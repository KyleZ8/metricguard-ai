"""Embedding-based complaint theme analysis for MetricGuard AI.

This module owns the fifth step of the MetricGuard workflow::

    metric movement -> data quality -> anomaly -> drivers -> TEXT THEMES -> explanation

``driver_analysis.py`` establishes *where* the corrected dispute-rate spike lives
(travel merchants on the mobile channel). This module asks whether customers
were saying the same thing: did complaint themes rise in that same business
area, in their own words?

Technique
---------
Sentence embeddings are the primary representation, following ``TECHNIQUE_LOG.md``.
Each narrative and each theme anchor is embedded into one vector space, and every
complaint is assigned to the theme whose anchor it is closest to by cosine
similarity. Embeddings are used rather than keyword counting because customers
describe the same problem in different language: "the same travel charge appears
twice" and "I was billed twice for my hotel" share almost no words but one
meaning.

The taxonomy is deliberately *controlled* rather than discovered. A fixed list of
business themes keeps the output in risk, product and operations language that a
manager can act on, and keeps the labels stable month over month so a trend means
something. Unsupervised clustering is offered separately, as a way to spot themes
the taxonomy is missing.

Embedding backends
------------------
Three, resolved in this order by :func:`resolve_embedder`:

1. an embedder the caller injects (a callable or a precomputed matrix) -- this is
   what the tests use, so they never touch the network;
2. :class:`SentenceTransformerEmbedder`, the primary path;
3. :class:`HashingEmbedder`, a dependency-free deterministic fallback so the
   module still runs where no model is cached.

Which backend produced a given report is always recorded on the report itself.
Silently swapping representations would make two months of theme trends
incomparable, which matters more here than in most places: the numbers are meant
to be put in front of a manager.

Everything is deterministic and no LLM is called. The optional clustering stretch
uses scikit-learn with a fixed ``random_state``; the taxonomy path never needs it.

Run directly to print the full report::

    python src/text_theme_analysis.py
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metric_engine import resolve_periods, resolve_tables
from quality_checks import DATA_DIR, TABLE_ACCOUNTS, TABLE_COMPLAINTS, _is_blank

try:  # Optional: only needed for the emerging-theme clustering stretch.
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where sklearn is absent
    KMeans = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False


try:  # Optional: the primary embedding backend.
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where the package is absent
    SentenceTransformer = None  # type: ignore[assignment]
    SENTENCE_TRANSFORMERS_AVAILABLE = False


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

UNCLEAR_THEME = "unclear_or_other"

# The controlled business taxonomy. Several anchor phrasings per theme, because a
# single sentence is a narrow target: a complaint is scored against its closest
# anchor, so more phrasings widen what the theme legitimately catches without
# loosening what it means.
THEME_TAXONOMY: Mapping[str, tuple[str, ...]] = {
    "unclear_merchant_descriptor": (
        "There is a purchase on my statement that I do not recognize and the merchant name is unclear.",
        "The merchant descriptor is confusing and I cannot tell which business charged me.",
        "I cannot identify the store behind this charge from the name on my statement.",
    ),
    "duplicate_looking_travel_charge": (
        "The same travel charge appears twice and I cannot tell whether it is pending or posted.",
        "I was billed twice for the same hotel or airline purchase.",
        "A duplicate-looking charge was posted for my trip and I want one removed.",
    ),
    "foreign_transaction_fee_confusion": (
        "I do not understand why a foreign transaction fee appeared after I used the card while traveling.",
        "An unexpected fee was added after an international purchase and I need a plain explanation.",
        "The fee label is confusing and I cannot tell what triggered it.",
    ),
    "mobile_dispute_submission_friction": (
        "I disputed a charge in the mobile app but the status in the app has not changed.",
        "The app will not let me file or track my dispute properly.",
        "The app and statement do not show the same status for my disputed activity.",
    ),
    "delayed_dispute_resolution": (
        "My dispute is taking too long to resolve and nobody has updated me.",
        "I need help understanding why my dispute was closed with explanation.",
        "The dispute has been open for weeks without a decision.",
    ),
    "failed_mobile_autopay": (
        "My autopay failed in the mobile app and now my account shows a late payment warning.",
        "The payment looked successful on my phone but later showed as returned.",
        "The app said payment scheduled, but the balance did not update on time.",
    ),
    "fraud_security_concern": (
        "I received a fraud alert and could not verify the transaction quickly.",
        "I worry my account may have been used without permission.",
        "A purchase was declined while I was traveling and then a similar charge posted later.",
    ),
    UNCLEAR_THEME: (
        "The statement is difficult to understand and I need help reading the activity.",
        "I contacted support but still do not understand the account activity.",
        "I have a general question about my account.",
    ),
}

THEME_NAMES: tuple[str, ...] = tuple(THEME_TAXONOMY)

TEXT_COLUMN = "complaint_narrative"

# Segment fields carried on the complaint record itself.
COMPLAINT_SEGMENT_FIELDS = ("merchant_category", "channel", "customer_segment", "fico_band")

# Segment fields worth joining in from the account dimension.
ACCOUNT_JOIN_FIELDS = ("product_type", "region")

DEFAULT_SEGMENT_FIELDS = ("merchant_category", "channel")

COMPLAINT_FACT_COLUMNS = (
    "complaint_id",
    "account_id",
    "complaint_date",
    "month",
    "narrative",
    "issue",
    "submitted_via",
)

THEME_SUMMARY_COLUMNS = (
    "theme_name",
    "previous_complaints",
    "current_complaints",
    "complaint_change",
    "previous_share",
    "current_share",
    "share_change",
    "avg_similarity",
)

SEGMENT_THEME_COLUMNS = (
    "segment_name",
    "segment_value",
    "theme_name",
    "previous_complaints",
    "current_complaints",
    "complaint_change",
    "previous_share",
    "current_share",
    "share_change",
    "avg_similarity",
)

REPRESENTATIVE_COLUMNS = (
    "theme_name",
    "rank",
    "complaint_id",
    "month",
    "similarity",
    "issue",
    "merchant_category",
    "channel",
    "narrative",
)

CLUSTER_COLUMNS = (
    "cluster_id",
    "complaints",
    "share_of_complaints",
    "dominant_theme",
    "dominant_theme_share",
    "avg_similarity_to_taxonomy",
    "representative_narrative",
)


@dataclass(frozen=True, eq=False)
class ThemeAssignment:
    """Complaints labelled with a theme, plus the vectors behind the labels.

    ``eq=False`` because a generated ``__eq__`` would compare DataFrames
    element-wise and raise on truth-value ambiguity.
    """

    facts: pd.DataFrame
    embeddings: np.ndarray
    backend_name: str


@dataclass(frozen=True, eq=False)
class TextThemeReport:
    """Dashboard-ready complaint-theme frames for one period comparison."""

    current_period: str
    previous_period: str
    backend_name: str
    labelled_complaints: pd.DataFrame
    theme_summary: pd.DataFrame
    segment_themes: pd.DataFrame
    representative_complaints: pd.DataFrame
    emerging_clusters: pd.DataFrame | None


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------


class HashingEmbedder:
    """Deterministic hashed bag-of-ngrams vectoriser. No downloads, no fitting.

    This is the fallback, not the technique: it matches on shared wording, so it
    misses the paraphrase cases sentence embeddings are chosen for. It exists so
    the module and its tests still run where no model is cached, and so a report
    can always be produced -- with ``backend_name`` saying plainly that this is
    what produced it.

    Determinism comes from BLAKE2b rather than Python's ``hash``, which is
    randomised per process.
    """

    name = "hashing"

    def __init__(self, dimensions: int = 512, ngram_range: tuple[int, int] = (1, 2)) -> None:
        if dimensions < 1:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self.dimensions = dimensions
        self.ngram_range = ngram_range

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimensions

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float64)
        low, high = self.ngram_range
        for row, text in enumerate(texts):
            tokens = self._tokenize(text)
            for size in range(low, high + 1):
                for start in range(len(tokens) - size + 1):
                    matrix[row, self._bucket(" ".join(tokens[start : start + size]))] += 1.0
        return matrix

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)


class SentenceTransformerEmbedder:
    """The primary backend: a sentence-transformers model.

    Loading is attempted from the local model cache first, so an already-cached
    model works with no network access.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence_transformers is not installed")
        self.model_name = model_name
        self.name = f"sentence_transformer:{model_name}"
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=256,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float64,
        )

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)


Embedder = Callable[[Sequence[str]], np.ndarray]


def resolve_embedder(embedder: Embedder | None = None) -> tuple[Embedder, str]:
    """Return an embedding callable and the name to record on the report.

    An injected embedder always wins, which is how tests avoid the network.
    Otherwise the sentence-transformer model is tried and the hashing fallback
    is used only if it cannot be loaded.
    """
    if embedder is not None:
        name = getattr(embedder, "name", None) or getattr(type(embedder), "__name__", "injected")
        return embedder, str(name)

    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            backend = SentenceTransformerEmbedder()
            return backend, backend.name
        except Exception:  # noqa: BLE001 - any load failure falls back deliberately
            pass

    fallback = HashingEmbedder()
    return fallback, fallback.name


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so a dot product is a cosine similarity.

    Zero rows (an empty narrative under a bag-of-words backend) stay zero rather
    than becoming NaN; they score 0 against every anchor.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of ``left`` and every row of ``right``."""
    return (
        _l2_normalize(np.asarray(left, dtype=np.float64))
        @ _l2_normalize(np.asarray(right, dtype=np.float64)).T
    )


# ---------------------------------------------------------------------------
# 1 and 2. Complaint fact table
# ---------------------------------------------------------------------------


def complaint_fact_table(
    tables: Mapping[str, pd.DataFrame] | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Complaints with a month key, clean narrative text, and segment columns."""
    resolved = resolve_tables(tables, data_dir)
    complaints = resolved[TABLE_COMPLAINTS].copy()

    complaints["complaint_date"] = pd.to_datetime(complaints["date_received"])
    complaints["month"] = complaints["complaint_date"].dt.to_period("M").astype(str)
    complaints["narrative"] = (
        complaints[TEXT_COLUMN]
        .astype("object")
        .where(complaints[TEXT_COLUMN].notna(), "")
        .astype(str)
        .str.strip()
    )

    accounts = resolved.get(TABLE_ACCOUNTS)
    if accounts is not None:
        join_fields = [
            field
            for field in ACCOUNT_JOIN_FIELDS
            if field in accounts.columns and field not in complaints.columns
        ]
        if join_fields:
            complaints = complaints.merge(
                accounts[["account_id"] + join_fields], on="account_id", how="left"
            )

    for field in COMPLAINT_SEGMENT_FIELDS + ACCOUNT_JOIN_FIELDS:
        if field in complaints.columns:
            # ~_is_blank(...), not ~complaints[field].notna(): merchant_category can
            # carry the planted missing-value defect through from its related
            # transaction, which is NaN after a CSV round-trip but a literal "" after
            # a Parquet round-trip (data/sample/). notna() alone would only label the
            # CSV case __missing__ and let "" through unlabeled on Parquet-loaded data.
            complaints[field] = (
                complaints[field]
                .astype("object")
                .where(~_is_blank(complaints[field]), "__missing__")
                .astype(str)
            )

    leading = [column for column in COMPLAINT_FACT_COLUMNS if column in complaints.columns]
    remaining = [column for column in complaints.columns if column not in leading]
    return complaints[leading + remaining].reset_index(drop=True)


def available_segment_fields(facts: pd.DataFrame) -> tuple[str, ...]:
    """Segment fields actually present on the complaint fact table."""
    return tuple(
        field for field in COMPLAINT_SEGMENT_FIELDS + ACCOUNT_JOIN_FIELDS if field in facts.columns
    )


# ---------------------------------------------------------------------------
# 4, 5 and 6. Theme assignment
# ---------------------------------------------------------------------------


def assign_complaint_themes(
    facts: pd.DataFrame,
    embedder: Embedder | None = None,
    taxonomy: Mapping[str, Sequence[str]] = THEME_TAXONOMY,
    min_similarity: float | None = None,
    text_column: str = "narrative",
) -> ThemeAssignment:
    """Label each complaint with its closest business theme.

    Anchors and narratives are embedded in a single call so both live in the
    same vector space. A complaint scores against its *closest anchor* per theme,
    then takes the best-scoring theme.

    Blank narratives, and any narrative the backend maps to a zero vector, are
    assigned ``unclear_or_other`` at similarity 0.0: with no signal to compare,
    an argmax over an all-zero row would otherwise return whichever theme
    happens to be listed first. ``min_similarity``, when
    given, routes weak matches to the same place -- useful, but calibrate it per
    backend, since raw cosine ranges differ sharply between a neural model and
    the hashing fallback.
    """
    if not taxonomy:
        raise ValueError("taxonomy must define at least one theme")

    resolved_embedder, backend_name = resolve_embedder(embedder)

    anchor_texts: list[str] = []
    anchor_theme: list[str] = []
    for theme_name, anchors in taxonomy.items():
        for anchor in anchors:
            anchor_texts.append(anchor)
            anchor_theme.append(theme_name)
    anchor_theme_array = np.array(anchor_theme)

    narratives = facts[text_column].fillna("").astype(str).tolist()
    has_text = np.array([bool(text.strip()) for text in narratives])

    vectors = np.asarray(resolved_embedder(anchor_texts + narratives), dtype=np.float64)
    if vectors.shape[0] != len(anchor_texts) + len(narratives):
        raise ValueError(
            f"embedder returned {vectors.shape[0]} vectors for "
            f"{len(anchor_texts) + len(narratives)} texts"
        )
    anchor_vectors = vectors[: len(anchor_texts)]
    narrative_vectors = vectors[len(anchor_texts) :]

    similarities = cosine_similarity_matrix(narrative_vectors, anchor_vectors)

    theme_names = list(taxonomy)
    per_theme = np.column_stack(
        [similarities[:, anchor_theme_array == theme].max(axis=1) for theme in theme_names]
    )
    best_index = per_theme.argmax(axis=1)
    best_similarity = per_theme[np.arange(len(per_theme)), best_index]
    assigned = np.array(theme_names, dtype=object)[best_index]

    # A blank narrative, or one the backend maps to a zero vector, carries no
    # signal to compare: label it unclear rather than letting argmax pick the
    # first theme off an all-zero similarity row.
    has_signal = has_text & (np.linalg.norm(narrative_vectors, axis=1) > 0)

    fallback_theme = UNCLEAR_THEME if UNCLEAR_THEME in taxonomy else theme_names[-1]
    assigned = np.where(has_signal, assigned, fallback_theme)
    best_similarity = np.where(has_signal, best_similarity, 0.0)
    if min_similarity is not None:
        assigned = np.where(best_similarity >= min_similarity, assigned, fallback_theme)

    labelled = facts.copy()
    labelled["theme_name"] = pd.Categorical(assigned, categories=theme_names)
    labelled["theme_similarity"] = best_similarity.astype(float)
    labelled["embedding_backend"] = backend_name

    return ThemeAssignment(facts=labelled, embeddings=narrative_vectors, backend_name=backend_name)


def _resolve_assignment(
    assignment: ThemeAssignment | pd.DataFrame | None,
    tables: Mapping[str, pd.DataFrame] | None,
    embedder: Embedder | None,
    min_similarity: float | None,
    data_dir: Path,
) -> pd.DataFrame:
    """Accept a prepared assignment, or build one from the tables."""
    if isinstance(assignment, ThemeAssignment):
        return assignment.facts
    if isinstance(assignment, pd.DataFrame):
        return assignment
    facts = complaint_fact_table(tables, data_dir)
    return assign_complaint_themes(facts, embedder, min_similarity=min_similarity).facts


def _resolve_periods_from(
    labelled: pd.DataFrame, current_period: str | None, previous_period: str | None
) -> tuple[str, str]:
    months = pd.DataFrame({"month": sorted(labelled["month"].unique())})
    # Reusing metric_engine's resolver keeps period defaulting and error
    # messages identical across every analysis module.
    return resolve_periods(months, current_period, previous_period)


def _movement(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    current_period: str,
    previous_period: str,
    share_within: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Period-over-period complaint counts, shares and mean similarity."""
    window = frame[frame["month"].isin([current_period, previous_period])]

    counts = window.groupby(list(group_columns) + ["month"], observed=False).size().unstack("month")
    for period in (previous_period, current_period):
        if period not in counts.columns:
            counts[period] = 0
    counts = counts.fillna(0)

    out = pd.DataFrame(
        {
            "previous_complaints": counts[previous_period].astype(int),
            "current_complaints": counts[current_period].astype(int),
        }
    ).reset_index()
    out["complaint_change"] = out["current_complaints"] - out["previous_complaints"]

    # Share is computed inside the population the reader is looking at: the whole
    # book for the summary table, one segment value for the segment table.
    if share_within:
        previous_total = out.groupby(list(share_within))["previous_complaints"].transform("sum")
        current_total = out.groupby(list(share_within))["current_complaints"].transform("sum")
    else:
        previous_total = pd.Series(out["previous_complaints"].sum(), index=out.index)
        current_total = pd.Series(out["current_complaints"].sum(), index=out.index)

    out["previous_share"] = (
        out["previous_complaints"].divide(previous_total).where(previous_total > 0)
    )
    out["current_share"] = out["current_complaints"].divide(current_total).where(current_total > 0)
    out["share_change"] = out["current_share"] - out["previous_share"]

    similarity = (
        window.groupby(list(group_columns), observed=False)["theme_similarity"]
        .mean()
        .reset_index()
        .rename(columns={"theme_similarity": "avg_similarity"})
    )
    return out.merge(similarity, on=list(group_columns), how="left")


# ---------------------------------------------------------------------------
# 7. Theme summary table
# ---------------------------------------------------------------------------


def theme_summary_table(
    current_period: str | None = None,
    previous_period: str | None = None,
    assignment: ThemeAssignment | pd.DataFrame | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    embedder: Embedder | None = None,
    min_similarity: float | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Complaint theme mix and movement between two periods.

    Both a count change and a share change are reported: total complaint volume
    rose in the demo month, so a theme can grow in count while losing share. The
    share column is what says a theme grew *faster than the book*.
    """
    labelled = _resolve_assignment(assignment, tables, embedder, min_similarity, data_dir)
    resolved_current, resolved_previous = _resolve_periods_from(
        labelled, current_period, previous_period
    )

    summary = _movement(labelled, ["theme_name"], resolved_current, resolved_previous)
    summary["theme_name"] = summary["theme_name"].astype(str)

    return (
        summary[list(THEME_SUMMARY_COLUMNS)]
        .sort_values(["complaint_change", "theme_name"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 8. Segment theme table
# ---------------------------------------------------------------------------


def segment_theme_table(
    current_period: str | None = None,
    previous_period: str | None = None,
    segment_fields: Sequence[str] = DEFAULT_SEGMENT_FIELDS,
    assignment: ThemeAssignment | pd.DataFrame | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    embedder: Embedder | None = None,
    min_similarity: float | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Theme movement inside each segment value.

    This is the table that connects text back to ``driver_analysis.py``: it says
    whether the themes that grew did so in the same merchant category and channel
    the corrected KPI movement came from. Shares are computed within a segment
    value, so a row reads "of travel complaints this month, this share were
    duplicate-looking charges".
    """
    labelled = _resolve_assignment(assignment, tables, embedder, min_similarity, data_dir)
    resolved_current, resolved_previous = _resolve_periods_from(
        labelled, current_period, previous_period
    )

    frames: list[pd.DataFrame] = []
    for field in segment_fields:
        if field not in labelled.columns:
            continue
        movement = _movement(
            labelled,
            [field, "theme_name"],
            resolved_current,
            resolved_previous,
            share_within=[field],
        )
        movement = movement.rename(columns={field: "segment_value"})
        movement.insert(0, "segment_name", field)
        frames.append(movement)

    if not frames:
        return pd.DataFrame(columns=list(SEGMENT_THEME_COLUMNS))

    segments = pd.concat(frames, ignore_index=True)
    segments["segment_value"] = segments["segment_value"].astype(str)
    segments["theme_name"] = segments["theme_name"].astype(str)

    return (
        segments[list(SEGMENT_THEME_COLUMNS)]
        .sort_values(
            ["segment_name", "complaint_change", "segment_value", "theme_name"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 9. Representative complaints
# ---------------------------------------------------------------------------


def representative_complaints(
    assignment: ThemeAssignment | pd.DataFrame | None = None,
    top_n: int = 3,
    period: str | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    embedder: Embedder | None = None,
    min_similarity: float | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """The clearest examples of each theme, ranked by similarity.

    These are the evidence rows. A theme count is only trustworthy if an analyst
    can read the complaints behind it and agree with the label, so every theme in
    the summary should be traceable to real narratives here.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}")

    labelled = _resolve_assignment(assignment, tables, embedder, min_similarity, data_dir)
    if period is not None:
        labelled = labelled[labelled["month"].eq(period)]

    columns = [column for column in REPRESENTATIVE_COLUMNS if column not in ("rank",)]
    available = [column for column in columns if column in labelled.columns]

    ranked = labelled.sort_values(
        ["theme_name", "theme_similarity", "complaint_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    top = ranked.groupby("theme_name", observed=True, group_keys=False).head(top_n).copy()
    top["rank"] = top.groupby("theme_name", observed=True).cumcount() + 1
    top = top.rename(columns={"theme_similarity": "similarity"})
    top["theme_name"] = top["theme_name"].astype(str)

    ordered = [
        column for column in REPRESENTATIVE_COLUMNS if column in top.columns or column in available
    ]
    return top[[column for column in ordered if column in top.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stretch: unsupervised emerging themes
# ---------------------------------------------------------------------------


def cluster_emerging_themes(
    assignment: ThemeAssignment,
    n_clusters: int = 8,
    period: str | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Cluster complaint embeddings to surface themes the taxonomy may be missing.

    The taxonomy path stays primary; this is a discovery aid. A cluster whose
    complaints spread thinly across taxonomy labels, or whose mean similarity to
    every anchor is low, is a candidate for a theme the business has not named
    yet.

    Deterministic: fixed ``random_state`` and an explicit ``n_init``.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is not installed, so emerging-theme clustering is unavailable. "
            "The taxonomy path does not require it."
        )

    facts = assignment.facts
    embeddings = assignment.embeddings
    mask = np.ones(len(facts), dtype=bool)
    if period is not None:
        mask = facts["month"].eq(period).to_numpy()

    selected_facts = facts[mask]
    selected_vectors = _l2_normalize(embeddings[mask])
    if len(selected_facts) < n_clusters:
        raise ValueError(
            f"need at least {n_clusters} complaints to form {n_clusters} clusters, "
            f"got {len(selected_facts)}"
        )

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(selected_vectors)

    rows: list[dict[str, object]] = []
    for cluster_id in range(n_clusters):
        members = selected_facts[labels == cluster_id]
        if members.empty:
            continue
        theme_counts = members["theme_name"].astype(str).value_counts()
        centroid = model.cluster_centers_[cluster_id]
        distances = selected_vectors[labels == cluster_id] @ centroid
        representative = members.iloc[int(np.argmax(distances))]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "complaints": len(members),
                "share_of_complaints": len(members) / len(selected_facts),
                "dominant_theme": str(theme_counts.index[0]),
                "dominant_theme_share": float(theme_counts.iloc[0] / len(members)),
                "avg_similarity_to_taxonomy": float(members["theme_similarity"].mean()),
                "representative_narrative": str(representative["narrative"]),
            }
        )

    return (
        pd.DataFrame(rows, columns=list(CLUSTER_COLUMNS))
        .sort_values(["complaints", "cluster_id"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 10. Report assembly
# ---------------------------------------------------------------------------


def build_text_theme_report(
    current_period: str | None = None,
    previous_period: str | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    embedder: Embedder | None = None,
    min_similarity: float | None = None,
    segment_fields: Sequence[str] = DEFAULT_SEGMENT_FIELDS,
    top_n: int = 3,
    include_clusters: bool = True,
    n_clusters: int = 8,
    data_dir: Path = DATA_DIR,
) -> TextThemeReport:
    """Assemble every dashboard-ready complaint-theme frame in one pass.

    Complaints are embedded once here and the vectors are reused by every table,
    so a full report costs a single encode pass.
    """
    facts = complaint_fact_table(tables, data_dir)
    assignment = assign_complaint_themes(facts, embedder, min_similarity=min_similarity)
    resolved_current, resolved_previous = _resolve_periods_from(
        assignment.facts, current_period, previous_period
    )

    clusters: pd.DataFrame | None = None
    if include_clusters and SKLEARN_AVAILABLE:
        clusters = cluster_emerging_themes(
            assignment, n_clusters=n_clusters, period=resolved_current
        )

    return TextThemeReport(
        current_period=resolved_current,
        previous_period=resolved_previous,
        backend_name=assignment.backend_name,
        labelled_complaints=assignment.facts,
        theme_summary=theme_summary_table(
            resolved_current, resolved_previous, assignment=assignment
        ),
        segment_themes=segment_theme_table(
            resolved_current, resolved_previous, segment_fields, assignment=assignment
        ),
        representative_complaints=representative_complaints(
            assignment, top_n=top_n, period=resolved_current
        ),
        emerging_clusters=clusters,
    )


if __name__ == "__main__":
    report = build_text_theme_report()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 70)

    print("MetricGuard AI - complaint theme analysis")
    print(f"{report.current_period} vs {report.previous_period}   backend: {report.backend_name}")
    print("=" * 118)

    print()
    print("Theme summary")
    print("-" * 118)
    print(report.theme_summary.round(4).to_string(index=False))

    print()
    print("Theme movement inside travel and mobile (the corrected KPI driver segments)")
    print("-" * 118)
    focus = report.segment_themes[report.segment_themes["segment_value"].isin(["travel", "mobile"])]
    print(focus.head(12).round(4).to_string(index=False))

    print()
    print("Representative complaints")
    print("-" * 118)
    print(
        report.representative_complaints[
            ["theme_name", "rank", "similarity", "merchant_category", "channel", "narrative"]
        ]
        .head(16)
        .round(3)
        .to_string(index=False)
    )

    print()
    if report.emerging_clusters is None:
        print("Emerging-theme clustering: skipped (scikit-learn not installed)")
    else:
        print("Emerging-theme clusters")
        print("-" * 118)
        print(report.emerging_clusters.round(3).to_string(index=False))
