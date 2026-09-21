"""Tests for the MetricGuard AI complaint theme analysis module.

These run under pytest. Because pytest is not installed in every environment
this project is developed in, the file is also directly runnable, matching the
other test modules in this project::

    python tests/test_text_theme_analysis.py

No test downloads a model. Logic tests inject a tiny deterministic embedder
whose vectors can be worked out by hand; the real-data tests use the offline
:class:`HashingEmbedder`. One optional test exercises the sentence-transformer
backend and skips itself cleanly when no model is cached locally.

Two things these tests are careful about.

*Counts, not shares.* Complaint volume and the travel segment both grow sharply
in August, so a theme can grow strongly in count while its share of a fast-growing
segment falls. Share-change assertions on a single segment/theme cell flip on
small data movements; count growth plus concentration-over-base-rate does not.

*Backend-dependent separability.* The offline ``HashingEmbedder`` is a bag of
words, so it cannot separate "the same travel charge appears twice" from "the same
charge appears twice" -- they differ by one token. The sentence-transformer model
separates them cleanly. Tests that depend on that separation therefore assert the
underlying *data* property (travel wording only appears on travel rows), which
holds regardless of backend, and check the classifier's ranking only where the
real model is available.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quality_checks import TABLE_ACCOUNTS, TABLE_COMPLAINTS, load_tables  # noqa: E402
from text_theme_analysis import (  # noqa: E402
    CLUSTER_COLUMNS,
    REPRESENTATIVE_COLUMNS,
    SEGMENT_THEME_COLUMNS,
    SKLEARN_AVAILABLE,
    THEME_NAMES,
    THEME_SUMMARY_COLUMNS,
    THEME_TAXONOMY,
    UNCLEAR_THEME,
    HashingEmbedder,
    TextThemeReport,
    ThemeAssignment,
    assign_complaint_themes,
    build_text_theme_report,
    cluster_emerging_themes,
    complaint_fact_table,
    cosine_similarity_matrix,
    representative_complaints,
    resolve_embedder,
    segment_theme_table,
    theme_summary_table,
)

SPIKE_MONTH = "2026-08"
PRIOR_MONTH = "2026-07"

DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
DOCS_DIR = PROJECT_ROOT / "docs"
CSV_FILES = (
    "accounts.csv",
    "account_monthly_snapshot.csv",
    "transactions.csv",
    "complaints.csv",
    "metric_definitions.csv",
)

# Themes that genuinely rise in the generated August data.
RISING_THEMES = (
    "duplicate_looking_travel_charge",
    "unclear_merchant_descriptor",
    "mobile_dispute_submission_friction",
    "delayed_dispute_resolution",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TOY_TAXONOMY = {
    "duplicate_looking_travel_charge": ("the same travel charge appears twice",),
    "foreign_transaction_fee_confusion": ("an unexpected fee appeared",),
    "failed_mobile_autopay": ("my autopay failed",),
    UNCLEAR_THEME: ("general question about the account",),
}


def toy_embedder(texts):
    """A hand-checkable 4-dimensional one-hot embedder.

    Each dimension fires on one keyword, so every similarity in the toy tests
    can be computed on paper. Text matching no keyword becomes a zero vector,
    which is how the "no signal" path gets exercised.
    """
    rows = []
    for text in texts:
        low = str(text).lower()
        rows.append(
            [
                1.0 if ("twice" in low or "duplicate" in low) else 0.0,
                1.0 if "fee" in low else 0.0,
                1.0 if "autopay" in low else 0.0,
                1.0 if "general" in low else 0.0,
            ]
        )
    return np.array(rows, dtype=float)


toy_embedder.name = "toy"  # type: ignore[attr-defined]


def _toy_facts() -> pd.DataFrame:
    rows = [
        ("C1", "2026-01", "I was billed twice for my hotel", "duplicate_looking_travel_charge"),
        ("C2", "2026-01", "there is a strange fee on my card", "foreign_transaction_fee_confusion"),
        ("C3", "2026-01", "my autopay failed again", "failed_mobile_autopay"),
        ("C4", "2026-01", "general question about billing", UNCLEAR_THEME),
        ("C5", "2026-02", "I was billed twice for my flight", "duplicate_looking_travel_charge"),
        ("C6", "2026-02", "the same charge appears twice", "duplicate_looking_travel_charge"),
        ("C7", "2026-02", "an unexpected fee showed up", "foreign_transaction_fee_confusion"),
        ("C8", "2026-02", "", UNCLEAR_THEME),
        ("C9", "2026-02", "   ", UNCLEAR_THEME),
        ("C10", "2026-02", "zzz nothing recognisable here", UNCLEAR_THEME),
    ]
    facts = pd.DataFrame(rows, columns=["complaint_id", "month", "narrative", "expected_theme"])
    facts["account_id"] = "A1"
    facts["complaint_date"] = pd.to_datetime(facts["month"] + "-15")
    facts["issue"] = "Problem with a purchase shown on your statement"
    facts["submitted_via"] = "Web"
    facts["merchant_category"] = ["travel", "fee", "payment"] * 3 + ["travel"]
    facts["channel"] = "mobile"
    return facts


@lru_cache(maxsize=1)
def _tables() -> dict[str, pd.DataFrame]:
    return load_tables()


@lru_cache(maxsize=1)
def _facts() -> pd.DataFrame:
    return complaint_fact_table(_tables())


@lru_cache(maxsize=1)
def _assignment() -> ThemeAssignment:
    return assign_complaint_themes(_facts(), HashingEmbedder())


@lru_cache(maxsize=1)
def _summary() -> pd.DataFrame:
    return theme_summary_table(assignment=_assignment())


@lru_cache(maxsize=1)
def _segments() -> pd.DataFrame:
    return segment_theme_table(assignment=_assignment())


def _theme_row(summary: pd.DataFrame, theme_name: str) -> pd.Series:
    matches = summary[summary["theme_name"].eq(theme_name)]
    assert len(matches) == 1, f"expected one row for {theme_name}, got {len(matches)}"
    return matches.iloc[0]


def _segment_theme_row(segments: pd.DataFrame, value: str, theme_name: str) -> pd.Series:
    matches = segments[segments["segment_value"].eq(value) & segments["theme_name"].eq(theme_name)]
    assert len(matches) == 1, f"expected one row for {value}/{theme_name}, got {len(matches)}"
    return matches.iloc[0]


def _raises(exception_type, callable_, *args, **kwargs):
    try:
        callable_(*args, **kwargs)
    except exception_type as error:
        return error
    raise AssertionError(f"expected {exception_type.__name__} but nothing was raised")


# ---------------------------------------------------------------------------
# Cosine similarity and embedding backends
# ---------------------------------------------------------------------------


def test_cosine_similarity_matches_a_hand_computation():
    left = np.array([[1.0, 0.0], [1.0, 1.0]])
    right = np.array([[1.0, 0.0], [0.0, 1.0]])

    similarity = cosine_similarity_matrix(left, right)

    np.testing.assert_allclose(similarity[0], [1.0, 0.0])
    np.testing.assert_allclose(similarity[1], [np.sqrt(0.5), np.sqrt(0.5)])


def test_cosine_similarity_is_scale_invariant():
    single = np.array([[3.0, 4.0]])
    scaled = np.array([[30.0, 40.0]])

    np.testing.assert_allclose(cosine_similarity_matrix(single, scaled), np.array([[1.0]]))


def test_cosine_similarity_treats_a_zero_vector_as_zero_not_nan():
    similarity = cosine_similarity_matrix(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))

    assert np.isfinite(similarity).all()
    assert similarity[0, 0] == 0.0


def test_injected_embedder_is_preferred_and_named():
    embedder, name = resolve_embedder(toy_embedder)

    assert embedder is toy_embedder
    assert name == "toy"


def test_hashing_embedder_is_deterministic_within_a_process():
    first = HashingEmbedder().encode(["a duplicate travel charge", "autopay failed"])
    second = HashingEmbedder().encode(["a duplicate travel charge", "autopay failed"])

    np.testing.assert_array_equal(first, second)


def test_hashing_embedder_is_stable_across_processes():
    # Python's built-in hash() is randomised per process, so a hashed vectoriser
    # built on it would silently produce different vectors on every run.
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from text_theme_analysis import HashingEmbedder\n"
        "v = HashingEmbedder(dimensions=64).encode(['duplicate travel charge'])[0]\n"
        "print(','.join(str(int(x)) for x in v))\n" % str(PROJECT_ROOT / "src")
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "12345")
    ]

    assert runs[0] == runs[1] != ""


def test_hashing_embedder_rejects_a_non_positive_dimension():
    error = _raises(ValueError, HashingEmbedder, 0)

    assert "dimensions" in str(error)


# ---------------------------------------------------------------------------
# Theme assignment with injected deterministic embeddings
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_toy_assignment_matches_the_expected_theme_for_every_complaint():
    facts = _toy_facts()

    labelled = assign_complaint_themes(facts, toy_embedder, taxonomy=TOY_TAXONOMY).facts

    assert labelled["theme_name"].astype(str).tolist() == facts["expected_theme"].tolist()


@pytest.mark.slow
def test_toy_assignment_reports_exact_similarity_scores():
    labelled = assign_complaint_themes(_toy_facts(), toy_embedder, taxonomy=TOY_TAXONOMY).facts
    scores = labelled.set_index("complaint_id")["theme_similarity"]

    # One-hot narrative against a one-hot anchor is a cosine of exactly 1.
    np.testing.assert_allclose(scores["C1"], 1.0)
    np.testing.assert_allclose(scores["C2"], 1.0)
    np.testing.assert_allclose(scores["C3"], 1.0)


@pytest.mark.slow
def test_blank_narratives_are_labelled_unclear_at_zero_similarity():
    labelled = assign_complaint_themes(_toy_facts(), toy_embedder, taxonomy=TOY_TAXONOMY).facts
    blanks = labelled[labelled["complaint_id"].isin(["C8", "C9"])]

    assert set(blanks["theme_name"].astype(str)) == {UNCLEAR_THEME}
    assert (blanks["theme_similarity"] == 0.0).all()


@pytest.mark.slow
def test_a_narrative_with_no_signal_is_labelled_unclear():
    # "zzz nothing recognisable here" embeds to a zero vector under the toy
    # backend, so there is nothing to compare against any anchor.
    labelled = assign_complaint_themes(_toy_facts(), toy_embedder, taxonomy=TOY_TAXONOMY).facts
    row = labelled[labelled["complaint_id"].eq("C10")].iloc[0]

    assert str(row["theme_name"]) == UNCLEAR_THEME
    assert row["theme_similarity"] == 0.0


def test_min_similarity_routes_weak_matches_to_unclear():
    facts = pd.DataFrame(
        {
            "complaint_id": ["C1"],
            "month": ["2026-01"],
            # Fires two dimensions, so its best cosine is sqrt(0.5) ~ 0.707.
            "narrative": ["a duplicate charge and an unexpected fee"],
        }
    )

    lenient = assign_complaint_themes(facts, toy_embedder, taxonomy=TOY_TAXONOMY).facts
    strict = assign_complaint_themes(
        facts, toy_embedder, taxonomy=TOY_TAXONOMY, min_similarity=0.9
    ).facts

    np.testing.assert_allclose(lenient.iloc[0]["theme_similarity"], np.sqrt(0.5))
    assert str(lenient.iloc[0]["theme_name"]) != UNCLEAR_THEME
    assert str(strict.iloc[0]["theme_name"]) == UNCLEAR_THEME


@pytest.mark.slow
def test_assignment_records_the_backend_that_produced_it():
    assignment = assign_complaint_themes(_toy_facts(), toy_embedder, taxonomy=TOY_TAXONOMY)

    assert assignment.backend_name == "toy"
    assert set(assignment.facts["embedding_backend"]) == {"toy"}


@pytest.mark.slow
def test_assignment_returns_one_embedding_row_per_complaint():
    facts = _toy_facts()

    assignment = assign_complaint_themes(facts, toy_embedder, taxonomy=TOY_TAXONOMY)

    assert assignment.embeddings.shape[0] == len(facts)


@pytest.mark.slow
def test_assignment_rejects_an_embedder_returning_the_wrong_row_count():
    def broken(texts):
        return np.ones((len(texts) - 1, 3))

    error = _raises(ValueError, assign_complaint_themes, _toy_facts(), broken, TOY_TAXONOMY)

    assert "vectors" in str(error)


@pytest.mark.slow
def test_assignment_rejects_an_empty_taxonomy():
    error = _raises(ValueError, assign_complaint_themes, _toy_facts(), toy_embedder, {})

    assert "at least one theme" in str(error)


def test_every_taxonomy_theme_is_reachable_from_its_own_anchor():
    # Each anchor should classify as its own theme; otherwise the taxonomy has
    # two themes that are not actually distinguishable.
    anchors = [(theme, text) for theme, texts in THEME_TAXONOMY.items() for text in texts]
    facts = pd.DataFrame(
        {
            "complaint_id": [f"A{index}" for index in range(len(anchors))],
            "month": "2026-01",
            "narrative": [text for _, text in anchors],
        }
    )

    labelled = assign_complaint_themes(facts, HashingEmbedder()).facts

    assert labelled["theme_name"].astype(str).tolist() == [theme for theme, _ in anchors]


# ---------------------------------------------------------------------------
# Complaint fact table
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_complaint_fact_table_has_the_required_fields():
    facts = _facts()

    for column in (
        "complaint_id",
        "account_id",
        "complaint_date",
        "month",
        "narrative",
        "issue",
        "submitted_via",
        "merchant_category",
    ):
        assert column in facts.columns, column
    assert len(facts) == 5_300


@pytest.mark.slow
def test_complaint_fact_table_joins_account_segments():
    facts = _facts()

    for column in ("product_type", "region"):
        assert column in facts.columns
        assert facts[column].notna().all()


def test_complaint_fact_table_labels_blank_segment_values_as_missing():
    tables = {
        TABLE_ACCOUNTS: pd.DataFrame(
            {
                "account_id": ["A1"],
                "fico_band": [">660"],
                "customer_segment": ["affluent"],
                "product_type": ["travel_rewards"],
                "region": ["West"],
            }
        ),
        TABLE_COMPLAINTS: pd.DataFrame(
            {
                "complaint_id": ["C1"],
                "account_id": ["A1"],
                "date_received": ["2026-08-15"],
                "product": ["Credit card"],
                "issue": ["Problem with a purchase shown on your statement"],
                "complaint_narrative": ["I was charged twice."],
                "submitted_via": ["Web"],
                "merchant_category": ["   "],
                "channel": [""],
            }
        ),
    }

    facts = complaint_fact_table(tables)

    assert facts.loc[0, "merchant_category"] == "__missing__"
    assert facts.loc[0, "channel"] == "__missing__"


@pytest.mark.slow
def test_complaint_months_are_well_formed():
    months = _facts()["month"]

    assert months.str.fullmatch(r"\d{4}-\d{2}").all()
    assert months.nunique() == 8


@pytest.mark.slow
def test_narratives_are_stripped_and_never_null():
    narratives = _facts()["narrative"]

    assert narratives.notna().all()
    assert (narratives == narratives.str.strip()).all()


# ---------------------------------------------------------------------------
# 7. Theme summary table
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_theme_summary_has_the_documented_schema():
    summary = _summary()

    assert list(summary.columns) == list(THEME_SUMMARY_COLUMNS)
    assert set(summary["theme_name"]) == set(THEME_NAMES)


@pytest.mark.slow
def test_theme_summary_counts_reconcile_with_the_labelled_complaints():
    summary = _summary()
    labelled = _assignment().facts

    for period, column in (
        (PRIOR_MONTH, "previous_complaints"),
        (SPIKE_MONTH, "current_complaints"),
    ):
        assert summary[column].sum() == int(labelled["month"].eq(period).sum())


@pytest.mark.slow
def test_theme_summary_changes_are_consistent_with_their_levels():
    summary = _summary()

    assert (
        summary["complaint_change"]
        == summary["current_complaints"] - summary["previous_complaints"]
    ).all()
    np.testing.assert_allclose(
        summary["share_change"].to_numpy(dtype=float),
        (summary["current_share"] - summary["previous_share"]).to_numpy(dtype=float),
    )


@pytest.mark.slow
def test_theme_shares_sum_to_one_in_each_period():
    summary = _summary()

    np.testing.assert_allclose(summary["previous_share"].sum(), 1.0)
    np.testing.assert_allclose(summary["current_share"].sum(), 1.0)


@pytest.mark.slow
def test_theme_summary_periods_default_to_the_two_most_recent_months():
    report = build_text_theme_report(
        tables=_tables(), embedder=HashingEmbedder(), include_clusters=False
    )

    assert report.current_period == SPIKE_MONTH
    assert report.previous_period == PRIOR_MONTH


@pytest.mark.slow
def test_theme_summary_accepts_explicit_periods():
    summary = theme_summary_table("2026-03", "2026-01", assignment=_assignment())

    assert summary["current_complaints"].sum() == 661
    assert summary["previous_complaints"].sum() == 601


@pytest.mark.slow
def test_theme_summary_rejects_an_unknown_period():
    error = _raises(KeyError, theme_summary_table, "2031-01", PRIOR_MONTH, _assignment())

    assert "2031-01" in str(error)


@pytest.mark.slow
def test_average_similarity_is_within_the_cosine_range():
    similarities = _summary()["avg_similarity"].dropna()

    assert not similarities.empty
    assert (similarities >= -1.0).all() and (similarities <= 1.0).all()


# ---------------------------------------------------------------------------
# 8. Segment theme table
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_segment_theme_table_has_the_documented_schema():
    segments = _segments()

    assert list(segments.columns) == list(SEGMENT_THEME_COLUMNS)
    assert set(segments["segment_name"]) == {"merchant_category", "channel"}


@pytest.mark.slow
def test_segment_theme_shares_sum_to_one_within_each_segment_value():
    segments = _segments()
    totals = segments.groupby(["segment_name", "segment_value"])["current_share"].sum()
    populated = totals[totals > 0]

    np.testing.assert_allclose(populated.to_numpy(dtype=float), np.ones(len(populated)))


@pytest.mark.slow
def test_segment_theme_counts_reconcile_with_the_summary_totals():
    segments = _segments()
    channel_rows = segments[segments["segment_name"].eq("channel")]

    assert channel_rows["current_complaints"].sum() == _summary()["current_complaints"].sum()


@pytest.mark.slow
def test_segment_theme_table_can_use_other_segment_fields():
    segments = segment_theme_table(segment_fields=("fico_band",), assignment=_assignment())

    assert set(segments["segment_name"]) == {"fico_band"}
    assert set(segments["segment_value"]) == {"<=660", ">660"}


# ---------------------------------------------------------------------------
# 9. Representative complaints
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_representative_complaints_have_the_documented_schema():
    examples = representative_complaints(_assignment(), top_n=3)

    for column in REPRESENTATIVE_COLUMNS:
        assert column in examples.columns, column


@pytest.mark.slow
def test_representative_complaints_belong_to_the_theme_they_illustrate():
    examples = representative_complaints(_assignment(), top_n=3)
    labelled = _assignment().facts.set_index("complaint_id")

    for _, row in examples.iterrows():
        actual = str(labelled.loc[row["complaint_id"], "theme_name"])
        assert actual == row["theme_name"], (
            f"{row['complaint_id']} is {actual}, not {row['theme_name']}"
        )


@pytest.mark.slow
def test_representative_complaints_are_ranked_by_similarity():
    examples = representative_complaints(_assignment(), top_n=3)

    for theme, group in examples.groupby("theme_name"):
        similarities = group.sort_values("rank")["similarity"].tolist()
        assert similarities == sorted(similarities, reverse=True), theme
        assert group["rank"].tolist() == list(range(1, len(group) + 1))


@pytest.mark.slow
def test_representative_complaints_respect_top_n():
    examples = representative_complaints(_assignment(), top_n=2)

    assert (examples.groupby("theme_name").size() <= 2).all()


@pytest.mark.slow
def test_representative_complaints_can_focus_on_one_period():
    examples = representative_complaints(_assignment(), top_n=3, period=SPIKE_MONTH)

    assert set(examples["month"]) == {SPIKE_MONTH}


@pytest.mark.slow
def test_representative_complaints_rejects_a_non_positive_top_n():
    error = _raises(ValueError, representative_complaints, _assignment(), 0)

    assert "top_n" in str(error)


# ---------------------------------------------------------------------------
# 10. Report assembly
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_report_exposes_every_dashboard_frame():
    report = build_text_theme_report(
        tables=_tables(), embedder=HashingEmbedder(), include_clusters=False
    )

    assert isinstance(report, TextThemeReport)
    for frame in (
        report.labelled_complaints,
        report.theme_summary,
        report.segment_themes,
        report.representative_complaints,
    ):
        assert isinstance(frame, pd.DataFrame) and not frame.empty


@pytest.mark.slow
def test_report_records_the_embedding_backend():
    report = build_text_theme_report(
        tables=_tables(), embedder=HashingEmbedder(), include_clusters=False
    )

    # Which representation produced the numbers must be visible on the report:
    # two months scored by different backends are not comparable.
    assert report.backend_name == "hashing"


@pytest.mark.slow
def test_report_is_deterministic():
    first = build_text_theme_report(
        tables=_tables(), embedder=HashingEmbedder(), include_clusters=False
    )
    second = build_text_theme_report(
        tables=_tables(), embedder=HashingEmbedder(), include_clusters=False
    )

    pd.testing.assert_frame_equal(first.theme_summary, second.theme_summary)
    pd.testing.assert_frame_equal(first.segment_themes, second.segment_themes)


@pytest.mark.slow
def test_report_can_skip_clustering():
    report = build_text_theme_report(
        tables=_tables(), embedder=HashingEmbedder(), include_clusters=False
    )

    assert report.emerging_clusters is None


# ---------------------------------------------------------------------------
# Stretch: emerging-theme clustering
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_clustering_returns_the_documented_schema():
    if not SKLEARN_AVAILABLE:
        return
    clusters = cluster_emerging_themes(_assignment(), n_clusters=5, period=SPIKE_MONTH)

    assert list(clusters.columns) == list(CLUSTER_COLUMNS)
    assert len(clusters) <= 5


@pytest.mark.slow
def test_clusters_partition_the_period():
    if not SKLEARN_AVAILABLE:
        return
    clusters = cluster_emerging_themes(_assignment(), n_clusters=5, period=SPIKE_MONTH)
    august = int(_assignment().facts["month"].eq(SPIKE_MONTH).sum())

    assert clusters["complaints"].sum() == august
    np.testing.assert_allclose(clusters["share_of_complaints"].sum(), 1.0)


@pytest.mark.slow
def test_clustering_is_deterministic():
    if not SKLEARN_AVAILABLE:
        return
    pd.testing.assert_frame_equal(
        cluster_emerging_themes(_assignment(), n_clusters=5, period=SPIKE_MONTH),
        cluster_emerging_themes(_assignment(), n_clusters=5, period=SPIKE_MONTH),
    )


@pytest.mark.slow
def test_clustering_rejects_more_clusters_than_complaints():
    if not SKLEARN_AVAILABLE:
        return
    error = _raises(ValueError, cluster_emerging_themes, _assignment(), 100_000, SPIKE_MONTH)

    assert "at least" in str(error)


# ---------------------------------------------------------------------------
# The business story: text themes vs the corrected KPI drivers
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dispute_related_themes_rise_in_august():
    summary = _summary()

    for theme in RISING_THEMES:
        row = _theme_row(summary, theme)
        assert row["complaint_change"] > 0, f"{theme} did not rise"

    # Individually a theme can grow in count yet lose share to a faster-growing
    # neighbour, so the share claim is made collectively: dispute-related themes
    # take a larger slice of the August book than they did in July.
    dispute_related = summary[summary["theme_name"].isin(RISING_THEMES)]
    assert dispute_related["share_change"].sum() > 0


@pytest.mark.slow
def test_the_largest_theme_increases_are_all_dispute_related():
    summary = _summary().sort_values("complaint_change", ascending=False)
    top_four = set(summary.head(4)["theme_name"])

    assert top_four == set(RISING_THEMES)


@pytest.mark.slow
def test_dispute_issue_volume_is_what_actually_grows_in_august():
    # The generator raises dispute-issue complaints only; fee and payment issue
    # volume does not grow. Any claim that fee or autopay themes "drove" the
    # August spike has to survive this table first.
    complaints = _tables()[TABLE_COMPLAINTS]
    complaints = complaints.assign(
        month=pd.to_datetime(complaints["date_received"]).dt.to_period("M").astype(str)
    )
    by_issue = (
        complaints[complaints["month"].isin([PRIOR_MONTH, SPIKE_MONTH])]
        .groupby(["issue", "month"])
        .size()
        .unstack("month")
    )

    dispute_issue = "Problem with a purchase shown on your statement"
    dispute_growth = (
        by_issue.loc[dispute_issue, SPIKE_MONTH] - by_issue.loc[dispute_issue, PRIOR_MONTH]
    )
    assert dispute_growth > 150

    for issue in ("Fees or interest", "Problem when making payments"):
        change = by_issue.loc[issue, SPIKE_MONTH] - by_issue.loc[issue, PRIOR_MONTH]
        assert change < dispute_growth / 10, f"{issue} grew unexpectedly ({change})"


@pytest.mark.slow
def test_travel_complaints_concentrate_the_duplicate_charge_theme():
    # driver_analysis.py puts the corrected KPI movement in travel merchants.
    # The text should agree: within travel complaints, duplicate-looking charges
    # grew sharply and are over-represented relative to the whole book.
    row = _segment_theme_row(_segments(), "travel", "duplicate_looking_travel_charge")

    assert row["complaint_change"] > 0
    assert row["current_complaints"] > 2 * row["previous_complaints"]

    labelled = _assignment().facts
    theme = labelled[labelled["theme_name"].astype(str).eq("duplicate_looking_travel_charge")]
    base_rate = float(labelled["merchant_category"].eq("travel").mean())
    theme_rate = float(theme["merchant_category"].eq("travel").mean())

    assert theme_rate > 2 * base_rate, f"theme is {theme_rate:.3f} travel vs {base_rate:.3f} base"


@pytest.mark.slow
def test_mobile_complaints_concentrate_dispute_submission_friction():
    # The other half of the driver finding was the mobile channel.
    row = _segment_theme_row(_segments(), "mobile", "mobile_dispute_submission_friction")
    assert row["complaint_change"] > 0

    labelled = _assignment().facts
    theme = labelled[labelled["theme_name"].astype(str).eq("mobile_dispute_submission_friction")]
    base_rate = float(labelled["channel"].eq("mobile").mean())
    theme_rate = float(theme["channel"].eq("mobile").mean())

    assert theme_rate > 2 * base_rate, f"theme is {theme_rate:.3f} mobile vs {base_rate:.3f} base"


@pytest.mark.slow
def test_travel_complaints_take_a_much_larger_share_of_the_august_book():
    labelled = _assignment().facts
    july = labelled[labelled["month"].eq(PRIOR_MONTH)]
    august = labelled[labelled["month"].eq(SPIKE_MONTH)]

    july_share = float(july["merchant_category"].eq("travel").mean())
    august_share = float(august["merchant_category"].eq("travel").mean())

    # Share rather than a growth ratio: total complaint volume also rises, so a
    # ratio-of-ratios sits close to its threshold and flips on small movements.
    assert august_share > 2 * july_share, f"travel share {july_share:.3f} -> {august_share:.3f}"


@pytest.mark.slow
def test_the_directional_story_holds_under_the_sentence_transformer_backend():
    # Skips cleanly when no model is cached; never downloads.
    try:
        from text_theme_analysis import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    except Exception:  # noqa: BLE001 - absent package or absent local model
        return

    summary = theme_summary_table(assignment=assign_complaint_themes(_facts(), embedder))
    for theme in RISING_THEMES:
        assert _theme_row(summary, theme)["complaint_change"] > 0, theme


# ---------------------------------------------------------------------------
# Narrative alignment: text must agree with the row it was generated from
#
# These assert properties of the generated corpus, not of the classifier, so
# they hold under every embedding backend. They are the reason the dashboard's
# representative-evidence rows are believable.
# ---------------------------------------------------------------------------

TRAVEL_VOCABULARY = r"travel|airline|hotel|trip|international"
APP_VOCABULARY = r"\bapp\b|my phone"


@lru_cache(maxsize=1)
def _raw_complaints() -> pd.DataFrame:
    complaints = _tables()[TABLE_COMPLAINTS].copy()
    complaints["lower"] = complaints["complaint_narrative"].str.lower()
    return complaints


@pytest.mark.slow
def test_travel_vocabulary_never_appears_on_a_non_travel_purchase():
    # The defect this fixes: a grocery dispute described as "the same travel
    # charge appears twice", which contradicts the row shown beside it.
    complaints = _raw_complaints()
    offenders = complaints[
        complaints["lower"].str.contains(TRAVEL_VOCABULARY, regex=True)
        & complaints["merchant_category"].ne("travel")
        & complaints["merchant_category"].ne("fee")
    ]

    assert offenders.empty, offenders["complaint_narrative"].head(3).tolist()


@pytest.mark.slow
def test_the_only_non_travel_rows_with_travel_wording_are_foreign_fee_rows():
    # A foreign transaction fee is caused by international spending, so this
    # wording is context, not contradiction. It is confined to fee rows.
    complaints = _raw_complaints()
    carriers = complaints[complaints["lower"].str.contains(TRAVEL_VOCABULARY, regex=True)]
    non_travel = carriers[carriers["merchant_category"].ne("travel")]

    assert set(non_travel["merchant_category"]) <= {"fee"}
    assert non_travel["lower"].str.contains("foreign transaction fee").all()


@pytest.mark.slow
def test_most_travel_complaints_actually_use_travel_wording():
    complaints = _raw_complaints()
    travel = complaints[complaints["merchant_category"].eq("travel")]
    rate = float(travel["lower"].str.contains(TRAVEL_VOCABULARY, regex=True).mean())

    # Not 100%: some travel complaints are deliberately vague, which is realistic.
    assert rate > 0.8, f"only {rate:.2f} of travel complaints use travel wording"


@pytest.mark.slow
def test_foreign_transaction_fee_wording_only_appears_on_foreign_fee_rows():
    complaints = _raw_complaints()
    transactions = _tables()["transactions"].set_index("transaction_id")
    carriers = complaints[complaints["lower"].str.contains("foreign transaction fee")]
    names = transactions.loc[carriers["related_transaction_id"], "merchant_name"]

    assert not carriers.empty
    assert set(names) == {"FOREIGN TRANSACTION FEE"}


@pytest.mark.slow
def test_autopay_wording_only_appears_on_failed_mobile_payments():
    complaints = _raw_complaints()
    transactions = _tables()["transactions"].set_index("transaction_id")
    carriers = complaints[complaints["lower"].str.contains("autopay")]
    related = transactions.loc[carriers["related_transaction_id"]]

    assert not carriers.empty
    assert set(related["channel"]) == {"mobile"}
    assert set(related["payment_failed"]) == {1}


@pytest.mark.slow
def test_app_wording_is_concentrated_on_mobile_rows():
    complaints = _raw_complaints()
    carriers = complaints[complaints["lower"].str.contains(APP_VOCABULARY, regex=True)]
    base_rate = float(complaints["channel"].eq("mobile").mean())
    carrier_rate = float(carriers["channel"].eq("mobile").mean())

    assert carrier_rate > 3 * base_rate, f"{carrier_rate:.3f} vs {base_rate:.3f} base"


@pytest.mark.slow
def test_non_mobile_app_wording_is_only_the_deliberately_vague_narrative():
    # One vague narrative mentions the app without claiming a mobile purchase.
    # A customer can view any transaction in the app, so this is messiness
    # rather than contradiction -- but it should be the only source.
    complaints = _raw_complaints()
    stray = complaints[
        complaints["lower"].str.contains(APP_VOCABULARY, regex=True)
        & complaints["channel"].ne("mobile")
    ]

    assert (
        stray["complaint_narrative"]
        .str.startswith("The app and the statement do not show the same thing")
        .all()
    )


@pytest.mark.slow
def test_top_representatives_never_contradict_their_own_columns():
    # The dashboard shows these rows as evidence, so they carry the most weight.
    examples = representative_complaints(_assignment(), top_n=5, period=SPIKE_MONTH)
    lowered = examples["narrative"].str.lower()

    travel_claims = examples[lowered.str.contains(TRAVEL_VOCABULARY, regex=True)]
    assert set(travel_claims["merchant_category"]) <= {"travel", "fee"}

    autopay_claims = examples[lowered.str.contains("autopay")]
    assert set(autopay_claims["channel"]) <= {"mobile"}


@pytest.mark.slow
def test_mobile_friction_representatives_are_mobile_rows():
    examples = representative_complaints(_assignment(), top_n=5, period=SPIKE_MONTH)
    friction = examples[examples["theme_name"].eq("mobile_dispute_submission_friction")]

    assert not friction.empty
    assert set(friction["channel"]) == {"mobile"}


@pytest.mark.slow
def test_duplicate_travel_representatives_are_travel_rows_under_the_real_model():
    # The hashing fallback cannot separate "the same travel charge appears twice"
    # from "the same charge appears twice" -- one token apart in a bag of words.
    # The sentence-transformer model can, so the ranking claim is checked there.
    # Skips cleanly when no model is cached; never downloads.
    try:
        from text_theme_analysis import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    except Exception:  # noqa: BLE001 - absent package or absent local model
        return

    assignment = assign_complaint_themes(_facts(), embedder)
    examples = representative_complaints(assignment, top_n=3, period=SPIKE_MONTH)
    travel_theme = examples[examples["theme_name"].eq("duplicate_looking_travel_charge")]

    assert not travel_theme.empty
    assert set(travel_theme["merchant_category"]) == {"travel"}


@pytest.mark.slow
def test_the_vague_pool_gives_unclear_or_other_a_real_population():
    summary = _summary()
    unclear = _theme_row(summary, UNCLEAR_THEME)

    # Before narratives were context-aware, no complaint was generic enough to
    # land here and the bucket was permanently empty.
    assert unclear["current_complaints"] > 0
    assert unclear["previous_complaints"] > 0


# ---------------------------------------------------------------------------
# The regenerated corpus must agree with itself and with GROUND_TRUTH.md
# ---------------------------------------------------------------------------


def _ground_truth_text() -> str:
    return (DOCS_DIR / "ground_truth.md").read_text(encoding="utf-8")


def _ground_truth_probe_table() -> pd.DataFrame:
    text = _ground_truth_text()
    marker = "## Complaint Narrative Theme Movement"
    body = text[text.index(marker) :]
    rows = [line for line in body.splitlines() if line.startswith("| ")]
    header = [cell.strip() for cell in rows[0].strip("|").split("|")]
    records = [[cell.strip() for cell in line.strip("|").split("|")] for line in rows[2:]]
    frame = pd.DataFrame(records, columns=header)
    for column in ("july_complaints", "august_complaints", "change"):
        frame[column] = frame[column].astype(int)
    return frame


def test_ground_truth_publishes_the_narrative_theme_movement():
    table = _ground_truth_probe_table()

    assert not table.empty
    assert (table["change"] == table["august_complaints"] - table["july_complaints"]).all()


@pytest.mark.slow
def test_ground_truth_probe_counts_match_the_complaint_data():
    probes = {
        "unclear merchant descriptor": r"descriptor|do not recognize|cannot identify|cannot tell which",
        "duplicate-looking charge": r"twice|duplicate",
        "travel-related charge": TRAVEL_VOCABULARY,
        "mobile app dispute friction": r"\bapp\b",
        "foreign transaction fee": r"foreign transaction fee",
        "failed autopay": r"autopay",
        "fraud or security concern": r"fraud|without permission",
    }
    complaints = _raw_complaints()
    months = pd.to_datetime(complaints["date_received"]).dt.strftime("%Y-%m")
    table = _ground_truth_probe_table().set_index("narrative_theme")

    for label, pattern in probes.items():
        for month, column in ((PRIOR_MONTH, "july_complaints"), (SPIKE_MONTH, "august_complaints")):
            expected = int(
                complaints.loc[months == month, "lower"].str.contains(pattern, regex=True).sum()
            )
            assert table.loc[label, column] == expected, f"{label} / {month}"


def test_ground_truth_prose_does_not_contradict_its_own_probe_table():
    text = _ground_truth_text()
    table = _ground_truth_probe_table()
    rising = set(table[table["change"] > 0]["narrative_theme"])
    falling = set(table[table["change"] <= 0]["narrative_theme"])

    sentence = next(
        line
        for line in text.splitlines()
        if line.startswith("Complaint narrative themes that rise")
    )
    for theme in rising:
        assert theme in sentence, f"{theme} rises but is not listed as rising"
    for theme in falling:
        assert theme not in sentence, f"{theme} does not rise but is listed as rising"


def test_ground_truth_still_reports_the_planted_defects():
    text = _ground_truth_text()

    assert "Duplicate source transactions from replayed batch: 165" in text
    assert "Missing merchant categories from card-processor batch: 1,384" in text


@pytest.mark.slow
def test_generated_tables_are_internally_consistent():
    tables = _tables()
    complaints = tables[TABLE_COMPLAINTS]
    accounts = tables[TABLE_ACCOUNTS]
    transactions = tables["transactions"]

    assert len(accounts) == 12_000
    assert len(transactions) == 929_838
    assert len(complaints) == 5_300

    assert complaints["complaint_id"].is_unique
    assert complaints["account_id"].isin(set(accounts["account_id"])).all()
    assert complaints["related_transaction_id"].isin(set(transactions["transaction_id"])).all()


@pytest.mark.slow
def test_complaint_segment_columns_match_their_source_rows():
    tables = _tables()
    complaints = tables[TABLE_COMPLAINTS]
    related = (
        tables["transactions"].set_index("transaction_id").loc[complaints["related_transaction_id"]]
    )
    account = tables[TABLE_ACCOUNTS].set_index("account_id").loc[complaints["account_id"]]

    assert (complaints["channel"].to_numpy() == related["channel"].to_numpy()).all()
    assert (complaints["fico_band"].to_numpy() == account["fico_band"].to_numpy()).all()
    assert (
        complaints["customer_segment"].to_numpy() == account["customer_segment"].to_numpy()
    ).all()

    category = complaints["merchant_category"].fillna("")
    related_category = pd.Series(related["merchant_category"].to_numpy()).fillna("")
    assert (category.to_numpy() == related_category.to_numpy()).all()


@pytest.mark.slow
def test_complaints_are_never_received_before_their_transaction():
    tables = _tables()
    complaints = tables[TABLE_COMPLAINTS]
    related = (
        tables["transactions"].set_index("transaction_id").loc[complaints["related_transaction_id"]]
    )

    received = pd.to_datetime(complaints["date_received"]).to_numpy()
    occurred = pd.to_datetime(related["transaction_date"]).to_numpy()
    assert (received >= occurred).all()


# ---------------------------------------------------------------------------
# Generated data must not be touched
# ---------------------------------------------------------------------------


def _csv_fingerprints() -> dict[str, tuple[str, int]]:
    return {
        name: (
            hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest(),
            (DATA_DIR / name).stat().st_size,
        )
        for name in CSV_FILES
    }


@pytest.mark.slow
def test_building_the_report_does_not_modify_any_generated_csv():
    before = _csv_fingerprints()

    build_text_theme_report(embedder=HashingEmbedder(), include_clusters=False)

    assert _csv_fingerprints() == before


@pytest.mark.slow
def test_text_theme_analysis_never_writes_to_the_data_directory():
    before = sorted(path.name for path in DATA_DIR.iterdir())

    build_text_theme_report(tables=_tables(), embedder=HashingEmbedder(), include_clusters=False)

    assert sorted(path.name for path in DATA_DIR.iterdir()) == before


# ---------------------------------------------------------------------------
# Direct runner, used when pytest is unavailable
# ---------------------------------------------------------------------------


def _run_without_pytest() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]

    failures: list[tuple[str, BaseException]] = []
    for name, test in tests:
        try:
            test()
        except BaseException as error:  # noqa: BLE001 - a runner reports everything
            failures.append((name, error))
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok   {name}")

    print()
    print(f"{len(tests) - len(failures)} passed, {len(failures)} failed, {len(tests)} total")
    if failures:
        print()
        print("Failures:")
        for name, error in failures:
            print(f"  - {name}: {type(error).__name__}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_without_pytest())
