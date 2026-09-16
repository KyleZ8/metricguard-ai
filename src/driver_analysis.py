"""Corrected dispute-rate driver analysis for MetricGuard AI.

This module owns the fourth step of the MetricGuard workflow::

    metric movement -> data quality -> anomaly detection -> DRIVER ANALYSIS -> ...

``quality_checks.py`` asks whether the records can be trusted. ``metric_engine.py``
quantifies how much of the reported spike was duplicate rows and how much
survives correction. This module answers what is left: **which parts of the
portfolio explain the movement that is actually real?**

Every number here is computed on corrected source events. The module routes all
of its input through :func:`metric_engine.deduplicate_transactions`, so a
replayed batch can never be mistaken for a business driver. That matters
concretely in this dataset: all 165 replayed rows are disputed August travel
purchases, so driver analysis run on raw data would overstate exactly the
segment the analyst is trying to size.

Two different questions get two different tables, and confusing them is the
classic driver-analysis mistake:

``count`` drivers (``contribution_share_of_positive_dispute_change``)
    Where did the extra disputes come from in absolute terms? Large segments
    dominate, because they have the volume to move a portfolio total.

``rate`` drivers (``rate_change``)
    Where did customer experience actually deteriorate? Small segments can lead
    here, which is why every row carries ``min_denominator_flag``.

Contribution share is computed **within** each ``segment_name``, not pooled
across all of them. Pooling would be meaningless: ``merchant_category`` and
``channel`` each partition the same population, so their changes sum to the same
total and a pooled share would double-count every dispute.

Everything is deterministic pandas. No LLM call. The optional decision-tree
stretch uses scikit-learn with a fixed ``random_state``; when scikit-learn is
absent the rest of the module is unaffected.

Run directly to print the full driver report::

    python src/driver_analysis.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from metric_engine import (
    METRIC_NAME,
    PURCHASE_TRANSACTION_TYPE,
    deduplicate_transactions,
    resolve_periods,
    resolve_tables,
)
from quality_checks import DATA_DIR, TABLE_ACCOUNTS, TABLE_TRANSACTIONS


try:  # The decision-tree stretch is optional by design.
    from sklearn.tree import DecisionTreeClassifier, _tree as _sklearn_tree

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where sklearn is absent
    DecisionTreeClassifier = None  # type: ignore[assignment]
    _sklearn_tree = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False


# Segment fields carried on the transaction itself.
TRANSACTION_SEGMENT_FIELDS = ("merchant_category", "channel")

# Segment fields that must be joined in from the account dimension.
ACCOUNT_SEGMENT_FIELDS = ("fico_band", "customer_segment", "product_type", "region")

SEGMENT_FIELDS = TRANSACTION_SEGMENT_FIELDS + ACCOUNT_SEGMENT_FIELDS

SUPPORTED_INTERACTIONS = (
    ("merchant_category", "channel"),
    ("merchant_category", "fico_band"),
    ("channel", "customer_segment"),
)

# A null dimension becomes its own visible bucket rather than being dropped.
# The July card-processor batch left 1,384 purchases without a merchant
# category; silently discarding them would quietly shrink a denominator the
# headline KPI still counts.
MISSING_LABEL = "__missing__"

# Below this many purchases in either period, a segment rate is too noisy to
# rank on. At a ~1.3% base rate, 500 purchases implies roughly 6 disputes and a
# relative standard error near 40%, so the rate moves on almost nothing.
DEFAULT_MIN_DENOMINATOR = 500

SEGMENT_DRIVER_COLUMNS = (
    "segment_name",
    "segment_value",
    "previous_disputed",
    "current_disputed",
    "disputed_change",
    "previous_purchases",
    "current_purchases",
    "purchase_change",
    "previous_dispute_rate",
    "current_dispute_rate",
    "rate_change",
    "contribution_share_of_positive_dispute_change",
    "min_denominator_flag",
)

INTERACTION_KEY_COLUMNS = (
    "segment_a_name",
    "segment_a_value",
    "segment_b_name",
    "segment_b_value",
)

INTERACTION_DRIVER_COLUMNS = INTERACTION_KEY_COLUMNS + SEGMENT_DRIVER_COLUMNS[2:]

_MEASURE_COLUMNS = SEGMENT_DRIVER_COLUMNS[2:]

DECISION_TREE_RULE_COLUMNS = (
    "rule",
    "depth",
    "purchases",
    "disputed",
    "dispute_rate",
    "lift_vs_overall",
    "share_of_purchases",
)


@dataclass(frozen=True, eq=False)
class DriverReport:
    """Dashboard-ready driver frames for one period comparison.

    ``eq=False`` because a generated ``__eq__`` would compare DataFrames
    element-wise and raise on truth-value ambiguity.
    """

    metric_name: str
    current_period: str
    previous_period: str
    segment_drivers: pd.DataFrame
    top_count_drivers: pd.DataFrame
    top_rate_deterioration: pd.DataFrame
    interaction_heatmap_data: pd.DataFrame
    decision_tree_rules: pd.DataFrame | None


# ---------------------------------------------------------------------------
# Corrected fact table
# ---------------------------------------------------------------------------


def corrected_purchase_facts(
    tables: Mapping[str, pd.DataFrame] | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Deduplicated purchase transactions enriched with account segments.

    This is the single fact table every driver view is built from. Routing all
    of them through :func:`metric_engine.deduplicate_transactions` is what makes
    the whole module "corrected" rather than raw.
    """
    resolved = resolve_tables(tables, data_dir)
    transactions = resolved[TABLE_TRANSACTIONS]

    deduped = deduplicate_transactions(transactions)
    purchases = deduped[deduped["transaction_type"].eq(PURCHASE_TRANSACTION_TYPE)].copy()
    purchases["month"] = pd.to_datetime(purchases["transaction_date"]).dt.to_period("M").astype(str)

    accounts = resolved.get(TABLE_ACCOUNTS)
    if accounts is not None:
        join_columns = ["account_id"] + [
            field for field in ACCOUNT_SEGMENT_FIELDS if field in accounts.columns
        ]
        purchases = purchases.merge(accounts[join_columns], on="account_id", how="left")

    for field in SEGMENT_FIELDS:
        if field in purchases.columns:
            purchases[field] = (
                purchases[field].astype("object").where(purchases[field].notna(), MISSING_LABEL).astype(str)
            )

    return purchases


def available_segment_fields(facts: pd.DataFrame) -> tuple[str, ...]:
    """Segment fields actually present on the fact table."""
    return tuple(field for field in SEGMENT_FIELDS if field in facts.columns)


def _resolve_facts_and_periods(
    tables: Mapping[str, pd.DataFrame] | None,
    current_period: str | None,
    previous_period: str | None,
    data_dir: Path,
) -> tuple[pd.DataFrame, str, str]:
    """Build the corrected fact table and settle which two months to compare."""
    facts = corrected_purchase_facts(tables, data_dir)
    months = pd.DataFrame({"month": sorted(facts["month"].unique())})
    # Reusing metric_engine's resolver keeps period defaulting and the error
    # messages identical across the two modules.
    resolved_current, resolved_previous = resolve_periods(months, current_period, previous_period)
    return facts, resolved_current, resolved_previous


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Rate that yields NaN rather than inf when a period has no purchases."""
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    den = pd.to_numeric(denominator, errors="coerce").astype(float)
    return num.divide(den).where(den.gt(0))


def _movement_frame(
    facts: pd.DataFrame,
    group_columns: Sequence[str],
    current_period: str,
    previous_period: str,
    min_denominator: int,
) -> pd.DataFrame:
    """Period-over-period dispute movement for an arbitrary grouping.

    Shared by the single-field and interaction views so both compute their
    measures identically.
    """
    window = facts[facts["month"].isin([current_period, previous_period])]
    grouped = (
        window.groupby(list(group_columns) + ["month"], dropna=False)["is_disputed"]
        .agg(disputed="sum", purchases="size")
        .unstack("month")
    )

    # A pair present in only one period must still appear, with a zero on the
    # missing side, or the analyst never sees that it started or stopped.
    for measure in ("disputed", "purchases"):
        for period in (previous_period, current_period):
            if (measure, period) not in grouped.columns:
                grouped[(measure, period)] = 0
    grouped = grouped.fillna(0)

    frame = pd.DataFrame(
        {
            "previous_disputed": grouped[("disputed", previous_period)].astype(int),
            "current_disputed": grouped[("disputed", current_period)].astype(int),
            "previous_purchases": grouped[("purchases", previous_period)].astype(int),
            "current_purchases": grouped[("purchases", current_period)].astype(int),
        }
    ).reset_index()

    frame["disputed_change"] = frame["current_disputed"] - frame["previous_disputed"]
    frame["purchase_change"] = frame["current_purchases"] - frame["previous_purchases"]
    frame["previous_dispute_rate"] = _safe_rate(frame["previous_disputed"], frame["previous_purchases"])
    frame["current_dispute_rate"] = _safe_rate(frame["current_disputed"], frame["current_purchases"])
    frame["rate_change"] = frame["current_dispute_rate"] - frame["previous_dispute_rate"]
    frame["min_denominator_flag"] = (
        frame[["previous_purchases", "current_purchases"]].min(axis=1) < min_denominator
    )
    return frame


def _add_contribution_share(frame: pd.DataFrame, partition_by: str | None = None) -> pd.DataFrame:
    """Share of the total positive dispute increase attributable to each row.

    Computed within ``partition_by`` when given, because each segment field
    partitions the same population independently.
    """
    out = frame.copy()

    def _share(group: pd.Series) -> pd.Series:
        positive_total = group[group > 0].sum()
        if positive_total <= 0:
            return pd.Series(0.0, index=group.index)
        return group.where(group > 0, 0.0) / positive_total

    if partition_by is None:
        out["contribution_share_of_positive_dispute_change"] = _share(out["disputed_change"])
    else:
        out["contribution_share_of_positive_dispute_change"] = (
            out.groupby(partition_by, group_keys=False)["disputed_change"].apply(_share)
        )
    return out


# ---------------------------------------------------------------------------
# 1. Segment driver table
# ---------------------------------------------------------------------------


def segment_driver_table(
    current_period: str | None = None,
    previous_period: str | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    min_denominator: int = DEFAULT_MIN_DENOMINATOR,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Corrected dispute movement for every segment value of every segment field.

    Periods default to the two most recent months present in the corrected data.
    """
    facts, resolved_current, resolved_previous = _resolve_facts_and_periods(
        tables, current_period, previous_period, data_dir
    )

    frames: list[pd.DataFrame] = []
    for field in available_segment_fields(facts):
        movement = _movement_frame(
            facts, [field], resolved_current, resolved_previous, min_denominator
        )
        movement = movement.rename(columns={field: "segment_value"})
        movement.insert(0, "segment_name", field)
        frames.append(movement)

    if not frames:
        return pd.DataFrame(columns=list(SEGMENT_DRIVER_COLUMNS))

    drivers = pd.concat(frames, ignore_index=True)
    drivers = _add_contribution_share(drivers, partition_by="segment_name")
    drivers["segment_value"] = drivers["segment_value"].astype(str)

    return (
        drivers[list(SEGMENT_DRIVER_COLUMNS)]
        .sort_values(
            ["segment_name", "disputed_change", "segment_value"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def count_driver_table(drivers: pd.DataFrame) -> pd.DataFrame:
    """Segments ranked by how many additional disputes they contributed."""
    return drivers.sort_values(
        ["disputed_change", "segment_name", "segment_value"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Rate deterioration table
# ---------------------------------------------------------------------------


def rate_deterioration_table(
    drivers: pd.DataFrame | None = None,
    demote_flagged: bool = True,
    **kwargs: object,
) -> pd.DataFrame:
    """Segments ranked by dispute-rate deterioration, worst first.

    Rows whose smaller period has fewer than ``min_denominator`` purchases keep
    their ``min_denominator_flag`` and, by default, sort below every unflagged
    row. A 40%-relative-error rate at the top of a ranked table is how an
    analyst ends up investigating noise. Pass ``demote_flagged=False`` for a
    pure ``rate_change`` ordering.

    Accepts a prepared ``drivers`` frame, or builds one from the same keyword
    arguments :func:`segment_driver_table` takes.
    """
    frame = segment_driver_table(**kwargs) if drivers is None else drivers.copy()  # type: ignore[arg-type]

    sort_columns = ["rate_change", "segment_name", "segment_value"]
    ascending = [False, True, True]
    if demote_flagged:
        sort_columns.insert(0, "min_denominator_flag")
        ascending.insert(0, True)

    # NaN rate_change (a segment absent from one period) sorts last either way.
    return frame.sort_values(
        sort_columns, ascending=ascending, na_position="last", kind="mergesort"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Interaction driver table
# ---------------------------------------------------------------------------


def interaction_driver_table(
    segment_a: str,
    segment_b: str,
    current_period: str | None = None,
    previous_period: str | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    min_denominator: int = DEFAULT_MIN_DENOMINATOR,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Corrected dispute movement for a pair of segment fields.

    A single-field view can hide a driver that lives in the intersection: travel
    and mobile can each look moderate while travel-on-mobile is extreme.
    Contribution share is computed across the whole table here, because the pair
    partitions the population exactly once.
    """
    if segment_a == segment_b:
        raise ValueError(f"segment_a and segment_b must differ, both were {segment_a!r}")
    for field in (segment_a, segment_b):
        if field not in SEGMENT_FIELDS:
            raise ValueError(f"unknown segment field {field!r}; expected one of {list(SEGMENT_FIELDS)}")

    facts, resolved_current, resolved_previous = _resolve_facts_and_periods(
        tables, current_period, previous_period, data_dir
    )
    for field in (segment_a, segment_b):
        if field not in facts.columns:
            raise KeyError(f"segment field {field!r} is not available on the loaded tables")

    movement = _movement_frame(
        facts, [segment_a, segment_b], resolved_current, resolved_previous, min_denominator
    )
    movement = movement.rename(
        columns={segment_a: "segment_a_value", segment_b: "segment_b_value"}
    )
    movement.insert(0, "segment_a_name", segment_a)
    movement.insert(2, "segment_b_name", segment_b)
    movement["segment_a_value"] = movement["segment_a_value"].astype(str)
    movement["segment_b_value"] = movement["segment_b_value"].astype(str)

    movement = _add_contribution_share(movement, partition_by=None)

    return (
        movement[list(INTERACTION_DRIVER_COLUMNS)]
        .sort_values(
            ["disputed_change", "segment_a_value", "segment_b_value"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def heatmap_matrix(interaction_table: pd.DataFrame, value_column: str = "rate_change") -> pd.DataFrame:
    """Pivot an interaction table into a matrix for heatmap rendering."""
    if value_column not in interaction_table.columns:
        raise KeyError(f"{value_column!r} is not a column of the interaction table")
    return interaction_table.pivot(
        index="segment_a_value", columns="segment_b_value", values=value_column
    ).sort_index()


# ---------------------------------------------------------------------------
# 5. Optional shallow decision tree
# ---------------------------------------------------------------------------

MAX_TREE_DEPTH = 4


def _tree_rules(model, feature_names: Sequence[str]) -> dict[int, tuple[str, int]]:
    """Readable condition path and depth for each leaf of a fitted tree.

    One-hot features are named ``field=value``, so a ``<= 0.5`` branch reads as
    "is not that value" and a ``> 0.5`` branch as "is that value".
    """
    tree = model.tree_
    paths: dict[int, tuple[str, int]] = {}

    def walk(node: int, conditions: list[str], depth: int) -> None:
        if tree.feature[node] == _sklearn_tree.TREE_UNDEFINED:
            rule = " AND ".join(conditions) if conditions else "all corrected purchases"
            paths[node] = (rule, depth)
            return
        field, _, value = feature_names[tree.feature[node]].partition("=")
        walk(tree.children_left[node], conditions + [f"{field} != {value}"], depth + 1)
        walk(tree.children_right[node], conditions + [f"{field} == {value}"], depth + 1)

    walk(0, [], 0)
    return paths


def decision_tree_segments(
    current_period: str | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    max_depth: int = 3,
    min_samples_leaf: int = 1_000,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Shallow decision tree over categorical segments, returned as leaf rules.

    The tree is a search tool, not a predictive model: at a ~1.6% base rate it
    would score well by predicting "no dispute" everywhere. What is useful is
    the partition it finds -- which combination of categorical segments isolates
    the highest-dispute population -- expressed as readable rules the analyst can
    check against the count and rate tables.

    Deterministic: fixed ``random_state``, sorted one-hot columns, and a
    ``min_samples_leaf`` floor that keeps leaves large enough to interpret.

    Raises ``ImportError`` when scikit-learn is unavailable; the rest of this
    module does not depend on it.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is not installed, so the decision-tree segmentation stretch is "
            "unavailable. Every other driver function works without it."
        )
    if not 1 <= max_depth <= MAX_TREE_DEPTH:
        raise ValueError(f"max_depth must be between 1 and {MAX_TREE_DEPTH}, got {max_depth}")

    facts, resolved_current, _ = _resolve_facts_and_periods(tables, current_period, None, data_dir)
    period_facts = facts[facts["month"].eq(resolved_current)]
    if period_facts.empty:
        return pd.DataFrame(columns=list(DECISION_TREE_RULE_COLUMNS))

    fields = list(available_segment_fields(period_facts))
    features = pd.get_dummies(period_facts[fields].astype(str), prefix_sep="=")
    features = features.reindex(sorted(features.columns), axis=1).astype(np.uint8)
    target = period_facts["is_disputed"].astype(int).to_numpy()

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=0,
    ).fit(features.to_numpy(), target)

    leaf_ids = model.apply(features.to_numpy())
    rules = _tree_rules(model, features.columns.tolist())

    summary = (
        pd.DataFrame({"leaf": leaf_ids, "is_disputed": target})
        .groupby("leaf")["is_disputed"]
        .agg(disputed="sum", purchases="size")
        .reset_index()
    )
    summary["rule"] = summary["leaf"].map(lambda leaf: rules[leaf][0])
    summary["depth"] = summary["leaf"].map(lambda leaf: rules[leaf][1])
    summary["dispute_rate"] = summary["disputed"] / summary["purchases"]

    overall_rate = float(target.mean())
    summary["lift_vs_overall"] = (
        summary["dispute_rate"] / overall_rate if overall_rate else np.nan
    )
    summary["share_of_purchases"] = summary["purchases"] / len(period_facts)

    return (
        summary[list(DECISION_TREE_RULE_COLUMNS)]
        .sort_values(["dispute_rate", "purchases"], ascending=[False, False], kind="mergesort")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 4. Report assembly
# ---------------------------------------------------------------------------


def build_driver_report(
    current_period: str | None = None,
    previous_period: str | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    top_n: int = 5,
    interaction: tuple[str, str] = SUPPORTED_INTERACTIONS[0],
    min_denominator: int = DEFAULT_MIN_DENOMINATOR,
    include_decision_tree: bool = True,
    data_dir: Path = DATA_DIR,
) -> DriverReport:
    """Assemble every dashboard-ready driver frame in one pass.

    ``decision_tree_rules`` is ``None`` when scikit-learn is unavailable or when
    the caller opts out; no other frame depends on it.
    """
    resolved_tables = resolve_tables(tables, data_dir)
    drivers = segment_driver_table(
        current_period, previous_period, resolved_tables, min_denominator, data_dir
    )
    facts, resolved_current, resolved_previous = _resolve_facts_and_periods(
        resolved_tables, current_period, previous_period, data_dir
    )

    tree_rules: pd.DataFrame | None = None
    if include_decision_tree and SKLEARN_AVAILABLE:
        tree_rules = decision_tree_segments(
            resolved_current, resolved_tables, data_dir=data_dir
        )

    return DriverReport(
        metric_name=METRIC_NAME,
        current_period=resolved_current,
        previous_period=resolved_previous,
        segment_drivers=drivers,
        top_count_drivers=count_driver_table(drivers).head(top_n).reset_index(drop=True),
        top_rate_deterioration=rate_deterioration_table(drivers).head(top_n).reset_index(drop=True),
        interaction_heatmap_data=interaction_driver_table(
            interaction[0],
            interaction[1],
            resolved_current,
            resolved_previous,
            resolved_tables,
            min_denominator,
            data_dir,
        ),
        decision_tree_rules=tree_rules,
    )


if __name__ == "__main__":
    report = build_driver_report()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    print(f"MetricGuard AI - corrected {report.metric_name} driver analysis")
    print(f"{report.current_period} vs {report.previous_period} (deduplicated source events)")
    print("=" * 110)

    print()
    print("Top count drivers (share of the positive dispute increase, within each segment field)")
    print("-" * 110)
    print(
        report.top_count_drivers[
            [
                "segment_name",
                "segment_value",
                "previous_disputed",
                "current_disputed",
                "disputed_change",
                "contribution_share_of_positive_dispute_change",
                "min_denominator_flag",
            ]
        ].to_string(index=False)
    )

    print()
    print("Top rate deterioration (unflagged segments first)")
    print("-" * 110)
    print(
        report.top_rate_deterioration[
            [
                "segment_name",
                "segment_value",
                "previous_dispute_rate",
                "current_dispute_rate",
                "rate_change",
                "current_purchases",
                "min_denominator_flag",
            ]
        ].to_string(index=False)
    )

    print()
    print("Interaction: merchant_category x channel (top 10 by dispute change)")
    print("-" * 110)
    print(
        report.interaction_heatmap_data.head(10)[
            [
                "segment_a_value",
                "segment_b_value",
                "disputed_change",
                "previous_dispute_rate",
                "current_dispute_rate",
                "rate_change",
                "current_purchases",
                "min_denominator_flag",
            ]
        ].to_string(index=False)
    )

    print()
    if report.decision_tree_rules is None:
        print("Decision-tree segmentation: skipped (scikit-learn not installed)")
    else:
        print("Decision-tree leaf rules (max_depth=3, corrected purchases)")
        print("-" * 110)
        print(report.decision_tree_rules.head(8).to_string(index=False))
