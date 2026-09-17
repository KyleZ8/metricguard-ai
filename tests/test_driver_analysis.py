"""Tests for the MetricGuard AI corrected dispute-rate driver analysis.

These run under pytest. Because pytest is not installed in every environment
this project is developed in, the file is also directly runnable, matching the
other test modules in this project::

    python tests/test_driver_analysis.py

The most important assertions in this file are the provenance ones. All 165
replayed rows in the synthetic data are ``travel`` purchases on the ``mobile``
channel -- exactly the cell the demo story points at -- so driver analysis run
on raw transactions would overstate the headline driver by 91%. Several tests
below exist specifically to prove that cannot happen.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from driver_analysis import (  # noqa: E402
    ACCOUNT_SEGMENT_FIELDS,
    DECISION_TREE_RULE_COLUMNS,
    DEFAULT_MIN_DENOMINATOR,
    INTERACTION_DRIVER_COLUMNS,
    MAX_TREE_DEPTH,
    MISSING_LABEL,
    SEGMENT_DRIVER_COLUMNS,
    SEGMENT_FIELDS,
    SKLEARN_AVAILABLE,
    SUPPORTED_INTERACTIONS,
    DriverReport,
    build_driver_report,
    corrected_purchase_facts,
    count_driver_table,
    decision_tree_segments,
    heatmap_matrix,
    interaction_driver_table,
    rate_deterioration_table,
    segment_driver_table,
)
from metric_engine import duplicate_source_transactions  # noqa: E402
from quality_checks import TABLE_ACCOUNTS, TABLE_TRANSACTIONS, load_tables  # noqa: E402


SPIKE_MONTH = "2026-08"
PRIOR_MONTH = "2026-07"

# Ground truth from data/synthetic/GROUND_TRUTH.md and the planted replay batch.
EXPECTED_DUPLICATE_ROWS = 165
EXPECTED_TRAVEL_MOBILE_DISPUTED_LE660 = 111
EXPECTED_TRAVEL_MOBILE_DISPUTED_GT660 = 169

DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
CSV_FILES = (
    "accounts.csv",
    "account_monthly_snapshot.csv",
    "transactions.csv",
    "complaints.csv",
    "metric_definitions.csv",
)


@lru_cache(maxsize=1)
def _tables() -> dict[str, pd.DataFrame]:
    return load_tables()


@lru_cache(maxsize=1)
def _drivers() -> pd.DataFrame:
    return segment_driver_table(tables=_tables())


@lru_cache(maxsize=1)
def _interaction() -> pd.DataFrame:
    return interaction_driver_table("merchant_category", "channel", tables=_tables())


@lru_cache(maxsize=1)
def _report() -> DriverReport:
    return build_driver_report(tables=_tables())


def _segment_row(drivers: pd.DataFrame, segment_name: str, segment_value: str) -> pd.Series:
    matches = drivers[
        drivers["segment_name"].eq(segment_name) & drivers["segment_value"].eq(segment_value)
    ]
    assert len(matches) == 1, f"expected one row for {segment_name}={segment_value}, got {len(matches)}"
    return matches.iloc[0]


def _interaction_row(table: pd.DataFrame, value_a: str, value_b: str) -> pd.Series:
    matches = table[table["segment_a_value"].eq(value_a) & table["segment_b_value"].eq(value_b)]
    assert len(matches) == 1, f"expected one row for {value_a} x {value_b}, got {len(matches)}"
    return matches.iloc[0]


def _raises(exception_type, callable_, *args, **kwargs):
    try:
        callable_(*args, **kwargs)
    except exception_type as error:
        return error
    raise AssertionError(f"expected {exception_type.__name__} but nothing was raised")


def _toy_tables() -> dict[str, pd.DataFrame]:
    """A hand-built pair of tables whose arithmetic can be checked by eye.

    2026-01 travel/mobile: 10 purchases,  1 disputed -> 0.10
    2026-02 travel/mobile: 10 purchases,  3 disputed -> 0.30, change +2
    2026-01 grocery/web:   10 purchases,  1 disputed -> 0.10
    2026-02 grocery/web:   10 purchases,  1 disputed -> 0.10, change  0
    Plus a deliberately tiny 4-purchase gas/web segment, and one replayed
    travel/mobile dispute in 2026-02 that deduplication must remove.
    """
    accounts = pd.DataFrame(
        {
            "account_id": ["A1", "A2"],
            "fico_band": ["<=660", ">660"],
            "customer_segment": ["student", "affluent"],
            "product_type": ["student_card", "travel_rewards"],
            "region": ["South", "West"],
        }
    )

    rows: list[dict[str, object]] = []
    counter = 0

    def add(month: str, category: str, channel: str, count: int, disputed: int) -> None:
        nonlocal counter
        for index in range(count):
            counter += 1
            rows.append(
                {
                    "source_transaction_id": f"S{counter}",
                    "transaction_id": f"T{counter}",
                    "account_id": "A1" if index % 2 == 0 else "A2",
                    "transaction_date": f"{month}-05",
                    "created_at": f"{month}-05 01:00",
                    "transaction_type": "purchase",
                    "merchant_category": category,
                    "channel": channel,
                    "is_disputed": 1 if index < disputed else 0,
                }
            )

    add("2026-01", "travel", "mobile", 10, 1)
    add("2026-02", "travel", "mobile", 10, 3)
    add("2026-01", "grocery", "web", 10, 1)
    add("2026-02", "grocery", "web", 10, 1)
    add("2026-01", "gas", "web", 4, 0)
    add("2026-02", "gas", "web", 4, 1)

    transactions = pd.DataFrame(rows)
    # A replayed travel/mobile dispute: same source event, new row id.
    replay = transactions[
        transactions["transaction_date"].eq("2026-02-05")
        & transactions["merchant_category"].eq("travel")
        & transactions["is_disputed"].eq(1)
    ].head(1).copy()
    replay["transaction_id"] = "T_REPLAY"
    replay["created_at"] = "2026-02-09 09:00"
    transactions = pd.concat([transactions, replay], ignore_index=True)

    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
    transactions["created_at"] = pd.to_datetime(transactions["created_at"])
    return {TABLE_ACCOUNTS: accounts, TABLE_TRANSACTIONS: transactions}


# ---------------------------------------------------------------------------
# Provenance: driver analysis runs on deduplicated transactions
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_planted_replay_is_entirely_travel_on_mobile():
    # This is why deduplication is not optional for driver analysis: every
    # replayed row sits in the exact cell the demo story investigates.
    removed = duplicate_source_transactions(_tables()[TABLE_TRANSACTIONS])

    assert len(removed) == EXPECTED_DUPLICATE_ROWS
    assert set(removed["merchant_category"]) == {"travel"}
    assert set(removed["channel"]) == {"mobile"}
    assert set(removed["is_disputed"]) == {1}


@pytest.mark.slow
def test_corrected_facts_contain_no_duplicated_source_events():
    facts = corrected_purchase_facts(_tables())

    assert facts["source_transaction_id"].duplicated().sum() == 0


@pytest.mark.slow
def test_injecting_more_replayed_rows_does_not_change_the_driver_table():
    # The strongest available proof that deduplication is applied: replaying
    # rows that already exist must be a no-op on every driver number.
    tables = dict(_tables())
    transactions = tables[TABLE_TRANSACTIONS]
    replay = transactions[transactions["source_transaction_id"].isin(
        duplicate_source_transactions(transactions)["source_transaction_id"]
    )].copy()
    replay["transaction_id"] = "REPLAY_" + replay["transaction_id"].astype(str)
    replay["created_at"] = replay["created_at"] + pd.Timedelta(days=1)
    tables[TABLE_TRANSACTIONS] = pd.concat([transactions, replay], ignore_index=True)

    pd.testing.assert_frame_equal(segment_driver_table(tables=tables), _drivers())


@pytest.mark.slow
def test_corrected_travel_mobile_change_is_smaller_than_the_raw_change():
    tables = _tables()
    corrected = _interaction_row(_interaction(), "travel", "mobile")["disputed_change"]

    raw = tables[TABLE_TRANSACTIONS]
    purchases = raw[raw["transaction_type"].eq("purchase")].copy()
    purchases["month"] = pd.to_datetime(purchases["transaction_date"]).dt.to_period("M").astype(str)
    cell = purchases[
        purchases["merchant_category"].eq("travel") & purchases["channel"].eq("mobile")
    ]
    by_month = cell.groupby("month")["is_disputed"].sum()
    raw_change = int(by_month.get(SPIKE_MONTH, 0) - by_month.get(PRIOR_MONTH, 0))

    assert corrected < raw_change
    # Every one of the 165 replayed rows lands in this single cell.
    assert raw_change - corrected == EXPECTED_DUPLICATE_ROWS


@pytest.mark.slow
def test_toy_driver_table_reflects_deduplication():
    # The toy replay would make travel/mobile look like 4 disputes out of 11.
    drivers = segment_driver_table(tables=_toy_tables())
    travel = _segment_row(drivers, "merchant_category", "travel")

    assert travel["current_purchases"] == 10
    assert travel["current_disputed"] == 3
    assert travel["current_dispute_rate"] == 0.3
    assert travel["previous_dispute_rate"] == 0.1
    np.testing.assert_allclose(travel["rate_change"], 0.2)


# ---------------------------------------------------------------------------
# 1. Segment driver table
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_segment_driver_table_has_the_documented_schema():
    drivers = _drivers()

    assert list(drivers.columns) == list(SEGMENT_DRIVER_COLUMNS)
    assert not drivers.empty


@pytest.mark.slow
def test_segment_driver_table_covers_every_segment_field():
    drivers = _drivers()

    assert set(drivers["segment_name"]) == set(SEGMENT_FIELDS)


@pytest.mark.slow
def test_account_segments_were_joined_in():
    facts = corrected_purchase_facts(_tables())

    for field in ACCOUNT_SEGMENT_FIELDS:
        assert field in facts.columns
        assert facts[field].notna().all()


@pytest.mark.slow
def test_change_columns_are_consistent_with_their_levels():
    drivers = _drivers()

    assert (
        drivers["disputed_change"] == drivers["current_disputed"] - drivers["previous_disputed"]
    ).all()
    assert (
        drivers["purchase_change"] == drivers["current_purchases"] - drivers["previous_purchases"]
    ).all()


@pytest.mark.slow
def test_rates_equal_disputed_over_purchases():
    drivers = _drivers()
    populated = drivers[drivers["current_purchases"] > 0]

    np.testing.assert_allclose(
        populated["current_dispute_rate"],
        populated["current_disputed"] / populated["current_purchases"],
    )
    np.testing.assert_allclose(
        populated["rate_change"],
        populated["current_dispute_rate"] - populated["previous_dispute_rate"],
    )


@pytest.mark.slow
def test_count_columns_are_integers():
    drivers = _drivers()

    for column in (
        "previous_disputed",
        "current_disputed",
        "disputed_change",
        "previous_purchases",
        "current_purchases",
        "purchase_change",
    ):
        assert pd.api.types.is_integer_dtype(drivers[column]), column


@pytest.mark.slow
def test_contribution_share_sums_to_one_within_each_segment_field():
    # Each segment field partitions the same population, so shares must total
    # 1.0 per field. A pooled share across fields would double-count disputes.
    totals = _drivers().groupby("segment_name")["contribution_share_of_positive_dispute_change"].sum()

    np.testing.assert_allclose(totals.to_numpy(), np.ones(len(totals)))


@pytest.mark.slow
def test_contribution_share_is_zero_for_segments_that_did_not_grow():
    drivers = _drivers()
    non_growing = drivers[drivers["disputed_change"] <= 0]

    assert not non_growing.empty
    assert (non_growing["contribution_share_of_positive_dispute_change"] == 0).all()


@pytest.mark.slow
def test_a_zero_denominator_segment_yields_nan_not_infinity():
    missing = _segment_row(_drivers(), "merchant_category", MISSING_LABEL)

    assert missing["current_purchases"] == 0
    assert np.isnan(missing["current_dispute_rate"])
    assert np.isnan(missing["rate_change"])


@pytest.mark.slow
def test_missing_merchant_category_is_kept_as_a_visible_bucket():
    # The July batch left 1,384 purchases uncategorised. Dropping them silently
    # would shrink a denominator the headline KPI still counts.
    missing = _segment_row(_drivers(), "merchant_category", MISSING_LABEL)

    assert missing["previous_purchases"] == 1_384


def test_corrected_purchase_facts_labels_blank_segment_values_as_missing():
    tables = _toy_tables()
    transactions = tables[TABLE_TRANSACTIONS].copy()
    target = transactions.index[transactions["merchant_category"].eq("travel")][0]
    transactions.loc[target, "merchant_category"] = "   "
    tables[TABLE_TRANSACTIONS] = transactions

    facts = corrected_purchase_facts(tables)

    assert MISSING_LABEL in set(facts["merchant_category"])
    assert "   " not in set(facts["merchant_category"])


@pytest.mark.slow
def test_driver_table_is_deterministic():
    pd.testing.assert_frame_equal(segment_driver_table(tables=_tables()), _drivers())


@pytest.mark.slow
def test_driver_table_is_invariant_to_input_row_order():
    tables = dict(_tables())
    tables[TABLE_TRANSACTIONS] = tables[TABLE_TRANSACTIONS].sample(frac=1.0, random_state=5)

    pd.testing.assert_frame_equal(segment_driver_table(tables=tables), _drivers())


@pytest.mark.slow
def test_periods_default_to_the_two_most_recent_months():
    report = _report()

    assert report.current_period == SPIKE_MONTH
    assert report.previous_period == PRIOR_MONTH


@pytest.mark.slow
def test_explicit_periods_are_honoured():
    drivers = segment_driver_table("2026-03", "2026-01", tables=_tables())
    march = segment_driver_table(tables=_tables())

    assert not drivers.equals(march)
    assert drivers["current_purchases"].sum() > 0


@pytest.mark.slow
def test_unknown_period_is_rejected():
    error = _raises(KeyError, segment_driver_table, "2030-01", PRIOR_MONTH, _tables())

    assert "2030-01" in str(error)


# ---------------------------------------------------------------------------
# The business story
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_travel_is_the_strongest_count_driver():
    top = count_driver_table(_drivers()).iloc[0]

    assert top["segment_name"] == "merchant_category"
    assert top["segment_value"] == "travel"
    assert top["disputed_change"] > 0


@pytest.mark.slow
def test_travel_explains_most_of_the_merchant_category_increase():
    travel = _segment_row(_drivers(), "merchant_category", "travel")

    assert travel["contribution_share_of_positive_dispute_change"] > 0.80


@pytest.mark.slow
def test_travel_is_also_the_strongest_rate_deterioration():
    top = rate_deterioration_table(_drivers()).iloc[0]

    assert top["segment_value"] == "travel"
    assert not bool(top["min_denominator_flag"])
    assert top["rate_change"] > 0.02


@pytest.mark.slow
def test_mobile_is_the_leading_channel_driver():
    channels = _drivers()[_drivers()["segment_name"].eq("channel")]
    top_channel = channels.sort_values("disputed_change", ascending=False).iloc[0]

    assert top_channel["segment_value"] == "mobile"
    assert top_channel["rate_change"] > 0


@pytest.mark.slow
def test_subprime_band_deteriorates_faster_than_prime():
    # KPI_RESEARCH.md attributes the remaining movement partly to FICO <= 660.
    drivers = _drivers()
    subprime = _segment_row(drivers, "fico_band", "<=660")
    prime = _segment_row(drivers, "fico_band", ">660")

    assert subprime["rate_change"] > prime["rate_change"]


@pytest.mark.slow
def test_travel_on_mobile_is_the_strongest_interaction_cell():
    interaction = _interaction()

    assert interaction.iloc[0]["segment_a_value"] == "travel"
    assert interaction.iloc[0]["segment_b_value"] == "mobile"

    reliable = interaction[~interaction["min_denominator_flag"]]
    worst_rate = reliable.sort_values("rate_change", ascending=False).iloc[0]
    assert (worst_rate["segment_a_value"], worst_rate["segment_b_value"]) == ("travel", "mobile")


@pytest.mark.slow
def test_travel_on_mobile_rate_more_than_doubles():
    cell = _interaction_row(_interaction(), "travel", "mobile")

    assert cell["current_dispute_rate"] > 2 * cell["previous_dispute_rate"]


@pytest.mark.slow
def test_travel_fico_interaction_matches_ground_truth_counts():
    table = interaction_driver_table("merchant_category", "fico_band", tables=_tables())

    # GROUND_TRUTH.md records travel/mobile disputes by FICO band after dedup;
    # summing the two bands over all channels must exceed those cell counts.
    travel_rows = table[table["segment_a_value"].eq("travel")]
    assert travel_rows["current_disputed"].sum() >= (
        EXPECTED_TRAVEL_MOBILE_DISPUTED_LE660 + EXPECTED_TRAVEL_MOBILE_DISPUTED_GT660
    )


# ---------------------------------------------------------------------------
# 2. Rate deterioration and small-denominator flagging
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_rate_deterioration_is_sorted_worst_first_among_unflagged_rows():
    table = rate_deterioration_table(_drivers())
    unflagged = table[~table["min_denominator_flag"]]["rate_change"].dropna()

    assert unflagged.tolist() == sorted(unflagged.tolist(), reverse=True)


@pytest.mark.slow
def test_flagged_segments_sort_below_every_unflagged_segment():
    table = rate_deterioration_table(_drivers())
    flags = table["min_denominator_flag"].tolist()

    assert flags == sorted(flags)


@pytest.mark.slow
def test_pure_rate_ordering_is_available_when_requested():
    table = rate_deterioration_table(_drivers(), demote_flagged=False)
    changes = table["rate_change"].dropna().tolist()

    assert changes == sorted(changes, reverse=True)


@pytest.mark.slow
def test_the_uncategorised_bucket_is_flagged_on_the_real_data():
    missing = _segment_row(_drivers(), "merchant_category", MISSING_LABEL)

    assert bool(missing["min_denominator_flag"])


@pytest.mark.slow
def test_a_tiny_segment_is_flagged_and_a_large_one_is_not():
    drivers = segment_driver_table(tables=_toy_tables(), min_denominator=5)

    tiny = _segment_row(drivers, "merchant_category", "gas")  # 4 purchases per period
    large = _segment_row(drivers, "merchant_category", "travel")  # 10 per period

    assert bool(tiny["min_denominator_flag"])
    assert not bool(large["min_denominator_flag"])


@pytest.mark.slow
def test_flag_uses_the_smaller_of_the_two_periods():
    drivers = _drivers()
    smaller_period = drivers[["previous_purchases", "current_purchases"]].min(axis=1)

    expected = smaller_period < DEFAULT_MIN_DENOMINATOR
    assert drivers["min_denominator_flag"].tolist() == expected.tolist()


@pytest.mark.slow
def test_raising_the_threshold_flags_more_segments():
    lenient = segment_driver_table(tables=_tables(), min_denominator=1)
    strict = segment_driver_table(tables=_tables(), min_denominator=100_000)

    assert int(lenient["min_denominator_flag"].sum()) < int(strict["min_denominator_flag"].sum())


@pytest.mark.slow
def test_rate_deterioration_can_build_its_own_driver_table():
    table = rate_deterioration_table(tables=_tables())

    assert list(table.columns) == list(SEGMENT_DRIVER_COLUMNS)
    assert table.iloc[0]["segment_value"] == "travel"


# ---------------------------------------------------------------------------
# 3. Interaction driver table
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_interaction_table_has_the_documented_schema():
    interaction = _interaction()

    assert list(interaction.columns) == list(INTERACTION_DRIVER_COLUMNS)
    assert set(interaction["segment_a_name"]) == {"merchant_category"}
    assert set(interaction["segment_b_name"]) == {"channel"}


@pytest.mark.slow
def test_all_three_required_interaction_pairs_are_supported():
    for segment_a, segment_b in SUPPORTED_INTERACTIONS:
        table = interaction_driver_table(segment_a, segment_b, tables=_tables())

        assert not table.empty, f"{segment_a} x {segment_b} produced no rows"
        assert list(table.columns) == list(INTERACTION_DRIVER_COLUMNS)


@pytest.mark.slow
def test_interaction_cells_sum_to_the_single_field_totals():
    interaction = _interaction()
    travel_cells = interaction[interaction["segment_a_value"].eq("travel")]
    travel_total = _segment_row(_drivers(), "merchant_category", "travel")

    assert travel_cells["current_disputed"].sum() == travel_total["current_disputed"]
    assert travel_cells["current_purchases"].sum() == travel_total["current_purchases"]


@pytest.mark.slow
def test_interaction_contribution_shares_sum_to_one():
    total = _interaction()["contribution_share_of_positive_dispute_change"].sum()

    np.testing.assert_allclose(total, 1.0)


@pytest.mark.slow
def test_interaction_flags_small_cells():
    interaction = _interaction()
    flagged = interaction[interaction["min_denominator_flag"]]

    assert not flagged.empty
    assert (flagged[["previous_purchases", "current_purchases"]].min(axis=1) < DEFAULT_MIN_DENOMINATOR).all()


@pytest.mark.slow
def test_interaction_rejects_a_repeated_segment_field():
    error = _raises(ValueError, interaction_driver_table, "channel", "channel", None, None, _tables())

    assert "must differ" in str(error)


@pytest.mark.slow
def test_interaction_rejects_an_unknown_segment_field():
    error = _raises(
        ValueError, interaction_driver_table, "merchant_category", "state", None, None, _tables()
    )

    assert "unknown segment field" in str(error)


@pytest.mark.slow
def test_heatmap_matrix_pivots_the_interaction_table():
    matrix = heatmap_matrix(_interaction(), "rate_change")

    assert "travel" in matrix.index
    assert "mobile" in matrix.columns
    np.testing.assert_allclose(
        matrix.loc["travel", "mobile"],
        _interaction_row(_interaction(), "travel", "mobile")["rate_change"],
    )


@pytest.mark.slow
def test_heatmap_matrix_rejects_an_unknown_value_column():
    error = _raises(KeyError, heatmap_matrix, _interaction(), "not_a_column")

    assert "not_a_column" in str(error)


# ---------------------------------------------------------------------------
# 4. Driver report
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_driver_report_exposes_the_three_dashboard_frames():
    report = _report()

    assert isinstance(report.top_count_drivers, pd.DataFrame)
    assert isinstance(report.top_rate_deterioration, pd.DataFrame)
    assert isinstance(report.interaction_heatmap_data, pd.DataFrame)
    for frame in (
        report.top_count_drivers,
        report.top_rate_deterioration,
        report.interaction_heatmap_data,
    ):
        assert not frame.empty


@pytest.mark.slow
def test_driver_report_respects_top_n():
    report = build_driver_report(tables=_tables(), top_n=3)

    assert len(report.top_count_drivers) == 3
    assert len(report.top_rate_deterioration) == 3


@pytest.mark.slow
def test_driver_report_headlines_travel():
    report = _report()

    assert report.top_count_drivers.iloc[0]["segment_value"] == "travel"
    assert report.top_rate_deterioration.iloc[0]["segment_value"] == "travel"


@pytest.mark.slow
def test_driver_report_interaction_defaults_to_merchant_category_by_channel():
    report = _report()

    assert set(report.interaction_heatmap_data["segment_a_name"]) == {"merchant_category"}
    assert set(report.interaction_heatmap_data["segment_b_name"]) == {"channel"}


@pytest.mark.slow
def test_driver_report_can_use_a_different_interaction_pair():
    report = build_driver_report(
        tables=_tables(), interaction=("channel", "customer_segment"), include_decision_tree=False
    )

    assert set(report.interaction_heatmap_data["segment_a_name"]) == {"channel"}
    assert set(report.interaction_heatmap_data["segment_b_name"]) == {"customer_segment"}


@pytest.mark.slow
def test_driver_report_can_skip_the_decision_tree():
    report = build_driver_report(tables=_tables(), include_decision_tree=False)

    assert report.decision_tree_rules is None


# ---------------------------------------------------------------------------
# 5. Decision tree stretch
# ---------------------------------------------------------------------------


def test_decision_tree_is_available_in_this_environment():
    # Documents which branch of the stretch these tests exercise.
    assert SKLEARN_AVAILABLE, "scikit-learn missing; the tree stretch is skipped by design"


@pytest.mark.slow
def test_decision_tree_returns_readable_leaf_rules():
    if not SKLEARN_AVAILABLE:
        return
    rules = decision_tree_segments(tables=_tables())

    assert list(rules.columns) == list(DECISION_TREE_RULE_COLUMNS)
    assert not rules.empty
    assert rules["rule"].map(lambda text: isinstance(text, str) and text.strip()).all()


@pytest.mark.slow
def test_decision_tree_isolates_the_travel_mobile_population():
    if not SKLEARN_AVAILABLE:
        return
    top = decision_tree_segments(tables=_tables()).iloc[0]

    assert "merchant_category == travel" in top["rule"]
    assert "channel == mobile" in top["rule"]
    assert top["lift_vs_overall"] > 3.0


@pytest.mark.slow
def test_decision_tree_leaves_partition_the_period():
    if not SKLEARN_AVAILABLE:
        return
    rules = decision_tree_segments(tables=_tables())
    facts = corrected_purchase_facts(_tables())
    august = facts[facts["month"].eq(SPIKE_MONTH)]

    assert rules["purchases"].sum() == len(august)
    assert rules["disputed"].sum() == int(august["is_disputed"].sum())
    np.testing.assert_allclose(rules["share_of_purchases"].sum(), 1.0)


@pytest.mark.slow
def test_decision_tree_respects_the_depth_ceiling():
    if not SKLEARN_AVAILABLE:
        return
    rules = decision_tree_segments(tables=_tables(), max_depth=2)

    assert rules["depth"].max() <= 2


@pytest.mark.slow
def test_decision_tree_rejects_a_depth_above_the_ceiling():
    if not SKLEARN_AVAILABLE:
        return
    error = _raises(ValueError, decision_tree_segments, None, _tables(), MAX_TREE_DEPTH + 1)

    assert "max_depth" in str(error)


@pytest.mark.slow
def test_decision_tree_is_deterministic():
    if not SKLEARN_AVAILABLE:
        return
    pd.testing.assert_frame_equal(
        decision_tree_segments(tables=_tables()), decision_tree_segments(tables=_tables())
    )


@pytest.mark.slow
def test_decision_tree_lift_is_consistent_with_the_overall_rate():
    if not SKLEARN_AVAILABLE:
        return
    rules = decision_tree_segments(tables=_tables())
    overall = rules["disputed"].sum() / rules["purchases"].sum()

    np.testing.assert_allclose(rules["lift_vs_overall"], rules["dispute_rate"] / overall)


@pytest.mark.slow
def test_decision_tree_uses_only_categorical_segment_features():
    if not SKLEARN_AVAILABLE:
        return
    rules = decision_tree_segments(tables=_tables())
    mentioned = {
        condition.split(" ")[0]
        for rule in rules["rule"]
        if rule != "all corrected purchases"
        for condition in rule.split(" AND ")
    }

    assert mentioned.issubset(set(SEGMENT_FIELDS))


# ---------------------------------------------------------------------------
# Generated data must not be touched
# ---------------------------------------------------------------------------


def _csv_fingerprints() -> dict[str, tuple[str, int]]:
    fingerprints: dict[str, tuple[str, int]] = {}
    for name in CSV_FILES:
        path = DATA_DIR / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        fingerprints[name] = (digest, path.stat().st_size)
    return fingerprints


@pytest.mark.slow
def test_running_the_full_driver_report_does_not_modify_any_generated_csv():
    before = _csv_fingerprints()

    build_driver_report()
    interaction_driver_table("merchant_category", "fico_band")
    rate_deterioration_table(tables=load_tables())

    assert _csv_fingerprints() == before


@pytest.mark.slow
def test_driver_analysis_never_writes_to_the_data_directory():
    before = sorted(path.name for path in DATA_DIR.iterdir())

    build_driver_report(tables=_tables())

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
