"""Tests for the MetricGuard AI metric calculation and remediation module.

These run under pytest. Because pytest is not installed in every environment
this project is developed in, the file is also directly runnable, matching
``tests/test_quality_checks.py``::

    python tests/test_metric_engine.py

Assertions come in three kinds:

* arithmetic pinned to small hand-built fixtures, where the expected rate can be
  checked by eye;
* values pinned to ``data/synthetic/GROUND_TRUTH.md`` for the real tables; and
* invariance properties -- shuffling input rows, or changing which row of a
  duplicate pair survives, must not move the metric.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metric_engine import (  # noqa: E402
    METRIC_NAME,
    MONTHLY_TREND_COLUMNS,
    PERIOD_COMPARISON_COLUMNS,
    REMEDIATION_IMPACT_COLUMNS,
    VARIANT_CORRECTED,
    VARIANT_RAW,
    MetricReport,
    aggregate_metric_trend,
    available_finance_kpis,
    build_finance_metric_report,
    build_metric_report,
    deduplicate_transactions,
    duplicate_source_transactions,
    generic_monthly_trend_table,
    generic_period_comparison_table,
    generic_remediation_impact_table,
    generic_segment_driver_table,
    metric_spec,
    monthly_dispute_rate,
    monthly_trend_table,
    period_comparison_table,
    period_months_from_trend,
    remediation_impact_table,
    resolve_periods,
    resolve_tables,
    to_engine_metric_frame,
)
from quality_checks import (  # noqa: E402
    TABLE_ACCOUNTS,
    TABLE_COMPLAINTS,
    TABLE_METRIC_DEFINITIONS,
    TABLE_SNAPSHOTS,
    TABLE_TRANSACTIONS,
    load_tables,
)


# Ground truth from data/synthetic/GROUND_TRUTH.md.
SPIKE_MONTH = "2026-08"
PRIOR_MONTH = "2026-07"
EXPECTED_DUPLICATE_ROWS = 165

GROUND_TRUTH_TREND = {
    #  month:    (purchases, disputed, unique_source_events)
    "2026-01": (105_267, 1_440, 105_267),
    "2026-02": (106_086, 1_468, 106_086),
    "2026-03": (105_487, 1_396, 105_487),
    "2026-04": (105_578, 1_411, 105_578),
    "2026-05": (106_167, 1_418, 106_167),
    "2026-06": (105_847, 1_487, 105_847),
    "2026-07": (105_611, 1_406, 105_611),
    "2026-08": (105_834, 1_871, 105_669),
}


@lru_cache(maxsize=1)
def _tables() -> dict[str, pd.DataFrame]:
    return load_tables()


FINANCE_KPIS = (
    "dispute_rate",
    "fraud_claim_rate",
    "payment_failure_rate",
    "delinquency_rate_30dpd_balance",
    "net_charge_off_rate_proxy",
    "complaint_rate",
    "fee_complaint_share",
)


@lru_cache(maxsize=1)
def _transactions() -> pd.DataFrame:
    return _tables()[TABLE_TRANSACTIONS]


@lru_cache(maxsize=1)
def _trend() -> pd.DataFrame:
    return monthly_trend_table(_transactions())


@lru_cache(maxsize=1)
def _report() -> MetricReport:
    return build_metric_report(_tables())


def _toy_transactions() -> pd.DataFrame:
    """A hand-built table with arithmetic simple enough to verify by eye.

    2026-01: 4 purchases, 1 disputed                      -> raw = corrected = 0.25
    2026-02: 6 purchase rows, 3 disputed, but two of them
             are a replay of one source event             -> raw 3/6 = 0.5
                                                             corrected 2/5 = 0.4
    One payment row exists so the purchase filter is exercised.
    """
    rows = [
        # (source_id, transaction_id, date, created_at, type, disputed)
        ("S1", "T1", "2026-01-05", "2026-01-05 01:00", "purchase", 1),
        ("S2", "T2", "2026-01-06", "2026-01-06 01:00", "purchase", 0),
        ("S3", "T3", "2026-01-07", "2026-01-07 01:00", "purchase", 0),
        ("S4", "T4", "2026-01-08", "2026-01-08 01:00", "purchase", 0),
        ("S5", "T5", "2026-02-03", "2026-02-03 01:00", "purchase", 1),
        ("S6", "T6", "2026-02-04", "2026-02-04 01:00", "purchase", 1),
        # S6 replayed: same event, later ingestion, duplicate disputed row.
        ("S6", "T7", "2026-02-04", "2026-02-09 09:00", "purchase", 1),
        ("S8", "T8", "2026-02-05", "2026-02-05 01:00", "purchase", 0),
        ("S9", "T9", "2026-02-06", "2026-02-06 01:00", "purchase", 0),
        ("S10", "T10", "2026-02-07", "2026-02-07 01:00", "purchase", 0),
        ("S11", "T11", "2026-02-08", "2026-02-08 01:00", "payment", 0),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "source_transaction_id",
            "transaction_id",
            "transaction_date",
            "created_at",
            "transaction_type",
            "is_disputed",
        ],
    )
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
    frame["created_at"] = pd.to_datetime(frame["created_at"])
    return frame


def _raises(exception_type, callable_, *args, **kwargs):
    try:
        callable_(*args, **kwargs)
    except exception_type as error:
        return error
    raise AssertionError(f"expected {exception_type.__name__} but nothing was raised")


# ---------------------------------------------------------------------------
# 1. Loading and accepting tables
# ---------------------------------------------------------------------------


def test_resolve_tables_returns_all_five_synthetic_tables_when_none_supplied():
    tables = resolve_tables()

    assert set(tables) == {
        TABLE_ACCOUNTS,
        TABLE_SNAPSHOTS,
        TABLE_TRANSACTIONS,
        TABLE_COMPLAINTS,
        TABLE_METRIC_DEFINITIONS,
    }
    assert not tables[TABLE_TRANSACTIONS].empty


def test_resolve_tables_passes_supplied_tables_through_without_reading_disk():
    toy = {TABLE_TRANSACTIONS: _toy_transactions()}

    resolved = resolve_tables(toy)

    assert resolved[TABLE_TRANSACTIONS] is toy[TABLE_TRANSACTIONS]


def test_resolve_tables_accepts_the_older_engine_key_naming():
    tables = _tables()
    engine_style = {
        "transactions": tables[TABLE_TRANSACTIONS],
        "snapshots": tables[TABLE_SNAPSHOTS],
    }

    resolved = resolve_tables(engine_style)

    assert TABLE_SNAPSHOTS in resolved
    assert "snapshots" not in resolved


def test_build_metric_report_accepts_injected_tables():
    report = build_metric_report({TABLE_TRANSACTIONS: _toy_transactions()})

    assert report.current_period == "2026-02"
    assert report.previous_period == "2026-01"


# ---------------------------------------------------------------------------
# 3. Deduplication
# ---------------------------------------------------------------------------


def test_deduplication_removes_the_planted_replay_rows():
    transactions = _transactions()

    deduped = deduplicate_transactions(transactions)

    assert len(transactions) - len(deduped) == EXPECTED_DUPLICATE_ROWS
    assert deduped["source_transaction_id"].duplicated().sum() == 0


def test_deduplication_keeps_the_earliest_created_at_row():
    deduped = deduplicate_transactions(_toy_transactions())

    survivor = deduped[deduped["source_transaction_id"].eq("S6")]
    assert len(survivor) == 1
    assert survivor.iloc[0]["transaction_id"] == "T6"


def test_deduplication_preserves_original_row_order():
    transactions = _transactions()

    deduped = deduplicate_transactions(transactions)

    assert deduped.index.is_monotonic_increasing
    assert deduped["transaction_id"].tolist() == transactions.loc[deduped.index, "transaction_id"].tolist()


def test_deduplication_is_invariant_to_input_row_order():
    toy = _toy_transactions()
    shuffled = toy.sample(frac=1.0, random_state=7)

    from_ordered = set(deduplicate_transactions(toy)["transaction_id"])
    from_shuffled = set(deduplicate_transactions(shuffled)["transaction_id"])

    assert from_ordered == from_shuffled


def test_deduplication_breaks_created_at_ties_deterministically():
    toy = _toy_transactions()
    # Force an exact tie between the two S6 rows.
    toy.loc[toy["transaction_id"].eq("T7"), "created_at"] = toy.loc[
        toy["transaction_id"].eq("T6"), "created_at"
    ].iloc[0]

    first = deduplicate_transactions(toy)
    second = deduplicate_transactions(toy.sample(frac=1.0, random_state=3))

    assert set(first["transaction_id"]) == set(second["transaction_id"])
    assert first[first["source_transaction_id"].eq("S6")].iloc[0]["transaction_id"] == "T6"


def test_deduplication_keeps_rows_without_a_source_event_id():
    toy = _toy_transactions()
    toy.loc[toy["transaction_id"].isin(["T8", "T9"]), "source_transaction_id"] = np.nan

    deduped = deduplicate_transactions(toy)

    assert {"T8", "T9"}.issubset(set(deduped["transaction_id"]))


def test_deduplication_is_a_no_op_on_already_unique_source_events():
    unique = _transactions().drop_duplicates("source_transaction_id").head(2_000)

    deduped = deduplicate_transactions(unique)

    pd.testing.assert_frame_equal(deduped, unique)


def test_deduplication_requires_the_source_event_key():
    toy = _toy_transactions().drop(columns=["source_transaction_id"])

    error = _raises(KeyError, deduplicate_transactions, toy)

    assert "source_transaction_id" in str(error)


def test_duplicate_source_transactions_returns_exactly_the_removed_rows():
    transactions = _transactions()

    removed = duplicate_source_transactions(transactions)
    kept = deduplicate_transactions(transactions)

    assert len(removed) == EXPECTED_DUPLICATE_ROWS
    assert len(removed) + len(kept) == len(transactions)
    assert removed.index.intersection(kept.index).empty


def test_removed_duplicates_are_all_disputed_august_purchases():
    removed = duplicate_source_transactions(_transactions())

    assert set(removed["transaction_type"]) == {"purchase"}
    assert set(removed["is_disputed"]) == {1}
    months = pd.to_datetime(removed["transaction_date"]).dt.to_period("M").astype(str)
    assert set(months) == {SPIKE_MONTH}


# ---------------------------------------------------------------------------
# 2 and 4. Raw and corrected monthly dispute rate
# ---------------------------------------------------------------------------


def test_toy_raw_dispute_rate_is_hand_checkable():
    raw = monthly_dispute_rate(_toy_transactions(), deduped=False).set_index("month")

    assert raw.loc["2026-01", "dispute_rate"] == 0.25
    assert raw.loc["2026-02", "dispute_rate"] == 0.5
    assert raw.loc["2026-02", "purchase_transactions"] == 6
    assert raw.loc["2026-02", "disputed_purchases"] == 3


def test_toy_corrected_dispute_rate_drops_the_replayed_dispute():
    corrected = monthly_dispute_rate(_toy_transactions(), deduped=True).set_index("month")

    assert corrected.loc["2026-02", "purchase_transactions"] == 5
    assert corrected.loc["2026-02", "disputed_purchases"] == 2
    assert corrected.loc["2026-02", "dispute_rate"] == 0.4
    # January had no duplicates, so correction must not move it.
    assert corrected.loc["2026-01", "dispute_rate"] == 0.25


def test_dispute_rate_counts_only_purchase_transactions():
    toy = _toy_transactions()

    raw = monthly_dispute_rate(toy, deduped=False).set_index("month")
    payments = int(toy["transaction_type"].eq("payment").sum())

    assert payments == 1
    assert raw["purchase_transactions"].sum() == len(toy) - payments


def test_monthly_dispute_rate_labels_its_variant_and_metric():
    raw = monthly_dispute_rate(_toy_transactions(), deduped=False)
    corrected = monthly_dispute_rate(_toy_transactions(), deduped=True)

    assert set(raw["variant"]) == {VARIANT_RAW}
    assert set(corrected["variant"]) == {VARIANT_CORRECTED}
    assert set(raw["metric_name"]) == {METRIC_NAME}


def test_monthly_dispute_rate_handles_a_table_with_no_purchases():
    payments_only = _toy_transactions().query("transaction_type == 'payment'")

    result = monthly_dispute_rate(payments_only)

    assert result.empty
    assert "dispute_rate" in result.columns


def test_raw_monthly_counts_match_ground_truth():
    trend = _trend().set_index("month")

    for month, (purchases, disputed, _) in GROUND_TRUTH_TREND.items():
        assert trend.loc[month, "purchase_transactions_raw"] == purchases, month
        assert trend.loc[month, "disputed_purchases_raw"] == disputed, month


def test_corrected_monthly_denominator_matches_ground_truth_unique_source_events():
    trend = _trend().set_index("month")

    for month, (_, _, unique_source_events) in GROUND_TRUTH_TREND.items():
        assert trend.loc[month, "purchase_transactions_corrected"] == unique_source_events, month


def test_ground_truth_dispute_rates_are_reproduced_to_four_decimals():
    trend = _trend().set_index("month")

    assert round(float(trend.loc[SPIKE_MONTH, "dispute_rate_raw"]), 4) == 0.0177
    assert round(float(trend.loc[SPIKE_MONTH, "dispute_rate_corrected"]), 4) == 0.0161
    assert round(float(trend.loc[PRIOR_MONTH, "dispute_rate_raw"]), 4) == 0.0133


def test_months_before_the_replay_are_identical_raw_and_corrected():
    trend = _trend()
    untouched = trend[trend["month"] < SPIKE_MONTH]

    assert len(untouched) == 7
    assert (untouched["duplicate_purchase_rows_removed"] == 0).all()
    assert (untouched["dispute_rate_difference"] == 0).all()
    pd.testing.assert_series_equal(
        untouched["dispute_rate_raw"],
        untouched["dispute_rate_corrected"],
        check_names=False,
    )


def test_correction_lowers_august_but_leaves_it_above_july():
    trend = _trend().set_index("month")

    raw = float(trend.loc[SPIKE_MONTH, "dispute_rate_raw"])
    corrected = float(trend.loc[SPIKE_MONTH, "dispute_rate_corrected"])
    july = float(trend.loc[PRIOR_MONTH, "dispute_rate_raw"])

    # This is the whole demo story: part data quality, part real movement.
    assert corrected < raw
    assert corrected > july * 1.15


def test_metric_is_invariant_to_which_row_of_a_duplicate_pair_survives():
    # Deduplication keeps the earliest created_at. In this dataset the replay
    # batch is often the earlier row, so it is worth proving the choice cannot
    # move the metric: both rows of every pair agree on the fields it reads.
    transactions = _transactions()
    latest_kept = (
        transactions.sort_values(["source_transaction_id", "created_at", "transaction_id"])
        .drop_duplicates("source_transaction_id", keep="last")
        .sort_index()
    )

    from_earliest = monthly_dispute_rate(transactions, deduped=True).set_index("month")["dispute_rate"]
    from_latest = monthly_dispute_rate(latest_kept, deduped=False).set_index("month")["dispute_rate"]

    pd.testing.assert_series_equal(from_earliest, from_latest, check_names=False)


# ---------------------------------------------------------------------------
# 7a. Monthly trend table
# ---------------------------------------------------------------------------


def test_monthly_trend_table_has_the_documented_schema():
    trend = _trend()

    assert list(trend.columns) == list(MONTHLY_TREND_COLUMNS)
    assert len(trend) == len(GROUND_TRUTH_TREND)


def test_monthly_trend_months_are_sorted_and_unique():
    trend = _trend()

    assert trend["month"].is_unique
    assert trend["month"].tolist() == sorted(trend["month"].tolist())


def test_monthly_trend_count_columns_are_integers():
    trend = _trend()

    for column in (
        "purchase_transactions_raw",
        "disputed_purchases_raw",
        "purchase_transactions_corrected",
        "disputed_purchases_corrected",
        "duplicate_purchase_rows_removed",
        "duplicate_disputed_rows_removed",
    ):
        assert pd.api.types.is_integer_dtype(trend[column]), column


def test_monthly_trend_is_internally_consistent():
    trend = _trend()

    assert (
        trend["duplicate_purchase_rows_removed"]
        == trend["purchase_transactions_raw"] - trend["purchase_transactions_corrected"]
    ).all()
    assert (
        trend["duplicate_disputed_rows_removed"]
        == trend["disputed_purchases_raw"] - trend["disputed_purchases_corrected"]
    ).all()
    np.testing.assert_allclose(
        trend["dispute_rate_difference"],
        trend["dispute_rate_raw"] - trend["dispute_rate_corrected"],
    )


def test_monthly_trend_rates_equal_numerator_over_denominator():
    trend = _trend()

    np.testing.assert_allclose(
        trend["dispute_rate_raw"],
        trend["disputed_purchases_raw"] / trend["purchase_transactions_raw"],
    )
    np.testing.assert_allclose(
        trend["dispute_rate_corrected"],
        trend["disputed_purchases_corrected"] / trend["purchase_transactions_corrected"],
    )


def test_monthly_trend_only_ever_removes_rows():
    trend = _trend()

    assert (trend["duplicate_purchase_rows_removed"] >= 0).all()
    assert (trend["dispute_rate_difference"] >= 0).all()


# ---------------------------------------------------------------------------
# 5 and 7b. Period comparison
# ---------------------------------------------------------------------------


def test_period_comparison_has_the_documented_schema_and_both_variants():
    comparison = _report().period_comparison

    assert list(comparison.columns) == list(PERIOD_COMPARISON_COLUMNS)
    assert comparison["variant"].tolist() == [VARIANT_RAW, VARIANT_CORRECTED]


def test_period_comparison_defaults_to_the_two_most_recent_months():
    report = _report()

    assert report.current_period == SPIKE_MONTH
    assert report.previous_period == PRIOR_MONTH


def test_period_comparison_change_arithmetic_is_consistent():
    comparison = _report().period_comparison

    np.testing.assert_allclose(
        comparison["absolute_change"],
        comparison["current_value"] - comparison["previous_value"],
    )
    np.testing.assert_allclose(
        comparison["percent_change"],
        comparison["absolute_change"] / comparison["previous_value"],
    )


def test_period_comparison_numerators_and_denominators_match_the_trend():
    trend = _trend().set_index("month")
    comparison = _report().period_comparison.set_index("variant")

    assert comparison.loc[VARIANT_RAW, "current_numerator"] == trend.loc[SPIKE_MONTH, "disputed_purchases_raw"]
    assert (
        comparison.loc[VARIANT_CORRECTED, "current_denominator"]
        == trend.loc[SPIKE_MONTH, "purchase_transactions_corrected"]
    )


def test_raw_movement_looks_larger_than_corrected_movement():
    comparison = _report().period_comparison.set_index("variant")

    assert (
        comparison.loc[VARIANT_RAW, "percent_change"]
        > comparison.loc[VARIANT_CORRECTED, "percent_change"]
    )
    assert comparison.loc[VARIANT_CORRECTED, "percent_change"] > 0


def test_period_comparison_accepts_explicit_periods():
    comparison = period_comparison_table(_trend(), "2026-03", "2026-01")

    assert set(comparison["current_period"]) == {"2026-03"}
    assert set(comparison["previous_period"]) == {"2026-01"}


def test_toy_period_comparison_is_hand_checkable():
    comparison = period_comparison_table(monthly_trend_table(_toy_transactions())).set_index("variant")

    raw = comparison.loc[VARIANT_RAW]
    assert raw["previous_value"] == 0.25
    assert raw["current_value"] == 0.5
    assert raw["absolute_change"] == 0.25
    assert raw["percent_change"] == 1.0

    corrected = comparison.loc[VARIANT_CORRECTED]
    assert corrected["current_value"] == 0.4
    np.testing.assert_allclose(corrected["percent_change"], 0.6)


def test_unknown_period_is_rejected_with_the_available_months_listed():
    error = _raises(KeyError, period_comparison_table, _trend(), "2027-01", PRIOR_MONTH)

    assert "2027-01" in str(error)
    assert SPIKE_MONTH in str(error)


def test_earliest_month_cannot_be_used_as_a_current_period_by_default():
    error = _raises(ValueError, resolve_periods, _trend(), "2026-01")

    assert "earliest" in str(error)


def test_a_single_month_of_history_cannot_be_compared():
    single_month = _trend().head(1)

    error = _raises(ValueError, resolve_periods, single_month)

    assert "at least two months" in str(error)


# ---------------------------------------------------------------------------
# 6 and 7c. Remediation impact
# ---------------------------------------------------------------------------


def test_remediation_impact_has_the_documented_schema_and_one_row():
    impact = _report().remediation_impact

    assert list(impact.columns) == list(REMEDIATION_IMPACT_COLUMNS)
    assert len(impact) == 1


def test_remediation_impact_reports_the_six_required_quantities_for_august():
    impact = _report().remediation_impact.iloc[0]

    assert impact["period"] == SPIKE_MONTH
    assert round(float(impact["raw_dispute_rate"]), 4) == 0.0177
    assert round(float(impact["corrected_dispute_rate"]), 4) == 0.0161
    assert round(float(impact["dispute_rate_difference"]), 4) == 0.0015
    assert impact["raw_disputed_count"] == 1_871
    assert impact["corrected_disputed_count"] == 1_706
    assert impact["duplicate_disputed_rows_removed"] == EXPECTED_DUPLICATE_ROWS


def test_remediation_impact_is_internally_consistent():
    impact = _report().remediation_impact.iloc[0]

    assert (
        impact["raw_disputed_count"] - impact["corrected_disputed_count"]
        == impact["duplicate_disputed_rows_removed"]
    )
    assert (
        impact["raw_purchase_count"] - impact["corrected_purchase_count"]
        == impact["duplicate_purchase_rows_removed"]
    )
    np.testing.assert_allclose(
        impact["dispute_rate_difference"],
        impact["raw_dispute_rate"] - impact["corrected_dispute_rate"],
    )


def test_remediation_impact_is_all_zero_for_a_month_without_duplicates():
    impact = remediation_impact_table(_trend(), PRIOR_MONTH).iloc[0]

    assert impact["duplicate_disputed_rows_removed"] == 0
    assert impact["duplicate_purchase_rows_removed"] == 0
    assert impact["dispute_rate_difference"] == 0.0
    assert impact["raw_dispute_rate"] == impact["corrected_dispute_rate"]


def test_remediation_impact_defaults_to_the_latest_month():
    impact = remediation_impact_table(_trend()).iloc[0]

    assert impact["period"] == SPIKE_MONTH


def test_remediation_impact_rejects_an_unknown_period():
    error = _raises(KeyError, remediation_impact_table, _trend(), "1999-01")

    assert "1999-01" in str(error)


def test_toy_remediation_impact_is_hand_checkable():
    impact = remediation_impact_table(monthly_trend_table(_toy_transactions()), "2026-02").iloc[0]

    assert impact["raw_dispute_rate"] == 0.5
    assert impact["corrected_dispute_rate"] == 0.4
    np.testing.assert_allclose(impact["dispute_rate_difference"], 0.1)
    assert impact["raw_disputed_count"] == 3
    assert impact["corrected_disputed_count"] == 2
    assert impact["duplicate_disputed_rows_removed"] == 1


# ---------------------------------------------------------------------------
# Report assembly and determinism
# ---------------------------------------------------------------------------


def test_metric_report_exposes_the_three_dashboard_frames():
    report = _report()

    assert report.metric_name == METRIC_NAME
    for frame in (report.monthly_trend, report.period_comparison, report.remediation_impact):
        assert isinstance(frame, pd.DataFrame)
        assert not frame.empty


def test_building_the_report_twice_gives_identical_frames():
    first = build_metric_report(_tables())
    second = build_metric_report(_tables())

    pd.testing.assert_frame_equal(first.monthly_trend, second.monthly_trend)
    pd.testing.assert_frame_equal(first.period_comparison, second.period_comparison)
    pd.testing.assert_frame_equal(first.remediation_impact, second.remediation_impact)


def test_trend_is_invariant_to_input_row_order():
    shuffled = _transactions().sample(frac=1.0, random_state=11)

    pd.testing.assert_frame_equal(monthly_trend_table(shuffled), _trend())


def test_report_frames_contain_no_infinite_values():
    report = _report()

    for frame in (report.monthly_trend, report.period_comparison, report.remediation_impact):
        numeric = frame.select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_a_zero_denominator_yields_nan_rather_than_infinity():
    empty_month = _toy_transactions().copy()
    empty_month["is_disputed"] = 0
    empty_month["transaction_type"] = "payment"
    trend = monthly_trend_table(pd.concat([_toy_transactions(), empty_month], ignore_index=True))

    assert np.isfinite(trend["dispute_rate_raw"]).all()


# ---------------------------------------------------------------------------
# Compatibility with the original metricguard_engine
# ---------------------------------------------------------------------------


def test_engine_adapter_emits_the_schema_the_old_engine_expects():
    frame = to_engine_metric_frame(_trend(), VARIANT_CORRECTED)

    assert {"month", "numerator", "denominator", "metric_value", "metric_name", "deduped"}.issubset(
        frame.columns
    )
    assert set(frame["deduped"]) == {True}


def test_engine_adapter_output_runs_through_the_old_anomaly_detector():
    from metricguard_engine import rolling_anomaly_flags

    flagged = rolling_anomaly_flags(to_engine_metric_frame(_trend(), VARIANT_CORRECTED))

    assert bool(flagged.set_index("month").loc[SPIKE_MONTH, "is_anomaly"])


def test_engine_adapter_output_runs_through_the_old_period_result():
    from metricguard_engine import metric_period_result

    result = metric_period_result(
        to_engine_metric_frame(_trend(), VARIANT_RAW), METRIC_NAME, SPIKE_MONTH, PRIOR_MONTH
    )

    assert result.current_numerator == 1_871
    np.testing.assert_allclose(result.current_value, 1_871 / 105_834)


def test_new_and_old_engines_agree_on_the_raw_dispute_rate():
    from metricguard_engine import monthly_dispute_rate as legacy_monthly_dispute_rate

    legacy = legacy_monthly_dispute_rate(_transactions(), deduped=False).set_index("month")
    current = _trend().set_index("month")

    np.testing.assert_allclose(
        current["dispute_rate_raw"].to_numpy(dtype=float),
        legacy["metric_value"].reindex(current.index).to_numpy(dtype=float),
    )


def test_engine_adapter_rejects_an_unknown_variant():
    error = _raises(ValueError, to_engine_metric_frame, _trend(), "deduped")

    assert "variant must be" in str(error)


# ---------------------------------------------------------------------------
# Generic finance KPI selector contract
# ---------------------------------------------------------------------------


def test_all_finance_kpis_are_exposed_in_a_stable_order():
    assert available_finance_kpis() == FINANCE_KPIS
    assert [metric_spec(name).metric_name for name in FINANCE_KPIS] == list(FINANCE_KPIS)


def test_every_finance_kpi_has_a_monthly_raw_and_corrected_trend():
    for metric_name in FINANCE_KPIS:
        trend = generic_monthly_trend_table(_tables(), metric_name)

        assert len(trend) == 8, metric_name
        assert set(trend["month"]) == set(GROUND_TRUTH_TREND)
        for column in (
            "raw_numerator",
            "raw_denominator",
            "raw_value",
            "corrected_numerator",
            "corrected_denominator",
            "corrected_value",
            "duplicate_numerator_removed",
            "duplicate_denominator_removed",
        ):
            assert column in trend.columns, (metric_name, column)
        assert trend["corrected_denominator"].gt(0).all(), metric_name
        assert np.isfinite(trend["corrected_value"]).all(), metric_name


def test_every_finance_kpi_supports_explicit_period_comparison_and_remediation():
    for metric_name in FINANCE_KPIS:
        trend = generic_monthly_trend_table(_tables(), metric_name)
        comparison = generic_period_comparison_table(
            trend, metric_name, SPIKE_MONTH, PRIOR_MONTH
        )
        impact = generic_remediation_impact_table(trend, metric_name, SPIKE_MONTH).iloc[0]

        assert set(comparison["variant"]) == {VARIANT_RAW, VARIANT_CORRECTED}
        assert impact["period"] == SPIKE_MONTH
        assert impact["metric_name"] == metric_name
        assert impact["raw_denominator"] >= impact["corrected_denominator"]
        assert impact["raw_numerator"] >= impact["corrected_numerator"]


def test_every_finance_kpi_builds_a_dashboard_report_with_segment_drivers():
    for metric_name in FINANCE_KPIS:
        report = build_finance_metric_report(
            metric_name, _tables(), current_period=SPIKE_MONTH, previous_period=PRIOR_MONTH
        )

        assert report.metric_name == metric_name
        assert report.current_period == SPIKE_MONTH
        assert report.previous_period == PRIOR_MONTH
        assert report.display_name == metric_spec(metric_name).display_name
        assert not report.monthly_trend.empty
        assert not report.period_comparison.empty
        assert not report.remediation_impact.empty
        assert not report.segment_drivers.empty
        assert report.metric_definition["metric_name"] == metric_name


def test_duplicate_remediation_only_changes_purchase_transaction_kpis_in_august():
    changed = []
    unchanged = []
    for metric_name in FINANCE_KPIS:
        impact = build_finance_metric_report(
            metric_name, _tables(), current_period=SPIKE_MONTH, previous_period=PRIOR_MONTH
        ).remediation_impact.iloc[0]
        if impact["duplicate_denominator_removed"] or impact["duplicate_numerator_removed"]:
            changed.append(metric_name)
        else:
            unchanged.append(metric_name)

    assert changed == ["dispute_rate", "fraud_claim_rate"]
    assert set(unchanged) == set(FINANCE_KPIS) - set(changed)


def test_quarterly_trend_aggregates_three_month_windows():
    monthly = generic_monthly_trend_table(_tables(), "dispute_rate")
    quarterly = aggregate_metric_trend(monthly, "quarterly")
    latest = quarterly.iloc[-1]
    months = ("2026-06", "2026-07", "2026-08")
    selected = monthly[monthly["month"].isin(months)]

    assert latest["month"] == "2026-06 to 2026-08"
    assert latest["period_months"] == months
    assert latest["corrected_numerator"] == selected["corrected_numerator"].sum()
    assert latest["corrected_denominator"] == selected["corrected_denominator"].sum()
    np.testing.assert_allclose(
        latest["corrected_value"],
        selected["corrected_numerator"].sum() / selected["corrected_denominator"].sum(),
    )


def test_semiannual_report_uses_six_month_windows_and_records_the_months():
    report = build_finance_metric_report("payment_failure_rate", _tables(), period_grain="semiannual")

    assert len(report.monthly_trend) == 3
    assert report.period_grain == "semiannual"
    assert report.current_period == "2026-03 to 2026-08"
    assert report.current_months == ("2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08")
    assert period_months_from_trend(report.monthly_trend, report.previous_period) == report.previous_months


def test_generic_segment_drivers_compare_full_period_windows():
    monthly = generic_segment_driver_table(
        _tables(), "dispute_rate", "2026-08", "2026-07", period_grain="monthly"
    )
    quarterly = generic_segment_driver_table(
        _tables(), "dispute_rate", "2026-06 to 2026-08", "2026-03 to 2026-05", period_grain="quarterly"
    )

    monthly_travel = monthly[
        monthly["segment_name"].eq("merchant_category") & monthly["segment_value"].eq("travel")
    ].iloc[0]
    quarterly_travel = quarterly[
        quarterly["segment_name"].eq("merchant_category") & quarterly["segment_value"].eq("travel")
    ].iloc[0]

    assert quarterly_travel["current_denominator"] > monthly_travel["current_denominator"]
    assert quarterly_travel["previous_denominator"] > monthly_travel["previous_denominator"]


def test_unknown_period_grain_is_rejected():
    error = _raises(ValueError, aggregate_metric_trend, generic_monthly_trend_table(_tables(), "dispute_rate"), "weekly")

    assert "period_grain must be" in str(error)


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
