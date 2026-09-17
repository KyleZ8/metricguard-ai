"""Tests for the MetricGuard AI data-quality and observability module.

These run under pytest. Because pytest is not installed in every environment
this project is developed in, the file is also directly runnable::

    python tests/test_quality_checks.py

Two kinds of assertion appear here:

* checks run against the real synthetic tables, pinned to the defects recorded
  in ``data/synthetic/GROUND_TRUTH.md``; and
* checks run against small mutated copies, which prove a check actually fires
  when its rule is violated. Without the second kind, a check that always
  returned "pass" would look healthy on clean data.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quality_checks import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    RESULT_COLUMNS,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARN,
    TABLE_ACCOUNTS,
    TABLE_COMPLAINTS,
    TABLE_METRIC_DEFINITIONS,
    TABLE_SNAPSHOTS,
    TABLE_TRANSACTIONS,
    Thresholds,
    _is_blank,
    check_accepted_values,
    check_amount_sign_validity,
    check_category_distribution_drift,
    check_duplicate_source_transaction_ids,
    check_kpi_anomaly_dispute_rate,
    check_metric_definition_governance,
    check_monthly_row_count_anomaly,
    check_null_rate_drift,
    check_posted_date_not_before_transaction_date,
    check_referential_integrity,
    check_required_field_completeness,
    floored_zscore,
    load_tables,
    population_stability_index,
    run_quality_checks,
    summarize_quality_checks,
)


# Ground truth recorded in data/synthetic/GROUND_TRUTH.md.
EXPECTED_DUPLICATE_SOURCE_ROWS = 165
EXPECTED_MISSING_MERCHANT_CATEGORY_ROWS = 1_384
DEFECT_MONTH_MISSING_CATEGORY = "2026-07"
SPIKE_MONTH = "2026-08"


# The transactions table has ~930k rows, so load once and share.
@lru_cache(maxsize=1)
def _tables() -> dict[str, pd.DataFrame]:
    return load_tables()


@lru_cache(maxsize=1)
def _report() -> pd.DataFrame:
    return run_quality_checks(_tables())


def _row(report: pd.DataFrame, check_name: str) -> pd.Series:
    matches = report[report["check_name"].eq(check_name)]
    assert len(matches) == 1, f"expected exactly one row for {check_name}, got {len(matches)}"
    return matches.iloc[0]


def _only(rows: list[dict[str, object]]) -> dict[str, object]:
    assert len(rows) == 1, f"expected a single check row, got {len(rows)}"
    return rows[0]


def _find(rows: list[dict[str, object]], check_name: str) -> dict[str, object]:
    matches = [row for row in rows if row["check_name"] == check_name]
    assert len(matches) == 1, f"expected exactly one row for {check_name}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Report contract
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_report_has_the_required_columns_in_order():
    report = _report()

    assert list(report.columns) == list(RESULT_COLUMNS)
    assert not report.empty


@pytest.mark.slow
def test_report_uses_only_the_allowed_status_severity_and_type_values():
    report = _report()

    assert set(report["status"]).issubset({STATUS_PASS, STATUS_WARN, STATUS_FAIL})
    assert set(report["severity"]).issubset({"none", "low", "medium", "high"})
    assert set(report["check_type"]).issubset({"rule_based", "statistical", "governance"})


@pytest.mark.slow
def test_passing_checks_carry_no_severity_and_failing_checks_do():
    report = _report()

    assert set(report.loc[report["status"].eq(STATUS_PASS), "severity"]) <= {"none"}
    non_passing = report[report["status"].ne(STATUS_PASS)]
    assert not non_passing.empty
    assert "none" not in set(non_passing["severity"])


@pytest.mark.slow
def test_affected_rows_is_a_non_negative_integer_column():
    report = _report()

    assert pd.api.types.is_integer_dtype(report["affected_rows"])
    assert (report["affected_rows"] >= 0).all()


@pytest.mark.slow
def test_every_required_check_family_is_present():
    names = set(_report()["check_name"])

    expected = {
        # 1 duplicate source events
        "duplicate_source_transaction_id",
        # 2 required field completeness, one row per table
        f"required_field_completeness__{TABLE_ACCOUNTS}",
        f"required_field_completeness__{TABLE_TRANSACTIONS}",
        f"required_field_completeness__{TABLE_COMPLAINTS}",
        f"required_field_completeness__{TABLE_SNAPSHOTS}",
        # 3 date ordering
        "posted_date_not_before_transaction_date",
        # 4 amount sign validity
        "amount_sign_validity",
        # 5 accepted values
        f"accepted_values__{TABLE_TRANSACTIONS}__transaction_type",
        f"accepted_values__{TABLE_TRANSACTIONS}__channel",
        f"accepted_values__{TABLE_TRANSACTIONS}__merchant_category",
        f"accepted_values__{TABLE_ACCOUNTS}__fico_band",
        f"accepted_values__{TABLE_ACCOUNTS}__product_type",
        f"accepted_values__{TABLE_COMPLAINTS}__issue",
        # 6 referential integrity
        f"referential_integrity__{TABLE_TRANSACTIONS}__account_id",
        f"referential_integrity__{TABLE_COMPLAINTS}__account_id",
        f"referential_integrity__{TABLE_SNAPSHOTS}__account_id",
        # 7 row-count anomaly
        f"monthly_row_count_anomaly__{TABLE_TRANSACTIONS}",
        f"monthly_row_count_anomaly__{TABLE_COMPLAINTS}",
        f"monthly_row_count_anomaly__{TABLE_SNAPSHOTS}",
        # 8 null-rate drift
        f"null_rate_drift__{TABLE_TRANSACTIONS}__merchant_category",
        # 9 category-distribution drift
        f"category_distribution_drift__{TABLE_TRANSACTIONS}__merchant_category",
        f"category_distribution_drift__{TABLE_TRANSACTIONS}__channel",
        # 10 KPI anomaly
        "kpi_anomaly__dispute_rate",
        # 11 metric governance
        "metric_definition_governance_completeness",
    }
    assert expected.issubset(names), f"missing checks: {sorted(expected - names)}"


@pytest.mark.slow
def test_check_names_are_unique():
    report = _report()

    assert report["check_name"].is_unique


@pytest.mark.slow
def test_report_leads_with_failures_then_warnings_then_passes():
    statuses = _report()["status"].tolist()
    rank = {STATUS_FAIL: 0, STATUS_WARN: 1, STATUS_PASS: 2}
    ranks = [rank[status] for status in statuses]

    assert ranks == sorted(ranks)


@pytest.mark.slow
def test_every_row_explains_itself_and_recommends_an_action():
    report = _report()

    for column in ("explanation", "recommended_action", "observed_value", "expected_value"):
        assert report[column].map(lambda text: isinstance(text, str) and text.strip()).all()

    passing = report[report["status"].eq(STATUS_PASS)]
    assert (passing["recommended_action"] == "No action required.").all()


@pytest.mark.slow
def test_running_the_checks_twice_gives_an_identical_report():
    first = run_quality_checks(_tables())
    second = run_quality_checks(_tables())

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.slow
def test_accepts_the_engine_table_key_naming():
    tables = _tables()
    engine_style = {
        "accounts": tables[TABLE_ACCOUNTS],
        "snapshots": tables[TABLE_SNAPSHOTS],
        "transactions": tables[TABLE_TRANSACTIONS],
        "complaints": tables[TABLE_COMPLAINTS],
        "metric_definitions": tables[TABLE_METRIC_DEFINITIONS],
    }

    pd.testing.assert_frame_equal(run_quality_checks(engine_style), _report())


# ---------------------------------------------------------------------------
# 1. Duplicate source_transaction_id
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_duplicate_source_transaction_check_finds_the_planted_replay():
    row = _row(_report(), "duplicate_source_transaction_id")

    assert row["status"] == STATUS_FAIL
    assert row["severity"] == "high"
    assert row["affected_rows"] == EXPECTED_DUPLICATE_SOURCE_ROWS
    assert row["table_name"] == TABLE_TRANSACTIONS


@pytest.mark.slow
def test_duplicate_source_transaction_check_passes_on_unique_ids():
    clean = _tables()[TABLE_TRANSACTIONS].drop_duplicates("source_transaction_id").head(500)
    row = _only(check_duplicate_source_transaction_ids(clean))

    assert row["status"] == STATUS_PASS
    assert row["affected_rows"] == 0


# ---------------------------------------------------------------------------
# 2. Required field completeness
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_completeness_fails_for_transactions_because_merchant_category_is_missing():
    row = _row(_report(), f"required_field_completeness__{TABLE_TRANSACTIONS}")

    assert row["status"] == STATUS_FAIL
    assert row["affected_rows"] == EXPECTED_MISSING_MERCHANT_CATEGORY_ROWS
    assert row["affected_columns"] == "merchant_category"


@pytest.mark.slow
def test_completeness_passes_for_the_other_three_tables():
    report = _report()

    for table in (TABLE_ACCOUNTS, TABLE_COMPLAINTS, TABLE_SNAPSHOTS):
        row = _row(report, f"required_field_completeness__{table}")
        assert row["status"] == STATUS_PASS, f"{table} unexpectedly incomplete"
        assert row["affected_rows"] == 0


def test_is_blank_catches_whitespace_and_empty_strings_on_both_string_dtypes():
    # pandas 3.x changed pd.api.types.is_string_dtype(): it no longer matches
    # plain object dtype (only the newer string/StringDtype), so _is_blank
    # must check both, not just one or the other. A regression here would
    # silently stop flagging blanks on whichever dtype it drops.
    expected = [True, True, True, False]

    object_series = pd.Series(["   ", "", None, "ok"], dtype=object)
    assert _is_blank(object_series).tolist() == expected

    string_series = pd.Series(["   ", "", None, "ok"], dtype="string")
    assert _is_blank(string_series).tolist() == expected

    # A non-string dtype must never be flagged just because it has a null.
    numeric_series = pd.Series([1, 2, 3, None])
    assert _is_blank(numeric_series).tolist() == [False, False, False, True]


@pytest.mark.slow
def test_completeness_treats_a_whitespace_only_string_as_missing():
    tables = dict(_tables())
    complaints = tables[TABLE_COMPLAINTS].head(50).copy()
    complaints.loc[complaints.index[0], "complaint_narrative"] = "   "
    tables[TABLE_COMPLAINTS] = complaints

    rows = check_required_field_completeness(tables)
    row = _find(rows, f"required_field_completeness__{TABLE_COMPLAINTS}")

    assert row["status"] == STATUS_FAIL
    assert row["affected_rows"] == 1
    assert row["affected_columns"] == "complaint_narrative"


@pytest.mark.slow
def test_completeness_fails_when_a_required_column_is_absent_entirely():
    tables = dict(_tables())
    tables[TABLE_ACCOUNTS] = tables[TABLE_ACCOUNTS].head(20).drop(columns=["fico_band"])

    row = _find(
        check_required_field_completeness(tables), f"required_field_completeness__{TABLE_ACCOUNTS}"
    )

    assert row["status"] == STATUS_FAIL
    assert "missing columns: fico_band" in row["observed_value"]


# ---------------------------------------------------------------------------
# 3. posted_date >= transaction_date
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_posted_date_ordering_passes_on_the_synthetic_data():
    row = _row(_report(), "posted_date_not_before_transaction_date")

    assert row["status"] == STATUS_PASS
    assert row["affected_rows"] == 0


@pytest.mark.slow
def test_posted_date_ordering_fails_when_a_transaction_posts_before_it_happens():
    transactions = _tables()[TABLE_TRANSACTIONS].head(100).copy()
    transactions.loc[transactions.index[0], "posted_date"] = transactions.loc[
        transactions.index[0], "transaction_date"
    ] - pd.Timedelta(days=1)

    row = _only(check_posted_date_not_before_transaction_date(transactions))

    assert row["status"] == STATUS_FAIL
    assert row["affected_rows"] == 1
    assert row["severity"] == "high"


@pytest.mark.slow
def test_posted_date_ordering_allows_same_day_posting():
    transactions = _tables()[TABLE_TRANSACTIONS].head(100).copy()
    transactions["posted_date"] = transactions["transaction_date"]

    assert _only(check_posted_date_not_before_transaction_date(transactions))["status"] == STATUS_PASS


# ---------------------------------------------------------------------------
# 4. Amount sign validity
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_amount_sign_validity_passes_on_the_synthetic_data():
    row = _row(_report(), "amount_sign_validity")

    assert row["status"] == STATUS_PASS
    assert row["affected_rows"] == 0


@pytest.mark.slow
def test_amount_sign_validity_catches_each_of_the_four_sign_rules():
    transactions = _tables()[TABLE_TRANSACTIONS].head(400).copy()
    cases = {
        "purchase_amount_not_positive": ("purchase", 0, -5.0),
        "fee_amount_not_positive": ("fee", 0, -5.0),
        "successful_payment_amount_not_negative": ("payment", 0, 25.0),
        "failed_payment_amount_not_zero": ("payment", 1, -25.0),
    }

    for index, (rule_name, (kind, failed, amount)) in enumerate(cases.items()):
        broken = transactions.copy()
        target = broken.index[index]
        broken.loc[target, ["transaction_type", "payment_failed", "transaction_amount"]] = [
            kind,
            failed,
            amount,
        ]

        row = _only(check_amount_sign_validity(broken))

        assert row["status"] == STATUS_FAIL, f"{rule_name} was not caught"
        assert row["affected_rows"] == 1, f"{rule_name} counted {row['affected_rows']} rows"
        assert f"{rule_name}=1" in row["observed_value"]


@pytest.mark.slow
def test_amount_sign_validity_accepts_a_zero_amount_failed_payment():
    transactions = _tables()[TABLE_TRANSACTIONS].head(200).copy()
    target = transactions.index[0]
    transactions.loc[target, ["transaction_type", "payment_failed", "transaction_amount"]] = [
        "payment",
        1,
        0.0,
    ]

    assert _only(check_amount_sign_validity(transactions))["status"] == STATUS_PASS


# ---------------------------------------------------------------------------
# 5. Accepted values
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_all_accepted_value_checks_pass_on_the_synthetic_data():
    report = _report()
    accepted_rows = report[report["check_name"].str.startswith("accepted_values__")]

    assert len(accepted_rows) == 6
    assert set(accepted_rows["status"]) == {STATUS_PASS}


@pytest.mark.slow
def test_accepted_values_fails_on_an_unmapped_category():
    tables = dict(_tables())
    transactions = tables[TABLE_TRANSACTIONS].head(200).copy()
    transactions.loc[transactions.index[0], "merchant_category"] = "crypto_atm"
    tables[TABLE_TRANSACTIONS] = transactions

    row = _find(
        check_accepted_values(tables), f"accepted_values__{TABLE_TRANSACTIONS}__merchant_category"
    )

    assert row["status"] == STATUS_FAIL
    assert row["affected_rows"] == 1
    assert "crypto_atm" in row["observed_value"]


@pytest.mark.slow
def test_accepted_values_ignores_nulls_so_completeness_is_not_double_counted():
    tables = dict(_tables())
    transactions = tables[TABLE_TRANSACTIONS].head(200).copy()
    transactions.loc[transactions.index[:3], "merchant_category"] = np.nan
    tables[TABLE_TRANSACTIONS] = transactions

    row = _find(
        check_accepted_values(tables), f"accepted_values__{TABLE_TRANSACTIONS}__merchant_category"
    )

    assert row["status"] == STATUS_PASS
    assert row["affected_rows"] == 0


def test_accepted_values_ignores_blank_strings_so_completeness_is_not_double_counted():
    tables = {
        TABLE_TRANSACTIONS: pd.DataFrame(
            {
                "transaction_type": ["purchase", "payment", "fee"],
                "channel": ["mobile", "web", "card_present"],
                "merchant_category": ["", "   ", np.nan],
            }
        ),
        TABLE_ACCOUNTS: pd.DataFrame(
            {
                "fico_band": [">660", "<=660"],
                "product_type": ["cash_rewards", "student_card"],
            }
        ),
        TABLE_COMPLAINTS: pd.DataFrame(
            {
                "issue": [
                    "Problem with a purchase shown on your statement",
                    "Fees or interest",
                ]
            }
        ),
    }

    row = _find(
        check_accepted_values(tables), f"accepted_values__{TABLE_TRANSACTIONS}__merchant_category"
    )

    assert row["status"] == STATUS_PASS
    assert row["affected_rows"] == 0


# ---------------------------------------------------------------------------
# 6. Referential integrity
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_referential_integrity_passes_for_all_three_child_tables():
    report = _report()

    for table in (TABLE_TRANSACTIONS, TABLE_COMPLAINTS, TABLE_SNAPSHOTS):
        row = _row(report, f"referential_integrity__{table}__account_id")
        assert row["status"] == STATUS_PASS, f"{table} has orphan account_id values"
        assert row["affected_rows"] == 0


@pytest.mark.slow
def test_referential_integrity_fails_on_an_orphan_account_id():
    tables = dict(_tables())
    complaints = tables[TABLE_COMPLAINTS].head(30).copy()
    complaints.loc[complaints.index[0], "account_id"] = "ACCT999999999"
    tables[TABLE_COMPLAINTS] = complaints

    row = _find(
        check_referential_integrity(tables), f"referential_integrity__{TABLE_COMPLAINTS}__account_id"
    )

    assert row["status"] == STATUS_FAIL
    assert row["affected_rows"] == 1
    assert row["severity"] == "high"


# ---------------------------------------------------------------------------
# 7. Monthly row-count anomaly
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_row_count_anomaly_flags_the_august_complaint_surge():
    row = _row(_report(), f"monthly_row_count_anomaly__{TABLE_COMPLAINTS}")

    assert row["status"] == STATUS_FAIL
    assert SPIKE_MONTH in row["observed_value"]


@pytest.mark.slow
def test_row_count_anomaly_passes_for_steady_transaction_and_snapshot_volume():
    report = _report()

    for table in (TABLE_TRANSACTIONS, TABLE_SNAPSHOTS):
        row = _row(report, f"monthly_row_count_anomaly__{table}")
        assert row["status"] == STATUS_PASS, f"{table} volume unexpectedly flagged"


@pytest.mark.slow
def test_row_count_anomaly_flags_a_duplicated_month_of_transactions():
    tables = dict(_tables())
    transactions = tables[TABLE_TRANSACTIONS]
    august = transactions[
        pd.to_datetime(transactions["transaction_date"]).dt.to_period("M").astype(str).eq(SPIKE_MONTH)
    ]
    tables[TABLE_TRANSACTIONS] = pd.concat([transactions, august], ignore_index=True)

    row = _find(
        check_monthly_row_count_anomaly(tables), f"monthly_row_count_anomaly__{TABLE_TRANSACTIONS}"
    )

    assert row["status"] == STATUS_FAIL
    assert SPIKE_MONTH in row["observed_value"]


# ---------------------------------------------------------------------------
# 8. Null-rate drift
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_null_rate_drift_localises_the_missing_category_batch_to_july():
    row = _row(_report(), f"null_rate_drift__{TABLE_TRANSACTIONS}__merchant_category")

    assert row["status"] == STATUS_FAIL
    assert DEFECT_MONTH_MISSING_CATEGORY in row["observed_value"]
    assert row["affected_rows"] == EXPECTED_MISSING_MERCHANT_CATEGORY_ROWS


@pytest.mark.slow
def test_null_rate_drift_passes_for_fields_that_are_never_null():
    report = _report()
    drift_rows = report[
        report["check_name"].str.startswith("null_rate_drift__")
        & ~report["check_name"].str.endswith("merchant_category")
    ]

    assert not drift_rows.empty
    assert set(drift_rows["status"]) == {STATUS_PASS}


@pytest.mark.slow
def test_null_rate_drift_ignores_a_movement_below_the_practical_floor():
    # One null in a ~116k-row month is a 0.00001 move: statistically loud against
    # a zero-variance baseline, operationally irrelevant.
    tables = dict(_tables())
    transactions = tables[TABLE_TRANSACTIONS].copy()
    august = pd.to_datetime(transactions["transaction_date"]).dt.to_period("M").astype(str).eq(
        SPIKE_MONTH
    )
    transactions.loc[transactions.index[august][0], "channel"] = np.nan
    tables[TABLE_TRANSACTIONS] = transactions

    row = _find(
        check_null_rate_drift(tables), f"null_rate_drift__{TABLE_TRANSACTIONS}__channel"
    )

    assert row["status"] == STATUS_PASS


# ---------------------------------------------------------------------------
# 9. Category-distribution drift
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_category_drift_warns_on_merchant_category_and_blames_the_missing_bucket():
    row = _row(_report(), f"category_distribution_drift__{TABLE_TRANSACTIONS}__merchant_category")

    assert row["status"] == STATUS_WARN
    assert DEFECT_MONTH_MISSING_CATEGORY in row["observed_value"]
    assert "__missing__" in row["observed_value"]
    assert row["affected_rows"] == EXPECTED_MISSING_MERCHANT_CATEGORY_ROWS


@pytest.mark.slow
def test_category_drift_passes_for_a_stable_channel_mix():
    row = _row(_report(), f"category_distribution_drift__{TABLE_TRANSACTIONS}__channel")

    assert row["status"] == STATUS_PASS


@pytest.mark.slow
def test_category_drift_fails_when_a_channel_is_remapped_wholesale():
    tables = dict(_tables())
    transactions = tables[TABLE_TRANSACTIONS].copy()
    august = pd.to_datetime(transactions["transaction_date"]).dt.to_period("M").astype(str).eq(
        SPIKE_MONTH
    )
    transactions.loc[august & transactions["channel"].eq("card_present"), "channel"] = "mobile"
    tables[TABLE_TRANSACTIONS] = transactions

    row = _find(
        check_category_distribution_drift(tables),
        f"category_distribution_drift__{TABLE_TRANSACTIONS}__channel",
    )

    assert row["status"] == STATUS_FAIL
    assert SPIKE_MONTH in row["observed_value"]


def test_population_stability_index_is_zero_for_an_unchanged_distribution():
    distribution = pd.Series({"travel": 0.2, "grocery": 0.5, "gas": 0.3})

    psi, _ = population_stability_index(distribution, distribution)

    assert psi == 0.0 or abs(psi) < 1e-9


def test_population_stability_index_grows_with_the_size_of_the_shift():
    baseline = pd.Series({"travel": 0.2, "grocery": 0.5, "gas": 0.3})
    small, _ = population_stability_index(
        pd.Series({"travel": 0.25, "grocery": 0.45, "gas": 0.30}), baseline
    )
    large, _ = population_stability_index(
        pd.Series({"travel": 0.60, "grocery": 0.20, "gas": 0.20}), baseline
    )

    assert 0 < small < large


def test_population_stability_index_handles_a_brand_new_bucket():
    baseline = pd.Series({"travel": 0.5, "grocery": 0.5})
    actual = pd.Series({"travel": 0.4, "grocery": 0.4, "__missing__": 0.2})

    psi, contributions = population_stability_index(actual, baseline)

    assert np.isfinite(psi)
    assert contributions.idxmax() == "__missing__"


# ---------------------------------------------------------------------------
# 10. KPI anomaly for dispute_rate
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dispute_rate_anomaly_flags_august():
    row = _row(_report(), "kpi_anomaly__dispute_rate")

    assert row["status"] == STATUS_FAIL
    assert row["severity"] == "high"
    assert SPIKE_MONTH in row["observed_value"]
    assert row["check_type"] == "statistical"


@pytest.mark.slow
def test_dispute_rate_anomaly_reports_the_disputed_purchase_count_for_the_spike_month():
    tables = _tables()
    row = _row(_report(), "kpi_anomaly__dispute_rate")

    transactions = tables[TABLE_TRANSACTIONS]
    purchases = transactions[transactions["transaction_type"].eq("purchase")]
    month = pd.to_datetime(purchases["transaction_date"]).dt.to_period("M").astype(str)
    expected = int(purchases.loc[month.eq(SPIKE_MONTH), "is_disputed"].sum())

    assert row["affected_rows"] == expected


@pytest.mark.slow
def test_dispute_rate_anomaly_passes_when_the_spike_month_is_removed():
    tables = dict(_tables())
    transactions = tables[TABLE_TRANSACTIONS]
    month = pd.to_datetime(transactions["transaction_date"]).dt.to_period("M").astype(str)
    tables[TABLE_TRANSACTIONS] = transactions[~month.eq(SPIKE_MONTH)]

    row = _only(check_kpi_anomaly_dispute_rate(tables))

    assert row["status"] == STATUS_PASS


@pytest.mark.slow
def test_statistical_checks_need_enough_history_before_they_score_anything():
    tables = dict(_tables())
    transactions = tables[TABLE_TRANSACTIONS]
    month = pd.to_datetime(transactions["transaction_date"]).dt.to_period("M").astype(str)
    tables[TABLE_TRANSACTIONS] = transactions[month.isin(["2026-01", "2026-02"])]

    row = _only(check_kpi_anomaly_dispute_rate(tables))

    assert row["status"] == STATUS_PASS
    assert row["observed_value"] == "insufficient history to score"


# ---------------------------------------------------------------------------
# 11. Metric definition governance
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_metric_definitions_are_fully_documented():
    row = _row(_report(), "metric_definition_governance_completeness")

    assert row["status"] == STATUS_PASS
    assert row["check_type"] == "governance"
    assert row["affected_rows"] == 0


@pytest.mark.slow
def test_metric_governance_fails_when_a_metric_lacks_known_limitations():
    definitions = _tables()[TABLE_METRIC_DEFINITIONS].copy()
    definitions.loc[definitions.index[0], "known_limitations"] = ""

    row = _only(check_metric_definition_governance(definitions))

    assert row["status"] == STATUS_FAIL
    assert row["affected_rows"] == 1
    assert row["affected_columns"] == "known_limitations"
    assert str(definitions.loc[definitions.index[0], "metric_name"]) in row["observed_value"]


@pytest.mark.slow
def test_metric_governance_fails_when_an_owner_column_is_absent():
    definitions = _tables()[TABLE_METRIC_DEFINITIONS].drop(columns=["owner"])

    row = _only(check_metric_definition_governance(definitions))

    assert row["status"] == STATUS_FAIL
    assert "missing columns: owner" in row["observed_value"]


# ---------------------------------------------------------------------------
# Rolling-baseline helpers
# ---------------------------------------------------------------------------


def test_floored_zscore_stays_finite_when_the_baseline_never_varied():
    score = floored_zscore(value=0.012, mean=0.0, std=0.0, min_abs_delta=0.002, fail_z=3.0)

    assert np.isfinite(score)
    assert score >= 3.0


def test_floored_zscore_suppresses_a_tiny_move_against_a_flat_baseline():
    score = floored_zscore(value=0.0001, mean=0.0, std=0.0, min_abs_delta=0.002, fail_z=3.0)

    assert abs(score) < 3.0


def test_floored_zscore_matches_the_plain_zscore_when_the_std_dominates():
    score = floored_zscore(value=12.0, mean=10.0, std=1.0, min_abs_delta=0.0, fail_z=3.0)

    assert score == 2.0


def test_floored_zscore_is_zero_when_the_value_equals_a_flat_baseline():
    assert floored_zscore(value=5.0, mean=5.0, std=0.0, min_abs_delta=0.0, fail_z=3.0) == 0.0


@pytest.mark.slow
def test_thresholds_are_configurable_and_change_the_verdict():
    strict = Thresholds(warn_z=1.0, fail_z=1.5, kpi_min_abs_delta=0.0)
    lenient = Thresholds(warn_z=50.0, fail_z=100.0)

    strict_row = _only(check_kpi_anomaly_dispute_rate(_tables(), strict))
    lenient_row = _only(check_kpi_anomaly_dispute_rate(_tables(), lenient))

    assert strict_row["status"] == STATUS_FAIL
    assert lenient_row["status"] == STATUS_PASS


def test_default_thresholds_follow_the_conventional_psi_bands():
    assert DEFAULT_THRESHOLDS.psi_warn == 0.10
    assert DEFAULT_THRESHOLDS.psi_fail == 0.25
    assert DEFAULT_THRESHOLDS.warn_z < DEFAULT_THRESHOLDS.fail_z


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_summary_counts_reconcile_with_the_report():
    report = _report()
    summary = summarize_quality_checks(report)

    assert summary["total_checks"] == len(report)
    assert summary["passed"] + summary["warned"] + summary["failed"] == len(report)
    assert 0.0 <= summary["pass_rate"] <= 1.0


@pytest.mark.slow
def test_summary_lists_the_failing_and_warning_checks():
    report = _report()
    summary = summarize_quality_checks(report)

    assert set(summary["blocking_checks"]) == set(
        report.loc[report["status"].eq(STATUS_FAIL), "check_name"]
    )
    assert set(summary["review_checks"]) == set(
        report.loc[report["status"].eq(STATUS_WARN), "check_name"]
    )
    assert "duplicate_source_transaction_id" in summary["blocking_checks"]


def test_summary_of_an_all_clean_report_is_a_perfect_pass_rate():
    clean = pd.DataFrame(
        [
            {column: "" for column in RESULT_COLUMNS} | {"status": STATUS_PASS, "check_name": name}
            for name in ("a", "b")
        ]
    )

    summary = summarize_quality_checks(clean)

    assert summary["pass_rate"] == 1.0
    assert summary["blocking_checks"] == []


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
