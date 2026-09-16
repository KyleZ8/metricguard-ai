from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metricguard_engine import (  # noqa: E402
    build_investigation_summary,
    data_quality_checks,
    deduplicate_transactions,
    load_synthetic_data,
    monthly_dispute_rate,
    segment_driver_analysis,
)


def test_deduplication_removes_replayed_source_events():
    data = load_synthetic_data()
    transactions = data["transactions"]
    deduped = deduplicate_transactions(transactions)

    assert len(deduped) < len(transactions)
    assert deduped["source_transaction_id"].duplicated().sum() == 0


def test_august_dispute_rate_remains_elevated_after_deduplication():
    data = load_synthetic_data()
    trend = monthly_dispute_rate(data["transactions"], deduped=True).set_index("month")

    july = trend.loc["2026-07", "metric_value"]
    august = trend.loc["2026-08", "metric_value"]

    assert august > july * 1.15


def test_raw_august_dispute_rate_is_higher_than_deduped_rate():
    data = load_synthetic_data()
    raw = monthly_dispute_rate(data["transactions"], deduped=False).set_index("month")
    deduped = monthly_dispute_rate(data["transactions"], deduped=True).set_index("month")

    assert raw.loc["2026-08", "metric_value"] > deduped.loc["2026-08", "metric_value"]


def test_data_quality_checks_detect_planted_issues():
    data = load_synthetic_data()
    checks = data_quality_checks(data["transactions"], data["complaints"], data["snapshots"])
    statuses = dict(zip(checks["check_name"], checks["status"]))

    assert statuses["duplicate_source_transactions"] == "fail"
    assert statuses["missing_merchant_category"] == "fail"


def test_driver_analysis_returns_capital_one_relevant_segments():
    data = load_synthetic_data()
    drivers = segment_driver_analysis(data["transactions"], data["accounts"])

    assert not drivers.empty
    assert {"merchant_category", "channel", "fico_band", "customer_segment"}.issubset(
        set(drivers["segment_name"])
    )


def test_build_investigation_summary_has_expected_outputs():
    summary = build_investigation_summary()

    assert summary["deduped_result"].metric_name == "deduped_dispute_rate"
    assert isinstance(summary["quality_checks"], pd.DataFrame)
    assert isinstance(summary["drivers"], pd.DataFrame)
    assert isinstance(summary["complaint_themes"], pd.DataFrame)
