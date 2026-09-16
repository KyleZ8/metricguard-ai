"""Tests for the MetricGuard AI action prioritisation layer.

Run directly when pytest is unavailable::

    python tests/test_action_engine.py

The action plan is deterministic decision support. These tests check that it
does not invent actions for calm months, separates data-quality remediation from
business investigation, and sizes scenario impact from the selected KPI frame.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_engine import (  # noqa: E402
    ACTION_COLUMNS,
    SCENARIO_COLUMNS,
    SCORECARD_COLUMNS,
    build_action_plan,
)
from driver_analysis import build_driver_report  # noqa: E402
from metric_engine import build_finance_metric_report, resolve_tables  # noqa: E402
from quality_checks import run_quality_checks  # noqa: E402
from text_theme_analysis import HashingEmbedder, build_text_theme_report  # noqa: E402


SPIKE_MONTH = "2026-08"
PRIOR_MONTH = "2026-07"


@lru_cache(maxsize=1)
def _tables() -> dict[str, pd.DataFrame]:
    return resolve_tables()


@lru_cache(maxsize=1)
def _quality() -> pd.DataFrame:
    return run_quality_checks(_tables())


def _selected_quality(*months: str) -> pd.DataFrame:
    quality = _quality()
    mask = quality["status"].eq("pass")
    for month in months:
        mask = mask | quality.apply(
            lambda row: any(month in str(row[column]) for column in ("observed_value", "expected_value", "explanation")),
            axis=1,
        )
    return quality[mask].reset_index(drop=True)


def _spike_report():
    metric = build_finance_metric_report("dispute_rate", _tables(), SPIKE_MONTH, PRIOR_MONTH)
    driver = build_driver_report(
        current_period=SPIKE_MONTH,
        previous_period=PRIOR_MONTH,
        tables=_tables(),
        include_decision_tree=False,
    )
    themes = build_text_theme_report(
        current_period=SPIKE_MONTH,
        previous_period=PRIOR_MONTH,
        tables=_tables(),
        embedder=HashingEmbedder(),
        include_clusters=False,
    )
    return build_action_plan(metric, _selected_quality(SPIKE_MONTH, PRIOR_MONTH), driver, themes)


def test_action_plan_has_the_documented_frames():
    report = _spike_report()

    assert tuple(report.recommendations.columns) == ACTION_COLUMNS
    assert tuple(report.scenario_analysis.columns) == SCENARIO_COLUMNS
    assert tuple(report.scorecard.columns) == SCORECARD_COLUMNS
    assert report.methodology


def test_august_prioritises_duplicate_ingestion_as_p1():
    actions = _spike_report().recommendations
    top = actions.iloc[0]

    assert top["priority"] == "P1"
    assert top["action_type"] == "Data quality control"
    assert top["target_area"] == "dispute platform replay"
    assert top["estimated_avoidable_events"] == 165.0
    assert top["confidence"] == "High"


def test_august_still_has_a_business_investigation_after_correction():
    actions = _spike_report().recommendations

    business = actions[actions["action_type"].eq("Business investigation")]
    assert not business.empty
    assert "travel" in business.iloc[0]["target_area"]
    assert business.iloc[0]["estimated_avoidable_events"] > 300
    assert business.iloc[0]["priority"] in {"P1", "P2"}


def test_kpi_anomaly_is_not_mislabelled_as_a_data_quality_fix():
    actions = _spike_report().recommendations

    quality_review = actions[actions["action_type"].eq("Data quality review")]
    assert not quality_review["recommendation"].str.contains("kpi_anomaly").any()


def test_interaction_action_targets_travel_mobile_for_the_spike():
    actions = _spike_report().recommendations

    case_review = actions[actions["action_type"].eq("Case review")]
    assert not case_review.empty
    assert case_review.iloc[0]["target_area"] == "travel x mobile"


def test_customer_text_action_uses_the_leading_theme_when_material():
    actions = _spike_report().recommendations

    customer = actions[actions["action_type"].eq("Customer experience")]
    assert not customer.empty
    assert customer.iloc[0]["target_area"] == "duplicate_looking_travel_charge"


def test_scenario_table_sizes_partial_improvements_from_the_top_action():
    report = _spike_report()
    scenarios = report.scenario_analysis

    assert scenarios["scenario"].tolist() == ["Conservative", "Base", "Stretch"]
    assert scenarios["target_area"].tolist() == ["merchant_category = travel"] * 3
    assert scenarios["avoidable_events"].tolist() == [78.6, 157.2, 235.9]
    assert scenarios["projected_current_value"].is_monotonic_decreasing


def test_scorecard_weights_sum_to_one():
    scorecard = _spike_report().scorecard

    np.testing.assert_allclose(scorecard["weight"].sum(), 1.0)
    assert scorecard["weighted_score"].between(0, 1).all()


def test_calm_month_without_quality_findings_is_monitor_only():
    metric = build_finance_metric_report("dispute_rate", _tables(), "2026-02", "2026-01")
    empty_quality = _quality().iloc[0:0].copy()
    report = build_action_plan(metric, empty_quality)

    assert report.decision_summary.startswith("Monitor only")
    assert len(report.recommendations) == 1
    assert report.recommendations.iloc[0]["action_type"] == "Monitoring"
    assert report.scenario_analysis.empty


def test_negative_month_does_not_receive_a_business_spike_action():
    metric = build_finance_metric_report("dispute_rate", _tables(), "2026-07", "2026-06")
    report = build_action_plan(metric, _selected_quality("2026-07", "2026-06"))

    assert not report.recommendations["action_type"].isin(["Business investigation", "Case review"]).any()
    assert "spike" in report.decision_summary.lower()


def test_non_dispute_kpi_gets_a_generic_action_plan():
    metric = build_finance_metric_report("payment_failure_rate", _tables(), "2026-08", "2026-07")
    report = build_action_plan(metric, _selected_quality("2026-08", "2026-07"))

    assert report.metric_name == "payment_failure_rate"
    assert not report.recommendations.empty
    assert set(report.scorecard["signal"]) == {
        "Corrected KPI movement",
        "Driver concentration",
        "Quality readiness",
        "Customer text support",
    }


def test_action_plan_is_deterministic():
    first = _spike_report()
    second = _spike_report()

    pd.testing.assert_frame_equal(first.recommendations, second.recommendations)
    pd.testing.assert_frame_equal(first.scenario_analysis, second.scenario_analysis)
    pd.testing.assert_frame_equal(first.scorecard, second.scorecard)


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
