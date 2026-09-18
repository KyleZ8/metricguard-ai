"""Tests for the MetricGuard AI Streamlit dashboard.

These run under pytest and are also directly runnable, matching the other test
modules in this project::

    python tests/test_streamlit_app.py

No Streamlit server is started. The app is written so its data layer
(``_build_dashboard_data``) is a plain function with no Streamlit dependency,
and the render functions are only reached through ``main()`` behind an
``if __name__ == "__main__"`` guard. That split is what makes the page testable
at all: importing the module runs no UI code, and the pipeline can be exercised
directly.

What is verified here is the contract between the app and ``src/``: that every
section the page renders has data behind it, that the app reads rather than
recomputes, that the grounding badge is surfaced, and that nothing writes to the
generated CSVs. Visual layout is checked by running the app, not by these tests.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "app"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit_app  # noqa: E402
from quality_checks import STATUS_FAIL, STATUS_PASS, STATUS_WARN  # noqa: E402

SPIKE_MONTH = "2026-08"
PRIOR_MONTH = "2026-07"

DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
CSV_FILES = (
    "accounts.csv",
    "account_monthly_snapshot.csv",
    "transactions.csv",
    "complaints.csv",
    "metric_definitions.csv",
)


@lru_cache(maxsize=1)
def _data() -> streamlit_app.DashboardData:
    """The full pipeline, built once and shared by every assertion."""
    return streamlit_app._build_dashboard_data()


def _raises(exception_type, callable_, *args, **kwargs):
    try:
        callable_(*args, **kwargs)
    except exception_type as error:
        return error
    raise AssertionError(f"expected {exception_type.__name__} but nothing was raised")


# ---------------------------------------------------------------------------
# Import and module contract
# ---------------------------------------------------------------------------


def test_the_app_module_imports_without_rendering():
    module = importlib.import_module("streamlit_app")

    assert module.APP_TITLE == "MetricGuard AI"
    assert callable(module.main)


def test_the_app_exposes_the_three_business_modes_and_the_kpi():
    assert streamlit_app.BUSINESS_MODES == ("Finance Risk", "Growth Funnel", "Product Health")
    assert streamlit_app.ACTIVE_BUSINESS_MODE in streamlit_app.BUSINESS_MODES
    assert streamlit_app.AVAILABLE_KPIS == (
        "dispute_rate",
        "fraud_claim_rate",
        "payment_failure_rate",
        "delinquency_rate_30dpd_balance",
        "net_charge_off_rate_proxy",
        "complaint_rate",
        "fee_complaint_share",
    )
    assert streamlit_app.KPI_LABELS["dispute_rate"] == "Dispute rate"


def test_every_render_function_the_page_needs_exists():
    for name in (
        "render_header",
        "render_metric_overview",
        "render_verdict",
        "render_trend",
        "render_data_quality",
        "render_drivers",
        "render_themes",
        "render_evidence",
        "render_action_plan",
        "render_explanation",
        "render_inactive_mode",
    ):
        assert callable(getattr(streamlit_app, name)), name


def test_raw_and_corrected_have_distinct_reserved_colours():
    assert streamlit_app.COLOR_RAW != streamlit_app.COLOR_CORRECTED
    # Status colours must never double as a data series colour.
    assert streamlit_app.COLOR_RAW not in streamlit_app.STATUS_COLORS.values()
    assert streamlit_app.COLOR_CORRECTED not in streamlit_app.STATUS_COLORS.values()


def test_every_status_has_a_colour_and_an_icon():
    for status in (STATUS_PASS, STATUS_WARN, STATUS_FAIL):
        assert status in streamlit_app.STATUS_COLORS
        assert status in streamlit_app.STATUS_ICONS


# ---------------------------------------------------------------------------
# The data layer feeding every section
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dashboard_data_carries_every_report_the_page_renders():
    data = _data()

    assert data.finance_metric_report is not None
    assert data.metric_report is not None
    assert data.driver_report is not None
    assert data.theme_report is not None
    assert data.action_report is not None
    assert data.evidence_packet is not None
    assert data.explanation is not None
    assert isinstance(data.quality_report, pd.DataFrame)
    assert isinstance(data.quality_summary, dict)


@pytest.mark.slow
def test_metric_section_has_the_numbers_the_overview_shows():
    impact = _data().finance_metric_report.remediation_impact.iloc[0]

    for column in (
        "raw_value",
        "corrected_value",
        "value_difference",
        "duplicate_numerator_removed",
        "raw_numerator",
        "corrected_numerator",
    ):
        assert column in impact.index, column
    assert impact["raw_value"] > impact["corrected_value"]


@pytest.mark.slow
def test_period_comparison_supplies_both_month_over_month_changes():
    comparison = _data().metric_report.period_comparison.set_index("variant")

    assert comparison.loc["raw", "percent_change"] > comparison.loc["corrected", "percent_change"]
    assert comparison.loc["corrected", "percent_change"] > 0


@pytest.mark.slow
def test_metric_report_keeps_the_full_monthly_history_available():
    trend = _data().finance_metric_report.monthly_trend

    assert len(trend) == 8
    assert SPIKE_MONTH in set(trend["month"])
    for column in ("raw_value", "corrected_value", "duplicate_numerator_removed"):
        assert column in trend.columns


@pytest.mark.slow
def test_selected_trend_frame_shows_only_the_chosen_periods():
    data = streamlit_app._build_dashboard_data(
        metric_name="dispute_rate",
        current_period="2026-07",
        previous_period="2026-06",
    )
    trend = streamlit_app.selected_trend_frame(data)

    assert trend["month"].tolist() == ["2026-06", "2026-07"]
    assert "2026-08" not in set(trend["month"])


@pytest.mark.slow
def test_selected_trend_frame_uses_rolling_period_labels_for_quarters():
    data = streamlit_app._build_dashboard_data(metric_name="dispute_rate", period_grain="quarterly")
    trend = streamlit_app.selected_trend_frame(data)

    assert trend["month"].tolist() == ["2026-05 to 2026-07", "2026-06 to 2026-08"]


@pytest.mark.slow
def test_quarterly_dashboard_data_uses_multi_month_windows():
    data = streamlit_app._build_dashboard_data(metric_name="dispute_rate", period_grain="quarterly")

    assert data.finance_metric_report.period_grain == "quarterly"
    assert data.finance_metric_report.current_period == "2026-06 to 2026-08"
    assert data.finance_metric_report.current_months == ("2026-06", "2026-07", "2026-08")
    assert len(data.finance_metric_report.monthly_trend) == 6
    assert data.explanation is None, (
        "grounded explanation is monthly-only until multi-month packets exist"
    )


@pytest.mark.slow
def test_semiannual_dashboard_data_uses_multi_month_windows():
    data = streamlit_app._build_dashboard_data(
        metric_name="payment_failure_rate", period_grain="semiannual"
    )

    assert data.finance_metric_report.period_grain == "semiannual"
    assert data.finance_metric_report.current_period == "2026-03 to 2026-08"
    assert len(data.finance_metric_report.current_months) == 6
    assert len(data.finance_metric_report.monthly_trend) == 3


@pytest.mark.slow
def test_selected_quality_report_changes_with_the_selected_period():
    june = streamlit_app._build_dashboard_data(
        metric_name="dispute_rate",
        current_period="2026-06",
        previous_period="2026-05",
    )
    august = streamlit_app._build_dashboard_data(
        metric_name="dispute_rate",
        current_period=SPIKE_MONTH,
        previous_period=PRIOR_MONTH,
    )

    june_open = streamlit_app.selected_quality_report(june)
    june_open = june_open[june_open["status"].isin([STATUS_FAIL, STATUS_WARN])]
    august_open = streamlit_app.selected_quality_report(august)
    august_open = august_open[august_open["status"].isin([STATUS_FAIL, STATUS_WARN])]

    assert june_open.empty
    assert "kpi_anomaly__dispute_rate" in set(august_open["check_name"])
    assert "null_rate_drift__transactions__merchant_category" in set(august_open["check_name"])


@pytest.mark.slow
def test_selected_quality_report_for_july_has_one_fail_and_one_warning():
    july = streamlit_app._build_dashboard_data(
        metric_name="dispute_rate",
        current_period="2026-07",
        previous_period="2026-06",
    )
    summary = streamlit_app.summarize_quality_checks(streamlit_app.selected_quality_report(july))
    status, label = streamlit_app._quality_label(summary)

    assert status == STATUS_FAIL
    assert summary["failed"] == 1
    assert summary["warned"] == 1
    assert label == "1 fail / 1 warn"


@pytest.mark.slow
def test_selected_month_explanation_uses_selected_quality_context():
    february = streamlit_app._build_dashboard_data(
        metric_name="dispute_rate",
        current_period="2026-02",
        previous_period="2026-01",
    )

    text = february.explanation.narrative_text()
    assert "All 15 relevant automated data-quality checks passed" in text
    assert "kpi_anomaly__dispute_rate" not in text
    assert "Confirm with the source-system owner" not in text
    assert "Reissue the 2026-02" not in text


@pytest.mark.slow
def test_negative_month_explanation_is_valid_and_not_an_escalation():
    july = streamlit_app._build_dashboard_data(
        metric_name="dispute_rate",
        current_period="2026-07",
        previous_period="2026-06",
    )

    text = july.explanation.narrative_text()
    assert july.explanation.validation.is_valid, july.explanation.validation.notes
    assert "fell 5.24%" in july.explanation.headline
    assert "Do not escalate 2026-07 as a spike" in text
    assert "replay-batch issue" not in text


@pytest.mark.slow
def test_selected_quality_report_changes_with_the_selected_kpi():
    dispute = streamlit_app._build_dashboard_data(metric_name="dispute_rate")
    delinquency = streamlit_app._build_dashboard_data(metric_name="delinquency_rate_30dpd_balance")

    dispute_tables = set(streamlit_app.selected_quality_report(dispute)["table_name"])
    delinquency_tables = set(streamlit_app.selected_quality_report(delinquency)["table_name"])

    assert "transactions" in dispute_tables
    assert "account_monthly_snapshot" in delinquency_tables
    assert "transactions" not in delinquency_tables


@pytest.mark.slow
def test_non_dispute_kpis_build_generic_dashboard_data_without_dispute_only_reports():
    for metric_name in streamlit_app.AVAILABLE_KPIS:
        data = streamlit_app._build_dashboard_data(
            metric_name=metric_name,
            current_period=SPIKE_MONTH,
            previous_period=PRIOR_MONTH,
        )

        assert data.finance_metric_report.metric_name == metric_name
        assert not data.finance_metric_report.segment_drivers.empty
        assert data.action_report is not None
        assert not data.action_report.recommendations.empty
        if metric_name == "dispute_rate":
            assert data.metric_report is not None
            assert data.driver_report is not None
            assert data.theme_report is not None
            assert data.explanation is not None
        else:
            assert data.metric_report is None
            assert data.driver_report is None
            assert data.theme_report is None
            assert data.explanation is None


@pytest.mark.slow
def test_quality_summary_counts_reconcile_with_the_report():
    data = _data()
    summary = data.quality_summary

    assert summary["total_checks"] == len(data.quality_report)
    assert summary["passed"] + summary["warned"] + summary["failed"] == summary["total_checks"]
    assert summary["failed"] > 0, "the demo depends on at least one failing check"


@pytest.mark.slow
def test_the_headline_quality_checks_the_panel_pins_are_present():
    report = _data().quality_report

    for check_name in streamlit_app.HEADLINE_CHECKS:
        rows = report[report["check_name"].eq(check_name)]
        assert len(rows) == 1, check_name
        assert rows.iloc[0]["status"] in {STATUS_FAIL, STATUS_WARN}


@pytest.mark.slow
def test_driver_section_has_both_tables_and_the_interaction():
    report = _data().driver_report

    assert not report.top_count_drivers.empty
    assert not report.top_rate_deterioration.empty
    assert not report.interaction_heatmap_data.empty
    assert report.top_count_drivers.iloc[0]["segment_value"] == "travel"


@pytest.mark.slow
def test_theme_section_has_summary_segments_and_examples():
    report = _data().theme_report

    assert not report.theme_summary.empty
    assert not report.segment_themes.empty
    assert not report.representative_complaints.empty
    assert report.backend_name, "the page displays which embedding backend ran"


@pytest.mark.slow
def test_evidence_table_has_the_columns_the_page_shows():
    examples = _data().theme_report.representative_complaints

    for column in ("theme_name", "rank", "similarity", "merchant_category", "channel", "narrative"):
        assert column in examples.columns, column
    assert examples["narrative"].str.len().min() > 0


@pytest.mark.slow
def test_the_travel_and_mobile_focus_rows_exist():
    segments = _data().theme_report.segment_themes
    focus = segments[segments["segment_value"].isin(["travel", "mobile"])]

    assert not focus.empty
    assert set(focus["segment_value"]) == {"travel", "mobile"}


@pytest.mark.slow
def test_action_plan_prioritises_recommendations_and_scenarios():
    report = _data().action_report

    assert report.decision_summary
    assert not report.recommendations.empty
    assert report.recommendations.iloc[0]["priority"] == "P1"
    assert report.recommendations.iloc[0]["target_area"] == "dispute platform replay"
    assert not report.scenario_analysis.empty
    assert not report.scorecard.empty


# ---------------------------------------------------------------------------
# Explanation and its grounding badge
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_dashboard_uses_the_deterministic_explanation_by_default():
    explanation = _data().explanation

    # The first dashboard version must never reach the network.
    assert explanation.generated_by == "deterministic-fallback"


@pytest.mark.slow
def test_the_explanation_exposes_a_validation_result_to_surface():
    validation = _data().explanation.validation

    assert validation.status in {STATUS_PASS, STATUS_FAIL}
    assert validation.is_valid, validation.notes
    assert len(validation.checked) > 0
    assert validation.unsupported == ()


@pytest.mark.slow
def test_the_explanation_has_every_field_the_panel_renders():
    explanation = _data().explanation

    for field_name in (
        "headline",
        "executive_summary",
        "what_changed",
        "data_quality_findings",
        "business_drivers",
        "customer_text_evidence",
        "recommended_next_steps",
        "limitations",
        "evidence_references",
        "generated_by",
    ):
        assert getattr(explanation, field_name), field_name


@pytest.mark.slow
def test_the_export_payload_is_non_empty_markdown():
    markdown = _data().explanation.to_markdown()

    assert markdown.startswith("# ")
    assert "## Recommended next steps" in markdown
    assert "Number check:" in markdown


def test_the_status_badge_never_relies_on_colour_alone():
    for status in (STATUS_PASS, STATUS_WARN, STATUS_FAIL):
        badge = streamlit_app.status_badge(status)
        assert streamlit_app.STATUS_ICONS[status] in badge
        assert status.upper() in badge


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_trend_chart_builds_with_both_series():
    data = _data()
    chart = streamlit_app.build_trend_chart(
        data.finance_metric_report.monthly_trend,
        current_period=data.finance_metric_report.current_period,
        previous_period=data.finance_metric_report.previous_period,
    )
    spec = chart.to_dict()

    assert spec is not None
    rendered = str(spec)
    assert "Raw (reported)" in rendered
    assert "Corrected" in rendered
    assert streamlit_app.COLOR_RAW in rendered
    assert streamlit_app.COLOR_CORRECTED in rendered


@pytest.mark.slow
def test_the_trend_chart_marks_the_selected_windows():
    data = streamlit_app._build_dashboard_data(metric_name="dispute_rate", period_grain="quarterly")
    chart = streamlit_app.build_trend_chart(
        streamlit_app.selected_trend_frame(data),
        current_period=data.finance_metric_report.current_period,
        previous_period=data.finance_metric_report.previous_period,
    )
    rendered = str(chart.to_dict())

    assert "2026-06 to 2026-08" in rendered
    assert "2026-05 to 2026-07" in rendered


@pytest.mark.slow
def test_the_heatmap_builds_with_a_diverging_scale_around_zero():
    chart = streamlit_app.build_interaction_heatmap(_data().driver_report.interaction_heatmap_data)
    spec = chart.to_dict()

    scale = spec["encoding"]["color"]["scale"]
    assert scale["domain"][1] == 0, "the diverging midpoint must sit at zero change"
    assert scale["range"] == [
        streamlit_app.COLOR_DIVERGE_LOW,
        streamlit_app.COLOR_DIVERGE_MID,
        streamlit_app.COLOR_DIVERGE_HIGH,
    ]


@pytest.mark.slow
def test_charts_paint_an_explicit_surface_so_the_palette_is_read_correctly():
    chart = streamlit_app.build_trend_chart(_data().metric_report.monthly_trend)
    spec = chart.to_dict()

    assert spec["config"]["background"] == streamlit_app.CHART_SURFACE
    assert spec["config"]["view"]["fill"] == streamlit_app.CHART_SURFACE


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def test_percent_helpers_render_readable_values():
    assert streamlit_app._percent(0.017679) == "1.77%"
    assert streamlit_app._signed_percent(0.327922) == "+32.8%"
    assert streamlit_app._signed_percent(-0.1) == "-10.0%"


def test_percent_helpers_survive_missing_values():
    assert streamlit_app._percent(None) == "n/a"
    assert streamlit_app._signed_percent(float("nan")) == "n/a"


# ---------------------------------------------------------------------------
# Rendering the real page
#
# Streamlit's AppTest executes the script headlessly, so these prove the page
# actually renders rather than only that its data layer works. The run is
# expensive -- it drives the whole pipeline -- so it is done once and shared.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _rendered_app():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=1800)
    app.run()
    return app


@pytest.mark.slow
def test_the_page_renders_without_raising():
    app = _rendered_app()

    assert not app.exception, [str(item.value) for item in app.exception]


@pytest.mark.slow
def test_the_rendered_page_shows_the_raw_and_corrected_rates():
    metrics = {metric.label: metric.value for metric in _rendered_app().metric}

    assert metrics["Reported value (raw)"] == "1.77%"
    assert metrics["Corrected value"] == "1.61%"
    assert metrics["Numerator removed"] == "165"


@pytest.mark.slow
def test_the_rendered_page_shows_both_month_over_month_deltas():
    deltas = {metric.label: metric.delta for metric in _rendered_app().metric}

    assert deltas["Reported value (raw)"] == "+32.8%"
    assert deltas["Corrected value"] == "+21.3%"


@pytest.mark.slow
def test_the_rendered_page_surfaces_the_data_quality_status():
    metrics = {metric.label: metric.value for metric in _rendered_app().metric}

    assert "fail" in str(metrics["Data quality"])


@pytest.mark.slow
def test_the_rendered_page_has_the_seven_investigation_tabs():
    app = _rendered_app()

    assert len(app.tabs) == 7


@pytest.mark.slow
def test_the_rendered_page_offers_the_mode_and_kpi_selectors():
    selections = {box.label: box.value for box in _rendered_app().selectbox}

    assert selections["Business mode"] == "Finance Risk"
    assert selections["KPI"] == "Dispute rate"
    assert selections["Analysis window"] == "Monthly"
    assert selections["Current period"] == SPIKE_MONTH
    assert selections["Comparison period"] == PRIOR_MONTH


@pytest.mark.slow
def test_the_rendered_page_offers_the_summary_download():
    # Download buttons have no dedicated AppTest accessor; reach them by name.
    labels = [element.label for element in _rendered_app().get("download_button")]

    assert any("Download manager summary" in label for label in labels)


@pytest.mark.slow
def test_an_unwired_business_mode_says_so_instead_of_showing_borrowed_numbers():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=1800)
    app.run()
    app.selectbox[0].select("Growth Funnel").run()

    assert not app.exception
    assert app.info, "no notice shown for an unwired mode"
    assert "not wired to data yet" in app.info[0].value
    # The credit-card metric row must not appear under a different label.
    assert not app.metric


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
def test_building_the_dashboard_does_not_modify_any_generated_csv():
    before = _csv_fingerprints()

    data = _data()
    data.explanation.to_markdown()
    streamlit_app.build_trend_chart(data.metric_report.monthly_trend)

    assert _csv_fingerprints() == before


@pytest.mark.slow
def test_the_dashboard_never_writes_to_the_data_directory():
    before = sorted(path.name for path in DATA_DIR.iterdir())

    _data()

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
