"""MetricGuard AI - dispute-rate investigation dashboard.

An internal analytics tool, not a landing page. It walks a fictional-card-issuer-style
analyst through one question in the order the question is actually answered:

    is the KPI alert real? -> can the data be trusted? -> what moved? ->
    what do customers say? -> what do I tell my manager?

This module renders only. Every number on screen is computed by ``src/`` and is
read here without modification, which is what keeps the dashboard consistent with
the test suite behind those modules.

Two visual conventions carry most of the meaning, and both are deliberate:

* **Raw is orange, corrected is blue, everywhere.** The reported figure is the
  one that triggered the alert and overstates the movement; the corrected figure
  is the one to act on. Holding the hues fixed across the metric row, the trend
  chart and the explanation means a reader learns the pairing once.
* **Status colour is reserved for pass / warn / fail** and never used for a data
  series, so a red mark always means "a check failed" and never "this is the
  second series".

Run it with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_engine import build_action_plan  # noqa: E402
from config import DATASET_SIZE  # noqa: E402
from driver_analysis import build_driver_report  # noqa: E402
from explanation_engine import build_evidence_packet, explain  # noqa: E402
from metric_engine import (  # noqa: E402
    METRIC_NAME as DISPUTE_RATE_METRIC,
)
from metric_engine import (
    PERIOD_GRAIN_MONTHLY,
    aggregate_metric_trend,
    available_finance_kpis,
    build_finance_metric_report,
    build_metric_report,
    generic_monthly_trend_table,
    metric_spec,
)
from quality_checks import (  # noqa: E402
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARN,
    TABLE_METRIC_DEFINITIONS,
    load_tables,
    run_quality_checks,
    summarize_quality_checks,
)
from text_theme_analysis import build_text_theme_report  # noqa: E402

APP_TITLE = "MetricGuard AI"
APP_SUBTITLE = "KPI investigation workspace - credit-card portfolio"

BUSINESS_MODES = ("Finance Risk", "Growth Funnel", "Product Health")
ACTIVE_BUSINESS_MODE = "Finance Risk"
AVAILABLE_KPIS = available_finance_kpis()
KPI_LABELS = {name: metric_spec(name).display_name for name in AVAILABLE_KPIS}
KPI_NAME_BY_LABEL = {label: name for name, label in KPI_LABELS.items()}
PERIOD_GRAIN_LABELS = {
    "Monthly": "monthly",
    "Rolling quarter": "quarterly",
    "Rolling 6 months": "semiannual",
}
PERIOD_GRAIN_NAME_BY_VALUE = {value: label for label, value in PERIOD_GRAIN_LABELS.items()}

SPIKE_MONTH = "2026-08"

# Palette. Categorical slots 1 and 2 for the two metric variants; the status four
# for check outcomes. Charts paint their own light surface so they stay legible
# whichever theme Streamlit is running.
COLOR_RAW = "#eb6834"  # categorical slot 2 - the reported, inflated figure
COLOR_CORRECTED = "#2a78d6"  # categorical slot 1 - the figure to act on
COLOR_DIVERGE_LOW = "#2a78d6"
COLOR_DIVERGE_MID = "#f0efec"
COLOR_DIVERGE_HIGH = "#e34948"

STATUS_COLORS = {
    STATUS_PASS: "#0ca30c",
    STATUS_WARN: "#fab219",
    STATUS_FAIL: "#d03b3b",
}
STATUS_ICONS = {STATUS_PASS: "●", STATUS_WARN: "▲", STATUS_FAIL: "■"}

CHART_SURFACE = "#fcfcfb"
CHART_INK = "#0b0b0b"
CHART_INK_MUTED = "#52514e"
CHART_GRID = "#e6e5e1"

# The two findings an analyst must see before believing any driver claim.
HEADLINE_CHECKS = (
    "duplicate_source_transaction_id",
    "null_rate_drift__transactions__merchant_category",
)


CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
  [data-testid="stMetricValue"] {{ font-size: 1.55rem; }}
  [data-testid="stMetricLabel"] {{ font-size: 0.78rem; letter-spacing: .01em; }}

  .mg-head {{ display:flex; align-items:baseline; gap:.75rem; flex-wrap:wrap;
             border-bottom:1px solid {CHART_GRID}; padding-bottom:.6rem; margin-bottom:1rem; }}
  .mg-head h1 {{ font-size:1.35rem; font-weight:650; margin:0; letter-spacing:-.01em; }}
  .mg-head .sub {{ font-size:.85rem; opacity:.68; }}

  .mg-verdict {{ border-left:3px solid {COLOR_CORRECTED}; padding:.65rem .9rem;
                 background:rgba(42,120,214,.06); border-radius:3px; margin:.2rem 0 1rem; }}
  .mg-verdict .h {{ font-size:1rem; font-weight:600; line-height:1.4; }}
  .mg-verdict .s {{ font-size:.86rem; opacity:.8; margin-top:.3rem; line-height:1.5; }}

  .mg-badge {{ display:inline-block; padding:.12rem .5rem; border-radius:3px;
               font-size:.74rem; font-weight:600; letter-spacing:.02em; }}
  .mg-note {{ font-size:.8rem; opacity:.65; margin-top:.35rem; }}
  .mg-legend {{ font-size:.8rem; opacity:.75; margin:.1rem 0 .6rem; }}
  .mg-swatch {{ display:inline-block; width:.62rem; height:.62rem; border-radius:2px;
                margin-right:.3rem; vertical-align:baseline; }}
  blockquote {{ border-left:2px solid {CHART_GRID}; margin:.4rem 0; padding-left:.8rem;
                font-size:.87rem; opacity:.9; }}
</style>
"""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DashboardData:
    """Every computed report the page renders, built once per session."""

    tables: dict[str, pd.DataFrame]
    finance_metric_report: object
    metric_report: object | None
    quality_report: pd.DataFrame
    quality_summary: dict
    driver_report: object | None
    theme_report: object | None
    action_report: object
    evidence_packet: object | None
    explanation: object | None


def _build_dashboard_data(
    metric_name: str = DISPUTE_RATE_METRIC,
    current_period: str | None = None,
    previous_period: str | None = None,
    period_grain: str = PERIOD_GRAIN_MONTHLY,
) -> DashboardData:
    """Run the full analysis pipeline.

    Kept free of Streamlit caching so tests can call it directly. The cached
    entry point below is what the page uses.
    """
    tables = load_tables()

    finance_metric_report = build_finance_metric_report(
        metric_name=metric_name,
        tables=tables,
        current_period=current_period,
        previous_period=previous_period,
        period_grain=period_grain,
    )
    quality_report = run_quality_checks(tables)
    metric_report = None
    driver_report = None
    theme_report = None
    action_report = None
    evidence_packet = None
    explanation = None

    if metric_name == DISPUTE_RATE_METRIC and period_grain == PERIOD_GRAIN_MONTHLY:
        metric_report = build_metric_report(
            tables,
            current_period=finance_metric_report.current_period,
            previous_period=finance_metric_report.previous_period,
        )
        driver_report = build_driver_report(
            tables=tables,
            current_period=finance_metric_report.current_period,
            previous_period=finance_metric_report.previous_period,
        )
        theme_report = build_text_theme_report(
            tables=tables,
            current_period=finance_metric_report.current_period,
            previous_period=finance_metric_report.previous_period,
        )

        evidence_packet = build_evidence_packet(
            metric_report=metric_report,
            quality_report=filter_quality_report_for_selection(
                quality_report,
                finance_metric_report,
            ),
            driver_report=driver_report,
            theme_report=theme_report,
        )
        # use_llm=False by default: the dashboard never reaches the network, so
        # the demo behaves identically with or without an API key.
        explanation = explain(evidence_packet, use_llm=False)

    action_report = build_action_plan(
        finance_metric_report=finance_metric_report,
        quality_report=filter_quality_report_for_selection(
            quality_report,
            finance_metric_report,
        ),
        driver_report=driver_report,
        theme_report=theme_report,
    )

    return DashboardData(
        tables=tables,
        finance_metric_report=finance_metric_report,
        metric_report=metric_report,
        quality_report=quality_report,
        quality_summary=summarize_quality_checks(quality_report),
        driver_report=driver_report,
        theme_report=theme_report,
        action_report=action_report,
        evidence_packet=evidence_packet,
        explanation=explanation,
    )


@st.cache_resource(show_spinner="Running the MetricGuard analysis pipeline...")
def load_dashboard_data(
    metric_name: str = DISPUTE_RATE_METRIC,
    current_period: str | None = None,
    previous_period: str | None = None,
    period_grain: str = PERIOD_GRAIN_MONTHLY,
) -> DashboardData:
    """Session-cached pipeline.

    ``cache_resource`` rather than ``cache_data``: the payload holds live report
    objects that should be shared, not copied per caller, and the sentence
    embeddings behind the theme report are far too expensive to recompute on
    every widget interaction.
    """
    return _build_dashboard_data(metric_name, current_period, previous_period, period_grain)


@st.cache_resource(show_spinner="Loading portfolio tables...")
def load_dashboard_tables() -> dict[str, pd.DataFrame]:
    """Load the synthetic portfolio once so controls can react quickly."""
    return load_tables()


def available_periods_for_kpi(
    metric_name: str,
    tables: dict[str, pd.DataFrame],
    period_grain: str = PERIOD_GRAIN_MONTHLY,
) -> tuple[str, ...]:
    """Reporting windows where the selected KPI has a computable denominator."""
    trend = aggregate_metric_trend(generic_monthly_trend_table(tables, metric_name), period_grain)
    valid = trend[trend["corrected_denominator"].gt(0)]
    return tuple(valid["month"].astype(str).tolist())


# ---------------------------------------------------------------------------
# Small render helpers
# ---------------------------------------------------------------------------


def _percent(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def _signed_percent(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:+.{digits}f}%"


def status_badge(status: str, label: str | None = None) -> str:
    """A status pill. Icon plus text, never colour alone."""
    color = STATUS_COLORS.get(status, CHART_INK_MUTED)
    icon = STATUS_ICONS.get(status, "●")
    return (
        f'<span class="mg-badge" style="background:{color}1f;color:{color};">'
        f"{icon} {label or status.upper()}</span>"
    )


def _chart_theme(chart: alt.Chart) -> alt.Chart:
    """Paint an explicit light surface and recessive hairline chrome.

    Fixing the surface means the validated light palette is always read against
    the surface it was validated on, whatever theme the app is running under.
    """
    return (
        chart.configure(background=CHART_SURFACE)
        .configure_view(fill=CHART_SURFACE, stroke=None)
        .configure_axis(
            grid=True,
            gridColor=CHART_GRID,
            gridWidth=1,
            domainColor=CHART_GRID,
            tickColor=CHART_GRID,
            labelColor=CHART_INK_MUTED,
            titleColor=CHART_INK_MUTED,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=CHART_INK,
            titleColor=CHART_INK_MUTED,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_text(color=CHART_INK)
    )


# ---------------------------------------------------------------------------
# 1. Header
# ---------------------------------------------------------------------------


def render_header(tables: dict[str, pd.DataFrame]) -> tuple[str, str, str, str | None, str | None]:
    st.markdown(
        f'<div class="mg-head"><h1>{APP_TITLE}</h1><span class="sub">{APP_SUBTITLE}</span></div>',
        unsafe_allow_html=True,
    )
    dataset_label = "sample (demo)" if DATASET_SIZE == "sample" else "full"
    st.caption(f"Dataset: {dataset_label} — set with the METRICGUARD_DATA environment variable")

    controls = st.columns([1.05, 1.15, 1.15, 1.55, 1.55])
    business_mode = controls[0].selectbox("Business mode", BUSINESS_MODES, index=0)
    kpi_label = controls[1].selectbox(
        "KPI",
        [KPI_LABELS[name] for name in AVAILABLE_KPIS],
        index=0,
    )
    kpi = KPI_NAME_BY_LABEL[kpi_label]
    grain_label = controls[2].selectbox("Analysis window", list(PERIOD_GRAIN_LABELS), index=0)
    period_grain = PERIOD_GRAIN_LABELS[grain_label]

    current_period: str | None = None
    previous_period: str | None = None
    if business_mode == ACTIVE_BUSINESS_MODE:
        periods = available_periods_for_kpi(kpi, tables, period_grain)
        current_options = periods[1:] if len(periods) > 1 else periods
        current_period = controls[3].selectbox(
            "Current period",
            current_options,
            index=len(current_options) - 1 if current_options else 0,
        )
        previous_options = tuple(period for period in periods if period < current_period)
        previous_period = controls[4].selectbox(
            "Comparison period",
            previous_options,
            index=len(previous_options) - 1 if previous_options else 0,
        )
    else:
        controls[3].markdown("**Current period**  \nNo data wired")
        controls[4].markdown("**Comparison period**  \nNo data wired")

    return business_mode, kpi, period_grain, current_period, previous_period


def _format_metric_label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _format_count(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.0f}"


def _quality_label(summary: dict) -> tuple[str, str]:
    failed = int(summary.get("failed", 0))
    warned = int(summary.get("warned", 0))
    if failed and warned:
        return STATUS_FAIL, f"{failed} fail / {warned} warn"
    if failed:
        return STATUS_FAIL, f"{failed} failed"
    if warned:
        return STATUS_WARN, f"{warned} warning" if warned == 1 else f"{warned} warnings"
    return STATUS_PASS, "all passed"


def _selected_kpi_title(data: DashboardData) -> str:
    report = data.finance_metric_report
    return f"{report.display_name} ({report.current_period} vs {report.previous_period})"


def generic_verdict_text(data: DashboardData) -> tuple[str, str]:
    report = data.finance_metric_report
    comparison = report.period_comparison.set_index("variant")
    impact = report.remediation_impact.iloc[0]
    corrected_change = comparison.loc["corrected", "percent_change"]
    raw_change = comparison.loc["raw", "percent_change"]
    direction = "rose" if comparison.loc["corrected", "absolute_change"] >= 0 else "fell"
    duplicate_removed = impact["duplicate_numerator_removed"]

    headline = (
        f"{report.display_name} {direction} {_signed_percent(corrected_change).replace('+', '')} "
        f"in {report.current_period}"
    )
    if duplicate_removed:
        summary = (
            f"Reported movement was {_signed_percent(raw_change)}, but the corrected movement is "
            f"{_signed_percent(corrected_change)} after removing {_format_count(duplicate_removed)} "
            f"duplicated numerator event(s). The generic driver table shows which segments explain "
            "the corrected movement."
        )
    else:
        summary = (
            f"Raw and corrected values are the same for this KPI because the known replay defect "
            f"does not affect its source table. The current value is {_percent(impact['corrected_value'])}, "
            f"compared with {_percent(comparison.loc['corrected', 'previous_value'])} in "
            f"{report.previous_period}."
        )
    return headline, summary


def _source_tables_for_report(report: object) -> set[str]:
    source_tables = str(report.metric_definition.get("source_tables", ""))
    tables = {item.strip() for item in source_tables.split(",") if item.strip()}
    tables.add(TABLE_METRIC_DEFINITIONS)
    return tables


def _metric_source_tables(data: DashboardData) -> set[str]:
    return _source_tables_for_report(data.finance_metric_report)


def _row_mentions_any_month(row: pd.Series, months: tuple[str, ...]) -> bool:
    haystack = " ".join(
        str(row.get(column, ""))
        for column in (
            "check_name",
            "observed_value",
            "expected_value",
            "explanation",
            "recommended_action",
        )
    )
    compact_haystack = haystack.replace("-", "")
    return any(month in haystack or month.replace("-", "") in compact_haystack for month in months)


def filter_quality_report_for_selection(
    quality_report: pd.DataFrame,
    finance_metric_report: object,
) -> pd.DataFrame:
    """Quality findings relevant to one KPI and reporting window."""
    tables = _source_tables_for_report(finance_metric_report)
    relevant = quality_report[quality_report["table_name"].isin(tables)].copy()
    months = finance_metric_report.current_months + finance_metric_report.previous_months
    if not months:
        return relevant

    period_mask = relevant.apply(lambda row: _row_mentions_any_month(row, months), axis=1)
    always_show = relevant["check_type"].isin(["rule", "contract"]) | relevant["status"].eq(
        STATUS_PASS
    )
    return relevant[period_mask | always_show].reset_index(drop=True)


def selected_quality_report(data: DashboardData) -> pd.DataFrame:
    """Quality findings relevant to the selected KPI and reporting window."""
    return filter_quality_report_for_selection(data.quality_report, data.finance_metric_report)


def render_inactive_mode(business_mode: str) -> None:
    """Say plainly that a mode has no data rather than showing borrowed numbers."""
    st.info(
        f"**{business_mode} mode is not wired to data yet.** The investigation engine is "
        "KPI-agnostic - the same metric, quality and driver pipeline runs on any "
        "rate metric with a numerator, a denominator and segment columns. The finance "
        "dataset now supports the portfolio KPIs; the growth and product modes are shown "
        "empty rather than filled with the "
        "credit-card numbers under a different label.",
        icon=":material/info:",
    )


# ---------------------------------------------------------------------------
# 2. Metric health overview
# ---------------------------------------------------------------------------


def render_metric_overview(data: DashboardData) -> None:
    report = data.finance_metric_report
    spec = metric_spec(report.metric_name)
    impact = report.remediation_impact.iloc[0]
    comparison = report.period_comparison.set_index("variant")
    selected_quality = selected_quality_report(data)
    summary = summarize_quality_checks(selected_quality)
    quality_status, quality_label = _quality_label(summary)

    columns = st.columns(6)
    # Only two cards carry a delta line, so pin a shared height to keep the row even.
    card = {"border": True, "height": 132}
    columns[0].metric(
        "Reported value (raw)",
        _percent(impact["raw_value"]),
        _signed_percent(comparison.loc["raw", "percent_change"]),
        delta_color="inverse",
        **card,
        help=f"As reported before remediation. Numerator: {spec.numerator_label}.",
    )
    columns[1].metric(
        "Corrected value",
        _percent(impact["corrected_value"]),
        _signed_percent(comparison.loc["corrected", "percent_change"]),
        delta_color="inverse",
        **card,
        help=f"After known data-quality remediation. Denominator: {spec.denominator_label}.",
    )
    columns[2].metric(
        "Overstatement",
        _percent(impact["value_difference"]),
        **card,
        help="Percentage points of the reported value contributed by duplicate rows.",
    )
    columns[3].metric(
        "Numerator removed",
        _format_count(impact["duplicate_numerator_removed"]),
        **card,
        help="Numerator events removed by the remediation step.",
    )
    columns[4].metric(
        f"{_format_metric_label(spec.numerator_label)} (raw / corrected)",
        f"{_format_count(impact['raw_numerator'])} / {_format_count(impact['corrected_numerator'])}",
        **card,
    )
    columns[5].metric(
        "Data quality",
        quality_label,
        **card,
        help=f"{summary['passed']} passed, {summary['warned']} warned, {summary['failed']} failed.",
    )

    st.markdown(
        f'<div class="mg-legend">'
        f'<span class="mg-swatch" style="background:{COLOR_RAW}"></span>Raw / reported'
        f'&nbsp;&nbsp;&nbsp;<span class="mg-swatch" style="background:{COLOR_CORRECTED}"></span>'
        f"Corrected&nbsp;&nbsp;&nbsp;{status_badge(quality_status, quality_label)}</div>",
        unsafe_allow_html=True,
    )


def render_verdict(data: DashboardData) -> None:
    """The answer, before the evidence."""
    if data.explanation is None:
        headline, summary = generic_verdict_text(data)
        st.markdown(
            f'<div class="mg-verdict"><div class="h">{headline}</div>'
            f'<div class="s">{summary}</div>'
            f'<div class="mg-note">{status_badge(STATUS_PASS, "computed from selected KPI")} '
            f"&nbsp; Deterministic summary, no LLM call</div></div>",
            unsafe_allow_html=True,
        )
        return

    explanation = data.explanation
    validation = explanation.validation
    badge = status_badge(
        STATUS_PASS if validation.is_valid else STATUS_FAIL,
        "numbers verified" if validation.is_valid else "unverified numbers",
    )
    st.markdown(
        f'<div class="mg-verdict"><div class="h">{explanation.headline}</div>'
        f'<div class="s">{explanation.executive_summary}</div>'
        f'<div class="mg-note">{badge} &nbsp; {len(validation.checked)} numbers checked '
        f"against the evidence packet &nbsp;·&nbsp; {explanation.generated_by}</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 3. Trend
# ---------------------------------------------------------------------------


def _standard_trend_frame(trend: pd.DataFrame) -> pd.DataFrame:
    """Accept the generic KPI frame or the legacy dispute-rate frame."""
    if "raw_value" in trend.columns:
        return trend.copy()
    return pd.DataFrame(
        {
            "month": trend["month"],
            "metric_name": DISPUTE_RATE_METRIC,
            "raw_numerator": trend["disputed_purchases_raw"],
            "raw_denominator": trend["purchase_transactions_raw"],
            "raw_value": trend["dispute_rate_raw"],
            "corrected_numerator": trend["disputed_purchases_corrected"],
            "corrected_denominator": trend["purchase_transactions_corrected"],
            "corrected_value": trend["dispute_rate_corrected"],
            "duplicate_numerator_removed": trend["duplicate_disputed_rows_removed"],
            "duplicate_denominator_removed": trend["duplicate_purchase_rows_removed"],
            "value_difference": trend["dispute_rate_difference"],
        }
    )


def build_trend_chart(
    trend: pd.DataFrame,
    y_title: str = "metric value",
    current_period: str | None = None,
    previous_period: str | None = None,
) -> alt.Chart:
    """Raw and corrected monthly KPI value on one axis."""
    trend = _standard_trend_frame(trend)
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "month": trend["month"],
                    "variant": "Raw (reported)",
                    "value": trend["raw_value"],
                    "numerator": trend["raw_numerator"],
                    "denominator": trend["raw_denominator"],
                }
            ),
            pd.DataFrame(
                {
                    "month": trend["month"],
                    "variant": "Corrected",
                    "value": trend["corrected_value"],
                    "numerator": trend["corrected_numerator"],
                    "denominator": trend["corrected_denominator"],
                }
            ),
        ],
        ignore_index=True,
    )

    color = alt.Color(
        "variant:N",
        scale=alt.Scale(domain=["Raw (reported)", "Corrected"], range=[COLOR_RAW, COLOR_CORRECTED]),
        legend=alt.Legend(title=None, orient="top", direction="horizontal"),
    )

    marker_rows = [
        {"month": previous_period, "selected_period": "Comparison"}
        if previous_period in set(trend["month"])
        else None,
        {"month": current_period, "selected_period": "Current"}
        if current_period in set(trend["month"])
        else None,
    ]
    markers = pd.DataFrame([row for row in marker_rows if row is not None])
    highlight = (
        alt.Chart(markers)
        .mark_rule(color=CHART_INK_MUTED, strokeWidth=1.5, opacity=0.55)
        .encode(x=alt.X("month:N"))
        if not markers.empty
        else alt.Chart(pd.DataFrame({"month": []})).mark_rule().encode(x=alt.X("month:N"))
    )

    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=44, filled=True))
        .encode(
            x=alt.X("month:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "value:Q",
                title=y_title,
                axis=alt.Axis(format=".2%"),
                scale=alt.Scale(zero=False, nice=True),
            ),
            color=color,
            tooltip=[
                alt.Tooltip("month:N", title="Month"),
                alt.Tooltip("variant:N", title="Series"),
                alt.Tooltip("value:Q", title="Value", format=".3%"),
                alt.Tooltip("numerator:Q", title="Numerator", format=","),
                alt.Tooltip("denominator:Q", title="Denominator", format=","),
            ],
        )
    )

    # Direct-label the endpoints only; the axis and tooltip carry the rest.
    endpoints = long[long["month"].eq(long["month"].max())]
    labels = (
        alt.Chart(endpoints)
        .mark_text(align="left", dx=8, fontSize=11, fontWeight=600)
        .encode(
            x=alt.X("month:N"),
            y=alt.Y("value:Q"),
            text=alt.Text("value:Q", format=".2%"),
            color=color,
        )
    )

    return _chart_theme((highlight + lines + labels).properties(height=290, padding={"right": 58}))


def selected_trend_frame(data: DashboardData) -> pd.DataFrame:
    """Trend rows that match the selected comparison/current windows."""
    report = data.finance_metric_report
    trend = _standard_trend_frame(report.monthly_trend)
    selected = trend[trend["month"].isin([report.previous_period, report.current_period])].copy()
    order = {report.previous_period: 0, report.current_period: 1}
    selected["_selected_order"] = selected["month"].map(order)
    return (
        selected.sort_values("_selected_order")
        .drop(columns="_selected_order")
        .reset_index(drop=True)
    )


def render_trend(data: DashboardData) -> None:
    report = data.finance_metric_report
    trend = selected_trend_frame(data)
    spec = metric_spec(report.metric_name)

    st.altair_chart(
        build_trend_chart(
            trend,
            spec.display_name.lower(),
            report.current_period,
            report.previous_period,
        ),
        width="stretch",
    )
    st.caption(
        "The orange series is the reported KPI and the blue series is the value after known "
        "data-quality remediation. This view shows only the selected comparison and current windows."
    )

    display = trend.rename(
        columns={
            "month": "Month",
            "raw_numerator": "Numerator (raw)",
            "raw_denominator": "Denominator (raw)",
            "raw_value": "Value (raw)",
            "corrected_numerator": "Numerator (corrected)",
            "corrected_denominator": "Denominator (corrected)",
            "corrected_value": "Value (corrected)",
            "duplicate_numerator_removed": "Numerator removed",
        }
    )
    st.dataframe(
        display[
            [
                "Month",
                "Numerator (raw)",
                "Denominator (raw)",
                "Value (raw)",
                "Numerator (corrected)",
                "Denominator (corrected)",
                "Value (corrected)",
                "Numerator removed",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Value (raw)": st.column_config.NumberColumn(format="percent"),
            "Value (corrected)": st.column_config.NumberColumn(format="percent"),
        },
    )


# ---------------------------------------------------------------------------
# 4. Data quality
# ---------------------------------------------------------------------------


def render_data_quality(data: DashboardData) -> None:
    report = selected_quality_report(data)
    summary = summarize_quality_checks(report)
    source_tables = ", ".join(sorted(_metric_source_tables(data)))
    selected_months = ", ".join(data.finance_metric_report.current_months)

    columns = st.columns(4)
    columns[0].metric("Relevant checks", summary["total_checks"], border=True)
    columns[1].metric("Failed", summary["failed"], border=True)
    columns[2].metric("Warnings", summary["warned"], border=True)
    columns[3].metric("Passed", summary["passed"], border=True)
    st.caption(
        f"Filtered to `{data.finance_metric_report.metric_name}` source tables ({source_tables}) "
        f"and the selected window ending in {selected_months}."
    )

    st.markdown("##### Findings that change how the metric should be read")
    highlighted = report[report["status"].isin([STATUS_FAIL, STATUS_WARN])]
    if highlighted.empty:
        st.success("No failing or warning checks are relevant to this KPI and selected window.")
    for check_name in HEADLINE_CHECKS:
        rows = report[report["check_name"].eq(check_name)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        st.markdown(
            f"{status_badge(row['status'])} &nbsp; **{row['check_name']}** &nbsp; "
            f"`{row['table_name']}` &nbsp;·&nbsp; {int(row['affected_rows']):,} rows affected",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="mg-note">{row["observed_value"]}<br/>'
            f"<em>{row['recommended_action']}</em></div>",
            unsafe_allow_html=True,
        )
        st.write("")

    open_checks = highlighted
    st.markdown("##### Relevant failing and warning checks")
    st.dataframe(
        open_checks[
            [
                "check_name",
                "check_type",
                "table_name",
                "status",
                "severity",
                "affected_rows",
                "observed_value",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"{summary['passed']} relevant checks passed and are not listed. "
        "Rule-based checks are contract violations; statistical checks compare a month "
        "against its own rolling baseline."
    )


# ---------------------------------------------------------------------------
# 5. Drivers
# ---------------------------------------------------------------------------


def build_interaction_heatmap(interaction: pd.DataFrame) -> alt.Chart:
    """Merchant category x channel rate movement.

    Diverging, because zero means "no change" and is the reference the reader
    needs: warm for deterioration, cool for improvement, neutral gray at zero.
    """
    frame = interaction.copy()
    limit = float(frame["rate_change"].abs().max() or 0.01)

    return _chart_theme(
        alt.Chart(frame)
        .mark_rect(stroke=CHART_SURFACE, strokeWidth=2)
        .encode(
            x=alt.X("segment_b_value:N", title="channel", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("segment_a_value:N", title="merchant category"),
            color=alt.Color(
                "rate_change:Q",
                title="rate change",
                scale=alt.Scale(
                    domain=[-limit, 0, limit],
                    range=[COLOR_DIVERGE_LOW, COLOR_DIVERGE_MID, COLOR_DIVERGE_HIGH],
                ),
                legend=alt.Legend(format=".1%", orient="right"),
            ),
            tooltip=[
                alt.Tooltip("segment_a_value:N", title="Merchant category"),
                alt.Tooltip("segment_b_value:N", title="Channel"),
                alt.Tooltip("previous_dispute_rate:Q", title="Previous rate", format=".2%"),
                alt.Tooltip("current_dispute_rate:Q", title="Current rate", format=".2%"),
                alt.Tooltip("rate_change:Q", title="Rate change", format=".2%"),
                alt.Tooltip("disputed_change:Q", title="Dispute change", format=","),
                alt.Tooltip("current_purchases:Q", title="Purchases", format=","),
                alt.Tooltip("min_denominator_flag:N", title="Small denominator"),
            ],
        )
        .properties(height=300)
    )


def render_drivers(data: DashboardData) -> None:
    if data.driver_report is None:
        report = data.finance_metric_report
        spec = metric_spec(report.metric_name)
        drivers = report.segment_drivers
        if drivers.empty:
            st.info("No segment driver rows are available for this KPI and period.")
            return

        top = drivers.head(15).rename(
            columns={
                "segment_name": "Segment",
                "segment_value": "Value",
                "previous_numerator": f"Prev {spec.numerator_label}",
                "current_numerator": f"Curr {spec.numerator_label}",
                "numerator_change": "Numerator change",
                "previous_value": "Prev value",
                "current_value": "Curr value",
                "absolute_change": "Change (pp)",
                "contribution_share_of_positive_change": "Share of positive movement",
                "min_denominator_flag": "Small n",
            }
        )
        st.markdown(f"##### Corrected segment drivers for {_selected_kpi_title(data)}")
        st.dataframe(
            top[
                [
                    "Segment",
                    "Value",
                    f"Prev {spec.numerator_label}",
                    f"Curr {spec.numerator_label}",
                    "Numerator change",
                    "Prev value",
                    "Curr value",
                    "Change (pp)",
                    "Share of positive movement",
                    "Small n",
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "Prev value": st.column_config.NumberColumn(format="percent"),
                "Curr value": st.column_config.NumberColumn(format="percent"),
                "Change (pp)": st.column_config.NumberColumn(format="percent"),
                "Share of positive movement": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption(
            "This table uses the same corrected fact table as the selected KPI. For complaint-rate "
            "KPIs, segment drivers should be read mainly as complaint-volume movement rather than "
            "a true per-account segment rate."
        )
        return

    report = data.driver_report
    interaction = report.interaction_heatmap_data

    if not interaction.empty:
        cell = interaction.iloc[0]
        st.markdown(
            f"**Strongest intersection — {cell['segment_a_value']} on "
            f"{cell['segment_b_value']}:** dispute rate moved "
            f"{_percent(cell['previous_dispute_rate'])} → "
            f"{_percent(cell['current_dispute_rate'])}, adding "
            f"{int(cell['disputed_change']):,} disputes on "
            f"{int(cell['current_purchases']):,} purchases."
        )

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("##### Count drivers — where the extra disputes came from")
        st.dataframe(
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
            ].rename(
                columns={
                    "segment_name": "Segment",
                    "segment_value": "Value",
                    "previous_disputed": "Prev",
                    "current_disputed": "Curr",
                    "disputed_change": "Change",
                    "contribution_share_of_positive_dispute_change": "Share of growth",
                    "min_denominator_flag": "Small n",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={"Share of growth": st.column_config.NumberColumn(format="percent")},
        )
        st.caption(
            "Share of growth is computed within each segment field, not pooled across "
            "them — every field partitions the same population."
        )

        st.markdown("##### Rate deterioration — where experience worsened")
        st.dataframe(
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
            ].rename(
                columns={
                    "segment_name": "Segment",
                    "segment_value": "Value",
                    "previous_dispute_rate": "Prev rate",
                    "current_dispute_rate": "Curr rate",
                    "rate_change": "Change (pp)",
                    "current_purchases": "Purchases",
                    "min_denominator_flag": "Small n",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "Prev rate": st.column_config.NumberColumn(format="percent"),
                "Curr rate": st.column_config.NumberColumn(format="percent"),
                "Change (pp)": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption(
            "Segments flagged **Small n** have too few purchases in one period for the "
            "rate to be stable; they sort below unflagged rows."
        )

    with right:
        st.markdown("##### Merchant category × channel")
        if interaction.empty:
            st.write("No interaction data available.")
        else:
            st.altair_chart(build_interaction_heatmap(interaction), width="stretch")
            st.caption(
                "Warm = dispute rate rose, cool = fell, neutral = unchanged. Hover a cell "
                "for rates, counts and whether its denominator is too small to trust."
            )
            st.markdown("##### Top interaction cells")
            st.markdown(
                """
| Column | How it is produced |
| --- | --- |
| Category | Group-by dimension from the corrected purchase fact table. |
| Channel | Second group-by dimension from the corrected purchase fact table. |
| Dispute change | Current disputed purchases minus comparison-period disputed purchases in that cell. |
| Rate change (pp) | Current dispute rate minus comparison-period dispute rate in that cell. |
| Purchases | Current-period purchase denominator for that cell, used to judge whether the rate is reliable. |
                """
            )
            st.dataframe(
                interaction.head(6)[
                    [
                        "segment_a_value",
                        "segment_b_value",
                        "disputed_change",
                        "rate_change",
                        "current_purchases",
                    ]
                ].rename(
                    columns={
                        "segment_a_value": "Category",
                        "segment_b_value": "Channel",
                        "disputed_change": "Dispute change",
                        "rate_change": "Rate change (pp)",
                        "current_purchases": "Purchases",
                    }
                ),
                width="stretch",
                hide_index=True,
                column_config={"Rate change (pp)": st.column_config.NumberColumn(format="percent")},
            )
            st.caption(
                "Each row is one merchant-category by channel combination, ranked by the corrected "
                "change in disputed purchases for the selected comparison."
            )


# ---------------------------------------------------------------------------
# 6. Complaint themes
# ---------------------------------------------------------------------------


def render_themes(data: DashboardData) -> None:
    if data.theme_report is None:
        st.info(
            "Complaint-theme analysis is currently wired to the dispute-rate investigation path. "
            "The selected KPI still has metric, quality and segment-driver evidence above.",
            icon=":material/info:",
        )
        return

    report = data.theme_report
    summary = report.theme_summary

    rising = summary[summary["complaint_change"] > 0].head(4)
    if not rising.empty:
        columns = st.columns(len(rising))
        for column, (_, row) in zip(columns, rising.iterrows()):
            column.metric(
                row["theme_name"].replace("_", " "),
                f"{int(row['current_complaints']):,}",
                f"{int(row['complaint_change']):+,} vs {report.previous_period}",
                delta_color="inverse",
                border=True,
            )

    st.markdown("##### Theme movement")
    st.dataframe(
        summary.rename(
            columns={
                "theme_name": "Theme",
                "previous_complaints": "Prev",
                "current_complaints": "Curr",
                "complaint_change": "Change",
                "previous_share": "Prev share",
                "current_share": "Curr share",
                "share_change": "Share change",
                "avg_similarity": "Avg similarity",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Prev share": st.column_config.NumberColumn(format="percent"),
            "Curr share": st.column_config.NumberColumn(format="percent"),
            "Share change": st.column_config.NumberColumn(format="percent"),
            "Avg similarity": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.markdown("##### Themes inside the driver segments")
    focus = report.segment_themes[report.segment_themes["segment_value"].isin(["travel", "mobile"])]
    st.dataframe(
        focus[
            [
                "segment_name",
                "segment_value",
                "theme_name",
                "previous_complaints",
                "current_complaints",
                "complaint_change",
                "current_share",
            ]
        ].rename(
            columns={
                "segment_name": "Segment",
                "segment_value": "Value",
                "theme_name": "Theme",
                "previous_complaints": "Prev",
                "current_complaints": "Curr",
                "complaint_change": "Change",
                "current_share": "Share of segment",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={"Share of segment": st.column_config.NumberColumn(format="percent")},
    )
    st.caption(
        f"Themes assigned by cosine similarity to a fixed business taxonomy. "
        f"Embedding backend: `{report.backend_name}`. Shares are computed within a "
        "segment value, so a theme can grow in count while losing share of a "
        "faster-growing segment."
    )


# ---------------------------------------------------------------------------
# 7. Evidence
# ---------------------------------------------------------------------------


def render_evidence(data: DashboardData) -> None:
    if data.theme_report is None:
        report = data.finance_metric_report
        st.markdown(f"##### Metric definition for {report.display_name}")
        definition = pd.DataFrame([report.metric_definition])
        st.dataframe(definition, width="stretch", hide_index=True)
        st.caption(
            "For non-dispute KPIs, this evidence tab shows the formal metric definition. "
            "Narrative complaint examples are only shown where they are tied to the grounded "
            "dispute investigation."
        )
        return

    examples = data.theme_report.representative_complaints
    st.markdown(
        "Top representative complaints for each theme in "
        f"{data.theme_report.current_period}. These are the rows behind the theme counts: "
        "if a label looks wrong, it should be visible here."
    )
    st.markdown(
        """
| Column | How it is produced |
| --- | --- |
| Theme | Embedding similarity: each narrative is compared with fixed business-theme anchor text, then assigned to the closest theme. |
| Rank | Within each theme, complaints are sorted by similarity; rank 1 is the clearest example. |
| Similarity | Cosine similarity between the complaint embedding and the assigned theme embedding. |
| Merchant category | Structured field from the linked transaction/complaint fact row, not inferred by the model. |
| Channel | Structured field from the linked transaction/complaint fact row, not inferred by the model. |
| Complaint narrative | Synthetic customer text generated from that row's issue/category/channel fields; used as readable evidence. |
        """
    )

    columns = [
        column
        for column in (
            "theme_name",
            "rank",
            "similarity",
            "merchant_category",
            "channel",
            "narrative",
        )
        if column in examples.columns
    ]
    st.dataframe(
        examples[columns].rename(
            columns={
                "theme_name": "Theme",
                "rank": "Rank",
                "similarity": "Similarity",
                "merchant_category": "Merchant category",
                "channel": "Channel",
                "narrative": "Complaint narrative",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Complaint narrative": st.column_config.TextColumn(width="large"),
            "Theme": st.column_config.TextColumn(width="medium"),
            "Similarity": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "Narratives are synthetic and are generated from each row's own merchant "
        "category, channel and issue, so a quote should not contradict the columns "
        "beside it."
    )


# ---------------------------------------------------------------------------
# 8. Action plan
# ---------------------------------------------------------------------------


def render_action_plan(data: DashboardData) -> None:
    report = data.action_report
    st.markdown(f"#### {report.decision_summary}")
    st.caption(
        "Actions are scored from selected-period evidence only. This is a prioritization "
        "aid for analysts, not an automated customer decision."
    )

    actions = report.recommendations.rename(
        columns={
            "priority": "Priority",
            "recommendation": "Recommendation",
            "action_type": "Action type",
            "target_area": "Target area",
            "owner": "Owner",
            "expected_benefit": "Expected benefit",
            "estimated_avoidable_events": "Avoidable events",
            "portfolio_rate_impact_pp": "Portfolio impact (pp)",
            "confidence": "Confidence",
            "effort": "Effort",
            "priority_score": "Score",
            "evidence_used": "Evidence used",
            "guardrail": "Guardrail",
        }
    )
    st.dataframe(
        actions[
            [
                "Priority",
                "Recommendation",
                "Target area",
                "Owner",
                "Avoidable events",
                "Portfolio impact (pp)",
                "Confidence",
                "Effort",
                "Score",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Recommendation": st.column_config.TextColumn(width="large"),
            "Portfolio impact (pp)": st.column_config.NumberColumn(format="percent"),
            "Score": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown("##### Evidence and guardrails")
        st.dataframe(
            actions[
                [
                    "Priority",
                    "Action type",
                    "Evidence used",
                    "Guardrail",
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "Evidence used": st.column_config.TextColumn(width="large"),
                "Guardrail": st.column_config.TextColumn(width="large"),
            },
        )

    with right:
        st.markdown("##### Scenario impact")
        if report.scenario_analysis.empty:
            st.info(
                "No scenario is shown because the selected window has no material avoidable-event estimate."
            )
        else:
            scenarios = report.scenario_analysis.rename(
                columns={
                    "scenario": "Scenario",
                    "target_area": "Target area",
                    "assumption": "Assumption",
                    "avoidable_events": "Avoidable events",
                    "portfolio_rate_impact_pp": "Portfolio impact (pp)",
                    "projected_current_value": "Projected KPI",
                }
            )
            st.dataframe(
                scenarios[
                    [
                        "Scenario",
                        "Assumption",
                        "Avoidable events",
                        "Portfolio impact (pp)",
                        "Projected KPI",
                    ]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "Assumption": st.column_config.TextColumn(width="large"),
                    "Portfolio impact (pp)": st.column_config.NumberColumn(format="percent"),
                    "Projected KPI": st.column_config.NumberColumn(format="percent"),
                },
            )

        st.markdown("##### Confidence scorecard")
        scorecard = report.scorecard.rename(
            columns={
                "signal": "Signal",
                "score": "Score",
                "weight": "Weight",
                "weighted_score": "Weighted score",
                "evidence": "Evidence",
            }
        )
        st.dataframe(
            scorecard,
            width="stretch",
            hide_index=True,
            column_config={
                "Score": st.column_config.NumberColumn(format="%.2f"),
                "Weight": st.column_config.NumberColumn(format="percent"),
                "Weighted score": st.column_config.NumberColumn(format="%.2f"),
                "Evidence": st.column_config.TextColumn(width="large"),
            },
        )

    st.markdown("##### Method")
    for item in report.methodology:
        st.markdown(f"- {item}")


# ---------------------------------------------------------------------------
# 9 and 10. Explanation and export
# ---------------------------------------------------------------------------


def render_explanation(data: DashboardData) -> None:
    if data.explanation is None:
        headline, summary = generic_verdict_text(data)
        report = data.finance_metric_report
        st.markdown(f"#### {headline}")
        st.write(summary)
        st.markdown("##### Metric definition")
        st.write(report.metric_definition.get("business_definition", "No definition available."))
        st.markdown("##### Recommended next steps")
        st.markdown("1. Review the top corrected segment drivers for business concentration.")
        st.markdown(
            "2. Check whether any failed data-quality rule touches this KPI's source table."
        )
        st.markdown("3. Escalate only the corrected KPI movement, not the raw movement.")
        st.download_button(
            "Download manager summary (.md)",
            data=f"# {headline}\n\n{summary}\n",
            file_name=f"metricguard_{report.metric_name}_{report.current_period}.md",
            mime="text/markdown",
            width="stretch",
        )
        return

    explanation = data.explanation
    validation = explanation.validation

    left, right = st.columns([3, 1.15])

    with left:
        st.markdown(f"#### {explanation.headline}")
        st.write(explanation.executive_summary)

        st.markdown("##### What changed")
        st.write(explanation.what_changed)
        st.markdown("##### Data quality")
        st.write(explanation.data_quality_findings)
        st.markdown("##### Business drivers")
        st.write(explanation.business_drivers)
        st.markdown("##### Customer text evidence")
        st.write(explanation.customer_text_evidence)

        st.markdown("##### Recommended next steps")
        for index, step in enumerate(explanation.recommended_next_steps, 1):
            st.markdown(f"{index}. {step}")

        st.markdown("##### Limitations")
        for item in explanation.limitations:
            st.markdown(f"- {item}")

    with right:
        st.markdown("##### Grounding check")
        st.markdown(
            status_badge(
                STATUS_PASS if validation.is_valid else STATUS_FAIL,
                "verified" if validation.is_valid else "unsupported numbers",
            ),
            unsafe_allow_html=True,
        )
        st.metric("Numbers checked", len(validation.checked))
        st.metric("Unsupported", len(validation.unsupported))
        if validation.unsupported:
            st.error("Not in the evidence packet: " + ", ".join(validation.unsupported))
        st.caption(validation.notes)
        st.markdown(f"**Generated by**  \n`{explanation.generated_by}`")
        st.caption(
            "Every number in the summary is extracted and matched against the evidence "
            "packet after generation. The prose is written from computed facts; it never "
            "calculates them."
        )

        st.markdown("##### Evidence references")
        for reference in explanation.evidence_references:
            st.markdown(f"- `{reference}`")

        st.download_button(
            "Download manager summary (.md)",
            data=explanation.to_markdown(),
            file_name=(
                f"metricguard_{data.metric_report.metric_name}_"
                f"{data.metric_report.current_period}.md"
            ),
            mime="text/markdown",
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":material/query_stats:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    tables = load_dashboard_tables()
    business_mode, kpi, period_grain, current_period, previous_period = render_header(tables)

    if business_mode != ACTIVE_BUSINESS_MODE:
        render_inactive_mode(business_mode)
        return

    data = load_dashboard_data(kpi, current_period, previous_period, period_grain)

    render_metric_overview(data)
    render_verdict(data)

    tabs = st.tabs(
        [
            "Trend",
            "Data quality",
            "Drivers",
            "Complaint themes",
            "Evidence",
            "Action plan",
            "Explanation & export",
        ]
    )
    with tabs[0]:
        render_trend(data)
    with tabs[1]:
        render_data_quality(data)
    with tabs[2]:
        render_drivers(data)
    with tabs[3]:
        render_themes(data)
    with tabs[4]:
        render_evidence(data)
    with tabs[5]:
        render_action_plan(data)
    with tabs[6]:
        render_explanation(data)


if __name__ == "__main__":
    main()
