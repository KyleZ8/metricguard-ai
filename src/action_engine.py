"""Action prioritisation for MetricGuard AI.

This module owns the decision layer of the workflow::

    metric movement -> quality -> drivers -> text evidence -> ACTION PLAN

The output is deliberately not a black-box recommender. Banking analytics teams
need a recommendation that can be challenged in review: what evidence was used,
what impact was estimated, how confidence was assigned, and what guardrail keeps
the recommendation from being over-applied.

Everything here is deterministic pandas. No model call, no randomness, and no
I/O. The caller passes in the already-built metric, quality, driver, and text
reports from the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ACTION_COLUMNS = (
    "priority",
    "recommendation",
    "action_type",
    "target_area",
    "owner",
    "expected_benefit",
    "estimated_avoidable_events",
    "portfolio_rate_impact_pp",
    "confidence",
    "effort",
    "priority_score",
    "evidence_used",
    "guardrail",
)

SCENARIO_COLUMNS = (
    "scenario",
    "target_area",
    "assumption",
    "avoidable_events",
    "portfolio_rate_impact_pp",
    "projected_current_value",
)

SCORECARD_COLUMNS = (
    "signal",
    "score",
    "weight",
    "weighted_score",
    "evidence",
)

MIN_MATERIAL_RELATIVE_INCREASE = 0.05
EFFORT_MULTIPLIERS = {"Low": 1.0, "Medium": 0.82, "High": 0.62}


@dataclass(frozen=True, eq=False)
class ActionPlanReport:
    """Dashboard-ready action plan frames for one selected KPI/window."""

    metric_name: str
    display_name: str
    current_period: str
    previous_period: str
    decision_summary: str
    recommendations: pd.DataFrame
    scenario_analysis: pd.DataFrame
    scorecard: pd.DataFrame
    methodology: tuple[str, ...]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _priority_label(score: float) -> str:
    if score >= 80:
        return "P1"
    if score >= 45:
        return "P2"
    return "P3"


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def _metric_rows(report: Any) -> tuple[pd.Series, pd.Series]:
    comparison = report.period_comparison
    raw = comparison[comparison["variant"].eq("raw")].iloc[0]
    corrected = comparison[comparison["variant"].eq("corrected")].iloc[0]
    return raw, corrected


def _remediation_row(report: Any) -> pd.Series:
    if report.remediation_impact.empty:
        return pd.Series(dtype=object)
    return report.remediation_impact.iloc[0]


def _open_quality_findings(quality_report: pd.DataFrame) -> pd.DataFrame:
    if quality_report.empty or "status" not in quality_report.columns:
        return pd.DataFrame(columns=quality_report.columns)
    return quality_report[quality_report["status"].isin(["fail", "warn"])].copy()


def _data_contract_findings(quality_report: pd.DataFrame) -> pd.DataFrame:
    """Open findings that represent data defects, not KPI movement itself."""
    findings = _open_quality_findings(quality_report)
    if findings.empty:
        return findings
    names = findings["check_name"].astype(str)
    return findings[
        ~names.str.startswith("kpi_anomaly__")
        & ~names.str.startswith("monthly_row_count_anomaly__")
    ].copy()


def _quality_confidence(row: pd.Series) -> float:
    severity = str(row.get("severity", "")).lower()
    status = str(row.get("status", "")).lower()
    base = 0.82 if status == "fail" else 0.62
    if severity == "high":
        base += 0.12
    elif severity == "medium":
        base += 0.06
    return min(base, 0.98)


def _best_generic_driver(report: Any) -> pd.Series | None:
    drivers = getattr(report, "segment_drivers", pd.DataFrame())
    if drivers is None or drivers.empty:
        return None
    candidates = drivers[
        drivers["numerator_change"].gt(0) & ~drivers["min_denominator_flag"].astype(bool)
    ].copy()
    if candidates.empty:
        return None
    return candidates.sort_values(
        ["numerator_change", "absolute_change", "contribution_share_of_positive_change"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]


def _best_interaction_driver(driver_report: Any | None) -> pd.Series | None:
    if driver_report is None:
        return None
    interaction = getattr(driver_report, "interaction_heatmap_data", pd.DataFrame())
    if interaction is None or interaction.empty:
        return None
    candidates = interaction[
        interaction["disputed_change"].gt(0) & ~interaction["min_denominator_flag"].astype(bool)
    ].copy()
    if candidates.empty:
        return None
    return candidates.sort_values(
        ["disputed_change", "rate_change", "contribution_share_of_positive_dispute_change"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]


def _best_theme(theme_report: Any | None) -> pd.Series | None:
    if theme_report is None:
        return None
    summary = getattr(theme_report, "theme_summary", pd.DataFrame())
    if summary is None or summary.empty:
        return None
    candidates = summary[summary["complaint_change"].gt(0)].copy()
    if candidates.empty:
        return None
    return candidates.sort_values(
        ["complaint_change", "share_change", "avg_similarity"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]


def _segment_avoidable_events(driver: pd.Series | None) -> float:
    if driver is None:
        return 0.0
    current_numerator = _to_float(driver.get("current_numerator"))
    current_denominator = _to_float(driver.get("current_denominator"))
    previous_value = _to_float(driver.get("previous_value"))
    return max(0.0, current_numerator - current_denominator * previous_value)


def _interaction_avoidable_events(driver: pd.Series | None) -> float:
    if driver is None:
        return 0.0
    current_disputed = _to_float(driver.get("current_disputed"))
    current_purchases = _to_float(driver.get("current_purchases"))
    previous_rate = _to_float(driver.get("previous_dispute_rate"))
    return max(0.0, current_disputed - current_purchases * previous_rate)


def _scorecard(
    corrected_change: float,
    top_driver: pd.Series | None,
    quality_findings: pd.DataFrame,
    theme: pd.Series | None,
) -> pd.DataFrame:
    """Transparent evidence-strength components behind business actions."""
    metric_score = (
        1.0
        if corrected_change >= MIN_MATERIAL_RELATIVE_INCREASE
        else (0.55 if corrected_change > 0 else 0.0)
    )
    driver_share = (
        _to_float(top_driver.get("contribution_share_of_positive_change"))
        if top_driver is not None
        else 0.0
    )
    driver_score = min(1.0, driver_share / 0.50) if driver_share > 0 else 0.0
    if top_driver is not None and bool(top_driver.get("min_denominator_flag", False)):
        driver_score *= 0.5

    blocking_quality = int(
        quality_findings[
            quality_findings["check_name"]
            .astype(str)
            .str.contains("duplicate|referential|completeness|drift", case=False, regex=True)
        ].shape[0]
    )
    quality_score = (
        max(0.25, 1.0 - min(blocking_quality, 3) * 0.20) if not quality_findings.empty else 1.0
    )

    theme_change = _to_float(theme.get("complaint_change")) if theme is not None else 0.0
    theme_similarity = _to_float(theme.get("avg_similarity")) if theme is not None else 0.0
    theme_score = min(1.0, 0.35 + theme_similarity) if theme_change > 0 else 0.0

    rows = [
        (
            "Corrected KPI movement",
            metric_score,
            0.30,
            f"Corrected percent change = {corrected_change:.4f}.",
        ),
        (
            "Driver concentration",
            driver_score,
            0.30,
            (
                f"Top driver share = {driver_share:.4f}."
                if top_driver is not None
                else "No positive driver with enough denominator."
            ),
        ),
        (
            "Quality readiness",
            quality_score,
            0.20,
            f"{len(quality_findings)} open fail/warn findings in the selected context.",
        ),
        (
            "Customer text support",
            theme_score,
            0.20,
            (
                f"Top complaint theme increased by {theme_change:.0f}."
                if theme is not None
                else "No selected-period complaint-theme report."
            ),
        ),
    ]
    frame = pd.DataFrame(rows, columns=["signal", "score", "weight", "evidence"])
    frame["weighted_score"] = frame["score"] * frame["weight"]
    return frame[list(SCORECARD_COLUMNS)]


def _recommendation_row(
    recommendation: str,
    action_type: str,
    target_area: str,
    owner: str,
    expected_benefit: str,
    estimated_avoidable_events: float,
    portfolio_rate_impact_pp: float,
    confidence_score: float,
    effort: str,
    evidence_used: str,
    guardrail: str,
    impact_score: float,
) -> dict[str, object]:
    effort_multiplier = EFFORT_MULTIPLIERS[effort]
    score = round(100.0 * impact_score * confidence_score * effort_multiplier, 1)
    return {
        "priority": _priority_label(score),
        "recommendation": recommendation,
        "action_type": action_type,
        "target_area": target_area,
        "owner": owner,
        "expected_benefit": expected_benefit,
        "estimated_avoidable_events": round(float(estimated_avoidable_events), 1),
        "portfolio_rate_impact_pp": round(float(portfolio_rate_impact_pp), 4),
        "confidence": _confidence_label(confidence_score),
        "effort": effort,
        "priority_score": score,
        "evidence_used": evidence_used,
        "guardrail": guardrail,
    }


def _scenario_table(
    report: Any,
    target_area: str,
    avoidable_events: float,
) -> pd.DataFrame:
    _, corrected = _metric_rows(report)
    current_value = _to_float(corrected["current_value"])
    denominator = _to_float(corrected["current_denominator"])
    if avoidable_events <= 0 or denominator <= 0:
        return pd.DataFrame(columns=list(SCENARIO_COLUMNS))

    rows = []
    for label, fraction in (
        ("Conservative", 0.25),
        ("Base", 0.50),
        ("Stretch", 0.75),
    ):
        events = avoidable_events * fraction
        rate_impact = events / denominator
        rows.append(
            {
                "scenario": label,
                "target_area": target_area,
                "assumption": f"{int(fraction * 100)}% of excess events return to comparison baseline",
                "avoidable_events": round(events, 1),
                "portfolio_rate_impact_pp": round(rate_impact, 4),
                "projected_current_value": max(0.0, current_value - rate_impact),
            }
        )
    return pd.DataFrame(rows, columns=list(SCENARIO_COLUMNS))


def build_action_plan(
    finance_metric_report: Any,
    quality_report: pd.DataFrame,
    driver_report: Any | None = None,
    theme_report: Any | None = None,
) -> ActionPlanReport:
    """Build prioritized actions for the selected KPI and reporting window."""
    raw, corrected = _metric_rows(finance_metric_report)
    remediation = _remediation_row(finance_metric_report)
    open_quality_findings = _open_quality_findings(quality_report)
    data_contract_findings = _data_contract_findings(quality_report)
    top_driver = _best_generic_driver(finance_metric_report)
    interaction = _best_interaction_driver(driver_report)
    theme = _best_theme(theme_report)

    corrected_change = _to_float(corrected["percent_change"])
    current_denominator = _to_float(corrected["current_denominator"])
    duplicate_numerator_removed = _to_float(remediation.get("duplicate_numerator_removed"))
    duplicate_denominator_removed = _to_float(remediation.get("duplicate_denominator_removed"))
    value_difference = _to_float(remediation.get("value_difference"))
    has_duplicate_remediation = duplicate_numerator_removed > 0 or duplicate_denominator_removed > 0
    material_increase = corrected_change >= MIN_MATERIAL_RELATIVE_INCREASE

    scorecard = _scorecard(corrected_change, top_driver, open_quality_findings, theme)
    business_confidence = float(scorecard["weighted_score"].sum())

    rows: list[dict[str, object]] = []
    if has_duplicate_remediation:
        impact_score = min(1.0, max(0.35, duplicate_numerator_removed / 150.0))
        rows.append(
            _recommendation_row(
                recommendation="Fix duplicate source-event ingestion control",
                action_type="Data quality control",
                target_area="dispute platform replay",
                owner="Data engineering",
                expected_benefit="Prevent false KPI inflation and avoid reissuing the same incident narrative.",
                estimated_avoidable_events=duplicate_numerator_removed,
                portfolio_rate_impact_pp=value_difference,
                confidence_score=0.95,
                effort="Low",
                evidence_used=(
                    f"Raw numerator exceeded corrected numerator by {duplicate_numerator_removed:.0f}; "
                    "corrected metrics remove replayed source events."
                ),
                guardrail="Do not treat removed duplicates as customer behavior; keep them in lineage/audit evidence.",
                impact_score=impact_score,
            )
        )

    if material_increase and top_driver is not None:
        segment = f"{top_driver['segment_name']} = {top_driver['segment_value']}"
        avoidable = _segment_avoidable_events(top_driver)
        impact_pp = avoidable / current_denominator if current_denominator else 0.0
        impact_score = min(
            1.0, max(0.30, _to_float(top_driver["contribution_share_of_positive_change"]))
        )
        rows.append(
            _recommendation_row(
                recommendation=f"Investigate the leading {finance_metric_report.display_name.lower()} driver",
                action_type="Business investigation",
                target_area=segment,
                owner="Risk analytics / operations",
                expected_benefit="Separate true customer or process movement from normal portfolio noise.",
                estimated_avoidable_events=avoidable,
                portfolio_rate_impact_pp=impact_pp,
                confidence_score=max(0.50, business_confidence),
                effort="Medium",
                evidence_used=(
                    f"{segment} has the largest corrected numerator increase "
                    f"({top_driver['numerator_change']:.0f}) among stable-denominator segments."
                ),
                guardrail="Validate operational cause before changing policy, messaging, or treatment rules.",
                impact_score=impact_score,
            )
        )

    if material_increase and interaction is not None:
        target = f"{interaction['segment_a_value']} x {interaction['segment_b_value']}"
        avoidable = _interaction_avoidable_events(interaction)
        impact_pp = avoidable / current_denominator if current_denominator else 0.0
        impact_score = min(
            1.0, max(0.35, _to_float(interaction["contribution_share_of_positive_dispute_change"]))
        )
        rows.append(
            _recommendation_row(
                recommendation="Pull case-level review for the top interaction cell",
                action_type="Case review",
                target_area=target,
                owner="Disputes operations",
                expected_benefit="Find whether the spike is concentrated in a specific channel journey.",
                estimated_avoidable_events=avoidable,
                portfolio_rate_impact_pp=impact_pp,
                confidence_score=max(0.62, business_confidence),
                effort="Medium",
                evidence_used=(
                    f"{target} is the largest corrected interaction cell by additional disputes "
                    f"({interaction['disputed_change']:.0f})."
                ),
                guardrail="Use this as investigation targeting, not an automated customer decision.",
                impact_score=impact_score,
            )
        )

    if material_increase and theme is not None:
        impact_score = min(0.85, max(0.25, _to_float(theme["current_share"])))
        rows.append(
            _recommendation_row(
                recommendation="Review customer-facing language and support handling for the top complaint theme",
                action_type="Customer experience",
                target_area=str(theme["theme_name"]),
                owner="Product / customer operations",
                expected_benefit="Reduce avoidable complaints and clarify whether customer language matches metric drivers.",
                estimated_avoidable_events=_to_float(theme["complaint_change"]),
                portfolio_rate_impact_pp=0.0,
                confidence_score=min(0.85, max(0.50, _to_float(theme["avg_similarity"]))),
                effort="Medium",
                evidence_used=(
                    f"Complaint theme increased by {theme['complaint_change']:.0f} and now represents "
                    f"{_to_float(theme['current_share']):.2%} of selected-period complaints."
                ),
                guardrail="Complaint text supports root-cause discovery; it is not proof of financial loss by itself.",
                impact_score=impact_score,
            )
        )

    for _, finding in data_contract_findings.head(2).iterrows():
        if has_duplicate_remediation and "duplicate" in str(finding.get("check_name", "")).lower():
            continue
        affected_rows = _to_float(finding.get("affected_rows"))
        denominator = max(current_denominator, 1.0)
        impact_score = min(0.90, max(0.20, affected_rows / denominator))
        rows.append(
            _recommendation_row(
                recommendation=f"Resolve selected-period quality finding: {finding['check_name']}",
                action_type="Data quality review",
                target_area=str(finding.get("table_name", "selected tables")),
                owner="Data quality / source owner",
                expected_benefit="Protect downstream segment analysis and reporting trust.",
                estimated_avoidable_events=0.0,
                portfolio_rate_impact_pp=0.0,
                confidence_score=_quality_confidence(finding),
                effort="Low",
                evidence_used=str(finding.get("explanation", "")),
                guardrail="Fix the data contract first; do not infer customer behavior from corrupted fields.",
                impact_score=impact_score,
            )
        )

    if not rows:
        rows.append(
            _recommendation_row(
                recommendation="Monitor the KPI instead of opening an incident",
                action_type="Monitoring",
                target_area=finance_metric_report.display_name,
                owner="Analytics owner",
                expected_benefit="Avoid alert fatigue when corrected movement is not material.",
                estimated_avoidable_events=0.0,
                portfolio_rate_impact_pp=0.0,
                confidence_score=0.70 if corrected_change <= 0 else 0.55,
                effort="Low",
                evidence_used=(
                    f"Corrected percent change is {corrected_change:.4f}; "
                    "no material corrected increase crossed the action threshold."
                ),
                guardrail="Re-run after the next data load and escalate only if movement persists or grows.",
                impact_score=0.55,
            )
        )

    recommendations = pd.DataFrame(rows, columns=list(ACTION_COLUMNS)).sort_values(
        ["priority_score", "recommendation"],
        ascending=[False, True],
        kind="mergesort",
    )
    recommendations["priority"] = recommendations["priority_score"].map(_priority_label)
    recommendations = recommendations.reset_index(drop=True)

    scenario_candidates = recommendations[
        recommendations["action_type"].isin(
            ["Business investigation", "Case review", "Customer experience"]
        )
        & recommendations["estimated_avoidable_events"].gt(0)
    ]
    scenario_source = (
        scenario_candidates.iloc[0]
        if not scenario_candidates.empty
        else recommendations[recommendations["estimated_avoidable_events"].gt(0)].head(1).iloc[0]
        if recommendations["estimated_avoidable_events"].gt(0).any()
        else None
    )
    scenario_target = scenario_source["target_area"] if scenario_source is not None else ""
    scenario_events = (
        _to_float(scenario_source["estimated_avoidable_events"])
        if scenario_source is not None
        else 0.0
    )
    scenario_analysis = _scenario_table(
        finance_metric_report, str(scenario_target), scenario_events
    )

    if has_duplicate_remediation and material_increase:
        decision_summary = (
            "Open a data-quality remediation and a targeted business investigation. "
            "The reported movement is partly inflated, but the corrected KPI still increased materially."
        )
    elif has_duplicate_remediation:
        decision_summary = "Prioritise data-quality remediation. After correction, the selected KPI does not justify a business spike response."
    elif material_increase:
        decision_summary = "Open a targeted business investigation. The corrected KPI increased materially and no duplicate-event remediation explains it."
    elif not data_contract_findings.empty:
        decision_summary = "Do not open a KPI spike incident, but resolve the selected quality findings before relying on segment labels."
    else:
        decision_summary = (
            "Monitor only. The selected corrected KPI movement does not clear the action threshold."
        )

    methodology = (
        "Quality gate: data defects and remediation are assessed before business recommendations.",
        "Counterfactual sizing: avoidable events estimate how many current-period numerator events would disappear if the target returned to its comparison-period rate.",
        "Evidence triangulation: KPI movement, driver concentration, quality findings, and complaint themes each contribute to confidence.",
        "Priority score: impact score times confidence score times an effort multiplier; governance guardrails are shown beside every action.",
    )

    return ActionPlanReport(
        metric_name=finance_metric_report.metric_name,
        display_name=finance_metric_report.display_name,
        current_period=finance_metric_report.current_period,
        previous_period=finance_metric_report.previous_period,
        decision_summary=decision_summary,
        recommendations=recommendations,
        scenario_analysis=scenario_analysis,
        scorecard=scorecard,
        methodology=methodology,
    )


if __name__ == "__main__":
    from metric_engine import build_finance_metric_report, resolve_tables
    from quality_checks import run_quality_checks

    tables = resolve_tables()
    metric_report = build_finance_metric_report("dispute_rate", tables)
    quality = run_quality_checks(tables)
    report = build_action_plan(metric_report, quality)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)
    print(report.decision_summary)
    print(report.recommendations.to_string(index=False))
    print(report.scenario_analysis.to_string(index=False))
