"""Grounded explanation layer for MetricGuard AI.

This module owns the last step of the MetricGuard workflow::

    metric -> data quality -> anomaly -> drivers -> text themes -> EXPLANATION

Every preceding module computes facts. This one only *communicates* them. The
split is the whole point of the project: in a financial-services setting a
dashboard narrative is only trustworthy if the numbers in it were produced by
auditable code, not by a language model.

How that separation is enforced
-------------------------------
1. :func:`build_evidence_packet` assembles a compact, JSON-serialisable packet
   from the four analysis modules. Nothing else is ever shown to the model.
2. Every numeric value placed in the packet is simultaneously registered in a
   :class:`SupportedNumbers` whitelist, in each form a writer might plausibly
   render it (a rate as ``0.0177`` and as ``1.77``, a count with and without a
   thousands separator, each rounded to a few decimal places).
3. The explanation is generated either deterministically or by the model.
4. :func:`validate_numbers` re-reads the finished prose, extracts every number,
   and checks it against that whitelist. The result travels with the
   explanation, so a reader can always see whether it was verified.

The model is therefore never asked to calculate anything, and cannot smuggle a
number past step 4 without it being flagged.

Two generation paths, one schema
--------------------------------
``deterministic``
    Templated prose built only from packet values. Always available, needs no
    API key and no network, and is what the tests exercise. It is a real
    fallback, not a stub: it produces the manager summary the demo shows.

``openai``
    Used when ``OPENAI_API_KEY`` is present, or when a client is injected. Low
    temperature, structured JSON output, and a system prompt that forbids
    introducing numbers. Its output goes through exactly the same validation.

Both return an :class:`Explanation` with identical fields, so downstream code
never branches on which one ran -- ``generated_by`` records it instead.

Run directly to print a manager summary::

    python src/explanation_engine.py
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from driver_analysis import DriverReport, build_driver_report
from metric_engine import MetricReport, build_metric_report
from quality_checks import (
    DATA_DIR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARN,
    run_quality_checks,
    summarize_quality_checks,
)
from text_theme_analysis import TextThemeReport, build_text_theme_report

DEFAULT_MODEL_ENV_VAR = "METRICGUARD_LLM_MODEL"
DEFAULT_MODEL = "gpt-5"
API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_TEMPERATURE = 0.1

GENERATOR_DETERMINISTIC = "deterministic-fallback"

EXPLANATION_FIELDS = (
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
)

# Numbers at or below this value are treated as structural rather than
# quantitative -- "the top 3 segments", "two checks failed". They are extremely
# common in prose and almost never carry a factual claim about the portfolio.
# The trade-off is explicit: a miscounted list length will not be caught. Metric
# values, rates, and population counts all sit far above this threshold.
DEFAULT_MATERIAL_INTEGER_THRESHOLD = 10
MATERIAL_RELATIVE_INCREASE_THRESHOLD = 0.05

# Standing caveats. These are properties of the method, not of one month's data,
# so they belong to the module rather than to any single run.
STANDING_LIMITATIONS = (
    "Driver and complaint findings are analytical signals, not causal proof; they identify "
    "where to investigate, not why the movement happened.",
    "All figures come from synthetic data generated for this project and describe no real "
    "customer, account, or portfolio.",
    "Dispute rate is measured on transaction date, so late-arriving disputes can revise the "
    "most recent period after publication.",
    "This summary supports analyst investigation only. It is not credit policy, financial "
    "advice, or a customer-level decision.",
)


class _ResponsesClient(Protocol):
    """The slice of the OpenAI client this module uses.

    Declaring it as a protocol is what lets a test inject a fake with no network
    access and no API key, while the production path passes a real ``OpenAI()``.
    """

    responses: Any


# ---------------------------------------------------------------------------
# Number whitelist
# ---------------------------------------------------------------------------


class SupportedNumbers:
    """Every numeric value the explanation is allowed to contain.

    A value is registered in each form a writer might reasonably produce, so
    that display rounding is accepted but invention is not. Registering
    ``0.017679`` as a rate also admits ``1.77``, ``1.8`` and ``0.0177``; it does
    not admit ``2.1``.
    """

    #: Decimal places a writer might round to when rendering a value.
    #
    # Zero is deliberately excluded. Rounding to whole numbers would make a
    # registered rate of 1.7679 accept an invented 2.1 -- both round to 2 -- which
    # is exactly the class of error this whitelist exists to catch. Integers are
    # unaffected, since rounding an integer to one decimal place is a no-op.
    #
    # One decimal place is kept so a legitimate "1.8%" rendering is accepted. The
    # residual tolerance that buys is small (a claim within half a tenth of a
    # real value); invented figures are not near-misses.
    ROUNDINGS = (1, 2, 3, 4, 6)

    def __init__(self, tolerance: float = 1e-9) -> None:
        self.tolerance = tolerance
        self._values: dict[float, str] = {}

    def _register(self, value: float, label: str) -> None:
        if value is None or not np.isfinite(value):
            return
        for digits in self.ROUNDINGS:
            self._values.setdefault(round(float(value), digits), label)

    def add(self, value: float | int | None, label: str) -> None:
        """Register a plain count or scalar."""
        if value is None:
            return
        self._register(float(value), label)

    def add_rate(self, value: float | None, label: str) -> None:
        """Register a proportion, and the percentage a writer would print."""
        if value is None or not np.isfinite(float(value)):
            return
        self._register(float(value), label)
        self._register(float(value) * 100.0, f"{label} (as percent)")
        if float(value) < 0:
            self._register(abs(float(value)), f"{label} (absolute magnitude)")
            self._register(abs(float(value)) * 100.0, f"{label} (absolute percent magnitude)")

    def add_many(self, values: Mapping[str, Any]) -> None:
        for label, value in values.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
                value, bool
            ):
                self.add(float(value), label)

    def contains(self, value: float) -> bool:
        if not np.isfinite(value):
            return False
        for digits in self.ROUNDINGS:
            if round(float(value), digits) in self._values:
                return True
        return False

    def label_for(self, value: float) -> str | None:
        for digits in self.ROUNDINGS:
            key = round(float(value), digits)
            if key in self._values:
                return self._values[key]
        return None

    def as_sorted_values(self) -> tuple[float, ...]:
        return tuple(sorted(self._values))

    def __len__(self) -> int:
        return len(self._values)


# Period tokens such as "2026-08" are masked before number extraction: their
# digits are labels, not quantities, and splitting them produces phantom numbers.
_PERIOD_PATTERN = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
_NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")


@dataclass(frozen=True)
class NumberValidation:
    """Whether every material number in an explanation is backed by evidence."""

    status: str
    checked: tuple[str, ...]
    supported: tuple[str, ...]
    unsupported: tuple[str, ...]
    ignored: tuple[str, ...]
    notes: str

    @property
    def is_valid(self) -> bool:
        return self.status == STATUS_PASS


def extract_numbers(text: str) -> list[str]:
    """Return the numeric tokens in ``text``, ignoring period labels."""
    masked = _PERIOD_PATTERN.sub(" ", text)
    return [match.group(0) for match in _NUMBER_PATTERN.finditer(masked)]


def _token_to_float(token: str) -> float | None:
    cleaned = token.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def validate_numbers(
    text: str,
    supported: SupportedNumbers,
    material_integer_threshold: int = DEFAULT_MATERIAL_INTEGER_THRESHOLD,
) -> NumberValidation:
    """Check every material number in ``text`` against the evidence whitelist.

    Small integers are treated as structural and reported under ``ignored``
    rather than silently dropped, so the exemption stays visible to a reviewer.
    """
    checked: list[str] = []
    supported_tokens: list[str] = []
    unsupported_tokens: list[str] = []
    ignored_tokens: list[str] = []

    for token in extract_numbers(text):
        value = _token_to_float(token)
        if value is None:
            continue
        is_structural = (
            float(value).is_integer()
            and abs(value) <= material_integer_threshold
            and "%" not in token
            and "." not in token
        )
        if is_structural and not supported.contains(value):
            ignored_tokens.append(token)
            continue
        checked.append(token)
        if supported.contains(value):
            supported_tokens.append(token)
        else:
            unsupported_tokens.append(token)

    if unsupported_tokens:
        notes = (
            f"{len(unsupported_tokens)} number(s) in the explanation are not present in the "
            "evidence packet: " + ", ".join(sorted(set(unsupported_tokens)))
        )
        status = STATUS_FAIL
    else:
        notes = f"All {len(supported_tokens)} material number(s) trace to the evidence packet." + (
            f" {len(ignored_tokens)} small integer(s) treated as structural."
            if ignored_tokens
            else ""
        )
        status = STATUS_PASS

    return NumberValidation(
        status=status,
        checked=tuple(checked),
        supported=tuple(supported_tokens),
        unsupported=tuple(unsupported_tokens),
        ignored=tuple(ignored_tokens),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Evidence packet
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EvidencePacket:
    """The only material the explanation layer is allowed to describe.

    ``facts`` is the JSON-serialisable payload handed to the model.
    ``supported_numbers`` is the whitelist built from it at the same time, which
    is what makes post-generation validation possible at all.
    """

    facts: dict[str, Any]
    supported_numbers: SupportedNumbers

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.facts, indent=indent, sort_keys=False, default=str)

    @property
    def metric_name(self) -> str:
        return str(self.facts["metric_name"])

    @property
    def current_period(self) -> str:
        return str(self.facts["current_period"])

    @property
    def previous_period(self) -> str:
        return str(self.facts["previous_period"])


def _round(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if not np.isfinite(number) else round(number, digits)
    return value


def build_evidence_packet(
    metric_report: MetricReport | None = None,
    quality_report: pd.DataFrame | None = None,
    driver_report: DriverReport | None = None,
    theme_report: TextThemeReport | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    top_n: int = 3,
    data_dir: Path = DATA_DIR,
) -> EvidencePacket:
    """Assemble the evidence packet from the four analysis modules.

    Any report not supplied is computed. Supplying them is the normal path for a
    dashboard (which has already rendered these tables) and for tests, which
    build the pipeline once and reuse it.
    """
    if metric_report is None:
        metric_report = build_metric_report(tables=tables, data_dir=data_dir)
    if quality_report is None:
        quality_report = run_quality_checks(tables=tables, data_dir=data_dir)
    if driver_report is None:
        driver_report = build_driver_report(tables=tables, data_dir=data_dir)
    if theme_report is None:
        theme_report = build_text_theme_report(tables=tables, data_dir=data_dir)

    supported = SupportedNumbers()

    # --- metric movement, raw and corrected -------------------------------
    impact = metric_report.remediation_impact.iloc[0]
    comparison = metric_report.period_comparison.set_index("variant")
    raw = comparison.loc["raw"]
    corrected = comparison.loc["corrected"]

    metric_facts = {
        "raw_current_rate": _round(impact["raw_dispute_rate"]),
        "raw_previous_rate": _round(raw["previous_value"]),
        "corrected_current_rate": _round(impact["corrected_dispute_rate"]),
        "corrected_previous_rate": _round(corrected["previous_value"]),
        "raw_percent_change": _round(raw["percent_change"]),
        "corrected_percent_change": _round(corrected["percent_change"]),
        "rate_difference_from_duplicates": _round(impact["dispute_rate_difference"]),
        "raw_current_disputed": int(impact["raw_disputed_count"]),
        "corrected_current_disputed": int(impact["corrected_disputed_count"]),
        "duplicate_rows_removed": int(impact["duplicate_disputed_rows_removed"]),
        "raw_current_purchases": int(impact["raw_purchase_count"]),
        "corrected_current_purchases": int(impact["corrected_purchase_count"]),
    }
    for key in (
        "raw_current_rate",
        "raw_previous_rate",
        "corrected_current_rate",
        "corrected_previous_rate",
        "raw_percent_change",
        "corrected_percent_change",
        "rate_difference_from_duplicates",
    ):
        supported.add_rate(metric_facts[key], f"metric.{key}")
    for key in (
        "raw_current_disputed",
        "corrected_current_disputed",
        "duplicate_rows_removed",
        "raw_current_purchases",
        "corrected_current_purchases",
    ):
        supported.add(metric_facts[key], f"metric.{key}")

    # --- data quality ------------------------------------------------------
    quality_summary = summarize_quality_checks(quality_report)
    supported.add(quality_summary["total_checks"], "quality.total_checks")
    supported.add(quality_summary["failed"], "quality.failed")
    supported.add(quality_summary["warned"], "quality.warned")
    supported.add(quality_summary["passed"], "quality.passed")
    supported.add_rate(quality_summary["pass_rate"], "quality.pass_rate")

    def _quality_rows(status: str) -> list[dict[str, Any]]:
        rows = []
        for _, row in quality_report[quality_report["status"].eq(status)].iterrows():
            supported.add(int(row["affected_rows"]), f"quality.{row['check_name']}.affected_rows")
            rows.append(
                {
                    "check_name": str(row["check_name"]),
                    "check_type": str(row["check_type"]),
                    "table_name": str(row["table_name"]),
                    "severity": str(row["severity"]),
                    "affected_rows": int(row["affected_rows"]),
                    "observed_value": str(row["observed_value"]),
                    "recommended_action": str(row["recommended_action"]),
                }
            )
        return rows

    quality_facts = {
        "total_checks": int(quality_summary["total_checks"]),
        "failed": int(quality_summary["failed"]),
        "warned": int(quality_summary["warned"]),
        "passed": int(quality_summary["passed"]),
        "failing_checks": _quality_rows(STATUS_FAIL),
        "warning_checks": _quality_rows(STATUS_WARN),
    }

    # --- drivers -----------------------------------------------------------
    def _driver_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for _, row in frame.head(top_n).iterrows():
            name = f"{row['segment_name']}={row['segment_value']}"
            supported.add(int(row["disputed_change"]), f"driver.{name}.disputed_change")
            supported.add(int(row["current_disputed"]), f"driver.{name}.current_disputed")
            supported.add(int(row["previous_disputed"]), f"driver.{name}.previous_disputed")
            supported.add_rate(row["current_dispute_rate"], f"driver.{name}.current_rate")
            supported.add_rate(row["previous_dispute_rate"], f"driver.{name}.previous_rate")
            supported.add_rate(row["rate_change"], f"driver.{name}.rate_change")
            supported.add_rate(
                row["contribution_share_of_positive_dispute_change"], f"driver.{name}.contribution"
            )
            rows.append(
                {
                    "segment_name": str(row["segment_name"]),
                    "segment_value": str(row["segment_value"]),
                    "previous_disputed": int(row["previous_disputed"]),
                    "current_disputed": int(row["current_disputed"]),
                    "disputed_change": int(row["disputed_change"]),
                    "previous_dispute_rate": _round(row["previous_dispute_rate"]),
                    "current_dispute_rate": _round(row["current_dispute_rate"]),
                    "rate_change": _round(row["rate_change"]),
                    "contribution_share": _round(
                        row["contribution_share_of_positive_dispute_change"]
                    ),
                    "small_denominator_flag": bool(row["min_denominator_flag"]),
                }
            )
        return rows

    interaction = driver_report.interaction_heatmap_data
    top_interaction: dict[str, Any] | None = None
    if not interaction.empty:
        cell = interaction.iloc[0]
        name = f"{cell['segment_a_value']}x{cell['segment_b_value']}"
        supported.add(int(cell["disputed_change"]), f"interaction.{name}.disputed_change")
        supported.add(int(cell["current_disputed"]), f"interaction.{name}.current_disputed")
        supported.add_rate(cell["current_dispute_rate"], f"interaction.{name}.current_rate")
        supported.add_rate(cell["previous_dispute_rate"], f"interaction.{name}.previous_rate")
        supported.add_rate(cell["rate_change"], f"interaction.{name}.rate_change")
        top_interaction = {
            "segment_a_name": str(cell["segment_a_name"]),
            "segment_a_value": str(cell["segment_a_value"]),
            "segment_b_name": str(cell["segment_b_name"]),
            "segment_b_value": str(cell["segment_b_value"]),
            "previous_disputed": int(cell["previous_disputed"]),
            "current_disputed": int(cell["current_disputed"]),
            "disputed_change": int(cell["disputed_change"]),
            "previous_dispute_rate": _round(cell["previous_dispute_rate"]),
            "current_dispute_rate": _round(cell["current_dispute_rate"]),
            "rate_change": _round(cell["rate_change"]),
        }

    driver_facts = {
        "top_count_drivers": _driver_rows(driver_report.top_count_drivers),
        "top_rate_drivers": _driver_rows(driver_report.top_rate_deterioration),
        "top_interaction_driver": top_interaction,
    }

    # --- complaint text ----------------------------------------------------
    theme_rows = []
    for _, row in theme_report.theme_summary.head(top_n).iterrows():
        theme = str(row["theme_name"])
        supported.add(int(row["current_complaints"]), f"theme.{theme}.current")
        supported.add(int(row["previous_complaints"]), f"theme.{theme}.previous")
        supported.add(int(row["complaint_change"]), f"theme.{theme}.change")
        supported.add_rate(row["current_share"], f"theme.{theme}.current_share")
        supported.add_rate(row["share_change"], f"theme.{theme}.share_change")
        theme_rows.append(
            {
                "theme_name": theme,
                "previous_complaints": int(row["previous_complaints"]),
                "current_complaints": int(row["current_complaints"]),
                "complaint_change": int(row["complaint_change"]),
                "current_share": _round(row["current_share"]),
                "share_change": _round(row["share_change"]),
            }
        )

    example_rows = []
    top_theme_names = [row["theme_name"] for row in theme_rows]
    examples = theme_report.representative_complaints.copy()
    examples["theme_name"] = examples["theme_name"].astype(str)
    for theme in top_theme_names:
        match = examples[examples["theme_name"].eq(theme)]
        if match.empty:
            continue
        row = match.iloc[0]
        example_rows.append(
            {
                "theme_name": str(row["theme_name"]),
                "merchant_category": str(row.get("merchant_category", "")),
                "channel": str(row.get("channel", "")),
                "narrative": str(row["narrative"]),
            }
        )
        if len(example_rows) >= top_n:
            break

    theme_facts = {
        "embedding_backend": theme_report.backend_name,
        "top_themes": theme_rows,
        "representative_complaints": example_rows,
    }

    facts: dict[str, Any] = {
        "metric_name": metric_report.metric_name,
        "current_period": metric_report.current_period,
        "previous_period": metric_report.previous_period,
        "metric": metric_facts,
        "data_quality": quality_facts,
        "drivers": driver_facts,
        "text_themes": theme_facts,
        "limitations": list(STANDING_LIMITATIONS),
    }
    return EvidencePacket(facts=facts, supported_numbers=supported)


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Explanation:
    """A manager-ready explanation plus the proof that its numbers check out."""

    headline: str
    executive_summary: str
    what_changed: str
    data_quality_findings: str
    business_drivers: str
    customer_text_evidence: str
    recommended_next_steps: tuple[str, ...]
    limitations: tuple[str, ...]
    evidence_references: tuple[str, ...]
    generated_by: str
    validation: NumberValidation

    def to_dict(self) -> dict[str, Any]:
        payload = {field_name: getattr(self, field_name) for field_name in EXPLANATION_FIELDS}
        for key, value in payload.items():
            if isinstance(value, tuple):
                payload[key] = list(value)
        payload["validation"] = asdict(self.validation)
        return payload

    def narrative_text(self) -> str:
        """Every prose field joined, which is what number validation reads."""
        parts = [
            self.headline,
            self.executive_summary,
            self.what_changed,
            self.data_quality_findings,
            self.business_drivers,
            self.customer_text_evidence,
            *self.recommended_next_steps,
            *self.limitations,
        ]
        return "\n".join(str(part) for part in parts)

    def to_markdown(self) -> str:
        steps = "\n".join(
            f"{index}. {step}" for index, step in enumerate(self.recommended_next_steps, 1)
        )
        limits = "\n".join(f"- {item}" for item in self.limitations)
        references = "\n".join(f"- `{item}`" for item in self.evidence_references)
        return (
            f"# {self.headline}\n\n"
            f"{self.executive_summary}\n\n"
            f"## What changed\n{self.what_changed}\n\n"
            f"## Data quality\n{self.data_quality_findings}\n\n"
            f"## Business drivers\n{self.business_drivers}\n\n"
            f"## Customer text evidence\n{self.customer_text_evidence}\n\n"
            f"## Recommended next steps\n{steps}\n\n"
            f"## Limitations\n{limits}\n\n"
            f"## Evidence references\n{references}\n\n"
            f"---\nGenerated by: {self.generated_by}  \n"
            f"Number check: {self.validation.status} - {self.validation.notes}\n"
        )


def _percent(value: Any, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def _abs_percent(value: Any, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{abs(float(value)) * 100:.{digits}f}%"


def _count(value: Any) -> str:
    return f"{int(value):,}"


# ---------------------------------------------------------------------------
# Deterministic path
# ---------------------------------------------------------------------------


def build_deterministic_explanation(packet: EvidencePacket) -> Explanation:
    """Compose the explanation from packet values alone.

    Written as templates over the packet, so it cannot state a number the
    evidence does not contain -- it still goes through the same validator, which
    is what makes that claim checkable rather than assumed.
    """
    facts = packet.facts
    metric = facts["metric"]
    quality = facts["data_quality"]
    drivers = facts["drivers"]
    themes = facts["text_themes"]
    current = facts["current_period"]
    previous = facts["previous_period"]
    name = facts["metric_name"]

    raw_change = metric["raw_percent_change"]
    corrected_change = metric["corrected_percent_change"]
    duplicates = metric["duplicate_rows_removed"]
    has_duplicates = bool(duplicates)

    still_elevated = (
        corrected_change is not None
        and np.isfinite(float(corrected_change))
        and float(corrected_change) > 0
    )
    material_increase = (
        still_elevated and float(corrected_change) >= MATERIAL_RELATIVE_INCREASE_THRESHOLD
    )

    if has_duplicates and still_elevated:
        headline = (
            f"{name} rose {_percent(raw_change)} in {current}, but {_percent(corrected_change)} "
            "once duplicate source events are removed"
        )
        verdict = (
            "Part of the reported movement is a data-quality artifact and part of it is real. "
            "Both need a different response, so they should not be escalated as one number."
        )
    elif has_duplicates:
        headline = f"{name} movement in {current} is explained by duplicate source events"
        verdict = (
            "After removing duplicated source events the metric is no longer elevated, so this "
            "looks like a pipeline problem rather than a customer or risk problem."
        )
    elif still_elevated:
        headline = f"{name} rose {_abs_percent(corrected_change)} in {current} with no duplicate records found"
        verdict = (
            "No duplicate source events were found in the selected period, so raw and corrected "
            "figures match. Treat this as ordinary metric movement, not a replay-batch issue."
        )
    else:
        direction = "fell" if float(corrected_change or 0) < 0 else "was flat"
        headline = f"{name} {direction} {_abs_percent(corrected_change)} in {current} with no duplicate records found"
        verdict = (
            "No duplicate source events were found in the selected period, and there is no "
            "portfolio-level increase to escalate."
        )

    if has_duplicates:
        executive_summary = (
            f"{name} for {current} was {_percent(metric['raw_current_rate'])} as reported, against "
            f"{_percent(metric['raw_previous_rate'])} in {previous}. After deduplicating replayed "
            f"source events the corrected {current} figure is "
            f"{_percent(metric['corrected_current_rate'])}, a corrected change of "
            f"{_percent(corrected_change)} rather than the headline {_percent(raw_change)}. "
            f"{verdict}"
        )
        what_changed = (
            f"Reported {name} moved from {_percent(metric['raw_previous_rate'])} in {previous} to "
            f"{_percent(metric['raw_current_rate'])} in {current}, built on "
            f"{_count(metric['raw_current_disputed'])} disputed purchases out of "
            f"{_count(metric['raw_current_purchases'])}. Removing "
            f"{_count(duplicates)} duplicated source events leaves "
            f"{_count(metric['corrected_current_disputed'])} disputed purchases out of "
            f"{_count(metric['corrected_current_purchases'])}, a corrected rate of "
            f"{_percent(metric['corrected_current_rate'])}. Duplicate records therefore inflated the "
            f"reported rate by {_percent(metric['rate_difference_from_duplicates'])} in absolute terms "
            "(percentage points), not as a share of it."
        )
    else:
        executive_summary = (
            f"{name} for {current} was {_percent(metric['raw_current_rate'])} as reported, against "
            f"{_percent(metric['raw_previous_rate'])} in {previous}. The corrected figure is the same "
            f"at {_percent(metric['corrected_current_rate'])}, because no duplicated source events "
            f"were removed from the selected period. The corrected change is {_percent(corrected_change)}. "
            f"{verdict}"
        )
        what_changed = (
            f"Reported {name} moved from {_percent(metric['raw_previous_rate'])} in {previous} to "
            f"{_percent(metric['raw_current_rate'])} in {current}, built on "
            f"{_count(metric['raw_current_disputed'])} disputed purchases out of "
            f"{_count(metric['raw_current_purchases'])}. The corrected numerator and denominator are "
            f"unchanged at {_count(metric['corrected_current_disputed'])} disputed purchases out of "
            f"{_count(metric['corrected_current_purchases'])}, so the raw and corrected rates both read "
            f"{_percent(metric['corrected_current_rate'])}."
        )

    if quality["failing_checks"] or quality["warning_checks"]:
        lead = quality["failing_checks"][:2] or quality["warning_checks"][:2]
        detail = "; ".join(
            f"{item['check_name']} ({_count(item['affected_rows'])} rows, {item['severity']} severity)"
            for item in lead
        )
        suffix = (
            "Deduplication has been applied to every figure labelled corrected above; the remaining findings are open."
            if has_duplicates
            else "These findings should be reviewed, but they are not duplicate-row remediation items for this period."
        )
        data_quality_findings = (
            f"{quality['failed']} of {quality['total_checks']} relevant automated checks failed and "
            f"{quality['warned']} raised a warning. The findings that affect this selected period are: "
            f"{detail}. {suffix}"
        )
    else:
        data_quality_findings = (
            f"All {quality['total_checks']} relevant automated data-quality checks passed, so the reported "
            "movement is measured on records that meet the current contract."
        )

    count_drivers = drivers["top_count_drivers"]
    rate_drivers = drivers["top_rate_drivers"]
    interaction = drivers["top_interaction_driver"]

    if count_drivers and material_increase:
        top = count_drivers[0]
        driver_sentences = [
            f"The largest contributor to the increase is "
            f"{top['segment_name']} = {top['segment_value']}, up "
            f"{_count(top['disputed_change'])} disputes and accounting for "
            f"{_percent(top['contribution_share'])} of the growth within that segment field."
        ]
        if rate_drivers:
            worst = rate_drivers[0]
            driver_sentences.append(
                f"By rate rather than volume, {worst['segment_name']} = {worst['segment_value']} "
                f"deteriorated most, from {_percent(worst['previous_dispute_rate'])} to "
                f"{_percent(worst['current_dispute_rate'])}."
            )
        if interaction:
            driver_sentences.append(
                f"The two cuts intersect: {interaction['segment_a_value']} on "
                f"{interaction['segment_b_value']} moved from "
                f"{_percent(interaction['previous_dispute_rate'])} to "
                f"{_percent(interaction['current_dispute_rate'])}, adding "
                f"{_count(interaction['disputed_change'])} disputes on its own."
            )
        business_drivers = " ".join(driver_sentences)
    elif count_drivers:
        top = count_drivers[0]
        movement_note = (
            "increased, but the movement is below the materiality threshold for escalation"
            if still_elevated
            else "did not increase"
        )
        business_drivers = (
            f"The portfolio-level {name} {movement_note}, so the driver table should be read as "
            "local segment movement rather than an escalation list. The largest positive local "
            f"movement is {top['segment_name']} = {top['segment_value']}, up "
            f"{_count(top['disputed_change'])} disputes. Use this as context only; the selected "
            "movement does not call for spike remediation."
        )
    else:
        business_drivers = "No segment driver stood out in the corrected data."

    top_themes = themes["top_themes"]
    if top_themes:
        lead_theme = top_themes[0]
        theme_detail = ", ".join(
            f"{item['theme_name']} ({_count(item['previous_complaints'])} to "
            f"{_count(item['current_complaints'])})"
            for item in top_themes
        )
        example = next(
            (
                item
                for item in themes["representative_complaints"]
                if item["theme_name"] == lead_theme["theme_name"]
            ),
            themes["representative_complaints"][0] if themes["representative_complaints"] else None,
        )
        quote = f' A representative complaint reads: "{example["narrative"]}"' if example else ""
        if material_increase:
            opening = "Complaint themes move in the same direction as the corrected metric."
        else:
            opening = (
                "Complaint themes are supporting evidence only here, because the selected metric "
                "movement is not a material portfolio-level spike."
            )
        customer_text_evidence = (
            f"{opening} The largest theme movement is {lead_theme['theme_name']}, up "
            f"{_count(lead_theme['complaint_change'])} complaints and now "
            f"{_percent(lead_theme['current_share'])} of the month's complaints. "
            f"Theme movements: {theme_detail}.{quote} "
            f"Themes were assigned with {themes['embedding_backend']}."
        )
    else:
        customer_text_evidence = "No complaint theme moved materially between the two periods."

    recommended_next_steps = []
    if has_duplicates:
        recommended_next_steps.extend(
            [
                f"Reissue the {current} {name} using deduplicated source events; the corrected figure is "
                f"{_percent(metric['corrected_current_rate'])}, not {_percent(metric['raw_current_rate'])}.",
                "Confirm with the source-system owner whether the replayed batch was reprocessed upstream, "
                "and whether any earlier period is affected.",
            ]
        )
    elif material_increase:
        recommended_next_steps.append(
            f"Keep the published {current} {name} at {_percent(metric['corrected_current_rate'])}; "
            "raw and corrected values match for this period."
        )
    elif still_elevated:
        recommended_next_steps.append(
            f"Monitor {current} rather than escalating it as a spike; the corrected {name} moved "
            f"{_percent(corrected_change)} with no duplicate records removed."
        )
    else:
        recommended_next_steps.append(
            f"Do not escalate {current} as a spike; the corrected {name} is "
            f"{_percent(metric['corrected_current_rate'])} and fell {_abs_percent(corrected_change)}."
        )
    if count_drivers and material_increase:
        top = count_drivers[0]
        recommended_next_steps.append(
            f"Review {top['segment_name']} = {top['segment_value']} dispute handling with the "
            "operations owner to establish whether the increase reflects customer behaviour, a "
            "merchant issue, or a process change."
        )
    if interaction and material_increase:
        recommended_next_steps.append(
            f"Pull case-level detail for {interaction['segment_a_value']} disputes on "
            f"{interaction['segment_b_value']} before the next reporting cycle."
        )
    if quality["failing_checks"] or quality["warning_checks"]:
        recommended_next_steps.append(
            "Review the relevant open data-quality findings before using segment labels operationally."
        )
    recommended_next_steps.append(
        "Re-run this investigation after the next data load to confirm whether the movement persists."
    )

    references = [
        f"metric_engine.remediation_impact[{current}]",
        f"metric_engine.period_comparison[{previous}->{current}]",
    ]
    references += [
        f"quality_checks[{item['check_name']}]" for item in quality["failing_checks"][:3]
    ]
    references += [
        f"driver_analysis.top_count_drivers[{item['segment_name']}={item['segment_value']}]"
        for item in count_drivers[:2]
    ]
    if interaction:
        references.append(
            "driver_analysis.interaction_heatmap_data"
            f"[{interaction['segment_a_value']}x{interaction['segment_b_value']}]"
        )
    references += [
        f"text_theme_analysis.theme_summary[{item['theme_name']}]" for item in top_themes[:2]
    ]

    explanation = Explanation(
        headline=headline,
        executive_summary=executive_summary,
        what_changed=what_changed,
        data_quality_findings=data_quality_findings,
        business_drivers=business_drivers,
        customer_text_evidence=customer_text_evidence,
        recommended_next_steps=tuple(recommended_next_steps),
        limitations=tuple(facts["limitations"]),
        evidence_references=tuple(references),
        generated_by=GENERATOR_DETERMINISTIC,
        validation=NumberValidation(STATUS_PASS, (), (), (), (), "not yet validated"),
    )
    return _revalidate(explanation, packet)


def _revalidate(explanation: Explanation, packet: EvidencePacket) -> Explanation:
    validation = validate_numbers(explanation.narrative_text(), packet.supported_numbers)
    return Explanation(
        **{field_name: getattr(explanation, field_name) for field_name in EXPLANATION_FIELDS},
        validation=validation,
    )


# ---------------------------------------------------------------------------
# OpenAI path
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior data analyst at a large credit-card issuer, writing a short
investigation summary for an analytics manager.

You will receive a JSON evidence packet produced by a deterministic analytics
pipeline. It is your ONLY source of facts.

Hard rules:
1. Do NOT calculate anything. Every number you write must already appear in the
   evidence packet. Do not sum, average, difference, or re-express values.
2. Do NOT introduce any number that is not in the packet, including estimates,
   approximations, dollar amounts, or percentages you derived yourself.
3. Distinguish the reported (raw) metric from the corrected metric. If duplicate
   records were removed, say plainly how much of the movement was data quality
   and how much survives correction.
4. Describe drivers as where to investigate, not as proven causes.
5. Do not give credit policy guidance, financial advice, or customer-level
   decisions.
6. If the evidence is mixed or thin, say so rather than writing a confident
   story.

Write for a manager who has ninety seconds. Be concrete and plain.
"""

EXPLANATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "headline",
        "executive_summary",
        "what_changed",
        "data_quality_findings",
        "business_drivers",
        "customer_text_evidence",
        "recommended_next_steps",
        "limitations",
        "evidence_references",
    ],
    "properties": {
        "headline": {"type": "string"},
        "executive_summary": {"type": "string"},
        "what_changed": {"type": "string"},
        "data_quality_findings": {"type": "string"},
        "business_drivers": {"type": "string"},
        "customer_text_evidence": {"type": "string"},
        "recommended_next_steps": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "evidence_references": {"type": "array", "items": {"type": "string"}},
    },
}


def resolve_model(model: str | None = None) -> str:
    """Model id, configurable by environment variable."""
    return model or os.environ.get(DEFAULT_MODEL_ENV_VAR) or DEFAULT_MODEL


def _extract_output_text(response: Any) -> str:
    """Read the text payload off a Responses API result.

    ``output_text`` is the documented convenience accessor; the walk is a
    fallback so a client stub that only populates ``output`` still works.
    """
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        for piece in getattr(item, "content", None) or []:
            value = getattr(piece, "text", None)
            if isinstance(value, str):
                chunks.append(value)
    if not chunks:
        raise ValueError("model response contained no text output")
    return "".join(chunks)


def build_llm_explanation(
    packet: EvidencePacket,
    client: _ResponsesClient | None = None,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Explanation:
    """Generate the explanation with the OpenAI Responses API.

    ``client`` is injectable, which is how the tests exercise this path with a
    fake and no network access. The returned explanation is validated exactly
    like the deterministic one -- the model gets no special trust.
    """
    resolved_model = resolve_model(model)

    if client is None:
        from openai import OpenAI  # imported lazily so the module works without the package

        client = OpenAI()

    response = client.responses.create(
        model=resolved_model,
        temperature=temperature,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Evidence packet:\n"
                    f"{packet.to_json()}\n\n"
                    "Write the investigation summary. Use only numbers that appear above."
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "metricguard_explanation",
                "schema": EXPLANATION_JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    payload = json.loads(_extract_output_text(response))
    missing = [key for key in EXPLANATION_JSON_SCHEMA["required"] if key not in payload]
    if missing:
        raise ValueError(f"model response is missing required fields: {missing}")

    explanation = Explanation(
        headline=str(payload["headline"]),
        executive_summary=str(payload["executive_summary"]),
        what_changed=str(payload["what_changed"]),
        data_quality_findings=str(payload["data_quality_findings"]),
        business_drivers=str(payload["business_drivers"]),
        customer_text_evidence=str(payload["customer_text_evidence"]),
        recommended_next_steps=tuple(str(item) for item in payload["recommended_next_steps"]),
        limitations=tuple(str(item) for item in payload["limitations"]),
        evidence_references=tuple(str(item) for item in payload["evidence_references"]),
        generated_by=f"openai:{resolved_model}",
        validation=NumberValidation(STATUS_PASS, (), (), (), (), "not yet validated"),
    )
    return _revalidate(explanation, packet)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def explain(
    packet: EvidencePacket | None = None,
    client: _ResponsesClient | None = None,
    model: str | None = None,
    use_llm: bool | None = None,
    on_unsupported_numbers: str = "flag",
    temperature: float = DEFAULT_TEMPERATURE,
    tables: Mapping[str, pd.DataFrame] | None = None,
    data_dir: Path = DATA_DIR,
) -> Explanation:
    """Produce the manager-ready explanation.

    ``use_llm`` defaults to "yes if a client was injected or an API key is set".
    ``on_unsupported_numbers`` decides what happens when validation finds a
    number with no evidence behind it:

    ``flag``
        keep the explanation and report the failure on ``validation`` (default;
        the dashboard shows the badge and the reader decides)
    ``fallback``
        discard it and return the deterministic explanation instead
    ``raise``
        refuse to return an ungrounded explanation at all
    """
    if on_unsupported_numbers not in {"flag", "fallback", "raise"}:
        raise ValueError(
            f"on_unsupported_numbers must be flag, fallback or raise; got {on_unsupported_numbers!r}"
        )

    if packet is None:
        packet = build_evidence_packet(tables=tables, data_dir=data_dir)

    if use_llm is None:
        use_llm = client is not None or bool(os.environ.get(API_KEY_ENV_VAR))

    if not use_llm:
        return build_deterministic_explanation(packet)

    explanation = build_llm_explanation(packet, client=client, model=model, temperature=temperature)

    if explanation.validation.is_valid:
        return explanation
    if on_unsupported_numbers == "raise":
        raise ValueError(
            "generated explanation contains numbers absent from the evidence packet: "
            + ", ".join(explanation.validation.unsupported)
        )
    if on_unsupported_numbers == "fallback":
        return build_deterministic_explanation(packet)
    return explanation


if __name__ == "__main__":
    evidence = build_evidence_packet()
    result = explain(evidence)

    print(result.to_markdown())
    print("=" * 78)
    print(f"Evidence packet numbers whitelisted: {len(evidence.supported_numbers)}")
    print(f"Numbers checked in explanation:      {len(result.validation.checked)}")
    print(f"Unsupported numbers:                 {len(result.validation.unsupported)}")
