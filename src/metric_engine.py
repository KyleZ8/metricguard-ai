"""Metric calculation and duplicate remediation for MetricGuard AI.

This module owns the second step of the MetricGuard workflow::

    metric movement -> data quality check -> anomaly detection -> ...

``quality_checks.py`` answers "can we trust these records?". This module answers
the next question: "given what we found, what is the metric actually worth?" It
computes ``dispute_rate`` two ways -- as the dashboard reported it, and as it
stands once replayed source events are removed -- so the gap between them can be
quantified rather than asserted.

The MVP metric, matching ``metric_definitions.csv`` and ``DATA_DICTIONARY.md``::

    dispute_rate = disputed purchase transactions / total purchase transactions

Everything here is deterministic pandas. No LLM call, no randomness, no I/O
beyond reading the synthetic CSVs. The same tables always produce the same
frames, which is what makes the numbers safe to put in front of a manager.

Three dashboard-ready frames come out of :func:`build_metric_report`:

``monthly_trend``
    One row per month: raw and corrected rate side by side, plus the duplicate
    rows removed. This is the trend chart and the anomaly-detection input.
``period_comparison``
    One row per variant (raw, corrected) comparing a current and previous
    period. This is the "Metric Health Overview" panel.
``remediation_impact``
    One row quantifying what deduplication did to the current period. This is
    the "part data quality, part real movement" story in numbers.

Run directly to print all three::

    python src/metric_engine.py
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# quality_checks owns table loading and the canonical table names. Importing
# them keeps a single loader in the project rather than a third private copy.
from quality_checks import (
    DATA_DIR,
    TABLE_ACCOUNTS,
    TABLE_COMPLAINTS,
    TABLE_METRIC_DEFINITIONS,
    TABLE_SNAPSHOTS,
    TABLE_TRANSACTIONS,
    load_tables,
)

METRIC_NAME = "dispute_rate"
PURCHASE_TRANSACTION_TYPE = "purchase"

VARIANT_RAW = "raw"
VARIANT_CORRECTED = "corrected"

# Deduplication identifies a replayed upstream event by this key and breaks
# created_at ties on transaction_id so the result never depends on row order.
SOURCE_EVENT_KEY = "source_transaction_id"
INGESTION_TIMESTAMP = "created_at"
ROW_KEY = "transaction_id"

MONTHLY_TREND_COLUMNS = (
    "month",
    "purchase_transactions_raw",
    "disputed_purchases_raw",
    "dispute_rate_raw",
    "purchase_transactions_corrected",
    "disputed_purchases_corrected",
    "dispute_rate_corrected",
    "duplicate_purchase_rows_removed",
    "duplicate_disputed_rows_removed",
    "dispute_rate_difference",
)

PERIOD_COMPARISON_COLUMNS = (
    "metric_name",
    "variant",
    "current_period",
    "previous_period",
    "current_value",
    "previous_value",
    "absolute_change",
    "percent_change",
    "current_numerator",
    "current_denominator",
    "previous_numerator",
    "previous_denominator",
)

REMEDIATION_IMPACT_COLUMNS = (
    "metric_name",
    "period",
    "raw_dispute_rate",
    "corrected_dispute_rate",
    "dispute_rate_difference",
    "raw_disputed_count",
    "corrected_disputed_count",
    "duplicate_disputed_rows_removed",
    "raw_purchase_count",
    "corrected_purchase_count",
    "duplicate_purchase_rows_removed",
)

_COUNT_COLUMNS = (
    "purchase_transactions_raw",
    "disputed_purchases_raw",
    "purchase_transactions_corrected",
    "disputed_purchases_corrected",
    "duplicate_purchase_rows_removed",
    "duplicate_disputed_rows_removed",
)

# metricguard_engine.py keys the snapshot table as "snapshots".
_TABLE_ALIASES = {"snapshots": TABLE_SNAPSHOTS, "snapshot": TABLE_SNAPSHOTS}


@dataclass(frozen=True, eq=False)
class MetricReport:
    """The three dashboard frames plus the periods they describe.

    ``eq=False`` because the generated ``__eq__`` would compare DataFrames
    element-wise and raise on truth-value ambiguity.
    """

    metric_name: str
    current_period: str
    previous_period: str
    monthly_trend: pd.DataFrame
    period_comparison: pd.DataFrame
    remediation_impact: pd.DataFrame


@dataclass(frozen=True)
class FinanceMetricSpec:
    """Definition for one selectable finance KPI."""

    metric_name: str
    display_name: str
    business_definition: str
    numerator_label: str
    denominator_label: str
    metric_family: str
    higher_is_bad: bool
    segment_fields: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class FinanceMetricReport:
    """Generic report used by the dashboard KPI selector."""

    metric_name: str
    display_name: str
    period_grain: str
    current_period: str
    previous_period: str
    current_months: tuple[str, ...]
    previous_months: tuple[str, ...]
    monthly_trend: pd.DataFrame
    period_comparison: pd.DataFrame
    remediation_impact: pd.DataFrame
    segment_drivers: pd.DataFrame
    metric_definition: dict[str, object]


FINANCE_METRIC_SPECS: dict[str, FinanceMetricSpec] = {
    "dispute_rate": FinanceMetricSpec(
        metric_name="dispute_rate",
        display_name="Dispute rate",
        business_definition="Share of purchase transactions that generated a customer dispute.",
        numerator_label="disputed purchases",
        denominator_label="purchase transactions",
        metric_family="transaction_purchase",
        higher_is_bad=True,
        segment_fields=(
            "merchant_category",
            "channel",
            "customer_segment",
            "product_type",
            "fico_band",
            "region",
        ),
    ),
    "fraud_claim_rate": FinanceMetricSpec(
        metric_name="fraud_claim_rate",
        display_name="Fraud claim rate",
        business_definition="Share of purchase transactions associated with a fraud claim.",
        numerator_label="fraud-claimed purchases",
        denominator_label="purchase transactions",
        metric_family="transaction_purchase",
        higher_is_bad=True,
        segment_fields=("merchant_category", "channel", "product_type", "fico_band", "region"),
    ),
    "payment_failure_rate": FinanceMetricSpec(
        metric_name="payment_failure_rate",
        display_name="Payment failure rate",
        business_definition="Share of payment attempts that failed or were returned.",
        numerator_label="failed payments",
        denominator_label="payment attempts",
        metric_family="transaction_payment",
        higher_is_bad=True,
        segment_fields=("channel", "product_type", "fico_band", "customer_segment", "region"),
    ),
    "delinquency_rate_30dpd_balance": FinanceMetricSpec(
        metric_name="delinquency_rate_30dpd_balance",
        display_name="30+ DPD balance rate",
        business_definition="Balance-based 30+ day delinquency proxy for active credit-card accounts.",
        numerator_label="30+ DPD balance",
        denominator_label="statement balance",
        metric_family="snapshot_balance",
        higher_is_bad=True,
        segment_fields=("product_type", "fico_band", "customer_segment", "region"),
    ),
    "net_charge_off_rate_proxy": FinanceMetricSpec(
        metric_name="net_charge_off_rate_proxy",
        display_name="Net charge-off proxy",
        business_definition="Annualized synthetic charge-off balance divided by statement balance.",
        numerator_label="annualized charge-off balance",
        denominator_label="statement balance",
        metric_family="snapshot_balance",
        higher_is_bad=True,
        segment_fields=("product_type", "fico_band", "customer_segment", "region"),
    ),
    "complaint_rate": FinanceMetricSpec(
        metric_name="complaint_rate",
        display_name="Complaint rate",
        business_definition="Customer complaints per 1,000 active accounts (amendment A4).",
        numerator_label="complaints",
        denominator_label="active accounts (x1,000)",
        metric_family="complaint_volume",
        higher_is_bad=True,
        # Account-level segment fields only (A4): a complaint has no
        # independent "population" for a complaint-only attribute like
        # issue/submitted_via, so complaint_rate is not computed for those --
        # see generic_segment_driver_table's complaint_rate branch.
        segment_fields=(
            "product_type",
            "customer_segment",
            "fico_band",
            "region",
        ),
    ),
    "fee_complaint_share": FinanceMetricSpec(
        metric_name="fee_complaint_share",
        display_name="Fee complaint share",
        business_definition="Share of complaints related to fees or interest.",
        numerator_label="fee or interest complaints",
        denominator_label="complaints",
        metric_family="complaint_share",
        higher_is_bad=True,
        # "issue" excluded (A11): every complaint in the issue="Fees or
        # interest" group is definitionally a fee complaint, so that one cut
        # is tautological (1.0 for that value, 0.0 for every other issue).
        segment_fields=(
            "submitted_via",
            "product_type",
            "customer_segment",
            "fico_band",
            "region",
        ),
    ),
}

FINANCE_KPIS = tuple(FINANCE_METRIC_SPECS)

PERIOD_GRAIN_MONTHLY = "monthly"
PERIOD_GRAIN_QUARTERLY = "quarterly"
PERIOD_GRAIN_SEMIANNUAL = "semiannual"
PERIOD_GRAINS = (
    PERIOD_GRAIN_MONTHLY,
    PERIOD_GRAIN_QUARTERLY,
    PERIOD_GRAIN_SEMIANNUAL,
)
PERIOD_GRAIN_MONTHS = {
    PERIOD_GRAIN_MONTHLY: 1,
    PERIOD_GRAIN_QUARTERLY: 3,
    PERIOD_GRAIN_SEMIANNUAL: 6,
}

GENERIC_MONTHLY_TREND_COLUMNS = (
    "month",
    "period_grain",
    "start_month",
    "end_month",
    "period_months",
    "metric_name",
    "raw_numerator",
    "raw_denominator",
    "raw_value",
    "corrected_numerator",
    "corrected_denominator",
    "corrected_value",
    "duplicate_numerator_removed",
    "duplicate_denominator_removed",
    "value_difference",
)

GENERIC_REMEDIATION_COLUMNS = (
    "metric_name",
    "period",
    "raw_value",
    "corrected_value",
    "value_difference",
    "raw_numerator",
    "corrected_numerator",
    "duplicate_numerator_removed",
    "raw_denominator",
    "corrected_denominator",
    "duplicate_denominator_removed",
)

GENERIC_SEGMENT_DRIVER_COLUMNS = (
    "segment_name",
    "segment_value",
    "previous_numerator",
    "current_numerator",
    "numerator_change",
    "previous_denominator",
    "current_denominator",
    "denominator_change",
    "previous_value",
    "current_value",
    "absolute_change",
    "percent_change",
    "contribution_share_of_positive_change",
    "min_denominator_flag",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Accept either this project's or the older engine's table key naming."""
    return {_TABLE_ALIASES.get(name, name): frame for name, frame in tables.items()}


def _month_key(frame: pd.DataFrame, date_column: str = "transaction_date") -> pd.Series:
    """Return a ``YYYY-MM`` period key, tolerating unparsed date columns."""
    return pd.to_datetime(frame[date_column]).dt.to_period("M").astype(str)


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise rate that yields NaN rather than inf on a zero denominator."""
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    den = pd.to_numeric(denominator, errors="coerce").astype(float)
    return num.divide(den).where(den.gt(0))


def _safe_scalar_rate(numerator: float, denominator: float) -> float:
    if denominator is None or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _require_period(trend: pd.DataFrame, period: str, label: str) -> None:
    if period not in set(trend["month"]):
        available = ", ".join(trend["month"].tolist())
        raise KeyError(
            f"{label} {period!r} is not in the monthly trend. Available months: {available}"
        )


def _period_width(period_grain: str) -> int:
    try:
        return PERIOD_GRAIN_MONTHS[period_grain]
    except KeyError as error:
        raise ValueError(
            f"period_grain must be one of {list(PERIOD_GRAINS)}, got {period_grain!r}"
        ) from error


def _period_label(months: tuple[str, ...]) -> str:
    if len(months) == 1:
        return months[0]
    return f"{months[0]} to {months[-1]}"


def rolling_period_windows(months: tuple[str, ...] | list[str], period_grain: str) -> pd.DataFrame:
    """Return rolling reporting windows for a monthly series."""
    width = _period_width(period_grain)
    ordered_months = tuple(str(month) for month in months)
    rows: list[dict[str, object]] = []
    for end_index in range(width - 1, len(ordered_months)):
        window = ordered_months[end_index - width + 1 : end_index + 1]
        rows.append(
            {
                "month": _period_label(window),
                "period_grain": period_grain,
                "start_month": window[0],
                "end_month": window[-1],
                "period_months": tuple(window),
            }
        )
    return pd.DataFrame(
        rows, columns=["month", "period_grain", "start_month", "end_month", "period_months"]
    )


def period_months_from_trend(trend: pd.DataFrame, period: str) -> tuple[str, ...]:
    """Months contained in a selected period label."""
    _require_period(trend, period, "period")
    row = trend[trend["month"].eq(period)].iloc[0]
    months = row.get("period_months", (period,))
    if isinstance(months, tuple):
        return months
    if isinstance(months, list):
        return tuple(str(month) for month in months)
    return (str(months),)


def available_finance_kpis() -> tuple[str, ...]:
    """KPI names available in the finance dashboard."""
    return FINANCE_KPIS


def metric_spec(metric_name: str) -> FinanceMetricSpec:
    """Return the configured KPI spec, with a clear error for unknown names."""
    try:
        return FINANCE_METRIC_SPECS[metric_name]
    except KeyError as error:
        raise KeyError(
            f"unknown metric {metric_name!r}; expected one of {list(FINANCE_KPIS)}"
        ) from error


def _metric_definition(
    tables: Mapping[str, pd.DataFrame], spec: FinanceMetricSpec
) -> dict[str, object]:
    definitions = tables.get(TABLE_METRIC_DEFINITIONS)
    if definitions is None or "metric_name" not in definitions.columns:
        return {
            "metric_name": spec.metric_name,
            "business_definition": spec.business_definition,
            "numerator": spec.numerator_label,
            "denominator": spec.denominator_label,
        }
    rows = definitions[definitions["metric_name"].eq(spec.metric_name)]
    if rows.empty:
        return {
            "metric_name": spec.metric_name,
            "business_definition": spec.business_definition,
            "numerator": spec.numerator_label,
            "denominator": spec.denominator_label,
        }
    return {str(key): _json_scalar(value) for key, value in rows.iloc[0].to_dict().items()}


def _json_scalar(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _generic_group(
    frame: pd.DataFrame,
    group_columns: list[str],
    numerator: str | Callable[[pd.DataFrame], pd.Series],
    denominator: str | Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    """Aggregate numerator/denominator by month plus optional segments."""
    if frame.empty:
        return pd.DataFrame(columns=group_columns + ["numerator", "denominator", "metric_value"])

    working = frame.copy()
    if callable(numerator):
        working["_metric_numerator"] = numerator(working)
    else:
        working["_metric_numerator"] = working[numerator]

    if denominator is None:
        working["_metric_denominator"] = 1
    elif callable(denominator):
        working["_metric_denominator"] = denominator(working)
    else:
        working["_metric_denominator"] = working[denominator]

    grouped = (
        working.groupby(group_columns, dropna=False)
        .agg(numerator=("_metric_numerator", "sum"), denominator=("_metric_denominator", "sum"))
        .reset_index()
    )
    grouped["metric_value"] = _safe_rate(grouped["numerator"], grouped["denominator"])
    return grouped


def _transaction_metric_source(
    tables: Mapping[str, pd.DataFrame],
    spec: FinanceMetricSpec,
    corrected: bool,
) -> tuple[pd.DataFrame, str | Callable[[pd.DataFrame], pd.Series], None]:
    tx = tables[TABLE_TRANSACTIONS]
    tx = deduplicate_transactions(tx) if corrected else tx.copy()
    if spec.metric_family == "transaction_purchase":
        tx = tx[tx["transaction_type"].eq(PURCHASE_TRANSACTION_TYPE)].copy()
        numerator = "is_disputed" if spec.metric_name == "dispute_rate" else "is_fraud_claim"
    elif spec.metric_family == "transaction_payment":
        tx = tx[tx["transaction_type"].eq("payment")].copy()
        numerator = "payment_failed"
    else:
        raise ValueError(f"{spec.metric_name} is not a transaction metric")

    tx["month"] = _month_key(tx, "transaction_date")
    accounts = tables.get(TABLE_ACCOUNTS)
    if accounts is not None:
        join_fields = [
            field
            for field in ("fico_band", "customer_segment", "product_type", "region")
            if field in accounts.columns and field not in tx.columns
        ]
        if join_fields:
            tx = tx.merge(accounts[["account_id"] + join_fields], on="account_id", how="left")
    return tx, numerator, None


def _snapshot_metric_source(
    tables: Mapping[str, pd.DataFrame],
    spec: FinanceMetricSpec,
) -> tuple[pd.DataFrame, Callable[[pd.DataFrame], pd.Series], str]:
    snapshots = tables[TABLE_SNAPSHOTS].copy()
    snapshots = snapshots[snapshots["active_flag"].eq(1)].copy()
    snapshots["month"] = snapshots["snapshot_month"].astype(str)
    accounts = tables.get(TABLE_ACCOUNTS)
    if accounts is not None:
        join_fields = [
            field
            for field in ("fico_band", "customer_segment", "product_type", "region")
            if field in accounts.columns and field not in snapshots.columns
        ]
        if join_fields:
            snapshots = snapshots.merge(
                accounts[["account_id"] + join_fields], on="account_id", how="left"
            )

    if spec.metric_name == "delinquency_rate_30dpd_balance":
        def numerator(frame: pd.DataFrame) -> pd.Series:
            return frame["statement_balance"] * frame["is_30dpd"]

    elif spec.metric_name == "net_charge_off_rate_proxy":
        def numerator(frame: pd.DataFrame) -> pd.Series:
            return frame["charge_off_balance"] * 12.0

    else:
        raise ValueError(f"{spec.metric_name} is not a snapshot metric")
    return snapshots, numerator, "statement_balance"


def _complaint_monthly_trend(
    tables: Mapping[str, pd.DataFrame],
    spec: FinanceMetricSpec,
) -> pd.DataFrame:
    complaints = tables[TABLE_COMPLAINTS].copy()
    complaints["month"] = _month_key(complaints, "date_received")

    if spec.metric_name == "fee_complaint_share":
        grouped = _generic_group(
            complaints,
            ["month"],
            lambda frame: frame["issue"].eq("Fees or interest").astype(int),
            None,
        )
        return grouped

    comp_count = complaints.groupby("month").agg(numerator=("complaint_id", "count"))
    snapshots = tables[TABLE_SNAPSHOTS]
    active = (
        snapshots[snapshots["active_flag"].eq(1)]
        .groupby("snapshot_month")
        .agg(denominator=("account_id", "nunique"))
    )
    grouped = comp_count.join(active, how="outer").fillna(0).reset_index(names="month")
    # Per 1,000 active accounts (amendment A4), not a raw complaints/accounts
    # fraction -- complaint counts are small relative to the account base, so
    # an unscaled rate reads as a string of zeros.
    grouped["metric_value"] = _safe_rate(grouped["numerator"], grouped["denominator"]) * 1000
    return grouped[["month", "numerator", "denominator", "metric_value"]]


def _complaint_segment_source(
    tables: Mapping[str, pd.DataFrame],
    spec: FinanceMetricSpec,
) -> tuple[pd.DataFrame, str | Callable[[pd.DataFrame], pd.Series], None]:
    complaints = tables[TABLE_COMPLAINTS].copy()
    complaints["month"] = _month_key(complaints, "date_received")
    if spec.metric_name == "fee_complaint_share":
        def numerator(frame: pd.DataFrame) -> pd.Series:
            return frame["issue"].eq("Fees or interest").astype(int)

    else:
        def numerator(frame: pd.DataFrame) -> pd.Series:
            return pd.Series(1, index=frame.index)

    return complaints, numerator, None


def _metric_source(
    tables: Mapping[str, pd.DataFrame],
    spec: FinanceMetricSpec,
    corrected: bool = True,
) -> tuple[
    pd.DataFrame,
    str | Callable[[pd.DataFrame], pd.Series],
    str | Callable[[pd.DataFrame], pd.Series] | None,
]:
    if spec.metric_family.startswith("transaction"):
        return _transaction_metric_source(tables, spec, corrected)
    if spec.metric_family.startswith("snapshot"):
        return _snapshot_metric_source(tables, spec)
    return _complaint_segment_source(tables, spec)


def generic_monthly_metric(
    tables: Mapping[str, pd.DataFrame],
    metric_name: str,
    corrected: bool = True,
) -> pd.DataFrame:
    """Monthly KPI series for any finance metric."""
    spec = metric_spec(metric_name)
    if spec.metric_family.startswith("complaint"):
        grouped = _complaint_monthly_trend(tables, spec)
    else:
        source, numerator, denominator = _metric_source(tables, spec, corrected)
        grouped = _generic_group(source, ["month"], numerator, denominator)
    grouped = grouped.sort_values("month").reset_index(drop=True)
    grouped["metric_name"] = spec.metric_name
    grouped["variant"] = VARIANT_CORRECTED if corrected else VARIANT_RAW
    return grouped[["month", "metric_name", "variant", "numerator", "denominator", "metric_value"]]


def generic_monthly_trend_table(
    tables: Mapping[str, pd.DataFrame],
    metric_name: str,
) -> pd.DataFrame:
    """Raw/corrected monthly trend for any finance KPI.

    Deduplication is meaningful for transaction metrics. For snapshot and
    complaint metrics, raw and corrected are intentionally identical.
    """
    spec = metric_spec(metric_name)
    raw = generic_monthly_metric(tables, metric_name, corrected=False)
    corrected = (
        generic_monthly_metric(tables, metric_name, corrected=True)
        if spec.metric_family.startswith("transaction")
        else raw.assign(variant=VARIANT_CORRECTED)
    )

    trend = raw.merge(
        corrected,
        on=["month", "metric_name"],
        how="outer",
        suffixes=("_raw", "_corrected"),
    ).sort_values("month")
    for column in (
        "numerator_raw",
        "denominator_raw",
        "numerator_corrected",
        "denominator_corrected",
    ):
        trend[column] = trend[column].fillna(0)
    trend["raw_value"] = _safe_rate(trend["numerator_raw"], trend["denominator_raw"])
    trend["corrected_value"] = _safe_rate(
        trend["numerator_corrected"], trend["denominator_corrected"]
    )
    trend["duplicate_numerator_removed"] = trend["numerator_raw"] - trend["numerator_corrected"]
    trend["duplicate_denominator_removed"] = (
        trend["denominator_raw"] - trend["denominator_corrected"]
    )
    trend["value_difference"] = trend["raw_value"] - trend["corrected_value"]
    trend["period_grain"] = PERIOD_GRAIN_MONTHLY
    trend["start_month"] = trend["month"]
    trend["end_month"] = trend["month"]
    trend["period_months"] = trend["month"].map(lambda month: (str(month),))
    return trend.rename(
        columns={
            "numerator_raw": "raw_numerator",
            "denominator_raw": "raw_denominator",
            "numerator_corrected": "corrected_numerator",
            "denominator_corrected": "corrected_denominator",
        }
    )[list(GENERIC_MONTHLY_TREND_COLUMNS)].reset_index(drop=True)


def aggregate_metric_trend(
    monthly_trend: pd.DataFrame,
    period_grain: str = PERIOD_GRAIN_MONTHLY,
) -> pd.DataFrame:
    """Aggregate a monthly KPI trend into rolling reporting windows."""
    _period_width(period_grain)
    monthly = monthly_trend.sort_values("month").reset_index(drop=True)
    if period_grain == PERIOD_GRAIN_MONTHLY:
        return monthly.copy()

    windows = rolling_period_windows(tuple(monthly["month"].astype(str)), period_grain)
    rows: list[dict[str, object]] = []
    for _, window in windows.iterrows():
        months = tuple(window["period_months"])
        selected = monthly[monthly["month"].isin(months)]
        raw_numerator = float(selected["raw_numerator"].sum())
        raw_denominator = float(selected["raw_denominator"].sum())
        corrected_numerator = float(selected["corrected_numerator"].sum())
        corrected_denominator = float(selected["corrected_denominator"].sum())
        raw_value = _safe_scalar_rate(raw_numerator, raw_denominator)
        corrected_value = _safe_scalar_rate(corrected_numerator, corrected_denominator)
        rows.append(
            {
                "month": window["month"],
                "period_grain": period_grain,
                "start_month": window["start_month"],
                "end_month": window["end_month"],
                "period_months": months,
                "metric_name": selected["metric_name"].iloc[0],
                "raw_numerator": raw_numerator,
                "raw_denominator": raw_denominator,
                "raw_value": raw_value,
                "corrected_numerator": corrected_numerator,
                "corrected_denominator": corrected_denominator,
                "corrected_value": corrected_value,
                "duplicate_numerator_removed": raw_numerator - corrected_numerator,
                "duplicate_denominator_removed": raw_denominator - corrected_denominator,
                "value_difference": raw_value - corrected_value,
            }
        )
    return pd.DataFrame(rows, columns=list(GENERIC_MONTHLY_TREND_COLUMNS))


def generic_period_comparison_table(
    trend: pd.DataFrame,
    metric_name: str,
    current_period: str | None = None,
    previous_period: str | None = None,
) -> pd.DataFrame:
    """Generic raw/corrected period comparison."""
    resolved_current, resolved_previous = resolve_periods(trend, current_period, previous_period)
    indexed = trend.set_index("month")
    rows: list[dict[str, object]] = []
    for variant in (VARIANT_RAW, VARIANT_CORRECTED):
        current = indexed.loc[resolved_current]
        previous = indexed.loc[resolved_previous]
        current_value = float(current[f"{variant}_value"])
        previous_value = float(previous[f"{variant}_value"])
        absolute_change = current_value - previous_value
        rows.append(
            {
                "metric_name": metric_name,
                "variant": variant,
                "current_period": resolved_current,
                "previous_period": resolved_previous,
                "current_value": current_value,
                "previous_value": previous_value,
                "absolute_change": absolute_change,
                "percent_change": _safe_scalar_rate(absolute_change, previous_value),
                "current_numerator": float(current[f"{variant}_numerator"]),
                "current_denominator": float(current[f"{variant}_denominator"]),
                "previous_numerator": float(previous[f"{variant}_numerator"]),
                "previous_denominator": float(previous[f"{variant}_denominator"]),
            }
        )
    return pd.DataFrame(rows, columns=list(PERIOD_COMPARISON_COLUMNS))


def generic_remediation_impact_table(
    trend: pd.DataFrame,
    metric_name: str,
    period: str | None = None,
) -> pd.DataFrame:
    """Generic raw/corrected impact row for one period."""
    resolved_period = period if period is not None else trend["month"].iloc[-1]
    _require_period(trend, resolved_period, "period")
    row = trend.set_index("month").loc[resolved_period]
    return pd.DataFrame(
        [
            {
                "metric_name": metric_name,
                "period": resolved_period,
                "raw_value": float(row["raw_value"]),
                "corrected_value": float(row["corrected_value"]),
                "value_difference": float(row["value_difference"]),
                "raw_numerator": float(row["raw_numerator"]),
                "corrected_numerator": float(row["corrected_numerator"]),
                "duplicate_numerator_removed": float(row["duplicate_numerator_removed"]),
                "raw_denominator": float(row["raw_denominator"]),
                "corrected_denominator": float(row["corrected_denominator"]),
                "duplicate_denominator_removed": float(row["duplicate_denominator_removed"]),
            }
        ],
        columns=list(GENERIC_REMEDIATION_COLUMNS),
    )


def generic_segment_driver_table(
    tables: Mapping[str, pd.DataFrame],
    metric_name: str,
    current_period: str | None = None,
    previous_period: str | None = None,
    period_grain: str = PERIOD_GRAIN_MONTHLY,
    min_denominator: int = 100,
) -> pd.DataFrame:
    """Segment-level movement for any selected KPI, using corrected inputs."""
    spec = metric_spec(metric_name)
    trend = aggregate_metric_trend(generic_monthly_trend_table(tables, metric_name), period_grain)
    resolved_current, resolved_previous = resolve_periods(trend, current_period, previous_period)
    current_months = period_months_from_trend(trend, resolved_current)
    previous_months = period_months_from_trend(trend, resolved_previous)

    def _finish(grouped: pd.DataFrame, field: str, value_scale: float = 1.0) -> pd.DataFrame | None:
        """[field, _metric_period, numerator, denominator] -> the standard output columns."""
        if grouped.empty:
            return None
        pivot = grouped.set_index([field, "_metric_period"])[["numerator", "denominator"]].unstack(
            "_metric_period"
        )
        for measure in ("numerator", "denominator"):
            for period in (resolved_previous, resolved_current):
                if (measure, period) not in pivot.columns:
                    pivot[(measure, period)] = 0.0
        pivot = pivot.fillna(0)
        out = pd.DataFrame(
            {
                "segment_value": pivot.index,
                "previous_numerator": pivot[("numerator", resolved_previous)].astype(float),
                "current_numerator": pivot[("numerator", resolved_current)].astype(float),
                "previous_denominator": pivot[("denominator", resolved_previous)].astype(float),
                "current_denominator": pivot[("denominator", resolved_current)].astype(float),
            }
        ).reset_index(drop=True)
        out["segment_name"] = field
        out["numerator_change"] = out["current_numerator"] - out["previous_numerator"]
        out["denominator_change"] = out["current_denominator"] - out["previous_denominator"]
        out["previous_value"] = (
            _safe_rate(out["previous_numerator"], out["previous_denominator"]) * value_scale
        )
        out["current_value"] = (
            _safe_rate(out["current_numerator"], out["current_denominator"]) * value_scale
        )
        out["absolute_change"] = out["current_value"] - out["previous_value"]
        out["percent_change"] = out.apply(
            lambda row: _safe_scalar_rate(row["absolute_change"], row["previous_value"]),
            axis=1,
        )
        positive_total = out.loc[out["numerator_change"] > 0, "numerator_change"].sum()
        out["contribution_share_of_positive_change"] = (
            out["numerator_change"].where(out["numerator_change"] > 0, 0.0) / positive_total
            if positive_total > 0
            else 0.0
        )
        out["min_denominator_flag"] = (
            out[["previous_denominator", "current_denominator"]].min(axis=1) < min_denominator
        )
        return out

    frames: list[pd.DataFrame] = []
    if spec.metric_name == "complaint_rate":
        # complaint_rate's denominator is active accounts, not complaints
        # (amendment A4), so it cannot come from the same source frame as the
        # numerator the way every other KPI's does. Build each side from its
        # own table, keyed by segment value and period, then join them.
        complaints = tables[TABLE_COMPLAINTS].copy()
        complaints["month"] = _month_key(complaints, "date_received")
        snapshots = tables[TABLE_SNAPSHOTS].copy()
        snapshots["month"] = snapshots["snapshot_month"].astype(str)
        active_accounts = snapshots[snapshots["active_flag"].eq(1)].copy()
        accounts = tables.get(TABLE_ACCOUNTS)
        if accounts is not None:
            # complaints already carries customer_segment/fico_band natively
            # (copied at generation time); only join fields it's missing, to
            # avoid a _x/_y suffix collision that would hide both copies.
            complaint_join_fields = [
                field
                for field in spec.segment_fields
                if field in accounts.columns and field not in complaints.columns
            ]
            if complaint_join_fields:
                complaints = complaints.merge(
                    accounts[["account_id"] + complaint_join_fields], on="account_id", how="left"
                )
            snapshot_join_fields = [
                field
                for field in spec.segment_fields
                if field in accounts.columns and field not in active_accounts.columns
            ]
            if snapshot_join_fields:
                active_accounts = active_accounts.merge(
                    accounts[["account_id"] + snapshot_join_fields], on="account_id", how="left"
                )
        period_complaints = pd.concat(
            [
                complaints[complaints["month"].isin(previous_months)].assign(
                    _metric_period=resolved_previous
                ),
                complaints[complaints["month"].isin(current_months)].assign(
                    _metric_period=resolved_current
                ),
            ],
            ignore_index=True,
        )
        period_accounts = pd.concat(
            [
                active_accounts[active_accounts["month"].isin(previous_months)].assign(
                    _metric_period=resolved_previous
                ),
                active_accounts[active_accounts["month"].isin(current_months)].assign(
                    _metric_period=resolved_current
                ),
            ],
            ignore_index=True,
        )
        for field in spec.segment_fields:
            if field not in complaints.columns or field not in period_accounts.columns:
                continue
            numerator_counts = (
                period_complaints.groupby([field, "_metric_period"]).size().rename("numerator")
            )
            denominator_counts = (
                period_accounts.groupby([field, "_metric_period"])["account_id"]
                .nunique()
                .rename("denominator")
            )
            grouped = (
                pd.concat([numerator_counts, denominator_counts], axis=1).fillna(0).reset_index()
            )
            out = _finish(grouped, field, value_scale=1000.0)
            if out is not None:
                frames.append(out)
    else:
        source, numerator, denominator = _metric_source(tables, spec, corrected=True)
        period_source = pd.concat(
            [
                source[source["month"].isin(previous_months)].assign(_metric_period=resolved_previous),
                source[source["month"].isin(current_months)].assign(_metric_period=resolved_current),
            ],
            ignore_index=True,
        )
        for field in spec.segment_fields:
            if field not in period_source.columns:
                continue
            grouped = _generic_group(period_source, [field, "_metric_period"], numerator, denominator)
            out = _finish(grouped, field)
            if out is not None:
                frames.append(out)

    if not frames:
        return pd.DataFrame(columns=list(GENERIC_SEGMENT_DRIVER_COLUMNS))
    return (
        pd.concat(frames, ignore_index=True)[list(GENERIC_SEGMENT_DRIVER_COLUMNS)]
        .sort_values(
            ["numerator_change", "absolute_change", "segment_name"], ascending=[False, False, True]
        )
        .reset_index(drop=True)
    )


def build_finance_metric_report(
    metric_name: str = METRIC_NAME,
    tables: Mapping[str, pd.DataFrame] | None = None,
    current_period: str | None = None,
    previous_period: str | None = None,
    period_grain: str = PERIOD_GRAIN_MONTHLY,
    data_dir: Path = DATA_DIR,
) -> FinanceMetricReport:
    """Build the generic dashboard report for any finance KPI."""
    resolved_tables = resolve_tables(tables, data_dir)
    spec = metric_spec(metric_name)
    trend = aggregate_metric_trend(
        generic_monthly_trend_table(resolved_tables, metric_name),
        period_grain,
    )
    resolved_current, resolved_previous = resolve_periods(trend, current_period, previous_period)
    return FinanceMetricReport(
        metric_name=spec.metric_name,
        display_name=spec.display_name,
        period_grain=period_grain,
        current_period=resolved_current,
        previous_period=resolved_previous,
        current_months=period_months_from_trend(trend, resolved_current),
        previous_months=period_months_from_trend(trend, resolved_previous),
        monthly_trend=trend,
        period_comparison=generic_period_comparison_table(
            trend, metric_name, resolved_current, resolved_previous
        ),
        remediation_impact=generic_remediation_impact_table(trend, metric_name, resolved_current),
        segment_drivers=generic_segment_driver_table(
            resolved_tables, metric_name, resolved_current, resolved_previous, period_grain
        ),
        metric_definition=_metric_definition(resolved_tables, spec),
    )


# ---------------------------------------------------------------------------
# 1. Load or accept the synthetic tables
# ---------------------------------------------------------------------------


def resolve_tables(
    tables: Mapping[str, pd.DataFrame] | None = None, data_dir: Path = DATA_DIR
) -> dict[str, pd.DataFrame]:
    """Return the synthetic tables, loading them only when none were supplied.

    Accepting a caller-supplied mapping is what lets the dashboard load once and
    lets tests inject small fixtures without touching disk.
    """
    if tables is None:
        return load_tables(data_dir)
    return _normalize_tables(tables)


def transactions_from(
    tables: Mapping[str, pd.DataFrame] | None = None, data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """Convenience accessor for the one table every metric here needs."""
    return resolve_tables(tables, data_dir)[TABLE_TRANSACTIONS]


# ---------------------------------------------------------------------------
# 3. Deduplicate replayed source events
# ---------------------------------------------------------------------------


def deduplicate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per ``source_transaction_id``: the earliest ``created_at``.

    Ties on ``created_at`` are broken by ``transaction_id`` so the result is
    independent of input row order, and surviving rows are returned in their
    original order so downstream frames stay stable.

    Rows with a null ``source_transaction_id`` are all kept: without the upstream
    key there is no evidence they are replays, and silently collapsing them would
    understate the denominator.

    A lineage caveat specific to this dataset: the replay batch carries an
    earlier ``created_at`` than many of the original rows, so "earliest wins"
    often retains the replayed row rather than the original. That does not move
    ``dispute_rate``, because both rows of every duplicate pair agree on
    transaction_date, transaction_type and is_disputed -- but it does mean the
    surviving row's ingestion_batch_id should not be treated as authoritative
    lineage.
    """
    if SOURCE_EVENT_KEY not in transactions.columns:
        raise KeyError(f"transactions must contain {SOURCE_EVENT_KEY!r} to deduplicate")

    identified = transactions[transactions[SOURCE_EVENT_KEY].notna()]

    sort_columns = [SOURCE_EVENT_KEY]
    if INGESTION_TIMESTAMP in transactions.columns:
        sort_columns.append(INGESTION_TIMESTAMP)
    if ROW_KEY in transactions.columns:
        sort_columns.append(ROW_KEY)

    winners = (
        identified.sort_values(sort_columns, kind="mergesort")
        .drop_duplicates(SOURCE_EVENT_KEY, keep="first")
        .index
    )

    keep = transactions[SOURCE_EVENT_KEY].isna() | transactions.index.isin(winners)
    return transactions[keep]


def duplicate_source_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """The rows deduplication removes, for the dashboard's evidence table."""
    kept = deduplicate_transactions(transactions).index
    return transactions[~transactions.index.isin(kept)]


# ---------------------------------------------------------------------------
# 2 and 4. Raw and corrected monthly dispute rate
# ---------------------------------------------------------------------------


def monthly_dispute_rate(transactions: pd.DataFrame, deduped: bool = False) -> pd.DataFrame:
    """Monthly ``dispute_rate`` = disputed purchases / purchase transactions.

    Set ``deduped=True`` for the corrected series. Months are keyed on
    ``transaction_date``, i.e. when the transaction happened, not when it loaded.
    """
    source = deduplicate_transactions(transactions) if deduped else transactions
    purchases = source[source["transaction_type"].eq(PURCHASE_TRANSACTION_TYPE)]

    variant = VARIANT_CORRECTED if deduped else VARIANT_RAW
    if purchases.empty:
        empty = pd.DataFrame(
            columns=["month", "disputed_purchases", "purchase_transactions", "dispute_rate"]
        )
        empty["variant"] = pd.Series(dtype=object)
        empty["metric_name"] = pd.Series(dtype=object)
        return empty

    grouped = (
        purchases.groupby(_month_key(purchases))["is_disputed"]
        .agg(disputed_purchases="sum", purchase_transactions="size")
        .sort_index()
        .reset_index(names="month")
    )
    grouped["disputed_purchases"] = grouped["disputed_purchases"].astype(int)
    grouped["purchase_transactions"] = grouped["purchase_transactions"].astype(int)
    grouped["dispute_rate"] = _safe_rate(
        grouped["disputed_purchases"], grouped["purchase_transactions"]
    )
    grouped["variant"] = variant
    grouped["metric_name"] = METRIC_NAME
    return grouped


# ---------------------------------------------------------------------------
# 7a. Monthly trend table
# ---------------------------------------------------------------------------


def monthly_trend_table(transactions: pd.DataFrame) -> pd.DataFrame:
    """Raw and corrected dispute rate per month, with the duplicates removed."""
    raw = monthly_dispute_rate(transactions, deduped=False)
    corrected = monthly_dispute_rate(transactions, deduped=True)

    trend = raw.merge(
        corrected,
        on="month",
        how="outer",
        suffixes=("_raw", "_corrected"),
    ).sort_values("month")

    for column in (
        "disputed_purchases_raw",
        "purchase_transactions_raw",
        "disputed_purchases_corrected",
        "purchase_transactions_corrected",
    ):
        trend[column] = trend[column].fillna(0).astype(int)

    trend["duplicate_purchase_rows_removed"] = (
        trend["purchase_transactions_raw"] - trend["purchase_transactions_corrected"]
    )
    trend["duplicate_disputed_rows_removed"] = (
        trend["disputed_purchases_raw"] - trend["disputed_purchases_corrected"]
    )
    trend["dispute_rate_raw"] = _safe_rate(
        trend["disputed_purchases_raw"], trend["purchase_transactions_raw"]
    )
    trend["dispute_rate_corrected"] = _safe_rate(
        trend["disputed_purchases_corrected"], trend["purchase_transactions_corrected"]
    )
    trend["dispute_rate_difference"] = trend["dispute_rate_raw"] - trend["dispute_rate_corrected"]

    return trend[list(MONTHLY_TREND_COLUMNS)].reset_index(drop=True)


def monthly_trend_table_sql(tables: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Same shape as :func:`monthly_trend_table`, computed by the sql/ DuckDB layer (CC4).

    Not the default path -- :func:`monthly_trend_table` (pandas) is what the
    rest of the app calls -- but a real, callable integration point, proven
    identical to it by src/verify_sql_parity.py. Exists so the SQL layer is
    something the metric engine actually calls, not just a side script.
    """
    from sql_engine import build_connection, monthly_dispute_rate_sql

    con = build_connection(tables)
    try:
        raw = monthly_dispute_rate_sql(con, deduped=False)
        corrected = monthly_dispute_rate_sql(con, deduped=True)
    finally:
        con.close()

    trend = raw.merge(corrected, on="month", how="outer", suffixes=("_raw", "_corrected")).sort_values(
        "month"
    )
    trend = trend.rename(
        columns={
            "numerator_raw": "disputed_purchases_raw",
            "denominator_raw": "purchase_transactions_raw",
            "metric_value_raw": "dispute_rate_raw",
            "numerator_corrected": "disputed_purchases_corrected",
            "denominator_corrected": "purchase_transactions_corrected",
            "metric_value_corrected": "dispute_rate_corrected",
        }
    )
    for column in (
        "disputed_purchases_raw",
        "purchase_transactions_raw",
        "disputed_purchases_corrected",
        "purchase_transactions_corrected",
    ):
        trend[column] = trend[column].fillna(0).astype(int)
    trend["duplicate_purchase_rows_removed"] = (
        trend["purchase_transactions_raw"] - trend["purchase_transactions_corrected"]
    )
    trend["duplicate_disputed_rows_removed"] = (
        trend["disputed_purchases_raw"] - trend["disputed_purchases_corrected"]
    )
    trend["dispute_rate_difference"] = trend["dispute_rate_raw"] - trend["dispute_rate_corrected"]
    return trend[list(MONTHLY_TREND_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5 and 7b. Period comparison table
# ---------------------------------------------------------------------------


def resolve_periods(
    trend: pd.DataFrame,
    current_period: str | None = None,
    previous_period: str | None = None,
) -> tuple[str, str]:
    """Default to the two most recent months present in the trend."""
    months = trend["month"].tolist()
    if len(months) < 2:
        raise ValueError(f"need at least two months to compare, got {len(months)}")

    resolved_current = current_period if current_period is not None else months[-1]
    _require_period(trend, resolved_current, "current_period")

    if previous_period is not None:
        resolved_previous = previous_period
    else:
        position = months.index(resolved_current)
        if position == 0:
            raise ValueError(
                f"current_period {resolved_current!r} is the earliest month, so it has no prior period"
            )
        resolved_previous = months[position - 1]
    _require_period(trend, resolved_previous, "previous_period")

    return resolved_current, resolved_previous


def period_comparison_table(
    trend: pd.DataFrame,
    current_period: str | None = None,
    previous_period: str | None = None,
) -> pd.DataFrame:
    """Compare two periods for both the raw and the corrected series.

    ``percent_change`` is a fraction, not a display percentage, and is NaN when
    the previous value is zero or missing.
    """
    resolved_current, resolved_previous = resolve_periods(trend, current_period, previous_period)
    indexed = trend.set_index("month")

    rows: list[dict[str, object]] = []
    for variant, rate_column, numerator_column, denominator_column in (
        (VARIANT_RAW, "dispute_rate_raw", "disputed_purchases_raw", "purchase_transactions_raw"),
        (
            VARIANT_CORRECTED,
            "dispute_rate_corrected",
            "disputed_purchases_corrected",
            "purchase_transactions_corrected",
        ),
    ):
        current = indexed.loc[resolved_current]
        previous = indexed.loc[resolved_previous]
        current_value = float(current[rate_column])
        previous_value = float(previous[rate_column])
        absolute_change = current_value - previous_value

        rows.append(
            {
                "metric_name": METRIC_NAME,
                "variant": variant,
                "current_period": resolved_current,
                "previous_period": resolved_previous,
                "current_value": current_value,
                "previous_value": previous_value,
                "absolute_change": absolute_change,
                "percent_change": _safe_scalar_rate(absolute_change, previous_value),
                "current_numerator": int(current[numerator_column]),
                "current_denominator": int(current[denominator_column]),
                "previous_numerator": int(previous[numerator_column]),
                "previous_denominator": int(previous[denominator_column]),
            }
        )

    return pd.DataFrame(rows, columns=list(PERIOD_COMPARISON_COLUMNS))


# ---------------------------------------------------------------------------
# 6 and 7c. Remediation impact table
# ---------------------------------------------------------------------------


def remediation_impact_table(trend: pd.DataFrame, period: str | None = None) -> pd.DataFrame:
    """Quantify what deduplication did to one period's dispute rate.

    This is the number that separates "the dashboard was wrong" from "the
    business moved": how much of the reported rate was duplicate rows, and how
    much survives correction.
    """
    resolved_period = period if period is not None else trend["month"].iloc[-1]
    _require_period(trend, resolved_period, "period")
    row = trend.set_index("month").loc[resolved_period]

    return pd.DataFrame(
        [
            {
                "metric_name": METRIC_NAME,
                "period": resolved_period,
                "raw_dispute_rate": float(row["dispute_rate_raw"]),
                "corrected_dispute_rate": float(row["dispute_rate_corrected"]),
                "dispute_rate_difference": float(row["dispute_rate_difference"]),
                "raw_disputed_count": int(row["disputed_purchases_raw"]),
                "corrected_disputed_count": int(row["disputed_purchases_corrected"]),
                "duplicate_disputed_rows_removed": int(row["duplicate_disputed_rows_removed"]),
                "raw_purchase_count": int(row["purchase_transactions_raw"]),
                "corrected_purchase_count": int(row["purchase_transactions_corrected"]),
                "duplicate_purchase_rows_removed": int(row["duplicate_purchase_rows_removed"]),
            }
        ],
        columns=list(REMEDIATION_IMPACT_COLUMNS),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_metric_report(
    tables: Mapping[str, pd.DataFrame] | None = None,
    current_period: str | None = None,
    previous_period: str | None = None,
    data_dir: Path = DATA_DIR,
) -> MetricReport:
    """Build all three dashboard frames in one pass.

    Periods default to the two most recent months in the data, so the report is
    correct without hardcoding a demo date.
    """
    transactions = transactions_from(tables, data_dir)
    trend = monthly_trend_table(transactions)
    resolved_current, resolved_previous = resolve_periods(trend, current_period, previous_period)

    return MetricReport(
        metric_name=METRIC_NAME,
        current_period=resolved_current,
        previous_period=resolved_previous,
        monthly_trend=trend,
        period_comparison=period_comparison_table(trend, resolved_current, resolved_previous),
        remediation_impact=remediation_impact_table(trend, resolved_current),
    )


if __name__ == "__main__":
    report = build_metric_report()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print(f"MetricGuard AI - {report.metric_name} calculation and duplicate remediation")
    print("=" * 100)
    print()
    print("Monthly trend")
    print("-" * 100)
    print(report.monthly_trend.to_string(index=False))
    print()
    print(f"Period comparison: {report.current_period} vs {report.previous_period}")
    print("-" * 100)
    print(report.period_comparison.to_string(index=False))
    print()
    print(f"Duplicate remediation impact: {report.current_period}")
    print("-" * 100)
    print(report.remediation_impact.T.to_string(header=False))
