"""Data-quality and observability checks for MetricGuard AI.

This module answers the first question in the MetricGuard workflow:

    Before we explain a KPI movement, can we trust the underlying records?

It is deliberately deterministic and LLM-free. Every check is a pure function of
the synthetic tables, so the same data always produces the same report. The LLM
layer later explains these facts; it never produces them.

Three families of checks are implemented:

``rule_based``
    Hard contract violations that are true or false for a given row: duplicate
    upstream event IDs, required-field completeness, date ordering, amount-sign
    validity, accepted values, and referential integrity.

``statistical``
    Observability checks that compare a month against its own rolling history:
    row-count anomalies, null-rate drift, category-distribution drift (PSI), and
    KPI anomaly for ``dispute_rate``.

``governance``
    Metric-definition completeness, supporting the metadata / business-definition
    story in ``PROJECT_SPEC.md``.

Every check returns one row of :data:`RESULT_COLUMNS`. Two column conventions are
worth stating explicitly:

* ``affected_rows`` is always an integer and is the machine-readable field.
* ``observed_value`` / ``expected_value`` are human-readable strings, because the
  checks are heterogeneous (a count, a rate, a z-score, a PSI value) and a single
  numeric column could not carry all of them without losing meaning.

Run directly to print the full report::

    python src/quality_checks.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from config import DATA_DIR

RESULT_COLUMNS = (
    "check_name",
    "check_type",
    "table_name",
    "status",
    "severity",
    "affected_rows",
    "affected_columns",
    "observed_value",
    "expected_value",
    "explanation",
    "recommended_action",
)

STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"

SEVERITY_NONE = "none"
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Canonical table names match the physical CSV file names so that the report
# reads like a data-quality report against real tables.
TABLE_ACCOUNTS = "accounts"
TABLE_SNAPSHOTS = "account_monthly_snapshot"
TABLE_TRANSACTIONS = "transactions"
TABLE_COMPLAINTS = "complaints"
TABLE_METRIC_DEFINITIONS = "metric_definitions"

# The engine in ``metricguard_engine.py`` keys the snapshot table as "snapshots".
# Accepting both keeps this module drop-in compatible with its loader.
_TABLE_ALIASES = {"snapshots": TABLE_SNAPSHOTS, "snapshot": TABLE_SNAPSHOTS}

# Fields that must never be null. Optional analytical fields are deliberately
# excluded: complaints.merchant_category is documented as an optional link in
# DATA_DICTIONARY.md, so a null there is not a completeness failure.
REQUIRED_FIELDS: Mapping[str, tuple[str, ...]] = {
    TABLE_ACCOUNTS: (
        "account_id",
        "open_date",
        "product_type",
        "customer_segment",
        "fico_band",
        "region",
        "active_flag",
        "credit_limit",
    ),
    TABLE_TRANSACTIONS: (
        "transaction_id",
        "source_transaction_id",
        "account_id",
        "transaction_date",
        "posted_date",
        "merchant_category",
        "channel",
        "transaction_amount",
        "transaction_type",
        "is_disputed",
        "ingestion_batch_id",
        "source_system",
    ),
    TABLE_COMPLAINTS: (
        "complaint_id",
        "account_id",
        "date_received",
        "product",
        "issue",
        "complaint_narrative",
        "submitted_via",
    ),
    TABLE_SNAPSHOTS: (
        "snapshot_month",
        "account_id",
        "active_flag",
        "statement_balance",
        "credit_limit",
        "is_30dpd",
    ),
}

# Accepted value domains, checked on the table where each field is authoritative.
# Nulls are excluded here on purpose: missing values are already reported by the
# completeness check and should not be double-counted as unexpected values.
ACCEPTED_VALUES: Mapping[tuple[str, str], tuple[str, ...]] = {
    (TABLE_TRANSACTIONS, "transaction_type"): ("purchase", "payment", "fee", "refund"),
    (TABLE_TRANSACTIONS, "channel"): ("card_present", "web", "mobile", "recurring", "system"),
    (TABLE_TRANSACTIONS, "merchant_category"): (
        "grocery",
        "restaurant",
        "gas",
        "online_retail",
        "subscription",
        "travel",
        "health",
        "education",
        "fee",
        "payment",
    ),
    (TABLE_ACCOUNTS, "fico_band"): (">660", "<=660"),
    (TABLE_ACCOUNTS, "product_type"): (
        "cash_rewards",
        "travel_rewards",
        "student_card",
        "secured_card",
        "venture_style",
    ),
    (TABLE_COMPLAINTS, "issue"): (
        "Problem with a purchase shown on your statement",
        "Fees or interest",
        "Problem when making payments",
        "Problem with fraud alerts or security",
    ),
}

# child table -> (foreign key column, parent table, parent key column)
REFERENTIAL_INTEGRITY_RULES: tuple[tuple[str, str, str, str], ...] = (
    (TABLE_TRANSACTIONS, "account_id", TABLE_ACCOUNTS, "account_id"),
    (TABLE_COMPLAINTS, "account_id", TABLE_ACCOUNTS, "account_id"),
    (TABLE_SNAPSHOTS, "account_id", TABLE_ACCOUNTS, "account_id"),
)

# How to derive a "YYYY-MM" period key for each event table.
MONTH_SOURCE: Mapping[str, tuple[str, str]] = {
    TABLE_TRANSACTIONS: ("transaction_date", "date"),
    TABLE_COMPLAINTS: ("date_received", "date"),
    TABLE_SNAPSHOTS: ("snapshot_month", "month"),
}

# Fields watched for null-rate drift, with the practical-significance floor used
# to suppress statistically large but operationally irrelevant movement.
NULL_RATE_MONITORED_FIELDS: tuple[tuple[str, str], ...] = (
    (TABLE_TRANSACTIONS, "merchant_category"),
    (TABLE_TRANSACTIONS, "channel"),
    (TABLE_TRANSACTIONS, "transaction_amount"),
    (TABLE_TRANSACTIONS, "posted_date"),
    (TABLE_COMPLAINTS, "issue"),
    (TABLE_COMPLAINTS, "complaint_narrative"),
    (TABLE_SNAPSHOTS, "statement_balance"),
    (TABLE_SNAPSHOTS, "is_30dpd"),
)

CATEGORY_DRIFT_FIELDS: tuple[tuple[str, str], ...] = (
    (TABLE_TRANSACTIONS, "merchant_category"),
    (TABLE_TRANSACTIONS, "channel"),
)

METRIC_DEFINITION_REQUIRED_FIELDS: tuple[str, ...] = (
    "metric_name",
    "business_definition",
    "numerator",
    "denominator",
    "grain",
    "owner",
    "source_tables",
    "known_limitations",
)

MISSING_BUCKET = "__missing__"


@dataclass(frozen=True)
class Thresholds:
    """Tunable thresholds for the statistical observability checks.

    ``window`` / ``min_periods`` define the rolling baseline: each month is
    compared against up to ``window`` preceding months and is only evaluated once
    ``min_periods`` months of history exist.

    ``warn_z`` / ``fail_z`` are the z-score bands.

    ``psi_warn`` / ``psi_fail`` follow the conventional Population Stability Index
    reading: below 0.10 is stable, 0.10-0.25 is a moderate shift, and 0.25 or
    above is a significant shift.
    """

    window: int = 6
    min_periods: int = 3
    warn_z: float = 2.0
    fail_z: float = 3.0
    psi_warn: float = 0.10
    psi_fail: float = 0.25
    # Practical-significance floors, in the natural unit of each check.
    row_count_min_abs_delta: float = 0.0
    null_rate_min_abs_delta: float = 0.002
    kpi_min_abs_delta: float = 0.001


DEFAULT_THRESHOLDS = Thresholds()


# ---------------------------------------------------------------------------
# Loading and small helpers
# ---------------------------------------------------------------------------


def load_tables(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the synthetic tables keyed by their physical table name."""
    return {
        TABLE_ACCOUNTS: pd.read_csv(data_dir / "accounts.csv", parse_dates=["open_date"]),
        TABLE_SNAPSHOTS: pd.read_csv(
            data_dir / "account_monthly_snapshot.csv",
            parse_dates=["snapshot_loaded_at"],
        ),
        TABLE_TRANSACTIONS: pd.read_csv(
            data_dir / "transactions.csv",
            parse_dates=["transaction_date", "posted_date", "created_at"],
        ),
        TABLE_COMPLAINTS: pd.read_csv(
            data_dir / "complaints.csv", parse_dates=["date_received"]
        ),
        TABLE_METRIC_DEFINITIONS: pd.read_csv(data_dir / "metric_definitions.csv"),
    }


def _normalize_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Accept either this module's or the engine's table key naming."""
    return {_TABLE_ALIASES.get(name, name): df for name, df in tables.items()}


def _result(
    check_name: str,
    check_type: str,
    table_name: str,
    status: str,
    severity: str,
    affected_rows: int,
    affected_columns: str,
    observed_value: str,
    expected_value: str,
    explanation: str,
    recommended_action: str,
) -> dict[str, object]:
    return {
        "check_name": check_name,
        "check_type": check_type,
        "table_name": table_name,
        "status": status,
        "severity": severity,
        "affected_rows": int(affected_rows),
        "affected_columns": affected_columns,
        "observed_value": observed_value,
        "expected_value": expected_value,
        "explanation": explanation,
        "recommended_action": recommended_action,
    }


def _severity_for(status: str, failed: str, warned: str = SEVERITY_MEDIUM) -> str:
    if status == STATUS_FAIL:
        return failed
    if status == STATUS_WARN:
        return warned
    return SEVERITY_NONE


def _is_blank(series: pd.Series) -> pd.Series:
    """Null, or a string that is empty once stripped."""
    blank = series.isna()
    if series.dtype == object:
        blank = blank | series.astype("string").str.strip().eq("").fillna(False)
    return blank


def _month_key(df: pd.DataFrame, table_name: str) -> pd.Series:
    """Return a ``YYYY-MM`` period key for an event table."""
    column, kind = MONTH_SOURCE[table_name]
    values = df[column]
    if kind == "month":
        return values.astype(str)
    return pd.to_datetime(values).dt.to_period("M").astype(str)


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Rolling-baseline machinery
# ---------------------------------------------------------------------------


def rolling_baseline(
    series: pd.Series, window: int, min_periods: int
) -> tuple[pd.Series, pd.Series]:
    """Mean and population std of the ``window`` months preceding each month.

    ``shift(1)`` keeps the current month out of its own baseline, so a month can
    never mask its own anomaly.
    """
    prior = series.shift(1)
    mean = prior.rolling(window=window, min_periods=min_periods).mean()
    std = prior.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return mean, std


def floored_zscore(
    value: float, mean: float, std: float, min_abs_delta: float, fail_z: float
) -> float:
    """Z-score with the baseline std floored by a practical-significance term.

    A perfectly stable history has ``std == 0``, which would make an ordinary
    z-score infinite for any movement at all. Flooring the std at
    ``min_abs_delta / fail_z`` fixes that and gives the floor a second, useful
    meaning: a deviation smaller than ``min_abs_delta`` can never reach
    ``fail_z``, so tiny-but-statistically-loud movement is suppressed.
    """
    if not np.isfinite(value) or not np.isfinite(mean):
        return float("nan")
    std_floor = (min_abs_delta / fail_z) if fail_z > 0 else 0.0
    effective_std = max(std if np.isfinite(std) else 0.0, std_floor)
    if effective_std <= 0:
        return 0.0
    return float((value - mean) / effective_std)


@dataclass(frozen=True)
class AnomalyScan:
    """The most extreme month found while scanning a monthly series."""

    month: str | None
    value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    evaluated_months: int


def scan_monthly_anomaly(
    series: pd.Series, thresholds: Thresholds, min_abs_delta: float
) -> AnomalyScan:
    """Score every month against its own rolling baseline, return the worst one.

    Scanning all months rather than only the latest one matters here: a defect
    can land in any month, and the July merchant-category gap in the synthetic
    data would be invisible to a latest-month-only check.
    """
    ordered = series.sort_index()
    mean, std = rolling_baseline(ordered, thresholds.window, thresholds.min_periods)
    scored = [
        (
            month,
            float(ordered.loc[month]),
            float(mean.loc[month]),
            float(std.loc[month]),
            floored_zscore(
                float(ordered.loc[month]),
                float(mean.loc[month]),
                float(std.loc[month]),
                min_abs_delta,
                thresholds.fail_z,
            ),
        )
        for month in ordered.index
        if np.isfinite(mean.loc[month])
    ]
    if not scored:
        return AnomalyScan(None, float("nan"), float("nan"), float("nan"), float("nan"), 0)
    worst = max(scored, key=lambda row: abs(row[4]) if np.isfinite(row[4]) else -1.0)
    return AnomalyScan(worst[0], worst[1], worst[2], worst[3], worst[4], len(scored))


def _status_for_z(scan: AnomalyScan, thresholds: Thresholds) -> str:
    if scan.month is None or not np.isfinite(scan.z_score):
        return STATUS_PASS
    magnitude = abs(scan.z_score)
    if magnitude >= thresholds.fail_z:
        return STATUS_FAIL
    if magnitude >= thresholds.warn_z:
        return STATUS_WARN
    return STATUS_PASS


def population_stability_index(
    actual: pd.Series, expected: pd.Series, epsilon: float = 1e-6
) -> tuple[float, pd.Series]:
    """PSI between two category distributions, plus per-bucket contributions.

    Both inputs are proportion vectors over the same buckets. Values are clipped
    at ``epsilon`` so a bucket that is absent from one side cannot produce an
    infinite term.
    """
    buckets = actual.index.union(expected.index)
    a = actual.reindex(buckets).fillna(0.0).clip(lower=epsilon)
    e = expected.reindex(buckets).fillna(0.0).clip(lower=epsilon)
    contributions = (a - e) * np.log(a / e)
    return float(contributions.sum()), contributions


# ---------------------------------------------------------------------------
# Rule-based checks
# ---------------------------------------------------------------------------


def check_duplicate_source_transaction_ids(transactions: pd.DataFrame) -> list[dict[str, object]]:
    """Check 1: a source event should appear exactly once in transactions."""
    duplicated_rows = int(transactions["source_transaction_id"].duplicated(keep="first").sum())
    distinct_ids = int(
        transactions.loc[
            transactions["source_transaction_id"].duplicated(keep=False),
            "source_transaction_id",
        ].nunique()
    )
    status = STATUS_FAIL if duplicated_rows else STATUS_PASS

    batches = ""
    if duplicated_rows:
        offenders = transactions.loc[
            transactions["source_transaction_id"].duplicated(keep=False), "ingestion_batch_id"
        ]
        batches = ", ".join(str(b) for b in offenders.value_counts().head(3).index)

    return [
        _result(
            check_name="duplicate_source_transaction_id",
            check_type="rule_based",
            table_name=TABLE_TRANSACTIONS,
            status=status,
            severity=_severity_for(status, SEVERITY_HIGH),
            affected_rows=duplicated_rows,
            affected_columns="source_transaction_id",
            observed_value=f"{duplicated_rows} duplicate rows across {distinct_ids} source IDs",
            expected_value="0 duplicate rows",
            explanation=(
                "Each upstream event should land once. Repeated source_transaction_id "
                "values indicate a replayed or double-loaded file, which inflates every "
                "count-based KPI built on this table."
                + (f" Top batches: {batches}." if batches else "")
            ),
            recommended_action=(
                "Deduplicate on source_transaction_id (keep earliest created_at) before "
                "reporting, and confirm the replayed batch with the source-system owner."
                if duplicated_rows
                else "No action required."
            ),
        )
    ]


def check_required_field_completeness(
    tables: Mapping[str, pd.DataFrame]
) -> list[dict[str, object]]:
    """Check 2: required fields must be populated on every row."""
    results: list[dict[str, object]] = []
    for table_name, fields in REQUIRED_FIELDS.items():
        frame = tables[table_name]
        present = [field for field in fields if field in frame.columns]
        absent = [field for field in fields if field not in frame.columns]

        blank_mask = pd.Series(False, index=frame.index)
        per_field: list[tuple[str, int]] = []
        for field in present:
            field_blank = _is_blank(frame[field])
            count = int(field_blank.sum())
            if count:
                per_field.append((field, count))
            blank_mask = blank_mask | field_blank

        affected_rows = int(blank_mask.sum())
        status = STATUS_FAIL if (affected_rows or absent) else STATUS_PASS
        detail = ", ".join(f"{field}={count}" for field, count in per_field) or "none"
        if absent:
            detail = f"missing columns: {', '.join(absent)}; " + detail

        results.append(
            _result(
                check_name=f"required_field_completeness__{table_name}",
                check_type="rule_based",
                table_name=table_name,
                status=status,
                severity=_severity_for(status, SEVERITY_HIGH),
                affected_rows=affected_rows,
                affected_columns=", ".join(field for field, _ in per_field) or "",
                observed_value=f"{affected_rows} incomplete rows ({detail})",
                expected_value="0 incomplete rows",
                explanation=(
                    "Required fields drive KPI numerators, denominators, and segment "
                    f"drilldowns for {table_name}. Nulls silently drop rows from grouped "
                    "analysis and bias the segments that remain."
                ),
                recommended_action=(
                    "Trace the null rows back to their ingestion batch and source system, "
                    "and exclude or backfill them before publishing segment cuts."
                    if status == STATUS_FAIL
                    else "No action required."
                ),
            )
        )
    return results


def check_posted_date_not_before_transaction_date(
    transactions: pd.DataFrame,
) -> list[dict[str, object]]:
    """Check 3: posted_date must be on or after transaction_date."""
    transaction_date = pd.to_datetime(transactions["transaction_date"])
    posted_date = pd.to_datetime(transactions["posted_date"])
    violations = int((posted_date < transaction_date).sum())
    status = STATUS_FAIL if violations else STATUS_PASS

    lag = (posted_date - transaction_date).dt.days
    return [
        _result(
            check_name="posted_date_not_before_transaction_date",
            check_type="rule_based",
            table_name=TABLE_TRANSACTIONS,
            status=status,
            severity=_severity_for(status, SEVERITY_HIGH),
            affected_rows=violations,
            affected_columns="posted_date, transaction_date",
            observed_value=(
                f"{violations} rows with posted_date < transaction_date; "
                f"observed posting lag min={_fmt(float(lag.min()), 0)} "
                f"max={_fmt(float(lag.max()), 0)} days"
            ),
            expected_value="0 rows with posted_date < transaction_date",
            explanation=(
                "A transaction cannot post before it occurs. Negative posting lag points "
                "to a timezone or batch-dating defect, which shifts events between "
                "reporting months and distorts month-over-month KPI comparisons."
            ),
            recommended_action=(
                "Inspect the affected source system's date handling and restate the "
                "impacted reporting months."
                if violations
                else "No action required."
            ),
        )
    ]


def check_amount_sign_validity(transactions: pd.DataFrame) -> list[dict[str, object]]:
    """Check 4: transaction_amount sign must match the transaction semantics."""
    amount = pd.to_numeric(transactions["transaction_amount"], errors="coerce")
    transaction_type = transactions["transaction_type"]
    payment_failed = pd.to_numeric(transactions["payment_failed"], errors="coerce")

    is_payment = transaction_type.eq("payment")
    rules = {
        "purchase_amount_not_positive": transaction_type.eq("purchase") & ~amount.gt(0),
        "fee_amount_not_positive": transaction_type.eq("fee") & ~amount.gt(0),
        "successful_payment_amount_not_negative": (
            is_payment & payment_failed.eq(0) & ~amount.lt(0)
        ),
        "failed_payment_amount_not_zero": is_payment & payment_failed.eq(1) & ~amount.eq(0),
    }

    violation_mask = pd.Series(False, index=transactions.index)
    breakdown: list[str] = []
    for rule_name, mask in rules.items():
        filled = mask.fillna(False)
        breakdown.append(f"{rule_name}={int(filled.sum())}")
        violation_mask = violation_mask | filled

    affected_rows = int(violation_mask.sum())
    status = STATUS_FAIL if affected_rows else STATUS_PASS

    return [
        _result(
            check_name="amount_sign_validity",
            check_type="rule_based",
            table_name=TABLE_TRANSACTIONS,
            status=status,
            severity=_severity_for(status, SEVERITY_HIGH),
            affected_rows=affected_rows,
            affected_columns="transaction_amount, transaction_type, payment_failed",
            observed_value=f"{affected_rows} violations ({'; '.join(breakdown)})",
            expected_value=(
                "purchase > 0; fee > 0; successful payment < 0; failed payment = 0"
            ),
            explanation=(
                "Amount sign encodes direction of money movement. A purchase or fee "
                "recorded as a credit, or a failed payment carrying a non-zero amount, "
                "corrupts purchase volume and payment-failure reporting."
            ),
            recommended_action=(
                "Reconcile the offending rows against the source ledger and correct the "
                "sign convention in the ingestion mapping."
                if affected_rows
                else "No action required."
            ),
        )
    ]


def check_accepted_values(tables: Mapping[str, pd.DataFrame]) -> list[dict[str, object]]:
    """Check 5: categorical fields must stay inside their documented domain.

    Nulls are ignored here; the completeness check already owns them.
    """
    results: list[dict[str, object]] = []
    for (table_name, column), accepted in ACCEPTED_VALUES.items():
        frame = tables[table_name]
        values = frame[column]
        non_null = values[~values.isna()]
        unexpected_mask = ~non_null.astype(str).isin(accepted)
        affected_rows = int(unexpected_mask.sum())
        status = STATUS_FAIL if affected_rows else STATUS_PASS
        unexpected_values = (
            ", ".join(sorted(non_null[unexpected_mask].astype(str).unique())[:5])
            if affected_rows
            else "none"
        )

        results.append(
            _result(
                check_name=f"accepted_values__{table_name}__{column}",
                check_type="rule_based",
                table_name=table_name,
                status=status,
                severity=_severity_for(status, SEVERITY_MEDIUM),
                affected_rows=affected_rows,
                affected_columns=column,
                observed_value=(
                    f"{affected_rows} rows outside the accepted domain "
                    f"(unexpected values: {unexpected_values})"
                ),
                expected_value=f"values in [{', '.join(accepted)}]",
                explanation=(
                    f"{table_name}.{column} is a governed dimension used for segment "
                    "drilldowns. A new or misspelled value fragments a segment and makes "
                    "period-over-period comparisons inconsistent."
                ),
                recommended_action=(
                    "Confirm whether the new value is a legitimate business change; if so "
                    "update the metric definition and this domain, otherwise fix the mapping."
                    if affected_rows
                    else "No action required."
                ),
            )
        )
    return results


def check_referential_integrity(tables: Mapping[str, pd.DataFrame]) -> list[dict[str, object]]:
    """Check 6: every child foreign key must resolve to a parent row."""
    results: list[dict[str, object]] = []
    for child_table, child_column, parent_table, parent_column in REFERENTIAL_INTEGRITY_RULES:
        child = tables[child_table]
        parent_keys = set(tables[parent_table][parent_column].dropna().astype(str))
        child_keys = child[child_column].astype(str)
        orphan_mask = ~child_keys.isin(parent_keys)
        affected_rows = int(orphan_mask.sum())
        orphan_keys = int(child_keys[orphan_mask].nunique())
        status = STATUS_FAIL if affected_rows else STATUS_PASS

        results.append(
            _result(
                check_name=f"referential_integrity__{child_table}__{child_column}",
                check_type="rule_based",
                table_name=child_table,
                status=status,
                severity=_severity_for(status, SEVERITY_HIGH),
                affected_rows=affected_rows,
                affected_columns=child_column,
                observed_value=(
                    f"{affected_rows} orphan rows across {orphan_keys} unresolved "
                    f"{child_column} values"
                ),
                expected_value=f"every {child_table}.{child_column} present in "
                f"{parent_table}.{parent_column}",
                explanation=(
                    f"Segment analysis joins {child_table} to {parent_table} to attach "
                    "product, FICO band, customer segment, and region. Orphan keys drop "
                    "out of that join, so the drivers view silently covers fewer rows than "
                    "the headline KPI."
                ),
                recommended_action=(
                    "Check load ordering between the two tables and re-run the child load "
                    "after the parent dimension is current."
                    if affected_rows
                    else "No action required."
                ),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Statistical observability checks
# ---------------------------------------------------------------------------


def check_monthly_row_count_anomaly(
    tables: Mapping[str, pd.DataFrame], thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> list[dict[str, object]]:
    """Check 7: monthly volume should track its own rolling history."""
    results: list[dict[str, object]] = []
    for table_name in (TABLE_TRANSACTIONS, TABLE_COMPLAINTS, TABLE_SNAPSHOTS):
        frame = tables[table_name]
        counts = _month_key(frame, table_name).value_counts().sort_index()
        scan = scan_monthly_anomaly(
            counts.astype(float), thresholds, thresholds.row_count_min_abs_delta
        )
        status = _status_for_z(scan, thresholds)

        results.append(
            _result(
                check_name=f"monthly_row_count_anomaly__{table_name}",
                check_type="statistical",
                table_name=table_name,
                status=status,
                severity=_severity_for(status, SEVERITY_HIGH, SEVERITY_MEDIUM),
                affected_rows=int(scan.value) if status != STATUS_PASS and np.isfinite(scan.value) else 0,
                affected_columns=MONTH_SOURCE[table_name][0],
                observed_value=(
                    f"{scan.month}: {_fmt(scan.value, 0)} rows, z={_fmt(scan.z_score, 2)} "
                    f"(baseline mean={_fmt(scan.baseline_mean, 1)}, "
                    f"std={_fmt(scan.baseline_std, 1)}, {scan.evaluated_months} months scored)"
                    if scan.month
                    else "insufficient history to score"
                ),
                expected_value=f"|z| < {thresholds.warn_z} against a {thresholds.window}-month rolling baseline",
                explanation=(
                    "A month whose row count departs from its rolling baseline usually "
                    "means a replayed file, a partial load, or a genuine volume shift. "
                    "Ruling out volume artifacts first prevents a pipeline problem from "
                    "being read as a business movement."
                ),
                recommended_action=(
                    f"Reconcile the {scan.month} load against source-system control totals "
                    "before trusting that month's KPIs."
                    if status != STATUS_PASS
                    else "No action required."
                ),
            )
        )
    return results


def check_null_rate_drift(
    tables: Mapping[str, pd.DataFrame], thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> list[dict[str, object]]:
    """Check 8: the null rate of an important field should be stable over time.

    Completeness (check 2) says whether nulls exist; this says whether they
    suddenly appeared, which is what localises the defect to a batch and month.
    """
    results: list[dict[str, object]] = []
    for table_name, column in NULL_RATE_MONITORED_FIELDS:
        frame = tables[table_name]
        month = _month_key(frame, table_name)
        blank = _is_blank(frame[column])
        null_rate = blank.groupby(month).mean().sort_index()
        scan = scan_monthly_anomaly(
            null_rate.astype(float), thresholds, thresholds.null_rate_min_abs_delta
        )
        status = _status_for_z(scan, thresholds)
        affected_rows = (
            int(blank[month.eq(scan.month)].sum()) if status != STATUS_PASS and scan.month else 0
        )

        results.append(
            _result(
                check_name=f"null_rate_drift__{table_name}__{column}",
                check_type="statistical",
                table_name=table_name,
                status=status,
                severity=_severity_for(status, SEVERITY_HIGH, SEVERITY_MEDIUM),
                affected_rows=affected_rows,
                affected_columns=column,
                observed_value=(
                    f"{scan.month}: null rate={_fmt(scan.value)}, z={_fmt(scan.z_score, 2)} "
                    f"(baseline mean={_fmt(scan.baseline_mean)}, std={_fmt(scan.baseline_std)})"
                    if scan.month
                    else "insufficient history to score"
                ),
                expected_value=(
                    f"|z| < {thresholds.warn_z} and null-rate move < "
                    f"{thresholds.null_rate_min_abs_delta:.3f}"
                ),
                explanation=(
                    f"A step change in the {column} null rate marks the month a feed "
                    "started dropping the field. Rows missing a dimension disappear from "
                    "segment cuts even though they still count toward the headline KPI."
                ),
                recommended_action=(
                    f"Identify the ingestion batches supplying {column} in {scan.month} and "
                    "backfill or quarantine them before publishing segment analysis."
                    if status != STATUS_PASS
                    else "No action required."
                ),
            )
        )
    return results


def check_category_distribution_drift(
    tables: Mapping[str, pd.DataFrame], thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> list[dict[str, object]]:
    """Check 9: category mix should be stable, measured by PSI.

    Missing values get their own ``__missing__`` bucket so that a field going
    blank registers as a mix shift rather than quietly shrinking the denominator.
    """
    results: list[dict[str, object]] = []
    for table_name, column in CATEGORY_DRIFT_FIELDS:
        frame = tables[table_name]
        month = _month_key(frame, table_name)
        values = frame[column].where(~_is_blank(frame[column]), MISSING_BUCKET).astype(str)

        counts = pd.crosstab(month, values)
        distribution = counts.div(counts.sum(axis=1), axis=0).sort_index()

        worst_month: str | None = None
        worst_psi = float("nan")
        worst_bucket = ""
        scored_months = 0
        for position, current_month in enumerate(distribution.index):
            prior = distribution.iloc[max(0, position - thresholds.window) : position]
            if len(prior) < thresholds.min_periods:
                continue
            scored_months += 1
            psi, contributions = population_stability_index(
                distribution.loc[current_month], prior.mean(axis=0)
            )
            if not np.isfinite(worst_psi) or psi > worst_psi:
                worst_month, worst_psi, worst_bucket = (
                    current_month,
                    psi,
                    str(contributions.idxmax()),
                )

        if worst_month is None:
            status = STATUS_PASS
        elif worst_psi >= thresholds.psi_fail:
            status = STATUS_FAIL
        elif worst_psi >= thresholds.psi_warn:
            status = STATUS_WARN
        else:
            status = STATUS_PASS

        # Report the rows behind the bucket that moved most, not the whole month:
        # that is the population an analyst actually has to go and reconcile.
        affected_rows = 0
        if status != STATUS_PASS and worst_month:
            affected_rows = int((month.eq(worst_month) & values.eq(worst_bucket)).sum())

        results.append(
            _result(
                check_name=f"category_distribution_drift__{table_name}__{column}",
                check_type="statistical",
                table_name=table_name,
                status=status,
                severity=_severity_for(status, SEVERITY_HIGH, SEVERITY_MEDIUM),
                affected_rows=affected_rows,
                affected_columns=column,
                observed_value=(
                    f"{worst_month}: PSI={_fmt(worst_psi)} "
                    f"(largest contributor: {worst_bucket}, {scored_months} months scored)"
                    if worst_month
                    else "insufficient history to score"
                ),
                expected_value=(
                    f"PSI < {thresholds.psi_warn:.2f} against a "
                    f"{thresholds.window}-month rolling baseline mix"
                ),
                explanation=(
                    f"Population Stability Index compares this month's {column} mix with "
                    "its recent baseline. Conventionally, PSI below 0.10 is stable, "
                    "0.10-0.25 is a moderate shift, and 0.25 or more is significant. A mix "
                    "shift can move a rate-based KPI without any single segment's rate "
                    "changing at all."
                ),
                recommended_action=(
                    f"Compare the {worst_bucket} share in {worst_month} against source "
                    "counts to separate a mapping change from real customer behavior."
                    if status != STATUS_PASS
                    else "No action required."
                ),
            )
        )
    return results


def monthly_dispute_rate_series(transactions: pd.DataFrame) -> pd.Series:
    """Monthly dispute rate: disputed purchases / purchase transactions.

    Matches the ``dispute_rate`` definition in ``metric_definitions.csv``. The
    raw (non-deduplicated) series is used on purpose, because this check exists
    to reproduce what the dashboard alert showed before remediation.
    """
    purchases = transactions[transactions["transaction_type"].eq("purchase")].copy()
    month = pd.to_datetime(purchases["transaction_date"]).dt.to_period("M").astype(str)
    grouped = purchases.groupby(month)["is_disputed"].agg(["sum", "count"])
    return (grouped["sum"] / grouped["count"]).sort_index()


def check_kpi_anomaly_dispute_rate(
    tables: Mapping[str, pd.DataFrame], thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> list[dict[str, object]]:
    """Check 10: dispute_rate should track its own rolling baseline."""
    transactions = tables[TABLE_TRANSACTIONS]
    series = monthly_dispute_rate_series(transactions)
    scan = scan_monthly_anomaly(series, thresholds, thresholds.kpi_min_abs_delta)
    status = _status_for_z(scan, thresholds)

    affected_rows = 0
    if status != STATUS_PASS and scan.month:
        purchases = transactions[transactions["transaction_type"].eq("purchase")]
        month = pd.to_datetime(purchases["transaction_date"]).dt.to_period("M").astype(str)
        affected_rows = int(purchases.loc[month.eq(scan.month), "is_disputed"].sum())

    return [
        _result(
            check_name="kpi_anomaly__dispute_rate",
            check_type="statistical",
            table_name=TABLE_TRANSACTIONS,
            status=status,
            severity=_severity_for(status, SEVERITY_HIGH, SEVERITY_MEDIUM),
            affected_rows=affected_rows,
            affected_columns="is_disputed, transaction_type, transaction_date",
            observed_value=(
                f"{scan.month}: dispute_rate={_fmt(scan.value)}, z={_fmt(scan.z_score, 2)} "
                f"(baseline mean={_fmt(scan.baseline_mean)}, std={_fmt(scan.baseline_std)})"
                if scan.month
                else "insufficient history to score"
            ),
            expected_value=(
                f"|z| < {thresholds.warn_z} and move < {thresholds.kpi_min_abs_delta:.3f} "
                f"against a {thresholds.window}-month rolling baseline"
            ),
            explanation=(
                "This is the metric side of the investigation: the KPI itself is scored "
                "against its own history. On its own an anomaly here is only an alert. It "
                "becomes a finding only once the rule-based and drift checks above have "
                "ruled out duplicate, missing, or mis-mapped source records."
            ),
            recommended_action=(
                f"Recompute {scan.month} dispute_rate on deduplicated source events, then "
                "run segment driver analysis on the remaining movement."
                if status != STATUS_PASS
                else "No action required."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Metric governance check
# ---------------------------------------------------------------------------


def check_metric_definition_governance(
    metric_definitions: pd.DataFrame,
) -> list[dict[str, object]]:
    """Check 11: every registered metric must carry its governance metadata."""
    absent_columns = [
        field for field in METRIC_DEFINITION_REQUIRED_FIELDS if field not in metric_definitions.columns
    ]
    present_columns = [
        field for field in METRIC_DEFINITION_REQUIRED_FIELDS if field in metric_definitions.columns
    ]

    incomplete_mask = pd.Series(False, index=metric_definitions.index)
    incomplete_fields: list[str] = []
    for field in present_columns:
        field_blank = _is_blank(metric_definitions[field])
        if field_blank.any():
            incomplete_fields.append(f"{field}={int(field_blank.sum())}")
        incomplete_mask = incomplete_mask | field_blank

    affected_rows = int(incomplete_mask.sum())
    status = STATUS_FAIL if (affected_rows or absent_columns) else STATUS_PASS

    incomplete_metrics = ""
    if affected_rows and "metric_name" in metric_definitions.columns:
        incomplete_metrics = ", ".join(
            metric_definitions.loc[incomplete_mask, "metric_name"].astype(str).head(5)
        )

    detail = "; ".join(incomplete_fields) or "none"
    if absent_columns:
        detail = f"missing columns: {', '.join(absent_columns)}; " + detail

    return [
        _result(
            check_name="metric_definition_governance_completeness",
            check_type="governance",
            table_name=TABLE_METRIC_DEFINITIONS,
            status=status,
            severity=_severity_for(status, SEVERITY_MEDIUM),
            affected_rows=affected_rows,
            affected_columns=", ".join(
                field.split("=")[0] for field in incomplete_fields
            )
            or "",
            observed_value=(
                f"{len(metric_definitions) - affected_rows} of {len(metric_definitions)} "
                f"metrics fully documented (gaps: {detail})"
                + (f"; incomplete: {incomplete_metrics}" if incomplete_metrics else "")
            ),
            expected_value=(
                "every metric documents "
                f"{', '.join(METRIC_DEFINITION_REQUIRED_FIELDS)}"
            ),
            explanation=(
                "A KPI is only trustworthy if its business definition, numerator, "
                "denominator, grain, owner, lineage, and known limitations are recorded. "
                "Without those, two teams can compute the same metric name differently and "
                "a movement cannot be attributed to a definition change."
            ),
            recommended_action=(
                "Assign an owner to each undocumented metric and complete the missing "
                "governance fields before the metric is published to a dashboard."
                if status == STATUS_FAIL
                else "No action required."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_quality_checks(
    tables: Mapping[str, pd.DataFrame] | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Run every check and return one row per check.

    Parameters
    ----------
    tables:
        Mapping of table name to DataFrame. Accepts either this module's naming
        or ``metricguard_engine.load_synthetic_data`` naming. Loaded from
        ``data_dir`` when omitted.
    thresholds:
        Thresholds for the statistical checks.

    Returns
    -------
    pandas.DataFrame
        Columns are exactly :data:`RESULT_COLUMNS`. Rows are ordered fail, then
        warn, then pass, so the report leads with what needs attention.
    """
    resolved = _normalize_tables(tables if tables is not None else load_tables(data_dir))

    rows: list[dict[str, object]] = []
    rows += check_duplicate_source_transaction_ids(resolved[TABLE_TRANSACTIONS])
    rows += check_required_field_completeness(resolved)
    rows += check_posted_date_not_before_transaction_date(resolved[TABLE_TRANSACTIONS])
    rows += check_amount_sign_validity(resolved[TABLE_TRANSACTIONS])
    rows += check_accepted_values(resolved)
    rows += check_referential_integrity(resolved)
    rows += check_monthly_row_count_anomaly(resolved, thresholds)
    rows += check_null_rate_drift(resolved, thresholds)
    rows += check_category_distribution_drift(resolved, thresholds)
    rows += check_kpi_anomaly_dispute_rate(resolved, thresholds)
    rows += check_metric_definition_governance(resolved[TABLE_METRIC_DEFINITIONS])

    results = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
    status_rank = {STATUS_FAIL: 0, STATUS_WARN: 1, STATUS_PASS: 2}
    severity_rank = {SEVERITY_HIGH: 0, SEVERITY_MEDIUM: 1, SEVERITY_LOW: 2, SEVERITY_NONE: 3}
    results = results.sort_values(
        by=["status", "severity", "affected_rows", "check_name"],
        key=lambda column: (
            column.map(status_rank)
            if column.name == "status"
            else column.map(severity_rank)
            if column.name == "severity"
            else -column
            if column.name == "affected_rows"
            else column
        ),
        kind="stable",
    )
    return results.reset_index(drop=True)


def summarize_quality_checks(results: pd.DataFrame) -> dict[str, object]:
    """Condense a results frame into headline numbers for the dashboard."""
    counts = results["status"].value_counts()
    failed = int(counts.get(STATUS_FAIL, 0))
    warned = int(counts.get(STATUS_WARN, 0))
    passed = int(counts.get(STATUS_PASS, 0))
    total = int(len(results))
    return {
        "total_checks": total,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        # Warnings count as half a failure: they need review, not remediation.
        "pass_rate": (passed + 0.5 * warned) / total if total else float("nan"),
        "blocking_checks": results.loc[results["status"].eq(STATUS_FAIL), "check_name"].tolist(),
        "review_checks": results.loc[results["status"].eq(STATUS_WARN), "check_name"].tolist(),
    }


if __name__ == "__main__":
    report = run_quality_checks()
    summary = summarize_quality_checks(report)

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)

    print("MetricGuard AI - data quality and observability report")
    print("=" * 78)
    print(
        report[
            [
                "check_name",
                "check_type",
                "table_name",
                "status",
                "severity",
                "affected_rows",
                "observed_value",
            ]
        ].to_string(index=False)
    )
    print()
    print("Summary")
    print("-" * 78)
    for key, value in summary.items():
        print(f"{key}: {value}")
