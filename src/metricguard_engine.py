"""Deterministic analysis engine for MetricGuard AI.

This module computes the factual evidence that the LLM layer will later explain.
It intentionally keeps metric calculation, data-quality checks, anomaly flags,
and driver analysis outside the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"


@dataclass(frozen=True)
class MetricResult:
    metric_name: str
    current_period: str
    previous_period: str
    current_value: float
    previous_value: float
    absolute_change: float
    percent_change: float
    current_numerator: float
    current_denominator: float
    previous_numerator: float
    previous_denominator: float


def load_synthetic_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all synthetic tables."""
    return {
        "accounts": pd.read_csv(data_dir / "accounts.csv", parse_dates=["open_date"]),
        "snapshots": pd.read_csv(data_dir / "account_monthly_snapshot.csv"),
        "transactions": pd.read_csv(
            data_dir / "transactions.csv",
            parse_dates=["transaction_date", "posted_date", "created_at"],
        ),
        "complaints": pd.read_csv(data_dir / "complaints.csv", parse_dates=["date_received"]),
        "metric_definitions": pd.read_csv(data_dir / "metric_definitions.csv"),
    }


def add_month(df: pd.DataFrame, date_col: str, month_col: str = "month") -> pd.DataFrame:
    out = df.copy()
    out[month_col] = out[date_col].dt.to_period("M").astype(str)
    return out


def deduplicate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Remove replayed upstream events using source_transaction_id."""
    return transactions.sort_values("created_at").drop_duplicates("source_transaction_id", keep="first")


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return float("nan")
    return float(numerator / denominator)


def monthly_dispute_rate(transactions: pd.DataFrame, deduped: bool = False) -> pd.DataFrame:
    tx = deduplicate_transactions(transactions) if deduped else transactions.copy()
    purchase = tx[tx["transaction_type"] == "purchase"].copy()
    purchase = add_month(purchase, "transaction_date")
    grouped = purchase.groupby("month").agg(
        numerator=("is_disputed", "sum"),
        denominator=("transaction_id", "count"),
        unique_source_events=("source_transaction_id", "nunique"),
    )
    grouped["metric_value"] = grouped["numerator"] / grouped["denominator"]
    grouped["metric_name"] = "dispute_rate"
    grouped["deduped"] = deduped
    return grouped.reset_index()


def monthly_payment_failure_rate(transactions: pd.DataFrame) -> pd.DataFrame:
    payments = transactions[transactions["transaction_type"] == "payment"].copy()
    payments = add_month(payments, "transaction_date")
    grouped = payments.groupby("month").agg(
        numerator=("payment_failed", "sum"),
        denominator=("transaction_id", "count"),
    )
    grouped["metric_value"] = grouped["numerator"] / grouped["denominator"]
    grouped["metric_name"] = "payment_failure_rate"
    return grouped.reset_index()


def monthly_complaint_rate(complaints: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    comp = add_month(complaints, "date_received")
    comp_count = comp.groupby("month").agg(numerator=("complaint_id", "count"))
    active = snapshots[snapshots["active_flag"] == 1].groupby("snapshot_month").agg(
        denominator=("account_id", "nunique")
    )
    grouped = comp_count.join(active, how="left")
    grouped["metric_value"] = grouped["numerator"] / grouped["denominator"]
    grouped["metric_name"] = "complaint_rate"
    return grouped.reset_index(names="month")


def monthly_delinquency_rate_balance(snapshots: pd.DataFrame) -> pd.DataFrame:
    active = snapshots[snapshots["active_flag"] == 1].copy()
    grouped = active.groupby("snapshot_month").agg(
        numerator=("statement_balance", lambda s: s[active.loc[s.index, "is_30dpd"] == 1].sum()),
        denominator=("statement_balance", "sum"),
    )
    grouped["metric_value"] = grouped["numerator"] / grouped["denominator"]
    grouped["metric_name"] = "delinquency_rate_30dpd_balance"
    return grouped.reset_index(names="month")


def metric_period_result(metric_df: pd.DataFrame, metric_name: str, current_period: str, previous_period: str) -> MetricResult:
    metric = metric_df.set_index("month")
    cur = metric.loc[current_period]
    prev = metric.loc[previous_period]
    current_value = float(cur["metric_value"])
    previous_value = float(prev["metric_value"])
    absolute_change = current_value - previous_value
    percent_change = absolute_change / previous_value if previous_value else float("nan")
    return MetricResult(
        metric_name=metric_name,
        current_period=current_period,
        previous_period=previous_period,
        current_value=current_value,
        previous_value=previous_value,
        absolute_change=absolute_change,
        percent_change=float(percent_change),
        current_numerator=float(cur["numerator"]),
        current_denominator=float(cur["denominator"]),
        previous_numerator=float(prev["numerator"]),
        previous_denominator=float(prev["denominator"]),
    )


def data_quality_checks(transactions: pd.DataFrame, complaints: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Run transparent data-quality checks for the demo."""
    checks = []

    duplicate_source_count = int(transactions["source_transaction_id"].duplicated().sum())
    checks.append(
        {
            "check_name": "duplicate_source_transactions",
            "status": "fail" if duplicate_source_count else "pass",
            "severity": "high" if duplicate_source_count else "none",
            "affected_rows": duplicate_source_count,
            "explanation": "Upstream source_transaction_id appears more than once, suggesting a replayed file or duplicated source event.",
        }
    )

    missing_category_count = int(transactions["merchant_category"].fillna("").eq("").sum())
    checks.append(
        {
            "check_name": "missing_merchant_category",
            "status": "fail" if missing_category_count else "pass",
            "severity": "medium" if missing_category_count else "none",
            "affected_rows": missing_category_count,
            "explanation": "Merchant category is required for driver analysis and may be missing from a card-processor batch.",
        }
    )

    invalid_posted_dates = int((transactions["posted_date"] < transactions["transaction_date"]).sum())
    checks.append(
        {
            "check_name": "posted_before_transaction_date",
            "status": "fail" if invalid_posted_dates else "pass",
            "severity": "high" if invalid_posted_dates else "none",
            "affected_rows": invalid_posted_dates,
            "explanation": "Posted date should not be earlier than transaction date.",
        }
    )

    invalid_amounts = int(
        (
            (transactions["transaction_type"].eq("purchase") & transactions["transaction_amount"].le(0))
            | (transactions["transaction_type"].eq("fee") & transactions["transaction_amount"].le(0))
        ).sum()
    )
    checks.append(
        {
            "check_name": "invalid_purchase_or_fee_amount",
            "status": "fail" if invalid_amounts else "pass",
            "severity": "high" if invalid_amounts else "none",
            "affected_rows": invalid_amounts,
            "explanation": "Purchase and fee amounts should be positive.",
        }
    )

    tx = add_month(transactions, "created_at", "load_month")
    monthly_rows = tx.groupby("load_month").size()
    row_spike = 0
    if len(monthly_rows) >= 3:
        baseline = monthly_rows.iloc[:-1]
        threshold = baseline.mean() + 3 * baseline.std(ddof=0)
        row_spike = int(monthly_rows.iloc[-1] > threshold)
    checks.append(
        {
            "check_name": "monthly_ingestion_row_count_spike",
            "status": "warn" if row_spike else "pass",
            "severity": "medium" if row_spike else "none",
            "affected_rows": int(monthly_rows.iloc[-1]) if row_spike else 0,
            "explanation": "Latest monthly ingestion volume is unusually high relative to recent history.",
        }
    )

    complaint_missing_narrative = int(complaints["complaint_narrative"].fillna("").str.strip().eq("").sum())
    checks.append(
        {
            "check_name": "missing_complaint_narrative",
            "status": "warn" if complaint_missing_narrative else "pass",
            "severity": "low" if complaint_missing_narrative else "none",
            "affected_rows": complaint_missing_narrative,
            "explanation": "Complaint narrative is useful for NLP theme extraction.",
        }
    )

    invalid_snapshot_balance = int((snapshots["statement_balance"] < 0).sum())
    checks.append(
        {
            "check_name": "negative_statement_balance",
            "status": "fail" if invalid_snapshot_balance else "pass",
            "severity": "high" if invalid_snapshot_balance else "none",
            "affected_rows": invalid_snapshot_balance,
            "explanation": "Statement balances should not be negative in this synthetic credit-risk snapshot.",
        }
    )

    return pd.DataFrame(checks)


def rolling_anomaly_flags(metric_df: pd.DataFrame, window: int = 4, z_threshold: float = 2.0) -> pd.DataFrame:
    """Flag simple time-series anomalies using prior rolling mean/std."""
    out = metric_df.sort_values("month").copy()
    out["baseline_mean"] = out["metric_value"].shift(1).rolling(window=window, min_periods=3).mean()
    out["baseline_std"] = out["metric_value"].shift(1).rolling(window=window, min_periods=3).std(ddof=0)
    out["z_score"] = (out["metric_value"] - out["baseline_mean"]) / out["baseline_std"]
    out["is_anomaly"] = out["z_score"].abs() >= z_threshold
    return out


def segment_driver_analysis(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    current_period: str = "2026-08",
    previous_period: str = "2026-07",
    segments: Iterable[str] = ("merchant_category", "channel", "fico_band", "customer_segment", "product_type", "region"),
) -> pd.DataFrame:
    """Find segments contributing to a dispute-rate movement."""
    tx = deduplicate_transactions(transactions)
    purchase = tx[tx["transaction_type"] == "purchase"].copy()
    purchase = add_month(purchase, "transaction_date")
    purchase = purchase.merge(
        accounts[["account_id", "fico_band", "customer_segment", "product_type", "region"]],
        on="account_id",
        how="left",
    )

    rows = []
    for segment in segments:
        grouped = (
            purchase[purchase["month"].isin([current_period, previous_period])]
            .groupby(["month", segment], dropna=False)
            .agg(disputed=("is_disputed", "sum"), purchases=("transaction_id", "count"))
            .reset_index()
        )
        pivot = grouped.pivot(index=segment, columns="month", values=["disputed", "purchases"]).fillna(0)
        for value in (previous_period, current_period):
            if ("disputed", value) not in pivot:
                pivot[("disputed", value)] = 0
            if ("purchases", value) not in pivot:
                pivot[("purchases", value)] = 0
        pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
        pivot = pivot.reset_index().rename(columns={segment: "segment_value"})
        pivot["segment_name"] = segment
        pivot["previous_rate"] = pivot.apply(
            lambda r: _safe_rate(r[f"disputed_{previous_period}"], r[f"purchases_{previous_period}"]),
            axis=1,
        )
        pivot["current_rate"] = pivot.apply(
            lambda r: _safe_rate(r[f"disputed_{current_period}"], r[f"purchases_{current_period}"]),
            axis=1,
        )
        pivot["disputed_change"] = pivot[f"disputed_{current_period}"] - pivot[f"disputed_{previous_period}"]
        pivot["purchase_change"] = pivot[f"purchases_{current_period}"] - pivot[f"purchases_{previous_period}"]
        rows.append(
            pivot[
                [
                    "segment_name",
                    "segment_value",
                    f"disputed_{previous_period}",
                    f"disputed_{current_period}",
                    "disputed_change",
                    f"purchases_{previous_period}",
                    f"purchases_{current_period}",
                    "purchase_change",
                    "previous_rate",
                    "current_rate",
                ]
            ]
        )

    drivers = pd.concat(rows, ignore_index=True)
    total_positive_change = drivers.loc[drivers["disputed_change"] > 0, "disputed_change"].sum()
    drivers["contribution_share_of_positive_change"] = np.where(
        drivers["disputed_change"] > 0,
        drivers["disputed_change"] / total_positive_change if total_positive_change else 0,
        0,
    )
    return drivers.sort_values("disputed_change", ascending=False).reset_index(drop=True)


def complaint_theme_summary(complaints: pd.DataFrame, current_period: str = "2026-08", previous_period: str = "2026-07") -> pd.DataFrame:
    """Summarize complaint issue movement before deeper NLP is added."""
    comp = add_month(complaints, "date_received")
    grouped = (
        comp[comp["month"].isin([current_period, previous_period])]
        .groupby(["month", "issue"])
        .agg(complaints=("complaint_id", "count"))
        .reset_index()
    )
    pivot = grouped.pivot(index="issue", columns="month", values="complaints").fillna(0)
    for value in (previous_period, current_period):
        if value not in pivot:
            pivot[value] = 0
    pivot = pivot.reset_index().rename(columns={previous_period: "previous_complaints", current_period: "current_complaints"})
    pivot["complaint_change"] = pivot["current_complaints"] - pivot["previous_complaints"]
    pivot["percent_change"] = pivot.apply(
        lambda r: r["complaint_change"] / r["previous_complaints"] if r["previous_complaints"] else np.nan,
        axis=1,
    )
    return pivot.sort_values("complaint_change", ascending=False).reset_index(drop=True)


def build_investigation_summary(current_period: str = "2026-08", previous_period: str = "2026-07") -> dict[str, object]:
    data = load_synthetic_data()
    raw_metric = monthly_dispute_rate(data["transactions"], deduped=False)
    deduped_metric = monthly_dispute_rate(data["transactions"], deduped=True)
    raw_result = metric_period_result(raw_metric, "raw_dispute_rate", current_period, previous_period)
    deduped_result = metric_period_result(deduped_metric, "deduped_dispute_rate", current_period, previous_period)
    anomaly = rolling_anomaly_flags(deduped_metric)
    quality = data_quality_checks(data["transactions"], data["complaints"], data["snapshots"])
    drivers = segment_driver_analysis(data["transactions"], data["accounts"], current_period, previous_period)
    themes = complaint_theme_summary(data["complaints"], current_period, previous_period)
    return {
        "raw_result": raw_result,
        "deduped_result": deduped_result,
        "metric_trend": deduped_metric,
        "anomaly": anomaly,
        "quality_checks": quality,
        "drivers": drivers,
        "complaint_themes": themes,
    }


if __name__ == "__main__":
    summary = build_investigation_summary()
    print(summary["raw_result"])
    print(summary["deduped_result"])
    print(summary["quality_checks"])
    print(summary["drivers"].head(10))
    print(summary["complaint_themes"].head(10))
