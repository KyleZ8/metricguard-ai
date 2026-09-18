"""Parity check: SQL results (sql_engine.py) vs. the pandas engine (CC4).

Proves the DuckDB queries in sql/ produce the same numbers as the existing
pandas computations in metric_engine.py / driver_analysis.py, on the same
loaded tables. Compares, per amendment: monthly dispute rate (raw and
corrected), fraud claim rate, payment failure rate, and the corrected
dispute-rate segment breakdown by merchant_category, channel, and their
interaction.

Exits non-zero and prints every mismatch if anything differs; exits 0 and
prints a one-line summary per check if everything matches.

Run with::

    python src/verify_sql_parity.py
"""

from __future__ import annotations

import sys

import pandas as pd

from driver_analysis import corrected_purchase_facts
from metric_engine import generic_monthly_trend_table, monthly_dispute_rate, resolve_tables
from quality_checks import TABLE_TRANSACTIONS
from sql_engine import (
    build_connection,
    monthly_dispute_rate_sql,
    monthly_kpi_sql,
    segment_dispute_rate_sql,
    segment_interaction_dispute_rate_sql,
)


def _compare(label: str, pandas_frame: pd.DataFrame, sql_frame: pd.DataFrame, keys: list[str], value_cols: list[str]) -> list[str]:
    """Return a list of human-readable mismatch descriptions (empty if none)."""
    left = pandas_frame[keys + value_cols].sort_values(keys).reset_index(drop=True)
    right = sql_frame[keys + value_cols].sort_values(keys).reset_index(drop=True)

    problems: list[str] = []
    if len(left) != len(right):
        problems.append(f"{label}: row count differs (pandas={len(left)}, sql={len(right)})")
        return problems

    for col in value_cols:
        left_vals = left[col].astype(float).to_numpy()
        right_vals = right[col].astype(float).to_numpy()
        if not (left_vals == right_vals).all():
            import numpy as np

            close = np.isclose(left_vals, right_vals, rtol=0, atol=1e-9)
            if close.all():
                continue
            bad_idx = [i for i, ok in enumerate(close) if not ok]
            for i in bad_idx[:5]:
                key_repr = {k: left.loc[i, k] for k in keys}
                problems.append(
                    f"{label}: {col} differs at {key_repr}: pandas={left_vals[i]!r} sql={right_vals[i]!r}"
                )
    return problems


def verify() -> list[str]:
    tables = resolve_tables()
    transactions = tables[TABLE_TRANSACTIONS]
    con = build_connection(tables)

    problems: list[str] = []

    # 1. Monthly dispute rate, raw and corrected.
    for deduped in (False, True):
        pandas_frame = monthly_dispute_rate(transactions, deduped=deduped).rename(
            columns={
                "disputed_purchases": "numerator",
                "purchase_transactions": "denominator",
                "dispute_rate": "metric_value",
            }
        )
        sql_frame = monthly_dispute_rate_sql(con, deduped=deduped)
        label = f"monthly_dispute_rate(deduped={deduped})"
        problems += _compare(label, pandas_frame, sql_frame, ["month"], ["numerator", "denominator", "metric_value"])

    # 2. Monthly fraud claim rate and payment failure rate, via the generic
    #    trend table (same corrected-transactions source the pandas engine
    #    uses for every finance KPI except raw dispute_rate).
    for metric_name in ("fraud_claim_rate", "payment_failure_rate"):
        pandas_frame = generic_monthly_trend_table(tables, metric_name).rename(
            columns={
                "corrected_numerator": "numerator",
                "corrected_denominator": "denominator",
                "corrected_value": "metric_value",
            }
        )
        sql_frame = monthly_kpi_sql(con, metric_name)
        problems += _compare(
            f"monthly_{metric_name}", pandas_frame, sql_frame, ["month"], ["numerator", "denominator", "metric_value"]
        )

    # 3. Segment dispute rate by merchant_category and by channel: compare
    #    per-month, per-segment-value disputed/purchase counts against
    #    driver_analysis.corrected_purchase_facts() aggregated the same way.
    facts = corrected_purchase_facts(tables=tables)
    for field in ("merchant_category", "channel"):
        pandas_frame = (
            facts.groupby(["month", field])
            .agg(numerator=("is_disputed", "sum"), denominator=("transaction_id", "count"))
            .reset_index()
            .rename(columns={field: "segment_value"})
        )
        sql_frame = segment_dispute_rate_sql(con, field)
        problems += _compare(
            f"segment_dispute_rate[{field}]", pandas_frame, sql_frame, ["month", "segment_value"], ["numerator", "denominator"]
        )

    # 4. merchant_category x channel interaction.
    pandas_interaction = (
        facts.groupby(["month", "merchant_category", "channel"])
        .agg(numerator=("is_disputed", "sum"), denominator=("transaction_id", "count"))
        .reset_index()
    )
    sql_interaction = segment_interaction_dispute_rate_sql(con)
    problems += _compare(
        "segment_dispute_rate[merchant_category x channel]",
        pandas_interaction,
        sql_interaction,
        ["month", "merchant_category", "channel"],
        ["numerator", "denominator"],
    )

    con.close()
    return problems


if __name__ == "__main__":
    checks = [
        "monthly_dispute_rate(deduped=False)",
        "monthly_dispute_rate(deduped=True)",
        "monthly_fraud_claim_rate",
        "monthly_payment_failure_rate",
        "segment_dispute_rate[merchant_category]",
        "segment_dispute_rate[channel]",
        "segment_dispute_rate[merchant_category x channel]",
    ]
    found = verify()
    if found:
        print(f"SQL/pandas PARITY FAILED — {len(found)} mismatch(es):")
        for line in found:
            print(f"  - {line}")
        sys.exit(1)
    print(f"SQL/pandas parity OK — {len(checks)} checks, all exact matches:")
    for check in checks:
        print(f"  - {check}")
