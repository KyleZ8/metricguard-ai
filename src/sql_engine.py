"""DuckDB SQL layer for MetricGuard AI (CC4).

Owns the ``sql/`` directory: deduplicated views, monthly KPIs, and segment
KPIs, run against DuckDB rather than pandas. The queries are hand-written SQL
(``sql/01_clean_views.sql``, ``02_kpis.sql``, ``03_segments.sql``), not
generated -- this module's job is only to register the already-loaded tables
into a DuckDB connection, run those files in order, and hand back the
resulting views as pandas DataFrames with the same shape the equivalent
``metric_engine``/``driver_analysis`` pandas function produces, so the two
can be compared directly (see :mod:`verify_sql_parity`).

Tables are registered from whatever ``quality_checks.load_tables()`` already
returned (full CSV or demo Parquet, dtypes already normalized), not read
fresh from disk here -- the SQL layer answers "does the same data give the
same answer in SQL," not "can SQL also parse CSV/Parquet."

Run directly to print every KPI/segment view::

    python src/sql_engine.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from config import PROJECT_ROOT
from quality_checks import TABLE_ACCOUNTS, TABLE_COMPLAINTS, TABLE_SNAPSHOTS, TABLE_TRANSACTIONS, load_tables

SQL_DIR = PROJECT_ROOT / "sql"

# DuckDB table name -> canonical table key (quality_checks.TABLE_*). The SQL
# files are written against these DuckDB names because they read like the
# real tables a warehouse would have, not the Python dict keys.
_DUCKDB_TABLE_NAMES = {
    "transactions": TABLE_TRANSACTIONS,
    "accounts": TABLE_ACCOUNTS,
    "complaints": TABLE_COMPLAINTS,
    "account_monthly_snapshot": TABLE_SNAPSHOTS,
}

_SQL_FILES = ("01_clean_views.sql", "02_kpis.sql", "03_segments.sql")


def build_connection(tables: dict[str, pd.DataFrame] | None = None) -> duckdb.DuckDBPyConnection:
    """Register the loaded tables and run every sql/ file. Views stay live on the connection."""
    if tables is None:
        tables = load_tables()

    con = duckdb.connect(database=":memory:")
    for duckdb_name, table_key in _DUCKDB_TABLE_NAMES.items():
        con.register(duckdb_name, tables[table_key])

    for filename in _SQL_FILES:
        sql_text = (SQL_DIR / filename).read_text(encoding="utf-8")
        con.execute(sql_text)

    return con


def monthly_dispute_rate_sql(con: duckdb.DuckDBPyConnection, deduped: bool) -> pd.DataFrame:
    """Same shape as metric_engine.monthly_dispute_rate(transactions, deduped=...)."""
    view = "monthly_dispute_rate_corrected" if deduped else "monthly_dispute_rate_raw"
    frame = con.execute(f"SELECT * FROM {view}").df()
    frame["metric_name"] = "dispute_rate"
    frame["deduped"] = deduped
    return frame


def monthly_kpi_sql(con: duckdb.DuckDBPyConnection, metric_name: str) -> pd.DataFrame:
    """One of dispute_rate (corrected), fraud_claim_rate, payment_failure_rate."""
    view = {
        "dispute_rate": "monthly_dispute_rate_corrected",
        "fraud_claim_rate": "monthly_fraud_claim_rate",
        "payment_failure_rate": "monthly_payment_failure_rate",
    }[metric_name]
    return con.execute(f"SELECT * FROM {view}").df()


def segment_dispute_rate_sql(con: duckdb.DuckDBPyConnection, segment_field: str) -> pd.DataFrame:
    """Corrected dispute rate by month x segment_value, for merchant_category or channel."""
    view = {
        "merchant_category": "segment_dispute_rate_merchant_category",
        "channel": "segment_dispute_rate_channel",
    }[segment_field]
    return con.execute(f"SELECT * FROM {view}").df()


def segment_interaction_dispute_rate_sql(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Corrected dispute rate by month x merchant_category x channel."""
    return con.execute("SELECT * FROM segment_dispute_rate_merchant_category_channel").df()


if __name__ == "__main__":
    connection = build_connection()

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print("Monthly dispute rate (raw)")
    print(monthly_dispute_rate_sql(connection, deduped=False).to_string(index=False))
    print()
    print("Monthly dispute rate (corrected)")
    print(monthly_dispute_rate_sql(connection, deduped=True).to_string(index=False))
    print()
    print("Monthly fraud claim rate")
    print(monthly_kpi_sql(connection, "fraud_claim_rate").to_string(index=False))
    print()
    print("Monthly payment failure rate")
    print(monthly_kpi_sql(connection, "payment_failure_rate").to_string(index=False))
    print()
    print("Segment dispute rate by merchant_category (2026-08, top 5 by numerator)")
    seg = segment_dispute_rate_sql(connection, "merchant_category")
    print(seg[seg["month"] == "2026-08"].sort_values("numerator", ascending=False).head(5).to_string(index=False))
