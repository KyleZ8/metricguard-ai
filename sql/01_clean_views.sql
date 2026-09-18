-- Deduplication and validated views (CC4).
--
-- Run against DuckDB with the loaded tables registered under these exact
-- names: transactions, accounts, complaints, account_monthly_snapshot
-- (src/sql_engine.py does this from the same tables quality_checks.load_tables()
-- returns, so the SQL layer sees the same data the pandas engine does, in
-- either dataset size).
--
-- transactions_deduped mirrors metric_engine.deduplicate_transactions():
-- a replayed dispute-platform file creates duplicate rows that share
-- source_transaction_id but not transaction_id. Keep one row per
-- source_transaction_id -- the earliest created_at, ties broken by
-- transaction_id so the result never depends on row order.
CREATE OR REPLACE VIEW transactions_deduped AS
SELECT * EXCLUDE (_rn)
FROM (
    SELECT
        t.*,
        ROW_NUMBER() OVER (
            PARTITION BY source_transaction_id
            ORDER BY created_at ASC, transaction_id ASC
        ) AS _rn
    FROM transactions t
)
WHERE _rn = 1;
