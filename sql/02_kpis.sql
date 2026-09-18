-- Monthly KPIs: dispute rate, fraud claim rate, payment failure rate (CC4).
--
-- Depends on sql/01_clean_views.sql (transactions_deduped) having been run
-- first in the same DuckDB connection.
--
-- Each KPI is produced twice: _raw (as first reported, no correction) and
-- _corrected (against transactions_deduped) for dispute_rate, matching
-- metric_engine.monthly_dispute_rate(deduped=False/True). fraud_claim_rate
-- and payment_failure_rate are corrected-only, matching how the pandas
-- engine reports them (metric_engine._transaction_metric_source always
-- deduplicates for every metric except the raw side of dispute_rate).

CREATE OR REPLACE VIEW monthly_dispute_rate_raw AS
SELECT
    strftime(transaction_date, '%Y-%m') AS month,
    SUM(CASE WHEN is_disputed = 1 THEN 1 ELSE 0 END) AS numerator,
    COUNT(*) AS denominator,
    CAST(SUM(CASE WHEN is_disputed = 1 THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS metric_value
FROM transactions
WHERE transaction_type = 'purchase'
GROUP BY 1
ORDER BY 1;

CREATE OR REPLACE VIEW monthly_dispute_rate_corrected AS
SELECT
    strftime(transaction_date, '%Y-%m') AS month,
    SUM(CASE WHEN is_disputed = 1 THEN 1 ELSE 0 END) AS numerator,
    COUNT(*) AS denominator,
    CAST(SUM(CASE WHEN is_disputed = 1 THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS metric_value
FROM transactions_deduped
WHERE transaction_type = 'purchase'
GROUP BY 1
ORDER BY 1;

CREATE OR REPLACE VIEW monthly_fraud_claim_rate AS
SELECT
    strftime(transaction_date, '%Y-%m') AS month,
    SUM(CASE WHEN is_fraud_claim = 1 THEN 1 ELSE 0 END) AS numerator,
    COUNT(*) AS denominator,
    CAST(SUM(CASE WHEN is_fraud_claim = 1 THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS metric_value
FROM transactions_deduped
WHERE transaction_type = 'purchase'
GROUP BY 1
ORDER BY 1;

CREATE OR REPLACE VIEW monthly_payment_failure_rate AS
SELECT
    strftime(transaction_date, '%Y-%m') AS month,
    SUM(CASE WHEN payment_failed = 1 THEN 1 ELSE 0 END) AS numerator,
    COUNT(*) AS denominator,
    CAST(SUM(CASE WHEN payment_failed = 1 THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS metric_value
FROM transactions_deduped
WHERE transaction_type = 'payment'
GROUP BY 1
ORDER BY 1;
