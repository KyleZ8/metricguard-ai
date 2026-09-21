-- Segment KPIs: corrected dispute rate by merchant category, channel, and
-- their combination (CC4).
--
-- Depends on sql/01_clean_views.sql (transactions_deduped) having been run
-- first in the same DuckDB connection. All corrected (deduplicated) --
-- driver_analysis.py routes every segment view through
-- metric_engine.deduplicate_transactions() for the same reason.
--
-- A blank merchant_category (the planted card-processor defect, ~1,384
-- purchases on full data) is normalized to the literal string '__missing__'
-- rather than dropped or left blank, matching driver_analysis.MISSING_LABEL:
-- silently discarding it would shrink a denominator the headline KPI still
-- counts. TRIM/NULLIF/COALESCE together treat both an empty string and NULL
-- as missing, since the same logical blank value round-trips differently
-- through CSV (NULL) and Parquet (empty string) -- see .ai/PARITY_CC3.md.

CREATE OR REPLACE VIEW segment_dispute_rate_merchant_category AS
SELECT
    strftime(transaction_date, '%Y-%m') AS month,
    COALESCE(NULLIF(TRIM(merchant_category), ''), '__missing__') AS segment_value,
    SUM(CASE WHEN is_disputed = 1 THEN 1 ELSE 0 END) AS numerator,
    COUNT(*) AS denominator
FROM transactions_deduped
WHERE transaction_type = 'purchase'
GROUP BY 1, 2
ORDER BY 1, 2;

CREATE OR REPLACE VIEW segment_dispute_rate_channel AS
SELECT
    strftime(transaction_date, '%Y-%m') AS month,
    channel AS segment_value,
    SUM(CASE WHEN is_disputed = 1 THEN 1 ELSE 0 END) AS numerator,
    COUNT(*) AS denominator
FROM transactions_deduped
WHERE transaction_type = 'purchase'
GROUP BY 1, 2
ORDER BY 1, 2;

CREATE OR REPLACE VIEW segment_dispute_rate_merchant_category_channel AS
SELECT
    strftime(transaction_date, '%Y-%m') AS month,
    COALESCE(NULLIF(TRIM(merchant_category), ''), '__missing__') AS merchant_category,
    channel,
    SUM(CASE WHEN is_disputed = 1 THEN 1 ELSE 0 END) AS numerator,
    COUNT(*) AS denominator
FROM transactions_deduped
WHERE transaction_type = 'purchase'
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;
