# SQL layer

DuckDB queries, run in order against the same tables `quality_checks.load_tables()`
returns (so they see whichever dataset `METRICGUARD_DATA` selected). Loaded and
executed by `src/sql_engine.py`; proven to match the pandas engine exactly by
`src/verify_sql_parity.py` (`python src/verify_sql_parity.py`).

| File | Owns |
|---|---|
| `01_clean_views.sql` | `transactions_deduped` — one row per `source_transaction_id`, the earliest `created_at` (matches `metric_engine.deduplicate_transactions`) |
| `02_kpis.sql` | Monthly dispute rate (raw + corrected), fraud claim rate, payment failure rate |
| `03_segments.sql` | Corrected dispute rate by `merchant_category`, `channel`, and their interaction |

`metric_engine.monthly_trend_table_sql()` is the callable integration point —
same output shape as the pandas `monthly_trend_table()`, computed through this
SQL layer instead.

## Complaint KPI definitions (amendments A4, A11)

`complaint_rate` and `fee_complaint_share` are computed in `metric_engine.py`
(`_complaint_monthly_trend`, `generic_segment_driver_table`'s `complaint_rate`
branch), not here — a KPI catalog note either way, since A4 asked for the
definition to be documented in both `sql/` and `docs/`:

- **`complaint_rate`** = complaints ÷ active accounts, **× 1,000** ("complaints
  per 1,000 active accounts"). Before this fix, the segment-level cut divided
  complaint counts by complaint counts (always 1.0) because it reused the same
  generic numerator/denominator machinery every other KPI uses, and complaints
  and their denominator (accounts) don't live in the same source table.
  Segment cuts are restricted to fields that exist at the account level
  (`product_type`, `customer_segment`, `fico_band`, `region`) — a
  complaint-only attribute like `issue` has no independent account
  population to divide by, so `complaint_rate` is not computed for those.
- **`fee_complaint_share`** = complaints with `issue = "Fees or interest"` ÷
  all complaints, by segment. The `issue` segment cut was removed (A11):
  every complaint in the `issue = "Fees or interest"` group is
  definitionally a fee complaint, so that one cut is tautological (1.0 for
  that value, 0.0 for every other issue) — it was measuring the grouping
  key against itself, not a real relationship.
