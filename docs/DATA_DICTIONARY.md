# Synthetic Data Dictionary

The project will use synthetic data. The data should be realistic enough to support a fictional-card-issuer-style Senior Data Analyst project, but it must not contain real customer data.

## Dataset Design

Use four synthetic data tables:

1. `accounts.csv`
2. `account_monthly_snapshot.csv`
3. `transactions.csv`
4. `complaints.csv`

And one metric governance table:

5. `metric_definitions.csv`

## accounts.csv

One row per synthetic credit-card account.

Fields:

- `account_id`
  - Synthetic account identifier.
- `open_date`
  - Date the account opened.
- `product_type`
  - Example values: `cash_rewards`, `travel_rewards`, `student_card`, `secured_card`, `venture_style`.
- `customer_segment`
  - Example values: `student`, `young_professional`, `mass_market`, `affluent`.
- `fico_band`
  - Example values: `>660`, `<=660`.
- `region`
  - Example values: `Northeast`, `South`, `Midwest`, `West`.
- `state`
  - Two-letter state abbreviation.
- `channel_origin`
  - Example values: `branch`, `web`, `mobile`, `partner`.
- `active_flag`
  - 1 if active, 0 otherwise.
- `current_balance`
  - Synthetic current card balance.
- `credit_limit`
  - Synthetic credit limit.

## account_monthly_snapshot.csv

One row per synthetic credit-card account per month.

Fields:

- `snapshot_month`
  - Month of the account snapshot.
- `account_id`
  - Joins to `accounts.csv`.
- `active_flag`
  - 1 if the account is active in the month, 0 otherwise.
- `statement_balance`
  - Synthetic period-end statement balance.
- `credit_limit`
  - Credit limit for the account.
- `utilization_rate`
  - Statement balance divided by credit limit.
- `days_past_due`
  - Synthetic days past due.
- `is_30dpd`
  - 1 if account is at least 30 days past due, 0 otherwise.
- `charge_off_balance`
  - Synthetic charged-off balance proxy.
- `source_system`
  - Example value: `card_servicing_platform`.
- `snapshot_loaded_at`
  - Timestamp when snapshot was loaded.

Purpose:

- Compute balance-based 30+ day delinquency proxy.
- Compute charge-off proxy.
- Provide realistic denominator context for credit-risk KPIs.
- Support risk segmentation by FICO band, product, and customer segment.

## transactions.csv

One row per synthetic card transaction or account event.

Fields:

- `transaction_id`
  - Synthetic row-level transaction identifier.
- `source_transaction_id`
  - Synthetic upstream event identifier. This is used to detect replayed/duplicated source events.
- `account_id`
  - Joins to `accounts.csv`.
- `transaction_date`
  - Date of transaction.
- `posted_date`
  - Date transaction posted.
- `merchant_category`
  - Example values: `travel`, `grocery`, `restaurant`, `gas`, `online_retail`, `subscription`, `health`, `education`, `fee`, `payment`.
- `merchant_name`
  - Synthetic merchant descriptor.
- `channel`
  - Example values: `card_present`, `web`, `mobile`, `recurring`, `system`.
- `transaction_amount`
  - Purchase, fee, refund, or payment amount.
- `transaction_type`
  - Example values: `purchase`, `fee`, `payment`, `refund`.
- `is_disputed`
  - 1 if disputed, 0 otherwise.
- `is_fraud_claim`
  - 1 if fraud claim, 0 otherwise.
- `payment_failed`
  - 1 if related payment failed, 0 otherwise.
- `ingestion_batch_id`
  - Simulates data pipeline batch.
- `source_system`
  - Example values: `card_processor`, `dispute_platform`, `payment_gateway`.
- `created_at`
  - Ingestion timestamp.

Purpose:

- Compute dispute rate.
- Compute fraud claim rate.
- Compute payment failure rate.
- Compute purchase volume.
- Create data-quality issues through duplicate source IDs, missing categories, date gaps, and batch spikes.

## complaints.csv

One row per synthetic customer complaint.

Fields inspired by CFPB complaint fields:

- `complaint_id`
  - Synthetic complaint identifier.
- `account_id`
  - Joins to `accounts.csv`.
- `date_received`
  - Date complaint was received.
- `product`
  - Example value: `Credit card`.
- `sub_product`
  - Example value: `General-purpose credit card or charge card`.
- `issue`
  - Example values: `Problem with a purchase shown on your statement`, `Fees or interest`, `Problem when making payments`, `Problem with fraud alerts or security`.
- `sub_issue`
  - More specific complaint category.
- `complaint_narrative`
  - Synthetic customer text.
- `submitted_via`
  - Example values: `Web`, `Phone`, `Mobile app`, `Referral`.
- `company_response`
  - Example values: `Closed with explanation`, `Closed with monetary relief`, `In progress`.
- `timely_response`
  - `Yes` or `No`.
- `merchant_category`
  - Optional link to merchant category.
- `channel`
  - Optional customer channel.

Purpose:

- Compute complaint rate.
- Compute dispute-related complaint share.
- Compute fee-related complaint share.
- Extract complaint themes with NLP.
- Provide evidence for LLM explanations.

## metric_definitions.csv

One row per KPI.

Fields:

- `metric_name`
- `business_definition`
- `numerator`
- `denominator`
- `grain`
- `refresh_frequency`
- `owner`
- `source_tables`
- `known_limitations`

Purpose:

> Supports the data-governance story: metadata, lineage, business definitions, and well-managed metrics.

## First Demo KPI Formulas

### dispute_rate

```text
dispute_rate = disputed_transactions / total_purchase_transactions
```

Use this as the main demo metric.

### fraud_claim_rate

```text
fraud_claim_rate = fraud_claim_transactions / total_purchase_transactions
```

### payment_failure_rate

```text
payment_failure_rate = failed_payments / total_payment_attempts
```

### delinquency_rate_30dpd

Preferred synthetic balance-based version:

```text
delinquency_rate_30dpd_balance =
sum(statement_balance where is_30dpd = 1) / sum(statement_balance for active accounts)
```

Simpler account-count version:

```text
delinquency_rate_30dpd = accounts_with_days_past_due_30plus / active_accounts
```

Note:

Public a fictional card issuer disclosures use balance-based delinquency rates. The project should use the balance-based version in the main demo and keep the account-count version only as an explanatory backup.

### purchase_volume

```text
purchase_volume = sum(transaction_amount where transaction_type = purchase)
```

### complaint_rate

```text
complaint_rate = total_complaints / active_accounts
```

### fee_complaint_share

```text
fee_complaint_share = fee_related_complaints / total_complaints
```

## Planned Synthetic Anomalies

Include at least one real business anomaly:

- elevated dispute rate in travel merchant category
- increase concentrated in mobile channel
- stronger effect in `<=660` FICO segment
- complaint narratives mention foreign transaction fees, unrecognized merchants, and duplicate charges

Include at least one data-quality anomaly:

- duplicate dispute records in one ingestion batch
- missing merchant category for a subset of records
- delayed posted dates for one source system
- row-count spike caused by file replay

## Current Generated Dataset Profile

Generated by:

```text
src/generate_synthetic_data.py
```

Current row counts:

- `accounts.csv`: 12,000 rows
- `account_monthly_snapshot.csv`: 96,000 rows
- `transactions.csv`: 929,838 rows
- `complaints.csv`: 5,300 rows
- `metric_definitions.csv`: 7 rows

Current planted defects:

- 165 duplicate upstream source transactions from `DISPUTE_PLATFORM_20260818_REPLAY_01`
- 1,384 missing merchant categories from a card-processor batch

Current planted business signal:

- August 2026 dispute rate rises from a roughly 1.3% baseline to 1.77% raw.
- After deduplicating replayed source transactions, August 2026 dispute rate remains elevated at 1.61%.
- This supports the intended story: data quality inflated the alert, but a real business movement remains.
