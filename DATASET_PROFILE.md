# Synthetic Dataset Profile

Generated on: September 6, 2026

Generator:

```text
src/generate_synthetic_data.py
```

Output folder:

```text
data/synthetic/
```

## Files Created

- `accounts.csv`
- `account_monthly_snapshot.csv`
- `transactions.csv`
- `complaints.csv`
- `metric_definitions.csv`
- `GROUND_TRUTH.md`

## Row Counts

- Accounts: 12,000
- Monthly account snapshots: 96,000
- Transactions: 929,838
- Complaints: 5,300
- Metric definitions: 7

## Why The Data Looks Realistic

The synthetic data includes multiple real-world characteristics that an analyst would expect in credit-card risk, disputes, payments, and customer operations data:

- account-level product mix
- FICO-band segmentation
- customer segments
- geographic regions and states
- acquisition channels
- monthly balance snapshots
- statement balances and utilization
- days-past-due fields
- transaction dates vs. posted dates
- merchant categories
- messy merchant descriptors
- source systems
- ingestion batch IDs
- duplicate upstream source events
- missing categories from a bad batch
- customer complaint narratives written from each row's own context
- CFPB-style complaint fields
- metric definitions with numerator, denominator, owner, grain, and limitations

## Capital One-Style KPI Story

The first demo focuses on a credit-card dispute-rate alert.

Alert:

> August 2026 dispute rate is unusually high.

What the data contains:

1. A data-quality issue:
   - a dispute-platform replay batch creates duplicate upstream source events

2. A real business movement:
   - even after deduplication, dispute rate remains elevated

3. Business drivers:
   - travel merchant activity
   - mobile channel
   - FICO-band differences
   - student and young-professional segments

4. Customer text signal:
   - duplicate-looking travel charges
   - mobile dispute submission friction
   - unclear merchant descriptors
   - dispute status confusion
   - foreign-transaction-fee and failed-autopay themes are present, but not August growth drivers

## Complaint Narrative Alignment

Complaint narratives are written from the seed row's `merchant_category`, `channel`,
`transaction_type`, `merchant_name`, `payment_failed` and `is_fraud_claim`, not from
`issue` alone.

This matters for the dashboard's evidence table. Under the earlier generator a grocery
dispute could be described as "the same travel charge appears twice", so a representative
complaint could contradict the structured columns shown beside it.

Current alignment in the generated corpus:

- travel wording appears on 459 of 533 travel-category complaints
- travel wording appears on 0 non-travel purchase complaints; the only non-travel rows
  carrying it are 201 `fee` rows whose narrative explains a foreign transaction fee, where
  international spending is the actual cause
- foreign-transaction-fee wording appears only on `FOREIGN TRANSACTION FEE` rows
- autopay wording appears only on failed mobile payments
- app wording is concentrated on mobile-channel rows

Some deliberate messiness remains: about 6% of complaints are written vaguely
("I contacted support but still do not understand this account activity"). These stay
generic, so they never contradict their row, and they give the theme classifier a
realistic `unclear_or_other` population instead of an unrealistically tidy corpus.

## Dispute Rate Pattern

The planted monthly pattern is recorded in:

```text
data/synthetic/GROUND_TRUTH.md
```

Current pattern:

- January to July baseline: around 1.3% to 1.4%
- August raw dispute rate: 1.77%
- August deduped dispute rate: 1.61%

Interpretation:

> The replayed dispute batch inflated the KPI, but did not fully explain the spike.

This creates a strong demo because the tool can say:

> Part of the dashboard alert is bad data. Part of it is a real customer/risk issue.

## Data Governance Story

`metric_definitions.csv` supports the Capital One SDA story around business definitions, metric ownership, metadata, and lineage.

Each metric includes:

- metric name
- business definition
- numerator
- denominator
- grain
- refresh frequency
- owner
- source tables
- known limitations

This lets the final app show that the project is not just analytics, but well-managed analytics.

## Safety And Privacy

All data is synthetic.

The dataset uses fake:

- account IDs
- transaction IDs
- complaint IDs
- merchant descriptors
- narratives
- balances
- events

No real customer data is used.
