# KPI Research Notes

This file records authentic KPI ideas for a Capital One Senior Data Analyst-focused project.

Research date: September 5, 2026

## Role Signals From Capital One SDA Postings

Capital One describes Senior Data Analyst work around three broad categories:

- innovation
- business intelligence
- data management

Important phrases from current postings:

- build and maintain well-managed data solutions
- solve complex business and product challenges
- lead product analytics to measure and improve product/platform health
- build tools, techniques, metrics, and dashboards
- drive meaningful insights on business strategies
- data quality management
- metadata, lineage, and business definitions
- monitor and report on data quality
- Python, R, Spark, SQL
- BI visualization tools
- Snowflake and AWS
- structured and unstructured data

Sources:

- https://www.capitalonecareers.com/job/mclean/senior-data-analyst-analytics-and-transformation/1732/98634904320
- https://www.capitalonecareers.com/job/mclean/senior-data-analyst-risk-product-data/1732/98246876496

Implication for this project:

> The project should not only show modeling. It should show metric ownership, BI thinking, data quality, governance, and business explanation.

## Public Capital One / Credit-Card Metrics

Capital One's public earnings materials and SEC credit metric disclosures use several authentic credit-card and banking KPIs.

### Net Charge-Off Rate

Definition from Capital One SEC monthly credit metric disclosure:

> Net charge-off rate is calculated by dividing annualized net charge-offs by average loans held for investment for the specified loan category.

Source:

- https://www.sec.gov/Archives/edgar/data/927628/000092762825000241/ex991june2025creditmetrics.htm

Why it matters:

> This is a core credit-risk loss metric. It tells the business how much loan balance is being written off after recoveries.

Project use:

- Use as an advanced KPI or proxy.
- In synthetic data, compute a simplified charge-off proxy:

```text
charge_off_rate = charged_off_balance / average_balance
```

### 30+ Day Delinquency Rate

Definition from Capital One SEC monthly credit metric disclosure:

> 30+ day performing delinquency rate is calculated by dividing 30+ day performing delinquent loans by period-end loans held for investment for the specified loan category.

Source:

- https://www.sec.gov/Archives/edgar/data/927628/000092762825000241/ex991june2025creditmetrics.htm

Federal Reserve framing:

> Credit card delinquency rates track the fraction of balances at least 30 days past due, excluding severely derogatory balances.

Source:

- https://www.federalreserve.gov/econres/notes/feds-notes/predicting-credit-card-delinquency-rates-20250228.html

Why it matters:

> Delinquency is an early credit-risk indicator. It usually moves before charge-offs.

Project use:

```text
delinquency_rate_30dpd = balance_30plus_dpd / period_end_balance
```

### Purchase Volume

Capital One reports domestic card purchase volume as a selected performance metric in its credit-card segment.

Source:

- https://investor.capitalone.com/static-files/63e07fa4-c440-4f70-b5e0-e02380962c8a

Why it matters:

> Purchase volume measures customer spending activity and card engagement.

Project use:

```text
purchase_volume = sum(purchase_amount)
```

### Period-End Loans Held For Investment

Capital One reports period-end loans held for investment in segment-level summaries.

Source:

- https://investor.capitalone.com/static-files/63e07fa4-c440-4f70-b5e0-e02380962c8a

Why it matters:

> It is the denominator behind several credit-risk metrics and gives scale/context to movement.

Project use:

```text
period_end_balance = sum(statement_balance)
```

### Average Loans Held For Investment

Capital One uses average loans held for investment as the denominator for net charge-off rate.

Source:

- https://investor.capitalone.com/static-files/63e07fa4-c440-4f70-b5e0-e02380962c8a

Project use:

```text
average_balance = mean(statement_balance)
```

### Total Net Revenue Margin

Capital One reports total net revenue margin in domestic card selected performance metrics.

Source:

- https://investor.capitalone.com/static-files/63e07fa4-c440-4f70-b5e0-e02380962c8a

Why it matters:

> It connects risk and customer behavior to business economics.

Project use:

- Mention in research and optional dashboard context.
- Do not make it the first demo KPI because synthetic calculation would require more financial assumptions.

### Refreshed FICO Mix

Capital One reports refreshed FICO score mix for domestic card, including above/below 660 groups.

Source:

- https://investor.capitalone.com/static-files/63e07fa4-c440-4f70-b5e0-e02380962c8a

Why it matters:

> FICO mix is a common segmentation lens for credit-card risk analytics.

Project use:

Use FICO band as a segment field:

```text
fico_band = prime / near_prime / subprime
```

or:

```text
fico_band = >660 / <=660
```

## Complaint / Customer-Experience Metrics

The CFPB Consumer Complaint Database is a useful public reference for authentic customer complaint fields.

CFPB states that the database lets users explore financial product and service complaints, view trends, read complaints, and download data.

Source:

- https://www.consumerfinance.gov/data-research/consumer-complaints/

CFPB field references include:

- date received
- product
- sub-product
- issue
- sub-issue
- consumer complaint narrative
- company public response
- company
- state
- submitted via
- company response to consumer
- timely response
- complaint ID

Sources:

- https://cfpb.github.io/api/ccdb/fields.html
- https://www.consumerfinance.gov/complaint/data-use/

Project use:

Synthetic complaint fields should mirror this structure enough to feel realistic.

Useful complaint KPIs:

### Complaint Rate

Definition:

```text
complaint_rate = complaint_count / active_accounts
```

Why it matters:

> Measures customer pain relative to account base.

### Dispute-Related Complaint Share

Definition:

```text
dispute_complaint_share = dispute_related_complaints / total_complaints
```

Why it matters:

> Indicates whether dispute/friction issues are becoming a larger part of customer feedback.

### Fee-Related Complaint Share

Definition:

```text
fee_complaint_share = fee_related_complaints / total_complaints
```

Why it matters:

> Helps detect customer confusion or dissatisfaction around fees.

### Timely Response Rate

Inspired by CFPB complaint fields.

Definition:

```text
timely_response_rate = timely_responses / total_complaints
```

Why it matters:

> Operational KPI for complaint handling and customer operations.

## Recommended First Demo KPIs

The first demo should use these KPIs:

1. `dispute_rate`
   - Main KPI.
   - Easy to understand.
   - Strong connection to card operations, fraud, customer experience, and risk.

2. `complaint_rate`
   - Customer-experience supporting KPI.
   - Lets us use NLP on complaint text.

3. `payment_failure_rate`
   - Operational supporting KPI.
   - Strong product analytics connection.

4. `delinquency_rate_30dpd`
   - Authentic credit-risk KPI.
   - Strong Capital One relevance.

5. `purchase_volume`
   - Business activity context.
   - Helps avoid analyzing rates without denominator/business scale.

## Recommended Demo Story

Dashboard alert:

> Raw dispute rate increased 32.8% month over month, from 1.33% in July 2026 to 1.77% in August 2026.

MetricGuard investigation:

- Data-quality checks detect a duplicate-file issue in one ingestion batch.
- After deduplication, dispute rate is still elevated at 1.61%, but the corrected increase is smaller at 21.3%.
- Driver analysis shows that most of the remaining increase comes from travel merchant categories, mobile channel, and customers with FICO <= 660.
- Complaint NLP shows rising themes around "foreign transaction fee", "unrecognized merchant", and "duplicate charge".
- The AI explanation summarizes the finding and recommends next analyses.

Manager-ready conclusion:

> The spike is partially inflated by duplicate records, but not entirely explained by data quality. After removing duplicates, dispute rate remains 21.3% above July, mainly driven by travel merchant transactions in mobile cardholder activity and supported by a rise in fee/dispute-related complaint themes.
