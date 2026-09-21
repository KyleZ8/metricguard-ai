# MetricGuard AI Project Spec

## Working Title

MetricGuard AI

## Priority Audience

Primary target role:

> Senior Data Analyst at a fictional card issuer

Primary user inside the product:

> A fictional-card-issuer-style data analyst who owns KPI monitoring, dashboard reliability, risk/product reporting, or customer operations analytics for a credit-card business.

Secondary users:

- analytics manager
- risk manager
- product manager
- business intelligence team member
- data governance analyst

## Job-Relevant Motivation

Target-role postings emphasize:

- solving complex business and product challenges
- product analytics
- BI metrics and dashboards
- self-service data tools
- data quality management
- metadata, lineage, and business definitions
- data access governance
- Python, R, Spark, SQL
- Snowflake and AWS exposure
- working with structured and unstructured data

Sources:

- a fictional card issuer's Senior Data Analyst - Analytics & Transformation job posting (McLean, VA)
- a fictional card issuer's Senior Data Analyst - Risk Product Data job posting (McLean, VA)

## Product Problem

Analysts and business partners often see KPI movement before they know whether the movement is real.

Example:

> Credit-card dispute rate increased by 32.8% this month.

The analyst must answer:

- Did the metric really move?
- Is the change caused by duplicate rows, missing data, source freshness, schema drift, or a metric-definition problem?
- If the movement is real, which segment, product, channel, or merchant category drove it?
- Do customer complaints explain the movement?
- What should the analyst tell a manager?

## Product Promise

MetricGuard AI helps analysts determine whether a KPI movement is real or caused by data-quality issues, then explains the likely drivers in plain English with evidence from structured data and customer text.

## Demo Scenario

The first demo will focus on a fictional-card-issuer-style credit-card risk and operations use case.

Scenario:

> A dashboard shows monthly credit-card dispute rate rising from 1.33% in July 2026 to 1.77% in August 2026. MetricGuard AI investigates whether the spike is real or caused by data-quality problems, identifies the strongest segment drivers, mines related customer complaint themes, and writes a manager-ready explanation.

The demo should include both:

1. A true business movement:
   - dispute rate increases for a specific segment, such as young cardholders using travel-related merchants after a product or policy change.

2. A data-quality issue:
   - duplicate dispute records, missing merchant categories, delayed batch arrival, or a changed metric definition.

This lets the tool show the difference between:

```text
real metric movement
```

and:

```text
false signal caused by data quality
```

## User Flow

1. User opens the dashboard.
2. User selects business mode:
   - Finance Risk
   - Growth Funnel
   - Product Health
3. User uploads or loads sample synthetic data.
4. User selects one of the active Finance Risk KPIs.
5. User selects an analysis window:
   - monthly
   - rolling quarter
   - rolling 6 months
6. User selects the current period and comparison period for that window.
7. App shows metric overview:
   - current value
   - previous value
   - absolute change
   - percent change
   - anomaly flag
   - data-quality status
   - corrected value after data remediation
8. App runs data-quality checks:
   - missing required fields
   - duplicate event IDs
   - invalid date gaps
   - impossible values
   - suspicious row-count changes
   - unexpected category drift
9. App runs driver analysis:
   - product
   - customer segment
   - FICO band
   - merchant category
   - channel
   - region
10. App runs NLP theme analysis on customer text for the grounded monthly dispute-rate path:
   - complaint topic keywords
   - fee/dispute/fraud language
   - representative comments
11. App generates a grounded explanation for monthly `dispute_rate` and deterministic KPI summaries for the other finance KPIs/windows:
   - what changed
   - whether the metric movement looks real
   - what data issues were found
   - which segments drove the movement
   - what evidence supports the conclusion
   - what the analyst should check next
12. App produces an analyst action plan:
   - recommended action
   - target segment or source-system area
   - estimated avoidable events
   - confidence
   - effort
   - governance guardrail
   - conservative/base/stretch scenario impact
13. User can ask follow-up questions in a chat panel.
14. User can export a short manager-ready report.

## UI Requirements

The interface should feel like a serious internal analytics tool, not a marketing site and not a chatbot-first toy.

Core UI sections:

- business-mode selector
- active Finance Risk KPI selector
- analysis-window selector
- current-period and comparison-period selectors
- Metric Health Overview
- Trend chart with anomaly highlight
- Data Quality Panel
- Driver Analysis table/chart
- Complaint Theme/NLP panel
- Evidence table
- Action Plan panel
- AI Explanation panel
- Analyst Chat panel
- Export Report button

## Technical Principle

Use deterministic code for facts and calculations.

Use the LLM for language explanation.

```text
Python/SQL-style logic = exact metrics, validation, grouping, anomaly detection
NLP/LLM = text themes, plain-language summaries, analyst Q&A
```

This is especially important for financial services, where numbers must be grounded and auditable.

## Growth And Product Adaptability

The project is a fictional card issuer-first, but the engine should generalize.

The core workflow:

```text
metric movement
-> data quality checks
-> anomaly detection
-> driver analysis
-> text theme mining
-> grounded explanation
```

Active Finance Risk KPIs:

- dispute rate
- fraud claim rate
- payment failure rate
- delinquency rate
- net charge-off proxy
- complaint rate
- fee-related complaint share

Growth Funnel adaptation:

- conversion rate
- activation rate
- retention rate
- churn rate
- campaign response rate
- funnel drop-off rate

Product Health adaptation:

- feature adoption rate
- active-user rate
- session success rate
- crash/error rate
- support-ticket rate
- experiment metric movement

## Final Deliverables

- working Streamlit app
- synthetic dataset
- project README
- KPI research note
- technique log
- final written report
- demo script/interview talking points
