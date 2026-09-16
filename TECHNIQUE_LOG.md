# Technique Log

This file explains every technique used in MetricGuard AI in interview-ready language.

## Principle

Every technique must answer:

- What problem does it solve?
- Why is it appropriate?
- What input does it use?
- What output does it produce?
- How could it fail?
- How does it connect to Capital One SDA work?

## Planned Techniques

### Data Quality Checks

Problem solved:

> A KPI spike may be caused by broken data, not real business movement.

Checks:

- missing required fields
- duplicate IDs
- invalid dates
- impossible values
- row-count anomalies
- source-system freshness delays
- category drift

Input:

- synthetic accounts, transactions, complaints, and metric definitions

Output:

- data-quality score
- list of failed checks
- affected rows or segments

Capital One SDA connection:

> Capital One postings explicitly emphasize data quality management, metadata, lineage, business definitions, and tools to monitor/report data quality.

Implemented in:

```text
src/metricguard_engine.py
```

Current checks:

- duplicate upstream `source_transaction_id`
- missing merchant category
- posted date earlier than transaction date
- invalid positive-amount transactions
- monthly ingestion row-count spike
- missing complaint narrative
- negative statement balance

Interview explanation:

> I intentionally treated data quality as part of the product, not an afterthought. Before explaining a KPI movement, the engine checks whether the underlying records are complete, deduplicated, timely, and valid. This is important in financial analytics because a dashboard spike can create unnecessary escalations if it is caused by a replayed batch or broken source feed.

### Metric Calculation

Problem solved:

> Business users need consistent KPI definitions.

Input:

- metric definitions
- synthetic transaction/account/complaint tables

Output:

- current KPI value
- previous-period KPI value
- absolute and percent change
- denominator context

Capital One SDA connection:

> Matches BI and dashboard ownership work: designing tools, techniques, metrics, and dashboards for business insights.

Implemented metrics:

- raw dispute rate
- deduplicated dispute rate
- payment failure rate
- complaint rate
- balance-based 30+ day delinquency proxy

Current main demo result:

```text
July 2026 dispute rate: 1.33%
August 2026 raw dispute rate: 1.77%
August 2026 deduped dispute rate: 1.61%
Raw month-over-month change: +32.8%
Corrected month-over-month change: +21.3%
```

Interview explanation:

> The first result shows exactly why metric governance matters. The raw dashboard number overstated the issue because of duplicated source events, but after deduplication the KPI was still materially elevated. That gives the analyst a more precise story: part data-quality issue, part real business movement.

### Anomaly Detection

Problem solved:

> Analysts need to know whether a metric movement is unusual relative to history.

Initial method:

- rolling mean and rolling standard deviation
- z-score anomaly flag

Why start simple:

> It is transparent, explainable, and appropriate for a class project/interview demo.

Possible later method:

- Isolation Forest for multivariate anomaly detection

Input:

- time series of KPI values

Output:

- anomaly score
- anomaly flag
- expected range

Failure modes:

- seasonality can create false positives
- short history makes baseline unstable
- one-time business events may look anomalous but be explainable

Implemented in:

```text
rolling_anomaly_flags()
```

Interview explanation:

> I started with rolling z-score anomaly detection because it is transparent and easy to defend. For an SDA role, explainability matters: a stakeholder can understand "this month's metric is above the recent rolling baseline" more easily than a black-box alert.

### Segment Driver Analysis

Problem solved:

> A KPI moved, but the analyst needs to know which segment drove the movement.

Implemented methods:

- corrected segment contribution analysis
- rate deterioration analysis
- two-way interaction driver analysis
- optional shallow decision-tree segmentation

Why these methods fit:

> These are common business intelligence and product/risk analytics techniques. They are more useful than a black-box model at this stage because the analyst needs an auditable explanation: which segment changed, by how much, and whether the movement came from volume or rate deterioration.

Inputs:

- deduplicated purchase transactions
- account segment fields
- current month
- previous month

Segment fields:

- merchant category
- channel
- FICO band
- customer segment
- product type
- region

Output:

- count-driver table
- rate-deterioration table
- interaction driver table
- heatmap-ready matrix
- optional readable decision-tree rules

Implemented in:

```text
src/driver_analysis.py
```

Current demo findings:

```text
Top count driver:
merchant_category = travel
July disputed purchases: 204
August disputed purchases: 521
Change: +317
Share of merchant-category dispute growth: 88.8%

Top rate driver:
merchant_category = travel
July dispute rate: 1.94%
August dispute rate: 4.90%
Change: +2.96 percentage points

Top interaction:
merchant_category = travel and channel = mobile
July dispute rate: 2.33%
August dispute rate: 6.61%
Change: +4.27 percentage points
```

Optional decision-tree stretch:

```text
merchant_category == travel AND channel == mobile AND fico_band == <=660
Purchases: 1,243
Disputed: 111
Dispute rate: 8.93%
Lift vs overall: 5.53x
```

Important design decisions:

- all driver analysis runs on corrected, deduplicated transactions
- count drivers and rate drivers are separated because they answer different questions
- contribution share is calculated within each segment field to avoid double-counting
- missing segment values are kept as a visible `__missing__` bucket
- small-denominator segments are flagged because rates on tiny groups are noisy
- the decision tree is shallow and used for segmentation, not black-box prediction

Capital One SDA connection:

> Useful for risk/product analytics because it turns a dashboard movement into an action-oriented explanation.

Failure modes:

- segment analysis can show correlation but does not prove causality
- small segments can produce unstable rates
- missing or wrong dimensions can hide the real driver
- a decision tree can overfit if allowed to grow too deep

Interview explanation:

> After confirming the KPI spike was not fully explained by duplicate data, I decomposed the corrected dispute-rate movement by business segments. I separated count contribution from rate deterioration, because a large segment may explain most of the added disputes while a smaller segment may have the worst customer experience deterioration. I also added interaction analysis and a shallow decision tree to detect combined drivers like travel on mobile for lower-FICO customers.

Manager-ready interpretation:

> The corrected August dispute-rate increase is concentrated in travel-related mobile purchases. The data-quality replay inflated the spike, but even after removing duplicates, travel and mobile remain the strongest business drivers.

### NLP Complaint Theme Analysis

Problem solved:

> Customer complaint text contains signal that structured metrics alone may miss.

Implemented approach:

- sentence-embedding theme classification
- business taxonomy mapping
- semantic clustering for emerging themes
- representative complaint retrieval
- theme trend comparison across periods and segments

Why not make TF-IDF the main technique:

> TF-IDF is useful and explainable, but it mainly sees exact words and phrases. This project needs to understand messy complaint language where customers describe the same issue in different ways. Because the user already has a TF-IDF project, MetricGuard should lead with embeddings and use keyword methods only as supporting evidence.

Core idea:

> Convert each complaint narrative into a semantic vector, compare it to business-theme vectors, assign the closest theme, and then measure which themes increased in the same segment where the corrected KPI moved.

Recommended business taxonomy:

- unclear merchant descriptor
- duplicate-looking travel charge
- foreign transaction fee confusion
- mobile dispute submission friction
- delayed dispute resolution
- failed mobile autopay
- fraud/security concern
- generic service complaint

Input:

- synthetic complaint narratives
- complaint dates
- complaint issue/product fields
- account segment fields
- corrected driver-analysis segments

Output:

- theme-level complaint counts
- theme share by period
- month-over-month theme lift
- theme concentration in travel/mobile and other driver segments
- rising complaint themes
- representative complaint examples
- optional cluster labels for newly emerging themes

Implemented in:

```text
src/text_theme_analysis.py
```

Current demo findings:

```text
Top rising August themes:
duplicate_looking_travel_charge: 87 -> 147 (+60)
mobile_dispute_submission_friction: 76 -> 121 (+45)
unclear_merchant_descriptor: 111 -> 146 (+35)
delayed_dispute_resolution: 33 -> 65 (+32)

Travel complaints:
duplicate_looking_travel_charge: 22 -> 61

Mobile complaints:
mobile_dispute_submission_friction: 45 -> 79
```

Important evidence distinction:

> Keyword probes show foreign-transaction-fee and failed-autopay wording exists and rises slightly after the generator alignment fix, but the single-label embedding classifier does not rank those as August KPI drivers. The strongest text story remains travel duplicate charges, mobile dispute friction, unclear descriptors, and dispute-status delays. This is intentional: use the classifier for theme attribution and probes for generator validation.

Course connection:

> Uses text preprocessing, embeddings, semantic similarity, clustering, and grounded text evidence. TF-IDF or class-based keyword scoring can still be used to produce readable labels, but embeddings are the main representation.

Capital One SDA connection:

> Matches work with unstructured data and customer-risk/product insights.

Why this fits industry practice:

> Many analytics teams do not have clean labeled complaint data for every new issue. Embedding-based theme matching is useful because it combines modern NLP with business control: the model helps compare meaning, but the taxonomy keeps the output aligned to risk, product, and operations language.

Failure modes:

- embeddings can group complaints by wording style instead of business meaning
- theme labels can be too broad or too narrow
- synthetic complaint text may be cleaner than real call-center text
- semantic similarity can assign a theme even when the complaint is ambiguous

Guardrails:

- keep confidence scores
- expose representative complaints
- allow an `unclear_or_other` theme
- report counts and rates, not just labels
- do not let an LLM invent themes without evidence

Interview explanation:

> I used embedding-based complaint theme analysis rather than simple keyword counting because customers describe similar problems in different language. The system maps narratives to a controlled business taxonomy, then measures which themes increased in the same segments that drove the corrected KPI spike. This connects unstructured customer text to a governed KPI investigation.

### LLM Grounded Explanation

Problem solved:

> Analysts need to explain technical findings to managers and partners.

Implemented approach:

- deterministic evidence packet construction
- constrained manager-summary generation
- optional OpenAI Responses API call when an API key is configured
- deterministic fallback explanation when no API key is available
- citation-style references back to computed tables

Input:

- computed KPI values
- data-quality results
- driver-analysis table
- NLP themes
- evidence rows

Output:

- plain-English diagnosis
- manager-ready summary
- suggested next analysis
- risk/control caveats
- evidence references

Guardrail:

> The LLM must explain computed facts. It should not invent numbers, calculate totals, or make unsupported claims.

Design rule:

> Code builds a compact evidence packet from the metric, quality, driver, and text-theme modules. The LLM receives only that packet and must return structured fields such as `headline`, `manager_summary`, `evidence`, `recommended_actions`, and `limitations`. Any number in the explanation must already exist in the packet.

Implemented in:

```text
src/explanation_engine.py
```

Current demo output:

```text
Generated by: deterministic-fallback
Evidence packet numbers whitelisted: 263
Numbers checked in explanation: 39
Unsupported numbers: 0

Headline:
dispute_rate rose 32.79% in 2026-08, but 21.27% once duplicate source events are removed
```

Current evidence chain:

```text
Raw dispute_rate: 1.77%
Corrected dispute_rate: 1.61%
Duplicate source events removed: 165
Top business driver: merchant_category = travel
Top interaction: travel on mobile
Top text theme: duplicate_looking_travel_charge
Representative quote: The same airline charge appears twice and I cannot tell whether one is still pending.
```

Why this fits industry practice:

> Modern analytics copilots are most defensible when deterministic systems compute facts and the language model only helps communicate them. For a finance/risk analytics project, the important skill is not simply "calling an LLM"; it is showing that AI output is grounded, auditable, and separated from metric calculation.

Failure modes:

- the LLM may overstate causality from correlation
- it may introduce a number not present in the evidence packet
- it may ignore a data-quality caveat
- it may write a confident summary when the evidence is mixed

Guardrails to implement:

- low temperature
- structured output schema
- whitelist every numeric value from the evidence packet
- post-generation validation that rejects unsupported numbers
- explicit wording that findings are analytical signals, not causal proof
- no financial advice, credit policy recommendation, or customer-level decisioning

Capital One SDA connection:

> Shows AI-enabled insight generation while preserving data governance and metric trust.

Interview explanation:

> I used an LLM only after the deterministic analytics pipeline had already computed the KPI movement, data-quality issues, corrected metric, driver tables, and complaint themes. The model's job is to translate those facts into a manager-ready explanation, not to calculate or invent evidence. I also validate the output so every number it mentions comes from the evidence packet.

### Dashboard Investigation Workflow

Problem solved:

> Analysts need more than individual tables. They need a guided product experience that starts with the KPI alert, checks whether the data can be trusted, separates raw from corrected movement, identifies business drivers, connects customer text evidence, and produces a manager-ready summary.

Implemented approach:

- Streamlit dashboard
- Plotly/Altair charts
- cached analysis report construction
- evidence-first layout
- exportable manager summary

Why this fits:

> A Senior Data Analyst project should not look like disconnected notebook cells. The UI should demonstrate product thinking: what the analyst sees first, which decisions the tool helps them make, and how each technical module supports a business workflow.

Recommended page flow:

```text
1. Investigation header
2. Metric Health Overview
3. Raw vs corrected KPI trend
4. Data Quality Panel
5. Driver Analysis Panel
6. Complaint Theme Panel
7. Evidence Table
8. Grounded Explanation Panel
```

Design principles:

- show the answer first, then the evidence
- distinguish raw metric from corrected metric visually
- make data-quality failures visible before driver claims
- keep NLP themes connected to the travel/mobile driver
- show the explanation validation badge
- avoid chatbot-first design
- make the layout adaptable to Finance Risk, Growth Funnel, and Product Health modes

Input:

- metric report
- quality report
- driver report
- text theme report
- grounded explanation

Output:

- demo-ready analytics dashboard
- manager-ready explanation export
- analyst evidence tables

Failure modes:

- too many tables can hide the story
- over-styled dashboards can feel less credible for risk analytics
- showing LLM prose without validation can weaken trust
- evidence rows can distract if they contradict structured fields

Capital One SDA connection:

> This turns the project into a self-service analytics tool: KPI monitoring, dashboard reliability, data quality, driver analysis, customer text insight, and manager communication in one workflow.

Interview explanation:

> I designed the dashboard around the actual analyst workflow, not around charts for their own sake. The user first sees whether the KPI alert is real, then sees the data-quality checks, corrected metric, segment drivers, complaint themes, and a validated explanation. That makes the project feel like an internal analytics product rather than a one-off notebook.

### Multi-KPI Metric Workbench

Problem solved:

> A real analytics team does not investigate only one metric. The same analyst may need to review disputes, fraud claims, failed payments, delinquency, charge-offs, complaint volume, and fee-related complaints, while changing the month being compared.

Implemented approach:

- governed KPI catalog from `metric_definitions.csv`
- reusable numerator / denominator metric engine
- raw and corrected trend for every active Finance Risk KPI
- selectable current period and comparison period
- monthly, rolling-quarter, and rolling-six-month analysis windows
- selected-window quality filtering by KPI source table and relevant months
- generic corrected segment-driver table for every active KPI
- dispute-rate-specific NLP and grounded explanation retained for the richest demo path

Active finance KPIs:

```text
dispute_rate
fraud_claim_rate
payment_failure_rate
delinquency_rate_30dpd_balance
net_charge_off_rate_proxy
complaint_rate
fee_complaint_share
```

Why this fits industry practice:

> Mature BI and risk analytics teams treat metrics as governed objects: each KPI has a business definition, numerator, denominator, owner, source tables, limitations, and allowed cuts. The dashboard now reflects that practice. The UI selector chooses a metric definition, then the backend computes the selected KPI from source data rather than relabeling one hardcoded chart.

Important boundary:

> The metric, quality, trend, and driver layers now work across all active Finance Risk KPIs and across monthly, rolling-quarter, and rolling-six-month views. Because the synthetic data currently covers January through August 2026, quarter and six-month views use rolling windows rather than fixed calendar periods. The complaint-theme and validated LLM explanation layer remains tied to monthly `dispute_rate`, because that path has the strongest grounded text evidence. For the other KPIs and wider windows, the app shows a deterministic computed summary instead of pretending the dispute-specific NLP evidence applies.

Capital One SDA connection:

> This demonstrates dashboard ownership, metric governance, data-quality remediation, self-service period comparison, and reusable KPI analytics. The strongest interview point is that the tool separates a general metric platform from a deeper KPI-specific investigation workflow.

Interview explanation:

> I generalized the dashboard from one hardcoded dispute-rate investigation into a Finance Risk metric workbench. Each KPI is defined by a governed numerator and denominator, can be compared across user-selected monthly, quarterly, or six-month windows, and produces raw versus corrected values plus segment drivers. I also made the data-quality panel context-aware, so it filters to the selected KPI's source tables and relevant months. I kept the richer NLP and LLM explanation scoped to monthly dispute rate because that is where the text evidence is grounded, which is the right tradeoff for an auditable financial analytics product.

### Action Prioritization

Problem solved:

> A dashboard that stops at "the KPI moved" still leaves the analyst with the hardest part: what should the business do first? MetricGuard now converts the selected evidence into a ranked action plan while keeping every recommendation auditable.

Implemented approach:

- data-quality gating before business action
- counterfactual impact sizing
- segment-based action targeting
- evidence triangulation across metric movement, quality checks, drivers, and complaint themes
- confidence scoring
- impact × confidence × effort prioritization
- conservative/base/stretch scenario analysis
- explicit governance guardrail beside every action

Why this fits industry practice:

> In regulated financial analytics, the strongest solution is not a black-box recommender. The stronger pattern is controlled decision support: deterministic calculations, transparent scoring, clear ownership, and a documented limitation for every recommendation. This mirrors how banking teams handle model and analytics risk: recommendations should be traceable, reviewable, and appropriate to the evidence.

Scoring logic:

```text
priority_score = impact_score × confidence_score × effort_multiplier × 100
```

Inputs:

- selected KPI comparison
- selected-window quality findings
- corrected segment-driver table
- corrected interaction-driver table when available
- complaint-theme report when available

Outputs:

- prioritized recommendation table
- scenario-impact table
- confidence scorecard
- methodology notes

Capital One SDA connection:

> This moves the project from reporting into decision support. A Senior Data Analyst can explain not only what happened to a KPI, but also which action is worth taking, how much benefit it might create, what evidence supports it, and what governance guardrail prevents misuse.

Interview explanation:

> I added an action-prioritization layer that ranks recommended next steps from the selected evidence. It first gates on data quality, then sizes business impact with a counterfactual baseline: how many current-period numerator events might be avoidable if the target segment returned to the comparison-period rate. It combines metric movement, driver concentration, quality readiness, and customer text support into a confidence score, then applies an effort multiplier. I chose this transparent scoring approach instead of a black-box recommender because the use case is financial services analytics, where recommendations need to be auditable and easy to challenge.
