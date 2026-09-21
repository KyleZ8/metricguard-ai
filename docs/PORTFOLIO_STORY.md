# MetricGuard AI Portfolio Story

Use this document to explain the project in resumes, interviews, GitHub, LinkedIn, and presentations.

## 30-Second Pitch

MetricGuard AI is an internal analytics dashboard for financial-services KPI investigation. It helps analysts determine whether a KPI movement is real, caused by data-quality issues, or both. The tool validates source data, computes raw and corrected KPI values, identifies segment drivers, analyzes customer complaint themes, prioritizes recommended actions, and generates a grounded manager-ready explanation.

The main demo investigates a credit-card dispute-rate spike. The raw dashboard shows a 32.8% increase, but after removing duplicate source events the corrected increase is 21.3%. That means part of the spike is a data pipeline issue, and part of it is a real business movement concentrated in travel and mobile activity.

## One-Sentence Resume Version

Built MetricGuard AI, a Streamlit-based financial analytics dashboard that detects data-quality-driven KPI inflation, computes corrected KPI movement, identifies segment and complaint-text drivers, prioritizes analyst actions, and generates grounded manager-ready explanations.

## Resume Bullets

- Built an end-to-end KPI investigation dashboard for a fictional-card-issuer-style credit-card portfolio, combining data-quality checks, raw vs corrected metric calculation, anomaly detection, segment driver analysis, NLP complaint themes, and action prioritization.
- Developed deterministic pandas modules for seven Finance Risk KPIs, including dispute rate, fraud claim rate, payment failure rate, delinquency proxy, charge-off proxy, complaint rate, and fee-related complaint share.
- Detected and quantified a synthetic source-system replay issue that inflated August dispute rate from a corrected 1.61% to a reported 1.77%, removing 165 duplicate disputed purchase rows.
- Implemented corrected driver analysis showing `merchant_category = travel` contributed 317 additional disputed purchases and `travel x mobile` was the strongest interaction cell.
- Used sentence-embedding similarity to classify complaint narratives into business themes, then validated that representative complaints did not contradict structured segment fields.
- Added an evidence-grounded explanation layer that validates every generated number against a computed evidence packet before surfacing the manager summary.
- Built an action-prioritization engine using data-quality gating, counterfactual impact sizing, confidence scoring, and impact x effort ranking to convert investigation results into business recommendations.

## LinkedIn / GitHub Short Description

MetricGuard AI is a portfolio project focused on financial-services KPI trust. It investigates whether a dashboard alert reflects real business movement or data-quality noise, then turns the evidence into a corrected metric story, driver analysis, complaint-theme evidence, prioritized action plan, and manager-ready explanation.

## Demo Script

### 1. Open With The Business Problem

"This project is built around a common analytics problem: a KPI moves, but the analyst does not know yet whether the business changed or the data broke. In financial services, that distinction matters because a false alert can create unnecessary escalation, while a real movement needs fast root-cause analysis."

### 2. Show The Metric Overview

"The dashboard starts with the reported dispute rate. August 2026 shows 1.77%, compared with 1.33% in July, which looks like a 32.8% increase."

"But the corrected value is 1.61%. The difference comes from duplicate source events. So the first conclusion is not simply 'disputes spiked.' The correct conclusion is: part data-quality issue, part real movement."

### 3. Show Data Quality

"Before explaining drivers, the tool checks whether the metric can be trusted. It catches duplicate source transaction IDs, missing merchant categories, null-rate drift, category distribution drift, and row-count anomalies."

"This is important because driver analysis on raw data would overstate the travel/mobile segment. The data-quality layer protects the business from acting on inflated evidence."

### 4. Show Trend

"The Trend tab compares raw and corrected KPI values for the selected period. The orange line is reported, the blue line is corrected. The user can change monthly, rolling-quarter, or rolling-six-month windows."

### 5. Show Drivers

"Once the metric is corrected, the dashboard identifies where the real movement remains. The strongest merchant-category driver is travel. The strongest interaction cell is travel on mobile."

"I split count drivers from rate drivers because they answer different business questions. Count drivers explain where the portfolio impact came from; rate drivers explain where customer experience deteriorated most."

### 6. Show Complaint Themes

"The complaint text analysis checks whether customer language supports the same story. Instead of relying only on keyword counts, it uses embedding similarity against a fixed business-theme taxonomy. That keeps the output interpretable and auditable."

### 7. Show Evidence

"The Evidence tab shows representative complaints behind each theme. This is there so an analyst can challenge the label. If the label says duplicate-looking travel charge, the example should actually look like that and should not contradict merchant category or channel."

### 8. Show Action Plan

"The Action Plan is the decision layer. It ranks what the analyst should do next. The first priority is fixing duplicate source-event ingestion, because that inflated the reported KPI. The second priority is investigating the travel driver, because the corrected metric is still materially elevated."

"The scenarios estimate how many events could be avoided if the target segment returns partially to its comparison-period baseline."

### 9. Show Explanation & Export

"Finally, the dashboard creates a manager-ready summary. The LLM is not allowed to calculate metrics. It receives a structured evidence packet, and every number in the explanation is validated against that packet after generation."

### 10. Close

"The main point is that this is not only a dashboard. It is a metric investigation workflow: trust the data, correct the metric, identify drivers, connect text evidence, prioritize action, and communicate the result."

## Interview Talking Points

### Why This Fits A a fictional card issuer's Senior Data Analyst Role

This project maps well to Senior Data Analyst work because it combines metric ownership, dashboard reliability, data quality, business interpretation, and communication. fictional-card-issuer-style analytics work often requires analysts to define KPIs, monitor trends, investigate anomalies, partner with business teams, and explain findings clearly.

### Why Not Just Use A Black-Box ML Model?

Because the problem is not only prediction. The user needs to trust, explain, and defend the result. In financial services, transparent methods are often stronger than black-box methods when the output informs business escalation, reporting, or operational action.

### Why Use Synthetic Data?

The project uses synthetic data because real credit-card transaction and complaint data is sensitive. The synthetic data is still designed to behave like realistic portfolio data: account segments, transaction types, payment failures, disputes, complaints, monthly snapshots, duplicate source events, missing fields, and planted business signals.

### What Makes The Project More Than A Notebook?

The project is built as a reusable analytics product. It has a dashboard, governed KPI definitions, reusable engines, automated tests, source-of-truth documentation, and a workflow that matches how an analyst would investigate a real KPI issue.

### How The LLM Is Controlled

The LLM is used for language, not calculation. The project builds an evidence packet first, then asks for a structured explanation. After generation, the validator extracts numbers from the text and checks that each one appears in the evidence packet. Unsupported numbers are flagged or replaced by a deterministic fallback.

## Technique Cheat Sheet

| Technique | Simple explanation | Why it matters |
| --- | --- | --- |
| Data-quality checks | Tests whether source data is complete, valid, and deduplicated | Prevents false KPI escalation |
| Raw vs corrected metrics | Shows reported value beside remediated value | Separates data issue from real movement |
| Rolling z-score | Compares a month to recent baseline | Flags unusual KPI movement |
| PSI drift | Measures whether a category distribution changed | Finds population/source mix shifts |
| Contribution analysis | Measures which segment caused the count increase | Explains portfolio-level movement |
| Rate deterioration | Measures which segment's rate worsened most | Finds experience or risk deterioration |
| Interaction analysis | Looks at two segment fields together | Finds concentrated root-cause cells |
| Decision tree segmentation | Produces readable high-risk segment rules | Finds interactions without manual guessing |
| Embedding similarity | Matches complaint text to business themes by meaning | Better than exact keyword matching |
| Evidence packet | Structured facts sent to explanation layer | Keeps generated summaries grounded |
| Number validation | Checks generated numbers against evidence | Reduces hallucination risk |
| Action scoring | Ranks actions by impact, confidence, and effort | Turns analysis into decisions |

## STAR Interview Example

### Situation

A credit-card portfolio dashboard showed a sharp increase in dispute rate. The business needed to know whether the movement was real before escalating it.

### Task

I needed to build a tool that could validate the data, correct the KPI if needed, identify business drivers, and produce a manager-ready explanation.

### Action

I built MetricGuard AI with deterministic Python modules for data-quality checks, KPI calculation, deduplication, anomaly detection, segment drivers, complaint-theme analysis, action prioritization, and grounded explanation. I also built a Streamlit dashboard so the investigation could be followed step by step.

### Result

The tool showed that the raw August dispute rate increased 32.8%, but the corrected increase was 21.3% after removing 165 duplicate disputed purchase rows. It then identified travel and mobile as the strongest business drivers and produced a prioritized action plan: fix duplicate ingestion first, then investigate the travel driver.

## Claude Code Review Prompt

```text
Read README.md, PROJECT_SPEC.md, TECHNIQUE_LOG.md, DATA_DICTIONARY.md, and docs/PORTFOLIO_STORY.md.

Review MetricGuard AI from a portfolio/interview perspective. Check whether the README and portfolio story accurately describe the codebase and avoid overclaiming. Focus on a fictional card issuer's Senior Data Analyst relevance, clarity of technical explanation, whether the action-prioritization module sounds realistic for regulated financial analytics, and whether any resume bullet should be softened or strengthened.

Do not rewrite the whole project. Suggest concise edits only, and flag any claim that is not supported by the implemented code or synthetic data.
```
