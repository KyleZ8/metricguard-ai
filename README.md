# MetricGuard AI

MetricGuard AI is a financial-services analytics dashboard that helps analysts decide whether a KPI movement is real, caused by data quality, or both.

The project is designed for a Capital One-style Senior Data Analyst use case: a credit-card business sees a monthly KPI spike, and the analyst needs to validate the data, correct the metric, identify drivers, connect customer complaints, prioritize action, and produce a manager-ready explanation.

## Why This Project Exists

Business teams often see dashboard movement before they know whether the movement can be trusted.

Example demo scenario:

> Credit-card dispute rate rises from 1.33% in July 2026 to 1.77% in August 2026.

MetricGuard investigates:

- Did the metric truly move?
- Did duplicate records or missing fields inflate the alert?
- Which merchant, channel, product, or customer segments drove the corrected movement?
- Do customer complaints support the same story?
- What action should the analyst recommend?

## Demo Result

The August dispute-rate spike is partly a data-quality issue and partly a real business movement.

| Metric | July 2026 | August 2026 | Change |
| --- | ---: | ---: | ---: |
| Raw dispute rate | 1.33% | 1.77% | +32.8% |
| Corrected dispute rate | 1.33% | 1.61% | +21.3% |

Key findings:

- 165 duplicate disputed purchase rows inflated the raw August KPI.
- After deduplication, the corrected KPI is still materially elevated.
- `merchant_category = travel` is the leading corrected count and rate driver.
- `travel x mobile` is the strongest interaction cell.
- Complaint themes also increase around duplicate-looking travel charges and mobile dispute friction.
- The action plan prioritizes data-quality remediation first, then targeted business investigation.

## Product Workflow

The dashboard follows the real analyst workflow:

```text
KPI alert
-> data-quality checks
-> raw vs corrected metric
-> segment drivers
-> complaint-theme analysis
-> action prioritization
-> grounded manager explanation
```

Dashboard tabs:

- Trend
- Data quality
- Drivers
- Complaint themes
- Evidence
- Action plan
- Explanation & export

## Techniques Used

| Area | Technique |
| --- | --- |
| Metric governance | KPI catalog with numerator, denominator, owner, source tables, and limitations |
| Data quality | Completeness, accepted values, referential integrity, duplicate source-event checks, date validity, row-count anomaly, null-rate drift, PSI drift |
| Metric correction | Deterministic source-event deduplication |
| Anomaly detection | Rolling baseline z-score with practical variance floor |
| Driver analysis | Count contribution, rate deterioration, interaction heatmap, shallow decision-tree segmentation |
| NLP | Sentence embedding similarity against business-theme anchors, with deterministic hashing fallback |
| LLM safety | Evidence packet, structured explanation schema, post-generation number validation |
| Decision support | Data-quality gating, counterfactual impact sizing, confidence scorecard, impact x effort prioritization |

## Active Finance KPIs

- Dispute rate
- Fraud claim rate
- Payment failure rate
- 30+ day delinquency balance rate
- Net charge-off proxy
- Complaint rate
- Fee-related complaint share

The richest end-to-end path is monthly `dispute_rate`, because it has metric correction, driver analysis, complaint NLP, action planning, and grounded explanation.

## How To Run

From the project root:

```bash
streamlit run app/streamlit_app.py
```

Direct test runners:

```bash
python tests/test_quality_checks.py
python tests/test_metric_engine.py
python tests/test_driver_analysis.py
python tests/test_text_theme_analysis.py
python tests/test_explanation_engine.py
python tests/test_action_engine.py
python tests/test_streamlit_app.py
```

## Project Structure

```text
metricguard-ai/
  app/
    streamlit_app.py
  src/
    generate_synthetic_data.py
    quality_checks.py
    metric_engine.py
    driver_analysis.py
    text_theme_analysis.py
    explanation_engine.py
    action_engine.py
  data/synthetic/
    accounts.csv
    transactions.csv
    account_monthly_snapshot.csv
    complaints.csv
    metric_definitions.csv
    GROUND_TRUTH.md
  tests/
  docs/
  PROJECT_SPEC.md
  KPI_RESEARCH.md
  TECHNIQUE_LOG.md
  DATA_DICTIONARY.md
  WORKFLOW.md
```

## Portfolio Positioning

This project demonstrates:

- data quality and metric trust
- financial-services KPI monitoring
- dashboard/product thinking
- reusable metric engineering
- structured and unstructured data analysis
- explainable AI with evidence grounding
- action-oriented analytics

Interview pitch:

> I built MetricGuard AI as an internal analytics tool for financial-services KPI investigation. It starts from a dashboard alert, validates whether the data can be trusted, corrects the metric when duplicate source events are found, identifies corrected business drivers, mines complaint text for supporting evidence, prioritizes recommended actions, and generates a manager-ready explanation with number validation. The goal was not just to show a KPI moved, but to help an analyst decide what to do next.
