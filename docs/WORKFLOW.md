# MetricGuard AI Workflow

Primary target: a fictional card issuer's Senior Data Analyst.

Secondary target: data analyst, product analyst, and growth analyst roles at finance and technology companies.

## Project North Star

MetricGuard AI should answer one practical analyst question:

> Did this KPI really move, or is the movement caused by data quality, metric definition, or pipeline issues?

The project should show that the analyst can combine:

- business KPI thinking
- SQL/Python-style data validation
- anomaly detection
- segment driver analysis
- NLP on customer text
- LLM-based explanation
- dashboard/product design

## Where We Build

Use this folder as the project root:

```text
Final Project/metricguard-ai/
```

All specs, research, datasets, app code, tests, and final-report drafts should live here.

Recommended structure:

```text
metricguard-ai/
  WORKFLOW.md
  PROJECT_SPEC.md
  KPI_RESEARCH.md
  TECHNIQUE_LOG.md
  DATA_DICTIONARY.md
  data/
    synthetic/
  src/
  app/
  tests/
  docs/
```

## Tool Roles

### Codex

Use Codex as the lead project partner.

Codex should own:

- project strategy
- a fictional card issuer's Senior Data Analyst positioning
- research synthesis
- KPI selection and definitions
- architecture decisions
- code review
- test planning
- final report drafting
- keeping written project context up to date

Codex is useful because it can work across conversation, local files, terminal commands, web research, and documents in one place.

### Claude Code

Use Claude Code as a coding collaborator, especially for isolated implementation tasks.

Claude Code should own bounded tasks like:

- implement one analysis function
- improve one Streamlit page
- refactor one module
- write unit tests for one file
- fix one error message
- improve styling in one component

Do not ask Claude Code to redesign the whole project from scratch unless the shared docs are already updated.

## Handoff Rule

Never hand off vague context.

Bad handoff:

```text
Build the project we discussed.
```

Good handoff:

```text
Read WORKFLOW.md, PROJECT_SPEC.md, KPI_RESEARCH.md, TECHNIQUE_LOG.md, and DATA_DICTIONARY.md.
Then implement the metric quality checks described in PROJECT_SPEC.md.
Only edit files under src/quality_checks.py and tests/test_quality_checks.py.
Preserve the existing public function names.
Run the tests before stopping.
```

## Shared Source Of Truth

To avoid token-limit and context-loss issues, keep decisions in files:

- `PROJECT_SPEC.md`: what the product is and who uses it
- `KPI_RESEARCH.md`: authentic KPIs and why they matter
- `TECHNIQUE_LOG.md`: every technical method used, why we used it, and how to explain it
- `DATA_DICTIONARY.md`: synthetic dataset fields and meanings
- `WORKFLOW.md`: how Codex and Claude Code cooperate

When an important decision changes, update the relevant file before coding further.

## Technique Discussion Rule

Every technique must be explainable in interview language.

For each technique, document:

- what problem it solves
- why it is appropriate
- what input it uses
- what output it produces
- how it could fail
- how it connects to a fictional card issuer's Senior Data Analyst work

Examples:

- data quality checks
- metric definition validation
- time-series anomaly detection
- segment contribution analysis
- TF-IDF keyword extraction
- topic clustering
- LLM explanation with evidence grounding

## Target-Role Priority

Prioritize problems that match analytics work in financial services:

- metric trust
- data quality
- dashboard reliability
- risk monitoring
- customer complaints
- payment/dispute/fraud operations
- governance and business definitions
- evidence-backed executive explanations

The demo should feel like a tool for an analyst supporting risk, card, payments, fraud/dispute, or customer operations teams.

## Growth And Product Adaptability

Build the app around a flexible metric-investigation engine.

Fictional-card-issuer finance demo KPIs can include:

- dispute rate
- fraud claim rate
- payment failure rate
- delinquency rate
- complaint rate
- fee-related complaint share

Growth/product versions can reuse the same engine with different KPIs:

- activation rate
- conversion rate
- retention rate
- churn rate
- feature adoption rate
- experiment metric movement

The app should make this adaptability visible with a `business mode` selector:

```text
Finance Risk
Growth Funnel
Product Health
```

The core analysis stays the same:

```text
metric movement -> data quality check -> anomaly detection -> driver analysis -> text themes -> grounded explanation
```

## Development Process

1. Research authentic, fictional-card-issuer-relevant KPIs.
2. Lock the project spec.
3. Design synthetic datasets.
4. Build the deterministic analysis engine.
5. Add NLP text analysis.
6. Add the LLM explanation layer.
7. Build the dashboard UI.
8. Add tests and demo data.
9. Write the final report.
10. Prepare interview talking points.

## Quality Bar

The final project should be demoable in 3 to 5 minutes.

It should show:

- one metric spike
- whether the spike is real or data-quality-driven
- top segment drivers
- related customer text themes
- evidence rows
- a manager-ready explanation
- a clear note that code computes facts and the LLM explains them

