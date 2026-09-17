"""Tests for the MetricGuard AI grounded explanation layer.

These run under pytest and are also directly runnable, matching the other test
modules in this project::

    python tests/test_explanation_engine.py

No test contacts the OpenAI API. The LLM path is exercised through an injected
fake client that records the request it was given, so the prompt, the model id,
the temperature and the structured-output request are all assertable without a
key and without a network call. ``OPENAI_API_KEY`` is neutralised for the whole
module so a developer who happens to have one exported cannot turn these into
live calls by accident.

The real pipeline is built once and shared: four analysis modules over ~930k
transactions is slow, and every integration assertion can read the same packet.
Logic tests that do not need real data use a hand-built packet instead.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Neutralised before importing the engine so no code path can pick up a real key.
os.environ.pop("OPENAI_API_KEY", None)

from explanation_engine import (  # noqa: E402
    DEFAULT_MATERIAL_INTEGER_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_MODEL_ENV_VAR,
    EXPLANATION_FIELDS,
    EXPLANATION_JSON_SCHEMA,
    GENERATOR_DETERMINISTIC,
    STANDING_LIMITATIONS,
    SYSTEM_PROMPT,
    EvidencePacket,
    Explanation,
    NumberValidation,
    SupportedNumbers,
    build_deterministic_explanation,
    build_evidence_packet,
    build_llm_explanation,
    explain,
    extract_numbers,
    resolve_model,
    validate_numbers,
)
from driver_analysis import build_driver_report  # noqa: E402
from metric_engine import build_metric_report  # noqa: E402
from quality_checks import STATUS_FAIL, STATUS_PASS, load_tables, run_quality_checks  # noqa: E402
from text_theme_analysis import HashingEmbedder, build_text_theme_report  # noqa: E402


SPIKE_MONTH = "2026-08"
PRIOR_MONTH = "2026-07"

DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
CSV_FILES = (
    "accounts.csv",
    "account_monthly_snapshot.csv",
    "transactions.csv",
    "complaints.csv",
    "metric_definitions.csv",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _tables() -> dict[str, pd.DataFrame]:
    return load_tables()


@lru_cache(maxsize=1)
def _packet() -> EvidencePacket:
    """One real evidence packet, shared by every integration assertion.

    The theme report uses the offline hashing backend so the suite never needs a
    model download; the explanation layer does not care which backend ran.
    """
    tables = _tables()
    return build_evidence_packet(
        metric_report=build_metric_report(tables),
        quality_report=run_quality_checks(tables),
        driver_report=build_driver_report(tables=tables, include_decision_tree=False),
        theme_report=build_text_theme_report(
            tables=tables, embedder=HashingEmbedder(), include_clusters=False
        ),
    )


@lru_cache(maxsize=1)
def _explanation() -> Explanation:
    return build_deterministic_explanation(_packet())


def _toy_packet() -> EvidencePacket:
    """A small packet whose every number can be checked by eye."""
    supported = SupportedNumbers()
    supported.add_rate(0.02, "metric.raw_current_rate")
    supported.add_rate(0.01, "metric.raw_previous_rate")
    supported.add_rate(0.015, "metric.corrected_current_rate")
    supported.add_rate(0.01, "metric.corrected_previous_rate")
    supported.add_rate(1.0, "metric.raw_percent_change")
    supported.add_rate(0.5, "metric.corrected_percent_change")
    supported.add_rate(0.005, "metric.rate_difference")
    for value, label in (
        (200, "raw_disputed"),
        (150, "corrected_disputed"),
        (50, "duplicate_rows_removed"),
        (10_000, "raw_purchases"),
        (9_950, "corrected_purchases"),
        (40, "quality.total"),
        (2, "quality.failed"),
        (1, "quality.warned"),
        (37, "quality.passed"),
        (99, "driver.change"),
        (100, "driver.current_disputed"),
        (25, "theme.change"),
        (26, "theme.current_complaints"),
    ):
        supported.add(value, label)
    supported.add_rate(0.5, "share")
    supported.add_rate(0.01, "rate_change")

    facts = {
        "metric_name": "dispute_rate",
        "current_period": "2026-02",
        "previous_period": "2026-01",
        "metric": {
            "raw_current_rate": 0.02,
            "raw_previous_rate": 0.01,
            "corrected_current_rate": 0.015,
            "corrected_previous_rate": 0.01,
            "raw_percent_change": 1.0,
            "corrected_percent_change": 0.5,
            "rate_difference_from_duplicates": 0.005,
            "raw_current_disputed": 200,
            "corrected_current_disputed": 150,
            "duplicate_rows_removed": 50,
            "raw_current_purchases": 10_000,
            "corrected_current_purchases": 9_950,
        },
        "data_quality": {
            "total_checks": 40,
            "failed": 2,
            "warned": 1,
            "passed": 37,
            "failing_checks": [
                {
                    "check_name": "duplicate_source_transaction_id",
                    "check_type": "rule_based",
                    "table_name": "transactions",
                    "severity": "high",
                    "affected_rows": 50,
                    "observed_value": "50 duplicate rows",
                    "recommended_action": "Deduplicate.",
                }
            ],
            "warning_checks": [],
        },
        "drivers": {
            "top_count_drivers": [
                {
                    "segment_name": "merchant_category",
                    "segment_value": "travel",
                    "previous_disputed": 1,
                    "current_disputed": 100,
                    "disputed_change": 99,
                    "previous_dispute_rate": 0.01,
                    "current_dispute_rate": 0.02,
                    "rate_change": 0.01,
                    "contribution_share": 0.5,
                    "small_denominator_flag": False,
                }
            ],
            "top_rate_drivers": [],
            "top_interaction_driver": None,
        },
        "text_themes": {
            "embedding_backend": "hashing",
            "top_themes": [
                {
                    "theme_name": "duplicate_looking_travel_charge",
                    "previous_complaints": 1,
                    "current_complaints": 26,
                    "complaint_change": 25,
                    "current_share": 0.5,
                    "share_change": 0.01,
                }
            ],
            "representative_complaints": [
                {
                    "theme_name": "duplicate_looking_travel_charge",
                    "merchant_category": "travel",
                    "channel": "mobile",
                    "narrative": "I was billed twice for the same hotel reservation.",
                }
            ],
        },
        "limitations": list(STANDING_LIMITATIONS),
    }
    return EvidencePacket(facts=facts, supported_numbers=supported)


def _valid_llm_payload() -> dict[str, object]:
    """A model response that stays inside the toy packet's numbers."""
    return {
        "headline": "dispute_rate rose in 2026-02 but half the move was duplicate records",
        "executive_summary": (
            "Reported dispute_rate was 2.00% against 1.00% the month before. After removing "
            "50 duplicated source events the corrected figure is 1.50%."
        ),
        "what_changed": "200 disputed purchases fell to 150 after deduplication.",
        "data_quality_findings": "2 of 40 checks failed, including duplicate_source_transaction_id.",
        "business_drivers": "merchant_category = travel added 99 disputes.",
        "customer_text_evidence": "duplicate_looking_travel_charge rose by 25 complaints.",
        "recommended_next_steps": ["Reissue the corrected metric.", "Contact the source owner."],
        "limitations": ["Signals, not causal proof."],
        "evidence_references": ["metric_engine.remediation_impact[2026-02]"],
    }


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.output_text = text


class _FakeResponses:
    def __init__(self, owner: "FakeOpenAIClient") -> None:
        self._owner = owner

    def create(self, **kwargs: object) -> _FakeResponse:
        self._owner.calls.append(kwargs)
        return _FakeResponse(json.dumps(self._owner.payload))


class FakeOpenAIClient:
    """Stands in for ``openai.OpenAI`` with no network and no key."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []
        self.responses = _FakeResponses(self)


def _raises(exception_type, callable_, *args, **kwargs):
    try:
        callable_(*args, **kwargs)
    except exception_type as error:
        return error
    raise AssertionError(f"expected {exception_type.__name__} but nothing was raised")


# ---------------------------------------------------------------------------
# Evidence packet schema
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_evidence_packet_has_the_documented_top_level_sections():
    facts = _packet().facts

    assert set(facts) == {
        "metric_name",
        "current_period",
        "previous_period",
        "metric",
        "data_quality",
        "drivers",
        "text_themes",
        "limitations",
    }


@pytest.mark.slow
def test_evidence_packet_carries_every_required_metric_field():
    metric = _packet().facts["metric"]

    for key in (
        "raw_current_rate",
        "raw_previous_rate",
        "corrected_current_rate",
        "corrected_previous_rate",
        "raw_percent_change",
        "corrected_percent_change",
        "duplicate_rows_removed",
        "raw_current_disputed",
        "corrected_current_disputed",
    ):
        assert key in metric, key
        assert metric[key] is not None, key


@pytest.mark.slow
def test_evidence_packet_carries_quality_drivers_and_themes():
    facts = _packet().facts

    assert facts["data_quality"]["failing_checks"], "no failing checks captured"
    assert facts["drivers"]["top_count_drivers"], "no count drivers captured"
    assert facts["drivers"]["top_rate_drivers"], "no rate drivers captured"
    assert facts["drivers"]["top_interaction_driver"] is not None
    assert facts["text_themes"]["top_themes"], "no themes captured"
    assert facts["text_themes"]["representative_complaints"], "no example complaints captured"
    assert facts["limitations"] == list(STANDING_LIMITATIONS)


@pytest.mark.slow
def test_evidence_packet_matches_the_underlying_reports():
    packet = _packet()
    impact = build_metric_report(_tables()).remediation_impact.iloc[0]
    metric = packet.facts["metric"]

    assert metric["raw_current_disputed"] == int(impact["raw_disputed_count"])
    assert metric["duplicate_rows_removed"] == int(impact["duplicate_disputed_rows_removed"])
    np.testing.assert_allclose(metric["raw_current_rate"], float(impact["raw_dispute_rate"]), atol=1e-6)


@pytest.mark.slow
def test_evidence_packet_periods_are_the_two_most_recent_months():
    packet = _packet()

    assert packet.current_period == SPIKE_MONTH
    assert packet.previous_period == PRIOR_MONTH
    assert packet.metric_name == "dispute_rate"


@pytest.mark.slow
def test_evidence_packet_is_json_serialisable():
    payload = json.loads(_packet().to_json())

    assert payload["current_period"] == SPIKE_MONTH
    assert isinstance(payload["metric"]["raw_current_rate"], float)


@pytest.mark.slow
def test_evidence_packet_whitelists_its_own_numbers():
    packet = _packet()
    metric = packet.facts["metric"]

    assert len(packet.supported_numbers) > 50
    assert packet.supported_numbers.contains(metric["raw_current_disputed"])
    assert packet.supported_numbers.contains(metric["raw_current_rate"])
    # Rates are registered as percentages too, because that is how prose renders them.
    assert packet.supported_numbers.contains(metric["raw_current_rate"] * 100)


@pytest.mark.slow
def test_evidence_packet_can_build_itself_from_tables():
    # Every report omitted, so the packet computes them. Kept small by reusing
    # the already-loaded tables rather than re-reading the CSVs.
    packet = build_evidence_packet(
        metric_report=build_metric_report(_tables()),
        quality_report=run_quality_checks(_tables()),
        driver_report=build_driver_report(tables=_tables(), include_decision_tree=False),
        theme_report=build_text_theme_report(
            tables=_tables(), embedder=HashingEmbedder(), include_clusters=False
        ),
    )

    assert packet.facts["metric"]["duplicate_rows_removed"] == 165


# ---------------------------------------------------------------------------
# Number whitelist and validation
# ---------------------------------------------------------------------------


def test_supported_numbers_accepts_display_rounding():
    supported = SupportedNumbers()
    supported.add_rate(0.017679, "rate")

    assert supported.contains(0.017679)
    assert supported.contains(0.0177)
    assert supported.contains(1.7679)
    assert supported.contains(1.77)
    assert supported.contains(1.8)


def test_supported_numbers_accepts_negative_change_as_a_positive_magnitude():
    supported = SupportedNumbers()
    supported.add_rate(-0.0524, "corrected_percent_change")

    assert supported.contains(-5.24)
    assert supported.contains(5.24)
    assert validate_numbers("The metric fell 5.24%.", supported).is_valid


def test_supported_numbers_rejects_a_different_value():
    supported = SupportedNumbers()
    supported.add_rate(0.017679, "rate")

    assert not supported.contains(2.1)
    assert not supported.contains(4200)


def test_supported_numbers_records_a_label_for_traceability():
    supported = SupportedNumbers()
    supported.add(1871, "metric.raw_current_disputed")

    assert supported.label_for(1871) == "metric.raw_current_disputed"


def test_number_extraction_handles_currency_percent_and_separators():
    tokens = extract_numbers("Rate 1.77% on 105,834 purchases costing $12.50 each")

    assert "1.77%" in tokens
    assert "105,834" in tokens
    assert "$12.50" in tokens


def test_number_extraction_ignores_period_labels():
    # "2026-08" is a label; splitting it would invent a 2026 and an 08.
    tokens = extract_numbers("In 2026-08 the rate was 1.77%")

    assert tokens == ["1.77%"]


def test_validation_passes_when_every_number_is_supported():
    supported = SupportedNumbers()
    supported.add_rate(0.0177, "rate")
    supported.add(1871, "count")

    result = validate_numbers("The rate was 1.77% across 1,871 disputes.", supported)

    assert result.status == STATUS_PASS
    assert result.is_valid
    assert not result.unsupported


def test_validation_catches_an_invented_number():
    supported = SupportedNumbers()
    supported.add_rate(0.0177, "rate")

    result = validate_numbers("The rate was 1.77% across 4,200 disputes.", supported)

    assert result.status == STATUS_FAIL
    assert not result.is_valid
    assert "4,200" in result.unsupported
    assert "4,200" in result.notes


def test_validation_catches_an_invented_dollar_amount():
    supported = SupportedNumbers()
    supported.add(165, "duplicates")

    result = validate_numbers("165 duplicates cost roughly $48,000 in write-offs.", supported)

    assert result.status == STATUS_FAIL
    assert "$48,000" in result.unsupported


def test_validation_treats_small_integers_as_structural_but_reports_them():
    supported = SupportedNumbers()
    supported.add_rate(0.0177, "rate")

    result = validate_numbers("The top 3 segments explain the 1.77% rate.", supported)

    assert result.is_valid
    assert "3" in result.ignored
    assert "small integer" in result.notes


def test_a_large_invented_integer_is_never_treated_as_structural():
    supported = SupportedNumbers()
    result = validate_numbers("There were 9000 disputes.", supported)

    assert not result.is_valid
    assert "9000" in result.unsupported


def test_the_structural_threshold_is_configurable():
    supported = SupportedNumbers()

    lenient = validate_numbers("We saw 7 issues.", supported, material_integer_threshold=10)
    strict = validate_numbers("We saw 7 issues.", supported, material_integer_threshold=0)

    assert lenient.is_valid
    assert not strict.is_valid


def test_a_small_integer_that_is_real_evidence_is_still_checked():
    supported = SupportedNumbers()
    supported.add(5, "quality.failed")

    result = validate_numbers("5 checks failed.", supported)

    assert result.is_valid
    assert "5" in result.supported
    assert "5" not in result.ignored


# ---------------------------------------------------------------------------
# Deterministic explanation
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_deterministic_explanation_has_every_required_field():
    explanation = _explanation()

    for field_name in EXPLANATION_FIELDS:
        assert hasattr(explanation, field_name), field_name
    payload = explanation.to_dict()
    assert set(EXPLANATION_FIELDS).issubset(payload)
    assert "validation" in payload


@pytest.mark.slow
def test_deterministic_explanation_needs_no_api_key():
    assert not os.environ.get("OPENAI_API_KEY")

    explanation = explain(_packet())

    assert explanation.generated_by == GENERATOR_DETERMINISTIC
    assert explanation.executive_summary


@pytest.mark.slow
def test_deterministic_explanation_passes_its_own_number_validation():
    validation = _explanation().validation

    assert validation.is_valid, validation.notes
    assert len(validation.supported) > 10


@pytest.mark.slow
def test_explanation_states_the_raw_and_corrected_rates():
    text = _explanation().narrative_text()
    metric = _packet().facts["metric"]

    assert f"{metric['raw_current_rate'] * 100:.2f}%" in text
    assert f"{metric['corrected_current_rate'] * 100:.2f}%" in text
    assert f"{metric['raw_previous_rate'] * 100:.2f}%" in text


@pytest.mark.slow
def test_explanation_states_the_key_counts():
    text = _explanation().narrative_text()
    metric = _packet().facts["metric"]

    assert f"{metric['raw_current_disputed']:,}" in text
    assert f"{metric['corrected_current_disputed']:,}" in text
    assert f"{metric['duplicate_rows_removed']:,}" in text


@pytest.mark.slow
def test_explanation_reports_the_replay_and_the_corrected_metric():
    explanation = _explanation()
    text = explanation.narrative_text().lower()

    assert "duplicat" in text
    assert "corrected" in text
    # The headline has to carry the split, not bury it in a later paragraph.
    assert "duplicate" in explanation.headline.lower()


@pytest.mark.slow
def test_explanation_separates_data_quality_from_real_movement():
    text = _explanation().narrative_text().lower()

    assert "data-quality artifact" in text or "data quality" in text
    assert "part of it is real" in text or "remains" in text


@pytest.mark.slow
def test_explanation_names_the_travel_and_mobile_drivers():
    explanation = _explanation()
    drivers = explanation.business_drivers.lower()

    assert "travel" in drivers
    assert "mobile" in drivers
    assert "merchant_category" in drivers


@pytest.mark.slow
def test_explanation_cites_complaint_themes_with_an_example():
    explanation = _explanation()
    evidence = explanation.customer_text_evidence

    top_theme = _packet().facts["text_themes"]["top_themes"][0]["theme_name"]
    assert top_theme in evidence
    assert '"' in evidence, "no representative complaint quoted"


@pytest.mark.slow
def test_representative_example_supports_the_lead_theme():
    explanation = _explanation()
    text_facts = _packet().facts["text_themes"]
    top_theme = text_facts["top_themes"][0]["theme_name"]
    matching_examples = [
        item["narrative"]
        for item in text_facts["representative_complaints"]
        if item["theme_name"] == top_theme
    ]

    assert matching_examples
    assert any(example in explanation.customer_text_evidence for example in matching_examples)


@pytest.mark.slow
def test_explanation_lists_actionable_next_steps():
    steps = _explanation().recommended_next_steps

    assert len(steps) >= 3
    assert any("dedup" in step.lower() for step in steps)


@pytest.mark.slow
def test_explanation_carries_the_standing_limitations():
    explanation = _explanation()

    assert explanation.limitations == tuple(STANDING_LIMITATIONS)
    joined = " ".join(explanation.limitations).lower()
    assert "not causal proof" in joined
    assert "synthetic" in joined


@pytest.mark.slow
def test_explanation_references_each_contributing_module():
    references = " ".join(_explanation().evidence_references)

    for module in ("metric_engine", "quality_checks", "driver_analysis", "text_theme_analysis"):
        assert module in references, module


@pytest.mark.slow
def test_explanation_renders_as_markdown_with_the_validation_badge():
    markdown = _explanation().to_markdown()

    assert markdown.startswith("# ")
    assert "## Recommended next steps" in markdown
    assert "Number check: pass" in markdown


@pytest.mark.slow
def test_deterministic_explanation_is_reproducible():
    first = build_deterministic_explanation(_packet())
    second = build_deterministic_explanation(_packet())

    assert first.to_dict() == second.to_dict()


@pytest.mark.slow
def test_toy_packet_explanation_uses_only_toy_numbers():
    explanation = build_deterministic_explanation(_toy_packet())

    assert explanation.validation.is_valid, explanation.validation.notes
    assert "2.00%" in explanation.narrative_text()
    assert "1.50%" in explanation.narrative_text()


@pytest.mark.slow
def test_explanation_handles_a_clean_month_with_no_duplicates():
    packet = _toy_packet()
    packet.facts["metric"]["duplicate_rows_removed"] = 0
    packet.facts["data_quality"].update({"failing_checks": [], "warning_checks": [], "failed": 0})
    packet.supported_numbers.add(0, "metric.duplicate_rows_removed")

    explanation = build_deterministic_explanation(packet)

    assert "no duplicate" in explanation.headline.lower()
    assert "passed" in explanation.data_quality_findings.lower()


# ---------------------------------------------------------------------------
# OpenAI path, driven by an injected fake client
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_llm_path_uses_the_injected_client_and_never_the_network():
    client = FakeOpenAIClient(_valid_llm_payload())

    explanation = build_llm_explanation(_toy_packet(), client=client)

    assert len(client.calls) == 1
    assert explanation.generated_by == f"openai:{DEFAULT_MODEL}"
    assert explanation.headline == _valid_llm_payload()["headline"]


@pytest.mark.slow
def test_llm_request_carries_the_grounding_instruction_and_the_packet():
    client = FakeOpenAIClient(_valid_llm_payload())

    build_llm_explanation(_toy_packet(), client=client)
    request = client.calls[0]
    system_message, user_message = request["input"]

    assert system_message["role"] == "system"
    assert "Do NOT introduce any number" in system_message["content"]
    assert "Do NOT calculate anything" in system_message["content"]
    assert user_message["role"] == "user"
    assert "dispute_rate" in user_message["content"]
    assert "duplicate_rows_removed" in user_message["content"]


@pytest.mark.slow
def test_llm_request_asks_for_low_temperature_and_structured_json():
    client = FakeOpenAIClient(_valid_llm_payload())

    build_llm_explanation(_toy_packet(), client=client)
    request = client.calls[0]

    assert request["temperature"] <= 0.2
    text_format = request["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    assert text_format["schema"] == EXPLANATION_JSON_SCHEMA


def test_llm_json_schema_requires_every_output_field():
    required = set(EXPLANATION_JSON_SCHEMA["required"])
    expected = set(EXPLANATION_FIELDS) - {"generated_by"}

    assert required == expected
    assert EXPLANATION_JSON_SCHEMA["additionalProperties"] is False


@pytest.mark.slow
def test_model_is_configurable_by_environment_variable():
    previous = os.environ.get(DEFAULT_MODEL_ENV_VAR)
    os.environ[DEFAULT_MODEL_ENV_VAR] = "gpt-test-model"
    try:
        assert resolve_model() == "gpt-test-model"
        client = FakeOpenAIClient(_valid_llm_payload())
        explanation = build_llm_explanation(_toy_packet(), client=client)
        assert client.calls[0]["model"] == "gpt-test-model"
        assert explanation.generated_by == "openai:gpt-test-model"
    finally:
        if previous is None:
            os.environ.pop(DEFAULT_MODEL_ENV_VAR, None)
        else:
            os.environ[DEFAULT_MODEL_ENV_VAR] = previous


@pytest.mark.slow
def test_an_explicit_model_argument_wins_over_the_environment():
    client = FakeOpenAIClient(_valid_llm_payload())

    build_llm_explanation(_toy_packet(), client=client, model="gpt-explicit")

    assert client.calls[0]["model"] == "gpt-explicit"


@pytest.mark.slow
def test_a_grounded_model_response_passes_validation():
    client = FakeOpenAIClient(_valid_llm_payload())

    explanation = build_llm_explanation(_toy_packet(), client=client)

    assert explanation.validation.is_valid, explanation.validation.notes


@pytest.mark.slow
def test_an_invented_number_from_the_model_is_caught():
    payload = _valid_llm_payload()
    payload["business_drivers"] = "Travel disputes cost the portfolio $4,200,000 last month."
    client = FakeOpenAIClient(payload)

    explanation = build_llm_explanation(_toy_packet(), client=client)

    assert not explanation.validation.is_valid
    assert any("4,200,000" in token for token in explanation.validation.unsupported)


@pytest.mark.slow
def test_flag_mode_keeps_the_ungrounded_explanation_but_marks_it():
    payload = _valid_llm_payload()
    payload["what_changed"] = "Roughly 7,777 disputes were filed."
    client = FakeOpenAIClient(payload)

    explanation = explain(_toy_packet(), client=client, on_unsupported_numbers="flag")

    assert explanation.generated_by.startswith("openai:")
    assert not explanation.validation.is_valid
    assert "7,777" in explanation.validation.unsupported


@pytest.mark.slow
def test_fallback_mode_replaces_an_ungrounded_explanation():
    payload = _valid_llm_payload()
    payload["what_changed"] = "Roughly 7,777 disputes were filed."
    client = FakeOpenAIClient(payload)

    explanation = explain(_toy_packet(), client=client, on_unsupported_numbers="fallback")

    assert explanation.generated_by == GENERATOR_DETERMINISTIC
    assert explanation.validation.is_valid


@pytest.mark.slow
def test_raise_mode_refuses_an_ungrounded_explanation():
    payload = _valid_llm_payload()
    payload["what_changed"] = "Roughly 7,777 disputes were filed."
    client = FakeOpenAIClient(payload)

    error = _raises(
        ValueError, explain, _toy_packet(), client, None, None, "raise"
    )

    assert "7,777" in str(error)


@pytest.mark.slow
def test_explain_rejects_an_unknown_unsupported_number_policy():
    error = _raises(
        ValueError, explain, _toy_packet(), None, None, False, "ignore-everything"
    )

    assert "on_unsupported_numbers" in str(error)


@pytest.mark.slow
def test_a_model_response_missing_a_field_is_rejected():
    payload = _valid_llm_payload()
    del payload["business_drivers"]
    client = FakeOpenAIClient(payload)

    error = _raises(ValueError, build_llm_explanation, _toy_packet(), client)

    assert "business_drivers" in str(error)


@pytest.mark.slow
def test_explain_defaults_to_the_deterministic_path_without_a_key_or_client():
    assert not os.environ.get("OPENAI_API_KEY")

    explanation = explain(_toy_packet())

    assert explanation.generated_by == GENERATOR_DETERMINISTIC


@pytest.mark.slow
def test_explain_uses_the_llm_when_a_client_is_injected():
    client = FakeOpenAIClient(_valid_llm_payload())

    explanation = explain(_toy_packet(), client=client)

    assert explanation.generated_by.startswith("openai:")


@pytest.mark.slow
def test_use_llm_false_ignores_an_injected_client():
    client = FakeOpenAIClient(_valid_llm_payload())

    explanation = explain(_toy_packet(), client=client, use_llm=False)

    assert explanation.generated_by == GENERATOR_DETERMINISTIC
    assert client.calls == []


# ---------------------------------------------------------------------------
# Generated data must not be touched
# ---------------------------------------------------------------------------


def _csv_fingerprints() -> dict[str, tuple[str, int]]:
    return {
        name: (
            hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest(),
            (DATA_DIR / name).stat().st_size,
        )
        for name in CSV_FILES
    }


@pytest.mark.slow
def test_building_an_explanation_does_not_modify_any_generated_csv():
    before = _csv_fingerprints()

    explanation = explain(_packet())
    explanation.to_markdown()

    assert _csv_fingerprints() == before


@pytest.mark.slow
def test_the_explanation_layer_never_writes_to_the_data_directory():
    before = sorted(path.name for path in DATA_DIR.iterdir())

    build_deterministic_explanation(_packet())

    assert sorted(path.name for path in DATA_DIR.iterdir()) == before


# ---------------------------------------------------------------------------
# Direct runner, used when pytest is unavailable
# ---------------------------------------------------------------------------


def _run_without_pytest() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]

    failures: list[tuple[str, BaseException]] = []
    for name, test in tests:
        try:
            test()
        except BaseException as error:  # noqa: BLE001 - a runner reports everything
            failures.append((name, error))
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok   {name}")

    print()
    print(f"{len(tests) - len(failures)} passed, {len(failures)} failed, {len(tests)} total")
    if failures:
        print()
        print("Failures:")
        for name, error in failures:
            print(f"  - {name}: {type(error).__name__}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_without_pytest())
