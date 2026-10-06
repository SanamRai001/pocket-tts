"""Tests for the versioned expressive planner interchange."""

import json

import pytest

from pocket_tts.expressive import (
    AffectVector,
    AffectivePromptRouter,
    ExpressivePlan,
    ExpressiveSegment,
    StyleProfile,
)
from pocket_tts.expressive_planner import (
    ExpressivePlanValidationError,
    build_planner_prompt,
    heuristic_plan,
    plan_from_dict,
    plan_from_json,
    plan_to_json,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "segments": [
            {
                "text": "We actually won!",
                "affect": {
                    "valence": 0.9,
                    "arousal": 0.85,
                    "dominance": 0.35,
                    "intensity": 0.95,
                },
                "style": None,
                "pause_after_ms": 180,
                "confidence": 0.9,
            },
            {
                "text": "I wish Dad were here.",
                "affect": {
                    "valence": -0.7,
                    "arousal": -0.35,
                    "dominance": -0.25,
                    "intensity": 0.7,
                },
                "style": None,
                "pause_after_ms": 0,
                "confidence": 0.95,
            },
        ],
    }


def test_plan_from_dict_parses_valid_versioned_payload() -> None:
    source = "We actually won! I wish Dad were here."

    plan = plan_from_dict(_payload(), source_text=source)

    assert len(plan.segments) == 2
    assert plan.segments[0].pause_after_ms == 180
    assert plan.segments[0].confidence == 0.9
    assert plan.segments[1].affect.valence == -0.7


def test_plan_rejects_planner_rewrite_when_source_text_is_provided() -> None:
    payload = _payload()
    segments = payload["segments"]
    assert isinstance(segments, list)
    first = segments[0]
    assert isinstance(first, dict)
    first["text"] = "We finally won!"

    with pytest.raises(ExpressivePlanValidationError, match="changed the spoken text"):
        plan_from_dict(payload, source_text="We actually won! I wish Dad were here.")


def test_plan_rejects_unknown_fields_instead_of_silently_ignoring_them() -> None:
    payload = _payload()
    payload["secret_instruction"] = "make it dramatic"

    with pytest.raises(ExpressivePlanValidationError, match="unsupported field"):
        plan_from_dict(payload)


def test_plan_rejects_unavailable_exact_style_override() -> None:
    payload = _payload()
    segments = payload["segments"]
    assert isinstance(segments, list)
    first = segments[0]
    assert isinstance(first, dict)
    first["style"] = "whisper"

    with pytest.raises(ExpressivePlanValidationError, match="not available"):
        plan_from_dict(payload, allowed_styles={"neutral", "happy", "sad"})


def test_low_confidence_damps_affect_toward_neutral() -> None:
    router = AffectivePromptRouter(
        (
            StyleProfile("neutral", "neutral.wav", AffectVector()),
            StyleProfile(
                "excited",
                "excited.wav",
                AffectVector(valence=0.9, arousal=0.9, dominance=0.4, intensity=0.9),
            ),
        ),
        default_style="neutral",
    )
    uncertain = ExpressiveSegment(
        "Maybe this is good news.",
        affect=AffectVector(valence=0.9, arousal=0.9, dominance=0.4, intensity=0.9),
        confidence=0.0,
    )

    assert router.resolve(uncertain).name == "neutral"


def test_plan_json_round_trip_preserves_contract_fields() -> None:
    plan = ExpressivePlan(
        segments=(
            ExpressiveSegment(
                "Hello there.",
                affect=AffectVector(valence=0.2, arousal=0.1, intensity=0.3),
                pause_after_ms=50,
                confidence=0.75,
            ),
        )
    )

    encoded = plan_to_json(plan)
    decoded = plan_from_json(encoded, source_text="Hello there.")

    assert decoded == plan
    assert json.loads(encoded)["schema_version"] == "1.0"


def test_planner_prompt_is_provider_neutral_and_contains_style_bank() -> None:
    prompt = build_planner_prompt(
        "I cannot believe it.",
        available_styles={"neutral", "happy", "sad"},
    )

    assert "Return exactly one JSON object" in prompt
    assert "Preserve every spoken word" in prompt
    assert '"happy"' in prompt
    assert "I cannot believe it." in prompt


def test_heuristic_plan_is_conservative_and_preserves_text() -> None:
    source = "We actually won! I wish Dad were here."

    plan = heuristic_plan(source)

    assert len(plan.segments) == 2
    assert plan.segments[0].affect.valence > 0
    assert plan.segments[0].affect.arousal > 0
    assert plan.segments[1].affect.valence < 0
    assert all(segment.confidence <= 0.55 for segment in plan.segments)


def test_heuristic_plan_defaults_to_near_neutral_without_emotional_cues() -> None:
    plan = heuristic_plan("The package is on the table.")

    segment = plan.segments[0]
    assert segment.affect == AffectVector()
    assert segment.confidence < 0.3
