"""Tests for lightweight expressive route preview."""

import json
from pathlib import Path

from scripts.inspect_expressive_routes import build_route_preview


def test_route_preview_maps_positive_high_arousal_to_happy(tmp_path: Path) -> None:
    bank = {
        "schema_version": "1.0",
        "bank_id": "test-bank",
        "speaker_id": "speaker",
        "language": "english",
        "rights_confirmed": True,
        "recording_protocol_version": "1.0",
        "styles": [
            {
                "name": "neutral",
                "audio_path": "neutral.wav",
                "affect": {
                    "valence": 0.0,
                    "arousal": 0.0,
                    "dominance": 0.0,
                    "intensity": 0.0,
                },
            },
            {
                "name": "happy",
                "audio_path": "happy.wav",
                "affect": {
                    "valence": 0.85,
                    "arousal": 0.75,
                    "dominance": 0.2,
                    "intensity": 0.9,
                },
            },
        ],
    }
    corpus = {
        "schema_version": "1.0",
        "items": [
            {
                "id": "success",
                "purpose": "high-arousal positive",
                "text": "We did it!",
                "reference_plan": {
                    "schema_version": "1.0",
                    "segments": [
                        {
                            "text": "We did it!",
                            "affect": {
                                "valence": 0.9,
                                "arousal": 0.9,
                                "dominance": 0.2,
                                "intensity": 0.95,
                            },
                            "style": None,
                            "pause_after_ms": 0,
                            "confidence": 1.0,
                        }
                    ],
                },
            }
        ],
    }

    bank_path = tmp_path / "bank.json"
    corpus_path = tmp_path / "corpus.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")

    report = build_route_preview(bank_path, corpus_path)

    items = report["items"]
    assert isinstance(items, list)
    first = items[0]
    assert isinstance(first, dict)
    assert first["route"] == ["happy"]
    assert report["style_usage"] == {"happy": 1}
    warnings = report["warnings"]
    assert isinstance(warnings, list)
    assert not any("routes to neutral" in str(warning) for warning in warnings)
