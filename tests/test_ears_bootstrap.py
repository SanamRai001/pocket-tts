"""Tests for the EARS same-speaker research bootstrap."""

import pytest

from scripts.bootstrap_ears_research_bank import (
    DEFAULT_SPEAKER,
    STYLES,
    build_manifest,
    source_path,
)


def test_ears_manifest_matches_core_eight_style_bank() -> None:
    manifest = build_manifest(DEFAULT_SPEAKER)

    styles = manifest["styles"]
    assert isinstance(styles, list)
    assert [style["name"] for style in styles] == [
        "neutral",
        "warm",
        "happy",
        "excited",
        "calm",
        "sad",
        "angry",
        "fearful",
    ]
    assert len(STYLES) == 8


def test_ears_source_path_uses_native_emotion_and_optional_enhancement() -> None:
    assert source_path("p031", "sadness") == "ears/p031/emo_sadness_freeform.wav"
    assert (
        source_path("p003", "anger", enhanced=True)
        == "ears/p003/emo_anger_freeform_enhanced.wav"
    )


def test_ears_source_path_rejects_non_emotion_speaker() -> None:
    with pytest.raises(ValueError, match="unsupported EARS emotion speaker"):
        source_path("p010", "neutral")
