"""Tests for the research-only Expresso bootstrap helper."""

import re

import pytest

from scripts.bootstrap_expresso_research_bank import (
    LICENSE_ID,
    _FILE_RE,
    _choose_speaker,
    _research_notice,
    _speaker_for_file,
)


def test_expresso_filename_channel_maps_to_correct_speaker() -> None:
    channel1 = _FILE_RE.match(
        "expresso/ex04-ex02_confused_001_channel1_499s.wav"
    )
    channel2 = _FILE_RE.match(
        "expresso/ex04-ex02_confused_001_channel2_499s.wav"
    )

    assert isinstance(channel1, re.Match)
    assert isinstance(channel2, re.Match)
    assert _speaker_for_file(channel1) == "ex04"
    assert _speaker_for_file(channel2) == "ex02"


def test_choose_speaker_prefers_broadest_style_coverage() -> None:
    speaker, styles = _choose_speaker(
        {
            "ex01": {
                "default": "neutral.wav",
                "whisper": "whisper.wav",
            },
            "ex02": {
                "default": "neutral.wav",
                "whisper": "whisper.wav",
                "confused": "confused.wav",
            },
        },
        requested=None,
    )

    assert speaker == "ex02"
    assert set(styles) == {"default", "whisper", "confused"}


def test_choose_speaker_rejects_neutral_only_bank() -> None:
    with pytest.raises(ValueError, match="no expressive style"):
        _choose_speaker(
            {"ex01": {"default": "neutral.wav"}},
            requested="ex01",
        )


def test_research_notice_is_explicit_about_noncommercial_scope() -> None:
    notice = _research_notice(
        revision="abc123",
        speaker="ex04",
        selected_sources=["expresso/example.wav"],
    )

    assert LICENSE_ID in notice
    assert "non-commercial research/evaluation only" in notice
    assert "expresso/example.wav" in notice
    assert "replace this bank" in notice
