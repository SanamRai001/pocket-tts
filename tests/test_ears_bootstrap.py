"""Tests for the EARS same-speaker research bootstrap."""

import wave
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile

from scripts.bootstrap_ears_research_bank import (
    DEFAULT_SPEAKER,
    STYLES,
    _write_pcm16_wav,
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


def test_ears_float_wav_is_normalized_to_pcm16(tmp_path: Path) -> None:
    source = tmp_path / "float.wav"
    destination = tmp_path / "pcm16.wav"
    samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    scipy.io.wavfile.write(source, 24000, samples)

    _write_pcm16_wav(source, destination)

    with wave.open(str(destination), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1

    sample_rate, converted = scipy.io.wavfile.read(destination)
    assert sample_rate == 24000
    assert converted.dtype == np.int16
    assert converted[0] == -32767
    assert converted[-1] == 32767
