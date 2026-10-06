"""Tests for dependency-light expressive evaluation tooling."""

import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile

from scripts.score_expressive_eval import _technical_metrics
from scripts.summarize_expressive_blind_eval import _score


def test_technical_metrics_detect_duration_level_and_clipping(tmp_path: Path) -> None:
    sample_rate = 1000
    audio = np.zeros(1000, dtype=np.float32)
    audio[:100] = 1.0
    path = tmp_path / "clip.wav"
    scipy.io.wavfile.write(path, sample_rate, audio)

    metrics = _technical_metrics(path)

    assert metrics["sample_rate"] == 1000
    assert metrics["samples"] == 1000
    assert metrics["duration_seconds"] == 1.0
    assert metrics["finite"] is True
    assert metrics["peak_abs"] == 1.0
    assert metrics["near_clip_fraction"] == 0.1
    assert metrics["silence_fraction"] == 0.9


def test_technical_metrics_handles_integer_pcm(tmp_path: Path) -> None:
    path = tmp_path / "pcm16.wav"
    scipy.io.wavfile.write(
        path,
        8000,
        np.array([0, 16384, -16384, 32767], dtype=np.int16),
    )

    metrics = _technical_metrics(path)

    assert metrics["sample_rate"] == 8000
    assert metrics["finite"] is True
    assert 0.99 < float(metrics["peak_abs"]) <= 1.0


def test_human_rating_score_parser_accepts_blank_and_valid_scores() -> None:
    assert _score("", "naturalness_a_1_5", "item") is None
    assert _score("4", "naturalness_a_1_5", "item") == 4.0


def test_blind_answer_key_shape_is_json_friendly(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1.0",
        "answers": [{"id": "one", "A": "expressive", "B": "baseline-neutral"}],
    }
    path = tmp_path / "answer-key.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    decoded = json.loads(path.read_text(encoding="utf-8"))

    assert decoded["answers"][0]["A"] == "expressive"
