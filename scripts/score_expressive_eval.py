"""Score an expressive Pocket TTS A/B run without adding runtime dependencies.

Tier 0 technical diagnostics use only Pocket TTS's existing dependencies.
Optional research metrics (ASR WER, WavLM speaker similarity, UTMOS) are
loaded only when explicitly requested and require the project's dev environment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io.wavfile
import scipy.signal
import torch

from pocket_tts.expressive_prompt_bank import load_prompt_bank


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a neutral-vs-expressive evaluation run."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory produced by scripts/generate_expressive_eval.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Score report path (default: <run_dir>/scores.json)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for optional neural scorers (default: cpu)",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help=(
            "Optional Hugging Face ASR model for WER. "
            "Example for CPU: openai/whisper-base.en"
        ),
    )
    parser.add_argument(
        "--speaker-similarity",
        action="store_true",
        help="Score speaker similarity with microsoft/wavlm-base-plus-sv",
    )
    parser.add_argument(
        "--utmos",
        action="store_true",
        help="Score UTMOS quality (requires utmos-pytorch dev dependency)",
    )
    return parser


def _load_float_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, raw = scipy.io.wavfile.read(path)
    audio = np.asarray(raw)

    if audio.ndim == 2:
        audio = audio.astype(np.float64).mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32, copy=False)

    if audio.ndim != 1:
        raise ValueError(f"{path} is not mono-compatible: shape={audio.shape}")
    if audio.size == 0:
        raise ValueError(f"{path} contains no audio samples")
    return int(sample_rate), audio


def _resample(audio: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    return scipy.signal.resample_poly(audio, up, down).astype(np.float32)


def _technical_metrics(path: Path) -> dict[str, float | int | bool]:
    sample_rate, audio = _load_float_mono(path)
    finite = bool(np.isfinite(audio).all())
    if not finite:
        finite_audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        finite_audio = audio

    absolute = np.abs(finite_audio)
    peak = float(absolute.max())
    rms = float(np.sqrt(np.mean(np.square(finite_audio, dtype=np.float64))))
    rms_dbfs = float(20.0 * math.log10(max(rms, 1e-12)))
    return {
        "sample_rate": sample_rate,
        "samples": int(audio.size),
        "duration_seconds": float(audio.size / sample_rate),
        "finite": finite,
        "peak_abs": peak,
        "rms": rms,
        "rms_dbfs": rms_dbfs,
        "dc_offset": float(np.mean(finite_audio)),
        "near_clip_fraction": float(np.mean(absolute >= 0.999)),
        "silence_fraction": float(np.mean(absolute < 1e-4)),
    }


def _device_for_pipeline(device: str) -> int | str:
    lowered = device.lower()
    if lowered == "cpu":
        return -1
    if lowered.startswith("cuda:"):
        return int(lowered.split(":", 1)[1])
    if lowered == "cuda":
        return 0
    return device


def _build_asr(model_name: str, device: str):
    try:
        import jiwer
        from transformers import pipeline
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError as exc:
        raise RuntimeError(
            "WER scoring requires the project dev dependencies "
            "(transformers, jiwer, whisper-normalizer)"
        ) from exc

    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=_device_for_pipeline(device),
    )
    normalize = EnglishTextNormalizer()

    def transcribe(path: Path) -> tuple[str, str]:
        sample_rate, audio = _load_float_mono(path)
        audio16 = _resample(audio, sample_rate, 16000)
        result = recognizer({"array": audio16, "sampling_rate": 16000})
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise RuntimeError(f"ASR returned an unexpected result for {path}")
        raw_text = result["text"]
        return raw_text, normalize(raw_text)

    return transcribe, normalize, jiwer


def _build_speaker_embedder(device: torch.device):
    try:
        from transformers import AutoFeatureExtractor, WavLMForXVector
    except ImportError as exc:
        raise RuntimeError(
            "speaker similarity requires the project dev dependency transformers"
        ) from exc

    model_name = "microsoft/wavlm-base-plus-sv"
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = WavLMForXVector.from_pretrained(model_name).to(device).eval()

    def embed(path: Path) -> torch.Tensor:
        sample_rate, audio = _load_float_mono(path)
        audio16 = _resample(audio, sample_rate, 16000)
        inputs = extractor(
            audio16,
            sampling_rate=16000,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            return model(**inputs).embeddings[0]

    return embed


def _build_utmos(device: torch.device):
    try:
        from utmos_pytorch import UTMOSScoreTorch
    except ImportError as exc:
        raise RuntimeError(
            "UTMOS scoring requires the project dev dependency utmos-pytorch"
        ) from exc

    scorer = UTMOSScoreTorch(device=str(device))

    def score(path: Path) -> float:
        sample_rate, audio = _load_float_mono(path)
        audio16 = _resample(audio, sample_rate, 16000)
        waveform = torch.from_numpy(audio16)[None, None].to(device)
        with torch.no_grad():
            return float(scorer.score(waveform))

    return score


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _delta(expressive: float | None, baseline: float | None) -> float | None:
    if expressive is None or baseline is None:
        return None
    return float(expressive - baseline)


def main() -> None:
    args = _parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    items = report.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{report_path} contains no evaluation items")

    device = torch.device(args.device)
    asr = _build_asr(args.asr_model, args.device) if args.asr_model else None
    speaker_embed = _build_speaker_embedder(device) if args.speaker_similarity else None
    utmos = _build_utmos(device) if args.utmos else None

    speaker_reference_embedding: torch.Tensor | None = None
    if speaker_embed is not None:
        bank_path = Path(str(report["prompt_bank"]))
        bank = load_prompt_bank(bank_path)
        neutral_audio = bank.root / bank.entry("neutral").audio_path
        speaker_reference_embedding = speaker_embed(neutral_audio)

    scored_items: list[dict[str, Any]] = []
    baseline_wer_refs: list[str] = []
    baseline_wer_hyps: list[str] = []
    expressive_wer_refs: list[str] = []
    expressive_wer_hyps: list[str] = []

    for raw_item in items:
        if not isinstance(raw_item, dict):
            raise ValueError("report items must be objects")
        item_id = str(raw_item["id"])
        text = raw_item.get("text")
        if asr is not None and not isinstance(text, str):
            raise ValueError(
                "report.json does not contain reference text; regenerate the run "
                "with the current generate_expressive_eval.py"
            )

        baseline_path = run_dir / str(raw_item["baseline_wav"])
        expressive_path = run_dir / str(raw_item["expressive_wav"])

        baseline: dict[str, Any] = {
            "technical": _technical_metrics(baseline_path),
        }
        expressive: dict[str, Any] = {
            "technical": _technical_metrics(expressive_path),
        }

        if asr is not None:
            transcribe, normalize, jiwer = asr
            baseline_raw, baseline_hyp = transcribe(baseline_path)
            expressive_raw, expressive_hyp = transcribe(expressive_path)
            reference = normalize(text)
            baseline["asr_text"] = baseline_raw
            expressive["asr_text"] = expressive_raw
            baseline["wer"] = float(jiwer.wer(reference, baseline_hyp))
            expressive["wer"] = float(jiwer.wer(reference, expressive_hyp))
            baseline_wer_refs.append(reference)
            baseline_wer_hyps.append(baseline_hyp)
            expressive_wer_refs.append(reference)
            expressive_wer_hyps.append(expressive_hyp)

        if speaker_embed is not None:
            assert speaker_reference_embedding is not None
            baseline_embedding = speaker_embed(baseline_path)
            expressive_embedding = speaker_embed(expressive_path)
            baseline["speaker_similarity"] = float(
                torch.nn.functional.cosine_similarity(
                    baseline_embedding[None],
                    speaker_reference_embedding[None],
                    dim=-1,
                ).item()
            )
            expressive["speaker_similarity"] = float(
                torch.nn.functional.cosine_similarity(
                    expressive_embedding[None],
                    speaker_reference_embedding[None],
                    dim=-1,
                ).item()
            )

        if utmos is not None:
            baseline["utmos"] = utmos(baseline_path)
            expressive["utmos"] = utmos(expressive_path)

        scored_items.append(
            {
                "id": item_id,
                "purpose": raw_item.get("purpose"),
                "route": raw_item.get("route"),
                "baseline": baseline,
                "expressive": expressive,
            }
        )

    baseline_first = [
        float(item["baseline_first_chunk_seconds"])
        for item in items
        if isinstance(item, dict) and item.get("baseline_first_chunk_seconds") is not None
    ]
    expressive_first = [
        float(item["expressive_first_chunk_seconds"])
        for item in items
        if isinstance(item, dict) and item.get("expressive_first_chunk_seconds") is not None
    ]
    baseline_speed = [
        float(item["baseline_speed_x"])
        for item in items
        if isinstance(item, dict) and item.get("baseline_speed_x") is not None
    ]
    expressive_speed = [
        float(item["expressive_speed_x"])
        for item in items
        if isinstance(item, dict) and item.get("expressive_speed_x") is not None
    ]

    baseline_rms = [
        float(item["baseline"]["technical"]["rms_dbfs"]) for item in scored_items
    ]
    expressive_rms = [
        float(item["expressive"]["technical"]["rms_dbfs"]) for item in scored_items
    ]
    baseline_clip = [
        float(item["baseline"]["technical"]["near_clip_fraction"])
        for item in scored_items
    ]
    expressive_clip = [
        float(item["expressive"]["technical"]["near_clip_fraction"])
        for item in scored_items
    ]

    aggregate: dict[str, Any] = {
        "num_items": len(scored_items),
        "baseline_mean_first_chunk_seconds": _mean(baseline_first),
        "expressive_mean_first_chunk_seconds": _mean(expressive_first),
        "first_chunk_delta_seconds": _delta(
            _mean(expressive_first), _mean(baseline_first)
        ),
        "baseline_mean_speed_x": _mean(baseline_speed),
        "expressive_mean_speed_x": _mean(expressive_speed),
        "speed_x_delta": _delta(_mean(expressive_speed), _mean(baseline_speed)),
        "baseline_mean_rms_dbfs": _mean(baseline_rms),
        "expressive_mean_rms_dbfs": _mean(expressive_rms),
        "rms_dbfs_delta": _delta(_mean(expressive_rms), _mean(baseline_rms)),
        "baseline_mean_near_clip_fraction": _mean(baseline_clip),
        "expressive_mean_near_clip_fraction": _mean(expressive_clip),
    }

    if asr is not None:
        _, _, jiwer = asr
        aggregate["baseline_corpus_wer"] = float(
            jiwer.wer(baseline_wer_refs, baseline_wer_hyps)
        )
        aggregate["expressive_corpus_wer"] = float(
            jiwer.wer(expressive_wer_refs, expressive_wer_hyps)
        )
        aggregate["wer_delta"] = (
            aggregate["expressive_corpus_wer"] - aggregate["baseline_corpus_wer"]
        )

    if speaker_embed is not None:
        baseline_values = [
            float(item["baseline"]["speaker_similarity"]) for item in scored_items
        ]
        expressive_values = [
            float(item["expressive"]["speaker_similarity"]) for item in scored_items
        ]
        aggregate["baseline_mean_speaker_similarity"] = _mean(baseline_values)
        aggregate["expressive_mean_speaker_similarity"] = _mean(expressive_values)
        aggregate["speaker_similarity_delta"] = _delta(
            _mean(expressive_values), _mean(baseline_values)
        )

    if utmos is not None:
        baseline_values = [float(item["baseline"]["utmos"]) for item in scored_items]
        expressive_values = [float(item["expressive"]["utmos"]) for item in scored_items]
        aggregate["baseline_mean_utmos"] = _mean(baseline_values)
        aggregate["expressive_mean_utmos"] = _mean(expressive_values)
        aggregate["utmos_delta"] = _delta(
            _mean(expressive_values), _mean(baseline_values)
        )

    scored_report = {
        "schema_version": "1.0",
        "source_report": str(report_path),
        "model_ref": report.get("model_ref"),
        "bank_id": report.get("bank_id"),
        "optional_metrics": {
            "asr_model": args.asr_model,
            "speaker_similarity": args.speaker_similarity,
            "utmos": args.utmos,
            "device": args.device,
        },
        "aggregate": aggregate,
        "items": scored_items,
    }

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_dir / "scores.json"
    )
    output.write_text(json.dumps(scored_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"Scores: {output}")


if __name__ == "__main__":
    main()
