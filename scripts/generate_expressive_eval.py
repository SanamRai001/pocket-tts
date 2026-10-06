"""Generate neutral-vs-expressive audio for the fixed Phase 3A evaluation corpus."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import scipy.io.wavfile
import torch

from pocket_tts import TTSModel
from pocket_tts.expressive import AffectivePromptRouter, ExpressiveGenerator
from pocket_tts.expressive_planner import plan_from_dict
from pocket_tts.expressive_prompt_bank import (
    PromptBankValidationError,
    load_prompt_bank,
    style_profiles_from_bank,
    validate_bank_files,
)

_DEFAULT_CORPUS = Path("docs/expressive-speech/eval-corpus-v1.json")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate baseline neutral and affect-routed Pocket TTS audio from a "
            "compiled expressive prompt bank."
        )
    )
    parser.add_argument("prompt_bank", type=Path, help="Compiled prompt-bank manifest")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_DEFAULT_CORPUS,
        help=f"Evaluation corpus JSON (default: {_DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/expressive-eval"),
        help="Directory for generated WAVs and report.json",
    )
    model = parser.add_mutually_exclusive_group()
    model.add_argument(
        "--language",
        default=None,
        help="Pocket TTS language/config key (default: english)",
    )
    model.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Custom Pocket TTS YAML config",
    )
    parser.add_argument(
        "--max-cached-states",
        type=int,
        default=2,
        help="Maximum expressive prompt states held in memory (default: 2)",
    )
    return parser


def _load_corpus(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("evaluation corpus must be a version 1.0 object")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("evaluation corpus must contain a non-empty items array")

    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise ValueError(f"items[{index}] must be an object")
        item_id = raw.get("id")
        text = raw.get("text")
        plan = raw.get("reference_plan")
        if not isinstance(item_id, str) or not _SAFE_ID.fullmatch(item_id):
            raise ValueError(f"items[{index}].id must match {_SAFE_ID.pattern!r}")
        if item_id in seen:
            raise ValueError(f"duplicate evaluation item id {item_id!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"items[{index}].text must be a non-empty string")
        if not isinstance(plan, dict):
            raise ValueError(f"items[{index}].reference_plan must be an object")
        seen.add(item_id)
        result.append(raw)
    return result


def _write_wav(path: Path, sample_rate: int, audio: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(path, sample_rate, audio.detach().cpu().numpy())


def _timed_stream(
    function: Callable[[], Iterator[torch.Tensor]],
) -> tuple[torch.Tensor, float, float]:
    started = time.perf_counter()
    first_chunk_seconds: float | None = None
    chunks: list[torch.Tensor] = []
    for chunk in function():
        if first_chunk_seconds is None:
            first_chunk_seconds = time.perf_counter() - started
        chunks.append(chunk)
    elapsed = time.perf_counter() - started
    if not chunks:
        raise RuntimeError("generation produced no audio chunks")
    assert first_chunk_seconds is not None
    return torch.cat(chunks, dim=0), first_chunk_seconds, elapsed


def main() -> None:
    args = _parser().parse_args()
    if args.max_cached_states < 0:
        raise ValueError("--max-cached-states cannot be negative")

    if args.config is not None:
        config = args.config.resolve()
        model_ref = f"config:{config}"
        model = TTSModel.load_model(config=config)
    else:
        language = args.language or "english"
        model_ref = f"language:{language}"
        model = TTSModel.load_model(language=language)

    bank = load_prompt_bank(args.prompt_bank)
    problems = validate_bank_files(bank, require_states=True)
    if problems:
        raise PromptBankValidationError("\n".join(problems))

    profiles = style_profiles_from_bank(
        bank,
        expected_model_ref=model_ref,
        model=model,
    )
    by_name = {profile.name: profile for profile in profiles}
    neutral = by_name["neutral"]

    router = AffectivePromptRouter(profiles, default_style="neutral")
    expressive = ExpressiveGenerator(
        model,
        router,
        max_cached_states=args.max_cached_states,
    )
    neutral_state = model.get_state_for_audio_prompt(neutral.source)

    corpus = _load_corpus(args.corpus)
    output_root = args.output_dir.resolve()
    baseline_dir = output_root / "baseline-neutral"
    expressive_dir = output_root / "expressive"
    output_root.mkdir(parents=True, exist_ok=True)

    report_items: list[dict[str, object]] = []
    for item in corpus:
        item_id = str(item["id"])
        text = str(item["text"])
        reference_plan = item["reference_plan"]
        assert isinstance(reference_plan, dict)

        plan = plan_from_dict(
            reference_plan,
            source_text=text,
            allowed_styles=bank.style_names,
        )
        route = expressive.routing_trace(plan)

        baseline_audio, baseline_first_chunk, baseline_elapsed = _timed_stream(
            lambda: model.generate_audio_stream(neutral_state, text)
        )
        expressive_audio, expressive_first_chunk, expressive_elapsed = _timed_stream(
            lambda: expressive.generate_audio_stream(plan)
        )

        baseline_path = baseline_dir / f"{item_id}.wav"
        expressive_path = expressive_dir / f"{item_id}.wav"
        _write_wav(baseline_path, model.sample_rate, baseline_audio)
        _write_wav(expressive_path, model.sample_rate, expressive_audio)

        baseline_audio_seconds = baseline_audio.numel() / model.sample_rate
        expressive_audio_seconds = expressive_audio.numel() / model.sample_rate
        report_items.append(
            {
                "id": item_id,
                "purpose": item.get("purpose"),
                "text": text,
                "route": list(route),
                "baseline_wav": str(baseline_path.relative_to(output_root)),
                "expressive_wav": str(expressive_path.relative_to(output_root)),
                "baseline_first_chunk_seconds": baseline_first_chunk,
                "expressive_first_chunk_seconds": expressive_first_chunk,
                "baseline_generation_seconds": baseline_elapsed,
                "expressive_generation_seconds": expressive_elapsed,
                "baseline_audio_seconds": baseline_audio_seconds,
                "expressive_audio_seconds": expressive_audio_seconds,
                "baseline_speed_x": (
                    baseline_audio_seconds / baseline_elapsed if baseline_elapsed > 0 else None
                ),
                "expressive_speed_x": (
                    expressive_audio_seconds / expressive_elapsed
                    if expressive_elapsed > 0
                    else None
                ),
            }
        )
        print(f"{item_id}: {' -> '.join(route)}")

    report = {
        "schema_version": "1.0",
        "prompt_bank": str(bank.manifest_path),
        "bank_id": bank.bank_id,
        "model_ref": model_ref,
        "sample_rate": model.sample_rate,
        "max_cached_states": args.max_cached_states,
        "items": report_items,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation output: {output_root}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
