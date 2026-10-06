"""Run the complete research-only expressive Pocket TTS experiment.

This orchestrates the existing Phase 3C tools:
1. bootstrap a non-commercial Expresso prompt bank
2. compile WAV prompts to fast-loading safetensors states
3. generate the fixed neutral-vs-expressive A/B corpus
4. run dependency-light technical scoring
5. prepare a blinded listening set

The script intentionally stops before heavy optional metrics such as ASR WER,
WavLM speaker similarity, or UTMOS. Those remain separate opt-in evaluation
steps because they can download large models and are not required for the
first listening experiment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full research-only Expresso expressive-TTS experiment."
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("runs/expressive-research-v1"),
        help="Root directory for bank, generated audio, scores, and blind set.",
    )
    parser.add_argument(
        "--language",
        default="english",
        help="Pocket TTS language/config key (default: english).",
    )
    parser.add_argument(
        "--speaker",
        default=None,
        help="Optional Expresso speaker id such as ex04.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Kyutai tts-voices revision used by the research bootstrap.",
    )
    parser.add_argument(
        "--max-cached-states",
        type=int,
        default=2,
        help="Maximum expressive prompt states held in memory.",
    )
    parser.add_argument(
        "--skip-blind",
        action="store_true",
        help="Skip preparation of the blinded A/B listening set.",
    )
    return parser


def _run(command: list[str]) -> None:
    printable = subprocess.list2cmdline(command)
    print(f"\n>>> {printable}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = _parser().parse_args()
    if args.max_cached_states < 0:
        raise ValueError("--max-cached-states cannot be negative")

    repo_root = Path(__file__).resolve().parent.parent
    work_root = args.work_dir.expanduser().resolve()
    bank_dir = work_root / "prompt-bank"
    eval_dir = work_root / "evaluation"
    bank_manifest = bank_dir / "prompt-bank.json"
    compiled_manifest = bank_dir / "prompt-bank.compiled.json"

    python = sys.executable
    scripts = repo_root / "scripts"

    bootstrap = [
        python,
        str(scripts / "bootstrap_expresso_research_bank.py"),
        "--output-dir",
        str(bank_dir),
        "--revision",
        args.revision,
    ]
    if args.speaker:
        bootstrap.extend(["--speaker", args.speaker])
    _run(bootstrap)

    _run(
        [
            python,
            str(scripts / "export_expressive_prompt_bank.py"),
            str(bank_manifest),
            "--language",
            args.language,
        ]
    )

    _run(
        [
            python,
            str(scripts / "generate_expressive_eval.py"),
            str(compiled_manifest),
            "--language",
            args.language,
            "--output-dir",
            str(eval_dir),
            "--max-cached-states",
            str(args.max_cached_states),
        ]
    )

    _run(
        [
            python,
            str(scripts / "score_expressive_eval.py"),
            str(eval_dir),
        ]
    )

    if not args.skip_blind:
        _run(
            [
                python,
                str(scripts / "prepare_expressive_blind_eval.py"),
                str(eval_dir),
            ]
        )

    scores_path = eval_dir / "scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    aggregate = scores.get("aggregate", {})

    print("\n=== Expressive research experiment complete ===")
    print(f"Work directory: {work_root}")
    print(f"Compiled prompt bank: {compiled_manifest}")
    print(f"A/B audio: {eval_dir}")
    print(f"Scores: {scores_path}")
    if not args.skip_blind:
        print(f"Blind listening set: {eval_dir / 'blind'}")

    if isinstance(aggregate, dict):
        print("\nTier 0 aggregate:")
        print(json.dumps(aggregate, indent=2))

    print(
        "\nReminder: the Expresso bootstrap is CC BY-NC 4.0 and is "
        "non-commercial research/evaluation only."
    )
    print(
        "Do not make product-quality conclusions until an appropriately "
        "authorized/licensed same-speaker bank has also been evaluated."
    )


if __name__ == "__main__":
    main()
