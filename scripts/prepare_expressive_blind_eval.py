"""Prepare a deterministic blinded A/B listening set for expressive Pocket TTS."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Randomize baseline-vs-expressive WAV order and create a human-rating worksheet."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory produced by scripts/generate_expressive_eval.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Blind set directory (default: <run_dir>/blind)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261006,
        help="Deterministic A/B randomization seed",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    items = report.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{report_path} contains no evaluation items")

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "blind"
    )
    audio_dir = output_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    trials: list[dict[str, object]] = []
    answer_key: list[dict[str, str]] = []

    for raw_item in items:
        if not isinstance(raw_item, dict):
            raise ValueError("report items must be objects")
        item_id = str(raw_item["id"])
        baseline = run_dir / str(raw_item["baseline_wav"])
        expressive = run_dir / str(raw_item["expressive_wav"])
        if not baseline.is_file() or not expressive.is_file():
            raise FileNotFoundError(
                f"missing A/B audio for {item_id}: {baseline} / {expressive}"
            )

        expressive_is_a = bool(rng.getrandbits(1))
        sources = {
            "A": expressive if expressive_is_a else baseline,
            "B": baseline if expressive_is_a else expressive,
        }
        conditions = {
            "A": "expressive" if expressive_is_a else "baseline-neutral",
            "B": "baseline-neutral" if expressive_is_a else "expressive",
        }

        trial_audio: dict[str, str] = {}
        for label, source in sources.items():
            destination = audio_dir / f"{item_id}_{label}.wav"
            shutil.copy2(source, destination)
            trial_audio[label] = str(destination.relative_to(output_root))

        trials.append(
            {
                "id": item_id,
                "purpose": raw_item.get("purpose"),
                "text": raw_item.get("text"),
                "audio_a": trial_audio["A"],
                "audio_b": trial_audio["B"],
            }
        )
        answer_key.append(
            {
                "id": item_id,
                "A": conditions["A"],
                "B": conditions["B"],
            }
        )

    trials_path = output_root / "trials.json"
    trials_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seed": args.seed,
                "instructions": (
                    "Rate every trial before opening answer-key.json. "
                    "A and B are randomized per item."
                ),
                "trials": trials,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    answer_key_path = output_root / "answer-key.json"
    answer_key_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seed": args.seed,
                "answers": answer_key,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    ratings_path = output_root / "ratings.csv"
    with ratings_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "preferred",
                "emotional_fit_a_1_5",
                "emotional_fit_b_1_5",
                "naturalness_a_1_5",
                "naturalness_b_1_5",
                "speaker_consistency_a_1_5",
                "speaker_consistency_b_1_5",
                "transition_smoothness_a_1_5",
                "transition_smoothness_b_1_5",
                "notes",
            ]
        )
        for trial in trials:
            writer.writerow([trial["id"], "", "", "", "", "", "", "", "", "", ""])

    print(f"Blind trials: {trials_path}")
    print(f"Ratings sheet: {ratings_path}")
    print("Do not open answer-key.json until ratings are complete.")


if __name__ == "__main__":
    main()
