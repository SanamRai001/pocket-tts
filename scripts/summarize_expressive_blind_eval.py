"""Unblind and summarize expressive Pocket TTS human A/B ratings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


_SCORE_FIELDS = (
    "emotional_fit",
    "naturalness",
    "speaker_consistency",
    "transition_smoothness",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unblind ratings.csv and summarize expressive-vs-neutral preferences."
    )
    parser.add_argument(
        "blind_dir",
        type=Path,
        help="Directory produced by scripts/prepare_expressive_blind_eval.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary JSON path (default: <blind_dir>/summary.json)",
    )
    return parser


def _score(value: str, field: str, item_id: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        number = float(stripped)
    except ValueError as exc:
        raise ValueError(f"{item_id}: {field} must be numeric or blank") from exc
    if number < 1 or number > 5:
        raise ValueError(f"{item_id}: {field} must be in [1, 5]")
    return number


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    args = _parser().parse_args()
    blind_dir = args.blind_dir.expanduser().resolve()
    answers_payload = json.loads((blind_dir / "answer-key.json").read_text(encoding="utf-8"))
    answers = answers_payload.get("answers")
    if not isinstance(answers, list):
        raise ValueError("answer-key.json has no answers array")

    answer_by_id: dict[str, dict[str, str]] = {}
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("answer-key answers must be objects")
        item_id = str(answer["id"])
        answer_by_id[item_id] = {
            "A": str(answer["A"]),
            "B": str(answer["B"]),
        }

    rows: list[dict[str, str]] = []
    with (blind_dir / "ratings.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: value or "" for key, value in row.items()})

    condition_scores: dict[str, dict[str, list[float]]] = {
        "expressive": {field: [] for field in _SCORE_FIELDS},
        "baseline-neutral": {field: [] for field in _SCORE_FIELDS},
    }
    preference_counts = {
        "expressive": 0,
        "baseline-neutral": 0,
        "tie": 0,
        "unrated": 0,
    }
    per_item: list[dict[str, object]] = []

    for row in rows:
        item_id = row.get("id", "")
        if item_id not in answer_by_id:
            raise ValueError(f"ratings.csv contains unknown item {item_id!r}")
        answer = answer_by_id[item_id]

        preferred = row.get("preferred", "").strip().upper()
        if not preferred:
            preference_counts["unrated"] += 1
            preferred_condition = None
        elif preferred in {"TIE", "T"}:
            preference_counts["tie"] += 1
            preferred_condition = "tie"
        elif preferred in {"A", "B"}:
            preferred_condition = answer[preferred]
            preference_counts[preferred_condition] += 1
        else:
            raise ValueError(
                f"{item_id}: preferred must be A, B, Tie, or blank"
            )

        item_scores: dict[str, dict[str, float | None]] = {
            "expressive": {},
            "baseline-neutral": {},
        }
        for label in ("A", "B"):
            condition = answer[label]
            suffix = label.lower()
            for field in _SCORE_FIELDS:
                column = f"{field}_{suffix}_1_5"
                value = _score(row.get(column, ""), column, item_id)
                item_scores[condition][field] = value
                if value is not None:
                    condition_scores[condition][field].append(value)

        per_item.append(
            {
                "id": item_id,
                "preferred_condition": preferred_condition,
                "scores": item_scores,
                "notes": row.get("notes", ""),
            }
        )

    means = {
        condition: {
            field: _mean(values)
            for field, values in fields.items()
        }
        for condition, fields in condition_scores.items()
    }
    score_deltas = {
        field: (
            means["expressive"][field] - means["baseline-neutral"][field]
            if means["expressive"][field] is not None
            and means["baseline-neutral"][field] is not None
            else None
        )
        for field in _SCORE_FIELDS
    }

    rated_preferences = (
        preference_counts["expressive"]
        + preference_counts["baseline-neutral"]
        + preference_counts["tie"]
    )
    expressive_win_rate = (
        preference_counts["expressive"] / rated_preferences
        if rated_preferences
        else None
    )

    summary = {
        "schema_version": "1.0",
        "num_trials": len(rows),
        "preference_counts": preference_counts,
        "expressive_win_rate_including_ties_in_denominator": expressive_win_rate,
        "mean_scores": means,
        "expressive_minus_baseline_score_deltas": score_deltas,
        "items": per_item,
    }

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else blind_dir / "summary.json"
    )
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, indent=2))
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
