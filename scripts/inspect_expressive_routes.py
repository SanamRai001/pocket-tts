"""Preview expressive routing without loading Pocket TTS model weights.

This is a cheap Phase 3C sanity check. It evaluates the prompt-bank geometry
against the fixed emotional corpus before spending time compiling states or
synthesizing audio.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pocket_tts.expressive import AffectivePromptRouter
from pocket_tts.expressive_planner import plan_from_dict
from pocket_tts.expressive_prompt_bank import load_prompt_bank, style_profiles_from_bank

_DEFAULT_CORPUS = Path("docs/expressive-speech/eval-corpus-v1.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview affect-to-style routing for an expressive prompt bank."
    )
    parser.add_argument("prompt_bank", type=Path, help="Prompt-bank manifest")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_DEFAULT_CORPUS,
        help=f"Evaluation corpus JSON (default: {_DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    return parser


def _load_items(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("evaluation corpus must be a version 1.0 object")

    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("evaluation corpus must contain a non-empty items array")

    result: list[dict[str, object]] = []
    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise ValueError(f"items[{index}] must be an object")
        if not isinstance(raw.get("text"), str):
            raise ValueError(f"items[{index}].text must be a string")
        if not isinstance(raw.get("reference_plan"), dict):
            raise ValueError(f"items[{index}].reference_plan must be an object")
        result.append(raw)
    return result


def main() -> None:
    args = _parser().parse_args()
    bank = load_prompt_bank(args.prompt_bank)
    profiles = style_profiles_from_bank(bank, prefer_exported=False)
    router = AffectivePromptRouter(profiles, default_style="neutral")
    items = _load_items(args.corpus)

    style_counts: Counter[str] = Counter()
    warnings: list[str] = []
    routed_items: list[dict[str, object]] = []

    for raw in items:
        item_id = str(raw.get("id", "unknown"))
        text = str(raw["text"])
        reference_plan = raw["reference_plan"]
        assert isinstance(reference_plan, dict)

        plan = plan_from_dict(
            reference_plan,
            source_text=text,
            allowed_styles=bank.style_names,
        )
        routes = tuple(profile.name for profile in router.route_plan(plan))
        style_counts.update(routes)

        segment_rows: list[dict[str, object]] = []
        for index, (segment, route) in enumerate(zip(plan.segments, routes, strict=True)):
            if segment.affect.intensity >= 0.45 and route == "neutral":
                warnings.append(
                    f"{item_id}[{index}] has affect intensity "
                    f"{segment.affect.intensity:.2f} but routes to neutral"
                )
            segment_rows.append(
                {
                    "index": index,
                    "text": segment.text,
                    "affect": {
                        "valence": segment.affect.valence,
                        "arousal": segment.affect.arousal,
                        "dominance": segment.affect.dominance,
                        "intensity": segment.affect.intensity,
                    },
                    "route": route,
                }
            )

        routed_items.append(
            {
                "id": item_id,
                "purpose": raw.get("purpose"),
                "route": list(routes),
                "segments": segment_rows,
            }
        )
        print(f"{item_id}: {' -> '.join(routes)}")

    unused_styles = sorted(set(bank.style_names) - set(style_counts))
    if unused_styles:
        warnings.append(
            "styles unused by fixed corpus: " + ", ".join(unused_styles)
        )

    report = {
        "schema_version": "1.0",
        "bank_id": bank.bank_id,
        "speaker_id": bank.speaker_id,
        "available_styles": list(bank.style_names),
        "style_usage": dict(sorted(style_counts.items())),
        "unused_styles": unused_styles,
        "warnings": warnings,
        "items": routed_items,
    }

    print("\nStyle usage:")
    for style, count in sorted(style_counts.items()):
        print(f"  {style}: {count}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nRoute preview: {output}")


if __name__ == "__main__":
    main()
