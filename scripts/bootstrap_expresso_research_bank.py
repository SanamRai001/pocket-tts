"""Bootstrap a research-only expressive bank from Kyutai's Expresso voice clips.

Expresso is licensed CC BY-NC 4.0. This helper is intentionally research-only:
it creates a local prompt bank for non-commercial evaluation and debugging, not
the default/product voice-bank path.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "kyutai/tts-voices"
LICENSE_ID = "CC-BY-NC-4.0"
SOURCE_URL = "https://huggingface.co/kyutai/tts-voices/tree/main/expresso"

_FILE_RE = re.compile(
    r"^expresso/(?P<speaker1>ex\d+)-(?P<speaker2>ex\d+)_"
    r"(?P<kind>[a-z0-9_]+?)_\d+_channel(?P<channel>[12])_\d+s\.wav$"
)


@dataclass(frozen=True)
class Prototype:
    """Routing coordinates for one Expresso performance style."""

    output_name: str
    valence: float
    arousal: float
    dominance: float
    intensity: float


# Routing prototypes for a bootstrap experiment, not psychological ground truth.
_PROTOTYPES = {
    "default": Prototype("neutral", 0.0, 0.0, 0.0, 0.0),
    "happy": Prototype("happy", 0.80, 0.45, 0.20, 0.78),
    "sad": Prototype("sad", -0.78, -0.48, -0.35, 0.75),
    "angry": Prototype("angry", -0.82, 0.82, 0.78, 0.92),
    "calm": Prototype("calm", 0.22, -0.68, 0.12, 0.50),
    "awe": Prototype("awe", 0.62, 0.55, -0.05, 0.72),
    "desire": Prototype("desire", 0.52, 0.28, -0.10, 0.62),
    "disgusted": Prototype("disgusted", -0.78, 0.42, 0.32, 0.78),
    "laughing": Prototype("laughing", 0.85, 0.75, 0.10, 0.85),
    "fast": Prototype("fast", 0.20, 0.75, 0.10, 0.65),
    "whisper": Prototype("whisper", 0.00, -0.70, -0.40, 0.45),
    "confused": Prototype("confused", -0.35, 0.35, -0.55, 0.55),
    "projected": Prototype("projected", 0.00, 0.65, 0.75, 0.70),
    "enunciated": Prototype("enunciated", 0.10, 0.05, 0.25, 0.30),
    "narration": Prototype("narration", 0.20, -0.20, 0.15, 0.35),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-commercial research prompt bank from Kyutai's "
            "Expresso clips."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/expresso-research-bank"),
        help="Local bank directory (default: runs/expresso-research-bank)",
    )
    parser.add_argument(
        "--speaker",
        default=None,
        help=(
            "Optional Expresso speaker id such as ex04. If omitted, choose "
            "the speaker with the broadest supported style coverage."
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face repository revision (default: main)",
    )
    return parser


def _speaker_for_file(match: re.Match[str]) -> str:
    """Return the speaker that owns the selected stereo channel."""

    return (
        match.group("speaker1")
        if match.group("channel") == "1"
        else match.group("speaker2")
    )


def _discover(revision: str) -> dict[str, dict[str, str]]:
    """Discover one deterministic clip per supported style and speaker."""

    api = HfApi()
    files = api.list_repo_files(REPO_ID, revision=revision)

    candidates: dict[str, dict[str, str]] = defaultdict(dict)
    for filename in files:
        match = _FILE_RE.match(filename)
        if match is None:
            continue

        kind = match.group("kind")
        if kind not in _PROTOTYPES:
            continue

        speaker = _speaker_for_file(match)
        existing = candidates[speaker].get(kind)
        if existing is None or filename < existing:
            candidates[speaker][kind] = filename

    return dict(candidates)


def _choose_speaker(
    candidates: dict[str, dict[str, str]],
    requested: str | None,
) -> tuple[str, dict[str, str]]:
    """Choose a speaker with a neutral/default anchor and broad style coverage."""

    if requested is not None:
        if requested not in candidates:
            available = ", ".join(sorted(candidates))
            raise ValueError(
                f"speaker {requested!r} has no supported Expresso clips; "
                f"available speakers: {available}"
            )
        styles = candidates[requested]
        if "default" not in styles:
            raise ValueError(
                f"speaker {requested!r} has no default clip, so it cannot anchor neutral"
            )
        if len(styles) < 2:
            raise ValueError(
                f"speaker {requested!r} has no expressive style beyond the neutral anchor"
            )
        return requested, styles

    ranked = sorted(
        (
            (len(styles), speaker, styles)
            for speaker, styles in candidates.items()
            if "default" in styles and len(styles) >= 2
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked:
        raise RuntimeError("no Expresso speaker with a default clip was discovered")

    _, speaker, styles = ranked[0]
    return speaker, styles


def _research_notice(
    *,
    revision: str,
    speaker: str,
    selected_sources: list[str],
) -> str:
    sources = "\n".join(f"- {path}" for path in selected_sources)
    return f"""# Research-only Expresso prompt bank

This bank is a bootstrap/debug asset, not the product/default voice bank.

Source: {SOURCE_URL}
Repository: {REPO_ID}
Revision: {revision}
Speaker: {speaker}
License: {LICENSE_ID}
Usage scope: non-commercial research/evaluation only

Selected source files:
{sources}

The V/A/D/I coordinates in prompt-bank.json are experimental routing
prototypes. They are not dataset labels or claims about psychological ground
truth.

Before any commercial/product use, replace this bank with an appropriately
authorized/licensed voice bank.
"""


def main() -> None:
    args = _parser().parse_args()
    output_root = args.output_dir.expanduser().resolve()
    recordings = output_root / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)

    candidates = _discover(args.revision)
    speaker, source_styles = _choose_speaker(candidates, args.speaker)

    manifest_styles: list[dict[str, object]] = []
    selected_sources: list[str] = []

    for kind in sorted(source_styles, key=lambda name: (name != "default", name)):
        prototype = _PROTOTYPES[kind]
        source = source_styles[kind]
        cached = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=source,
                revision=args.revision,
            )
        )
        destination = recordings / f"{prototype.output_name}.wav"
        shutil.copy2(cached, destination)
        selected_sources.append(source)

        manifest_styles.append(
            {
                "name": prototype.output_name,
                "audio_path": f"recordings/{destination.name}",
                "state_path": None,
                "audio_sha256": None,
                "affect": {
                    "valence": prototype.valence,
                    "arousal": prototype.arousal,
                    "dominance": prototype.dominance,
                    "intensity": prototype.intensity,
                },
                "prompt_text": None,
            }
        )

    manifest = {
        "schema_version": "1.0",
        "bank_id": f"expresso-{speaker}-research-v1",
        "speaker_id": f"expresso-{speaker}",
        "language": "english",
        "rights_confirmed": True,
        "recording_protocol_version": "1.0",
        "styles": manifest_styles,
    }
    manifest_path = output_root / "prompt-bank.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    notice_path = output_root / "RESEARCH_ONLY.md"
    notice_path.write_text(
        _research_notice(
            revision=args.revision,
            speaker=speaker,
            selected_sources=selected_sources,
        ),
        encoding="utf-8",
    )

    print(f"Research bank: {manifest_path}")
    print(f"Speaker: {speaker}")
    print(f"Styles: {', '.join(str(style['name']) for style in manifest_styles)}")
    print(f"License notice: {notice_path}")
    print("This bank is non-commercial research/evaluation only.")


if __name__ == "__main__":
    main()
