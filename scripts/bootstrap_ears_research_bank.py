"""Bootstrap a research-only expressive bank from Kyutai's EARS voice clips.

Kyutai exposes p003 and p031 with multiple emo_*_freeform.wav recordings from
the same speaker specifically for emotion experiments. EARS is CC BY-NC 4.0,
so this helper is strictly for non-commercial research/evaluation.
"""

from __future__ import annotations

import argparse
import json
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from huggingface_hub import hf_hub_download

REPO_ID = "kyutai/tts-voices"
LICENSE_ID = "CC-BY-NC-4.0"
SOURCE_URL = "https://huggingface.co/kyutai/tts-voices/tree/main/ears"
SUPPORTED_SPEAKERS = ("p003", "p031")
DEFAULT_SPEAKER = "p031"


@dataclass(frozen=True)
class EarsStyle:
    """One research routing style backed by a native EARS emotion recording."""

    source_emotion: str
    valence: float
    arousal: float
    dominance: float
    intensity: float


# V/A/D/I values are our routing prototypes; source_emotion names are native
# EARS/Kyutai file labels.
STYLES: dict[str, EarsStyle] = {
    "neutral": EarsStyle("neutral", 0.0, 0.0, 0.0, 0.0),
    "warm": EarsStyle("contentment", 0.45, -0.15, 0.05, 0.45),
    "happy": EarsStyle("amusement", 0.75, 0.35, 0.20, 0.70),
    "excited": EarsStyle("amazement", 0.85, 0.85, 0.35, 0.95),
    "calm": EarsStyle("serenity", 0.25, -0.65, 0.10, 0.50),
    "sad": EarsStyle("sadness", -0.75, -0.50, -0.35, 0.75),
    "angry": EarsStyle("anger", -0.80, 0.80, 0.75, 0.95),
    "fearful": EarsStyle("fear", -0.75, 0.75, -0.70, 0.90),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-commercial same-speaker emotional prompt bank "
            "from Kyutai's EARS clips."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/ears-research-bank"),
        help="Local bank directory (default: runs/ears-research-bank)",
    )
    parser.add_argument(
        "--speaker",
        choices=SUPPORTED_SPEAKERS,
        default=DEFAULT_SPEAKER,
        help=f"EARS emotion speaker (default: {DEFAULT_SPEAKER})",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face repository revision (default: main)",
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use Kyutai's enhanced/cleaned EARS WAVs instead of the original WAVs",
    )
    return parser


def source_path(speaker: str, source_emotion: str, *, enhanced: bool = False) -> str:
    """Return the EARS source path for one speaker/emotion recording."""

    if speaker not in SUPPORTED_SPEAKERS:
        raise ValueError(
            f"unsupported EARS emotion speaker {speaker!r}; "
            f"choose one of {', '.join(SUPPORTED_SPEAKERS)}"
        )
    suffix = "_enhanced" if enhanced else ""
    return f"ears/{speaker}/emo_{source_emotion}_freeform{suffix}.wav"


def build_manifest(speaker: str) -> dict[str, object]:
    """Build our standard prompt-bank manifest for an EARS emotion speaker."""

    if speaker not in SUPPORTED_SPEAKERS:
        raise ValueError(
            f"unsupported EARS emotion speaker {speaker!r}; "
            f"choose one of {', '.join(SUPPORTED_SPEAKERS)}"
        )

    return {
        "schema_version": "1.0",
        "bank_id": f"ears-{speaker}-research-v1",
        "speaker_id": f"ears-{speaker}",
        "language": "english",
        "rights_confirmed": True,
        "recording_protocol_version": "1.0",
        "styles": [
            {
                "name": name,
                "audio_path": f"recordings/{name}.wav",
                "state_path": None,
                "audio_sha256": None,
                "affect": {
                    "valence": style.valence,
                    "arousal": style.arousal,
                    "dominance": style.dominance,
                    "intensity": style.intensity,
                },
                "prompt_text": None,
            }
            for name, style in STYLES.items()
        ],
    }


def _research_notice(
    *,
    revision: str,
    speaker: str,
    enhanced: bool,
    selected_sources: list[str],
) -> str:
    sources = "\n".join(f"- {path}" for path in selected_sources)
    variant = "enhanced/cleaned" if enhanced else "original"
    return f"""# Research-only EARS prompt bank

This bank is a bootstrap/debug asset, not the product/default voice bank.

Source: {SOURCE_URL}
Repository: {REPO_ID}
Revision: {revision}
Speaker: {speaker}
Audio variant: {variant}
Local conversion: PCM16 WAV for Pocket TTS compatibility
License: {LICENSE_ID}
Usage scope: non-commercial research/evaluation only

Kyutai exposes speakers p003 and p031 with multiple emo_*_freeform.wav files
specifically to experiment with one speaker across multiple emotions.

Selected source files:
{sources}

The source emotion names come from EARS/Kyutai. The V/A/D/I coordinates in
prompt-bank.json are our experimental routing prototypes and are not official
dataset annotations or universal psychological ground truth.

Before any commercial/product use, replace this bank with an appropriately
authorized/licensed voice bank.
"""


def _write_pcm16_wav(source: Path, destination: Path) -> None:
    """Convert a WAV into PCM16 so Pocket TTS's built-in WAV reader can open it."""

    sample_rate, raw = scipy.io.wavfile.read(source)
    audio = np.asarray(raw)

    if audio.size == 0:
        raise ValueError(f"EARS source contains no audio samples: {source}")

    if np.issubdtype(audio.dtype, np.floating):
        finite = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        pcm16 = np.rint(np.clip(finite, -1.0, 1.0) * 32767.0).astype(np.int16)
    elif audio.dtype == np.int16:
        pcm16 = audio
    elif np.issubdtype(audio.dtype, np.signedinteger):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        pcm16 = np.rint(
            np.clip(audio.astype(np.float64) / scale, -1.0, 1.0) * 32767.0
        ).astype(np.int16)
    elif np.issubdtype(audio.dtype, np.unsignedinteger):
        info = np.iinfo(audio.dtype)
        midpoint = (info.max + 1) / 2.0
        normalized = (audio.astype(np.float64) - midpoint) / midpoint
        pcm16 = np.rint(np.clip(normalized, -1.0, 1.0) * 32767.0).astype(np.int16)
    else:
        raise ValueError(f"unsupported EARS WAV dtype {audio.dtype} for {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(destination, int(sample_rate), pcm16)

    # Validate the exact reader path Pocket TTS uses before continuing.
    with wave.open(str(destination), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise RuntimeError(f"PCM16 conversion failed for {destination}")


def main() -> None:
    args = _parser().parse_args()
    output_root = args.output_dir.expanduser().resolve()
    recordings = output_root / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)

    selected_sources: list[str] = []
    for name, style in STYLES.items():
        source = source_path(
            args.speaker,
            style.source_emotion,
            enhanced=args.enhanced,
        )
        cached = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=source,
                revision=args.revision,
            )
        )
        destination = recordings / f"{name}.wav"
        _write_pcm16_wav(cached, destination)
        selected_sources.append(source)

    manifest = build_manifest(args.speaker)
    manifest_path = output_root / "prompt-bank.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    notice_path = output_root / "RESEARCH_ONLY.md"
    notice_path.write_text(
        _research_notice(
            revision=args.revision,
            speaker=args.speaker,
            enhanced=args.enhanced,
            selected_sources=selected_sources,
        ),
        encoding="utf-8",
    )

    print(f"Research bank: {manifest_path}")
    print(f"Speaker: {args.speaker}")
    print(f"Styles: {', '.join(STYLES)}")
    print(f"License notice: {notice_path}")
    print("This bank is non-commercial research/evaluation only.")


if __name__ == "__main__":
    main()
