"""Compile an expressive prompt-bank manifest into fast-loading safetensors states."""

from __future__ import annotations

import argparse
from pathlib import Path

from pocket_tts import TTSModel
from pocket_tts.expressive_prompt_bank import compile_prompt_bank


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export every expressive reference recording in a prompt-bank manifest "
            "to Pocket TTS .safetensors states."
        )
    )
    parser.add_argument("manifest", type=Path, help="Source prompt-bank JSON manifest")
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Compiled manifest path (default: <manifest>.compiled.json)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Directory for exported states (default: ./states beside the manifest)",
    )
    model = parser.add_mutually_exclusive_group()
    model.add_argument(
        "--language",
        default=None,
        help="Pocket TTS built-in language/config key (default: english)",
    )
    model.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Custom Pocket TTS YAML config",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing exported state files",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    language = args.language
    config = args.config

    if config is not None:
        model_ref = f"config:{config.resolve()}"
        tts_model = TTSModel.load_model(config=config)
    else:
        effective_language = language or "english"
        model_ref = f"language:{effective_language}"
        tts_model = TTSModel.load_model(language=effective_language)

    compiled = compile_prompt_bank(
        tts_model,
        args.manifest,
        model_ref=model_ref,
        output_manifest=args.output_manifest,
        state_dir=args.state_dir,
        overwrite=args.overwrite,
    )

    print(f"Compiled prompt bank: {compiled.manifest_path}")
    print(f"Model reference: {compiled.compiled_for.model_ref if compiled.compiled_for else model_ref}")
    print(f"Styles: {', '.join(compiled.style_names)}")


if __name__ == "__main__":
    main()
