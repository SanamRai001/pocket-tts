"""Tests for expressive prompt-bank manifests and compilation."""

import json
from pathlib import Path

import pytest
import torch

from pocket_tts.expressive_prompt_bank import (
    PromptBankValidationError,
    compile_prompt_bank,
    load_prompt_bank,
    style_profiles_from_bank,
    validate_bank_files,
    validate_compiled_bank_model,
)
from pocket_tts.modules.stateful_module import ModelState


class FakePromptBankModel:
    origin: Path | None = None

    def __init__(self) -> None:
        self.prompts: list[Path] = []

    def get_state_for_audio_prompt(
        self, audio_conditioning: str | Path | torch.Tensor, truncate: bool = False
    ) -> ModelState:
        assert truncate is True
        prompt = Path(audio_conditioning)
        self.prompts.append(prompt)
        return {"fake": {"cache": torch.tensor([float(len(self.prompts))])}}


def _manifest(rights_confirmed: bool = True) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "bank_id": "speaker-a-en-v1",
        "speaker_id": "speaker-a",
        "language": "english",
        "rights_confirmed": rights_confirmed,
        "recording_protocol_version": "1.0",
        "styles": [
            {
                "name": "neutral",
                "audio_path": "recordings/neutral.wav",
                "affect": {
                    "valence": 0.0,
                    "arousal": 0.0,
                    "dominance": 0.0,
                    "intensity": 0.0,
                },
                "prompt_text": "This is the same text.",
            },
            {
                "name": "happy",
                "audio_path": "recordings/happy.wav",
                "affect": {
                    "valence": 0.8,
                    "arousal": 0.5,
                    "dominance": 0.2,
                    "intensity": 0.8,
                },
                "prompt_text": "This is the same text.",
            },
        ],
    }


def _write_bank(tmp_path: Path, *, rights_confirmed: bool = True) -> Path:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "neutral.wav").write_bytes(b"neutral-audio")
    (recordings / "happy.wav").write_bytes(b"happy-audio")
    manifest = tmp_path / "prompt-bank.json"
    manifest.write_text(
        json.dumps(_manifest(rights_confirmed=rights_confirmed)),
        encoding="utf-8",
    )
    return manifest


def test_load_prompt_bank_resolves_portable_manifest() -> None:
    manifest = _write_bank(pytest.ensuretemp("prompt-bank-load"))
    bank = load_prompt_bank(manifest)

    assert bank.bank_id == "speaker-a-en-v1"
    assert bank.style_names == ("neutral", "happy")
    assert bank.entry("happy").audio_path == Path("recordings/happy.wav")


def test_prompt_bank_requires_neutral_style(tmp_path: Path) -> None:
    payload = _manifest()
    styles = payload["styles"]
    assert isinstance(styles, list)
    styles.pop(0)
    manifest = tmp_path / "prompt-bank.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromptBankValidationError, match="neutral"):
        load_prompt_bank(manifest)


def test_prompt_bank_rejects_paths_that_escape_bank_directory(tmp_path: Path) -> None:
    payload = _manifest()
    styles = payload["styles"]
    assert isinstance(styles, list)
    first = styles[0]
    assert isinstance(first, dict)
    first["audio_path"] = "../neutral.wav"
    manifest = tmp_path / "prompt-bank.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromptBankValidationError, match="cannot escape"):
        load_prompt_bank(manifest)


def test_compile_prompt_bank_exports_states_and_hashes(tmp_path: Path) -> None:
    manifest = _write_bank(tmp_path)
    model = FakePromptBankModel()

    compiled = compile_prompt_bank(
        model,
        manifest,
        model_ref="language:english",
    )

    assert compiled.manifest_path.name == "prompt-bank.compiled.json"
    assert compiled.compiled_for is not None
    assert compiled.compiled_for.model_ref == "language:english"
    assert all(entry.audio_sha256 is not None for entry in compiled.entries)
    assert all(entry.state_path is not None for entry in compiled.entries)
    assert validate_bank_files(compiled, require_states=True) == ()

    profiles = style_profiles_from_bank(
        compiled,
        expected_model_ref="language:english",
        model=model,
    )
    assert all(Path(profile.source).suffix == ".safetensors" for profile in profiles)


def test_compile_requires_voice_rights_confirmation(tmp_path: Path) -> None:
    manifest = _write_bank(tmp_path, rights_confirmed=False)

    with pytest.raises(PromptBankValidationError, match="rights_confirmed"):
        compile_prompt_bank(
            FakePromptBankModel(),
            manifest,
            model_ref="language:english",
        )


def test_compiled_bank_rejects_different_model_reference(tmp_path: Path) -> None:
    compiled = compile_prompt_bank(
        FakePromptBankModel(),
        _write_bank(tmp_path),
        model_ref="language:english",
    )

    with pytest.raises(PromptBankValidationError, match="not 'language:french'"):
        validate_compiled_bank_model(
            compiled,
            model_ref="language:french",
        )
