"""Portable same-speaker prompt banks for expressive Pocket TTS."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pocket_tts import export_model_state
from pocket_tts.expressive import AffectVector, StyleProfile, VoiceSource
from pocket_tts.modules.stateful_module import ModelState

PROMPT_BANK_SCHEMA_VERSION = "1.0"
RECORDING_PROTOCOL_VERSION = "1.0"
_STYLE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class PromptBankValidationError(ValueError):
    """Raised when a prompt-bank manifest is invalid or unsafe to compile."""


class PromptBankModel(Protocol):
    """Minimal Pocket TTS surface required to compile a prompt bank."""

    origin: Path | None

    def get_state_for_audio_prompt(
        self, audio_conditioning: VoiceSource, truncate: bool = False
    ) -> ModelState: ...


@dataclass(frozen=True)
class PromptBankEntry:
    """One expressive reference performance from a single bank speaker."""

    name: str
    audio_path: Path
    affect: AffectVector
    prompt_text: str | None = None
    state_path: Path | None = None
    audio_sha256: str | None = None


@dataclass(frozen=True)
class CompiledFor:
    """Model identity recorded when prompt states are exported."""

    model_ref: str
    config_sha256: str | None = None


@dataclass(frozen=True)
class PromptBank:
    """A versioned same-speaker expressive reference bank."""

    manifest_path: Path
    bank_id: str
    speaker_id: str
    language: str
    rights_confirmed: bool
    recording_protocol_version: str
    entries: tuple[PromptBankEntry, ...]
    compiled_for: CompiledFor | None = None

    @property
    def root(self) -> Path:
        return self.manifest_path.parent

    @property
    def style_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    def entry(self, name: str) -> PromptBankEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(name)


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PromptBankValidationError(f"{path} must be an object")

    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise PromptBankValidationError(f"{path} contains a non-string key")
        result[key] = item
    return result


def _reject_unknown(data: Mapping[str, object], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise PromptBankValidationError(
            f"{path} contains unsupported field(s): {', '.join(unknown)}"
        )


def _required_string(data: Mapping[str, object], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PromptBankValidationError(f"{path}.{key} must be a non-empty string")
    return value.strip()


def _relative_path(value: object, path: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PromptBankValidationError(f"{path} must be a non-empty relative path")
    parsed = Path(value)
    if parsed.is_absolute():
        raise PromptBankValidationError(f"{path} must be relative for portability")
    if ".." in parsed.parts:
        raise PromptBankValidationError(f"{path} cannot escape the prompt-bank directory")
    return parsed


def _finite_number(value: object, path: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromptBankValidationError(f"{path} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < low or number > high:
        raise PromptBankValidationError(
            f"{path} must be finite and in [{low}, {high}], got {value}"
        )
    return number


def _affect(value: object, path: str) -> AffectVector:
    data = _mapping(value, path)
    _reject_unknown(data, {"valence", "arousal", "dominance", "intensity"}, path)
    missing = [
        key
        for key in ("valence", "arousal", "dominance", "intensity")
        if key not in data
    ]
    if missing:
        raise PromptBankValidationError(
            f"{path} is missing field(s): {', '.join(missing)}"
        )

    return AffectVector(
        valence=_finite_number(data["valence"], f"{path}.valence", -1.0, 1.0),
        arousal=_finite_number(data["arousal"], f"{path}.arousal", -1.0, 1.0),
        dominance=_finite_number(data["dominance"], f"{path}.dominance", -1.0, 1.0),
        intensity=_finite_number(data["intensity"], f"{path}.intensity", 0.0, 1.0),
    )


def load_prompt_bank(path: str | Path) -> PromptBank:
    """Load and strictly validate a prompt-bank manifest."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromptBankValidationError(
            f"{manifest_path} is not valid JSON: {exc.msg}"
        ) from exc

    data = _mapping(decoded, "$")
    _reject_unknown(
        data,
        {
            "schema_version",
            "bank_id",
            "speaker_id",
            "language",
            "rights_confirmed",
            "recording_protocol_version",
            "styles",
            "compiled_for",
        },
        "$",
    )

    if data.get("schema_version") != PROMPT_BANK_SCHEMA_VERSION:
        raise PromptBankValidationError(
            f"$.schema_version must be {PROMPT_BANK_SCHEMA_VERSION!r}"
        )

    rights_confirmed = data.get("rights_confirmed")
    if not isinstance(rights_confirmed, bool):
        raise PromptBankValidationError("$.rights_confirmed must be a boolean")

    protocol = data.get("recording_protocol_version", RECORDING_PROTOCOL_VERSION)
    if protocol != RECORDING_PROTOCOL_VERSION:
        raise PromptBankValidationError(
            f"$.recording_protocol_version must be {RECORDING_PROTOCOL_VERSION!r}"
        )

    raw_styles = data.get("styles")
    if not isinstance(raw_styles, list) or not raw_styles:
        raise PromptBankValidationError("$.styles must be a non-empty array")

    entries: list[PromptBankEntry] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(raw_styles):
        path_label = f"$.styles[{index}]"
        entry = _mapping(raw_entry, path_label)
        _reject_unknown(
            entry,
            {
                "name",
                "audio_path",
                "state_path",
                "audio_sha256",
                "affect",
                "prompt_text",
            },
            path_label,
        )

        name = _required_string(entry, "name", path_label)
        if not _STYLE_NAME_RE.fullmatch(name):
            raise PromptBankValidationError(
                f"{path_label}.name must match {_STYLE_NAME_RE.pattern!r}"
            )
        if name in seen:
            raise PromptBankValidationError(f"duplicate style name {name!r}")
        seen.add(name)

        prompt_text_value = entry.get("prompt_text")
        if prompt_text_value is not None and (
            not isinstance(prompt_text_value, str) or not prompt_text_value.strip()
        ):
            raise PromptBankValidationError(
                f"{path_label}.prompt_text must be null or a non-empty string"
            )

        state_value = entry.get("state_path")
        state_path = (
            _relative_path(state_value, f"{path_label}.state_path")
            if state_value is not None
            else None
        )

        audio_hash = entry.get("audio_sha256")
        if audio_hash is not None and (
            not isinstance(audio_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", audio_hash)
        ):
            raise PromptBankValidationError(
                f"{path_label}.audio_sha256 must be a lowercase SHA-256 hex digest"
            )

        entries.append(
            PromptBankEntry(
                name=name,
                audio_path=_relative_path(entry.get("audio_path"), f"{path_label}.audio_path"),
                affect=_affect(entry.get("affect"), f"{path_label}.affect"),
                prompt_text=(
                    prompt_text_value.strip()
                    if isinstance(prompt_text_value, str)
                    else None
                ),
                state_path=state_path,
                audio_sha256=audio_hash,
            )
        )

    if "neutral" not in seen:
        raise PromptBankValidationError(
            "prompt bank must include a 'neutral' style for conservative fallback"
        )

    compiled_for_value = data.get("compiled_for")
    compiled_for: CompiledFor | None = None
    if compiled_for_value is not None:
        compiled_data = _mapping(compiled_for_value, "$.compiled_for")
        _reject_unknown(compiled_data, {"model_ref", "config_sha256"}, "$.compiled_for")
        model_ref = _required_string(compiled_data, "model_ref", "$.compiled_for")
        config_sha256 = compiled_data.get("config_sha256")
        if config_sha256 is not None and (
            not isinstance(config_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", config_sha256)
        ):
            raise PromptBankValidationError(
                "$.compiled_for.config_sha256 must be null or a lowercase SHA-256 digest"
            )
        compiled_for = CompiledFor(model_ref=model_ref, config_sha256=config_sha256)

    return PromptBank(
        manifest_path=manifest_path,
        bank_id=_required_string(data, "bank_id", "$"),
        speaker_id=_required_string(data, "speaker_id", "$"),
        language=_required_string(data, "language", "$"),
        rights_confirmed=rights_confirmed,
        recording_protocol_version=protocol,
        entries=tuple(entries),
        compiled_for=compiled_for,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(model: PromptBankModel) -> str | None:
    origin = model.origin
    if origin is None:
        return None
    try:
        path = Path(origin)
    except TypeError:
        return None
    if not path.is_file():
        return None
    return _sha256(path)


def _portable_relative(path: Path, root: Path) -> Path:
    relative = Path(os.path.relpath(path, root))
    if ".." in relative.parts:
        raise PromptBankValidationError(
            f"{path} is outside {root}; compiled prompt-bank paths must stay portable"
        )
    return relative


def prompt_bank_to_dict(bank: PromptBank) -> dict[str, object]:
    """Serialize a prompt bank using paths relative to its manifest."""

    payload: dict[str, object] = {
        "schema_version": PROMPT_BANK_SCHEMA_VERSION,
        "bank_id": bank.bank_id,
        "speaker_id": bank.speaker_id,
        "language": bank.language,
        "rights_confirmed": bank.rights_confirmed,
        "recording_protocol_version": bank.recording_protocol_version,
        "styles": [
            {
                "name": entry.name,
                "audio_path": entry.audio_path.as_posix(),
                "state_path": entry.state_path.as_posix() if entry.state_path else None,
                "audio_sha256": entry.audio_sha256,
                "affect": {
                    "valence": entry.affect.valence,
                    "arousal": entry.affect.arousal,
                    "dominance": entry.affect.dominance,
                    "intensity": entry.affect.intensity,
                },
                "prompt_text": entry.prompt_text,
            }
            for entry in bank.entries
        ],
    }
    if bank.compiled_for is not None:
        payload["compiled_for"] = {
            "model_ref": bank.compiled_for.model_ref,
            "config_sha256": bank.compiled_for.config_sha256,
        }
    return payload


def validate_compiled_bank_model(
    bank: PromptBank,
    *,
    model_ref: str,
    model: PromptBankModel | None = None,
) -> None:
    """Reject exported prompt states compiled for different model weights."""

    if bank.compiled_for is None:
        raise PromptBankValidationError(
            "prompt bank has no compiled_for metadata; recompile it for this model"
        )
    if bank.compiled_for.model_ref != model_ref:
        raise PromptBankValidationError(
            f"prompt bank was compiled for {bank.compiled_for.model_ref!r}, "
            f"not {model_ref!r}"
        )

    expected_hash = bank.compiled_for.config_sha256
    if model is not None and expected_hash is not None:
        actual_hash = _config_sha256(model)
        if actual_hash is not None and actual_hash != expected_hash:
            raise PromptBankValidationError(
                "Pocket TTS config changed since this bank was compiled; "
                "re-export the prompt states"
            )


def style_profiles_from_bank(
    bank: PromptBank,
    *,
    prefer_exported: bool = True,
    expected_model_ref: str | None = None,
    model: PromptBankModel | None = None,
) -> tuple[StyleProfile, ...]:
    """Build router profiles, preferring fast exported states when available."""

    if prefer_exported and expected_model_ref is not None:
        validate_compiled_bank_model(bank, model_ref=expected_model_ref, model=model)

    profiles: list[StyleProfile] = []
    for entry in bank.entries:
        audio = bank.root / entry.audio_path
        state = bank.root / entry.state_path if entry.state_path is not None else None
        source: Path
        if prefer_exported and state is not None and state.is_file():
            source = state
        else:
            source = audio
        profiles.append(StyleProfile(entry.name, source, entry.affect))
    return tuple(profiles)


def compile_prompt_bank(
    model: PromptBankModel,
    manifest_path: str | Path,
    *,
    model_ref: str,
    output_manifest: str | Path | None = None,
    state_dir: str | Path | None = None,
    overwrite: bool = False,
) -> PromptBank:
    """Export all prompt WAVs to fast-loading model states and write a compiled manifest."""

    if not model_ref.strip():
        raise PromptBankValidationError("model_ref cannot be empty")

    bank = load_prompt_bank(manifest_path)
    if not bank.rights_confirmed:
        raise PromptBankValidationError(
            "rights_confirmed must be true before voice prompt states can be exported"
        )

    source_manifest = bank.manifest_path
    output = (
        Path(output_manifest).expanduser().resolve()
        if output_manifest is not None
        else source_manifest.with_name(f"{source_manifest.stem}.compiled.json")
    )
    states_root = (
        Path(state_dir).expanduser().resolve()
        if state_dir is not None
        else source_manifest.parent / "states"
    )
    states_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    compiled_entries: list[PromptBankEntry] = []
    for entry in bank.entries:
        audio = bank.root / entry.audio_path
        if not audio.is_file():
            raise PromptBankValidationError(
                f"audio prompt for style {entry.name!r} does not exist: {audio}"
            )

        destination = states_root / f"{entry.name}.safetensors"
        if destination.exists() and not overwrite:
            raise PromptBankValidationError(
                f"state already exists for style {entry.name!r}: {destination}; "
                "pass overwrite=True to replace it"
            )

        model_state = model.get_state_for_audio_prompt(audio, truncate=True)
        export_model_state(model_state, destination)

        compiled_entries.append(
            PromptBankEntry(
                name=entry.name,
                audio_path=_portable_relative(audio, output.parent),
                affect=entry.affect,
                prompt_text=entry.prompt_text,
                state_path=_portable_relative(destination, output.parent),
                audio_sha256=_sha256(audio),
            )
        )

    compiled = PromptBank(
        manifest_path=output,
        bank_id=bank.bank_id,
        speaker_id=bank.speaker_id,
        language=bank.language,
        rights_confirmed=True,
        recording_protocol_version=bank.recording_protocol_version,
        entries=tuple(compiled_entries),
        compiled_for=CompiledFor(
            model_ref=model_ref,
            config_sha256=_config_sha256(model),
        ),
    )
    output.write_text(
        json.dumps(prompt_bank_to_dict(compiled), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return compiled


def validate_bank_files(
    bank: PromptBank,
    *,
    require_states: bool = False,
) -> tuple[str, ...]:
    """Return deterministic file-integrity problems without loading the TTS model."""

    problems: list[str] = []
    for entry in bank.entries:
        audio = bank.root / entry.audio_path
        if not audio.is_file():
            problems.append(f"{entry.name}: missing audio file {audio}")
            continue
        if entry.audio_sha256 is not None and _sha256(audio) != entry.audio_sha256:
            problems.append(f"{entry.name}: audio SHA-256 mismatch")

        if require_states:
            if entry.state_path is None:
                problems.append(f"{entry.name}: no exported state_path")
            elif not (bank.root / entry.state_path).is_file():
                problems.append(
                    f"{entry.name}: missing exported state {bank.root / entry.state_path}"
                )
    return tuple(problems)
