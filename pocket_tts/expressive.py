"""Experimental expressive-speech routing for Pocket TTS.

This module deliberately leaves the acoustic model unchanged. It maps a continuous
affect description to one of several reference-audio prompt states and then delegates
generation to the existing Pocket TTS streaming path.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

import torch

from pocket_tts.modules.stateful_module import ModelState

VoiceSource: TypeAlias = str | Path | torch.Tensor


def _require_finite_range(name: str, value: float, low: float, high: float) -> None:
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}], got {value}")


@dataclass(frozen=True)
class AffectVector:
    """Continuous affect coordinates used by the prompt router.

    Valence, arousal, and dominance use [-1, 1]. Intensity uses [0, 1].
    Neutral speech is represented by all zeros.
    """

    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    intensity: float = 0.0

    def __post_init__(self) -> None:
        _require_finite_range("valence", self.valence, -1.0, 1.0)
        _require_finite_range("arousal", self.arousal, -1.0, 1.0)
        _require_finite_range("dominance", self.dominance, -1.0, 1.0)
        _require_finite_range("intensity", self.intensity, 0.0, 1.0)


@dataclass(frozen=True)
class StyleProfile:
    """One reference prompt and its prototype affect coordinates."""

    name: str
    source: VoiceSource
    affect: AffectVector = AffectVector()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("style profile name cannot be empty")


@dataclass(frozen=True)
class ExpressiveSegment:
    """A text segment plus semantic affect requested by a planner."""

    text: str
    affect: AffectVector = AffectVector()
    style: str | None = None
    pause_after_ms: int = 0

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("expressive segment text cannot be empty")
        if self.style is not None and not self.style.strip():
            raise ValueError("style override cannot be blank")
        if self.pause_after_ms < 0:
            raise ValueError("pause_after_ms cannot be negative")


@dataclass(frozen=True)
class ExpressivePlan:
    """Ordered expressive segments for one utterance or passage."""

    segments: tuple[ExpressiveSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("expressive plan must contain at least one segment")


@dataclass(frozen=True)
class RoutingWeights:
    """Weights for affect distance and style-switch hysteresis."""

    valence: float = 1.0
    arousal: float = 1.2
    dominance: float = 0.35
    intensity: float = 0.8
    switch_penalty: float = 0.08

    def __post_init__(self) -> None:
        for name in ("valence", "arousal", "dominance", "intensity", "switch_penalty"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")


class ExpressiveModel(Protocol):
    """Small protocol implemented by TTSModel and easy to fake in unit tests."""

    @property
    def device(self) -> torch.device: ...

    @property
    def sample_rate(self) -> int: ...

    def get_state_for_audio_prompt(
        self, audio_conditioning: VoiceSource, truncate: bool = False
    ) -> ModelState: ...

    def generate_audio_stream(
        self,
        model_state: ModelState,
        text_to_generate: str,
        *,
        stop: threading.Event | None = None,
    ) -> Iterator[torch.Tensor]: ...


class AffectivePromptRouter:
    """Resolve continuous affect to a discrete prompt profile.

    The switch penalty provides lightweight hysteresis: when two profiles are almost
    equally suitable, retaining the previous style wins. This reduces audible style
    flapping without adding model parameters or another neural network.
    """

    def __init__(
        self,
        profiles: Sequence[StyleProfile],
        default_style: str,
        weights: RoutingWeights | None = None,
    ) -> None:
        self._profiles = tuple(profiles)
        if not self._profiles:
            raise ValueError("at least one style profile is required")

        self._by_name = {profile.name: profile for profile in self._profiles}
        if len(self._by_name) != len(self._profiles):
            raise ValueError("style profile names must be unique")
        if default_style not in self._by_name:
            raise ValueError(f"default style {default_style!r} is not in the style bank")

        self.default_style = default_style
        self.weights = weights or RoutingWeights()

    @property
    def profiles(self) -> tuple[StyleProfile, ...]:
        return self._profiles

    def _score(
        self,
        target: AffectVector,
        profile: StyleProfile,
        previous_style: str | None,
    ) -> float:
        w = self.weights
        p = profile.affect
        score = (
            w.valence * (target.valence - p.valence) ** 2
            + w.arousal * (target.arousal - p.arousal) ** 2
            + w.dominance * (target.dominance - p.dominance) ** 2
            + w.intensity * (target.intensity - p.intensity) ** 2
        )
        if previous_style is not None and profile.name != previous_style:
            score += w.switch_penalty
        return score

    def resolve(
        self,
        segment: ExpressiveSegment,
        previous_style: str | None = None,
    ) -> StyleProfile:
        if segment.style is not None:
            try:
                return self._by_name[segment.style]
            except KeyError as exc:
                raise ValueError(f"unknown style override {segment.style!r}") from exc

        return min(
            self._profiles,
            key=lambda profile: (
                self._score(segment.affect, profile, previous_style),
                profile.name != self.default_style,
                profile.name,
            ),
        )

    def route_plan(self, plan: ExpressivePlan) -> tuple[StyleProfile, ...]:
        result: list[StyleProfile] = []
        previous_style: str | None = None
        for segment in plan.segments:
            profile = self.resolve(segment, previous_style)
            result.append(profile)
            previous_style = profile.name
        return tuple(result)


class ExpressiveGenerator:
    """Render an ExpressivePlan through the unchanged Pocket TTS model."""

    def __init__(
        self,
        model: ExpressiveModel,
        router: AffectivePromptRouter,
        max_cached_states: int = 2,
    ) -> None:
        if max_cached_states < 0:
            raise ValueError("max_cached_states cannot be negative")
        self.model = model
        self.router = router
        self.max_cached_states = max_cached_states
        self._state_cache: OrderedDict[str, ModelState] = OrderedDict()

    @property
    def cached_styles(self) -> tuple[str, ...]:
        return tuple(self._state_cache)

    def clear_cache(self) -> None:
        self._state_cache.clear()

    def _state_for(self, profile: StyleProfile) -> ModelState:
        cached = self._state_cache.get(profile.name)
        if cached is not None:
            self._state_cache.move_to_end(profile.name)
            return cached

        state = self.model.get_state_for_audio_prompt(profile.source)
        if self.max_cached_states > 0:
            self._state_cache[profile.name] = state
            self._state_cache.move_to_end(profile.name)
            while len(self._state_cache) > self.max_cached_states:
                self._state_cache.popitem(last=False)
        return state

    def routing_trace(self, plan: ExpressivePlan) -> tuple[str, ...]:
        return tuple(profile.name for profile in self.router.route_plan(plan))

    def generate_audio_stream(
        self,
        plan: ExpressivePlan,
        stop: threading.Event | None = None,
    ) -> Iterator[torch.Tensor]:
        stop_event = stop or threading.Event()
        previous_style: str | None = None

        for segment in plan.segments:
            if stop_event.is_set():
                break

            profile = self.router.resolve(segment, previous_style)
            state = self._state_for(profile)
            yield from self.model.generate_audio_stream(
                state, segment.text, stop=stop_event
            )

            if segment.pause_after_ms and not stop_event.is_set():
                pause_samples = round(self.model.sample_rate * segment.pause_after_ms / 1000)
                if pause_samples:
                    yield torch.zeros(
                        pause_samples,
                        dtype=torch.float32,
                        device=self.model.device,
                    )

            previous_style = profile.name

    def generate_audio(
        self,
        plan: ExpressivePlan,
        stop: threading.Event | None = None,
    ) -> torch.Tensor:
        chunks = list(self.generate_audio_stream(plan, stop=stop))
        if not chunks:
            return torch.empty(0, dtype=torch.float32, device=self.model.device)
        return torch.cat(chunks, dim=0)
