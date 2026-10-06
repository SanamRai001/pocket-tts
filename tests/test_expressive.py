"""Unit tests for the dependency-free expressive routing layer."""

from collections.abc import Iterator
import threading

import pytest
import torch

from pocket_tts.expressive import (
    AffectVector,
    AffectivePromptRouter,
    ExpressiveGenerator,
    ExpressivePlan,
    ExpressiveSegment,
    RoutingWeights,
    StyleProfile,
    VoiceSource,
)
from pocket_tts.modules.stateful_module import ModelState


class FakeExpressiveModel:
    def __init__(self) -> None:
        self.load_calls: list[str] = []
        self.generate_calls: list[str] = []

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def sample_rate(self) -> int:
        return 1000

    def get_state_for_audio_prompt(
        self, audio_conditioning: VoiceSource, truncate: bool = False
    ) -> ModelState:
        del truncate
        source = str(audio_conditioning)
        self.load_calls.append(source)
        return {"fake": {"id": torch.tensor([len(self.load_calls)])}}

    def generate_audio_stream(
        self,
        model_state: ModelState,
        text_to_generate: str,
        *,
        stop: threading.Event | None = None,
    ) -> Iterator[torch.Tensor]:
        del model_state
        if stop is not None and stop.is_set():
            return
        self.generate_calls.append(text_to_generate)
        yield torch.ones(len(text_to_generate), dtype=torch.float32)


def _profiles() -> tuple[StyleProfile, ...]:
    return (
        StyleProfile("neutral", "neutral.safetensors", AffectVector()),
        StyleProfile(
            "happy",
            "happy.safetensors",
            AffectVector(valence=0.8, arousal=0.65, dominance=0.3, intensity=0.8),
        ),
        StyleProfile(
            "sad",
            "sad.safetensors",
            AffectVector(valence=-0.8, arousal=-0.45, dominance=-0.3, intensity=0.7),
        ),
    )


def test_affect_vector_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="valence"):
        AffectVector(valence=1.1)

    with pytest.raises(ValueError, match="intensity"):
        AffectVector(intensity=-0.1)


def test_router_uses_nearest_affect_profile() -> None:
    router = AffectivePromptRouter(_profiles(), default_style="neutral")
    segment = ExpressiveSegment(
        "We actually did it!",
        AffectVector(valence=0.9, arousal=0.7, dominance=0.2, intensity=0.9),
    )

    assert router.resolve(segment).name == "happy"


def test_router_hysteresis_avoids_small_style_flaps() -> None:
    profiles = (
        StyleProfile("calm", "calm.wav", AffectVector(arousal=0.1, intensity=0.5)),
        StyleProfile("excited", "excited.wav", AffectVector(arousal=0.9, intensity=0.5)),
    )
    router = AffectivePromptRouter(
        profiles,
        default_style="calm",
        weights=RoutingWeights(
            valence=0.0,
            arousal=1.0,
            dominance=0.0,
            intensity=0.0,
            switch_penalty=0.08,
        ),
    )
    segment = ExpressiveSegment(
        "It could go either way.",
        AffectVector(arousal=0.51, intensity=0.5),
    )

    assert router.resolve(segment, previous_style="calm").name == "calm"
    assert router.resolve(segment, previous_style=None).name == "excited"


def test_explicit_style_override_wins_and_unknown_style_fails() -> None:
    router = AffectivePromptRouter(_profiles(), default_style="neutral")
    sad_affect = AffectVector(valence=-1.0, arousal=-0.8, intensity=1.0)

    assert router.resolve(
        ExpressiveSegment("Say it brightly.", sad_affect, style="happy")
    ).name == "happy"

    with pytest.raises(ValueError, match="unknown style"):
        router.resolve(ExpressiveSegment("No.", style="missing"))


def test_generator_keeps_state_cache_bounded_with_lru_eviction() -> None:
    model = FakeExpressiveModel()
    router = AffectivePromptRouter(_profiles(), default_style="neutral")
    generator = ExpressiveGenerator(model, router, max_cached_states=2)
    plan = ExpressivePlan(
        segments=(
            ExpressiveSegment("one", style="neutral"),
            ExpressiveSegment("two", style="happy"),
            ExpressiveSegment("three", style="sad"),
            ExpressiveSegment("four", style="neutral"),
        )
    )

    list(generator.generate_audio_stream(plan))

    assert model.load_calls == [
        "neutral.safetensors",
        "happy.safetensors",
        "sad.safetensors",
        "neutral.safetensors",
    ]
    assert generator.cached_styles == ("sad", "neutral")


def test_generator_preserves_streaming_and_inserts_requested_pause() -> None:
    model = FakeExpressiveModel()
    router = AffectivePromptRouter(_profiles(), default_style="neutral")
    generator = ExpressiveGenerator(model, router)
    plan = ExpressivePlan(
        segments=(
            ExpressiveSegment("hello", style="neutral", pause_after_ms=25),
            ExpressiveSegment("world", style="happy"),
        )
    )

    chunks = list(generator.generate_audio_stream(plan))

    assert model.generate_calls == ["hello", "world"]
    assert [chunk.numel() for chunk in chunks] == [5, 25, 5]
    assert torch.count_nonzero(chunks[1]).item() == 0
    assert generator.routing_trace(plan) == ("neutral", "happy")
