# Project State

## Goal

Evolve this fork of Kyutai Pocket TTS into a context-aware expressive TTS while preserving the
project's defining properties: small model size, CPU-first inference, streaming, voice cloning,
simple dependencies, and easy upstream synchronization.

## Repository state

- Repository: `SanamRai001/pocket-tts`
- Default branch: `main`
- Work branch: `feat/expressive-tts-foundation`
- Phase-1 base commit: `41cbc84af539ea78a804ffca5f9c6edc1a22ce44`
- Fork was unmodified before this phase: one `main` branch and no fork-specific PRs.
- Upstream behavior is intentionally untouched in Phase 1.

## Current phase

**Phase 1 - expressive routing foundation**

The architecture is documented in `docs/expressive-speech/ARCHITECTURE.md`.

### Decisions

- Keep `TTSModel` and all existing CLI/server behavior unchanged.
- Represent semantic emotion continuously with valence/arousal/dominance/intensity.
- Start with reference-prompt routing instead of adding model parameters.
- Use same-speaker prompt banks for identity consistency.
- Add a style-switch penalty to reduce rapid routing changes.
- Cache only two prompt states by default to bound memory.
- Do not interpolate exported `ModelState` tensors without experiments.
- Keep ChatGPT/LLM planning optional and outside the synthesis core.
- Do not add a dependency in Phase 1.

## Changes in this phase

- Added `pocket_tts/expressive.py`:
  - `AffectVector`
  - `StyleProfile`
  - `ExpressiveSegment`
  - `ExpressivePlan`
  - `RoutingWeights`
  - `AffectivePromptRouter`
  - `ExpressiveGenerator`
  - bounded LRU prompt-state cache
  - streaming rendering and optional inter-segment silence
- Added focused unit tests using a fake model, so the tests do not download weights.
- Added the expressive architecture/research roadmap.

## Expected UX

Normal Pocket TTS users see no change.

Opt-in users can prepare several expressive recordings of one speaker, describe each with affect
coordinates, and feed a semantically planned passage to `ExpressiveGenerator`. The router selects
the closest reference performance while applying hysteresis between adjacent segments.

## Verification

Not yet verified on the user's machine in this phase.

Run the lightweight checks first:

```powershell
uv run pytest tests/test_expressive.py -v
uv run ty check
uvx pre-commit run --all-files
```

Expected:

- all expressive unit tests pass
- `ty` reports no type errors
- pre-commit makes no remaining changes/failures

Do not run full model-generation tests until the lightweight checks are green.

## Risks

1. **Reference leakage:** Pocket TTS copies more than emotion from a prompt, including cadence and
   acoustic conditions. Emotional prompts must be clean and intentionally recorded.
2. **Speaker drift:** using different speakers for different emotions will sound like speaker
   switching, not emotional delivery.
3. **Boundary artifacts:** segment-by-segment generation can create audible transitions. Phase 1
   measures this before adding crossfades or teacher-forcing continuity.
4. **Planner errors:** semantic affect prediction can be wrong even when synthesis works perfectly.
   Planner quality and TTS quality must be evaluated separately.
5. **Cache trade-off:** a two-state cache bounds memory but repeated switching among many WAV
   prompts can trigger re-encoding. Exported `.safetensors` prompt states are preferred.
6. **No continuous acoustic interpolation yet:** affect is continuous semantically, but Phase 1
   renders through the nearest validated prompt profile.

## Acceptance criteria for Phase 1

- No regression to the existing public API.
- No new runtime dependency.
- No increase to model parameter count or weight size.
- Existing streaming path remains the synthesis engine.
- Routing is deterministic and testable without model downloads.
- Prompt-state memory is explicitly bounded.
- Exact style overrides work.
- Automatic routing uses continuous affect and avoids trivial style flapping.

## Next phase

After the lightweight checks pass:

1. define the JSON interchange schema for semantic planners
2. add a ChatGPT-ready planner prompt/schema without making OpenAI a runtime dependency
3. create a same-speaker prompt-bank recording protocol
4. add a small evaluation fixture for mixed-emotion passages
5. measure boundary quality, RTF, memory, WER, speaker similarity, and emotion agreement
6. only then decide whether an affect-prefix adapter is justified
