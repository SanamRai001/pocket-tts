# Project State

## Goal

Evolve this fork of Kyutai Pocket TTS into a context-aware expressive TTS while preserving the
project's defining properties: small model size, CPU-first inference, streaming, voice cloning,
simple dependencies, and easy upstream synchronization.

## Repository state

- Repository: `SanamRai001/pocket-tts`
- Default branch: `main`
- Work branch: `feat/expressive-tts-foundation`
- Draft PR: `#1 Experimental expressive TTS routing foundation`
- Phase base commit: `41cbc84af539ea78a804ffca5f9c6edc1a22ce44`
- `main` remains untouched by the expressive work.
- The fork was clean/upstream-equivalent before this branch.

## Current milestone

**Expressive foundation milestone: Phase 1 + Phase 2 implemented, verification deferred.**

The user explicitly chose to continue building before spending time on local/full verification.
Tests are present in the branch, but results must not be claimed until they are actually run.

Architecture:

- `docs/expressive-speech/ARCHITECTURE.md`
- `docs/expressive-speech/PLANNER_PROTOCOL.md`
- `docs/expressive-speech/expressive-plan-v1.schema.json`

## Phase 1 - Affective Prompt Routing

Implemented:

- continuous `AffectVector`
  - valence
  - arousal
  - dominance
  - intensity
- `StyleProfile` prompt bank entries
- `ExpressiveSegment` and `ExpressivePlan`
- deterministic nearest-profile routing
- style-switch hysteresis
- explicit style overrides
- bounded LRU prompt-state cache, default size 2
- streaming rendering through the unchanged Pocket TTS generator
- optional inter-segment silence
- planner-confidence damping toward neutral

### Phase 1 design decisions

- Keep `TTSModel` and existing CLI/server behavior unchanged.
- Add no model parameters.
- Add no model weights.
- Add no runtime dependencies.
- Keep semantic emotion understanding outside the acoustic model.
- Do not interpolate exported `ModelState` tensors without experiments.
- Prefer same-speaker recordings across every emotional style.

## Phase 2 - Planner Interchange

Implemented:

- versioned planner contract: `schema_version = "1.0"`
- strict JSON parsing
- machine-readable JSON Schema
- serialization helpers
- ChatGPT/provider-neutral planner prompt builder
- source-text preservation validation
- rejection of unknown planner fields
- affect numeric range validation
- optional allowed-style allow-list
- confidence-aware conservative routing
- prompt-injection boundary: planner output is treated as untrusted structured input
- dependency-free offline English heuristic fallback

### Important planner rule

The planner decides **how the supplied text should sound**.

It must never decide **what text should be spoken**.

When `source_text` is supplied, the parser checks that all planned segments reconstruct the
original text apart from whitespace normalization. Paraphrasing, translation, omissions,
insertions, and "helpful" grammar corrections are rejected.

### Offline fallback

`heuristic_plan()` uses only a tiny English lexical/punctuation heuristic.

It deliberately:

- has no model dependency
- uses modest confidence
- gets damped toward neutral
- is not presented as equivalent to ChatGPT/contextual semantic reasoning

Its purpose is graceful offline behavior, not state-of-the-art emotion recognition.

## Performance / size impact so far

Expected architectural impact:

- additional neural parameters: **0**
- additional model weights: **0**
- mandatory dependencies added: **0**
- routing complexity: O(number of prompt styles) per semantic segment
- prompt-state memory: explicitly bounded by the LRU cache

These are architectural properties of the code. Runtime speed/memory numbers still require
measurement.

## Files added or changed by the expressive branch

Core:

- `pocket_tts/expressive.py`
- `pocket_tts/expressive_planner.py`

Tests:

- `tests/test_expressive.py`
- `tests/test_expressive_planner.py`

Docs/contracts:

- `docs/expressive-speech/ARCHITECTURE.md`
- `docs/expressive-speech/PLANNER_PROTOCOL.md`
- `docs/expressive-speech/expressive-plan-v1.schema.json`
- `docs/PROJECT_STATE.md`

No existing Pocket TTS synthesis/model implementation has been modified.

## Verification status

**Deferred intentionally.**

Do not claim these pass until the user runs them or CI results are explicitly reviewed.

Focused checks for the current milestone:

```powershell
git fetch origin
git switch feat/expressive-tts-foundation
git pull --ff-only

uv run pytest tests/test_expressive.py tests/test_expressive_planner.py -v
uv run ty check
uvx pre-commit run --all-files
```

After those are green, the repository's full suite is:

```powershell
uv run pytest -n 3 -v
```

A real audio evaluation is also required before calling expressive synthesis successful.

## Known risks

1. **Reference leakage**
   Pocket TTS prompts carry more than emotion: cadence, acoustic conditions, accent, and other
   characteristics can leak into output.

2. **Speaker drift**
   Different speakers across emotional prompts will sound like speaker switching. The first real
   bank should use one consenting speaker across all styles.

3. **Segment boundary artifacts**
   Independent prompt-conditioned segments may create audible transitions. This needs listening
   evaluation before adding crossfades or teacher-forcing continuity.

4. **Planner errors**
   A semantic planner can misunderstand sarcasm, ambiguity, mixed emotions, or narrative context.
   Confidence damping reduces overacting but does not solve planner accuracy.

5. **English-only heuristic fallback**
   The local heuristic is intentionally small and English-specific. Other languages should remain
   neutral or use a capable external planner until a justified fallback exists.

6. **No continuous acoustic control yet**
   Affect is continuous at the semantic layer, but V1 renders through discrete prompt profiles.

## Next phase

**Phase 3A - prompt-bank and first real expressive-audio experiment**

Before training or changing FlowLM:

1. define a same-speaker recording protocol
2. define the minimum useful style bank and prototype V/A/D/I coordinates
3. add a prompt-bank manifest format
4. support exporting prompt WAVs to cached `.safetensors` states
5. create a fixed mixed-emotion evaluation script/corpus
6. compare baseline Pocket TTS vs expressive routing on actual audio
7. measure:
   - time to first chunk
   - real-time factor
   - memory
   - WER
   - speaker similarity
   - audio quality
   - emotion agreement
   - human preference

Only after those results should we decide whether Phase 4's tiny learned affect-prefix adapter is
worth the complexity.
