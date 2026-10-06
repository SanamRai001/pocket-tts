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

**Expressive foundation milestone: Phase 1 + Phase 2 + Phase 3A infrastructure implemented,
verification deferred.**

The user explicitly chose to continue building before spending time on local/full verification.
Tests are present in the branch, but results must not be claimed until they are actually run.

Phase 3A code/tooling is complete enough to capture and compile a real same-speaker bank. The one
external input still required for the controlled experiment is the authorized reference audio
itself.

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

## Phase 3A - Same-Speaker Prompt Bank + A/B Experiment

Implemented:

- versioned prompt-bank manifest
- machine-readable prompt-bank JSON Schema
- required neutral anchor
- portable relative-path validation
- generic `rights_confirmed` guard before voice-state export
- source-audio SHA-256 fingerprints
- model/config compatibility metadata for compiled states
- model/config mismatch rejection
- same-speaker eight-style example bank
- controlled recording protocol
- prompt WAV -> `.safetensors` bank compiler
- fixed hand-authored emotional evaluation corpus
- neutral-vs-expressive A/B audio generation script
- timing, duration, routing, and speed report
- output placed under the already-ignored `runs/` directory

The first bank prototypes are:

- neutral
- warm
- happy
- excited
- calm
- sad
- angry
- fearful

Their initial V/A/D/I coordinates are experimental prototypes and should be tuned from real
evaluation rather than treated as universal labels.

### Controlled experiment principle

The first recordings use the same speaker and the same text across all styles. This isolates
prosody/style better than giving every emotion different words.

The fixed evaluation corpus contains manual affect trajectories. That separates synthesis/routing
quality from planner quality for the first A/B experiment.

### Public research bootstrap

Kyutai's public Expresso voice collection contains same-speaker expressive 10-second clips and can
be useful for a temporary research/debug bank. The Expresso material is CC BY-NC 4.0 and
non-commercial, so it is not the default product-oriented bank.

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
- `pocket_tts/expressive_prompt_bank.py`

Scripts:

- `scripts/export_expressive_prompt_bank.py`
- `scripts/generate_expressive_eval.py`

Tests:

- `tests/test_expressive.py`
- `tests/test_expressive_planner.py`
- `tests/test_expressive_prompt_bank.py`

Docs/contracts:

- `docs/expressive-speech/ARCHITECTURE.md`
- `docs/expressive-speech/PLANNER_PROTOCOL.md`
- `docs/expressive-speech/RECORDING_PROTOCOL.md`
- `docs/expressive-speech/expressive-plan-v1.schema.json`
- `docs/expressive-speech/expressive-prompt-bank-v1.schema.json`
- `docs/expressive-speech/prompt-bank.example.json`
- `docs/expressive-speech/eval-corpus-v1.json`
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

uv run pytest tests/test_expressive.py tests/test_expressive_planner.py tests/test_expressive_prompt_bank.py -v
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

**Phase 3B - run the first real prompt-bank experiment and add objective scoring**

The infrastructure is ready. The next meaningful step is data + measurement:

1. capture/select one authorized same-speaker bank using `RECORDING_PROTOCOL.md`
2. compile it with `scripts/export_expressive_prompt_bank.py`
3. generate the fixed A/B set with `scripts/generate_expressive_eval.py`
4. listen for:
   - emotional distinctness
   - speaker consistency
   - clipping/noise leakage
   - abrupt segment boundaries
   - overacting/underacting
5. add objective evaluation:
   - WER
   - speaker similarity
   - quality proxy such as UTMOS
   - emotion agreement
   - peak memory
   - time to first chunk
6. compare prompt-bank routing against ordinary neutral Pocket TTS

Do not train the affect-prefix adapter yet. The current prompt-conditioned ceiling needs to be
measured first.
