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

**Expressive foundation milestone: Phase 1 + Phase 2 + Phase 3A + Phase 3B evaluation tooling
implemented, verification and real-audio measurement deferred.**

The user explicitly chose to continue building before spending time on local/full verification.
Tests are present in the branch, but results must not be claimed until they are actually run.

The software side of the first controlled expressive experiment is now complete. For a product-
oriented conclusion, an authorized same-speaker emotional reference bank is still required.

For research/debugging before that bank exists, the branch includes two non-commercial bootstraps:
EARS is the preferred controlled same-speaker emotion bank, while Expresso remains a secondary
conversational stress-test bank.

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
- timing, duration, routing, speed, and time-to-first-chunk report
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

Preferred: Kyutai's EARS voice collection exposes speakers `p003` and `p031` with many
`emo_*_freeform.wav` recordings for the same speaker, which better isolates emotion from
speaker/session changes.

Secondary: Expresso remains useful as a conversational stress test with 10-second expressive
segments.

Both EARS and Expresso material used here are CC BY-NC 4.0 and non-commercial only.

## Phase 3B - Measurement and Blind Evaluation Tooling

Implemented:

- time-to-first-audio measurement for baseline and expressive streams
- dependency-light waveform diagnostics:
  - finite sample check
  - peak level
  - RMS / RMS dBFS
  - DC offset
  - near-clipping fraction
  - near-silence fraction
- optional ASR WER scoring through a user-selected Hugging Face ASR model
- optional WavLM speaker similarity using microsoft/wavlm-base-plus-sv
- optional UTMOS quality scoring
- deterministic blinded A/B audio randomization
- ratings CSV for emotional fit, naturalness, speaker consistency, and transition smoothness
- separate answer key
- unblinding and aggregate preference/score summary
- evaluation decision gates documented in EVALUATION_PROTOCOL.md

The heavy scorers remain evaluation-only and are never imported by the normal Pocket TTS runtime
path unless explicitly requested by the evaluation script.

Emotion agreement is intentionally not wired to one arbitrary classifier yet. A suitable emotion
scorer should be selected only after the first real bank exists, because label ontology and
continuous V/A/D compatibility matter.

Peak native-process memory is also still unmeasured; adding a misleading Python-only memory metric
would be worse than leaving it explicit for the real experiment.

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
- `scripts/score_expressive_eval.py`
- `scripts/prepare_expressive_blind_eval.py`
- `scripts/summarize_expressive_blind_eval.py`
- `scripts/bootstrap_ears_research_bank.py`
- `scripts/bootstrap_expresso_research_bank.py`
- `scripts/run_expressive_research_experiment.py`
- `scripts/inspect_expressive_routes.py`

Tests:

- `tests/test_expressive.py`
- `tests/test_expressive_planner.py`
- `tests/test_expressive_prompt_bank.py`
- `tests/test_expressive_eval_tools.py`
- `tests/test_ears_bootstrap.py`
- `tests/test_expresso_bootstrap.py`
- `tests/test_expressive_route_preview.py`

Docs/contracts:

- `docs/expressive-speech/ARCHITECTURE.md`
- `docs/expressive-speech/PLANNER_PROTOCOL.md`
- `docs/expressive-speech/RECORDING_PROTOCOL.md`
- `docs/expressive-speech/EVALUATION_PROTOCOL.md`
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

uv run pytest tests/test_expressive.py tests/test_expressive_planner.py tests/test_expressive_prompt_bank.py tests/test_expressive_eval_tools.py tests/test_ears_bootstrap.py tests/test_expresso_bootstrap.py tests/test_expressive_route_preview.py -v
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

6. **Source WAV encoding compatibility**
   Public research datasets may use valid WAV encodings that Python's standard `wave` module
   cannot decode. The EARS bootstrap now normalizes downloaded clips to PCM16 before Pocket TTS
   sees them.

7. **No continuous acoustic control yet**
   Affect is continuous at the semantic layer, but V1 renders through discrete prompt profiles.

## Research-only bootstrap path

Preferred controlled bank:

`scripts/bootstrap_ears_research_bank.py`

- supports Kyutai EARS emotion speakers `p003` and `p031`
- uses the same speaker across neutral/contentment/amusement/amazement/serenity/sadness/anger/fear
- maps those native source emotions to the project's eight research roles
- defaults to original recordings, with optional `--enhanced` cleaned WAVs
- writes the standard prompt-bank manifest and RESEARCH_ONLY.md

Secondary stress-test bank:

`scripts/bootstrap_expresso_research_bank.py`

- automatically selects one speaker with a neutral/default anchor
- uses the broadest supported conversational style coverage
- is useful for testing robustness to less controlled expressive references

Both are CC BY-NC 4.0 and non-commercial research/evaluation only.

The one-command orchestrator defaults to EARS:

```powershell
uv run --no-dev python scripts/run_expressive_research_experiment.py
```

Expresso remains selectable:

```powershell
uv run --no-dev python scripts/run_expressive_research_experiment.py --source expresso
```

The runner writes `route-preview.json` before loading model weights, compiles prompt states,
generates the fixed A/B corpus, writes Tier 0 scores, and prepares the blinded listening set under
`runs/expressive-research-v1`.

A manual GitHub Actions workflow also exists at
`.github/workflows/expressive-research-experiment.yml`. GitHub only exposes
`workflow_dispatch` once the workflow file exists on the repository default branch, so while this
work remains isolated on the feature branch the immediate execution path is the local one-command
runner. The workflow is a post-merge/cloud convenience, not a claimed current CI run.

## Next phase

**Phase 3C - run the first prompt-bank experiment**

The implementation and evaluation harness are ready. The next meaningful work is empirical.

Phase 3C hardening completed before the first expensive run:

- one-command experiment is rerun-safe (`--overwrite` on compiled prompt states)
- child scripts execute with the repository root as their working directory
- lightweight research commands use `uv --no-dev` so Transformers/UTMOS/torchaudio are not
  installed unless optional research scoring is requested
- a pre-synthesis route preview is written before model loading
- route preview warns when high-intensity segments collapse to neutral
- route preview reports style usage and unused styles
- new route-preview and bootstrap behavior has focused network-free tests
- EARS source WAVs are normalized to PCM16 during bootstrap because the source files can use
  IEEE-float WAV format (`wFormatTag=3`), which Python 3.10's built-in `wave` reader rejects
- PCM16 conversion is validated through the same `wave` path Pocket TTS uses before synthesis

For research/debugging, run the complete lightweight experiment:

```powershell
uv run --no-dev python scripts/run_expressive_research_experiment.py
```

For product-oriented evaluation, capture/select one authorized same-speaker bank using
`RECORDING_PROTOCOL.md`, then run the individual compile/generate/score tools against that bank.

After audio exists:

1. inspect the Tier 0 technical report
2. complete the blind A/B ratings before opening the answer key
3. optionally run WER, WavLM similarity, and UTMOS
4. inspect failures by category:
   - prompt quality
   - V/A/D/I bank geometry
   - segment boundary quality
   - Pocket TTS conditioning ceiling
5. only after those results:
   - tune prompt prototypes
   - consider boundary smoothing/teacher forcing
   - select an emotion-agreement scorer
   - measure native peak memory
   - decide whether the learned affect-prefix adapter is justified

Do not train the affect-prefix adapter yet. The current prompt-conditioned ceiling still needs to
be measured first.
