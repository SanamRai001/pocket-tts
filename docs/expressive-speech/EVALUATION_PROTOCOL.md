# Expressive Evaluation Protocol v1

## Goal

Measure whether Affective Prompt Routing improves emotional delivery without damaging the
properties that make Pocket TTS useful.

The first comparison is deliberately controlled:

- A: ordinary Pocket TTS using the neutral reference
- B: the same Pocket TTS model using hand-authored affect trajectories and prompt-bank routing

Text, model weights, speaker, and corpus stay fixed. This isolates routing/synthesis before planner
quality is introduced.

## Evaluation layers

The lightweight bootstrap/generate/score/listening workflow uses `uv run --no-dev` on purpose.
That avoids installing the large optional research-scoring stack until it is explicitly needed.


### Tier 0: always available

No additional scorer model is required.

The generation report records:

- routing trace
- audio duration
- total generation time
- real-time speed
- time to first audio chunk

The lightweight scorer additionally measures:

- finite/non-finite samples
- peak amplitude
- RMS and RMS dBFS
- DC offset
- near-clipping fraction
- near-silence fraction

These are regression guards, not emotion metrics.

### Tier 1: optional research scorers

These use the project dev environment and may download pretrained models:

- ASR WER for intelligibility
- WavLM speaker similarity
- UTMOS quality proxy

They are evaluation-time only and are never imported by the normal Pocket TTS synthesis path.

For speaker similarity, the scorer follows the upstream Pocket TTS evaluation principle and uses
microsoft/wavlm-base-plus-sv.

For CPU-oriented WER experiments, pass a smaller ASR model explicitly, for example
openai/whisper-base.en. Do not compare WER values produced by different ASR models as though they
were directly equivalent.

## Human blind A/B

Emotion appropriateness needs human listening.

The blind tool randomizes whether A or B is baseline for each item. The listener rates:

- preferred clip
- emotional fit
- naturalness
- speaker consistency
- transition smoothness
- notes

Do not open answer-key.json until ratings are complete.

Use a 1-5 scale. Transition smoothness can be left blank for single-style items.

## Fixed corpus

Use docs/expressive-speech/eval-corpus-v1.json.

It includes:

- neutral factual speech
- warm reassurance
- excitement
- calm reassurance
- restrained sadness
- controlled anger
- fear
- joy to grief
- anxiety to relief
- nostalgia to sadness

The trajectories are hand-authored so the first experiment does not mix planner errors with
routing/synthesis errors.

## Optional research-only bootstrap

If no authorized custom bank is available yet, the repository can create a temporary bank from
Kyutai's public Expresso voice clips:

~~~powershell
uv run --no-dev python scripts/bootstrap_expresso_research_bank.py
~~~

The helper:

- discovers Expresso clips from the Kyutai voice repository
- chooses one speaker with a neutral/default anchor and the broadest supported style coverage
- downloads only the selected clips
- writes a normal Pocket TTS prompt-bank manifest
- writes RESEARCH_ONLY.md beside the bank
- keeps all generated assets under runs/ by default

Expresso is CC BY-NC 4.0 and is **non-commercial only**. This path exists strictly for research,
debugging, and architecture validation.

After bootstrapping:

~~~powershell
uv run --no-dev python scripts/export_expressive_prompt_bank.py .\runs\expresso-research-bank\prompt-bank.json --language english
~~~

Then use the compiled manifest with the same A/B generation and scoring workflow below.

Do not use the Expresso bootstrap as the product/default bank. Replace it with an appropriately
authorized/licensed same-speaker bank before any commercial use.

## Complete workflow

### 1. Record the bank

Follow docs/expressive-speech/RECORDING_PROTOCOL.md and use one authorized speaker across styles.

### 2. Compile reference states

~~~powershell
uv run --no-dev python scripts/export_expressive_prompt_bank.py .\my-prompt-bank\prompt-bank.json --language english
~~~

### 3. Generate neutral-vs-expressive audio

~~~powershell
uv run --no-dev python scripts/generate_expressive_eval.py .\my-prompt-bank\prompt-bank.compiled.json --language english --output-dir .\runs\expressive-eval-v1
~~~

The report includes time-to-first-chunk so expressive routing cannot hide a streaming regression
behind a good total real-time factor.

### 4. Run lightweight scoring

~~~powershell
uv run --no-dev python scripts/score_expressive_eval.py .\runs\expressive-eval-v1
~~~

### 5. Optional WER

~~~powershell
uv run python scripts/score_expressive_eval.py .\runs\expressive-eval-v1 --asr-model openai/whisper-base.en
~~~

### 6. Optional speaker similarity and UTMOS

~~~powershell
uv run python scripts/score_expressive_eval.py .\runs\expressive-eval-v1 --speaker-similarity --utmos
~~~

These can be slow on CPU and are not prerequisites for the first listening test.

### 7. Prepare the blind listening set

~~~powershell
uv run --no-dev python scripts/prepare_expressive_blind_eval.py .\runs\expressive-eval-v1
~~~

This creates the randomized audio set, trials.json, ratings.csv, and a separate answer-key.json.

### 8. Rate every item before unblinding

For preferred, enter A, B, or Tie. Fill the 1-5 score columns where applicable.

### 9. Unblind and summarize

~~~powershell
uv run --no-dev python scripts/summarize_expressive_blind_eval.py .\runs\expressive-eval-v1\blind
~~~

The summary reports condition-level means, preference counts, and expressive-minus-baseline score
deltas.

## Decision gates

### Gate A: technical health

Required:

- no NaN/Inf audio
- no systematic clipping increase
- no pathological silence
- CPU speed remains usable
- time-to-first-chunk does not regress unacceptably

### Gate B: intelligibility and identity

Desired:

- WER not materially worse than neutral baseline
- speaker similarity stays close to baseline
- no broad UTMOS quality collapse

### Gate C: emotional usefulness

Desired:

- expressive condition wins preference on emotionally loaded items
- emotional-fit scores improve
- multi-emotion transitions are acceptable
- neutral factual speech does not become overacted

### Gate D: identify the actual bottleneck

Failures should be classified before changing architecture:

1. prompt quality problem
2. V/A/D/I bank-geometry problem
3. segment-boundary problem
4. Pocket TTS conditioning ceiling
5. planner problem

Do not respond to every failure by making the model larger.

## When Phase 4 is justified

A learned affect-prefix adapter is justified only if evaluation shows that prompt routing already
creates useful emotional differences, speaker identity remains stable, and the remaining measured
limitation is coarse/discrete control or transition smoothness.

Until then, prompt-conditioned routing is the smaller and safer solution.
