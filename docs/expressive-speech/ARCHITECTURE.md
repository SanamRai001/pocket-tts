# Expressive Pocket TTS Architecture

## Goal

Add text-aware emotional delivery while preserving the properties that make Pocket TTS
valuable: about 100M generative parameters, CPU-first inference, streaming, voice cloning,
small installation surface, and easy upstream rebases.

The first rule is therefore: **do not turn Pocket TTS into a large instruction-TTS model**.

## What the existing model already gives us

Pocket TTS conditions generation with an encoded audio prompt. Kyutai explicitly notes that
the prompt carries voice color, emotion, accent, cadence, and acoustic conditions. That means
the cheapest expressive-control surface already exists: reference audio.

The current model state is not a simple speaker embedding. It contains stateful transformer
data after the voice prompt has conditioned the model. For that reason, arbitrary linear
interpolation of two exported model states is not considered safe. We will validate any latent
mixing experimentally before it becomes part of the design.

## Design principles

1. **Core model first.** Existing `TTSModel` behavior remains unchanged by default.
2. **No mandatory LLM.** Semantic emotion planning is an optional layer outside synthesis.
3. **Continuous semantics, discrete rendering first.** Represent affect continuously, then map
   it to a small bank of real reference performances.
4. **Same speaker across styles.** Neutral/happy/sad/etc. prompts should be recordings of the
   same person whenever speaker identity matters.
5. **Bounded memory.** Style prompt states are lazy-loaded through a small LRU cache.
6. **Streaming stays streaming.** Style routing happens between semantic segments; each segment
   still uses Pocket TTS's normal streaming generator.
7. **Measure before training.** Do not add parameters until the prompt-routing ceiling is known.

## Inspiration

This design borrows principles rather than implementations:

- Kyutai Pocket TTS technical report:
  https://kyutai.org/pocket-tts-technical-report/
- Continuous Audio Language Models (CALM):
  https://arxiv.org/abs/2509.06926
- Global Style Tokens (GST), which demonstrated compact style-bank control:
  https://arxiv.org/abs/1803.09017
- EmoSphere-TTS, which motivates continuous affect/intensity rather than only class labels:
  https://arxiv.org/abs/2406.07803
- EXPRESSO, which provides expressive speech styles and useful evaluation/data ideas:
  https://arxiv.org/abs/2308.05725
- Parler-TTS, which demonstrates the usability of natural-language delivery descriptions:
  https://huggingface.co/parler-tts/parler-tts-mini-expresso
- StyleTTS 2 is a quality reference, not an architectural target for this fork:
  https://arxiv.org/abs/2306.07691

## Phase 1: Affective Prompt Routing (APR)

Working name: **Affective Prompt Routing**.

A semantic planner divides text into meaningful segments. Each segment carries:

- valence in [-1, 1]
- arousal in [-1, 1]
- dominance in [-1, 1]
- intensity in [0, 1]
- optional exact style override
- optional pause after the segment

A style bank contains reference prompts with prototype coordinates in the same space.

Example:

```text
neutral -> V= 0.00 A= 0.00 D= 0.00 I=0.00
happy   -> V= 0.80 A= 0.65 D= 0.30 I=0.80
sad     -> V=-0.80 A=-0.45 D=-0.30 I=0.70
```

For target affect x and candidate profile p, routing uses:

```text
score(p) =
    wv * (Vx - Vp)^2
  + wa * (Ax - Ap)^2
  + wd * (Dx - Dp)^2
  + wi * (Ix - Ip)^2
  + switch_penalty * [p != previous_style]
```

The final term is lightweight hysteresis. It prevents rapid style switching when two candidates
are nearly tied, which should reduce audible discontinuities without another model.

An explicit style override bypasses nearest-profile routing.

### Why this is cheap

APR adds:

- no model parameters
- no model weights
- no third-party dependency
- O(number_of_styles) scalar routing work per segment
- at most a configurable number of cached prompt states

The expensive operation remains the same Pocket TTS generation that already existed.

### Recommended prompt storage

For development, profiles may point to WAV files. For repeated use, export each emotional prompt
to a `.safetensors` state and route to those files. Loading an exported state is much cheaper than
re-encoding the audio prompt.

Keep the prompt bank small at first. A useful initial set is:

```text
neutral
warm
happy
excited
sad
calm
angry
fearful
```

Do not collect eight different speakers. Record one consenting speaker performing all eight.

## Planner boundary

The synthesizer should not know whether affect came from ChatGPT, another LLM, a tiny local
classifier, application metadata, or a human editor.

Target architecture:

```text
text
  |
  v
semantic planner (optional, replaceable)
  |
  v
ExpressivePlan: text segments + V/A/D/intensity
  |
  v
Affective Prompt Router
  |
  v
bounded prompt-state cache
  |
  v
unchanged Pocket TTS streaming generation
  |
  v
audio
```

This separation is important. A hosted LLM can provide excellent contextual understanding when
available, but Pocket TTS must remain useful offline and on-device.

## Why not infer emotion from punctuation alone?

Punctuation is a useful prosody hint but not semantic understanding. The same sentence can be
sarcastic, relieved, grieving, or excited depending on context. We can use punctuation in a tiny
offline fallback later, but it should not define the architecture.

## Why not blend exported ModelState values now?

The exported state contains transformer/KV state after conditioning, not an isolated style vector.
Linear interpolation may create states the model never saw in training. Kyutai's own CALM work is
careful about where interpolation/guidance is applied. Therefore Phase 1 only selects validated
prompt states.

Continuous acoustic control remains a research track, not a hidden assumption.

## Phase 2: Planner interchange

Add a stable JSON plan schema and adapters:

1. manual JSON
2. ChatGPT/LLM-produced JSON
3. dependency-free local heuristic fallback
4. optional tiny local emotion classifier, only if its accuracy/size trade-off is worthwhile

The base package must not gain a mandatory transformer/LLM dependency.

## Phase 3: Evaluation harness

A feature is not successful just because it sounds expressive once. Compare baseline Pocket TTS
against APR on a fixed corpus.

Track:

- WER / intelligibility
- speaker similarity
- UTMOS or another quality proxy
- emotion recognition agreement on generated audio
- real-time factor on CPU
- peak RSS / prompt-state memory
- time to first audio chunk
- human pairwise preference for naturalness and emotional fit

Include emotionally mixed paragraphs, not only isolated emotional sentences.

## Phase 4: Continuous control experiment

Only after APR has a measured baseline, test a tiny learned **affect-prefix adapter**:

```text
[V, A, D, I] -> tiny MLP -> one/few conditioning vectors -> FlowLM prefix
```

A 4 -> small-hidden -> model-dimension projector can remain tiny relative to the 100M-parameter
model. It would require training/fine-tuning and careful distillation, but it is a much more
principled continuous-control path than interpolating arbitrary KV states.

Candidate training ideas:

- pseudo-label expressive data with V/A/D estimates
- preserve speaker identity separately from affect
- train a larger teacher if needed, then distill back to the six-layer student
- keep the affect adapter optional/config-driven
- evaluate whether it improves smooth intensity control enough to justify any added parameters

## Phase 5: Hierarchical emotion trajectory

If phrase-level routing works, move from one emotion per sentence to an emotion trajectory across
the utterance. Recent emotional-TTS work suggests that word/phrase/utterance levels carry different
information. We should only add this after boundary artifacts and planner stability are solved.

## Non-goals for the early fork

- replacing Mimi
- replacing CALM/FlowLM
- bundling a large language model
- requiring GPU inference
- adding a diffusion model to the runtime
- changing normal `TTSModel.generate_audio*` behavior
- claiming latent blending works before experiments show it

## Phase 1 API sketch

```python
from pocket_tts import TTSModel
from pocket_tts.expressive import (
    AffectVector,
    AffectivePromptRouter,
    ExpressiveGenerator,
    ExpressivePlan,
    ExpressiveSegment,
    StyleProfile,
)

model = TTSModel.load_model()

profiles = (
    StyleProfile("neutral", "voices/me-neutral.safetensors", AffectVector()),
    StyleProfile(
        "happy",
        "voices/me-happy.safetensors",
        AffectVector(valence=0.8, arousal=0.6, dominance=0.2, intensity=0.8),
    ),
    StyleProfile(
        "sad",
        "voices/me-sad.safetensors",
        AffectVector(valence=-0.8, arousal=-0.4, dominance=-0.2, intensity=0.7),
    ),
)

router = AffectivePromptRouter(profiles, default_style="neutral")
expressive = ExpressiveGenerator(model, router)

plan = ExpressivePlan(
    segments=(
        ExpressiveSegment(
            "I cannot believe we actually won!",
            AffectVector(valence=0.95, arousal=0.9, dominance=0.3, intensity=0.95),
        ),
        ExpressiveSegment(
            "I just wish Dad were here to see it.",
            AffectVector(valence=-0.65, arousal=-0.35, dominance=-0.2, intensity=0.65),
        ),
    )
)

audio = expressive.generate_audio(plan)
```

The semantic planner is intentionally absent from this example. It is the next layer, not a
requirement for the synthesizer.
