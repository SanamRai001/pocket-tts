# Expressive Planner Protocol v1

## Purpose

The planner protocol separates **understanding how text should sound** from **generating the
waveform**.

Pocket TTS stays a small CPU-first synthesizer. A semantic planner can be ChatGPT, another LLM,
a local model, application logic, or a human-authored JSON file. Every planner must produce the
same versioned `ExpressivePlan` contract.

This prevents provider lock-in and keeps large language models out of the Pocket TTS runtime.

## Contract

Machine-readable schema:

`docs/expressive-speech/expressive-plan-v1.schema.json`

Python helpers:

`pocket_tts/expressive_planner.py`

Current schema version:

`1.0`

Example:

```json
{
  "schema_version": "1.0",
  "segments": [
    {
      "text": "I cannot believe we actually won!",
      "affect": {
        "valence": 0.95,
        "arousal": 0.9,
        "dominance": 0.3,
        "intensity": 0.95
      },
      "style": null,
      "pause_after_ms": 180,
      "confidence": 0.98
    },
    {
      "text": "I just wish Dad were here to see it.",
      "affect": {
        "valence": -0.65,
        "arousal": -0.35,
        "dominance": -0.2,
        "intensity": 0.65
      },
      "style": null,
      "pause_after_ms": 0,
      "confidence": 0.94
    }
  ]
}
```

## Affect dimensions

### Valence

Emotional positivity.

```text
-1.0  strongly negative
 0.0  neutral
+1.0  strongly positive
```

### Arousal

Activation/energy, not positivity.

```text
-1.0  subdued / low activation
 0.0  neutral
+1.0  highly activated
```

This separates, for example:

- excitement: positive + high arousal
- peaceful happiness: positive + low arousal
- anger: negative + high arousal
- grief: negative + low arousal

### Dominance

Perceived control or forcefulness.

```text
-1.0  vulnerable / submissive
 0.0  neutral
+1.0  forceful / in control
```

### Intensity

How strongly the affect should influence delivery.

```text
0.0  neutral delivery
1.0  strongest justified expression
```

## Confidence

`confidence` represents confidence in the semantic emotion estimate, not audio quality.

The router automatically moves uncertain affect toward neutral:

```text
effective_affect = predicted_affect * confidence
```

Example:

```text
predicted arousal = 0.9
confidence        = 0.2
effective arousal = 0.18
```

This is deliberately conservative. If a planner is unsure whether a sentence is excited,
sarcastic, anxious, or neutral, overacting is usually worse than slightly neutral delivery.

Exact `style` overrides are not damped because they represent an explicit routing instruction.

## Text preservation

A semantic planner is allowed to annotate and split the source text. It is **not** allowed to
rewrite the spoken content.

When the caller supplies `source_text`, `plan_from_json()` validates that all segment text,
joined in order, matches the source apart from whitespace normalization.

This catches:

- paraphrasing
- omitted words
- inserted words
- translation
- "helpful" grammar correction
- prompt-injection text that convinces the planner to replace the requested speech

For TTS, the planner decides **how to say the text**, never **what text to say**.

## Style override

`style` is optional.

Normally the planner should return:

```json
"style": null
```

and allow Affective Prompt Routing to choose the closest profile.

An explicit style should be used only when application context or a human instruction requires a
specific known style.

The parser can receive `allowed_styles`. If a planner invents a style that is not present in the
active prompt bank, validation fails before synthesis.

## Pause control

`pause_after_ms` is deliberately simple.

Normal recommendations:

- 0 ms: ordinary continuation
- 80-250 ms: slight rhetorical separation
- 250-700 ms: meaningful emotional transition
- >700 ms: unusual; use only when strongly justified

The parser accepts 0-10,000 ms as a safety-bounded range. The planner prompt asks models to stay
within natural shorter values unless context clearly calls for more.

This is separate from future native pause/prosody control inside the acoustic model.

## ChatGPT integration

Pocket TTS does not import an OpenAI SDK.

Instead:

```python
from pocket_tts.expressive_planner import build_planner_prompt, plan_from_json

prompt = build_planner_prompt(
    source_text,
    available_styles={"neutral", "happy", "sad", "calm", "excited"},
)

# Send `prompt` to ChatGPT or another capable model.
planner_json = call_your_model(prompt)

plan = plan_from_json(
    planner_json,
    source_text=source_text,
    allowed_styles={"neutral", "happy", "sad", "calm", "excited"},
)
```

The same parser works whether the JSON came from an API, local model, UI editor, test fixture, or
file.

## Prompt-injection boundary

The generated planner prompt explicitly instructs the semantic model to treat source text as
content rather than instructions.

That is still not considered sufficient security by itself.

The important enforcement is deterministic after model output:

1. strict JSON parsing
2. version check
3. additional fields rejected
4. numeric ranges checked
5. style names optionally allow-listed
6. source-text preservation verified

LLM output is treated as untrusted structured input.

## Segmentation principle

Use the **fewest segments that capture meaningful emotional changes**.

Bad:

```text
one segment / every few words
```

This creates unnecessary prompt switches and can damage continuity.

Also bad:

```text
one segment / entire emotional paragraph
```

when the paragraph changes from excitement to reflection to grief.

Preferred:

```text
semantic/prosodic transition
        ↓
segment boundary
```

The acoustic engine should not be forced to switch styles unless there is a useful reason.

## V1 non-goals

Planner Protocol v1 does not include:

- chain-of-thought or explanations
- word-level pitch curves
- phoneme durations
- arbitrary SSML
- per-word emotions
- model-specific hidden vectors
- acoustic latent interpolation
- a required cloud model
- a required local classifier

Those can be explored without breaking this contract if later research justifies them.

## Compatibility policy

Future compatible additions should be introduced through a new schema version when they change
semantics.

The synthesizer should fail clearly on unknown schema versions rather than silently guessing.

This keeps planner output reproducible and makes evaluation across versions possible.
