# Expressive Prompt-Bank Recording Protocol v1

## Purpose

This protocol creates a **controlled same-speaker reference bank** for the first real expressive
Pocket TTS experiment.

The objective is not to collect a large emotional dataset. It is to create a small, reproducible
set of high-quality prompt recordings that let us test whether Pocket TTS's existing audio-prompt
conditioning is already strong enough for useful emotional delivery.

## Core experimental rule

For the first bank, use:

- one speaker
- one microphone
- one room
- one recording chain
- one script
- multiple emotional performances

Do not change the text between emotional styles in the first experiment.

Keeping the lexical content constant helps isolate the effect of prosody/style from the effect of
different words.

## Voice rights

Only use a voice that you are authorized to use.

Before compiling a bank, set:

```json
"rights_confirmed": true
```

The compiler refuses to export voice states while this is false.

This field means the person building the bank has confirmed that the voice is self-owned,
consented, or otherwise licensed for this use. It is not a substitute for keeping any real consent
or license record that applies to the source recording.

## Initial bank

The v1 example uses eight references:

| Style | Valence | Arousal | Dominance | Intensity | Performance target |
|---|---:|---:|---:|---:|---|
| neutral | 0.00 | 0.00 | 0.00 | 0.00 | ordinary natural speech |
| warm | +0.45 | -0.15 | +0.05 | 0.45 | friendly, close, reassuring |
| happy | +0.75 | +0.35 | +0.20 | 0.70 | genuine positive energy |
| excited | +0.85 | +0.85 | +0.35 | 0.95 | energetic but still intelligible |
| calm | +0.25 | -0.65 | +0.10 | 0.50 | slow, composed, grounded |
| sad | -0.75 | -0.50 | -0.35 | 0.75 | restrained sadness, not theatrical crying |
| angry | -0.80 | +0.80 | +0.75 | 0.95 | controlled anger, firm and forceful |
| fearful | -0.75 | +0.75 | -0.70 | 0.90 | alert, vulnerable, tense |

These coordinates are **starting prototypes**, not universal psychological truth.

They exist so the routing experiment is reproducible. We should tune them from listening,
automatic emotion measurements, and pairwise preference tests.

## Parallel script

Use the exact same words for every style:

> I thought it might happen today. I waited by the window for a moment, then took a breath and
> said, "All right, I'm ready." After that, everything started to change.

This script is intentionally semantically flexible. It can be performed neutrally, happily,
fearfully, angrily, sadly, calmly, or with excitement without requiring the actor to change the
words.

Do not improvise extra words.

## Target duration

Aim for roughly **8-12 seconds of useful speech** per selected take.

This is a practical first target because Pocket TTS's predefined voice prompts are around this
scale, while current models can also handle longer prompt prefixes.

Do not stretch or rush the performance just to hit an exact duration. Natural delivery matters
more than a stopwatch.

## Recording format

Recommended capture:

- mono WAV
- 24 kHz or 48 kHz
- uncompressed PCM or clean lossless WAV
- no MP3/AAC source
- no music
- no noise suppression artifacts
- no reverb effect
- no pitch correction
- no artificial time stretching

Pocket TTS resamples the prompt internally, so the important factor is clean source quality rather
than matching the model's output sample rate exactly.

## Physical consistency

Keep these as constant as possible across styles:

- microphone
- interface
- gain
- microphone distance
- mouth angle
- room
- chair/standing position
- recording software
- input processing

Emotion should be the main variable.

If angry is recorded 5 cm from the mic while sad is recorded 40 cm away, Pocket TTS may learn
microphone-distance differences along with emotion.

## Level and clipping

Do not chase maximum loudness.

Record enough headroom that excited/angry peaks do not clip.

A quieter clean take is preferable to a clipped emotional take.

Avoid changing gain between styles unless absolutely necessary.

## Start and ending

Start speaking naturally without a long artificial silence.

At the end:

1. finish the final word naturally
2. stop acting the emotion
3. leave a short clean pause before ending the recording

Pocket TTS already normalizes voice-prompt endings onto a pause before encoding. The recording
should still end cleanly rather than cutting through a final phoneme or breath.

## Number of takes

Record **three takes per style**.

Do not combine them.

For each style choose one final take using this priority:

1. naturalness
2. speaker identity consistency
3. emotional clarity
4. intelligibility
5. clean audio
6. target intensity

The most extreme performance is not automatically the best prompt.

## Performance directions

### Neutral

Speak as if explaining something ordinary to one person.

Avoid:

- announcer voice
- intentional happiness
- intentional sadness
- exaggerated rhythm

This is the anchor of the bank.

### Warm

Speak to someone you trust and want to reassure.

Target:

- gentle smile
- slightly softer energy
- natural closeness
- not sleepy

### Happy

The event is genuinely good.

Target:

- positive pitch movement
- light energy
- genuine smile
- avoid turning it into excitement

### Excited

Something important just succeeded.

Target:

- faster energy
- stronger pitch movement
- higher activation
- remain clearly intelligible

Do not shout directly into the microphone.

### Calm

You are composed and certain that things will be okay.

Target:

- lower activation
- steady rhythm
- confident breathing
- no whispering

### Sad

You are disappointed or grieving, but still trying to speak clearly.

Target:

- lower energy
- reduced pitch movement
- vulnerability
- restrained emotion

Avoid fake sobbing in v1. Non-verbal crying can become a confounder.

### Angry

You are genuinely upset but still controlling yourself.

Target:

- high activation
- firmer articulation
- stronger dominance
- controlled intensity

Avoid screaming. We need emotional prosody, not microphone overload.

### Fearful

You believe something may be wrong.

Target:

- high activation
- lower perceived control
- tension/uncertainty
- clear words

Do not add gasps or extra words in v1.

## File layout

Recommended local layout:

```text
my-prompt-bank/
├── prompt-bank.json
├── recordings/
│   ├── neutral.wav
│   ├── warm.wav
│   ├── happy.wav
│   ├── excited.wav
│   ├── calm.wav
│   ├── sad.wav
│   ├── angry.wav
│   └── fearful.wav
└── states/
```

WAV and safetensors files are already ignored by this repository's Git configuration.

The manifest can remain local too if speaker metadata should not be published.

## Create the manifest

Start from:

`docs/expressive-speech/prompt-bank.example.json`

Copy it beside the recordings and change:

- `bank_id`
- `speaker_id`
- `rights_confirmed`
- file names if needed

Keep the initial affect coordinates unchanged for the first controlled experiment.

## Compile the bank

For the default English model:

```powershell
uv run python scripts/export_expressive_prompt_bank.py .\my-prompt-bank\prompt-bank.json --language english
```

This will:

1. load Pocket TTS once
2. encode each WAV prompt
3. export a fast-loading `.safetensors` state for each style
4. SHA-256 fingerprint every source recording
5. write a compiled manifest
6. record which Pocket TTS model/config the states were compiled for

Default outputs:

```text
my-prompt-bank/
├── prompt-bank.json
├── prompt-bank.compiled.json
└── states/
    ├── neutral.safetensors
    ├── warm.safetensors
    └── ...
```

## Why compiled states are model-specific

A prompt state is not merely an audio embedding. It contains the state of Pocket TTS after the
prompt has conditioned the streaming model.

Therefore an exported state should not silently be reused after switching model weights.

The compiled manifest stores:

- `model_ref`
- config SHA-256 when the config exists locally

The loader can reject a bank compiled for a different model reference.

If the Pocket TTS config/weights change, re-export the bank from the original WAV files.

## First experiment discipline

Do not tune eight things at once.

For the first comparison:

1. use the same model
2. use one speaker
3. use the same parallel prompt text
4. keep the initial V/A/D/I prototypes
5. use the hand-authored evaluation trajectories
6. compare against ordinary neutral Pocket TTS

Only after hearing/measuring that baseline should we alter:

- coordinates
- reference scripts
- number of styles
- prompt duration
- boundary smoothing
- learned emotion adapters

## What success means

Phase 3A is successful if the bank produces **clearly different emotional delivery while preserving
speaker identity and intelligibility**, without meaningfully damaging Pocket TTS's CPU-first
properties.

It does not need to solve continuous emotional speech perfectly.

The purpose of Phase 3A is to measure how far the existing model can go before we spend complexity
on training a new control mechanism.
