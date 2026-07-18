# Granular Ambient Soundscape

This proposal supersedes most of
[14-audio-synthesis.md](14-audio-synthesis.md). The note/chord-based
approach there (stochastic chord progression FSM, arpeggiation, BPM
scheduling, subtractive SynthDef voices) produced something *like* an
evolving ambient soundscape by borrowing song machinery. This document
replaces that machinery with granular synthesis: continuously running
"grain cloud" voices whose parameters drift stochastically and are
steered by the simulation state around the camera.

Sections of doc 14 that remain in force unchanged:

- **Synth engine** — Supersonic boot sequence, autoplay-policy user
  gesture, `setup` event recovery, `postMessage` transport.
- **Camera neighbourhood voxel stats** — the 16^3 neighbourhood stats
  (mean, median, max, min, range), normalized 0..1, smoothed on the
  main thread.
- The global FX chain in `synth.js` (level -> lpf -> reverb on
  dedicated buses).

Everything in doc 14 under "Music composition" and "Algorithmic music
MVP" is superseded.

## Motivation

Chord progressions impose discrete harmonic events on a simulation
that is continuous and textural. Reaction-diffusion fields don't have
downbeats. A granular soundscape matches the medium better:

- Sound is a statistical cloud of many short grains, so it can evolve
  continuously along every axis (density, pitch, brightness, stereo
  spread) instead of stepping between chords.
- Randomness is intrinsic (grain timing, pitch jitter, buffer
  position spray) rather than bolted on via stochastic FSM
  transitions.
- The simulation stats can drive the *texture itself*, not just a
  filter cutoff on top of fixed notes.

## Data flow (unchanged shape, new consumer)

* `simulation (wasm)` -> `scalar lattice` -> `compute_worker.js`
* `compute_worker.js` -> `camera neighbourhood stats` -> `main.js`
* `main.js` -> `smoothed stats` -> `soundscape.js` (conductor) -> `synth.js`
* `synth.js` -> OSC over Supersonic -> scsynth (AudioWorklet)

`music.js` (progression FSM, arpeggiator, note scheduling) is retired
and replaced by `soundscape.js`.

## Where grains are generated: inside scsynth

There are two ways to do granular synthesis with this stack:

1. **Per-grain messages from the main thread** — schedule each grain
   as its own `/s_new`. Rejected: an ambient texture wants tens to
   hundreds of grains per second across voices; over `postMessage`
   transport that means constant message traffic, timing jitter tied
   to main-thread load (the same thread running WebGL), and GC
   pressure.
2. **Grain clouds inside long-running SynthDefs** — each voice is a
   single synth node that generates its own grain stream using
   SuperCollider granular UGens (`GrainBuf`, `GrainSin`/`GrainFM`)
   triggered internally by `Dust`/`Impulse`. The main thread only
   nudges control-rate parameters with `/n_set` a few times per
   second.

Option 2 is the design. It is robust to main-thread jank, keeps the
message rate trivial, and puts all sample-accurate timing where it
belongs (the audio thread). The trade-off is that grain behaviour must
be expressed as SynthDef *parameters* rather than arbitrary JS logic;
the parameter set below is chosen so the conductor still has full
macro control.

## Custom SynthDefs

The Sonic Pi synthdef collection currently loaded from unpkg has no
general-purpose granular cloud synth, so this feature introduces our
own compiled SynthDefs.

- SuperCollider sources live in `audio/synthdefs/*.scd` in this repo.
- They are compiled offline with sclang
  (`SynthDef(...).writeDefFile(...)`) and the resulting `.scsyndef`
  binaries are committed to `web/synthdefs/`.
- `synth.js` loads them via `loadSynthDef` with a path/URL into the
  local `web/synthdefs/` directory; the Sonic Pi bases stay configured
  for the FX synthdefs (`fx_level`, `fx_lpf`, `fx_reverb`) and for
  source samples.
- A short build note in `audio/README.md` MUST document the compile
  step so the binaries can be regenerated.

### `grain-cloud-buf` (buffer granulation)

The workhorse texture voice: granulates a loaded sample buffer with
`GrainBuf`. Control-rate parameters (all set via `/n_set`, all clamped
in the SynthDef with `.clip`):

| param        | meaning                                              | range        |
|--------------|------------------------------------------------------|--------------|
| `buf`        | source buffer index                                  | int          |
| `density`    | mean grains/sec (drives `Dust.kr`)                   | 0.5 .. 80    |
| `graindur`   | grain duration seconds                               | 0.02 .. 0.8  |
| `durrand`    | random spread on grain duration (multiplier)         | 0 .. 1       |
| `rate`       | centre playback rate (pitch), as a ratio             | 0.25 .. 4    |
| `raterand`   | random pitch spread in semitones, bipolar            | 0 .. 12      |
| `pos`        | centre read position in buffer                       | 0 .. 1       |
| `posspray`   | random spread around `pos`                           | 0 .. 0.5     |
| `posdrift`   | speed of internal `LFNoise2` drift on `pos`          | 0 .. 1       |
| `panwidth`   | stereo scatter of grains                             | 0 .. 1       |
| `amp`        | voice gain                                           | 0 .. 1       |
| `attack`, `release` | envelope for fade-in/out on scene changes     | seconds      |
| `out_bus`    | routed to `BUS_SYNTH` (existing FX chain input)      | int          |

Internal slow modulators (`LFNoise1`/`LFNoise2` at 0.02–0.2 Hz) wobble
`pos`, `density` and `panwidth` around their set values so the texture
is never static even when the conductor is idle.

### `grain-cloud-sin` (oscillator grains)

A "shimmer" voice using `GrainSin` (or `GrainFM`): no buffer, grains
are pitched sine/FM blips. Same density/duration/pan/amp parameters as
above, plus:

| param       | meaning                                            |
|-------------|----------------------------------------------------|
| `freq`      | centre frequency (Hz)                              |
| `freqspread`| bipolar random spread in semitones                 |
| `ratios`    | — not a param; consonant transposition is done by the conductor choosing `freq` (see pitch policy) |

### `sub-drone`

One near-static low sine/triangle drone (`freq`, `amp`, `attack`,
`release`) to anchor the low end. This is the only survivor of the
"drone" concept from doc 14, reduced from a chord progression to a
single slowly re-pitched tone.

## Source material

Granulation sources come from the already-configured Sonic Pi sample
collection (`sampleBaseURL`), starting with the `ambi_*` family —
these are long, tonal, and granulate well. `synth.js` loads a small
pool (3–5 buffers) at boot via Supersonic's sample loading. Scene
changes (below) select which buffer each `grain-cloud-buf` voice
reads. Custom recorded sources can be added to `web/samples/` later
without changing the architecture.

## Pitch policy (no chords, still consonant)

There is no scale, chord, or progression machinery. To keep the
result consonant rather than uniformly noisy:

- A single **tonal anchor** frequency exists at any time (e.g. A at
  55 Hz), owned by the conductor.
- Voice pitches (`rate` for buffer clouds, `freq` for sine clouds,
  `freq` for the drone) are the anchor multiplied by a ratio drawn
  from a small weighted set of simple just-intonation ratios:
  `{1, 2, 3/2, 4/3, 5/4, 9/8, 16/9}` and octave shifts.
- Per-grain pitch jitter (`raterand`/`freqspread`) supplies the
  "random" microtexture; the ratio set supplies coherence at the macro
  level.
- The anchor itself moves rarely (on some scene changes) and only by
  a ratio from the same set, giving a slow, keyless harmonic drift.

## The conductor: `soundscape.js`

Replaces `music.js`. It owns no audio; it computes parameter targets
and calls `synth.js` setters. Three timescales:

1. **Continuous (every animation frame, piggybacking on the existing
   `updateSmoothedStats` loop in `main.js`)** — smoothed camera stats
   are mapped to macro parameters and applied at a throttled rate
   (SHOULD be ≤ 10 `/n_set` batches per second per voice).
2. **Drift (internal, ~0.1–1 Hz)** — each macro parameter has a
   seeded random walk (reuse the xorshift `createRng` from `music.js`)
   producing a slowly moving *bias* that is summed with the
   stat-driven value, so the soundscape evolves even when the camera
   and simulation are still. Walks MUST be mean-reverting (clamp +
   pull toward a home value) so parameters never pin at their limits.
3. **Scenes (every 30–90 s, randomized)** — a discrete re-roll:
   choose per-voice source buffers from the pool, possibly move the
   tonal anchor, re-draw voice ratios, and choose new home values for
   the drift walks. Scene transitions MUST be smooth: outgoing
   settings cross-fade via voice `attack`/`release` envelopes or
   parameter ramps over ≥ 5 s — no hard cuts.

### Stat -> parameter mappings

All inputs are normalized 0..1 and all outputs MUST clamp (unchanged
requirement from doc 14). Initial mapping, expected to be tuned:

| stat            | drives                                            | intuition                                   |
|-----------------|---------------------------------------------------|---------------------------------------------|
| mean            | global LPF cutoff; grain `density` (up)           | denser medium -> brighter, busier cloud     |
| max vs. iso     | reverb `room`/`mix` (inverse, as in doc 14); `graindur` (near surface -> shorter grains) | open space -> huge wash; near solids -> dry and gritty |
| range           | `raterand`/`freqspread`, `posspray`               | heterogeneous neighbourhood -> wilder spread|
| median − mean   | balance between buf-cloud and sin-cloud voice amps| skewed field -> shifts timbre               |
| min             | sub-drone `amp` (up as min rises)                 | fully occupied space -> weighty low end     |

The existing `mapMaxToReverbRoom` iso-relative logic carries over;
`mapMeanToCutoff` carries over for the global LPF.

## Changes to `synth.js`

`synth.js` keeps its boot, FX-chain, and node-management code. Added:

- `loadLocalSynthDefs()` / buffer-pool loading during `bootAudio()`.
- `createVoice(synthdefName, params)` -> `{ nodeId, set(params),
  free({ release }) }`; voices are `/s_new`-ed into `GROUP_SYNTHS`
  with `out_bus: BUS_SYNTH` so the existing FX chain applies.
- The `setup` (recovery) handler MUST rebuild live voices as well as
  the FX chain; `soundscape.js` re-applies its current targets after
  recovery.
- `noteOn` and the one-shot note path remain for debugging but are no
  longer used by the soundscape.

## MVP — definition of done

The definition of done MUST be:

- Two `grain-cloud-buf` voices (different `ambi_*` buffers) plus one
  `grain-cloud-sin` voice and one `sub-drone`, all running
  continuously after the existing audio-boot user gesture.
- Grain generation happens entirely inside scsynth; the main thread
  sends only throttled `/n_set` batches.
- Smoothed camera stats audibly steer at least: LPF cutoff (mean),
  reverb room (max vs. iso), and grain density (mean).
- Seeded drift walks keep the texture evolving with the camera
  parked; a fixed seed reproduces the same evolution.
- Scene changes occur on a randomized 30–90 s timer with ≥ 5 s
  cross-fades and occasional tonal-anchor moves from the ratio set.
- All parameters clamp in both JS and the SynthDefs; no chord,
  progression, or BPM machinery remains in the audio path.
- `music.js` is deleted or reduced to shared helpers (`createRng`,
  `clamp01`, the two surviving stat-mapping functions).
