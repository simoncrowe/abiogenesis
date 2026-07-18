# Audio synthdefs

SuperCollider sources for the granular soundscape voices
(see [`architecture/15-granular-ambient.md`](../architecture/15-granular-ambient.md)).

- `synthdefs/*.scd` — SuperCollider sources for the custom voices:
  - `grain-cloud-buf` — buffer granulation (`GrainBuf`)
  - `grain-cloud-sin` — pitched sine-blip grains (`GrainSin`)
  - `sub-drone` — a single slowly re-pitched low tone
- `build.scd` — compiles all of the above to `.scsyndef` binaries.

The compiled binaries live in [`../web/synthdefs/`](../web/synthdefs/) and are
committed to the repo so the app needs no build step at runtime. `synth.js`
loads them from that directory via `loadSynthDef`. The Sonic Pi FX synthdefs
(`fx_level`, `fx_lpf`, `fx_reverb`) and the granulation source samples continue
to load from the Supersonic unpkg bases.

## Regenerating the binaries

Requires [SuperCollider](https://supercollider.github.io/) (`sclang`) on
`PATH`. From the repo root:

```sh
# Headless machines need an offscreen Qt platform.
QT_QPA_PLATFORM=offscreen sclang audio/build.scd
```

`build.scd` compiles every `synthdefs/*.scd` and writes the matching
`<name>.scsyndef` into `web/synthdefs/`. It does not require a running scsynth
server — `SynthDef(...).writeDefFile(...)` serialises the def to disk directly.
Commit the regenerated `.scsyndef` files.

## Parameters

Every voice clamps its control-rate parameters inside the SynthDef (mirroring
the JS-side clamps in `soundscape.js`), so an out-of-range `/n_set` can never
push a value past its documented limit. Each voice also runs internal slow
`LFNoise`/`SinOsc` modulators so the texture is never completely static, and
carries an `asr` envelope (`gate` + `doneAction: 2`) so the conductor can
cross-fade voices in and out. Voices write to `out_bus`, which the conductor
points at `BUS_SYNTH` so the global FX chain (level → lpf → reverb) applies.
