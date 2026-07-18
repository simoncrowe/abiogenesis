// soundscape.js — the conductor.
//
// Replaces music.js. It owns no audio: it computes parameter *targets* from the
// smoothed camera stats and calls synth.js setters. Grain generation happens
// entirely inside scsynth (see audio/synthdefs/*.scd); the conductor only
// nudges control-rate params a few times per second.
//
// Three timescales (see architecture/15-granular-ambient.md):
//   1. Continuous — every frame, throttled to ~10 Hz per voice: smoothed stats
//      map to macro params.
//   2. Drift      — seeded, mean-reverting random walks add a slowly moving bias
//      so the texture evolves even when the camera is parked.
//   3. Scenes     — every 30-90 s: re-roll buffers, ratios, and drift homes, and
//      occasionally move the tonal anchor. Transitions cross-fade over >= 5 s.

import {
  createVoice,
  getBufferPool,
  onAudioRecovered,
  setFilter,
  setReverb,
} from "./synth.js";
import {
  clamp01,
  createRng,
  mapMaxToReverbRoom,
  mapMeanToCutoff,
} from "./music.js";

// ---- helpers ----------------------------------------------------------------

function lerp(a, b, t) {
  return a + (b - a) * clamp01(t);
}

function clamp(v, lo, hi) {
  const n = Number(v);
  if (!Number.isFinite(n)) return lo;
  return Math.max(lo, Math.min(hi, n));
}

function pickWeighted(rng, entries) {
  let total = 0;
  for (const [, w] of entries) total += w;
  let r = rng() * total;
  for (const [value, w] of entries) {
    r -= w;
    if (r <= 0) return value;
  }
  return entries[entries.length - 1][0];
}

// Simple just-intonation ratio set (plus octaves) keeps voices consonant with
// the tonal anchor without any scale/chord machinery.
const RATIOS = [
  [1 / 1, 3],
  [9 / 8, 1],
  [5 / 4, 2],
  [4 / 3, 2],
  [3 / 2, 3],
  [16 / 9, 1],
  [2 / 1, 2],
];
// Low, drone-friendly subset.
const LOW_RATIOS = [
  [1 / 1, 4],
  [9 / 8, 1],
  [4 / 3, 2],
  [3 / 2, 3],
];

function pickRatio(rng) {
  return pickWeighted(rng, RATIOS);
}

function foldInto(freq, lo, hi) {
  let f = freq;
  while (f > hi) f *= 0.5;
  while (f < lo) f *= 2;
  return f;
}

// Mean-reverting random walk producing a slowly moving bias. Clamped and pulled
// toward `home` so it never pins at its limits.
function makeWalk(rng, { sigma, pull, home = 0, min, max }) {
  return {
    value: home,
    home,
    sigma,
    pull,
    min,
    max,
    step() {
      const n = rng() * 2 - 1;
      this.value += n * this.sigma;
      this.value += (this.home - this.value) * this.pull;
      this.value = Math.max(this.min, Math.min(this.max, this.value));
      return this.value;
    },
  };
}

// ---- tuning constants -------------------------------------------------------

const APPLY_MS = 100; // <= 10 /n_set batches per second per voice
const XFADE_S = 6; // scene cross-fade / initial fade (>= 5 s required)
const SCENE_MIN_S = 30;
const SCENE_MAX_S = 90;

const ANCHOR_INIT_HZ = 55; // A1
const ANCHOR_LO = 45;
const ANCHOR_HI = 75;

const BASE_AMP = { buf: 0.16, sin: 0.12, drone: 0.24 };

// Stats used before the first camera-voxel-stats message arrives.
const NEUTRAL_STATS = { mean: 0.5, max: 0.5, min: 0.5, range: 0.0, median: 0.5 };

const BUF_SYNTHDEF = "grain-cloud-buf";
const SIN_SYNTHDEF = "grain-cloud-sin";
const DRONE_SYNTHDEF = "sub-drone";

// ---- conductor --------------------------------------------------------------

export function createSoundscape({ seed = 1 } = {}) {
  const rng = createRng(seed);

  let running = false;
  let lastApplyAt = 0;
  let nextSceneAt = 0;

  let anchorHz = ANCHOR_INIT_HZ;

  // Latest debug snapshot for the HUD.
  const debug = { cutoff: 0, room: 0, mix: 0, anchor: anchorHz, scene: 0 };
  let sceneCount = 0;

  function newWalks() {
    return {
      density: makeWalk(rng, { sigma: 1.5, pull: 0.03, min: -12, max: 12 }),
      dur: makeWalk(rng, { sigma: 0.02, pull: 0.04, min: -0.15, max: 0.15 }),
      pos: makeWalk(rng, { sigma: 0.03, pull: 0.05, min: -0.3, max: 0.3 }),
      pan: makeWalk(rng, { sigma: 0.02, pull: 0.05, min: -0.2, max: 0.2 }),
    };
  }

  // Re-home the drift walks so the "still" evolution wanders to new territory.
  function rehomeWalks(w) {
    w.density.home = rng() * 8 - 4;
    w.dur.home = rng() * 0.1 - 0.05;
    w.pos.home = rng() * 0.3 - 0.15;
    w.pan.home = rng() * 0.2 - 0.1;
  }

  function pool() {
    const p = getBufferPool();
    return p.length ? p : [0];
  }

  function pickBuf() {
    const p = pool();
    return p[Math.floor(rng() * p.length) % p.length];
  }

  // Voice descriptors. handle is the synth.js voice; the rest is scene state.
  const voices = {
    buf: [
      { handle: null, bufnum: 0, rate: 1, homePos: 0.5, walks: newWalks() },
      { handle: null, bufnum: 0, rate: 1, homePos: 0.5, walks: newWalks() },
    ],
    sin: { handle: null, freq: 220, walks: newWalks() },
    drone: { handle: null, freq: ANCHOR_INIT_HZ, walks: newWalks() },
  };

  // ---- pitch / buffer rolls -------------------------------------------------

  function rollBufPitch() {
    const ratio = pickRatio(rng);
    const oct = pickWeighted(rng, [[0.5, 1], [1, 2], [2, 1]]);
    return ratio * oct;
  }

  function rollSinFreq() {
    const ratio = pickRatio(rng);
    const octMul = pickWeighted(rng, [[2, 1], [4, 2], [8, 1]]);
    return anchorHz * ratio * octMul;
  }

  function rollDroneFreq() {
    const ratio = pickWeighted(rng, LOW_RATIOS);
    return foldInto(anchorHz * ratio, 30, 90);
  }

  // ---- macro-param computation ----------------------------------------------

  function derive(stats, iso) {
    const mean = clamp01(stats.mean);
    const max = clamp01(stats.max);
    const min = clamp01(stats.min);
    const range = clamp01(stats.range);
    const median = clamp01(stats.median);

    // 1 in open space, 0 near/inside solids (max approaches iso).
    const isoT = clamp01(iso);
    const openness = clamp01((isoT - max) / Math.max(1e-6, isoT));
    // median vs mean skew shifts timbre between buffer and sine clouds.
    const skew = Math.max(-1, Math.min(1, median - mean));
    const balance = clamp01(0.5 + skew); // higher -> favour the sine cloud

    return { mean, max, min, range, median, openness, balance };
  }

  function bufMacros(v, d) {
    const w = v.walks;
    return {
      density: clamp(lerp(2, 45, d.mean) + w.density.value, 0.5, 80),
      graindur: clamp(lerp(0.06, 0.5, d.openness) + w.dur.value, 0.02, 0.8),
      durrand: 0.35,
      raterand: clamp(lerp(0, 7, d.range), 0, 12),
      posspray: clamp(lerp(0.03, 0.4, d.range), 0, 0.5),
      pos: clamp(v.homePos + w.pos.value, 0, 1),
      posdrift: 0.3,
      panwidth: clamp(0.55 + w.pan.value, 0, 1),
      amp: BASE_AMP.buf * (0.5 + 0.5 * (1 - d.balance)),
    };
  }

  function sinMacros(v, d) {
    const w = v.walks;
    return {
      density: clamp(lerp(1.5, 30, d.mean) + w.density.value, 0.5, 80),
      graindur: clamp(lerp(0.1, 0.4, d.openness) + w.dur.value, 0.02, 0.8),
      durrand: 0.3,
      freqspread: clamp(lerp(0, 6, d.range), 0, 12),
      panwidth: clamp(0.7 + w.pan.value, 0, 1),
      amp: BASE_AMP.sin * (0.5 + 0.5 * d.balance),
    };
  }

  function droneMacros(_v, d) {
    return { amp: clamp(lerp(0.05, BASE_AMP.drone, d.min), 0, 1) };
  }

  // ---- spawning / scenes ----------------------------------------------------

  function spawnAll(fade) {
    const d = derive(NEUTRAL_STATS, 0.5);
    for (const v of voices.buf) {
      v.bufnum = pickBuf();
      v.rate = rollBufPitch();
      v.homePos = 0.2 + rng() * 0.6;
      const m = bufMacros(v, d);
      v.handle = createVoice(BUF_SYNTHDEF, {
        buf: v.bufnum,
        rate: v.rate,
        attack: fade,
        release: XFADE_S,
        ...m,
      });
    }

    voices.sin.freq = rollSinFreq();
    v_spawnSin(fade, d);

    voices.drone.freq = rollDroneFreq();
    voices.drone.handle = createVoice(DRONE_SYNTHDEF, {
      freq: voices.drone.freq,
      attack: fade,
      release: XFADE_S,
      ...droneMacros(voices.drone, d),
    });
  }

  function v_spawnSin(fade, d) {
    voices.sin.handle = createVoice(SIN_SYNTHDEF, {
      freq: voices.sin.freq,
      attack: fade,
      release: XFADE_S,
      ...sinMacros(voices.sin, d),
    });
  }

  function scene(stats, iso) {
    sceneCount++;
    const d = derive(stats, iso);

    // Occasionally drift the tonal anchor by a ratio from the same set.
    if (rng() < 0.35) {
      anchorHz = foldInto(anchorHz * pickRatio(rng), ANCHOR_LO, ANCHOR_HI);
    }

    // Cross-fade every voice: release the old node over XFADE_S while a fresh
    // node with new buffer/pitch fades in over XFADE_S. No hard cuts.
    for (const v of voices.buf) {
      rehomeWalks(v.walks);
      v.handle?.free({ release: XFADE_S });
      v.bufnum = pickBuf();
      v.rate = rollBufPitch();
      v.homePos = 0.2 + rng() * 0.6;
      v.handle = createVoice(BUF_SYNTHDEF, {
        buf: v.bufnum,
        rate: v.rate,
        attack: XFADE_S,
        release: XFADE_S,
        ...bufMacros(v, d),
      });
    }

    rehomeWalks(voices.sin.walks);
    voices.sin.handle?.free({ release: XFADE_S });
    voices.sin.freq = rollSinFreq();
    v_spawnSin(XFADE_S, d);

    rehomeWalks(voices.drone.walks);
    voices.drone.handle?.free({ release: XFADE_S });
    voices.drone.freq = rollDroneFreq();
    voices.drone.handle = createVoice(DRONE_SYNTHDEF, {
      freq: voices.drone.freq,
      attack: XFADE_S,
      release: XFADE_S,
      ...droneMacros(voices.drone, d),
    });

    debug.anchor = anchorHz;
    debug.scene = sceneCount;
  }

  function stepWalks() {
    for (const v of voices.buf) {
      v.walks.density.step();
      v.walks.dur.step();
      v.walks.pos.step();
      v.walks.pan.step();
    }
    voices.sin.walks.density.step();
    voices.sin.walks.dur.step();
    voices.sin.walks.pan.step();
  }

  function applyContinuous(stats, iso) {
    const d = derive(stats, iso);

    for (const v of voices.buf) {
      v.handle?.set(bufMacros(v, d));
    }
    voices.sin.handle?.set(sinMacros(voices.sin, d));
    voices.drone.handle?.set(droneMacros(voices.drone, d));

    // Global FX (carries over from doc 14): mean -> LPF cutoff,
    // max-vs-iso -> reverb room/mix.
    const cutoff = mapMeanToCutoff(d.mean);
    const room = mapMaxToReverbRoom(d.max, iso);
    const mix = 0.2 + 0.35 * room;
    setFilter({ cutoff, res: 0.45 });
    setReverb({ room, mix });

    debug.cutoff = cutoff;
    debug.room = room;
    debug.mix = mix;
  }

  // Re-apply current targets after an audio-context recovery. synth.js has
  // already rebuilt the voice nodes; force an apply on the next tick.
  onAudioRecovered(() => {
    if (running) lastApplyAt = 0;
  });

  return {
    start(stats = NEUTRAL_STATS, iso = 0.5) {
      if (running) return;
      running = true;
      lastApplyAt = 0;
      sceneCount = 0;
      anchorHz = ANCHOR_INIT_HZ;
      spawnAll(XFADE_S);
      applyContinuous(stats, iso);
      const now = performance.now();
      lastApplyAt = now;
      nextSceneAt = now + (SCENE_MIN_S + rng() * (SCENE_MAX_S - SCENE_MIN_S)) * 1000;
    },

    stop() {
      if (!running) return;
      running = false;
      for (const v of voices.buf) {
        v.handle?.free({ release: XFADE_S });
        v.handle = null;
      }
      voices.sin.handle?.free({ release: XFADE_S });
      voices.sin.handle = null;
      voices.drone.handle?.free({ release: XFADE_S });
      voices.drone.handle = null;
    },

    // Called every animation frame from main.js.
    update(_dt, tNow, stats, iso) {
      if (!running) return;

      if (lastApplyAt === 0 || tNow - lastApplyAt >= APPLY_MS) {
        stepWalks();
        applyContinuous(stats, iso);
        lastApplyAt = tNow;
      }

      if (tNow >= nextSceneAt) {
        scene(stats, iso);
        nextSceneAt = tNow + (SCENE_MIN_S + rng() * (SCENE_MAX_S - SCENE_MIN_S)) * 1000;
      }
    },

    getDebug() {
      return debug;
    },

    get running() {
      return running;
    },
  };
}
