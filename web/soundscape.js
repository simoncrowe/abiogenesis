// Granular soundscape conductor (architecture/15-granular-ambient.md).
// Owns no audio: computes parameter targets from smoothed camera stats plus
// seeded drift walks, and nudges long-running scsynth voices via synth.js.
import {
  createVoice,
  isAudioBooted,
  getBufferPool,
  onAudioRecovered,
  setFilter,
  setReverb,
} from "./synth.js";

// ---- Shared helpers (carried over from the retired music.js) ----

export function clamp01(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

export function createRng(seed = 1234) {
  let s = (seed >>> 0) || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 2 ** 32;
  };
}

export function mapMeanToCutoff(mean01) {
  const t = clamp01(mean01);
  // Sonic Pi cutoff is roughly 0..130-ish; keep it musical.
  return 45 + t * 65;
}

export function mapMaxToReverbRoom(max01, volumeThreshold01 = 0.5) {
  // Use the meshing volume threshold (iso) as the scale: when the local max scalar
  // value approaches iso, the camera is near a surface we consider "solid".
  const maxT = clamp01(max01);
  const isoT = clamp01(volumeThreshold01);

  // 1 when max=0 (open space), 0 when max>=iso (near/inside solids).
  const denom = Math.max(1e-6, isoT);
  let room = clamp01((isoT - maxT) / denom);

  // Emphasize the near-surface region.
  room *= room;
  return 0.5 + (room * 2); // Scale reverb room size
}

// ---- Local helpers ----

function clamp(v, lo, hi) {
  const n = Number(v);
  if (!Number.isFinite(n)) return lo;
  return Math.max(lo, Math.min(hi, n));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function pick(rng, arr) {
  return arr[Math.max(0, Math.min(arr.length - 1, Math.floor(rng() * arr.length)))];
}

// Simple just-intonation ratios; jitter inside the grains supplies microtexture,
// this set supplies macro-level coherence. Weighted toward the most consonant.
const RATIOS = [1, 2, 3 / 2, 4 / 3, 5 / 4, 9 / 8, 16 / 9];
const RATIO_WEIGHTS = [4, 2, 3, 2, 2, 1, 1];
const RATIO_WEIGHT_TOTAL = RATIO_WEIGHTS.reduce((a, b) => a + b, 0);

function pickRatio(rng) {
  let r = rng() * RATIO_WEIGHT_TOTAL;
  for (let i = 0; i < RATIOS.length; i++) {
    r -= RATIO_WEIGHTS[i];
    if (r < 0) return RATIOS[i];
  }
  return RATIOS[0];
}

function foldFreqIntoRange(freq, lo, hi) {
  let f = clamp(freq, 1, 20000);
  while (f > hi) f /= 2;
  while (f < lo) f *= 2;
  return clamp(f, lo, hi);
}

// Mean-reverting seeded random walk: wanders around `home`, clamped, pulled back
// so parameters never pin at their limits. Stepped on a fixed timestep so a
// fixed seed reproduces the same evolution regardless of frame rate.
function createWalk(rng, { home, min, max, step, revert = 0.08 }) {
  let value = clamp(home, min, max);
  let homeValue = value;
  return {
    get value() {
      return value;
    },
    step(dt) {
      value += (homeValue - value) * revert * dt + (rng() * 2 - 1) * step * dt;
      value = clamp(value, min, max);
    },
    rehome(h) {
      homeValue = clamp(h, min, max);
    },
  };
}

const BASE_ANCHOR_HZ = 55; // A1
const ANCHOR_LO_HZ = 36;
const ANCHOR_HI_HZ = 88;

const XFADE_S = 6; // scene cross-fade (spec: >= 5s)
const WALK_DT_S = 0.5; // fixed drift timestep
const SEND_INTERVAL_S = 0.15; // <= 10 /n_set batches per second per voice
const SCENE_MIN_S = 30;
const SCENE_MAX_S = 90;

const CLOUD_NAMES = ["bufA", "bufB", "sin"];
const VOICE_NAMES = ["bufA", "bufB", "sin", "drone"];

// Per-param epsilons: skip /n_set churn for inaudible changes.
const SEND_EPS = {
  cutoff: 0.2,
  density: 0.15,
  graindur: 0.005,
  freq: 0.5,
  rate: 0.004,
  raterand: 0.05,
  freqspread: 0.05,
  pos: 0.004,
  posspray: 0.004,
  panwidth: 0.005,
  amp: 0.002,
};

export function createSoundscape({ seed = 1, getIso } = {}) {
  const rng = createRng(seed);

  let running = false;
  let timeS = 0; // internal clock (sum of update dts)
  let walkAccS = 0;
  let sendAccS = 0;
  let nextSceneAtS = Infinity;
  let sceneCount = 0;
  let anchorHz = BASE_ANCHOR_HZ;
  let lastStats = null;
  let lastTargets = null;

  const walks = {
    cutoff: createWalk(rng, { home: 0, min: -10, max: 10, step: 4 }),
    densA: createWalk(rng, { home: 0, min: -1, max: 1, step: 0.3 }),
    densB: createWalk(rng, { home: 0, min: -1, max: 1, step: 0.3 }),
    densSin: createWalk(rng, { home: 0, min: -1, max: 1, step: 0.3 }),
    grain: createWalk(rng, { home: 0, min: -0.6, max: 0.6, step: 0.2 }),
    posA: createWalk(rng, { home: 0.5, min: 0.05, max: 0.95, step: 0.07 }),
    posB: createWalk(rng, { home: 0.5, min: 0.05, max: 0.95, step: 0.07 }),
    panA: createWalk(rng, { home: 0.7, min: 0.25, max: 1, step: 0.1 }),
    panB: createWalk(rng, { home: 0.7, min: 0.25, max: 1, step: 0.1 }),
    panSin: createWalk(rng, { home: 0.8, min: 0.3, max: 1, step: 0.1 }),
    droneAmp: createWalk(rng, { home: 0, min: -0.06, max: 0.06, step: 0.03 }),
  };
  const walkList = Object.values(walks);

  // handle: return value of createVoice; null when not running.
  const voices = { bufA: null, bufB: null, sin: null, drone: null };
  // Per-scene tuning: source buffers and consonant ratios.
  const tuning = {
    bufA: { bufnum: 10, ratio: 1, octave: 1 },
    bufB: { bufnum: 11, ratio: 3 / 2, octave: 1 },
    sin: { ratio: 2, octave: 4 },
  };
  const lastSent = { bufA: {}, bufB: {}, sin: {}, drone: {}, fx: {} };

  function rehomeWalks() {
    walks.cutoff.rehome(lerp(-8, 8, rng()));
    walks.densA.rehome(lerp(-0.7, 0.7, rng()));
    walks.densB.rehome(lerp(-0.7, 0.7, rng()));
    walks.densSin.rehome(lerp(-0.7, 0.7, rng()));
    walks.grain.rehome(lerp(-0.4, 0.4, rng()));
    walks.posA.rehome(lerp(0.15, 0.85, rng()));
    walks.posB.rehome(lerp(0.15, 0.85, rng()));
    walks.panA.rehome(lerp(0.4, 0.95, rng()));
    walks.panB.rehome(lerp(0.4, 0.95, rng()));
    walks.panSin.rehome(lerp(0.5, 1, rng()));
    walks.droneAmp.rehome(lerp(-0.04, 0.04, rng()));
  }

  function drawTuning() {
    const pool = getBufferPool();
    const i = Math.floor(rng() * pool.length);
    let j = Math.floor(rng() * Math.max(1, pool.length - 1));
    if (j >= i) j = (j + 1) % pool.length;
    tuning.bufA = { bufnum: pool[i].bufnum, ratio: pickRatio(rng), octave: pick(rng, [0.5, 1, 1]) };
    tuning.bufB = { bufnum: pool[j].bufnum, ratio: pickRatio(rng), octave: pick(rng, [0.5, 1, 2]) };
    tuning.sin = { ratio: pickRatio(rng), octave: pick(rng, [2, 4, 4, 8]) };
  }

  function maybeMoveAnchor(probability) {
    if (rng() >= probability) return;
    let ratio = pickRatio(rng);
    if (rng() < 0.5) ratio = 1 / ratio;
    anchorHz = foldFreqIntoRange(anchorHz * ratio, ANCHOR_LO_HZ, ANCHOR_HI_HZ);
  }

  function scheduleNextScene() {
    nextSceneAtS = timeS + SCENE_MIN_S + rng() * (SCENE_MAX_S - SCENE_MIN_S);
  }

  function computeTargets(stats) {
    const iso = clamp01(typeof getIso === "function" ? getIso() : 0.5) || 0.5;
    const mean = clamp01(stats?.mean ?? 0.5);
    const max = clamp01(stats?.max ?? 0.5);
    const min = clamp01(stats?.min ?? 0);
    const range = clamp01(stats?.range ?? 0.25);
    const median = clamp01(stats?.median ?? mean);

    // 1 in open space, 0 at/near solid surfaces (relative to the iso threshold).
    const openness = clamp01((iso - max) / Math.max(1e-6, iso));
    // Skewed field shifts timbre between buffer clouds and the sine shimmer.
    const balance = clamp01(0.5 + (median - mean) * 2);

    const cutoff = clamp(mapMeanToCutoff(mean) + walks.cutoff.value, 30, 125);
    const room = clamp(mapMaxToReverbRoom(max, iso), 0, 3);
    const mix = clamp01(0.2 + 0.35 * clamp01((room - 0.5) / 2));

    const densityBase = lerp(3, 32, mean);
    const graindur = clamp(lerp(0.05, 0.4, openness) * 2 ** walks.grain.value, 0.02, 0.8);
    const raterand = clamp(lerp(0.3, 7, range), 0, 12);
    const freqspread = clamp(lerp(0.2, 5, range), 0, 12);
    const posspray = clamp(lerp(0.02, 0.35, range), 0, 0.5);

    const bufAmp = clamp(0.26 * (1.15 - 0.7 * balance), 0, 1);
    const sinAmp = clamp(0.2 * (0.3 + 0.9 * balance), 0, 1);
    const droneAmp = clamp(lerp(0.02, 0.3, min) + walks.droneAmp.value, 0, 0.4);

    const anchorRate = anchorHz / BASE_ANCHOR_HZ;

    const bufVoice = (t, posWalk, panWalk, densWalk) => ({
      buf: t.bufnum,
      density: clamp(densityBase * 2 ** densWalk.value, 0.5, 80),
      graindur,
      durrand: 0.5,
      rate: clamp(anchorRate * t.ratio * t.octave, 0.25, 4),
      raterand,
      pos: clamp(posWalk.value, 0, 1),
      posspray,
      posdrift: 0.3,
      panwidth: clamp(panWalk.value, 0, 1),
      amp: bufAmp,
    });

    return {
      fx: { cutoff, room, mix },
      bufA: bufVoice(tuning.bufA, walks.posA, walks.panA, walks.densA),
      bufB: bufVoice(tuning.bufB, walks.posB, walks.panB, walks.densB),
      sin: {
        freq: clamp(anchorHz * tuning.sin.ratio * tuning.sin.octave, 60, 4000),
        freqspread,
        density: clamp(densityBase * 0.6 * 2 ** walks.densSin.value, 0.5, 80),
        graindur: clamp(graindur * 0.75, 0.02, 0.8),
        durrand: 0.5,
        panwidth: clamp(walks.panSin.value, 0, 1),
        amp: sinAmp,
      },
      drone: {
        freq: anchorHz > 70 ? anchorHz / 2 : anchorHz,
        amp: droneAmp,
      },
    };
  }

  function sendVoice(name, params, force) {
    const handle = voices[name];
    if (!handle) return;
    const prev = lastSent[name];
    const changed = {};
    let any = false;
    for (const [k, v] of Object.entries(params)) {
      if (!Number.isFinite(v)) continue;
      const eps = SEND_EPS[k] ?? 0.003;
      if (force || !(k in prev) || Math.abs(v - prev[k]) > eps) {
        changed[k] = v;
        prev[k] = v;
        any = true;
      }
    }
    if (any) handle.set(changed);
  }

  function applyTargets(targets, force = false) {
    lastTargets = targets;
    const prev = lastSent.fx;
    if (force || !(("cutoff" in prev)) || Math.abs(targets.fx.cutoff - prev.cutoff) > SEND_EPS.cutoff) {
      prev.cutoff = targets.fx.cutoff;
      setFilter({ cutoff: targets.fx.cutoff, res: 0.45 });
    }
    const roomChanged = !("room" in prev) || Math.abs(targets.fx.room - prev.room) > 0.01;
    const mixChanged = !("mix" in prev) || Math.abs(targets.fx.mix - prev.mix) > 0.005;
    if (force || roomChanged || mixChanged) {
      prev.room = targets.fx.room;
      prev.mix = targets.fx.mix;
      setReverb({ room: targets.fx.room, mix: targets.fx.mix });
    }
    for (const name of VOICE_NAMES) sendVoice(name, targets[name], force);
  }

  function spawnCloudVoices(targets, fadeInS) {
    voices.bufA = createVoice("grain-cloud-buf", { ...targets.bufA, attack: fadeInS, release: XFADE_S });
    voices.bufB = createVoice("grain-cloud-buf", { ...targets.bufB, attack: fadeInS, release: XFADE_S });
    voices.sin = createVoice("grain-cloud-sin", { ...targets.sin, attack: fadeInS, release: XFADE_S });
    lastSent.bufA = { ...targets.bufA };
    lastSent.bufB = { ...targets.bufB };
    lastSent.sin = { ...targets.sin };
  }

  // Discrete re-roll: new buffers/ratios/walk homes, occasional anchor move.
  // Outgoing cloud voices release over XFADE_S while replacements attack.
  function changeScene() {
    sceneCount++;
    maybeMoveAnchor(0.35);
    drawTuning();
    rehomeWalks();

    for (const name of CLOUD_NAMES) {
      voices[name]?.free({ release: XFADE_S });
      voices[name] = null;
      lastSent[name] = {};
    }

    const targets = computeTargets(lastStats);
    spawnCloudVoices(targets, XFADE_S);
    // The drone persists across scenes; its freq glides (5s lag in the SynthDef).
    sendVoice("drone", targets.drone, true);
    lastTargets = targets;

    scheduleNextScene();
  }

  // Re-apply everything after an engine recover (synth.js has already
  // re-created the voice nodes with their last-known params).
  onAudioRecovered(() => {
    if (!running) return;
    for (const name of Object.keys(lastSent)) lastSent[name] = {};
    applyTargets(computeTargets(lastStats), true);
  });

  return {
    get running() {
      return running;
    },

    start() {
      if (running) return;
      if (!isAudioBooted()) throw new Error("audio not booted (call bootAudio() after a user gesture)");
      running = true;
      timeS = 0;
      walkAccS = 0;
      sendAccS = 0;
      sceneCount = 1;

      maybeMoveAnchor(0.5);
      drawTuning();
      rehomeWalks();

      const targets = computeTargets(lastStats);
      spawnCloudVoices(targets, 4);
      voices.drone = createVoice("sub-drone", { ...targets.drone, attack: 6, release: XFADE_S });
      lastSent.drone = { ...targets.drone };
      applyTargets(targets, true);

      scheduleNextScene();
    },

    stop() {
      if (!running) return;
      running = false;
      nextSceneAtS = Infinity;
      for (const name of VOICE_NAMES) {
        voices[name]?.free({ release: 4 });
        voices[name] = null;
        lastSent[name] = {};
      }
      lastSent.fx = {};
    },

    // Called every animation frame from main.js with smoothed 0..1 stats.
    update(dt, stats) {
      if (!running) return;
      if (stats) lastStats = stats;

      const d = clamp(dt, 0, 0.1);
      timeS += d;

      walkAccS += d;
      while (walkAccS >= WALK_DT_S) {
        walkAccS -= WALK_DT_S;
        for (const w of walkList) w.step(WALK_DT_S);
      }

      if (timeS >= nextSceneAtS) changeScene();

      sendAccS += d;
      if (sendAccS >= SEND_INTERVAL_S) {
        sendAccS = 0;
        applyTargets(computeTargets(lastStats));
      }
    },

    getState() {
      return {
        running,
        sceneCount,
        anchorHz,
        nextSceneInS: running ? Math.max(0, nextSceneAtS - timeS) : 0,
        fx: lastTargets ? { ...lastTargets.fx } : null,
      };
    },
  };
}
