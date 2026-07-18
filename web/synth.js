import { SuperSonic } from "https://unpkg.com/supersonic-scsynth@latest/dist/supersonic.js";

const DEFAULT_SYNTHDEF = "sonic-pi-dark_ambience";

const SUPERSONIC_VERSION = "latest";
const CORE_BASE = `https://unpkg.com/supersonic-scsynth-core@${SUPERSONIC_VERSION}/`;
const SYNTHDEFS_BASE = `https://unpkg.com/supersonic-scsynth-synthdefs@${SUPERSONIC_VERSION}/synthdefs/`;
const SAMPLES_BASE = `https://unpkg.com/supersonic-scsynth-samples@${SUPERSONIC_VERSION}/samples/`;

let supersonic = null;
let bootPromise = null;
let nextNodeId = 1000;

const GROUP_SYNTHS = 100;
const GROUP_FX = 101;

// Match Sonic Pi / Supersonic examples: synths -> level -> lpf -> reverb -> out.
// Keep each stage on its own bus to avoid accidental feedback/summing.
const BUS_SYNTH = 20;
const BUS_LEVEL_TO_LPF = 22;
const BUS_LPF_TO_REVERB = 24;

const NODE_LEVEL = 2000;
const NODE_LPF = 2001;
const NODE_REVERB = 2002;

const LEVEL_SYNTHDEF = "sonic-pi-fx_level";
const LPF_SYNTHDEF = "sonic-pi-fx_lpf";
const REVERB_SYNTHDEF = "sonic-pi-fx_reverb";

// Custom granular voices compiled from audio/synthdefs/*.scd (see audio/README.md).
// Loaded from our own web/synthdefs/ directory rather than the Sonic Pi unpkg base;
// loadSynthDef treats any string containing "/" as a URL and uses it verbatim.
const LOCAL_SYNTHDEF_DIR = new URL("./synthdefs/", import.meta.url).href;
const LOCAL_SYNTHDEFS = ["grain-cloud-buf", "grain-cloud-sin", "sub-drone"];

// Granulation source pool from the already-configured Sonic Pi sample collection.
// The `ambi_*` family is long, tonal, and granulates well. Buffer indices are
// assigned locally; nothing else in this app allocates buffers.
const SAMPLE_POOL = [
  "ambi_choir.flac",
  "ambi_drone.flac",
  "ambi_glass_hum.flac",
  "ambi_haunted_hum.flac",
  "ambi_lunar_land.flac",
];
const BUF_POOL_BASE = 10;

let fxReady = false;
let booted = false;

// { bufnum, name } for each loaded granulation source.
const bufferPool = [];

// Long-running granular voices: nodeId -> { synthdef, params }. Kept so the
// `setup` (recovery) handler can rebuild them after an audio-context restart.
const liveVoices = new Map();

let onRecoveredCb = null;

function ensureInstance() {
  if (supersonic) return supersonic;

  supersonic = new SuperSonic({
    // Core runtime assets (AudioWorklet + WASM)
    workerBaseURL: `${CORE_BASE}workers/`,
    wasmBaseURL: `${CORE_BASE}wasm/`,
    workletUrl: `${CORE_BASE}workers/scsynth_audio_worklet.js`,

    // Synth definition assets (Sonic Pi collection)
    synthdefBaseURL: SYNTHDEFS_BASE,

    // Optional samples (only required by sample-based synthdefs)
    sampleBaseURL: SAMPLES_BASE,

    mode: "postMessage",
  });

  // Recreate persistent node tree on boot/recover.
  supersonic.on("setup", async () => {
    fxReady = false;

    // Groups must exist even if synthdefs aren't loaded yet.
    supersonic.send("/g_new", GROUP_SYNTHS, 0, 0);
    supersonic.send("/g_new", GROUP_FX, 1, GROUP_SYNTHS);
    await supersonic.sync();

    // On first boot the synthdefs/samples aren't loaded yet; bootAudio() does
    // that. On recovery (context restart) `booted` is already true, so rebuild
    // the whole live tree: FX chain, sample buffers, then the granular voices.
    if (booted) {
      await ensureGlobalFxChain();
      await loadBufferPool();
      rebuildLiveVoices();
      // Let the conductor re-apply its current parameter targets.
      onRecoveredCb?.();
    }
  });

  return supersonic;
}

export function getSupersonic() {
  return ensureInstance();
}

export function isAudioBooted() {
  return Boolean(supersonic?.initialized);
}

export async function bootAudio({
  synthdefs = [DEFAULT_SYNTHDEF, LEVEL_SYNTHDEF, LPF_SYNTHDEF, REVERB_SYNTHDEF],
} = {}) {
  if (bootPromise) return bootPromise;

  bootPromise = (async () => {
    const s = ensureInstance();
    await s.init();
    await s.loadSynthDefs(synthdefs);
    await loadLocalSynthDefs();
    await s.sync();

    // Now that synthdefs are loaded, we can safely create the FX chain.
    await ensureGlobalFxChain();

    // Load the granulation source buffers used by grain-cloud-buf voices.
    await loadBufferPool();

    booted = true;
    return s;
  })();

  try {
    return await bootPromise;
  } catch (e) {
    bootPromise = null;
    throw e;
  }
}

export function noteOn({
  synthdef = DEFAULT_SYNTHDEF,
  note = 60,
  amp = 0.25,
  attack = 0.01,
  release = 1.5,
  pan,
  outBus,
} = {}) {
  const s = ensureInstance();
  if (!s.initialized) throw new Error("audio not booted (call bootAudio() after a user gesture)");

  const nodeId = nextNodeId;
  nextNodeId = (nextNodeId + 1) | 0;

  const resolvedOutBus = Number.isFinite(outBus)
    ? Math.trunc(outBus)
    : (fxReady ? BUS_SYNTH : 0);

  const args = [
    "/s_new",
    synthdef,
    nodeId,
    0,
    GROUP_SYNTHS,
    "out_bus",
    resolvedOutBus,
    "note",
    Math.trunc(note),
    "amp",
    Math.max(0, Math.min(2, Number(amp) || 0)),
    "attack",
    Math.max(0, Number(attack) || 0),
    "release",
    Math.max(0.01, Number(release) || 0.01),
  ];

  if (Number.isFinite(pan)) {
    args.push("pan", Math.max(-1, Math.min(1, Number(pan))));
  }

  s.send(...args);

  return nodeId;
}

export async function ensureGlobalFxChain() {
  const s = ensureInstance();
  if (!s.initialized) return false;
  if (!s.loadedSynthDefs?.has?.(REVERB_SYNTHDEF)) return false;
  if (!s.loadedSynthDefs?.has?.(LPF_SYNTHDEF)) return false;
  if (!s.loadedSynthDefs?.has?.(LEVEL_SYNTHDEF)) return false;

  // Recreate FX chain (idempotent: it overwrites our chosen node ids).
  s.send("/n_free", NODE_LEVEL);
  s.send("/n_free", NODE_LPF);
  s.send("/n_free", NODE_REVERB);

  // Gain staging first (prevents clipping with chords / reverb).
  s.send(
    "/s_new",
    LEVEL_SYNTHDEF,
    NODE_LEVEL,
    0,
    GROUP_FX,
    "in_bus",
    BUS_SYNTH,
    "out_bus",
    BUS_LEVEL_TO_LPF,
    "amp",
    0.7,
  );

  // Global low-pass before reverb.
  s.send(
    "/s_new",
    LPF_SYNTHDEF,
    NODE_LPF,
    3,
    NODE_LEVEL,
    "in_bus",
    BUS_LEVEL_TO_LPF,
    "out_bus",
    BUS_LPF_TO_REVERB,
    "cutoff",
    90.0,
    "res",
    0.5,
  );

  // Reverb after LPF.
  s.send(
    "/s_new",
    REVERB_SYNTHDEF,
    NODE_REVERB,
    3,
    NODE_LPF,
    "in_bus",
    BUS_LPF_TO_REVERB,
    "out_bus",
    0,
    "mix",
    0.25,
    "room",
    0.5,
  );

  await s.sync();
  fxReady = true;
  return true;
}

// Backwards-compatible name (older code calls this).
export const ensureGlobalReverb = ensureGlobalFxChain;

export function setReverb({ mix, room } = {}) {
  if (!fxReady) return;
  const params = {};
  if (Number.isFinite(mix)) params.mix = Math.max(0, Math.min(1, Number(mix)));
  if (Number.isFinite(room)) params.room = Math.max(0, Math.min(1, Number(room)));
  setNode(NODE_REVERB, params);
}

export function setFilter({ cutoff, res } = {}) {
  if (!fxReady) return;
  const params = {};
  if (Number.isFinite(cutoff)) params.cutoff = Number(cutoff);
  if (Number.isFinite(res)) params.res = Math.max(0, Math.min(1, Number(res)));
  setNode(NODE_LPF, params);
}

export function setMasterLevel({ amp } = {}) {
  if (!fxReady) return;
  if (!Number.isFinite(amp)) return;
  setNode(NODE_LEVEL, { amp: Math.max(0, Math.min(2, Number(amp))) });
}

export function getAudioRouting() {
  return {
    groups: { synths: GROUP_SYNTHS, fx: GROUP_FX },
    buses: { synth: BUS_SYNTH, levelToLpf: BUS_LEVEL_TO_LPF, lpfToReverb: BUS_LPF_TO_REVERB },
    nodes: { level: NODE_LEVEL, lpf: NODE_LPF, reverb: NODE_REVERB },
  };
}

export function setNode(nodeId, params = {}) {
  const s = ensureInstance();
  if (!s.initialized) return;
  if (!Number.isFinite(nodeId)) return;

  const flat = [];
  for (const [k, v] of Object.entries(params)) {
    if (typeof k !== "string" || k.length === 0) continue;
    if (!Number.isFinite(v)) continue;
    flat.push(k, v);
  }
  if (flat.length === 0) return;

  s.send("/n_set", Math.trunc(nodeId), ...flat);
}

export function freeNode(nodeId) {
  const s = ensureInstance();
  if (!s.initialized) return;
  if (!Number.isFinite(nodeId)) return;
  s.send("/n_free", Math.trunc(nodeId));
}

// --- Granular voices ---------------------------------------------------------

export async function loadLocalSynthDefs() {
  const s = ensureInstance();
  for (const name of LOCAL_SYNTHDEFS) {
    await s.loadSynthDef(`${LOCAL_SYNTHDEF_DIR}${name}.scsyndef`);
  }
}

export async function loadBufferPool() {
  const s = ensureInstance();
  if (!s.initialized) return [];

  bufferPool.length = 0;
  for (let i = 0; i < SAMPLE_POOL.length; i++) {
    bufferPool.push({ bufnum: BUF_POOL_BASE + i, name: SAMPLE_POOL[i] });
  }

  await Promise.all(bufferPool.map((b) => s.loadSample(b.bufnum, b.name)));
  await s.sync();
  return getBufferPool();
}

export function getBufferPool() {
  return bufferPool.map((b) => b.bufnum);
}

function flattenParams(params = {}) {
  const flat = [];
  for (const [k, v] of Object.entries(params)) {
    if (typeof k !== "string" || k.length === 0) continue;
    if (!Number.isFinite(v)) continue;
    flat.push(k, v);
  }
  return flat;
}

function spawnVoiceNode(synthdef, nodeId, params) {
  const s = ensureInstance();
  s.send("/s_new", synthdef, nodeId, 0, GROUP_SYNTHS, ...flattenParams(params));
}

function rebuildLiveVoices() {
  for (const [nodeId, rec] of liveVoices) {
    spawnVoiceNode(rec.synthdef, nodeId, rec.params);
  }
}

// Create a long-running granular voice. Returns a small handle; grain
// generation happens entirely inside scsynth, so the caller only nudges
// control-rate params via `set()` a few times per second.
export function createVoice(synthdefName, params = {}) {
  const s = ensureInstance();
  if (!s.initialized) throw new Error("audio not booted (call bootAudio() after a user gesture)");

  const nodeId = nextNodeId;
  nextNodeId = (nextNodeId + 1) | 0;

  const merged = { gate: 1, out_bus: BUS_SYNTH, ...params };
  spawnVoiceNode(synthdefName, nodeId, merged);
  liveVoices.set(nodeId, { synthdef: synthdefName, params: { ...merged } });

  return {
    nodeId,
    set(next = {}) {
      const rec = liveVoices.get(nodeId);
      if (rec) {
        for (const [k, v] of Object.entries(next)) {
          if (Number.isFinite(v)) rec.params[k] = v;
        }
      }
      setNode(nodeId, next);
    },
    free({ release } = {}) {
      liveVoices.delete(nodeId);
      if (!s.initialized) return;
      // gate 0 triggers the asr release; the synthdef's doneAction frees the node.
      if (Number.isFinite(release)) {
        s.send("/n_set", nodeId, "release", Math.max(0.01, Number(release)), "gate", 0);
      } else {
        s.send("/n_set", nodeId, "gate", 0);
      }
    },
  };
}

// Register a callback fired after an audio-context recovery has rebuilt the FX
// chain, buffers, and voices, so the conductor can re-apply its targets.
export function onAudioRecovered(cb) {
  onRecoveredCb = typeof cb === "function" ? cb : null;
}
