import init, { GrayScottParams, Simulation } from "../wasm/web/pkg/abiogenesis.js";

const CTRL_EPOCH = 0;
const CTRL_FRONT_INDEX = 1;

const TIMING_LAST_PUBLISH_MS = 0;
const TIMING_PERIOD_MS = 1;

const DEFAULT_DIMS = 192;
const DEFAULT_PERIOD_MS = 200;

let wasm;
let sim;

let ctrlI32;
let vViews;
let chunkMinViews;
let chunkMaxViews;

let timingI64;

let chunkMinWasm;
let chunkMaxWasm;

let dims = DEFAULT_DIMS;
let periodMs = DEFAULT_PERIOD_MS;

let currentSeed = 1337;
let simConfig = {
  strategyId: "gray_scott",
  params: { du: 0.16, dv: 0.08, feed: 0.037, kill: 0.06 },
  dt: 0.1,
  ticksPerSecond: 5,
  seeding: {
    type: "classic",
    noiseAmp: 0.01,
    sphereCount: 20,
    sphereRadius01: 0.05,
    u: 0.5,
    v: 0.25,
  },
};

let stepTimer = null;

// Exponential moving average of `Simulation.step(1)` duration.
// Used to start the next step early enough that publishing lands near `periodMs`.
let emaStepMs = 120;

let lastPublishMs = Date.now();

let totalSteps = 0;
let stepsWindow = 0;
let lastRateAt = 0;

function publishSnapshot(nowMs) {
  if (!sim || !ctrlI32 || !vViews || !chunkMinViews || !chunkMaxViews) return;

  const front = Atomics.load(ctrlI32, CTRL_FRONT_INDEX);
  const back = front ^ 1;

  const ptr = sim.v_ptr();
  const len = sim.v_len();
  const v = new Float32Array(wasm.memory.buffer, ptr, len);

  vViews[back].set(v);
  if (chunkMinWasm && chunkMaxWasm) {
    chunkMinViews[back].set(chunkMinWasm);
    chunkMaxViews[back].set(chunkMaxWasm);
  }

  // Publish: flip front, bump epoch.
  Atomics.store(ctrlI32, CTRL_FRONT_INDEX, back);
  Atomics.add(ctrlI32, CTRL_EPOCH, 1);

  lastPublishMs = nowMs;

  if (timingI64) {
    Atomics.store(timingI64, TIMING_LAST_PUBLISH_MS, BigInt(nowMs));
    Atomics.store(timingI64, TIMING_PERIOD_MS, BigInt(periodMs));
  }
}

function scheduleNextStep() {
  const targetPublishMs = lastPublishMs + periodMs;
  const startAtMs = targetPublishMs - emaStepMs;
  const delayMs = Math.max(0, startAtMs - Date.now());
  if (stepTimer) clearTimeout(stepTimer);
  stepTimer = setTimeout(stepOnceAndPublish, delayMs);
}

function stepOnceAndPublish() {
  if (!sim) return;

  const t0 = performance.now();
  sim.step(1);
  const nowMs = Date.now();
  publishSnapshot(nowMs);
  const dtMs = performance.now() - t0;

  totalSteps += 1;
  stepsWindow += 1;
  emaStepMs = emaStepMs * 0.8 + dtMs * 0.2;

  const now = performance.now();
  if (lastRateAt === 0) lastRateAt = now;
  const elapsed = now - lastRateAt;
  if (elapsed >= 1000) {
    const stepsPerSec = stepsWindow / (elapsed / 1000);
    stepsWindow = 0;
    lastRateAt = now;
    self.postMessage({ type: "sim_stats", stepsPerSec, totalSteps });
  }

  scheduleNextStep();
}

function mergeSimConfig(update) {
  if (!update || typeof update !== "object") return;

  if (typeof update.strategyId === "string") simConfig.strategyId = update.strategyId;

  if (typeof update.dt === "number") simConfig.dt = update.dt;
  if (typeof update.ticksPerSecond === "number") simConfig.ticksPerSecond = update.ticksPerSecond;

  if (update.params && typeof update.params === "object") {
    simConfig.params = {
      ...simConfig.params,
      ...update.params,
    };
  }

  if (update.seeding && typeof update.seeding === "object") {
    simConfig.seeding = {
      ...simConfig.seeding,
      ...update.seeding,
    };
  }
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function makeSeededRng(seed) {
  // Simple deterministic LCG (matches the style used in the Rust codebase).
  let state = (seed ^ 0x9e3779b9) >>> 0;
  return {
    nextFloat() {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 2 ** 32;
    },
  };
}

function seedClassic(
  sim,
  dims,
  seed,
  {
    noiseAmp = 0.01,
    sphereCount = 20,
    sphereRadius01 = 0.05,
    sphereRadiusJitter01 = 0.4,
    u = 0.5,
    v = 0.25,
  } = {},
) {
  const vPtr = sim.v_ptr();
  const vLen = sim.v_len();
  const vView = new Float32Array(wasm.memory.buffer, vPtr, vLen);

  // If the WASM build exposes U accessors, seed U + V together.
  const hasU = typeof sim.u_ptr === "function" && typeof sim.u_len === "function";
  const uView = hasU ? new Float32Array(wasm.memory.buffer, sim.u_ptr(), sim.u_len()) : null;

  // Base state: U=1, V=0 everywhere.
  vView.fill(0);
  if (uView) uView.fill(1);

  const rng = makeSeededRng(seed);
  const amp = Number(noiseAmp) || 0;

  if (amp > 0) {
    for (let i = 0; i < vView.length; i++) {
      const nu = (rng.nextFloat() * 2 - 1) * amp;
      const nv = (rng.nextFloat() * 2 - 1) * amp;

      if (uView) uView[i] = clamp01(uView[i] + nu);
      vView[i] = clamp01(vView[i] + nv);
    }
  }

  // Random sphere perturbations (with radius variation).
  const baseRadius = Math.max(1, Math.floor(dims * Math.max(0, Math.min(1, sphereRadius01))));
  const jitter = Math.max(0, Math.min(1, Number(sphereRadiusJitter01) || 0));

  const uu = clamp01(Number(u) || 0);
  const vv = clamp01(Number(v) || 0);
  const count = Math.max(0, Math.min(1000, Math.trunc(sphereCount)));

  for (let n = 0; n < count; n++) {
    const scale = 1 - jitter + rng.nextFloat() * (2 * jitter);
    const radius = Math.max(1, Math.round(baseRadius * scale));
    const r2 = radius * radius;

    const minC = radius;
    const maxC = dims - 1 - radius;

    // Keep the entire sphere inside the volume (no boundary intersections).
    // If the radius is too large for the grid, fall back to a centered sphere.
    const cx = maxC >= minC ? minC + Math.floor(rng.nextFloat() * (maxC - minC + 1)) : Math.floor(dims / 2);
    const cy = maxC >= minC ? minC + Math.floor(rng.nextFloat() * (maxC - minC + 1)) : Math.floor(dims / 2);
    const cz = maxC >= minC ? minC + Math.floor(rng.nextFloat() * (maxC - minC + 1)) : Math.floor(dims / 2);

    const x0 = cx - radius;
    const x1 = cx + radius;
    const y0 = cy - radius;
    const y1 = cy + radius;
    const z0 = cz - radius;
    const z1 = cz + radius;

    for (let z = z0; z <= z1; z++) {
      const dz = z - cz;
      const dz2 = dz * dz;
      for (let y = y0; y <= y1; y++) {
        const dy = y - cy;
        const dy2 = dy * dy;
        const base = dims * (y + dims * z);
        for (let x = x0; x <= x1; x++) {
          const dx = x - cx;
          if (dx * dx + dy2 + dz2 > r2) continue;
          const i = base + x;
          if (uView) uView[i] = uu;
          vView[i] = vv;
        }
      }
    }
  }
}

async function ensureWasm() {
  if (!wasm) wasm = await init();
}

async function restartSimulation() {
  await ensureWasm();

  if (stepTimer) {
    clearTimeout(stepTimer);
    stepTimer = null;
  }

  emaStepMs = 120;
  lastRateAt = 0;
  stepsWindow = 0;
  totalSteps = 0;

  if (simConfig.strategyId !== "gray_scott") {
    throw new Error(`unknown simulation strategy: ${String(simConfig.strategyId)}`);
  }

  const params = new GrayScottParams();
  params.set_du(Number(simConfig.params?.du ?? 0.16));
  params.set_dv(Number(simConfig.params?.dv ?? 0.08));
  params.set_feed(Number(simConfig.params?.feed ?? 0.037));
  params.set_kill(Number(simConfig.params?.kill ?? 0.06));

  sim = new Simulation(dims, dims, dims, currentSeed, params);
  sim.set_dt(Number(simConfig.dt ?? 0.1));

  const seeding = simConfig.seeding ?? {};
  if (seeding.type === "perlin") {
    sim.seed_perlin(
      Number(seeding.frequency ?? 6.0),
      Number(seeding.octaves ?? 4),
      Number(seeding.v_bias ?? 0.0),
      Number(seeding.v_amp ?? 1.0),
    );
  } else if (seeding.type === "classic") {
    seedClassic(sim, dims, currentSeed, seeding);
  } else {
    throw new Error(`unknown seeding type: ${String(seeding.type)}`);
  }

  const chunkLen = sim.chunk_v_len();
  chunkMinWasm = new Float32Array(wasm.memory.buffer, sim.chunk_v_min_ptr(), chunkLen);
  chunkMaxWasm = new Float32Array(wasm.memory.buffer, sim.chunk_v_max_ptr(), chunkLen);

  // Warm start: create two keyframes ASAP so meshing can interpolate.
  sim.step(1);
  publishSnapshot(Date.now());
  sim.step(1);
  publishSnapshot(Date.now());

  self.postMessage({ type: "sim_ready" });
  scheduleNextStep();
}

function clampTicksPerSecond(tps) {
  if (!Number.isFinite(tps)) return 5;
  return Math.max(1, Math.min(30, Math.trunc(tps)));
}

function applyTicksPerSecond(tps) {
  const ticksPerSecond = clampTicksPerSecond(tps);
  simConfig.ticksPerSecond = ticksPerSecond;
  periodMs = Math.max(1, Math.round(1000 / ticksPerSecond));

  if (timingI64) {
    Atomics.store(timingI64, TIMING_PERIOD_MS, BigInt(periodMs));
  }

  scheduleNextStep();
}

function applyLiveConfigUpdate(update) {
  if (!sim) return false;

  let changed = false;

  if (typeof update.dt === "number") {
    sim.set_dt(Number(simConfig.dt));
    changed = true;
  }

  if (typeof update.ticksPerSecond === "number") {
    applyTicksPerSecond(update.ticksPerSecond);
    changed = true;
  }

  return changed;
}

self.onmessage = (e) => {
  const msg = e.data;
  if (!msg || typeof msg !== "object") return;

  if (msg.type === "init") {
    const ctrl = msg.ctrl;
    const vSabs = msg.vSabs;
    const chunkMinSabs = msg.chunkMinSabs;
    const chunkMaxSabs = msg.chunkMaxSabs;
    const timing = msg.timing;

    if (Number.isFinite(msg.dims)) dims = Math.trunc(msg.dims);
    if (Number.isFinite(msg.periodMs)) periodMs = Math.trunc(msg.periodMs);

    ctrlI32 = new Int32Array(ctrl);
    vViews = [new Float32Array(vSabs[0]), new Float32Array(vSabs[1])];
    chunkMinViews = [new Float32Array(chunkMinSabs[0]), new Float32Array(chunkMinSabs[1])];
    chunkMaxViews = [new Float32Array(chunkMaxSabs[0]), new Float32Array(chunkMaxSabs[1])];
    timingI64 = timing ? new BigInt64Array(timing) : null;

    const nowMs = Date.now();
    lastPublishMs = nowMs;
    if (timingI64) {
      Atomics.store(timingI64, TIMING_LAST_PUBLISH_MS, BigInt(nowMs));
      Atomics.store(timingI64, TIMING_PERIOD_MS, BigInt(periodMs));
    }

    currentSeed = Number.isFinite(msg.seed) ? (Math.trunc(msg.seed) >>> 0) : 1337;
    if (msg.simConfig) mergeSimConfig(msg.simConfig);

    // Keep `periodMs` consistent with config (default: 5 ticks/sec).
    if (Number.isFinite(simConfig.ticksPerSecond)) {
      periodMs = Math.max(1, Math.round(1000 / clampTicksPerSecond(simConfig.ticksPerSecond)));
      if (timingI64) Atomics.store(timingI64, TIMING_PERIOD_MS, BigInt(periodMs));
    }

    void restartSimulation().catch((err) => {
      self.postMessage({ type: "error", message: String(err?.stack || err) });
    });
    return;
  }

  if (msg.type === "sim_config") {
    const update = msg.config && typeof msg.config === "object" ? msg.config : msg;

    const prevStrategyId = simConfig.strategyId;
    const prevParams = { ...simConfig.params };
    mergeSimConfig(update);

    const paramsChanged =
      prevParams.du !== simConfig.params.du ||
      prevParams.dv !== simConfig.params.dv ||
      prevParams.feed !== simConfig.params.feed ||
      prevParams.kill !== simConfig.params.kill ||
      update.seeding;

    if (prevStrategyId !== simConfig.strategyId || paramsChanged) {
      void restartSimulation().catch((err) => {
        self.postMessage({ type: "error", message: String(err?.stack || err) });
      });
      return;
    }

    applyLiveConfigUpdate(update);
  }
};

