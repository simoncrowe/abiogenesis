import init, {
  GrayScottParams,
  Simulation,
  ScalarFieldMesher,
  StochasticRdmeParams,
  StochasticRdmeSimulation,
  CahnHilliardParams,
  CahnHilliardSimulation,
  ExcitableMediaParams,
  ExcitableMediaSimulation,
  ReplicatorMutatorParams,
  ReplicatorMutatorSimulation,
  init_thread_pool,
  rayon_num_threads,
} from "../wasm/web/pkg/abiogenesis.js";

const DEFAULT_DIMS = 192;
const DEFAULT_TICKS_PER_SECOND = 5;

// Run the simulation as fast as possible, but only publish snapshots at `periodMs`.
// Meshing interpolates between the most recent two published snapshots.
const STEP_TIME_SLICE_MS = 12;
const STEP_BATCH = 8;

const MESH_INTERVAL_MS = 16;

let wasm;

let sim = null;
let simKind = null; // "gray_scott" | "stochastic_rdme" | "cahn_hilliard" | "excitable_media" | "replicator_mutator"
let mesher = null;
let mesherDims = null;

let dims = DEFAULT_DIMS;
let ticksPerSecond = DEFAULT_TICKS_PER_SECOND;
let periodMs = Math.max(1, Math.round(1000 / ticksPerSecond));

let currentSeed = 1337;
let simConfig = {
  strategyId: "gray_scott",
  params: { du: 0.16, dv: 0.08, feed: 0.037, kill: 0.06 },
  dt: 0.1,
  ticksPerSecond: DEFAULT_TICKS_PER_SECOND,
  seeding: {
    type: "classic",
    noiseAmp: 0.01,
    sphereCount: 20,
    sphereRadius01: 0.05,
    u: 0.5,
    v: 0.25,
  },
};

let camPos = [0, 0, 0.25];
let viewRadius = 0.35;
let iso = 0.5;
let color = [0.15, 0.65, 0.9, 0.9];
let gradMagGain = 12.0;

let cameraDirty = true;

let lastMeshAt = 0;
let lastKeyframeEpoch = 0;

let meshIntervalMs = MESH_INTERVAL_MS;

let isRunning = false;
let loopTimer = null;

let lastPublishMs = Date.now();
let nextPublishMs = lastPublishMs + periodMs;

let totalSteps = 0;
let stepsWindow = 0;
let lastRateAt = 0;

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function toU32(v, fallback) {
  const n = Number(v);
  if (!Number.isFinite(n)) return (Math.trunc(fallback) >>> 0);
  return (Math.max(0, Math.min(2 ** 32 - 1, Math.trunc(n))) >>> 0);
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

  const hasU = typeof sim.u_ptr === "function" && typeof sim.u_len === "function";
  const uView = hasU ? new Float32Array(wasm.memory.buffer, sim.u_ptr(), sim.u_len()) : null;

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

function lerp(a, b, t) {
  return a + (b - a) * t;
}

// Gradient-magnitude colormap.
const colorRamps = {
  rainbow(t) {
    let h = Math.max(0, Math.min(1, t));
    h = (h * 6) % 6;
    const i = Math.floor(h);
    const f = h - i;

    const q = 1 - f;

    switch (i) {
      case 0:
        return [1, f, 0];
      case 1:
        return [q, 1, 0];
      case 2:
        return [0, 1, f];
      case 3:
        return [0, q, 1];
      case 4:
        return [f, 0, 1];
      default:
        return [1, 0, q];
    }
  },
};

const ACTIVE_RAMP = colorRamps.rainbow;

function gradMagToT(gradMag) {
  const g = Math.max(0, gradMag);
  return 1 - Math.exp(-g * gradMagGain);
}

function recolorByGradientMagnitude(normals, colors) {
  // Leave other sims as-is.
  // CH tends to produce much larger scalar gradients at the phase boundary, which
  // makes the ramp saturate. We apply a cheap per-strategy remap to keep the
  // control usable without changing the global behavior.
  const vertexCount = Math.floor(normals.length / 3);
  if (vertexCount <= 0) return;

  const alpha = colors.length >= 4 ? colors[3] : 1;

  const k = simKind === "cahn_hilliard" ? 0.15 : 1.0;

  for (let i = 0; i < vertexCount; i++) {
    const nx = normals[i * 3 + 0];
    const ny = normals[i * 3 + 1];
    const nz = normals[i * 3 + 2];
    const g = Math.hypot(nx, ny, nz) * k;
    const t = gradMagToT(g);
    const [r, gg, b] = ACTIVE_RAMP(t);

    colors[i * 4 + 0] = r;
    colors[i * 4 + 1] = gg;
    colors[i * 4 + 2] = b;
    colors[i * 4 + 3] = alpha;
  }
}

function stopLoop() {
  isRunning = false;
  if (loopTimer) clearTimeout(loopTimer);
  loopTimer = null;
}

function startLoop() {
  if (isRunning) return;
  isRunning = true;
  loop();
}

function copyFloat32(ptr, len) {
  return new Float32Array(wasm.memory.buffer, ptr, len).slice();
}

function copyU32(ptr, len) {
  return new Uint32Array(wasm.memory.buffer, ptr, len).slice();
}

function clampTicksPerSecond(tps) {
  if (!Number.isFinite(tps)) return DEFAULT_TICKS_PER_SECOND;
  return Math.max(1, Math.min(60, Math.trunc(tps)));
}

function applyTicksPerSecond(tps) {
  ticksPerSecond = clampTicksPerSecond(tps);
  periodMs = Math.max(1, Math.round(1000 / ticksPerSecond));

  const nowMs = Date.now();
  lastPublishMs = nowMs;
  nextPublishMs = nowMs + periodMs;
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

  if (typeof update.exportMode === "string" || update.exportMode === null) {
    simConfig.exportMode = update.exportMode;
  }
}

async function ensureWasm(threadCount) {
  if (!wasm) wasm = await init();

  const requestedThreads = Number.isFinite(threadCount) ? Math.trunc(threadCount) : 0;

  // wasm-bindgen-rayon requires a SharedArrayBuffer-backed WebAssembly memory.
  // In Chromium, this generally requires crossOriginIsolated (COOP/COEP headers).
  const hasSAB = typeof SharedArrayBuffer !== "undefined";
  const isIsolated = self.crossOriginIsolated === true;
  const hasSharedMemory = wasm?.memory?.buffer instanceof SharedArrayBuffer;
  const hasThreadInit = typeof init_thread_pool === "function";

  const canUseThreads = hasSAB && isIsolated && hasSharedMemory && hasThreadInit;

  // Always publish thread status so the UI can report it.
  if (!canUseThreads) {
    let reason = "unknown";
    if (!hasSAB) reason = "no_sab";
    else if (!isIsolated) reason = "not_isolated";
    else if (!hasSharedMemory) reason = "no_shared_memory";
    else if (!hasThreadInit) reason = "no_thread_init";

    self.postMessage({
      type: "thread_info",
      status: "unavailable",
      reason,
      requestedThreads,
      threads: 1,
      rayonThreads: null,
      hasSAB,
      isIsolated,
      hasSharedMemory,
      hasThreadInit,
    });

    if (requestedThreads > 0) {
      console.warn(
        "Wasm threads unavailable; continuing single-threaded.",
        { hasSAB, isIsolated, hasSharedMemory, hasThreadInit, requestedThreads, reason },
      );
    }

    return;
  }

  if (requestedThreads <= 0) {
    self.postMessage({
      type: "thread_info",
      status: "disabled",
      reason: "not_requested",
      requestedThreads,
      threads: 1,
      rayonThreads: null,
      hasSAB,
      isIsolated,
      hasSharedMemory,
      hasThreadInit,
    });
    return;
  }

  try {
    await init_thread_pool(requestedThreads);

    let rayonThreads = null;
    try {
      if (typeof rayon_num_threads === "function") {
        rayonThreads = rayon_num_threads();
      }
    } catch {
      // Ignore: just diagnostic.
    }

    self.postMessage({
      type: "thread_info",
      status: "enabled",
      reason: "ok",
      requestedThreads,
      threads: requestedThreads,
      rayonThreads,
      hasSAB,
      isIsolated,
      hasSharedMemory,
      hasThreadInit,
    });
  } catch (e) {
    console.error("init_thread_pool failed; continuing single-threaded", e);

    self.postMessage({
      type: "thread_info",
      status: "failed",
      reason: "init_failed",
      requestedThreads,
      threads: 1,
      rayonThreads: null,
      hasSAB,
      isIsolated,
      hasSharedMemory,
      hasThreadInit,
    });

    self.postMessage({
      type: "error",
      message:
        "Failed to start Wasm thread pool. Ensure the page is crossOriginIsolated (COOP/COEP headers) and rebuild with shared-memory enabled.",
    });
  }
}

function ensureMesher() {
  if (!wasm) return;
  if (!mesher || mesherDims !== dims) {
    mesher = new ScalarFieldMesher(dims, dims, dims);
    mesherDims = dims;
    lastKeyframeEpoch = 0;
    lastMeshAt = 0;
    meshIntervalMs = MESH_INTERVAL_MS;
    cameraDirty = true;
  }
}

function publishKeyframe(nowMs) {
  if (!sim || !mesher) return;

  if (typeof sim.recompute_chunk_ranges_from_v === "function") {
    sim.recompute_chunk_ranges_from_v();
  }

  if (simKind === "gray_scott") {
    mesher.push_keyframe_from_gray_scott(sim);
  } else if (simKind === "stochastic_rdme") {
    mesher.push_keyframe_from_stochastic_rdme(sim);
  } else if (simKind === "cahn_hilliard") {
    mesher.push_keyframe_from_cahn_hilliard(sim);
  } else if (simKind === "excitable_media") {
    mesher.push_keyframe_from_excitable_media(sim);
  } else if (simKind === "replicator_mutator") {
    mesher.push_keyframe_from_replicator_mutator(sim);
  }

  lastKeyframeEpoch += 1;
  lastPublishMs = nowMs;
  nextPublishMs = nowMs + periodMs;
}

function lerpFactor() {
  if (periodMs <= 0) return 1;
  return Math.max(0, Math.min(1, (Date.now() - lastPublishMs) / periodMs));
}

function maybeBuildMesh() {
  if (!mesher) return;

  const now = performance.now();
  if (lastMeshAt !== 0 && now - lastMeshAt < meshIntervalMs) return;

  const t = lerpFactor();

  cameraDirty = false;

  const t0 = performance.now();
  mesher.generate_mesh_visible_lerp(
    t,
    camPos[0],
    camPos[1],
    camPos[2],
    viewRadius,
    iso,
    color[0],
    color[1],
    color[2],
    color[3],
  );
  const meshMs = performance.now() - t0;

  // Throttle mesh generation based on how expensive it was.
  // (If meshing takes 200ms, trying again 16ms later just starves simulation stepping.)
  meshIntervalMs = Math.max(MESH_INTERVAL_MS, Math.min(1000, meshMs * 1.5));

  // Track the time *after* meshing so build cost counts towards throttling.
  lastMeshAt = performance.now();

  const positions = copyFloat32(mesher.mesh_positions_ptr(), mesher.mesh_positions_len());
  const normals = copyFloat32(mesher.mesh_normals_ptr(), mesher.mesh_normals_len());
  const colors = copyFloat32(mesher.mesh_colors_ptr(), mesher.mesh_colors_len());
  const indices = copyU32(mesher.mesh_indices_ptr(), mesher.mesh_indices_len());

  recolorByGradientMagnitude(normals, colors);

  self.postMessage(
    {
      type: "mesh",
      positions: positions.buffer,
      normals: normals.buffer,
      colors: colors.buffer,
      indices: indices.buffer,
      indexCount: indices.length,
      vertexCount: mesher.mesh_vertex_count(),
      meshMs,
      epoch: lastKeyframeEpoch,
    },
    [positions.buffer, normals.buffer, colors.buffer, indices.buffer],
  );
}

function updateStepsPerSecondStats() {
  const now = performance.now();
  if (lastRateAt === 0) lastRateAt = now;

  const elapsed = now - lastRateAt;
  if (elapsed < 1000) return;

  const stepsPerSec = stepsWindow / (elapsed / 1000);
  stepsWindow = 0;
  lastRateAt = now;
  self.postMessage({ type: "sim_stats", stepsPerSec, totalSteps });
}

function loop() {
  if (!isRunning || !sim) return;

  const sliceStart = performance.now();

  while (performance.now() - sliceStart < STEP_TIME_SLICE_MS) {
    const nowMs = Date.now();
    if (nowMs >= nextPublishMs) {
      publishKeyframe(nowMs);
      continue;
    }

    sim.step(STEP_BATCH);
    totalSteps += STEP_BATCH;
    stepsWindow += STEP_BATCH;
  }

  updateStepsPerSecondStats();
  maybeBuildMesh();

  loopTimer = setTimeout(loop, 0);
}

async function restartSimulation() {
  stopLoop();

  lastRateAt = 0;
  stepsWindow = 0;
  totalSteps = 0;

  ensureMesher();

  const strategyId = String(simConfig.strategyId || "gray_scott");
  simKind = strategyId;

  if (strategyId === "gray_scott") {
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
  } else if (strategyId === "stochastic_rdme") {
    const params = new StochasticRdmeParams();

    params.set_df(Number(simConfig.params?.df ?? 0.2));
    params.set_da(Number(simConfig.params?.da ?? 0.05));
    params.set_di(Number(simConfig.params?.di ?? 0.02));

    params.set_k1(Number(simConfig.params?.k1 ?? 0.002));
    params.set_k2(Number(simConfig.params?.k2 ?? 0.02));
    params.set_k3(Number(simConfig.params?.k3 ?? 0.001));

    params.set_feed_base(Number(simConfig.params?.feedBase ?? 2.0));
    params.set_feed_noise_amp(Number(simConfig.params?.feedNoiseAmp ?? 0.35));
    params.set_feed_noise_scale(toU32(simConfig.params?.feedNoiseScale, 8));

    params.set_d_a(Number(simConfig.params?.decayA ?? 0.01));
    params.set_d_i(Number(simConfig.params?.decayI ?? 0.005));
    params.set_d_f(Number(simConfig.params?.decayF ?? 0.0));

    params.set_eta_scale(Number(simConfig.params?.etaScale ?? 0.25));

    params.set_aliveness_alpha(Number(simConfig.params?.alivenessAlpha ?? 0.25));
    params.set_aliveness_gain(Number(simConfig.params?.alivenessGain ?? 0.05));

    // Optional stability knob (defaults to 1 in Rust).
    if (typeof params.set_substeps === "function") {
      params.set_substeps(toU32(simConfig.params?.substeps, 1));
    }

    sim = new StochasticRdmeSimulation(dims, dims, dims, currentSeed, params);
    sim.set_dt(Number(simConfig.dt ?? 0.05));

    const seeding = simConfig.seeding ?? {};
    if (!seeding.type || seeding.type === "perlin") {
      sim.seed_perlin(
        Number(seeding.frequency ?? 6.0),
        toU32(seeding.octaves, 4),
        toU32(seeding.baseF, 50),
        toU32(seeding.baseI, 0),
        Number(seeding.aBias ?? 0.0),
        Number(seeding.aAmp ?? 20.0),
      );
    } else if (seeding.type === "spheres") {
      sim.seed_spheres(
        Number(seeding.radius01 ?? 0.05),
        toU32(seeding.sphereCount, 20),
        toU32(seeding.baseF, 50),
        toU32(seeding.baseA, 0),
        toU32(seeding.baseI, 0),
        toU32(seeding.sphereF, 25),
        toU32(seeding.sphereA, 20),
        toU32(seeding.sphereI, 0),
        Number(seeding.aNoiseProb ?? 0.02),
      );
    } else {
      throw new Error(`unknown rdme seeding type: ${String(seeding.type)}`);
    }
  } else if (strategyId === "cahn_hilliard") {
    const params = new CahnHilliardParams();
 
    params.set_a(Number(simConfig.params?.a ?? 1.0));
    params.set_kappa(Number(simConfig.params?.kappa ?? 1.0));
    params.set_m(Number(simConfig.params?.m ?? 0.2));

    if (typeof params.set_substeps === "function") {
      params.set_substeps(toU32(simConfig.params?.substeps, 4));
    }
    if (typeof params.set_pass_mode === "function") {
      params.set_pass_mode(toU32(simConfig.params?.passMode, 2));
    }
    if (typeof params.set_approx_mode === "function") {
      params.set_approx_mode(toU32(simConfig.params?.approxMode, 0));
    }

    // Seeding controls.
    if (typeof params.set_phi_mean === "function") {
      params.set_phi_mean(Number(simConfig.params?.phiMean ?? 0.0));
    }
    if (typeof params.set_noise_amp === "function") {
      params.set_noise_amp(Number(simConfig.params?.noiseAmp ?? 0.01));
    }

    sim = new CahnHilliardSimulation(dims, dims, dims, currentSeed, params);
    sim.set_dt(Number(simConfig.dt ?? 0.002));

    const seeding = simConfig.seeding ?? {};

    // Deterministic spinodal noise init (optionally override params).
    const mean = Number(seeding.phiMean ?? simConfig.params?.phiMean ?? 0.0);
    const amp = Number(seeding.noiseAmp ?? simConfig.params?.noiseAmp ?? 0.02);
    if (typeof sim.seed_spinodal === "function") {
      sim.seed_spinodal(mean, amp);
    }

    // Export mode selection (separate from initialization).
    // For backwards compatibility, if exportMode isn't set, we pick something based on seeding.
    const modeId = simConfig.exportMode;

    if (modeId === "phase") {
      if (typeof sim.set_export_mode === "function") sim.set_export_mode(0);
      if (typeof sim.set_export_gain === "function") sim.set_export_gain(6.0);
    } else if (modeId === "phase_tanh") {
      if (typeof sim.set_export_mode === "function") sim.set_export_mode(2);
      if (typeof sim.set_export_gain === "function") sim.set_export_gain(0.6);
    } else if (modeId === "membranes") {
      if (typeof sim.set_export_mode === "function") sim.set_export_mode(1);
      if (typeof sim.set_export_gain === "function") sim.set_export_gain(4.0);
    } else if (modeId === "energy") {
      if (typeof sim.set_export_mode === "function") sim.set_export_mode(3);
      if (typeof sim.set_export_gain === "function") sim.set_export_gain(2.0);
    } else {
      // Fallback mapping based on seeding type.
      if (!seeding.type || seeding.type === "spinodal" || seeding.type === "droplets") {
        if (typeof sim.set_export_mode === "function") sim.set_export_mode(2);
        if (typeof sim.set_export_gain === "function") sim.set_export_gain(0.6);
      } else if (seeding.type === "membranes") {
        if (typeof sim.set_export_mode === "function") sim.set_export_mode(1);
        if (typeof sim.set_export_gain === "function") sim.set_export_gain(4.0);
      } else if (seeding.type === "energy") {
        if (typeof sim.set_export_mode === "function") sim.set_export_mode(3);
        if (typeof sim.set_export_gain === "function") sim.set_export_gain(2.0);
      } else {
        throw new Error(`unknown cahn-hilliard seeding type: ${String(seeding.type)}`);
      }
    }
  } else if (strategyId === "excitable_media") {
    const params = new ExcitableMediaParams();

    params.set_epsilon(Number(simConfig.params?.epsilon ?? 0.03));
    params.set_a(Number(simConfig.params?.a ?? 0.75));
    params.set_b(Number(simConfig.params?.b ?? 0.02));
    params.set_du(Number(simConfig.params?.du ?? 1.0));
    params.set_dv(Number(simConfig.params?.dv ?? 0.0));

    if (typeof params.set_substeps === "function") {
      params.set_substeps(toU32(simConfig.params?.substeps, 1));
    }

    // Optional: seeding presets can also tweak params/dt (we apply before constructing).
    const seeding = simConfig.seeding ?? {};
    if (seeding.preset === "A") {
      // A: self-sustaining turbulence (more nucleation)
      params.set_epsilon(Number(seeding.epsilon ?? simConfig.params?.epsilon ?? 0.03));
      params.set_a(Number(seeding.a ?? simConfig.params?.a ?? 0.85));
      params.set_b(Number(seeding.b ?? simConfig.params?.b ?? 0.01));
      params.set_du(Number(seeding.du ?? simConfig.params?.du ?? 1.0));
      params.set_dv(Number(seeding.dv ?? simConfig.params?.dv ?? 0.0));
    } else if (seeding.preset === "B") {
      // B: longer-lived scroll-wave-ish regime (fewer, larger sources)
      params.set_epsilon(Number(seeding.epsilon ?? simConfig.params?.epsilon ?? 0.04));
      params.set_a(Number(seeding.a ?? simConfig.params?.a ?? 0.80));
      params.set_b(Number(seeding.b ?? simConfig.params?.b ?? 0.02));
      params.set_du(Number(seeding.du ?? simConfig.params?.du ?? 0.9));
      params.set_dv(Number(seeding.dv ?? simConfig.params?.dv ?? 0.0));
    }

    sim = new ExcitableMediaSimulation(dims, dims, dims, currentSeed, params);
    sim.set_dt(Number(seeding.dt ?? simConfig.dt ?? 0.01));

    if (!seeding.type || seeding.type === "random") {
      sim.seed_random(Number(seeding.noiseAmp ?? 0.02), Number(seeding.excitedProb ?? 0.002));
    } else if (seeding.type === "sources") {
      sim.seed_sources(
        toU32(seeding.sourceCount, 8),
        Number(seeding.radius01 ?? 0.06),
        Number(seeding.uPeak ?? 1.0),
      );
    } else {
      throw new Error(`unknown excitable-media seeding type: ${String(seeding.type)}`);
    }
  } else if (strategyId === "replicator_mutator") {
    const params = new ReplicatorMutatorParams();

    // Core model params.
    if (typeof params.set_types === "function") params.set_types(toU32(simConfig.params?.types, 4));

    params.set_g_base(Number(simConfig.params?.gBase ?? 0.06));
    params.set_g_spread(Number(simConfig.params?.gSpread ?? 0.20));
    params.set_d_r(Number(simConfig.params?.dR ?? 0.03));

    params.set_feed_rate(Number(simConfig.params?.feedRate ?? 0.04));
    params.set_d_f(Number(simConfig.params?.dF ?? 0.01));

    params.set_mu(Number(simConfig.params?.mu ?? 0.003));

    params.set_d_r_diff(Number(simConfig.params?.diffR ?? 0.01));
    params.set_d_f_diff(Number(simConfig.params?.diffF ?? 0.20));

    if (typeof params.set_substeps === "function") {
      params.set_substeps(toU32(simConfig.params?.substeps, 2));
    }

    sim = new ReplicatorMutatorSimulation(dims, dims, dims, currentSeed, params);
    sim.set_dt(Number(simConfig.dt ?? 0.02));

    const seeding = simConfig.seeding ?? {};
    if (!seeding.type || seeding.type === "uniform") {
      sim.seed_uniform(Number(seeding.noiseAmp ?? 0.02), Number(seeding.rBase ?? 0.012), Number(seeding.fInit ?? 0.8));
    } else if (seeding.type === "regions") {
      sim.seed_regions(Number(seeding.noiseAmp ?? 0.01), Number(seeding.rPeak ?? 0.08), Number(seeding.fInit ?? 0.8));
    } else if (seeding.type === "gradient") {
      sim.seed_gradient_niches(
        Number(seeding.noiseAmp ?? 0.02),
        Number(seeding.rBase ?? 0.012),
        Number(seeding.fInit ?? 0.8),
        Number(seeding.feedBase ?? 0.04),
        Number(seeding.feedAmp ?? 1.0),
        toU32(seeding.axis, 0),
      );
    } else {
      throw new Error(`unknown replicator-mutator seeding type: ${String(seeding.type)}`);
    }

    // Optional live feed field update.
    if (seeding.type === "gradient") {
      sim.set_feed_gradient(Number(seeding.feedBase ?? 0.04), Number(seeding.feedAmp ?? 1.0), toU32(seeding.axis, 0));
    } else {
      sim.set_feed_uniform(Number(seeding.feedRate ?? simConfig.params?.feedRate ?? 0.04));
    }
  } else {
    throw new Error(`unknown simulation strategy: ${String(strategyId)}`);
  }

  applyTicksPerSecond(simConfig.ticksPerSecond);

  // Warm start: create two keyframes ASAP so meshing can interpolate.
  sim.step(1);
  publishKeyframe(Date.now());
  sim.step(1);
  publishKeyframe(Date.now());

  self.postMessage({ type: "sim_ready" });
  self.postMessage({ type: "mesh_ready" });

  startLoop();
}

function handleCamera(msg) {
  let changed = false;

  if (Array.isArray(msg.pos) && msg.pos.length === 3) {
    const next = [msg.pos[0], msg.pos[1], msg.pos[2]];
    if (next[0] !== camPos[0] || next[1] !== camPos[1] || next[2] !== camPos[2]) {
      camPos = next;
      changed = true;
    }
  }

  if (typeof msg.radius === "number" && msg.radius !== viewRadius) {
    viewRadius = msg.radius;
    changed = true;
  }

  if (typeof msg.iso === "number" && msg.iso !== iso) {
    iso = msg.iso;
    changed = true;
  }

  if (typeof msg.gradMagGain === "number" && msg.gradMagGain !== gradMagGain) {
    gradMagGain = msg.gradMagGain;
    changed = true;
  }

  if (Array.isArray(msg.color) && msg.color.length === 4) {
    const next = [msg.color[0], msg.color[1], msg.color[2], msg.color[3]];
    if (next[0] !== color[0] || next[1] !== color[1] || next[2] !== color[2] || next[3] !== color[3]) {
      color = next;
      changed = true;
    }
  }

  if (changed) cameraDirty = true;
}

function applyLiveConfigUpdate(update) {
  if (!sim) return;

  if (typeof update.dt === "number") {
    sim.set_dt(Number(simConfig.dt));
  }

  if (typeof update.ticksPerSecond === "number") {
    applyTicksPerSecond(update.ticksPerSecond);
  }
}

self.onmessage = (e) => {
  const msg = e.data;
  if (!msg || typeof msg !== "object") return;

  if (msg.type === "init") {
    dims = Number.isFinite(msg.dims) ? Math.trunc(msg.dims) : DEFAULT_DIMS;
    currentSeed = Number.isFinite(msg.seed) ? (Math.trunc(msg.seed) >>> 0) : 1337;

    if (msg.simConfig) mergeSimConfig(msg.simConfig);

    void ensureWasm(msg.threadCount)
      .then(() => restartSimulation())
      .catch((err) => {
        self.postMessage({ type: "error", message: String(err?.stack || err) });
      });

    return;
  }

  if (msg.type === "camera") {
    handleCamera(msg);
    return;
  }

  if (msg.type === "sim_config") {
    const update = msg.config && typeof msg.config === "object" ? msg.config : msg;

    const prevStrategyId = simConfig.strategyId;
    const prevSeedingType = simConfig.seeding?.type;
    const prevExportMode = simConfig.exportMode;

    mergeSimConfig(update);

    const strategyChanged = prevStrategyId !== simConfig.strategyId;
    const paramsChanged = !!(update.params && Object.keys(update.params).length > 0);
    const seedingChanged = !!update.seeding || prevSeedingType !== simConfig.seeding?.type;
    const exportChanged = Object.prototype.hasOwnProperty.call(update, "exportMode") || prevExportMode !== simConfig.exportMode;

    if (strategyChanged || paramsChanged || seedingChanged) {
      void restartSimulation().catch((err) => {
        self.postMessage({ type: "error", message: String(err?.stack || err) });
      });
      return;
    }

    // Allow changing export mode without restarting the simulation.
    if (exportChanged && simKind === "cahn_hilliard" && sim && typeof sim.set_export_mode === "function") {
      const modeId = simConfig.exportMode;
      if (modeId === "phase") {
        sim.set_export_mode(0);
      } else if (modeId === "phase_tanh") {
        sim.set_export_mode(2);
      } else if (modeId === "membranes") {
        sim.set_export_mode(1);
      } else if (modeId === "energy") {
        sim.set_export_mode(3);
      }
      // Ensure a fresh v field is available for meshing.
      if (typeof sim.recompute_chunk_ranges_from_v === "function") {
        sim.recompute_chunk_ranges_from_v();
      }
      lastMeshAt = 0;
    }

    applyLiveConfigUpdate(update);
  }
};
