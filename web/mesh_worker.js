import init, { ScalarFieldMesher } from "../wasm/web/pkg/abiogenesis.js";

const CTRL_EPOCH = 0;
const CTRL_FRONT_INDEX = 1;

const TIMING_LAST_PUBLISH_MS = 0;
const TIMING_PERIOD_MS = 1;

const DEFAULT_DIMS = 192;
const MESH_INTERVAL_MS = 16;

let wasm;
let mesher;
let mesherDims = null;
let targetDims = null;
let hasPostedReady = false;

let ctrlI32;
let vViews;
let chunkMinViews;
let chunkMaxViews;
let timingI64;

let camPos = [0, 0, 0.25];
let viewRadius = 0.35;
let iso = 0.5;
let color = [0.15, 0.65, 0.9, 0.9];

let lastEpoch = -1;
let lastMeshAt = 0;
let cameraDirty = true;

function copyFloat32(ptr, len) {
  return new Float32Array(wasm.memory.buffer, ptr, len).slice();
}

function copyU32(ptr, len) {
  return new Uint32Array(wasm.memory.buffer, ptr, len).slice();
}

function inferDimsFromVoxelBuffer(buf) {
  const n = buf?.length ?? 0;
  if (!Number.isFinite(n) || n <= 0) return null;
  const d = Math.round(Math.cbrt(n));
  if (d > 0 && d * d * d === n) return d;
  return null;
}

function ensureMesher() {
  if (!wasm) return;

  if (!targetDims && vViews?.[0]) {
    targetDims = inferDimsFromVoxelBuffer(vViews[0]);
  }

  const dims = targetDims ?? DEFAULT_DIMS;
  if (!mesher || mesherDims !== dims) {
    mesher = new ScalarFieldMesher(dims, dims, dims);
    mesherDims = dims;
    lastEpoch = -1;
    cameraDirty = true;
  }

  if (!hasPostedReady && mesher) {
    hasPostedReady = true;
    self.postMessage({ type: "mesh_ready" });
    loop();
  }
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

// Gradient-magnitude colormaps.
// The mesher outputs normals as (unnormalized) scalar-field gradients, so |normal|
// is a cheap proxy for local "thickness-ish" detail.
const colorRamps = {
  redToBlue(t) {
    const tt = Math.max(0, Math.min(1, t));
    return [lerp(1, 0, tt), 0, lerp(0, 1, tt)];
  },

  // Simple HSV-style rainbow. t=0 -> red, then yellow, green, cyan, blue, magenta, back to red.
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
let gradMagGain = 18.0;

function gradMagToT(gradMag) {
  const g = Math.max(0, gradMag);
  return 1 - Math.exp(-g * gradMagGain);
}

function recolorByGradientMagnitude(normals, colors) {
  const vertexCount = Math.floor(normals.length / 3);
  if (vertexCount <= 0) return;

  const alpha = colors.length >= 4 ? colors[3] : 1;

  for (let i = 0; i < vertexCount; i++) {
    const nx = normals[i * 3 + 0];
    const ny = normals[i * 3 + 1];
    const nz = normals[i * 3 + 2];
    const gradMag = Math.hypot(nx, ny, nz);
    const t = gradMagToT(gradMag);
    const [r, g, b] = ACTIVE_RAMP(t);

    colors[i * 4 + 0] = r;
    colors[i * 4 + 1] = g;
    colors[i * 4 + 2] = b;
    colors[i * 4 + 3] = alpha;
  }
}

function lerpFactor() {
  if (!timingI64) return 1;
  const lastPublishMs = Number(Atomics.load(timingI64, TIMING_LAST_PUBLISH_MS) || 0n);
  const periodMs = Number(Atomics.load(timingI64, TIMING_PERIOD_MS) || 0n);
  if (!Number.isFinite(lastPublishMs) || !Number.isFinite(periodMs) || periodMs <= 0) return 1;
  return Math.max(0, Math.min(1, (Date.now() - lastPublishMs) / periodMs));
}

function maybePushKeyframe() {
  if (!ctrlI32 || !vViews || !chunkMinViews || !chunkMaxViews || !mesher) return false;

  const epoch = Atomics.load(ctrlI32, CTRL_EPOCH);
  if (epoch === lastEpoch) return false;
  lastEpoch = epoch;

  const front = Atomics.load(ctrlI32, CTRL_FRONT_INDEX);
  mesher.push_keyframe_with_chunk_ranges(vViews[front], chunkMinViews[front], chunkMaxViews[front]);
  return true;
}

function buildMesh() {
  if (!mesher) return;

  const now = performance.now();
  if (lastMeshAt !== 0 && now - lastMeshAt < MESH_INTERVAL_MS) return;

  const keyframeChanged = maybePushKeyframe();
  const t = lerpFactor();

  // If nothing changed and we’re at (or past) the end of the lerp window, skip.
  if (!keyframeChanged && !cameraDirty && t >= 0.999) return;

  lastMeshAt = now;
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
      epoch: lastEpoch,
    },
    [positions.buffer, normals.buffer, colors.buffer, indices.buffer],
  );
}

function loop() {
  buildMesh();
  setTimeout(loop, 0);
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

self.onmessage = (e) => {
  const msg = e.data;
  if (!msg || typeof msg !== "object") return;

  if (msg.type === "init") {
    ctrlI32 = new Int32Array(msg.ctrl);

    const vSabs = msg.vSabs;
    const chunkMinSabs = msg.chunkMinSabs;
    const chunkMaxSabs = msg.chunkMaxSabs;

    vViews = [new Float32Array(vSabs[0]), new Float32Array(vSabs[1])];
    chunkMinViews = [new Float32Array(chunkMinSabs[0]), new Float32Array(chunkMinSabs[1])];
    chunkMaxViews = [new Float32Array(chunkMaxSabs[0]), new Float32Array(chunkMaxSabs[1])];
    timingI64 = msg.timing ? new BigInt64Array(msg.timing) : null;

    targetDims = Number.isFinite(msg.dims) ? Math.trunc(msg.dims) : inferDimsFromVoxelBuffer(vViews[0]);
    ensureMesher();
    return;
  }

  if (msg.type === "camera") {
    handleCamera(msg);
  }
};

(async () => {
  wasm = await init();
  ensureMesher();
})().catch((err) => {
  self.postMessage({ type: "error", message: String(err?.stack || err) });
});
