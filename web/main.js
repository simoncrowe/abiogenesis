const canvas = document.querySelector("#c");
const statsEl = document.querySelector("#stats");
const seedInput = document.querySelector("#seed");
const simStrategySelect = document.querySelector("#simStrategy");
const simInitSelect = document.querySelector("#simInit");
const simParamsEl = document.querySelector("#simParams");
const volumeThresholdInput = document.querySelector("#volumeThreshold");
const viewRadiusInput = document.querySelector("#viewRadius");
const gradMagGainInput = document.querySelector("#gradMagGain");
const restartBtn = document.querySelector("#restart");

function normalizeSeed(value) {
  const n = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(n)) return Date.now() >>> 0;
  return n >>> 0;
}

function resizeCanvasToDisplaySize(c) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const w = Math.floor(c.clientWidth * dpr);
  const h = Math.floor(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
    return true;
  }
  return false;
}

function mat4Identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

function mat4Mul(a, b) {
  // Column-major (OpenGL/WebGL) matrix multiply: out = a * b.
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function mat4Perspective(fovyRad, aspect, near, far) {
  const f = 1.0 / Math.tan(fovyRad / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (far + near) * nf,
    -1,
    0,
    0,
    (2 * far * near) * nf,
    0,
  ]);
}

function vec3Add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vec3Scale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function vec3Normalize(a) {
  const len = Math.hypot(a[0], a[1], a[2]);
  if (len < 1e-8) return [0, 0, 1];
  return [a[0] / len, a[1] / len, a[2] / len];
}

function vec3Dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vec3Cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(sh) || "shader compile failed");
  }
  return sh;
}

function createProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p) || "program link failed");
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return p;
}

const vsSource = `#version 300 es
precision highp float;

layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNor;
layout(location=2) in vec4 aCol;

uniform mat4 uViewProj;

out vec3 vNor;
out vec4 vCol;
out vec3 vPos;

void main() {
  vNor = aNor;
  vCol = aCol;
  vPos = aPos;
  gl_Position = uViewProj * vec4(aPos, 1.0);
}
`;

const fsSource = `#version 300 es
precision highp float;

in vec3 vNor;
in vec4 vCol;
in vec3 vPos;

uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform vec3 uFogColor;
uniform float uFogDensity;

out vec4 outColor;

void main() {
  vec3 n = normalize(vNor);
  float ndl = max(dot(n, normalize(uLightDir)), 0.0);
  vec3 base = vCol.rgb;
  vec3 lit = base * (0.25 + 0.75 * ndl);

  float d = length(vPos - uCamPos);
  float fog = 1.0 - exp(-uFogDensity * d * d);
  vec3 rgb = mix(lit, uFogColor, fog);

  outColor = vec4(rgb, vCol.a);
}
`;

class FlyCamera {
  constructor() {
    this.pos = [0, 0, 0.25];
    this.yaw = 0;
    this.pitch = 0;
    this.moveSpeed = 0.75;
  }

  viewMatrix() {
    const f = this.forward();

    // Build an orthonormal basis (right-handed).
    const worldUp = [0, 1, 0];
    let r = vec3Cross(f, worldUp);
    if (Math.hypot(r[0], r[1], r[2]) < 1e-6) {
      // Forward is too close to worldUp; pick a different up.
      r = vec3Cross(f, [1, 0, 0]);
    }
    r = vec3Normalize(r);
    const u = vec3Normalize(vec3Cross(r, f));

    const tx = -vec3Dot(r, this.pos);
    const ty = -vec3Dot(u, this.pos);
    const tz = vec3Dot(f, this.pos);

    // Column-major storage, but these are ROWS of the view matrix.
    // (This is easy to get wrong and causes "orbiting" when looking around.)
    return new Float32Array([
      r[0],
      u[0],
      -f[0],
      0,
      r[1],
      u[1],
      -f[1],
      0,
      r[2],
      u[2],
      -f[2],
      0,
      tx,
      ty,
      tz,
      1,
    ]);
  }

  forward() {
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    return vec3Normalize([sy * cp, -sp, -cy * cp]);
  }

  right() {
    // Used for movement; derived from current forward direction.
    const f = this.forward();
    const worldUp = [0, 1, 0];
    let r = vec3Cross(f, worldUp);
    if (Math.hypot(r[0], r[1], r[2]) < 1e-6) {
      r = vec3Cross(f, [1, 0, 0]);
    }
    return vec3Normalize(r);
  }
}

function createMeshGpu(gl) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  const posBuf = gl.createBuffer();
  const norBuf = gl.createBuffer();
  const colBuf = gl.createBuffer();
  const idxBuf = gl.createBuffer();

  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, norBuf);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);

  gl.bindVertexArray(null);

  return {
    vao,
    posBuf,
    norBuf,
    colBuf,
    idxBuf,
    indexCount: 0,
    vertexCount: 0,
  };
}

function uploadMeshFromBuffers(gl, gpuMesh, msg) {
  const pos = new Float32Array(msg.positions);
  const nor = new Float32Array(msg.normals);
  const col = new Float32Array(msg.colors);
  const idx = new Uint32Array(msg.indices);

  gl.bindVertexArray(gpuMesh.vao);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpuMesh.posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpuMesh.norBuf);
  gl.bufferData(gl.ARRAY_BUFFER, nor, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpuMesh.colBuf);
  gl.bufferData(gl.ARRAY_BUFFER, col, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gpuMesh.idxBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.DYNAMIC_DRAW);

  gl.bindVertexArray(null);

  gpuMesh.indexCount = idx.length;
  gpuMesh.vertexCount = msg.vertexCount || 0;
}


async function main() {
  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
  if (!gl) throw new Error("WebGL2 not supported");

  const program = createProgram(gl, vsSource, fsSource);
  const uViewProjLoc = gl.getUniformLocation(program, "uViewProj");
  const uLightDirLoc = gl.getUniformLocation(program, "uLightDir");
  const uCamPosLoc = gl.getUniformLocation(program, "uCamPos");
  const uFogColorLoc = gl.getUniformLocation(program, "uFogColor");
  const uFogDensityLoc = gl.getUniformLocation(program, "uFogDensity");

  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const cam = new FlyCamera();

  let viewRadius = 0.6;
  let volumeThreshold = 0.25;
  let gradMagGain = 12.0;
  const meshColor = [0.15, 0.65, 0.9, 0.75];

  const fogColor = [0.04, 0.06, 0.14];
  const fogDensity = 10.0;

  const gpuMesh = createMeshGpu(gl);

  const keys = new Set();
  window.addEventListener("keydown", (e) => {
    if (e.code === "Escape") {
      document.exitPointerLock?.();
      return;
    }
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }
    keys.add(e.code);
  });
  window.addEventListener("keyup", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }
    keys.delete(e.code);
  });

  canvas.addEventListener("click", async () => {
    if (document.pointerLockElement !== canvas) {
      await canvas.requestPointerLock();
    }
  });

  const lookSensitivity = 0.0015 * 0.75;
  window.addEventListener("mousemove", (e) => {
    if (document.pointerLockElement !== canvas) return;
    // Non-inverted: moving mouse right => look right, mouse up => look up.
    cam.yaw += e.movementX * lookSensitivity;
    cam.pitch += e.movementY * lookSensitivity;
    cam.pitch = Math.max(-1.55, Math.min(1.55, cam.pitch));
  });

  window.addEventListener("wheel", (e) => {
    const sign = Math.sign(e.deltaY);
    cam.moveSpeed = Math.max(0.1, Math.min(8.0, cam.moveSpeed * (sign > 0 ? 0.9 : 1.1)));
  });

  if (!globalThis.crossOriginIsolated || typeof SharedArrayBuffer === "undefined") {
    throw new Error(
      "SharedArrayBuffer requires crossOriginIsolated (COOP/COEP headers).",
    );
  }


  const simStrategies = {
    gray_scott: {
      id: "gray_scott",
      name: "Gray–Scott reaction–diffusion",
      params: [
        {
          key: "dims",
          path: ["dims"],
          label: "Grid size (dims)",
          min: 16,
          max: 256,
          step: 1,
          defaultValue: 128,
          requiresRestart: true,
        },
        {
          key: "du",
          path: ["params", "du"],
          label: "U diffusion rate (du)",
          min: 0,
          max: 1,
          step: 0.001,
          defaultValue: 0.16,
          requiresRestart: true,
        },
        {
          key: "dv",
          path: ["params", "dv"],
          label: "V diffusion rate (dv)",
          min: 0,
          max: 1,
          step: 0.001,
          defaultValue: 0.08,
          requiresRestart: true,
        },
        {
          key: "feed",
          path: ["params", "feed"],
          label: "Feed rate (U replenishment)",
          min: 0,
          max: 0.1,
          step: 0.0001,
          defaultValue: 0.0367,
          requiresRestart: true,
        },
        {
          key: "kill",
          path: ["params", "kill"],
          label: "Kill rate (V removal)",
          min: 0,
          max: 0.1,
          step: 0.0001,
          defaultValue: 0.0649,
          requiresRestart: true,
        },
        {
          key: "dt",
          path: ["dt"],
          label: "Time step (dt)",
          min: 0.001,
          max: 1,
          step: 0.001,
          defaultValue: 0.1,
          requiresRestart: false,
        },
        {
          key: "ticksPerSecond",
          path: ["ticksPerSecond"],
          label: "Snapshot ticks per second",
          min: 1,
          max: 60,
          step: 1,
          defaultValue: 30,
          requiresRestart: false,
        },
      ],
      seedings: [
        {
          id: "perlin",
          name: "Perlin noise",
          config: { type: "perlin", frequency: 5.0, octaves: 5, v_bias: 0.0, v_amp: 1.0 },
        },
        {
          id: "classic",
          name: "Random spheres + noise (long-lived)",
          config: {
            type: "classic",
            noiseAmp: 0.01,
            sphereCount: 20,
            sphereRadius01: 0.05,
            sphereRadiusJitter01: 0.4,
            u: 0.5,
            v: 0.25,
          },
        },
      ],
    },

    stochastic_rdme: {
      id: "stochastic_rdme",
      name: "Stochastic RDME (τ-leaping)",
      params: [
        { key: "dims", path: ["dims"], label: "Grid size (dims)", min: 16, max: 256, step: 1, defaultValue: 64, requiresRestart: true },
        { key: "df", path: ["params", "df"], label: "Food diffusion (Df)", min: 0, max: 1, step: 0.001, defaultValue: 0.2, requiresRestart: true },
        { key: "da", path: ["params", "da"], label: "Catalyst diffusion (Da)", min: 0, max: 1, step: 0.001, defaultValue: 0.05, requiresRestart: true },
        { key: "di", path: ["params", "di"], label: "Inhibitor diffusion (Di)", min: 0, max: 1, step: 0.001, defaultValue: 0.02, requiresRestart: true },

        { key: "k1", path: ["params", "k1"], label: "Autocatalysis rate (k1)", min: 0, max: 0.01, step: 0.0001, defaultValue: 0.002, requiresRestart: true },
        { key: "k2", path: ["params", "k2"], label: "Inhibitor production (k2)", min: 0, max: 0.2, step: 0.001, defaultValue: 0.02, requiresRestart: true },
        { key: "k3", path: ["params", "k3"], label: "Inhibition rate (k3)", min: 0, max: 0.01, step: 0.0001, defaultValue: 0.001, requiresRestart: true },

        { key: "feedBase", path: ["params", "feedBase"], label: "Feed base rate", min: 0, max: 20, step: 0.1, defaultValue: 2.0, requiresRestart: true },
        { key: "feedNoiseAmp", path: ["params", "feedNoiseAmp"], label: "Feed noise amplitude", min: 0, max: 2, step: 0.01, defaultValue: 0.35, requiresRestart: true },
        { key: "feedNoiseScale", path: ["params", "feedNoiseScale"], label: "Feed noise scale (voxels)", min: 1, max: 64, step: 1, defaultValue: 8, requiresRestart: true },

        { key: "decayA", path: ["params", "decayA"], label: "Catalyst decay (dA)", min: 0, max: 0.2, step: 0.001, defaultValue: 0.01, requiresRestart: true },
        { key: "decayI", path: ["params", "decayI"], label: "Inhibitor decay (dI)", min: 0, max: 0.2, step: 0.001, defaultValue: 0.005, requiresRestart: true },
        { key: "decayF", path: ["params", "decayF"], label: "Food decay (dF)", min: 0, max: 0.2, step: 0.001, defaultValue: 0.0, requiresRestart: true },

        { key: "alivenessAlpha", path: ["params", "alivenessAlpha"], label: "Aliveness inhibitor weight (α)", min: 0, max: 5, step: 0.01, defaultValue: 0.25, requiresRestart: true },
        { key: "alivenessGain", path: ["params", "alivenessGain"], label: "Aliveness gain", min: 0.0001, max: 1, step: 0.001, defaultValue: 0.05, requiresRestart: true },

        { key: "dt", path: ["dt"], label: "Time step (dt)", min: 0.001, max: 0.2, step: 0.001, defaultValue: 0.05, requiresRestart: false },
        { key: "ticksPerSecond", path: ["ticksPerSecond"], label: "Snapshot ticks per second", min: 1, max: 60, step: 1, defaultValue: 30, requiresRestart: false },
      ],
      seedings: [
        {
          id: "spheres",
          name: "Catalyst spheres + noise",
          config: {
            type: "spheres",
            radius01: 0.05,
            sphereCount: 20,
            baseF: 50,
            baseA: 0,
            baseI: 0,
            sphereF: 25,
            sphereA: 20,
            sphereI: 0,
            aNoiseProb: 0.02,
          },
        },
      ],
    },
  };

  const simConfig = {
    strategyId: "gray_scott",
    dims: 128,
    params: {},
    dt: 0.1,
    ticksPerSecond: 5,
    seeding: simStrategies.gray_scott.seedings[0].config,
  };

  resetSimConfigForStrategy(simConfig.strategyId);


  let simWorker = null;
  let meshWorker = null;

  let lastMeshMs = 0;
  let lastMeshEpoch = 0;
  let lastSimStepsPerSec = 0;
  let lastSimTotalSteps = 0;
  let meshReady = false;

  function stopWorkers() {
    simWorker?.terminate();
    meshWorker?.terminate();
    simWorker = null;
    meshWorker = null;
    meshReady = false;
  }

  function clampDims(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return 128;
    return Math.max(16, Math.min(256, Math.trunc(n)));
  }

  function startWorkers(seed) {
    stopWorkers();

    const dims = clampDims(simConfig.dims);
    const n = dims * dims * dims;

    // Reset UI + mesh.
    statsEl.textContent = "";
    gpuMesh.indexCount = 0;
    gpuMesh.vertexCount = 0;
    lastMeshMs = 0;
    lastMeshEpoch = 0;
    lastSimStepsPerSec = 0;
    lastSimTotalSteps = 0;
    lastCamSendAt = 0;

    const vSabs = [
      new SharedArrayBuffer(n * Float32Array.BYTES_PER_ELEMENT),
      new SharedArrayBuffer(n * Float32Array.BYTES_PER_ELEMENT),
    ];

    // Must match `CHUNK` in `wasm/src/sim.rs`.
    const chunkSize = 16;
    const cubes = dims - 1;
    const chunkN = Math.ceil(cubes / chunkSize);
    const chunkTotal = chunkN * chunkN * chunkN;

    const chunkMinSabs = [
      new SharedArrayBuffer(chunkTotal * Float32Array.BYTES_PER_ELEMENT),
      new SharedArrayBuffer(chunkTotal * Float32Array.BYTES_PER_ELEMENT),
    ];
    const chunkMaxSabs = [
      new SharedArrayBuffer(chunkTotal * Float32Array.BYTES_PER_ELEMENT),
      new SharedArrayBuffer(chunkTotal * Float32Array.BYTES_PER_ELEMENT),
    ];

    // Shared control layout:
    // ctrlI32[0] = epoch
    // ctrlI32[1] = frontIndex (0/1)
    const ctrl = new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT * 2);
    const ctrlI32 = new Int32Array(ctrl);
    ctrlI32[0] = 0;
    ctrlI32[1] = 0;

    // timingI64[0] = last publish Date.now() (ms)
    // timingI64[1] = target period (ms)
    const timing = new SharedArrayBuffer(BigInt64Array.BYTES_PER_ELEMENT * 2);
    const ticksPerSecond = Math.max(1, Math.min(60, Math.trunc(simConfig.ticksPerSecond ?? 5)));
    const periodMs = Math.max(1, Math.round(1000 / ticksPerSecond));

    const timingI64 = new BigInt64Array(timing);
    timingI64[0] = BigInt(Date.now());
    timingI64[1] = BigInt(periodMs);

    simWorker = new Worker(new URL("./sim_worker.js", import.meta.url), { type: "module" });
    meshWorker = new Worker(new URL("./mesh_worker.js", import.meta.url), { type: "module" });

    simWorker.postMessage({
      type: "init",
      ctrl,
      vSabs,
      chunkMinSabs,
      chunkMaxSabs,
      timing,
      seed,
      dims,
      periodMs,
      simConfig,
    });
    meshWorker.postMessage({ type: "init", ctrl, vSabs, chunkMinSabs, chunkMaxSabs, timing, dims });

    simWorker.onmessage = (e) => {
      const msg = e.data;
      if (!msg || typeof msg !== "object") return;

      if (msg.type === "sim_ready") {
        return;
      }

      if (msg.type === "sim_stats") {
        lastSimStepsPerSec = msg.stepsPerSec || 0;
        lastSimTotalSteps = msg.totalSteps || 0;
        return;
      }

      if (msg.type === "error") {
        console.error(msg.message);
        statsEl.textContent = String(msg.message);
      }
    };

    meshWorker.onmessage = (e) => {
      const msg = e.data;
      if (!msg || typeof msg !== "object") return;

      if (msg.type === "mesh_ready") {
        meshReady = true;
        return;
      }

      if (msg.type === "mesh") {
        uploadMeshFromBuffers(gl, gpuMesh, msg);
        lastMeshMs = msg.meshMs || 0;
        lastMeshEpoch = msg.epoch || 0;
        return;
      }

      if (msg.type === "error") {
        console.error(msg.message);
        statsEl.textContent = String(msg.message);
      }
    };
  }

  function restartFromUi() {
    const seed = normalizeSeed(seedInput?.value);
    if (seedInput) seedInput.value = String(seed);
    startWorkers(seed);
  }

  function parseClampedFloat(value, fallback, min, max) {
    const n = Number.parseFloat(String(value ?? ""));
    if (!Number.isFinite(n)) return fallback;
    return Math.max(min, Math.min(max, n));
  }

  const urlSeed = new URLSearchParams(globalThis.location?.search ?? "").get("seed");
  const initialSeed = normalizeSeed(urlSeed ?? seedInput?.value ?? 1337);
  if (seedInput) seedInput.value = String(initialSeed);

  function decimalsForStep(step) {
    const s = String(step ?? "");
    const idx = s.indexOf(".");
    if (idx === -1) return 0;
    return s.length - idx - 1;
  }

  function setNumberInputValue(input, value, step) {
    const decimals = decimalsForStep(step);
    input.value = Number(value).toFixed(decimals);
  }

  function resetSimConfigForStrategy(strategyId) {
    const strategy = simStrategies[strategyId];
    if (!strategy) throw new Error(`unknown sim strategy: ${String(strategyId)}`);

    simConfig.strategyId = strategyId;

    // Rebuild params/dt from defaults in the schema.
    simConfig.params = {};
    for (const p of strategy.params) {
      if (p.path.length === 2 && p.path[0] === "params") {
        simConfig.params[p.path[1]] = p.defaultValue;
      } else if (p.path.length === 1 && p.path[0] === "dims") {
        simConfig.dims = p.defaultValue;
      } else if (p.path.length === 1 && p.path[0] === "dt") {
        simConfig.dt = p.defaultValue;
      } else if (p.path.length === 1 && p.path[0] === "ticksPerSecond") {
        simConfig.ticksPerSecond = p.defaultValue;
      }
    }

    simConfig.seeding = strategy.seedings?.[0]?.config ?? simConfig.seeding;
  }

  function getSimConfigValue(path) {
    let cur = simConfig;
    for (const key of path) {
      if (!cur || typeof cur !== "object") return undefined;
      cur = cur[key];
    }
    return cur;
  }

  function setSimConfigValue(path, value) {
    if (path.length === 1) {
      simConfig[path[0]] = value;
      return;
    }

    if (path.length === 2 && path[0] === "params") {
      simConfig.params[path[1]] = value;
      return;
    }

    throw new Error(`unsupported sim config path: ${path.join(".")}`);
  }

  function buildSimConfigUpdate(path, value) {
    if (path.length === 1) return { [path[0]]: value };
    if (path.length === 2 && path[0] === "params") return { params: { [path[1]]: value } };
    throw new Error(`unsupported sim config update path: ${path.join(".")}`);
  }

  function renderSimParams() {
    const strategy = simStrategies[simConfig.strategyId];
    if (!simParamsEl || !strategy) return;

    simParamsEl.textContent = "";

    for (const p of strategy.params) {
      const input = document.createElement("input");
      input.type = "number";
      input.step = String(p.step);
      input.min = String(p.min);
      input.max = String(p.max);

      const label = document.createElement("label");
      label.append(p.label, input);

      const initialValue = getSimConfigValue(p.path) ?? p.defaultValue;
      setNumberInputValue(input, initialValue, p.step);

        const applyValue = (format) => {
        const cur = getSimConfigValue(p.path) ?? p.defaultValue;
        const next = parseClampedFloat(input.value, cur, p.min, p.max);
        setSimConfigValue(p.path, next);
        if (format) setNumberInputValue(input, next, p.step);

        // `dims` changes require reallocating shared buffers.
        if (p.path.length === 1 && p.path[0] === "dims") {
          restartFromUi();
          return;
        }

        simWorker?.postMessage({ type: "sim_config", config: buildSimConfigUpdate(p.path, next) });
      };


      if (p.requiresRestart) {
        input.addEventListener("change", () => applyValue(true));
      } else {
        input.addEventListener("input", () => applyValue(false));
        input.addEventListener("change", () => applyValue(true));
      }

      simParamsEl.appendChild(label);
    }
  }

  function renderSimInitSelect() {
    if (!simInitSelect) return;

    const strategy = simStrategies[simConfig.strategyId];
    simInitSelect.textContent = "";

    for (const s of strategy.seedings ?? []) {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = s.name;
      simInitSelect.appendChild(opt);
    }

    const curType = simConfig.seeding?.type;
    const selected = (strategy.seedings ?? []).find((s) => s.config.type === curType) ??
      strategy.seedings?.[0];
    if (selected) simInitSelect.value = selected.id;

    simInitSelect.onchange = () => {
      const next = (strategy.seedings ?? []).find((s) => s.id === simInitSelect.value);
      if (!next) return;
      simConfig.seeding = next.config;
      simWorker?.postMessage({ type: "sim_config", config: { seeding: simConfig.seeding } });
    };
  }

  function renderSimStrategySelect() {
    if (!simStrategySelect) return;

    simStrategySelect.textContent = "";

    for (const strategy of Object.values(simStrategies)) {
      const opt = document.createElement("option");
      opt.value = strategy.id;
      opt.textContent = strategy.name;
      simStrategySelect.appendChild(opt);
    }

    simStrategySelect.value = simConfig.strategyId;

    simStrategySelect.onchange = () => {
      const nextId = simStrategySelect.value;
      const prevDims = simConfig.dims;
      resetSimConfigForStrategy(nextId);
      renderSimInitSelect();
      renderSimParams();

      if (simConfig.dims !== prevDims) {
        restartFromUi();
        return;
      }

      simWorker?.postMessage({ type: "sim_config", config: simConfig });
    };
  }

  renderSimStrategySelect();
  renderSimInitSelect();
  renderSimParams();

  if (volumeThresholdInput) volumeThresholdInput.value = volumeThreshold.toFixed(2);
  if (viewRadiusInput) viewRadiusInput.value = viewRadius.toFixed(2);
  if (gradMagGainInput) gradMagGainInput.value = String(gradMagGain);

  volumeThresholdInput?.addEventListener("input", () => {
    volumeThreshold = parseClampedFloat(volumeThresholdInput.value, volumeThreshold, 0, 1);
  });
  volumeThresholdInput?.addEventListener("change", () => {
    volumeThreshold = parseClampedFloat(volumeThresholdInput.value, volumeThreshold, 0, 1);
    volumeThresholdInput.value = volumeThreshold.toFixed(2);
  });

  viewRadiusInput?.addEventListener("input", () => {
    viewRadius = parseClampedFloat(viewRadiusInput.value, viewRadius, 0.05, 5);
  });
  viewRadiusInput?.addEventListener("change", () => {
    viewRadius = parseClampedFloat(viewRadiusInput.value, viewRadius, 0.05, 5);
    viewRadiusInput.value = viewRadius.toFixed(2);
  });

  gradMagGainInput?.addEventListener("input", () => {
    gradMagGain = parseClampedFloat(gradMagGainInput.value, gradMagGain, 0, 50);
  });
  gradMagGainInput?.addEventListener("change", () => {
    gradMagGain = parseClampedFloat(gradMagGainInput.value, gradMagGain, 0, 50);
    gradMagGainInput.value = String(gradMagGain);
  });

  restartBtn?.addEventListener("click", () => {
    document.exitPointerLock?.();
    restartFromUi();
  });

  seedInput?.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    restartFromUi();
  });

  let lastCamSendAt = 0;
  startWorkers(initialSeed);
  function sendCamera() {
    const now = performance.now();
    if (now - lastCamSendAt < 33) return;
    lastCamSendAt = now;

    if (!meshReady || !meshWorker) return;
    meshWorker.postMessage({
      type: "camera",
      pos: cam.pos,
      radius: viewRadius,
      iso: volumeThreshold,
      color: meshColor,
      gradMagGain,
    });
  }

  const moveSpeedScale = 0.5;
  function moveCam(dt) {
    const v = cam.moveSpeed * dt * moveSpeedScale;
    let delta = [0, 0, 0];
    const f = cam.forward();
    const r = cam.right();

    if (keys.has("KeyW")) delta = vec3Add(delta, vec3Scale(f, v));
    if (keys.has("KeyS")) delta = vec3Add(delta, vec3Scale(f, -v));
    if (keys.has("KeyD")) delta = vec3Add(delta, vec3Scale(r, v));
    if (keys.has("KeyA")) delta = vec3Add(delta, vec3Scale(r, -v));

    cam.pos = vec3Add(cam.pos, delta);
  }


  let lastT = performance.now();
  function render(tNow) {
    const dt = Math.min(0.05, (tNow - lastT) / 1000);
    lastT = tNow;

    resizeCanvasToDisplaySize(canvas);
    gl.viewport(0, 0, canvas.width, canvas.height);

    moveCam(dt);
    sendCamera();

    const aspect = canvas.width / canvas.height;
    const proj = mat4Perspective((60 * Math.PI) / 180, aspect, 0.01, 100.0);
    const view = cam.viewMatrix();
    const viewProj = mat4Mul(proj, view);

    gl.clearColor(fogColor[0], fogColor[1], fogColor[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(program);
    gl.uniformMatrix4fv(uViewProjLoc, false, viewProj);
    gl.uniform3f(uLightDirLoc, -0.4, 0.8, 0.2);
    gl.uniform3f(uCamPosLoc, cam.pos[0], cam.pos[1], cam.pos[2]);
    gl.uniform3f(uFogColorLoc, fogColor[0], fogColor[1], fogColor[2]);
    gl.uniform1f(uFogDensityLoc, fogDensity);

    gl.bindVertexArray(gpuMesh.vao);
    if (gpuMesh.indexCount > 0) {
      // Disable depth writes so inner surfaces can blend through.
      //gl.depthMask(false);
      gl.drawElements(gl.TRIANGLES, gpuMesh.indexCount, gl.UNSIGNED_INT, 0);
      gl.depthMask(true);
    }
    gl.bindVertexArray(null);

    const vtx = gpuMesh.vertexCount;
    statsEl.textContent = `verts ${vtx.toLocaleString()}  sim ${lastSimStepsPerSec.toFixed(1)} steps/s  mesh ${lastMeshMs.toFixed(1)}ms  epoch ${lastMeshEpoch}  steps ${Math.floor(lastSimTotalSteps).toLocaleString()}`;

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
}

main().catch((e) => {
  console.error(e);
  statsEl.textContent = String(e);
});
