import { FlyCamera, createMouseFlightController } from "./camera.js";
import { createMeshProgram } from "./shaders.js";
import { createHudController } from "./config.js";

const canvas = document.querySelector("#c");
const statsEl = document.querySelector("#stats");

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

function mat4Mul(a, b) {
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

  const { program: meshProgram, uniforms: meshUniforms } = createMeshProgram(gl);

  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  // Premultiplied alpha.
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  const fogColor = [0.04, 0.06, 0.14];
  const fogDensity = 10.0;
  const lightDir = [-0.4, 0.8, 0.2];

  const cam = new FlyCamera();
  const cameraController = createMouseFlightController({ canvas, camera: cam });

  const gpuMesh = createMeshGpu(gl);

  // Worker lifecycle + stats.
  let computeWorker = null;
  let workerReady = false;
  let lastMeshMs = 0;
  let lastMeshEpoch = 0;
  let lastSimStepsPerSec = 0;
  let lastSimTotalSteps = 0;
  let threadInfo = null;

  function stopWorkers() {
    computeWorker?.terminate();
    computeWorker = null;
    workerReady = false;
  }

  function startWorkers(seed, simConfig) {
    stopWorkers();

    const dims = Number.isFinite(simConfig?.dims) ? Math.max(16, Math.min(256, Math.trunc(simConfig.dims))) : 128;

    // Reset UI + mesh.
    gpuMesh.indexCount = 0;
    gpuMesh.vertexCount = 0;
    lastMeshMs = 0;
    lastMeshEpoch = 0;
    lastSimStepsPerSec = 0;
    lastSimTotalSteps = 0;
    threadInfo = null;

    computeWorker = new Worker(new URL("./compute_worker.js", import.meta.url), { type: "module" });

    const canUseWasmThreads = (globalThis.crossOriginIsolated === true) && (typeof SharedArrayBuffer !== "undefined");
    const threadCount = (canUseWasmThreads && navigator.hardwareConcurrency)
      ? Math.max(1, Math.min(8, navigator.hardwareConcurrency))
      : undefined;

    computeWorker.postMessage({
      type: "init",
      seed,
      dims,
      simConfig,
      threadCount,
    });

    computeWorker.onmessage = (e) => {
      const msg = e.data;
      if (!msg || typeof msg !== "object") return;

      if (msg.type === "sim_ready") return;
      if (msg.type === "mesh_ready") {
        workerReady = true;
        return;
      }

      if (msg.type === "thread_info") {
        threadInfo = msg;
        return;
      }

      if (msg.type === "sim_stats") {
        lastSimStepsPerSec = msg.stepsPerSec || 0;
        lastSimTotalSteps = msg.totalSteps || 0;
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
        if (statsEl) statsEl.textContent = String(msg.message);
      }
    };
  }

  const hud = createHudController({
    onRestart: (seed, simConfig) => startWorkers(seed, simConfig),
    getWorker: () => computeWorker,
  });

  // Camera -> worker updates.
  let lastCamSendAt = 0;
  function sendCamera() {
    const now = performance.now();
    if (now - lastCamSendAt < 33) return;
    lastCamSendAt = now;

    if (!workerReady || !computeWorker) return;

    const camSettings = hud.getCameraSettings();
    computeWorker.postMessage({
      type: "camera",
      pos: cam.pos,
      radius: camSettings.radius,
      iso: camSettings.iso,
      color: camSettings.color,
      gradMagGain: camSettings.gradMagGain,
    });
  }

  function formatThreadStatus() {
    if (!threadInfo || typeof threadInfo !== "object") return "thr/act (?/?)";

    const threads = Number.isFinite(threadInfo.threads) ? Math.max(1, Math.trunc(threadInfo.threads)) : 1;
    const rayonThreads = Number.isFinite(threadInfo.rayonThreads) ? Math.max(1, Math.trunc(threadInfo.rayonThreads)) : null;
    const active = rayonThreads ?? "?";

    const reasonMap = {
      no_sab: "noSAB",
      not_isolated: "noCOOP/COEP",
      no_shared_memory: "noSharedMem",
      no_thread_init: "noInit",
      init_failed: "initFail",
      not_requested: "off",
    };
    const shortReason = reasonMap[String(threadInfo.reason)] || String(threadInfo.reason || "");

    if (threadInfo.status === "enabled") return `thr/act (${threads}/${active})`;
    if (threadInfo.status === "disabled") return `thr/act (${threads}/${active})${shortReason ? ` (${shortReason})` : ""}`;
    if (threadInfo.status === "unavailable") return `thr/act (${threads}/${active})${shortReason ? ` (mt ${shortReason})` : " (mt n/a)"}`;
    if (threadInfo.status === "failed") return `thr/act (${threads}/${active})${shortReason ? ` (mt ${shortReason})` : " (mt fail)"}`;
    return `thr/act (${threads}/${active})`;
  }

  let lastT = performance.now();
  function render(tNow) {
    const dt = Math.min(0.05, (tNow - lastT) / 1000);
    lastT = tNow;

    resizeCanvasToDisplaySize(canvas);
    gl.viewport(0, 0, canvas.width, canvas.height);

    cameraController.update(dt);
    sendCamera();

    const aspect = canvas.width / canvas.height;
    const proj = mat4Perspective((60 * Math.PI) / 180, aspect, 0.01, 100.0);
    const view = cam.viewMatrix();
    const viewProj = mat4Mul(proj, view);

    // Background.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.clearColor(fogColor[0], fogColor[1], fogColor[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(meshProgram);
    gl.uniformMatrix4fv(meshUniforms.uViewProj, false, viewProj);
    gl.uniform3f(meshUniforms.uLightDir, lightDir[0], lightDir[1], lightDir[2]);
    gl.uniform3f(meshUniforms.uCamPos, cam.pos[0], cam.pos[1], cam.pos[2]);
    gl.uniform3f(meshUniforms.uFogColor, fogColor[0], fogColor[1], fogColor[2]);
    gl.uniform1f(meshUniforms.uFogDensity, fogDensity);

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    gl.bindVertexArray(gpuMesh.vao);
    if (gpuMesh.indexCount > 0) {
      gl.depthMask(true);
      gl.drawElements(gl.TRIANGLES, gpuMesh.indexCount, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);

    if (statsEl) {
      const vtx = gpuMesh.vertexCount;
      statsEl.textContent = `verts ${vtx.toLocaleString()}  sim ${lastSimStepsPerSec.toFixed(1)} steps/s  mesh ${lastMeshMs.toFixed(1)}ms  ${formatThreadStatus()}  epoch ${lastMeshEpoch}  steps ${Math.floor(lastSimTotalSteps).toLocaleString()}`;
    }

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
}

main().catch((e) => {
  console.error(e);
  if (statsEl) statsEl.textContent = String(e);
});
