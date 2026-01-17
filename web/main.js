import { FlyCamera, createMouseFlightController } from "./camera.js";
import { createAoBlurProgram, createAoCompositeProgram, createAoProgram, createMeshProgram } from "./shaders.js";
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

function mat4Invert(m) {
  const inv = new Float32Array(16);

  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

  let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return null;

  det = 1.0 / det;
  for (let i = 0; i < 16; i++) inv[i] *= det;
  return inv;
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
  const { program: aoProgram, vao: aoVao, uniforms: aoUniforms } = createAoProgram(gl);
  const { program: aoBlurProgram, vao: aoBlurVao, uniforms: aoBlurUniforms } = createAoBlurProgram(gl);
  const { program: compositeProgram, vao: compositeVao, uniforms: compositeUniforms } = createAoCompositeProgram(gl);

  function createSceneTargets(width, height) {
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);

    const colorTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, colorTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colorTex, 0);

    const normalTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, normalTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, normalTex, 0);

    const depthTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, depthTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, width, height, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depthTex, 0);

    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);

    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    if (!ok) {
      gl.deleteFramebuffer(fbo);
      gl.deleteTexture(colorTex);
      gl.deleteTexture(normalTex);
      gl.deleteTexture(depthTex);
      return null;
    }

    return {
      fbo,
      colorTex,
      normalTex,
      depthTex,
      width,
      height,
    };
  }

  function deleteSceneTargets(t) {
    if (!t) return;
    gl.deleteFramebuffer(t.fbo);
    gl.deleteTexture(t.colorTex);
    gl.deleteTexture(t.normalTex);
    gl.deleteTexture(t.depthTex);
  }

  let sceneTargets = null;
  function ensureSceneTargets(width, height) {
    if (sceneTargets && sceneTargets.width === width && sceneTargets.height === height) return;
    deleteSceneTargets(sceneTargets);
    sceneTargets = createSceneTargets(width, height);
  }

  function createAoTargets(width, height) {
    const fboRaw = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboRaw);

    const aoRawTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, aoRawTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, null);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, aoRawTex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);

    const okRaw = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;

    const fboTmp = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboTmp);

    const aoTmpTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, aoTmpTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, null);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, aoTmpTex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);

    const okTmp = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;

    const fboBlur = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboBlur);

    const aoBlurTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, aoBlurTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, null);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, aoBlurTex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);

    const okBlur = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    if (!okRaw || !okTmp || !okBlur) {
      gl.deleteFramebuffer(fboRaw);
      gl.deleteFramebuffer(fboTmp);
      gl.deleteFramebuffer(fboBlur);
      gl.deleteTexture(aoRawTex);
      gl.deleteTexture(aoTmpTex);
      gl.deleteTexture(aoBlurTex);
      return null;
    }

    return {
      fboRaw,
      fboTmp,
      fboBlur,
      aoRawTex,
      aoTmpTex,
      aoBlurTex,
      width,
      height,
    };
  }

  function deleteAoTargets(t) {
    if (!t) return;
    gl.deleteFramebuffer(t.fboRaw);
    gl.deleteFramebuffer(t.fboTmp);
    gl.deleteFramebuffer(t.fboBlur);
    gl.deleteTexture(t.aoRawTex);
    gl.deleteTexture(t.aoTmpTex);
    gl.deleteTexture(t.aoBlurTex);
  }

  let aoTargets = null;
  function ensureAoTargets(width, height) {
    const w = Math.max(1, Math.floor(width / 2));
    const h = Math.max(1, Math.floor(height / 2));
    if (aoTargets && aoTargets.width === w && aoTargets.height === h) return;
    deleteAoTargets(aoTargets);
    aoTargets = createAoTargets(w, h);
  }

  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  // Premultiplied alpha.
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  const fogColor = [0.04, 0.06, 0.14];
  const fogDensity = 10.0;
  const lightDir = [-0.4, 0.8, 0.2];
  const near = 0.01;
  const far = 100.0;

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
    ensureSceneTargets(canvas.width, canvas.height);
    ensureAoTargets(canvas.width, canvas.height);

    cameraController.update(dt);
    sendCamera();

    const fovyRad = (60 * Math.PI) / 180;
    const tanHalfFovy = Math.tan(fovyRad / 2);

    const aspect = canvas.width / canvas.height;
    const proj = mat4Perspective(fovyRad, aspect, near, far);
    const invProj = mat4Invert(proj);
    if (!invProj) {
      requestAnimationFrame(render);
      return;
    }

    const view = cam.viewMatrix();
    const viewProj = mat4Mul(proj, view);

    if (!sceneTargets || !aoTargets) {
      requestAnimationFrame(render);
      return;
    }

    const camSettings = hud.getCameraSettings();

    // Scene pass (color + normal + depth).
    gl.bindFramebuffer(gl.FRAMEBUFFER, sceneTargets.fbo);
    gl.viewport(0, 0, sceneTargets.width, sceneTargets.height);
    gl.clearBufferfv(gl.COLOR, 0, new Float32Array([fogColor[0], fogColor[1], fogColor[2], 1]));
    gl.clearBufferfv(gl.COLOR, 1, new Float32Array([0.5, 0.5, 1.0, 1.0]));
    gl.clear(gl.DEPTH_BUFFER_BIT);

    gl.useProgram(meshProgram);
    gl.uniformMatrix4fv(meshUniforms.uViewProj, false, viewProj);
    gl.uniformMatrix4fv(meshUniforms.uView, false, view);
    gl.uniform3f(meshUniforms.uLightDir, lightDir[0], lightDir[1], lightDir[2]);
    gl.uniform3f(meshUniforms.uCamPos, cam.pos[0], cam.pos[1], cam.pos[2]);
    gl.uniform3f(meshUniforms.uFogColor, fogColor[0], fogColor[1], fogColor[2]);
    gl.uniform1f(meshUniforms.uFogDensity, fogDensity);
    gl.uniform1f(meshUniforms.uLightIntensity, camSettings.lightIntensity);
    gl.uniform1f(meshUniforms.uSssEnabled, camSettings.sssEnabled ? 1.0 : 0.0);
    gl.uniform1f(meshUniforms.uSssWrap, camSettings.sssWrap);
    gl.uniform1f(meshUniforms.uSssBackStrength, camSettings.sssBackStrength);
    gl.uniform1f(meshUniforms.uSssBackPower, camSettings.sssBackPower);

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    gl.bindVertexArray(gpuMesh.vao);
    if (gpuMesh.indexCount > 0) {
      gl.depthMask(true);
      gl.drawElements(gl.TRIANGLES, gpuMesh.indexCount, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);

    // SSAO pass (outputs AO factor into aoRawTex).
    gl.bindFramebuffer(gl.FRAMEBUFFER, aoTargets.fboRaw);
    gl.viewport(0, 0, aoTargets.width, aoTargets.height);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.clearBufferfv(gl.COLOR, 0, new Float32Array([1, 1, 1, 1]));

    gl.useProgram(aoProgram);
    gl.bindVertexArray(aoVao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTargets.depthTex);
    gl.uniform1i(aoUniforms.uDepth, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, sceneTargets.normalTex);
    gl.uniform1i(aoUniforms.uNormal, 1);

    gl.uniform2f(aoUniforms.uInvResolution, 1.0 / canvas.width, 1.0 / canvas.height);
    gl.uniformMatrix4fv(aoUniforms.uProj, false, proj);
    gl.uniformMatrix4fv(aoUniforms.uInvProj, false, invProj);
    gl.uniform1f(aoUniforms.uTanHalfFovy, tanHalfFovy);
    gl.uniform1f(aoUniforms.uFogDensity, fogDensity);
    gl.uniform1f(aoUniforms.uNear, near);
    gl.uniform1f(aoUniforms.uFar, far);
    gl.uniform1f(aoUniforms.uAoEnabled, camSettings.aoEnabled ? 1.0 : 0.0);
    gl.uniform1f(aoUniforms.uAoIntensity, camSettings.aoIntensity);
    gl.uniform1f(aoUniforms.uAoRadiusPx, camSettings.aoRadiusPx);
    gl.uniform1i(aoUniforms.uAoSampleCount, Math.max(1, Math.min(16, Math.trunc(camSettings.aoSamples ?? 8))));
    gl.uniform1f(aoUniforms.uAoBias, camSettings.aoBias);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Depth-aware blur pass (two-pass separable).
    gl.useProgram(aoBlurProgram);
    gl.bindVertexArray(aoBlurVao);

    const blurRadius = Math.max(0, Math.min(4, Math.trunc(camSettings.aoSoftness ?? 3)));

    // Horizontal.
    gl.bindFramebuffer(gl.FRAMEBUFFER, aoTargets.fboTmp);
    gl.viewport(0, 0, aoTargets.width, aoTargets.height);
    gl.clearBufferfv(gl.COLOR, 0, new Float32Array([1, 1, 1, 1]));

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, aoTargets.aoRawTex);
    gl.uniform1i(aoBlurUniforms.uAo, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, sceneTargets.depthTex);
    gl.uniform1i(aoBlurUniforms.uDepth, 1);

    gl.uniform2f(aoBlurUniforms.uInvResolution, 1.0 / aoTargets.width, 1.0 / aoTargets.height);
    gl.uniform2f(aoBlurUniforms.uDirection, 1.0, 0.0);
    gl.uniform1i(aoBlurUniforms.uBlurRadius, blurRadius);
    gl.uniform1f(aoBlurUniforms.uNear, near);
    gl.uniform1f(aoBlurUniforms.uFar, far);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Vertical.
    gl.bindFramebuffer(gl.FRAMEBUFFER, aoTargets.fboBlur);
    gl.viewport(0, 0, aoTargets.width, aoTargets.height);
    gl.clearBufferfv(gl.COLOR, 0, new Float32Array([1, 1, 1, 1]));

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, aoTargets.aoTmpTex);
    gl.uniform1i(aoBlurUniforms.uAo, 0);

    gl.uniform2f(aoBlurUniforms.uDirection, 0.0, 1.0);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Composite pass.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);

    gl.useProgram(compositeProgram);
    gl.bindVertexArray(compositeVao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTargets.colorTex);
    gl.uniform1i(compositeUniforms.uColor, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, aoTargets.aoBlurTex);
    gl.uniform1i(compositeUniforms.uAo, 1);

    gl.uniform3f(compositeUniforms.uFogColor, fogColor[0], fogColor[1], fogColor[2]);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0);
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
