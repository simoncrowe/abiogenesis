import { FlyCamera, createMouseFlightController, mat4Invert, mat4Mul, mat4Perspective } from "./camera.js";
import { createAoBlurProgram, createAoCompositeProgram, createAoProgram, createMeshProgram } from "./shaders.js";
import { createHudController } from "./config.js";
import { bootAudio, ensureGlobalReverb, isAudioBooted, noteOn } from "./synth.js";
import { createSoundscape } from "./soundscape.js";
import { MSG_TYPES } from "./msg_types.js";

const canvas = document.querySelector("#c");
const statsEl = document.querySelector("#stats");
const audioBootBtn = document.querySelector("#audioBoot");
const audioTestBtn = document.querySelector("#audioTest");
const audioMusicBtn = document.querySelector("#audioMusic");
const audioStatusEl = document.querySelector("#audioStatus");
const audioStatsEl = document.querySelector("#audioStats");

function setAudioStatus(text) {
  if (!audioStatusEl) return;
  audioStatusEl.textContent = text ? `(${text})` : "";
}

function setAudioStats(text) {
  if (!audioStatsEl) return;
  audioStatsEl.textContent = text || "";
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
  const normals = new Float32Array(msg.normals);
  const col = new Float32Array(msg.colors);
  const idx = new Uint32Array(msg.indices);

  gl.bindVertexArray(gpuMesh.vao);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpuMesh.posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpuMesh.norBuf);
  gl.bufferData(gl.ARRAY_BUFFER, normals, gl.DYNAMIC_DRAW);

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
      type: MSG_TYPES.INIT,
      seed,
      dims,
      simConfig,
      threadCount,
    });

    computeWorker.onmessage = (e) => {
      const msg = e.data;
      if (!msg || typeof msg !== "object") return;

      if (msg.type === MSG_TYPES.SIM_READY) return;
      if (msg.type === MSG_TYPES.MESH_READY) {
        workerReady = true;
        return;
      }

      if (msg.type === MSG_TYPES.THREAD_INFO) {
        threadInfo = msg;
        return;
      }

      if (msg.type === MSG_TYPES.SIM_STATS) {
        lastSimStepsPerSec = msg.stepsPerSec || 0;
        lastSimTotalSteps = msg.totalSteps || 0;
        return;
      }

      // Camera neighbourhood scalar stats (normalized 0..1), driving the soundscape.
      if (msg.type === MSG_TYPES.CAMERA_VOXEL_STATS) {
        if (Number.isFinite(msg.mean)) statsTarget.mean = Math.max(0, Math.min(1, msg.mean));
        if (Number.isFinite(msg.median)) statsTarget.median = Math.max(0, Math.min(1, msg.median));
        if (Number.isFinite(msg.max)) statsTarget.max = Math.max(0, Math.min(1, msg.max));
        if (Number.isFinite(msg.min)) statsTarget.min = Math.max(0, Math.min(1, msg.min));
        if (Number.isFinite(msg.range)) statsTarget.range = Math.max(0, Math.min(1, msg.range));
        return;
      }

      if (msg.type === MSG_TYPES.MESH) {
        uploadMeshFromBuffers(gl, gpuMesh, msg);
        lastMeshMs = msg.meshMs || 0;
        lastMeshEpoch = msg.epoch || 0;
        return;
      }

      if (msg.type === MSG_TYPES.ERROR) {
        console.error(msg.message);
        if (statsEl) statsEl.textContent = String(msg.message);
      }
    };
  }

  // Smoothed camera-neighbourhood stats (normalized 0..1).
  // Must be initialized before `createHudController` triggers the first worker start.
  const statsTarget = { mean: 0.5, median: 0.5, max: 0.5, min: 0.5, range: 0.0 };
  const statsSmooth = { mean: 0.5, median: 0.5, max: 0.5, min: 0.5, range: 0.0 };
  // Lower smoothing lag so voxel stats influence synth params faster.
  const SMOOTH_TAU_S = 0.25;

  const updateSmoothedStats = (dt) => {
    const a = 1 - Math.exp(-Math.max(0, dt) / SMOOTH_TAU_S);
    for (const k of Object.keys(statsSmooth)) {
      statsSmooth[k] += (statsTarget[k] - statsSmooth[k]) * a;
    }
  };

  const hud = createHudController({
    onRestart: (seed, simConfig) => startWorkers(seed, simConfig),
    getWorker: () => computeWorker,
  });

  // The conductor: maps smoothed camera stats to granular-voice parameters.
  // Fixed seed -> reproducible drift/scene evolution.
  const soundscape = createSoundscape({ seed: 0xC0FFEE });

  const iso01 = () => {
    const v = Number(hud.getCameraSettings()?.iso);
    return Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0.5;
  };

  const refreshAudioUi = () => {
    const ok = isAudioBooted();
    if (audioTestBtn) audioTestBtn.disabled = !ok;
    if (audioMusicBtn) audioMusicBtn.disabled = !ok;
    if (audioMusicBtn) audioMusicBtn.textContent = soundscape.running ? "Stop soundscape" : "Start soundscape";
    setAudioStatus(ok ? (soundscape.running ? "soundscape" : "ready") : "off");
  };

  refreshAudioUi();

  audioBootBtn?.addEventListener("click", async () => {
    try {
      setAudioStatus("booting...");
      await bootAudio();
      await ensureGlobalReverb();
      refreshAudioUi();
    } catch (e) {
      console.error(e);
      setAudioStatus("error");
    }
  });

  audioTestBtn?.addEventListener("click", () => {
    try {
      // A slightly low note so it's clearly audible.
      noteOn({ note: 48, amp: 0.22, attack: 0.02, release: 2.5 });
    } catch (e) {
      console.error(e);
      refreshAudioUi();
    }
  });

  audioMusicBtn?.addEventListener("click", () => {
    try {
      if (soundscape.running) soundscape.stop();
      else soundscape.start(statsSmooth, iso01());
      refreshAudioUi();
    } catch (e) {
      console.error(e);
      refreshAudioUi();
    }
  });

  // Camera -> worker updates.
  let lastCamSendAt = 0;
  function sendCamera() {
    const now = performance.now();
    if (now - lastCamSendAt < 20) return;
    lastCamSendAt = now;

    if (!workerReady || !computeWorker) return;

    const camSettings = hud.getCameraSettings();
    computeWorker.postMessage({
      type: MSG_TYPES.CAMERA,
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
  let lastAudioHudAt = 0;
  function render(tNow) {
    const dt = Math.min(0.05, (tNow - lastT) / 1000);
    lastT = tNow;

    resizeCanvasToDisplaySize(canvas);
    ensureSceneTargets(canvas.width, canvas.height);
    ensureAoTargets(canvas.width, canvas.height);

    cameraController.update(dt);
    sendCamera();

    updateSmoothedStats(dt);
    // The conductor (soundscape.js) owns all audio-parameter updates: it throttles
    // its own /n_set batches and drives the FX chain, so nothing is done here.
    if (isAudioBooted()) {
      soundscape.update(dt, tNow, statsSmooth, iso01());
    }

    if (tNow - lastAudioHudAt > 100) {
      lastAudioHudAt = tNow;
      const d = soundscape.getDebug();
      setAudioStats(
        `mean ${statsSmooth.mean.toFixed(3)}  median ${statsSmooth.median.toFixed(3)}  ` +
        `max ${statsSmooth.max.toFixed(3)}  min ${statsSmooth.min.toFixed(3)}  range ${statsSmooth.range.toFixed(3)}\n` +
        `cutoff ${d.cutoff.toFixed(1)}  reverb room ${d.room.toFixed(3)}  mix ${d.mix.toFixed(3)}  ` +
        `anchor ${d.anchor.toFixed(1)}Hz  scene ${d.scene}`,
      );
    }

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
