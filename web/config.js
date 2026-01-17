const seedInput = document.querySelector("#seed");
const simStrategySelect = document.querySelector("#simStrategy");
const simInitSelect = document.querySelector("#simInit");
const simParamsEl = document.querySelector("#simParams");
const volumeThresholdInput = document.querySelector("#volumeThreshold");
const viewRadiusInput = document.querySelector("#viewRadius");
const meshOpacityInput = document.querySelector("#meshOpacity");
const lightIntensityInput = document.querySelector("#lightIntensity");
const sssEnabledInput = document.querySelector("#sssEnabled");
const sssWrapInput = document.querySelector("#sssWrap");
const sssBackStrengthInput = document.querySelector("#sssBackStrength");
const sssBackPowerInput = document.querySelector("#sssBackPower");
const aoEnabledInput = document.querySelector("#aoEnabled");
const aoIntensityInput = document.querySelector("#aoIntensity");
const aoRadiusInput = document.querySelector("#aoRadius");
const aoSamplesInput = document.querySelector("#aoSamples");
const aoSoftnessInput = document.querySelector("#aoSoftness");
const aoBiasInput = document.querySelector("#aoBias");
const gradMagGainInput = document.querySelector("#gradMagGain");
const restartBtn = document.querySelector("#restart");

function normalizeSeed(value) {
  const n = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(n)) return Date.now() >>> 0;
  return n >>> 0;
}

function parseClampedFloat(value, fallback, min, max) {
  const n = Number.parseFloat(String(value ?? ""));
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

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

export function createHudController({
  onRestart,
  getWorker,
}) {
  if (typeof onRestart !== "function") throw new Error("createHudController requires onRestart");

  // Display settings
  let viewRadius = 0.6;
  let volumeThreshold = 0.25;
  let gradMagGain = 12.0;

  const meshColor = [0.15, 0.65, 0.9, 0.75];
  let meshOpacity = meshColor[3];

  let lightIntensity = 1.0;

  let sssEnabled = false;
  let sssWrap = 0.4;
  let sssBackStrength = 0.4;
  let sssBackPower = 2.0;

  let aoEnabled = false;
  let aoIntensity = 1.5;
  let aoRadiusPx = 16.0;
  let aoSamples = 8;
  let aoSoftness = 3;
  let aoBias = 0.002;


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
          label: "Kill rate (V decay)",
          min: 0,
          max: 0.1,
          step: 0.0001,
          defaultValue: 0.0649,
          requiresRestart: true,
        },
        {
          key: "dt",
          path: ["dt"],
          label: "Simulation timestep (dt)",
          min: 0.001,
          max: 1,
          step: 0.001,
          defaultValue: 0.1,
          requiresRestart: false,
        },
        {
          key: "ticksPerSecond",
          path: ["ticksPerSecond"],
          label: "Publish rate (ticks/s)",
          min: 1,
          max: 60,
          step: 1,
          defaultValue: 5,
          requiresRestart: false,
        },
      ],
      seedings: [
        {
          id: "perlin",
          name: "Perlin noise",
          config: {
            type: "perlin",
            frequency: 6.0,
            octaves: 4,
            v_bias: 0.0,
            v_amp: 1.0,
          },
        },
        {
          id: "classic",
          name: "Classic (spheres + noise)",
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
      name: "Stochastic RDME",
      params: [
        { key: "dims", path: ["dims"], label: "Grid size (dims)", min: 16, max: 256, step: 1, defaultValue: 128, requiresRestart: true },
        { key: "dt", path: ["dt"], label: "Simulation timestep (dt)", min: 0.001, max: 1, step: 0.001, defaultValue: 0.05, requiresRestart: false },
        { key: "ticksPerSecond", path: ["ticksPerSecond"], label: "Publish rate (ticks/s)", min: 1, max: 60, step: 1, defaultValue: 5, requiresRestart: false },
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

  function resetSimConfigForStrategy(strategyId) {
    const strategy = simStrategies[strategyId];
    if (!strategy) throw new Error(`unknown sim strategy: ${String(strategyId)}`);

    simConfig.strategyId = strategyId;

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

        if (p.path.length === 1 && p.path[0] === "dims") {
          restartFromUi();
          return;
        }

        if (p.requiresRestart) {
          restartFromUi();
          return;
        }

        getWorker()?.postMessage({ type: "sim_config", config: buildSimConfigUpdate(p.path, next) });
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
    const selected = (strategy.seedings ?? []).find((s) => s.config.type === curType) ?? strategy.seedings?.[0];
    if (selected) simInitSelect.value = selected.id;

    simInitSelect.onchange = () => {
      const next = (strategy.seedings ?? []).find((s) => s.id === simInitSelect.value);
      if (!next) return;
      simConfig.seeding = next.config;
      getWorker()?.postMessage({ type: "sim_config", config: { seeding: simConfig.seeding } });
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

      getWorker()?.postMessage({ type: "sim_config", config: simConfig });
    };
  }

  function restartFromUi() {
    const seed = normalizeSeed(seedInput?.value);
    if (seedInput) seedInput.value = String(seed);
    onRestart(seed, structuredClone(simConfig));
  }

  // Initial UI values.
  renderSimStrategySelect();
  renderSimInitSelect();
  renderSimParams();

  if (volumeThresholdInput) volumeThresholdInput.value = volumeThreshold.toFixed(2);
  if (viewRadiusInput) viewRadiusInput.value = viewRadius.toFixed(2);
  if (gradMagGainInput) gradMagGainInput.value = String(gradMagGain);
  if (meshOpacityInput) meshOpacityInput.value = meshOpacity.toFixed(2);
  if (lightIntensityInput) setNumberInputValue(lightIntensityInput, lightIntensity, lightIntensityInput.step);
  if (sssEnabledInput) sssEnabledInput.checked = sssEnabled;
  if (sssWrapInput) setNumberInputValue(sssWrapInput, sssWrap, sssWrapInput.step);
  if (sssBackStrengthInput) setNumberInputValue(sssBackStrengthInput, sssBackStrength, sssBackStrengthInput.step);
  if (sssBackPowerInput) setNumberInputValue(sssBackPowerInput, sssBackPower, sssBackPowerInput.step);

  const updateSssUi = () => {
    const enabled = sssEnabled;
    if (sssWrapInput) sssWrapInput.disabled = !enabled;
    if (sssBackStrengthInput) sssBackStrengthInput.disabled = !enabled;
    if (sssBackPowerInput) sssBackPowerInput.disabled = !enabled;
  };

  sssEnabledInput?.addEventListener("change", () => {
    sssEnabled = sssEnabledInput.checked === true;
    updateSssUi();
  });

  updateSssUi();

  if (aoEnabledInput) aoEnabledInput.checked = aoEnabled;
  if (aoIntensityInput) setNumberInputValue(aoIntensityInput, aoIntensity, aoIntensityInput.step);
  if (aoRadiusInput) setNumberInputValue(aoRadiusInput, aoRadiusPx, aoRadiusInput.step);
  if (aoSamplesInput) setNumberInputValue(aoSamplesInput, aoSamples, aoSamplesInput.step);
  if (aoSoftnessInput) setNumberInputValue(aoSoftnessInput, aoSoftness, aoSoftnessInput.step);
  if (aoBiasInput) setNumberInputValue(aoBiasInput, aoBias, aoBiasInput.step);

  const updateAoUi = () => {
    const enabled = aoEnabled;
    if (aoIntensityInput) aoIntensityInput.disabled = !enabled;
    if (aoRadiusInput) aoRadiusInput.disabled = !enabled;
    if (aoSamplesInput) aoSamplesInput.disabled = !enabled;
    if (aoSoftnessInput) aoSoftnessInput.disabled = !enabled;
    if (aoBiasInput) aoBiasInput.disabled = !enabled;
  };

  aoEnabledInput?.addEventListener("change", () => {
    aoEnabled = aoEnabledInput.checked === true;
    updateAoUi();
  });

  updateAoUi();

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

  meshOpacityInput?.addEventListener("input", () => {
    meshOpacity = parseClampedFloat(meshOpacityInput.value, meshOpacity, 0, 1);
    meshColor[3] = meshOpacity;
  });
  meshOpacityInput?.addEventListener("change", () => {
    meshOpacity = parseClampedFloat(meshOpacityInput.value, meshOpacity, 0, 1);
    meshColor[3] = meshOpacity;
    meshOpacityInput.value = meshOpacity.toFixed(2);
  });

  lightIntensityInput?.addEventListener("input", () => {
    lightIntensity = parseClampedFloat(lightIntensityInput.value, lightIntensity, 0, 3);
  });
  lightIntensityInput?.addEventListener("change", () => {
    lightIntensity = parseClampedFloat(lightIntensityInput.value, lightIntensity, 0, 3);
    setNumberInputValue(lightIntensityInput, lightIntensity, lightIntensityInput.step);
  });

  sssWrapInput?.addEventListener("input", () => {
    sssWrap = parseClampedFloat(sssWrapInput.value, sssWrap, 0, 1);
  });
  sssWrapInput?.addEventListener("change", () => {
    sssWrap = parseClampedFloat(sssWrapInput.value, sssWrap, 0, 1);
    setNumberInputValue(sssWrapInput, sssWrap, sssWrapInput.step);
  });

  sssBackStrengthInput?.addEventListener("input", () => {
    sssBackStrength = parseClampedFloat(sssBackStrengthInput.value, sssBackStrength, 0, 2);
  });
  sssBackStrengthInput?.addEventListener("change", () => {
    sssBackStrength = parseClampedFloat(sssBackStrengthInput.value, sssBackStrength, 0, 2);
    setNumberInputValue(sssBackStrengthInput, sssBackStrength, sssBackStrengthInput.step);
  });

  sssBackPowerInput?.addEventListener("input", () => {
    sssBackPower = parseClampedFloat(sssBackPowerInput.value, sssBackPower, 0.1, 8);
  });
  sssBackPowerInput?.addEventListener("change", () => {
    sssBackPower = parseClampedFloat(sssBackPowerInput.value, sssBackPower, 0.1, 8);
    setNumberInputValue(sssBackPowerInput, sssBackPower, sssBackPowerInput.step);
  });

  aoIntensityInput?.addEventListener("input", () => {
    aoIntensity = parseClampedFloat(aoIntensityInput.value, aoIntensity, 0, 3);
  });
  aoIntensityInput?.addEventListener("change", () => {
    aoIntensity = parseClampedFloat(aoIntensityInput.value, aoIntensity, 0, 3);
    setNumberInputValue(aoIntensityInput, aoIntensity, aoIntensityInput.step);
  });

  aoRadiusInput?.addEventListener("input", () => {
    aoRadiusPx = parseClampedFloat(aoRadiusInput.value, aoRadiusPx, 1, 64);
  });
  aoRadiusInput?.addEventListener("change", () => {
    aoRadiusPx = parseClampedFloat(aoRadiusInput.value, aoRadiusPx, 1, 64);
    setNumberInputValue(aoRadiusInput, aoRadiusPx, aoRadiusInput.step);
  });

  aoSamplesInput?.addEventListener("input", () => {
    aoSamples = Math.trunc(parseClampedFloat(aoSamplesInput.value, aoSamples, 4, 16));
  });
  aoSamplesInput?.addEventListener("change", () => {
    aoSamples = Math.trunc(parseClampedFloat(aoSamplesInput.value, aoSamples, 4, 16));
    setNumberInputValue(aoSamplesInput, aoSamples, aoSamplesInput.step);
  });

  aoSoftnessInput?.addEventListener("input", () => {
    aoSoftness = Math.trunc(parseClampedFloat(aoSoftnessInput.value, aoSoftness, 0, 4));
  });
  aoSoftnessInput?.addEventListener("change", () => {
    aoSoftness = Math.trunc(parseClampedFloat(aoSoftnessInput.value, aoSoftness, 0, 4));
    setNumberInputValue(aoSoftnessInput, aoSoftness, aoSoftnessInput.step);
  });

  aoBiasInput?.addEventListener("input", () => {
    aoBias = parseClampedFloat(aoBiasInput.value, aoBias, 0, 0.02);
  });
  aoBiasInput?.addEventListener("change", () => {
    aoBias = parseClampedFloat(aoBiasInput.value, aoBias, 0, 0.02);
    setNumberInputValue(aoBiasInput, aoBias, aoBiasInput.step);
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

  const urlSeed = new URLSearchParams(globalThis.location?.search ?? "").get("seed");
  const initialSeed = normalizeSeed(urlSeed ?? seedInput?.value ?? 1337);
  if (seedInput) seedInput.value = String(initialSeed);

  // Fire initial start.
  onRestart(initialSeed, structuredClone(simConfig));

  return {
    getCameraSettings() {
      return {
        radius: viewRadius,
        iso: volumeThreshold,
        color: meshColor,
        gradMagGain,
        lightIntensity,
        sssEnabled,
        sssWrap,
        sssBackStrength,
        sssBackPower,
        aoEnabled,
        aoIntensity,
        aoRadiusPx,
        aoSamples,
        aoSoftness,
        aoBias,
      };
    },


    getSimConfig() {
      return structuredClone(simConfig);
    },
  };
}
