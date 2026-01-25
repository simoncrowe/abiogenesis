const seedInput = document.querySelector("#seed");
const simStrategySelect = document.querySelector("#simStrategy");
const simInitSelect = document.querySelector("#simInit");
const simExportSelect = document.querySelector("#simExport");
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


  const simExportModes = {
    // Shared IDs; not every strategy will expose all.
    phase: { id: "phase", name: "Phase field" },
    phase_tanh: { id: "phase_tanh", name: "Phase field (tanh)" },
    membranes: { id: "membranes", name: "Active membranes (|mu|)" },
    energy: { id: "energy", name: "Energy density" },
  };

  const simStrategies = {
    gray_scott: {
      id: "gray_scott",
      name: "Gray-Scott reaction-diffusion",
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

        { key: "df", path: ["params", "df"], label: "Diffusion F (df)", min: 0, max: 2, step: 0.001, defaultValue: 0.2, requiresRestart: true },
        { key: "da", path: ["params", "da"], label: "Diffusion A (da)", min: 0, max: 2, step: 0.001, defaultValue: 0.05, requiresRestart: true },
        { key: "di", path: ["params", "di"], label: "Diffusion I (di)", min: 0, max: 2, step: 0.001, defaultValue: 0.02, requiresRestart: true },

        { key: "k1", path: ["params", "k1"], label: "Rate k1 (A+F->2A)", min: 0, max: 0.1, step: 0.00001, defaultValue: 0.002, requiresRestart: true },
        { key: "k2", path: ["params", "k2"], label: "Rate k2 (A->A+I)", min: 0, max: 0.5, step: 0.0001, defaultValue: 0.02, requiresRestart: true },
        { key: "k3", path: ["params", "k3"], label: "Rate k3 (A+I->I)", min: 0, max: 0.1, step: 0.00001, defaultValue: 0.001, requiresRestart: true },

        { key: "feedBase", path: ["params", "feedBase"], label: "Feed base", min: 0, max: 10, step: 0.001, defaultValue: 2.0, requiresRestart: true },
        { key: "feedNoiseAmp", path: ["params", "feedNoiseAmp"], label: "Feed noise amp", min: 0, max: 2, step: 0.001, defaultValue: 0.35, requiresRestart: true },
        { key: "feedNoiseScale", path: ["params", "feedNoiseScale"], label: "Feed noise scale", min: 1, max: 64, step: 1, defaultValue: 8, requiresRestart: true },

        { key: "decayA", path: ["params", "decayA"], label: "Decay A", min: 0, max: 1, step: 0.0001, defaultValue: 0.01, requiresRestart: true },
        { key: "decayI", path: ["params", "decayI"], label: "Decay I", min: 0, max: 1, step: 0.0001, defaultValue: 0.005, requiresRestart: true },
        { key: "decayF", path: ["params", "decayF"], label: "Decay F", min: 0, max: 1, step: 0.0001, defaultValue: 0.0, requiresRestart: true },

        { key: "etaScale", path: ["params", "etaScale"], label: "Noise scale (eta)", min: 0, max: 2, step: 0.001, defaultValue: 0.25, requiresRestart: true },
        { key: "substeps", path: ["params", "substeps"], label: "Substeps", min: 1, max: 8, step: 1, defaultValue: 1, requiresRestart: true },

        { key: "alivenessAlpha", path: ["params", "alivenessAlpha"], label: "Aliveness alpha", min: 0, max: 2, step: 0.001, defaultValue: 0.25, requiresRestart: true },
        { key: "alivenessGain", path: ["params", "alivenessGain"], label: "Aliveness gain", min: 0.00001, max: 1, step: 0.00001, defaultValue: 0.05, requiresRestart: true },
      ],
      seedings: [
        {
          id: "perlin",
          name: "Perlin noise",
          config: {
            type: "perlin",
            frequency: 6.0,
            octaves: 4,
            baseF: 50,
            baseI: 0,
            aBias: 0.0,
            aAmp: 20.0,
          },
        },
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

    cahn_hilliard: {
      id: "cahn_hilliard",
      name: "Cahn-Hilliard phase field",
      exportModes: [
        simExportModes.phase_tanh,
        simExportModes.phase,
        simExportModes.membranes,
        simExportModes.energy,
      ],
      params: [
        { key: "dims", path: ["dims"], label: "Grid size (dims)", min: 16, max: 256, step: 1, defaultValue: 128, requiresRestart: true },
        { key: "dt", path: ["dt"], label: "Simulation timestep (dt)", min: 0.0001, max: 0.2, step: 0.0001, defaultValue: 0.002, requiresRestart: false },
        { key: "ticksPerSecond", path: ["ticksPerSecond"], label: "Publish rate (ticks/s)", min: 1, max: 60, step: 1, defaultValue: 5, requiresRestart: false },

        { key: "a", path: ["params", "a"], label: "Double-well strength (A)", min: 0, max: 5, step: 0.001, defaultValue: 1.0, requiresRestart: true },
        { key: "kappa", path: ["params", "kappa"], label: "Interface width (kappa)", min: 0, max: 5, step: 0.001, defaultValue: 0.6, requiresRestart: true },
        { key: "m", path: ["params", "m"], label: "Mobility (M)", min: 0, max: 5, step: 0.001, defaultValue: 0.2, requiresRestart: true },
        { key: "substeps", path: ["params", "substeps"], label: "Substeps", min: 1, max: 16, step: 1, defaultValue: 2, requiresRestart: true },
        { key: "passMode", path: ["params", "passMode"], label: "Passes (2=full,1=fast)", min: 1, max: 2, step: 1, defaultValue: 1, requiresRestart: true },
        { key: "approxMode", path: ["params", "approxMode"], label: "2-pass approx (0/1)", min: 0, max: 1, step: 1, defaultValue: 0, requiresRestart: true },

        { key: "phiMean", path: ["params", "phiMean"], label: "Mean phi", min: -1, max: 1, step: 0.001, defaultValue: -0.45, requiresRestart: true },
        { key: "noiseAmp", path: ["params", "noiseAmp"], label: "Init noise amp", min: 0, max: 0.2, step: 0.001, defaultValue: 0.02, requiresRestart: true },
      ],
      seedings: [
        {
          id: "spinodal",
          name: "Spinodal (noise)",
          config: {
            type: "spinodal",
            phiMean: 0.0,
            noiseAmp: 0.02,
          },
        },
        {
          id: "droplets",
          name: "Droplets (biased)",
          config: {
            type: "droplets",
            phiMean: -0.45,
            noiseAmp: 0.02,
          },
        },
        {
          id: "membranes",
          name: "Active membranes",
          config: {
            type: "membranes",
            phiMean: 0.0,
            noiseAmp: 0.02,
          },
        },
        {
          id: "energy",
          name: "Energy density",
          config: {
            type: "energy",
            phiMean: -0.45,
            noiseAmp: 0.02,
          },
        },
      ],
    },

    excitable_media: {
      id: "excitable_media",
      name: "Excitable media (Barkley)",
      params: [
        { key: "dims", path: ["dims"], label: "Grid size (dims)", min: 16, max: 256, step: 1, defaultValue: 128, requiresRestart: true },
        { key: "dt", path: ["dt"], label: "Simulation timestep (dt)", min: 0.0001, max: 0.05, step: 0.0001, defaultValue: 0.01, requiresRestart: false },
        { key: "ticksPerSecond", path: ["ticksPerSecond"], label: "Publish rate (ticks/s)", min: 1, max: 60, step: 1, defaultValue: 5, requiresRestart: false },

        { key: "epsilon", path: ["params", "epsilon"], label: "Timescale sep (epsilon)", min: 0.001, max: 0.2, step: 0.0001, defaultValue: 0.03, requiresRestart: true },
        { key: "a", path: ["params", "a"], label: "Threshold (a)", min: 0.1, max: 2, step: 0.0001, defaultValue: 0.85, requiresRestart: true },
        { key: "b", path: ["params", "b"], label: "Bias (b)", min: 0, max: 0.2, step: 0.0001, defaultValue: 0.01, requiresRestart: true },

        { key: "du", path: ["params", "du"], label: "Diffusion u (Du)", min: 0, max: 3, step: 0.001, defaultValue: 1.0, requiresRestart: true },
        { key: "dv", path: ["params", "dv"], label: "Diffusion v (Dv)", min: 0, max: 1, step: 0.001, defaultValue: 0.0, requiresRestart: true },

        { key: "substeps", path: ["params", "substeps"], label: "Substeps", min: 1, max: 8, step: 1, defaultValue: 1, requiresRestart: true },
      ],
      seedings: [
        {
          id: "random",
          name: "Random excitation",
          config: {
            type: "random",
            noiseAmp: 0.02,
            excitedProb: 0.002,
          },
        },
        {
          id: "turbulence",
          name: "A: Turbulence (self-sustaining)",
          config: {
            type: "random",
            noiseAmp: 0.02,
            excitedProb: 0.01,
            preset: "A",
          },
        },
        {
          id: "sources",
          name: "Seeded sources",
          config: {
            type: "sources",
            sourceCount: 8,
            radius01: 0.06,
            uPeak: 1.0,
          },
        },
        {
          id: "scroll",
          name: "B: Scroll waves (stable)",
          config: {
            type: "sources",
            sourceCount: 4,
            radius01: 0.10,
            uPeak: 1.2,
            preset: "B",
          },
        },
      ],
    },

    replicator_mutator: {
      id: "replicator_mutator",
      name: "Replicator-mutator ecology",
      params: [
        { key: "dims", path: ["dims"], label: "Grid size (dims)", min: 16, max: 256, step: 1, defaultValue: 128, requiresRestart: true },
        { key: "dt", path: ["dt"], label: "Simulation timestep (dt)", min: 0.0001, max: 0.2, step: 0.0001, defaultValue: 0.02, requiresRestart: false },
        { key: "ticksPerSecond", path: ["ticksPerSecond"], label: "Publish rate (ticks/s)", min: 1, max: 60, step: 1, defaultValue: 5, requiresRestart: false },

        { key: "types", path: ["params", "types"], label: "Types (K)", min: 2, max: 8, step: 1, defaultValue: 4, requiresRestart: true },

        // These defaults aim for long-lived dynamics without runaway biomass.
        { key: "gBase", path: ["params", "gBase"], label: "Growth base (g0)", min: 0, max: 1, step: 0.0001, defaultValue: 0.06, requiresRestart: true },
        { key: "gSpread", path: ["params", "gSpread"], label: "Growth spread", min: 0, max: 2, step: 0.0001, defaultValue: 0.20, requiresRestart: true },
        { key: "dR", path: ["params", "dR"], label: "Replicator decay (dR)", min: 0, max: 1, step: 0.0001, defaultValue: 0.03, requiresRestart: true },

        { key: "feedRate", path: ["params", "feedRate"], label: "Feed rate", min: 0, max: 1, step: 0.0001, defaultValue: 0.04, requiresRestart: true },
        { key: "dF", path: ["params", "dF"], label: "Feed decay (dF)", min: 0, max: 1, step: 0.0001, defaultValue: 0.01, requiresRestart: true },

        { key: "mu", path: ["params", "mu"], label: "Mutation rate (mu)", min: 0, max: 0.05, step: 0.0001, defaultValue: 0.003, requiresRestart: true },

        { key: "diffR", path: ["params", "diffR"], label: "Diffusion R (DR)", min: 0, max: 0.2, step: 0.0001, defaultValue: 0.01, requiresRestart: true },
        { key: "diffF", path: ["params", "diffF"], label: "Diffusion F (DF)", min: 0, max: 2, step: 0.0001, defaultValue: 0.20, requiresRestart: true },

        { key: "substeps", path: ["params", "substeps"], label: "Substeps", min: 1, max: 8, step: 1, defaultValue: 2, requiresRestart: true },
      ],
      seedings: [
        {
          id: "uniform",
          name: "Uniform low density + noise",
          config: {
            type: "uniform",
            noiseAmp: 0.02,
            rBase: 0.012,
            fInit: 0.8,
          },
        },
        {
          id: "regions",
          name: "Localized regions",
          config: {
            type: "regions",
            noiseAmp: 0.01,
            rPeak: 0.08,
            fInit: 0.8,
          },
        },
        {
          id: "gradient",
          name: "Gradient-driven niches",
          config: {
            type: "gradient",
            noiseAmp: 0.02,
            rBase: 0.012,
            fInit: 0.8,
            feedBase: 0.04,
            feedAmp: 1.0,
            axis: 0,
          },
        },
      ],
    },

    lenia: {
      id: "lenia",
      name: "Continuous CA (Lenia)",
      params: [
        { key: "dims", path: ["dims"], label: "Grid size (dims)", min: 16, max: 256, step: 1, defaultValue: 64, requiresRestart: true },
        { key: "dt", path: ["dt"], label: "Simulation timestep (dt)", min: 0.001, max: 1, step: 0.001, defaultValue: 0.010, requiresRestart: false },
        { key: "ticksPerSecond", path: ["ticksPerSecond"], label: "Publish rate (ticks/s)", min: 1, max: 60, step: 1, defaultValue: 5, requiresRestart: false },

        { key: "radius", path: ["params", "radius"], label: "Kernel radius (R)", min: 1, max: 10, step: 1, defaultValue: 5, requiresRestart: true },
        { key: "mu", path: ["params", "mu"], label: "Growth mu", min: 0, max: 1, step: 0.0001, defaultValue: 0.15, requiresRestart: true },
        { key: "sigma", path: ["params", "sigma"], label: "Growth sigma", min: 0.0001, max: 0.2, step: 0.0001, defaultValue: 0.03, requiresRestart: true },
        { key: "sharpness", path: ["params", "sharpness"], label: "Kernel sharpness", min: 0.1, max: 4, step: 0.1, defaultValue: 0.5, requiresRestart: true },
      ],
      seedings: [
        {
          id: "blobs",
          name: "Seeded blobs",
          config: { type: "blobs", blobCount: 12, radius01: 0.08, peak: 1.0 },
        },
        {
          id: "noise",
          name: "Random noise",
          config: { type: "noise", amp: 0.02 },
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
    exportMode: null,
  };

  // Ensure defaults are applied for the initial strategy.
  resetSimConfigForStrategy(simConfig.strategyId);

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
    simConfig.exportMode = strategy.exportModes?.[0]?.id ?? null;
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

  function renderSimExportSelect() {
    if (!simExportSelect) return;

    const strategy = simStrategies[simConfig.strategyId];
    simExportSelect.textContent = "";

    for (const m of strategy.exportModes ?? []) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.name;
      simExportSelect.appendChild(opt);
    }

    if ((strategy.exportModes ?? []).length === 0) {
      simExportSelect.disabled = true;
      return;
    }

    simExportSelect.disabled = false;
    simExportSelect.value = simConfig.exportMode ?? strategy.exportModes?.[0]?.id;

    simExportSelect.onchange = () => {
      simConfig.exportMode = simExportSelect.value;
      getWorker()?.postMessage({ type: "sim_config", config: { exportMode: simConfig.exportMode } });
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

      // Strategy-specific default isosurface threshold.
      // (User can still override via the "volume threshold" slider afterward.)
      if (nextId === "stochastic_rdme") {
        volumeThreshold = 0.10;

        gradMagGain = 10.0;
        // RDME tends to look best slightly closer-in by default.
        viewRadius = 0.40;
        if (viewRadiusInput) viewRadiusInput.value = viewRadius.toFixed(2);
      } else if (nextId === "gray_scott") {
        volumeThreshold = 0.25;
      } else if (nextId === "cahn_hilliard") {
        // CH exports a phase boundary centered at v=0.5.
        volumeThreshold = 0.50;

        // Smaller view radius tends to work better with CH droplets.
        viewRadius = 0.35;
        if (viewRadiusInput) viewRadiusInput.value = viewRadius.toFixed(2);

        // CH interface gradients are typically sharper than Gray-Scott.
        // Use a lower default gain so the ramp doesn't instantly saturate.
        gradMagGain = 1.0;
        if (gradMagGainInput) gradMagGainInput.value = String(gradMagGain);
      } else if (nextId === "excitable_media") {
        // Excitable media exported field is raw activator u (unbounded-ish), but iso around 0.5
        // tends to pick out the wavefront volume nicely.
        volumeThreshold = 0.50;

        // Wavefronts have fairly strong gradients; keep the gain moderate.
        gradMagGain = 4.0;
        if (gradMagGainInput) gradMagGainInput.value = String(gradMagGain);
      } else if (nextId === "replicator_mutator") {
        // Biomass tends to live in a lower range; slightly lower iso works better.
        volumeThreshold = 0.15;

        gradMagGain = 3.0;
        if (gradMagGainInput) gradMagGainInput.value = String(gradMagGain);
      } else if (nextId === "lenia") {
        volumeThreshold = 0.50;

        // Lenia fields are smooth; use a lower gain so the ramp doesn't saturate.
        gradMagGain = 1.0;
        if (gradMagGainInput) gradMagGainInput.value = String(gradMagGain);
      }
      if (volumeThresholdInput) volumeThresholdInput.value = volumeThreshold.toFixed(2);

      renderSimInitSelect();
      renderSimExportSelect();
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
  renderSimExportSelect();
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
