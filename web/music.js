function clamp01(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function pick(rng, arr) {
  return arr[Math.max(0, Math.min(arr.length - 1, Math.floor(rng() * arr.length)))];
}

function createRng(seed = 1234) {
  let s = (seed >>> 0) || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 2 ** 32;
  };
}

function chordMaj7(root) {
  return [root, root + 4, root + 7, root + 11];
}

function chordMin7(root) {
  return [root, root + 3, root + 7, root + 10];
}

function chordDom7(root) {
  return [root, root + 4, root + 7, root + 10];
}

// Minimal ambient-ish state machine in C major.
// States yield 7th chords; transitions are mildly stochastic.
const PROGRESSION = {
  // "_" is a pause/rest.
  _: { chord: null, next: ["I", "IV", "vi"] },

  I: { chord: (oct) => chordMaj7(60 + 12 * oct), next: ["IV", "vi", "ii"] },
  ii: { chord: (oct) => chordMin7(62 + 12 * oct), next: ["V", "IV"] },
  IV: { chord: (oct) => chordMaj7(65 + 12 * oct), next: ["I", "V", "ii", "_"] },
  V: { chord: (oct) => chordDom7(67 + 12 * oct), next: ["I", "vi"] },
  vi: { chord: (oct) => chordMin7(69 + 12 * oct), next: ["IV", "ii", "V"] },
};

export function createMusicEngine({
  bpm = 120,
  seed = 1,
  arpeggiate = false,
  onNote,
  getNoteParams,
} = {}) {
  if (typeof onNote !== "function") throw new Error("createMusicEngine requires onNote(note, params)");

  const rng = createRng(seed);

  let running = false;
  let timer = null;

  let state = "I";
  let chord = PROGRESSION[state].chord ? PROGRESSION[state].chord(0) : null;
  let arpIdx = 0;
  let stepInBar = 0;

  const secondsPerBeat = 60 / Math.max(1, bpm);
  const stepSeconds = secondsPerBeat / 2; // 8ths
  const stepMs = Math.max(10, Math.round(stepSeconds * 1000));

  const arpOrder = [0, 2, 1, 3];

  function nextChord() {
    const next = pick(rng, PROGRESSION[state].next);
    state = next;

    const oct = rng() < 0.15 ? 1 : 0;
    const def = PROGRESSION[state];
    chord = typeof def.chord === "function" ? def.chord(oct) : null;
    arpIdx = 0;
  }

  function tick() {
    if (!running) return;

    if (stepInBar === 0) {
      // Change chord each bar.
      if (rng() < 0.85) nextChord();
    }

    if (chord) {
      const paramsBase = typeof getNoteParams === "function" ? (getNoteParams() || {}) : {};

      if (arpeggiate) {
        const chordIdx = arpOrder[arpIdx % arpOrder.length];
        const note = chord[chordIdx];
        arpIdx++;
        onNote(note, { ...paramsBase, chordIndex: chordIdx, chordSize: chord.length });
      } else {
        // No arpeggiation: play the full chord once per bar.
        if (stepInBar === 0) {
          for (let i = 0; i < chord.length; i++) {
            onNote(chord[i], { ...paramsBase, chordIndex: i, chordSize: chord.length });
          }
        }
      }
    }

    stepInBar = (stepInBar + 1) % 8; // 8 eighth-notes per bar (4/4)
  }

  return {
    start() {
      if (running) return;
      running = true;
      timer = setInterval(tick, stepMs);
      tick();
    },
    stop() {
      running = false;
      if (timer) clearInterval(timer);
      timer = null;
    },
    get running() {
      return running;
    },
    get stepSeconds() {
      return stepSeconds;
    },
    setBpm(nextBpm) {
      const b = Number(nextBpm);
      if (!Number.isFinite(b) || b <= 0) return;
      bpm = b;
    },
    setArpeggiate(next) {
      arpeggiate = Boolean(next);
    },
  };
}

export function mapMeanToCutoff(mean01) {
  const t = clamp01(mean01);
  // Sonic Pi cutoff is roughly 0..130-ish; keep it musical.
  return 45 + t * 65;
}

export function mapMaxToReverbRoom(max01, volumeThreshold01 = 0.5) {
  // Use the meshing volume threshold (iso) as the scale: when the local max scalar
  // value approaches iso, the camera is near a surface we consider "solid".
  const maxT = clamp01(max01);
  const isoT = clamp01(volumeThreshold01);

  // 1 when max=0 (open space), 0 when max>=iso (near/inside solids).
  const denom = Math.max(1e-6, isoT);
  let room = clamp01((isoT - maxT) / denom);

  // Emphasize the near-surface region.
  room *= room; 
  return 0.5 +  (room * 4) // Scale reverb room size
}
