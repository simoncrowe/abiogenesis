// Shared helpers for the granular soundscape (see soundscape.js).
//
// The chord-progression / arpeggiator machinery that used to live here was
// retired with architecture/15-granular-ambient.md; only these general-purpose
// helpers survive.

export function clamp01(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

// Seeded xorshift RNG. A fixed seed reproduces the same soundscape evolution.
export function createRng(seed = 1234) {
  let s = (seed >>> 0) || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 2 ** 32;
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
  return 0.5 + (room * 2); // Scale reverb room size
}
