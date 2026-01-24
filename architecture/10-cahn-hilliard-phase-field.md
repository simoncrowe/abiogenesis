# architecture/10-cahn-hilliard-phase-field.md
# architecture/10-cahn-hilliard-phase-field.md

## Purpose

Add a Cahn-Hilliard (CH) phase-field simulation backend to the existing PoC to provide
compartmentalization / droplet / membrane-like dynamics on an `n^3` lattice, suitable for 3D isosurface
extraction (marching tetrahedra) and efficient parallel stepping (Rust + Rayon, wasm threads).

CH fills a niche not covered by Gray-Scott or the current stochastic RDME/CLE-style simulation:
- spontaneous phase separation (droplets, bicontinuous phases)
- membrane-like interfaces (large gradients at boundaries)
- protocell-like compartments when later coupled to chemistry

Primary output: a single scalar field per voxel (phi or derived) exported to JS as f32.

In this codebase, that exported scalar should be presented as `v` (to match `ScalarFieldMesher` and the existing JS worker pipeline). Practically, `v` is expected to be in [0,1], so a simple mapping like `v = (phi + 1)/2` keeps the UI slider behavior consistent.

---

## Overview

Cahn-Hilliard is a conserved-order-parameter phase separation model:

- phi(x) is a scalar field (e.g., "composition" / "oil-water fraction")
- phi evolves to minimize a free energy, subject to conservation of total phi

Continuous form (standard):
1) Chemical potential:
   mu = f'(phi) - kappa * Laplacian(phi)

2) Evolution:
   dphi/dt = M * Laplacian(mu)

Where:
- f(phi) is a double-well potential (commonly (phi^2 - 1)^2 / 4)
- kappa controls interface width / surface tension
- M is mobility (diffusional timescale)

In discrete form this becomes a stencil-based update composed of local nonlinear terms + Laplacians.

---

## Simulation mode integration

The current codebase does not have a unified Rust-side `SimulationMode` enum/trait.
Instead, the JS worker chooses a strategy via `simConfig.strategyId` (e.g. `"gray_scott"`, `"stochastic_rdme"`) and instantiates a specific wasm-exported simulation type.

Practical integration pattern (match existing sims):
- Add a new strategy id: `"cahn_hilliard"`.
- Add a wasm-exported params type and simulation type:
  - `CahnHilliardParams`
  - `CahnHilliardSimulation`
- Match the existing JS contract used by `web/compute_worker.js` + `ScalarFieldMesher`:
  - `step(steps: usize)`
  - `set_dt(dt: f32)`
  - `v_ptr() -> u32` / `v_len() -> usize` (the mesher always consumes `v`)
  - `recompute_chunk_ranges_from_v()` + `chunk_v_min_ptr()` / `chunk_v_max_ptr()` / `chunk_v_len()` for iso-culling

Notes:
- In the current pipeline, "aliveness" is just the exported `v` scalar field.
- Chunk min/max are recomputed on publish (keyframes), not necessarily every simulation step.

---

## Lattice, indexing, boundaries

- Domain: `nx x ny x nz`, periodic boundaries (matches existing sims).
- Target: start at 128^3, but expect the UI to default closer to 192^3.

Indexing (match existing `idx(nx, ny, x, y, z)` helper):
- `i = x + nx * (y + ny * z)`

Neighbors for Laplacian:
- 6-neighbor stencil (matches existing gather-style diffusion kernels):
  - `L(phi)[i] = phi[x-1] + phi[x+1] + phi[y-1] + phi[y+1] + phi[z-1] + phi[z+1] - 6*phi[i]`

Implementation detail (match performance style in `wasm/src/gray_scott.rs` and `wasm/src/rdme.rs`):
- Precompute wrapped index tables once:
  - `x_minus[x]`, `x_plus[x]`, etc.
- Iterate in `z`-slice `par_chunks_mut(nxy)` form to avoid per-voxel div/mod in hot loops.

---

## State and buffers

Core state:
- `phi[i]: f32` conserved order parameter.

Buffers (match existing "double-buffered arrays" pattern):
- `phi0[n]` read
- `phi1[n]` write

Scratch:
- `mu[n]` scratch (single buffer).

Why two-pass first (practical):
- Existing sims already accept extra buffers when it keeps kernels simple.
- Two-pass makes it easy to validate `mu` and `phi` separately and keeps the inner loops clean.

Memory note:
- 3 buffers * `n` * 4 bytes = ~56MB at 192^3 (7.1M voxels).
- That is within reach for native and often for wasm, but is worth keeping in mind; if wasm memory becomes tight, the next optimization is to avoid storing `mu`.

---

## Free energy model (recognized default)

Use the standard double-well free energy density:
- f(phi) = A * (phi^2 - 1)^2 / 4

Then:
- f'(phi) = A * (phi^3 - phi)

Chemical potential:
- mu = A*(phi^3 - phi) - kappa * Laplacian(phi)

Notes:
- A controls strength of phase separation
- kappa controls interface thickness and surface tension-like behavior
- The stable phases are approximately phi ~ +1 and phi ~ -1

---

## Discrete update

Let:
- `L(phi)` be the 6-neighbor Laplacian.

Two-pass kernel (recommended initial implementation):

Pass 1 (compute `mu`):
- `mu[i] = A * (phi^3 - phi) - kappa * L(phi)[i]`

Pass 2 (update `phi`):
- `phi_next[i] = phi[i] + dt * M * L(mu)[i]`

Conservation / drift checks:
- Mean(`phi`) should be approximately conserved.
- In practice, floating point + boundaries + explicit Euler can drift; it is worth tracking mean(`phi`) in debug builds and/or occasionally in JS (sample a few slices).

Practical stability knobs (match existing RDME approach):
- Prefer adding `substeps: u32` to the params (internal loop doing `dt/substeps`) rather than clamping.
- Avoid clamping `phi` hard to [-1,1] as a first response; it breaks mass conservation and tends to produce sticky, unphysical interfaces.

---

## Numerical stability guidance

CH is more numerically sensitive than Gray-Scott because the update effectively behaves like a 4th-order diffusion term.

Practical controls (this codebase):
- Use a small displayed `dt` (UI value), and internally run `substeps` like RDME does.
- If you see checkerboarding or blow-up: reduce `dt`, increase `substeps`, and/or reduce `M`.

Notes:
- A semi-implicit spectral scheme (FFT) is the "real" standard for CH, but it is a major plumbing shift (FFT dependency, complex buffers, and different boundary assumptions). For this project, explicit + conservative timesteps is the pragmatic first step.

---

## Parallelization (Rust + Rayon + WASM threads)

This fits the existing gather-stencil parallelization model.

Implementation pattern (copy from `wasm/src/gray_scott.rs` and `wasm/src/rdme.rs`):
- Parallelize over `z` slices: `par_chunks_mut(nxy).enumerate()`.
- For each slice, compute neighbor offsets using `x_minus/x_plus/y_minus/y_plus/z_minus/z_plus`.

Pass structure:
- Pass 1: write `mu[z][y][x]` from `phi0`.
- Pass 2: write `phi1[z][y][x]` from `mu`.
- Swap buffers.

Wasm threading:
- The worker already has optional thread pool init via `init_thread_pool()` and will run single-thread if threads are unavailable. CH should just use Rayon in the same style; no extra JS-side plumbing required.

---

## Initialization strategies (must support at least two)

Match existing UI seeding conventions (see `web/config.js`): each strategy advertises a small set of seedings.

Recommended seedings for CH:

1) Spinodal ("noise"):
- `phi = phi_mean + uniform_noise`
- Defaults: `phi_mean = 0.0`, `noise_amp = 0.01`.

2) Droplets ("spheres"):
- Background `phi = -1` and seed `sphere_count` spheres of `phi = +1`.
- Keep it deterministic (seeded RNG) like existing JS/Rust seeding.

Practical note:
- Use the same seeding style as RDME/Gray-Scott: provide wasm methods like `seed_perlin(...)` / `seed_spheres(...)` or accept JS-side seeding by exposing `phi_ptr()`/`phi_len()`.
- If you only pick one, pick wasm-side seeding; JS-side direct writes make it easy to accidentally desync chunk min/max unless you recompute.

---

## Exported scalar field (what the mesher consumes)

In the current app, the mesher consumes a single scalar field called `v`.
For CH we should export a mapped scalar so it fits the existing [0,1] convention:
- `v = 0.5 + 0.5 * tanh(gain * phi)` (recommended default to keep gradient-based shading in range)
- `v = (phi + 1)/2` (simplest)

Why this mapping (practical):
- The phase boundary `phi = 0` becomes `v = 0.5` (easy default iso).
- Keeps the meshing/UI pipeline consistent with other modes that treat `v` as a normalized field.

Optional second visualization mode (later):
- Export `v = 1 - exp(-gain * |grad(phi)|)` to emphasize membranes/interfaces, similar to RDME's aliveness mapping.
- That would require either a separate `v` buffer or writing into `v` on publish.

---

## Parameter set (initial, non-binding)

Expose a minimal param set similar to other sims (`*_Params` with getters/setters):

- `a` (A): phase separation strength, start 1.0
- `kappa`: interface thickness, start 1.0
- `m` (M): mobility, start 1.0
- `dt`: start 0.01 (displayed dt)
- `substeps`: start 2..=8 (internal stability knob)

Suggested UI defaults (practical):
- `dt = 0.01` is more realistic for explicit CH than 0.1.
- If targeting 192^3, start with `substeps = 4` so users can still drag dt a bit without instant blow-up.

---

## Success criteria

Behavioral:
- Stable phase separation (spinodal -> coarsening) and/or droplet ripening.
- Clean `phi=0` surfaces without persistent numerical checkerboarding.

Technical:
- No atomics/locks; gather stencils only.
- No per-step allocations.
- Works in wasm single-thread, and scales under wasm threads when available.

Performance:
- Similar structure to RDME: 2 parallel passes over `n` per simulation step (+ optional publish-time chunk-range recompute).

---

## Extensions (explicitly not required for PoC)

1) CH + reactions (protocell coupling):
- add one or more reaction-diffusion fields that preferentially localize in one phase
- make M or kappa depend on local chemistry (active emulsions)

2) Hydrodynamics:
- couple to flow (LBM) for advective transport

3) Semi-implicit integrators:
- improve stability and allow larger dt

These should be designed as add-on modules, keeping CH core loosely coupled.

---
