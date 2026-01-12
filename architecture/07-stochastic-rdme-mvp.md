# architecture/07-stochastic-rdme-mvp.md

## Purpose
Add a stochastic Reaction–Diffusion Master Equation (RDME) simulation backend to the existing PoC (currently Gray–Scott), sharing the same lattice, stepping, and output plumbing.

The RDME system must:
- run on an n^3 lattice (target: 128^3),
- use integer or integer-like molecule counts per voxel,
- support sustained, noisy, far-from-equilibrium dynamics,
- export a single scalar "aliveness" field per voxel for marching tetrahedra.

---

## High-level model

Each voxel is a well-mixed compartment.

State per voxel:
- F: food
- A: autocatalyst
- I: inhibitor / waste

Dynamics:
- stochastic reactions inside voxels
- stochastic diffusion as random hops between neighboring voxels
- open system with feed and decay

Numerical scheme:
- operator splitting (Strang):
  - reaction half-step
  - diffusion full-step
  - reaction half-step

---

## Lattice and storage

- Domain: NxNxN, periodic boundaries
- Target: N = 128
- Species count: 3 (F, A, I)

Storage (Rust):
- arrays: [i32] or [f32] but treated as non-negative integers
- double-buffered per species for diffusion safety

Indexing:
- linear index = x + N*(y + N*z)

---

## Reactions (per voxel)

1) Feed (open system)  
   Ø -> F  
   rate = s(x,y,z)

2) Autocatalysis  
   A + F -> 2A  
   rate = k1 * A * F

3) Inhibitor production  
   A -> A + I  
   rate = k2 * A

4) Inhibition  
   A + I -> I  
   rate = k3 * A * I

5) Decay  
   A -> Ø   rate = dA * A  
   I -> Ø   rate = dI * I  
   (optional) F -> Ø   rate = dF * F

Notes:
- rates are dimensionless; stability enforced via timestep choice
- reactions consuming species must never overdraw counts

---

## Reaction step (τ-leaping)

Per voxel, per reaction r:
- compute propensity a_r
- sample firings:
  - K_r ~ Poisson(a_r * dt)
  - OR (preferred for consuming reactions):
    K_r ~ Binomial(N, 1 - exp(-k * dt))

Apply stoichiometry updates:
- clamp to prevent negative counts
- if clamping occurs frequently, reduce dt

---

## Diffusion step (stochastic hopping)

Goal: implement diffusion in a way that is statistically reasonable **and** practical for a dense 128^3 wasm simulation.

For each species S ∈ {F, A, I}:
- diffusion coefficient `D_S`
- voxel spacing `h` (MVP uses `h = 1`)

In RDME form, each molecule has a per-neighbor hop propensity `D_S / h^2`.

**Practical MVP sampling (approximate but fast):**

Per voxel with `count = N_S`:
- total hop rate: `λ_total = N_S * 6 * D_S / h^2`
- sample number of hop events for this voxel over `dt`:
  - `L ~ Poisson(λ_total * dt)`
  - clamp: `L = min(L, N_S)`
- distribute the `L` hop events to 6 neighbors using a multinomial draw.
  - MVP implementation can use sequential binomial splitting rather than an explicit multinomial.

Update (double-buffered per species):
- `next[i] += N_S - L`
- `next[neighbor_dir] += L_dir` for each direction

Notes:
- This avoids looping once per molecule (which can explode when counts are large).
- If the simulation homogenizes too quickly or becomes too noisy, tune `dt`, `D_S`, and baseline counts.

Diffusion ordering (typical):
- `D_F > D_A >= D_I`

---

## Source field s(x,y,z)

Required for vibrancy.

Implement at least one:
- constant background + noise
- vent:
  s(x,y,z) = high inside a cylinder or sphere, low elsewhere
- gradient along one axis

s(x,y,z) is static or slowly varying.

---

## Timestep constraints

- diffusion stability:
  dt <= h^2 / (6 * max(D_S))
- reaction stability:
  expected firings per step should be O(1–10), not >> 1

dt is global and fixed.

---

## Aliveness scalar (output field)

Export **one scalar per voxel** for visualization.

In the current codebase, this scalar is consumed by the WebGL UI and mesher as the field named `v`, with an isovalue slider that assumes the field is roughly in `[0, 1]`.

**Requirement (practical integration):**
- exported scalar must be `f32`
- exported scalar should be **bounded and normalized** (target `[0, 1]`)
- higher = more "alive"
- stable under noise (avoid raw F)

Recommended mapping:
- `raw = max(0, A - alpha * I)`
- `aliveness = 1 - exp(-gain * raw)`
- clamp to `[0, 1]`

This field is passed to marching tetrahedra as-is.

---

## Integration with existing PoC

The current PoC selects simulation behavior via the JS-side `strategyId` (see `web/main.js` and `web/sim_worker.js`).

**Practical MVP integration plan:**
- Keep the existing Gray–Scott wasm export `Simulation` unchanged.
- Add a new wasm export `StochasticRdmeSimulation`.
- Update `web/sim_worker.js` to construct the correct wasm simulation class based on `simConfig.strategyId`.

**Compatibility contract (what the sim worker needs):**
- `step(steps: usize)`
- `set_dt(dt: f32)`
- `v_ptr() -> u32` and `v_len() -> usize` (exported scalar field)
- `chunk_v_min_ptr() -> u32`, `chunk_v_max_ptr() -> u32`, `chunk_v_len() -> usize`
- `recompute_chunk_ranges_from_v()` (called before publishing snapshots)

RDME owns:
- species count arrays (integer counts)
- RNG (seedable; deterministic)
- reaction + diffusion operators
- aliveness computation into the exported `v` scalar field

---

## Determinism and RNG

The broader architecture requires determinism.

- RNG must be explicit and seedable.
- Given identical `(seed, params, dims, dt)` the RDME simulation must produce identical results.
- Implementation may use:
  - a single deterministic RNG advanced in a fixed voxel iteration order, or
  - per-voxel/per-step derived RNG streams (e.g. `hash(seed, voxel_index, step, stage)`).

No use of nondeterministic browser RNGs.

---

## Success criteria

- System does NOT homogenize quickly
- Exhibits:
  - nucleation
  - extinction
  - recolonization
  - persistent spatial structure
- Aliveness isosurface evolves nontrivially over long runs

---

## Out of scope (for now)

- exact SSA (Gillespie)
- advection / flow
- membranes / compartments
- evolution of reaction rules

These may be layered later.
