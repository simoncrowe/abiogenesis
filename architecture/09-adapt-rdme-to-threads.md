# architecture/09-adapt-rdme-to-threads

## Purpose

Provide an architecture for replacing (or approximating) an RDME (integer-count, jump-process) implementation with a
parallel-friendly lattice simulation that scales cleanly under Rust + WASM + Rayon.

Target constraints:
- n^3 lattice (initial target: 128^3)
- high throughput on CPU threads (Rayon), compatible with WASM threads when enabled
- deterministic memory access patterns (stencil gathers), no atomic scatter updates
- exports a single f32 "aliveness" scalar field per voxel for marching tetrahedra / WebGL

The replacement model should be a recognized variant of stochastic spatial chemistry.

## Feasibility at 128^3

This design is feasible under the stated constraints.

Memory sketch (f32, N=128 => 2,097,152 voxels):
- one scalar field: ~8.0 MiB (N^3 * 4 bytes)
- 3 species, double-buffered: 3 * 2 * 8.0 MiB = ~48.0 MiB
- aliveness: ~8.0 MiB
- optional source field s(x,y,z): ~8.0 MiB
- total: ~56–72 MiB depending on whether s is stored vs computed on the fly

This is fine natively; in WASM it is reasonable as long as the module memory is sized accordingly and you avoid extra per-voxel temporary arrays.

Compute sketch:
- update is memory-bandwidth bound (6-neighbor gathers + a few FMAs + RNG)
- Rayon scales well because there are no atomics and no cross-thread writes
- in WASM, threading requires `SharedArrayBuffer` and cross-origin isolation; otherwise it falls back to single-thread and should still run at reduced resolution or reduced step rate

---

## Rationale: why RDME is hard to parallelize

A typical RDME diffusion step is "scatter":
- each voxel samples hops and writes to 6 neighbors
- this creates write conflicts between threads

Avoiding conflicts usually requires atomics, multi-pass scheduling, or storing directional outflows, each with cost and complexity.

Therefore, we replace RDME diffusion (jump process) with a recognized continuous approximation that preserves:
- local reactions
- diffusion
- intrinsic noise

...while using a read-neighbors / write-self update.

---

## Selected recognized variant: Chemical Langevin Equation with diffusion (Reaction–Diffusion CLE / SPDE)

Use the Chemical Langevin Equation (CLE) / reaction–diffusion SPDE as an approximation to RDME.

Important feasibility note:
- CLE is reliable when local copy numbers are not extremely small.
- If you treat the state as *concentration-like* f32 fields (not literal molecule counts), you can still get the desired "intrinsic noise" look by calibrating the noise amplitude (see `eta_scale`) while keeping the update stable.

Model per voxel is a stochastic differential equation:
- deterministic drift from reaction kinetics
- additive/multiplicative noise term derived from propensities
- diffusion implemented as Laplacian

This is commonly referred to as:
- Chemical Langevin Equation (CLE)
- Stochastic Reaction–Diffusion Equation (SRDE / SPDE)
- Reaction–diffusion with intrinsic noise (system-size expansion style)

Key property:
- diffusion is a stencil gather (Laplacian), fully parallelizable
- noise is local (per voxel), fully parallelizable

---

## State representation

Species per voxel (example set aligned with RDME brief):
- F: food
- A: autocatalyst
- I: inhibitor / waste

Representation:
- store concentrations as f32 (or f64) arrays
- enforce non-negativity by clamping after update (or use a positivity-preserving scheme)

Buffers:
- double-buffer per species for diffusion/stencil updates:
  - F0, A0, I0 (read)
  - F1, A1, I1 (write)
- optional: aliveness[] f32 buffer exported to JS

Indexing:
- linear index = x + N*(y + N*z)

Boundaries:
- periodic (recommended)

---

## Dynamics

### Deterministic drift (same reaction network as RDME, continuous form)

Reactions (local):
1) Feed:
   dF/dt += s(x,y,z)

2) Autocatalysis:
   A + F -> 2A
   dA/dt +=  k1 * A * F
   dF/dt += -k1 * A * F

3) Inhibitor production:
   A -> A + I
   dI/dt +=  k2 * A

4) Inhibition:
   A + I -> I
   dA/dt += -k3 * A * I

5) Decay:
   dA/dt += -dA * A
   dI/dt += -dI * I
   (optional) dF/dt += -dF * F

### Diffusion (stencil gather)

For each species S in {F, A, I}:
- dS/dt += D_S * Laplacian(S)

Laplacian stencil:
- 6-neighbor (recommended for simplicity), or 26-neighbor if needed
- periodic wrap for neighbors

### Noise term (CLE-style)

For each voxel and each reaction channel r with propensity a_r:
- add noise to affected species:
  delta_X += nu_r * sqrt(max(a_r, 0)) * Normal(0, 1) * sqrt(dt) * eta_scale

Notes:
- nu_r is stoichiometric change vector for reaction r
- eta_scale is a global knob to tune visible stochasticity (start near 0.1–1.0)

Numerical/physical guardrails (needed for feasibility in f32 + large dt ranges):
- clamp propensities: `a_r = max(a_r, 0)` before sqrt
- clamp noise magnitude if needed: `sigma = min(sigma, sigma_max)` (optional but practical)
- after applying drift+noise, clamp state to non-negative
- consider a soft floor: `X = max(X, eps)` for species that appear in denominators

This produces intrinsic reaction noise analogous to RDME fluctuations without per-event sampling.

Optional (advanced):
- add diffusion noise (usually skip; it increases high-frequency speckle and cost)

---

## Numerical integrator

Use explicit Euler–Maruyama (sufficient for PoC).

Per voxel:
- compute deterministic drift from reactions + diffusion
- compute stochastic increments from reaction noise
- update: `X_new = X_old + drift * dt + noise`

Stability considerations:
- keep dt consistent with diffusion CFL constraint:
  `dt <= h^2 / (6 * max(D_S))` for 6-neighbor stencil
- reaction stiffness can also limit dt (especially with large k1 * A * F)

Practical stabilization knobs (useful to actually hit the "long-lived dynamics" goal):
- substep diffusion+reaction: do 2–4 smaller internal steps per displayed step (still parallel)
- cap extreme concentrations: `X = min(X, X_max)` (optional; prevents blow-ups)
- keep `dt` small enough that the noise term does not dominate drift everywhere

Positivity:
- after update, clamp:
  `X_new = max(X_new, 0)`

---

## Parallelization strategy (Rust + Rayon + WASM threads)

Core rule:
- each voxel update writes only to its own index in output buffers

Per simulation step:
1) parallel over voxels:
   - read neighbors from (F0, A0, I0)
   - compute Laplacians
   - compute reaction drift
   - compute propensities a_r (for noise magnitudes)
   - draw local normal randoms (stateless RNG)
   - write updated (F1, A1, I1)

2) swap buffers (F0<->F1, etc.)

3) parallel over voxels:
   - compute aliveness[i] from (A0, I0, optionally F0)

This maps cleanly to:
- native: Rayon always enabled
- wasm: Rayon enabled only if WASM threads + SharedArrayBuffer available; otherwise fallback to single-thread loops

No atomics are required.

---

## RNG architecture (thread-safe and reproducible)

Requirement:
- no shared RNG state across threads
- deterministic outputs given seed, step, and voxel index (preferred)

Approach (recommended):
- stateless counter-based RNG per sample:
  `u = hash64(global_seed, timestep, voxel_index, stream_id)`
  -> convert u into uniform float(s)
  -> convert to Normal(0,1) via Box–Muller

Streams:
- use a small fixed set of `stream_id`s per voxel per step (e.g. 0..K)
- generate 2 normals at a time via Box–Muller to reduce hash/uniform cost

Reason:
- WASM + Rayon works best with deterministic, allocation-free RNG
- avoids per-thread RNG objects and avoids nondeterminism from work-stealing order

---

## "Aliveness" scalar output

Export aliveness[] f32 (length N^3) in WASM linear memory.

Recommended definition (choose one; start with simplest):
- aliveness = A - alpha * I
- or aliveness = A / (1 + I)
- or aliveness = log(1 + A)

Constraints:
- stable under noise
- produces meaningful isosurfaces
- avoid using raw F as primary aliveness (it is a driver, not "living" structure)

---

## Source field s(x,y,z)

A source field is required for long-lived nontrivial dynamics.

Implement at least one:
- vent cylinder / sphere:
  - high s inside region, low outside
- gradient along an axis
- low-level background feed + spatial noise

s(x,y,z) should be:
- cheap to compute (precompute array or analytic function)
- stable over time (static initially)

---

## Integration with existing PoC

Add a new backend implementing the same trait/interface as Gray–Scott:

enum SimulationMode {
  GrayScott,
  StochasticCLE,   // replacement for RDME
}

Shared responsibilities:
- lattice sizing, periodic indexing helpers
- stepping loop / time accumulation
- WASM exports and typed-array views
- marching tetrahedra pipeline consumes aliveness[] only

Backend responsibilities (StochasticCLE):
- buffers (F,A,I) double-buffered
- parameter set (D_S, k1,k2,k3, dA,dI,dF, dt, eta_scale, alpha)
- source field s(x,y,z)
- update kernel (parallel)

---

## Benefits vs RDME

- Same qualitative ingredients: reactions + diffusion + intrinsic noise
- Far simpler parallelization model: gather stencil + local noise
- Predictable memory access and high scaling under Rayon
- Clean WASM integration (no atomics, no scatter)

Tradeoffs / risks (and mitigations):
- Not exact low-copy-number RDME; best when values are locally moderate
- Euler–Maruyama can destabilize with large dt; mitigate with smaller dt or 2–4 substeps
- f32 + strong multiplicative noise can create negative excursions; clamp and cap noise if needed
- Visual and structural "aliveness" is typically strong once rates/source/noise are tuned

---

## Validation / success criteria

Behavioral:
- does not homogenize quickly
- shows nucleation, extinction, recolonization, traveling fronts, patchiness
- aliveness isosurfaces remain active over long runs

Technical:
- stable under Rayon parallelism (native + wasm threads)
- no data races, no atomics required
- deterministic with fixed seed (desired for debugging; optional for "live" mode)

Performance:
- one simulation step is a single parallel pass over N^3 for the update (+ optional pass for aliveness)
- no per-step allocations
- no per-step neighbor index tables (compute wrapped indices on the fly, or precompute only x/y/z wrap tables)

---

## Out of scope (initially)

- exact RDME event simulation
- diffusion noise
- adaptive dt
- membranes / compartments (phase field) - can be layered later

---

## Suggested initial parameter sketch (non-binding)

Assume lattice spacing h = 1.

Diffusion ordering:
- D_F high, D_A medium, D_I low-to-medium
- example: D_F=1.0, D_A=0.2, D_I=0.1

Reactions (example ranges):
- k1 (A+F->2A): 0.5 .. 4.0
- k2 (A->A+I): 0.01 .. 0.2
- k3 (A+I->I): 0.1 .. 2.0
- dA,dI: 0.001 .. 0.05

Source:
- vent feed inside a sphere, `s_in=0.5 .. 2.0`, `s_out=0.0 .. 0.05`

Noise:
- eta_scale = 0.1 .. 1.0 (start 0.25)
- if you see salt-and-pepper speckle with no structures, reduce eta_scale
- clamp negative values to 0 post-step

dt:
- for the diffusion example above, CFL bound is ~ `dt <= 1/(6*max(D)) = ~0.166`
- start with dt=0.02 .. 0.05 (or do 2–4 substeps if you need a larger displayed step)

