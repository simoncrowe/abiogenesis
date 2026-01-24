# Replicator-Mutator Reaction-Diffusion 

## Purpose

Add a Replicator–Mutator Reaction–Diffusion (RMRD) ecology backend to the existing PoC.
This backend introduces heredity, competition, mutation, and spatial selection on an n^3 lattice,
while remaining easy to parallelize (Rust + Rayon, WASM threads).

This fills a niche not covered by:
- Gray–Scott (pattern formation without heredity)
- RDME (local stochastic dynamics without explicit lineages)
- Cahn–Hilliard (compartments without selection)
- Excitable media (signaling without reproduction)

Primary contribution:
- spatially stabilized evolution-like dynamics
- coexistence, invasion fronts, extinction, and niche partitioning
- “species” or “strategies” competing for shared resources

Primary output: a single scalar "aliveness" or "biomass" field per voxel exported to JS as f32.

---

## Conceptual model

Each voxel hosts continuous-valued concentrations of multiple replicator types.

Key ingredients:
- replication from shared resources (food / energy)
- decay / death
- mutation between types
- diffusion (spatial mixing)
- optional inhibitors / waste

The model is deterministic but can optionally include local noise.
All updates are stencil-gather + local nonlinear kinetics.

---

## Selected recognized framework

This backend is based on standard forms of:
- spatial replicator equations
- replicator–mutator dynamics
- reaction–diffusion ecology (PDE-based population models)

General continuous form:

For each replicator type i:

dR_i/dt =
  R_i * (fitness_i(R, F, W, ...) - phi_mean)
  + sum_j (M_ij * R_j - M_ji * R_i)
  + D_i * Laplacian(R_i)

Where:
- R_i is concentration of type i
- fitness_i is local growth rate
- phi_mean is optional normalization / carrying-capacity control
- M_ij is mutation rate from j -> i
- D_i is diffusion coefficient

---

## Simulation mode integration

Practical note (current codebase): simulation backends are exposed as separate wasm-bindgen classes (e.g. `GrayScottSimulation`, `CahnHilliardSimulation`) and selected in `web/compute_worker.js` by `strategyId`.

So, implement this backend as:
- `ReplicatorMutatorParams` (wasm-bindgen)
- `ReplicatorMutatorSimulation` (wasm-bindgen)

Shared responsibilities (same contract as the existing backends):
- lattice dimensions, indexing helpers, periodic boundaries
- stepping loop / dt
- export a single f32 scalar field (`v`) per voxel for meshing
- provide chunk min/max ranges for iso culling (computed on publish, not every step)

ReplicatorMutator backend owns:
- replicator buffers `R_k` (double-buffered)
- resource buffer `F` (strongly recommended)
- optional waste/inhibitor buffer `W` (defer for v1 if needed)
- mutation coupling (implemented as a small sparse pattern; see below)
- parameters (growth/decay, diffusion, mutation, dt)
- initialization + stepping kernels

---

## Lattice and indexing

- Domain: N x N x N
- Target: N = 128
- Boundaries: periodic

Indexing:
- i = x + N*(y + N*z)

Stencil:
- 6-neighbor Laplacian per field (initial implementation)

---

## State representation

For performance and simplicity in this PoC, use a fixed, small number of types `K` selected at construction time (recommend default `K=4`, allow up to 8).

Required fields:
- `R[k][i]` : concentration of replicator type `k` at voxel `i`

Strongly recommended (v1):
- `F[i]` : shared food / energy resource

Deferable (v2):
- `W[i]` : waste / inhibitor

All fields stored as `Vec<f32>`.

Buffers:
- double-buffer each evolving field:
  - `R0[k]`, `R1[k]` for each k
  - `F0/F1`

Data layout practicality:
- Prefer **struct-of-arrays**: `Vec<Vec<f32>>` or a single flat `Vec<f32>` with indexing `R[k*n + i]`.
- For wasm+Rayon cache locality and fewer allocations, a single flat `Vec<f32>` is typically best:
  - `r0: Vec<f32>` length `K*N^3`
  - `r1: Vec<f32>` length `K*N^3`
  - `r0[k*n + i]`

---

## Reaction kinetics (example baseline)

### Resource dynamics (optional but recommended)

Feed:
- Ø -> F      rate = s(x,y,z)

Consumption:
- R_i + F -> 2R_i    rate = g_i * R_i * F

Decay:
- R_i -> Ø           rate = d_i * R_i
- F -> Ø             rate = d_F * F

Waste (optional):
- R_i -> R_i + W     rate = w_i * R_i
- R_i + W -> W       rate = h_i * R_i * W
- W -> Ø             rate = d_W * W

This structure:
- creates competition for limited resources
- penalizes overgrowth
- allows local extinction and recolonization

---

## Replicator–mutator coupling

Mutation is modeled as linear coupling between replicator fields.

For practicality/performance in v1:
- implement a **fixed sparse topology** (no dynamic KxK matrix per voxel)
- default to a **linear chain** (nearest-neighbor in type index):
  - `0 <-> 1 <-> 2 <-> ... <-> K-1`

This lets the update do O(K) work per voxel (not O(K^2)).

Recommended parameterization:
- a single scalar `mu` (symmetric neighbor mutation rate)
- per-type asymmetry can be added later if needed

Per type k:
- outflow: `mu * R_k` to each neighbor (1 or 2 neighbors)
- inflow: `mu * R_neighbor`

So for interior types:
- `dR_k += mu*(R_{k-1} + R_{k+1} - 2*R_k)`

Edge types have only one neighbor:
- `dR_0 += mu*(R_1 - R_0)`
- `dR_{K-1} += mu*(R_{K-2} - R_{K-1})`

---

## Diffusion

For each field S ∈ {R_i, F (and optionally W)}:
- `dS/dt += D_S * Laplacian(S)`

Typical ordering:
- `D_F` high (resource spreads easily)
- `D_R` low/medium (replicators spread slowly)

Performance note:
- diffusion dominates cost (6 neighbor reads per field).
- Add the same optimization used in `excitable_media`: **skip Laplacian computation when D==0**.

Practical v1 recommendation:
- use a single shared `D_R` for all replicator types (simplifies UI and reduces parameter surface)
- keep `D_F` separate

Diffusion uses gather stencil only.

---

## Discrete update (explicit Euler)

For each voxel i:

1) Compute Laplacians for all active fields.
2) Compute reaction terms (growth, decay, mutation).
3) Update fields:
   S1[i] = S0[i] + dt * (reaction + diffusion)

4) Enforce non-negativity:
   S1[i] = max(S1[i], 0)

Double-buffer all fields and swap after step.

---

## Optional normalization / carrying capacity

To prevent runaway total biomass:

Option A (resource-limited growth):
- rely on F depletion only (preferred, more physical)

Option B (soft normalization):
- compute total biomass B = sum_i R_i
- apply:
  dR_i/dt -= lambda * R_i * B

Option C (hard cap):
- if B > B_max, scale all R_i proportionally
  (not recommended; distorts dynamics)

Initial recommendation:
- use resource limitation (Option A) only

---

## Noise (optional, parallel-safe)

Optional intrinsic noise can be added locally:

- dR_i += sigma * sqrt(max(R_i, 0)) * Normal(0,1) * sqrt(dt)

Noise is:
- local per voxel
- stateless RNG seeded by (voxel, timestep, type)

Do not add noise to diffusion.

---

## Parallelization (Rust + Rayon + WASM threads)

All updates are local gather + write-self:

- each voxel update reads:
  - local fields
  - neighbors for Laplacian
- each voxel writes:
  - its own output fields only

Parallel passes per step:
1) parallel over voxels:
   - compute Laplacians
   - compute reaction + mutation
   - write next buffers

2) swap buffers

3) parallel over voxels:
   - compute aliveness scalar

No atomics, no locks, deterministic memory access.

Single-thread fallback uses identical code paths.

---

## Initialization strategies

Must support at least three:

1) Uniform low-density + noise:
- all R_i small random values
- F initialized high
- leads to spontaneous symmetry breaking

2) Localized seeding:
- seed different replicator types in different regions
- background empty
- observe invasion fronts and competition

3) Gradient-driven niches:
- spatial gradient in resource feed s(x,y,z)
- observe stable spatial differentiation of types

---

## Aliveness scalar output

Export one f32 per voxel (`v`), consistent with the rest of the codebase.

Recommended options:

Option A (total biomass):
- `v = sum_k R_k`

Option B (metabolic activity):
- `v = sum_k (g_k * R_k * F)`

Initial recommendation (v1):
- `v = sum_k R_k`

Performance note:
- compute `v` only when publishing a keyframe (same as RDME/CH patterns), or compute it incrementally only if needed.
- keep a `v_dirty` flag so stepping can skip `v` work.

---

## Parameter sketch (non-binding)

Typical starting scales:

- K (types): 3–5
- g_i: similar magnitude, small variation between types
- d_i: slightly lower than growth in high-F regions
- mu_ij: 1e-3 – 1e-2 (relative scale)
- D_R: small
- D_F: large
- dt: constrained by diffusion stability

Tune empirically.

---

## Success criteria

Behavioral:
- persistent coexistence of multiple replicator types
- visible invasion fronts and competitive exclusion
- stable spatial niches over long runs

Technical:
- stable under Rayon parallelism (native + wasm threads)
- no atomics, no race conditions
- reasonable performance at 128^3 with K ≤ 8

---

## Extensions (explicitly not required for PoC)

- explicit genotype–phenotype mapping
- adaptive mutation rates
- coupling to Cahn–Hilliard compartments
- evolutionary changes to reaction parameters

These should be layered later without altering the core kernel.

---
