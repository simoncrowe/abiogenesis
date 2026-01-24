# Continuous Cellular Automata (Lenia)

## Purpose

Add a Continuous Cellular Automata (CCA) backend of the Lenia / smooth-CA class to the existing PoC.
This backend explores life-like self-maintaining, motile, and adaptive-looking structures on an n^3 lattice,
while remaining highly parallelizable (Rust + Rayon, WASM threads).

This fills a niche distinct from:
- Gray–Scott (reaction–diffusion growth patterns)
- RDME (stochastic micro-ecologies)
- Cahn–Hilliard (compartments / membranes)
- Excitable media (wave signaling)
- Replicator–mutator ecology (selection and heredity)

Primary contribution:
- persistent self-organized “organisms”
- motion, deformation, and interaction without explicit agents
- minimal rule set with rich emergent behavior

Primary output: a single scalar field per voxel exported to JS as f32.

---

## Conceptual model

Continuous cellular automata generalize classic CA by:
- replacing discrete states with continuous values
- replacing hard neighborhood rules with smooth convolution kernels
- applying nonlinear growth functions

General update:
1) compute neighborhood influence via convolution
2) apply a nonlinear growth function
3) integrate in time

State is continuous and bounded (typically [0, 1]).

---

## Selected recognized framework

This backend is inspired by:
- Lenia (Bert Wang-Chak Chan, 2019)
- SmoothLife (Rafler, 2011)

Core components:
- state field A(x) ∈ [0,1]
- convolution kernel K(r)
- growth function G(u)

These are standard and well-documented in the literature.

---

## Simulation mode integration

Practical note (current codebase): simulation backends are exposed as separate wasm-bindgen classes and selected in `web/compute_worker.js` by `strategyId`.

So implement this backend as:
- `LeniaParams` (wasm-bindgen)
- `LeniaSimulation` (wasm-bindgen)

Shared responsibilities (same contract as existing backends):
- lattice dimensions and indexing helpers
- stepping loop / dt
- export a single f32 scalar field (`v`) per voxel for meshing
- provide chunk min/max ranges for iso culling (computed on publish)

Lenia backend owns:
- state buffers `a0`, `a1`
- kernel offsets+weights (precomputed once per sim, reused each step)
- growth function parameters
- initialization and stepping kernels

---

## Lattice, indexing, boundaries

- Domain: N x N x N
- Target: N = 128
- Boundaries: periodic (recommended)

Indexing:
- i = x + N*(y + N*z)

Neighborhood:
- spherical neighborhood defined by kernel radius R
- isotropic weighting preferred for life-like behavior

---

## State and buffers

Core field:
- A[i] : f32, typically constrained to [0, 1]

Buffers:
- A0[N^3] read
- A1[N^3] write

Double-buffering is mandatory.

Optional derived buffer:
- aliveness[i] (exported to JS)

---

## Convolution kernel

The kernel defines neighborhood influence.

General form:
- `K(r)` where `r` is distance from center voxel
- `K(r)` is radially symmetric and compactly supported (`r <= R`)

Discrete implementation (practical for this repo):
- precompute a list of **integer offsets** `(dx,dy,dz)` within radius `R`
- precompute a corresponding list of **weights** `w_j`
- normalize weights so `sum(w_j)=1`

At runtime, for each voxel:
- `U = sum_j w_j * A0[x+dx_j, y+dy_j, z+dz_j]`

Performance note:
- keep `R` modest. In 3D, neighborhood size grows as O(R^3).
- for `N=128`, start with `R=4..6`.

Implementation note (avoid per-voxel modulo):
- store offset indices as **wrapped index deltas** per axis, or compute neighbor coordinates using the precomputed `x_minus/x_plus` style tables.
- simplest v1: store `(dx,dy,dz,w)` and use integer wrap per axis; then optimize if needed.

---

## Growth function

Growth function maps neighborhood influence to state change.

Standard Lenia growth:
- G(u) = 2 * exp(-((u - mu)^2) / (2 * sigma^2)) - 1

Where:
- u is convolution result at voxel
- mu is preferred neighborhood activation
- sigma controls tolerance

Other smooth bell-shaped functions are acceptable.

---

## Discrete update

For each voxel i:

1) Compute neighborhood influence:
   U = sum_j K(j) * A0[i + offset_j]

2) Compute growth:
   g = G(U)

3) Integrate:
   A1[i] = A0[i] + dt * g

4) Clamp:
   A1[i] = clamp(A1[i], 0, 1)

Notes:
- dt controls responsiveness; too large causes instability
- clamp is essential to keep values bounded

---

## Numerical stability guidance

- dt should be small enough that A changes smoothly
- kernel radius R and dt jointly affect stability
- if patterns explode or vanish:
  - reduce dt
  - reduce kernel amplitude
  - adjust mu and sigma

Lenia is typically forgiving numerically.

---

## Parallelization (Rust + Rayon + WASM threads)

Lenia is parallel-friendly:
- convolution is gather-only
- each voxel writes only to itself
- no atomics, no locks

Practical implementation detail for this codebase:
- compute `v` lazily on publish (via `v_dirty`) rather than every step
- compute chunk min/max ranges on publish (`recompute_chunk_ranges_from_v`)

Parallel passes per step:
1) parallel over voxels (z-slices):
   - compute convolution
   - compute growth
   - write `a1`

2) swap buffers, mark `v_dirty=true`

Kernel offsets and weights are read-only and shared.

Single-thread fallback uses the same kernel.

---

## Initialization strategies

Must support at least three:

1) Random noise:
- A0[i] ∈ small random range near 0
- leads to spontaneous emergence or extinction depending on parameters

2) Seeded organisms:
- place small blobs of A ≈ 1 in empty background
- observe persistence, motion, replication-like behavior

3) Imported seeds:
- predefined patterns loaded into the lattice
- useful for reproducible experiments

---

## Aliveness scalar output

Export one f32 per voxel (`v`) for meshing.

Recommended (v1):
- `v = A`

Alternative (motion emphasis):
- `v = |A1 - A0|` (temporal activity)

Practical note:
- keep a `v_dirty` flag and compute `v` only when publishing a keyframe.

Initial recommendation:
- `v = A`

---

## Kernel and parameter sets

Provide a mechanism to define multiple "species" via parameter presets:
- kernel radius R
- kernel shape parameters
- mu, sigma
- dt

Allow runtime switching of parameter sets.

---

## Performance considerations

- convolution cost scales with kernel size
- keep R modest (e.g., 3–8 lattice units) for 128^3
- precompute kernel offsets once

Memory footprint is minimal (two f32 fields + kernel data).

---

## Success criteria

Behavioral:
- persistent, localized, self-maintaining structures
- motion and interaction between structures
- diversity of behaviors across parameter sets

Technical:
- stable under Rayon parallelism (native and wasm threads)
- no atomics, no race conditions
- acceptable performance at 128^3

---

## Extensions (explicitly not required for PoC)

- multi-channel Lenia (multiple interacting fields)
- adaptive kernels or growth functions
- evolutionary search over parameter space
- coupling to other backends (e.g., CH compartments)

These should be layered later without altering the core kernel.

---
