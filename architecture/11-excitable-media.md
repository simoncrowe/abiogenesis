# Excitable Media  

## Purpose

Add an excitable media simulation backend (Barkley / FitzHugh–Nagumo class) to the existing PoC.
This backend provides wave-based, signaling-oriented, proto-neural / proto-circulatory dynamics
on an n^3 lattice, suitable for efficient parallelization (Rust + Rayon, WASM threads) and
3D isosurface extraction via marching tetahedra.

Excitable media fills a niche distinct from:
- Gray–Scott (growth / pattern-forming, plant-like)
- RDME (local stochastic dynamics, animal-like micro-ecologies)
- Cahn–Hilliard (compartments / membranes)

Primary contribution:
- long-range coordination via traveling waves
- refractory dynamics and spatiotemporal memory
- scroll waves and wave turbulence in 3D

Primary output: a single scalar field per voxel (activator or derived) exported to JS as f32.

---

## Overview

Excitable media are reaction–diffusion systems with:
- a fast activator variable
- a slow recovery (inhibitor) variable
- a thresholded nonlinear response

Canonical continuous form (two-field):
- u(x): activator
- v(x): inhibitor / recovery

General structure:
- du/dt = reaction_u(u, v) + D_u * Laplacian(u)
- dv/dt = reaction_v(u, v) + D_v * Laplacian(v)

These systems support:
- excitation when u crosses a threshold
- refractory periods governed by v
- traveling wavefronts and spiral / scroll waves

They are deterministic, local, and stencil-based.

---

## Selected recognized variants

Two widely used, minimal excitable systems:

### Option A: Barkley model (recommended initial implementation)

Barkley equations:
- du/dt = (1/epsilon) * u * (1 - u) * (u - (v + b)/a) + D_u * Laplacian(u)
- dv/dt = u - v + D_v * Laplacian(v)

Properties:
- simple polynomial nonlinearities
- robust wave dynamics
- widely used for 2D/3D spiral and scroll wave studies
- numerically forgiving relative to some alternatives

### Option B: FitzHugh–Nagumo (FHN)

FHN equations (one common form):
- du/dt = u - (u^3)/3 - v + I + D_u * Laplacian(u)
- dv/dt = (u + a - b*v)/tau + D_v * Laplacian(v)

Properties:
- more biologically inspired
- slightly stiffer numerically
- well-known in neuroscience contexts

Initial recommendation:
- Implement Barkley first (fewer parameters, easier tuning)
- Keep interface flexible enough to swap kinetics later

---

## Simulation mode integration

Add a backend variant compatible with the existing simulation trait/interface:

enum SimulationMode {
  GrayScott,
  RDME,
  CahnHilliard,
  ExcitableMedia,
}

Shared responsibilities:
- lattice dimensions and indexing helpers
- periodic boundary support
- stepping loop / dt
- WASM exports for scalar fields
- marching tetrahedra consumes one "aliveness" scalar field

ExcitableMedia backend owns:
- u and v buffers (double-buffered)
- parameters (epsilon, a, b, D_u, D_v, dt, etc.)
- initialization and stepping kernels

---

## Lattice, indexing, boundaries

- Domain: N x N x N
- Target: N = 128
- Boundaries: periodic (recommended for scroll wave persistence)

Indexing:
- i = x + N*(y + N*z)

Neighbor access:
- 6-neighbor stencil for Laplacian (initial)
- extension to 26-neighbor stencil is optional but not required

---

## State and buffers

Core fields:
- u[i]: activator (f32)
- v[i]: inhibitor / recovery (f32)

Buffers:
- u0, v0 : read
- u1, v1 : write

Double-buffering is mandatory to avoid read/write hazards.

Optional derived buffers:
- aliveness[i] (f32), exported to JS

---

## Discrete update (Barkley model)

Let:
- L(u) = Laplacian(u)
- L(v) = Laplacian(v)

Reaction terms (per voxel):
- ru = (1/epsilon) * u * (1 - u) * (u - (v + b)/a)
- rv = u - v

Full update:
- u1[i] = u0[i] + dt * (ru + D_u * L(u0)[i])
- v1[i] = v0[i] + dt * (rv + D_v * L(v0)[i])

Notes:
- epsilon controls timescale separation (small epsilon => fast excitation)
- a, b control excitability threshold and wave morphology
- D_v is often set to 0 or << D_u (inhibitor diffuses slowly or not at all)

---

## Discrete update (FitzHugh–Nagumo, optional alternative)

Reaction terms:
- ru = u - (u^3)/3 - v + I
- rv = (u + a - b*v)/tau

Update:
- u1[i] = u0[i] + dt * (ru + D_u * L(u0)[i])
- v1[i] = v0[i] + dt * (rv + D_v * L(v0)[i])

Only one variant should be active at a time.

---

## Numerical stability and timestep guidance

Excitable media are typically less stiff than Cahn–Hilliard but more sensitive than Gray–Scott.

Guidelines:
- choose dt such that wavefronts propagate smoothly without oscillation
- if waves fragment or numerical noise dominates:
  - reduce dt
  - reduce D_u
  - increase epsilon (Barkley)

Diffusion stability constraint (6-neighbor stencil):
- dt <= h^2 / (6 * max(D_u, D_v))

Explicit Euler is sufficient for PoC-scale simulations.

---

## Parallelization (Rust + Rayon + WASM threads)

Excitable media is trivially parallelizable:

- each voxel update:
  - reads u0, v0 and neighbor values
  - writes u1, v1
- no atomics
- no shared mutable state

Parallel passes per step:
1) parallel over voxels:
   - compute Laplacians
   - compute reaction terms
   - write u1, v1

2) swap buffers

3) parallel over voxels:
   - compute aliveness[i]

This scales well under Rayon and maps cleanly to WASM threads when enabled.

Provide single-thread fallback when threads are unavailable.

---

## Initialization strategies

Must support at least two modes:

1) Random excitation (classic):
- u0 initialized near resting state (e.g., u ~ 0)
- v0 initialized near resting state
- add small random noise to u (or a few random excited voxels)

Produces spontaneous wave nucleation and turbulence.

2) Seeded wave sources:
- set localized regions with elevated u
- background at rest
- optional gradients or obstacles

Produces persistent scroll waves and wave interactions.

Optional:
- spatial heterogeneity in parameters (a, b, epsilon) to create niches

---

## Aliveness scalar output

Provide one exported f32 field per voxel.

Recommended options:

Option A (activator):
- aliveness[i] = u[i]
Simple, intuitive, produces clean wave isosurfaces.

Option B (excitation indicator):
- aliveness[i] = max(u[i] - u_threshold, 0)
Highlights active wavefronts only.

Option C (energy-like):
- aliveness[i] = u[i]^2 + v[i]^2
Shows persistent oscillatory regions.

Initial recommendation:
- use aliveness = u

---

## Parameter sketch (Barkley, non-binding)

These are starting points only; tune empirically.

- epsilon : 0.02 – 0.05
- a       : 0.7 – 0.9
- b       : 0.01 – 0.05
- D_u     : 1.0
- D_v     : 0.0 – 0.2
- dt      : small enough for smooth propagation (e.g., 0.01)

Scroll waves in 3D often require careful tuning of epsilon and a.

---

## Success criteria

Behavioral:
- sustained traveling waves or scroll waves in 3D
- clear refractory behavior (no immediate re-excitation)
- rich spatiotemporal structure over long runs

Technical:
- stable under Rayon parallelism (native and wasm threads)
- no atomics or locks
- consistent behavior across thread counts

---

## Extensions (explicitly not required for PoC)

- coupling excitable media to chemistry (reaction–diffusion fields)
- obstacles / anisotropic diffusion
- parameter evolution or adaptation
- bidirectional coupling to phase fields (active membranes)

These should be layered later without altering the core excitable-media kernel.

---

