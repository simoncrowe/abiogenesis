# architecture/05-cahn-hilliard-phase-field.md

## Purpose

Add a Cahn–Hilliard (CH) phase-field simulation backend to the existing PoC to provide
compartmentalization / droplet / membrane-like dynamics on an n^3 lattice, suitable for 3D isosurface
extraction (marching tetrahedra) and efficient parallel stepping (Rust + Rayon, WASM threads).

CH fills a niche not covered by Gray–Scott or RDME:
- spontaneous phase separation (droplets, bicontinuous phases)
- membrane-like interfaces (large gradients at boundaries)
- protocell-like compartments when later coupled to chemistry

Primary output: a single scalar field per voxel (phi or derived) exported to JS as f32.

---

## Overview

Cahn–Hilliard is a conserved-order-parameter phase separation model:

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

Add a backend variant compatible with the existing simulation trait/interface:

enum SimulationMode {
  GrayScott,
  RDME,
  CahnHilliard,
}

Shared responsibilities:
- lattice dimensions, indexing helpers, periodic boundary support
- stepping loop / dt
- WASM exports for scalar fields (typed array view into wasm memory)
- marching tetrahedra consumes an "aliveness" scalar field

CH backend owns:
- phi buffers (double-buffer)
- optional mu buffer (or compute on the fly)
- parameters (M, kappa, A, dt, etc.)
- initialization and stepping implementation

---

## Lattice, indexing, boundaries

- Domain: N x N x N, periodic boundaries recommended
- Target: N = 128 initially

Indexing:
- i = x + N*(y + N*z)

Neighbors for Laplacian:
- 6-neighbor stencil (recommended initial implementation):
  Laplacian(phi)[i] = sum(phi[neighbors]) - 6*phi[i]
  (scaled by 1/h^2 if using physical spacing h != 1)

Periodic wrap:
- x-1 wraps to N-1, etc.

---

## State and buffers

Core field:
- phi[i] : f32 (or f64 if needed), conserved scalar

Buffers:
- phi0[N^3] read
- phi1[N^3] write

Optional scratch buffers (choose one approach):
A) On-the-fly (minimal memory, more compute):
- compute Laplacian(phi) and Laplacian(mu) within a single kernel without storing mu

B) Two-pass with scratch (more memory, simpler to validate):
- mu[N^3] scratch (or mu0/mu1 if double-buffering mu)
- pass 1: compute mu from phi0
- pass 2: compute phi1 using Laplacian(mu)

Initial recommendation:
- Implement two-pass first (mu scratch) for clarity and correctness
- Optimize to one-pass later if needed

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
- The stable phases are approximately phi ≈ +1 and phi ≈ -1

---

## Discrete update

Let:
- L(phi) = Laplacian(phi)
- L(mu)  = Laplacian(mu)

Pass 1 (compute mu):
- mu[i] = A*(phi0[i]^3 - phi0[i]) - kappa * L(phi0)[i]

Pass 2 (update phi):
- phi1[i] = phi0[i] + dt * M * L(mu)[i]

Conservation:
- CH is mass-conserving in theory; discretization should preserve mean(phi) approximately
- In practice, small drift can occur; track mean(phi) to verify stability

Optional clamping:
- Avoid hard clamping phi in CH; it can distort conservation and interface dynamics
- If numerics blow up, reduce dt or use a more stable integrator instead of clamping

---

## Numerical stability guidance

CH is more numerically sensitive than Gray–Scott due to effectively 4th-order diffusion.

Practical controls:
- Use small dt initially; increase carefully
- If using h = 1 and 6-neighbor Laplacian, dt often needs to be much smaller than for Gray–Scott
- If unstable (checkerboarding / blow-up), reduce dt and/or M and/or kappa

Optional integrator upgrades (future):
- semi-implicit spectral methods (FFT) are standard for CH but not required for PoC
- explicit Euler is acceptable for PoC if dt is conservative

---

## Parallelization (Rust + Rayon + WASM threads)

CH is "embarrassingly parallel" with gather stencils:

- Each voxel write depends only on phi0 (and mu if two-pass)
- No atomics required
- Double-buffer ensures no write hazards

Parallel passes:
1) mu pass:
   parallel over i:
     read phi0 + neighbors
     write mu[i]

2) phi pass:
   parallel over i:
     read mu + neighbors
     write phi1[i]

3) swap phi0/phi1

4) compute aliveness scalar:
   parallel over i:
     aliveness[i] = <definition> from phi0 (or derived)

WASM:
- Use Rayon only if WASM threads + SharedArrayBuffer are available
- Provide single-thread fallback path (same kernels, sequential loops)

---

## Initialization strategies (must support at least two)

1) Spinodal decomposition (classic):
- phi0[i] = phi_mean + noise
- phi_mean in (-1, +1); common starting point is 0.0
- noise small, e.g. uniform in [-0.01, 0.01]

This produces bicontinuous structure then coarsens into droplets/domains.

2) Droplet seeding:
- set background phi = -1 (or +1)
- seed one or multiple spheres with phi = +1 (or -1)
- optional small noise

This produces distinct protocell-like droplets.

Notes:
- Total mean(phi) controls volume fraction of each phase (droplet vs bicontinuous)

---

## Aliveness scalar output (for marching tetrahedra)

Provide one exported f32 field, recommended options:

Option A (interfaces / membranes):
- aliveness[i] = |grad(phi)|  (approx via central differences)
This highlights boundaries and produces membrane-like isosurfaces.

Option B (phase occupancy):
- aliveness[i] = phi
This yields solid droplet surfaces at a chosen iso-value (e.g., phi = 0).

Option C (chemical potential magnitude):
- aliveness[i] = |mu|  (requires mu field)
This highlights active interfacial dynamics but is less interpretable visually.

Initial recommendation:
- Output phi as primary field (simple, robust)
- Optionally add a mode to output |grad(phi)| for membrane visualization

---

## Parameter set (initial, non-binding)

These are starting points; tune empirically for stable dynamics at 128^3.

- A     : 1.0
- kappa : 1.0 (increase for thicker interfaces; decrease for sharper)
- M     : 1.0 (controls coarsening speed)
- dt    : start small (e.g., 0.01 in lattice units), adjust upward cautiously

If coarsening is too fast:
- reduce M or dt

If interfaces are too thin / noisy:
- increase kappa

If separation is weak:
- increase A

---

## Success criteria

Behavioral:
- produces stable phase separation structures (droplets or bicontinuous domains)
- evolution continues over long runs (coarsening / ripening is visible)
- interface surfaces are clean and visually meaningful in marching tetrahedra output

Technical:
- stable under Rayon parallelism (native and wasm threads)
- no atomics, no locks
- mean(phi) remains approximately conserved (monitor drift)

---

## Extensions (explicitly not required for PoC)

1) CH + reactions (protocell coupling):
- add one or more reaction–diffusion fields that preferentially localize in one phase
- make M or kappa depend on local chemistry (active emulsions)

2) Hydrodynamics:
- couple to flow (LBM) for advective transport

3) Semi-implicit integrators:
- improve stability and allow larger dt

These should be designed as add-on modules, keeping CH core loosely coupled.

---
