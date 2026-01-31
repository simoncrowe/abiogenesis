# Rule System (Strategy Pattern with PoC Autocatalytic Model)

## Purpose

The rule system defines how local lattice state evolves over time.

The simulation engine MUST NOT encode specific chemistry or dynamics.
Instead, all domain behavior is delegated to a pluggable rule system
selected at runtime.

For the MVP, the **Grey-Scott Reaction–diffusion systems**  MUST be implemented as a proof of concept.

---

## Architectural Requirements

### Decoupling

- The simulation engine MUST depend only on a rule interface / trait.
- Rule systems MUST be swappable at runtime.
- The engine MUST NOT assume:
  - specific species semantics
  - conservation laws
  - reversibility
  - biological meaning

---

### Rule System Responsibilities

A rule system is responsible for:

- defining the per-cell state it operates on
- updating local lattice state each tick
- optionally contributing scalar fields for visualization

A rule system MUST:
- operate locally (cell or small neighborhood)
- be deterministic
- avoid global state mutation

---

## Rule Interface (Conceptual)

A rule system SHALL expose functionality equivalent to:

- initialize_cell(cell_state, position)
- update_cell(cell_state, neighbor_states)
- optional: contribute_scalar(cell_state, accumulator)

The exact Rust trait definition is left to implementation,
but the interface MUST allow:

- inspection of neighboring cells
- local mutation of state
- independence from visualization logic

---

## MVP Rule System: Gray–Scott Reaction–Diffusion Model

### Rationale

The **Gray–Scott reaction–diffusion system** is a canonical artificial
chemistry model known to produce:

- autocatalysis
- spontaneous pattern formation
- self-sustaining localized structures
- division-like and motile patterns (in some regimes)

It is:
- simple
- deterministic
- local
- extensively documented and understood

This makes it an ideal starting point for an autopoiesis-focused MVP.

---

### Model Overview

Each lattice cell stores two scalar quantities:

- U: concentration of substrate (food)
- V: concentration of autocatalyst

The local dynamics are defined by the reactions:

    U + 2V -> 3V
    V -> ∅

Combined with diffusion of U and V across the lattice.

This system is autocatalytic because:
- V catalyzes its own production using U as a resource.

---

### Discrete Lattice Adaptation

For the MVP:

- U and V are stored per lattice cell (float or fixed-point).
- Diffusion is implemented via local neighborhood averaging
  (e.g. 6- or 26-neighbor stencil).
- Reaction terms are applied locally after diffusion.

The update per tick is:

1. Diffuse U and V to neighbors
2. Apply reaction equations locally
3. Clamp values to valid ranges

---

### Parameters (Configurable)

The rule system MUST expose parameters, including:

- diffusion rate of U
- diffusion rate of V
- feed rate of U
- kill rate of V

These parameters MUST be adjustable in code to explore
different behavioral regimes.

---

### Determinism

- No randomness is required.
- Given identical initial conditions and parameters,
  the system MUST evolve deterministically.

---

## Scalar Fields Provided by the Rule System

The Gray–Scott rule system MUST provide at least:

- V concentration as a scalar field (primary isosurface candidate)

Optional additional scalars:
- U concentration
- U + V total density
- |dV/dt| (activity)

These scalars feed directly into the isosurface meshing pipeline.

---

## Visualization Implications

- Isosurfaces extracted from V naturally form:
  - blobs
  - filaments
  - dividing structures
- These structures:
  - emerge without explicit entities
  - are not cells
  - align well with autopoiesis themes

---

## Extensibility Beyond MVP

The rule system architecture MUST make it possible to later add:

- lattice gas collision rules
- discrete particle species
- stochastic variants
- hybrid continuous/discrete models

The Gray–Scott model is a starting point, not a constraint.
