# Autopoiesis Isosurface MVP – Architecture Overview

## Purpose

This project implements a minimal, extensible system for exploring autopoiesis-like
emergent structure using:

- a large lattice-based simulation ("lattice gas"-style)
- swappable local rule systems
- scalar-field-driven isosurface visualization
- WebGL 2.0 rendering via a fullscreen canvas

The system is explicitly NOT a life simulation and does not model cells, organisms,
rigid bodies, gravity, or collisions.

The MVP prioritizes:
- simplicity
- determinism
- architectural flexibility
- visual legibility

---

## High-Level Architecture

The system is divided into four loosely coupled layers:

1. Simulation Core (Rust / wasm)
2. Rule System (pluggable strategies)
3. Visualization Geometry (isosurface extraction)
4. Rendering + Camera (JavaScript + WebGL 2.0)

Data ownership flows outward:
- Simulation state lives in wasm memory
- Geometry buffers are generated in wasm
- JavaScript only reads buffers and renders

---

## Key Architectural Constraints

- World size must be configurable (initial target: 10,000^3 logical cells)
- Rule systems must be swappable at runtime
- Visualization must support multiple meshes from the same lattice state
- Architecture should make time-as-space and backward/forward navigation feasible
  without requiring them in the MVP

---

## Non-Goals (MVP)

- Perfect physical realism
- Reversible computation
- GPU-based simulation
- Biological cell abstraction
