# abiogenesis

Abiogenesis-inspired pattern exploration MVP:
- 3D Gray–Scott reaction–diffusion simulation (Rust → wasm)
- Isosurface extraction in wasm (marching tetrahedra)
- WebGL 2.0 viewer + free-fly camera (JavaScript)
- Simulation + meshing run in a Web Worker (smooth rendering)

## Build + Run

Prerequisites:
- Rust toolchain
- `wasm-pack`

Build wasm:

- `wasm-pack build wasm --release --target web --out-dir web/pkg`

This generates the wasm bindings at `wasm/web/pkg/` (imported by the web workers).

Serve the repo root with a static server that can set headers.

Because the app uses `SharedArrayBuffer`, it must be `crossOriginIsolated`
(COOP/COEP headers) and served from a secure context (HTTPS).

The included `Caddyfile` is configured to serve the repo root with the
necessary headers and `tls internal` for local HTTPS.

- `caddy run --config ./Caddyfile`

Note: your browser may warn about the certificate until you trust Caddy's local CA.

Open:
- `https://localhost:8080/web/`

## Controls

- Move: `W/A/S/D`
- Look: click the canvas for mouse-look
- Speed: mouse wheel
