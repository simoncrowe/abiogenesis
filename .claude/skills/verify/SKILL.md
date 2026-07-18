---
name: verify
description: Build/launch/drive recipe for verifying autopoiesis (WebGL2 + WASM sim + Supersonic audio) headlessly.
---

# Verifying autopoiesis

## Serve

The repo ships a Caddyfile that serves the repo root over HTTPS with the
COOP/COEP headers needed for `crossOriginIsolated` (SharedArrayBuffer/threads):

```sh
caddy run --config Caddyfile   # page at https://localhost:8080/web/
```

The WASM pkg is committed at `wasm/web/pkg/` — no build step needed unless
Rust sources changed (then `wasm-pack`, see repo README).

## Drive headless

Headless Chromium works for WebGL2 *and* WebAudio/AudioWorklet with:

```sh
chromium --headless=new --remote-debugging-port=9222 --no-sandbox \
  --user-data-dir=<tmp-profile> --ignore-certificate-errors \
  --autoplay-policy=no-user-gesture-required \
  --use-angle=swiftshader --enable-unsafe-swiftshader about:blank
```

Gotchas:
- `--disable-gpu` kills WebGL2 and `main()` aborts before wiring the audio
  buttons — use SwiftShader instead.
- Use a fresh `--user-data-dir` or the browser serves stale cached JS modules.
- Launch chromium and caddy with `setsid ... < /dev/null &` so they survive
  the shell; they get reaped between tool calls otherwise.
- No node/deno on this machine. Drive CDP with python + `websocket-client`
  (venv in scratchpad). `Runtime.evaluate` with `awaitPromise` covers
  everything: click `#audioBoot`, poll `#audioStatus` for `(ready)` (boot
  fetches wasm+samples from unpkg, allow ~60s), click `#audioMusic`.

## Observe audio

- `#audioStats` div shows smoothed voxel stats and the soundscape conductor
  state (cutoff/room/mix/anchor/scene) — updated ~10Hz.
- For actual audio evidence, in page context:
  `const m = await import('./synth.js'); const s = m.getSupersonic();`
  then connect an AnalyserNode to `s.node` and compute RMS.
- scsynth node tree: listen on `s.on('in:text', ...)` and send
  `s.send('/g_queryTree', 0, 0)` — the reply lists group 100 (voices) and
  group 101 (fx chain).
