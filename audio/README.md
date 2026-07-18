# Audio SynthDefs

SuperCollider sources for the custom granular-soundscape SynthDefs
(see [architecture/15-granular-ambient.md](../architecture/15-granular-ambient.md)).

- `synthdefs/*.scd` — one SynthDef per file; the SynthDef is the file's
  value (no side effects), so the compile script can load and write it.
- Compiled `.scsyndef` binaries are committed to `web/synthdefs/` and
  loaded by `web/synth.js` at boot.

## Regenerating the binaries

Requires SuperCollider (`sclang` on PATH; tested with 3.14):

```sh
sclang audio/compile_synthdefs.scd
```

This writes `web/synthdefs/<name>.scsyndef` for every source file and
exits. Commit the regenerated binaries alongside any `.scd` change.
