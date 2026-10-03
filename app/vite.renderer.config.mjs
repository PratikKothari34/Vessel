import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// app/package.json now declares "type": "module", so this file is real ESM
// and `__dirname` does not exist in it. vite happens to transpile the config,
// which masked that - don't rely on it.
const here = dirname(fileURLToPath(import.meta.url));

/* The app's only UI build. Tauri compiles `out/renderer` straight into the
   .exe (tauri.conf.json `frontendDist` + the `bundled-ui` feature), and
   serves this same tree from `devUrl` in a dev build.

   Originally written to reproduce electron-vite 2.3.0's renderer preset
   exactly, so the switch away from it was verifiable by hash rather than by
   eye. Electron went on 2026-10-03; these options stay because each was
   re-checked on its own merits, not because the preset set them. The comments
   say why each one earns its place -- changing any of them moves every asset
   hash. */

export default defineConfig({
  root: resolve(here, 'src/renderer'),

  // Relative asset URLs. A release build loads the page from `tauri://
  // localhost`, a non-http origin where an absolute `/assets/...` resolves to
  // the filesystem root and every asset 404s. Still required with one host:
  // this single character is what makes the bundle origin-agnostic.
  base: './',

  plugins: [react()],

  // Serves `devUrl` from tauri.conf.json. The port is fixed and `strictPort`
  // is on deliberately: the binary has that URL compiled in (see the
  // bundled-ui note in src-tauri/Cargo.toml), so vite silently falling back
  // to 5174 would leave the window on ERR_CONNECTION_REFUSED with nothing in
  // either log to explain it. Better to fail loudly here.
  //
  // `host` is deliberately left unset. vite then binds 127.0.0.1 only, which
  // is what contains CVE-wise the three dev-server advisories open against
  // vite 5.4.21 as of 2026-10-03 (esbuild CORS, optimized-deps path traversal,
  // launch-editor UNC/NTLMv2 on Windows): each needs a remote origin to reach
  // this port. The patched line is vite 8, a Rolldown rewrite that moves every
  // asset hash -- not a change to make without re-establishing the byte
  // comparison. Do not add `host: true` or `--host` to make the dev server
  // reachable from another machine without revisiting all three.
  server: {
    port: 5173,
    strictPort: true,
  },

  build: {
    outDir: resolve(here, 'out/renderer'),
    emptyOutDir: true,

    // Inherited from Electron 31's engine. Electron is gone, so this was
    // measured again on 2026-10-03 to see whether the floor still earns its
    // place: chrome126, chrome130, chrome140 and chrome150 all build
    // BYTE-IDENTICAL output, so nothing here is being down-levelled and
    // raising the number buys exactly nothing.
    //
    // `esnext` is the only value that differs, and it is worse. It makes vite
    // skip the esbuild transform altogether, which is what pretty-prints the
    // rollup output while `minify: false` is set -- 8,166 readable lines
    // become 1,822 dense ones and the file drops 274 kB -> 214 kB. The syntax
    // is identical either way (same async/arrow/??/?. counts), so that 60 kB
    // is purely the indentation that makes a user's stack trace legible.
    // Inside a 20 MB binary that is a trade in the wrong direction; see the
    // `minify` note below, which is the same argument.
    target: 'chrome126',

    // The polyfill exists for browsers without modulepreload. Neither host
    // needs it, and it would prepend dead code to the entry chunk.
    modulePreload: { polyfill: false },

    // Unminified on purpose, inherited from the electron-vite preset: the
    // bundle ships inside a desktop binary, so download size is irrelevant,
    // and a readable stack trace from a user's log is worth more than bytes.
    minify: false,

    // Nothing gzips this bundle in transit; measuring it only slowed the build.
    reportCompressedSize: false,
  },
});
