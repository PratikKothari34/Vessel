/**
 * Every cross-directory path this repo builds at runtime or build time.
 *
 * These are ASSEMBLED paths: `path.join(here, '..', '..', 'src-tauri')`,
 * `"frontendDist": "../app/out/renderer"`, `CARGO_MANIFEST_DIR.parent()`. None
 * of them appears in the source as the string it resolves to, so a grep for the
 * joined path finds every literal reference and silently misses these.
 *
 * That is not hypothetical. Flattening `app/src/renderer/src/` to `app/ui/` on
 * 2026-10-17 updated six references found by searching for `src/renderer` and
 * missed the seventh - `api-transport.test.js` built its path from separate
 * `path.resolve` segments - which failed 17 tests only once the suite ran.
 *
 * So the structure gets a test. Moving a directory now fails HERE, by name,
 * instead of somewhere downstream or not at all.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..', '..');

/** Read a file from the repo, or fail the test naming what is missing. */
function read(rel) {
  const p = path.join(ROOT, rel);
  assert.ok(fs.existsSync(p), `${rel} is missing from the repo`);
  return fs.readFileSync(p, 'utf8');
}

test('the renderer sources sit where vite roots itself', () => {
  const cfg = read('app/vite.renderer.config.mjs');
  const m = cfg.match(/root:\s*resolve\(here,\s*'([^']+)'\)/);
  assert.ok(m, "vite config no longer sets `root` the way this test reads it");

  const root = path.join(ROOT, 'app', m[1]);
  assert.ok(fs.existsSync(root), `vite roots at app/${m[1]}, which does not exist`);

  // index.html is the entry vite loads, and it must sit AT the root.
  const indexHtml = path.join(root, 'index.html');
  assert.ok(fs.existsSync(indexHtml), `no index.html at vite's root (app/${m[1]})`);

  // ...and the module it points at has to be there too. This is the pair that
  // broke in the move: the root changed and the entry's relative src with it.
  const entry = fs.readFileSync(indexHtml, 'utf8').match(/src="(\.\/[^"]+\.jsx)"/);
  assert.ok(entry, 'index.html no longer references a relative .jsx entry');
  assert.ok(
    fs.existsSync(path.join(root, entry[1])),
    `index.html points at ${entry[1]}, which is not beside it`,
  );
});

test('every module the unit tests import is where they say it is', () => {
  // The seventh reference. Both spellings, because the one that broke was the
  // one assembled from segments - a test that only knew the joined form would
  // have reproduced the original miss exactly.
  const dir = path.join(ROOT, 'test', 'unit');
  const joined = /['"`](\.\.\/\.\.\/app\/[^'"`]+\.(?:js|mjs|jsx))['"`]/g;
  const segmented = /path\.resolve\(\s*__dirname\s*,([^)]*)\)/g;

  let checked = 0;
  for (const name of fs.readdirSync(dir).filter((f) => f.endsWith('.test.js'))) {
    const src = fs.readFileSync(path.join(dir, name), 'utf8');

    for (const [, rel] of src.matchAll(joined)) {
      const target = path.resolve(dir, rel);
      assert.ok(fs.existsSync(target), `${name} imports ${rel}, which does not exist`);
      checked += 1;
    }

    for (const [, args] of src.matchAll(segmented)) {
      const parts = [...args.matchAll(/'([^']+)'/g)].map((x) => x[1]);
      if (!parts.includes('app')) continue;
      const target = path.resolve(dir, ...parts);
      assert.ok(
        fs.existsSync(target),
        `${name} assembles ${parts.join('/')}, which does not exist`,
      );
      checked += 1;
    }
  }

  // Without this the loop passes by finding nothing - the failure mode of
  // every "scan the tree and assert" test.
  assert.ok(checked >= 5, `expected to check at least 5 imports, checked ${checked}`);
});

test('tauri loads the bundle vite writes', () => {
  const conf = JSON.parse(read('src-tauri/tauri.conf.json'));
  const dist = conf.build.frontendDist;
  assert.ok(dist.startsWith('../'), `frontendDist ${dist} is no longer repo-relative`);

  const cfg = read('app/vite.renderer.config.mjs');
  const out = cfg.match(/outDir:\s*resolve\(here,\s*'([^']+)'\)/);
  assert.ok(out, "vite config no longer sets `outDir` the way this test reads it");

  assert.equal(
    path.resolve(ROOT, 'src-tauri', dist),
    path.resolve(ROOT, 'app', out[1]),
    'tauri reads a different directory than vite writes',
  );
});

test('the icons tauri bundles are the ones the generator writes', () => {
  const conf = JSON.parse(read('src-tauri/tauri.conf.json'));
  for (const rel of conf.bundle.icon) {
    assert.ok(
      fs.existsSync(path.join(ROOT, 'src-tauri', rel)),
      `tauri.conf.json bundles ${rel}, which is not in the repo`,
    );
  }

  // make-tauri-icons.mjs writes them via join(here, '..', '..', 'src-tauri',
  // 'icons') - assembled, and from a directory that is NOT the one it lives
  // beside, so moving app/build/ breaks it with nothing to grep for.
  const gen = read('app/build/make-tauri-icons.mjs');
  const m = gen.match(/const out = join\(here,([^)]*)\)/);
  assert.ok(m, 'make-tauri-icons.mjs no longer assembles its output path');
  const parts = [...m[1].matchAll(/'([^']+)'/g)].map((x) => x[1]);
  assert.equal(
    path.resolve(ROOT, 'app', 'build', ...parts),
    path.resolve(ROOT, 'src-tauri', 'icons'),
    'the icon generator writes somewhere tauri does not read',
  );
});

test('the Modelfile is one level up from the crate that reads it', () => {
  // modelfile.rs: PathBuf::from(CARGO_MANIFEST_DIR).parent().join("Modelfile").
  // Assembled in Rust, so neither a JS nor a Rust grep for a path finds it.
  const src = read('src-core/src/inference/modelfile.rs');
  assert.match(src, /CARGO_MANIFEST_DIR/, 'modelfile.rs no longer derives its default path');
  assert.match(src, /\.parent\(\)/, 'modelfile.rs no longer goes up from the crate dir');

  assert.ok(
    fs.existsSync(path.resolve(ROOT, 'src-core', '..', 'Modelfile')),
    'no Modelfile where src-core/../Modelfile resolves',
  );
});
