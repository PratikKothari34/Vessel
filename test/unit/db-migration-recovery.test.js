'use strict';

/**
 * The plaintext->encrypted migration's crash window.
 *
 * The migration swaps two files: rename(main -> backup), then
 * rename(tmp -> main). A process that dies between them leaves the main DB file
 * ABSENT while `.encrypting` holds the finished encrypted copy. Nothing
 * downstream notices on its own -- isPlaintextDb sees no file and returns
 * false, so no migration runs, and the encrypted open then creates a fresh
 * EMPTY database in the gap. The user's library is still on disk, in two files
 * the app never opens again, and the boot that lost it printed nothing.
 *
 * These cover the recovery added 2026-09-30 and the zero-byte guard the Rust
 * track already had. They use plain files rather than real databases: the
 * function under test is pure filesystem bookkeeping, and driving it with the
 * real driver would need a keychain this process must never touch.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const db = require(path.resolve(__dirname, '..', '..', 'src', 'backend', 'db.js'));
const { completeInterruptedMigration, isPlaintextDb } = db._internals;

function scratch() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vessel-mig-'));
  return { dir, main: path.join(dir, 'scenario.db') };
}

test('a migration interrupted mid-swap is completed rather than lost', () => {
  const { dir, main } = scratch();
  fs.writeFileSync(`${main}.encrypting`, 'ENCRYPTED-COPY');
  fs.writeFileSync(`${main}.plaintext-backup`, 'ORIGINAL');
  // Sidecars of the absent main file describe the plaintext original and would
  // corrupt reads of the encrypted file taking its place.
  fs.writeFileSync(`${main}-wal`, 'stale');
  assert.equal(fs.existsSync(main), false, 'the main file is absent, as after the crash');

  completeInterruptedMigration(main);

  assert.equal(fs.readFileSync(main, 'utf8'), 'ENCRYPTED-COPY', 'the staged copy is promoted');
  assert.equal(fs.existsSync(`${main}.encrypting`), false, 'the staging file is consumed');
  assert.equal(fs.existsSync(`${main}-wal`), false, 'the stale sidecar is cleared');
  assert.equal(
    fs.readFileSync(`${main}.plaintext-backup`, 'utf8'), 'ORIGINAL',
    'the backup is left exactly where it is',
  );
  fs.rmSync(dir, { recursive: true, force: true });
});

test('a staged copy never overwrites a live database', () => {
  // The other half: a crash BEFORE the first rename leaves `.encrypting` beside
  // an intact main file. Promoting it there would replace the live database
  // with a partial copy -- strictly worse than the crash.
  const { dir, main } = scratch();
  fs.writeFileSync(main, 'LIVE-DATABASE');
  fs.writeFileSync(`${main}.encrypting`, 'partial garbage');

  completeInterruptedMigration(main);

  assert.equal(fs.readFileSync(main, 'utf8'), 'LIVE-DATABASE', 'the live file is untouched');
  assert.equal(fs.existsSync(`${main}.encrypting`), false, 'the stale staging file is dropped');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('a zero-byte staging file is discarded, not promoted', () => {
  const { dir, main } = scratch();
  fs.writeFileSync(`${main}.encrypting`, '');

  completeInterruptedMigration(main);

  assert.equal(fs.existsSync(main), false, 'an empty staging file must not become the database');
  assert.equal(fs.existsSync(`${main}.encrypting`), false);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('no staging file means nothing to do', () => {
  const { dir, main } = scratch();
  fs.writeFileSync(main, 'LIVE');
  completeInterruptedMigration(main);
  assert.equal(fs.readFileSync(main, 'utf8'), 'LIVE');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('a zero-byte database is not treated as plaintext', async () => {
  // An empty file opens fine with the plain driver. Reading that as "plaintext"
  // runs a whole migration to produce an empty encrypted database plus a
  // pointless backup of nothing. Guarded without touching a driver: the size
  // check returns before any connect happens.
  const { dir, main } = scratch();
  fs.writeFileSync(main, '');
  assert.equal(await isPlaintextDb(main), false);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('a missing database is not treated as plaintext', async () => {
  const { dir, main } = scratch();
  assert.equal(await isPlaintextDb(main), false);
  fs.rmSync(dir, { recursive: true, force: true });
});
