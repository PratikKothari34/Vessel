'use strict';

/**
 * encrypt-existing-db.js — one-time migration: copy an existing PLAINTEXT
 * scenario.db into a new ENCRYPTED database using the keychain key.
 *
 * Why this exists: enabling encryption-at-rest means libSQL opens the file with
 * `encryptionKey`. An existing plaintext file can't be opened that way
 * (SQLITE_NOTADB). This script reads the plaintext DB with no key, recreates the
 * schema + rows in a new file opened WITH the key, then swaps it in. The
 * original is preserved as `<db>.plaintext.bak` so nothing is lost.
 *
 * Run once:  node src/backend/scripts/encrypt-existing-db.js
 * Safe to skip if you have no existing data (fresh installs start encrypted).
 */

const path = require('path');
const fs = require('fs');
const { createClient } = require('@libsql/client');
const keystore = require('../keystore');

const LOCAL_DB_PATH = process.env.LOCAL_DB_PATH || './data/scenario.db';

function fileUrl(p) { return `file:${path.resolve(p)}`; }

// Tables to copy, parent-first so foreign keys resolve on insert.
const TABLES = ['characters', 'conversations', 'turns', 'variants', 'archive'];

async function main() {
  const src = path.resolve(LOCAL_DB_PATH);
  if (!fs.existsSync(src)) {
    console.log(`No existing DB at ${src} — nothing to migrate (fresh installs start encrypted).`);
    return;
  }

  const key = await keystore.getDbEncryptionKey();
  if (!key) {
    console.error('No encryption key available (keychain unreachable and DB_ENCRYPTION_KEY unset). Aborting.');
    process.exitCode = 1;
    return;
  }

  // Verify the source is actually plaintext (and not already encrypted).
  const plain = createClient({ url: fileUrl(src) });
  try {
    await plain.execute('SELECT 1');
  } catch (e) {
    console.error('Could not open the existing DB as plaintext. It may already be encrypted — aborting to avoid data loss.');
    console.error('Detail:', e.message);
    process.exitCode = 1;
    return;
  }

  const tmp = src + '.enc.tmp';
  // Clean any stale temp from a prior aborted run.
  for (const suffix of ['', '-wal', '-shm', '-journal']) fs.rmSync(tmp + suffix, { force: true });

  const enc = createClient({ url: fileUrl(tmp), encryptionKey: key });

  // Recreate the exact schema in the encrypted target by replaying the source's
  // own CREATE statements (tables + indexes, including the vector index).
  const schema = await plain.execute(
    "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
  );
  await enc.execute('PRAGMA foreign_keys = OFF;');
  for (const row of schema.rows) {
    await enc.execute(row.sql);
  }

  // Copy rows table-by-table.
  let total = 0;
  for (const t of TABLES) {
    let rows;
    try { rows = await plain.execute(`SELECT * FROM ${t}`); }
    catch { continue; } // table may not exist in older DBs
    for (const r of rows.rows) {
      const cols = Object.keys(r);
      const placeholders = cols.map(() => '?').join(', ');
      const args = cols.map((c) => r[c]);
      await enc.execute({
        sql: `INSERT INTO ${t} (${cols.map((c) => `"${c}"`).join(', ')}) VALUES (${placeholders})`,
        args,
      });
      total++;
    }
    console.log(`  copied ${rows.rows.length} rows from ${t}`);
  }
  await enc.execute('PRAGMA foreign_keys = ON;');

  // Close both clients. NOTE: libSQL keeps the native file handle open for the
  // life of THIS process even after close(), so renaming the source here throws
  // EBUSY on Windows. The fix: this process exits, then a *fresh* process (with
  // no open handles) performs the pure-filesystem swap via `--swap-only`.
  try { plain.close(); } catch { /* ignore */ }
  try { enc.close(); } catch { /* ignore */ }

  console.log(`\nMigrated ${total} rows into the encrypted temp DB (${path.basename(tmp)}).`);
  console.log('Finalizing the swap in a clean process...');

  // Re-exec the swap phase in a new process AFTER this one fully exits.
  const { spawn } = require('child_process');
  const child = spawn(process.execPath, [__filename, '--swap-only'], {
    detached: true, stdio: 'inherit',
  });
  child.unref();
}

// Phase 2: pure filesystem swap, run in a clean process (no libSQL handles).
// Retries each rename because the parent process may take a moment to fully
// exit and release the source file's handle on Windows.
async function swapOnly() {
  const src = path.resolve(LOCAL_DB_PATH);
  const tmp = src + '.enc.tmp';
  const bak = src + '.plaintext.bak';
  if (!fs.existsSync(tmp)) {
    console.error('No encrypted temp DB found — run the migration first.');
    process.exitCode = 1;
    return;
  }

  const renameWithRetry = async (from, to) => {
    for (let i = 0; i < 40; i++) {
      try { fs.renameSync(from, to); return; }
      catch (e) {
        if ((e.code === 'EBUSY' || e.code === 'EPERM') && i < 39) {
          await new Promise((r) => setTimeout(r, 250));
          continue;
        }
        throw e;
      }
    }
  };

  for (const suffix of ['', '-wal', '-shm', '-journal']) {
    const from = src + suffix;
    if (fs.existsSync(from)) await renameWithRetry(from, bak + suffix);
  }
  await renameWithRetry(tmp, src);
  for (const suffix of ['-wal', '-shm', '-journal']) {
    if (fs.existsSync(tmp + suffix)) await renameWithRetry(tmp + suffix, src + suffix);
  }
  console.log('Swap complete. Encrypted DB is now scenario.db; original preserved as scenario.db.plaintext.bak.');
  console.log('Verify the app works, then you may delete the .plaintext.bak file.');
}

if (process.argv.includes('--swap-only')) {
  swapOnly().catch((e) => { console.error('Swap failed:', e.message); process.exitCode = 1; });
} else {
  main().catch((e) => {
    console.error('Migration failed:', e.message);
    process.exitCode = 1;
  });
}
