/**
 * Loaded with --import before every test file, in every test worker.
 *
 * As of 2026-10-03 the Node backend is gone and the only tests left here are
 * renderer unit tests over pure modules -- they open no database, read no
 * keychain and look at no environment. So nothing in the current suite needs
 * this file.
 *
 * It stays because of what it defends against, which did not go away: the
 * keystore talks to the REAL OS keychain and LOCAL_DB_PATH defaults to the
 * user's real database. A test that reached either would overwrite live data,
 * and the failure is silent. This runs early enough to make that impossible --
 * it is the only hook evaluated before the first test file -- so a future test
 * that does open something starts out pointed somewhere safe.
 *
 * Delete it only along with the last test that could ever touch persistent
 * state, not merely because today's tests do not.
 */

import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function insideRepo(p) {
  const rel = path.relative(PROJECT_ROOT, path.resolve(p));
  return Boolean(rel) && !rel.startsWith('..') && !path.isAbsolute(rel);
}

// Presence-only check in the data layer, so any value disables cloud sync.
process.env.VESSEL_NO_SYNC = '1';

// dotenv does not overwrite a key that is already present -- '' counts as
// present -- so these stay empty even if a stray .env is ever in reach.
process.env.TURSO_DATABASE_URL = '';
process.env.TURSO_AUTH_TOKEN = '';

// Never let a test process generate or read a real key.
if (!process.env.DB_ENCRYPTION_KEY) {
  process.env.DB_ENCRYPTION_KEY = crypto.randomBytes(32).toString('hex');
}

const current = process.env.LOCAL_DB_PATH;
if (!current || insideRepo(current)) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vessel-guard-'));
  process.env.LOCAL_DB_PATH = path.join(dir, 'guard.db');
}
