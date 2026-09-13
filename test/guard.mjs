/**
 * Loaded with --import before every test file, in every test worker.
 *
 * Unit tests require the backend modules directly, and several of them capture
 * configuration at require time -- db.js and settings.js both read
 * LOCAL_DB_PATH at module scope, which is why setting it from inside a test
 * after the require has already happened does nothing at all.
 *
 * So the safe value has to be in place before any test file is evaluated, and
 * this is the only hook that runs that early. It is a backstop, not the primary
 * mechanism: nothing in the unit tests opens a database, and the integration
 * tests pass an explicit environment to a spawned child. It exists so that a
 * future test which does open one cannot reach the real file by accident.
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

// Presence-only in db.js, so any value disables cloud sync.
process.env.VESSEL_NO_SYNC = '1';

// dotenv does not overwrite a key that is already present -- '' counts as
// present -- so these stay empty even if a stray .env is ever in reach.
process.env.TURSO_DATABASE_URL = '';
process.env.TURSO_AUTH_TOKEN = '';

// Never let keystore generate or read a real key for a test process.
if (!process.env.DB_ENCRYPTION_KEY) {
  process.env.DB_ENCRYPTION_KEY = crypto.randomBytes(32).toString('hex');
}

const current = process.env.LOCAL_DB_PATH;
if (!current || insideRepo(current)) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vessel-guard-'));
  process.env.LOCAL_DB_PATH = path.join(dir, 'guard.db');
}
