'use strict';

/**
 * db.js — Turso / libSQL data layer for Scenario_Chat.
 *
 * Local-first: the source of truth is a local SQLite file (LOCAL_DB_PATH).
 * If TURSO_DATABASE_URL + TURSO_AUTH_TOKEN are set, the client becomes an
 * embedded replica with OFFLINE WRITES — writes hit the local file first and
 * sync to Turso cloud in the background (backup + multi-device). With no Turso
 * creds the app is 100% local and needs no account.
 *
 * Vector search is native (libSQL): archived turns store a 768-dim F32_BLOB
 * embedding indexed with libsql_vector_idx; retrieval uses vector_top_k +
 * vector_distance_cos. (Verified against @libsql/client 0.14.0.)
 *
 * Encryption at rest: the local SQLite file is opened with libSQL's native
 * `encryptionKey` (whole-file, transparent — vector search still works because
 * decryption happens in-process). The key comes from the OS keychain via
 * keystore.js; with no key available the DB opens unencrypted (local-only,
 * no-keychain fallback). The Turso cloud leg is already TLS (https/libsql://).
 */

const path = require('path');
const fs = require('fs');
const { createClient } = require('@libsql/client');
const keystore = require('./keystore');

const EMBED_DIM = 768; // nomic-embed-text output dimension

const LOCAL_DB_PATH = process.env.LOCAL_DB_PATH || './data/scenario.db';
const TURSO_DATABASE_URL = (process.env.TURSO_DATABASE_URL || '').trim();
// Sync is *possible* if a URL is configured; the token is resolved at connect
// time from the keychain (with .env fallback), so we confirm SYNC_ENABLED then.
const TURSO_URL_SET = Boolean(TURSO_DATABASE_URL);
let SYNC_ENABLED = false;
let _encryptedAtRest = false;
const SYNC_INTERVAL = (() => {
  const v = parseInt(process.env.TURSO_SYNC_INTERVAL, 10);
  return Number.isFinite(v) && v >= 0 ? v : 60;
})();

let _db = null;
let _syncTimer = null;

function localFileUrl() {
  // libSQL wants a file: URL; resolve to an absolute path so cwd doesn't matter.
  const abs = path.resolve(LOCAL_DB_PATH);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  return `file:${abs}`;
}

// Build the client. Embedded replica (offline writes) when synced, else a plain
// local file. syncUrl + authToken turn the local file into a replica of the
// remote primary; offline:true makes local writes authoritative until synced.
// `encryptionKey` (when present) encrypts the LOCAL replica file at rest only —
// it is independent of the remote Turso DB (which is managed/encrypted by Turso
// and reached over TLS). The key is machine-local (keychain), so an encrypted
// local file is NOT portable to another machine by copy; use sync for that.
function makeClient({ authToken, encryptionKey }) {
  const synced = Boolean(TURSO_URL_SET && authToken);
  const enc = encryptionKey ? { encryptionKey } : {};
  if (synced) {
    return createClient({
      url: localFileUrl(),
      syncUrl: TURSO_DATABASE_URL,
      authToken,
      syncInterval: SYNC_INTERVAL > 0 ? SYNC_INTERVAL : undefined,
      offline: true,
      ...enc,
    });
  }
  return createClient({ url: localFileUrl(), ...enc });
}

// Schema. Vector index is created on the archive embedding column so retrieval
// can use vector_top_k. FKs cascade so deleting a character/conversation cleans
// up its turns + archive.
async function initSchema(db) {
  await db.execute('PRAGMA foreign_keys = ON;');

  await db.execute(`
    CREATE TABLE IF NOT EXISTS characters (
      id         TEXT PRIMARY KEY,
      name       TEXT NOT NULL,
      avatar     TEXT DEFAULT '',
      persona    TEXT DEFAULT '',
      greeting   TEXT DEFAULT '',
      sampling   TEXT DEFAULT '{}',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
  `);

  await db.execute(`
    CREATE TABLE IF NOT EXISTS conversations (
      id           TEXT PRIMARY KEY,
      character_id TEXT REFERENCES characters(id) ON DELETE CASCADE,
      title        TEXT DEFAULT '',
      summary      TEXT DEFAULT '',
      created_at   TEXT NOT NULL,
      updated_at   TEXT NOT NULL
    );
  `);

  // verbatim recent turns (kept in full in the live window)
  await db.execute(`
    CREATE TABLE IF NOT EXISTS turns (
      id              INTEGER PRIMARY KEY AUTOINCREMENT,
      conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role            TEXT NOT NULL,
      content         TEXT NOT NULL,
      created_at      TEXT NOT NULL
    );
  `);
  await db.execute(
    `CREATE INDEX IF NOT EXISTS turns_conv_idx ON turns(conversation_id, id);`
  );

  // Swipe variants: alternate generations for an assistant turn. The parent
  // turns.content always mirrors the ACTIVE variant (so memory reads turns
  // unchanged). Only the latest assistant turn is regenerated in practice, but
  // variants persist so you can swipe between them after reopening the chat.
  await db.execute(`
    CREATE TABLE IF NOT EXISTS variants (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      turn_id    INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
      content    TEXT NOT NULL,
      is_active  INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL
    );
  `);
  await db.execute(
    `CREATE INDEX IF NOT EXISTS variants_turn_idx ON variants(turn_id, id);`
  );

  // archive: frozen older turns + embedding for retrieval
  await db.execute(`
    CREATE TABLE IF NOT EXISTS archive (
      id              INTEGER PRIMARY KEY AUTOINCREMENT,
      conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role            TEXT NOT NULL,
      content         TEXT NOT NULL,
      embedding       F32_BLOB(${EMBED_DIM}),
      created_at      TEXT NOT NULL
    );
  `);
  await db.execute(
    `CREATE INDEX IF NOT EXISTS archive_conv_idx ON archive(conversation_id, id);`
  );
  // Native vector index for approximate nearest-neighbour retrieval.
  await db.execute(
    `CREATE INDEX IF NOT EXISTS archive_vec_idx ON archive(libsql_vector_idx(embedding));`
  );

  await runMigrations(db);
}

// Additive migrations for existing DBs. Each is guarded so re-running is safe.
async function runMigrations(db) {
  const cols = await db.execute(`PRAGMA table_info(characters);`);
  const has = (name) => cols.rows.some((r) => r.name === name);
  const addCol = async (name, ddl) => { if (!has(name)) await db.execute(`ALTER TABLE characters ADD COLUMN ${ddl};`); };

  // narration vs dialogue control
  await addCol('response_style', `response_style TEXT DEFAULT 'balanced'`);
  // c.ai-style profile fields
  await addCol('tagline', `tagline TEXT DEFAULT ''`);        // short hook under the name
  await addCol('about', `about TEXT DEFAULT ''`);            // longer public blurb
  await addCol('chat_starters', `chat_starters TEXT DEFAULT '[]'`); // JSON array of opening prompts
  await addCol('tags', `tags TEXT DEFAULT '[]'`);           // JSON array of category labels
}

// Initialise once. On a synced client, pull the remote state before creating
// the schema so we don't fork from an existing cloud DB.
async function getDb() {
  if (_db) return _db;

  // Resolve secrets from the OS keychain (with env fallbacks) before connecting.
  const [encryptionKey, authToken] = await Promise.all([
    keystore.getDbEncryptionKey(),
    keystore.getTursoToken(),
  ]);
  _encryptedAtRest = Boolean(encryptionKey);
  SYNC_ENABLED = Boolean(TURSO_URL_SET && authToken);
  if (!_encryptedAtRest) {
    console.warn('[db] local database is NOT encrypted at rest (no keychain key and no DB_ENCRYPTION_KEY).');
  }

  const db = makeClient({ authToken, encryptionKey });
  if (SYNC_ENABLED) {
    try { await db.sync(); } catch (e) {
      console.warn('[db] initial sync failed (continuing local-first):', e.message);
    }
  }
  await initSchema(db);
  _db = db;

  // Background sync heartbeat (syncInterval already auto-syncs, but this also
  // pushes local offline writes up on a steady cadence).
  if (SYNC_ENABLED && SYNC_INTERVAL > 0) {
    _syncTimer = setInterval(() => {
      db.sync().catch((e) => console.warn('[db] periodic sync failed:', e.message));
    }, SYNC_INTERVAL * 1000);
    if (_syncTimer.unref) _syncTimer.unref();
  }
  return _db;
}

// Force a sync now (call on shutdown / on demand). No-op when local-only.
async function syncNow() {
  if (!_db || !SYNC_ENABLED) return false;
  await _db.sync();
  return true;
}

function isSyncEnabled() {
  return SYNC_ENABLED;
}

function isEncryptedAtRest() {
  return _encryptedAtRest;
}

// Encode a JS number[] as the JSON string vector32() expects.
function toVectorArg(arr) {
  return `[${arr.join(',')}]`;
}

module.exports = {
  getDb,
  syncNow,
  isSyncEnabled,
  isEncryptedAtRest,
  toVectorArg,
  EMBED_DIM,
  // Live snapshot — sync/encryption are resolved at getDb() time, so read these
  // via getters rather than capturing module-load values.
  get _config() {
    return {
      LOCAL_DB_PATH,
      SYNC_ENABLED,
      SYNC_INTERVAL,
      ENCRYPTED_AT_REST: _encryptedAtRest,
      TURSO_DATABASE_URL: TURSO_URL_SET ? '(set)' : '',
    };
  },
};
