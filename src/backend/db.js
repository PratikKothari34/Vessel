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
 */

const path = require('path');
const fs = require('fs');
const { createClient } = require('@libsql/client');

const EMBED_DIM = 768; // nomic-embed-text output dimension

const LOCAL_DB_PATH = process.env.LOCAL_DB_PATH || './data/scenario.db';
const TURSO_DATABASE_URL = (process.env.TURSO_DATABASE_URL || '').trim();
const TURSO_AUTH_TOKEN = (process.env.TURSO_AUTH_TOKEN || '').trim();
const SYNC_ENABLED = Boolean(TURSO_DATABASE_URL && TURSO_AUTH_TOKEN);
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
function makeClient() {
  if (SYNC_ENABLED) {
    return createClient({
      url: localFileUrl(),
      syncUrl: TURSO_DATABASE_URL,
      authToken: TURSO_AUTH_TOKEN,
      syncInterval: SYNC_INTERVAL > 0 ? SYNC_INTERVAL : undefined,
      offline: true,
    });
  }
  return createClient({ url: localFileUrl() });
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
}

// Initialise once. On a synced client, pull the remote state before creating
// the schema so we don't fork from an existing cloud DB.
async function getDb() {
  if (_db) return _db;
  const db = makeClient();
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

// Encode a JS number[] as the JSON string vector32() expects.
function toVectorArg(arr) {
  return `[${arr.join(',')}]`;
}

module.exports = {
  getDb,
  syncNow,
  isSyncEnabled,
  toVectorArg,
  EMBED_DIM,
  _config: { LOCAL_DB_PATH, SYNC_ENABLED, SYNC_INTERVAL, TURSO_DATABASE_URL: SYNC_ENABLED ? '(set)' : '' },
};
