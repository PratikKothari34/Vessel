'use strict';

/**
 * db.js — Turso data layer for Vessel (@tursodatabase/sync).
 *
 * Local-first: the source of truth is a local SQLite file (LOCAL_DB_PATH). If
 * a Turso database URL + auth token are configured, the client also syncs to
 * Turso cloud via the push/pull protocol (backup + multi-device). With no
 * Turso creds the app is 100% local and needs no account.
 *
 * Credentials come from the user, not the build: the URL from data/settings.json
 * (set in-app via PUT /settings; falls back to TURSO_DATABASE_URL in dev) and
 * the token from the OS keychain (falls back to TURSO_AUTH_TOKEN). Nothing is
 * baked into the installer.
 *
 * NOTE (2026-07): migrated off @libsql/client. Turso retired the old embedded-
 * replica sync protocol (syncUrl + offline:true) server-side, so this uses the
 * new @tursodatabase/sync engine (explicit push()/pull()). That engine has NO
 * native vector search (no libsql_vector_idx / vector_top_k), so retrieval moved
 * to in-JS cosine over stored embeddings (see memory.js retrieve()). Embeddings
 * are stored as raw little-endian Float32 blobs.
 *
 * The rest of the backend still calls db.execute({sql,args}) / .rows /
 * .rowsAffected — a compatibility shim below preserves that contract over the
 * new better-sqlite3-style API (prepare/all/get/run), so only db.js + the vector
 * query in memory.js changed.
 *
 * Encryption at rest: the local SQLite file is encrypted with libSQL-style
 * whole-file encryption (aes256gcm) via the new engine's `encryption` option.
 * The key comes from the OS keychain via keystore.js; with no key available the
 * DB opens unencrypted (local-only, no-keychain fallback). The Turso cloud leg
 * is TLS (https/libsql://).
 */

const path = require('path');
const fs = require('fs');
const keystore = require('./keystore');
const settings = require('./settings');

// @tursodatabase/sync is ESM-only ("type":"module"). This backend is CommonJS,
// and in the packaged app it runs under Electron's bundled Node 20, where a
// static require() of an ES module throws ERR_REQUIRE_ESM. Load it via a
// memoized dynamic import() instead — the only form that works across Node 20
// (packaged) and Node 22+ (dev).
let _connect = null;
async function getConnect() {
  if (!_connect) {
    ({ connect: _connect } = await import('@tursodatabase/sync'));
  }
  return _connect;
}

const EMBED_DIM = 768; // nomic-embed-text output dimension

const LOCAL_DB_PATH = process.env.LOCAL_DB_PATH || './data/scenario.db';

// Sync URL: settings.json wins when its key is present (the in-app Settings
// panel wrote it; '' there means the user explicitly disabled sync). An absent
// key falls back to the env var so dev .env setups keep working.
function resolveSyncUrl() {
  const s = settings.load();
  if (typeof s.tursoUrl === 'string') return s.tursoUrl.trim();
  return (process.env.TURSO_DATABASE_URL || '').trim();
}
let _syncUrlInUse = ''; // what this boot actually connected with
let SYNC_ENABLED = false;
let _encryptedAtRest = false;
const SYNC_INTERVAL = (() => {
  const v = parseInt(process.env.TURSO_SYNC_INTERVAL, 10);
  return Number.isFinite(v) && v >= 0 ? v : 60;
})();

let _db = null;      // the shim-wrapped client used by the rest of the app
let _raw = null;     // the underlying @tursodatabase/sync Database (push/pull)
let _syncTimer = null;

function localAbsPath() {
  const abs = path.resolve(LOCAL_DB_PATH);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  return abs;
}

// The sync engine writes sidecar files next to the DB (scenario.db-info,
// scenario.db-wal*, etc). If the MAIN db file is gone but sidecars remain
// (crash mid-write, manual deletion, partial restore), connect() throws
// "main DB file doesn't exist, but metadata is" and the app can't boot at all.
// Detect that orphaned state and clear the stale sidecars so a clean replica
// bootstraps instead. Only fires when the main file is absent, so it never
// touches a healthy DB.
function clearOrphanedSyncMetadata(abs) {
  if (fs.existsSync(abs)) return; // main file present → nothing to clean
  const dir = path.dirname(abs);
  const base = path.basename(abs);
  let cleared = 0;
  for (const f of fs.readdirSync(dir)) {
    // sidecars are "<dbfile>-<suffix>"; never delete the main file (absent) or
    // unrelated files.
    if (f.startsWith(`${base}-`)) {
      try { fs.rmSync(path.join(dir, f), { force: true }); cleared++; } catch { /* ignore */ }
    }
  }
  if (cleared) console.warn(`[db] cleared ${cleared} orphaned sync metadata file(s) (main DB was missing).`);
}

// The "<dbfile>-info" sidecar ties the local file's sync state to ONE remote
// database. If the configured remote changed since the last connect (user
// pointed the app at their own Turso DB), that stale metadata would make the
// engine pull/push against the wrong generation. Removing just the -info file
// turns a remote switch into the supported "existing local DB starts syncing
// now" bootstrap; the local data and WAL are untouched.
function clearSyncMetadataIfRemoteChanged(abs, syncUrl) {
  if (!syncUrl) return;
  const last = String(settings.load().lastSyncUrl || '');
  if (syncUrl === last) return;
  if (!last) return; // never synced before → nothing stale to clear
  const info = `${abs}-info`;
  try {
    if (fs.existsSync(info)) {
      fs.rmSync(info, { force: true });
      console.warn('[db] sync remote changed — cleared stale sync metadata; re-bootstrapping against the new remote.');
    }
  } catch (e) {
    console.warn('[db] could not clear stale sync metadata:', e.message);
  }
}

// ---- Compatibility shim ---------------------------------------------------
// The codebase was written against @libsql/client:
//   db.execute('SELECT ...')                 -> { rows, rowsAffected }
//   db.execute({ sql, args: [...] })         -> { rows, rowsAffected }
// The new engine exposes db.all()/db.get()/db.run()/db.exec() instead. This
// wrapper re-implements execute() so nothing else has to change.
//
// Reads (SELECT/PRAGMA/WITH... that returns rows) go through all(); writes go
// through run() (which reports { changes, lastInsertRowid }). We detect reads by
// the leading keyword — good enough for this app's fully-known query set.
function isReadQuery(sql) {
  const head = sql.replace(/^\s+/, '').slice(0, 12).toUpperCase();
  return head.startsWith('SELECT') || head.startsWith('PRAGMA') || head.startsWith('WITH');
}

function wrap(raw) {
  return {
    async execute(q) {
      const sql = typeof q === 'string' ? q : q.sql;
      const args = typeof q === 'string' ? [] : (q.args || []);
      if (isReadQuery(sql)) {
        const rows = await raw.all(sql, ...args);
        return { rows, rowsAffected: 0 };
      }
      const info = await raw.run(sql, ...args);
      return { rows: [], rowsAffected: Number(info?.changes || 0), lastInsertRowid: info?.lastInsertRowid };
    },
    // multi-statement DDL (schema init) — no bind params
    async exec(sql) { return raw.exec(sql); },
    raw,
  };
}

// ---- Schema ---------------------------------------------------------------
// No vector index: the new engine has no libsql_vector_idx. `embedding` is a
// plain blob column holding raw Float32 bytes; retrieval scans + cosines in JS.
async function initSchema(db) {
  await db.exec(`PRAGMA foreign_keys = ON;`);

  await db.exec(`
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

  await db.exec(`
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
  await db.exec(`
    CREATE TABLE IF NOT EXISTS turns (
      id              INTEGER PRIMARY KEY AUTOINCREMENT,
      conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role            TEXT NOT NULL,
      content         TEXT NOT NULL,
      created_at      TEXT NOT NULL
    );
  `);
  await db.exec(`CREATE INDEX IF NOT EXISTS turns_conv_idx ON turns(conversation_id, id);`);

  // Swipe variants: alternate generations for an assistant turn. turns.content
  // mirrors the ACTIVE variant so memory reads turns unchanged.
  await db.exec(`
    CREATE TABLE IF NOT EXISTS variants (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      turn_id    INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
      content    TEXT NOT NULL,
      is_active  INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL
    );
  `);
  await db.exec(`CREATE INDEX IF NOT EXISTS variants_turn_idx ON variants(turn_id, id);`);

  // archive: frozen older turns + embedding (raw Float32 blob) for retrieval.
  await db.exec(`
    CREATE TABLE IF NOT EXISTS archive (
      id              INTEGER PRIMARY KEY AUTOINCREMENT,
      conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role            TEXT NOT NULL,
      content         TEXT NOT NULL,
      embedding       BLOB,
      created_at      TEXT NOT NULL
    );
  `);
  await db.exec(`CREATE INDEX IF NOT EXISTS archive_conv_idx ON archive(conversation_id, id);`);

  await runMigrations(db);
}

// Additive migrations for existing DBs. Each guarded so re-running is safe.
async function runMigrations(db) {
  const cols = await db.raw.all(`PRAGMA table_info(characters);`);
  const has = (name) => cols.some((r) => r.name === name);
  const addCol = async (name, ddl) => { if (!has(name)) await db.exec(`ALTER TABLE characters ADD COLUMN ${ddl};`); };

  await addCol('response_style', `response_style TEXT DEFAULT 'balanced'`);
  await addCol('tagline', `tagline TEXT DEFAULT ''`);
  await addCol('about', `about TEXT DEFAULT ''`);
  await addCol('chat_starters', `chat_starters TEXT DEFAULT '[]'`);
  await addCol('tags', `tags TEXT DEFAULT '[]'`);
}

// ---- Connect --------------------------------------------------------------
async function getDb() {
  if (_db) return _db;

  const [encryptionKey, authToken] = await Promise.all([
    keystore.getDbEncryptionKey(),
    keystore.getTursoToken(),
  ]);
  _encryptedAtRest = Boolean(encryptionKey);
  const syncUrl = resolveSyncUrl();
  _syncUrlInUse = syncUrl;
  SYNC_ENABLED = Boolean(syncUrl && authToken);
  if (!_encryptedAtRest) {
    console.warn('[db] local database is NOT encrypted at rest (no keychain key and no DB_ENCRYPTION_KEY).');
  }

  const dbPath = localAbsPath();
  clearOrphanedSyncMetadata(dbPath);
  if (SYNC_ENABLED) clearSyncMetadataIfRemoteChanged(dbPath, syncUrl);
  const opts = { path: dbPath, clientName: 'vessel' };
  if (SYNC_ENABLED) {
    opts.url = syncUrl;
    opts.authToken = authToken;
  }
  if (encryptionKey) {
    // keystore returns a 64-char hex string (32 bytes); the engine wants hexkey.
    opts.experimental = ['encryption'];
    opts.encryption = { cipher: 'aes256gcm', hexkey: encryptionKey };
  }

  const connect = await getConnect();
  let raw;
  try {
    raw = await connect(opts);
  } catch (e) {
    if (!SYNC_ENABLED) throw e;
    // A bad URL / revoked token / offline remote must NOT brick the app: the
    // engine contacts the remote during connect() and throws (e.g. "Host not
    // found"), which would stop the backend from ever listening — leaving the
    // user no way to reopen Settings and fix the credentials. Fall back to a
    // local-only connect instead.
    console.warn('[db] cloud connect failed — starting LOCAL-ONLY (fix sync settings in-app):', e.message);
    SYNC_ENABLED = false;
    delete opts.url;
    delete opts.authToken;
    raw = await connect(opts);
  }
  _raw = raw;
  _db = wrap(raw);

  // Pull remote state before creating schema so we don't fork an existing cloud DB.
  if (SYNC_ENABLED) {
    try { await raw.pull(); } catch (e) {
      console.warn('[db] initial pull failed (continuing local-first):', e.message);
    }
    // Remember which remote this file's sync metadata now belongs to, so a
    // future URL change can detect it and clear the stale metadata.
    try { settings.save({ lastSyncUrl: syncUrl }); } catch { /* non-fatal */ }
  }

  await initSchema(_db);

  // Push our schema/rows up once, then start a push+pull heartbeat.
  if (SYNC_ENABLED) {
    try { await raw.push(); } catch (e) {
      console.warn('[db] initial push failed (continuing local-first):', e.message);
    }
    if (SYNC_INTERVAL > 0) {
      _syncTimer = setInterval(async () => {
        try { await raw.pull(); await raw.push(); }
        catch (e) { console.warn('[db] periodic sync failed:', e.message); }
      }, SYNC_INTERVAL * 1000);
      if (_syncTimer.unref) _syncTimer.unref();
    }
  }
  return _db;
}

// Force a full sync now (call on shutdown / on demand). No-op when local-only.
async function syncNow() {
  if (!_raw || !SYNC_ENABLED) return false;
  await _raw.pull();
  await _raw.push();
  return true;
}

function isSyncEnabled() { return SYNC_ENABLED; }
function isEncryptedAtRest() { return _encryptedAtRest; }

// ---- Embedding blob codec -------------------------------------------------
// Store embeddings as raw little-endian Float32 bytes (the vector32() SQL func
// no longer exists). encode: number[] -> Buffer; decode: Buffer -> Float32Array.
function encodeEmbedding(arr) {
  const f = Float32Array.from(arr);
  return Buffer.from(f.buffer, f.byteOffset, f.byteLength);
}
function decodeEmbedding(buf) {
  if (!buf) return null;
  // rows may hand back Buffer, Uint8Array, or ArrayBuffer depending on the driver.
  const b = Buffer.isBuffer(buf) ? buf
    : buf instanceof Uint8Array ? Buffer.from(buf.buffer, buf.byteOffset, buf.byteLength)
    : Buffer.from(buf);
  // guard against a truncated/garbage blob
  if (b.byteLength % 4 !== 0) return null;
  // COPY into a fresh, 4-byte-aligned Float32Array rather than viewing over the
  // source buffer: a driver that returns a BLOB as a slice of a pooled
  // ArrayBuffer could hand back a non-4-aligned byteOffset, which would make a
  // Float32Array VIEW constructor throw. A freshly allocated ArrayBuffer is
  // always aligned; copy the bytes into it (cheap for 768 floats).
  const ab = new ArrayBuffer(b.byteLength);
  new Uint8Array(ab).set(b);
  return new Float32Array(ab);
}

module.exports = {
  getDb,
  resolveSyncUrl,
  syncNow,
  isSyncEnabled,
  isEncryptedAtRest,
  encodeEmbedding,
  decodeEmbedding,
  EMBED_DIM,
  get _config() {
    return {
      LOCAL_DB_PATH,
      SYNC_ENABLED,
      SYNC_INTERVAL,
      ENCRYPTED_AT_REST: _encryptedAtRest,
      TURSO_DATABASE_URL: _syncUrlInUse ? '(set)' : '',
    };
  },
};
