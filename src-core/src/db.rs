//! Data layer. Port of `src/backend/db.js`, including the part that matters
//! most: the two-driver split.
//!
//! Encryption at rest is real, via turso's aes256gcm whole-file encryption. The
//! catch — and the reason there are two drivers — is that the *sync* engine has
//! no local-file encryption. It accepts a remote-encryption key for the cloud
//! leg only. So:
//!
//! ```text
//!   local-only      -> turso::Builder + aes256gcm   (ENCRYPTED)
//!   cloud sync on   -> turso::sync::Builder         (plaintext, warned)
//! ```
//!
//! The choice is made HERE, at runtime, from live config — never baked in at
//! build or install time. Flipping sync in Settings changes drivers on the next
//! start.
//!
//! Decision 0001 verified that this crate at 0.7.2 opens the exact container
//! `@tursodatabase/database` 0.7.2 writes (canary row read back, wrong-key
//! negative control fails at page 1). The version is pinned with `=` for that
//! reason; the container carries a format version byte.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use serde_json::{Map, Value as Json};
use tokio::sync::OnceCell;
use turso::Value as TValue;

use crate::{config, keystore, settings};

/// Why encryption is off, when it is. A bare boolean cannot distinguish "the
/// user turned on cloud sync and accepted the tradeoff" from "the keychain
/// broke and we silently wrote their stories in cleartext", and those need very
/// different warnings in the UI.
// kebab-case, not lowercase: the renderer keys its explanation table off these
// exact strings, and the Node backend emitted `no-key`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UnencryptedReason {
    /// Cloud sync owns the file; the sync engine cannot encrypt locally.
    Sync,
    /// No keychain key and no `DB_ENCRYPTION_KEY`. Degraded.
    NoKey,
    /// The plaintext -> encrypted migration failed. Degraded.
    Migration,
}

/// The open database handle. It is held for the process's whole lifetime: a
/// dropped `Connection` recycles itself into a pool that the `Database` owns, so
/// letting the handle go would pull that pool out from under every live
/// connection. Both variants are read - to open connections, and (sync only) to
/// `push`/`pull`.
enum Handle {
    Local(turso::Database),
    Sync(turso::sync::Database),
}

/// How many reads may be in flight at once.
///
/// Not a tuning knob - a correctness one. A turso `Connection` rejects
/// overlapping use with "concurrent use forbidden", and this app overlaps
/// constantly: a turn records while the previous turn's fold is still embedding
/// in the background, and the sidebar refreshes on top of both. One shared
/// connection turns every one of those into an error.
///
/// Four, because the concurrent work is bounded and known - the streaming turn,
/// its background maintenance, a UI read, and the sync heartbeat - and idle
/// connections are not free. A fifth reader waits a few milliseconds instead of
/// failing.
const POOL_SIZE: usize = 4;

/// Reader connections that are open but not in use. `slots` is what bounds the
/// pool; `idle` only decides whether a lease reuses a connection or opens one.
struct Pool {
    idle: std::sync::Mutex<Vec<turso::Connection>>,
    slots: tokio::sync::Semaphore,
}

/// A connection borrowed from the pool, returned on drop.
struct Lease<'a> {
    pool: &'a Pool,
    conn: Option<turso::Connection>,
    _slot: tokio::sync::SemaphorePermit<'a>,
}

impl std::ops::Deref for Lease<'_> {
    type Target = turso::Connection;
    fn deref(&self) -> &turso::Connection {
        self.conn
            .as_ref()
            .expect("a lease holds its connection until it is dropped")
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            // A poisoned lock means another task panicked mid-query. The
            // connection is still usable and the list is still a list, so take
            // it back rather than leaking a slot's worth of connection.
            let mut idle = self.pool.idle.lock().unwrap_or_else(|p| p.into_inner());
            if idle.len() < POOL_SIZE {
                idle.push(conn);
            }
        }
    }
}

/// One writer, many readers - the engine's own rule, made explicit.
///
/// MEASURED: eight concurrent inserts over eight connections fail six times with
/// "database is locked". The engine takes a single write lock for the whole file
/// and does NOT wait for it, so overlapping writers do not queue, they error.
/// Serializing writes here is what turns that into a short wait, and it costs
/// nothing real: the writes are single statements against a local file.
///
/// Reads stay parallel, which is where the time actually goes - retrieval scans
/// every archived vector in a conversation.
pub struct Db {
    writer: tokio::sync::Mutex<turso::Connection>,
    readers: Pool,
    handle: Handle,
    sync_enabled: bool,
    encrypted_at_rest: bool,
    unencrypted_reason: Option<UnencryptedReason>,
    path: PathBuf,
}

impl Db {
    /// Borrow a reader, opening one if the pool has none idle.
    async fn reader(&self) -> Result<Lease<'_>> {
        let slot = self
            .readers
            .slots
            .acquire()
            .await
            .map_err(|_| anyhow!("the database connection pool was closed"))?;
        if let Some(conn) = self
            .readers
            .idle
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop()
        {
            return Ok(Lease {
                pool: &self.readers,
                conn: Some(conn),
                _slot: slot,
            });
        }
        let conn = match &self.handle {
            Handle::Local(db) => db.connect()?,
            Handle::Sync(s) => s.connect().await?,
        };
        prepare_connection(&conn).await?;
        Ok(Lease {
            pool: &self.readers,
            conn: Some(conn),
            _slot: slot,
        })
    }

    pub fn is_sync_enabled(&self) -> bool {
        self.sync_enabled
    }
    pub fn is_encrypted_at_rest(&self) -> bool {
        self.encrypted_at_rest
    }
    /// `None` when encrypted; otherwise why not.
    pub fn unencrypted_reason(&self) -> Option<UnencryptedReason> {
        if self.encrypted_at_rest {
            None
        } else {
            self.unencrypted_reason
        }
    }
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Force a full sync now (on shutdown or on demand). No-op when local-only.
    pub async fn sync_now(&self) -> Result<bool> {
        match &self.handle {
            Handle::Sync(s) if self.sync_enabled => {
                s.pull().await?;
                s.push().await?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    // ---- query helpers ----------------------------------------------------
    // Every value is a bound parameter. The only string interpolation anywhere
    // in this file is placeholder generation and schema-introspection
    // identifiers, both of which come from our own code, never from input.

    pub async fn query(&self, sql: &str, params: Vec<TValue>) -> Result<Vec<Map<String, Json>>> {
        let conn = self.reader().await?;
        let mut rows = conn
            .query(sql, turso::params_from_iter(params))
            .await
            .with_context(|| format!("query failed: {}", first_words(sql)))?;
        let names = rows.column_names();
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let mut obj = Map::with_capacity(names.len());
            for (i, name) in names.iter().enumerate() {
                obj.insert(name.clone(), value_to_json(row.get_value(i)?));
            }
            out.push(obj);
        }
        Ok(out)
    }

    /// First row only, or `None`.
    pub async fn query_one(
        &self,
        sql: &str,
        params: Vec<TValue>,
    ) -> Result<Option<Map<String, Json>>> {
        Ok(self.query(sql, params).await?.into_iter().next())
    }

    /// Raw rows, for callers that want blobs without a JSON round trip (the
    /// retrieval scan reads 768-component embeddings — turning those into JSON
    /// arrays and back would dominate the scan).
    pub async fn query_values(&self, sql: &str, params: Vec<TValue>) -> Result<Vec<Vec<TValue>>> {
        let conn = self.reader().await?;
        let mut rows = conn.query(sql, turso::params_from_iter(params)).await?;
        let width = rows.column_count();
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let mut v = Vec::with_capacity(width);
            for i in 0..width {
                v.push(row.get_value(i)?);
            }
            out.push(v);
        }
        Ok(out)
    }

    pub async fn execute(&self, sql: &str, params: Vec<TValue>) -> Result<u64> {
        let conn = self.writer.lock().await;
        conn.execute(sql, turso::params_from_iter(params))
            .await
            .with_context(|| format!("execute failed: {}", first_words(sql)))
    }

    /// An INSERT that reports the rowid it wrote.
    ///
    /// `last_insert_rowid` is a property of a CONNECTION, not of the database,
    /// so reading it after a separate `execute` returns whatever that connection
    /// last inserted. Holding the write lock across both is what ties the id to
    /// the statement rather than to whoever wrote last.
    pub async fn insert(&self, sql: &str, params: Vec<TValue>) -> Result<i64> {
        let conn = self.writer.lock().await;
        conn.execute(sql, turso::params_from_iter(params))
            .await
            .with_context(|| format!("insert failed: {}", first_words(sql)))?;
        Ok(conn.last_insert_rowid())
    }
}

/// Enough of a statement to identify it in an error, without pasting a whole
/// query (which can carry user text in a LIKE) into a log line.
fn first_words(sql: &str) -> String {
    sql.split_whitespace().take(4).collect::<Vec<_>>().join(" ")
}

pub fn value_to_json(v: TValue) -> Json {
    match v {
        TValue::Null => Json::Null,
        TValue::Integer(i) => Json::from(i),
        TValue::Real(f) => serde_json::Number::from_f64(f)
            .map(Json::Number)
            .unwrap_or(Json::Null),
        TValue::Text(s) => Json::String(s),
        // Blobs are embeddings. No caller sends one to the renderer; surfacing
        // the length keeps a debug dump readable without shipping 774 bytes.
        TValue::Blob(b) => Json::from(b.len()),
    }
}

// ---- Sidecar hygiene -------------------------------------------------------

/// The sync engine writes sidecars next to the DB (`-info`, `-wal*`, ...). If
/// the MAIN file is gone but sidecars remain (crash mid-write, manual deletion,
/// partial restore), connect throws "main DB file doesn't exist, but metadata
/// is" and the app cannot boot at all. Clear the stale sidecars so a clean
/// replica bootstraps instead. Only fires when the main file is absent, so it
/// never touches a healthy DB.
fn clear_orphaned_sync_metadata(abs: &Path) {
    if abs.exists() {
        return;
    }
    let (Some(dir), Some(base)) = (abs.parent(), abs.file_name().and_then(|s| s.to_str())) else {
        return;
    };
    let prefix = format!("{base}-");
    let mut cleared = 0usize;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue };
            // Sidecars are "<dbfile>-<suffix>"; never touch the main file
            // (absent here) or anything unrelated.
            if name.starts_with(&prefix) && std::fs::remove_file(e.path()).is_ok() {
                cleared += 1;
            }
        }
    }
    if cleared > 0 {
        tracing::warn!(
            "[db] cleared {cleared} orphaned sync metadata file(s) (main DB was missing)."
        );
    }
}

/// The `-info` sidecar ties the local file's sync state to ONE remote database.
/// If the configured remote changed since the last connect, that stale metadata
/// would make the engine pull/push against the wrong generation. Removing just
/// `-info` turns a remote switch into the supported "existing local DB starts
/// syncing now" bootstrap; local data and WAL are untouched.
fn clear_sync_metadata_if_remote_changed(abs: &Path, sync_url: &str) {
    if sync_url.is_empty() {
        return;
    }
    let last = settings::get_str("lastSyncUrl").unwrap_or_default();
    if last.is_empty() || last == sync_url {
        return; // never synced before, or same remote: nothing stale
    }
    let info = PathBuf::from(format!("{}-info", abs.display()));
    if info.exists() {
        match std::fs::remove_file(&info) {
            Ok(()) => tracing::warn!(
                "[db] sync remote changed - cleared stale sync metadata; re-bootstrapping against the new remote."
            ),
            Err(e) => tracing::warn!("[db] could not clear stale sync metadata: {e}"),
        }
    }
}

// ---- Encryption ------------------------------------------------------------

fn encryption_opts(hexkey: &str) -> turso::EncryptionOpts {
    turso::EncryptionOpts {
        cipher: "aes256gcm".to_string(),
        hexkey: hexkey.to_string(),
    }
}

async fn open_local_plain(path: &Path) -> Result<turso::Database> {
    Ok(turso::Builder::new_local(&path.to_string_lossy())
        .build()
        .await?)
}

async fn open_local_encrypted(path: &Path, hexkey: &str) -> Result<turso::Database> {
    Ok(turso::Builder::new_local(&path.to_string_lossy())
        .experimental_encryption(true)
        .with_encryption(encryption_opts(hexkey))
        .build()
        .await?)
}

/// Is this file already plaintext? The encrypted driver refuses a plaintext file
/// ("Decryption failed for page=1") and the plain driver refuses an encrypted
/// one, so probe cheaply: try to read the schema with no key. Succeeding means
/// plaintext; failing means encrypted (or unreadable, which the real open
/// reports properly).
async fn is_plaintext_db(path: &Path) -> bool {
    // A missing file has nothing to migrate. So does a zero-byte one, and that
    // case matters: an empty file opens fine with the plain driver, which would
    // read as "plaintext" and run a whole migration to produce an empty
    // encrypted database plus a pointless `.plaintext-backup`.
    match std::fs::metadata(path) {
        Ok(m) if m.len() > 0 => {}
        _ => return false,
    }
    let Ok(db) = open_local_plain(path).await else {
        return false;
    };
    let Ok(conn) = db.connect() else { return false };
    conn.query(
        "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1",
        (),
    )
    .await
    .is_ok()
}

/// One-time migration: copy an existing PLAINTEXT database into a new encrypted
/// one, then swap it into place. Encryption is whole-file — the encrypted driver
/// cannot open a plaintext DB — so without this an existing install would break
/// the moment encryption switches on.
///
/// The original is kept as `.plaintext-backup` so a failure can never lose
/// someone's stories; the rename only happens after the new file is fully
/// written.
async fn migrate_plaintext_to_encrypted(path: &Path, hexkey: &str) -> Result<()> {
    let disp = path.display().to_string();
    let tmp = PathBuf::from(format!("{disp}.encrypting"));
    let backup = PathBuf::from(format!("{disp}.plaintext-backup"));

    // A previous interrupted attempt could leave these behind.
    for suffix in ["", "-wal", "-info", "-changes", "-wal-revert"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", tmp.display()));
    }

    tracing::info!("[db] encrypting the existing local database (one-time migration)...");

    let total = {
        let src = open_local_plain(path).await?.connect()?;
        let dst = open_local_encrypted(&tmp, hexkey).await?.connect()?;

        // Fold the source WAL into its main file FIRST. Most of a live
        // database's recent rows sit in the -wal sidecar, so without this the
        // backup kept below would be an empty shell and the sidecar cleanup
        // would discard the only copy of that data.
        let _ = src.execute("PRAGMA wal_checkpoint(truncate)", ()).await;

        // `__turso_internal_%` objects are engine-managed (autoincrement
        // sequences) and are rejected if re-created by hand.
        let tables = {
            let mut rows = src
                .query(
                    "SELECT name, sql FROM sqlite_master WHERE type='table'
                       AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '__turso_internal_%'
                       AND sql IS NOT NULL",
                    (),
                )
                .await?;
            let mut v = Vec::new();
            while let Some(r) = rows.next().await? {
                let (TValue::Text(name), TValue::Text(sql)) = (r.get_value(0)?, r.get_value(1)?)
                else {
                    continue;
                };
                v.push((name, sql));
            }
            v
        };

        for (_, ddl) in &tables {
            dst.execute_batch(ddl).await?;
        }

        let mut total = 0usize;
        for (name, _) in &tables {
            // `name` comes from sqlite_master, not from input.
            let mut rows = src.query(&format!("SELECT * FROM {name}"), ()).await?;
            let cols = rows.column_names();
            let placeholders = vec!["?"; cols.len()].join(",");
            let insert = format!(
                "INSERT INTO {name} ({}) VALUES ({placeholders})",
                cols.join(",")
            );
            while let Some(r) = rows.next().await? {
                let mut vals = Vec::with_capacity(cols.len());
                for i in 0..cols.len() {
                    vals.push(r.get_value(i)?);
                }
                dst.execute(&insert, turso::params_from_iter(vals)).await?;
                total += 1;
            }
        }

        let mut idx = src
            .query(
                "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL
                   AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '__turso_internal_%'",
                (),
            )
            .await?;
        while let Some(r) = idx.next().await? {
            if let TValue::Text(sql) = r.get_value(0)? {
                // A failure here is fine: init_schema recreates every index.
                let _ = dst.execute_batch(&sql).await;
            }
        }

        let _ = dst.execute("PRAGMA wal_checkpoint(truncate)", ()).await;
        total
    }; // both connections dropped here, releasing the files before the rename

    // Swap: keep the plaintext original as a backup rather than deleting it.
    // (Its data is all in the main file now, thanks to the checkpoint above.)
    std::fs::rename(path, &backup).context("could not set the plaintext file aside")?;
    // Drop the OLD file's sidecars — they describe the plaintext DB and would
    // corrupt reads of the encrypted one taking its place.
    for suffix in ["-wal", "-info", "-changes", "-wal-revert"] {
        let _ = std::fs::remove_file(format!("{disp}{suffix}"));
    }
    std::fs::rename(&tmp, path).context("could not move the encrypted file into place")?;
    for suffix in ["-wal", "-info", "-changes", "-wal-revert"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", tmp.display()));
    }

    tracing::info!(
        "[db] migration complete: {total} rows encrypted. The previous plaintext file is kept at\n     {}\n     Delete it once you have confirmed the app works - it is NOT encrypted.",
        backup.display()
    );
    Ok(())
}

// ---- Schema ----------------------------------------------------------------
// No vector index: this engine has no libsql_vector_idx. `embedding` is a plain
// blob column holding the codec's bytes; retrieval scans and cosines in Rust.

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS characters (
  id             TEXT PRIMARY KEY,
  name           TEXT NOT NULL,
  avatar         TEXT DEFAULT '',
  persona        TEXT DEFAULT '',
  greeting       TEXT DEFAULT '',
  sampling       TEXT DEFAULT '{}',
  -- Added later, and repeated in the migration list for databases that predate
  -- them. Declaring them here as well saves a fresh install five ALTER TABLE
  -- statements on first launch; the migration then finds them present.
  response_style TEXT DEFAULT 'balanced',
  tagline        TEXT DEFAULT '',
  about          TEXT DEFAULT '',
  chat_starters  TEXT DEFAULT '[]',
  tags           TEXT DEFAULT '[]',
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
  id           TEXT PRIMARY KEY,
  character_id TEXT REFERENCES characters(id) ON DELETE CASCADE,
  title        TEXT DEFAULT '',
  summary      TEXT DEFAULT '',
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);

-- The sidebar reads every conversation ordered by recency, filtered by
-- character. Without this it is a full scan plus a sort on every refresh.
CREATE INDEX IF NOT EXISTS conversations_char_idx ON conversations(character_id, updated_at DESC);

-- verbatim recent turns (kept in full in the live window)
CREATE TABLE IF NOT EXISTS turns (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role            TEXT NOT NULL,
  content         TEXT NOT NULL,
  created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_conv_idx ON turns(conversation_id, id);

-- Swipe variants: alternate generations for an assistant turn. turns.content
-- mirrors the ACTIVE variant so memory reads turns unchanged.
CREATE TABLE IF NOT EXISTS variants (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  turn_id    INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
  content    TEXT NOT NULL,
  is_active  INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS variants_turn_idx ON variants(turn_id, id);

-- archive: frozen older turns + embedding blob for retrieval.
CREATE TABLE IF NOT EXISTS archive (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role            TEXT NOT NULL,
  content         TEXT NOT NULL,
  embedding       BLOB,
  created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS archive_conv_idx ON archive(conversation_id, id);
"#;

/// Additive migrations for existing DBs. Each guarded so re-running is safe.
async fn run_migrations(conn: &turso::Connection) -> Result<()> {
    let mut have = Vec::new();
    let mut rows = conn.query("PRAGMA table_info(characters)", ()).await?;
    while let Some(r) = rows.next().await? {
        if let TValue::Text(name) = r.get_value(1)? {
            have.push(name);
        }
    }
    for (col, ddl) in [
        ("response_style", "response_style TEXT DEFAULT 'balanced'"),
        ("tagline", "tagline TEXT DEFAULT ''"),
        ("about", "about TEXT DEFAULT ''"),
        ("chat_starters", "chat_starters TEXT DEFAULT '[]'"),
        ("tags", "tags TEXT DEFAULT '[]'"),
    ] {
        if !have.iter().any(|c| c == col) {
            conn.execute_batch(&format!("ALTER TABLE characters ADD COLUMN {ddl};"))
                .await?;
        }
    }
    Ok(())
}

/// Per-connection state. `foreign_keys` is NOT a database property - it is set
/// per connection and defaults to off - so every connection the pool opens must
/// set it, or `ON DELETE CASCADE` quietly stops firing and deleting a character
/// leaves its conversations and turns behind.
async fn prepare_connection(conn: &turso::Connection) -> Result<()> {
    Ok(conn.execute_batch("PRAGMA foreign_keys = ON;").await?)
}

async fn init_schema(conn: &turso::Connection) -> Result<()> {
    prepare_connection(conn).await?;
    conn.execute_batch(SCHEMA).await?;
    run_migrations(conn).await
}

// ---- Connect ---------------------------------------------------------------

/// Readers open on first use, not at startup: a session that only ever writes
/// should not pay to open three connections it will never read from. The
/// connection that opened the file becomes the writer, since it is the one that
/// already ran the schema.
fn new_pool() -> Pool {
    Pool {
        idle: std::sync::Mutex::new(Vec::new()),
        slots: tokio::sync::Semaphore::new(POOL_SIZE),
    }
}

static DB: OnceCell<Arc<Db>> = OnceCell::const_new();

/// Memoized on the initialization itself, not on the resolved handle. Anything
/// weaker only closes the window after connect finishes, and everything in
/// between is an await: the first turn asks for the DB several times at once, so
/// two callers would each open the file, each run the schema, and — the part
/// that actually destroys data — each start the one-time plaintext->encrypted
/// migration over the same file.
pub async fn get() -> Result<Arc<Db>> {
    DB.get_or_try_init(|| async { connect().await.map(Arc::new) })
        .await
        .cloned()
}

/// Already-initialized handle, if any. Used by shutdown, which must not open a
/// database just to close it.
pub fn get_if_ready() -> Option<Arc<Db>> {
    DB.get().cloned()
}

/// A throwaway, unencrypted database at an explicit path, for tests only.
///
/// Every other way into a `Db` resolves `LOCAL_DB_PATH` and can therefore reach
/// the user's real file. Modules whose tests write and delete rows - `memory`,
/// above all - get a handle built from a path they chose themselves, and
/// `#[cfg(test)]` keeps it out of every shipped binary.
#[cfg(test)]
pub(crate) async fn open_scratch(path: &Path) -> Result<Db> {
    let db = open_local_plain(path).await?;
    let conn = db.connect()?;
    init_schema(&conn).await?;
    Ok(Db {
        writer: tokio::sync::Mutex::new(conn),
        readers: new_pool(),
        handle: Handle::Local(db),
        sync_enabled: false,
        encrypted_at_rest: false,
        unencrypted_reason: Some(UnencryptedReason::NoKey),
        path: path.to_path_buf(),
    })
}

async fn connect() -> Result<Db> {
    let path = config::local_db_abs()?;
    let encryption_key = keystore::db_encryption_key();
    let auth_token = keystore::turso_token();

    let sync_url = if config::no_sync() {
        tracing::warn!("[db] VESSEL_NO_SYNC is set: cloud sync is OFF for this process.");
        String::new()
    } else {
        settings::resolve_sync_url()
    };
    let want_sync = !sync_url.is_empty() && !auth_token.is_empty();

    clear_orphaned_sync_metadata(&path);
    if want_sync {
        clear_sync_metadata_if_remote_changed(&path, &sync_url);
    }

    if want_sync {
        // Cloud sync: the sync engine owns the file and cannot encrypt it.
        match turso::sync::Builder::new_remote(&path.to_string_lossy())
            .with_remote_url(sync_url.clone())
            .with_auth_token(auth_token)
            .with_client_name("vessel")
            .build()
            .await
        {
            Ok(sdb) => {
                // Only warn once sync actually came up; a failed connect falls
                // back to the encrypted local path and the warning would be wrong.
                tracing::warn!(
                    "[db] cloud sync is ON, so the local database is NOT encrypted at rest: the sync engine\n     has no local-file encryption (it only encrypts the cloud leg). Treat this file as\n     sensitive plaintext: {}\n     Turn sync off in Settings to get an encrypted local database.",
                    path.display()
                );
                let conn = sdb.connect().await?;

                // Pull remote state before creating schema so we don't fork an
                // existing cloud DB.
                if let Err(e) = sdb.pull().await {
                    tracing::warn!("[db] initial pull failed (continuing local-first): {e}");
                }
                let _ = settings::save(
                    [("lastSyncUrl".to_string(), Json::String(sync_url.clone()))]
                        .into_iter()
                        .collect(),
                );

                init_schema(&conn).await?;

                if let Err(e) = sdb.push().await {
                    tracing::warn!("[db] initial push failed (continuing local-first): {e}");
                }
                start_sync_heartbeat(sdb.clone());

                return Ok(Db {
                    writer: tokio::sync::Mutex::new(conn),
                    readers: new_pool(),
                    handle: Handle::Sync(sdb),
                    sync_enabled: true,
                    encrypted_at_rest: false,
                    unencrypted_reason: Some(UnencryptedReason::Sync),
                    path,
                });
            }
            Err(e) => {
                // A bad URL / revoked token / offline remote must NOT brick the
                // app: the engine contacts the remote during connect and throws,
                // which would stop the app from ever opening — leaving the user
                // no way to reach Settings and fix the credentials. Fall back to
                // local-only, which also re-enables encryption.
                tracing::warn!(
                    "[db] cloud connect failed - starting LOCAL-ONLY (fix sync settings in-app): {e}"
                );
            }
        }
    }

    let (db, encrypted, reason) = open_local(&path, encryption_key.as_deref()).await?;
    let conn = db.connect()?;
    init_schema(&conn).await?;
    Ok(Db {
        writer: tokio::sync::Mutex::new(conn),
        readers: new_pool(),
        handle: Handle::Local(db),
        sync_enabled: false,
        encrypted_at_rest: encrypted,
        unencrypted_reason: reason,
        path,
    })
}

/// Open the local-only path. Handles the one-time plaintext->encrypted
/// migration and degrades to plaintext rather than failing to boot — except for
/// the one case that must always fail loudly (see below).
async fn open_local(
    path: &Path,
    key: Option<&str>,
) -> Result<(turso::Database, bool, Option<UnencryptedReason>)> {
    let Some(key) = key else {
        tracing::warn!(
            "[db] local database is NOT encrypted at rest (no keychain key and no DB_ENCRYPTION_KEY)."
        );
        return Ok((
            open_local_plain(path).await?,
            false,
            Some(UnencryptedReason::NoKey),
        ));
    };

    if is_plaintext_db(path).await {
        if let Err(e) = migrate_plaintext_to_encrypted(path, key).await {
            // Keep the user's data reachable.
            tracing::error!(
                "[db] encryption migration FAILED - keeping the existing database as-is: {e}"
            );
            return Ok((
                open_local_plain(path).await?,
                false,
                Some(UnencryptedReason::Migration),
            ));
        }
    }

    match open_local_encrypted(path, key).await {
        Ok(db) => Ok((db, true, None)),
        Err(e) => {
            // We HAVE a key and the file is not plaintext, so the file exists
            // and is encrypted with a DIFFERENT key (rotated or lost keychain
            // entry, a DB restored from another machine) or is damaged. Falling
            // back to the plain driver is never right: it cannot open this file
            // anyway, and when the file is merely missing it silently creates a
            // NEW plaintext database that quietly records every future story in
            // cleartext. Fail loudly instead — the encrypted data is left
            // untouched, so recovering the original key recovers the stories.
            tracing::error!("[db] could not open the encrypted database: {e}");
            Err(anyhow!(
                "Could not open the encrypted database at {}: {e}. Refusing to fall back to an \
                 unencrypted database - your existing data is still encrypted and untouched.",
                path.display()
            ))
        }
    }
}

/// Background push+pull. Reports an outage once and its recovery once, not every
/// interval: an unreachable remote is a steady state, and a line a minute
/// forever buries everything else without adding a single fact.
fn start_sync_heartbeat(sdb: turso::sync::Database) {
    let interval = config::sync_interval_secs();
    if interval == 0 {
        return;
    }
    tokio::spawn(async move {
        let failing = AtomicBool::new(false);
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        ticker.tick().await; // the first tick fires immediately; skip it
        loop {
            ticker.tick().await;
            let result = async {
                sdb.pull().await?;
                sdb.push().await
            }
            .await;
            match result {
                Ok(()) => {
                    if failing.swap(false, Ordering::Relaxed) {
                        tracing::warn!("[db] sync recovered.");
                    }
                }
                Err(e) => {
                    if !failing.swap(true, Ordering::Relaxed) {
                        tracing::warn!("[db] sync failing (silenced until it recovers): {e}");
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    //! These tests drive the driver-selection helpers DIRECTLY, with an explicit
    //! path and key. They never read `LOCAL_DB_PATH`, never call [`get`], and so
    //! can never reach the real database - which matters more than usual here,
    //! because one of the paths under test rewrites the file it is given.

    use super::*;
    use turso::params;

    const KEY_A: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
    const KEY_B: &str = "ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100";

    fn scratch() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("scenario.db");
        (dir, path)
    }

    async fn seed_plaintext(path: &Path, rows: &[(&str, &str)]) {
        let conn = open_local_plain(path).await.unwrap().connect().unwrap();
        init_schema(&conn).await.unwrap();
        for (id, name) in rows {
            conn.execute(
                "INSERT INTO characters (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                params![*id, *name, "2026-01-01", "2026-01-01"],
            )
            .await
            .unwrap();
        }
        conn.execute("PRAGMA wal_checkpoint(truncate)", ())
            .await
            .ok();
    }

    async fn names(conn: &turso::Connection) -> Vec<String> {
        let mut rows = conn
            .query("SELECT name FROM characters ORDER BY id", ())
            .await
            .unwrap();
        let mut out = Vec::new();
        while let Some(r) = rows.next().await.unwrap() {
            if let TValue::Text(t) = r.get_value(0).unwrap() {
                out.push(t);
            }
        }
        out
    }

    // ---- Concurrency ------------------------------------------------------
    // The app overlaps database work constantly: a turn records while the
    // previous turn's fold embeds in the background and the sidebar refreshes on
    // top of both. These pin the two ways that used to break.

    async fn scratch_db() -> (tempfile::TempDir, Arc<Db>) {
        let (dir, path) = scratch();
        (dir, Arc::new(open_scratch(&path).await.unwrap()))
    }

    fn conv_row(id: &str) -> Vec<TValue> {
        vec![
            TValue::Text(id.into()),
            TValue::Text("t".into()),
            TValue::Text("2026-01-01".into()),
            TValue::Text("2026-01-01".into()),
        ]
    }

    const INSERT_CONV: &str =
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)";

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn overlapping_writes_wait_instead_of_failing() {
        // MEASURED before the writer lock existed: eight of these came back
        // "database is locked". The engine takes one write lock for the whole
        // file and does not wait for it, so overlapping writers error rather
        // than queue.
        let (_dir, db) = scratch_db().await;
        let mut tasks = Vec::new();
        for i in 0..16 {
            let db = db.clone();
            tasks.push(tokio::spawn(async move {
                db.execute(INSERT_CONV, conv_row(&format!("id-{i}"))).await
            }));
        }
        for t in tasks {
            t.await.unwrap().expect("a concurrent write must not fail");
        }
        let rows = db
            .query("SELECT COUNT(*) AS n FROM conversations", vec![])
            .await
            .unwrap();
        assert_eq!(rows[0]["n"], 16);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reads_keep_running_while_writes_are_in_flight() {
        // Serializing writes must not serialize reads with them: retrieval scans
        // every archived vector in a conversation, and it runs during a turn.
        let (_dir, db) = scratch_db().await;
        let mut tasks = Vec::new();
        for i in 0..8 {
            let db = db.clone();
            tasks.push(tokio::spawn(async move {
                db.execute(INSERT_CONV, conv_row(&format!("w-{i}")))
                    .await
                    .map(|_| ())
            }));
        }
        for _ in 0..8 {
            let db = db.clone();
            tasks.push(tokio::spawn(async move {
                for _ in 0..10 {
                    db.query("SELECT COUNT(*) AS n FROM conversations", vec![])
                        .await?;
                }
                Ok(())
            }));
        }
        for t in tasks {
            t.await
                .unwrap()
                .expect("a read must not fail because a write is running");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_insert_reports_its_own_row_not_the_last_one_written() {
        // `last_insert_rowid` belongs to a connection, so reading it after a
        // separate `execute` would hand a turn somebody else's rowid - and the
        // variant rows written against that id would attach to the wrong turn.
        let (_dir, db) = scratch_db().await;
        db.execute(INSERT_CONV, conv_row("c1")).await.unwrap();

        let mut tasks = Vec::new();
        for i in 0..16 {
            let db = db.clone();
            tasks.push(tokio::spawn(async move {
                let content = format!("turn-{i}");
                let id = db
                    .insert(
                        "INSERT INTO turns (conversation_id, role, content, created_at)
                         VALUES (?, ?, ?, ?)",
                        vec![
                            TValue::Text("c1".into()),
                            TValue::Text("user".into()),
                            TValue::Text(content.clone()),
                            TValue::Text("2026-01-01".into()),
                        ],
                    )
                    .await
                    .unwrap();
                (id, content)
            }));
        }

        let mut seen = std::collections::HashSet::new();
        for t in tasks {
            let (id, content) = t.await.unwrap();
            assert!(seen.insert(id), "two inserts reported the same rowid: {id}");
            let row = db
                .query_one(
                    "SELECT content FROM turns WHERE id = ?",
                    vec![TValue::Integer(id)],
                )
                .await
                .unwrap()
                .expect("the reported rowid must exist");
            assert_eq!(
                row["content"],
                serde_json::json!(content),
                "rowid {id} names another row"
            );
        }
    }

    #[tokio::test]
    async fn deleting_a_character_still_cascades() {
        // `foreign_keys` is per connection and defaults to OFF. If a pooled
        // connection ever skipped the pragma, a delete would leave the rows
        // behind instead of erroring, and nothing else would notice.
        let (_dir, db) = scratch_db().await;
        db.execute(
            "INSERT INTO characters (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            vec![
                TValue::Text("ch1".into()),
                TValue::Text("Aria".into()),
                TValue::Text("2026-01-01".into()),
                TValue::Text("2026-01-01".into()),
            ],
        )
        .await
        .unwrap();
        db.execute(
            "INSERT INTO conversations (id, character_id, title, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?)",
            vec![
                TValue::Text("c1".into()),
                TValue::Text("ch1".into()),
                TValue::Text("t".into()),
                TValue::Text("2026-01-01".into()),
                TValue::Text("2026-01-01".into()),
            ],
        )
        .await
        .unwrap();

        db.execute(
            "DELETE FROM characters WHERE id = ?",
            vec![TValue::Text("ch1".into())],
        )
        .await
        .unwrap();

        let rows = db
            .query("SELECT COUNT(*) AS n FROM conversations", vec![])
            .await
            .unwrap();
        assert_eq!(
            rows[0]["n"], 0,
            "the character went but its conversation stayed"
        );
    }

    #[tokio::test]
    async fn a_fresh_local_database_is_encrypted() {
        let (_dir, path) = scratch();
        let (db, encrypted, reason) = open_local(&path, Some(KEY_A)).await.unwrap();
        assert!(encrypted);
        assert_eq!(reason, None);
        init_schema(&db.connect().unwrap()).await.unwrap();
        drop(db);

        // The proof that encryption is real, not a flag we set: the plain driver
        // cannot read the file back. It gives up at connect, before a query is
        // even possible, so accept a failure at either step.
        let readable = match open_local_plain(&path).await.and_then(|d| Ok(d.connect()?)) {
            Ok(conn) => conn.query("SELECT name FROM characters", ()).await.is_ok(),
            Err(_) => false,
        };
        assert!(
            !readable,
            "an encrypted file must not be readable without the key"
        );
    }

    #[tokio::test]
    async fn the_wrong_key_fails_loudly_instead_of_starting_a_new_plaintext_db() {
        let (_dir, path) = scratch();
        let (db, _, _) = open_local(&path, Some(KEY_A)).await.unwrap();
        init_schema(&db.connect().unwrap()).await.unwrap();
        drop(db);

        let err = open_local(&path, Some(KEY_B))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Refusing to fall back to an unencrypted database"),
            "unexpected error: {err}"
        );
        // The original is untouched, so recovering the key recovers the stories.
        let (db, encrypted, _) = open_local(&path, Some(KEY_A)).await.unwrap();
        assert!(encrypted);
        assert_eq!(names(&db.connect().unwrap()).await.len(), 0);
    }

    #[tokio::test]
    async fn an_existing_plaintext_database_is_migrated_and_backed_up() {
        let (_dir, path) = scratch();
        seed_plaintext(&path, &[("a", "Ada"), ("b", "Bo")]).await;
        assert!(is_plaintext_db(&path).await);

        let (db, encrypted, reason) = open_local(&path, Some(KEY_A)).await.unwrap();
        assert!(encrypted, "migration should leave us encrypted");
        assert_eq!(reason, None);
        assert_eq!(names(&db.connect().unwrap()).await, vec!["Ada", "Bo"]);
        drop(db);

        let backup = PathBuf::from(format!("{}.plaintext-backup", path.display()));
        assert!(backup.exists(), "the pre-migration file must be kept");
        assert!(
            !is_plaintext_db(&path).await,
            "the live file is encrypted now"
        );
    }

    #[tokio::test]
    async fn migration_keeps_rows_that_were_still_in_the_wal() {
        // The checkpoint inside the migration is what makes this pass: without
        // it the backup is an empty shell and the sidecar cleanup throws away
        // the only copy of these rows.
        let (_dir, path) = scratch();
        {
            let conn = open_local_plain(&path).await.unwrap().connect().unwrap();
            init_schema(&conn).await.unwrap();
            conn.execute(
                "INSERT INTO characters (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                params!["w", "Wal", "2026-01-01", "2026-01-01"],
            )
            .await
            .unwrap();
            // Deliberately NO checkpoint here.
        }
        let (db, encrypted, _) = open_local(&path, Some(KEY_A)).await.unwrap();
        assert!(encrypted);
        assert_eq!(names(&db.connect().unwrap()).await, vec!["Wal"]);
    }

    #[tokio::test]
    async fn without_a_key_the_database_is_plaintext_and_says_why() {
        let (_dir, path) = scratch();
        let (db, encrypted, reason) = open_local(&path, None).await.unwrap();
        assert!(!encrypted);
        assert_eq!(reason, Some(UnencryptedReason::NoKey));
        init_schema(&db.connect().unwrap()).await.unwrap();
    }

    #[tokio::test]
    async fn an_empty_file_is_not_treated_as_a_plaintext_database() {
        let (_dir, path) = scratch();
        std::fs::write(&path, b"").unwrap();
        assert!(!is_plaintext_db(&path).await);

        let (db, encrypted, _) = open_local(&path, Some(KEY_A)).await.unwrap();
        assert!(encrypted);
        drop(db);
        let backup = PathBuf::from(format!("{}.plaintext-backup", path.display()));
        assert!(
            !backup.exists(),
            "nothing was migrated, so nothing to back up"
        );
    }

    #[test]
    fn orphaned_sync_sidecars_are_cleared_only_when_the_main_file_is_gone() {
        let (_dir, path) = scratch();
        let sidecar = PathBuf::from(format!("{}-info", path.display()));
        let neighbour = path.parent().unwrap().join("unrelated.db");

        // Main file present: nothing is touched.
        std::fs::write(&path, b"x").unwrap();
        std::fs::write(&sidecar, b"y").unwrap();
        std::fs::write(&neighbour, b"z").unwrap();
        clear_orphaned_sync_metadata(&path);
        assert!(sidecar.exists());

        // Main file gone: the stale sidecar goes, the neighbour stays.
        std::fs::remove_file(&path).unwrap();
        clear_orphaned_sync_metadata(&path);
        assert!(!sidecar.exists());
        assert!(neighbour.exists(), "only this DB's sidecars may be removed");
    }

    #[tokio::test]
    async fn schema_and_migrations_are_idempotent() {
        let (_dir, path) = scratch();
        let db = open_local_plain(&path).await.unwrap();
        let conn = db.connect().unwrap();
        init_schema(&conn).await.unwrap();
        init_schema(&conn).await.unwrap();
        run_migrations(&conn).await.unwrap();

        let mut cols = Vec::new();
        let mut rows = conn
            .query("PRAGMA table_info(characters)", ())
            .await
            .unwrap();
        while let Some(r) = rows.next().await.unwrap() {
            if let TValue::Text(n) = r.get_value(1).unwrap() {
                cols.push(n);
            }
        }
        for expected in [
            "response_style",
            "tagline",
            "about",
            "chat_starters",
            "tags",
        ] {
            assert!(
                cols.contains(&expected.to_string()),
                "missing column {expected}"
            );
        }
    }
}
