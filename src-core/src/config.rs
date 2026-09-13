//! Environment-derived configuration.
//!
//! Read once at startup and handed around, rather than re-read at each use, so
//! there is a single place where a default lives and no chance of two callers
//! disagreeing about what a variable meant.

use std::env;
use std::path::PathBuf;

/// nomic-embed-text output dimension.
pub const EMBED_DIM: usize = 768;

/// Keychain service name. Kept as the app's former name on purpose: the DB
/// encryption key and the Turso token are already stored under it, and renaming
/// orphans them — which makes the existing encrypted local DB permanently
/// unreadable. Do not change without a migration.
pub const KEYCHAIN_SERVICE: &str = "scenario-chat";
pub const KEY_ACCOUNT_DB: &str = "db-encryption-key";
pub const KEY_ACCOUNT_TURSO: &str = "turso-auth-token";

fn env_trim(key: &str) -> String {
    env::var(key).unwrap_or_default().trim().to_string()
}

/// The local database path. Default kept at `./data/scenario.db` — renaming it
/// orphans the DB and its sidecars (`-info`, `-wal*`).
pub fn local_db_path() -> PathBuf {
    let raw = env::var("LOCAL_DB_PATH").unwrap_or_else(|_| "./data/scenario.db".into());
    PathBuf::from(raw)
}

/// Absolute local DB path, with the parent directory created.
pub fn local_db_abs() -> std::io::Result<PathBuf> {
    let abs = std::path::absolute(local_db_path())?;
    if let Some(dir) = abs.parent() {
        std::fs::create_dir_all(dir)?;
    }
    Ok(abs)
}

/// Presence-only, so `VESSEL_NO_SYNC=0` and a bare `VESSEL_NO_SYNC=` both
/// disable sync. A harness that mentions this variable at all means it, and the
/// failure mode of guessing wrong is pushing test data into a live database.
pub fn no_sync() -> bool {
    env::var_os("VESSEL_NO_SYNC").is_some()
}

pub fn turso_url_env() -> String {
    env_trim("TURSO_DATABASE_URL")
}

pub fn turso_token_env() -> String {
    env_trim("TURSO_AUTH_TOKEN")
}

pub fn db_encryption_key_override() -> String {
    env_trim("DB_ENCRYPTION_KEY")
}

/// Seconds between background push+pull. 0 disables the heartbeat.
pub fn sync_interval_secs() -> u64 {
    match env::var("TURSO_SYNC_INTERVAL").ok().and_then(|v| v.parse::<i64>().ok()) {
        Some(v) if v >= 0 => v as u64,
        _ => 60,
    }
}

/// Write int8 embedding blobs by default; `EMBED_QUANTIZE=0` falls back to
/// legacy f32. Both formats stay readable either way.
pub fn embed_quantize() -> bool {
    env::var("EMBED_QUANTIZE").map(|v| v != "0").unwrap_or(true)
}
