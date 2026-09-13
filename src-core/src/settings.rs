//! Non-secret runtime settings persisted next to the local DB
//! (`data/settings.json`), written by the in-app Settings panel. A packaged
//! install has no `.env`, so this file is how an end user's own config (their
//! Turso database URL, say) survives restarts.
//!
//! Secrets do NOT live here — the Turso auth token goes to the OS keychain via
//! [`crate::keystore`].
//!
//! Precedence: a key PRESENT in this file wins over the matching env var (`""`
//! means the user explicitly cleared it); an ABSENT key falls back to the env,
//! so dev setups keep working untouched.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use serde_json::{Map, Value};

use crate::config;

type Settings = Map<String, Value>;

fn cache() -> &'static Mutex<Option<Settings>> {
    static CACHE: OnceLock<Mutex<Option<Settings>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// `settings.json` sits beside the database file, so a relocated DB takes its
/// settings with it.
pub fn file_path() -> PathBuf {
    let abs = std::path::absolute(config::local_db_path())
        .unwrap_or_else(|_| config::local_db_path());
    let dir = abs.parent().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    dir.join("settings.json")
}

pub fn load() -> Settings {
    let mut guard = cache().lock().expect("settings cache poisoned");
    if let Some(v) = guard.as_ref() {
        return v.clone();
    }
    // Missing or corrupt file means defaults, not a failure to boot: this file
    // is user-editable and a half-written one must never stop the app starting,
    // which is the only way back in to fix it.
    let loaded = std::fs::read_to_string(file_path())
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| match v {
            Value::Object(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default();
    *guard = Some(loaded.clone());
    loaded
}

/// Merge `patch` over the current settings and persist. Returns the merged map.
pub fn save(patch: Settings) -> anyhow::Result<Settings> {
    let mut next = load();
    for (k, v) in patch {
        next.insert(k, v);
    }
    let path = file_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut body = serde_json::to_string_pretty(&Value::Object(next.clone()))?;
    body.push('\n');
    std::fs::write(&path, body)?;
    *cache().lock().expect("settings cache poisoned") = Some(next.clone());
    Ok(next)
}

/// Read a string setting, if present.
pub fn get_str(key: &str) -> Option<String> {
    load().get(key).and_then(Value::as_str).map(str::to_string)
}

/// Sync URL: `settings.json` wins when its key is PRESENT (the Settings panel
/// wrote it; `""` there means the user explicitly disabled sync). An absent key
/// falls back to the env var so dev `.env` setups keep working.
pub fn resolve_sync_url() -> String {
    match get_str("tursoUrl") {
        Some(v) => v.trim().to_string(),
        None => config::turso_url_env(),
    }
}

/// Test-only: drop the memoized copy so a test can point at a different file.
#[cfg(test)]
pub fn reset_cache_for_test() {
    *cache().lock().expect("settings cache poisoned") = None;
}
