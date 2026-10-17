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
    let abs =
        std::path::absolute(config::local_db_path()).unwrap_or_else(|_| config::local_db_path());
    let dir = abs
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    dir.join("settings.json")
}

/// A poisoned lock means some other caller panicked while holding it. The cache
/// is a plain map either way, so take it back rather than turning one panic into
/// a permanent inability to read settings - which would take the Settings panel,
/// and with it the only way to fix a bad sync URL, down with it.
fn lock_cache() -> std::sync::MutexGuard<'static, Option<Settings>> {
    cache().lock().unwrap_or_else(|p| p.into_inner())
}

fn load_cached(guard: &mut Option<Settings>) -> Settings {
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

pub fn load() -> Settings {
    load_cached(&mut lock_cache())
}

/// Merge `patch` over the current settings and persist. Returns the merged map.
///
/// The whole read-merge-write runs under the cache lock. Two savers that
/// interleave around it would each merge onto the same base and the second
/// would drop the first one's key - and the keys here decide which database
/// driver the app opens on next start.
pub fn save(patch: Settings) -> anyhow::Result<Settings> {
    let mut guard = lock_cache();
    let mut next = load_cached(&mut guard);
    for (k, v) in patch {
        next.insert(k, v);
    }
    let path = file_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut body = serde_json::to_string_pretty(&Value::Object(next.clone()))?;
    body.push('\n');
    write_atomically(&path, body.as_bytes())?;
    *guard = Some(next.clone());
    Ok(next)
}

/// Write through a sibling temp file and rename over the target.
///
/// A plain write truncates first, so losing power or crashing between the
/// truncate and the flush leaves a zero-length or half-written file - which
/// parses as "no settings", which silently drops the user's Turso URL and
/// reopens their database on the other driver. The rename is atomic, so a
/// reader sees either the old file or the new one.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension("json.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        // Rename only orders the directory entry. Without this the new file can
        // still be empty on disk when the entry pointing at it is durable.
        f.sync_all()?;
    }
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// These tests deliberately avoid [`load`], [`save`] and [`file_path`].
    /// Those three go through a process-global `OnceLock` cache and a path
    /// derived from `LOCAL_DB_PATH`, and lib tests share one process and run
    /// concurrently - a test that set that variable would race every other
    /// test and could write into the user's real `data/`. The integration
    /// harness in `tests/common/mod.rs` is where env-dependent paths belong.
    /// What is left is still the part that loses data when it is wrong.

    #[test]
    fn a_settings_write_replaces_the_file_instead_of_rewriting_it_in_place() {
        // The durability property the whole function exists for, and the one
        // that is NOT observable from the finished file: a staged rename and a
        // plain truncating write both end with the right bytes on disk. What
        // separates them is whether the target is ever opened for writing. A
        // truncating write empties it first, so a crash in that gap leaves a
        // zero-length file, which parses as "no settings", which drops the
        // user's Turso URL and silently reopens their database on the other
        // driver.
        //
        // Caught by holding the old file open across the write. On Windows a
        // rename over a file that still has an open handle fails, while the
        // staged write renames the STAGING file and never touches this one -
        // so the handle keeps reading the bytes it was opened on. An earlier
        // version of this test only compared the finished contents and passed
        // happily against a truncating implementation: it asserted nothing.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.json");
        let old_body: &[u8] = b"{\"tursoUrl\":\"libsql://old\"}\n";
        std::fs::write(&path, old_body).unwrap();

        let mut reader = std::fs::File::open(&path).unwrap();

        write_atomically(&path, b"{\"tursoUrl\":\"libsql://new\"}\n")
            .expect("a staged write lands even while the old file is held open");

        // The held handle still sees the ORIGINAL bytes: it refers to the file
        // the rename displaced, which is what makes the swap atomic for a
        // concurrent reader rather than merely eventually-correct.
        let mut held = Vec::new();
        {
            use std::io::Read;
            reader.read_to_end(&mut held).unwrap();
        }
        assert_eq!(
            held, old_body,
            "a reader open across the write saw torn or empty content"
        );

        let after = std::fs::read_to_string(&path).unwrap();
        assert!(after.contains("libsql://new"), "{after}");
    }

    #[test]
    fn the_staging_file_is_a_sibling_and_does_not_outlive_the_write() {
        // `with_extension` REPLACES the extension, so the staging name has to
        // be checked rather than assumed: a temp path that landed outside the
        // directory would fail the rename on a different volume, and one left
        // behind would be read as settings by nothing but would accumulate.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.json");
        write_atomically(&path, b"{}\n").unwrap();

        let strays: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n != "settings.json")
            .collect();
        assert!(
            strays.is_empty(),
            "staging file outlived the write: {strays:?}"
        );
    }

    #[test]
    fn a_corrupt_or_wrongly_shaped_file_reads_as_defaults_rather_than_failing() {
        // This file is user-editable and is the only way back in to fix a bad
        // sync URL, so a half-written or hand-mangled one must never stop the
        // app starting. Covers the three shapes the parse chain drops: invalid
        // JSON, valid JSON that is not an object, and an empty file.
        for body in ["{\"tursoUrl\":", "[1,2,3]", "", "null", "\"a string\""] {
            let parsed = serde_json::from_str::<Value>(body)
                .ok()
                .and_then(|v| match v {
                    Value::Object(m) => Some(m),
                    _ => None,
                })
                .unwrap_or_default();
            assert!(
                parsed.is_empty(),
                "{body:?} should read as defaults, got {parsed:?}"
            );
        }
    }

    #[test]
    fn a_present_but_empty_sync_url_is_not_the_same_as_an_absent_one() {
        // The precedence rule the module header states: PRESENT wins over env
        // ("" means the user explicitly cleared it), ABSENT falls back. Getting
        // these two confused is what would make clearing sync in the panel
        // silently re-enable it from a stale `.env`.
        let mut present: Settings = Map::new();
        present.insert("tursoUrl".to_string(), Value::String("  ".to_string()));
        assert_eq!(
            present
                .get("tursoUrl")
                .and_then(Value::as_str)
                .map(str::trim),
            Some(""),
            "a whitespace-only URL must resolve to empty, which disables sync"
        );

        let absent: Settings = Map::new();
        assert!(
            absent.get("tursoUrl").is_none(),
            "an absent key must stay absent so the env fallback can apply"
        );
    }

    #[test]
    fn a_non_string_setting_is_ignored_rather_than_stringified() {
        // `get_str` goes through `as_str`, so a hand-edited `"tursoUrl": 7`
        // reads as absent and falls back to the env, instead of becoming the
        // literal "7" and being handed to the sync engine as a remote URL.
        let mut m: Settings = Map::new();
        m.insert("tursoUrl".to_string(), Value::from(7));
        m.insert("other".to_string(), Value::Bool(true));
        assert!(m.get("tursoUrl").and_then(Value::as_str).is_none());
        assert!(m.get("other").and_then(Value::as_str).is_none());
    }
}
