//! Secrets at rest via the OS keychain (Windows Credential Manager here).
//!
//! Holds two secrets:
//!   - `db-encryption-key`: 32-byte random key (hex) that encrypts the local
//!     database file at rest via turso's aes256gcm whole-file encryption (see
//!     [`crate::db`] — the sync engine accepts no such option). Generated once
//!     on first run and reused after.
//!   - `turso-auth-token`: optional Turso cloud auth token, kept out of `.env`
//!     so it isn't left in plaintext on disk.
//!
//! The store encrypts values with DPAPI per user, so the key never lives on disk
//! in cleartext. If the store can't be reached we fail safe: the DB key falls
//! back to an env var if explicitly provided, otherwise encryption is disabled
//! rather than silently inventing a key we cannot persist — which would brick
//! the existing DB on the next launch.

use keyring::{Entry, Error as KeyringError};
use rand::RngCore;

use crate::config::{KEYCHAIN_SERVICE, KEY_ACCOUNT_DB, KEY_ACCOUNT_TURSO};

/// The Windows Credential Manager `target_name` the Electron build already uses.
///
/// This is the whole reason this function exists. The two tracks reach the same
/// Credential Manager through different libraries, and the libraries disagree
/// about how to turn (service, account) into a target name:
///
///   - keytar 7.9.0 (`keytar_win.cc:120`) writes `service + '/' + account`
///     -> `scenario-chat/db-encryption-key`
///   - keyring 4.2.0's Windows store (`cred.rs:53`, default divider '.') writes
///     `user + '.' + service` -> `db-encryption-key.scenario-chat`
///
/// Same SERVICE, same account, two different credentials. Left alone, the Rust
/// track reads an empty store, `db_encryption_key` takes its generate-on-empty
/// branch, and the app mints a SECOND key while the real one sits untouched
/// under keytar's name - the exact orphaning that the `scenario-chat` service
/// name was frozen to prevent, arriving by a different route.
///
/// So we pin the target name explicitly rather than let the store compose one.
/// keytar's spelling wins because it is the one with the user's live data
/// behind it; the Rust side is the newcomer and adapts.
///
/// The target name is only half of it - see [`get`] and [`set`] for the blob
/// encoding, which the two libraries also disagree about.
fn target_name(account: &str) -> String {
    format!("{KEYCHAIN_SERVICE}/{account}")
}

/// Read a secret the way keytar wrote it: the blob is raw UTF-8.
///
/// `Entry::get_password` cannot be used here. keyring's Windows store writes
/// passwords as UTF-16LE and decodes them the same way
/// (`windows-native-keyring-store/src/utils.rs:80` and `:275`), while keytar
/// hands `CredWrite` the UTF-8 bytes directly
/// (`keytar_win.cc:133`, `CredentialBlobSize = password.size()`).
///
/// Point keyring at a credential keytar wrote and it decodes those UTF-8 bytes
/// as UTF-16: a 64-character hex key comes back as 32 CJK characters. It does
/// not error - it returns a plausible-looking String - which is the dangerous
/// part, because that string would then be used as the database key.
///
/// `get_secret` is the one accessor that hands back the blob untouched, so we
/// take the bytes and do the decoding ourselves.
fn get(account: &str) -> Result<Option<String>, KeyringError> {
    match entry(account)?.get_secret() {
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(v) => Ok(Some(v)),
            // A credential that is neither valid UTF-8 nor ours. Treat it as
            // absent rather than guessing; the caller's empty-store path is
            // safe, and silently inventing a decoding is not.
            Err(_) => {
                tracing::warn!("[keystore] credential for {account} is not valid UTF-8; ignoring.");
                Ok(None)
            }
        },
        Err(KeyringError::NoEntry) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Write a secret the way keytar reads it: raw UTF-8, no UTF-16 round trip.
fn set(account: &str, value: &str) -> Result<(), KeyringError> {
    entry(account)?.set_secret(value.as_bytes())
}

fn entry(account: &str) -> Result<Entry, KeyringError> {
    // `Entry::new` is also what installs the process-wide default store (it is
    // guarded by a `LazyLock` in keyring's v1 wrapper). `keyring_core::Entry`
    // has the modifier constructor but does NOT run that init, so we go through
    // the wrapper first and let its error surface unchanged.
    let _ = Entry::new(KEYCHAIN_SERVICE, account)?;
    let target = target_name(account);
    let mods = std::collections::HashMap::from([("target", target.as_str())]);
    let inner = keyring_core::Entry::new_with_modifiers(KEYCHAIN_SERVICE, account, &mods)?;
    Ok(Entry { inner })
}

/// `Ok(None)` means "no such credential", which is a normal first-run state and
/// must not be confused with "the keychain is unreachable".
fn read(account: &str) -> Result<Option<String>, KeyringError> {
    get(account)
}

/// Resolve the local-DB encryption key (hex), or `None` if encryption cannot be
/// enabled. Order:
///   1. `DB_ENCRYPTION_KEY` env var (explicit override — lets advanced users BYO key).
///   2. keychain value (created on first run if absent).
///   3. `None` → caller opens the DB unencrypted.
///
/// Returning a *fresh* random key when one already exists would make the
/// existing encrypted DB unreadable, so we only ever generate when the keychain
/// has nothing AND no override is set.
pub fn db_encryption_key() -> Option<String> {
    let override_key = crate::config::db_encryption_key_override();
    if !override_key.is_empty() {
        return Some(override_key);
    }

    match read(KEY_ACCOUNT_DB) {
        Ok(Some(k)) if !k.is_empty() => Some(k),
        Ok(_) => {
            let mut buf = [0u8; 32]; // 256-bit
            rand::thread_rng().fill_bytes(&mut buf);
            let key = hex::encode(buf);
            match set(KEY_ACCOUNT_DB, &key) {
                Ok(()) => {
                    tracing::info!(
                        "[keystore] generated and stored a new local-DB encryption key."
                    );
                    Some(key)
                }
                // No safe place to persist it: staying plaintext is recoverable,
                // encrypting with a key we just lost is not.
                Err(e) => {
                    tracing::warn!("[keystore] could not store a generated DB key: {e}");
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!("[keystore] could not access DB key in keychain: {e}");
            None
        }
    }
}

/// Resolve the Turso auth token. Prefers the keychain; falls back to the env var
/// for backward compatibility. If the token only exists in the env, migrate it
/// into the keychain so future launches don't depend on `.env`.
pub fn turso_token() -> String {
    let env_token = crate::config::turso_token_env();
    match read(KEY_ACCOUNT_TURSO) {
        Ok(Some(t)) if !t.is_empty() => t,
        Ok(_) => {
            if !env_token.is_empty() {
                match set(KEY_ACCOUNT_TURSO, &env_token) {
                    Ok(()) => tracing::info!(
                        "[keystore] migrated TURSO_AUTH_TOKEN from .env into the OS keychain."
                    ),
                    Err(e) => tracing::warn!("[keystore] could not migrate Turso token: {e}"),
                }
            }
            env_token
        }
        Err(e) => {
            tracing::warn!("[keystore] could not access Turso token in keychain: {e}");
            env_token
        }
    }
}

/// Store (or clear, when empty) the Turso auth token. Returns `false` when the
/// keychain is unavailable — the caller must surface that rather than silently
/// dropping the secret, because we never write it to disk.
pub fn set_turso_token(token: &str) -> bool {
    let t = token.trim();
    let result = if t.is_empty() {
        entry(KEY_ACCOUNT_TURSO).and_then(|e| match e.delete_credential() {
            // Clearing a token that was never set is the requested end state.
            Err(KeyringError::NoEntry) => Ok(()),
            other => other,
        })
    } else {
        set(KEY_ACCOUNT_TURSO, t)
    };
    match result {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!("[keystore] could not write Turso token to keychain: {e}");
            false
        }
    }
}

/// Whether the OS credential store answered at all. Used by the UI to explain a
/// failed save instead of showing a silent no-op.
pub fn keychain_available() -> bool {
    !matches!(entry(KEY_ACCOUNT_DB), Err(KeyringError::NoDefaultStore))
}
