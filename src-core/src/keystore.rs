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

fn entry(account: &str) -> Result<Entry, KeyringError> {
    Entry::new(KEYCHAIN_SERVICE, account)
}

/// `Ok(None)` means "no such credential", which is a normal first-run state and
/// must not be confused with "the keychain is unreachable".
fn read(account: &str) -> Result<Option<String>, KeyringError> {
    match entry(account)?.get_password() {
        Ok(v) => Ok(Some(v)),
        Err(KeyringError::NoEntry) => Ok(None),
        Err(e) => Err(e),
    }
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
            match entry(KEY_ACCOUNT_DB).and_then(|e| e.set_password(&key)) {
                Ok(()) => {
                    tracing::info!("[keystore] generated and stored a new local-DB encryption key.");
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
                match entry(KEY_ACCOUNT_TURSO).and_then(|e| e.set_password(&env_token)) {
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
    let result = entry(KEY_ACCOUNT_TURSO).and_then(|e| {
        if t.is_empty() {
            match e.delete_credential() {
                // Clearing a token that was never set is the requested end state.
                Err(KeyringError::NoEntry) => Ok(()),
                other => other,
            }
        } else {
            e.set_password(t)
        }
    });
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
