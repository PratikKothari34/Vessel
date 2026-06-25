'use strict';

/**
 * keystore.js — secrets at rest via the OS keychain (Windows Credential Manager
 * on this machine, libsecret / Keychain elsewhere) through keytar.
 *
 * Holds two secrets:
 *   - db-key:     32-byte random key (hex) that encrypts the local SQLite file
 *                 at rest via libSQL's native `encryptionKey`. Generated once on
 *                 first run and reused forever after.
 *   - turso-token: optional Turso cloud auth token, migrated out of .env so it
 *                 isn't left in plaintext on disk.
 *
 * Keytar stores values encrypted by the OS (DPAPI per-user on Windows), so the
 * key never lives on disk in cleartext. If keytar can't load (e.g. a stripped
 * environment), we fail safe: the DB key falls back to an env var if explicitly
 * provided, otherwise encryption is disabled rather than silently inventing a
 * key we can't persist (which would brick the existing DB on next launch).
 */

const crypto = require('crypto');

const SERVICE = 'scenario-chat';
const DB_KEY_ACCOUNT = 'db-encryption-key';
const TURSO_ACCOUNT = 'turso-auth-token';

let _keytar = null;
let _keytarTried = false;

// Lazy, soft require: keytar is native and may be unavailable. Never throw —
// callers degrade gracefully.
function keytar() {
  if (_keytarTried) return _keytar;
  _keytarTried = true;
  try {
    _keytar = require('keytar');
  } catch (e) {
    console.warn('[keystore] keytar unavailable — secrets will not use the OS keychain:', e.message);
    _keytar = null;
  }
  return _keytar;
}

function isAvailable() {
  return Boolean(keytar());
}

/**
 * Resolve the local-DB encryption key (hex string), or null if encryption can't
 * be enabled. Order:
 *   1. DB_ENCRYPTION_KEY env var (explicit override — lets advanced users BYO key).
 *   2. keychain value (created on first run if absent).
 *   3. null → caller opens the DB unencrypted (no keychain + no override).
 *
 * IMPORTANT: returning a *fresh* random key when one already exists would make
 * the existing encrypted DB unreadable. So we only ever generate when the
 * keychain has nothing AND no env override is set.
 */
async function getDbEncryptionKey() {
  const override = (process.env.DB_ENCRYPTION_KEY || '').trim();
  if (override) return override;

  const kt = keytar();
  if (!kt) return null; // no safe place to persist a generated key → stay plaintext

  try {
    let key = await kt.getPassword(SERVICE, DB_KEY_ACCOUNT);
    if (!key) {
      key = crypto.randomBytes(32).toString('hex'); // 256-bit
      await kt.setPassword(SERVICE, DB_KEY_ACCOUNT, key);
      console.log('[keystore] generated and stored a new local-DB encryption key.');
    }
    return key;
  } catch (e) {
    console.warn('[keystore] could not access DB key in keychain:', e.message);
    return null;
  }
}

/**
 * Resolve the Turso auth token. Prefers the keychain; falls back to the env var
 * for backward compatibility. If the token only exists in env, transparently
 * migrate it into the keychain so future launches don't depend on .env.
 */
async function getTursoToken() {
  const kt = keytar();
  const envToken = (process.env.TURSO_AUTH_TOKEN || '').trim();

  if (!kt) return envToken; // can't reach keychain — use env as-is

  try {
    let token = await kt.getPassword(SERVICE, TURSO_ACCOUNT);
    if (!token && envToken) {
      await kt.setPassword(SERVICE, TURSO_ACCOUNT, envToken);
      token = envToken;
      console.log('[keystore] migrated TURSO_AUTH_TOKEN from .env into the OS keychain.');
    }
    return token || envToken || '';
  } catch (e) {
    console.warn('[keystore] could not access Turso token in keychain:', e.message);
    return envToken;
  }
}

module.exports = { isAvailable, getDbEncryptionKey, getTursoToken };
