'use strict';

/**
 * settings.js — non-secret runtime settings persisted next to the local DB
 * (data/settings.json), written by PUT /settings from the in-app Settings
 * panel. A packaged install has no .env, so this file is how an end user's
 * own config (e.g. their Turso database URL) survives restarts.
 *
 * Secrets do NOT live here — the Turso auth token goes to the OS keychain
 * via keystore.js.
 *
 * Precedence: a key PRESENT in this file wins over the matching env var
 * ('' means the user explicitly cleared it); an ABSENT key falls back to
 * .env, so dev setups keep working untouched.
 */

const path = require('path');
const fs = require('fs');

const LOCAL_DB_PATH = process.env.LOCAL_DB_PATH || './data/scenario.db';
const FILE = path.join(path.dirname(path.resolve(LOCAL_DB_PATH)), 'settings.json');

let _cache = null;

function load() {
  if (_cache) return _cache;
  try {
    const v = JSON.parse(fs.readFileSync(FILE, 'utf8'));
    _cache = v && typeof v === 'object' && !Array.isArray(v) ? v : {};
  } catch {
    _cache = {}; // missing or corrupt file → defaults
  }
  return _cache;
}

function save(patch) {
  const next = { ...load(), ...patch };
  fs.mkdirSync(path.dirname(FILE), { recursive: true });
  // Write a sibling and rename over the target. A plain write truncates first,
  // so crashing between the truncate and the flush leaves a half-written file —
  // which parses as "no settings", which silently drops the user's Turso URL
  // and reopens their database on the other driver. rename() is atomic.
  const tmp = FILE + '.tmp';
  try {
    fs.writeFileSync(tmp, JSON.stringify(next, null, 2) + '\n');
    fs.renameSync(tmp, FILE);
  } catch (e) {
    try { fs.unlinkSync(tmp); } catch { /* nothing to clean up */ }
    throw e;
  }
  _cache = next;
  return next;
}

module.exports = { load, save, FILE };
