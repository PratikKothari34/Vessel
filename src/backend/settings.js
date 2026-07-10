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
  fs.writeFileSync(FILE, JSON.stringify(next, null, 2) + '\n');
  _cache = next;
  return next;
}

module.exports = { load, save, FILE };
