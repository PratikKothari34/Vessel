'use strict';

/**
 * characters.js — CRUD for roleplay characters.
 *
 * A character is the persona the model plays: name, avatar, persona text,
 * opening greeting, and optional per-character sampling overrides (temperature,
 * top_p, ...). The persona is injected as a system message at chat time, so
 * each character behaves distinctly without rebuilding the model.
 */

const crypto = require('crypto');
const { getDb } = require('./db');

function nowIso() {
  return new Date().toISOString();
}

function newId() {
  return crypto.randomUUID();
}

// ids we mint are UUIDs; reject anything else to keep them safe in queries/paths.
function isValidId(id) {
  return typeof id === 'string' && /^[a-zA-Z0-9_-]{1,64}$/.test(id);
}

// Sampling overrides are stored as a JSON string. Keep only known numeric keys
// so a client can't smuggle arbitrary Ollama options through.
const SAMPLING_KEYS = ['temperature', 'top_p', 'top_k', 'min_p', 'repeat_penalty', 'num_ctx'];
function cleanSampling(s) {
  const out = {};
  if (s && typeof s === 'object') {
    for (const k of SAMPLING_KEYS) {
      const v = Number(s[k]);
      if (Number.isFinite(v)) out[k] = v;
    }
  }
  return out;
}

const RESPONSE_STYLES = ['balanced', 'dialogue', 'narration-light'];
function cleanStyle(s) {
  return RESPONSE_STYLES.includes(s) ? s : 'balanced';
}

function rowToCharacter(row) {
  let sampling = {};
  try { sampling = JSON.parse(row.sampling || '{}'); } catch { /* default {} */ }
  return {
    id: row.id,
    name: row.name,
    avatar: row.avatar || '',
    persona: row.persona || '',
    greeting: row.greeting || '',
    sampling,
    responseStyle: cleanStyle(row.response_style),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

async function listCharacters() {
  const db = await getDb();
  const res = await db.execute('SELECT * FROM characters ORDER BY updated_at DESC');
  return res.rows.map(rowToCharacter);
}

async function getCharacter(id) {
  if (!isValidId(id)) return null;
  const db = await getDb();
  const res = await db.execute({ sql: 'SELECT * FROM characters WHERE id = ?', args: [id] });
  return res.rows.length ? rowToCharacter(res.rows[0]) : null;
}

async function createCharacter({ name, avatar, persona, greeting, sampling, responseStyle }) {
  const clean = String(name || '').trim();
  if (!clean) throw new Error('Character name is required.');
  const db = await getDb();
  const id = newId();
  const ts = nowIso();
  await db.execute({
    sql: `INSERT INTO characters (id, name, avatar, persona, greeting, sampling, response_style, created_at, updated_at)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    args: [
      id, clean.slice(0, 120), String(avatar || ''),
      String(persona || ''), String(greeting || ''),
      JSON.stringify(cleanSampling(sampling)), cleanStyle(responseStyle), ts, ts,
    ],
  });
  return getCharacter(id);
}

async function updateCharacter(id, patch) {
  if (!isValidId(id)) throw new Error('Invalid character id.');
  const existing = await getCharacter(id);
  if (!existing) return null;
  const merged = {
    name: patch.name != null ? String(patch.name).trim().slice(0, 120) : existing.name,
    avatar: patch.avatar != null ? String(patch.avatar) : existing.avatar,
    persona: patch.persona != null ? String(patch.persona) : existing.persona,
    greeting: patch.greeting != null ? String(patch.greeting) : existing.greeting,
    sampling: patch.sampling != null ? cleanSampling(patch.sampling) : existing.sampling,
    responseStyle: patch.responseStyle != null ? cleanStyle(patch.responseStyle) : existing.responseStyle,
  };
  if (!merged.name) throw new Error('Character name is required.');
  const db = await getDb();
  await db.execute({
    sql: `UPDATE characters SET name=?, avatar=?, persona=?, greeting=?, sampling=?, response_style=?, updated_at=?
          WHERE id=?`,
    args: [
      merged.name, merged.avatar, merged.persona, merged.greeting,
      JSON.stringify(merged.sampling), merged.responseStyle, nowIso(), id,
    ],
  });
  return getCharacter(id);
}

// Deletes the character and (via FK cascade) all its conversations/turns/archive.
async function deleteCharacter(id) {
  if (!isValidId(id)) throw new Error('Invalid character id.');
  const db = await getDb();
  const res = await db.execute({ sql: 'DELETE FROM characters WHERE id = ?', args: [id] });
  return res.rowsAffected > 0;
}

module.exports = {
  newId,
  isValidId,
  listCharacters,
  getCharacter,
  createCharacter,
  updateCharacter,
  deleteCharacter,
};
