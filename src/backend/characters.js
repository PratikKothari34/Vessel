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

// Field length caps (defense in depth — the body limit is the only other bound).
// avatar is short on purpose: a URL fits; a giant data: URI does not (bloats DB + sync).
const CAP = { avatar: 4096, about: 8000, persona: 16000, greeting: 8000 };
function cap(v, max) { return String(v == null ? '' : v).slice(0, max); }

// Avatar is rendered into an <img src>. Persist only web image URLs / inline
// image data; reject anything else (file:, javascript:, etc.) so a hostile or
// imported value can never make a client fetch a local path. Empty = use the
// generated glyph fallback.
function cleanAvatar(v) {
  const s = cap(v, CAP.avatar).trim();
  if (/^https?:\/\//i.test(s) || /^data:image\//i.test(s)) return s;
  return '';
}

// Normalize a string array (chat starters / tags): trim, drop empties, cap size.
function cleanList(a, { max = 12, maxLen = 200 } = {}) {
  if (!Array.isArray(a)) return [];
  return a.map((x) => String(x == null ? '' : x).trim().slice(0, maxLen))
    .filter(Boolean)
    .slice(0, max);
}

function parseList(json) {
  try { const v = JSON.parse(json || '[]'); return Array.isArray(v) ? v : []; }
  catch { return []; }
}

function rowToCharacter(row) {
  let sampling = {};
  try { sampling = JSON.parse(row.sampling || '{}'); } catch { /* default {} */ }
  return {
    id: row.id,
    name: row.name,
    avatar: row.avatar || '',
    tagline: row.tagline || '',
    about: row.about || '',
    persona: row.persona || '',
    greeting: row.greeting || '',
    chatStarters: parseList(row.chat_starters),
    tags: parseList(row.tags),
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

async function createCharacter({ name, avatar, tagline, about, persona, greeting, chatStarters, tags, sampling, responseStyle }) {
  const clean = String(name || '').trim();
  if (!clean) throw new Error('Character name is required.');
  const db = await getDb();
  const id = newId();
  const ts = nowIso();
  await db.execute({
    sql: `INSERT INTO characters
            (id, name, avatar, tagline, about, persona, greeting, chat_starters, tags, sampling, response_style, created_at, updated_at)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    args: [
      id, clean.slice(0, 120), cleanAvatar(avatar),
      cap(tagline, 200), cap(about, CAP.about),
      cap(persona, CAP.persona), cap(greeting, CAP.greeting),
      JSON.stringify(cleanList(chatStarters)), JSON.stringify(cleanList(tags, { maxLen: 40 })),
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
    avatar: patch.avatar != null ? cleanAvatar(patch.avatar) : existing.avatar,
    tagline: patch.tagline != null ? cap(patch.tagline, 200) : existing.tagline,
    about: patch.about != null ? cap(patch.about, CAP.about) : existing.about,
    persona: patch.persona != null ? cap(patch.persona, CAP.persona) : existing.persona,
    greeting: patch.greeting != null ? cap(patch.greeting, CAP.greeting) : existing.greeting,
    chatStarters: patch.chatStarters != null ? cleanList(patch.chatStarters) : existing.chatStarters,
    tags: patch.tags != null ? cleanList(patch.tags, { maxLen: 40 }) : existing.tags,
    sampling: patch.sampling != null ? cleanSampling(patch.sampling) : existing.sampling,
    responseStyle: patch.responseStyle != null ? cleanStyle(patch.responseStyle) : existing.responseStyle,
  };
  if (!merged.name) throw new Error('Character name is required.');
  const db = await getDb();
  await db.execute({
    sql: `UPDATE characters SET name=?, avatar=?, tagline=?, about=?, persona=?, greeting=?,
            chat_starters=?, tags=?, sampling=?, response_style=?, updated_at=?
          WHERE id=?`,
    args: [
      merged.name, merged.avatar, merged.tagline, merged.about, merged.persona, merged.greeting,
      JSON.stringify(merged.chatStarters), JSON.stringify(merged.tags),
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
