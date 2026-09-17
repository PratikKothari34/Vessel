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
// so a client can't smuggle arbitrary Ollama options through, and clamp each to
// a sane range so a hostile/imported value can't request a runaway context
// window (OOM / GPU exhaustion) or out-of-domain sampling params.
const SAMPLING_BOUNDS = {
  temperature: [0, 2],
  top_p: [0, 1],
  top_k: [0, 1000],
  min_p: [0, 1],
  repeat_penalty: [0, 4],
  num_ctx: [256, 131072],
  num_predict: [16, 4096],
};
const SAMPLING_KEYS = Object.keys(SAMPLING_BOUNDS);
function cleanSampling(s) {
  const out = {};
  if (s && typeof s === 'object') {
    for (const k of SAMPLING_KEYS) {
      const raw = s[k];
      // Number() is not a validator: it maps null, '', false and [] all to 0,
      // which is IN RANGE for every bound here. A client sending
      // { temperature: null } to mean "leave it alone" -- and JSON.stringify
      // turns NaN and Infinity into null too -- would otherwise have pinned the
      // character to greedy decoding forever. Absent means absent.
      if (raw === null || raw === undefined || raw === '') continue;
      if (typeof raw === 'boolean' || typeof raw === 'object') continue;
      const v = Number(raw);
      if (!Number.isFinite(v)) continue;
      const [min, max] = SAMPLING_BOUNDS[k];
      out[k] = Math.min(Math.max(v, min), max);
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

// Normalize a string array (chat starters / tags): drop anything that is not a
// string, trim, drop empties, cap size.
//
// Dropping rather than stringifying: String({}) is "[object Object]", which is
// content the user never typed being rendered as a tag. A non-string entry is
// a malformed list, not a list with an odd entry in it, and the Rust track has
// always read it that way.
function cleanList(a, { max = 12, maxLen = 200 } = {}) {
  if (!Array.isArray(a)) return [];
  return a.filter((x) => typeof x === 'string')
    .map((x) => x.trim().slice(0, maxLen))
    .filter(Boolean)
    .slice(0, max);
}

// Read a stored JSON string array back, applying the same caps the write path
// applies. See rowToCharacter for why the read path re-cleans at all.
function parseList(json, opts) {
  try { const v = JSON.parse(json || '[]'); return cleanList(Array.isArray(v) ? v : [], opts); }
  catch { return []; }
}

// The clamps above run on the way IN, which only covers rows this process
// wrote. A row can also arrive from the sync remote, written by another device
// or an older build, or be restored from a backup - none of those went through
// createCharacter. So the three fields that carry real power are re-cleaned on
// the way OUT as well, which is what response_style has always done:
//
//   sampling  spreads straight into the engine's options, so an unclamped
//             num_ctx is an OOM and an unknown key is a smuggled option
//   avatar    lands in an <img src>, so a file: URL makes a client fetch a
//             local path
//   the lists are rendered, and JSON.parse alone lets objects through where the
//             UI expects strings
//
// The prose fields are left as stored: they are inert text, already capped on
// write and bounded by the body limit, and truncating them here would quietly
// shorten a persona that is merely long.
function rowToCharacter(row) {
  let sampling = {};
  try { sampling = JSON.parse(row.sampling || '{}'); } catch { /* default {} */ }
  return {
    id: row.id,
    name: row.name,
    avatar: cleanAvatar(row.avatar),
    tagline: row.tagline || '',
    about: row.about || '',
    persona: row.persona || '',
    greeting: row.greeting || '',
    chatStarters: parseList(row.chat_starters),
    tags: parseList(row.tags, { maxLen: 40 }),
    sampling: cleanSampling(sampling),
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
  isValidId,
  listCharacters,
  getCharacter,
  createCharacter,
  updateCharacter,
  deleteCharacter,
  // Exposed for the tests that plant a row the write path never saw, which is
  // what a synced or restored row looks like from here. Not part of the API.
  _internals: { rowToCharacter },
};
