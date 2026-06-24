'use strict';

/**
 * memory.js — long-term memory engine (Turso/libSQL backed).
 *
 * Keeps the model's live window small (fast) while preserving unlimited history:
 *
 *   live window = [character persona] + [conversation rules]
 *               + [rolling summary] + [retrieved snippets]
 *               + [last N verbatim turns] + [newest user message]
 *
 * - Rolling summary : SUMMARIZER_MODEL (gemma3:4b) condenses old turns.
 * - Retrieval       : EMBED_MODEL (nomic-embed-text) embeds turns; native libSQL
 *                     vector_top_k finds relevant archived turns by cosine.
 * - Persistence     : Turso tables (see db.js). No JSON files.
 *
 * Adapted from the reference Natsumura memory.js (JSON + in-JS cosine) — same
 * algorithm, swapped storage to SQL and cosine to native vector search.
 */

const crypto = require('crypto');
const { getDb, toVectorArg, EMBED_DIM } = require('./db');

// ---- Config --------------------------------------------------------------
const OLLAMA_HOST = (process.env.OLLAMA_HOST || 'http://localhost:11434').replace(/\/+$/, '');
const SUMMARIZER_MODEL = process.env.SUMMARIZER_MODEL || 'gemma3:4b';
const EMBED_MODEL = process.env.EMBED_MODEL || 'nomic-embed-text';
const EMBED_NUM_GPU = (() => {
  const v = parseInt(process.env.EMBED_NUM_GPU, 10);
  return Number.isFinite(v) ? v : 0;
})();

function intEnv(name, def, { min = 1 } = {}) {
  const v = parseInt(process.env[name], 10);
  return Number.isFinite(v) && v >= min ? v : def;
}
function floatEnv(name, def, { min = -Infinity, max = Infinity } = {}) {
  const v = parseFloat(process.env[name]);
  return Number.isFinite(v) && v >= min && v <= max ? v : def;
}

const SUMMARIZER_NUM_CTX = intEnv('SUMMARIZER_NUM_CTX', 8192, { min: 512 });
const VERBATIM_TURNS = intEnv('VERBATIM_TURNS', 8);
let SUMMARIZE_THRESHOLD = intEnv('SUMMARIZE_THRESHOLD', 12);
if (SUMMARIZE_THRESHOLD <= VERBATIM_TURNS) SUMMARIZE_THRESHOLD = VERBATIM_TURNS + 4;
const RETRIEVE_K = intEnv('RETRIEVE_K', 4);
const RETRIEVE_MIN_SCORE = floatEnv('RETRIEVE_MIN_SCORE', 0.45, { min: -1, max: 1 });
const MAX_SUMMARY_CHARS = intEnv('MAX_SUMMARY_CHARS', 6000, { min: 500 });

function nowIso() { return new Date().toISOString(); }
function newId() { return crypto.randomUUID(); }
function isValidId(id) { return typeof id === 'string' && /^[a-zA-Z0-9_-]{1,64}$/.test(id); }

// ---- Conversation state --------------------------------------------------

// Ensure a conversation row exists; create it (optionally bound to a character)
// if missing. Returns the conversation row.
async function ensureConversation(id, characterId = null) {
  const db = await getDb();
  const res = await db.execute({ sql: 'SELECT * FROM conversations WHERE id = ?', args: [id] });
  if (res.rows.length) return res.rows[0];
  const ts = nowIso();
  await db.execute({
    sql: `INSERT INTO conversations (id, character_id, title, summary, created_at, updated_at)
          VALUES (?, ?, '', '', ?, ?)`,
    args: [id, characterId, ts, ts],
  });
  const again = await db.execute({ sql: 'SELECT * FROM conversations WHERE id = ?', args: [id] });
  return again.rows[0];
}

async function getSummary(id) {
  const db = await getDb();
  const res = await db.execute({ sql: 'SELECT summary FROM conversations WHERE id = ?', args: [id] });
  return res.rows.length ? (res.rows[0].summary || '') : '';
}

async function touchConversation(id) {
  const db = await getDb();
  await db.execute({ sql: 'UPDATE conversations SET updated_at = ? WHERE id = ?', args: [nowIso(), id] });
}

// Recent verbatim turns in chronological order.
async function getVerbatim(id) {
  const db = await getDb();
  const res = await db.execute({
    sql: 'SELECT role, content FROM turns WHERE conversation_id = ? ORDER BY id ASC',
    args: [id],
  });
  return res.rows.map((r) => ({ role: r.role, content: r.content }));
}

async function hasArchive(id) {
  const db = await getDb();
  const res = await db.execute({
    sql: 'SELECT 1 FROM archive WHERE conversation_id = ? LIMIT 1', args: [id],
  });
  return res.rows.length > 0;
}

// ---- Ollama helpers ------------------------------------------------------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function fetchRetry(url, opts, tries = 3) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, opts);
      if (res.ok) return res;
      if (res.status < 500) throw new Error(`${res.status} ${await res.text().catch(() => '')}`);
      lastErr = new Error(`${res.status} ${await res.text().catch(() => '')}`);
    } catch (e) { lastErr = e; }
    if (i < tries - 1) await sleep(400 * (i + 1));
  }
  throw lastErr;
}

async function embed(text) {
  const prompt = typeof text === 'string' ? text : String(text == null ? '' : text);
  if (!prompt.trim()) throw new Error('embed: empty text');
  const body = { model: EMBED_MODEL, prompt };
  if (EMBED_NUM_GPU >= 0) body.options = { num_gpu: EMBED_NUM_GPU };
  const res = await fetchRetry(`${OLLAMA_HOST}/api/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!Array.isArray(data.embedding) || data.embedding.length === 0) {
    throw new Error('embed: model returned no embedding');
  }
  if (data.embedding.length !== EMBED_DIM) {
    throw new Error(`embed: expected ${EMBED_DIM} dims, got ${data.embedding.length}`);
  }
  return data.embedding;
}

async function generate(model, prompt) {
  const res = await fetchRetry(`${OLLAMA_HOST}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model, prompt, stream: false,
      options: { num_ctx: SUMMARIZER_NUM_CTX },
    }),
  });
  const data = await res.json();
  return (data.response || '').trim();
}

// ---- Per-conversation lock ----------------------------------------------
// Serialize turns per conversation so post-stream bookkeeping (summarize/embed)
// never races the next request's state read/write.
const _locks = new Map();
const LOCK_WAIT_MS = intEnv('LOCK_WAIT_MS', 120000, { min: 1000 });

async function acquireLock(id) {
  const prev = _locks.get(id) || Promise.resolve();
  let release;
  const next = new Promise((r) => { release = r; });
  _locks.set(id, prev.then(() => next));

  let timer;
  const waited = new Promise((r) => { timer = setTimeout(r, LOCK_WAIT_MS); });
  await Promise.race([prev.catch(() => {}), waited]);
  clearTimeout(timer);

  let released = false;
  return () => {
    if (released) return;
    released = true;
    release();
    if (_locks.get(id) === next) _locks.delete(id);
  };
}

// ---- Retrieval -----------------------------------------------------------

/**
 * Up to k archived turns from THIS conversation most relevant to queryText,
 * above the score threshold, in chronological order.
 *
 * libSQL vector_top_k searches the whole index (can't pre-filter by
 * conversation), so we over-fetch then filter by conversation_id + score.
 */
async function retrieve(conversationId, queryText, k = RETRIEVE_K) {
  if (k <= 0 || typeof queryText !== 'string' || !queryText.trim()) return [];
  if (!(await hasArchive(conversationId))) return [];

  const db = await getDb();
  const q = await embed(queryText);
  const qArg = toVectorArg(q);
  // Over-fetch: the global vector index may interleave other conversations'
  // turns, so pull extra then filter by conversation_id below.
  // NOTE: vector_top_k's k MUST be a literal integer — a bound `?` param is
  // rejected ("third parameter (k) must be a non-negative integer"). fetchK is
  // an internally-computed int (never user input), so inlining it is safe.
  const fetchK = Math.max(Math.min(k, 1000) * 8, 32);

  const res = await db.execute({
    sql: `
      SELECT a.id, a.role, a.content,
             1 - vector_distance_cos(a.embedding, vector32(?)) AS score
      FROM vector_top_k('archive_vec_idx', vector32(?), ${fetchK}) AS vt
      JOIN archive a ON a.id = vt.id
      WHERE a.conversation_id = ?
      ORDER BY score DESC`,
    args: [qArg, qArg, conversationId],
  });

  const hits = res.rows
    .filter((r) => Number(r.score) >= RETRIEVE_MIN_SCORE)
    .slice(0, k)
    .sort((a, b) => Number(a.id) - Number(b.id)); // chronological

  return hits.map((r) => ({ role: r.role, content: r.content, score: Number(r.score) }));
}

// ---- Summarization -------------------------------------------------------

function renderTurns(turns, assistantName = 'Character') {
  return turns
    .map((t) => `${t.role === 'user' ? 'User' : assistantName}: ${t.content}`)
    .join('\n');
}

async function summarize(priorSummary, turns, assistantName) {
  const prompt = [
    'You are a story archivist. Maintain a running summary of an ongoing',
    'roleplay so it can be remembered later. Update the summary below with the',
    'new exchanges. Preserve concrete facts: character names, relationships,',
    'locations, plot events, decisions, and unresolved threads. Be faithful and',
    'concise. Do not add disclaimers, opinions, or content not present in the',
    'text. Output ONLY the updated summary prose.',
    '',
    '=== CURRENT SUMMARY ===',
    priorSummary || '(none yet)',
    '',
    '=== NEW EXCHANGES TO FOLD IN ===',
    renderTurns(turns, assistantName),
    '',
    '=== UPDATED SUMMARY ===',
  ].join('\n');
  return generate(SUMMARIZER_MODEL, prompt);
}

// ---- Context assembly ----------------------------------------------------

const RETRIEVAL_HEADER =
  '[Relevant earlier moments recalled from this story — use them for continuity:]';
const SUMMARY_HEADER = '[Story so far — summary of earlier events:]';

function isValidMsg(m) {
  return m && typeof m === 'object' &&
    (m.role === 'system' || m.role === 'user' || m.role === 'assistant') &&
    typeof m.content === 'string';
}

/**
 * Build the messages array the chat model receives. `leadingSystems` are
 * caller-supplied system messages (character persona + conversation rules)
 * placed first.
 */
async function buildContext(conversationId, incoming, leadingSystems = []) {
  const valid = Array.isArray(incoming) ? incoming.filter(isValidMsg) : [];
  const latestUser = [...valid].reverse().find((m) => m.role === 'user') || null;
  const queryText = latestUser ? latestUser.content : '';

  const messages = [];
  for (const s of leadingSystems) if (isValidMsg(s)) messages.push(s);

  const summary = await getSummary(conversationId);
  if (summary) messages.push({ role: 'system', content: `${SUMMARY_HEADER}\n${summary}` });

  let retrieved = [];
  if (queryText) {
    try { retrieved = await retrieve(conversationId, queryText); } catch { retrieved = []; }
  }
  if (retrieved.length) {
    messages.push({ role: 'system', content: `${RETRIEVAL_HEADER}\n${renderTurns(retrieved)}` });
  }

  const verbatim = await getVerbatim(conversationId);
  for (const t of verbatim) messages.push({ role: t.role, content: t.content });
  if (latestUser && !alreadyLast(verbatim, latestUser)) {
    messages.push({ role: 'user', content: latestUser.content });
  }

  return { messages, retrieved, latestUser };
}

function alreadyLast(verbatim, msg) {
  const last = verbatim[verbatim.length - 1];
  return last && last.role === msg.role && last.content === msg.content;
}

// Remove the most recent assistant turn (regenerate/rephrase).
async function dropLastAssistant(conversationId) {
  const db = await getDb();
  const res = await db.execute({
    sql: 'SELECT id, role FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT 1',
    args: [conversationId],
  });
  if (res.rows.length && res.rows[0].role === 'assistant') {
    await db.execute({ sql: 'DELETE FROM turns WHERE id = ?', args: [res.rows[0].id] });
    return true;
  }
  return false;
}

// ---- Post-turn bookkeeping ----------------------------------------------

/**
 * Record the user turn + assistant reply. When verbatim count exceeds the
 * threshold, fold the oldest turns into the rolling summary and archive them
 * with embeddings. Runs after streaming so the user never waits on it.
 */
async function recordTurn(conversationId, userMessage, assistantReply, assistantName = 'Character') {
  const db = await getDb();
  const verbatim = await getVerbatim(conversationId);

  if (userMessage && !alreadyLast(verbatim, userMessage)) {
    await db.execute({
      sql: 'INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)',
      args: [conversationId, 'user', userMessage.content, nowIso()],
    });
  }
  await db.execute({
    sql: 'INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)',
    args: [conversationId, 'assistant', assistantReply, nowIso()],
  });
  await touchConversation(conversationId);

  const count = (await db.execute({
    sql: 'SELECT COUNT(*) AS n FROM turns WHERE conversation_id = ?', args: [conversationId],
  })).rows[0].n;

  if (Number(count) <= SUMMARIZE_THRESHOLD) {
    return { archived: 0, summarized: false };
  }

  // Fold the oldest (count - VERBATIM_TURNS) turns into summary + archive.
  const toArchiveCount = Number(count) - VERBATIM_TURNS;
  const oldest = (await db.execute({
    sql: 'SELECT id, role, content FROM turns WHERE conversation_id = ? ORDER BY id ASC LIMIT ?',
    args: [conversationId, toArchiveCount],
  })).rows;

  const priorSummary = await getSummary(conversationId);
  let updated;
  try {
    updated = await summarize(priorSummary, oldest, assistantName);
    if (updated.length > MAX_SUMMARY_CHARS) updated = updated.slice(updated.length - MAX_SUMMARY_CHARS);
  } catch (e) {
    // Summarizer down: keep turns verbatim rather than lose them.
    return { archived: 0, summarized: false, error: e.message };
  }
  await db.execute({
    sql: 'UPDATE conversations SET summary = ?, updated_at = ? WHERE id = ?',
    args: [updated, nowIso(), conversationId],
  });

  // Move folded turns from verbatim -> archive (with embeddings).
  let archived = 0;
  for (const turn of oldest) {
    let embeddingArg = null;
    try {
      const vec = await embed(`${turn.role}: ${turn.content}`);
      embeddingArg = toVectorArg(vec);
    } catch { /* keep content, drop vector (not retrievable) */ }

    if (embeddingArg) {
      await db.execute({
        sql: `INSERT INTO archive (conversation_id, role, content, embedding, created_at)
              VALUES (?, ?, ?, vector32(?), ?)`,
        args: [conversationId, turn.role, turn.content, embeddingArg, nowIso()],
      });
    } else {
      await db.execute({
        sql: `INSERT INTO archive (conversation_id, role, content, embedding, created_at)
              VALUES (?, ?, ?, NULL, ?)`,
        args: [conversationId, turn.role, turn.content, nowIso()],
      });
    }
    await db.execute({ sql: 'DELETE FROM turns WHERE id = ?', args: [turn.id] });
    archived++;
  }

  return { archived, summarized: true };
}

// ---- Read-only introspection (UI sidebar + memory inspector) -------------

async function listConversations(characterId = null) {
  const db = await getDb();
  const sql = characterId
    ? `SELECT c.*, (SELECT content FROM turns t WHERE t.conversation_id = c.id ORDER BY t.id DESC LIMIT 1) AS last_turn,
              (SELECT COUNT(*) FROM turns t WHERE t.conversation_id = c.id) +
              (SELECT COUNT(*) FROM archive a WHERE a.conversation_id = c.id) AS turn_count
       FROM conversations c WHERE c.character_id = ? ORDER BY c.updated_at DESC`
    : `SELECT c.*, (SELECT content FROM turns t WHERE t.conversation_id = c.id ORDER BY t.id DESC LIMIT 1) AS last_turn,
              (SELECT COUNT(*) FROM turns t WHERE t.conversation_id = c.id) +
              (SELECT COUNT(*) FROM archive a WHERE a.conversation_id = c.id) AS turn_count
       FROM conversations c ORDER BY c.updated_at DESC`;
  const res = await db.execute(characterId ? { sql, args: [characterId] } : sql);
  return res.rows.map((r) => ({
    id: r.id,
    characterId: r.character_id,
    title: r.title || '',
    createdAt: r.created_at,
    updatedAt: r.updated_at,
    hasSummary: Boolean(r.summary && r.summary.length),
    turnCount: Number(r.turn_count || 0),
    preview: r.last_turn ? String(r.last_turn).slice(0, 120) : '',
  }));
}

// Full inspectable view (summary + verbatim + archived turns, no raw vectors).
async function getConversation(id) {
  if (!isValidId(id)) return null;
  const db = await getDb();
  const c = await db.execute({ sql: 'SELECT * FROM conversations WHERE id = ?', args: [id] });
  if (!c.rows.length) return null;
  const row = c.rows[0];
  const verbatim = await getVerbatim(id);
  const arch = await db.execute({
    sql: `SELECT role, content, embedding IS NOT NULL AS has_embedding
          FROM archive WHERE conversation_id = ? ORDER BY id ASC`,
    args: [id],
  });
  return {
    id: row.id,
    characterId: row.character_id,
    title: row.title || '',
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    summary: row.summary || '',
    verbatim,
    archive: arch.rows.map((r) => ({
      role: r.role, content: r.content, hasEmbedding: Boolean(r.has_embedding),
    })),
  };
}

async function setTitle(id, title) {
  if (!isValidId(id)) throw new Error('setTitle: invalid id');
  const db = await getDb();
  await db.execute({
    sql: 'UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?',
    args: [typeof title === 'string' ? title.slice(0, 200) : '', nowIso(), id],
  });
  return { title: typeof title === 'string' ? title.slice(0, 200) : '' };
}

async function deleteConversation(id) {
  if (!isValidId(id)) throw new Error('deleteConversation: invalid id');
  const db = await getDb();
  const res = await db.execute({ sql: 'DELETE FROM conversations WHERE id = ?', args: [id] });
  return res.rowsAffected > 0;
}

module.exports = {
  newId,
  isValidId,
  ensureConversation,
  buildContext,
  recordTurn,
  acquireLock,
  dropLastAssistant,
  listConversations,
  getConversation,
  setTitle,
  deleteConversation,
  // exported for testing
  embed,
  retrieve,
  summarize,
  _config: {
    SUMMARIZER_MODEL, SUMMARIZER_NUM_CTX, EMBED_MODEL, EMBED_NUM_GPU,
    VERBATIM_TURNS, SUMMARIZE_THRESHOLD, RETRIEVE_K, RETRIEVE_MIN_SCORE,
  },
};
