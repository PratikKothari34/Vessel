'use strict';

/**
 * memory.js — long-term memory engine (Turso backed; see db.js for the driver).
 *
 * Keeps the model's live window small (fast) while preserving unlimited history:
 *
 *   live window = [character persona] + [conversation rules]
 *               + [rolling summary] + [retrieved snippets]
 *               + [last N verbatim turns] + [newest user message]
 *
 * - Rolling summary : SUMMARIZER_MODEL (gemma3:4b) condenses old turns.
 * - Retrieval       : EMBED_MODEL (nomic-embed-text) embeds turns; archived
 *                     embeddings are cosine-ranked in JS (see retrieve()).
 * - Persistence     : Turso tables (see db.js). No JSON files.
 *
 * Adapted from the reference Natsumura memory.js (JSON + in-JS cosine) — same
 * algorithm, swapped storage to SQL.
 */

const crypto = require('crypto');
const { getDb, encodeEmbedding, decodeEmbedding, EMBED_DIM } = require('./db');
const inference = require('./inference');

// ---- Config --------------------------------------------------------------
const CHAT_MODEL = process.env.OLLAMA_MODEL || 'vessel';
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

// Ollama keys a resident model instance on (model, num_ctx) -- not on the model
// alone. (On llama-server this whole problem is gone: one process, one window,
// fixed at launch, nothing to evict. The backend warns and ignores the value.)
// A summariser that differs on EITHER evicts the chat model, and the next
// user message pays a full reload before its first token. Measured on an 8 GB
// 4060: gemma3:4b costs 19.4 s, vessel at a mismatched num_ctx costs 18.6 s, and
// vessel at the SAME num_ctx costs 3 ms. So when the summariser is the chat
// model, default its window to the chat window and keep the one slot warm.
const CHAT_NUM_CTX = intEnv('OLLAMA_NUM_CTX', 12288, { min: 256 });
const SUMMARIZER_NUM_CTX = intEnv(
  'SUMMARIZER_NUM_CTX',
  SUMMARIZER_MODEL === CHAT_MODEL ? CHAT_NUM_CTX : 8192,
  { min: 512 },
);
const VERBATIM_TURNS = intEnv('VERBATIM_TURNS', 8);
let SUMMARIZE_THRESHOLD = intEnv('SUMMARIZE_THRESHOLD', 12);
if (SUMMARIZE_THRESHOLD <= VERBATIM_TURNS) SUMMARIZE_THRESHOLD = VERBATIM_TURNS + 4;
const RETRIEVE_K = intEnv('RETRIEVE_K', 4);
const RETRIEVE_MIN_SCORE = floatEnv('RETRIEVE_MIN_SCORE', 0.45, { min: -1, max: 1 });
const MAX_SUMMARY_CHARS = intEnv('MAX_SUMMARY_CHARS', 6000, { min: 500 });
// How many conversations keep their archive vectors resident. 768 floats x 4 B
// = 3 KB per archived turn, so a 1,000-turn story costs ~3 MB.
const ARCHIVE_CACHE_CONVS = intEnv('ARCHIVE_CACHE_CONVS', 8);
// Parallel embed calls when folding turns into the archive. Ollama serializes
// work on one model, but overlapping the HTTP + tokenize legs still helps, and
// this whole phase now runs underneath the summarizer call anyway.
const EMBED_CONCURRENCY = intEnv('EMBED_CONCURRENCY', 4);

function nowIso() { return new Date().toISOString(); }
function newId() { return crypto.randomUUID(); }
function isValidId(id) { return typeof id === 'string' && /^[a-zA-Z0-9_-]{1,64}$/.test(id); }

// ---- Conversation state --------------------------------------------------

// Ensure a conversation row exists; create it (optionally bound to a character)
// if missing. Returns the conversation row.
//
// character_id is a FK, so a characterId that no longer exists would make the
// INSERT fail with a raw "FOREIGN KEY constraint failed". Callers should reject
// a missing character up front (see /chat), but drop the binding rather than
// throwing if one slips through: an unbound conversation is recoverable, a
// hard 500 mid-chat is not.
async function ensureConversation(id, characterId = null) {
  const db = await getDb();
  const res = await db.execute({ sql: 'SELECT * FROM conversations WHERE id = ?', args: [id] });
  if (res.rows.length) return res.rows[0];
  let boundId = characterId || null;
  if (boundId) {
    const c = await db.execute({ sql: 'SELECT 1 FROM characters WHERE id = ?', args: [boundId] });
    if (!c.rows.length) boundId = null;
  }
  const ts = nowIso();
  await db.execute({
    sql: `INSERT INTO conversations (id, character_id, title, summary, created_at, updated_at)
          VALUES (?, ?, '', '', ?, ?)`,
    args: [id, boundId, ts, ts],
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

// ---- Engine calls --------------------------------------------------------
// Transport and per-backend quirks live in inference/; what stays here is the
// part that is about memory, not about the engine: the dimension contract the
// archive depends on, and the summarizer's window.

async function embed(text) {
  const prompt = typeof text === 'string' ? text : String(text == null ? '' : text);
  if (!prompt.trim()) throw new Error('embed: empty text');
  const vec = await inference.embed(EMBED_MODEL, prompt, { numGpu: EMBED_NUM_GPU });
  // Every archived row is stored at EMBED_DIM. A model that returns anything
  // else would write vectors that can never be compared against the existing
  // ones, so this fails the turn rather than corrupting the archive.
  if (vec.length !== EMBED_DIM) {
    throw new Error(`embed: expected ${EMBED_DIM} dims, got ${vec.length}`);
  }
  return vec;
}

function generate(model, prompt) {
  return inference.generate(model, prompt, { numCtx: SUMMARIZER_NUM_CTX });
}

// ---- Per-conversation lock ----------------------------------------------
// Serialize turns per conversation so post-stream bookkeeping (summarize/embed)
// never races the next request's state read/write.
const _locks = new Map();
const LOCK_WAIT_MS = intEnv('LOCK_WAIT_MS', 120000, { min: 1000 });

// waitMs caps how long the caller queues behind the holder before giving up and
// proceeding anyway. A turn is willing to wait out a whole generation; a delete
// is not -- it should not leave a UI button spinning for two minutes because a
// stream is stuck, and proceeding unlocked is exactly the old behaviour.
async function acquireLock(id, waitMs = LOCK_WAIT_MS) {
  const prev = _locks.get(id) || Promise.resolve();
  let release;
  const next = new Promise((r) => { release = r; });
  _locks.set(id, prev.then(() => next));

  let timer;
  const waited = new Promise((r) => { timer = setTimeout(r, waitMs); });
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

// Scale a vector to unit length IN PLACE. Every vector that enters the cache and
// every query vector is normalized on the way in, which turns cosine similarity
// into a plain dot product: the two sqrt() calls and the two norm accumulator
// loops per candidate row disappear from the hot scan entirely.
function normalizeInPlace(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  if (n === 0) return v;
  const inv = 1 / Math.sqrt(n);
  for (let i = 0; i < v.length; i++) v[i] *= inv;
  return v;
}

// Cosine similarity of two ALREADY-NORMALIZED vectors.
function dot(a, b) {
  let d = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) d += a[i] * b[i];
  return d;
}

// ---- Archive vector cache ------------------------------------------------
// Retrieval used to re-read and re-decode every archived embedding on EVERY
// turn: 3 KB per row off disk, a Float32 decode per row, then two sqrt loops per
// row. None of that changes between turns -- the archive is append-only.
//
// So keep the decoded, normalized vectors resident per conversation and load
// only rows newer than the high-water mark. Steady state is zero row reads and
// zero decodes; a turn that archives adds a handful.
//
// Correctness against cloud sync: a pull can insert archive rows with ids BELOW
// our high-water mark (another device's autoincrement), or a restore can shrink
// the table. Neither is caught by a max(id) check alone, so we carry the row
// count too and rebuild from scratch whenever it drops.
//
// LRU by insertion order: delete-then-set makes the first key the oldest.
const _archiveCache = new Map(); // convId -> { ids:[], vecs:[], maxId, rowsSeen }

function _touchCache(conversationId) {
  let e = _archiveCache.get(conversationId);
  if (e) _archiveCache.delete(conversationId);
  else e = { ids: [], vecs: [], maxId: 0, rowsSeen: 0 };
  _archiveCache.set(conversationId, e);
  while (_archiveCache.size > ARCHIVE_CACHE_CONVS) {
    _archiveCache.delete(_archiveCache.keys().next().value);
  }
  return e;
}

function _resetCacheEntry(e) {
  e.ids.length = 0; e.vecs.length = 0; e.maxId = 0; e.rowsSeen = 0;
}

// Drop a conversation's cached vectors (conversation deleted).
function forgetArchive(conversationId) { _archiveCache.delete(conversationId); }

// Rows newer than `sinceId`. NOT filtered on `embedding IS NOT NULL`: the count
// returned has to be comparable with COUNT(*) below, so nulls are skipped here in
// JS instead of in SQL.
async function _fetchArchiveRows(db, conversationId, sinceId) {
  return (await db.execute({
    sql: `SELECT id, embedding FROM archive
          WHERE conversation_id = ? AND id > ?
          ORDER BY id ASC`,
    args: [conversationId, sinceId],
  })).rows;
}

function _absorbRows(e, rows) {
  for (const r of rows) {
    const vec = decodeEmbedding(r.embedding);
    if (!vec || vec.length !== EMBED_DIM) continue; // no vector -> not retrievable
    e.ids.push(Number(r.id));
    e.vecs.push(normalizeInPlace(vec));
  }
}

async function loadArchiveVectors(db, conversationId) {
  const e = _touchCache(conversationId);

  // One indexed aggregate replaces the old hasArchive() existence probe AND
  // tells us whether anything changed since last turn.
  const agg = (await db.execute({
    sql: 'SELECT COUNT(*) AS n, MAX(id) AS m FROM archive WHERE conversation_id = ?',
    args: [conversationId],
  })).rows[0] || {};
  const total = Number(agg.n || 0);
  const maxId = Number(agg.m || 0);

  if (total === 0) { _resetCacheEntry(e); return e; }
  // Rows vanished, or the table was restored to an older state: the high-water
  // mark means nothing now.
  if (total < e.rowsSeen || maxId < e.maxId) _resetCacheEntry(e);

  // maxId moving with the count unchanged means a row was replaced, not appended
  // -- rare, but the fetch below then returns more rows than expected and forces
  // the rebuild, which is exactly right.
  const expectedNew = total - e.rowsSeen;
  if (expectedNew > 0 || maxId > e.maxId) {
    let rows = await _fetchArchiveRows(db, conversationId, e.maxId);
    // A sync pull can land rows with ids BELOW our high-water mark -- another
    // device's autoincrement is independent of ours. Those are invisible to the
    // id > maxId fetch, and the only way to notice is that fewer rows came back
    // than the count says appeared. Then, and only then, rebuild from scratch.
    if (rows.length !== expectedNew) {
      _resetCacheEntry(e);
      rows = await _fetchArchiveRows(db, conversationId, 0);
    }
    _absorbRows(e, rows);
    e.maxId = maxId;
  }
  e.rowsSeen = total;
  return e;
}

/**
 * Up to k archived turns from THIS conversation most relevant to queryText,
 * above the score threshold, in chronological order.
 *
 * The @tursodatabase/sync engine has no native vector search, so ranking happens
 * in JS. Three things keep that cheap:
 *   - vectors are cached decoded and normalized (see above), so the scan is one
 *     multiply-add loop per row and nothing else;
 *   - the scan reads NO prose -- only the k winners' text is fetched, so a
 *     2,000-turn archive costs four row reads per turn instead of 2,000;
 *   - the archive is checked BEFORE the embed, so a conversation that has not
 *     archived anything yet never pays the 20-80 ms CPU embed at all.
 */
async function retrieve(conversationId, queryText, k = RETRIEVE_K) {
  if (k <= 0 || typeof queryText !== 'string' || !queryText.trim()) return [];

  const db = await getDb();
  const cache = await loadArchiveVectors(db, conversationId);
  if (!cache.ids.length) return [];

  const q = normalizeInPlace(Float32Array.from(await embed(queryText)));

  const scored = [];
  for (let i = 0; i < cache.ids.length; i++) {
    const score = dot(q, cache.vecs[i]);
    if (score >= RETRIEVE_MIN_SCORE) scored.push({ id: cache.ids[i], score });
  }
  if (!scored.length) return [];

  const top = scored
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .sort((a, b) => a.id - b.id); // chronological

  const rows = (await db.execute({
    sql: `SELECT id, role, content FROM archive
          WHERE conversation_id = ? AND id IN (${top.map(() => '?').join(',')})`,
    args: [conversationId, ...top.map((t) => t.id)],
  })).rows;
  const byId = new Map(rows.map((r) => [Number(r.id), r]));

  return top
    .map((t) => {
      const r = byId.get(t.id);
      return r ? { role: r.role, content: r.content, score: t.score } : null;
    })
    .filter(Boolean);
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

// The recall block moved from the middle of the prompt to the tail (see
// buildContext). Sitting after the verbatim window it is the last thing the model
// reads before the new user turn, so it needs to say plainly that it is NOT the
// current scene — otherwise the model continues from a retrieved fragment.
const RETRIEVAL_HEADER =
  '[Recall — excerpts from EARLIER in this same story, pulled up for continuity. ' +
  'They already happened and are out of order. Do not treat them as the present ' +
  'moment and do not continue from them:]';
const RETRIEVAL_FOOTER = '[End of recall. The present moment is the conversation above.]';
const SUMMARY_HEADER = '[Story so far — summary of earlier events:]';

// Content must be non-empty: an empty/whitespace user message would otherwise
// be recorded as a junk turn (the UI blocks empty sends, but the API is open).
function isValidMsg(m) {
  return m && typeof m === 'object' &&
    (m.role === 'system' || m.role === 'user' || m.role === 'assistant') &&
    typeof m.content === 'string' && m.content.trim().length > 0;
}

/**
 * Build the messages array the chat model receives.
 *
 * ORDER IS THE OPTIMIZATION. Ollama reuses its KV cache only up to the first
 * token that differs from the previous request, so anything volatile poisons
 * everything after it. The old order put the recall block in the middle:
 *
 *   persona | summary | RECALL | verbatim turns | new user
 *                        ^ different every turn
 *
 * which re-prefilled the entire verbatim window (~2-4K tokens) on every single
 * message. The volatile blocks now sit at the tail instead:
 *
 *   persona | summary | verbatim turns | RECALL | director | new user
 *   ------------ append-only ---------/  ---- volatile ----/
 *
 * The prefix only ever grows, except when summarization fires and rewrites the
 * summary — once every SUMMARIZE_THRESHOLD-VERBATIM_TURNS turns. metrics.js
 * reports the shared-prefix fraction so this is measured, not assumed.
 *
 * Putting recall last also helps on quality: the model weights recent context
 * more heavily, and recall is the part we most want it to actually use.
 *
 * `leadingSystems` are stable per-conversation system messages (character
 * persona). `trailingSystems` are per-request ones (the director note) and
 * belong in the volatile tail.
 */
async function buildContext(conversationId, incoming, leadingSystems = [], trailingSystems = []) {
  const valid = Array.isArray(incoming) ? incoming.filter(isValidMsg) : [];
  const latestUser = [...valid].reverse().find((m) => m.role === 'user') || null;
  const queryText = latestUser ? latestUser.content : '';

  // Independent reads; the retrieve() leg may include a CPU embed round trip.
  const t0 = Date.now();
  const [summary, verbatim, retrieved] = await Promise.all([
    getSummary(conversationId),
    getVerbatim(conversationId),
    queryText
      ? retrieve(conversationId, queryText).catch(() => [])
      : Promise.resolve([]),
  ]);
  const retrieveMs = Date.now() - t0;

  const messages = [];
  const stable = [];
  for (const m of leadingSystems) if (isValidMsg(m)) stable.push(m);
  if (summary) stable.push({ role: 'system', content: `${SUMMARY_HEADER}\n${summary}` });
  for (const t of verbatim) stable.push({ role: t.role, content: t.content });
  messages.push(...stable);

  if (retrieved.length) {
    messages.push({
      role: 'system',
      content: `${RETRIEVAL_HEADER}\n${renderTurns(retrieved)}\n${RETRIEVAL_FOOTER}`,
    });
  }
  for (const m of trailingSystems) if (isValidMsg(m)) messages.push(m);

  if (latestUser && !alreadyLast(verbatim, latestUser)) {
    messages.push({ role: 'user', content: latestUser.content });
  }

  const chars = (arr) => arr.reduce((n, m) => n + m.content.length, 0);
  const personaChars = chars(leadingSystems.filter(isValidMsg));
  const stats = {
    messageCount: messages.length,
    promptChars: chars(messages),
    // What the append-only prefix costs. Compare against prefillReuse in
    // metrics.js: the two should track once the reorder is doing its job.
    stablePrefixChars: chars(stable),
    personaChars,
    summaryChars: summary ? summary.length : 0,
    verbatimCount: verbatim.length,
    verbatimChars: chars(verbatim),
    retrievedCount: retrieved.length,
    retrievedChars: chars(retrieved),
    directorChars: chars(trailingSystems.filter(isValidMsg)),
    newUserChars: latestUser ? latestUser.content.length : 0,
    retrieveMs,
  };

  return { messages, retrieved, latestUser, stats };
}

function alreadyLast(verbatim, msg) {
  const last = verbatim[verbatim.length - 1];
  return last && last.role === msg.role && last.content === msg.content;
}

// ---- Variants (swipe between alternate generations) ---------------------

// The last assistant turn row, or null.
async function getLastAssistantTurn(conversationId) {
  const db = await getDb();
  const res = await db.execute({
    sql: 'SELECT id, role, content FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT 1',
    args: [conversationId],
  });
  const row = res.rows[0];
  return row && row.role === 'assistant' ? { id: Number(row.id), content: row.content } : null;
}

// Ensure a turn has at least its current content registered as a variant.
// (Older turns created before variants existed get back-filled lazily.)
async function ensureBaseVariant(turnId, content) {
  const db = await getDb();
  const have = await db.execute({ sql: 'SELECT COUNT(*) AS n FROM variants WHERE turn_id = ?', args: [turnId] });
  if (Number(have.rows[0].n) === 0) {
    await db.execute({
      sql: 'INSERT INTO variants (turn_id, content, is_active, created_at) VALUES (?, ?, 1, ?)',
      args: [turnId, content, nowIso()],
    });
  }
}

// Append a new variant to an assistant turn, make it active, and mirror it into
// turns.content (so memory/summary read the chosen text). Returns variant info.
async function appendVariant(turnId, content) {
  const db = await getDb();
  await ensureBaseVariant(turnId, content); // no-op if base already exists
  await db.execute({
    sql: 'INSERT INTO variants (turn_id, content, is_active, created_at) VALUES (?, ?, 0, ?)',
    args: [turnId, content, nowIso()],
  });
  // Activate the just-inserted variant (highest id for this turn).
  const last = await db.execute({
    sql: 'SELECT id FROM variants WHERE turn_id = ? ORDER BY id DESC LIMIT 1', args: [turnId],
  });
  const variantId = Number(last.rows[0].id);
  await db.execute({ sql: 'UPDATE variants SET is_active = 0 WHERE turn_id = ?', args: [turnId] });
  await db.execute({ sql: 'UPDATE variants SET is_active = 1 WHERE id = ?', args: [variantId] });
  await db.execute({ sql: 'UPDATE turns SET content = ? WHERE id = ?', args: [content, turnId] });
  return variantId;
}

// Pick which variant of a turn is active. Mirrors its text into turns.content.
async function setActiveVariant(conversationId, turnId, variantId) {
  const db = await getDb();
  // Validate the variant belongs to a turn in this conversation.
  const v = await db.execute({
    sql: `SELECT v.content FROM variants v JOIN turns t ON t.id = v.turn_id
          WHERE v.id = ? AND v.turn_id = ? AND t.conversation_id = ?`,
    args: [variantId, turnId, conversationId],
  });
  if (!v.rows.length) throw new Error('Variant not found for this conversation/turn.');
  await db.execute({ sql: 'UPDATE variants SET is_active = 0 WHERE turn_id = ?', args: [turnId] });
  await db.execute({ sql: 'UPDATE variants SET is_active = 1 WHERE id = ?', args: [variantId] });
  await db.execute({ sql: 'UPDATE turns SET content = ? WHERE id = ?', args: [v.rows[0].content, turnId] });
  return { turnId, variantId, content: v.rows[0].content };
}

// All variants for a turn, oldest first, with the active index.
async function getVariants(turnId) {
  const db = await getDb();
  const res = await db.execute({
    sql: 'SELECT id, content, is_active FROM variants WHERE turn_id = ? ORDER BY id ASC',
    args: [turnId],
  });
  const list = res.rows.map((r) => ({ id: Number(r.id), content: r.content, active: Boolean(r.is_active) }));
  const activeIndex = Math.max(0, list.findIndex((v) => v.active));
  return { variants: list, activeIndex };
}

// ---- Post-turn bookkeeping ----------------------------------------------

// Embed a batch of turns with bounded concurrency. Returns blobs positionally;
// a failed embed yields null, which stores the turn WITHOUT a vector — the text
// survives, it just cannot be retrieved.
async function embedAll(turns) {
  const out = new Array(turns.length).fill(null);
  let next = 0;
  const worker = async () => {
    for (;;) {
      const i = next++;
      if (i >= turns.length) return;
      try {
        out[i] = encodeEmbedding(await embed(`${turns[i].role}: ${turns[i].content}`));
      } catch { out[i] = null; }
    }
  };
  const lanes = Math.max(1, Math.min(EMBED_CONCURRENCY, turns.length));
  await Promise.all(Array.from({ length: lanes }, worker));
  return out;
}

// SQLite caps bound parameters per statement (999 by default). Batched writes
// below stay well under it by chunking.
function chunk(arr, size) {
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}
const ARCHIVE_INSERT_CHUNK = 100; // x5 params = 500 bound values per statement
const DELETE_ID_CHUNK = 400;

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
  // Register this reply as the turn's first (active) variant so swiping works.
  const newTurn = await db.execute({
    sql: 'SELECT id FROM turns WHERE conversation_id = ? AND role = ? ORDER BY id DESC LIMIT 1',
    args: [conversationId, 'assistant'],
  });
  await db.execute({
    sql: 'INSERT INTO variants (turn_id, content, is_active, created_at) VALUES (?, ?, 1, ?)',
    args: [Number(newTurn.rows[0].id), assistantReply, nowIso()],
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

  // The summarizer call is by far the most expensive thing in this phase — a
  // full generate on SUMMARIZER_MODEL — and the embeds do not depend on it.
  // Running both legs at once hides the ENTIRE embedding phase underneath the
  // summarize call. This runs while the per-conversation lock is held, so every
  // millisecond saved here is a millisecond the user's next message is not
  // blocked behind bookkeeping for the previous one.
  const [summaryResult, embeddings] = await Promise.all([
    summarize(priorSummary, oldest, assistantName).then(
      (text) => ({ ok: true, text }),
      (err) => ({ ok: false, error: err.message }),
    ),
    embedAll(oldest),
  ]);

  if (!summaryResult.ok) {
    // Summarizer down: keep turns verbatim rather than lose them. The embeds we
    // computed alongside are discarded; they will be recomputed next attempt.
    return { archived: 0, summarized: false, error: summaryResult.error };
  }
  let updated = summaryResult.text;
  if (updated.length > MAX_SUMMARY_CHARS) updated = updated.slice(updated.length - MAX_SUMMARY_CHARS);

  await db.execute({
    sql: 'UPDATE conversations SET summary = ?, updated_at = ? WHERE id = ?',
    args: [updated, nowIso(), conversationId],
  });

  // Move folded turns from verbatim -> archive. Two batched statements instead
  // of 2N single-row ones. A multi-row INSERT is one statement and therefore
  // already atomic, which matters because the sync shim exposes no transaction
  // API for us to wrap the old loop in.
  const ts = nowIso();
  const rows = oldest.map((turn, i) => ({ turn, blob: embeddings[i] }));

  for (const part of chunk(rows, ARCHIVE_INSERT_CHUNK)) {
    const args = [];
    for (const { turn, blob } of part) args.push(conversationId, turn.role, turn.content, blob, ts);
    await db.execute({
      sql: `INSERT INTO archive (conversation_id, role, content, embedding, created_at)
            VALUES ${part.map(() => '(?, ?, ?, ?, ?)').join(', ')}`,
      args,
    });
  }
  for (const part of chunk(oldest.map((t) => t.id), DELETE_ID_CHUNK)) {
    await db.execute({
      sql: `DELETE FROM turns WHERE id IN (${part.map(() => '?').join(',')})`,
      args: part,
    });
  }

  return { archived: oldest.length, summarized: true };
}

// Record ONLY a user turn (no assistant reply). Used when a stream is stopped
// before any reply text arrives, so the user's message still survives a reload.
async function recordUserTurn(conversationId, userMessage) {
  if (!userMessage || typeof userMessage.content !== 'string' || !userMessage.content.trim()) return false;
  const db = await getDb();
  const verbatim = await getVerbatim(conversationId);
  if (alreadyLast(verbatim, userMessage)) return false;
  await db.execute({
    sql: 'INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)',
    args: [conversationId, 'user', userMessage.content, nowIso()],
  });
  await touchConversation(conversationId);
  return true;
}

// Delete a conversation row only if it is truly empty (no turns, no archive,
// no title/summary). Cleans up the row /chat pre-creates when the request then
// fails (e.g. Ollama down) — safe to call on any conversation.
async function deleteConversationIfEmpty(id) {
  if (!isValidId(id)) return false;
  const db = await getDb();
  forgetArchive(id);
  const res = await db.execute({
    sql: `DELETE FROM conversations WHERE id = ?
          AND title = '' AND summary = ''
          AND NOT EXISTS (SELECT 1 FROM turns t WHERE t.conversation_id = conversations.id)
          AND NOT EXISTS (SELECT 1 FROM archive a WHERE a.conversation_id = conversations.id)`,
    args: [id],
  });
  return res.rowsAffected > 0;
}

// Record a regenerated reply: append it as a new variant of the LAST assistant
// turn (no new turn, no re-archiving). Returns the turn + variant ids, or null
// if there's no assistant turn to regenerate.
async function recordRegeneration(conversationId, assistantReply) {
  const last = await getLastAssistantTurn(conversationId);
  if (!last) return null;
  const variantId = await appendVariant(last.id, assistantReply);
  await touchConversation(conversationId);
  return { turnId: last.id, variantId };
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
  // Verbatim turns WITH ids, so the UI can swipe variants on the last assistant.
  const vres = await db.execute({
    sql: 'SELECT id, role, content FROM turns WHERE conversation_id = ? ORDER BY id ASC',
    args: [id],
  });
  const verbatim = [];
  const lastAssistant = await getLastAssistantTurn(id);
  for (const t of vres.rows) {
    const turn = { turnId: Number(t.id), role: t.role, content: t.content };
    // Attach variants only to the latest assistant turn (the swipeable one).
    if (lastAssistant && Number(t.id) === lastAssistant.id) {
      const v = await getVariants(lastAssistant.id);
      if (v.variants.length > 1) {
        turn.variants = v.variants;
        turn.activeIndex = v.activeIndex;
      }
    }
    verbatim.push(turn);
  }

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

// Returns the new title, or null when the conversation doesn't exist.
async function setTitle(id, title) {
  if (!isValidId(id)) throw new Error('setTitle: invalid id');
  const clean = typeof title === 'string' ? title.slice(0, 200) : '';
  const db = await getDb();
  const res = await db.execute({
    sql: 'UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?',
    args: [clean, nowIso(), id],
  });
  return res.rowsAffected > 0 ? { title: clean } : null;
}

async function deleteConversation(id) {
  if (!isValidId(id)) throw new Error('deleteConversation: invalid id');
  const db = await getDb();
  const res = await db.execute({ sql: 'DELETE FROM conversations WHERE id = ?', args: [id] });
  forgetArchive(id); // cached vectors would outlive the rows they describe
  return res.rowsAffected > 0;
}

module.exports = {
  newId,
  isValidId,
  ensureConversation,
  buildContext,
  recordTurn,
  recordUserTurn,
  recordRegeneration,
  deleteConversationIfEmpty,
  acquireLock,
  getLastAssistantTurn,
  setActiveVariant,
  listConversations,
  getConversation,
  setTitle,
  deleteConversation,
  forgetArchive,
  _config: {
    CHAT_MODEL, CHAT_NUM_CTX,
    SUMMARIZER_MODEL, SUMMARIZER_NUM_CTX, EMBED_MODEL, EMBED_NUM_GPU,
    VERBATIM_TURNS, SUMMARIZE_THRESHOLD, RETRIEVE_K, RETRIEVE_MIN_SCORE,
    ARCHIVE_CACHE_CONVS, EMBED_CONCURRENCY,
  },
};
