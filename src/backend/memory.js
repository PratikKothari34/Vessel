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
const { getDb, encodeEmbedding, decodeEmbeddingInt8, EMBED_DIM } = require('./db');
const inference = require('./inference');

// ---- Config --------------------------------------------------------------
const CHAT_MODEL = process.env.OLLAMA_MODEL || 'vessel';
const SUMMARIZER_MODEL = process.env.SUMMARIZER_MODEL || 'gemma3:4b';
const EMBED_MODEL = process.env.EMBED_MODEL || 'nomic-embed-text';
const EMBED_NUM_GPU = (() => {
  const v = parseInt(process.env.EMBED_NUM_GPU, 10);
  return Number.isFinite(v) ? v : 0;
})();
// 0 = CPU, and CPU is the default for the same reason the embedder's is. The
// chat model owns 6.1 GB of an 8 GB card; a 3 GB summariser alongside it does
// not get evicted under llama-server, it oversubscribes the card and the driver
// pages 2.3 GB of the chat model out to system RAM -- measured, chat decode
// 47.0 -> 20.8 tok/s. On CPU the summariser costs the GPU nothing at all.
const SUMMARIZER_NUM_GPU = (() => {
  const v = parseInt(process.env.SUMMARIZER_NUM_GPU, 10);
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
// Whether the rolling summary is maintained at all.
//
// Off, folding still happens: old turns are still embedded, still moved to the
// archive, and still reachable through retrieve(). What goes away is the
// condensed narrative of everything that fell out of the window -- so the model
// keeps whatever the vector search surfaces for THIS message, and nothing else.
//
// It exists because the summariser is the single most expensive thing in the
// pipeline and the only one whose value is a judgement call. Turning it off is
// how that judgement gets measured rather than assumed.
//
// DEFAULT IS OFF, and that is a measurement, not a guess. A 2,000-exchange A/B
// with everything else identical -- same character, same script, same models,
// folding and retrieval left on in both arms:
//
//   recall        33.8% on vs 39.4% off   (p=0.30 -- no difference)
//   accuracy      44.3% on vs 61.8% off   (p=0.009, when it committed)
//   wrong-fact    41 on vs 7 off          (p<0.0001)
//   summariser    836 min on vs 0 off     (98.4% CPU duty, 667 min of stall)
//
// The summary did not change how much got remembered. It changed how the model
// failed: holding forty facts co-resident in lossy prose let it answer with a
// DIFFERENT established fact, confidently, six times as often. Retrieval alone
// carried the same recall at 2,394 prompt tokens against 4,023.
//
// The defects that run exposed are fixed below (clampSummary, stripTranscript,
// and the prompt). Those fixes are UNMEASURED -- shipping them on by default
// would repeat the mistake the experiment was run to catch. Set
// SUMMARY_ENABLED=1 to re-run the A/B against the fixed summariser.
const SUMMARY_ENABLED = !/^(0|false|off|no)$/i.test(
  String(process.env.SUMMARY_ENABLED ?? '0').trim(),
);
const VERBATIM_TURNS = intEnv('VERBATIM_TURNS', 8);
let SUMMARIZE_THRESHOLD = intEnv('SUMMARIZE_THRESHOLD', 12);
if (SUMMARIZE_THRESHOLD <= VERBATIM_TURNS) SUMMARIZE_THRESHOLD = VERBATIM_TURNS + 4;

// Hard ceiling on how many verbatim rows the READ path will include.
//
// Folding is a background job now (see runMaintenance), so between the turn
// that crosses the threshold and the fold finishing, `turns` legitimately holds
// more rows than the design allows. A CPU summary takes 95-239 s -- long enough
// for dozens of turns to pile up behind it -- and without a ceiling the prompt
// would grow with that backlog and overflow the context window.
//
// SUMMARIZE_THRESHOLD + 2 is exactly the high-water mark the old synchronous
// design already reached: the threshold, plus the user+assistant pair whose
// arrival crossed it. So in the steady state this changes nothing.
const VERBATIM_CEILING = intEnv(
  'VERBATIM_CEILING', SUMMARIZE_THRESHOLD + 2, { min: VERBATIM_TURNS },
);

// Most turns one background fold will summarise at once.
//
// The synchronous design capped this implicitly: it folded on the turn that
// crossed the threshold, so there were never more than a handful of rows to
// fold. Background folding removes that guarantee -- a slow summariser plus a
// fast typist can leave dozens of turns waiting -- and handing all of them to
// one generate would blow SUMMARIZER_NUM_CTX and silently truncate the oldest,
// which is the exact failure the summary exists to prevent.
//
// A fold that hits this cap asks for another pass instead of taking a bigger
// bite. Successive passes fold the remainder, oldest first, each one seeing the
// previous pass's summary as its prior.
const MAX_FOLD_TURNS = intEnv('MAX_FOLD_TURNS', 24, { min: 2 });
const RETRIEVE_K = intEnv('RETRIEVE_K', 4);
const RETRIEVE_MIN_SCORE = floatEnv('RETRIEVE_MIN_SCORE', 0.45, { min: -1, max: 1 });
const MAX_SUMMARY_CHARS = intEnv('MAX_SUMMARY_CHARS', 6000, { min: 500 });
// The prompt never used to state a length, so the summarizer answered with
// whatever it felt like -- measured at 5,792 to 12,422 chars against a 6,000
// cap. The overflow is cut by the front-truncation below, which removes the
// OLDEST text: the part already condensed several times over and no longer
// present in the verbatim window, so it is the only part that cannot be
// recovered. Asking for a bound fixes both ends -- gemma3:4b came back at 5,160
// chars keeping 12 of 13 facts, versus 12,422 chars with 6,422 amputated.
//
// The target is deliberately well under the cap: models treat a character count
// as a loose hint and overshoot roughly twofold. Truncation stays as the
// backstop for when they ignore it entirely.
const SUMMARY_TARGET_CHARS = intEnv(
  'SUMMARY_TARGET_CHARS', Math.round(MAX_SUMMARY_CHARS * 0.4), { min: 200 },
);
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

// Recent verbatim turns in chronological order, capped at VERBATIM_CEILING.
//
// Ordering DESC + reversing takes the NEWEST rows, which is the right end to
// keep: anything past the ceiling is a backlog the background fold is already
// working on, and the archive keeps every word of it either way.
async function getVerbatim(id, limit = VERBATIM_CEILING) {
  const db = await getDb();
  const res = await db.execute({
    sql: 'SELECT role, content FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT ?',
    args: [id, limit],
  });
  return [...res.rows].reverse().map((r) => ({ role: r.role, content: r.content }));
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
  return inference.generate(model, prompt, {
    numCtx: SUMMARIZER_NUM_CTX,
    numGpu: SUMMARIZER_NUM_GPU,
  });
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
  // Keep a reference to the CHAIN we store, not to `next`. The tail-identity
  // check below has to compare against what is actually in the map: comparing
  // against `next` never matched, so no entry was ever removed and _locks grew
  // by one dead promise per conversation for the life of the process.
  const chain = prev.then(() => next);
  _locks.set(id, chain);

  let timer;
  const waited = new Promise((r) => { timer = setTimeout(r, waitMs); });
  await Promise.race([prev.catch(() => {}), waited]);
  clearTimeout(timer);

  let released = false;
  return () => {
    if (released) return;
    released = true;
    release();
    // Only the LAST waiter may drop the key; anyone queued behind us has
    // already replaced the tail with their own chain.
    if (_locks.get(id) === chain) _locks.delete(id);
  };
}

// ---- Retrieval -----------------------------------------------------------

// Scale a vector to unit length IN PLACE. Every vector that enters the cache and
// every query vector is normalized on the way in, which turns cosine similarity
// into a plain dot product: the two sqrt() calls and the two norm accumulator
// loops per candidate row disappear from the hot scan entirely.
// null means the vector has no usable direction and the caller must drop it:
// either it is all zero, or a component is not finite. The second case is the
// one that matters. A blob can carry an Infinity -- a legacy f32 row is raw
// bytes, and a sync pull brings rows this device never wrote -- and an infinite
// component makes the norm Infinity, the scaled components NaN, and every score
// against that row NaN. NaN loses no comparison, so such a row would rank above
// real matches AND, once it became the cutoff, let every remaining row through.
// One bad blob would quietly replace retrieval with noise for the whole
// conversation, so it is rejected here instead.
function normalizeInPlace(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  if (n === 0 || !Number.isFinite(n)) return null;
  const inv = 1 / Math.sqrt(n);
  for (let i = 0; i < v.length; i++) v[i] *= inv;
  return v;
}

// ---- Archive vector cache ------------------------------------------------
// Retrieval used to re-read and re-decode every archived embedding on EVERY
// turn: 3 KB per row off disk, a Float32 decode per row, then two sqrt loops per
// row. None of that changes between turns -- the archive is append-only.
//
// So keep the vectors resident per conversation and load only rows newer than
// the high-water mark. Steady state is zero row reads and zero decodes; a turn
// that archives adds a handful.
//
// They are held as ONE flat int8 matrix, exactly as they sit on disk, plus a
// per-row inverse norm -- not as a vector of decoded, normalized Float32Arrays.
// Cosine is invariant to positive scaling, so scoring raw int8 against its own
// norm gives the same number dequantizing would (see decodeEmbeddingInt8), and
// the layout pays off twice over on a long story. At 40,000 archived turns,
// measured: cold fill 130 ms -> 43 ms, because the dequantize pass and 40,000
// Float32Array allocations both disappear; resident 123 MB -> 31 MB, which at
// ARCHIVE_CACHE_CONVS=8 is the difference between a gigabyte of cache and a
// quarter of one. The scan itself does NOT get faster -- measured at ~35 ms
// either way, and it is 1% of a turn -- so the win claimed here is memory and
// load time, not scan time.
//
// Correctness against cloud sync: a pull can insert archive rows with ids BELOW
// our high-water mark (another device's autoincrement), or a restore can shrink
// the table. Neither is caught by a max(id) check alone, so we carry the row
// count too and rebuild from scratch whenever it drops.
//
// LRU by insertion order: delete-then-set makes the first key the oldest.
const _archiveCache = new Map(); // convId -> cache entry, see _newCacheEntry

const ARCHIVE_CACHE_MIN_CAP = 256;

function _newCacheEntry() {
  return {
    ids: new Float64Array(ARCHIVE_CACHE_MIN_CAP),     // archive rowid per row
    mat: new Int8Array(ARCHIVE_CACHE_MIN_CAP * EMBED_DIM),
    norms: new Float32Array(ARCHIVE_CACHE_MIN_CAP),   // 1 / |row|, precomputed
    n: 0,                                             // rows held
    cap: ARCHIVE_CACHE_MIN_CAP,
    maxId: 0,
    rowsSeen: 0,
  };
}

// Grow to hold `want` rows. Geometric, so filling an archive of any size costs
// O(log n) copies rather than one per row.
function _ensureCapacity(e, want) {
  if (want <= e.cap) return;
  let cap = e.cap;
  while (cap < want) cap *= 2;
  const ids = new Float64Array(cap);
  const mat = new Int8Array(cap * EMBED_DIM);
  const norms = new Float32Array(cap);
  ids.set(e.ids.subarray(0, e.n));
  mat.set(e.mat.subarray(0, e.n * EMBED_DIM));
  norms.set(e.norms.subarray(0, e.n));
  e.ids = ids; e.mat = mat; e.norms = norms; e.cap = cap;
}

function _touchCache(conversationId) {
  let e = _archiveCache.get(conversationId);
  if (e) _archiveCache.delete(conversationId);
  else e = _newCacheEntry();
  _archiveCache.set(conversationId, e);
  while (_archiveCache.size > ARCHIVE_CACHE_CONVS) {
    _archiveCache.delete(_archiveCache.keys().next().value);
  }
  return e;
}

// Rewind to empty. The buffers are kept: a reset is followed by a refill of the
// same conversation, so the capacity earned is exactly the capacity wanted.
function _resetCacheEntry(e) {
  e.n = 0; e.maxId = 0; e.rowsSeen = 0;
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
  _ensureCapacity(e, e.n + rows.length);
  for (const r of rows) {
    const off = e.n * EMBED_DIM;
    // Writes into the row slot without committing it. A rejected row leaves
    // whatever it managed to write behind, which the next row overwrites,
    // because e.n only moves on success.
    if (!decodeEmbeddingInt8(r.embedding, e.mat, off)) continue;
    let sq = 0;
    for (let i = 0; i < EMBED_DIM; i++) { const x = e.mat[off + i]; sq += x * x; }
    if (sq === 0) continue; // no direction -> matches nothing, and 1/0 is not a score
    e.norms[e.n] = 1 / Math.sqrt(sq);
    e.ids[e.n] = Number(r.id);
    e.n++;
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
  if (!cache.n) return [];

  // A query with no direction matches nothing, and scoring against it would
  // return NaN for every row.
  const q = normalizeInPlace(Float32Array.from(await embed(queryText)));
  if (!q) return [];

  // Bounded selection, not a sort. The old path allocated one { id, score }
  // object per row above the threshold and then sorted all of them to keep k --
  // O(n log n) comparisons plus n allocations, on a list that grows with the
  // story. Here each row costs one compare against the current k-th best, and
  // only a winner pays an O(k) insert. k is 4, so that insert is free.
  const topId = new Float64Array(k);
  const topScore = new Float64Array(k);
  let filled = 0;
  let cutoff = RETRIEVE_MIN_SCORE;
  const { ids, mat, norms, n } = cache;
  for (let i = 0, off = 0; i < n; i++, off += EMBED_DIM) {
    // Raw int8 against the unit query, then scaled by the row's own inverse
    // norm: that product IS the cosine, because the row's dequantized form is
    // a positive multiple of these components.
    let d = 0;
    for (let j = 0; j < EMBED_DIM; j++) d += q[j] * mat[off + j];
    const score = d * norms[i];
    // Negated rather than `score < cutoff`: NaN fails every comparison, so this
    // form drops it where the direct one would let it through. Nothing reaching
    // here can be NaN any more -- an int8 row cannot hold one and the norm is
    // finite and positive by construction -- but the guard is free, and it is
    // the last line of defence if a future format lets one back in.
    if (!(score >= cutoff)) continue;
    // Descending insert; drops the weakest once full.
    let j = filled < k ? filled++ : k - 1;
    while (j > 0 && topScore[j - 1] < score) {
      topScore[j] = topScore[j - 1];
      topId[j] = topId[j - 1];
      j--;
    }
    topScore[j] = score;
    topId[j] = ids[i];
    // Once k winners are held, nothing weaker than the weakest can ever win.
    if (filled === k) cutoff = topScore[k - 1];
  }
  if (!filled) return [];

  const top = Array.from({ length: filled }, (_, i) => ({ id: topId[i], score: topScore[i] }))
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

// How far into a truncated summary we will look for a clean break.
//
// Front-truncation cuts at a character offset, so the survivor starts mid-word
// by construction -- the 2,000-exchange run ended with a summary whose first
// characters were "hreads". That fragment is not just ugly: it becomes the next
// fold's CURRENT SUMMARY, so every later fold inherits and re-compresses a
// broken opening, and the damage compounds for the life of the conversation.
//
// The lookahead is capped rather than unbounded. Prose that carries no sentence
// break in its first fifth is not prose, and discarding more than that to find
// one would cost more than the fragment does.
const SUMMARY_CUT_LOOKAHEAD = 0.2;

/**
 * Trim a summary to `max` characters from the FRONT, landing on a boundary.
 *
 * Front rather than back because the oldest material is the part already
 * condensed several times over and no longer present in the verbatim window --
 * the only part that cannot be recovered from anywhere else. Preference order
 * is paragraph break, then sentence end, then word break, then the raw cut:
 * each step down is a worse opening, and the raw cut is what we had before.
 */
function clampSummary(text, max = MAX_SUMMARY_CHARS) {
  const s = String(text ?? '');
  if (s.length <= max) return s;
  const cut = s.slice(s.length - max);
  const window = Math.floor(max * SUMMARY_CUT_LOOKAHEAD);
  // The cut may already have landed between words. Advancing to the next
  // boundary would then discard a whole word to fix nothing -- the defect is
  // starting mid-WORD, and this start is not.
  const startsClean = /\s/.test(s.charAt(s.length - max - 1) || ' ');

  // A blank line is the only boundary guaranteed not to be mid-thought.
  const para = cut.indexOf('\n\n');
  if (para >= 0 && para < window) {
    const out = cut.slice(para + 2).trimStart();
    if (out) return out;
  }

  // Sentence end, including one closing quote or bracket after the stop.
  const m = /[.!?]["')\]]?\s+/.exec(cut.slice(0, window + 4));
  if (m) {
    const out = cut.slice(m.index + m[0].length);
    if (out) return out;
  }

  if (startsClean) return cut;

  // Backstop: at least do not start mid-word.
  const word = cut.search(/\s/);
  if (word >= 0 && word < window) {
    const out = cut.slice(word + 1).trimStart();
    if (out) return out;
  }
  return cut;
}

// A speaker label at the head of a line: `User:` or the character's own name.
//
// Deliberately NOT a general `^\w+:` — real summary prose opens lines with
// "Note:", "Kestrel Station: a wreck", and any other colon it likes. Only the
// two labels renderTurns actually emits are stripped, so the rule can only ever
// remove text the summariser copied out of its own input.
function transcriptLineRe(assistantName) {
  const esc = (x) => String(x).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const who = ['User', assistantName && String(assistantName).trim()]
    .filter(Boolean).map(esc).join('|');
  return new RegExp(`^\\s*(?:${who})\\s*:`, 'i');
}

/**
 * Drop transcript lines the summariser copied instead of summarising.
 *
 * Measured failure: across 226 folds the summariser degenerated from narrative
 * into concatenation, and the final 6,000-char summary ended in raw dialogue --
 * "User: Do you dream about the Petrel?" / "Dr. Ilse Varga-Mbeki: ...". A
 * transcript inside the summary is strictly worse than no summary: it burns the
 * character budget at full length, and it re-teaches the model the label format
 * the persona spends a paragraph forbidding.
 *
 * Returns the cleaned text and how many lines went, so the caller can tell a
 * light touch-up from a summary that was nothing but transcript.
 */
function stripTranscript(text, assistantName) {
  const s = String(text ?? '');
  if (!s) return { text: s, stripped: 0 };
  const re = transcriptLineRe(assistantName);
  const lines = s.split('\n');
  const kept = [];
  let stripped = 0;
  for (const line of lines) {
    if (re.test(line)) { stripped += 1; continue; }
    kept.push(line);
  }
  if (!stripped) return { text: s, stripped: 0 };
  // Collapse the blank runs the removed lines leave behind.
  const out = kept.join('\n').replace(/\n{3,}/g, '\n\n').trim();
  return { text: out, stripped };
}

// Least a cleaned summary may be, relative to the one it would replace, before
// we keep the old one instead.
//
// If stripping took most of the reply, the summariser did not write a summary
// with some transcript in it -- it wrote a transcript. Overwriting a good prior
// summary with the handful of prose lines that survived loses more than the
// fold gains, and the same non-destructive reasoning already governs the
// SUMMARY_ENABLED=0 path: never replace something real with a stub.
const SUMMARY_MIN_KEEP_RATIO = 0.35;

function summaryIsUsable(cleaned, priorSummary) {
  if (!cleaned) return false;
  const prior = String(priorSummary || '');
  if (prior.length < 400) return true; // nothing worth protecting yet
  return cleaned.length >= prior.length * SUMMARY_MIN_KEEP_RATIO;
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
    // Measured over 226 folds: the summariser drifts out of narrative into
    // copying the exchanges back verbatim, labels and all, which spends the
    // whole character budget on dialogue it was asked to compress.
    'Write third-person narrative prose in continuous paragraphs. Do NOT write',
    'a transcript: never begin a line with a speaker label such as "User:" or',
    'the character name followed by a colon, and do not reproduce dialogue word',
    'for word. Report what was said and decided, do not replay it.',
    '',
    // The same run invented a name for the user out of nothing, wrote it into
    // the summary, and thereafter fed it back on every single turn -- the model
    // had no way to tell an invented name from an established one.
    'Use ONLY names the story has actually established. Never invent a name for',
    'anyone, and never promote a person who was merely mentioned into a',
    'participant in the scene. If the user has not named their character, call',
    'them "the user" and nothing else.',
    `Keep the updated summary under ${SUMMARY_TARGET_CHARS} characters. If it `
      + 'would run longer, compress the OLDEST material hardest and keep the '
      + 'newest exchanges intact.',
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
  // Walk backwards for the newest user message instead of filter + copy +
  // reverse + find: three intermediate arrays over the whole incoming list, to
  // locate one element that is almost always the last one.
  let latestUser = null;
  if (Array.isArray(incoming)) {
    for (let i = incoming.length - 1; i >= 0; i--) {
      const m = incoming[i];
      if (m && m.role === 'user' && isValidMsg(m)) { latestUser = m; break; }
    }
  }
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
  // Counted while building instead of re-walked afterwards: leadingSystems was
  // filtered twice (once here, once for the stats) and every block was measured
  // by a separate reduce over an array we had just constructed.
  let personaChars = 0;
  for (const m of leadingSystems) {
    if (!isValidMsg(m)) continue;
    stable.push(m);
    personaChars += m.content.length;
  }
  if (summary) stable.push({ role: 'system', content: `${SUMMARY_HEADER}\n${summary}` });
  for (const t of verbatim) stable.push({ role: t.role, content: t.content });
  // push(...stable) spreads the whole window onto the call stack; a long
  // verbatim window plus a long summary is enough arguments to matter.
  for (const m of stable) messages.push(m);

  if (retrieved.length) {
    messages.push({
      role: 'system',
      content: `${RETRIEVAL_HEADER}\n${renderTurns(retrieved)}\n${RETRIEVAL_FOOTER}`,
    });
  }
  let directorChars = 0;
  for (const m of trailingSystems) {
    if (!isValidMsg(m)) continue;
    messages.push(m);
    directorChars += m.content.length;
  }

  if (latestUser && !alreadyLast(verbatim, latestUser)) {
    messages.push({ role: 'user', content: latestUser.content });
  }

  const chars = (arr) => arr.reduce((n, m) => n + m.content.length, 0);
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
    directorChars,
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

// Back-fill the variant a turn should already have, seeded from its OWN text.
//
// recordTurn writes a base variant with every assistant reply, so the normal
// path never needs this. Two kinds of row do: turns written before the variants
// table existed, and turns pulled from an older synced database -- runMigrations
// adds columns, it never back-fills rows.
//
// The seed has to come from turns.content, not from the caller. Seeding it with
// the caller's NEW text made the first regenerate on such a turn overwrite the
// original reply with the regeneration and leave two identical variants, so the
// swipe the feature exists for had nothing to swipe back to. Silent, and
// unrecoverable once turns.content was mirrored over too.
async function ensureBaseVariant(turnId) {
  const db = await getDb();
  const have = await db.execute({ sql: 'SELECT COUNT(*) AS n FROM variants WHERE turn_id = ?', args: [turnId] });
  if (Number(have.rows[0].n) > 0) return;
  const turn = await db.execute({ sql: 'SELECT content FROM turns WHERE id = ?', args: [turnId] });
  if (!turn.rows.length) return;
  await db.execute({
    sql: 'INSERT INTO variants (turn_id, content, is_active, created_at) VALUES (?, ?, 1, ?)',
    args: [turnId, turn.rows[0].content, nowIso()],
  });
}

// Append a new variant to an assistant turn, make it active, and mirror it into
// turns.content (so memory/summary read the chosen text). Returns variant info.
async function appendVariant(turnId, content) {
  const db = await getDb();
  await ensureBaseVariant(turnId); // no-op if the base already exists
  // Clear the siblings BEFORE inserting, so the new row can land active and the
  // three writes this took are two. The driver reports the inserted rowid, so
  // the id is the row we just wrote rather than whatever a follow-up
  // ORDER BY id DESC found -- same reasoning as recordTurn.
  await db.execute({ sql: 'UPDATE variants SET is_active = 0 WHERE turn_id = ?', args: [turnId] });
  const inserted = await db.execute({
    sql: 'INSERT INTO variants (turn_id, content, is_active, created_at) VALUES (?, ?, 1, ?)',
    args: [turnId, content, nowIso()],
  });
  let variantId = Number(inserted.lastInsertRowid);
  if (!Number.isInteger(variantId) || variantId <= 0) {
    // Defensive: a driver that does not report lastInsertRowid falls back to the
    // old lookup rather than returning an id the UI cannot address.
    const last = await db.execute({
      sql: 'SELECT id FROM variants WHERE turn_id = ? ORDER BY id DESC LIMIT 1', args: [turnId],
    });
    variantId = Number(last.rows[0].id);
  }
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
  // One timestamp for the whole exchange: the user turn, the reply and its base
  // variant belong to the same moment, and three separate new Date() calls could
  // otherwise straddle a second boundary and order the rows by clock skew.
  const ts = nowIso();

  if (userMessage && !alreadyLast(verbatim, userMessage)) {
    await db.execute({
      sql: 'INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)',
      args: [conversationId, 'user', userMessage.content, ts],
    });
  }
  // The driver reports the inserted rowid, so the variant row is tied to the
  // turn we just wrote rather than to whatever "newest assistant turn" happened
  // to be by the time a follow-up SELECT ran. One round trip fewer, and no
  // window for the two to disagree.
  const inserted = await db.execute({
    sql: 'INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)',
    args: [conversationId, 'assistant', assistantReply, ts],
  });
  let turnId = Number(inserted.lastInsertRowid);
  if (!Number.isInteger(turnId) || turnId <= 0) {
    // Defensive: a driver that does not report lastInsertRowid falls back to the
    // old lookup rather than writing an orphaned variant.
    const newTurn = await db.execute({
      sql: 'SELECT id FROM turns WHERE conversation_id = ? AND role = ? ORDER BY id DESC LIMIT 1',
      args: [conversationId, 'assistant'],
    });
    turnId = Number(newTurn.rows[0].id);
  }
  // Register this reply as the turn's first (active) variant so swiping works.
  await db.execute({
    sql: 'INSERT INTO variants (turn_id, content, is_active, created_at) VALUES (?, ?, 1, ?)',
    args: [turnId, assistantReply, ts],
  });
  await touchConversation(conversationId);

  const count = (await db.execute({
    sql: 'SELECT COUNT(*) AS n FROM turns WHERE conversation_id = ?', args: [conversationId],
  })).rows[0].n;

  if (Number(count) <= SUMMARIZE_THRESHOLD) {
    return { archived: 0, summarized: false };
  }

  // Everything above is the fast write phase, and it is done: the turn is
  // durable and the reply is already on screen. What follows -- summarise,
  // embed, archive -- is maintenance, and it is slow: the summariser runs on
  // CPU (see docs/MASTER.md, "Where the summariser runs: CPU") at 95-239 s.
  //
  // The caller holds the per-conversation turn lock across this function, so
  // doing that work here makes the user's NEXT message wait behind it. Measured
  // at 31,839 ms and 35,619 ms for a summarising turn against 2.3-6.1 s for
  // every other one. Hand it to the background instead; the read path is built
  // to tolerate the stale state that creates (see VERBATIM_CEILING).
  scheduleMaintenance(conversationId, assistantName);
  return { archived: 0, summarized: false, maintenanceScheduled: true };
}

// ---- Background maintenance ---------------------------------------------
// Folding old turns into the rolling summary is maintenance, not part of a
// turn. It runs OFF the turn lock, so a summary never delays the next message.
//
// What that costs: for as long as a fold is in flight, readers see the previous
// summary and a longer-than-designed verbatim window. Both are safe. The
// summary is behind, not wrong; the extra verbatim turns are the very text the
// summary is missing, so the model sees that material either way -- in full
// rather than condensed. VERBATIM_CEILING bounds how much of it reaches the
// prompt.
//
// One run per conversation at a time. A request arriving mid-run sets `rerun`
// instead of starting a second one: the next pass re-reads the turn count and
// picks up whatever arrived meanwhile, which is strictly better than two runs
// racing over the same oldest rows.
const _maintenance = new Map(); // conversationId -> { promise, rerun, cancelled }

// What folding actually costs.
//
// It runs OFF the turn lock, so none of it appears in a request timing: from
// the outside a conversation that summarises and one that does not look
// identical until the machine runs out of CPU. Six counters is the whole price
// of being able to see it.
const _foldCost = {
  folds: 0, summaries: 0, archived: 0,
  summarizeMs: 0, embedMs: 0, waitMs: 0,
  // Summary hygiene, so the two known degeneration modes are visible in
  // /health rather than only in a post-hoc read of the stored summary:
  // transcript lines the summariser copied instead of compressing, and folds
  // where so little prose survived that the prior summary was kept instead.
  transcriptLines: 0, summariesRejected: 0,
};
// `pending` is how many conversations are folding right now, which is the
// only way a caller can tell a quiet queue from one that has not started.
function foldStats() { return { ..._foldCost, pending: _maintenance.size }; }

function scheduleMaintenance(conversationId, assistantName) {
  const existing = _maintenance.get(conversationId);
  if (existing) {
    existing.rerun = true;
    return existing.promise;
  }

  const state = { rerun: false, cancelled: false, draining: false, promise: null };
  _maintenance.set(conversationId, state);
  state.promise = (async () => {
    try {
      for (;;) {
        state.rerun = false;
        await runMaintenance(conversationId, assistantName, state);
        // No await between the loop body returning and this check, so a
        // scheduleMaintenance() call cannot slip in and set `rerun` on a state
        // we are about to discard.
        if (state.cancelled || !state.rerun) break;
      }
    } catch (err) {
      // A failed fold is recoverable by construction: nothing was deleted, so
      // the turns are still verbatim and the next turn schedules another try.
      console.error(`[memory] maintenance failed for ${conversationId}:`, err.message);
    } finally {
      if (_maintenance.get(conversationId) === state) _maintenance.delete(conversationId);
    }
  })();
  return state.promise;
}

// Every in-flight fold, so shutdown can flush rather than drop them.
function awaitAllMaintenance() {
  return Promise.all([..._maintenance.values()].map((s) => s.promise.catch(() => {})));
}

// Stop a fold from WRITING. Not an abort -- the expensive legs (summarise +
// embed) are already in flight and interrupting them saves nothing -- but it
// guarantees no rows land for a conversation the user just deleted.
function cancelMaintenance(conversationId) {
  const s = _maintenance.get(conversationId);
  if (s) {
    s.cancelled = true;
    s.rerun = false;
    s.draining = false;
  }
}

/**
 * One fold: summarise + embed + archive the turns above the verbatim window.
 * Re-reads its own state, so it is safe to call at any time and cheap when
 * there is nothing to do.
 */
async function runMaintenance(conversationId, assistantName, state) {
  const db = await getDb();
  const count = (await db.execute({
    sql: 'SELECT COUNT(*) AS n FROM turns WHERE conversation_id = ?', args: [conversationId],
  })).rows[0].n;

  // SUMMARIZE_THRESHOLD is the TRIGGER; VERBATIM_TURNS is the TARGET. Once a
  // drain is under way the trigger has already fired, so later passes keep
  // going down to the target -- otherwise a capped fold would stop at the
  // threshold and leave the window permanently larger than the design.
  const draining = Boolean(state && state.draining);
  const floor = draining ? VERBATIM_TURNS : SUMMARIZE_THRESHOLD;
  if (Number(count) <= floor) {
    if (state) state.draining = false;
    return { archived: 0, summarized: false };
  }

  // Fold the oldest (count - VERBATIM_TURNS) turns into summary + archive,
  // at most MAX_FOLD_TURNS of them in one pass.
  const pending = Number(count) - VERBATIM_TURNS;
  const toArchiveCount = Math.min(pending, MAX_FOLD_TURNS);
  // Ask for another pass rather than a bigger prompt. Set before the long
  // awaits so a cancel during them still wins: cancelMaintenance clears both.
  if (state) {
    state.draining = pending > toArchiveCount;
    if (state.draining) state.rerun = true;
  }
  const oldest = (await db.execute({
    sql: 'SELECT id, role, content FROM turns WHERE conversation_id = ? ORDER BY id ASC LIMIT ?',
    args: [conversationId, toArchiveCount],
  })).rows;

  const priorSummary = await getSummary(conversationId);

  // The summarizer call is by far the most expensive thing in this phase — a
  // full generate on SUMMARIZER_MODEL — and the embeds do not depend on it.
  // Running both legs at once hides the ENTIRE embedding phase underneath the
  // summarize call. Nothing waits on this any more, but the window of stale
  // reads is still worth shortening, and the embedder is on CPU too.
  const legStarted = Date.now();
  let summarizeMs = 0;
  let embedMs = 0;
  const [summaryResult, embeddings] = await Promise.all([
    SUMMARY_ENABLED
      ? summarize(priorSummary, oldest, assistantName).then(
        (text) => { summarizeMs = Date.now() - legStarted; return { ok: true, text }; },
        (err) => { summarizeMs = Date.now() - legStarted; return { ok: false, error: err.message }; },
      )
      // Not an empty summary -- no summary write at all. An empty one would
      // overwrite whatever a previous, enabled run had already built.
      : Promise.resolve({ ok: true, skipped: true }),
    embedAll(oldest).then((out) => { embedMs = Date.now() - legStarted; return out; }),
  ]);
  // Both legs run together, so waitMs is what the pair cost, not their sum.
  _foldCost.summarizeMs += summarizeMs;
  _foldCost.embedMs += embedMs;
  _foldCost.waitMs += Date.now() - legStarted;

  if (!summaryResult.ok) {
    // Summarizer down: keep turns verbatim rather than lose them. The embeds we
    // computed alongside are discarded; they will be recomputed next attempt.
    return { archived: 0, summarized: false, error: summaryResult.error };
  }

  // Those two awaits are the long ones -- minutes, on CPU. The conversation can
  // be deleted in that time, and writing archive rows for a row that no longer
  // exists would resurrect a deleted story. Everything below this point is
  // await-free apart from its own writes, so checking here is enough.
  if (state && state.cancelled) return { archived: 0, summarized: false, cancelled: true };
  const alive = await db.execute({
    sql: 'SELECT 1 AS ok FROM conversations WHERE id = ?', args: [conversationId],
  });
  if (!alive.rows.length) return { archived: 0, summarized: false, cancelled: true };

  let wroteSummary = false;
  if (!summaryResult.skipped) {
    // Clean BEFORE clamping: stripping transcript can bring an over-long reply
    // back under the cap on its own, and clamping first would spend the budget
    // holding dialogue we are about to discard anyway.
    const cleaned = stripTranscript(summaryResult.text, assistantName);
    _foldCost.transcriptLines += cleaned.stripped;
    if (summaryIsUsable(cleaned.text, priorSummary)) {
      const updated = clampSummary(cleaned.text, MAX_SUMMARY_CHARS);
      await db.execute({
        sql: 'UPDATE conversations SET summary = ?, updated_at = ? WHERE id = ?',
        args: [updated, nowIso(), conversationId],
      });
      wroteSummary = true;
    } else {
      // Keep the prior summary. The turns still fold -- they are embedded and
      // archived below, so nothing is lost, it just is not narrated.
      _foldCost.summariesRejected += 1;
    }
  }

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

  _foldCost.folds += 1;
  _foldCost.archived += oldest.length;
  if (wroteSummary) _foldCost.summaries += 1;

  return { archived: oldest.length, summarized: wroteSummary };
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
  cancelMaintenance(id);
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
  // Named columns, not c.*: the only thing the sidebar wants from `summary` is
  // whether there is one, and a folded conversation's summary runs to a couple
  // of thousand characters. Selecting it would serialise all of that across the
  // driver for every row, on every refresh, to produce one boolean. Same reason
  // the preview is truncated in SQL rather than in JS.
  const sql = `SELECT c.id, c.character_id, c.title, c.created_at, c.updated_at,
              LENGTH(COALESCE(c.summary, '')) > 0 AS has_summary,
              (SELECT substr(t.content, 1, 120) FROM turns t
                WHERE t.conversation_id = c.id ORDER BY t.id DESC LIMIT 1) AS last_turn,
              (SELECT COUNT(*) FROM turns t WHERE t.conversation_id = c.id) +
              (SELECT COUNT(*) FROM archive a WHERE a.conversation_id = c.id) AS turn_count
       FROM conversations c${characterId ? ' WHERE c.character_id = ?' : ''}
       ORDER BY c.updated_at DESC`;
  const res = await db.execute(characterId ? { sql, args: [characterId] } : sql);
  return res.rows.map((r) => ({
    id: r.id,
    characterId: r.character_id,
    title: r.title || '',
    createdAt: r.created_at,
    updatedAt: r.updated_at,
    hasSummary: Boolean(Number(r.has_summary)),
    turnCount: Number(r.turn_count || 0),
    preview: r.last_turn || '',
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
  // A background fold may be mid-summary for this conversation. Tell it not to
  // write; it checks the flag (and re-checks that this row still exists) right
  // before its first write.
  cancelMaintenance(id);
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
  awaitAllMaintenance,
  cancelMaintenance,
  foldStats,
  // Exported for tests: both are pure string transforms and the cases that
  // matter (mid-word cut, transcript degeneration) are exactly the ones a
  // 2,000-turn run took 14 hours to surface.
  clampSummary,
  stripTranscript,
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
  // Pure vector helpers, exported for the unit tests only. They sit below the
  // database and the engine, so testing them through retrieve() would need both
  // just to exercise arithmetic.
  _internals: { normalizeInPlace, _absorbRows, _newCacheEntry },
  _config: {
    CHAT_MODEL, CHAT_NUM_CTX,
    SUMMARIZER_MODEL, SUMMARIZER_NUM_CTX, SUMMARIZER_NUM_GPU, EMBED_MODEL, EMBED_NUM_GPU,
    SUMMARY_ENABLED, MAX_SUMMARY_CHARS, SUMMARY_TARGET_CHARS,
    VERBATIM_TURNS, VERBATIM_CEILING, MAX_FOLD_TURNS, SUMMARIZE_THRESHOLD, RETRIEVE_K, RETRIEVE_MIN_SCORE,
    ARCHIVE_CACHE_CONVS, EMBED_CONCURRENCY,
  },
};
