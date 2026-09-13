'use strict';

/**
 * metrics.js — generation telemetry (Stage 0 of docs/decisions/0001).
 *
 * Every Ollama chat stream ends with a `done: true` object carrying the numbers
 * that decide every tuning question in docs/MASTER.md:
 *
 *   prompt_eval_count     tokens actually prefilled  -> the real live-window size
 *   prompt_eval_duration  ns spent prefilling        -> prefill throughput
 *   eval_count/_duration  generated tokens + ns      -> decode throughput
 *   load_duration         ns spent loading weights   -> non-zero == model evicted
 *
 * Before this file those numbers were JSON.parsed and dropped on the floor.
 * Nothing here touches the response path: records land after res.end().
 *
 * `prefillReuse` is the point of the Stage 2 prompt reorder. Ollama re-prefills
 * from the first token that differs from the previous request's prompt, so a
 * block that changes every turn (retrieval) poisons everything after it. We keep
 * the previous prompt's text per conversation, measure the shared prefix, and
 * report it as a fraction. Reorder working == this trends toward 1.0.
 */

function intEnv(name, def, min, max) {
  const v = parseInt(process.env[name], 10);
  return Number.isFinite(v) && v >= min && v <= max ? v : def;
}

const RING = intEnv('METRICS_RING', 200, 20, 5000);
const _ring = [];
// conversationId -> the previous turn's outbound messages, by reference. Not a
// concatenated copy: the prefix walk below compares message by message, so the
// whole prompt never has to be materialised as one string.
const _prevPrompt = new Map();
const PREV_MAX = intEnv('METRICS_PREV_MAX', 32, 1, 512);

const ns = (v) => (Number.isFinite(v) && v > 0 ? v : 0);
const msOf = (n) => Math.round(ns(n) / 1e5) / 10; // ns -> ms, 0.1ms resolution
const tps = (count, duration) => (ns(duration) ? Math.round((count / (duration / 1e9)) * 10) / 10 : null);

/**
 * Shared-prefix length in characters between the previous and current prompt
 * for this conversation, as a fraction of the current prompt. This is the
 * closest proxy we have from outside the engine for how much of the KV cache
 * Ollama could reuse.
 */
function promptChars(messages) {
  let n = 0;
  for (let i = 0; i < messages.length; i++) n += messages[i].content.length;
  return n;
}

// Shared prefix between two message arrays, in characters. Whole messages are
// compared by identity first -- buildContext reuses the same content strings
// turn to turn, so an unchanged block costs one pointer compare instead of
// thousands of charCodeAt calls -- and only the first message that actually
// differs is walked character by character.
function sharedPrefixChars(prev, cur) {
  let shared = 0;
  const n = Math.min(prev.length, cur.length);
  for (let i = 0; i < n; i++) {
    const a = prev[i];
    const b = cur[i];
    if (a.role !== b.role) break;
    if (a.content === b.content) { shared += a.content.length; continue; }
    const m = Math.min(a.content.length, b.content.length);
    let j = 0;
    while (j < m && a.content.charCodeAt(j) === b.content.charCodeAt(j)) j++;
    shared += j;
    break;
  }
  return shared;
}

function prefixReuse(conversationId, messages) {
  const prev = _prevPrompt.get(conversationId);
  // Bounded LRU: delete-then-set keeps insertion order == recency.
  _prevPrompt.delete(conversationId);
  _prevPrompt.set(conversationId, messages);
  while (_prevPrompt.size > PREV_MAX) _prevPrompt.delete(_prevPrompt.keys().next().value);
  if (!Array.isArray(prev) || !prev.length || !messages.length) return null;
  const total = promptChars(messages);
  if (!total) return null;
  return Math.round((sharedPrefixChars(prev, messages) / total) * 1000) / 1000;
}

function forgetConversation(conversationId) { _prevPrompt.delete(conversationId); }

/**
 * Record one generation. `done` is the final Ollama chunk; `window` is the
 * composition breakdown from memory.buildContext(); `promptText` is the
 * concatenated outbound messages (used only for the reuse estimate).
 */
function record({ conversationId, characterId, model, backend, done, window, promptMessages, aborted }) {
  const d = done || {};
  const promptTokens = Number.isFinite(d.prompt_eval_count) ? d.prompt_eval_count : null;
  const evalTokens = Number.isFinite(d.eval_count) ? d.eval_count : null;
  // llama-server only. `cached_tokens` is llama.cpp's `timings.cache_n` -- how
  // many prompt tokens it genuinely served from the KV cache. prefillReuse
  // below is a char-prefix ESTIMATE that works on any backend; this is the
  // measurement, and the two should track each other closely.
  const cachedTokens = Number.isFinite(d.cached_tokens) ? d.cached_tokens : null;

  const rec = {
    at: new Date().toISOString(),
    conversationId: conversationId || null,
    characterId: characterId || null,
    backend: backend || 'ollama',
    model: model || null,
    aborted: Boolean(aborted),
    promptTokens,
    evalTokens,
    promptMs: msOf(d.prompt_eval_duration),
    evalMs: msOf(d.eval_duration),
    totalMs: msOf(d.total_duration),
    loadMs: msOf(d.load_duration),
    prefillTps: promptTokens != null ? tps(promptTokens, d.prompt_eval_duration) : null,
    decodeTps: evalTokens != null ? tps(evalTokens, d.eval_duration) : null,
    // Measured chars/token for THIS window. The MASTER.md token estimates are
    // all char-derived; this is what replaces the divide-by-4 guess.
    charsPerToken: promptTokens && window && window.promptChars
      ? Math.round((window.promptChars / promptTokens) * 100) / 100
      : null,
    prefillReuse: Array.isArray(promptMessages) && conversationId
      ? prefixReuse(conversationId, promptMessages)
      : null,
    cachedTokens,
    cacheReuse: cachedTokens != null && promptTokens
      ? Math.round((cachedTokens / promptTokens) * 1000) / 1000
      : null,
    window: window || null,
  };

  _ring.push(rec);
  // shift() is O(n) per call; splice the overflow off in one move instead. In
  // practice this is a single element, but a lowered METRICS_RING at runtime
  // would otherwise re-index the array once per dropped record.
  if (_ring.length > RING) _ring.splice(0, _ring.length - RING);
  return rec;
}

function pct(sorted, p) {
  if (!sorted.length) return null;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[i];
}

function summarize(recs) {
  const withTokens = recs.filter((r) => Number.isFinite(r.promptTokens));
  const prompt = withTokens.map((r) => r.promptTokens).sort((a, b) => a - b);
  const decode = recs.map((r) => r.decodeTps).filter(Number.isFinite).sort((a, b) => a - b);
  const prefill = recs.map((r) => r.prefillTps).filter(Number.isFinite).sort((a, b) => a - b);
  const reuse = recs.map((r) => r.prefillReuse).filter(Number.isFinite).sort((a, b) => a - b);
  const cacheReuse = recs.map((r) => r.cacheReuse).filter(Number.isFinite).sort((a, b) => a - b);
  const cpt = recs.map((r) => r.charsPerToken).filter(Number.isFinite).sort((a, b) => a - b);
  return {
    samples: recs.length,
    // The number that decides num_ctx. Compare p95 against Modelfile num_ctx.
    promptTokens: { p50: pct(prompt, 50), p95: pct(prompt, 95), max: prompt[prompt.length - 1] ?? null },
    decodeTps: { p50: pct(decode, 50), p95: pct(decode, 95) },
    prefillTps: { p50: pct(prefill, 50), p95: pct(prefill, 95) },
    // Stage 2 target: this should climb toward 1.0 once retrieval moves to the tail.
    prefillReuse: { p50: pct(reuse, 50), p95: pct(reuse, 95) },
    // Engine-reported, not estimated. Null on Ollama, which does not expose it.
    cacheReuse: cacheReuse.length
      ? { p50: pct(cacheReuse, 50), p95: pct(cacheReuse, 95), samples: cacheReuse.length }
      : null,
    charsPerToken: { p50: pct(cpt, 50) },
    // Non-zero load time means the model was evicted between turns — the
    // signature of VRAM pressure, not of a cold start, when it recurs.
    reloads: recs.filter((r) => r.loadMs > 50).length,
  };
}

function snapshot({ limit = 50, conversationId = null } = {}) {
  const recs = conversationId ? _ring.filter((r) => r.conversationId === conversationId) : _ring;
  return {
    config: { ring: RING },
    summary: summarize(recs),
    recent: recs.slice(-limit),
  };
}

function reset() { _ring.length = 0; _prevPrompt.clear(); }

module.exports = { record, snapshot, reset, forgetConversation };
