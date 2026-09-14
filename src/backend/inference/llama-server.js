'use strict';

/**
 * llama.cpp `llama-server` inference backend (Stage 3 of docs/decisions/0001).
 *
 * Why this exists, in one line: Ollama keys a resident model instance on
 * (model, num_ctx), so the summarizer evicts the chat model and the next user
 * message pays a 19.4 s reload. llama-server holds one model for the life of the
 * process, serves chat and summarization from the same weights, and reports how
 * many prompt tokens it actually reused instead of making us estimate it.
 *
 * Shape of the swap:
 *
 *   Ollama                          llama-server
 *   ------                          ------------
 *   num_ctx per request             `-c` at launch (nothing can evict)
 *   /api/chat, NDJSON               /v1/chat/completions, SSE
 *   /api/generate (applies TEMPLATE) /v1/chat/completions, one user message
 *   /api/embeddings                 /v1/embeddings (separate instance/port)
 *   prompt_eval_count (est. reuse)  timings.cache_n (measured reuse)
 *
 * The chat model and the embedding model are two processes here -- one
 * llama-server serves one model. index.js lets the embedder stay on Ollama
 * while chat moves, so the swap can be done one half at a time.
 *
 * Output contract is identical to ollama.js: `raw` is an Ollama-shaped chat
 * chunk, synthesized here, so the SSE wire and the renderer do not change.
 */

const { fetchRetry, errorBody, nsFromMs, trimSlash, MAX_FRAME_BYTES, UNDELIMITED } = require('./util');
const modelfile = require('./modelfile');

const CHAT_HOST = trimSlash(process.env.LLAMA_CHAT_URL || 'http://127.0.0.1:8080');
const MODEL = process.env.LLAMA_CHAT_MODEL || process.env.OLLAMA_MODEL || 'vessel';
// llama.cpp reuses the longest common prefix of the KV cache when this is on.
// It is the default upstream; sent explicitly so the behaviour is in our code,
// not in whatever build the user happens to be running.
const CACHE_PROMPT = process.env.LLAMA_CACHE_PROMPT !== '0';
// Attaches a `timings` object to every streamed chunk, which is the only way to
// read `cache_n` (tokens served from the KV cache) on the OpenAI-compatible
// endpoint. Costs one small JSON object per token.
const TIMINGS_PER_TOKEN = process.env.LLAMA_TIMINGS !== '0';

// ---- Sampling translation -------------------------------------------------
// Characters carry Ollama-named sampling options (characters.js cleans them).
// Map the ones llama.cpp understands and drop the rest loudly-once, rather than
// forwarding unknown keys and getting a 400 mid-conversation.
const SAMPLING_MAP = {
  num_predict: 'max_tokens',
  temperature: 'temperature',
  top_p: 'top_p',
  top_k: 'top_k',
  min_p: 'min_p',
  typical_p: 'typical_p',
  repeat_penalty: 'repeat_penalty',
  repeat_last_n: 'repeat_last_n',
  presence_penalty: 'presence_penalty',
  frequency_penalty: 'frequency_penalty',
  mirostat: 'mirostat',
  mirostat_tau: 'mirostat_tau',
  mirostat_eta: 'mirostat_eta',
  seed: 'seed',
  stop: 'stop',
};
// num_ctx is a launch flag here, not a request field. Silently dropping it
// would mean a character asking for a bigger window gets a smaller one with no
// sign, so it is warned about explicitly, once per key.
const _warned = new Set();
function warnOnce(key, msg) {
  if (_warned.has(key)) return;
  _warned.add(key);
  console.warn(`  !! llama-server: ${msg}`);
}

// ---- Model-side defaults ---------------------------------------------------
// Ollama bakes every Modelfile PARAMETER into the model and applies it to any
// request that does not override that key. llama-server loads the bare GGUF and
// applies nothing, so without this the model runs at llama.cpp's stock sampling
// (temperature 0.8, no min_p, no repeat window) instead of the tuned values the
// Modelfile declares. Read from the same file so there is one source of truth.
//
// The Modelfile SYSTEM is deliberately NOT applied here. Measured: Ollama drops
// it whenever the client sends a system message of its own, and the app always
// does. server.js carries it in the persona message instead, which is what
// makes the two backends produce the same prompt.
const MODELFILE = modelfile.load();
const MODEL_PARAMS = { ...MODELFILE.params };
// The window is a launch flag here; translateSampling would warn about it on
// every startup for a value nobody passed in this request.
delete MODEL_PARAMS.num_ctx;

function translateSampling(options = {}) {
  const out = {};
  for (const [k, v] of Object.entries(options)) {
    if (v === undefined || v === null) continue;
    if (k === 'num_ctx') {
      warnOnce('num_ctx', `num_ctx=${v} ignored -- llama-server takes the context window from its own -c flag at launch. Start it with -c ${v}.`);
      continue;
    }
    const mapped = SAMPLING_MAP[k];
    if (!mapped) { warnOnce(k, `sampling option "${k}" has no llama.cpp equivalent and was dropped.`); continue; }
    out[mapped] = v;
  }
  return out;
}

// ---- Ollama-shaped chunk synthesis ---------------------------------------

function contentChunk(text) {
  return JSON.stringify({
    model: MODEL,
    created_at: new Date().toISOString(),
    message: { role: 'assistant', content: text },
    done: false,
  });
}

/**
 * Final chunk, in the shape metrics.js already parses.
 *
 * `timings.prompt_n` counts only the tokens llama.cpp actually had to process;
 * `cache_n` counts the ones it reused. Ollama's prompt_eval_count is the WHOLE
 * prompt with a near-zero duration when it hits cache, so the two are summed
 * here -- otherwise "promptTokens p95 vs num_ctx", the number that decides the
 * window size, would silently shrink by whatever the cache absorbed.
 */
function doneChunk({ timings, usage, finishReason }) {
  const t = timings || {};
  const u = usage || {};
  const cached = Number.isFinite(t.cache_n) ? t.cache_n : null;
  const processed = Number.isFinite(t.prompt_n) ? t.prompt_n : null;
  const promptTokens = processed != null
    ? processed + (cached || 0)
    : (Number.isFinite(u.prompt_tokens) ? u.prompt_tokens : null);
  const evalTokens = Number.isFinite(t.predicted_n)
    ? t.predicted_n
    : (Number.isFinite(u.completion_tokens) ? u.completion_tokens : null);

  const promptNs = nsFromMs(t.prompt_ms);
  const evalNs = nsFromMs(t.predicted_ms);

  const chunk = {
    model: MODEL,
    created_at: new Date().toISOString(),
    message: { role: 'assistant', content: '' },
    done: true,
    done_reason: finishReason || 'stop',
    // load_duration is always 0: the weights are resident for the life of the
    // process. That is the whole point of the stage -- a non-zero value here
    // would mean llama-server restarted under us.
    load_duration: 0,
    total_duration: promptNs + evalNs,
  };
  if (promptTokens != null) chunk.prompt_eval_count = promptTokens;
  if (promptNs) chunk.prompt_eval_duration = promptNs;
  if (evalTokens != null) chunk.eval_count = evalTokens;
  if (evalNs) chunk.eval_duration = evalNs;
  // Not an Ollama field. metrics.js reads it when present and reports measured
  // KV reuse alongside the char-prefix estimate that works on both backends.
  if (cached != null) chunk.cached_tokens = cached;
  return chunk;
}

// ---- Chat ------------------------------------------------------------------

async function chatStream({ messages, options, signal }) {
  const body = {
    model: MODEL,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    cache_prompt: CACHE_PROMPT,
    // Modelfile first, request second: a character's own sampling still wins,
    // exactly as it does on Ollama.
    ...translateSampling(MODEL_PARAMS),
    ...translateSampling(options),
  };
  if (TIMINGS_PER_TOKEN) body.timings_per_token = true;

  const res = await fetch(`${CHAT_HOST}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    return { ok: false, status: res.status, detail: await errorBody(res) };
  }

  return { ok: true, events: iterate(res) };
}

async function* iterate(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  // Last-writer-wins: llama.cpp repeats timings on each chunk and puts usage on
  // the final one, so carrying the most recent of each gives the true totals.
  let timings = null;
  let usage = null;
  let finishReason = null;
  let sawDone = false;

  const frame = function* (payload) {
    if (payload === '[DONE]') { sawDone = true; return; }
    let obj;
    try { obj = JSON.parse(payload); } catch { return; } // keepalive / partial
    if (obj.error) {
      yield { error: String(obj.error.message || obj.error) };
      return;
    }
    if (obj.timings) timings = obj.timings;
    if (obj.usage) usage = obj.usage;
    const choice = obj.choices && obj.choices[0];
    if (choice) {
      if (choice.finish_reason) finishReason = choice.finish_reason;
      const text = choice.delta && typeof choice.delta.content === 'string' ? choice.delta.content : '';
      if (text) yield { raw: contentChunk(text), delta: text, done: null };
    }
  };

  // SSE frames are separated by a blank line; a frame may carry several
  // `data:` lines that concatenate.
  const drain = function* (flush) {
    for (;;) {
      const idx = buffer.indexOf('\n\n');
      if (idx === -1) break;
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let payload = '';
      for (const line of raw.split('\n')) {
        const l = line.trim();
        if (l.startsWith('data:')) payload += l.slice(5).trim();
      }
      if (payload) yield* frame(payload);
    }
    if (flush && buffer.trim()) {
      let payload = '';
      for (const line of buffer.trim().split('\n')) {
        const l = line.trim();
        if (l.startsWith('data:')) payload += l.slice(5).trim();
      }
      buffer = '';
      if (payload) yield* frame(payload);
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // Normalize CRLF so the \n\n frame split holds regardless of proxy.
    if (buffer.indexOf('\r') !== -1) buffer = buffer.replace(/\r\n/g, '\n');
    yield* drain(false);
    // Same bound as the NDJSON reader, and for the same reason: a host that
    // streams without ever closing a frame must fail loudly.
    if (buffer.length > MAX_FRAME_BYTES) {
      await reader.cancel().catch(() => {});
      yield { error: UNDELIMITED };
      return;
    }
  }
  yield* drain(true);

  // Only synthesize a done chunk if the stream actually finished. An aborted
  // read must record as aborted, exactly as it does on Ollama.
  if (sawDone || finishReason || usage || timings) {
    const chunk = doneChunk({ timings, usage, finishReason });
    yield { raw: JSON.stringify(chunk), delta: '', done: chunk };
  }
}

// ---- Completion (summarizer) ----------------------------------------------
// Deliberately the chat endpoint and not /completion: Ollama's /api/generate
// applies the model's chat TEMPLATE, so a raw completion here would change what
// the summarizer sees and make the two backends produce different summaries for
// the same history. `model` is accepted and ignored -- one server, one model.
async function generate(_model, prompt, { numCtx } = {}) {
  if (numCtx) {
    warnOnce('gen_num_ctx', `summarizer num_ctx=${numCtx} ignored -- one llama-server process, one window (-c).`);
  }
  const res = await fetchRetry(`${CHAT_HOST}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: 'user', content: prompt }],
      stream: false,
      // The summary prompt is a different prefix every time and would evict the
      // chat conversation's slot. Give it its own slot instead of poisoning the
      // one the story is using.
      cache_prompt: false,
      // Deliberately NOT the Modelfile's sampling. Those values (temperature
      // 0.9, min_p 0.05) are tuned to make roleplay prose surprising, which is
      // the opposite of what a summary needs. On Ollama the summarizer is a
      // different model with its own defaults, so there is no parity to keep
      // here -- only a right answer, which is low variance.
      temperature: 0.3,
      top_p: 0.9,
      repeat_penalty: 1.05,
    }),
  });
  const data = await res.json();
  const choice = data.choices && data.choices[0];
  const text = choice && choice.message && typeof choice.message.content === 'string'
    ? choice.message.content : '';
  return text.trim();
}

// ---- Embeddings ------------------------------------------------------------
// A second llama-server, started with --embedding and the embedding GGUF.
// LLAMA_EMBED_URL must point at it; index.js keeps the embedder on Ollama by
// default so the chat swap can be tested on its own.
const EMBED_HOST = trimSlash(process.env.LLAMA_EMBED_URL || 'http://127.0.0.1:8081');

async function embed(model, text) {
  const res = await fetchRetry(`${EMBED_HOST}/v1/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: model || 'embed', input: text }),
  });
  const data = await res.json();
  const row = data && Array.isArray(data.data) ? data.data[0] : null;
  let vec = row && row.embedding;
  // With --pooling none llama.cpp returns one vector per token. We want a
  // single vector; that configuration is a misconfiguration for our use, so say
  // so rather than silently picking row 0.
  if (Array.isArray(vec) && Array.isArray(vec[0])) {
    if (vec.length !== 1) {
      throw new Error(`embed: llama-server returned ${vec.length} token vectors -- start it with --pooling mean`);
    }
    vec = vec[0];
  }
  if (!Array.isArray(vec) || vec.length === 0) throw new Error('embed: model returned no embedding');
  return vec;
}

module.exports = {
  name: 'llama-server',
  host: CHAT_HOST,
  model: MODEL,
  // The context window is fixed at launch. server.js uses this to decide
  // whether to send num_ctx at all, and to warn when the two disagree.
  acceptsNumCtx: false,
  chatStream,
  generate,
  embed,
  describe: () => ({
    backend: 'llama-server',
    host: CHAT_HOST,
    model: MODEL,
    embedHost: EMBED_HOST,
    cachePrompt: CACHE_PROMPT,
    modelParams: MODEL_PARAMS,
  }),
  // Exported for the verification rig; not used on the request path.
  _translateSampling: translateSampling,
  _doneChunk: doneChunk,
  _modelParams: MODEL_PARAMS,
};
