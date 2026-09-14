'use strict';

/**
 * Ollama inference backend.
 *
 * Behaviour-identical to the code that lived inline in server.js / memory.js
 * before Stage 3. In particular the upstream NDJSON line is carried through
 * UNTOUCHED as `raw`: the SSE wire the renderer parses is an Ollama chat chunk,
 * and this backend must keep producing it byte for byte.
 */

const { fetchRetry, errorBody, MAX_FRAME_BYTES, UNDELIMITED } = require('./util');

const HOST = (process.env.OLLAMA_HOST || 'http://localhost:11434').replace(/\/+$/, '');
const CHAT_URL = `${HOST}/api/chat`;
const MODEL = process.env.OLLAMA_MODEL || 'vessel';

/**
 * Stream a chat completion.
 *
 * Resolves to `{ ok: false, status, detail }` when the upstream refuses before
 * the body starts, or `{ ok: true, events }` where `events` is an async
 * iterator of:
 *   { raw, delta, done }  raw = the exact line to forward on the SSE wire
 *   { error }             upstream reported an error mid-stream
 *
 * Transport failures (unreachable, aborted before headers) throw, so the caller
 * can tell "never started" from "started and failed".
 */
async function chatStream({ messages, options, signal }) {
  const res = await fetch(CHAT_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: MODEL, messages, stream: true, options }),
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

  const parse = (line) => {
    if (!line) return null;
    let obj;
    try { obj = JSON.parse(line); } catch { return { raw: line }; } // forward, don't interpret
    if (obj.error) return { raw: line, error: String(obj.error) };
    return {
      raw: line,
      delta: obj.message && typeof obj.message.content === 'string' ? obj.message.content : '',
      done: obj.done === true ? obj : null,
    };
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf('\n')) !== -1) {
      const evt = parse(buffer.slice(0, nl).trim());
      buffer = buffer.slice(nl + 1);
      if (evt) yield evt;
    }
    // Checked after draining, so a chunk carrying many whole lines is never
    // mistaken for one runaway line.
    if (buffer.length > MAX_FRAME_BYTES) {
      await reader.cancel().catch(() => {});
      yield { raw: '', error: UNDELIMITED };
      return;
    }
  }
  const tail = parse(buffer.trim());
  if (tail) yield tail;
}

// Non-streaming completion, used by the summarizer. Ollama's /api/generate
// applies the model's own TEMPLATE unless raw:true, so the prompt arrives
// wrapped exactly as a chat turn would be.
async function generate(model, prompt, { numCtx, numGpu } = {}) {
  const options = {};
  if (numCtx) options.num_ctx = numCtx;
  // numGpu 0 pins the model to CPU. The summariser uses it for the same reason
  // the embedder does: on an 8 GB card the chat model already owns 6.1 GB, so a
  // second model on the GPU does not evict it (llama-server cannot be evicted)
  // but does oversubscribe the card, and the driver pages the chat model's
  // working set out to system RAM. See docs/MASTER.md, "Eviction is gone;
  // contention replaced it".
  if (Number.isFinite(numGpu) && numGpu >= 0) options.num_gpu = numGpu;
  const res = await fetchRetry(`${HOST}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, prompt, stream: false, options }),
  });
  const data = await res.json();
  return (data.response || '').trim();
}

async function embed(model, text, { numGpu } = {}) {
  const body = { model, prompt: text };
  if (Number.isFinite(numGpu) && numGpu >= 0) body.options = { num_gpu: numGpu };
  const res = await fetchRetry(`${HOST}/api/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!Array.isArray(data.embedding) || data.embedding.length === 0) {
    throw new Error('embed: model returned no embedding');
  }
  return data.embedding;
}

module.exports = {
  name: 'ollama',
  host: HOST,
  model: MODEL,
  // Ollama takes the context window per request, so tuning it never needs an
  // `ollama create`. llama-server takes it at launch instead -- see that file.
  acceptsNumCtx: true,
  chatStream,
  generate,
  embed,
  describe: () => ({ backend: 'ollama', host: HOST, model: MODEL, chatUrl: CHAT_URL }),
};
