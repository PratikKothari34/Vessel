'use strict';

/**
 * inference/ — the one place that knows which engine is generating tokens.
 *
 * Stage 3 of docs/decisions/0001. Everything above this module (server.js,
 * memory.js) talks to a single interface:
 *
 *   chatStream({ messages, options, signal })
 *       -> { ok:false, status, detail }        upstream refused before the body
 *       -> { ok:true, events }                 async iterator of
 *              { raw, delta, done }  raw = the SSE line to forward verbatim
 *              { error }             upstream failed mid-stream
 *   generate(model, prompt, { numCtx, numGpu }) -> string
 *   embed(model, text, { numGpu })       -> number[]
 *
 * `raw` is always an Ollama-shaped chat chunk. That is the wire format the
 * renderer parses (app/src/renderer/src/lib/api.js) and it predates the split;
 * the llama-server backend synthesizes it rather than forcing a renderer change
 * that would buy nothing.
 *
 * Chat, summarization and embedding are chosen SEPARATELY, and all three default
 * to Ollama. One llama-server process serves exactly one model, so a backend
 * that is right for chat is not automatically right for the two support models.
 *
 * That is not a hypothetical. INFERENCE_BACKEND used to move the summarizer with
 * chat, and because llama-server's generate() accepts-and-ignores its `model`
 * argument, SUMMARIZER_MODEL was then silently discarded and the roleplay chat
 * model wrote the summaries. Measured, it kept 8 of 13 facts and invented named
 * characters, which a rolling summary then feeds back as canon on every later
 * turn. Recall loss degrades a story; confabulation corrupts it. So the
 * summarizer gets its own switch and stays on Ollama, where SUMMARIZER_MODEL
 * means something and the model can be pinned to CPU (SUMMARIZER_NUM_GPU=0).
 */

const ollama = require('./ollama');
const llamaServer = require('./llama-server');

const BACKENDS = { ollama, 'llama-server': llamaServer };

function pick(value, fallback, varName) {
  const key = String(value || '').trim().toLowerCase();
  if (!key) return BACKENDS[fallback];
  const b = BACKENDS[key];
  if (!b) {
    throw new Error(
      `${varName}="${value}" is not a known backend. Use one of: ${Object.keys(BACKENDS).join(', ')}.`,
    );
  }
  return b;
}

const chat = pick(process.env.INFERENCE_BACKEND, 'ollama', 'INFERENCE_BACKEND');
// Default: whatever chat is NOT responsible for stays where it was. Moving the
// embedder is an independent, separately-verifiable step.
const embedder = pick(process.env.EMBED_BACKEND, 'ollama', 'EMBED_BACKEND');
// Same reasoning as the embedder, plus one of its own: llama-server ignores the
// model argument, so pointing the summarizer at it silently replaces
// SUMMARIZER_MODEL with the chat model. Opt in deliberately or not at all.
const summarizer = pick(process.env.SUMMARIZER_BACKEND, 'ollama', 'SUMMARIZER_BACKEND');

/**
 * The context window the chat engine is actually running with, or null when the
 * backend cannot be asked. llama-server fixes it at launch (`-c`), so a mismatch
 * with OLLAMA_NUM_CTX is a real misconfiguration worth reporting at startup
 * rather than discovering as a truncated story.
 */
async function probeContext({ timeoutMs = 1500 } = {}) {
  if (chat.name !== 'llama-server') return null;
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  try {
    const res = await fetch(`${chat.host}/props`, { signal: ac.signal });
    if (!res.ok) return null;
    const data = await res.json();
    const n = data && data.default_generation_settings && data.default_generation_settings.n_ctx;
    return Number.isFinite(n) ? n : null;
  } catch {
    return null; // not running yet is not an error here -- /health reports that
  } finally {
    clearTimeout(timer);
  }
}

module.exports = {
  name: chat.name,
  host: chat.host,
  model: chat.model,
  acceptsNumCtx: chat.acceptsNumCtx,

  chatStream: (args) => chat.chatStream(args),
  generate: (model, prompt, opts) => summarizer.generate(model, prompt, opts),
  embed: (model, text, opts) => embedder.embed(model, text, opts),

  probeContext,
  describe: () => ({
    chat: chat.describe(),
    // Only the host: which MODEL gets embedded is memory.js's EMBED_MODEL, not
    // the adapter's chat model, and printing that here read as if the chat model
    // were producing the vectors.
    embed: { backend: embedder.name, host: embedder.host },
    // Which MODEL summarizes is memory.js's SUMMARIZER_MODEL -- but only while
    // this backend honours it. llama-server does not, so name the backend here
    // and let /health show the mismatch instead of hiding it.
    summarize: { backend: summarizer.name, host: summarizer.host, honoursModel: summarizer.name !== 'llama-server' },
  }),

  _embedder: embedder,
};
