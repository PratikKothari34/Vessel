'use strict';

/**
 * A stand-in for Ollama.
 *
 * The backend talks to exactly three endpoints (/api/chat, /api/generate,
 * /api/embeddings) and this serves all three with the same wire shapes, so the
 * real inference adapter, the real SSE relay and the real memory engine are
 * exercised end to end. Only the model is fake.
 *
 * That matters for more than speed. A real model makes every assertion
 * probabilistic -- reply text, token counts and timings all move -- and a test
 * that cannot state the expected answer cannot fail honestly. Here the reply is
 * scripted, `prompt_eval_count` is whatever the test asked for, and the
 * embedding is a pure function of the text, so retrieval ranking is decidable.
 *
 * It also records every request, which is how the prompt-order tests assert on
 * what the model was actually sent rather than on what the code meant to send.
 */

const http = require('http');

const EMBED_DIM = 768;

// A deterministic unit-ish vector derived from the text. Same text -> same
// vector, similar text -> similar vector, which is all retrieval needs to be
// testable. (xorshift32, seeded by an FNV-1a hash of the text.)
function fakeEmbedding(text, dim = EMBED_DIM) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  let s = h || 1;
  const out = new Array(dim);
  for (let i = 0; i < dim; i++) {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const parts = [];
    req.on('data', (c) => parts.push(c));
    req.on('end', () => {
      const raw = Buffer.concat(parts).toString('utf8');
      try { resolve(raw ? JSON.parse(raw) : {}); } catch (e) { reject(e); }
    });
    req.on('error', reject);
  });
}

/**
 * start() resolves to a handle:
 *   url            where to point OLLAMA_HOST
 *   chatRequests   every /api/chat body received, in order
 *   genRequests    every /api/generate body
 *   embedRequests  every /api/embeddings body
 *   script         mutable behaviour knobs (see defaults below)
 *   close()
 */
async function start(opts = {}) {
  const state = {
    chatRequests: [],
    genRequests: [],
    embedRequests: [],
    script: {
      // Text the chat endpoint streams back, one chunk per word.
      reply: 'This is a scripted reply from the fake model.',
      // ms between chunks; keeps "stop mid-stream" tests deterministic.
      chunkDelayMs: 0,
      // ms before the first byte of the chat body.
      firstByteDelayMs: 0,
      // Set to a status code to refuse the chat request outright.
      chatStatus: 0,
      // Set to a string to emit an Ollama in-band {"error": ...} line.
      chatError: null,
      // What /api/generate returns (the rolling summary).
      summary: 'SUMMARY: the story so far.',
      // Set to true to make /api/generate fail, so a fold is abandoned.
      generateFails: false,
      // Set to true to make /api/embeddings return nothing usable.
      embedFails: false,
      // Numbers the final done chunk reports.
      promptEvalCount: 100,
      evalCount: 10,
      ...(opts.script || {}),
    },
  };

  const server = http.createServer(async (req, res) => {
    const json = (code, body) => {
      const s = JSON.stringify(body);
      res.writeHead(code, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(s) });
      res.end(s);
    };

    let body = {};
    try { body = await readBody(req); } catch { return json(400, { error: 'bad json' }); }

    if (req.url === '/api/chat') {
      state.chatRequests.push(body);
      const sc = state.script;
      if (sc.chatStatus) return json(sc.chatStatus, { error: 'scripted refusal' });

      res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
      if (sc.firstByteDelayMs) await new Promise((r) => setTimeout(r, sc.firstByteDelayMs));

      if (sc.chatError) {
        res.write(`${JSON.stringify({ error: sc.chatError })}\n`);
        return res.end();
      }

      // One NDJSON line per word, exactly as Ollama streams tokens.
      const words = sc.reply.length ? sc.reply.split(/(?<=\s)/) : [];
      for (const w of words) {
        if (res.writableEnded || res.destroyed) return;
        res.write(`${JSON.stringify({
          model: body.model, created_at: new Date().toISOString(),
          message: { role: 'assistant', content: w }, done: false,
        })}\n`);
        if (sc.chunkDelayMs) await new Promise((r) => setTimeout(r, sc.chunkDelayMs));
      }
      if (res.writableEnded || res.destroyed) return;
      res.write(`${JSON.stringify({
        model: body.model, created_at: new Date().toISOString(),
        message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop',
        total_duration: 1e9, load_duration: 1e6,
        prompt_eval_count: sc.promptEvalCount, prompt_eval_duration: 5e8,
        eval_count: sc.evalCount, eval_duration: 5e8,
      })}\n`);
      return res.end();
    }

    if (req.url === '/api/generate') {
      state.genRequests.push(body);
      if (state.script.generateFails) return json(500, { error: 'scripted generate failure' });
      return json(200, { model: body.model, response: state.script.summary, done: true });
    }

    if (req.url === '/api/embeddings') {
      state.embedRequests.push(body);
      if (state.script.embedFails) return json(200, { embedding: [] });
      return json(200, { embedding: fakeEmbedding(String(body.prompt || '')) });
    }

    return json(404, { error: `no fake route for ${req.url}` });
  });

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();

  return {
    url: `http://127.0.0.1:${port}`,
    port,
    get chatRequests() { return state.chatRequests; },
    get genRequests() { return state.genRequests; },
    get embedRequests() { return state.embedRequests; },
    script: state.script,
    reset() {
      state.chatRequests.length = 0;
      state.genRequests.length = 0;
      state.embedRequests.length = 0;
    },
    close() {
      return new Promise((resolve) => {
        server.closeAllConnections?.();
        server.close(resolve);
      });
    },
  };
}

module.exports = { start, fakeEmbedding, EMBED_DIM };
