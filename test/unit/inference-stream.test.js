'use strict';

/**
 * The bound on a stream that never delimits itself.
 *
 * Both backends accumulate bytes until they see the end of a line (Ollama's
 * NDJSON) or the end of a frame (llama-server's SSE). A well-behaved engine
 * sends one small object per token, so the buffer holds a few hundred bytes at
 * most -- but nothing in either protocol promises a delimiter ever arrives, and
 * the engine host is a field the user types into. Point it at a stale port, a
 * proxy, or anything else that streams without one, and the buffer grew until
 * the process died, with no error to explain it.
 *
 * Both readers capture their host at require time, so the server below has to
 * be listening and the environment set BEFORE the backend is required -- which
 * is why the requires are inside the tests rather than at the top.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const path = require('node:path');

const backendDir = path.resolve(__dirname, '..', '..', 'src', 'backend', 'inference');
const { MAX_FRAME_BYTES, UNDELIMITED } = require(path.join(backendDir, 'util.js'));

let server;
let base;

// Answers every route with twice the bound in one unbroken run of bytes: no
// newline, so no NDJSON line ever completes and no SSE frame ever closes.
test.before(async () => {
  server = http.createServer((_req, res) => {
    res.on('error', () => {});
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
    const filler = 'x'.repeat(64 * 1024);
    let sent = 0;
    const pump = () => {
      while (sent < MAX_FRAME_BYTES * 2) {
        sent += filler.length;
        if (!res.write(filler)) { res.once('drain', pump); return; }
      }
      res.end();
    };
    pump();
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}`;
  process.env.OLLAMA_HOST = base;
  process.env.LLAMA_CHAT_URL = base;
});

test.after(async () => {
  server.closeAllConnections?.();
  await new Promise((r) => server.close(r));
});

const ask = { messages: [{ role: 'user', content: 'hi' }], options: {} };

async function drain(events) {
  const out = [];
  for await (const e of events) out.push(e);
  return out;
}

test('an ollama stream that never ends a line is reported, not buffered', async () => {
  const ollama = require(path.join(backendDir, 'ollama.js'));
  const res = await ollama.chatStream(ask);
  assert.ok(res.ok, 'the response itself was fine -- the body is the problem');

  const events = await drain(res.events);
  assert.equal(events.length, 1, `one report, not a hang: ${JSON.stringify(events).slice(0, 200)}`);
  assert.equal(events[0].error, UNDELIMITED);
});

test('a llama-server stream that never closes a frame is reported, not buffered', async () => {
  const llama = require(path.join(backendDir, 'llama-server.js'));
  const res = await llama.chatStream(ask);
  assert.ok(res.ok);

  const events = await drain(res.events);
  assert.equal(events.length, 1, `one report, not a hang: ${JSON.stringify(events).slice(0, 200)}`);
  assert.equal(events[0].error, UNDELIMITED);
});
