'use strict';

/**
 * fetchRetry, and the retry policy it encodes.
 *
 * The policy is the interesting part: a 5xx or a dropped connection is worth
 * another attempt because the model server is often mid-load, but a 4xx is a
 * bad request and retrying it just triples the latency of a failure the caller
 * already has the answer to.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const path = require('node:path');

const util = require(path.resolve(__dirname, '..', '..', 'src', 'backend', 'inference', 'util.js'));

// A server whose handler is swapped per test, so each case controls exactly
// what the N-th attempt sees.
async function withServer(handler, fn) {
  let calls = 0;
  const server = http.createServer((req, res) => { calls++; handler(req, res, calls); });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const url = `http://127.0.0.1:${server.address().port}/`;
  try { return await fn(url, () => calls); }
  finally {
    server.closeAllConnections?.();
    await new Promise((r) => server.close(r));
  }
}

test('a 200 is returned on the first attempt', async () => {
  await withServer((_req, res) => res.end('ok'), async (url, calls) => {
    const res = await util.fetchRetry(url, {});
    assert.equal(res.status, 200);
    assert.equal(calls(), 1);
  });
});

test('a 5xx is retried and a later success is returned', async () => {
  await withServer((_req, res, n) => {
    if (n < 3) { res.writeHead(503); res.end('loading model'); return; }
    res.end('ok');
  }, async (url, calls) => {
    const res = await util.fetchRetry(url, {}, 3);
    assert.equal(res.status, 200);
    assert.equal(calls(), 3);
  });
});

test('a 4xx is not retried -- it will stay bad', async () => {
  await withServer((_req, res) => { res.writeHead(404); res.end('no such model'); },
    async (url, calls) => {
      await assert.rejects(util.fetchRetry(url, {}, 3), /404/);
      assert.equal(calls(), 1, 'retrying a client error only delays the failure');
    });
});

test('the last 5xx body is what surfaces after the tries run out', async () => {
  await withServer((_req, res, n) => { res.writeHead(500); res.end(`attempt ${n}`); },
    async (url, calls) => {
      await assert.rejects(util.fetchRetry(url, {}, 2), /attempt 2/);
      assert.equal(calls(), 2);
    });
});

test('a transport failure is retried too', async () => {
  // Destroy the socket without a response: this is what a model server that is
  // still starting up looks like from here.
  await withServer((req, res, n) => {
    if (n < 2) { res.socket.destroy(); return; }
    res.end('ok');
  }, async (url, calls) => {
    const res = await util.fetchRetry(url, {}, 3);
    assert.equal(res.status, 200);
    assert.equal(calls(), 2);
  });
});

test('an unreachable host rejects rather than hanging', async () => {
  // Port 1 on loopback: nothing listens, connection refused immediately.
  await assert.rejects(util.fetchRetry('http://127.0.0.1:1/', {}, 1));
});

test('an abort signal is honoured and not retried around', async () => {
  await withServer((_req, res) => { /* never respond */ }, async (url, calls) => {
    const ac = new AbortController();
    const p = util.fetchRetry(url, { signal: ac.signal }, 3);
    setTimeout(() => ac.abort(), 50);
    await assert.rejects(p, (e) => e.name === 'AbortError' || /abort/i.test(e.message));
    assert.equal(calls(), 1, 'a user-cancelled request must not be re-issued');
  });
});

test('a JSON error body comes back parsed', async () => {
  await withServer((_req, res) => {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'model not found' }));
  }, async (url) => {
    const res = await fetch(url);
    assert.deepEqual(await util.errorBody(res), { error: 'model not found' });
  });
});

test('a non-JSON error page comes back as text rather than as nothing', async () => {
  // The old read-as-JSON-then-fall-back-to-text pattern always produced '':
  // the failed JSON read had already disturbed the body. This is the case the
  // fallback existed for -- an engine host that is not an engine.
  await withServer((_req, res) => {
    res.writeHead(502, { 'Content-Type': 'text/html' });
    res.end('<html><body>Bad Gateway</body></html>');
  }, async (url) => {
    const body = await util.errorBody(await fetch(url));
    assert.match(body, /Bad Gateway/);
  });
});

test('a runaway error body is cut and says so', async () => {
  await withServer((_req, res) => {
    res.writeHead(500);
    res.end('x'.repeat(util.MAX_ERROR_BODY * 4));
  }, async (url) => {
    const body = await util.errorBody(await fetch(url));
    assert.ok(body.length <= util.MAX_ERROR_BODY + 3, `kept ${body.length} chars`);
    assert.ok(body.endsWith('...'), 'a cut body must say it was cut');
  });
});

test('nsFromMs converts llama.cpp float milliseconds to Ollama nanoseconds', () => {
  assert.equal(util.nsFromMs(1), 1e6);
  assert.equal(util.nsFromMs(1234.5), 1234500000);
  // Zero, negatives and non-numbers all mean "no measurement", which metrics.js
  // reads as absent rather than as a real zero-duration generation.
  assert.equal(util.nsFromMs(0), 0);
  assert.equal(util.nsFromMs(-5), 0);
  assert.equal(util.nsFromMs(NaN), 0);
  assert.equal(util.nsFromMs(undefined), 0);
  assert.equal(util.nsFromMs('100'), 0);
});

test('trimSlash normalises a host so URLs never double up', () => {
  assert.equal(util.trimSlash('http://h:1/'), 'http://h:1');
  assert.equal(util.trimSlash('http://h:1///'), 'http://h:1');
  assert.equal(util.trimSlash('http://h:1'), 'http://h:1');
  assert.equal(util.trimSlash(undefined), '');
  assert.equal(util.trimSlash(null), '');
});

test('sleep resolves after the requested delay', async () => {
  const t0 = Date.now();
  await util.sleep(40);
  assert.ok(Date.now() - t0 >= 35);
});
