'use strict';

/**
 * Spawns the real backend against a throwaway database and a fake model.
 *
 * The whole point is isolation, and the isolation here is not decorative --
 * two separate incidents came from getting it wrong. A test backend that finds
 * the user's Turso credentials writes scratch rows into their live cloud
 * database, and a test backend that misses LOCAL_DB_PATH runs the one-time
 * plaintext->encrypted migration over their real local one. So:
 *
 *   cwd = a scratch directory, so dotenv finds NO .env at all. This is the
 *         load-bearing part: .env holds real credentials and dotenv reads it
 *         from the process cwd, not from the script's directory.
 *   LOCAL_DB_PATH is absolute, under the scratch directory, and asserted to be
 *         outside the repo before anything is spawned. db.js and settings.js
 *         both capture it at require time, so it can only be passed in the
 *         environment -- never assigned to process.env by a test.
 *   VESSEL_NO_SYNC is presence-only, so no empty-string quirk can defeat it.
 *   TURSO_DATABASE_URL / TURSO_AUTH_TOKEN are set (empty) rather than unset:
 *         dotenv does not overwrite a key that is already present, even when
 *         its value is ''. Belt and braces on top of the missing .env.
 *   DB_ENCRYPTION_KEY is a scratch key, so keystore never asks the real OS
 *         keychain for one -- and never generates and stores one either.
 *
 * And then the banner is read back and `-> sync: local-only` is REQUIRED. If
 * the child ever reports `sync: enabled`, it is killed before a single request
 * is sent. Everything above is reasoning about what should happen; this is the
 * only line that checks what did.
 */

const { spawn } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');

const fake = require('./fake-inference');

const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const SERVER_JS = path.join(PROJECT_ROOT, 'src', 'backend', 'server.js');

const BOOT_TIMEOUT_MS = 30000;

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

// Refuse to run against anything inside the repo. A scratch path that silently
// resolved to ./data/scenario.db is exactly how the real database got encrypted.
function assertScratch(dbPath) {
  const abs = path.resolve(dbPath);
  const rel = path.relative(PROJECT_ROOT, abs);
  const insideRepo = rel && !rel.startsWith('..') && !path.isAbsolute(rel);
  if (insideRepo) {
    throw new Error(`refusing to start a test backend with LOCAL_DB_PATH inside the repo: ${abs}`);
  }
  return abs;
}

/**
 * start(opts) -> handle
 *   opts.env     extra environment for the child (tuning knobs)
 *   opts.script  fake-model behaviour (see fake-inference.js)
 *
 * handle:
 *   url, port, dir, dbPath
 *   model        the fake inference handle (requests, script, reset)
 *   stdout()     everything the child has printed
 *   get/post/put/del/req   fetch wrappers rooted at the backend
 *   sse(body)    POST /chat and collect the stream (see collectSse)
 *   stop()       shut the child and the fake model down
 */
async function start(opts = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vessel-test-'));
  const dbPath = assertScratch(path.join(dir, 'data', 'scratch.db'));
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });

  const model = await fake.start({ script: opts.script });
  const port = await freePort();

  const env = {
    ...process.env,
    PORT: String(port),
    // --- isolation (see the header) -----------------------------------------
    VESSEL_NO_SYNC: '1',
    LOCAL_DB_PATH: dbPath,
    TURSO_DATABASE_URL: '',
    TURSO_AUTH_TOKEN: '',
    // A fresh key per start, so the keystore never touches the real OS keychain
    // and one test can never read another's database.
    //
    // This is also a trap for any harness that RESTARTS the backend against a
    // database it wrote earlier: the second start mints a different key and
    // every read fails with "Decryption failed for page=1". Long-running or
    // resumable harnesses must pin their own key through `opts.env`, which
    // spreads below and therefore wins.
    DB_ENCRYPTION_KEY: crypto.randomBytes(32).toString('hex'),
    // --- the model ----------------------------------------------------------
    OLLAMA_HOST: model.url,
    OLLAMA_MODEL: 'test-model',
    SUMMARIZER_MODEL: 'test-summarizer',
    EMBED_MODEL: 'test-embedder',
    INFERENCE_BACKEND: 'ollama',
    EMBED_BACKEND: 'ollama',
    SUMMARIZER_BACKEND: 'ollama',
    // Electron sets this when it re-executes itself as Node; inherited here it
    // confuses nothing, but clearing it keeps the child a plain Node process.
    ELECTRON_RUN_AS_NODE: undefined,
    ...(opts.env || {}),
  };
  for (const k of Object.keys(env)) if (env[k] === undefined) delete env[k];

  const child = spawn(process.execPath, [SERVER_JS], {
    cwd: dir, // <- no .env here, which is the entire point
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });

  let out = '';
  child.stdout.setEncoding('utf8');
  child.stderr.setEncoding('utf8');
  child.stdout.on('data', (d) => { out += d; });
  child.stderr.on('data', (d) => { out += d; });

  let exited = null;
  child.on('exit', (code, signal) => { exited = { code, signal }; });

  const url = `http://127.0.0.1:${port}`;

  const kill = () => {
    try { child.kill(); } catch { /* already gone */ }
  };

  // Wait for the banner, not for a port to open: the banner is what carries the
  // sync verdict, and there is no safe request to make before reading it.
  const deadline = Date.now() + BOOT_TIMEOUT_MS;
  while (!/Vessel backend on http/.test(out)) {
    if (exited) {
      await model.close();
      throw new Error(`backend exited during boot (code ${exited.code}):\n${out}`);
    }
    if (Date.now() > deadline) {
      kill(); await model.close();
      throw new Error(`backend did not boot within ${BOOT_TIMEOUT_MS} ms:\n${out}`);
    }
    await new Promise((r) => setTimeout(r, 25));
  }

  // The banner prints inference/model/rules before sync, so give the remaining
  // startup lines a moment to land, then insist on the verdict.
  const syncDeadline = Date.now() + 5000;
  while (!/-> sync: /.test(out) && Date.now() < syncDeadline) {
    await new Promise((r) => setTimeout(r, 25));
  }
  if (!/-> sync: local-only/.test(out)) {
    kill(); await model.close();
    throw new Error(`test backend did not report local-only sync -- refusing to continue:\n${out}`);
  }

  async function req(method, route, body, init = {}) {
    const res = await fetch(url + route, {
      method,
      headers: { ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}), ...(init.headers || {}) },
      body: body === undefined ? undefined : (typeof body === 'string' ? body : JSON.stringify(body)),
      ...init,
    });
    const text = await res.text();
    let json = null;
    try { json = text ? JSON.parse(text) : null; } catch { /* not json */ }
    return { status: res.status, headers: res.headers, text, json };
  }

  return {
    url, port, dir, dbPath, model, child,
    stdout: () => out,
    req,
    get: (r, init) => req('GET', r, undefined, init),
    post: (r, b, init) => req('POST', r, b, init),
    put: (r, b, init) => req('PUT', r, b, init),
    patch: (r, b, init) => req('PATCH', r, b, init),
    del: (r, b, init) => req('DELETE', r, b, init),
    sse: (body, init) => collectSse(url + '/chat', body, init),
    async stop() {
      const ended = new Promise((r) => child.once('exit', r));
      kill();
      await Promise.race([ended, new Promise((r) => setTimeout(r, 5000))]);
      await model.close();
      try { fs.rmSync(dir, { recursive: true, force: true }); } catch { /* windows lock */ }
    },
  };
}

/**
 * POST /chat and decode the SSE body into something assertable:
 *   { status, meta, text, events, errors, raw }
 * `onChunk` can abort mid-stream, which is how the stop-button path is tested.
 */
async function collectSse(endpoint, body, init = {}) {
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(init.headers || {}) },
    body: JSON.stringify(body),
    signal: init.signal,
  });

  if (!res.headers.get('content-type')?.includes('text/event-stream')) {
    const text = await res.text();
    let json = null;
    try { json = text ? JSON.parse(text) : null; } catch { /* not json */ }
    return { status: res.status, stream: false, json, text, meta: null, events: [], errors: [] };
  }

  const out = { status: res.status, stream: true, meta: null, text: '', events: [], errors: [] };
  const decoder = new TextDecoder();
  let buf = '';

  try {
    for await (const chunk of res.body) {
      buf += decoder.decode(chunk, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const evtMatch = /^event:\s*(.+)$/m.exec(frame);
        const dataMatch = /^data:\s*([\s\S]*)$/m.exec(frame);
        if (!dataMatch) continue;
        let data = null;
        try { data = JSON.parse(dataMatch[1]); } catch { continue; }
        const name = evtMatch ? evtMatch[1].trim() : 'message';
        out.events.push({ event: name, data });
        if (name === 'meta') out.meta = data;
        else if (name === 'error') out.errors.push(data.error);
        else if (data.message && typeof data.message.content === 'string') out.text += data.message.content;
        if (init.onChunk) await init.onChunk(out);
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') throw err;
    out.aborted = true;
  }

  return out;
}

module.exports = { start, collectSse, PROJECT_ROOT };
