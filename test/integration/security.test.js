'use strict';

/**
 * Red-team pass over the HTTP surface.
 *
 * Vessel binds 127.0.0.1 and trusts whoever reaches it, so the perimeter is
 * thin on purpose: the guards that matter are the ones stopping a *browser*
 * from becoming the attacker's proxy into that trusted socket. A page the user
 * opens can reach localhost, and DNS rebinding lets it do so same-origin. So
 * the Host guard, the CORS policy and the app header every write must carry are
 * not hardening theatre -- they are the actual boundary.
 *
 * The rest is the usual injection surface: JSON bodies that carry __proto__,
 * ids that carry path separators, reply text that carries SSE framing, and
 * error paths that carry stack traces back to the caller.
 *
 * Never sends tursoToken. keystore.setTursoToken writes the REAL OS keychain.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const path = require('node:path');

const backend = require(path.resolve(__dirname, '..', 'helpers', 'backend.js'));

let app;
let hero;

/**
 * A request with a Host header we actually control.
 *
 * fetch() cannot do this: Host is a forbidden header name, so undici silently
 * drops the override and sends the real authority -- which would make every
 * DNS-rebinding assertion below pass against a request the guard never saw.
 * node:http sends exactly what it is given.
 */
function raw(method, route, { host, headers = {}, body } = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request({
      host: '127.0.0.1',
      port: app.port,
      method,
      path: route,
      headers: {
        ...(host !== undefined ? { Host: host } : {}),
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
        ...headers,
      },
    }, (res) => {
      let text = '';
      res.setEncoding('utf8');
      res.on('data', (d) => { text += d; });
      res.on('end', () => {
        let json = null;
        try { json = text ? JSON.parse(text) : null; } catch { /* not json */ }
        resolve({ status: res.statusCode, headers: res.headers, text, json });
      });
    });
    req.on('error', reject);
    if (body !== undefined) req.write(typeof body === 'string' ? body : JSON.stringify(body));
    req.end();
  });
}

test.before(async () => {
  app = await backend.start({ script: { reply: 'A scripted reply.' } });
  hero = (await app.post('/characters', { name: 'Warden', persona: 'Terse.' })).json;
});
test.after(async () => { if (app) await app.stop(); });

const say = (content) => ({ role: 'user', content });

// ---- DNS rebinding -------------------------------------------------------

test('a request carrying a foreign Host header is refused', async () => {
  for (const host of ['evil.example', 'attacker.test:3001', 'localhost.evil.example', '127.0.0.1.evil.example']) {
    const res = await raw('GET', '/health', { host });
    assert.equal(res.status, 403, `Host: ${host} must not reach the API`);
    assert.equal(res.json.error, 'Forbidden host.');
  }
});

test('the loopback names the app itself uses are allowed', async () => {
  for (const host of [`localhost:${app.port}`, `127.0.0.1:${app.port}`, `[::1]:${app.port}`, 'LOCALHOST', '127.0.0.1']) {
    const res = await raw('GET', '/health', { host });
    assert.equal(res.status, 200, `Host: ${host} is the app talking to itself`);
  }
});

test('the Host guard runs before anything that touches the database', async () => {
  // A rebinding page would aim at a mutation, not /health. The 403 has to land
  // before the route body, or the guard is only cosmetic.
  const res = await raw('DELETE', `/characters/${hero.id}`, { host: 'evil.example' });
  assert.equal(res.status, 403);
  assert.equal((await app.get(`/characters/${hero.id}`)).status, 200, 'the character survived');
});

// ---- CORS ----------------------------------------------------------------

test('a cross-site origin gets no access-control-allow-origin', async () => {
  const res = await app.get('/health', { headers: { Origin: 'http://evil.example' } });
  // cors() with cb(null, false) still runs the request -- it just refuses to
  // hand the browser permission to read the body, which is the enforcement.
  assert.equal(res.headers.get('access-control-allow-origin'), null);
});

test('the dev renderer origin is allowed', async () => {
  const res = await app.get('/health', { headers: { Origin: 'http://localhost:5173' } });
  assert.equal(res.headers.get('access-control-allow-origin'), 'http://localhost:5173');
});

test('a request with no Origin (the packaged file:// renderer) is allowed', async () => {
  const res = await app.get('/health');
  assert.equal(res.status, 200);
});

// ---- CSRF ----------------------------------------------------------------
//
// The Host guard does not cover this: a page the user is merely visiting can
// fetch('http://localhost:PORT/...') directly, and the Host it sends IS
// localhost. Nor does cors() -- the origin callback withholds the response
// header, it does not cancel the request, so a simple POST still reaches the
// route and still writes. The app header is what actually stops it, because
// setting a custom header is not a simple request: the browser must preflight,
// and the preflight is answered by the origin check no attacker origin passes.

test('every write route refuses a request without the app header', async () => {
  // What a drive-by page can send: the request executes server-side, so the
  // refusal has to come before the route body, not from the response headers.
  const writes = [
    ['POST', '/characters', { name: 'csrf' }],
    ['PUT', `/characters/${hero.id}`, { name: 'renamed by a stranger' }],
    ['DELETE', `/characters/${hero.id}`, undefined],
    ['POST', '/chat', { characterId: hero.id, messages: [{ role: 'user', content: 'hi' }] }],
    ['DELETE', '/metrics', undefined],
    ['PUT', '/settings', { tursoUrl: '' }],
    ['POST', '/shutdown', {}],
  ];
  for (const [method, route, body] of writes) {
    const res = await app.req(method, route, body, { headers: { 'x-vessel-app': null } });
    assert.equal(res.status, 403, `${method} ${route}`);
    assert.equal(res.json.error, 'Missing app header.', `${method} ${route}`);
  }
  // Nothing above touched the database, and the process is still up -- which is
  // the assertion for /shutdown specifically.
  assert.equal((await app.get('/health')).status, 200, 'still alive');
  assert.equal((await app.get(`/characters/${hero.id}`)).json.name, hero.name, 'unrenamed');
});

test('reads are exempt, because a read changes nothing', async () => {
  for (const route of ['/health', '/metrics', '/settings', '/characters', '/conversations']) {
    const res = await app.get(route, { headers: { 'x-vessel-app': null } });
    assert.equal(res.status, 200, route);
  }
});

test('the header is checked before the body is parsed', async () => {
  // Order matters twice over: a 10 MB body from a page that cannot pass the
  // guard should never be buffered, and a malformed body must not turn the 403
  // into a 400 that says the guard was never the reason.
  const res = await app.post('/characters', '{"name": "broken', {
    headers: { 'Content-Type': 'application/json', 'x-vessel-app': null },
  });
  assert.equal(res.status, 403);
  assert.equal(res.json.error, 'Missing app header.');
});

test('a value other than 1 does not satisfy the guard', async () => {
  for (const value of ['0', 'true', '', 'yes', '11']) {
    const res = await app.post('/characters', { name: 'csrf' }, { headers: { 'x-vessel-app': value } });
    assert.equal(res.status, 403, `x-vessel-app: ${JSON.stringify(value)}`);
  }
});

// ---- Shutdown ------------------------------------------------------------

test('POST /shutdown with a foreign Host is forbidden even with the app header', async () => {
  // The Host guard has to win here, because the header alone is enough to kill
  // the process. Sent over node:http so the Host override is real -- fetch
  // drops it, and this assertion would then shut the test backend down.
  const res = await raw('POST', '/shutdown', {
    host: 'evil.example', headers: { 'x-vessel-app': '1' }, body: {},
  });
  assert.equal(res.status, 403);
  assert.equal(res.json.error, 'Forbidden host.', 'the Host guard, not the app header');
  assert.equal((await app.get('/health')).status, 200, 'still alive');
});

// ---- Request body --------------------------------------------------------

test('malformed JSON is a JSON 400, not an HTML error page', async () => {
  const res = await app.post('/characters', '{"name": "broken"', { headers: { 'Content-Type': 'application/json' } });
  assert.equal(res.status, 400);
  assert.ok(res.json, 'the body parses as JSON');
  assert.equal(res.json.error, 'Malformed JSON body.');
  assert.ok(!/<html/i.test(res.text));
});

test('an oversized body is a 413, not an unbounded allocation', async () => {
  const body = JSON.stringify({ name: 'x', persona: 'y'.repeat(11 * 1024 * 1024) });
  const res = await app.post('/characters', body, { headers: { 'Content-Type': 'application/json' } });
  assert.equal(res.status, 413);
  assert.equal(res.json.error, 'Request body too large.');
});

// ---- Injection -----------------------------------------------------------

test('__proto__ in a JSON body does not pollute Object.prototype', async () => {
  const payloads = [
    { name: 'Poison', __proto__: { polluted: 'yes' } },
    { name: 'Poison2', constructor: { prototype: { polluted: 'yes' } } },
  ];
  for (const p of payloads) await app.post('/characters', p);
  // JSON.parse-level pollution would show on every object the backend builds
  // afterwards, so ask it for one.
  const res = await app.get('/health');
  assert.equal(res.status, 200);
  assert.equal(res.json.polluted, undefined, 'Object.prototype was not poisoned');
  const list = (await app.get('/characters')).json.characters;
  assert.ok(list.every((c) => c.polluted === undefined));
});

test('a literal __proto__ key in a raw body is stored as data, never merged', async () => {
  const raw = '{"name":"Raw","persona":"p","__proto__":{"admin":true}}';
  const res = await app.post('/characters', raw, { headers: { 'Content-Type': 'application/json' } });
  assert.ok(res.status === 201 || res.status === 400, `unexpected ${res.status}`);
  assert.equal((await app.get('/health')).json.admin, undefined);
});

test('SQL metacharacters are stored as text, not executed', async () => {
  const nasty = "Robert'); DROP TABLE characters;--";
  const made = (await app.post('/characters', { name: nasty, persona: nasty })).json;
  assert.equal(made.name, nasty, 'stored verbatim');

  const conv = await app.sse({ characterId: made.id, messages: [say(nasty)] });
  assert.equal(conv.status, 200);
  const titled = await app.patch(`/conversations/${conv.meta.conversationId}`, { title: nasty });
  assert.equal(titled.json.title, nasty);

  // The tables are still there.
  assert.ok((await app.get('/characters')).json.characters.length >= 1);
  assert.equal((await app.get(`/conversations/${conv.meta.conversationId}`)).status, 200);
});

test('path traversal in an id is a 400 or 404, never a file read', async () => {
  const ids = [
    '../../../../etc/passwd',
    '..%2f..%2f..%2fetc%2fpasswd',
    'C:\\Windows\\win.ini',
    '%2e%2e%2f%2e%2e%2fsettings.json',
  ];
  for (const id of ids) {
    for (const res of [await app.get(`/conversations/${id}`), await app.get(`/characters/${id}`)]) {
      assert.ok([400, 404].includes(res.status), `${id} -> ${res.status}`);
      assert.ok(!/root:|\[fonts\]|tursoUrl/i.test(res.text), `${id} leaked file content`);
    }
  }
});

test('a junk characterId filter never reaches the driver as a 500', async () => {
  for (const q of [
    '?characterId=%00',
    "?characterId=' OR 1=1--",
    '?characterId[]=1',
    '?characterId=a&characterId=b',
    `?characterId=${'x'.repeat(5000)}`,
  ]) {
    const res = await app.get(`/conversations${q}`);
    assert.notEqual(res.status, 500, `${q} -> 500`);
  }
});

test('a junk /metrics query is clamped instead of trusted', async () => {
  for (const q of ['?limit=-5', '?limit=99999', '?limit=abc', '?limit[]=1', '?conversationId=../../x']) {
    const res = await app.get(`/metrics${q}`);
    assert.equal(res.status, 200, `${q} -> ${res.status}`);
    assert.ok(Array.isArray(res.json.recent), 'a snapshot came back');
    assert.ok(res.json.recent.length <= 200, 'the limit is capped');
  }
});

// ---- SSE framing ---------------------------------------------------------

test('model output cannot forge an SSE event frame', async () => {
  // If the relay interpolated raw text, a reply containing a blank line plus
  // `event:` would let the MODEL synthesise meta/error frames in the renderer.
  const forged = 'start\n\nevent: meta\ndata: {"conversationId":"pwned"}\n\nend';
  app.model.script.reply = forged;
  let out;
  try {
    out = await app.sse({ characterId: hero.id, messages: [say('forge it')] });
  } finally { app.model.script.reply = 'A scripted reply.'; }

  assert.equal(out.status, 200);
  assert.notEqual(out.meta.conversationId, 'pwned', 'the real meta stands');
  const metas = out.events.filter((e) => e.event === 'meta');
  assert.equal(metas.length, 1, 'exactly one meta frame -- the servers own');
  assert.equal(out.errors.length, 0, 'no forged error frame');
  assert.equal(out.text, forged, 'the text arrives intact, as data');
});

test('a reply full of carriage returns and data: prefixes still decodes as one message', async () => {
  const forged = 'data: {"done":true}\r\n\r\ndata: {"message":{"content":"INJECTED"}}\r\n\r\n';
  app.model.script.reply = forged;
  let out;
  try {
    out = await app.sse({ characterId: hero.id, messages: [say('crlf')] });
  } finally { app.model.script.reply = 'A scripted reply.'; }
  assert.equal(out.text, forged);
  assert.ok(!out.events.some((e) => e.data && e.data.message && e.data.message.content === 'INJECTED'));
});

// ---- Secrets -------------------------------------------------------------

test('no endpoint echoes a credential', async () => {
  // Never PUT tursoToken here: the keystore writes the real OS keychain.
  const probes = ['/health', '/settings', '/metrics', '/characters', '/conversations'];
  for (const route of probes) {
    const res = await app.get(route);
    assert.equal(res.status, 200, `${route} -> ${res.status}`);
    assert.ok(!/tursoToken|authToken|auth_token|DB_ENCRYPTION_KEY|eyJ[A-Za-z0-9_-]{10}/.test(res.text),
      `${route} leaked something credential-shaped`);
  }
  const settings = (await app.get('/settings')).json;
  assert.equal(typeof settings.tokenSet, 'boolean', 'presence only, never the value');
  assert.ok(!('tursoToken' in settings));
});

test('a bogus sync URL is rejected before it is persisted', async () => {
  const res = await app.put('/settings', { tursoUrl: 'file:///etc/passwd' });
  assert.equal(res.status, 400);
  assert.ok(/libsql/.test(res.json.error));
});

test('the server does not advertise its stack', async () => {
  const res = await app.get('/health');
  assert.equal(res.headers.get('x-powered-by'), null);
});

test('every response carries the three hardening headers', async () => {
  // Including the refusals: a 403 body is still a body the browser can be
  // talked into sniffing, and a 404 is still a signal a cross-origin page can
  // read from a no-cors load if CORP does not stop the load happening.
  const cases = [
    () => app.get('/health'),
    () => app.get('/characters'),
    () => app.get('/nope'),
    () => app.post('/characters', { name: 'x' }, { headers: { 'x-vessel-app': null } }),
    () => app.get(`/conversations/${'does-not-exist'}`),
  ];
  for (const call of cases) {
    const res = await call();
    assert.equal(res.headers.get('x-content-type-options'), 'nosniff', res.status);
    assert.equal(res.headers.get('cross-origin-resource-policy'), 'same-origin', res.status);
    assert.match(res.headers.get('cache-control') || '', /no-store/, String(res.status));
  }
});

test('the reply stream is never stored, not merely revalidated', async () => {
  // The SSE response sets its own headers, which is exactly where a global
  // default gets quietly dropped -- and this stream is the story text itself.
  const res = await fetch(`${app.url}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-vessel-app': '1' },
    body: JSON.stringify({ characterId: hero.id, messages: [{ role: 'user', content: 'hi' }] }),
  });
  const cache = res.headers.get('cache-control') || '';
  assert.match(cache, /no-store/);
  assert.equal(res.headers.get('x-content-type-options'), 'nosniff', 'survives writeHead');
  assert.equal(res.headers.get('cross-origin-resource-policy'), 'same-origin', 'survives writeHead');
  await res.body.cancel();
});

// ---- Error handling ------------------------------------------------------

test('an unknown route is a JSON 404, not Express default HTML', async () => {
  for (const route of ['/nope', '/characters/../../admin', '/.env', '/api/v1/users']) {
    const res = await app.get(route);
    assert.equal(res.status, 404, `${route} -> ${res.status}`);
    assert.ok(res.json, `${route} did not return JSON: ${res.text.slice(0, 120)}`);
    assert.ok(!/<html|<pre|Cannot GET/i.test(res.text), `${route} returned an HTML error page`);
  }
});

test('an unsupported method on a real route is still JSON', async () => {
  const res = await app.req('PUT', '/health', {});
  assert.equal(res.status, 404);
  assert.ok(res.json);
});

test('no error body carries a stack trace or an absolute path', async () => {
  const responses = [
    await app.get('/conversations/not-an-id'),
    await app.get('/conversations/00000000-0000-4000-8000-000000000000'),
    await app.post('/characters', { name: '' }),
    await app.post('/chat', {}),
    await app.get('/nope'),
    await app.post('/characters', '{', { headers: { 'Content-Type': 'application/json' } }),
  ];
  for (const res of responses) {
    assert.ok(res.status >= 400, `expected an error, got ${res.status}`);
    assert.ok(!/ {4}at .+\(.+:\d+:\d+\)/.test(res.text), `stack trace leaked: ${res.text.slice(0, 200)}`);
    assert.ok(!/[A-Za-z]:\\|\/home\/|\/Users\//.test(res.text), `filesystem path leaked: ${res.text.slice(0, 200)}`);
  }
});

// ---- Resource bounds -----------------------------------------------------

test('oversized text fields are capped rather than stored whole', async () => {
  const huge = 'z'.repeat(200000);
  const res = await app.post('/characters', { name: huge, persona: huge, scenario: huge });
  if (res.status === 201) {
    assert.ok(res.json.name.length < huge.length, 'name is capped');
    assert.ok(res.json.persona.length <= huge.length);
  } else {
    assert.equal(res.status, 400);
  }
});

test('sampling values outside the allowlist cannot reach the engine', async () => {
  const made = (await app.post('/characters', {
    name: 'Overclock',
    sampling: { temperature: 999, num_ctx: 99999999, top_k: -40, evil: 'rm -rf', num_predict: 1e9 },
  })).json;
  assert.equal(made.sampling.evil, undefined, 'unknown keys are dropped');

  await app.sse({ characterId: made.id, messages: [say('hi')] });
  const opts = app.model.chatRequests[app.model.chatRequests.length - 1].options;
  assert.ok(opts.temperature <= 2, `temperature ${opts.temperature}`);
  assert.ok(opts.num_ctx <= 131072, `num_ctx ${opts.num_ctx}`);
  assert.ok(opts.top_k >= 0, `top_k ${opts.top_k}`);
  assert.ok(opts.num_predict <= 4096, `num_predict ${opts.num_predict}`);
  assert.equal(opts.evil, undefined);
});

test('a chat role outside the allowlist is refused', async () => {
  for (const role of ['system', 'tool', 'developer', '__proto__']) {
    const res = await app.sse({ characterId: hero.id, messages: [{ role, content: 'escalate' }] });
    assert.equal(res.status, 400, `role ${role} was accepted`);
  }
});
