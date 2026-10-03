/**
 * The renderer's data seam, after the HTTP transport was removed on
 * 2026-10-03.
 *
 * `api.js` decides its transport ONCE, at module load, from
 * `window.__TAURI__`. That makes load order part of its behaviour, so every
 * scenario below imports a fresh copy with a cache-busting query rather than
 * reusing one -- a single shared import would test only whichever global
 * happened to be set first.
 *
 * What is worth asserting here is the half that has no second chance: there is
 * no HTTP fallback any more, so a missing bridge used to silently take the
 * other code path and now has to fail in a way somebody can act on. Nothing
 * here opens a database, a socket or a keychain.
 */

'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { test } = require('node:test');

const API = pathToFileURL(
  path.resolve(__dirname, '..', '..', 'app', 'src', 'renderer', 'src', 'lib', 'api.js')
).href;

let seq = 0;
// A fresh module instance per scenario: the transport is captured at load.
function loadApi() {
  return import(`${API}?n=${seq++}`);
}

// Minimal stand-in for what the Tauri host injects. `calls` records every
// invoke so a test can assert on the command name and its arguments.
function fakeTauri(opts = {}) {
  const { resolve } = opts;
  // Keyed on presence, not truthiness: `{}` and `null` are both rejections the
  // real core can send, and both are falsy.
  const rejects = 'reject' in opts;
  const calls = [];
  const host = {
    core: {
      invoke(cmd, args) {
        calls.push({ cmd, args });
        if (rejects) return Promise.reject(opts.reject);
        return Promise.resolve(typeof resolve === 'function' ? resolve(cmd, args) : resolve);
      },
      Channel: class {
        constructor() {
          this.onmessage = null;
        }
      },
    },
  };
  return { host, calls };
}

function withHost(host) {
  globalThis.window = host ? { __TAURI__: host } : {};
}

test.afterEach(() => {
  delete globalThis.window;
});

test('without the IPC bridge every api call rejects, rather than reaching for a transport that is gone', async () => {
  withHost(null);
  const { api } = await loadApi();

  // Every method, not just one: the old file had two objects and it was the
  // shape of the export that decided which you got.
  const methods = Object.keys(api);
  assert.ok(methods.length >= 12, `expected the whole surface, saw ${methods.length}`);

  for (const name of methods) {
    let p;
    assert.doesNotThrow(() => { p = api[name]('x', 'y', 'z'); }, `${name} must not throw synchronously`);
    await assert.rejects(
      () => p,
      /application window|IPC bridge/i,
      `${name} should refuse without a host`
    );
  }
});

test('the refusal rejects rather than throwing, and names the cause', async () => {
  withHost(null);
  const { api } = await loadApi();

  // A synchronous throw would escape the caller entirely: the components call
  // this from React handlers with a .catch and no try/catch, so a throw lands
  // on window.onerror and the UI never shows the reason.
  let p;
  assert.doesNotThrow(() => { p = api.health(); });
  assert.ok(p instanceof Promise);

  await assert.rejects(() => p, (err) => {
    assert.ok(err instanceof Error);
    // The failure mode this replaced: reading .core of undefined.
    assert.doesNotMatch(err.message, /undefined|null|Cannot read/i);
    assert.match(err.message, /application window/);
    return true;
  });
});

test('streamChat without a bridge reports through onError and still returns an abort function', async () => {
  withHost(null);
  const { streamChat } = await loadApi();

  const errors = [];
  let done = 0;
  // Handlers are wired AFTER the call returns, which is why the report has to
  // be asynchronous -- a synchronous onError would fire into nothing.
  const abort = streamChat({ messages: [] }, {
    onError: (m) => errors.push(m),
    onDone: () => { done++; },
  });

  assert.equal(typeof abort, 'function', 'callers always call abort() on unmount');
  assert.deepEqual(errors, [], 'must not have reported before the caller returned');

  await new Promise((r) => setImmediate(r));

  assert.equal(errors.length, 1, 'exactly one report');
  assert.match(errors[0], /application window|IPC bridge/i);
  assert.equal(done, 0, 'a stream that never started is not done');
  assert.doesNotThrow(() => abort(), 'abort on a dead stream is a no-op');
});

test('streamChat without a bridge does not throw when no handlers are passed at all', async () => {
  withHost(null);
  const { streamChat } = await loadApi();

  // Chat.jsx always passes handlers, but the signature defaults to {} and an
  // unhandled rejection here would take down the renderer.
  let abort;
  assert.doesNotThrow(() => { abort = streamChat({ messages: [] }); });
  await new Promise((r) => setImmediate(r));
  assert.doesNotThrow(() => abort());
});

test('with the bridge present every call goes over IPC under its documented command name', async () => {
  const { host, calls } = fakeTauri({
    resolve: (cmd) => {
      if (cmd === 'list_characters') return { characters: [{ id: 'c1' }] };
      if (cmd === 'list_conversations') return { conversations: [{ id: 'v1' }] };
      return { ok: true };
    },
  });
  withHost(host);
  const { api } = await loadApi();

  assert.deepEqual(await api.listCharacters(), [{ id: 'c1' }], 'unwraps .characters');
  assert.deepEqual(await api.listConversations(), [{ id: 'v1' }], 'unwraps .conversations');
  await api.health();
  await api.deleteCharacter('c9');

  const names = calls.map((c) => c.cmd);
  assert.deepEqual(names, ['list_characters', 'list_conversations', 'health', 'delete_character']);
  // snake_case is the Rust command name; a camelCase slip would fail at runtime only.
  for (const n of names) assert.doesNotMatch(n, /[A-Z]/, `${n} must be snake_case`);
});

test('listConversations sends an explicit null rather than an absent field', async () => {
  const { host, calls } = fakeTauri({ resolve: { conversations: [] } });
  withHost(host);
  const { api } = await loadApi();

  await api.listConversations();
  // The command signature takes Option<String>; omitting the key entirely
  // deserializes differently from a null on the Rust side.
  assert.ok('characterId' in calls[0].args, 'the key must be present');
  assert.equal(calls[0].args.characterId, null);

  await api.listConversations('abc');
  assert.equal(calls[1].args.characterId, 'abc');
});

test('saveSettings carries confirmClearToken across the bridge', async () => {
  const { host, calls } = fakeTauri({ resolve: {} });
  withHost(host);
  const { api } = await loadApi();

  // Dropping this field is what made a confirmed "turn off sync" fail instead
  // of apply: save_settings refuses to clear a stored token without it.
  await api.saveSettings({ tursoUrl: '', tursoToken: null, confirmClearToken: true });
  assert.equal(calls[0].args.confirmClearToken, true);
  assert.ok('tursoToken' in calls[0].args);
});

test('a rejected command is flattened to one Error whatever shape it arrived in', async () => {
  for (const [rejection, expected] of [
    [{ error: 'Ollama is not running.' }, /Ollama is not running/],
    [{ detail: 'disk full' }, /disk full/],
    [new Error('already an Error'), /already an Error/],
    ['a bare string', /a bare string/],
    [{}, /Request failed/],
    [null, /Request failed/],
  ]) {
    const { host } = fakeTauri({ reject: rejection });
    withHost(host);
    const { api } = await loadApi();

    await assert.rejects(() => api.health(), (err) => {
      assert.ok(err instanceof Error, `${JSON.stringify(rejection)} must flatten to an Error`);
      assert.match(err.message, expected);
      return true;
    });
  }
});

test('the module exposes no HTTP transport any more', async () => {
  const { host } = fakeTauri({ resolve: {} });
  withHost(host);
  const mod = await loadApi();

  // The export surface is the contract App.jsx and Chat.jsx import.
  assert.deepEqual(Object.keys(mod).sort(), ['api', 'streamChat']);
  assert.equal(typeof mod.streamChat, 'function');
  // relaunch used to be null under HTTP when the shell had not injected it;
  // over IPC it is always a real command.
  assert.equal(typeof mod.api.relaunch, 'function');
});

test('a streamed turn delivers meta, chunks and done in order and accumulates the text', async () => {
  const { host, calls } = fakeTauri({ resolve: {} });
  withHost(host);
  const { streamChat } = await loadApi();

  const seen = [];
  let full = null;
  streamChat({ messages: [{ role: 'user', content: 'hi' }] }, {
    onMeta: (m) => seen.push(['meta', m.conversationId]),
    onToken: (delta, acc) => seen.push(['token', delta, acc]),
    onDone: (text) => { full = text; },
    onError: (e) => seen.push(['error', e]),
  });

  const channel = calls[0].args.onEvent;
  channel.onmessage({ type: 'meta', conversationId: 'v1', characterId: 'c1', recalled: [] });
  channel.onmessage({ type: 'chunk', delta: 'Hel' });
  channel.onmessage({ type: 'chunk', delta: 'lo' });
  channel.onmessage({ type: 'done' });

  assert.deepEqual(seen, [
    ['meta', 'v1'],
    ['token', 'Hel', 'Hel'],
    ['token', 'lo', 'Hello'],
  ]);
  assert.equal(full, 'Hello', 'onDone receives the whole reply, not the last delta');
});

test('an unknown event type is ignored rather than treated as a chunk', async () => {
  const { host, calls } = fakeTauri({ resolve: {} });
  withHost(host);
  const { streamChat } = await loadApi();

  let tokens = 0;
  let done = null;
  streamChat({ messages: [] }, { onToken: () => { tokens++; }, onDone: (t) => { done = t; } });

  const channel = calls[0].args.onEvent;
  // A newer core adding an event must not corrupt the reply text.
  channel.onmessage({ type: 'heartbeat' });
  channel.onmessage({ type: 'chunk', delta: 'x' });
  channel.onmessage({ type: 'done' });

  assert.equal(tokens, 1);
  assert.equal(done, 'x');
});

test('a stop issued before the first event still cancels, once the id arrives', async () => {
  const { host, calls } = fakeTauri({ resolve: {} });
  withHost(host);
  const { streamChat } = await loadApi();

  // The real case: the model is still loading, the user hits stop, and no
  // conversation id has come back yet. Dropping it would leave the turn running.
  const abort = streamChat({ messages: [] }, {});
  abort();

  assert.deepEqual(calls.map((c) => c.cmd), ['chat'], 'nothing to cancel yet');

  calls[0].args.onEvent.onmessage({ type: 'meta', conversationId: 'v7' });

  const cancels = calls.filter((c) => c.cmd === 'cancel_chat');
  assert.equal(cancels.length, 1, 'the deferred stop fires on meta');
  assert.equal(cancels[0].args.conversationId, 'v7');
});

test('a stop on a turn that already has an id cancels immediately', async () => {
  const { host, calls } = fakeTauri({ resolve: {} });
  withHost(host);
  const { streamChat } = await loadApi();

  const abort = streamChat({ conversationId: 'v3', messages: [] }, {});
  abort();

  const cancels = calls.filter((c) => c.cmd === 'cancel_chat');
  assert.equal(cancels.length, 1);
  assert.equal(cancels[0].args.conversationId, 'v3');
});

test('a chat command that rejects before streaming reports once, and not after a done', async () => {
  const { host } = fakeTauri({ reject: { error: 'model missing' } });
  withHost(host);
  const { streamChat } = await loadApi();

  const errors = [];
  streamChat({ messages: [] }, { onError: (m) => errors.push(m) });
  await new Promise((r) => setImmediate(r));

  assert.deepEqual(errors, ['model missing'], 'the rejection is the only report the caller gets');
});

test('a rejection after a completed stream is swallowed, because done already reported', async () => {
  let rejectChat;
  const calls = [];
  const host = {
    core: {
      invoke(cmd, args) {
        calls.push({ cmd, args });
        if (cmd === 'chat') return new Promise((_, rej) => { rejectChat = rej; });
        return Promise.resolve({});
      },
      Channel: class { constructor() { this.onmessage = null; } },
    },
  };
  withHost(host);
  const { streamChat } = await loadApi();

  const errors = [];
  let done = 0;
  streamChat({ messages: [] }, { onError: (m) => errors.push(m), onDone: () => { done++; } });

  calls[0].args.onEvent.onmessage({ type: 'done' });
  rejectChat({ error: 'late failure' });
  await new Promise((r) => setImmediate(r));

  assert.equal(done, 1);
  assert.deepEqual(errors, [], 'a finished turn must not also report an error');
});

test('a cancel that itself fails does not surface to the user', async () => {
  const calls = [];
  const host = {
    core: {
      invoke(cmd, args) {
        calls.push({ cmd, args });
        // The turn is already over on the Rust side; cancelling it is moot.
        if (cmd === 'cancel_chat') return Promise.reject({ error: 'no such turn' });
        return Promise.resolve({});
      },
      Channel: class { constructor() { this.onmessage = null; } },
    },
  };
  withHost(host);
  const { streamChat } = await loadApi();

  const errors = [];
  const abort = streamChat({ conversationId: 'v1', messages: [] }, {
    onError: (m) => errors.push(m),
  });
  abort();
  await new Promise((r) => setImmediate(r));

  assert.deepEqual(errors, [], 'a failed cancel is not something the user can act on');
});

test('an error event mid-stream is reported and does not end the turn as done', async () => {
  const { host, calls } = fakeTauri({ resolve: {} });
  withHost(host);
  const { streamChat } = await loadApi();

  const errors = [];
  let done = 0;
  streamChat({ messages: [] }, { onError: (m) => errors.push(m), onDone: () => { done++; } });

  const channel = calls[0].args.onEvent;
  channel.onmessage({ type: 'chunk', delta: 'partial' });
  channel.onmessage({ type: 'error', error: 'engine died' });

  assert.deepEqual(errors, ['engine died']);
  assert.equal(done, 0);
});
