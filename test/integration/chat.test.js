'use strict';

/**
 * POST /chat -- the SSE path, end to end.
 *
 * Two things are being checked here and they are not the same thing. The first
 * is the wire: status codes, the meta event, the chunk relay, what survives a
 * stop. The second, and the more valuable one, is WHAT THE MODEL WAS SENT --
 * the fake model records every request body, so the prompt order that the whole
 * Stage 2 reorder rests on can be asserted directly instead of inferred from a
 * reuse percentage:
 *
 *   persona | summary | verbatim turns | RECALL | director | new user
 *   ------------ append-only ---------/  ---- volatile ----/
 *
 * Anything volatile that drifts up into the prefix silently costs a full
 * re-prefill on every turn, and nothing about the app breaks when it does. That
 * makes it exactly the kind of regression a test has to catch.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const backend = require(path.resolve(__dirname, '..', 'helpers', 'backend.js'));

const DEFAULT_REPLY = 'The lamp gutters. "You came back," she says.';

let app;
let hero;

test.before(async () => {
  app = await backend.start({ script: { reply: DEFAULT_REPLY } });
  hero = (await app.post('/characters', {
    name: 'Mira',
    persona: 'A lighthouse keeper who speaks in short sentences.',
    responseStyle: 'dialogue',
  })).json;
});
test.after(async () => { if (app) await app.stop(); });

// The body of the last /api/chat the backend made.
const lastPrompt = () => app.model.chatRequests[app.model.chatRequests.length - 1];
const roles = (msgs) => msgs.map((m) => m.role);
const say = (content) => ({ role: 'user', content });

test('a chat streams meta, then tokens, then a done chunk', async () => {
  const out = await app.sse({ characterId: hero.id, messages: [say('Hello?')] });
  assert.equal(out.status, 200);
  assert.equal(out.stream, true);
  assert.ok(out.meta, 'a meta event must arrive before any token');
  assert.match(out.meta.conversationId, /^[0-9a-f-]{36}$/);
  assert.equal(out.meta.characterId, hero.id);
  assert.deepEqual(out.meta.recalled, [], 'nothing to recall on the first turn');
  assert.equal(out.text, DEFAULT_REPLY, 'every chunk is relayed, in order');
  assert.deepEqual(out.errors, []);

  assert.equal(out.events[0].event, 'meta', 'meta is the first frame on the wire');
  const done = out.events[out.events.length - 1].data;
  assert.equal(done.done, true, 'the final chunk carries the telemetry');
  assert.equal(done.prompt_eval_count, app.model.script.promptEvalCount);
});

test('the persona leads the prompt and the new user message ends it', async () => {
  await app.sse({ characterId: hero.id, messages: [say('Who are you?')] });
  const { messages } = lastPrompt();
  assert.equal(messages[0].role, 'system');
  assert.match(messages[0].content, /roleplaying as the character "Mira"/);
  assert.match(messages[0].content, /lighthouse keeper/, 'the character persona is carried');
  assert.match(messages[0].content, /ALWAYS give the character spoken dialogue/, 'responseStyle applies');
  assert.deepEqual(messages[messages.length - 1], { role: 'user', content: 'Who are you?' });
});

test('the reply is persisted and replayed as verbatim history on the next turn', async () => {
  const first = await app.sse({ characterId: hero.id, messages: [say('First message.')] });
  const convId = first.meta.conversationId;
  await app.sse({ conversationId: convId, messages: [say('Second message.')] });

  const { messages } = lastPrompt();
  assert.deepEqual(roles(messages), ['system', 'user', 'assistant', 'user']);
  assert.equal(messages[1].content, 'First message.');
  assert.equal(messages[2].content, DEFAULT_REPLY);
  assert.equal(messages[3].content, 'Second message.');

  const conv = (await app.get(`/conversations/${convId}`)).json;
  assert.equal(conv.verbatim.length, 4);
  assert.equal(conv.characterId, hero.id, 'the character stays bound without re-sending characterId');
});

test('the director note sits in the volatile tail, not in the cacheable prefix', async () => {
  const first = await app.sse({ characterId: hero.id, messages: [say('Tell me a story.')] });
  const convId = first.meta.conversationId;
  await app.sse({
    conversationId: convId,
    director: 'Make her colder.',
    messages: [say('And then?')],
  });

  const { messages } = lastPrompt();
  const last = messages[messages.length - 1];
  const beforeLast = messages[messages.length - 2];
  assert.equal(last.role, 'user');
  assert.equal(last.content, 'And then?');
  assert.equal(beforeLast.role, 'system');
  assert.match(beforeLast.content, /Director note/);
  assert.match(beforeLast.content, /Make her colder\./);
  // The director must never be the thing that shifts the stable prefix.
  assert.equal(messages.findIndex((m) => /Director note/.test(m.content)), messages.length - 2);
});

test('a director note is never recorded as story', async () => {
  const first = await app.sse({ characterId: hero.id, messages: [say('Hello.')] });
  const convId = first.meta.conversationId;
  await app.sse({ conversationId: convId, director: 'Be brief.', messages: [say('Go on.')] });
  const conv = (await app.get(`/conversations/${convId}`)).json;
  assert.equal(conv.verbatim.length, 4);
  assert.ok(!conv.verbatim.some((t) => /Be brief/.test(t.content)), 'the note is steering, not narrative');
});

test('a director-only request continues the scene without inventing a user turn', async () => {
  const first = await app.sse({ characterId: hero.id, messages: [say('Say something.')] });
  const convId = first.meta.conversationId;

  const out = await app.sse({ conversationId: convId, director: 'Describe the weather.' });
  assert.equal(out.status, 200);
  assert.equal(out.text, DEFAULT_REPLY);

  const conv = (await app.get(`/conversations/${convId}`)).json;
  assert.deepEqual(conv.verbatim.map((t) => t.role), ['user', 'assistant', 'assistant'],
    'a second reply is appended with no user turn between');
  assert.ok(!conv.verbatim.some((t) => /Describe the weather/.test(t.content)));
});

test('regenerate appends a variant instead of a new turn, and hides the old reply', async () => {
  const first = await app.sse({ characterId: hero.id, messages: [say('Roll once.')] });
  const convId = first.meta.conversationId;

  app.model.script.reply = 'A different take entirely.';
  let again;
  try {
    again = await app.sse({ conversationId: convId, regenerate: true });
  } finally { app.model.script.reply = DEFAULT_REPLY; }
  assert.equal(again.status, 200);
  assert.equal(again.text, 'A different take entirely.');

  // The model must not see its own previous reply when re-rolling, and the
  // prompting user message must not be duplicated.
  const { messages } = lastPrompt();
  assert.deepEqual(roles(messages), ['system', 'user']);
  assert.equal(messages[1].content, 'Roll once.');

  const conv = (await app.get(`/conversations/${convId}`)).json;
  assert.equal(conv.verbatim.length, 2, 'still one user turn and one assistant turn');
  const turn = conv.verbatim[1];
  assert.equal(turn.variants.length, 2);
  assert.equal(turn.activeIndex, 1, 'the fresh roll becomes the active variant');
  assert.equal(turn.content, 'A different take entirely.');
});

test('regenerating a conversation with no reply is a 400, not a crash', async () => {
  const res = await app.post('/chat', { characterId: hero.id, regenerate: true });
  assert.equal(res.status, 400);
  assert.match(res.json.error, /Nothing to regenerate/);
});

test('a request with nothing usable in it is rejected', async () => {
  assert.equal((await app.post('/chat', {})).status, 400);
  assert.equal((await app.post('/chat', { messages: [] })).status, 400);
  assert.equal((await app.post('/chat', { messages: 'hello' })).status, 400);
  const junk = [
    { role: 'user', content: '   ' },
    { role: 'user', content: 42 },
    { role: 'ghost', content: 'boo' },
    null,
  ];
  for (const m of junk) {
    const res = await app.post('/chat', { characterId: hero.id, messages: [m] });
    assert.equal(res.status, 400, `expected 400 for ${JSON.stringify(m)}`);
  }
});

test('junk messages cannot slip a second assistant turn into an existing conversation', async () => {
  // The regression this guards: a whitespace-only message used to be dropped,
  // leaving the stored history non-empty, so the model replied to the PREVIOUS
  // turn and a second assistant turn landed with no user turn between them --
  // breaking the alternation the summarizer depends on.
  const first = await app.sse({ characterId: hero.id, messages: [say('Real message.')] });
  const convId = first.meta.conversationId;
  const res = await app.post('/chat', { conversationId: convId, messages: [say('   ')] });
  assert.equal(res.status, 400);
  const conv = (await app.get(`/conversations/${convId}`)).json;
  assert.deepEqual(conv.verbatim.map((t) => t.role), ['user', 'assistant']);
});

test('a characterId that no longer exists is a 404, not an opaque foreign-key 500', async () => {
  const doomed = (await app.post('/characters', { name: 'Gone' })).json;
  await app.del(`/characters/${doomed.id}`);
  const res = await app.post('/chat', { characterId: doomed.id, messages: [say('hi')] });
  assert.equal(res.status, 404);
  assert.match(res.json.error, /Character not found/);
});

test('an unreachable engine is a 503 and leaves no empty conversation behind', async () => {
  const dead = await backend.start({ env: { OLLAMA_HOST: 'http://127.0.0.1:1' } });
  try {
    const res = await dead.post('/chat', { messages: [say('anyone there?')] });
    assert.equal(res.status, 503);
    assert.match(res.json.error, /Cannot reach/);
    assert.equal((await dead.get('/conversations')).json.conversations.length, 0,
      'the row ensureConversation pre-created must be cleaned up');
  } finally { await dead.stop(); }
});

test('an engine that refuses the request surfaces as 502, and a missing model as 404', async () => {
  const before = (await app.get('/conversations')).json.conversations.length;
  app.model.script.chatStatus = 500;
  try {
    const res = await app.post('/chat', { characterId: hero.id, messages: [say('hi')] });
    assert.equal(res.status, 502);
  } finally { app.model.script.chatStatus = 0; }

  app.model.script.chatStatus = 404;
  try {
    const res = await app.post('/chat', { characterId: hero.id, messages: [say('hi')] });
    assert.equal(res.status, 404, 'a missing model is a caller problem, not a bad gateway');
  } finally { app.model.script.chatStatus = 0; }

  assert.equal((await app.get('/conversations')).json.conversations.length, before,
    'neither refusal may leave an empty conversation');
});

test('an in-band engine error becomes an SSE error event and records nothing', async () => {
  const before = (await app.get('/conversations')).json.conversations.length;
  app.model.script.chatError = 'model runner crashed';
  let convId;
  try {
    const out = await app.sse({ characterId: hero.id, messages: [say('hi')] });
    // Headers are already sent by then, so the failure has to travel in-band.
    assert.equal(out.status, 200);
    assert.equal(out.text, '');
    assert.equal(out.errors.length, 1);
    assert.match(out.errors[0], /model runner crashed/);
    convId = out.meta.conversationId;
  } finally { app.model.script.chatError = null; }

  assert.equal((await app.get(`/conversations/${convId}`)).status, 404,
    'nothing was generated, so nothing is kept');
  assert.equal((await app.get('/conversations')).json.conversations.length, before);
});

test('stopping mid-stream keeps the partial reply rather than losing the turn', async () => {
  const long = 'one two three four five six seven eight nine ten';
  app.model.script.reply = long;
  app.model.script.chunkDelayMs = 15;
  const ac = new AbortController();
  let convId = null;
  try {
    const out = await app.sse(
      { characterId: hero.id, messages: [say('Count for me.')] },
      {
        signal: ac.signal,
        onChunk: (o) => {
          if (!convId && o.meta) convId = o.meta.conversationId;
          if (o.text.split(' ').length >= 3) ac.abort();
        },
      },
    );
    assert.equal(out.aborted, true);
  } catch (err) {
    if (err.name !== 'AbortError') throw err;
  } finally {
    app.model.script.chunkDelayMs = 0;
    app.model.script.reply = DEFAULT_REPLY;
  }

  assert.ok(convId, 'meta arrived before the abort');
  // The backend records after the stream closes; give it a moment to land.
  let conv = null;
  for (let i = 0; i < 40; i++) {
    conv = (await app.get(`/conversations/${convId}`)).json;
    if (conv && conv.verbatim.length >= 2) break;
    await new Promise((r) => setTimeout(r, 50));
  }
  assert.equal(conv.verbatim[0].content, 'Count for me.', 'the user message survives a stop');
  assert.equal(conv.verbatim[1].role, 'assistant');
  assert.ok(conv.verbatim[1].content.length > 0, 'so does whatever the model had written');
  assert.ok(conv.verbatim[1].content.length < long.length, 'and it is genuinely partial');
});

test('the per-character sampling overrides reach the engine, bounded', async () => {
  const tuned = (await app.post('/characters', {
    name: 'Tuned',
    sampling: { temperature: 0.4, num_predict: 99999 },
  })).json;
  await app.sse({ characterId: tuned.id, messages: [say('hi')] });
  const { options, model } = lastPrompt();
  assert.equal(model, 'test-model');
  assert.equal(options.temperature, 0.4);
  assert.equal(options.num_predict, 4096, 'clamped by the sampling allowlist, not by the caller');
  assert.ok(Number.isInteger(options.num_ctx), 'ollama takes the window per request');
});

test('a chat without a character still works and carries no persona', async () => {
  const out = await app.sse({ messages: [say('Just talk to me.')] });
  assert.equal(out.status, 200);
  assert.equal(out.meta.characterId, null);
  const { messages } = lastPrompt();
  assert.deepEqual(roles(messages), ['user'], 'no character means no system message at all');
});

test('every generation is measured', async () => {
  await app.del('/metrics');
  const out = await app.sse({ characterId: hero.id, messages: [say('Measure me.')] });
  const snap = (await app.get(`/metrics?conversationId=${out.meta.conversationId}`)).json;
  assert.equal(snap.summary.samples, 1);
  const rec = snap.recent[0];
  assert.equal(rec.promptTokens, app.model.script.promptEvalCount);
  assert.equal(rec.characterId, hero.id);
  assert.equal(rec.backend, 'ollama');
  assert.ok(rec.window.promptChars > 0);
  assert.equal(rec.window.messageCount, 2);
});
