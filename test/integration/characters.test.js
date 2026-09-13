'use strict';

/**
 * Character CRUD, against the real backend and a real (scratch) database.
 *
 * Most of what matters here is the sanitizers. A character is not just data --
 * its persona becomes a system message the model obeys, its avatar becomes an
 * <img src> in the renderer, and its sampling becomes options on the inference
 * request. Each of those is a place where an imported or hostile value reaches
 * something that acts on it, so each one is tested at the boundary rather than
 * trusted to the UI that normally produces it.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const backend = require(path.resolve(__dirname, '..', 'helpers', 'backend.js'));

let app;
test.before(async () => { app = await backend.start(); });
test.after(async () => { if (app) await app.stop(); });

const create = (body) => app.post('/characters', body);

test('a character round-trips through create and read', async () => {
  const res = await create({
    name: 'Ada',
    persona: 'A precise, dry engineer.',
    greeting: 'Well?',
    tagline: 'Builds things',
    about: 'Long form description.',
    chatStarters: ['Hello', 'What are you building?'],
    tags: ['engineer', 'dry'],
    responseStyle: 'dialogue',
    sampling: { temperature: 0.8 },
  });
  assert.equal(res.status, 201);
  const c = res.json;
  assert.match(c.id, /^[0-9a-f-]{36}$/, 'ids are minted server-side as UUIDs');
  assert.equal(c.name, 'Ada');
  assert.deepEqual(c.chatStarters, ['Hello', 'What are you building?']);
  assert.deepEqual(c.tags, ['engineer', 'dry']);
  assert.equal(c.responseStyle, 'dialogue');
  assert.deepEqual(c.sampling, { temperature: 0.8 });

  const read = await app.get(`/characters/${c.id}`);
  assert.equal(read.status, 200);
  assert.deepEqual(read.json, c);
});

test('a name is required, and whitespace is not a name', async () => {
  assert.equal((await create({})).status, 400);
  assert.equal((await create({ name: '' })).status, 400);
  assert.equal((await create({ name: '   \n\t ' })).status, 400);
  assert.equal((await create({ name: null })).status, 400);
});

test('a client-supplied id is ignored -- the server mints its own', async () => {
  const res = await create({ id: 'attacker-chosen-id', name: 'Spoof' });
  assert.equal(res.status, 201);
  assert.notEqual(res.json.id, 'attacker-chosen-id');
  assert.equal((await app.get('/characters/attacker-chosen-id')).status, 404);
});

test('text fields are capped', async () => {
  const res = await create({
    name: 'N'.repeat(500),
    tagline: 'T'.repeat(500),
    about: 'A'.repeat(20000),
    persona: 'P'.repeat(40000),
    greeting: 'G'.repeat(20000),
  });
  assert.equal(res.status, 201);
  assert.equal(res.json.name.length, 120);
  assert.equal(res.json.tagline.length, 200);
  assert.equal(res.json.about.length, 8000);
  assert.equal(res.json.persona.length, 16000);
  assert.equal(res.json.greeting.length, 8000);
});

test('only web image URLs survive as an avatar', async () => {
  const keep = [
    'https://example.com/a.png',
    'http://example.com/a.png',
    'data:image/png;base64,iVBORw0KGgo=',
    'DATA:IMAGE/SVG+XML,<svg/>',
  ];
  for (const avatar of keep) {
    const res = await create({ name: 'A', avatar });
    assert.equal(res.json.avatar, avatar.trim(), `expected to keep ${avatar}`);
  }
  // Each of these would make the renderer fetch or execute something local.
  const drop = [
    'javascript:alert(1)',
    'file:///C:/Windows/win.ini',
    'data:text/html,<script>alert(1)</script>',
    '//evil.example.com/a.png',
    'vbscript:msgbox(1)',
    '  javascript:alert(1)  ',
    'jAvAsCrIpT:alert(1)',
  ];
  for (const avatar of drop) {
    const res = await create({ name: 'A', avatar });
    assert.equal(res.json.avatar, '', `expected to reject ${avatar}`);
  }
});

test('sampling is an allowlist, and every value is clamped', async () => {
  const res = await create({
    name: 'Sampler',
    sampling: {
      temperature: 99,          // > max 2
      top_p: -1,                // < min 0
      top_k: 5000,              // > max 1000
      min_p: 0.3,               // in range
      repeat_penalty: 1.1,      // in range
      num_ctx: 1e9,             // would exhaust the GPU
      num_predict: 1,           // < min 16
      // Not on the list: none of these may reach the inference request.
      num_gpu: 0,
      seed: 1234,
      mirostat: 2,
      stop: ['</s>'],
      __proto__: { polluted: true },
    },
  });
  assert.deepEqual(res.json.sampling, {
    temperature: 2,
    top_p: 0,
    top_k: 1000,
    min_p: 0.3,
    repeat_penalty: 1.1,
    num_ctx: 131072,
    num_predict: 16,
  });
});

test('non-numeric sampling values are dropped rather than coerced to garbage', async () => {
  const res = await create({
    name: 'Sampler2',
    // NaN and Infinity arrive as null: JSON has no way to spell either.
    sampling: { temperature: 'hot', top_p: null, top_k: [], min_p: {}, repeat_penalty: NaN, num_ctx: false },
  });
  assert.deepEqual(res.json.sampling, {}, 'none of these are numbers, however Number() coerces them');
});

test('a numeric string is still a number', async () => {
  // The editor sliders round-trip through form values, so '0.8' is normal input.
  const res = await create({ name: 'Sampler3', sampling: { temperature: '0.8', top_k: '40' } });
  assert.deepEqual(res.json.sampling, { temperature: 0.8, top_k: 40 });
});

test('an explicit zero is kept -- it means greedy, not absent', async () => {
  const res = await create({ name: 'Sampler4', sampling: { temperature: 0 } });
  assert.deepEqual(res.json.sampling, { temperature: 0 });
});

test('a non-object sampling value yields an empty override set', async () => {
  for (const sampling of ['temperature=2', 42, true, ['temperature', 2]]) {
    const res = await create({ name: 'S', sampling });
    assert.deepEqual(res.json.sampling, {});
  }
});

test('an unknown response style falls back to balanced', async () => {
  assert.equal((await create({ name: 'A', responseStyle: 'unhinged' })).json.responseStyle, 'balanced');
  assert.equal((await create({ name: 'A', responseStyle: 42 })).json.responseStyle, 'balanced');
  assert.equal((await create({ name: 'A' })).json.responseStyle, 'balanced');
  for (const s of ['balanced', 'dialogue', 'narration-light']) {
    assert.equal((await create({ name: 'A', responseStyle: s })).json.responseStyle, s);
  }
});

test('lists are trimmed, emptied and bounded', async () => {
  const res = await create({
    name: 'Lists',
    chatStarters: ['  spaced  ', '', '   ', null, 'x'.repeat(500), ...Array.from({ length: 20 }, (_, i) => `s${i}`)],
    tags: ['  tag  ', 'y'.repeat(100), ...Array.from({ length: 20 }, (_, i) => `t${i}`)],
  });
  const { chatStarters, tags } = res.json;
  assert.equal(chatStarters.length, 12, 'at most 12 starters');
  assert.equal(chatStarters[0], 'spaced');
  assert.equal(chatStarters[1].length, 200, 'each starter capped at 200 chars');
  assert.equal(tags.length, 12);
  assert.equal(tags[0], 'tag');
  assert.equal(tags[1].length, 40, 'tags are capped shorter than starters');
});

test('a non-array list is an empty list, not a crash', async () => {
  const res = await create({ name: 'L', chatStarters: 'hello', tags: { a: 1 } });
  assert.deepEqual(res.json.chatStarters, []);
  assert.deepEqual(res.json.tags, []);
});

test('update merges: absent fields are left alone, null is treated as absent', async () => {
  const { json: c } = await create({ name: 'Before', persona: 'keep me', tags: ['a'] });
  const res = await app.put(`/characters/${c.id}`, { name: 'After' });
  assert.equal(res.status, 200);
  assert.equal(res.json.name, 'After');
  assert.equal(res.json.persona, 'keep me');
  assert.deepEqual(res.json.tags, ['a']);
  assert.notEqual(res.json.updatedAt, res.json.createdAt);

  const nulls = await app.put(`/characters/${c.id}`, { persona: null, tags: null });
  assert.equal(nulls.json.persona, 'keep me');
  assert.deepEqual(nulls.json.tags, ['a']);
});

test('update can clear a field with an empty string', async () => {
  const { json: c } = await create({ name: 'Clearable', persona: 'text' });
  const res = await app.put(`/characters/${c.id}`, { persona: '' });
  assert.equal(res.json.persona, '');
});

test('update cannot blank the name', async () => {
  const { json: c } = await create({ name: 'Named' });
  assert.equal((await app.put(`/characters/${c.id}`, { name: '   ' })).status, 400);
  assert.equal((await app.get(`/characters/${c.id}`)).json.name, 'Named');
});

test('unknown and malformed ids are 404, never 500', async () => {
  assert.equal((await app.get('/characters/00000000-0000-4000-8000-000000000000')).status, 404);
  assert.equal((await app.get('/characters/not a valid id')).status, 404);
  assert.equal((await app.put('/characters/00000000-0000-4000-8000-000000000000', { name: 'x' })).status, 404);
  assert.equal((await app.del('/characters/00000000-0000-4000-8000-000000000000')).status, 404);
});

test('SQL metacharacters in a name are stored as text', async () => {
  const nasty = "Robert'); DROP TABLE characters;--";
  const { json: c } = await create({ name: nasty });
  assert.equal(c.name, nasty);
  // The table is still there and still holds this row.
  assert.equal((await app.get(`/characters/${c.id}`)).json.name, nasty);
  assert.equal((await app.get('/characters')).status, 200);
});

test('delete removes the character and cascades to its conversations', async () => {
  const { json: c } = await create({ name: 'Doomed', persona: 'x' });
  const chat = await app.sse({ characterId: c.id, messages: [{ role: 'user', content: 'hello' }] });
  const convId = chat.meta.conversationId;
  assert.equal((await app.get(`/conversations/${convId}`)).status, 200);

  assert.equal((await app.del(`/characters/${c.id}`)).status, 200);
  assert.equal((await app.get(`/characters/${c.id}`)).status, 404);
  assert.equal(
    (await app.get(`/conversations/${convId}`)).status, 404,
    'the conversation must go with the character (FK cascade)',
  );
});

test('the list is ordered most-recently-updated first', async () => {
  const before = (await app.get('/characters')).json.characters.length;
  const { json: a } = await create({ name: 'Older' });
  await new Promise((r) => setTimeout(r, 5));
  const { json: b } = await create({ name: 'Newer' });
  let list = (await app.get('/characters')).json.characters;
  assert.equal(list.length, before + 2);
  assert.equal(list[0].id, b.id);

  await new Promise((r) => setTimeout(r, 5));
  await app.put(`/characters/${a.id}`, { tagline: 'touched' });
  list = (await app.get('/characters')).json.characters;
  assert.equal(list[0].id, a.id, 'an update moves a character to the front');
});

test('deleting a character mid-stream cascades without a foreign key failure', async () => {
  // Same race DELETE /conversations/:id was fixed for, one level up: the cascade
  // lands between the stream closing and the turn being written.
  const { json: c } = await create({ name: 'Doomed mid-flight', persona: 'x' });
  const originalReply = app.model.script.reply;
  app.model.script.reply = 'one two three four five six seven eight nine ten';
  app.model.script.chunkDelayMs = 30;
  let convId = null;
  let deleteStatus = null;
  try {
    const streamed = app.sse(
      { characterId: c.id, messages: [{ role: 'user', content: 'Delete my character.' }] },
      {
        onChunk: async (o) => {
          if (convId || !o.meta) return;
          convId = o.meta.conversationId;
          deleteStatus = (await app.del(`/characters/${c.id}`)).status;
        },
      },
    );
    assert.equal((await streamed).status, 200);
  } finally {
    app.model.script.chunkDelayMs = 0;
    app.model.script.reply = originalReply;
  }

  assert.equal(deleteStatus, 200, 'the delete must not sit behind the generation or fail');
  await new Promise((r) => setTimeout(r, 300));
  assert.equal((await app.get(`/characters/${c.id}`)).status, 404);
  assert.equal((await app.get(`/conversations/${convId}`)).status, 404,
    'a cascaded conversation must not come back when the stream finishes');
  assert.ok(!/FOREIGN KEY constraint failed/.test(app.stdout()), 'no FK error was logged');
});

test('an invalid character id is a 400 on delete, not a swallowed 500', async () => {
  assert.equal((await app.del('/characters/not a valid id')).status, 400);
});

test('deleting a character with many conversations cascades all of them at once', async () => {
  // The lock waits are taken concurrently. Serialised, a character with this
  // many conversations would bound the request at n x DELETE_LOCK_WAIT_MS.
  const { json: c } = await create({ name: 'Prolific', persona: 'x' });
  const ids = [];
  for (let i = 0; i < 6; i++) {
    const out = await app.sse({ characterId: c.id, messages: [{ role: 'user', content: `turn ${i}` }] });
    ids.push(out.meta.conversationId);
  }
  assert.equal(new Set(ids).size, 6, 'six distinct conversations');

  const started = Date.now();
  assert.equal((await app.del(`/characters/${c.id}`)).status, 200);
  assert.ok(Date.now() - started < 5000, 'the delete does not queue behind six separate waits');

  for (const id of ids) assert.equal((await app.get(`/conversations/${id}`)).status, 404, id);
  assert.ok(!/FOREIGN KEY constraint failed/.test(app.stdout()));
});
