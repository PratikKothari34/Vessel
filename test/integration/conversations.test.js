'use strict';

/**
 * Conversation lifecycle: list, read, rename, delete, and variant selection.
 *
 * The load-bearing case in here is the last one. Deleting a conversation while
 * a generation is still streaming into it used to fail 5 times out of 5 on the
 * turns -> conversations foreign key, because the delete landed between the
 * stream closing and the turn being written. The fix was to abort the live
 * stream and then take the same per-conversation lock /chat holds; this file is
 * what keeps that fix honest.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const backend = require(path.resolve(__dirname, '..', 'helpers', 'backend.js'));

let app;
let hero;

test.before(async () => {
  app = await backend.start({ script: { reply: 'A short scripted reply.' } });
  hero = (await app.post('/characters', { name: 'Keeper', persona: 'Terse.' })).json;
});
test.after(async () => { if (app) await app.stop(); });

const say = (content) => ({ role: 'user', content });

// Start a conversation and return its id.
async function seed(characterId = hero.id, text = 'Opening line.') {
  const out = await app.sse({ characterId, messages: [say(text)] });
  assert.equal(out.status, 200);
  return out.meta.conversationId;
}

test('a conversation appears in the list with a preview and a turn count', async () => {
  const id = await seed(hero.id, 'Something memorable.');
  const list = (await app.get('/conversations')).json.conversations;
  const row = list.find((c) => c.id === id);
  assert.ok(row, 'the new conversation is listed');
  assert.equal(row.characterId, hero.id);
  assert.equal(row.turnCount, 2, 'user + assistant');
  assert.equal(row.hasSummary, false, 'nothing has been folded yet');
  assert.equal(row.preview, 'A short scripted reply.', 'the preview is the newest turn');
  assert.equal(row.title, '');
});

test('the list filters by character', async () => {
  const other = (await app.post('/characters', { name: 'Other' })).json;
  const mine = await seed(hero.id);
  const theirs = await seed(other.id);

  const filtered = (await app.get(`/conversations?characterId=${other.id}`)).json.conversations;
  assert.ok(filtered.every((c) => c.characterId === other.id));
  assert.ok(filtered.some((c) => c.id === theirs));
  assert.ok(!filtered.some((c) => c.id === mine));
});

test('an unknown characterId filters to nothing rather than erroring', async () => {
  const res = await app.get('/conversations?characterId=00000000-0000-4000-8000-000000000000');
  assert.equal(res.status, 200);
  assert.deepEqual(res.json.conversations, []);
});

test('a repeated characterId parameter is rejected, not passed through to SQL', async () => {
  // Express's extended query parser turns ?a=1&a=2 into an ARRAY. Handed
  // straight to a bound parameter that is an opaque driver-level failure.
  const res = await app.get('/conversations?characterId=aaa&characterId=bbb');
  assert.notEqual(res.status, 500, 'a malformed query string is the caller error it looks like');
  assert.ok(res.status === 200 || res.status === 400, `unexpected status ${res.status}`);
  if (res.status === 200) assert.deepEqual(res.json.conversations, []);
});

test('a bracketed characterId parameter is rejected too', async () => {
  const res = await app.get('/conversations?characterId[evil]=1');
  assert.notEqual(res.status, 500);
  assert.ok(res.status === 200 || res.status === 400, `unexpected status ${res.status}`);
});

test('reading a conversation returns summary, verbatim and archive', async () => {
  const id = await seed();
  const conv = (await app.get(`/conversations/${id}`)).json;
  assert.equal(conv.id, id);
  assert.equal(conv.characterId, hero.id);
  assert.equal(conv.summary, '');
  assert.deepEqual(conv.archive, []);
  assert.deepEqual(conv.verbatim.map((t) => t.role), ['user', 'assistant']);
  assert.ok(Number.isInteger(conv.verbatim[0].turnId));
  assert.ok(!('variants' in conv.verbatim[1]), 'one variant is not a swipe set');
});

test('an invalid id is a 400 and an unknown one is a 404', async () => {
  assert.equal((await app.get('/conversations/not a valid id')).status, 400);
  assert.equal((await app.get('/conversations/00000000-0000-4000-8000-000000000000')).status, 404);
  assert.equal((await app.patch('/conversations/not a valid id', { title: 'x' })).status, 400);
  assert.equal((await app.del('/conversations/not a valid id')).status, 400);
  assert.equal((await app.del('/conversations/00000000-0000-4000-8000-000000000000')).status, 404);
});

test('a title can be set, replaced, cleared and is capped', async () => {
  const id = await seed();
  assert.equal((await app.patch(`/conversations/${id}`, { title: 'First night' })).json.title, 'First night');
  assert.equal((await app.patch(`/conversations/${id}`, { title: 'Second night' })).json.title, 'Second night');
  assert.equal((await app.patch(`/conversations/${id}`, {})).json.title, '', 'no title means clear it');
  assert.equal((await app.patch(`/conversations/${id}`, { title: 'T'.repeat(500) })).json.title.length, 200);
  assert.equal((await app.patch(`/conversations/${id}`, { title: 42 })).json.title, '',
    'a non-string title is not stringified into the sidebar');
});

test('an active variant can be selected, and it becomes the canonical turn text', async () => {
  const id = await seed(hero.id, 'Roll for me.');
  app.model.script.reply = 'Second take.';
  try {
    await app.sse({ conversationId: id, regenerate: true });
  } finally { app.model.script.reply = 'A short scripted reply.'; }

  let conv = (await app.get(`/conversations/${id}`)).json;
  const turn = conv.verbatim[1];
  assert.equal(turn.variants.length, 2);
  assert.equal(turn.activeIndex, 1);

  const first = turn.variants[0];
  const res = await app.put(`/conversations/${id}/active-variant`, {
    turnId: turn.turnId, variantId: first.id,
  });
  assert.equal(res.status, 200);
  assert.equal(res.json.content, 'A short scripted reply.');

  conv = (await app.get(`/conversations/${id}`)).json;
  assert.equal(conv.verbatim[1].content, 'A short scripted reply.');
  assert.equal(conv.verbatim[1].activeIndex, 0);

  // And the model is sent the selected text on the next turn, not the other one.
  await app.sse({ conversationId: id, messages: [say('Carry on.')] });
  const sent = app.model.chatRequests[app.model.chatRequests.length - 1].messages;
  assert.ok(sent.some((m) => m.role === 'assistant' && m.content === 'A short scripted reply.'));
  assert.ok(!sent.some((m) => m.content === 'Second take.'));
});

test('a variant from another conversation cannot be selected', async () => {
  const a = await seed(hero.id, 'Conversation A.');
  const b = await seed(hero.id, 'Conversation B.');
  const convA = (await app.get(`/conversations/${a}`)).json;
  const convB = (await app.get(`/conversations/${b}`)).json;
  const res = await app.put(`/conversations/${b}/active-variant`, {
    turnId: convA.verbatim[1].turnId, variantId: 1,
  });
  assert.equal(res.status, 400, 'the variant must belong to a turn in THIS conversation');
  assert.equal((await app.get(`/conversations/${b}`)).json.verbatim[1].content, convB.verbatim[1].content);
});

test('active-variant requires integer ids', async () => {
  const id = await seed();
  for (const body of [{}, { turnId: 1 }, { variantId: 1 }, { turnId: '1', variantId: '1' }, { turnId: 1.5, variantId: 2 }]) {
    const res = await app.put(`/conversations/${id}/active-variant`, body);
    assert.equal(res.status, 400, `expected 400 for ${JSON.stringify(body)}`);
  }
});

test('deleting a conversation removes it and its turns', async () => {
  const id = await seed();
  assert.equal((await app.del(`/conversations/${id}`)).json.deleted, true);
  assert.equal((await app.get(`/conversations/${id}`)).status, 404);
  assert.ok(!(await app.get('/conversations')).json.conversations.some((c) => c.id === id));
  assert.equal((await app.del(`/conversations/${id}`)).status, 404, 'a second delete is a 404, not a 500');
});

test('deleting mid-stream aborts the generation instead of failing a foreign key', async () => {
  app.model.script.reply = 'one two three four five six seven eight nine ten';
  app.model.script.chunkDelayMs = 30;
  let convId = null;
  let deleteStatus = null;
  try {
    const streamed = app.sse(
      { characterId: hero.id, messages: [say('Delete me mid-flight.')] },
      {
        onChunk: async (o) => {
          if (convId || !o.meta) return;
          convId = o.meta.conversationId;
          deleteStatus = (await app.del(`/conversations/${convId}`)).status;
        },
      },
    );
    const out = await streamed;
    assert.equal(out.status, 200, 'the stream itself never errors -- it is cut short');
  } finally {
    app.model.script.chunkDelayMs = 0;
    app.model.script.reply = 'A short scripted reply.';
  }

  assert.ok(convId, 'meta arrived');
  assert.equal(deleteStatus, 200, 'the delete must not sit behind the generation or fail');
  // Give the post-stream record path its chance to (not) resurrect the row.
  await new Promise((r) => setTimeout(r, 300));
  assert.equal((await app.get(`/conversations/${convId}`)).status, 404,
    'a deleted conversation must not come back when the stream finishes');
  assert.ok(!(await app.get('/conversations')).json.conversations.some((c) => c.id === convId));
  assert.ok(!/FOREIGN KEY constraint failed/.test(app.stdout()), 'no FK error was logged');
});

test('the list is ordered most-recently-updated first', async () => {
  const older = await seed(hero.id, 'Older conversation.');
  await new Promise((r) => setTimeout(r, 5));
  const newer = await seed(hero.id, 'Newer conversation.');
  let list = (await app.get('/conversations')).json.conversations;
  assert.equal(list[0].id, newer);

  await new Promise((r) => setTimeout(r, 5));
  await app.sse({ conversationId: older, messages: [say('Back to this one.')] });
  list = (await app.get('/conversations')).json.conversations;
  assert.equal(list[0].id, older, 'a new turn moves a conversation to the top');
});
