'use strict';

/**
 * Regeneration variants, and the back-fill that stands behind them.
 *
 * recordTurn writes a base variant alongside every assistant reply, so in the
 * normal path there is always something to swipe back to. Two kinds of row have
 * no variant at all: turns written before the variants table existed, and turns
 * pulled from an older synced database -- runMigrations adds columns, it never
 * back-fills rows.
 *
 * The back-fill used to be seeded from the CALLER's text, which meant the first
 * regenerate on such a turn wrote the regeneration in as the "original", mirrored
 * it over turns.content, and left two identical variants. The reply being
 * replaced was gone, the swipe led nowhere, and nothing reported either. These
 * tests run against a real scratch database (guard.mjs puts LOCAL_DB_PATH in a
 * per-worker temp directory) because the defect lives in what the rows end up
 * holding, not in any value a function returns.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const backend = path.resolve(__dirname, '..', '..', 'src', 'backend');
const { getDb } = require(backend + '/db.js');
const memory = require(backend + '/memory.js');

const user = (content) => ({ role: 'user', content });

let seq = 0;
// One conversation per test: these write rows, and a shared id would let the
// order tests run in decide what each of them sees.
async function freshConversation() {
  const id = 'conv-variants-' + (++seq);
  await memory.ensureConversation(id, null);
  return id;
}

// Strip a turn of its variant rows, which is the state a pre-variants or
// freshly-synced row arrives in.
async function stripVariants(turnId) {
  const db = await getDb();
  await db.execute({ sql: 'DELETE FROM variants WHERE turn_id = ?', args: [turnId] });
}

async function variantsOf(turnId) {
  const db = await getDb();
  const res = await db.execute({
    sql: 'SELECT id, content, is_active FROM variants WHERE turn_id = ? ORDER BY id ASC',
    args: [turnId],
  });
  return res.rows.map((r) => ({
    id: Number(r.id),
    content: r.content,
    active: Boolean(Number(r.is_active)),
  }));
}

async function turnContent(turnId) {
  const db = await getDb();
  const res = await db.execute({ sql: 'SELECT content FROM turns WHERE id = ?', args: [turnId] });
  return res.rows.length ? res.rows[0].content : null;
}

test('a reply is registered as its own first variant', async () => {
  const conv = await freshConversation();
  await memory.recordTurn(conv, user('hello'), 'the original reply', 'Character');

  const last = await memory.getLastAssistantTurn(conv);
  const rows = await variantsOf(last.id);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].content, 'the original reply');
  assert.ok(rows[0].active);
});

test('regenerating keeps the original as a variant and activates the new one', async () => {
  const conv = await freshConversation();
  await memory.recordTurn(conv, user('hello'), 'the original reply', 'Character');
  const last = await memory.getLastAssistantTurn(conv);

  const { variantId } = await memory.recordRegeneration(conv, 'the second roll');

  const rows = await variantsOf(last.id);
  assert.deepEqual(rows.map((r) => r.content), ['the original reply', 'the second roll']);
  assert.deepEqual(rows.map((r) => r.active), [false, true]);
  assert.equal(rows[1].id, variantId, 'the returned id is the row just written');
  assert.equal(await turnContent(last.id), 'the second roll');
});

test('regenerating a turn with NO variant rows preserves the original reply', async () => {
  // The defect, stated as a test. A turn from before the variants table, or one
  // just pulled from an older synced database, has no variant row -- and the
  // back-fill has to read turns.content rather than trusting the new text it was
  // handed, or the reply it is supposed to be preserving is the one it destroys.
  const conv = await freshConversation();
  await memory.recordTurn(conv, user('hello'), 'the reply that predates variants', 'Character');
  const last = await memory.getLastAssistantTurn(conv);
  await stripVariants(last.id);

  await memory.recordRegeneration(conv, 'the second roll');

  const rows = await variantsOf(last.id);
  assert.equal(rows.length, 2, 'the base was back-filled, not overwritten');
  assert.equal(rows[0].content, 'the reply that predates variants', 'the original survived');
  assert.equal(rows[1].content, 'the second roll');
  assert.notEqual(rows[0].content, rows[1].content, 'two identical variants is the bug');
});

test('a back-filled original can still be swiped back to', async () => {
  // Preserving the row is only half of it: the point of the variant is that the
  // user can return to the reply the regeneration replaced.
  const conv = await freshConversation();
  await memory.recordTurn(conv, user('hello'), 'the reply that predates variants', 'Character');
  const last = await memory.getLastAssistantTurn(conv);
  await stripVariants(last.id);
  await memory.recordRegeneration(conv, 'the second roll');

  const rows = await variantsOf(last.id);
  const back = await memory.setActiveVariant(conv, last.id, rows[0].id);

  assert.equal(back.content, 'the reply that predates variants');
  assert.equal(await turnContent(last.id), 'the reply that predates variants');
});

test('exactly one variant is active after any number of rolls', async () => {
  const conv = await freshConversation();
  await memory.recordTurn(conv, user('hello'), 'roll one', 'Character');
  const last = await memory.getLastAssistantTurn(conv);

  for (const text of ['roll two', 'roll three', 'roll four']) {
    await memory.recordRegeneration(conv, text);
    const rows = await variantsOf(last.id);
    assert.equal(rows.filter((r) => r.active).length, 1, 'after ' + text);
    assert.equal(rows[rows.length - 1].active, true, 'the newest roll is the active one');
  }

  const rows = await variantsOf(last.id);
  assert.deepEqual(
    rows.map((r) => r.content),
    ['roll one', 'roll two', 'roll three', 'roll four'],
    'oldest first, which is the order the swipe UI reads',
  );
  assert.equal(await turnContent(last.id), 'roll four');
});

test('a conversation with nothing to regenerate reports that instead of writing', async () => {
  const conv = await freshConversation();
  assert.equal(await memory.recordRegeneration(conv, 'a reply to nothing'), null);

  // And a conversation whose last turn is the USER's is equally not a target:
  // appending there would attach an assistant variant to a user turn.
  await memory.recordUserTurn(conv, user('hello'));
  assert.equal(await memory.recordRegeneration(conv, 'a reply to nothing'), null);
});

test('the swipe set is only offered once there is something to swipe between', async () => {
  // getConversation attaches variants to the last assistant turn only when there
  // is more than one, so a first-time reader sees no swipe affordance at all.
  const conv = await freshConversation();
  await memory.recordTurn(conv, user('hello'), 'the only reply', 'Character');

  let view = await memory.getConversation(conv);
  let turn = view.verbatim[view.verbatim.length - 1];
  assert.ok(!('variants' in turn));

  await memory.recordRegeneration(conv, 'the second roll');
  view = await memory.getConversation(conv);
  turn = view.verbatim[view.verbatim.length - 1];
  assert.equal(turn.variants.length, 2);
  assert.equal(turn.activeIndex, 1);
});

test('a variant of another conversation cannot be activated', async () => {
  // The join in setActiveVariant is the authorization check: a variant id on its
  // own says nothing about which conversation it belongs to.
  const mine = await freshConversation();
  const theirs = await freshConversation();
  await memory.recordTurn(mine, user('hello'), 'my reply', 'Character');
  await memory.recordTurn(theirs, user('hello'), 'their reply', 'Character');

  const myTurn = await memory.getLastAssistantTurn(mine);
  const theirTurn = await memory.getLastAssistantTurn(theirs);
  const theirVariants = await variantsOf(theirTurn.id);

  await assert.rejects(
    () => memory.setActiveVariant(mine, myTurn.id, theirVariants[0].id),
    /Variant not found/,
  );
  assert.equal(await turnContent(myTurn.id), 'my reply');
});
