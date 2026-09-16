'use strict';

/**
 * The fold: summarise + embed + archive, and the recall that reads it back.
 *
 * This is the part of the system that decides what the model remembers about a
 * story it can no longer see, so the tests below care about three separate
 * things:
 *
 *   1. The drain lands on VERBATIM_TURNS, not on SUMMARIZE_THRESHOLD. The
 *      threshold is the TRIGGER, the window is the TARGET, and a capped fold
 *      that stops at the trigger leaves the window permanently oversized. That
 *      bug is invisible from the outside -- everything still works, the prompt
 *      is just bigger forever.
 *   2. A failed summariser loses nothing. Turns are only deleted after the
 *      summary comes back, so an engine that is down means a bigger window, not
 *      an amputated story.
 *   3. Recall ranks by cosine over the archive, and the recall block lands in
 *      the volatile TAIL of the prompt.
 *
 * The fold thresholds are pushed right down via the environment so a handful of
 * turns exercises what normally takes a dozen.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const backend = require(path.resolve(__dirname, '..', 'helpers', 'backend.js'));
const { fakeEmbedding } = require(path.resolve(__dirname, '..', 'helpers', 'fake-inference.js'));

const FOLD_ENV = {
  // The rolling summary ships OFF (see SUMMARY_ENABLED in memory.js -- a
  // 2,000-exchange A/B found it changed how the model failed, not how much it
  // remembered, for 836 minutes of CPU). Every test in this file is about the
  // summariser itself, so it is turned back on here rather than in the shared
  // helper, which would leave the shipped default with no coverage at all.
  SUMMARY_ENABLED: '1',
  VERBATIM_TURNS: '2',
  SUMMARIZE_THRESHOLD: '4',
  MAX_FOLD_TURNS: '2',   // forces a capped fold, so the drain needs two passes
  RETRIEVE_K: '1',
  RETRIEVE_MIN_SCORE: '-1', // the fake embedder is a hash, so nothing is "similar"
};

const say = (content) => ({ role: 'user', content });

function cosine(a, b) {
  let d = 0; let na = 0; let nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

// Drive `n` exchanges through a conversation, returning its id.
async function exchange(app, convId, characterId, texts) {
  let id = convId;
  for (const t of texts) {
    const out = await app.sse(id ? { conversationId: id, messages: [say(t)] } : { characterId, messages: [say(t)] });
    assert.equal(out.status, 200, `turn "${t}" failed`);
    id = out.meta.conversationId;
  }
  return id;
}

// The fold runs in the background, off the turn lock, so every assertion about
// it has to wait for it rather than assume it already happened.
async function waitFor(fn, what, ms = 10000) {
  const deadline = Date.now() + ms;
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 50));
  }
}

let app;
let hero;

test.before(async () => {
  app = await backend.start({
    env: FOLD_ENV,
    script: { reply: 'Reply.', summary: 'SUMMARY: they met at the lighthouse.' },
  });
  hero = (await app.post('/characters', { name: 'Mira', persona: 'Terse.' })).json;
});
test.after(async () => { if (app) await app.stop(); });

test('nothing is folded while the conversation is under the threshold', async () => {
  const id = await exchange(app, null, hero.id, ['one', 'two']); // 4 turns, threshold 4
  await new Promise((r) => setTimeout(r, 300));
  const conv = (await app.get(`/conversations/${id}`)).json;
  assert.equal(conv.verbatim.length, 4);
  assert.deepEqual(conv.archive, []);
  assert.equal(conv.summary, '');
  assert.equal(app.model.genRequests.length, 0, 'the summariser was never called');
});

test('a capped fold drains all the way down to the verbatim window', async () => {
  app.model.reset();
  const id = await exchange(app, null, hero.id, ['a1', 'a2', 'a3']); // 6 turns

  const conv = await waitFor(async () => {
    const c = (await app.get(`/conversations/${id}`)).json;
    return c.archive.length >= 4 ? c : null;
  }, 'the archive to fill');

  // MAX_FOLD_TURNS is 2, so this took two passes. Stopping at the trigger would
  // have left 4 verbatim turns; the target is VERBATIM_TURNS.
  assert.equal(conv.verbatim.length, 2, 'the window lands on VERBATIM_TURNS, not on the threshold');
  assert.equal(conv.archive.length, 4);
  assert.deepEqual(conv.archive.map((t) => t.role), ['user', 'assistant', 'user', 'assistant']);
  assert.equal(conv.archive[0].content, 'a1', 'oldest first');
  assert.ok(conv.archive.every((t) => t.hasEmbedding), 'every archived turn is retrievable');
  assert.equal(conv.summary, 'SUMMARY: they met at the lighthouse.');
  assert.ok(app.model.genRequests.length >= 2, 'one summariser call per pass');
});

test('the summariser is given the prior summary and the turns being folded', async () => {
  app.model.reset();
  const id = await exchange(app, null, hero.id, ['b1', 'b2', 'b3']);
  await waitFor(async () => (await app.get(`/conversations/${id}`)).json.archive.length >= 4, 'the fold');

  const first = app.model.genRequests[0];
  assert.equal(first.model, 'test-summarizer', 'the summariser is its own model');
  assert.equal(first.stream, false);
  assert.match(first.prompt, /=== CURRENT SUMMARY ===\s*\(none yet\)/);
  assert.match(first.prompt, /User: b1/);
  assert.match(first.prompt, /Mira: Reply\./, 'turns are rendered with the character name, not "assistant"');
  assert.match(first.prompt, /Keep the updated summary under \d+ characters/);

  const second = app.model.genRequests[1];
  assert.match(second.prompt, /=== CURRENT SUMMARY ===\s*SUMMARY: they met at the lighthouse\./,
    'the second pass folds into the first pass result');
});

test('the summary is replayed to the model as a system message inside the prefix', async () => {
  app.model.reset();
  const id = await exchange(app, null, hero.id, ['c1', 'c2', 'c3']);
  await waitFor(async () => (await app.get(`/conversations/${id}`)).json.archive.length >= 4, 'the fold');

  await app.sse({ conversationId: id, messages: [say('c4')] });
  const { messages } = app.model.chatRequests[app.model.chatRequests.length - 1];
  const summaryIdx = messages.findIndex((m) => /Story so far/.test(m.content));
  assert.ok(summaryIdx !== -1, 'the rolling summary reaches the model');
  assert.match(messages[summaryIdx].content, /they met at the lighthouse/);
  assert.equal(messages[summaryIdx].role, 'system');
  assert.equal(summaryIdx, 1, 'persona, then summary: both stable, both in the cacheable prefix');
  assert.equal(messages[messages.length - 1].content, 'c4');
});

test('recall pulls the best-matching archived turn into the volatile tail', async () => {
  app.model.reset();
  const id = await exchange(app, null, hero.id, ['d1 lantern', 'd2 harbour', 'd3 rope']);
  const folded = await waitFor(async () => {
    const c = (await app.get(`/conversations/${id}`)).json;
    return c.archive.length >= 4 ? c : null;
  }, 'the fold');

  // The fake embedder is a hash, not a semantic model: two different strings
  // give two independent vectors whose cosine sits near zero, so "similar" text
  // proves nothing and the ranking of unrelated strings is a coin flip inside
  // int8 quantization error. Querying with the exact stored string is the only
  // fixture that makes the winner decidable -- it scores 1.0 against its own row
  // and ~0 against every other, which is precisely the ranking being tested.
  // The archive embeds "role: content", so the query has to carry that prefix.
  const target = folded.archive.find((t) => t.content === 'd1 lantern');
  const query = `${target.role}: ${target.content}`;
  const q = fakeEmbedding(query);
  const scored = folded.archive
    .map((t) => ({ content: t.content, score: cosine(q, fakeEmbedding(`${t.role}: ${t.content}`)) }))
    .sort((a, b) => b.score - a.score);
  assert.ok(scored[0].score - scored[1].score > 0.01,
    'fixture is degenerate: the top two are within int8 quantization error of each other');

  const out = await app.sse({ conversationId: id, messages: [say(query)] });
  assert.equal(out.meta.recalled.length, 1, 'RETRIEVE_K bounds it');
  assert.equal(out.meta.recalled[0].content, scored[0].content, 'the highest cosine wins');
  assert.ok(Math.abs(out.meta.recalled[0].score - scored[0].score) < 0.01,
    'the reported score matches the int8 round trip');

  // And the block itself sits AFTER the verbatim window, immediately before the
  // new user message -- the entire point of the Stage 2 reorder.
  const { messages } = app.model.chatRequests[app.model.chatRequests.length - 1];
  const recallIdx = messages.findIndex((m) => /^\[Recall/.test(m.content));
  assert.equal(recallIdx, messages.length - 2);
  assert.equal(messages[recallIdx].role, 'system');
  assert.match(messages[recallIdx].content, /End of recall\. The present moment is the conversation above\./);
  assert.match(messages[recallIdx].content, new RegExp(scored[0].content));
});

test('recall is scoped to one conversation', async () => {
  app.model.reset();
  const a = await exchange(app, null, hero.id, ['zebra crossing', 'zebra again', 'zebra third']);
  await waitFor(async () => (await app.get(`/conversations/${a}`)).json.archive.length >= 4, 'fold a');
  const b = await exchange(app, null, hero.id, ['unrelated one', 'unrelated two', 'unrelated three']);
  await waitFor(async () => (await app.get(`/conversations/${b}`)).json.archive.length >= 4, 'fold b');

  const out = await app.sse({ conversationId: b, messages: [say('zebra crossing')] });
  assert.equal(out.meta.recalled.length, 1);
  assert.ok(!/zebra/.test(out.meta.recalled[0].content),
    'another conversation archive must never leak into this story');
});

test('a summariser failure keeps the turns verbatim instead of losing them', async () => {
  app.model.reset();
  app.model.script.generateFails = true;
  let id;
  try {
    id = await exchange(app, null, hero.id, ['e1', 'e2', 'e3']);
    await waitFor(async () => app.model.genRequests.length > 0, 'the summariser attempt');
    await new Promise((r) => setTimeout(r, 400));
    const conv = (await app.get(`/conversations/${id}`)).json;
    assert.equal(conv.archive.length, 0, 'nothing may be archived without a summary to replace it');
    assert.equal(conv.verbatim.length, 6, 'the turns are all still there');
    assert.equal(conv.summary, '');
  } finally { app.model.script.generateFails = false; }

  // And the next turn retries, so a transient outage self-heals.
  await app.sse({ conversationId: id, messages: [say('e4')] });
  const conv = await waitFor(async () => {
    const c = (await app.get(`/conversations/${id}`)).json;
    return c.archive.length > 0 ? c : null;
  }, 'the retried fold');
  assert.equal(conv.summary, 'SUMMARY: they met at the lighthouse.');
});

test('an embedder failure archives the text anyway, just unretrievable', async () => {
  app.model.reset();
  app.model.script.embedFails = true;
  let id;
  try {
    id = await exchange(app, null, hero.id, ['f1', 'f2', 'f3']);
    const conv = await waitFor(async () => {
      const c = (await app.get(`/conversations/${id}`)).json;
      return c.archive.length >= 4 ? c : null;
    }, 'the fold');
    assert.ok(conv.archive.every((t) => !t.hasEmbedding));
    assert.equal(conv.archive[0].content, 'f1', 'the prose survives -- only recall is lost');
  } finally { app.model.script.embedFails = false; }

  // A conversation with no usable vectors must not break the next turn.
  const out = await app.sse({ conversationId: id, messages: [say('f4')] });
  assert.equal(out.status, 200);
  assert.deepEqual(out.meta.recalled, []);
});

test('deleting a conversation mid-fold does not resurrect it', async () => {
  app.model.reset();
  const id = await exchange(app, null, hero.id, ['g1', 'g2', 'g3']);
  // Delete while the background fold is still in its summarise+embed legs.
  assert.equal((await app.del(`/conversations/${id}`)).status, 200);
  await new Promise((r) => setTimeout(r, 800));
  assert.equal((await app.get(`/conversations/${id}`)).status, 404);
  assert.ok(!(await app.get('/conversations')).json.conversations.some((c) => c.id === id));
  assert.ok(!/FOREIGN KEY constraint failed/.test(app.stdout()));
});

test('the window stats reported per turn describe the real prompt', async () => {
  app.model.reset();
  await app.del('/metrics');
  const id = await exchange(app, null, hero.id, ['h1', 'h2', 'h3']);
  await waitFor(async () => (await app.get(`/conversations/${id}`)).json.archive.length >= 4, 'the fold');
  await app.sse({ conversationId: id, messages: [say('h4 and a query')] });

  const rec = (await app.get(`/metrics?conversationId=${id}`)).json.recent.pop();
  const { messages } = app.model.chatRequests[app.model.chatRequests.length - 1];
  const total = messages.reduce((n, m) => n + m.content.length, 0);
  assert.equal(rec.window.messageCount, messages.length);
  assert.equal(rec.window.promptChars, total);
  assert.ok(rec.window.summaryChars > 0);
  assert.ok(rec.window.retrievedCount > 0);
  assert.ok(rec.window.personaChars > 0);
  assert.equal(rec.window.newUserChars, 'h4 and a query'.length);
  assert.ok(rec.window.stablePrefixChars < total, 'the recall block is outside the stable prefix');
});
