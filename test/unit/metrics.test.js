'use strict';

/**
 * Generation telemetry.
 *
 * prefillReuse is the number the whole Stage 2 prompt reorder is judged by, so
 * it has to mean what it claims: the fraction of the CURRENT prompt that shares
 * a prefix with the previous one. Getting that wrong does not break the app --
 * it quietly tells us a reorder worked when it did not.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const metrics = require(path.resolve(__dirname, '..', '..', 'src', 'backend', 'metrics.js'));

const sys = (c) => ({ role: 'system', content: c });
const user = (c) => ({ role: 'user', content: c });
const bot = (c) => ({ role: 'assistant', content: c });

const done = (over = {}) => ({
  prompt_eval_count: 1000, prompt_eval_duration: 1e9,
  eval_count: 100, eval_duration: 2e9,
  total_duration: 3e9, load_duration: 0,
  ...over,
});

test.beforeEach(() => metrics.reset());

test('the first turn of a conversation has no previous prompt to compare', () => {
  const rec = metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('persona'), user('hi')] });
  assert.equal(rec.prefillReuse, null);
});

test('an unchanged prompt reports full reuse', () => {
  const msgs = [sys('persona'), user('hi')];
  metrics.record({ conversationId: 'c1', done: done(), promptMessages: msgs });
  const rec = metrics.record({ conversationId: 'c1', done: done(), promptMessages: msgs.map((m) => ({ ...m })) });
  assert.equal(rec.prefillReuse, 1);
});

test('appending to the tail keeps the whole prefix', () => {
  const stable = [sys('A'.repeat(900)), user('B'.repeat(100))];
  metrics.record({ conversationId: 'c1', done: done(), promptMessages: stable });
  // 1000 shared chars out of 1100 total.
  const next = [...stable, bot('C'.repeat(100))];
  const rec = metrics.record({ conversationId: 'c1', done: done(), promptMessages: next });
  assert.equal(rec.prefillReuse, Math.round((1000 / 1100) * 1000) / 1000);
});

test('a change in the FIRST message destroys the whole prefix', () => {
  // This is the failure the prompt order exists to prevent: anything volatile
  // placed early re-prefills everything after it.
  metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('recall: X'), user('hi')] });
  const rec = metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('recall: Y'), user('hi')] });
  // "recall: " is 8 shared chars, out of 8 + 1 + 2 = 11.
  assert.equal(rec.prefillReuse, Math.round((8 / 11) * 1000) / 1000);
});

test('a role change at the same index stops the walk', () => {
  metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('same'), user('x')] });
  const rec = metrics.record({ conversationId: 'c1', done: done(), promptMessages: [user('same'), user('x')] });
  assert.equal(rec.prefillReuse, 0);
});

test('a shorter follow-up prompt does not read past the end of either array', () => {
  metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('abc'), user('def'), bot('ghi')] });
  const rec = metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('abc')] });
  assert.equal(rec.prefillReuse, 1);
});

test('a message that is a strict prefix of the previous one counts only its own length', () => {
  metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('abcdefghij')] });
  const rec = metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('abcde')] });
  assert.equal(rec.prefillReuse, 1);
});

test('reuse is tracked per conversation, not globally', () => {
  metrics.record({ conversationId: 'a', done: done(), promptMessages: [sys('AAAA')] });
  metrics.record({ conversationId: 'b', done: done(), promptMessages: [sys('BBBB')] });
  const a = metrics.record({ conversationId: 'a', done: done(), promptMessages: [sys('AAAA')] });
  assert.equal(a.prefillReuse, 1, 'conversation b must not have evicted or poisoned a');
});

test('forgetConversation drops the stored prompt so a reused id starts clean', () => {
  metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('AAAA')] });
  metrics.forgetConversation('c1');
  const rec = metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('AAAA')] });
  assert.equal(rec.prefillReuse, null);
});

test('the previous-prompt map is bounded', () => {
  // PREV_MAX defaults to 32. Touch 40 conversations, then come back to the first:
  // it must have been evicted rather than retained forever.
  for (let i = 0; i < 40; i++) {
    metrics.record({ conversationId: `c${i}`, done: done(), promptMessages: [sys(`p${i}`)] });
  }
  const first = metrics.record({ conversationId: 'c0', done: done(), promptMessages: [sys('p0')] });
  assert.equal(first.prefillReuse, null, 'oldest entry should have been evicted');
  const last = metrics.record({ conversationId: 'c39', done: done(), promptMessages: [sys('p39')] });
  assert.equal(last.prefillReuse, 1, 'most recent entry should still be held');
});

test('an empty prompt array reports null instead of dividing by zero', () => {
  metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('x')] });
  const rec = metrics.record({ conversationId: 'c1', done: done(), promptMessages: [] });
  assert.equal(rec.prefillReuse, null);
});

test('messages with no characters at all report null, not NaN', () => {
  metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('')] });
  const rec = metrics.record({ conversationId: 'c1', done: done(), promptMessages: [sys('')] });
  assert.equal(rec.prefillReuse, null);
});

test('derived rates come out of the done chunk', () => {
  const rec = metrics.record({
    conversationId: 'c1',
    done: done({ prompt_eval_count: 1200, prompt_eval_duration: 2e9, eval_count: 60, eval_duration: 3e9 }),
    window: { promptChars: 4800 },
  });
  assert.equal(rec.promptTokens, 1200);
  assert.equal(rec.prefillTps, 600);        // 1200 tokens / 2 s
  assert.equal(rec.decodeTps, 20);          // 60 tokens / 3 s
  assert.equal(rec.charsPerToken, 4);       // 4800 chars / 1200 tokens
  assert.equal(rec.promptMs, 2000);
});

test('an aborted generation records the abort and nothing it cannot know', () => {
  const rec = metrics.record({ conversationId: 'c1', done: null, aborted: true });
  assert.equal(rec.aborted, true);
  assert.equal(rec.promptTokens, null);
  assert.equal(rec.decodeTps, null);
  assert.equal(rec.charsPerToken, null);
});

test('cacheReuse is reported only when the engine supplies cached_tokens', () => {
  const ollama = metrics.record({ conversationId: 'c1', done: done() });
  assert.equal(ollama.cachedTokens, null);
  assert.equal(ollama.cacheReuse, null);
  const llama = metrics.record({ conversationId: 'c2', done: done({ cached_tokens: 900 }) });
  assert.equal(llama.cacheReuse, 0.9);
});

test('snapshot summarises, filters by conversation and honours the limit', () => {
  for (let i = 0; i < 5; i++) {
    metrics.record({ conversationId: 'a', done: done({ prompt_eval_count: 100 * (i + 1) }) });
  }
  metrics.record({ conversationId: 'b', done: done({ prompt_eval_count: 9000 }) });

  const all = metrics.snapshot({ limit: 50 });
  assert.equal(all.summary.samples, 6);
  assert.equal(all.summary.promptTokens.max, 9000);

  const onlyA = metrics.snapshot({ limit: 50, conversationId: 'a' });
  assert.equal(onlyA.summary.samples, 5);
  assert.equal(onlyA.summary.promptTokens.max, 500);

  assert.equal(metrics.snapshot({ limit: 2 }).recent.length, 2);
});

test('a reload is counted only when load time is real', () => {
  metrics.record({ conversationId: 'c1', done: done({ load_duration: 1e6 }) });   // 1 ms
  metrics.record({ conversationId: 'c1', done: done({ load_duration: 19e9 }) });  // 19 s
  assert.equal(metrics.snapshot({}).summary.reloads, 1);
});

test('the ring buffer is bounded and keeps the newest records', () => {
  // RING defaults to 200.
  for (let i = 0; i < 260; i++) {
    metrics.record({ conversationId: 'c1', done: done({ prompt_eval_count: i }) });
  }
  const snap = metrics.snapshot({ limit: 1000 });
  assert.equal(snap.recent.length, 200);
  assert.equal(snap.recent[snap.recent.length - 1].promptTokens, 259);
  assert.equal(snap.recent[0].promptTokens, 60);
});

test('a record with no conversation id is stored but skips the reuse estimate', () => {
  const rec = metrics.record({ done: done(), promptMessages: [sys('x')] });
  assert.equal(rec.prefillReuse, null);
  assert.equal(metrics.snapshot({}).summary.samples, 1);
});

test('a truncated answer is distinguishable from a short one', () => {
  // The reason `matched` exists. A caller that asks for 500 records and gets 50
  // cannot otherwise tell "that is all there is" from "the ring gave you what
  // it felt like". Both answers used to look identical on the wire.
  for (let i = 0; i < 40; i++) metrics.record({ conversationId: 'c1', done: done() });

  const all = metrics.snapshot({ limit: 500 });
  assert.equal(all.config.matched, 40);
  assert.equal(all.config.returned, 40);
  assert.equal(all.recent.length, 40);

  const cut = metrics.snapshot({ limit: 10 });
  assert.equal(cut.config.matched, 40, 'matched counts what the ring holds, not what it returned');
  assert.equal(cut.config.returned, 10);
  assert.equal(cut.recent.length, 10);
});

test('matched counts the filtered conversation, not the whole ring', () => {
  for (let i = 0; i < 7; i++) metrics.record({ conversationId: 'c1', done: done() });
  for (let i = 0; i < 3; i++) metrics.record({ conversationId: 'c2', done: done() });
  assert.equal(metrics.snapshot({ limit: 100, conversationId: 'c2' }).config.matched, 3);
  assert.equal(metrics.snapshot({ limit: 100 }).config.matched, 10);
});

test('ringSize reports the configured ring, which is what the route clamps to', () => {
  // /metrics used to clamp `limit` to a literal 200, so a larger METRICS_RING
  // was unreadable past its newest 200 records however it was configured.
  // The route now clamps to this, so the two can never drift apart again.
  assert.equal(metrics.ringSize(), metrics.snapshot({}).config.ring);
  assert.ok(metrics.ringSize() >= 20, 'the ring floor is 20');
});
