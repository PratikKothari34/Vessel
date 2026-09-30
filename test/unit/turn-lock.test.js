'use strict';

/**
 * The per-conversation turn lock.
 *
 * It serializes post-stream bookkeeping (summarize/embed/archive) against the
 * next request's read-modify-write of the same conversation. Waiting is bounded
 * on purpose -- past LOCK_WAIT_MS the caller proceeds WITHOUT the lock, so a
 * delete never leaves a button spinning behind a stuck stream.
 *
 * That valve is also the one event in the app that can interleave two writers on
 * one conversation, and every call site discards the return value. It was
 * completely silent until 2026-09-30, and the Rust track had `timed_out` under
 * test while this track had no lock test at all. These pin both halves: that the
 * lock IS exclusive, and that giving up on it is visible when it happens.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const memory = require(path.resolve(__dirname, '..', '..', 'src', 'backend', 'memory.js'));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

test('a lock serializes holders of the same conversation', async () => {
  let inside = 0;
  let maxInside = 0;
  const worker = async () => {
    const release = await memory.acquireLock('lock-conv-a', 5000);
    inside += 1;
    maxInside = Math.max(maxInside, inside);
    await sleep(15);
    inside -= 1;
    release();
  };
  await Promise.all([worker(), worker(), worker()]);
  assert.equal(maxInside, 1, 'only one holder at a time');
});

test('two different conversations do not wait on each other', async () => {
  const a = await memory.acquireLock('lock-conv-b1', 5000);
  const t0 = Date.now();
  const b = await memory.acquireLock('lock-conv-b2', 5000);
  const waited = Date.now() - t0;
  a(); b();
  assert.ok(waited < 200, `an unrelated conversation waited ${waited}ms`);
});

test('waiting past the deadline proceeds unlocked, and says so', async () => {
  // The deliberate valve. A caller that gives up on the lock must not do it
  // silently: this is the only path that can produce interleaved writes, and
  // without a line in the log the resulting mess has no findable cause.
  const warnings = [];
  const realWarn = console.warn;
  console.warn = (...a) => warnings.push(a.join(' '));
  try {
    const held = await memory.acquireLock('lock-conv-c', 5000);
    assert.equal(held.timedOut, false, 'an uncontended lock is really held');

    const gaveUp = await memory.acquireLock('lock-conv-c', 30);
    assert.equal(gaveUp.timedOut, true, 'the second holder did not get the lock');

    held();
    gaveUp();
  } finally {
    console.warn = realWarn;
  }
  assert.equal(warnings.length, 1, 'exactly one warning for one expired wait');
  assert.match(warnings[0], /WITHOUT the turn lock/);
  assert.match(warnings[0], /lock-conv-c/, 'the warning names the conversation');
});

test('releasing twice is a no-op rather than an error', async () => {
  const release = await memory.acquireLock('lock-conv-d', 1000);
  release();
  release();
  // The lock must still be grantable afterwards -- a double release that
  // resolved someone else's chain would strand the next waiter.
  const again = await memory.acquireLock('lock-conv-d', 1000);
  assert.equal(again.timedOut, false);
  again();
});

test('the registry does not grow across acquire/release cycles', async () => {
  // A dead entry per conversation for the life of the process is how this
  // leaked before the tail-identity check; the Rust side refcounts instead.
  for (let i = 0; i < 200; i += 1) {
    const release = await memory.acquireLock(`lock-churn-${i}`, 1000);
    release();
  }
  assert.equal(memory._internals.lockCount(), 0, 'every key is dropped once nobody holds it');
});
