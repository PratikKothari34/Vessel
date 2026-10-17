'use strict';

/**
 * Who a finished chat stream belongs to.
 *
 * The defect this pins is a real desync, not a cosmetic one. Aborting a stream
 * does not detach the channel handler -- it sets a flag and fires a
 * best-effort cancel -- so when the user clicks another scenario mid-generation
 * the original `onDone` and `onError` still run, holding the first
 * conversation's id in their closure. Three observed consequences: the reply
 * was appended to the conversation now on screen, the stale id was then
 * reloaded over it, and the error path restored the OTHER conversation's
 * message array under the current heading.
 *
 * `stopAndReload` always had this guard. The completion paths did not, so the
 * cases below are ordered the way they actually happen: start a stream, switch
 * away, then let the result land.
 *
 * Dynamic import because the module is ESM under a CommonJS package.
 */

const test = require('node:test');
const assert = require('node:assert/strict');

let ownsStream;
test.before(async () => {
  ({ ownsStream } = await import('../../app/ui/lib/stream-owner.mjs'));
});

test('a reply that lands while its own conversation is open is still owned', () => {
  assert.equal(ownsStream('conv-A', 'conv-A'), true);
});

test('a reply that lands after the user switched scenarios is disowned', () => {
  // The case that put conversation A's text into conversation B's transcript.
  assert.equal(ownsStream('conv-A', 'conv-B'), false);
});

test('a stream never adopted by the server cannot be placed anywhere', () => {
  // A brand-new scenario is null until `onMeta` mints an id. A result still
  // holding null has no home, and guessing means writing it into whatever is
  // open.
  assert.equal(ownsStream(null, 'conv-B'), false);
  assert.equal(ownsStream(null, null), false);
});

test('a result landing after the user started a fresh scenario is disowned', () => {
  // newConversation() sets convId back to null while a stream is in flight.
  assert.equal(ownsStream('conv-A', null), false);
});

test('a non-string id compares false rather than throwing', () => {
  // Called from a stream callback during an unmount race, where a throw escapes
  // to window.onerror instead of any component's error state.
  assert.equal(ownsStream(undefined, 'conv-A'), false);
  assert.equal(ownsStream('conv-A', undefined), false);
  assert.equal(ownsStream(42, 42), false);
  assert.equal(ownsStream('', ''), false);
});
