'use strict';

/**
 * The renderer's three shared helpers: a hue, a set of initials, a relative
 * timestamp.
 *
 * The module is 26 lines of pure functions and had no coverage at all, which is
 * exactly why three defects sat in it. None of them was theoretical - each one
 * is reachable from a value the app really handles:
 *
 *   - `initials('   ')` returned '' rather than the '?' fallback, because the
 *     emptiness test ran on the raw string and a whitespace-only name is
 *     truthy. The avatar rendered blank. Rust rejects an empty name at
 *     `characters.rs`, but a name of spaces is not empty there either.
 *   - `initials` indexed by UTF-16 code UNIT, so a name starting with an astral
 *     character (an emoji) had its surrogate pair split and returned a LONE
 *     high surrogate - text that is not well-formed, rendering as a
 *     replacement glyph.
 *   - `relTime` on an unparseable timestamp leaked the literal string
 *     'Invalid Date' into the conversation rail, because NaN fails every
 *     comparison and falls through to toLocaleDateString().
 *
 * `hashHue` is covered for totality too: it is called during render beside
 * `initials`, and it used to throw a TypeError on a non-string.
 *
 * Dynamic import because the module is ESM under a CommonJS package.
 */

const test = require('node:test');
const assert = require('node:assert/strict');

let hashHue, initials, relTime;
test.before(async () => {
  ({ hashHue, initials, relTime } = await import(
    '../../app/src/renderer/src/lib/util.js'
  ));
});

// --- initials -------------------------------------------------------------

test('initials: ordinary names', () => {
  assert.equal(initials('Ana Lee'), 'AL');
  assert.equal(initials('Ana'), 'AN');
  assert.equal(initials('a'), 'A');
  // Middle names are ignored: first and LAST, not first and second.
  assert.equal(initials('Ana Maria Lee'), 'AL');
});

test('initials: a whitespace-only name falls back, it does not render blank', () => {
  // The defect: '   ' is truthy, so the old guard passed it through and
  // split/slice produced ''. A blank avatar is indistinguishable from a
  // broken one.
  for (const blank of ['   ', '\t', '\n', ' \t\n ', ' '.trim()]) {
    assert.equal(initials(blank), '?', `blank input ${JSON.stringify(blank)}`);
  }
});

test('initials: surrounding whitespace does not change the result', () => {
  assert.equal(initials('  Ana Lee  '), 'AL');
  assert.equal(initials('\tAna\n'), 'AN');
});

test('initials: astral characters stay well-formed', () => {
  // The defect: slice(0, 2) cut a surrogate pair in half and returned a lone
  // high surrogate. Assert on well-formedness, not just the rendered value -
  // a lone surrogate can still compare equal to a lookalike by accident.
  const emoji = '\u{1F525}'; // one astral code point, two UTF-16 code units
  const single = initials(emoji);
  assert.equal(Array.from(single).length, 1, 'one glyph, not half a pair');
  assert.equal(single.isWellFormed(), true);

  const pair = initials(`${emoji} x`);
  assert.equal(pair.isWellFormed(), true);
  assert.equal(pair, `${emoji}X`);

  // Two astral glyphs in a one-word name: take two whole code points.
  const two = initials('\u{1F525}\u{1F9EA}');
  assert.equal(Array.from(two).length, 2);
  assert.equal(two.isWellFormed(), true);
});

test('initials: a non-string never throws', () => {
  for (const bad of [null, undefined, 0, 42, {}, [], true, NaN]) {
    assert.equal(initials(bad), '?', `input ${JSON.stringify(bad)}`);
  }
});

// --- relTime --------------------------------------------------------------

test('relTime: empty input renders nothing', () => {
  for (const empty of ['', null, undefined, 0]) {
    assert.equal(relTime(empty), '');
  }
});

test('relTime: an unparseable timestamp renders nothing, not "Invalid Date"', () => {
  // The defect: NaN fails every comparison, so the function fell through to
  // toLocaleDateString() and printed the literal string into the UI.
  for (const bad of ['not a date', 'yesterday', '2026-13-45', '{}']) {
    const out = relTime(bad);
    assert.equal(out, '', `input ${JSON.stringify(bad)}`);
    assert.ok(!out.includes('Invalid'), 'must never surface "Invalid Date"');
  }
});

test('relTime: each bucket', () => {
  const ago = (ms) => new Date(Date.now() - ms).toISOString();
  assert.equal(relTime(ago(0)), 'just now');
  assert.equal(relTime(ago(59_000)), 'just now');
  assert.equal(relTime(ago(60_000)), '1m ago');
  assert.equal(relTime(ago(59 * 60_000)), '59m ago');
  assert.equal(relTime(ago(3600_000)), '1h ago');
  assert.equal(relTime(ago(23 * 3600_000)), '23h ago');
  assert.equal(relTime(ago(86_400_000)), '1d ago');
  assert.equal(relTime(ago(6 * 86_400_000)), '6d ago');
  // Past a week it becomes an absolute date, so assert only that it is not a
  // relative string - the exact format is the host locale's business.
  const old = relTime(ago(8 * 86_400_000));
  assert.ok(!old.endsWith('ago'), `expected an absolute date, got ${old}`);
  assert.ok(old.length > 0);
});

test('relTime: a future timestamp reads as "just now", never negative', () => {
  // Clock skew between two synced devices is the real source of these.
  for (const ahead of [1000, 3600_000, 30 * 86_400_000]) {
    const out = relTime(new Date(Date.now() + ahead).toISOString());
    assert.equal(out, 'just now', `+${ahead}ms`);
    assert.ok(!out.includes('-'), 'no negative figure');
  }
});

// --- hashHue --------------------------------------------------------------

test('hashHue: deterministic and in range', () => {
  const names = ['Ana Lee', 'ana lee', '', 'x', '\u{1F525}', 'A'.repeat(500)];
  for (const n of names) {
    const h = hashHue(n);
    assert.equal(Number.isInteger(h), true, `integer for ${JSON.stringify(n)}`);
    assert.ok(h >= 0 && h < 360, `0..359 for ${JSON.stringify(n)}, got ${h}`);
    assert.equal(hashHue(n), h, 'same input, same hue');
  }
  // Case matters - these are different characters and should tint differently.
  assert.notEqual(hashHue('Ana Lee'), hashHue('ana lee'));
});

test('hashHue: a non-string never throws', () => {
  // It is called during render beside initials(); it used to throw a TypeError
  // reading .length off null, which would unmount the chat.
  for (const bad of [null, undefined, 0, 42, {}, [], true]) {
    const h = hashHue(bad);
    assert.equal(Number.isInteger(h), true, `input ${JSON.stringify(bad)}`);
    assert.ok(h >= 0 && h < 360);
  }
});
