'use strict';

/**
 * Summary hygiene: the two ways a rolling summary corrupts itself.
 *
 * Both were found by a 2,000-exchange A/B, not by reading the code, and both
 * compound rather than merely degrade -- the stored summary is fed back as the
 * next fold's prior, so a defect written once is re-read and re-compressed for
 * the life of the conversation.
 *
 *   1. Front-truncation cuts at a character offset, so the survivor starts
 *      mid-word. That run ended holding a summary whose first characters were
 *      "hreads" -- a decapitated "threads".
 *   2. The summariser drifts out of narrative into copying its own input back,
 *      labels and all, spending the whole character budget on the dialogue it
 *      was asked to compress.
 *
 * Neither shows up in a short test conversation, which is exactly why they need
 * pinning here rather than being left to the next long run to rediscover.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const backend = path.resolve(__dirname, '..', '..', 'src', 'backend');
const { clampSummary, stripTranscript, foldStats, _config } = require(path.join(backend, 'memory.js'));

// ---- truncation ----------------------------------------------------------

// The lookahead is a FRACTION of the cap (20%), so these use realistic caps and
// compute the cut position rather than hand-picking offsets: at a toy cap of 30
// the whole lookahead is six characters and no boundary is ever in reach.

test('a summary is truncated from the front, keeping the newest material', () => {
  // The tail is what the model just learned. Cutting the end would throw that
  // away and keep what has already been condensed twice.
  assert.equal(clampSummary('abcdef', 3), 'def');
  assert.equal(clampSummary('abc', 10), 'abc');
});

test('a mid-word cut advances to the next whole word', () => {
  // The exact defect from the 2,000-exchange run: the stored summary opened
  // "hreads", a decapitated "threads", and then fed the next fold as its prior.
  const tail = ' include the manifest and the berth number.';
  const text = `${'a'.repeat(200)} threads${tail}`;
  const cap = 'hreads'.length + tail.length;

  assert.equal(text.slice(text.length - cap).slice(0, 6), 'hreads', 'the raw cut is the defect');
  assert.equal(clampSummary(text, cap), tail.trimStart());
});

test('a cut that already fell between words keeps that word', () => {
  // The defect is starting mid-WORD, and this start is not. Advancing anyway
  // would discard a whole word to fix nothing.
  const tail = 'He never asked about it again.';
  const text = `${'a'.repeat(200)} ${tail}`;
  assert.equal(clampSummary(text, tail.length), tail);
});

test('a blank line is preferred over a word break', () => {
  // It is the only boundary guaranteed not to be mid-thought.
  const tail = 'Second paragraph opens here and continues onward.';
  const text = `${'a'.repeat(200)}\n\n${tail}`;
  const cap = 10 + 2 + tail.length; // the cut is ten a's, the break, then the tail
  assert.equal(clampSummary(text, cap), tail);
});

test('a sentence end is preferred over a word break', () => {
  const tail = 'Second sentence opens here and continues onward.';
  const text = `${'a'.repeat(200)}. ${tail}`;
  const cap = 10 + 2 + tail.length;
  assert.equal(clampSummary(text, cap), tail);
});

test('the result is always a suffix that starts a word', () => {
  // The invariant the whole helper exists for, checked across the cap range
  // rather than at one hand-chosen size.
  const prose = 'She kept the manifest hidden below the berth plating. '
    + 'He never asked about it again, and she never offered. '
    + 'The salvage crew logged it as scrap and moved on.\n\n'
    + 'Later the broker came asking, and that was worse.';
  for (const cap of [40, 60, 100, 150, 200]) {
    const out = clampSummary(prose, cap);
    assert.ok(out.length <= cap, `cap ${cap} produced ${out.length}`);
    assert.ok(out.length > 0, `cap ${cap} produced nothing`);
    assert.ok(prose.endsWith(out), `cap ${cap} did not return a suffix`);
    if (out !== prose) {
      const prev = prose.charAt(prose.length - out.length - 1);
      assert.match(prev, /\s/, `cap ${cap} started mid-word after ${JSON.stringify(prev)}`);
    }
  }
});

test('truncation is always bounded and never empty', () => {
  // No boundary exists anywhere in the lookahead: the raw cut is the backstop,
  // which is exactly what this replaced, so it must still be bounded.
  assert.equal(clampSummary('x'.repeat(200), 50).length, 50);
  assert.equal(clampSummary('', 6000), '');
  assert.equal(clampSummary(null, 6000), '');
  assert.equal(clampSummary(undefined, 6000), '');
  for (const cap of [10, 50, 500, 6000]) {
    const out = clampSummary('word '.repeat(4000), cap);
    assert.ok(out.length <= cap, `cap ${cap} produced ${out.length}`);
    assert.ok(out.length > 0, `cap ${cap} produced nothing`);
  }
});

// ---- transcript stripping ------------------------------------------------

test('transcript lines are stripped out of a summary', () => {
  const text = [
    'She agreed to the salvage.',
    'User: Do you dream about the Petrel?',
    'Ilse Varga: Long breath in -- "I never did."',
    'The manifest stayed hidden.',
  ].join('\n');

  const out = stripTranscript(text, 'Ilse Varga');
  assert.equal(out.stripped, 2);
  assert.match(out.text, /She agreed/);
  assert.match(out.text, /The manifest/);
  assert.doesNotMatch(out.text, /User:/);
  assert.doesNotMatch(out.text, /Ilse Varga:/);
});

test('prose that merely contains a colon is left alone', () => {
  // A general ^\w+: rule would eat both of these. The match is restricted to
  // the two labels renderTurns actually emits, so it can only ever remove text
  // the summariser copied out of its own input.
  const prose = 'Note: she kept it.\nKestrel Station: a wreck she inherited.';
  const out = stripTranscript(prose, 'Ilse Varga');
  assert.equal(out.stripped, 0);
  assert.equal(out.text, prose);
});

test('a character name with regex metacharacters is matched literally', () => {
  // Personas are user-authored. An unescaped name would either throw on the
  // RegExp construction or match far more than it should.
  const name = 'Dr. Ilse (Varga-Mbeki) [ret.]';
  const out = stripTranscript(`${name}: spoken line\nkept prose`, name);
  assert.equal(out.stripped, 1);
  assert.equal(out.text, 'kept prose');

  // And the metacharacters must not have matched anything else.
  assert.equal(stripTranscript('Dr! Ilse xVargay-Mbeki z: line', name).stripped, 0);
});

test('stripping is a no-op on clean prose and on nothing', () => {
  const clean = 'She agreed to the salvage. The manifest stayed hidden.';
  assert.deepEqual(stripTranscript(clean, 'Ilse'), { text: clean, stripped: 0 });
  assert.deepEqual(stripTranscript('', 'Ilse'), { text: '', stripped: 0 });
  assert.deepEqual(stripTranscript(null, 'Ilse'), { text: '', stripped: 0 });
});

test('the blank runs left by removed lines are collapsed', () => {
  const text = 'Prose.\n\nUser: a\nUser: b\n\nMore prose.';
  const out = stripTranscript(text, 'Ilse');
  assert.equal(out.stripped, 2);
  assert.doesNotMatch(out.text, /\n{3,}/);
});

// ---- the toggle and its instrumentation ---------------------------------

test('the summary ships off by default', () => {
  // Not a preference. Over 2,000 exchanges with folding and retrieval left on
  // in both arms, the summary left recall unchanged (33.8% vs 39.4%, p=0.30)
  // while cutting accuracy-when-committed from 61.8% to 44.3% (p=0.009) and
  // answering with the WRONG established fact 41 times against 7 -- for 836
  // minutes of summariser CPU against zero.
  assert.equal(_config.SUMMARY_ENABLED, false);
});

test('fold cost is observable, including the two degeneration modes', () => {
  // These counters are what a re-run of that A/B reads. Without
  // transcriptLines and summariesRejected the fixes above would be unfalsifiable
  // outside a manual read of the stored summary.
  const stats = foldStats();
  for (const key of [
    'folds', 'summaries', 'archived',
    'summarizeMs', 'embedMs', 'waitMs',
    'transcriptLines', 'summariesRejected', 'pending',
  ]) {
    assert.equal(typeof stats[key], 'number', `foldStats().${key}`);
  }
});

test('foldStats hands back a copy, not the live counters', () => {
  const stats = foldStats();
  stats.folds = 9999;
  assert.notEqual(foldStats().folds, 9999);
});
