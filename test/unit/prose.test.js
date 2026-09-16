'use strict';

/**
 * The reply renderer's interpretation layer.
 *
 * This is the last thing that touches a reply before the user reads it, and
 * every defect here is visible: dialogue that never gets colored, a paragraph
 * that arrives as one wall of text, an action marker rendered as literal
 * asterisks. It had no coverage at all until the 1000-exchange stress test
 * surfaced the mixed-quote bug by eye, which is a slow way to find out.
 *
 * Dynamic import because the module is ESM under a CommonJS package.
 */

const test = require('node:test');
const assert = require('node:assert/strict');

let toBlocks, splitLongBlock, tokenize;
test.before(async () => {
  ({ toBlocks, splitLongBlock, tokenize } = await import(
    '../../app/src/renderer/src/lib/prose.mjs'
  ));
});

const kinds = (line) => tokenize(line).map((t) => t.kind);
const texts = (line) => tokenize(line).map((t) => t.text);

// ---- Spans ---------------------------------------------------------------

test('straight, curly and mixed quotes all read as dialogue', async () => {
  // The bug from the original report: the 8B model opens with a straight quote
  // and closes with a curly one often enough that a same-style-only pattern
  // leaves real dialogue uncolored.
  for (const line of ['"hello"', '“hello”', '"hello”', '“hello"']) {
    assert.deepEqual(kinds(line), ['dialogue'], line);
    assert.equal(tokenize(line)[0].text, line, 'the marks stay, they are the styling');
  }
});

test('an action keeps its text and loses its delimiters', async () => {
  for (const line of ['*leans in*', '_leans in_']) {
    const [tok] = tokenize(line);
    assert.equal(tok.kind, 'action');
    assert.equal(tok.text, 'leans in', 'the reader never sees the marker');
  }
});

test('prose between spans survives as its own token', async () => {
  const line = 'She paused. "Go on," *he said*, and waited.';
  assert.deepEqual(kinds(line), ['text', 'dialogue', 'text', 'action', 'text']);
  assert.ok(texts(line).join('').includes('and waited.'));
});

test('every character of the line appears exactly once in the tokens', async () => {
  // The invariant that matters: tokenizing is a partition, so nothing the model
  // wrote is dropped and nothing is duplicated. Actions are the one exception,
  // and only by their two delimiters.
  const lines = [
    'plain prose with no spans at all',
    '"dialogue" then *action* then _narration_ then tail',
    'unclosed "quote runs to the end of the line',
    'unclosed *action runs to the end of the line',
    '',
    '   ',
    '"back" to "back" spans',
    'punctuation: -- ... !? (parens) [brackets]',
  ];
  for (const line of lines) {
    const rebuilt = tokenize(line)
      .map((t) => (t.kind === 'action' ? '*' + t.text + '*' : t.text))
      .join('');
    // An action written with underscores rebuilds with asterisks, so compare on
    // a form where the two markers are interchangeable.
    assert.equal(rebuilt.replace(/[*_]/g, '#'), line.replace(/[*_]/g, '#'), line);
  }
});

test('an unterminated span is left as plain text, not swallowed', async () => {
  assert.deepEqual(kinds('he said "and then nothing'), ['text']);
  assert.deepEqual(kinds('*he reaches for it'), ['text']);
});

test('an empty line tokenizes to nothing', async () => {
  assert.deepEqual(tokenize(''), []);
});

test('the shared pattern does not carry state between calls', async () => {
  // SPAN_RE is a module-level /g regex, so a leaked lastIndex would make the
  // SECOND call on the same input return something different from the first.
  const line = '"one" and *two*';
  const first = tokenize(line);
  for (let i = 0; i < 5; i++) assert.deepEqual(tokenize(line), first, 'call ' + i);
});

// ---- Blocks --------------------------------------------------------------

test('blank lines separate paragraphs and extra ones do not widen the gap', async () => {
  assert.deepEqual(toBlocks('one\n\ntwo\n\n\n\nthree'), ['one', 'two', 'three']);
});

test('a single newline stays inside its block', async () => {
  assert.deepEqual(toBlocks('one\ntwo'), ['one\ntwo']);
});

test('the literal words blank line never reach the reader', async () => {
  // The model writes the marker instead of leaving an empty line often enough
  // that it has to be stripped rather than styled.
  for (const marker of ['blank line', 'BLANK LINE', '(blank line)', 'Blank Lines']) {
    const out = toBlocks('before\n' + marker + '\nafter');
    assert.ok(!out.join('\n').toLowerCase().includes('blank'), marker);
    assert.ok(out.join(' ').includes('before'), marker);
    assert.ok(out.join(' ').includes('after'), marker);
  }
});

test('empty and null input render nothing rather than throwing', async () => {
  for (const input of [undefined, null, '', '   ', '\n\n\n']) {
    assert.deepEqual(toBlocks(input), [], JSON.stringify(input));
  }
});

// ---- Forced splitting ----------------------------------------------------

const sentence = (n) => 'This is sentence number ' + n + ', long enough to matter here.';
const wall = (n) => Array.from({ length: n }, (_, i) => sentence(i + 1)).join(' ');

test('a short block is left exactly as it is', async () => {
  const short = sentence(1) + ' ' + sentence(2);
  assert.ok(short.length < 240);
  assert.deepEqual(splitLongBlock(short), [short]);
});

test('a block the model formatted itself is never re-split', async () => {
  // Its own line breaks are the signal that the formatting was deliberate.
  const formatted = wall(8).replace('. This is sentence number 4', '.\nThis is sentence number 4');
  assert.ok(formatted.length > 240);
  assert.ok(formatted.includes('\n'));
  assert.deepEqual(splitLongBlock(formatted), [formatted]);
});

test('a long wall of text is split two sentences to a paragraph', async () => {
  const out = splitLongBlock(wall(6));
  assert.equal(out.length, 3);
  for (const p of out) assert.ok(p.startsWith('This is sentence'), p);
  assert.equal(out.join(' '), wall(6), 'no text is lost or reordered');
});

test('an odd sentence count leaves a one-sentence tail, not an empty paragraph', async () => {
  const out = splitLongBlock(wall(5));
  assert.equal(out.length, 3);
  assert.equal(out[2], sentence(5));
});

test('a sentence end inside a span is not a split point', async () => {
  // Tearing a line of dialogue in half across two paragraphs is the failure
  // this guards: the terminator belongs to the character, not to the layout.
  const block = 'He waited. "Is that it? Is that really all of it? You are sure?" '
    + 'She said nothing for a while. *He looked away. He counted. He gave up.* '
    + 'The room stayed quiet for a long time after that, and neither of them moved. '
    + 'Outside, the rain kept on against the window, steady and entirely indifferent.';
  assert.ok(block.length > 240);
  for (const p of splitLongBlock(block)) {
    assert.equal((p.match(/"/g) || []).length % 2, 0, 'quotes balanced in: ' + p);
    assert.equal((p.match(/\*/g) || []).length % 2, 0, 'actions balanced in: ' + p);
  }
});

test('a long block with nothing safe to split on is left whole', async () => {
  const unsplittable = '"' + 'and on and on it goes, '.repeat(20) + '"';
  assert.ok(unsplittable.length > 240);
  assert.deepEqual(splitLongBlock(unsplittable), [unsplittable]);
});

test('a decimal point does not cut a paragraph in half', async () => {
  // A terminator only counts when a space follows it, which is what keeps 3.5
  // from reading as the end of a sentence.
  const out = splitLongBlock('It took 3.5 seconds. ' + wall(4));
  assert.ok(out[0].startsWith('It took 3.5 seconds.'), out[0]);
});

test('splitting never drops or duplicates a character', async () => {
  for (const n of [2, 3, 7, 12]) {
    const block = wall(n);
    assert.equal(splitLongBlock(block).join(' '), block, n + ' sentences');
  }
});

test('toBlocks applies the split, so the component sees final paragraphs', async () => {
  const out = toBlocks(wall(6));
  assert.equal(out.length, 3, 'one long block became three paragraphs');
});
