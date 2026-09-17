'use strict';

/**
 * What a stored character row turns into on the way out.
 *
 * The clamps in createCharacter only cover rows THIS process wrote. Rows also
 * arrive from the sync remote, written by another device or an older build, and
 * a user can restore a backup. None of those went through the write path, so
 * the read path has to re-clean the fields that carry real power: sampling
 * spreads straight into the engine's options, avatar lands in an <img src>, and
 * the lists are rendered.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const { _internals } = require(path.resolve(__dirname, '..', '..', 'src', 'backend', 'characters.js'));
const { rowToCharacter } = _internals;

// A row as the sync engine would hand it back: every column a string, and none
// of it checked by anything in this process.
function row(over = {}) {
  return {
    id: 'c1', name: 'Drift', avatar: '', tagline: '', about: '', persona: '', greeting: '',
    chat_starters: '[]', tags: '[]', sampling: '{}', response_style: 'balanced',
    created_at: '2026-01-01T00:00:00.000Z', updated_at: '2026-01-01T00:00:00.000Z',
    ...over,
  };
}

test('an unclamped sampling value from a planted row cannot reach the engine', () => {
  const got = rowToCharacter(row({
    sampling: JSON.stringify({ num_ctx: 99999999, temperature: 99, top_k: -40, evil: 'rm -rf' }),
  }));
  assert.equal(got.sampling.evil, undefined, 'unknown keys are dropped');
  assert.equal(got.sampling.num_ctx, 131072);
  assert.equal(got.sampling.temperature, 2);
  assert.equal(got.sampling.top_k, 0);
});

test('a hostile avatar from a planted row never reaches an img src', () => {
  assert.equal(rowToCharacter(row({ avatar: 'file:///C:/Windows/win.ini' })).avatar, '');
  assert.equal(rowToCharacter(row({ avatar: 'javascript:alert(1)' })).avatar, '');
  assert.equal(rowToCharacter(row({ avatar: 'data:text/html,<script>' })).avatar, '');
  assert.equal(
    rowToCharacter(row({ avatar: 'https://example.test/a.png' })).avatar,
    'https://example.test/a.png',
    'a legitimate avatar still survives',
  );
});

test('a planted list is capped in both count and entry length', () => {
  const got = rowToCharacter(row({
    tags: JSON.stringify([{ not: 'a string' }, 'x'.repeat(4000), '', '  spaced  ']),
    chat_starters: JSON.stringify(new Array(500).fill('s')),
  }));
  assert.deepEqual(got.tags, ['x'.repeat(40), 'spaced'], 'objects and empties drop, the rest caps');
  assert.equal(got.chat_starters, undefined);
  assert.equal(got.chatStarters.length, 12);
});

test('a corrupt column falls back to the empty value rather than throwing', () => {
  const got = rowToCharacter(row({ sampling: 'not json', tags: '{', chat_starters: '"a string"' }));
  assert.deepEqual(got.sampling, {});
  assert.deepEqual(got.tags, []);
  assert.deepEqual(got.chatStarters, []);
});

test('an unknown response style still falls back, as it always has', () => {
  assert.equal(rowToCharacter(row({ response_style: 'obedient' })).responseStyle, 'balanced');
  assert.equal(rowToCharacter(row({ response_style: 'dialogue' })).responseStyle, 'dialogue');
});

test('an ordinary row passes through with its values intact', () => {
  const got = rowToCharacter(row({
    name: 'Ilse', persona: 'a quiet archivist', tagline: 'keeper of the stacks',
    tags: JSON.stringify(['calm', 'archivist']),
    sampling: JSON.stringify({ temperature: 0.85, num_ctx: 12288 }),
  }));
  assert.equal(got.name, 'Ilse');
  assert.equal(got.persona, 'a quiet archivist');
  assert.deepEqual(got.tags, ['calm', 'archivist']);
  assert.deepEqual(got.sampling, { temperature: 0.85, num_ctx: 12288 });
});
