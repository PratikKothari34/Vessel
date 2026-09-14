'use strict';

/**
 * The vector arithmetic under retrieval.
 *
 * A stored embedding is bytes, and not all of them came from this machine: a
 * legacy row is a raw little-endian f32 array with no header to validate, and a
 * sync pull brings rows another device wrote. An infinite component in one of
 * them makes the norm infinite, every scaled component NaN, and every score
 * against that row NaN -- and NaN loses no comparison, so the row would sort
 * above real matches and then, as the k-th best, let every remaining row clear
 * the cutoff. One bad blob would replace retrieval with noise for the whole
 * conversation, silently. These tests pin the rejection that stops it.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const backend = path.resolve(__dirname, '..', '..', 'src', 'backend');
const { EMBED_DIM } = require(path.join(backend, 'db.js'));
const { _internals } = require(path.join(backend, 'memory.js'));
const { normalizeInPlace, _absorbRows } = _internals;

// The legacy on-disk shape: raw f32, no header. Chosen deliberately -- it is
// the format with nothing to validate, so it is the one poison arrives in.
function rawBlob(values) {
  const f = Float32Array.from(values);
  return Buffer.from(f.buffer, f.byteOffset, f.byteLength);
}

function unit(seed) {
  const v = new Float32Array(EMBED_DIM);
  for (let i = 0; i < EMBED_DIM; i++) v[i] = Math.sin(seed + i * 0.01);
  return v;
}

test('normalizing turns a dot product into a cosine', () => {
  const v = Float32Array.from([3, 4]);
  assert.ok(normalizeInPlace(v));
  assert.ok(Math.abs(v[0] - 0.6) < 1e-6);
  assert.ok(Math.abs(v[1] - 0.8) < 1e-6);
});

test('a vector with no direction is rejected rather than scaled', () => {
  assert.equal(normalizeInPlace(Float32Array.from([0, 0, 0, 0])), null, 'all zero');
  // Not f32::MAX: JS accumulates the norm as a double, so a component at the
  // top of the f32 range squares to a finite number here and normalizes fine.
  // The Rust core sums in f32 and rejects it. The divergence costs at most one
  // unreachable row on one of the two builds, which is not worth a wider type
  // on the hot loop to close.
  for (const bad of [Infinity, -Infinity, NaN]) {
    assert.equal(
      normalizeInPlace(Float32Array.from([1, bad, 1])),
      null,
      `${bad} was accepted`,
    );
  }
});

test('a row whose blob decodes to nonsense is not cached', () => {
  const e = { ids: [], vecs: [] };
  const poisoned = unit(1);
  poisoned[7] = Infinity;
  _absorbRows(e, [
    { id: 1, embedding: rawBlob(poisoned) },
    { id: 2, embedding: rawBlob(unit(0)) },
  ]);
  assert.deepEqual(e.ids, [2], 'the poisoned row was cached');
  assert.equal(e.vecs.length, 1, 'and its vector with it');
});

test('a row of the wrong width is not cached either', () => {
  const e = { ids: [], vecs: [] };
  _absorbRows(e, [{ id: 1, embedding: rawBlob([1, 2, 3]) }]);
  assert.deepEqual(e.ids, []);
});
