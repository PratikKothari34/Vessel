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
const { normalizeInPlace, _absorbRows, _newCacheEntry } = _internals;

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

function cachedIds(e) {
  return Array.from(e.ids.subarray(0, e.n));
}

test('a row whose blob decodes to nonsense is not cached', () => {
  const e = _newCacheEntry();
  const poisoned = unit(1);
  poisoned[7] = Infinity;
  _absorbRows(e, [
    { id: 1, embedding: rawBlob(poisoned) },
    { id: 2, embedding: rawBlob(unit(0)) },
  ]);
  assert.deepEqual(cachedIds(e), [2], 'the poisoned row was cached');
  assert.equal(e.n, 1, 'and its vector with it');
});

test('a row of the wrong width is not cached either', () => {
  const e = _newCacheEntry();
  _absorbRows(e, [{ id: 1, embedding: rawBlob([1, 2, 3]) }]);
  assert.equal(e.n, 0);
});

test('a rejected row does not consume the slot the next row needs', () => {
  // The decode writes into the row slot before the row is known to be good, so
  // a rejection leaves debris behind. The next accepted row has to land on the
  // same slot and overwrite it, or every row after a bad one scores garbage.
  const e = _newCacheEntry();
  const good = unit(0);
  _absorbRows(e, [
    { id: 1, embedding: rawBlob(new Float32Array(EMBED_DIM)) }, // all zero: no direction
    { id: 2, embedding: rawBlob(good) },
  ]);
  assert.deepEqual(cachedIds(e), [2]);
  // Scoring the surviving row against its own direction must be a clean 1.
  const q = normalizeInPlace(Float32Array.from(good));
  let d = 0;
  for (let i = 0; i < EMBED_DIM; i++) d += q[i] * e.mat[i];
  assert.ok(Math.abs(d * e.norms[0] - 1) < 0.01, `debris survived: ${d * e.norms[0]}`);
});

test('the cache grows past its initial capacity without losing a row', () => {
  // Geometric growth copies the matrix; an off-by-one in the copy would corrupt
  // rows that were already resident rather than fail loudly.
  const e = _newCacheEntry();
  const start = e.cap;
  const rows = [];
  for (let i = 0; i < start + 5; i++) rows.push({ id: i + 1, embedding: rawBlob(unit(i)) });
  _absorbRows(e, rows);
  assert.equal(e.n, start + 5);
  assert.ok(e.cap > start, 'capacity never grew');
  assert.equal(e.ids[0], 1, 'the first row was lost in the copy');
  assert.equal(e.ids[e.n - 1], start + 5);
  // The very first row still has to score 1 against itself after the move.
  const q = normalizeInPlace(unit(0));
  let d = 0;
  for (let i = 0; i < EMBED_DIM; i++) d += q[i] * e.mat[i];
  assert.ok(Math.abs(d * e.norms[0] - 1) < 0.01, 'row 0 was corrupted by the grow');
});

test('scoring raw int8 against its own norm is the cosine, not an approximation', () => {
  // The cache skips dequantizing because cosine ignores the per-vector scale.
  // This pins that equivalence: the int8 score must match the score computed the
  // long way -- dequantize to f32, normalize, dot -- to within the quantization
  // error already present on disk, NOT to within something looser.
  const e = _newCacheEntry();
  const rows = [];
  for (let i = 0; i < 8; i++) rows.push({ id: i + 1, embedding: rawBlob(unit(i * 3)) });
  _absorbRows(e, rows);
  assert.equal(e.n, 8);

  const q = normalizeInPlace(unit(4.5));
  for (let r = 0; r < e.n; r++) {
    const off = r * EMBED_DIM;
    let fast = 0;
    for (let i = 0; i < EMBED_DIM; i++) fast += q[i] * e.mat[off + i];
    fast *= e.norms[r];

    // the long way, from the same int8 components
    const deq = new Float32Array(EMBED_DIM);
    for (let i = 0; i < EMBED_DIM; i++) deq[i] = e.mat[off + i] * 0.0037; // any positive scale
    normalizeInPlace(deq);
    let slow = 0;
    for (let i = 0; i < EMBED_DIM; i++) slow += q[i] * deq[i];

    assert.ok(Math.abs(fast - slow) < 1e-5, `row ${r}: ${fast} vs ${slow}`);
  }
});
