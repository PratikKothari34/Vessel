'use strict';

/**
 * The embedding blob codec.
 *
 * This is the one piece of the system where a silent bug is unrecoverable: the
 * blobs are already on disk, so a codec that writes something the decoder reads
 * differently corrupts every archived turn in the story rather than failing
 * loudly. The tests below therefore check the byte layout itself, not just the
 * round trip, and they check that the two formats stay mutually distinguishable.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const DB_PATH = path.resolve(__dirname, '..', '..', 'src', 'backend', 'db.js');

// db.js reads EMBED_QUANTIZE at module scope, so the f32 fallback can only be
// exercised by re-requiring it under a different environment.
function loadDb(env = {}) {
  const saved = {};
  for (const [k, v] of Object.entries(env)) { saved[k] = process.env[k]; process.env[k] = v; }
  delete require.cache[DB_PATH];
  const mod = require(DB_PATH);
  for (const [k, v] of Object.entries(saved)) {
    if (v === undefined) delete process.env[k]; else process.env[k] = v;
  }
  delete require.cache[DB_PATH];
  return mod;
}

const db = loadDb({ EMBED_QUANTIZE: '1' });
const dbF32 = loadDb({ EMBED_QUANTIZE: '0' });

// A realistic embedding: 768 components, unit norm, mean |component| ~ 1/sqrt(768).
function unitVector(dim = 768, seed = 12345) {
  let s = seed >>> 0;
  const v = new Array(dim);
  let norm = 0;
  for (let i = 0; i < dim; i++) {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    const x = (s / 0xffffffff) * 2 - 1;
    v[i] = x;
    norm += x * x;
  }
  norm = Math.sqrt(norm);
  for (let i = 0; i < dim; i++) v[i] /= norm;
  return v;
}

function cosine(a, b) {
  let dot = 0; let na = 0; let nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

// Dequantize a stored blob the way production does: decodeEmbeddingInt8 is the
// only decoder the app ships, and it hands back int8 components rather than
// floats because cosine does not need the scale. The encode-side properties
// below are still about direction, so recover the float view through it and
// compare directions -- that keeps the coverage pointed at shipping code
// instead of at a decoder nothing but a test would call.
function dequantize(blob, scale = null) {
  const mat = new Int8Array(db.EMBED_DIM);
  if (!db.decodeEmbeddingInt8(blob, mat, 0)) return null;
  const s = scale === null ? 1 : scale;
  return Array.from(mat, (q) => q * s);
}

test('int8 blob has the documented layout', () => {
  const v = unitVector();
  const buf = db.encodeEmbedding(v);
  assert.equal(buf[0], 0xe0, 'magic byte');
  assert.equal(buf[1], 0x01, 'format version');
  assert.equal(buf.byteLength, 6 + 768, '6-byte header + one byte per component');
  assert.ok(buf.readFloatLE(2) > 0, 'per-vector scale is positive');
  // 774 is not a multiple of 4, so it can never be mistaken for an f32 blob.
  assert.notEqual(buf.byteLength % 4, 0);
});

test('int8 round trip keeps cosine within the documented error', () => {
  const v = unitVector();
  const back = dequantize(db.encodeEmbedding(v));
  assert.equal(back.length, 768);
  const err = 1 - cosine(v, back);
  assert.ok(err < 0.002, `cosine moved by ${err}, documented bound is 0.002`);
});

test('int8 quantization is per-vector, so a tiny vector keeps its direction', () => {
  // Every component ~1e-6. A global scale would collapse this to zeros; the
  // per-vector scale must keep the full int8 range and therefore the direction.
  const v = unitVector().map((x) => x * 1e-6);
  const back = dequantize(db.encodeEmbedding(v));
  assert.ok(back.some((x) => x !== 0), 'vector did not collapse to zeros');
  assert.ok(1 - cosine(v, back) < 0.002);
});

test('an all-zero vector gets scale 1 rather than a divide by zero', () => {
  const v = new Array(768).fill(0);
  const buf = db.encodeEmbedding(v);
  assert.equal(buf.readFloatLE(2), 1, 'scale 1 rather than a divide by zero');
  const q8 = Array.from(new Int8Array(buf.buffer, buf.byteOffset + 6, 768));
  assert.ok(q8.every((x) => x === 0), 'components stay zero, not NaN or wrapped');
  // The retrieval decoder still accepts it: an int8 row cannot carry poison, and
  // a zero row scores 0 against everything, which is the honest answer for a
  // turn with no direction.
  assert.equal(db.decodeEmbeddingInt8(buf, new Int8Array(db.EMBED_DIM), 0), true);
});

test('components clamp to the int8 range instead of wrapping', () => {
  // maxAbs sits at both ends, so the middle components quantize well inside
  // range; the check is that nothing ever wraps to the opposite sign.
  const v = new Array(768).fill(0.5);
  v[0] = 1; v[1] = -1; v[2] = 0.5; v[3] = -0.5; v[767] = 1;
  const back = dequantize(db.encodeEmbedding(v));
  assert.ok(back[0] > 0 && back[1] < 0 && back[2] > 0 && back[3] < 0);
  assert.ok(back.every((x) => x >= -127 && x <= 127), 'nothing wrapped past the range');
});

test('legacy f32 blobs still decode when int8 is the write format', () => {
  const v = unitVector(768, 777);
  const legacy = dbF32.encodeEmbedding(v);
  assert.equal(legacy.byteLength, 768 * 4);
  // Decoded by the CURRENT decoder, which is what an old row gets after upgrade.
  const back = dequantize(legacy);
  assert.equal(back.length, 768);
  assert.ok(1 - cosine(v, back) < 0.002, 'direction survives the f32 -> int8 fold');
});

test('decode accepts Buffer, Uint8Array and ArrayBuffer', () => {
  const v = unitVector(768, 99);
  const buf = db.encodeEmbedding(v);
  const fromBuffer = dequantize(buf);
  const fromU8 = dequantize(new Uint8Array(buf));
  const fromAb = dequantize(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  assert.deepEqual(fromU8, fromBuffer);
  assert.deepEqual(fromAb, fromBuffer);
});

test('the codec is stable across encodes', () => {
  const v = unitVector(768, 424242);
  assert.deepEqual(db.encodeEmbedding(v), db.encodeEmbedding(v));
});

// ---- decodeEmbeddingInt8: the retrieval cache's decoder -------------------
// It skips the dequantize pass entirely and writes int8 straight into a shared
// matrix. That makes it the one decoder whose rejections matter for ranking:
// anything it lets through is scored, and a non-finite component that got in
// would make every score against that row NaN.

test('the int8 decoder writes a row at its offset and leaves neighbours alone', () => {
  const v = unitVector();
  const blob = db.encodeEmbedding(v);
  const mat = new Int8Array(3 * db.EMBED_DIM).fill(7);
  assert.equal(db.decodeEmbeddingInt8(blob, mat, db.EMBED_DIM), true);

  const body = new Int8Array(blob.buffer, blob.byteOffset + 6, db.EMBED_DIM);
  for (let i = 0; i < db.EMBED_DIM; i++) {
    assert.equal(mat[db.EMBED_DIM + i], body[i], `component ${i}`);
  }
  assert.equal(mat[0], 7, 'wrote before its slot');
  assert.equal(mat[2 * db.EMBED_DIM], 7, 'wrote past its slot');
});

test('a legacy f32 row is quantized on the way in, keeping its direction', () => {
  const v = unitVector(768, 777);
  const blob = dbF32.encodeEmbedding(v);
  const mat = new Int8Array(db.EMBED_DIM);
  assert.equal(db.decodeEmbeddingInt8(blob, mat, 0), true);
  assert.ok(cosine(Array.from(mat), v) > 0.999, 'direction lost in quantization');
});

test('the int8 decoder rejects a legacy row carrying a non-finite component', () => {
  // The whole reason it returns a boolean. int8 cannot hold Infinity, so this is
  // the only door poison can come through.
  for (const bad of [Infinity, -Infinity, NaN]) {
    const v = unitVector();
    v[13] = bad;
    const mat = new Int8Array(db.EMBED_DIM);
    assert.equal(db.decodeEmbeddingInt8(dbF32.encodeEmbedding(v), mat, 0), false, `${bad}`);
  }
});

test('the int8 decoder rejects rows of the wrong width, both formats', () => {
  const mat = new Int8Array(db.EMBED_DIM);
  assert.equal(db.decodeEmbeddingInt8(db.encodeEmbedding(unitVector(64)), mat, 0), false, 'short int8');
  assert.equal(db.decodeEmbeddingInt8(dbF32.encodeEmbedding(unitVector(64)), mat, 0), false, 'short f32');
  assert.equal(db.decodeEmbeddingInt8(Buffer.alloc(0), mat, 0), false, 'empty');
  assert.equal(db.decodeEmbeddingInt8(null, mat, 0), false, 'missing');
});

test('the int8 decoder rejects a blob whose scale header is corrupt', () => {
  const blob = Buffer.from(db.encodeEmbedding(unitVector()));
  blob.writeFloatLE(NaN, 2);
  assert.equal(db.decodeEmbeddingInt8(blob, new Int8Array(db.EMBED_DIM), 0), false);
});

test('the int8 decoder survives an unaligned slice of a pooled buffer', () => {
  // Same hazard the f32 decoder guards: a driver can hand back a BLOB whose
  // byteOffset is not 4-aligned, which makes a Float32Array view throw.
  const v = unitVector();
  const blob = dbF32.encodeEmbedding(v);
  const pool = Buffer.alloc(blob.byteLength + 3);
  blob.copy(pool, 3);
  const unaligned = pool.subarray(3);
  assert.notEqual(unaligned.byteOffset % 4, 0, 'test did not actually misalign');
  const mat = new Int8Array(db.EMBED_DIM);
  assert.equal(db.decodeEmbeddingInt8(unaligned, mat, 0), true);
  assert.ok(cosine(Array.from(mat), v) > 0.999);
});
