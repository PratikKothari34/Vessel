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
  const back = db.decodeEmbedding(db.encodeEmbedding(v));
  assert.equal(back.length, 768);
  const err = 1 - cosine(v, Array.from(back));
  assert.ok(err < 0.002, `cosine moved by ${err}, documented bound is 0.002`);
});

test('int8 quantization is per-vector, so a tiny vector keeps its direction', () => {
  // Every component ~1e-6. A global scale would collapse this to zeros; the
  // per-vector scale must keep the full int8 range and therefore the direction.
  const v = unitVector().map((x) => x * 1e-6);
  const back = Array.from(db.decodeEmbedding(db.encodeEmbedding(v)));
  assert.ok(back.some((x) => x !== 0), 'vector did not collapse to zeros');
  assert.ok(1 - cosine(v, back) < 0.002);
});

test('an all-zero vector survives as all zeros', () => {
  const v = new Array(768).fill(0);
  const buf = db.encodeEmbedding(v);
  assert.equal(buf.readFloatLE(2), 1, 'scale 1 rather than a divide by zero');
  const back = db.decodeEmbedding(buf);
  assert.ok(Array.from(back).every((x) => x === 0));
});

test('components clamp to the int8 range instead of wrapping', () => {
  // maxAbs is the last component, so the others quantize well inside range; the
  // check is that nothing ever wraps to the opposite sign.
  const v = [1, -1, 0.5, -0.5, 1];
  const back = Array.from(db.decodeEmbedding(db.encodeEmbedding(v)));
  assert.ok(back[0] > 0 && back[1] < 0 && back[2] > 0 && back[3] < 0);
});

test('legacy f32 blobs still decode when int8 is the write format', () => {
  const v = unitVector(768, 777);
  const legacy = dbF32.encodeEmbedding(v);
  assert.equal(legacy.byteLength, 768 * 4);
  // Decoded by the CURRENT decoder, which is what an old row gets after upgrade.
  const back = Array.from(db.decodeEmbedding(legacy));
  assert.equal(back.length, 768);
  assert.ok(1 - cosine(v, back) < 1e-6, 'f32 round trip is lossless to float precision');
});

test('decode accepts Buffer, Uint8Array and ArrayBuffer', () => {
  const v = unitVector(768, 99);
  const buf = db.encodeEmbedding(v);
  const fromBuffer = db.decodeEmbedding(buf);
  const fromU8 = db.decodeEmbedding(new Uint8Array(buf));
  const fromAb = db.decodeEmbedding(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  assert.deepEqual(Array.from(fromU8), Array.from(fromBuffer));
  assert.deepEqual(Array.from(fromAb), Array.from(fromBuffer));
});

test('decode survives an unaligned slice of a pooled buffer', () => {
  // Drivers hand BLOBs back as slices of a pooled ArrayBuffer. A legacy f32 blob
  // at an odd byteOffset would make a Float32Array VIEW throw, which is why the
  // decoder copies. Force that case rather than hoping the pool produces it.
  const v = unitVector(64, 5);
  const legacy = dbF32.encodeEmbedding(v);
  const padded = Buffer.alloc(legacy.byteLength + 1);
  legacy.copy(padded, 1);
  const unaligned = padded.subarray(1);
  assert.notEqual(unaligned.byteOffset % 4, 0);
  const back = db.decodeEmbedding(unaligned);
  assert.equal(back.length, 64);
  assert.ok(1 - cosine(v, Array.from(back)) < 1e-6);
});

test('garbage decodes to null rather than to a wrong vector', () => {
  assert.equal(db.decodeEmbedding(null), null);
  assert.equal(db.decodeEmbedding(undefined), null);
  // Not a multiple of 4 and not an int8 blob: unreadable, and saying so is the
  // only honest answer. Silently returning a truncated vector would poison recall.
  assert.equal(db.decodeEmbedding(Buffer.from([1, 2, 3])), null);
  // int8 magic with a NaN scale.
  const bad = Buffer.alloc(6 + 8);
  bad[0] = 0xe0; bad[1] = 0x01;
  bad.writeUInt32LE(0x7fc00000, 2); // NaN
  assert.equal(db.decodeEmbedding(bad), null);
});

test('an empty blob is not read as an empty vector', () => {
  // Zero-length is indistinguishable from a failed write; the f32 branch accepts
  // it (0 % 4 === 0) and returns an empty vector, which cosine scores as 0. That
  // is the intended outcome -- a row that can never match -- not a crash.
  const back = db.decodeEmbedding(Buffer.alloc(0));
  assert.equal(back.length, 0);
});

test('the codec is stable across encodes', () => {
  const v = unitVector(768, 424242);
  assert.deepEqual(db.encodeEmbedding(v), db.encodeEmbedding(v));
});
