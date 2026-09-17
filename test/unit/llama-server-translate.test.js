'use strict';

/**
 * The two pure transforms inside the llama-server backend.
 *
 * Both were exported for a one-off verification rig that proved the backend on
 * the real GPU. The rig is gone; the exports stayed, and with them the logic
 * went unverified -- which is the worse half of the trade, because both
 * transforms fail SILENTLY. A sampling key that llama.cpp spells differently is
 * dropped and the model runs at stock settings; a prompt-token count that
 * ignores the KV cache shrinks by whatever the cache absorbed, and that number
 * is the one the context-window size is chosen from.
 *
 * Requiring the backend reads the environment and the Modelfile. Neither opens
 * a socket or a database, so this file requires it at the top like any other.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const llama = require(path.resolve(__dirname, '..', '..', 'src', 'backend', 'inference', 'llama-server.js'));
const translate = llama._translateSampling;
const doneChunk = llama._doneChunk;

// ---- translateSampling ----------------------------------------------------

test('the sampling names llama.cpp spells differently are renamed, not dropped', () => {
  const out = translate({ num_predict: 512, repeat_penalty: 1.1, repeat_last_n: 64 });
  assert.equal(out.max_tokens, 512, 'the OpenAI-compatible endpoint calls it max_tokens');
  assert.equal(out.repeat_penalty, 1.1);
  assert.equal(out.repeat_last_n, 64);
  assert.ok(!('num_predict' in out), 'the Ollama spelling must not survive');
});

test('the names both engines share pass through unchanged', () => {
  const out = translate({ temperature: 0.7, top_p: 0.9, top_k: 40, min_p: 0.05, seed: 7 });
  assert.deepEqual(out, { temperature: 0.7, top_p: 0.9, top_k: 40, min_p: 0.05, seed: 7 });
});

test('a null or undefined value is skipped rather than sent as null', () => {
  // A character that clears a field leaves the key behind with no value. Sent
  // as null it is a type error at the engine, not a default.
  const out = translate({ temperature: null, top_p: undefined, top_k: 40 });
  assert.deepEqual(out, { top_k: 40 });
});

test('num_ctx is dropped with a warning, because the window is a launch flag', () => {
  const warned = [];
  const real = console.warn;
  console.warn = (m) => warned.push(String(m));
  try {
    const out = translate({ num_ctx: 8192, temperature: 0.7 });
    assert.deepEqual(out, { temperature: 0.7 });
  } finally {
    console.warn = real;
  }
  // warnOnce keeps a module-level set, so the message may already have been
  // spent by an earlier require. Either it warned here or it warned before --
  // what must never happen is num_ctx silently reaching the request body.
  if (warned.length) assert.match(warned[0], /num_ctx=8192 ignored/);
});

test('an option with no llama.cpp equivalent is dropped, not forwarded raw', () => {
  const key = 'not_a_real_sampling_option_' + Date.now();
  const warned = [];
  const real = console.warn;
  console.warn = (m) => warned.push(String(m));
  try {
    const out = translate({ [key]: 1, temperature: 0.7 });
    assert.deepEqual(out, { temperature: 0.7 });
  } finally {
    console.warn = real;
  }
  assert.equal(warned.length, 1, 'a fresh key warns exactly once');
  assert.match(warned[0], new RegExp(key));
});

test('an empty options object translates to an empty body fragment', () => {
  assert.deepEqual(translate(), {});
  assert.deepEqual(translate({}), {});
});

test('the Modelfile defaults carry no num_ctx into the request body', () => {
  // The window is a launch flag for this backend. Leaving num_ctx in the
  // defaults would warn on every single startup about a value no character
  // asked for, and the warning is the only sign a real one was ignored.
  assert.ok(!('num_ctx' in llama._modelParams), 'stripped before translation');
  assert.deepEqual(translate(llama._modelParams).num_ctx, undefined);
});

// ---- doneChunk ------------------------------------------------------------

test('the prompt token count is the processed tokens PLUS the cached ones', () => {
  // llama.cpp reports only what it had to process; Ollama reports the whole
  // prompt. Reporting only prompt_n would make a well-cached conversation look
  // like it uses a fraction of the window it actually fills.
  const c = doneChunk({ timings: { prompt_n: 120, cache_n: 3880, predicted_n: 64 } });
  assert.equal(c.prompt_eval_count, 4000);
  assert.equal(c.cached_tokens, 3880, 'the measured reuse is carried through');
  assert.equal(c.eval_count, 64);
});

test('with no cache reported, the processed count stands alone', () => {
  const c = doneChunk({ timings: { prompt_n: 120, predicted_n: 8 } });
  assert.equal(c.prompt_eval_count, 120);
  assert.ok(!('cached_tokens' in c), 'absent is not zero -- zero would read as a cold cache');
});

test('usage is the fallback when timings carry no counts', () => {
  const c = doneChunk({ usage: { prompt_tokens: 900, completion_tokens: 30 } });
  assert.equal(c.prompt_eval_count, 900);
  assert.equal(c.eval_count, 30);
});

test('timings win over usage when both are present', () => {
  const c = doneChunk({
    timings: { prompt_n: 100, cache_n: 20, predicted_n: 5 },
    usage: { prompt_tokens: 999, completion_tokens: 999 },
  });
  assert.equal(c.prompt_eval_count, 120);
  assert.equal(c.eval_count, 5);
});

test('durations are nanoseconds, and load_duration is always zero', () => {
  const c = doneChunk({ timings: { prompt_ms: 2, predicted_ms: 3, prompt_n: 1, predicted_n: 1 } });
  assert.equal(c.prompt_eval_duration, 2e6);
  assert.equal(c.eval_duration, 3e6);
  assert.equal(c.total_duration, 5e6);
  assert.equal(c.load_duration, 0, 'a non-zero value would mean the server restarted under us');
});

test('a chunk with nothing measurable is still a well-formed done chunk', () => {
  const c = doneChunk({});
  assert.equal(c.done, true);
  assert.equal(c.done_reason, 'stop');
  assert.equal(c.message.role, 'assistant');
  assert.equal(c.message.content, '');
  assert.equal(c.total_duration, 0);
  assert.ok(!('prompt_eval_count' in c) && !('eval_count' in c), 'unknown is absent, not zero');
});

test('the finish reason is carried when the engine gives one', () => {
  assert.equal(doneChunk({ finishReason: 'length' }).done_reason, 'length');
});
