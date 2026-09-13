'use strict';

/**
 * The Modelfile parser.
 *
 * This file is the single source of truth for what the model IS -- the global
 * roleplay rules and the sampling defaults. Ollama bakes them in at
 * `ollama create` time; llama-server knows nothing about any of it and has to
 * be handed the same values. A parser bug here does not throw, it produces a
 * different product on one backend: no SYSTEM means the model starts refusing
 * and writing the user's actions.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..', '..');
const modelfile = require(path.join(ROOT, 'src', 'backend', 'inference', 'modelfile.js'));

test('the repo Modelfile parses and carries a SYSTEM block', () => {
  const m = modelfile.load();
  assert.equal(m.found, true, `expected a Modelfile at ${m.path}`);
  assert.ok(m.from, 'FROM is required for ollama create to work');
  assert.ok(m.system && m.system.length > 100, 'SYSTEM is what suppresses refusals on both backends');
});

test('a triple-quoted SYSTEM keeps its own markup', () => {
  // A `#` or a PARAMETER-looking line inside the prompt is prose, not markup.
  const m = modelfile.parse([
    'FROM ./model.gguf',
    'SYSTEM """',
    'You are a character.',
    '# not a comment, this is part of the prompt',
    'PARAMETER temperature 9 -- also prose',
    '"""',
    'PARAMETER temperature 0.85',
  ].join('\n'));
  assert.match(m.system, /# not a comment/);
  assert.match(m.system, /PARAMETER temperature 9/);
  assert.equal(m.params.temperature, 0.85, 'the real PARAMETER outside the block still applies');
});

test('numbers are coerced, text is not', () => {
  const m = modelfile.parse([
    'PARAMETER temperature 0.85',
    'PARAMETER top_k 40',
    'PARAMETER num_ctx 12288',
    'PARAMETER mirostat_tau -5.0',
    'PARAMETER some_name gemma3',
  ].join('\n'));
  assert.equal(m.params.temperature, 0.85);
  assert.equal(m.params.top_k, 40);
  assert.equal(m.params.num_ctx, 12288);
  assert.equal(m.params.mirostat_tau, -5);
  assert.equal(m.params.some_name, 'gemma3', 'a version-like token stays a string');
});

test('stop is the one key that accumulates', () => {
  const m = modelfile.parse([
    'PARAMETER stop "<|im_end|>"',
    'PARAMETER stop "User:"',
    'PARAMETER stop "Assistant:"',
  ].join('\n'));
  assert.deepEqual(m.params.stop, ['<|im_end|>', 'User:', 'Assistant:']);
});

test('a single stop is still a list, so consumers never branch on arity', () => {
  const m = modelfile.parse('PARAMETER stop "<|im_end|>"');
  assert.deepEqual(m.params.stop, ['<|im_end|>']);
});

test('a repeated non-stop PARAMETER is last-wins, as Ollama treats it', () => {
  // It must not become an array: MODEL_PARAMS is spread straight into the
  // llama.cpp request body, and temperature: [0.5, 0.9] is not a temperature.
  const m = modelfile.parse('PARAMETER temperature 0.5\nPARAMETER temperature 0.9');
  assert.equal(m.params.temperature, 0.9);
});

test('comments, blank lines and casing are handled', () => {
  const m = modelfile.parse([
    '# a comment',
    '',
    'from ./lower.gguf',
    '   parameter   TOP_P   0.9   ',
  ].join('\n'));
  assert.equal(m.from, './lower.gguf');
  assert.equal(m.params.top_p, 0.9, 'parameter keys normalise to lower case');
});

test('quoted values are unquoted and escapes resolved', () => {
  const m = modelfile.parse('SYSTEM "line one\\nline two with a \\"quote\\""');
  assert.equal(m.system, 'line one\nline two with a "quote"');
});

test('CRLF files parse the same as LF ones', () => {
  const lf = modelfile.parse('FROM ./m.gguf\nPARAMETER top_k 40');
  const crlf = modelfile.parse('FROM ./m.gguf\r\nPARAMETER top_k 40');
  assert.deepEqual(crlf, lf);
});

test('a missing file reports found:false instead of throwing', () => {
  const m = modelfile.load(path.join(os.tmpdir(), 'definitely-not-a-modelfile-xyz'));
  assert.equal(m.found, false);
  assert.equal(m.system, null);
  assert.deepEqual(m.params, {});
  assert.ok(m.error, 'the reason is kept so the banner can explain itself');
});

test('LLAMA_MODELFILE overrides the default path', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vessel-mf-'));
  const p = path.join(dir, 'Modelfile');
  fs.writeFileSync(p, 'FROM ./x.gguf\nSYSTEM "override"\n', 'utf8');
  const saved = process.env.LLAMA_MODELFILE;
  process.env.LLAMA_MODELFILE = p;
  try {
    const m = modelfile.load();
    assert.equal(m.found, true);
    assert.equal(m.system, 'override');
    assert.equal(m.path, p);
  } finally {
    if (saved === undefined) delete process.env.LLAMA_MODELFILE; else process.env.LLAMA_MODELFILE = saved;
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('an explicit argument beats the environment variable', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vessel-mf-'));
  const a = path.join(dir, 'A');
  const b = path.join(dir, 'B');
  fs.writeFileSync(a, 'SYSTEM "from-arg"\n', 'utf8');
  fs.writeFileSync(b, 'SYSTEM "from-env"\n', 'utf8');
  const saved = process.env.LLAMA_MODELFILE;
  process.env.LLAMA_MODELFILE = b;
  try {
    assert.equal(modelfile.load(a).system, 'from-arg');
  } finally {
    if (saved === undefined) delete process.env.LLAMA_MODELFILE; else process.env.LLAMA_MODELFILE = saved;
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('an empty file parses to an empty model, not to garbage', () => {
  const m = modelfile.parse('');
  assert.deepEqual(m, { from: null, system: null, template: null, params: {} });
});
