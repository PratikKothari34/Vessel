'use strict';

/**
 * Reads the repo `Modelfile`.
 *
 * On Ollama, `SYSTEM` and every `PARAMETER` line here are baked into the model
 * at `ollama create` time: the server applies them to every request whether or
 * not the client sends anything. llama-server loads the raw GGUF blob and knows
 * nothing about any of it -- the blob carries weights and a chat template, and
 * that is all.
 *
 * So a naive backend swap silently drops the global roleplay system prompt and
 * the sampling defaults, and the model starts refusing, moralizing, and writing
 * for the user. That is not a transport difference; it is a different product.
 *
 * The fix is to read the same file Ollama read, so there is still exactly one
 * source of truth for what the model is.
 */

const fs = require('fs');
const path = require('path');

const DEFAULT_PATH = path.resolve(__dirname, '..', '..', '..', 'Modelfile');

// A Modelfile value can be a bare token, a quoted string, or a """block""".
const TRIPLE = /^(SYSTEM|TEMPLATE|ADAPTER)\s+"""([\s\S]*?)"""/gim;

function unquote(s) {
  const t = String(s).trim();
  if (t.length >= 2 && t[0] === '"' && t[t.length - 1] === '"') {
    return t.slice(1, -1).replace(/\\n/g, '\n').replace(/\\"/g, '"');
  }
  return t;
}

function coerce(v) {
  const t = unquote(v);
  if (t === '') return t;
  const n = Number(t);
  return Number.isFinite(n) && /^[-+]?[0-9.]+(e[-+]?\d+)?$/i.test(t) ? n : t;
}

/**
 * @returns {{ from: string|null, system: string|null, template: string|null,
 *             params: Object, path: string, found: boolean }}
 */
function parse(text) {
  const out = { from: null, system: null, template: null, params: {} };

  // Pull the block-quoted directives out first, so a `#` or a PARAMETER-looking
  // line inside the system prompt is not mistaken for markup.
  let rest = String(text).replace(TRIPLE, (_m, key, body) => {
    const k = key.toUpperCase();
    if (k === 'SYSTEM') out.system = body.trim();
    else if (k === 'TEMPLATE') out.template = body;
    return '';
  });

  for (const line of rest.split(/\r?\n/)) {
    const l = line.trim();
    if (!l || l.startsWith('#')) continue;
    let m = /^FROM\s+(.+)$/i.exec(l);
    if (m) { out.from = unquote(m[1]); continue; }
    m = /^SYSTEM\s+(.+)$/i.exec(l);
    if (m) { out.system = unquote(m[1]); continue; }
    m = /^TEMPLATE\s+(.+)$/i.exec(l);
    if (m) { out.template = unquote(m[1]); continue; }
    m = /^PARAMETER\s+(\S+)\s+(.+)$/i.exec(l);
    if (m) {
      const key = m[1].toLowerCase();
      const val = coerce(m[2]);
      // `stop` is the one key Ollama allows more than once.
      if (key in out.params) {
        out.params[key] = [].concat(out.params[key], val);
      } else {
        out.params[key] = key === 'stop' ? [val] : val;
      }
      continue;
    }
  }
  return out;
}

function load(file) {
  const p = file || process.env.LLAMA_MODELFILE || DEFAULT_PATH;
  try {
    const parsed = parse(fs.readFileSync(p, 'utf8'));
    return { ...parsed, path: p, found: true };
  } catch (err) {
    return {
      from: null, system: null, template: null, params: {},
      path: p, found: false, error: err.message,
    };
  }
}

module.exports = { load, parse, DEFAULT_PATH };
