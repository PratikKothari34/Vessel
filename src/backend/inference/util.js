'use strict';

// Shared helpers for the inference backends.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Retry only what is worth retrying: a 5xx or a transport error. A 4xx is a bad
// request and will stay bad, so it throws on the first attempt.
async function fetchRetry(url, opts, tries = 3) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, opts);
      if (res.ok) return res;
      if (res.status < 500) throw new Error(`${res.status} ${await res.text().catch(() => '')}`);
      lastErr = new Error(`${res.status} ${await res.text().catch(() => '')}`);
    } catch (e) { lastErr = e; }
    if (i < tries - 1) await sleep(400 * (i + 1));
  }
  throw lastErr;
}

// Ollama reports durations in nanoseconds; llama.cpp reports them in float
// milliseconds. metrics.js speaks nanoseconds, so llama-server converts here.
const nsFromMs = (ms) => (Number.isFinite(ms) && ms > 0 ? Math.round(ms * 1e6) : 0);

const trimSlash = (s) => String(s || '').replace(/\/+$/, '');

module.exports = { sleep, fetchRetry, nsFromMs, trimSlash };
