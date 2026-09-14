'use strict';

// Shared helpers for the inference backends.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Largest a stream buffer may grow while waiting for a delimiter.
//
// Both readers accumulate bytes until they see the end of a line or a frame. A
// well-behaved engine sends one small object per token, so the buffer never
// holds more than a few hundred bytes -- but nothing in the protocol promises a
// delimiter will ever arrive, and the engine host is a user-editable field.
// Point it at something that streams without one and the buffer grows until the
// process dies, with no error to explain it.
const MAX_FRAME_BYTES = 1024 * 1024;

// What a stream that never delimits itself is told to say.
const UNDELIMITED =
  'The engine sent more than a megabyte with no frame boundary. It may not be a model server.';

// How much of a failed response is worth keeping. The body of an error becomes
// the detail the renderer shows; an engine host that is not an engine at all --
// a stale port, a proxy, a login page -- answers with a whole HTML document, and
// reading it in full both buffers it and puts it in front of the user.
const MAX_ERROR_BODY = 2048;

// Read a failed response, stopping once there is enough to name the failure.
//
// Reading it as JSON and falling back to text does not work: the first read
// disturbs the body, so the fallback always throws and the detail comes back
// empty -- which is exactly the case (a non-JSON error page) the fallback was
// there for. Read the bytes once, then decide what they are.
async function errorBody(res) {
  let text = '';
  try {
    const reader = res.body && res.body.getReader ? res.body.getReader() : null;
    if (!reader) {
      text = await res.text();
    } else {
      const decoder = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        text += decoder.decode(value, { stream: true });
        if (text.length >= MAX_ERROR_BODY) { await reader.cancel().catch(() => {}); break; }
      }
    }
  } catch { /* a body that cannot be read is simply no detail */ }
  if (text.length > MAX_ERROR_BODY) text = text.slice(0, MAX_ERROR_BODY) + '...';
  try { return JSON.parse(text); } catch { return text; }
}

// Retry only what is worth retrying: a 5xx or a transport error. A 4xx is a bad
// request and will stay bad, and an abort is the user saying stop -- both give
// up immediately.
//
// The give-up has to be a flag rather than a `throw` inside the try, which is
// what it used to be: the catch below is in the same block, so it swallowed the
// 4xx straight back into lastErr and the loop retried it anyway. Every "model
// not found" cost three requests and 1.2 s of backoff before reporting a 404 it
// knew on the first attempt, and a cancelled generation paid the same.
async function fetchRetry(url, opts, tries = 3) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    let fatal = false;
    try {
      const res = await fetch(url, opts);
      if (res.ok) return res;
      const body = await errorBody(res);
      const detail = `${res.status} ${typeof body === 'string' ? body : JSON.stringify(body)}`;
      if (res.status < 500) fatal = true;
      lastErr = new Error(detail);
    } catch (e) {
      lastErr = e;
      // A cancelled request must not be re-issued: the caller has already gone.
      if (e && (e.name === 'AbortError' || (opts && opts.signal && opts.signal.aborted))) fatal = true;
    }
    if (fatal) break;
    if (i < tries - 1) await sleep(400 * (i + 1));
  }
  throw lastErr;
}

// Ollama reports durations in nanoseconds; llama.cpp reports them in float
// milliseconds. metrics.js speaks nanoseconds, so llama-server converts here.
const nsFromMs = (ms) => (Number.isFinite(ms) && ms > 0 ? Math.round(ms * 1e6) : 0);

const trimSlash = (s) => String(s || '').replace(/\/+$/, '');

module.exports = {
  sleep, fetchRetry, errorBody, nsFromMs, trimSlash,
  MAX_ERROR_BODY, MAX_FRAME_BYTES, UNDELIMITED,
};
