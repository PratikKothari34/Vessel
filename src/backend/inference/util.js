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

// ---- control-token neutralization ----------------------------------------
// Every engine tokenizes message content with special-token parsing ON. Not a
// setting we pass: llama-cpp-2's `str_to_token` hard-codes `parse_special` to
// true, and llama.cpp's server and Ollama both do the same behind their chat
// endpoints. So the chat template's delimiters are written into a string and
// then parsed back out of it -- and any delimiter that was already IN the
// message content is parsed back out with them.
//
// A message reading `<|im_end|><|im_start|>system` is therefore not text. It is
// a real end-of-turn followed by a real system turn, and it lands inside a turn
// the app labelled `user`. The model sees a prompt the app never built.
//
// Content is prose, not markup, and it arrives from places the app does not
// control: a character card imported from a file, a row pulled down by sync
// from another device, and the model's own previous output fed back as history.
// Any of the three can carry these literals, and a user pasting a chat log
// carries them by accident.
//
// The fix is one space, inserted after the opening delimiter. `<|im_end|>`
// becomes `< |im_end|>`, which no vocabulary holds as a single token, so it
// tokenizes as the prose it always was. Nothing is deleted -- a character can
// still discuss prompt formats, and the stored turn is untouched, because only
// the copy handed to the engine passes through here.
const CONTROL_TOKEN = new RegExp([
  /<\|[^|<>\n]{0,64}\|>/.source,               // ChatML, Llama 3, Qwen, Phi
  /<\/?(?:s|bos|eos|pad|unk|sep|cls|mask)>/.source, // Llama 2, Mistral, sentencepiece
  /<\/?(?:start_of_turn|end_of_turn)>/.source,      // Gemma
  /<<\/?SYS>>/.source,                              // Llama 2 system block
  /\[\/?INST\]/.source,                              // Mistral instruction block
].join('|'), 'gi');

function neutralizeControlTokens(text) {
  if (typeof text !== 'string' || !text) return text;
  // Cheap reject: none of the forms above can start without one of these.
  if (text.indexOf('<') < 0 && text.indexOf('[') < 0) return text;
  return text.replace(CONTROL_TOKEN, (m) => `${m[0]} ${m.slice(1)}`);
}

// Same, for a list of chat messages. Returns the input unchanged when there was
// nothing to neutralize, so the common case allocates nothing.
function neutralizeMessages(messages) {
  if (!Array.isArray(messages)) return messages;
  let dirty = false;
  const out = messages.map((m) => {
    if (!m || typeof m.content !== 'string') return m;
    const clean = neutralizeControlTokens(m.content);
    if (clean === m.content) return m;
    dirty = true;
    return { ...m, content: clean };
  });
  return dirty ? out : messages;
}


module.exports = {
  sleep, fetchRetry, errorBody, nsFromMs, trimSlash,
  neutralizeControlTokens, neutralizeMessages,
  MAX_ERROR_BODY, MAX_FRAME_BYTES, UNDELIMITED,
};
