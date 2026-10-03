// The renderer's only data seam.
//
// One transport: every call is a Tauri IPC command and the chat stream is a
// channel. The HTTP half that spoke to a Node/Express backend on :3001 was
// removed on 2026-10-03 along with the backend itself -- there is no HTTP
// listener anywhere in the Rust track, so there is nothing left to fall back
// to. If `window.__TAURI__` is missing the app is not running in its shell and
// cannot work at all, which is worth failing loudly about rather than papering
// over with a second code path that cannot succeed.
//
// The shapes below are the ones the Express contract defined, because the Rust
// commands were written against it. Keeping them is not legacy: components were
// built on them, and over IPC the delta arrives as text, so the per-token JSON
// parse the old SSE path needed is simply gone.

const TAURI = typeof window !== 'undefined' && window.__TAURI__ ? window.__TAURI__ : null;

// A command rejects with the serialized error struct ({ error, detail }).
// Flatten whatever comes back to one Error.
function toError(e) {
  if (e instanceof Error) return e;
  if (e && typeof e === 'object') return new Error(e.error || e.detail || 'Request failed.');
  return new Error(String(e || 'Request failed.'));
}

const NO_HOST = 'Vessel must run inside its application window. The IPC bridge is unavailable.';

// Rejects rather than throws. Callers are React handlers that attach .catch and
// have no try/catch around the call, so a synchronous throw here escapes to
// window.onerror instead of their error state -- every other failure on this
// seam arrives as a rejection, and this one has to match.
function invoke(cmd, args) {
  if (!TAURI) return Promise.reject(new Error(NO_HOST));
  return TAURI.core.invoke(cmd, args).catch((e) => {
    throw toError(e);
  });
}

export const api = {
  health: () => invoke('health'),

  listCharacters: () => invoke('list_characters').then((d) => d.characters),
  createCharacter: (c) => invoke('create_character', { patch: c }),
  updateCharacter: (id, c) => invoke('update_character', { id, patch: c }),
  deleteCharacter: (id) => invoke('delete_character', { id }),

  getSettings: () => invoke('get_settings'),
  // confirmClearToken must cross the bridge: save_settings refuses to delete the
  // stored token without it, so dropping the field here would make the UI's
  // confirmed "turn off sync" fail instead of apply.
  saveSettings: (patch) =>
    invoke('save_settings', {
      tursoUrl: patch.tursoUrl,
      tursoToken: patch.tursoToken,
      confirmClearToken: patch.confirmClearToken,
    }),

  listConversations: (characterId) =>
    invoke('list_conversations', { characterId: characterId || null }).then((d) => d.conversations),
  getConversation: (id) => invoke('get_conversation', { id }),
  deleteConversation: (id) => invoke('delete_conversation', { id }),
  setActiveVariant: (id, turnId, variantId) =>
    invoke('set_active_variant', { id, turnId, variantId }),

  // Restarting is how a sync credential change takes effect: swapping the live
  // DB handle mid-run would race in-flight bookkeeping.
  relaunch: () => invoke('relaunch'),
};

/**
 * Stream a chat reply.
 *
 * @param {object} payload { characterId?, conversationId?, messages, regenerate?, director? }
 * @param {object} handlers { onMeta(meta), onToken(text, full), onDone(fullText), onError(msg) }
 * @returns {function} abort() -- stops the in-flight generation.
 */
export function streamChat(payload, { onMeta, onToken, onDone, onError } = {}) {
  if (!TAURI) {
    const err = new Error(NO_HOST);
    // Report asynchronously so the caller has returned and wired its handlers.
    Promise.resolve().then(() => onError && onError(err.message));
    return () => {};
  }

  const channel = new TAURI.core.Channel();
  let full = '';
  // A stop can land before the first event, when the model is still loading and
  // no conversation id has come back yet. Remember it and cancel the moment the
  // id arrives, rather than dropping the request on the floor.
  let convId = payload.conversationId || null;
  let stopped = false;
  let finished = false;

  const stop = () => {
    if (convId) invoke('cancel_chat', { conversationId: convId }).catch(() => {});
  };

  channel.onmessage = (evt) => {
    switch (evt.type) {
      case 'meta':
        convId = evt.conversationId;
        if (stopped) stop();
        onMeta && onMeta({
          conversationId: evt.conversationId,
          characterId: evt.characterId,
          recalled: evt.recalled,
        });
        break;
      case 'chunk':
        full += evt.delta;
        onToken && onToken(evt.delta, full);
        break;
      case 'error':
        onError && onError(evt.error);
        break;
      case 'done':
        finished = true;
        onDone && onDone(full);
        break;
      default:
        break;
    }
  };

  invoke('chat', { request: payload, onEvent: channel }).catch((err) => {
    // A rejection means nothing was streamed -- the turn never started -- so
    // `done` will not arrive and this is the only report the caller gets.
    if (!finished) onError && onError(err.message);
  });

  return () => {
    stopped = true;
    stop();
  };
}
