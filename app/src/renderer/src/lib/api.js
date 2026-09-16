// The renderer's only data seam.
//
// Two hosts, one surface. Under Tauri every call is an IPC command and the chat
// stream is a channel; under Electron it is the loopback HTTP backend and SSE.
// Which one is live is decided once, at load, from what the host injected -- so
// no component below this file knows or cares.
//
// The two transports return the SAME shapes, because the Rust commands were
// written against the Express contract. Where they differ is cost: over IPC the
// delta arrives as text, so the per-token JSON parse the SSE path needs is gone.

const TAURI = typeof window !== 'undefined' && window.__TAURI__ ? window.__TAURI__ : null;

const BASE =
  (typeof window !== 'undefined' && window.scenario && window.scenario.backendUrl) ||
  'http://localhost:3001';

// A command rejects with the serialized error struct ({ error, detail }); HTTP
// rejects with a body of the same shape. Flatten both to one Error.
function toError(e) {
  if (e instanceof Error) return e;
  if (e && typeof e === 'object') return new Error(e.error || e.detail || 'Request failed.');
  return new Error(String(e || 'Request failed.'));
}

function invoke(cmd, args) {
  return TAURI.core.invoke(cmd, args).catch((e) => {
    throw toError(e);
  });
}

// Proof to the backend that this request came from the app and not from a page
// the user happens to have open. A cross-site fetch cannot set a custom header
// without a preflight, and the preflight is refused for any origin but ours.
// Sent on every request; the backend only insists on it for the ones that write.
const APP_HEADER = { 'X-Vessel-App': '1' };

async function json(method, path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json', ...APP_HEADER } : APP_HEADER,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || data.detail || `HTTP ${res.status}`);
  return data;
}

const tauriApi = {
  health: () => invoke('health'),

  listCharacters: () => invoke('list_characters').then((d) => d.characters),
  createCharacter: (c) => invoke('create_character', { patch: c }),
  updateCharacter: (id, c) => invoke('update_character', { id, patch: c }),
  deleteCharacter: (id) => invoke('delete_character', { id }),

  getSettings: () => invoke('get_settings'),
  saveSettings: (patch) =>
    invoke('save_settings', { tursoUrl: patch.tursoUrl, tursoToken: patch.tursoToken }),

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

const httpApi = {
  health: () => json('GET', '/health'),

  listCharacters: () => json('GET', '/characters').then((d) => d.characters),
  createCharacter: (c) => json('POST', '/characters', c),
  updateCharacter: (id, c) => json('PUT', `/characters/${id}`, c),
  deleteCharacter: (id) => json('DELETE', `/characters/${id}`),

  getSettings: () => json('GET', '/settings'),
  saveSettings: (patch) => json('PUT', '/settings', patch),

  listConversations: (characterId) =>
    json('GET', `/conversations${characterId ? `?characterId=${encodeURIComponent(characterId)}` : ''}`)
      .then((d) => d.conversations),
  getConversation: (id) => json('GET', `/conversations/${id}`),
  deleteConversation: (id) => json('DELETE', `/conversations/${id}`),
  setActiveVariant: (id, turnId, variantId) =>
    json('PUT', `/conversations/${id}/active-variant`, { turnId, variantId }),

  relaunch:
    typeof window !== 'undefined' && window.scenario && window.scenario.relaunch
      ? () => window.scenario.relaunch()
      : null,
};

export const api = TAURI ? tauriApi : httpApi;

/**
 * Stream a chat reply.
 *
 * @param {object} payload { characterId?, conversationId?, messages, regenerate?, director? }
 * @param {object} handlers { onMeta(meta), onToken(text, full), onDone(fullText), onError(msg) }
 * @returns {function} abort() -- stops the in-flight generation.
 */
export function streamChat(payload, handlers = {}) {
  return TAURI ? streamViaChannel(payload, handlers) : streamViaSse(payload, handlers);
}

function streamViaChannel(payload, { onMeta, onToken, onDone, onError } = {}) {
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

// EventSource cannot POST, so the SSE stream is parsed manually from fetch().
function streamViaSse(payload, { onMeta, onToken, onDone, onError } = {}) {
  const controller = new AbortController();
  let full = '';

  (async () => {
    let res;
    try {
      res = await fetch(`${BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...APP_HEADER },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
    } catch (err) {
      if (err.name !== 'AbortError') onError && onError(err.message);
      return;
    }

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      onError && onError(data.error || data.detail || `HTTP ${res.status}`);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let event = 'message';

    const handleEvent = (evt, dataStr) => {
      if (evt === 'meta') {
        try { onMeta && onMeta(JSON.parse(dataStr)); } catch { /* ignore */ }
        return;
      }
      if (evt === 'error') {
        try { onError && onError(JSON.parse(dataStr).error); } catch { onError && onError(dataStr); }
        return;
      }
      // default: raw Ollama chunk
      try {
        const obj = JSON.parse(dataStr);
        if (obj.message && typeof obj.message.content === 'string') {
          full += obj.message.content;
          onToken && onToken(obj.message.content, full);
        }
      } catch { /* non-JSON line */ }
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buffer.indexOf('\n\n')) !== -1) {
          const raw = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          event = 'message';
          let dataStr = '';
          for (const line of raw.split('\n')) {
            if (line.startsWith('event:')) event = line.slice(6).trim();
            else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
          }
          if (dataStr) handleEvent(event, dataStr);
        }
      }
      onDone && onDone(full);
    } catch (err) {
      if (err.name !== 'AbortError') onError && onError(err.message);
    }
  })();

  return () => controller.abort();
}
