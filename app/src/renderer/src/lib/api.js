// API client for the local Scenario_Chat backend.
// EventSource can't POST, so /chat streaming is parsed manually from fetch().

const BASE =
  (typeof window !== 'undefined' && window.scenario && window.scenario.backendUrl) ||
  'http://localhost:3001';

async function json(method, path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || data.detail || `HTTP ${res.status}`);
  return data;
}

export const api = {
  health: () => json('GET', '/health'),

  // Characters
  listCharacters: () => json('GET', '/characters').then((d) => d.characters),
  getCharacter: (id) => json('GET', `/characters/${id}`),
  createCharacter: (c) => json('POST', '/characters', c),
  updateCharacter: (id, c) => json('PUT', `/characters/${id}`, c),
  deleteCharacter: (id) => json('DELETE', `/characters/${id}`),

  // Conversations
  listConversations: (characterId) =>
    json('GET', `/conversations${characterId ? `?characterId=${characterId}` : ''}`).then(
      (d) => d.conversations
    ),
  getConversation: (id) => json('GET', `/conversations/${id}`),
  renameConversation: (id, title) => json('PATCH', `/conversations/${id}`, { title }),
  deleteConversation: (id) => json('DELETE', `/conversations/${id}`),
};

/**
 * Stream a chat reply via SSE.
 *
 * @param {object} payload { characterId?, conversationId?, messages, regenerate? }
 * @param {object} handlers { onMeta(meta), onToken(text), onDone(fullText), onError(msg) }
 * @returns {function} abort() — cancels the in-flight stream.
 */
export function streamChat(payload, handlers = {}) {
  const { onMeta, onToken, onDone, onError } = handlers;
  const controller = new AbortController();
  let full = '';

  (async () => {
    let res;
    try {
      res = await fetch(`${BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
