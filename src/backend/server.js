'use strict';

require('dotenv').config();

const express = require('express');
const cors = require('cors');
const db = require('./db');
const memory = require('./memory');
const characters = require('./characters');

const app = express();

// ---- Config --------------------------------------------------------------
const PORT = process.env.PORT || 3001;
const OLLAMA_HOST = (process.env.OLLAMA_HOST || 'http://localhost:11434').replace(/\/+$/, '');
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'scenario-chat';
const OLLAMA_CHAT_URL = `${OLLAMA_HOST}/api/chat`;

// Response-style rules (narration vs dialogue control). Addresses the common
// failure where the model only narrates the scene instead of speaking as the
// character (seen in the character.ai reference).
const STYLE_RULES = {
  balanced: '',
  dialogue:
    'Response style: ALWAYS give the character spoken dialogue when addressed. Lead with what the character SAYS (in quotes). Keep scene narration minimal — at most one short line of action. Never reply with narration only.',
  'narration-light':
    'Response style: keep narration brief and focused. Prioritize the character speaking and reacting over describing the scene.',
};

// Build the character persona system message. The base model carries global
// roleplay behavior; this injects the specific character per conversation.
function buildPersonaMessage(character) {
  if (!character) return null;
  const parts = [];
  parts.push(`You are roleplaying as the character "${character.name}". Stay fully in character as ${character.name}.`);
  if (character.persona && character.persona.trim()) {
    parts.push(`Character details:\n${character.persona.trim()}`);
  }
  const styleRule = STYLE_RULES[character.responseStyle] || '';
  if (styleRule) parts.push(styleRule);
  return { role: 'system', content: parts.join('\n\n') };
}

// Director / OOC note: a meta-instruction that steers the model WITHOUT becoming
// part of the story. Injected as a high-priority system message, never recorded.
function buildDirectorMessage(director) {
  if (typeof director !== 'string' || !director.trim()) return null;
  return {
    role: 'system',
    content: `[Director note — out of character. Follow this instruction for how you write from now on, but do NOT mention it in the story or break character to acknowledge it:]\n${director.trim()}`,
  };
}

// ---- Middleware ----------------------------------------------------------
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || 'http://localhost:5173')
  .split(',').map((o) => o.trim()).filter(Boolean);
app.use(cors({
  origin(origin, cb) {
    if (!origin || ALLOWED_ORIGINS.includes(origin)) return cb(null, true);
    return cb(null, false);
  },
}));
app.use(express.json({ limit: '10mb' }));

app.use((err, _req, res, next) => {
  if (err) {
    const tooBig = err.type === 'entity.too.large';
    return res.status(tooBig ? 413 : 400).json({
      error: tooBig ? 'Request body too large.' : 'Malformed JSON body.',
      detail: err.message,
    });
  }
  next();
});

// ---- Health --------------------------------------------------------------
app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    model: OLLAMA_MODEL,
    ollama: OLLAMA_HOST,
    sync: { enabled: db.isSyncEnabled(), interval: db._config.SYNC_INTERVAL },
    encryptedAtRest: db.isEncryptedAtRest(),
    memory: {
      summarizer: memory._config.SUMMARIZER_MODEL,
      embedder: memory._config.EMBED_MODEL,
      verbatimTurns: memory._config.VERBATIM_TURNS,
      summarizeThreshold: memory._config.SUMMARIZE_THRESHOLD,
    },
  });
});

// ---- Characters ----------------------------------------------------------
app.get('/characters', async (_req, res) => {
  try { res.json({ characters: await characters.listCharacters() }); }
  catch (err) { res.status(500).json({ error: 'Failed to list characters.', detail: err.message }); }
});

app.post('/characters', async (req, res) => {
  try { res.status(201).json(await characters.createCharacter(req.body || {})); }
  catch (err) { res.status(400).json({ error: 'Failed to create character.', detail: err.message }); }
});

app.get('/characters/:id', async (req, res) => {
  try {
    const c = await characters.getCharacter(req.params.id);
    if (!c) return res.status(404).json({ error: 'Character not found.' });
    res.json(c);
  } catch (err) { res.status(500).json({ error: 'Failed to read character.', detail: err.message }); }
});

app.put('/characters/:id', async (req, res) => {
  try {
    const c = await characters.updateCharacter(req.params.id, req.body || {});
    if (!c) return res.status(404).json({ error: 'Character not found.' });
    res.json(c);
  } catch (err) { res.status(400).json({ error: 'Failed to update character.', detail: err.message }); }
});

app.delete('/characters/:id', async (req, res) => {
  try {
    const removed = await characters.deleteCharacter(req.params.id);
    if (!removed) return res.status(404).json({ error: 'Character not found.' });
    res.json({ deleted: true });
  } catch (err) { res.status(400).json({ error: 'Failed to delete character.', detail: err.message }); }
});

// ---- Conversations -------------------------------------------------------
app.get('/conversations', async (req, res) => {
  try {
    const characterId = req.query.characterId || null;
    res.json({ conversations: await memory.listConversations(characterId) });
  } catch (err) { res.status(500).json({ error: 'Failed to list conversations.', detail: err.message }); }
});

app.get('/conversations/:id', async (req, res) => {
  if (!memory.isValidId(req.params.id)) return res.status(400).json({ error: 'Invalid conversation id.' });
  try {
    const conv = await memory.getConversation(req.params.id);
    if (!conv) return res.status(404).json({ error: 'Conversation not found.' });
    res.json(conv);
  } catch (err) { res.status(500).json({ error: 'Failed to read conversation.', detail: err.message }); }
});

app.patch('/conversations/:id', async (req, res) => {
  if (!memory.isValidId(req.params.id)) return res.status(400).json({ error: 'Invalid conversation id.' });
  try { res.json(await memory.setTitle(req.params.id, (req.body || {}).title)); }
  catch (err) { res.status(500).json({ error: 'Failed to rename conversation.', detail: err.message }); }
});

app.delete('/conversations/:id', async (req, res) => {
  if (!memory.isValidId(req.params.id)) return res.status(400).json({ error: 'Invalid conversation id.' });
  try {
    const removed = await memory.deleteConversation(req.params.id);
    if (!removed) return res.status(404).json({ error: 'Conversation not found.' });
    res.json({ deleted: true });
  } catch (err) { res.status(500).json({ error: 'Failed to delete conversation.', detail: err.message }); }
});

/**
 * POST /chat — SSE stream.
 * Body: { characterId?, conversationId?, messages: [{role,content}], regenerate? }
 *   - Assembles the live window from memory (persona + summary + retrieval + verbatim).
 *   - Emits `event: meta` (conversationId + recalled snippets), then Ollama chunks.
 *   - Post-stream: records the turn off the response path (summarize/embed).
 */
app.post('/chat', async (req, res) => {
  const { messages, conversationId, characterId, regenerate, director } = req.body || {};
  const isRegenerate = Boolean(regenerate);
  const hasDirector = typeof director === 'string' && director.trim().length > 0;

  // messages may be empty for a regenerate (re-roll) or a director-only note
  // (steer + continue without a new user turn). Otherwise require a real message.
  if (!Array.isArray(messages) || (messages.length === 0 && !isRegenerate && !hasDirector)) {
    return res.status(400).json({ error: 'Invalid input: "messages" must be a non-empty array.' });
  }
  // A director-only request has no story user turn to record.
  const directorOnly = hasDirector && !isRegenerate &&
    !messages.some((m) => m && m.role === 'user' && typeof m.content === 'string' && m.content.trim());

  let releaseLock = null;
  let convId = null;
  let character = null;
  let latestUser = null;
  let retrieved = [];
  let outboundMessages = messages;

  try {
    convId = memory.isValidId(conversationId) ? conversationId : memory.newId();
    releaseLock = await memory.acquireLock(convId);

    // Resolve character: explicit characterId, else the conversation's bound one.
    let charId = characters.isValidId(characterId) ? characterId : null;
    const convRow = await memory.ensureConversation(convId, charId);
    if (!charId && convRow.character_id) charId = convRow.character_id;
    if (charId) character = await characters.getCharacter(charId);

    const personaMsg = buildPersonaMessage(character);
    const directorMsg = buildDirectorMessage(director);
    const leading = [personaMsg, directorMsg].filter(Boolean);

    // Regenerate: keep the assistant turn (we append a variant to it). Build
    // from the persisted verbatim window, then drop the trailing assistant so the
    // model re-rolls from the prompting user message. (No incoming turn — the
    // user message is already in verbatim; passing it again would duplicate it.)
    let contextInput = messages;
    if (isRegenerate) {
      const lastAssistant = await memory.getLastAssistantTurn(convId);
      if (!lastAssistant) {
        if (releaseLock) releaseLock();
        return res.status(400).json({ error: 'Nothing to regenerate: this conversation has no previous reply.' });
      }
      contextInput = [];
    }

    // Director-only: no new user turn — just steer + continue from current state.
    if (directorOnly) contextInput = [];

    const built = await memory.buildContext(convId, contextInput, leading);
    // On regenerate, drop the last assistant turn from the outbound window so the
    // model doesn't see its own previous reply when re-rolling.
    if (isRegenerate) {
      const lastIdx = [...built.messages].reverse().findIndex((m) => m.role === 'assistant');
      if (lastIdx !== -1) built.messages.splice(built.messages.length - 1 - lastIdx, 1);
    }
    outboundMessages = built.messages;
    // directorOnly has no story user turn to record afterwards.
    latestUser = directorOnly ? null : built.latestUser;
    retrieved = built.retrieved || [];

    if (!outboundMessages.some((m) => m.role !== 'system')) {
      if (releaseLock) releaseLock();
      return res.status(400).json({
        error: isRegenerate
          ? 'Nothing to regenerate: this conversation has no previous turn.'
          : 'No valid messages: each must have a known role and string content.',
      });
    }
  } catch (err) {
    if (releaseLock) releaseLock();
    return res.status(500).json({ error: 'Memory subsystem failed to assemble context.', detail: err.message });
  }

  // Abort upstream if the client disconnects. Listen on res (real disconnect),
  // not req (fires immediately after body parse).
  const abortController = new AbortController();
  let clientAborted = false;
  res.on('close', () => {
    if (!res.writableEnded) { clientAborted = true; abortController.abort(); }
  });

  // Per-character sampling overrides (cleaned in characters.js).
  const options = character && character.sampling ? { ...character.sampling } : undefined;

  let ollamaRes;
  try {
    ollamaRes = await fetch(OLLAMA_CHAT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: OLLAMA_MODEL, messages: outboundMessages, stream: true, ...(options ? { options } : {}) }),
      signal: abortController.signal,
    });
  } catch (err) {
    if (releaseLock) releaseLock();
    if (err.name === 'AbortError') return;
    return res.status(503).json({ error: 'Cannot reach Ollama. Is it running?', detail: err.message, ollama: OLLAMA_HOST });
  }

  if (!ollamaRes.ok) {
    let detail;
    try { detail = await ollamaRes.json(); } catch { detail = await ollamaRes.text().catch(() => ''); }
    if (releaseLock) releaseLock();
    return res.status(ollamaRes.status === 404 ? 404 : 502).json({ error: `Ollama returned ${ollamaRes.status}.`, detail });
  }

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  if (typeof res.flushHeaders === 'function') res.flushHeaders();

  const recalled = retrieved.map((t) => ({
    role: t.role, content: t.content, score: typeof t.score === 'number' ? t.score : null,
  }));
  res.write(`event: meta\ndata: ${JSON.stringify({ conversationId: convId, characterId: character ? character.id : null, recalled })}\n\n`);

  const sendError = (message) => {
    try { res.write(`event: error\ndata: ${JSON.stringify({ error: message })}\n\n`); } catch { /* gone */ }
  };

  let replyText = '';
  let ollamaError = null;

  try {
    const reader = ollamaRes.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const handleLine = (line) => {
      if (!line) return;
      res.write(`data: ${line}\n\n`);
      try {
        const obj = JSON.parse(line);
        if (obj.error) { ollamaError = String(obj.error); sendError(`Ollama error: ${ollamaError}`); return; }
        if (obj.message && typeof obj.message.content === 'string') replyText += obj.message.content;
      } catch { /* non-JSON line */ }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        handleLine(line);
      }
    }
    handleLine(buffer.trim());
    res.end();
  } catch (err) {
    if (err.name !== 'AbortError') sendError(`Stream interrupted: ${err.message}`);
    try { res.end(); } catch { /* closed */ }
  }

  // Post-stream bookkeeping — off the response path. Skip on error/disconnect.
  // A director-only message (no user turn, no regenerate) steers behavior but is
  // never recorded; it still produces a reply we DO record as a normal turn.
  try {
    if (!ollamaError && !clientAborted && replyText.trim()) {
      const assistantName = character ? character.name : 'Character';
      if (isRegenerate) {
        await memory.recordRegeneration(convId, replyText.trim());
      } else {
        await memory.recordTurn(convId, latestUser, replyText.trim(), assistantName);
      }
    }
  } catch (err) {
    console.error(`[memory] record failed for ${convId}:`, err.message);
  } finally {
    if (releaseLock) releaseLock();
  }
});

// Set which variant of an assistant turn is active (swipe selection). Updates
// the canonical turn text used for memory/summary.
app.put('/conversations/:id/active-variant', async (req, res) => {
  if (!memory.isValidId(req.params.id)) return res.status(400).json({ error: 'Invalid conversation id.' });
  const { turnId, variantId } = req.body || {};
  if (!Number.isInteger(turnId) || !Number.isInteger(variantId)) {
    return res.status(400).json({ error: 'turnId and variantId (integers) are required.' });
  }
  try {
    const result = await memory.setActiveVariant(req.params.id, turnId, variantId);
    res.json(result);
  } catch (err) {
    res.status(400).json({ error: 'Failed to set active variant.', detail: err.message });
  }
});

// ---- Startup -------------------------------------------------------------
async function start() {
  await db.getDb(); // init schema + (optional) initial sync before serving
  app.listen(PORT, '127.0.0.1', () => {
    console.log(`Scenario_Chat backend on http://localhost:${PORT}`);
    console.log(`  -> ollama: ${OLLAMA_CHAT_URL} | model: ${OLLAMA_MODEL}`);
    console.log(`  -> sync: ${db.isSyncEnabled() ? 'enabled' : 'local-only'}`);
    console.log(`  -> memory: summarizer=${memory._config.SUMMARIZER_MODEL}, embedder=${memory._config.EMBED_MODEL}`);
  });
}

// Flush to cloud on shutdown (no-op when local-only).
async function shutdown() {
  try { await db.syncNow(); } catch { /* best effort */ }
  process.exit(0);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

start().catch((err) => {
  console.error('Failed to start backend:', err.message);
  process.exit(1);
});
