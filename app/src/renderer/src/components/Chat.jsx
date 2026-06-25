import React, { useState, useEffect, useRef, useCallback } from 'react';
import { api, streamChat } from '../lib/api';
import Avatar from './Avatar.jsx';
import MessageText from './MessageText.jsx';
import MemoryInspector from './MemoryInspector.jsx';
import { relTime } from '../lib/util';
import './chat.css';

// Recognize an inline OOC/director command: leading "//" or "/ooc ".
function parseDirector(text) {
  const t = text.trim();
  if (t.startsWith('//')) return t.slice(2).trim();
  if (/^\/ooc\b/i.test(t)) return t.replace(/^\/ooc\b/i, '').trim();
  return null;
}

export default function Chat({ character, conversationId, onBack, onConversation, onEditCharacter }) {
  const [convId, setConvId] = useState(conversationId);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState('');
  const [conversations, setConversations] = useState([]);
  const [recalled, setRecalled] = useState([]);
  const [showMemory, setShowMemory] = useState(false);
  const [error, setError] = useState(null);
  const [directorMode, setDirectorMode] = useState(false);

  const abortRef = useRef(null);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  const refreshConversations = useCallback(async () => {
    try { setConversations(await api.listConversations(character.id)); } catch { /* ignore */ }
  }, [character.id]);

  useEffect(() => { refreshConversations(); }, [refreshConversations]);

  // Load the active conversation's turns (with variant metadata on the last one).
  // Greeting shown as the character's opening line. It's display-only (never a
  // stored turn), so we prepend it to every view of this character's chats.
  const greetingMsg = character.greeting
    ? [{ role: 'assistant', content: character.greeting, greeting: true }]
    : [];

  const loadConversation = useCallback(async (id) => {
    if (!id) {
      setMessages([...greetingMsg]);
      return;
    }
    try {
      const c = await api.getConversation(id);
      const archived = c.archive.map((t) => ({ role: t.role, content: t.content }));
      const verbatim = c.verbatim.map((t) => ({
        role: t.role,
        content: t.content,
        turnId: t.turnId,
        variants: t.variants || null,
        activeIndex: t.activeIndex ?? 0,
      }));
      setMessages([...greetingMsg, ...archived, ...verbatim]);
    } catch {
      // A transient load failure (backend blip) must not wipe the visible chat.
      // Keep whatever is on screen; the next successful load reconciles it. Only
      // fall back to the greeting when there's nothing shown yet.
      setMessages((m) => (m.length ? m : [...greetingMsg]));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [character]);

  useEffect(() => {
    setConvId(conversationId);
    loadConversation(conversationId);
  }, [conversationId, loadConversation]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streamText]);

  const send = (regenerate = false) => {
    if (streaming) return;
    const raw = input.trim();

    // Director / OOC: either toggle is on, or the message uses a // or /ooc prefix.
    let director = null;
    let storyText = raw;
    if (!regenerate) {
      const prefixCmd = parseDirector(raw);
      if (prefixCmd !== null) { director = prefixCmd; storyText = ''; }
      else if (directorMode) { director = raw; storyText = ''; }
    }

    if (!regenerate && !director && !storyText) return;

    setError(null);

    let next = messages;
    if (!regenerate) {
      if (director) {
        // Director note: show a subtle OOC chip, don't add a story user bubble.
        next = [...messages, { role: 'director', content: director }];
      } else {
        next = [...messages, { role: 'user', content: storyText }];
      }
      setMessages(next);
      setInput('');
      if (directorMode) setDirectorMode(false);
    } else {
      // Regenerate: keep the last assistant bubble; we'll attach a new variant.
    }

    setStreaming(true);
    setStreamText('');

    const payload = {
      characterId: character.id,
      conversationId: convId || undefined,
      regenerate,
      director: director || undefined,
      // Story turn only when there's real text; director-only & regenerate send none.
      messages: !regenerate && storyText ? [{ role: 'user', content: storyText }] : [],
    };

    // The conversation id this stream belongs to (may be minted server-side and
    // arrive via onMeta). Captured here so onDone reloads the RIGHT conversation,
    // not the stale closure value.
    let streamConvId = convId;

    abortRef.current = streamChat(payload, {
      onMeta: (meta) => {
        if (meta.conversationId) {
          streamConvId = meta.conversationId;
          if (meta.conversationId !== convId) {
            setConvId(meta.conversationId);
            onConversation && onConversation(meta.conversationId);
          }
        }
        setRecalled(meta.recalled || []);
      },
      onToken: (_t, full) => setStreamText(full),
      onDone: async (full) => {
        setStreaming(false);
        setStreamText('');
        // Optimistic append for a new reply (not regenerate, which updates the
        // existing bubble in place). Immediately reconciled by the reload below,
        // which carries authoritative variant ids/counts. The append is the
        // fallback shown if the reload fails.
        if (!regenerate && full && full.trim()) {
          setMessages((m) => [...m, { role: 'assistant', content: full.trim() }]);
        }
        if (streamConvId) await loadConversation(streamConvId);
        refreshConversations();
      },
      onError: (msg) => {
        setStreaming(false);
        setStreamText('');
        setError(msg);
      },
    });
  };

  const stop = () => { if (abortRef.current) abortRef.current(); setStreaming(false); };

  const swipeVariant = async (msgIndex, dir) => {
    const msg = messages[msgIndex];
    if (!msg.variants || !msg.turnId) return;
    const newIdx = msg.activeIndex + dir;
    if (newIdx < 0 || newIdx >= msg.variants.length) return;
    const variant = msg.variants[newIdx];
    try {
      await api.setActiveVariant(convId, msg.turnId, variant.id);
      setMessages((ms) => ms.map((m, i) =>
        i === msgIndex ? { ...m, content: variant.content, activeIndex: newIdx } : m
      ));
    } catch (e) { setError(e.message); }
  };

  const newConversation = () => {
    if (streaming) stop();
    setConvId(null);
    onConversation && onConversation(null);
    setMessages([...greetingMsg]);
    setRecalled([]);
    inputRef.current?.focus();
  };

  const openConversation = (id) => { if (streaming) stop(); onConversation && onConversation(id); };

  const deleteConv = async (e, id) => {
    e.stopPropagation();
    await api.deleteConversation(id);
    if (id === convId) newConversation();
    refreshConversations();
  };

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const lastAssistantIndex = (() => {
    for (let i = messages.length - 1; i >= 0; i--) if (messages[i].role === 'assistant') return i;
    return -1;
  })();

  const directorActive = directorMode || parseDirector(input) !== null;

  return (
    <div className="chat-layout">
      <aside className="chat-rail">
        <button className="rail-back btn btn-ghost" onClick={onBack}>← All characters</button>

        <div className="rail-char" onClick={onEditCharacter} title="Edit character">
          <Avatar character={character} size={44} ring />
          <div className="rail-char-meta">
            <span className="rail-char-name">{character.name}</span>
            <span className="rail-char-edit">edit persona</span>
          </div>
        </div>

        <button className="btn btn-primary rail-new" onClick={newConversation}>+ New scenario</button>

        <div className="rail-list">
          <p className="kicker rail-list-head">Scenarios</p>
          {conversations.length === 0 && <p className="rail-empty">No saved scenarios.</p>}
          {conversations.map((c) => (
            <div key={c.id} className={`rail-item ${c.id === convId ? 'active' : ''}`} onClick={() => openConversation(c.id)}>
              <div className="rail-item-main">
                <span className="rail-item-title">{c.title || 'Untitled scenario'}</span>
                <span className="rail-item-preview">{c.preview || '…'}</span>
              </div>
              <div className="rail-item-side">
                <span className="rail-item-time">{relTime(c.updatedAt)}</span>
                <button className="rail-del" onClick={(e) => deleteConv(e, c.id)} title="Delete">✕</button>
              </div>
            </div>
          ))}
        </div>
      </aside>

      <section className="chat-main">
        <header className="chat-head">
          <div className="chat-head-title">
            <Avatar character={character} size={32} />
            <span>{character.name}</span>
          </div>
          <button className="btn btn-ghost" onClick={() => setShowMemory(true)} disabled={!convId}>◆ Memory</button>
        </header>

        <div className="chat-stream" ref={scrollRef}>
          <div className="stream-inner">
            {messages.map((m, i) => {
              if (m.role === 'director') {
                return (
                  <div key={i} className="director-chip">
                    <span className="director-tag">Director</span>
                    {m.content}
                  </div>
                );
              }
              const isLastAssistant = i === lastAssistantIndex;
              const hasVariants = isLastAssistant && m.variants && m.variants.length > 1;
              return (
                <div key={i} className={`msg-row ${m.role}`} style={{ animationDelay: `${Math.min(i, 6) * 30}ms` }}>
                  {m.role === 'assistant' && <Avatar character={character} size={36} />}
                  <div className={`bubble ${m.role}`}>
                    <MessageText text={m.content} />
                    {hasVariants && (
                      <div className="variant-bar">
                        <button className="variant-arrow" disabled={m.activeIndex === 0 || streaming} onClick={() => swipeVariant(i, -1)}>◀</button>
                        <span className="variant-count">{m.activeIndex + 1} / {m.variants.length}</span>
                        <button className="variant-arrow" disabled={m.activeIndex === m.variants.length - 1 || streaming} onClick={() => swipeVariant(i, 1)}>▶</button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {streaming && (
              <div className="msg-row assistant">
                <Avatar character={character} size={36} />
                <div className="bubble assistant">
                  {streamText ? <MessageText text={streamText} /> : <div className="typing"><span /><span /><span /></div>}
                </div>
              </div>
            )}

            {error && <div className="chat-error">⚠ {error}</div>}
          </div>
        </div>

        <footer className="chat-input-bar">
          {/* Regenerate only when the last message is a REAL persisted assistant
              turn (not the display-only greeting). */}
          {!streaming && lastAssistantIndex === messages.length - 1 &&
            messages.length > 0 && !messages[lastAssistantIndex].greeting && (
            <button className="btn btn-ghost regen" onClick={() => send(true)} title="Generate another variant">
              ↻ Regenerate
            </button>
          )}
          <div className={`input-wrap ${directorActive ? 'director' : ''}`}>
            <button
              className={`director-toggle ${directorActive ? 'on' : ''}`}
              onClick={() => setDirectorMode((v) => !v)}
              title="Director mode — give the AI an out-of-character instruction (or type // before your message)"
            >
              ◈
            </button>
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder={directorActive
                ? 'Director note to the AI (out of character)…'
                : `Write the next moment with ${character.name}…  (// for director)`}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKey}
              rows={1}
            />
            {streaming ? (
              <button className="send-btn stop" onClick={stop}>■</button>
            ) : (
              <button className="send-btn" onClick={() => send()} disabled={!input.trim()}>↑</button>
            )}
          </div>
        </footer>
      </section>

      {showMemory && (
        <MemoryInspector conversationId={convId} recalled={recalled} onClose={() => setShowMemory(false)} />
      )}
    </div>
  );
}
