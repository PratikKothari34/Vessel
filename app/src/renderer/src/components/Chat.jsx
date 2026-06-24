import React, { useState, useEffect, useRef, useCallback } from 'react';
import { api, streamChat } from '../lib/api';
import Avatar from './Avatar.jsx';
import MessageText from './MessageText.jsx';
import MemoryInspector from './MemoryInspector.jsx';
import { relTime } from '../lib/util';
import './chat.css';

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

  const abortRef = useRef(null);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  const refreshConversations = useCallback(async () => {
    try { setConversations(await api.listConversations(character.id)); } catch { /* ignore */ }
  }, [character.id]);

  // Load conversation list + the active conversation's turns.
  useEffect(() => { refreshConversations(); }, [refreshConversations]);

  useEffect(() => {
    setConvId(conversationId);
    if (conversationId) {
      api.getConversation(conversationId).then((c) => {
        const turns = [...c.archive, ...c.verbatim].map((t) => ({ role: t.role, content: t.content }));
        setMessages(turns);
      }).catch(() => setMessages([]));
    } else {
      // Fresh conversation: seed with the character's greeting.
      setMessages(character.greeting ? [{ role: 'assistant', content: character.greeting }] : []);
    }
  }, [conversationId, character]);

  // Autoscroll on new content.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streamText]);

  const send = (regenerate = false) => {
    if (streaming) return;
    const text = input.trim();
    if (!regenerate && !text) return;

    setError(null);
    let next = messages;
    if (!regenerate) {
      next = [...messages, { role: 'user', content: text }];
      setMessages(next);
      setInput('');
    } else {
      // drop last assistant locally for the re-roll
      next = [...messages];
      if (next.length && next[next.length - 1].role === 'assistant') next = next.slice(0, -1);
      setMessages(next);
    }

    setStreaming(true);
    setStreamText('');

    const payload = {
      characterId: character.id,
      conversationId: convId || undefined,
      regenerate,
      messages: regenerate ? [] : [{ role: 'user', content: text }],
    };

    abortRef.current = streamChat(payload, {
      onMeta: (meta) => {
        if (meta.conversationId && meta.conversationId !== convId) {
          setConvId(meta.conversationId);
          onConversation && onConversation(meta.conversationId);
        }
        setRecalled(meta.recalled || []);
      },
      onToken: (_t, full) => setStreamText(full),
      onDone: (full) => {
        setStreaming(false);
        setStreamText('');
        if (full.trim()) setMessages((m) => [...m, { role: 'assistant', content: full.trim() }]);
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

  const newConversation = () => {
    if (streaming) stop();
    setConvId(null);
    onConversation && onConversation(null);
    setMessages(character.greeting ? [{ role: 'assistant', content: character.greeting }] : []);
    setRecalled([]);
    inputRef.current?.focus();
  };

  const openConversation = (id) => {
    if (streaming) stop();
    onConversation && onConversation(id);
  };

  const deleteConv = async (e, id) => {
    e.stopPropagation();
    await api.deleteConversation(id);
    if (id === convId) newConversation();
    refreshConversations();
  };

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div className="chat-layout">
      {/* ── History sidebar ── */}
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
            <div
              key={c.id}
              className={`rail-item ${c.id === convId ? 'active' : ''}`}
              onClick={() => openConversation(c.id)}
            >
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

      {/* ── Conversation column ── */}
      <section className="chat-main">
        <header className="chat-head">
          <div className="chat-head-title">
            <Avatar character={character} size={32} />
            <span>{character.name}</span>
          </div>
          <button className="btn btn-ghost" onClick={() => setShowMemory(true)} disabled={!convId}>
            ◆ Memory
          </button>
        </header>

        <div className="chat-stream" ref={scrollRef}>
          <div className="stream-inner">
            {messages.map((m, i) => (
              <div key={i} className={`msg-row ${m.role}`} style={{ animationDelay: `${Math.min(i, 6) * 30}ms` }}>
                {m.role === 'assistant' && <Avatar character={character} size={36} />}
                <div className={`bubble ${m.role}`}>
                  <MessageText text={m.content} />
                </div>
              </div>
            ))}

            {streaming && (
              <div className="msg-row assistant">
                <Avatar character={character} size={36} />
                <div className="bubble assistant">
                  {streamText ? <MessageText text={streamText} /> : (
                    <div className="typing"><span /><span /><span /></div>
                  )}
                </div>
              </div>
            )}

            {error && <div className="chat-error">⚠ {error}</div>}
          </div>
        </div>

        <footer className="chat-input-bar">
          {!streaming && messages.length > 0 && messages[messages.length - 1].role === 'assistant' && (
            <button className="btn btn-ghost regen" onClick={() => send(true)} title="Regenerate last reply">
              ↻ Regenerate
            </button>
          )}
          <div className="input-wrap">
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder={`Write the next moment with ${character.name}…`}
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
