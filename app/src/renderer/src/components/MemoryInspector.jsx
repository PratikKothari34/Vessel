import React, { useEffect, useState } from 'react';
import { api } from '../lib/api';
import './memory.css';

// Read-only window into the conversation's long-term memory: rolling summary,
// verbatim recent turns, archived turns, and what was recalled this turn.
export default function MemoryInspector({ conversationId, recalled, onClose }) {
  const [conv, setConv] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if (!conversationId) return;
    api.getConversation(conversationId).then(setConv).catch((e) => setErr(e.message));
  }, [conversationId]);

  return (
    <div className="overlay-backdrop" onClick={onClose}>
      <div className="overlay-panel memory-panel" onClick={(e) => e.stopPropagation()}>
        <div className="overlay-head">
          <div>
            <p className="kicker">The Vault</p>
            <h2 className="overlay-title">Memory</h2>
          </div>
          <button className="overlay-close" onClick={onClose}>✕</button>
        </div>

        <div className="overlay-body">
          {err && <p className="mem-err">{err}</p>}
          {!conv && !err && <p className="mem-dim">No memory recorded yet.</p>}

          {conv && (
            <>
              <section className="mem-section">
                <h4 className="mem-h">Recalled this turn</h4>
                {recalled && recalled.length ? (
                  recalled.map((r, i) => (
                    <div key={i} className="mem-snippet recalled">
                      <span className="mem-score">{(r.score * 100).toFixed(0)}%</span>
                      <span className="mem-snippet-text">{r.content.slice(0, 220)}</span>
                    </div>
                  ))
                ) : (
                  <p className="mem-dim">Nothing retrieved — recent turns covered it.</p>
                )}
              </section>

              <section className="mem-section">
                <h4 className="mem-h">Rolling summary</h4>
                {conv.summary ? (
                  <div className="mem-summary">{conv.summary}</div>
                ) : (
                  <p className="mem-dim">Not enough history to summarize yet.</p>
                )}
              </section>

              <section className="mem-section">
                <h4 className="mem-h">
                  Verbatim window <span className="mem-count">{conv.verbatim.length}</span>
                </h4>
                <div className="mem-turns">
                  {conv.verbatim.map((t, i) => (
                    <div key={i} className={`mem-turn ${t.role}`}>
                      <span className="mem-role">{t.role === 'user' ? 'You' : 'Char'}</span>
                      <span>{t.content.slice(0, 160)}{t.content.length > 160 ? '…' : ''}</span>
                    </div>
                  ))}
                </div>
              </section>

              <section className="mem-section">
                <h4 className="mem-h">
                  Archived <span className="mem-count">{conv.archive.length}</span>
                </h4>
                {conv.archive.length ? (
                  <div className="mem-turns">
                    {conv.archive.map((t, i) => (
                      <div key={i} className={`mem-turn ${t.role} archived`}>
                        <span className="mem-role">{t.role === 'user' ? 'You' : 'Char'}</span>
                        <span>{t.content.slice(0, 140)}{t.content.length > 140 ? '…' : ''}</span>
                        {t.hasEmbedding && <span className="mem-vec" title="indexed for retrieval">◆</span>}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="mem-dim">Nothing archived — story still fits the live window.</p>
                )}
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
