import React from 'react';
import Avatar from './Avatar.jsx';
import './gallery.css';

export default function Gallery({ characters, loading, onOpen, onNew, onEdit }) {
  return (
    <div className="gallery-scroll">
      <div className="gallery-inner">
        <div className="gallery-head">
          <div>
            <p className="kicker">Local · Private · Unfiltered</p>
            <h1 className="gallery-title">Choose a <em>persona</em></h1>
            <p className="gallery-sub">
              Every conversation lives on this machine. Pick a character to begin a scenario,
              or forge a new one.
            </p>
          </div>
          <button className="btn btn-primary" onClick={onNew}>+ New Character</button>
        </div>

        {loading ? (
          <div className="gallery-empty"><span className="loading-pulse" /> Loading cast…</div>
        ) : characters.length === 0 ? (
          <div className="gallery-empty empty-state">
            <div className="empty-glyph" />
            <h3>No characters yet</h3>
            <p>Create your first persona to start roleplaying.</p>
            <button className="btn btn-primary" onClick={onNew}>+ New Character</button>
          </div>
        ) : (
          <div className="card-grid">
            {characters.map((c, i) => (
              <article
                key={c.id}
                className="char-card"
                style={{ animationDelay: `${i * 55}ms` }}
                onClick={() => onOpen(c)}
              >
                <div className="card-glow" />
                <div className="card-top">
                  <Avatar character={c} size={56} ring />
                  <button
                    className="card-edit"
                    onClick={(e) => { e.stopPropagation(); onEdit(c); }}
                    title="Edit character"
                  >
                    ✎
                  </button>
                </div>
                <h3 className="card-name">{c.name}</h3>
                <p className="card-tagline">
                  {c.tagline
                    ? c.tagline
                    : c.persona
                      ? c.persona.slice(0, 110) + (c.persona.length > 110 ? '…' : '')
                      : 'No description.'}
                </p>
                {c.tags && c.tags.length > 0 && (
                  <div className="card-tags">
                    {c.tags.slice(0, 3).map((t) => (
                      <span key={t} className="card-tag">{t}</span>
                    ))}
                  </div>
                )}
                <div className="card-foot">
                  <span className="card-cta">Enter scenario →</span>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
