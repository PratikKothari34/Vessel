import React, { useState } from 'react';
import Avatar from './Avatar.jsx';
import './editor.css';

const SLIDERS = [
  { key: 'temperature', label: 'Temperature', min: 0, max: 1.5, step: 0.05, def: 0.9, hint: 'Creativity / randomness' },
  { key: 'top_p', label: 'Top-P', min: 0.1, max: 1, step: 0.05, def: 0.95, hint: 'Nucleus sampling' },
  { key: 'repeat_penalty', label: 'Repeat penalty', min: 1, max: 1.5, step: 0.05, def: 1.1, hint: 'Discourages repetition' },
];

export default function CharacterEditor({ character, onClose, onSave, onDelete }) {
  const editing = character && character.id;
  const [name, setName] = useState(character?.name || '');
  const [avatar, setAvatar] = useState(character?.avatar || '');
  const [tagline, setTagline] = useState(character?.tagline || '');
  const [about, setAbout] = useState(character?.about || '');
  const [persona, setPersona] = useState(character?.persona || '');
  const [greeting, setGreeting] = useState(character?.greeting || '');
  // chat starters: one per line in the textarea; tags: comma-separated.
  const [startersText, setStartersText] = useState((character?.chatStarters || []).join('\n'));
  const [tagsText, setTagsText] = useState((character?.tags || []).join(', '));
  const [sampling, setSampling] = useState(character?.sampling || {});
  const [responseStyle, setResponseStyle] = useState(character?.responseStyle || 'balanced');
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);

  const setSlider = (key, v) => setSampling((s) => ({ ...s, [key]: Number(v) }));

  const save = async () => {
    if (!name.trim()) { setErr('Name is required.'); return; }
    setSaving(true); setErr(null);
    const chatStarters = startersText.split('\n').map((s) => s.trim()).filter(Boolean);
    const tags = tagsText.split(',').map((s) => s.trim()).filter(Boolean);
    try {
      await onSave({ name, avatar, tagline, about, persona, greeting, chatStarters, tags, sampling, responseStyle });
      // On success the parent unmounts this editor; reset anyway so the button
      // never gets stuck on "Saving…" if it stays mounted.
      setSaving(false);
    } catch (e) { setErr(e.message); setSaving(false); }
  };

  const preview = { name: name || 'New Character', avatar };

  return (
    <div className="overlay-backdrop" onClick={onClose}>
      <div className="overlay-panel editor-panel" onClick={(e) => e.stopPropagation()}>
        <div className="overlay-head">
          <div>
            <p className="kicker">{editing ? 'Edit dossier' : 'New dossier'}</p>
            <h2 className="overlay-title">{editing ? name || 'Character' : 'Forge a character'}</h2>
          </div>
          <button className="overlay-close" onClick={onClose}>✕</button>
        </div>

        <div className="overlay-body editor-grid">
          <div className="editor-left">
            <div className="editor-preview">
              <Avatar character={preview} size={96} ring />
              <span className="editor-preview-name">{preview.name}</span>
            </div>
            <label className="field-label">Avatar URL / path</label>
            <input className="input" value={avatar} onChange={(e) => setAvatar(e.target.value)} placeholder="https://… or leave blank" />

            <label className="field-label" style={{ marginTop: 22 }}>Sampling</label>
            {SLIDERS.map((s) => {
              // Coerce defensively: a stored/imported/synced sampling value may
              // be a non-number (e.g. a stringified JSON value); Number(...) +
              // fallback keeps .toFixed and the range input from crashing render.
              const raw = Number(sampling[s.key]);
              const val = Number.isFinite(raw) ? raw : s.def;
              return (
                <div key={s.key} className="slider-row">
                  <div className="slider-top">
                    <span className="slider-label">{s.label}</span>
                    <span className="slider-val">{val.toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min={s.min} max={s.max} step={s.step} value={val}
                    onChange={(e) => setSlider(s.key, e.target.value)}
                    className="slider"
                  />
                  <span className="slider-hint">{s.hint}</span>
                </div>
              );
            })}
          </div>

          <div className="editor-right">
            <label className="field-label">Name</label>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Aria Vance" autoFocus />

            <label className="field-label" style={{ marginTop: 18 }}>Tagline</label>
            <input
              className="input" value={tagline} onChange={(e) => setTagline(e.target.value)}
              placeholder="Short hook shown on the card. e.g. A billionaire interested in you"
            />

            <label className="field-label" style={{ marginTop: 18 }}>Tags <span className="field-hint">comma-separated</span></label>
            <input
              className="input" value={tagsText} onChange={(e) => setTagsText(e.target.value)}
              placeholder="Romance, Lifestyle, Drama"
            />

            <label className="field-label" style={{ marginTop: 18 }}>Persona <span className="field-hint">defines behavior</span></label>
            <textarea
              className="textarea" value={persona} onChange={(e) => setPersona(e.target.value)}
              placeholder="Who is this character? Personality, voice, backstory, quirks, the world they inhabit…"
              style={{ minHeight: 160 }}
            />

            <label className="field-label" style={{ marginTop: 18 }}>About <span className="field-hint">public blurb, optional</span></label>
            <textarea
              className="textarea" value={about} onChange={(e) => setAbout(e.target.value)}
              placeholder="A short description shown on the character's profile."
              style={{ minHeight: 70 }}
            />

            <label className="field-label" style={{ marginTop: 18 }}>Opening greeting</label>
            <textarea
              className="textarea" value={greeting} onChange={(e) => setGreeting(e.target.value)}
              placeholder='The first line they say when a scenario begins. e.g. "You again."'
              style={{ minHeight: 80 }}
            />

            <label className="field-label" style={{ marginTop: 18 }}>Chat starters <span className="field-hint">one per line</span></label>
            <textarea
              className="textarea" value={startersText} onChange={(e) => setStartersText(e.target.value)}
              placeholder={"Suggested openers shown on a new chat.\nWhat's your favorite movie?\nTell me about your day."}
              style={{ minHeight: 90 }}
            />

            <label className="field-label" style={{ marginTop: 18 }}>Response style</label>
            <div className="style-options">
              {[
                { v: 'balanced', t: 'Balanced', d: 'Mix of dialogue & narration' },
                { v: 'dialogue', t: 'Dialogue-first', d: 'Always speaks; minimal narration' },
                { v: 'narration-light', t: 'Light narration', d: 'Brief scene-setting' },
              ].map((o) => (
                <button
                  key={o.v}
                  type="button"
                  className={`style-opt ${responseStyle === o.v ? 'active' : ''}`}
                  onClick={() => setResponseStyle(o.v)}
                >
                  <span className="style-opt-t">{o.t}</span>
                  <span className="style-opt-d">{o.d}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="overlay-foot">
          {editing && (
            <button
              className="btn btn-danger"
              onClick={async () => {
                // Deleting a character removes it and every scenario/memory
                // tied to it. Irreversible — confirm first.
                if (window.confirm(`Delete "${name || 'this character'}"? This removes the character and all its scenarios and memory. This cannot be undone.`)) {
                  try { await onDelete(character.id); }
                  catch (e) { setErr(`Could not delete: ${e.message}`); }
                }
              }}
            >
              Delete
            </button>
          )}
          {err && <span className="editor-err">{err}</span>}
          <div style={{ flex: 1 }} />
          <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={save} disabled={saving}>
            {saving ? 'Saving…' : editing ? 'Save changes' : 'Create character'}
          </button>
        </div>
      </div>
    </div>
  );
}
