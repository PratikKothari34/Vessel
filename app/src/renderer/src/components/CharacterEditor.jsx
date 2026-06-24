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
  const [persona, setPersona] = useState(character?.persona || '');
  const [greeting, setGreeting] = useState(character?.greeting || '');
  const [sampling, setSampling] = useState(character?.sampling || {});
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);

  const setSlider = (key, v) => setSampling((s) => ({ ...s, [key]: Number(v) }));

  const save = async () => {
    if (!name.trim()) { setErr('Name is required.'); return; }
    setSaving(true); setErr(null);
    try {
      await onSave({ name, avatar, persona, greeting, sampling });
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
              const val = sampling[s.key] ?? s.def;
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

            <label className="field-label" style={{ marginTop: 18 }}>Persona</label>
            <textarea
              className="textarea" value={persona} onChange={(e) => setPersona(e.target.value)}
              placeholder="Who is this character? Personality, voice, backstory, quirks, the world they inhabit…"
              style={{ minHeight: 180 }}
            />

            <label className="field-label" style={{ marginTop: 18 }}>Opening greeting</label>
            <textarea
              className="textarea" value={greeting} onChange={(e) => setGreeting(e.target.value)}
              placeholder='The first line they say when a scenario begins. e.g. "You again."'
              style={{ minHeight: 90 }}
            />
          </div>
        </div>

        <div className="overlay-foot">
          {editing && (
            <button className="btn btn-danger" onClick={() => onDelete(character.id)}>Delete</button>
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
