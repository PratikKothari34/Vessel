import React from 'react';
import './settings.css';

// Read-only view of backend/runtime config. Editing lives in the .env file —
// surfaced here so the user knows what's wired up.
export default function Settings({ health, onClose }) {
  const rows = health && health.status === 'ok'
    ? [
        ['Chat model', health.model],
        ['Ollama host', health.ollama],
        ['Summarizer', health.memory?.summarizer],
        ['Embedder', health.memory?.embedder],
        ['Verbatim turns', health.memory?.verbatimTurns],
        ['Summarize threshold', health.memory?.summarizeThreshold],
        ['Cloud sync', health.sync?.enabled ? `enabled (${health.sync.interval}s)` : 'local-only'],
      ]
    : [];

  return (
    <div className="overlay-backdrop" onClick={onClose}>
      <div className="overlay-panel settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="overlay-head">
          <div>
            <p className="kicker">Configuration</p>
            <h2 className="overlay-title">Settings</h2>
          </div>
          <button className="overlay-close" onClick={onClose}>✕</button>
        </div>

        <div className="overlay-body">
          <div className={`settings-status ${health?.status === 'ok' ? 'ok' : 'down'}`}>
            <span className="dot" />
            {health?.status === 'ok' ? 'Backend connected' : 'Backend offline — is Ollama running?'}
          </div>

          {rows.length > 0 && (
            <div className="settings-table">
              {rows.map(([k, v]) => (
                <div key={k} className="settings-row">
                  <span className="settings-key">{k}</span>
                  <span className="settings-val">{String(v)}</span>
                </div>
              ))}
            </div>
          )}

          <div className="settings-note">
            <p className="kicker">Editing config</p>
            <p>
              Models, memory tuning, and Turso cloud sync are configured in the
              <code> .env </code> file at the project root (copy from <code>.env.example</code>).
              Restart the app after changing it.
            </p>
            <p style={{ marginTop: 10 }}>
              Leave <code>TURSO_DATABASE_URL</code> blank to stay fully local. Fill it in
              (with <code>TURSO_AUTH_TOKEN</code>) to enable encrypted cloud backup + multi-device sync.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
