import React, { useEffect, useState } from 'react';
import { api } from '../lib/api';
import './settings.css';

// Runtime config view + cloud-sync setup. Sync credentials are the user's own
// Turso database: URL saved to the backend's settings.json, token to the OS
// keychain (never echoed back — only "saved / not saved"). Changes apply on
// the next app start, so a successful save offers a restart.
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

  const [cfg, setCfg] = useState(null); // { tursoUrl, tokenSet, keychain, syncActive }
  const [url, setUrl] = useState('');
  const [token, setToken] = useState('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  const [err, setErr] = useState('');

  useEffect(() => {
    api.getSettings()
      .then((s) => { setCfg(s); setUrl(s.tursoUrl || ''); })
      .catch((e) => setErr(`Could not load sync settings: ${e.message}`));
  }, []);

  const save = async (patch) => {
    setBusy(true); setErr(''); setNote('');
    try {
      await api.saveSettings(patch);
      const s = await api.getSettings();
      setCfg(s); setUrl(s.tursoUrl || ''); setToken('');
      setNote('Saved. Restart Vessel to apply.');
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  const onSave = () => {
    const patch = { tursoUrl: url };
    if (token.trim()) patch.tursoToken = token;
    save(patch);
  };
  const onDisable = () => save({ tursoUrl: '', tursoToken: '' });

  const canRelaunch = typeof window !== 'undefined' && window.scenario?.relaunch;

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

          <div className="settings-sync">
            <p className="kicker">Cloud sync — your own Turso database</p>
            <p className="settings-sync-blurb">
              Optional. Create a free database at <code>turso.tech</code>, then paste its URL and
              an auth token here. Leave blank to stay fully local. The token is stored in the
              OS keychain, not on disk.
            </p>

            <label className="field-label" htmlFor="turso-url">Database URL</label>
            <input
              id="turso-url"
              className="input"
              type="text"
              placeholder="libsql://your-db-your-org.turso.io"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              disabled={busy}
              spellCheck={false}
            />

            <label className="field-label" htmlFor="turso-token">Auth token</label>
            <input
              id="turso-token"
              className="input"
              type="password"
              placeholder={cfg?.tokenSet ? 'Saved — type here to replace' : 'Paste your database token'}
              value={token}
              onChange={(e) => setToken(e.target.value)}
              disabled={busy}
            />

            {cfg && !cfg.keychain && (
              <p className="settings-sync-warn">
                OS keychain unavailable — the token can’t be stored securely, so sync can’t be enabled.
              </p>
            )}

            <div className="settings-sync-actions">
              <button className="btn btn-primary" onClick={onSave} disabled={busy || !cfg}>
                Save sync settings
              </button>
              {(cfg?.tursoUrl || cfg?.tokenSet) && (
                <button className="btn btn-danger" onClick={onDisable} disabled={busy}>
                  Disable sync
                </button>
              )}
              {note && canRelaunch && (
                <button className="btn" onClick={() => window.scenario.relaunch()}>
                  Restart now
                </button>
              )}
            </div>

            {note && <p className="settings-sync-note">{note}</p>}
            {err && <p className="settings-sync-warn">{err}</p>}
          </div>

          <div className="settings-note">
            <p className="kicker">Advanced config</p>
            <p>
              Models and memory tuning come from the <code>.env</code> file at the project root
              (dev setups — copy from <code>.env.example</code>). Sync credentials set above
              override <code>.env</code>. Restart the app after changing either.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
