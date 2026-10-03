import React, { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';
import { privacyState, syncState } from '../lib/settings-state.mjs';
import './settings.css';

/* Settings holds what a user can act on, and nothing else.
   Two things qualify: whether their writing is private, and the sync
   credentials -- the only editable values in the app.

   Deliberately NOT here: model ids, host URLs, and the memory tuning numbers.
   They are real, but they come from `.env`, cannot be changed from this window,
   and are shown where they are useful anyway -- the chat model in the header,
   memory internals in the Memory inspector. Printing them here told the user
   something true and unactionable on the way in.

   One page, not tabs: with the diagnostics gone there are two short sections,
   and the tab chrome promised depth that was not behind it. */

function Card({ tone = 'plain', title, children }) {
  return (
    <div className={`st-card is-${tone}`}>
      <h3 className="st-card-title">{title}</h3>
      <p className="st-card-text">{children}</p>
    </div>
  );
}

// Cloud-sync setup. Credentials are the user's own Turso database: URL saved to
// settings.json, token to the OS keychain (never echoed back -- only
// "saved / not saved"). Changes apply on the next app start, so a save that
// changes the running state says so and offers a restart.
export default function Settings({ health, onRefreshHealth, onClose }) {
  const privacy = privacyState(health);

  const [cfg, setCfg] = useState(null); // { tursoUrl, tokenSet, keychain, syncActive }
  const [url, setUrl] = useState('');
  const [token, setToken] = useState('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  const [err, setErr] = useState('');
  // Turning sync off deletes the auth token from the OS keychain, and a
  // packaged install has no .env to recover it from -- so it is a destructive
  // action, not a toggle. It costs a typed confirmation.
  const [confirming, setConfirming] = useState(false);
  const [confirmText, setConfirmText] = useState('');

  const load = useCallback(
    () =>
      api.getSettings()
        .then((s) => { setCfg(s); setUrl(s.tursoUrl || ''); })
        .catch((e) => setErr(`Could not load sync settings: ${e.message}`)),
    [],
  );

  // Read both when the panel opens, not just at app launch: this panel asserts
  // live state, and the app's health snapshot may predate whatever has happened
  // since the window was opened.
  useEffect(() => {
    load();
    if (onRefreshHealth) onRefreshHealth();
  }, [load, onRefreshHealth]);

  const save = async (patch) => {
    setBusy(true); setErr(''); setNote('');
    try {
      await api.saveSettings(patch);
      await load();
      setToken('');
      setNote('Saved.');
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  const onSave = () => {
    // Saving a blank URL over a configured one silently disables sync -- the
    // same outcome as the destructive button, reached by a control labelled
    // "Save". Route it through the same confirmation instead.
    if (!url.trim() && (cfg?.tursoUrl || cfg?.tokenSet)) {
      setConfirming(true);
      return;
    }
    const patch = { tursoUrl: url };
    if (token.trim()) patch.tursoToken = token;
    save(patch);
  };

  const onDisable = () => {
    setConfirming(false);
    setConfirmText('');
    // confirmClearToken is required by the backend before it will delete the
    // keychain entry; without it the request is refused outright.
    save({ tursoUrl: '', tursoToken: '', confirmClearToken: true });
  };

  // Both hosts can restart themselves; a browser-only dev renderer cannot.
  const canRelaunch = Boolean(api.relaunch);

  // Derived in settings-state.mjs so the suite can pin every combination of
  // saved settings against boot state without a DOM.
  const { ready, running, configured, pending, wasCleared } = syncState(cfg, health);

  return (
    <div className="overlay-backdrop" onClick={onClose}>
      <div className="overlay-panel settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="overlay-head">
          <div>
            <p className="kicker">Configuration</p>
            <h2 className="overlay-title">Settings</h2>
          </div>
          <button className="overlay-close" onClick={onClose} aria-label="Close settings">
            &#10005;
          </button>
        </div>

        <div className="overlay-body st-body">
          {health && health.status !== 'ok' && (
            <Card tone="danger" title="Vessel is not running">
              Vessel cannot reach its backend, so what is below may be out of date. If you run
              models on this machine, check that Ollama is running, then restart the app.
            </Card>
          )}

          {/* ── Privacy: the one read-only fact worth stating ────────────── */}
          {privacy && (
            <Card tone={privacy.tone} title={privacy.title}>
              {privacy.detail}
            </Card>
          )}

          {/* ── Sync: the only editable surface ──────────────────────────── */}
          <section className="st-section">
            <h3 className="st-section-title">Cloud sync</h3>

            {/* Until the settings load, `running` can only fall back to the
                app's health snapshot -- the stale value this whole change
                exists to stop trusting. Say nothing rather than flash it. */}
            <div className={`st-state ${ready && running ? 'is-on' : ''}`}>
              <span className="st-state-dot" />
              <div className="st-state-text">
                <strong>{!ready ? 'Checking sync...' : running ? 'Sync is on' : 'Sync is off'}</strong>
                <span>
                  {!ready
                    ? ''
                    : running
                    ? `Pushing and pulling every ${health?.sync?.interval ?? 60} seconds.`
                    : wasCleared
                      ? 'Your conversations stay on this machine. The cloud copy is untouched, but the saved token was deleted -- paste it again to resume.'
                      : 'Your conversations never leave this machine.'}
                </span>
              </div>
            </div>

            {/* Configured and running differ only until the next start, so name
                which one the user is looking at instead of a bare "restart to
                apply" that reads as the app being unfinished. */}
            {pending && (
              <p className="st-msg is-warn">
                {configured
                  ? 'Sync starts the next time you open Vessel.'
                  : 'Sync stops the next time you open Vessel. It is still running now.'}
                {canRelaunch && (
                  <button className="btn btn-ghost st-link st-inline-link" onClick={() => api.relaunch()}>
                    Restart now
                  </button>
                )}
              </p>
            )}

            <p className="st-prose">
              Vessel runs no server of its own. Sync copies your conversations to a database
              <strong> you own</strong> at <code>turso.tech</code>. Leave these blank to stay
              fully local.
            </p>

            <div className="st-field">
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
            </div>

            <div className="st-field">
              <label className="field-label" htmlFor="turso-token">Auth token</label>
              <input
                id="turso-token"
                className="input"
                type="password"
                placeholder={cfg?.tokenSet ? 'Saved - type here to replace' : 'Paste your database token'}
                value={token}
                onChange={(e) => setToken(e.target.value)}
                disabled={busy}
              />
              <p className="st-field-note">
                {cfg?.tokenSet
                  ? 'A token is saved in your OS keychain, never on disk.'
                  : 'Stored in your OS keychain, never on disk.'}
              </p>
            </div>

            {cfg && !cfg.keychain && (
              <p className="st-msg is-danger">
                Your OS keychain is unavailable, so the token cannot be stored securely and sync
                cannot be turned on.
              </p>
            )}

            <div className="st-actions">
              <button className="btn btn-primary" onClick={onSave} disabled={busy || !cfg}>
                {busy ? 'Saving...' : 'Save'}
              </button>
              {(cfg?.tursoUrl || cfg?.tokenSet) && !confirming && (
                <button
                  className="btn btn-ghost st-danger-link"
                  onClick={() => setConfirming(true)}
                  disabled={busy}
                >
                  Turn off sync
                </button>
              )}
            </div>

            {confirming && (
              <div className="st-confirm">
                <h4 className="st-confirm-title">Turn off sync?</h4>
                <p className="st-confirm-text">
                  This deletes the saved auth token from your keychain. Vessel cannot get it
                  back &mdash; you would need to copy it from your Turso dashboard again. Your
                  conversations stay on this machine and the cloud copy is left untouched.
                </p>
                <label className="field-label" htmlFor="confirm-off">
                  Type <strong>off</strong> to confirm
                </label>
                <input
                  id="confirm-off"
                  className="input"
                  type="text"
                  value={confirmText}
                  onChange={(e) => setConfirmText(e.target.value)}
                  autoComplete="off"
                  spellCheck={false}
                  disabled={busy}
                />
                <div className="st-actions">
                  <button
                    className="btn btn-danger"
                    onClick={onDisable}
                    disabled={busy || confirmText.trim().toLowerCase() !== 'off'}
                  >
                    Turn off sync
                  </button>
                  <button
                    className="btn btn-ghost"
                    onClick={() => { setConfirming(false); setConfirmText(''); }}
                    disabled={busy}
                  >
                    Keep sync on
                  </button>
                </div>
              </div>
            )}

            {note && <p className="st-msg is-ok">{note}</p>}
            {err && <p className="st-msg is-danger">{err}</p>}
          </section>
        </div>
      </div>
    </div>
  );
}
