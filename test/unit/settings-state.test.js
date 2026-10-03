'use strict';

/**
 * What the Settings panel tells the user about sync and privacy.
 *
 * This had no coverage, and the defect it hid was not cosmetic: the panel read
 * the app's launch-time health snapshot, so after a user turned sync off it went
 * on insisting sync was on until the next restart. The same screen is where
 * someone decides whether their writing is private, so a stale or flattering
 * reading there is the worst kind.
 *
 * The cases below are the real state combinations, including the one that
 * actually happened: an accidental "turn off sync" that deleted the stored
 * token while the session kept syncing.
 *
 * Dynamic import because the module is ESM under a CommonJS package.
 */

const test = require('node:test');
const assert = require('node:assert/strict');

let privacyState, syncState;
test.before(async () => {
  ({ privacyState, syncState } = await import(
    '../../app/src/renderer/src/lib/settings-state.mjs'
  ));
});

const URL = 'libsql://db-org.turso.io';

// ---- Privacy -------------------------------------------------------------

test('an encrypted database reads as encrypted, with no alarm', () => {
  const s = privacyState({ status: 'ok', encryptedAtRest: true });
  assert.equal(s.tone, 'ok');
  assert.match(s.title, /encrypted/i);
});

test('sync explains the tradeoff and never claims encryption', () => {
  // The user chose this, so it warns rather than alarms -- but it must not
  // imply a setting exists that turns sync and encryption on together.
  const s = privacyState({
    status: 'ok',
    encryptedAtRest: false,
    unencryptedReason: 'sync',
  });
  assert.equal(s.tone, 'warn');
  assert.match(s.detail, /cannot both be on/i);
  assert.match(s.detail, /no backup/i);
});

test('a broken keychain is a danger, not a tradeoff', () => {
  // Nobody opted into this, so it must not be dressed in the same tone as a
  // choice the user made.
  const s = privacyState({
    status: 'ok',
    encryptedAtRest: false,
    unencryptedReason: 'no-key',
  });
  assert.equal(s.tone, 'danger');
  assert.match(s.detail, /keychain/i);
});

test('a half-finished migration is a danger too', () => {
  const s = privacyState({
    status: 'ok',
    encryptedAtRest: false,
    unencryptedReason: 'migration',
  });
  assert.equal(s.tone, 'danger');
  assert.match(s.detail, /did not finish/i);
});

test('an unknown state renders nothing rather than reassurance', () => {
  // A down backend must never read as "your conversations are encrypted".
  assert.equal(privacyState(null), null);
  assert.equal(privacyState({ status: 'down' }), null);
  assert.equal(privacyState(undefined), null);
});

// ---- Sync ----------------------------------------------------------------

test('before the settings load, the panel admits it does not know yet', () => {
  // The fallback is the stale health snapshot -- the very value this module
  // exists to stop trusting. It must not be presented as settled.
  const s = syncState(null, { sync: { enabled: true } });
  assert.equal(s.ready, false);
  assert.equal(s.pending, false, 'nothing is pending against a guess');
  assert.equal(s.wasCleared, false, 'nothing is "lost" against a guess');
});

test('sync configured and running needs no restart', () => {
  const s = syncState(
    { tursoUrl: URL, tokenSet: true, syncActive: true },
    { sync: { enabled: true } },
  );
  assert.equal(s.running, true);
  assert.equal(s.configured, true);
  assert.equal(s.pending, false);
});

test('never configured is simply off, with nothing lost', () => {
  const s = syncState(
    { tursoUrl: '', tokenSet: false, syncActive: false },
    { sync: { enabled: false } },
  );
  assert.equal(s.running, false);
  assert.equal(s.configured, false);
  assert.equal(s.pending, false);
  assert.equal(s.wasCleared, false, 'nothing was ever there to lose');
});

test('credentials saved this session start syncing on the next launch', () => {
  const s = syncState(
    { tursoUrl: URL, tokenSet: true, syncActive: false },
    { sync: { enabled: false } },
  );
  assert.equal(s.running, false, 'this boot never connected');
  assert.equal(s.configured, true);
  assert.equal(s.pending, true, 'the restart turns it on');
});

test('an accidental turn-off is reported as a loss, not as privacy', () => {
  // The incident this module was written for. Sync is still running because the
  // boot connected before the change, but both credentials are gone -- and a
  // packaged install has no .env to recover them from. Reporting this as
  // "your conversations never leave this machine" hides a destroyed secret
  // behind a reassuring sentence.
  const s = syncState(
    { tursoUrl: '', tokenSet: false, syncActive: true },
    { sync: { enabled: true } },
  );
  assert.equal(s.running, true, 'this boot is still syncing');
  assert.equal(s.configured, false, 'the next boot will not');
  assert.equal(s.pending, true, 'the user must be told a restart changes this');
  assert.equal(s.wasCleared, true, 'and that a credential was deleted');
});

test('a kept token is not a cleared one, even with the URL gone', () => {
  // Only the URL was cleared, so nothing irreversible happened: the token is
  // still in the keychain and pasting a URL back is enough. Claiming a loss
  // here would send the user to their Turso dashboard for no reason.
  const s = syncState(
    { tursoUrl: '', tokenSet: true, syncActive: true },
    { sync: { enabled: true } },
  );
  assert.equal(s.pending, true);
  assert.equal(s.wasCleared, false);
});

test('the boot state wins over the health snapshot once settings load', () => {
  // The whole bug in one assertion: health says sync is on, the settings say
  // this boot did not connect, and the settings are the ones that know.
  const s = syncState(
    { tursoUrl: '', tokenSet: false, syncActive: false },
    { sync: { enabled: true } },
  );
  assert.equal(s.running, false);
});
