/**
 * What the Settings panel says about sync and privacy.
 *
 * Kept apart from the component for the same reason as prose.mjs: this is where
 * the decisions are, and where the bug was. The panel spent its life reporting
 * a launch-time snapshot as live state, so after a user turned sync off it kept
 * saying sync was on until the next restart. Deriving that here, as plain data
 * in and plain data out, lets the suite pin every combination without a DOM.
 *
 * The distinction that fixes it: `/settings` returns BOTH what the next boot
 * will do (`tursoUrl`) and what this one actually connected with
 * (`syncActive`). The old panel read neither -- it read the app's health
 * snapshot -- and conflating the three is what let it lie.
 *
 * .mjs rather than .js because app/package.json declares no type, so Node reads
 * a .js file here as CommonJS and the tests could not import it.
 */

/**
 * Is the local database encrypted, and if not, why not?
 *
 * Answers for the database THIS BOOT opened, not for whatever the form fields
 * are currently set to. `unencryptedReason` is what separates a tradeoff the
 * user chose (sync) from a failure they never saw (keychain, half-done
 * migration) -- only the failure warrants alarm, so only it gets the danger
 * tone.
 *
 * Returns null when there is no usable health reading: an unknown state must
 * render as nothing, never as reassurance.
 */
export function privacyState(health) {
  if (!health || health.status !== 'ok') return null;

  if (health.encryptedAtRest) {
    return {
      tone: 'ok',
      title: 'Your conversations are encrypted',
      detail: 'They are stored on this machine only, with the key in your OS keychain.',
    };
  }

  if (health.unencryptedReason === 'sync') {
    // Name the trade as a trade. Cloud sync and local encryption are mutually
    // exclusive here, and implying otherwise sends the user hunting for a
    // setting that would turn both on.
    return {
      tone: 'warn',
      title: 'Readable on this machine',
      detail:
        'Cloud sync and local encryption cannot both be on -- the sync engine cannot open an encrypted file. '
        + 'Anyone with access to your files can read your conversations. Keeping sync means accepting that; '
        + 'turning it off gives you an encrypted database but no backup.',
    };
  }

  return {
    tone: 'danger',
    title: 'Readable on this machine',
    detail:
      health.unencryptedReason === 'no-key'
        ? 'Your OS keychain was unavailable, so no encryption key could be stored. Restarting usually fixes this.'
        : 'Encrypting your database did not finish. Restarting usually fixes this.',
  };
}

/**
 * Derive what the sync section should claim, from the saved settings (`cfg`,
 * null until they load) and the app's health reading.
 *
 *   ready      -- have the real settings arrived? Until they have, every field
 *                 below is a guess from the stale health snapshot, so the panel
 *                 must say "checking" rather than flash a value it will correct.
 *   running    -- what THIS BOOT connected with. Authoritative for "is it on".
 *   configured -- what the saved settings say the NEXT boot will do.
 *   pending    -- the two disagree, so a restart changes something. The panel
 *                 names which direction instead of a bare "restart to apply",
 *                 which reads as the app being unfinished.
 *   wasCleared -- sync is running but both the URL and the token are gone: the
 *                 user turned it off this session and lost a credential Vessel
 *                 cannot regenerate. Indistinguishable from "never configured"
 *                 without this, and only one of them means something was lost.
 */
export function syncState(cfg, health) {
  const ready = Boolean(cfg);
  const running = ready ? Boolean(cfg.syncActive) : Boolean(health?.sync?.enabled);
  const configured = Boolean(cfg?.tursoUrl);

  return {
    ready,
    running,
    configured,
    // Both only mean anything once the real settings are in hand; computing
    // them from the fallback would be comparing a guess against a guess.
    pending: ready && running !== configured,
    wasCleared: ready && running && !configured && !cfg.tokenSet,
  };
}
