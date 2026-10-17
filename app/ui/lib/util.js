// Small shared helpers for the renderer.
//
// Every function here takes UNTRUSTED input: a character name is hand-entered,
// imported, or synced from another device, and a timestamp can come from a row
// written by an older schema. They are called during render, so a throw here
// unmounts the chat - each one is total over its argument type rather than
// relying on the caller to pre-validate.

// Deterministic ember/steel hue per character name, for avatar fallbacks.
export function hashHue(str) {
  // Called with whatever the caller has; a non-string must not throw mid-render.
  const s = typeof str === 'string' ? str : String(str ?? '');
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return h;
}

export function initials(name) {
  if (typeof name !== 'string') return '?';
  // Trim BEFORE the emptiness test: a whitespace-only name is truthy but has no
  // initials, and returning '' there renders a blank avatar instead of the '?'.
  const trimmed = name.trim();
  if (!trimmed) return '?';
  const parts = trimmed.split(/\s+/);
  // Index by code POINT, not code unit. slice(0, 2) on an astral character
  // splits its surrogate pair and yields a lone high surrogate, which is not
  // well-formed text and renders as a replacement glyph.
  const glyphs = (word) => Array.from(word);
  if (parts.length === 1) return glyphs(parts[0]).slice(0, 2).join('').toUpperCase();
  const first = glyphs(parts[0])[0];
  const last = glyphs(parts[parts.length - 1])[0];
  return (first + last).toUpperCase();
}

export function relTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  // An unparseable value gives NaN time; every comparison below is then false
  // and toLocaleDateString() renders the string 'Invalid Date' into the UI.
  if (Number.isNaN(d.getTime())) return '';
  const s = (Date.now() - d.getTime()) / 1000;
  // A future timestamp (clock skew between synced devices) reads as 'just now'
  // rather than falling through to a negative minutes-ago figure.
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 604800) return `${Math.floor(s / 86400)}d ago`;
  return d.toLocaleDateString();
}
