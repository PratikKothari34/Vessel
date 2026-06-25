import React, { useState, useEffect } from 'react';
import { hashHue, initials } from '../lib/util';

// Only allow web image URLs and inline image data. A character's avatar string
// is untrusted (hand-entered, imported, or synced from another device); without
// this a crafted value like file:///C:/... could make the app fetch a local
// file. javascript: can't execute via <img>, but we drop it too for clarity.
function safeAvatarSrc(src) {
  if (typeof src !== 'string' || !src.trim()) return null;
  const s = src.trim();
  if (/^https?:\/\//i.test(s)) return s;
  if (/^data:image\//i.test(s)) return s;
  return null;
}

// Character avatar: image if provided, else a deterministic tinted glyph.
// If the image fails to load (dead URL, bad path, offline), fall back to the glyph.
export default function Avatar({ character, size = 48, ring = false }) {
  const name = character?.name || '?';
  const hue = hashHue(name);
  const px = `${size}px`;
  const [imgFailed, setImgFailed] = useState(false);

  // Reset the failure flag when the avatar source changes.
  useEffect(() => { setImgFailed(false); }, [character?.avatar]);

  const style = {
    width: px,
    height: px,
    borderRadius: size > 80 ? '8px' : '50%',
    fontSize: `${size * 0.38}px`,
  };

  const ringStyle = ring
    ? { boxShadow: `0 0 0 1px var(--line-strong), 0 0 18px var(--ember-faint)` }
    : {};

  const safeSrc = safeAvatarSrc(character?.avatar);
  if (safeSrc && !imgFailed) {
    return (
      <img
        className="avatar avatar-img"
        src={safeSrc}
        alt={name}
        onError={() => setImgFailed(true)}
        style={{ ...style, ...ringStyle, objectFit: 'cover' }}
      />
    );
  }

  return (
    <div
      className="avatar avatar-glyph"
      style={{
        ...style,
        ...ringStyle,
        background: `radial-gradient(120% 120% at 30% 20%, hsl(${hue} 45% 22%), hsl(${(hue + 40) % 360} 35% 10%))`,
        color: `hsl(${hue} 60% 78%)`,
      }}
    >
      {initials(name)}
    </div>
  );
}
