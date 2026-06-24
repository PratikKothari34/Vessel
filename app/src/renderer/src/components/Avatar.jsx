import React, { useState, useEffect } from 'react';
import { hashHue, initials } from '../lib/util';

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

  if (character?.avatar && !imgFailed) {
    return (
      <img
        className="avatar avatar-img"
        src={character.avatar}
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
