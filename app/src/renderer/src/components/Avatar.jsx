import React from 'react';
import { hashHue, initials } from '../lib/util';

// Character avatar: image if provided, else a deterministic ember-tinted glyph.
export default function Avatar({ character, size = 48, ring = false }) {
  const name = character?.name || '?';
  const hue = hashHue(name);
  const px = `${size}px`;

  const style = {
    width: px,
    height: px,
    borderRadius: size > 80 ? '8px' : '50%',
    fontSize: `${size * 0.38}px`,
  };

  const ringStyle = ring
    ? { boxShadow: `0 0 0 1px var(--line-strong), 0 0 18px var(--ember-faint)` }
    : {};

  if (character?.avatar) {
    return (
      <img
        className="avatar avatar-img"
        src={character.avatar}
        alt={name}
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
