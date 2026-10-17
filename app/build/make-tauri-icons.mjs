// Generates the icon set Tauri's bundler expects (`src-tauri/icons/`, listed in
// tauri.conf.json) from the same source art in this directory.
//
// Step 2 of 2: `make-icon.mjs` cuts `icon.ico` + `icon.png` from the source
// artwork first, and this copies/resizes them into the shape Tauri wants.
//
// ONE-SHOT ARTWORK TOOL - `sharp` is deliberately NOT a dependency of this repo
// (nothing in the app needs an image pipeline at runtime or build time). Install
// it ad hoc when the artwork actually changes:
//   cd app && npm i --no-save sharp && node build/make-tauri-icons.mjs
// The committed PNG/ICO outputs are what the build consumes, not this script.
import sharp from 'sharp';
import { mkdirSync, copyFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, 'icon-source.png');
const out = join(here, '..', '..', 'src-tauri', 'icons');
mkdirSync(out, { recursive: true });

const sizes = [
  [32, '32x32.png'],
  [128, '128x128.png'],
  [256, '128x128@2x.png'],
];

for (const [px, name] of sizes) {
  await sharp(src).resize(px, px, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .png().toFile(join(out, name));
  console.log(`wrote ${name} (${px}px)`);
}

// The .ico is already cut at the right sizes for this app — reuse it rather
// than re-encoding and risking a different set of embedded resolutions.
copyFileSync(join(here, 'icon.ico'), join(out, 'icon.ico'));
console.log('copied icon.ico');
