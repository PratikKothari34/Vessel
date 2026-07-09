// Builds app/build/icon.ico (16..256, the Windows max) and a 4K app/build/icon.png
// from the source flask artwork, using sharp.
//
// Pipeline: detect the flask's bounding box (bright pixels vs the dark bg) →
// crop tight (this drops the bottom UI-icon row and the bottom-right watermark) →
// composite onto a dark rounded-square tile with a subtle radial glow → export.
//
// Run from repo root: node app/build/make-icon.mjs
import sharp from 'sharp';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(__dirname, 'icon-source.png');
const ICO_SIZES = [16, 24, 32, 48, 64, 128, 256];
const TILE_4K = 1024; // master tile size we render the icon at, then downscale
const BG = { r: 10, g: 10, b: 15 }; // #0a0a0f app background

// --- 1. find the flask bounding box -----------------------------------------
async function findBBox() {
  const { data, info } = await sharp(SRC).raw().toBuffer({ resolveWithObject: true });
  const { width, height, channels: ch } = info;
  const THRESH = 78; // luminance above this = flask/glow (higher: ignore the blurry bg shelf)
  // Count lit pixels per column/row, then take the bbox of columns/rows whose
  // lit-count exceeds a density floor — a single stray reflection won't widen it.
  const colCount = new Uint32Array(width);
  const rowCount = new Uint32Array(height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * ch;
      const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      if (lum > THRESH) { colCount[x]++; rowCount[y]++; }
    }
  }
  const colFloor = Math.round(height * 0.12); // column needs >=12% of its height lit to count as flask body
  const rowFloor = Math.round(width * 0.04);
  let minX = width, maxX = 0, minY = height, maxY = 0;
  for (let x = 0; x < width; x++) if (colCount[x] > colFloor) { if (x < minX) minX = x; if (x > maxX) maxX = x; }
  for (let y = 0; y < height; y++) if (rowCount[y] > rowFloor) { if (y < minY) minY = y; if (y > maxY) maxY = y; }
  // The flask is horizontally centered in the source; symmetrize the X bbox
  // around the image center so a one-sided table reflection can't skew the crop.
  const cx = width / 2;
  const halfX = Math.max(cx - minX, maxX - cx);
  minX = Math.max(0, Math.round(cx - halfX));
  maxX = Math.min(width - 1, Math.round(cx + halfX));
  return { minX, maxX, minY, maxY, width, height };
}

// --- 2. rounded-rect mask + radial glow tile (as raw RGBA PNG buffers) -------
function roundedMaskSVG(size, radius) {
  return Buffer.from(
    `<svg width="${size}" height="${size}"><rect x="0" y="0" width="${size}" height="${size}" rx="${radius}" ry="${radius}" fill="#fff"/></svg>`
  );
}
function tileBackgroundSVG(size) {
  const r = Math.round(size * 0.16);
  const cx = size * 0.5, cy = size * 0.46;
  // dark base + a soft cyan/violet radial bloom behind where the orb will sit
  return Buffer.from(`
    <svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="bloom" cx="50%" cy="46%" r="55%">
          <stop offset="0%"  stop-color="#2a3550" stop-opacity="0.9"/>
          <stop offset="45%" stop-color="#171826" stop-opacity="0.7"/>
          <stop offset="100%" stop-color="#0a0a0f" stop-opacity="1"/>
        </radialGradient>
        <linearGradient id="wash" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%"  stop-color="#101018"/>
          <stop offset="100%" stop-color="#14101c"/>
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="${size}" height="${size}" rx="${r}" ry="${r}" fill="url(#wash)"/>
      <rect x="0" y="0" width="${size}" height="${size}" rx="${r}" ry="${r}" fill="url(#bloom)"/>
    </svg>`);
}

// --- 3. assemble the master tile --------------------------------------------
async function buildMasterTile() {
  const bbox = await findBBox();
  // pad the crop a touch, but crop the BOTTOM tighter to drop the icon row + base
  const padX = Math.round((bbox.maxX - bbox.minX) * 0.03);
  const cropLeft = Math.max(0, bbox.minX - padX);
  const cropTop = Math.max(0, bbox.minY - padX);
  const cropRight = Math.min(bbox.width, bbox.maxX + padX);
  // cut the bottom at ~82% of the flask height: removes the UI-icon row + stand,
  // keeps the orb centered. (icon row sits low in the body.)
  const flaskH = bbox.maxY - bbox.minY;
  const cropBottom = Math.min(bbox.height, bbox.minY + Math.round(flaskH * 0.74));
  const cropW = cropRight - cropLeft;
  const cropH = cropBottom - cropTop;
  console.log(`bbox x${bbox.minX}-${bbox.maxX} y${bbox.minY}-${bbox.maxY}`);
  console.log(`crop ${cropW}x${cropH} at (${cropLeft},${cropTop}) — bottom trimmed for icons/stand`);

  // extract the flask, scale it to sit inside ~78% of the tile, centered
  const inner = Math.round(TILE_4K * 0.88);
  const flask = await sharp(SRC)
    .extract({ left: cropLeft, top: cropTop, width: cropW, height: cropH })
    .resize({ width: inner, height: inner, fit: 'inside' })
    .png()
    .toBuffer();
  const fMeta = await sharp(flask).metadata();

  const tileBg = await sharp(tileBackgroundSVG(TILE_4K)).png().toBuffer();
  const composed = await sharp(tileBg)
    .composite([{
      input: flask,
      left: Math.round((TILE_4K - fMeta.width) / 2),
      top: Math.round((TILE_4K - fMeta.height) / 2) + Math.round(TILE_4K * 0.02),
    }])
    .png()
    .toBuffer();

  // clip everything to the rounded-square mask
  const mask = await sharp(roundedMaskSVG(TILE_4K, Math.round(TILE_4K * 0.16))).png().toBuffer();
  const tile = await sharp(composed)
    .composite([{ input: mask, blend: 'dest-in' }])
    .png()
    .toBuffer();
  return tile;
}

// --- 4. pack multi-size ICO from PNG frames ---------------------------------
function buildICO(frames) {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0); header.writeUInt16LE(1, 2); header.writeUInt16LE(frames.length, 4);
  const dir = Buffer.alloc(16 * frames.length);
  let offset = 6 + dir.length;
  const blobs = [];
  frames.forEach((f, i) => {
    const o = i * 16;
    dir[o] = f.size >= 256 ? 0 : f.size;
    dir[o + 1] = f.size >= 256 ? 0 : f.size;
    dir.writeUInt16LE(1, o + 4);
    dir.writeUInt16LE(32, o + 6);
    dir.writeUInt32LE(f.png.length, o + 8);
    dir.writeUInt32LE(offset, o + 12);
    offset += f.png.length;
    blobs.push(f.png);
  });
  return Buffer.concat([header, dir, ...blobs]);
}

// --- run --------------------------------------------------------------------
const tile = await buildMasterTile();

// 4K PNG (upscale the 1024 master to 3840 with a good kernel)
const png4k = await sharp(tile).resize(3840, 3840, { kernel: 'lanczos3' }).png().toBuffer();
fs.writeFileSync(path.join(__dirname, 'icon.png'), png4k);
console.log(`icon.png written (3840x3840, ${(png4k.length / 1e6).toFixed(1)} MB)`);

// ICO frames
const frames = [];
for (const S of ICO_SIZES) {
  const png = await sharp(tile).resize(S, S, { kernel: 'lanczos3' }).png().toBuffer();
  frames.push({ size: S, png });
}
fs.writeFileSync(path.join(__dirname, 'icon.ico'), buildICO(frames));
console.log(`icon.ico written (sizes ${ICO_SIZES.join('/')})`);
