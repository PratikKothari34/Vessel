import React from 'react';

// Render roleplay prose as character.ai-style spaced paragraphs:
//   - blank-line-separated blocks become individual <p> with even spacing
//     (extra blank lines are collapsed so gaps stay uniform)
//   - a single newline inside a block becomes a <br>
//   - "quoted dialogue" gets accent emphasis; *actions* / _narration_ get italic
//     dim styling. Keeps the cinematic feel without a full markdown engine.
export default function MessageText({ text }) {
  // Hard guarantee: the 8B model sometimes writes the literal words "blank line"
  // (or "BLANK LINE") where it should have left an empty line. Strip any such
  // marker so it never renders as visible text. Match it whether it sits on its
  // own line or is wedged inline.
  const cleaned = String(text || '').replace(/^[ \t]*\(?blank ?lines?\)?[ \t]*$/gim, '');

  // Split into paragraphs on one-or-more blank lines; drop empty blocks so the
  // model emitting \n\n\n doesn't create oversized gaps.
  const rawBlocks = cleaned
    .split(/\n[ \t]*\n+/)
    .map((b) => b.trim())
    .filter((b) => b && !/^\(?blank ?lines?\)?$/i.test(b));

  // Hard guarantee: the 8B model sometimes packs several sentences into one
  // block with no blank line. Force-split any over-long block at sentence
  // boundaries so the reply always reads as spaced paragraphs, regardless of
  // what the model emitted.
  const blocks = rawBlocks.flatMap(splitLongBlock);

  return (
    <div className="msg-text">
      {blocks.map((block, bi) => (
        <p key={bi} className={block.startsWith('Scene:') ? 'rp-scene' : ''}>
          {renderBlock(block)}
        </p>
      ))}
    </div>
  );
}

// Force-split an over-long block into shorter paragraphs at sentence boundaries.
// A block is left untouched if it's short or already contains its own line
// breaks (the model deliberately formatted it). Sentence ends are only honored
// when they fall OUTSIDE "quotes" and *action* spans, so dialogue/actions never
// get torn apart. Sentences are then grouped 2-at-a-time into paragraphs.
function splitLongBlock(block) {
  const SOFT_LEN = 240; // chars; below this, leave it alone
  if (block.length <= SOFT_LEN || block.includes('\n')) return [block];

  // Walk the block, tracking quote/asterisk/underscore depth, and cut after a
  // sentence terminator only when we're not inside a span.
  const sentences = [];
  let start = 0;
  let inQuote = false;
  let inStar = false;
  let inUnder = false;
  for (let i = 0; i < block.length; i++) {
    const ch = block[i];
    if (ch === '"' || ch === '“' || ch === '”') inQuote = !inQuote;
    else if (ch === '*' && !inQuote) inStar = !inStar;
    else if (ch === '_' && !inQuote) inUnder = !inUnder;
    const open = inQuote || inStar || inUnder;
    if (!open && (ch === '.' || ch === '!' || ch === '?')) {
      // consume any run of terminators/closing quote, require a following space
      let j = i;
      while (j + 1 < block.length && '.!?"’”'.includes(block[j + 1])) j++;
      if (j + 1 >= block.length || block[j + 1] === ' ') {
        sentences.push(block.slice(start, j + 1).trim());
        start = j + 1;
        i = j;
      }
    }
  }
  if (start < block.length) sentences.push(block.slice(start).trim());
  const clean = sentences.filter(Boolean);
  if (clean.length <= 1) return [block]; // nothing safe to split on

  // Group two sentences per paragraph so beats stay tight but not choppy.
  const out = [];
  for (let i = 0; i < clean.length; i += 2) {
    out.push(clean.slice(i, i + 2).join(' '));
  }
  return out;
}

// Tokenize a block on quotes and *...* / _..._ spans, preserving single newlines
// inside the block as <br>.
function renderBlock(block) {
  const out = [];
  let k = 0;
  block.split('\n').forEach((line, li) => {
    if (li > 0) out.push(<br key={`br-${k++}`} />);
    // Recognize straight ("..."), curly (“...”), AND mixed ("...” or “...")
    // double quotes. The 8B model frequently opens with a straight quote and
    // closes with a curly one (or vice versa); a same-style-only regex would
    // leave that dialogue uncolored, which is the exact bug from the original
    // report. Match any opening quote char paired with any closing quote char.
    const re = /(["“][^"“”]*["”]|\*[^*]+\*|_[^_]+_)/g;
    let last = 0;
    let m;
    while ((m = re.exec(line)) !== null) {
      if (m.index > last) out.push(<span key={k++}>{line.slice(last, m.index)}</span>);
      const tok = m[0];
      if (tok.startsWith('"') || tok.startsWith('“'))
        out.push(<span key={k++} className="rp-dialogue">{tok}</span>);
      else out.push(<em key={k++} className="rp-action">{tok.slice(1, -1)}</em>);
      last = m.index + tok.length;
    }
    if (last < line.length) out.push(<span key={k++}>{line.slice(last)}</span>);
  });
  return out;
}
