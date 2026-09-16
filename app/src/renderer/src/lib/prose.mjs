/**
 * Turn a model reply into the blocks and spans the message renderer draws.
 *
 * Kept apart from the component because this is where the reply is actually
 * interpreted -- and where the bugs have been. It is plain data in, plain data
 * out, so the suite can exercise it without a DOM: the component below it only
 * maps the tokens onto elements.
 *
 * .mjs rather than .js because app/package.json declares no type, so Node reads
 * a .js file here as CommonJS and the tests could not import it.
 */

// The 8B model sometimes writes the literal words "blank line" (or "BLANK
// LINE") where it should have left an empty one. Strip the marker so it never
// renders as visible text, whether it sits on its own line or inline.
const BLANK_MARKER_LINE = /^[ \t]*\(?blank ?lines?\)?[ \t]*$/gim;
const BLANK_MARKER_WHOLE = /^\(?blank ?lines?\)?$/i;

// One-or-more blank lines separate paragraphs. Collapsed, so a model emitting
// \n\n\n does not open an oversized gap.
const PARAGRAPH_BREAK = /\n[ \t]*\n+/;

// Recognize straight ("..."), curly (“...”), AND mixed ("...” or “...") double
// quotes. The 8B model frequently opens with a straight quote and closes with a
// curly one, and a same-style-only pattern left that dialogue uncolored -- the
// bug from the original report. Match any opening quote char against any
// closing one.
//
// One instance for the module rather than one per line: tokenize runs on every
// line of every message on every render, and a /g regex carries no state but
// lastIndex, which the loop resets before it reads it.
const SPAN_RE = /(["“][^"“”]*["”]|\*[^*]+\*|_[^_]+_)/g;

const SOFT_LEN = 240;   // chars; a block below this is left alone
const PER_PARAGRAPH = 2; // sentences per forced paragraph

/** Reply text -> the paragraphs to render, in order. */
export function toBlocks(text) {
  return String(text || '')
    .replace(BLANK_MARKER_LINE, '')
    .split(PARAGRAPH_BREAK)
    .map((b) => b.trim())
    .filter((b) => b && !BLANK_MARKER_WHOLE.test(b))
    .flatMap(splitLongBlock);
}

/**
 * Force-split an over-long block at sentence boundaries, so the reply reads as
 * spaced paragraphs whatever the model emitted. A short block, or one that
 * already carries its own line breaks, is left alone -- there the formatting
 * was deliberate. Sentence ends only count OUTSIDE "quotes", *actions* and
 * _narration_, so a span is never torn in half.
 */
export function splitLongBlock(block) {
  if (block.length <= SOFT_LEN || block.includes('\n')) return [block];

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
    if (inQuote || inStar || inUnder) continue;
    if (ch !== '.' && ch !== '!' && ch !== '?') continue;
    // Consume any run of terminators and closing quotes, then require a space:
    // "Mr. Smith" and "3.5" are not sentence ends.
    let j = i;
    while (j + 1 < block.length && '.!?"’”'.includes(block[j + 1])) j++;
    if (j + 1 < block.length && block[j + 1] !== ' ') continue;
    sentences.push(block.slice(start, j + 1).trim());
    start = j + 1;
    i = j;
  }
  if (start < block.length) sentences.push(block.slice(start).trim());

  const clean = sentences.filter(Boolean);
  if (clean.length <= 1) return [block]; // nothing safe to split on

  const out = [];
  for (let i = 0; i < clean.length; i += PER_PARAGRAPH) {
    out.push(clean.slice(i, i + PER_PARAGRAPH).join(' '));
  }
  return out;
}

/**
 * One line -> [{ kind, text }], kind being 'text', 'dialogue' or 'action'.
 * An action's text is the inside of the span; the delimiters are styling, not
 * content, and the reader should not see them. A quote keeps its marks.
 */
export function tokenize(line) {
  const out = [];
  SPAN_RE.lastIndex = 0;
  let last = 0;
  let m;
  while ((m = SPAN_RE.exec(line)) !== null) {
    if (m.index > last) out.push({ kind: 'text', text: line.slice(last, m.index) });
    const tok = m[0];
    const quoted = tok.startsWith('"') || tok.startsWith('“');
    out.push(quoted
      ? { kind: 'dialogue', text: tok }
      : { kind: 'action', text: tok.slice(1, -1) });
    last = m.index + tok.length;
  }
  if (last < line.length) out.push({ kind: 'text', text: line.slice(last) });
  return out;
}
