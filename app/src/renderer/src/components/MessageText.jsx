import React from 'react';

// Render roleplay prose as character.ai-style spaced paragraphs:
//   - blank-line-separated blocks become individual <p> with even spacing
//     (extra blank lines are collapsed so gaps stay uniform)
//   - a single newline inside a block becomes a <br>
//   - "quoted dialogue" gets accent emphasis; *actions* / _narration_ get italic
//     dim styling. Keeps the cinematic feel without a full markdown engine.
export default function MessageText({ text }) {
  // Split into paragraphs on one-or-more blank lines; drop empty blocks so the
  // model emitting \n\n\n doesn't create oversized gaps.
  const blocks = String(text || '').split(/\n[ \t]*\n+/).map((b) => b.trim()).filter(Boolean);

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

// Tokenize a block on quotes and *...* / _..._ spans, preserving single newlines
// inside the block as <br>.
function renderBlock(block) {
  const out = [];
  let k = 0;
  block.split('\n').forEach((line, li) => {
    if (li > 0) out.push(<br key={`br-${k++}`} />);
    const re = /("[^"]*"|\*[^*]+\*|_[^_]+_)/g;
    let last = 0;
    let m;
    while ((m = re.exec(line)) !== null) {
      if (m.index > last) out.push(<span key={k++}>{line.slice(last, m.index)}</span>);
      const tok = m[0];
      if (tok.startsWith('"')) out.push(<span key={k++} className="rp-dialogue">{tok}</span>);
      else out.push(<em key={k++} className="rp-action">{tok.slice(1, -1)}</em>);
      last = m.index + tok.length;
    }
    if (last < line.length) out.push(<span key={k++}>{line.slice(last)}</span>);
  });
  return out;
}
