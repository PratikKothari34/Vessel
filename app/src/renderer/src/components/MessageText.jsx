import React from 'react';
import { toBlocks, tokenize } from '../lib/prose.mjs';

// Render roleplay prose as character.ai-style spaced paragraphs:
//   - blank-line-separated blocks become individual <p> with even spacing
//   - a single newline inside a block becomes a <br>
//   - "quoted dialogue" gets accent emphasis; *actions* / _narration_ get italic
//     dim styling. Keeps the cinematic feel without a full markdown engine.
//
// Everything that decides WHAT the spans are lives in lib/prose.mjs, which the
// suite can exercise without a DOM. This file only turns tokens into elements.
export default function MessageText({ text }) {
  return (
    <div className="msg-text">
      {toBlocks(text).map((block, bi) => (
        <p key={bi} className={block.startsWith('Scene:') ? 'rp-scene' : ''}>
          {renderBlock(block)}
        </p>
      ))}
    </div>
  );
}

// A block's lines, with the single newlines between them kept as <br>.
function renderBlock(block) {
  const out = [];
  let k = 0;
  block.split('\n').forEach((line, li) => {
    if (li > 0) out.push(<br key={`br-${k++}`} />);
    for (const tok of tokenize(line)) {
      if (tok.kind === 'dialogue') out.push(<span key={k++} className="rp-dialogue">{tok.text}</span>);
      else if (tok.kind === 'action') out.push(<em key={k++} className="rp-action">{tok.text}</em>);
      else out.push(<span key={k++}>{tok.text}</span>);
    }
  });
  return out;
}
