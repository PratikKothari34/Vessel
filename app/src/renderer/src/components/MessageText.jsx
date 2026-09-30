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
//
// Memoized on `text`, and that is not a micro-optimization. Streaming calls
// setStreamText on EVERY token, so Chat re-renders per token and re-runs its
// whole `messages.map`. Without this, each token re-parses every settled message
// in the conversation through toBlocks + tokenize - O(tokens x history) parses
// for one reply, on the thread that also has to paint. The settled messages
// cannot have changed, because a new token only ever appends to the streaming
// bubble, which is a separate element with its own changing `text`.
//
// The default shallow compare is exactly right here: the only prop is a string,
// so it is compared by value, not identity.
function MessageText({ text }) {
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

export default React.memo(MessageText);

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
