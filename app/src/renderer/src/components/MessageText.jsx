import React from 'react';

// Render roleplay prose: "quoted dialogue" gets accent emphasis, *actions* and
// _narration_ get italic dim styling. Keeps the cinematic feel without a full
// markdown engine.
export default function MessageText({ text }) {
  const parts = [];
  let key = 0;

  // Tokenize on quotes and *...* / _..._ spans.
  const regex = /("[^"]*"|\*[^*]+\*|_[^_]+_)/g;
  let last = 0;
  let m;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) parts.push(<span key={key++}>{text.slice(last, m.index)}</span>);
    const tok = m[0];
    if (tok.startsWith('"')) {
      parts.push(<span key={key++} className="rp-dialogue">{tok}</span>);
    } else {
      parts.push(<em key={key++} className="rp-action">{tok.slice(1, -1)}</em>);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) parts.push(<span key={key++}>{text.slice(last)}</span>);

  // Preserve line breaks (scene headers etc.)
  return (
    <div className="msg-text">
      {text.split('\n').map((line, i) => {
        const lineParts = renderLine(line);
        return (
          <p key={i} className={line.startsWith('Scene:') ? 'rp-scene' : ''}>
            {lineParts.length ? lineParts : ' '}
          </p>
        );
      })}
    </div>
  );

  function renderLine(line) {
    const out = [];
    let k = 0;
    const re = /("[^"]*"|\*[^*]+\*|_[^_]+_)/g;
    let l = 0, mm;
    while ((mm = re.exec(line)) !== null) {
      if (mm.index > l) out.push(<span key={k++}>{line.slice(l, mm.index)}</span>);
      const t = mm[0];
      if (t.startsWith('"')) out.push(<span key={k++} className="rp-dialogue">{t}</span>);
      else out.push(<em key={k++} className="rp-action">{t.slice(1, -1)}</em>);
      l = mm.index + t.length;
    }
    if (l < line.length) out.push(<span key={k++}>{line.slice(l)}</span>);
    return out;
  }
}
