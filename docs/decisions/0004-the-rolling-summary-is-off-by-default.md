# 0004 — The rolling summary is off by default

**Status:** accepted
**Supersedes:** nothing — amends 0003, which stands for how the summariser runs
**Numbers:** `docs/MASTER.md`, "The 2,000-exchange A/B"

## Decision

`SUMMARY_ENABLED` defaults to `0`. Folding, embedding, archiving and retrieval
are unchanged and stay on: turns still fold out of the verbatim window, still
embed, and are still reachable through `retrieve`. What goes away is the
narrative of what fell out.

Everything 0003 decided still holds **when the summary is on** — its own
backend, `gemma3:4b`, CPU, background maintenance off the turn lock.

Three defects the measurement exposed are fixed, in both tracks:

1. **`clampSummary` / `clamp_summary`** replaces raw front-truncation. It lands
   on a paragraph break, else a sentence end, else a word break, else the raw
   cut.
2. **`stripTranscript` / `strip_transcript`** drops lines the summariser copied
   back verbatim, matched against the two labels `renderTurns` actually emits —
   not a general `^\w+:`, which would eat "Note:" and "Kestrel Station: a wreck".
   If what survives is under 35% of the length of the summary it would replace,
   the prior summary is kept instead (`summaryIsUsable`) -- what came back was a
   transcript, not a summary with some transcript in it.
3. **The prompt** now demands third-person prose, forbids speaker labels, and
   forbids inventing a name for anyone.

These fixes are **unmeasured**. That is exactly why they ship behind a default
of off: turning them on by default would repeat the mistake this experiment was
run to catch. `SUMMARY_ENABLED=1` re-runs the A/B against the fixed summariser.

## Why

2,000 exchanges per arm, one character, one script, identical models, folding
and retrieval on in both. 40 planted facts probed at +40, +180, +600 and +1400
turns, 240 explicit adult turns for compliance.

| | summary ON | summary OFF | |
|---|---|---|---|
| fact recall | 33.8% | 39.4% | p=0.30 — **no difference** |
| accuracy when it committed | 44.3% | **61.8%** | p=0.009 |
| commit rate | 76.3% | 63.7% | p=0.015 |
| answered with the WRONG established fact | 41 | **7** | p<0.0001 |
| invented a value | 27 | 32 | p=0.47 — no difference |
| facts wrong at least once | 34/40 | **18/40** | |
| prompt tokens p50 | 4,023 | **2,394** | |
| latency p50 / TTFT p50 | 2,282 / 589 ms | **1,625 / 332 ms** | |
| summariser CPU | **836 min** | 0 | 98.4% duty, 667 min of stall |
| refusals / character breaks on 240 adult turns | 0 / 0 | 0 / 0 | |

**The summary did not change how much was remembered. It changed how the model
failed.** Recall moved by 5.6 points in the direction of *off* and did not clear
noise at any distance. What moved decisively was the shape of the errors: given
forty facts co-resident in lossy prose, the model answered with a *different*
established fact six times as often — entity collapse, not forgetting. Inventing
a value outright, the failure a summary is supposed to prevent, was statistically
identical in both arms.

Retrieval alone carried the same recall on 40% fewer prompt tokens, at 71% of the
latency, for none of the CPU.

### Why recall was flat

Retrieval does the remembering. `RETRIEVE_K=4` over the embedded archive is
unaffected by `SUMMARY_ENABLED`, and a planted fact is a concrete noun phrase —
exactly what cosine retrieval is good at. The summary was a second, lossier copy
of material already reachable, and its cost was paid on every single turn.

### What the run exposed in the summariser itself

Both defects compound, because the stored summary is fed back as the next fold's
prior — written once, then re-read and re-compressed for the life of the
conversation:

- The ON arm ended holding a summary that **opened mid-word**, `"hreads"` — a
  decapitated "threads" — because `MAX_SUMMARY_CHARS` was enforced by
  `slice(len - 6000)`.
- Over 226 folds the summariser **drifted out of narrative into transcript**,
  copying its input back with `User:` / character-name labels. Stripping those
  lines from the real stored summary reclaimed 6,000 → 4,565 chars: a quarter of
  the budget spent replaying dialogue it was asked to compress.
- It **invented a name for the user** (`Teodor`, never given), wrote it into the
  summary, and thereafter fed it back on every turn as established fact.

The `Modelfile` gained the matching rules for the chat model: never invent a name
for the user, and admit a gap in character rather than guess. The run declined
4 times against 68 wrong answers — the model had no in-character way to say it
did not remember.

## Consequences

- A conversation that ran with the summary on keeps its stored summary; it is
  still read. Nothing migrates, and nothing needs to: the next fold under
  `SUMMARY_ENABLED=1` rewrites it wholesale through the fixed path.
- `foldStats()` gains `transcriptLines` and `summariesRejected`, so a re-run can
  falsify the two fixes without reading the stored summary by hand.
- `_foldCost.summaries` and `recordTurn`'s `summarized` now follow the actual
  write, not merely a non-skipped summariser call.
- The result is bounded by what was measured: one model, one character, one
  scripted conversation, probes that are concrete noun phrases. It says nothing
  about a summary as a *reading surface* for the user, which is a different job.
