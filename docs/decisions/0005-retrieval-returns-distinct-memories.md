# 0005 — Retrieval returns distinct memories, not k copies of one

**Status:** accepted
**Supersedes:** nothing — amends 0004, which stands for what retrieval is *for*
**Numbers:** `docs/MASTER.md`, "The 20,000-exchange run"

## Decision

Retrieval's top-k is a set of *memories*, not a set of rows. Two archived turns
whose vectors sit at or above `RETRIEVE_DUP_MAX` cosine of one another are one
memory and share one slot: the better-scoring copy takes it, the other is
dropped rather than given a slot of its own.

`RETRIEVE_DUP_MAX` defaults to `0.97`. It is deliberately high. The failure
being corrected is duplicates, not neighbours, and a lower bar starts discarding
genuinely different turns that happen to share a subject — which is most of a
long conversation about one thing.

The comparison runs on winners, not on rows: a candidate is only compared
against what is already held, which it reaches only by clearing the score
cutoff. That is at most k row-against-row dot products a handful of times per
query, against one query-against-row product for every row in the archive.
Measured cost is nil — 79–85 ms per probe with the rule and without it, because
the embed round-trip dominates both.

In both tracks: `src/backend/memory.js` and `src-core/src/memory.rs`.

## Why

The 20,000-exchange run's recall fell from 52.5% at +40 turns to 0.0% at +4000,
which reads as a limit on how far back memory reaches. It was not.

The probes are asked once per distance, so the fifth ask met four earlier copies
of the same question already in the archive, every one of them closer to the
query than the answer was. Retrieval was returning the user's own question, four
times over, at 0.94 each.

| probe distance | recall | unique rows in the top-4 |
|---|---|---|
| +40 | 52.5% | 2.95 |
| +180 | 55.0% | 3.20 |
| +600 | 32.5% | 2.83 |
| +1400 | 27.5% | 2.00 |
| +4000 | **0.0%** | **1.00** |

Distance was a confound. The defect is in the second column, and it is present
in every row of it.

**The general shape:** a bounded top-k with no diversity rule spends its whole
budget on one thing as soon as that thing is in the corpus more than k times.
Nothing about this is specific to a probe question — a greeting, a line the
character returns to, a question the user asks again every few hundred turns all
do it. A long conversation repeating itself is not an edge case; it is what a
long conversation is.

Replaying all 40 probes through the real `retrieve()` against a snapshot of the
run's own database, so both arms see identical rows:

| | before | after |
|---|---|---|
| recall | 0/40 | **35/40 (87.5%)** |
| unique rows in the top-4 | 1.00 | 4.00 |
| per-probe latency | 79–85 ms | 79–85 ms |

### Why 0.97 and not a rounder number

Swept on that same snapshot rather than picked:

| `RETRIEVE_DUP_MAX` | recall |
|---|---|
| 0.85 | 82.5% — over-suppression starting |
| 0.90 | 87.5% |
| 0.95 | 87.5% |
| **0.97** | **87.5%** |
| 0.99 | 87.5% |
| 1.0 | 20% — float equality fails on bit-identical rows |

0.97 sits in the middle of the plateau, clear of the edge where distinct
memories start being merged and of the edge where the comparison stops working
at all.

### What this does not do

It does not deduplicate the archive. Both rows stay stored, both stay
searchable, and a later query that matches one and not the other still finds it.
The rule is about what a single answer is built from.

It has no "off". `RETRIEVE_DUP_MAX=1.0` is the loosest setting the clamp allows,
and at 1.0 bit-identical rows still merge — correctly. A cosine above 1 is not a
cosine.

### Limits

One run, one script, one embedding model. The probes are concrete noun phrases,
which favour retrieval, and they are asked in a pattern that guarantees the
duplicates — that is why the defect showed up here and not in the 2,000-exchange
A/B, where each fact is probed at fewer distances. The threshold is tuned
against `nomic-embed-text`; a different embedder puts "almost the same" at a
different cosine and would want its own sweep.
