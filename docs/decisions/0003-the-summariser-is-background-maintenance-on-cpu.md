# 0003 — The summariser is background maintenance, on CPU

**Status:** accepted
**Supersedes:** nothing
**Numbers:** `docs/MASTER.md`

## Decision

Three things, which are really one thing:

1. **The summariser picks its own backend.** `SUMMARIZER_BACKEND` (default
   `ollama`) is separate from `INFERENCE_BACKEND`. `/health` reports
   `summarize.honoursModel`.
2. **The summariser runs on CPU.** `SUMMARIZER_NUM_GPU=0`, the same call
   `EMBED_NUM_GPU=0` already makes, for the same reason.
3. **Folding is background maintenance, not part of a turn.** `recordTurn` does
   the durable write and returns; `runMaintenance` summarises, embeds and
   archives off the per-conversation turn lock.

`SUMMARIZER_MODEL` stays `gemma3:4b`. It does not become the chat model.

## Why

### The routing bug made the other two invisible

`inference/index.js` sent `generate()` to the **chat** backend. llama-server
accepts a `model` argument and ignores it — one server, one model — so with
`INFERENCE_BACKEND=llama-server` set, every summary was written by `vessel`
while `/health` reported `gemma3:4b`.

That is not a cosmetic mismatch. Scored on 13 facts the transcript actually
contains, over a two-round rolling update:

| summariser | facts kept | invented entities |
|---|---|---|
| `gemma3:4b` | 12–13 / 13 | 0 |
| `vessel` | 8 / 13 | names characters that do not exist |

**Confabulation, not recall, is the metric.** A rolling summary is read back as
canon on every later turn, so an invented character does not degrade gracefully
— it becomes permanent story fact. `vessel` is tuned to write surprising prose;
asked to condense, it keeps writing.

So `SUMMARIZER_MODEL=vessel` is rejected on quality, and the summariser needs a
backend it can actually choose.

### The GPU has no room for a second model

llama-server holds 6,163 MiB of an 8,188 MiB card. Every placement was measured:

| placement | VRAM | chat decode | verdict |
|---|---|---|---|
| GPU, co-resident | 3.0 GB | 47.0 → **20.8 tok/s** | 2.3 GB of chat model paged to system RAM |
| GPU, `keep_alive: 0` | transient | — | 8–32 s reload **per summary** |
| CPU | **0 MiB** | **unchanged** | chosen |

No llama-server configuration makes both fit: 4.6 GB of weights plus a 1.5 GB KV
pool, against 3.0 GB for the summariser and an 8.19 GB card. `-c 8192` and q8_0
KV together save 1.25 GB and still leave 5.4 + 3.0 > 8.19.

Not being evictable is not the same as being free. llama-server solved
*eviction*; it did not solve *contention*.

### CPU is only affordable if nobody waits for it

CPU costs 95–239 s per summary against ~19 s on the GPU. That is unacceptable on
a read path and irrelevant off one — and the code had it on the read path:
`server.js` holds `memory.acquireLock(convId)` across the whole `/chat` handler,
including `recordTurn`'s `summarize()`. Measured, the summarising turn took
**31,839 ms and 35,619 ms** against 2.3–6.1 s for every other turn.

So the fold moved off the lock. `recordTurn` now does only the durable write —
insert the turn, register the variant, touch the conversation — and schedules
`runMaintenance`, which summarises, embeds and archives in the background under
its own per-conversation coalescing.

**What that trades away:** while a fold is in flight, readers see the previous
summary and a longer-than-designed verbatim window. Both are safe. The summary
is *behind*, not wrong, and the extra verbatim turns are exactly the material
the summary is missing — the model sees it either way, in full rather than
condensed. `VERBATIM_CEILING` bounds how much of that backlog reaches the
prompt so a slow fold cannot grow the context window without limit.

Verified with the summariser held for 8 s per call, `SUMMARIZE_THRESHOLD=6`,
`VERBATIM_TURNS=4`:

- slowest turn **44 ms** against the 8,000 ms summariser — no turn waits
- nine turns land during one fold; the prompt still carries exactly 8 verbatim
  rows while the database holds 18
- six schedule calls collapse to **two** folds; 18 rows, no duplicates,
  settling at `VERBATIM_TURNS` and stopping
- deleting a conversation mid-fold leaves it deleted

### The summary also had no length bound

Separate defect, found on the way. `MAX_SUMMARY_CHARS` was enforced by
**front-truncation** — `updated.slice(updated.length - 6000)` — and nothing told
the model about the cap, so gemma answered with 5,792–12,422 chars. The overflow
was cut off the **front**: the oldest, most-condensed material, the part that can
no longer be recovered from the verbatim window.

`SUMMARY_TARGET_CHARS` now steers the prompt, defaulting to
`MAX_SUMMARY_CHARS * 0.4`. Measured, the bound costs one fact and halves
generation time; steering below ~1,800 makes the model drop material instead of
compressing it, scoring the same 8/13 as `vessel`. Front-truncation stays as the
backstop it was always meant to be.

## Consequences

- A summary can be up to `SHUTDOWN_MAINTENANCE_MS` (15 s) from landing when the
  app closes. Nothing is lost: the turns are still verbatim and the next launch
  folds them.
- A crash mid-fold leaves the turns verbatim and re-folds later. The archive
  insert and the turn delete are still two statements, as they were before.
- `recordTurn` no longer reports `archived` / `summarized` truthfully — it
  returns `maintenanceScheduled`. Nothing consumed the old fields.
- `--parallel 1` is now required at llama-server launch. Unrelated to the
  summariser, found in the same pass: `--parallel` *divides* `-c`, so the shipped
  default of 4 was truncating every conversation to a quarter of its window.

## Amended by stage 4b

`INFERENCE_BACKEND=llama-local` (decision 0001, stage 4b — the in-process
llama.cpp engine, Rust track only, behind a build feature) is a deliberate
exception to points 1 and 2. It holds exactly one model, so it chats and
summarises off the same resident weights: `SUMMARIZER_MODEL` is ignored and the
summary runs wherever the chat model is, GPU included.

That is not a regression of this decision, it is the thing this decision was
pricing. The CPU rule exists because a second model on an 8 GB card evicts the
first; with one model there is nothing to evict. Everywhere else — ollama,
llama-server — points 1 and 2 stand unchanged, and point 3 stands everywhere.

Also amended by 0004: all of the above applies only when `SUMMARY_ENABLED=1`,
which is no longer the default.
