# MASTER

Consolidated numbers for Vessel. **Not an authority** — loses to `CLAUDE.md`
and `docs/decisions/`, wins on numbers.

Every figure is tagged **[M]** measured or **[E]** estimated. An [E] is a claim
waiting to be checked — replace it the moment a measurement exists.

## Current footprint

| Item | Size | |
|---|---|---|
| `win-unpacked` total | 300 MB | [M] |
| `Vessel.exe` (Chromium + Node) | 173 MB | [M] |
| `locales/` | 39 MB | [M] |
| `resources/` (backend + addons) | 41 MB | [M] |
| GL / swiftshader / d3dcompiler / vulkan | 20 MB | [M] |
| NSIS installer | 87 MB | [M] |

Electron doubles as the Node runtime — `app/src/main/index.js:60` spawns the
backend via `process.execPath` with `ELECTRON_RUN_AS_NODE=1`. No separate Node
is shipped. Three N-API addons in `resources/`: `@tursodatabase/database`,
`@tursodatabase/sync`, `keytar`.

## Models

| Model | Role | Size | |
|---|---|---|---|
| chat (Llama-3.1-8B Q4_K_M) | `vessel` | 4.92 GB | [M] |
| `gemma3:4b` | summariser | 3.34 GB | [M] |
| `nomic-embed-text` | embeddings, 768-dim | 0.27 GB | [M] |
| **total** | | **8.53 GB** | [M] |

Models are ~96% of on-disk footprint. The entire Electron shell is ~3.5%.

## KV cache

Llama-3.1-8B: 32 layers, 8 KV heads (GQA), 128 head dim.

```
per token = 2 x 32 x 8 x 128 x 2 bytes (f16) = 131,072 B = 128 KB
```

| `num_ctx` | dtype | KV | resident total | |
|---|---|---|---|---|
| 32768 | f16 | 4.00 GiB | **9.52 GB** | [M] |
| 12288 | f16 | 1.50 GiB | 6.35 GB | [M] |
| 32768 | q8_0 | 2.00 GiB | 7.56 GB | [M] |
| 12288 | q8_0 | 0.75 GiB | **5.60 GB** | [M] |

KV is allocated eagerly at `num_ctx`. The estimate was `4.9 weights + 4.0 KV +
0.5 compute = 9.4 GB`; `/api/ps` reports **9.52 GB**, and every other row lands
within 0.2 GB of the same arithmetic. Against an 8 GB RTX 4060 that spills.

## Stage 1 result — `num_ctx` is the lever, not KV dtype

All four arms, same 3,206-token prompt, each started from 7,956 MiB free:

| config | VRAM | CPU | GPU% | prefill t/s | decode t/s | |
|---|---|---|---|---|---|---|
| f16 / 32768 (Modelfile default) | 6.25 GB | 3.27 GB | 65.6% | 1,145 | **13.69** | [M] |
| f16 / 12288 | 6.35 GB | 0 | **100%** | 1,638 | **39.37** | [M] |
| q8_0 / 32768 | 6.37 GB | 1.19 GB | 84.3% | 1,410 | 23.02 | [M] |
| q8_0 / 12288 | 5.60 GB | 0 | **100%** | 1,581 | 37.30 | [M] |

**2.88x on decode from one value.** f16 / 12288 repeated at 37.71 / 39.42 / 39.37
across three runs.

`OLLAMA_KV_CACHE_TYPE=q8_0` is **not** adopted: once the model fits, quantised KV
measured *slower* than f16 (37.30 vs 39.37) because the dequant is pure overhead.
Its 0.75 GB remains the cheapest way to buy headroom later — a
speculative-decoding draft model — not a win on its own.

Measuring this needs a clean GPU. `ollama.exe` spawns `llama-server.exe` workers
that **outlive it**, and Ollama picks its layer offload from free VRAM at load
time — so a stale worker silently makes the next arm look worse. The first run of
this matrix showed GPU% *falling* as the model shrank, which is that bug, not a
result.

## Model eviction — the hidden cost of a second model

Ollama keys a resident instance on **(model, num_ctx)**, not on the model alone.
Anything that differs on either axis evicts the chat model, and the next user
message pays a full reload before its first token:

| summariser | chat reload after a summary | |
|---|---|---|
| `gemma3:4b` (current) | **19,365 ms** | [M] |
| `vessel`, `num_ctx` 8192 vs chat 12288 | 18,597 ms | [M] |
| `vessel`, `num_ctx` matched | **3 ms** | [M] |

The mismatched-window row is the trap: switching `SUMMARIZER_MODEL` to the chat
model buys nothing on its own. Both axes have to match. `memory.js` therefore
defaults `SUMMARIZER_NUM_CTX` to `OLLAMA_NUM_CTX` whenever the summariser *is*
the chat model, and `server.js` warns at startup if an explicit value breaks it.

This whole section is an **Ollama** problem. llama-server holds its one model in
an allocation it owns for the life of the process, so a second prompt cannot
evict the first — see *Stage 3 result*. Under `INFERENCE_BACKEND=llama-server`
the 19.4 s is 0 ms. (The KV pool is *not* unified and the extra slots are not
free context — see *Launch it with `--parallel 1`*.)

### Eviction is gone; contention replaced it

Not being evictable is not the same as being free. llama-server holds 6,163 MiB
of an 8,188 MiB card, so loading `gemma3:4b` (3.0 GB, confirmed `100% GPU` in
`ollama ps`) puts the card at 7,874 MiB — 96% full — and the Windows driver
pages the chat model's working set out to system RAM to make room:

| | value | |
|---|---|---|
| llama-server resident, alone | 6,163 MiB | [M] |
| both models resident | 7,874 MiB of 8,188 | [M] |
| llama-server resident, **during** the summary | **3,867 MiB** — 2.3 GB paged out | [M] |
| chat decode, summariser absent | 47.0 tok/s | [M] |
| chat decode, summariser resident | **20.8 tok/s** (2.26x) | [M] |
| chat decode, co-resident, separate run | 15.4 -> 38.0 tok/s (2.47x) | [M] |
| gemma load into the remaining VRAM | **8,319 ms** warm, **32,163 ms** cold | [M] |

The load figure is the one that kills `keep_alive: 0`. Unloading the summariser
after every summary sounds like the fix the backend swap unlocks — it cannot be
evicted, so the squeeze should last only as long as the call — but reloading a
3 GB model into 2 GB of free VRAM costs 8–32 s **per summary**, against 3.9 s on
an empty card. That is worse than the 19.4 s it was meant to avoid, and
`memory.js` holds the per-conversation lock across `summarize()`, so the user's
next message waits behind all of it.

No llama-server configuration makes both fit. The 8B chat weights are ~4.6 GB
and a 12,288-token unified KV pool is ~1.5 GB; dropping to `-c 8192` saves
0.5 GB and q8_0 KV saves another 0.75 GB, which still leaves 5.4 + 3.0 GB against
an 8.19 GB card. **The summariser cannot live on this GPU.** It runs on CPU —
the same decision `EMBED_NUM_GPU=0` already makes for the embedder.

### `SUMMARIZER_MODEL` was silently ignored

`inference/index.js` routed `generate()` to the **chat** backend. llama-server
accepts a `model` argument and ignores it — one server, one model — so in the
shipped Stage 3 config every summary was written by `vessel`, not `gemma3:4b`,
while `/health` reported gemma. The summariser now picks its own backend
(`SUMMARIZER_BACKEND`, default `ollama`), and `/health` reports
`summarize.honoursModel` so the failure cannot go quiet again.

That matters because the substitution is not neutral. Scored on 13 facts the
transcript actually contains, over a two-round rolling update [M]:

| summariser | facts kept | invented entities | |
|---|---|---|---|
| `gemma3:4b` | **12–13 / 13** | **0** | [M] |
| `vessel` (what was actually running) | 8 / 13 | names characters that do not exist | [M] |

**Confabulation, not recall, is the metric that matters.** A rolling summary is
read back as canon on every subsequent turn, so an invented character does not
degrade gracefully — it becomes permanent story fact. `vessel` is tuned to write
surprising prose; asked to condense, it keeps writing. `SUMMARIZER_MODEL=vessel`
is rejected on quality, not on cost.

### The summary had no length bound

`memory.js` capped the stored summary at `MAX_SUMMARY_CHARS` (6,000) by
**front-truncating**: `updated.slice(updated.length - 6000)`. Nothing told the
model about the cap, so gemma answered with 5,792–12,422 chars and the overflow
was cut off the **front** — amputating the oldest, most-condensed material, the
part that can no longer be recovered from the verbatim window. The model was
paying to generate tokens that were then thrown away, and throwing away the ones
that mattered most.

Sweeping an explicit bound in the prompt, gemma on CPU, round 2 scored /13 [M]:

| bound | time | chars | facts | truncated |
|---|---|---|---|---|
| none | 238,937 ms | 12,422 | 13/13 | **−6,422 off the front** |
| 2,500 | 95,270 ms | 5,160 | 12/13 | no |
| 2,500 (repeat) | 136,255 ms | 5,125 | 12/13 | no |
| 2,500 (repeat) | 172,312 ms | 8,709 | 12/13 | −2,709 |
| 1,800 | 123,390 ms | 6,418 | **8/13** | −418 |

The bound is a steer, not a limit — gemma overshoots it by 2–3.5x — but steering
at 2,500 lands inside the 6,000 cap most of the time, costs 1 fact, and halves
generation time. Steering at 1,800 makes the model drop material instead of
compressing it: 8/13 is the same score `vessel` gets. `SUMMARY_TARGET_CHARS`
defaults to `MAX_SUMMARY_CHARS * 0.4` (2,400) for exactly that reason, and the
front-truncation stays as the backstop it was always meant to be.

### Where the summariser runs: CPU

Four placements were measured. Only one leaves the chat model alone:

| placement | VRAM cost | chat decode | quality | verdict |
|---|---|---|---|---|
| GPU, co-resident | 3.0 GB | 47.0 → **20.8 tok/s** | 12–13/13 | rejected — 2.26x slowdown |
| GPU, `keep_alive: 0` | transient | n/a | 12–13/13 | rejected — 8–32 s reload *per summary* |
| `SUMMARIZER_MODEL=vessel` | 0 (shared) | unchanged | **8/13, confabulates** | rejected on quality |
| **CPU (`SUMMARIZER_NUM_GPU=0`)** | **0 MiB** | **unchanged** | **12–13/13, 0 invented** | **chosen** [M] |

CPU costs 95–239 s per summary against ~19 s on the GPU. That is the trade the
app can actually afford: summarisation is a background maintenance job, not a
read path, and VRAM was verified flat at 6,171–6,173 MiB across a full six-turn
run with gemma showing `3.0 GB / 100% CPU` in `ollama ps` [M]. The same decision
`EMBED_NUM_GPU=0` already makes for the embedder, for the same reason.

### Folding moved off the turn lock

CPU is only affordable if nobody waits for it, and the code made them wait.
`server.js` holds `memory.acquireLock(convId)` across the whole `/chat` handler
including `recordTurn`'s `summarize()`, so the summarising turn cost **31,839 ms
and 35,619 ms** against 2.3–6.1 s for every other turn [M].

`recordTurn` now does only the durable write — insert the turn, register the
variant, touch the conversation — and hands the fold to `runMaintenance`, which
summarises, embeds and archives in the background. Coalesced per conversation: a
request arriving mid-fold sets a rerun flag rather than starting a second run.

Verified with the summariser held 8,000 ms per call, `SUMMARIZE_THRESHOLD=6`,
`VERBATIM_TURNS=4` [M]:

| | result |
|---|---|
| slowest turn, against an 8,000 ms summariser | **44 ms** (every turn 16–44 ms) |
| verbatim rows in the DB during the fold | 18 |
| verbatim rows in the **prompt** during the fold | **8** (`VERBATIM_CEILING`) |
| 6 schedule calls → folds actually run | **2**, 18 rows total, no duplicates |
| settles at | `VERBATIM_TURNS` = 4, then stops |
| delete mid-fold | stays deleted (404) |

The cost is stale reads while a fold is in flight: the previous summary, and a
verbatim window longer than the design. Both are safe — the summary is behind,
not wrong, and the extra verbatim turns are exactly the material it is missing,
so the model sees that content either way, in full rather than condensed.
`VERBATIM_CEILING` (default `SUMMARIZE_THRESHOLD + 2`, the old synchronous
design's own high-water mark) bounds how much of the backlog reaches the prompt.

## Live window ceiling

The memory architecture caps the window by construction:

| Component | Source | Tokens | |
|---|---|---|---|
| Modelfile SYSTEM | `Modelfile`, 2,211 chars | 466 | [M] |
| character persona | `server.js` `buildPersonaMessage` | ~250 | [E] |
| rolling summary | `MAX_SUMMARY_CHARS=6000` | ~1,500 | [E] |
| retrieved | `RETRIEVE_K=4` x 512 | ~2,048 | [E] |
| verbatim | `SUMMARIZE_THRESHOLD=12` | ~4,200 | [E] |
| new message + `num_predict` | `server.js` = 512 | ~660 | [E] |
| **worst case** | | **~9,700** | [E] |

The 466-token `SYSTEM` row is new. Ollama had been dropping it whenever the
client sent its own system message, which the app always does, so it was never
in the prompt — decision `0002` and the measurement behind it. It now sits at
the head of the cacheable prefix, prefilled once per conversation
(`cache_n` p50 0.985), so the recurring cost is zero. Every prompt-token figure
measured below this line predates the fix; add ~466 to the prefix, not to the
per-turn prefill.

Measured, driving 24 turns with every block at production size (5,799-char
summary, 4 recalled turns, 10 verbatim turns — 18,708 prompt chars):

| | Tokens | |
|---|---|---|
| p50 | 4,677 | [M] |
| p95 | 5,986 | [M] |
| max | 5,988 | [M] |
| chars per token | 4 | [M] |

The estimate was conservative — it assumed ~4,200 verbatim tokens where the real
window holds ~2,260. **`OLLAMA_NUM_CTX=12288`** is ~2x the measured max and ~1.3x
the worst-case estimate, and it is sent per request (`server.js`), so changing it
never means rebuilding the Ollama model.

## Optimization ledger

Ranked by bytes saved.

| Lever | Saves | Cost |
|---|---|---|
| `num_ctx` 32768 -> 12288 | 3.17 GB VRAM, **2.88x decode** [M] | **done** — `OLLAMA_NUM_CTX`, per request |
| Ollama -> llama-server | 0.37 GB VRAM, **+18% decode**, +38% prefill, **19.4 s eviction -> 0** [M] | **done** — `INFERENCE_BACKEND`, both backends pass the same suite |
| `SUMMARIZER_MODEL=vessel` (drop gemma3:4b) | 3.34 GB disk + **19.4 s per summary** [M] | **rejected on quality** — 8/13 facts and invents named characters; see *Summariser* [M] |
| KV `q8_0` | 0.75 GiB VRAM, **-5% decode** [M] | rejected for now; see Stage 1 result |
| summariser on CPU (`SUMMARIZER_NUM_GPU=0`) | **3.0 GB VRAM**, chat decode 20.8 -> **47.0 tok/s** [M] | **done** — costs 95–239 s per summary, which nothing waits on |
| fold off the turn lock (background maintenance) | summarising turn **35.6 s -> 44 ms** [M] | **done** — readers see a stale summary until the fold lands; bounded by `VERBATIM_CEILING` |
| llama-server `--parallel 1` | **`n_ctx_slot` 3,072 -> 12,288** at identical VRAM and decode [M] | **done** — none; the other three slots were never used |
| summary length steer (`SUMMARY_TARGET_CHARS`) | up to **6,422 chars** no longer generated then front-truncated away [M] | **done** — 1 fact of 13, and gemma overshoots the steer 2–3.5x |
| Q4_K_M -> IQ4_XS | ~0.47 GB disk + VRAM | negligible quality |
| 384-dim embedder (bge-small / MiniLM) | ~0.18 GB disk, 2x faster cosine | slight recall loss |
| int8 embeddings in DB | 3072 -> 774 B/turn (3.97x) [M] | **done** — max cosine error 1.2e-3, ranking identical |
| Electron -> Tauri | 0.19 GB disk, ~150 MB RAM [E] | **code-complete (stage 4a)** — 239 lines of main+preload became a 524-line shell; unverified at runtime, see *Stage 4a* |
| Node backend -> in-process Rust | 0.04 GB disk, ~70 MB RAM [E] | **code-complete (stage 4a)** — 3,996 lines became 5,690, no loopback HTTP left |

**Banked so far:** VRAM 9.52 GB spilling -> **6.03 GB fully resident** [M].
Decode 13.69 -> **46.4 tok/s p50** end to end on the real GPU [M] — **3.39x**,
from two levers: `num_ctx` 32768 -> 12288, then Ollama -> llama-server. Prefill
1,145 -> ~2,200 tok/s. The summariser's 19.4 s eviction is 0, its VRAM is 0, and
the turn that triggers it no longer pays for it: **35.6 s -> 44 ms** [M].
Usable context per conversation is 4x what shipped (`--parallel 1`). Disk is
unchanged at 8.53 GB — every disk lever left is a model swap, and each one is a
quality call.

**Endpoint:** disk 8.5 GB -> ~4.6 GB. VRAM -> ~5.2 GB (resident, ~2.8 GB
headroom for a speculative-decoding draft model).

The shell is **3% of the disk win and 5% of the RAM win.** It goes last.

## Known hot paths

| Issue | Where | Status |
|---|---|---|
| Retrieval block sits mid-prefix, changes every turn | `memory.js` `buildContext` | **fixed** — moved to the tail, after the verbatim window |
| `embed()` blocks before the chat call, on CPU | `memory.js`, `EMBED_NUM_GPU=0` | **fixed** — the archive is checked first, so a conversation with nothing archived never pays the embed; the rest runs under `summarize()` |
| Full archive scan + Float32 decode per message | `memory.js` `retrieve` | **fixed** — vectors cached per conversation, normalised once, scanned with a dot product; the cache rebuilds when a sync pull lands rows below the high-water id, which a `max(id)` check alone misses |
| Archive loop holds the per-conversation lock | `memory.js` `recordTurn` | **fixed** — embeds run concurrently under the summariser call; writes batched to 2 statements |
| `prompt_eval_count` parsed then discarded | `server.js` | **fixed** — `metrics.js`, exposed at `GET /metrics` |
| Second model evicts the chat model | `memory.js` `SUMMARIZER_MODEL` | open **on Ollama** — 19.4 s reload per summary; closing it is a quality call. **Gone on llama-server** — measured 1,889 ms with `loadMs: 0` |
| Modelfile `SYSTEM` never reached the model | `server.js` `buildPersonaMessage` | **fixed** — Ollama drops a Modelfile SYSTEM whenever the client sends one; the app always does. Read from the `Modelfile` at runtime and prepended to the persona message, so both backends send it. Decision `0002` |
| Delete races the post-stream record | `server.js` `DELETE /conversations/:id` | **fixed** — the delete aborts any live stream for that conversation, then takes the same lock; 5/5 FOREIGN KEY failures before, 0/5 after |

## Stage 2 result — prompt order

Same 24 turns, same content, production block sizes. The only variable is where
the recall block sits. Ollama re-prefills from the first token that differs from
the previous request, so a volatile block poisons everything after it.

| | recall mid-prompt | recall at tail | |
|---|---|---|---|
| prompt tokens p50 | 4,677 | 4,677 | [M] |
| prefill reuse p50 | 0.408 | **0.717** | [M] |
| reuse on recall turns (median) | 0.373 | **0.688** | [M] |
| stable prefix | 1,566 / 18,708 (8%) | **16,451 / 18,708 (88%)** | [M] |

~2,769 -> ~1,324 tokens re-prefilled per turn at p50. The order is now
`persona | summary | verbatim | recall | director | new user` — append-only except
when summarisation fires and rewrites the summary.

Confirmed on the live stack — real `vessel`, real memory, 16 turns, summariser
and retrieval both firing: prompt tokens p50 2,870 / max 4,305, prefill reuse
**p50 0.705 / p95 0.846**, recall on 9 of 16 turns, 4.16 chars per token, final
window 16,538 chars of which 12,380 are stable prefix [M]. The tail-recall
number survives contact with a real summariser rewriting the summary mid-run.

## Stage 3 result — llama-server beats Ollama on every axis measured

Same GGUF blob, same 8 GB RTX 4060 Laptop, same `num_ctx` 12288. Ollama column is
the Stage 1 f16/12288 figure, re-confirmed live (`ollama ps`: `vessel:latest /
6.4 GB / 100% GPU / context 12288`).

| | Ollama | llama-server | |
|---|---|---|---|
| resident VRAM | 6.4 GB | **6.03 GB** | [M] |
| decode | 39.4 tok/s | **46.5 tok/s** (+18%) | [M] |
| prefill | ~1,600 tok/s | **~2,200–2,360 tok/s** | [M] |
| model load | 10,456 ms cold, 19,365 ms after eviction | **2,577 ms once, then never** | [M] |
| KV reuse | estimated from char diffs | **measured, `cache_n`** | [M] |

### Launch it with `--parallel 1`

An earlier draft of this doc claimed `n_slots 4`, `n_ctx_slot 12288`,
`kv_unified true` — one shared KV pool with four free slots. That was wrong on
two counts, and the live `/props` disagrees with it [M].

`--parallel N` **divides** `-c`, it does not multiply it. `-c 12288 --parallel 4`
gives `n_ctx_slot 3072`: each conversation gets a quarter of the window, not all
of it. And `kv_unified` reported `'false'`, so the slots are four private pools,
not one shared one — there is no borrowing between them.

Vessel is a single-user desktop app that already serialises the summariser after
the turn (see *Concurrency is a trap*, below). It never needs a second slot, and
paying for three unused ones costs three quarters of the context window.

| `--parallel` | `n_ctx_slot` | VRAM | decode | |
|---|---|---|---|---|
| 4 (was shipped) | 3,072 | 6,163 MiB | 46.2 tok/s | [M] |
| **1 (required)** | **12,288** | 6,163 MiB | 46.2 tok/s | [M] |

Same VRAM, same decode, 4x the usable window. `--parallel 1` is not a tuning
preference; the shipped value was silently truncating long conversations to a
quarter of the context they were configured for. `.env.example` documents the
corrected launch line.

### The eviction cost is gone, not reduced

Sequence prime -> warm -> summariser (`cache_prompt: false`, 5,200-token distinct
prompt) -> chat again:

| | Ollama | llama-server | |
|---|---|---|---|
| chat turn after a summary | 19,365 ms reload first | **1,889 ms, `loadMs: 0`** | [M] |
| KV reuse across the summary | n/a (evicted) | **0.996** | [M] |
| VRAM across the summary | model swap | 6,171 -> 6,173 MiB | [M] |

### Concurrency is a trap

Running chat and summariser simultaneously finishes sooner in wall-clock —
7,543 ms vs 13,159 ms serial — but chat decode collapses to **3.41 tok/s** [M].
An 8B model on a 4060 is compute-bound; two streams share one SM pool. The
existing code already serialises the summariser after the turn. Keep it.

### End to end on the real GPU

17 turns through the real product (character create -> chat -> archive ->
summarise -> delete), `INFERENCE_BACKEND=llama-server`:

| | | |
|---|---|---|
| decode | p50 **46.4** / p95 47.1 tok/s | [M] |
| KV reuse (`cache_n`) | p50 **0.985** / p95 0.999 | [M] |
| prompt tokens | p50 1,489 / p95 2,469 | [M] |
| `loadMs` | **0 on every turn** | [M] |
| chars per token | **4.81** | [M] |
| archive rows / summary | 24 / 1,913 chars | [M] |
| refusal or assistant-voice markers | **none** | [M] |

`charsPerToken` 4.81 measured vs 4.0 assumed above — the token estimates in
*Live window ceiling* are ~20% conservative, which leaves `OLLAMA_NUM_CTX=12288`
with even more headroom than stated.

### Launch requirement

Ollama's bundled `llama-server.exe` does **not** auto-discover its CUDA backend.
Without pointing at it explicitly the server prints `no usable GPU found`,
ignores `-ngl`, and runs on CPU — a silent ~10x loss:

```
set GGML_BACKEND_PATH=%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll
llama-server -m <model.gguf> -c 12288 -ngl 99 --host 127.0.0.1 --port 8080
```

`cuda_v13` because `nvidia-smi` reports CUDA UMD 13.4. Full notes in
`.env.example`.

## Stage 4a result - the Rust core

Decision 0001 in code. Two crates in one workspace: `src-core` (`vessel-core`,
no GUI dependency of any kind) and `src-tauri` (the shell). The renderer is the
same React/Vite build, untouched.

What the port removes is not lines, it is a whole tier. The Electron build ran
an Express server on 127.0.0.1:3001 and talked to it over loopback HTTP; the
Tauri build calls the core directly through typed commands, so there is no
socket, no origin, no CORS policy, no Host allowlist and no port to collide
with. Roughly half of the Node security suite defended a perimeter that no
longer exists - see the module doc on `src-core/tests/security.rs` for which
tests were carried across and which were deleted, and why.

| Area | Electron track | Rust track | |
|---|---|---|---|
| backend / core, production only | 3,996 | 5,690 | [M] |
| of which `inference/` | 700 | 1,626 | [M] |
| shell (main + preload / Tauri) | 239 | 524 | [M] |
| tests | 2,766 | 3,058 | [M] |
| test count | 154 | 150 | [M] |
| renderer JSX | 1,233 | shared, unchanged | [M] |
| renderer CSS | 752 | shared, unchanged | [M] |

Rust is longer per unit of behaviour and that is the trade: explicit error
types, no prototype chain to smuggle a key through, and a compiler that rejects
the shape of bug the Node suite had to test for. The test counts are close
because both suites cover the same behaviour; the Rust side folds 126 of its
150 into the modules they test, so a failure names the function rather than an
endpoint.

**Not yet verified at runtime.** The shell needs a visible desktop window
(WebView2), which no test here can open. The IPC ACL in
`src-tauri/capabilities/default.json` and `permissions/ipc.toml` has never been
exercised against a live renderer. Stage 4b - in-process llama.cpp via
`llama-cpp-2`, which deletes the `reqwest` dependency and the last HTTP hop -
needs Visual Studio Build Tools with the C++ workload and the CUDA Toolkit, and
is blocked on those being installed.
