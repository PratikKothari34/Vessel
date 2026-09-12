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

This whole section is an **Ollama** problem. llama-server holds one model in one
unified KV pool across four slots, so a second prompt cannot evict the first —
see *Stage 3 result*. Under `INFERENCE_BACKEND=llama-server` the 19.4 s is 0 ms
and `SUMMARIZER_MODEL=vessel` costs nothing at all.

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
| `SUMMARIZER_MODEL=vessel` (drop gemma3:4b) | 3.34 GB disk + **19.4 s per summary** [M] | test summary quality — the one open call. Free on llama-server: one model, no eviction [M] |
| KV `q8_0` | 0.75 GiB VRAM, **-5% decode** [M] | rejected for now; see Stage 1 result |
| Q4_K_M -> IQ4_XS | ~0.47 GB disk + VRAM | negligible quality |
| 384-dim embedder (bge-small / MiniLM) | ~0.18 GB disk, 2x faster cosine | slight recall loss |
| int8 embeddings in DB | 3072 -> 774 B/turn (3.97x) [M] | **done** — max cosine error 1.2e-3, ranking identical |
| Electron -> Tauri | 0.19 GB disk, ~150 MB RAM | 234 lines -> Rust |
| Node backend -> in-process Rust | 0.04 GB disk, ~70 MB RAM | 3,449 lines -> Rust |

**Banked so far:** VRAM 9.52 GB spilling -> **6.03 GB fully resident** [M].
Decode 13.69 -> **46.4 tok/s p50** end to end on the real GPU [M] — **3.39x**,
from two levers: `num_ctx` 32768 -> 12288, then Ollama -> llama-server. Prefill
1,145 -> ~2,200 tok/s. The summariser's 19.4 s eviction is 0. Disk is unchanged
at 8.53 GB — every disk lever left is a model swap, and each one is a quality
call.

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

Server reports `n_slots 4`, `n_ctx_slot 12288`, `kv_unified true` — one KV pool,
four slots, nothing to evict.

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

## Code size

| Area | Lines | |
|---|---|---|
| backend (`src/backend/`) | 3,449 | [M] |
| of which `inference/` (Stage 3) | 700 | [M] |
| renderer JSX | 1,233 | [M] |
| renderer CSS | 752 | [M] |
| `main/index.js` + `preload/index.js` | 234 | [M] |
