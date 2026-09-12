# MASTER

Consolidated numbers for Vessel. **Not an authority** — loses to `CLAUDE.md`
and `docs/decisions/`, wins on numbers.

Every figure is tagged **[M]** measured or **[E]** estimated. Replacing [E] with
[M] is the point of Stage 0.

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
| chat (Llama-3.1-8B Q4_K_M) | `vessel` | ~4.9 GB | [E] check actual GGUF |
| `gemma3:4b` | summariser | ~3.3 GB | [E] |
| `nomic-embed-text` | embeddings, 768-dim | ~0.27 GB | [E] |
| **total** | | **~8.5 GB** | [E] |

Models are ~96% of on-disk footprint. The entire Electron shell is ~3.5%.

## KV cache

Llama-3.1-8B: 32 layers, 8 KV heads (GQA), 128 head dim.

```
per token = 2 x 32 x 8 x 128 x 2 bytes (f16) = 131,072 B = 128 KB
```

| `num_ctx` | dtype | KV | |
|---|---|---|---|
| 32768 (current) | f16 | **4.00 GiB** | [E] arithmetic |
| 12288 | f16 | 1.50 GiB | [E] |
| 12288 | q8_0 | **0.75 GiB** | [E] |

KV is allocated eagerly at `num_ctx`. Current total demand:
`4.9 weights + 4.0 KV + 0.5 compute = 9.4 GB` against an 8 GB RTX 4060 —
**the model is spilling to CPU.**

## Live window ceiling

The memory architecture caps the window by construction:

| Component | Source | Tokens | |
|---|---|---|---|
| Modelfile SYSTEM + persona | `Modelfile` | ~1,100 | [E] |
| rolling summary | `MAX_SUMMARY_CHARS=6000` | ~1,500 | [E] |
| retrieved | `RETRIEVE_K=4` x 512 | ~2,048 | [E] |
| verbatim | `SUMMARIZE_THRESHOLD=12` | ~4,200 | [E] |
| new message + `num_predict` | `server.js:23` = 512 | ~660 | [E] |
| **worst case** | | **~9,500** | [E] |

`num_ctx 32768` is ~3.5x the ceiling and ~6x typical. **Measure before setting.**

## Optimization ledger

Ranked by bytes saved.

| Lever | Saves | Cost |
|---|---|---|
| `num_ctx` 32768 -> 12288 | ~2.5 GiB VRAM | one value, after measuring |
| `SUMMARIZER_MODEL=vessel` (drop gemma3:4b) | ~3.3 GB disk | test summary quality |
| KV `q8_0` | ~0.75 GiB VRAM | one flag |
| Q4_K_M -> IQ4_XS | ~0.47 GB disk + VRAM | negligible quality |
| 384-dim embedder (bge-small / MiniLM) | ~0.18 GB disk, 2x faster cosine | slight recall loss |
| int8 embeddings in DB | 3072 -> 768 B/turn (4x) | negligible |
| Electron -> Tauri | 0.19 GB disk, ~150 MB RAM | 234 lines -> Rust |
| Node backend -> in-process Rust | 0.04 GB disk, ~70 MB RAM | 2175 lines -> Rust |

**Endpoint:** disk 8.8 GB -> ~4.6 GB. VRAM 9.4 GB (spilling) -> ~5.2 GB
(resident, ~2.8 GB headroom for a speculative-decoding draft model).

The shell is **3% of the disk win and 5% of the RAM win.** It goes last.

## Known hot paths

| Issue | Where | Effect |
|---|---|---|
| Retrieval block sits mid-prefix, changes every turn | `memory.js` `buildContext` | Invalidates KV reuse for the whole verbatim window — re-prefills ~2-4K tokens per turn |
| `embed()` blocks before the chat call, on CPU | `memory.js:135`, `EMBED_NUM_GPU=0` | 20-80 ms on the critical path |
| Full archive scan + Float32 decode per message | `memory.js` `retrieve` | 3072 B/row read + decode + 2 sqrt loops per row |
| Archive loop holds the per-conversation lock | `memory.js` `recordTurn` | Serial embed + 2 writes x N turns; next message blocks |
| `prompt_eval_count` parsed then discarded | `server.js:439` | The measurement needed for Stage 0 is already in the stream |

## Code size

| Area | Lines | |
|---|---|---|
| backend (`src/backend/`) | 2,175 | [M] |
| renderer JSX | 1,223 | [M] |
| renderer CSS | 752 | [M] |
| `main/index.js` + `preload/index.js` | 234 | [M] |
