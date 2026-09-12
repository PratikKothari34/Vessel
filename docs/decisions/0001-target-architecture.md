# 0001 — Target architecture: Tauri + Rust + in-process llama.cpp

**Status:** accepted
**Supersedes:** nothing
**Numbers:** `docs/MASTER.md`

## Decision

Vessel moves to:

| Layer | Choice |
|---|---|
| Shell | Tauri 2.x (WebView2) |
| Renderer | existing React / Vite build, **unchanged** |
| Core | Rust |
| Inference | llama.cpp in-process (`llama-cpp-2`) |
| DB | `libsql` / `turso` Rust crate |
| Secrets | `keyring` crate — `SERVICE` stays `scenario-chat` |

No Node, no sidecar, no N-API addons, no loopback HTTP.

## Why not the alternatives

- **Electron (current).** Its one real advantage is that the shell doubles as
  the Node runtime, so the backend runs free. That advantage disappears the
  moment the backend is Rust. Costs ~250 MB of shell and ~150 MB RAM for
  nothing on the generation path.
- **React Native Windows. Rejected.** No DOM — ~1,975 lines of JSX and CSS
  would be rewritten. `app/src/renderer/src/lib/api.js:75` streams with
  `res.body.getReader()`, which RN's fetch does not implement. Spawning the
  backend needs a hand-written C++/C# native module. Its only performance
  argument is native text rendering, worth microseconds against a ~25 ms
  inter-token budget. Weakest desktop maturity of the three.
- Mobile is **not** a goal. If it becomes one, it is a second client against
  the same schema, not a migration.

## Why in-process llama.cpp is the real reason

Footprint is not the motive — it is 3% of the total win. These are unreachable
from any HTTP-based design:

- **Summarisation reuses the already-loaded chat weights** at zero marginal
  VRAM and zero marginal disk. Deletes `gemma3:4b` outright (~3.3 GB).
- **Persistent per-conversation KV cache** (slot save/restore). Resuming a story
  costs a file read instead of a full prefill.
- **Pipelining** — prefill the stable prefix while the embed/retrieve runs,
  hiding the CPU embed entirely.
- Direct token callbacks. No HTTP, no NDJSON parse, no SSE re-encode.
- Full control of `num_ctx`, KV dtype, flash attention, speculative decoding.

## Staging

Each stage ships and measures independently. Every stage survives the next.

| # | Stage | Notes |
|---|---|---|
| 0 | Instrument | `prompt_eval_count` is already in the stream at `server.js:439` and discarded. ~3 lines. |
| 1 | KV quant test | `OLLAMA_KV_CACHE_TYPE=q8_0` — confirms the VRAM spill hypothesis with no code change. |
| 2 | Prompt reorder + retrieval fixes | Pure algorithm. Ports to any stack unchanged. |
| 3 | Ollama -> llama-server | Still Node. Gains slot reuse and KV control; removes the three-model install wall. |
| 4 | Rust + Tauri | In-process llama.cpp, persistent KV, single binary. |

Stages 0-2 are free and carry forward verbatim. Stage 3 proves llama.cpp on the
target GPU before committing to Rust. **Stage 4 must not start before 0-3** —
building it first hardcodes guesses that measurement would have corrected.

## Open risk

Whether the Rust `turso`/`libsql` crate opens the **same aes256gcm whole-file
format** that `@tursodatabase/database` writes.

If it does not, existing encrypted databases need an export path through the
Node backend before cutover, and that backend must stay alive through the
transition. This is the single unknown that can change the shape of Stage 4.
**Answer it before writing Stage 4 code.**
