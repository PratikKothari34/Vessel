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

| # | Stage | Status |
|---|---|---|
| 0 | Instrument | **done** — `metrics.js`, `GET /metrics`. |
| 1 | KV quant test | **done, and it answered differently than expected.** The spill is real (9.52 GB resident, 3.27 GB on CPU), but `num_ctx` is the lever, not KV dtype: 32768 -> 12288 takes decode 13.69 -> 39.37 tok/s, while `q8_0` at 12288 is *slower* than f16. Adopted `OLLAMA_NUM_CTX=12288`, rejected `q8_0`. Numbers in `docs/MASTER.md`. |
| 2 | Prompt reorder + retrieval fixes | **done** — prefill reuse p50 0.408 -> 0.717; stable prefix 8% -> 88%. |
| 3 | Ollama -> llama-server | next. Still Node. Gains slot reuse and KV control; removes the three-model install wall. |
| 4 | Rust + Tauri | In-process llama.cpp, persistent KV, single binary. Open risk below is now closed. |

Stage 1 also surfaced the cost Stage 3 and Stage 4 are meant to delete. Ollama
keys a resident model on **(model, num_ctx)**, so the summariser evicts the chat
model on every summary: the next user message waits **19.4 s** for a reload. That
is not a tuning problem — it is the "summarisation reuses the already-loaded chat
weights" argument above, priced.

Stages 0-2 are free and carry forward verbatim. Stage 3 proves llama.cpp on the
target GPU before committing to Rust. **Stage 4 must not start before 0-3** —
building it first hardcodes guesses that measurement would have corrected.

## Open risk — CLOSED

> Whether the Rust `turso`/`libsql` crate opens the **same aes256gcm whole-file
> format** that `@tursodatabase/database` writes.

**It does.** Measured, not reasoned:

1. `@tursodatabase/database` 0.7.2 — the version Vessel ships — wrote a throwaway
   database with the exact options `db.js` passes
   (`{ cipher: 'aes256gcm', hexkey }`) and inserted a canary row. The file starts
   `54 75 72 73 6f 00 02` (`Turso\0\x02`), not `SQLite format 3` — the encrypted
   container is Turso's own whole-file format, so this could not be assumed.
2. A Rust binary against the `turso` crate at the matching 0.7.2, using
   `Builder::new_local(path).experimental_encryption(true).with_encryption(
   EncryptionOpts { cipher: "aes256gcm", hexkey })`, read the canary back:
   `RESULT=OK note=Text("stage4-canary")`.
3. Negative control, so that `OK` means something: the same binary with a wrong
   key fails at the first page —
   `RESULT=OPEN_FAILED error=Decryption failed for page=1`.

`EncryptionOpts { cipher: String, hexkey: String }` is the same shape as the JS
option object, and the Rust sync engine omits at-rest encryption for the same
reason the JS one does — so the two-driver split in `db.js` carries over to Rust
unchanged rather than being a Node-specific workaround.

**Consequence:** Stage 4 needs **no export path** and no transitional Node
backend. An existing encrypted database opens directly, with the key still read
from the OS keychain under `SERVICE = 'scenario-chat'`.

Re-verify if the pinned crate version moves: this was proven at 0.7.2 on both
sides, and the container carries a format version byte (`\x02`).
