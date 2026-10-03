# Vessel

Long stories that don't lose the thread. Local roleplay for Windows, running on
[Ollama](https://ollama.com) — nothing leaves your machine.

Vessel keeps a story coherent past the model's context window. Old turns fold
out of the live window and are archived with embeddings, then recalled by
relevance when they matter again, so the conversation keeps its history without
the live window growing. Multiple characters, each with its own persona, and the
local database is encrypted at rest. Optional cloud backup + multi-device sync
via [Turso](https://turso.tech) — which trades the encrypted local file for a
synced one (see Privacy).

- **Long-term memory** — a small, fast live window plus embedding-based retrieval
  over everything archived, so long stories stay coherent without slowing down.
- **Runs on your hardware** — no external API, no account, no request leaving
  the machine. Whatever model you point it at is the model you get. Ships
  configured for the Natsumura storytelling/roleplay model.
- **Multi-character** — create personas (name, persona, greeting, avatar, sampling)
  and switch between them.
- **Swipe variants** — regenerate a reply to get alternates; swipe `◀ 2/3 ▶`
  between them. All variants persist; the one you pick becomes canonical for memory.
- **Director / OOC mode** — steer the AI with out-of-character instructions
  (e.g. "focus on dialogue, less narration") via the ◈ toggle or a `//` prefix.
  Director notes guide behavior but are never written into the story or memory.
- **Response style** — per-character setting (balanced / dialogue-first /
  light-narration) to stop the model from only narrating instead of speaking.
- **Local-first storage** — SQLite (Turso) on disk, encrypted at rest with a key
  in the OS keychain. Cloud sync is opt-in and trades that encryption for a
  synced file (see Privacy).

---

## Architecture

```
Tauri shell (Rust) ──IPC──> vessel-core ──HTTP──> Ollama
     │                          │
  React renderer           Turso (@tursodatabase/sync)  (local file; optional cloud sync)
  (Vite)                   characters / conversations / turns / archive(+embeddings)
```

There is no HTTP port in this path: every route is a `#[tauri::command]` in
`src-tauri/src/lib.rs`, and the renderer reaches it over IPC. The React UI in
`app/` is shared infrastructure, not a second app - `tauri.conf.json` points
`frontendDist` at `app/out/renderer` and compiles it into the binary.

There is one implementation. The Node/Express backend that used to serve this
on port 3001 was removed on 2026-10-03, along with the Electron shell that
spawned it - its logic lives in `src-core/` and its routes are Tauri commands.

The model's live window is kept small - 12,288 tokens, not the 32,768 the model
will accept - because that is the single largest speed lever measured: it freed
3.17 GB of VRAM and tripled decode (`docs/MASTER.md`, Stage 1). Older turns fold
out of it and are **archived with embeddings** (nomic-embed-text); relevant ones are
recalled per message by cosine-ranking the stored embeddings in JS (the Turso
sync engine has no native vector search).

Those recalled turns are distinct by construction. A long conversation repeats
itself, and near-identical turns score near-identically against a query, so
without a rule against it a single repeated line wins every slot and the reply is
built on one memory instead of four. Turns within `RETRIEVE_DUP_MAX` cosine of an
already-chosen one share its slot; measured over a 20,000-exchange run this moved
recall at +4000 turns from 0/40 to 35/40 at no measurable cost. See
`docs/decisions/0005-retrieval-returns-distinct-memories.md`.

Ollama is the default, not the only option. `INFERENCE_BACKEND=llama-server`
points the same backend at a llama.cpp server instead, which on an RTX 4060 8GB
measured faster on every axis - decode 46.5 vs 39.4 tok/s, prefill ~2,200 vs
~1,600 - and holds one model for the life of the process so nothing can evict
it. It needs a server you start yourself; `.env.example` has the launch line and
the flags that matter.

A **rolling summary** (gemma3:4b) can narrate what fell out of the window as
well, but it ships **off**. Measured over 2,000 exchanges it left recall
unchanged (33.8% vs 39.4%, p=0.30) while cutting accuracy when the model
committed to an answer from 61.8% to 44.3%, and cost 836 minutes of CPU against
zero. Set `SUMMARY_ENABLED=1` to turn it on; see
`docs/decisions/0004-the-rolling-summary-is-off-by-default.md`.

---

## Prerequisites

1. **Ollama** running locally, with the three models Vessel employs - one per
   role, and only the summariser is optional:
   ```bash
   ollama pull Tohur/natsumura-storytelling-rp-llama-3.1:8b   # chat, 4.9 GB
   ollama pull nomic-embed-text                               # embeddings, 274 MB
   ollama pull gemma3:4b                                      # summariser, 3.3 GB
   ```
   `gemma3:4b` is only read when `SUMMARY_ENABLED=1`, which is not the default,
   so you can skip it until you turn the rolling summary on.
2. **The chat model itself** - `vessel`, built from the included `Modelfile`:
   ```bash
   ollama create vessel -f Modelfile
   ```
   This is a local alias over the Natsumura pull above, not a second download:
   it fixes `num_ctx` at 32768, sets the sampling (temperature 0.9, top_p 0.95,
   min_p 0.05, repeat_penalty 1.1) and carries the global roleplay persona.
   `OLLAMA_MODEL` names the alias, so pointing it at the base model directly
   runs without any of that.

Node.js is **not** required to run Vessel - the app is a single Rust binary. It
is needed only to build the renderer and to run the Node test suite.

---

## Run (development)

```bash
# 1. renderer deps (the repo root has none - it is the Rust app plus a test
#    harness, and `cargo` fetches its own)
cd app && npm install

# 2. (optional) config — copy and edit if you want cloud sync or different tuning
cp ../.env.example ../.env

# 3. build the UI, then the binary
npm run build:renderer
cd .. && cargo build -p vessel --release --features bundled-ui
```

Run `target/release/vessel.exe`. The `bundled-ui` feature is not optional —
without it the binary compiles in `devUrl` and opens on
`ERR_CONNECTION_REFUSED`; see `src-tauri/Cargo.toml` for why that is decided at
compile time.

**Working on the UI?** `cd app && npm run dev` serves it on
`http://localhost:5173`, which is the `devUrl` in `tauri.conf.json`, so a
`cargo build` *without* `bundled-ui` gives you hot reload. Re-run
`npm run build:renderer` before any `bundled-ui` build — `frontendDist` is read
at compile time, so a stale bundle is embedded silently. Verify with
`grep -c "index-<hash>.js" target/release/vessel.exe`.

---

## Configuration (`.env`)

All optional — sane defaults work for a local-only setup. See `.env.example` for the
full list. Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `vessel` | Chat model |
| `SUMMARY_ENABLED` | `0` | Maintain the rolling summary. Off by default — see above |
| `SUMMARIZER_MODEL` | `gemma3:4b` | Rolling-summary model, when enabled |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model (768-dim) |
| `LOCAL_DB_PATH` | `./data/scenario.db` | Local SQLite file |
| `TURSO_DATABASE_URL` | *(blank)* | Set to enable cloud sync |
| `TURSO_AUTH_TOKEN` | *(blank)* | Turso auth token |
| `TURSO_SYNC_INTERVAL` | `60` | Background push/pull cadence (seconds); `0` = startup/shutdown only |
| `VERBATIM_TURNS` | `8` | Recent turns kept verbatim |
| `SUMMARIZE_THRESHOLD` | `12` | When to archive old turns — see the note below |
| `RETRIEVE_K` | `4` | Max recalled turns per message |
| `RETRIEVE_MIN_SCORE` | `0.45` | Min cosine similarity (0–1) for a recalled turn to count as relevant |
| `RETRIEVE_DUP_MAX` | `0.97` | Cosine above which two recalled turns count as one memory and share one slot |
| `MAX_SUMMARY_CHARS` | `6000` | Hard cap on rolling-summary length |
| `SUMMARIZER_NUM_CTX` | `8192` | Summarizer context window |

> `SUMMARIZE_THRESHOLD` must sit above `VERBATIM_TURNS` — archiving can't trigger
> below the verbatim window. If you set it lower, Vessel raises it to
> `VERBATIM_TURNS + 4` when it starts; `SUMMARIZE_THRESHOLD=6` with
> `VERBATIM_TURNS=8` runs as `12`, not `6`. Like all `.env` tuning, it's read from
> your config each launch — nothing here is fixed when the app is built.

### Cloud sync (optional)

Local-first with offline writes — the app always works offline; the cloud is a
backup/mirror you can restore from or read on another machine. Every user brings
their **own** Turso database; no credentials ship with the app.

1. Create a Turso DB: `turso db create vessel`
2. Get the URL + token:
   ```bash
   turso db show vessel --url
   turso db tokens create vessel
   ```
3. In the app: **Settings → Cloud sync**, paste the URL + token, save, restart.
   The URL is persisted to `data/settings.json`; the token goes to the OS
   keychain (never written to disk). Dev alternative: `TURSO_DATABASE_URL` /
   `TURSO_AUTH_TOKEN` in `.env` — in-app values override `.env`.

The schema is created on both local and remote (retrieval is in-JS cosine — the
sync engine has no native vector index).

> **Storage tip:** keep `LOCAL_DB_PATH` **outside** a OneDrive/Dropbox-synced
> folder — file-syncers can lock the SQLite file mid-write.

---

## Install (Windows)

Grab the latest `Vessel Setup *.exe` from the
[Releases](../../releases) page and run it (one-click, per-user install).
You still need **Ollama + the three models** (see Prerequisites) on the machine.
Your data lives in `%APPDATA%/Vessel/data/` and survives updates.

### Or build it yourself

```bash
cd app && npm run build:renderer && cd ..
cargo build -p vessel --release --features bundled-ui   # -> target/release/vessel.exe
```

That is the whole app: one ~20 MB binary with the UI compiled in, no Node
runtime and no backend process to spawn. It still requires **Ollama + the
models** on the target machine.

For an NSIS installer instead of a bare binary, `tauri.conf.json` is already
configured for it (`currentUser` install mode), but it needs the Tauri CLI,
which is not a dependency of this repo:

```bash
cargo install tauri-cli --locked
cargo tauri build          # -> src-tauri/target/release/bundle/nsis/
```

Installers under `app/dist/` are from the retired Electron build and are kept
for provenance only.

---

## Project layout

```
Vessel/
├── Modelfile               # ollama create vessel -f Modelfile
├── .env.example
├── src-core/               # vessel-core: the whole application, in Rust
│   ├── db.rs               # Turso sync client + schema + embedding codec
│   ├── memory.rs           # summary + retrieval engine
│   ├── chat.rs             # turn assembly + streaming
│   ├── characters.rs       # character CRUD
│   ├── keystore.rs         # OS keychain: DB encryption key + Turso token
│   ├── settings.rs         # data/settings.json, atomic write
│   ├── metrics.rs          # generation ring
│   └── inference/          # ollama / llama-server adapters behind one trait
├── src-tauri/              # the desktop shell: window + #[tauri::command]s
├── app/                    # React UI (Vite) — compiled into the Tauri binary
│   ├── vite.renderer.config.mjs   # the only UI build
│   └── src/renderer/src/   # React UI (Gallery, Chat, Editor, Settings, Memory)
├── test/                   # node:test over the renderer's pure modules
└── docs/
    ├── MASTER.md           # every measured number, in one place
    └── decisions/          # why the load-bearing choices were made
```

`src-core/` holds all the logic and `src-tauri/` the shell, which is what the
installer builds, tracking `docs/decisions/0001-target-architecture.md`. The
Node implementation this replaced was removed on 2026-10-03.

---

## Tests

Two suites, no test dependencies in either - the repo installs nothing to run
them.

```bash
cargo test -p vessel-core     # 201 tests — the application
npm test                      # 32 tests — the renderer's pure modules
```

`cargo test` covers the data layer, memory and retrieval, chat assembly,
the inference adapters, and a security suite that drives the real code paths;
every test that touches storage gets its own temporary database. `npm test`
covers the two renderer modules with logic worth asserting on - prose
tokenizing and the settings state machine - and opens no database, keychain or
model.

---

## Privacy

Everything is local by default: the LLM, the database, the conversations. No
telemetry, no external calls except to your own Ollama instance. Cloud sync is
strictly opt-in and only activates when you provide Turso credentials.

### Encryption at rest

The local database is encrypted with **aes256gcm** using a 256-bit key generated
on first run and stored in the OS keychain (Windows Credential Manager). The file
cannot be opened with a wrong key or no key at all. `/health` reports the live
state as `encryptedAtRest`, and Settings shows it too.

> **Cloud sync and local encryption are mutually exclusive right now.** The Turso
> sync engine has no local-file encryption — it only encrypts the cloud leg — so
> Vessel picks the storage driver at startup based on your settings:
>
> | Mode | Local file | Cloud backup |
> |---|---|---|
> | **Local-only** (default) | **encrypted** (aes256gcm) | — |
> | **Cloud sync on** | plaintext (warned at startup) | yes, over TLS |
>
> Turning sync off in Settings gets you an encrypted local database on the next
> start. If sync is configured but the remote is unreachable, Vessel falls back to
> local-only — and encrypts.
>
> **Upgrading an existing install:** an older plaintext database is migrated to
> encrypted automatically on first launch. The original is kept beside it as
> `scenario.db.plaintext-backup` — delete that file once you've confirmed things
> work, since it is *not* encrypted.

**If encryption can't be turned on, Vessel says so.** When a key exists but the
encrypted database cannot be decrypted with it — a rotated or lost keychain
entry, a restored backup from another machine, a damaged file — the backend
refuses to start rather than silently reopening your stories in plaintext. Your
existing data is left untouched and still encrypted, so recovering the original
key recovers the stories. In the rarer case where no key can be stored at all
(the OS keychain is unavailable), the app still runs, but `/health` reports *why*
encryption is off and Settings shows a warning instead of a quiet "no".

---

## License

[GPL-3.0](LICENSE) — free to use, modify, and redistribute; derivatives must
stay under the same license.
