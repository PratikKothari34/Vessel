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
Electron main ──spawns──> Node/Express backend (127.0.0.1) ──HTTP──> Ollama
     │                          │
  React renderer           Turso (@tursodatabase/sync)  (local file; optional cloud sync)
  (Vite)                   characters / conversations / turns / archive(+embeddings)
```

The model's live window is kept small (32K) for speed. Older turns fold out of it
and are **archived with embeddings** (nomic-embed-text); relevant ones are
recalled per message by cosine-ranking the stored embeddings in JS (the Turso
sync engine has no native vector search).

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

1. **Node.js 18+**
2. **Ollama** running locally, with three models:
   ```bash
   ollama pull Tohur/natsumura-storytelling-rp-llama-3.1:8b
   ollama pull gemma3:4b
   ollama pull nomic-embed-text
   ```
3. **The custom chat model** (built from the included `Modelfile`):
   ```bash
   ollama create vessel -f Modelfile
   ```

---

## Run (development)

```bash
# 1. backend deps (repo root)
npm install

# 2. app deps
cd app && npm install

# 3. (optional) config — copy and edit if you want cloud sync or different tuning
cp ../.env.example ../.env

# 4. launch (starts backend + Electron window)
npm run dev
```

The app spawns the backend automatically and waits for it to be healthy before
showing the window.

> **Note:** if `ELECTRON_RUN_AS_NODE=1` is set in your shell, the dev launcher
> (`app/scripts/dev.mjs`) clears it for the app process — otherwise Electron would
> run headless as plain Node.

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
| `MAX_SUMMARY_CHARS` | `6000` | Hard cap on rolling-summary length |
| `SUMMARIZER_NUM_CTX` | `8192` | Summarizer context window |

> `SUMMARIZE_THRESHOLD` must sit above `VERBATIM_TURNS` — archiving can't trigger
> below the verbatim window. If you set it lower, the backend raises it to
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

### Or build the installer yourself

```bash
cd app
npm run package      # -> app/dist/*.exe (NSIS installer)
```

The backend (`src/`, `node_modules`, `Modelfile`) is bundled into the app's
resources. The installed app still requires **Ollama + the models** on the target
machine.

---

## Project layout

```
Vessel/
├── Modelfile               # ollama create vessel -f Modelfile
├── .env.example
├── src/backend/
│   ├── server.js           # Express + SSE /chat + REST
│   ├── db.js               # Turso sync client + schema + embedding codec
│   ├── memory.js           # summary + retrieval engine
│   ├── characters.js       # character CRUD
│   └── inference/          # ollama / llama-server adapters behind one interface
├── app/                    # Electron + React (Vite)
│   └── src/
│       ├── main/           # spawns backend, creates window
│       ├── preload/
│       └── renderer/src/   # React UI (Gallery, Chat, Editor, Settings, Memory)
├── test/                   # node:test suites — see Tests below
└── docs/
    ├── MASTER.md           # every measured number, in one place
    └── decisions/          # why the load-bearing choices were made
```

A Rust port of the backend lives beside this one in `src-core/` (all the logic)
and `src-tauri/` (a Tauri shell), tracking
`docs/decisions/0001-target-architecture.md`. It is not what the installer
builds, and nothing above depends on it.

---

## Tests

Two suites, no test dependencies in either.

```bash
npm test                      # 202 tests — the Node backend, unit + integration
cargo test -p vessel-core     # 165 tests — the Rust core
```

`npm test` spawns its own backend on a scratch database with a fake inference
engine, so it never touches your real data, your keychain or a model. It needs
nothing running.

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
