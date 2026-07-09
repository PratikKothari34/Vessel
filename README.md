# Vessel

A local, private, **uncensored** character.ai-style roleplay desktop app for Windows.
Multiple characters, each with its own persona; long-term memory that survives the
model's context window; everything runs on your machine via [Ollama](https://ollama.com).
Optional encrypted cloud backup + multi-device sync via [Turso](https://turso.tech).

- **Local LLM** — no external API, no content filtering. Built on the Natsumura
  storytelling/roleplay model.
- **Multi-character** — create personas (name, persona, greeting, avatar, sampling),
  switch between them like character.ai.
- **Long-term memory** — small fast live window + rolling summary + embedding-based
  retrieval, so long stories stay coherent without slowing down.
- **Swipe variants** — regenerate a reply to get alternates; swipe `◀ 2/3 ▶`
  between them. All variants persist; the one you pick becomes canonical for memory.
- **Director / OOC mode** — steer the AI with out-of-character instructions
  (e.g. "focus on dialogue, less narration") via the ◈ toggle or a `//` prefix.
  Director notes guide behavior but are never written into the story or memory.
- **Response style** — per-character setting (balanced / dialogue-first /
  light-narration) to stop the model from only narrating instead of speaking.
- **Local-first storage** — SQLite (Turso) on disk; cloud sync is opt-in.

---

## Architecture

```
Electron main ──spawns──> Node/Express backend (127.0.0.1) ──HTTP──> Ollama
     │                          │
  React renderer           Turso (@tursodatabase/sync)  (local file; optional cloud sync)
  (Vite)                   characters / conversations / turns / archive(+embeddings)
```

The model's live window is kept small (32K) for speed. Older turns are folded into
a **rolling summary** (gemma3:4b) and **archived with embeddings** (nomic-embed-text);
relevant ones are recalled per message by cosine-ranking the stored embeddings in
JS (the Turso sync engine has no native vector search).

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
| `SUMMARIZER_MODEL` | `gemma3:4b` | Rolling-summary model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model (768-dim) |
| `LOCAL_DB_PATH` | `./data/scenario.db` | Local SQLite file |
| `TURSO_DATABASE_URL` | *(blank)* | Set to enable cloud sync |
| `TURSO_AUTH_TOKEN` | *(blank)* | Turso auth token |
| `VERBATIM_TURNS` | `8` | Recent turns kept verbatim |
| `SUMMARIZE_THRESHOLD` | `12` | When to archive old turns |
| `RETRIEVE_K` | `4` | Max recalled turns per message |

### Cloud sync (optional)

Local-first with offline writes — the app always works offline; the cloud is a
backup/mirror you can restore from or read on another machine.

1. Create a Turso DB: `turso db create vessel`
2. Get the URL + token:
   ```bash
   turso db show vessel --url
   turso db tokens create vessel
   ```
3. Put them in `.env` as `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN`, restart.

The schema (including the vector index) is created on both local and remote.

> **Storage tip:** keep `LOCAL_DB_PATH` **outside** a OneDrive/Dropbox-synced
> folder — file-syncers can lock the SQLite file mid-write.

---

## Build a Windows installer

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
│   └── characters.js       # character CRUD
└── app/                    # Electron + React (Vite)
    └── src/
        ├── main/           # spawns backend, creates window
        ├── preload/
        └── renderer/src/   # React UI (Gallery, Chat, Editor, Settings, Memory)
```

---

## Privacy

Everything is local by default: the LLM, the database, the conversations. No
telemetry, no external calls except to your own Ollama instance. Cloud sync is
strictly opt-in and only activates when you provide Turso credentials.
