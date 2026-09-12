# 0002 — The model's rules live in the prompt, not in the model

**Status:** accepted
**Supersedes:** nothing
**Numbers:** `docs/MASTER.md`

## Decision

The `Modelfile`'s `SYSTEM` block and its `PARAMETER` lines are read at runtime by
`src/backend/inference/modelfile.js` and applied by Vessel itself:

- **`SYSTEM`** is prepended to the per-character persona message in
  `server.js:buildPersonaMessage`.
- **`PARAMETER`** values are sent as sampling defaults by the llama-server
  adapter, under anything the request supplies.

The `Modelfile` stays the single source of truth for both. Nothing is duplicated
into code.

## Why

Two separate problems, one cause: **baked-in model config is not portable, and on
Ollama it was not even applied.**

### 1. The global roleplay rules had never reached the model

Ollama renders a Modelfile `SYSTEM` only when the client sends no system message
of its own. When the client does send one, that message *becomes* the system
prompt and the Modelfile's is dropped entirely.

`buildPersonaMessage` always produces a system message. So the `SYSTEM` block —
the "no refusal, no moralizing, never write for the user" rules that are the
entire point of an uncensored roleplay app — was dead config on every chat turn
the app has ever made.

Measured against the real Ollama on `vessel`, `num_predict: 1`:

| Request | `prompt_eval_count` | Fixed prefix |
|---|---|---|
| no client system message | 492 | 492 tokens (template + Modelfile SYSTEM) |
| client system message, short | 96 | **26 tokens** (template only) |
| client system message, 6x longer | 445 | **26 tokens** (template only) |

The fixed prefix collapses from 492 to 26 the moment a client system message
exists. The 2,211-character `SYSTEM` was not in the prompt.

### 2. llama-server has no concept of a Modelfile at all

Stage 3 loads the raw GGUF blob. The blob carries weights and a chat template;
`SYSTEM`, `temperature`, `min_p`, `repeat_last_n` live in Ollama's manifest, not
in it. A backend swap that ignored this would have silently shipped a different
product on llama-server: stock llama.cpp sampling and no behavioural rules.

Reading the `Modelfile` fixes both at once and makes the two backends generate
from the same prompt with the same sampling — which is the only thing that makes
the Stage 3 A/B numbers comparable.

## Cost

+2,211 characters (~470 tokens) at the **head** of the cacheable prefix. It is
prefilled once per conversation and served from the KV cache on every later turn;
measured steady-state reuse on the real GPU is `cache_n` p50 **0.985**. The
recurring cost is zero.

## Rejected

- **Copy the SYSTEM text into `server.js`.** Two sources of truth for what the
  model is. The next `ollama create` would silently disagree with the running app.
- **Have the llama-server adapter inject the `SYSTEM` itself.** It would then
  appear on llama-server and not on Ollama — the exact divergence this decision
  exists to remove. The adapter deliberately does not touch messages.
- **Apply the Modelfile's sampling to the summarizer too.** Those values are
  tuned to make roleplay prose surprising, which is wrong for a summary. The
  summarizer sends its own low-variance sampling; see `llama-server.js:generate`.

## Verification

- `rig/verify.js` — "global roleplay rules reached the model" and "global rules
  sent exactly once", asserted on **both** backends.
- `rig/stage3-adapter.js` — Modelfile parameters on the llama.cpp wire, character
  sampling overriding them, and the adapter adding no system message of its own.
- Negative control: removing the prepend fails exactly those assertions on both
  arms and nothing else; removing the parameter merge fails exactly the three
  parameter assertions.
