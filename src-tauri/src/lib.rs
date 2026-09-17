//! The desktop shell: a window, a command surface, and nothing else.
//!
//! Every route the Express backend exposed is a `#[tauri::command]` here that
//! calls into [`vessel_core`], and `/chat`'s SSE stream is a Tauri channel. What
//! that deletes is the entire loopback perimeter - the DNS-rebinding Host guard,
//! the CORS allowlist, the CSRF app header, the body-size limit, the
//! port-reclaim retry loop - because there is no socket for anything to reach.
//! The IPC boundary is the process boundary.
//!
//! Errors keep the shape the renderer already parses: `{ error, detail? }`. The
//! HTTP status is gone, so anything that carried meaning in a status carries it
//! in a field instead - see [`vessel_core::chat::ErrorKind`].

use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value as Json};
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager};

use vessel_core::inference::Engine;
use vessel_core::{characters, chat, db, inference, keystore, memory, metrics, prompt, settings};

/// Cap on the shutdown wait for a background fold.
///
/// Losing one means the next launch re-summarises from scratch, so it is worth
/// waiting for - but a CPU summary can take 239 s and nobody should watch a
/// close button spin for four minutes. On timeout the turns are still verbatim
/// and untouched, so the next turn simply schedules another fold.
const SHUTDOWN_MAINTENANCE: Duration = Duration::from_secs(15);

// ---- Error shape ----------------------------------------------------------

/// What a failed command returns. `detail` carries the underlying message the
/// way the JSON error bodies did; the renderer shows `error` and logs `detail`.
#[derive(Debug, Serialize)]
pub struct CmdError {
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

impl CmdError {
    fn new(error: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            detail: None,
        }
    }
    fn detailed(error: impl Into<String>, e: impl std::fmt::Display) -> Self {
        Self {
            error: error.into(),
            detail: Some(e.to_string()),
        }
    }
}

type Cmd<T> = Result<T, CmdError>;

/// The open database, or a command error. Every data command starts here.
async fn open() -> Cmd<Arc<db::Db>> {
    db::get()
        .await
        .map_err(|e| CmdError::detailed("Database unavailable.", e))
}

// ---- Health ---------------------------------------------------------------

#[tauri::command]
async fn health() -> Cmd<Json> {
    let db = open().await?;
    let m = memory::config();
    let chat_engine = inference::chat().ok();
    Ok(json!({
        "status": "ok",
        "model": chat_engine.map(|e| e.model()),
        // `ollama` is the host of whatever engine serves chat. The key keeps its
        // name because Settings.jsx reads it; `inference` is the honest,
        // backend-aware version.
        "ollama": chat_engine.map(|e| e.host()),
        "inference": inference::describe(),
        "sync": {
            "enabled": db.is_sync_enabled(),
            "interval": vessel_core::config::sync_interval_secs(),
        },
        "encryptedAtRest": db.is_encrypted_at_rest(),
        // Why encryption is off, when it is - lets Settings tell an accepted
        // tradeoff (cloud sync) from a silent failure the user should act on.
        "unencryptedReason": db.unencrypted_reason(),
        "memory": {
            "summarizer": m.summarizer_model,
            "embedder": m.embed_model,
            "verbatimTurns": m.verbatim_turns,
            "summarizeThreshold": m.summarize_threshold,
        },
    }))
}

// ---- Metrics --------------------------------------------------------------

/// Stage 0 of decision 0001: the measurement that turned `num_ctx`, KV dtype and
/// the live-window ceiling from arithmetic into numbers. Local-only, no secrets,
/// no message text - counts and durations only.
#[tauri::command]
fn metrics_snapshot(limit: Option<usize>, conversation_id: Option<String>) -> Json {
    // Clamped to the ring, not to a literal. A hard 200 made METRICS_RING
    // unreadable above 200: the caller configured a larger ring, asked for it,
    // and was handed the newest 200 with nothing to say so.
    let limit = limit.unwrap_or(50).min(metrics::ring_size());
    let conv = conversation_id.filter(|id| memory::is_valid_id(id));
    metrics::snapshot(limit, conv.as_deref())
}

#[tauri::command]
fn metrics_reset() {
    metrics::reset();
}

// ---- Settings -------------------------------------------------------------

/// End users configure their OWN Turso database here (a packaged install ships
/// no `.env` and no baked-in credentials). The URL is persisted to
/// `settings.json`; the token goes to the OS keychain only - never written to
/// disk, never echoed back. Changes apply on next start: swapping the live DB
/// handle would race in-flight bookkeeping chains.
#[tauri::command]
async fn get_settings() -> Cmd<Json> {
    Ok(json!({
        "tursoUrl": settings::resolve_sync_url(),
        "tokenSet": !keystore::turso_token().is_empty(),
        "keychain": keystore::keychain_available(),
        // What the CURRENT boot connected with, which is not necessarily what is
        // configured above.
        "syncActive": db::get_if_ready().is_some_and(|d| d.is_sync_enabled()),
    }))
}

/// Accepts either field alone: the renderer saves the URL and the token
/// separately, and an absent field must not clear the stored one.
#[tauri::command]
async fn save_settings(turso_url: Option<String>, turso_token: Option<String>) -> Cmd<Json> {
    if let Some(raw) = turso_url {
        let url: String = raw.trim().chars().take(2048).collect();
        if !url.is_empty() && !is_sync_url(&url) {
            return Err(CmdError::new(
                "Database URL must start with libsql:// or https://.",
            ));
        }
        let mut patch = serde_json::Map::new();
        patch.insert("tursoUrl".into(), Json::String(url));
        settings::save(patch).map_err(|e| CmdError::detailed("Could not save settings.", e))?;
    }
    if let Some(raw) = turso_token {
        let token: String = raw.trim().chars().take(8192).collect();
        if !keystore::set_turso_token(&token) {
            return Err(CmdError::new(
                "OS keychain unavailable \u{2014} the token cannot be stored securely. Sync stays off.",
            ));
        }
    }
    Ok(json!({ "ok": true, "restartRequired": true }))
}

/// `^(libsql|https?)://\S+$`, without pulling in a regex engine for one check.
fn is_sync_url(url: &str) -> bool {
    let rest = ["libsql://", "https://", "http://"].iter().find_map(|p| {
        url.get(..p.len())
            .filter(|s| s.eq_ignore_ascii_case(p))
            .map(|_| &url[p.len()..])
    });
    matches!(rest, Some(r) if !r.is_empty() && !r.chars().any(char::is_whitespace))
}

// ---- Characters -----------------------------------------------------------

#[tauri::command]
async fn list_characters() -> Cmd<Json> {
    let list = characters::list()
        .await
        .map_err(|e| CmdError::detailed("Failed to list characters.", e))?;
    Ok(json!({ "characters": list }))
}

#[tauri::command]
async fn create_character(patch: characters::CharacterPatch) -> Cmd<characters::Character> {
    characters::create(patch)
        .await
        .map_err(|e| CmdError::detailed("Failed to create character.", e))?
        // `create` returns an Option because the INSERT is followed by a read
        // back; a None here means the row vanished between the two, which only a
        // concurrent delete of a just-created id could do.
        .ok_or_else(|| CmdError::new("Failed to create character."))
}

#[tauri::command]
async fn get_character(id: String) -> Cmd<characters::Character> {
    characters::get(&id)
        .await
        .map_err(|e| CmdError::detailed("Failed to read character.", e))?
        .ok_or_else(|| CmdError::new("Character not found."))
}

#[tauri::command]
async fn update_character(
    id: String,
    patch: characters::CharacterPatch,
) -> Cmd<characters::Character> {
    characters::update(&id, patch)
        .await
        .map_err(|e| CmdError::detailed("Failed to update character.", e))?
        .ok_or_else(|| CmdError::new("Character not found."))
}

/// The character's conversations go with it (`ON DELETE CASCADE`), so this has
/// to do everything [`delete_conversation`] does - stop live streams, stop
/// background folds, take each per-conversation lock - or the cascade lands in
/// the middle of a write and fails the `turns -> conversations` foreign key.
/// Same race, one level up.
#[tauri::command]
async fn delete_character(id: String) -> Cmd<Json> {
    if !characters::is_valid_id(&id) {
        return Err(CmdError::new("Invalid character id."));
    }
    let db = open().await?;
    let owned = memory::list_conversations(&db, Some(&id))
        .await
        .map_err(|e| CmdError::detailed("Failed to delete character.", e))?;

    for c in &owned {
        chat::cancel(&c.id);
        memory::cancel_maintenance(&c.id);
    }
    // Concurrently, not one after another: a character with 40 conversations
    // would otherwise serialise 40 bounded waits, making the ceiling 40 x
    // DELETE_LOCK_WAIT instead of one of them. `acquire_lock` resolves either
    // way - it proceeds unlocked once the wait is spent - so there is no
    // partial-acquire case to unwind. The guards are held until this returns.
    let _locks: Vec<memory::TurnLock> = futures_util::future::join_all(
        owned
            .iter()
            .map(|c| memory::acquire_lock(&c.id, chat::DELETE_LOCK_WAIT)),
    )
    .await;

    let removed = characters::delete(&id)
        .await
        .map_err(|e| CmdError::detailed("Failed to delete character.", e))?;
    if !removed {
        return Err(CmdError::new("Character not found."));
    }
    // Cached vectors, metrics and KV caches are dropped only after the delete
    // commits: a failed delete must leave them intact. The KV cache matters most
    // of the three - it is the persona and the story in another form, and this
    // is the path that deletes a character's whole history at once.
    for c in &owned {
        memory::forget_archive(&c.id);
        metrics::forget_conversation(&c.id);
        inference::forget_conversation(&c.id);
    }
    Ok(json!({ "deleted": true }))
}

// ---- Conversations --------------------------------------------------------

#[tauri::command]
async fn list_conversations(character_id: Option<String>) -> Cmd<Json> {
    if let Some(id) = character_id.as_deref() {
        if !characters::is_valid_id(id) {
            return Err(CmdError::new("Invalid character id."));
        }
    }
    let db = open().await?;
    let list = memory::list_conversations(&db, character_id.as_deref())
        .await
        .map_err(|e| CmdError::detailed("Failed to list conversations.", e))?;
    Ok(json!({ "conversations": list }))
}

#[tauri::command]
async fn get_conversation(id: String) -> Cmd<memory::ConversationDetail> {
    if !memory::is_valid_id(&id) {
        return Err(CmdError::new("Invalid conversation id."));
    }
    let db = open().await?;
    memory::get_conversation(&db, &id)
        .await
        .map_err(|e| CmdError::detailed("Failed to read conversation.", e))?
        .ok_or_else(|| CmdError::new("Conversation not found."))
}

#[tauri::command]
async fn set_conversation_title(id: String, title: String) -> Cmd<Json> {
    if !memory::is_valid_id(&id) {
        return Err(CmdError::new("Invalid conversation id."));
    }
    let db = open().await?;
    let title = memory::set_title(&db, &id, &title)
        .await
        .map_err(|e| CmdError::detailed("Failed to rename conversation.", e))?
        .ok_or_else(|| CmdError::new("Conversation not found."))?;
    Ok(json!({ "id": id, "title": title }))
}

/// Stop any generation still writing into this conversation, then take the same
/// per-conversation lock a chat turn holds. Recording a turn happens AFTER the
/// stream closes, so without both, the delete interleaves with that write and
/// fails the `turns -> conversations` foreign key. Measured on the Node build:
/// 5 of 5 deletes issued mid-stream failed that way before, 0 of 5 after.
#[tauri::command]
async fn delete_conversation(id: String) -> Cmd<Json> {
    if !memory::is_valid_id(&id) {
        return Err(CmdError::new("Invalid conversation id."));
    }
    let db = open().await?;
    chat::cancel(&id);
    let _lock = memory::acquire_lock(&id, chat::DELETE_LOCK_WAIT).await;
    let removed = memory::delete_conversation(&db, &id)
        .await
        .map_err(|e| CmdError::detailed("Failed to delete conversation.", e))?;
    metrics::forget_conversation(&id);
    if !removed {
        return Err(CmdError::new("Conversation not found."));
    }
    Ok(json!({ "deleted": true }))
}

/// Which variant of an assistant turn is active (swipe selection). Updates the
/// canonical turn text that memory and the summary read.
#[tauri::command]
async fn set_active_variant(
    id: String,
    turn_id: i64,
    variant_id: i64,
) -> Cmd<memory::ActiveVariant> {
    if !memory::is_valid_id(&id) {
        return Err(CmdError::new("Invalid conversation id."));
    }
    let db = open().await?;
    memory::set_active_variant(&db, &id, turn_id, variant_id)
        .await
        .map_err(|e| CmdError::detailed("Failed to set active variant.", e))
}

// ---- Chat -----------------------------------------------------------------

/// The renderer's end of the stream. A channel is ordered and needs no framing,
/// so the SSE parser the Electron build carried is gone with it.
struct ChannelSink(Channel<chat::ChatEvent>);

impl chat::Sink for ChannelSink {
    fn send(&self, event: chat::ChatEvent) {
        // A closed channel means the window went away mid-generation. The turn
        // still finishes and still records, exactly as the aborted-SSE path did.
        if let Err(e) = self.0.send(event) {
            tracing::debug!("chat channel closed: {e}");
        }
    }
}

#[tauri::command]
async fn chat(
    request: chat::ChatRequest,
    on_event: Channel<chat::ChatEvent>,
) -> Result<String, chat::ChatError> {
    chat::run(request, Arc::new(ChannelSink(on_event))).await
}

/// Stop the generation(s) running on a conversation. Returns how many were
/// signalled, so the renderer can tell a real stop from a stale click.
#[tauri::command]
fn cancel_chat(conversation_id: String) -> usize {
    chat::cancel(&conversation_id)
}

// ---- Lifecycle ------------------------------------------------------------

/// Push whatever is local up to the cloud. No-op when sync is off. The Electron
/// build reached this through `POST /shutdown` because Windows kills the backend
/// child before SIGTERM lands; in one process it is just a call.
#[tauri::command]
async fn flush_sync() -> Cmd<bool> {
    let db = open().await?;
    db.sync_now()
        .await
        .map_err(|e| CmdError::detailed("Sync failed.", e))
}

/// Restart the app so a settings change takes effect. Replaces
/// `window.scenario.relaunch`.
#[tauri::command]
fn relaunch(app: AppHandle) {
    app.restart()
}

/// Everything the Node build printed once at listen time. Assembled as one write
/// rather than six, so an async warning cannot land in the middle of it.
fn banner(db: &db::Db) {
    let m = memory::config();
    let mf = inference::modelfile::load();
    let gb = prompt::global_behavior();
    let num_ctx = chat::default_num_ctx();
    let chat_engine = inference::chat().ok();
    let (name, host, model, takes_ctx) = match chat_engine {
        Some(e) => (
            e.name(),
            e.host().to_string(),
            e.model().to_string(),
            e.accepts_num_ctx(),
        ),
        None => ("none", String::new(), String::new(), false),
    };
    let embedder = inference::embedder().map(|e| e.name()).unwrap_or("none");

    let mut lines = vec![
        "Vessel".to_string(),
        format!(
            "  -> inference: {name} @ {host} | model: {model} | num_ctx: {num_ctx}{}",
            if takes_ctx {
                ""
            } else {
                " (from -c at launch)"
            }
        ),
        format!(
            "  -> model rules: {}{}",
            if gb.is_empty() {
                "NONE".to_string()
            } else {
                format!("{} chars from Modelfile", gb.chars().count())
            },
            if mf.found {
                ""
            } else {
                " (Modelfile not found)"
            }
        ),
        format!(
            "  -> sync: {}",
            if db.is_sync_enabled() {
                "enabled"
            } else {
                "local-only"
            }
        ),
        format!(
            "  -> memory: summarizer={}, embedder={} ({embedder})",
            m.summarizer_model, m.embed_model
        ),
    ];
    // Without the Modelfile rules the model falls back to its own alignment and
    // starts refusing, moralizing, and writing the user's actions - the exact
    // behaviours that SYSTEM block exists to suppress. Silent degradation, so
    // say it loudly.
    if gb.is_empty() {
        lines.push(format!("  !! No SYSTEM found in {}.", mf.path.display()));
        lines.push("     Replies will not carry the global roleplay rules on any backend.".into());
    }
    // Same model, different window = two resident instances, so every summary
    // evicts the chat model and the next message pays a full reload (~19 s
    // measured). Only reachable by setting SUMMARIZER_NUM_CTX explicitly, and
    // only meaningful on a backend that takes the window per request.
    if takes_ctx && m.summarizer_model == model && m.summarizer_num_ctx != num_ctx {
        lines.push(format!(
            "  !! SUMMARIZER_NUM_CTX={} does not match num_ctx={num_ctx}.",
            m.summarizer_num_ctx
        ));
        lines.push(
            "     The summariser shares the chat model but not its slot, so each summary".into(),
        );
        lines
            .push("     reloads the model. Unset SUMMARIZER_NUM_CTX to keep one slot warm.".into());
    }
    tracing::info!("{}", lines.join("\n"));

    // On llama-server the window is whatever it was launched with. A silent
    // mismatch shows up later as a truncated story, so ask the server.
    tokio::spawn(async move {
        if let Some(n) = inference::probe_context().await {
            if n != num_ctx {
                tracing::warn!(
                    "llama-server is running at n_ctx={n}, but OLLAMA_NUM_CTX={num_ctx}.\n\
                     The window the model actually has is {n}. Relaunch it with -c {num_ctx}."
                );
            }
        }
    });
}

/// Wait for a background fold, then flush to the cloud. Called once, on the way
/// out.
async fn shutdown() {
    let _ = tokio::time::timeout(SHUTDOWN_MAINTENANCE, memory::await_all_maintenance()).await;
    if let Some(db) = db::get_if_ready() {
        let _ = db.sync_now().await;
    }
}

pub fn run() {
    // `.env` is a developer convenience; a packaged install has none and every
    // value falls back to its code default.
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_target(false)
        .init();

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            health,
            metrics_snapshot,
            metrics_reset,
            get_settings,
            save_settings,
            list_characters,
            create_character,
            get_character,
            update_character,
            delete_character,
            list_conversations,
            get_conversation,
            set_conversation_title,
            delete_conversation,
            set_active_variant,
            chat,
            cancel_chat,
            flush_sync,
            relaunch,
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            // The window is created hidden and shown here, so the user never
            // sees an empty frame while the schema opens and the first sync runs.
            tauri::async_runtime::spawn(async move {
                match db::get().await {
                    Ok(db) => banner(&db),
                    // Not fatal at this point: the window still opens and every
                    // command reports the same failure with its own message,
                    // which is far more useful than a dead process.
                    Err(e) => tracing::error!("database unavailable: {e}"),
                }
                if let Some(w) = handle.get_webview_window("main") {
                    let _ = w.show();
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to start Vessel")
        .run(|_app, event| {
            // `ExitRequested` fires before the process tears down, which is the
            // only point where a fold still in flight can be waited on.
            if let tauri::RunEvent::ExitRequested { .. } = event {
                tauri::async_runtime::block_on(shutdown());
            }
        });
}
