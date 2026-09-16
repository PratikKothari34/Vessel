//! The chat turn, end to end.
//!
//! This is `POST /chat` from the Node backend with the HTTP removed. The shape
//! is unchanged - assemble the window, stream, record afterwards - because the
//! ordering in it is load-bearing:
//!
//! 1. Take the per-conversation lock, so a delete cannot interleave with the
//!    write that follows the stream.
//! 2. Assemble the window through [`crate::memory::build_context`].
//! 3. Stream, forwarding deltas to the caller's sink.
//! 4. Record telemetry, THEN the turn, THEN release the lock.
//!
//! What HTTP used to provide and now has to be explicit:
//!
//! - **Backpressure and framing.** SSE gave us both. A [`Sink`] is a plain
//!   callback, so the transport decides; the Tauri shell hands each event to an
//!   `ipc::Channel`, which is ordered and lock-free.
//! - **Client disconnect.** `res.on('close')` aborted the upstream request. The
//!   equivalent is [`cancel`], called by the shell when the user stops a
//!   generation and by a delete that needs the stream out of the way.
//!
//! What is deliberately NOT carried over is the raw wire line. The Express
//! version forwarded each engine chunk verbatim and the renderer re-parsed JSON
//! per token to dig out `message.content`. Over IPC the delta is sent as text,
//! so that parse disappears from the renderer's hot path.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::Duration;

use anyhow::Result;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

use crate::characters::{self, Character};
use crate::db;
use crate::inference::{self, ChatStart, Engine, Message, StreamEvent};
use crate::memory::{self, Retrieved};
use crate::metrics;
use crate::prompt;

// ---- Config ---------------------------------------------------------------

/// Default reply-length ceiling, in tokens. Bounds runaway "essay" replies even
/// when the model ignores the short-paragraph rule; a character can raise it
/// with its own `sampling.num_predict`. ~512 tokens is a few tight paragraphs.
pub fn default_num_predict() -> u32 {
    std::env::var("MAX_REPLY_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(512)
}

/// Context window, sent per request so the installed Ollama model is never
/// rebuilt to change it. The Modelfile's own `num_ctx 32768` measured 9.52 GB
/// resident against an 8 GB card - 3.27 GB spilled to CPU, decode down to 13.7
/// tok/s. At 12288 the model is 100% GPU-resident and decodes at 39.4. KV
/// quantisation is deliberately NOT used: once the model fits, q8_0 measured
/// slightly slower than f16.
///
/// On llama-server this is a LAUNCH flag (`-c`), not a request field, so it is
/// not sent there - only read, to check the running server agrees.
pub fn default_num_ctx() -> u32 {
    std::env::var("OLLAMA_NUM_CTX")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|v| *v >= 256)
        .unwrap_or(12288)
}

/// A delete aborts any live stream before taking the lock, so this only has to
/// cover the post-stream record (summarize + embed), not the generation.
pub const DELETE_LOCK_WAIT: Duration = Duration::from_secs(30);

// ---- Cancellation ---------------------------------------------------------

/// A one-shot stop signal. `AbortController`, with the two things this code
/// actually uses: a flag to test and a future to race against.
#[derive(Clone, Default)]
pub struct Cancel {
    flagged: Arc<AtomicBool>,
    notify: Arc<tokio::sync::Notify>,
}

impl Cancel {
    pub fn cancel(&self) {
        self.flagged.store(true, Ordering::SeqCst);
        // `notify_waiters` would be lost if nobody is parked yet.
        self.notify.notify_one();
    }

    pub fn is_cancelled(&self) -> bool {
        self.flagged.load(Ordering::SeqCst)
    }

    /// Resolves once cancelled. Safe to race in a `select!`: the permit
    /// `notify_one` leaves behind means a cancel that arrives before the wait
    /// still wakes it.
    pub async fn cancelled(&self) {
        while !self.is_cancelled() {
            self.notify.notified().await;
        }
    }
}

/// In-flight streams, keyed by conversation. Vec-valued because a regenerate can
/// overlap the turn it re-rolls.
fn live() -> &'static StdMutex<HashMap<String, Vec<Cancel>>> {
    static LIVE: OnceLock<StdMutex<HashMap<String, Vec<Cancel>>>> = OnceLock::new();
    LIVE.get_or_init(Default::default)
}

fn lock_live() -> std::sync::MutexGuard<'static, HashMap<String, Vec<Cancel>>> {
    match live().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

fn register(conversation_id: &str, c: &Cancel) {
    lock_live()
        .entry(conversation_id.to_string())
        .or_default()
        .push(c.clone());
}

fn unregister(conversation_id: &str, c: &Cancel) {
    let mut reg = lock_live();
    if let Some(v) = reg.get_mut(conversation_id) {
        v.retain(|x| !Arc::ptr_eq(&x.flagged, &c.flagged));
        if v.is_empty() {
            reg.remove(conversation_id);
        }
    }
}

/// Stop every generation still writing into a conversation. Returns how many
/// were signalled, which is what tells a caller whether to expect the lock to
/// free up shortly.
pub fn cancel(conversation_id: &str) -> usize {
    let reg = lock_live();
    match reg.get(conversation_id) {
        Some(v) => {
            for c in v {
                c.cancel();
            }
            v.len()
        }
        None => 0,
    }
}

// ---- Wire types -----------------------------------------------------------

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub conversation_id: Option<String>,
    pub character_id: Option<String>,
    pub regenerate: bool,
    pub director: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Recalled {
    pub role: String,
    pub content: String,
    pub score: f32,
}

/// What the renderer receives, in order: exactly one `Meta`, then any number of
/// `Chunk`, then exactly one terminator (`Done`, or `Error` followed by `Done`).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ChatEvent {
    Meta {
        #[serde(rename = "conversationId")]
        conversation_id: String,
        #[serde(rename = "characterId")]
        character_id: Option<String>,
        recalled: Vec<Recalled>,
    },
    /// One token's worth of text. Already extracted from the engine's wire
    /// format, so the renderer only appends.
    Chunk {
        delta: String,
    },
    /// The generation failed partway. The stream ends after this.
    Error {
        error: String,
    },
    Done,
}

/// Anything that can take a [`ChatEvent`]. The shell implements it over a Tauri
/// channel; the tests implement it over a `Vec`.
pub trait Sink: Send + Sync {
    fn send(&self, event: ChatEvent);
}

impl<F: Fn(ChatEvent) + Send + Sync> Sink for F {
    fn send(&self, event: ChatEvent) {
        self(event)
    }
}

/// A failure BEFORE any token was produced, so the caller can still report a
/// status rather than a half-written stream.
///
/// `kind` is what the renderer switches on; it replaces the HTTP status the
/// Express version returned, and carries the same distinctions: bad input, a
/// thing that is not there, an engine that cannot be reached, and an engine that
/// answered with a refusal.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatError {
    pub kind: ErrorKind,
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub character_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ErrorKind {
    /// The request cannot produce a turn (empty, junk, nothing to regenerate).
    BadRequest,
    /// A character or a previous reply the request named does not exist.
    NotFound,
    /// The engine is not reachable at all.
    Unavailable,
    /// The engine answered, and said no.
    Refused,
    /// Something below the request failed - the database, the memory subsystem.
    Internal,
}

impl ChatError {
    fn new(kind: ErrorKind, error: impl Into<String>) -> Self {
        Self {
            kind,
            error: error.into(),
            detail: None,
            character_id: None,
        }
    }
    fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.detail {
            Some(d) => write!(f, "{}: {d}", self.error),
            None => f.write_str(&self.error),
        }
    }
}

impl std::error::Error for ChatError {}

// ---- Request validation ---------------------------------------------------

fn has_usable_message(messages: &[Message]) -> bool {
    messages
        .iter()
        .any(|m| matches!(m.role.as_str(), "user" | "assistant") && !m.content.trim().is_empty())
}

fn has_user_message(messages: &[Message]) -> bool {
    messages
        .iter()
        .any(|m| m.role == "user" && !m.content.trim().is_empty())
}

/// What kind of turn this is. Resolving it once, up front, removes the three
/// separate re-derivations the Express handler did further down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// A new user message.
    Normal,
    /// Re-roll the last reply; no new user turn.
    Regenerate,
    /// A director note alone: steer and continue, nothing to record.
    DirectorOnly,
}

/// Ceiling on the text one request may carry.
///
/// The Express build got this from `express.json({ limit: "10mb" })`; over IPC
/// there is no body parser to impose it, so the check moves here. It is not a
/// trust boundary - the renderer is the only caller - it is a bound: without it
/// a pasted novel is written to the database and handed to the engine, and the
/// failure surfaces as an out-of-memory or a stalled model instead of a
/// sentence the user can act on.
const MAX_REQUEST_BYTES: usize = 10 * 1024 * 1024;

fn request_bytes(req: &ChatRequest) -> usize {
    req.messages
        .iter()
        .map(|m| m.content.len() + m.role.len())
        .sum::<usize>()
        + req.director.as_deref().map_or(0, str::len)
}

fn classify(req: &ChatRequest) -> Result<Shape, ChatError> {
    let has_director = req
        .director
        .as_deref()
        .is_some_and(|d| !d.trim().is_empty());

    if request_bytes(req) > MAX_REQUEST_BYTES {
        return Err(ChatError::new(
            ErrorKind::BadRequest,
            "Message is too large. Trim it and send again.",
        ));
    }

    // `messages` may legitimately be absent for a regenerate or a director-only
    // note. Otherwise a real message is required.
    if req.messages.is_empty() && !req.regenerate && !has_director {
        return Err(ChatError::new(
            ErrorKind::BadRequest,
            "Invalid input: \"messages\" must be a non-empty array.",
        ));
    }

    // Reject messages carrying no usable content. Without this such a request
    // falls through to a normal generation on an EXISTING conversation: the junk
    // message is dropped, but the stored history keeps the window non-empty, so
    // the model replies to the previous turn and a SECOND assistant turn is
    // appended with no user turn between them - breaking the alternation the
    // summarizer and the model both depend on. A fresh conversation already
    // errored out; only existing ones leaked.
    if !req.messages.is_empty()
        && !req.regenerate
        && !has_director
        && !has_usable_message(&req.messages)
    {
        return Err(ChatError::new(
            ErrorKind::BadRequest,
            "No valid messages: each must have a known role and non-empty string content.",
        ));
    }

    Ok(if req.regenerate {
        Shape::Regenerate
    } else if has_director && !has_user_message(&req.messages) {
        Shape::DirectorOnly
    } else {
        Shape::Normal
    })
}

// ---- The turn -------------------------------------------------------------

/// Everything assembled before the first token. Split out so the long function
/// below reads as the three phases it actually has.
struct Prepared {
    conversation_id: String,
    character: Option<Character>,
    outbound: Vec<Message>,
    latest_user: Option<Message>,
    retrieved: Vec<Retrieved>,
    stats: Option<crate::metrics::Window>,
}

async fn prepare(
    db: &db::Db,
    req: &ChatRequest,
    shape: Shape,
    conversation_id: &str,
) -> Result<Prepared, ChatError> {
    let internal = |e: anyhow::Error| {
        ChatError::new(
            ErrorKind::Internal,
            "Memory subsystem failed to assemble context.",
        )
        .with_detail(e.to_string())
    };

    // Resolve the character: the explicit id, else the conversation's bound one.
    //
    // A well-formed id that no longer exists must be caught HERE.
    // `conversations.character_id` is a foreign key, so binding a new
    // conversation to a missing character fails the INSERT with a raw "FOREIGN
    // KEY constraint failed". That is reachable in normal use - the character was
    // deleted on another synced device, or in a second window, while this chat
    // was open - so report it as the not-found it is, and let the UI send the
    // user back to the gallery instead of retrying something that can never work.
    let mut char_id = req
        .character_id
        .clone()
        .filter(|c| characters::is_valid_id(c));
    if let Some(id) = char_id.clone() {
        let exists = characters::get(&id).await.map_err(internal)?;
        if exists.is_none() {
            let mut e = ChatError::new(ErrorKind::NotFound, "Character not found.");
            e.character_id = Some(id);
            return Err(e);
        }
    }

    let conv = memory::ensure_conversation(db, conversation_id, char_id.as_deref())
        .await
        .map_err(internal)?;
    if char_id.is_none() {
        char_id = conv.character_id.clone();
    }
    let character = match char_id.as_deref() {
        Some(id) => characters::get(id).await.map_err(internal)?,
        None => None,
    };

    // Persona is stable for the whole conversation and belongs in the cacheable
    // prefix. The director note changes per request, so it joins recall in the
    // volatile tail - see the ordering note on `memory::build_context`.
    let leading: Vec<Message> = prompt::persona_message(character.as_ref())
        .into_iter()
        .collect();
    let trailing: Vec<Message> = prompt::director_message(req.director.as_deref())
        .into_iter()
        .collect();

    // A regenerate builds from the PERSISTED verbatim window: the user message
    // is already there, and passing it again would duplicate it. Director-only
    // likewise has no new turn to add.
    let incoming: &[Message] = match shape {
        Shape::Normal => &req.messages,
        Shape::Regenerate | Shape::DirectorOnly => &[],
    };

    if shape == Shape::Regenerate
        && memory::get_last_assistant_turn(db, conversation_id)
            .await
            .map_err(internal)?
            .is_none()
    {
        return Err(ChatError::new(
            ErrorKind::BadRequest,
            "Nothing to regenerate: this conversation has no previous reply.",
        ));
    }

    let mut built = memory::build_context(db, conversation_id, incoming, &leading, &trailing)
        .await
        .map_err(internal)?;

    // On a regenerate, drop the last assistant turn from the outbound window so
    // the model does not see its own previous reply while re-rolling.
    if shape == Shape::Regenerate {
        if let Some(pos) = built.messages.iter().rposition(|m| m.role == "assistant") {
            built.messages.remove(pos);
        }
    }

    // Nothing but system messages means there is no turn to answer.
    if !built.messages.iter().any(|m| m.role != "system") {
        return Err(ChatError::new(
            ErrorKind::BadRequest,
            if shape == Shape::Regenerate {
                "Nothing to regenerate: this conversation has no previous turn."
            } else {
                "No valid messages: each must have a known role and string content."
            },
        ));
    }

    Ok(Prepared {
        conversation_id: conversation_id.to_string(),
        character,
        outbound: built.messages,
        // A director-only request has no story user turn to record afterwards.
        latest_user: if shape == Shape::DirectorOnly {
            None
        } else {
            built.latest_user
        },
        retrieved: built.retrieved,
        stats: Some(built.stats),
    })
}

/// Per-character sampling overrides, with the reply-length ceiling applied.
///
/// `num_ctx` goes in FIRST so a character's own `sampling.num_ctx` still wins,
/// and is omitted entirely on a backend that fixes the window at launch -
/// sending it there is a silently ignored field.
fn build_options(character: Option<&Character>) -> Map<String, Json> {
    let engine = inference::chat().ok();
    let mut options = Map::new();
    if engine.is_some_and(|e| e.accepts_num_ctx()) {
        options.insert("num_ctx".into(), Json::from(default_num_ctx()));
    }
    options.insert("num_predict".into(), Json::from(default_num_predict()));
    if let Some(c) = character {
        for (k, v) in &c.sampling {
            options.insert(k.clone(), v.clone());
        }
    }
    options
}

/// Run one chat turn, streaming events into `sink`.
///
/// Returns the conversation id on success. An `Err` means nothing was streamed -
/// the caller can report it as a status. Once the first event is sent, every
/// later failure arrives as [`ChatEvent::Error`] instead.
pub async fn run(req: ChatRequest, sink: Arc<dyn Sink>) -> Result<String, ChatError> {
    let shape = classify(&req)?;

    let db = db::get().await.map_err(|e| {
        ChatError::new(ErrorKind::Internal, "Database unavailable.").with_detail(e.to_string())
    })?;

    let conv_id = req
        .conversation_id
        .clone()
        .filter(|id| memory::is_valid_id(id))
        .unwrap_or_else(memory::new_id);

    // Held for the whole turn, INCLUDING the post-stream record. A delete that
    // wants this conversation waits here rather than interleaving with the write.
    let _lock = memory::acquire_lock(&conv_id, memory::config().lock_wait).await;

    let prep = match prepare(&db, &req, shape, &conv_id).await {
        Ok(p) => p,
        Err(e) => {
            // Do not leave behind the empty row `ensure_conversation` may have
            // just created for a chat that then failed.
            let _ = memory::delete_conversation_if_empty(&db, &conv_id).await;
            return Err(e);
        }
    };

    let cancel = Cancel::default();
    register(&conv_id, &cancel);
    // From here every exit must unregister, so the body is wrapped and the
    // cleanup happens once, below.
    let result = stream_and_record(&db, prep, shape, &cancel, sink).await;
    unregister(&conv_id, &cancel);
    result
}

async fn stream_and_record(
    db: &db::Db,
    prep: Prepared,
    shape: Shape,
    cancel: &Cancel,
    sink: Arc<dyn Sink>,
) -> Result<String, ChatError> {
    let conv_id = prep.conversation_id.clone();
    let engine = inference::chat().map_err(|e| {
        ChatError::new(
            ErrorKind::Unavailable,
            "No inference backend is configured.",
        )
        .with_detail(e.to_string())
    })?;
    let options = build_options(prep.character.as_ref());

    let start = tokio::select! {
        biased;
        // A stop that lands before the engine answers must not wait out a model
        // load, which can take tens of seconds.
        _ = cancel.cancelled() => {
            finish_aborted_before_start(db, &prep).await;
            return Ok(conv_id);
        }
        r = engine.chat_stream(prep.outbound.clone(), &options) => r,
    };

    let stream = match start {
        Ok(ChatStart::Streaming(s)) => s,
        Ok(ChatStart::Refused { status, detail }) => {
            let _ = memory::delete_conversation_if_empty(db, &conv_id).await;
            return Err(ChatError::new(
                if status == 404 {
                    ErrorKind::NotFound
                } else {
                    ErrorKind::Refused
                },
                format!("{} returned {status}.", engine.name()),
            )
            .with_detail(detail.to_string()));
        }
        Err(e) => {
            // Nothing generated and nothing recorded: drop the row this request
            // pre-created for a new chat.
            let _ = memory::delete_conversation_if_empty(db, &conv_id).await;
            return Err(ChatError::new(
                ErrorKind::Unavailable,
                format!("Cannot reach {}. Is it running?", engine.name()),
            )
            .with_detail(e.to_string()));
        }
    };

    sink.send(ChatEvent::Meta {
        conversation_id: conv_id.clone(),
        character_id: prep.character.as_ref().map(|c| c.id.clone()),
        recalled: prep
            .retrieved
            .iter()
            .map(|r| Recalled {
                role: r.role.clone(),
                content: r.content.clone(),
                score: r.score,
            })
            .collect(),
    });

    let mut reply = String::new();
    let mut engine_error: Option<String> = None;
    // The final chunk carries prompt_eval_count / eval_count / the durations.
    // Ollama sends it; the llama-server adapter synthesizes the same shape from
    // llama.cpp's `timings` and adds cached_tokens on top.
    let mut done = None;

    let mut stream = stream;
    loop {
        let evt = tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            e = stream.next() => match e { Some(e) => e, None => break },
        };
        match evt {
            StreamEvent::Chunk { delta, .. } => {
                if !delta.is_empty() {
                    reply.push_str(&delta);
                    sink.send(ChatEvent::Chunk { delta });
                }
            }
            StreamEvent::Done { stats, .. } => done = Some(*stats),
            StreamEvent::Error { message } => {
                sink.send(ChatEvent::Error {
                    error: format!("{} error: {message}", engine.name()),
                });
                engine_error = Some(message);
            }
        }
    }
    // Dropping the stream closes the upstream connection, which is what actually
    // stops the engine generating.
    drop(stream);
    sink.send(ChatEvent::Done);

    // Telemetry FIRST: it is pure in-memory bookkeeping and must not be skipped
    // by anything below. A stopped stream carries no done chunk, so an aborted
    // generation records only that it happened.
    metrics::record(metrics::Sample {
        conversation_id: Some(&conv_id),
        character_id: prep.character.as_ref().map(|c| c.id.as_str()),
        model: Some(engine.model()),
        backend: engine.name(),
        done: done.as_ref(),
        window: prep.stats,
        prompt_messages: Some(prep.outbound),
        aborted: done.is_none() && engine_error.is_none(),
    });

    // A director-only message steers behaviour and is never recorded; it still
    // produces a reply that IS recorded as a normal turn. A stream the user
    // stopped is also recorded - the partial reply if any tokens arrived, else
    // just the user's message - so nothing on screen silently vanishes on
    // reload. Only an engine error skips recording.
    let partial = reply.trim();
    let outcome = if engine_error.is_none() && !partial.is_empty() {
        let name = prep
            .character
            .as_ref()
            .map(|c| c.name.as_str())
            .unwrap_or("Character");
        if shape == Shape::Regenerate {
            memory::record_regeneration(db, &conv_id, partial)
                .await
                .map(|_| ())
        } else {
            memory::record_turn(db, &conv_id, prep.latest_user.as_ref(), partial, name)
                .await
                .map(|_| ())
        }
    } else if let (None, Some(user)) = (engine_error.as_ref(), prep.latest_user.as_ref()) {
        memory::record_user_turn(db, &conv_id, user)
            .await
            .map(|_| ())
    } else {
        // Nothing recorded (an engine error, or an aborted director/regenerate
        // with no text) - drop the conversation row if this request created it.
        memory::delete_conversation_if_empty(db, &conv_id)
            .await
            .map(|_| ())
    };
    if let Err(e) = outcome {
        tracing::error!("[memory] record failed for {conv_id}: {e}");
    }

    Ok(conv_id)
}

/// The user stopped before the engine sent anything - a model still loading,
/// usually. Keep their message so it survives a reload.
async fn finish_aborted_before_start(db: &db::Db, prep: &Prepared) {
    let outcome = match prep.latest_user.as_ref() {
        Some(u) => memory::record_user_turn(db, &prep.conversation_id, u)
            .await
            .map(|_| ()),
        None => memory::delete_conversation_if_empty(db, &prep.conversation_id)
            .await
            .map(|_| ()),
    };
    if let Err(e) = outcome {
        tracing::error!(
            "[memory] abort cleanup failed for {}: {e}",
            prep.conversation_id
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            messages,
            ..Default::default()
        }
    }

    #[test]
    fn a_plain_message_is_a_normal_turn() {
        assert_eq!(
            classify(&req(vec![Message::new("user", "hi")])).unwrap(),
            Shape::Normal
        );
    }

    #[test]
    fn an_empty_request_is_refused_unless_it_has_another_job() {
        let e = classify(&req(vec![])).unwrap_err();
        assert_eq!(e.kind, ErrorKind::BadRequest);
        // A regenerate and a director note both legitimately carry no messages.
        assert_eq!(
            classify(&ChatRequest {
                regenerate: true,
                ..Default::default()
            })
            .unwrap(),
            Shape::Regenerate
        );
        assert_eq!(
            classify(&ChatRequest {
                director: Some("be colder".into()),
                ..Default::default()
            })
            .unwrap(),
            Shape::DirectorOnly
        );
    }

    #[test]
    fn messages_that_carry_no_content_are_refused_not_silently_dropped() {
        // Dropping them used to append a second assistant turn with no user turn
        // between, which breaks the alternation the summarizer depends on.
        let e = classify(&req(vec![
            Message::new("user", "   "),
            Message::new("tool", "x"),
        ]))
        .unwrap_err();
        assert_eq!(e.kind, ErrorKind::BadRequest);
        assert!(e.error.contains("No valid messages"));
    }

    #[test]
    fn a_director_note_alongside_a_real_message_is_still_a_normal_turn() {
        let r = ChatRequest {
            messages: vec![Message::new("user", "hi")],
            director: Some("be colder".into()),
            ..Default::default()
        };
        assert_eq!(classify(&r).unwrap(), Shape::Normal);
    }

    #[test]
    fn a_blank_director_note_does_not_count_as_one() {
        let r = ChatRequest {
            director: Some("   ".into()),
            ..Default::default()
        };
        assert_eq!(classify(&r).unwrap_err().kind, ErrorKind::BadRequest);
    }

    #[test]
    fn a_request_larger_than_the_ceiling_is_refused_before_anything_opens() {
        // The Express body limit is gone with the HTTP; this is what replaces it.
        let r = req(vec![Message::new(
            "user",
            "x".repeat(MAX_REQUEST_BYTES + 1),
        )]);
        assert_eq!(classify(&r).unwrap_err().kind, ErrorKind::BadRequest);
    }

    #[test]
    fn the_ceiling_counts_the_whole_request_not_one_message() {
        // Many merely-large messages add up to the same problem as one huge one.
        let half = "x".repeat(MAX_REQUEST_BYTES / 2 + 1);
        let r = req(vec![
            Message::new("user", half.clone()),
            Message::new("user", half),
        ]);
        assert_eq!(classify(&r).unwrap_err().kind, ErrorKind::BadRequest);
    }

    #[test]
    fn a_request_at_the_ceiling_still_goes_through() {
        let mut r = req(vec![Message::new(
            "user",
            "x".repeat(MAX_REQUEST_BYTES - 4),
        )]);
        r.messages[0].role = "user".into();
        assert_eq!(classify(&r).unwrap(), Shape::Normal);
    }

    #[test]
    fn regenerate_wins_over_everything_else() {
        let r = ChatRequest {
            messages: vec![Message::new("user", "hi")],
            regenerate: true,
            director: Some("be colder".into()),
            ..Default::default()
        };
        assert_eq!(classify(&r).unwrap(), Shape::Regenerate);
    }

    #[test]
    fn a_character_sampling_override_beats_the_defaults() {
        let mut c = Character {
            id: "c1".into(),
            name: "Aria".into(),
            avatar: String::new(),
            tagline: String::new(),
            about: String::new(),
            persona: String::new(),
            greeting: String::new(),
            chat_starters: vec![],
            tags: vec![],
            sampling: Map::new(),
            response_style: "balanced".into(),
            created_at: String::new(),
            updated_at: String::new(),
        };
        c.sampling.insert("num_predict".into(), Json::from(2048));
        c.sampling.insert("temperature".into(), Json::from(0.9));
        let opts = build_options(Some(&c));
        assert_eq!(opts["num_predict"], Json::from(2048));
        assert_eq!(opts["temperature"], Json::from(0.9));

        // And the ceiling still applies when the character says nothing.
        let bare = build_options(None);
        assert_eq!(bare["num_predict"], Json::from(default_num_predict()));
    }

    #[tokio::test]
    async fn cancelling_resolves_even_if_it_lands_before_the_wait() {
        // The race this guards: `notify_waiters` drops a signal with nobody
        // parked, which would hang a stream that was stopped the instant it
        // started.
        let c = Cancel::default();
        c.cancel();
        assert!(c.is_cancelled());
        tokio::time::timeout(Duration::from_millis(500), c.cancelled())
            .await
            .expect("a cancel issued before the wait must still wake it");
    }

    #[tokio::test]
    async fn cancelling_a_conversation_signals_every_stream_on_it() {
        let a = Cancel::default();
        let b = Cancel::default();
        register("conv-x", &a);
        register("conv-x", &b);
        // A regenerate overlapping its own turn is the case this covers.
        assert_eq!(cancel("conv-x"), 2);
        assert!(a.is_cancelled() && b.is_cancelled());

        unregister("conv-x", &a);
        unregister("conv-x", &b);
        assert_eq!(cancel("conv-x"), 0, "and the registry does not leak");
        assert!(!lock_live().contains_key("conv-x"));
    }

    #[tokio::test]
    async fn conversations_do_not_cancel_each_others_streams() {
        let mine = Cancel::default();
        let theirs = Cancel::default();
        register("conv-a", &mine);
        register("conv-b", &theirs);
        cancel("conv-a");
        assert!(mine.is_cancelled());
        assert!(
            !theirs.is_cancelled(),
            "a delete must not stop an unrelated chat"
        );
        unregister("conv-a", &mine);
        unregister("conv-b", &theirs);
    }

    #[test]
    fn an_error_reads_as_one_line_with_its_detail() {
        let e =
            ChatError::new(ErrorKind::Unavailable, "Cannot reach ollama.").with_detail("timed out");
        assert_eq!(e.to_string(), "Cannot reach ollama.: timed out");
        let bare = ChatError::new(ErrorKind::BadRequest, "Nope.");
        assert_eq!(bare.to_string(), "Nope.");
    }
}
