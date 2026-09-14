//! End-to-end tests for one chat turn.
//!
//! Unit tests can prove the pieces; only this can prove the ORDER, which is
//! where every bug in the Node version lived: a turn recorded before the stream
//! closed, a conversation row left behind by a request that failed, a stop that
//! lost the user's message. So this runs the real [`chat::run`] against a real
//! database and a real HTTP engine - just not the user's, and not a model.
//!
//! ## Isolation
//!
//! An integration test is its own process, so the environment set in [`setup`]
//! is this process alone and cannot leak into a running app:
//!
//! - `LOCAL_DB_PATH` points at a temp directory. The user's `data/scenario.db`
//!   is never opened, and neither is the plaintext -> encrypted migration.
//! - `VESSEL_NO_SYNC` is set, so nothing is pushed anywhere.
//! - `DB_ENCRYPTION_KEY` is a fixed test key, so the OS keychain is never asked
//!   for the real one.
//! - `OLLAMA_HOST` points at the mock below, so no model is ever loaded.
//!
//! ## The mock engine
//!
//! A hand-written HTTP/1.1 server on a loopback port, ~80 lines, rather than a
//! server framework in dev-dependencies. It speaks the three endpoints the
//! Ollama adapter uses and closes the connection to delimit the body, which is
//! also what makes a cancelled stream observable: when the client drops the
//! socket mid-generation, the write fails and the handler stops.
//!
//! Behaviour is steered by the prompt text, because that is the only part of the
//! request a test controls end to end: `FAIL404` gets a 404 before any stream,
//! `MIDFAIL` gets a chunk and then an error line, `SLOW` gets a chunk and then a
//! long pause.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::time::Duration;

use vessel_core::chat::{self, ChatEvent, ChatRequest, ErrorKind};
use vessel_core::inference::Message;
use vessel_core::{db, memory, metrics};

// ---- The mock engine ------------------------------------------------------

fn body_of(req: &str) -> &str {
    req.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("")
}

/// 768 deterministic, non-zero floats. Zeros would normalise to NaN, and a
/// constant vector would make every retrieval score identical - neither tells us
/// anything about the ranking.
fn fake_embedding(text: &str) -> String {
    let seed = text.len() as f32;
    let values: Vec<String> = (0..768)
        .map(|i| format!("{:.6}", 0.1 + ((i as f32 * 0.37 + seed) % 1.0) * 0.5))
        .collect();
    format!("{{\"embedding\":[{}]}}", values.join(","))
}

fn write_json(sock: &mut TcpStream, status: &str, body: &str) {
    let _ = write!(
        sock,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
}

/// The NDJSON lines the Ollama adapter expects: content chunks, then a final
/// object carrying the telemetry.
fn chat_lines(text: &str) -> Vec<String> {
    let mut lines: Vec<String> = text
        .split_inclusive(' ')
        .map(|w| format!("{{\"message\":{{\"role\":\"assistant\",\"content\":\"{w}\"}},\"done\":false}}"))
        .collect();
    lines.push(
        "{\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":120,\
         \"prompt_eval_duration\":2000000000,\"eval_count\":8,\"eval_duration\":1000000000,\
         \"total_duration\":3000000000,\"load_duration\":0}"
            .to_string(),
    );
    lines
}

fn handle(mut sock: TcpStream) {
    let mut reader = BufReader::new(sock.try_clone().expect("clone socket"));
    let mut head = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        head.push_str(&line);
        if line == "\r\n" {
            break;
        }
    }
    let len: usize = head
        .lines()
        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().ok()))
        .flatten()
        .unwrap_or(0);
    let mut body = vec![0u8; len];
    let _ = reader.read_exact(&mut body);
    let req = format!("{head}\r\n{}", String::from_utf8_lossy(&body));
    let body = body_of(&req).to_string();
    let path = head.split_whitespace().nth(1).unwrap_or("").to_string();

    match path.as_str() {
        "/api/embeddings" => write_json(&mut sock, "200 OK", &fake_embedding(&body)),
        "/api/generate" => {
            write_json(&mut sock, "200 OK", "{\"response\":\"They talked. Then they stopped.\"}")
        }
        "/api/chat" => {
            if body.contains("FAIL404") {
                write_json(&mut sock, "404 Not Found", "{\"error\":\"model not found\"}");
                return;
            }
            let slow = body.contains("SLOW");
            let midfail = body.contains("MIDFAIL");
            let _ = write!(
                sock,
                "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n"
            );
            let _ = sock.flush();
            if midfail {
                // A chunk, then the shape Ollama uses to report a failure that
                // only becomes visible after generation has started.
                for line in [
                    "{\"message\":{\"role\":\"assistant\",\"content\":\"Partial\"},\"done\":false}",
                    "{\"error\":\"an error was encountered while running the model\"}",
                ] {
                    if writeln!(sock, "{line}").is_err() || sock.flush().is_err() {
                        return;
                    }
                }
                return;
            }
            for line in chat_lines("Hello there friend.") {
                if writeln!(sock, "{line}").is_err() || sock.flush().is_err() {
                    return;
                }
                if slow {
                    // Long enough that the test cancels first. A dropped socket
                    // surfaces as a write error on the next line, which is how
                    // this loop learns the generation was stopped.
                    for _ in 0..100 {
                        std::thread::sleep(Duration::from_millis(50));
                        if write!(sock, " ").is_err() || sock.flush().is_err() {
                            return;
                        }
                    }
                }
            }
        }
        _ => write_json(&mut sock, "404 Not Found", "{}"),
    }
}

// ---- Setup ----------------------------------------------------------------

fn temp_root() -> &'static std::path::PathBuf {
    static ROOT: OnceLock<std::path::PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("vessel-chat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    })
}

/// Runs exactly once, before anything reads config. Every value here is read
/// live from the environment by `vessel_core::config`, so setting them first is
/// what makes the rest of this file safe.
fn setup() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock engine");
        let port = listener.local_addr().expect("addr").port();
        std::thread::spawn(move || {
            for sock in listener.incoming().flatten() {
                std::thread::spawn(move || handle(sock));
            }
        });

        let root = temp_root();
        std::env::set_var("LOCAL_DB_PATH", root.join("scenario.db"));
        std::env::set_var("VESSEL_NO_SYNC", "1");
        std::env::set_var("DB_ENCRYPTION_KEY", "0".repeat(64));
        std::env::set_var("INFERENCE_BACKEND", "ollama");
        std::env::set_var("OLLAMA_HOST", format!("http://127.0.0.1:{port}"));
        std::env::set_var("OLLAMA_MODEL", "test-model");
        std::env::set_var("EMBED_MODEL", "test-embed");
        std::env::set_var("SUMMARIZER_MODEL", "test-summarizer");
        // Small enough that a handful of turns crosses the fold threshold.
        std::env::set_var("VERBATIM_TURNS", "2");
        std::env::set_var("SUMMARIZE_THRESHOLD", "4");
    });
}

async fn open() -> Arc<db::Db> {
    setup();
    let db = db::get().await.expect("scratch database");
    assert!(
        db.path().starts_with(temp_root()),
        "refusing to run against {}",
        db.path().display()
    );
    assert!(!db.is_sync_enabled(), "the test database must never sync");
    db
}

/// Collects the stream so a test can assert on the order, not just the text.
#[derive(Default)]
struct Collector(Mutex<Vec<ChatEvent>>);

impl chat::Sink for Collector {
    fn send(&self, event: ChatEvent) {
        self.0.lock().expect("collector").push(event);
    }
}

impl Collector {
    fn text(&self) -> String {
        self.0
            .lock()
            .expect("collector")
            .iter()
            .filter_map(|e| match e {
                ChatEvent::Chunk { delta } => Some(delta.clone()),
                _ => None,
            })
            .collect()
    }
    fn kinds(&self) -> Vec<&'static str> {
        self.0
            .lock()
            .expect("collector")
            .iter()
            .map(|e| match e {
                ChatEvent::Meta { .. } => "meta",
                ChatEvent::Chunk { .. } => "chunk",
                ChatEvent::Error { .. } => "error",
                ChatEvent::Done => "done",
            })
            .collect()
    }
    fn conv_id(&self) -> String {
        self.0
            .lock()
            .expect("collector")
            .iter()
            .find_map(|e| match e {
                ChatEvent::Meta { conversation_id, .. } => Some(conversation_id.clone()),
                _ => None,
            })
            .expect("meta")
    }
    fn errors(&self) -> Vec<String> {
        self.0
            .lock()
            .expect("collector")
            .iter()
            .filter_map(|e| match e {
                ChatEvent::Error { error } => Some(error.clone()),
                _ => None,
            })
            .collect()
    }
}

fn user(text: &str) -> ChatRequest {
    ChatRequest { messages: vec![Message::new("user", text)], ..Default::default() }
}

async fn run(req: ChatRequest) -> (Result<String, chat::ChatError>, Arc<Collector>) {
    let sink = Arc::new(Collector::default());
    let out = chat::run(req, sink.clone()).await;
    (out, sink)
}

// ---- Tests ----------------------------------------------------------------

#[tokio::test]
async fn a_turn_streams_then_records_in_that_order() {
    let db = open().await;
    let (out, sink) = run(user("Hello?")).await;
    let id = out.expect("the turn should succeed");

    let kinds = sink.kinds();
    assert_eq!(kinds.first(), Some(&"meta"), "meta must arrive before any token");
    assert_eq!(kinds.last(), Some(&"done"), "done must be the terminator");
    assert!(kinds.iter().filter(|k| **k == "meta").count() == 1, "exactly one meta");
    assert!(kinds.iter().filter(|k| **k == "done").count() == 1, "exactly one done");
    assert_eq!(sink.text(), "Hello there friend.");

    // Recorded only after the stream closed, which is why it is readable now.
    let conv = memory::get_conversation(&db, &id).await.unwrap().expect("conversation");
    let roles: Vec<&str> = conv.verbatim.iter().map(|t| t.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant"], "one exchange, in order");
    assert_eq!(conv.verbatim[0].content, "Hello?");
    assert_eq!(conv.verbatim[1].content, "Hello there friend.");
}

#[tokio::test]
async fn the_generation_is_measured_even_though_nothing_was_logged() {
    let _db = open().await;
    let (out, _) = run(user("Measure me.")).await;
    let id = out.expect("the turn should succeed");

    let snap = metrics::snapshot(10, Some(&id));
    let recent = snap["recent"].as_array().expect("recent");
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0]["promptTokens"], 120, "the done chunk's numbers are kept");
    assert_eq!(recent[0]["evalTokens"], 8);
    assert_eq!(recent[0]["backend"], "ollama");
    assert_eq!(recent[0]["aborted"], false);
    // The window the prompt was built from travels with the record; without it
    // chars-per-token cannot be computed and every estimate stays a guess.
    assert!(recent[0]["window"]["promptChars"].as_u64().unwrap_or(0) > 0);
}

#[tokio::test]
async fn an_engine_that_refuses_leaves_no_conversation_behind() {
    let db = open().await;
    let id = memory::new_id();
    let req = ChatRequest {
        messages: vec![Message::new("user", "FAIL404 please")],
        conversation_id: Some(id.clone()),
        ..Default::default()
    };
    let (out, sink) = run(req).await;

    let err = out.expect_err("a 404 from the engine is not a successful turn");
    assert_eq!(err.kind, ErrorKind::NotFound);
    assert!(sink.kinds().is_empty(), "nothing may be streamed before the engine answers");
    assert!(
        memory::get_conversation(&db, &id).await.unwrap().is_none(),
        "the row this request created must be swept away"
    );
}

#[tokio::test]
async fn a_stopped_generation_keeps_what_was_written() {
    let db = open().await;
    let id = memory::new_id();
    let req = ChatRequest {
        messages: vec![Message::new("user", "SLOW down")],
        conversation_id: Some(id.clone()),
        ..Default::default()
    };
    let sink = Arc::new(Collector::default());
    let task = tokio::spawn(chat::run(req, sink.clone()));

    // Let the first chunk land, then stop it the way the UI does.
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        if !sink.text().is_empty() {
            break;
        }
    }
    assert!(!sink.text().is_empty(), "the mock should have sent a chunk by now");
    assert_eq!(chat::cancel(&id), 1, "the stream must be registered under its conversation");

    let out = tokio::time::timeout(Duration::from_secs(10), task)
        .await
        .expect("a cancelled turn must not wait out the generation")
        .expect("task");
    assert_eq!(out.expect("a stop is not a failure"), id);
    assert_eq!(sink.kinds().last(), Some(&"done"), "a stopped stream still terminates");

    // Nothing on screen may silently vanish on reload: the partial reply is
    // recorded, not discarded.
    let conv = memory::get_conversation(&db, &id).await.unwrap().expect("conversation");
    assert_eq!(conv.verbatim.len(), 2);
    assert_eq!(conv.verbatim[1].role, "assistant");
    assert!(!conv.verbatim[1].content.is_empty());

    let snap = metrics::snapshot(10, Some(&id));
    assert_eq!(snap["recent"][0]["aborted"], true, "an aborted turn is recorded as one");
}

#[tokio::test]
async fn a_regenerated_reply_becomes_a_second_variant_of_the_same_turn() {
    let db = open().await;
    let (out, _) = run(user("Say something.")).await;
    let id = out.expect("first turn");

    let req = ChatRequest {
        conversation_id: Some(id.clone()),
        regenerate: true,
        ..Default::default()
    };
    let (out, sink) = run(req).await;
    assert_eq!(out.expect("regenerate"), id);
    assert_eq!(sink.text(), "Hello there friend.");

    let conv = memory::get_conversation(&db, &id).await.unwrap().expect("conversation");
    assert_eq!(conv.verbatim.len(), 2, "a re-roll replaces the reply, it does not append one");
    let variants = conv.verbatim[1].variants.as_ref().expect("the latest reply carries variants");
    assert_eq!(variants.len(), 2, "the original and the re-roll");
    assert_eq!(conv.verbatim[1].active_index, Some(1), "the newest is the active one");
}

#[tokio::test]
async fn nothing_to_regenerate_is_refused_rather_than_answered() {
    let _db = open().await;
    let req = ChatRequest {
        conversation_id: Some(memory::new_id()),
        regenerate: true,
        ..Default::default()
    };
    let (out, sink) = run(req).await;
    let err = out.expect_err("an empty conversation has no reply to re-roll");
    assert_eq!(err.kind, ErrorKind::BadRequest);
    assert!(sink.kinds().is_empty());
}

#[tokio::test]
async fn a_director_note_steers_without_becoming_a_turn() {
    let db = open().await;
    let (out, _) = run(user("Begin.")).await;
    let id = out.expect("first turn");

    let req = ChatRequest {
        conversation_id: Some(id.clone()),
        director: Some("be colder".into()),
        ..Default::default()
    };
    let (out, _) = run(req).await;
    assert_eq!(out.expect("director-only turn"), id);

    let conv = memory::get_conversation(&db, &id).await.unwrap().expect("conversation");
    let roles: Vec<&str> = conv.verbatim.iter().map(|t| t.role.as_str()).collect();
    // The note itself is never stored; the reply it produced is.
    assert_eq!(roles, ["user", "assistant", "assistant"]);
    assert!(!conv.verbatim.iter().any(|t| t.content.contains("be colder")));
}

#[tokio::test]
async fn a_request_with_nothing_usable_in_it_never_reaches_the_engine() {
    let db = open().await;
    let id = memory::new_id();
    let req = ChatRequest {
        messages: vec![Message::new("user", "   "), Message::new("tool", "x")],
        conversation_id: Some(id.clone()),
        ..Default::default()
    };
    let (out, sink) = run(req).await;

    let err = out.expect_err("whitespace is not a message");
    assert_eq!(err.kind, ErrorKind::BadRequest);
    assert!(sink.kinds().is_empty());
    assert!(memory::get_conversation(&db, &id).await.unwrap().is_none());
}

#[tokio::test]
async fn a_character_that_no_longer_exists_reports_itself_instead_of_a_foreign_key() {
    let db = open().await;
    let id = memory::new_id();
    let req = ChatRequest {
        messages: vec![Message::new("user", "Hi")],
        conversation_id: Some(id.clone()),
        // Well-formed, so it passes the id check and reaches the existence
        // check - which is the whole point of the test.
        character_id: Some(memory::new_id()),
        ..Default::default()
    };
    let (out, _) = run(req).await;

    let err = out.expect_err("a deleted character cannot be chatted with");
    assert_eq!(err.kind, ErrorKind::NotFound);
    assert!(err.character_id.is_some(), "the UI needs to know WHICH character is gone");
    assert!(memory::get_conversation(&db, &id).await.unwrap().is_none());
}

#[tokio::test]
async fn an_error_after_the_first_token_is_reported_and_nothing_is_recorded() {
    // The hard case: the request succeeded, tokens were streamed, and the model
    // fell over anyway. The text on screen is not a reply, so it must not be
    // saved as one - and the conversation it created must not survive either.
    let db = open().await;
    let id = memory::new_id();
    let req = ChatRequest {
        messages: vec![Message::new("user", "MIDFAIL now")],
        conversation_id: Some(id.clone()),
        ..Default::default()
    };
    let (out, sink) = run(req).await;

    assert_eq!(out.expect("the stream opened, so the call itself succeeded"), sink.conv_id());
    assert_eq!(sink.text(), "Partial", "what arrived before the failure is still shown");
    let errors = sink.errors();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("an error was encountered"), "got {:?}", errors[0]);
    assert_eq!(sink.kinds().last(), Some(&"done"), "the stream still terminates");

    assert!(
        memory::get_conversation(&db, &id).await.unwrap().is_none(),
        "a failed generation leaves no conversation in the sidebar"
    );
}

#[tokio::test]
async fn a_healthy_stream_carries_no_error_event() {
    let _db = open().await;
    let (out, sink) = run(user("Fine.")).await;
    out.expect("turn");
    assert!(sink.errors().is_empty());
}

#[tokio::test]
async fn folding_runs_in_the_background_and_the_summary_lands() {
    let db = open().await;
    let id = memory::new_id();
    // VERBATIM_TURNS=2 and SUMMARIZE_THRESHOLD=4, so a handful of exchanges
    // crosses the threshold and the oldest turns fold into a summary.
    for i in 0..4 {
        let req = ChatRequest {
            messages: vec![Message::new("user", format!("Turn {i}."))],
            conversation_id: Some(id.clone()),
            ..Default::default()
        };
        let (out, _) = run(req).await;
        out.expect("turn");
    }
    memory::await_all_maintenance().await;

    let conv = memory::get_conversation(&db, &id).await.unwrap().expect("conversation");
    assert!(!conv.summary.is_empty(), "the fold should have written a summary");
    assert!(!conv.archive.is_empty(), "and moved the oldest turns to the archive");
    assert!(
        conv.archive.iter().all(|t| t.has_embedding),
        "every archived turn needs a vector, or it can never be recalled"
    );
    assert!(conv.verbatim.len() <= 4, "the verbatim window stays bounded");
}
