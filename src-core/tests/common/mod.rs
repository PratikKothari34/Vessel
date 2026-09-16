//! Shared harness for the integration tests: a scratch database, a mock engine,
//! and a sink that records the stream.
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
//! A hand-written HTTP/1.1 server on a loopback port, ~100 lines, rather than a
//! server framework in dev-dependencies. It speaks the three endpoints the
//! Ollama adapter uses and closes the connection to delimit the body, which is
//! also what makes a cancelled stream observable: when the client drops the
//! socket mid-generation, the write fails and the handler stops.
//!
//! Behaviour is steered by the prompt text, because that is the only part of the
//! request a test controls end to end: `FAIL404` gets a 404 before any stream,
//! `MIDFAIL` gets a chunk and then an error line, `SLOW` gets a chunk and then a
//! long pause. A test that needs to control the reply text itself takes
//! [`steer`] instead, which also serializes those tests against each other.

#![allow(dead_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, MutexGuard, Once, OnceLock};
use std::time::Duration;

use vessel_core::chat::{self, ChatEvent, ChatRequest};
use vessel_core::inference::Message;
use vessel_core::{db, memory};

// ---- Steering -------------------------------------------------------------

/// The reply the mock sends instead of its default, when a test has set one.
fn override_reply() -> &'static Mutex<Option<String>> {
    static R: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(None))
}

/// The last `/api/chat` body the mock received, so a test can assert on what
/// actually reached the engine rather than on what it hoped was sent.
fn last_chat() -> &'static Mutex<String> {
    static L: OnceLock<Mutex<String>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(String::new()))
}

fn steer_lock() -> &'static Mutex<()> {
    static S: OnceLock<Mutex<()>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(()))
}

/// Hold the engine still: this test alone decides the reply, and it alone reads
/// [`Steer::last_prompt`]. Both are process-wide state, so the guard is what
/// keeps two such tests from reading each other's.
pub struct Steer(Option<MutexGuard<'static, ()>>);

pub fn steer(reply: &str) -> Steer {
    let guard = steer_lock().lock().unwrap_or_else(|p| p.into_inner());
    *override_reply().lock().unwrap_or_else(|p| p.into_inner()) = Some(reply.to_string());
    last_chat().lock().unwrap_or_else(|p| p.into_inner()).clear();
    Steer(Some(guard))
}

impl Steer {
    /// The prompt the engine was handed, verbatim JSON.
    pub fn last_prompt(&self) -> String {
        last_chat().lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
}

impl Drop for Steer {
    fn drop(&mut self) {
        *override_reply().lock().unwrap_or_else(|p| p.into_inner()) = None;
        self.0.take();
    }
}

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
///
/// The chunk text is serialized rather than interpolated. A test that feeds the
/// engine's mouth with quotes, newlines or `data:` prefixes - which is the whole
/// point of the framing tests - would otherwise produce invalid JSON and prove
/// only that the mock is broken.
pub const DEFAULT_REPLY: &str = "Hello there friend.";

fn chat_lines(text: &str) -> Vec<String> {
    let mut lines: Vec<String> = text
        .split_inclusive(' ')
        .map(|w| {
            serde_json::json!({ "message": { "role": "assistant", "content": w }, "done": false })
                .to_string()
        })
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
            last_chat().lock().unwrap_or_else(|p| p.into_inner()).clone_from(&body);
            if body.contains("FAIL404") {
                write_json(&mut sock, "404 Not Found", "{\"error\":\"model not found\"}");
                return;
            }
            let slow = body.contains("SLOW");
            let midfail = body.contains("MIDFAIL");
            let reply = override_reply()
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
                .unwrap_or_else(|| DEFAULT_REPLY.to_string());
            let _ = write!(
                sock,
                "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n"
            );
            let _ = sock.flush();
            if body.contains("NODELIM") {
                // A body that never closes a frame: the shape a stale port or a
                // proxy has, and the one case where the reader's buffer is the
                // thing under test rather than what it parses.
                let filler = "x".repeat(64 * 1024);
                for _ in 0..24 {
                    if write!(sock, "{filler}").is_err() || sock.flush().is_err() {
                        return;
                    }
                }
                return;
            }
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
            for line in chat_lines(&reply) {
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

pub fn temp_root() -> &'static std::path::PathBuf {
    static ROOT: OnceLock<std::path::PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let dir = std::env::temp_dir()
            .join(format!("vessel-it-{}-{}", std::process::id(), binary_tag()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    })
}

/// Two test binaries can share a process id across runs but never a name, and
/// they must not share a database file either.
fn binary_tag() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "unknown".into())
}

/// Runs exactly once, before anything reads config. Every value here is read
/// live from the environment by `vessel_core::config`, so setting them first is
/// what makes the rest of these tests safe.
pub fn setup() {
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
        // The summary ships OFF by default (see Config::summary_enabled). These
        // tests exercise the summariser path itself, so they turn it back on
        // explicitly -- config() is a one-shot read, so it cannot be toggled
        // per-test inside one binary. The default's own parsing is covered by
        // the_summary_is_off_unless_asked_for in memory.rs.
        std::env::set_var("SUMMARY_ENABLED", "1");
    });
}

pub async fn open() -> Arc<db::Db> {
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

// ---- The sink -------------------------------------------------------------

/// Collects the stream so a test can assert on the order, not just the text.
#[derive(Default)]
pub struct Collector(Mutex<Vec<ChatEvent>>);

impl chat::Sink for Collector {
    fn send(&self, event: ChatEvent) {
        self.0.lock().expect("collector").push(event);
    }
}

impl Collector {
    pub fn text(&self) -> String {
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
    pub fn kinds(&self) -> Vec<&'static str> {
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
    pub fn conv_id(&self) -> String {
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
    pub fn errors(&self) -> Vec<String> {
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
    /// How many frames of a kind arrived. A forged frame shows up here as a
    /// second `meta` or an `error` nobody sent.
    pub fn count(&self, kind: &str) -> usize {
        self.kinds().iter().filter(|k| **k == kind).count()
    }
}

pub fn user(text: &str) -> ChatRequest {
    ChatRequest { messages: vec![Message::new("user", text)], ..Default::default() }
}

pub async fn run(req: ChatRequest) -> (Result<String, chat::ChatError>, Arc<Collector>) {
    let sink = Arc::new(Collector::default());
    let out = chat::run(req, sink.clone()).await;
    (out, sink)
}

/// A conversation id nothing has used yet.
pub fn new_id() -> String {
    memory::new_id()
}
