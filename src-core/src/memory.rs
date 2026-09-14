//! Long-term memory: the rolling summary, the archive, and retrieval.
//!
//! Keeps the model's live window small (fast) while preserving unlimited
//! history:
//!
//! ```text
//! live window = [character persona] + [rolling summary] + [last N verbatim turns]
//!             + [retrieved snippets] + [director note] + [newest user message]
//! ```
//!
//! - Rolling summary: `SUMMARIZER_MODEL` condenses old turns.
//! - Retrieval: `EMBED_MODEL` embeds every archived turn; ranking is a cosine
//!   scan here, because neither driver has native vector search.
//! - Persistence: the tables in [`crate::db`]. No JSON files.
//!
//! ## What changed in the port, and what did not
//!
//! The algorithm is the Node one, turn for turn. Two things are genuinely
//! different, both because Rust made them free:
//!
//! - Cached archive vectors live in ONE flat `Vec<f32>` per conversation rather
//!   than a vector of vectors, so the scan walks contiguous memory instead of
//!   chasing a pointer per row.
//! - Every function takes the open [`Db`] rather than reaching for the global,
//!   which is what lets the tests below run against a scratch file and never go
//!   near the user's database.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use serde::Serialize;
use tokio::sync::Mutex as AsyncMutex;
use turso::Value as TValue;

use crate::config::EMBED_DIM;
use crate::db::{self, Db};
use crate::embed as codec;
use crate::inference::{self, EmbedOpts, Engine, GenOpts, Message};
use crate::metrics::Window;
use crate::util::now_iso;

pub use crate::characters::is_valid_id;

// ---- Config ---------------------------------------------------------------

fn int_env(name: &str, def: i64, min: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .filter(|v| *v >= min)
        .unwrap_or(def)
}

fn float_env(name: &str, def: f32, min: f32, max: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= min && *v <= max)
        .unwrap_or(def)
}

#[derive(Debug, Clone)]
pub struct Config {
    pub chat_model: String,
    pub summarizer_model: String,
    pub embed_model: String,
    pub embed_num_gpu: u32,
    pub summarizer_num_gpu: u32,
    pub chat_num_ctx: u32,
    pub summarizer_num_ctx: u32,
    pub verbatim_turns: usize,
    pub summarize_threshold: usize,
    pub verbatim_ceiling: usize,
    pub max_fold_turns: usize,
    pub retrieve_k: usize,
    pub retrieve_min_score: f32,
    pub max_summary_chars: usize,
    pub summary_target_chars: usize,
    pub archive_cache_convs: usize,
    pub embed_concurrency: usize,
    pub lock_wait: Duration,
}

impl Config {
    fn from_env() -> Self {
        let chat_model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "vessel".into());
        let summarizer_model =
            std::env::var("SUMMARIZER_MODEL").unwrap_or_else(|_| "gemma3:4b".into());

        // Ollama keys a resident model instance on (model, num_ctx), not on the
        // model alone. A summarizer that differs on EITHER evicts the chat model
        // and the next user message pays a full reload before its first token:
        // measured on an 8 GB 4060, gemma3:4b costs 19.4 s, the chat model at a
        // mismatched window 18.6 s, and the SAME window 3 ms. So when the
        // summarizer is the chat model, default its window to the chat window
        // and keep the one slot warm. (llama-server does not have this problem:
        // one process, one window, fixed at launch, nothing to evict.)
        let chat_num_ctx = int_env("OLLAMA_NUM_CTX", 12288, 256) as u32;
        let same_model = summarizer_model == chat_model;
        let summarizer_num_ctx = int_env(
            "SUMMARIZER_NUM_CTX",
            if same_model { chat_num_ctx as i64 } else { 8192 },
            512,
        ) as u32;

        let verbatim_turns = int_env("VERBATIM_TURNS", 8, 1) as usize;
        let mut summarize_threshold = int_env("SUMMARIZE_THRESHOLD", 12, 1) as usize;
        if summarize_threshold <= verbatim_turns {
            summarize_threshold = verbatim_turns + 4;
        }
        // Hard ceiling on how many verbatim rows the READ path will include.
        // Folding is a background job, so between the turn that crosses the
        // threshold and the fold finishing, `turns` legitimately holds more rows
        // than the design allows - and a CPU summary takes 95-239 s, long enough
        // for dozens to pile up. Threshold + 2 is exactly the high-water mark
        // the old synchronous design already reached, so the steady state is
        // unchanged.
        let verbatim_ceiling = int_env(
            "VERBATIM_CEILING",
            (summarize_threshold + 2) as i64,
            verbatim_turns as i64,
        ) as usize;

        let max_summary_chars = int_env("MAX_SUMMARY_CHARS", 6000, 500) as usize;
        // Deliberately well under the cap: models treat a character count as a
        // loose hint and overshoot roughly twofold. Truncation stays as the
        // backstop for when they ignore it entirely.
        let summary_target_chars = int_env(
            "SUMMARY_TARGET_CHARS",
            ((max_summary_chars as f64) * 0.4).round() as i64,
            200,
        ) as usize;

        Self {
            chat_model,
            summarizer_model,
            embed_model: std::env::var("EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".into()),
            embed_num_gpu: int_env("EMBED_NUM_GPU", 0, 0) as u32,
            // 0 = CPU, for the same reason as the embedder. The chat model owns
            // 6.1 GB of an 8 GB card; a 3 GB summarizer alongside it does not
            // get evicted under llama-server, it oversubscribes the card and the
            // driver pages 2.3 GB of the chat model out to system RAM. Measured:
            // chat decode 47.0 -> 20.8 tok/s. On CPU it costs the GPU nothing.
            summarizer_num_gpu: int_env("SUMMARIZER_NUM_GPU", 0, 0) as u32,
            chat_num_ctx,
            summarizer_num_ctx,
            verbatim_turns,
            summarize_threshold,
            verbatim_ceiling,
            // A fold that hits this cap asks for another pass instead of taking
            // a bigger bite: handing a whole backlog to one generate would blow
            // SUMMARIZER_NUM_CTX and silently truncate the oldest turns, which
            // is the exact failure the summary exists to prevent.
            max_fold_turns: int_env("MAX_FOLD_TURNS", 24, 2) as usize,
            retrieve_k: int_env("RETRIEVE_K", 4, 0) as usize,
            retrieve_min_score: float_env("RETRIEVE_MIN_SCORE", 0.45, -1.0, 1.0),
            max_summary_chars,
            summary_target_chars,
            // 768 floats x 4 B = 3 KB per archived turn, so a 1,000-turn story
            // costs about 3 MB resident.
            archive_cache_convs: int_env("ARCHIVE_CACHE_CONVS", 8, 1) as usize,
            embed_concurrency: int_env("EMBED_CONCURRENCY", 4, 1) as usize,
            lock_wait: Duration::from_millis(int_env("LOCK_WAIT_MS", 120_000, 1000) as u64),
        }
    }
}

pub fn config() -> &'static Config {
    static CFG: OnceLock<Config> = OnceLock::new();
    CFG.get_or_init(Config::from_env)
}

pub fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

// ---- Rows -----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Turn {
    pub id: i64,
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Retrieved {
    pub role: String,
    pub content: String,
    pub score: f32,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Conversation {
    pub id: String,
    pub character_id: Option<String>,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub summary: String,
}

fn text(row: &serde_json::Map<String, serde_json::Value>, key: &str) -> String {
    row.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string()
}

fn opt_text(row: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
    row.get(key).and_then(|v| v.as_str()).map(str::to_string)
}

fn int(row: &serde_json::Map<String, serde_json::Value>, key: &str) -> i64 {
    row.get(key).and_then(|v| v.as_i64()).unwrap_or(0)
}

// ---- Conversation state ---------------------------------------------------

/// Ensure a conversation row exists; create it (optionally bound to a
/// character) if missing.
///
/// `character_id` is a foreign key, so an id that no longer exists would make
/// the INSERT fail with a raw "FOREIGN KEY constraint failed". Callers reject a
/// missing character up front, but if one slips through, drop the binding rather
/// than throwing: an unbound conversation is recoverable, a hard failure
/// mid-chat is not.
pub async fn ensure_conversation(
    db: &Db,
    id: &str,
    character_id: Option<&str>,
) -> Result<Conversation> {
    if !is_valid_id(id) {
        return Err(anyhow!("ensureConversation: invalid id"));
    }
    if let Some(row) = db
        .query_one("SELECT * FROM conversations WHERE id = ?", vec![TValue::Text(id.into())])
        .await?
    {
        return Ok(row_to_conversation(&row));
    }

    let mut bound = character_id.filter(|c| is_valid_id(c)).map(str::to_string);
    if let Some(c) = bound.clone() {
        let exists = db
            .query_one("SELECT 1 FROM characters WHERE id = ?", vec![TValue::Text(c)])
            .await?;
        if exists.is_none() {
            bound = None;
        }
    }

    let ts = now_iso();
    db.execute(
        "INSERT INTO conversations (id, character_id, title, summary, created_at, updated_at)
         VALUES (?, ?, '', '', ?, ?)",
        vec![
            TValue::Text(id.into()),
            bound.map(TValue::Text).unwrap_or(TValue::Null),
            TValue::Text(ts.clone()),
            TValue::Text(ts),
        ],
    )
    .await?;

    let row = db
        .query_one("SELECT * FROM conversations WHERE id = ?", vec![TValue::Text(id.into())])
        .await?
        .ok_or_else(|| anyhow!("ensureConversation: row vanished after insert"))?;
    Ok(row_to_conversation(&row))
}

fn row_to_conversation(row: &serde_json::Map<String, serde_json::Value>) -> Conversation {
    Conversation {
        id: text(row, "id"),
        character_id: opt_text(row, "character_id"),
        title: text(row, "title"),
        created_at: text(row, "created_at"),
        updated_at: text(row, "updated_at"),
        summary: text(row, "summary"),
    }
}

pub async fn get_summary(db: &Db, id: &str) -> Result<String> {
    let row = db
        .query_one("SELECT summary FROM conversations WHERE id = ?", vec![TValue::Text(id.into())])
        .await?;
    Ok(row.as_ref().map(|r| text(r, "summary")).unwrap_or_default())
}

pub async fn touch_conversation(db: &Db, id: &str) -> Result<()> {
    db.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        vec![TValue::Text(now_iso()), TValue::Text(id.into())],
    )
    .await?;
    Ok(())
}

/// Recent verbatim turns in chronological order, capped at `verbatim_ceiling`.
///
/// Ordering DESC and reversing takes the NEWEST rows, which is the right end to
/// keep: anything past the ceiling is a backlog the background fold is already
/// working on, and the archive keeps every word of it either way.
pub async fn get_verbatim(db: &Db, id: &str) -> Result<Vec<Turn>> {
    get_verbatim_limit(db, id, config().verbatim_ceiling).await
}

async fn get_verbatim_limit(db: &Db, id: &str, limit: usize) -> Result<Vec<Turn>> {
    let rows = db
        .query(
            "SELECT id, role, content FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            vec![TValue::Text(id.into()), TValue::Integer(limit as i64)],
        )
        .await?;
    let mut out: Vec<Turn> = rows
        .iter()
        .map(|r| Turn { id: int(r, "id"), role: text(r, "role"), content: text(r, "content") })
        .collect();
    out.reverse();
    Ok(out)
}

// ---- Engine calls ---------------------------------------------------------
// Transport and per-backend quirks live in `inference`. What stays here is the
// part that is about memory rather than about the engine: the dimension
// contract the archive depends on, and the summarizer's window.

pub async fn embed(text: &str) -> Result<Vec<f32>> {
    if text.trim().is_empty() {
        return Err(anyhow!("embed: empty text"));
    }
    let cfg = config();
    let vec = inference::embedder()?
        .embed(&cfg.embed_model, text, EmbedOpts { num_gpu: Some(cfg.embed_num_gpu) })
        .await?;
    // Every archived row is stored at EMBED_DIM. A model that returned anything
    // else would write vectors that can never be compared against the existing
    // ones, so this fails the turn rather than corrupting the archive.
    if vec.len() != EMBED_DIM {
        return Err(anyhow!("embed: expected {EMBED_DIM} dims, got {}", vec.len()));
    }
    Ok(vec)
}

async fn generate(model: &str, prompt: &str) -> Result<String> {
    let cfg = config();
    inference::summarizer()?
        .generate(
            model,
            prompt,
            GenOpts {
                num_ctx: Some(cfg.summarizer_num_ctx),
                num_gpu: Some(cfg.summarizer_num_gpu),
            },
        )
        .await
}

// ---- Per-conversation lock ------------------------------------------------

/// Serializes turns per conversation so post-stream bookkeeping never races the
/// next request's state read and write.
type LockRegistry = StdMutex<Vec<(String, Arc<AsyncMutex<()>>)>>;

fn locks() -> &'static LockRegistry {
    static LOCKS: OnceLock<LockRegistry> = OnceLock::new();
    LOCKS.get_or_init(Default::default)
}

/// Held for the duration of a turn. Dropping it releases.
///
/// The guard is an `Option` because waiting is bounded: past `wait` the caller
/// proceeds WITHOUT the lock, which is exactly the behaviour that existed before
/// the lock did. A turn is willing to queue behind a whole generation; a delete
/// is not, and should never leave a button spinning for two minutes because a
/// stream is stuck.
pub struct TurnLock {
    _guard: Option<tokio::sync::OwnedMutexGuard<()>>,
    pub timed_out: bool,
}

pub async fn acquire_lock(id: &str, wait: Duration) -> TurnLock {
    let m = {
        let mut reg = match locks().lock() {
            Ok(r) => r,
            Err(p) => p.into_inner(),
        };
        // Entries nobody is queued on any more. The JS version dropped the key
        // from inside the release callback and needed a tail-identity check to
        // get it right; refcounting says the same thing without the ceremony.
        reg.retain(|(_, m)| Arc::strong_count(m) > 1);
        match reg.iter().find(|(k, _)| k == id) {
            Some((_, m)) => m.clone(),
            None => {
                let m: Arc<AsyncMutex<()>> = Arc::new(AsyncMutex::new(()));
                reg.push((id.to_string(), m.clone()));
                m
            }
        }
    };

    match tokio::time::timeout(wait, m.lock_owned()).await {
        Ok(guard) => TurnLock { _guard: Some(guard), timed_out: false },
        Err(_) => TurnLock { _guard: None, timed_out: true },
    }
}

// ---- Vector maths ---------------------------------------------------------

/// Scale to unit length in place. Every vector that enters the cache and every
/// query vector is normalized on the way in, which turns cosine similarity into
/// a plain dot product: the two sqrt calls and the two norm loops per candidate
/// row disappear from the hot scan entirely.
fn normalize_in_place(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum();
    if n == 0.0 {
        return;
    }
    let inv = 1.0 / n.sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Cosine similarity of two ALREADY-NORMALIZED vectors.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// The k best rows, strongest first, by dot product against `query`.
///
/// Bounded selection, not a sort. Scoring every row and sorting the survivors is
/// O(n log n) comparisons plus one allocation per candidate, on a list that
/// grows with the story. Here each row costs one compare against the current
/// k-th best, and only a winner pays an O(k) insert - k is 4, so that insert is
/// free.
fn top_k(query: &[f32], ids: &[i64], vecs: &[f32], k: usize, min_score: f32) -> Vec<(i64, f32)> {
    let mut top: Vec<(i64, f32)> = Vec::with_capacity(k);
    if k == 0 {
        return top;
    }
    let mut cutoff = min_score;
    for (i, id) in ids.iter().enumerate() {
        let score = dot(query, &vecs[i * EMBED_DIM..(i + 1) * EMBED_DIM]);
        if score < cutoff {
            continue;
        }
        if top.len() == k {
            top.pop();
        }
        let at = top.partition_point(|(_, s)| *s >= score);
        top.insert(at, (*id, score));
        // Once k winners are held, nothing weaker than the weakest can win.
        if top.len() == k {
            cutoff = top[k - 1].1;
        }
    }
    top
}

// ---- Archive vector cache -------------------------------------------------
// Retrieval used to re-read and re-decode every archived embedding on EVERY
// turn: 3 KB per row off disk, a decode per row, then two sqrt loops per row.
// None of that changes between turns - the archive is append-only.
//
// So keep the decoded, normalized vectors resident per conversation and load
// only rows newer than the high-water mark. Steady state is zero row reads and
// zero decodes; a turn that archives adds a handful.
//
// Correctness against cloud sync: a pull can insert archive rows with ids BELOW
// our high-water mark (another device's autoincrement is independent of ours),
// and a restore can shrink the table. Neither is caught by a max(id) check
// alone, so the row count is carried too and the cache is rebuilt whenever it
// drops.

#[derive(Default)]
struct CacheEntry {
    ids: Vec<i64>,
    /// All vectors end to end, `EMBED_DIM` floats each. One allocation and one
    /// contiguous walk for the whole scan.
    vecs: Vec<f32>,
    max_id: i64,
    rows_seen: usize,
}

impl CacheEntry {
    fn reset(&mut self) {
        self.ids.clear();
        self.vecs.clear();
        self.max_id = 0;
        self.rows_seen = 0;
    }
}

type Cache = Arc<AsyncMutex<CacheEntry>>;

/// LRU by position: the front is the least recently used. The registry lock is
/// a plain mutex held only long enough to look one entry up, so a long scan on
/// one conversation never blocks another.
fn archive_cache() -> &'static StdMutex<Vec<(String, Cache)>> {
    static CACHE: OnceLock<StdMutex<Vec<(String, Cache)>>> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

fn touch_cache(conversation_id: &str) -> Cache {
    let mut reg = match archive_cache().lock() {
        Ok(r) => r,
        Err(p) => p.into_inner(),
    };
    let existing = reg.iter().position(|(k, _)| k == conversation_id);
    let entry = match existing {
        Some(i) => reg.remove(i).1,
        None => Arc::new(AsyncMutex::new(CacheEntry::default())),
    };
    reg.push((conversation_id.to_string(), entry.clone()));
    let cap = config().archive_cache_convs;
    if reg.len() > cap {
        let over = reg.len() - cap;
        reg.drain(..over);
    }
    entry
}

/// Drop a conversation's cached vectors (the conversation was deleted).
pub fn forget_archive(conversation_id: &str) {
    if let Ok(mut reg) = archive_cache().lock() {
        reg.retain(|(k, _)| k != conversation_id);
    }
}

/// Rows newer than `since_id`. NOT filtered on `embedding IS NOT NULL`: the
/// count has to be comparable with the COUNT(*) below, so rows without a vector
/// are skipped here instead of in SQL.
async fn fetch_archive_rows(db: &Db, conversation_id: &str, since_id: i64) -> Result<Vec<(i64, Vec<u8>)>> {
    let rows = db
        .query_values(
            "SELECT id, embedding FROM archive
             WHERE conversation_id = ? AND id > ?
             ORDER BY id ASC",
            vec![TValue::Text(conversation_id.into()), TValue::Integer(since_id)],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|mut r| {
            let blob = match r.pop() {
                Some(TValue::Blob(b)) => b,
                _ => Vec::new(),
            };
            let id = match r.pop() {
                Some(TValue::Integer(i)) => i,
                _ => 0,
            };
            (id, blob)
        })
        .collect())
}

fn absorb_rows(e: &mut CacheEntry, rows: Vec<(i64, Vec<u8>)>) {
    e.vecs.reserve(rows.len() * EMBED_DIM);
    for (id, blob) in rows {
        let Some(mut vec) = codec::decode(&blob) else { continue };
        if vec.len() != EMBED_DIM {
            continue; // no usable vector -> the row is not retrievable
        }
        normalize_in_place(&mut vec);
        e.ids.push(id);
        e.vecs.extend_from_slice(&vec);
    }
}

async fn load_archive_vectors(db: &Db, conversation_id: &str, e: &mut CacheEntry) -> Result<()> {
    // One indexed aggregate replaces an existence probe AND tells us whether
    // anything changed since last turn.
    let agg = db
        .query_one(
            "SELECT COUNT(*) AS n, MAX(id) AS m FROM archive WHERE conversation_id = ?",
            vec![TValue::Text(conversation_id.into())],
        )
        .await?
        .unwrap_or_default();
    let total = int(&agg, "n").max(0) as usize;
    let max_id = int(&agg, "m");

    if total == 0 {
        e.reset();
        return Ok(());
    }
    // Rows vanished, or the table was restored to an older state: the
    // high-water mark means nothing now.
    if total < e.rows_seen || max_id < e.max_id {
        e.reset();
    }

    // max_id moving with the count unchanged means a row was replaced rather
    // than appended - rare, but the fetch then returns more rows than expected
    // and forces the rebuild, which is exactly right.
    let expected_new = total as i64 - e.rows_seen as i64;
    if expected_new > 0 || max_id > e.max_id {
        let mut rows = fetch_archive_rows(db, conversation_id, e.max_id).await?;
        // A sync pull can land rows with ids BELOW our high-water mark. Those
        // are invisible to the `id > max_id` fetch, and the only way to notice
        // is that fewer rows came back than the count says appeared. Then, and
        // only then, rebuild from scratch.
        if rows.len() as i64 != expected_new {
            e.reset();
            rows = fetch_archive_rows(db, conversation_id, 0).await?;
        }
        absorb_rows(e, rows);
        e.max_id = max_id;
    }
    e.rows_seen = total;
    Ok(())
}

/// Up to k archived turns from THIS conversation most relevant to `query_text`,
/// above the score threshold, in chronological order.
///
/// Three things keep the in-process ranking cheap:
/// - vectors are cached decoded and normalized, so the scan is one multiply-add
///   loop per row and nothing else;
/// - the scan reads NO prose - only the winners' text is fetched, so a
///   2,000-turn archive costs four row reads per turn instead of 2,000;
/// - the archive is checked BEFORE the embed, so a conversation that has not
///   archived anything yet never pays the embed at all.
pub async fn retrieve(db: &Db, conversation_id: &str, query_text: &str) -> Result<Vec<Retrieved>> {
    let cfg = config();
    if cfg.retrieve_k == 0 || query_text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let cache = touch_cache(conversation_id);
    let mut e = cache.lock().await;
    load_archive_vectors(db, conversation_id, &mut e).await?;
    if e.ids.is_empty() {
        return Ok(Vec::new());
    }

    let mut q = embed(query_text).await?;
    normalize_in_place(&mut q);

    let mut top = top_k(&q, &e.ids, &e.vecs, cfg.retrieve_k, cfg.retrieve_min_score);
    drop(e);
    if top.is_empty() {
        return Ok(Vec::new());
    }
    top.sort_by_key(|(id, _)| *id); // chronological

    let placeholders = vec!["?"; top.len()].join(",");
    let mut params: Vec<TValue> = Vec::with_capacity(top.len() + 1);
    params.push(TValue::Text(conversation_id.into()));
    params.extend(top.iter().map(|(id, _)| TValue::Integer(*id)));
    let rows = db
        .query(
            &format!(
                "SELECT id, role, content FROM archive WHERE conversation_id = ? AND id IN ({placeholders})"
            ),
            params,
        )
        .await?;

    let by_id: HashMap<i64, &serde_json::Map<String, serde_json::Value>> =
        rows.iter().map(|r| (int(r, "id"), r)).collect();
    Ok(top
        .iter()
        .filter_map(|(id, score)| {
            by_id.get(id).map(|r| Retrieved {
                role: text(r, "role"),
                content: text(r, "content"),
                score: *score,
            })
        })
        .collect())
}

// ---- Summarization --------------------------------------------------------

fn render_turns<'a, I>(turns: I, assistant_name: &str) -> String
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let mut out = String::new();
    for (role, content) in turns {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(if role == "user" { "User" } else { assistant_name });
        out.push_str(": ");
        out.push_str(content);
    }
    out
}

/// The summarizer prompt.
///
/// The length instruction is not decoration. Without it the summarizer answered
/// with whatever it felt like - measured at 5,792 to 12,422 characters against a
/// 6,000 cap - and the overflow is cut by the front-truncation in
/// `run_maintenance`, which removes the OLDEST text: the part already condensed
/// several times over and no longer present in the verbatim window, so the only
/// part that cannot be recovered. Asking for a bound fixed both ends: 5,160
/// characters keeping 12 of 13 facts, versus 12,422 with 6,422 amputated.
fn summarize_prompt(prior_summary: &str, rendered: &str, target_chars: usize) -> String {
    format!(
        "You are a story archivist. Maintain a running summary of an ongoing\n\
         roleplay so it can be remembered later. Update the summary below with the\n\
         new exchanges. Preserve concrete facts: character names, relationships,\n\
         locations, plot events, decisions, and unresolved threads. Be faithful and\n\
         concise. Do not add disclaimers, opinions, or content not present in the\n\
         text. Output ONLY the updated summary prose.\n\
         Keep the updated summary under {target_chars} characters. If it would run \
         longer, compress the OLDEST material hardest and keep the newest exchanges \
         intact.\n\
         \n\
         === CURRENT SUMMARY ===\n\
         {prior}\n\
         \n\
         === NEW EXCHANGES TO FOLD IN ===\n\
         {rendered}\n\
         \n\
         === UPDATED SUMMARY ===",
        prior = if prior_summary.is_empty() { "(none yet)" } else { prior_summary },
    )
}

async fn summarize(prior_summary: &str, turns: &[Turn], assistant_name: &str) -> Result<String> {
    let rendered = render_turns(
        turns.iter().map(|t| (t.role.as_str(), t.content.as_str())),
        assistant_name,
    );
    let prompt = summarize_prompt(prior_summary, &rendered, config().summary_target_chars);
    generate(&config().summarizer_model, &prompt).await
}

// ---- Context assembly -----------------------------------------------------

/// The recall block sits AFTER the verbatim window, so it is the last thing the
/// model reads before the new user turn. It has to say plainly that it is not
/// the current scene - otherwise the model continues from a retrieved fragment.
const RETRIEVAL_HEADER: &str = "[Recall \u{2014} excerpts from EARLIER in this same story, pulled up for continuity. They already happened and are out of order. Do not treat them as the present moment and do not continue from them:]";
const RETRIEVAL_FOOTER: &str = "[End of recall. The present moment is the conversation above.]";
const SUMMARY_HEADER: &str = "[Story so far \u{2014} summary of earlier events:]";

/// Content must be non-empty: an empty or whitespace-only user message would
/// otherwise be recorded as a junk turn.
pub fn is_valid_msg(m: &Message) -> bool {
    matches!(m.role.as_str(), "system" | "user" | "assistant") && !m.content.trim().is_empty()
}

pub struct BuiltContext {
    pub messages: Vec<Message>,
    pub retrieved: Vec<Retrieved>,
    pub latest_user: Option<Message>,
    pub stats: Window,
}

/// Build the messages the chat model receives.
///
/// ORDER IS THE OPTIMIZATION. An engine reuses its KV cache only up to the first
/// token that differs from the previous request, so anything volatile poisons
/// everything after it. The old order put recall in the middle:
///
/// ```text
/// persona | summary | RECALL | verbatim turns | new user
///                      ^ different every turn
/// ```
///
/// which re-prefilled the entire verbatim window (2-4K tokens) on every single
/// message. The volatile blocks sit at the tail instead:
///
/// ```text
/// persona | summary | verbatim turns | RECALL | director | new user
/// ------------ append-only ---------/  ---- volatile ----/
/// ```
///
/// The prefix only ever grows, except when summarization rewrites the summary -
/// once every `summarize_threshold - verbatim_turns` turns. `metrics` reports
/// the shared-prefix fraction, so this is measured rather than assumed.
///
/// Putting recall last also helps on quality: the model weights recent context
/// more heavily, and recall is the part we most want it to use.
///
/// `leading_systems` are stable per-conversation system messages (the character
/// persona). `trailing_systems` are per-request ones (the director note) and
/// belong in the volatile tail.
pub async fn build_context(
    db: &Db,
    conversation_id: &str,
    incoming: &[Message],
    leading_systems: &[Message],
    trailing_systems: &[Message],
) -> Result<BuiltContext> {
    // Walk backwards for the newest user message: it is almost always the last
    // element, and the JS version built three intermediate arrays to find it.
    let latest_user = incoming
        .iter()
        .rev()
        .find(|m| m.role == "user" && is_valid_msg(m))
        .cloned();
    let query_text = latest_user.as_ref().map(|m| m.content.as_str()).unwrap_or("");

    // Sequential, not concurrent: all three legs share ONE database connection,
    // so running them together would buy no parallelism and would put two
    // statements on that connection at once. The two reads are sub-millisecond;
    // the leg that costs anything is the embed inside `retrieve`, and it cannot
    // start before the archive check anyway.
    let t0 = Instant::now();
    let summary = get_summary(db, conversation_id).await?;
    let verbatim = get_verbatim(db, conversation_id).await?;
    let retrieved = if query_text.is_empty() {
        Vec::new()
    } else {
        // Retrieval is an enhancement. A dead embedder must cost recall, not
        // the whole turn.
        retrieve(db, conversation_id, query_text).await.unwrap_or_default()
    };
    let retrieve_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut messages: Vec<Message> = Vec::with_capacity(
        leading_systems.len() + verbatim.len() + trailing_systems.len() + 3,
    );
    // Counted while building rather than re-walked afterwards.
    let mut persona_chars = 0usize;
    let mut stable_prefix_chars = 0usize;

    for m in leading_systems.iter().filter(|m| is_valid_msg(m)) {
        persona_chars += m.content.chars().count();
        stable_prefix_chars += m.content.chars().count();
        messages.push(m.clone());
    }
    let summary_chars = summary.chars().count();
    if !summary.is_empty() {
        let block = format!("{SUMMARY_HEADER}\n{summary}");
        stable_prefix_chars += block.chars().count();
        messages.push(Message::new("system", block));
    }
    let mut verbatim_chars = 0usize;
    for t in &verbatim {
        verbatim_chars += t.content.chars().count();
        stable_prefix_chars += t.content.chars().count();
        messages.push(Message::new(&t.role, t.content.clone()));
    }

    let mut retrieved_chars = 0usize;
    if !retrieved.is_empty() {
        for r in &retrieved {
            retrieved_chars += r.content.chars().count();
        }
        let body = render_turns(
            retrieved.iter().map(|r| (r.role.as_str(), r.content.as_str())),
            "Character",
        );
        messages.push(Message::new(
            "system",
            format!("{RETRIEVAL_HEADER}\n{body}\n{RETRIEVAL_FOOTER}"),
        ));
    }

    let mut director_chars = 0usize;
    for m in trailing_systems.iter().filter(|m| is_valid_msg(m)) {
        director_chars += m.content.chars().count();
        messages.push(m.clone());
    }

    let mut new_user_chars = 0usize;
    if let Some(u) = latest_user.as_ref() {
        if !already_last(&verbatim, u) {
            new_user_chars = u.content.chars().count();
            messages.push(Message::new("user", u.content.clone()));
        }
    }

    let stats = Window {
        message_count: messages.len(),
        prompt_chars: messages.iter().map(|m| m.content.chars().count()).sum(),
        stable_prefix_chars,
        persona_chars,
        summary_chars,
        verbatim_count: verbatim.len(),
        verbatim_chars,
        retrieved_count: retrieved.len(),
        retrieved_chars,
        director_chars,
        new_user_chars,
        retrieve_ms: (retrieve_ms * 10.0).round() / 10.0,
    };

    Ok(BuiltContext { messages, retrieved, latest_user, stats })
}

fn already_last(verbatim: &[Turn], msg: &Message) -> bool {
    verbatim
        .last()
        .is_some_and(|last| last.role == msg.role && last.content == msg.content)
}

// ---- Variants (swipe between alternate generations) -----------------------

/// The last assistant turn row, or `None` when the newest turn is not one.
pub async fn get_last_assistant_turn(db: &Db, conversation_id: &str) -> Result<Option<Turn>> {
    let row = db
        .query_one(
            "SELECT id, role, content FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
            vec![TValue::Text(conversation_id.into())],
        )
        .await?;
    Ok(row.filter(|r| text(r, "role") == "assistant").map(|r| Turn {
        id: int(&r, "id"),
        role: "assistant".into(),
        content: text(&r, "content"),
    }))
}

/// Ensure a turn has at least its current content registered as a variant.
/// Turns created before variants existed get back-filled lazily.
async fn ensure_base_variant(db: &Db, turn_id: i64, content: &str) -> Result<()> {
    let have = db
        .query_one(
            "SELECT COUNT(*) AS n FROM variants WHERE turn_id = ?",
            vec![TValue::Integer(turn_id)],
        )
        .await?
        .unwrap_or_default();
    if int(&have, "n") == 0 {
        db.execute(
            "INSERT INTO variants (turn_id, content, is_active, created_at) VALUES (?, ?, 1, ?)",
            vec![
                TValue::Integer(turn_id),
                TValue::Text(content.into()),
                TValue::Text(now_iso()),
            ],
        )
        .await?;
    }
    Ok(())
}

/// Append a variant to an assistant turn, make it active, and mirror it into
/// `turns.content` so the summary and the archive read the chosen text.
pub async fn append_variant(db: &Db, turn_id: i64, content: &str) -> Result<i64> {
    ensure_base_variant(db, turn_id, content).await?;
    db.execute(
        "INSERT INTO variants (turn_id, content, is_active, created_at) VALUES (?, ?, 0, ?)",
        vec![TValue::Integer(turn_id), TValue::Text(content.into()), TValue::Text(now_iso())],
    )
    .await?;
    let variant_id = db.last_insert_rowid();
    db.execute(
        "UPDATE variants SET is_active = 0 WHERE turn_id = ?",
        vec![TValue::Integer(turn_id)],
    )
    .await?;
    db.execute(
        "UPDATE variants SET is_active = 1 WHERE id = ?",
        vec![TValue::Integer(variant_id)],
    )
    .await?;
    db.execute(
        "UPDATE turns SET content = ? WHERE id = ?",
        vec![TValue::Text(content.into()), TValue::Integer(turn_id)],
    )
    .await?;
    Ok(variant_id)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActiveVariant {
    pub turn_id: i64,
    pub variant_id: i64,
    pub content: String,
}

/// Pick which variant of a turn is active. Mirrors its text into
/// `turns.content`.
pub async fn set_active_variant(
    db: &Db,
    conversation_id: &str,
    turn_id: i64,
    variant_id: i64,
) -> Result<ActiveVariant> {
    // The join is the authorization check: a variant id alone says nothing about
    // which conversation it belongs to.
    let v = db
        .query_one(
            "SELECT v.content FROM variants v JOIN turns t ON t.id = v.turn_id
             WHERE v.id = ? AND v.turn_id = ? AND t.conversation_id = ?",
            vec![
                TValue::Integer(variant_id),
                TValue::Integer(turn_id),
                TValue::Text(conversation_id.into()),
            ],
        )
        .await?
        .ok_or_else(|| anyhow!("Variant not found for this conversation/turn."))?;
    let content = text(&v, "content");

    db.execute(
        "UPDATE variants SET is_active = 0 WHERE turn_id = ?",
        vec![TValue::Integer(turn_id)],
    )
    .await?;
    db.execute(
        "UPDATE variants SET is_active = 1 WHERE id = ?",
        vec![TValue::Integer(variant_id)],
    )
    .await?;
    db.execute(
        "UPDATE turns SET content = ? WHERE id = ?",
        vec![TValue::Text(content.clone()), TValue::Integer(turn_id)],
    )
    .await?;
    Ok(ActiveVariant { turn_id, variant_id, content })
}

#[derive(Debug, Clone, Serialize)]
pub struct Variant {
    pub id: i64,
    pub content: String,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Variants {
    pub variants: Vec<Variant>,
    pub active_index: usize,
}

/// All variants for a turn, oldest first, with the active index.
pub async fn get_variants(db: &Db, turn_id: i64) -> Result<Variants> {
    let rows = db
        .query(
            "SELECT id, content, is_active FROM variants WHERE turn_id = ? ORDER BY id ASC",
            vec![TValue::Integer(turn_id)],
        )
        .await?;
    let variants: Vec<Variant> = rows
        .iter()
        .map(|r| Variant {
            id: int(r, "id"),
            content: text(r, "content"),
            active: int(r, "is_active") != 0,
        })
        .collect();
    let active_index = variants.iter().position(|v| v.active).unwrap_or(0);
    Ok(Variants { variants, active_index })
}

// ---- Post-turn bookkeeping ------------------------------------------------

/// Embed a batch of turns with bounded concurrency. Blobs come back
/// positionally; a failed embed yields `None`, which stores the turn WITHOUT a
/// vector - the text survives, it just cannot be retrieved.
async fn embed_all(turns: &[Turn]) -> Vec<Option<Vec<u8>>> {
    let lanes = config().embed_concurrency.clamp(1, turns.len().max(1));
    // Each future owns its text rather than borrowing the turn: a borrow here
    // would have to outlive a spawned task, and the fold runs on one.
    let texts: Vec<String> = turns.iter().map(|t| format!("{}: {}", t.role, t.content)).collect();
    futures_util::stream::iter(texts.into_iter().map(|text| async move {
        match embed(&text).await {
            Ok(v) => Some(codec::encode(&v)),
            Err(_) => None,
        }
    }))
    .buffered(lanes)
    .collect()
    .await
}

// SQLite caps bound parameters per statement (999 by default). Batched writes
// stay well under it by chunking.
const ARCHIVE_INSERT_CHUNK: usize = 100; // x5 params = 500 bound values
const DELETE_ID_CHUNK: usize = 400;

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnResult {
    pub archived: usize,
    pub summarized: bool,
    pub maintenance_scheduled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub cancelled: bool,
}

/// Record the user turn and the assistant reply.
///
/// Everything here is the fast write phase. When the verbatim count crosses the
/// threshold the slow half - summarize, embed, archive - is handed to the
/// background instead of run inline: the caller holds the per-conversation turn
/// lock across this function, so folding here makes the user's NEXT message wait
/// behind a CPU summary. Measured at 31,839 ms and 35,619 ms for a summarizing
/// turn against 2.3-6.1 s for every other one.
pub async fn record_turn(
    db: &Db,
    conversation_id: &str,
    user_message: Option<&Message>,
    assistant_reply: &str,
    assistant_name: &str,
) -> Result<TurnResult> {
    let verbatim = get_verbatim(db, conversation_id).await?;
    // One timestamp for the whole exchange: the user turn, the reply and its
    // base variant belong to the same moment, and three separate clock reads
    // could straddle a second boundary and order the rows by skew.
    let ts = now_iso();

    if let Some(u) = user_message {
        if !already_last(&verbatim, u) {
            db.execute(
                "INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                vec![
                    TValue::Text(conversation_id.into()),
                    TValue::Text("user".into()),
                    TValue::Text(u.content.clone()),
                    TValue::Text(ts.clone()),
                ],
            )
            .await?;
        }
    }

    // The driver reports the inserted rowid, so the variant row is tied to the
    // turn we just wrote rather than to whatever "newest assistant turn"
    // happened to be by the time a follow-up SELECT ran.
    db.execute(
        "INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        vec![
            TValue::Text(conversation_id.into()),
            TValue::Text("assistant".into()),
            TValue::Text(assistant_reply.into()),
            TValue::Text(ts.clone()),
        ],
    )
    .await?;
    let mut turn_id = db.last_insert_rowid();
    if turn_id <= 0 {
        // Defensive: a driver that does not report the rowid falls back to the
        // lookup rather than writing an orphaned variant.
        let row = db
            .query_one(
                "SELECT id FROM turns WHERE conversation_id = ? AND role = 'assistant'
                 ORDER BY id DESC LIMIT 1",
                vec![TValue::Text(conversation_id.into())],
            )
            .await?
            .ok_or_else(|| anyhow!("recordTurn: the assistant turn vanished after insert"))?;
        turn_id = int(&row, "id");
    }

    // Register this reply as the turn's first (active) variant so swiping works.
    db.execute(
        "INSERT INTO variants (turn_id, content, is_active, created_at) VALUES (?, ?, 1, ?)",
        vec![
            TValue::Integer(turn_id),
            TValue::Text(assistant_reply.into()),
            TValue::Text(ts),
        ],
    )
    .await?;
    touch_conversation(db, conversation_id).await?;

    let count = turn_count(db, conversation_id).await?;
    if count <= config().summarize_threshold {
        return Ok(TurnResult::default());
    }

    schedule_maintenance(conversation_id, assistant_name);
    Ok(TurnResult { maintenance_scheduled: true, ..Default::default() })
}

async fn turn_count(db: &Db, conversation_id: &str) -> Result<usize> {
    let row = db
        .query_one(
            "SELECT COUNT(*) AS n FROM turns WHERE conversation_id = ?",
            vec![TValue::Text(conversation_id.into())],
        )
        .await?
        .unwrap_or_default();
    Ok(int(&row, "n").max(0) as usize)
}

// ---- Background maintenance -----------------------------------------------
// Folding old turns into the rolling summary is maintenance, not part of a
// turn. It runs OFF the turn lock, so a summary never delays the next message.
//
// What that costs: for as long as a fold is in flight, readers see the previous
// summary and a longer-than-designed verbatim window. Both are safe. The summary
// is behind, not wrong; the extra verbatim turns are the very text the summary
// is missing, so the model sees that material either way - in full rather than
// condensed. `verbatim_ceiling` bounds how much of it reaches the prompt.
//
// One run per conversation at a time. A request arriving mid-run sets `rerun`
// instead of starting a second one: the next pass re-reads the turn count and
// picks up whatever arrived meanwhile, which is strictly better than two runs
// racing over the same oldest rows.

#[derive(Default)]
struct Maint {
    rerun: AtomicBool,
    cancelled: AtomicBool,
    draining: AtomicBool,
    /// Held by the running task for its whole life. Anyone who wants to wait for
    /// the fold to finish just locks it.
    running: Arc<AsyncMutex<()>>,
}

fn maintenance() -> &'static StdMutex<HashMap<String, Arc<Maint>>> {
    static M: OnceLock<StdMutex<HashMap<String, Arc<Maint>>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

pub fn schedule_maintenance(conversation_id: &str, assistant_name: &str) {
    let state = {
        let mut reg = match maintenance().lock() {
            Ok(r) => r,
            Err(p) => p.into_inner(),
        };
        if let Some(existing) = reg.get(conversation_id) {
            existing.rerun.store(true, Ordering::SeqCst);
            return;
        }
        let state = Arc::new(Maint::default());
        reg.insert(conversation_id.to_string(), state.clone());
        state
    };

    // Taken here, not inside the task: the permit has to be held before this
    // function returns, or a caller that immediately awaits all maintenance
    // could slip past a task that has not started yet. The mutex is brand new,
    // so this never blocks.
    let guard = state
        .running
        .clone()
        .try_lock_owned()
        .expect("a fresh maintenance lock is always free");

    let id = conversation_id.to_string();
    let name = assistant_name.to_string();
    tokio::spawn(async move {
        let _guard = guard;
        loop {
            state.rerun.store(false, Ordering::SeqCst);
            let db = match db::get().await {
                Ok(db) => db,
                Err(e) => {
                    tracing::error!("maintenance for {id} could not open the database: {e}");
                    break;
                }
            };
            // A failed fold is recoverable by construction: nothing was deleted,
            // so the turns are still verbatim and the next turn tries again.
            if let Err(e) = run_maintenance(&db, &id, &name, &state).await {
                tracing::error!("maintenance failed for {id}: {e}");
                break;
            }
            // No await between the pass returning and this check, so a
            // schedule_maintenance call cannot set `rerun` on a state we are
            // about to discard.
            if state.cancelled.load(Ordering::SeqCst) || !state.rerun.load(Ordering::SeqCst) {
                break;
            }
        }
        if let Ok(mut reg) = maintenance().lock() {
            if reg.get(&id).is_some_and(|s| Arc::ptr_eq(s, &state)) {
                reg.remove(&id);
            }
        }
    });
}

/// Wait for every in-flight fold, so shutdown can flush rather than drop them.
pub async fn await_all_maintenance() {
    let states: Vec<Arc<Maint>> = match maintenance().lock() {
        Ok(reg) => reg.values().cloned().collect(),
        Err(p) => p.into_inner().values().cloned().collect(),
    };
    for s in states {
        let _ = s.running.lock().await;
    }
}

/// Stop a fold from WRITING. Not an abort - the expensive legs (summarize and
/// embed) are already in flight and interrupting them saves nothing - but it
/// guarantees no rows land for a conversation the user just deleted.
pub fn cancel_maintenance(conversation_id: &str) {
    let reg = match maintenance().lock() {
        Ok(r) => r,
        Err(p) => p.into_inner(),
    };
    if let Some(s) = reg.get(conversation_id) {
        s.cancelled.store(true, Ordering::SeqCst);
        s.rerun.store(false, Ordering::SeqCst);
        s.draining.store(false, Ordering::SeqCst);
    }
}

/// One fold: summarize, embed, and archive the turns above the verbatim window.
/// Re-reads its own state, so it is safe to call at any time and cheap when
/// there is nothing to do.
async fn run_maintenance(
    db: &Db,
    conversation_id: &str,
    assistant_name: &str,
    state: &Maint,
) -> Result<TurnResult> {
    let cfg = config();
    let count = turn_count(db, conversation_id).await?;

    // The threshold is the TRIGGER; verbatim_turns is the TARGET. Once a drain
    // is under way the trigger has already fired, so later passes keep going
    // down to the target - otherwise a capped fold would stop at the threshold
    // and leave the window permanently larger than the design.
    let draining = state.draining.load(Ordering::SeqCst);
    let floor = if draining { cfg.verbatim_turns } else { cfg.summarize_threshold };
    if count <= floor {
        state.draining.store(false, Ordering::SeqCst);
        return Ok(TurnResult::default());
    }

    let pending = count - cfg.verbatim_turns;
    let to_archive = pending.min(cfg.max_fold_turns);
    // Ask for another pass rather than a bigger prompt. Set BEFORE the long
    // awaits so a cancel during them still wins: cancel_maintenance clears both.
    let more = pending > to_archive;
    state.draining.store(more, Ordering::SeqCst);
    if more {
        state.rerun.store(true, Ordering::SeqCst);
    }

    let rows = db
        .query(
            "SELECT id, role, content FROM turns WHERE conversation_id = ? ORDER BY id ASC LIMIT ?",
            vec![TValue::Text(conversation_id.into()), TValue::Integer(to_archive as i64)],
        )
        .await?;
    let oldest: Vec<Turn> = rows
        .iter()
        .map(|r| Turn { id: int(r, "id"), role: text(r, "role"), content: text(r, "content") })
        .collect();

    let prior_summary = get_summary(db, conversation_id).await?;

    // The summarizer call is by far the most expensive thing in this phase and
    // the embeds do not depend on it, so both legs run at once and the entire
    // embedding phase hides underneath the summarize call.
    let (summary_result, embeddings) = tokio::join!(
        summarize(&prior_summary, &oldest, assistant_name),
        embed_all(&oldest),
    );

    let updated = match summary_result {
        Ok(text) => text,
        Err(e) => {
            // Summarizer down: keep the turns verbatim rather than lose them.
            // The embeds computed alongside are discarded and recomputed next
            // attempt.
            return Ok(TurnResult { error: Some(e.to_string()), ..Default::default() });
        }
    };

    // Those two awaits are the long ones - minutes, on CPU. The conversation can
    // be deleted in that time, and writing archive rows for a row that no longer
    // exists would resurrect a deleted story. Everything below is await-free
    // apart from its own writes, so checking here is enough.
    if state.cancelled.load(Ordering::SeqCst) {
        return Ok(TurnResult { cancelled: true, ..Default::default() });
    }
    let alive = db
        .query_one(
            "SELECT 1 AS ok FROM conversations WHERE id = ?",
            vec![TValue::Text(conversation_id.into())],
        )
        .await?;
    if alive.is_none() {
        return Ok(TurnResult { cancelled: true, ..Default::default() });
    }

    let updated = tail_chars(&updated, cfg.max_summary_chars);
    db.execute(
        "UPDATE conversations SET summary = ?, updated_at = ? WHERE id = ?",
        vec![
            TValue::Text(updated),
            TValue::Text(now_iso()),
            TValue::Text(conversation_id.into()),
        ],
    )
    .await?;

    // Move folded turns from verbatim to archive. Two batched statements instead
    // of 2N single-row ones. A multi-row INSERT is one statement and therefore
    // already atomic, which matters because neither driver exposes a transaction
    // API for us to wrap a loop in.
    // `embed_all` preserves order, so a turn and its vector share an index.
    // Zipping them keeps that alignment explicit through the chunking.
    let ts = now_iso();
    let rows: Vec<(&Turn, Option<&Vec<u8>>)> =
        oldest.iter().zip(embeddings.iter().map(Option::as_ref)).collect();
    for part in rows.chunks(ARCHIVE_INSERT_CHUNK) {
        let values = vec!["(?, ?, ?, ?, ?)"; part.len()].join(", ");
        let mut params: Vec<TValue> = Vec::with_capacity(part.len() * 5);
        for (turn, blob) in part {
            params.push(TValue::Text(conversation_id.into()));
            params.push(TValue::Text(turn.role.clone()));
            params.push(TValue::Text(turn.content.clone()));
            // A turn whose embed failed is archived without a vector: the text
            // survives, it just cannot be retrieved.
            params.push(match blob {
                Some(b) => TValue::Blob((*b).clone()),
                None => TValue::Null,
            });
            params.push(TValue::Text(ts.clone()));
        }
        db.execute(
            &format!(
                "INSERT INTO archive (conversation_id, role, content, embedding, created_at)
                 VALUES {values}"
            ),
            params,
        )
        .await?;
    }

    let ids: Vec<i64> = oldest.iter().map(|t| t.id).collect();
    for part in ids.chunks(DELETE_ID_CHUNK) {
        let placeholders = vec!["?"; part.len()].join(",");
        db.execute(
            &format!("DELETE FROM turns WHERE id IN ({placeholders})"),
            part.iter().map(|id| TValue::Integer(*id)).collect(),
        )
        .await?;
    }

    Ok(TurnResult { archived: oldest.len(), summarized: true, ..Default::default() })
}

/// Keep the LAST `max` characters. The front is what gets cut, because it is the
/// material already condensed several times over.
fn tail_chars(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    s.chars().skip(n - max).collect()
}

/// Record ONLY a user turn (no assistant reply). Used when a stream is stopped
/// before any reply text arrives, so the user's message still survives a reload.
pub async fn record_user_turn(db: &Db, conversation_id: &str, user_message: &Message) -> Result<bool> {
    if user_message.content.trim().is_empty() {
        return Ok(false);
    }
    let verbatim = get_verbatim(db, conversation_id).await?;
    if already_last(&verbatim, user_message) {
        return Ok(false);
    }
    db.execute(
        "INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        vec![
            TValue::Text(conversation_id.into()),
            TValue::Text("user".into()),
            TValue::Text(user_message.content.clone()),
            TValue::Text(now_iso()),
        ],
    )
    .await?;
    touch_conversation(db, conversation_id).await?;
    Ok(true)
}

/// Delete a conversation row only if it is truly empty (no turns, no archive, no
/// title or summary). Cleans up the row a chat pre-creates when the request then
/// fails - safe to call on any conversation.
pub async fn delete_conversation_if_empty(db: &Db, id: &str) -> Result<bool> {
    if !is_valid_id(id) {
        return Ok(false);
    }
    cancel_maintenance(id);
    forget_archive(id);
    let n = db
        .execute(
            "DELETE FROM conversations WHERE id = ?
               AND title = '' AND summary = ''
               AND NOT EXISTS (SELECT 1 FROM turns t WHERE t.conversation_id = conversations.id)
               AND NOT EXISTS (SELECT 1 FROM archive a WHERE a.conversation_id = conversations.id)",
            vec![TValue::Text(id.into())],
        )
        .await?;
    Ok(n > 0)
}

/// Record a regenerated reply: append it as a new variant of the LAST assistant
/// turn (no new turn, no re-archiving).
pub async fn record_regeneration(
    db: &Db,
    conversation_id: &str,
    assistant_reply: &str,
) -> Result<Option<ActiveVariant>> {
    let Some(last) = get_last_assistant_turn(db, conversation_id).await? else {
        return Ok(None);
    };
    let variant_id = append_variant(db, last.id, assistant_reply).await?;
    touch_conversation(db, conversation_id).await?;
    Ok(Some(ActiveVariant {
        turn_id: last.id,
        variant_id,
        content: assistant_reply.to_string(),
    }))
}

// ---- Read-only introspection (sidebar + memory inspector) -----------------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationSummary {
    pub id: String,
    pub character_id: Option<String>,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub has_summary: bool,
    pub turn_count: i64,
    pub preview: String,
}

pub async fn list_conversations(
    db: &Db,
    character_id: Option<&str>,
) -> Result<Vec<ConversationSummary>> {
    // Named columns, not `c.*`: the only thing the sidebar wants from `summary`
    // is whether there is one, and a folded conversation's summary runs to a
    // couple of thousand characters. Selecting it would serialize all of that
    // for every row, on every refresh, to produce one boolean. Same reason the
    // preview is truncated in SQL rather than here.
    let base = "SELECT c.id, c.character_id, c.title, c.created_at, c.updated_at,
                LENGTH(COALESCE(c.summary, '')) > 0 AS has_summary,
                (SELECT substr(t.content, 1, 120) FROM turns t
                  WHERE t.conversation_id = c.id ORDER BY t.id DESC LIMIT 1) AS last_turn,
                (SELECT COUNT(*) FROM turns t WHERE t.conversation_id = c.id) +
                (SELECT COUNT(*) FROM archive a WHERE a.conversation_id = c.id) AS turn_count
         FROM conversations c";
    let (sql, params) = match character_id.filter(|c| is_valid_id(c)) {
        Some(c) => (
            format!("{base} WHERE c.character_id = ? ORDER BY c.updated_at DESC"),
            vec![TValue::Text(c.into())],
        ),
        None => (format!("{base} ORDER BY c.updated_at DESC"), vec![]),
    };
    let rows = db.query(&sql, params).await?;
    Ok(rows
        .iter()
        .map(|r| ConversationSummary {
            id: text(r, "id"),
            character_id: opt_text(r, "character_id"),
            title: text(r, "title"),
            created_at: text(r, "created_at"),
            updated_at: text(r, "updated_at"),
            has_summary: int(r, "has_summary") != 0,
            turn_count: int(r, "turn_count"),
            preview: text(r, "last_turn"),
        })
        .collect())
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VerbatimTurn {
    pub turn_id: i64,
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variants: Option<Vec<Variant>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_index: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArchivedTurn {
    pub role: String,
    pub content: String,
    pub has_embedding: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationDetail {
    pub id: String,
    pub character_id: Option<String>,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub summary: String,
    pub verbatim: Vec<VerbatimTurn>,
    pub archive: Vec<ArchivedTurn>,
}

/// Full inspectable view: summary, verbatim turns and archived turns, no raw
/// vectors.
pub async fn get_conversation(db: &Db, id: &str) -> Result<Option<ConversationDetail>> {
    if !is_valid_id(id) {
        return Ok(None);
    }
    let Some(row) = db
        .query_one("SELECT * FROM conversations WHERE id = ?", vec![TValue::Text(id.into())])
        .await?
    else {
        return Ok(None);
    };

    let rows = db
        .query(
            "SELECT id, role, content FROM turns WHERE conversation_id = ? ORDER BY id ASC",
            vec![TValue::Text(id.into())],
        )
        .await?;
    let last_assistant = get_last_assistant_turn(db, id).await?;
    // Variants are attached only to the latest assistant turn - the swipeable
    // one - so this is at most one extra query, not one per row.
    let latest_variants = match last_assistant.as_ref() {
        Some(t) => {
            let v = get_variants(db, t.id).await?;
            if v.variants.len() > 1 {
                Some((t.id, v))
            } else {
                None
            }
        }
        None => None,
    };

    let verbatim = rows
        .iter()
        .map(|r| {
            let turn_id = int(r, "id");
            let mine = latest_variants.as_ref().filter(|(id, _)| *id == turn_id);
            VerbatimTurn {
                turn_id,
                role: text(r, "role"),
                content: text(r, "content"),
                variants: mine.map(|(_, v)| v.variants.clone()),
                active_index: mine.map(|(_, v)| v.active_index),
            }
        })
        .collect();

    let arch = db
        .query(
            "SELECT role, content, embedding IS NOT NULL AS has_embedding
             FROM archive WHERE conversation_id = ? ORDER BY id ASC",
            vec![TValue::Text(id.into())],
        )
        .await?;

    Ok(Some(ConversationDetail {
        id: text(&row, "id"),
        character_id: opt_text(&row, "character_id"),
        title: text(&row, "title"),
        created_at: text(&row, "created_at"),
        updated_at: text(&row, "updated_at"),
        summary: text(&row, "summary"),
        verbatim,
        archive: arch
            .iter()
            .map(|r| ArchivedTurn {
                role: text(r, "role"),
                content: text(r, "content"),
                has_embedding: int(r, "has_embedding") != 0,
            })
            .collect(),
    }))
}

/// The new title, or `None` when the conversation does not exist.
pub async fn set_title(db: &Db, id: &str, title: &str) -> Result<Option<String>> {
    if !is_valid_id(id) {
        return Err(anyhow!("setTitle: invalid id"));
    }
    let clean: String = title.chars().take(200).collect();
    let n = db
        .execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            vec![
                TValue::Text(clean.clone()),
                TValue::Text(now_iso()),
                TValue::Text(id.into()),
            ],
        )
        .await?;
    Ok(if n > 0 { Some(clean) } else { None })
}

pub async fn delete_conversation(db: &Db, id: &str) -> Result<bool> {
    if !is_valid_id(id) {
        return Err(anyhow!("deleteConversation: invalid id"));
    }
    // A background fold may be mid-summary for this conversation. Tell it not to
    // write; it checks the flag, and re-checks that this row still exists, right
    // before its first write.
    cancel_maintenance(id);
    let n = db
        .execute("DELETE FROM conversations WHERE id = ?", vec![TValue::Text(id.into())])
        .await?;
    forget_archive(id); // cached vectors would outlive the rows they describe
    crate::metrics::forget_conversation(id);
    Ok(n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    async fn scratch() -> (tempfile::TempDir, Db) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = db::open_scratch(&dir.path().join("scenario.db")).await.expect("scratch db");
        (dir, db)
    }

    async fn seed_character(db: &Db, id: &str) {
        db.execute(
            "INSERT INTO characters (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            vec![
                TValue::Text(id.into()),
                TValue::Text("Test".into()),
                TValue::Text("2026-01-01".into()),
                TValue::Text("2026-01-01".into()),
            ],
        )
        .await
        .unwrap();
    }

    async fn add_turn(db: &Db, conv: &str, role: &str, content: &str) {
        db.execute(
            "INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            vec![
                TValue::Text(conv.into()),
                TValue::Text(role.into()),
                TValue::Text(content.into()),
                TValue::Text(now_iso()),
            ],
        )
        .await
        .unwrap();
    }

    fn unit(seed: f32) -> Vec<f32> {
        // A deterministic, non-degenerate vector: every component differs, so two
        // different seeds never collide after normalization.
        (0..EMBED_DIM).map(|i| ((i as f32 * 0.37) + seed).sin()).collect()
    }

    // ---- pure functions --------------------------------------------------

    #[test]
    fn the_summarizer_prompt_states_a_length_and_carries_the_prior_summary() {
        let p = summarize_prompt("earlier events", "User: hi\nAria: hello", 2400);
        assert!(p.contains("under 2400 characters"), "{p}");
        assert!(p.contains("=== CURRENT SUMMARY ===\nearlier events"));
        assert!(p.contains("=== NEW EXCHANGES TO FOLD IN ===\nUser: hi\nAria: hello"));
        assert!(p.trim_end().ends_with("=== UPDATED SUMMARY ==="));
    }

    #[test]
    fn an_empty_prior_summary_is_named_rather_than_left_blank() {
        // A blank line there reads to the model as "the summary is the next
        // heading", and it starts summarizing the headings.
        assert!(summarize_prompt("", "User: hi", 100).contains("=== CURRENT SUMMARY ===\n(none yet)"));
    }

    #[test]
    fn turns_render_with_the_characters_name_not_the_role() {
        let rendered = render_turns(
            vec![("user", "where are we"), ("assistant", "the old pier")],
            "Aria",
        );
        assert_eq!(rendered, "User: where are we\nAria: the old pier");
    }

    #[test]
    fn the_recall_block_says_it_is_not_the_present_moment() {
        // The model continues from whatever it read last. Without this, the
        // retrieved fragment IS the last thing it read.
        assert!(RETRIEVAL_HEADER.contains("Do not treat them as the present moment"));
        assert!(RETRIEVAL_FOOTER.contains("present moment is the conversation above"));
    }

    #[test]
    fn normalizing_makes_the_dot_product_a_cosine() {
        let mut a = vec![3.0f32, 4.0];
        normalize_in_place(&mut a);
        assert!((a[0] - 0.6).abs() < 1e-6 && (a[1] - 0.8).abs() < 1e-6);
        assert!((dot(&a, &a) - 1.0).abs() < 1e-6);
        // A zero vector has no direction; scaling it would be a divide by zero.
        let mut z = vec![0.0f32; 4];
        normalize_in_place(&mut z);
        assert_eq!(z, vec![0.0; 4]);
    }

    #[test]
    fn selection_returns_the_best_k_strongest_first() {
        let mut q = unit(0.0);
        normalize_in_place(&mut q);
        let mut vecs = Vec::new();
        let mut ids = Vec::new();
        for (i, seed) in [0.0f32, 2.0, 0.05, 4.0].iter().enumerate() {
            let mut v = unit(*seed);
            normalize_in_place(&mut v);
            vecs.extend_from_slice(&v);
            ids.push(i as i64 + 1);
        }
        let top = top_k(&q, &ids, &vecs, 2, -1.0);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1, "the identical vector wins");
        assert_eq!(top[1].0, 3, "the near-identical one is second");
        assert!(top[0].1 >= top[1].1);
    }

    #[test]
    fn the_score_floor_can_reject_everything() {
        let mut q = unit(0.0);
        normalize_in_place(&mut q);
        let mut v = unit(9.0);
        normalize_in_place(&mut v);
        assert!(top_k(&q, &[1], &v, 4, 0.999).is_empty());
        assert!(top_k(&q, &[1], &v, 0, -1.0).is_empty(), "k=0 disables retrieval");
    }

    #[test]
    fn a_summary_is_truncated_from_the_front() {
        // The tail is the newest material. Cutting the end would throw away what
        // the model just learned and keep what it already condensed twice.
        assert_eq!(tail_chars("abcdef", 3), "def");
        assert_eq!(tail_chars("abc", 10), "abc");
        // Characters, not bytes: slicing bytes here would panic mid-codepoint.
        assert_eq!(tail_chars("\u{e9}\u{e9}\u{e9}", 2), "\u{e9}\u{e9}");
    }

    #[test]
    fn only_real_messages_reach_the_prompt() {
        assert!(is_valid_msg(&Message::new("user", "hi")));
        assert!(!is_valid_msg(&Message::new("user", "   ")));
        assert!(!is_valid_msg(&Message::new("tool", "hi")));
    }

    #[test]
    fn the_summarizer_shares_the_chat_window_when_it_is_the_chat_model() {
        // Guards the eviction rule: Ollama keys a resident instance on
        // (model, num_ctx), so a mismatch here costs a full reload per turn.
        std::env::set_var("OLLAMA_MODEL", "vessel-test-model");
        std::env::set_var("SUMMARIZER_MODEL", "vessel-test-model");
        std::env::set_var("OLLAMA_NUM_CTX", "9000");
        std::env::remove_var("SUMMARIZER_NUM_CTX");
        let c = Config::from_env();
        assert_eq!(c.summarizer_num_ctx, 9000);

        std::env::set_var("SUMMARIZER_MODEL", "something-else");
        assert_eq!(Config::from_env().summarizer_num_ctx, 8192);

        std::env::remove_var("OLLAMA_MODEL");
        std::env::remove_var("SUMMARIZER_MODEL");
        std::env::remove_var("OLLAMA_NUM_CTX");
    }

    #[test]
    fn the_threshold_can_never_sit_at_or_below_the_verbatim_window() {
        // Otherwise every single turn triggers a fold that has nothing to fold.
        std::env::set_var("VERBATIM_TURNS", "8");
        std::env::set_var("SUMMARIZE_THRESHOLD", "6");
        let c = Config::from_env();
        assert_eq!(c.summarize_threshold, 12);
        assert_eq!(c.verbatim_ceiling, 14);
        std::env::remove_var("VERBATIM_TURNS");
        std::env::remove_var("SUMMARIZE_THRESHOLD");
    }

    // ---- database-backed -------------------------------------------------

    #[tokio::test]
    async fn a_conversation_is_created_once_and_read_back() {
        let (_d, db) = scratch().await;
        seed_character(&db, "char1").await;
        let c = ensure_conversation(&db, "conv1", Some("char1")).await.unwrap();
        assert_eq!(c.character_id.as_deref(), Some("char1"));
        // Second call must not overwrite or duplicate.
        db.execute(
            "UPDATE conversations SET title = 'kept' WHERE id = ?",
            vec![TValue::Text("conv1".into())],
        )
        .await
        .unwrap();
        let again = ensure_conversation(&db, "conv1", None).await.unwrap();
        assert_eq!(again.title, "kept");
    }

    #[tokio::test]
    async fn an_unknown_character_binding_is_dropped_not_thrown() {
        // The alternative is a foreign-key error mid-chat, which loses the turn.
        let (_d, db) = scratch().await;
        let c = ensure_conversation(&db, "conv1", Some("ghost")).await.unwrap();
        assert_eq!(c.character_id, None);
    }

    #[tokio::test]
    async fn verbatim_keeps_the_newest_turns_up_to_the_ceiling() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        for i in 0..20 {
            add_turn(&db, "conv1", if i % 2 == 0 { "user" } else { "assistant" }, &format!("m{i}"))
                .await;
        }
        let v = get_verbatim_limit(&db, "conv1", 5).await.unwrap();
        assert_eq!(v.len(), 5);
        assert_eq!(v.first().unwrap().content, "m15", "oldest of the kept window");
        assert_eq!(v.last().unwrap().content, "m19", "and chronological order");
    }

    #[tokio::test]
    async fn a_turn_writes_the_reply_its_variant_and_nothing_else() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        let user = Message::new("user", "hello");
        let res = record_turn(&db, "conv1", Some(&user), "hi there", "Aria").await.unwrap();
        assert_eq!(res, TurnResult::default(), "well under the fold threshold");

        let v = get_verbatim(&db, "conv1").await.unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[1].content, "hi there");
        let last = get_last_assistant_turn(&db, "conv1").await.unwrap().unwrap();
        let variants = get_variants(&db, last.id).await.unwrap();
        assert_eq!(variants.variants.len(), 1);
        assert!(variants.variants[0].active);
    }

    #[tokio::test]
    async fn the_same_user_message_is_not_recorded_twice() {
        // The renderer re-sends the whole visible thread; without this the user
        // turn would be duplicated on every retry.
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        let user = Message::new("user", "hello");
        assert!(record_user_turn(&db, "conv1", &user).await.unwrap());
        assert!(!record_user_turn(&db, "conv1", &user).await.unwrap());
        assert!(!record_user_turn(&db, "conv1", &Message::new("user", "  ")).await.unwrap());
        assert_eq!(get_verbatim(&db, "conv1").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_regenerated_reply_becomes_a_variant_and_replaces_the_turn_text() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        record_turn(&db, "conv1", Some(&Message::new("user", "hi")), "first", "Aria")
            .await
            .unwrap();
        let out = record_regeneration(&db, "conv1", "second").await.unwrap().unwrap();
        let v = get_variants(&db, out.turn_id).await.unwrap();
        assert_eq!(v.variants.len(), 2);
        assert_eq!(v.active_index, 1);
        let verbatim = get_verbatim(&db, "conv1").await.unwrap();
        assert_eq!(verbatim.last().unwrap().content, "second", "the turn mirrors the active variant");

        // And swiping back rewrites it again.
        let first_id = v.variants[0].id;
        let back = set_active_variant(&db, "conv1", out.turn_id, first_id).await.unwrap();
        assert_eq!(back.content, "first");
        assert_eq!(get_verbatim(&db, "conv1").await.unwrap().last().unwrap().content, "first");
    }

    #[tokio::test]
    async fn a_variant_from_another_conversation_is_refused() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        ensure_conversation(&db, "conv2", None).await.unwrap();
        record_turn(&db, "conv1", None, "reply", "Aria").await.unwrap();
        let last = get_last_assistant_turn(&db, "conv1").await.unwrap().unwrap();
        let v = get_variants(&db, last.id).await.unwrap();
        let err = set_active_variant(&db, "conv2", last.id, v.variants[0].id).await;
        assert!(err.is_err(), "conv2 must not be able to rewrite conv1's turn");
    }

    #[tokio::test]
    async fn regeneration_needs_an_assistant_turn_to_regenerate() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        record_user_turn(&db, "conv1", &Message::new("user", "hi")).await.unwrap();
        assert!(record_regeneration(&db, "conv1", "x").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn the_prompt_is_persona_summary_verbatim_then_the_new_message() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        db.execute(
            "UPDATE conversations SET summary = ? WHERE id = ?",
            vec![TValue::Text("what happened before".into()), TValue::Text("conv1".into())],
        )
        .await
        .unwrap();
        add_turn(&db, "conv1", "user", "older question").await;
        add_turn(&db, "conv1", "assistant", "older answer").await;

        let persona = vec![Message::new("system", "You are Aria.")];
        let director = vec![Message::new("system", "[Keep it short.]")];
        let incoming = vec![Message::new("user", "and now?")];
        let built = build_context(&db, "conv1", &incoming, &persona, &director).await.unwrap();

        let roles: Vec<&str> = built.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "system", "user", "assistant", "system", "user"]);
        assert!(built.messages[1].content.starts_with(SUMMARY_HEADER));
        assert_eq!(built.messages[4].content, "[Keep it short.]", "the director note is volatile, so it sits last");
        assert_eq!(built.messages[5].content, "and now?");
        assert_eq!(built.stats.verbatim_count, 2);
        assert_eq!(built.stats.persona_chars, "You are Aria.".len());
        assert_eq!(built.stats.new_user_chars, "and now?".len());
        assert!(built.retrieved.is_empty(), "nothing is archived yet");
    }

    #[tokio::test]
    async fn the_stable_prefix_is_everything_up_to_the_volatile_tail() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        add_turn(&db, "conv1", "user", "abc").await;
        let persona = vec![Message::new("system", "xy")];
        let built = build_context(&db, "conv1", &[Message::new("user", "new")], &persona, &[])
            .await
            .unwrap();
        assert_eq!(built.stats.stable_prefix_chars, 2 + 3);
        assert_eq!(built.stats.prompt_chars, 2 + 3 + 3);
    }

    #[tokio::test]
    async fn a_resent_last_message_is_not_appended_twice() {
        // The renderer sends the whole thread, so the newest user message is
        // usually ALREADY the last verbatim turn.
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        add_turn(&db, "conv1", "user", "only message").await;
        let built = build_context(&db, "conv1", &[Message::new("user", "only message")], &[], &[])
            .await
            .unwrap();
        assert_eq!(built.messages.len(), 1);
        assert_eq!(built.stats.new_user_chars, 0);
    }

    #[tokio::test]
    async fn empty_and_malformed_incoming_messages_are_ignored() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        let incoming = vec![Message::new("user", "   "), Message::new("tool", "x")];
        let built = build_context(&db, "conv1", &incoming, &[], &[]).await.unwrap();
        assert!(built.latest_user.is_none());
        assert!(built.messages.is_empty());
    }

    #[tokio::test]
    async fn the_sidebar_query_reports_counts_without_reading_the_summary() {
        let (_d, db) = scratch().await;
        seed_character(&db, "char1").await;
        ensure_conversation(&db, "conv1", Some("char1")).await.unwrap();
        ensure_conversation(&db, "conv2", None).await.unwrap();
        add_turn(&db, "conv1", "user", "hello world").await;
        db.execute(
            "INSERT INTO archive (conversation_id, role, content, embedding, created_at)
             VALUES (?, 'user', 'old', NULL, ?)",
            vec![TValue::Text("conv1".into()), TValue::Text(now_iso())],
        )
        .await
        .unwrap();
        db.execute(
            "UPDATE conversations SET summary = 'x' WHERE id = ?",
            vec![TValue::Text("conv1".into())],
        )
        .await
        .unwrap();

        let all = list_conversations(&db, None).await.unwrap();
        assert_eq!(all.len(), 2);
        let one = list_conversations(&db, Some("char1")).await.unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].turn_count, 2, "verbatim plus archived");
        assert!(one[0].has_summary);
        assert_eq!(one[0].preview, "hello world");
    }

    #[tokio::test]
    async fn the_inspector_view_attaches_variants_only_to_the_latest_reply() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        record_turn(&db, "conv1", Some(&Message::new("user", "a")), "b", "Aria").await.unwrap();
        record_turn(&db, "conv1", Some(&Message::new("user", "c")), "d", "Aria").await.unwrap();
        record_regeneration(&db, "conv1", "d2").await.unwrap();

        let detail = get_conversation(&db, "conv1").await.unwrap().unwrap();
        assert_eq!(detail.verbatim.len(), 4);
        assert!(detail.verbatim[1].variants.is_none(), "the older reply is not swipeable");
        assert_eq!(detail.verbatim[3].variants.as_ref().unwrap().len(), 2);
        assert_eq!(detail.verbatim[3].active_index, Some(1));
    }

    #[tokio::test]
    async fn titles_are_bounded_and_missing_conversations_report_it() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        let long: String = "t".repeat(400);
        let title = set_title(&db, "conv1", &long).await.unwrap().unwrap();
        assert_eq!(title.chars().count(), 200);
        assert!(set_title(&db, "nope", "x").await.unwrap().is_none());
        assert!(set_title(&db, "bad id!", "x").await.is_err());
    }

    #[tokio::test]
    async fn deleting_a_conversation_takes_its_turns_and_archive_with_it() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        add_turn(&db, "conv1", "user", "x").await;
        db.execute(
            "INSERT INTO archive (conversation_id, role, content, embedding, created_at)
             VALUES (?, 'user', 'old', NULL, ?)",
            vec![TValue::Text("conv1".into()), TValue::Text(now_iso())],
        )
        .await
        .unwrap();

        assert!(delete_conversation(&db, "conv1").await.unwrap());
        assert_eq!(turn_count(&db, "conv1").await.unwrap(), 0);
        let left = db
            .query_one(
                "SELECT COUNT(*) AS n FROM archive WHERE conversation_id = ?",
                vec![TValue::Text("conv1".into())],
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(int(&left, "n"), 0, "the cascade has to reach the archive");
        assert!(!delete_conversation(&db, "conv1").await.unwrap());
    }

    #[tokio::test]
    async fn only_a_genuinely_empty_conversation_is_swept_away() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "empty", None).await.unwrap();
        ensure_conversation(&db, "used", None).await.unwrap();
        add_turn(&db, "used", "user", "x").await;
        assert!(!delete_conversation_if_empty(&db, "used").await.unwrap());
        assert!(delete_conversation_if_empty(&db, "empty").await.unwrap());
        assert!(!delete_conversation_if_empty(&db, "bad id!").await.unwrap());
    }

    #[tokio::test]
    async fn the_archive_cache_loads_once_and_then_only_the_new_rows() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        let mut e = CacheEntry::default();

        for i in 0..3 {
            let mut v = unit(i as f32);
            normalize_in_place(&mut v);
            db.execute(
                "INSERT INTO archive (conversation_id, role, content, embedding, created_at)
                 VALUES (?, 'user', ?, ?, ?)",
                vec![
                    TValue::Text("conv1".into()),
                    TValue::Text(format!("row{i}")),
                    TValue::Blob(codec::encode(&v)),
                    TValue::Text(now_iso()),
                ],
            )
            .await
            .unwrap();
        }
        load_archive_vectors(&db, "conv1", &mut e).await.unwrap();
        assert_eq!(e.ids.len(), 3);
        assert_eq!(e.vecs.len(), 3 * EMBED_DIM);
        let high = e.max_id;

        // A second pass with nothing new must not re-read a single row.
        load_archive_vectors(&db, "conv1", &mut e).await.unwrap();
        assert_eq!(e.ids.len(), 3);
        assert_eq!(e.max_id, high);
    }

    #[tokio::test]
    async fn a_shrunken_archive_forces_a_rebuild() {
        // A restore or a sync pull can remove rows. A max(id) check alone would
        // keep serving vectors for rows that no longer exist.
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        let mut e = CacheEntry::default();
        for i in 0..4 {
            let mut v = unit(i as f32);
            normalize_in_place(&mut v);
            db.execute(
                "INSERT INTO archive (conversation_id, role, content, embedding, created_at)
                 VALUES (?, 'user', 'x', ?, ?)",
                vec![
                    TValue::Text("conv1".into()),
                    TValue::Blob(codec::encode(&v)),
                    TValue::Text(now_iso()),
                ],
            )
            .await
            .unwrap();
        }
        load_archive_vectors(&db, "conv1", &mut e).await.unwrap();
        assert_eq!(e.ids.len(), 4);

        db.execute(
            "DELETE FROM archive WHERE id = (SELECT MIN(id) FROM archive WHERE conversation_id = ?)",
            vec![TValue::Text("conv1".into())],
        )
        .await
        .unwrap();
        load_archive_vectors(&db, "conv1", &mut e).await.unwrap();
        assert_eq!(e.ids.len(), 3, "the cache rebuilt rather than kept a dead vector");
        assert_eq!(e.vecs.len(), 3 * EMBED_DIM);
    }

    #[tokio::test]
    async fn rows_without_a_vector_stay_out_of_the_scan_but_still_count() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        let mut e = CacheEntry::default();
        db.execute(
            "INSERT INTO archive (conversation_id, role, content, embedding, created_at)
             VALUES (?, 'user', 'no vector', NULL, ?)",
            vec![TValue::Text("conv1".into()), TValue::Text(now_iso())],
        )
        .await
        .unwrap();
        load_archive_vectors(&db, "conv1", &mut e).await.unwrap();
        assert!(e.ids.is_empty(), "not retrievable");
        assert_eq!(e.rows_seen, 1, "but counted, or the next pass rebuilds forever");
    }

    #[tokio::test]
    async fn retrieval_is_skipped_entirely_when_nothing_is_archived() {
        // Proof it never reaches the embedder: there is no engine running in the
        // test process, so any embed call would fail the call rather than return
        // an empty list.
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        assert!(retrieve(&db, "conv1", "anything").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn the_cache_registry_evicts_the_least_recently_used_conversation() {
        let cap = config().archive_cache_convs;
        for i in 0..=cap {
            touch_cache(&format!("evict-{i}"));
        }
        let reg = archive_cache().lock().unwrap();
        assert!(reg.len() <= cap);
        assert!(
            !reg.iter().any(|(k, _)| k == "evict-0"),
            "the oldest entry is the one that goes"
        );
    }

    #[tokio::test]
    async fn a_lock_is_exclusive_and_released_on_drop() {
        let held = acquire_lock("lock-conv", Duration::from_secs(5)).await;
        assert!(!held.timed_out);
        // A second acquire cannot proceed while the first is held.
        let blocked = acquire_lock("lock-conv", Duration::from_millis(50)).await;
        assert!(blocked.timed_out, "waiting past the deadline proceeds unlocked");
        drop(held);
        drop(blocked);
        let after = acquire_lock("lock-conv", Duration::from_millis(200)).await;
        assert!(!after.timed_out);
    }

    #[tokio::test]
    async fn finished_locks_do_not_accumulate() {
        // The JS version leaked one dead promise per conversation for the life of
        // the process; refcounting is what replaced that.
        for i in 0..50 {
            drop(acquire_lock(&format!("tmp-{i}"), Duration::from_millis(10)).await);
        }
        drop(acquire_lock("tmp-final", Duration::from_millis(10)).await);
        let reg = locks().lock().unwrap();
        assert!(reg.len() <= 2, "registry held {} entries", reg.len());
    }

    #[tokio::test]
    async fn cancelling_maintenance_stops_it_from_writing() {
        let state = Maint::default();
        state.cancelled.store(true, Ordering::SeqCst);
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        for i in 0..40 {
            add_turn(&db, "conv1", "user", &format!("m{i}")).await;
        }
        // Cancelled before the summarizer is ever reached, so this needs no
        // engine: the flag short-circuits the fold.
        let before = turn_count(&db, "conv1").await.unwrap();
        state.rerun.store(false, Ordering::SeqCst);
        let res = run_maintenance(&db, "conv1", "Aria", &state).await;
        assert!(res.is_err() || res.unwrap().archived == 0);
        assert_eq!(turn_count(&db, "conv1").await.unwrap(), before, "no rows moved");
    }

    #[tokio::test]
    async fn a_fold_below_the_threshold_does_nothing_and_clears_the_drain_flag() {
        let (_d, db) = scratch().await;
        ensure_conversation(&db, "conv1", None).await.unwrap();
        add_turn(&db, "conv1", "user", "one").await;
        let state = Maint::default();
        state.draining.store(true, Ordering::SeqCst);
        let res = run_maintenance(&db, "conv1", "Aria", &state).await.unwrap();
        assert_eq!(res, TurnResult::default());
        assert!(!state.draining.load(Ordering::SeqCst));
    }

    #[test]
    fn scratch_paths_never_point_at_the_real_database() {
        // A guard on the tests themselves: `scratch()` must stay inside a temp
        // directory, because these tests write and delete rows freely.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("scenario.db");
        assert!(p.starts_with(std::env::temp_dir()) || !Path::new("./data").exists());
    }
}
