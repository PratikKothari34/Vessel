//! Generation telemetry (stage 0 of decision 0001).
//!
//! Every chat stream ends with a `done` object carrying the numbers that decide
//! every tuning question in `docs/MASTER.md`:
//!
//! ```text
//! prompt_eval_count     tokens actually prefilled  -> the real live-window size
//! prompt_eval_duration  ns spent prefilling        -> prefill throughput
//! eval_count/_duration  generated tokens + ns      -> decode throughput
//! load_duration         ns spent loading weights   -> non-zero == model evicted
//! ```
//!
//! Nothing here sits on the response path: records land after the stream ends.
//!
//! `prefill_reuse` is the point of the stage 2 prompt reorder. An engine
//! re-prefills from the first token that differs from the previous request, so a
//! block that changes every turn (retrieval) poisons everything after it. We
//! keep the previous prompt per conversation, measure the shared prefix, and
//! report it as a fraction. Reorder working == this trends toward 1.0.
//!
//! `cache_reuse` is the same quantity MEASURED rather than estimated, and only
//! llama-server reports it. The two should track each other closely; when they
//! do not, the estimate is what is wrong.

use serde::Serialize;
use serde_json::json;
use std::sync::Mutex;

use crate::inference::{DoneStats, Message};
use crate::util::now_iso;

fn int_env(name: &str, def: usize, min: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v >= min && *v <= max)
        .unwrap_or(def)
}

fn ring_size() -> usize {
    int_env("METRICS_RING", 200, 20, 5000)
}
fn prev_max() -> usize {
    int_env("METRICS_PREV_MAX", 32, 1, 512)
}

/// ns -> ms at 0.1 ms resolution.
fn ms_of(n: Option<u64>) -> f64 {
    (n.unwrap_or(0) as f64 / 1e5).round() / 10.0
}

/// Tokens per second at one decimal, or `None` when the engine reported no
/// duration to divide by.
fn tps(count: u64, duration: Option<u64>) -> Option<f64> {
    match duration {
        Some(d) if d > 0 => Some(((count as f64 / (d as f64 / 1e9)) * 10.0).round() / 10.0),
        _ => None,
    }
}

fn round_to(v: f64, places: i32) -> f64 {
    let f = 10f64.powi(places);
    (v * f).round() / f
}

/// The composition of one live window, as `memory::build_context` measured it.
///
/// Character counts, not bytes: the JS side counted UTF-16 units and every
/// number derived from these (chars/token, the reuse fraction) is compared
/// against what that build recorded.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Window {
    pub message_count: usize,
    pub prompt_chars: usize,
    /// Everything up to and including the verbatim turns - the part that is
    /// supposed to be byte-identical to last turn's prompt.
    pub stable_prefix_chars: usize,
    pub persona_chars: usize,
    pub summary_chars: usize,
    pub verbatim_count: usize,
    pub verbatim_chars: usize,
    pub retrieved_count: usize,
    pub retrieved_chars: usize,
    pub director_chars: usize,
    pub new_user_chars: usize,
    pub retrieve_ms: f64,
}

/// One recorded generation. Serialized straight to the metrics UI, so the field
/// names are the renderer's contract.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Record {
    pub at: String,
    pub conversation_id: Option<String>,
    pub character_id: Option<String>,
    pub backend: String,
    pub model: Option<String>,
    pub aborted: bool,
    pub prompt_tokens: Option<u64>,
    pub eval_tokens: Option<u64>,
    pub prompt_ms: f64,
    pub eval_ms: f64,
    pub total_ms: f64,
    pub load_ms: f64,
    pub prefill_tps: Option<f64>,
    pub decode_tps: Option<f64>,
    /// Measured chars/token for THIS window. The `MASTER.md` token estimates are
    /// all char-derived; this is what replaces the divide-by-four guess.
    pub chars_per_token: Option<f64>,
    pub prefill_reuse: Option<f64>,
    pub cached_tokens: Option<u64>,
    pub cache_reuse: Option<f64>,
    pub window: Option<Window>,
}

/// What `record` needs to know that the `done` chunk does not carry.
pub struct Sample<'a> {
    pub conversation_id: Option<&'a str>,
    pub character_id: Option<&'a str>,
    pub model: Option<&'a str>,
    pub backend: &'a str,
    pub done: Option<&'a DoneStats>,
    pub window: Option<Window>,
    /// The outbound prompt, used only for the prefix-reuse estimate. Taken by
    /// value because the store keeps it for the next turn to compare against.
    pub prompt_messages: Option<Vec<Message>>,
    pub aborted: bool,
}

#[derive(Default)]
struct Store {
    ring: Vec<Record>,
    /// conversation id -> that conversation's previous prompt. A plain vector
    /// rather than a map plus a recency list: it is bounded at 512 entries by
    /// `METRICS_PREV_MAX` and touched once per generation, so the linear scan is
    /// free and the ordering IS the recency, which is the whole trick the JS
    /// delete-then-set relied on.
    prev: Vec<(String, Vec<Message>)>,
}

fn store() -> &'static Mutex<Store> {
    static STORE: std::sync::OnceLock<Mutex<Store>> = std::sync::OnceLock::new();
    STORE.get_or_init(Default::default)
}

fn prompt_chars(messages: &[Message]) -> usize {
    messages.iter().map(|m| m.content.chars().count()).sum()
}

/// Shared prefix between two prompts, in characters.
///
/// Whole messages are compared first - an unchanged block costs one length
/// check and a memcmp instead of a character walk - and only the first message
/// that actually differs is walked.
fn shared_prefix_chars(prev: &[Message], cur: &[Message]) -> usize {
    let mut shared = 0usize;
    for (a, b) in prev.iter().zip(cur.iter()) {
        if a.role != b.role {
            break;
        }
        if a.content == b.content {
            shared += a.content.chars().count();
            continue;
        }
        shared += a
            .content
            .chars()
            .zip(b.content.chars())
            .take_while(|(x, y)| x == y)
            .count();
        break;
    }
    shared
}

/// Record this prompt as the conversation's latest and return how much of it
/// the previous one already covered. `None` on the first turn, when there is
/// nothing to compare against.
fn prefix_reuse(st: &mut Store, conversation_id: &str, messages: Vec<Message>) -> Option<f64> {
    let prev = match st.prev.iter().position(|(id, _)| id == conversation_id) {
        Some(i) => Some(st.prev.remove(i).1),
        None => None,
    };
    st.prev.push((conversation_id.to_string(), messages));
    let cap = prev_max();
    if st.prev.len() > cap {
        let over = st.prev.len() - cap;
        st.prev.drain(..over);
    }
    let prev = prev?;
    let cur = &st.prev.last()?.1;
    if prev.is_empty() || cur.is_empty() {
        return None;
    }
    let total = prompt_chars(cur);
    if total == 0 {
        return None;
    }
    Some(round_to(shared_prefix_chars(&prev, cur) as f64 / total as f64, 3))
}

pub fn forget_conversation(conversation_id: &str) {
    if let Ok(mut st) = store().lock() {
        st.prev.retain(|(id, _)| id != conversation_id);
    }
}

pub fn record(sample: Sample<'_>) {
    let d = sample.done.cloned().unwrap_or_default();
    let prompt_tokens = d.prompt_eval_count;
    let eval_tokens = d.eval_count;
    let cached_tokens = d.cached_tokens;

    let mut st = match store().lock() {
        Ok(st) => st,
        Err(p) => p.into_inner(), // a panic elsewhere must not silence telemetry
    };

    let prefill_reuse = match (sample.conversation_id, sample.prompt_messages) {
        (Some(id), Some(msgs)) if !msgs.is_empty() => prefix_reuse(&mut st, id, msgs),
        _ => None,
    };

    let rec = Record {
        at: now_iso(),
        conversation_id: sample.conversation_id.map(str::to_string),
        character_id: sample.character_id.map(str::to_string),
        backend: sample.backend.to_string(),
        model: sample.model.map(str::to_string),
        aborted: sample.aborted,
        prompt_tokens,
        eval_tokens,
        prompt_ms: ms_of(d.prompt_eval_duration),
        eval_ms: ms_of(d.eval_duration),
        total_ms: ms_of(d.total_duration),
        load_ms: ms_of(d.load_duration),
        prefill_tps: prompt_tokens.and_then(|n| tps(n, d.prompt_eval_duration)),
        decode_tps: eval_tokens.and_then(|n| tps(n, d.eval_duration)),
        chars_per_token: match (prompt_tokens, sample.window.as_ref()) {
            (Some(t), Some(w)) if t > 0 && w.prompt_chars > 0 => {
                Some(round_to(w.prompt_chars as f64 / t as f64, 2))
            }
            _ => None,
        },
        prefill_reuse,
        cached_tokens,
        cache_reuse: match (cached_tokens, prompt_tokens) {
            (Some(c), Some(t)) if t > 0 => Some(round_to(c as f64 / t as f64, 3)),
            _ => None,
        },
        window: sample.window,
    };

    st.ring.push(rec);
    // Drain the overflow in one move rather than shifting per dropped record:
    // lowering METRICS_RING at runtime would otherwise re-index the whole ring
    // once for every record it sheds.
    let cap = ring_size();
    if st.ring.len() > cap {
        let over = st.ring.len() - cap;
        st.ring.drain(..over);
    }
}

fn pct(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let i = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    Some(sorted[i.saturating_sub(1).min(sorted.len() - 1)])
}

fn sorted_by<F: Fn(&Record) -> Option<f64>>(recs: &[&Record], f: F) -> Vec<f64> {
    let mut v: Vec<f64> = recs.iter().filter_map(|r| f(r)).filter(|x| x.is_finite()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

fn band(sorted: &[f64]) -> serde_json::Value {
    json!({ "p50": pct(sorted, 50.0), "p95": pct(sorted, 95.0) })
}

pub fn summarize(recs: &[&Record]) -> serde_json::Value {
    let prompt = sorted_by(recs, |r| r.prompt_tokens.map(|n| n as f64));
    let decode = sorted_by(recs, |r| r.decode_tps);
    let prefill = sorted_by(recs, |r| r.prefill_tps);
    let reuse = sorted_by(recs, |r| r.prefill_reuse);
    let cache_reuse = sorted_by(recs, |r| r.cache_reuse);
    let cpt = sorted_by(recs, |r| r.chars_per_token);

    json!({
        "samples": recs.len(),
        // The number that decides num_ctx. Compare p95 against the Modelfile.
        "promptTokens": {
            "p50": pct(&prompt, 50.0),
            "p95": pct(&prompt, 95.0),
            "max": prompt.last().copied(),
        },
        "decodeTps": band(&decode),
        "prefillTps": band(&prefill),
        // Stage 2 target: climbs toward 1.0 once retrieval moves to the tail.
        "prefillReuse": band(&reuse),
        // Engine-reported, not estimated. Null on Ollama, which does not expose it.
        "cacheReuse": if cache_reuse.is_empty() {
            serde_json::Value::Null
        } else {
            json!({
                "p50": pct(&cache_reuse, 50.0),
                "p95": pct(&cache_reuse, 95.0),
                "samples": cache_reuse.len(),
            })
        },
        "charsPerToken": { "p50": pct(&cpt, 50.0) },
        // Non-zero load time means the model was evicted between turns - the
        // signature of VRAM pressure, not of a cold start, when it recurs.
        "reloads": recs.iter().filter(|r| r.load_ms > 50.0).count(),
    })
}

pub fn snapshot(limit: usize, conversation_id: Option<&str>) -> serde_json::Value {
    let st = match store().lock() {
        Ok(st) => st,
        Err(p) => p.into_inner(),
    };
    // References, not copies. Every `Record` owns several `String`s, so cloning
    // the ring to read it allocated once per string per record - for a view that
    // then serializes at most `limit` of them and reads the rest only to compute
    // percentiles. A vector of pointers costs one allocation total.
    let recs: Vec<&Record> = match conversation_id {
        Some(id) => st
            .ring
            .iter()
            .filter(|r| r.conversation_id.as_deref() == Some(id))
            .collect(),
        None => st.ring.iter().collect(),
    };
    let start = recs.len().saturating_sub(limit);
    json!({
        "config": { "ring": ring_size() },
        "summary": summarize(&recs),
        "recent": recs[start..],
    })
}

pub fn reset() {
    if let Ok(mut st) = store().lock() {
        st.ring.clear();
        st.prev.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(parts: &[(&str, &str)]) -> Vec<Message> {
        parts.iter().map(|(r, c)| Message::new(r, *c)).collect()
    }

    #[test]
    fn durations_convert_to_tenths_of_a_millisecond() {
        assert_eq!(ms_of(Some(2_000_000_000)), 2000.0);
        assert_eq!(ms_of(Some(1_234_567)), 1.2, "1.234567 ms lands on a tenth");
        assert_eq!(ms_of(Some(123_456)), 0.1);
        assert_eq!(ms_of(None), 0.0);
        assert_eq!(tps(40, Some(1_000_000_000)), Some(40.0));
        assert_eq!(tps(40, Some(0)), None, "a missing duration is not zero tok/s");
    }

    #[test]
    fn an_unchanged_prefix_counts_whole_messages() {
        let a = msgs(&[("system", "persona"), ("user", "hello there")]);
        let b = msgs(&[("system", "persona"), ("user", "hello world")]);
        // "persona" in full, then "hello " before the two diverge.
        assert_eq!(shared_prefix_chars(&a, &b), 7 + 6);
    }

    #[test]
    fn a_changed_role_stops_the_walk_immediately() {
        let a = msgs(&[("system", "same"), ("user", "same")]);
        let b = msgs(&[("user", "same"), ("user", "same")]);
        assert_eq!(shared_prefix_chars(&a, &b), 0);
    }

    #[test]
    fn the_prefix_walk_counts_characters_not_bytes() {
        // A prompt full of accented prose would otherwise report a reuse
        // fraction built from two different units.
        let a = msgs(&[("user", "\u{e9}\u{e9}\u{e9}x")]);
        let b = msgs(&[("user", "\u{e9}\u{e9}\u{e9}y")]);
        assert_eq!(shared_prefix_chars(&a, &b), 3);
        assert_eq!(prompt_chars(&a), 4);
    }

    #[test]
    fn reuse_is_none_on_the_first_turn_then_a_fraction() {
        let mut st = Store::default();
        let first = msgs(&[("system", "abcd")]);
        assert_eq!(prefix_reuse(&mut st, "c1", first.clone()), None);
        let second = msgs(&[("system", "abcd"), ("user", "efgh")]);
        // 4 of 8 characters carried over.
        assert_eq!(prefix_reuse(&mut st, "c1", second), Some(0.5));
    }

    #[test]
    fn conversations_do_not_read_each_others_prompts() {
        let mut st = Store::default();
        prefix_reuse(&mut st, "c1", msgs(&[("system", "abcd")]));
        assert_eq!(
            prefix_reuse(&mut st, "c2", msgs(&[("system", "abcd")])),
            None,
            "a second conversation must start with no prior, not with c1's"
        );
    }

    #[test]
    fn percentiles_pick_the_nearest_rank() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(pct(&v, 50.0), Some(2.0));
        assert_eq!(pct(&v, 95.0), Some(4.0));
        assert_eq!(pct(&[], 50.0), None);
    }

    #[test]
    fn a_summary_ignores_records_that_carry_no_numbers() {
        let mk = |prompt: Option<u64>, load: f64| Record {
            at: String::new(),
            conversation_id: None,
            character_id: None,
            backend: "ollama".into(),
            model: None,
            aborted: false,
            prompt_tokens: prompt,
            eval_tokens: None,
            prompt_ms: 0.0,
            eval_ms: 0.0,
            total_ms: 0.0,
            load_ms: load,
            prefill_tps: None,
            decode_tps: None,
            chars_per_token: None,
            prefill_reuse: None,
            cached_tokens: None,
            cache_reuse: None,
            window: None,
        };
        let recs = [mk(Some(100), 0.0), mk(None, 0.0), mk(Some(300), 900.0)];
        let s = summarize(&recs.iter().collect::<Vec<_>>());
        assert_eq!(s["samples"], 3);
        assert_eq!(s["promptTokens"]["max"], 300.0);
        assert_eq!(s["reloads"], 1, "only the record that reloaded weights counts");
        assert!(s["cacheReuse"].is_null(), "no backend reported KV reuse");
    }
}
