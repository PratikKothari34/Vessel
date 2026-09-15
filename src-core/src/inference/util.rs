//! Shared helpers for the inference backends.

use anyhow::{anyhow, Result};
use std::time::Duration;

/// One shared client for every backend call.
///
/// Building a `reqwest::Client` per request throws away the connection pool and
/// the TLS session cache, and these are hot paths: one embed call per turn on
/// both legs of retrieval, plus a summarize call. The pool also keeps the
/// keep-alive connection to llama-server open between turns.
pub fn client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            // No read timeout on purpose: a chat stream is open for as long as
            // the model is generating, and the connect timeout is what actually
            // catches "llama-server is not running".
            .connect_timeout(Duration::from_secs(5))
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .expect("reqwest client")
    })
}

/// Retry only what is worth retrying: a 5xx or a transport error.
///
/// A 4xx is a bad request and will stay bad, so it gives up immediately. The JS
/// version needed a `fatal` flag to express that, because a `throw` inside its
/// `try` was caught by the `catch` in the same block and retried anyway - every
/// "model not found" then cost three requests and 1.2 s of backoff to report a
/// 404 it knew on the first attempt. Here the control flow is explicit.
pub async fn fetch_retry(req: reqwest::RequestBuilder, tries: u32) -> Result<reqwest::Response> {
    let mut last: Option<anyhow::Error> = None;
    for i in 0..tries {
        let attempt = req
            .try_clone()
            .ok_or_else(|| anyhow!("request body is not replayable"))?;
        match attempt.send().await {
            Ok(res) if res.status().is_success() => return Ok(res),
            Ok(res) => {
                let status = res.status();
                let body = error_body(res).await;
                last = Some(anyhow!("{} {}", status.as_u16(), body));
                if !status.is_server_error() {
                    break;
                }
            }
            Err(e) => {
                let fatal = e.is_builder();
                last = Some(e.into());
                if fatal {
                    break;
                }
            }
        }
        if i + 1 < tries {
            tokio::time::sleep(Duration::from_millis(400 * (i as u64 + 1))).await;
        }
    }
    Err(last.unwrap_or_else(|| anyhow!("request failed")))
}

/// Largest a stream buffer may grow while waiting for a delimiter.
///
/// Both readers accumulate bytes until they see the end of a line or a frame.
/// A well-behaved engine sends one small object per token, so the buffer never
/// holds more than a few hundred bytes - but nothing in the protocol promises a
/// delimiter will ever arrive, and `OLLAMA_HOST` is a user-editable field. Point
/// it at something that streams without one and the buffer grows until the
/// process dies, with no error to explain it. A megabyte is far past any real
/// frame and small enough that hitting it is a report, not an outage.
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

/// What a stream that never delimits itself is told to say.
pub const UNDELIMITED: &str =
    "The engine sent more than a megabyte with no frame boundary. It may not be a model server.";

/// How much of a failed response is worth keeping.
///
/// The body of an error becomes the `detail` the renderer shows. An engine host
/// that is not an engine at all - a stale port, a proxy, a login page - answers
/// with a whole HTML document, and reading it in full both buffers it and puts
/// it in front of the user. A couple of lines is what identifies the failure.
const MAX_ERROR_BODY: usize = 2048;

/// Read a failed response, stopping once there is enough to name the failure.
async fn error_body(res: reqwest::Response) -> String {
    use futures_util::StreamExt;
    let mut out = String::new();
    let mut body = res.bytes_stream();
    while let Some(Ok(chunk)) = body.next().await {
        out.push_str(&String::from_utf8_lossy(&chunk));
        if out.len() >= MAX_ERROR_BODY {
            break;
        }
    }
    clip(out)
}

/// Cut to [`MAX_ERROR_BODY`] on a character boundary, marking the cut. Slicing
/// on the byte index alone panics whenever a multi-byte character straddles it.
fn clip(mut s: String) -> String {
    if s.len() <= MAX_ERROR_BODY {
        return s;
    }
    let mut end = MAX_ERROR_BODY;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    s.push_str("...");
    s
}

/// Ollama reports durations in nanoseconds; llama.cpp reports them in float
/// milliseconds. The metrics module speaks nanoseconds, so llama-server
/// converts here.
pub fn ns_from_ms(ms: f64) -> u64 {
    if ms.is_finite() && ms > 0.0 {
        (ms * 1e6).round() as u64
    } else {
        0
    }
}

pub fn trim_slash(s: &str) -> String {
    s.trim_end_matches('/').to_string()
}

/// Warn about a dropped or ignored option once per key, not once per request.
///
/// A per-request warning for a value the user set once is noise that trains
/// people to ignore the log; a silent drop is worse, because a character asking
/// for a bigger context window would just quietly get a smaller one.
pub fn warn_once(key: &str, msg: &str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: std::sync::OnceLock<Mutex<HashSet<String>>> = std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut g = match seen.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if g.insert(key.to_string()) {
        tracing::warn!("{msg}");
    }
}

/// Build the Ollama-shaped chat chunk the renderer parses.
///
/// Shared rather than duplicated because it is the wire contract: decision 0001
/// keeps the renderer unchanged, so every backend that is not Ollama has to
/// synthesize exactly this shape, and two copies of it drift the moment one is
/// edited.
pub fn ollama_chunk(model: &str, text: &str) -> String {
    serde_json::json!({
        "model": model,
        "created_at": crate::util::now_iso(),
        "message": { "role": "assistant", "content": text },
        "done": false,
    })
    .to_string()
}

/// The final chunk of a stream, carrying the numbers.
pub fn ollama_done(model: &str, stats: &crate::inference::DoneStats) -> String {
    let mut chunk = serde_json::json!({
        "model": model,
        "created_at": crate::util::now_iso(),
        "message": { "role": "assistant", "content": "" },
        "done": true,
        "done_reason": stats.done_reason.clone().unwrap_or_else(|| "stop".into()),
        "load_duration": stats.load_duration.unwrap_or(0),
        "total_duration": stats.total_duration.unwrap_or(0),
    });
    let obj = chunk.as_object_mut().expect("object");
    // Absent rather than null, exactly as the Ollama chunk has them: the
    // renderer and the metrics reader both treat a missing key as "unknown".
    for (key, val) in [
        ("prompt_eval_count", stats.prompt_eval_count),
        ("prompt_eval_duration", stats.prompt_eval_duration),
        ("eval_count", stats.eval_count),
        ("eval_duration", stats.eval_duration),
        // Not an Ollama field. The metrics module reads it when present and
        // reports measured KV reuse alongside the char-prefix estimate that
        // works on both backends.
        ("cached_tokens", stats.cached_tokens),
    ] {
        if let Some(v) = val {
            obj.insert(key.to_string(), serde_json::json!(v));
        }
    }
    chunk.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nanoseconds_from_float_milliseconds() {
        assert_eq!(ns_from_ms(1.5), 1_500_000);
        assert_eq!(ns_from_ms(0.0), 0);
        assert_eq!(ns_from_ms(-3.0), 0);
        assert_eq!(ns_from_ms(f64::NAN), 0);
    }

    #[test]
    fn a_runaway_error_body_is_cut_at_a_character_boundary() {
        // An engine host that is not an engine answers with a whole document,
        // and a byte-index slice through one of its characters panics.
        let out = clip("\u{e9}".repeat(MAX_ERROR_BODY));
        assert!(out.len() <= MAX_ERROR_BODY + 3, "kept {} bytes", out.len());
        assert!(out.ends_with("..."), "a cut body must say it was cut");
        assert!(out.trim_end_matches('.').chars().all(|c| c == '\u{e9}'));
    }

    #[test]
    fn a_short_error_body_is_kept_whole() {
        assert_eq!(clip("model not found".into()), "model not found");
    }

    #[test]
    fn trailing_slashes_go() {
        assert_eq!(trim_slash("http://x:8080///"), "http://x:8080");
        assert_eq!(trim_slash("http://x:8080"), "http://x:8080");
    }
}
