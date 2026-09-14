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
                let body = res.text().await.unwrap_or_default();
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
    fn trailing_slashes_go() {
        assert_eq!(trim_slash("http://x:8080///"), "http://x:8080");
        assert_eq!(trim_slash("http://x:8080"), "http://x:8080");
    }
}
