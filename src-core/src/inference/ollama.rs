//! Ollama inference backend.
//!
//! Behaviour-identical to the Node adapter it replaces. In particular the
//! upstream NDJSON line is carried through UNTOUCHED as `raw`: the wire the
//! renderer parses is an Ollama chat chunk, and this backend must keep producing
//! it byte for byte.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures_util::StreamExt;
use serde_json::{json, Map, Value as Json};

use super::util::{client, fetch_retry, trim_slash, MAX_FRAME_BYTES, UNDELIMITED};
use super::{ChatStart, DoneStats, EmbedOpts, Engine, GenOpts, Message, StreamEvent};

pub struct Ollama {
    host: String,
    model: String,
}

impl Ollama {
    pub fn from_env() -> Self {
        Self {
            host: trim_slash(
                &std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into()),
            ),
            model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "vessel".into()),
        }
    }

    fn chat_url(&self) -> String {
        format!("{}/api/chat", self.host)
    }
}

impl Engine for Ollama {
    fn name(&self) -> &'static str {
        "ollama"
    }
    fn host(&self) -> &str {
        &self.host
    }
    fn model(&self) -> &str {
        &self.model
    }
    /// Ollama takes the context window per request, so tuning it never needs an
    /// `ollama create`. llama-server takes it at launch instead.
    fn accepts_num_ctx(&self) -> bool {
        true
    }
    /// Ollama loads whichever model the request names, so SUMMARIZER_MODEL and
    /// EMBED_MODEL mean what they say. That is why the summarizer defaults here.
    fn honours_model(&self) -> bool {
        true
    }
    /// Ollama exposes no prompt cache to key, so there is nothing to tell it.
    fn keys_kv_by_conversation(&self) -> bool {
        false
    }

    /// Ollama holds no per-conversation state, so there is nothing to drop.
    fn forget_conversation(&self, _conversation: &str) {}
    fn describe(&self) -> Json {
        json!({
            "backend": "ollama",
            "host": self.host,
            "model": self.model,
            "chatUrl": self.chat_url(),
        })
    }

    async fn chat_stream(
        &self,
        messages: Vec<Message>,
        options: &Map<String, Json>,
    ) -> Result<ChatStart> {
        let res = client()
            .post(self.chat_url())
            .json(&json!({
                "model": self.model,
                "messages": messages,
                "stream": true,
                "options": options,
            }))
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status().as_u16();
            let text = res.text().await.unwrap_or_default();
            let detail = serde_json::from_str(&text).unwrap_or(Json::String(text));
            return Ok(ChatStart::Refused { status, detail });
        }

        Ok(ChatStart::Streaming(Box::pin(iterate(res))))
    }

    /// Non-streaming completion, used by the summarizer. `/api/generate` applies
    /// the model own TEMPLATE unless `raw` is set, so the prompt arrives wrapped
    /// exactly as a chat turn would be.
    async fn generate(&self, model: &str, prompt: &str, opts: GenOpts) -> Result<String> {
        let mut options = Map::new();
        if let Some(n) = opts.num_ctx.filter(|n| *n > 0) {
            options.insert("num_ctx".into(), json!(n));
        }
        if let Some(n) = opts.num_gpu {
            options.insert("num_gpu".into(), json!(n));
        }
        let req = client()
            .post(format!("{}/api/generate", self.host))
            .json(&json!({
                "model": model,
                "prompt": prompt,
                "stream": false,
                "options": options,
            }));
        let data: Json = fetch_retry(req, 3).await?.json().await?;
        Ok(data
            .get("response")
            .and_then(Json::as_str)
            .unwrap_or("")
            .trim()
            .to_string())
    }

    async fn embed(&self, model: &str, text: &str, opts: EmbedOpts) -> Result<Vec<f32>> {
        let mut body = json!({ "model": model, "prompt": text });
        if let Some(n) = opts.num_gpu {
            body["options"] = json!({ "num_gpu": n });
        }
        let req = client()
            .post(format!("{}/api/embeddings", self.host))
            .json(&body);
        let data: Json = fetch_retry(req, 3).await?.json().await?;
        let vec: Vec<f32> = match data.get("embedding") {
            Some(Json::Array(a)) if !a.is_empty() => {
                a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()
            }
            _ => return Err(anyhow!("embed: model returned no embedding")),
        };
        Ok(vec)
    }
}

/// Split the NDJSON body into lines and interpret each one.
fn iterate(res: reqwest::Response) -> impl futures_util::Stream<Item = StreamEvent> + Send {
    stream! {
        let mut body = res.bytes_stream();
        // Bytes are appended to a String rather than decoded per chunk: a
        // multi-byte character can straddle a chunk boundary, and pushing the
        // valid prefix forward is what keeps it from becoming U+FFFD.
        let mut buffer = Vec::<u8>::new();
        while let Some(chunk) = body.next().await {
            let Ok(chunk) = chunk else { break };
            buffer.extend_from_slice(&chunk);
            while let Some(nl) = buffer.iter().position(|b| *b == b'\n') {
                let line = String::from_utf8_lossy(&buffer[..nl]).trim().to_string();
                buffer.drain(..=nl);
                if let Some(evt) = parse_line(&line) {
                    yield evt;
                }
            }
            // Checked after draining, so a chunk carrying many whole lines is
            // never mistaken for one runaway line.
            if buffer.len() > MAX_FRAME_BYTES {
                yield StreamEvent::Error { message: UNDELIMITED.to_string() };
                return;
            }
        }
        let tail = String::from_utf8_lossy(&buffer).trim().to_string();
        if let Some(evt) = parse_line(&tail) {
            yield evt;
        }
    }
}

fn parse_line(line: &str) -> Option<StreamEvent> {
    if line.is_empty() {
        return None;
    }
    // Forward, do not interpret: an unparseable line is still the wire.
    let Ok(obj) = serde_json::from_str::<Json>(line) else {
        return Some(StreamEvent::Chunk {
            raw: line.to_string(),
            delta: String::new(),
        });
    };
    if let Some(e) = obj.get("error") {
        return Some(StreamEvent::Error {
            message: e
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| e.to_string()),
        });
    }
    if obj.get("done") == Some(&Json::Bool(true)) {
        let stats: DoneStats = serde_json::from_value(obj).unwrap_or_default();
        return Some(StreamEvent::Done {
            raw: line.to_string(),
            stats: Box::new(stats),
        });
    }
    let delta = obj
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Json::as_str)
        .unwrap_or("")
        .to_string();
    Some(StreamEvent::Chunk {
        raw: line.to_string(),
        delta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_content_chunk_yields_its_delta_and_the_untouched_line() {
        let line =
            r#"{"model":"vessel","message":{"role":"assistant","content":"Hi"},"done":false}"#;
        match parse_line(line).unwrap() {
            StreamEvent::Chunk { raw, delta } => {
                assert_eq!(delta, "Hi");
                assert_eq!(raw, line, "the wire line must be forwarded byte for byte");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_final_chunk_carries_the_telemetry() {
        let line = r#"{"done":true,"done_reason":"stop","prompt_eval_count":120,
            "prompt_eval_duration":2000000000,"eval_count":40,"eval_duration":1000000000,
            "total_duration":3000000000,"load_duration":0}"#;
        match parse_line(line).unwrap() {
            StreamEvent::Done { stats, .. } => {
                assert_eq!(stats.prompt_eval_count, Some(120));
                assert_eq!(stats.eval_count, Some(40));
                assert_eq!(stats.done_reason.as_deref(), Some("stop"));
                assert_eq!(stats.cached_tokens, None, "Ollama does not report KV reuse");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_unparseable_line_is_forwarded_rather_than_swallowed() {
        match parse_line("not json").unwrap() {
            StreamEvent::Chunk { raw, delta } => {
                assert_eq!(raw, "not json");
                assert!(delta.is_empty());
            }
            other => panic!("{other:?}"),
        }
        assert!(parse_line("").is_none());
    }

    #[test]
    fn an_error_object_ends_the_stream_as_an_error() {
        let line = r#"{"error":"model not found"}"#;
        match parse_line(line).unwrap() {
            StreamEvent::Error { message } => assert_eq!(message, "model not found"),
            other => panic!("{other:?}"),
        }
    }
}
