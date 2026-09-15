//! llama.cpp `llama-server` backend (stage 3 of decision 0001).
//!
//! Why this exists, in one line: Ollama keys a resident model instance on
//! (model, num_ctx), so the summarizer evicts the chat model and the next user
//! message pays a 19.4 s reload. llama-server holds one model for the life of
//! the process, serves chat and summarization from the same weights, and reports
//! how many prompt tokens it actually reused instead of making us estimate it.
//!
//! Shape of the swap:
//!
//! ```text
//! Ollama                           llama-server
//! ------                           ------------
//! num_ctx per request              `-c` at launch (nothing can evict)
//! /api/chat, NDJSON                /v1/chat/completions, SSE
//! /api/generate (applies TEMPLATE) /v1/chat/completions, one user message
//! /api/embeddings                  /v1/embeddings (separate instance/port)
//! prompt_eval_count (est. reuse)   timings.cache_n (measured reuse)
//! ```
//!
//! The chat model and the embedding model are two processes here - one
//! llama-server serves one model. The selector in the parent module lets the
//! embedder stay on Ollama while chat moves, so the swap can be done one half at
//! a time.
//!
//! Output contract is identical to the Ollama adapter: `raw` is an Ollama-shaped
//! chat chunk, synthesized here, so the wire and the renderer do not change.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures_util::StreamExt;
use serde_json::{json, Map, Value as Json};

use super::util::{
    client, fetch_retry, ns_from_ms, ollama_chunk, ollama_done, trim_slash, warn_once,
    MAX_FRAME_BYTES, UNDELIMITED,
};
use super::{modelfile, ChatStart, DoneStats, EmbedOpts, Engine, GenOpts, Message, StreamEvent};

/// Characters carry Ollama-named sampling options. Map the ones llama.cpp
/// understands and drop the rest loudly-once, rather than forwarding unknown
/// keys and getting a 400 mid-conversation.
const SAMPLING_MAP: &[(&str, &str)] = &[
    ("num_predict", "max_tokens"),
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("top_k", "top_k"),
    ("min_p", "min_p"),
    ("typical_p", "typical_p"),
    ("repeat_penalty", "repeat_penalty"),
    ("repeat_last_n", "repeat_last_n"),
    ("presence_penalty", "presence_penalty"),
    ("frequency_penalty", "frequency_penalty"),
    ("mirostat", "mirostat"),
    ("mirostat_tau", "mirostat_tau"),
    ("mirostat_eta", "mirostat_eta"),
    ("seed", "seed"),
    ("stop", "stop"),
];

pub struct LlamaServer {
    host: String,
    embed_host: String,
    model: String,
    cache_prompt: bool,
    timings_per_token: bool,
    /// The Modelfile PARAMETER lines, minus `num_ctx`.
    ///
    /// Ollama bakes every Modelfile PARAMETER into the model and applies it to
    /// any request that does not override that key. llama-server loads the bare
    /// GGUF and applies nothing, so without this the model runs at llama.cpp
    /// stock sampling (temperature 0.8, no min_p, no repeat window) instead of
    /// the tuned values the Modelfile declares.
    ///
    /// The Modelfile SYSTEM is deliberately NOT applied. Measured: Ollama drops
    /// it whenever the client sends a system message of its own, and the app
    /// always does. The persona message carries it instead, which is what makes
    /// the two backends produce the same prompt.
    model_params: Map<String, Json>,
}

impl LlamaServer {
    pub fn from_env() -> Self {
        let mut model_params = modelfile::load().params.clone();
        // The window is a launch flag here; translate_sampling would warn about
        // it on every startup for a value nobody passed in this request.
        model_params.remove("num_ctx");
        Self {
            host: trim_slash(
                &std::env::var("LLAMA_CHAT_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".into()),
            ),
            // A second llama-server, started with --embedding and the embedding
            // GGUF. The selector keeps the embedder on Ollama by default so the
            // chat swap can be tested on its own.
            embed_host: trim_slash(
                &std::env::var("LLAMA_EMBED_URL").unwrap_or_else(|_| "http://127.0.0.1:8081".into()),
            ),
            model: std::env::var("LLAMA_CHAT_MODEL")
                .or_else(|_| std::env::var("OLLAMA_MODEL"))
                .unwrap_or_else(|_| "vessel".into()),
            // llama.cpp reuses the longest common prefix of the KV cache when
            // this is on. It is the default upstream; sent explicitly so the
            // behaviour is in our code, not in whatever build is installed.
            cache_prompt: std::env::var("LLAMA_CACHE_PROMPT").as_deref() != Ok("0"),
            // Attaches a `timings` object to every streamed chunk, which is the
            // only way to read `cache_n` on the OpenAI-compatible endpoint.
            // Costs one small JSON object per token.
            timings_per_token: std::env::var("LLAMA_TIMINGS").as_deref() != Ok("0"),
            model_params,
        }
    }
}

/// Translate Ollama-named sampling keys into llama.cpp ones.
///
/// `num_ctx` is a launch flag here, not a request field. Silently dropping it
/// would mean a character asking for a bigger window gets a smaller one with no
/// sign, so it is warned about explicitly, once per key.
fn translate_sampling(options: &Map<String, Json>, out: &mut Map<String, Json>) {
    for (k, v) in options {
        if v.is_null() {
            continue;
        }
        if k == "num_ctx" {
            warn_once(
                "num_ctx",
                &format!(
                    "llama-server: num_ctx={v} ignored - llama-server takes the context window \
                     from its own -c flag at launch. Start it with -c {v}."
                ),
            );
            continue;
        }
        match SAMPLING_MAP.iter().find(|(from, _)| from == k) {
            Some((_, to)) => {
                out.insert((*to).to_string(), v.clone());
            }
            None => warn_once(
                k,
                &format!("llama-server: sampling option \"{k}\" has no llama.cpp equivalent and was dropped."),
            ),
        }
    }
}

/// Build the final chunk in the shape the metrics module already parses.
///
/// `timings.prompt_n` counts only the tokens llama.cpp actually had to process;
/// `cache_n` counts the ones it reused. Ollama's `prompt_eval_count` is the
/// WHOLE prompt with a near-zero duration when it hits cache, so the two are
/// summed here - otherwise "promptTokens p95 vs num_ctx", the number that
/// decides the window size, would silently shrink by whatever the cache
/// absorbed.
fn done_stats(timings: Option<&Json>, usage: Option<&Json>, finish_reason: Option<&str>) -> DoneStats {
    let num = |o: Option<&Json>, k: &str| o.and_then(|t| t.get(k)).and_then(Json::as_f64);
    let int = |o: Option<&Json>, k: &str| o.and_then(|t| t.get(k)).and_then(Json::as_u64);

    let cached = int(timings, "cache_n");
    let processed = int(timings, "prompt_n");
    let prompt_tokens = match processed {
        Some(p) => Some(p + cached.unwrap_or(0)),
        None => int(usage, "prompt_tokens"),
    };
    let eval_tokens = int(timings, "predicted_n").or_else(|| int(usage, "completion_tokens"));

    let prompt_ns = ns_from_ms(num(timings, "prompt_ms").unwrap_or(0.0));
    let eval_ns = ns_from_ms(num(timings, "predicted_ms").unwrap_or(0.0));

    DoneStats {
        done_reason: Some(finish_reason.unwrap_or("stop").to_string()),
        prompt_eval_count: prompt_tokens,
        prompt_eval_duration: (prompt_ns > 0).then_some(prompt_ns),
        eval_count: eval_tokens,
        eval_duration: (eval_ns > 0).then_some(eval_ns),
        total_duration: Some(prompt_ns + eval_ns),
        // Always 0: the weights are resident for the life of the process. That
        // is the whole point of the stage - a non-zero value here would mean
        // llama-server restarted under us.
        load_duration: Some(0),
        cached_tokens: cached,
    }
}

impl Engine for LlamaServer {
    fn name(&self) -> &'static str {
        "llama-server"
    }
    fn host(&self) -> &str {
        &self.host
    }
    fn model(&self) -> &str {
        &self.model
    }
    /// The context window is fixed at launch. Callers use this to decide whether
    /// to send num_ctx at all, and to warn when the two disagree.
    fn accepts_num_ctx(&self) -> bool {
        false
    }
    /// One process, one model, named by `-m` at launch. The `model` field of a
    /// request is accepted and ignored, so SUMMARIZER_MODEL silently does
    /// nothing here - the health view says so rather than hiding it.
    fn honours_model(&self) -> bool {
        false
    }
    fn describe(&self) -> Json {
        json!({
            "backend": "llama-server",
            "host": self.host,
            "model": self.model,
            "embedHost": self.embed_host,
            "cachePrompt": self.cache_prompt,
            "modelParams": self.model_params,
        })
    }

    async fn chat_stream(&self, messages: Vec<Message>, options: &Map<String, Json>) -> Result<ChatStart> {
        let mut sampling = Map::new();
        // Modelfile first, request second: a character own sampling still wins,
        // exactly as it does on Ollama.
        translate_sampling(&self.model_params, &mut sampling);
        translate_sampling(options, &mut sampling);

        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "stream_options": { "include_usage": true },
            "cache_prompt": self.cache_prompt,
        });
        let obj = body.as_object_mut().expect("object");
        obj.extend(sampling);
        if self.timings_per_token {
            obj.insert("timings_per_token".into(), Json::Bool(true));
        }

        let res = client()
            .post(format!("{}/v1/chat/completions", self.host))
            .json(&body)
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status().as_u16();
            let text = res.text().await.unwrap_or_default();
            let detail = serde_json::from_str(&text).unwrap_or(Json::String(text));
            return Ok(ChatStart::Refused { status, detail });
        }

        Ok(ChatStart::Streaming(Box::pin(iterate(res, self.model.clone()))))
    }

    /// Deliberately the chat endpoint and not `/completion`: Ollama's
    /// `/api/generate` applies the model chat TEMPLATE, so a raw completion here
    /// would change what the summarizer sees and make the two backends produce
    /// different summaries for the same history. `model` is accepted and ignored
    /// - one server, one model.
    async fn generate(&self, _model: &str, prompt: &str, opts: GenOpts) -> Result<String> {
        if let Some(n) = opts.num_ctx.filter(|n| *n > 0) {
            warn_once(
                "gen_num_ctx",
                &format!("llama-server: summarizer num_ctx={n} ignored - one process, one window (-c)."),
            );
        }
        let req = client()
            .post(format!("{}/v1/chat/completions", self.host))
            .json(&json!({
                "model": self.model,
                "messages": [{ "role": "user", "content": prompt }],
                "stream": false,
                // The summary prompt is a different prefix every time and would
                // evict the chat conversation slot. Give it its own slot instead
                // of poisoning the one the story is using.
                "cache_prompt": false,
                // Deliberately NOT the Modelfile sampling. Those values
                // (temperature 0.9, min_p 0.05) are tuned to make roleplay prose
                // surprising, which is the opposite of what a summary needs. On
                // Ollama the summarizer is a different model with its own
                // defaults, so there is no parity to keep here - only a right
                // answer, which is low variance.
                "temperature": 0.3,
                "top_p": 0.9,
                "repeat_penalty": 1.05,
            }));
        let data: Json = fetch_retry(req, 3).await?.json().await?;
        let text = data
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(Json::as_str)
            .unwrap_or("");
        Ok(text.trim().to_string())
    }

    async fn embed(&self, model: &str, text: &str, _opts: EmbedOpts) -> Result<Vec<f32>> {
        let req = client()
            .post(format!("{}/v1/embeddings", self.embed_host))
            .json(&json!({ "model": if model.is_empty() { "embed" } else { model }, "input": text }));
        let data: Json = fetch_retry(req, 3).await?.json().await?;
        let row = data.get("data").and_then(|d| d.get(0));
        let mut vec = row.and_then(|r| r.get("embedding"));

        // With --pooling none llama.cpp returns one vector per token. We want a
        // single vector; that configuration is a misconfiguration for our use,
        // so say so rather than silently picking row 0.
        if let Some(Json::Array(outer)) = vec {
            if matches!(outer.first(), Some(Json::Array(_))) {
                if outer.len() != 1 {
                    return Err(anyhow!(
                        "embed: llama-server returned {} token vectors - start it with --pooling mean",
                        outer.len()
                    ));
                }
                vec = outer.first();
            }
        }

        match vec {
            Some(Json::Array(a)) if !a.is_empty() => {
                Ok(a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
            }
            _ => Err(anyhow!("embed: model returned no embedding")),
        }
    }
}

/// State carried across SSE frames.
///
/// Last-writer-wins: llama.cpp repeats timings on each chunk and puts usage on
/// the final one, so carrying the most recent of each gives the true totals.
#[derive(Default)]
struct Tail {
    timings: Option<Json>,
    usage: Option<Json>,
    finish_reason: Option<String>,
    saw_done: bool,
}

fn iterate(res: reqwest::Response, model: String) -> impl futures_util::Stream<Item = StreamEvent> + Send {
    stream! {
        let mut body = res.bytes_stream();
        let mut buffer = Vec::<u8>::new();
        let mut tail = Tail::default();
        let server = LlamaServerChunk { model: model.clone() };

        while let Some(chunk) = body.next().await {
            let Ok(chunk) = chunk else { break };
            buffer.extend_from_slice(&chunk);
            // SSE frames are separated by a blank line; a frame may carry
            // several `data:` lines that concatenate. CRLF is normalised so the
            // split holds regardless of proxy.
            while let Some(idx) = find_frame_end(&buffer) {
                let raw = String::from_utf8_lossy(&buffer[..idx.0]).to_string();
                buffer.drain(..idx.1);
                for evt in frame(&payload_of(&raw), &mut tail, &server) {
                    yield evt;
                }
            }
            // Same bound as the NDJSON reader, and for the same reason: a host
            // that streams without ever closing a frame must fail loudly.
            if buffer.len() > MAX_FRAME_BYTES {
                yield StreamEvent::Error { message: UNDELIMITED.to_string() };
                return;
            }
        }
        let rest = String::from_utf8_lossy(&buffer).trim().to_string();
        if !rest.is_empty() {
            for evt in frame(&payload_of(&rest), &mut tail, &server) {
                yield evt;
            }
        }

        // Only synthesize a done chunk if the stream actually finished. An
        // aborted read must record as aborted, exactly as it does on Ollama.
        if tail.saw_done || tail.finish_reason.is_some() || tail.usage.is_some() || tail.timings.is_some() {
            let stats = done_stats(tail.timings.as_ref(), tail.usage.as_ref(), tail.finish_reason.as_deref());
            yield StreamEvent::Done { raw: ollama_done(&model, &stats), stats: Box::new(stats) };
        }
    }
}

struct LlamaServerChunk {
    model: String,
}

impl LlamaServerChunk {
    fn content(&self, text: &str) -> String {
        ollama_chunk(&self.model, text)
    }
}

/// Offset of the end of the first complete frame, and how many bytes to drop.
/// Handles both LF and CRLF separators without rewriting the buffer.
fn find_frame_end(buf: &[u8]) -> Option<(usize, usize)> {
    for i in 0..buf.len().saturating_sub(1) {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some((i, i + 2));
        }
        if i + 3 < buf.len() && &buf[i..i + 4] == b"\r\n\r\n" {
            return Some((i, i + 4));
        }
    }
    None
}

/// Concatenate the `data:` lines of one frame.
fn payload_of(raw: &str) -> String {
    let mut payload = String::new();
    for line in raw.split('\n') {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("data:") {
            payload.push_str(rest.trim());
        }
    }
    payload
}

fn frame(payload: &str, tail: &mut Tail, chunk: &LlamaServerChunk) -> Vec<StreamEvent> {
    if payload.is_empty() {
        return Vec::new();
    }
    if payload == "[DONE]" {
        tail.saw_done = true;
        return Vec::new();
    }
    // Keepalive or partial: not an error, just nothing to report.
    let Ok(obj) = serde_json::from_str::<Json>(payload) else { return Vec::new() };

    if let Some(e) = obj.get("error") {
        let message = e
            .get("message")
            .and_then(Json::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| e.as_str().map(str::to_string).unwrap_or_else(|| e.to_string()));
        return vec![StreamEvent::Error { message }];
    }
    if let Some(t) = obj.get("timings") {
        tail.timings = Some(t.clone());
    }
    if let Some(u) = obj.get("usage") {
        if !u.is_null() {
            tail.usage = Some(u.clone());
        }
    }
    let Some(choice) = obj.get("choices").and_then(|c| c.get(0)) else { return Vec::new() };
    if let Some(r) = choice.get("finish_reason").and_then(Json::as_str) {
        tail.finish_reason = Some(r.to_string());
    }
    match choice.get("delta").and_then(|d| d.get("content")).and_then(Json::as_str) {
        Some(text) if !text.is_empty() => {
            vec![StreamEvent::Chunk { raw: chunk.content(text), delta: text.to_string() }]
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, Json)]) -> Map<String, Json> {
        pairs.iter().map(|(k, v)| ((*k).to_string(), v.clone())).collect()
    }

    #[test]
    fn ollama_named_sampling_becomes_llama_cpp_named_sampling() {
        let mut out = Map::new();
        translate_sampling(
            &map(&[
                ("num_predict", json!(512)),
                ("temperature", json!(0.9)),
                ("repeat_penalty", json!(1.1)),
            ]),
            &mut out,
        );
        assert_eq!(out.get("max_tokens"), Some(&json!(512)));
        assert_eq!(out.get("temperature"), Some(&json!(0.9)));
        assert_eq!(out.get("repeat_penalty"), Some(&json!(1.1)));
        assert!(!out.contains_key("num_predict"));
    }

    #[test]
    fn the_context_window_and_unknown_keys_are_dropped_not_forwarded() {
        // Forwarding either one gets a 400 from llama.cpp mid-conversation.
        let mut out = Map::new();
        translate_sampling(&map(&[("num_ctx", json!(32768)), ("nonsense", json!(1))]), &mut out);
        assert!(out.is_empty(), "{out:?}");
    }

    #[test]
    fn a_request_option_overrides_the_modelfile_value() {
        let mut out = Map::new();
        translate_sampling(&map(&[("temperature", json!(0.9))]), &mut out);
        translate_sampling(&map(&[("temperature", json!(0.2))]), &mut out);
        assert_eq!(out.get("temperature"), Some(&json!(0.2)));
    }

    #[test]
    fn prompt_tokens_are_processed_plus_cached() {
        // The whole point: reporting only `prompt_n` would make the p95 that
        // sizes the context window shrink by whatever the KV cache absorbed.
        let t = json!({ "prompt_n": 20, "cache_n": 1000, "prompt_ms": 50.0, "predicted_n": 8, "predicted_ms": 200.0 });
        let s = done_stats(Some(&t), None, Some("stop"));
        assert_eq!(s.prompt_eval_count, Some(1020));
        assert_eq!(s.cached_tokens, Some(1000));
        assert_eq!(s.eval_count, Some(8));
        assert_eq!(s.prompt_eval_duration, Some(50_000_000));
        assert_eq!(s.eval_duration, Some(200_000_000));
        assert_eq!(s.total_duration, Some(250_000_000));
        assert_eq!(s.load_duration, Some(0), "the weights never reload here");
    }

    #[test]
    fn usage_covers_for_missing_timings() {
        let u = json!({ "prompt_tokens": 300, "completion_tokens": 12 });
        let s = done_stats(None, Some(&u), None);
        assert_eq!(s.prompt_eval_count, Some(300));
        assert_eq!(s.eval_count, Some(12));
        assert_eq!(s.done_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn the_synthesized_done_line_is_ollama_shaped() {
        let t = json!({ "prompt_n": 10, "cache_n": 5, "prompt_ms": 1.0, "predicted_n": 2, "predicted_ms": 2.0 });
        let s = done_stats(Some(&t), None, Some("length"));
        let raw: Json = serde_json::from_str(&ollama_done("vessel", &s)).unwrap();
        assert_eq!(raw["done"], json!(true));
        assert_eq!(raw["done_reason"], json!("length"));
        assert_eq!(raw["message"]["content"], json!(""));
        assert_eq!(raw["prompt_eval_count"], json!(15));
        assert_eq!(raw["cached_tokens"], json!(5));
    }

    #[test]
    fn absent_telemetry_is_absent_rather_than_null() {
        let s = done_stats(None, None, None);
        let raw: Json = serde_json::from_str(&ollama_done("vessel", &s)).unwrap();
        for key in ["prompt_eval_count", "eval_count", "cached_tokens", "prompt_eval_duration"] {
            assert!(raw.get(key).is_none(), "{key} should be absent, got {raw}");
        }
    }

    #[test]
    fn sse_frames_split_on_a_blank_line_in_either_line_ending() {
        assert_eq!(find_frame_end(b"data: a\n\ndata: b\n\n"), Some((7, 9)));
        assert_eq!(find_frame_end(b"data: a\r\n\r\nrest"), Some((7, 11)));
        assert_eq!(find_frame_end(b"data: a\n"), None);
    }

    #[test]
    fn a_frame_concatenates_its_data_lines_and_ignores_the_rest() {
        assert_eq!(payload_of("event: x\ndata: {\"a\":\ndata: 1}"), "{\"a\":1}");
        assert_eq!(payload_of(": keepalive"), "");
    }

    #[test]
    fn a_delta_becomes_an_ollama_shaped_chunk() {
        let mut tail = Tail::default();
        let c = LlamaServerChunk { model: "vessel".into() };
        let evts = frame(
            r#"{"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}"#,
            &mut tail,
            &c,
        );
        match &evts[0] {
            StreamEvent::Chunk { raw, delta } => {
                assert_eq!(delta, "Hi");
                let parsed: Json = serde_json::from_str(raw).unwrap();
                assert_eq!(parsed["message"]["content"], json!("Hi"));
                assert_eq!(parsed["done"], json!(false));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_terminator_is_recorded_but_emits_nothing() {
        let mut tail = Tail::default();
        let c = LlamaServerChunk { model: "vessel".into() };
        assert!(frame("[DONE]", &mut tail, &c).is_empty());
        assert!(tail.saw_done);
    }
}
