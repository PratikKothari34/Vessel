//! The one place that knows which engine is generating tokens.
//!
//! Stage 3 of decision 0001 established this seam in the Node backend;
//! stage 4 carries it over. Everything above it talks to one interface:
//!
//! ```text
//! chat_stream(messages, options) -> Refused { status, detail }   upstream said no before the body
//!                                -> Streaming(stream of StreamEvent)
//! generate(model, prompt, opts)  -> String
//! embed(model, text, opts)       -> Vec<f32>
//! ```
//!
//! [`StreamEvent::Chunk`] carries `raw`: an Ollama-shaped chat chunk, which is
//! the wire format the renderer already parses. It predates the split, and the
//! llama-server backend synthesizes it rather than forcing a renderer change
//! that would buy nothing - decision 0001 keeps the renderer unchanged.
//!
//! Chat, summarization and embedding are chosen SEPARATELY, and all three
//! default to Ollama. One llama-server process serves exactly one model, so a
//! backend that is right for chat is not automatically right for the two
//! support models.
//!
//! That is not hypothetical. `INFERENCE_BACKEND` used to move the summarizer
//! with chat, and because llama-server accepts-and-ignores the `model` argument,
//! `SUMMARIZER_MODEL` was then silently discarded and the roleplay chat model
//! wrote the summaries. Measured, it kept 8 of 13 facts and invented named
//! characters, which a rolling summary then feeds back as canon on every later
//! turn. Recall loss degrades a story; confabulation corrupts it. So the
//! summarizer gets its own switch and stays on Ollama, where `SUMMARIZER_MODEL`
//! means something and the model can be pinned to CPU.
//!
//! ## Why an enum and not `dyn Engine`
//!
//! The backend set is closed and known at build time, so dispatch is a match
//! rather than a vtable: no boxing on the embed path (two calls per turn), and
//! when stage 4b adds the in-process llama.cpp engine the compiler names every
//! site that has to handle it instead of letting a missing impl reach runtime.

pub mod gguf;
/// The in-process engine. Gated because it links llama.cpp: a build without the
/// feature carries no native dependency and no CUDA toolchain requirement, and
/// that stays the default so the repo still builds anywhere. `gguf` is NOT
/// gated - it is pure path logic, and its tests are worth running either way.
#[cfg(feature = "local-llama")]
pub mod llama_local;
pub mod llama_server;
pub mod modelfile;
pub mod ollama;
pub mod util;

use anyhow::{anyhow, Result};
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value as Json};
use std::pin::Pin;

/// One message in the outbound prompt.
///
/// The JS version relied on `buildContext` handing back the SAME content string
/// turn to turn so the metrics prefix walk could compare by pointer. Nothing
/// here depends on that: equal strings compare by length first and then memcmp,
/// which is fast enough at prompt scale, and the correctness no longer rests on
/// an allocation detail upstream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.to_string(),
            content: content.into(),
        }
    }
}

/// The numbers the final chunk carries. Durations are nanoseconds, Ollama's
/// unit; the llama-server adapter converts its float milliseconds on the way in
/// so there is exactly one unit above this layer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DoneStats {
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    /// llama-server only: tokens genuinely served from the KV cache
    /// (`timings.cache_n`). Ollama does not expose it.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A token. `raw` is the line to forward to the renderer verbatim; `delta`
    /// is the text to accumulate. A line the backend could not interpret also
    /// arrives here, with an empty delta, so it is still forwarded rather than
    /// swallowed.
    Chunk { raw: String, delta: String },
    /// The generation finished. `raw` is the final wire line.
    Done { raw: String, stats: Box<DoneStats> },
    /// Upstream reported a failure mid-stream. The stream ends after this.
    Error { message: String },
}

/// What `chat_stream` gives back.
///
/// `Refused` is upstream saying no before the body starts - a bad model name, a
/// bad request - which is a reportable status, not a transport failure. A
/// transport failure (nothing listening, DNS, a drop before headers) is an
/// `Err`, so the caller can tell "never started" from "started and failed".
pub enum ChatStart {
    Refused { status: u16, detail: Json },
    Streaming(ChatStream),
}

pub type ChatStream = Pin<Box<dyn Stream<Item = StreamEvent> + Send>>;

/// Options for the non-streaming completion used by the summarizer.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenOpts {
    pub num_ctx: Option<u32>,
    /// 0 pins the model to CPU. On an 8 GB card the chat model already owns
    /// 6.1 GB, so a second model on the GPU does not evict it - llama-server
    /// cannot be evicted - but does oversubscribe the card, and the driver pages
    /// the chat model working set out to system RAM.
    pub num_gpu: Option<u32>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EmbedOpts {
    pub num_gpu: Option<u32>,
}

/// The contract every backend satisfies. Stage 4b implements it a third time,
/// in-process, and nothing above this module changes.
#[allow(async_fn_in_trait)] // Not used through `dyn`; see the module header.
pub trait Engine {
    fn name(&self) -> &'static str;
    fn host(&self) -> &str;
    fn model(&self) -> &str;
    /// Whether the context window is a per-request option. Ollama takes it per
    /// request; llama-server fixes it at launch with `-c`.
    fn accepts_num_ctx(&self) -> bool;
    /// Whether the `model` argument to `generate` and `embed` selects anything.
    ///
    /// Ollama swaps models per request, so it does. The other two serve exactly
    /// the weights they were started with and accept-and-ignore the argument -
    /// which is how `SUMMARIZER_MODEL` once got silently replaced by the roleplay
    /// chat model. Asking the backend beats comparing `name()` against a string
    /// literal that a new backend would not update.
    fn honours_model(&self) -> bool;
    fn describe(&self) -> Json;

    async fn chat_stream(
        &self,
        messages: Vec<Message>,
        options: &Map<String, Json>,
    ) -> Result<ChatStart>;
    async fn generate(&self, model: &str, prompt: &str, opts: GenOpts) -> Result<String>;
    async fn embed(&self, model: &str, text: &str, opts: EmbedOpts) -> Result<Vec<f32>>;
}

pub enum Backend {
    Ollama(ollama::Ollama),
    LlamaServer(llama_server::LlamaServer),
    #[cfg(feature = "local-llama")]
    LlamaLocal(llama_local::LlamaLocal),
}

macro_rules! dispatch {
    // The async arm comes FIRST: `$self:expr` would otherwise swallow the
    // `async` token and fail to parse `self` as a closure.
    //
    // Each match arm returns a DIFFERENT opaque future type, so the await has to
    // happen inside the arm rather than on the match.
    (async $self:expr, $method:ident($($arg:expr),*)) => {
        match $self {
            Backend::Ollama(b) => b.$method($($arg),*).await,
            Backend::LlamaServer(b) => b.$method($($arg),*).await,
            #[cfg(feature = "local-llama")]
            Backend::LlamaLocal(b) => b.$method($($arg),*).await,
        }
    };
    ($self:expr, $method:ident($($arg:expr),*)) => {
        match $self {
            Backend::Ollama(b) => b.$method($($arg),*),
            Backend::LlamaServer(b) => b.$method($($arg),*),
            #[cfg(feature = "local-llama")]
            Backend::LlamaLocal(b) => b.$method($($arg),*),
        }
    };
}

impl Engine for Backend {
    fn name(&self) -> &'static str {
        dispatch!(self, name())
    }
    fn host(&self) -> &str {
        dispatch!(self, host())
    }
    fn model(&self) -> &str {
        dispatch!(self, model())
    }
    fn accepts_num_ctx(&self) -> bool {
        dispatch!(self, accepts_num_ctx())
    }
    fn honours_model(&self) -> bool {
        dispatch!(self, honours_model())
    }
    fn describe(&self) -> Json {
        dispatch!(self, describe())
    }
    async fn chat_stream(
        &self,
        messages: Vec<Message>,
        options: &Map<String, Json>,
    ) -> Result<ChatStart> {
        dispatch!(async self, chat_stream(messages, options))
    }
    async fn generate(&self, model: &str, prompt: &str, opts: GenOpts) -> Result<String> {
        dispatch!(async self, generate(model, prompt, opts))
    }
    async fn embed(&self, model: &str, text: &str, opts: EmbedOpts) -> Result<Vec<f32>> {
        dispatch!(async self, embed(model, text, opts))
    }
}

/// The names `pick` accepts. Listed once so the error message cannot drift out
/// of step with the match below it.
const BACKENDS: &[&str] = &["ollama", "llama-server", "llama-local"];

/// Construct one backend.
///
/// Fallible because the in-process engine resolves a GGUF path and loads it:
/// "the model file is not where I looked" is a startup error with something to
/// act on, not a per-request failure found halfway through the first message.
/// The two HTTP backends cannot fail here - nothing is contacted until a
/// request is made.
fn build(kind: &str) -> Result<Backend> {
    Ok(match kind {
        "llama-server" => Backend::LlamaServer(llama_server::LlamaServer::from_env()),
        #[cfg(feature = "local-llama")]
        "llama-local" => Backend::LlamaLocal(llama_local::LlamaLocal::from_env()?),
        #[cfg(not(feature = "local-llama"))]
        "llama-local" => {
            // Naming a backend the binary cannot contain is a build problem, and
            // saying so beats "not a known backend" - the name IS known, this
            // build just does not carry it.
            return Err(anyhow!(
                "llama-local was selected, but this build was compiled without the local-llama \
                 feature. Rebuild with --features local-llama-cuda (or local-llama for CPU), or \
                 use llama-server."
            ));
        }
        _ => Backend::Ollama(ollama::Ollama::from_env()),
    })
}

/// Resolve one backend name from the environment, rejecting anything unknown.
///
/// A typo must not silently fall back to the default: `INFERENCE_BACKEND=llama`
/// would then run the whole session on Ollama while the user believed otherwise,
/// and the only symptom would be a performance number that never improved.
fn pick(var: &str, fallback: &str) -> Result<Backend> {
    let raw = std::env::var(var).unwrap_or_default();
    let key = raw.trim().to_ascii_lowercase();
    if key.is_empty() {
        return build(fallback);
    }
    if !BACKENDS.contains(&key.as_str()) {
        return Err(anyhow!(
            "{var}=\"{raw}\" is not a known backend. Use one of: {}.",
            BACKENDS.join(", ")
        ));
    }
    build(&key)
}

struct Engines {
    chat: Backend,
    embedder: Backend,
    summarizer: Backend,
}

/// Built once. A bad backend name is a startup failure, not a per-request one,
/// so the error is captured here and returned from every accessor.
fn engines() -> Result<&'static Engines> {
    static ENGINES: std::sync::OnceLock<Result<Engines, String>> = std::sync::OnceLock::new();
    ENGINES
        .get_or_init(|| {
            (|| {
                Ok(Engines {
                    chat: pick("INFERENCE_BACKEND", "ollama")?,
                    // Whatever chat is not responsible for stays where it was.
                    // Moving the embedder is an independent, separately
                    // verifiable step.
                    embedder: pick("EMBED_BACKEND", "ollama")?,
                    // Same reasoning, plus one of its own: llama-server ignores
                    // the model argument, so pointing the summarizer at it
                    // silently replaces SUMMARIZER_MODEL with the chat model.
                    // Opt in deliberately or not at all.
                    summarizer: pick("SUMMARIZER_BACKEND", "ollama")?,
                })
            })()
            .map_err(|e: anyhow::Error| e.to_string())
        })
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
}

pub fn chat() -> Result<&'static Backend> {
    Ok(&engines()?.chat)
}
pub fn embedder() -> Result<&'static Backend> {
    Ok(&engines()?.embedder)
}
pub fn summarizer() -> Result<&'static Backend> {
    Ok(&engines()?.summarizer)
}

/// The context window the chat engine is actually running with, or `None` when
/// the backend cannot be asked. llama-server fixes it at launch, so a mismatch
/// with `OLLAMA_NUM_CTX` is a real misconfiguration worth reporting at startup
/// rather than discovering as a truncated story.
pub async fn probe_context() -> Option<u32> {
    let chat = chat().ok()?;
    #[cfg(feature = "local-llama")]
    if let Backend::LlamaLocal(b) = chat {
        // Answered by the engine thread, so a reply also proves the weights
        // finished loading rather than echoing the config back.
        return b.probe_ctx().await;
    }
    let Backend::LlamaServer(_) = chat else {
        return None;
    };
    let res = util::client()
        .get(format!("{}/props", chat.host()))
        .timeout(std::time::Duration::from_millis(1500))
        .send()
        .await
        .ok()?;
    if !res.status().is_success() {
        return None;
    }
    let data: Json = res.json().await.ok()?;
    // Not running yet is not an error here - the health command reports that.
    data.get("default_generation_settings")?
        .get("n_ctx")?
        .as_u64()
        .map(|n| n as u32)
}

pub fn describe() -> Json {
    let Ok(e) = engines() else {
        return json!({ "error": engines().err().map(|e| e.to_string()) });
    };
    json!({
        "chat": e.chat.describe(),
        // Only the host: WHICH model gets embedded is the memory module's
        // EMBED_MODEL, not the adapter chat model, and printing that here read
        // as if the chat model were producing the vectors.
        "embed": { "backend": e.embedder.name(), "host": e.embedder.host() },
        // Which model summarizes is the memory module SUMMARIZER_MODEL - but
        // only while that backend honours it. llama-server does not, so name the
        // backend here and let the health command show the mismatch instead of
        // hiding it.
        "summarize": {
            "backend": e.summarizer.name(),
            "host": e.summarizer.host(),
            "honoursModel": e.summarizer.honours_model(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_backend_name_is_a_startup_error_not_a_silent_default() {
        // Reads a variable nothing else uses, so it cannot disturb the cached
        // engines built from the real ones.
        std::env::set_var("VESSEL_TEST_BACKEND", "llama");
        let err = match pick("VESSEL_TEST_BACKEND", "ollama") {
            Err(e) => e.to_string(),
            Ok(b) => panic!("a typo silently selected {}", b.name()),
        };
        assert!(err.contains("not a known backend"), "{err}");
        std::env::remove_var("VESSEL_TEST_BACKEND");
    }

    #[test]
    fn an_unset_or_blank_backend_takes_the_fallback() {
        std::env::set_var("VESSEL_TEST_BACKEND2", "   ");
        assert_eq!(
            pick("VESSEL_TEST_BACKEND2", "ollama")
                .map(|b| b.name())
                .unwrap_or("err"),
            "ollama"
        );
        std::env::remove_var("VESSEL_TEST_BACKEND2");
        assert_eq!(
            pick("VESSEL_TEST_BACKEND2", "llama-server")
                .map(|b| b.name())
                .unwrap_or("err"),
            "llama-server"
        );
    }

    #[test]
    fn only_ollama_takes_the_context_window_per_request() {
        assert!(build("ollama").unwrap().accepts_num_ctx());
        assert!(!build("llama-server").unwrap().accepts_num_ctx());
    }

    #[test]
    fn only_ollama_lets_the_caller_choose_the_model() {
        // What the health view prints as honoursModel. It used to be a string
        // comparison against "llama-server", which a third backend would have
        // silently passed.
        assert!(build("ollama").unwrap().honours_model());
        assert!(!build("llama-server").unwrap().honours_model());
    }

    /// A name this build cannot carry must not read as a typo.
    #[test]
    #[cfg(not(feature = "local-llama"))]
    fn a_backend_the_build_omits_says_so_rather_than_calling_it_unknown() {
        std::env::set_var("VESSEL_TEST_BACKEND3", "llama-local");
        let err = match pick("VESSEL_TEST_BACKEND3", "ollama") {
            Err(e) => e.to_string(),
            Ok(b) => panic!("a feature-gated backend built as {}", b.name()),
        };
        assert!(err.contains("compiled without the local-llama"), "{err}");
        std::env::remove_var("VESSEL_TEST_BACKEND3");
    }
}
