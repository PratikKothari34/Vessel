//! In-process llama.cpp (stage 4b of decision 0001).
//!
//! Stage 3 removed Ollama; this removes the last HTTP hop. The weights are
//! loaded into this process and the tokens come back through a channel, not a
//! socket, which is what buys the four things stage 4a could not have:
//!
//! - **Summarization reuses the chat weights.** `generate` runs on the same
//!   model, on its own KV sequence, so there is no second model to load and no
//!   slot for a summary to evict. That is the 3.3 GB `gemma3:4b` deletion the
//!   decision names.
//! - **The KV cache survives the turn.** The tokens decoded last turn stay in
//!   sequence 0, so the next prompt only pays for the tokens that actually
//!   changed - and `cached_tokens` becomes a measured prefix length rather than
//!   an estimate.
//! - **The window, the KV dtype and the offload split are ours**, not a launch
//!   flag on a process somebody else started.
//! - **No process to start.** Nothing to health-check, nothing left running.
//!
//! ## Why a thread and a channel
//!
//! `LlamaContext` borrows its model and is not `Send`, and a decode is a
//! blocking call that owns a CUDA stream for its duration. Both facts point the
//! same way: one dedicated OS thread owns the backend, the model and the context
//! for the life of the process, and everything else talks to it over channels.
//! Requests serialize there, which matches how the llama-server track was
//! already run (`--parallel 1`) and keeps an 8 GB card from being
//! oversubscribed by two live contexts.
//!
//! The channel is a tokio one rather than `std::sync::mpsc` for one concrete
//! reason: the engine is held in a `static` by the parent module, so it must be
//! `Sync`, and `std::sync::mpsc::Sender` is not. The worker drains it with
//! `blocking_recv`, which is exactly what that call is for.
//!
//! The thread starts loading at construction rather than on the first request,
//! so a cold start overlaps with the window opening instead of landing on the
//! first message. A load failure does not kill the thread - it is remembered and
//! returned from every job, because "the channel closed" is the same information
//! with the useful part removed.
//!
//! ## Output contract
//!
//! Unchanged, and deliberately so: `raw` is the same Ollama-shaped chat chunk
//! the other two backends emit, built by the same helpers in `util`. The
//! renderer never learns that this exists.

use anyhow::{anyhow, Result};
use async_stream::stream;
use serde_json::{json, Map, Value as Json};
use std::path::PathBuf;
use std::time::Instant;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{LlamaStateSeqFlags, SeqState};

use super::util::{ollama_chunk, ollama_done, warn_once};
use super::{
    gguf, modelfile, ChatStart, DoneStats, EmbedOpts, Engine, GenOpts, Message, StreamEvent,
};

/// The conversation on screen lives here and is never cleared between turns.
/// That is the persistent prompt cache. A switch to a different conversation
/// does not clear it either - it is serialized out and put back later, which is
/// what `ParkedCache` holds.
const SEQ_CHAT: i32 = 0;
/// The summarizer lives here and is cleared after every call. A summary prompt
/// is a different prefix every time; letting it share sequence 0 would evict the
/// story's cache to save nothing.
const SEQ_AUX: i32 = 1;

/// Read an env var as a number, falling back rather than failing: a typo in a
/// tuning knob must not stop the app from starting.
fn env_num<T: std::str::FromStr>(key: &str, fallback: T) -> T {
    match std::env::var(key) {
        Ok(raw) => raw.trim().parse().unwrap_or_else(|_| {
            warn_once(
                key,
                &format!("llama-local: {key}=\"{raw}\" is not a number; using the default."),
            );
            fallback
        }),
        Err(_) => fallback,
    }
}

/// KV cache element type. F16 is the default because stage 1 of decision 0001
/// measured `q8_0` and rejected it; the knob exists so that decision can be
/// re-measured on this backend without a code change.
fn kv_type(raw: &str) -> Option<KvCacheType> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "f16" => Some(KvCacheType::F16),
        "f32" => Some(KvCacheType::F32),
        "q8_0" => Some(KvCacheType::Q8_0),
        "q5_1" => Some(KvCacheType::Q5_1),
        "q5_0" => Some(KvCacheType::Q5_0),
        "q4_1" => Some(KvCacheType::Q4_1),
        "q4_0" => Some(KvCacheType::Q4_0),
        _ => None,
    }
}

/// Sampling, in llama.cpp terms, with the Ollama-named knobs already resolved.
///
/// The defaults are llama.cpp's own, and they matter: Ollama bakes Modelfile
/// PARAMETER lines into the model while this backend loads a bare GGUF, so
/// anything not set here runs at stock settings. That is the same trap the
/// llama-server adapter documents, one layer lower.
#[derive(Debug, Clone, PartialEq)]
struct Sampling {
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_p: f32,
    typical_p: f32,
    repeat_penalty: f32,
    repeat_last_n: i32,
    presence_penalty: f32,
    frequency_penalty: f32,
    seed: u32,
    /// `<= 0` means "until end-of-generation or the window runs out".
    num_predict: i32,
    stop: Vec<String>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            typical_p: 1.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            // llama.cpp's LLAMA_DEFAULT_SEED: draw a fresh one per run.
            seed: u32::MAX,
            num_predict: -1,
            stop: Vec::new(),
        }
    }
}

impl Sampling {
    /// Apply one option map over the top of whatever is already set.
    ///
    /// Called twice - Modelfile first, request second - so a character's own
    /// sampling still wins, exactly as it does on the other two backends.
    fn apply(&mut self, options: &Map<String, Json>) {
        for (k, v) in options {
            if v.is_null() {
                continue;
            }
            let f = v.as_f64().map(|n| n as f32);
            let i = v.as_i64().map(|n| n as i32);
            let known = match k.as_str() {
                "temperature" => f.map(|n| self.temperature = n).is_some(),
                "top_p" => f.map(|n| self.top_p = n).is_some(),
                "top_k" => i.map(|n| self.top_k = n).is_some(),
                "min_p" => f.map(|n| self.min_p = n).is_some(),
                "typical_p" => f.map(|n| self.typical_p = n).is_some(),
                "repeat_penalty" => f.map(|n| self.repeat_penalty = n).is_some(),
                "repeat_last_n" => i.map(|n| self.repeat_last_n = n).is_some(),
                "presence_penalty" => f.map(|n| self.presence_penalty = n).is_some(),
                "frequency_penalty" => f.map(|n| self.frequency_penalty = n).is_some(),
                "num_predict" => i.map(|n| self.num_predict = n).is_some(),
                "seed" => v.as_i64().map(|n| self.seed = n as u32).is_some(),
                "stop" => match v {
                    Json::String(s) => {
                        self.stop = vec![s.clone()];
                        true
                    }
                    Json::Array(a) => {
                        self.stop = a
                            .iter()
                            .filter_map(Json::as_str)
                            .map(str::to_string)
                            .collect();
                        true
                    }
                    _ => false,
                },
                // The window is a property of the context, which is built once
                // at load. Saying so beats generating in a smaller one silently.
                "num_ctx" => {
                    warn_once(
                        "local_num_ctx",
                        &format!(
                            "llama-local: num_ctx={v} ignored - the window is fixed when the model \
                             loads. Set LLAMA_NUM_CTX and restart."
                        ),
                    );
                    true
                }
                // Not a sampling knob. `chat_stream` reads it to decide which
                // conversation's KV cache this turn runs against, and it is
                // named here so the unknown-option warning does not fire for a
                // field this backend is the one that asked for.
                "conversation_id" => true,
                _ => false,
            };
            if !known {
                warn_once(
                    k,
                    &format!(
                        "llama-local: sampling option \"{k}\" has no llama.cpp equivalent, or its \
                         value is the wrong type, and was dropped."
                    ),
                );
            }
        }
    }

    fn chain(&self, n_vocab: i32) -> LlamaSampler {
        // Order is llama.cpp's own: penalties act on raw logits, the truncating
        // samplers narrow the tail, temperature scales what is left, and `dist`
        // draws from it.
        LlamaSampler::chain_simple([
            LlamaSampler::penalties(
                n_vocab,
                self.repeat_last_n,
                self.repeat_penalty,
                self.frequency_penalty,
                self.presence_penalty,
            ),
            LlamaSampler::top_k(self.top_k),
            LlamaSampler::typical(self.typical_p, 1),
            LlamaSampler::top_p(self.top_p, 1),
            LlamaSampler::min_p(self.min_p, 1),
            LlamaSampler::temp(self.temperature),
            LlamaSampler::dist(self.seed),
        ])
    }

    /// What the summarizer runs with: low variance, and deliberately NOT the
    /// Modelfile values. Those are tuned to make roleplay prose surprising,
    /// which is the opposite of what a summary needs. Same reasoning and the
    /// same numbers as the llama-server adapter.
    fn for_summary() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.9,
            repeat_penalty: 1.05,
            ..Self::default()
        }
    }
}

/// Turns token bytes into the text that is safe to send right now.
///
/// Two things force a delay:
/// - a multi-byte character can straddle two tokens, and half of one is not
///   text;
/// - a stop string can straddle several tokens, and no part of it may be shown.
///
/// The second is why the holdback is computed per token rather than fixed. The
/// obvious "hold back the longest stop minus one" is off by one: when a stop
/// lands exactly at the end of a token that window still releases its first
/// byte, and the user sees a stray newline a moment before the turn ends. What
/// is held back instead is the longest suffix of the text so far that is also
/// the START of a stop string. If the rest of it arrives, none of it was ever
/// sent; if it does not, the held bytes go out with the next token.
struct Emitter {
    /// Bytes that are not yet a whole character.
    pending: Vec<u8>,
    /// Everything decoded so far, the held-back tail included.
    text: String,
    /// How much of `text` has already gone out.
    emitted: usize,
    stops: Vec<String>,
    /// Where a stop string starts, once one has landed. Nothing from here on is
    /// ever emitted, and it is where the reply ends.
    stop: Option<usize>,
}

impl Emitter {
    fn new(stops: Vec<String>) -> Self {
        // An empty stop string matches at position 0 and would end every turn
        // before its first token.
        let stops = stops.into_iter().filter(|s| !s.is_empty()).collect();
        Self {
            pending: Vec::new(),
            text: String::new(),
            emitted: 0,
            stops,
            stop: None,
        }
    }

    /// Feed one token's bytes. Returns the text that is safe to send now.
    fn push(&mut self, bytes: &[u8]) -> Option<String> {
        self.pending.extend_from_slice(bytes);
        self.decode();
        if self.stop.is_none() {
            self.stop = self
                .stops
                .iter()
                .filter_map(|s| self.text.find(s.as_str()))
                .min();
        }
        let limit = self.stop.unwrap_or_else(|| self.holdback());
        self.take(limit)
    }

    /// Move every complete character out of `pending` and into `text`.
    fn decode(&mut self) {
        while !self.pending.is_empty() {
            let cut = match std::str::from_utf8(&self.pending) {
                Ok(_) => self.pending.len(),
                Err(e) if e.valid_up_to() > 0 => e.valid_up_to(),
                // A byte that begins no sequence at all: drop it rather than let
                // it wedge the buffer for the rest of the generation.
                Err(e) if e.error_len().is_some() => {
                    self.pending.drain(..1);
                    continue;
                }
                // An incomplete character at the front. Wait for the rest of it.
                Err(_) => return,
            };
            let head: Vec<u8> = self.pending.drain(..cut).collect();
            if let Ok(s) = String::from_utf8(head) {
                self.text.push_str(&s);
            }
        }
    }

    /// The furthest point that can be sent without showing the beginning of a
    /// stop string. Compared as bytes, so a multi-byte character inside a stop
    /// string cannot panic the slice.
    fn holdback(&self) -> usize {
        let text = self.text.as_bytes();
        let mut limit = text.len();
        for stop in &self.stops {
            let bytes = stop.as_bytes();
            let most = (bytes.len() - 1).min(text.len());
            for k in (1..=most).rev() {
                if text[text.len() - k..] == bytes[..k] {
                    limit = limit.min(text.len() - k);
                    break;
                }
            }
        }
        // Never cut a character in half on the way out.
        while limit > 0 && !self.text.is_char_boundary(limit) {
            limit -= 1;
        }
        limit
    }

    fn take(&mut self, limit: usize) -> Option<String> {
        if limit <= self.emitted {
            return None;
        }
        let out = self.text[self.emitted..limit].to_string();
        self.emitted = limit;
        Some(out)
    }

    /// Everything still held back, once the generation is over. A trailing
    /// incomplete character is dropped: it is not text yet and never will be.
    fn flush(&mut self) -> Option<String> {
        let end = self.stop.unwrap_or(self.text.len());
        self.take(end)
    }

    fn hit_stop(&self) -> bool {
        self.stop.is_some()
    }

    /// The reply, with anything from the stop string on cut away.
    fn finished(&self) -> &str {
        &self.text[..self.stop.unwrap_or(self.text.len())]
    }
}

/// What the engine asks the worker to do. Every variant carries its own reply
/// channel: the worker never holds a reference to the caller.
enum Job {
    Chat {
        messages: Vec<Message>,
        sampling: Box<Sampling>,
        /// Which conversation's KV cache this turn belongs to. `None` means the
        /// caller did not say, and every such turn shares one sequence - the
        /// behaviour this engine had before parking existed.
        conversation: Option<String>,
        out: UnboundedSender<StreamEvent>,
    },
    Generate {
        prompt: String,
        out: tokio::sync::oneshot::Sender<Result<String>>,
    },
    /// Load state, for the startup probe. It queues behind the load, which is
    /// what makes it a real health check rather than a config echo.
    Info(tokio::sync::oneshot::Sender<Json>),
    /// A conversation is gone; drop its KV cache. Fire-and-forget on purpose -
    /// the caller is a delete that has already committed, and there is nothing
    /// useful it could do with a reply.
    Forget(String),
}

pub struct LlamaLocal {
    gguf: PathBuf,
    model: String,
    n_ctx: u32,
    n_gpu_layers: u32,
    kv: String,
    model_params: Map<String, Json>,
    jobs: UnboundedSender<Job>,
}

impl LlamaLocal {
    /// Resolve the configuration and start the worker.
    ///
    /// Fallible on purpose: a GGUF that cannot be found is a startup error with
    /// an actionable message, not a per-request failure discovered halfway
    /// through the first story.
    pub fn from_env() -> Result<Self> {
        let model = std::env::var("LLAMA_CHAT_MODEL")
            .or_else(|_| std::env::var("OLLAMA_MODEL"))
            .unwrap_or_else(|_| "vessel".into());
        let gguf = gguf::resolve(&model)?;

        let mut model_params = modelfile::load().params.clone();
        // The window is a load-time property here; `apply` would warn about it
        // on every startup for a value nobody passed in this request.
        model_params.remove("num_ctx");

        // Stage 1 of decision 0001 adopted 12288 and the rest of the config
        // still writes it in Ollama's variable, so that one is read as the
        // fallback rather than inventing a second source of truth.
        let n_ctx: u32 = env_num("LLAMA_NUM_CTX", env_num("OLLAMA_NUM_CTX", 12288u32));
        // 99 is "all of them". An 8 GB card holds the measured 6.03 GB working
        // set, so partial offload is a fallback for a smaller card, not the
        // normal case.
        let n_gpu_layers: u32 = env_num("LLAMA_NGL", 99u32);
        let kv = std::env::var("LLAMA_KV_TYPE").unwrap_or_else(|_| "f16".into());
        if kv_type(&kv).is_none() {
            return Err(anyhow!(
                "LLAMA_KV_TYPE=\"{kv}\" is not a KV cache type. Use one of: f16, f32, q8_0, q5_1, \
                 q5_0, q4_1, q4_0."
            ));
        }

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let cfg = WorkerCfg {
            gguf: gguf.clone(),
            n_ctx,
            n_gpu_layers,
            kv: kv.clone(),
            n_batch: env_num("LLAMA_N_BATCH", 2048u32).max(1),
            n_threads: env_num("LLAMA_THREADS", default_threads()),
            // Parking costs host RAM, and how much depends on the model: an 8B
            // at f16 serializes about 131 KB per token, so a conversation that
            // filled a 12288-token window is ~1.6 GB. The default holds one of
            // those plus several short ones. 0 turns parking off and restores
            // the single-sequence behaviour.
            kv_cache_bytes: env_num("LLAMA_KV_CACHE_MB", 2048usize).saturating_mul(1024 * 1024),
        };
        std::thread::Builder::new()
            .name("llama-local".into())
            .spawn(move || worker(cfg, rx))
            .map_err(|e| anyhow!("llama-local: could not start the engine thread - {e}"))?;

        Ok(Self {
            gguf,
            model,
            n_ctx,
            n_gpu_layers,
            kv,
            model_params,
            jobs: tx,
        })
    }

    /// The window the context was actually built with, asked of the engine
    /// rather than read back off our own config - so the answer also proves the
    /// model loaded. `None` means it has not finished loading, or it failed;
    /// either way the health command is the place that reports it, not this.
    pub async fn probe_ctx(&self) -> Option<u32> {
        let info = self.probe().await?;
        info.get("nCtx").and_then(Json::as_u64).map(|n| n as u32)
    }

    /// Ask the engine thread what it is holding right now.
    ///
    /// Queued behind whatever it is doing, and bounded: a reply proves the
    /// weights loaded, and no reply inside the window means the thread is busy
    /// or gone - either way the caller reports "unknown", never a stale answer.
    pub async fn probe(&self) -> Option<Json> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.send(Job::Info(tx)).ok()?;
        tokio::time::timeout(std::time::Duration::from_millis(1500), rx)
            .await
            .ok()?
            .ok()
    }

    fn send(&self, job: Job) -> Result<()> {
        self.jobs
            .send(job)
            .map_err(|_| anyhow!("llama-local: the engine thread has stopped; restart the app."))
    }
}

fn default_threads() -> i32 {
    std::thread::available_parallelism().map_or(4, |n| n.get() as i32)
}

impl Engine for LlamaLocal {
    fn name(&self) -> &'static str {
        "llama-local"
    }
    /// Nothing is listening anywhere. Named rather than left blank so the health
    /// view reads as a deliberate answer instead of a missing one.
    fn host(&self) -> &str {
        "in-process"
    }
    fn model(&self) -> &str {
        &self.model
    }
    /// One KV cache per conversation, parked and restored on a switch - so it
    /// does want to be told which conversation a turn belongs to.
    fn keys_kv_by_conversation(&self) -> bool {
        true
    }

    /// Queued behind whatever the worker is doing, so a delete during a reply
    /// lands after that reply rather than pulling the cells out from under it.
    /// A dead worker means a dead cache; nothing to forget.
    fn forget_conversation(&self, conversation: &str) {
        let _ = self.jobs.send(Job::Forget(conversation.to_string()));
    }
    /// Fixed when the context is built, same as llama-server.
    fn accepts_num_ctx(&self) -> bool {
        false
    }
    /// One model is loaded, so the `model` argument to `generate` is accepted
    /// and ignored - and the health view has to say so, or a `SUMMARIZER_MODEL`
    /// that does nothing looks like it is working.
    fn honours_model(&self) -> bool {
        false
    }
    fn describe(&self) -> Json {
        json!({
            "backend": "llama-local",
            "host": "in-process",
            "model": self.model,
            // The file name, not the path: which quantization is loaded is the
            // first thing to check when the numbers move, and the directory is
            // the user's home.
            "gguf": self.gguf.file_name().map(|n| n.to_string_lossy().to_string()),
            "numCtx": self.n_ctx,
            "nGpuLayers": self.n_gpu_layers,
            "kvType": self.kv,
            "modelParams": self.model_params,
        })
    }

    async fn chat_stream(
        &self,
        messages: Vec<Message>,
        options: &Map<String, Json>,
    ) -> Result<ChatStart> {
        let mut sampling = Sampling::default();
        sampling.apply(&self.model_params);
        sampling.apply(options);

        // Which conversation this turn belongs to, so the worker can put that
        // conversation's KV cache back before it decodes. `chat.rs` sends it
        // only to a backend that says it keys its cache this way, so `None`
        // here means the caller genuinely has no id, not that it forgot.
        let conversation = options
            .get("conversation_id")
            .and_then(Json::as_str)
            .map(str::to_owned);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        self.send(Job::Chat {
            messages,
            sampling: Box::new(sampling),
            conversation,
            out: tx,
        })?;
        Ok(ChatStart::Streaming(Box::pin(stream! {
            while let Some(evt) = rx.recv().await {
                yield evt;
            }
        })))
    }

    async fn generate(&self, _model: &str, prompt: &str, opts: GenOpts) -> Result<String> {
        if let Some(n) = opts.num_ctx.filter(|n| *n > 0) {
            warn_once(
                "local_gen_num_ctx",
                &format!("llama-local: summarizer num_ctx={n} ignored - one context, one window."),
            );
        }
        if opts.num_gpu == Some(0) {
            // On Ollama this pinned the summarizer to CPU so it could not evict
            // the chat model. There is nothing to evict here: it is the same
            // weights, already resident, on its own KV sequence.
            warn_once(
                "local_gen_num_gpu",
                "llama-local: summarizer num_gpu=0 ignored - it shares the loaded chat weights.",
            );
        }
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.send(Job::Generate {
            prompt: prompt.to_string(),
            out: tx,
        })?;
        rx.await
            .map_err(|_| anyhow!("llama-local: the engine thread stopped mid-generation."))?
    }

    /// Deliberately unimplemented.
    ///
    /// Embedding needs a SECOND model - `nomic-embed-text` is not this model -
    /// and a context built with pooling on, which is a different context rather
    /// than a different call. Loading it here would put a second set of weights
    /// on an 8 GB card to save one loopback call that is not on the hot path.
    /// The selector already lets the embedder stay where it is, so the honest
    /// answer is to say so.
    async fn embed(&self, _model: &str, _text: &str, _opts: EmbedOpts) -> Result<Vec<f32>> {
        Err(anyhow!(
            "llama-local does not embed - it loads one chat model. Leave EMBED_BACKEND on ollama \
             or llama-server."
        ))
    }
}

struct WorkerCfg {
    gguf: PathBuf,
    n_ctx: u32,
    n_gpu_layers: u32,
    kv: String,
    n_batch: u32,
    n_threads: i32,
    /// Host memory this engine may hold for conversations that are not on
    /// screen. See `LLAMA_KV_CACHE_MB`.
    kv_cache_bytes: usize,
}

/// Owns the backend, the model and the context, and answers jobs until the
/// engine is dropped.
///
/// The backend is a local here rather than a field anywhere: it must outlive
/// every model and context built from it, and a stack frame that nothing returns
/// from is the simplest way to guarantee that.
fn worker(cfg: WorkerCfg, mut rx: UnboundedReceiver<Job>) {
    let backend = match LlamaBackend::init() {
        Ok(b) => b,
        Err(e) => {
            return serve_error(
                rx,
                &format!("llama-local: llama.cpp would not initialise - {e}"),
            )
        }
    };
    let params = LlamaModelParams::default().with_n_gpu_layers(cfg.n_gpu_layers);
    let model = match LlamaModel::load_from_file(&backend, &cfg.gguf, &params) {
        Ok(m) => m,
        Err(e) => {
            return serve_error(
                rx,
                &format!("llama-local: could not load {} - {e}", cfg.gguf.display()),
            )
        }
    };
    let mut session = match Session::new(&cfg, &backend, &model) {
        Ok(s) => s,
        Err(e) => return serve_error(rx, &e.to_string()),
    };
    tracing::info!(
        model = %cfg.gguf.display(),
        n_ctx = session.ctx.n_ctx(),
        n_gpu_layers = cfg.n_gpu_layers,
        kv = %cfg.kv,
        "llama-local: loaded"
    );
    while let Some(job) = rx.blocking_recv() {
        session.handle(job, &cfg);
    }
}

/// Answer every job with the same failure, for as long as anyone asks. The
/// thread stays alive so the message survives; a closed channel would replace it
/// with nothing.
fn serve_error(mut rx: UnboundedReceiver<Job>, message: &str) {
    tracing::error!("{message}");
    while let Some(job) = rx.blocking_recv() {
        match job {
            Job::Chat { out, .. } => {
                let _ = out.send(StreamEvent::Error {
                    message: message.to_string(),
                });
            }
            Job::Generate { out, .. } => {
                let _ = out.send(Err(anyhow!("{message}")));
            }
            Job::Info(out) => {
                let _ = out.send(json!({ "loaded": false, "error": message }));
            }
            // Nothing ever loaded, so nothing is parked.
            Job::Forget(_) => {}
        }
    }
}

/// One conversation's KV cache, serialized out of the context so a different
/// conversation can have the cells.
///
/// The bytes are host memory. `LlamaStateSeqFlags::ON_DEVICE` would keep the
/// copy in VRAM and restore faster, which is exactly the wrong trade on an 8 GB
/// card the weights already own 6 GB of - the whole point of parking is to stop
/// paying for a prefill, not to buy that back in evictions.
struct Parked<S> {
    id: String,
    /// What the restored sequence holds, so the next turn's prefix walk has
    /// something real to compare the new prompt against.
    tokens: Vec<LlamaToken>,
    state: S,
    /// Cached rather than asked of `state` on every comparison: the budget
    /// arithmetic runs on every switch and the answer never changes.
    bytes: usize,
}

/// Least-recently-used parked conversations, bounded by total bytes.
///
/// Bytes and not a count, because the sizes are not comparable: a two-turn
/// conversation parks a few megabytes and one that filled the window parks
/// ~1.6 GB. A count-based cap would either hold several of the large ones - and
/// swap the machine - or evict four small ones to make room for nothing.
struct ParkedCache<S> {
    budget: usize,
    bytes: usize,
    /// Oldest first; the back is the most recently parked.
    entries: Vec<Parked<S>>,
}

impl<S> ParkedCache<S> {
    fn new(budget: usize) -> Self {
        Self {
            budget,
            bytes: 0,
            entries: Vec::new(),
        }
    }

    /// Remove and return one conversation's cache, giving its bytes back to the
    /// budget.
    fn take(&mut self, id: &str) -> Option<Parked<S>> {
        let at = self.entries.iter().position(|e| e.id == id)?;
        let entry = self.entries.remove(at);
        self.bytes -= entry.bytes;
        Some(entry)
    }

    /// Park one conversation, evicting the oldest until it fits.
    ///
    /// `false` means it does not fit even in an empty cache, so nothing was
    /// evicted to make room for something that was never going to be kept. The
    /// caller has lost those cells either way; the difference is whether the
    /// budget is the reason, which is the only version worth a log line.
    fn put(&mut self, entry: Parked<S>) -> bool {
        // A re-park replaces its older self rather than sitting beside it.
        let _ = self.take(&entry.id);
        if entry.bytes > self.budget {
            return false;
        }
        while self.bytes + entry.bytes > self.budget {
            let old = self.entries.remove(0);
            self.bytes -= old.bytes;
        }
        self.bytes += entry.bytes;
        self.entries.push(entry);
        true
    }
}

struct Session<'a> {
    model: &'a LlamaModel,
    ctx: LlamaContext<'a>,
    /// The name the renderer sees on every chunk, read once from the GGUF so the
    /// health view and the wire agree about what is loaded.
    name: String,
    /// Exactly the tokens currently held in `SEQ_CHAT`. The prefix this shares
    /// with the next prompt is what does not have to be decoded again.
    resident: Vec<LlamaToken>,
    /// Which conversation those tokens belong to. `None` before the first turn,
    /// and after a switch that had nothing to restore.
    resident_id: Option<String>,
    /// Set when a clear failed, so the cells `SEQ_CHAT` holds no longer match
    /// what `resident`/`resident_id` say they do. While it is set, the
    /// same-conversation fast path is off and the next switch clears again -
    /// otherwise a turn that aborted on a failed clear would be followed by one
    /// that took the fast path straight into the cells it never dropped.
    dirty: bool,
    /// Every other conversation this process has seen, up to the budget.
    parked: ParkedCache<SeqState>,
}

impl<'a> Session<'a> {
    fn new(cfg: &WorkerCfg, backend: &LlamaBackend, model: &'a LlamaModel) -> Result<Self> {
        let kv = kv_type(&cfg.kv).unwrap_or(KvCacheType::F16);
        let params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(cfg.n_ctx))
            .with_n_batch(cfg.n_batch)
            .with_n_threads(cfg.n_threads)
            .with_n_threads_batch(cfg.n_threads)
            // Two: the story in sequence 0, the summarizer in sequence 1.
            .with_n_seq_max(2)
            .with_type_k(kv)
            .with_type_v(kv);
        let ctx = model
            .new_context(backend, params)
            .map_err(|e| anyhow!("llama-local: could not build a context - {e}"))?;
        let name = model
            .meta_val_str("general.name")
            .unwrap_or_else(|_| "vessel".into());
        Ok(Self {
            model,
            ctx,
            name,
            resident: Vec::new(),
            resident_id: None,
            dirty: false,
            parked: ParkedCache::new(cfg.kv_cache_bytes),
        })
    }

    fn handle(&mut self, job: Job, cfg: &WorkerCfg) {
        match job {
            Job::Chat {
                messages,
                sampling,
                conversation,
                out,
            } => {
                if let Err(e) = self.chat(messages, &sampling, conversation, cfg, &out) {
                    let _ = out.send(StreamEvent::Error {
                        message: e.to_string(),
                    });
                }
            }
            Job::Generate { prompt, out } => {
                let _ = out.send(self.summarize(&prompt, cfg));
            }
            Job::Info(out) => {
                let _ = out.send(json!({
                    "loaded": true,
                    "residentTokens": self.resident.len(),
                    "nCtx": self.ctx.n_ctx(),
                    "parkedConversations": self.parked.entries.len(),
                    "parkedBytes": self.parked.bytes,
                    "parkedBudgetBytes": self.parked.budget,
                }));
            }
            Job::Forget(id) => self.forget(&id),
        }
    }

    /// Drop everything this session holds for a conversation that no longer
    /// exists - parked or resident.
    ///
    /// A failed clear is only logged here, unlike in `switch_to`: nothing is
    /// about to decode into those cells, and the failure sets `dirty`, so the
    /// next switch clears again before anything does. Losing the delete over it
    /// would be worse.
    fn forget(&mut self, id: &str) {
        let parked = self.parked.take(id).is_some();
        let was_resident = self.resident_id.as_deref() == Some(id);
        if was_resident {
            // Cleared, not parked: `park_current` would put it straight back.
            self.resident_id = None;
            self.resident.clear();
            if let Err(e) = self.clear_chat_seq() {
                tracing::warn!("llama-local: {e}");
            }
        }
        if parked || was_resident {
            tracing::debug!(conversation = %id, "llama-local: KV cache dropped");
        }
    }

    /// Turn messages into tokens the way the GGUF's own template says to.
    ///
    /// The template is read from the model file rather than hard-coded: this
    /// backend is meant to load whatever the other two were pointed at, and a
    /// wrong template is a silent quality loss, not an error.
    fn tokenize(&self, messages: &[Message]) -> Result<Vec<LlamaToken>> {
        let template = self
            .model
            .chat_template(None)
            .map_err(|e| anyhow!("llama-local: the model carries no usable chat template - {e}"))?;
        let chat = messages
            .iter()
            .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                anyhow!("llama-local: a message could not be passed to the template - {e}")
            })?;
        let prompt = self
            .model
            .apply_chat_template(&template, &chat, true)
            .map_err(|e| anyhow!("llama-local: the chat template failed - {e}"))?;

        // The template writes its own special tokens as text and `str_to_token`
        // parses them, so BOS must NOT be added blindly - Gemma's template emits
        // `<bos>` itself, and a second one shifts every position by a token the
        // model never saw in training. Add it only when the template did not.
        let mut tokens = self
            .model
            .str_to_token(&prompt, AddBos::Never)
            .map_err(|e| anyhow!("llama-local: the prompt could not be tokenized - {e}"))?;
        let bos = self.model.token_bos();
        if bos.0 >= 0 && tokens.first() != Some(&bos) {
            tokens.insert(0, bos);
        }
        Ok(tokens)
    }

    /// Decode `tokens[from..]` into `seq`, asking for logits only on the last
    /// one - the rest are prefill and their logits are never read.
    fn decode_prompt(
        &mut self,
        tokens: &[LlamaToken],
        from: usize,
        seq: i32,
        cfg: &WorkerCfg,
    ) -> Result<()> {
        let chunk = cfg.n_batch as usize;
        let mut batch = LlamaBatch::new(chunk, 1);
        let mut i = from;
        while i < tokens.len() {
            let end = (i + chunk).min(tokens.len());
            batch.clear();
            for (k, token) in tokens[i..end].iter().enumerate() {
                batch
                    .add(*token, (i + k) as i32, &[seq], i + k == tokens.len() - 1)
                    .map_err(|e| anyhow!("llama-local: prompt batch overflow - {e}"))?;
            }
            self.ctx
                .decode(&mut batch)
                .map_err(|e| anyhow!("llama-local: prefill failed - {e}"))?;
            i = end;
        }
        Ok(())
    }

    /// Make `id` the conversation that owns `SEQ_CHAT`.
    ///
    /// Without this, every conversation shares one sequence, so switching to
    /// another one finds a prefix no longer than the persona the two happen to
    /// have in common and re-decodes the rest of the window. Parking serializes
    /// the outgoing cache to host memory and puts the incoming one back, which
    /// turns a switch from a prefill into a memcpy.
    ///
    /// Failing to park or to restore costs a prefill and nothing else, so both
    /// are swallowed - a turn that refuses to start because a cache could not be
    /// saved would be strictly worse than a slow one. Failing to CLEAR is not in
    /// that class: decoding from position zero into a sequence that still holds
    /// another conversation's cells produces a confident, wrong reply, so that
    /// one is returned.
    fn switch_to(&mut self, id: Option<&str>) -> Result<()> {
        if !self.dirty && self.resident_id.as_deref() == id {
            return Ok(());
        }
        self.park_current();
        // Whatever is still in the sequence now belongs to somebody else, or to
        // nobody. Either way it is not this turn's.
        self.clear_chat_seq()?;
        self.resident.clear();
        self.resident_id = id.map(str::to_owned);

        let Some(id) = id else { return Ok(()) };
        let Some(entry) = self.parked.take(id) else {
            return Ok(());
        };
        let restored = self.ctx.state_seq_set(&entry.state, SEQ_CHAT);
        match restored {
            Ok(()) => {
                self.resident = entry.tokens;
                tracing::debug!(
                    conversation = %id,
                    tokens = self.resident.len(),
                    "llama-local: KV cache restored"
                );
            }
            Err(e) => {
                // The cells are in an unknown state after a partial read, so
                // drop them and take the full prefill.
                self.clear_chat_seq()?;
                warn_once(
                    "local_kv_restore",
                    &format!(
                        "llama-local: a parked KV cache could not be restored ({e}); it re-prefills."
                    ),
                );
            }
        }
        Ok(())
    }

    /// Drop every cell `SEQ_CHAT` holds, and record whether it worked.
    fn clear_chat_seq(&mut self) -> Result<()> {
        match self
            .ctx
            .clear_kv_cache_seq(Some(SEQ_CHAT as u32), None, None)
        {
            Ok(_) => {
                self.dirty = false;
                Ok(())
            }
            Err(e) => {
                self.dirty = true;
                Err(anyhow!(
                    "llama-local: the KV cache could not be cleared - {e}"
                ))
            }
        }
    }

    /// Serialize whatever `SEQ_CHAT` holds into the cache, so the cells can be
    /// handed to the next conversation without losing the work that filled them.
    fn park_current(&mut self) {
        let Some(id) = self.resident_id.take() else {
            return;
        };
        if self.resident.is_empty() || self.parked.budget == 0 {
            return;
        }
        let captured = self
            .ctx
            .state_seq_get(SEQ_CHAT, LlamaStateSeqFlags::empty());
        match captured {
            Ok(state) => {
                let bytes = state.byte_len();
                let tokens = std::mem::take(&mut self.resident);
                let entry = Parked {
                    id,
                    tokens,
                    state,
                    bytes,
                };
                if !self.parked.put(entry) {
                    warn_once(
                        "local_kv_budget",
                        &format!(
                            concat!(
                                "llama-local: a {} MiB KV cache is larger than ",
                                "LLAMA_KV_CACHE_MB, so that conversation re-prefills on every ",
                                "switch. Raise the budget, or lower LLAMA_NUM_CTX."
                            ),
                            bytes / (1024 * 1024)
                        ),
                    );
                }
            }
            Err(e) => warn_once(
                "local_kv_park",
                &format!("llama-local: a KV cache could not be parked ({e}); it re-prefills."),
            ),
        }
    }

    fn chat(
        &mut self,
        messages: Vec<Message>,
        sampling: &Sampling,
        conversation: Option<String>,
        cfg: &WorkerCfg,
        out: &UnboundedSender<StreamEvent>,
    ) -> Result<()> {
        let started = Instant::now();
        let tokens = self.tokenize(&messages)?;
        if tokens.is_empty() {
            // `tokenize` prepends BOS, so this needs a model that has no BOS
            // token AND a template that rendered nothing. Guarded rather than
            // assumed away: the prefix clamp below subtracts one from the
            // length, and on an empty prompt that underflows.
            return Err(anyhow!(
                "llama-local: the chat template produced an empty prompt."
            ));
        }
        let n_ctx = self.ctx.n_ctx() as usize;
        if tokens.len() + 16 >= n_ctx {
            // Refusing beats truncating: the caller trimmed history to fit a
            // window it believes in, and dropping the front of the prompt
            // silently removes the persona first.
            return Err(anyhow!(
                "llama-local: the prompt is {} tokens and the window is {n_ctx}. Lower the history \
                 budget or raise LLAMA_NUM_CTX.",
                tokens.len()
            ));
        }

        // After the window check, not before: a prompt this engine refuses must
        // not cost the conversation that is currently resident its cache.
        self.switch_to(conversation.as_deref())?;

        // Measured, not estimated. One token is always re-decoded: logits come
        // out of a decode, so reusing the whole prompt would leave nothing to
        // sample from.
        let reuse = common_prefix(&self.resident, &tokens).min(tokens.len() - 1);
        self.ctx
            .clear_kv_cache_seq(Some(SEQ_CHAT as u32), Some(reuse as u32), None)
            .map_err(|e| anyhow!("llama-local: could not trim the KV cache - {e}"))?;
        // Anything past the reused prefix is gone. Say so before the next
        // fallible call, or a failure there leaves `resident` lying about what
        // the cache holds and the turn after it reuses tokens that are not real.
        self.resident.truncate(reuse);

        self.decode_prompt(&tokens, reuse, SEQ_CHAT, cfg)?;
        self.resident.clone_from(&tokens);
        let prompt_ns = started.elapsed().as_nanos() as u64;

        let eval_started = Instant::now();
        let mut sampler = sampling.chain(self.model.n_vocab());
        // The penalty window has to see the prompt, or a repeat penalty measured
        // against an empty history is not the sampler the other backends run.
        // This is what llama.cpp's own server does.
        sampler.accept_many(&tokens);

        let mut emitter = Emitter::new(sampling.stop.clone());
        let mut batch = LlamaBatch::new(1, 1);
        let mut produced: u64 = 0;
        let mut done_reason = "stop";

        loop {
            let token = sampler.sample(&self.ctx, -1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }

            let bytes = self.piece(token)?;
            if let Some(text) = emitter.push(&bytes) {
                let raw = ollama_chunk(&self.name, &text);
                if out.send(StreamEvent::Chunk { raw, delta: text }).is_err() {
                    // The receiver is gone: the user stopped, or navigated away.
                    // Everything decoded so far stays in the KV cache, which is
                    // the point - the next turn reuses it.
                    done_reason = "abort";
                    break;
                }
            }
            if emitter.hit_stop() {
                break;
            }

            produced += 1;
            if sampling.num_predict > 0 && produced >= sampling.num_predict as u64 {
                done_reason = "length";
                break;
            }
            let pos = self.resident.len();
            if pos >= n_ctx {
                done_reason = "length";
                break;
            }
            self.resident.push(token);
            batch.clear();
            batch
                .add(token, pos as i32, &[SEQ_CHAT], true)
                .map_err(|e| anyhow!("llama-local: decode batch overflow - {e}"))?;
            self.ctx
                .decode(&mut batch)
                .map_err(|e| anyhow!("llama-local: decode failed - {e}"))?;
        }

        if done_reason != "abort" {
            if let Some(rest) = emitter.flush() {
                let raw = ollama_chunk(&self.name, &rest);
                let _ = out.send(StreamEvent::Chunk { raw, delta: rest });
            }
        }

        let eval_ns = eval_started.elapsed().as_nanos() as u64;
        let stats = DoneStats {
            done_reason: Some(done_reason.to_string()),
            // Same convention as the llama-server adapter: the WHOLE prompt,
            // reused tokens included, so the number that sizes the window does
            // not shrink by whatever the cache absorbed.
            prompt_eval_count: Some(tokens.len() as u64),
            prompt_eval_duration: Some(prompt_ns),
            eval_count: Some(produced),
            eval_duration: Some(eval_ns),
            total_duration: Some(prompt_ns + eval_ns),
            // The weights are resident for the life of the process. A non-zero
            // value here would mean something reloaded them behind our back.
            load_duration: Some(0),
            // This many tokens were already in the KV cache and were not decoded
            // again - the same field llama-server fills from `timings.cache_n`.
            cached_tokens: Some(reuse as u64),
        };
        let _ = out.send(StreamEvent::Done {
            raw: ollama_done(&self.name, &stats),
            stats: Box::new(stats),
        });
        Ok(())
    }

    /// The summarizer. Same weights, its own KV sequence, cleared afterwards.
    fn summarize(&mut self, prompt: &str, cfg: &WorkerCfg) -> Result<String> {
        let tokens = self.tokenize(&[Message::new("user", prompt)])?;
        let n_ctx = self.ctx.n_ctx() as usize;
        if tokens.len() + 16 >= n_ctx {
            return Err(anyhow!(
                "llama-local: the summary prompt is {} tokens and the window is {n_ctx}.",
                tokens.len()
            ));
        }
        // Start clean every time: this prefix differs on every call, so there is
        // nothing to reuse and a stale tail would be read as context.
        let _ = self
            .ctx
            .clear_kv_cache_seq(Some(SEQ_AUX as u32), None, None);
        let result = self.summarize_inner(&tokens, cfg);
        // Give the memory back whether or not it worked - the story's cache is
        // sharing this context.
        let _ = self
            .ctx
            .clear_kv_cache_seq(Some(SEQ_AUX as u32), None, None);
        result
    }

    fn summarize_inner(&mut self, tokens: &[LlamaToken], cfg: &WorkerCfg) -> Result<String> {
        let sampling = Sampling::for_summary();
        self.decode_prompt(tokens, 0, SEQ_AUX, cfg)?;
        let mut sampler = sampling.chain(self.model.n_vocab());
        sampler.accept_many(tokens);

        let mut emitter = Emitter::new(Vec::new());
        let mut batch = LlamaBatch::new(1, 1);
        let mut pos = tokens.len() as i32;
        // A summary that never stops would hold the engine for the rest of the
        // window. The caller's budget is prose-sized; this is the hard stop.
        let budget = (self.ctx.n_ctx() as i32 - pos).min(2048);

        for _ in 0..budget {
            let token = sampler.sample(&self.ctx, -1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            let bytes = self.piece(token)?;
            let _ = emitter.push(&bytes);
            batch.clear();
            batch
                .add(token, pos, &[SEQ_AUX], true)
                .map_err(|e| anyhow!("llama-local: summary batch overflow - {e}"))?;
            self.ctx
                .decode(&mut batch)
                .map_err(|e| anyhow!("llama-local: summary decode failed - {e}"))?;
            pos += 1;
        }
        let _ = emitter.flush();
        Ok(emitter.finished().trim().to_string())
    }

    /// One token's bytes. 32 covers every piece in a normal vocabulary; the
    /// retry is for the long added tokens some fine-tunes carry.
    fn piece(&self, token: LlamaToken) -> Result<Vec<u8>> {
        self.model
            .token_to_piece_bytes(token, 32, false, None)
            .or_else(|_| self.model.token_to_piece_bytes(token, 1024, false, None))
            .map_err(|e| anyhow!("llama-local: a token could not be decoded - {e}"))
    }
}

/// How many leading tokens two sequences share.
fn common_prefix(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts(v: Json) -> Map<String, Json> {
        v.as_object().expect("an object").clone()
    }

    #[test]
    fn ollama_named_options_land_on_the_llama_cpp_fields() {
        let mut s = Sampling::default();
        s.apply(&opts(json!({
            "temperature": 0.9, "top_p": 0.8, "top_k": 30, "min_p": 0.02,
            "repeat_penalty": 1.1, "repeat_last_n": 128, "num_predict": 256,
            "stop": ["\nUser:"]
        })));
        assert_eq!(s.temperature, 0.9);
        assert_eq!(s.top_k, 30);
        assert_eq!(s.repeat_last_n, 128);
        assert_eq!(s.num_predict, 256);
        assert_eq!(s.stop, vec!["\nUser:".to_string()]);
    }

    #[test]
    fn the_request_wins_over_the_modelfile_because_it_is_applied_second() {
        let mut s = Sampling::default();
        s.apply(&opts(json!({ "temperature": 0.9 })));
        s.apply(&opts(json!({ "temperature": 0.2 })));
        assert_eq!(s.temperature, 0.2);
    }

    #[test]
    fn a_null_option_does_not_overwrite_a_set_value() {
        let mut s = Sampling::default();
        s.apply(&opts(json!({ "temperature": 0.9 })));
        s.apply(&opts(json!({ "temperature": null })));
        assert_eq!(s.temperature, 0.9);
    }

    #[test]
    fn an_unknown_option_is_dropped_and_leaves_everything_else_alone() {
        let mut s = Sampling::default();
        s.apply(&opts(json!({ "mirostat": 2, "temperature": 0.4 })));
        assert_eq!(
            s,
            Sampling {
                temperature: 0.4,
                ..Sampling::default()
            }
        );
    }

    #[test]
    fn a_stop_string_can_arrive_as_one_string_or_as_a_list() {
        let mut s = Sampling::default();
        s.apply(&opts(json!({ "stop": "END" })));
        assert_eq!(s.stop, vec!["END".to_string()]);
        s.apply(&opts(json!({ "stop": ["A", "B"] })));
        assert_eq!(s.stop, vec!["A".to_string(), "B".to_string()]);
    }

    #[test]
    fn the_summary_sampler_is_low_variance_and_not_the_roleplay_one() {
        let s = Sampling::for_summary();
        assert!(s.temperature < Sampling::default().temperature);
        assert_eq!(s.temperature, 0.3);
    }

    #[test]
    fn an_unknown_kv_type_is_rejected_rather_than_silently_f16() {
        assert!(kv_type("q8_0").is_some());
        assert!(kv_type("").is_some());
        assert!(kv_type("q3_k").is_none());
    }

    #[test]
    fn a_character_split_across_two_tokens_is_never_emitted_in_halves() {
        let mut e = Emitter::new(Vec::new());
        assert_eq!(e.push(b"caf").as_deref(), Some("caf"));
        assert_eq!(e.push(&[0xC3]), None);
        assert_eq!(e.push(&[0xA9]).as_deref(), Some("\u{e9}"));
        assert_eq!(e.finished(), "caf\u{e9}");
    }

    #[test]
    fn not_one_byte_of_a_stop_string_is_ever_emitted() {
        // The stop lands exactly at a token boundary, which is the case a fixed
        // "longest stop minus one" window gets wrong.
        let mut e = Emitter::new(vec!["\nUser:".to_string()]);
        let mut seen = String::new();
        for piece in ["hello", "\nUser", ":"] {
            if let Some(out) = e.push(piece.as_bytes()) {
                seen.push_str(&out);
            }
        }
        assert!(e.hit_stop());
        assert_eq!(seen, "hello");
        assert_eq!(e.flush(), None);
        assert_eq!(e.finished(), "hello");
    }

    #[test]
    fn a_stop_that_never_completes_releases_what_it_was_holding() {
        let mut e = Emitter::new(vec!["\nUser:".to_string()]);
        assert_eq!(e.push(b"hello").as_deref(), Some("hello"));
        // Looks like the start of the stop, so it waits.
        assert_eq!(e.push(b"\nUse"), None);
        // It was not. Nothing is lost and nothing arrives out of order.
        assert_eq!(e.push(b"d to be").as_deref(), Some("\nUsed to be"));
        assert!(!e.hit_stop());
    }

    #[test]
    fn text_that_could_not_begin_a_stop_is_not_held_back_at_all() {
        let mut e = Emitter::new(vec!["END".to_string()]);
        assert_eq!(
            e.push(b"a long stretch of prose").as_deref(),
            Some("a long stretch of prose")
        );
    }

    #[test]
    fn an_invalid_byte_does_not_wedge_the_buffer() {
        let mut e = Emitter::new(Vec::new());
        assert_eq!(e.push(&[0xFF, 111, 107, 33]).as_deref(), Some("ok!"));
    }

    // ---- Parked KV caches ------------------------------------------------
    //
    // The eviction policy is tested on its own rather than through a loaded
    // model: the payload is opaque to the cache, so a unit that never links
    // llama.cpp proves everything the policy is responsible for.

    fn parked(id: &str, bytes: usize) -> Parked<()> {
        Parked {
            id: id.into(),
            tokens: Vec::new(),
            state: (),
            bytes,
        }
    }

    fn ids<S>(c: &ParkedCache<S>) -> Vec<&str> {
        c.entries.iter().map(|e| e.id.as_str()).collect()
    }

    #[test]
    fn the_oldest_parked_conversation_is_the_one_evicted() {
        let mut c = ParkedCache::new(300);
        assert!(c.put(parked("a", 100)));
        assert!(c.put(parked("b", 100)));
        assert!(c.put(parked("c", 100)));
        assert_eq!(ids(&c), ["a", "b", "c"]);
        assert_eq!(c.bytes, 300);

        assert!(c.put(parked("d", 100)));
        assert_eq!(ids(&c), ["b", "c", "d"], "a was the oldest");
        assert_eq!(c.bytes, 300);
    }

    #[test]
    fn one_large_cache_evicts_as_many_small_ones_as_it_needs() {
        let mut c = ParkedCache::new(300);
        c.put(parked("a", 50));
        c.put(parked("b", 50));
        c.put(parked("c", 50));
        assert!(c.put(parked("big", 250)));
        assert_eq!(ids(&c), ["c", "big"]);
        assert_eq!(c.bytes, 300);
    }

    #[test]
    fn re_parking_replaces_rather_than_double_counting() {
        // The same conversation is parked again every time the user leaves it,
        // and a second entry would both leak bytes and let a stale state win the
        // lookup.
        let mut c = ParkedCache::new(300);
        c.put(parked("a", 100));
        c.put(parked("a", 120));
        assert_eq!(ids(&c), ["a"]);
        assert_eq!(c.bytes, 120);
    }

    #[test]
    fn taking_a_cache_out_returns_its_bytes_to_the_budget() {
        let mut c = ParkedCache::new(300);
        c.put(parked("a", 100));
        c.put(parked("b", 100));
        let got = c.take("a").expect("a was parked");
        assert_eq!(got.bytes, 100);
        assert_eq!(c.bytes, 100);
        assert!(c.take("a").is_none(), "taking is not a peek");
        assert_eq!(ids(&c), ["b"]);
    }

    #[test]
    fn a_cache_bigger_than_the_whole_budget_evicts_nothing() {
        // The important half is the eviction: throwing away three usable caches
        // to make room for something that will not be kept is strictly worse
        // than refusing.
        let mut c = ParkedCache::new(300);
        c.put(parked("a", 100));
        c.put(parked("b", 100));
        assert!(!c.put(parked("huge", 400)));
        assert_eq!(ids(&c), ["a", "b"]);
        assert_eq!(c.bytes, 200);
    }

    #[test]
    fn taking_a_conversation_that_was_never_parked_leaves_the_rest_alone() {
        let mut c = ParkedCache::new(100);
        assert!(c.put(parked("a", 10)));
        assert!(c.take("never-here").is_none());
        assert_eq!(ids(&c), ["a"]);
        assert_eq!(c.bytes, 10);
    }

    #[test]
    fn a_zero_budget_keeps_nothing() {
        let mut c = ParkedCache::new(0);
        assert!(!c.put(parked("a", 1)));
        assert!(c.entries.is_empty());
        assert_eq!(c.bytes, 0);
    }

    #[test]
    fn a_reparked_conversation_is_the_newest_again() {
        // Re-parking has to refresh the position too, or the conversation the
        // user keeps coming back to becomes the one evicted first.
        let mut c = ParkedCache::new(300);
        c.put(parked("a", 100));
        c.put(parked("b", 100));
        c.put(parked("a", 100));
        assert_eq!(ids(&c), ["b", "a"]);
        c.put(parked("d", 200));
        assert_eq!(ids(&c), ["a", "d"], "b was the oldest by then");
    }

    #[test]
    fn the_reusable_prefix_is_the_shared_head_and_nothing_after_it() {
        let a: Vec<LlamaToken> = [1, 2, 3, 4].iter().map(|n| LlamaToken(*n)).collect();
        let b: Vec<LlamaToken> = [1, 2, 9, 4].iter().map(|n| LlamaToken(*n)).collect();
        assert_eq!(common_prefix(&a, &b), 2);
        assert_eq!(common_prefix(&a, &a), 4);
        assert_eq!(common_prefix(&a, &[]), 0);
    }
}
