//! Measure the in-process engine on this machine.
//!
//! Decision 0001 stage 4b picked llama.cpp in-process over the llama-server HTTP
//! hop, and over Ollama, on the argument that removing the hop and keeping the
//! KV cache in the same process is faster. That argument has never had a number
//! attached to it on this hardware, and a stage that ships on an unmeasured
//! claim is a stage that cannot be defended later.
//!
//! So this reports four things, in the order they matter:
//!
//!   1. Cold prefill - tokens/second decoding a prompt the engine has not seen.
//!   2. Decode - tokens/second generating, which is what the reader watches.
//!   3. Warm prefill - the same conversation one turn later, where the shared
//!      prefix is already resident and only the new tail is decoded.
//!   4. Restored prefill - the same conversation after another one took the
//!      context away and gave it back. This is what LLAMA_KV_CACHE_MB buys; if
//!      it does not land near the warm number, parking is not working.
//!
//! Run it with the toolchain wrapper, which is the only thing on this machine
//! that hands MSVC the right headers:
//!
//!   powershell -ExecutionPolicy Bypass -File src-core\build-local-llama.ps1 \
//!       -Cuda -Release -Command run -Example bench-local-llama
//!
//! It reads the same environment as the app (LLAMA_CHAT_MODEL / OLLAMA_MODEL,
//! LLAMA_NUM_CTX, LLAMA_N_GPU_LAYERS, LLAMA_KV_CACHE_MB), so what it measures is
//! the configuration that would actually run.

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use serde_json::{Map, Value as Json};
use vessel_core::inference::{ChatStart, Engine, Message, StreamEvent};

/// Big enough that prefill is the dominant cost and a cache hit is unmistakable,
/// small enough to leave room for the reply inside a default window.
const PROMPT_PARAGRAPHS: usize = 60;

/// Long enough that the first few tokens' warm-up stops dominating the rate -
/// a 64-token reply measured over 18 tokens is a warm-up measurement wearing a
/// decode label. Short enough that four turns still finish in a minute.
const REPLY_TOKENS: u64 = 128;

#[derive(Default)]
struct Timing {
    prompt_tokens: u64,
    prompt_ms: f64,
    eval_tokens: u64,
    eval_ms: f64,
}

impl Timing {
    fn prefill_tps(&self) -> f64 {
        rate(self.prompt_tokens, self.prompt_ms)
    }
    fn decode_tps(&self) -> f64 {
        rate(self.eval_tokens, self.eval_ms)
    }
}

fn rate(tokens: u64, ms: f64) -> f64 {
    if ms <= 0.0 {
        0.0
    } else {
        tokens as f64 * 1000.0 / ms
    }
}

fn ms(ns: Option<u64>) -> f64 {
    ns.unwrap_or(0) as f64 / 1e6
}

/// A prompt with a long, stable head and a short, unique tail - the shape a real
/// conversation has, and the only shape that tells a prefix hit from a miss.
fn prompt(turn: usize) -> Vec<Message> {
    let head = "The lamp-keeper walked the seawall every night, counting the \
                lights that still burned and noting the ones that did not. The \
                ledger was older than the town, and every hand that had kept it \
                wrote smaller than the last.\n"
        .repeat(PROMPT_PARAGRAPHS);
    vec![
        // Deliberately asks for length: the reply has to outrun warm-up, or
        // the decode rate is measuring the first tokens rather than the rate.
        Message::new(
            "system",
            "You are a careful narrator. Answer at length, in full paragraphs.",
        ),
        Message::new("user", head),
        Message::new(
            "user",
            format!("Question {turn}: what did the keeper count?"),
        ),
    ]
}

fn options(conversation: &str) -> Map<String, Json> {
    let mut o = Map::new();
    o.insert("conversation_id".into(), Json::from(conversation));
    o.insert("num_predict".into(), Json::from(REPLY_TOKENS));
    // Greedy: the numbers should not move between runs because a sampler rolled
    // differently.
    o.insert("temperature".into(), Json::from(0.0));
    o
}

async fn turn(engine: &impl Engine, conversation: &str, n: usize) -> Result<Timing> {
    let start = match engine
        .chat_stream(prompt(n), &options(conversation))
        .await?
    {
        ChatStart::Refused { status, detail } => {
            return Err(anyhow!("refused with {status}: {detail}"))
        }
        ChatStart::Streaming(s) => s,
    };
    let mut stream = start;
    let mut timing = Timing::default();
    while let Some(ev) = stream.next().await {
        match ev {
            StreamEvent::Chunk { .. } => {}
            StreamEvent::Error { message } => return Err(anyhow!(message)),
            StreamEvent::Done { stats, .. } => {
                timing.prompt_tokens = stats.prompt_eval_count.unwrap_or(0);
                timing.prompt_ms = ms(stats.prompt_eval_duration);
                timing.eval_tokens = stats.eval_count.unwrap_or(0);
                timing.eval_ms = ms(stats.eval_duration);
            }
        }
    }
    Ok(timing)
}

fn row(label: &str, t: &Timing) {
    println!(
        "{label:<18} {:>7} tok {:>9.0} ms {:>9.1} tok/s   | decode {:>5} tok {:>8.0} ms {:>7.1} tok/s",
        t.prompt_tokens,
        t.prompt_ms,
        t.prefill_tps(),
        t.eval_tokens,
        t.eval_ms,
        t.decode_tps(),
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    // The app loads .env the same way; without it the model name and the layer
    // count would be defaults rather than the running configuration.
    let _ = dotenvy::dotenv();

    let engine = vessel_core::inference::llama_local::LlamaLocal::from_env()?;
    println!("{}\n", serde_json::to_string_pretty(&engine.describe())?);
    println!(
        "{:<18} {:>11} {:>12} {:>14}   | {}",
        "phase", "prefill", "", "", "decode"
    );

    // 1 + 2. Nothing is resident, so this prefill is the whole prompt.
    let cold = turn(&engine, "bench-a", 1).await?;
    row("cold", &cold);

    // 3. Same conversation, same head, new tail: only the tail should decode.
    let warm = turn(&engine, "bench-a", 2).await?;
    row("warm (prefix)", &warm);

    // Another conversation takes the context. Its own numbers are not the point;
    // what matters is what happens to bench-a afterwards.
    let other = turn(&engine, "bench-b", 1).await?;
    row("other conv", &other);

    // 4. Back to bench-a. Without parking this is a cold prefill again.
    let restored = turn(&engine, "bench-a", 3).await?;
    row("restored", &restored);

    println!();
    let saved = cold.prefill_tps();
    if saved > 0.0 {
        println!(
            "prefix reuse cut prefill to {:.0}% of cold; a restored cache to {:.0}%.",
            100.0 * warm.prompt_ms / cold.prompt_ms.max(f64::MIN_POSITIVE),
            100.0 * restored.prompt_ms / cold.prompt_ms.max(f64::MIN_POSITIVE),
        );
    }
    // The budget read-out, straight from the engine thread. If `restored`
    // matched `cold` rather than `warm`, this is the line that says why: a
    // parked size at or over the budget means the cache never fit.
    if let Some(info) = engine.probe().await {
        println!(
            "parked: {} conversation(s), {} MiB of a {} MiB budget; {} token(s) resident.",
            info["parkedConversations"].as_u64().unwrap_or(0),
            info["parkedBytes"].as_u64().unwrap_or(0) / (1024 * 1024),
            info["parkedBudgetBytes"].as_u64().unwrap_or(0) / (1024 * 1024),
            info["residentTokens"].as_u64().unwrap_or(0),
        );
    }
    Ok(())
}
