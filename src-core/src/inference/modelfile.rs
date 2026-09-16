//! Reads the repo `Modelfile`.
//!
//! On Ollama, `SYSTEM` and every `PARAMETER` line there are baked into the model
//! at `ollama create` time: the server applies them to every request whether or
//! not the client sends anything. llama-server loads the raw GGUF blob and knows
//! nothing about any of it - the blob carries weights and a chat template, and
//! that is all.
//!
//! So a naive backend swap silently drops the global roleplay system prompt and
//! the sampling defaults, and the model starts refusing, moralizing, and writing
//! for the user. That is not a transport difference; it is a different product.
//!
//! The fix is to read the same file Ollama read, so there is still exactly one
//! source of truth for what the model is.

use serde_json::{Map, Value as Json};
use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub struct Modelfile {
    pub from: Option<String>,
    pub system: Option<String>,
    pub template: Option<String>,
    pub params: Map<String, Json>,
    pub path: PathBuf,
    pub found: bool,
    pub error: Option<String>,
}

/// The repo root, relative to the compiled crate. `CARGO_MANIFEST_DIR` is
/// `src-core`, so the Modelfile sits one level up - the same file the Electron
/// build reads from three levels up its own tree.
fn default_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join("Modelfile"))
        .unwrap_or_else(|| PathBuf::from("Modelfile"))
}

fn unquote(s: &str) -> String {
    let t = s.trim();
    let b = t.as_bytes();
    if b.len() >= 2 && b[0] == b'"' && b[b.len() - 1] == b'"' {
        return t[1..t.len() - 1].replace("\\n", "\n").replace("\\\"", "\"");
    }
    t.to_string()
}

/// A bare token that looks like a number becomes one; everything else stays a
/// string. The shape test matters: `1.1.1` parses as nothing, and `0x10` must
/// not become 16, so the characters are checked before the parse.
fn coerce(v: &str) -> Json {
    let t = unquote(v);
    if t.is_empty() {
        return Json::String(t);
    }
    let numeric = t
        .chars()
        .all(|c| c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E'));
    if numeric {
        if let Ok(n) = t.parse::<f64>() {
            if let Some(n) = serde_json::Number::from_f64(n) {
                return Json::Number(n);
            }
        }
    }
    Json::String(t)
}

/// Pull the triple-quoted block directives out first, so a `#` or a
/// PARAMETER-looking line inside the system prompt is not mistaken for markup.
/// Returns the text with those blocks removed.
fn take_blocks(text: &str, out: &mut Modelfile) -> String {
    let mut rest = String::with_capacity(text.len());
    let mut cur = text;
    loop {
        let Some((head, key, after_key)) = find_block_head(cur) else {
            rest.push_str(cur);
            return rest;
        };
        let Some(end) = after_key.find("\"\"\"") else {
            rest.push_str(cur);
            return rest;
        };
        rest.push_str(head);
        let body = &after_key[..end];
        match key.as_str() {
            "SYSTEM" => out.system = Some(body.trim().to_string()),
            "TEMPLATE" => out.template = Some(body.to_string()),
            _ => {} // ADAPTER: recognised so its body cannot be parsed as lines.
        }
        cur = &after_key[end + 3..];
    }
}

/// Finds the next block directive opening at a line start. Returns the text
/// before it, the uppercased key, and the text after the opening quotes.
fn find_block_head(text: &str) -> Option<(&str, String, &str)> {
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let start = offset;
        offset += line.len();
        let t = line.trim_start();
        let indent = line.len() - t.len();
        for key in ["SYSTEM", "TEMPLATE", "ADAPTER"] {
            if t.len() < key.len() || !t[..key.len()].eq_ignore_ascii_case(key) {
                continue;
            }
            let after = &t[key.len()..];
            let trimmed = after.trim_start();
            if !trimmed.starts_with("\"\"\"") {
                continue;
            }
            let open = start + indent + key.len() + (after.len() - trimmed.len()) + 3;
            return Some((&text[..start], key.to_string(), &text[open..]));
        }
    }
    None
}

pub fn parse(text: &str) -> Modelfile {
    let mut out = Modelfile::default();
    let rest = take_blocks(text, &mut out);

    for line in rest.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if let Some(v) = strip_keyword(l, "FROM") {
            out.from = Some(unquote(v));
            continue;
        }
        if let Some(v) = strip_keyword(l, "SYSTEM") {
            out.system = Some(unquote(v));
            continue;
        }
        if let Some(v) = strip_keyword(l, "TEMPLATE") {
            out.template = Some(unquote(v));
            continue;
        }
        if let Some(v) = strip_keyword(l, "PARAMETER") {
            let mut it = v.splitn(2, char::is_whitespace);
            let (Some(key), Some(val)) = (it.next(), it.next()) else {
                continue;
            };
            let key = key.to_ascii_lowercase();
            let val = coerce(val.trim());
            // `stop` is the one key Ollama allows more than once; it accumulates
            // and is always a list, so consumers never have to branch on arity.
            // Every other key is last-wins, as Ollama treats it. Accumulating
            // those too turned a duplicated line into `temperature: [0.5, 0.9]`,
            // which the llama-server adapter spreads straight into the request.
            if key == "stop" {
                match out.params.entry(key).or_insert_with(|| Json::Array(vec![])) {
                    Json::Array(a) => a.push(val),
                    slot => *slot = Json::Array(vec![slot.take(), val]),
                }
            } else {
                out.params.insert(key, val);
            }
            continue;
        }
    }
    out
}

/// `KEYWORD <value>` at a line start, case-insensitive, with at least one space.
fn strip_keyword<'a>(line: &'a str, keyword: &str) -> Option<&'a str> {
    if line.len() <= keyword.len() || !line[..keyword.len()].eq_ignore_ascii_case(keyword) {
        return None;
    }
    let rest = &line[keyword.len()..];
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let v = rest.trim();
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

pub fn load() -> &'static Modelfile {
    static LOADED: std::sync::OnceLock<Modelfile> = std::sync::OnceLock::new();
    LOADED.get_or_init(|| {
        let p = std::env::var("LLAMA_MODELFILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_path());
        match std::fs::read_to_string(&p) {
            Ok(text) => Modelfile {
                path: p,
                found: true,
                ..parse(&text)
            },
            Err(e) => Modelfile {
                path: p,
                found: false,
                error: Some(e.to_string()),
                ..Default::default()
            },
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SAMPLE: &str = concat!(
        "FROM some/model:8b\n",
        "# a comment\n",
        "PARAMETER num_ctx 32768\n",
        "PARAMETER temperature 0.9\n",
        "PARAMETER stop \"<eot>\"\n",
        "PARAMETER stop \"<end>\"\n",
        "SYSTEM \"\"\"Line one.\n",
        "# not a comment\n",
        "PARAMETER not_a_param 1\n",
        "\"\"\"\n",
    );

    #[test]
    fn reads_from_parameters_and_the_block_system_prompt() {
        let m = parse(SAMPLE);
        assert_eq!(m.from.as_deref(), Some("some/model:8b"));
        assert_eq!(m.params.get("num_ctx"), Some(&json!(32768.0)));
        assert_eq!(m.params.get("temperature"), Some(&json!(0.9)));
        let sys = m.system.unwrap();
        assert!(sys.starts_with("Line one."));
        assert!(sys.contains("# not a comment"), "{sys}");
    }

    #[test]
    fn a_parameter_line_inside_the_system_block_is_prose_not_markup() {
        let m = parse(SAMPLE);
        assert!(!m.params.contains_key("not_a_param"));
    }

    #[test]
    fn stop_accumulates_and_everything_else_is_last_wins() {
        let m = parse(SAMPLE);
        assert_eq!(m.params.get("stop"), Some(&json!(["<eot>", "<end>"])));
        let m2 = parse("PARAMETER temperature 0.5\nPARAMETER temperature 0.9\n");
        assert_eq!(m2.params.get("temperature"), Some(&json!(0.9)));
    }

    #[test]
    fn the_real_modelfile_carries_the_tuned_sampling() {
        // Guards the actual file, because dropping these silently changes what
        // the model is on the llama-server backend.
        let m = load();
        assert!(
            m.found,
            "Modelfile not found at {:?}: {:?}",
            m.path, m.error
        );
        for key in ["num_ctx", "temperature", "top_p", "min_p", "repeat_penalty"] {
            assert!(m.params.contains_key(key), "missing PARAMETER {key}");
        }
        assert!(m
            .system
            .as_deref()
            .unwrap_or("")
            .contains("ONLY your own character"));
    }
}
