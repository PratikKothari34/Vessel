//! Finding the GGUF file the in-process engine loads (stage 4b of decision 0001).
//!
//! The HTTP backends never needed this: Ollama and llama-server both own their
//! model store and take a NAME. An in-process llama.cpp takes a PATH, and this
//! repo deliberately stores none - `.env.example` documents `llama-server -m
//! <model.gguf>` and leaves the path to the operator.
//!
//! So there are two ways to answer "which file":
//!
//! 1. `LLAMA_GGUF` - an explicit path. Always wins.
//! 2. The Ollama blob store, keyed by the same model name the other two backends
//!    already use (`LLAMA_CHAT_MODEL` / `OLLAMA_MODEL`).
//!
//! (2) matters because the weights are already on disk and are already the ones
//! the app has been measured against - but Ollama stores them content-addressed,
//! as `blobs/sha256-<64 hex>` with no extension, so the file cannot be found by
//! looking for `*.gguf`. The manifest is what maps a name to a digest.
//!
//! Resolution is pure up to the final `exists` check, so the parsing is tested
//! on every build - including builds without the `local-llama` feature, where
//! none of the engine itself compiles.

use anyhow::{anyhow, Context, Result};
use serde_json::Value as Json;
use std::path::{Path, PathBuf};

/// The layer that holds the weights. Ollama tags the template, the system
/// prompt, the licence and the parameters as separate layers in the same
/// manifest; only this one is a GGUF.
const MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

/// Split an Ollama model reference into (namespace, name, tag).
///
/// `vessel` -> (library, vessel, latest), which is the form the repo uses.
/// A publisher prefix and an explicit tag are both accepted because the store on
/// this machine holds both shapes.
pub fn parse_model_ref(reference: &str) -> (String, String, String) {
    let reference = reference.trim();
    let (namespace, rest) = match reference.split_once('/') {
        Some((ns, rest)) if !ns.is_empty() && !rest.is_empty() => (ns, rest),
        _ => ("library", reference),
    };
    let (name, tag) = match rest.split_once(':') {
        Some((n, t)) if !n.is_empty() && !t.is_empty() => (n, t),
        _ => (rest, "latest"),
    };
    (namespace.to_string(), name.to_string(), tag.to_string())
}

/// The blob FILE NAME for the weights layer of a manifest.
///
/// Ollama writes the digest as `sha256:<hex>` inside the manifest and stores the
/// file as `sha256-<hex>` - a colon is not a legal path character on Windows.
/// Translating that is the whole trick.
///
/// The hex is validated rather than trusted: it becomes a path segment, and a
/// manifest is a file on disk that nothing in this app wrote.
pub fn weights_blob_name(manifest: &str) -> Result<String> {
    let doc: Json = serde_json::from_str(manifest).context("manifest is not JSON")?;
    let layers = doc
        .get("layers")
        .and_then(Json::as_array)
        .ok_or_else(|| anyhow!("manifest has no layers array"))?;
    let digest = layers
        .iter()
        .find(|l| l.get("mediaType").and_then(Json::as_str) == Some(MODEL_MEDIA_TYPE))
        .and_then(|l| l.get("digest"))
        .and_then(Json::as_str)
        .ok_or_else(|| anyhow!("manifest has no {MODEL_MEDIA_TYPE} layer"))?;
    let (algo, hex) = digest
        .split_once(':')
        .ok_or_else(|| anyhow!("digest \"{digest}\" is not <algo>:<hex>"))?;
    let algo_ok = !algo.is_empty() && algo.chars().all(|c| c.is_ascii_alphanumeric());
    let hex_ok = !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit());
    if !algo_ok || !hex_ok {
        return Err(anyhow!("digest \"{digest}\" does not carry a hex hash"));
    }
    Ok(format!("{algo}-{hex}"))
}

/// Root of the Ollama model store. `OLLAMA_MODELS` is Ollama's own override; the
/// default is the directory it creates on install.
fn store_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("OLLAMA_MODELS") {
        if !dir.trim().is_empty() {
            return Some(PathBuf::from(dir.trim()));
        }
    }
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok()?;
    Some(Path::new(&home).join(".ollama").join("models"))
}

/// `<root>/manifests/registry.ollama.ai/<namespace>/<name>/<tag>`
pub fn manifest_path(root: &Path, reference: &str) -> PathBuf {
    let (namespace, name, tag) = parse_model_ref(reference);
    root.join("manifests").join("registry.ollama.ai").join(namespace).join(name).join(tag)
}

/// Resolve the weights path, or say precisely what to do about it.
///
/// The error text is the point of this function. "file not found" on a path the
/// user never typed is unactionable, so both routes are named and the explicit
/// one is offered as the fix.
pub fn resolve(model_ref: &str) -> Result<PathBuf> {
    if let Ok(raw) = std::env::var("LLAMA_GGUF") {
        let path = PathBuf::from(raw.trim());
        if path.as_os_str().is_empty() {
            // Set-but-blank is a half-finished edit, not a request to fall back.
            return Err(anyhow!(
                "LLAMA_GGUF is set but empty. Give it a path to a .gguf file, or unset it."
            ));
        }
        if !path.is_file() {
            return Err(anyhow!("LLAMA_GGUF points at {}, which is not a file.", path.display()));
        }
        return Ok(path);
    }

    let root = store_root().ok_or_else(|| {
        anyhow!("LLAMA_GGUF is unset and no home directory is set, so the Ollama store cannot be found.")
    })?;
    let manifest = manifest_path(&root, model_ref);
    let text = std::fs::read_to_string(&manifest).map_err(|e| {
        anyhow!(
            "LLAMA_GGUF is unset, so the weights were looked up as Ollama model \"{model_ref}\" at \
             {} - {e}. Either pull that model with Ollama, or set LLAMA_GGUF to a .gguf path.",
            manifest.display()
        )
    })?;
    let blob = weights_blob_name(&text).with_context(|| format!("reading {}", manifest.display()))?;
    let path = root.join("blobs").join(&blob);
    if !path.is_file() {
        return Err(anyhow!(
            "Ollama model \"{model_ref}\" names blob {blob}, but {} is missing. The store is \
             incomplete; re-pull it or set LLAMA_GGUF.",
            path.display()
        ));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_name_resolves_to_the_library_namespace_and_latest_tag() {
        assert_eq!(parse_model_ref("vessel"), ("library".into(), "vessel".into(), "latest".into()));
    }

    #[test]
    fn a_publisher_prefix_and_an_explicit_tag_are_both_kept() {
        assert_eq!(
            parse_model_ref("Tohur/natsumura-storytelling-rp-llama-3.1:8b"),
            ("Tohur".into(), "natsumura-storytelling-rp-llama-3.1".into(), "8b".into())
        );
        assert_eq!(parse_model_ref("gemma3:4b"), ("library".into(), "gemma3".into(), "4b".into()));
    }

    #[test]
    fn the_weights_layer_is_picked_out_of_a_manifest_that_has_others() {
        // Shape taken from the store on this machine: the GGUF is neither the
        // first layer nor the last, so anything that guessed by position would
        // pass on some models and fail on others.
        let manifest = r#"{"layers":[
            {"mediaType":"application/vnd.ollama.image.template","digest":"sha256:0046e5db","size":1},
            {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:79e4df2a","size":2},
            {"mediaType":"application/vnd.ollama.image.license","digest":"sha256:5e7fe605","size":3}
        ]}"#;
        assert_eq!(weights_blob_name(manifest).unwrap(), "sha256-79e4df2a");
    }

    #[test]
    fn a_manifest_with_no_weights_layer_is_an_error_not_an_empty_path() {
        let manifest =
            r#"{"layers":[{"mediaType":"application/vnd.ollama.image.license","digest":"sha256:ab"}]}"#;
        let err = weights_blob_name(manifest).unwrap_err().to_string();
        assert!(err.contains("no application/vnd.ollama.image.model layer"), "{err}");
    }

    #[test]
    fn a_digest_that_is_not_algo_colon_hex_cannot_become_a_path_segment() {
        for bad in [
            r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"79e4df2a"}]}"#,
            r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:../../etc"}]}"#,
            r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:"}]}"#,
            r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":":abcd"}]}"#,
        ] {
            assert!(weights_blob_name(bad).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn the_manifest_path_is_built_the_way_ollama_lays_the_store_out() {
        let got = manifest_path(Path::new("/store"), "vessel");
        let tail: PathBuf =
            ["manifests", "registry.ollama.ai", "library", "vessel", "latest"].iter().collect();
        assert!(got.ends_with(tail), "{}", got.display());
    }
}
