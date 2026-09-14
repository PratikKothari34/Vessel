//! CRUD for roleplay characters.
//!
//! A character is the persona the model plays: name, avatar, persona text,
//! opening greeting, and optional per-character sampling overrides. The persona
//! is injected as a system message at chat time, so each character behaves
//! distinctly without rebuilding the model.
//!
//! Everything a client sends is normalised here rather than at the command
//! boundary, because these rows are also reachable by import and by sync from
//! another machine - neither of which passes through a command handler.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};
use turso::Value as TValue;

use crate::db;

/// Ids we mint are UUIDs. Anything else is rejected so an id can never carry
/// something that matters to a query or a path.
pub fn is_valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Sampling overrides are stored as a JSON string. Only these keys survive, so a
/// client cannot smuggle arbitrary inference options through, and each is
/// clamped so a hostile or imported value cannot request a runaway context
/// window (OOM / GPU exhaustion) or out-of-domain sampling.
const SAMPLING_BOUNDS: &[(&str, f64, f64)] = &[
    ("temperature", 0.0, 2.0),
    ("top_p", 0.0, 1.0),
    ("top_k", 0.0, 1000.0),
    ("min_p", 0.0, 1.0),
    ("repeat_penalty", 0.0, 4.0),
    ("num_ctx", 256.0, 131072.0),
    ("num_predict", 16.0, 4096.0),
];

/// Keep only known numeric keys, clamped.
///
/// The JS version needed a paragraph of defence here because its Number() cast
/// maps null, the empty string, false and the empty array all to 0 - which is IN
/// RANGE for every bound below, so a temperature sent as null to mean "leave it
/// alone" would have pinned the character to greedy decoding forever. Rust draws
/// that line for free, but the rule is the same and worth stating: absent means
/// absent, and only a real number (or a string that parses as one) counts.
pub fn clean_sampling(s: Option<&Json>) -> Map<String, Json> {
    let mut out = Map::new();
    let Some(Json::Object(obj)) = s else { return out };
    for (key, min, max) in SAMPLING_BOUNDS {
        let v = match obj.get(*key) {
            Some(Json::Number(n)) => n.as_f64(),
            // A number that arrived as a string still counts; "" and "abc" do not.
            Some(Json::String(t)) => t.trim().parse::<f64>().ok(),
            _ => None,
        };
        let Some(v) = v.filter(|v| v.is_finite()) else { continue };
        if let Some(n) = serde_json::Number::from_f64(v.clamp(*min, *max)) {
            out.insert((*key).to_string(), Json::Number(n));
        }
    }
    out
}

const RESPONSE_STYLES: [&str; 3] = ["balanced", "dialogue", "narration-light"];

pub fn clean_style(s: Option<&str>) -> String {
    match s {
        Some(v) if RESPONSE_STYLES.contains(&v) => v.to_string(),
        _ => "balanced".to_string(),
    }
}

// Field length caps (defence in depth - the request body limit is the only other
// bound). `avatar` is short on purpose: a URL fits; a giant data: URI does not,
// and it would bloat both the database and the sync leg.
const CAP_AVATAR: usize = 4096;
const CAP_ABOUT: usize = 8000;
const CAP_PERSONA: usize = 16000;
const CAP_GREETING: usize = 8000;
const CAP_NAME: usize = 120;
const CAP_TAGLINE: usize = 200;

/// Truncate to `max` CHARACTERS, not bytes: slicing a UTF-8 string at a byte
/// offset can land mid-codepoint and panic, and the JS original counted UTF-16
/// units, so byte-slicing would disagree with it as well.
fn cap(v: &str, max: usize) -> String {
    v.chars().take(max).collect()
}

fn cap_opt(v: Option<&Json>, max: usize) -> String {
    cap(v.and_then(Json::as_str).unwrap_or(""), max)
}

/// The avatar is rendered into an img src. Persist only web image URLs and
/// inline image data; reject everything else (file:, javascript:, ...) so a
/// hostile or imported value can never make a client fetch a local path. Empty
/// means the generated glyph fallback.
pub fn clean_avatar(v: &str) -> String {
    let s = cap(v, CAP_AVATAR);
    let s = s.trim();
    let lower = s.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("data:image/")
    {
        s.to_string()
    } else {
        String::new()
    }
}

/// Trim, drop empties, cap both the number of entries and each entry's length.
pub fn clean_list(a: Option<&Json>, max: usize, max_len: usize) -> Vec<String> {
    let Some(Json::Array(items)) = a else { return Vec::new() };
    items
        .iter()
        .map(|x| cap(x.as_str().unwrap_or("").trim(), max_len))
        .filter(|s| !s.is_empty())
        .take(max)
        .collect()
}

fn parse_list(json: Option<&Json>) -> Vec<String> {
    let raw = json.and_then(Json::as_str).unwrap_or("[]");
    match serde_json::from_str::<Json>(raw) {
        Ok(Json::Array(a)) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Character {
    pub id: String,
    pub name: String,
    pub avatar: String,
    pub tagline: String,
    pub about: String,
    pub persona: String,
    pub greeting: String,
    pub chat_starters: Vec<String>,
    pub tags: Vec<String>,
    pub sampling: Map<String, Json>,
    pub response_style: String,
    pub created_at: String,
    pub updated_at: String,
}

fn s(row: &Map<String, Json>, key: &str) -> String {
    row.get(key).and_then(Json::as_str).unwrap_or("").to_string()
}

fn row_to_character(row: &Map<String, Json>) -> Character {
    let sampling = match row
        .get("sampling")
        .and_then(Json::as_str)
        .map(serde_json::from_str::<Json>)
    {
        Some(Ok(Json::Object(m))) => m,
        _ => Map::new(),
    };
    Character {
        id: s(row, "id"),
        name: s(row, "name"),
        avatar: s(row, "avatar"),
        tagline: s(row, "tagline"),
        about: s(row, "about"),
        persona: s(row, "persona"),
        greeting: s(row, "greeting"),
        chat_starters: parse_list(row.get("chat_starters")),
        tags: parse_list(row.get("tags")),
        sampling,
        response_style: clean_style(row.get("response_style").and_then(Json::as_str)),
        created_at: s(row, "created_at"),
        updated_at: s(row, "updated_at"),
    }
}

/// The shape a client sends. Every field is optional; `create` requires a name
/// and `update` treats an absent field as "leave it alone".
///
/// `Option<Json>` rather than `Option<String>` on purpose: it keeps null and
/// absent distinguishable, which is the difference between "clear this" and "do
/// not touch this", and it lets the normalisers above reject one wrong-typed
/// field instead of a deserialize error rejecting the whole request.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterPatch {
    pub name: Option<Json>,
    pub avatar: Option<Json>,
    pub tagline: Option<Json>,
    pub about: Option<Json>,
    pub persona: Option<Json>,
    pub greeting: Option<Json>,
    pub chat_starters: Option<Json>,
    pub tags: Option<Json>,
    pub sampling: Option<Json>,
    pub response_style: Option<Json>,
}

/// null means the same as absent - the JS original compared against null, which
/// is true for both.
fn present(v: &Option<Json>) -> Option<&Json> {
    match v {
        Some(Json::Null) | None => None,
        Some(x) => Some(x),
    }
}

pub async fn list() -> Result<Vec<Character>> {
    let db = db::get().await?;
    let rows = db
        .query("SELECT * FROM characters ORDER BY updated_at DESC", vec![])
        .await?;
    Ok(rows.iter().map(row_to_character).collect())
}

pub async fn get(id: &str) -> Result<Option<Character>> {
    if !is_valid_id(id) {
        return Ok(None);
    }
    let db = db::get().await?;
    let row = db
        .query_one("SELECT * FROM characters WHERE id = ?", vec![TValue::Text(id.into())])
        .await?;
    Ok(row.as_ref().map(row_to_character))
}

pub async fn create(p: CharacterPatch) -> Result<Option<Character>> {
    let name = cap(
        present(&p.name).and_then(Json::as_str).unwrap_or("").trim(),
        CAP_NAME,
    );
    if name.is_empty() {
        return Err(anyhow!("Character name is required."));
    }

    let db = db::get().await?;
    let id = uuid::Uuid::new_v4().to_string();
    let ts = crate::util::now_iso();

    db.execute(
        "INSERT INTO characters
           (id, name, avatar, tagline, about, persona, greeting, chat_starters, tags,
            sampling, response_style, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        vec![
            TValue::Text(id.clone()),
            TValue::Text(name),
            TValue::Text(clean_avatar(present(&p.avatar).and_then(Json::as_str).unwrap_or(""))),
            TValue::Text(cap_opt(present(&p.tagline), CAP_TAGLINE)),
            TValue::Text(cap_opt(present(&p.about), CAP_ABOUT)),
            TValue::Text(cap_opt(present(&p.persona), CAP_PERSONA)),
            TValue::Text(cap_opt(present(&p.greeting), CAP_GREETING)),
            TValue::Text(serde_json::to_string(&clean_list(present(&p.chat_starters), 12, 200))?),
            TValue::Text(serde_json::to_string(&clean_list(present(&p.tags), 12, 40))?),
            TValue::Text(serde_json::to_string(&clean_sampling(present(&p.sampling)))?),
            TValue::Text(clean_style(present(&p.response_style).and_then(Json::as_str))),
            TValue::Text(ts.clone()),
            TValue::Text(ts),
        ],
    )
    .await?;

    get(&id).await
}

pub async fn update(id: &str, p: CharacterPatch) -> Result<Option<Character>> {
    if !is_valid_id(id) {
        return Err(anyhow!("Invalid character id."));
    }
    let Some(existing) = get(id).await? else { return Ok(None) };

    let name = match present(&p.name) {
        Some(v) => cap(v.as_str().unwrap_or("").trim(), CAP_NAME),
        None => existing.name,
    };
    if name.is_empty() {
        return Err(anyhow!("Character name is required."));
    }

    let avatar = match present(&p.avatar) {
        Some(v) => clean_avatar(v.as_str().unwrap_or("")),
        None => existing.avatar,
    };
    let tagline = match present(&p.tagline) {
        Some(v) => cap(v.as_str().unwrap_or(""), CAP_TAGLINE),
        None => existing.tagline,
    };
    let about = match present(&p.about) {
        Some(v) => cap(v.as_str().unwrap_or(""), CAP_ABOUT),
        None => existing.about,
    };
    let persona = match present(&p.persona) {
        Some(v) => cap(v.as_str().unwrap_or(""), CAP_PERSONA),
        None => existing.persona,
    };
    let greeting = match present(&p.greeting) {
        Some(v) => cap(v.as_str().unwrap_or(""), CAP_GREETING),
        None => existing.greeting,
    };
    let starters = match present(&p.chat_starters) {
        Some(v) => clean_list(Some(v), 12, 200),
        None => existing.chat_starters,
    };
    let tags = match present(&p.tags) {
        Some(v) => clean_list(Some(v), 12, 40),
        None => existing.tags,
    };
    let sampling = match present(&p.sampling) {
        Some(v) => clean_sampling(Some(v)),
        None => existing.sampling,
    };
    let style = match present(&p.response_style) {
        Some(v) => clean_style(v.as_str()),
        None => existing.response_style,
    };

    let db = db::get().await?;
    db.execute(
        "UPDATE characters SET name=?, avatar=?, tagline=?, about=?, persona=?, greeting=?,
           chat_starters=?, tags=?, sampling=?, response_style=?, updated_at=?
         WHERE id=?",
        vec![
            TValue::Text(name),
            TValue::Text(avatar),
            TValue::Text(tagline),
            TValue::Text(about),
            TValue::Text(persona),
            TValue::Text(greeting),
            TValue::Text(serde_json::to_string(&starters)?),
            TValue::Text(serde_json::to_string(&tags)?),
            TValue::Text(serde_json::to_string(&sampling)?),
            TValue::Text(style),
            TValue::Text(crate::util::now_iso()),
            TValue::Text(id.into()),
        ],
    )
    .await?;

    get(id).await
}

/// Deletes the character and, by foreign-key cascade, all its conversations,
/// turns and archive rows.
pub async fn delete(id: &str) -> Result<bool> {
    if !is_valid_id(id) {
        return Err(anyhow!("Invalid character id."));
    }
    let db = db::get().await?;
    Ok(db
        .execute("DELETE FROM characters WHERE id = ?", vec![TValue::Text(id.into())])
        .await?
        > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sampling_keeps_only_known_keys_and_clamps_them() {
        let out = clean_sampling(Some(&json!({
            "temperature": 5.0,
            "top_p": -1,
            "num_ctx": 999_999,
            "num_predict": "512",
            "seed": 42,
            "mirostat": 2
        })));
        assert_eq!(out.get("temperature").unwrap(), &json!(2.0));
        assert_eq!(out.get("top_p").unwrap(), &json!(0.0));
        assert_eq!(out.get("num_ctx").unwrap(), &json!(131072.0));
        assert_eq!(out.get("num_predict").unwrap(), &json!(512.0));
        assert!(!out.contains_key("seed"), "unknown keys must not survive");
        assert!(!out.contains_key("mirostat"));
    }

    #[test]
    fn a_null_sampling_value_means_absent_not_zero() {
        // This is the bug the JS version had to defend against by hand: every
        // one of these casts to 0 there, and 0 is in range for all of them.
        let out = clean_sampling(Some(&json!({
            "temperature": null,
            "top_p": "",
            "top_k": false,
            "min_p": [],
            "repeat_penalty": {}
        })));
        assert!(out.is_empty(), "got {out:?}");
    }

    #[test]
    fn avatars_are_limited_to_web_images() {
        assert_eq!(clean_avatar("https://example.test/a.png"), "https://example.test/a.png");
        assert_eq!(clean_avatar("  HTTP://example.test/a.png "), "HTTP://example.test/a.png");
        assert_eq!(clean_avatar("data:image/png;base64,AAAA"), "data:image/png;base64,AAAA");
        assert_eq!(clean_avatar("file:///C:/Windows/win.ini"), "");
        assert_eq!(clean_avatar("javascript:alert(1)"), "");
        assert_eq!(clean_avatar("data:text/html,<script>"), "");
        assert_eq!(clean_avatar("/etc/passwd"), "");
    }

    #[test]
    fn an_over_long_avatar_is_rejected_rather_than_truncated_into_validity() {
        // Truncating first and matching second is the safe order: a 5000-char
        // data:text/html URI must not become a valid-looking 4096-char one.
        let long = format!("data:text/html,{}", "A".repeat(5000));
        assert_eq!(clean_avatar(&long), "");
    }

    #[test]
    fn caps_count_characters_not_bytes() {
        // Four-byte codepoints: a byte-based cap would slice mid-character.
        let s: String = "\u{1F600}".repeat(10);
        assert_eq!(cap(&s, 3).chars().count(), 3);
    }

    #[test]
    fn lists_are_trimmed_deduped_of_blanks_and_bounded() {
        let out = clean_list(Some(&json!(["  a  ", "", "   ", "b", 7, null])), 12, 200);
        assert_eq!(out, vec!["a", "b"]);
        let many: Vec<Json> = (0..50).map(|i| json!(format!("t{i}"))).collect();
        assert_eq!(clean_list(Some(&Json::Array(many)), 12, 40).len(), 12);
        assert_eq!(clean_list(Some(&json!("not an array")), 12, 40).len(), 0);
        assert_eq!(clean_list(None, 12, 40).len(), 0);
    }

    #[test]
    fn ids_are_restricted_to_what_we_mint() {
        assert!(is_valid_id(&uuid::Uuid::new_v4().to_string()));
        assert!(!is_valid_id(""));
        assert!(!is_valid_id("../../etc/passwd"));
        assert!(!is_valid_id("a'; DROP TABLE characters--"));
        assert!(!is_valid_id(&"a".repeat(65)));
    }

    #[test]
    fn unknown_response_styles_fall_back_to_balanced() {
        assert_eq!(clean_style(Some("dialogue")), "dialogue");
        assert_eq!(clean_style(Some("shakespeare")), "balanced");
        assert_eq!(clean_style(None), "balanced");
    }
}
