//! The system messages that shape a generation: the character persona and the
//! out-of-character director note.
//!
//! This lived in `server.js` because that is where the route was. It is not
//! transport, though - it is the product - so it moves into the core with
//! everything else the shell must not own.
//!
//! ## Why the global behaviour is carried here
//!
//! On Ollama, `SYSTEM` in the Modelfile is baked into the model at
//! `ollama create` time. It looks like it always applies. It does not: MEASURED,
//! when the client sends any system message of its own, Ollama uses THAT as the
//! model's system prompt and the Modelfile SYSTEM is never rendered. A bare
//! `/api/chat` call prefixes 492 tokens; the same call with a client system
//! message prefixes 26. Since [`persona_message`] always produces a system
//! message, that SYSTEM had never once reached the model.
//!
//! Reading the Modelfile and prepending it here is what actually applies it -
//! and it is also what makes llama-server, which loads the raw GGUF and knows
//! nothing of Modelfiles, behave identically. It sits at the head of the
//! cacheable prefix, so it is prefilled once per conversation and reused on
//! every later turn.

use std::sync::OnceLock;

use crate::characters::Character;
use crate::inference::modelfile;
use crate::inference::Message;

/// The Modelfile's `SYSTEM`, read once.
pub fn global_behavior() -> &'static str {
    static GB: OnceLock<String> = OnceLock::new();
    GB.get_or_init(|| {
        modelfile::load()
            .system
            .as_deref()
            .unwrap_or_default()
            .trim()
            .to_string()
    })
}

/// Appended to EVERY persona message, whatever the response style.
///
/// [`global_behavior`] asks for paragraphs too, but that instruction sits far
/// from the point of generation and the model dilutes it. Restating it
/// concretely at the END of the per-character system message - which lands right
/// before the turn - is what actually makes replies break into spaced
/// paragraphs.
const FORMAT_RULE: &str = "Formatting (REQUIRED): keep your reply SHORT \u{2014} 2 to 4 short paragraphs, then STOP. Do not write a long multi-paragraph scene; leave room for the user to respond. Separate each paragraph with one empty line (press Enter twice). Do NOT write the words \"blank line\" or any label between paragraphs \u{2014} just leave the empty line. Never answer in a single block. ALWAYS wrap every spoken line in straight double quotes, like \"this\", with no exceptions \u{2014} even one-line replies must put the spoken words in quotes. Put spoken dialogue on its own paragraph, and put actions/narration in separate paragraphs around it. CRITICAL: write ONLY your own character. Never narrate, decide, or quote the user \u{2014} do not describe the user's actions, words, body, gaze, thoughts, or feelings (\"you take a sip\", \"you let out a breath\", \"a flicker of surprise on your face\", \"you say...\"). End your reply at the moment it becomes the user's turn, then stop and wait.";

/// Narration vs dialogue control. Addresses the common failure where the model
/// only narrates the scene instead of speaking as the character.
fn style_rule(style: &str) -> &'static str {
    match style {
        "dialogue" => "Response style: ALWAYS give the character spoken dialogue when addressed. Lead with what the character SAYS (in quotes) on its own paragraph. Surround it with short separate paragraphs of action/reaction. Never reply with narration only.",
        "narration-light" => "Response style: keep narration brief and focused. Prioritize the character speaking and reacting over describing the scene.",
        _ => "", // "balanced", and anything unrecognised
    }
}

/// The persona system message: global roleplay behaviour first, then this
/// specific character.
pub fn persona_message(character: Option<&Character>) -> Option<Message> {
    let c = character?;
    // Sized once rather than grown: the persona runs to a few KB and this is
    // built on every turn.
    let mut parts: Vec<String> = Vec::with_capacity(6);

    let gb = global_behavior();
    if !gb.is_empty() {
        parts.push(gb.to_string());
    }
    parts.push(format!(
        "You are roleplaying as the character \"{name}\". Stay fully in character as {name}.",
        name = c.name
    ));
    let persona = c.persona.trim();
    if !persona.is_empty() {
        parts.push(format!("Character details:\n{persona}"));
    }
    // The single most common failure is the model speaking or acting FOR the
    // user. Stating the boundary concretely, with the character's name in it, is
    // far more reliable than the abstract global rule alone.
    parts.push(format!(
        "Control ONLY {name}. You write {name}'s words, actions, thoughts, and reactions \u{2014} nothing else. Never write what the user does, says, thinks, or feels: do not describe the user's body, gaze, sensations, or emotions (\"a shiver runs down your spine\", \"you can't look away\", \"you feel...\") \u{2014} those are the user's to write, not yours. Describe only what {name} perceives and does. End every reply at a point that hands control back to the user, then stop and wait for their response.",
        name = c.name
    ));
    let rule = style_rule(&c.response_style);
    if !rule.is_empty() {
        parts.push(rule.to_string());
    }
    parts.push(FORMAT_RULE.to_string());

    Some(Message::new("system", parts.join("\n\n")))
}

/// The director / OOC note: a meta-instruction that steers the model WITHOUT
/// becoming part of the story. Injected as a system message, never recorded.
pub fn director_message(director: Option<&str>) -> Option<Message> {
    let text = director?.trim();
    if text.is_empty() {
        return None;
    }
    Some(Message::new(
        "system",
        format!("[Director note \u{2014} out of character. Follow this instruction for how you write from now on, but do NOT mention it in the story or break character to acknowledge it:]\n{text}"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn character(name: &str, persona: &str, style: &str) -> Character {
        Character {
            id: "c1".into(),
            name: name.into(),
            avatar: String::new(),
            tagline: String::new(),
            about: String::new(),
            persona: persona.into(),
            greeting: String::new(),
            chat_starters: vec![],
            tags: vec![],
            sampling: Default::default(),
            response_style: style.into(),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn no_character_means_no_persona_message() {
        assert!(persona_message(None).is_none());
    }

    #[test]
    fn the_persona_names_the_character_in_the_boundary_rule() {
        // The abstract rule alone does not hold; the name is what makes it stick.
        let m =
            persona_message(Some(&character("Aria", "A lighthouse keeper.", "balanced"))).unwrap();
        assert_eq!(m.role, "system");
        assert!(m.content.contains("roleplaying as the character \"Aria\""));
        assert!(m.content.contains("Control ONLY Aria."));
        assert!(m
            .content
            .contains("Character details:\nA lighthouse keeper."));
    }

    #[test]
    fn an_empty_persona_leaves_out_the_details_block_entirely() {
        // Not an empty heading: a bare "Character details:" tells the model the
        // character has none, which is worse than saying nothing.
        let m = persona_message(Some(&character("Aria", "   ", "balanced"))).unwrap();
        assert!(!m.content.contains("Character details:"));
    }

    #[test]
    fn the_format_rule_is_last_because_the_tail_is_what_the_model_obeys() {
        let m = persona_message(Some(&character("Aria", "x", "dialogue"))).unwrap();
        assert!(m.content.trim_end().ends_with("then stop and wait."));
        assert!(m
            .content
            .contains("ALWAYS give the character spoken dialogue"));
        let style_at = m.content.find("Response style:").unwrap();
        let format_at = m.content.find("Formatting (REQUIRED)").unwrap();
        assert!(style_at < format_at);
    }

    #[test]
    fn an_unknown_response_style_adds_no_rule_rather_than_failing() {
        let m = persona_message(Some(&character("Aria", "x", "not-a-style"))).unwrap();
        assert!(!m.content.contains("Response style:"));
        let balanced = persona_message(Some(&character("Aria", "x", "balanced"))).unwrap();
        assert_eq!(m.content, balanced.content);
    }

    #[test]
    fn a_director_note_is_marked_out_of_character() {
        let m = director_message(Some("  be colder  ")).unwrap();
        assert!(m.content.starts_with("[Director note"));
        assert!(m.content.ends_with("\nbe colder"), "trimmed, not raw");
        assert!(m.content.contains("do NOT mention it in the story"));
    }

    #[test]
    fn a_blank_director_note_is_no_note() {
        assert!(director_message(None).is_none());
        assert!(director_message(Some("")).is_none());
        assert!(director_message(Some("   \n ")).is_none());
    }
}
