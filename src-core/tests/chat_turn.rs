//! End-to-end tests for one chat turn.
//!
//! Unit tests can prove the pieces; only this can prove the ORDER, which is
//! where every bug in the Node version lived: a turn recorded before the stream
//! closed, a conversation row left behind by a request that failed, a stop that
//! lost the user's message. So this runs the real [`chat::run`] against a real
//! database and a real HTTP engine - just not the user's, and not a model.
//!
//! The scratch database, the mock engine and the recording sink are in
//! [`common`], shared with the security pass next door.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{open, run, user, Collector};
use turso::Value as TValue;
use vessel_core::chat::{self, ChatRequest, ErrorKind};
use vessel_core::inference::Message;
use vessel_core::{db, memory, metrics};

// ---- Tests ----------------------------------------------------------------

#[tokio::test]
async fn a_turn_streams_then_records_in_that_order() {
    let db = open().await;
    let (out, sink) = run(user("Hello?")).await;
    let id = out.expect("the turn should succeed");

    let kinds = sink.kinds();
    assert_eq!(kinds.first(), Some(&"meta"), "meta must arrive before any token");
    assert_eq!(kinds.last(), Some(&"done"), "done must be the terminator");
    assert!(kinds.iter().filter(|k| **k == "meta").count() == 1, "exactly one meta");
    assert!(kinds.iter().filter(|k| **k == "done").count() == 1, "exactly one done");
    assert_eq!(sink.text(), "Hello there friend.");

    // Recorded only after the stream closed, which is why it is readable now.
    let conv = memory::get_conversation(&db, &id).await.unwrap().expect("conversation");
    let roles: Vec<&str> = conv.verbatim.iter().map(|t| t.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant"], "one exchange, in order");
    assert_eq!(conv.verbatim[0].content, "Hello?");
    assert_eq!(conv.verbatim[1].content, "Hello there friend.");
}

#[tokio::test]
async fn the_generation_is_measured_even_though_nothing_was_logged() {
    let _db = open().await;
    let (out, _) = run(user("Measure me.")).await;
    let id = out.expect("the turn should succeed");

    let snap = metrics::snapshot(10, Some(&id));
    let recent = snap["recent"].as_array().expect("recent");
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0]["promptTokens"], 120, "the done chunk's numbers are kept");
    assert_eq!(recent[0]["evalTokens"], 8);
    assert_eq!(recent[0]["backend"], "ollama");
    assert_eq!(recent[0]["aborted"], false);
    // The window the prompt was built from travels with the record; without it
    // chars-per-token cannot be computed and every estimate stays a guess.
    assert!(recent[0]["window"]["promptChars"].as_u64().unwrap_or(0) > 0);
}

#[tokio::test]
async fn an_engine_that_refuses_leaves_no_conversation_behind() {
    let db = open().await;
    let id = memory::new_id();
    let req = ChatRequest {
        messages: vec![Message::new("user", "FAIL404 please")],
        conversation_id: Some(id.clone()),
        ..Default::default()
    };
    let (out, sink) = run(req).await;

    let err = out.expect_err("a 404 from the engine is not a successful turn");
    assert_eq!(err.kind, ErrorKind::NotFound);
    assert!(sink.kinds().is_empty(), "nothing may be streamed before the engine answers");
    assert!(
        memory::get_conversation(&db, &id).await.unwrap().is_none(),
        "the row this request created must be swept away"
    );
}

#[tokio::test]
async fn a_stopped_generation_keeps_what_was_written() {
    let db = open().await;
    let id = memory::new_id();
    let req = ChatRequest {
        messages: vec![Message::new("user", "SLOW down")],
        conversation_id: Some(id.clone()),
        ..Default::default()
    };
    let sink = Arc::new(Collector::default());
    let task = tokio::spawn(chat::run(req, sink.clone()));

    // Let the first chunk land, then stop it the way the UI does.
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        if !sink.text().is_empty() {
            break;
        }
    }
    assert!(!sink.text().is_empty(), "the mock should have sent a chunk by now");
    assert_eq!(chat::cancel(&id), 1, "the stream must be registered under its conversation");

    let out = tokio::time::timeout(Duration::from_secs(10), task)
        .await
        .expect("a cancelled turn must not wait out the generation")
        .expect("task");
    assert_eq!(out.expect("a stop is not a failure"), id);
    assert_eq!(sink.kinds().last(), Some(&"done"), "a stopped stream still terminates");

    // Nothing on screen may silently vanish on reload: the partial reply is
    // recorded, not discarded.
    let conv = memory::get_conversation(&db, &id).await.unwrap().expect("conversation");
    assert_eq!(conv.verbatim.len(), 2);
    assert_eq!(conv.verbatim[1].role, "assistant");
    assert!(!conv.verbatim[1].content.is_empty());

    let snap = metrics::snapshot(10, Some(&id));
    assert_eq!(snap["recent"][0]["aborted"], true, "an aborted turn is recorded as one");
}

#[tokio::test]
async fn a_regenerated_reply_becomes_a_second_variant_of_the_same_turn() {
    let db = open().await;
    let (out, _) = run(user("Say something.")).await;
    let id = out.expect("first turn");

    let req = ChatRequest {
        conversation_id: Some(id.clone()),
        regenerate: true,
        ..Default::default()
    };
    let (out, sink) = run(req).await;
    assert_eq!(out.expect("regenerate"), id);
    assert_eq!(sink.text(), "Hello there friend.");

    let conv = memory::get_conversation(&db, &id).await.unwrap().expect("conversation");
    assert_eq!(conv.verbatim.len(), 2, "a re-roll replaces the reply, it does not append one");
    let variants = conv.verbatim[1].variants.as_ref().expect("the latest reply carries variants");
    assert_eq!(variants.len(), 2, "the original and the re-roll");
    assert_eq!(conv.verbatim[1].active_index, Some(1), "the newest is the active one");
}

#[tokio::test]
async fn nothing_to_regenerate_is_refused_rather_than_answered() {
    let _db = open().await;
    let req = ChatRequest {
        conversation_id: Some(memory::new_id()),
        regenerate: true,
        ..Default::default()
    };
    let (out, sink) = run(req).await;
    let err = out.expect_err("an empty conversation has no reply to re-roll");
    assert_eq!(err.kind, ErrorKind::BadRequest);
    assert!(sink.kinds().is_empty());
}

#[tokio::test]
async fn a_director_note_steers_without_becoming_a_turn() {
    let db = open().await;
    let (out, _) = run(user("Begin.")).await;
    let id = out.expect("first turn");

    let req = ChatRequest {
        conversation_id: Some(id.clone()),
        director: Some("be colder".into()),
        ..Default::default()
    };
    let (out, _) = run(req).await;
    assert_eq!(out.expect("director-only turn"), id);

    let conv = memory::get_conversation(&db, &id).await.unwrap().expect("conversation");
    let roles: Vec<&str> = conv.verbatim.iter().map(|t| t.role.as_str()).collect();
    // The note itself is never stored; the reply it produced is.
    assert_eq!(roles, ["user", "assistant", "assistant"]);
    assert!(!conv.verbatim.iter().any(|t| t.content.contains("be colder")));
}

#[tokio::test]
async fn a_request_with_nothing_usable_in_it_never_reaches_the_engine() {
    let db = open().await;
    let id = memory::new_id();
    let req = ChatRequest {
        messages: vec![Message::new("user", "   "), Message::new("tool", "x")],
        conversation_id: Some(id.clone()),
        ..Default::default()
    };
    let (out, sink) = run(req).await;

    let err = out.expect_err("whitespace is not a message");
    assert_eq!(err.kind, ErrorKind::BadRequest);
    assert!(sink.kinds().is_empty());
    assert!(memory::get_conversation(&db, &id).await.unwrap().is_none());
}

#[tokio::test]
async fn a_character_that_no_longer_exists_reports_itself_instead_of_a_foreign_key() {
    let db = open().await;
    let id = memory::new_id();
    let req = ChatRequest {
        messages: vec![Message::new("user", "Hi")],
        conversation_id: Some(id.clone()),
        // Well-formed, so it passes the id check and reaches the existence
        // check - which is the whole point of the test.
        character_id: Some(memory::new_id()),
        ..Default::default()
    };
    let (out, _) = run(req).await;

    let err = out.expect_err("a deleted character cannot be chatted with");
    assert_eq!(err.kind, ErrorKind::NotFound);
    assert!(err.character_id.is_some(), "the UI needs to know WHICH character is gone");
    assert!(memory::get_conversation(&db, &id).await.unwrap().is_none());
}

#[tokio::test]
async fn an_error_after_the_first_token_is_reported_and_nothing_is_recorded() {
    // The hard case: the request succeeded, tokens were streamed, and the model
    // fell over anyway. The text on screen is not a reply, so it must not be
    // saved as one - and the conversation it created must not survive either.
    let db = open().await;
    let id = memory::new_id();
    let req = ChatRequest {
        messages: vec![Message::new("user", "MIDFAIL now")],
        conversation_id: Some(id.clone()),
        ..Default::default()
    };
    let (out, sink) = run(req).await;

    assert_eq!(out.expect("the stream opened, so the call itself succeeded"), sink.conv_id());
    assert_eq!(sink.text(), "Partial", "what arrived before the failure is still shown");
    let errors = sink.errors();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("an error was encountered"), "got {:?}", errors[0]);
    assert_eq!(sink.kinds().last(), Some(&"done"), "the stream still terminates");

    assert!(
        memory::get_conversation(&db, &id).await.unwrap().is_none(),
        "a failed generation leaves no conversation in the sidebar"
    );
}

#[tokio::test]
async fn a_healthy_stream_carries_no_error_event() {
    let _db = open().await;
    let (out, sink) = run(user("Fine.")).await;
    out.expect("turn");
    assert!(sink.errors().is_empty());
}

#[tokio::test]
async fn folding_runs_in_the_background_and_the_summary_lands() {
    let db = open().await;
    let id = memory::new_id();
    // VERBATIM_TURNS=2 and SUMMARIZE_THRESHOLD=4, so a handful of exchanges
    // crosses the threshold and the oldest turns fold into a summary.
    for i in 0..4 {
        let req = ChatRequest {
            messages: vec![Message::new("user", format!("Turn {i}."))],
            conversation_id: Some(id.clone()),
            ..Default::default()
        };
        let (out, _) = run(req).await;
        out.expect("turn");
    }
    memory::await_all_maintenance().await;

    let conv = memory::get_conversation(&db, &id).await.unwrap().expect("conversation");
    assert!(!conv.summary.is_empty(), "the fold should have written a summary");
    assert!(!conv.archive.is_empty(), "and moved the oldest turns to the archive");
    assert!(
        conv.archive.iter().all(|t| t.has_embedding),
        "every archived turn needs a vector, or it can never be recalled"
    );
    assert!(conv.verbatim.len() <= 4, "the verbatim window stays bounded");
}


// ---- Variants on a turn that has none -------------------------------------

/// Strip a turn of its variant rows, which is the state a pre-variants row --
/// or one pulled from an older synced database -- arrives in. Migrations add
/// columns; they never back-fill rows.
async fn strip_variants(db: &db::Db, turn_id: i64) {
    db.execute("DELETE FROM variants WHERE turn_id = ?", vec![TValue::Integer(turn_id)])
        .await
        .expect("strip variants");
}

#[tokio::test]
async fn regenerating_a_turn_with_no_variant_rows_preserves_the_original_reply() {
    // The back-fill has to read turns.content, not the new text it was handed.
    // Seeded from the caller, the first re-roll on such a turn wrote the
    // regeneration in as the "original", mirrored it over turns.content, and
    // left two identical variants -- the reply it was meant to preserve was the
    // one it destroyed, and the swipe led nowhere.
    let db = open().await;
    let id = memory::new_id();
    memory::ensure_conversation(&db, &id, None).await.expect("conversation");
    memory::record_turn(&db, &id, None, "the reply that predates variants", "Character")
        .await
        .expect("record");

    let last = memory::get_last_assistant_turn(&db, &id).await.unwrap().expect("a reply");
    strip_variants(&db, last.id).await;

    memory::record_regeneration(&db, &id, "the second roll").await.expect("regenerate");

    let v = memory::get_variants(&db, last.id).await.expect("variants");
    assert_eq!(v.variants.len(), 2, "the base was back-filled, not overwritten");
    assert_eq!(v.variants[0].content, "the reply that predates variants", "the original survived");
    assert_eq!(v.variants[1].content, "the second roll");
    assert_eq!(v.active_index, 1, "the newest roll is the active one");
}

#[tokio::test]
async fn a_back_filled_original_can_still_be_swiped_back_to() {
    // Keeping the row is only half of it: the point of a variant is that the
    // user can return to the reply the regeneration replaced.
    let db = open().await;
    let id = memory::new_id();
    memory::ensure_conversation(&db, &id, None).await.expect("conversation");
    memory::record_turn(&db, &id, None, "the reply that predates variants", "Character")
        .await
        .expect("record");
    let last = memory::get_last_assistant_turn(&db, &id).await.unwrap().expect("a reply");
    strip_variants(&db, last.id).await;
    memory::record_regeneration(&db, &id, "the second roll").await.expect("regenerate");

    let v = memory::get_variants(&db, last.id).await.expect("variants");
    let back = memory::set_active_variant(&db, &id, last.id, v.variants[0].id)
        .await
        .expect("swipe back");

    assert_eq!(back.content, "the reply that predates variants");
    let now = memory::get_last_assistant_turn(&db, &id).await.unwrap().expect("a reply");
    assert_eq!(now.content, "the reply that predates variants", "turns.content follows the swipe");
}

#[tokio::test]
async fn exactly_one_variant_is_active_after_any_number_of_rolls() {
    let db = open().await;
    let id = memory::new_id();
    memory::ensure_conversation(&db, &id, None).await.expect("conversation");
    memory::record_turn(&db, &id, None, "roll one", "Character").await.expect("record");
    let last = memory::get_last_assistant_turn(&db, &id).await.unwrap().expect("a reply");

    for text in ["roll two", "roll three", "roll four"] {
        memory::record_regeneration(&db, &id, text).await.expect("regenerate");
        let v = memory::get_variants(&db, last.id).await.expect("variants");
        assert_eq!(
            v.variants.iter().filter(|x| x.active).count(),
            1,
            "after {text}"
        );
        assert_eq!(v.active_index, v.variants.len() - 1, "the newest roll is active");
    }

    let v = memory::get_variants(&db, last.id).await.expect("variants");
    let texts: Vec<&str> = v.variants.iter().map(|x| x.content.as_str()).collect();
    assert_eq!(
        texts,
        ["roll one", "roll two", "roll three", "roll four"],
        "oldest first, which is the order the swipe UI reads"
    );
}
