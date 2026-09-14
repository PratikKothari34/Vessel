//! Vessel core - the Rust replacement for the Node backend (decision 0001,
//! stage 4).
//!
//! Nothing here knows about Tauri, windows, or WebView2. The shell crate
//! (`src-tauri`) wraps these functions in `#[tauri::command]` handlers and owns
//! everything that touches the desktop; this crate owns everything that touches
//! the user's data.
//!
//! Two things carry over from the Node backend unchanged, both deliberately:
//!
//! - The **two-driver split** in [`db`]: local-only gets genuine aes256gcm
//!   whole-file encryption, cloud sync does not (its engine has no local-file
//!   encryption), and the choice is made at runtime from live config, never
//!   baked in at build time.
//! - The **embedding blob format** in [`embed`], byte for byte, so the Electron
//!   build and this one can read each other's rows while both exist.
//!
//! What does NOT carry over is the loopback HTTP server. Every Express route
//! becomes a Tauri command, which deletes the whole DNS-rebinding / CORS /
//! Host-guard perimeter: there is no socket for a browser to reach.

pub mod characters;
pub mod config;
pub mod db;
pub mod embed;
pub mod inference;
pub mod keystore;
pub mod memory;
pub mod metrics;
pub mod settings;
pub mod util;
