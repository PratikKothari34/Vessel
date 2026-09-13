//! Vessel's desktop shell (decision 0001, stage 4).
//!
//! This crate owns the window and the command surface. It holds no application
//! logic: every route the Express backend exposed becomes a `#[tauri::command]`
//! here that calls into [`vessel_core`], and `/chat`'s SSE stream becomes a
//! Tauri event channel.
//!
//! There is **no loopback HTTP server**. That deletes the whole DNS-rebinding /
//! CORS / Host-guard perimeter the Node backend had to defend, because there is
//! no socket for a browser to reach in the first place.

pub fn run() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("failed to start Vessel");
}
