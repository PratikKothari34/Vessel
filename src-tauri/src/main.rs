// Windows: no console window on a GUI build. In debug we keep it, because the
// banner and every tracing line go to stdout and that is the only place they
// are readable without a log file.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    vessel_shell::run()
}
