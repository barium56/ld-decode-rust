//! Optional `.log` file mirroring the terminal output.
//!
//! `install` opens (truncating) `<outname>.log`; every `tracing` line and
//! every `teeprintln!` is then written both to the terminal and to that file,
//! so the `.log` always contains the full console output of the run (matching
//! the reference `ld-decode`'s `<outname>.log`).

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// The open log file, if `install` was called.
pub static LOG_FILE: OnceLock<Mutex<File>> = OnceLock::new();

/// Create (truncating) the log file at `path`. Idempotent: only the first
/// call wins. A failure to open the file is reported to stderr by the caller
/// and simply disables file logging.
pub fn install(path: &Path) -> io::Result<()> {
    if LOG_FILE.get().is_none() {
        let f = File::create(path)?;
        let _ = LOG_FILE.set(Mutex::new(f));
    }
    Ok(())
}

/// Write one line (plus newline, like `eprintln!`) to stderr and, if
/// installed, to the log file.
pub fn tee_stderr(line: &str) {
    let _ = io::stderr().write_all(line.as_bytes());
    let _ = io::stderr().write_all(b"\n");
    if let Some(m) = LOG_FILE.get() {
        if let Ok(mut f) = m.lock() {
            let _ = f.write_all(line.as_bytes());
            let _ = f.write_all(b"\n");
        }
    }
}

/// Append bytes to the log file (used by the CLI's tracing writer). Returns
/// the input length; the file write failing is not fatal.
pub fn file_write(buf: &[u8]) -> usize {
    if let Some(m) = LOG_FILE.get() {
        if let Ok(mut f) = m.lock() {
            let _ = f.write_all(buf);
        }
    }
    buf.len()
}

/// Flush the log file (used by the CLI's tracing writer on flush events).
pub fn file_flush() {
    if let Some(m) = LOG_FILE.get() {
        if let Ok(mut f) = m.lock() {
            let _ = f.flush();
        }
    }
}

/// `eprintln!` that also appends the line to the installed `.log` file.
#[macro_export]
macro_rules! teeprintln {
    ($($arg:tt)*) => {
        $crate::logging::tee_stderr(&format!($($arg)*));
    };
}