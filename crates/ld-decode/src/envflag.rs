//! Cached environment flags for hot paths.
//!
//! `std::env::var_os` takes the process-wide environment lock and walks the
//! environment block on every call. The decoder asks about the same handful of
//! `LD_*` switches dozens of times per field (and per demod block), so each
//! site owns a `CachedFlag` instead: one relaxed atomic load after the first
//! probe. The value is read once per process, which is what the debug
//! switches want anyway (`LD_DUMP_*` destinations are not re-read mid-run).

use std::sync::atomic::{AtomicU8, Ordering};

const UNKNOWN: u8 = 0;
const ABSENT: u8 = 1;
const PRESENT: u8 = 2;

/// A single-shot cached `is this environment variable set?` probe.
pub(crate) struct CachedFlag(AtomicU8);

impl CachedFlag {
    pub(crate) const fn new() -> Self {
        Self(AtomicU8::new(UNKNOWN))
    }

    /// `true` when the variable is present (any value, including empty).
    #[inline]
    pub(crate) fn get(&self, name: &str) -> bool {
        match self.0.load(Ordering::Relaxed) {
            PRESENT => true,
            ABSENT => false,
            _ => {
                let present = std::env::var_os(name).is_some();
                self.0
                    .store(if present { PRESENT } else { ABSENT }, Ordering::Relaxed);
                present
            }
        }
    }

    /// `true` when the variable is present and non-empty.
    #[inline]
    pub(crate) fn get_nonempty(&self, name: &str) -> bool {
        if !self.get(name) {
            return false;
        }
        // `get` only caches presence; the value read is rare (dump dirs).
        std::env::var(name).map(|v| !v.is_empty()).unwrap_or(false)
    }
}
