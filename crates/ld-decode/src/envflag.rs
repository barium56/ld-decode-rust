//! Cached environment flags for hot paths.
//!
//! `std::env::var_os` takes the process-wide environment lock and walks the
//! environment block on every call. The decoder asks about the same handful of
//! `LD_*` switches dozens of times per field (and per demod block), so each
//! site owns a `CachedFlag` instead: one relaxed atomic load after the first
//! probe. The value is read once per process, which is what the debug
//! switches want anyway (`LD_DUMP_*` destinations are not re-read mid-run).

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

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

/// A single-shot cached environment *value* (the readloc filters and dump
/// destinations that go with a [`CachedFlag`]).
///
/// Some probes are only reached once per line — or once per zero crossing —
/// and re-reading the variable on every call is far from free: on Windows
/// `std::env::var_os` takes a process-wide read lock and issues a
/// `GetEnvironmentVariableW` syscall, so a few tens of thousands of them per
/// field cost milliseconds *and* serialize otherwise independent pool workers
/// on the shared lock. Values are cached once, like the flags: the `LD_DUMP_*`
/// switches are not meant to be re-pointed mid-run.
pub(crate) struct CachedVar {
    name: &'static str,
    val: OnceLock<Option<std::ffi::OsString>>,
}

impl CachedVar {
    pub(crate) const fn new(name: &'static str) -> Self {
        Self {
            name,
            val: OnceLock::new(),
        }
    }

    /// The raw value (`std::env::var_os`), or `None` when absent. Present but
    /// empty is `Some("")`, exactly as `var_os` reports it.
    #[inline]
    pub(crate) fn get(&self) -> Option<&std::ffi::OsStr> {
        self.val
            .get_or_init(|| std::env::var_os(self.name))
            .as_deref()
    }

    #[inline]
    pub(crate) fn is_present(&self) -> bool {
        self.get().is_some()
    }

    /// `true` when set to a non-empty value.
    #[inline]
    pub(crate) fn is_nonempty(&self) -> bool {
        self.get().map(|v| !v.is_empty()).unwrap_or(false)
    }

    /// The value as `&str`, or `""` when absent or not valid UTF-8 (matches
    /// `std::env::var(name).unwrap_or_default()`).
    #[inline]
    pub(crate) fn str_or_empty(&self) -> &str {
        self.get().and_then(|v| v.to_str()).unwrap_or("")
    }
}
