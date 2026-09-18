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

#[cfg(test)]
mod tests {
    use super::*;

    /// An unset variable reports the same shape the uncached reads did: `None`
    /// from `var_os`, `false` from the presence probes, `""` from the
    /// `var(...).unwrap_or_default()` form. (The present case is left to the
    /// dump-identity checks; `set_var` is process-global and racy under the
    /// parallel test harness.)
    #[test]
    fn unset_variables_match_the_uncached_reads() {
        let var = CachedVar::new("LD_CACHEDVAR_TEST_UNSET");
        assert_eq!(var.get(), None);
        assert!(!var.is_present());
        assert!(!var.is_nonempty());
        assert_eq!(var.str_or_empty(), "");

        let flag = CachedFlag::new();
        assert!(!flag.get("LD_CACHEDVAR_TEST_UNSET"));
        assert!(!flag.get_nonempty("LD_CACHEDVAR_TEST_UNSET"));
    }

    /// The audit that keeps `79cf57a` from growing back.
    ///
    /// A direct `std::env::var*` in a per-line or per-block function costs a
    /// process-wide `ENV_LOCK` plus a `GetEnvironmentVariableW` syscall per
    /// call — tens of thousands per field — and serializes pool workers on the
    /// shared lock. That is what made the burst stage cost 4.3 ms of CPU per
    /// field (see the envelope commit and `work/bench_log.md`).
    ///
    /// Hot functions must go through `CachedFlag`/`CachedVar`; only the
    /// *whole-field* and once-per-process probes may read the environment
    /// directly. This test pins the set of functions that still do, so adding a
    /// probe to a per-line path fails here first.
    const AUDITED: &[(&str, &[(&str, usize)])] = &[
        (
            "decode/field.rs",
            &[
                // Whole-field or whole-run probes only.
                ("compute_burst_offsets", 4),
                ("compute_linelocs", 2),
                ("computewow_scaled", 1),
                ("downscale", 17),
                ("getpulses", 3),
                ("gl0_trace", 1),
                ("llstages_ok", 1),
                ("process", 4),
                ("refine_linelocs_burst", 1),
                ("refine_linelocs_hsync", 9),
            ],
        ),
        (
            "decode/demodblock.rs",
            // All three sit behind a cached `LD_DUMP_PIPE`/once-flag gate.
            &[("demod_block_cpu", 3)],
        ),
        (
            "optimized/scale_field.rs",
            &[("scale_field_sinc", 1), ("sinc_lut", 1)],
        ),
        ("decode/vits.rs", &[("compute_vits_metrics", 1)]),
        (
            "decode/dropouts.rs",
            &[("detect_dropouts", 1), ("dropout_detect_demod", 1)],
        ),
    ];

    /// The name of the function a `fn` line defines, or `None` for any other
    /// line. Handles the qualifiers this codebase uses (`pub`, `pub(crate)`,
    /// `unsafe`, `const`, `async`, `extern`).
    fn fn_name(line: &str) -> Option<String> {
        let mut rest = line.trim_start();
        loop {
            let head = rest.split(' ').next()?;
            if head == "fn" {
                break;
            }
            let is_qualifier = matches!(head, "pub" | "unsafe" | "const" | "async" | "extern")
                || head.starts_with("pub(");
            if !is_qualifier {
                return None;
            }
            rest = rest[head.len()..].trim_start();
        }
        let after = rest.strip_prefix("fn")?;
        let name: String = after
            .trim_start()
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            None
        } else {
            Some(name)
        }
    }

    fn env_reads_by_function(src: &str) -> Vec<(String, usize)> {
        let mut current = "<module>".to_string();
        let mut counts: Vec<(String, usize)> = Vec::new();
        for line in src.lines() {
            if let Some(name) = fn_name(line) {
                current = name;
            }
            if line.contains("std::env::var") {
                match counts.iter_mut().find(|(n, _)| *n == current) {
                    Some((_, c)) => *c += 1,
                    None => counts.push((current.clone(), 1)),
                }
            }
        }
        counts.sort();
        counts
    }

    #[test]
    fn no_direct_env_reads_in_hot_functions() {
        for (file, expected) in AUDITED {
            let src = match *file {
                "decode/field.rs" => include_str!("decode/field.rs"),
                "decode/demodblock.rs" => include_str!("decode/demodblock.rs"),
                "optimized/scale_field.rs" => include_str!("optimized/scale_field.rs"),
                "decode/vits.rs" => include_str!("decode/vits.rs"),
                "decode/dropouts.rs" => include_str!("decode/dropouts.rs"),
                other => panic!("unlisted file {other}"),
            };
            let got = env_reads_by_function(src);
            let want: Vec<(String, usize)> = {
                let mut v: Vec<(String, usize)> = expected
                    .iter()
                    .map(|(n, c)| (n.to_string(), *c))
                    .collect();
                v.sort();
                v
            };
            assert_eq!(
                got, want,
                "a direct `std::env::var*` moved into (or out of) a function in {file}: \
                 per-line and per-block paths must use envflag::CachedFlag/CachedVar"
            );
        }
    }
}
