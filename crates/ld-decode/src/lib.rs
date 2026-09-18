#![cfg_attr(nightly_portable_simd, feature(portable_simd))]

mod decode;
mod envflag;
pub mod ffi_ducc;
pub mod logging;
mod optimized;
mod request;
mod spec;
mod vec_utils;

pub use decode::{
    Decoder, DecoderMetadata, DropOuts, FieldInfoEntry, LumaOutput, VitsMetrics, WriteableField,
    BLOCKSIZE,
};
pub use request::{ColorSystem, DecodeRequest, WowInterpolation};
pub use spec::DecoderSpec;

/// Demodulation thread budget set by [`set_worker_threads`] (0 = unset).
static DEMOD_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Configure the worker pools. `n` is the *demodulation* thread budget; the
/// global pool (which runs the serial tail's data-parallel sections: the sinc
/// gather, block assembly, split/filter stages) gets a quarter of it. The two
/// phases overlap, so the split matters more than the total: the demod pool
/// saturates its half of the machine, while the tail's sections are short and
/// lose more from oversubscription than they gain from extra threads (measured
/// on a 16-thread box: 12 demod + 3 tail beats 12 + 12 by ~7%).
///
/// `LD_TAIL_POOL` overrides the tail pool size (default `n/4`, min 1). Re-checked
/// 2026-09-18 under the old balance (demod-binding): a sweep of 2/3/4/6 tail
/// threads at `-j 9` was flat — the sinc gather and wow eval shrank but
/// `pffold` grew by exactly as much, since the demod prefetch was the bound.
/// Under the new balance (driver-binding, see balance notes), the tail's slack
/// is no longer fully consumed, so widening it buys again — gate any such run
/// on bitparity, the demod pool still must not be starved.
/// Call before the first decode.
pub fn set_worker_threads(n: usize) {
    let n = n.max(1);
    DEMOD_THREADS.store(n, std::sync::atomic::Ordering::Relaxed);
    let tail = std::env::var("LD_TAIL_POOL")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&t| t > 0)
        .unwrap_or_else(|| (n / 4).max(1));
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(tail)
        .build_global();
}

/// The tail (global rayon) pool size that [`set_worker_threads`] built.
pub fn tail_pool_threads() -> usize {
    rayon::current_num_threads()
}

/// The demodulation thread budget (the global pool size when unset).
pub(crate) fn demod_threads() -> usize {
    match DEMOD_THREADS.load(std::sync::atomic::Ordering::Relaxed) {
        0 => rayon::current_num_threads(),
        n => n,
    }
}

pub type DeterministicHashMap<K, V> = std::collections::HashMap<
    K,
    V,
    std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>,
>;
