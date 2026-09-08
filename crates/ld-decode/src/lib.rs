#![cfg_attr(nightly_portable_simd, feature(portable_simd))]

mod decode;
mod ffi_ducc;
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

/// Configure the global rayon pool used for parallel demodulation/downscaling.
/// Call before the first decode; default is the machine's logical core count.
pub fn set_worker_threads(n: usize) {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n.max(1))
        .build_global();
}

pub type DeterministicHashMap<K, V> = std::collections::HashMap<
    K,
    V,
    std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>,
>;
