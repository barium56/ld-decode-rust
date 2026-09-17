// Builds the vendored ducc0 FFT library (the exact version scipy 1.18.0 uses
// behind `scipy.fft`, i.e. `scipy.fft._duccfft`) plus a small C shim into
// static libs linked into this crate.
//
// The DEFAULT engine must be compiled exactly the way scipy's `_duccfft`
// wheel is built on x86-64 Windows/MSVC to reproduce its rounding bit-for-bit:
// the "homegrown SIMD" path using only SSE2 (no AVX/FMA), and single-threaded.
// clang-cl uses MSVC ABI/STL and emits COFF objects linkable by rustc's MSVC
// linker.
//
// EXPERIMENTAL ENGINES (user-sanctioned 2026-09-17): the same shim TU is also
// compiled twice more — `avx2fma` (-mavx2 -mfma, the historically measured
// 2.7x-faster but differently-rounding build) and `avx2` (-mavx2 -mfma with
// -ffp-contract=off, testing the FMA-contraction hypothesis: ducc's kernels
// use no explicit FMA, so if the AVX2 divergence comes from compiler
// contraction, disabling it should round identically to SSE2 at lane width 4).
// All three link into one binary and are selected at RUNTIME via LD_FFT_ENGINE
// (default sse2), so there is no build flag that could silently ship a
// non-default engine. Each variant's extern "C" entry points get a distinct
// symbol prefix (DUCCQ_PREFIX); ducc internals are anonymous-namespace
// templates instantiated per TU, so the three copies cannot clash.
fn main() {
    println!("cargo:rerun-if-changed=../../vendor");

    let vendor = "../../vendor";
    let out = std::env::var("OUT_DIR").unwrap();

    // Engine-independent ducc translation units (no SIMD, compiled once).
    cc::Build::new()
        .cpp(true)
        .compiler(find_clang_cl())
        .define("DUCC0_NO_LOWLEVEL_THREADING", None)
        .define("_CRT_SECURE_NO_WARNINGS", None)
.flag_if_supported("/EHsc")
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/O2")
        .warnings(false)
        .include(vendor)
        .file(format!("{vendor}/ducc0/infra/threading.cc"))
        .file(format!("{vendor}/ducc0/infra/string_utils.cc"))
        .compile("duccfft_base");

    // The FFT shim, once per engine. The source is copied to a distinct
    // object-stem per engine so the cc crate's object files cannot collide.
    let shim = std::fs::read_to_string(format!("{vendor}/ducc_ffi.cc"))
        .expect("read vendor/ducc_ffi.cc");
    for (name, extra, prefix) in [
        ("sse2", &[][..], "duccq_sse2_"),
        ("avx2fma", &["-mavx2", "-mfma"][..], "duccq_avx2fma_"),
        (
            "avx2",
            &["-mavx2", "-mfma", "-ffp-contract=off"][..],
            "duccq_avx2_",
        ),
    ] {
        let src = format!("{out}/ducc_ffi_{name}.cc");
        std::fs::write(&src, &shim).expect("write shim copy");
        let mut b = cc::Build::new();
        b.cpp(true)
            .compiler(find_clang_cl())
            .define("DUCC0_NO_LOWLEVEL_THREADING", None)
            .define("_CRT_SECURE_NO_WARNINGS", None)
            .define("DUCCQ_PREFIX", prefix)
.flag_if_supported("/EHsc")
            .flag_if_supported("/std:c++17")
            .flag_if_supported("/O2")
            .warnings(false)
            .include(vendor)
            .file(&src);
        // Unconditional: cc's flag_if_supported probe silently drops -mavx2
        // under clang-cl (measured 2026-09-17 — an "avx2fma" build compiled
        // via flag_if_supported ran at sse2 speed, i.e. the flags never
        // reached the TU). clang-cl accepts GNU-style -m/-f flags directly.
        for f in extra {
            b.flag(f);
        }
        b.compile(&format!("duccfft_{name}"));
    }
}

fn find_clang_cl() -> String {
    // Prefer an explicit full path so the build is reproducible even if the
    // LLVM bin dir is not on PATH.
    for cand in [
        "C:\\Program Files\\LLVM\\bin\\clang-cl.exe",
        "C:\\Program Files (x86)\\LLVM\\bin\\clang-cl.exe",
    ] {
        if std::path::Path::new(cand).exists() {
            return cand.to_string();
        }
    }
    "clang-cl".to_string()
}