// Builds the vendored ducc0 FFT library (the exact version scipy 1.18.0 uses
// behind `scipy.fft`, i.e. `scipy.fft._duccfft`) plus a small C shim into a
// static lib linked into this crate.
//
// ducc0 must be compiled exactly the way scipy's `_duccfft` wheel is built on
// x86-64 Windows/MSVC to reproduce its rounding bit-for-bit: the "homegrown
// SIMD" path using only SSE2 (no AVX/FMA), and single-threaded. clang-cl uses
// MSVC ABI/STL and emits COFF objects linkable by rustc's MSVC linker.
fn main() {
    println!("cargo:rerun-if-changed=../../vendor");

    let vendor = "../../vendor";
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
        .file(format!("{vendor}/ducc_ffi.cc"))
        .file(format!("{vendor}/ducc0/infra/threading.cc"))
        .file(format!("{vendor}/ducc0/infra/string_utils.cc"))
        .compile("duccfft");
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