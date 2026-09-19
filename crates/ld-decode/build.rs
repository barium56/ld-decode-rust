// Builds the vendored ducc0 FFT library (the exact version scipy 1.18.0 uses
// behind `scipy.fft`, i.e. `scipy.fft._duccfft`) plus a small C shim into
// static libs linked into this crate.
//
// PARITY RULE (do not change without re-verifying the goldens): the DEFAULT
// engine must be compiled exactly the way scipy's `_duccfft` wheel is built on
// x86-64 to reproduce its rounding bit-for-bit: the "homegrown SIMD" path at
// the 128-bit (SSE2 / NEON) width, no AVX/FMA, and single-threaded. On Windows
// that means clang-cl with MSVC ABI/STL; on Linux/macOS it means clang++ or
// g++ with the GNU dialect. The FFT kernels' operation order is defined by
// ducc0's source, not by the compiler, and ducc0 picks its SIMD width from the
// compiler's predefined macros (`__SSE2__`, `__AVX2__` -- see
// `ducc0/infra/simd.h`), so a default-baseline x86-64 build on either OS
// selects the same 128-bit kernels and rounds identically.
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
//
// `-mavx2 -mfma` is x86-only; on other architectures the three names are built
// from the same portable TU so the FFI surface stays uniform (engine selection
// is CPU-gated at runtime and never picks an AVX2 engine off x86 -- see
// `ffi_ducc::cpu_has_avx2`).
//
// To type-check (`cargo check`) on a host without a C++ toolchain, set
// LD_SKIP_VENDOR_FFT=1: the native build is skipped entirely. Checking does not
// link, so the missing symbols do not matter; a real build would fail loudly.
fn main() {
    println!("cargo:rerun-if-changed=../../vendor");
    println!("cargo:rerun-if-env-changed=LD_SKIP_VENDOR_FFT");

    if std::env::var_os("LD_SKIP_VENDOR_FFT").is_some() {
        println!(
            "cargo:warning=LD_SKIP_VENDOR_FFT is set: skipping the vendored ducc0 build. \
             This is a check-only mode; the resulting binary cannot link."
        );
        return;
    }

    let target_arch = var("CARGO_CFG_TARGET_ARCH");
    let target_env = var("CARGO_CFG_TARGET_ENV");
    let target_os = var("CARGO_CFG_TARGET_OS");
    let msvc = target_env == "msvc";
    let x86 = target_arch == "x86_64" || target_arch == "x86";

    let vendor = "../../vendor";
    let out = std::env::var("OUT_DIR").unwrap();

    // Engine-independent ducc translation units (no SIMD, compiled once).
    let mut base = cc::Build::new();
    base.cpp(true)
        .define("DUCC0_NO_LOWLEVEL_THREADING", None)
        .pic(true)
        .warnings(false)
        .include(vendor)
        .file(format!("{vendor}/ducc0/infra/threading.cc"))
        .file(format!("{vendor}/ducc0/infra/string_utils.cc"));
    language_flags(&mut base, msvc);
    if let Some(cxx) = find_cxx(msvc) {
        base.compiler(cxx);
    }
    base.compile("duccfft_base");

    // The FFT shim, once per engine. The source is copied to a distinct
    // object-stem per engine so the cc crate's object files cannot collide.
    let shim = std::fs::read_to_string(format!("{vendor}/ducc_ffi.cc"))
        .expect("read vendor/ducc_ffi.cc");
    let engines: &[(&str, &[&str], &str)] = if x86 {
        &[
            ("sse2", &[], "duccq_sse2_"),
            ("avx2fma", &["-mavx2", "-mfma"], "duccq_avx2fma_"),
            (
                "avx2",
                &["-mavx2", "-mfma", "-ffp-contract=off"],
                "duccq_avx2_",
            ),
        ]
    } else {
        // No x86 SIMD: build the same portable TU under all three names. The
        // AVX2 engines are an x86-64 experiment; off x86 they are aliases, and
        // they are unreachable because the runtime feature gate is false.
        &[
            ("sse2", &[], "duccq_sse2_"),
            ("avx2fma", &[], "duccq_avx2fma_"),
            ("avx2", &[], "duccq_avx2_"),
        ]
    };
    for (name, extra, prefix) in engines.iter().copied() {
        let src = format!("{out}/ducc_ffi_{name}.cc");
        std::fs::write(&src, &shim).expect("write shim copy");
        let mut b = cc::Build::new();
        b.cpp(true)
            .define("DUCC0_NO_LOWLEVEL_THREADING", None)
            .define("DUCCQ_PREFIX", prefix)
            .pic(true)
            .warnings(false)
            .include(vendor)
            .file(&src);
        language_flags(&mut b, msvc);
        // `/O2`, and it stays `/O2`: an `/O3` shim (bit-identical by
        // construction — clang-cl's `/O3` implies no fast-math — and 4/4 on
        // b3sum) measured **neutral** on a clean 4-round interleaved
        // `-l 2000` A/B (43.15 vs 42.99 FPS mean, 3/4 pairs positive), so it
        // was reverted rather than kept as dead weight. The 76% of demod CPU
        // that lives in this TU is dominated by memory traffic, not by the
        // inlining/loop opts `/O3` adds.
        if let Some(cxx) = find_cxx(msvc) {
            b.compiler(cxx);
        }
        // Unconditional: cc's flag_if_supported probe silently drops -mavx2
        // under clang-cl (measured 2026-09-17 — an "avx2fma" build compiled
        // via flag_if_supported ran at sse2 speed, i.e. the flags never
        // reached the TU). clang-cl and clang++/g++ both accept GNU-style
        // -m/-f flags directly.
        for f in extra {
            b.flag(f);
        }
        b.compile(&format!("duccfft_{name}"));
    }

    // The C++ runtime is not on rustc's default link line outside MSVC (where
    // the CRT directives embedded in the objects pull it in). Without this the
    // final link fails on undefined std::* symbols.
    match target_os.as_str() {
        "linux" | "android" | "freebsd" | "netbsd" | "openbsd" | "dragonfly" | "illumos"
        | "solaris" => println!("cargo:rustc-link-lib=stdc++"),
        "macos" | "ios" | "tvos" | "watchos" => println!("cargo:rustc-link-lib=c++"),
        _ => {}
    }
}

fn var(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

/// C++ dialect flags for the target's toolchain. The optimization level is
/// `/O2` on MSVC and `-O2` elsewhere; no fast-math is enabled by default and
/// none is added (the arithmetic must stay IEEE and contraction-free — without
/// `-mfma` there is no FMA instruction to contract into on x86-64).
fn language_flags(b: &mut cc::Build, msvc: bool) {
    if msvc {
        b.define("_CRT_SECURE_NO_WARNINGS", None)
            .flag_if_supported("/EHsc")
            .flag_if_supported("/std:c++17")
            .flag_if_supported("/O2");
    } else {
        b.flag("-std=c++17").flag("-O2");
    }
}

/// Pick the C++ compiler. An explicit `CXX` always wins (cc honours it), so
/// only answer when it is unset. Otherwise prefer clang so both platforms
/// build the same source with the same compiler family, and fall back to
/// whatever `cc` would have chosen.
fn find_cxx(msvc: bool) -> Option<String> {
    if let Ok(cxx) = std::env::var("CXX") {
        if !cxx.trim().is_empty() {
            return None;
        }
    }
    if msvc {
        // Prefer an explicit full path so the build is reproducible even if
        // the LLVM bin dir is not on PATH.
        for cand in [
            "C:\\Program Files\\LLVM\\bin\\clang-cl.exe",
            "C:\\Program Files (x86)\\LLVM\\bin\\clang-cl.exe",
        ] {
            if std::path::Path::new(cand).exists() {
                return Some(cand.to_string());
            }
        }
        return Some("clang-cl".to_string());
    }
    for cand in ["clang++", "clang++-19", "clang++-18", "clang++-17", "g++"] {
        if on_path(cand) {
            return Some(cand.to_string());
        }
    }
    None
}

fn on_path(exe: &str) -> bool {
    std::process::Command::new(exe)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}


