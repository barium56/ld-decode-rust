//! FFI bindings to the vendored ducc0 FFT library (the exact FFT behind
//! `scipy.fft` in scipy >= 1.18), compiled via `build.rs`.
//!
//! The DEFAULT engine (`sse2`) reproduces scipy's `_duccfft` rounding
//! **bit-for-bit** on x86-64, which is required for the decoder's output to
//! match the reference bit-exactly.
//!
//! EXPERIMENTAL ENGINES (user-sanctioned 2026-09-17): the binary also carries
//! `avx2fma` (-mavx2 -mfma — the historically 2.7x-faster but
//! differently-rounding build) and `avx2` (-mavx2 -mfma -ffp-contract=off —
//! tests whether the divergence is compiler FMA contraction; ducc's kernels
//! use no explicit FMA, so without contraction a 4-lane AVX2 build should
//! round identically to SSE2). Selection is runtime-only via `LD_FFT_ENGINE`
//! (values: `sse2` default | `avx2fma` | `avx2`), gated on CPU features, with
//! the active engine logged once at startup. The default path is untouched.

use rustfft::num_complex::Complex64;
use std::os::raw::c_int;
use std::sync::OnceLock;

unsafe extern "C" {
    fn duccq_sse2_fft(n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_sse2_ifft(n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_sse2_rfft(n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_sse2_irfft(n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_avx2fma_fft(n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_avx2fma_ifft(n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_avx2fma_rfft(n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_avx2fma_irfft(n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_avx2_fft(n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_avx2_ifft(n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_avx2_rfft(n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_avx2_irfft(n: c_int, inp: *const f64, out: *mut f64);
    #[cfg(test)]
    fn duccq_sse2_simdlen() -> c_int;
    #[cfg(test)]
    fn duccq_sse2_native_simdlen() -> c_int;
    #[cfg(test)]
    fn duccq_avx2fma_simdlen() -> c_int;
    #[cfg(test)]
    fn duccq_avx2fma_native_simdlen() -> c_int;
    #[cfg(test)]
    fn duccq_avx2_simdlen() -> c_int;
    #[cfg(test)]
    fn duccq_avx2_native_simdlen() -> c_int;
    fn duccq_sse2_ifft_batch_rows(k: c_int, n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_avx2fma_ifft_batch_rows(k: c_int, n: c_int, inp: *const f64, out: *mut f64);
    fn duccq_avx2_ifft_batch_rows(k: c_int, n: c_int, inp: *const f64, out: *mut f64);
}

/// The FFT engine backing the ducc FFI entry points.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FftEngine {
    /// scipy-wheel replica (-msse2 only): the bit-exact default.
    Sse2,
    /// -mavx2 -mfma: fastest ducc build, historically divergent rounding.
    Avx2Fma,
    /// -mavx2 -mfma -ffp-contract=off: FMA-contraction-hypothesis build.
    Avx2,
}

impl FftEngine {
    pub fn name(self) -> &'static str {
        match self {
            FftEngine::Sse2 => "sse2",
            FftEngine::Avx2Fma => "avx2fma",
            FftEngine::Avx2 => "avx2",
        }
    }

    fn resolve(requested: Option<&str>) -> Result<FftEngine, String> {
        let eng = match requested.map(str::trim).filter(|s| !s.is_empty()) {
            None => FftEngine::Sse2,
            Some(s) if s.eq_ignore_ascii_case("sse2") => FftEngine::Sse2,
            Some(s) if s.eq_ignore_ascii_case("avx2fma") => FftEngine::Avx2Fma,
            Some(s) if s.eq_ignore_ascii_case("avx2") => FftEngine::Avx2,
            Some(other) => {
                return Err(format!(
                    "unknown LD_FFT_ENGINE '{other}' (valid: sse2 | avx2fma | avx2)"
                ))
            }
        };
        // Runtime CPU gating: refusing loudly beats an illegal instruction.
        if (matches!(eng, FftEngine::Avx2Fma | FftEngine::Avx2) && !cpu_has_avx2())
            || (matches!(eng, FftEngine::Avx2Fma) && !cpu_has_fma())
        {
            return Err(format!(
                "LD_FFT_ENGINE={} requested but this CPU lacks the required features",
                eng.name()
            ));
        }
        Ok(eng)
    }
}

/// Runtime AVX2 availability. False on every non-x86 target: the AVX2 engines
/// exist only because x86-64 is the parity target, and on other architectures
/// `build.rs` compiles all three names from the same portable TU, so selecting
/// one would silently mean "not the engine you asked for".
pub(crate) fn cpu_has_avx2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Runtime FMA availability (same gating rules as [`cpu_has_avx2`]).
pub(crate) fn cpu_has_fma() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

static ENGINE: OnceLock<FftEngine> = OnceLock::new();

#[cfg(test)]
static TEST_OVERRIDE: std::sync::RwLock<Option<FftEngine>> = std::sync::RwLock::new(None);

/// Resolve and install the FFT engine from the `LD_FFT_ENGINE` env var (once
/// per process). Called by the CLI before decoding starts; errors are fatal
/// and reported verbatim.
pub fn init_engine() -> Result<FftEngine, String> {
    if let Some(e) = ENGINE.get() {
        return Ok(*e);
    }
    let requested = std::env::var("LD_FFT_ENGINE").ok();
    let eng = FftEngine::resolve(requested.as_deref())?;
    let _ = ENGINE.set(eng);
    Ok(eng)
}

/// The active engine (default `sse2` if [`init_engine`] was not called).
pub fn active_engine() -> FftEngine {
    *ENGINE.get_or_init(|| FftEngine::Sse2)
}

/// Resolve the engine once and cache the four entry points.
fn eng() -> (
    unsafe extern "C" fn(c_int, *const f64, *mut f64),
    unsafe extern "C" fn(c_int, *const f64, *mut f64),
    unsafe extern "C" fn(c_int, *const f64, *mut f64),
    unsafe extern "C" fn(c_int, *const f64, *mut f64),
    unsafe extern "C" fn(c_int, c_int, *const f64, *mut f64),
) {
    #[cfg(test)]
    {
        if let Ok(g) = TEST_OVERRIDE.read() {
            if let Some(e) = *g {
                return match e {
                    FftEngine::Sse2 => (
                        duccq_sse2_fft,
                        duccq_sse2_ifft,
                        duccq_sse2_rfft,
                        duccq_sse2_irfft,
                        duccq_sse2_ifft_batch_rows,
                    ),
                    FftEngine::Avx2Fma => (
                        duccq_avx2fma_fft,
                        duccq_avx2fma_ifft,
                        duccq_avx2fma_rfft,
                        duccq_avx2fma_irfft,
                        duccq_avx2fma_ifft_batch_rows,
                    ),
                    FftEngine::Avx2 => (
                        duccq_avx2_fft,
                        duccq_avx2_ifft,
                        duccq_avx2_rfft,
                        duccq_avx2_irfft,
                        duccq_avx2_ifft_batch_rows,
                    ),
                };
            }
        }
    }
    match active_engine() {
        FftEngine::Sse2 => (
            duccq_sse2_fft,
            duccq_sse2_ifft,
            duccq_sse2_rfft,
            duccq_sse2_irfft,
            duccq_sse2_ifft_batch_rows,
        ),
        FftEngine::Avx2Fma => (
            duccq_avx2fma_fft,
            duccq_avx2fma_ifft,
            duccq_avx2fma_rfft,
            duccq_avx2fma_irfft,
            duccq_avx2fma_ifft_batch_rows,
        ),
        FftEngine::Avx2 => (
            duccq_avx2_fft,
            duccq_avx2_ifft,
            duccq_avx2_rfft,
            duccq_avx2_irfft,
            duccq_avx2_ifft_batch_rows,
        ),
    }
}

// `Complex64` is `#[repr(C)] { re: f64, im: f64 }`, i.e. its memory layout is
// exactly the interleaved (re,im) pair stream that the C shim consumes, so we
// pass buffers straight through without any intermediate copy.

fn uninit_complex(len: usize) -> Vec<Complex64> {
    let mut v = Vec::with_capacity(len);
    // ducc0 writes every element of the output, so leaving them uninitialized
    // until the FFI call is fine.
    unsafe { v.set_len(len) };
    v
}

/// Forward complex FFT (no normalization).
pub fn fft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len() as c_int;
    let mut out = uninit_complex(x.len());
    let (f_fft, _, _, _, _) = eng();
    unsafe {
        f_fft(n, x.as_ptr() as *const f64, out.as_mut_ptr() as *mut f64);
    }
    out
}

/// Inverse complex FFT (normalized by 1/n).
pub fn ifft(x: &[Complex64]) -> Vec<Complex64> {
    let mut out = Vec::new();
    ifft_into(x, &mut out);
    out
}

/// [`ifft`] into a caller-owned buffer that is reused across calls.
///
/// `out` is resized to `x.len()` and every element is written by ducc, so the
/// buffer's previous contents can never leak into the result: reusing it is a
/// pure allocation/cache win and cannot change a single value.
pub fn ifft_into(x: &[Complex64], out: &mut Vec<Complex64>) {
    let n = x.len() as c_int;
    out.clear();
    out.reserve(x.len());
    // ducc writes all n output elements before returning.
    unsafe { out.set_len(x.len()) };
    let (_, f_ifft, _, _, _) = eng();
    unsafe {
        f_ifft(n, x.as_ptr() as *const f64, out.as_mut_ptr() as *mut f64);
    }
}


/// Batched inverse complex FFT: `k` independent transforms of length `n` held
/// as `k` contiguous rows, normalized by `1/n` like [`ifft`].
///
/// Results are bit-identical to `k` separate [`ifft_into`] calls (gated by
/// `ifft_batch_matches_scalar_transforms`), but ducc executes them ~1.83x
/// faster per transform at `n = blocklen`: its FFT SIMD vectorizes *across*
/// independent transforms, and the 1-D entry point never reaches those kernels
/// (its `exec_simple` passes the scalar type index to the plan), while the
/// multi-transform machinery disables vectorization for long transforms unless
/// the batch entry opts in (see the vendored-ducc `force_simul_batch` patch).
///
/// Both slices must hold exactly `k*n` values; ducc writes every output element
/// before returning, so the destination's prior contents are never read.
pub fn ifft_batch_rows(k: usize, n: usize, x: &[Complex64], out: &mut [Complex64]) {
    debug_assert_eq!(x.len(), k * n);
    debug_assert_eq!(out.len(), k * n);
    let (_, _, _, _, f_batch) = eng();
    unsafe {
        f_batch(
            k as c_int,
            n as c_int,
            x.as_ptr() as *const f64,
            out.as_mut_ptr() as *mut f64,
        );
    }
}

/// Forward real FFT -> half spectrum (len `n/2+1` complex), unnormalized.
pub fn rfft(x: &[f64]) -> Vec<Complex64> {
    let n = x.len() as c_int;
    let mut out = uninit_complex(x.len() / 2 + 1);
    let (_, _, f_rfft, _, _) = eng();
    unsafe {
        f_rfft(n, x.as_ptr(), out.as_mut_ptr() as *mut f64);
    }
    out
}

/// Forward FFT of real data, matching `scipy.fft.fft` bit-for-bit: pocketfft
/// computes a real (r2c) transform of the first `n/2+1` bins and reflects the
/// rest (`X[n-k] = conj(X[k])`), which is *not* the same rounding as a full
/// complex c2c of the data with zero imaginary parts.
pub fn fft_real_full(x: &[f64]) -> Vec<Complex64> {
    let mut out = Vec::new();
    fft_real_full_into(x, &mut out);
    out
}

/// [`fft_real_full`] into a caller-owned buffer that is reused across calls.
///
/// The r2c half spectrum lands in `out[0..n/2+1]` and the reflected half is
/// then built **in place** (`out[n-k] = conj(out[k])`), so the separate
/// `half` buffer and the copy out of it are gone. Every element is written
/// (ducc writes the first `n/2+1`, the loop below fills the rest), and the
/// read range `1..n/2` never overlaps the write range `n/2+1..n`, so the
/// values are identical to the allocating form by construction.
pub fn fft_real_full_into(x: &[f64], out: &mut Vec<Complex64>) {
    let n = x.len();
    out.clear();
    out.reserve(n);
    // ducc fills bins 0..=n/2; the reflection loop fills n/2+1..n.
    unsafe { out.set_len(n) };
    let (_, _, f_rfft, _, _) = eng();
    unsafe {
        f_rfft(n as c_int, x.as_ptr(), out.as_mut_ptr() as *mut f64);
    }
    for k in (1..n / 2).rev() {
        let c = out[k];
        out[n - k] = Complex64::new(c.re, -c.im);
    }
}


/// Inverse real FFT from a half spectrum, `n` real out, normalized by 1/n.
pub fn irfft(spectrum: &[Complex64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n];
    let (_, _, _, f_irfft, _) = eng();
    unsafe {
        f_irfft(n as c_int, spectrum.as_ptr() as *const f64, out.as_mut_ptr());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference outputs were produced by `scipy.fft` (scipy 1.18.0 in the
    // bundled 7.3.0 release python). The FFI must reproduce them bit-for-bit.
    fn to_complex(data: &[[f64; 2]]) -> Vec<Complex64> {
        data.iter().map(|d| Complex64::new(d[0], d[1])).collect()
    }

    /// Forward FFT at 1024 must reproduce scipy 1.18.0 (`scipy.fft._duccfft`)
    /// bit-for-bit. Reference produced by the bundled 7.3.0 release python.
    #[test]
    fn fft_matches_scipy_1024() {
        let input_const: &[[f64; 2]] = include!("../tests/data/scipy_in_1024.rs");
        let want_const: &[[f64; 2]] = include!("../tests/data/scipy_out_1024.rs");
        let x = to_complex(input_const);
        let want = to_complex(want_const);
        let y = fft(&x);
        assert_eq!(y.len(), want.len());
        for i in 0..want.len() {
            assert_eq!(y[i].re.to_bits(), want[i].re.to_bits(), "idx {i} re");
            assert_eq!(y[i].im.to_bits(), want[i].im.to_bits(), "idx {i} im");
        }
    }

    // PARITY PROBE: the reusable-buffer forms exist only to remove per-call
    // allocations and the intermediate `half` buffer; a reused buffer whose
    // stale contents or length leaked into a result would be a parity bug, so
    // pin that reuse (across repeated calls and across changing sizes) is
    // observationally identical to a fresh allocation.
    #[test]
    fn reused_buffers_match_fresh_allocations() {
        let n = 32768usize;
        let xr: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.3819660112501051).fract() - 0.5)
            .collect();
        let mut buf = Vec::new();
        for round in 0..3 {
            // Alternate full length and a shorter one: a buffer reused without
            // resetting its length would keep the tail of the previous call.
            let src: &[f64] = if round % 2 == 0 { &xr } else { &xr[..1024] };
            fft_real_full_into(src, &mut buf);
            let want = fft_real_full(src);
            assert_eq!(buf.len(), want.len());
            for i in 0..want.len() {
                assert_eq!(buf[i].re.to_bits(), want[i].re.to_bits(), "real_full idx {i} re");
                assert_eq!(buf[i].im.to_bits(), want[i].im.to_bits(), "real_full idx {i} im");
            }
        }

        let xc: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(xr[i], xr[(i + 7) % n]))
            .collect();
        let mut ibuf = Vec::new();
        for round in 0..3 {
            let src: &[Complex64] = if round % 2 == 0 { &xc } else { &xc[..1024] };
            ifft_into(src, &mut ibuf);
            let want = ifft(src);
            assert_eq!(ibuf.len(), want.len());
            for i in 0..want.len() {
                assert_eq!(ibuf[i].re.to_bits(), want[i].re.to_bits(), "ifft idx {i} re");
                assert_eq!(ibuf[i].im.to_bits(), want[i].im.to_bits(), "ifft idx {i} im");
            }
        }
    }

    // PARITY PROBE: does the c2r path reproduce the
    // real parts of the c2c backward transform bit-for-bit? No - even on an
    // exactly Hermitian spectrum the real parts differ (relative ~1e-7, e.g.
    // c2c 4.82243740404508792e-6 vs c2r 4.82243740410059907e-6 at n=32768),
    // and the pipeline's spectra are only approximately Hermitian anyway.
    // The c2r path would halve the six discarded-imaginary iffts in
    // demodblock (~27% of demod CPU by LD_DEMODTIME) and is therefore
    // permanently off the table under the bit-parity directive. This test
    // pins the divergence so the trade-off is never re-derived.
    #[test]
    fn irfft_matches_c2c_real_parts() {
        let n = 32768usize;
        // Hermitian spectrum built exactly like fft_real_full: forward r2c of
        // a real signal, then reflect bins 1..n/2 with negated imaginary part.
        let sig: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.3819660112501051).fract() - 0.5)
            .collect();
        let half = rfft(&sig);
        let herm = {
            let mut out = Vec::with_capacity(n);
            out.extend_from_slice(&half);
            for k in (1..n / 2).rev() {
                let c = half[k];
                out.push(Complex64::new(c.re, -c.im));
            }
            out
        };
        // Pipeline-like spectrum: small asymmetric perturbation, like the
        // MTF/rounded-filter products in demodblock.
        let mut nonherm = herm.clone();
        for (i, v) in nonherm.iter_mut().enumerate() {
            let s = ((i as f64) * 0.104).sin();
            v.re *= 1.0 + 1e-9 * s;
            v.im *= 1.0 + 1.2e-9 * (s + 0.3);
        }
        for (name, spec) in [("hermitian", &herm), ("nonherm", &nonherm)] {
            let c2c = ifft(spec);
            let c2r = irfft(spec, n);
            let mut worst = 0u64;
            let mut worst_idx = 0usize;
            for i in 0..n {
                let d = c2c[i].re.to_bits() as i64 - c2r[i].to_bits() as i64;
                let d = d.unsigned_abs();
                if d > worst {
                    worst = d;
                    worst_idx = i;
                }
            }
            println!(
                "{name}: exact={} worst_bits={worst} at {worst_idx}",
                worst == 0
            );
            if worst != 0 {
                println!(
                    "  c2c[{worst_idx}]={:+.17e} c2r[{worst_idx}]={:+.17e}",
                    c2c[worst_idx].re, c2r[worst_idx]
                );
            }
            // The whole point of this probe: under bit parity, c2r can never
            // replace the c2c iffts. Pin the divergence so a future change
            // that accidentally switches paths fails here.
            assert!(worst != 0, "{name}: c2r unexpectedly matched c2c");
        }
    }

    // Pipeline-size parity gate. The 1024-point test above is a canary, but the
    // demod kernel only ever transforms 32768-point blocks (`blocklen`), so the
    // size that actually decides output parity needs its own bit-for-bit check
    // against the reference library. Goldens: `scripts/gen_fft32768_ref.py`
    // under the bundled 7.3.0 python (scipy 1.18.0, numpy 2.4.6), raw
    // little-endian f64, complex files interleaved re/im.
    fn data_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name)
    }

    fn read_f64s(name: &str) -> Vec<f64> {
        let raw = std::fs::read(data_path(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        raw.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn read_cf(name: &str) -> Vec<Complex64> {
        let v = read_f64s(name);
        v.chunks_exact(2)
            .map(|c| Complex64::new(c[0], c[1]))
            .collect()
    }

    fn assert_same_bits(got: &[Complex64], want: &[Complex64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for i in 0..want.len() {
            assert_eq!(got[i].re.to_bits(), want[i].re.to_bits(), "{what}: idx {i} re");
            assert_eq!(got[i].im.to_bits(), want[i].im.to_bits(), "{what}: idx {i} im");
        }
    }

    // PARITY PROBE (2026-09-16 record, SUPERSEDED 2026-09-17 — see
    // `engine_census_vs_scipy_goldens`): the original AVX2 experiment (flags
    // hand-added to build.rs) measured 2.7x faster but divergent at idx 1 of
    // every golden. Re-measured with the runtime-selectable engines: on the
    // CURRENT toolchain, verified-AVX2 codegen (FMA/ymm instructions confirmed
    // in the objects) is bit-identical to scipy on every golden and 4/4
    // byte-identical end-to-end — and ~20% SLOWER per transform pair
    // (1412 vs 1181 us), i.e. the old speed record is also stale (likely
    // clang-cl fp-contraction/codegen drift). The engines stay available via
    // LD_FFT_ENGINE with sse2 as the default; do not flip the default on
    // micro-benchmarks alone.
    /// c2c forward and inverse at the pipeline's block length.
    #[test]
    fn fft_matches_scipy_32768() {
        let x = read_cf("fft32768_in.f64");
        assert_eq!(x.len(), 32768);
        assert_same_bits(&fft(&x), &read_cf("fft32768_fwd.f64"), "fft 32768");
        assert_same_bits(&ifft(&x), &read_cf("fft32768_inv.f64"), "ifft 32768");
    }

    /// `scipy.fft.fft` of a *real* array — what the kernel's `indata_fft` and
    /// `demod_fft` are: pocketfft computes the r2c half and reflects it, so this
    /// pins the r2c result *and* the reflection in one comparison.
    #[test]
    fn fft_real_full_matches_scipy_32768() {
        let real = read_f64s("rfft32768_in.f64");
        assert_eq!(real.len(), 32768);
        let got = fft_real_full(&real);
        let want = read_cf("rfft32768_full.f64");
        assert_eq!(got.len(), want.len());
        for i in 0..want.len() {
            // Bins 0 and n/2 are mathematically real for a real input; measured,
            // scipy's c2c gives +0.0 there while the r2c + conjugate-reflection
            // path gives -0.0 (2 cells of 65536). That signed zero is all that
            // differs, it stays +-0.0 through the filter products, and every
            // consumer of the forward spectra of real data reads the affected
            // cells only as a zero imaginary term — which is why the decodes are
            // byte-identical to the reference. Pinned here so the exception stays
            // checked instead of assumed; everything else must match bit-for-bit.
            let trivial_im = i == 0 || i == want.len() / 2;
            assert_eq!(got[i].re.to_bits(), want[i].re.to_bits(), "idx {i} re");
            if trivial_im {
                assert_eq!(got[i].im, 0.0, "idx {i}: trivial bin im must be zero");
                assert_eq!(want[i].im, 0.0, "idx {i}: golden trivial bin im");
            } else {
                assert_eq!(got[i].im.to_bits(), want[i].im.to_bits(), "idx {i} im");
            }
        }
    }

    // CENSUS (experimental engines): for each engine, how far is its output
    // from the scipy goldens? Runs all golden comparisons with the engine
    // forced, reporting divergence stats instead of asserting. The sse2 row
    // must be zero (it is the default build pinned by the tests above); the
    // avx2/avx2fma rows quantify what a future "make AVX2 bit-perfect" effort
    // would have to eliminate. `--nocapture` to see the table.
    fn census_engine(e: FftEngine) {
        // 1024 c2c canary.
        let input_const: &[[f64; 2]] = include!("../tests/data/scipy_in_1024.rs");
        let want_const: &[[f64; 2]] = include!("../tests/data/scipy_out_1024.rs");
        let x: Vec<Complex64> = input_const
            .iter()
            .map(|d| Complex64::new(d[0], d[1]))
            .collect();
        let want: Vec<Complex64> = want_const
            .iter()
            .map(|d| Complex64::new(d[0], d[1]))
            .collect();
        *TEST_OVERRIDE.write().unwrap() = Some(e);
        let y = fft(&x);
        let mut cells = 0u64;
        let mut worst_rel = 0.0f64;
        let mut worst_idx = 0usize;
        for i in 0..want.len() {
            for (g, w) in [(y[i].re, want[i].re), (y[i].im, want[i].im)] {
                if g.to_bits() != w.to_bits() {
                    cells += 1;
                    let d = (g - w).abs();
                    let rel = d / w.abs().max(1e-300);
                    if rel > worst_rel {
                        worst_rel = rel;
                        worst_idx = i;
                    }
                }
            }
        }
        println!(
            "{:8} c2c 1024: divergent cells {cells}/{} worst_rel {:.3e} at idx {worst_idx}",
            e.name(),
            want.len() * 2,
            worst_rel
        );

        // 32768 golden set: c2c fwd/inv + r2c-full.
        let xg = read_cf("fft32768_in.f64");
        for (what, got, want) in [
            ("fwd", fft(&xg), read_cf("fft32768_fwd.f64")),
            ("inv", ifft(&xg), read_cf("fft32768_inv.f64")),
        ] {
            let mut cells = 0u64;
            let mut worst_rel = 0.0f64;
            let mut worst_idx = 0usize;
            let mut ulp = 0u64;
            for i in 0..want.len() {
                for (g, w) in [(got[i].re, want[i].re), (got[i].im, want[i].im)] {
                    if g.to_bits() != w.to_bits() {
                        cells += 1;
                        let dg = g.to_bits() as i64 - w.to_bits() as i64;
                        let dw = w.to_bits() as i64 - g.to_bits() as i64;
                        ulp = ulp.max(dg.unsigned_abs().max(dw.unsigned_abs()));
                        let rel = (g - w).abs() / w.abs().max(1e-300);
                        if rel > worst_rel {
                            worst_rel = rel;
                            worst_idx = i;
                        }
                    }
                }
            }
            println!(
                "{e2:8} c2c 32768 {what}: divergent cells {cells}/{} worst_rel {worst_rel:.3e} at idx {worst_idx} max_ulp_gap {ulp}",
                want.len() * 2,
                e2 = e.name()
            );
        }
        let real = read_f64s("rfft32768_in.f64");
        let got = fft_real_full(&real);
        let want = read_cf("rfft32768_full.f64");
        let mut cells = 0u64;
        let mut worst_rel = 0.0f64;
        for i in 0..want.len() {
            let trivial_im = i == 0 || i == want.len() / 2;
            for (g, w) in [(got[i].re, want[i].re), (got[i].im, want[i].im)] {
                if g.to_bits() != w.to_bits() && !(trivial_im && g == 0.0 && w == 0.0) {
                    cells += 1;
                    let rel = (g - w).abs() / w.abs().max(1e-300);
                    worst_rel = worst_rel.max(rel);
                }
            }
        }
        println!(
            "{:8} r2c-full 32768: divergent cells {cells}/{} worst_rel {:.3e}",
            e.name(),
            want.len() * 2,
            worst_rel
        );
        *TEST_OVERRIDE.write().unwrap() = None;
    }

    #[test]
    fn engine_census_vs_scipy_goldens() {
        if !cpu_has_avx2() || !cpu_has_fma() {
            println!("skip: CPU lacks avx2/fma");
            return;
        }
        for e in [FftEngine::Sse2, FftEngine::Avx2, FftEngine::Avx2Fma] {
            census_engine(e);
        }
    }

    /// [`ifft_batch_rows`] must stay bit-identical to the scalar per-transform
    /// `ifft_into` it replaces in the demod kernel, and the pipeline's own batch
    /// sizes (2 for the early group, 4 for the late group) are the ones that
    /// matter. This is the gate for the vendored-ducc `force_simul_batch`
    /// opt-in, which hands the plan the SIMD type index that the 1-D entry
    /// point never passes; it is bit-exact because each SIMD lane runs the same
    /// operation sequence for its own transform.
    #[test]
    fn ifft_batch_matches_scalar_transforms() {
        let n = 32768usize;
        let golden_in = read_cf("fft32768_in.f64");
        let golden_inv = read_cf("fft32768_inv.f64");
        assert_eq!(golden_in.len(), n);
        // Every engine can be selected at runtime (`LD_FFT_ENGINE`), so every
        // engine's batch path has to agree with the scalar path and the golden.
        let have_avx2 = cpu_has_avx2() && cpu_has_fma();
        let engines: &[FftEngine] = if have_avx2 {
            &[FftEngine::Sse2, FftEngine::Avx2, FftEngine::Avx2Fma]
        } else {
            println!("note: CPU lacks avx2/fma; sse2 only");
            &[FftEngine::Sse2]
        };
        let mut st = 0xd1b5_4a32_d192_ed03u64;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            ((st >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        for engine in engines {
        *TEST_OVERRIDE.write().unwrap() = Some(*engine);
        for k in [1usize, 2, 3, 4, 6, 8] {
            let mut x = vec![Complex64::new(0.0, 0.0); k * n];
            // Row 0 is the scipy 1.18 golden input, so the batch is also
            // checked against the reference library's own numbers.
            x[..n].copy_from_slice(&golden_in);
            for t in 1..k {
                for i in 0..n {
                    let scale = 2f64.powi(((t + i % 5) as i32 % 30) - 15);
                    x[t * n + i] = Complex64::new(next() * scale, next() * scale);
                }
            }
            let mut want = vec![Complex64::new(0.0, 0.0); k * n];
            for t in 0..k {
                let row: Vec<Complex64> = x[t * n..(t + 1) * n].to_vec();
                want[t * n..(t + 1) * n].copy_from_slice(&ifft(&row));
            }
            let mut got = vec![Complex64::new(0.0, 0.0); k * n];
            ifft_batch_rows(k, n, &x, &mut got);
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g.re.to_bits(),
                    w.re.to_bits(),
                    "batched ifft ({}) k={k} diverged at flat index {i} (re)",
                    engine.name()
                );
                assert_eq!(
                    g.im.to_bits(),
                    w.im.to_bits(),
                    "batched ifft ({}) k={k} diverged at flat index {i} (im)",
                    engine.name()
                );
            }
            for i in 0..n {
                assert_eq!(
                    got[i].re.to_bits(),
                    golden_inv[i].re.to_bits(),
                    "batched ifft ({}) k={k} row 0 differs from the scipy golden at bin {i} (re)",
                    engine.name()
                );
                assert_eq!(
                    got[i].im.to_bits(),
                    golden_inv[i].im.to_bits(),
                    "batched ifft ({}) k={k} row 0 differs from the scipy golden at bin {i} (im)",
                    engine.name()
                );
            }
        }
        }
        *TEST_OVERRIDE.write().unwrap() = None;
    }

    /// SIMD lane counts compiled into each engine: `(fft1d_simdlen<double>,
    /// native_simd<double>::size())`. Asserts rather than prints, because the
    /// first version of this build silently dropped `-mavx2` under clang-cl
    /// (`flag_if_supported` probed a flag the driver then discarded), leaving
    /// three identical SSE2 engines behind three names.
    #[test]
    fn engine_simd_widths() {
        // The lane counts are an x86-64 statement: off x86, build.rs compiles
        // all three engines from the same portable TU, so there is no
        // "4-lane" engine to assert about.
        if !cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            println!("skip: not x86; the AVX2 engines are x86-only");
            return;
        }
        unsafe {
            let sse2 = (duccq_sse2_simdlen(), duccq_sse2_native_simdlen());
            let avx2fma = (duccq_avx2fma_simdlen(), duccq_avx2fma_native_simdlen());
            let avx2 = (duccq_avx2_simdlen(), duccq_avx2_native_simdlen());
            println!(
                "engine widths: sse2 {sse2:?}  avx2 {avx2:?}  avx2fma {avx2fma:?} \
                 (fft1d_simdlen<double>, native_simd<double>::size())"
            );
            assert_eq!(sse2, (2, 2), "sse2 engine must be the 128-bit replica");
            assert_eq!(avx2, (4, 4), "avx2 engine lost its -mavx2 flags");
            assert_eq!(avx2fma, (4, 4), "avx2fma engine lost its -mavx2 flags");
        }
    }

    /// Per-size timing for every engine, from L1-resident to the pipeline's
    /// block length. All three are equal at every size, which is the signature
    /// of the engines never differing in executed code: the 1-D contiguous
    /// entry point ducc takes here hands the plan the *scalar* type index, so
    /// the SIMD kernels are compiled in but never run. Only the batched entry
    /// (`ifft_batch_rows`) reaches them, and there the lane width still makes no
    /// difference — the win comes from batching independent transforms.
    #[test]
    fn engine_size_sweep() {
        if !cpu_has_avx2() || !cpu_has_fma() {
            println!("skip: CPU lacks avx2/fma");
            return;
        }
        let engines = [FftEngine::Sse2, FftEngine::Avx2, FftEngine::Avx2Fma];
        for n in [1024usize, 4096, 16384, 32768] {
            let xc: Vec<Complex64> = (0..n)
                .map(|i| {
                    Complex64::new(
                        ((i as f64) * 0.38).fract() - 0.5,
                        ((i as f64) * 0.11).fract(),
                    )
                })
                .collect();
            // Build plans for every engine first.
            for e in engines {
                *TEST_OVERRIDE.write().unwrap() = Some(e);
                let s = fft(&xc);
                std::hint::black_box(&ifft(&s));
            }
            let mut res: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            for _ in 0..9 {
                for (i, e) in engines.iter().enumerate() {
                    *TEST_OVERRIDE.write().unwrap() = Some(*e);
                    let iters = (1 << 20) / n;
                    let t0 = std::time::Instant::now();
                    for _ in 0..iters {
                        let s = fft(&xc);
                        let s2 = ifft(&s);
                        std::hint::black_box(&s2);
                    }
                    res[i].push(t0.elapsed().as_secs_f64() * 1e6 / iters as f64);
                }
            }
            *TEST_OVERRIDE.write().unwrap() = None;
            let mut line = format!("n={n:>6}:");
            for (i, e) in engines.iter().enumerate() {
                res[i].sort_by(|a, b| a.partial_cmp(b).unwrap());
                line.push_str(&format!("  {} {:>7.2} us", e.name(), res[i][res[i].len() / 2]));
            }
            println!("{line}");
        }
    }

    /// Flags-effectiveness check + interleaved engine comparison.
    ///
    /// The engines are cycled round-robin inside the measurement loop (a
    /// single sequential pass is not trustworthy: the first engine measured
    /// also pays plan construction and any cold-cache effects) and each
    /// result is the median of many rounds. Measures both transform shapes
    /// the pipeline actually runs at the block length: a c2c forward+inverse
    /// pair and the r2c (`real_full`) forward + inverse pair.
    #[test]
    fn engine_speed_census() {
        if !cpu_has_avx2() || !cpu_has_fma() {
            println!("skip: CPU lacks avx2/fma");
            return;
        }
        let n = 32768usize;
        let xc: Vec<Complex64> = (0..n)
            .map(|i| {
                Complex64::new(
                    ((i as f64) * 0.38).fract() - 0.5,
                    ((i as f64) * 0.11).fract(),
                )
            })
            .collect();
        let xr: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.3819660112501051).fract() - 0.5).collect();

        let engines = [FftEngine::Sse2, FftEngine::Avx2, FftEngine::Avx2Fma];
        // Build every engine's plan first so construction never lands in a
        // timed sample.
        let mut buf = Vec::new();
        for e in engines {
            *TEST_OVERRIDE.write().unwrap() = Some(e);
            let s = fft(&xc);
            std::hint::black_box(&ifft(&s));
            fft_real_full_into(&xr, &mut buf);
            std::hint::black_box(&buf);
        }

        let mut c2c: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut r2c: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let rounds = 15;
        let iters = 20;
        for _ in 0..rounds {
            for (i, e) in engines.iter().enumerate() {
                *TEST_OVERRIDE.write().unwrap() = Some(*e);
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let s = fft(&xc);
                    let s2 = ifft(&s);
                    std::hint::black_box(&s2);
                }
                c2c[i].push(t0.elapsed().as_secs_f64() * 1e6 / iters as f64);

                let t1 = std::time::Instant::now();
                for _ in 0..iters {
                    fft_real_full_into(&xr, &mut buf);
                    let s2 = ifft(&buf);
                    std::hint::black_box(&s2);
                }
                r2c[i].push(t1.elapsed().as_secs_f64() * 1e6 / iters as f64);
            }
        }
        *TEST_OVERRIDE.write().unwrap() = None;
        for (i, e) in engines.iter().enumerate() {
            let med = |v: &mut Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            };
            println!(
                "{:8}: c2c fft+ifft {:>6.0} us (min {:>6.0}) | r2c+ifft {:>6.0} us (min {:>6.0})",
                e.name(),
                med(&mut c2c[i]),
                c2c[i][0],
                med(&mut r2c[i]),
                r2c[i][0]
            );
        }
    }

    // PERF PROBE (temporary): split `fft_real_full_into` into its r2c and its
    // Hermitian reflection to see which one holds the time.
    #[test]
    fn bench_reflection_split() {
        use std::time::Instant;
        let n = 32768usize;
        let xr: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.3819660112501051).fract() - 0.5).collect();
        let iters = 400;
        let mut sink = 0.0f64;
        let mut buf: Vec<Complex64> = Vec::new();
        let t0 = Instant::now();
        for _ in 0..iters {
            fft_real_full_into(&xr, &mut buf);
            sink += buf[7].re;
        }
        eprintln!("PERF rfft+reflect into reused buf: {:?}/call", t0.elapsed() / iters);
        let t0 = Instant::now();
        for _ in 0..iters {
            let h = rfft(&xr);
            sink += h[7].re;
        }
        eprintln!("PERF rfft (fresh alloc): {:?}/call", t0.elapsed() / iters);
        let t0 = Instant::now();
        for _ in 0..iters {
            for k in (1..n / 2).rev() {
                let c = buf[k];
                buf[n - k] = Complex64::new(c.re, -c.im);
            }
            sink += buf[3].re;
        }
        eprintln!("PERF reflection loop only: {:?}/call", t0.elapsed() / iters);
        let t0 = Instant::now();
        for _ in 0..iters {
            let (head, tail) = buf.split_at_mut(n / 2);
            for (dst, src) in tail.iter_mut().rev().zip(head[1..].iter()) {
                dst.re = src.re;
                dst.im = -src.im;
            }
            sink += buf[3].re;
        }
        eprintln!("PERF reflection split_at_mut form: {:?}/call", t0.elapsed() / iters);
        eprintln!("PERF sink {sink}");
    }

    // PERF PROBE: measure plan-cache reuse for the pipeline sizes.
    #[test]
    fn bench_plan_reuse() {
        use std::time::Instant;
        let n = 32768usize;
        let x: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = (i as f64) * 0.618033988749895;
                Complex64::new(t.fract() - 0.5, (t * 1.7).fract() - 0.5)
            })
            .collect();
        let t0 = Instant::now();
        let _ = fft(&x);
        let _ = ifft(&x);
        eprintln!("PERF first fft+ifft (cold): {:?}", t0.elapsed());
        let iters = 200;
        let mut sink = 0.0f64;
        let t0 = Instant::now();
        for _ in 0..iters {
            let f = fft(&x);
            let b = ifft(&f);
            sink += b[1].re;
        }
        eprintln!("PERF fft+ifft n=32768: {:?}/pair, sink {:.6}", t0.elapsed() / iters, sink);
        let xr: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.3819660112501051).fract() - 0.5).collect();
        let t0 = Instant::now();
        let _ = rfft(&xr);
        eprintln!("PERF first rfft (cold): {:?}", t0.elapsed());
        let t0 = Instant::now();
        for _ in 0..iters {
            let h = rfft(&xr);
            let b = irfft(&h, n);
            sink += b[1];
        }
        eprintln!("PERF rfft+irfft n=32768: {:?}/pair, sink {:.6}", t0.elapsed() / iters, sink);
    }

    // PERF PROBE: does the global plan cache (a single Mutex around a 10-entry
    // array) serialise concurrent FFT calls across worker threads?
    #[test]
    fn bench_fft_thread_scaling() {
        use std::time::Instant;
        const N: usize = 32768;
        let xr: Vec<f64> = (0..N).map(|i| ((i as f64) * 0.3819660112501051).fract() - 0.5).collect();
        let xc: Vec<Complex64> = (0..N)
            .map(|i| Complex64::new(xr[i], xr[(i + 1) % N]))
            .collect();
        let iters = 100usize;
        let work = |xr: &[f64], xc: &[Complex64], iters: usize| -> f64 {
            let mut sink = 0.0f64;
            for _ in 0..iters {
                let h = rfft(xr);
                let b = irfft(&h, N);
                sink += b[1];
                let f = fft(xc);
                let c = ifft(&f);
                sink += c[1].re;
            }
            sink
        };
        let s = work(&xr, &xc, 20);
        let t0 = Instant::now();
        let s1 = work(&xr, &xc, iters);
        let single = t0.elapsed().as_secs_f64();
        eprintln!("PERF 1 thread: {:.3} ms/iter, sink {:.6}", single * 1000.0 / iters as f64, s1 + s);

        for threads in [4usize, 8, 12] {
            let t0 = Instant::now();
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    let xr = xr.clone();
                    let xc = xc.clone();
                    std::thread::spawn(move || work(&xr, &xc, iters))
                })
                .collect();
            let mut tot = 0.0f64;
            for h in hs {
                tot += h.join().unwrap();
            }
            let par = t0.elapsed().as_secs_f64();
            eprintln!(
                "PERF {threads} threads: {:.3} ms/iter/thread, sink {:.6}, scaling {:.2}x",
                par * 1000.0 / iters as f64,
                tot,
                single * threads as f64 / par
            );
        }
    }

    // TEMP probe: real linked lib vs scipy at all pipeline sizes. Reads dumps
    // from a scratch directory outside the repo, so it is ignored by default;
    // run it with `cargo test -- --ignored` where those dumps exist.
    #[test]
    #[ignore = "needs the work/ducc_gate dumps from a scratch scipy run"]
    fn probe_sizes_vs_scipy() {
        let dir = "C:/Software/freebuff_rust_ld-decode/work/ducc_gate";
        for n in [4096usize, 8192, 16384, 32768, 983072] {
            let raw = std::fs::read(format!("{dir}/input_{n}.bin")).unwrap();
            let x: Vec<Complex64> = (0..n)
                .map(|i| {
                    let re = f64::from_le_bytes(raw[i * 16..i * 16 + 8].try_into().unwrap());
                    let im = f64::from_le_bytes(raw[i * 16 + 8..i * 16 + 16].try_into().unwrap());
                    Complex64::new(re, im)
                })
                .collect();
            for (name, y) in [("fwd", fft(&x)), ("inv", ifft(&x))] {
                let want_raw = std::fs::read(format!("{dir}/scipy_{name}_{n}.bin")).unwrap();
                let mut ndiff = 0usize;
                let mut maxbits = 0u64;
                for i in 0..n {
                    let wr = f64::from_le_bytes(want_raw[i * 16..i * 16 + 8].try_into().unwrap());
                    let wi = f64::from_le_bytes(want_raw[i * 16 + 8..i * 16 + 16].try_into().unwrap());
                    for (a, b) in [(y[i].re, wr), (y[i].im, wi)] {
                        if a.to_bits() != b.to_bits() {
                            ndiff += 1;
                            maxbits = maxbits.max(a.to_bits().abs_diff(b.to_bits()));
                        }
                    }
                }
                eprintln!("probe n={n} {name}: ndiff={ndiff}/{} maxbitdist={maxbits}", n * 2);
            }
            // r2c (real input -> half spectrum)
            let rraw = std::fs::read(format!("{dir}/input_r_{n}.bin")).unwrap();
            let r: Vec<f64> = (0..n)
                .map(|i| f64::from_le_bytes(rraw[i * 8..i * 8 + 8].try_into().unwrap()))
                .collect();
            let half = rfft(&r);
            let want_raw = std::fs::read(format!("{dir}/scipy_r2c_{n}.bin")).unwrap();
            let mut ndiff = 0usize;
            for i in 0..half.len() {
                let wr = f64::from_le_bytes(want_raw[i * 16..i * 16 + 8].try_into().unwrap());
                let wi = f64::from_le_bytes(want_raw[i * 16 + 8..i * 16 + 16].try_into().unwrap());
                if half[i].re.to_bits() != wr.to_bits() {
                    ndiff += 1;
                }
                if half[i].im.to_bits() != wi.to_bits() {
                    ndiff += 1;
                }
            }
            eprintln!("probe n={n} r2c: ndiff={ndiff}/{}", half.len() * 2);
        }
    }
}