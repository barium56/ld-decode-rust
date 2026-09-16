//! FFI bindings to the vendored ducc0 FFT library (the exact FFT behind
//! `scipy.fft` in scipy >= 1.18), compiled via `build.rs`.
//!
//! These reproduce scipy's `_duccfft` rounding **bit-for-bit** on x86-64 when
//! the library is built the same way (SSE2 homegrown SIMD, single-threaded),
//! which is required for the decoder's output to match the reference bit-exactly.

use rustfft::num_complex::Complex64;
use std::os::raw::c_int;

unsafe extern "C" {
    /// Forward complex FFT, no normalization. `in`/`out` hold `2n` doubles (n complex).
    fn duccq_fft(n: c_int, inp: *const f64, out: *mut f64);
    /// Inverse complex FFT, normalized by 1/n.
    fn duccq_ifft(n: c_int, inp: *const f64, out: *mut f64);
    /// Forward real FFT -> half spectrum, `n/2+1` complex out (unnormalized).
    fn duccq_rfft(n: c_int, inp: *const f64, out: *mut f64);
    /// Inverse real FFT from a half spectrum, `n` real out, normalized by 1/n.
    fn duccq_irfft(n: c_int, inp: *const f64, out: *mut f64);
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
    unsafe {
        duccq_fft(n, x.as_ptr() as *const f64, out.as_mut_ptr() as *mut f64);
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
    unsafe {
        duccq_ifft(n, x.as_ptr() as *const f64, out.as_mut_ptr() as *mut f64);
    }
}


/// Forward real FFT -> half spectrum (len `n/2+1` complex), unnormalized.
pub fn rfft(x: &[f64]) -> Vec<Complex64> {
    let n = x.len() as c_int;
    let mut out = uninit_complex(x.len() / 2 + 1);
    unsafe {
        duccq_rfft(n, x.as_ptr(), out.as_mut_ptr() as *mut f64);
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
    unsafe {
        duccq_rfft(n as c_int, x.as_ptr(), out.as_mut_ptr() as *mut f64);
    }
    for k in (1..n / 2).rev() {
        let c = out[k];
        out[n - k] = Complex64::new(c.re, -c.im);
    }
}


/// Inverse real FFT from a half spectrum, `n` real out, normalized by 1/n.
pub fn irfft(spectrum: &[Complex64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n];
    unsafe {
        duccq_irfft(n as c_int, spectrum.as_ptr() as *const f64, out.as_mut_ptr());
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

    // PARITY PROBE (measured, REJECTED 2026-09-16): an AVX2 build of the
    // vendored ducc0 (`-mavx2 -mfma` added next to `/O2` in build.rs) is 2.7x
    // faster on exactly the transform the decoder spends its time in — with a
    // forced rebuild, fft+ifft n=32768 goes 1072 us -> 394 us per pair and
    // rfft+irfft 682 us -> 344 us — but it rounds differently at *every* size
    // this crate uses: this test fails at idx 1, `fft_real_full` at idx 1 and
    // the 1024 canary at idx 1. Every transform the pipeline runs is
    // 32768-point (2 r2c + 6 c2c per block, 21 blocks per field, plus small
    // audio slices), so there is no subset of sizes where AVX2 could be
    // enabled — permanently off the table under the bit-parity directive, like
    // the c2r path. To re-measure: add the two flags to build.rs, `touch` it
    // (cargo does not rebuild on an env var change) and run these tests. Do not
    // leave the flags in a default build.
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