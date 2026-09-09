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
    let n = x.len() as c_int;
    let mut out = uninit_complex(x.len());
    unsafe {
        duccq_ifft(n, x.as_ptr() as *const f64, out.as_mut_ptr() as *mut f64);
    }
    out
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
    let n = x.len();
    let half = rfft(x); // n/2+1 values
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&half);
    for k in (1..n / 2).rev() {
        let c = half[k];
        out.push(Complex64::new(c.re, -c.im));
    }
    out
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

    // TEMP probe: real linked lib vs scipy at all pipeline sizes.
    #[test]
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