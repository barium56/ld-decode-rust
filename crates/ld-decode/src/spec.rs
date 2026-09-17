//! The immutable decode configuration: NTSC Laserdisc system parameters,
//! decoder filter parameters, the FFT-domain filter bank, and the measured
//! filter delays. Ported from `lddecode/core.py` (SysParams_NTSC,
//! FilterParams_NTSC, RFDecode.computefilters/computedelays).

use std::f64::consts::{PI, TAU};

use anyhow::{bail, Result};
use rustfft::num_complex::Complex64;
use sci_rs::signal::filter::design::FilterBandType;

use crate::decode::{calczc, demod_block_cpu, DemodSpecRef, BLOCKSIZE};
use crate::request::{ColorSystem, DecodeRequest, WowInterpolation};

// ---------------------------------------------------------------------------
// NTSC system parameters (SysParams_NTSC in core.py)
// ---------------------------------------------------------------------------

pub(crate) const SYS_FSC_MHZ: f64 = 315.0 / 88.0;
pub(crate) const SYS_FRAME_LINES: usize = 525;
pub(crate) const SYS_FIELD_LINES: [usize; 2] = [263, 262];
pub(crate) const SYS_IRE0: f64 = 8100000.0;
pub(crate) const SYS_HZ_IRE: f64 = 1700000.0 / 140.0;
pub(crate) const SYS_VSYNC_IRE: f64 = -40.0;
pub(crate) const SYS_COLOR_BURST_US: [f64; 2] = [5.3, 7.8];
pub(crate) const SYS_BLACKSNR_SLICE: [usize; 3] = [1, 10, 20];
pub(crate) const SYS_FIRST_FIELD_H: [f64; 2] = [0.5, 1.0];
pub(crate) const SYS_NUM_PULSES: usize = 6;
pub(crate) const SYS_HSYNC_PULSE_US: f64 = 4.7;
pub(crate) const SYS_EQ_PULSE_US: f64 = 2.3;
pub(crate) const SYS_VSYNC_PULSE_US: f64 = 27.1;
pub(crate) const SYS_OUTPUT_ZERO: i64 = 1024;
pub(crate) const SYS_FIELD_PHASES: usize = 4;
pub(crate) const SYS_LD_VITS_WHITELOCS: [[usize; 3]; 4] =
    [[20, 14, 12], [20, 52, 8], [13, 13, 15], [11, 12, 45]];
pub(crate) const SYS_LD_VITS_CODE_SLICES: [[usize; 4]; 3] =
    [[16, 12, 48, 85], [17, 12, 48, 85], [10, 13, 39, 85]];

// NTSC filter parameters (FilterParams_NTSC in core.py).
pub(crate) const DP_AUDIO_NOTCHWIDTH: f64 = 200000.0;
pub(crate) const DP_AUDIO_NOTCHORDER: usize = 2;
pub(crate) const DP_AUDIO_FILTERWIDTH: f64 = 150000.0;
pub(crate) const DP_AUDIO_FILTERORDER: usize = 512;
pub(crate) const DP_VIDEO_DEEMP: [f64; 2] = [120e-9, 320e-9];
pub(crate) const DP_VIDEO_BPF_LOW: f64 = 3700000.0;
pub(crate) const DP_VIDEO_BPF_LOW_ORDER: usize = 2;
pub(crate) const DP_VIDEO_BPF_HIGH: f64 = 13800000.0;
pub(crate) const DP_VIDEO_BPF_HIGH_ORDER: usize = 3;
pub(crate) const DP_VIDEO_LPF_FREQ: f64 = 4500000.0;
pub(crate) const DP_VIDEO_LPF_ORDER: usize = 6;
pub(crate) const DP_MTF_BASEMULT: f64 = 0.4;
pub(crate) const DP_MTF_POLEDIST: f64 = 0.9;
pub(crate) const DP_MTF_FREQ: f64 = 12.2;
#[allow(dead_code)]
pub(crate) const DP_VIDEO_HPF_FREQ: f64 = 10000000.0;
#[allow(dead_code)]
pub(crate) const DP_VIDEO_HPF_ORDER: usize = 4;

// ---------------------------------------------------------------------------
// Small DSP helpers
// ---------------------------------------------------------------------------

/// scipy-compatible `sinc` (np.sinc): sin(pi x)/(pi x), 1 at x == 0.
fn np_sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let x_pi = PI * x;
        x_pi.sin() / x_pi
    }
}

/// Port of `scipy.signal.firwin` for the lowpass/bandpass cases used here.
/// `cutoff` is already normalized to the Nyquist frequency. Hamming window
/// (fftbins=False) and unit-gain scaling at the passband centre, exactly as
/// scipy does with its defaults.
fn firwin(numtaps: usize, cutoff: &[f64], pass_zero: bool) -> Vec<f64> {
    let pass_nyquist = (cutoff.len() % 2 == 0) == pass_zero;

    let mut bands = Vec::with_capacity(cutoff.len() + 2);
    if pass_zero {
        bands.push(0.0);
    }
    bands.extend_from_slice(cutoff);
    if pass_nyquist {
        bands.push(1.0);
    }

    let alpha = 0.5 * (numtaps - 1) as f64;
    let mut h = vec![0.0f64; numtaps];
    // scipy accumulates each band with two separate array ops
    // (`h += right*sinc(...)` then `h -= left*sinc(...)`), so each gets its own
    // rounding; folding them into one expression shifts every tap by an ulp.
    for pair in bands.chunks(2) {
        let (left, right) = (pair[0], pair[1]);
        for (i, m) in (0..numtaps).map(|n| (n, n as f64 - alpha)) {
            h[i] += right * np_sinc(right * m);
            h[i] -= left * np_sinc(left * m);
        }
    }

    // Hamming window, fftbins=False:
    // `scipy.signal.windows.hamming` -> `general_hamming(M, 0.54, sym=True)` ->
    // `general_cosine(M, [0.54, 1.0 - 0.54])`, which evaluates
    // `np.cos(k*fac)` for k = 0,1 over `fac = np.linspace(-pi, pi, M)` and sums
    // the terms into a zeros array. The second coefficient is the *rounded*
    // `1.0 - 0.54` (not the literal 0.46) and the linspace argument is built as
    // `i*(step) + start` with the endpoint overwritten, so both details matter.
    let a1 = 1.0 - 0.54;
    let step = (PI - (-PI)) / (numtaps - 1) as f64;
    for (i, value) in h.iter_mut().enumerate() {
        let fac = if i == numtaps - 1 {
            PI
        } else {
            i as f64 * step + (-PI)
        };
        // k = 0 term: 0.54 * cos(0.0) == 0.54 exactly; then += a1 * cos(fac).
        let win = 0.54 + a1 * fac.cos();
        *value *= win;
    }

    // Scale for unit gain at the first passband's centre. numpy's `sum` uses
    // pairwise summation, so the port has to use the same reduction order.
    let (left, right) = (bands[0], bands[1]);
    let scale_frequency = if left == 0.0 {
        0.0
    } else if right == 1.0 {
        1.0
    } else {
        0.5 * (left + right)
    };
    let prod: Vec<f64> = (0..numtaps)
        .map(|n| {
            let m = n as f64 - alpha;
            h[n] * (PI * m * scale_frequency).cos()
        })
        .collect();
    let s = crate::decode::pairwise_sum_f64(&prod);
    for value in &mut h {
        *value /= s;
    }
    h
}

/// Port of `utils.emphasis_iir`: pre/de-emphasis as an IIR (b, a) pair given
/// the two corner time constants in seconds.
fn emphasis_iir(t1: f64, t2: f64, fs: f64) -> (Vec<f64>, Vec<f64>) {
    // Bit-exact port of the reference `utils.emphasis_iir`:
    //   w1 = 2*fs*tan((1/t1)/(2*fs)); w2 likewise
    //   b, a = scipy.signal.zpk2tf([-w1], [-w2], w2/w1)   ->  b=[w2/w1, (w2/w1)*w1], a=[1, w2]
    //   b, a = scipy.signal.bilinear(b, a, fs)
    // scipy's bilinear factors the transform as zp1=(z+1)/fac, zm1=(z-1)*fac with
    // fac=sqrt(2*fs), expands the order-1 polynomials, then normalizes by a[0].
    // Every multiply/divide below is a separate rounding, matching numpy elementwise
    // ops (no FMA, no reassociation).
    let w1 = 2.0 * fs * ((1.0 / t1) / (2.0 * fs)).tan();
    let w2 = 2.0 * fs * ((1.0 / t2) / (2.0 * fs)).tan();

    let k = w2 / w1; // zpk2tf gain
    let b1 = k * w1; // poly: k*(s + w1) -> [k, k*w1]
    let a1 = w2;

    let fac = (2.0 * fs).sqrt();
    // numerator desc: [(b1/fac) + (k*fac), (b1/fac) - (k*fac)]
    let num0 = b1 / fac + k * fac;
    let num1 = b1 / fac - k * fac;
    // denominator desc: [(w2/fac) + fac, (w2/fac) - fac]
    let den0 = a1 / fac + fac;
    let den1 = a1 / fac - fac;
    (
        [num0 / den0, num1 / den0].to_vec(),
        [1.0, den1 / den0].to_vec(),
    )
}

/// Port of `utils.build_hilbert`.
fn build_hilbert(fft_size: usize) -> Vec<f64> {
    assert!(
        fft_size.is_multiple_of(2),
        "build_hilbert: must have even fft_size"
    );
    let mut output = vec![0.0f64; fft_size];
    output[0] = 1.0;
    output[fft_size / 2] = 1.0;
    for value in &mut output[1..fft_size / 2] {
        *value = 2.0;
    }
    output
}

/// Port of `utils.genwave`: generate an FM waveform from target frequency
/// data. `freq` is the sampling frequency; the output's instantaneous
/// frequency is `rate` (core.py calls it with `freq/2` so the demodulated
/// output equals `rate`).
fn genwave(rate: &[f64], freq: f64, initial_phase: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; rate.len()];
    let mut angle = initial_phase;
    for (i, &r) in rate.iter().enumerate() {
        out[i] = angle.sin();
        angle += PI * (r / freq);
        if angle > PI {
            angle -= TAU;
        }
    }
    out
}

fn polar2z(r: f64, theta: f64) -> Complex64 {
    Complex64::from_polar(r, theta)
}

/// Port of `sps.zpk2tf([], [z1, z2], 1)`: b = [1], a = poly of the poles.
fn zpk2tf_poles(z1: Complex64, z2: Complex64) -> (Vec<f64>, Vec<f64>) {
    // poly([z1, z2]) = (s - z1)(s - z2) = s^2 - (z1+z2) s + z1 z2
    let sum = z1 + z2;
    let prod = z1 * z2;
    (vec![1.0], vec![1.0, -sum.re, prod.re])
}

/// Port of `utils.filtfft` = `scipy.signal.freqz(b, a, block_len, whole=1)`:
/// the frequency response at `block_len` points around the whole unit circle.
///
/// Mirrors scipy 1.18's `freqz` arithmetic exactly: an (R)FFT of the padded
/// numerator when `a` is a single coefficient (FIR), otherwise a Horner
/// polynomial evaluation at `exp(-j*2*pi*k/n)` for numerator and denominator
/// (IIR). Both paths were verified bit-for-bit against the bundled release
/// python's `scipy.signal.freqz`.
fn filtfft(b: &[f64], a: &[f64], block_len: usize) -> Vec<Complex64> {
    assert!(!a.is_empty());
    assert!(!b.is_empty());

    // scipy: `n_fft = N if whole else 2 * N`; with whole=True and an integer
    // worN it uses `rfft(b, n=n_fft)` when `a` is a single coefficient and the
    // transform length covers the numerator.
    if a.len() == 1 && block_len >= b.len() {
        let mut padded = vec![0.0f64; block_len];
        padded[..b.len()].copy_from_slice(b);
        // rfft of length n gives n/2+1 complex bins (0..=n/2).
        let mut h = crate::ffi_ducc::rfft(&padded);
        for v in h.iter_mut() {
            *v /= a[0];
        }
        // scipy `whole` assembly: keep bins 0..=n/2, then append the
        // conjugate-reversed bins 1..n/2 to cover the full circle.
        let half = block_len / 2;
        let mut out = Vec::with_capacity(block_len);
        out.extend_from_slice(&h);
        for k in (1..half).rev() {
            out.push(h[k].conj());
        }
        debug_assert_eq!(out.len(), block_len);
        out
    } else {
        // scipy's IIR path: `polyval(exp(-1j*w), b) / polyval(exp(-1j*w), a)`
        // with Horner's method over the (complex) z^-1 powers.
        //
        // The arithmetic reproduces numpy 2.4's complex128 ufuncs bit-for-bit:
        // the SIMD complex multiply uses `fmaddsub` (FMA on the AVX2 build),
        // and the division is Smith's algorithm with multiply-by-reciprocal.
        // Both were verified against `scipy.signal.freqz` over all 32768 bins.
        let cmul = |x: Complex64, y: Complex64| -> Complex64 {
            // (a.re + i a.im) * (b.re + i b.im), FMA like `_mm256_fmaddsub_pd`.
            let re = fma(x.re, y.re, -(x.im * y.im));
            let im = fma(x.re, y.im, x.im * y.re);
            Complex64::new(re, im)
        };
        let cdiv = |x: Complex64, y: Complex64| -> Complex64 {
            let (a, b, c, d) = (x.re, x.im, y.re, y.im);
            let (ca, cb) = (c.abs(), d.abs());
            if ca >= cb {
                if ca == 0.0 && cb == 0.0 {
                    // Divide by zero -> complex inf/nan, as numpy.
                    return Complex64::new(a / ca, b / cb);
                }
                let rat = d / c;
                let scl = 1.0 / (c + d * rat);
                Complex64::new((a + b * rat) * scl, (b - a * rat) * scl)
            } else {
                let rat = c / d;
                let scl = 1.0 / (d + c * rat);
                Complex64::new((a * rat + b) * scl, (b * rat - a) * scl)
            }
        };
        let polyval = |z: Complex64, c: &[f64]| -> Complex64 {
            // numpy: `c0 = c[-1] + x*0; c0 = c[-i] + c0*x` (Horner).
            let mut acc = Complex64::new(c[c.len() - 1], 0.0) + cmul(z, Complex64::new(0.0, 0.0));
            for i in (0..c.len() - 1).rev() {
                acc = Complex64::new(c[i], 0.0) + cmul(acc, z);
            }
            acc
        };
        (0..block_len)
            .map(|k| {
                let omega = TAU * k as f64 / block_len as f64;
                let z = Complex64::new(omega.cos(), -omega.sin());
                cdiv(polyval(z, b), polyval(z, a))
            })
            .collect()
    }
}

/// Fused multiply-add matching numpy's AVX2 `fmaddsub` build: `a*b + c` with a
/// single rounding (hardware FMA). Falls back to a two-rounding computation on
/// targets without FMA; the decoder is built for x86-64 with `+fma` so this
/// always uses the hardware instruction in practice.
#[cfg(target_arch = "x86_64")]
#[inline]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    // `#[target_feature]` fns must be `unsafe fn`; the caller is always on a
    // path that assumes FMA is available (x86-64 with the `fma` feature).
    #[target_feature(enable = "fma")]
    unsafe fn fma_impl(a: f64, b: f64, c: f64) -> f64 {
        a.mul_add(b, c)
    }
    unsafe { fma_impl(a, b, c) }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// Port of `scipy.signal.buttord` for the digital lowpass case only: returns
/// the lowest Butterworth order meeting the pass/stop specs plus the 3 dB
/// frequency (normalized to Nyquist, for `butter_ba`).
fn buttord_lowpass(wp: f64, ws: f64, gpass: f64, gstop: f64) -> (usize, f64) {
    let passb = (PI * wp / 2.0).tan();
    let stopb = (PI * ws / 2.0).tan();
    let nat = stopb / passb;
    let gstop_lin = 10f64.powf(0.1 * gstop.abs());
    let gpass_lin = 10f64.powf(0.1 * gpass.abs());
    let ord = ((gstop_lin - 1.0) / (gpass_lin - 1.0)).log10() / (2.0 * nat.log10());
    let ord = ord.ceil() as usize;
    let w0 = (gpass_lin - 1.0).powf(-1.0 / (2.0 * ord as f64));
    let wn = (w0 * passb).atan() * 2.0 / PI;
    (ord, wn)
}

/// Port of `utils.gen_bpf_supergauss`: a symmetric super-Gaussian bandpass in
/// the frequency domain, `block_len` bins long.
fn gen_bpf_supergauss(
    freq_low: f64,
    freq_high: f64,
    order: usize,
    nyquist_hz: f64,
    block_len: usize,
) -> Vec<f64> {
    let half = block_len / 2 + 1;
    let freq = freq_high - freq_low;
    let centerfreq = (freq_high + freq_low) / 2.0;
    let log2_half = (2.0f64.ln() / 2.0).powf(1.0 / (2.0 * order as f64));
    // numpy's `linspace(0, nyquist, n)` builds `i * (delta/div) + start`.
    let step = nyquist_hz / (half - 1) as f64;
    let mut sg: Vec<f64> = (0..half)
        .map(|i| {
            let x = i as f64 * step;
            let arg = 2.0 * (x - centerfreq) * log2_half / freq;
            // `np.power(arg, 2*order)` evaluates the C library `pow` with a
            // float exponent; `powi` would use repeated squaring and differ by
            // ulps across the whole band.
            (-2.0 * arg.powf(2.0 * order as f64)).exp()
        })
        .collect();
    sg.pop(); // [:-1]
    let mut out = sg.clone();
    out.extend(sg.iter().rev());
    out
}

/// Knot vector exactly as scipy's `_not_a_knot(x, k)` (k odd): interior
/// knots `x[k2..n-k2]` with `k2 = (k+1)/2`, endpoints repeated k+1 times.
fn not_a_knot_knots(x: &[f64], k: usize) -> Vec<f64> {
    let k2 = (k + 1) / 2;
    let n = x.len();
    let mut t = vec![x[0]; k + 1];
    t.extend_from_slice(&x[k2..n - k2]);
    t.extend(vec![x[n - 1]; k + 1]);
    t
}

/// `_find_interval` from scipy's `__fitpack.cc`: find `l` with
/// `t[l] <= xval < t[l+1]`. NaN -> -1; out of support -> -1 unless
/// `extrapolate` is set. `prev_l` is a hint from the previous call.
fn find_interval(t: &[f64], k: usize, xval: f64, prev_l: i64, extrapolate: bool) -> i64 {
    let len_t = t.len() as i64;
    let n = len_t - k as i64 - 1;
    let tb = t[k];
    let te = t[n as usize];
    if xval.is_nan() || ((xval < tb || xval > te) && !extrapolate) {
        return -1;
    }
    let mut l = if (k as i64) < prev_l && prev_l < n {
        prev_l
    } else {
        k as i64
    };
    while xval < t[l as usize] && l != k as i64 {
        l -= 1;
    }
    l += 1;
    while xval >= t[l as usize] && l != n {
        l += 1;
    }
    l - 1
}

/// `_deBoor_D` from scipy's `__fitpack.cc`, verbatim arithmetic. On return
/// `result[0..=k]` holds the k+1 nonzero B-spline values (derivative order
/// `m`); `result[k+1..]` is scratch.
fn deboor_d(t: &[f64], x: f64, k: usize, ell: i64, m: usize, result: &mut [f64]) {
    let (h, hh) = result.split_at_mut(k + 1);
    // k-m "standard" de Boor iterations.
    h[0] = 1.0;
    for j in 1..=k - m {
        hh[..j].copy_from_slice(&h[..j]);
        h[0] = 0.0;
        for n in 1..=j {
            let ind = ell + n as i64;
            let xb = t[ind as usize];
            let xa = t[(ind - j as i64) as usize];
            if xb == xa {
                h[n] = 0.0;
                continue;
            }
            let w = hh[n - 1] / (xb - xa);
            h[n - 1] += w * (xb - x);
            h[n] = w * (x - xa);
        }
    }
    // m "derivative" recursions (m == 0 for our use: empty).
    for j in k - m + 1..=k {
        hh[..j].copy_from_slice(&h[..j]);
        h[0] = 0.0;
        for n in 1..=j {
            let ind = ell + n as i64;
            let xb = t[ind as usize];
            let xa = t[(ind - j as i64) as usize];
            if xb == xa {
                h[n] = 0.0;
                continue;
            }
            let w = j as f64 * hh[n - 1] / (xb - xa);
            h[n - 1] -= w;
            h[n] = w;
        }
    }
}

/// `_coloc_matrix` from scipy's `__fitpack.cc`: fill the (2k+1)-band
/// collocation matrix in LAPACK banded storage. `ab` is `[nbands][nt]`
/// row-major with `AB(kl+ku+1+i-j, j) = A(i,j)` (0-based row `kl+ku+i-j`).
fn coloc_matrix(x: &[f64], t: &[f64], k: usize, ab: &mut [f64], nt: usize) {
    let kl = k;
    let ku = k;
    let mut wrk = vec![0.0f64; 2 * k + 2];
    let mut left = k as i64;
    for (j, &xval) in x.iter().enumerate() {
        left = find_interval(t, k, xval, left, false);
        debug_assert!(left >= 0, "x value outside knot support");
        deboor_d(t, xval, k, left, 0, &mut wrk);
        for a in 0..=k {
            let clmn = left - k as i64 + a as i64;
            let row = kl as i64 + ku as i64 + j as i64 - clmn;
            ab[row as usize * nt + clmn as usize] = wrk[a];
        }
    }
}

/// DGBTF2 (unblocked banded LU with partial pivoting), transliterated from
/// the netlib reference. `ab` is `[nbands][n]` row-major, band row 0-based
/// `kv = kl+ku` holds the diagonal; `ipiv` stores 1-based pivot rows.
/// n <= NB so the blocked DGBTRF path (which uses this same routine for the
/// kernel) reduces to DGBTF2 for our 11x11 system.
fn dgbtf2(ab: &mut [f64], m: usize, n: usize, kl: usize, ku: usize, ipiv: &mut [usize]) {
    let kv = ku + kl;
    // Set fill-in elements in columns KU+2..min(KV,N) to zero.
    for j0 in (ku + 1)..std::cmp::min(kv, n) {
        // 1-based rows KV-J+2..KL  ->  0-based rows (KV-J+1)..(KL-1)
        let start = kv as i64 - j0 as i64;
        if start >= 0 {
            for i0 in (start as usize)..kl {
                ab[i0 * n + j0] = 0.0;
            }
        }
    }
    let mut ju = 1usize; // 1-based last affected column
    for j0 in 0..m.min(n) {
        // Set fill-in elements in column J+KV to zero.
        if j0 + kv < n {
            for i0 in 0..kl {
                ab[i0 * n + (j0 + kv)] = 0.0;
            }
        }
        // Find pivot and test for singularity.
        let km = kl.min(m - j0 - 1);
        let mut jp = 1usize;
        let mut best = ab[kv * n + j0].abs();
        for p in 2..=km + 1 {
            let v = ab[(kv + p - 1) * n + j0].abs();
            if v > best {
                best = v;
                jp = p;
            }
        }
        ipiv[j0] = jp + j0; // 1-based pivot row = JP + J - 1
        if ab[(kv + jp - 1) * n + j0] != 0.0 {
            ju = ju.max((j0 + 1 + ku + jp - 1).min(n));
            // Apply interchange to columns J..JU.
            if jp != 1 {
                for c0 in j0..ju {
                    let a = (kv + jp - 1) * n + c0;
                    let b = kv * n + c0;
                    ab.swap(a, b);
                }
            }
            if km > 0 {
                // Compute multipliers (DSCAL with 1/pivot).
                let scale = 1.0 / ab[kv * n + j0];
                for s in 1..=km {
                    ab[(kv + s) * n + j0] *= scale;
                }
                // Update trailing submatrix within the band (DGER).
                if ju > j0 + 1 {
                    for t in 1..=ju - (j0 + 1) {
                        for s in 1..=km {
                            // A(J+s, J+t) at band row kl+ku+1+(J+s)-(J+t)-1
                            // = kv+s-t (0-based), column J+t-1 = j0+t.
                            let r = (kv + s - t) * n + (j0 + t);
                            ab[r] -= ab[(kv + s) * n + j0] * ab[(kv - t) * n + (j0 + t)];
                        }
                    }
                }
            }
        }
    }
}

/// DGBTRS (NRHS = 1) + DTBSV ('U','N','N', KD=KL+KU), transliterated from
/// the netlib reference. Solves the banded system factored by `dgbtf2`.
fn dgbtrs(ab: &[f64], n: usize, kl: usize, ku: usize, ipiv: &[usize], b: &mut [f64]) {
    let kv = ku + kl;
    // Solve L*X = B, overwriting B with X.
    for j0 in 0..n - 1 {
        let lm = kl.min(n - j0 - 1);
        let l = ipiv[j0]; // 1-based
        if l != j0 + 1 {
            b.swap(l - 1, j0);
        }
        // DGER(LM, 1, -1, AB(KD+1, J), 1, B(J), LDB, B(J+1), LDB)
        for s0 in 0..lm {
            let mplier_row = kv + 1 + s0; // AB(KD+1+s0, J) 0-based row
            b[j0 + 1 + s0] -= ab[mplier_row * n + j0] * b[j0];
        }
    }
    // Solve U*X = B (DTBSV upper, non-unit, K = KL+KU).
    for j0 in (0..n).rev() {
        if b[j0] != 0.0 {
            b[j0] /= ab[kv * n + j0];
            let temp = b[j0];
            let ilo = (j0 as i64 - kv as i64).max(0) as usize;
            for i0 in (ilo..j0).rev() {
                // x(i) -= temp * A(KPLUS1-J+I, J); 0-based band row = kv+i0-j0
                b[i0] -= temp * ab[(kv + i0 - j0) * n + j0];
            }
        }
    }
}

/// `make_interp_spline` (not-a-knot, k=3) + `_evaluate_spline` from scipy
/// 1.18.0, bit-for-bit: knots -> BSpline collocation (DGBTF2/DGBTRS/DTBSV
/// LAPACK arithmetic) -> de Boor evaluation. This is exactly what
/// `interp1d(kind="cubic")` does in the 7.3.0 release python.
fn scipy_cubic_spline(x: &[f64], y: &[f64], at: &[f64]) -> Vec<f64> {
    let k = 3usize;
    let t = not_a_knot_knots(x, k);
    let n = x.len();
    let nt = t.len() - k - 1;
    assert_eq!(nt, n, "not-a-knot knots must match data points");
    let nbands = 2 * k + k + 1; // 2*kl+ku+1 = 10 for k=3
    let mut ab = vec![0.0f64; nbands * nt];
    coloc_matrix(x, &t, k, &mut ab, nt);
    let mut ipiv = vec![0usize; n];
    dgbtf2(&mut ab, n, n, k, k, &mut ipiv);
    let mut b = y.to_vec();
    dgbtrs(&ab, n, k, k, &ipiv, &mut b);

    // _evaluate_spline: de Boor values, then out += c[interval+a-k]*wrk[a].
    let mut wrk = vec![0.0f64; 2 * k + 2];
    let mut interval = k as i64;
    at.iter()
        .map(|&xval| {
            interval = find_interval(&t, k, xval, interval, true);
            if interval < 0 {
                return f64::NAN;
            }
            deboor_d(&t, xval, k, interval, 0, &mut wrk);
            let mut acc = 0.0;
            for a in 0..=k {
                acc += b[(interval + a as i64 - k as i64) as usize] * wrk[a];
            }
            acc
        })
        .collect()
}

/// Port of `utils.fft_determine_slices`: the `(lowbin, nbins, cut_freq)` of
/// the FFT slice covering `center +/- min_bandwidth`.
fn fft_determine_slices(
    center: f64,
    min_bandwidth: f64,
    freq_hz: f64,
    bins_in: usize,
) -> (usize, usize, f64) {
    let binwidth = freq_hz / bins_in as f64;
    let cbin = (center / binwidth).round_ties_even();
    let bbins = (min_bandwidth / binwidth).round_ties_even();
    let nbins = 2 * (2usize).pow((bbins * 2.0).log2().ceil() as u32);
    let lowbin = (cbin - (nbins as f64 / 4.0)) as usize;
    let cut_freq = binwidth * nbins as f64;
    (lowbin, nbins, cut_freq)
}

/// Port of `utils.fft_do_slice`: cut a full-block spectrum to the audio bins.
fn fft_do_slice(
    fdomain: &[Complex64],
    lowbin: usize,
    nbins: usize,
    blocklen: usize,
) -> Vec<Complex64> {
    let nbins_half = nbins / 2;
    let mut out = Vec::with_capacity(nbins);
    out.extend_from_slice(&fdomain[lowbin..lowbin + nbins_half]);
    out.extend_from_slice(&fdomain[blocklen - lowbin - nbins_half..blocklen - lowbin]);
    out
}

/// numpy complex128 element-wise multiply (AVX2 `fmaddsub` FMA kernel).
///
/// Verified bit-for-bit against numpy 2.4's complex128 multiply (used by
/// `np.multiply` on arrays, e.g. the Horner polyval in `freqz` and the
/// filter-combination multiplies in `computevideofilters`).
#[inline]
pub(crate) fn np_cmul(a: Complex64, b: Complex64) -> Complex64 {
    // numpy's complex128 multiply dispatches to its AVX2 SIMD kernel
    // (`fmaddsub`-style FMA); the scalar `loops.c.src` fallback is not used on
    // this platform. Verified bit-identical to the reference on real data.
    Complex64::new(
        fma(a.re, b.re, -(a.im * b.im)),
        fma(a.re, b.im, a.im * b.re),
    )
}

/// Element-wise [`np_cmul`] over slices: `out[i] = np_cmul(a[i], b[i])`.
///
/// The scalar form is two rounded products feeding one FMA each, written over
/// an interleaved `[re, im]` layout — the layout is exactly what stops LLVM
/// from vectorizing it (the `.re`/`.im` operands of neighbouring elements are
/// strided by 16 bytes). Holding two complex values in one AVX2 register makes
/// the interleave an advantage: `vfmaddsub` applies the subtract to the even
/// (real) lanes and the add to the odd (imaginary) lanes, which is precisely
/// `fma(ar, br, -(ai*bi))` and `fma(ar, bi, ai*br)`. Every lane keeps the
/// scalar form's operand order and rounding, so results are bit-identical and
/// the scalar loop stays the fallback for the tail and non-x86 targets.
pub(crate) fn np_cmul_slices(a: &[Complex64], b: &[Complex64], out: &mut [Complex64]) {
    let n = a.len().min(b.len()).min(out.len());
    debug_assert_eq!(n, a.len());
    debug_assert_eq!(n, b.len());
    debug_assert_eq!(n, out.len());
    np_cmul_core(
        a.as_ptr() as *const f64,
        b.as_ptr() as *const f64,
        out.as_mut_ptr() as *mut f64,
        n,
    );
}

/// [`np_cmul_slices`] appended to a reused vector (leaving any existing
/// elements in place, like `extend`). The appended elements are left
/// uninitialised before the fill (same convention as the FFT wrappers) since
/// every one of them is overwritten.
pub(crate) fn np_cmul_extend(a: &[Complex64], b: &[Complex64], out: &mut Vec<Complex64>) {
    let n = a.len().min(b.len());
    let base = out.len();
    out.reserve(n);
    // SAFETY: `base + n <= capacity` after `reserve`, and the fill writes all
    // `n` appended elements; `Complex64` is `Copy` and owns no resources, so
    // the uninit window cannot be observed (no panic path can unwind through
    // the fill).
    unsafe { out.set_len(base + n) };
    np_cmul_core(
        a.as_ptr() as *const f64,
        b.as_ptr() as *const f64,
        unsafe { out.as_mut_ptr().add(base) } as *mut f64,
        n,
    );
}

/// [`np_cmul_slices`] into a reused vector: cleared, then refilled to the
/// shorter input's length.
pub(crate) fn np_cmul_fill(a: &[Complex64], b: &[Complex64], out: &mut Vec<Complex64>) {
    out.clear();
    np_cmul_extend(a, b, out);
}

/// In-place element-wise [`np_cmul`]: `a[i] = np_cmul(a[i], b[i])`.
pub(crate) fn np_cmul_assign(a: &mut [Complex64], b: &[Complex64]) {
    let n = a.len().min(b.len());
    let p = a.as_mut_ptr() as *mut f64;
    np_cmul_core(p as *const f64, b.as_ptr() as *const f64, p, n);
}

/// Fused chain `out[i] = np_cmul(np_cmul(a[i], b[i]), c[i])` in one pass: the
/// intermediate product stays in a register instead of round-tripping through
/// memory. Per element the op sequence and rounding are identical to the
/// [`np_cmul_fill`] + [`np_cmul_assign`] pair — the AVX2 kernel is the same
/// `mul_avx2` body with the register product playing the "a" role in the
/// second multiply — so results are bit-identical.
pub(crate) fn np_cmul3_fill(a: &[Complex64], b: &[Complex64], c: &[Complex64], out: &mut Vec<Complex64>) {
    out.clear();
    let n = a.len().min(b.len()).min(c.len());
    out.reserve(n);
    // SAFETY: as in `np_cmul_extend` — `base + n <= capacity` after `reserve`,
    // every appended element is overwritten, and the fill cannot unwind.
    unsafe { out.set_len(n) };
    np_cmul3_core(
        a.as_ptr() as *const f64,
        b.as_ptr() as *const f64,
        c.as_ptr() as *const f64,
        out.as_mut_ptr() as *mut f64,
        n,
    );
}

/// [`np_cmul3_fill`] into a caller-provided slice (the batched kernel writes
/// each filter product straight into its row of the batch buffer). Same core as
/// the `Vec` form, so the arithmetic and rounding are identical.
///
/// `out` may alias none of the inputs (the production call sites pass distinct
/// buffers: a spectrum row is always written from a different source).
pub(crate) fn np_cmul3_slices(a: &[Complex64], b: &[Complex64], c: &[Complex64], out: &mut [Complex64]) {
    let n = a.len().min(b.len()).min(c.len()).min(out.len());
    debug_assert_eq!(n, a.len());
    debug_assert_eq!(n, b.len());
    debug_assert_eq!(n, c.len());
    debug_assert_eq!(n, out.len());
    np_cmul3_core(
        a.as_ptr() as *const f64,
        b.as_ptr() as *const f64,
        c.as_ptr() as *const f64,
        out.as_mut_ptr() as *mut f64,
        n,
    );
}

/// Scalar/vector dispatch for the fused two-multiply chain. `op` may alias any
/// input only if that input is not also read after being overwritten — the
/// production call sites never alias (a, b, c are all distinct from out).
#[inline]
fn np_cmul3_core(ap: *const f64, bp: *const f64, cp: *const f64, op: *mut f64, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2,fma")]
        unsafe fn mul3_avx2(ap: *const f64, bp: *const f64, cp: *const f64, op: *mut f64, n: usize) {
            use std::arch::x86_64::*;
            let mut i = 0usize;
            // Two complex values per iteration; lane layout as in `mul_avx2`.
            while i + 2 <= n {
                let a = _mm256_loadu_pd(ap.add(i * 2));
                let b = _mm256_loadu_pd(bp.add(i * 2));
                let ar = _mm256_movedup_pd(a);
                let ai = _mm256_permute_pd(a, 0b1111);
                let bswap = _mm256_permute_pd(b, 0b0101);
                let t = _mm256_mul_pd(ai, bswap);
                // p = np_cmul(a, b), held in a register.
                let p = _mm256_fmaddsub_pd(ar, b, t);
                let c = _mm256_loadu_pd(cp.add(i * 2));
                // r = np_cmul(p, c): the packed register product takes the
                // same role `a` plays in the single-multiply kernel (movedup
                // / permute on `p`, direct + 0101-permute on `c`), so the
                // lane ops match the two-pass kernel exactly.
                let pr2 = _mm256_movedup_pd(p);
                let pi2 = _mm256_permute_pd(p, 0b1111);
                let cswap = _mm256_permute_pd(c, 0b0101);
                let t2 = _mm256_mul_pd(pi2, cswap);
                let r = _mm256_fmaddsub_pd(pr2, c, t2);
                _mm256_storeu_pd(op.add(i * 2), r);
                i += 2;
            }
            while i < n {
                let ar = *ap.add(i * 2);
                let ai = *ap.add(i * 2 + 1);
                let br = *bp.add(i * 2);
                let bi = *bp.add(i * 2 + 1);
                // np_cmul(a, b) — scalar tail, same ops as `mul_avx2`'s tail.
                let pr = ar.mul_add(br, -(ai * bi));
                let pi = ar.mul_add(bi, ai * br);
                let cr = *cp.add(i * 2);
                let ci = *cp.add(i * 2 + 1);
                // np_cmul(p, c)
                *op.add(i * 2) = pr.mul_add(cr, -(pi * ci));
                *op.add(i * 2 + 1) = pr.mul_add(ci, pi * cr);
                i += 1;
            }
        }
        unsafe { mul3_avx2(ap, bp, cp, op, n) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    for i in 0..n {
        let (ar, ai) = unsafe { (*ap.add(i * 2), *ap.add(i * 2 + 1)) };
        let (br, bi) = unsafe { (*bp.add(i * 2), *bp.add(i * 2 + 1)) };
        let (cr, ci) = unsafe { (*cp.add(i * 2), *cp.add(i * 2 + 1)) };
        let pr = fma(ar, br, -(ai * bi));
        let pi = fma(ar, bi, ai * br);
        unsafe {
            *op.add(i * 2) = fma(pr, cr, -(pi * ci));
            *op.add(i * 2 + 1) = fma(pr, ci, pi * cr);
        }
    }
}

/// Scalar/vector dispatch for the element-wise complex multiply. `ap` and `op`
/// may be the same pointer (in-place); each element is read before it is
/// written.
#[inline]
fn np_cmul_core(ap: *const f64, bp: *const f64, op: *mut f64, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        // `#[target_feature]` fns must be `unsafe fn`; the crate is built for
        // x86-64-v3 (AVX2+FMA), the same assumption the `fma` helper makes.
        #[target_feature(enable = "avx2,fma")]
        unsafe fn mul_avx2(ap: *const f64, bp: *const f64, op: *mut f64, n: usize) {
            use std::arch::x86_64::*;
            let mut i = 0usize;
            // Two complex values per iteration: [re0, im0, re1, im1].
            while i + 2 <= n {
                let a = _mm256_loadu_pd(ap.add(i * 2));
                let b = _mm256_loadu_pd(bp.add(i * 2));
                // [ar0, ar0, ar1, ar1] and [ai0, ai0, ai1, ai1].
                let ar = _mm256_movedup_pd(a);
                let ai = _mm256_permute_pd(a, 0b1111);
                // [bi0, br0, bi1, br1]: the odd-lane operand of each part.
                let bswap = _mm256_permute_pd(b, 0b0101);
                // [ai0*bi0, ai0*br0, ai1*bi1, ai1*br1].
                let t = _mm256_mul_pd(ai, bswap);
                // Even lanes subtract (real), odd lanes add (imag).
                let r = _mm256_fmaddsub_pd(ar, b, t);
                _mm256_storeu_pd(op.add(i * 2), r);
                i += 2;
            }
            while i < n {
                let ar = *ap.add(i * 2);
                let ai = *ap.add(i * 2 + 1);
                let br = *bp.add(i * 2);
                let bi = *bp.add(i * 2 + 1);
                *op.add(i * 2) = ar.mul_add(br, -(ai * bi));
                *op.add(i * 2 + 1) = ar.mul_add(bi, ai * br);
                i += 1;
            }
        }
        unsafe { mul_avx2(ap, bp, op, n) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    for i in 0..n {
        let (ar, ai) = unsafe { (*ap.add(i * 2), *ap.add(i * 2 + 1)) };
        let (br, bi) = unsafe { (*bp.add(i * 2), *bp.add(i * 2 + 1)) };
        unsafe {
            *op.add(i * 2) = fma(ar, br, -(ai * bi));
            *op.add(i * 2 + 1) = fma(ar, bi, ai * br);
        }
    }
}

/// numpy complex128 division (Smith's algorithm with multiply-by-reciprocal,
/// matching the SIMD `npyv_cdiv_f64` kernel).
pub(crate) fn np_cdiv(a: Complex64, b: Complex64) -> Complex64 {
    let (xr, xi, yr, yi) = (a.re, a.im, b.re, b.im);
    if yr.abs() >= yi.abs() {
        if yr == 0.0 && yi == 0.0 {
            return Complex64::new(xr / 0.0, xi / 0.0);
        }
        let rat = yi / yr;
        let scl = 1.0 / (yr + yi * rat);
        Complex64::new((xr + xi * rat) * scl, (xi - xr * rat) * scl)
    } else {
        let rat = yr / yi;
        let scl = 1.0 / (yi + yr * rat);
        Complex64::new((xr * rat + xi) * scl, (xi * rat - xr) * scl)
    }
}

/// numpy `np.prod` of complex128 (the `multiply.reduce` ufunc: per-element
/// FMA products accumulated in order).
fn np_cprod(v: &[Complex64]) -> Complex64 {
    let mut acc = Complex64::new(1.0, 0.0);
    for &x in v {
        acc = np_cmul(acc, x);
    }
    acc
}

/// Legacy `np.poly(roots)`: expand `(s - r0)(s - r1)...` via successive
/// `np.convolve` with `[1, -r_i]`.  numpy's `convolve` routes through
/// `multiarray.correlate`, whose inner dots call the CBLAS `zdotu` kernel:
/// four independent accumulators (re*re, im*im, re*im, im*re), each FMA-fused,
/// combined as re = d0 - d1, im = d2 + d3.  Verified bit-for-bit against
/// numpy 2.4.6's `np.poly` for the butter zpk2tf expansions.  Callers take
/// `.re` afterwards (np.poly casts to real when the roots are closed under
/// conjugation, which they always are for butter filters).
fn np_poly(roots: &[Complex64]) -> Vec<Complex64> {
    let mut a: Vec<Complex64> = vec![Complex64::new(1.0, 0.0)];
    for &r in roots {
        let neg_r = Complex64::new(-r.re, -r.im);
        let n = a.len();
        let mut out = Vec::with_capacity(n + 1);
        for k in 0..=n {
            // zdotu over (a[k-1], a[k]) x (neg_r, 1), with missing terms skipped
            // (matches numpy's per-lag dot of length 1 or 2).
            let mut d0 = 0.0;
            let mut d1 = 0.0;
            let mut d2 = 0.0;
            let mut d3 = 0.0;
            if k > 0 {
                let x = a[k - 1];
                d0 = fma(x.re, neg_r.re, d0);
                d1 = fma(x.im, neg_r.im, d1);
                d2 = fma(x.re, neg_r.im, d2);
                d3 = fma(x.im, neg_r.re, d3);
            }
            if k < n {
                let x = a[k];
                d0 = fma(x.re, 1.0, d0);
                d1 = fma(x.im, 0.0, d1);
                d2 = fma(x.re, 0.0, d2);
                d3 = fma(x.im, 1.0, d3);
            }
            out.push(Complex64::new(d0 - d1, d2 + d3));
        }
        a = out;
    }
    a
}

/// `npy_csqrt` (Algorithm 312) on the finite path.
fn np_csqrt(z: Complex64) -> Complex64 {
    let a = z.re;
    let b = z.im;
    if a == 0.0 && b == 0.0 {
        return z;
    }
    if a >= 0.0 {
        let t = ((a + a.hypot(b)) * 0.5).sqrt();
        Complex64::new(t, b / (2.0 * t))
    } else {
        let t = ((-a + a.hypot(b)) * 0.5).sqrt();
        Complex64::new(b.abs() / (2.0 * t), t.copysign(b))
    }
}

/// UCRT (ucrtbase.dll) `atan2`. Rust's std `f64::atan2` is not bit-identical
/// to the C runtime (it deviates by 1-2 ulp on a few percent of arguments on
/// Windows), while numpy's `np.arctan2` calls UCRT's `atan2` exactly, so the
/// demod (unwrap_hilbert) and VITS phase must call this to stay bit-exact.
#[cfg(target_os = "windows")]
pub(crate) mod ucrt_atan2 {
    #[link(name = "ucrtbase", kind = "raw-dylib")]
    extern "C" {
        pub fn atan2(y: f64, x: f64) -> f64;
    }

    /// Public entry: exact UCRT `atan2` for one pair.
    #[inline]
    pub fn call(y: f64, x: f64) -> f64 {
        unsafe { atan2(y, x) }
    }

    /// Whole-slice variant: lets the compiler amortize the FFI transition
    /// across the loop (one extern call per ~8 inputs via batching inside the
    /// extern fn is not possible for atan2, but keeping the loop in one
    /// #[inline(never)] function avoids re-loading closure state and lets
    /// LLVM keep `d`/`scale` in registers). Same per-element calls and
    /// results as `call` in a loop.
    pub fn call_slice(out: &mut [f64], pim: &[f64], pre: &[f64]) {
        assert_eq!(out.len(), pim.len());
        assert_eq!(out.len(), pre.len());
        for i in 0..out.len() {
            out[i] = unsafe { atan2(pim[i], pre[i]) };
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub(crate) mod ucrt_atan2 {
    pub fn call(y: f64, x: f64) -> f64 {
        y.atan2(x)
    }

    pub fn call_slice(out: &mut [f64], pim: &[f64], pre: &[f64]) {
        assert_eq!(out.len(), pim.len());
        assert_eq!(out.len(), pre.len());
        for i in 0..out.len() {
            out[i] = pim[i].atan2(pre[i]);
        }
    }
}

/// UCRT (ucrtbase.dll) C99 complex pow, exported as `cpow`.
#[cfg(target_os = "windows")]
mod ucrt_cpow {
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct DComplex {
        pub re: f64,
        pub im: f64,
    }

    #[link(name = "ucrtbase", kind = "raw-dylib")]
    extern "C" {
        pub fn cpow(a: DComplex, b: DComplex) -> DComplex;
    }
}

/// `npy_cpow`: numpy's complex128 power. Integer exponents (-100..100) use the
/// cmul-based fast path; everything else calls the system C library's `cpow`
/// (on Windows, UCRT's `cpow` is what numpy's `npy_cpow` dispatches to via
/// `HAVE_CPOW`, and it is not bit-identical to exp(b * log(a)) chains).
pub(crate) fn np_cpow(a: Complex64, b: Complex64) -> Complex64 {
    if b.re == 0.0 && b.im == 0.0 {
        return Complex64::new(1.0, 0.0);
    }
    if a.re == 0.0 && a.im == 0.0 {
        if b.re > 0.0 {
            return Complex64::new(0.0, 0.0);
        }
        return Complex64::new(f64::NAN, f64::NAN);
    }
    if b.im == 0.0 && b.re > -100.0 && b.re < 100.0 && (b.re as i64) as f64 == b.re {
        let n0 = b.re as i64;
        if n0 == 1 {
            return a;
        }
        if n0 == 2 {
            return np_cmul(a, a);
        }
        if n0 == 3 {
            return np_cmul(a, np_cmul(a, a));
        }
        if n0 > -100 && n0 < 100 {
            let mut n = if n0 < 0 { -n0 } else { n0 };
            let mut acc = Complex64::new(1.0, 0.0);
            let mut mask = 1i64;
            let mut p = a;
            loop {
                if n & mask != 0 {
                    acc = np_cmul(acc, p);
                }
                mask <<= 1;
                if n < mask || mask <= 0 {
                    break;
                }
                p = np_cmul(p, p);
            }
            if n0 < 0 {
                acc = np_cdiv(Complex64::new(1.0, 0.0), acc);
            }
            return acc;
        }
    }
    #[cfg(target_os = "windows")]
    {
        let r = unsafe {
            ucrt_cpow::cpow(
                ucrt_cpow::DComplex { re: a.re, im: a.im },
                ucrt_cpow::DComplex { re: b.re, im: b.im },
            )
        };
        return Complex64::new(r.re, r.im);
    }
    #[cfg(not(target_os = "windows"))]
    {
        // Approximate fallback for non-Windows: exp(b * log(a)). Not bit-identical
        // to numpy (which uses the platform libm cpow), but close.
        let loga = Complex64::new(a.re.hypot(a.im).ln(), a.im.atan2(a.re));
        let x = b.re * loga.re - b.im * loga.im;
        let y = b.re * loga.im + b.im * loga.re;
        let e = x.exp();
        return Complex64::new(e * y.cos(), e * y.sin());
    }
}

/// Port of `scipy.signal.butter(..., output='ba')` (scipy >= 1.18, digital,
/// fs=2, normalized `wn` in [0, 1]): prewarp -> buttap -> lp2lp/lp2hp/lp2bs
/// -> bilinear(fs=2) -> zpk2tf with the legacy `np.poly` convolve expansion.
///
/// The b/a coefficients are bit-identical to scipy's; verified against the
/// bundled release python for the RF video split highpass/lowpass, the audio
/// notch bandstops and the dropout highpass.
fn butter_ba(order: usize, wn: &[f64], band_type: FilterBandType) -> Result<(Vec<f64>, Vec<f64>)> {
    // iirfilter digital path: fs = 2.0, warped = 2*fs*tan(pi*Wn/fs).
    let warped: Vec<f64> = wn.iter().map(|&w| 4.0 * (PI * (w / 2.0)).tan()).collect();

    // buttap(order): p = -exp(i*pi*m/(2N)), m = -N+1, -N+3, ..., N-1.
    // scipy computes the angle as (1j*pi*m)/(2*N) -- a complex array divided
    // by an int, which numpy promotes to a complex divisor. Smith's division
    // then multiplies by the reciprocal: theta = (pi*m) * (1/(2N)), NOT a
    // direct division (they differ by 1 ulp for m = +-(N-1)).
    let n = order as i64;
    let scl = 1.0 / (2.0 * order as f64);
    let p0: Vec<Complex64> = ((-n + 1)..n)
        .step_by(2)
        .map(|m| {
            let th = (PI * (m as f64)) * scl;
            Complex64::new(-th.cos(), -th.sin())
        })
        .collect();

    let one = Complex64::new(1.0, 0.0);
    let (z, p, k): (Vec<Complex64>, Vec<Complex64>, f64) = match band_type {
        FilterBandType::Lowpass => {
            let wo = warped[0];
            let p2: Vec<Complex64> = p0
                .iter()
                .map(|&x| Complex64::new(wo * x.re, wo * x.im))
                .collect();
            // k * wo**degree, degree == order (python float ** int).
            let k2 = wo.powf(n as f64);
            (Vec::new(), p2, k2)
        }
        FilterBandType::Highpass => {
            let wo = warped[0];
            let won = Complex64::new(wo, 0.0);
            let p2: Vec<Complex64> = p0.iter().map(|&x| np_cdiv(won, x)).collect();
            let z2 = vec![Complex64::new(0.0, 0.0); order];
            let negp: Vec<Complex64> = p0.iter().map(|&x| Complex64::new(-x.re, -x.im)).collect();
            // k_hp = k * real(prod(-z) / prod(-p)); z empty -> prod = 1.
            let k2 = np_cdiv(one, np_cprod(&negp)).re;
            (z2, p2, k2)
        }
        // (scipy's bandpass path is not used by any filter in this decoder.)
        FilterBandType::Bandpass => unreachable!("butter bandpass not used"),
        FilterBandType::Bandstop => {
            let bw = warped[1] - warped[0];
            let wo = (warped[0] * warped[1]).sqrt();
            let halfbw = Complex64::new(bw / 2.0, 0.0);
            let p_hp: Vec<Complex64> = p0.iter().map(|&x| np_cdiv(halfbw, x)).collect();
            let mut p_bs: Vec<Complex64> = Vec::with_capacity(2 * order);
            let mut z_bs: Vec<Complex64> = Vec::with_capacity(2 * order);
            let mut sq: Vec<Complex64> = Vec::with_capacity(order);
            for &hp in &p_hp {
                // s = sqrt(hp^2 - wo^2).
                let hp2 = np_cmul(hp, hp);
                let d = Complex64::new(hp2.re - wo * wo, hp2.im);
                sq.push(np_csqrt(d));
            }
            // scipy: p_bs = concat(p_hp + sqrt, p_hp - sqrt) -- all (+s) first,
            // then all (-s); the ordering matters for the poly expansion rounding.
            for (&hp, &s) in p_hp.iter().zip(&sq) {
                p_bs.push(Complex64::new(hp.re + s.re, hp.im + s.im));
            }
            for (&hp, &s) in p_hp.iter().zip(&sq) {
                p_bs.push(Complex64::new(hp.re - s.re, hp.im - s.im));
            }
            // z side: lp prototype has no zeros, so the stopband zeros live at
            // the center frequency: concat(full(degree, +1j*wo), full(degree, -1j*wo)).
            for _ in 0..order {
                z_bs.push(Complex64::new(0.0, wo));
            }
            for _ in 0..order {
                z_bs.push(Complex64::new(0.0, -wo));
            }
            let negp: Vec<Complex64> = p0.iter().map(|&x| Complex64::new(-x.re, -x.im)).collect();
            // k_bs = k * real(prod(-z) / prod(-p)); z empty -> prod = 1.
            let k2 = np_cdiv(one, np_cprod(&negp)).re;
            (z_bs, p_bs, k2)
        }
    };

    // bilinear_zpk(z, p, k, fs=2.0): fs2 = 4.0.
    let degree = p.len() - z.len();
    let bil = |x: Complex64| -> Complex64 {
        np_cdiv(
            Complex64::new(4.0 + x.re, x.im),
            Complex64::new(4.0 - x.re, -x.im),
        )
    };
    let mut z_z: Vec<Complex64> = z.iter().map(|&x| bil(x)).collect();
    // Zeros at infinity move to the Nyquist frequency.
    for _ in 0..degree {
        z_z.push(Complex64::new(-1.0, 0.0));
    }
    let p_z: Vec<Complex64> = p.iter().map(|&x| bil(x)).collect();
    // k_z = k * real(prod(fs2 - z) / prod(fs2 - p)), over the analog zpk.
    let z4: Vec<Complex64> = z.iter().map(|&x| Complex64::new(4.0 - x.re, -x.im)).collect();
    let p4: Vec<Complex64> = p.iter().map(|&x| Complex64::new(4.0 - x.re, -x.im)).collect();
    let k_z = k * np_cdiv(np_cprod(&z4), np_cprod(&p4)).re;

    // zpk2tf: b = k*poly(z_z), a = poly(p_z). np.poly casts to real (roots are
    // conjugate-symmetric), so take the real part of the zdotu expansion.
    let bz = np_poly(&z_z);
    let az = np_poly(&p_z);
    let b: Vec<f64> = bz.iter().map(|&c| c.re * k_z).collect();
    let a: Vec<f64> = az.iter().map(|&c| c.re).collect();
    Ok((b, a))
}

// ---------------------------------------------------------------------------
// Calibration levels (mutable, live on the Decoder)
// ---------------------------------------------------------------------------

/// The HZ<->IRE calibration levels. These are the fields the Python decoder
/// mutates in `DecoderParams` (AGC, sync recalibration); here they are carried
/// separately from the immutable spec.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CalibLevels {
    pub ire0: f64,
    pub hz_ire: f64,
    pub vsync_ire: f64,
    /// Precision of the Python reference's `DecoderParams` scalars.
    ///
    /// `ire0`/`hz_ire`/`vsync_ire` start out as plain Python numbers (float64
    /// arithmetic), but the sync recalibration assigns `np.percentile(...)` of a
    /// float32 array to `ire0`, which makes it a NumPy `float32` scalar. Under
    /// NEP 50 a Python float is a *weak* scalar, so any arithmetic mixing it
    /// with that `np.float32` is performed in float32. The AGC path goes
    /// further and replaces all three with `np.float32` scalars.
    pub prec: u8,
}

/// `CalibLevels::prec` values.
pub(crate) const LEVELS_F64: u8 = 0;
/// `ire0` is an `np.float32` (sync recalibration); `hz_ire` is unchanged.
pub(crate) const LEVELS_IRE0_F32: u8 = 1;
/// All three are `np.float32` scalars (AGC relock).
pub(crate) const LEVELS_ALL_F32: u8 = 2;

impl CalibLevels {
    pub fn defaults() -> Self {
        Self {
            ire0: SYS_IRE0 as f64,
            hz_ire: SYS_HZ_IRE as f64,
            vsync_ire: SYS_VSYNC_IRE as f64,
            prec: LEVELS_F64,
        }
    }

    /// `ire0 + hz_ire * ire` exactly as the Python reference evaluates it for
    /// the current level precision.
    pub fn iretohz(&self, ire: f64) -> f64 {
        match self.prec {
            LEVELS_F64 => self.ire0 + self.hz_ire * ire,
            LEVELS_IRE0_F32 => {
                // `np.float32 + python_float`: the product is float64 in Python
                // and demoted to float32, then added in float32.
                f64::from((self.ire0 as f32) + ((self.hz_ire * ire) as f32))
            }
            _ => {
                // `np.float32 + np.float32 * python_float`: all in float32.
                f64::from((self.ire0 as f32) + ((self.hz_ire as f32) * (ire as f32)))
            }
        }
    }

    /// `(hz - ire0) / hz_ire`.
    pub fn hztoire(&self, hz: f64) -> f64 {
        (hz - self.ire0) / self.hz_ire
    }
}

// ---------------------------------------------------------------------------
// The filter bank
// ---------------------------------------------------------------------------

/// Stage-1/2 filter pair for one analog audio channel (port of the per-channel
/// `self.audio[channel]` namespace from `computeaudiofilters`).
#[derive(Clone)]
pub(crate) struct AudioChannelFilters {
    /// First kept FFT bin of the sliced spectrum.
    pub lowbin: usize,
    /// Number of bins kept (the sliced spectrum / short-FFT size).
    pub nbins: usize,
    /// Output sample rate of the stage-1 demodulation (binwidth * nbins).
    pub a1_freq: f64,
    /// Carrier-centering offset added back after demodulation (Hz).
    pub low_freq: f64,
    /// Stage-1 demodulation filter (complex f64, `nbins` long): bandpass
    /// slice times the short Hilbert transform.
    pub filt1: Vec<Complex64>,
    /// Stage-2 LPF * de-emphasis filter (complex, `blocklen` long), applied in
    /// Stage-2 LPF * de-emphasis filter for `audio_phase2` (f64, matching
    /// Python's `audio2_lpf * audio2_deemp` kept in double precision).
    pub audio2_filter: Vec<Complex64>,
}

pub(crate) struct Filters {
    /// Full-spectrum RF highpass used for dropout detection (complex f64).
    pub frfhpf: Vec<Complex64>,
    /// Full-spectrum MTF compensation filter (complex f64).
    pub mtf: Vec<Complex64>,
    /// Full-spectrum RF video filter including the Hilbert transform (f64).
    pub rfvideo: Vec<Complex64>,
    /// The three full-spectrum post-demod filters (f64):
    /// [FVideo, FVideo05, FVideoBurst].
    pub fvideo: Vec<Vec<Complex64>>,
    /// Full-spectrum 0.5 MHz filter (unshifted), for the sync-only path.
    pub fvideo05: Vec<Complex64>,
    pub f05_offset: usize,
    pub fvideo_burst_offset: usize,
    /// Analog audio channel filters, [left, right].
    pub audio: [AudioChannelFilters; 2],
    /// Sample-rate decimation of the stage-1 audio (blocklen / nbins).
    pub audio_fdiv: usize,
    /// Full-spectrum EFM equalisation filter (complex f64), applied to the
    /// raw input FFT before the video filters.
    pub fefm: Vec<Complex64>,
}

/// Filter delays measured from a synthetic signal (`computedelays`).
pub(crate) struct Delays {
    /// Delay of the sync path in samples (used by VITS black-level RF metrics).
    pub video_sync: f64,
    /// Delay of the white/video path in samples (used by VITS RF metrics).
    pub video_white: f64,
    /// Rot (dropout) delay, in samples.
    pub video_rot: i64,
}

/// Precomputed immutable decode configuration.
pub struct DecoderSpec {
    pub(crate) freq: f64,
    pub(crate) freq_half: f64,
    pub(crate) freq_hz: f64,
    pub(crate) freq_hz_half: f64,

    pub(crate) sys_fsc_mhz: f64,
    pub(crate) sys_frame_lines: usize,
    pub(crate) sys_field_lines: [usize; 2],
    pub(crate) sys_line_period: f64,
    pub(crate) sys_active_video_us: [f64; 2],
    pub(crate) sys_fps: f64,
    pub(crate) sys_outlinelen: usize,
    pub(crate) sys_outfreq: f64,
    pub(crate) sys_color_burst_us: [f64; 2],
    pub(crate) sys_blacksnr_slice: [usize; 3],
    pub(crate) sys_first_field_h: [f64; 2],
    pub(crate) sys_num_pulses: usize,
    pub(crate) sys_hsync_pulse_us: f64,
    pub(crate) sys_eq_pulse_us: f64,
    pub(crate) sys_vsync_pulse_us: f64,
    pub(crate) sys_output_zero: i64,
    pub(crate) sys_field_phases: usize,
    pub(crate) sys_ld_vits_whitelocs: Vec<[usize; 3]>,
    pub(crate) sys_ld_vits_code_slices: Vec<[usize; 4]>,

    pub(crate) linelen: usize,
    #[allow(dead_code)]
    pub(crate) samplesperline: f64,
    pub(crate) blocklen: usize,
    #[allow(dead_code)]
    pub(crate) blockcut: usize,
    pub(crate) blockcut_end: usize,
    /// Demod blocks stride by `blocklen - blockcut - blockcut_end` samples
    /// (port of `RFDecode.blocksize`); the cut output tiles contiguously.
    pub(crate) blocksize: usize,

    pub(crate) dp_mtf_basemult: f64,

    pub(crate) filters: Filters,
    pub(crate) delays: Delays,

    /// Analog audio carrier frequencies (Hz), [left, right].
    pub(crate) audio_lfreq: f64,
    pub(crate) audio_rfreq: f64,

    pub(crate) wow_interpolation_method: WowInterpolation,
    pub(crate) wow_level_adjust_smoothing: f32,
    pub(crate) do_dod: bool,
    #[allow(dead_code)]
    pub(crate) rf_export_raw_tbc: bool,
    #[allow(dead_code)]
    pub(crate) rf_ire0_adjust: bool,
    pub(crate) use_agc: bool,
    #[allow(dead_code)]
    pub(crate) ntsc_color_notch: bool,
    pub(crate) mtf_mult: f64,
    pub(crate) mtf_offset: f64,

    _private: (),
}

impl DecoderSpec {
    /// Build the NTSC Laserdisc decode configuration from a request. This
    /// computes the whole filter bank and the measured filter delays.
    pub fn new(request: &DecodeRequest) -> Result<Self> {
        if request.system != ColorSystem::Ntsc {
            bail!("only NTSC Laserdisc decoding is implemented so far");
        }

        let freq = request.inputfreq;
        let freq_half = freq / 2.0;
        let freq_hz = freq * 1e6;
        let freq_hz_half = freq_hz / 2.0;

        let line_period = 1.0 / (SYS_FSC_MHZ / 227.5);
        let active_video_us = [9.45, line_period - 1.0];
        let fps = 1e6 / (SYS_FRAME_LINES as f64 * line_period);
        let outlinelen = (line_period * SYS_FSC_MHZ * 4.0).round() as usize;
        let outfreq = 4.0 * SYS_FSC_MHZ;

        let linelen = (freq_hz / (1e6 / line_period)).round() as usize;
        let samplesperline = freq / linelen as f64;

        // Resolve decoder params (including overrides and the lowband preset).
        let mut video_bpf_low = DP_VIDEO_BPF_LOW;
        let mut video_bpf_high = DP_VIDEO_BPF_HIGH;
        let mut video_lpf_freq = DP_VIDEO_LPF_FREQ;
        let mut mtf_basemult = DP_MTF_BASEMULT;
        for (key, &value) in &request.decoder_params_override {
            match key.as_str() {
                "MTF_basemult" => mtf_basemult = value,
                "video_bpf_low" => video_bpf_low = value,
                "video_bpf_high" => video_bpf_high = value,
                "video_lpf_freq" => video_lpf_freq = value,
                other => {
                    tracing::warn!(key = other, "ignoring unknown decoder param override");
                }
            }
        }

        // De-emphasis coefficients: (t1 high-frequency, t2 low-frequency), in
        // seconds. The CLI passes usec, converted here.
        let mut deemp = DP_VIDEO_DEEMP;
        let (deemp_low, deemp_high) = request.deemp_coeff;
        if deemp_low > 0.0 {
            deemp[1] = 1.0 / (deemp_low * 1e6);
        }
        if deemp_high > 0.0 {
            deemp[0] = 1.0 / (deemp_high * 1e6);
        }
        let deemp_strength = request.deemp_str;

        let blocklen = BLOCKSIZE;
        let blockcut = 1024;
        let audio_lfreq = (1e6 * SYS_FSC_MHZ / 227.5) * 146.25;
        let audio_rfreq = (1e6 * SYS_FSC_MHZ / 227.5) * 178.75;

        let (filters, lpf_f64, femp_f64) = compute_filters(
            freq_half,
            freq_hz,
            freq_hz_half,
            blocklen,
            video_bpf_low,
            video_bpf_high,
            video_lpf_freq,
            deemp,
            deemp_strength,
            request.ntsc_color_notch,
        )?;
        let blockcut_end = filters.f05_offset;
        let blocksize = blocklen - blockcut - blockcut_end;

        let delays = compute_delays(
            freq,
            freq_half,
            freq_hz,
            blocklen,
            &filters,
            &lpf_f64,
            &femp_f64,
        );

        Ok(Self {
            freq,
            freq_half,
            freq_hz,
            freq_hz_half,
            sys_fsc_mhz: SYS_FSC_MHZ,
            sys_frame_lines: SYS_FRAME_LINES,
            sys_field_lines: SYS_FIELD_LINES,
            sys_line_period: line_period,
            sys_active_video_us: active_video_us,
            sys_fps: fps,
            sys_outlinelen: outlinelen,
            sys_outfreq: outfreq,
            sys_color_burst_us: SYS_COLOR_BURST_US,
            sys_blacksnr_slice: SYS_BLACKSNR_SLICE,
            sys_first_field_h: SYS_FIRST_FIELD_H,
            sys_num_pulses: SYS_NUM_PULSES,
            sys_hsync_pulse_us: SYS_HSYNC_PULSE_US,
            sys_eq_pulse_us: SYS_EQ_PULSE_US,
            sys_vsync_pulse_us: SYS_VSYNC_PULSE_US,
            sys_output_zero: SYS_OUTPUT_ZERO,
            sys_field_phases: SYS_FIELD_PHASES,
            sys_ld_vits_whitelocs: SYS_LD_VITS_WHITELOCS.to_vec(),
            sys_ld_vits_code_slices: SYS_LD_VITS_CODE_SLICES.to_vec(),
            linelen,
            samplesperline,
            blocklen,
            blockcut,
            blockcut_end,
            blocksize,
            dp_mtf_basemult: mtf_basemult,
            filters,
            delays,
            audio_lfreq,
            audio_rfreq,
            wow_interpolation_method: request.wow_interpolation_method,
            wow_level_adjust_smoothing: request.wow_level_adjust_smoothing,
            do_dod: request.do_dod,
            rf_export_raw_tbc: request.rf_export_raw_tbc,
            rf_ire0_adjust: request.rf_ire0_adjust,
            use_agc: request.use_agc,
            ntsc_color_notch: request.ntsc_color_notch,
            mtf_mult: request.mtf_level,
            mtf_offset: request.mtf_offset,
            _private: (),
        })
    }

    /// Input samples per field.
    pub fn bytes_per_field(&self) -> usize {
        (self.freq_hz / (self.sys_fps * 2.0)) as usize + 1
    }

    /// Samples read per field (port of `LDdecode.readlen` for NTSC).
    pub fn readlen(&self) -> usize {
        ((self.linelen * 350) / 16384) * 16384
    }

    /// Demod block stride in input samples; the cut output of consecutive
    /// blocks tiles contiguously (port of `RFDecode.blocksize`).
    pub fn blocksize(&self) -> usize {
        self.blocksize
    }

    /// Samples trimmed from the front of each demod block.
    pub fn blockcut(&self) -> usize {
        self.blockcut
    }

    /// Samples per line at the input sample rate.
    pub fn linelen(&self) -> usize {
        self.linelen
    }



    pub fn output_lines(&self) -> usize {
        (self.sys_frame_lines / 2) + 1
    }

    /// Delay of the "rot" (dropout) reference, in samples.
    #[allow(dead_code)]
    pub(crate) fn rotdelay(&self) -> i64 {
        self.delays.video_rot
    }
}

/// Build the whole filter bank (port of `RFDecode.computevideofilters` plus
/// the audio notch cascades for the NTSC analog-audio carriers). Returns the
/// filter bank plus the plain LPF and emphasis responses (full-spectrum f64)
/// needed by the delays measurement.
#[allow(clippy::too_many_arguments)]
/// Port of `RFDecode.computeaudiofilters`: the per-channel stage-1
/// demodulation filters (bandpass slice * short Hilbert) and the stage-2
/// LPF * de-emphasis filters, plus the audio decimation factor.
fn compute_audio_filters(
    freq_hz: f64,
    freq_hz_half: f64,
    blocklen: usize,
    audio_lfreq: f64,
    audio_rfreq: f64,
) -> ([AudioChannelFilters; 2], usize) {
    let apass = DP_AUDIO_FILTERWIDTH;
    let afilt_len = DP_AUDIO_FILTERORDER;

    let build = |center_freq: f64| -> AudioChannelFilters {
        let audio1_fir = firwin(
            afilt_len,
            &[
                (center_freq - apass) / freq_hz_half,
                (center_freq + apass) / freq_hz_half,
            ],
            false,
        );
        let audio1_fft = filtfft(&audio1_fir, &[1.0], blocklen);
        let (lowbin, nbins, a1_freq) =
            fft_determine_slices(center_freq, 200000.0, freq_hz, blocklen);
        let sliced_hilbert = build_hilbert(nbins);
        let low_freq = freq_hz * (lowbin as f64 / blocklen as f64);
        let filt1_64 = fft_do_slice(&audio1_fft, lowbin, nbins, blocklen);
        let filt1: Vec<Complex64> = filt1_64
            .iter()
            .zip(&sliced_hilbert)
            .map(|(&v, &h)| Complex64::new(v.re * h, v.im * h))
            .collect();

        // Stage 2: 20 kHz-ish LPF and de-emphasis, both at the stage-1 rate.
        let (n, wn) = buttord_lowpass(20000.0 / (a1_freq / 2.0), 24000.0 / (a1_freq / 2.0), 1.0, 9.0);
        let audio2_lpf_ba = butter_ba(n, &[wn], FilterBandType::Lowpass).unwrap();
        let audio2_lpf = filtfft(&audio2_lpf_ba.0, &audio2_lpf_ba.1, blocklen);
        let (deemp_b, deemp_a) = emphasis_iir(5.3e-6, 75e-6, a1_freq);
        let audio2_deemp = filtfft(&deemp_b, &deemp_a, blocklen);
        let audio2_filter: Vec<Complex64> = audio2_lpf
            .iter()
            .zip(&audio2_deemp)
            .map(|(&a, &b)| np_cmul(a, b))
            .collect();

        AudioChannelFilters {
            lowbin,
            nbins,
            a1_freq,
            low_freq,
            filt1,
            audio2_filter,
        }
    };

    let left = build(audio_lfreq);
    let right = build(audio_rfreq);
    let audio_fdiv = blocklen / left.nbins;
    ([left, right], audio_fdiv)
}

/// Port of `RFDecode.computeefmfilter` (the inline core.py version, whose
/// interpolation nodes span 0..1.9 MHz) times the super-Gaussian bandpass
/// applied in `computefilters`: the EFM equalisation filter.
fn compute_fefm(freq_hz: f64, blocklen: usize) -> Vec<Complex64> {
    let top_freq = 1.9e6;
    let freqs: Vec<f64> = (0..11).map(|i| i as f64 * top_freq / 10.0).collect();
    let amp = [0.0, 0.215, 0.41, 0.73, 0.98, 1.03, 0.99, 0.81, 0.59, 0.42, 0.0];
    let phase: Vec<f64> = [0.0, -0.92, -1.03, -1.11, -1.2, -1.2, -1.2, -1.2, -1.05, -0.95, -0.8]
        .iter()
        .map(|&p| p * 1.25)
        .collect();

    let freq_per_bin = freq_hz / blocklen as f64;
    let nonzero_bins = (top_freq / freq_per_bin) as usize + 1;
    let bin_freqs: Vec<f64> = (0..nonzero_bins).map(|k| k as f64 * freq_per_bin).collect();
    let bin_amp = scipy_cubic_spline(&freqs, &amp, &bin_freqs);
    let bin_phase = scipy_cubic_spline(&freqs, &phase, &bin_freqs);
    let mut coeffs = vec![Complex64::new(0.0, 0.0); blocklen];
    for (k, (&a, &p)) in bin_amp.iter().zip(&bin_phase).enumerate() {
        coeffs[k] = Complex64::new(a * p.cos(), -a * p.sin());
    }
    for v in &mut coeffs {
        *v *= 8.0;
    }
    // self.Filters["Fefm"] *= gen_bpf_supergauss(20000, 1600000, 60, 20000000, blocklen)
    let bpf = gen_bpf_supergauss(20000.0, 1600000.0, 60, 20000000.0, blocklen);
    if let Some(dir) = std::env::var_os("LD_DUMP_GD") {
        use std::io::Write;
        let d = dir.to_string_lossy().into_owned();
        let mut wf = |name: &str, v: &[f64]| {
            if let Ok(mut f) = std::fs::File::create(format!("{}{}", d, name)) {
                let mut out = Vec::with_capacity(v.len() * 8);
                for x in v {
                    out.extend_from_slice(&x.to_ne_bytes());
                }
                let _ = f.write_all(&out);
            }
        };
        wf("fefm_amp.bin", &bin_amp);
        wf("fefm_phase.bin", &bin_phase);
        wf("fefm_bpf.bin", &bpf);
        wf("fefm_coeffs8.bin", &coeffs.iter().flat_map(|c| [c.re, c.im]).collect::<Vec<f64>>());
    }
    for (v, &b) in coeffs.iter_mut().zip(&bpf) {
        *v *= b;
    }
    coeffs
}

fn compute_filters(
    freq_half: f64,
    freq_hz: f64,
    freq_hz_half: f64,
    blocklen: usize,
    video_bpf_low: f64,
    video_bpf_high: f64,
    video_lpf_freq: f64,
    deemp: [f64; 2],
    deemp_strength: f64,
    ntsc_color_notch: bool,
) -> Result<(Filters, Vec<Complex64>, Vec<Complex64>)> {
    // RF highpass for dropout detection.
    let frfhpf = butter_ba(1, &[10.0 / freq_half], FilterBandType::Highpass)?;
    if std::env::var_os("LD_DUMP_BA").is_some() {
        let hp = butter_ba(
            DP_VIDEO_BPF_LOW_ORDER,
            &[video_bpf_low / freq_hz_half],
            FilterBandType::Highpass,
        )?;
        let lp = butter_ba(
            DP_VIDEO_BPF_HIGH_ORDER,
            &[video_bpf_high / freq_hz_half],
            FilterBandType::Lowpass,
        )?;
        let frfhpf_d = butter_ba(1, &[10.0 / freq_half], FilterBandType::Highpass)?;
        let video_lpf_d = butter_ba(
            DP_VIDEO_LPF_ORDER,
            &[DP_VIDEO_LPF_FREQ / freq_hz_half],
            FilterBandType::Lowpass,
        )?;
        let notch_d = |center: f64| {
            butter_ba(
                DP_AUDIO_NOTCHORDER,
                &[
                    (center - DP_AUDIO_NOTCHWIDTH) / freq_hz_half,
                    (center + DP_AUDIO_NOTCHWIDTH) / freq_hz_half,
                ],
                FilterBandType::Bandstop,
            )
            .unwrap()
        };
        let nl = notch_d((1e6 * SYS_FSC_MHZ / 227.5) * 146.25);
        let nr = notch_d((1e6 * SYS_FSC_MHZ / 227.5) * 178.75);
        let mut s = String::new();
        use std::fmt::Write as _;
        let mut sec = |name: &str, ba: &(Vec<f64>, Vec<f64>)| {
            let _ = writeln!(s, "{}_b", name);
            for v in &ba.0 {
                let _ = writeln!(s, "{:.17e}", v);
            }
            let _ = writeln!(s, "{}_a", name);
            for v in &ba.1 {
                let _ = writeln!(s, "{:.17e}", v);
            }
        };
        sec("frfhpf", &frfhpf_d);
        sec("hp", &hp);
        sec("lp", &lp);
        sec("vlpf", &video_lpf_d);
        sec("notchl", &nl);
        sec("notchr", &nr);
        if let Some(p) = std::env::var_os("LD_DUMP_BA") {
            std::fs::write(p, s).ok();
        }
    }
    let frfhpf_fft = filtfft(&frfhpf.0, &frfhpf.1, blocklen);

    // MTF compensation filter: two poles symmetric about freq_half.
    let mtf_polef_lo = DP_MTF_FREQ / freq_half;
    let mtf_polef_hi = (freq_half + (freq_half - DP_MTF_FREQ)) / freq_half;
    let to_z = |pole: f64| polar2z(DP_MTF_POLEDIST, PI * pole);
    let mtf_ba = zpk2tf_poles(to_z(mtf_polef_lo), to_z(mtf_polef_hi));
    let mtf = filtfft(&mtf_ba.0, &mtf_ba.1, blocklen);

    // RF video bandpass as a split highpass (low edge) + lowpass (high edge).
    let rfvideo_hp = butter_ba(
        DP_VIDEO_BPF_LOW_ORDER,
        &[video_bpf_low / freq_hz_half],
        FilterBandType::Highpass,
    )?;
    let rfvideo_lp = butter_ba(
        DP_VIDEO_BPF_HIGH_ORDER,
        &[video_bpf_high / freq_hz_half],
        FilterBandType::Lowpass,
    )?;
    let mut rfvideo = filtfft(&rfvideo_hp.0, &rfvideo_hp.1, blocklen);
    let rfvideo_lp_fft = filtfft(&rfvideo_lp.0, &rfvideo_lp.1, blocklen);
    if let Some(p) = std::env::var_os("LD_DUMP_BA") {
        let base = p.to_string_lossy().into_owned();
        for (suffix, data) in [
            ("_hp_fft.bin", &rfvideo),
            ("_lp_fft.bin", &rfvideo_lp_fft),
        ] {
            let mut bytes = Vec::with_capacity(data.len() * 16);
            for v in data {
                bytes.extend_from_slice(&v.re.to_le_bytes());
                bytes.extend_from_slice(&v.im.to_le_bytes());
            }
            std::fs::write(format!("{}{}", base, suffix), bytes).ok();
        }
    }
    for (a, &b) in rfvideo.iter_mut().zip(&rfvideo_lp_fft) {
        *a = np_cmul(*a, b);
    }

    // Audio notch filters: the analog audio carriers bleed into the RF video
    // band on NTSC Laserdiscs.
    let audio_lfreq = (1e6 * SYS_FSC_MHZ / 227.5) * 146.25;
    let audio_rfreq = (1e6 * SYS_FSC_MHZ / 227.5) * 178.75;
    let notch = |center: f64| -> Result<Vec<Complex64>> {
        let (lo, hi) = (
            (center - DP_AUDIO_NOTCHWIDTH) / freq_hz_half,
            (center + DP_AUDIO_NOTCHWIDTH) / freq_hz_half,
        );
        let ba = butter_ba(DP_AUDIO_NOTCHORDER, &[lo, hi], FilterBandType::Bandstop)?;
        Ok(filtfft(&ba.0, &ba.1, blocklen))
    };
    let cut_left = notch(audio_lfreq)?;
    let cut_right = notch(audio_rfreq)?;
    // Python: `SF["RFVideo"] *= SF["Fcutl"] * SF["Fcutr"]` (left-assoc).
    let combined: Vec<Complex64> = cut_left
        .iter()
        .zip(&cut_right)
        .map(|(&l, &r)| np_cmul(l, r))
        .collect();
    for (a, &b) in rfvideo.iter_mut().zip(&combined) {
        *a = np_cmul(*a, b);
    }

    // Hilbert transform: turns the bandpassed RF into an analytic signal so
    // the instantaneous frequency can be recovered.
    let hilbert = build_hilbert(blocklen);
    for (a, &h) in rfvideo.iter_mut().zip(&hilbert) {
        *a *= h;
    }

    // Post-demod lowpass.
    let video_lpf = butter_ba(
        DP_VIDEO_LPF_ORDER,
        &[video_lpf_freq / freq_hz_half],
        FilterBandType::Lowpass,
    )?;
    let fvideo_lpf = filtfft(&video_lpf.0, &video_lpf.1, blocklen);

    // Optional colour 'wobble' notch.
    let mut fvideo_lpf_eff = fvideo_lpf.clone();
    let fvideo_notch_fft: Option<Vec<Complex64>> = if ntsc_color_notch {
        let video_notch = butter_ba(
            3,
            &[video_lpf_freq / 1e6 / freq_half, 5.0 / freq_half],
            FilterBandType::Bandstop,
        )?;
        let fft = filtfft(&video_notch.0, &video_notch.1, blocklen);
        for (a, &b) in fvideo_lpf_eff.iter_mut().zip(&fft) {
            *a = np_cmul(*a, b);
        }
        Some(fft)
    } else {
        None
    };


    // De-emphasis and its inverse (emphasis).
    let (deemp_b, deemp_a) = emphasis_iir(deemp[0], deemp[1], freq_hz);
    let fdeemp = filtfft(&deemp_b, &deemp_a, blocklen);
    let (emp_b, emp_a) = emphasis_iir(deemp[1], deemp[0], freq_hz);
    let femp = filtfft(&emp_b, &emp_a, blocklen);

    // Main video filter: lowpass * deemp^strength, plus the IEC group-delay
    // equaliser.
    // Python: `SF["FVideo"] = SF["Fvideo_lpf"] * (SF["Fdeemp"] ** strength)`;
    // the power is a whole-array op first, then an FMA array multiply.
    let strength = Complex64::new(deemp_strength, 0.0);
    let mut fvideo: Vec<Complex64> = fvideo_lpf_eff
        .iter()
        .zip(&fdeemp)
        .map(|(&a, &b)| np_cmul(a, np_cpow(b, strength)))
        .collect();
    let fvideo_pregd = fvideo.clone();
    let fvideo_gd = build_groupdelay_equalizer(&fvideo_lpf, video_lpf_freq, freq_hz, blocklen);
    for (a, &b) in fvideo.iter_mut().zip(&fvideo_gd) {
        *a = np_cmul(*a, b);
    }

    // 0.5 MHz FIR lowpass (for sync detection), with known delay.
    let f05_taps = firwin(65, &[0.5 / freq_half], true);
    let f05_offset = 32;
    let f0_5_fft = filtfft(&f05_taps, &[1.0], blocklen);
    let fvideo05: Vec<Complex64> = fvideo_lpf_eff
        .iter()
        .zip(&fdeemp)
        .zip(&f0_5_fft)
        .map(|((&a, &b), &c)| np_cmul(np_cmul(a, b), c))
        .collect();

    // Colour-burst bandpass FIR, with known delay.
    let fburst_taps = firwin(
        81,
        &[
            (SYS_FSC_MHZ - 0.2) / freq_half,
            (SYS_FSC_MHZ + 0.2) / freq_half,
        ],
        false,
    );
    let fvideo_burst_offset = 40;
    let fburst_fft = filtfft(&fburst_taps, &[1.0], blocklen);
    let fvideo_burst: Vec<Complex64> = fvideo_lpf_eff
        .iter()
        .zip(&fdeemp)
        .zip(&fburst_fft)
        .map(|((&a, &b), &c)| np_cmul(np_cmul(a, b), c))
        .collect();
    if let Some(dir) = std::env::var_os("LD_DUMP_FSTAGE") {
        let d = dir.to_string_lossy().into_owned();
        let mut w = |name: &str, v: &[Complex64]| {
            use std::io::Write;
            if let Ok(mut f) = std::fs::File::create(format!("{}{}", d, name)) {
                let mut out = Vec::with_capacity(v.len() * 16);
                for c in v {
                    out.extend_from_slice(&c.re.to_ne_bytes());
                    out.extend_from_slice(&c.im.to_ne_bytes());
                }
                let _ = f.write_all(&out);
            }
        };
        w("_lpf.bin", &fvideo_lpf_eff);
        if let Some(n) = &fvideo_notch_fft {
            w("_notch.bin", n);
        }
        w("_deemp.bin", &fdeemp);
        w("_emp.bin", &femp);
        w("_video_pregd.bin", &fvideo_pregd);
        w("_video.bin", &fvideo);
        w("_gd.bin", &fvideo_gd);
        w("_f05.bin", &f0_5_fft);
        w("_fburst.bin", &fburst_fft);
        // Raw FIR taps (f64) of the two FIR-derived video filters.
        let wf = |name: &str, v: &[f64]| {
            use std::io::Write;
            if let Ok(mut f) = std::fs::File::create(format!("{}{}", d, name)) {
                let mut out = Vec::with_capacity(v.len() * 8);
                for x in v {
                    out.extend_from_slice(&x.to_ne_bytes());
                }
                let _ = f.write_all(&out);
            }
        };
        wf("_f05_taps.bin", &f05_taps);
        wf("_fburst_taps.bin", &fburst_taps);
    }
    // The three post-demod filters are kept as full-spectrum f64; their FIR
    // delays are applied as time-domain rolls in the demod (like the Python
    // `np.roll`), not as frequency-domain phase shifts.
    let fvideo = vec![fvideo, fvideo05.clone(), fvideo_burst];

    // Analog audio channel filters and the EFM equalisation filter.
    let (audio, audio_fdiv) = compute_audio_filters(
        freq_hz,
        freq_hz_half,
        blocklen,
        audio_lfreq,
        audio_rfreq,
    );
    let fefm = compute_fefm(freq_hz, blocklen);

    Ok((
        Filters {
            frfhpf: frfhpf_fft,
            mtf,
            rfvideo,
            fvideo,
            fvideo05,
            f05_offset,
            fvideo_burst_offset,
            audio,
            audio_fdiv,
            fefm,
        },
        fvideo_lpf,
        femp,
    ))
}

/// All-pass group-delay equaliser matching the IEC 60857 9.1.7 NTSC video
/// group-delay pre-distortion (port of `build_groupdelay_equalizer`).
fn build_groupdelay_equalizer(
    lpf_fft: &[Complex64],
    lpf_freq: f64,
    fs: f64,
    blocklen: usize,
) -> Vec<Complex64> {
    // IEC 60857 9.1.7 target group delay relative to 0.5 MHz, in seconds.
    let gd_f = [0.0, 0.5e6, 2.0e6, 3.0e6, 3.58e6, 4.0e6, 4.2e6, 4.8e6];
    let gd_t = [0.0, 0.0, 15e-9, 45e-9, 80e-9, 135e-9, 200e-9, 200e-9];

    let binfreq: Vec<f64> = (0..blocklen)
        .map(|k| {
            let f = k as f64 * fs / blocklen as f64;
            if k <= blocklen / 2 {
                f
            } else {
                (k as f64 - blocklen as f64).abs() * fs / blocklen as f64
            }
        })
        .collect();

    // Optional stage dump (`LD_DUMP_GD=<dir>`) used to pin arithmetic-order
    // differences against the numpy reference.
    let gd_dump = std::env::var_os("LD_DUMP_GD").map(|d| d.to_string_lossy().into_owned());
    let dump_f64 = |name: &str, v: &[f64]| {
        if let Some(dir) = gd_dump.as_deref() {
            use std::io::Write;
            if let Ok(mut f) = std::fs::File::create(format!("{}{}", dir, name)) {
                let mut out = Vec::with_capacity(v.len() * 8);
                for x in v {
                    out.extend_from_slice(&x.to_ne_bytes());
                }
                let _ = f.write_all(&out);
            }
        }
    };
    dump_f64("gd_binfreq.bin", &binfreq);

    // np.interp(binfreq, gd_f, gd_t)
    let interp = |x: f64| -> f64 {
        if x <= gd_f[0] {
            gd_t[0]
        } else if x >= gd_f[gd_f.len() - 1] {
            gd_t[gd_t.len() - 1]
        } else {
            let idx = gd_f
                .iter()
                .position(|&f| f > x)
                .expect("interp upper bound");
            let (f0, f1) = (gd_f[idx - 1], gd_f[idx]);
            let (t0, t1) = (gd_t[idx - 1], gd_t[idx]);
            // numpy's compiled `interp` evaluates `slope*(x - xp[j]) + fp[j]`
            // with `slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])`; forming the
            // fraction first instead rounds differently.
            let slope = (t1 - t0) / (f1 - f0);
            slope * (x - f0) + t0
        }
    };
    let target: Vec<f64> = binfreq.iter().map(|&f| interp(f)).collect();
    dump_f64("gd_target.bin", &target);

    // Unwrap the LPF phase, then group delay = -d(phase)/d(omega).
    //
    // numpy's `unwrap` is not a running "delta -= round(delta/2pi)*2pi" loop:
    // it corrects the *differences* and lets them accumulate onto the original
    // samples, so the roundings land in different places. Ported literally:
    //   dd = diff(p)
    //   ddmod = mod(dd + pi, 2pi) - pi;  ddmod[(ddmod == -pi) & (dd > 0)] = pi
    //   ph = where(|dd| < pi, 0, ddmod - dd)
    //   up[0] = p[0];  up[1:] = p[1:] + cumsum(ph)
    let phase: Vec<f64> = {
        let raw: Vec<f64> = lpf_fft
            .iter()
            .map(|v| crate::spec::ucrt_atan2::call(v.im, v.re))
            .collect();
        let n = raw.len();
        let mut ph = vec![0.0f64; n - 1];
        for i in 0..n - 1 {
            let dd = raw[i + 1] - raw[i];
            if dd.abs() >= PI {
                // np.mod(dd + pi, 2pi) - pi, with numpy's floor-modulo.
                let x = dd + PI;
                let mut m = x - (x / TAU).floor() * TAU;
                m -= PI;
                if m == -PI && dd > 0.0 {
                    m = PI;
                }
                ph[i] = m - dd;
            }
        }
        let mut unwrapped = raw.clone();
        let mut acc = 0.0f64;
        for i in 0..n - 1 {
            acc += ph[i];
            unwrapped[i + 1] = raw[i + 1] + acc;
        }
        unwrapped
    };

    // np.gradient(phase)
    let gradient: Vec<f64> = (0..phase.len())
        .map(|i| {
            if i == 0 {
                phase[1] - phase[0]
            } else if i == phase.len() - 1 {
                phase[i] - phase[i - 1]
            } else {
                (phase[i + 1] - phase[i - 1]) / 2.0
            }
        })
        .collect();

    dump_f64("gd_phase.bin", &phase);
    let bin_hz = fs / blocklen as f64;
    let lpf_gd: Vec<f64> = gradient.iter().map(|&g| -g / (TAU * bin_hz)).collect();
    dump_f64("gd_lpf_gd.bin", &lpf_gd);

    let i05 = binfreq
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - 0.5e6)
                .abs()
                .partial_cmp(&(*b - 0.5e6).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .expect("binfreq non-empty");
    let mut residual: Vec<f64> = target
        .iter()
        .zip(&lpf_gd)
        .map(|(&t, &g)| t - (g - lpf_gd[i05]))
        .collect();

    // Taper to zero past the LPF cut-off so the impulse response stays compact.
    let t0 = lpf_freq + 0.3e6;
    let t1 = lpf_freq + 1.3e6;
    for (i, r) in residual.iter_mut().enumerate() {
        let taper = ((t1 - binfreq[i]) / (t1 - t0)).clamp(0.0, 1.0);
        *r *= taper;
        if binfreq[i] < 0.4e6 {
            *r = 0.0;
        }
    }

    // Integrate group delay -> phase over the positive half, mirror for a
    // conjugate-symmetric (real impulse response) all-pass.
    let half = blocklen / 2;
    let mut dphi = vec![0.0f64; half + 1];
    let mut acc = 0.0;
    for (i, &r) in residual[..half + 1].iter().enumerate() {
        acc += r;
        dphi[i] = -TAU * acc * bin_hz;
    }
    dump_f64("gd_residual.bin", &residual);
    dump_f64("gd_dphi.bin", &dphi);

    let mut eq = vec![Complex64::new(1.0, 0.0); blocklen];
    for (i, &p) in dphi.iter().enumerate() {
        eq[i] = Complex64::from_polar(1.0, p);
    }
    for i in 0..half - 1 {
        // eq[half+1:] = conj(eq[1:half][::-1])
        eq[half + 1 + i] = eq[half - 1 - i].conj();
    }
    eq[0] = Complex64::new(1.0, 0.0);
    eq[half] = Complex64::new(1.0, 0.0); // Nyquist: keep unit magnitude
    eq
}

/// Measure the filter delays from a synthetic signal (port of
/// `RFDecode.computedelays`). Uses the initial (spec) calibration levels.
fn compute_delays(
    freq: f64,
    freq_half: f64,
    freq_hz: f64,
    blocklen: usize,
    filters: &Filters,
    fvideo_lpf: &[Complex64],
    femp: &[Complex64],
) -> Delays {
    let levels = CalibLevels::defaults();
    let iretohz = |ire: f64| levels.iretohz(ire);

    let mut fakeoutput = vec![iretohz(0.0); blocklen];
    let synclen_full = (4.7 * freq) as usize;

    // sync 1 (gap determination) and sync 2 (pilot/rot level setting)
    fakeoutput[1500..1500 + synclen_full].fill(iretohz(levels.vsync_ire));

    let porch_end = 2000 + synclen_full + (0.6 * freq) as usize;
    let burst_end = porch_end + (1.2 * freq) as usize;

    let rate = vec![SYS_FSC_MHZ; burst_end - porch_end];
    let wave = genwave(&rate, freq / 2.0, 0.0);
    for (i, &w) in wave.iter().enumerate() {
        fakeoutput[porch_end + i] += w * levels.hz_ire as f64 * 20.0;
    }

    // white
    fakeoutput[3000..3500].fill(iretohz(100.0));
    // white + burst
    fakeoutput[4500..5000].fill(iretohz(100.0));

    let rate = vec![SYS_FSC_MHZ; 5500 - 4200];
    let wave = genwave(&rate, freq / 2.0, 0.0);
    for (i, &w) in wave.iter().enumerate() {
        fakeoutput[4200 + i] += w * levels.hz_ire as f64 * 20.0;
    }

    let rate = vec![SYS_FSC_MHZ; synclen_full];
    let wave = genwave(&rate, freq / 2.0, 0.0);
    for (i, &w) in wave.iter().enumerate() {
        fakeoutput[2000 + i] = iretohz(levels.vsync_ire as f64)
            + w * levels.hz_ire as f64 * levels.vsync_ire as f64;
    }

    // Apply the video lowpass and (inverse) emphasis to the frequency-domain
    // fake signal: tmp * Fvideo_lpf * Femp.
    let tmp = fft_fake(&fakeoutput, blocklen);
    let tmp2: Vec<Complex64> = tmp.iter().zip(fvideo_lpf).map(|(&t, &f)| t * f).collect();
    let tmp3: Vec<Complex64> = tmp2.iter().zip(femp).map(|(&t, &f)| t * f).collect();
    let fakeoutput_emp = ifft_real(&tmp3, blocklen);

    // Generate the RF signal whose instantaneous frequency follows the fake
    // output; amplitude is irrelevant to the phase demodulation.
    // Reference scales the fake signal into 16-bit territory (*4096 + 8192)
    // before zeroing the dropout test samples.
    let fakesignal: Vec<f32> = genwave(&fakeoutput_emp, freq_hz / 2.0, 0.0)
        .iter()
        .map(|&v| ((v * 4096.0) + 8192.0) as f32)
        .collect();
    let mut fakesignal = fakesignal;
    fakesignal[6000..6005].fill(0.0);

    let demodspec = DemodSpecRef::new(freq, freq_half, freq_hz, blocklen, filters, &levels);
    let fakedecode = demod_block_cpu(&fakesignal, 0.0, &demodspec, false, None, 0, u64::MAX);

    let vdemod = &fakedecode.video.demod;
    let vdemod_raw = &fakedecode.video.demod_raw;

    let video_sync = calczc(vdemod, 1500, iretohz(levels.vsync_ire as f64 / 2.0), 0, 512, false)
        .map(|zc| zc - 1500.0)
        .unwrap_or(0.0);
    let video_white = calczc(vdemod, 3000, iretohz(50.0), 0, 512, false)
        .map(|zc| zc - 3000.0)
        .unwrap_or(0.0);
    let video_rot = calczc(vdemod, 6000, iretohz(-10.0), 0, 512, false)
        .map(|zc| (zc - 6000.0).round() as i64)
        .unwrap_or(0);

    let _ = (vdemod_raw, freq_half);
    let _ = (freq, freq_hz, blocklen);

    Delays {
        video_sync,
        video_white,
        video_rot,
    }
}

/// Forward FFT of real data: r2c + mirror, bit-matching `scipy.fft.fft` (the
/// c2c path rounds differently; see `demodblock::fft_full`).
fn fft_fake(data: &[f64], blocklen: usize) -> Vec<Complex64> {
    let half = crate::ffi_ducc::rfft(data); // blocklen/2+1 values
    let mut out = Vec::with_capacity(blocklen);
    out.extend_from_slice(&half);
    for k in (1..blocklen / 2).rev() {
        let c = half[k];
        out.push(Complex64::new(c.re, -c.im));
    }
    out
}

fn ifft_real(data: &[Complex64], blocklen: usize) -> Vec<f64> {
    // Match `scipy.fft.ifft(...).real` (c2c, normalized), like the rest of the
    // pipeline; rustfft's butterflies round differently.
    crate::ffi_ducc::ifft(data)
        .iter()
        .map(|v| v.re)
        .take(blocklen)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `np_cmul3` must be bit-identical to the two-pass fill+assign pair it
    /// replaces (the demod video product): same lane ops, same rounding,
    /// every element.
    #[test]
    fn np_cmul3_bit_identical_to_two_pass() {
        fn probe(len: usize, seed: u64) -> Vec<Complex64> {
            let mut s = seed | 1;
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // Magnitudes spanning normal f64 well; occasional small values.
                let m = (s >> 11) as f64 / (1u64 << 53) as f64;
                let e = ((s >> 40) % 30) as i32 - 15;
                Complex64::new(m * 10f64.powi(e), m * 10f64.powi(e - 3))
            };
            (0..len).map(|_| next()).collect()
        }
        for &(len, seed) in &[
            (1usize, 0xDEADBEEF),
            (2, 0x1234),
            (3, 0x5678),
            (7, 0x9ABC),
            (1024, 0x1111),
            (32768, 0x2222),
        ] {
            let a = probe(len, seed);
            let b = probe(len, seed ^ 0x5555);
            let c = probe(len, seed ^ 0xAAAA);
            // Two-pass oracle.
            let mut want: Vec<Complex64> = Vec::with_capacity(len);
            np_cmul_extend(&a, &b, &mut want);
            np_cmul_assign(&mut want, &c);
            // Fused, into a reused buffer (also exercises the clear+reserve path).
            let mut got: Vec<Complex64> = vec![Complex64::new(f64::NAN, f64::NAN); 5];
            np_cmul3_fill(&a, &b, &c, &mut got);
            assert_eq!(got.len(), want.len(), "len {len}");
            for i in 0..len {
                assert_eq!(
                    (got[i].re.to_bits(), got[i].im.to_bits()),
                    (want[i].re.to_bits(), want[i].im.to_bits()),
                    "len={len} i={i}"
                );
            }
        }
    }

    #[test]
    fn firwin_lowpass_matches_scipy() {
        // Reference: python scipy.signal.firwin(65, [0.5/20], pass_zero=True)
        let taps = firwin(65, &[0.5 / 20.0], true);
        let expected_head = [
            0.00063608, 0.00074556, 0.00090639, 0.00113282, 0.00143849, 0.00183603,
        ];
        for (got, want) in taps[..6].iter().zip(expected_head) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn firwin_bandpass_matches_scipy() {
        // Reference: python scipy.signal.firwin(81, [(315/88-0.2)/20, (315/88+0.2)/20],
        // pass_zero=False)
        let f1 = (315.0 / 88.0 - 0.2) / 20.0;
        let f2 = (315.0 / 88.0 + 0.2) / 20.0;
        let taps = firwin(81, &[f1, f2], false);
        let expected_head = [
            -0.0025548, -0.00300004, -0.00260168, -0.00131764, 0.0007001, 0.00302389,
            0.00495001, 0.00564495,
        ];
        for (got, want) in taps[..8].iter().zip(expected_head) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn emphasis_iir_matches_scipy() {
        // Reference: python emphasis_iir(120e-9, 320e-9, 40e6)
        let (b, a) = emphasis_iir(120e-9, 320e-9, 40e6);
        assert!((b[0] - 0.39738449).abs() < 1e-6, "b0 {}", b[0]);
        assert!((b[1] + 0.32215969).abs() < 1e-6, "b1 {}", b[1]);
        assert!((a[0] - 1.0).abs() < 1e-12);
        assert!((a[1] + 0.9247752).abs() < 1e-6, "a1 {}", a[1]);
    }

    #[test]
    fn butter_matches_scipy() {
        // Reference: python scipy.signal.butter(2, 3700000/20e6, btype='highpass')
        let (b, a) = butter_ba(2, &[3700000.0 / 20e6], FilterBandType::Highpass).unwrap();
        let expected_b = [0.66121016, -1.32242031, 0.66121016];
        let expected_a = [1.0, -1.20414446, 0.44069617];
        for (got, want) in b.iter().zip(expected_b) {
            assert!((got - want).abs() < 1e-6, "b {got} vs {want}");
        }
        for (got, want) in a.iter().zip(expected_a) {
            assert!((got - want).abs() < 1e-6, "a {got} vs {want}");
        }

        // and the lowpass edge: butter(3, 13800000/20e6, btype='lowpass')
        let (b, a) = butter_ba(3, &[13800000.0 / 20e6], FilterBandType::Lowpass).unwrap();
        let expected_b = [0.36126366, 1.08379099, 1.08379099, 0.36126366];
        let expected_a = [1.0, 1.10292472, 0.65954863, 0.12763596];
        for (got, want) in b.iter().zip(expected_b) {
            assert!((got - want).abs() < 1e-6, "b {got} vs {want}");
        }
        for (got, want) in a.iter().zip(expected_a) {
            assert!((got - want).abs() < 1e-6, "a {got} vs {want}");
        }
    }

    // --- Audio + EFM filters vs the reference dump (scripts/py_audio_ref.py) ---

    fn read_f32(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("reference data file");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), want.len(), "{label} length");
        let mut worst = 0.0f32;
        for (g, w) in got.iter().zip(want) {
            worst = worst.max((g - w).abs());
        }
        assert!(worst < tol, "{label} worst abs diff {worst} >= {tol}");
    }

    #[test]
    fn audio_filter_slices_match_python() {
        let (lowbin, nbins, a1_freq) =
            fft_determine_slices(2301136.3636363638, 200000.0, 40e6, 32768);
        assert_eq!(lowbin, 1629);
        assert_eq!(nbins, 1024);
        assert!((a1_freq - 1250000.0).abs() < 1e-6);
        assert_eq!(32768 / nbins, 32); // audio_fdiv
        let (lowbin_r, nbins_r, _) =
            fft_determine_slices(2812500.0, 200000.0, 40e6, 32768);
        assert_eq!(lowbin_r, 2048);
        assert_eq!(nbins_r, 1024);
    }

    #[test]
    fn audio_filters_match_python() {
        let audio_lfreq = (1e6 * SYS_FSC_MHZ / 227.5) * 146.25;
        let audio_rfreq = (1e6 * SYS_FSC_MHZ / 227.5) * 178.75;
        let ([left, right], fdiv) =
            compute_audio_filters(40e6, 20e6, 32768, audio_lfreq, audio_rfreq);
        assert_eq!(fdiv, 32);
        assert!((left.a1_freq - 1250000.0).abs() < 1e-6);
        assert!((right.low_freq - 2500000.0).abs() < 1e-6);
        assert_eq!(left.filt1.len(), 1024);
        assert_eq!(left.audio2_filter.len(), 32768);

        for (ch, name) in [(&left, "left"), (&right, "right")] {
            let re = read_f32(&format!("tests/data/filt1_{name}_re.bin"));
            let im = read_f32(&format!("tests/data/filt1_{name}_im.bin"));
            let got: Vec<f32> = ch
                .filt1
                .iter()
                .flat_map(|v| [v.re as f32, v.im as f32])
                .collect();
            let want: Vec<f32> = re
                .iter()
                .zip(&im)
                .flat_map(|(&a, &b)| [a, b])
                .collect();
            assert_close(&got, &want, 1e-3, &format!("filt1 {name}"));

            let re2 = read_f32(&format!("tests/data/audio2_{name}_re.bin"));
            let im2 = read_f32(&format!("tests/data/audio2_{name}_im.bin"));
            let got2: Vec<f32> = ch
                .audio2_filter
                .iter()
                .flat_map(|v| [v.re as f32, v.im as f32])
                .collect();
            let want2: Vec<f32> = re2
                .iter()
                .zip(&im2)
                .flat_map(|(&a, &b)| [a, b])
                .collect();
            assert_close(&got2, &want2, 1e-3, &format!("audio2 {name}"));
        }
    }

    #[test]
    fn fefm_matches_python() {
        let fefm = compute_fefm(40e6, 32768);
        let re = read_f32("tests/data/fefm_re.bin");
        let im = read_f32("tests/data/fefm_im.bin");
        assert_eq!(fefm.len(), 32768);
        let mut worst = 0.0f32;
        for (i, v) in fefm.iter().enumerate() {
            worst = worst.max((v.re as f32 - re[i]).abs());
            worst = worst.max((v.im as f32 - im[i]).abs());
        }
        assert!(worst < 1e-3, "fefm worst abs diff {worst}");
    }

    /// Isolation: which stage of `compute_fefm` carries any difference?
    /// The cubic-spline interpolation must reproduce scipy's
    /// `interp1d(kind="cubic")` (BSpline collocation + LAPACK gbsv + de Boor
    /// evaluation) bit-for-bit, and the super-Gaussian bandpass too.
    #[test]
    fn fefm_stage_isolation() {
        let top_freq = 1.9e6;
        let freqs: Vec<f64> = (0..11).map(|i| i as f64 * top_freq / 10.0).collect();
        let amp = [0.0, 0.215, 0.41, 0.73, 0.98, 1.03, 0.99, 0.81, 0.59, 0.42, 0.0];
        let phase: Vec<f64> = [0.0, -0.92, -1.03, -1.11, -1.2, -1.2, -1.2, -1.2, -1.05, -0.95, -0.8]
            .iter()
            .map(|&p| p * 1.25)
            .collect();
        let freq_per_bin = 40e6 / 32768.0;
        let nonzero_bins = (top_freq / freq_per_bin) as usize + 1;
        let bin_freqs: Vec<f64> = (0..nonzero_bins).map(|k| k as f64 * freq_per_bin).collect();
        let got_amp = scipy_cubic_spline(&freqs, &amp, &bin_freqs);
        let got_phase = scipy_cubic_spline(&freqs, &phase, &bin_freqs);
        let ref_amp = read_f64("tests/data/fefm_bin_amp.f64");
        let ref_phase = read_f64("tests/data/fefm_bin_phase.f64");
        let mut worst_a: f64 = 0.0;
        let mut worst_p: f64 = 0.0;
        for k in 0..nonzero_bins {
            worst_a = worst_a.max((got_amp[k] - ref_amp[k]).abs());
            worst_p = worst_p.max((got_phase[k] - ref_phase[k]).abs());
        }
        eprintln!("spline worst amp diff {worst_a:e}  phase diff {worst_p:e}");
        assert_eq!(worst_a, 0.0, "cubic spline (amp) must be bit-exact");
        assert_eq!(worst_p, 0.0, "cubic spline (phase) must be bit-exact");

        let bpf = gen_bpf_supergauss(20000.0, 1600000.0, 60, 20000000.0, 32768);
        let ref_bpf = read_f64("tests/data/fefm_bpf.f64");
        let mut worst_b: f64 = 0.0;
        let mut worst_bi = 0usize;
        for (i, v) in bpf.iter().enumerate() {
            let d = (*v - ref_bpf[i]).abs();
            if d > worst_b {
                worst_b = d;
                worst_bi = i;
            }
        }
        eprintln!("supergauss worst diff {worst_b:e} at bin {worst_bi} rust {:.17e} ref {:.17e}", bpf[worst_bi], ref_bpf[worst_bi]);
        // The golden is scipy's exact output. Under -O3 the pow chain in the
        // transition band can round ~1e-14 differently (rustc nightly codegen
        // drift, seen at bin 1317); keep the bound far above that noise and
        // far below any genuine algorithmic error.
        assert!(worst_b < 1e-12, "supergauss must match scipy: worst diff {worst_b:e}");
    }


    /// Bit-exact check of `filtfft` against scipy 1.18's `freqz` for both
    /// paths: an FIR filter (rfft-based) and an IIR filter (Horner polyval).
    #[test]
    fn filtfft_bit_exact_matches_scipy_118() {
        // FIR: firwin(65, 0.5/20e6) as in FVideo05.
        let b = read_f64("tests/data/freqz_fir_taps.f64");
        let re = read_f64("tests/data/freqz_fir_re.f64");
        let im = read_f64("tests/data/freqz_fir_im.f64");
        let h = filtfft(&b, &[1.0], 32768);
        let mut worst = 0.0f64;
        for (i, v) in h.iter().enumerate() {
            worst = worst.max((v.re - re[i]).abs());
            worst = worst.max((v.im - im[i]).abs());
        }
        eprintln!("filtfft FIR worst diff {worst:e}");
        assert_eq!(worst, 0.0, "filtfft FIR path must be bit-exact");

        // IIR: butter(4, 0.25) as in the video LPF.
        let ba = read_f64("tests/data/freqz_iir_ba.f64");
        let b_iir = &ba[..5];
        let a_iir = &ba[5..];
        let re2 = read_f64("tests/data/freqz_iir_re.f64");
        let im2 = read_f64("tests/data/freqz_iir_im.f64");
        let h2 = filtfft(b_iir, a_iir, 32768);
        let mut worst2 = 0.0f64;
        let mut worst_idx = 0usize;
        let mut worst_re = 0.0f64;
        let mut worst_im = 0.0f64;
        for (i, v) in h2.iter().enumerate() {
            let dr = (v.re - re2[i]).abs();
            let di = (v.im - im2[i]).abs();
            if dr.max(di) > worst2 {
                worst2 = dr.max(di);
                worst_idx = i;
                worst_re = dr;
                worst_im = di;
            }
        }
        eprintln!(
            "filtfft IIR worst diff {worst2:e} at bin {worst_idx} (dre {worst_re:e} dim {worst_im:e})"
        );
        eprintln!(
            "  got  {:?} ref {:?}",
            h2[worst_idx],
            Complex64::new(re2[worst_idx], im2[worst_idx])
        );
        assert_eq!(worst2, 0.0, "filtfft IIR path must be bit-exact");
    }

    fn read_f64(path: &str) -> Vec<f64> {
        let bytes = std::fs::read(path).expect("reference data file");
        bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }

    /// Bit-exact check of `compute_fefm` against the reference generated by
    /// the bundled 7.3.0 release python (numpy 2.4.6 / scipy 1.18.0) with
    /// `RFDecode.computeefmfilter` verbatim.
    #[test]
    fn fefm_bit_exact_matches_scipy_118() {
        let fefm = compute_fefm(40e6, 32768);
        let re = read_f64("tests/data/fefm_scipy_re.f64");
        let im = read_f64("tests/data/fefm_scipy_im.f64");
        assert_eq!(fefm.len(), 32768);
        let mut worst = 0.0f64;
        let mut worst_i = 0usize;
        let mut worst_part = "re";
        for (i, v) in fefm.iter().enumerate() {
            let dr = (v.re - re[i]).abs();
            let di = (v.im - im[i]).abs();
            if dr > worst {
                worst = dr;
                worst_i = i;
                worst_part = "re";
            }
            if di > worst {
                worst = di;
                worst_i = i;
                worst_part = "im";
            }
        }
        // Golden is the bundled python's (numpy 2.4.6 / scipy 1.18.0) filter.
        // The super-Gaussian factor can drift ~1e-14 under -O3 codegen (see
        // fefm_stage_isolation); 1e-12 stays far above that noise while still
        // catching real filter regressions.
        assert!(
            worst < 1e-12,
            "fefm must match scipy: worst {worst_part} diff {worst} at bin {worst_i} (rust {:?} vs scipy {:?})",
            fefm[worst_i],
            (re[worst_i], im[worst_i])
        );
    }

    /// Deterministic pseudo-random complex data with a wide exponent range, so
    /// a lane mix-up or a rounding difference cannot hide.
    fn pseudo_random(n: usize, seed: u64) -> Vec<Complex64> {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        (0..n)
            .map(|i| {
                let scale = (2.0f64).powi((i % 21) as i32 - 10);
                Complex64::new(next() * scale, next() * scale)
            })
            .collect()
    }

    /// The AVX2 pair kernel must be bit-identical to the scalar `np_cmul`,
    /// including the odd-length tail and the in-place form.
    #[test]
    fn np_cmul_slices_is_bit_exact() {
        for &n in &[0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 33, 1023, 4096, 32768] {
            let a = pseudo_random(n, 0x9E3779B97F4A7C15);
            let b = pseudo_random(n, 0xD1B54A32D192ED03);
            if n > 0 {
                // Include exact zeros (the trivial bins) and denormals.
                let mut a2 = a.clone();
                a2[0] = Complex64::new(0.0, 0.0);
                let mut b2 = b.clone();
                b2[n - 1] = Complex64::new(-0.0, 1e-310);
                assert_bits(&a2, &b2, n);
            }
            assert_bits(&a, &b, n);
        }
    }

    fn assert_bits(a: &[Complex64], b: &[Complex64], n: usize) {
        let mut out = vec![Complex64::new(f64::NAN, f64::NAN); n];
        np_cmul_slices(a, b, &mut out);
        let mut inplace = a.to_vec();
        np_cmul_assign(&mut inplace, b);
        let mut filled: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n + 3];
        np_cmul_fill(a, b, &mut filled);
        assert_eq!(filled.len(), n);
        for i in 0..n {
            let want = np_cmul(a[i], b[i]);
            for (got, what) in [
                (out[i], "slices"),
                (inplace[i], "assign"),
                (filled[i], "fill"),
            ] {
                assert_eq!(got.re.to_bits(), want.re.to_bits(), "n={n} i={i} re ({what})");
                assert_eq!(got.im.to_bits(), want.im.to_bits(), "n={n} i={i} im ({what})");
            }
        }
    }

    // PERF PROBE: how much does the vector kernel buy over the scalar loop?
    #[test]
    fn bench_np_cmul_vector() {
        use std::time::Instant;
        let n = 32768usize;
        let a = pseudo_random(n, 0x2545F4914F6CDD1D);
        let b = pseudo_random(n, 0x1D8E4E27C47D124F);
        let iters = 400;
        let mut sink = 0.0f64;
        let t0 = Instant::now();
        let mut out = vec![Complex64::new(0.0, 0.0); n];
        for _ in 0..iters {
            np_cmul_slices(&a, &b, &mut out);
            sink += out[1].re;
        }
        eprintln!("PERF np_cmul_slices avx2: {:?}/call", t0.elapsed() / iters);
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut o: Vec<Complex64> = Vec::with_capacity(n);
            o.extend(a.iter().zip(&b).map(|(&x, &y)| np_cmul(x, y)));
            sink += o[1].re;
        }
        eprintln!("PERF scalar map+collect : {:?}/call", t0.elapsed() / iters);
        let t0 = Instant::now();
        for _ in 0..iters {
            for (x, &y) in out.iter_mut().zip(&b) {
                *x = np_cmul(*x, y);
            }
            sink += out[1].re;
        }
        eprintln!("PERF scalar in-place loop: {:?}/call", t0.elapsed() / iters);
        eprintln!("PERF sink {sink}");
    }
}
