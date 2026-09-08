//! Field resampling with the ld-decode sinc-LUT scaler.
//!
//! This is the Rust port of `utils.scale_field` from the Python ld-decode:
//! the line-location spline (computed from expected vs. actual line locations)
//! gives per-output-sample coordinates and wow factors, and the wow factor is
//! used both as a level adjustment and (through the coordinate) to pick the
//! fractional-phase row of the Kaiser-windowed sinc lookup table. The spline
//! machinery (knots/coefficients and evaluation) is ported from the tape-decode
//! Rust project, which in turn matched the scipy `make_interp_spline` used by
//! both Python codebases.

use anyhow::{bail, Result};
use rayon::prelude::*;

use super::sinc::{build_kaiser_lut, SINC_PHASE_COUNT, SINC_TAP_COUNT};

/// Median (mean of the two middle order statistics for an even count) of a
/// slice, in place (reorders).
fn median_f64(values: &mut [f64]) -> f64 {
    assert!(!values.is_empty());
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        let (left, &mut hi, _) = values.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater)
        });
        let lo = *left
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater))
            .unwrap();
        (lo + hi) / 2.0
    } else {
        let (_, &mut median, _) = values.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater)
        });
        median
    }
}

/// Parameters for one [`scale_field_sinc`] call.
#[derive(Clone, Copy)]
pub(crate) struct SincScaleParams {
    pub lineoffset: usize,
    pub outwidth: usize,
    pub wow_level_adjust_smoothing: f32,
    pub level_adjust_threshold: f64,
}

/// Resample `buf` into `dsout` (length `linesout * outwidth`) using the
/// precomputed per-output-sample coordinates and wow factors from the
/// line-location spline. Port of `utils.scale_field` (Python ld-decode).
///
/// `interpolated_pixel_locs` and `wowfactors` must have at least
/// `outwidth * (lineoffset + 1) + dsout.len()` entries.
pub(crate) fn scale_field_sinc(
    buf: &[f32],
    dsout: &mut [f32],
    interpolated_pixel_locs: &[f64],
    wowfactors: &[f64],
    sinc_lut: &[f32],
    params: SincScaleParams,
) {
    assert_eq!(sinc_lut.len(), (SINC_PHASE_COUNT + 1) * SINC_TAP_COUNT);
    let level_adjust_threshold = params.level_adjust_threshold;

    // Average out unusual per-line spikes in wow: these indicate an hsync TBC
    // error rather than real playback-speed variation, so fall back to the
    // average wow to avoid a bright/dark line.
    let mut wow_copy = wowfactors.to_vec();
    let median = median_f64(&mut wow_copy);
    let mad = {
        let mut diffs: Vec<f64> = wow_copy.iter().map(|&w| (w - median).abs()).collect();
        median_f64(&mut diffs)
    };
    let threshold = if mad > 0.0 {
        level_adjust_threshold * mad
    } else {
        0.001 // fallback for no variance
    };

    let mut level_adjusts: Vec<f64> = wowfactors
        .iter()
        .map(|&w| {
            if (w - median).abs() > threshold {
                median
            } else {
                w
            }
        })
        .collect();

    if params.wow_level_adjust_smoothing > 0.0 {
        // Removes oscillating brightness variations: a low-pass filter that
        // smooths sudden brightness changes while staying reactive enough for
        // low-frequency wow.
        let alpha = 1.0 / (f64::from(params.wow_level_adjust_smoothing) * params.outwidth as f64);
        let one_minus_alpha = 1.0 - alpha;
        for i in 1..level_adjusts.len() {
            level_adjusts[i] = alpha * level_adjusts[i] + one_minus_alpha * level_adjusts[i - 1];
        }
    }

    let half_taps_m1 = (SINC_TAP_COUNT / 2) - 1;
    let dsout_start = params.outwidth * (params.lineoffset + 1);

    // Every output sample is an independent gather from `buf`, so the
    // interpolation is parallelized across chunks of the output (the per
    // sample f32/f64 arithmetic is unchanged, keeping the output bit-identical
    // to the serial loop).
    let mut dsout = dsout;
    let chunk = 4096usize;
    dsout
        .par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(ci, out)| {
            let base = dsout_start + ci * chunk;
            for (j, slot) in out.iter_mut().enumerate() {
                let i = base + j;
                // Compensates for the amplitude/frequency shift caused by FM
                // demodulation under varying playback speed.
                let level_adjust = level_adjusts[i];

                // Reconstruct the waveform at the proper fractional sample
                // position, undoing wow-induced timing variations. Clamp into
                // the interior so the 16-tap window stays in bounds; valid
                // fields never reach the edge, and the Python original raised
                // there (which its caller turned into a dropped field).
                let coord = (interpolated_pixel_locs[i] as f32)
                    .max(half_taps_m1 as f32)
                    .min((buf.len() - SINC_TAP_COUNT - half_taps_m1) as f32);
                let coord_int = coord as usize;
                let frac = coord - coord_int as f32;

                // Fractional phase: Python (numba) linearly interpolates
                // between the two adjacent LUT phase rows, with `alpha`
                // computed in f32.
                let phase_pos = frac * SINC_PHASE_COUNT as f32;
                let phase_start = phase_pos as usize;
                let phase_end = phase_start + 1;
                let alpha = phase_pos - phase_start as f32;

                let w_start = &sinc_lut[phase_start * SINC_TAP_COUNT..];
                let w_end = &sinc_lut[phase_end * SINC_TAP_COUNT..];

                let start = coord_int - half_taps_m1;
                // numba types `result = 0.0` as float64, so the accumulator
                // is f64 while each product is computed in f32.
                let mut result = 0.0f64;
                for t in 0..SINC_TAP_COUNT {
                    let ws = w_start[t];
                    let interp = ws + alpha * (w_end[t] - ws); // f32
                    result += (buf[start + t] * interp) as f64;
                }

                // The final level_adjust * result multiply happens in f64,
                // then the value is rounded to f32 on the store.
                *slot = (level_adjust * result) as f32;
            }
        });
}

// ============================================================================
// Interpolating spline (knots + coefficients) over the line locations
// ============================================================================

/// Build an interpolating spline of degree `k` over (x, y) for the wow
/// interpolation modes (k=1 linear, k=2 quadratic, k=3 natural cubic).
/// Ported from the tape-decode Rust project.
pub(crate) fn make_interp_spline_scaled(x: &[f64], y: &[f64], k: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    match k {
        1 => make_interp_spline::<1>(x, y, false),
        2 => make_interp_spline::<2>(x, y, false),
        3 => make_interp_spline::<3>(x, y, true),
        _ => bail!("unsupported spline degree"),
    }
}

/// Construct an interpolating B-spline (knots + coefficients) for degree `k`.
/// `natural` selects the zero-second-derivative boundary condition at both
/// ends, used for the cubic case; otherwise not-a-knot knots are used.
fn make_interp_spline<const K: usize>(
    x: &[f64],
    y: &[f64],
    natural: bool,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = x.len();
    if n != y.len() {
        bail!("make_interp_spline: x and y length mismatch");
    }
    if n < 2 {
        bail!("make_interp_spline: need at least two points");
    }

    // special-case k=1: t = [x0, x, x_{-1}], c = y (Lyche and Morken, Eq.(2.16)).
    if K == 1 {
        let mut t = Vec::with_capacity(n + 2);
        t.push(x[0]);
        t.extend_from_slice(x);
        t.push(x[n - 1]);
        return Ok((t, y.to_vec()));
    }

    // Construct the knot vector.
    let t: Vec<f64> = if natural {
        // _augknt(x, k): k copies of x0, x, k copies of x_{-1}.
        let mut t = Vec::with_capacity(n + 2 * K);
        for _ in 0..K {
            t.push(x[0]);
        }
        t.extend_from_slice(x);
        for _ in 0..K {
            t.push(x[n - 1]);
        }
        t
    } else {
        // _not_a_knot(x, k).
        let interior: Vec<f64> = if K % 2 == 1 {
            let k2 = K.div_ceil(2);
            x[k2..n - k2].to_vec()
        } else {
            let k2 = K / 2;
            let mids: Vec<f64> = (0..n - 1).map(|i| (x[i + 1] + x[i]) / 2.0).collect();
            mids[k2..mids.len() - k2].to_vec()
        };
        let mut t = Vec::with_capacity(2 * (K + 1) + interior.len());
        for _ in 0..=K {
            t.push(x[0]);
        }
        t.extend_from_slice(&interior);
        for _ in 0..=K {
            t.push(x[n - 1]);
        }
        t
    };

    let nt = t.len() - K - 1;
    let (nleft, nright) = if natural { (1usize, 1usize) } else { (0usize, 0usize) };
    if nt != n + nleft + nright {
        bail!("make_interp_spline: knot/condition count mismatch");
    }

    // Build the collocation matrix with boundary derivative rows.
    let mut a = vec![vec![0.0; nt]; nt];
    let mut rhs = vec![0.0; nt];

    // Left boundary derivative rows (natural: 2nd derivative == 0 at x[0]).
    if nleft > 0 {
        let span = bspline_find_span(&t, K, nt, x[0]);
        let ders = bspline_ders_basis::<K>(&t, span, x[0], 2);
        for j in 0..=K {
            a[0][span - K + j] = ders[2][j];
        }
        rhs[0] = 0.0;
    }

    // Collocation rows: spline value at each data point equals y.
    for i in 0..n {
        let span = bspline_find_span(&t, K, nt, x[i]);
        let ders = bspline_ders_basis::<K>(&t, span, x[i], 0);
        let row = nleft + i;
        for j in 0..=K {
            a[row][span - K + j] = ders[0][j];
        }
        rhs[row] = y[i];
    }

    // Right boundary derivative rows.
    if nright > 0 {
        let span = bspline_find_span(&t, K, nt, x[n - 1]);
        let ders = bspline_ders_basis::<K>(&t, span, x[n - 1], 2);
        let row = nt - nright;
        for j in 0..=K {
            a[row][span - K + j] = ders[2][j];
        }
        rhs[row] = 0.0;
    }

    let c = solve_dense(a, rhs)?;
    Ok((t, c))
}

/// Evaluate the spline value and first derivative at `x`. `span` is the
/// running knot-span index (updated in place) so consecutive evaluations walk
/// forward efficiently. Ported from the tape-decode Rust project.
pub(crate) fn eval_spline_value_deriv_k<const K: usize>(
    t: &[f64],
    c: &[f64],
    nt: usize,
    span: &mut usize,
    x: f64,
) -> (f64, f64) {
    if x <= t[K] {
        *span = K;
    } else if x >= t[nt] {
        *span = nt - 1;
    } else {
        while *span + 1 < nt && x >= t[*span + 1] {
            *span += 1;
        }
    }
    if K == 1 {
        let s = *span;
        let inv_h = 1.0 / (t[s + 1] - t[s]);
        let b0 = (t[s + 1] - x) * inv_h;
        let b1 = (x - t[s]) * inv_h;
        let c0 = c[s - 1];
        let c1 = c[s];
        // Value matches scipy's evaluate_spline bit-exactly. The derivative,
        // however, must reproduce FITPACK `_deBoor_D` (m = 1) followed by the
        // ascending accumulation `out += c * wrk` that scipy 1.18 performs in
        // `_dierckx.evaluate_spline` -- not the NURBS A2.3 basis, whose last-ulp
        // arithmetic differs.
        let mut deriv = c0 * (-inv_h);
        deriv += c1 * inv_h;
        return (b0 * c0 + b1 * c1, deriv);
    }
    let ders = bspline_ders_basis::<K>(t, *span, x, 1);
    let cofs = &c[*span - K..=*span];
    let mut value = 0.0;
    let mut deriv = 0.0;
    let dh = fitpack_deboor_deriv::<K>(t, *span, x);
    for j in 0..=K {
        let cj = cofs[j];
        value += ders[0][j] * cj;
        deriv += dh[j] * cj;
    }
    (value, deriv)
}

/// First-derivative basis values at `x` in knot span `span`, replicating
/// scipy's FITPACK `_deBoor_D` recursion with `m = 1` exactly (same operation
/// order and the same `xb == xa` guard), so that the wow factors match
/// python's `spl(x, 1)` bit-for-bit.
fn fitpack_deboor_deriv<const K: usize>(t: &[f64], ell: usize, x: f64) -> [f64; 4] {
    debug_assert!(K <= 3);
    let mut h = [0.0f64; 4];
    let mut hh = [0.0f64; 4];
    h[0] = 1.0;

    // k - m = K - 1 standard de Boor iterations (m == 1).
    for j in 1..K {
        hh[..j].copy_from_slice(&h[..j]);
        h[0] = 0.0;
        for n in 1..=j {
            let ind = ell + n;
            let xb = t[ind];
            let xa = t[ind - j];
            if xb == xa {
                h[n] = 0.0;
                continue;
            }
            let w = hh[n - 1] / (xb - xa);
            h[n - 1] += w * (xb - x);
            h[n] = w * (x - xa);
        }
    }

    // The single derivative recursion at level j == k.
    let j = K;
    hh[..j].copy_from_slice(&h[..j]);
    h[0] = 0.0;
    for n in 1..=j {
        let ind = ell + n;
        let xb = t[ind];
        let xa = t[ind - j];
        if xb == xa {
            h[n] = 0.0;
            continue;
        }
        let w = (j as f64) * hh[n - 1] / (xb - xa);
        h[n - 1] -= w;
        h[n] = w;
    }

    h
}

/// Evaluate the k+1 nonzero B-spline basis functions and their derivatives up
/// to order `d` at `x` in knot span `span`. Returns ders[order][j],
/// j = 0..=k indexing the basis functions B_{span-k+j}. Algorithm A2.3 from
/// "The NURBS Book". Degree is bounded by 3 here, so scratch storage is
/// fixed-size.
fn bspline_ders_basis<const K: usize>(t: &[f64], span: usize, x: f64, d: usize) -> [[f64; 4]; 3] {
    let p = K;
    debug_assert!(p <= 3 && d <= 2);
    let mut ndu = [[0.0f64; 4]; 4];
    let mut left = [0.0f64; 4];
    let mut right = [0.0f64; 4];
    ndu[0][0] = 1.0;
    for j in 1..=p {
        left[j] = x - t[span + 1 - j];
        right[j] = t[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    let mut ders = [[0.0f64; 4]; 3];
    for j in 0..=p {
        ders[0][j] = ndu[j][p];
    }

    let mut a = [[0.0f64; 4]; 2];
    for r in 0..=p {
        let mut s1 = 0usize;
        let mut s2 = 1usize;
        a[0][0] = 1.0;
        for kk in 1..=d {
            let mut acc = 0.0;
            let rk = r as isize - kk as isize;
            let pk = p as isize - kk as isize;
            if r >= kk {
                a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                acc = a[s2][0] * ndu[rk as usize][pk as usize];
            }
            let j1 = if rk >= -1 { 1 } else { (-rk) as usize };
            let j2 = if (r as isize - 1) <= pk {
                kk - 1
            } else {
                p - r
            };
            for j in j1..=j2 {
                a[s2][j] =
                    (a[s1][j] - a[s1][j - 1]) / ndu[(pk + 1) as usize][(rk + j as isize) as usize];
                acc += a[s2][j] * ndu[(rk + j as isize) as usize][pk as usize];
            }
            if (r as isize) <= pk {
                a[s2][kk] = -a[s1][kk - 1] / ndu[(pk + 1) as usize][r];
                acc += a[s2][kk] * ndu[r][pk as usize];
            }
            ders[kk][r] = acc;
            std::mem::swap(&mut s1, &mut s2);
        }
    }

    // Multiply through by the correct factors (Eq. [2.9] in "The NURBS Book").
    let mut r = p;
    for kk in 1..=d {
        for j in 0..=p {
            ders[kk][j] *= r as f64;
        }
        r *= p - kk;
    }
    ders
}

/// Find the knot span index `i` such that t[i] <= x < t[i+1], clamped to the
/// valid evaluation range [k, nt-1] so out-of-bounds x uses the edge polynomial.
fn bspline_find_span(t: &[f64], k: usize, nt: usize, x: f64) -> usize {
    if x <= t[k] {
        return k;
    }
    if x >= t[nt] {
        return nt - 1;
    }
    let mut lo = k;
    let mut hi = nt;
    let mut mid = (lo + hi) / 2;
    while x < t[mid] || x >= t[mid + 1] {
        if x < t[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
        mid = (lo + hi) / 2;
    }
    mid
}

/// Solve a dense linear system A x = b in place using Gaussian elimination
/// with partial pivoting. `a` is row-major n x n; returns the solution vector.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best == 0.0 {
            bail!("Collocation matrix is singular.");
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        let inv = 1.0 / a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] * inv;
            if factor != 0.0 {
                for c in col..n {
                    a[row][c] -= factor * a[col][c];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for c in (i + 1)..n {
            sum -= a[i][c] * x[c];
        }
        x[i] = sum / a[i][i];
    }
    Ok(x)
}

/// Convenience: the kaiser sinc LUT, built once and cached.
pub(crate) fn sinc_lut() -> &'static [f32] {
    use std::sync::OnceLock;
    static LUT: OnceLock<Vec<f32>> = OnceLock::new();
    LUT.get_or_init(|| {
        let lut = build_kaiser_lut();
        if let Some(p) = std::env::var_os("LD_DUMP_LUT") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::File::create(&p) {
                for v in &lut {
                    let _ = writeln!(f, "{:08x}", v.to_bits());
                }
            }
        }
        lut
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_spline_reproduces_points() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 1.0, 0.0, 1.0];
        let (t, c) = make_interp_spline_scaled(&x, &y, 1).unwrap();
        let nt = t.len() - 2;
        let mut span = 1usize;
        for i in 0..4 {
            let (value, _) = eval_spline_value_deriv_k::<1>(&t, &c, nt, &mut span, x[i]);
            assert!((value - y[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn sinc_scaler_passes_through_aligned_signal() {
        // With identity wow factors and coordinates, resampling a constant
        // signal must preserve it exactly (LUT rows sum to 1).
        let lut = build_kaiser_lut();
        let buf = vec![7.0f32; 1000];
        let outwidth = 100;
        let lineoffset = 2;
        let eval_count = outwidth * (lineoffset + 1) + 300;
        let locs: Vec<f64> = (0..eval_count).map(|i| i as f64).collect();
        let wow = vec![1.0f64; eval_count];
        let mut out = vec![0.0f32; 300];
        scale_field_sinc(
            &buf,
            &mut out,
            &locs,
            &wow,
            &lut,
            SincScaleParams {
                lineoffset,
                outwidth,
                wow_level_adjust_smoothing: 0.0,
                level_adjust_threshold: 15.0,
            },
        );
        for &v in &out {
            assert!((v - 7.0).abs() < 1e-4, "got {v}");
        }
    }
}
