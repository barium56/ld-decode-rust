//! Bit-exact ports of UCRT's `exp`, `log`, `log1p` and of the `_dexp` /
//! `clogl` / `cexp` / `cpow` composition that `numpy.power` reaches through the
//! C99 `cpow` on Windows.
//!
//! Why this exists: `numpy.power` on complex128 calls the platform's `cpow`.
//! UCRT's `cpow` is `cexp(clogl(z) * w)` for every input except a real
//! non-negative base with a real exponent, which it hands to `pow`. glibc
//! instead computes `pow(|z|, w)` on the modulus, so the two disagree far
//! beyond rounding -- measured 39% of MTF-filter elements differing, up to
//! 3 ulp, at every MTF level -- and the MTF power spectrum is what the demod
//! multiplies with, so that is the last residue keeping a Linux decode off the
//! Windows reference bytes. The ports below make Linux compute what UCRT
//! computes.
//!
//! Provenance: `ucrtbase.dll`'s disassembly. `work/pe_dump.py` reads the
//! constants and tables out of the DLL by RVA, `work/gen_ucrt_tables.py`
//! regenerates `ucrt_exp_log_tables.rs`, and
//! `work/numprobe/src/bin/cpowcheck.rs` diffs this port against the real
//! functions on Windows (see the module test for the pinned results).
//!
//! The FMA variants are the ones ported, because every one of these functions
//! starts with a CPU-feature dispatch and this machine's UCRT takes the
//! AVX2+FMA path. That was settled for `sin`/`cos` by implementing both
//! variants and diffing against the real function (0 mismatches for FMA, 9816
//! for the scalar one), and the same dispatch is present here.
//!
//! Returns `None` where the port deliberately stops short -- non-positive or
//! subnormal `log`/`log1p` arguments, out-of-range `sin`/`cos`/`atan2` -- so
//! the caller falls back to the platform library instead of silently
//! returning a wrong value. The decoder's cpow arguments are MTF filter
//! elements and levels, i.e. magnitudes in `[0.4, 6]`.

// On Windows the decoder calls UCRT directly and this module is referenced only
// by its own tests, so most of it looks dead there.
#![allow(dead_code)]

use super::ucrt_exp_log_tables::{EXP_T1, EXP_T2, LOG_HI, LOG_INV, LOG_LO};
use super::{ucrt_atan2, ucrt_math};

#[inline(always)]
fn bits(x: f64) -> u64 {
    x.to_bits()
}

#[inline(always)]
fn val(b: u64) -> f64 {
    f64::from_bits(b)
}

/// UCRT evaluates with hardware FMA throughout; `f64::mul_add` is the
/// correctly-rounded fused operation, so it reproduces `vfmadd*` exactly.
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

const SIGN: u64 = 1 << 63;
const MANT_MASK: u64 = 0x000f_ffff_ffff_ffff;
/// The `2^(j/64)` table is used twice per element: whole, and as its 24-bit
/// head (the low 28 mantissa bits cleared).
const HEAD_CLEAR: u64 = (1 << 28) - 1;

// `exp` constants.
const EXP_K: f64 = f64::from_bits(0x4057_1547_652b_82fe); // 92.33248261689366 = 64/ln2
const EXP_CH: f64 = f64::from_bits(0xbf86_2e42_fefa_0000); // -0.010830424696223417
const EXP_CL: f64 = f64::from_bits(0xbd1c_f79a_bc9e_3b39); // -2.5728046223276688e-14
const EXP_P1: f64 = f64::from_bits(0x3f56_c16c_16c1_6c17); // 1/720
const EXP_P2: f64 = f64::from_bits(0x3f81_1111_1111_1111); // 1/120
const EXP_P3: f64 = f64::from_bits(0x3fa5_5555_5555_5555); // 1/24
const EXP_P4: f64 = f64::from_bits(0x3fc5_5555_5555_5555); // 1/6
const EXP_P5: f64 = f64::from_bits(0x3fe0_0000_0000_0000); // 0.5
const EXP_TINY: u64 = 0x3e50_0000_0000_0000; // 2^-26
const EXP_HI: f64 = f64::from_bits(0x4086_2e42_fefa_39ef); // 709.782712893384
const EXP_LO: f64 = f64::from_bits(0xc087_4046_dfef_d9d0); // -744.0346068132731

// `log` constants.
const LOG_INDEX: u64 = 0x000f_f000_0000_0000; // the index's 8 mantissa bits
const LOG_ROUND: u64 = 0x0000_0800_0000_0000; // the bit below them, shifted in
const LOG_HALF: u64 = 0x3fe0_0000_0000_0000; // 0.5, as an exponent field
const LOG_NEAR: f64 = f64::from_bits(0x3fb0_0000_0000_0000); // 0.0625
const LOG_C0: f64 = f64::from_bits(0x3fb5_5555_5555_54e6); // 0.08333333333333179
const LOG_C1: f64 = f64::from_bits(0x3f89_9999_99ba_c6d4); // 0.012500000003771751
const LOG_C2: f64 = f64::from_bits(0x3f62_4923_07f1_519f); // 0.0022321399879194482
const LOG_C3: f64 = f64::from_bits(0x3f3c_8034_c85d_fff0); // 0.0004348877777076146
const LOG_LN2_MID: f64 = f64::from_bits(0x3fe6_2e42_e000_0000); // 0.6931471228599548
const LOG_LN2_LO: f64 = f64::from_bits(0x3e6e_fa39_ef35_793c); // 5.7699990475432854e-08
const LOG_P0: f64 = f64::from_bits(0x3fc5_5555_5555_5555); // 1/6
const LOG_P1: f64 = f64::from_bits(0x3fc9_9999_9999_999a); // 0.2
const LOG_P2: f64 = f64::from_bits(0x3fd0_0000_0000_0000); // 0.25
const LOG_P3: f64 = f64::from_bits(0x3fd5_5555_5555_5555); // 1/3
const LOG_P4: f64 = f64::from_bits(0x3fe0_0000_0000_0000); // 0.5

// `_dexp` (the exponential `cexp` uses) constants.
const DEXP_LOG2E: f64 = f64::from_bits(0x3ff7_1547_652b_82fe); // 1.4426950408889634
const DEXP_LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_f800_0000); // 0.6931471675634384
const DEXP_LN2_LO: f64 = f64::from_bits(0x3e4b_e8e7_bcd5_e4f2); // 1.2996506893889889e-08
const DEXP_C1: f64 = f64::from_bits(0x3f01_52b7_41a5_e84b); // 3.304120783105597e-05
const DEXP_C2: f64 = f64::from_bits(0x3f50_3fa0_8157_2e11); // 0.0009917323526335046
const DEXP_C3: f64 = f64::from_bits(0x3f8c_70e8_daf3_bd0b); // 0.01388723295391838
const DEXP_C4: f64 = f64::from_bits(0x3fbc_718f_8c12_4358); // 0.11110779924116565
const DEXP_TINY: f64 = f64::from_bits(0x3c90_0000_0000_0000); // 2^-54
const DEXP_RANGE: f64 = 1842.0;

/// UCRT `exp`, FMA variant: `n = trunc(x*64/ln2)`, a degree-5 polynomial for
/// `exp(r) - 1`, and `2^(j/64) * 2^k` applied as a table split plus a
/// bit-pattern exponent add.
pub(crate) fn exp(x: f64) -> f64 {
    // The entry gate is `(x <= 709.78...) && !(x < -744.03...)`; anything else
    // (including NaN) leaves through UCRT's exception helpers. The decoder's
    // exponents are small, so the saturated answers here are for completeness.
    if x.is_nan() {
        return x;
    }
    if x > EXP_HI {
        return f64::INFINITY;
    }
    if x < EXP_LO {
        return 0.0;
    }
    if bits(x) & !SIGN <= EXP_TINY {
        // |x| <= 2^-26: `1 + x` needs no series.
        return 1.0 + x;
    }
    let nd = (x * EXP_K).trunc();
    let n = nd as i32;
    let mut r = fma(nd, EXP_CH, x);
    r = nd * EXP_CL + r;
    let mut p = EXP_P1;
    p = fma(r, p, EXP_P2);
    p = fma(r, p, EXP_P3);
    p = fma(r, p, EXP_P4);
    p = fma(r, p, EXP_P5);
    let v = fma(p, r * r, r);
    let j = (n & 63) as usize;
    let head = val(EXP_T1[j] & !HEAD_CLEAR);
    let mut y = v * val(EXP_T1[j]);
    y += val(EXP_T2[j]);
    y += head;
    exp_scale(y, n >> 6)
}

/// The tail of UCRT's `exp`: scale by `2^k`, either by adding `k` to the
/// exponent field (exact, and the only path a normal result takes) or, when the
/// result would be subnormal, by multiplying by `2^(k+1074)`.
fn exp_scale(y: f64, k: i32) -> f64 {
    if k > -1022 || (k == -1022 && y >= 1.0) {
        // `paddq` on the bit pattern: the low 64 bits wrap, and for a mantissa
        // in [1, 2) the add cannot carry into the exponent's own bits.
        val(bits(y).wrapping_add((k as u64) << 52))
    } else {
        let s = 1u64 << ((k + 1074) as u32);
        y * val(s)
    }
}

/// UCRT `log`, FMA variant, for a positive finite argument. `None` for a
/// subnormal or non-positive argument: subnormals go through a normalization
/// path whose register use is ambiguous in the listing, and the decoder cannot
/// reach one (its arguments are filter magnitudes in `[0.4, 6]`).
pub(crate) fn log(x: f64) -> Option<f64> {
    let e = ((bits(x) >> 52) & 0x7ff) as i32;
    if !(x > 0.0) || e == 0 || e == 0x7ff {
        return None;
    }
    Some(log_normal(x, (e - 1023) as f64))
}

/// The shared body of `log` for a normal positive argument, given the binary
/// exponent as a double.
fn log_normal(x: f64, k: f64) -> f64 {
    let b = bits(x);
    let near = x - 1.0;
    if near.abs() < LOG_NEAR {
        // |x - 1| < 1/16: the atanh series in f = (x-1)/(x+1), rearranged the
        // way UCRT does (t - t*f*(...) rather than 2*atanh(f) directly).
        let t = near;
        let d = t + 2.0;
        let f = t / d;
        let tt = t * f;
        let u = f + f;
        let z = u * u;
        let mut c1 = fma(z, LOG_C1, LOG_C0);
        let c2 = fma(z, LOG_C3, LOG_C2);
        let w3 = u * z;
        c1 *= w3;
        let w7 = u * (w3 * w3);
        c1 = fma(c2, w7, c1);
        return t + (c1 - tt);
    }
    // The index is the top mantissa bits with the next one down added in; the
    // add can carry, which is why the tables have a 257th entry.
    let idx_bits = (b & LOG_INDEX) + ((b & LOG_ROUND) << 1);
    let idx = (idx_bits >> 44) as usize;
    // `m_hi` keeps the sum whole: when the two terms carry, the sum is exactly
    // 0x0010000000000000, which ORs with the 0.5 exponent into 1.0 -- i.e. the
    // mantissa is so close to 2 that the table point is the next binade.
    let m_low = val((b & MANT_MASK) | LOG_HALF);
    let m_hi = val(idx_bits | LOG_HALF);
    let f = (m_hi - m_low) * val(LOG_INV[idx]);
    let z = f * f;
    let mut p = LOG_P0;
    let mut q = LOG_P3;
    p = fma(f, p, LOG_P1);
    q = fma(f, q, LOG_P4);
    let z2 = z * z;
    p = fma(f, p, LOG_P2);
    let mut acc = fma(q, z, f);
    acc = fma(p, z2, acc);
    let hi = fma(k, LOG_LN2_MID, val(LOG_HI[idx]));
    let lo = val(LOG_LO[idx]) + (k * LOG_LN2_LO - acc);
    hi + lo
}

/// UCRT `log1p`: `log(1+x)` with a one-term correction for the cancellation in
/// `(1+x) - 1`.
pub(crate) fn log1p(x: f64) -> Option<f64> {
    if x == 0.0 {
        return Some(x);
    }
    if x == f64::INFINITY {
        return Some(x);
    }
    if x < -1.0 {
        return Some(f64::NAN);
    }
    if x == -1.0 {
        return Some(f64::NEG_INFINITY);
    }
    let d = x + 1.0;
    let e = ((bits(d) >> 52) & 0x7ff) as i32;
    if e == 0 || e == 0x7ff {
        return None;
    }
    let l = log_normal(d, (e - 1023) as f64);
    let corr = ((d - 1.0) - x) / d;
    Some(l - corr)
}

/// UCRT `_dexp(x, y, 0)`: `y * exp(x)` with the reduction and the `2^k` scaling
/// UCRT uses, i.e. *not* `y * exp(x)` rounded afterwards. `cexp` calls it twice,
/// with `cos(im)` and `sin(im)`.
pub(crate) fn exp_mul(x: f64, y: f64) -> f64 {
    if x > DEXP_RANGE {
        return y * f64::INFINITY;
    }
    if x < -DEXP_RANGE {
        return y * 0.0;
    }
    let q = x * DEXP_LOG2E;
    let n = (q + if q >= 0.0 { 0.5 } else { -0.5 }).trunc() as i32;
    let kd = n as f64;
    let r = (x - kd * DEXP_LN2_HI) - kd * DEXP_LN2_LO;
    let mant = if r <= -DEXP_TINY || r >= DEXP_TINY {
        // exp(r) = (1 + A + B) / (1 + A - B) -- UCRT's rational form, with
        // A = (c2*r^2 + c4)*r^2 and B = ((c1*r^2 + c3)*r^2 + 0.5)*r.
        let r2 = r * r;
        let a = (r2 * DEXP_C2 + DEXP_C4) * r2;
        let b = ((r2 * DEXP_C1 + DEXP_C3) * r2 + 0.5) * r;
        ((b + a) + 1.0) / ((a - b) + 1.0)
    } else {
        // |r| < 2^-54: exp(r) rounds to 1, so the product is exactly y.
        1.0
    };
    scale_pow2(mant * y, n)
}

/// Exact scaling by `2^m` for the exponent range the decoder reaches (`|m|` a
/// few); outside it, the standard two-step split, which is what `_dscale` does.
fn scale_pow2(x: f64, m: i32) -> f64 {
    if m == 0 || !x.is_finite() || x == 0.0 {
        return x;
    }
    if (-1022..=1023).contains(&m) {
        return x * val(((m + 1023) as u64) << 52);
    }
    if m > 1023 {
        return scale_pow2(scale_pow2(x, 1023), m - 1023);
    }
    scale_pow2(scale_pow2(x, -1022), m + 1022)
}

/// UCRT `cexp`: `exp(re) * (cos(im), sin(im))`, each part computed by `_dexp` so
/// the multiply happens before the exponent scaling, exactly as UCRT orders it.
pub(crate) fn cexp(re: f64, im: f64) -> Option<(f64, f64)> {
    let c = ucrt_math::cos(im)?;
    let s = ucrt_math::sin(im)?;
    Some((exp_mul(re, c), exp_mul(re, s)))
}

/// The 26-bit head `_d_int(x, 26)` produces: the low `1075 - exponent - 26`
/// mantissa bits cleared, with every lower 16-bit word zeroed. The `log` split
/// path only calls it for `x` in `[0.4, 0.9]`, where the clear count is 27..36.
fn d_int26(x: f64) -> f64 {
    let b = bits(x);
    let e = ((b >> 52) & 0x7ff) as i32;
    let n = 1075 - e - 26;
    if n <= 0 {
        return x;
    }
    if n >= 53 {
        return val(b & !MANT_MASK);
    }
    let word = (n >> 4) as u32;
    let mask = ((1u64 << (n & 15)) - 1) << (word * 16);
    let lower = if word == 0 { 0 } else { (1u64 << (word * 16)) - 1 };
    val(b & !mask & !lower)
}

/// UCRT `clogl`: `(log|z|, atan2(im, re))`, with the modulus from the scaled
/// `0.5*log1p((min/max)^2) + log(max)` form or -- when both parts are within
/// `[0.4, 0.9]` -- from a `_d_int` split around zero so the `log1p` sees
/// `|z|^2 - 1` without cancellation.
pub(crate) fn clogl(re: f64, im: f64) -> Option<(f64, f64)> {
    let theta = ucrt_atan2::atan2(im, re)?;
    let (a, b) = (re.abs(), im.abs());
    let (mx, mn) = if a >= b { (a, b) } else { (b, a) };
    if mx == 0.0 {
        return Some((f64::NEG_INFINITY, theta));
    }
    let m = if mx > 0.9 || mn < 0.4 {
        let r = mn / mx;
        0.5 * log1p(r * r)? + log(mx)?
    } else {
        let mh = d_int26(mx);
        let nh = d_int26(mn);
        let t = ((mn - nh) * (nh + mn) + (mx - mh) * (mh + mx)) + ((mh * mh - 1.0) + nh * nh);
        0.5 * log1p(t)?
    };
    Some((m, theta))
}

/// UCRT `cpow`: `cexp(clogl(z) * w)`, except for a real non-negative base with a
/// real exponent, which goes to `pow`. The complex product is the plain
/// four-multiply form, in the disassembly's order.
///
/// The `pow` branch is the one piece not ported here: UCRT's `pow` is a
/// double-double `log2`/`exp2` pair, a separate job, and the composition
/// `exp(y*log(x))` from the ports above is *not* it (measured 45% of arguments
/// differing, up to 4 ulp). On Linux this branch therefore still calls the
/// platform `pow`, which the reference disagrees with on ~0.2% of arguments.
/// It is reached once per MTF level -- the DC bin is the only exactly-real
/// element of 32768, measured on a real capture -- and was identical to UCRT on
/// every level sampled so far.
pub(crate) fn cpow(ar: f64, ai: f64, br: f64, bi: f64) -> Option<(f64, f64)> {
    // Both parts of the base (im exactly zero, not merely a rounding away) and
    // the exponent must be real, and the base non-negative; the disassembly
    // tests exactly that, in that order.
    if ai == 0.0 && ar >= 0.0 && bi == 0.0 {
        return Some((ar.powf(br), 0.0));
    }
    let (m, t) = clogl(ar, ai)?;
    cexp(m * br - t * bi, m * bi + t * br)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every expected value was read out of the real UCRT on Windows
    /// (`work/numprobe/src/bin/explogpins.rs`), never computed here, so this
    /// pins the port to the reference rather than to itself. One case per
    /// branch: `exp`'s tiny/medium/subnormal tails and both saturating ends;
    /// `log`'s near-1 series, its table path, the index that carries into the
    /// 257th table entry, and both ends of the decoder's range; `log1p` around
    /// -1 and 0; and decoder-shaped `clogl`/`cexp`/`cpow` pairs, including the
    /// MTF filter element `LD_DUMP_MTFPOW` recorded at index 1 of a capture.
    #[test]
    fn exp_matches_ucrt_bit_patterns() {
        assert_eq!(exp(0.0).to_bits(), 0x3ff0000000000000);
        assert_eq!(exp(1.0).to_bits(), 0x4005bf0a8b145769);
        assert_eq!(exp(-1.0).to_bits(), 0x3fd78b56362cef38);
        assert_eq!(exp(2.0).to_bits(), 0x401d8e64b8d4ddae);
        assert_eq!(exp(-2.0).to_bits(), 0x3fc152aaa3bf81cc);
        assert_eq!(exp(1e-9).to_bits(), 0x3ff000000044b830); // |x| <= 2^-26
        assert_eq!(exp(-1e-9).to_bits(), 0x3fefffffff768fa1);
        assert_eq!(exp(0.010830424696223417).to_bits(), 0x3ff02c9a3e777fec); // one bucket step
        assert_eq!(exp(-0.010830424696223417).to_bits(), 0x3fefa7c1819e91bd);
        assert_eq!(exp(1.0 / 3.0).to_bits(), 0x3ff6546db1ba2d13);
        assert_eq!(exp(20.5).to_bits(), 0x41c7d6c4f0bcdd5c);
        assert_eq!(exp(-20.5).to_bits(), 0x3e157a3afeed00ab);
        assert_eq!(exp(700.0).to_bits(), 0x7f0d945df4f8ec8e); // subnormal tail
        assert_eq!(exp(-700.0).to_bits(), 0x00d14f2b0fb9307f);
        assert_eq!(exp(-740.0).to_bits(), 0x0000000000000055);
        assert_eq!(exp(709.0).to_bits(), 0x7fdd422d2be5dc9b);
        assert_eq!(exp(710.0), f64::INFINITY); // saturates rather than wraps
    }

    #[test]
    fn log_matches_ucrt_bit_patterns() {
        assert_eq!(log(1.0).unwrap().to_bits(), 0x0000000000000000);
        assert_eq!(log(0.5).unwrap().to_bits(), 0xbfe62e42fefa39ef);
        assert_eq!(log(0.413269546833581).unwrap().to_bits(), 0xbfec46e75c46c72d);
        assert_eq!(log(0.9).unwrap().to_bits(), 0xbfbaf8e8210a415c);
        assert_eq!(log(1.05).unwrap().to_bits(), 0x3fa8fb063ef2c7ef);
        assert_eq!(log(0.98).unwrap().to_bits(), 0xbf94b004bce0abf7); // near-1 series
        assert_eq!(log(1.02).unwrap().to_bits(), 0x3f944723d272a7f6);
        assert_eq!(log(2.0).unwrap().to_bits(), 0x3fe62e42fefa39ef);
        assert_eq!(log(5.5).unwrap().to_bits(), 0x3ffb46a5ef8151a0);
        assert_eq!(log(0.001).unwrap().to_bits(), 0xc01ba18a998fffa0);
        assert_eq!(log(1000.0).unwrap().to_bits(), 0x401ba18a998fffa0);
        assert_eq!(log(1e9).unwrap().to_bits(), 0x4034b927f32bffb8);
        assert_eq!(log(1e-9).unwrap().to_bits(), 0xc034b927f32bffb8);
        // 1.998187023536248: the index add carries into the 257th table entry.
        assert_eq!(log(1.998187023536248).unwrap().to_bits(), 0x3fe626d51719ec44);
    }

    #[test]
    fn log1p_matches_ucrt_bit_patterns() {
        assert_eq!(log1p(0.0).unwrap().to_bits(), 0x0000000000000000);
        assert_eq!(log1p(1e-12).unwrap().to_bits(), 0x3d719799812de065);
        assert_eq!(log1p(-0.5).unwrap().to_bits(), 0xbfe62e42fefa39ef);
        assert_eq!(log1p(0.5).unwrap().to_bits(), 0x3fd9f323ecbf984c);
        assert_eq!(log1p(1.0).unwrap().to_bits(), 0x3fe62e42fefa39ef);
        assert_eq!(log1p(1e6).unwrap().to_bits(), 0x402ba18abb1dedc8);
        assert_eq!(log1p(-0.999999).unwrap().to_bits(), 0xc02ba18a998fc064);
        assert_eq!(log1p(5e-3).unwrap().to_bits(), 0x3f746dd0fad67274);
    }

    #[test]
    fn cpow_matches_ucrt_bit_patterns() {
        let (re, im) = (0.413269546833581, 7.302115235329318e-5);
        assert_eq!(clogl(re, im).unwrap().0.to_bits(), 0xbfec46e753e55e19);
        assert_eq!(clogl(re, im).unwrap().1.to_bits(), 0x3f2728c71622297b);
        assert_eq!(cexp(re, im).unwrap().0.to_bits(), 0x3ff8302357a35b39);
        assert_eq!(cexp(re, im).unwrap().1.to_bits(), 0x3f1cf024275092db);
        assert_eq!(clogl(re, 0.0).unwrap().0.to_bits(), 0xbfec46e75c46c72d);
        assert_eq!(clogl(re, 0.0).unwrap().1.to_bits(), 0x0000000000000000);
        assert_eq!(cexp(re, 0.0).unwrap().0.to_bits(), 0x3ff8302358b852c1);
        assert_eq!(cexp(re, 0.0).unwrap().1.to_bits(), 0x0000000000000000);
        assert_eq!(clogl(1.4, -0.9).unwrap().0.to_bits(), 0x3fe04d32d8fdf7d6);
        assert_eq!(clogl(1.4, -0.9).unwrap().1.to_bits(), 0xbfe2486589dba9e0);
        assert_eq!(cexp(1.4, -0.9).unwrap().0.to_bits(), 0x40042a80674b228f);
        assert_eq!(cexp(1.4, -0.9).unwrap().1.to_bits(), 0xc00969919bd8bf2e);
        assert_eq!(clogl(0.5, 0.5).unwrap().0.to_bits(), 0xbfd62e42fefa39ef);
        assert_eq!(clogl(0.5, 0.5).unwrap().1.to_bits(), 0x3fe921fb54442d18);
        assert_eq!(cexp(0.5, 0.5).unwrap().0.to_bits(), 0x3ff726751e511e89);
        assert_eq!(cexp(0.5, 0.5).unwrap().1.to_bits(), 0x3fe94b46e77c3f11);
        assert_eq!(clogl(5.4, 0.2).unwrap().0.to_bits(), 0x3ffafe4c2a0c27af);
        assert_eq!(clogl(5.4, 0.2).unwrap().1.to_bits(), 0x3fa2f44cf5f2007b);
        assert_eq!(cexp(5.4, 0.2).unwrap().0.to_bits(), 0x406b1fc6e400504d);
        assert_eq!(cexp(5.4, 0.2).unwrap().1.to_bits(), 0x4045fe4b060fe625);
        assert_eq!(clogl(-2.0, 1.0).unwrap().0.to_bits(), 0x3fe9c041f7ed8d33);
        assert_eq!(clogl(-2.0, 1.0).unwrap().1.to_bits(), 0x40056c6e7397f5ae);
        assert_eq!(cexp(-2.0, 1.0).unwrap().0.to_bits(), 0x3fb2b81f02dce73a);
        assert_eq!(cexp(-2.0, 1.0).unwrap().1.to_bits(), 0x3fbd2749568d34e3);
        assert_eq!(
            cpow(0.413269546833581, 7.302115235329318e-5, 0.4, 0.0)
                .unwrap()
                .0
                .to_bits(),
            0x3fe678da78447f9a
        );
        assert_eq!(
            cpow(0.413269546833581, 7.302115235329318e-5, 0.4, 0.0)
                .unwrap()
                .1
                .to_bits(),
            0x3f0a059971bc1922
        );
        // The one real-base case: `cpow` hands this to `pow`. It is the DC bin
        // of an MTF filter -- the only exactly-real element of 32768 -- and the
        // one branch the port leaves to the platform library.
        assert_eq!(
            cpow(0.413269546833581, 0.0, 0.4, 0.0).unwrap().0.to_bits(),
            0x3fe678da76dae8dd
        );
        assert_eq!(cpow(0.413269546833581, 0.0, 0.4, 0.0).unwrap().1.to_bits(), 0x0);
        assert_eq!(cpow(1.4, -0.9, 0.28, 0.0).unwrap().0.to_bits(), 0x3ff237a6ae7edb94);
        assert_eq!(cpow(1.4, -0.9, 0.28, 0.0).unwrap().1.to_bits(), 0xbfc783f6aff747e0);
        assert_eq!(cpow(0.5, 0.5, 0.128, 0.0).unwrap().0.to_bits(), 0x3fee74f7ae54e86e);
        assert_eq!(cpow(0.5, 0.5, 0.128, 0.0).unwrap().1.to_bits(), 0x3fb893e76951d1ca);
    }

    /// The argument classes the port deliberately does not cover must report
    /// `None`, so the caller falls back instead of returning a wrong value.
    #[test]
    fn declines_arguments_it_does_not_cover() {
        assert!(log(0.0).is_none());
        assert!(log(-1.0).is_none());
        assert!(log(f64::NAN).is_none());
        assert!(log(1e-320).is_none()); // subnormal
        assert!(log1p(f64::NEG_INFINITY).unwrap().is_nan());
        assert!(log1p(-2.0).unwrap().is_nan());
        assert_eq!(log1p(-1.0).unwrap(), f64::NEG_INFINITY);
        assert!(clogl(f64::INFINITY, 0.0).is_none());
    }
}
