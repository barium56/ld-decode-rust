//! Bit-exact port of UCRT's `pow`, FMA variant, for the argument class the
//! decoder reaches.
//!
//! Why this exists: `cpow` hands a real non-negative base with a real exponent
//! to `pow`, and that was the last branch of `ucrt_exp_log::cpow` still served
//! by the platform library. It is not academic -- measured on the decoder's own
//! arguments (the MTF filter's DC bin with the MTF level, 1 010 pairs from a
//! real window, `work/mtf_dc_pairs.py` + `work/pow_diff.py`), **one pair
//! differs between UCRT and glibc by 1 ulp**, and a 1-ulp difference in a
//! filter coefficient is exactly the kind of thing that eventually flips a
//! marginal dropout or whiteloc decision. `np_cpow` now calls this first on
//! every platform and falls back to `powf` only for arguments outside the class
//! below.
//!
//! Provenance: `ucrtbase.dll` (`pow` at RVA 0x2c3e0, FMA body at 0x2d021,
//! selected by the same CPU-feature dispatch as `sin`/`cos`/`atan2`).
//! `work/gen_pow_tables.py` extracts the tables, `work/pe_dump.py` the
//! constants, and `work/numprobe/src/bin/ucrtpow.rs` is the gate: with this
//! transcription it reports **0 mismatches over 2 400 000 cases** (the
//! decoder's class densely sampled, plus a wider sweep) and 0 on the 1 010 real
//! pairs.
//!
//! Algorithm: `x^y = exp(y * ln(x))`, every intermediate carried in
//! double-double.
//!
//!   * `ln(x)` = `k*ln2 + ln(m)`: the mantissa is split against a 257-entry
//!     table point, whose reciprocal `2/(1+j/256)` comes from a head/tail table
//!     pair (`POW_T1`/`POW_T2`), and the series `f + f^2/2 + f^3/3 + ...` from
//!     a Horner chain in the same shape the `log` port uses,
//!   * `y * ln(x)` as four products with `y` split at 26 significant bits
//!     (`y & 0xfffffffff8000000`), so the leading product is exact,
//!   * `exp` of that with `n = rint(z*64/ln2)`, `k = n >> 6`, `j = n & 63`, a
//!     64-entry `2^(j/64)` head/tail pair (`POW_EA`/`EXP_T2`) and the `1/n!`
//!     Horner chain.
//!
//! The operation order is the listing's, deliberately: the double-double pairs
//! only cancel in that order, and several of the additions look
//! interchangeable but are not (a `w - (f + s)` where the parenthesisation is
//! what carries the rounding error, for instance).

// Used on every platform since 2026-09-20 (via `ucrt_exp_log::cpow`); before
// that Windows called the real UCRT and this module was referenced only by its
// own tests there.
#![allow(dead_code)]

use super::ucrt_exp_log_tables::{EXP_T2, LOG_HI, LOG_LO};
use super::ucrt_pow_tables::{POW_EA, POW_T1, POW_T2};

#[inline(always)]
fn val(b: u64) -> f64 {
    f64::from_bits(b)
}

/// UCRT evaluates with hardware FMA throughout; `mul_add` is the
/// correctly-rounded fused multiply-add, so it reproduces `vfmadd*` exactly.
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

// Masks and constants, from `work/pe_dump.py` at the addresses in the listing.
const IDX_MASK: u64 = 0x000f_f000_0000_0000;
const ROUND_BIT: u64 = 0x0000_0800_0000_0000;
const MANT_MASK: u64 = 0x000f_ffff_ffff_ffff;
/// The log tables are indexed in the `[0.5, 1)` binade (the same convention the
/// `log` port uses), so the table point and the mantissa are built by OR-ing
/// this exponent field into their bit patterns.
const HALF_EXP: u64 = 0x3fe0_0000_0000_0000;
/// `y` is split at 26 significant bits; the leading product `y_hi * ln_hi` is
/// then exact, which is what the rest of the pair is built on. The log's head
/// uses the same mask to clear the low 27 mantissa bits.
const SPLIT_27: u64 = 0xffff_ffff_f800_0000;

const LN2_HI: u64 = 0x3fe6_2e42_e000_0000; // 0.6931471228599548
const LN2_LO: u64 = 0x3e6e_fa39_ef35_793c; // 5.7699990475432854e-08
const C1_7: u64 = 0x3fc2_4924_9249_2494; // 0.1428571428571429 (1/7)
const C1_6: u64 = 0x3fc5_5555_5555_5555; // 1/6
const C1_5: u64 = 0x3fc9_9999_9999_999a; // 0.2
const C1_4: u64 = 0x3fd0_0000_0000_0000; // 1/4
const C1_3: u64 = 0x3fd5_5555_5555_5555; // 1/3
const C1_2: u64 = 0x3fe0_0000_0000_0000; // 1/2

const N_SCALE: u64 = 0x4057_1547_652b_82fe; // 92.33248261689366 = 64/ln2
const LN2_64_HI: u64 = 0x3f86_2e42_f000_0000; // 0.010830424260348082 = ln2/64
const LN2_64_LO: u64 = 0xbdfd_f473_de6a_f278; // -4.359010638708991e-10
const E1_720: u64 = 0x3f56_c16c_16c1_6c17; // 1/720
const E1_120: u64 = 0x3f81_1111_1111_1111; // 1/120
const E1_24: u64 = 0x3fa5_5555_5555_5555; // 1/24
const E1_6: u64 = 0x3fc5_5555_5555_5555; // 1/6
const E1_2: u64 = 0x3fe0_0000_0000_0000; // 0.5
const ONE: u64 = 0x3ff0_0000_0000_0000;

/// The listing's bounds on `z * 64/ln2`, above and below which `pow` goes to
/// its overflow/underflow paths.
const HI_LIMIT: f64 = 65536.0;
const LO_LIMIT: f64 = -68800.0;

/// UCRT `pow` for the decoder's class; `None` outside it so the caller can fall
/// back to the platform library.
///
/// Covered: a positive normal `x != 1`, a finite non-zero `y`, and a result
/// comfortably inside the normal range. Declined: subnormal/zero/negative/inf/
/// NaN `x` or `y`, `x == 1` (UCRT answers 1 exactly), `y` an odd integer with
/// negative `x` (a separate sign branch), and the overflow/subnormal-result
/// tails.
pub(crate) fn pow(x: f64, y: f64) -> Option<f64> {
    let xb = x.to_bits();
    let yb = y.to_bits();
    let xe = ((xb >> 52) & 0x7ff) as i32;
    if xb & (1 << 63) != 0 || xe == 0 || xe == 0x7ff || xb == ONE {
        return None;
    }
    let ye = ((yb >> 52) & 0x7ff) as i32;
    if y == 0.0 || ye == 0 || ye == 0x7ff {
        return None;
    }

    let ed = (xe - 1023) as f64; // the binary exponent as a double

    // ---- ln(x), double-double --------------------------------------------
    let idx_bits = (xb & IDX_MASK) + ((xb & ROUND_BIT) << 1);
    let j = (idx_bits >> 44) as usize;
    let tp = val(idx_bits | HALF_EXP);
    let m = val((xb & MANT_MASK) | HALF_EXP);
    let d = tp - m;
    let a1 = d * val(POW_T1[j]);
    let a2 = d * val(POW_T2[j]);
    let f = a1 + a2;
    let e0 = (a1 - f) + a2; // the exact two-sum error of that product
    let f2 = f * f;
    let mut q = fma(f, val(C1_7), val(C1_6));
    q = fma(f, q, val(C1_5));
    q = fma(f, q, val(C1_4));
    q = fma(f, q, val(C1_3));
    q = fma(f, q, val(C1_2));
    let poly = fma(f2, q, e0);
    let w = fma(ed, val(LN2_LO), -poly) + val(LOG_LO[j]);
    let s = w - f;
    let l = fma(ed, val(LN2_HI), val(LOG_HI[j]));
    let lp = l + s;
    let lp_hi = val(lp.to_bits() & SPLIT_27);
    let u = f + s;
    let l_lo = (((l - lp) + s) + (w - u)) + (lp - lp_hi);

    // ---- y * ln(x), double-double ----------------------------------------
    let y_hi = val(yb & SPLIT_27);
    let y_lo = y - y_hi;
    let p1 = y_lo * l_lo;
    let p2 = y_lo * lp_hi;
    let p3 = y_hi * l_lo;
    let p4 = y_hi * lp_hi;
    let s1 = p1 + p2;
    let s2 = s1 + p3;
    let z_hi = p4 + s2;
    let z_lo = (p4 - z_hi) + s2;

    // ---- exp(z), double-double -------------------------------------------
    let scaled = z_hi * val(N_SCALE);
    if scaled > HI_LIMIT || scaled < LO_LIMIT {
        return None;
    }
    let n = scaled.round_ties_even() as i32; // `vcvtpd2dq` rounds to nearest even
    let j2 = (n & 0x3f) as usize;
    let k = n >> 6;
    if k <= -1022 || k >= 1024 {
        return None; // the subnormal-result and huge-scale tails
    }
    let nd = n as f64;
    let r = fma(-nd, val(LN2_64_HI), z_hi) + nd * val(LN2_64_LO) + z_lo;
    let mut p = fma(r, val(E1_720), val(E1_120));
    p = fma(r, p, val(E1_24));
    p = fma(r, p, val(E1_6));
    p = fma(r, p, val(E1_2));
    p = fma(r, p, val(ONE));
    let poly = p * r; // e^r - 1
    let eb = val(EXP_T2[j2]);
    let ea = val(POW_EA[j2]);
    // (EA*poly) + ((EB*poly) + EB) + EA, in the listing's order.
    let hi = ea * poly + (eb * poly + eb);
    let scale = ((k + 1023) as u64) << 52;
    Some((hi + ea) * val(scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every expected value was read out of the real UCRT on Windows
    /// (`work/numprobe/src/bin/powpairs.rs`), never computed here, so this pins
    /// the port to the reference rather than to itself. The first two cases are
    /// the ones the decoder actually hits: the MTF filter's DC bin (0.413...
    /// from a real capture) raised to a real MTF level. The second is the pair
    /// on which glibc's `pow` differs from UCRT's by 1 ulp -- which is the
    /// whole reason this module exists.
    #[test]
    fn pow_matches_ucrt_bit_patterns() {
        assert_eq!(pow(0.413269546833581, 0.4).unwrap().to_bits(), 0x3fe678da76dae8dd);
        assert_eq!(
            pow(0.413269546833581, 0.26203478260869567).unwrap().to_bits(),
            0x3fe962bfae3ef6fc
        );
        assert_eq!(
            pow(0.413269546833581, 0.262034782608696).unwrap().to_bits(),
            0x3fe962bfae3ef6f9
        );
        assert_eq!(
            pow(0.413269546833581, 0.127999999999999).unwrap().to_bits(),
            0x3fec93e6a752e89f
        );
        assert_eq!(pow(0.5, 0.128).unwrap().to_bits(), 0x3fed487e0cf699aa);
        assert_eq!(pow(1.4, 0.28).unwrap().to_bits(), 0x3ff194a7e0e2ed3d);
        assert_eq!(pow(5.4, 0.2).unwrap().to_bits(), 0x3ff66b085f95e5b7);
        assert_eq!(pow(6.0, 0.5).unwrap().to_bits(), 0x4003988e1409212e);
        assert_eq!(pow(0.4, 1.0).unwrap().to_bits(), 0x3fd999999999999a);
        assert_eq!(pow(2.0, 0.5).unwrap().to_bits(), 0x3ff6a09e667f3bcd);
        assert_eq!(pow(3.9, 0.77).unwrap().to_bits(), 0x4006d07ede6e7689);
        assert_eq!(pow(1.5, -2.5).unwrap().to_bits(), 0x3fd7398bf1d1ee70);
        assert_eq!(pow(0.4, 1e-06).unwrap().to_bits(), 0x3feffffe141204b8);
        assert_eq!(pow(6.0, 0.999999).unwrap().to_bits(), 0x4017fffd2e8b0176);
        assert_eq!(pow(0.75, 0.333333333333333).unwrap().to_bits(), 0x3fed12ed0af1a280);
        assert_eq!(
            pow(1.0000000000000002, 0.5).unwrap().to_bits(),
            0x3ff0000000000000
        );
    }

    /// The argument classes the port deliberately does not cover must report
    /// `None`, so the real-base branch falls back to the platform `pow` instead
    /// of returning a value from an unreproduced branch.
    #[test]
    fn declines_arguments_it_does_not_cover() {
        assert!(pow(1.0, 0.5).is_none()); // UCRT answers 1 exactly
        assert!(pow(0.0, 0.5).is_none());
        assert!(pow(-2.0, 0.5).is_none()); // the negative-base sign branch
        assert!(pow(0.5, 0.0).is_none());
        assert!(pow(0.5, f64::NAN).is_none());
        assert!(pow(f64::INFINITY, 0.5).is_none());
        assert!(pow(0.5, 1e-320).is_none()); // subnormal exponent
        assert!(pow(1e-300, 3.0).is_none()); // subnormal result tail
        assert!(pow(1e300, 3.0).is_none()); // overflow tail
    }
}
