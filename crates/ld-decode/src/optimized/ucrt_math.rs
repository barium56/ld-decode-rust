//! Bit-exact ports of UCRT's `cos` and `sin`, reconstructed from the
//! disassembly of `ucrtbase.dll`.
//!
//! Why this exists: on Windows every libm call on the parity path resolves to
//! UCRT, and UCRT is not glibc -- they disagree by 1-2 ulp on a few percent of
//! arguments (measured over 200k cases per function; `cpow` by far more). A
//! Linux build that calls glibc therefore cannot reproduce the Windows
//! reference byte-for-byte, which is the contract this project holds itself
//! to. These functions are the UCRT values, evaluated with the same operation
//! order, so Linux can produce them without linking UCRT.
//!
//! Provenance, so this can be re-derived if it is ever doubted:
//! - `llvm-objdump -d --disassemble-symbols=cos,sin ucrtbase.dll` for the code,
//!   and a PE reader for the coefficients it references in rodata.
//! - UCRT contains **two** implementations of each function and picks one at
//!   run time from a CPU-feature flag (`cmp dword [rip+0xc6e1d], 0` at entry):
//!   a scalar variant and a VEX/FMA-encoded one. This is the FMA variant, which
//!   is the one this machine takes -- measured by implementing both and diffing
//!   against the real UCRT: the scalar variant mismatched 9 816 of 400 000 cases
//!   while the FMA variant mismatched 0. The two variants differ from each other
//!   in 8 970 of those cases, so the reference corpus is necessarily the output
//!   of one specific variant.
//! - Validated against the real UCRT on Windows over 1 131 090 arguments
//!   covering the whole decoder range (`work/numprobe/src/bin/ucrtsincos.rs`):
//!   0 mismatches for both functions.
//!
//! Rust's `mul_add` is the correctly-rounded fused multiply-add, so it matches
//! `vfmadd`/`vfnmadd` exactly; a plain `a * b + c` would not.

// --- polynomial coefficients (ucrtbase.dll rodata) ---------------------------
/// cos kernel, in the order UCRT combines them.
const C1400: f64 = 0.041666666666666664; // +1/4!
const C1410: f64 = -0.0013888888888887398; // -1/6!
const C1420: f64 = 2.4801587298767044e-05; // +1/8!
const C1430: f64 = -2.755731727234489e-07; // -1/10!
const C1440: f64 = 2.0876146382372144e-09; // +1/12!
const C1450: f64 = -1.138263981623609e-11; // -1/14!
/// sin kernel polynomial.
const C1460: f64 = -0.16666666666666666; // -1/3!
const C1470: f64 = 0.00833333333333095; // +1/5!
const C1480: f64 = -0.00019841269836761127; // -1/7!
const C1490: f64 = 2.7557316103728802e-06; // +1/9!
const C14A0: f64 = -2.5051132068021698e-08; // -1/11!
const C14B0: f64 = 1.5918144304485914e-10; // -1/13!

// --- argument-reduction constants (ucrtbase.dll rodata) ----------------------
const TWO_OVER_PI: f64 = 0.6366197723675814;
const PI_OVER_2: f64 = 1.5707963267948966; // round(pi/2)
const PI_OVER_2_MID: f64 = 6.123233995736757e-17;
const PI_OVER_2_LO: f64 = 8.478427660368898e-32; // fdlibm's pio2_3t
const ROUND_MAGIC: f64 = 6755399441055744.0; // 1.5 * 2^52

// --- branch thresholds (ucrtbase.dll rodata) ---------------------------------
const TINY: f64 = 7.450580596923828e-09; // 2^-27
const SMALL: f64 = 0.0001220703125; // 2^-13
const QUARTER_PI: f64 = 0.7853981633974483;
/// Above this UCRT switches to a second, out-of-line reducer that is not
/// ported. Every argument the decoder produces is below 2*pi, so this is a
/// guard rail rather than a used boundary.
const HUGE: f64 = 20000000.0;
const SIN_SMALL_C: f64 = 0.16666666666666666; // 1/6, sin's 2^-27..2^-13 branch

/// `ucrtbase+0x2f540`: reduce `x >= 0` to `x - n*(pi/2)`, returning the high
/// and low parts of the remainder and `n & 3`.
///
/// Three-word pi/2 split plus FMA error terms, exactly as UCRT accumulates
/// them; `ROUND_MAGIC` rounds `x * 2/pi` to nearest under the current FPU mode
/// without an explicit conversion.
#[inline]
fn reduce(x: f64) -> (f64, f64, u32) {
    let mut n = x.mul_add(TWO_OVER_PI, ROUND_MAGIC);
    n -= ROUND_MAGIC;
    let ni = n as i32 as u32;

    let mut t = (-n).mul_add(PI_OVER_2, x); // x - n*pi/2 (high part only)
    let prod = n * PI_OVER_2_MID; // n * A
    let prod_err = PI_OVER_2_MID.mul_add(n, -prod); // exact error of that product
    let mut e = t - prod;
    let mut c = t - e;
    c -= prod;
    let r_hi = (-n).mul_add(PI_OVER_2_MID, t);
    t = r_hi;
    e -= t;
    e += c;
    e -= prod_err;
    let r_lo = (-n).mul_add(PI_OVER_2_LO, e);
    (r_hi, r_lo, ni & 3)
}

/// `cos(r) = 1 - z/2 + z^2*P(z)` with the compensated `1 - z/2` and the
/// `-r_hi*r_lo` correction (`ucrtbase+0x29fc6`).
#[inline]
fn cos_kernel(r_hi: f64, r_lo: f64) -> f64 {
    let z = r_hi * r_hi;
    let half_z = z * 0.5;
    let hi = 1.0 - half_z;
    let mut acc = 1.0 - hi;
    acc -= half_z;
    acc = (-r_hi).mul_add(r_lo, acc);

    // Note the first step's operand roles: UCRT computes `fma(z, C1450, C1440)`
    // -- coefficient as multiplicand, accumulator as addend -- before switching
    // to `fma(z, acc, c)`. Written as plain descending Horner. Swapping the two
    // roles is worth ~1e-11 absolute, which the validation harness catches.
    let zz = z * z;
    let mut p = C1440;
    p = C1450.mul_add(z, p);
    p = p.mul_add(z, C1430);
    p = p.mul_add(z, C1420);
    p = p.mul_add(z, C1410);
    p = p.mul_add(z, C1400);
    p = p.mul_add(zz, acc);
    p + hi
}

/// `sin(r) = r + r^3*P(z)` with the `r_lo` correction (`ucrtbase+0x2fa0f`).
#[inline]
fn sin_kernel(r_hi: f64, r_lo: f64) -> f64 {
    let z = r_hi * r_hi;
    let mut p = C14A0;
    p = C14B0.mul_add(z, p);
    p = p.mul_add(z, C1490);
    p = p.mul_add(z, C1480);
    p = p.mul_add(z, C1470);

    let r3 = r_hi * z;
    let mut t = r3 * p;
    let half_lo = r_lo * 0.5;
    t = half_lo - t;
    t *= z;
    t -= r_lo;
    t = (-r3).mul_add(C1460, t);
    r_hi - t
}

/// `ucrtbase!cos` (FMA variant) for the decoder's argument range.
///
/// Returns `None` outside that range, where UCRT calls an unported reducer; the
/// caller must then fall back to the platform library rather than silently use
/// a wrong value.
pub fn cos(x: f64) -> Option<f64> {
    if !x.is_finite() {
        return None;
    }
    let a = x.abs();
    if a >= HUGE {
        return None;
    }
    // Below pi/4 the sign is irrelevant: cos is even, and UCRT reduces |x|.
    if a < SMALL {
        return Some(if a >= TINY {
            let half_x = x * 0.5; // single FMA: 1 - (0.5*x)*x
            (-half_x).mul_add(x, 1.0)
        } else {
            1.0
        });
    }
    if a < QUARTER_PI {
        let z = x * x;
        let mut acc = C1450;
        acc = z.mul_add(acc, C1440);
        acc = z.mul_add(acc, C1430);
        acc = z.mul_add(acc, C1420);
        acc = z.mul_add(acc, C1410);
        acc = z.mul_add(acc, C1400);
        acc = z.mul_add(acc, -0.5);
        return Some(z.mul_add(acc, 1.0));
    }

    // cos(r + n*pi/2): even n keeps the cosine kernel, odd n swaps to sine.
    let (r_hi, r_lo, n) = reduce(a);
    let val = if n & 1 == 0 {
        cos_kernel(r_hi, r_lo)
    } else {
        sin_kernel(r_hi, r_lo)
    };
    Some(if n.wrapping_add(1) & 2 != 0 { -val } else { val })
}

/// `ucrtbase!sin` (FMA variant) for the decoder's argument range.
pub fn sin(x: f64) -> Option<f64> {
    if !x.is_finite() {
        return None;
    }
    let a = x.abs();
    if a >= HUGE {
        return None;
    }
    if a < SMALL {
        if a < TINY {
            // sin(x) rounds to x here; UCRT's extra arithmetic only raises flags.
            return Some(x);
        }
        let x3 = x * x * x;
        return Some((-x3).mul_add(SIN_SMALL_C, x));
    }
    if a < QUARTER_PI {
        let z = x * x;
        let mut p = C14B0;
        p = z.mul_add(p, C14A0);
        p = z.mul_add(p, C1490);
        p = z.mul_add(p, C1480);
        p = z.mul_add(p, C1470);
        p = z.mul_add(p, C1460);
        let x3 = x * z;
        return Some(x3.mul_add(p, x));
    }

    // sin(r + n*pi/2): even n keeps the sine kernel, odd n swaps to cosine
    // (the `jb` at ucrtbase+0x2fa0d jumps *to* the cosine kernel on odd n).
    let (r_hi, r_lo, n) = reduce(a);
    let val = if n & 1 != 0 {
        cos_kernel(r_hi, r_lo)
    } else {
        sin_kernel(r_hi, r_lo)
    };
    let mut out = if n & 2 != 0 { -val } else { val };
    if x.is_sign_negative() {
        out = -out;
    }
    Some(out)
}

/// Fallbacks for arguments outside the ported range: UCRT's behaviour there is
/// not reproduced, so the platform library is used instead of a guess. The
/// debug assertion is deliberate -- every argument the decoder produces is
/// well under 2*pi, so reaching it means a new call site needs the large-arg
/// reducer ported rather than a silent loss of parity.
#[inline]
pub fn cos_or_libm(x: f64) -> f64 {
    match cos(x) {
        Some(v) => v,
        None => {
            debug_assert!(false, "ucrt_math::cos: {x:?} is outside the ported range");
            platform_cos(x)
        }
    }
}

#[inline]
pub fn sin_or_libm(x: f64) -> f64 {
    match sin(x) {
        Some(v) => v,
        None => {
            debug_assert!(false, "ucrt_math::sin: {x:?} is outside the ported range");
            platform_sin(x)
        }
    }
}

/// The platform library's cosine, used only outside the ported range. On
/// Windows this is UCRT itself, so behaviour there is unchanged either way.
#[cfg(target_os = "windows")]
#[inline]
fn platform_cos(x: f64) -> f64 {
    #[link(name = "ucrtbase", kind = "raw-dylib")]
    extern "C" {
        fn cos(x: f64) -> f64;
    }
    unsafe { cos(x) }
}

#[cfg(not(target_os = "windows"))]
#[inline]
fn platform_cos(x: f64) -> f64 {
    x.cos()
}

#[cfg(target_os = "windows")]
#[inline]
fn platform_sin(x: f64) -> f64 {
    #[link(name = "ucrtbase", kind = "raw-dylib")]
    extern "C" {
        fn sin(x: f64) -> f64;
    }
    unsafe { sin(x) }
}

#[cfg(not(target_os = "windows"))]
#[inline]
fn platform_sin(x: f64) -> f64 {
    x.sin()
}

// --- C ABI shims for the vendored C++ FFT -----------------------------------
//
// `vendor/ducc0/math/unity_roots.h` builds every FFT twiddle with the platform
// libm's `cos`/`sin`, which is where the bulk of the Linux-vs-Windows FFT drift
// came from. On non-Windows targets `build.rs` defines `DUCC_UCRT_SHIM` and the
// header calls these instead, so the two platforms build identical twiddles
// from identical numbers.

#[cfg(not(target_os = "windows"))]
#[no_mangle]
pub extern "C" fn ld_ucrt_cos(x: f64) -> f64 {
    cos_or_libm(x)
}

#[cfg(not(target_os = "windows"))]
#[no_mangle]
pub extern "C" fn ld_ucrt_sin(x: f64) -> f64 {
    sin_or_libm(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every expected value below was read out of the real UCRT on Windows
    /// (`work/numprobe/src/bin/echo.rs`), not computed here, so this pins the
    /// port to the reference rather than to itself. A change of operation
    /// order fails here instead of silently moving FFT output.
    #[test]
    fn cos_matches_ucrt_bit_patterns() {
        // twiddle-grid value (exercises the medium/reduction path)
        assert_eq!(
            cos(std::f64::consts::FRAC_PI_4).unwrap().to_bits(),
            0x3fe6a09e667f3bcd
        );
        assert_eq!(cos(0.5).unwrap().to_bits(), 0x3fec1528065b7d50);
        assert_eq!(cos(3.0).unwrap().to_bits(), 0xbfefae04be85e5d2);
        assert_eq!(cos(-2.5).unwrap().to_bits(), 0xbfe9a2f7ef858b7d);
        assert_eq!(cos(6.283185307179586).unwrap().to_bits(), 0x3ff0000000000000);
        // tiny branch: 2^-27 and below return exactly 1.0
        assert_eq!(cos(1e-12).unwrap().to_bits(), 0x3ff0000000000000);
        assert_eq!(cos(1e-9).unwrap().to_bits(), 0x3ff0000000000000);
        // outside the ported range the caller must fall back
        assert!(cos(3.0e7).is_none());
        assert!(cos(f64::NAN).is_none());
    }

    #[test]
    fn sin_matches_ucrt_bit_patterns() {
        assert_eq!(
            sin(std::f64::consts::FRAC_PI_4).unwrap().to_bits(),
            0x3fe6a09e667f3bcd
        );
        assert_eq!(sin(0.5).unwrap().to_bits(), 0x3fdeaee8744b05f0);
        assert_eq!(sin(3.0).unwrap().to_bits(), 0x3fc210386db6d55b);
        assert_eq!(sin(-2.5).unwrap().to_bits(), 0xbfe326af0dcfcab0);
        // sin(2*pi) is not zero: the reduced argument keeps the low part
        assert_eq!(sin(6.283185307179586).unwrap().to_bits(), 0xbcb1a62633145c07);
        // tiny branch: sin(x) rounds to x itself
        assert_eq!(sin(1e-12).unwrap().to_bits(), 1e-12f64.to_bits());
        assert!(sin(3.0e7).is_none());
    }

    #[test]
    fn sin_is_odd_and_cos_is_even_through_the_reducer() {
        for x in [0.5, 3.0, 2.5, 6.283185307179586] {
            assert_eq!(sin(-x).unwrap().to_bits(), (-sin(x).unwrap()).to_bits());
            assert_eq!(cos(-x).unwrap().to_bits(), cos(x).unwrap().to_bits());
        }
    }
}
