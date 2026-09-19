//! Bit-exact port of UCRT's `atan2`, reconstructed from the disassembly of
//! `ucrtbase.dll` (`llvm-objdump -d --disassemble-symbols=atan2`).
//!
//! Shape of the original: `atan` is *inlined* into `atan2`, so this single
//! function contains both the argument reduction (`|y|`, `|x|` sign handling,
//! a swap so the ratio is <= 1) and the polynomial/table `atan` core. The
//! value is carried as a **pair** (two doubles that are added at the end) so
//! the final `pi/2 -` and `pi -` adjustments keep their low bits.
//!
//! Two branches:
//! - `z <= 1/16`: polynomial in `z^2` plus a residual term computed in
//!   extended precision, so the result does not lose the bits the ratio
//!   `y/x` already dropped.
//! - `z > 1/16`: `n = round(z*256)`, `atan(n/256)` from a 241-entry table
//!   split into two arrays, plus a reduced argument `r` and a small
//!   correction. The reduction scales the larger operand into `[1,2)` by
//!   `2^d` split across two factors so neither step overflows.
//!
//! UCRT calls out to helpers in several places; all of them are exception or
//! errno plumbing (the value is already in `xmm6`/`xmm10` before the call, and
//! the result registers are set after it), so they are not reproduced.
//!
//! Returns `None` outside the range this port covers -- NaN/inf operands and
//! subnormal operands -- so the caller falls back to the platform library
//! rather than silently returning a wrong value. The decoder never produces
//! either.


// On Windows the decoder calls UCRT directly and this module is referenced only
// by its own tests, so most of it looks dead there.
#![allow(dead_code)]

// ucrtbase.dll rodata.
const A_TINY: f64 = 1e-08; // 0x1800c50b8
const PI_LO: f64 = 3.178650954705639e-08; // 0x1800c50c0
const INV_256: f64 = 0.00390625; // 0x1800c50c8
const SIXTEENTH: f64 = 0.0625; // 0x1800c50d0
const C090: f64 = 0.09002981028544979; // 0x1800c50d8
const C111: f64 = 0.11110736283514526; // 0x1800c50e0
const C142: f64 = 0.1428571356180717; // 0x1800c50e8
const C199A: f64 = 0.19999918038989142; // 0x1800c50f0
const C199B: f64 = 0.19999999999393223; // 0x1800c50f8
const C333A: f64 = 0.33333333333224097; // 0x1800c5100
const C333B: f64 = 0.3333333333333317; // 0x1800c5108
const PI_HI: f64 = 3.1415926218032837; // 0x1800c5110
const SCALE_256: f64 = 256.0; // 0x1800c5118
const PI2_LO: f64 = 6.123233995736766e-17; // 0x1800c4080
const HALF: f64 = 0.5; // 0x1800d15c8
const PI2_HI: f64 = 1.5707963267948966; // 0x1800d1668
const PI: f64 = 3.141592653589793; // 0x1800d1698

/// `atan2(y, x)` with UCRT's exact rounding, or `None` if the port does not
/// cover these operands.
pub fn atan2(y: f64, x: f64) -> Option<f64> {
    let bx = x.to_bits();
    let by = y.to_bits();
    let sx = (bx >> 63) != 0;
    let sy = (by >> 63) != 0;
    let abs_x = bx & 0x7fff_ffff_ffff_ffff;
    let abs_y = by & 0x7fff_ffff_ffff_ffff;

    // NaN or inf: UCRT hands these to helpers (0x180033f9c).
    if abs_x > 0x7ff0_0000_0000_0000 || abs_y > 0x7ff0_0000_0000_0000 {
        return None;
    }
    if abs_y == 0 {
        // 0x1800320ee: y is zero. atan2(+-0, x>0) = +-0; x<0 gives +-pi.
        if sx {
            return Some(if sy { -PI } else { PI });
        }
        return Some(y);
    }
    if abs_x == 0 {
        // 0x180032101: x is zero. atan2(+y, +-0) = +pi/2, atan2(-y, +-0) = -pi/2.
        return Some(if sy { -PI2_HI } else { PI2_HI });
    }

    let mut exp_x = ((bx >> 52) & 0x7ff) as i32;
    let mut exp_y = ((by >> 52) & 0x7ff) as i32;
    // Subnormal operands take a different scaling path that is not ported.
    if exp_x == 0 || exp_y == 0 {
        return None;
    }

    // 0x18003212b: when both magnitudes are below 0.5 the original adds 1 to
    // each exponent (`bits + 0x4000000000000000`), i.e. multiplies both by 2,
    // and recomputes the exponent difference from the scaled values. The ratio
    // is unchanged but the exponent `d` used by the table branch is not.
    let (mut vx, mut vy) = (x, y);
    if exp_x < 0x3fd && exp_y < 0x3fd {
        vx = x * 2.0;
        vy = y * 2.0;
        exp_x = ((vx.to_bits() >> 52) & 0x7ff) as i32;
        exp_y = ((vy.to_bits() >> 52) & 0x7ff) as i32;
    }
    let esi = exp_y - exp_x;

    // 0x18003221d: |y| overwhelmingly larger than |x|.
    if esi > 56 {
        return Some(if sy { -PI2_HI } else { PI2_HI });
    }
    // 0x180032241: |y| overwhelmingly smaller than |x|.
    if esi < -28 {
        if !sx {
            if esi >= -1074 {
                // 0x180032275 -> 0x18003235b: |y| tiny, x > 0: the answer is
                // the ratio itself, no reduction at all.
                if esi >= -1022 {
                    return Some(vy / vx);
                }
                // 0x180032281: subnormal-scaled ratio, not ported.
                return None;
            }
            // 0x18003225b: y is smaller than 2^-1074 * |x|.
            return Some(if sy { -0.0 } else { 0.0 });
        }
    }
    // 0x180032369: very small ratio with x < 0 gives +-pi.
    if esi < -56 && sx {
        return Some(if sy { -PI } else { PI });
    }

    // 0x1800323d0: work on magnitudes, with the larger operand in `lg`.
    let mut lg = if sx { -vx } else { vx };
    let mut sm = if sy { -vy } else { vy };
    let swapped = sm > lg;
    if swapped {
        core::mem::swap(&mut lg, &mut sm);
    }
    let z = sm / lg;

    // The value is a pair of doubles summed at the end: `lo` + `hi`.
    let (mut lo, mut hi);
    if z <= SIXTEENTH {
        // 0x18003257f: polynomial branch.
        if A_TINY > z {
            lo = 0.0;
            hi = z;
        } else {
            let z2 = z * z; // xmm4
            let lg_hi = f64::from_bits(lg.to_bits() & 0xffff_ffff_0000_0000); // xmm3
            let z_hi = f64::from_bits(z.to_bits() & 0xffff_ffff_0000_0000); // xmm2
            // 0x1800325cc-0x180032622: the residual of `sm - z*lg`, computed
            // from the truncated pieces so it keeps the bits `z` dropped.
            let mut t1 = lg - lg_hi; // xmm1
            let mut acc = sm; // xmm8
            t1 *= z_hi;
            acc -= z_hi * lg_hi;
            acc -= t1;
            let mut c = C111; // xmm1
            acc -= (z - z_hi) * lg;
            acc /= lg;
            let mut t = z2 * C090;
            c -= t;
            t = C142;
            c *= z2;
            t -= c;
            let mut c2 = C199B;
            t *= z2;
            c2 -= t;
            let mut t2 = C333B;
            c2 *= z2;
            let z3 = z2 * z;
            t2 -= c2;
            t2 *= z3;
            acc -= t2;
            acc += z;
            lo = 0.0;
            hi = acc;
        }
    } else {
        // 0x18003241e: table branch.
        let n = (z * SCALE_256 + HALF) as i32; // cvttsd2si
        let idx = (n - 16) as usize;
        let t_lo = ATAN_TBL_HI[idx]; // xmm10, loaded from 0x1800c41a0
        let t_hi = ATAN_TBL_LO[idx]; // xmm7,  loaded from 0x1800c4930
        let m = (n as f64) * INV_256; // xmm6

        // 0x180032458-0x1800324ee: scale `lg` into [1,2) by 2^d, split into
        // 2^e * 2^f so neither factor overflows on its own, and scale `sm` by
        // 2^e only.
        let d = 0x3ff - (((lg.to_bits() >> 52) & 0x7ff) as i32);
        let e = d / 2; // truncates toward zero, as `(d - sign(d)) >> 1`
        let f = d - e;
        let two_e = f64::from_bits(((e + 0x3ff) as u64) << 52);
        let two_f = f64::from_bits(((f + 0x3ff) as u64) << 52);
        let big = two_e * lg * two_f; // X, in [1,2)
        let small = two_e * sm * two_f; // xmm3: sm scaled the same way as lg

        // 0x1800324fd-0x18003254d: r = (small - X*m) / (X + m*small), with the
        // product and the split of X both carrying their residual terms.
        let x_hi = f64::from_bits(big.to_bits() & 0xffff_ffff_f800_0000); // xmm2
        let mut p = big - x_hi; // xmm1
        let mut acc = small; // xmm4
        acc -= x_hi * m;
        p *= m;
        let mut msmall = m * small; // xmm6
        acc -= p;
        msmall += big;
        acc /= msmall;
        let r = acc; // xmm4

        // 0x180032551-0x18003257a: t_hi + r - r^3 * (C333A - r^2*C199A).
        let r2 = r * r; // xmm2
        let mut poly = r + t_hi; // xmm5
        let mut c = C333A; // xmm1
        c -= r2 * C199A;
        c *= r2;
        c *= r;
        poly -= c;
        lo = t_lo;
        hi = poly;
    }

    // 0x180032681: quadrant fix-ups, still on the pair.
    if swapped {
        lo = PI2_HI - lo;
        hi = PI2_LO - hi;
    }
    if sx {
        lo = PI_HI - lo;
        hi = PI_LO - hi;
    }
    let mut v = lo + hi;
    if sy {
        v = -v;
    }
    Some(v)
}

// ---------------------------------------------------------------------------
// Generated tables (ucrtbase.dll rodata).
// ---------------------------------------------------------------------------

// Generated from ucrtbase.dll rodata (see work/pe_dump.py). Do not hand-edit.
// ATAN_TBL_HI[i] / ATAN_TBL_LO[i] are UCRT's split of atan((i+16)/256):
// index i corresponds to n = i + 16 in 16..=256.
pub(crate) const ATAN_TBL_HI: [f64; 241] = [
    f64::from_bits(0x3faff55b00000000), f64::from_bits(0x3fb0f99e00000000), f64::from_bits(0x3fb1f86d00000000), f64::from_bits(0x3fb2f71900000000),
    f64::from_bits(0x3fb3f59f00000000), f64::from_bits(0x3fb4f3fd00000000), f64::from_bits(0x3fb5f23200000000), f64::from_bits(0x3fb6f03b00000000),
    f64::from_bits(0x3fb7ee1800000000), f64::from_bits(0x3fb8ebc500000000), f64::from_bits(0x3fb9e94100000000), f64::from_bits(0x3fbae68a00000000),
    f64::from_bits(0x3fbbe39e00000000), f64::from_bits(0x3fbce07c00000000), f64::from_bits(0x3fbddd2100000000), f64::from_bits(0x3fbed98c00000000),
    f64::from_bits(0x3fbfd5ba00000000), f64::from_bits(0x3fc068d500000000), f64::from_bits(0x3fc0e6ad00000000), f64::from_bits(0x3fc1646500000000),
    f64::from_bits(0x3fc1e1fa00000000), f64::from_bits(0x3fc25f6e00000000), f64::from_bits(0x3fc2dcbd00000000), f64::from_bits(0x3fc359e800000000),
    f64::from_bits(0x3fc3d6ee00000000), f64::from_bits(0x3fc453ce00000000), f64::from_bits(0x3fc4d08700000000), f64::from_bits(0x3fc54d1800000000),
    f64::from_bits(0x3fc5c98100000000), f64::from_bits(0x3fc645bf00000000), f64::from_bits(0x3fc6c1d400000000), f64::from_bits(0x3fc73dbd00000000),
    f64::from_bits(0x3fc7b97b00000000), f64::from_bits(0x3fc8350b00000000), f64::from_bits(0x3fc8b06e00000000), f64::from_bits(0x3fc92ba300000000),
    f64::from_bits(0x3fc9a6a800000000), f64::from_bits(0x3fca217e00000000), f64::from_bits(0x3fca9c2300000000), f64::from_bits(0x3fcb169600000000),
    f64::from_bits(0x3fcb90d700000000), f64::from_bits(0x3fcc0ae500000000), f64::from_bits(0x3fcc84bf00000000), f64::from_bits(0x3fccfe6500000000),
    f64::from_bits(0x3fcd77d500000000), f64::from_bits(0x3fcdf11000000000), f64::from_bits(0x3fce6a1400000000), f64::from_bits(0x3fcee2e100000000),
    f64::from_bits(0x3fcf5b7500000000), f64::from_bits(0x3fcfd3d100000000), f64::from_bits(0x3fd025fa00000000), f64::from_bits(0x3fd061ee00000000),
    f64::from_bits(0x3fd09dc500000000), f64::from_bits(0x3fd0d97e00000000), f64::from_bits(0x3fd1151a00000000), f64::from_bits(0x3fd1509700000000),
    f64::from_bits(0x3fd18bf500000000), f64::from_bits(0x3fd1c73500000000), f64::from_bits(0x3fd2025500000000), f64::from_bits(0x3fd23d5600000000),
    f64::from_bits(0x3fd2783700000000), f64::from_bits(0x3fd2b2f700000000), f64::from_bits(0x3fd2ed9800000000), f64::from_bits(0x3fd3281800000000),
    f64::from_bits(0x3fd3627700000000), f64::from_bits(0x3fd39cb400000000), f64::from_bits(0x3fd3d6d100000000), f64::from_bits(0x3fd410cb00000000),
    f64::from_bits(0x3fd44aa400000000), f64::from_bits(0x3fd4845a00000000), f64::from_bits(0x3fd4bdee00000000), f64::from_bits(0x3fd4f75f00000000),
    f64::from_bits(0x3fd530ad00000000), f64::from_bits(0x3fd569d800000000), f64::from_bits(0x3fd5a2e000000000), f64::from_bits(0x3fd5dbc300000000),
    f64::from_bits(0x3fd6148400000000), f64::from_bits(0x3fd64d1f00000000), f64::from_bits(0x3fd6859700000000), f64::from_bits(0x3fd6bdea00000000),
    f64::from_bits(0x3fd6f61900000000), f64::from_bits(0x3fd72e2200000000), f64::from_bits(0x3fd7660700000000), f64::from_bits(0x3fd79dc600000000),
    f64::from_bits(0x3fd7d56000000000), f64::from_bits(0x3fd80cd400000000), f64::from_bits(0x3fd8442200000000), f64::from_bits(0x3fd87b4b00000000),
    f64::from_bits(0x3fd8b24d00000000), f64::from_bits(0x3fd8e92900000000), f64::from_bits(0x3fd91fde00000000), f64::from_bits(0x3fd9566d00000000),
    f64::from_bits(0x3fd98cd500000000), f64::from_bits(0x3fd9c31600000000), f64::from_bits(0x3fd9f93000000000), f64::from_bits(0x3fda2f2300000000),
    f64::from_bits(0x3fda64ee00000000), f64::from_bits(0x3fda9a9200000000), f64::from_bits(0x3fdad00f00000000), f64::from_bits(0x3fdb056400000000),
    f64::from_bits(0x3fdb3a9100000000), f64::from_bits(0x3fdb6f9600000000), f64::from_bits(0x3fdba47300000000), f64::from_bits(0x3fdbd92800000000),
    f64::from_bits(0x3fdc0db400000000), f64::from_bits(0x3fdc421900000000), f64::from_bits(0x3fdc765500000000), f64::from_bits(0x3fdcaa6800000000),
    f64::from_bits(0x3fdcde5300000000), f64::from_bits(0x3fdd121500000000), f64::from_bits(0x3fdd45ae00000000), f64::from_bits(0x3fdd791f00000000),
    f64::from_bits(0x3fddac6700000000), f64::from_bits(0x3fdddf8500000000), f64::from_bits(0x3fde127b00000000), f64::from_bits(0x3fde454800000000),
    f64::from_bits(0x3fde77eb00000000), f64::from_bits(0x3fdeaa6500000000), f64::from_bits(0x3fdedcb600000000), f64::from_bits(0x3fdf0ede00000000),
    f64::from_bits(0x3fdf40dd00000000), f64::from_bits(0x3fdf72b200000000), f64::from_bits(0x3fdfa45d00000000), f64::from_bits(0x3fdfd5e000000000),
    f64::from_bits(0x3fe0039c00000000), f64::from_bits(0x3fe01c3400000000), f64::from_bits(0x3fe034b700000000), f64::from_bits(0x3fe04d2500000000),
    f64::from_bits(0x3fe0657e00000000), f64::from_bits(0x3fe07dc300000000), f64::from_bits(0x3fe095f300000000), f64::from_bits(0x3fe0ae0e00000000),
    f64::from_bits(0x3fe0c61400000000), f64::from_bits(0x3fe0de0500000000), f64::from_bits(0x3fe0f5e200000000), f64::from_bits(0x3fe10daa00000000),
    f64::from_bits(0x3fe1255d00000000), f64::from_bits(0x3fe13cfb00000000), f64::from_bits(0x3fe1548500000000), f64::from_bits(0x3fe16bfa00000000),
    f64::from_bits(0x3fe1835a00000000), f64::from_bits(0x3fe19aa500000000), f64::from_bits(0x3fe1b1dc00000000), f64::from_bits(0x3fe1c8fe00000000),
    f64::from_bits(0x3fe1e00b00000000), f64::from_bits(0x3fe1f70400000000), f64::from_bits(0x3fe20de800000000), f64::from_bits(0x3fe224b700000000),
    f64::from_bits(0x3fe23b7100000000), f64::from_bits(0x3fe2521700000000), f64::from_bits(0x3fe268a900000000), f64::from_bits(0x3fe27f2600000000),
    f64::from_bits(0x3fe2958e00000000), f64::from_bits(0x3fe2abe200000000), f64::from_bits(0x3fe2c22100000000), f64::from_bits(0x3fe2d84c00000000),
    f64::from_bits(0x3fe2ee6200000000), f64::from_bits(0x3fe3046400000000), f64::from_bits(0x3fe31a5200000000), f64::from_bits(0x3fe3302b00000000),
    f64::from_bits(0x3fe345f000000000), f64::from_bits(0x3fe35ba000000000), f64::from_bits(0x3fe3713d00000000), f64::from_bits(0x3fe386c500000000),
    f64::from_bits(0x3fe39c3900000000), f64::from_bits(0x3fe3b19800000000), f64::from_bits(0x3fe3c6e400000000), f64::from_bits(0x3fe3dc1c00000000),
    f64::from_bits(0x3fe3f13f00000000), f64::from_bits(0x3fe4064f00000000), f64::from_bits(0x3fe41b4a00000000), f64::from_bits(0x3fe4303200000000),
    f64::from_bits(0x3fe4450600000000), f64::from_bits(0x3fe459c600000000), f64::from_bits(0x3fe46e7200000000), f64::from_bits(0x3fe4830a00000000),
    f64::from_bits(0x3fe4978f00000000), f64::from_bits(0x3fe4ac0000000000), f64::from_bits(0x3fe4c05e00000000), f64::from_bits(0x3fe4d4a800000000),
    f64::from_bits(0x3fe4e8de00000000), f64::from_bits(0x3fe4fd0100000000), f64::from_bits(0x3fe5111000000000), f64::from_bits(0x3fe5250c00000000),
    f64::from_bits(0x3fe538f500000000), f64::from_bits(0x3fe54cca00000000), f64::from_bits(0x3fe5608d00000000), f64::from_bits(0x3fe5743c00000000),
    f64::from_bits(0x3fe587d800000000), f64::from_bits(0x3fe59b6000000000), f64::from_bits(0x3fe5aed600000000), f64::from_bits(0x3fe5c23900000000),
    f64::from_bits(0x3fe5d58900000000), f64::from_bits(0x3fe5e8c600000000), f64::from_bits(0x3fe5fbf000000000), f64::from_bits(0x3fe60f0800000000),
    f64::from_bits(0x3fe6220d00000000), f64::from_bits(0x3fe634ff00000000), f64::from_bits(0x3fe647de00000000), f64::from_bits(0x3fe65aab00000000),
    f64::from_bits(0x3fe66d6600000000), f64::from_bits(0x3fe6800e00000000), f64::from_bits(0x3fe692a400000000), f64::from_bits(0x3fe6a52700000000),
    f64::from_bits(0x3fe6b79800000000), f64::from_bits(0x3fe6c9f700000000), f64::from_bits(0x3fe6dc4400000000), f64::from_bits(0x3fe6ee7f00000000),
    f64::from_bits(0x3fe700a700000000), f64::from_bits(0x3fe712be00000000), f64::from_bits(0x3fe724c300000000), f64::from_bits(0x3fe736b600000000),
    f64::from_bits(0x3fe7489700000000), f64::from_bits(0x3fe75a6700000000), f64::from_bits(0x3fe76c2400000000), f64::from_bits(0x3fe77dd100000000),
    f64::from_bits(0x3fe78f6b00000000), f64::from_bits(0x3fe7a0f400000000), f64::from_bits(0x3fe7b26c00000000), f64::from_bits(0x3fe7c3d300000000),
    f64::from_bits(0x3fe7d52800000000), f64::from_bits(0x3fe7e66c00000000), f64::from_bits(0x3fe7f79e00000000), f64::from_bits(0x3fe808c000000000),
    f64::from_bits(0x3fe819d000000000), f64::from_bits(0x3fe82ad000000000), f64::from_bits(0x3fe83bbe00000000), f64::from_bits(0x3fe84c9c00000000),
    f64::from_bits(0x3fe85d6900000000), f64::from_bits(0x3fe86e2500000000), f64::from_bits(0x3fe87ed000000000), f64::from_bits(0x3fe88f6b00000000),
    f64::from_bits(0x3fe89ff500000000), f64::from_bits(0x3fe8b06f00000000), f64::from_bits(0x3fe8c0d900000000), f64::from_bits(0x3fe8d13200000000),
    f64::from_bits(0x3fe8e17a00000000), f64::from_bits(0x3fe8f1b300000000), f64::from_bits(0x3fe901db00000000), f64::from_bits(0x3fe911f300000000),
    f64::from_bits(0x3fe921fb00000000),
];
pub(crate) const ATAN_TBL_LO: [f64; 241] = [
    f64::from_bits(0x3e56e59fbd38db2c), f64::from_bits(0x3e64e3aa54dedf96), f64::from_bits(0x3e67e105ab1bda88), f64::from_bits(0x3e48c5254d013fd0),
    f64::from_bits(0x3e2cf8ab3ad62670), f64::from_bits(0x3e59dca4bec80468), f64::from_bits(0x3e53f4b5ec98a8da), f64::from_bits(0x3e6b9d49619d81fe),
    f64::from_bits(0x3e43017887460934), f64::from_bits(0x3e511e3eca0b9944), f64::from_bits(0x3e54f3f73c5a332e), f64::from_bits(0x3e5c71c8ae0e00a6),
    f64::from_bits(0x3e67cde0f86fbdc7), f64::from_bits(0x3e570f328c889c72), f64::from_bits(0x3e5c07ae9b994efe), f64::from_bits(0x3e40c8021d7b1698),
    f64::from_bits(0x3e635585edb8cb22), f64::from_bits(0x3e70842567b30e96), f64::from_bits(0x3e799e811031472e), f64::from_bits(0x3e6041821416bcee),
    f64::from_bits(0x3e7f6086e4dc96f4), f64::from_bits(0x3e471a535c5f1b58), f64::from_bits(0x3e765f743fe63ca1), f64::from_bits(0x3e7dbd733472d014),
    f64::from_bits(0x3e7d18cc4d8b0d1d), f64::from_bits(0x3e78c12553c8fb29), f64::from_bits(0x3e753b49e2e8f991), f64::from_bits(0x3e77422ae148c141),
    f64::from_bits(0x3e4e3ec269df56a8), f64::from_bits(0x3e7ff6754e7e0ac9), f64::from_bits(0x3e7131267b1b5aad), f64::from_bits(0x3e7d14fa403a94bc),
    f64::from_bits(0x3e62f396c089a3d8), f64::from_bits(0x3e7c731d78fa95bb), f64::from_bits(0x3e7c50f385177399), f64::from_bits(0x3e6f41409c6f2c20),
    f64::from_bits(0x3e7d2d90c4c39ec0), f64::from_bits(0x3e680420696f2106), f64::from_bits(0x3e4b40327943a2e8), f64::from_bits(0x3e65d35e02f3d2a2),
    f64::from_bits(0x3e64a498288117b0), f64::from_bits(0x3e635da119afb324), f64::from_bits(0x3e714e85cdb9a908), f64::from_bits(0x3e638754e5547b9a),
    f64::from_bits(0x3e7be40ae6ce3246), f64::from_bits(0x3e70c993b3bea7e7), f64::from_bits(0x3e71d2dd89ac3359), f64::from_bits(0x3e61476603332c46),
    f64::from_bits(0x3e7f25901bac55b7), f64::from_bits(0x3e7f881b7c826e28), f64::from_bits(0x3e7441996d698d20), f64::from_bits(0x3e8407ac521ea089),
    f64::from_bits(0x3e82fb0c6c4b1723), f64::from_bits(0x3e8ca135966a3e18), f64::from_bits(0x3e6b1218e4d646e4), f64::from_bits(0x3e6d4e72a350d288),
    f64::from_bits(0x3e84617e2f04c329), f64::from_bits(0x3e6096ec41e82650), f64::from_bits(0x3e79f91f25773e6e), f64::from_bits(0x3e659c0820f1d674),
    f64::from_bits(0x3e602bf7a2df1064), f64::from_bits(0x3e8fb36bfc40508f), f64::from_bits(0x3e7ea08f3f8dc892), f64::from_bits(0x3e73ed6254656a0e),
    f64::from_bits(0x3e6b83f5e5e69c58), f64::from_bits(0x3e8d6ec2af768592), f64::from_bits(0x3e6493889a226f94), f64::from_bits(0x3e85ad8fa65279ba),
    f64::from_bits(0x3e6b615784d45434), f64::from_bits(0x3e809a184368f145), f64::from_bits(0x3e761a2439b0d91c), f64::from_bits(0x3e7ce1a65e39a978),
    f64::from_bits(0x3e832a39a93b6a66), f64::from_bits(0x3e81c3699af804e7), f64::from_bits(0x3e575e0f4e44ede8), f64::from_bits(0x3e8f77ced1a7a83b),
    f64::from_bits(0x3e284e7f0cb1b500), f64::from_bits(0x3e8ec6b838b02dfe), f64::from_bits(0x3e83ebf4dfbeda87), f64::from_bits(0x3e89397aed9cb475),
    f64::from_bits(0x3e707937bc239c54), f64::from_bits(0x3e8aa754553131b6), f64::from_bits(0x3e74a05d407c45dc), f64::from_bits(0x3e8132231a206dd0),
    f64::from_bits(0x3e72d8ecfdd69c88), f64::from_bits(0x3e7a852c74218606), f64::from_bits(0x3e871bf2baeebb50), f64::from_bits(0x3e483d7db7491820),
    f64::from_bits(0x3e6ca50d92b6da14), f64::from_bits(0x3e56f5cde8530298), f64::from_bits(0x3e7f343198910740), f64::from_bits(0x3e70e8d241ccd80a),
    f64::from_bits(0x3e71535ac619e6c8), f64::from_bits(0x3e77316041c36cd2), f64::from_bits(0x3e7985a000637d8e), f64::from_bits(0x3e6f2f29858c0a68),
    f64::from_bits(0x3e8879847f96d909), f64::from_bits(0x3e8ab3d319e12e42), f64::from_bits(0x3e75088162dfc4c2), f64::from_bits(0x3e605749a1cd9d8c),
    f64::from_bits(0x3e5da65c6c6b8618), f64::from_bits(0x3e6739bf7df1ad64), f64::from_bits(0x3e6bc31252aa3340), f64::from_bits(0x3e5e528191ad3aa8),
    f64::from_bits(0x3e8929d93df19f18), f64::from_bits(0x3e5ff11eb693a080), f64::from_bits(0x3e455ae3f145a3a0), f64::from_bits(0x3e7cbcd8c6c0ca82),
    f64::from_bits(0x3e70cb04d425d304), f64::from_bits(0x3e79adfcab5be678), f64::from_bits(0x3e893d90c5662508), f64::from_bits(0x3e768489bd35ff40),
    f64::from_bits(0x3e3586ed3da2b7e0), f64::from_bits(0x3e87604d2e850eee), f64::from_bits(0x3e7ac1d12bfb53d8), f64::from_bits(0x3e39b3d468274740),
    f64::from_bits(0x3e7fc5d68d10e53c), f64::from_bits(0x3e88f9e51884becb), f64::from_bits(0x3e8a87f0869c06d1), f64::from_bits(0x3e831e7279f685fa),
    f64::from_bits(0x3e46a8282f9719b0), f64::from_bits(0x3e60d2724a8a44e0), f64::from_bits(0x3e8a60524b11ad4e), f64::from_bits(0x3e575fdf832750f0),
    f64::from_bits(0x3e8cf06902e4cd36), f64::from_bits(0x3e6e82422d4f6d10), f64::from_bits(0x3e524a091063e6c0), f64::from_bits(0x3e78a1a172dc6f38),
    f64::from_bits(0x3e929b6619f8a92d), f64::from_bits(0x3e79274d9c1b70c8), f64::from_bits(0x3e50c34b1fbb7930), f64::from_bits(0x3e6639866c20eb50),
    f64::from_bits(0x3e86d6d0f6832e9e), f64::from_bits(0x3e9af54def99f25e), f64::from_bits(0x3e916cfc52a00262), f64::from_bits(0x3e8dcc1e83569c32),
    f64::from_bits(0x3e937f7a551ed425), f64::from_bits(0x3e9f6360adc98887), f64::from_bits(0x3e92c6ec8d35a2c1), f64::from_bits(0x3e8bd44df84cb036),
    f64::from_bits(0x3e9117cf826e310e), f64::from_bits(0x3e9ca533f332cfc9), f64::from_bits(0x3e90f208509dbc2e), f64::from_bits(0x3e8cd07d93c945de),
    f64::from_bits(0x3e957bdfd67e6d72), f64::from_bits(0x3e7aab89c516c658), f64::from_bits(0x3e63e823b1a1b8a0), f64::from_bits(0x3e8307464a9d6d3c),
    f64::from_bits(0x3e9c5993cd438843), f64::from_bits(0x3e9ba2fca02ab554), f64::from_bits(0x3e801a5b6983a268), f64::from_bits(0x3e6273d1b350efc8),
    f64::from_bits(0x3e864c238c37b0c6), f64::from_bits(0x3e6aded07370a300), f64::from_bits(0x3e878091197eb47e), f64::from_bits(0x3e74b0f245e0dabc),
    f64::from_bits(0x3e9080d9794e2eaf), f64::from_bits(0x3e8d4ec242b60c76), f64::from_bits(0x3e4221d2f940caa0), f64::from_bits(0x3e7cdbc42b2bba5c),
    f64::from_bits(0x3e6cce37bb440840), f64::from_bits(0x3e96c1d999cf1dd0), f64::from_bits(0x3e5bed8a07eb0870), f64::from_bits(0x3e769ed88f490e3c),
    f64::from_bits(0x3e6cd41719b73ef0), f64::from_bits(0x3e9cbc4ac95b41b7), f64::from_bits(0x3e9238f1b890f5d7), f64::from_bits(0x3e750c4282259cc4),
    f64::from_bits(0x3e9713d2de87b3e2), f64::from_bits(0x3e81d5a7d2255276), f64::from_bits(0x3e9c0dfd48227ac1), f64::from_bits(0x3e91c964dab76753),
    f64::from_bits(0x3e86de56d5704496), f64::from_bits(0x3e84aeb71fd19968), f64::from_bits(0x3e8fbf91c57b1918), f64::from_bits(0x3e9d6bef7fbe5d9a),
    f64::from_bits(0x3e9464d3dc249066), f64::from_bits(0x3e9638e2ec4d9073), f64::from_bits(0x3e716f4a7247ea7c), f64::from_bits(0x3e31a0a740f1d440),
    f64::from_bits(0x3e86edbb0114a33c), f64::from_bits(0x3e7dbee8bf1d513c), f64::from_bits(0x3e95b8bdb0248f73), f64::from_bits(0x3e97de3d3f5eac64),
    f64::from_bits(0x3e8ee24187ae448a), f64::from_bits(0x3e9e06c591ec5192), f64::from_bits(0x3e74e3861a332738), f64::from_bits(0x3e7a9599dcc2bfe4),
    f64::from_bits(0x3e6f732fbad43468), f64::from_bits(0x3e9eb9f573b727d9), f64::from_bits(0x3e98b212a2eb9897), f64::from_bits(0x3e9384884c167215),
    f64::from_bits(0x3e90e2d363020051), f64::from_bits(0x3e92820879fbd022), f64::from_bits(0x3e9a1ab9893e4b30), f64::from_bits(0x3e82d1b817a24478),
    f64::from_bits(0x3e615d7b8ded4878), f64::from_bits(0x3e78968f9db3a5e4), f64::from_bits(0x3e971c4171fe135f), f64::from_bits(0x3e96d80f605d0d8c),
    f64::from_bits(0x3e7c91f043691590), f64::from_bits(0x3e839f8a15fce2b2), f64::from_bits(0x3e455beda9d94b80), f64::from_bits(0x3e8b12c15d60949a),
    f64::from_bits(0x3e924167b312bfe3), f64::from_bits(0x3e90ab8633070277), f64::from_bits(0x3e854554ebbc80ee), f64::from_bits(0x3e60204aef5a4bb8),
    f64::from_bits(0x3e98af08c679cf2c), f64::from_bits(0x3e90852a330ae6c8), f64::from_bits(0x3e86d3eb9ec32916), f64::from_bits(0x3e8685cb7fcbbafe),
    f64::from_bits(0x3e91f751c1e0bd95), f64::from_bits(0x3e5705b1b0f72560), f64::from_bits(0x3e9b98d8d808ca92), f64::from_bits(0x3e62ea22c75cc980),
    f64::from_bits(0x3e97aba62bca0350), f64::from_bits(0x3e9d73833442278c), f64::from_bits(0x3e95a5ca1fb18bf9), f64::from_bits(0x3e61a6092b6ecf28),
    f64::from_bits(0x3e744fd049aac104), f64::from_bits(0x3e2c114fd8df5180), f64::from_bits(0x3e95972f130feae5), f64::from_bits(0x3e7ca034a55fe198),
    f64::from_bits(0x3e96e2b149990227), f64::from_bits(0x3e7b00000294592c), f64::from_bits(0x3e98b9bdc442620e), f64::from_bits(0x3e8d94fdfabf3e4e),
    f64::from_bits(0x3e85db30b145ad9a), f64::from_bits(0x3e8e3e1eb95022b0), f64::from_bits(0x3e9d5b8b45442bd6), f64::from_bits(0x3e97a046231ecd2e),
    f64::from_bits(0x3e9feafe3ef55232), f64::from_bits(0x3e9839e7bfd78267), f64::from_bits(0x3e645cf49d6fa900), f64::from_bits(0x3e4be3132b27f380),
    f64::from_bits(0x3e9533980bb84f9f), f64::from_bits(0x3e5889e2ce3ba390), f64::from_bits(0x3e7f7778c3ad0cc8), f64::from_bits(0x3e846660cec4eba2),
    f64::from_bits(0x3e85110b4611a626),
];

#[cfg(test)]
mod tests {
    use super::atan2;

    /// Every expected value was read out of the real UCRT on Windows
    /// (`work/numprobe/src/bin/atan2pins.rs`), not computed here, so this pins
    /// the port to the reference rather than to itself. One case per branch:
    /// both polynomial and table paths, the index edges, the swap, all four
    /// quadrants, the small-magnitude x2 scaling, wide exponent gaps, and the
    /// zero shortcuts.
    #[test]
    fn atan2_matches_ucrt_bit_patterns() {
        assert_eq!(atan2(0.05, 1.0).unwrap().to_bits(), 0x3fa9942597929f27); // path A
        assert_eq!(atan2(0.001, 1.0).unwrap().to_bits(), 0x3f50624d77516e16);
        assert_eq!(atan2(1e-7, 1.0).unwrap().to_bits(), 0x3e7ad7f29abcaf2f);
        assert_eq!(atan2(1e-12, 1.0).unwrap().to_bits(), 0x3d719799812dea11); // z < 1e-8
        assert_eq!(atan2(0.1, 1.0).unwrap().to_bits(), 0x3fb983e282e2cc4d); // path B
        assert_eq!(atan2(0.5, 1.0).unwrap().to_bits(), 0x3fddac670561bb4f);
        assert_eq!(atan2(1.0, 1.0).unwrap().to_bits(), 0x3fe921fb54442d18); // index edge
        assert_eq!(
            atan2(0.7071067811865476, 0.7071067811865476)
                .unwrap()
                .to_bits(),
            0x3fe921fb54442d18
        );
        assert_eq!(atan2(2.0, 1.0).unwrap().to_bits(), 0x3ff1b6e192ebbe44); // swap
        assert_eq!(atan2(1234500.0, 3500.0).unwrap().to_bits(), 0x3ff9165e75edf9df);
        assert_eq!(atan2(1.0, -1.0).unwrap().to_bits(), 0x4002d97c7f3321d2); // quadrants
        assert_eq!(atan2(-1.0, 1.0).unwrap().to_bits(), 0xbfe921fb54442d18);
        assert_eq!(atan2(-1.0, -1.0).unwrap().to_bits(), 0xc002d97c7f3321d2);
        assert_eq!(atan2(0.01, -7.5).unwrap().to_bits(), 0x40091f404766d57a);
        assert_eq!(atan2(0.3, 0.4).unwrap().to_bits(), 0x3fe4978fa3269ee1); // both < 0.5
        assert_eq!(atan2(0.01, 0.02).unwrap().to_bits(), 0x3fddac670561bb4f);
        assert_eq!(atan2(0.49, 0.1).unwrap().to_bits(), 0x3ff5e9630a7aaa8e);
        assert_eq!(atan2(100000.0, 1.0).unwrap().to_bits(), 0x3ff921f0d7e968a8); // wide gaps
        assert_eq!(atan2(1.0, 100000.0).unwrap().to_bits(), 0x3ee4f8b588e06853);
        assert_eq!(atan2(1e-5, 100000.0).unwrap().to_bits(), 0x3ddb7cdfd9d7bdbb);
        assert_eq!(atan2(0.0, 1.0).unwrap().to_bits(), 0x0000000000000000); // zeros
        assert_eq!(atan2(0.0, -1.0).unwrap().to_bits(), 0x400921fb54442d18);
        assert_eq!(atan2(1.0, 0.0).unwrap().to_bits(), 0x3ff921fb54442d18);
        assert_eq!(atan2(-1.0, 0.0).unwrap().to_bits(), 0xbff921fb54442d18);
        // decoder-shaped: both operands large and comparable
        assert_eq!(atan2(871110.0, -1440200.0).unwrap().to_bits(), 0x4004c7e8d373a0d1);
        assert_eq!(
            atan2(-4209800.0, 3331100.0).unwrap().to_bits(),
            0xbfecd843efe72857
        );
        assert_eq!(
            atan2(10011000.0, 9999900.0).unwrap().to_bits(),
            0x3fe926869d4a54b3
        );
    }

    /// The operand classes the port deliberately does not cover must report
    /// `None`, so the caller falls back instead of returning a wrong value.
    #[test]
    fn atan2_declines_operands_it_does_not_cover() {
        assert!(atan2(f64::NAN, 1.0).is_none());
        assert!(atan2(1.0, f64::INFINITY).is_none());
        assert!(atan2(1e-320, 1.0).is_none());
    }
}
