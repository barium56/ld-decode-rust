//! Bit-exact replica of the ducc0 transform plans the decoder actually runs
//! (test-only).
//!
//! The point of this module is to answer a question the OpenCL probe could not:
//! **is ducc's FFT reproducible by any other engine, bit for bit?** The probe's
//! first kernels were Stockham radix-2 transforms, which are mathematically
//! right and numerically different from ducc -- so they could never be a
//! `--gpu` backend under this project's parity contract. The answer turns out
//! to be yes, on one condition: reproduce *ducc's own plan*, operation for
//! operation, not merely the transform.
//!
//! Everything ducc does on the decode path is plain IEEE double `+`, `-`, `*`
//! in a fixed order -- no fused multiply-add, because the vendored build is a
//! baseline SSE2 build and the 1-D entry points run the scalar pass type. So a
//! different engine that issues the same operations in the same order produces
//! the same bits, whatever the hardware. That is what this module encodes, and
//! what `gpu_probe::gpu_bitexact_probe` then runs on the device.
//!
//! The plan for the shapes the demod kernel issues (`n = 32768`):
//!
//! * `c2c(32768)` -- `cfft_multipass(1, 1, 32768)` over `factorize(32768) =
//!   [8; 5]`, i.e. five `cfftp8` passes with `(l1, ido)` = `(1, 4096)`,
//!   `(8, 512)`, `(64, 64)`, `(512, 8)`, `(4096, 1)`. Twiddles are
//!   `UnityRoots(32768)`, `rfct = 1`.
//! * `r2c(32768)` / `c2r(32768)` -- `rfftp_complexify`, because
//!   `rfftpass::make_pass` takes the complex route for `ip > 10000`. It is a
//!   `c2c(16384)` (`factorize` -> `[8, 8, 8, 8, 4]`, so four `cfftp8` passes
//!   plus one `cfftp4`) wrapped in the half-complex pre/post processing, with
//!   `UnityRoots(32768)` for that wrapper and a separate `UnityRoots(16384)`
//!   inside.
//!
//! What is *not* replicated, deliberately: nothing on these shapes reaches
//! `cfftp_vecpass` or the SIMD pass dispatch, because `simd_exists<double, 4>`
//! is false in the vendored SSE2 build and `cfft_multipass` builds its
//! sub-passes with `vectorize = false`. A 4-lane (`LD_FFT_ENGINE=avx2*`) build
//! *does* take `cfftp_vecpass` for the 16384 inner transform and decomposes by
//! lane width -- that engine is already documented as non-parity, and this
//! replica would not match it.
//!
//! Two ducc details are easy to get wrong and are called out at their sites
//! below: the buffer ping-pong (`cfftp8` with `l1 == 1` runs *in place* and
//! does not swap, everything else writes to the second buffer and does), and
//! `cfftp8`'s `ido == 1` pass, which writes its output *transposed*
//! (`ch[k + l1*m]` from `cc[b + 8*k]`).
#![allow(dead_code)]

use crate::optimized::ucrt_math::{cos_or_libm, sin_or_libm};

/// One complex value, `(re, im)`. Kept as a pair rather than a `Complex64` so
/// the operation order is visible at the call site.
pub type C = (f64, f64);

#[inline]
fn add(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}

#[inline]
fn sub(a: C, b: C) -> C {
    (a.0 - b.0, a.1 - b.1)
}

/// `Cmplx<double>::operator*` -- `(r*br - i*bi, r*bi + i*br)`.
#[inline]
fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline]
fn conj(a: C) -> C {
    (a.0, -a.1)
}

// ---------------------------------------------------------------------------
// UnityRoots
// ---------------------------------------------------------------------------

/// `ducc0::detail_unity_roots::UnityRoots<double, Cmplx<double>>`.
///
/// A two-table decomposition: `v1[i]` holds `calc(i)` and `v2[j]` holds
/// `calc(j*(mask+1))`, and `operator[]` multiplies the pair. The twiddles are
/// built with the UCRT `cos`/`sin` -- the same port the vendored header calls
/// under `DUCC_UCRT_SHIM` -- so this reproduces ducc's table on both platforms.
pub struct UnityRoots {
    n: usize,
    mask: usize,
    shift: u32,
    v1: Vec<C>,
    v2: Vec<C>,
}

/// `UnityRoots::calc`: eight-way symmetry around the unit circle, `ang =
/// 0.25*pi/n` evaluated in `double` (the vendored header's comment explains
/// why the `long double` literal had to go).
fn calc(mut x: usize, n: usize, ang: f64) -> C {
    x <<= 3;
    if x < 4 * n {
        if x < 2 * n {
            if x < n {
                return (
                    cos_or_libm(x as f64 * ang),
                    sin_or_libm(x as f64 * ang),
                );
            }
            let y = 2 * n - x;
            return (sin_or_libm(y as f64 * ang), cos_or_libm(y as f64 * ang));
        }
        x -= 2 * n;
        if x < n {
            return (
                -sin_or_libm(x as f64 * ang),
                cos_or_libm(x as f64 * ang),
            );
        }
        let y = 2 * n - x;
        return (
            -cos_or_libm(y as f64 * ang),
            sin_or_libm(y as f64 * ang),
        );
    }
    x = 8 * n - x;
    if x < 2 * n {
        if x < n {
            return (
                cos_or_libm(x as f64 * ang),
                -sin_or_libm(x as f64 * ang),
            );
        }
        let y = 2 * n - x;
        return (
            sin_or_libm(y as f64 * ang),
            -cos_or_libm(y as f64 * ang),
        );
    }
    x -= 2 * n;
    if x < n {
        return (
            -sin_or_libm(x as f64 * ang),
            -cos_or_libm(x as f64 * ang),
        );
    }
    let y = 2 * n - x;
    (
        -cos_or_libm(y as f64 * ang),
        -sin_or_libm(y as f64 * ang),
    )
}

impl UnityRoots {
    pub fn new(n: usize) -> Self {
        const PI: f64 = 3.141592653589793238462643383279502884197;
        let ang = 0.25 * PI / (n as f64);
        let nval = (n + 2) / 2;
        let mut shift = 1u32;
        while (1usize << shift) * (1usize << shift) < nval {
            shift += 1;
        }
        let mask = (1usize << shift) - 1;
        let mut v1 = Vec::with_capacity(mask + 1);
        v1.push((1.0, 0.0));
        for i in 1..=mask {
            v1.push(calc(i, n, ang));
        }
        let v2len = (nval + mask) / (mask + 1);
        let mut v2 = Vec::with_capacity(v2len);
        v2.push((1.0, 0.0));
        for i in 1..v2len {
            v2.push(calc(i * (mask + 1), n, ang));
        }
        Self {
            n,
            mask,
            shift,
            v1,
            v2,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn v1(&self) -> &[C] {
        &self.v1
    }

    pub fn v2(&self) -> &[C] {
        &self.v2
    }

    pub fn mask(&self) -> usize {
        self.mask
    }

    pub fn shift(&self) -> u32 {
        self.shift
    }

    /// `UnityRoots::operator[](idx)`; `rfct` is 1 for every plan here (the
    /// roots table is always built for the full transform length).
    pub fn get(&self, idx: usize) -> C {
        assert!(idx <= self.n, "UnityRoots::get({idx}) out of range n={}", self.n);
        let lo = idx <= self.n / 2;
        let i = if lo { idx } else { self.n - idx };
        let m = cmul(self.v1[i & self.mask], self.v2[i >> self.shift]);
        if lo {
            m
        } else {
            (m.0, -m.1)
        }
    }
}

// ---------------------------------------------------------------------------
// c2c
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    P2,
    P4,
    P8,
}

/// One radix pass, with the twiddle table ducc would have built for it.
pub struct Pass {
    pub kind: Kind,
    pub l1: usize,
    pub ido: usize,
    /// `(ip-1)*(ido-1)` twiddles, indexed `x + (i-1)*(ip-1)` for `x = j-1`.
    pub wa: Vec<C>,
}

/// `Rtx90<fwd>`. Inverse (fwd == false): `(r, i) -> (-i, r)`.
#[inline]
fn rotx90(fwd: bool, a: &mut C) {
    let t = a.0;
    if fwd {
        a.0 = a.1;
        a.1 = -t;
    } else {
        a.0 = -a.1;
        a.1 = t;
    }
}

const HSQT2: f64 = 0.707106781186547524400844362104849;

#[inline]
fn rotx45(fwd: bool, a: &mut C) {
    let t = a.0;
    if fwd {
        a.0 = HSQT2 * (a.0 + a.1);
        a.1 = HSQT2 * (a.1 - t);
    } else {
        a.0 = HSQT2 * (a.0 - a.1);
        a.1 = HSQT2 * (a.1 + t);
    }
}

#[inline]
fn rotx135(fwd: bool, a: &mut C) {
    let t = a.0;
    if fwd {
        a.0 = HSQT2 * (a.1 - a.0);
        a.1 = HSQT2 * (-t - a.1);
    } else {
        a.0 = HSQT2 * (-a.0 - a.1);
        a.1 = HSQT2 * (t - a.1);
    }
}

#[inline]
fn pminplace(a: &mut C, b: &mut C) {
    let t = *a;
    *a = add(t, *b);
    *b = sub(t, *b);
}

/// `special_mul<fwd>`: the forward form is the conjugate-twiddle product, the
/// inverse form the plain one.
#[inline]
fn special_mul(fwd: bool, v: C, w: C) -> C {
    if fwd {
        (v.0 * w.0 + v.1 * w.1, v.1 * w.0 - v.0 * w.1)
    } else {
        (v.0 * w.0 - v.1 * w.1, v.0 * w.1 + v.1 * w.0)
    }
}

/// `cfftp8::exec_`, `l1 == 1` / `ido == 1` form: no twiddles.
fn butterfly8_plain(fwd: bool, v: &[C; 8]) -> [C; 8] {
    let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7);
    a1 = add(v[1], v[5]);
    a5 = sub(v[1], v[5]);
    a3 = add(v[3], v[7]);
    a7 = sub(v[3], v[7]);
    pminplace(&mut a1, &mut a3);
    rotx90(fwd, &mut a3);
    rotx90(fwd, &mut a7);
    pminplace(&mut a5, &mut a7);
    rotx45(fwd, &mut a5);
    rotx135(fwd, &mut a7);
    a0 = add(v[0], v[4]);
    a4 = sub(v[0], v[4]);
    a2 = add(v[2], v[6]);
    a6 = sub(v[2], v[6]);
    let o0 = add(add(a0, a2), a1);
    let o4 = sub(add(a0, a2), a1);
    let o2 = add(sub(a0, a2), a3);
    let o6 = sub(sub(a0, a2), a3);
    rotx90(fwd, &mut a6);
    let o1 = add(add(a4, a6), a5);
    let o5 = sub(add(a4, a6), a5);
    let o3 = add(sub(a4, a6), a7);
    let o7 = sub(sub(a4, a6), a7);
    [o0, o1, o2, o3, o4, o5, o6, o7]
}

/// `cfftp8::exec_`, `l1 > 1 && ido > 1` form for one `i >= 1`.
fn butterfly8_twiddled(fwd: bool, v: &[C; 8], wa: &[C]) -> [C; 8] {
    let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7);
    a1 = add(v[1], v[5]);
    a5 = sub(v[1], v[5]);
    a3 = add(v[3], v[7]);
    a7 = sub(v[3], v[7]);
    rotx90(fwd, &mut a7);
    pminplace(&mut a1, &mut a3);
    rotx90(fwd, &mut a3);
    pminplace(&mut a5, &mut a7);
    rotx45(fwd, &mut a5);
    rotx135(fwd, &mut a7);
    a0 = add(v[0], v[4]);
    a4 = sub(v[0], v[4]);
    a2 = add(v[2], v[6]);
    a6 = sub(v[2], v[6]);
    pminplace(&mut a0, &mut a2);
    let o0 = add(a0, a1);
    let o4 = special_mul(fwd, sub(a0, a1), wa[3]);
    let o2 = special_mul(fwd, add(a2, a3), wa[1]);
    let o6 = special_mul(fwd, sub(a2, a3), wa[5]);
    rotx90(fwd, &mut a6);
    pminplace(&mut a4, &mut a6);
    let o1 = special_mul(fwd, add(a4, a5), wa[0]);
    let o5 = special_mul(fwd, sub(a4, a5), wa[4]);
    let o3 = special_mul(fwd, add(a6, a7), wa[2]);
    let o7 = special_mul(fwd, sub(a6, a7), wa[6]);
    [o0, o1, o2, o3, o4, o5, o6, o7]
}

/// `cfftp4::exec_`, `ido == 1` form: no twiddles.
fn butterfly4_plain(fwd: bool, v: &[C; 4]) -> [C; 4] {
    let t2 = add(v[0], v[2]);
    let t1 = sub(v[0], v[2]);
    let t3 = add(v[1], v[3]);
    let mut t4 = sub(v[1], v[3]);
    rotx90(fwd, &mut t4);
    [
        add(t2, t3),
        add(t1, t4),
        sub(t2, t3),
        sub(t1, t4),
    ]
}

/// `cfftp4::exec_`, `l1 > 1 && ido > 1` form for one `i >= 1`.
fn butterfly4_twiddled(fwd: bool, v: &[C; 4], wa: &[C]) -> [C; 4] {
    let t2 = add(v[0], v[2]);
    let t1 = sub(v[0], v[2]);
    let t3 = add(v[1], v[3]);
    let mut t4 = sub(v[1], v[3]);
    rotx90(fwd, &mut t4);
    [
        add(t2, t3),
        special_mul(fwd, add(t1, t4), wa[0]),
        special_mul(fwd, sub(t2, t3), wa[1]),
        special_mul(fwd, sub(t1, t4), wa[2]),
    ]
}

impl Pass {
    /// `rfct` is ducc's `roots->size()/N`: 1 when the pass spans the whole
    /// transform, and the block ratio for a pass inside a nested multipass
    /// (`32768/256 = 128` for the 256-point sub-transforms of a 32768 FFT).
    /// Dropping it silently builds a *different*, wrong twiddle table.
    fn new(ip: usize, l1: usize, ido: usize, roots: &UnityRoots, rfct: usize) -> Self {
        let kind = match ip {
            2 => Kind::P2,
            4 => Kind::P4,
            8 => Kind::P8,
            _ => panic!("ducc_plan: radix {ip} is not implemented"),
        };
        let mut wa = vec![(0.0, 0.0); (ip - 1) * (ido - 1)];
        for i in 1..ido {
            for j in 1..ip {
                wa[(j - 1) + (i - 1) * (ip - 1)] = roots.get(rfct * j * l1 * i);
            }
        }
        Self {
            kind,
            l1,
            ido,
            wa,
        }
    }

    fn wa_at(&self, ip: usize, x: usize, i: usize) -> C {
        self.wa[x + (i - 1) * (ip - 1)]
    }

    /// Execute this pass from `src` into `dst`. Both buffers are the full
    /// transform length; the index maps below are ducc's own.
    pub fn exec(&self, fwd: bool, src: &[C], dst: &mut [C]) {
        match self.kind {
            Kind::P8 => {
                if self.l1 == 1 {
                    // CC(a,b) = cc[a + ido*b], in place. `a == 0` is the
                    // untwiddled block (its twiddles are all 1); `a >= 1`
                    // takes `WA(x, a)`, exactly as the generic form does.
                    let ido = self.ido;
                    for a in 0..ido {
                        let mut v = [(0.0, 0.0); 8];
                        for b in 0..8 {
                            v[b] = src[a + ido * b];
                        }
                        let o = if a == 0 {
                            butterfly8_plain(fwd, &v)
                        } else {
                            butterfly8_twiddled(fwd, &v, &self.wa[(a - 1) * 7..(a - 1) * 7 + 7])
                        };
                        for m in 0..8 {
                            dst[a + ido * m] = o[m];
                        }
                    }
                } else if self.ido == 1 {
                    // CC(b,k) = cc[b + 8*k] -> CH(k,m) = ch[k + l1*m]: transposed.
                    let l1 = self.l1;
                    for k in 0..l1 {
                        let mut v = [(0.0, 0.0); 8];
                        for b in 0..8 {
                            v[b] = src[b + 8 * k];
                        }
                        let o = butterfly8_plain(fwd, &v);
                        for m in 0..8 {
                            dst[k + l1 * m] = o[m];
                        }
                    }
                } else {
                    // `CC(i,b,k) = cc[i + ido*(b + 8*k)]` in, `CH(i,k,m) =
                    // ch[i + ido*(k + l1*m)]` out: the pass transposes the
                    // radix digit `b` (stride ido) into `m` (stride l1*ido)
                    // and the transform digit `k` the other way. Using one
                    // layout for both is the trap -- it still fills every
                    // element, with the wrong ones.
                    let (l1, ido) = (self.l1, self.ido);
                    for k in 0..l1 {
                        for a in 0..ido {
                            let base_in = a + ido * 8 * k;
                            let base_out = a + ido * k;
                            let mut v = [(0.0, 0.0); 8];
                            for b in 0..8 {
                                v[b] = src[base_in + ido * b];
                            }
                            let o = if a == 0 {
                                butterfly8_plain(fwd, &v)
                            } else {
                                butterfly8_twiddled(fwd, &v, &self.wa[(a - 1) * 7..(a - 1) * 7 + 7])
                            };
                            for m in 0..8 {
                                dst[base_out + ido * l1 * m] = o[m];
                            }
                        }
                    }
                }
            }
            Kind::P4 => {
                if self.ido == 1 {
                    let l1 = self.l1;
                    for k in 0..l1 {
                        let v = [src[4 * k], src[4 * k + 1], src[4 * k + 2], src[4 * k + 3]];
                        let o = butterfly4_plain(fwd, &v);
                        for m in 0..4 {
                            dst[k + l1 * m] = o[m];
                        }
                    }
                } else {
                    let (l1, ido) = (self.l1, self.ido);
                    for k in 0..l1 {
                        for a in 0..ido {
                            let base_in = a + ido * 4 * k;
                            let base_out = a + ido * k;
                            let v = [
                                src[base_in],
                                src[base_in + ido],
                                src[base_in + ido * 2],
                                src[base_in + ido * 3],
                            ];
                            let o = if a == 0 {
                                butterfly4_plain(fwd, &v)
                            } else {
                                butterfly4_twiddled(fwd, &v, &self.wa[(a - 1) * 3..(a - 1) * 3 + 3])
                            };
                            for m in 0..4 {
                                dst[base_out + ido * l1 * m] = o[m];
                            }
                        }
                    }
                }
            }
            Kind::P2 => {
                let (l1, ido) = (self.l1, self.ido);
                if ido == 1 {
                    for b in 0..l1 {
                        let (x0, x1) = (src[2 * b], src[2 * b + 1]);
                        dst[b] = add(x0, x1);
                        dst[b + l1] = sub(x0, x1);
                    }
                } else {
                    for k in 0..l1 {
                        for a in 0..ido {
                            let base_in = a + ido * 2 * k;
                            let base_out = a + ido * k;
                            let (x0, x1) = (src[base_in], src[base_in + ido]);
                            dst[base_out] = add(x0, x1);
                            dst[base_out + ido * l1] = if a == 0 {
                                sub(x0, x1)
                            } else {
                                special_mul(fwd, sub(x0, x1), self.wa_at(2, 0, a))
                            };
                        }
                    }
                }
            }
        }
    }
}

/// `cfftpass::factorize`: 8s, then 4s, then a single 2 moved to the front, then
/// odd factors. `32768 = 8^5` and `16384 = 8^4 * 4`, so these two shapes are
/// pure radix-8 (and one radix-4) multipasses.
fn factorize_c(mut n: usize) -> Vec<usize> {
    let mut f = Vec::new();
    while n & 7 == 0 {
        f.push(8);
        n >>= 3;
    }
    while n & 3 == 0 {
        f.push(4);
        n >>= 2;
    }
    if n & 1 == 0 {
        n >>= 1;
        f.push(2);
        let last = f.len() - 1;
        f.swap(0, last);
    }
    let mut d = 3;
    while d * d <= n {
        while n % d == 0 {
            f.push(d);
            n /= d;
        }
        d += 2;
    }
    if n > 1 {
        f.push(n);
    }
    f
}

/// Run a single-level pass chain on one block, ping-ponging exactly as
/// `cfft_multipass::exec_` does for `l1 == 1 && ido == 1`.
fn chain_run(passes: &[Pass], fwd: bool, ip: usize, block: &[C]) -> Vec<C> {
    let mut a = block.to_vec();
    let mut b = vec![(0.0, 0.0); ip];
    let mut in_a = true;
    for p in passes {
        // `cfftp8` with `l1 == 1` works in place and returns its input
        // pointer, so no swap happens; every other pass writes to the second
        // buffer and swaps. Getting this wrong yields a permutation of the
        // right values, never the wrong bits of the right answer, which is why
        // it had to be traced rather than guessed.
        let in_place = p.kind == Kind::P8 && p.l1 == 1;
        if in_place {
            let tmp = if in_a { a.clone() } else { b.clone() };
            if in_a {
                p.exec(fwd, &tmp, &mut a);
            } else {
                p.exec(fwd, &tmp, &mut b);
            }
        } else if in_a {
            p.exec(fwd, &a, &mut b);
            in_a = false;
        } else {
            p.exec(fwd, &b, &mut a);
            in_a = true;
        }
    }
    if in_a {
        a
    } else {
        b
    }
}

/// `util1d::prime_factors`.
fn prime_factors(mut n: usize) -> Vec<usize> {
    let mut f = Vec::new();
    while n & 1 == 0 {
        n >>= 1;
        f.push(2);
    }
    let mut d = 3;
    while d * d <= n {
        while n % d == 0 {
            f.push(d);
            n /= d;
        }
        d += 2;
    }
    if n > 1 {
        f.push(n);
    }
    f
}

/// One node of ducc's plan tree.
///
/// `cfft_multipass` is not one algorithm: for `ip <= 10000` it is a straight
/// chain of radix passes over `factorize(ip)`, and above that it splits the
/// prime factors into two balanced "packets" and recurses -- the four-step FFT.
/// The decoder's 32768-point transforms always take the second route
/// (`prime_factors(32768)` is fifteen 2s -> packets 256 and 128), so a replica
/// that only models the chain disagrees at the 1e-11 level and looks like a
/// twiddle bug.
pub struct Node {
    pub ip: usize,
    pub l1: usize,
    pub ido: usize,
    /// `roots->size()/N` for the outer twiddle of this node.
    pub rfct: usize,
    pub body: Body,
}

pub enum Body {
    /// A chain of radix passes over one block of `ip` points.
    Chain(Vec<Pass>),
    /// Two packets: `a` then `b`, each an independent transform of its own
    /// length, with `b` writing transposed into the destination.
    Packets(Box<Node>, Box<Node>),
}

fn build_node(ip: usize, l1: usize, ido: usize, roots: &UnityRoots) -> Node {
    // `cfft_multipass`: `lim = 10000`.
    // Two different ratios, and confusing them is silent until the values come
    // out wrong: a *pass* spans `ip` points (`rfct = roots->size()/ip`), while
    // the *node* applies its outer twiddles across `ip*l1*ido` points.
    let node_rfct = roots.len() / (ip * l1 * ido);
    if ip <= 10000 {
        let pass_rfct = roots.len() / ip;
        let mut passes = Vec::new();
        let mut l1l = 1usize;
        for f in factorize_c(ip) {
            passes.push(Pass::new(f, l1l, ip / (f * l1l), roots, pass_rfct));
            l1l *= f;
        }
        return Node {
            ip,
            l1,
            ido,
            rfct: node_rfct,
            body: Body::Chain(passes),
        };
    }
    let mut f = prime_factors(ip);
    f.sort_unstable_by(|a, b| b.cmp(a));
    let (mut p0, mut p1) = (1usize, 1usize);
    for x in f {
        if p0 > p1 {
            p1 *= x;
        } else {
            p0 *= x;
        }
    }
    Node {
        ip,
        l1,
        ido,
        rfct: node_rfct,
        body: Body::Packets(
            Box::new(build_node(p0, 1, ip / p0, roots)),
            Box::new(build_node(p1, p0, ip / (p1 * p0), roots)),
        ),
    }
}

impl Node {
    /// One `ip`-point transform of this node, ignoring the `l1`/`ido` role it
    /// plays in its parent.
    ///
    /// This is the distinction that has to be right: a node's stored
    /// `l1`/`ido` describe how its *parent* gathers blocks, not its own
    /// internal layout, so a gathered block is transformed as a standalone
    /// unit of length `ip` (which for a packet node means recursing with
    /// `l1 == ido == 1`, not re-applying the parent's gather).
    fn exec_unit(&self, fwd: bool, roots: &UnityRoots, src: &[C], dst: &mut [C]) {
        match &self.body {
            Body::Chain(passes) => {
                let res = chain_run(passes, fwd, self.ip, &src[..self.ip]);
                dst[..self.ip].copy_from_slice(&res);
            }
            Body::Packets(a, b) => {
                let mut mid = vec![(0.0, 0.0); self.ip];
                a.exec(fwd, roots, src, &mut mid);
                b.exec(fwd, roots, &mid, dst);
            }
        }
    }

    /// Execute this node over `src` (length `ip*l1*ido`) into `dst`.
    ///
    /// The three branches are `cfft_multipass::exec_`'s: a plain chain at the
    /// top level, a stride-`ido` gather per `l1 == 1` block with the outer
    /// twiddles applied after it, and a contiguous-block transpose for
    /// `ido == 1`. `l1 > 1 && ido > 1` is `MR_fail` in ducc.
    fn exec(&self, fwd: bool, roots: &UnityRoots, src: &[C], dst: &mut [C]) {
        if self.l1 == 1 && self.ido == 1 {
            self.exec_unit(fwd, roots, src, dst);
            return;
        }
        if self.l1 == 1 {
            let (ip, ido, rfct) = (self.ip, self.ido, self.rfct);
            let mut v = vec![(0.0, 0.0); ip];
            let mut res = vec![(0.0, 0.0); ip];
            for i in 0..ido {
                for (m, slot) in v.iter_mut().enumerate() {
                    *slot = src[i + ido * m];
                }
                self.exec_unit(fwd, roots, &v, &mut res);
                // `if (i != 0)` in ducc, then every output but the first
                // carries `roots[rfct*m*i]`; `m == 0` is copied untouched.
                for m in 0..ip {
                    dst[i + ido * m] = if m == 0 {
                        res[m]
                    } else {
                        special_mul(fwd, res[m], roots.get(rfct * m * i))
                    };
                }
            }
            return;
        }
        if self.ido == 1 {
            let (ip, l1) = (self.ip, self.l1);
            let mut block = vec![(0.0, 0.0); ip];
            let mut res = vec![(0.0, 0.0); ip];
            for n in 0..l1 {
                block.copy_from_slice(&src[n * ip..n * ip + ip]);
                self.exec_unit(fwd, roots, &block, &mut res);
                for m in 0..ip {
                    dst[n + m * l1] = res[m];
                }
            }
            return;
        }
        panic!("ducc_plan: l1 > 1 && ido > 1 is unreachable in ducc");
    }
}

/// A complex-to-complex plan: ducc's pass tree for one length.
pub struct C2cPlan {
    pub n: usize,
    pub root: Node,
    roots: UnityRoots,
}

impl C2cPlan {
    pub fn new(n: usize) -> Self {
        let roots = UnityRoots::new(n);
        let root = build_node(n, 1, 1, &roots);
        Self { n, root, roots }
    }

    pub fn roots(&self) -> &UnityRoots {
        &self.roots
    }

    /// The top-level pass chain, when the plan is a single-level one (the sizes
    /// the small-n tests use). Empty for a packet plan.
    pub fn passes(&self) -> &[Pass] {
        match &self.root.body {
            Body::Chain(passes) => passes,
            Body::Packets(..) => &[],
        }
    }

    /// Run the plan. `fwd == false` is ducc's `BACKWARD` (the inverse), which
    /// is what the demod kernel's complex inverse transforms use.
    pub fn execute(&self, fwd: bool, input: &[C]) -> Vec<C> {
        assert_eq!(input.len(), self.n);
        let mut out = vec![(0.0, 0.0); self.n];
        self.root.exec(fwd, &self.roots, input, &mut out);
        out
    }

    /// `pocketfft_c::exec`'s tail: `res[i] *= fct`, one multiply per component.
    pub fn execute_scaled(&self, fwd: bool, input: &[C], fct: f64) -> Vec<C> {
        let mut out = self.execute(fwd, input);
        if fct != 1.0 {
            for c in out.iter_mut() {
                c.0 *= fct;
                c.1 *= fct;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// r2c / c2r -- `rfftp_complexify`
// ---------------------------------------------------------------------------

/// `rfftp_complexify::exec_<true>`: an `n/2` complex forward transform of the
/// real input reinterpreted as complex, then the half-complex unpacking.
pub fn r2c(n: usize, input: &[f64]) -> Vec<C> {
    assert_eq!(input.len(), n);
    assert_eq!(n % 2, 0);
    let roots = UnityRoots::new(n);
    let inner = C2cPlan::new(n / 2);
    let ccc: Vec<C> = (0..n / 2).map(|i| (input[2 * i], input[2 * i + 1])).collect();
    let res = inner.execute(true, &ccc);

    let mut out = vec![0.0f64; n];
    out[0] = res[0].0 + res[0].1;
    let mut i = 1usize;
    let mut xi = n / 2 - 1;
    while i <= xi {
        let (ri, rxi) = (res[i], res[xi]);
        // `xe = res[i] + res[xi].conj()`.
        let xe = (ri.0 + rxi.0, ri.1 - rxi.1);
        // `xo = (res[i].i + res[xi].i, res[xi].r - res[i].r) * roots[i].conj()`.
        let xo = cmul((ri.1 + rxi.1, rxi.0 - ri.0), conj(roots.get(i)));
        out[2 * i - 1] = 0.5 * (xe.0 + xo.0);
        out[2 * i] = 0.5 * (xe.1 + xo.1);
        out[2 * xi - 1] = 0.5 * (xe.0 - xo.0);
        out[2 * xi] = 0.5 * (xo.1 - xe.1);
        i += 1;
        xi -= 1;
    }
    out[n - 1] = res[0].0 - res[0].1;

    // general_r2c's output packing: `(r[0], 0)`, then `(r[2k-1], r[2k])`, then
    // the Nyquist `(r[n-1], 0)`.
    let mut sp = Vec::with_capacity(n / 2 + 1);
    sp.push((out[0], 0.0));
    let mut i = 1usize;
    while i < n - 1 {
        sp.push((out[i], out[i + 1]));
        i += 2;
    }
    if i < n {
        sp.push((out[i], 0.0));
    }
    sp
}

/// `rfftp_complexify::exec_<false>`: the half-complex packing, an `n/2`
/// complex inverse transform, then `fct = 1/n` on the way out.
pub fn c2r(n: usize, spectrum: &[C]) -> Vec<f64> {
    assert_eq!(n % 2, 0);
    let roots = UnityRoots::new(n);
    let inner = C2cPlan::new(n / 2);

    // general_c2r's input packing: the Nyquist imaginary part is dropped.
    let mut pack = vec![0.0f64; n];
    pack[0] = spectrum[0].0;
    let mut i = 1usize;
    let mut ii = 1usize;
    while i < n - 1 {
        pack[i] = spectrum[ii].0;
        pack[i + 1] = spectrum[ii].1;
        i += 2;
        ii += 1;
    }
    if i < n {
        pack[i] = spectrum[ii].0;
    }

    let mut cch = vec![(0.0, 0.0); n / 2];
    cch[0] = (pack[0] + pack[n - 1], pack[0] - pack[n - 1]);
    let mut i = 1usize;
    let mut xi = n / 2 - 1;
    while i <= xi {
        let t1 = (pack[2 * i - 1], pack[2 * i]);
        let t2 = (pack[2 * xi - 1], -pack[2 * xi]);
        let xe = add(t1, t2);
        let xo = cmul(sub(t1, t2), roots.get(i));
        cch[i] = (xe.0 - xo.1, xe.1 + xo.0);
        cch[xi] = (xe.0 + xo.1, -xe.1 + xo.0);
        i += 1;
        xi -= 1;
    }

    let res = inner.execute(false, &cch);
    let fct = 1.0 / (n as f64);
    let mut out = Vec::with_capacity(n);
    for c in &res {
        out.push(c.0 * fct);
        out.push(c.1 * fct);
    }
    out
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi_ducc;
    use rustfft::num_complex::Complex64;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed | 1;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    fn to_c(v: &[Complex64]) -> Vec<C> {
        v.iter().map(|c| (c.re, c.im)).collect()
    }

    fn to_complex(v: &[C]) -> Vec<Complex64> {
        v.iter().map(|c| Complex64::new(c.0, c.1)).collect()
    }

    /// Count differing bit patterns, so a failure reports how far off it is
    /// rather than just "no".
    fn report(name: &str, want: &[f64], got: &[f64]) -> usize {
        assert_eq!(want.len(), got.len(), "{name}: length");
        let mut bad = 0;
        let mut worst = 0i64;
        for (a, b) in want.iter().zip(got) {
            if a.to_bits() != b.to_bits() {
                bad += 1;
                let d = (a.to_bits() as i64).wrapping_sub(b.to_bits() as i64).abs();
                worst = worst.max(d);
            }
        }
        if bad > 0 {
            eprintln!("{name}: {bad}/{} differ, worst {worst} ulp", want.len());
        }
        bad
    }

    fn flatten(v: &[C]) -> Vec<f64> {
        v.iter().flat_map(|c| [c.0, c.1]).collect()
    }

    #[test]
    fn debug_n32() {
        for (label, make) in [
            ("impulse", 0usize),
            ("dc", 1),
            ("ramp", 2),
        ] {
            let n = 32usize;
            let plan = C2cPlan::new(n);
            let input: Vec<Complex64> = (0..n)
                .map(|i| match make {
                    0 => Complex64::new(if i == 0 { 1.0 } else { 0.0 }, 0.0),
                    1 => Complex64::new(1.0, 0.0),
                    _ => Complex64::new(i as f64, 0.0),
                })
                .collect();
            let want = ffi_ducc::ifft(&input);
            let got = plan.execute_scaled(false, &to_c(&input), 1.0 / n as f64);
            eprintln!("-- {label}");
            for i in 0..8 {
                eprintln!(
                    "  [{i}] want {:+.17e} {:+.17e}   got {:+.17e} {:+.17e}",
                    want[i].re, want[i].im, got[i].0, got[i].1
                );
            }
        }
    }

    #[test]
    fn debug_packet_structure() {
        let n = 16384usize;
        let plan = C2cPlan::new(n);
        let cases: [(&str, fn(usize, usize) -> C); 3] = [
            ("dc", |_, _| (1.0, 0.0)),
            ("impulse", |i, _| if i == 0 { (1.0, 0.0) } else { (0.0, 0.0) }),
            ("bin1", |i, n| {
                let a = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                (a.cos(), a.sin())
            }),
        ];
        for (label, f) in cases {
            let input: Vec<C> = (0..n).map(|i| f(i, n)).collect();
            let want = plan_want(&input);
            let got = plan.execute_scaled(false, &input, 1.0 / n as f64);
            let mut bad = 0;
            let mut first = None;
            for i in 0..n {
                if want[i].0.to_bits() != got[i].0.to_bits()
                    || want[i].1.to_bits() != got[i].1.to_bits()
                {
                    bad += 1;
                    if first.is_none() {
                        first = Some(i);
                    }
                }
            }
            eprintln!("{label}: {bad}/{n} bad, first {first:?}");
            if let Some(i) = first {
                for k in i..(i + 4).min(n) {
                    eprintln!(
                        "  [{k}] want {:+.6e}{:+.6e}i got {:+.6e}{:+.6e}i",
                        want[k].0, want[k].1, got[k].0, got[k].1
                    );
                }
            }
        }
    }

    fn plan_want(input: &[C]) -> Vec<C> {
        let c: Vec<Complex64> = input.iter().map(|v| Complex64::new(v.0, v.1)).collect();
        ffi_ducc::ifft(&c)
            .iter()
            .map(|z| (z.re, z.im))
            .collect()
    }

    /// Small sizes, to isolate the butterfly and the twiddle table from the
    /// multipass bookkeeping.
    #[test]
    fn replica_matches_ducc_small_n() {
        for n in [8usize, 16, 32, 64, 128, 4096] {
            let plan = C2cPlan::new(n);
            let mut r = lcg(n as u64);
            let input: Vec<Complex64> = (0..n)
                .map(|_| Complex64::new(r() * 8.0, r() * 8.0))
                .collect();
            let want = flatten(&to_c(&ffi_ducc::ifft(&input)));
            let got = flatten(&plan.execute_scaled(false, &to_c(&input), 1.0 / n as f64));
            let bad = report(&format!("c2c-{n}"), &want, &got);
            assert_eq!(
                bad,
                0,
                "n={n} passes {:?}",
                plan.passes()
                    .iter()
                    .map(|p| (p.kind, p.l1, p.ido))
                    .collect::<Vec<_>>()
            );
        }
    }

    /// The gate: the replica must reproduce ducc's `c2c(BACKWARD, 1/n)` bit for
    /// bit, on data with the dynamic range the decoder feeds it.
    #[test]
    fn replica_matches_ducc_c2c_32768() {
        let n = 32768usize;
        let plan = C2cPlan::new(n);
        match &plan.root.body {
            Body::Packets(a, b) => {
                assert_eq!((plan.root.ip, plan.root.l1, plan.root.ido), (32768, 1, 1));
                let mut sizes = vec![a.ip, b.ip];
                sizes.sort_unstable();
                assert_eq!(sizes, vec![128, 256], "packets of 32768 are 256 and 128");
                assert!(matches!(a.body, Body::Chain(_)) && matches!(b.body, Body::Chain(_)));
            }
            Body::Chain(_) => panic!("32768 must be a packet plan, not a chain"),
        }
        let mut ok = 0;
        for seed in [1u64, 7, 12345] {
            let mut r = lcg(seed);
            let input: Vec<Complex64> = (0..n)
                .map(|_| Complex64::new(r() * 4096.0, r() * 64.0))
                .collect();
            let want = flatten(&to_c(&ffi_ducc::ifft(&input)));
            let got = flatten(&plan.execute_scaled(false, &to_c(&input), 1.0 / n as f64));
            assert_eq!(
                report(&format!("c2c-32768 seed {seed}"), &want, &got),
                0,
                "replica is not bit-exact at seed {seed}"
            );
            ok += 1;
        }
        eprintln!("c2c(32768): {ok}/3 seeds bit-identical to ducc");
    }

    /// Same for the 16384 inner transform the real path uses.
    #[test]
    fn replica_matches_ducc_c2c_16384() {
        let n = 16384usize;
        let plan = C2cPlan::new(n);
        match &plan.root.body {
            Body::Packets(a, b) => {
                assert_eq!(plan.root.ip, 16384);
                for child in [a, b] {
                    assert_eq!(child.ip, 128);
                    match &child.body {
                        Body::Chain(passes) => {
                            let kinds: Vec<Kind> = passes.iter().map(|p| p.kind).collect();
                            assert_eq!(
                                kinds,
                                vec![Kind::P2, Kind::P8, Kind::P8],
                                "a 128-point packet is radix 2 then two radix-8 passes"
                            );
                        }
                        Body::Packets(..) => panic!("packets of 16384 are single-level"),
                    }
                }
            }
            Body::Chain(_) => panic!("16384 must be a packet plan"),
        }
        let mut r = lcg(99);
        let input: Vec<Complex64> = (0..n)
            .map(|_| Complex64::new(r() * 4096.0, r() * 64.0))
            .collect();
        let want = flatten(&to_c(&ffi_ducc::ifft(&input)));
        let got = flatten(&plan.execute_scaled(false, &to_c(&input), 1.0 / n as f64));
        assert_eq!(report("c2c-16384", &want, &got), 0);
    }

    /// Real forward transform, against `ffi_ducc::rfft`.
    #[test]
    fn replica_matches_ducc_rfft_32768() {
        let n = 32768usize;
        let mut r = lcg(4242);
        let input: Vec<f64> = (0..n).map(|_| r() * 8192.0).collect();
        let want = flatten(&to_c(&ffi_ducc::rfft(&input)));
        let got = flatten(&r2c(n, &input));
        assert_eq!(report("rfft-32768", &want, &got), 0);
    }

    /// Real inverse, against `ffi_ducc::irfft`.
    #[test]
    fn replica_matches_ducc_irfft_32768() {
        let n = 32768usize;
        let mut r = lcg(777);
        let spectrum: Vec<Complex64> = (0..n / 2 + 1)
            .map(|_| Complex64::new(r() * 1024.0, r() * 1024.0))
            .collect();
        let want = ffi_ducc::irfft(&spectrum, n);
        let got = c2r(n, &to_c(&spectrum));
        assert_eq!(report("irfft-32768", &want, &got), 0);
    }

    /// A round trip through the replica alone: this one needs no ducc, so it
    /// distinguishes "the pack/unpack is self-consistent" from "it matches".
    ///
    /// `c2r(r2c(x))` is the only meaningful round trip: a half spectrum carries
    /// no mirror relation *within* its own bins (bin `k`'s conjugate mirror is
    /// bin `n-k`, which is not stored), so writing one in destroys the input
    /// rather than symmetrizing it.
    #[test]
    fn replica_round_trip_is_identity() {
        let n = 32768usize;
        let mut r = lcg(31337);
        let input: Vec<f64> = (0..n).map(|_| r() * 4096.0).collect();
        let back = c2r(n, &r2c(n, &input));
        let mut worst = 0.0f64;
        for (a, b) in input.iter().zip(&back) {
            worst = worst.max((a - b).abs());
        }
        eprintln!("round trip max abs error {worst:.3e}");
        assert!(worst < 1e-7, "round trip error {worst}");
    }
}
