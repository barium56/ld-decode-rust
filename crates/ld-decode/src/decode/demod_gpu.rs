//! OpenCL demodulator: [`demod_block_cpu`](super::demodblock::demod_block_cpu)
//! for a whole batch of blocks at once, in single or double precision (the
//! kernels are built with `REAL` = `float` or `double`).
//!
//! Same signal path and the same filters as the CPU kernel (see the header of
//! `demodblock.rs`); the FFTs are a radix-8 Stockham transform with twiddles
//! tabulated in double precision. Its FFTs are not the reference's, so it is
//! not bit-identical by construction; in double precision the differences
//! stay far below the f32/16-bit roundings of the outputs (a whole test
//! capture decoded byte-identical to the CPU). `LD_GPU_CHECK` compares the
//! two kernels block by block.
//!
//! Single precision: the FM discriminator's output is a frequency near 8 MHz that
//! moves ~7 kHz per IRE. Carried as is in single precision, that constant
//! would leave only ~1 Hz of resolution and the post-demod FFTs would round
//! at that scale. So the discriminator writes `f - c` (c = the 50 IRE
//! frequency), the clamp and the post-demod filters run on that, and
//! `c * H(0)` is added back to each filtered channel: the filters are linear,
//! so it is the same quantity, now resolved to ~0.1 Hz.

use std::sync::Mutex;

use anyhow::Result;
use rustfft::num_complex::Complex64;

use super::demodblock::{audio_stage1, BlockDecode, DemodSpecRef, VideoChannels};
use crate::opencl::{Arg, Buffer, Gpu, Kernel, Pinned, Queue};

/// Blocks per GPU submission (bounds the device memory: ~10 buffers of
/// `MAX_BATCH * blocklen` complex values).
const MAX_BATCH: usize = 64;

const SOURCE: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline REAL2 cmul(REAL2 a, REAL2 b) { return (REAL2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
inline REAL2 cmulf(REAL2 a, REAL2 b) { return cmul(a, b); }

// Real samples -> complex rows.
__kernel void load_real(__global const float* in, __global REAL2* out) {
    int i = get_global_id(0), b = get_global_id(1);
    if (i >= N) return;
    out[b * N + i] = (REAL2)((REAL)in[b * N + i], (REAL)0);
}

// The same from 16-bit codes: sample = code + off (exact for integers).
__kernel void load_u16(__global const ushort* in, const float off, __global REAL2* out) {
    int i = get_global_id(0), b = get_global_id(1);
    if (i >= N) return;
    out[b * N + i] = (REAL2)((REAL)in[b * N + i] + (REAL)off, (REAL)0);
}

// v * exp(s * i * pi/4 * k) for k = 1, 2, 3 (s = -1 forward, +1 inverse).
#define R2 ((REAL)0.70710678118654752)
inline REAL2 w8_1(REAL2 v, float s) { return (REAL2)(R2 * (v.x - s * v.y), R2 * (v.y + s * v.x)); }
inline REAL2 w8_2(REAL2 v, float s) { return (REAL2)(-s * v.y, s * v.x); }
inline REAL2 w8_3(REAL2 v, float s) { return (REAL2)(R2 * (-v.x - s * v.y), R2 * (-v.y + s * v.x)); }

// One radix-8 Stockham pass (Govindaraju et al.): sub-transform length Ns.
// tw[m] = exp(-2 pi i m / N), tabulated in double precision.
__kernel void fft_pass(__global const REAL2* in, __global REAL2* out, __global const REAL2* tw,
                       const int Ns, const float s) {
    int j = get_global_id(0), b = get_global_id(1);
    if (j >= N / 8) return;
    __global const REAL2* x = in + (size_t)b * N;
    __global REAL2* y = out + (size_t)b * N;
    int k = j % Ns;
    REAL2 v[8];
    for (int r = 0; r < 8; r++) v[r] = x[j + r * (N / 8)];
    if (Ns > 1) {
        int step = N / (Ns * 8);
        for (int r = 1; r < 8; r++) {
            REAL2 t = tw[(k * r * step) & (N - 1)];
            if (s > 0.0f) t.y = -t.y;
            v[r] = cmul(v[r], t);
        }
    }
    // DFT-8: radix-2 split into two DFT-4.
    REAL2 a0 = v[0] + v[4], a4 = v[0] - v[4];
    REAL2 a1 = v[1] + v[5], a5 = w8_1(v[1] - v[5], s);
    REAL2 a2 = v[2] + v[6], a6 = w8_2(v[2] - v[6], s);
    REAL2 a3 = v[3] + v[7], a7 = w8_3(v[3] - v[7], s);
    REAL2 c0 = a0 + a2, c2 = a0 - a2, c1 = a1 + a3, c3 = w8_2(a1 - a3, s);
    REAL2 d0 = a4 + a6, d2 = a4 - a6, d1 = a5 + a7, d3 = w8_2(a5 - a7, s);
    REAL2 o[8];
    o[0] = c0 + c1; o[4] = c0 - c1; o[2] = c2 + c3; o[6] = c2 - c3;
    o[1] = d0 + d1; o[5] = d0 - d1; o[3] = d2 + d3; o[7] = d2 - d3;
    int idx = (j / Ns) * Ns * 8 + k;
    for (int r = 0; r < 8; r++) y[idx + r * Ns] = o[r];
}

// Hermitian full spectrum of x[:N/2+1] * f (what irfft inverts: the imaginary
// parts of the DC and Nyquist bins are dropped).
inline REAL2 herm(__global const REAL2* x, __global const REAL2* f, int k) {
    if (k == 0 || k == N / 2) return (REAL2)(cmul(x[k], f[k]).x, (REAL)0);
    if (k < N / 2) return cmul(x[k], f[k]);
    REAL2 v = cmul(x[N - k], f[N - k]);
    return (REAL2)(v.x, -v.y);
}

// The block spectrum's three products: RF video (* MTF power), EFM, RF highpass.
__kernel void rf_products(__global const REAL2* X, __global const REAL2* rfvideo, __global const REAL2* mtfpow,
                          const int use_pow, __global const REAL2* fefm, __global const REAL2* frfhpf,
                          __global REAL2* H, __global REAL2* E, __global REAL2* R) {
    int k = get_global_id(0), b = get_global_id(1);
    if (k >= N) return;
    __global const REAL2* x = X + (size_t)b * N;
    size_t o = (size_t)b * N + k;
    REAL2 h = cmul(x[k], rfvideo[k]);
    if (use_pow) h = cmul(h, mtfpow[k]);
    H[o] = h;
    E[o] = cmul(x[k], fefm[k]);
    R[o] = herm(x, frfhpf, k);
}

// FM discriminator, minus c; `raw` keeps the unclamped value, `D` the clamped
// one as a complex row for the post-demod FFT.
__kernel void discriminate(__global const REAL2* h, __global REAL* raw, __global REAL2* D,
                           const double scale, const double c, const REAL lo, const REAL hi) {
    int n = get_global_id(0), b = get_global_id(1);
    if (n >= N) return;
    size_t o = (size_t)b * N + n;
    REAL v;
    if (n == 0) {
        v = (REAL)(-c);
    } else {
        REAL2 z = h[o], w = h[o - 1];
        REAL re = z.x * w.x + z.y * w.y, im = z.y * w.x - z.x * w.y;
        double d = (double)atan2(im, re);
        if (d < 0.0) d += 6.283185307179586;
        v = (REAL)(d * scale - c);
    }
    raw[o] = v;
    D[o] = (REAL2)(clamp(v, lo, hi), (REAL)0);
}

// The three post-demod filters as Hermitian spectra.
__kernel void video_products(__global const REAL2* Y, __global const REAL2* f0, __global const REAL2* f1,
                             __global const REAL2* f2, __global REAL2* V0, __global REAL2* V1, __global REAL2* V2) {
    int k = get_global_id(0), b = get_global_id(1);
    if (k >= N) return;
    __global const REAL2* y = Y + (size_t)b * N;
    size_t o = (size_t)b * N + k;
    V0[o] = herm(y, f0, k);
    V1[o] = herm(y, f1, k);
    V2[o] = herm(y, f2, k);
}

// Cut and pack every output channel of the batch.
__kernel void extract(__global const REAL2* V0, __global const REAL2* V1, __global const REAL2* V2,
                      __global const REAL* raw, __global const REAL2* R, __global const REAL2* E,
                      const int cstart, const int clen, const int rstart, const int rlen,
                      const REAL add0, const REAL add1, const REAL add2, const REAL c,
                      __global float* o_demod, __global float* o_05, __global float* o_burst,
                      __global float* o_raw, __global float* o_rfhpf, __global short* o_efm) {
    int j = get_global_id(0), b = get_global_id(1);
    const REAL inv = (REAL)1 / (REAL)N;
    size_t base = (size_t)b * N;
    if (j < clen) {
        size_t i = base + cstart + j, oo = (size_t)b * clen + j;
        o_demod[oo] = (float)(V0[i].x * inv + add0);
        o_05[oo] = (float)(V1[i].x * inv + add1);
        o_burst[oo] = (float)(V2[i].x * inv + add2);
        o_raw[oo] = (float)(raw[i] + c);
        o_efm[oo] = convert_short_rtz(clamp(E[i].x * inv, (REAL)-32768, (REAL)32767));
    }
    if (j < rlen) {
        o_rfhpf[(size_t)b * rlen + j] = (float)(R[base + rstart + j].x * inv);
    }
}

// Spectrum bins the CPU's analog-audio stage reads: per block, `n` bins from
// each of the four starts in `starts`.
__kernel void gather_bins(__global const REAL2* X, __global const int* starts, const int n, __global REAL2* out) {
    int j = get_global_id(0), b = get_global_id(1);
    if (j >= 4 * n) return;
    int run = j / n, i = j % n;
    out[(size_t)b * 4 * n + j] = X[(size_t)b * N + min(starts[run] + i, N - 1)];
}
"#;

/// Per-decoder GPU state: its queue, kernels and buffers.
struct State {
    q: Queue,
    k_load: Kernel,
    k_load16: Kernel,
    k_fft: Kernel,
    k_rf: Kernel,
    k_disc: Kernel,
    k_video: Kernel,
    k_extract: Kernel,
    k_bins: Kernel,
    tw: Buffer,
    rfvideo: Buffer,
    mtfpow: Buffer,
    fefm: Buffer,
    frfhpf: Buffer,
    fvideo: [Buffer; 3],
    starts: Buffer,
    /// Work rows: input/scratch, spectrum, H, E, R, raw, D, V0..V2 are carved
    /// from these as the pipeline goes.
    rows: [Buffer; 6],
    raw: Buffer,
    in16: Buffer,
    /// Page-locked staging for the input and for the outputs.
    up: Pinned,
    down: Pinned,
    out_f: [Buffer; 5],
    out_efm: Buffer,
    bins: Buffer,
    /// MTF power uploaded last (its level), to skip re-uploading.
    pow_level: Option<f64>,
}

pub(crate) struct GpuDemod {
    blocklen: usize,
    /// Double precision (`REAL` = `double`).
    f64: bool,
    /// DC gain of the three post-demod filters (to add `c * H(0)` back).
    fvideo_dc: [f64; 3],
    state: Mutex<State>,
}

/// Complex values as the kernels' `REAL2`, in bytes.
fn cplx_bytes(v: &[Complex64], f64: bool) -> Vec<u8> {
    if f64 {
        v.iter().flat_map(|c| [c.re.to_ne_bytes(), c.im.to_ne_bytes()]).flatten().collect()
    } else {
        v.iter().flat_map(|c| [(c.re as f32).to_ne_bytes(), (c.im as f32).to_ne_bytes()]).flatten().collect()
    }
}

/// A `REAL` kernel argument.
fn real(v: f64, f64: bool) -> Arg<'static> {
    if f64 {
        Arg::F64(v)
    } else {
        Arg::F32(v as f32)
    }
}

impl GpuDemod {
    pub(crate) fn new(spec: &DemodSpecRef, f64: bool) -> Result<Self> {
        let n = spec.blocklen;
        anyhow::ensure!(n.is_power_of_two() && n.trailing_zeros() % 3 == 0, "GPU demod needs blocklen = 8^k (got {n})");
        let gpu = Gpu::get()?;
        let mut ks = gpu
            .kernels(
                SOURCE,
                &format!("-DN={n} {}", if f64 { "-DREAL=double -DREAL2=double2" } else { "-DREAL=float -DREAL2=float2" }),
                &["load_real", "fft_pass", "rf_products", "discriminate", "video_products", "extract", "gather_bins", "load_u16"],
            )?
            .into_iter();
        let mut k = || ks.next().unwrap();
        let (k_load, k_fft, k_rf, k_disc, k_video, k_extract, k_bins, k_load16) = (k(), k(), k(), k(), k(), k(), k(), k());
        let q = gpu.queue()?;
        let f = spec.filters;
        let upload = |data: &[Complex64]| -> Result<Buffer> {
            let bytes = cplx_bytes(data, f64);
            let b = gpu.buffer(bytes.len())?;
            q.write(&b, 0, &bytes)?;
            Ok(b)
        };
        let tw: Vec<Complex64> = (0..n)
            .map(|m| Complex64::from_polar(1.0, -std::f64::consts::TAU * m as f64 / n as f64))
            .collect();
        let state = State {
            tw: upload(&tw)?,
            rfvideo: upload(&f.rfvideo)?,
            mtfpow: gpu.buffer(n * if f64 { 16 } else { 8 })?,
            fefm: upload(&f.fefm)?,
            frfhpf: upload(&f.frfhpf_rfft)?,
            fvideo: [upload(&f.fvideo_rfft[0])?, upload(&f.fvideo_rfft[1])?, upload(&f.fvideo_rfft[2])?],
            starts: gpu.buffer(16)?,
            rows: [gpu.buffer(8)?, gpu.buffer(8)?, gpu.buffer(8)?, gpu.buffer(8)?, gpu.buffer(8)?, gpu.buffer(8)?],
            raw: gpu.buffer(4)?,
            in16: gpu.buffer(2)?,
            up: gpu.pinned(4)?,
            down: gpu.pinned(4)?,
            out_f: [gpu.buffer(4)?, gpu.buffer(4)?, gpu.buffer(4)?, gpu.buffer(4)?, gpu.buffer(4)?],
            out_efm: gpu.buffer(2)?,
            bins: gpu.buffer(8)?,
            pow_level: None,
            q,
            k_load,
            k_load16,
            k_fft,
            k_rf,
            k_disc,
            k_video,
            k_extract,
            k_bins,
        };
        let fvideo_dc = [f.fvideo_rfft[0][0].re, f.fvideo_rfft[1][0].re, f.fvideo_rfft[2][0].re];
        Ok(Self { blocklen: n, f64, fvideo_dc, state: Mutex::new(state) })
    }

    /// Demodulate `blocks` (each `blocklen` samples), cut as the CPU kernel
    /// does with `cut = true`.
    pub(crate) fn demod(
        &self,
        blocks: &[&[f32]],
        mtf_level: f64,
        mtf_pow: Option<&[Complex64]>,
        rotdelay: i64,
        spec: &DemodSpecRef,
    ) -> Result<Vec<BlockDecode>> {
        let mut out = Vec::with_capacity(blocks.len());
        for chunk in blocks.chunks(MAX_BATCH) {
            out.extend(self.batch(chunk, mtf_level, mtf_pow, rotdelay, spec)?);
        }
        Ok(out)
    }

    fn batch(
        &self,
        blocks: &[&[f32]],
        mtf_level: f64,
        mtf_pow: Option<&[Complex64]>,
        rotdelay: i64,
        spec: &DemodSpecRef,
    ) -> Result<Vec<BlockDecode>> {
        let n = self.blocklen;
        let nb = blocks.len();
        let mut guard = self.state.lock().unwrap();
        let st = &mut *guard;
        let f64 = self.f64;
        let esz = if f64 { 16 } else { 8 };
        let row_bytes = nb * n * esz;
        for r in &mut st.rows {
            r.ensure(row_bytes)?;
        }
        st.raw.ensure(nb * n * esz / 2)?;
        let q = &st.q;

        // Input, as complex rows (rows[0] = staging, then rows[1] = X). Integer
        // samples spanning at most 65536 values (every 16-bit or 8-bit
        // capture) go up as 16-bit codes from their minimum: half the bytes,
        // the same values.
        let (lo, hi) = blocks.iter().flat_map(|b| &b[..n]).fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        let integral = hi - lo <= 65535.0 && blocks.iter().all(|b| b[..n].iter().all(|v| v.fract() == 0.0));
        if integral {
            st.up.ensure(nb * n * 2)?;
            let codes = st.up.slice_mut::<u16>(nb * n);
            for (dst, b) in codes.chunks_exact_mut(n).zip(blocks) {
                for (d, &v) in dst.iter_mut().zip(&b[..n]) {
                    *d = (v - lo) as u16;
                }
            }
            st.in16.ensure(nb * n * 2)?;
            q.upload(&st.in16, 0, &st.up, 0, nb * n * 2)?;
            st.k_load16.args(&[Arg::Buf(&st.in16), Arg::F32(lo), Arg::Buf(&st.rows[0])])?;
            q.launch(&st.k_load16, &[n, nb], 256)?;
        } else {
            st.up.ensure(nb * n * 4)?;
            for (dst, b) in st.up.slice_mut::<f32>(nb * n).chunks_exact_mut(n).zip(blocks) {
                dst.copy_from_slice(&b[..n]);
            }
            q.upload(&st.raw, 0, &st.up, 0, nb * n * 4)?;
            st.k_load.args(&[Arg::Buf(&st.raw), Arg::Buf(&st.rows[0])])?;
            q.launch(&st.k_load, &[n, nb], 256)?;
        }
        // FFT rows[0] -> lands in rows[1] after 5 passes (odd count).
        let fft = |src: usize, tmp: usize, sign: f32| -> Result<usize> {
            let (mut a, mut b) = (src, tmp);
            let mut ns = 1usize;
            while ns < n {
                st.k_fft.args(&[Arg::Buf(&st.rows[a]), Arg::Buf(&st.rows[b]), Arg::Buf(&st.tw), Arg::I32(ns as i32), Arg::F32(sign)])?;
                q.launch(&st.k_fft, &[n / 8, nb], 256)?;
                std::mem::swap(&mut a, &mut b);
                ns *= 8;
            }
            Ok(a)
        };
        let x = fft(0, 1, -1.0)?;
        let free = if x == 1 { 0 } else { 1 };

        // MTF power, uploaded when its level changed.
        let use_pow = mtf_pow.is_some();
        if let Some(p) = mtf_pow {
            if st.pow_level != Some(mtf_level) {
                q.write(&st.mtfpow, 0, &cplx_bytes(p, f64))?;
                st.pow_level = Some(mtf_level);
            }
        }
        // H -> rows[2], E -> rows[3], R -> rows[4].
        st.k_rf.args(&[
            Arg::Buf(&st.rows[x]),
            Arg::Buf(&st.rfvideo),
            Arg::Buf(&st.mtfpow),
            Arg::I32(use_pow as i32),
            Arg::Buf(&st.fefm),
            Arg::Buf(&st.frfhpf),
            Arg::Buf(&st.rows[2]),
            Arg::Buf(&st.rows[3]),
            Arg::Buf(&st.rows[4]),
        ])?;
        q.launch(&st.k_rf, &[n, nb], 256)?;

        // Analog-audio bins, gathered from the block spectrum before it goes.
        let audio_starts: Vec<i32> = (0..2)
            .flat_map(|ch| {
                let af = &spec.filters.audio[ch];
                let half = af.nbins / 2;
                [af.lowbin as i32, (n - af.lowbin - half) as i32]
            })
            .collect();
        let half_bins: Vec<usize> = (0..2).map(|ch| spec.filters.audio[ch].nbins / 2).collect();
        let run = half_bins[0].max(half_bins[1]);
        st.bins.ensure(nb * 4 * run * esz)?;
        q.write(&st.starts, 0, &audio_starts)?;
        st.k_bins.args(&[Arg::Buf(&st.rows[x]), Arg::Buf(&st.starts), Arg::I32(run as i32), Arg::Buf(&st.bins)])?;
        q.launch(&st.k_bins, &[4 * run, nb], 64)?;

        // Inverse transforms: H (hilbert), E (EFM), R (RF highpass). Each lands
        // in the free row after 5 passes; move the roles accordingly.
        let h = fft(2, x, 1.0)?; // x's row is dead now
        let hilbert = h;
        let spare = if h == 2 { x } else { 2 };
        let e = fft(3, spare, 1.0)?;
        let spare2 = if e == 3 { spare } else { 3 };
        let r = fft(4, spare2, 1.0)?;
        let spare3 = if r == 4 { spare2 } else { 4 };
        let _ = free;

        // Discriminator: raw -> st.raw, clamped D -> rows[spare3].
        let c = spec.levels.iretohz(50.0);
        let scale = spec.freq_hz / std::f64::consts::TAU;
        let lo = 1_500_000.0 - c;
        let hi = spec.freq_hz * 0.75 - c;
        st.k_disc.args(&[
            Arg::Buf(&st.rows[hilbert]),
            Arg::Buf(&st.raw),
            Arg::Buf(&st.rows[spare3]),
            Arg::F64(scale),
            Arg::F64(c),
            real(lo, f64),
            real(hi, f64),
        ])?;
        q.launch(&st.k_disc, &[n, nb], 256)?;
        // The hilbert row is free from here; the post-demod FFT uses it.
        let y = fft(spare3, hilbert, -1.0)?;
        let other = if y == spare3 { hilbert } else { spare3 };
        // Free rows for V0..V2: `other`, and the two of {0..5} not holding
        // E, R or Y.
        let used = [e, r, y, other];
        let mut vrows = vec![other];
        vrows.extend((0..6).filter(|i| !used.contains(i)).take(2));
        st.k_video.args(&[
            Arg::Buf(&st.rows[y]),
            Arg::Buf(&st.fvideo[0]),
            Arg::Buf(&st.fvideo[1]),
            Arg::Buf(&st.fvideo[2]),
            Arg::Buf(&st.rows[vrows[0]]),
            Arg::Buf(&st.rows[vrows[1]]),
            Arg::Buf(&st.rows[vrows[2]]),
        ])?;
        q.launch(&st.k_video, &[n, nb], 256)?;
        // Inverse of each video row, with Y's row as the ping-pong partner.
        let mut v_out = [0usize; 3];
        let mut tmp = y;
        for i in 0..3 {
            let o = fft(vrows[i], tmp, 1.0)?;
            v_out[i] = o;
            tmp = if o == vrows[i] { tmp } else { vrows[i] };
        }

        // Cut ranges, as the CPU kernel.
        let cstart = spec.blockcut.min(n);
        let cend = n.saturating_sub(spec.blockcut_end);
        let clen = cend.saturating_sub(cstart);
        let rstart = (spec.blockcut as i64 - rotdelay).clamp(0, n as i64) as usize;
        let rstop_raw = -((spec.blockcut_end as i64) + rotdelay);
        let rstop = if rstop_raw < 0 { (n as i64 + rstop_raw).clamp(0, n as i64) } else { rstop_raw.clamp(0, n as i64) } as usize;
        let rlen = rstop.saturating_sub(rstart);
        for (i, b) in st.out_f.iter_mut().enumerate() {
            b.ensure(nb * if i == 4 { rlen } else { clen } * 4)?;
        }
        st.out_efm.ensure(nb * clen * 2)?;
        let add: Vec<f64> = self.fvideo_dc.iter().map(|g| c * g).collect();
        st.k_extract.args(&[
            Arg::Buf(&st.rows[v_out[0]]),
            Arg::Buf(&st.rows[v_out[1]]),
            Arg::Buf(&st.rows[v_out[2]]),
            Arg::Buf(&st.raw),
            Arg::Buf(&st.rows[r]),
            Arg::Buf(&st.rows[e]),
            Arg::I32(cstart as i32),
            Arg::I32(clen as i32),
            Arg::I32(rstart as i32),
            Arg::I32(rlen as i32),
            real(add[0], f64),
            real(add[1], f64),
            real(add[2], f64),
            real(c, f64),
            Arg::Buf(&st.out_f[0]),
            Arg::Buf(&st.out_f[1]),
            Arg::Buf(&st.out_f[2]),
            Arg::Buf(&st.out_f[3]),
            Arg::Buf(&st.out_f[4]),
            Arg::Buf(&st.out_efm),
        ])?;
        q.launch(&st.k_extract, &[clen.max(rlen), nb], 256)?;

        // Everything back through the pinned buffer, in one wait.
        let sizes: Vec<usize> = (0..5).map(|i| nb * if i == 4 { rlen } else { clen } * 4).collect();
        let (efm_bytes, bin_bytes) = (nb * clen * 2, nb * 4 * run * esz);
        let total = sizes.iter().sum::<usize>() + efm_bytes + bin_bytes;
        st.down.ensure(total)?;
        let mut off = 0;
        for (i, &sz) in sizes.iter().enumerate() {
            q.download(&st.out_f[i], 0, &mut st.down, off, sz)?;
            off += sz;
        }
        q.download(&st.out_efm, 0, &mut st.down, off, efm_bytes)?;
        off += efm_bytes;
        q.download(&st.bins, 0, &mut st.down, off, bin_bytes)?;
        q.finish()?;
        let bytes = st.down.slice::<u8>(total);
        let mut off = 0;
        let mut take = |len: usize| {
            let b = &bytes[off..off + len];
            off += len;
            b
        };
        let ch: Vec<Vec<f32>> = sizes.iter().map(|&sz| take(sz).chunks_exact(4).map(|w| f32::from_ne_bytes([w[0], w[1], w[2], w[3]])).collect()).collect();
        let efm: Vec<i16> = take(efm_bytes).chunks_exact(2).map(|w| i16::from_ne_bytes([w[0], w[1]])).collect();
        let bins: Vec<Complex64> = take(bin_bytes)
            .chunks_exact(esz)
            .map(|w| {
                if f64 {
                    Complex64::new(f64::from_ne_bytes(w[..8].try_into().unwrap()), f64::from_ne_bytes(w[8..].try_into().unwrap()))
                } else {
                    Complex64::new(
                        f64::from(f32::from_ne_bytes(w[..4].try_into().unwrap())),
                        f64::from(f32::from_ne_bytes(w[4..].try_into().unwrap())),
                    )
                }
            })
            .collect();
        drop(guard);

        // Assemble per block; the analog audio's stage 1 runs on the CPU from
        // the gathered bins.
        let res = (0..nb)
            .map(|b| {
                let cl = |i: usize| ch[i][b * clen..(b + 1) * clen].to_vec();
                let audio = [0usize, 1].map(|a| {
                    let base = b * 4 * run + 2 * a * run;
                    let h = half_bins[a];
                    audio_stage1(&bins[base..base + h], &bins[base + run..base + run + h], a, spec, true)
                });
                BlockDecode {
                    video: VideoChannels {
                        demod: cl(0),
                        demod_raw: cl(3),
                        demod_05: cl(1),
                        demod_burst: cl(2),
                        audio,
                        efm: efm[b * clen..(b + 1) * clen].to_vec(),
                    },
                    rfhpf: ch[4][b * rlen..(b + 1) * rlen].to_vec(),
                }
            })
            .collect();
        Ok(res)
    }
}
