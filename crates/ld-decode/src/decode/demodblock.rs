//! Per-block RF demodulation (port of `RFDecode.demodblock_cpu` /
//! `demodblock_sync` from the Python ld-decode).
//!
//! Signal path (mirroring the Python reference exactly):
//! - `indata_fft = fft(data[:blocklen])` (complex f64 FFT)
//! - rfhpf = `ifft(indata_fft * Frfhpf).real` cut to
//!   `[blockcut - rotdelay : -blockcut_end - rotdelay]`, cast to f32
//! - EFM = `ifft(indata_fft * Fefm).real` cut to
//!   `[blockcut : -blockcut_end]`, clipped to i16
//! - `indata_fft_filt = indata_fft * RFVideo` (* MTF**mtf_level when nonzero)
//! - hilbert = ifft(indata_fft_filt); demod = unwrap_hilbert(freq_hz)
//! - demod_fft = fft(clip(demod, 1.5e6, freq_hz*0.75))
//! - out_video = ifft(demod_fft * FVideo).real etc., each `astype(f32)` when
//!   stored, cut to `[blockcut : -blockcut_end]`
//! - analog audio: sliced short-FFT demod, `astype(f32)`, cut to
//!   `[blockcut//fdiv : -blockcut_end//fdiv]`
//!
//! All FFTs are the vendored ducc0 library via `crate::ffi_ducc`, which
//! reproduces `scipy.fft` (scipy >= 1.18) bit-for-bit; the demod runs in f64
//! and only downcasts where the Python does (`.astype(np.float32)`).

use rustfft::num_complex::Complex64;

use crate::ffi_ducc;
use crate::spec::{
    np_cmul, np_cmul3_fill, np_cmul_assign, np_cmul_extend, np_cmul_fill, np_cpow,
    CalibLevels, Filters,
};

/// A borrow of everything a demodulation block needs from the spec, so the
/// demod functions stay decoupled from the full `DecoderSpec`.
pub(crate) struct DemodSpecRef<'a> {
    pub freq: f64,
    pub freq_half: f64,
    pub freq_hz: f64,
    pub blocklen: usize,
    pub blockcut: usize,
    pub blockcut_end: usize,
    pub filters: &'a Filters,
    pub levels: &'a CalibLevels,
}

impl<'a> DemodSpecRef<'a> {
    /// Fresh reference (used by the one-shot delay measurement in `spec.rs`).
    pub fn new(
        freq: f64,
        freq_half: f64,
        freq_hz: f64,
        blocklen: usize,
        filters: &'a Filters,
        levels: &'a CalibLevels,
    ) -> Self {
        Self::with_plans(freq, freq_half, freq_hz, blocklen, filters, levels)
    }

    /// Share the spec's configuration.
    pub fn with_plans(
        freq: f64,
        freq_half: f64,
        freq_hz: f64,
        blocklen: usize,
        filters: &'a Filters,
        levels: &'a CalibLevels,
    ) -> Self {
        Self {
            freq,
            freq_half,
            freq_hz,
            blocklen,
            blockcut: 1024,
            blockcut_end: filters.f05_offset,
            filters,
            levels,
        }
    }

    #[allow(dead_code)]
    pub fn iretohz(&self, ire: f64) -> f64 {
        self.levels.iretohz(ire)
    }
}

/// The video channels produced by one demodulated block (before or after the
/// overlap-save cut). Stored as f32 exactly where the Python reference casts
/// to float32.
#[derive(Clone)]
pub(crate) struct VideoChannels {
    /// Regular (filtered) video output.
    pub demod: Vec<f32>,
    /// Unfiltered instantaneous frequency.
    pub demod_raw: Vec<f32>,
    /// 0.5 MHz lowpass path, used for sync detection.
    pub demod_05: Vec<f32>,
    /// Colour-burst bandpass path.
    pub demod_burst: Vec<f32>,
    /// Stage-1 demodulated analog audio, [left, right] (decimated by fdiv).
    pub audio: [Vec<f32>; 2],
    /// EFM equalised signal, clipped to i16.
    pub efm: Vec<i16>,
}

/// One demodulated block: the video channels plus the (always-cut) RF
/// highpass used for dropout detection.
#[derive(Clone)]
pub(crate) struct BlockDecode {
    pub video: VideoChannels,
    pub rfhpf: Vec<f32>,
}

/// Per-thread work buffers for [`demod_block_cpu`], reused across blocks.
///
/// The kernel runs ~21 times per field and each call used to allocate about a
/// dozen 128-512 KB buffers, nearly all of them holding a pure intermediate.
/// Every buffer here is either fully written by the step that produces it or
/// explicitly initialised, so reusing one cannot change a value: the kernel's
/// arithmetic, operand order and rounding are untouched. Sizes are refitted per
/// call, so any block length this crate demodulates works.
#[derive(Default)]
struct Scratch {
    /// f64 view of the block's samples, feeding the input r2c.
    samples: Vec<f64>,
    /// `fft_real_full` of the block (full Hermitian spectrum); later reused to
    /// hold `demod_fft`, whose input spectrum is dead by then.
    indata: Vec<Complex64>,
    /// Spectrum intermediate: filter products, then the video-filter product.
    spec_buf: Vec<Complex64>,
    /// Inverse-FFT output intermediate.
    out_buf: Vec<Complex64>,
    /// Video-channel filter product.
    ch_spec: Vec<Complex64>,
    /// Video-channel inverse-FFT output.
    ch_out: Vec<Complex64>,
    /// `unwrap_hilbert` output, clamped in place for the second r2c.
    demod_buf: Vec<f64>,
}

thread_local! {
    /// `Option` so the kernel can move the buffers out for the duration of a
    /// block (a panic just means the next block reallocates them).
    static SCRATCH: std::cell::RefCell<Option<Box<Scratch>>> =
        std::cell::RefCell::new(None);
}

/// Port of `utils.unwrap_hilbert`: recover the instantaneous frequency (Hz,
/// in the range [0, freq_hz)) of an analytic (complex) signal via the
/// conjugate-product FM discriminator, in f64 like the reference.
pub(crate) fn unwrap_hilbert(hilbert: &[Complex64], freq_hz: f64) -> Vec<f64> {
    let mut out = Vec::new();
    unwrap_hilbert_into(hilbert, freq_hz, &mut out);
    out
}

/// [`unwrap_hilbert`] into a caller-owned buffer that is reused across blocks.
///
/// The loop writes `1..len` only — index 0 stays the `0.0` the reference's
/// `vec![0.0; len]` starts with — so the buffer is truncated to the new length
/// and index 0 is set explicitly; every other element is overwritten.
pub(crate) fn unwrap_hilbert_into(hilbert: &[Complex64], freq_hz: f64, out: &mut Vec<f64>) {
    use std::f64::consts::TAU;
    let len = hilbert.len();
    if out.len() != len {
        out.clear();
        out.resize(len, 0.0);
    }
    if len == 0 {
        return;
    }
    out[0] = 0.0;
    let scale = freq_hz / TAU;
    // Single pass: the conjugate-product components, the UCRT `atan2` and the
    // wrap/normalise step are all per-index independent, so they are fused.
    // The previous shape materialised `pre` and `pim` — two f64 arrays per
    // call — only to consume them once, then walked `out` a second time to
    // normalise. Arithmetic per element is unchanged:
    //
    // numba's jitted complex128 multiply uses the plain 4-product formula
    // (no FMA), NOT numpy's SIMD fmaddsub kernel, and
    // `d < 0 ? (d + TAU) * scale : d * scale` is the original
    // `if d < 0 { d += TAU }; d * scale`. Verified bit-for-bit against numba
    // 0.62 complex multiply on real data.
    for i in 1..len {
        let z = hilbert[i];
        let w = hilbert[i - 1];
        let (a, bb) = (z.re, z.im);
        let (c, dd) = (w.re, -w.im); // conj(w)
        let pre = a * c - bb * dd;
        let pim = a * dd + bb * c;
        let d = crate::spec::ucrt_atan2::call(pim, pre);
        out[i] = if d < 0.0 { (d + TAU) * scale } else { d * scale };
    }
}



/// Cut of a demodulated channel: `[blockcut : -blockcut_end]` (Python slice).
fn cut_block(data: &[f32], spec: &DemodSpecRef) -> Vec<f32> {
    let start = spec.blockcut.min(data.len());
    let end = data.len().saturating_sub(spec.blockcut_end);
    if start >= end {
        Vec::new()
    } else {
        data[start..end].to_vec()
    }
}

/// The output range one channel keeps: `[blockcut : -blockcut_end]` when the
/// caller asked for a cut, the whole array otherwise.
fn kept_range(len: usize, spec: &DemodSpecRef, cut: bool) -> (usize, usize) {
    if cut {
        (
            spec.blockcut.min(len),
            len.saturating_sub(spec.blockcut_end),
        )
    } else {
        (0, len)
    }
}

/// `cut(roll(ifft(x).re as f32))` in a single pass.
///
/// `offset` is the `np.roll(..., -offset)` shift and `start..end` the kept
/// slice. Rolling first and slicing afterwards visits exactly the source
/// indices `(i + offset) % len`, so gathering them in one go is byte-identical
/// to the old convert-everything / `rotate_left` / copy-the-middle chain, with
/// two passes over a 32768-element buffer per channel removed.
fn rolled_f32_range(data: &[Complex64], offset: usize, start: usize, end: usize) -> Vec<f32> {
    let len = data.len();
    let end = end.min(len);
    if len == 0 || start >= end {
        return Vec::new();
    }
    let off = offset % len;
    let n = end - start;
    // `start` and `off` are both `< len`, so one conditional subtract lands on
    // the first source index; from there the gather runs off the end of `data`
    // at most once, i.e. it is two contiguous runs. Splitting them removes the
    // per-element wrap branch and the bounds check that the single loop needed
    // (measured 16.0 -> 6.1 us for a 32768-element block, same values).
    let s0 = {
        let s = start + off;
        if s >= len {
            s - len
        } else {
            s
        }
    };
    let first = (len - s0).min(n);
    let mut out: Vec<f32> = Vec::with_capacity(n);
    out.extend(data[s0..s0 + first].iter().map(|c| c.re as f32));
    out.extend(data[..n - first].iter().map(|c| c.re as f32));
    out
}

/// `cut(demod.astype(f32))` for the raw channel, built at the kept range only.
fn f32_at_range(demod: &[f64], spec: &DemodSpecRef, cut: bool) -> Vec<f32> {
    let (start, end) = kept_range(demod.len(), spec, cut);
    if start >= end {
        return Vec::new();
    }
    demod[start..end].iter().map(|&v| v as f32).collect()
}

/// `cut_rfhpf`, taken straight off the complex inverse-FFT output: `.re` and
/// the f32 cast are the same value the old chain (complex -> f64 Vec -> f32
/// Vec -> cut) produced, with two arrays' worth of traffic removed. The slice
/// arithmetic is unchanged, including the negative-index stop.
fn cut_rfhpf(data: &[Complex64], spec: &DemodSpecRef, rotdelay: i64) -> Vec<f32> {
    let len = data.len() as i64;
    let start_raw = spec.blockcut as i64 - rotdelay;
    // Python slice stop = -blockcut_end - rotdelay (negative index).
    let stop_raw = -((spec.blockcut_end as i64) + rotdelay);
    let start = start_raw.clamp(0, len);
    let stop = if stop_raw < 0 {
        (len + stop_raw).clamp(0, len)
    } else {
        stop_raw.clamp(0, len)
    };
    if start >= stop {
        Vec::new()
    } else {
        data[start as usize..stop as usize]
            .iter()
            .map(|v| v.re as f32)
            .collect()
    }
}





/// Write one stage-dump file into the LD_DUMP_PIPE dir (first block only).
fn pipe_write(dir: &str, name: &str, bytes: &[u8]) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::File::create(std::path::Path::new(dir).join(name)) {
        let _ = f.write_all(bytes);
    }
}

fn pipe_f64(v: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn pipe_cf(v: &[Complex64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 16);
    for &c in v {
        out.extend_from_slice(&c.re.to_le_bytes());
        out.extend_from_slice(&c.im.to_le_bytes());
    }
    out
}

/// Precompute `MTF ** mtf_level` for one demodulation pass. The whole field
/// (window + prefetch) is demodulated at a single MTF, so this is computed
/// once per field and shared by every block; the per-element values are
/// identical to computing them inside each block, so outputs stay bit-identical.
pub(crate) fn compute_mtf_pow(mtf: &[Complex64], mtf_level: f64) -> Vec<Complex64> {
    mtf.iter()
        .map(|f| {
            if mtf_level == 0.0 {
                Complex64::new(1.0, 0.0)
            } else {
                np_cpow(*f, Complex64::new(mtf_level, 0.0))
            }
        })
        .collect()
}

/// Demodulate one block (port of `demodblock_cpu`).
///
/// `data` must have at least `blocklen` samples (the caller guarantees it).
/// When `cut`, the channels are trimmed like the Python reference; `rfhpf` is
/// always trimmed. `mtf_pow` is the precomputed `MTF ** mtf_level` for this
/// pass (all blocks of a field share the same MTF); `None` skips the MTF
/// multiply entirely (mtf 0, e.g. the synthetic delay measurement).
#[allow(clippy::too_many_arguments)]
/// Cheap always-on profiling counters for the demod kernel: total calls and
/// total CPU nanoseconds spent inside them. Used by the LD_TIMING breakdown to
/// tell how much demod work each field actually triggers.
pub(crate) mod demod_prof {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(crate) static CALLS: AtomicU64 = AtomicU64::new(0);
    pub(crate) static NANOS: AtomicU64 = AtomicU64::new(0);

    /// Stage labels for the `LD_DEMODTIME` breakdown, in `STAGE_NANOS` order.
    pub(crate) const STAGE_NAMES: [&str; 13] = [
        "indata_rfft",
        "rfhpf_cmul",
        "rfhpf_ifft",
        "efm_cmul",
        "efm_ifft",
        "audio",
        "video_cmul",
        "hilbert_ifft",
        "unwrap",
        "clip_rfft",
        "demod_raw",
        "fvideo",
        "cut",
    ];

    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU64 = AtomicU64::new(0);
    pub(crate) static STAGE_NANOS: [AtomicU64; 13] = [
        ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,
    ];

    /// `LD_DEMODTIME`: only then does the kernel pay for the stage timers.
    pub(crate) fn enabled() -> bool {
        static ON: crate::envflag::CachedFlag = crate::envflag::CachedFlag::new();
        ON.get("LD_DEMODTIME")
    }

    pub(crate) fn snapshot() -> (u64, u64) {
        (
            CALLS.load(Ordering::Relaxed),
            NANOS.load(Ordering::Relaxed),
        )
    }

    pub(crate) fn stage_snapshot() -> [u64; 13] {
        std::array::from_fn(|i| STAGE_NANOS[i].load(Ordering::Relaxed))
    }
}

/// Rolling stage timer for the `LD_DEMODTIME` breakdown. One instance is
/// created per demod block and `mark`ed at each stage boundary; the elapsed
/// time since the previous mark is charged to the stage that was current.
/// With the probe off (`on == false`) `mark` is a branch plus a move.
pub(crate) struct StageT {
    on: bool,
    stage: usize,
    at: std::time::Instant,
}

impl StageT {
    #[inline]
    pub(crate) fn new(stage: usize) -> Self {
        let on = demod_prof::enabled();
        StageT {
            on,
            stage,
            at: std::time::Instant::now(),
        }
    }

    /// Charge the time since the last mark to the current stage, then switch.
    #[inline]
    pub(crate) fn mark(&mut self, stage: usize) {
        if self.on {
            demod_prof::STAGE_NANOS[self.stage].fetch_add(
                self.at.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.at = std::time::Instant::now();
        }
        self.stage = stage;
    }
}

impl Drop for StageT {
    #[inline]
    fn drop(&mut self) {
        if self.on {
            demod_prof::STAGE_NANOS[self.stage].fetch_add(
                self.at.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }
}

/// Per-field deltas of the stage counters — summed over every worker thread,
/// so the total is the field's demod **CPU**, not its wall time.
pub(crate) fn stage_deltas() -> [u64; 13] {
    use std::sync::Mutex;
    static PREV: Mutex<[u64; 13]> = Mutex::new([0; 13]);
    let now = demod_prof::stage_snapshot();
    let mut prev = PREV.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = [0u64; 13];
    for i in 0..13 {
        out[i] = now[i].saturating_sub(prev[i]);
        prev[i] = now[i];
    }
    out
}

pub(crate) struct DemodProf(std::time::Instant);

impl DemodProf {
    #[inline]
    pub(crate) fn start() -> Self {
        let _ = demod_prof::CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DemodProf(std::time::Instant::now())
    }
}

impl Drop for DemodProf {
    #[inline]
    fn drop(&mut self) {
        demod_prof::NANOS.fetch_add(
            self.0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

pub(crate) fn demod_block_cpu(
    data: &[f32],
    mtf_level: f64,
    spec: &DemodSpecRef,
    cut: bool,
    mtf_pow: Option<&[Complex64]>,
    rotdelay: i64,
    block_no: u64,
) -> BlockDecode {
    let blocklen = spec.blocklen;
    let _prof = DemodProf::start();
    let mut st = StageT::new(0);

    // Stage-dump harness: every call dumps this block's intermediates and the
    // filters into $LD_DUMP_PIPE as s{n}_{stage}.bin (n = call sequence), so
    // blocks can be matched between a Python and a Rust run by content.
    // The switch is cached: this probe runs on every block of every field.
    static DUMP_PIPE: crate::envflag::CachedFlag = crate::envflag::CachedFlag::new();
    let pipe_dir = if DUMP_PIPE.get("LD_DUMP_PIPE") {
        std::env::var_os("LD_DUMP_PIPE").map(|p| p.to_string_lossy().into_owned())
    } else {
        None
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    static PIPE_SEQ: AtomicUsize = AtomicUsize::new(0);
    // `LD_DUMP_PIPE_BLOCK` restricts the dump to a comma-separated list of block
    // numbers, so only the blocks under investigation are written instead of the
    // whole capture's worth of stage files.
    let dump: Option<usize> = match pipe_dir.as_ref() {
        Some(_) => {
            let want = std::env::var("LD_DUMP_PIPE_BLOCK").ok();
            let wanted = |b: u64| match want.as_deref() {
                None | Some("") => true,
                Some(list) => list
                    .split(',')
                    .filter_map(|s| s.trim().parse::<u64>().ok())
                    .any(|x| x == b),
            };
            if wanted(block_no) {
                Some(PIPE_SEQ.fetch_add(1, Ordering::SeqCst))
            } else {
                None
            }
        }
        None => None,
    };
    if let Some(s) = dump {
        let dir = pipe_dir.as_deref().unwrap();
        let pfx = format!("s{}_", s);
        let input: Vec<u8> = data[..blocklen]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        pipe_write(dir, &format!("{}input.bin", pfx), &input);
        if s == 0 {
            pipe_write(dir, &format!("{}rfvideo.bin", pfx), &pipe_cf(&spec.filters.rfvideo));
            pipe_write(dir, &format!("{}fefm.bin", pfx), &pipe_cf(&spec.filters.fefm));
            pipe_write(dir, &format!("{}frfhpf.bin", pfx), &pipe_cf(&spec.filters.frfhpf));
            pipe_write(dir, &format!("{}mtf.bin", pfx), &pipe_cf(&spec.filters.mtf));
            pipe_write(dir, &format!("{}fvideo.bin", pfx), &pipe_cf(&spec.filters.fvideo[0]));
            pipe_write(dir, &format!("{}fvideo05.bin", pfx), &pipe_cf(&spec.filters.fvideo[1]));
            pipe_write(dir, &format!("{}fvideoburst.bin", pfx), &pipe_cf(&spec.filters.fvideo[2]));
        }
        pipe_write(
            dir,
            &format!("{}mtf_level.bin", pfx),
            &pipe_f64(&[mtf_level]),
        );
        if let Some(pow) = mtf_pow {
            pipe_write(dir, &format!("{}mtf_pow.bin", pfx), &pipe_cf(pow));
        }
    }

    // Reusable work buffers for this block (see `Scratch`).
    let mut sc = SCRATCH.with(|c| c.borrow_mut().take().unwrap_or_default());
    let Scratch {
        samples,
        indata,
        spec_buf,
        out_buf,
        ch_spec,
        ch_out,
        demod_buf,
    } = &mut *sc;

    // indata_fft = npfft.fft(data[:blocklen])  -- f64 (the f32 input samples
    // are exact integers, identical to Python's int16-as-float64 input).
    samples.clear();
    samples.extend(data[..blocklen].iter().map(|&v| f64::from(v)));
    ffi_ducc::fft_real_full_into(samples, indata);
    st.mark(1);

    // Dropout-detection RF highpass. Python cuts with `video_rot` during
    // field decode (delays set), and with 0 during the setup fakedecode
    // (delays not yet computed) — the caller passes the matching value.
    // The product is built directly into a fresh buffer instead of a
    // clone-then-overwrite pass; per element the arithmetic is the same
    // np_cmul, so the spectrum is bit-identical and one full-buffer read
    // plus one write per block is gone.
    np_cmul_fill(indata, &spec.filters.frfhpf, spec_buf);
    st.mark(2);
    ffi_ducc::ifft_into(spec_buf, out_buf);
    if let Some(s) = dump {
        let rfhpf_f64: Vec<f64> = out_buf.iter().map(|v| v.re).collect();
        pipe_write(
            pipe_dir.as_deref().unwrap(),
            &format!("s{}_rfhpf.bin", s),
            &pipe_f64(&rfhpf_f64),
        );
    }
    let rfhpf = cut_rfhpf(out_buf, spec, rotdelay);
    st.mark(3);

    // EFM: efm_out = npfft.ifft(indata_fft * Fefm); .real; clip to i16; cut.
    // Same direct-build as rfhpf_spec above (no clone-then-overwrite pass).
    np_cmul_fill(indata, &spec.filters.fefm, spec_buf);
    st.mark(4);
    ffi_ducc::ifft_into(spec_buf, out_buf);
    if let Some(s) = dump {
        let efm_f64: Vec<f64> = out_buf.iter().map(|v| v.re).collect();
        pipe_write(
            pipe_dir.as_deref().unwrap(),
            &format!("s{}_efm.bin", s),
            &pipe_f64(&efm_f64),
        );
    }
    // The clip is element-wise, so building only the kept range yields exactly
    // the elements the full-length build then sliced out.
    let efm: Vec<i16> = {
        let (start, end) = if cut {
            let start = spec.blockcut.min(out_buf.len());
            let end = out_buf.len().saturating_sub(spec.blockcut_end);
            (start.min(end), end)
        } else {
            (0, out_buf.len())
        };
        out_buf[start..end]
            .iter()
            .map(|v| (v.re.clamp(-32768.0, 32767.0)) as i16)
            .collect()
    };
    st.mark(5);

    // Analog audio stage 1: per-channel sliced bandpass demod.
    let fdiv = spec.filters.audio_fdiv;
    let mut audio: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    for (ch, out) in audio.iter_mut().enumerate() {
        let af = &spec.filters.audio[ch];
        // fft_do_slice(indata_fft)
        let nbins_half = af.nbins / 2;
        // Slice-and-multiply fused: the previous shape materialised the slice
        // copy only to overwrite every element with the filt1 product. The
        // per-element arithmetic (np_cmul) and the slice order are unchanged —
        // the two contiguous runs of the reference's `chain` are now two
        // appends, which also drops the intermediate `Vec` and its ifft input
        // copy.
        let mut sliced: Vec<Complex64> = Vec::with_capacity(af.nbins);
        np_cmul_extend(
            &indata[af.lowbin..af.lowbin + nbins_half],
            &af.filt1[..nbins_half],
            &mut sliced,
        );
        np_cmul_extend(
            &indata[blocklen - af.lowbin - nbins_half..blocklen - af.lowbin],
            &af.filt1[nbins_half..af.nbins],
            &mut sliced,
        );
        let a1 = ffi_ducc::ifft(&sliced);
        // a1u = unwrap_hilbert(a1, a1_freq) + low_freq
        let a1u = unwrap_hilbert(&a1, af.a1_freq);
        let a1u_f32: Vec<f32> = a1u.iter().map(|&v| (v + af.low_freq) as f32).collect();
        // Cut scaled by fdiv.
        let cut_start = (spec.blockcut / fdiv).min(a1u_f32.len());
        let cut_end = a1u_f32.len().saturating_sub(spec.blockcut_end / fdiv);
        *out = if cut && cut_start < cut_end {
            a1u_f32[cut_start..cut_end].to_vec()
        } else {
            a1u_f32
        };
    }

    st.mark(6);
    // indata_fft_filt = indata_fft * RFVideo  (* MTF**mtf_level when nonzero).
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::SeqCst) {
            if let Some(p) = std::env::var_os("LD_DUMP_RFVIDEO") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::File::create(&p) {
                    for v in &spec.filters.rfvideo {
                        let _ = writeln!(f, "{:.17e} {:.17e}", v.re, v.im);
                    }
                    let _ = writeln!(f, "fvideo0");
                    for v in &spec.filters.fvideo[0] {
                        let _ = writeln!(f, "{:.17e} {:.17e}", v.re, v.im);
                    }
                }
            }
        }
    }
    // Python: `indata_fft_filt = indata_fft * RFVideo` (FMA array multiply),
    // then `*= MTF ** mtf_level` (whole-array power, then FMA multiply). The
    // chain is fused into ONE pass over the spectrum: the intermediate
    // `np_cmul(v, f)` product stays in a register instead of round-tripping
    // through `spec_buf`, removing one full-buffer read + write per block.
    // Per element the op sequence is exactly `np_cmul(np_cmul(v, f), mf)` —
    // the same lane ops and rounding as the two-pass form — so the spectrum
    // is bit-identical (same argument the reference's own fused/two-pass
    // split already relies on).
    match mtf_pow {
        Some(mf) => np_cmul3_fill(indata, &spec.filters.rfvideo, mf, spec_buf),
        None => np_cmul_fill(indata, &spec.filters.rfvideo, spec_buf),
    }

    st.mark(7);
    ffi_ducc::ifft_into(spec_buf, out_buf);
    if let Some(s) = dump {
        pipe_write(
            pipe_dir.as_deref().unwrap(),
            &format!("s{}_hilbert.bin", s),
            &pipe_cf(out_buf),
        );
    }
    st.mark(8);
    unwrap_hilbert_into(out_buf, spec.freq_hz, demod_buf);
    if let Some(s) = dump {
        pipe_write(
            pipe_dir.as_deref().unwrap(),
            &format!("s{}_demod.bin", s),
            &pipe_f64(demod_buf),
        );
    }

    // The raw channel comes off the *unclamped* demod (the reference reads
    // `demod` before any clipped copy exists); the clamp then runs in place,
    // holding exactly the values the reference's separate array held, and
    // nothing reads the unclamped demod afterwards.
    let demod_raw = f32_at_range(demod_buf, spec, cut);
    // demod_fft = fft(clip(demod, 1500000, freq_hz * 0.75))
    let freq_hz = spec.freq_hz;
    for v in demod_buf.iter_mut() {
        *v = (*v).clamp(1_500_000.0, freq_hz * 0.75);
    }
    st.mark(9);
    // `demod_fft`, written over `indata`: that spectrum is dead here (the audio
    // slice and the video-filter product were its last users).
    ffi_ducc::fft_real_full_into(demod_buf, indata);
    if let Some(s) = dump {
        pipe_write(
            pipe_dir.as_deref().unwrap(),
            &format!("s{}_demod_fft.bin", s),
            &pipe_cf(indata),
        );
    }
    st.mark(10);

    let mut video = VideoChannels {
        demod: Vec::new(),
        demod_raw,
        demod_05: Vec::new(),
        demod_burst: Vec::new(),
        audio,
        efm,
    };
    st.mark(11);

    // The three post-demod filters, each with its own known delay rolled in
    // the time domain, exactly like the reference. Reused scratch buffers hold
    // the filtered spectrum and the ifft output (no per-channel clone of the
    // demod spectrum), and the output is converted straight to f32; identical
    // arithmetic.
    let (cstart, cend) = kept_range(indata.len(), spec, cut);
    for (i, filter) in spec.filters.fvideo.iter().enumerate() {
        np_cmul_fill(indata, filter, ch_spec);
        ffi_ducc::ifft_into(ch_spec, ch_out);
        if let Some(s) = dump {
            if i < 2 {
                pipe_write(
                    pipe_dir.as_deref().unwrap(),
                    &format!(
                        "s{}_{}.bin",
                        s,
                        if i == 0 { "out_video" } else { "out_video05" }
                    ),
                    &pipe_f64(&ch_out.iter().map(|v| v.re).collect::<Vec<f64>>()),
                );
            }
        }
        let offset = match i {
            0 => 0usize,
            1 => spec.filters.f05_offset,
            2 => spec.filters.fvideo_burst_offset,
            _ => unreachable!(),
        };
        let out_f32 = rolled_f32_range(ch_out, offset, cstart, cend);
        match i {
            0 => video.demod = out_f32,
            1 => video.demod_05 = out_f32,
            2 => video.demod_burst = out_f32,
            _ => unreachable!(),
        }
    }
    st.mark(12);
    // Hand the work buffers back for the next block on this thread.
    SCRATCH.with(|c| *c.borrow_mut() = Some(sc));
    BlockDecode { video, rfhpf }
}

/// Demodulate only the 0.5 MHz path (port of `demodblock_sync`), used for
/// vertical-sync detection.
#[allow(dead_code)]
pub(crate) fn demod_block_sync(data: &[f32], spec: &DemodSpecRef, cut: bool) -> Vec<f32> {
    let blocklen = spec.blocklen;
    let indata_fft = ffi_ducc::fft_real_full(&data[..blocklen].iter().map(|&v| f64::from(v)).collect::<Vec<f64>>());

    let mut filtered = indata_fft;
    for (v, &f) in filtered.iter_mut().zip(&spec.filters.rfvideo) {
        *v = np_cmul(*v, f);
    }

    let hilbert = ffi_ducc::ifft(&filtered);
    let demod = unwrap_hilbert(&hilbert, spec.freq_hz);

    let freq_hz = spec.freq_hz;
    let clipped: Vec<f64> = demod
        .iter()
        .map(|&d| d.clamp(1_500_000.0, freq_hz * 0.75))
        .collect();
    let demod_fft = ffi_ducc::fft_real_full(&clipped);

    let mut out_spec = demod_fft;
    for (v, &f) in out_spec.iter_mut().zip(&spec.filters.fvideo05) {
        *v = np_cmul(*v, f);
    }
    let out_f64: Vec<f64> = ffi_ducc::ifft(&out_spec).iter().map(|v| v.re).collect();
    let mut sync: Vec<f32> = out_f64.iter().map(|&v| v as f32).collect();

    // np.roll(sync, -f05_offset)
    let offset = spec.filters.f05_offset;
    if !sync.is_empty() {
        let n = offset % sync.len();
        sync.rotate_left(n);
    }

    if cut {
        sync = cut_block(&sync, spec);
    }
    sync
}
#[cfg(test)]
mod tests {
    use super::*;

    fn probe_data(len: usize, seed: u64) -> Vec<Complex64> {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        (0..len)
            .map(|_| Complex64::new(next(), next()))
            .collect()
    }

    /// Reference implementation of the roll+cut gather: the single loop with
    /// the per-element wrap condition and bounds-checked indexing that the
    /// production split-range version replaced. Kept here as the semantics the
    /// production version must still reproduce exactly.
    fn rolled_reference(data: &[Complex64], offset: usize, start: usize, end: usize) -> Vec<f32> {
        let len = data.len();
        if len == 0 || start >= end {
            return Vec::new();
        }
        let off = offset % len;
        let mut out = Vec::with_capacity(end.min(len) - start);
        for i in start..end.min(len) {
            let s = i + off;
            let s = if s >= len { s - len } else { s };
            out.push(data[s].re as f32);
        }
        out
    }

    #[test]
    fn rolled_f32_range_matches_reference() {
        let len = 32768usize;
        let data = probe_data(len, 0x243F6A8885A308D3);
        for &(start, end) in &[
            (512usize, 32256usize),
            (0, 100),
            (30000, 32768),
            (10, 11),
            (0, 32768),
            (32767, 32768),
            // `start == end == len` (empty cut); `start` is always `<= len`
            // here, being `blockcut.min(len)` at the call site.
            (32768, 32768),
        ] {
            for &off in &[0usize, 1, 7, 511, 16384, 32767, 32768, 40000] {
                let want = rolled_reference(&data, off, start, end);
                let got = rolled_f32_range(&data, off, start, end);
                assert_eq!(got.len(), want.len(), "len off={off} {start}..{end}");
                for i in 0..want.len() {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "off={off} start={start} end={end} i={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn bench_rolled_gather() {
        use std::time::Instant;
        let len = 32768usize;
        let data = probe_data(len, 0x13198A2E03707344);
        let (start, end) = (512usize, 32256usize);
        let iters = 400;
        let mut sink = 0.0f32;
        let t0 = Instant::now();
        for _ in 0..iters {
            let o = rolled_reference(&data, 0, start, end);
            sink += o[3];
        }
        eprintln!(
            "PERF rolled reference (off=0)  : {:?}/call",
            t0.elapsed() / iters
        );
        let t0 = Instant::now();
        for _ in 0..iters {
            let o = rolled_reference(&data, 511, start, end);
            sink += o[3];
        }
        eprintln!(
            "PERF rolled reference (off=511): {:?}/call",
            t0.elapsed() / iters
        );
        let t0 = Instant::now();
        for _ in 0..iters {
            let o = rolled_f32_range(&data, 0, start, end);
            sink += o[3];
        }
        eprintln!("PERF rolled split (off=0)      : {:?}/call", t0.elapsed() / iters);
        let t0 = Instant::now();
        for _ in 0..iters {
            let o = rolled_f32_range(&data, 511, start, end);
            sink += o[3];
        }
        eprintln!("PERF rolled split (off=511)    : {:?}/call", t0.elapsed() / iters);
        // Allocation cost alone: allocate + touch one element only.
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut o: Vec<f32> = Vec::with_capacity(end - start);
            o.push(1.0);
            sink += o[0];
        }
        eprintln!("PERF empty alloc of that size : {:?}/call", t0.elapsed() / iters);
        eprintln!("PERF sink {sink}");
    }
}
