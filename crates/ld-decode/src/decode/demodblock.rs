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
use crate::spec::{np_cmul, np_cpow, CalibLevels, Filters};

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

/// Port of `utils.unwrap_hilbert`: recover the instantaneous frequency (Hz,
/// in the range [0, freq_hz)) of an analytic (complex) signal via the
/// conjugate-product FM discriminator, in f64 like the reference.
pub(crate) fn unwrap_hilbert(hilbert: &[Complex64], freq_hz: f64) -> Vec<f64> {
    use std::f64::consts::TAU;
    let len = hilbert.len();
    let mut out = vec![0.0f64; len];
    if len == 0 {
        return out;
    }
    let scale = freq_hz / TAU;
    // Stage 1: conjugate-product components. numba's jitted complex128
    // multiply uses the plain 4-product formula (no FMA), NOT numpy's SIMD
    // fmaddsub kernel. Verified bit-for-bit against numba 0.62 complex
    // multiply on real data.
    let mut pim = vec![0.0f64; len];
    let mut pre = vec![0.0f64; len];
    for i in 1..len {
        let z = hilbert[i];
        let w = hilbert[i - 1];
        let (a, bb) = (z.re, z.im);
        let (c, dd) = (w.re, -w.im); // conj(w)
        pre[i] = a * c - bb * dd;
        pim[i] = a * dd + bb * c;
    }
    // Stage 2: one tight atan2 loop over the slice (same UCRT calls, same
    // results, but no per-iteration closure state).
    crate::spec::ucrt_atan2::call_slice(&mut out, &pim, &pre);
    for i in 1..len {
        let mut d = out[i];
        if d < 0.0 {
            d += TAU;
        }
        out[i] = d * scale;
    }
    out
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

fn cut_rfhpf(data: &[f32], spec: &DemodSpecRef, rotdelay: i64) -> Vec<f32> {
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
        data[start as usize..stop as usize].to_vec()
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

    // Stage-dump harness: every call dumps this block's intermediates and the
    // filters into $LD_DUMP_PIPE as s{n}_{stage}.bin (n = call sequence), so
    // blocks can be matched between a Python and a Rust run by content.
    let pipe_dir = std::env::var_os("LD_DUMP_PIPE").map(|p| p.to_string_lossy().into_owned());
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

    // indata_fft = npfft.fft(data[:blocklen])  -- f64 (the f32 input samples
    // are exact integers, identical to Python's int16-as-float64 input).
    let indata_fft = ffi_ducc::fft_real_full(&data[..blocklen].iter().map(|&v| f64::from(v)).collect::<Vec<f64>>());

    // Dropout-detection RF highpass. Python cuts with `video_rot` during
    // field decode (delays set), and with 0 during the setup fakedecode
    // (delays not yet computed) — the caller passes the matching value.
    let mut rfhpf_spec = indata_fft.clone();
    for (v, &f) in rfhpf_spec.iter_mut().zip(&spec.filters.frfhpf) {
        *v = np_cmul(*v, f);
    }
    let rfhpf_full = ffi_ducc::ifft(&rfhpf_spec);
    let rfhpf_f64: Vec<f64> = rfhpf_full.iter().map(|v| v.re).collect();
    if let Some(s) = dump {
        pipe_write(
            pipe_dir.as_deref().unwrap(),
            &format!("s{}_rfhpf.bin", s),
            &pipe_f64(&rfhpf_f64),
        );
    }
    let rfhpf_f32: Vec<f32> = rfhpf_f64.iter().map(|&v| v as f32).collect();
    let rfhpf = cut_rfhpf(&rfhpf_f32, spec, rotdelay);

    // EFM: efm_out = npfft.ifft(indata_fft * Fefm); .real; clip to i16; cut.
    let mut efm_spec = indata_fft.clone();
    for (v, &f) in efm_spec.iter_mut().zip(&spec.filters.fefm) {
        *v = np_cmul(*v, f);
    }
    let efm_full = ffi_ducc::ifft(&efm_spec);
    if let Some(s) = dump {
        let efm_f64: Vec<f64> = efm_full.iter().map(|v| v.re).collect();
        pipe_write(
            pipe_dir.as_deref().unwrap(),
            &format!("s{}_efm.bin", s),
            &pipe_f64(&efm_f64),
        );
    }
    let mut efm: Vec<i16> = efm_full
        .iter()
        .map(|v| (v.re.clamp(-32768.0, 32767.0)) as i16)
        .collect();
    if cut {
        let start = spec.blockcut.min(efm.len());
        let end = efm.len().saturating_sub(spec.blockcut_end);
        efm = efm[start.min(end)..end].to_vec();
    }

    // Analog audio stage 1: per-channel sliced bandpass demod.
    let fdiv = spec.filters.audio_fdiv;
    let mut audio: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    for (ch, out) in audio.iter_mut().enumerate() {
        let af = &spec.filters.audio[ch];
        // fft_do_slice(indata_fft)
        let nbins_half = af.nbins / 2;
        let mut sliced: Vec<Complex64> = indata_fft[af.lowbin..af.lowbin + nbins_half]
            .iter()
            .chain(&indata_fft[blocklen - af.lowbin - nbins_half..blocklen - af.lowbin])
            .cloned()
            .collect();
        // a1 = ifft(sliced * filt1)
        for (v, &f) in sliced.iter_mut().zip(&af.filt1) {
            *v = np_cmul(*v, f);
        }
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
    let mut filtered = indata_fft;
    // Python: `indata_fft_filt = indata_fft * RFVideo` (FMA array multiply),
    // then `*= MTF ** mtf_level` (whole-array power, then FMA multiply).
    if let Some(mtf_pow) = mtf_pow {
        if mtf_pow.len() == spec.filters.rfvideo.len() {
            for (v, (&f, &mf)) in filtered.iter_mut().zip(spec.filters.rfvideo.iter().zip(mtf_pow)) {
                *v = np_cmul(np_cmul(*v, f), mf);
            }
        } else {
            for (v, &f) in filtered.iter_mut().zip(&spec.filters.rfvideo) {
                *v = np_cmul(*v, f);
            }
            for (v, &f) in filtered.iter_mut().zip(mtf_pow) {
                *v = np_cmul(*v, f);
            }
        }
    } else {
        for (v, &f) in filtered.iter_mut().zip(&spec.filters.rfvideo) {
            *v = np_cmul(*v, f);
        }
    }

    let hilbert = ffi_ducc::ifft(&filtered);
    if let Some(s) = dump {
        pipe_write(
            pipe_dir.as_deref().unwrap(),
            &format!("s{}_hilbert.bin", s),
            &pipe_cf(&hilbert),
        );
    }
    let demod = unwrap_hilbert(&hilbert, spec.freq_hz);
    if let Some(s) = dump {
        pipe_write(
            pipe_dir.as_deref().unwrap(),
            &format!("s{}_demod.bin", s),
            &pipe_f64(&demod),
        );
    }

    // demod_fft = fft(clip(demod, 1500000, freq_hz * 0.75))
    let freq_hz = spec.freq_hz;
    let clipped: Vec<f64> = demod
        .iter()
        .map(|&d| d.clamp(1_500_000.0, freq_hz * 0.75))
        .collect();
    let demod_fft = ffi_ducc::fft_real_full(&clipped);
    if let Some(s) = dump {
        pipe_write(
            pipe_dir.as_deref().unwrap(),
            &format!("s{}_demod_fft.bin", s),
            &pipe_cf(&demod_fft),
        );
    }

    let mut video = VideoChannels {
        demod: Vec::new(),
        demod_raw: demod.iter().map(|&v| v as f32).collect(),
        demod_05: Vec::new(),
        demod_burst: Vec::new(),
        audio,
        efm,
    };

    // The three post-demod filters, each with its own known delay rolled in
    // the time domain, exactly like the reference. A single reusable scratch
    // holds the filtered spectrum (no per-channel clone of `demod_fft`), and
    // the ifft output is converted straight to f32; identical arithmetic.
    let mut ch_scratch: Vec<Complex64> = Vec::with_capacity(demod_fft.len());
    for (i, filter) in spec.filters.fvideo.iter().enumerate() {
        ch_scratch.clear();
        for (&s, &f) in demod_fft.iter().zip(filter) {
            ch_scratch.push(np_cmul(s, f));
        }
        let ifft_res = ffi_ducc::ifft(&ch_scratch);
        if let Some(s) = dump {
            if i < 2 {
                pipe_write(
                    pipe_dir.as_deref().unwrap(),
                    &format!(
                        "s{}_{}.bin",
                        s,
                        if i == 0 { "out_video" } else { "out_video05" }
                    ),
                    &pipe_f64(&ifft_res.iter().map(|v| v.re).collect::<Vec<f64>>()),
                );
            }
        }
        let mut out_f32: Vec<f32> = ifft_res.iter().map(|v| v.re as f32).collect();
        let offset = match i {
            0 => 0usize,
            1 => spec.filters.f05_offset,
            2 => spec.filters.fvideo_burst_offset,
            _ => unreachable!(),
        };
        if offset > 0 && !out_f32.is_empty() {
            let n = offset % out_f32.len();
            out_f32.rotate_left(n);
        }
        match i {
            0 => video.demod = out_f32,
            1 => video.demod_05 = out_f32,
            2 => video.demod_burst = out_f32,
            _ => unreachable!(),
        }
    }
    if cut {
        video.demod = cut_block(&video.demod, spec);
        video.demod_raw = cut_block(&video.demod_raw, spec);
        video.demod_05 = cut_block(&video.demod_05, spec);
        video.demod_burst = cut_block(&video.demod_burst, spec);
    }
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