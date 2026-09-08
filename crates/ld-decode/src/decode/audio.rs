//! Analog audio post-processing (port of `RFDecode.audio_phase2` and the
//! `downscale_audio` helpers from core.py).
//!
//! `audio_phase2` runs the stage-2 LPF * de-emphasis filter over the whole
//! field's stage-1 audio with an overlap-save scheme, then `downscale_audio`
//! resamples it to 44.1 kHz interleaved int16 stereo for the `.pcm` output.

use crate::spec::DecoderSpec;
use rayon::prelude::*;

/// Port of `findpeaks(array, low)` from utils.py (faithful, including its
/// quirks): values below `low` are zeroed, then the returned indices are
/// `loc - 1` for every `loc` where `array[loc] > array[-1]` (the last
/// element) and `array[loc + 1] > array[loc]`. `loc - 1` can be -1; callers
/// clamp like the Python `max(0, ...)` slices.
fn findpeaks(array: &[f32], low: f32) -> Vec<i64> {
    let mut array2 = array.to_vec();
    for v in &mut array2 {
        if *v < low {
            *v = 0.0;
        }
    }
    let last = *array2.last().unwrap_or(&0.0);
    let mut peaks = Vec::new();
    for loc in 0..array2.len().saturating_sub(1) {
        if array2[loc] > last && array2[loc + 1] > array2[loc] {
            peaks.push(loc as i64 - 1);
        }
    }
    peaks
}

/// Port of `RFDecode.runfilter_audio_phase2`: one overlap-save window of the
/// stage-2 filter for both channels. `start` may be negative (Python's
/// negative-index slicing semantics: `frame[start : start + blocklen]`).
///
/// The FFT runs in f64 through the ducc backend, exactly like Python: the
/// float32 stage-1 audio is center-subtracted in float32 (numpy in-place),
/// then zero-padded into a float64 array (`np.zeros_like(audio2_filter)`)
/// before the transform, and the float64 filter + center re-add round in f64.
fn runfilter_audio_phase2(
    spec: &DecoderSpec,
    frame_audio: &[Vec<f32>; 2],
    start: i64,
) -> [Vec<f64>; 2] {
    let blocklen = spec.blocklen;
    let centers = [spec.audio_lfreq, spec.audio_rfreq];

    // Serial prologue: slice + center-subtract both channels, then derive the
    // clip indices from channel 0 (whose raw is untouched by the replacement
    // before `findpeaks` runs, so splitting the old channel loop here keeps
    // every per-sample operation identical to the sequential port).
    let mut raws: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    for ch in 0..2 {
        let len = frame_audio[ch].len() as i64;
        // Python slice frame_audio[ch][start : start + blocklen]
        let (s, e) = if start < 0 {
            let s = (len + start).max(0);
            let e = (len + start + blocklen as i64).min(len);
            (s as usize, e as usize)
        } else {
            let s = start.min(len) as usize;
            let e = (start + blocklen as i64).min(len) as usize;
            (s, e)
        };
        let mut raw: Vec<f32> = if s < e {
            frame_audio[ch][s..e].to_vec()
        } else {
            Vec::new()
        };
        let center = centers[ch] as f32;
        for v in &mut raw {
            *v -= center;
        }
        raws[ch] = raw;
    }
    let clips = findpeaks(&raws[0], 500000.0);

    // The channels are independent once the clip indices exist, so run their
    // FFT-dominated tails in parallel (each output element is computed by the
    // exact same code path — bit-identical f64 results).
    let channels: Vec<Vec<f64>> = (0..2)
        .into_par_iter()
        .map(|ch| {
            let mut raw = raws[ch].clone();
            for &l in &clips {
                let replacelen = 8i64;
                let lo = (l - replacelen).max(0) as usize;
                let hi = ((l + replacelen) as usize).min(raw.len());
                for v in &mut raw[lo..hi] {
                    *v = 0.0;
                }
            }

        // Zero-pad into an f64 block (Python's `np.zeros_like(audio2_filter)`
        // upcasts the f32 stage-1 audio to f64), then the ducc FFT matches
        // `scipy.fft.fft` of the real input bit-for-bit.
        let mut a2_in = vec![0.0f64; blocklen];
        let n = raw.len().min(blocklen);
        for (dst, &s) in a2_in[..n].iter_mut().zip(&raw[..n]) {
            *dst = f64::from(s);
        }
        let mut a2_fft = crate::ffi_ducc::fft_real_full(&a2_in);
        for (v, &f) in a2_fft.iter_mut().zip(&spec.filters.audio[ch].audio2_filter) {
            *v *= f;
        }
            let out_full = crate::ffi_ducc::ifft(&a2_fft);
            let mut output = Vec::with_capacity(raw.len());
            for v in out_full.iter().take(raw.len()) {
                output.push(v.re + centers[ch]);
            }
            output
        })
        .collect();
    let mut outputs: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut channels = channels.into_iter();
    outputs[0] = channels.next().expect("channel 0");
    outputs[1] = channels.next().expect("channel 1");
    outputs
}

/// Port of `RFDecode.audio_phase2`: overlap-save stage-2 filtering of the
/// field's stage-1 audio.
///
/// Returns f64: Python's fast path (one block covering the whole field)
/// returns the raw f64 phase-2 result, and the slow path's f32 array stores
/// are rounded f64 -> f32 then widened back (same stored values).
pub(crate) fn audio_phase2(spec: &DecoderSpec, field_audio: &[Vec<f32>; 2]) -> [Vec<f64>; 2] {
    let blocklen = spec.blocklen;
    let len = field_audio[0].len();

    // Copy the first block in its entirety, to keep audio and video samples aligned.
    let tmp = runfilter_audio_phase2(spec, field_audio, 0);
    if tmp[0].len() >= len {
        // Python's `return tmp[: len(output_audio2)]` keeps the f64 result.
        let mut out: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        for ch in 0..2 {
            out[ch] = tmp[ch][..len].to_vec();
        }
        return out;
    }
    // Slow path: Python's `output_audio2` is an f32 array (dtype of the
    // stage-1 audio), so every store rounds f64 -> f32 (round-half-even).
    let mut output_audio2: [Vec<f32>; 2] = [vec![0.0f32; len], vec![0.0f32; len]];
    for ch in 0..2 {
        let n = tmp[ch].len().min(len);
        for (d, &s) in output_audio2[ch][..n].iter_mut().zip(&tmp[ch][..n]) {
            *d = s as f32;
        }
    }

    let askip = 512; // length of filters that needs to be chopped out of the ifft
    let sjump = blocklen - askip;

    let mut ostart = tmp[0].len();
    let mut sample = sjump;
    while sample < len.saturating_sub(sjump) {
        let tmp = runfilter_audio_phase2(spec, field_audio, sample as i64);
        let oend = ostart + tmp[0].len() - askip;
        for ch in 0..2 {
            for (d, &s) in output_audio2[ch][ostart..oend].iter_mut().zip(&tmp[ch][askip..]) {
                *d = s as f32;
            }
        }
        ostart += tmp[0].len() - askip;
        sample += sjump;
    }

    let tmp = runfilter_audio_phase2(spec, field_audio, len as i64 - blocklen as i64 - 1);
    if tmp[0].len() > askip {
        let tail = tmp[0].len() - askip;
        for ch in 0..2 {
            let dst_start = len - tail;
            for (d, &s) in output_audio2[ch][dst_start..].iter_mut().zip(&tmp[ch][askip..]) {
                *d = s as f32;
            }
        }
    }

    // Widen back so the rest of the pipeline sees f64 (the values are exactly
    // what Python's f32 array holds).
    let mut out: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for ch in 0..2 {
        out[ch] = output_audio2[ch].iter().map(|&v| f64::from(v)).collect();
    }
    out
}

/// Port of `_downscale_audio_compute_locs_and_swow`.
fn downscale_audio_compute_locs_and_swow(
    lineinfo: &[f64],
    line_period: f64,
    linelen: f64,
    linecount: usize,
    timeoffset: f64,
    freq: f64,
    scale: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    // Negative frequencies are a multiple of the HSYNC clock (--ntsc_audio_rate).
    let (timeoffset, freq) = if freq < 0.0 {
        (0.0, (1e6 / line_period) * -freq)
    } else {
        (timeoffset, freq)
    };
    let frametime = linecount as f64 / (1e6 / line_period);
    let soundgap = 1.0 / freq;

    // arange(timeoffset, frametime + soundgap/2, soundgap) -- np.arange's
    // element count is ceil((stop - start)/step).
    let n = (((frametime + soundgap / 2.0 - timeoffset) / soundgap).ceil() as usize).max(0);
    let arange: Vec<f64> = (0..n).map(|i| timeoffset + i as f64 * soundgap).collect();

    let mut locs = vec![0.0f64; n];
    let mut swow = vec![0.0f64; n];

    for (i, &t) in arange.iter().enumerate() {
        let linenum = ((t * 1e6) / line_period) + 1.0;
        let intlinenum = linenum as usize;

        let (lineloc_cur, lineloc_next) = if linenum < 0.0 {
            let cur = lineinfo[0] + linelen * linenum;
            (cur, cur + linelen)
        } else if lineinfo.len() > linenum as usize + 2 {
            (lineinfo[intlinenum], lineinfo[intlinenum + 1])
        } else {
            let cur = lineinfo[lineinfo.len() - 2];
            (cur, cur + linelen)
        };

        let mut sampleloc = lineloc_cur;
        sampleloc += (lineloc_next - lineloc_cur) * (linenum - linenum.floor());

        let mut sw = (lineloc_next - lineloc_cur) / linelen;
        sw = (sw - 1.0) + 1.0;
        if i > 0 && (sw - swow[i - 1]).abs() > 0.015 {
            sw = swow[i - 1];
        }
        swow[i] = sw;
        locs[i] = sampleloc / scale as f64;
    }

    (locs, swow, arange, frametime)
}

/// Port of `dsa_rescale_and_clip`. numba runs this on a float64 value and
/// `np.round` rounds half-to-even, so match both.
fn dsa_rescale_and_clip(infloat: f64) -> i16 {
    let value = (infloat * 32767.0 / 371081.0).round_ties_even() as i32;
    value.clamp(-32766, 32766) as i16
}

/// Port of `_downscale_audio_to_output`: decimate the stage-2 audio to
/// interleaved int16 at the output rate.
fn downscale_audio_to_output(
    locs: &[f64],
    swow: &[f64],
    audio_left: &[f64],
    audio_right: &[f64],
    audio_lfreq: f64,
    audio_rfreq: f64,
) -> (Vec<i16>, bool) {
    let mut output = vec![0i16; 2 * locs.len().saturating_sub(1)];
    let mut failed = false;

    for i in 0..locs.len().saturating_sub(1) {
        let start = locs[i] as usize;
        let end = locs[i + 1] as usize;
        if end > start && end < audio_left.len() {
            // numba's `np.mean` on float64: sequential f64 sum divided by the
            // f64 count.
            let mean = |s: usize, e: usize, data: &[f64]| -> f64 {
                data[s..e].iter().sum::<f64>() / (e - s) as f64
            };
            let output_left = mean(start, end, audio_left) * swow[i] - audio_lfreq;
            let output_right = mean(start, end, audio_right) * swow[i] - audio_rfreq;
            // Flipping audio here to line up with ralf/he010 digital sample.
            output[i * 2] = -dsa_rescale_and_clip(output_left);
            output[i * 2 + 1] = -dsa_rescale_and_clip(output_right);
        } else {
            failed = true;
        }
    }
    (output, failed)
}

/// Port of `downscale_audio`: resample the stage-2 audio to the output rate
/// (positive `freq` in Hz, e.g. 44100) and return interleaved int16 samples.
pub(crate) fn downscale_audio(
    spec: &DecoderSpec,
    audio: &[Vec<f64>; 2],
    lineinfo: &[f64],
    linecount: usize,
    timeoffset: f64,
    freq: f64,
    startloc: u64,
) -> (Vec<i16>, f64) {
    let (locs, swow, arange, frametime) = downscale_audio_compute_locs_and_swow(
        lineinfo,
        spec.sys_line_period,
        spec.linelen as f64,
        linecount,
        timeoffset,
        freq,
        spec.filters.audio_fdiv,
    );
    if let Some(p) = std::env::var_os("LD_DUMP_AUDIO") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
            let _ = writeln!(f, "# startloc={} linecount={} timeoffset={:.17} freq={} len={} {}", startloc, linecount, timeoffset, freq, audio[0].len(), audio[1].len());
            let n = 4000.min(audio[0].len());
            for i in 0..n {
                let _ = writeln!(f, "a {} {}", audio[0][i], audio[1][i]);
            }
            let n2 = 4000.min(locs.len());
            for i in 0..n2 {
                let _ = writeln!(f, "l {:.17} {:.17}", locs[i], swow[i]);
            }
        }
    }

    let (output16, failed) = downscale_audio_to_output(
        &locs,
        &swow,
        &audio[0],
        &audio[1],
        spec.audio_lfreq,
        spec.audio_rfreq,
    );
    if failed {
        tracing::warn!("Analog audio processing error, muting samples");
    }

    let next = *arange.last().unwrap_or(&0.0) - frametime;
    (output16, next)
}
