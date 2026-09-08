//! Field-level decoding (port of `Field` and `FieldNTSC` from the Python
//! ld-decode): sync-pulse detection, the vblank state machine, line-location
//! computation and refinement, colour-burst phase tracking, and the final
//! wow-compensated downscale to the TBC picture.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use crate::decode::demodblock::VideoChannels;
use rayon::prelude::*;

use crate::optimized::{
    eval_spline_at, eval_spline_value_deriv_k, make_interp_spline_scaled, scale_field_sinc,
    sinc_lut,
    SincScaleParams,
};
use crate::spec::{CalibLevels, DecoderSpec, SYS_HZ_IRE, SYS_IRE0};

// Vblank state machine states (matching the Python HSYNC..EQPL2 order).
pub(crate) const STATE_HSYNC: u8 = 0;
pub(crate) const STATE_EQPL1: u8 = 1;
pub(crate) const STATE_VSYNC: u8 = 2;
pub(crate) const STATE_EQPL2: u8 = 3;

/// One detected pulse (start + length in samples).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pulse {
    pub start: f64,
    pub len: f64,
}

/// A pulse accepted by the vblank state machine.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ValidPulse {
    pub state: u8,
    pub start: f64,
    pub len: f64,
    pub good: bool,
}

/// Everything the decoder needs from the previously decoded field.
#[derive(Clone)]
pub(crate) struct PrevField {
    pub linelocs: Vec<f64>,
    pub linecount: usize,
    pub is_first_field: bool,
    pub sync_confidence: i64,
    pub valid: bool,
    pub startloc: u64,
    pub demod: Vec<f32>,
    pub lineoffset: usize,
    pub outlinecount: usize,
    pub phase_adjust_median: f64,
    pub field_phase_id: i64,
    /// TBC picture of the previous field (for 3D comb metrics).
    pub dspicture: Vec<u16>,
    pub out_scale: f64,
}

impl PrevField {
    /// Port of `Field.skip_check`, evaluated on the previous field.
    fn skip_check(&self, spec: &DecoderSpec, levels: &CalibLevels) -> i64 {
        let mut score = 0i64;
        let mut vsync_lines = 0i64;
        let vsync_ire = levels.vsync_ire;

        for l in self.outlinecount..self.outlinecount + 8 {
            let l_adj = l + self.lineoffset;
            let _begin = f64::from(self.linelocs[l_adj]);
            let _length = spec.sys_line_period;
            let linelen = _length * line_freq_from_linelen_impl(spec, self.linelocs.len());
            let start = _begin as usize;
            let end = (_begin + linelen) as usize + 1;
            if start >= end || end > self.demod.len() {
                continue;
            }
            let median = median_f32(&mut self.demod[start..end].to_vec());
            let line_ire = levels.hztoire(f64::from(median));

            if inrange(f64::from(line_ire), f64::from(vsync_ire - 10.0), f64::from(vsync_ire / 2.0))
            {
                vsync_lines += 1;
            } else if inrange(f64::from(line_ire), -5.0, 5.0) {
                score += 1;
            } else {
                score -= 1;
            }
        }

        if vsync_lines >= 2 {
            100
        } else if vsync_lines == 1 && score > 0 {
            50
        } else if score > 0 {
            25
        } else {
            0
        }
    }
}

/// The demodulated input window a field decodes from.
pub(crate) struct FieldData {
    pub input: Vec<f32>,
    pub video: VideoChannels,
    pub rfhpf: Vec<f32>,
    /// Stage-2 filtered analog audio, [left, right] (after `audio_phase2`);
    /// f64 like Python's fast-path result.
    pub audio: [Vec<f64>; 2],
    /// EFM equalised signal (i16), for the .efm output.
    pub efm: Vec<i16>,
    pub startloc: u64,
}

/// A decoded field. Owns a working copy of the calibration levels (the
/// `getpulses` retry may recalibrate ire0) and its own `Arc` handle to the
/// spec, so a field can outlive any borrow of the decoder that built it.
#[allow(dead_code)]
pub(crate) struct Field {
    pub spec: Arc<DecoderSpec>,
    pub levels: CalibLevels,
    pub data: FieldData,
    pub prevfield: Option<PrevField>,
    pub fields_written: usize,
    pub readloc: u64,
    pub initphase: bool,

    pub inlinelen: usize,
    pub outlinelen: usize,
    pub lineoffset: usize,
    pub valid: bool,
    pub sync_confidence: i64,
    pub outlinecount: usize,
    pub linecount: Option<usize>,

    pub rawpulses: Vec<Pulse>,
    pub validpulses: Vec<ValidPulse>,
    pub linelocs: Vec<f64>,
    pub linelocs0: Vec<f64>,
    pub linelocs1: Vec<f64>,
    pub linelocs2: Vec<f64>,
    /// The decoder MTF level used for this field's demodulation.
    pub mtf_level: f64,
    pub linebad: Vec<bool>,
    pub nextfieldoffset: Option<f64>,
    pub vblank_next: Option<f64>,
    pub is_first_field: Option<bool>,
    pub needrerun: bool,
    pub skipdetected: bool,
    pub phase_adjust_median: f64,
    pub field_phase_id: i64,
    pub burstmedian: f64,
    pub linecode: Vec<Option<i64>>,
    pub out_scale: f64,
    /// The TBC picture (u16) once downscaled.
    pub dspicture: Vec<u16>,
    /// The raw (f32) downscaled luma.
    /// Downscaled interleaved int16 audio for the .pcm output.
    pub dsaudio: Vec<i16>,
    /// EFM slice for the .efm output (fed to the PLL at write time).
    pub efmout: Vec<i16>,
    pub lt: Timings,
}

/// Timing expectations derived from the measured pulses (port of `get_timings`).
#[derive(Clone, Default)]
pub(crate) struct Timings {
    pub hsync: (f64, f64),
    pub eq: (f64, f64),
    pub vsync: (f64, f64),
    pub hsync_median: f64,
    pub hsync_offset: f64,
}

fn inrange(a: f64, mi: f64, ma: f64) -> bool {
    a >= mi && a <= ma
}

// ---------------------------------------------------------------------------
// Zero-crossing and pulse detection (utils.calczc / findpulses)
// ---------------------------------------------------------------------------

/// Port of `calczc_findfirst`.
fn calczc_findfirst(data: &[f32], target: f64, rising: bool) -> Option<usize> {
    if rising {
        (1..data.len()).find(|&i| f64::from(data[i - 1]) < target && f64::from(data[i]) >= target)
    } else {
        (1..data.len()).find(|&i| f64::from(data[i - 1]) > target && f64::from(data[i]) <= target)
    }
}

/// Port of `calczc_do`: linear-interpolated zero crossing.
pub(crate) fn calczc_do(
    data: &[f32],
    start_offset: usize,
    target: f64,
    edge: i32,
    count: usize,
) -> Option<f64> {
    let icount = count + 1;
    let edge = if edge == 0 {
        if f64::from(data.get(start_offset).copied().unwrap_or(target as f32)) < target {
            1
        } else {
            -1
        }
    } else {
        edge
    };

    let end = (start_offset + icount).min(data.len());
    let loc = calczc_findfirst(&data[start_offset..end], target, edge == 1)?;

    let x = start_offset + loc;
    let a = f64::from(data[x - 1]) - target;
    let b = f64::from(data[x]) - target;

    if b - a != 0.0 {
        let y = -a / (-a + b);
        Some(x as f64 - 1.0 + y)
    } else {
        Some(x as f64 - 1.0)
    }
}

/// Port of `calczc_do` when the Python target is a float32 scalar (e.g.
/// `(porch_level + sync_level) / 2`): numba keeps the arithmetic in f32.
pub(crate) fn calczc_do_f32(
    data: &[f32],
    start_offset: usize,
    target: f32,
    edge: i32,
    count: usize,
) -> Option<f64> {
    let icount = count + 1;
    let edge = if edge == 0 {
        if data.get(start_offset).copied().unwrap_or(target) < target {
            1
        } else {
            -1
        }
    } else {
        edge
    };

    let end = (start_offset + icount).min(data.len());
    let loc = calczc_findfirst_f32(&data[start_offset..end], target, edge == 1)?;

    let x = start_offset + loc;
    let a = data[x - 1] - target;
    let b = data[x] - target;

    if let Some(p) = std::env::var_os("LD_DUMP_ZC2_RAW") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
            let _ = writeln!(f, "x={} a={:.17} b={:.17} target={:.17} d0={:.9} d1={:.9}", x, a, b, target, data[x - 1], data[x]);
        }
    }

    if b - a != 0.0 {
        let y = -a / (-a + b);
        Some(x as f64 - 1.0 + f64::from(y))
    } else {
        Some(x as f64 - 1.0)
    }
}

fn calczc_findfirst_f32(data: &[f32], target: f32, rising: bool) -> Option<usize> {
    if rising {
        (1..data.len()).find(|&i| data[i - 1] < target && data[i] >= target)
    } else {
        (1..data.len()).find(|&i| data[i - 1] > target && data[i] <= target)
    }
}

/// Port of `calczc`. `edge`: -1 falling, 0 either, 1 rising.
pub(crate) fn calczc(
    data: &[f32],
    start_offset: usize,
    target: f64,
    edge: i32,
    count: usize,
    reverse: bool,
) -> Option<f64> {
    if reverse {
        let end = start_offset.min(data.len().saturating_sub(1));
        let mut rev: Vec<f32> = data[..=end].iter().rev().copied().collect();
        let rev_zc = calczc_do(&mut rev, 0, target, edge, count);
        rev_zc.map(|zc| start_offset as f64 - zc)
    } else {
        calczc_do(data, start_offset, target, edge, count)
    }
}

/// Port of `findpulses_numba_raw` + `findpulses`: locate pulses by looking at
/// areas of `sync_ref` below `high`.
pub(crate) fn findpulses(sync_ref: &[f32], high: f64) -> Vec<Pulse> {
    let mut in_pulse = sync_ref.first().map_or(false, |&v| f64::from(v) <= high);
    let mut starts = Vec::new();
    let mut lengths = Vec::new();
    let mut cur_start = 0usize;

    for (pos, &value) in sync_ref.iter().enumerate() {
        if in_pulse {
            if f64::from(value) > high {
                let length = pos - cur_start;
                // Python's findpulses_numba_raw defaults: min_synclen=0,
                // max_synclen=5000.  Long sub-threshold runs (noise in the
                // no-signal zones) must be dropped exactly like python.
                if length <= 5000 && cur_start != 0 {
                    starts.push(cur_start as f64);
                    lengths.push(length as f64);
                }
                in_pulse = false;
            }
        } else if f64::from(value) <= high {
            cur_start = pos;
            in_pulse = true;
        }
    }

    starts
        .into_iter()
        .zip(lengths)
        .map(|(start, len)| Pulse { start, len })
        .collect()
}

fn median_f32(values: &mut [f32]) -> f32 {
    assert!(!values.is_empty());
    let mid = values.len() / 2;
    let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater);
    if values.len().is_multiple_of(2) {
        let (left, &mut hi, _) = values.select_nth_unstable_by(mid, cmp);
        let lo = *left.iter().max_by(|a, b| cmp(a, b)).unwrap();
        (lo + hi) / 2.0
    } else {
        let (_, &mut median, _) = values.select_nth_unstable_by(mid, cmp);
        median
    }
}

fn median_f64_(values: &mut [f64]) -> f64 {
    assert!(!values.is_empty());
    let mid = values.len() / 2;
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater);
    if values.len().is_multiple_of(2) {
        let (left, &mut hi, _) = values.select_nth_unstable_by(mid, cmp);
        let lo = *left.iter().max_by(|a, b| cmp(a, b)).unwrap();
        (lo + hi) / 2.0
    } else {
        let (_, &mut median, _) = values.select_nth_unstable_by(mid, cmp);
        median
    }
}

fn mean_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn rms(values: &[f32]) -> f32 {
    let mean = mean_f32(values);
    let square_mean = values
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f32>()
        / values.len() as f32;
    square_mean.sqrt()
}

fn line_freq_from_linelen_impl(spec: &DecoderSpec, linelen: usize) -> f64 {
    let samplesperline = spec.freq / linelen as f64;
    samplesperline * linelen as f64
}

// ---------------------------------------------------------------------------
// Field construction
// ---------------------------------------------------------------------------

impl Field {
    pub fn new(
        spec: Arc<DecoderSpec>,
        levels: CalibLevels,
        data: FieldData,
        prevfield: Option<PrevField>,
        fields_written: usize,
        readloc: u64,
        initphase: bool,
        mtf_level: f64,
    ) -> Self {
        let outlinecount = (spec.sys_frame_lines / 2) + 1;
        let linelen = spec.linelen;
        let outlinelen = spec.sys_outlinelen;
        let mut field = Field {
            spec,
            levels,
            data,
            prevfield,
            fields_written,
            readloc,
            initphase,
            inlinelen: linelen,
            outlinelen,
            lineoffset: 0,
            valid: false,
            sync_confidence: 100,
            outlinecount,
            linecount: None,
            rawpulses: Vec::new(),
            validpulses: Vec::new(),
            linelocs: Vec::new(),
            linelocs0: Vec::new(),
            linelocs1: Vec::new(),
            linelocs2: Vec::new(),
            mtf_level,
            linebad: Vec::new(),
            nextfieldoffset: None,
            vblank_next: None,
            is_first_field: None,
            needrerun: false,
            skipdetected: false,
            phase_adjust_median: 0.0,
            field_phase_id: 1,
            burstmedian: 0.0,
            linecode: vec![None; 3],
            out_scale: 0.0,
            dspicture: Vec::new(),
            dsaudio: Vec::new(),
            efmout: Vec::new(),
            lt: Timings::default(),
        };
        field.update_out_scale();
        field
    }

    fn update_out_scale(&mut self) {
        // NTSC: (0xC800 - 0x0400) / (100 - vsync_ire)
        self.out_scale = (51200.0 - 1024.0) / (100.0 - self.levels.vsync_ire);
    }

    pub fn is_first_field(&self) -> bool {
        self.is_first_field.unwrap_or(false)
    }

    /// Snapshot the state a later field needs from this one (port of the
    /// `prevfield` chain in the Python decoder).
    pub fn to_prevfield(&self) -> PrevField {
        PrevField {
            linelocs: self.linelocs.clone(),
            linecount: self.linecount.unwrap_or(0),
            is_first_field: self.is_first_field(),
            sync_confidence: self.sync_confidence,
            valid: self.valid,
            startloc: self.data.startloc,
            demod: self.data.video.demod.clone(),
            lineoffset: self.lineoffset,
            outlinecount: self.outlinecount,
            phase_adjust_median: self.phase_adjust_median,
            field_phase_id: self.field_phase_id,
            dspicture: self.dspicture.clone(),
            out_scale: self.out_scale,
        }
    }

    // -----------------------------------------------------------------------
    // Line geometry helpers
    // -----------------------------------------------------------------------

    pub fn get_linelen(&self, line: Option<usize>, linelocs: Option<&[f64]>) -> f64 {
        let linelocs = match linelocs {
            Some(l) => l,
            None if !self.linelocs.is_empty() => &self.linelocs,
            None => return self.inlinelen as f64,
        };
        let Some(line) = line else {
            return self.inlinelen as f64;
        };
        if line + 1 >= linelocs.len() {
            return self.inlinelen as f64;
        }

        let linecount = self.linecount.unwrap_or(0);
        let length = if line >= linecount + self.lineoffset {
            f64::from(linelocs[line]) - f64::from(linelocs[line - 1])
        } else if line > 0 {
            (f64::from(linelocs[line + 1]) - f64::from(linelocs[line - 1])) / 2.0
        } else {
            f64::from(linelocs[line + 1]) - f64::from(linelocs[line])
        };

        if length <= 0.0 {
            self.inlinelen as f64
        } else {
            length
        }
    }

    pub fn get_linefreq(&self, line: Option<usize>, linelocs: Option<&[f64]>) -> f64 {
        let length = self.get_linelen(line, linelocs);
        // samplesperline = freq / linelen (MHz / samples)
        let samplesperline = self.spec.freq / self.inlinelen as f64;
        samplesperline * length
    }

    pub fn usectoinpx(&self, x: f64, line: Option<usize>, linelocs: Option<&[f64]>) -> f64 {
        x * self.get_linefreq(line, linelocs)
    }

    pub fn inpxtousec(&self, x: f64, line: Option<usize>) -> f64 {
        x / self.get_linefreq(line, None)
    }

    pub fn usectooutpx(&self, x: f64) -> f64 {
        x * self.spec.sys_outfreq
    }

    /// Port of `lineslice`: a slice (start, end) of the pre-TBC demod window
    /// for line `l`.
    pub fn lineslice(
        &self,
        l: usize,
        begin: Option<f64>,
        length: Option<f64>,
        linelocs: Option<&[f64]>,
        begin_offset: f64,
    ) -> (usize, usize) {
        let linelocs = match linelocs {
            Some(l) => l,
            None => &self.linelocs,
        };
        let l_adj = l + self.lineoffset;
        if l_adj >= linelocs.len() {
            return (0, 0);
        }
        let mut _begin = linelocs[l_adj];
        if let Some(begin) = begin {
            _begin += self.usectoinpx(begin, Some(l_adj), Some(linelocs));
        }
        let _length = match length {
            Some(l) => l,
            None => self.spec.sys_line_period,
        };
        let _length = self.usectoinpx(_length, None, Some(linelocs));

        ((_begin + begin_offset) as usize, (_begin + _length + begin_offset) as usize + 1)
    }

    /// Port of `lineslice_tbc`: a slice of the TBC picture for (pre-TBC) line `l`.
    pub fn lineslice_tbc(&self, l: usize, begin: Option<f64>, length: Option<f64>) -> (usize, usize) {
        let outlinelen = self.outlinelen as f64;
        let mut _begin = outlinelen * (l - 1) as f64;
        let begin_offset = match begin {
            Some(b) => self.usectooutpx(b),
            None => 0.0,
        };
        _begin += begin_offset;
        let _length = match length {
            Some(l) => self.usectooutpx(l),
            None => outlinelen,
        };
        (_begin.round() as usize, (_begin + _length).round() as usize)
    }

    // -----------------------------------------------------------------------
    // Level conversions
    // -----------------------------------------------------------------------

    pub fn hz_to_output_array(&self, input: &[f32]) -> Vec<u16> {
        let scale = self.out_scale / self.levels.hz_ire;
        let offset = self.spec.sys_output_zero as f64
            - self.levels.vsync_ire * self.out_scale
            - self.levels.ire0 * scale;
        input
            .iter()
            .map(|&sample| {
                let value = f64::from(sample) * scale + offset + 0.5;
                value.clamp(0.0, 65535.0) as u16
            })
            .collect()
    }

    #[allow(dead_code)]
    pub fn hz_to_output_scalar(&self, input: f64) -> f64 {
        let mut reduced = (input - self.levels.ire0) / self.levels.hz_ire;
        reduced -= self.levels.vsync_ire;
        (((reduced * self.out_scale) + self.spec.sys_output_zero as f64).clamp(0.0, 65535.0) + 0.5)
            as u16 as f64
    }

    pub fn output_to_ire(&self, output: f64) -> f64 {
        ((output - self.spec.sys_output_zero as f64) / self.out_scale)
            + self.levels.vsync_ire
    }

    // -----------------------------------------------------------------------
    // Sync detection
    // -----------------------------------------------------------------------

    /// Port of `getpulses`: find raw sync pulses, recalibrating ire0 once on
    /// the first field if nothing is found.
    pub fn getpulses(&mut self, do_retry: bool) -> Vec<Pulse> {
        if let Some(p) = std::env::var_os("LD_DUMP_PULSES") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&p)
            {
                let _ = writeln!(f, "# getpulses do_retry={} readloc={}", do_retry, self.readloc);
                for pl in findpulses(&self.data.video.demod_05, self.levels.iretohz(-20.0)) {
                    let _ = writeln!(f, "{} {}", pl.start, pl.len);
                }
            }
        }
        let pulse_hz_min = self.levels.iretohz(self.levels.vsync_ire - 20.0);
        let pulse_hz_max = self.levels.iretohz(-20.0);
        let _ = pulse_hz_min;
        if let Some(p) = std::env::var_os("LD_DUMP_PULSE_THRESH") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&p)
            {
                let _ = writeln!(
                    f,
                    "# readloc={} high={:.17} ire0={:.17} hz_ire={:.17} vsync_ire={:.17}",
                    self.readloc,
                    pulse_hz_max,
                    self.levels.ire0,
                    self.levels.hz_ire,
                    self.levels.vsync_ire
                );
            }
        }
        if self.readloc == 0 || self.readloc == 39957120 {
            if let Some(p) = std::env::var_os("LD_DUMP_DEMOD05") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    let _ = writeln!(f, "# pass rl={} fields_written={} demodlen={} d05len={} ire0={:.17}", self.readloc, self.fields_written, self.data.video.demod.len(), self.data.video.demod_05.len(), self.levels.ire0);
                    for v in self.data.video.demod_05.iter() {
                        let _ = writeln!(f, "{:.17}", f64::from(*v));
                    }
                }
            }
        }

        let pulses = findpulses(&self.data.video.demod_05, pulse_hz_max);

        if pulses.is_empty() {
            if do_retry && self.fields_written == 0 {
                // Recalibrate sync levels and retry. Python computes
                // `np.percentile(demod_05, 15)` (linear method), whose index is
                // `(n-1)*q/100` and whose value is numpy's `_lerp` of the two
                // neighboring order statistics: `b - (b-a)*(1-t)` for t >= 0.5,
                // else `a + (b-a)*t`, with the difference computed in float32
                // and the final value rounded back to float32.
                let mut sorted = self.data.video.demod_05.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = sorted.len();
                let vi = (n as f64 - 1.0) * 0.15;
                let lo = vi.floor() as usize;
                let frac = vi - lo as f64;
                let a = sorted[lo];
                let b = sorted[lo + 1];
                let diff = f64::from(b - a);
                let v = if frac >= 0.5 {
                    f64::from(b) - diff * (1.0 - frac)
                } else {
                    f64::from(a) + diff * frac
                };
                self.levels.ire0 = f64::from(v as f32);
                return self.getpulses(false);
            }
            return pulses;
        }

        // Determine sync level from the vsync (long) pulses.
        let mut vsync_locs = Vec::new();
        let mut vsync_means = Vec::new();
        let minlength = self.usectoinpx(10.0, None, None);

        for (i, p) in pulses.iter().enumerate() {
            if p.len > minlength {
                vsync_locs.push(i);
                let start = (p.start + self.spec.freq) as usize;
                let end = (p.start + p.len - self.spec.freq) as usize;
                if end <= self.data.video.demod_05.len() {
                    vsync_means.push(mean_f32(&self.data.video.demod_05[start..end]));
                }
            }
        }

        if vsync_means.is_empty() {
            return Vec::new();
        }

        let synclevel = median_f32(&mut vsync_means);

        if (self.levels.hztoire(f64::from(synclevel)) - self.levels.vsync_ire).abs() < 5.0 {
            return pulses;
        }

        // Compute black level from the eq pulses around the vsync and retry.
        let mut black_means = Vec::new();
        let last = *vsync_locs.last().unwrap();
        for i in vsync_locs[0].saturating_sub(5)..vsync_locs[0] {
            if i >= pulses.len() {
                continue;
            }
            let p = pulses[i];
            if inrange(p.len, self.spec.freq * 0.75, self.spec.freq * 2.5) {
                let start = (p.start + self.spec.freq * 5.0) as usize;
                let end = (p.start + self.spec.freq * 20.0) as usize;
                if end <= self.data.video.demod_05.len() {
                    black_means.push(mean_f32(&self.data.video.demod_05[start..end]));
                }
            }
        }
        for i in last + 1..last + 6 {
            if i >= pulses.len() {
                continue;
            }
            let p = pulses[i];
            if inrange(p.len, self.spec.freq * 0.75, self.spec.freq * 2.5) {
                let start = (p.start + self.spec.freq * 5.0) as usize;
                let end = (p.start + self.spec.freq * 20.0) as usize;
                if end <= self.data.video.demod_05.len() {
                    black_means.push(mean_f32(&self.data.video.demod_05[start..end]));
                }
            }
        }

        if black_means.is_empty() {
            return Vec::new();
        }
        let blacklevel = median_f32(&mut black_means);

        // Python: both levels are np.float32, so the division stays f32.
        let pulse_hz_min = f64::from(synclevel) - self.levels.hz_ire * 10.0;
        let pulse_hz_max = f64::from((blacklevel + synclevel) / 2.0);
        let _ = pulse_hz_min;

        findpulses(&self.data.video.demod_05, pulse_hz_max)
    }

    /// Port of `get_timings`.
    pub fn get_timings(&self) -> Timings {
        let pulses = &self.rawpulses;
        let hsync_typical = self.usectoinpx(self.spec.sys_hsync_pulse_us, None, None);

        let hsync_checkmin = self.usectoinpx(self.spec.sys_hsync_pulse_us - 1.75, None, None);
        let hsync_checkmax = self.usectoinpx(self.spec.sys_hsync_pulse_us + 2.0, None, None);

        let mut hlens = Vec::new();
        for p in pulses {
            if inrange(p.len, hsync_checkmin, hsync_checkmax) {
                hlens.push(p.len);
            }
        }

        let mut lt = Timings::default();
        if !hlens.is_empty() {
            lt.hsync_median = median_f64_(&mut hlens);
        } else {
            lt.hsync_median = self.spec.sys_hsync_pulse_us;
        }

        let hsync_min = lt.hsync_median + self.usectoinpx(-0.5, None, None);
        let hsync_max = lt.hsync_median + self.usectoinpx(0.5, None, None);
        lt.hsync = (hsync_min, hsync_max);
        lt.hsync_offset = lt.hsync_median - hsync_typical;

        let eq_min = self.usectoinpx(self.spec.sys_eq_pulse_us - 0.5, None, None) + lt.hsync_offset;
        let eq_max = self.usectoinpx(self.spec.sys_eq_pulse_us + 0.5, None, None) + lt.hsync_offset;
        lt.eq = (eq_min, eq_max);

        let vsync_min =
            self.usectoinpx(self.spec.sys_vsync_pulse_us * 0.5, None, None) + lt.hsync_offset;
        let vsync_max =
            self.usectoinpx(self.spec.sys_vsync_pulse_us + 1.0, None, None) + lt.hsync_offset;
        lt.vsync = (vsync_min, vsync_max);

        lt
    }

    fn pulse_qualitycheck(&self, prevpulse: Option<&ValidPulse>, state: u8, pulse: &Pulse) -> bool {
        let Some(prev) = prevpulse else {
            return false;
        };
        let exprange = if prev.state > 0 && state > 0 {
            (0.4, 0.6)
        } else if prev.state == 0 && state == 0 {
            (0.9, 1.1)
        } else {
            (0.4, 1.1)
        };
        let linelen = (pulse.start - prev.start) / self.inlinelen as f64;
        inrange(linelen, exprange.0, exprange.1)
    }

    /// Port of `run_vblank_state_machine`.
    fn run_vblank_state_machine(&self, pulses: &[Pulse], lt: &Timings) -> (bool, Vec<ValidPulse>) {
        let mut done = false;
        let mut validpulses: Vec<ValidPulse> = Vec::new();
        let mut state_end = 0.0f64;
        let mut state_length: Option<f64> = None;

        for p in pulses {
            let mut spulse: Option<(u8, &Pulse)> = None;
            let state = validpulses.last().map(|v| v.state).unwrap_or(255);

            if state == 255 {
                // First valid pulse must be a regular HSYNC.
                if inrange(p.len, lt.hsync.0, lt.hsync.1) {
                    spulse = Some((STATE_HSYNC, p));
                }
            } else if state == STATE_HSYNC {
                if inrange(p.len, lt.hsync.0, lt.hsync.1) {
                    spulse = Some((STATE_HSYNC, p));
                } else if inrange(p.len, lt.eq.0, lt.eq.1) {
                    spulse = Some((STATE_EQPL1, p));
                    state_length = Some(self.spec.sys_num_pulses as f64 / 2.0);
                } else if inrange(p.len, lt.vsync.0, lt.vsync.1) {
                    spulse = Some((STATE_VSYNC, p));
                }
            } else if state == STATE_EQPL1 {
                if inrange(p.len, lt.eq.0, lt.eq.1) {
                    spulse = Some((STATE_EQPL1, p));
                } else if inrange(p.len, lt.vsync.0, lt.vsync.1) {
                    spulse = Some((STATE_VSYNC, p));
                    state_length = Some(self.spec.sys_num_pulses as f64 / 2.0);
                } else if inrange(p.len, lt.hsync.0, lt.hsync.1) {
                    spulse = Some((STATE_HSYNC, p));
                }
            } else if state == STATE_VSYNC {
                if inrange(p.len, lt.eq.0, lt.eq.1) {
                    spulse = Some((STATE_EQPL2, p));
                    state_length = Some(self.spec.sys_num_pulses as f64 / 2.0);
                } else if inrange(p.len, lt.vsync.0, lt.vsync.1) {
                    spulse = Some((STATE_VSYNC, p));
                } else if p.start > state_end && inrange(p.len, lt.hsync.0, lt.hsync.1) {
                    spulse = Some((STATE_HSYNC, p));
                }
            } else if state == STATE_EQPL2 {
                if inrange(p.len, lt.eq.0, lt.eq.1) {
                    spulse = Some((STATE_EQPL2, p));
                } else if inrange(p.len, lt.hsync.0, lt.hsync.1) {
                    spulse = Some((STATE_HSYNC, p));
                    done = true;
                }
            }

            if let Some((s, pulse)) = spulse {
                if s != state {
                    if pulse.start < state_end {
                        spulse = None;
                    } else if let Some(sl) = state_length {
                        state_end = pulse.start + ((sl - 0.1) * self.inlinelen as f64);
                        state_length = None;
                    }
                }
            }

            if let Some((s, pulse)) = spulse {
                let good = self.pulse_qualitycheck(validpulses.last(), s, pulse);
                validpulses.push(ValidPulse {
                    state: s,
                    start: pulse.start,
                    len: pulse.len,
                    good,
                });
            }

            if done {
                return (done, validpulses);
            }
        }

        (done, validpulses)
    }

    /// Port of `refinepulses`.
    fn refinepulses(&mut self) -> Vec<ValidPulse> {
        self.lt = self.get_timings();
        let lt = self.lt.clone();

        let rawpulses = self.rawpulses.clone();
        let mut valid_pulses: Vec<ValidPulse> = Vec::new();

        let mut i = 0usize;
        while i < rawpulses.len() {
            let curpulse = rawpulses[i];
            if inrange(curpulse.len, lt.hsync.0, lt.hsync.1) {
                let good = self.pulse_qualitycheck(valid_pulses.last(), STATE_HSYNC, &curpulse);
                valid_pulses.push(ValidPulse {
                    state: STATE_HSYNC,
                    start: curpulse.start,
                    len: curpulse.len,
                    good,
                });
                i += 1;
            } else if i > 2
                && inrange(rawpulses[i].len, lt.eq.0, lt.eq.1)
                && valid_pulses.last().map(|v| v.state) == Some(STATE_HSYNC)
            {
                let window_end = (i + 24).min(rawpulses.len());
                let start_idx = i.saturating_sub(2);
                let (done, vblank_pulses) =
                    self.run_vblank_state_machine(&rawpulses[start_idx..window_end], &lt);
                if done {
                    for vp in vblank_pulses.iter().skip(2) {
                        valid_pulses.push(*vp);
                    }
                    i += vblank_pulses.len().saturating_sub(2);
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        valid_pulses
    }

    /// Port of `getBlankRange`.
    fn get_blank_range(
        &self,
        validpulses: &[ValidPulse],
        start: usize,
    ) -> (Option<usize>, Option<usize>) {
        let vp_type: Vec<u8> = validpulses.iter().map(|p| p.state).collect();
        let vsyncs: Vec<usize> = vp_type
            .iter()
            .enumerate()
            .skip(start)
            .filter(|(_, &t)| t == STATE_VSYNC)
            .map(|(i, _)| i)
            .collect();
        let Some(&firstvsync) = vsyncs.first() else {
            return (None, None);
        };
        // `enumerate` already yields absolute indices, so `firstvsync` is
        // absolute (Python's `+ start` exists only because `np.where` on a
        // sliced array returns slice-relative indices).
        if firstvsync < 10 {
            return (None, None);
        }

        for newstart in (firstvsync - 10)..(firstvsync - 4) {
            let blank_locs: Vec<usize> = vp_type
                .iter()
                .enumerate()
                .skip(newstart)
                .filter(|(_, &t)| t > 0)
                .map(|(i, _)| i)
                .collect();
            let Some(&firstblank) = blank_locs.first() else {
                continue;
            };
            let hsync_locs: Vec<usize> = vp_type
                .iter()
                .enumerate()
                .skip(firstblank)
                .filter(|(_, &t)| t == 0)
                .map(|(i, _)| i)
                .collect();
            let Some(&first_hsync) = hsync_locs.first() else {
                continue;
            };
            // Python: hsync_locs[0] is relative to firstblank, so
            // lastblank = hsync_locs[0] + firstblank - 1 = first_hsync - 1.
            let lastblank = first_hsync - 1;
            if lastblank - firstblank > 12 {
                return (Some(firstblank), Some(lastblank));
            }
        }

        (None, None)
    }

    #[allow(dead_code)]
    fn get_vblank_length(&self, is_first_field: bool) -> f64 {
        let _ = is_first_field;
        // NTSC
        self.spec.sys_num_pulses as f64 * 3.0 * 0.5 + 1.0
    }

    /// Port of `processVBlank`.
    #[allow(clippy::type_complexity)]
    fn process_vblank(
        &mut self,
        validpulses: &[ValidPulse],
        start: usize,
        limit: Option<usize>,
    ) -> (Option<f64>, Option<bool>, Option<usize>, i64) {
        let (firstblank, lastblank) = self.get_blank_range(validpulses, start);

        let lastvalid = match limit {
            Some(l) => start + l,
            None => validpulses.len(),
        };
        let (Some(firstblank), Some(lastblank)) = (firstblank, lastblank) else {
            return (None, None, None, 0);
        };
        if firstblank > lastvalid {
            return (None, None, None, 0);
        }

        let loc_presync = validpulses[firstblank - 1].start;

        let pt: Vec<u8> = validpulses[firstblank..].iter().map(|v| v.state).collect();
        let pstart: Vec<f64> = validpulses[firstblank..].iter().map(|v| v.start).collect();
        let plen: Vec<f64> = validpulses[firstblank..].iter().map(|v| v.len).collect();

        let num_pulses = self.spec.sys_num_pulses;

        for state in [STATE_VSYNC, STATE_EQPL1, STATE_EQPL2] {
            let mut grouploc: Option<usize> = None;

            for j in 0..(lastblank - firstblank) {
                if j + num_pulses + 4 > pt.len() {
                    break;
                }
                if pt[j..j + num_pulses].iter().all(|&t| t == state) {
                    let sum_next: usize = pt[j..j + num_pulses + 4]
                        .iter()
                        .map(|&t| (t == state) as usize)
                        .sum();
                    if sum_next != num_pulses {
                        break;
                    }

                    let gaps = diff2(&pstart[j..j + num_pulses]);
                    let lengths = diff(&plen[j..j + num_pulses]);
                    let max_gap = gaps.iter().cloned().fold(0.0f64, f64::max);
                    let max_len = lengths.iter().cloned().fold(0.0f64, f64::max);
                    if max_gap < self.spec.freq * 0.2 && max_len < self.spec.freq * 0.2 {
                        grouploc = Some(j);
                        break;
                    }
                }
            }

            let Some(grouploc) = grouploc else {
                continue;
            };

            let setbegin = validpulses[firstblank + grouploc];
            let firstloc = setbegin.start;

            // distance of the first pulse of this block to line 1
            let distfroml1 = ((state as f64 - 1.0) * self.spec.sys_num_pulses as f64) * 0.5;

            let dist = (firstloc - loc_presync) / self.inlinelen as f64;
            let hdist = (dist * 2.0).round();

            let mut isfirstfield = (hdist % 2.0)
                == if self.spec.sys_first_field_h[1] as f64 != 1.0 {
                    1.0
                } else {
                    0.0
                };
            if ((distfroml1 * 2.0) % 2.0) != 0.0 {
                isfirstfield = !isfirstfield;
            }

            let eqgap = self.spec.sys_first_field_h[isfirstfield as usize];
            let line0 = firstloc - ((eqgap + distfroml1) * self.inlinelen as f64);

            return (Some(line0), Some(isfirstfield), Some(firstblank), 100);
        }

        // Fallback: check line 0 and the first/last eq pulses.
        if lastblank + 1 < validpulses.len()
            && validpulses[firstblank - 1].good
            && validpulses[firstblank].good
            && validpulses[lastblank].good
            && validpulses[lastblank + 1].good
        {
            let gap1 = validpulses[firstblank].start - validpulses[firstblank - 1].start;
            let gap2 = validpulses[lastblank + 1].start - validpulses[lastblank].start;

            // NTSC only.
            if inrange(
                (gap2 + gap1).abs(),
                self.inlinelen as f64 * 1.4,
                self.inlinelen as f64 * 1.6,
            ) {
                let isfirstfield = inrange(gap1 / self.inlinelen as f64, 0.95, 1.05);
                return (
                    Some(validpulses[firstblank - 1].start),
                    Some(isfirstfield),
                    Some(firstblank),
                    50,
                );
            }
            // Python sets sync_confidence = 0 in this else branch; the final
            // bare return below leaves it untouched.
            self.sync_confidence = 0;
            return (None, None, None, 0);
        }

        (None, None, None, 0)
    }

    /// Port of `computeLineLen`.
    fn compute_line_len(&self, validpulses: &[ValidPulse]) -> f64 {
        // longest run of HSYNC (state 0)
        let mut longrun: (usize, i64) = (0, -1);
        let mut currun: Option<(usize, i64)> = None;
        for (i, v) in validpulses.iter().enumerate() {
            if v.state != 0 {
                if let Some(cur) = currun {
                    if cur.1 > longrun.1 {
                        longrun = cur;
                    }
                }
                currun = None;
            } else if let Some(cur) = &mut currun {
                cur.1 += 1;
            } else {
                currun = Some((i, 0));
            }
        }
        if let Some(cur) = currun {
            if cur.1 > longrun.1 {
                longrun = cur;
            }
        }

        let mut linelens = Vec::new();
        // Python iterates range(longrun[0]+1, longrun[0]+longrun[1]); with no
        // pulses longrun[1] stays -1 and the range is simply empty.  Casting
        // -1 to usize would wrap, so guard and clamp like python's range.
        if longrun.1 >= 0 {
            let end = (longrun.0 + longrun.1 as usize).min(validpulses.len());
            for i in (longrun.0 + 1)..end {
                let linelen = validpulses[i].start - validpulses[i - 1].start;
                if inrange(linelen / self.inlinelen as f64, 0.95, 1.05) {
                    linelens.push(linelen);
                }
            }
        }

        if linelens.is_empty() {
            self.inlinelen as f64
        } else {
            linelens.iter().sum::<f64>() / linelens.len() as f64
        }
    }

    fn gl0_trace(&self, msg: &str) {
        use std::io::Write;
        if let Some(p) = std::env::var_os("LD_TRACE_GL0") {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                let _ = writeln!(f, "{} {}", self.readloc, msg);
            }
        }
    }

    /// Port of `getLine0`.
    fn get_line0(
        &mut self,
        validpulses: &[ValidPulse],
        meanlinelen: f64,
    ) -> (Option<f64>, Option<bool>) {
        self.sync_confidence = 100;

        let limit = if self
            .prevfield
            .as_ref()
            .is_some_and(|p| p.valid && p.skip_check(&self.spec, &self.levels) >= 50)
        {
            Some(100)
        } else {
            None
        };

        let (line0loc_local, is_first_local, firstblank_local, conf_local) =
            self.process_vblank(validpulses, 0, limit);

        let mut line0loc_next: Option<f64> = None;
        let mut is_first_next: Option<bool> = None;
        let mut conf_next: i64 = 0;

        if let (Some(_), Some(firstblank_local)) = (line0loc_local, firstblank_local) {
            let (vblank_next, is_not_first_next, _, conf) =
                self.process_vblank(validpulses, firstblank_local + 40, None);
            self.vblank_next = vblank_next;
            if let Some(vblank_next) = self.vblank_next {
                is_first_next = Some(!is_not_first_next.unwrap_or(true));
                let fieldlen = meanlinelen
                    * self.spec.sys_field_lines
                        [if is_first_next.unwrap() { 0 } else { 1 }]
                        as f64;
                line0loc_next = Some((vblank_next - fieldlen).round());
                if line0loc_next.unwrap() < 0.0 {
                    self.sync_confidence = 10;
                }
            }
            conf_next = conf;
        } else {
            self.vblank_next = None;
        }

        // Use the previous field's end to compute a possible line 0.
        let mut line0loc_prev: Option<f64> = None;
        let mut is_first_prev: Option<bool> = None;
        let mut conf_prev: i64 = 0;
        if let Some(prev) = &self.prevfield {
            if prev.valid {
                let frameoffset = self.data.startloc as f64 - prev.startloc as f64;
                line0loc_prev = Some(f64::from(prev.linelocs[prev.linecount]) - frameoffset);
                is_first_prev = Some(!prev.is_first_field);
                conf_prev = prev.sync_confidence;
            }
        }

        if line0loc_local.is_some() && line0loc_next.is_some() && line0loc_prev.is_some() {
            let votes = [
                is_first_local.unwrap_or(false),
                is_first_prev.unwrap_or(false),
                is_first_next.unwrap_or(false),
            ]
            .iter()
            .filter(|&&v| v)
            .count();
            let is_first_all = votes >= 2;
            let mut locs = [
                line0loc_local.unwrap(),
                line0loc_next.unwrap(),
                line0loc_prev.unwrap(),
            ];
            let median = median_f64_(&mut locs);
            self.gl0_trace(&format!(
                "med local={:?} conf_local={} next={:?} prev={:?} conf_prev={} conf={}",
                line0loc_local, conf_local, line0loc_next, line0loc_prev, conf_prev, self.sync_confidence
            ));
            return (Some(median), Some(is_first_all));
        }

        if let Some(line0loc_local) = line0loc_local {
            if conf_local > 50 {
                self.sync_confidence = self.sync_confidence.min(90);
                self.gl0_trace(&format!(
                    "local local={:?} conf_local={} next={:?} prev={:?} conf_prev={} conf={}",
                    Some(line0loc_local), conf_local, line0loc_next, line0loc_prev, conf_prev, self.sync_confidence
                ));
                return (Some(line0loc_local), is_first_local);
            }
        }
        if let Some(line0loc_prev) = line0loc_prev {
            let new_sync_confidence = (conf_prev - 10).max(0).max(10);
            self.sync_confidence = self.sync_confidence.min(new_sync_confidence);
            self.gl0_trace(&format!(
                "prev local={:?} conf_local={} next={:?} prev={:?} conf_prev={} conf={}",
                line0loc_local, conf_local, line0loc_next, Some(line0loc_prev), conf_prev, self.sync_confidence
            ));
            return (Some(line0loc_prev), is_first_prev);
        }
        if let Some(line0loc_next) = line0loc_next {
            self.sync_confidence = conf_next;
            self.gl0_trace(&format!(
                "next local={:?} conf_local={} next={:?} prev={:?} conf_prev={} conf={}",
                line0loc_local, conf_local, Some(line0loc_next), line0loc_prev, conf_prev, self.sync_confidence
            ));
            return (Some(line0loc_next), is_first_next);
        }

        self.gl0_trace(&format!(
            "none local={:?} conf_local={} next={:?} prev={:?} conf_prev={} conf={}",
            line0loc_local, conf_local, line0loc_next, line0loc_prev, conf_prev, self.sync_confidence
        ));
        (None, None)
    }

    /// Port of `compute_linelocs`. Returns (linelocs, linebad, nextfieldoffset).
    /// An empty linelocs means failure.
    fn compute_linelocs(&mut self, spec: &DecoderSpec) -> (Vec<f64>, Vec<bool>, Option<f64>) {
        self.rawpulses = self.getpulses(true);
        if self.rawpulses.is_empty() {
            if self.fields_written > 0 {
                tracing::error!("Unable to find any sync pulses, skipping one field");
                return (Vec::new(), Vec::new(), None);
            }
            tracing::error!("Unable to find any sync pulses, skipping one second");
            return (Vec::new(), Vec::new(), Some(spec.freq_hz));
        }

        let validpulses = self.refinepulses();
        let meanlinelen = self.compute_line_len(&validpulses);
        let (line0loc, is_first_field) = self.get_line0(&validpulses, meanlinelen);
        if let Some(p) = std::env::var_os("LD_DUMP_VP") {
            use std::io::Write;
            let filter = std::env::var("LD_DUMP_VP_RL").unwrap_or_default();
            if filter.is_empty()
                || filter
                    .split(',')
                    .any(|s| s.parse::<u64>().ok() == Some(self.readloc))
            {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&p)
                {
                    let _ = writeln!(
                        f,
                        "# rl={} n={} meanlinelen={:.17} line0loc={:?}",
                        self.readloc,
                        validpulses.len(),
                        meanlinelen,
                        line0loc
                    );
                    for vp in &validpulses {
                        let _ = writeln!(f, "{} {:.6} {:.6} {}", vp.state, vp.start, vp.len, vp.good);
                    }
                }
            }
        }
        self.is_first_field = is_first_field;

        self.linecount = Some(if is_first_field.unwrap_or(false) { 263 } else { 262 });
        let linecount = self.linecount.unwrap();

        // Number of lines to actually process; the entire following VSYNC is
        // processed.
        let proclines = self.outlinecount + self.lineoffset + 10;

        // lastlineloc comes from the next vblank (may be None).
        let lastlineloc = self.vblank_next;

        self.skipdetected = match (line0loc, lastlineloc) {
            (Some(line0loc), Some(lastlineloc)) => {
                let numlines = (lastlineloc - line0loc) / self.inlinelen as f64;
                numlines < (linecount as f64 - 5.0)
            }
            _ => false,
        };

        let Some(line0loc) = line0loc else {
            if !self.initphase {
                tracing::error!("Unable to determine start of field - dropping field");
            }
            return (Vec::new(), Vec::new(), Some((self.inlinelen * 200) as f64));
        };

        // If we don't have enough data at the end, move on to the next field.
        let lastline = (self.rawpulses.last().unwrap().start - line0loc) / meanlinelen;
        if lastline < proclines as f64 {
            // `line0loc` is measured from the start of this field's demod
            // window, so `line0loc - meanlinelen*20` can be negative here
            // (the vsync sits at the very start of a window that has almost
            // no data). The reference decoder never reaches this branch in
            // practice (its 40M-sample read window still holds junk after the
            // last field, so `getpulses` comes up empty and the EOF path
            // fires instead), where adding that value to the absolute
            // `fdoffset` would step *backward* and loop forever re-decoding
            // the same blocks. Clamp to a forward step like the "dropping
            // field" branch below so the tail of a capture always advances
            // and the decode reaches EOF.
            let mut nfo = line0loc - (meanlinelen * 20.0);
            if nfo <= 0.0 {
                nfo = (self.inlinelen * 200) as f64;
            }
            return (Vec::new(), Vec::new(), Some(nfo));
        }

        let mut linelocs_dict: HashMap<isize, f64> = HashMap::new();
        let mut linelocs_dist: HashMap<isize, f64> = HashMap::new();

        for vp in &validpulses {
            let lineloc = (vp.start - line0loc) / meanlinelen;
            let mut rlineloc = lineloc.round();
            let mut lineloc_distance = (lineloc - rlineloc).abs();

            if self.skipdetected {
                let lineloc_end = linecount as f64
                    - ((lastlineloc.unwrap_or(line0loc) - vp.start) / meanlinelen);
                let rlineloc_end = lineloc_end.round();
                let lineloc_end_distance = (lineloc_end - rlineloc_end).abs();

                if vp.state == 0 && rlineloc > 23.0 && lineloc_end_distance < lineloc_distance {
                    rlineloc = rlineloc_end;
                    lineloc_distance = lineloc_end_distance;
                }
            }

            let rlineloc_i = rlineloc as isize;

            // only record if it's closer to the (probable) beginning of the line
            if lineloc_distance > self.spec_hsync_tolerance()
                || (linelocs_dict.contains_key(&rlineloc_i)
                    && lineloc_distance > linelocs_dist[&rlineloc_i])
            {
                continue;
            }

            // skip non-regular lines (non-hsync) that don't seem to be in
            // valid order, or hsync lines in the vblank area
            if rlineloc > 0.0 && !vp.good {
                if vp.state > 0 || (vp.state == 0 && rlineloc < 10.0) {
                    continue;
                }
            }

            linelocs_dict.insert(rlineloc_i, vp.start);
            linelocs_dist.insert(rlineloc_i, lineloc_distance);
        }

        let mut rv_err = vec![false; proclines];

        let linelocs: Vec<f64> = (0..proclines)
            .map(|l| linelocs_dict.get(&(l as isize)).copied().unwrap_or(-1.0))
            .collect();
        let mut linelocs_filled = linelocs.clone();

        self.linelocs0 = linelocs.clone();

        if linelocs_filled[0] < 0.0 {
            let mut next_valid: Option<usize> = None;
            for i in 0..=self.outlinecount {
                if linelocs[i] > 0.0 {
                    next_valid = Some(i);
                    break;
                }
            }

            let Some(next_valid) = next_valid else {
                return (
                    Vec::new(),
                    Vec::new(),
                    Some(line0loc + (self.inlinelen as f64 * self.outlinecount as f64 - 7.0)),
                );
            };

            linelocs_filled[0] = linelocs_filled[next_valid] - (next_valid as f64 * meanlinelen);

            if linelocs_filled[0] < self.inlinelen as f64 {
                return (
                    Vec::new(),
                    Vec::new(),
                    Some(line0loc + (self.inlinelen as f64 * self.outlinecount as f64 - 7.0)),
                );
            }
        }

        for l in 1..proclines {
            if linelocs_filled[l] < 0.0 {
                rv_err[l] = true;

                let mut prev_valid: Option<usize> = None;
                let mut next_valid: Option<usize> = None;

                for i in (0..=l).rev() {
                    if linelocs[i] > 0.0 {
                        prev_valid = Some(i);
                        break;
                    }
                }
                for i in l..=self.outlinecount {
                    if linelocs[i] > 0.0 {
                        next_valid = Some(i);
                        break;
                    }
                }

                if let Some(next_valid) = next_valid {
                    if let Some(prev_valid) = prev_valid {
                        let avglen = (linelocs_filled[next_valid] - linelocs_filled[prev_valid])
                            as f64
                            / (next_valid - prev_valid) as f64;
                        linelocs_filled[l] =
                            linelocs_filled[prev_valid] + (avglen * (l - prev_valid) as f64);
                    } else {
                        let avglen = self.inlinelen as f64;
                        linelocs_filled[l] = linelocs_filled[next_valid]
                            - (avglen * (next_valid - l) as f64);
                    }
                } else if let Some(prev_valid) = prev_valid {
                    let avglen = self.inlinelen as f64;
                    linelocs_filled[l] =
                        linelocs_filled[prev_valid] + (avglen * (l - prev_valid) as f64);
                }
            }
        }

        let nextfield = match self.vblank_next {
            Some(vblank_next) => vblank_next - (self.inlinelen as f64 * 8.0),
            None => f64::from(linelocs_filled[self.outlinecount - 7]),
        };

        (linelocs_filled, rv_err, Some(nextfield))
    }

    fn spec_hsync_tolerance(&self) -> f64 {
        // rf.hsync_tolerance in the Python decoder.
        0.4
    }

    /// Port of `refine_linelocs_hsync`.
    fn refine_linelocs_hsync(&mut self, _spec: &DecoderSpec) -> Vec<f64> {
        if self.readloc == 1341417600 {
            if let Some(p) = std::env::var_os("LD_DUMP_LEVELS") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    let _ = writeln!(f, "# levels ire0={:.17} hz_ire={:.17} vsync_ire={:.17}", self.levels.ire0, self.levels.hz_ire, self.levels.vsync_ire);
                }
            }
        }
        let mut linelocs2 = self.linelocs1.clone();
        let demod_05 = &self.data.video.demod_05;

        for i in 0..self.linelocs1.len() {
            // skip VSYNC lines (they handle pulses differently)
            if inrange(i as f64, 3.0, 6.0) {
                self.linebad[i] = true;
                continue;
            }

            // refine beginning of hsync
            let ll1 = (f64::from(self.linelocs1[i]) - self.spec.freq) as usize;
            let target = self.levels.iretohz(self.levels.vsync_ire / 2.0);
            let zc = calczc(
                demod_05,
                ll1,
                target,
                0,
                (self.spec.freq * 2.0) as usize,
                false,
            );

            if let Some(p) = std::env::var_os("LD_DUMP_ZC") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    let _ = writeln!(f, "# readloc={} do_retry={}", self.readloc, 1);
                    let _ = writeln!(f, "{} ll1={} target={:.17} zc={:?}", i, ll1, target, zc);
                }
            }

            if let (Some(zc), false) = (zc, self.linebad[i]) {
                linelocs2[i] = zc;

                // The hsync area, burst, and porches should not leave
                // -50 to 30 IRE.
                let hsync_start = (zc - self.spec.freq * 0.75) as usize;
                let hsync_end = (zc + self.spec.freq * 8.0) as usize;
                if hsync_end <= demod_05.len() {
                    let hsync_area = &demod_05[hsync_start..hsync_end];
                    let min_h = hsync_area.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_h = hsync_area.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    if f64::from(min_h) < self.levels.iretohz(-55.0) || f64::from(max_h) > self.levels.iretohz(30.0) {
                        self.linebad[i] = true;
                        linelocs2[i] = self.linelocs1[i];
                    } else {
                        let porch_start = (zc + self.spec.freq * 8.0) as usize;
                        let porch_end = (zc + self.spec.freq * 9.0) as usize;
                        let sync_start = (zc + self.spec.freq * 1.0) as usize;
                        let sync_end = (zc + self.spec.freq * 2.5) as usize;
                        // Python's nb_median (numba np.median) computes the
                        // median in f32 but returns float64; the average then
                        // happens in f64.
                        let porch_level =
                            f64::from(median_f32(&mut demod_05[porch_start..porch_end].to_vec()));
                        let sync_level =
                            f64::from(median_f32(&mut demod_05[sync_start..sync_end].to_vec()));

                        let zc2 = calczc(
                            demod_05,
                            ll1,
                            (porch_level + sync_level) / 2.0,
                            0,
                            400,
                            false,
                        );

                        if let Some(p) = std::env::var_os("LD_DUMP_ZC2") {
                            use std::io::Write;
                            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                                let _ = writeln!(f, "{} zc={:.17} porch={:.9} sync={:.9} zc2={:?}", i, zc, porch_level, sync_level, zc2);
                            }
                        }

                        match zc2 {
                            Some(zc2) if (zc2 - zc).abs() < self.spec.freq / 2.0 => {
                                linelocs2[i] = zc2;
                            }
                            _ => {
                                self.linebad[i] = true;
                            }
                        }
                    }
                } else {
                    self.linebad[i] = true;
                }
            } else {
                self.linebad[i] = true;
            }

            if self.linebad[i] {
                linelocs2[i] = self.linelocs1[i];
            }
        }

        linelocs2
    }

    /// Port of `compute_deriv_error`.
    pub fn compute_deriv_error(&self, linelocs: &[f64], baserr: &[bool]) -> Vec<bool> {
        let mut derr1 = vec![false; linelocs.len()];
        if linelocs.len() >= 3 {
            for i in 1..linelocs.len() - 1 {
                let d2 = linelocs[i + 1] - 2.0 * linelocs[i] + linelocs[i - 1];
                derr1[i] = d2.abs() > 4.0;
            }
        }
        let mut derr2 = vec![false; linelocs.len()];
        if linelocs.len() >= 3 {
            for i in 2..linelocs.len() {
                let d2 = linelocs[i] - 2.0 * linelocs[i - 1] + linelocs[i - 2];
                derr2[i] = d2.abs() > 4.0;
            }
        }

        (0..linelocs.len())
            .map(|i| baserr[i] | derr1[i] | derr2[i])
            .collect()
    }

    /// Port of `fix_badlines`.
    fn fix_badlines(
        &mut self,
        _spec: &DecoderSpec,
        linelocs_in: &[f64],
        linelocs_backup_in: Option<&[f64]>,
    ) -> Vec<f64> {
        let linebad = self.compute_deriv_error(linelocs_in, &self.linebad);
        self.linebad = linebad;
        let mut linelocs = linelocs_in.to_vec();

        if let Some(backup) = linelocs_backup_in {
            for (l, v) in linelocs.iter_mut().enumerate() {
                if v.is_nan() {
                    *v = backup[l];
                }
            }
        }

        for l in 0..linelocs.len() {
            if !self.linebad[l] {
                continue;
            }
            let mut prevgood = l as isize - 1;
            while prevgood >= 0 && self.linebad[prevgood as usize] {
                prevgood -= 1;
            }
            let mut nextgood = l + 1;
            while nextgood < linelocs.len() && self.linebad[nextgood] {
                nextgood += 1;
            }

            let firstcheck = 1; // NTSC
            if prevgood >= firstcheck && nextgood < linelocs.len() + self.lineoffset {
                let gap = (linelocs[nextgood] - linelocs[prevgood as usize])
                    / (nextgood - prevgood as usize) as f64;
                linelocs[l] = gap * (l as f64 - prevgood as f64) + linelocs[prevgood as usize];
            }
        }

        linelocs
    }

    // -----------------------------------------------------------------------
    // Wow-compensated downscale
    // -----------------------------------------------------------------------

    /// Port of `computewow_scaled`: build the spline over line locations and
    /// evaluate per-output-sample coordinates and wow factors.
    pub fn computewow_scaled(&mut self) -> Result<(Vec<f64>, Vec<f64>)> {
        let actual_linelocs: Vec<f64> = self.linelocs.iter().map(|&v| f64::from(v)).collect();
        let expected_linelocs: Vec<f64> = (0..actual_linelocs.len())
            .map(|i| i as f64 * self.inlinelen as f64)
            .collect();

        let outscale = self.inlinelen as f64 / self.outlinelen as f64;
        let outsamples = self.outlinecount * self.outlinelen;
        let outline_offset = (self.lineoffset + 1) * self.outlinelen;

        let k = self.spec.wow_interpolation_method.spline_degree();
        let (t, c) = make_interp_spline_scaled(&expected_linelocs, &actual_linelocs, k)?;
        let nt = t.len() - k - 1;

        let eval_count = outsamples + outline_offset;

        // x is strictly increasing (i * outscale), so every knot span is
        // precomputable in one amortized-O(n) serial pass; the per-point
        // spline arithmetic is then independent and runs in parallel. Same
        // arithmetic as the serial loop, bit-for-bit.
        let xs: Vec<f64> = (0..eval_count).map(|i| i as f64 * outscale).collect();
        let mut spans = vec![k; eval_count];
        {
            let mut span = k;
            for (sp, &x) in spans.iter_mut().zip(xs.iter()) {
                if x <= t[k] {
                    span = k;
                } else if x >= t[nt] {
                    span = nt - 1;
                } else {
                    while span + 1 < nt && x >= t[span + 1] {
                        span += 1;
                    }
                }
                *sp = span;
            }
        }

        let mut interpolated_pixel_locs = vec![0.0f64; eval_count];
        let mut wowfactors = vec![0.0f64; eval_count];
        {
            let chunk = 16384usize;
            let n_chunks = eval_count.div_ceil(chunk);
            let locs_out: Vec<&mut [f64]> = interpolated_pixel_locs.chunks_mut(chunk).collect();
            let wows_out: Vec<&mut [f64]> = wowfactors.chunks_mut(chunk).collect();
            let spans_par: Vec<&[usize]> = spans.chunks(chunk).collect();
            let xs_par: Vec<&[f64]> = xs.chunks(chunk).collect();
            locs_out
                .into_par_iter()
                .zip(wows_out)
                .zip(spans_par)
                .zip(xs_par)
                .enumerate()
                .for_each(|(ci, (((locs, wows), sp), xsp))| {
                    let lo = ci * chunk;
                    for (k_i, (loc_slot, wow_slot)) in locs.iter_mut().zip(wows.iter_mut()).enumerate() {
                        let i = lo + k_i;
                        let (loc, wow) = match k {
                            1 => eval_spline_at::<1>(&t, &c, nt, sp[k_i], xsp[k_i]),
                            2 => eval_spline_at::<2>(&t, &c, nt, sp[k_i], xsp[k_i]),
                            _ => eval_spline_at::<3>(&t, &c, nt, sp[k_i], xsp[k_i]),
                        };
                        *loc_slot = loc;
                        *wow_slot = wow;
                    }
                });
        }

        Ok((interpolated_pixel_locs, wowfactors))
    }

    /// Port of `Field.downscale` (video path only; analog audio is not yet
    /// ported). `final_=true` converts the luma to u16.
    pub fn downscale(
        &mut self,
        linesout: usize,
        outwidth: usize,
        channel: usize,
        final_: bool,
        audio_freq: f64,
        audio_offset: f64,
    ) -> Result<Vec<f32>> {
        let sub = std::env::var_os("LD_SUBTIME").is_some();
        let t0 = std::time::Instant::now();
        let (interpolated_pixel_locs, wowfactors) = self.computewow_scaled()?;
        let t_wow = t0.elapsed().as_nanos() as u64;
        if let Some(p) = std::env::var_os("LD_DUMP_WOW") {
            use std::io::Write;
            let filter = std::env::var("LD_DUMP_WOW_FILTER").unwrap_or_default();
            let ok = filter.is_empty()
                || filter.split(',').any(|s| {
                    s == self.data.startloc.to_string() || s == self.readloc.to_string()
                });
            if ok {
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    let _ = writeln!(f, "# startloc={} linesout={} outwidth={}", self.data.startloc, linesout, outwidth);
                    for (a, b) in interpolated_pixel_locs.iter().zip(wowfactors.iter()) {
                        let _ = writeln!(f, "{:.17} {:.17}", a, b);
                    }
                }
            }
        }

        if let Some(p) = std::env::var_os("LD_DUMP_SCALE_IN") {
            use std::io::Write;
            let filter = std::env::var("LD_DUMP_SCALE_FILTER").unwrap_or_default();
            let ok = filter.is_empty()
                || filter.split(',').any(|s| s == self.data.startloc.to_string());
            if ok {
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    let _ = writeln!(f, "# startloc={} linesout={} outwidth={}", self.data.startloc, linesout, outwidth);
                }
            }
        }
        let mut dsout = vec![0.0f32; linesout * outwidth];
        let channel_data = match channel {
            0 => &self.data.video.demod,
            1 => &self.data.video.demod_raw,
            2 => &self.data.video.demod_05,
            3 => &self.data.video.demod_burst,
            _ => unreachable!(),
        };
        if let Some(p) = std::env::var_os("LD_DUMP_SCALE_IN") {
            use std::io::Write;
            let filter = std::env::var("LD_DUMP_SCALE_FILTER").unwrap_or_default();
            let ok = filter.is_empty()
                || filter.split(',').any(|s| s == self.data.startloc.to_string());
            if ok {
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    for v in channel_data.iter() {
                        let _ = writeln!(f, "{:.9e}", v);
                    }
                }
            }
        }
        let t1 = std::time::Instant::now();
        scale_field_sinc(
            channel_data,
            &mut dsout,
            &interpolated_pixel_locs,
            &wowfactors,
            sinc_lut(),
            SincScaleParams {
                lineoffset: self.lineoffset,
                outwidth,
                wow_level_adjust_smoothing: self.spec.wow_level_adjust_smoothing,
                level_adjust_threshold: 15.0,
            },
        );
        let t_sinc = t1.elapsed().as_nanos() as u64;

        let t2 = std::time::Instant::now();
        if final_ {
            self.dspicture = self.hz_to_output_array(&dsout);
        }

        if let Some(p) = std::env::var_os("LD_DUMP_FIELDCHAIN") {
            use std::io::Write;
            let filter = std::env::var("LD_DUMP_FIELDCHAIN_RL").unwrap_or_default();
            let ok = filter.is_empty()
                || filter.split(',').any(|s| {
                    s == self.readloc.to_string() || s == self.data.startloc.to_string()
                });
            if ok {
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    let _ = writeln!(
                        f,
                        "# readloc={} startloc={} lineoffset={} outwidth={} out_scale={:.17} ire0={:.17} hz_ire={:.17} vsync_ire={:.17} smoothing={} mtf_level={:.17}",
                        self.readloc,
                        self.data.startloc,
                        self.lineoffset,
                        outwidth,
                        self.out_scale,
                        self.levels.ire0,
                        self.levels.hz_ire,
                        self.levels.vsync_ire,
                        self.spec.wow_level_adjust_smoothing,
                        self.mtf_level
                    );
                    let _ = writeln!(f, "D {}", channel_data.len());
                    for v in channel_data.iter() {
                        let _ = writeln!(f, "{:.9e}", v);
                    }
                    let _ = writeln!(f, "O {}", dsout.len());
                    for v in dsout.iter() {
                        let _ = writeln!(f, "{:.9e}", v);
                    }
                    let _ = writeln!(f, "P {}", self.dspicture.len());
                    for v in self.dspicture.iter() {
                        let _ = writeln!(f, "{}", v);
                    }
                }
            }
        }
        let t_hz = t2.elapsed().as_nanos() as u64;

        // Analog audio: resample the stage-2 audio to the output rate.
        let t3 = std::time::Instant::now();
        if audio_freq != 0.0 {
            let linecount = self.linecount.unwrap_or(self.outlinecount);
            let (dsaudio, _next) = crate::decode::audio::downscale_audio(
                &self.spec,
                &self.data.audio,
                &self.linelocs,
                linecount,
                audio_offset,
                audio_freq,
                self.data.startloc,
            );
            self.dsaudio = dsaudio;
        }
        let t_aud = t3.elapsed().as_nanos() as u64;

        // EFM: slice the equalised signal between the first and last lines.
        let t4 = std::time::Instant::now();
        if self.data.efm.len() > 2 && self.linelocs.len() > 2 {
            let linecount = self.linecount.unwrap_or(self.outlinecount);
            let start = self.linelocs[1] as usize;
            let end = self
                .linelocs
                .get(linecount + 1)
                .map(|&l| l as usize)
                .unwrap_or(self.data.efm.len());
            if start < end && end <= self.data.efm.len() {
                if self.readloc == 1341417600 {
                    if let Some(p) = std::env::var_os("LD_DUMP_LINELOCS_FULL") {
                        use std::io::Write;
                        if let Ok(mut f) = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(&p)
                        {
                            for v in self.linelocs.iter() {
                                let _ = writeln!(f, "{:.17}", v);
                            }
                        }
                    }
                }
                if let Some(p) = std::env::var_os("LD_DUMP_LINELOCS") {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&p)
                    {
                        let _ = writeln!(
                            f,
                            "{} {} {} {} {} {}",
                            linecount,
                            start,
                            end,
                            self.data.efm.len(),
                            self.readloc,
                            self.readloc
                        );
                    }
                }
                if let Some(p) = std::env::var_os("LD_DUMP_EFMHEAD") {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::File::create(&p) {
                        let n = self.data.efm.len().min(200000);
                        for v in &self.data.efm[..n] {
                            let _ = writeln!(f, "{}", v);
                        }
                    }
                }
                if let Some(p) = std::env::var_os("LD_DUMP_D05HEAD") {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::File::create(&p) {
                        let n = self.data.video.demod_05.len().min(200000);
                        for v in &self.data.video.demod_05[..n] {
                            let _ = writeln!(f, "{}", v);
                        }
                    }
                }
                if let Some(p) = std::env::var_os("LD_DUMP_INPUTHEAD") {
                    use std::io::Write;
                    if self.readloc == 1341417600 {
                    if let Ok(mut f) = std::fs::File::create(&p) {
                        let _ = writeln!(f, "# startloc={} blockcut={} blockcut_end={} readloc={}", self.data.startloc, self.spec.blockcut, self.spec.blockcut_end, self.readloc);
                        let n = self.data.input.len().min(200000);
                        for v in &self.data.input[..n] {
                            let _ = writeln!(f, "{}", v);
                        }
                    }
                    }
                }
                if let Some(p) = std::env::var_os("LD_DUMP_DEMODHEAD") {
                    use std::io::Write;
                    if self.data.startloc == 1414799168 || self.data.startloc == 1417494688 {
                        let mut fname = p.clone();
                        fname.push(format!("_{}", self.data.startloc));
                        if let Ok(mut f) = std::fs::File::create(&fname) {
                            let _ = writeln!(f, "# startloc={} mtf_level={}", self.data.startloc, self.mtf_level);
                            for v in &self.data.video.demod {
                                let _ = writeln!(f, "{}", v);
                            }
                        }
                    }
                }
                if let Some(p) = std::env::var_os("LD_DUMP_LL1") {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&p)
                    {
                        let _ = writeln!(f, "# rl={} start={} end={}", self.readloc, start, end);
                        for v in &self.linelocs {
                            let _ = writeln!(f, "{:.6}", v);
                        }
                    }
                }
                self.efmout = self.data.efm[start..end].to_vec();
            }
        }
        let t_efm = t4.elapsed().as_nanos() as u64;
        if sub {
            eprintln!("SUBTIME wow={:.3} sinc={:.3} hz={:.3} aud={:.3} efm={:.3} ms", t_wow as f64 / 1e6, t_sinc as f64 / 1e6, t_hz as f64 / 1e6, t_aud as f64 / 1e6, t_efm as f64 / 1e6);
        }

        Ok(dsout)
    }

    // -----------------------------------------------------------------------
    // NTSC colour burst tracking
    // -----------------------------------------------------------------------

    pub fn get_burstlevel(&self, l: usize, linelocs: Option<&[f64]>) -> f64 {
        let (start, end) = self.lineslice(l, Some(5.5), Some(2.4), linelocs, 0.0);
        if start >= end || end > self.data.video.demod.len() {
            return 0.0;
        }
        let burstarea = &self.data.video.demod[start..end];
        f64::from(rms(burstarea)) * std::f64::consts::SQRT_2
    }

    fn calc_burstmedian(&self, _spec: &DecoderSpec) -> f64 {
        let mut burstlevel = Vec::new();
        for l in 11..264 {
            burstlevel.push(self.get_burstlevel(l, None));
        }
        let median = if burstlevel.is_empty() {
            0.0
        } else {
            median_f64_(&mut burstlevel)
        };
        median / self.levels.hz_ire
    }

    /// Port of `compute_line_bursts`.
    fn compute_line_bursts(
        &self,
        linelocs: &[f64],
        line: usize,
        prev_phaseadjust: f64,
    ) -> (Option<bool>, f64) {
        let line = line + self.lineoffset;
        // calczc works from integers, so get the start and remainder.
        let s = linelocs[line] as usize;
        let s_rem = linelocs[line] - s as f64;

        if let Some(p) = std::env::var_os("LD_DUMP_BURSTAREA") {
            use std::io::Write;
            let filter = std::env::var("LD_DUMP_BURSTAREA_RL").unwrap_or_default();
            if filter.split(',').any(|x| x.parse::<u64>().ok() == Some(self.readloc)) && line == 8 {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&p)
                {
                    let _ = writeln!(
                        f,
                        "# rl={} line={} s={} burstlen={} demodlen={}",
                        self.readloc,
                        line,
                        s,
                        self.data.video.demod_burst.len(),
                        self.data.video.demod.len()
                    );
                    for v in self.data.video.demod_burst.iter().skip(s).take(300) {
                        let _ = writeln!(f, "{:.9}", v);
                    }
                }
            }
        }

        let lfreq = self.get_linefreq(Some(line), Some(linelocs));

        let fsc_mhz_inv = 1.0 / self.spec.sys_fsc_mhz;

        // approximate burst beginning/end in usecs
        let bstime = 21.0 * fsc_mhz_inv;
        let betime = 28.0 * fsc_mhz_inv;

        let bstart = (bstime * lfreq) as usize;
        let bend = (betime * lfreq) as usize;

        // Python slices `demod_burst[s+bstart : s+bend]` and only bails when
        // that slice is empty; a partially out-of-range slice is used as-is.
        let burst_len = self.data.video.demod_burst.len();
        let demod_len = self.data.video.demod.len();
        if s + bstart >= burst_len || s + bstart >= demod_len {
            return (None, 0.0);
        }
        let start = s + bstart;
        let end = (s + bend).min(burst_len).min(demod_len);
        let burstarea_raw = &self.data.video.demod_burst[start..end];
        let mean = mean_f32(burstarea_raw);
        let burstarea: Vec<f32> = burstarea_raw.iter().map(|&v| v - mean).collect();
        let threshold = rms(&burstarea);

        let burstarea_demod = &self.data.video.demod[start..end];
        let dmean = mean_f32(burstarea_demod);
        let demod_mean_removed: Vec<f32> = burstarea_demod.iter().map(|&v| v - dmean).collect();
        let absmax = demod_mean_removed.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
        if f64::from(absmax) > 30.0 * self.levels.hz_ire {
            return (None, 0.0);
        }

        let zcburstdiv = (lfreq * fsc_mhz_inv) / 2.0;

        // Apply phase adjustment from the previous frame/line if available.
        let mut phase_adjust = -prev_phaseadjust;

        // A proper colour burst should have ~12-13 zero crossings.
        let mut isrising = [false; 16];
        let mut zcs = [0.0f32; 16];

        let mut zc_count = 0usize;
        let mut rising_count = 0usize;
        for _pass in 0..2 {
            let (zc, pa, rc) = clb_findbursts(
                &mut isrising,
                &mut zcs,
                &burstarea,
                0,
                burstarea.len() - 1,
                threshold,
                bstart,
                s_rem,
                zcburstdiv,
                phase_adjust,
            );
            zc_count = zc;
            phase_adjust = pa;
            rising_count = rc;
        }

        let rising = rising_count > (zc_count / 2);
        if let Some(p) = std::env::var_os("LD_DUMP_BURSTAREA") {
            use std::io::Write;
            let filter = std::env::var("LD_DUMP_BURSTAREA_RL").unwrap_or_default();
            if filter.split(',').any(|x| x.parse::<u64>().ok() == Some(self.readloc)) && line == 8 {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&p)
                {
                    let _ = writeln!(
                        f,
                        "# line8 threshold={:.17} absmax={:.17} limit={:.17} ba_len={} zc_count={} rising_count={} rising={:?} pa={:.9}",
                        threshold,
                        f64::from(absmax),
                        30.0 * self.levels.hz_ire,
                        burstarea.len(),
                        zc_count,
                        rising_count,
                        rising,
                        phase_adjust
                    );
                }
            }
        }
        (Some(rising), -phase_adjust)
    }

    /// Port of `compute_burst_offsets`.
    fn compute_burst_offsets(&mut self, linelocs: &[f64]) -> (bool, HashMap<usize, f64>) {
        let mut rising_sum = 0usize;
        let mut adjs: HashMap<usize, f64> = HashMap::new();

        for l in 0..266 {
            let mut prev_phaseadjust = self.phase_adjust_median;
            if prev_phaseadjust == 0.0 {
                if let Some(prev) = &self.prevfield {
                    prev_phaseadjust = prev.phase_adjust_median;
                }
            }

            let (rising, phase_adjust) = self.compute_line_bursts(linelocs, l, prev_phaseadjust);
            if let Some(p) = std::env::var_os("LD_DUMP_CLB") {
                use std::io::Write;
                let filter = std::env::var("LD_DUMP_CLB_RL").unwrap_or_default();
                if filter.is_empty()
                    || filter.split(',').any(|s| s.parse::<u64>().ok() == Some(self.readloc))
                {
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&p)
                    {
                        let _ = writeln!(
                            f,
                            "# rl={} l={} rising={:?} pa={:.9} prev={:.9}",
                            self.readloc, l, rising, phase_adjust, prev_phaseadjust
                        );
                    }
                }
            }
            let Some(rising) = rising else {
                continue;
            };

            // For adjustments, 1/2 the phase_adjust value is used for better
            // results.
            adjs.insert(l, phase_adjust / 2.0);

            let even_line = l % 2 == 0;
            if even_line && rising {
                rising_sum += 1;
            }
        }

        // If more than half of the lines have rising phase alignment, it's
        // (probably) field 1 or 4.
        let field14 = rising_sum > (adjs.len() / 4);

        // Python does `np.median([adjs[a] for a in adjs]) * 2` unconditionally:
        // an empty list or any NaN entry makes the median NaN, and that NaN
        // then poisons `phase_adjust_median` for subsequent fields (it is only
        // replaced when `== 0`, and NaN never compares equal). Mirror exactly.
        let mut vals: Vec<f64> = adjs.values().copied().collect();
        let med = if vals.is_empty() || vals.iter().any(|v| v.is_nan()) {
            f64::NAN
        } else {
            median_f64_(&mut vals)
        };
        self.phase_adjust_median = med * 2.0;

        (field14, adjs)
    }

    /// Port of `refine_linelocs_burst`.
    fn refine_linelocs_burst(&mut self, spec: &DecoderSpec, linelocs: &[f64]) -> Vec<f64> {
        let mut linelocs_adj = linelocs.to_vec();

        let (field14, adjs_new) = self.compute_burst_offsets(&linelocs_adj);

        for l in 1..266 {
            if !adjs_new.contains_key(&l) {
                if l < self.linebad.len() {
                    self.linebad[l] = true;
                }
            }
        }

        // Compute the adjustments for each line but *do not* apply, so
        // outliers can be bypassed.
        let mut adjs: HashMap<usize, f64> = HashMap::new();
        for l in 0..266.min(linelocs_adj.len()) {
            if !linelocs_adj[l].is_nan() && !self.linebad[l] {
                let lfreq = self.get_linefreq(Some(l), Some(linelocs));
                if let Some(adj) = adjs_new.get(&l) {
                    adjs.insert(l, f64::from(*adj) * lfreq * (1.0 / spec.sys_fsc_mhz));
                }
            }
        }

        if !adjs.is_empty() {
            let mut adj_vals: Vec<f64> = adjs.values().copied().collect();
            // Python: np.median of the adj values; any NaN propagates.
            let adjs_median = if adj_vals.iter().any(|v| v.is_nan()) {
                f64::NAN
            } else {
                median_f64_(&mut adj_vals)
            };
            let mut lastvalid_adj = adjs_median;

            for l in 0..266.min(linelocs_adj.len()) {
                if let Some(adj) = adjs.get(&l) {
                    if inrange(adj - adjs_median, -2.0, 2.0) {
                        linelocs_adj[l] += *adj;
                        lastvalid_adj = *adj;
                    } else {
                        linelocs_adj[l] += lastvalid_adj;
                    }
                } else {
                    linelocs_adj[l] += lastvalid_adj;
                }
            }

            // (first field, field14) -> fieldPhaseID
            let is_first = self.is_first_field.unwrap_or(false);
            self.field_phase_id = match (is_first, field14) {
                (true, true) => 1,
                (false, false) => 2,
                (true, false) => 3,
                (false, true) => 4,
            };
        } else {
            self.field_phase_id = 1;
        }

        linelocs_adj
    }

    /// Port of `decodephillipscode`.
    fn decodephillipscode(&self, _spec: &DecoderSpec, linenum: usize) -> Option<i64> {
        let linestart = f64::from(self.linelocs[linenum]);
        let data = &self.data.video.demod;

        let mut curzc = calczc(
            data,
            (linestart + self.usectoinpx(2.0, None, None)) as usize,
            self.levels.iretohz(50.0),
            0,
            self.usectoinpx(12.0, None, None) as usize,
            false,
        );

        let mut zc: Vec<(f64, bool)> = Vec::new();
        while let Some(c) = curzc {
            let idx = (c - self.usectoinpx(0.5, None, None)) as usize;
            let bit = idx < data.len() && f64::from(data[idx]) < self.levels.iretohz(50.0);
            zc.push((c, bit));
            curzc = calczc(
                data,
                (c + self.usectoinpx(1.9, None, None)) as usize,
                self.levels.iretohz(50.0),
                0,
                self.usectoinpx(0.2, None, None) as usize,
                false,
            );
        }

        if zc.len() != 24 {
            return None;
        }
        let usecgap: Vec<f64> = zc.windows(2).map(|w| self.inpxtousec(w[1].0 - w[0].0, None)).collect();
        let min_gap = usecgap.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_gap = usecgap.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !(min_gap > 1.85 && max_gap < 2.15) {
            return None;
        }

        let mut linecode: i64 = 0;
        for b in (0..24).step_by(4) {
            linecode *= 0x10;
            let mut byte = 0u8;
            for (k, bit) in zc[b..b + 4].iter().enumerate() {
                if bit.1 {
                    byte |= 1 << (3 - k);
                }
            }
            linecode += i64::from(byte);
        }

        Some(linecode)
    }

    /// Port of `compute_syncconf`.
    pub fn compute_syncconf(&mut self) -> i64 {
        let mut newconf = 100i64;

        let linecount = self.linecount.unwrap_or(0);
        let end = (self.lineoffset + linecount).min(self.linelocs.len());
        let mut lld2max = 0.0f64;
        if self.lineoffset + 2 <= end {
            for i in self.lineoffset..end - 2 {
                let lld2 = self.linelocs[i + 2] - 2.0 * self.linelocs[i + 1] + self.linelocs[i];
                lld2max = lld2max.max(lld2);
            }
            if lld2max > 4.0 {
                newconf = 45;
            }
        }
        newconf = newconf.max(0);
        let in_conf = self.sync_confidence;
        self.sync_confidence = self.sync_confidence.min(newconf);
        self.gl0_trace(&format!(
            "syncconf in={} newconf={} out={} lld2max={:.6} n={}",
            in_conf, newconf, self.sync_confidence, lld2max, (end - self.lineoffset).saturating_sub(2)
        ));
        self.sync_confidence
    }

    pub fn get_vsync_lines(&self) -> Vec<usize> {
        let mut rv = Vec::new();
        let end = if self.is_first_field() { 10 } else { 9 };
        for i in 1..end {
            rv.push(i);
        }
        rv
    }

    /// The `get_vblank_length` value plus a small margin (port of the same).
    #[allow(dead_code)]
    pub fn vsync_area_lines(&self) -> usize {
        (self.get_vblank_length(self.is_first_field()) + 0.6) as usize
    }

    /// Port of `Field.process` + the NTSC additions from `FieldNTSC.process`.
    pub fn process(&mut self) -> Result<()> {
        let spec = self.spec.clone();
        let (linelocs1, linebad, nextfieldoffset) = self.compute_linelocs(&spec);
        if linelocs1.is_empty() {
            self.nextfieldoffset = match nextfieldoffset {
                Some(v) => Some(v),
                None => Some((self.inlinelen * 200) as f64),
            };
            return Ok(());
        }
        self.linelocs1 = linelocs1;
        self.linebad = linebad;
        self.nextfieldoffset = nextfieldoffset;
        if self.readloc == 1334631232 || self.readloc == 1341417600 {
            if let Some(p) = std::env::var_os("LD_DUMP_LLSTAGES") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    let _ = writeln!(f, "# ll1 readloc={}", self.readloc);
                    for v in self.linelocs1.iter() { let _ = writeln!(f, "{:.17}", v); }
                }
            }
        }

        self.linebad = self.compute_deriv_error(&self.linelocs1, &self.linebad);
        self.linelocs2 = self.refine_linelocs_hsync(&spec);
        if self.readloc == 1334631232 || self.readloc == 1341417600 {
            if let Some(p) = std::env::var_os("LD_DUMP_LLSTAGES") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    let _ = writeln!(f, "# ll2 readloc={}", self.readloc);
                    for v in self.linelocs2.iter() { let _ = writeln!(f, "{:.17}", v); }
                }
            }
        }
        self.linebad = self.compute_deriv_error(&self.linelocs2, &self.linebad);
        self.linelocs = self.linelocs2.clone();
        self.valid = true;

        // ---- NTSC-specific process ----
        self.update_out_scale();
        if !self.valid {
            return Ok(());
        }

        self.linecode = [
            self.decodephillipscode(&spec, 16 + self.lineoffset),
            self.decodephillipscode(&spec, 17 + self.lineoffset),
            self.decodephillipscode(&spec, 18 + self.lineoffset),
        ]
        .to_vec();

        let linelocs2 = self.linelocs2.clone();
        let linelocs3 = self.refine_linelocs_burst(&spec, &linelocs2);
        let linelocs4 = self.fix_badlines(&spec, &linelocs3, Some(&linelocs2));
        if self.readloc == 1334631232 {
            if let Some(p) = std::env::var_os("LD_DUMP_LLFINAL") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    let _ = writeln!(f, "# ll4");
                    for v in linelocs4.iter() { let _ = writeln!(f, "{:.17}", v); }
                }
            }
        }
        self.burstmedian = self.calc_burstmedian(&spec);

        // Subcarrier phase offset in degrees, calibrated for correct NTSC
        // burst phase (~147 deg) at the output.
        let fsc_phase_deg = 117.25;
        let shift_samples = (fsc_phase_deg / 360.0) / self.spec.sys_fsc_mhz * self.spec.freq;
        self.linelocs = linelocs4.iter().map(|&v| v - shift_samples).collect();

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// clb_findbursts (utils.clb_findbursts)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn clb_findbursts(
    isrising: &mut [bool; 16],
    zcs: &mut [f32; 16],
    burstarea: &[f32],
    i: usize,
    endburstarea: usize,
    threshold: f32,
    bstart: usize,
    s_rem: f64,
    zcburstdiv: f64,
    mut phase_adjust: f64,
) -> (usize, f64, usize) {
    let mut zc_count = 0usize;
    let mut rising_count = 0usize;
    let mut j = i;

    isrising.fill(false);
    zcs.fill(0.0);

    while j < endburstarea && zc_count < zcs.len() {
        if burstarea[j].abs() > threshold {
            let zc = calczc_do(burstarea, j, 0.0, 0, 16);
            if let Some(zc) = zc {
                isrising[zc_count] = burstarea[j] < 0.0;
                zcs[zc_count] = zc as f32;
                zc_count += 1;
                j = zc as usize + 1;
            } else {
                break;
            }
        } else {
            j += 1;
        }
    }

    if zc_count > 0 {
        // Computed over the full fixed-size arrays, exactly like the numba
        // original (zero-filled entries contribute their (near-integer)
        // cycle count).
        let zc_cycles: Vec<f64> = zcs
            .iter()
            .map(|&zc| (bstart as f64 + f64::from(zc) - s_rem) / zcburstdiv + phase_adjust)
            .collect();
        let zc_rounds: Vec<i64> = zc_cycles.iter().map(|&c| (c + 0.5) as i64).collect();
        let diffs: Vec<f64> = zc_rounds
            .iter()
            .zip(&zc_cycles)
            .map(|(&r, &c)| r as f64 - c)
            .collect();
        let mut diffs = diffs;
        phase_adjust += median_f64_(&mut diffs);
        rising_count = (0..16)
            .filter(|&k| isrising[k] != (zc_rounds[k] % 2 != 0))
            .count();
    }

    (zc_count, phase_adjust, rising_count)
}

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

fn diff(values: &[f64]) -> Vec<f64> {
    values.windows(2).map(|w| w[1] - w[0]).collect()
}

fn diff2(values: &[f64]) -> Vec<f64> {
    let d = diff(values);
    diff(&d)
}

/// Port of `LDdecode.detectLevels`: returns (sync_hz, ire0_hz, ire100_hz).
pub(crate) fn detect_levels(field: &Field) -> (f64, f64, f64) {
    let spec = &field.spec;
    let mut sync_hzs: Vec<f64> = Vec::new();
    let mut ire0_hzs: Vec<f64> = Vec::new();
    let mut ire100_hzs: Vec<f32> = Vec::new();

    for wl in spec
        .sys_ld_vits_whitelocs
        .iter()
        .map(|w| (w[0], w[1], w[2], 50))
        .chain(spec.sys_ld_vits_code_slices.iter().map(|w| (w[0], w[1], w[2], w[3])))
    {
        let (start, end) = field.lineslice(wl.0, Some(wl.1 as f64), Some(wl.2 as f64), None, 0.0);
        if start >= end || end > field.data.video.demod.len() {
            continue;
        }
        let cut = &field.data.video.demod[start..end];
        let mut sorted = cut.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // np.percentile default 'linear' method: index = (n-1)*p/100 with
        // linear interpolation between neighbors, computed in float32.
        let idx = (sorted.len() as f64 - 1.0) * (wl.3 as f64 / 100.0);
        let lo = idx.floor() as usize;
        let frac = idx - lo as f64;
        let freq = if lo + 1 < sorted.len() {
            let f = frac as f32;
            sorted[lo] + (sorted[lo + 1] - sorted[lo]) * f
        } else {
            sorted[lo]
        };
        // NOTE: the Python computes IRE against the *spec* levels (spec=True).
        let freq_ire = (f64::from(freq) - SYS_IRE0 as f64) / SYS_HZ_IRE as f64;

        if inrange(freq_ire, 95.0, 110.0) {
            ire100_hzs.push(freq);
        }
    }

    let output_lines = spec.output_lines();
    for l in 12..output_lines {
        let (sa, ea) = field.lineslice(l, Some(0.25), Some(4.0), None, 0.0);
        let begin_ire0 = spec.sys_color_burst_us[1];
        let end_ire0 = spec.sys_active_video_us[0];
        let (sb, eb) = field.lineslice(
            l,
            Some(begin_ire0 + 0.25),
            Some(end_ire0 - begin_ire0 - 0.5),
            None,
            0.0,
        );

        // compute wow adjustment
        let thislinelen = f64::from(
            field.linelocs[l + field.lineoffset] - field.linelocs[l + field.lineoffset - 1],
        );
        let adj = field.inlinelen as f64 / thislinelen;

        if inrange(adj, 0.98, 1.02) {
            if sa < ea && ea <= field.data.video.demod_05.len() {
                let med = median_f32(&mut field.data.video.demod_05[sa..ea].to_vec());
                if let Some(p) = std::env::var_os("LD_DUMP_LEVELS_RAW") {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                        let _ = writeln!(f, "# sync l={} sa={} ea={} adj={:.17} med={:.9}", l, sa, ea, adj, med);
                    }
                }
                sync_hzs.push(f64::from(med) / adj);
            }
            if sb < eb && eb <= field.data.video.demod_05.len() {
                let med = median_f32(&mut field.data.video.demod_05[sb..eb].to_vec());
                if let Some(p) = std::env::var_os("LD_DUMP_LEVELS_RAW") {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                        let _ = writeln!(f, "# ire0 l={} sb={} eb={} adj={:.17} med={:.9}", l, sb, eb, adj, med);
                    }
                }
                ire0_hzs.push(f64::from(med) / adj);
            }
        }
    }

    if let Some(p) = std::env::var_os("LD_DUMP_LEVELS_RAW") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
            let _ = writeln!(f, "# sync_hzs len={}", sync_hzs.len());
            for v in sync_hzs.iter() { let _ = writeln!(f, "{:.17}", v); }
            let _ = writeln!(f, "# ire0_hzs len={}", ire0_hzs.len());
            for v in ire0_hzs.iter() { let _ = writeln!(f, "{:.17}", v); }
            let _ = writeln!(f, "# ire100_hzs len={}", ire100_hzs.len());
            for v in ire100_hzs.iter() { let _ = writeln!(f, "{:.17}", v); }
        }
    }

    let vsync_hz = field.levels.iretohz(field.levels.vsync_ire);

    let m_synchz = if sync_hzs.is_empty() {
        vsync_hz
    } else {
        median_f64_(&mut sync_hzs)
    };
    let m_ire0hz = if ire0_hzs.is_empty() {
        field.levels.iretohz(0.0)
    } else {
        median_f64_(&mut ire0_hzs)
    };
    let m_ire100hz = if ire100_hzs.is_empty() {
        field.levels.iretohz(100.0)
    } else {
        f64::from(median_f32(&mut ire100_hzs))
    };

    (m_synchz, m_ire0hz, m_ire100hz)
}
