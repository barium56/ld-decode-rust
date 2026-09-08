//! Dropout detection (port of `Field.dropout_detect` and friends from the
//! Python ld-decode). Dropouts are located on the RF highpass signal (large
//! excursions), cross-checked against absurd fluctuations of the demodulated
//! video, and reported in TBC picture coordinates.

use crate::decode::field::{Field, STATE_EQPL1, STATE_EQPL2, STATE_HSYNC, STATE_VSYNC};
use rayon::prelude::*;

fn inrange(a: f64, mi: f64, ma: f64) -> bool {
    a >= mi && a <= ma
}

/// Exact port of numba's `np.std` on a float32 array (`array_var_impl` in
/// numba's arraymath): mean accumulated in f32 (serial), then a float64
/// accumulator over f32 squared deviations. Any other accumulation differs in
/// the last ulp and can flip dropout boundary samples.
fn std_f32(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let len = values.len() as f32;
    let mean = values.iter().fold(0.0f32, |acc, &v| acc + v) / len;
    let ssd = values.iter().fold(0.0f64, |acc, &v| {
        let d = v - mean; // f32
        let sq = d * d; // f32, matching numba's val*conj(val) in f32
        acc + sq as f64
    });
    (ssd / values.len() as f64).sqrt()
}

/// Port of `dropout_detect_demod`: a per-sample error map over the demodulated
/// window.
fn dropout_detect_demod(field: &Field) -> Vec<bool> {
    let sub = std::env::var_os("LD_SUBTIME2").is_some();
    let s0 = std::time::Instant::now();
    let spec = &field.spec;

    let rfhpf = &field.data.rfhpf;
    let rfstd = std_f32(rfhpf);
    let s_std = s0.elapsed().as_nanos() as u64;

    // Combined pass: the original builds an `iserr_rf1` flag vector and then
    // copies it `rotdelay` samples later (same contents, shifts only), so we
    // write into the rotated index directly — bit-identical `iserr`.
    let rotdelay = spec.delays.video_rot.max(0) as usize;
    let mut iserr = vec![false; rfhpf.len()];
    let thr = rfstd * 3.0;
    // Destination-index space: element `gi` is fed by rfhpf[gi - rotdelay], so
    // each chunk writes only its own slice (element-wise identical to the
    // serial loop — same comparisons, same values, no shared state). Python
    // compares the f32 array against the f64 scalar, promoting per-sample.
    let stride = iserr.len().max(1).div_ceil(rayon::current_num_threads() * 16).max(1024);
    iserr.par_chunks_mut(stride).enumerate().for_each(|(ci, chunk)| {
        let base = ci * stride;
        let lo = rotdelay;
        let hi = lo + rfhpf.len();
        for (i, flag) in chunk.iter_mut().enumerate() {
            let gi = base + i;
            if gi >= lo && gi < hi {
                let v = f64::from(rfhpf[gi - lo]);
                if v < -thr || v > thr {
                    *flag = true;
                }
            }
        }
    });

    let s_p1 = s0.elapsed().as_nanos() as u64;
    // Build sets of min/max valid levels.
    let iretohz = |ire: f64| field.levels.iretohz(ire);
    let demod = &field.data.video.demod;
    let demod_05 = &field.data.video.demod_05;
    let mut valid_min = vec![iretohz(-50.0) as f32; demod.len()];
    let valid_max = iretohz(160.0) as f32;
    let mut valid_min05 = vec![iretohz(-30.0) as f32; demod_05.len()];
    let valid_max05 = iretohz(115.0) as f32;

    let hsync_len = field.lt.hsync.1 as usize;
    let vsync_ire = -40.0; // SysParams (spec) vsync_ire, NTSC
    let sync_min = iretohz(vsync_ire - 35.0) as f32;
    let sync_min_05 = iretohz(vsync_ire - 10.0) as f32;

    let vsync_lines = field.get_vsync_lines();
    let n = field.linelocs.len();

    for l in 1..n.saturating_sub(1) {
        let start = field.linelocs[l] as usize;
        let end = if vsync_lines.contains(&l) {
            field.linelocs[l + 1] as usize
        } else {
            start + hsync_len
        };
        let end = end.min(valid_min.len());
        for i in start..end {
            valid_min[i] = sync_min;
            if i < valid_min05.len() {
                valid_min05[i] = sync_min_05;
            }
        }
    }

    let s_vm = s0.elapsed().as_nanos() as u64;
    // Absurd fluctuations in pre-deemp demod can only be caused by dropouts.
    let demod_raw = &field.data.video.demod_raw;
    let freq_hz_half = spec.freq_hz_half as f32;

    // Element-wise synthesis: every `gi` reads only its own index of the
    // demod arrays and writes its own flag — parallel chunks are bit-exact.
    let stride = iserr.len().max(1).div_ceil(rayon::current_num_threads() * 16).max(1024);
    iserr.par_chunks_mut(stride).enumerate().for_each(|(ci, chunk)| {
        let gi0 = ci * stride;
        for (i, flag) in chunk.iter_mut().enumerate() {
            let gi = gi0 + i;
            if gi < demod_raw.len() && demod_raw[gi] > freq_hz_half {
                *flag = true;
            }
            if gi < demod.len() && (demod[gi] < valid_min[gi] || demod[gi] > valid_max) {
                *flag = true;
            }
            if gi < demod_05.len() && (demod_05[gi] < valid_min05[gi] || demod_05[gi] > valid_max05) {
                *flag = true;
            }
        }
    });

    let s_p2 = s0.elapsed().as_nanos() as u64;
    // Filter out dropouts outside the actual field.
    let lo = field.linelocs.get(field.lineoffset + 1).copied().unwrap_or(0.0) as usize;
    let hi = field
        .linelocs
        .get(field.lineoffset + field.linecount.unwrap_or(0) + 1)
        .copied()
        .unwrap_or(iserr.len() as f64) as usize;
    for i in 0..lo.min(iserr.len()) {
        iserr[i] = false;
    }
    for i in hi..iserr.len() {
        iserr[i] = false;
    }

    if sub {
        eprintln!("SUBTIME2 std={:.3} p1={:.3} vmin={:.3} p2={:.3} ms", s_std as f64 / 1e6, s_p1 as f64 / 1e6, s_vm as f64 / 1e6, s_p2 as f64 / 1e6);
    }
    iserr
}

/// Port of `build_errlist`: merge individual error samples into (start, end)
/// runs.
fn build_errlist(field: &Field, errmap: &[usize]) -> Vec<(f64, f64)> {
    let lineoffset_start = field.linelocs[field.lineoffset] as usize;

    // First error at/after the field start.
    let firsterr = match errmap.iter().find(|&&e| e >= lineoffset_start) {
        Some(&e) => e,
        None => return Vec::new(),
    };
    let mut errlist: Vec<(f64, f64)> = Vec::new();
    let mut curerr = (firsterr as f64, firsterr as f64);

    for &e in errmap {
        if e as f64 > curerr.0 && e as f64 <= curerr.1 + 20.0 {
            let mut pad = (e as f64 - curerr.0) * 1.7;
            pad = pad.min(spec_freq(field) * 12.0);
            let epad = curerr.0 + pad;
            curerr = (curerr.0, epad);
        } else if e as f64 > firsterr as f64 {
            errlist.push((curerr.0 - 8.0, curerr.1 + 4.0));
            curerr = (e as f64, e as f64);
        }
    }
    errlist.push(curerr);

    errlist
}

fn spec_freq(field: &Field) -> f64 {
    field.spec.freq
}

/// Port of `dropout_errlist_to_tbc`: convert raw-data coordinates to TBC
/// coordinates, splitting multi-line dropouts. Returns (line, startx, endx).
fn dropout_errlist_to_tbc(field: &Field, errlist: &[(f64, f64)]) -> Vec<(usize, f64, f64)> {
    let mut dropouts: Vec<(usize, f64, f64)> = Vec::new();
    if errlist.is_empty() {
        return dropouts;
    }

    let lineoffset = -(field.lineoffset as i64);
    let mut errlistc: Vec<(f64, f64)> = errlist.to_vec();
    let mut curerr = errlistc.remove(0);

    // Remove dropouts before the start of the frame so they don't cause the
    // rest to be skipped.
    let field_start = field.linelocs[field.lineoffset] as f64;
    while !errlistc.is_empty() && curerr.0 < field_start {
        curerr = errlistc.remove(0);
    }

    for line in field.lineoffset..field.lineoffset + field.linecount.unwrap_or(0) {
        let l0 = field.linelocs[line] as f64;
        let l1 = field.linelocs[line + 1] as f64;
        while inrange(curerr.0, l0, l1) {
            let start_rf_linepos = curerr.0 - l0;
            let mut start_linepos = start_rf_linepos / (l1 - l0);
            start_linepos = (start_linepos * field.outlinelen as f64).trunc();

            let end_rf_linepos = curerr.1 - l0;
            let mut end_linepos = end_rf_linepos / (l1 - l0);
            end_linepos = (end_linepos * field.outlinelen as f64).round();

            let first_line = line as i64 + 1 + lineoffset;

            if end_linepos > field.outlinelen as f64 {
                let num_lines = (end_linepos / field.outlinelen as f64).floor() as usize;

                // First line.
                dropouts.push((first_line as usize, start_linepos, field.outlinelen as f64));
                // Full lines in the middle.
                for n in 0..num_lines - 1 {
                    dropouts.push((
                        (first_line + n as i64 + 1) as usize,
                        0.0,
                        field.outlinelen as f64,
                    ));
                }
                // Leftover on the last line.
                dropouts.push((
                    (first_line + num_lines as i64) as usize,
                    0.0,
                    (end_linepos % field.outlinelen as f64).abs(),
                ));
            } else {
                dropouts.push((first_line as usize, start_linepos, end_linepos));
            }

            if !errlistc.is_empty() {
                curerr = errlistc.remove(0);
            } else {
                curerr = (f64::NAN, f64::NAN);
            }
        }
    }

    dropouts
}

/// Port of `dropout_detect`: returns (fieldLine, startx, endx) arrays, with
/// `fieldLine` 0-based like the JSON output.
pub(crate) fn detect_dropouts(field: &Field) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let s5 = std::time::Instant::now();
    let iserr = dropout_detect_demod(field);
    let s_demod = s5.elapsed().as_nanos() as u64;
    let errmap: Vec<usize> = iserr
        .iter()
        .enumerate()
        .filter(|(_, &e)| e)
        .map(|(i, _)| i)
        .collect();

    let mut rv_lines = Vec::new();
    let mut rv_starts = Vec::new();
    let mut rv_ends = Vec::new();

    if !errmap.is_empty() && errmap[errmap.len() - 1] as f64 > field.linelocs[field.lineoffset] as f64 {
        let errlist = build_errlist(field, &errmap);
        for (line, start, end) in dropout_errlist_to_tbc(field, &errlist) {
            rv_lines.push(line.saturating_sub(1));
            rv_starts.push(start as usize);
            rv_ends.push(end as usize);
        }
    }

    if std::env::var_os("LD_SUBTIME2").is_some() {
        eprintln!("SUBTIME2 errmap+list={:.3} ms", (s5.elapsed().as_nanos() as u64 - s_demod) as f64 / 1e6);
    }
    (rv_lines, rv_starts, rv_ends)
}

/// The vblank state machine pulse types (re-exported for the decoder's
/// convenience).
#[allow(unused)]
pub(crate) const _STATE_HSYNC: u8 = STATE_HSYNC;
#[allow(unused)]
pub(crate) const _STATE_EQPL1: u8 = STATE_EQPL1;
#[allow(unused)]
pub(crate) const _STATE_VSYNC: u8 = STATE_VSYNC;
#[allow(unused)]
pub(crate) const _STATE_EQPL2: u8 = STATE_EQPL2;
