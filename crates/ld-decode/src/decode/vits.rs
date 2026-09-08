//! VITS metrics (port of `LDdecode.computeMetrics` / `computeMetricsNTSC` and
//! the `CombNTSC` helper class from the Python ld-decode). These are the SNR /
//! IRE measurements reported per field in the TBC JSON and used for the
//! automatic MTF level (black-to-white RF ratio).

use crate::decode::field::Field;
use crate::spec::DecoderSpec;

/// What the decoder needs from a picture source for the (partial) NTSC comb
/// used by `calcLine19Info`. For the current field this is the field's own
/// TBC picture; for the previous field it is the stored `PrevField`.
pub(crate) struct CombInput<'a> {
    pub dspicture: &'a [u16],
    pub field_phase_id: i64,
    pub out_scale: f64,
}

/// The two JSON-reported metrics plus the MTF-driving RF ratio.
pub(crate) struct VitsOutcome {
    pub w_snr: Option<f64>,
    pub b_psnr: Option<f64>,
    pub black_to_white_rf_ratio: Option<f64>,
    /// Debug: the RF levels that feed the ratio (std of raw slices).
    pub white_rf_level: Option<f64>,
    pub black_line_rf_level: Option<f64>,
}

/// `np.round(x * 10^places) / 10^places` (round half to even).
fn roundfloat(fl: f64, places: i64) -> f64 {
    let r = 10f64.powi(places as i32);
    (fl * r).round_ties_even() / r
}

fn inrange(a: f64, mi: f64, ma: f64) -> bool {
    a >= mi && a <= ma
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        // `np.mean` uses pairwise summation (numpy-exact).
        pairwise_sum_f64(values) / values.len() as f64
    }
}

fn std(values: &[f64]) -> f64 {
    let m = mean(values);
    if values.is_empty() {
        0.0
    } else {
        (values.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / values.len() as f64).sqrt()
    }
}

/// Port of Python's `output_to_ire` applied to a uint16 dspicture slice.
/// Python keeps numpy uint16 dtype for `output - outputZero`, so pixels below
/// outputZero wrap around (e.g. 0 - 1024 -> 64512) and inflate the mean IRE.
/// The VITS checks use this wrapped arithmetic; `calcpsnr` avoids it by
/// converting to float first ("underflows at -40IRE" comment in core.py).
fn output_to_ire_u16(field: &Field, v: u16) -> f64 {
    (f64::from(v.wrapping_sub(field.spec.sys_output_zero as u16)) / field.out_scale)
        + field.levels.vsync_ire
}

/// Port of `LDdecode.calcsnr`. `psnr=true` fixes the signal at 100 IRE.
fn calcsnr(field: &Field, slice: (usize, usize), psnr: bool) -> Option<f64> {
    if slice.1 <= slice.0 || slice.1 > field.dspicture.len() {
        return None;
    }
    let data: Vec<f64> = field.dspicture[slice.0..slice.1]
        .iter()
        .map(|&v| field.output_to_ire(f64::from(v)))
        .collect();
    let signal = if psnr { 100.0 } else { mean(&data) };
    let noise = std(&data);
    if noise <= 0.0 || signal <= 0.0 {
        return None;
    }
    Some(20.0 * (signal / noise).log10())
}

/// Port of `CombNTSC`: a *partial* NTSC comb filter, just enough for the VITS
/// line-19 measurements.
struct CombNtsc<'a> {
    input: CombInput<'a>,
    cbuffer: Vec<f32>,
}

impl<'a> CombNtsc<'a> {
    fn new(input: CombInput<'a>) -> Self {
        let data = input.dspicture;
        let mut cbuffer = vec![0.0f32; data.len()];
        // cbuffer[i] = (data[i-2] + data[i+2]) / 2 - data[i]
        for i in 2..data.len().saturating_sub(2) {
            cbuffer[i] =
                (f32::from(data[i - 2]) + f32::from(data[i + 2])) / 2.0 - f32::from(data[i]);
        }
        Self { input, cbuffer }
    }

    /// Whether line `line` has positive colour-burst phase (based on the field
    /// phase ID and line parity).
    fn getlinephase(&self, line: usize) -> bool {
        let field_id = self.input.field_phase_id;
        if line % 2 == 0 {
            field_id == 1 || field_id == 4
        } else {
            field_id == 2 || field_id == 3
        }
    }

    /// Port of `splitIQ_line`: split the combed buffer into normalized I and Q
    /// arrays, each half the length of `cbuffer`.
    fn split_iq_line(&self, cbuffer: &[f32], line: usize) -> (Vec<f32>, Vec<f32>) {
        let linephase = self.getlinephase(line);

        let mut si = Vec::with_capacity(cbuffer.len() / 2);
        let mut sq = Vec::with_capacity(cbuffer.len() / 2);
        for (i, &v) in cbuffer.iter().enumerate() {
            if i % 2 == 0 {
                sq.push(v);
            } else {
                si.push(v);
            }
        }

        if !linephase {
            for i in (0..si.len()).step_by(2) {
                si[i] = -si[i];
            }
            for i in (1..sq.len()).step_by(2) {
                sq[i] = -sq[i];
            }
        } else {
            for i in (1..si.len()).step_by(2) {
                si[i] = -si[i];
            }
            for i in (0..sq.len()).step_by(2) {
                sq[i] = -sq[i];
            }
        }

        (si, sq)
    }

    /// Port of `calcLine19Info`: colour-burst level, phase (ideally ~147 deg)
    /// and unfiltered SNR from line 19.
    fn calc_line19_info(
        &self,
        l19_slice: (usize, usize),
        l19_slice_i70: (usize, usize),
        comb_field2: Option<&CombNtsc>,
    ) -> (Option<f64>, Option<f64>, Option<f64>) {
        if l19_slice_i70.1 > self.input.dspicture.len() {
            return (None, None, None);
        }
        let ire_out1 = &self.input.dspicture[l19_slice_i70.0..l19_slice_i70.1];
        if ire_out1.is_empty() {
            return (None, None, None);
        }
        let max1 = *ire_out1.iter().max().unwrap();
        let min1 = *ire_out1.iter().min().unwrap();
        // fail out if there is obviously bad data
        if !(max1 < 100 && min1 > 40) {
            return (None, None, None);
        }

        let mut cbuffer = self.cbuffer[l19_slice.0..l19_slice.1].to_vec();

        if let Some(cp) = comb_field2 {
            if l19_slice.1 > cp.input.dspicture.len() {
                return (None, None, None);
            }
            let ire_out2 = &cp.input.dspicture[l19_slice_i70.0..l19_slice_i70.1];
            let max2 = *ire_out2.iter().max().unwrap();
            let min2 = *ire_out2.iter().min().unwrap();
            if !(max2 < 100 && min2 > 40) {
                return (None, None, None);
            }
            let cb2 = &cp.cbuffer[l19_slice.0..l19_slice.1];
            for (a, &b) in cbuffer.iter_mut().zip(cb2) {
                *a = (*a - b) / 2.0;
            }
        }

        let (si, sq) = self.split_iq_line(&cbuffer, 19);
        let sl = 110..230.min(si.len());
        let si_sl = &si[sl.clone()];
        let sq_sl = &sq[sl.clone()];

        let cdata: Vec<f64> = si_sl
            .iter()
            .zip(sq_sl)
            .map(|(&a, &b)| f64::from(a * a + b * b).sqrt())
            .collect();
        if cdata.is_empty() {
            return (None, None, None);
        }

        let si_mean = mean(&si_sl.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
        let sq_mean = mean(&sq_sl.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
        let mut phase = crate::spec::ucrt_atan2::call(si_mean, sq_mean) * 180.0 / std::f64::consts::PI;
        if phase < 0.0 {
            phase += 360.0;
        }

        let signal = mean(&cdata);
        let noise = std(&cdata);
        if noise <= 0.0 {
            return (None, None, None);
        }
        let snr = 20.0 * (signal / noise).log10();

        (
            Some(signal / (2.0 * self.input.out_scale)),
            Some(phase),
            Some(snr),
        )
    }
}

/// Compute the per-field VITS metrics (port of `computeMetrics` for NTSC).
/// `fp` is the previous field, used for the 3D comb measurements.
pub(crate) fn compute_vits_metrics(
    spec: &DecoderSpec,
    field: &Field,
    fp: Option<&crate::decode::field::PrevField>,
) -> VitsOutcome {
    let mut metrics: Vec<(&str, f64)> = Vec::new();

    // Check for a white flag - only on earlier discs, and only on first
    // "frame" fields.
    let wf_slice = field.lineslice_tbc(11, Some(15.0), Some(40.0));
    if wf_slice.1 <= field.dspicture.len() && wf_slice.1 > wf_slice.0 {
        let wf_ire: Vec<f64> = field.dspicture[wf_slice.0..wf_slice.1]
            .iter()
            .map(|&v| output_to_ire_u16(field, v))
            .collect();
        if inrange(mean(&wf_ire), 92.0, 108.0) {
            if let Some(snr) = calcsnr(field, wf_slice, true) {
                metrics.push(("ntscWhiteFlagSNR", snr));
            }
        }
    }

    // Line-19 colour burst level/phase/SNR (for MTF compensation later).
    let l19_slice = field.lineslice_tbc(19, Some(0.0), Some(40.0));
    let l19_slice_i70 = field.lineslice_tbc(19, Some(14.0), Some(18.0));
    let comb = CombNtsc::new(CombInput {
        dspicture: &field.dspicture,
        field_phase_id: field.field_phase_id,
        out_scale: field.out_scale,
    });
    let fp_comb = fp.map(|p| {
        CombNtsc::new(CombInput {
            dspicture: &p.dspicture,
            field_phase_id: p.field_phase_id,
            out_scale: p.out_scale,
        })
    });
    let (level, phase, snr) = comb.calc_line19_info(l19_slice, l19_slice_i70, fp_comb.as_ref());
    if let Some(phase) = phase {
        metrics.push(("ntscLine19ColorPhase", phase));
    }
    if let Some(snr) = snr {
        metrics.push(("ntscLine19ColorRawSNR", snr));
    }

    let ire50_slice = field.lineslice_tbc(19, Some(36.0), Some(10.0));
    if let Some(snr) = calcsnr(field, ire50_slice, true) {
        metrics.push(("greyPSNR", snr));
    }
    if ire50_slice.1 <= field.dspicture.len() && ire50_slice.1 > ire50_slice.0 {
        let grey_ire: Vec<f64> = field.dspicture[ire50_slice.0..ire50_slice.1]
            .iter()
            .map(|&v| output_to_ire_u16(field, v))
            .collect();
        metrics.push(("greyIRE", mean(&grey_ire)));
    }

    let ire50_rawslice = field.lineslice(19, Some(36.0), Some(10.0), None, 0.0);
    let rawdata = &field.data.input;
    let delay_white = spec.delays.video_white as i64;
    if let Some(raw) = slice_shifted(rawdata, ire50_rawslice, delay_white) {
        metrics.push(("greyRFLevel", std_from_int16_f64(&raw)));
    }

    if !field.is_first_field() {
        if let Some(level) = level {
            metrics.push(("ntscLine19Burst70IRE", level));
        }
        if let (Some(snr3d), Some(fp)) = (snr3d_of(&comb, &l19_slice, &l19_slice_i70, fp), fp) {
            metrics.push(("ntscLine19Color3DRawSNR", snr3d));
            let _ = fp;
            let sl_cburst = field.lineslice_tbc(19, Some(5.5), Some(2.4));
            if sl_cburst.1 <= field.dspicture.len() && sl_cburst.1 <= fp.dspicture.len() {
                let diff: Vec<f64> = field.dspicture[sl_cburst.0..sl_cburst.1]
                    .iter()
                    .zip(&fp.dspicture[sl_cburst.0..sl_cburst.1])
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)) / 2.0)
                    .collect();
                let rms = (diff.iter().map(|&v| v * v).sum::<f64>() / diff.len() as f64).sqrt();
                metrics.push(("ntscLine19Burst0IRE", std::f64::consts::SQRT_2 * rms / field.out_scale));
            }
        }
    }

    // FIXME: these should probably be computed in the Field class.
    let mut white_rf_level: Option<f64> = None;

    for &wl in &spec.sys_ld_vits_whitelocs {
        let wl_slice = field.lineslice_tbc(wl[0], Some(wl[1] as f64), Some(wl[2] as f64));
        if wl_slice.1 > field.dspicture.len() || wl_slice.1 <= wl_slice.0 {
            continue;
        }
        let wl_ire: Vec<f64> = field.dspicture[wl_slice.0..wl_slice.1]
            .iter()
            .map(|&v| output_to_ire_u16(field, v))
            .collect();
        if std::env::var("LD_DUMP_WL").is_ok() {
            let m = mean(&wl_ire);
            let mu16: f64 = field.dspicture[wl_slice.0..wl_slice.1]
                .iter()
                .map(|&v| f64::from(v))
                .sum::<f64>()
                / (wl_slice.1 - wl_slice.0) as f64;
            eprintln!(
                "fw={} rl={} l={:?} sl={}:{} dsp={} oscale={:.17} vsync={:.17} ozer={} mean_u16={:.6} ire={:.9} inrange={}",
                field.fields_written,
                field.readloc as i64,
                wl,
                wl_slice.0,
                wl_slice.1,
                field.dspicture.len(),
                field.out_scale,
                field.levels.vsync_ire,
                spec.sys_output_zero,
                mu16,
                m,
                inrange(m, 90.0, 110.0)
            );
        }
        if inrange(mean(&wl_ire), 90.0, 110.0) {
            if let Some(snr) = calcsnr(field, wl_slice, true) {
                metrics.push(("wSNR", snr));
            }
            metrics.push(("whiteIRE", mean(&wl_ire)));

            let rawslice = field.lineslice(wl[0], Some(wl[1] as f64), Some(wl[2] as f64), None, 0.0);
            if let Some(raw) = slice_shifted(rawdata, rawslice, delay_white) {
                white_rf_level = Some(std_from_int16_f64(&raw));
                metrics.push(("whiteRFLevel", white_rf_level.unwrap()));
            }
            break;
        }
    }

    let bl_slice = field.lineslice(
        spec.sys_blacksnr_slice[0],
        Some(spec.sys_blacksnr_slice[1] as f64),
        Some(spec.sys_blacksnr_slice[2] as f64),
        None,
        0.0,
    );
    let bl_slicetbc = field.lineslice_tbc(
        spec.sys_blacksnr_slice[0],
        Some(spec.sys_blacksnr_slice[1] as f64),
        Some(spec.sys_blacksnr_slice[2] as f64),
    );

    let delay_sync = spec.delays.video_sync as i64;
    if let Some(raw) = slice_shifted(rawdata, bl_slice, delay_sync) {
        metrics.push(("blackLineRFLevel", std_from_int16_f64(&raw)));
    }

    if bl_slice.1 <= field.data.video.demod.len() && bl_slice.1 > bl_slice.0 {
        let demod_mean = mean_f32(&field.data.video.demod[bl_slice.0..bl_slice.1]);
        metrics.push(("blackLinePreTBCIRE", field.levels.hztoire(f64::from(demod_mean))));
    }

    if bl_slicetbc.1 <= field.dspicture.len() && bl_slicetbc.1 > bl_slicetbc.0 {
        let dsp_mean = mean_f64(&field.dspicture[bl_slicetbc.0..bl_slicetbc.1]);
        metrics.push(("blackLinePostTBCIRE", field.output_to_ire(dsp_mean)));
    }

    if let Some(snr) = calcsnr(field, bl_slicetbc, true) {
        metrics.push(("bPSNR", snr));
    }

    if let Some(white_rf_level) = white_rf_level {
        if let Some((_, black_rf)) = metrics
            .iter()
            .find(|(k, _)| *k == "blackLineRFLevel")
        {
            metrics.push(("blackToWhiteRFRatio", black_rf / white_rf_level));
        }
    }

    // Round the JSON-reported values.
    let get = |key: &str| {
        metrics
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| roundfloat(*v, 1))
            .filter(|v| v.is_finite())
    };
    let ratio = metrics
        .iter()
        .find(|(k, _)| *k == "blackToWhiteRFRatio")
        .map(|(_, v)| roundfloat(*v, 4))
        .filter(|v| v.is_finite());

    VitsOutcome {
        w_snr: get("wSNR"),
        b_psnr: get("bPSNR"),
        black_to_white_rf_ratio: ratio,
        white_rf_level: metrics
            .iter()
            .find(|(k, _)| *k == "whiteRFLevel")
            .map(|(_, v)| *v),
        black_line_rf_level: metrics
            .iter()
            .find(|(k, _)| *k == "blackLineRFLevel")
            .map(|(_, v)| *v),
    }
}

/// The 3D-comb line-19 SNR: recompute with the previous field's comb engaged.
fn snr3d_of(
    comb: &CombNtsc,
    l19_slice: &(usize, usize),
    l19_slice_i70: &(usize, usize),
    fp: Option<&crate::decode::field::PrevField>,
) -> Option<f64> {
    let fp_comb = fp.map(|p| {
        CombNtsc::new(CombInput {
            dspicture: &p.dspicture,
            field_phase_id: p.field_phase_id,
            out_scale: p.out_scale,
        })
    });
    let (_, _, snr) = comb.calc_line19_info(*l19_slice, *l19_slice_i70, fp_comb.as_ref());
    snr
}

/// `values[start - shift .. stop - shift]` as owned data (Python slice with a
/// shifted start/stop); None when the range is invalid.
fn slice_shifted(values: &[f32], slice: (usize, usize), shift: i64) -> Option<Vec<f32>> {
    let start = slice.0 as i64 - shift;
    let stop = slice.1 as i64 - shift;
    if start < 0 || stop <= start || stop > values.len() as i64 {
        return None;
    }
    Some(values[start as usize..stop as usize].to_vec())
}

fn mean_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn mean_f64(values: &[u16]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64
    }
}

/// numpy-compatible pairwise summation in f64: sequential sums over
/// 128-element blocks, then recursively over the block sums. This mirrors
/// `np.sum`/`np.mean`/`np.var` bit-for-bit, which matters here because the
/// reference computes the VITS RF levels with `np.std` on **int16** data
/// (numpy upcasts integer input to float64).
/// Bit-exact port of numpy's `DOUBLE_pairwise_sum` (loops_utils.h.src):
/// - n < 8: sequential sum seeded from -0.0
/// - n <= 128 (PW_BLOCKSIZE): 8 accumulators seeded with the first 8 elements,
///   combine as ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)), then trailing elements
/// - else: recurse on halves split at n2 = (n/2) rounded down to a multiple of 8
/// Validated bit-exact vs bundled numpy 2.4.6 np.sum/np.mean/np.std (13.5k cases).
pub(crate) fn pairwise_sum_f64(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &v in values {
            res += v;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&values[..8]);
        let mut i = 8usize;
        let limit = n - (n % 8);
        while i < limit {
            r[0] += values[i];
            r[1] += values[i + 1];
            r[2] += values[i + 2];
            r[3] += values[i + 3];
            r[4] += values[i + 4];
            r[5] += values[i + 5];
            r[6] += values[i + 6];
            r[7] += values[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += values[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum_f64(&values[..n2]) + pairwise_sum_f64(&values[n2..])
}

/// Port of `np.std` on the (integer-valued) RF input: the samples are exact
/// integers (int16), so numpy upcasts to float64 and computes the standard
/// deviation there. `values` holds the same samples as f32 (exact).
fn std_from_int16_f64(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let xs: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    let mean = pairwise_sum_f64(&xs) / n;
    let sq: Vec<f64> = xs.iter().map(|&x| {
        let d = x - mean;
        d * d
    }).collect();
    (pairwise_sum_f64(&sq) / n).sqrt()
}

#[cfg(test)]
mod pairwise_tests {
    use super::*;

    // Golden data generated by work/validate_pairwise.py against bundled
    // numpy 2.4.6 (np.sum / np.std bit-exact, 13.5k randomized cases).
    include!("../../../../../work/pairwise_golden_data.rs");

    fn cases() -> Vec<(f64, f64, &'static [f64])> {
        let all: [(f64, f64, &'static [f64]); GOLDEN_COUNT] = [
            GOLDEN_0, GOLDEN_1, GOLDEN_2, GOLDEN_3, GOLDEN_4, GOLDEN_5,
            GOLDEN_6, GOLDEN_7, GOLDEN_8, GOLDEN_9, GOLDEN_10,
        ];
        all.to_vec()
    }

    #[test]
    fn pairwise_sum_matches_numpy() {
        for (i, (exp_sum, _, data)) in cases().iter().enumerate() {
            let got = pairwise_sum_f64(data);
            assert_eq!(
                got.to_bits(),
                exp_sum.to_bits(),
                "pairwise sum mismatch case {i} (n={})",
                data.len()
            );
        }
    }

    #[test]
    fn std_matches_numpy() {
        // Golden std only valid for exact-int16 data (f32 conversion exact).
        for (i, (_, exp_std, data)) in cases().iter().enumerate() {
            if data.iter().any(|&v| v != v.trunc()) {
                continue;
            }
            let f32s: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let got = std_from_int16_f64(&f32s);
            assert_eq!(
                got.to_bits(),
                exp_std.to_bits(),
                "std mismatch case {i} (n={})",
                data.len()
            );
        }
    }
}
