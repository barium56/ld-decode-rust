//! The decoder driver (port of `LDdecode.readfield` / `decodefield` /
//! `buildmetadata` / `writeout` and friends from the Python ld-decode).
//!
//! The `Decoder` decodes a stream of f32 RF samples window by window. Each
//! call to [`Decoder::decode`] is one "readfield" pass: it locates the next
//! field, demodulates the needed blocks, runs the field-level sync/chroma
//! pipeline, computes VITS metrics and dropouts, applies automatic MTF/AGC
//! corrections (re-decoding once when they move the levels), and returns the
//! fields ready to write to the TBC output. When the supplied window does not
//! yet cover the next field and more input may still arrive, decoding pauses
//! with its state intact and reports the earliest sample offset still needed.

mod audio;
mod demodblock;
#[cfg(test)]
mod probe_test;
mod dropouts;
mod efm_pll;
mod field;
mod vits;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use rayon::prelude::*;
use serde::Serialize;

use crate::request::ColorSystem;
use rustfft::num_complex::Complex64;
use crate::spec::{CalibLevels, DecoderSpec};
use audio::audio_phase2;
use demodblock::{BlockDecode, VideoChannels};
use efm_pll::EfmPll;
use field::{Field, FieldData, PrevField};

pub(crate) use demodblock::{compute_mtf_pow, demod_block_cpu, DemodSpecRef};

/// Size of one FFT demodulation block (matches `BLOCKSIZE` in core.py).
pub const BLOCKSIZE: usize = 32 * 1024;

pub(crate) use field::calczc;

// ---------------------------------------------------------------------------
// Public output types
// ---------------------------------------------------------------------------

/// The SNR metrics reported per field in the TBC JSON.
#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VitsMetrics {
    #[serde(rename = "wSNR", skip_serializing_if = "Option::is_none")]
    pub w_snr: Option<f64>,
    #[serde(rename = "bPSNR", skip_serializing_if = "Option::is_none")]
    pub b_psnr: Option<f64>,
}

/// Detected dropouts, in TBC picture coordinates.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DropOuts {
    pub field_line: Vec<usize>,
    pub startx: Vec<usize>,
    pub endx: Vec<usize>,
}

/// VBI (Philips code) data decoded from the field's line codes.
#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VbiData {
    pub vbi_data: Vec<i64>,
}

/// Per-field information, serialized into the TBC JSON `fields` array.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FieldInfoEntry {
    pub is_first_field: bool,
    pub sync_conf: i64,
    pub seq_no: usize,
    pub disk_loc: f64,
    pub file_loc: u64,
    #[serde(rename = "medianBurstIRE")]
    pub median_burst_ire: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drop_outs: Option<DropOuts>,
    #[serde(rename = "fieldPhaseID")]
    pub field_phase_id: i64,
    /// Only present on completed fields: the reference's buildmetadata returns
    /// early (with these keys never set) when it gives up on a field pair
    /// ("Skipped field"), so the JSON must omit them there.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_faults: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vits_metrics: Option<VitsMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vbi: Option<VbiData>,
    /// Number of stereo sample pairs written to the .pcm file for this field.
    pub audio_samples: usize,
    /// Number of EFM T-values written to the .efm file for this field.
    pub efm_t_values: usize,
    /// AC3 symbols (always 0: AC3 decoding is not ported).
    pub ac3_symbols: usize,
}

/// Decoder-level metadata for the JSON sidecar (port of `build_json`).
#[derive(Clone)]
pub struct DecoderMetadata {
    pub system: &'static str,
    pub number_of_sequential_fields: usize,
    pub field_width: usize,
    pub sample_rate: f64,
    pub black_16b_ire: f64,
    pub white_16b_ire: f64,
    pub blanking_16b_ire: f64,
    pub field_height: usize,
    pub colour_burst_start: i64,
    pub colour_burst_end: i64,
    pub active_video_start: i64,
    pub active_video_end: i64,
}

/// The luma picture of a decoded field.
#[derive(Clone)]
pub enum LumaOutput {
    /// 16-bit encoded picture (the normal TBC output).
    Encoded(Vec<u16>),
    /// Raw pre-encoding luma.
    Raw(Vec<f32>),
}

impl LumaOutput {
    pub fn encoded(&self) -> &[u16] {
        match self {
            LumaOutput::Encoded(v) => v,
            LumaOutput::Raw(_) => &[],
        }
    }

    pub fn raw(&self) -> &[f32] {
        match self {
            LumaOutput::Raw(v) => v,
            LumaOutput::Encoded(_) => &[],
        }
    }
}

/// A decoded field ready to be written: its metadata plus the TBC picture,
/// analog audio (.pcm) and EFM data (.efm).
#[derive(Clone)]
pub struct WriteableField {
    pub info: FieldInfoEntry,
    pub luma: LumaOutput,
    /// Interleaved int16 stereo audio samples for the .pcm file.
    pub audio: Vec<i16>,
    /// Raw EFM samples (i16) before the PLL; the PLL runs at write time.
    pub efm_raw: Vec<i16>,
    /// EFM T-values (through the PLL) for the .efm file.
    pub efm: Vec<i8>,
    /// Per-line locations (input samples) within the field.
    pub linelocs: Vec<f64>,
}

impl WriteableField {
    pub fn new(info: FieldInfoEntry, luma: LumaOutput) -> Self {
        Self {
            info,
            luma,
            audio: Vec::new(),
            efm_raw: Vec::new(),
            efm: Vec::new(),
            linelocs: Vec::new(),
        }
    }

    pub fn info(&self) -> &FieldInfoEntry {
        &self.info
    }

    pub fn luma(&self) -> &LumaOutput {
        &self.luma
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

struct MetadataFieldState {
    out_scale: f64,
    outlinecount: usize,
}

enum FieldOutcome {
    /// The window does not yet cover the next field; the value is the earliest
    /// absolute sample offset needed.
    NeedData(u64),
    /// (field, offset to the next decode). Both None means end of input.
    Done(Option<Field>, Option<f64>),
}



/// Per-field wall-clock accumulation for the LD_TIMING breakdown (temporary
/// instrumentation; nanos per phase, set by decode_field/decode).
#[derive(Clone, Copy)]
struct DbgTiming {
    demod: u64,
    asm_: u64,
    asm_insert: u64,
    asm_extend: u64,
    phase2: u64,
    prefetch: u64,
    pf_fold: u64,
    proc: u64,
    df_rest: u64,
    meta_dod: u64,
    meta_vits: u64,
}

/// An in-flight background prefetch demodulation (one per field). The window
/// blocks of the current field are already demodulated and inserted, so the
/// prefetch results can be computed on the rayon pool while the field's
/// serial tail runs; the results are folded into the cache (in the same order
/// the serial schedule would) at the start of the next `decode_field`, before
/// anything reads the cache. The worker only ever touches the copies it owns,
/// never the cache, so no locking is needed.
struct PendingPrefetch {
    /// The MTF the blocks were demodulated with (the cache memo).
    mtf: f64,
    recv: std::sync::mpsc::Receiver<Vec<(u64, BlockDecode)>>,
}

#[derive(Default)]
struct DbgMeta {
    dod: u64,
    vits: u64,
}

pub struct Decoder {
    spec: Arc<DecoderSpec>,
    dbg: DbgTiming,
    levels: CalibLevels,
    fdoffset: u64,
    fdoffset_frac: f64,
    fields_written: usize,
    fieldinfo: Vec<FieldInfoEntry>,
    /// The last valid field, used as the previous field of the next decode.
    prevfield: Option<PrevField>,
    /// Line codes of the last decoded *first* field, for VBI frame decoding.
    firstfield_linecode: Option<Vec<Option<i64>>>,
    lastvalidfield: [Option<WriteableField>; 2],
    /// fieldinfo index of the most recent json entry for each field parity
    /// (python aliases the stored fi dict, so a filler's efm re-process must
    /// update the original entry's efmTValues too).
    lastvalid_json_idx: [Option<usize>; 2],
    /// (fields_written, readloc) of the last written field, for the analog
    /// audio A/V-sync offset (port of `LDdecode.lastFieldWritten`).
    last_written: Option<(f64, u64)>,
    /// EFM PLL, carried across fields (the EFM track is continuous).
    efm_pll: EfmPll,
    /// Vits metrics computed in the decode loop for the current field, reused
    /// by buildmetadata (wSNR/bPSNR don't depend on the previous field).
    cached_vits: Option<vits::VitsOutcome>,
    frame_number: Option<i64>,
    // CLV/CAV VBI state.
    is_clv: bool,
    early_clv: bool,
    clv_minutes: Option<i64>,
    clv_seconds: Option<i64>,
    clv_frame_num: Option<i64>,
    lead_in: bool,
    lead_out: bool,
    bw_ratios: Vec<f64>,
    mtf_level: f64,
    /// Port of Python's DemodCache: demodulated blocks keyed by block number.
    /// A block is demodulated once with the then-current MTF and reused even
    /// as the MTF drifts, until a redo flushes the whole cache (Python's
    /// `flush_demod` on forceredo). `demod_cache_order` tracks first-insertion
    /// order so the cache can be FIFO-pruned to Python's `cachesize` (256).
    /// Blocks are stored behind an `Arc` (like Python's shared cache views) so
    /// hits and field assembly don't deep-copy ~1.3 MB per block.
    demod_cache: HashMap<u64, (f64, Arc<BlockDecode>)>,
    demod_cache_order: VecDeque<u64>,
    /// Optional dedicated thread pool for the background prefetch demod
    /// (LD_PF_POOL threads). When unset, prefetch runs on the global rayon
    /// pool and the main thread's later `par_iter` stages (downscale,
    /// dropouts, audio phase 2) join-block behind the long prefetch tasks;
    /// a dedicated pool lets both run concurrently.
    pf_pool: Option<rayon::ThreadPool>,
    /// Two most recent check_mtf levels, oldest first. Python starts the next
    /// field's decode thread *before* the current field's checkMTF runs, so a
    /// field is demodulated with a level that lags our serial loop by one
    /// check step: the field decoded at iteration k uses the level from
    /// iteration k-2. Redos always use the fresh level.
    mtf_hist: VecDeque<f64>,
    pending_prefetch: Option<PendingPrefetch>,
    metadata: Option<MetadataFieldState>,
    output_lines: usize,
    bytes_per_field: f64,
    readlen: usize,
    /// Analog audio output rate in Hz (0 disables .pcm output; negative means
    /// HSYNC-locked, matching ld-decode's `--ntsc_audio_rate`).
    analog_audio_freq: f64,
    /// Whether to produce EFM (.efm) output.
    digital_audio: bool,
}


impl Decoder {
    pub fn new(spec: Arc<DecoderSpec>, fdoffset: u64) -> Self {
        let output_lines = spec.output_lines();
        let bytes_per_field = spec.bytes_per_field() as f64;
        let linelen = spec.linelen;
        let readlen = ((linelen * 350) / 16384) * 16384;
        Self {
            spec,
            dbg: DbgTiming {
                demod: 0,
                asm_: 0,
                asm_insert: 0,
                asm_extend: 0,
                phase2: 0,
                prefetch: 0,
                pf_fold: 0,
                proc: 0,
                df_rest: 0,
                meta_dod: 0,
                meta_vits: 0,
            },
            levels: CalibLevels::defaults(),
            fdoffset,
            fdoffset_frac: 0.0,
            fields_written: 0,
            fieldinfo: Vec::new(),
            prevfield: None,
            firstfield_linecode: None,
            lastvalidfield: [None, None],
            lastvalid_json_idx: [None, None],
            last_written: None,
            efm_pll: EfmPll::new(),
            cached_vits: None,
            frame_number: None,
            is_clv: false,
            early_clv: false,
            clv_minutes: None,
            clv_seconds: None,
            clv_frame_num: None,
            lead_in: false,
            lead_out: false,
            bw_ratios: Vec::new(),
            mtf_level: 1.0,
            demod_cache: HashMap::new(),
            demod_cache_order: VecDeque::new(),
            // Dedicated prefetch pool: default mirrors the global pool's size
            // (so `-j` keeps meaning the total parallelism); `LD_PF_POOL=0`
            // falls back to the global pool, any other value overrides.
            pf_pool: match std::env::var("LD_PF_POOL")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&n| n > 0)
            {
                Some(n) => Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(n)
                        .build()
                        .expect("failed to build prefetch pool"),
                ),
                None if std::env::var("LD_PF_POOL")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    == Some(0) =>
                {
                    None
                }
                None => Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(rayon::current_num_threads())
                        .build()
                        .expect("failed to build prefetch pool"),
                ),
            },
            mtf_hist: VecDeque::from([1.0, 1.0]),
            pending_prefetch: None,
            metadata: None,
            output_lines,
            bytes_per_field,
            readlen,
            analog_audio_freq: 44100.0,
            digital_audio: true,
        }
    }

    /// Configure the analog audio output rate (Hz; 0 disables .pcm output).
    pub fn set_analog_audio(&mut self, freq: f64) {
        self.analog_audio_freq = freq;
    }

    /// Enable/disable EFM output.
    pub fn set_digital_audio(&mut self, enabled: bool) {
        self.digital_audio = enabled;
    }

    /// Port of the `lastfieldwritten` analog-audio A/V sync offset: how far
    /// into the current field's audio the previous write ended, so consecutive
    /// fields stay sample-aligned.
    fn audio_offset(&self, field: &Field, last_written: Option<(f64, u64)>) -> f64 {
        let Some((last_count, last_loc)) = last_written else {
            return 0.0;
        };
        if self.analog_audio_freq < 16000.0 {
            return 0.0;
        }
        let rf_samples_per_field = self.spec.freq_hz / (self.spec.sys_fps * 2.0);
        let read_gap = (field.readloc as f64 - last_loc as f64) / rf_samples_per_field;
        let field_number = (last_count + read_gap).round_ties_even();
        let mut linecount = (self.spec.sys_field_lines[0] + self.spec.sys_field_lines[1]) as f64
            * (field_number / 2.0).floor();
        if !field.is_first_field() {
            linecount += self.spec.sys_field_lines[0] as f64;
        }
        let samples_per_line = (self.spec.sys_line_period / 1e6) / (1.0 / self.analog_audio_freq);
        let audsamp_count = linecount * samples_per_line;
        let audsamp_offset = audsamp_count - audsamp_count.floor();
        let out = if audsamp_offset > 0.5 {
            (1.0 - audsamp_offset) * (1.0 / self.analog_audio_freq)
        } else {
            -audsamp_offset * (1.0 / self.analog_audio_freq)
        };
        if let Some(p) = std::env::var_os("LD_DUMP_AUDIO") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                let _ = writeln!(f, "# AO startloc={} readloc={} lw=({}, {}) rfspf={} read_gap={} field_number={} linecount={} spl={} audsamp_count={} audsamp_offset={} out={}", field.data.startloc, field.readloc, last_count, last_loc, rf_samples_per_field, read_gap, field_number, linecount, samples_per_line, audsamp_count, audsamp_offset, out);
            }
        }
        out
    }

    pub fn spec(&self) -> &DecoderSpec {
        &self.spec
    }

    fn fdoffset_abs(&self) -> f64 {
        self.fdoffset as f64 + self.fdoffset_frac
    }

    fn advance_fdoffset(&mut self, offset: f64) {
        let advanced = self.fdoffset_frac + offset;
        let whole = advanced.floor();
        self.fdoffset = self.fdoffset.saturating_add_signed(whole as i64);
        self.fdoffset_frac = advanced - whole;
    }

    /// Decode fields from the window `data` (absolute samples starting at
    /// `data_start`). Returns (earliest input offset no longer needed, fields
    /// produced). `final_chunk` marks the true end of input.
    pub fn decode(
        &mut self,
        data: &[f32],
        data_start: u64,
        final_chunk: bool,
    ) -> Result<(u64, Vec<WriteableField>)> {
        let mut output: Vec<WriteableField> = Vec::new();

        let mut done = false;
        let mut adjusted = false;
        let mut redo: Option<f64> = None;
        let mut f: Option<Field> = None;
        let mut picture: Option<LumaOutput> = None;

        let mut t_iter = std::time::Instant::now();
        let mut t_down: u64 = 0;
        let mut t_vits: u64 = 0;
        let mut t_meta: u64 = 0;
        let mut t_out: u64 = 0;
        while !done {
            t_iter = std::time::Instant::now();
            let redo_flag = redo.is_some();
            // Python starts the next field's decode thread *before* the
            // current field's checkMTF runs, so the MTF applied to a field
            // lags our serial loop by one check step (iteration k uses the
            // level from iteration k-2). Redos always use the fresh level.
            let mtf_level = if redo_flag {
                self.mtf_level
            } else {
                *self.mtf_hist.front().unwrap_or(&self.mtf_level)
            };
            let (start, prevfield) = if let Some(redo_loc) = redo {
                (redo_loc, self.prevfield.clone())
            } else {
                (self.fdoffset_abs(), self.prevfield.clone())
            };
            let outcome = self.decode_field(
                data,
                data_start,
                final_chunk,
                start,
                prevfield,
                redo_flag,
                mtf_level,
            )?;

            let (mut field, offset) = match outcome {
                FieldOutcome::NeedData(needed) => return Ok((needed, output)),
                FieldOutcome::Done(f, offset) => (f, offset),
            };

            if redo_flag {
                // Only one redo, no matter what.
                done = true;
                redo = None;
            }

            // Process the previous run.
            if field.is_some() {
                // Python's `getpulses` recalibrates the *shared* RFDecode
                // params in place, so a no-pulses retry (lead-in junk) leaves
                // the recalibrated ire0 visible to every later field. Rust's
                // Field owns a working copy, so mirror the update back to the
                // decoder here; otherwise each new read window restarts from
                // the default levels and the lead-in lock diverges from the
                // reference (extra "skip one second" steps, later first
                // field, different AGC calibration).
                if let Some(f) = field.as_ref() {
                    self.levels = f.levels;
                }
                let off = offset.unwrap_or(0.0);
                self.advance_fdoffset(off);
            }

            // Mirror Python's `prevfield = f if (f and f.valid) else None`:
            // an invalid or absent field must clear the previous-field hint,
            // otherwise `get_line0` keeps voting with a stale `line0loc_prev`
            // (computed from a valid field far behind the current junk
            // window) and the resulting negative nextfieldoffset rewinds the
            // decode into an endless re-grind of the same junk region.
            if field.as_ref().is_none_or(|f| !f.valid) {
                self.prevfield = None;
            }

            if let Some(field) = field.as_mut() {
                if field.valid {
                    // Track the VBI frame number on every valid field so `--seek`
                    // can find a target frame while discarding output.
                    self.update_frame_number(field);

                    // Downscaling is time consuming, but there is no decode
                    // thread to overlap it with in this serial port.
                    let t_d0 = std::time::Instant::now();
                    let audio_offset = self.audio_offset(field, self.last_written);
                    let luma = field.downscale(
                        self.output_lines,
                        self.spec.sys_outlinelen,
                        0,
                        true,
                        self.analog_audio_freq,
                        audio_offset,
                    )?;
                    // downscale(final_=true) already encoded the luma into
                    // field.dspicture with the same levels; reuse it instead of
                    // recomputing the identical 1M-sample conversion.
                    picture = Some(LumaOutput::Encoded(field.dspicture.clone()));
                    self.metadata = Some(MetadataFieldState {
                        out_scale: field.out_scale,
                        outlinecount: field.outlinecount,
                    });

                    t_down += t_d0.elapsed().as_nanos() as u64;
                    let t_v0 = std::time::Instant::now();
                    let metrics = vits::compute_vits_metrics(&self.spec, field, None);
                    // Cache for buildmetadata: wSNR/bPSNR don't depend on the
                    // previous field, so the second call in buildmetadata is a
                    // pure duplicate for this field.
                    self.cached_vits = Some(metrics.clone());
                    if let Some(p) = std::env::var_os("LD_DUMP_BWRATIO") {
                        use std::io::Write;
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                            let _ = writeln!(f, "{} {:.17} {:.17} {:?} {:?}", self.fields_written, metrics.black_to_white_rf_ratio.unwrap_or(-1.0), self.mtf_level, metrics.white_rf_level, metrics.black_line_rf_level);
                        }
                    }
                    if let Some(ratio) = metrics.black_to_white_rf_ratio {
                        // Python only records the ratio on the first pass (it
                        // runs inside `if ... and adjusted is False`); the
                        // redo pass would otherwise double-count the same
                        // field and skew the MTF calibration.
                        if !adjusted {
                            let keep = if self.is_clv { 900 } else { 30 };
                            self.bw_ratios.push(ratio);
                            if self.bw_ratios.len() > keep {
                                self.bw_ratios.drain(..self.bw_ratios.len() - keep);
                            }
                            if let Some(p) = std::env::var_os("LD_DUMP_ISCLV") {
                                use std::io::Write;
                                if let Ok(mut f) = std::fs::OpenOptions::new()
                                    .create(true)
                                    .append(true)
                                    .open(&p)
                                {
                                    let _ = writeln!(
                                        f,
                                        "fw={} is_clv={} keep={} rl={} lc={:?}",
                                        self.fields_written,
                                        self.is_clv,
                                        keep,
                                        self.fdoffset_abs(),
                                        field.linecode
                                    );
                                }
                            }
                        }
                    }

                    redo = if !self.check_mtf() {
                        Some(self.fdoffset_abs() - offset.unwrap_or(0.0))
                    } else {
                        None
                    };

                    t_vits += t_v0.elapsed().as_nanos() as u64;
                    let t_v1 = std::time::Instant::now();

                    // Perform AGC changes on first fields only to prevent luma
                    // mismatch intra-field.
                    if self.spec.use_agc && field.is_first_field() && field.sync_confidence > 80 {
                        let (sync_hz, ire0_hz, ire100_hz) = field::detect_levels(field);

                        let actual_white_ire = self.levels.hztoire(ire100_hz);
                        let sync_ire_diff =
                            (self.levels.hztoire(sync_hz) - self.levels.vsync_ire).abs();
                        let whitediff =
                            (self.levels.hztoire(ire100_hz) - actual_white_ire).abs();
                        let ire0_diff = self.levels.hztoire(ire0_hz).abs();

                        let acceptable_diff = if self.fields_written > 0 { 2.0 } else { 0.5 };

                        if whitediff.max(ire0_diff).max(sync_ire_diff) > acceptable_diff {

                            let hz_ire = (ire100_hz - ire0_hz) / 100.0;
                            let vsync_ire = (sync_hz - ire0_hz) / hz_ire;

                            if vsync_ire > -20.0 {
                                tracing::warn!(
                                    "At field #{}, Auto-level detection malfunction (vsync IRE computed at {}, nominal ~= -40), possible disk skipping",
                                    self.fieldinfo.len(),
                                    vsync_ire
                                );
                            } else {
                                let redo_to = self.fdoffset_abs() - offset.unwrap_or(0.0);
                                if std::env::var_os("LD_TRACE_AGC").is_some() {
                                    crate::teeprintln!("AGC f#{} fdoffset={} offset={} redo_to={} ire0={:.3} hz_ire={:.3} vsync_ire={:.3}", self.fieldinfo.len(), self.fdoffset_abs(), offset.unwrap_or(0.0), redo_to, ire0_hz, hz_ire, vsync_ire);
                                }
                                redo = Some(redo_to);
                                self.levels.ire0 = ire0_hz;
                                self.levels.hz_ire = hz_ire;
                                self.levels.vsync_ire = vsync_ire;
                            }
                        }
                    }

                    t_vits += t_v1.elapsed().as_nanos() as u64;

                    if !adjusted && redo.is_some() {
                        adjusted = true;
                        let r = redo.unwrap();
                        self.fdoffset = r.floor() as u64;
                        self.fdoffset_frac = r - r.floor();
                    } else {
                        done = true;
                        let fieldlength =
                            (field.linelocs[self.output_lines] - field.linelocs[0])
                                / field.inlinelen as f64;
                        if field.sync_confidence < 50
                            && !inrange(
                                fieldlength,
                                self.output_lines as f64 - 2.0,
                                self.output_lines as f64 + 2.0,
                            )
                        {
                            tracing::warn!(
                                "Possible player skip detected - check output (field length {} lines)",
                                fieldlength
                            );
                        }
                    }
                }
            }

            // Advance the MTF history. A redo re-decodes with the fresh level
            // and resets the history so the next field (whose thread Python
            // launches with the current level) also uses it; otherwise the
            // normal one-step lag continues.
            if redo_flag {
                self.mtf_hist = VecDeque::from([self.mtf_level, self.mtf_level]);
            } else {
                self.mtf_hist.pop_front();
                self.mtf_hist.push_back(self.mtf_level);
            }

            if field.is_none() && offset.is_none() {
                // EOF, probably.
                return Ok((self.fdoffset, output));
            }

            f = field;
        }

        let Some(mut field) = f else {
            return Ok((self.fdoffset, output));
        };
        if !field.valid {
            return Ok((self.fdoffset, output));
        }

        // Only write a FirstField first.
        if self.fieldinfo.is_empty() && !field.is_first_field() {
            // Python sets `prevfield = f` here too (bottom of the decode loop)
            // and returns without running buildmetadata, so the next field
            // sees the post-getLine0 confidence (no syncconf min yet).
            self.prevfield = Some(field.to_prevfield());
            return Ok((self.fdoffset, output));
        }

        let t_m0 = std::time::Instant::now();
        let (fi, need_filler) = self.buildmetadata(&mut field)?;
        // Python's `prevfield` is the live field object, so the next field
        // reads sync_confidence AFTER compute_syncconf has min'd it with the
        // line-spacing score. Snapshot here, post-buildmetadata, to match.
        self.prevfield = Some(field.to_prevfield());
        t_meta += t_m0.elapsed().as_nanos() as u64;
        let idx = usize::from(field.is_first_field());
        let luma = picture.unwrap_or(LumaOutput::Encoded(Vec::new()));
        let mut wf = WriteableField::new(fi, luma);
        wf.audio = std::mem::take(&mut field.dsaudio);
        wf.efm_raw = std::mem::take(&mut field.efmout);
        wf.linelocs = field.linelocs.clone();
        self.lastvalidfield[idx] = Some(wf);

        let t_o0 = std::time::Instant::now();
        if need_filler {
            if let Some(other) = self.lastvalidfield[1 - idx].clone() {
                let orig_idx = self.lastvalid_json_idx[1 - idx];
                output.push(self.writeout(other)?);
                // Python's filler re-processes the stored field's efm through
                // the shared PLL and stores the result in the *same* fi dict
                // that was appended when the field was first written, so the
                // original json entry's efmTValues is overwritten with the
                // re-processed length. Mirror that aliasing here.
                if let Some(orig) = orig_idx {
                    let new_len = self.fieldinfo.last().map(|e| e.efm_t_values).unwrap_or(0);
                    self.fieldinfo[orig].efm_t_values = new_len;
                }
            }
            if let Some(current) = self.lastvalidfield[idx].clone() {
                output.push(self.writeout(current)?);
                self.lastvalid_json_idx[idx] = Some(self.fieldinfo.len() - 1);
            }
        } else if let Some(current) = self.lastvalidfield[idx].clone() {
            // Mirror LDdecode: the A/V-sync offset is computed from the last
            // written field, recorded before this write.
            self.last_written = Some((self.fields_written as f64, current.info.file_loc));
            output.push(self.writeout(current)?);
            self.lastvalid_json_idx[idx] = Some(self.fieldinfo.len() - 1);
        }
        t_out += t_o0.elapsed().as_nanos() as u64;

        if self.lead_out {
            // Python's main loop stops the decode right after the field that
            // completed a frame with two lead-out (0x80EEEE) Philips codes
            // (`ldd.leadOut` in main.py), so nothing further is decoded or
            // written. The current field is already in `output`.
            return Ok((self.fdoffset, output));
        }

        if std::env::var_os("LD_TIMING").is_some() {
            let dbg = self.dbg;
            let df_total =
                dbg.demod + dbg.asm_ + dbg.phase2 + dbg.prefetch + dbg.pf_fold + dbg.proc + dbg.df_rest;
            let t_iter_total = t_iter.elapsed().as_nanos() as u64;
            let rest = t_iter_total
                .saturating_sub(df_total + t_down + t_vits + t_meta + t_out);
            let ms = |n: u64| n as f64 / 1e6;
            crate::teeprintln!(
                "TIMING fw={} decf_total={:.2} demod={:.2} asm={:.2} asmI={:.2} asmE={:.2} phase2={:.2} prefetch={:.2} pffold={:.2} proc={:.2} dfrest={:.2} down={:.2} vits={:.2} meta={:.2} metaD={:.2} metaV={:.2} out={:.2} rest={:.2} iter={:.2}",
                self.fields_written,
                ms(df_total),
                ms(dbg.demod),
                ms(dbg.asm_),
                ms(dbg.asm_insert),
                ms(dbg.asm_extend),
                ms(dbg.phase2),
                ms(dbg.prefetch),
                ms(dbg.pf_fold),
                ms(dbg.proc),
                ms(dbg.df_rest),
                ms(t_down),
                ms(t_vits),
                ms(t_meta),
                ms(dbg.meta_dod),
                ms(dbg.meta_vits),
                ms(t_out),
                ms(rest),
                ms(t_iter_total),
            );
        }

        Ok((self.fdoffset, output))
    }

    /// Insert one demodulated block into the cache with Python's DemodCache
    /// semantics (memo + FIFO order). Identical across the window, serial
    /// prefetch and async-fold paths so the cache state each field observes
    /// never depends on which path inserted.
    fn cache_insert(&mut self, bnum: u64, mtf: f64, bd: BlockDecode) -> Arc<BlockDecode> {
        if std::env::var_os("LD_DUMP_CACHE").is_some() {
            use std::io::Write;
            if let Some(p) = std::env::var_os("LD_DUMP_CACHE") {
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    let _ = writeln!(f, "{} {:.17}", bnum, mtf);
                }
            }
        }
        let arc = Arc::new(bd);
        self.demod_cache.insert(bnum, (mtf, Arc::clone(&arc)));
        self.demod_cache_order.push_back(bnum);
        arc
    }

    /// FIFO-prune the cache to Python's `cachesize` (256).
    fn prune_cache(&mut self) {
        while self.demod_cache_order.len() > 256 {
            if let Some(old) = self.demod_cache_order.pop_front() {
                self.demod_cache.remove(&old);
            }
        }
    }

    /// Fold the previous field's background prefetch results into the cache.
    /// Called at the top of every `decode_field`, before the redo flush and
    /// before any window lookup, so the cache is always in the state the
    /// serial schedule would have produced (window inserts of field k, then
    /// prefetch inserts of field k, in that order).
    fn fold_pending_prefetch(&mut self) {
        let Some(p) = self.pending_prefetch.take() else {
            return;
        };
        match p.recv.recv() {
            Ok(results) => {
                for (bnum, bd) in results {
                    self.cache_insert(bnum, p.mtf, bd);
                }
                self.prune_cache();
            }
            Err(_) => {
                // The worker panicked; the blocks are simply not cached and
                // get demodulated on demand like any other miss.
            }
        }
    }

    /// Demodulate and process one field starting at absolute sample offset
    /// `start`.
    fn decode_field(
        &mut self,
        data: &[f32],
        data_start: u64,
        final_chunk: bool,
        start: f64,
        prevfield: Option<PrevField>,
        redo: bool,
        mtf_level: f64,
    ) -> Result<FieldOutcome> {
        let t_df0 = std::time::Instant::now();
        self.dbg.pf_fold = 0;
        let t_fold0 = std::time::Instant::now();
        self.fold_pending_prefetch();
        self.dbg.pf_fold = t_fold0.elapsed().as_nanos() as u64;
        // Owned Arc so the `&mut self` cache methods below don't fight this
        // borrow; the refcount bump per field is negligible.
        let spec = Arc::clone(&self.spec);
        if redo {
            // Python's `flush_demod` on forceredo: drop every cached demod so
            // the whole window is re-demodulated at the current MTF.
            self.demod_cache.clear();
            self.demod_cache_order.clear();
        }

        // Port of `decodefield`: `readloc` and `numblocks` use the LDdecode
        // `self.blocksize` field, which equals `rf.blocklen` (32768), while the
        // DemodCache internally re-floors the request onto ITS `blocksize`
        // (blocklen - blockcut - blockcut_end = 31712) grid. Keeping the two
        // grids separate matters: the blocklen-floored begin plus the
        // DemodCache's ceil-to-blocksize expansion reads one extra block
        // (31 here) that a pure-blocksize computation misses.
        let blocklen = spec.blocklen as u64;
        let blocksize = spec.blocksize as u64;
        let readloc = ((start - spec.blockcut as f64) as i64).max(0) as u64;
        let readloc_block = (readloc / blocklen) as usize;
        let numblocks = (self.readlen as u64 / blocklen) + 2;
        let begin = readloc_block as u64 * blocklen;
        if std::env::var_os("LD_DBG_ALL").is_some() {
            crate::teeprintln!("DBG start={:.1} fdoffset={} readloc={} readloc_block={} begin={} blocklen={} blocksize={} readlen={} data_start={} window_len={} final={}",
                start, self.fdoffset, readloc, readloc_block, begin, spec.blocklen, spec.blocksize, self.readlen, data_start, data.len(), final_chunk);
        }
        // DemodCache.read(begin, numblocks * blocklen): block b sits at
        // `b * blocksize` for b in [begin/blocksize, end/blocksize].
        let first_block = (begin / blocksize) as usize;
        let end = begin + numblocks as u64 * blocklen;
        let numblocks_read = (end / blocksize) as usize - first_block + 1;
        let block_begin = first_block as u64 * blocksize;

        // The window must cover [block_begin, first_block+numblocks_read) in
        // `blocksize` units, plus the DemodCache prefetch range ahead of the
        // window: Python prefetches `prefetch` blocks at the current MTF on
        // every read, and those cached blocks are what later fields reuse, so
        // we must demodulate them now (with this field's MTF) or the reuse
        // schedule diverges.
        let prefetch_blocks = (self.bytes_per_field * 4.0 / spec.blocksize as f64) as usize + 4;
        let needed_end = (first_block + numblocks_read + prefetch_blocks) as u64 * blocksize;
        if needed_end > data_start + data.len() as u64 && !final_chunk {
            return Ok(FieldOutcome::NeedData(block_begin as u64));
        }

        let levels = self.levels;
        let dspec = DemodSpecRef::with_plans(
            spec.freq,
            spec.freq_half,
            spec.freq_hz,
            spec.blocklen,
            &spec.filters,
            &levels,
        );
        let mtf = (mtf_level * spec.mtf_mult + spec.mtf_offset) * spec.dp_mtf_basemult;
        if let Some(p) = std::env::var_os("LD_DUMP_MTF") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                let _ = writeln!(f, "# fields_written={} mtf_level={:.17} mtf={:.17}", self.fields_written, mtf_level, mtf);
            }
        }

        // Pre-reserve the exact per-field channel sizes so the serial assembly
        // extends below never reallocate (pure capacity hint, no behavior
        // change).
        let per_block = spec.blocklen - spec.blockcut - spec.blockcut_end;
        let audiob = spec.blocklen / spec.filters.audio_fdiv.max(1) + 2;
        // Pre-sized FieldData; the per-block assembly below fills it. Under
        // the async-prefetch schedule the window blocks are already cached,
        // so this buffer is only produced once per field (reassigned from
        // the assembled `raw` inside the !reached_eof block).
        let mut rawdecode = FieldData {
            input: Vec::with_capacity(numblocks_read * per_block),
            video: VideoChannels {
                demod: Vec::with_capacity(numblocks_read * per_block),
                demod_raw: Vec::with_capacity(numblocks_read * per_block),
                demod_05: Vec::with_capacity(numblocks_read * per_block),
                demod_burst: Vec::with_capacity(numblocks_read * per_block),
                audio: [
                    Vec::with_capacity(numblocks_read * audiob),
                    Vec::with_capacity(numblocks_read * audiob),
                ],
                efm: Vec::with_capacity(numblocks_read * per_block),
            },
            rfhpf: Vec::with_capacity(numblocks_read * per_block),
            audio: [Vec::new(), Vec::new()],
            efm: Vec::new(),
            startloc: block_begin as u64,
        };

        let mut reached_eof = false;

        // The first demod block's offset within the window. All blocks are
        // contiguous at `blocksize` stride from here, so a single range check
        // covers every block (the `needed_end` check above guarantees the
        // window holds the whole range unless we're at EOF).
        let first_in_window = first_block as i64 * blocksize as i64 - data_start as i64;
        if let Some(p) = std::env::var_os("LD_DUMP_WINDOW") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                let _ = writeln!(f, "# fw={} start={:.3} readloc={} begin={} end={} first_block={} numblocks_read={} blocklen={} blocksize={} readlen={} mtf={:.17}", self.fields_written, start, readloc, begin, end, first_block, numblocks_read, spec.blocklen, spec.blocksize, self.readlen, mtf);
            }
        }
        if std::env::var_os("LD_DBG_ALL").is_some() {
            crate::teeprintln!("DBG2 begin={} first_block={} block_begin={} numblocks_read={} first_in_window={}",
                begin, first_block, block_begin, numblocks_read, first_in_window);
        }
        if first_in_window < 0 {
            // The window doesn't start early enough for this block; ask the
            // caller to keep data from `block_begin` on (mirrors the
            // DemodCache being unable to serve the range, which would
            // otherwise stall the decode loop).
            return Ok(FieldOutcome::NeedData(block_begin));
        }
        if first_in_window as usize + numblocks_read * spec.blocksize > data.len() {
            // True end of input (the requested blocks extend past EOF), like
            // the DemodCache returning None.
            reached_eof = true;
        }

        if !reached_eof {
            // Demodulate the blocks, reusing Python's DemodCache semantics: a
            // block is demodulated once (with the then-current MTF) and then
            // reused unconditionally until a redo flushes the whole cache. New
            // blocks are demodulated at the current MTF.
            let mut decoded: Vec<Option<Arc<BlockDecode>>> = Vec::with_capacity(numblocks_read);
            let mut window_mtfs: Vec<(u64, f64)> = Vec::with_capacity(numblocks_read);
            // Python's DemodCache re-demodulates a cached block when its
            // stored MTF deviates from the current target by more than
            // `MTF_tolerance` (0.05); reuse is only unconditional within that
            // band. Mirror exactly: an out-of-band block is treated as a miss
            // and re-demodulated at the current MTF.
            const MTF_TOLERANCE: f64 = 0.05;
            for i in 0..numblocks_read {
                let bnum = first_block as u64 + i as u64;
                if let Some((cached_mtf, cached)) = self.demod_cache.get(&bnum) {
                    if (*cached_mtf - mtf).abs() <= MTF_TOLERANCE {
                        decoded.push(Some(Arc::clone(cached)));
                        window_mtfs.push((bnum, *cached_mtf));
                    } else {
                        decoded.push(None);
                        window_mtfs.push((bnum, mtf));
                    }
                } else {
                    decoded.push(None);
                    window_mtfs.push((bnum, mtf));
                }
            }
            {
                let dump_cache = std::env::var_os("LD_DUMP_CACHE");
                let rl_filter = std::env::var("LD_DUMP_CACHE_RL").unwrap_or_default();
                let rl_ok = rl_filter
                    .split(',')
                    .any(|s| s.parse::<u64>().ok() == Some(readloc));
                if let Some(p) = dump_cache {
                    if rl_ok {
                        use std::io::Write;
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                            let _ = writeln!(f, "# rl={} cache_len={} order_len={} min={:?} max={:?}", readloc, self.demod_cache.len(), self.demod_cache_order.len(), self.demod_cache_order.front(), self.demod_cache_order.back());
                            for (bnum, bmtf) in &window_mtfs {
                                let _ = writeln!(f, "{} {:.17}", bnum, bmtf);
                            }
                        }
                    }
                }
            }
            // Collect the (block number, window offset) pairs that are not in
            // the cache, demodulate them in parallel (each block is an
            // independent pure function of its input slice, MTF and spec, so
            // the result per block is deterministic), then insert and assemble
            // serially in index order exactly like the serial port so the
            // cache/prune schedule and the output stay bit-identical.
            let missing: Vec<(u64, usize)> = (0..numblocks_read)
                .filter(|&i| decoded[i].is_none())
                .map(|i| {
                    (
                        first_block as u64 + i as u64,
                        first_in_window as usize + i * spec.blocksize,
                    )
                })
                .collect();
            // All blocks of this pass share the same MTF, so the
            // `MTF ** mtf_level` spectrum is computed lazily inside the
            // parallel closure: when every window block is a cache hit
            // (`missing` empty) the 16k complex pows never run, and when it
            // does run it lands inside the parallel demod timing instead of
            // the serial tail. Same value, same call sites as before.
            let mtf_pow: OnceLock<Option<Vec<Complex64>>> = OnceLock::new();
            let t_demod0 = std::time::Instant::now();
            let computed: Vec<BlockDecode> = missing
                .par_iter()
                .map(|&(_, off)| {
                    let pow = mtf_pow.get_or_init(|| {
                        if mtf != 0.0 {
                            Some(compute_mtf_pow(&spec.filters.mtf, mtf))
                        } else {
                            None
                        }
                    });
                    demod_block_cpu(&data[off..off + spec.blocklen], mtf, &dspec, true, pow.as_deref(), spec.delays.video_rot)
                })
                .collect();
            self.dbg.demod = t_demod0.elapsed().as_nanos() as u64;
            let t_asm0 = std::time::Instant::now();
            for (&(bnum, _), bd) in missing.iter().zip(computed) {
                let arc = self.cache_insert(bnum, mtf, bd);
                decoded[(bnum - first_block as u64) as usize] = Some(arc);
            }
            // FIFO-prune to the 256 most recently inserted blocks (Python's
            // `lru` list + `cachesize`; eviction never affects output because
            // windows only move forward and redo flushes everything anyway).
            self.prune_cache();
            self.dbg.asm_insert = t_asm0.elapsed().as_nanos() as u64;
            // Serial-schedule assembly (a straight ~140MB/field copy at DRAM
            // bandwidth — parallel scattering cannot beat the memory ceiling,
            // and an async variant measured neutral-to-negative because the
            // window demod is already fully prefetched), then the stage-2
            // audio filter, matching the reference exactly.
            let mut raw = FieldData {
                input: Vec::with_capacity(numblocks_read * per_block),
                video: VideoChannels {
                    demod: Vec::with_capacity(numblocks_read * per_block),
                    demod_raw: Vec::with_capacity(numblocks_read * per_block),
                    demod_05: Vec::with_capacity(numblocks_read * per_block),
                    demod_burst: Vec::with_capacity(numblocks_read * per_block),
                    audio: [
                        Vec::with_capacity(numblocks_read * audiob),
                        Vec::with_capacity(numblocks_read * audiob),
                    ],
                    efm: Vec::with_capacity(numblocks_read * per_block),
                },
                rfhpf: Vec::with_capacity(numblocks_read * per_block),
                audio: [Vec::new(), Vec::new()],
                efm: Vec::new(),
                startloc: block_begin as u64,
            };
            let t_ext0 = std::time::Instant::now();
            for (i, decoded) in decoded.into_iter().enumerate() {
                let decoded = decoded.expect("every block resolved");
                raw.video.demod.extend_from_slice(&decoded.video.demod);
                raw.video.demod_raw.extend_from_slice(&decoded.video.demod_raw);
                raw.video.demod_05.extend_from_slice(&decoded.video.demod_05);
                raw.video.demod_burst.extend_from_slice(&decoded.video.demod_burst);
                for ch in 0..2 {
                    raw.video.audio[ch]
                        .extend_from_slice(&decoded.video.audio[ch]);
                }
                raw.video.efm.extend_from_slice(&decoded.video.efm);
                raw.rfhpf.extend_from_slice(&decoded.rfhpf);
                let off = first_in_window as usize + i * spec.blocksize;
                let block = &data[off..off + spec.blocklen];
                let cut_end = spec.blockcut_end.min(block.len());
                raw.input
                    .extend_from_slice(&block[spec.blockcut..block.len() - cut_end]);
            }
            self.dbg.asm_extend = t_ext0.elapsed().as_nanos() as u64;
            let t_ph0 = std::time::Instant::now();
            let phase2 = audio_phase2(&spec, &raw.video.audio);
            self.dbg.phase2 = t_ph0.elapsed().as_nanos() as u64;
            raw.audio = phase2;
            raw.efm = std::mem::take(&mut raw.video.efm);
            rawdecode = raw;

            // Prefetch the next `prefetch_blocks` at the current MTF (Python's
            // `doread(toread_prefetch, MTF, prefetch=True)`), stopping at EOF
            // just like the loader returning short. Python's range starts at
            // `end // blocksize` — the *last* window block (skipped as cached)
            // — so it reaches one block fewer on the far end than starting
            // just past the window would.
            if let Some(p) = std::env::var_os("LD_DUMP_CACHEPROG") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    let _ = writeln!(f, "PF fw={} start_b={} n={} bpf={} blocksize={} freq_hz={} linelen={} mtf={:.17}", self.fields_written, first_block as u64 + numblocks_read as u64 - 1, prefetch_blocks, self.bytes_per_field, spec.blocksize, spec.freq_hz, spec.linelen, mtf);
                }
            }
            // Same treatment for the prefetch range: collect the uncached
            // block numbers first (stopping at EOF exactly like the serial
            // loop). Demodulation normally runs on the rayon pool in the
            // background while this field's serial tail (process, metadata,
            // downscale, writeout) executes, and the results are folded into
            // the cache at the top of the next `decode_field` — same insert
            // order/content as the serial schedule, so every field observes an
            // identical cache. `LD_NO_ASYNC_PREFETCH` forces the old inline
            // path (also used whenever there is nothing to prefetch).
            {
                let t_pf0 = std::time::Instant::now();
                let plan = self.prefetch_plan(data_start, data.len(), start);
                let prefetch: Vec<(u64, usize)> = plan
                    .into_iter()
                    .filter(|(bnum, _)| !self.demod_cache.contains_key(bnum))
                    .collect();
                let async_ok = !prefetch.is_empty()
                    && std::env::var_os("LD_NO_ASYNC_PREFETCH").is_none();
                if async_ok {
                    // The worker owns copies of its input slices, so it never
                    // races the caller's window buffer; it only computes and
                    // sends, never touches the cache.
                    let blocklen = spec.blocklen;
                    let freq = spec.freq;
                    let freq_half = spec.freq_half;
                    let freq_hz = spec.freq_hz;
                    let spec_arc = Arc::clone(&self.spec);
                    let levels = self.levels;
                    let inputs: Vec<(u64, Vec<f32>)> = prefetch
                        .iter()
                        .map(|&(bnum, off)| {
                            (bnum, data[off..off + blocklen].to_vec())
                        })
                        .collect();
                    let f_mtf = mtf;
                    let (tx, rx) = std::sync::mpsc::channel();
                    let spawn = |f: Box<dyn FnOnce() + Send + 'static>| match &self.pf_pool {
                        Some(pool) => pool.spawn(f),
                        None => rayon::spawn(f),
                    };
                    spawn(Box::new(move || {
                        let dspec = DemodSpecRef::with_plans(
                            freq,
                            freq_half,
                            freq_hz,
                            blocklen,
                            &spec_arc.filters,
                            &levels,
                        );
                        let mtf_pow = if f_mtf != 0.0 {
                            Some(compute_mtf_pow(&spec_arc.filters.mtf, f_mtf))
                        } else {
                            None
                        };
                        let results: Vec<(u64, BlockDecode)> = inputs
                            .into_par_iter()
                            .map(|(bnum, buf)| {
                                (bnum, demod_block_cpu(&buf, f_mtf, &dspec, true, mtf_pow.as_deref(), spec_arc.delays.video_rot))
                            })
                            .collect();
                        let _ = tx.send(results);
                    }));
                    self.pending_prefetch = Some(PendingPrefetch { mtf, recv: rx });
                } else {
                    let computed: Vec<BlockDecode> = prefetch
                        .par_iter()
                        .map(|&(_, off)| {
                            let pf_pow = if mtf != 0.0 {
                                Some(compute_mtf_pow(&spec.filters.mtf, mtf))
                            } else {
                                None
                            };
                            demod_block_cpu(&data[off..off + spec.blocklen], mtf, &dspec, true, pf_pow.as_deref(), spec.delays.video_rot)
                        })
                        .collect();
                    for ((bnum, _), bd) in prefetch.into_iter().zip(computed) {
                        self.cache_insert(bnum, mtf, bd);
                    }
                    self.prune_cache();
                }
                self.dbg.prefetch = t_pf0.elapsed().as_nanos() as u64;
            }
            if let Some(p) = std::env::var_os("LD_DUMP_CACHEPROG") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    for bnum in 42480..42530u64 {
                        if let Some((bmtf, _)) = self.demod_cache.get(&bnum) {
                            let _ = writeln!(f, "fw={} b={} {:.17}", self.fields_written, bnum, bmtf);
                        }
                    }
                }
            }
        }

        if reached_eof {
            return Ok(FieldOutcome::Done(None, None));
        }

        let mut field = Field::new(
            self.spec.clone(),
            self.levels,
            rawdecode,
            prevfield,
            // Python constructs the Field inside the decode thread, which is
            // launched one iteration before the field is written, so its
            // fields_written lags ours by one.
            self.fields_written.saturating_sub(1),
            block_begin,
            false,
            mtf_level,
        );
        let t_proc0 = std::time::Instant::now();
        field.process()?;
        self.dbg.proc = t_proc0.elapsed().as_nanos() as u64;
        let t_df_end = std::time::Instant::now();
        self.dbg.df_rest = (t_df_end
            .duration_since(t_df0)
            .as_nanos() as u64)
            .saturating_sub(
                self.dbg.demod
                    + self.dbg.asm_
                    + self.dbg.phase2
                    + self.dbg.prefetch
                    + self.dbg.pf_fold
                    + self.dbg.proc,
            );

        let offset = field.nextfieldoffset.map(|nfo| {
            nfo - (readloc as f64 - block_begin as f64)
        });
        let offset = if field.valid {
            offset
        } else {
            field.nextfieldoffset
        };
        if std::env::var_os("LD_DBG_ALL").is_some() {
            crate::teeprintln!("DBG3 start={:.1} valid={} off={:?} nfo={:?} ll={}",
                start, field.valid, offset, field.nextfieldoffset, field.linelocs1.first().copied().unwrap_or(-1.0));
        }

        Ok(FieldOutcome::Done(Some(field), offset))
    }

    /// The prefetch range for the field at absolute sample `start` (port of
    /// the DemodCache's `toread_prefetch`-wide lookahead): the `.last` window
    /// block plus the next `prefetch_blocks` blocks, as (block number, offset
    /// into the window data), stopping at the end of the available data. The
    /// caller filters blocks already present in the cache so a block is never
    /// demodulated twice with different MTFs, and the memos are reused by the
    /// later fields' windows.
    fn prefetch_plan(&self, data_start: u64, data_len: usize, start: f64) -> Vec<(u64, usize)> {
        let spec = &self.spec;
        let blocklen = spec.blocklen as u64;
        let blocksize = spec.blocksize as u64;
        let readloc = ((start - spec.blockcut as f64) as i64).max(0) as u64;
        let numblocks = (self.readlen as u64 / blocklen) + 2;
        let begin = (readloc / blocklen) as u64 * blocklen;
        let first_block = (begin / blocksize) as usize;
        let end = begin + numblocks * blocklen;
        let numblocks_read = (end / blocksize) as usize - first_block + 1;
        let prefetch_blocks = (self.bytes_per_field * 4.0 / spec.blocksize as f64) as usize + 4;
        let first_in_window = first_block as i64 * blocksize as i64 - data_start as i64;
        let mut out = Vec::with_capacity(prefetch_blocks);
        if first_in_window < 0 {
            return out;
        }
        for j in 0..prefetch_blocks {
            let bnum = first_block as u64 + numblocks_read as u64 - 1 + j as u64;
            let off = first_in_window as usize + (numblocks_read - 1 + j) * spec.blocksize;
            if off + spec.blocklen > data_len {
                break;
            }
            out.push((bnum, off));
        }
        out
    }

    /// Port of `checkMTF` (auto MTF only; the decoder always uses it).
    fn check_mtf(&mut self) -> bool {
        let oldmtf = self.mtf_level;
        if self.bw_ratios.is_empty() {
            return true;
        }
        // `np.mean` over the list (pairwise f64 summation, numpy-style) so the
        // level tracks the reference bit-for-bit.
        let mean = crate::decode::vits::pairwise_sum_f64(&self.bw_ratios) / self.bw_ratios.len() as f64;
        self.mtf_level = ((mean - 1.08) / 0.38).clamp(0.0, 1.0);
        if std::env::var_os("LD_TRACE_MTF").is_some() {
            crate::teeprintln!("MTFCHK fw={} fdoffset={:.1} old={:.17} new={:.17} nbw={} mean={:.17} redo={}",
                self.fields_written, self.fdoffset_abs(), oldmtf, self.mtf_level, self.bw_ratios.len(), mean, (self.mtf_level - oldmtf).abs() >= 0.05);
        }
        (self.mtf_level - oldmtf).abs() < 0.05
    }

    /// Port of `buildmetadata`: per-field JSON info plus whether a backfill
    /// (filler) field is needed.
    fn buildmetadata(&mut self, field: &mut Field) -> Result<(FieldInfoEntry, bool)> {
        let prevfi = self.fieldinfo.last().cloned();

        let mut fi = FieldInfoEntry {
            is_first_field: field.is_first_field(),
            sync_conf: field.compute_syncconf(),
            seq_no: self.fieldinfo.len() + 1,
            disk_loc: roundfloat(field.readloc as f64 / self.bytes_per_field, 1),
            file_loc: field.readloc,
            median_burst_ire: roundfloat(field.burstmedian, 3),
            field_phase_id: field.field_phase_id,
            decode_faults: None,
            drop_outs: None,
            vits_metrics: None,
            vbi: None,
            audio_samples: 0,
            efm_t_values: 0,
            ac3_symbols: 0,
        };

        if self.spec.do_dod {
            let t_dod0 = std::time::Instant::now();
            // Dropouts were already computed as a side task during downscale
            // (pure over the field's data); fall back to a synchronous call
            // only if downscale never ran for this field.
            let (lines, starts, ends) = field
                .dropouts_cached
                .clone()
                .unwrap_or_else(|| dropouts::detect_dropouts(field));
            self.dbg.meta_dod = t_dod0.elapsed().as_nanos() as u64;
            if !lines.is_empty() {
                fi.drop_outs = Some(DropOuts {
                    field_line: lines,
                    startx: starts,
                    endx: ends,
                });
            }
        }

        // This is a bitmap, not a counter; kept local so the "Skipped field"
        // early return below leaves decodeFaults out of the JSON (the
        // reference only assigns the key after these checks).
        let mut decode_faults: i64 = 0;
        if let Some(prevfi) = prevfi {
            let phase_ok = (fi.field_phase_id == 1
                && prevfi.field_phase_id == self.spec.sys_field_phases as i64)
                || (fi.field_phase_id == prevfi.field_phase_id + 1);
            if !phase_ok {
                tracing::warn!(
                    "At field #{}, Field phaseID sequence mismatch ({}->{}) (player may be paused)",
                    self.fieldinfo.len(),
                    prevfi.field_phase_id,
                    fi.field_phase_id
                );
                decode_faults |= 2;
            }

            if prevfi.is_first_field == fi.is_first_field {
                if inrange(fi.disk_loc - prevfi.disk_loc, 0.95, 1.05) {
                    decode_faults |= 1;
                    fi.is_first_field = !prevfi.is_first_field;
                    fi.sync_conf = 10;
                } else {
                    tracing::error!("Skipped field");
                    decode_faults |= 4;
                    fi.sync_conf = 0;
                    return Ok((fi, true));
                }
            }
        }

        let fp = self.prevfield.clone();
        let t_v2 = std::time::Instant::now();
        // Reuse the metrics computed in the decode loop for this same field;
        // only the fp-dependent line-19 metrics differ, and neither wSNR nor
        // bPSNR read those.
        let metrics = self
            .cached_vits
            .take()
            .unwrap_or_else(|| vits::compute_vits_metrics(&self.spec, field, fp.as_ref()));
        self.dbg.meta_vits = t_v2.elapsed().as_nanos() as u64;
        fi.decode_faults = Some(decode_faults);
        fi.vits_metrics = Some(VitsMetrics {
            w_snr: metrics.w_snr,
            b_psnr: metrics.b_psnr,
        });
        fi.vbi = Some(VbiData {
            vbi_data: field.linecode.iter().flatten().copied().collect(),
        });

        Ok((fi, false))
    }

    /// Update the VBI frame-number state from a just-decoded valid field. This
    /// runs for every valid field (not only written ones) so `--seek` can
    /// locate a target frame while discarding output.
    fn update_frame_number(&mut self, field: &Field) {
        // Python's `buildmetadata` returns early on the "Skipped field"
        // branch (prevfi.isFirstField == isFirstField with a diskLoc gap
        // outside [0.95, 1.05]) *before* `self.firstfield = f` and the
        // decodeFrameNumber call, so a skipped field never updates the stored
        // first field. Rust must mirror that, or the next frame pairs the
        // skipped field's linecodes (e.g. CLV codes on a junk field) and
        // `is_clv` sticks true where the reference trims bw_ratios to 30,
        // shifting the MTF mean and every redo decision after re-acquisition.
        let skipped = self.fieldinfo.last().is_some_and(|prev| {
            prev.is_first_field == field.is_first_field()
                && !inrange(
                    field.readloc as f64 / self.bytes_per_field - prev.disk_loc,
                    0.95,
                    1.05,
                )
        });
        if skipped {
            return;
        }
        self.frame_number = None;
        if field.is_first_field() {
            self.firstfield_linecode = Some(field.linecode.clone());
        } else if let Some(first_lc) = self.firstfield_linecode.clone() {
            self.frame_number = self.decode_frame_number(&first_lc, &field.linecode);
        }
    }

    /// Roughly seek to a field index (port of `roughseek`), as a multiple of
    /// the per-field sample count.
    pub fn rough_seek(&mut self, field_index: i64) {
        self.fdoffset = (field_index.max(0) as u64) * self.spec.bytes_per_field() as u64;
        self.fdoffset_frac = 0.0;
    }

    /// Position the decoder at an absolute input sample offset (used by the
    /// LD_START_SAMPLE diagnostic override).
    pub fn set_position_samples(&mut self, samples: u64) {
        self.fdoffset = samples;
        self.fdoffset_frac = 0.0;
    }

    /// The current decode position expressed as a field index.
    pub fn field_index(&self) -> i64 {
        (self.fdoffset / self.spec.bytes_per_field() as u64) as i64
    }

    /// The VBI frame number of the last decoded frame, if one was read.
    pub fn frame_number(&self) -> Option<i64> {
        self.frame_number
    }

    /// Python's `ldd.leadOut`: true once a frame's Philips codes showed two
    /// lead-out marks. The CLI stops decoding when this is set.
    pub fn lead_out(&self) -> bool {
        self.lead_out
    }

    /// The current decode position, in absolute input samples.
    pub fn position(&self) -> u64 {
        self.fdoffset
    }

    /// Port of `writeout`: run the EFM PLL, record the field info and hand the
    /// picture, audio and EFM back to the caller for writing.
    fn writeout(&mut self, wf: WriteableField) -> Result<WriteableField> {
        let mut wf = wf;
        wf.efm = if self.digital_audio {
            if let Some(pd) = std::env::var_os("LD_DUMP_PLL") {
                use std::io::Write;
                use std::sync::atomic::{AtomicUsize, Ordering};
                static PSEQ: AtomicUsize = AtomicUsize::new(0);
                let ps = PSEQ.fetch_add(1, Ordering::SeqCst);
                let dir = pd.to_string_lossy().into_owned();
                let mut f = std::fs::File::create(format!("{}/f{:03}_in.bin", dir, ps)).unwrap();
                for &v in &wf.efm_raw {
                    let _ = f.write_all(&v.to_le_bytes());
                }
                drop(f);
                let mut f = std::fs::File::create(format!("{}/f{:03}_linelocs.bin", dir, ps)).unwrap();
                for &v in &wf.linelocs {
                    let _ = f.write_all(&v.to_le_bytes());
                }
                drop(f);
                let out = self.efm_pll.process(&wf.efm_raw);
                let mut f = std::fs::File::create(format!("{}/f{:03}_out.bin", dir, ps)).unwrap();
                for &v in &out {
                    let _ = f.write_all(&[v as u8]);
                }
                drop(f);
                out
            } else {
                self.efm_pll.process(&wf.efm_raw)
            }
        } else {
            Vec::new()
        };
        wf.info.audio_samples = wf.audio.len() / 2;
        wf.info.efm_t_values = wf.efm.len();
        wf.info.ac3_symbols = 0;
        self.fieldinfo.push(wf.info.clone());
        self.fields_written += 1;
        Ok(wf)
    }

    /// Port of `decodeFrameNumber`: decode the Philips code data from both
    /// fields of a frame.
    fn decode_frame_number(
        &mut self,
        f1_linecode: &[Option<i64>],
        f2_linecode: &[Option<i64>],
    ) -> Option<i64> {
        // Python's `decodeFrameNumber` resets the CLV state at the top of
        // every call (self.isCLV = False; clvMinutes/Seconds/FrameNum = None),
        // so a CAV frame that follows CLV minutes is still seen as CAV. Rust
        // must mirror that: a stale `clv_minutes` from an earlier frame would
        // otherwise hit the CLV early-return below before the CAV branch is
        // reached, leaving `is_clv` stuck true and the bw_ratios history at
        // keep=900 where the reference trims it to 30 (which changes the MTF
        // mean and every subsequent redo decision).
        self.is_clv = false;
        self.clv_minutes = None;
        self.clv_seconds = None;
        self.clv_frame_num = None;
        self.early_clv = false;
        let mut leadout_count = 0i64;

        for &l in f1_linecode.iter().chain(f2_linecode) {
            let Some(l) = l else { continue };

            if l == 0x80EEEE {
                // lead-out reached
                leadout_count += 1;
                if leadout_count == 2 {
                    self.lead_out = true;
                }
            } else if l == 0x88FFFF {
                // lead-in
                self.lead_in = true;
            } else if (l & 0xF0DD00) == 0xF0DD00 {
                // CLV minutes/hours
                if let Ok(minutes) = decode_bcd(l & 0xFF) {
                    if let Ok(hours) = decode_bcd((l >> 16) & 0xF) {
                        self.clv_minutes = Some(minutes + hours * 60);
                        self.is_clv = true;
                    }
                }
            } else if (l & 0xF00000) == 0xF00000 {
                // CAV frame
                if let Ok(rv) = decode_bcd(l & 0x7FFFF) {
                    self.is_clv = false;
                    return Some(rv);
                }
            } else if (l & 0x80F000) == 0x80E000 {
                // CLV picture #
                if let (Ok(sec1s), Ok(frame)) = (decode_bcd((l >> 8) & 0xF), decode_bcd(l & 0xFF)) {
                    let sec10s = ((l >> 16) & 0xF) - 0xA;
                    if sec10s >= 0 {
                        self.clv_frame_num = Some(frame);
                        self.clv_seconds = Some(sec1s + 10 * sec10s);
                        self.is_clv = true;
                    }
                }
            }

            if let Some(clv_minutes) = self.clv_minutes {
                let minute_seconds = clv_minutes * 60;
                if let Some(clv_seconds) = self.clv_seconds {
                    // newer CLV
                    return Some((minute_seconds + clv_seconds) * 30 + self.clv_frame_num.unwrap_or(0));
                } else {
                    self.early_clv = true;
                    return Some(minute_seconds);
                }
            }
        }

        None
    }

    /// Decoder metadata for the JSON sidecar (port of `build_json`).
    pub fn metadata(&self) -> Option<DecoderMetadata> {
        let m = self.metadata.as_ref()?;
        let ire_to_output = |ire: f64| {
            let hz = self.levels.iretohz(ire);
            let mut reduced = (hz - self.levels.ire0) / self.levels.hz_ire;
            reduced -= self.levels.vsync_ire;
            (((reduced * m.out_scale) + self.spec.sys_output_zero as f64)
                .clamp(0.0, 65535.0)
                + 0.5) as u16 as f64
        };
        // Current burst adjustment (matches the 2/27/19 value in build_json).
        let badj = -1.4;
        let to_sample = |us: f64| ((us * self.spec.sys_outfreq) + badj).round() as i64;

        // NTSC setup (blackIRE = 7.5, matching main.py); blanking stays at 0 IRE.
        let black = ire_to_output(7.5);
        let blanking = ire_to_output(0.0);
        let white = ire_to_output(100.0);
        Some(DecoderMetadata {
            system: ColorSystem::Ntsc.as_str(),
            number_of_sequential_fields: self.fieldinfo.len(),
            field_width: self.spec.sys_outlinelen,
            sample_rate: self.spec.sys_outfreq * 1e6,
            black_16b_ire: black,
            white_16b_ire: white,
            blanking_16b_ire: blanking,
            field_height: m.outlinecount,
            colour_burst_start: to_sample(self.spec.sys_color_burst_us[0]),
            colour_burst_end: to_sample(self.spec.sys_color_burst_us[1]),
            active_video_start: to_sample(self.spec.sys_active_video_us[0]),
            active_video_end: to_sample(self.spec.sys_active_video_us[1]),
        })
    }
}

fn inrange(a: f64, mi: f64, ma: f64) -> bool {
    a >= mi && a <= ma
}

/// `np.round(x * 10^places) / 10^places` (round half to even).
fn roundfloat(fl: f64, places: i64) -> f64 {
    let r = 10f64.powi(places as i32);
    (fl * r).round_ties_even() / r
}

/// Read a BCD-encoded number; `Err` on any non-decimal digit.
fn decode_bcd(bcd: i64) -> Result<i64, ()> {
    if bcd == 0 {
        return Ok(0);
    }
    let digit = bcd & 0xF;
    if digit > 9 {
        return Err(());
    }
    Ok(10 * decode_bcd(bcd >> 4)? + digit)
}
