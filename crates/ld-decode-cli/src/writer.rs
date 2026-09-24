//! Output writer: the `.tbc` picture file plus the `.tbc.json` sidecar,
//! rewritten incrementally after every field so the JSON is always a complete,
//! valid document on disk (same scheme as the tape-decode CLI).

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::time::Instant;

use anyhow::{Context, Result};
use ld_decode::{DecoderMetadata, FieldInfoEntry, LumaOutput, WriteableField};
use serde::Serialize;

use crate::async_db::AsyncDbWriter;

/// The ld-decode version this port replicates bit-for-bit. The reference
/// release build reports `release:7.3.0`, which is parsed (mirroring
/// `build_json` in the reference) into `gitBranch=release` / `gitCommit=7.3.0`
/// in the `.tbc.json` `videoParameters`.
const REFERENCE_VERSION: &str = "release:7.3.0";

/// Opening of the JSON sidecar. During the decode only the fields array is
/// written (`{"fields":[ ... ]}\r\n`); the reference emits the
/// `pcmAudioParameters` / `videoParameters` header only at close time (see
/// `close`), so it is prepended then via an in-place rewrite of this file.
const FIELDS_OPEN: &[u8] = b"{\"fields\":[";

/// Whether a backfill re-write that lands at `fw2` still reaches the json entry
/// that was pushed for the same field at `fw1`.
///
/// The reference's main loop drains the pending field-info dicts after a
/// `readfield()` only when `ldd.fields_written < 100 || fields_written % 500 ==
/// 0` (main.py:506), and `JSONDumper.write()` serializes whatever
/// `fieldinfo.read()` hands it at that moment. `writeout` takes `fi` out of its
/// dataset tuple, so a backfill re-write pushes the *same* dict object again:
/// the earlier entry shows the re-processed values only while it is still
/// pending, i.e. only if this re-write happens at or before the next drain.
/// Below 100 the drain is that very field's own write, so a re-write never
/// reaches back to an earlier entry.
///
/// Validated against every reference json on hand (735,000+ fields: Diamond
/// Time CLV ldf, GGV1016 ldf, Pioneer GGV1069 ldf, s16 full, flac full,
/// gain1010): this rule matches all 22 repeated-field entries, whereas the
/// previous `fw1 >= 100` test mismatches the one at GGV1016 seqNo 69999
/// (`fw1=69999`, `fw2=70001`, next drain 70000).
fn json_alias_reaches_earlier(fw1: usize, fw2: usize) -> bool {
    let next_drain = if fw1 < 100 { fw1 } else { fw1.div_ceil(500) * 500 };
    fw2 <= next_drain
}

pub struct DecodeWriter {
    /// The picture stream. Boxed so the CLI can point it at stdout (`outfile
    /// == "-"`) instead of a `.tbc` file without a second code path; every
    /// other output is a real file (a pipe cannot carry the json header
    /// rewrite, which needs seek).
    outfile_video: BufWriter<Box<dyn Write + Send>>,
    outfile_audio: Option<BufWriter<File>>,
    outfile_efm: Option<BufWriter<File>>,
    /// Debug dump of the EFM samples fed to the PLL (mirrors ld-decode's
    /// --preEFM `.prefm` output). Only created when LD_DUMP_PREFM is set.
    outfile_pre_efm: Option<BufWriter<File>>,
    json_file: Option<File>,
    /// Byte offset of the array-closing `]` in the JSON file.
    /// Buffered per-field JSON entries. The reference keeps every `fi` dict
    /// in memory and dumps the array only at close time; rust must do the
    /// same so the filler aliasing (below) can be resolved before writing.
    json_entries: Vec<FieldInfoEntry>,
    /// fields_written at the time each json_entries entry was pushed (for the
    /// dumper-drain aliasing rule).
    json_push_fields: Vec<usize>,
    /// Scratch for the i8->u8 EFM byte conversion (reused across fields).
    efm_bytes: Vec<u8>,
    field_count: usize,
    first_field_write: Option<Instant>,
    last_field_write: Option<Instant>,
    /// SQLite metadata sidecar (`<out>.tbc.db`), written per field on a
    /// background thread (the db is derived data, off the critical path).
    db: Option<AsyncDbWriter>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PcmAudioParameters {
    bits: usize,
    is_little_endian: bool,
    is_signed: bool,
    sample_rate: usize,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct VideoParameters {
    number_of_sequential_fields: usize,
    os_info: String,
    version: String,
    git_branch: String,
    git_commit: String,
    system: String,
    field_width: usize,
    sample_rate: f64,
    black_16b_ire: f64,
    white_16b_ire: f64,
    blanking_16b_ire: f64,
    field_height: usize,
    colour_burst_start: i64,
    colour_burst_end: i64,
    active_video_start: i64,
    active_video_end: i64,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TbcMetadata {
    pcm_audio_parameters: PcmAudioParameters,
    video_parameters: VideoParameters,
}

impl DecodeWriter {
    pub fn new(
        luma: Box<dyn Write + Send>,
        audio: Option<File>,
        efm: Option<File>,
        pre_efm: Option<File>,
        json: Option<File>,
        db: Option<AsyncDbWriter>,
    ) -> Result<Self> {
        if let Some(mut json) = json.as_ref() {
            let mut chunk = FIELDS_OPEN.to_vec();
            chunk.extend_from_slice(b"]}\r\n");
            json.write_all(&chunk)?;
        }
        // 8MB OS-bypass buffers: each field's luma/audio/efm payload is
        // written as one or two large sequential writes; without buffering
        // every field pays several unbuffered file writes (~1-2ms on HDD,
        // measurable on the serial tail).
        const WBUF: usize = 8 * 1024 * 1024;
        Ok(Self {
            outfile_video: BufWriter::with_capacity(WBUF, luma),
            outfile_audio: audio.map(|f| BufWriter::with_capacity(WBUF / 4, f)),
            outfile_efm: efm.map(|f| BufWriter::with_capacity(WBUF / 4, f)),
            outfile_pre_efm: pre_efm.map(|f| BufWriter::with_capacity(WBUF / 4, f)),
            json_file: json,
            json_entries: Vec::new(),
            json_push_fields: Vec::new(),
    efm_bytes: Vec::new(),
            field_count: 0,
            first_field_write: None,
            last_field_write: None,
            db,
        })
    }

    pub fn write_writeable(
        &mut self,
        field: &WriteableField,
        metadata: Option<&DecoderMetadata>,
    ) -> Result<()> {
        // field_id for the database is the 0-based index of this field among
        // those written (the reference's `fields_written`, pre-increment).
        let field_id = self.field_count;
        if let (Some(db), Some(metadata)) = (&self.db, metadata) {
            db.write_field(&field.info, metadata, field_id);
        }

        let luma = field.luma();
        match luma {
            LumaOutput::Encoded(values) => write_u16_le(&mut self.outfile_video, values)?,
            LumaOutput::Raw(values) => write_f32_slice(&mut self.outfile_video, values)?,
        }

        if let Some(audio_file) = self.outfile_audio.as_mut() {
            if !field.audio.is_empty() {
                write_i16_slice(audio_file, &field.audio)?;
            }
        }
        if let Some(efm_file) = self.outfile_efm.as_mut() {
            if !field.efm.is_empty() {
                // Reuse the scratch buffer across fields (saves a ~15KB
                // alloc+copy per field on the serial write path).
                self.efm_bytes.clear();
                self.efm_bytes.reserve(field.efm.len());
                for v in &field.efm {
                    self.efm_bytes.push(*v as u8);
                }
                efm_file.write_all(&self.efm_bytes)?;
            }
        }
        if let Some(pre_efm_file) = self.outfile_pre_efm.as_mut() {
            if !field.efm_raw.is_empty() {
                write_i16_slice(pre_efm_file, &field.efm_raw)?;
            }
        }

        let now = Instant::now();
        let start = self.first_field_write.get_or_insert(now);
        self.last_field_write = Some(now);
        self.field_count += 1;

        if self.json_file.is_some() {
            // Python's json dumper thread serializes the fi dicts by reference,
            // but only drains them when the main loop calls write():
            //
            //     if ldd.fields_written < 100 or (ldd.fields_written % 500) == 0:
            //         jsondumper.write()          # main.py:506
            //
            // i.e. after every field for the first 100, then at each 500-field
            // boundary. `writeout` gets `fi` out of its dataset tuple, so a
            // backfill re-write of a field pushes the *same* dict object again;
            // the earlier json entry therefore shows the re-processed
            // efmTValues only if this re-write happened at or before the next
            // drain — once that drain has run, the entry was already serialized
            // with its own values. (The DB is snapshotted per INSERT and always
            // keeps its own.) Below 100 the next drain *is* this field's own
            // write, so a re-write never reaches the earlier entry.
            //
            // Checked against every reference json on hand — Diamond Time ldf
            // 71,710 fields, s16 168,800, gain1010 167,588, GGV1016 108,576,
            // flac-s300 168,566, Pioneer GGV1069 50,722 — this rule has no
            // violation; `orig_fw >= 100` has one, at GGV1016 seqNo 69999
            // (fw1=69999, fw2=70001, next drain 70000), where Python keeps the
            // entry's own 20265 and the old rule overwrote it with 20293.
            let dup_at = self
                .json_entries
                .iter()
                .rposition(|p| p.seq_no == field.info.seq_no)
                .map(|i| (i, self.json_push_fields[i]));
            self.json_entries.push(field.info.clone());
            self.json_push_fields.push(self.field_count);
            if let Some((i, orig_fw)) = dup_at {
                if json_alias_reaches_earlier(orig_fw, self.field_count) {
                    let prev = &mut self.json_entries[i];
                    prev.efm_t_values = field.info.efm_t_values;
                    prev.audio_samples = field.info.audio_samples;
                    prev.ac3_symbols = field.info.ac3_symbols;
                }
            }
        }

        const LOG_INTERVAL: usize = 500;
        let field_num = self.field_count;
        tracing::debug!("Written field {field_num}");
        if field_num.is_multiple_of(LOG_INTERVAL) {
            let elapsed = now.duration_since(*start).as_secs_f64();
            let fps = if elapsed > 0.0 {
                field_num as f64 / (elapsed * 2.0)
            } else {
                0.0
            };
            tracing::info!("Decoded {} fields so far in {:.3}s ({:.2} FPS)", field_num, elapsed, fps);
        }
        Ok(())
    }

    pub fn close(&mut self, metadata: Option<DecoderMetadata>) -> Result<()> {
        // Flush the buffered writers before the process drops them.
        self.outfile_video.flush()?;
        if let Some(a) = self.outfile_audio.as_mut() {
            a.flush()?;
        }
        if let Some(e) = self.outfile_efm.as_mut() {
            e.flush()?;
        }
        if let Some(pe) = self.outfile_pre_efm.as_mut() {
            pe.flush()?;
        }
        // Drain + join the background .tbc.db worker before reporting FPS.
        if let Some(db) = self.db.as_mut() {
            db.finish()?;
        }
        let field_count = self.field_count;
        let elapsed = match (self.first_field_write, self.last_field_write) {
            (Some(first), Some(last)) => last.duration_since(first).as_secs_f64(),
            _ => 0.0,
        };
        let fps = if elapsed > 0.0 {
            field_count as f64 / (elapsed * 2.0)
        } else {
            0.0
        };
        tracing::info!(
            "Decode finished: {} fields in {:.3}s ({:.2} FPS)",
            field_count,
            elapsed,
            fps
        );

        if let (Some(json_file), Some(metadata)) = (self.json_file.as_mut(), metadata) {
            // The `videoParameters` are only known once decoding is done (the
            // ire levels and field height come from the last decoded field, and
            // `numberOfSequentialFields` from the final count), so the header is
            // emitted here by rewriting the whole file, mirroring the
            // reference, whose `build_json` also writes the header only at
            // close time.
            //
            // Python's json dumper thread serializes each field's dict when it
            // arrives (pushed at its writeout), so a later in-place mutation of
            // the aliased dict never reaches the json. Entries keep the values
            // from their own writeout, duplicated fileLoc included.
            let mut chunk = Vec::new();
            append_header(&mut chunk, &metadata, field_count)?;
            for (i, e) in self.json_entries.iter().enumerate() {
                if i > 0 {
                    chunk.push(b',');
                }
                serde_json::to_writer(&mut chunk, e)?;
            }
            chunk.extend_from_slice(b"]}\r\n");

            json_file
                .seek(SeekFrom::Start(0))
                .with_context(|| "json: seek to start")?;
            json_file
                .write_all(&chunk)
                .with_context(|| "json: write header+fields")?;
            json_file
                .set_len(chunk.len() as u64)
                .with_context(|| "json: set_len")?;
        }
        Ok(())
    }
}

/// Append the reference layout: `"pcmAudioParameters":{...},"videoParameters":{...},"fields":[`
/// (everything of `TbcMetadata` except its trailing `}`) to `chunk`.
fn append_header(
    chunk: &mut Vec<u8>,
    metadata: &DecoderMetadata,
    field_count: usize,
) -> Result<()> {
    // Mirror the reference `build_json` version parsing: `release:7.3.0`
    // splits into branch `release` / commit `7.3.0`.
    let (git_branch, git_commit) = match REFERENCE_VERSION.split_once(':') {
        Some((b, c)) => (b.to_string(), c.to_string()),
        None => (String::new(), String::new()),
    };
    let tbc = TbcMetadata {
        pcm_audio_parameters: PcmAudioParameters {
            bits: 16,
            is_little_endian: true,
            is_signed: true,
            sample_rate: 44100,
        },
        video_parameters: VideoParameters {
            number_of_sequential_fields: field_count,
            os_info: platform_info(),
            version: REFERENCE_VERSION.to_string(),
            git_branch,
            git_commit,
            system: metadata.system.to_string(),
            field_width: metadata.field_width,
            sample_rate: metadata.sample_rate,
            black_16b_ire: metadata.black_16b_ire,
            white_16b_ire: metadata.white_16b_ire,
            blanking_16b_ire: metadata.blanking_16b_ire,
            field_height: metadata.field_height,
            colour_burst_start: metadata.colour_burst_start,
            colour_burst_end: metadata.colour_burst_end,
            active_video_start: metadata.active_video_start,
            active_video_end: metadata.active_video_end,
        },
    };
    let bytes = serde_json::to_vec(&tbc)?;
    chunk.extend_from_slice(&bytes[..bytes.len() - 1]);
    chunk.extend_from_slice(b",\"fields\":[");
    Ok(())
}

/// Port of Python's `f'{platform.system()}:{platform.release()}:{platform.version()}'`.
/// Best-effort: on Windows the release/version come from `ver`, on unix from
/// `uname -r`/`uname -v` (which is exactly what `platform.release()` and
/// `platform.version()` return there). If the query fails only the OS name is
/// emitted.
fn platform_info() -> String {
    let os = match std::env::consts::OS {
        "windows" => "Windows",
        "macos" => "Darwin",
        "linux" => "Linux",
        other => other,
    };
    if std::env::consts::OS == "windows" {
        if let Ok(out) = std::process::Command::new("cmd").args(["/c", "ver"]).output() {
            if let Ok(s) = String::from_utf8(out.stdout) {
                // e.g. "Microsoft Windows [Version 10.0.19045]"
                if let Some(v) = s
                    .split('[')
                    .nth(1)
                    .and_then(|x| x.split(']').next())
                    .and_then(|x| x.trim().strip_prefix("Version "))
                {
                    // platform.version() reports major.minor.build (no UBR).
                    let v3 = v.split('.').take(3).collect::<Vec<_>>().join(".");
                    let release = v3.split('.').next().unwrap_or(v3.as_str());
                    return format!("{os}:{release}:{v3}");
                }
            }
        }
    } else if let Some(release) = uname_field("-r") {
        let version = uname_field("-v").unwrap_or_default();
        return format!("{os}:{release}:{version}");
    }
    os.to_string()
}

#[cfg(unix)]
fn uname_field(flag: &str) -> Option<String> {
    let out = std::process::Command::new("uname").arg(flag).output().ok()?;
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

#[cfg(not(unix))]
fn uname_field(_flag: &str) -> Option<String> {
    None
}

fn write_u16_le(file: &mut dyn Write, values: &[u16]) -> Result<()> {
    // One write per field, not per sample: a per-element write_all is a
    // syscall per pixel and dominates the whole decode.
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    file.write_all(&bytes)?;
    Ok(())
}

fn write_f32_slice(file: &mut dyn Write, values: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    file.write_all(&bytes)?;
    Ok(())
}

fn write_i16_slice(file: &mut dyn Write, values: &[i16]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    file.write_all(&bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::json_alias_reaches_earlier;

    #[test]
    fn alias_stops_at_the_reference_drain_grid() {
        // GGV1016 seqNo 69999: pushed at fw1=69999, re-written at fw2=70001.
        // The 70000 drain serialized the earlier entry first, so Python keeps
        // that entry's own 20265 (the old `fw1 >= 100` test overwrote it with
        // the re-processed 20293).
        assert!(!json_alias_reaches_earlier(69_999, 70_001));
        assert!(!json_alias_reaches_earlier(69_998, 70_001));
        // Same 500-block: the re-write lands before the drain, so the shared
        // dict is still pending and both entries show the re-processed values.
        assert!(json_alias_reaches_earlier(69_800, 69_802));
        assert!(json_alias_reaches_earlier(69_900, 69_999));
        // Exactly on the drain point: the re-write happened during that
        // iteration, before the drain that follows it.
        assert!(json_alias_reaches_earlier(69_995, 70_000));
        // Below 100 the drain is per field, so a re-write never reaches back.
        assert!(!json_alias_reaches_earlier(99, 101));
        assert!(!json_alias_reaches_earlier(50, 52));
        // The first entry already past 100 still aliases within its block (the
        // ldf lead-in skip-back pairs).
        assert!(json_alias_reaches_earlier(100, 102));
        assert!(json_alias_reaches_earlier(100, 500));
        assert!(!json_alias_reaches_earlier(100, 501));
    }
}
