//! ld-decode command line interface: decode raw RF Laserdisc captures (NTSC)
//! to a `.tbc` picture plus `.tbc.json` sidecar.

mod db;
mod prefetch;
mod reader;
mod writer;

use std::fs::File;
use std::sync::Arc;

// The decode pipeline churns through hundreds of MB of transient FFT/scatter
// buffers per field; the Windows heap serializes large alloc/free bursts from
// the rayon workers. mimalloc handles that pattern much better. Pure allocator
// swap - arithmetic is untouched.
#[global_allocator]
static GLOBAL_ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::{Context as _, Result};
use clap::Parser;
use ld_decode::{ColorSystem, DecodeRequest, Decoder, DecoderSpec, BLOCKSIZE};
use reader::DecodeReader;
use writer::DecodeWriter;

#[derive(Parser)]
#[command(
    name = "ld-decode",
    about = "Decode raw RF Laserdisc captures (NTSC) into a TBC picture and JSON metadata"
)]
struct Args {
    /// Source file (.lds, .s16, .r16, .rf, .r30)
    infile: String,
    /// Base name for the output .tbc and .tbc.json files
    outfile: String,
    /// Input sample rate in MHz
    #[arg(long, default_value_t = 40.0)]
    inputfreq: f64,
    /// MTF compensation multiplier
    #[arg(long, default_value_t = 1.0)]
    mtf: f64,
    /// MTF compensation offset
    #[arg(long, default_value_t = 0.0)]
    mtf_offset: f64,
    /// Disable automatic gain control
    #[arg(long)]
    no_agc: bool,
    /// Disable dropout detection
    #[arg(long)]
    no_dod: bool,
    /// Notch filter on decoded video to reduce colour 'wobble'
    #[arg(long)]
    ntsc_color_notch: bool,
    /// Use lower-bandwidth (lowband) decode settings
    #[arg(long)]
    lowband: bool,
    /// De-emphasis strength multiplier
    #[arg(long, default_value_t = 1.0)]
    deemp_str: f64,
    /// De-emphasis time constants in usec: high,low (defaults kept when omitted)
    #[arg(long, value_delimiter = ',', num_args = 1..=2)]
    deemp: Vec<f64>,
    /// Rough jump to frame n of the capture (each frame is 2 fields; 0 = start)
    #[arg(long, short = 's', default_value_t = 0)]
    start: u64,
    /// Limit the decode to this many frames (each frame is 2 fields)
    #[arg(long, short = 'l')]
    length: Option<u64>,
    /// Seek to a specific frame number (needs VBI frame codes on the disc)
    #[arg(long, short = 'S')]
    seek: Option<i64>,
    /// Disable EFM (digital audio) output
    #[arg(long)]
    no_efm: bool,
    /// Disable analog audio decoding
    #[arg(long, alias = "disable-analogue-audio")]
    disable_analog_audio: bool,
    /// Number of worker threads for demodulation (default: cores, capped at 8)
    #[arg(long, short = 'j', default_value_t = default_threads())]
    threads: usize,
}

/// Default worker count: the workload is memory-bandwidth-bound, so more than
/// 8 threads rarely helps (hyperthreading usually makes it worse); matching
/// ld-decode's conservative default keeps other machines safe too.
fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(4)
}

/// tracing writer that appends every line to the installed `.log` file (see
/// `ld_decode::logging`), while the layer itself keeps writing to stdout.
struct LogFileWriter;

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogFileWriter {
    type Writer = LogFileSink;
    fn make_writer(&'a self) -> Self::Writer {
        LogFileSink
    }
}

struct LogFileSink;

impl std::io::Write for LogFileSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Ok(ld_decode::logging::file_write(buf))
    }
    fn flush(&mut self) -> std::io::Result<()> {
        ld_decode::logging::file_flush();
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Mirror the terminal output into `<outfile>.log` (like the reference
    // ld-decode). Opened before the tracing subscriber so every line of the
    // run lands in the file; a failure only disables file logging.
    let outfile = args.outfile.clone();
    if let Err(e) = ld_decode::logging::install(std::path::Path::new(&format!("{outfile}.log"))) {
        eprintln!("WARN: cannot create log file {outfile}.log: {e:#}");
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .with_writer(tracing_subscriber::fmt::writer::Tee::new(
            std::io::stdout,
            LogFileWriter,
        ))
        .init();

    let mut request = DecodeRequest::default();
    request.inputfreq = args.inputfreq;
    request.system = ColorSystem::Ntsc;
    request.mtf_level = args.mtf;
    request.mtf_offset = args.mtf_offset;
    request.use_agc = !args.no_agc;
    request.do_dod = !args.no_dod;
    request.ntsc_color_notch = args.ntsc_color_notch;
    request.lowband = args.lowband;
    request.deemp_str = args.deemp_str;
    if args.deemp.len() >= 1 {
        request.deemp_coeff.0 = args.deemp[0];
    }
    if args.deemp.len() >= 2 {
        request.deemp_coeff.1 = args.deemp[1];
    }

    ld_decode::set_worker_threads(args.threads);

    let spec = Arc::new(DecoderSpec::new(&request)?);
    tracing::info!(
        "System NTSC, {} MHz input, {} samples/line, {} lines/field",
        request.inputfreq,
        spec.linelen(),
        spec.output_lines()
    );

    let format = reader::infer_format(&args.infile)?;
    let file = File::open(&args.infile)
        .with_context(|| format!("opening {}", args.infile))?;
    let source = reader::open_source(&args.infile, file, format)?;
    let mut reader = DecodeReader::new(source);

    let outfile = args.outfile.clone();
    let luma = File::create(format!("{outfile}.tbc"))
        .with_context(|| format!("creating {outfile}.tbc"))?;
    // Opened read+write: `File::create` alone is write-only on Windows, and
    // the writer re-reads the fields array at close time to prepend the JSON
    // header (os error 5 otherwise).
    let json = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(format!("{outfile}.tbc.json"))
        .with_context(|| format!("creating {outfile}.tbc.json"))?;
    let audio = if args.disable_analog_audio {
        None
    } else {
        Some(
            File::create(format!("{outfile}.pcm"))
                .with_context(|| format!("creating {outfile}.pcm"))?,
        )
    };
    let efm = if args.no_efm {
        None
    } else {
        Some(
            File::create(format!("{outfile}.efm"))
                .with_context(|| format!("creating {outfile}.efm"))?,
        )
    };
    let pre_efm = std::env::var_os("LD_DUMP_PREFM").map(|p| {
        File::create(&p).unwrap_or_else(|e| panic!("creating prefm dump {}: {e}", p.to_string_lossy()))
    });
    // SQLite metadata sidecar, created fresh every run (like the reference,
    // which unlinks any pre-existing `<out>.tbc.db`).
    let db = db::DbWriter::create(std::path::Path::new(&format!("{outfile}.tbc.db")))?;
    let mut writer = DecodeWriter::new(luma, audio, efm, pre_efm, Some(json), Some(db))?;

    let mut decoder = Decoder::new(Arc::clone(&spec), 0);
    if args.disable_analog_audio {
        decoder.set_analog_audio(0.0);
    }
    decoder.set_digital_audio(!args.no_efm);

    if let Some(target) = args.seek {
        match run_seek(&mut reader, &spec, &mut decoder, target)? {
            true => tracing::info!("Finished seek, starting decode at frame {}", target),
            false => {
                ld_decode::teeprintln!("ERROR: Seeking to frame {} failed (no readable frame code on the disc?)", target);
                std::process::exit(1);
            }
        }
    }

    // Reference: `--start n` is a rough jump to frame n (`roughseek(n*2)`), so
    // position the reader just before the decoder's read begin (the decoder
    // skips straight to that field once decoding starts).
    let start_base = if args.start > 0 {
        if args.seek.is_none() {
            if let Some(s) = std::env::var_os("LD_START_SAMPLE") {
                let target = s.to_string_lossy().parse::<u64>()?;
                // Treat `target` as the *read* position (like setup_start's
                // read_start) and place the decoder a margin ahead of it, so
                // the decode behaves exactly like a normal --start run.
                let blocksize = spec.blocksize() as u64;
                let margin = spec.blockcut() as u64 + 2 * blocksize;
                tracing::info!("LD_START_SAMPLE: reading from absolute sample {target}");
                reader.seek_samples(target)?;
                decoder.set_position_samples(target + margin);
                target
            } else {
                let s = setup_start(&mut reader, &spec, &mut decoder, args.start)?;
                tracing::info!("read start at sample {s}");
                s
            }
        } else {
            0
        }
    } else {
        0
    };

    let max_frames = args.length;
    decode_all(&mut reader, &mut writer, spec, decoder, max_frames, start_base)?;
    Ok(())
}

/// Samples of rewind retained below the decoder's earliest need. MTF
/// calibration rewinds `fdoffset` by up to ~1 field before re-decoding, and
/// the demod cache can miss blocks there, so the window must still hold that
/// data. For raw files a backward re-seek is free, but for `.flac`/`.ldf`
/// (streaming decoders) every backward seek restarts the decoder and discards
/// from byte 0, costing seconds each. Keeping this band in the window turns
/// redo windows into in-memory hits.
fn decode_rewind(spec: &DecoderSpec) -> u64 {
    // MTF/AGC redos rewind to the current field's start, which can sit far
    // back in the lead-in junk (skip chains of tens of fields). Python serves
    // these from its raw-block cache (~126M samples); we keep a generous
    // sample band below the decoder's floor so junk-era redos hit the window
    // instead of forcing a reader re-seek. Anything beyond the band still
    // falls back to a backward re-seek in `decode_all`.
    24 * spec.bytes_per_field() as u64 + 4 * spec.blocksize() as u64
}

/// Sliding-window helper shared by `decode_all` and `run_seek`: read the next
/// chunk from `reader` into `window` (advancing `base`/`read_pos`/`final_chunk`).
/// Consume one prefetched read and append its samples to the window.
fn refill_take(
    pf: &mut prefetch::PrefetchReader,
    window: &mut Vec<f32>,
    chunk: usize,
    read_pos: &mut u64,
    final_chunk: &mut bool,
    initial_base: u64,
) -> Result<()> {
    if *final_chunk {
        return Ok(());
    }
    let (buf, read) = pf.take()?;
    window.extend_from_slice(&buf[..read]);
    *read_pos += read as u64;
    if read < chunk {
        *final_chunk = true;
    }
    let _ = initial_base;
    Ok(())
}

/// Drop the head of the window so it starts at `keep`, re-seeking the reader if
/// the window was fully drained.
fn keep_window(
    reader: &mut prefetch::PrefetchReader,
    window: &mut Vec<f32>,
    keep: u64,
    base: &mut u64,
    read_pos: &mut u64,
) -> Result<()> {
    if std::env::var_os("LD_TRACE_KEEP").is_some() {
        ld_decode::teeprintln!("KEEP keep={keep} base={} read_pos={} wlen={}", *base, *read_pos, window.len());
    }
    if keep > *read_pos {
        ld_decode::teeprintln!("KEEP RESEEK-FWD keep={keep} read_pos={}", *read_pos);
        reader.seek_samples(keep)?;
        *read_pos = keep;
        window.clear();
    } else if keep > *base {
        window.drain(..(keep - *base) as usize);
    } else {
        // `keep <= base`: the decoder has not consumed anything below `base`
        // (the head is `consumed - margin - rewind_band`, and the rewind band
        // already covers any MTF-redo backward step), so everything it could
        // still need is already in the window at or above `base`. Do NOT
        // re-seek the reader backward here: for streamed `.flac`/`.ldf` input
        // a backward seek restarts the decoder and discards from byte 0
        // (seconds each), and it is never necessary because the rewind band
        // keeps the needed data in the window.
    }
    *base = keep.max(*base);
    Ok(())
}

/// Decode the whole input serially over a sliding window of f32 samples.
/// Apply `--start n` (frames): set the decoder to roughseek frame n and move
/// the reader just before its read begin so `decode_all` samples forward from
/// there. Returns the block-aligned sample position the reader now sits at.
fn setup_start(
    reader: &mut DecodeReader,
    spec: &DecoderSpec,
    decoder: &mut Decoder,
    start_frame: u64,
) -> Result<u64> {
    // Reference: `ldd.roughseek(firstframe * 2)` -> fdoffset = frame*2 fields.
    decoder.rough_seek((start_frame as i64) * 2);
    let blocksize = spec.blocksize() as u64;
    let read_start = (decoder.position() as i64
        - (spec.blockcut() as i64 + 2 * blocksize as i64))
        .max(0) as u64;
    let read_start = read_start.div_euclid(blocksize) * blocksize;
    reader.seek_samples(read_start)?;
    Ok(read_start)
}

fn decode_all(
    reader: &mut DecodeReader,
    writer: &mut DecodeWriter,
    spec: Arc<DecoderSpec>,
    mut decoder: Decoder,
    max_frames: Option<u64>,
    initial_base: u64,
) -> Result<()> {
    // Overlap the blocking sample reads with the decode: wrap the reader in
    // the prefetch worker. All reads are served FIFO by one thread, so the
    // sample stream the decoder sees is identical to the synchronous path.
    let prefetch_reader = prefetch::spawn_prefetch(DecodeReader::new(std::mem::replace(
        &mut reader.source,
        Box::new(reader::NullSource),
    )))?;
    let mut pf = prefetch_reader;

    let chunk = spec.readlen() + 4 * spec.blocksize();
    let mut window: Vec<f32> = Vec::new();
    let mut read_buffer = vec![0.0f32; chunk];
    let mut base: u64 = initial_base;
    let mut read_pos: u64 = initial_base;
    let mut final_chunk = false;
    let blocksize = spec.blocksize() as u64;

    // Maximum number of fields to write (each frame is 2 fields).
    let max_fields = max_frames.map(|f| (f * 2) as usize);

    // Bound the retained window. `keep_window` trims to the decoder's
    // earliest need minus the rewind band, so the window must hold the band
    // PLUS the decoder's look-ahead (`readlen` + prefetch, ~119 demod blocks
    // ≈ 3.8M samples); otherwise the band alone exceeds the cap, refills
    // stop, the window end never advances, and the decode stalls on
    // NeedData forever (the read position must keep up with `consumed`).
    let max_keep =
        decode_rewind(&spec) + spec.readlen() as u64 + 130 * spec.blocksize() as u64;

    let mut fields_written = 0usize;
    let mut last_consumed = 0u64;
    let mut stalled = false;
    let t_start = std::time::Instant::now();
    loop {
        // Refill while the window is below the target so the read-ahead cannot
        // outrun the decoder's need by more than `max_keep` (previously the
        // head was drained past the decoder floor to bound the window, but
        // that forced a reader re-seek whenever MTF redo rewound slightly --
        // catastrophic for streamed `.flac` input). When the decoder stalled
        // (no fields, consumed frozen) it needs data beyond the cap -- a
        // NeedData the cap-sized window cannot serve -- so refill past the
        // cap until it makes progress.
        // Refill to the cap (the trim eats one chunk per iteration, so a
        // single refill per loop can never grow the window past the band;
        // loop until the cap is reached or input ends).
        while !final_chunk && (stalled || (window.len() as u64) < (max_keep as u64)) {
            // Queue the read, then block on its data. The queue is one deep:
            // the read issued at the end of this loop iteration overlaps with
            // the decode+write below.
            if pf.outstanding_reads() == 0 {
                pf.prefetch(chunk);
            }
            refill_take(&mut pf, &mut window, chunk, &mut read_pos, &mut final_chunk, initial_base)?;
            if !final_chunk {
                pf.prefetch(chunk);
            }
        }

        let t_d0 = std::time::Instant::now();
        let (consumed, fields) = decoder.decode(&window, base, final_chunk)?;
        if std::env::var_os("LD_TIMING").is_some() {
            ld_decode::teeprintln!("DEC us={} fields={} consumed={} wlen={}", t_d0.elapsed().as_micros(), fields.len(), consumed, window.len());
        }
        if std::env::var_os("LD_TRACE_KEEP").is_some() {
            ld_decode::teeprintln!("DEC consumed={consumed} fields={} base={} wlen={}", fields.len(), base, window.len());
        }
        // A NeedData below the retained window (MTF/AGC redo rewound further
        // back than the rewind band; python's 40M-sample reader always serves
        // these from memory) can never be satisfied by refilling, which only
        // appends. Re-seek the reader backward to just before the needed
        // block: a plain file seek for raw captures, a decoder restart for
        // streamed input (rare, and correct).
        if !final_chunk && fields.is_empty() && consumed < base {
            let new_base = consumed
                .saturating_sub(2 * blocksize)
                .div_euclid(blocksize)
                * blocksize;
            if std::env::var_os("LD_TRACE_KEEP").is_some() {
                ld_decode::teeprintln!("KEEP RESEEK-BACK need={consumed} base={base} -> {new_base}");
            }
            pf.seek_samples(new_base)?;
            window.clear();
            base = new_base;
            read_pos = new_base;
            stalled = false;
            last_consumed = consumed;
            continue;
        }
        let metadata = decoder.metadata();
        let t_w0 = std::time::Instant::now();
        for field in &fields {
            writer.write_writeable(field, metadata.as_ref())?;
        }
        if std::env::var_os("LD_TIMING").is_some() {
            ld_decode::teeprintln!("WRITE fields={} us={}", fields.len(), t_w0.elapsed().as_micros());
        }
        fields_written += fields.len();
        if decoder.lead_out() {
            // Python's main.py stops right after the lead-out frame's second
            // field is written (`ldd.leadOut`), decoding nothing further.
            break;
        }
        if std::env::var_os("LD_PROG").is_some() && fields_written % 100 == 0 {
            ld_decode::teeprintln!("PROG fields={} wall={:.1}s u64={} wlen={}", fields_written, t_start.elapsed().as_secs_f64(), consumed, window.len());
        }
        if max_fields.is_some_and(|m| fields_written >= m) {
            break;
        }

        if final_chunk {
            // Drain the rest of the window: each decode() call produces at
            // most one field, so keep calling it until it reports EOF (no
            // fields produced and the read position stops advancing).
            loop {
                let (consumed, fields) = decoder.decode(&window, base, true)?;
                let metadata = decoder.metadata();
                for field in &fields {
                    writer.write_writeable(field, metadata.as_ref())?;
                }
                fields_written += fields.len();
                if decoder.lead_out() {
                    break;
                }
                if max_fields.is_some_and(|m| fields_written >= m) {
                    break;
                }
                if fields.is_empty() && consumed == last_consumed {
                    break;
                }
                last_consumed = consumed;
                if consumed >= read_pos {
                    break;
                }
            }
            break;
        }
        // The decoder reads whole demod blocks starting at a block-aligned
        // position just before `consumed - blockcut` (its read begin is
        // aligned to `blocklen` and then floored to the demod `blocksize`), so
        // keep an extra `blocklen` of margin below that in the window so the
        // next decode call sees every block it needs. Retain a rewind band
        // below the floor too (see `decode_rewind`) so MTF redo never forces
        // a backward reader re-seek.
        let keep_from = consumed
            .saturating_sub(spec.blockcut() as u64 + BLOCKSIZE as u64)
            .div_euclid(blocksize)
            * blocksize;
        let head = keep_from
            .saturating_sub(decode_rewind(&spec))
            .div_euclid(blocksize)
            * blocksize;
        keep_window(&mut pf, &mut window, head, &mut base, &mut read_pos)?;
        stalled = fields.is_empty() && consumed == last_consumed;
        last_consumed = consumed;
    }

    if fields_written != 0 {
        tracing::info!("Completed: saving JSON and exiting.");
    } else {
        tracing::info!("Completed without handling any fields.");
    }

    writer.close(decoder.metadata())?;
    Ok(())
}

/// Locate the disc frame `target` using VBI frame codes, mirroring ld-decode's
/// `seek` / `seek_getframenr`. On success the decoder is left positioned at the
/// start of the target frame (and true is returned); a false return means the
/// frame could not be found (e.g. the disc has no frame codes).
fn run_seek(
    reader: &mut DecodeReader,
    spec: &DecoderSpec,
    decoder: &mut Decoder,
    target: i64,
) -> Result<bool> {
    // The buffer must cover the demod window (readlen + 2 blocks) plus the
    // DemodCache prefetch range (~46 blocks) or decode() stalls on NeedData
    // (this loop doesn't retry the way decode_all does).
    let chunk = spec.readlen() + 56 * spec.blocksize();
    let mut buffer = vec![0.0f32; chunk];
    let blocksize = spec.blocksize() as u64;

    // Start seeking from the current decode position (port of ld-decode's
    // `startframe = firstframe`).
    let mut startfield = decoder.field_index();

    for _retry in 0..3 {
        decoder.rough_seek(startfield);

        // Decode up to 10 fields looking for the first readable VBI frame code.
        let mut got: Option<(i64, i64)> = None; // (frame_number, field_index)
        let mut prev_pos = decoder.position();
        let mut fields_seen = 0i64;

        while fields_seen < 10 && got.is_none() {
            // (Re)build a fresh window from scratch at the decoder's position.
            let read_start = (decoder.position() as i64
                - (spec.blockcut() as i64 + 2 * blocksize as i64))
                .max(0) as u64;
            let read_start = read_start.div_euclid(blocksize) * blocksize;
            reader.seek_samples(read_start)?;
            let read = reader.read(&mut buffer)?;
            let final_chunk_local = read < chunk;
            let (consumed, _wrote) = decoder.decode(&buffer[..read], read_start, final_chunk_local)?;

            if let Some(fnr) = decoder.frame_number() {
                got = Some((fnr, decoder.field_index()));
                break;
            }

            let new_pos = decoder.position();
            if new_pos > prev_pos {
                // A field was actually decoded: keep scanning.
                fields_seen += 1;
                prev_pos = new_pos;
            } else {
                // No field decoded this call (EOF, decoder wait, or a blank
                // region): stop scanning this start position.
                let _ = consumed;
                break;
            }
        }

        let Some((fnr, fieldidx)) = got else {
            // Per ld-decode: an invalid start location falls back to the start
            // of the file once, otherwise seeking failed.
            if startfield != 0 {
                startfield = 0;
                continue;
            }
            return Ok(false);
        };

        if fnr == target {
            decoder.rough_seek(fieldidx);
            tracing::info!("Finished seeking, starting at frame {}", fnr);
            return Ok(true);
        }

        // Home in: fields are two per frame.
        startfield = fieldidx + (target - fnr) * 2 - 1;
        if startfield < 0 {
            startfield = 0;
        }
    }

    Ok(false)
}
