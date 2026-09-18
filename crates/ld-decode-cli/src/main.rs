//! ld-decode command line interface: decode raw RF Laserdisc captures (NTSC)
//! to a `.tbc` picture plus `.tbc.json` sidecar.

mod async_db;
mod async_writer;
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
    /// Base name for the output .tbc and .tbc.json files; "-" streams the
    /// .tbc picture to stdout (sidecar outputs are then disabled)
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
    /// Number of demodulation worker threads (default: 3/4 of the cores). The
    /// serial tail's parallel sections get a quarter of this; `LD_PF_POOL`
    /// overrides the split.
    #[arg(long, short = 'j', default_value_t = default_threads())]
    threads: usize,
}

/// Default demod thread count: half the logical cores plus one (9 on a 16-thread
/// box). The demod pool and the serial tail's data-parallel sections overlap, so
/// the split between them matters more than the total — and the ideal split
/// moves with how much work the tail has. It was three quarters of the cores
/// when the tail still carried the EFM PLL inline; with that offloaded the tail
/// needs less headroom and giving the (memory-bound) demod more cores pays
/// again: measured `-l 5000` on a 16-thread box, 2 rounds each —
/// `-j 9` 38.29/38.20, `-j 10` 37.87/37.65, `-j 8` 37.18/36.95,
/// `-j 12` 36.19/35.40 (the old default). Re-sweep after any change that
/// moves the demod/tail balance.
fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2 + 1).max(2))
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
    // With no positional arguments, show the same usage text as `--help`
    // instead of reporting missing required arguments.
    let args = if std::env::args_os().nth(1).is_none() {
        let mut help_args: Vec<_> = std::env::args_os().take(1).collect();
        help_args.push("--help".into());
        Args::parse_from(help_args)
    } else {
        Args::parse()
    };

    // Mirror the terminal output into `<outfile>.log` (like the reference
    // ld-decode). Opened before the tracing subscriber so every line of the
    // run lands in the file; a failure only disables file logging.
    let outfile = args.outfile.clone();
    // `-` means the picture goes to stdout: no `<outfile>.*` sidecars exist
    // then, and the console log has to go to stderr so it cannot interleave
    // with the binary picture bytes on stdout.
    let to_stdout = outfile == "-";
    if !to_stdout {
        if let Err(e) = ld_decode::logging::install(std::path::Path::new(&format!("{outfile}.log"))) {
            eprintln!("WARN: cannot create log file {outfile}.log: {e:#}");
        }
    }

    let fmt = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false);
    if to_stdout {
        fmt.with_writer(std::io::stderr).init();
        ld_decode::teeprintln!(
            "NOTE: outfile is \"-\": streaming the .tbc picture to stdout; .tbc.json/.pcm/.efm/.tbc.db and .log are disabled"
        );
    } else {
        fmt.with_writer(tracing_subscriber::fmt::writer::Tee::new(
            std::io::stdout,
            LogFileWriter,
        ))
        .init();
    }

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
    tracing::info!(
        "Worker threads: {} demod budget, {} tail pool (sinc gather / assembly)",
        args.threads,
        ld_decode::tail_pool_threads()
    );

    // Resolve the FFT engine before any transform runs. Default (no
    // LD_FFT_ENGINE) is the scipy-bit-exact sse2 build; the experimental
    // engines are opt-in and logged so no run is ever ambiguous.
    let engine = ld_decode::ffi_ducc::init_engine()
        .map_err(|e| anyhow::anyhow!(e))?;
    tracing::info!(
        "FFT engine: {}{}",
        engine.name(),
        if matches!(engine, ld_decode::ffi_ducc::FftEngine::Sse2) {
            " (scipy bit-exact default)"
        } else {
            " (EXPERIMENTAL — output will differ from the reference unless the engine is verified bit-exact)"
        }
    );

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
    let luma: Box<dyn std::io::Write + Send> = if to_stdout {
        Box::new(std::io::stdout())
    } else {
        Box::new(
            File::create(format!("{outfile}.tbc"))
                .with_context(|| format!("creating {outfile}.tbc"))?,
        )
    };
    // Opened read+write: `File::create` alone is write-only on Windows, and
    // the writer re-reads the fields array at close time to prepend the JSON
    // header (os error 5 otherwise).
    let json = if to_stdout {
        None
    } else {
        Some(
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(format!("{outfile}.tbc.json"))
                .with_context(|| format!("creating {outfile}.tbc.json"))?,
        )
    };
    let audio = if args.disable_analog_audio || to_stdout {
        None
    } else {
        Some(
            File::create(format!("{outfile}.pcm"))
                .with_context(|| format!("creating {outfile}.pcm"))?,
        )
    };
    let efm = if args.no_efm || to_stdout {
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
    let db = if to_stdout {
        None
    } else {
        Some(async_db::AsyncDbWriter::create(std::path::Path::new(
            &format!("{outfile}.tbc.db"),
        ))?)
    };
    // The writer's per-field file writes run on a dedicated thread (FIFO, so
    // the byte stream is identical to the inline loop) and overlap the decode
    // serial tail instead of extending it.
    let mut writer = async_writer::AsyncDecodeWriter::new(DecodeWriter::new(
        luma,
        audio,
        efm,
        pre_efm,
        json,
        db,
    )?);

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
    let seekable = reader.is_seekable();
    decode_all(&mut reader, &mut writer, spec, decoder, max_frames, start_base, seekable)?;
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
/// the window was fully drained. The drop is logical: the head is recorded as
/// a skip offset and the buffer is only compacted (memmoved) when the dead
/// prefix exceeds a quarter of the buffer, so a steady-state field loop never
/// memmoves the ~8M-sample window per field.
fn keep_window(
    reader: &mut prefetch::PrefetchReader,
    window: &mut Vec<f32>,
    keep: u64,
    base: &mut u64,
    read_pos: &mut u64,
    skip: &mut usize,
) -> Result<()> {
    if std::env::var_os("LD_TRACE_KEEP").is_some() {
        ld_decode::teeprintln!("KEEP keep={keep} base={} read_pos={} wlen={}", *base, *read_pos, window.len());
    }
    if keep > *read_pos {
        ld_decode::teeprintln!("KEEP RESEEK-FWD keep={keep} read_pos={}", *read_pos);
        reader.seek_samples(keep)?;
        *read_pos = keep;
        window.clear();
        *skip = 0;
    } else if keep > *base {
        *skip += (keep - *base) as usize;
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
    // Amortized compaction: only memmove when the dead prefix is large
    // relative to the live tail (cost amortizes to a few percent of a
    // per-field drain). Nothing above `base` is ever read again.
    if *skip > 0 && *skip * 4 >= window.len() - *skip {
        window.drain(..*skip);
        *skip = 0;
    }
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
    writer: &mut async_writer::AsyncDecodeWriter,
    spec: Arc<DecoderSpec>,
    mut decoder: Decoder,
    max_frames: Option<u64>,
    initial_base: u64,
    seekable: bool,
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
    // For seekable raw input the band only needs to cover MTF/AGC redo
    // (~1-2 fields): a larger rewind falls back to the cheap in-file
    // RESEEK-BACK path. A smaller band shrinks the sliding window, which
    // keeps trim/refill memcpy traffic off the wall clock. Streamed FLAC/ldf
    // keeps the full 24-field band (a backward seek restarts the decoder).
    let rewind_band = if seekable {
        2 * spec.bytes_per_field() as u64 + 4 * spec.blocksize() as u64
    } else {
        decode_rewind(&spec)
    };
    let max_keep = rewind_band + spec.readlen() as u64 + 130 * spec.blocksize() as u64;

    let mut fields_written = 0usize;
    let mut last_consumed = 0u64;
    let mut stalled = false;
    // Logical head of the window: `window[skip..]` is the live data starting
    // at sample `base`. Trims only advance `skip`; the buffer is compacted
    // lazily in `keep_window` (amortized O(1) memmove per field).
    let mut skip: usize = 0;
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
        // Live window length (excluding the compacted-away prefix).
        let live_len = window.len() - skip;
        while !final_chunk && (stalled || ((window.len() - skip) as u64) < (max_keep as u64)) {
            // Queue the read, then block on its data. The queue is two deep:
            // the reads issued here overlap with the decode+write below, so
            // the main thread's take() rarely waits on disk (the refills are
            // pure I/O and FIFO-ordered, so the sample stream is identical to
            // the synchronous path).
            while pf.outstanding_reads() < 2 {
                pf.prefetch(chunk);
            }
            refill_take(&mut pf, &mut window, chunk, &mut read_pos, &mut final_chunk, initial_base)?;
        }

        let t_d0 = std::time::Instant::now();
        let (consumed, fields) = decoder.decode(&window[skip..], base, final_chunk)?;
        if std::env::var_os("LD_TIMING").is_some() {
            ld_decode::teeprintln!("DEC us={} fields={} consumed={} wlen={}", t_d0.elapsed().as_micros(), fields.len(), consumed, live_len);
        }
        if std::env::var_os("LD_TRACE_KEEP").is_some() {
            ld_decode::teeprintln!("DEC consumed={consumed} fields={} base={} wlen={}", fields.len(), base, live_len);
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
            skip = 0;
            base = new_base;
            read_pos = new_base;
            stalled = false;
            last_consumed = consumed;
            continue;
        }
        let metadata = decoder.metadata();
        let t_w0 = std::time::Instant::now();
        for field in &fields {
            writer.write_writeable(field.clone(), metadata.as_ref())?;
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
            ld_decode::teeprintln!("PROG fields={} wall={:.1}s u64={} wlen={}", fields_written, t_start.elapsed().as_secs_f64(), consumed, live_len);
        }
        if max_fields.is_some_and(|m| fields_written >= m) {
            break;
        }

        if final_chunk {
            // Drain the rest of the window: each decode() call produces at
            // most one field, so keep calling it until it reports EOF (no
            // fields produced and the read position stops advancing).
            loop {
                let (consumed, fields) = decoder.decode(&window[skip..], base, true)?;
                let metadata = decoder.metadata();
                for field in &fields {
                    writer.write_writeable(field.clone(), metadata.as_ref())?;
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
            .saturating_sub(rewind_band)
            .div_euclid(blocksize)
            * blocksize;
        keep_window(&mut pf, &mut window, head, &mut base, &mut read_pos, &mut skip)?;
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
