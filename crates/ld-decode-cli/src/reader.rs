//! Input sample reader for raw Laserdisc RF captures. Unlike the tape
//! decoder's reader, samples are *not* normalized: ld-decode demodulates on
//! the raw 16-bit sample values, so the widened stream keeps them (`.rf`
//! float captures are scaled by 32768 to match, exactly like
//! `load_unpacked_data_float32`).

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use anyhow::{bail, Context as _, Result};
use claxon::FlacReader;

/// Input encoding of a raw capture.
#[derive(Clone, Copy, Debug)]
pub enum SampleFormat {
    /// Little-endian `i16`, one sample per word (`.s16`).
    S16Le,
    /// Little-endian `u16`, one sample per word (`.r16` / `.u16`).
    U16Le,
    /// Little-endian `f32` * 32768 (`.rf`).
    F32Le,
    /// Packed 10-bit DdD format (`.lds`): 4 samples per 5 bytes.
    Lds,
    /// Packed 10-bit `.r30` format: 3 samples per 4 bytes (deprecated).
    R30,
    /// FLAC-in-Ogg `.ldf` capture (lossless; see `LdfSource`).
    Ldf,
    /// Raw (non-Ogg) FLAC capture (`.flac`, the DdD `*.ddd.flac` files); the
    /// Python reference decodes these with PyAV resampled to s16, i.e. the
    /// raw 16-bit RF samples, so claxon's i32 samples stay unscaled.
    Flac,
}

/// Infer the sample format from the input filename extension.
pub fn infer_format(path: &str) -> Result<SampleFormat> {
    if path == "-" {
        bail!("cannot infer format from stdin; use --format");
    }
    let ext = path
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "s16" => Ok(SampleFormat::S16Le),
        "r16" | "u16" => Ok(SampleFormat::U16Le),
        "rf" => Ok(SampleFormat::F32Le),
        "lds" => Ok(SampleFormat::Lds),
        "r30" => Ok(SampleFormat::R30),
        "ldf" => Ok(SampleFormat::Ldf),
        "flac" => Ok(SampleFormat::Flac),
        other => bail!("cannot infer sample format from extension .{other}"),
    }
}

/// Open a raw sample source (regular file). Stdin is not supported for the
/// initial port. `path` is used by the `.ldf` source to reopen the file when
/// seeking.
pub fn open_source(path: &str, file: File, format: SampleFormat) -> Result<Box<dyn SampleSource>> {
    Ok(match format {
        SampleFormat::S16Le => Box::new(RawSource::new(file, 2, widen_s16)),
        SampleFormat::U16Le => Box::new(RawSource::new(file, 2, widen_u16)),
        SampleFormat::F32Le => Box::new(RawSource::new(file, 4, widen_f32)),
        SampleFormat::Lds => Box::new(PackedSource::new(file, 4, 5, |b, out| {
            let s = unpack_lds_group(b);
            out[..4].copy_from_slice(&s);
        })),
        SampleFormat::R30 => Box::new(PackedSource::new(file, 3, 4, |b, out| {
            let s = unpack_r30_group(b);
            out[..3].copy_from_slice(&s);
        })),
        SampleFormat::Ldf => open_ldf(path, file)?,
        SampleFormat::Flac => open_raw_flac(path, file, 0)?,
    })
}

/// Container of a FLAC capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Container {
    /// FLAC carried in an Ogg stream (`OggS` page header).
    Ogg,
    /// Bare FLAC bitstream (`fLaC` marker).
    Flac,
}

/// Bytes scanned for a container magic when the file does not start with one.
/// The Python reference reads `.ldf`/`.flac` through PyAV, which probes rather
/// than trusting offset 0, so a leading tag/header is transparent there.
const PROBE_WINDOW: usize = 64 * 1024;

/// First `OggS` or `fLaC` magic in `buf`, as `(container, byte offset)`.
fn find_container(buf: &[u8]) -> Option<(Container, usize)> {
    let find = |magic: &[u8]| buf.windows(magic.len()).position(|w| w == magic);
    match (find(b"OggS"), find(b"fLaC")) {
        (Some(o), Some(f)) if o <= f => Some((Container::Ogg, o)),
        (Some(_), Some(f)) => Some((Container::Flac, f)),
        (Some(o), None) => Some((Container::Ogg, o)),
        (None, Some(f)) => Some((Container::Flac, f)),
        (None, None) => None,
    }
}

/// Read as much of `file` as fits in `buf`, tolerating short reads.
fn read_fully(file: &mut File, buf: &mut [u8]) -> Result<usize> {
    let mut filled = 0usize;
    while filled < buf.len() {
        match file.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e).context("probing input container"),
        }
    }
    Ok(filled)
}

/// Describe what a file that holds neither container actually looks like.
fn unrecognised(path: &str, head: &[u8], size: u64) -> String {
    let hex = head
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ");
    let hint = match head {
        [0x1f, 0x8b, ..] => " (looks gzip-compressed)",
        [0x50, 0x4b, ..] => " (looks like a zip archive)",
        [b'I', b'D', b'3', ..] => " (ID3 tag but no FLAC stream behind it)",
        [b'R', b'I', b'F', b'F', ..] => " (looks like a RIFF/WAV file)",
        _ => "",
    };
    let detail = if head.is_empty() {
        "file is empty (0 bytes)".to_string()
    } else {
        format!("file is {size} bytes, starts with: {hex}")
    };
    format!(
        "{path}: no FLAC capture found in the first {PROBE_WINDOW} bytes{hint}\n  \
         {detail}\n  .ldf inputs must be FLAC -- Ogg-wrapped (as the Domesday \
         Duplicator writes them) or a bare FLAC stream"
    )
}

/// `.ldf` input. The documented container is FLAC-in-Ogg (`LdfSource`), but the
/// Python reference decodes `.ldf` through PyAV, which detects the container
/// itself: a `.ldf` holding a bare FLAC stream (capture tools that skip the Ogg
/// wrapper, files re-encoded later) must decode here too. Assuming `OggS` at
/// offset 0 failed those files with "No Ogg capture pattern found".
fn open_ldf(path: &str, mut file: File) -> Result<Box<dyn SampleSource>> {
    let mut head = vec![0u8; PROBE_WINDOW];
    let n = read_fully(&mut file, &mut head)?;
    head.truncate(n);
    let (container, offset) = match find_container(&head) {
        Some(found) => found,
        None => {
            let size = file.metadata().map(|m| m.len()).unwrap_or(0);
            let shown = &head[..head.len().min(16)];
            bail!("{}", unrecognised(path, shown, size));
        }
    };
    let offset = offset as u64;
    file.seek(SeekFrom::Start(offset))
        .with_context(|| format!("seeking {path}"))?;
    match container {
        Container::Ogg => Ok(Box::new(LdfSource::open(path, file, offset)?)),
        // Bare FLAC follows the `.flac` route (ffmpeg first, claxon fallback),
        // but ffmpeg only probes the stream from offset 0 -- a shifted marker
        // goes straight to claxon positioned on it.
        Container::Flac if offset == 0 => open_raw_flac(path, file, 0),
        Container::Flac => Ok(Box::new(RawFlacSource::open(path, file, offset)?)),
    }
}

/// Open a bare-FLAC capture: ffmpeg child first (C-speed decode, s16 semantics
/// identical to the Python reference's PyAV resampler), claxon fallback at
/// `offset` when ffmpeg is unavailable or cannot probe the stream.
fn open_raw_flac(path: &str, mut file: File, offset: u64) -> Result<Box<dyn SampleSource>> {
    if offset == 0 && std::env::var_os("LD_NO_FFMPEG").is_none() {
        if let Ok(src) = FfmpegFlacSource::spawn(path, 0) {
            return Ok(Box::new(src));
        }
    }
    if offset > 0 {
        file.seek(SeekFrom::Start(offset))
            .with_context(|| format!("seeking {path}"))?;
    }
    Ok(Box::new(RawFlacSource::open(path, file, offset)?))
}

/// Streams input as raw `f32` samples and seeks by sample index.
pub trait SampleSource: Send {
    /// Read up to `out.len()` samples; fewer than that means EOF.
    fn read(&mut self, out: &mut [f32]) -> Result<usize>;
    /// Seek to absolute sample `sample`.
    fn seek_samples(&mut self, sample: u64) -> Result<()>;
    /// Whether `seek_samples` is a cheap in-file seek (raw captures) rather
    /// than a decoder restart (streamed FLAC/ldf). The window manager uses
    /// this to pick the rewind-band size; it never changes the sample stream.
    fn is_seekable(&self) -> bool {
        false
    }
}

/// Fixed-width raw samples.
struct RawSource {
    file: File,
    bytes_per_sample: usize,
    widen: fn(&[u8], &mut [f32]),
}

impl RawSource {
    fn new(file: File, bytes_per_sample: usize, widen: fn(&[u8], &mut [f32])) -> Self {
        Self {
            file,
            bytes_per_sample,
            widen,
        }
    }
}

impl SampleSource for RawSource {
    fn is_seekable(&self) -> bool {
        true
    }

    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let want = out.len() * self.bytes_per_sample;
        let mut buf = vec![0u8; want];
        let mut filled = 0usize;
        while filled < want {
            let n = self.file.read(&mut buf[filled..])?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        let samples = filled / self.bytes_per_sample;
        (self.widen)(&buf[..samples * self.bytes_per_sample], &mut out[..samples]);
        Ok(samples)
    }

    fn seek_samples(&mut self, sample: u64) -> Result<()> {
        let target = sample
            .checked_mul(self.bytes_per_sample as u64)
            .context("seek offset too large")?;
        self.file.seek(SeekFrom::Start(target))?;
        Ok(())
    }
}

fn widen_s16(bytes: &[u8], out: &mut [f32]) {
    for (dst, word) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *dst = f32::from(i16::from_le_bytes([word[0], word[1]]));
    }
}

fn widen_u16(bytes: &[u8], out: &mut [f32]) {
    for (dst, word) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *dst = f32::from(u16::from_le_bytes([word[0], word[1]]));
    }
}

fn widen_f32(bytes: &[u8], out: &mut [f32]) {
    for (dst, word) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes([word[0], word[1], word[2], word[3]]) * 32768.0;
    }
}

/// Unpack one 5-byte `.lds` group into four 10-bit samples.
fn unpack_lds_group(b: &[u8]) -> [f32; 4] {
    let tenbit = [
        ((u16::from(b[0]) << 2) | (u16::from(b[1]) >> 6)) as i32,
        (((u16::from(b[1]) & 0x3F) << 4) | (u16::from(b[2]) >> 4)) as i32,
        (((u16::from(b[2]) & 0x0F) << 6) | (u16::from(b[3]) >> 2)) as i32,
        (((u16::from(b[3]) & 0x03) << 8) | u16::from(b[4])) as i32,
    ];
    tenbit.map(|t| ((t - 512) << 6) as f32)
}

/// Unpack one 4-byte `.r30` group into three 10-bit samples.
fn unpack_r30_group(b: &[u8]) -> [f32; 3] {
    let word = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    [
        f32::from((word & 0x3FF) as i16),
        f32::from(((word >> 10) & 0x3FF) as i16),
        f32::from(((word >> 20) & 0x3FF) as i16),
    ]
}

/// Generic packed 10-bit source: `samples_per_group` samples per `group_bytes`
/// bytes. A `pending` buffer holds the tail of a partially consumed group so
/// seeks to non-group-aligned sample offsets stay byte-exact.
struct PackedSource {
    file: File,
    samples_per_group: usize,
    group_bytes: usize,
    unpack: fn(&[u8], &mut [f32]),
    pending: Vec<f32>,
}

impl PackedSource {
    fn new(file: File, samples_per_group: usize, group_bytes: usize, unpack: fn(&[u8], &mut [f32])) -> Self {
        Self {
            file,
            samples_per_group,
            group_bytes,
            unpack,
            pending: Vec::new(),
        }
    }

    fn read_raw(&mut self, out: &mut [f32]) -> Result<usize> {
        let groups = out.len() / self.samples_per_group;
        let mut buf = vec![0u8; groups * self.group_bytes];
        let mut filled = 0usize;
        while filled < buf.len() {
            let n = self.file.read(&mut buf[filled..])?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        let groups = filled / self.group_bytes;
        for g in 0..groups {
            (self.unpack)(&buf[g * self.group_bytes..(g + 1) * self.group_bytes], &mut out[g * self.samples_per_group..]);
        }
        Ok(groups * self.samples_per_group)
    }
}

impl SampleSource for PackedSource {
    fn is_seekable(&self) -> bool {
        true
    }

    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let mut written = 0usize;
        // Serve any leftover samples from a non-aligned seek first.
        if !self.pending.is_empty() {
            let n = self.pending.len().min(out.len());
            out[..n].copy_from_slice(&self.pending[..n]);
            self.pending.drain(..n);
            written += n;
            if written == out.len() {
                return Ok(written);
            }
        }
        let got = self.read_raw(&mut out[written..])?;
        Ok(written + got)
    }

    fn seek_samples(&mut self, sample: u64) -> Result<()> {
        self.pending.clear();
        let byte_target = sample
            .checked_mul(self.group_bytes as u64)
            .context("seek offset too large")?;
        let start = byte_target / self.samples_per_group as u64;
        let offset = (byte_target % self.samples_per_group as u64) as usize;
        self.file.seek(SeekFrom::Start(start))?;
        if offset != 0 {
            // Read the group and keep the tail so the next read continues
            // exactly where the seek landed.
            let mut buf = vec![0u8; self.group_bytes];
            let mut filled = 0usize;
            while filled < self.group_bytes {
                let n = self.file.read(&mut buf[filled..])?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            if filled == self.group_bytes {
                let mut samples = vec![0.0f32; self.samples_per_group];
                (self.unpack)(&buf, &mut samples);
                self.pending.extend_from_slice(&samples[offset..]);
            }
        }
        Ok(())
    }
}

/// Wraps a boxed [`SampleSource`], forwarding reads and seeks.
pub struct DecodeReader {
    pub(crate) source: Box<dyn SampleSource>,
    eof: bool,
    /// Cached `source.is_seekable()` (cheap in-file seek vs decoder restart).
    seekable: bool,
}

impl DecodeReader {
    pub fn new(source: Box<dyn SampleSource>) -> Self {
        let seekable = source.is_seekable();
        Self { source, eof: false, seekable }
    }

    /// Whether `seek_samples` is a cheap in-file seek (raw file backends).
    pub fn is_seekable(&self) -> bool {
        self.seekable
    }

    pub fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        if self.eof {
            return Ok(0);
        }
        match self.source.read(out) {
            Ok(n) => Ok(n),
            Err(e) => {
                tracing::error!("{e:#}");
                self.eof = true;
                Ok(0)
            }
        }
    }

    pub fn seek_samples(&mut self, sample: u64) -> Result<()> {
        if self.eof {
            return Ok(());
        }
        if let Err(e) = self.source.seek_samples(sample) {
            tracing::error!("{e:#}");
            self.eof = true;
        }
        Ok(())
    }
}

/// `.ldf` input: an Ogg container holding the RF capture as a lossless mono
/// FLAC stream. ld-decode writes these with `-ar 40k` so FLAC accepts the
/// data, but each FLAC sample *is* one 40 MHz RF sample -- the declared rate
/// is just a container fiction. Ogg packets concatenate to a valid FLAC
/// bitstream (header packet, then one packet per frame), so we serve the
/// packet payloads to claxon as a plain byte stream.
struct LdfSource {
    path: String,
    /// Byte offset of the first Ogg page, from the container sniff (0 for the
    /// ordinary case; non-zero when the file has a leading tag/header).
    offset: u64,
    flac: FlacReader<PacketStream>,
    block: Vec<i32>,
    block_pos: usize,
}

/// Presents Ogg packet payloads as one continuous byte stream, dropping the
/// 9-byte Ogg-FLAC identification header (`0x7F "FLAC"` + version + header
/// packet count) that prefixes the first packet.
struct PacketStream {
    ogg: ogg::PacketReader<BufReader<File>>,
    current: Vec<u8>,
    pos: usize,
    skip: usize,
}

impl PacketStream {
    fn new(mut file: File, offset: u64) -> std::io::Result<Self> {
        if offset > 0 {
            file.seek(SeekFrom::Start(offset))?;
        }
        Ok(Self {
            ogg: ogg::PacketReader::new(BufReader::new(file)),
            current: Vec::new(),
            pos: 0,
            skip: 9,
        })
    }
}

impl Read for PacketStream {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        loop {
            if self.pos >= self.current.len() {
                match self.ogg.read_packet() {
                    Ok(Some(pkt)) => {
                        self.current = pkt.data;
                        self.pos = 0;
                    }
                    Ok(None) => return Ok(0),
                    Err(e) => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            e.to_string(),
                        ))
                    }
                }
            }
            if self.skip > 0 {
                let take = self.skip.min(self.current.len() - self.pos);
                self.pos += take;
                self.skip -= take;
                if self.skip > 0 {
                    continue;
                }
            }
            if self.pos < self.current.len() {
                let n = out.len().min(self.current.len() - self.pos);
                out[..n].copy_from_slice(&self.current[self.pos..self.pos + n]);
                self.pos += n;
                return Ok(n);
            }
        }
    }
}

impl LdfSource {
    fn open(path: &str, file: File, offset: u64) -> Result<Self> {
        let flac = FlacReader::new(PacketStream::new(file, offset)?)
            .context("opening .ldf FLAC stream")?;
        Ok(Self {
            path: path.to_string(),
            offset,
            flac,
            block: Vec::new(),
            block_pos: 0,
        })
    }

    /// Decode the next FLAC frame into `self.block`. Returns false at EOF.
    fn refill(&mut self) -> Result<bool> {
        match self.flac.blocks().read_next_or_eof(std::mem::take(&mut self.block)) {
            Ok(Some(block)) => {
                self.block = block.into_buffer();
                self.block_pos = 0;
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(e) => Err(anyhow::anyhow!("FLAC decode error: {e}")),
        }
    }
}

/// `.flac` input: a raw (non-Ogg) FLAC stream of the RF capture, decoded with
/// claxon exactly like the `.ldf` path (samples kept unscaled: the Python
/// reference resamples to s16, which for 16-bit RF captures preserves the
/// sample values).
struct RawFlacSource {
    path: String,
    /// Byte offset of the FLAC marker, from the container sniff (0 for ordinary
    /// `.flac` files; non-zero for a `.ldf` holding a shifted FLAC stream).
    offset: u64,
    flac: FlacReader<BufReader<File>>,
    block: Vec<i32>,
    block_pos: usize,
}

impl RawFlacSource {
    fn open(path: &str, mut file: File, offset: u64) -> Result<Self> {
        if offset > 0 {
            file.seek(SeekFrom::Start(offset))
                .with_context(|| format!("seeking {path}"))?;
        }
        let flac = FlacReader::new(BufReader::new(file)).context("opening .flac FLAC stream")?;
        Ok(Self {
            path: path.to_string(),
            offset,
            flac,
            block: Vec::new(),
            block_pos: 0,
        })
    }

    /// Decode the next FLAC frame into `self.block`. Returns false at EOF.
    fn refill(&mut self) -> Result<bool> {
        match self.flac.blocks().read_next_or_eof(std::mem::take(&mut self.block)) {
            Ok(Some(block)) => {
                self.block = block.into_buffer();
                self.block_pos = 0;
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(e) => Err(anyhow::anyhow!("FLAC decode error: {e}")),
        }
    }
}

/// Put the `ffmpeg` decode child below this process's priority.
///
/// The child only has to sustain the reader's ~137 MB/s of s16 and its decode
/// rate is ~234 MB/s (measured on the CLV capture: `-threads auto` 234,
/// `-threads 2` 231, `-threads 1` 123 MB/s against the 137 needed), so it has
/// headroom to lose the scheduling race and still keep the pipe full. That
/// matters because the demod pool is the binding constraint and everything it
/// loses to another runnable thread inflates `dcpu` directly: the `.flac` path
/// costs ~8 core-ms/field of demod inflation against s16, worth 9.6% FPS, and
/// the mechanism is visible in `LD_TIMING` (`dcpu` 80.6 -> 88.6, `asm`
/// 1.47 -> 1.58 — contention, not reader starvation).
///
/// Priority cannot change the decode's output: FLAC is lossless and ffmpeg's
/// sample values do not depend on when its threads run.
///
/// Unix uses `setpriority(PRIO_PROCESS, pid, 19)`; Windows
/// `SetPriorityClass(BELOW_NORMAL_PRIORITY_CLASS)`.
#[cfg(windows)]
fn demote_child_priority(child: &std::process::Child) {
    const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
    use std::os::windows::io::AsRawHandle;
    extern "system" {
        fn SetPriorityClass(hprocess: *mut core::ffi::c_void, class: u32) -> i32;
    }
    // Best effort: a failure just leaves the child at normal priority.
    unsafe {
        SetPriorityClass(
            child.as_raw_handle() as *mut core::ffi::c_void,
            BELOW_NORMAL_PRIORITY_CLASS,
        );
    }
}

#[cfg(unix)]
fn demote_child_priority(child: &std::process::Child) {
    const PRIO_PROCESS: i32 = 0;
    const PRIO_LOWEST: i32 = 19;
    extern "C" {
        fn setpriority(which: i32, who: u32, prio: i32) -> i32;
    }
    unsafe {
        setpriority(PRIO_PROCESS, child.id(), PRIO_LOWEST);
    }
}

/// `.flac` input decoded by an `ffmpeg` child process (C-speed FLAC decode,
/// output as raw s16le mono on stdout — the same sample values the Python
/// reference gets from its PyAV s16 resampler). Falls back to claxon (see
/// `RawFlacSource`) if ffmpeg cannot be spawned.
struct FfmpegFlacSource {
    path: String,
    child: std::process::Child,
    out: std::io::BufReader<std::process::ChildStdout>,
    out_bytes: Vec<u8>,
    out_pos: usize,
    /// Samples to skip before serving the first sample (see `spawn`).
    discard: u64,
}

impl FfmpegFlacSource {
    /// Spawn `ffmpeg` decoding `path`, positioned at (or just before) sample
    /// `from_sample`. The container rate is 40 kHz fiction = one RF sample per
    /// stream unit; `-ss` seeks by stream time.
    ///
    /// `-ss` is placed AFTER `-i` (output seeking): on these raw FLAC files the
    /// demuxer has no seek table, so input-side `-ss` guesses a byte offset from
    /// average bitrate and lands hundreds of thousands of samples late. Output
    /// seeking decodes from the start and discards to the exact stream time
    /// (the same accurate semantics the Python reference gets from PyAV).
    ///
    /// ffmpeg's output `-ss` seek (decode from the start, discard to the
    /// target time) is proven accurate only up to ~34.6B samples (the deepest
    /// verified seek). Beyond that it silently outputs nothing: the discard is
    /// bounded by the declared stream length, which these DdD files cap at
    /// ~37.1B samples while the real frame data extends ~3x further. When the
    /// target exceeds the proven range, spawn from the start (`-ss 0`) and
    /// discard the exact sample count in the reader — same full-file decode
    /// cost, guaranteed to deliver.
    fn spawn(path: &str, from_sample: u64) -> Result<Self> {
        if std::env::var_os("LD_TRACE_SEEK").is_some() {
            ld_decode::teeprintln!("FFMPEG SPAWN from_sample={from_sample}");
        }
        const MAX_SS_SAMPLE: u64 = 35_000_000_000;
        let (seconds, discard) = if from_sample > MAX_SS_SAMPLE {
            (0.0, from_sample)
        } else {
            (from_sample as f64 / 40_000.0, 0)
        };
        if std::env::var_os("LD_TRACE_SEEK").is_some() {
            ld_decode::teeprintln!("FFMPEG SPAWN -ss={seconds:.6}s discard={discard}");
        }
        let mut cmd = std::process::Command::new("ffmpeg");
        cmd.arg("-nostdin")
            .arg("-v")
            .arg("error")
            .arg("-i")
            .arg(path)
            .arg("-ss")
            .arg(format!("{seconds:.6}"))
            .arg("-f")
            .arg("s16le")
            .arg("-ac")
            .arg("1")
            .arg("pipe:1")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());
        let mut child = cmd.spawn().context("spawning ffmpeg for .flac decode")?;
        demote_child_priority(&child);
        let stdout = child.stdout.take().context("ffmpeg stdout")?;
        Ok(Self {
            path: path.to_string(),
            child,
            out: std::io::BufReader::with_capacity(1 << 20, stdout),
            out_bytes: Vec::new(),
            out_pos: 0,
            discard,
        })
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// Pull the next raw s16 block from ffmpeg into `out_bytes`.
    fn refill(&mut self) -> std::io::Result<bool> {
        self.out_bytes.resize(1 << 20, 0);
        let mut got = 0usize;
        while got < self.out_bytes.len() {
            match self.out.read(&mut self.out_bytes[got..]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        if std::env::var_os("LD_TRACE_SEEK").is_some() {
            ld_decode::teeprintln!("FFMPEG PIPE refill got={got}");
        }
        if got == 0 {
            let _ = self.child.wait();
            return Ok(false);
        }
        self.out_bytes.truncate(got);
        self.out_pos = 0;
        Ok(true)
    }
}

impl SampleSource for FfmpegFlacSource {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let mut written = 0usize;
        while written < out.len() {
            if self.out_pos >= self.out_bytes.len() && !self.refill()? {
                break;
            }
            if self.discard > 0 {
                let avail = (self.out_bytes.len() - self.out_pos) / 2;
                let skip = (avail as u64).min(self.discard) as usize;
                self.out_pos += skip * 2;
                self.discard -= skip as u64;
                continue;
            }
            let n = (out.len() - written).min((self.out_bytes.len() - self.out_pos) / 2);
            for i in 0..n {
                let lo = self.out_bytes[self.out_pos + i * 2];
                let hi = self.out_bytes[self.out_pos + i * 2 + 1];
                out[written + i] = f32::from(i16::from_le_bytes([lo, hi]));
            }
            written += n;
            self.out_pos += n * 2;
        }
        Ok(written)
    }

    fn seek_samples(&mut self, sample: u64) -> Result<()> {
        let target = flac_pyseek(sample);
        self.kill();
        *self = Self::spawn(&self.path, target)?;
        Ok(())
    }
}

impl Drop for FfmpegFlacSource {
    fn drop(&mut self) {
        self.kill();
    }
}

impl SampleSource for RawFlacSource {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let mut written = 0usize;
        while written < out.len() {
            if self.block_pos >= self.block.len() && !self.refill()? {
                break;
            }
            let n = (out.len() - written).min(self.block.len() - self.block_pos);
            for i in 0..n {
                out[written + i] = self.block[self.block_pos + i] as f32;
            }
            written += n;
            self.block_pos += n;
        }
        Ok(written)
    }

    fn seek_samples(&mut self, sample: u64) -> Result<()> {
        // Same restart-and-discard strategy as the `.ldf` source: FLAC frames
        // are independently decodable, but sample-level seeks need a decode
        // from the start (the Python reference does the same for small seeks).
        if std::env::var_os("LD_TRACE_SEEK").is_some() {
            ld_decode::teeprintln!("FLAC SEEK to {sample}");
        }
        let sample = flac_pyseek(sample);
        let mut file = File::open(&self.path)
            .with_context(|| format!("reopening {} for seek", self.path))?;
        if self.offset > 0 {
            file.seek(SeekFrom::Start(self.offset))?;
        }
        self.flac = FlacReader::new(BufReader::new(file))
            .context("reopening .flac FLAC stream")?;
        self.block.clear();
        self.block_pos = 0;

        let mut scratch = vec![0f32; 65536];
        let mut remaining = sample;
        while remaining > 0 {
            let want = remaining.min(scratch.len() as u64) as usize;
            let n = self.read(&mut scratch[..want])?;
            if n == 0 {
                break;
            }
            remaining -= n as u64;
        }
        Ok(())
    }
}

/// Python-compat shim for the LoadLDF `.flac` seek artifact (see
/// NOTES-flac-stock-mismatch.md). The reference decoder seeks via
/// PyAV `container.seek((sample / 1000 - sample_rate) us)` (microseconds!),
/// which for these DdD captures (40k container-rate fiction over 1:1 RF
/// samples) lands ~1.29 s in -- at the FLAC frame containing that time,
/// i.e. a frame with pts L = 4096 * floor((sample/1000 - 40000) / 102400)
/// samples (4096-sample FLAC frames, 0.04 container samples per us). The
/// reader then computes base = pts * 1000 (a 1000x scale error) and discards
/// `sample - base` samples, so the delivered stream starts at
/// `sample - 999 * L` instead of `sample`. Decoding the same bytes with the
/// same decode pipeline is bit-exact, so reproducing this delivery offset is
/// what makes rust's .flac output md5-identical to the python reference's.
///
/// `LD_NO_FLAC_PYSEEK=1` restores honest sample-exact seeking;
/// `LD_FLAC_SHIFT=n` overrides the computed shift for A/B tests.
fn flac_pyseek(sample: u64) -> u64 {
    if std::env::var_os("LD_NO_FLAC_PYSEEK").is_some() {
        return sample;
    }
    if let Ok(s) = std::env::var("LD_FLAC_SHIFT") {
        if let Ok(sh) = s.parse::<u64>() {
            return sample.saturating_sub(sh);
        }
    }
    let x = (sample / 1000).saturating_sub(40_000); // seek offset in us
    let l = (x / 102_400) * 4096; // pts of the FLAC frame the seek lands on
    let shift = 999 * l;
    sample.saturating_sub(shift)
}

impl SampleSource for LdfSource {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let mut written = 0usize;
        while written < out.len() {
            if self.block_pos >= self.block.len() && !self.refill()? {
                break;
            }
            let n = (out.len() - written).min(self.block.len() - self.block_pos);
            for i in 0..n {
                out[written + i] = self.block[self.block_pos + i] as f32;
            }
            written += n;
            self.block_pos += n;
        }
        Ok(written)
    }

    fn seek_samples(&mut self, sample: u64) -> Result<()> {
        // FLAC frames are independently decodable, but mapping a sample offset
        // to an Ogg page needs the granule positions; the simple correct
        // approach is to restart the decoder and discard samples (the Python
        // reference does the same for small seeks).
        let file = File::open(&self.path)
            .with_context(|| format!("reopening {} for seek", self.path))?;
        self.flac = FlacReader::new(PacketStream::new(file, self.offset)?)
            .context("reopening .ldf FLAC stream")?;
        self.block.clear();
        self.block_pos = 0;

        let mut scratch = vec![0f32; 65536];
        let mut remaining = sample;
        while remaining > 0 {
            let want = remaining.min(scratch.len() as u64) as usize;
            let n = self.read(&mut scratch[..want])?;
            if n == 0 {
                break;
            }
            remaining -= n as u64;
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_container_detects_magic_at_offset_zero() {
        assert_eq!(find_container(b"OggS\x00\x02\0\0"), Some((Container::Ogg, 0)));
        assert_eq!(
            find_container(b"fLaC\x00\x00\x00\x22"),
            Some((Container::Flac, 0))
        );
    }

    #[test]
    fn find_container_tolerates_a_leading_tag() {
        let mut ogg = b"junk".to_vec();
        ogg.extend_from_slice(b"OggS");
        assert_eq!(find_container(&ogg), Some((Container::Ogg, 4)));

        let mut flac = b"ID3\x04\x00\x00".to_vec();
        flac.extend_from_slice(&[0u8; 64]);
        flac.extend_from_slice(b"fLaC");
        assert_eq!(find_container(&flac), Some((Container::Flac, 70)));
    }

    #[test]
    fn find_container_reports_nothing_for_other_data() {
        assert_eq!(find_container(b""), None);
        assert_eq!(find_container(&[0u8; 4096]), None);
        // Raw s16 RF samples: neither container.
        assert_eq!(find_container(&[0x11, 0x22, 0x33, 0x44, 0x55]), None);
    }

    #[test]
    fn unrecognised_names_the_container_it_sees() {
        let gz = unrecognised("x.ldf", &[0x1f, 0x8b, 0x08, 0x00], 1234);
        assert!(gz.contains("gzip"), "{gz}");
        assert!(gz.contains("1f 8b 08 00"), "{gz}");
        assert!(gz.contains("1234"), "{gz}");
        assert!(unrecognised("x.ldf", &[], 0).contains("empty"));
    }
}

/// No-op source used only for Send assertions in tests/prefetch wiring.
pub struct NullSource;

impl SampleSource for NullSource {
    fn read(&mut self, _out: &mut [f32]) -> anyhow::Result<usize> {
        Ok(0)
    }
    fn seek_samples(&mut self, _sample: u64) -> anyhow::Result<()> {
        Ok(())
    }
}
