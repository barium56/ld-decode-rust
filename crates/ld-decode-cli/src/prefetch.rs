//! Async prefetching wrapper around `DecodeReader`.
//!
//! The decode loop is serial: refill the sample window (blocking read),
//! decode one field, write it out. The refill is pure I/O — independent of
//! the decode — so it runs on a background thread while the decoder works.
//! The wrapper preserves the exact byte order of the synchronous reader: one
//! worker serves requests FIFO and `take()` returns the head request's data,
//! so the decoder observes an identical sample stream. Seeks flush all
//! pending reads (matching `DecodeReader::seek_samples` semantics).

use anyhow::{bail, Context, Result};
use std::sync::mpsc::{channel, Receiver, Sender};

use super::reader::DecodeReader;

enum Request {
    /// Read up to `len` samples into a freshly allocated buffer; reply with
    /// (buffer, count). A count of 0 is a normal EOF marker.
    Read(usize),
    /// Seek to `sample` (flushing all pending reads); reply Done when done.
    Seek(u64),
    /// Quit the worker.
    Quit,
}

enum Reply {
    /// Filled buffer plus the number of valid samples.
    Data(Vec<f32>, usize),
    /// Seek completed.
    Done,
    /// The reader hit an error (already logged by `DecodeReader`).
    Err(String),
}

struct Worker {
    reader: DecodeReader,
    rx: Receiver<Request>,
    tx: Sender<Reply>,
}

impl Worker {
    fn run(mut self) {
        while let Ok(req) = self.rx.recv() {
            match req {
                Request::Read(len) => {
                    let mut buf = vec![0.0f32; len];
                    // `DecodeReader::read` logs its own errors and returns 0
                    // (EOF) after them, exactly like the synchronous path.
                    let n = self.reader.read(&mut buf).unwrap_or(0);
                    if self.tx.send(Reply::Data(buf, n)).is_err() {
                        return;
                    }
                }
                Request::Seek(sample) => {
                    self.reader.seek_samples(sample);
                    if self.tx.send(Reply::Done).is_err() {
                        return;
                    }
                }
                Request::Quit => return,
            }
        }
    }
}

/// Handle for the main thread. `take()` returns data for the request at the
/// head of the FIFO — the same order the synchronous loop would have issued
/// them.
pub struct PrefetchReader {
    tx: Sender<Request>,
    rx: Receiver<Reply>,
    /// Requests sent but not yet consumed; keeps the FIFO aligned.
    outstanding: usize,
    failed: bool,
}

// The worker owns the reader exclusively; the handle only passes owned
// buffers across the channel.
fn _assert_send() {
    fn is_send<T: Send>() {}
    is_send::<PrefetchReader>();
    fn requires_send<T: Send>(_: T) {}
    requires_send::<DecodeReader>(DecodeReader::new(Box::new(crate::reader::NullSource)));
}

impl PrefetchReader {
    /// Spawn the worker; the reader moves to the background thread.
    pub fn new(reader: DecodeReader) -> Result<Self> {
        let (req_tx, req_rx) = channel::<Request>();
        let (rep_tx, rep_rx) = channel::<Reply>();
        let worker = Worker {
            reader,
            rx: req_rx,
            tx: rep_tx,
        };
        std::thread::Builder::new()
            .name("prefetch-reader".into())
            .spawn(move || worker.run())
            .context("spawn prefetch reader thread")?;
        Ok(Self {
            tx: req_tx,
            rx: rep_rx,
            outstanding: 0,
            failed: false,
        })
    }

    /// Queue one read of `len` samples. Non-blocking.
    pub fn prefetch(&mut self, len: usize) {
        if self.failed {
            return;
        }
        if self.tx.send(Request::Read(len)).is_err() {
            self.failed = true;
            return;
        }
        self.outstanding += 1;
    }

    /// Block until the head request's data arrives; returns (buffer, count)
    /// where `count <= buffer.len()` samples are valid (count 0 = EOF).
    pub fn take(&mut self) -> Result<(Vec<f32>, usize)> {
        if self.failed {
            bail!("prefetch reader previously failed");
        }
        if self.outstanding == 0 {
            bail!("take() with no outstanding request");
        }
        match self.rx.recv() {
            Ok(Reply::Data(buf, n)) => {
                self.outstanding -= 1;
                Ok((buf, n))
            }
            Ok(Reply::Done) => {
                self.failed = true;
                bail!("prefetch reader: unexpected seek reply");
            }
            Ok(Reply::Err(e)) => {
                self.failed = true;
                bail!("prefetch reader error: {e}");
            }
            Err(_) => {
                self.failed = true;
                bail!("prefetch reader died");
            }
        }
    }

    /// Seek (flushing pending reads). Blocks until the seek completes, so the
    /// stream position after this call matches the synchronous reader.
    pub fn seek_samples(&mut self, sample: u64) -> Result<()> {
        if self.failed {
            bail!("prefetch reader previously failed");
        }
        // Discard pending read replies (FIFO; at most the queue depth).
        while self.outstanding > 0 {
            match self.rx.recv() {
                Ok(Reply::Data(..)) => self.outstanding -= 1,
                Ok(Reply::Done) | Ok(Reply::Err(_)) | Err(_) => {
                    self.failed = true;
                    bail!("prefetch reader: unexpected reply during seek flush");
                }
            }
        }
        if self.tx.send(Request::Seek(sample)).is_err() {
            self.failed = true;
            bail!("prefetch reader died");
        }
        match self.rx.recv() {
            Ok(Reply::Done) => Ok(()),
            _ => {
                self.failed = true;
                bail!("prefetch reader: seek failed");
            }
        }
    }

    /// Number of reads queued but not yet consumed.
    pub fn outstanding_reads(&self) -> usize {
        self.outstanding
    }

    /// Whether the worker is gone (worker thread exits with the process).
    pub fn is_failed(&self) -> bool {
        self.failed
    }
}

impl Drop for PrefetchReader {
    fn drop(&mut self) {
        let _ = self.tx.send(Request::Quit);
    }
}

/// Spawn the prefetch worker for `reader`.
pub fn spawn_prefetch(reader: DecodeReader) -> Result<PrefetchReader> {
    PrefetchReader::new(reader)
}

/// Queue depth: one in-flight read hides a full refill behind one field
/// decode; deeper queues only add memory.
pub const PREFETCH_DEPTH: usize = 1;
