//! Dedicated writer thread: `DecodeWriter` is fully self-contained state
//! (buffered output files, in-memory JSON entries, FPS counters), so per-field
//! writes can run on their own thread and overlap the decode's serial tail.
//! Commands are consumed strictly in FIFO order, so every byte written to the
//! output files (and the JSON close-time rewrite) is identical to the inline
//! call sequence. Bounded channel: the writer applies back-pressure instead of
//! growing memory unboundedly; the JSON close needs the final metadata, so
//! `close` carries it and `finish` reports errors that happened on the thread.

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use anyhow::Result;

use ld_decode::DecoderMetadata;

use crate::writer::DecodeWriter;

enum Cmd {
    Write(Box<ld_decode::WriteableField>, Option<DecoderMetadata>),
    Close(Option<DecoderMetadata>),
}

pub struct AsyncDecodeWriter {
    tx: Option<SyncSender<Cmd>>,
    handle: Option<JoinHandle<()>>,
    /// Fatal write error from the writer thread (broken pipe, full disk).
    /// Reported by `close` in place of the resulting channel disconnect.
    err: Arc<Mutex<Option<String>>>,
}

impl AsyncDecodeWriter {
    pub fn new(mut writer: DecodeWriter) -> Self {
        // Back-pressure bound: a handful of fields (~4.8MB luma each) is
        // enough to keep the writer busy without ballooning memory if the
        // decode ever outruns disk for a sustained stretch.
        const BOUND: usize = 4;
        let (tx, rx): (SyncSender<Cmd>, Receiver<Cmd>) = sync_channel(BOUND);
        let err = Arc::new(Mutex::new(None::<String>));
        let err_thread = Arc::clone(&err);
        let handle = std::thread::Builder::new()
            .name("writer".into())
            .stack_size(4 * 1024 * 1024)
            .spawn(move || {
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        Cmd::Write(field, metadata) => {
                            if let Err(e) = writer.write_writeable(&field, metadata.as_ref()) {
                                // Stop consuming: whatever broke the output
                                // (closed pipe, full disk) will break every
                                // later field too, so let the decode abort
                                // instead of running on with no output.
                                tracing::error!("writer thread: {e:#}");
                                *err_thread.lock().unwrap() = Some(format!("{e:#}"));
                                break;
                            }
                        }
                        Cmd::Close(metadata) => {
                            // Closing command: flush + json rewrite, then stop.
                            if let Err(e) = writer.close(metadata) {
                                tracing::error!("writer close: {e:#}");
                            }
                            break;
                        }
                    }
                }
            })
            .expect("failed to spawn writer thread");
        Self {
            tx: Some(tx),
            handle: Some(handle),
            err,
        }
    }

    #[inline]
    pub fn write_writeable(
        &self,
        field: ld_decode::WriteableField,
        metadata: Option<&DecoderMetadata>,
    ) -> Result<()> {
        // The metadata snapshot is small; clone it so each queued field keeps
        // exactly the value the inline path would have used.
        let owned = self
            .tx
            .as_ref()
            .expect("writer thread already joined")
            .send(Cmd::Write(Box::new(field), metadata.cloned()));
        if let Err(e) = owned {
            // Prefer the writer's own error (broken pipe, disk full) over the
            // channel disconnect it caused.
            if let Some(msg) = self.err.lock().unwrap().clone() {
                anyhow::bail!("writer: {msg}");
            }
            anyhow::bail!("writer thread gone: {e}");
        }
        Ok(())
    }

    /// Queue the close (flush + JSON rewrite) and wait for the thread.
    pub fn close(&mut self, metadata: Option<DecoderMetadata>) -> Result<()> {
        if let Some(tx) = self.tx.take() {
            if let Err(e) = tx.send(Cmd::Close(metadata)) {
                // The thread stopped on its own (a write error leaves the
                // loop): surface that error, not the channel disconnect.
                if let Some(msg) = self.err.lock().unwrap().clone() {
                    anyhow::bail!("writer: {msg}");
                }
                anyhow::bail!("writer thread gone: {e}");
            }
        }
        if let Some(handle) = self.handle.take() {
            handle.join().expect("writer thread panicked");
        }
        Ok(())
    }
}
