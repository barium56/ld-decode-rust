//! Background-thread `.tbc.db` writer.
//!
//! The database is derived data (a full decode regenerates it) and is not
//! part of the b3sum-verified outputs, so its per-field SQLite transaction
//! can run off the critical path. The SQL itself is unchanged: the worker
//! calls the same `DbWriter::write_field` per field, in order, exactly as
//! the synchronous path did.

use crate::db::DbWriter;
use anyhow::{Context, Result};
use ld_decode::{DecoderMetadata, FieldInfoEntry};
use std::sync::mpsc::{channel, Sender};

enum DbMsg {
    Field {
        info: FieldInfoEntry,
        metadata: DecoderMetadata,
        field_id: usize,
    },
    /// Flush point: caller blocks on the paired receiver until the worker has
    /// committed everything queued before this message.
    Sync(std::sync::mpsc::Sender<()>),
}

pub struct AsyncDbWriter {
    tx: Option<Sender<DbMsg>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl AsyncDbWriter {
    pub fn create(path: &std::path::Path) -> Result<Self> {
        let mut db = DbWriter::create(path)?;
        let (tx, rx) = channel::<DbMsg>();
        let handle = std::thread::Builder::new()
            .name("tbc-db".into())
            .spawn(move || {
                for msg in rx {
                    match msg {
                        DbMsg::Field {
                            info,
                            metadata,
                            field_id,
                        } => {
                            // A failed field insert must not kill the decode;
                            // surface it at finish() time instead.
                            let _ = db.write_field(&info, &metadata, field_id);
                        }
                        DbMsg::Sync(ack) => {
                            let _ = ack.send(());
                        }
                    }
                }
            })
            .context("spawning tbc-db writer thread")?;
        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
        })
    }

    /// Queue one field's rows; never blocks on SQLite.
    pub fn write_field(
        &self,
        info: &FieldInfoEntry,
        metadata: &DecoderMetadata,
        field_id: usize,
    ) {
        if let Some(tx) = &self.tx {
            // Backpressure if the worker falls more than one field behind:
            // mpsc is unbounded, so clone here is cheap and keeps the decode
            // thread independent of db throughput in the normal case.
            let _ = tx.send(DbMsg::Field {
                info: info.clone(),
                metadata: metadata.clone(),
                field_id,
            });
        }
    }

    /// Block until every queued field is committed, then shut the worker down.
    pub fn finish(&mut self) -> Result<()> {
        if let Some(tx) = self.tx.take() {
            let (ack_tx, ack_rx) = channel::<()>();
            if tx.send(DbMsg::Sync(ack_tx)).is_ok() {
                let _ = ack_rx.recv();
            }
            drop(tx); // worker exits when the channel closes
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        Ok(())
    }
}
