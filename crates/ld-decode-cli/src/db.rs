//! `.tbc.db` — SQLite metadata sidecar, port of `LDdecode.create_db_schema` /
//! `build_sqlite_metadata` and the per-field inserts in `writeout` from the
//! reference ld-decode.
//!
//! A fresh database is created per run (like the reference, which unlinks any
//! pre-existing file). One `capture` row plus one `pcm_audio_parameters` row
//! describe the whole decode; the `capture` row is inserted on the first
//! written field and re-UPDATEd after every field (the reference recomputes it
//! from `build_json()` each writeout, so the final contents only depend on the
//! last field). Each written field additionally contributes a `field_record`
//! row plus optional `vits_metrics`, `vbi` and `drop_outs` rows, mirroring the
//! per-writeout SQL in the reference.

use anyhow::{Context, Result};
use ld_decode::{DecoderMetadata, FieldInfoEntry};
use rusqlite::{params, Connection};

/// `gitBranch` / `gitCommit` reported in the `capture` table, from the
/// reference version string `release:7.3.0` (kept in sync with writer.rs).
const GIT_BRANCH: &str = "release";
const GIT_COMMIT: &str = "7.3.0";

/// Schema, mirroring `create_db_schema` in the reference (table names,
/// columns, constraints and `PRAGMA user_version = 1`).
const SCHEMA: &str = r#"
PRAGMA user_version = 1;

CREATE TABLE capture (
    capture_id INTEGER PRIMARY KEY,
    system TEXT NOT NULL CHECK (system IN ('NTSC','PAL','PAL_M')),
    decoder TEXT NOT NULL CHECK (decoder IN ('ld-decode','vhs-decode')),
    git_branch TEXT,
    git_commit TEXT,
    video_sample_rate REAL,
    active_video_start INTEGER,
    active_video_end INTEGER,
    field_width INTEGER,
    field_height INTEGER,
    number_of_sequential_fields INTEGER,
    colour_burst_start INTEGER,
    colour_burst_end INTEGER,
    is_mapped INTEGER CHECK (is_mapped IN (0,1)),
    is_subcarrier_locked INTEGER CHECK (is_subcarrier_locked IN (0,1)),
    is_widescreen INTEGER CHECK (is_widescreen IN (0,1)),
    white_16b_ire INTEGER,
    black_16b_ire INTEGER,
    blanking_16b_ire INTEGER,
    capture_notes TEXT
);

CREATE TABLE pcm_audio_parameters (
    capture_id INTEGER PRIMARY KEY REFERENCES capture(capture_id) ON DELETE CASCADE,
    bits INTEGER,
    is_signed INTEGER CHECK (is_signed IN (0,1)),
    is_little_endian INTEGER CHECK (is_little_endian IN (0,1)),
    sample_rate REAL
);

CREATE TABLE field_record (
    capture_id INTEGER NOT NULL REFERENCES capture(capture_id) ON DELETE CASCADE,
    field_id INTEGER NOT NULL,
    audio_samples INTEGER,
    decode_faults INTEGER,
    disk_loc REAL,
    efm_t_values INTEGER,
    field_phase_id INTEGER,
    file_loc INTEGER,
    is_first_field INTEGER CHECK (is_first_field IN (0,1)),
    median_burst_ire REAL,
    pad INTEGER CHECK (pad IN (0,1)),
    sync_conf INTEGER,
    ntsc_is_fm_code_data_valid INTEGER CHECK (ntsc_is_fm_code_data_valid IN (0,1)),
    ntsc_fm_code_data INTEGER,
    ntsc_field_flag INTEGER CHECK (ntsc_field_flag IN (0,1)),
    ntsc_is_video_id_data_valid INTEGER CHECK (ntsc_is_video_id_data_valid IN (0,1)),
    ntsc_video_id_data INTEGER,
    ntsc_white_flag INTEGER CHECK (ntsc_white_flag IN (0,1)),
    ac3_symbols INTEGER,
    PRIMARY KEY (capture_id, field_id)
);

CREATE TABLE vits_metrics (
    capture_id INTEGER NOT NULL,
    field_id INTEGER NOT NULL,
    b_psnr REAL,
    w_snr REAL,
    FOREIGN KEY (capture_id, field_id)
        REFERENCES field_record(capture_id, field_id) ON DELETE CASCADE,
    PRIMARY KEY (capture_id, field_id)
);

CREATE TABLE vbi (
    capture_id INTEGER NOT NULL,
    field_id INTEGER NOT NULL,
    vbi0 INTEGER NOT NULL,
    vbi1 INTEGER NOT NULL,
    vbi2 INTEGER NOT NULL,
    FOREIGN KEY (capture_id, field_id)
        REFERENCES field_record(capture_id, field_id) ON DELETE CASCADE,
    PRIMARY KEY (capture_id, field_id)
);

CREATE TABLE drop_outs (
    capture_id INTEGER NOT NULL,
    field_id INTEGER NOT NULL,
    field_line INTEGER NOT NULL,
    startx INTEGER NOT NULL,
    endx INTEGER NOT NULL,
    FOREIGN KEY (capture_id, field_id)
        REFERENCES field_record(capture_id, field_id) ON DELETE CASCADE,
    PRIMARY KEY (capture_id, field_id, field_line, startx, endx)
);

CREATE TABLE vitc (
    capture_id INTEGER NOT NULL,
    field_id INTEGER NOT NULL,
    vitc0 INTEGER NOT NULL,
    vitc1 INTEGER NOT NULL,
    vitc2 INTEGER NOT NULL,
    vitc3 INTEGER NOT NULL,
    vitc4 INTEGER NOT NULL,
    vitc5 INTEGER NOT NULL,
    vitc6 INTEGER NOT NULL,
    vitc7 INTEGER NOT NULL,
    FOREIGN KEY (capture_id, field_id)
        REFERENCES field_record(capture_id, field_id) ON DELETE CASCADE,
    PRIMARY KEY (capture_id, field_id)
);

CREATE TABLE closed_caption (
    capture_id INTEGER NOT NULL,
    field_id INTEGER NOT NULL,
    data0 INTEGER,
    data1 INTEGER,
    FOREIGN KEY (capture_id, field_id)
        REFERENCES field_record(capture_id, field_id) ON DELETE CASCADE,
    PRIMARY KEY (capture_id, field_id)
);
"#;

pub struct DbWriter {
    conn: Connection,
    /// capture_id of the inserted `capture` row (the reference inserts it on
    /// the first writeout and re-UPDATEs it afterwards).
    capture_id: Option<i64>,
}

impl DbWriter {
    /// Delete any existing file and create a fresh database with the schema.
    /// Mirrors the reference, which unlinks `<out>.tbc.db` before connecting.
    pub fn create(path: &std::path::Path) -> Result<Self> {
        if path.exists() {
            std::fs::remove_file(path)
                .with_context(|| format!("removing stale database {}", path.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("creating database {}", path.display()))?;
        conn.execute_batch(SCHEMA)
            .with_context(|| format!("initialising database {}", path.display()))?;
        Ok(Self {
            conn,
            capture_id: None,
        })
    }

    /// Record one written field, mirroring the reference `writeout`: ensure the
    /// `capture` / `pcm_audio_parameters` rows exist (insert on the first
    /// field, re-UPDATE afterwards) and insert the field's rows. All the
    /// field's SQL runs in one transaction committed per field, like the
    /// reference (which calls `commit()` at the end of every writeout).
    /// `field_id` is the 0-based index of this field among those written (the
    /// reference's `fields_written` counter, pre-increment).
    pub fn write_field(
        &mut self,
        info: &FieldInfoEntry,
        metadata: &DecoderMetadata,
        field_id: usize,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        let capture_id = match self.capture_id {
            Some(id) => {
                update_capture(&tx, id, metadata)?;
                update_pcm(&tx, id)?;
                id
            }
            None => {
                let id = insert_capture(&tx, metadata)?;
                insert_pcm(&tx, id)?;
                id
            }
        };
        self.capture_id = Some(capture_id);

        let field_id = field_id as i64;

        // decode_faults is stored as NULL when missing or 0 (reference:
        // `None if fi.get('decodeFaults') == 0 else fi.get('decodeFaults')`).
        let decode_faults: Option<i64> = match info.decode_faults {
            Some(v) if v != 0 => Some(v),
            _ => None,
        };

        tx.execute(
            "INSERT INTO field_record (
                capture_id, field_id, is_first_field, sync_conf, disk_loc,
                file_loc, median_burst_ire, field_phase_id, decode_faults,
                audio_samples, efm_t_values, ac3_symbols, pad
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                capture_id,
                field_id,
                info.is_first_field as i64,
                info.sync_conf,
                info.disk_loc,
                info.file_loc as i64,
                info.median_burst_ire,
                info.field_phase_id,
                decode_faults,
                info.audio_samples as i64,
                info.efm_t_values as i64,
                info.ac3_symbols as i64,
                0,
            ],
        )?;

        if let Some(metrics) = &info.vits_metrics {
            tx.execute(
                "INSERT INTO vits_metrics (
                    capture_id, field_id, w_snr, b_psnr
                ) VALUES (?1, ?2, ?3, ?4)",
                params![
                    capture_id,
                    field_id,
                    metrics.w_snr.unwrap_or(0.0),
                    metrics.b_psnr.unwrap_or(0.0),
                ],
            )?;
        }

        if let Some(vbi) = &info.vbi {
            if !vbi.vbi_data.is_empty() {
                // The reference pads to exactly three values.
                let mut vals = [0i64; 3];
                for (slot, v) in vals.iter_mut().zip(vbi.vbi_data.iter()) {
                    *slot = *v;
                }
                tx.execute(
                    "INSERT INTO vbi (
                        capture_id, field_id, vbi0, vbi1, vbi2
                    ) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![capture_id, field_id, vals[0], vals[1], vals[2]],
                )?;
            }
        }

        if let Some(drops) = &info.drop_outs {
            if !drops.field_line.is_empty() {
                let mut stmt = tx.prepare(
                    "INSERT INTO drop_outs (
                        capture_id, field_id, field_line, startx, endx
                    ) VALUES (?1, ?2, ?3, ?4, ?5)",
                )?;
                for i in 0..drops.field_line.len() {
                    stmt.execute(params![
                        capture_id,
                        field_id,
                        drops.field_line[i] as i64,
                        drops.startx[i] as i64,
                        drops.endx[i] as i64,
                    ])?;
                }
            }
        }

        tx.commit()?;
        Ok(())
    }
}

/// Capture-row values, mirroring `build_sqlite_metadata` reading `build_json`.
struct CaptureValues<'a> {
    system: &'a str,
    git_branch: &'static str,
    git_commit: &'static str,
    sample_rate: f64,
    active_video_start: i64,
    active_video_end: i64,
    field_width: i64,
    field_height: i64,
    number_of_sequential_fields: i64,
    colour_burst_start: i64,
    colour_burst_end: i64,
    white_16b_ire: f64,
    black_16b_ire: f64,
    blanking_16b_ire: f64,
}

impl<'a> CaptureValues<'a> {
    fn from_metadata(m: &'a DecoderMetadata) -> Self {
        Self {
            system: m.system,
            git_branch: GIT_BRANCH,
            git_commit: GIT_COMMIT,
            sample_rate: m.sample_rate,
            active_video_start: m.active_video_start,
            active_video_end: m.active_video_end,
            field_width: m.field_width as i64,
            field_height: m.field_height as i64,
            number_of_sequential_fields: m.number_of_sequential_fields as i64,
            colour_burst_start: m.colour_burst_start,
            colour_burst_end: m.colour_burst_end,
            white_16b_ire: m.white_16b_ire,
            black_16b_ire: m.black_16b_ire,
            blanking_16b_ire: m.blanking_16b_ire,
        }
    }
}

fn insert_capture(tx: &rusqlite::Transaction<'_>, m: &DecoderMetadata) -> Result<i64> {
    let v = CaptureValues::from_metadata(m);
    // Reference order of values for the 18 INSERT columns (is_mapped=0,
    // is_subcarrier_locked = (system == "NTSC"), is_widescreen=0).
    tx.execute(
        "INSERT INTO capture (
                system, decoder, git_branch, git_commit,
                video_sample_rate, active_video_start, active_video_end,
                field_width, field_height, number_of_sequential_fields,
                colour_burst_start, colour_burst_end,
                white_16b_ire, black_16b_ire, blanking_16b_ire,
                is_mapped, is_subcarrier_locked, is_widescreen
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
                      ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
            params![
                v.system,
                "ld-decode",
                v.git_branch,
                v.git_commit,
                v.sample_rate,
                v.active_video_start,
                v.active_video_end,
                v.field_width,
                v.field_height,
                v.number_of_sequential_fields,
                v.colour_burst_start,
                v.colour_burst_end,
                v.white_16b_ire,
                v.black_16b_ire,
                v.blanking_16b_ire,
                0,
                if v.system == "NTSC" { 1 } else { 0 },
                0,
            ],
    )?;
    Ok(tx.last_insert_rowid())
}

fn update_capture(tx: &rusqlite::Transaction<'_>, id: i64, m: &DecoderMetadata) -> Result<()> {
    let v = CaptureValues::from_metadata(m);
    tx.execute(
        "UPDATE capture SET
                system=?1, decoder=?2, git_branch=?3, git_commit=?4,
                video_sample_rate=?5, active_video_start=?6, active_video_end=?7,
                field_width=?8, field_height=?9, number_of_sequential_fields=?10,
                colour_burst_start=?11, colour_burst_end=?12,
                white_16b_ire=?13, black_16b_ire=?14, blanking_16b_ire=?15,
                is_mapped=?16, is_subcarrier_locked=?17, is_widescreen=?18
            WHERE capture_id = ?19",
            params![
                v.system,
                "ld-decode",
                v.git_branch,
                v.git_commit,
                v.sample_rate,
                v.active_video_start,
                v.active_video_end,
                v.field_width,
                v.field_height,
                v.number_of_sequential_fields,
                v.colour_burst_start,
                v.colour_burst_end,
                v.white_16b_ire,
                v.black_16b_ire,
                v.blanking_16b_ire,
                0,
                if v.system == "NTSC" { 1 } else { 0 },
                0,
                id,
            ],
    )?;
    Ok(())
}

fn insert_pcm(tx: &rusqlite::Transaction<'_>, capture_id: i64) -> Result<()> {
    tx.execute(
        "INSERT INTO pcm_audio_parameters (
            capture_id, bits, is_little_endian, is_signed, sample_rate
        ) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![capture_id, 16, 1, 1, 44100],
    )?;
    Ok(())
}

fn update_pcm(tx: &rusqlite::Transaction<'_>, capture_id: i64) -> Result<()> {
    tx.execute(
        "UPDATE pcm_audio_parameters SET
            bits=?1, is_little_endian=?2, is_signed=?3, sample_rate=?4
        WHERE capture_id = ?5",
        params![16, 1, 1, 44100, capture_id],
    )?;
    Ok(())
}
