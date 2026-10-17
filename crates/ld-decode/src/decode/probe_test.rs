//! Hermetic demod stage A/B probe (test-only).
//!
//! Usage: set env, then `cargo test -p ld-decode probe_block -- --ignored`.
//!
//! Reads a raw block window plus the RFVideo / MTF / FVideo filters as binary
//! bins and runs the exact NTSC video demod pipeline (indata_fft -> rfvideo
//! -> mtf_pow -> hilbert -> unwrap -> clip -> demod_fft -> out_video),
//! dumping every f64 intermediate so a Python replica can be diffed stage by
//! stage. Env:
//!   LD_PROBE_RAW      f64 little-endian, blocklen samples
//!   LD_PROBE_RFVIDEO  Complex64 f64 (re,im pairs), blocklen
//!   LD_PROBE_MTF      Complex64 f64, blocklen
//!   LD_PROBE_FVIDEO   Complex64 f64, blocklen
//!   LD_PROBE_FREQ_HZ  text (f64)
//!   LD_PROBE_MTF_LEVEL text (f64), default 1.0
//!   LD_PROBE_OUT      directory for stage dumps
//!
//! Dumps: indata_fft, filtered, hilbert, demod, demod_fft, out_video (f64
//! bins), out_video_f32 (f32 bin), mtf_pow (complex f64).
#![cfg(test)]

use crate::decode::demodblock::{compute_mtf_pow, unwrap_hilbert};
use crate::ffi_ducc;
use crate::spec::np_cmul;
use std::path::PathBuf;

fn rd(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn rd_f64(name: &str) -> Vec<f64> {
    let p = std::env::var(name).expect(name);
    let b = rd(&p);
    assert_eq!(b.len() % 8, 0, "{name} not f64");
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

type C64 = rustfft::num_complex::Complex64;

fn rd_cf(name: &str) -> Vec<C64> {
    let p = std::env::var(name).expect(name);
    let b = rd(&p);
    assert_eq!(b.len() % 16, 0, "{name} not c128");
    b.chunks_exact(16)
        .map(|c| {
            let re = f64::from_le_bytes(c[0..8].try_into().unwrap());
            let im = f64::from_le_bytes(c[8..16].try_into().unwrap());
            C64::new(re, im)
        })
        .collect()
}

fn wr(dir: &str, name: &str, bytes: &[u8]) {
    std::fs::write(PathBuf::from(dir).join(name), bytes).unwrap();
}

fn wr_f64(dir: &str, name: &str, v: &[f64]) {
    let mut b = Vec::with_capacity(v.len() * 8);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    wr(dir, name, &b);
}

fn wr_cf(dir: &str, name: &str, v: &[C64]) {
    let mut b = Vec::with_capacity(v.len() * 16);
    for x in v {
        b.extend_from_slice(&x.re.to_le_bytes());
        b.extend_from_slice(&x.im.to_le_bytes());
    }
    wr(dir, name, &b);
}

#[test]
#[ignore = "needs LD_PROBE_* dumps of a block window and its filters"]
fn probe_block() {
    let out = std::env::var("LD_PROBE_OUT").expect("LD_PROBE_OUT");
    let raw = rd_f64("LD_PROBE_RAW");
    let rfvideo = rd_cf("LD_PROBE_RFVIDEO");
    let mtf = rd_cf("LD_PROBE_MTF");
    let fvideo = rd_cf("LD_PROBE_FVIDEO");
    let freq_hz: f64 = std::env::var("LD_PROBE_FREQ_HZ")
        .expect("LD_PROBE_FREQ_HZ")
        .parse()
        .unwrap();
    let mtf_level: f64 = std::env::var("LD_PROBE_MTF_LEVEL")
        .unwrap_or_else(|_| "1.0".into())
        .parse()
        .unwrap();
    assert_eq!(raw.len(), rfvideo.len());
    assert_eq!(raw.len(), mtf.len());
    assert_eq!(raw.len(), fvideo.len());

    let indata_fft = ffi_ducc::fft_real_full(&raw);
    wr_cf(&out, "indata_fft.bin", &indata_fft);

    let mut filtered = indata_fft.clone();
    for (v, &f) in filtered.iter_mut().zip(&rfvideo) {
        *v = np_cmul(*v, f);
    }
    let mtf_pow = compute_mtf_pow(&mtf, mtf_level);
    for (v, &f) in filtered.iter_mut().zip(&mtf_pow) {
        *v = np_cmul(*v, f);
    }
    wr_cf(&out, "filtered.bin", &filtered);
    wr_cf(&out, "mtf_pow.bin", &mtf_pow);

    let hilbert = ffi_ducc::ifft(&filtered);
    wr_cf(&out, "hilbert.bin", &hilbert);

    // Dump the conjugate products feeding unwrap_hilbert for direct diffing.
    let mut prods: Vec<C64> = Vec::with_capacity(hilbert.len() - 1);
    for i in 1..hilbert.len() {
        prods.push(np_cmul(hilbert[i], hilbert[i - 1].conj()));
    }
    wr_cf(&out, "prod.bin", &prods);

    let demod = unwrap_hilbert(&hilbert, freq_hz);
    wr_f64(&out, "demod.bin", &demod);

    let clipped: Vec<f64> = demod
        .iter()
        .map(|&d| d.clamp(1_500_000.0, freq_hz * 0.75))
        .collect();
    let demod_fft = ffi_ducc::fft_real_full(&clipped);
    wr_cf(&out, "demod_fft.bin", &demod_fft);

    let mut ch: Vec<C64> = Vec::with_capacity(demod_fft.len());
    for (&s, &f) in demod_fft.iter().zip(&fvideo) {
        ch.push(np_cmul(s, f));
    }
    let ifft_res = ffi_ducc::ifft(&ch);
    let out_f64: Vec<f64> = ifft_res.iter().map(|v| v.re).collect();
    wr_f64(&out, "out_video.bin", &out_f64);
    let out_f32: Vec<f32> = out_f64.iter().map(|&v| v as f32).collect();
    let mut b = Vec::with_capacity(out_f32.len() * 4);
    for x in out_f32 {
        b.extend_from_slice(&x.to_le_bytes());
    }
    wr(&out, "out_video_f32.bin", &b);
}