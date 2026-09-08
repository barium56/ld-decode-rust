// Probe: decode samples of a raw .flac with claxon (accurate sequential
// decode from byte 0) at a given offset and print stats.
use claxon::FlacReader;
use std::fs::File;
use std::io::BufReader;

fn main() {
    let path = std::env::args().nth(1).expect("path");
    let n = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(100_000usize);
    let from = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0usize);
    let mut reader = FlacReader::new(BufReader::new(File::open(&path).unwrap())).unwrap();
    let info = reader.streaminfo();
    println!(
        "bps={} channels={} rate={} samples={}",
        info.bits_per_sample,
        info.channels,
        info.sample_rate,
        info.samples.map(|s| s as u64).unwrap_or(0)
    );
    let mut out: Vec<i32> = Vec::new();
    let mut skip = from;
    let mut got = 0usize;
    let mut spare: Vec<i32> = Vec::new();
    let mut skipped_tail: Option<Vec<i32>> = None;
    while (skip > 0 || got < n) && skipped_tail.is_none() {
        let block = match reader.blocks().read_next_or_eof(std::mem::take(&mut spare)) {
            Ok(Some(b)) => b,
            _ => break,
        };
        let mut samples = block.into_buffer();
        if skip > 0 {
            if samples.len() > skip {
                samples.drain(..skip);
                skip = 0;
            } else {
                skip -= samples.len();
                continue;
            }
        }
        let take = (n - got).min(samples.len());
        out.extend_from_slice(&samples[..take]);
        got += take;
    }
    if let Ok(dump) = std::env::var("LD_PROBE_DUMP") {
        use std::io::Write;
        let mut f = std::fs::File::create(&dump).unwrap();
        for &v in &out[..got] {
            f.write_all(&(v as i16).to_le_bytes()).unwrap();
        }
    }
    let arr = &out[..got];
    let min = arr.iter().min().unwrap();
    let max = arr.iter().max().unwrap();
    let mean: i64 = arr.iter().map(|&v| v as i64).sum::<i64>() / arr.len() as i64;
    println!("n={} min={} max={} mean={}", got, min, max, mean);
    print!("first 16: ");
    for v in &out[..16] {
        print!("{v} ");
    }
    println!();
}
