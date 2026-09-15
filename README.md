# ld-decode-rust

A from-scratch Rust port of the NTSC LaserDisc RF decoder from
[ld-decode](https://github.com/happycube/ld-decode) 7.3.0.

The goal is not "a decoder that produces similar pictures". The goal is a
decoder that produces **byte-identical output files** to the Python original:
same `.tbc` luma, same `.pcm` audio, same `.efm` digital-audio bits, same
`.tbc.json` metadata. Any change is measured against that bar, and a change that
moves a single byte is a bug, not a rounding detail. Speed is the reason the port
exists; parity is the constraint it is built under.

## What it does today

- **NTSC only.** PAL is deliberately not ported: it needs pilot-based lineloc
  refinement, a different V4300D notch and different field constants, and none
  of it has been verified. The code assumes NTSC.
- **Byte-parity verified end-to-end** on three input paths:
  - a full `s16` capture (168,800 fields) — 4/4 outputs identical,
  - a full `.ddd.flac` capture decoded from frame 300 — 4/4 outputs identical,
  - a full `.ldf` capture (50,722 fields) — 4/4 outputs identical.
- **Roughly 6–14x faster than the Python reference.** About 28–29 FPS at `-j 12`
  on a Ryzen 7 5800X3D (10,000 fields in ~6 minutes) against ~2–5 FPS for
  Python 3.12 + numpy 2.4.6 + scipy 1.18.0 on the same box.
- Produces `.tbc`, `.tbc.json`, `.pcm`, `.efm` and `.tbc.db`, plus a `.log` that
  mirrors the console output.

## Quick start

Prerequisites, all on Windows x86-64 (see [Why Windows only](#why-windows-only)):

- Rust **nightly** (`rustup toolchain install nightly-2026-08-29`), MSVC
  toolchain, and LLVM's `clang-cl` (the build falls back to `clang-cl` on
  `PATH`, and otherwise expects `C:\Program Files\LLVM\bin\clang-cl.exe`).

```bash
cargo build --release                     # produces target/release/ld-decode.exe
```

Decode a capture:

```bash
# whole file
target/release/ld-decode.exe -j 12 "capture.ddd.flac" out

# start at frame 1000, decode 1000 frames (frames, not samples; 2 fields each)
target/release/ld-decode.exe -j 12 -s 1000 -l 1000 "capture.s16" out
```

`-j 12` is the measured sweet spot on an 8-core/16-thread machine; the default
already scales with the core count (`3/4` of the logical cores, minimum 2).

## Command line

```
ld-decode.exe [OPTIONS] <INFILE> <OUTFILE>
```

`INFILE` is what to decode, `OUTFILE` is the base name for the outputs
(`out` produces `out.tbc`, `out.tbc.json`, `out.pcm`, `out.efm`, `out.tbc.db`
and `out.log`).

`OUTFILE` may also be `-`, which streams the `.tbc` picture to stdout as raw
little-endian `uint16` and disables every sidecar output (the JSON header
rewrite needs a seekable file, and the log would otherwise interleave with the
picture bytes — in this mode the console log goes to stderr instead):

```bash
# pipe the picture straight into something else
ld-decode.exe -j 12 "capture.s16" - > out.tbc
ld-decode.exe -j 12 "capture.s16" - | ffmpeg -f rawvideo ...
```

The bytes are identical to what a normal run writes, so `out.tbc` above is the
same file the `out` form would have produced. If the reader at the other end
closes the pipe early the decode stops immediately with an error instead of
decoding the rest of the disc into a broken pipe (same for a filled disk).

| Option | Default | Meaning |
| --- | --- | --- |
| `-s`, `--start <n>` | `0` | Rough jump to **frame** `n` of the capture (2 fields per frame). |
| `-l`, `--length <n>` | until EOF | Decode at most `n` **frames**. |
| `-S`, `--seek <n>` | off | Seek to a specific VBI frame number; needs readable CAV/CLV frame codes on the disc and fails gracefully without them. |
| `-j`, `--threads <n>` | `3/4` of logical cores | Demodulation worker threads. The serial tail's parallel sections get a quarter of this, so the split matters more than the total. |
| `--inputfreq <MHz>` | `40` | Input sample rate. |
| `--mtf <x>` | `1.0` | MTF compensation multiplier. |
| `--mtf-offset <x>` | `0.0` | MTF compensation offset. |
| `--deemp-str <x>` | `1.0` | De-emphasis strength multiplier. |
| `--deemp <high,low>` | reference constants | De-emphasis time constants in microseconds. |
| `--no-agc` | off | Disable automatic gain control. |
| `--no-dod` | off | Disable dropout detection. |
| `--ntsc-color-notch` | off | Notch filter on the decoded video, reduces colour wobble. |
| `--lowband` | off | Lower-bandwidth (lowband) decode settings. |
| `--no-efm` | off | Skip the EFM (digital audio) output. |
| `--disable-analog-audio` | off | Skip analog audio decoding. |

The output prefix doubles as the log path, so each run writes `<outfile>.log`
(unless `OUTFILE` is `-`, see above).

## Input formats

The format is inferred from the file extension; there is no stdin support and no
explicit format flag.

| Extension | Content |
| --- | --- |
| `.s16` | Raw signed 16-bit little-endian RF samples. |
| `.r16`, `.u16` | Raw unsigned 16-bit samples. |
| `.rf` | Raw 32-bit float samples. |
| `.lds` | Packed 10-bit DdD format (4 samples in 5 bytes). |
| `.r30` | Packed 10-bit legacy format (3 samples in 4 bytes). |
| `.ldf` | Ogg-FLAC capture, decoded in-process (claxon). Seeking restarts the decoder, as in Python. |
| `.flac`, `.ddd.flac` | Raw (non-Ogg) FLAC capture, decoded by an `ffmpeg` subprocess; set `LD_NO_FFMPEG=1` to fall back to the in-process decoder. |

## Output files

| File | Contents |
| --- | --- |
| `<out>.tbc` | Decoded luma field data (`uint16`, TBC'ed). |
| `<out>.tbc.json` | Field metadata: VITS, dropout lists, `fileLoc`s, audio/EFM parameters. Written with CRLF and the reference's exact key order. |
| `<out>.pcm` | Analog audio, 4 channels of signed 16-bit. |
| `<out>.efm` | EFM (digital audio) samples; `efmTValues` land in the JSON. |
| `<out>.tbc.db` | SQLite copy of the field metadata for tooling. Row-identical to Python's, **not** byte-identical: WAL mode and a different SQLite build change the page layout on purpose. |
| `<out>.log` | Everything the run printed, including the per-field timings when asked. |

## Verifying parity

The end-to-end gate is a hash comparison of all four decodable artifacts. On a
machine that has the reference outputs and the Python 7.3.0 tree:

```bash
ld-decode.exe -j 12 "capture.s16" rust_full
b3sum rust_full.tbc rust_full.pcm rust_full.efm rust_full.tbc.json
# compare against the hashes recorded for the Python run of the same input
```

For a change that cannot touch long-run state there is a much cheaper gate than
a full decode: a from-0 partial run is a **byte-exact prefix** of the Python
full-run artifacts. Decode `-l 500` (1000 fields, ~480 MB) and compare the
prefix without hashing 80 GB:

```bash
ld-decode.exe -j 12 -l 500 "capture.s16" head500
for x in tbc pcm efm; do
  cmp -n "$(stat -c%s head500.$x)" head500.$x "_stock_s16_full_7.3.0/capture.$x"
done
```

A from-0 `-l` run matches; a `-s` (windowed) run does not, for the reasons below.

Notes that matter when comparing runs:

- **Windowed and from-0 decodes of the same field differ** — in both
  implementations. AGC calibration state carries differently into the window.
  Compare windowed against windowed, from-0 against from-0.
- `.ddd.flac` comparisons always start at `-s 300`: the capture's lead-in has no
  signal, and that is the established A/B start for both ports.
- Deep `-s` seeks into a `.ddd.flac` are slow by design: the container has no
  seektable, so the reader decodes from the start and discards, exactly like the
  Python one.
- `.tbc.db` is compared row-by-row, never with `b3sum`.

## Environment variables

All `LD_*` switches are diagnostic or debugging hooks; with the environment
clean, the decoder is in its reference configuration.

| Variable | Effect |
| --- | --- |
| `LD_NO_FFMPEG=1` | Decode `.flac` with claxon instead of `ffmpeg`. |
| `LD_PF_POOL=n`, `LD_SIDE_POOL=n` | Override the demod/side worker pool split. |
| `LD_START_SAMPLE=n` | Start at an absolute sample instead of a frame. |
| `LD_NO_ASYNC_PREFETCH=1`, `LD_NO_DOD_PRE=1` | Disable the prefetch and dropout-predictor experiments. |
| `LD_TIMING=1` | Per-field phase timings (`proc`, `down`, `asm`, `dfrest`, `vits`, …). |
| `LD_PROCTIME=1`, `LD_DEMODTIME=1`, `LD_SUBTIME=1` | Coarser stage and sub-stage timings. |
| `LD_TRACE_MTF=1`, `LD_TRACE_AGC=1`, `LD_TRACE_KEEP=1`, `LD_TRACE_SEEK=1` | Text traces of the calibration and reader decisions. |
| `LD_DUMP_*` | Binary dumps of intermediate stages (dozens of them, mostly with a `_RL` readloc filter). Targeted debugging only — dumping every field makes runs crawl and fills disks. |

The full list, with the readloc filters and their formats, is in the source
(`crates/ld-decode/src/decode/` and `crates/ld-decode-cli/src/`).

## Tests and CI

```bash
cargo test -p ld-decode --release
```

The suite is mostly hermetic fidelity tests: the vendored FFT against stored
scipy 1.18.0 spectra, numpy's pairwise summation and `std` against 13.5k
generated cases, `butter`/`firwin`/`filtfft`/emphasis filters against scipy
coefficients, and the sinc LUT against its reference table. Expect
`19 passed, 3 ignored` plus one test that is filtered out or skipped:

- `pll_matches_stock_field_stream` needs `LD_PLL_DIR` pointing at a golden
  field stream dumped from a stock Python run; it fails loudly without it, so
  the CI commands pass `--skip pll_matches_stock_field_stream`.
- The three `#[ignore]`d tests (`probe_block`, `atan2_cmp`,
  `probe_sizes_vs_scipy`) read dumps that are not in the repo. Run them
  explicitly with `cargo test -- --ignored` where those dumps exist.

CI (`.github/workflows/`) is Windows x86-64 only and does what CI can do without
the captures: release build, the test suite in both the debug and the release
profile, and a smoke decode of a zero-signal file that must exit cleanly and
write all six output artifacts. The `b3sum` 4/4 gate needs the capture files and
hours of runtime, so it stays a local, manual check. The Rust toolchain is pinned
in CI to the nightly the parity tolerances are calibrated against; do not set
`RUSTFLAGS` in a workflow, because that would replace the `target-cpu=x86-64-v3`
setting in `.cargo/config.toml`.

## Why Windows only

The build is not portable as it stands, and that is intentional rather than
unfinished work:

- **The parity target is a specific Windows wheel.** `scipy.fft` dispatches to
  ducc0, and the shipped scipy 1.18.0 wheel for Windows x86-64 is built with the
  "homegrown SIMD" path using SSE2 only and single-threaded. The vendored ducc0
  in `vendor/` is compiled the same way (`crates/ld-decode/build.rs`) so its
  rounding matches bit-for-bit. An AVX2 build of ducc0 is measurably faster for
  the whole decode, and breaks a test with a 2-ulp difference — so SSE2 stays.
- **The compiler is part of the recipe.** `build.rs` drives `clang-cl` to emit
  MSVC-ABI COFF objects: matching rounding needs the same library build, and
  MSVC ABI/STL keeps the objects linkable by rustc's MSVC linker.
- **Some numerics are OS library calls.** Complex `pow` on Windows goes straight
  to `ucrtbase!cpow` through a `raw-dylib` declaration, because the UCRT
  implementation differs from any naive `exp`/`log` chain in the last bits.
- **The release configuration assumes Windows.** `.cargo/config.toml` sets
  `target-cpu=x86-64-v3` for the Rust code, the release profile keeps debug
  symbols deliberately, and the writer relies on Windows file semantics in one
  place (a `.tbc.json` whose header is rewritten at close needs read+write).

Porting to Linux or macOS is possible but is a project in itself: it would need a
ducc0 toolchain that reproduces the wheel used as ground truth on that platform,
plus a decision about which reference outputs count as canonical there.

## Why the code looks strange

Every numeric path replicates what a specific bundled library version computes —
not what is mathematically equivalent. Some consequences, all deliberate:

- FFTs go through the vendored ducc0, never `rustfft`, in the hot paths.
- Complex multiply uses numpy's FMA kernel and Smith's division-by-reciprocal;
  complex `pow` uses the UCRT via FFI.
- The Hilbert-unwrap path uses numba's plain four-product multiply, *not* numpy's
  FMA version — both are correct and they round differently.
- Summation of `bw_ratios` uses a bit-port of numpy's pairwise summation, because
  a 1-ulp mean would move a threshold and then a pixel.
- Sync thresholds and zero-crossings are computed in `f32` where NEP 50 dtype
  promotion makes the Python side `float32`.
- The sinc scaler uses `f32` per-tap products with an `f64` accumulator; widening
  the products to `f64` visibly changes luma.
- `uint16` wrap-around in IRE conversion is load-bearing: it rejects a field that
  a straight `f64` implementation would accept, and the reference does the same.
- The decoder's state machine follows Python's, including a MTF level that lags by
  one field and the exact prefetch/window block bookkeeping.

Do not "clean these up" without measuring: each one is there because the
alternative changed output bytes.

## Things that look like bugs but are not

- Windowed (`-s`) and from-0 decodes of the same field differ; Python does the
  same.
- `.flac` JSON `fileLoc`s are internally inconsistent with the delivered content,
  because PyAV seek under-delivers; the port replicates the content positions to
  stay byte-identical.
- `scripts/gen_test_signal.py` generates a synthetic signal that does not decode
  (its vblank structure does not match real sync timing). Python fails on it
  identically; it is a dev tool, not a parity bug.
- On one disc band the reference's own `uint16` arithmetic rejects a field the
  maths says should pass, and the port reproduces that.

## Repository layout

```
crates/ld-decode/       decoder library
  spec.rs               decode spec and every FFT filter
  ffi_ducc.rs           FFI to the vendored ducc0
  optimized/            sinc scaler, sosfiltfilt, fitpack deBoor, fast math
  decode/               demod block, field pipeline, VITS, dropouts, EFM PLL, audio
crates/ld-decode-cli/   the `ld-decode` binary
  reader.rs             all input formats and seek behaviour
  writer.rs, db.rs      .tbc/.tbc.json and .tbc.db writers
  prefetch.rs           window/refill loop
vendor/ducc0/           vendored FFT library, built by clang-cl
scripts/                Python reference generators and dev tools
```

## License

GPL-3.0-or-later, matching the ld-decode original it is ported from. The license
is declared in `Cargo.toml`; a `LICENSE` file still has to be added.
