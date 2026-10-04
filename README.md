# ld-decode-rust

A from-scratch Rust port of the NTSC LaserDisc RF decoder from
[ld-decode](https://github.com/happycube/ld-decode) 7.4.0, whose
`.tbc`/`.pcm`/`.efm` output is byte-identical to 7.3.0's.

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
- **Byte-parity verified end-to-end** on five full-disc paths:
  - `s16` capture (168,800 fields) — 4/4 outputs identical,
  - `.ddd.flac` from frame 300 (168,566 fields) — 4/4 outputs identical,
  - Pioneer GGV1069 `.ldf` (50,722 fields) — 4/4 outputs identical to the
    Python **7.4.0** release,
  - Diamond Time CLV `.ldf` (71,710 fields) — 4/4 outputs identical,
  - GGV1016 CAV `.ldf` (108,576 fields) — 4/4 outputs identical, JSON included.
- **Roughly 9–20x faster than the Python reference.** About 46 FPS for raw
  `s16` (`-j 8`) and ~45 FPS for `.ldf` (`-j 10`/`11`) on a Ryzen 7 5800X3D
  (10,000-field window), against ~2–5 FPS for Python 3.12 + numpy + scipy on the
  same box; the full GGV1069 `.ldf` takes 575 s (44.1 FPS). FPS is only
  meaningful with the machine otherwise idle and the output disk not full.
- Produces `.tbc`, `.tbc.json`, `.pcm`, `.efm` and `.tbc.db`, plus a `.log` that
  mirrors the console output.

## Quick start

Supported on **Windows x86-64 and Linux x86-64** (see
[Platform support](#platform-support)). One binary per platform; the only
runtime dependency is `ffmpeg` for the FLAC-family inputs.

Prerequisites:

- Rust **nightly** (`rustup toolchain install nightly-2026-08-29`).
- A C++17 compiler for the vendored ducc0 FFT (`build.rs`):
  - **Windows**: the MSVC toolchain plus LLVM's `clang-cl` on `PATH`.
  - **Linux**: `clang++` (preferred) or `g++`, and the C++ standard library
    headers — `sudo apt install build-essential clang`. Override with `CXX=...`.
- `ffmpeg` on `PATH` for `.flac` / `.ddd.flac` and Ogg `.ldf` inputs; the `.ldf`
  path falls back to the in-process claxon decoder with `LD_NO_FFMPEG=1`. Raw
  `.s16`/`.r16`/`.u16`/`.r8`/`.u8`/`.s8`/`.rf`/`.r30`/`.lds` decoding needs
  nothing else.

```bash
cargo build --release    # target/release/ld-decode.exe on Windows, ld-decode elsewhere

# whole file
target/release/ld-decode -j 10 "capture.ddd.flac" out

# start at frame 1000, decode 1000 frames (frames, not samples; 2 fields each)
target/release/ld-decode -j 10 -s 1000 -l 1000 "capture.s16" out
```

`-j` matters because the demodulation pool is throughput-bound: raising it past
the physical core count loses, and the optimum moves with the input. On an
8-core/16-thread machine, `s16` peaks around `-j 8` (~46 FPS) and `.ldf` at
`-j 10`/`11` (~45 FPS). The default is `logical * 5 / 8` (minimum 2).

## Command line

```
ld-decode [OPTIONS] <INFILE> <OUTFILE>
```

Running with no arguments prints the same usage text as `--help`.

`INFILE` is a file, or `-` for stdin (stdin needs `--format`; `-s`/`-S` can only
move forward there). `OUTFILE` is the base name for the outputs (`out` produces
`out.tbc`, `out.tbc.json`, `out.pcm`, `out.efm`, `out.tbc.db` and `out.log`).

`OUTFILE` may also be `-`, which streams the `.tbc` picture to stdout as raw
little-endian `uint16` and disables every sidecar output (the console log moves
to stderr). The bytes are identical to a normal run's; if the reader at the
other end closes the pipe early, the decode stops immediately with an error.

| Option | Default | Meaning |
| --- | --- | --- |
| `-s`, `--start <n>` | `0` | Rough jump to **frame** `n` of the capture (2 fields per frame). |
| `-l`, `--length <n>` | until EOF | Decode at most `n` **frames**. |
| `-S`, `--seek <n>` | off | Seek to a specific VBI frame number; needs readable CAV/CLV frame codes and fails gracefully without them. |
| `-j`, `--threads <n>` | `logical * 5 / 8`, min 2 | Demodulation worker threads; the serial tail's parallel sections get a quarter of this. |
| `--format <fmt>` | from the extension | Input sample format (required for stdin): `s16`, `r16`/`u16`, `r8`/`u8`, `s8`, `rf`, `lds`, `r30`, `ldf`, `flac`. |
| `--inputfreq <MHz>` | from FLAC metadata, else `40` | Input sample rate; a capture's declared kHz rate is its RF rate in MHz. |
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

## Input formats

The format is inferred from the file extension; `--format` overrides it.

| Extension | Content |
| --- | --- |
| `.s16` | Raw signed 16-bit little-endian samples. |
| `.r16`, `.u16` | Raw unsigned 16-bit samples. |
| `.r8`, `.u8` | Raw unsigned 8-bit samples, kept raw (0..255 — the reference's `uint8` view). |
| `.s8` | Signed 8-bit, scaled by 256 (`int8 * 256`). |
| `.rf` | Raw 32-bit float samples. |
| `.lds` | Packed 10-bit DdD format (4 samples in 5 bytes). |
| `.r30` | Packed 10-bit legacy format (3 samples in 4 bytes). |
| `.ldf` | FLAC capture. The container is sniffed like PyAV does: Ogg-wrapped FLAC is decoded by the same `ffmpeg` s16le subprocess as `.flac` (~2.7x claxon's throughput); `LD_NO_FFMPEG=1` falls back to in-process claxon. Seeks restart the child and discard the exact sample count (never `ffmpeg -ss`, matching Python's seek semantics). Bare FLAC is handled like `.flac`; trailing tags or a stray header before the magic are tolerated. |
| `.flac`, `.ddd.flac` | Raw (non-Ogg) FLAC capture, decoded by an `ffmpeg` subprocess; `LD_NO_FFMPEG=1` falls back to the in-process decoder. |

## Output files

| File | Contents |
| --- | --- |
| `<out>.tbc` | Decoded luma field data (`uint16`, TBC'ed). |
| `<out>.tbc.json` | Field metadata: VITS, dropout lists, `fileLoc`s, audio/EFM parameters. Written with CRLF and the reference's exact key order. |
| `<out>.pcm` | Analog audio, 4 channels of signed 16-bit. |
| `<out>.efm` | EFM (digital audio) samples; `efmTValues` land in the JSON. |
| `<out>.tbc.db` | SQLite copy of the field metadata. Row-identical to Python's, **not** byte-identical: WAL mode and a different SQLite build change the page layout on purpose. |
| `<out>.log` | Everything the run printed, including per-field timings when asked. |

## Verifying parity

The end-to-end gate is a **full-disc** decode: a hash comparison of all four
decodable artifacts (`.tbc`/`.pcm`/`.efm`/`.tbc.json`) against the Python
reference outputs for the same input. Only a full run crosses the long-run state
(AGC/MTF calibration, whiteloc/redo decisions, EFM PLL locking, reader seeking) —
a window cannot prove those. Everything else is proven with a windowed A/B —
1,000 frames, or a targeted window at the affected fields — run **interleaved**
(old and new binaries alternately) and compared pair by pair. `.tbc.db` is
compared row-by-row, never hashed.

- A from-0 `-l` run is a byte-exact prefix of the Python full-run artifacts; a
  `-s` (windowed) run is not. Compare windowed against windowed only — AGC
  calibration state carries differently into a window, in both implementations.
- `.ddd.flac` comparisons always start at `-s 300`: the capture's lead-in has no
  signal, and that is the established A/B start.
- Deep `-s` seeks into `.ddd.flac`/`.ldf` are slow by design: no seektable, so
  the reader decodes from the start and discards, exactly like Python.

## Environment variables

All `LD_*` switches are diagnostic hooks; with the environment clean, the decoder
is in its reference configuration.

| Variable | Effect |
| --- | --- |
| `LD_FFT_ENGINE=sse2\|avx2fma\|avx2` | Pick one of the three ducc0 builds linked into the binary. **`sse2` is the default and the only parity-safe choice**: the AVX2 engines evaluate a different, differently-rounded sequence and are not bit-exact, so they are for experiments only. The engine in use is logged at startup. |
| `LD_NO_FFMPEG=1` | Decode `.flac` and Ogg `.ldf` with claxon instead of `ffmpeg`. Bit-identical but slower (the demod pool starves on claxon's throughput), so `ffmpeg` stays the default. |
| `LD_PF_POOL=n`, `LD_SIDE_POOL=n` | Override the demod/side worker pool split. Both optima are measured and closed. |
| `LD_START_SAMPLE=n` | Start at an absolute sample instead of a frame. |
| `LD_NO_ASYNC_PREFETCH=1`, `LD_NO_DOD_PRE=1` | Disable the async demod prefetch and the dropout pre-pass. |
| `LD_PLLSPEC=1` | Print the EFM-PLL speculation commit/fallback counters. |
| `LD_TIMING=1` | Per-field phase timings plus the prefetch-health fields `pfspan`/`pfwork`/`mtfpow`/`mtfmiss`. |
| `LD_PROCTIME=1`, `LD_DEMODTIME=1`, `LD_SUBTIME=1` | Coarser stage and sub-stage timings. |
| `LD_TRACE_MTF=1`, `LD_TRACE_AGC=1`, `LD_TRACE_KEEP=1`, `LD_TRACE_SEEK=1` | Text traces of the calibration and reader decisions. |
| `LD_DUMP_*` | Binary dumps of intermediate stages (mostly with a `_RL` readloc filter). Targeted debugging only — they make runs crawl and fill disks. |

Anything reading an environment variable inside a per-line or per-sample loop
must go through `envflag::CachedVar`/`CachedFlag`: uncached lookups take a
process-global lock and cost 1.15 ms of wall time per field before caching.

## Tests and CI

```bash
cargo test -p ld-decode --release
```

The suite is mostly hermetic fidelity tests: the vendored FFT against stored
scipy 1.18.0 spectra (1024 and 32768, the size the pipeline transforms), numpy's
pairwise summation and `std`, the `butter`/`firwin`/`filtfft`/emphasis filters,
the sinc LUT, and the batched inverse FFT against the scalar form it replaced.
The goldens are **one committed set for both platforms** (the Windows scipy
values are the target), so the suite needs no Python and runs unchanged on
Windows and Linux. Expect **55 passed, 5 ignored**:

- `pll_matches_stock_field_stream` needs `LD_PLL_DIR` pointing at a golden field
  stream; CI passes `--skip pll_matches_stock_field_stream`. A local run without
  that variable reports one failure — the known environmental one, not a
  regression.
- The five `#[ignore]`d tests (`probe_block`, `atan2_cmp`,
  `probe_sizes_vs_scipy`, `kernel_shape_speed_census`,
  `construction_fingerprint_probe`) read dumps that are not in the repo or take
  minutes; run them with `cargo test -- --ignored` where their inputs exist.

CI (`.github/workflows/ci.yml`) runs the same job on Windows and Linux: release
build, tests in both profiles, and the shared `smoke.sh` — a zero-signal decode
that must exit cleanly and write `.tbc`/`.pcm`/`.efm`/`.tbc.db`/`.log` (and **no**
`.tbc.json`), plus the `.ldf` sniffing, pipe-to-stdout and argument-parsing
paths. `release.yml` is the only artifact recipe: `build-decode` runs on every
dispatch (versioned Windows `.zip`, Linux `.tar.gz`), and publishing is gated on
a `v*` tag. Linux artifacts build on `ubuntu-22.04` for a glibc 2.35 floor.

## Platform support

Windows x86-64 and Linux x86-64 are both supported and both parity-locked to the
same hashes. The parity-critical pieces are platform-independent or bit-exact
ports of the reference's math library:

- **The FFT is the vendored ducc0 at a fixed SIMD width** (128-bit SSE2,
  single-threaded — the configuration scipy 1.18.0's wheel uses). `build.rs`
  compiles it with clang-cl on Windows and clang++/g++ on Linux; ducc0 selects
  its width from compiler macros, not the OS, so the same kernels and rounding
  come out. The `engine_simd_widths` test asserts the compiled lane widths.
- **AVX2 builds are linked in but never used.** They are runtime-selectable via
  `LD_FFT_ENGINE`, but their 4-lane kernels evaluate a differently-rounded
  sequence — not bit-exact — so `sse2` stays the default for parity, despite the
  AVX2 engines measuring somewhat faster. (An earlier "all engines are
  identical" reading was an artifact of shared linker symbols; each engine has
  had its own entry symbols since 2026-09-22.)
- **The reference's libm calls are the parity target, and every one on the
  decode path is a bit-exact UCRT port** (`optimized/ucrt_math.rs`,
  `ucrt_atan2.rs`, `ucrt_exp_log.rs`, `ucrt_pow.rs`): `sin`/`cos`, `atan2`,
  `exp`/`log`/`log1p`, `cpow` and real-base `pow`. UCRT and glibc disagree by
  1-2 ulp (and `cpow`'s algorithm is genuinely different), and those values are
  baked into every FFT twiddle and filter, so Linux must not call glibc there.
  The ports are validated against the real UCRT over millions of arguments with
  0 mismatches, and they are authoritative on **both** platforms (not just
  Linux) so a Windows build does not depend on the host's `ucrtbase.dll`
  revision. A few construction-path sites (`tan`, `powf`/`log10`/`atan` in
  `buttord`, `hypot`) are still platform calls; they are measured identical and
  pinned by `construction_is_platform_independent`, which hashes the whole
  constructed filter bank on both platforms.
- **The Rust code is `target-cpu=x86-64-v3` on x86-64 only** (scoped in
  `.cargo/config.toml`). A `x86-64-v2` build is slower, so v3 stays.

One field is expected to differ between platforms: the `osInfo` string in
`.tbc.json`, which mirrors Python's `platform.system():release():version()`.
Everything else is common, and has therefore been verified at length: windowed
runs on all four input paths are byte-identical across platforms, and the
complete 168,800-field Linux/WSL2 s16 decode reproduces the Python reference
byte-for-byte on `.tbc`/`.pcm`/`.efm`.

Building on other architectures (aarch64, macOS) is untested: it compiles
best-effort, but the parity claim does not extend there — the reference wheels
use NEON/AVX2 kernels, a different rounding.

## Why the code looks strange

Every numeric path replicates what a specific bundled library version computes —
not what is mathematically equivalent. Do not "clean these up" without
measuring; each one is there because the alternative changed output bytes.

- FFTs go through the vendored ducc0, never `rustfft`.
- Complex multiply uses numpy's FMA kernel and Smith's division-by-reciprocal;
  complex `pow` goes through the UCRT port.
- The Hilbert-unwrap path uses numba's plain four-product multiply, not numpy's
  FMA version — both are correct and they round differently.
- `bw_ratios` summation is numpy's pairwise summation bit-for-bit.
- Sync thresholds and zero-crossings are `f32` where NEP 50 dtype promotion makes
  the Python side `float32`.
- The sinc scaler uses `f32` per-tap products with an `f64` accumulator.
- `uint16` wrap-around in IRE conversion is load-bearing; it rejects a field a
  straight `f64` implementation would accept.
- The state machine follows Python's, including the one-field MTF lag, the
  prefetch/window bookkeeping, unconditional reuse of demodulated blocks, and
  Python-truthy-tested redo targets.
- The EFM PLL's T-values are computed speculatively on a helper thread; a
  per-spawn token and a PLL-generation counter make a stale result unusable, so
  long-run output cannot differ.
- The MTF power spectrum is computed with a `par_iter` even though the memo
  lookup is serial: a miss costs ~3 ms of `cpow` on the prefetch batch's
  critical path with the demod pool idle.

## Things that look like bugs but are not

- Windowed (`-s`) and from-0 decodes of the same field differ; Python does the
  same.
- `.flac` JSON `fileLoc`s are internally inconsistent with the delivered content
  because PyAV seek under-delivers; the port replicates the content positions.
- `scripts/gen_test_signal.py` generates a synthetic signal that does not
  decode; Python fails on it identically. Dev tool, not a parity bug.
- On one disc band the reference's own `uint16` arithmetic rejects a field the
  maths says should pass, and the port reproduces that.
- A decode that handles no fields writes no `.tbc.json` at all (Python 7.4.0's
  dumper drops its `None` snapshot; 7.3.0 left a truncated temp file).

## Repository layout

```
crates/ld-decode/       decoder library
  spec.rs               decode spec and every FFT filter
  ffi_ducc.rs           FFI to the vendored ducc0 (three engine builds)
  envflag.rs            cached env lookups for hot paths
  optimized/            sinc scaler, sosfiltfilt, fitpack deBoor, UCRT math ports
  decode/               demod block, field pipeline, VITS, dropouts, EFM PLL, audio
crates/ld-decode-cli/   the `ld-decode` binary
  main.rs               argument parsing and the window/refill loop
  reader.rs             all input formats and seek behaviour
  writer.rs, db.rs      .tbc/.tbc.json and .tbc.db writers
  async_writer.rs       background writer thread
  async_db.rs           background `.tbc.db` thread
  prefetch.rs           window/refill loop
vendor/ducc0/           vendored FFT library, built by build.rs
vendor/ducc_ffi.cc      the C shim (compiled once per engine)
scripts/                Python reference generators and dev tools
```

`build.rs` compiles ducc0 three times (128-bit baseline, AVX2, AVX2 with FP
contraction off). Only the baseline is used by default — see
[Platform support](#platform-support). `LD_SKIP_VENDOR_FFT=1` skips that build
for a `cargo check` on a host without a C++ toolchain (check-only: no link).

## Performance work

Speed is the point of the port, so the measurement discipline is part of the
design. The pipeline is **CPU-saturated, not latency-bound**: ~97% of the
physical cores are busy, ~84% of that CPU is the demodulation kernel (mostly
ducc FFT). Two rules follow: deleting CPU anywhere pays (it frees cores for the
demod pool), and moving work between pools does not — a driver-side saving just
becomes pool wait. The reader can also starve the pool on compressed inputs
without showing up in decode CPU, which is why `.ldf` now uses the ffmpeg
subprocess.

Before proposing an optimization, check whether it is already recorded as a dead
end: finer demod task granularity (neutral, then 1.7% slower), cross-block
batching of the real transforms (1.9–2.0x isolated, 37.6 to 26.9 FPS in situ),
`/O3` on the ducc shim (neutral), a wider tail pool (flat), and the AVX2 engines
(faster isolated, not parity-safe). The common failure mode is an **isolated**
microbenchmark that reverses in situ, so a gain is only real once it is measured
interleaved on the real pipeline. Details in `AGENTS.md` and `work/bench_log.md`.

## License

GPL-3.0-or-later, matching the ld-decode original it is ported from. The license
is declared in `Cargo.toml`; a `LICENSE` file still has to be added.
