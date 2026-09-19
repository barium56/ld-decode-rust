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
- **Byte-parity verified end-to-end** on four input paths:
  - a full `s16` capture (168,800 fields) — 4/4 outputs identical,
  - a full `.ddd.flac` capture decoded from frame 300 — 4/4 outputs identical,
  - a full `.ldf` capture (50,722 fields, Pioneer GGV1069) — 4/4 outputs
    identical,
  - a full `.ldf` capture from a different disc (71,710 fields, Diamond Time
    CLV) — 4/4 outputs identical.
- **Roughly 9–20x faster than the Python reference.** About 43 FPS at `-j 9`
  on a Ryzen 7 5800X3D (10,000 fields in ~4 minutes) against ~2–5 FPS for
  Python 3.12 + numpy 2.4.6 + scipy 1.18.0 on the same box. FPS is only
  meaningful with the machine otherwise idle and the output disk not full: a
  nearly-full `F:` measurably collapses throughput.
- Produces `.tbc`, `.tbc.json`, `.pcm`, `.efm` and `.tbc.db`, plus a `.log` that
  mirrors the console output.

## Quick start

Supported on **Windows x86-64 and Linux x86-64** (see
[Platform support](#platform-support); one binary per platform, no runtime
dependencies beyond `ffmpeg` for the FLAC-family inputs).

Prerequisites:

- Rust **nightly** (`rustup toolchain install nightly-2026-08-29`) on both
  platforms.
- A C++17 compiler, used to build the vendored ducc0 FFT (`build.rs`):
  - **Windows**: the MSVC toolchain (Rust's default host) plus LLVM's
    `clang-cl`, found on `PATH` or at
    `C:\Program Files\LLVM\bin\clang-cl.exe`.
  - **Linux**: `clang++` (preferred) or `g++`, and the C++ standard library
    headers — `sudo apt install build-essential clang` on Debian/Ubuntu.
    Override the choice with `CXX=...` if needed.
- `ffmpeg` on `PATH` for `.flac` / `.ddd.flac` inputs (and for `.ldf` container
  sniffing). Raw `.s16`/`.r16`/`.u16`/`.rf`/`.r30`/`.lds` capture decoding needs
  nothing else.

```bash
cargo build --release    # target/release/ld-decode.exe on Windows, ld-decode elsewhere
```

Decode a capture:

```bash
# whole file
target/release/ld-decode -j 9 "capture.ddd.flac" out

# start at frame 1000, decode 1000 frames (frames, not samples; 2 fields each)
target/release/ld-decode -j 9 -s 1000 -l 1000 "capture.s16" out

# run with no arguments for the full usage text (same output as --help)
target/release/ld-decode
```

`-j 9` is the measured sweet spot on an 8-core/16-thread machine, and the
optimum is flat only over a narrow band: `-j` 8/9/10/11 measure 38.1/39.1/37.5/
36.9 FPS. The demodulation pool is throughput-bound, so raising it past the
physical core count loses (SMT sharing costs more than the extra worker gains):
`-j` 12/14/16 fall to 34.6/32.0/28.8 FPS. The default scales with the core count
(`logical/2 + 1`, minimum 2).

## Command line

```
ld-decode.exe [OPTIONS] <INFILE> <OUTFILE>
```

Running the program with no arguments prints this same usage text, exactly as
`--help` does.

`INFILE` is what to decode, `OUTFILE` is the base name for the outputs
(`out` produces `out.tbc`, `out.tbc.json`, `out.pcm`, `out.efm`, `out.tbc.db`
and `out.log`).

`OUTFILE` may also be `-`, which streams the `.tbc` picture to stdout as raw
little-endian `uint16` and disables every sidecar output (the JSON header
rewrite needs a seekable file, and the log would otherwise interleave with the
picture bytes — in this mode the console log goes to stderr instead):

```bash
# pipe the picture straight into something else
ld-decode.exe -j 9 "capture.s16" - > out.tbc
ld-decode.exe -j 9 "capture.s16" - | ffmpeg -f rawvideo ...
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
| `-j`, `--threads <n>` | `logical/2 + 1`, min 2 | Demodulation worker threads. The serial tail's parallel sections get a quarter of this, so the split matters more than the total. |
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
| `.ldf` | FLAC capture. The container is sniffed, like PyAV does for the Python reference: Ogg-wrapped FLAC (what the Domesday Duplicator writes) is decoded in-process by claxon with seeking restarting the decoder, while a bare FLAC stream is handled exactly like `.flac`. Trailing tags or a stray capture header before the magic are tolerated. |
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
ld-decode.exe -j 9 "capture.s16" rust_full
b3sum rust_full.tbc rust_full.pcm rust_full.efm rust_full.tbc.json
# compare against the hashes recorded for the Python run of the same input
```

The end-to-end gate is a **full-disc** decode: a change that can affect long-run
state (AGC/MTF calibration, whiteloc or redo decisions, EFM PLL locking, reader
seeking) cannot be proven by a window, and a full run costs hours plus ~80 GB of
output per side. Everything else should be proven with a windowed A/B first —
1000 frames, or a targeted window at the affected fields.

The standard A/B is **interleaved**: run the old and new binaries alternately in
the same directory and compare FPS pair by pair, because a single 2-round delta
is not trustworthy (a concurrent `cargo build` on the same box once produced a
+4% "win" that a clean 4-round re-measure showed to be noise).

```bash
ld-decode.exe -j 9 -s 1000 -l 1000 "capture.s16" new
ld-decode.exe -j 9 -s 1000 -l 1000 "capture.s16" old
# compare new.tbc/pcm/efm/tbc.json against old.* with b3sum
```

A from-0 `-l` run is also a **byte-exact prefix** of the Python full-run
artifacts, so a cheap prefix comparison is available when the change cannot
touch long-run state:

```bash
ld-decode.exe -j 9 -l 500 "capture.s16" head500
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
| `LD_FFT_ENGINE=sse2\|avx2fma\|avx2` | Pick one of the three ducc0 builds linked into the binary. **`sse2` is the default and the one to use**: all three are bit-identical, but both AVX2 engines measure ~21% *slower* in situ at `-j 9` (34.0/33.7 FPS vs 43.1/43.0) because of an AVX2/FMA frequency-offset penalty under load. The engine in use is logged at startup. |
| `LD_NO_FFMPEG=1` | Decode `.flac` with claxon instead of `ffmpeg`. Bit-identical, but slower (33.7 vs 37.1 FPS), so `ffmpeg` stays the default. |
| `LD_PF_POOL=n`, `LD_SIDE_POOL=n` | Override the demod/side worker pool split. Both optima are measured and closed: the demod pool is throughput-bound at one thread per physical core, and widening the tail/side pools is a loss. |
| `LD_START_SAMPLE=n` | Start at an absolute sample instead of a frame. |
| `LD_NO_ASYNC_PREFETCH=1`, `LD_NO_DOD_PRE=1` | Disable the async demod prefetch and the dropout pre-pass. |
| `LD_PLLSPEC=1` | Print the EFM-PLL speculation commit/fallback counters (a result is only installed when its token and state generation still match, so a fallback is always correct). |
| `LD_TIMING=1` | Per-field phase timings (`proc`, `down`, `asm`, `dfrest`, `vits`, …) plus the prefetch-health fields `pfspan`/`pfwork`/`mtfpow`/`mtfmiss`. |
| `LD_PROCTIME=1`, `LD_DEMODTIME=1`, `LD_SUBTIME=1` | Coarser stage and sub-stage timings. |
| `LD_PF_TIMELINE=<field>` | Dump one prefetch batch's per-unit start/end/thread timeline; this is what localized the serial `cpow` stall on the batch's critical path. |
| `LD_TRACE_MTF=1`, `LD_TRACE_AGC=1`, `LD_TRACE_KEEP=1`, `LD_TRACE_SEEK=1` | Text traces of the calibration and reader decisions. |
| `LD_DUMP_*` | Binary dumps of intermediate stages (dozens of them, mostly with a `_RL` readloc filter). Targeted debugging only — dumping every field makes runs crawl and fills disks. |

Anything that reads an environment variable inside a per-line or per-sample loop must go through `envflag::CachedVar`/`CachedFlag`: uncached `std::env::var_os` calls in `compute_line_bursts` and the zero-crossing path cost 1.15 ms of wall time per field before they were cached, because every lookup takes a process-global lock (and a `GetEnvironmentVariableW` syscall on Windows).

The full list, with the readloc filters and their formats, is in the source
(`crates/ld-decode/src/decode/` and `crates/ld-decode-cli/src/`).

## Tests and CI

```bash
cargo test -p ld-decode --release
```

The suite is mostly hermetic fidelity tests: the vendored FFT against stored
scipy 1.18.0 spectra (1024 and 32768, the size the pipeline actually
transforms), numpy's pairwise summation and `std` against generated cases,
`butter`/`firwin`/`filtfft`/emphasis filters against scipy coefficients, the
sinc LUT against its reference table, and the batched inverse FFT against the
per-transform scalar form it replaced. The stored spectra are committed per
platform (see [Platform support](#platform-support)), so the suite runs
unchanged on Windows and Linux with no extra tooling. Expect **36 passed, 3
ignored**, plus one test that is filtered out or skipped:

- `pll_matches_stock_field_stream` needs `LD_PLL_DIR` pointing at a golden
  field stream dumped from a stock Python run; it fails loudly without it, so
  the CI commands pass `--skip pll_matches_stock_field_stream`. A local run
  without that variable will therefore report one failure — that is the known
  environmental one, not a regression.
- The three `#[ignore]`d tests (`probe_block`, `atan2_cmp`,
  `probe_sizes_vs_scipy`) read dumps that are not in the repo. Run them
  explicitly with `cargo test -- --ignored` where those dumps exist.

Two known release-only fidelity failures exist in the notes for
`fefm_stage_isolation` and `fefm_bit_exact_matches_scipy_118` (nightly codegen
drift in `gen_bpf_supergauss`, ~1.4e-14); the tolerances sit at 1e-12 and the
debug profile passes exactly.

CI (`.github/workflows/ci.yml`) runs the same job on **Windows x86-64 and Linux
x86-64** and does what CI can do without the captures: release build, the test
suite in both the debug and the release profile, and the shared smoke test
(`.github/scripts/smoke.sh`, used by every workflow) — a zero-signal decode that
must exit cleanly and write all six output artifacts, plus the `.ldf`
container-sniffing, pipe-to-stdout and argument-parsing paths. `ffmpeg` is
installed in CI because the `.ldf` smoke test shells out to it to build a
bare-FLAC fixture. The `b3sum` 4/4 gate needs the capture files and hours of
runtime, so it stays a local, manual check. The Rust toolchain is pinned in CI to
the nightly the parity tolerances are calibrated against; do not set `RUSTFLAGS`
in a workflow, because that would replace the `target-cpu=x86-64-v3` setting in
`.cargo/config.toml`.

`release.yml` is the only distributable-artifact recipe (there is no separate
build-only workflow to drift out of sync): its `build-decode` job runs on every
dispatch and produces a versioned Windows `.zip` plus a Linux `.tar.gz` as run
artifacts, while the `release` job that publishes them is gated on a `v*` tag or
`create_release=true`. Linux artifacts are built on `ubuntu-22.04` rather than
`ubuntu-latest`, so the released binary links against glibc 2.35 and runs on
older distributions.

## Platform support

**Windows x86-64 and Linux x86-64 are both supported**, and both are held to the
same bar: byte-identical output files against the Windows Python 7.3.0 reference
— the same hashes on both platforms, not "close on each". What makes that
portable is that the parity-critical pieces are either platform-independent or
served by bit-exact ports of the one platform's math library:

- **The FFT is the vendored ducc0 source, at a fixed SIMD width.** `scipy.fft`
  dispatches to ducc0; the shipped scipy 1.18.0 wheel for Windows x86-64 uses
  the "homegrown SIMD" path at the 128-bit (SSE2) width, single-threaded.
  `crates/ld-decode/build.rs` compiles the vendored copy the same way on both
  platforms — clang-cl/MSVC on Windows, clang++ or g++ on Linux — with no
  AVX/FMA, so the same kernels are selected and the same rounding comes out.
  ducc0 picks its SIMD width from compiler predefined macros (`__SSE2__`,
  `__AVX2__`), not from the operating system, and the FFT's operation order is
  defined by ducc0's source, not by the compiler. The `engine_simd_widths` test
  asserts the compiled lane widths, so a silently dropped or added ISA flag
  fails the suite rather than quietly changing results.
- **AVX2 builds are linked in but not used.** `avx2`/`avx2fma` engines are
  runtime-selectable via `LD_FFT_ENGINE`; they are bit-identical on every golden
  and end-to-end, but measure **~21% slower** in situ at `-j 9` (34.0/33.7 FPS
  against 43.1/43.0) — the isolated microbenchmark that once showed all engines
  equal was measuring one transform on an idle core. So `sse2` stays the default
  for parity *and* speed, and any codegen change needs an in-situ A/B before it
  can be called neutral. Off x86 they are compiled from the same portable TU and
  the runtime feature gate never selects them.
- **The reference's libm calls are the parity target, and Linux does not call
  glibc for them.** `sin`/`cos` come from bit-exact ports of UCRT
  (`optimized/ucrt_math.rs`), because UCRT and glibc disagree by 1-2 ulp on a few
  percent of arguments and those values are baked into every FFT twiddle. The
  port reproduces the FMA variant, which is the one UCRT dispatches to on the CPU
  that produced the reference corpus, and is validated against the real UCRT over
  1.1 million arguments with 0 mismatches. **`atan2` is ported the same way**
  (`optimized/ucrt_atan2.rs`, 0 mismatches over 2.05 million arguments): it is the
  hottest parity call in the decoder at ~700 000 calls per field, and UCRT and
  glibc disagree on ~0.2% of arguments, so Linux must not call glibc there. The
  port costs nothing — 16.28 ns/call against glibc's 16.89 ns — and covers every
  operand class the decoder can produce, falling back to the platform library for
  NaN/inf and subnormal operands, which it cannot. C99 `cpow` (numpy's complex
  power, used for `MTF ** level` and for the de-emphasis exponent in the video
  filter) is the one remaining platform-bound call, and it is **measured to be
  the residual**: `LD_DUMP_MTFPOW` dumps the filter and the `cpow` result, and a
  cross-platform diff shows the MTF filter bit-identical while `cpow` differs on
  39% of its 32768 elements, by up to 3 ulp, for every level used. That leaves
  **one differing byte per 20 000 fields** (a 1-LSB luma value) in a full-disc
  Linux run, with `.efm` and `.pcm` byte-identical. Removing it needs UCRT's
  `clogl`/`cexp` (and the `log`/`exp`/`hypot` beneath them) ported the same way;
  UCRT computes `cexp(clog(z) * w)` where glibc uses a `pow` on the modulus,
  which is why they disagree by more than rounding.
  `numpy_sincos` (numpy's complex `exp`) is another explicit choice rather than an
  optimizer accident: on glibc `sincos` is *not* the same number as `cos`/`sin`,
  and `np.exp(-1j*w)` follows it across all 32768 bins of the `freqz` grid.
- **The Rust code is `target-cpu=x86-64-v3` on x86-64 only.**
  `.cargo/config.toml` scopes that to `cfg(target_arch = "x86_64")`, since the
  CPU name is invalid elsewhere. A `x86-64-v2` build of the same source is
  slower, so v3 stays.

One field is expected to differ between platforms: the `osInfo` string in
`.tbc.json`, which mirrors Python's
`platform.system():platform.release():platform.version()`. On Windows it comes
from `ver`, on unix from `uname -r`/`uname -v` — exactly as the Python reference
does on those platforms. Everything else is common: `.tbc`, `.pcm` and `.efm`
carry the same bytes, and therefore the same hashes, on both.

Measured on all four input paths (same window per path, sha256 of the raw
outputs, Linux build vs the Windows Python reference):

| path | window | `.tbc` (957 MB) | `.pcm` (2 942 940 samples) | `.efm` |
|---|---|---|---|---|
| s16 | `-s 1000 -l 1000` | identical | identical | identical |
| `.ldf` | `-s 300 -l 1000` | identical | identical | identical |
| `.flac` (ffmpeg) | `-s 300 -l 1000` | identical | identical | identical |
| `.flac` (claxon) | `-s 300 -l 1000` | identical | identical | identical |

The `.flac` rows used **different ffmpeg builds** (Ubuntu 4.4.2 vs the gyan
`N-112134` build on Windows), and the outputs are still byte-identical, so neither
the FLAC container nor the ffmpeg version is a parity risk. Before the UCRT ports
this table read 9 / 6 / 14 differing `.tbc` bytes (isolated luma values off by 1-3
LSB) and a ±1 LSB `.pcm` dither on about 7% of samples — `.efm` was already
identical everywhere, being hard-decision and drift-free.

Those rows are windowed runs. Over a long run the one unported call shows up: a
40 000-field Linux decode of the s16 capture differs from the Windows reference
in **one byte of 19 GB** of `.tbc` (field 11 656, line 126, a 1-LSB luma value),
with `.efm` and `.pcm` still identical. `LD_DUMP_MTFPOW` identifies it as `cpow`
(see the bullet above); everything else on the video path, including the MTF
filter it is applied to, is bit-identical across the two platforms.

The hermetic FFT/filter goldens live in `crates/ld-decode/tests/data` as **one
committed set for both platforms**, holding the values scipy 1.18.0 produces on
Windows — the parity target. They are generated from committed,
platform-independent inputs by `scripts/gen_scipy_fft_goldens.py`, so `cargo test`
needs no Python, no setup and no regeneration on either platform, including in
`release.yml`, whose two runners run the same command. `--verify` is the gate: it
recomputes the set and fails if the recipes or the pinned wheels have drifted from
the committed values (run it on Windows; elsewhere the local scipy's own drift
makes a mismatch expected, which is what `--census` measures instead).

Building on other architectures (aarch64, macOS) is untested: it compiles and
runs best-effort, but the parity claim does not extend there — the reference
wheel for those platforms uses NEON/AVX2 kernels, which is a different rounding.
Note also that `-mavx2 -mfma` is x86-only, so on such targets the three engine
names are built from one portable TU.

## Why the code looks strange

Every numeric path replicates what a specific bundled library version computes —
not what is mathematically equivalent. Some consequences, all deliberate:

- FFTs go through the vendored ducc0, never `rustfft`, in the hot paths.
- Complex multiply uses numpy's FMA kernel and Smith's division-by-reciprocal;
  complex `pow` is a direct FFI call to the platform C library (UCRT on Windows),
  never an `exp(b*log(a))` chain.
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
  one field and the exact prefetch/window block bookkeeping. A cached block is
  reused unconditionally once demodulated, redo targets are Python-truthy-tested
  (a `0.0` target cancels the redo), and the field read window is floored on the
  block-length grid while demodulation strides a different, coprime block size.
  All three of those moved linelocs or pixels when implemented "sensibly".
- The EFM PLL's T-values are produced in writeout order but *computed* on a
  helper thread during the same field's downscale. A per-spawn token plus a
  PLL-generation counter make a stale or superseded result unusable, and
  anything that fails either check falls back to the inline call — which is why
  long-run output cannot differ. Backfill writes of older fields advance the
  generation, so they can never pick up a speculation.
- The MTF power spectrum is computed with a `par_iter` even though the memo
  lookup is serial: a miss costs ~3 ms of `cpow` calls, and it sits on the
  prefetch batch's critical path with the whole demod pool idle. Parallelizing
  it took `iter` from 12.05 to 10.64 ms per field.

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
  ffi_ducc.rs           FFI to the vendored ducc0 (three engine builds)
  envflag.rs            cached env lookups for hot paths
  optimized/            sinc scaler, sosfiltfilt, fitpack deBoor, fast math
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

`build.rs` compiles ducc0's header-only templates three times (the 128-bit
baseline engine, AVX2, and AVX2 with FP contraction off) into one binary, using
clang-cl on Windows and clang++/g++ on Linux/macOS. Only the baseline engine is
used by default — see [Platform support](#platform-support). Set
`LD_SKIP_VENDOR_FFT=1` to skip that native build for a `cargo check` on a host
without a C++ toolchain (check-only: it will not link).

## Performance work

Speed is the point of the port, so the measurement discipline is part of the
design. The pipeline is **CPU-saturated, not latency-bound**: at `-j 9` roughly
97% of the eight physical cores are busy, and about 84% of that CPU is the
demodulation kernel, of which ~76% is ducc FFT. That has two consequences that
repeatedly decide experiments:

- **Deleting CPU anywhere pays**, because it frees cores and memory bandwidth for
  the demodulation pool. Moving work between pools does not.
- **The two sides sit within ~1 ms of each other** (driver chain ~9.3 ms, pool
  demand `dcpu/9` ~9.8 ms per field), and the prefetch fold wait correlates +0.88
  with that field's batch span. So a driver-side saving becomes pool wait, and
  only pool-side span or CPU cuts pay one for one.

Before proposing an optimization, check whether it is already recorded as a dead
end. Several plausible ones are, with the measurement that killed them: finer
demod task granularity (neutral, then 1.7% slower), cross-block batching of the
real transforms (1.9–2.0x faster isolated, 37.6 to 26.9 FPS in situ), `/O3` on
the ducc shim (neutral), a wider tail pool (flat), and the AVX2 engines (21%
slower). The common failure mode is an **isolated** microbenchmark that reverses
in situ, so a gain is only real once it is measured interleaved on the real
pipeline. Details and the full list are in `AGENTS.md` and `work/bench_log.md`.

## License

GPL-3.0-or-later, matching the ld-decode original it is ported from. The license
is declared in `Cargo.toml`; a `LICENSE` file still has to be added.
