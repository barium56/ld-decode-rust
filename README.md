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
target/release/ld-decode.exe -j 9 "capture.ddd.flac" out

# start at frame 1000, decode 1000 frames (frames, not samples; 2 fields each)
target/release/ld-decode.exe -j 9 -s 1000 -l 1000 "capture.s16" out

# run with no arguments for the full usage text (same output as --help)
target/release/ld-decode.exe
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

Anything that reads an environment variable inside a per-line or per-sample loop must go through `envflag::CachedVar`/`CachedFlag`: uncached `std::env::var_os` calls in `compute_line_bursts` and the zero-crossing path cost 1.15 ms of wall time per field before they were cached, because on Windows every lookup takes a process-global lock.

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
per-transform scalar form it replaced. Expect **36 passed, 3 ignored**, plus one
test that is filtered out or skipped:

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

CI (`.github/workflows/`) is Windows x86-64 only and does what CI can do without
the captures: release build, the test suite in both the debug and the release
profile, and a smoke decode of a zero-signal file that must exit cleanly and
write all six output artifacts. `ffmpeg` is installed in CI because the `.ldf`
container-sniffing smoke test shells out to it to build a bare-FLAC fixture. The
`b3sum` 4/4 gate needs the capture files and hours of runtime, so it stays a
local, manual check. The Rust toolchain is pinned in CI to the nightly the parity
tolerances are calibrated against; do not set `RUSTFLAGS` in a workflow, because
that would replace the `target-cpu=x86-64-v3` setting in `.cargo/config.toml`.

## Why Windows only

The build is not portable as it stands, and that is intentional rather than
unfinished work:

- **The parity target is a specific Windows wheel.** `scipy.fft` dispatches to
  ducc0, and the shipped scipy 1.18.0 wheel for Windows x86-64 is built with the
  "homegrown SIMD" path using SSE2 only and single-threaded. The vendored ducc0
  in `vendor/` is compiled the same way (`crates/ld-decode/build.rs`) so its
  rounding matches bit-for-bit. AVX2 builds of the same source are linked in as
  well and are bit-identical on every golden and end-to-end, but they measure
  **~21% slower** in situ at `-j 9` (34.0/33.7 FPS against 43.1/43.0) — the
  isolated microbenchmark that once showed all engines equal was measuring one
  transform on an idle core. So `sse2` stays the default for parity *and*
  speed, and any codegen change needs an in-situ A/B before it can be called
  neutral. (The Rust code's own `target-cpu=x86-64-v3` does not show the effect;
  a `x86-64-v2` build of the same source is slower, so v3 stays too.)
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
vendor/ducc0/           vendored FFT library, built by clang-cl
vendor/ducc_ffi.cc      the C shim (compiled once per engine)
scripts/                Python reference generators and dev tools
```

`build.rs` compiles ducc0's header-only templates three times (SSE2, AVX2, and
AVX2 with FP contraction off) into one binary. Only `sse2` is used by default —
see [Why Windows only](#why-windows-only).

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
