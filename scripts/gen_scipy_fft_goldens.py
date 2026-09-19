# Regenerates -- or measures -- the scipy reference outputs that the hermetic
# FFT and filter tests compare against.
#
# There is exactly ONE golden set, shared by every platform, and its values are
# what scipy 1.18.0 produces on Windows (UCRT). That is deliberate: the parity
# target of this project is the Windows Python reference.
#
# Why one set needs no platform conditionals any more:
#
# The vendored ducc0 computes its unity roots (twiddle factors) at plan time from
# the *platform C library's* sin/cos (`vendor/ducc0/math/unity_roots.h`). UCRT
# and glibc differ in the last bit or two on a few percent of arguments (and by
# far more on `cpow`), so the same FFT source produced different results on the
# two platforms -- historically making a second, Linux-specific golden set
# necessary. The port now removes the dependency instead: the twiddles and
# `numpy_sincos` are served on non-Windows targets by bit-exact ports of the UCRT
# functions (`crates/ld-decode/src/optimized/ucrt_math.rs`, wired in by
# `build.rs` via DUCC_UCRT_SHIM), so Linux reproduces these values exactly and
# `cargo test` needs no Python on either platform.
#
# The *inputs* (fft32768_in, rfft32768_in, scipy_in_1024, freqz_fir_taps,
# freqz_iir_ba) are committed data and platform-independent. Only the *outputs*
# are recomputed here, with the same scipy calls that produced the committed
# versions.
#
# Modes:
#   --verify   compare the computed values against the committed set, exit
#              non-zero on any mismatch. This proves the recipes below still
#              reproduce the set byte-for-byte, so an accidental edit to a
#              golden fails instead of silently widening a parity gate. Run it
#              on Windows, where the set was produced: on another platform the
#              local scipy's own drift makes a mismatch the expected result (use
#              --census for that instead).
#   --census   report *how far* the local scipy is from the committed set --
#              cells, ulp, first divergent indices with both bit patterns -- and
#              always exit 0. This measures the Python reference's own
#              cross-platform drift, i.e. exactly the quantity the UCRT ports
#              exist to neutralise. It does NOT involve the Rust decoder.
#   (default)  write the goldens into --out.
#
# Usage:
#   python3 scripts/gen_scipy_fft_goldens.py --verify            # on Windows
#   python3 scripts/gen_scipy_fft_goldens.py --census            # elsewhere
#   python3 scripts/gen_scipy_fft_goldens.py --out crates/ld-decode/tests/data
import argparse
import os
import re
import sys

import numpy as np
import scipy
import scipy.fft
import scipy.signal

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "crates", "ld-decode", "tests", "data")


def parse_rs_complex(path):
    """Parse the legacy `&[[re, im], ...]` constant form of the 1024 golden."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    rows = re.findall(r"\[\s*(-?[0-9.eE+-]+)\s*,\s*(-?[0-9.eE+-]+)\s*\]", text)
    if not rows:
        raise SystemExit(f"{path}: no numeric rows found")
    return np.array([complex(float(a), float(b)) for a, b in rows], dtype=np.complex128)


def read_cf(path):
    """Interleaved little-endian f64 pairs -> complex array."""
    a = np.fromfile(path, dtype="<f8")
    if a.size % 2:
        raise SystemExit(f"{path}: odd number of f64, not interleaved complex")
    return a[0::2] + 1j * a[1::2]


def input_1024():
    """The 1024-point canary's input: binary if present, else the legacy literal."""
    p = os.path.join(DATA, "scipy_in_1024.f64")
    if os.path.exists(p):
        return read_cf(p)
    return parse_rs_complex(os.path.join(DATA, "scipy_in_1024.rs"))


def compute():
    """{golden filename: values} for every platform-dependent golden.

    `*_re.f64` / `*_im.f64` entries hold the complex response and are split into
    two real files by `golden_files`; everything else is a complex spectrum
    written interleaved.
    """
    out = {}

    z = read_cf(os.path.join(DATA, "fft32768_in.f64"))
    out["fft32768_fwd.f64"] = scipy.fft.fft(z)
    out["fft32768_inv.f64"] = scipy.fft.ifft(z)

    real = np.fromfile(os.path.join(DATA, "rfft32768_in.f64"), dtype="<f8")
    # The pipeline's forward transform of real data: scipy's full c2c of the
    # real array (pocketfft computes it as r2c plus a conjugate reflection).
    out["rfft32768_full.f64"] = scipy.fft.fft(real)

    # 1024-point canary; the input is committed as a legacy `.rs` constant.
    out["scipy_out_1024.f64"] = scipy.fft.fft(input_1024())

    # freqz goldens: scipy.signal.freqz with whole=True, which is the call the
    # Rust `filtfft` replicates (n_fft = worN, rfft for the FIR numerator).
    taps = np.fromfile(os.path.join(DATA, "freqz_fir_taps.f64"), dtype="<f8")
    _, h = scipy.signal.freqz(taps, [1.0], worN=32768, whole=True)
    out["freqz_fir_re.f64"] = h

    ba = np.fromfile(os.path.join(DATA, "freqz_iir_ba.f64"), dtype="<f8")
    _, h2 = scipy.signal.freqz(ba[:5], ba[5:], worN=32768, whole=True)
    out["freqz_iir_re.f64"] = h2
    return out


# The 1024-point canary is compared like every other golden, but never written
# in binary form: the tests consume the `include!`d `scipy_out_1024.rs` literal,
# so a `.f64` beside it would be a file nothing reads (and the gate would check
# that instead of the literal). `read_reference` falls back to the `.rs`, which is
# how the Windows set has always been stored.
CANARY = "scipy_out_1024.f64"


def golden_files():
    """{filename: little-endian f64 array} exactly as the tests read them."""
    files = {}
    for name, v in compute().items():
        if name.endswith("_re.f64"):
            base = name[: -len("_re.f64")]
            files[name] = np.ascontiguousarray(v.real, dtype="<f8")
            files[base + "_im.f64"] = np.ascontiguousarray(v.imag, dtype="<f8")
        else:
            a = np.empty(2 * v.size, dtype="<f8")
            a[0::2] = v.real
            a[1::2] = v.imag
            files[name] = a
    return files


def rs_goldens():
    """(input, output) complex values for the goldens the tests `include!`.

    The 1024-point canary predates the binary `.f64` form and is consumed at
    *compile* time (`include!("../tests/data/scipy_out_1024.rs")`), so a
    platform's golden set has to carry its literals as well, not only the `.f64`
    files. The input is platform-independent committed data and is rewritten
    from the parsed values so the round-trip is exercised too.
    """
    zc = input_1024()
    return zc, scipy.fft.fft(zc)


def rs_text(values):
    """The `&[[re, im], ...]` literal form, CRLF and repr-rounded like the
    committed files (`repr(float(...))` round-trips an f64 exactly, so this is
    lossless; the `float()` unwrap matters because numpy 2's scalar repr is
    `np.float64(...)`, which is not a Rust literal)."""
    lines = ["&["]
    lines += [f"    [{float(v.real)!r}, {float(v.imag)!r}]," for v in values]
    lines.append("]")
    return "\n".join(lines) + "\n"


def write_rs(out_dir, shared_dir):
    """Write the literal goldens and prove they parse back bit-for-bit.

    The 1024-point *input* goes to the shared root (it is committed data, the
    same on both platforms); only the FFT *output* literal is platform-specific
    and goes to the platform directory.
    """
    zin, zout = rs_goldens()
    for name, vals, dest in (
        ("scipy_in_1024.rs", zin, shared_dir),
        ("scipy_out_1024.rs", zout, out_dir),
    ):
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(dest, name)
        with open(path, "w", newline="\r\n", encoding="utf-8") as f:
            f.write(rs_text(vals))
        back = parse_rs_complex(path)
        if back.shape != vals.shape or not np.array_equal(
            back.view("<u8"), vals.view("<u8")
        ):
            raise SystemExit(f"{path}: re-emitted literals do not round-trip")
        where = os.path.relpath(dest, os.path.dirname(DATA))
        print(f"  {name}: wrote {vals.size} complex literals ({where})")


def read_reference(out_dir, name):
    """The committed golden for `name`, or None.

    The 1024-point canary is still stored in the legacy `&[[re, im], ...]`
    constant form (`scipy_out_1024.rs`); accept either that or the binary form so
    the census and the verify control cover every golden the tests read.
    """
    path = os.path.join(out_dir, name)
    if os.path.exists(path):
        return np.fromfile(path, dtype="<f8")
    rs = path[: -len(".f64")] + ".rs"
    if os.path.exists(rs):
        v = parse_rs_complex(rs)
        a = np.empty(2 * v.size, dtype="<f8")
        a[0::2] = v.real
        a[1::2] = v.imag
        return a
    return None


def ulp_keys(bits):
    """Monotonic unsigned key over raw f64 bit patterns (IEEE total order)."""
    u = bits.view(np.uint64)
    neg = (u >> np.uint64(63)) != 0
    return np.where(neg, ~u, u | np.uint64(0x8000000000000000))


def census(out_dir):
    """Report how far the local scipy is from the files in `out_dir`."""
    print(f"census: local scipy {scipy.__version__} numpy {np.__version__}")
    print(f"census: compared against {os.path.normpath(out_dir)}")
    for name, want in sorted(golden_files().items()):
        got = read_reference(out_dir, name)
        if got is None:
            print(f"  {name}: MISSING reference file")
            continue
        if got.shape != want.shape:               
            print(f"  {name}: SHAPE {got.shape} vs {want.shape}")
            continue
        gb = got.view("<u8")
        wb = want.view("<u8")
        diff = gb != wb
        n = int(diff.sum())
        if n == 0:
            print(f"  {name}: identical ({want.size} f64)")
            continue
        idx = np.flatnonzero(diff)
        gk = ulp_keys(gb).astype(object)
        wk = ulp_keys(wb).astype(object)
        worst = 0
        worst_at = -1
        for i in idx:
            d = abs(int(gk[i]) - int(wk[i]))
            if d > worst:
                worst = d
                worst_at = int(i)
        rel = np.max(np.abs(got[idx] - want[idx]) / np.maximum(np.abs(want[idx]), 1e-300))
        print(
            f"  {name}: {n}/{want.size} cells differ; worst {worst} ulp at idx "
            f"{worst_at}; max rel {rel:.3e}"
        )
        for i in idx[:3]:
            print(
                f"      idx {int(i)}: reference {int(gb[i])} local {int(wb[i])} "
                f"({got[i]!r} vs {want[i]!r})"
            )
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Regenerate or measure the scipy reference goldens."
    )
    ap.add_argument("--out", default=DATA, help="golden directory to write into")
    ap.add_argument("--verify", action="store_true", help="compare, fail on mismatch")
    ap.add_argument("--census", action="store_true", help="report drift, never fail")
    args = ap.parse_args()

    out_dir = args.out
    ref_dir = out_dir

    if args.census:
        return census(ref_dir)

    files = golden_files()
    print(f"scipy {scipy.__version__} numpy {np.__version__}")
    print(f"local platform {sys.platform}: goldens {os.path.normpath(out_dir)}")
    if args.verify:
        print("  (verifying against the committed set)")
    else:
        print("  (writing: the committed set lives in one directory)")
    bad = []
    for name in sorted(files):
        want = files[name]
        path = os.path.join(out_dir, name)
        if name == CANARY and not args.verify:
            print(f"  {name}: not written (the tests consume scipy_out_1024.rs)")
            continue
        if args.verify:
            got = read_reference(ref_dir, name)
            if got is None:
                print(f"  {name}: MISSING (in {ref_dir})")
                bad.append(name)
                continue
            if got.shape == want.shape and np.array_equal(
                got.view("<u8"), want.view("<u8")
            ):
                print(f"  {name}: MATCH ({want.size} f64)")
            else:
                print(f"  {name}: DIFFER (shape {got.shape} vs {want.shape})")
                bad.append(name)
        else:
            os.makedirs(out_dir, exist_ok=True)
            want.tofile(path)
            print(f"  {name}: wrote {want.size} f64")

    if not bad and not args.verify and not args.census:
        write_rs(out_dir, args.out)

    if bad:
        print(
            f"FAILED: {len(bad)} golden(s) not reproduced: {', '.join(bad)}",
            file=sys.stderr,
        )
        if sys.platform.startswith("win"):
            print(
                "The recipes and the pinned wheels have drifted from the "
                "committed set; regenerate with the same script and no "
                "--verify only after understanding why.",
                file=sys.stderr,
            )
        else:
            print(
                "Expected outside Windows: the reference's own twiddles come "
                "from the platform libm, so this host's scipy differs from the "
                "committed set. Use --census to see by how much.",
                file=sys.stderr,
            )
        return 1
    if args.verify:
        print("OK: every computed golden reproduces the committed file byte-for-byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
