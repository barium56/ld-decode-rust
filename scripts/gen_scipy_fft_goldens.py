# Regenerates -- or measures -- the platform-dependent scipy reference outputs
# that the hermetic FFT and filter tests compare against.
#
# Why the goldens are platform-dependent at all:
#
# The vendored ducc0 computes its unity roots (twiddle factors) at plan time from
# the *platform C library's* sin/cos (`vendor/ducc0/math/unity_roots.h`, e.g.
# `v1[i] = {cos(i*ang), sin(i*ang)}`). UCRT's and glibc's sin/cos differ in the
# last bit or two, so the same FFT source produces results that differ by 1-2 ulp
# between Windows and Linux. Everything else about the FFT is identical (same
# source, same 128-bit SIMD width, same operation order), which is why the
# Windows build reproduces the Windows scipy wheel bit-for-bit and the Linux
# build reproduces the Linux scipy wheel bit-for-bit -- but the two platforms'
# goldens are NOT interchangeable.
#
# The *inputs* (fft32768_in, rfft32768_in, scipy_in_1024, freqz_fir_taps,
# freqz_iir_ba) are committed data and platform-independent. Only the *outputs*
# are recomputed here, with the same scipy calls that produced the committed
# Windows versions.
#
# Modes:
#   --verify   compare the computed values against the files in --out, exit
#              non-zero on any mismatch. Run this on Windows against the
#              committed goldens: it proves the recipes below still reproduce
#              them byte-for-byte, which is what makes a Linux regeneration
#              trustworthy.
#   --census   same comparison, but reports *how far* the local (Linux) scipy is
#              from the committed (Windows) goldens -- cells, ulp, first
#              divergent indices with both bit patterns -- and always exits 0.
#              This measures the Python reference's own cross-platform drift,
#              which is what decides whether the port can match a single
#              platform-independent golden set at all. It does NOT involve the
#              Rust decoder.
#   (default)  write the goldens into --out.
#
# Usage:
#   python3 scripts/gen_scipy_fft_goldens.py --census --out crates/ld-decode/tests/data
#   python3 scripts/gen_scipy_fft_goldens.py --out "$RUNNER_TEMP/goldens"
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
    """{filename: complex values} for the goldens the tests `include!` literally.

    The 1024-point canary predates the binary `.f64` form and is consumed at
    *compile* time (`include!("../tests/data/scipy_out_1024.rs")`), so a
    platform-local regeneration has to rewrite the literals as well, not only
    the `.f64` files. The input is platform-independent committed data; it is
    rewritten from the parsed values so the round-trip is exercised too.
    """
    zc = input_1024()
    return {"scipy_in_1024.rs": zc, "scipy_out_1024.rs": scipy.fft.fft(zc)}


def rs_text(values):
    """The `&[[re, im], ...]` literal form, CRLF and repr-rounded like the
    committed files (`repr(float(...))` round-trips an f64 exactly, so this is
    lossless; the `float()` unwrap matters because numpy 2's scalar repr is
    `np.float64(...)`, which is not a Rust literal)."""
    lines = ["&["]
    lines += [f"    [{float(v.real)!r}, {float(v.imag)!r}]," for v in values]
    lines.append("]")
    return "\n".join(lines) + "\n"


def write_rs(out_dir):
    """Write the literal goldens and prove they parse back bit-for-bit."""
    for name, vals in sorted(rs_goldens().items()):
        path = os.path.join(out_dir, name)
        with open(path, "w", newline="\r\n", encoding="utf-8") as f:
            f.write(rs_text(vals))
        back = parse_rs_complex(path)
        if back.shape != vals.shape or not np.array_equal(
            back.view("<u8"), vals.view("<u8")
        ):
            raise SystemExit(f"{path}: re-emitted literals do not round-trip")
        print(f"  {name}: wrote {vals.size} complex literals")


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
        description="Regenerate or measure the platform-dependent scipy goldens."
    )
    ap.add_argument("--out", default=DATA, help="directory to write/compare against")
    ap.add_argument("--verify", action="store_true", help="compare, fail on mismatch")
    ap.add_argument("--census", action="store_true", help="report drift, never fail")
    args = ap.parse_args()

    if args.census:
        return census(args.out)

    files = golden_files()
    print(f"scipy {scipy.__version__} numpy {np.__version__} (data dir {DATA})")
    bad = []
    for name in sorted(files):
        want = files[name]
        path = os.path.join(args.out, name)
        if args.verify:
            got = read_reference(args.out, name)
            if got is None:
                print(f"  {name}: MISSING (in {args.out})")
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
            os.makedirs(args.out, exist_ok=True)
            want.tofile(path)
            print(f"  {name}: wrote {want.size} f64")

    if not bad and not args.verify and not args.census:
        write_rs(args.out)

    if bad:
        print(
            f"FAILED: {len(bad)} golden(s) not reproduced: {', '.join(bad)}",
            file=sys.stderr,
        )
        return 1
    if args.verify:
        print("OK: every computed golden reproduces the committed file byte-for-byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
