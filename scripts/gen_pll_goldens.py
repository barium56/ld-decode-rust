"""Generate the committed EFM-PLL golden stream.

    python scripts/gen_pll_goldens.py [--ref <ld-decode tree>]

The Rust port of `efm_pll.EFM_PLL` is gated by
`decode::efm_pll::tests::pll_matches_stock_field_stream`, which feeds the
*stock* PLL's input through the Rust PLL and demands the same T-values byte for
byte. The goldens have to come from the reference's own numba implementation,
so this script imports it and runs it here -- never the Rust code.

The stream is synthetic on purpose. The real gate over capture data is
`LD_PLL_DIR`-driven (an `LD_DUMP_PLL` run of a 7.3.0 tree, several MB per
field); this one is a few hundred KB, needs no capture, and runs on every
`cargo test`, which is what keeps the PLL's arithmetic pinned once that 7.3.0
tree is gone.

The signal models the EFM track rather than a sinusoid: NRZ-I runs of 3..11 T
(T = 40e6 / 4321800 samples, the PLL's own base period), softened at the edges
and dithered, over three buffers with different stress -- nominal amplitude, a
weak/noisy stretch, and a flat dropout region with no zero crossings at all.
Three separate `process()` calls on one PLL instance carry the state across
buffer boundaries, exactly as fields do in a decode.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "crates", "ld-decode", "tests", "data")

BASE_PERIOD = 40000000.0 / 4321800.0  # the value EFM_PLL itself uses for T1
SEGMENTS = 3
SEG_LEN = 30000


def nrz_i_runs(rng, nsamples):
    """A +/-1 NRZ-I waveform whose zero crossings are 3..11 T apart."""
    out = np.empty(nsamples, dtype=np.float64)
    level = 1.0
    i = 0
    while i < nsamples:
        run = int(round(rng.randint(3, 12) * BASE_PERIOD))
        out[i : i + run] = level
        i += run
        level = -level
    return out


def make_segment(seg, rng):
    wave = nrz_i_runs(rng, SEG_LEN + 64)
    # Soften the edges: a real read-back signal is band-limited, so the PLL's
    # sub-sample interpolation sees a slope, not a step.
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    wave = np.convolve(wave, kernel / kernel.sum(), mode="same")

    if seg == 0:
        amp = 12000.0
    elif seg == 1:
        amp = 3500.0  # weak signal: interpolation is much more sensitive here
    else:
        amp = 12000.0
        # A dropout: a flat stretch with no zero crossings, exercising the
        # PLL's free-running branch.
        wave[10000:14000] = 0.0

    sig = wave * amp
    if seg == 1:
        sig = sig + rng.normal(0.0, 400.0, sig.shape)
    else:
        sig = sig + rng.normal(0.0, 30.0, sig.shape)
    return np.clip(np.rint(sig), -32768, 32767).astype(np.int16)[:SEG_LEN]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref",
        default=os.environ.get("LD_REF_TREE", os.path.join(HERE, "..", "..", "_ld-decode-7.4.0-release-ref")),
        help="ld-decode tree to take lddecode.efm_pll from",
    )
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.ref))

    from lddecode.efm_pll import EFM_PLL  # noqa: E402  (needs the path above)

    pll = EFM_PLL()
    rng = np.random.RandomState(0x5EED)
    os.makedirs(DATA, exist_ok=True)
    for seg in range(SEGMENTS):
        samples = make_segment(seg, rng)
        out = np.array(pll.process(samples), dtype=np.int8)
        in_path = os.path.join(DATA, "pll_seg%d_in.bin" % seg)
        out_path = os.path.join(DATA, "pll_seg%d_out.bin" % seg)
        samples.astype("<i2").tofile(in_path)
        out.astype("i1").tofile(out_path)
        hist = np.bincount(out.astype(np.int64) + 128, minlength=256)
        used = {int(v) - 128: int(c) for v, c in enumerate(hist) if c}
        print(
            "seg%d: %d samples -> %d symbols, T-values %s"
            % (seg, len(samples), len(out), sorted(used.items()))
        )
        for p in (in_path, out_path):
            print("  %s %s" % (os.path.basename(p), hashlib.sha256(open(p, "rb").read()).hexdigest()))


if __name__ == "__main__":
    main()
