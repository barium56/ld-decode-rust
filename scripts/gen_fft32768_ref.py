# Generates the scipy FFT goldens the pipeline-size parity tests read
# (`crates/ld-decode/tests/data/fft32768_*.f64`, `rfft32768_*.f64`).
#
# The hermetic FFT gate in the crate only covered n=1024 for a long time, but the
# demod kernel only ever transforms 32768-point blocks (`blocklen`), so the size
# that actually decides output parity had no bit-for-bit gate against the
# reference library. Run this with the bundled 7.3.0 python, which carries the
# scipy/numpy the port targets (scipy 1.18.0, numpy 2.4.6):
#
#   cd <repo>
#   PYTHONHOME="$PWD/_ld-decode-7.3.0-release-ref/python" \
#   PYTHONPATH="$PWD/_ld-decode-7.3.0-release-ref" \
#     _ld-decode-7.3.0-release-ref/python/python.exe \
#     ld-decode-rust/scripts/gen_fft32768_ref.py
#
# Output: raw little-endian f64 (complex files interleaved re, im).
import os

import numpy as np
import scipy
import scipy.fft

N = 32768
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "crates", "ld-decode", "tests", "data")

i = np.arange(N, dtype=np.float64)
# Int16-like RF samples with a smooth component on top: the same shape the demod
# feeds the transform (exact integers after the f32 cast) plus a fractional term,
# so the rounding is stressed rather than trivial.
re = np.sin(i * 0.0137) * 1.0e4 + ((i * 7919.0) % 65536.0 - 32768.0)
im = np.cos(i * 0.0311) * 1.0e3
z = re + 1j * im

# Real-input case: non-integer, never exactly representable in f32, so the r2c
# path is exercised the way `clip(demod)` does it (f32 cast before the r2c).
real = np.sin(i * 0.0137) * 1.0e4 + np.cos(i * 0.0043) * 37.0

fwd = scipy.fft.fft(z)
inv = scipy.fft.ifft(z)
real_full = scipy.fft.fft(real)  # pocketfft computes this as r2c + reflection


def wcf(name, v):
    a = np.empty(2 * v.size, dtype=np.float64)
    a[0::2] = v.real
    a[1::2] = v.imag
    a.tofile(os.path.join(OUT, name))


def wf(name, v):
    np.asarray(v, dtype=np.float64).tofile(os.path.join(OUT, name))


wcf("fft32768_in.f64", z)
wcf("fft32768_fwd.f64", fwd)
wcf("fft32768_inv.f64", inv)
wf("rfft32768_in.f64", real)
wcf("rfft32768_full.f64", real_full)

print("scipy", scipy.__version__, "numpy", np.__version__, "n", N)
print("wrote 5 goldens to", os.path.normpath(OUT))
