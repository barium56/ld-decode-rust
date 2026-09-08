#!/usr/bin/env python3
"""Compute the NTSC audio + EFM filters exactly like the reference ld-decode
(computeaudiofilters / computeefmfilter) and dump them to binary files so the
Rust port can be verified against them.

Usage: python scripts/py_audio_ref.py <outdir>
"""

import sys
import os

sys.path.insert(0, "../_ld-decode-ref")

import numpy as np
import scipy.signal as sps
import scipy.interpolate as spi
import lddecode.utils as utils

np.set_printoptions(precision=9, suppress=False)


def filtfft(filt, blocklen):
    return sps.freqz(filt[0], filt[1], blocklen, whole=1)[1]


def fft_determine_slices(center, min_bandwidth, freq_hz, bins_in):
    binwidth = freq_hz / bins_in
    cbin = np.round(center / binwidth)
    bbins = np.round(min_bandwidth / binwidth)
    nbins = 2 * (2 ** np.ceil(np.log2(bbins * 2)))
    nbins = int(nbins)
    lowbin = int(cbin - (nbins // 4))
    cut_freq = binwidth * nbins
    return lowbin, nbins, cut_freq


def fft_do_slice(fdomain, lowbin, nbins, blocklen):
    nbins_half = nbins // 2
    return np.concatenate(
        [
            fdomain[lowbin : lowbin + nbins_half],
            fdomain[blocklen - lowbin - nbins_half : blocklen - lowbin],
        ]
    )


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "audio_ref"
    os.makedirs(outdir, exist_ok=True)

    freq_hz = 40e6
    freq_half = 20e6
    freq_hz_half = 20e6
    blocklen = 32768

    audio_lfreq = (1000000 * 315 / 88 / 227.5) * 146.25
    audio_rfreq = (1000000 * 315 / 88 / 227.5) * 178.75
    print("audio_lfreq", audio_lfreq, "audio_rfreq", audio_rfreq)

    apass = 150000
    afilt_len = 512

    # ---- audio filters (computeaudiofilters) ----
    filt1_out = {}
    audio2_out = {}
    for channel, center_freq in zip(["left", "right"], [audio_lfreq, audio_rfreq]):
        audio1_fir = sps.firwin(
            afilt_len,
            [
                (center_freq - apass) / freq_hz_half,
                (center_freq + apass) / freq_hz_half,
            ],
            pass_zero=False,
        )
        lowbin, nbins, a1_freq = fft_determine_slices(center_freq, 200000, freq_hz, blocklen)
        sliced_hilbert = utils.build_hilbert(nbins)
        low_freq = freq_hz * (lowbin / blocklen)
        audio1_fft = filtfft([audio1_fir, [1.0]], blocklen)
        filt1 = fft_do_slice(audio1_fft, lowbin, nbins, blocklen) * sliced_hilbert

        N, Wn = sps.buttord(20000 / (a1_freq / 2), 24000 / (a1_freq / 2), 1, 9)
        audio2_lpf = filtfft(sps.butter(N, Wn), blocklen)
        audio2_deemp = filtfft(utils.emphasis_iir(5.3e-6, 75e-6, a1_freq), blocklen)
        audio2_filter = audio2_lpf * audio2_deemp

        print(f"{channel}: lowbin={lowbin} nbins={nbins} a1_freq={a1_freq} low_freq={low_freq} N={N} Wn={Wn} fdiv={blocklen // nbins}")

        filt1_out[channel] = filt1
        audio2_out[channel] = audio2_filter

        (filt1.real.astype("<f4")).tofile(os.path.join(outdir, f"filt1_{channel}_re.bin"))
        (filt1.imag.astype("<f4")).tofile(os.path.join(outdir, f"filt1_{channel}_im.bin"))
        (audio2_filter.real.astype("<f4")).tofile(os.path.join(outdir, f"audio2_{channel}_re.bin"))
        (audio2_filter.imag.astype("<f4")).tofile(os.path.join(outdir, f"audio2_{channel}_im.bin"))

    # ---- EFM filter (core.py's inline computeefmfilter: 0..1.9 MHz nodes) ----
    freqs = np.linspace(0.0e6, 1.9e6, num=11)
    amp = np.array([0.0, 0.215, 0.41, 0.73, 0.98, 1.03, 0.99, 0.81, 0.59, 0.42, 0.0])
    phase = np.array([0.0, -0.92, -1.03, -1.11, -1.2, -1.2, -1.2, -1.2, -1.05, -0.95, -0.8]) * 1.25
    a_interp = spi.interp1d(freqs, amp, kind="cubic")
    p_interp = spi.interp1d(freqs, phase, kind="cubic")
    freq_per_bin = freq_hz / blocklen
    nonzero_bins = int(freqs[-1] / freq_per_bin) + 1
    coeffs = np.zeros(blocklen, dtype=complex)
    bin_freqs = np.arange(nonzero_bins) * freq_per_bin
    bin_amp = a_interp(bin_freqs)
    bin_phase = p_interp(bin_freqs)
    coeffs[:nonzero_bins] = bin_amp * (np.cos(bin_phase) + (complex(0, -1) * np.sin(bin_phase)))
    fefm = coeffs * 8
    fefm *= utils.gen_bpf_supergauss(20000, 1600000, 60, 20000000, blocklen)
    fefm.real.astype("<f4").tofile(os.path.join(outdir, "fefm_re.bin"))
    fefm.imag.astype("<f4").tofile(os.path.join(outdir, "fefm_im.bin"))
    print("fefm len", len(fefm))

    print("done ->", outdir)


if __name__ == "__main__":
    main()
