#!/usr/bin/env python3
"""Generate a synthetic NTSC Laserdisc RF capture (.lds + .s16) for tests.

The signal is a simple FM representation of the video waveform: instantaneous
frequency (Hz) follows the IRE levels, and the RF output is a sinusoid whose
phase is the integral of that frequency (a real-valued FM signal, exactly what
the phase-discriminating ld-decode demodulator consumes).

Timing (NTSC, 40 Msps):
  line period 63.5556 us = 2542 samples
  field 1: 263 lines at line positions 0, 1, 2, ... (relative to frame)
  field 2: 262 lines starting half a line after field 1's last line
  frame: 525 lines

Each field's first 9 lines are the vertical blanking interval, whose pulses
sit on half-line positions:
  pre-equalizing: 6 x 2.3 us pulses
  vertical sync:  6 x 22.4 us pulses (the 27.1 us serration period minus the
                  4.7 us serration, which the 0.5 MHz sync lowpass merges)
  post-equalizing: 6 x 2.3 us pulses

Normal lines: 4.7 us HSYNC at -40 IRE, colour burst, then active video at a
chosen IRE level.

IRE -> Hz: 0 IRE = 8.1 MHz, 1 IRE = 1700000/140 Hz.
"""

import math
import struct
import sys

FREQ = 40e6  # sample rate, Hz
FSC = 315.0 / 88.0  # colour subcarrier, MHz
LINE_PERIOD = 1.0 / (FSC / 227.5) * 1e-6  # seconds (63.5556 us)
LINE_SAMPLES = int(round(FREQ * LINE_PERIOD))  # 2542
HALF_LINE = LINE_SAMPLES // 2
FRAME_LINES = 525
IRE0 = 8.1e6
HZ_IRE = 1.7e6 / 140.0


def iretohz(ire):
    return IRE0 + HZ_IRE * ire


def main():
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    outpath = sys.argv[2] if len(sys.argv) > 2 else "test.lds"

    # The instantaneous-frequency waveform, as a list of per-sample rates.
    total_samples = int((frames * FRAME_LINES + 20) * LINE_SAMPLES)
    rate = [iretohz(0.0)] * total_samples

    def emit(sample, freq):
        if 0 <= sample < total_samples:
            rate[sample] = freq

    def emit_run(start, length, freq):
        for i in range(start, min(start + length, total_samples)):
            rate[i] = freq

    def line_content(start):
        """Normal line: hsync + burst + active video at 50 IRE."""
        emit_run(start, int(4.7e-6 * FREQ), iretohz(-40.0))
        burst_start = start + int(5.3e-6 * FREQ)
        burst_len = int(2.5e-6 * FREQ)
        for i in range(burst_start, min(burst_start + burst_len, total_samples)):
            t = i - burst_start
            rate[i] = iretohz(0.0) + HZ_IRE * 20.0 * math.sin(
                math.tau * FSC * 1e6 * t / FREQ
            )
        emit_run(start + int(9.45e-6 * FREQ), int((LINE_PERIOD - 10.45e-6) * FREQ), iretohz(50.0))

    def vblank_content(start):
        """9 lines of vblank with half-line-spaced pulses."""
        for half in range(18):
            half_start = start + half * HALF_LINE
            if half < 6 or half >= 12:
                emit_run(half_start, int(2.3e-6 * FREQ), iretohz(-40.0))
            else:
                emit_run(half_start, int(22.4e-6 * FREQ), iretohz(-40.0))

    for frame in range(frames):
        for field_idx, nlines in enumerate((263, 262)):
            base = frame * FRAME_LINES
            field_line0 = base + (0 if field_idx == 0 else 262)
            shift = 0.0 if field_idx == 0 else 0.5  # lines
            for line in range(nlines):
                start = int((field_line0 + line + shift) * LINE_SAMPLES)
                if line < 9:
                    vblank_content(start)
                else:
                    line_content(start)

    # Convert to RF (FM) and scale to int16 like a real capture.
    out = [0.0] * total_samples
    phase = 0.0
    for i in range(total_samples):
        out[i] = math.sin(phase)
        phase += math.tau * rate[i] / FREQ
        if phase > math.tau:
            phase -= math.tau

    amplitude = 12000.0
    state = 12345
    samples = []
    for v in out:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        noise = ((state / 0x7FFFFFFF) - 0.5) * 2.0 * 200.0
        samples.append(int(round(v * amplitude + noise)))

    # Write packed 10-bit .lds (4 samples / 5 bytes).
    with open(outpath, "wb") as f:
        groups = len(samples) // 4
        for g in range(groups):
            ten = [(s >> 6) + 512 for s in samples[g * 4:(g + 1) * 4]]
            packed = bytes([
                (ten[0] & 0x3FC) >> 2,
                ((ten[0] & 0x003) << 6) | ((ten[1] & 0x3F0) >> 4),
                ((ten[1] & 0x00F) << 4) | ((ten[2] & 0x3C0) >> 6),
                ((ten[2] & 0x03F) << 2) | ((ten[3] & 0x300) >> 8),
                ten[3] & 0x0FF,
            ])
            f.write(packed)
    with open(outpath.replace(".lds", ".s16"), "wb") as f2:
        f2.write(struct.pack("<%dh" % len(samples), *samples))


if __name__ == "__main__":
    main()
