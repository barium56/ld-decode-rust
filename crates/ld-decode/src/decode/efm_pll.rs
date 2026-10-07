//! EFM phase-locked loop (port of `efm_pll.EFM_PLL` from ld-decode).
//!
//! The LaserDisc EFM track is NRZ-I, so only the zero-crossing *timing* of the
//! equalised EFM samples carries information. This class detects zero
//! crossings with sub-sample interpolation and feeds the resulting sample
//! deltas to a PLL that converts them to EFM T-values (1..11), one byte per
//! symbol.

/// The mutable part of the PLL state.
///
/// `EfmPll` can be snapshotted into this (and restored from it) so a field's
/// T-values can be computed off-thread and installed later *only* if nothing
/// else advanced the state meanwhile. The arithmetic is untouched — a restored
/// PLL produces bit-identical output to one that never left the thread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EfmPllState {
    zc_previous_input: i16,
    delta: f64,
    base_period: f64,
    minimum_period: f64,
    maximum_period: f64,
    period_adjust_base: f64,
    current_period: f64,
    phase_adjust: f64,
    ref_clock_time: f64,
    frequency_hysteresis: i32,
    t_counter: i8,
}

/// EFM PLL state machine, carried across the whole decode (the EFM signal is
/// continuous across field boundaries, so state must persist between writes).
pub(crate) struct EfmPll {
    // ZC detector state.
    zc_previous_input: i16,
    delta: f64,
    // PLL output buffer.
    pll_result: Vec<i8>,
    pll_result_count: usize,
    // PLL state.
    base_period: f64,
    minimum_period: f64,
    maximum_period: f64,
    period_adjust_base: f64,
    current_period: f64,
    phase_adjust: f64,
    ref_clock_time: f64,
    frequency_hysteresis: i32,
    t_counter: i8,
}

/// The rate the reference ld-decode hard-codes (40 MSPS).
#[cfg(test)]
const REFERENCE_RATE_HZ: f64 = 40000000.0;

/// EFM channel bit rate (T1 clock) in Hz.
const EFM_BIT_RATE_HZ: f64 = 4321800.0;

impl EfmPll {
    /// `sample_rate_hz` is the rate of the samples fed to `process` (the decode
    /// input rate): the T1 clock period is that many samples per EFM bit. At
    /// 40 MHz this is exactly the reference's `40000000.0 / 4321800.0`.
    pub fn new(sample_rate_hz: f64) -> Self {
        let base_period = sample_rate_hz / EFM_BIT_RATE_HZ; // T1 clock period
        Self {
            zc_previous_input: 0,
            delta: 0.0,
            pll_result: vec![0i8; 1 << 16],
            pll_result_count: 0,
            base_period,
            minimum_period: base_period * 0.90,  // -10% minimum
            maximum_period: base_period * 1.10,  // +10% maximum
            period_adjust_base: base_period * 0.0001, // Clock adjustment step
            current_period: base_period,
            phase_adjust: 0.0,
            ref_clock_time: 0.0,
            frequency_hysteresis: 0,
            t_counter: 1,
        }
    }

    /// Snapshot the mutable state (the T-value scratch buffer is excluded —
    /// `process` resets its length and overwrites every entry it emits).
    pub fn state(&self) -> EfmPllState {
        EfmPllState {
            zc_previous_input: self.zc_previous_input,
            delta: self.delta,
            base_period: self.base_period,
            minimum_period: self.minimum_period,
            maximum_period: self.maximum_period,
            period_adjust_base: self.period_adjust_base,
            current_period: self.current_period,
            phase_adjust: self.phase_adjust,
            ref_clock_time: self.ref_clock_time,
            frequency_hysteresis: self.frequency_hysteresis,
            t_counter: self.t_counter,
        }
    }

    /// Install a previously snapshotted state.
    pub fn set_state(&mut self, state: EfmPllState) {
        self.zc_previous_input = state.zc_previous_input;
        self.delta = state.delta;
        self.base_period = state.base_period;
        self.minimum_period = state.minimum_period;
        self.maximum_period = state.maximum_period;
        self.period_adjust_base = state.period_adjust_base;
        self.current_period = state.current_period;
        self.phase_adjust = state.phase_adjust;
        self.ref_clock_time = state.ref_clock_time;
        self.frequency_hysteresis = state.frequency_hysteresis;
        self.t_counter = state.t_counter;
    }

    /// Process a buffer of EFM samples (i16), returning the EFM T-values
    /// produced (one byte per symbol).
    pub fn process(&mut self, input_buffer: &[i16]) -> Vec<i8> {
        if self.pll_result.len() < input_buffer.len() {
            self.pll_result = vec![0i8; input_buffer.len()];
        }
        self.pll_result_count = 0;

        for &curr in input_buffer {
            let prev = self.zc_previous_input;

            // Have we seen a zero-crossing?
            if (prev < 0 && curr >= 0) || (prev >= 0 && curr < 0) {
                // Interpolate to get the ZC sub-sample position fraction.
                let fraction = (-prev) as f64 / (curr - prev) as f64;
                self.push_edge(self.delta + fraction);
                // Offset the next delta by the fractional part of the result
                // in order to maintain accuracy.
                self.delta = 1.0 - fraction;
            } else {
                self.delta += 1.0;
            }

            self.zc_previous_input = curr;
        }

        self.pll_result[..self.pll_result_count].to_vec()
    }

    fn push_edge(&mut self, sample_delta: f64) {
        while sample_delta >= self.ref_clock_time {
            let next_time = self.ref_clock_time + self.current_period + self.phase_adjust;
            self.ref_clock_time = next_time;

            // Note: the tCounter < 3 check causes an 'edge push' if T is 1 or 2
            // (invalid timing lengths for the NRZI data); also 'edge pull'
            // values greater than T11.
            if (sample_delta > next_time || self.t_counter < 3) && self.t_counter < 11 {
                self.phase_adjust = 0.0;
                self.t_counter += 1;
            } else {
                let edge_delta = sample_delta - (next_time - self.current_period / 2.0);
                self.phase_adjust = edge_delta * 0.005;

                // Adjust frequency based on error.
                if edge_delta < 0.0 {
                    if self.frequency_hysteresis < 0 {
                        self.frequency_hysteresis -= 1;
                    } else {
                        self.frequency_hysteresis = -1;
                    }
                } else if edge_delta > 0.0 {
                    if self.frequency_hysteresis > 0 {
                        self.frequency_hysteresis += 1;
                    } else {
                        self.frequency_hysteresis = 1;
                    }
                } else {
                    self.frequency_hysteresis = 0;
                }

                // Update the reference clock?
                if self.frequency_hysteresis < -1 || self.frequency_hysteresis > 1 {
                    let mut aper = self.period_adjust_base * edge_delta / self.current_period;

                    // If there's been a substantial gap since the last edge
                    // (e.g. a dropout), edge_delta can be very large here, so
                    // limit how much of an adjustment we're willing to make.
                    if aper < -self.period_adjust_base {
                        aper = -self.period_adjust_base;
                    } else if aper > self.period_adjust_base {
                        aper = self.period_adjust_base;
                    }

                    self.current_period += aper;

                    if self.current_period < self.minimum_period {
                        self.current_period = self.minimum_period;
                    } else if self.current_period > self.maximum_period {
                        self.current_period = self.maximum_period;
                    }
                }

                self.pll_result[self.pll_result_count] = self.t_counter;
                self.pll_result_count += 1;

                self.t_counter = 1;
            }
        }

        // Reset refClockTime ready for the next delta but keep any error to
        // maintain accuracy.
        self.ref_clock_time -= sample_delta;
    }
}

/// Run `input` through a PLL seeded with `state`, returning both the resulting
/// state and the T-values.
///
/// This is the off-thread form of `set_state(state); process(input)`: it builds
/// the same machine and runs the same `process`, so the bytes are identical by
/// construction rather than by re-implementation.
pub(crate) fn process_pll_detached(state: EfmPllState, input: &[i16]) -> (EfmPllState, Vec<i8>) {
    // `set_state` overwrites every field, including the periods, so the rate
    // given here is never used.
    let mut pll = EfmPll::new(state.base_period * EFM_BIT_RATE_HZ);
    pll.set_state(state);
    let out = pll.process(input);
    (pll.state(), out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random EFM-ish signal (clustered zero crossings,
    /// dropouts, flat runs) to exercise every branch of the state machine.
    fn pseudo_efm(seed: u64, n: usize) -> Vec<i16> {
        let mut s = seed | 1;
        let mut out = Vec::with_capacity(n);
        let mut v: i32 = 1000;
        for i in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let r = (s >> 33) as i32;
            if i % 997 == 0 {
                v = 0; // dropout-ish flat run
            } else if r % 3 == 0 {
                v = -v + (r % 21);
            } else {
                v += (r % 7) - 3;
            }
            out.push(v.clamp(-32768, 32767) as i16);
        }
        out
    }

    /// At 40 MHz the periods are the reference's exact constants (bit for bit),
    /// so the parity with Python ld-decode cannot move.
    #[test]
    fn pll_at_40mhz_keeps_the_reference_periods() {
        let base: f64 = 40000000.0 / 4321800.0;
        let s = EfmPll::new(40_000_000.0).state();
        assert_eq!(s.base_period.to_bits(), base.to_bits());
        assert_eq!(s.minimum_period.to_bits(), (base * 0.90).to_bits());
        assert_eq!(s.maximum_period.to_bits(), (base * 1.10).to_bits());
        assert_eq!(s.period_adjust_base.to_bits(), (base * 0.0001).to_bits());
        assert_eq!(s.current_period.to_bits(), base.to_bits());
    }

    /// At another input rate the T1 period follows it (30 MHz: 6.94 samples,
    /// not the 9.26 of 40 MHz) and the +/-10 % window is centred on it.
    #[test]
    fn pll_at_30mhz_scales_the_periods() {
        let base: f64 = 30000000.0 / 4321800.0;
        let s = EfmPll::new(30_000_000.0).state();
        assert_eq!(s.base_period.to_bits(), base.to_bits());
        assert_eq!(s.minimum_period.to_bits(), (base * 0.90).to_bits());
        assert_eq!(s.maximum_period.to_bits(), (base * 1.10).to_bits());
        assert_eq!(s.period_adjust_base.to_bits(), (base * 0.0001).to_bits());
        assert_eq!(s.current_period.to_bits(), base.to_bits());
        assert_ne!(s, EfmPll::new(40_000_000.0).state());
    }

    /// The detached path must be bit-identical to the inline path across
    /// carried state, including buffer-growth boundaries.
    #[test]
    fn detached_pll_matches_inline_bitwise() {
        let sizes = [1usize, 2, 17, 4096, 65535, 65536, 65537, 70000];
        let mut inline = EfmPll::new(REFERENCE_RATE_HZ);
        let mut detached_state = EfmPll::new(REFERENCE_RATE_HZ).state();
        for (k, &n) in sizes.iter().enumerate() {
            let input = pseudo_efm(0x9E3779B97F4A7C15 ^ (k as u64), n);
            let want = inline.process(&input);
            let (next, got) = process_pll_detached(detached_state, &input);
            assert_eq!(got.len(), want.len(), "len at size {n}");
            assert!(
                got == want,
                "detached PLL diverged from inline at size {n}"
            );
            assert_eq!(next, inline.state(), "state diverged at size {n}");
            detached_state = next;
        }
    }

    fn data_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name)
    }

    fn read_i16(path: &std::path::Path) -> Vec<i16> {
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    fn read_i8(path: &std::path::Path) -> Vec<i8> {
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .into_iter()
            .map(|b| b as i8)
            .collect()
    }

    /// Feed the stock PLL's own input through the Rust PLL and demand the same
    /// T-values, symbol for symbol.
    ///
    /// The default stream is the committed synthetic fixture
    /// (`tests/data/pll_seg{0,1,2}_{in,out}.bin`, generated by
    /// `scripts/gen_pll_goldens.py` from the reference's numba `EFM_PLL`): three
    /// buffers on one PLL instance, so the state carries across the boundaries
    /// exactly as it does across fields. No Python, capture or reference tree is
    /// needed to run it.
    ///
    /// Set `LD_PLL_DIR` to a directory of `fNNN_in.bin`/`fNNN_out.bin` from an
    /// `LD_DUMP_PLL` run to replay a real capture instead. That is the
    /// higher-fidelity probe, but it needs a 7.3.0 tree (7.4.0 dropped the dump
    /// hook) plus a capture, so it cannot be what `cargo test` depends on.
    #[test]
    fn pll_matches_stock_field_stream() {
        let pairs: Vec<(Vec<i16>, Vec<i8>)> = match std::env::var_os("LD_PLL_DIR") {
            Some(dir) => {
                let dir = std::path::PathBuf::from(dir);
                let mut v = Vec::new();
                for n in 0.. {
                    let in_path = dir.join(format!("f{n:03}_in.bin"));
                    if !in_path.exists() {
                        break;
                    }
                    v.push((
                        read_i16(&in_path),
                        read_i8(&dir.join(format!("f{n:03}_out.bin"))),
                    ));
                }
                v
            }
            None => (0..3)
                .map(|n| {
                    (
                        read_i16(&data_path(&format!("pll_seg{n}_in.bin"))),
                        read_i8(&data_path(&format!("pll_seg{n}_out.bin"))),
                    )
                })
                .collect(),
        };
        assert!(pairs.len() >= 3, "need at least 3 buffers to carry state");
        // Guard against a vacuous pass: every buffer must carry a real symbol
        // stream (the committed fixture has ~1470 symbols in total, and a real
        // capture dump many more).
        let symbols: usize = pairs.iter().map(|(_, w)| w.len()).sum();
        assert!(symbols >= 1000, "only {symbols} reference symbols");
        let mut pll = EfmPll::new(REFERENCE_RATE_HZ);
        for (n, (input, want)) in pairs.iter().enumerate() {
            let out = pll.process(input);
            let m = out.len().min(want.len());
            let neq = (0..m).filter(|&i| out[i] != want[i]).count();
            if neq > 0 {
                let i = (0..m).find(|&i| out[i] != want[i]).unwrap();
                panic!(
                    "buffer {}: {} of {} symbols differ; first at {} rust={} stock={} (len {} vs {})",
                    n, neq, m, i, out[i], want[i], out.len(), want.len()
                );
            }
            assert_eq!(out.len(), want.len(), "buffer {} length", n);
        }
    }
}
