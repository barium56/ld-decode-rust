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

impl Default for EfmPll {
    fn default() -> Self {
        Self::new()
    }
}

impl EfmPll {
    pub fn new() -> Self {
        let base_period = 40000000.0 / 4321800.0; // T1 clock period 40MSPS / bit-rate
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
    let mut pll = EfmPll::new();
    pll.set_state(state);
    let out = pll.process(input);
    (pll.state(), out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

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

    /// The detached path must be bit-identical to the inline path across
    /// carried state, including buffer-growth boundaries.
    #[test]
    fn detached_pll_matches_inline_bitwise() {
        let sizes = [1usize, 2, 17, 4096, 65535, 65536, 65537, 70000];
        let mut inline = EfmPll::new();
        let mut detached_state = EfmPll::new().state();
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

    // Feed the concatenated stock EFM input (fields 0..N from the paired
    // LD_DUMP_PLL run) through the Rust PLL and compare against the stock
    // PLL's outputs for the same fields. Set LD_PLL_DIR to the directory
    // holding fNNN_in.bin/fNNN_out.bin.
    #[test]
    fn pll_matches_stock_field_stream() {
        let dir = std::env::var("LD_PLL_DIR").expect("set LD_PLL_DIR");
        let mut n = 0usize;
        let mut pll = EfmPll::new();
        loop {
            let in_path = format!("{}/f{:03}_in.bin", dir, n);
            let mut f = match std::fs::File::open(&in_path) {
                Ok(f) => f,
                Err(_) => break,
            };
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            let input: Vec<i16> = buf
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
            let out = pll.process(&input);
            let mut want = Vec::new();
            std::fs::File::open(format!("{}/f{:03}_out.bin", dir, n))
                .unwrap()
                .read_to_end(&mut want)
                .unwrap();
            let want: Vec<i8> = want.into_iter().map(|b| b as i8).collect();
            let m = out.len().min(want.len());
            let neq = (0..m).filter(|&i| out[i] != want[i]).count();
            if neq > 0 {
                let i = (0..m).find(|&i| out[i] != want[i]).unwrap();
                panic!(
                    "field {}: {} of {} symbols differ; first at {} rust={} stock={} (len {} vs {})",
                    n, neq, m, i, out[i], want[i], out.len(), want.len()
                );
            }
            assert_eq!(out.len(), want.len(), "field {} length", n);
            n += 1;
        }
        assert!(n >= 3, "no fields found in {dir}");
    }
}
