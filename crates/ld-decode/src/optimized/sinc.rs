#![allow(dead_code)]

//! Kaiser-windowed sinc lookup table used by the field resampler.
//!
//! This regenerates the `sinc_lut.npz` table shipped with the Python
//! ld-decode (`build_kaiser_lut(kaiser_beta = 5, taps = 16, phases = 2**16)`);
//! the generated values match the shipped table to float32 rounding.

/// Sinc interpolation taps per output sample.
pub(crate) const SINC_TAP_COUNT: usize = 16;
/// Number of fractional phases tabulated (one row per phase, plus a duplicate
/// of the last phase so the final interpolation never runs past the table).
pub(crate) const SINC_PHASE_COUNT: usize = 1 << 16;

/// Modified Bessel function of the first kind, order 0, via the standard
/// ascending series. Converges quickly for the arguments used here (<= beta).
fn bessel_i0(x: f64) -> f64 {
    let x = x * x;
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let mut k = 1.0;
    loop {
        term *= x / (4.0 * k * k);
        let next = sum + term;
        if next == sum {
            break;
        }
        sum = next;
        k += 1.0;
    }
    sum
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let x_pi = std::f64::consts::PI * x;
        x_pi.sin() / x_pi
    }
}

fn kaiser_window(x: f64, a: f64, beta: f64, i0_beta: f64) -> f64 {
    let r = x / a;
    if !(-1.0..=1.0).contains(&r) {
        return 0.0;
    }
    let t = (1.0 - r * r).sqrt();
    bessel_i0(beta * t) / i0_beta
}

/// Build the downscale sinc lookup table used by `scale_field`.
///
/// Returns `SINC_PHASE_COUNT + 1` rows of `SINC_TAP_COUNT` f32 weights, flat
/// (row-major). The final row duplicates the previous one so the phase index
/// can run to `SINC_PHASE_COUNT` inclusive without bounds handling.
pub(crate) fn build_kaiser_lut() -> Vec<f32> {
    let beta = 5.0f64;
    let a = (SINC_TAP_COUNT / 2) as f64;
    let i0_beta = bessel_i0(beta);

    // Tap offsets: (a-1) down to -a, e.g. for 16 taps: 7..=-8.
    let offsets: Vec<f64> = (0..SINC_TAP_COUNT)
        .map(|i| (a - 1.0) - i as f64)
        .collect();

    let mut table = Vec::with_capacity((SINC_PHASE_COUNT + 1) * SINC_TAP_COUNT);

    // Match the Python generator bit-for-bit (numpy 1.x value-based casting):
    // each weight is rounded to f32 when stored, `s` accumulates the f64
    // weights, and the row is normalized with an f32-rounded sum in f32
    // arithmetic: row[j] = f32(w32[j]) / f32(s).
    let mut weights32 = [0.0f32; SINC_TAP_COUNT];

    for phase in 0..SINC_PHASE_COUNT {
        let phase_f = phase as f64 / SINC_PHASE_COUNT as f64;
        let mut sum = 0.0f64;
        for (j, &offset) in offsets.iter().enumerate() {
            let x = offset + phase_f;
            let weight = sinc(x) * kaiser_window(x, a, beta, i0_beta);
            weights32[j] = weight as f32;
            sum += weight;
        }
        let sum32 = sum as f32;
        for &weight in &weights32 {
            table.push(weight / sum32);
        }
    }

    // Duplicate the last phase.
    let last = table[table.len() - SINC_TAP_COUNT..].to_vec();
    table.extend_from_slice(&last);

    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_matches_reference() {
        // Reference values extracted from the Python ld-decode sinc_lut.npz.
        let lut = build_kaiser_lut();
        assert_eq!(lut.len(), (SINC_PHASE_COUNT + 1) * SINC_TAP_COUNT);

        // Phase 0 row: an exact alignment at tap 7 (offset 0), everything else
        // is float32 rounding residue from the normalization.
        let expected_phase0: [f32; 16] = [
            4.432085834437713e-18,
            -8.987014607284451e-18,
            1.4878434812264755e-17,
            -2.1551112216550367e-17,
            2.8183922824033373e-17,
            -3.383679902960441e-17,
            3.764031114374369e-17,
            1.0,
            3.764031114374369e-17,
            -3.383679902960441e-17,
            2.8183922824033373e-17,
            -2.1551112216550367e-17,
            1.4878434812264755e-17,
            -8.987014607284451e-18,
            4.432085834437713e-18,
            -1.4310536857848618e-18,
        ];
        for (got, want) in lut[0..16].iter().zip(expected_phase0) {
            assert!((got - want).abs() <= 2e-7, "{got} vs {want}");
        }

        // A mid-table row, phase 65536/2 = 0.5 (extracted from the npz).
        let expected_half: [f32; 16] = [
            -0.002983625279739499,
            0.008201505988836288,
            -0.01752941869199276,
            0.03299897536635399,
            -0.05823378637433052,
            0.10200048238039017,
            -0.19629739224910736,
            0.6318432688713074,
            0.6318432688713074,
            -0.19629739224910736,
            0.10200048238039017,
            -0.05823378637433052,
            0.03299897536635399,
            -0.01752941869199276,
            0.008201505988836288,
            -0.002983625279739499,
        ];
        let phase = SINC_PHASE_COUNT / 2;
        for (got, want) in lut[phase * 16..phase * 16 + 16]
            .iter()
            .zip(expected_half)
        {
            assert!((got - want).abs() <= 2e-7, "{got} vs {want}");
        }
    }
}
