use serde::Deserialize;

/// Colour encoding standard. Only NTSC is implemented for now; PAL is kept as
/// a variant so profiles can carry the field, but decoding rejects it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
pub enum ColorSystem {
    #[serde(rename = "NTSC")]
    Ntsc,
    #[serde(rename = "PAL")]
    Pal,
}

impl ColorSystem {
    pub fn as_str(self) -> &'static str {
        match self {
            ColorSystem::Ntsc => "NTSC",
            ColorSystem::Pal => "PAL",
        }
    }
}

/// Interpolation used by the wow level adjustment, mapped to a spline degree.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
pub enum WowInterpolation {
    Linear,
    Quadratic,
    Cubic,
}

impl WowInterpolation {
    pub fn spline_degree(self) -> usize {
        match self {
            WowInterpolation::Linear => 1,
            WowInterpolation::Quadratic => 2,
            WowInterpolation::Cubic => 3,
        }
    }
}

impl Default for WowInterpolation {
    fn default() -> Self {
        WowInterpolation::Linear
    }
}

/// Everything the decoder needs to know about a decode job. Mirrors the
/// relevant `RFDecode`/`LDdecode` constructor arguments from the Python
/// ld-decode (`main.py`), restricted to the NTSC video path.
#[derive(Clone, Debug)]
pub struct DecodeRequest {
    /// Input sample rate in MHz.
    pub inputfreq: f64,
    pub system: ColorSystem,

    // Video options.
    /// Substitute lower-bandwidth decode settings (FilterParams_*_lowband).
    pub lowband: bool,
    /// Notch filter on decoded video to reduce colour 'wobble'.
    pub ntsc_color_notch: bool,
    /// Custom de-emphasis time constants in usec (0 = keep defaults): (high, low).
    pub deemp_coeff: (f64, f64),
    /// De-emphasis strength multiplier.
    pub deemp_str: f64,
    /// MTF compensation multiplier (--MTF).
    pub mtf_level: f64,
    /// MTF compensation offset (--MTF_offset).
    pub mtf_offset: f64,

    /// Auto level control (AGC) on first fields (--noAGC disables).
    pub use_agc: bool,
    /// Enable dropout detection (--noDOD disables).
    pub do_dod: bool,

    /// Smoothing applied to the wow level adjustment (0 disables).
    pub wow_level_adjust_smoothing: f32,
    pub wow_interpolation_method: WowInterpolation,

    /// Overrides for DecoderParams (ire0/hz_ire/vsync_ire are carried
    /// separately as the mutable calibration levels).
    pub decoder_params_override: std::collections::HashMap<String, f64>,

    /// Write the raw (pre-TBC-conversion) luma as float instead of u16.
    pub rf_export_raw_tbc: bool,
    /// Adjust ire0 from the picture content (blanking-level tracking).
    pub rf_ire0_adjust: bool,
}

impl Default for DecodeRequest {
    fn default() -> Self {
        Self {
            inputfreq: 40.0,
            system: ColorSystem::Ntsc,
            lowband: false,
            ntsc_color_notch: false,
            deemp_coeff: (0.0, 0.0),
            deemp_str: 1.0,
            mtf_level: 1.0,
            mtf_offset: 0.0,
            use_agc: true,
            do_dod: true,
            wow_level_adjust_smoothing: 0.0,
            wow_interpolation_method: WowInterpolation::Linear,
            decoder_params_override: std::collections::HashMap::new(),
            rf_export_raw_tbc: false,
            rf_ire0_adjust: false,
        }
    }
}
