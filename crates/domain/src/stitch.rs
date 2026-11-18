//! Stitching policies, plans and results.

use std::sync::Arc;

use crate::geometry::Homography;
use crate::image::RgbaImage;
use crate::project::ImageId;

/// Tunable parameters for the whole registration/compositing pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct StitchOptions {
    /// Images larger than this are downscaled for *feature detection only*.
    /// Warping always uses the original resolution.
    pub max_analysis_dimension: u32,
    /// Upper limit on retained keypoints per image.
    pub max_features: usize,
    /// Lowe ratio test threshold. Lower is stricter.
    pub match_ratio: f32,
    /// Reprojection threshold for RANSAC, in analysis-image pixels.
    pub ransac_threshold: f64,
    /// Minimum number of geometrically consistent matches for a pair.
    pub min_inliers: usize,
    /// RANSAC iterations.
    pub ransac_iterations: usize,
    /// Hard limit on the output canvas size.
    pub max_canvas_dimension: u32,
    /// Hard limit on output canvas pixels (memory safety).
    pub max_canvas_pixels: u64,
    /// Feather width as a fraction of the shorter image side.
    pub feather_fraction: f32,
    /// Match the color balance of each layer to its reference image.
    pub normalize_colors: bool,
}

impl Default for StitchOptions {
    fn default() -> Self {
        Self {
            max_analysis_dimension: 1280,
            max_features: 3000,
            match_ratio: 0.78,
            ransac_threshold: 4.0,
            min_inliers: 12,
            ransac_iterations: 2500,
            max_canvas_dimension: 12_000,
            max_canvas_pixels: 110_000_000,
            feather_fraction: 0.08,
            normalize_colors: true,
        }
    }
}

impl StitchOptions {
    /// Clamps values into ranges that keep the pipeline well defined.
    pub fn sanitized(mut self) -> Self {
        self.max_analysis_dimension = self.max_analysis_dimension.clamp(320, 4096);
        self.max_features = self.max_features.clamp(100, 8000);
        self.match_ratio = self.match_ratio.clamp(0.5, 0.95);
        self.ransac_threshold = self.ransac_threshold.clamp(0.5, 20.0);
        self.min_inliers = self.min_inliers.clamp(6, 500);
        self.ransac_iterations = self.ransac_iterations.clamp(100, 20_000);
        self.max_canvas_dimension = self.max_canvas_dimension.clamp(512, 16_384);
        self.max_canvas_pixels = self.max_canvas_pixels.clamp(1_000_000, 150_000_000);
        self.feather_fraction = self.feather_fraction.clamp(0.0, 0.5);
        self
    }
}

/// Result of estimating one pairwise transform.
#[derive(Clone, Debug)]
pub struct HomographyEstimate {
    pub homography: Homography,
    /// Indices into the match list that agree with the homography.
    pub inliers: Vec<usize>,
    pub rms_error: f64,
    pub num_matches: usize,
}

/// One image's placement in the final canvas.
#[derive(Clone, Debug)]
pub struct ImagePlacement {
    pub image_id: ImageId,
    pub name: String,
    /// Maps source image coordinates into result-canvas coordinates.
    pub homography: Homography,
    /// Tone curve applied to the image before blending.
    pub color: ColorCurve,
}

/// Diagnostics for a candidate registration pair.
#[derive(Clone, Debug)]
pub struct PairDiagnostics {
    pub from: ImageId,
    pub to: ImageId,
    pub num_matches: usize,
    pub num_inliers: usize,
    pub rms_error: f64,
    pub accepted: bool,
    pub used_for_alignment: bool,
}

/// Human-readable pipeline diagnostics shown by the UI.
#[derive(Clone, Debug, Default)]
pub struct StitchDiagnostics {
    pub features_per_image: Vec<usize>,
    pub pairs: Vec<PairDiagnostics>,
    pub duration_ms: u128,
    pub output_width: u32,
    pub output_height: u32,
    pub output_scale: f64,
}

/// The stitched panorama.
#[derive(Clone, Debug)]
pub struct StitchResult {
    pub image: Arc<RgbaImage>,
    pub placements: Vec<ImagePlacement>,
    pub diagnostics: StitchDiagnostics,
}

/// Per-channel tone curve that maps a source image toward its reference.
///
/// A single multiplier cannot match two scans whose tone response differs:
/// the required gain depends on the pixel value. A 256-level lookup table per
/// channel captures that value dependence and is cheap to apply per pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorCurve {
    lut: [[u8; 256]; 3],
}

impl Default for ColorCurve {
    fn default() -> Self {
        Self::identity()
    }
}

impl ColorCurve {
    pub fn identity() -> Self {
        let mut lut = [[0u8; 256]; 3];
        for channel in &mut lut {
            for (level, entry) in channel.iter_mut().enumerate() {
                *entry = level as u8;
            }
        }
        Self { lut }
    }

    /// Builds a curve from a per-channel multiplier, saturating at 255.
    pub fn from_gain(gain: [f32; 3]) -> Self {
        let mut curve = Self::identity();
        for channel in 0..3 {
            for (level, entry) in curve.lut[channel].iter_mut().enumerate() {
                let value = level as f32 * gain[channel];
                *entry = value.round().clamp(0.0, 255.0) as u8;
            }
        }
        curve
    }

    /// Wraps precomputed lookup tables. Callers are responsible for keeping the
    /// tables monotone and within range.
    pub fn from_lut(lut: [[u8; 256]; 3]) -> Self {
        Self { lut }
    }

    pub fn is_identity(&self) -> bool {
        (0..3).all(|channel| {
            self.lut[channel]
                .iter()
                .enumerate()
                .all(|(level, value)| *value as usize == level)
        })
    }

    pub fn channel(&self, channel: usize) -> &[u8; 256] {
        &self.lut[channel]
    }

    pub fn lut(&self) -> &[[u8; 256]; 3] {
        &self.lut
    }

    /// Returns a curve that applies `self` first and `next` second.
    pub fn then(&self, next: &Self) -> Self {
        let mut lut = [[0u8; 256]; 3];
        for channel in 0..3 {
            for level in 0..256 {
                lut[channel][level] = next.lut[channel][self.lut[channel][level] as usize];
            }
        }
        Self { lut }
    }

    pub fn map(&self, rgb: [u8; 3]) -> [u8; 3] {
        [
            self.lut[0][rgb[0] as usize],
            self.lut[1][rgb[1] as usize],
            self.lut[2][rgb[2] as usize],
        ]
    }
}

/// A single source layer for the compositor port.
#[derive(Clone, Debug)]
pub struct WarpLayer {
    pub name: String,
    pub image: Arc<RgbaImage>,
    /// Source image coordinates -> canvas coordinates.
    pub homography: Homography,
    /// Per-channel tone curve applied before blending.
    pub color: ColorCurve,
}

/// Parameters passed to the compositor. The canvas size and layer transforms
/// have already been planned by the application layer.
#[derive(Clone, Debug)]
pub struct WarpOptions {
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub feather_fraction: f32,
}

/// Stage of a long-running stitch operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StitchStage {
    ExtractingFeatures,
    Matching,
    EstimatingTransforms,
    Compositing,
    Finished,
}

impl StitchStage {
    pub const fn label(self) -> &'static str {
        match self {
            Self::ExtractingFeatures => "Detecting features",
            Self::Matching => "Matching features",
            Self::EstimatingTransforms => "Estimating transforms",
            Self::Compositing => "Blending panorama",
            Self::Finished => "Finished",
        }
    }
}

/// A progress notification emitted by the application layer.
#[derive(Clone, Debug)]
pub struct StitchProgress {
    pub stage: StitchStage,
    /// Zero-based unit within the stage.
    pub current: usize,
    /// Total units in the stage. `None` for indeterminate progress.
    pub total: Option<usize>,
    pub message: String,
}

impl StitchProgress {
    pub fn new(
        stage: StitchStage,
        current: usize,
        total: usize,
        message: impl Into<String>,
    ) -> Self {
        Self {
            stage,
            current,
            total: Some(total),
            message: message.into(),
        }
    }

    pub fn fraction(&self) -> Option<f32> {
        match self.total {
            Some(total) if total > 0 => Some((self.current as f32 / total as f32).clamp(0.0, 1.0)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_curve_identity_leaves_levels_unchanged() {
        let curve = ColorCurve::identity();
        assert!(curve.is_identity());
        assert_eq!(curve.map([0, 128, 255]), [0, 128, 255]);
    }

    #[test]
    fn color_curve_gain_saturates_instead_of_wrapping() {
        let curve = ColorCurve::from_gain([2.0, 0.5, 1.0]);
        assert_eq!(curve.map([100, 100, 100]), [200, 50, 100]);
        assert_eq!(curve.map([200, 255, 10]), [255, 128, 10]);
    }

    #[test]
    fn color_curve_composition_applies_in_order() {
        let brighten = ColorCurve::from_gain([1.5, 1.0, 1.0]);
        let darken = ColorCurve::from_gain([0.5, 1.0, 1.0]);
        assert_eq!(brighten.then(&darken).map([100, 100, 100]), [75, 100, 100]);
        assert_eq!(darken.then(&brighten).map([100, 100, 100]), [75, 100, 100]);
    }

    #[test]
    fn sanitize_clamps_extremes() {
        let options = StitchOptions {
            max_analysis_dimension: 1,
            max_features: 0,
            match_ratio: 0.1,
            ransac_threshold: 500.0,
            min_inliers: 0,
            ransac_iterations: 0,
            max_canvas_dimension: 0,
            max_canvas_pixels: 0,
            feather_fraction: 4.0,
            normalize_colors: true,
        }
        .sanitized();
        assert!(options.max_analysis_dimension >= 320);
        assert!(options.max_features >= 100);
        assert!(options.match_ratio >= 0.5);
        assert!(options.max_canvas_dimension >= 512);
        assert!(options.feather_fraction <= 0.5);
    }
}
