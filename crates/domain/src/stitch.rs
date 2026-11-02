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
            max_canvas_pixels: 36_000_000,
            feather_fraction: 0.08,
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
        self.max_canvas_pixels = self.max_canvas_pixels.clamp(1_000_000, 64_000_000);
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

/// A single source layer for the compositor port.
#[derive(Clone, Debug)]
pub struct WarpLayer {
    pub name: String,
    pub image: Arc<RgbaImage>,
    /// Source image coordinates -> canvas coordinates.
    pub homography: Homography,
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
        }
        .sanitized();
        assert!(options.max_analysis_dimension >= 320);
        assert!(options.max_features >= 100);
        assert!(options.match_ratio >= 0.5);
        assert!(options.max_canvas_dimension >= 512);
        assert!(options.feather_fraction <= 0.5);
    }
}
