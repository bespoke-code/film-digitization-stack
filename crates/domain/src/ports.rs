//! Ports implemented by infrastructure adapters.

use std::sync::Arc;

use crate::features::{FeatureMatch, FeatureSet};
use crate::image::{GrayImage, RgbaImage};
use crate::metadata::PhotoMetadata;
use crate::project::ImageLocator;
use crate::stitch::{HomographyEstimate, StitchOptions, StitchProgress, WarpLayer, WarpOptions};
use crate::CoreResult;

/// An image and its preview, independent of where it came from.
#[derive(Clone, Debug)]
pub struct LoadedImage {
    pub locator: ImageLocator,
    pub name: String,
    pub image: RgbaImage,
    pub thumbnail: RgbaImage,
}

/// Durable storage for source images and exported results.
///
/// Native adapters use the filesystem, browser adapters use fetch/object URLs
/// and tests use memory.
pub trait ImageRepository: Send + Sync {
    fn load(&self, locator: &ImageLocator) -> CoreResult<LoadedImage>;
    fn save(&self, locator: &ImageLocator, image: &RgbaImage) -> CoreResult<()>;

    /// Saves an image and embeds descriptive metadata when the format supports it.
    fn save_with_metadata(
        &self,
        locator: &ImageLocator,
        image: &RgbaImage,
        metadata: Option<&PhotoMetadata>,
    ) -> CoreResult<()> {
        let _ = metadata;
        self.save(locator, image)
    }
}

/// Detects scale-space features. Production: SIFT. Tests: deterministic fakes.
pub trait FeatureExtractor: Send + Sync {
    fn extract(&self, image: &GrayImage) -> CoreResult<FeatureSet>;
}

/// Finds tentative correspondences between two feature sets.
pub trait FeatureMatcher: Send + Sync {
    /// `ratio` is the Lowe ratio test threshold.
    fn match_features(
        &self,
        query: &FeatureSet,
        train: &FeatureSet,
        ratio: f32,
    ) -> Vec<FeatureMatch>;
}

/// Robustly estimates the transform that maps query points onto train points.
pub trait HomographyEstimator: Send + Sync {
    fn estimate(
        &self,
        query: &FeatureSet,
        train: &FeatureSet,
        matches: &[FeatureMatch],
        options: &StitchOptions,
    ) -> CoreResult<HomographyEstimate>;
}

/// Warps layers into a common canvas and blends overlapping pixels.
pub trait Compositor: Send + Sync {
    fn composite(&self, layers: &[WarpLayer], options: &WarpOptions) -> CoreResult<RgbaImage>;
}

/// Monotonic clock used for diagnostics. Injected so tests are deterministic.
pub trait Clock: Send + Sync {
    fn now_ms(&self) -> u64;
}

/// Receives progress updates from long-running application use cases.
pub trait ProgressSink: Send + Sync {
    fn report(&self, progress: StitchProgress);
}

/// A sink that discards progress updates.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoProgress;

impl ProgressSink for NoProgress {
    fn report(&self, _progress: StitchProgress) {}
}

impl<F> ProgressSink for F
where
    F: Fn(StitchProgress) + Send + Sync,
{
    fn report(&self, progress: StitchProgress) {
        self(progress);
    }
}

/// A preview renderer port used by the UI core. Implementations must be fast
/// enough to be called once per frame; the wgpu adapter only uploads when the
/// image revision changes.
pub trait Renderer: Send {
    fn set_image(&mut self, image: Arc<RgbaImage>, revision: u64) -> CoreResult<()>;
    fn clear(&mut self);
}

/// Shared handle to an image repository.
pub type SharedRepository = Arc<dyn ImageRepository>;
/// Shared handle to a feature extractor.
pub type SharedExtractor = Arc<dyn FeatureExtractor>;
/// Shared handle to a feature matcher.
pub type SharedMatcher = Arc<dyn FeatureMatcher>;
/// Shared handle to a homography estimator.
pub type SharedEstimator = Arc<dyn HomographyEstimator>;
/// Shared handle to a compositor.
pub type SharedCompositor = Arc<dyn Compositor>;
