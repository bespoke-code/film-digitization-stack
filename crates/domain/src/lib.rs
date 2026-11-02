//! Pure domain model for scan-to-img.
//!
//! This crate has no dependency on a GUI, a GPU API, a filesystem layout or any
//! computer-vision library. It describes images, features, homographies,
//! projects, stitching plans and the ports (traits) that adapters implement.
//!
//! The dependency direction is always:
//!
//! ```text
//! UI adapter ──> UI core ──> application ──> domain <── adapters
//! ```
//!
//! Nothing in this crate may depend on an adapter.

pub mod error;
pub mod features;
pub mod geometry;
pub mod image;
pub mod ports;
pub mod project;
pub mod stitch;

pub use error::{CoreError, CoreResult};
pub use features::{Descriptor, Feature, FeatureMatch, FeatureSet, Keypoint};
pub use geometry::{Homography, Point};
pub use image::{GrayImage, RgbaImage};
pub use ports::{
    Clock, Compositor, FeatureExtractor, FeatureMatcher, HomographyEstimator, ImageRepository,
    LoadedImage, NoProgress, ProgressSink, Renderer, SharedCompositor, SharedEstimator,
    SharedExtractor, SharedMatcher, SharedRepository,
};
pub use project::{ImageId, ImageLocator, Project, SourceImage, MAX_IMAGES};
pub use stitch::{
    HomographyEstimate, ImagePlacement, PairDiagnostics, StitchDiagnostics, StitchOptions,
    StitchProgress, StitchResult, StitchStage, WarpLayer, WarpOptions,
};
