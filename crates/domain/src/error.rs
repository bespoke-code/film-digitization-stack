use thiserror::Error;

pub type CoreResult<T> = Result<T, CoreError>;

/// Errors produced by domain rules or by ports.
///
/// Adapters translate their native errors into this type so that the
/// application and UI layers never see a filesystem, codec or GPU error.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CoreError {
    #[error("image not found: {0}")]
    NotFound(String),

    #[error("could not decode image '{name}': {message}")]
    Decode { name: String, message: String },

    #[error("could not encode image '{name}': {message}")]
    Encode { name: String, message: String },

    #[error("unsupported image format for '{0}'")]
    UnsupportedFormat(String),

    #[error("I/O error for '{path}': {message}")]
    Io { path: String, message: String },

    #[error("invalid image: {0}")]
    InvalidImage(String),

    #[error("feature extraction failed: {0}")]
    FeatureExtraction(String),

    #[error("homography estimation failed: {0}")]
    Homography(String),

    #[error("compositing failed: {0}")]
    Compositing(String),

    #[error("rendering failed: {0}")]
    Rendering(String),

    #[error("the project already contains the maximum of {max} images")]
    TooManyImages { max: usize },

    #[error("the project contains no images")]
    EmptyProject,

    #[error("no stitched result is available yet; run the stitch operation first")]
    NoStitchResult,

    #[error("image {0} is not part of the project")]
    UnknownImage(u64),

    #[error("not enough overlap between '{0}' and the already registered images")]
    NotEnoughOverlap(String),
}
