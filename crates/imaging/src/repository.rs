//! Decode/encode images from the filesystem using the pure-Rust `image` crate.

use std::path::Path;

use image::{DynamicImage, ImageFormat, RgbaImage as ImageRgbaImage};
use stitch_domain::{CoreError, CoreResult, ImageLocator, ImageRepository, LoadedImage, RgbaImage};

/// Filesystem-backed [`ImageRepository`].
#[derive(Clone, Debug)]
pub struct FileImageRepository {
    /// Optional maximum side length applied at load time. Larger images are
    /// downscaled with bilinear filtering. `None` keeps original resolution.
    pub max_dimension: Option<u32>,
    /// Size of the generated list thumbnail (longest side).
    pub thumbnail_size: u32,
}

impl Default for FileImageRepository {
    fn default() -> Self {
        Self {
            max_dimension: Some(12_000),
            thumbnail_size: 320,
        }
    }
}

impl FileImageRepository {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_dimension(mut self, max_dimension: Option<u32>) -> Self {
        self.max_dimension = max_dimension;
        self
    }

    fn path(locator: &ImageLocator) -> &Path {
        Path::new(locator.as_str())
    }
}

impl ImageRepository for FileImageRepository {
    fn load(&self, locator: &ImageLocator) -> CoreResult<LoadedImage> {
        let path = Self::path(locator);
        if !path.exists() {
            return Err(CoreError::NotFound(locator.as_str().to_owned()));
        }
        if !path.is_file() {
            return Err(CoreError::Io {
                path: locator.as_str().to_owned(),
                message: "not a regular file".to_owned(),
            });
        }

        let decoded = image::open(path).map_err(|error| CoreError::Decode {
            name: locator.display_name(),
            message: error.to_string(),
        })?;
        let mut rgba = decoded.to_rgba8();
        if let Some(max_dimension) = self.max_dimension {
            let (width, height) = rgba.dimensions();
            if width.max(height) > max_dimension {
                let scale = max_dimension as f64 / width.max(height) as f64;
                let new_width = ((width as f64 * scale).round() as u32).max(1);
                let new_height = ((height as f64 * scale).round() as u32).max(1);
                rgba = image::imageops::resize(
                    &rgba,
                    new_width,
                    new_height,
                    image::imageops::FilterType::Lanczos3,
                );
            }
        }

        let image = to_domain(rgba.clone());
        let thumbnail = image.thumbnail(self.thumbnail_size.max(32));

        Ok(LoadedImage {
            locator: locator.clone(),
            name: locator.display_name(),
            image,
            thumbnail,
        })
    }

    fn save(&self, locator: &ImageLocator, image: &RgbaImage) -> CoreResult<()> {
        let path = Self::path(locator);
        let format = ImageFormat::from_path(path)
            .map_err(|_| CoreError::UnsupportedFormat(locator.as_str().to_owned()))?;
        if !matches!(
            format,
            ImageFormat::Png
                | ImageFormat::Jpeg
                | ImageFormat::WebP
                | ImageFormat::Bmp
                | ImageFormat::Tiff
        ) {
            return Err(CoreError::UnsupportedFormat(locator.as_str().to_owned()));
        }

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|error| CoreError::Io {
                    path: parent.to_string_lossy().into_owned(),
                    message: error.to_string(),
                })?;
            }
        }

        let rgba = ImageRgbaImage::from_raw(image.width(), image.height(), image.pixels().to_vec())
            .ok_or_else(|| {
                CoreError::InvalidImage(
                    "the output buffer does not match its dimensions".to_owned(),
                )
            })?;
        let mut dynamic = DynamicImage::ImageRgba8(rgba);
        if matches!(format, ImageFormat::Jpeg) {
            dynamic = DynamicImage::ImageRgb8(dynamic.to_rgb8());
        }
        dynamic
            .save_with_format(path, format)
            .map_err(|error| CoreError::Encode {
                name: locator.display_name(),
                message: error.to_string(),
            })
    }
}

fn to_domain(image: ImageRgbaImage) -> RgbaImage {
    let (width, height) = image.dimensions();
    RgbaImage::new(width, height, image.into_raw())
        .expect("image crate buffers always match their dimensions")
}

#[cfg(test)]
mod tests {
    use super::*;
    use stitch_domain::ImageRepository;

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.png");
        let locator = ImageLocator::new(path.to_string_lossy());
        let image = RgbaImage::from_fn(24, 16, |x, y| [x as u8 * 8, y as u8 * 12, 40, 255]);
        let repository = FileImageRepository::default();
        repository.save(&locator, &image).unwrap();
        let loaded = repository.load(&locator).unwrap();
        assert_eq!(loaded.image.width(), 24);
        assert_eq!(loaded.image.height(), 16);
        assert_eq!(loaded.image.pixel(3, 2), image.pixel(3, 2));
        assert!(loaded.thumbnail.width() <= 320);
    }

    #[test]
    fn missing_file_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let locator = ImageLocator::new(dir.path().join("missing.png").to_string_lossy());
        let error = FileImageRepository::default().load(&locator).unwrap_err();
        assert!(matches!(error, CoreError::NotFound(_)));
    }

    #[test]
    fn unsupported_extension_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let locator = ImageLocator::new(dir.path().join("out.xyz").to_string_lossy());
        let image = RgbaImage::filled(4, 4, [0, 0, 0, 255]);
        assert!(FileImageRepository::default()
            .save(&locator, &image)
            .is_err());
    }
}
