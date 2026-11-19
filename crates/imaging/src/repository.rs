//! Decode/encode images from the filesystem using the pure-Rust `image` crate.

use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

use image::codecs::jpeg::JpegEncoder;
use image::metadata::Orientation;
use image::{
    DynamicImage, ExtendedColorType, ImageDecoder, ImageFormat, ImageReader,
    RgbaImage as ImageRgbaImage,
};
use stitch_domain::{
    CoreError, CoreResult, ImageLocator, ImageRepository, LoadedImage, PhotoMetadata, RgbaImage,
    SaveOptions,
};

use crate::jpeg::jpeg_quality;
use crate::metadata::{embed_metadata, extract_metadata};

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

        let bytes = std::fs::read(path).map_err(|error| CoreError::Io {
            path: locator.as_str().to_owned(),
            message: error.to_string(),
        })?;
        let decode_error = |error: image::ImageError| CoreError::Decode {
            name: locator.display_name(),
            message: error.to_string(),
        };
        let mut decoder = ImageReader::new(Cursor::new(&bytes))
            .with_guessed_format()
            .map_err(|error| CoreError::Io {
                path: locator.as_str().to_owned(),
                message: error.to_string(),
            })?
            .into_decoder()
            .map_err(decode_error)?;
        // Cameras store portrait shots sideways and record the turn in EXIF.
        // Applying it here means every later step sees the upright photo, and
        // exports, which carry no orientation tag, stay upright too.
        let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
        let mut decoded = DynamicImage::from_decoder(decoder).map_err(decode_error)?;
        decoded.apply_orientation(orientation);
        let metadata = extract_metadata(&bytes);
        let jpeg_quality = jpeg_quality(&bytes);
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

        let image = to_domain(rgba);
        let thumbnail = image.thumbnail(self.thumbnail_size.max(32));

        Ok(LoadedImage {
            locator: locator.clone(),
            name: locator.display_name(),
            image,
            thumbnail,
            metadata,
            jpeg_quality,
        })
    }

    fn save(&self, locator: &ImageLocator, image: &RgbaImage) -> CoreResult<()> {
        self.save_with_options(locator, image, &SaveOptions::default())
    }

    fn save_with_options(
        &self,
        locator: &ImageLocator,
        image: &RgbaImage,
        options: &SaveOptions,
    ) -> CoreResult<()> {
        let (bytes, _) = self.encode(locator, image, options)?;
        self.write(locator, &bytes)
    }

    fn save_with_metadata(
        &self,
        locator: &ImageLocator,
        image: &RgbaImage,
        metadata: Option<&PhotoMetadata>,
    ) -> CoreResult<()> {
        self.save_with_metadata_and_options(locator, image, metadata, &SaveOptions::default())
    }

    fn save_with_metadata_and_options(
        &self,
        locator: &ImageLocator,
        image: &RgbaImage,
        metadata: Option<&PhotoMetadata>,
        options: &SaveOptions,
    ) -> CoreResult<()> {
        let Some(metadata) = metadata.filter(|metadata| !metadata.is_empty()) else {
            return self.save_with_options(locator, image, options);
        };
        let (bytes, format) = self.encode(locator, image, options)?;
        let bytes = embed_metadata(&bytes, format, metadata).unwrap_or(bytes);
        self.write(locator, &bytes)
    }
}

impl FileImageRepository {
    fn encode(
        &self,
        locator: &ImageLocator,
        image: &RgbaImage,
        options: &SaveOptions,
    ) -> CoreResult<(Vec<u8>, ImageFormat)> {
        let format = ImageFormat::from_path(Self::path(locator))
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
        let rgba = ImageRgbaImage::from_raw(image.width(), image.height(), image.pixels().to_vec())
            .ok_or_else(|| {
                CoreError::InvalidImage(
                    "the output buffer does not match its dimensions".to_owned(),
                )
            })?;
        let dynamic = DynamicImage::ImageRgba8(rgba);
        let mut bytes = Vec::new();
        if format == ImageFormat::Jpeg {
            let rgb = dynamic.to_rgb8();
            JpegEncoder::new_with_quality(&mut bytes, options.jpeg_quality.clamp(1, 100))
                .encode(
                    rgb.as_raw(),
                    rgb.width(),
                    rgb.height(),
                    ExtendedColorType::Rgb8,
                )
                .map_err(|error| CoreError::Encode {
                    name: locator.display_name(),
                    message: error.to_string(),
                })?;
        } else {
            dynamic
                .write_to(&mut Cursor::new(&mut bytes), format)
                .map_err(|error| CoreError::Encode {
                    name: locator.display_name(),
                    message: error.to_string(),
                })?;
        }
        Ok((bytes, format))
    }

    fn write(&self, locator: &ImageLocator, bytes: &[u8]) -> CoreResult<()> {
        let path = Self::path(locator);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|error| CoreError::Io {
                    path: parent.to_string_lossy().into_owned(),
                    message: error.to_string(),
                })?;
            }
        }
        write_atomically(path, bytes).map_err(|error| CoreError::Io {
            path: path.to_string_lossy().into_owned(),
            message: error.to_string(),
        })
    }
}

/// Writes next to the destination and renames into place so an interrupted
/// export never leaves a partial image behind.
fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_owned());
    let pid = std::process::id();
    for attempt in 0..100 {
        let temp = parent.join(format!(".{name}.{pid}-{attempt}.tmp"));
        let mut file = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        };
        let mut guard = TempFileGuard::new(temp);
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(guard.path(), path)?;
        guard.commit();
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not create a unique export temporary file",
    ))
}

struct TempFileGuard {
    path: PathBuf,
    committed: bool,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            committed: false,
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
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
    fn load_reports_source_jpeg_quality() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.jpg");
        let rgb = image::RgbImage::from_fn(32, 24, |x, y| {
            image::Rgb([(x * 5) as u8, (y * 7) as u8, 90])
        });
        let mut bytes = Vec::new();
        JpegEncoder::new_with_quality(&mut bytes, 87)
            .encode(rgb.as_raw(), 32, 24, ExtendedColorType::Rgb8)
            .unwrap();
        std::fs::write(&path, bytes).unwrap();

        let loaded = FileImageRepository::default()
            .load(&ImageLocator::new(path.to_string_lossy()))
            .unwrap();
        assert_eq!(loaded.jpeg_quality, Some(87));
    }

    #[test]
    fn save_with_options_uses_the_requested_jpeg_quality() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.jpg");
        let locator = ImageLocator::new(path.to_string_lossy());
        let image = RgbaImage::filled(32, 24, [10, 20, 30, 255]);
        FileImageRepository::default()
            .save_with_options(&locator, &image, &SaveOptions { jpeg_quality: 96 })
            .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(jpeg_quality(&bytes), Some(96));
    }

    /// Inserts an APP1 segment whose EXIF holds only an orientation tag.
    fn with_exif_orientation(jpeg: &[u8], orientation: u16) -> Vec<u8> {
        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"II");
        tiff.extend_from_slice(&42u16.to_le_bytes());
        tiff.extend_from_slice(&8u32.to_le_bytes());
        tiff.extend_from_slice(&1u16.to_le_bytes());
        tiff.extend_from_slice(&0x0112u16.to_le_bytes());
        tiff.extend_from_slice(&3u16.to_le_bytes());
        tiff.extend_from_slice(&1u32.to_le_bytes());
        tiff.extend_from_slice(&orientation.to_le_bytes());
        tiff.extend_from_slice(&[0, 0]);
        tiff.extend_from_slice(&0u32.to_le_bytes());
        let mut out = jpeg[..2].to_vec();
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&((2 + 6 + tiff.len()) as u16).to_be_bytes());
        out.extend_from_slice(b"Exif\0\0");
        out.extend_from_slice(&tiff);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    #[test]
    fn load_applies_the_exif_orientation() {
        // Left half dark, right half bright, stored sideways with
        // orientation 6: the viewer must turn it 90 degrees clockwise, which
        // moves the left half to the top.
        let rgb = image::RgbImage::from_fn(32, 16, |x, _| {
            if x < 16 {
                image::Rgb([10, 10, 10])
            } else {
                image::Rgb([240, 240, 240])
            }
        });
        let mut jpeg = Vec::new();
        JpegEncoder::new_with_quality(&mut jpeg, 95)
            .encode(rgb.as_raw(), 32, 16, ExtendedColorType::Rgb8)
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("portrait.jpg");
        std::fs::write(&path, with_exif_orientation(&jpeg, 6)).unwrap();

        let loaded = FileImageRepository::default()
            .load(&ImageLocator::new(path.to_string_lossy()))
            .unwrap();
        assert_eq!((loaded.image.width(), loaded.image.height()), (16, 32));
        assert!(
            loaded.image.pixel(8, 4)[0] < 60,
            "top must be the dark half"
        );
        assert!(loaded.image.pixel(8, 28)[0] > 200, "bottom must be bright");
        assert!(
            loaded.thumbnail.width() < loaded.thumbnail.height(),
            "the thumbnail must be upright too"
        );
    }

    #[test]
    fn missing_file_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let locator = ImageLocator::new(dir.path().join("missing.png").to_string_lossy());
        let error = FileImageRepository::default().load(&locator).unwrap_err();
        assert!(matches!(error, CoreError::NotFound(_)));
    }

    #[test]
    fn save_with_metadata_embeds_exif() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.png");
        let locator = ImageLocator::new(path.to_string_lossy());
        let image = RgbaImage::filled(24, 16, [1, 2, 3, 255]);
        let metadata = PhotoMetadata {
            camera: "Nikon FM2".to_owned(),
            film: "Kodak Portra 400".to_owned(),
            ..PhotoMetadata::default()
        };
        let repository = FileImageRepository::default();
        repository
            .save_with_metadata(&locator, &image, Some(&metadata))
            .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.windows(4).any(|window| window == b"eXIf"));
        assert!(String::from_utf8_lossy(&bytes).contains("Nikon FM2"));
        let loaded = repository.load(&locator).unwrap();
        assert_eq!(loaded.image.width(), 24);
        assert_eq!(loaded.metadata.camera, "Nikon FM2");
        assert_eq!(loaded.metadata.film, "Kodak Portra 400");
    }

    #[test]
    fn load_reads_metadata_from_jpeg_and_webp() {
        for extension in ["jpg", "webp"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(format!("image.{extension}"));
            let locator = ImageLocator::new(path.to_string_lossy());
            let image = RgbaImage::filled(24, 16, [1, 2, 3, 255]);
            let metadata = PhotoMetadata {
                camera: "Nikon FM2".to_owned(),
                iso: "400".to_owned(),
                ..PhotoMetadata::default()
            };
            let repository = FileImageRepository::default();
            repository
                .save_with_metadata(&locator, &image, Some(&metadata))
                .unwrap();
            let loaded = repository.load(&locator).unwrap();
            assert_eq!(loaded.metadata.camera, "Nikon FM2");
            assert_eq!(loaded.metadata.iso, "400");
        }
    }

    #[test]
    fn save_leaves_no_temporary_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.png");
        let locator = ImageLocator::new(path.to_string_lossy());
        let image = RgbaImage::filled(8, 8, [10, 20, 30, 255]);
        FileImageRepository::default()
            .save(&locator, &image)
            .unwrap();
        let mut entries: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        entries.sort();
        assert_eq!(entries, ["image.png"]);
    }

    #[test]
    fn failed_write_removes_the_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.png");
        std::fs::create_dir(&path).unwrap();
        let locator = ImageLocator::new(path.to_string_lossy());
        let image = RgbaImage::filled(8, 8, [10, 20, 30, 255]);
        let error = FileImageRepository::default()
            .save(&locator, &image)
            .unwrap_err();
        assert!(matches!(error, CoreError::Io { .. }));
        let mut entries: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        entries.sort();
        assert_eq!(entries, ["image.png"]);
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
