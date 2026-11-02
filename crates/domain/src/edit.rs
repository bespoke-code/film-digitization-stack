//! Negative inversion, levels and monochrome conversion.
//!
//! The math is pure and has no codec, GPU or UI dependency so the same
//! settings drive the preview, the histogram and exported files.

use std::borrow::Cow;

use crate::image::RgbaImage;

/// Input and output range for one channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelLevels {
    pub black: f32,
    pub white: f32,
    pub gamma: f32,
}

impl Default for ChannelLevels {
    fn default() -> Self {
        Self {
            black: 0.0,
            white: 1.0,
            gamma: 1.0,
        }
    }
}

impl ChannelLevels {
    pub fn sanitized(self) -> Self {
        let black = self.black.clamp(0.0, 0.98);
        let white = self.white.clamp(black + 0.01, 1.0);
        Self {
            black,
            white,
            gamma: self.gamma.clamp(0.05, 10.0),
        }
    }

    fn is_identity(self) -> bool {
        self.black <= f32::EPSILON
            && (self.white - 1.0).abs() <= f32::EPSILON
            && (self.gamma - 1.0).abs() <= f32::EPSILON
    }

    fn apply(self, value: f32) -> f32 {
        let normalized = ((value - self.black) / (self.white - self.black)).clamp(0.0, 1.0);
        normalized.powf(1.0 / self.gamma)
    }
}

/// Per-channel levels. The gray set applies after monochrome conversion.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Levels {
    pub red: ChannelLevels,
    pub green: ChannelLevels,
    pub blue: ChannelLevels,
    pub gray: ChannelLevels,
}

impl Levels {
    pub fn sanitized(self) -> Self {
        Self {
            red: self.red.sanitized(),
            green: self.green.sanitized(),
            blue: self.blue.sanitized(),
            gray: self.gray.sanitized(),
        }
    }

    fn is_identity(self) -> bool {
        self.red.is_identity()
            && self.green.is_identity()
            && self.blue.is_identity()
            && self.gray.is_identity()
    }
}

/// Integer crop rectangle in source-image coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl CropRect {
    pub fn clamped(self, image_width: u32, image_height: u32) -> Option<Self> {
        let x = self.x.min(image_width.saturating_sub(1));
        let y = self.y.min(image_height.saturating_sub(1));
        let width = self.width.min(image_width - x);
        let height = self.height.min(image_height - y);
        (width > 0 && height > 0).then_some(Self {
            x,
            y,
            width,
            height,
        })
    }
}

/// All adjustments for one photo.
#[derive(Clone, Debug, PartialEq)]
pub struct EditSettings {
    pub invert: bool,
    pub monochrome: bool,
    pub film_base: Option<[u8; 3]>,
    pub exposure: f32,
    pub contrast: f32,
    pub levels: Levels,
    pub crop: Option<CropRect>,
}

impl Default for EditSettings {
    fn default() -> Self {
        Self {
            invert: false,
            monochrome: false,
            film_base: None,
            exposure: 0.0,
            contrast: 0.0,
            levels: Levels::default(),
            crop: None,
        }
    }
}

impl EditSettings {
    pub fn sanitized(mut self) -> Self {
        self.exposure = self.exposure.clamp(-6.0, 6.0);
        self.contrast = self.contrast.clamp(-0.95, 0.95);
        self.levels = self.levels.sanitized();
        self
    }

    pub fn is_identity(&self) -> bool {
        !self.invert
            && !self.monochrome
            && self.film_base.is_none()
            && self.exposure.abs() <= f32::EPSILON
            && self.contrast.abs() <= f32::EPSILON
            && self.levels.is_identity()
            && self.crop.is_none()
    }

    /// Applies all adjustments at full resolution.
    pub fn develop(&self, image: &RgbaImage) -> RgbaImage {
        if self.is_identity() {
            return image.clone();
        }
        let cropped = self.cropped(image);
        self.develop_exact(cropped.as_ref())
    }

    /// Applies all adjustments and limits the longest side for display.
    pub fn develop_preview(&self, image: &RgbaImage, max_dimension: u32) -> RgbaImage {
        let cropped = self.cropped(image);
        let max_side = cropped.as_ref().width().max(cropped.as_ref().height());
        let fitted = if max_dimension > 0 && max_side > max_dimension {
            Cow::Owned(cropped.as_ref().resized_to_fit(max_dimension))
        } else {
            cropped
        };
        if self.is_identity() {
            return fitted.into_owned();
        }
        self.develop_exact(fitted.as_ref())
    }

    fn cropped<'a>(&self, image: &'a RgbaImage) -> Cow<'a, RgbaImage> {
        match self
            .crop
            .and_then(|crop| crop.clamped(image.width(), image.height()))
        {
            Some(crop) => Cow::Owned(image.cropped(crop.x, crop.y, crop.width, crop.height)),
            None => Cow::Borrowed(image),
        }
    }

    fn develop_exact(&self, image: &RgbaImage) -> RgbaImage {
        if self.is_identity() {
            return image.clone();
        }
        let settings = self.clone().sanitized();
        let luts = [
            channel_lut(&settings, 0),
            channel_lut(&settings, 1),
            channel_lut(&settings, 2),
        ];
        let gray_lut = gray_lut(&settings.levels.gray);
        let mut pixels = image.pixels().to_vec();
        for chunk in pixels.as_chunks_mut::<4>().0 {
            let r = luts[0][chunk[0] as usize];
            let g = luts[1][chunk[1] as usize];
            let b = luts[2][chunk[2] as usize];
            if settings.monochrome {
                let luma = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32)
                    .round()
                    .clamp(0.0, 255.0) as usize;
                let value = gray_lut[luma];
                chunk[0] = value;
                chunk[1] = value;
                chunk[2] = value;
            } else {
                chunk[0] = r;
                chunk[1] = g;
                chunk[2] = b;
            }
        }
        RgbaImage::new(image.width(), image.height(), pixels).expect("developed image keeps size")
    }
}

/// Estimates the film base from the bright edge of the frame.
///
/// The unexposed border is the brightest part of a negative, so the 90th
/// percentile of the outer band ignores the dark film rebate and dust.
pub fn suggest_film_base(image: &RgbaImage) -> [u8; 3] {
    let width = image.width();
    let height = image.height();
    let band = ((width.min(height) as f32 * 0.03).round() as u32).max(1);
    let step = (width.max(height) as usize / 2048).max(1);
    let mut channels: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut consider = |x: u32, y: u32| {
        let pixel = image.pixel(x, y);
        if pixel[3] == 0 {
            return;
        }
        for (channel, values) in channels.iter_mut().enumerate() {
            values.push(pixel[channel]);
        }
    };
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            if x < band || y < band || x + band >= width || y + band >= height {
                consider(x, y);
            }
            x += step as u32;
        }
        y += step as u32;
    }
    let mut out = [0u8; 3];
    for (channel, values) in channels.iter_mut().enumerate() {
        if values.is_empty() {
            continue;
        }
        values.sort_unstable();
        let index = values.len().saturating_sub(1) * 9 / 10;
        out[channel] = values[index];
    }
    out
}

fn channel_lut(settings: &EditSettings, channel: usize) -> [u8; 256] {
    let base = settings.film_base.map(|base| base[channel] as f32 / 255.0);
    let exposure = 2f32.powf(settings.exposure);
    let contrast = ((1.0 + settings.contrast) / (1.0 - settings.contrast)).clamp(0.01, 100.0);
    let levels = match channel {
        0 => settings.levels.red,
        1 => settings.levels.green,
        _ => settings.levels.blue,
    };
    let mut lut = [0u8; 256];
    for (value, slot) in lut.iter_mut().enumerate() {
        let mut normalized = value as f32 / 255.0;
        if settings.invert {
            if let Some(base) = base {
                normalized = ((normalized - base) / (1.0 - base).max(1e-4)).clamp(0.0, 1.0);
            }
            normalized = 1.0 - normalized;
        }
        normalized = (normalized - 0.5) * contrast + 0.5;
        normalized *= exposure;
        normalized = levels.apply(normalized);
        *slot = (normalized.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    lut
}

fn gray_lut(levels: &ChannelLevels) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (value, slot) in lut.iter_mut().enumerate() {
        let normalized = levels.apply(value as f32 / 255.0);
        *slot = (normalized.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inversion_flips_the_value() {
        let image = RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                [0, 0, 0, 255]
            } else {
                [255, 255, 255, 255]
            }
        });
        let settings = EditSettings {
            invert: true,
            ..EditSettings::default()
        };
        let developed = settings.develop(&image);
        assert_eq!(developed.pixel(0, 0), [255, 255, 255, 255]);
        assert_eq!(developed.pixel(1, 0), [0, 0, 0, 255]);
    }

    #[test]
    fn film_base_becomes_white_and_dense_negative_becomes_black() {
        let image = RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                [120, 80, 40, 255]
            } else {
                [255, 255, 255, 255]
            }
        });
        let settings = EditSettings {
            invert: true,
            film_base: Some([120, 80, 40]),
            ..EditSettings::default()
        };
        let developed = settings.develop(&image);
        assert_eq!(developed.pixel(0, 0), [255, 255, 255, 255]);
        assert_eq!(developed.pixel(1, 0), [0, 0, 0, 255]);
    }

    #[test]
    fn monochrome_sets_equal_channels() {
        let image = RgbaImage::filled(1, 1, [200, 40, 10, 255]);
        let settings = EditSettings {
            monochrome: true,
            ..EditSettings::default()
        };
        let pixel = settings.develop(&image).pixel(0, 0);
        assert_eq!(pixel[0], pixel[1]);
        assert_eq!(pixel[1], pixel[2]);
    }

    #[test]
    fn exposure_and_levels_change_the_range() {
        let image = RgbaImage::filled(1, 1, [64, 64, 64, 255]);
        let settings = EditSettings {
            exposure: 1.0,
            levels: Levels {
                red: ChannelLevels {
                    black: 0.0,
                    white: 0.5,
                    gamma: 1.0,
                },
                ..Levels::default()
            },
            ..EditSettings::default()
        };
        let pixel = settings.develop(&image).pixel(0, 0);
        assert_eq!(pixel[0], 255);
        assert_eq!(pixel[1], 128);
    }

    #[test]
    fn crop_limits_the_output() {
        let image = RgbaImage::filled(10, 10, [10, 20, 30, 255]);
        let settings = EditSettings {
            crop: Some(CropRect {
                x: 2,
                y: 3,
                width: 4,
                height: 5,
            }),
            ..EditSettings::default()
        };
        let developed = settings.develop(&image);
        assert_eq!((developed.width(), developed.height()), (4, 5));
    }

    #[test]
    fn suggested_film_base_follows_the_bright_border() {
        let image = RgbaImage::from_fn(100, 80, |x, y| {
            if x < 8 || y < 8 || x >= 92 || y >= 72 {
                [190, 120, 60, 255]
            } else {
                [30, 20, 10, 255]
            }
        });
        let base = suggest_film_base(&image);
        assert!(base[0] > base[1] && base[1] > base[2]);
        assert!(base[0] >= 180);
    }
}
