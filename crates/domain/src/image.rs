//! Small, dependency-free image types used by the domain.
//!
//! Keeping these types in the domain means every adapter (file decoding, GPU
//! preview, compositing) agrees on pixel layout without the domain depending on
//! a codec or graphics crate.

use crate::error::{CoreError, CoreResult};

/// 8-bit RGBA image with straight (non-premultiplied) alpha.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbaImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl RgbaImage {
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> CoreResult<Self> {
        let expected = width as usize * height as usize * 4;
        if width == 0 || height == 0 {
            return Err(CoreError::InvalidImage(
                "width and height must be non-zero".to_owned(),
            ));
        }
        if pixels.len() != expected {
            return Err(CoreError::InvalidImage(format!(
                "expected {expected} bytes for {width}x{height} RGBA, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn filled(width: u32, height: u32, color: [u8; 4]) -> Self {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for _ in 0..width as usize * height as usize {
            pixels.extend_from_slice(&color);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Builds an image from a closure returning `[r, g, b, a]` for each pixel.
    pub fn from_fn(width: u32, height: u32, mut f: impl FnMut(u32, u32) -> [u8; 4]) -> Self {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&f(x, y));
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub fn pixels_mut(&mut self) -> &mut [u8] {
        &mut self.pixels
    }

    pub fn into_pixels(self) -> Vec<u8> {
        self.pixels
    }

    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, color: [u8; 4]) {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        self.pixels[i..i + 4].copy_from_slice(&color);
    }

    /// Converts to normalized luminance in `0..=1` (Rec. 709 weights).
    pub fn to_gray(&self) -> GrayImage {
        let mut data = Vec::with_capacity(self.width as usize * self.height as usize);
        for chunk in self.pixels.as_chunks::<4>().0 {
            let r = chunk[0] as f32 / 255.0;
            let g = chunk[1] as f32 / 255.0;
            let b = chunk[2] as f32 / 255.0;
            data.push(0.2126 * r + 0.7152 * g + 0.0722 * b);
        }
        GrayImage {
            width: self.width,
            height: self.height,
            data,
        }
    }

    /// Bilinearly samples the image. Coordinates outside the image return `None`.
    pub fn sample_bilinear(&self, x: f64, y: f64) -> Option<[u8; 4]> {
        if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 {
            return None;
        }
        let max_x = self.width as f64 - 1.0;
        let max_y = self.height as f64 - 1.0;
        if x > max_x + 1e-9 || y > max_y + 1e-9 {
            return None;
        }
        let x = x.clamp(0.0, max_x);
        let y = y.clamp(0.0, max_y);
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;

        let p00 = self.pixel(x0, y0);
        let p10 = self.pixel(x1, y0);
        let p01 = self.pixel(x0, y1);
        let p11 = self.pixel(x1, y1);
        let mut out = [0u8; 4];
        for c in 0..4 {
            let top = p00[c] as f64 * (1.0 - fx) + p10[c] as f64 * fx;
            let bottom = p01[c] as f64 * (1.0 - fx) + p11[c] as f64 * fx;
            out[c] = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
        }
        Some(out)
    }

    /// Resizes to exactly `width` x `height` using bilinear interpolation.
    pub fn resize_to(&self, width: u32, height: u32) -> Self {
        if width == self.width && height == self.height {
            return self.clone();
        }
        let sx = self.width as f64 / width.max(1) as f64;
        let sy = self.height as f64 / height.max(1) as f64;
        Self::from_fn(width, height, |x, y| {
            let src_x = (x as f64 + 0.5) * sx - 0.5;
            let src_y = (y as f64 + 0.5) * sy - 0.5;
            self.sample_bilinear(
                src_x.clamp(0.0, self.width as f64 - 1.0),
                src_y.clamp(0.0, self.height as f64 - 1.0),
            )
            .unwrap_or([0, 0, 0, 0])
        })
    }

    /// Returns the size after a clockwise rotation by `degrees`.
    pub fn rotated_dimensions(width: u32, height: u32, degrees: f32) -> (u32, u32) {
        let degrees = degrees.rem_euclid(360.0);
        let quarter = degrees / 90.0;
        if (quarter - quarter.round()).abs() <= 1e-4 {
            return match quarter.round() as i64 % 4 {
                1 | 3 => (height, width),
                _ => (width, height),
            };
        }
        let (_, _, new_width, new_height) = rotation_bounds(width, height, degrees);
        (new_width, new_height)
    }

    /// Rotates clockwise by `degrees`. Quarter turns are lossless; other
    /// angles are bilinearly resampled and padded with transparency.
    pub fn rotated(&self, degrees: f32) -> Self {
        let degrees = degrees.rem_euclid(360.0);
        if degrees <= 1e-4 || 360.0 - degrees <= 1e-4 {
            return self.clone();
        }
        let quarter = degrees / 90.0;
        if (quarter - quarter.round()).abs() <= 1e-4 {
            return match quarter.round() as i64 % 4 {
                1 => self.rotated_quarter(1),
                2 => self.rotated_quarter(2),
                3 => self.rotated_quarter(3),
                _ => self.clone(),
            };
        }

        let (min_x, min_y, width, height) = rotation_bounds(self.width, self.height, degrees);
        let radians = f64::from(degrees).to_radians();
        let (sin, cos) = radians.sin_cos();
        let center_x = (self.width.saturating_sub(1)) as f64 / 2.0;
        let center_y = (self.height.saturating_sub(1)) as f64 / 2.0;

        Self::from_fn(width, height, |x, y| {
            let rx = x as f64 + min_x;
            let ry = y as f64 + min_y;
            let sx = center_x + cos * rx + sin * ry;
            let sy = center_y - sin * rx + cos * ry;
            self.sample_bilinear(sx, sy).unwrap_or([0, 0, 0, 0])
        })
    }

    fn rotated_quarter(&self, quarter: i64) -> Self {
        let (width, height) = match quarter % 4 {
            1 | 3 => (self.height, self.width),
            _ => (self.width, self.height),
        };
        Self::from_fn(width, height, |x, y| match quarter % 4 {
            1 => self.pixel(y, self.height - 1 - x),
            2 => self.pixel(self.width - 1 - x, self.height - 1 - y),
            3 => self.pixel(self.width - 1 - y, x),
            _ => self.pixel(x, y),
        })
    }

    /// Copies an axis-aligned rectangle, clamped to the image bounds.
    pub fn cropped(&self, x: u32, y: u32, width: u32, height: u32) -> Self {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        let width = width.min(self.width - x).max(1);
        let height = height.min(self.height - y).max(1);
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for row in y..y + height {
            let start = (row as usize * self.width as usize + x as usize) * 4;
            let end = start + width as usize * 4;
            pixels.extend_from_slice(&self.pixels[start..end]);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Downscales proportionally so that neither side exceeds `max_dimension`.
    /// Images that already fit are returned unchanged.
    pub fn resized_to_fit(&self, max_dimension: u32) -> Self {
        let max_side = self.width.max(self.height);
        if max_dimension == 0 || max_side <= max_dimension {
            return self.clone();
        }
        let scale = max_dimension as f64 / max_side as f64;
        let width = ((self.width as f64 * scale).round() as u32).max(1);
        let height = ((self.height as f64 * scale).round() as u32).max(1);
        self.resize_to(width, height)
    }

    /// Produces a thumbnail that fits inside `max_dimension`.
    pub fn thumbnail(&self, max_dimension: u32) -> Self {
        self.resized_to_fit(max_dimension)
    }

    pub fn total_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// Bounding box of a clockwise-rotated image in pixel-center coordinates.
fn rotation_bounds(width: u32, height: u32, degrees: f32) -> (f64, f64, u32, u32) {
    let radians = f64::from(degrees).to_radians();
    let (sin, cos) = radians.sin_cos();
    let center_x = (width.saturating_sub(1)) as f64 / 2.0;
    let center_y = (height.saturating_sub(1)) as f64 / 2.0;
    let last_x = width.saturating_sub(1) as f64;
    let last_y = height.saturating_sub(1) as f64;
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in [(0.0, 0.0), (last_x, 0.0), (last_x, last_y), (0.0, last_y)] {
        let dx = x - center_x;
        let dy = y - center_y;
        let rx = cos * dx - sin * dy;
        let ry = sin * dx + cos * dy;
        min_x = min_x.min(rx);
        min_y = min_y.min(ry);
        max_x = max_x.max(rx);
        max_y = max_y.max(ry);
    }
    (
        min_x,
        min_y,
        (max_x - min_x).round().max(0.0) as u32 + 1,
        (max_y - min_y).round().max(0.0) as u32 + 1,
    )
}

/// Single-channel floating point image in the range `0..=1`.
#[derive(Clone, Debug, PartialEq)]
pub struct GrayImage {
    width: u32,
    height: u32,
    data: Vec<f32>,
}

impl GrayImage {
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> CoreResult<Self> {
        if width == 0 || height == 0 {
            return Err(CoreError::InvalidImage(
                "width and height must be non-zero".to_owned(),
            ));
        }
        if data.len() != width as usize * height as usize {
            return Err(CoreError::InvalidImage(format!(
                "expected {} samples for {width}x{height} gray, got {}",
                width as usize * height as usize,
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn from_fn(width: u32, height: u32, mut f: impl FnMut(u32, u32) -> f32) -> Self {
        let mut data = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                data.push(f(x, y));
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn at(&self, x: u32, y: u32) -> f32 {
        self.data[y as usize * self.width as usize + x as usize]
    }

    pub fn at_signed(&self, x: i64, y: i64) -> Option<f32> {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return None;
        }
        Some(self.at(x as u32, y as u32))
    }

    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    pub fn variance(&self) -> f32 {
        let mean = self.mean();
        self.data
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / self.data.len().max(1) as f32
    }

    /// Bilinear sample in normalized image coordinates.
    pub fn sample(&self, x: f64, y: f64) -> Option<f32> {
        if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 {
            return None;
        }
        let max_x = self.width as f64 - 1.0;
        let max_y = self.height as f64 - 1.0;
        if x > max_x || y > max_y {
            return None;
        }
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = (x - x0 as f64) as f32;
        let fy = (y - y0 as f64) as f32;
        let top = self.at(x0, y0) * (1.0 - fx) + self.at(x1, y0) * fx;
        let bottom = self.at(x0, y1) * (1.0 - fx) + self.at(x1, y1) * fx;
        Some(top * (1.0 - fy) + bottom * fy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_round_trip_pixels() {
        let mut image = RgbaImage::filled(4, 3, [1, 2, 3, 4]);
        image.set_pixel(2, 1, [9, 8, 7, 6]);
        assert_eq!(image.pixel(2, 1), [9, 8, 7, 6]);
        assert_eq!(image.pixel(0, 0), [1, 2, 3, 4]);
    }

    #[test]
    fn luma_uses_rec709() {
        let image = RgbaImage::from_fn(1, 1, |_, _| [255, 255, 255, 255]);
        assert!((image.to_gray().at(0, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn resize_preserves_aspect() {
        let image = RgbaImage::filled(200, 100, [0, 0, 0, 255]);
        let resized = image.resized_to_fit(50);
        assert_eq!((resized.width(), resized.height()), (50, 25));
    }

    #[test]
    fn bilinear_sample_out_of_bounds() {
        let image = RgbaImage::filled(2, 2, [0, 0, 0, 255]);
        assert!(image.sample_bilinear(-0.1, 0.0).is_none());
        assert!(image.sample_bilinear(1.5, 0.0).is_none());
    }

    #[test]
    fn invalid_buffer_is_rejected() {
        assert!(RgbaImage::new(2, 2, vec![0; 15]).is_err());
        assert!(RgbaImage::new(0, 2, vec![]).is_err());
    }

    #[test]
    fn quarter_turns_rotate_pixels_losslessly() {
        let image = RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        });
        let clockwise = image.rotated(90.0);
        assert_eq!((clockwise.width(), clockwise.height()), (1, 2));
        assert_eq!(clockwise.pixel(0, 0), [255, 0, 0, 255]);
        assert_eq!(clockwise.pixel(0, 1), [0, 0, 255, 255]);

        let half = image.rotated(180.0);
        assert_eq!((half.width(), half.height()), (2, 1));
        assert_eq!(half.pixel(0, 0), [0, 0, 255, 255]);
        assert_eq!(half.pixel(1, 0), [255, 0, 0, 255]);

        let counter_clockwise = image.rotated(270.0);
        assert_eq!(counter_clockwise.pixel(0, 0), [0, 0, 255, 255]);
        assert_eq!(counter_clockwise.pixel(0, 1), [255, 0, 0, 255]);
    }

    #[test]
    fn four_quarter_turns_are_identity() {
        let image = RgbaImage::from_fn(5, 3, |x, y| [x as u8, y as u8, 7, 255]);
        let round_trip = image
            .rotated(90.0)
            .rotated(90.0)
            .rotated(90.0)
            .rotated(90.0);
        assert_eq!(round_trip, image);
    }

    #[test]
    fn arbitrary_rotation_expands_the_bounds() {
        let image = RgbaImage::filled(40, 20, [10, 20, 30, 255]);
        let rotated = image.rotated(30.0);
        assert!(rotated.width() > 40 && rotated.height() > 20);
        let dimensions = RgbaImage::rotated_dimensions(40, 20, 30.0);
        assert_eq!((rotated.width(), rotated.height()), dimensions);
    }
}
