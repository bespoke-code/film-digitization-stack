//! Negative inversion, levels and monochrome conversion.
//!
//! The math is pure and has no codec, GPU or UI dependency so the same
//! settings drive the preview, the histogram and exported files.

use std::borrow::Cow;

use rayon::prelude::*;

use crate::image::RgbaImage;

/// Below this many pixels the thread-pool overhead outweighs the parallelism.
const PARALLEL_PIXEL_THRESHOLD: usize = 1 << 16;

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

/// One control point on a tone curve in normalized input/output space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvePoint {
    pub x: f32,
    pub y: f32,
}

/// A monotone tone curve built from sorted control points.
///
/// The identity curve has two points, `(0, 0)` and `(1, 1)`, so an unused curve
/// costs nothing in the development lookup tables.
#[derive(Clone, Debug, PartialEq)]
pub struct ToneCurve {
    points: Vec<CurvePoint>,
}

impl Default for ToneCurve {
    fn default() -> Self {
        Self {
            points: vec![CurvePoint { x: 0.0, y: 0.0 }, CurvePoint { x: 1.0, y: 1.0 }],
        }
    }
}

impl ToneCurve {
    pub fn points(&self) -> &[CurvePoint] {
        &self.points
    }

    /// Replaces the control points, clamping and sorting them into a valid
    /// curve. Points that share an input value collapse into one.
    pub fn set_points(&mut self, mut points: Vec<CurvePoint>) {
        for point in &mut points {
            point.x = point.x.clamp(0.0, 1.0);
            point.y = point.y.clamp(0.0, 1.0);
        }
        points.sort_by(|a, b| a.x.total_cmp(&b.x));
        points.dedup_by(|a, b| (a.x - b.x).abs() <= 1e-4);
        if points.len() < 2 {
            self.points = ToneCurve::default().points;
        } else {
            self.points = points;
        }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn is_identity(&self) -> bool {
        self.points.len() == 2
            && self.points[0].x.abs() <= 1e-4
            && self.points[0].y.abs() <= 1e-4
            && (self.points[1].x - 1.0).abs() <= 1e-4
            && (self.points[1].y - 1.0).abs() <= 1e-4
    }

    /// Evaluates the curve with monotone cubic Hermite interpolation.
    pub fn evaluate(&self, x: f32) -> f32 {
        self.evaluate_with(x, &self.tangents())
    }

    /// Evaluates the curve with tangents that the caller computed once.
    fn evaluate_with(&self, x: f32, tangents: &[f32]) -> f32 {
        let x = x.clamp(0.0, 1.0);
        let points = &self.points;
        let last = points.len() - 1;
        if x <= points[0].x {
            return points[0].y;
        }
        if x >= points[last].x {
            return points[last].y;
        }

        let mut index = 0;
        while index < last && x > points[index + 1].x {
            index += 1;
        }
        let p0 = points[index];
        let p1 = points[index + 1];
        let h = p1.x - p0.x;
        if h <= 1e-6 {
            return p1.y;
        }

        let t = (x - p0.x) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        let y = h00 * p0.y + h10 * h * tangents[index] + h01 * p1.y + h11 * h * tangents[index + 1];
        y.clamp(0.0, 1.0)
    }

    /// Fritsch-Carlson tangents keep the interpolation monotone.
    fn tangents(&self) -> Vec<f32> {
        let points = &self.points;
        let count = points.len();
        let mut secants = Vec::with_capacity(count - 1);
        for pair in points.windows(2) {
            let dx = pair[1].x - pair[0].x;
            let dy = pair[1].y - pair[0].y;
            secants.push(if dx.abs() <= 1e-6 { 0.0 } else { dy / dx });
        }
        let mut tangents = vec![0.0; count];
        tangents[0] = secants[0];
        tangents[count - 1] = secants[count - 2];
        for index in 1..count - 1 {
            let left = secants[index - 1];
            let right = secants[index];
            tangents[index] = if left * right <= 0.0 {
                0.0
            } else {
                (left + right) * 0.5
            };
        }
        for index in 0..count - 1 {
            let secant = secants[index];
            if secant.abs() <= 1e-6 {
                tangents[index] = 0.0;
                tangents[index + 1] = 0.0;
                continue;
            }
            let a = tangents[index] / secant;
            let b = tangents[index + 1] / secant;
            let sum = a * a + b * b;
            if sum > 9.0 {
                let scale = 3.0 / sum.sqrt();
                tangents[index] = scale * a * secant;
                tangents[index + 1] = scale * b * secant;
            }
        }
        tangents
    }
}

/// Tone curves for the RGB composite, each channel and the gray channel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curves {
    pub rgb: ToneCurve,
    pub red: ToneCurve,
    pub green: ToneCurve,
    pub blue: ToneCurve,
    pub gray: ToneCurve,
}

impl Curves {
    pub fn sanitized(mut self) -> Self {
        self.rgb = self.rgb.sanitized();
        self.red = self.red.sanitized();
        self.green = self.green.sanitized();
        self.blue = self.blue.sanitized();
        self.gray = self.gray.sanitized();
        self
    }

    fn is_identity(&self) -> bool {
        self.rgb.is_identity()
            && self.red.is_identity()
            && self.green.is_identity()
            && self.blue.is_identity()
            && self.gray.is_identity()
    }
}

impl ToneCurve {
    fn sanitized(mut self) -> Self {
        let points = std::mem::take(&mut self.points);
        self.set_points(points);
        self
    }
}

/// Pixel counts per 8-bit level for each channel.
///
/// Counts are floating point because an adjusted histogram spreads one source
/// level across the output levels it covers. This removes the comb pattern that
/// 8-bit lookup-table quantisation would otherwise leave in the display.
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    pub red: [f32; 256],
    pub green: [f32; 256],
    pub blue: [f32; 256],
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            red: [0.0; 256],
            green: [0.0; 256],
            blue: [0.0; 256],
        }
    }
}

impl Histogram {
    /// Counts the levels in an image that already has its final tones.
    pub fn from_image(image: &RgbaImage) -> Self {
        let mut histogram = Self::default();
        let stride = sample_stride(image);
        for (index, pixel) in image.pixels().as_chunks::<4>().0.iter().enumerate() {
            if index % stride != 0 {
                continue;
            }
            histogram.red[pixel[0] as usize] += 1.0;
            histogram.green[pixel[1] as usize] += 1.0;
            histogram.blue[pixel[2] as usize] += 1.0;
        }
        histogram
    }

    pub fn peak(&self) -> f32 {
        self.red
            .iter()
            .chain(self.green.iter())
            .chain(self.blue.iter())
            .copied()
            .fold(0.0f32, f32::max)
            .max(1.0)
    }

    fn channel_mut(&mut self, channel: usize) -> &mut [f32; 256] {
        match channel {
            0 => &mut self.red,
            1 => &mut self.green,
            _ => &mut self.blue,
        }
    }
}

/// Caps the work a histogram does on large scans.
fn sample_stride(image: &RgbaImage) -> usize {
    (image.total_pixels() / 200_000).max(1)
}

/// Rec. 709 luma weights for gamma-encoded channels.
const LUMA_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Spreads one source level's count uniformly over the output levels it covers.
fn spread_interval(histogram: &mut [f32; 256], low: f32, high: f32, weight: f32) {
    let low = low.clamp(0.0, 1.0) * 255.0;
    let high = high.clamp(0.0, 1.0) * 255.0;
    let (low, high) = if low <= high {
        (low, high)
    } else {
        (high, low)
    };
    let width = high - low;
    if width <= 1e-6 {
        histogram[(low.round() as usize).min(255)] += weight;
        return;
    }
    let first = low.floor() as usize;
    let last = (high.floor() as usize).min(255);
    for (bin, slot) in histogram.iter_mut().enumerate().take(last + 1).skip(first) {
        let start = low.max(bin as f32);
        let end = high.min(bin as f32 + 1.0);
        let overlap = (end - start).max(0.0);
        if overlap > 0.0 {
            *slot += weight * overlap / width;
        }
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

/// Normalizes a clockwise angle into `0..360`.
pub fn normalized_rotation(degrees: f32) -> f32 {
    if !degrees.is_finite() {
        return 0.0;
    }
    let normalized = degrees.rem_euclid(360.0);
    if 360.0 - normalized <= 1e-4 {
        0.0
    } else {
        normalized
    }
}

/// True when an angle is a no-op for the viewer.
pub fn rotation_is_identity(degrees: f32) -> bool {
    let normalized = normalized_rotation(degrees);
    normalized <= 1e-4
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
    pub curves: Curves,
    pub crop: Option<CropRect>,
    /// Clockwise rotation in degrees, applied before the crop.
    pub rotation: f32,
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
            curves: Curves::default(),
            crop: None,
            rotation: 0.0,
        }
    }
}

impl EditSettings {
    pub fn sanitized(mut self) -> Self {
        self.exposure = self.exposure.clamp(-6.0, 6.0);
        self.contrast = self.contrast.clamp(-1.0, 1.0);
        self.levels = self.levels.sanitized();
        self.curves = self.curves.sanitized();
        self.rotation = normalized_rotation(self.rotation);
        self
    }

    pub fn is_identity(&self) -> bool {
        self.tone_is_identity() && self.crop.is_none() && rotation_is_identity(self.rotation)
    }

    fn tone_is_identity(&self) -> bool {
        !self.invert
            && !self.monochrome
            && self.film_base.is_none()
            && self.exposure.abs() <= f32::EPSILON
            && self.contrast.abs() <= f32::EPSILON
            && self.levels.is_identity()
            && self.curves.is_identity()
    }

    /// Applies all adjustments at full resolution.
    pub fn develop(&self, image: &RgbaImage) -> RgbaImage {
        if self.is_identity() {
            return image.clone();
        }
        let prepared = self.prepared(image);
        self.develop_exact(prepared.as_ref())
    }

    /// Applies tone and color adjustments to an image that is already prepared
    /// for display. Rotation and cropping are ignored.
    pub fn develop_tone(&self, image: &RgbaImage) -> RgbaImage {
        self.develop_exact(image)
    }

    /// Applies all adjustments and limits the longest side for display.
    pub fn develop_preview(&self, image: &RgbaImage, max_dimension: u32) -> RgbaImage {
        let prepared = self.prepared(image);
        let max_side = prepared.as_ref().width().max(prepared.as_ref().height());
        let fitted = if max_dimension > 0 && max_side > max_dimension {
            Cow::Owned(prepared.as_ref().resized_to_fit(max_dimension))
        } else {
            prepared
        };
        if self.is_identity() {
            return fitted.into_owned();
        }
        self.develop_exact(fitted.as_ref())
    }

    /// The source image after rotation and cropping, before tone adjustments.
    pub fn prepared<'a>(&self, image: &'a RgbaImage) -> Cow<'a, RgbaImage> {
        let rotated = if rotation_is_identity(self.rotation) {
            Cow::Borrowed(image)
        } else {
            Cow::Owned(image.rotated(normalized_rotation(self.rotation)))
        };
        match self
            .crop
            .and_then(|crop| crop.clamped(rotated.width(), rotated.height()))
        {
            Some(crop) => Cow::Owned(rotated.as_ref().cropped(
                crop.x,
                crop.y,
                crop.width,
                crop.height,
            )),
            None => rotated,
        }
    }

    fn develop_exact(&self, image: &RgbaImage) -> RgbaImage {
        if self.tone_is_identity() {
            return image.clone();
        }
        let settings = self.clone().sanitized();
        let luts = [
            channel_lut(&settings, 0),
            channel_lut(&settings, 1),
            channel_lut(&settings, 2),
        ];
        let gray_lut = settings
            .monochrome
            .then(|| gray_lut(&settings.levels.gray, &settings.curves.gray));
        let mut pixels = image.pixels().to_vec();
        let row_bytes = image.width() as usize * 4;
        let apply = |row: &mut [u8]| apply_luts(row, &luts, gray_lut.as_ref());
        if pixels.len() >= PARALLEL_PIXEL_THRESHOLD {
            pixels.par_chunks_mut(row_bytes).for_each(apply);
        } else {
            pixels.chunks_mut(row_bytes).for_each(apply);
        }
        RgbaImage::new(image.width(), image.height(), pixels).expect("developed image keeps size")
    }

    /// Counts the tones an image will have after these settings are applied.
    ///
    /// A direct count of the developed 8-bit pixels shows empty output levels
    /// whenever exposure or contrast stretches the range, because each 8-bit
    /// input level lands on a single 8-bit output level. Here every source
    /// level is treated as covering the continuous range between its two
    /// quantisation boundaries, so its count spreads over all output levels
    /// that range touches.
    pub fn histogram(&self, image: &RgbaImage) -> Histogram {
        if self.tone_is_identity() {
            return Histogram::from_image(image);
        }
        let settings = self.clone().sanitized();
        let stride = sample_stride(image);
        if settings.monochrome {
            settings.monochrome_histogram(image, stride)
        } else {
            settings.color_histogram(image, stride)
        }
    }

    fn color_histogram(&self, image: &RgbaImage, stride: usize) -> Histogram {
        let mut source = [[0.0f32; 256]; 3];
        for (index, pixel) in image.pixels().as_chunks::<4>().0.iter().enumerate() {
            if index % stride != 0 {
                continue;
            }
            for (channel, counts) in source.iter_mut().enumerate() {
                counts[pixel[channel] as usize] += 1.0;
            }
        }
        let mut histogram = Histogram::default();
        for (channel, counts) in source.iter().enumerate() {
            let intervals = channel_intervals(self, channel);
            let target = histogram.channel_mut(channel);
            for (value, count) in counts.iter().copied().enumerate() {
                if count > 0.0 {
                    let (low, high) = intervals[value];
                    spread_interval(target, low, high, count);
                }
            }
        }
        histogram
    }

    fn monochrome_histogram(&self, image: &RgbaImage, stride: usize) -> Histogram {
        let intervals = [
            channel_intervals(self, 0),
            channel_intervals(self, 1),
            channel_intervals(self, 2),
        ];
        let tangents = self.curves.gray.tangents();
        let mut histogram = Histogram::default();
        for (index, pixel) in image.pixels().as_chunks::<4>().0.iter().enumerate() {
            if index % stride != 0 {
                continue;
            }
            let mut low = 0.0;
            let mut high = 0.0;
            for (channel, weight) in LUMA_WEIGHTS.iter().copied().enumerate() {
                let (channel_low, channel_high) = intervals[channel][pixel[channel] as usize];
                low += weight * channel_low;
                high += weight * channel_high;
            }
            let low = gray_transform(&self.levels.gray, &self.curves.gray, low, &tangents);
            let high = gray_transform(&self.levels.gray, &self.curves.gray, high, &tangents);
            spread_interval(&mut histogram.red, low, high, 1.0);
        }
        histogram.green = histogram.red;
        histogram.blue = histogram.red;
        histogram
    }
}

fn apply_luts(row: &mut [u8], luts: &[[u8; 256]; 3], gray_lut: Option<&[u8; 256]>) {
    for chunk in row.as_chunks_mut::<4>().0 {
        let r = luts[0][chunk[0] as usize];
        let g = luts[1][chunk[1] as usize];
        let b = luts[2][chunk[2] as usize];
        match gray_lut {
            Some(gray_lut) => {
                // Fixed-point Rec. 709 weights: 0.2126/0.7152/0.0722 scaled by 2^16.
                let luma = (13933 * r as u32 + 46871 * g as u32 + 4732 * b as u32) >> 16;
                let value = gray_lut[luma as usize];
                chunk[0] = value;
                chunk[1] = value;
                chunk[2] = value;
            }
            None => {
                chunk[0] = r;
                chunk[1] = g;
                chunk[2] = b;
            }
        }
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
    let (_, curve) = channel_parts(settings, channel);
    let tangents = curve.tangents();
    let rgb_tangents = settings.curves.rgb.tangents();
    let mut lut = [0u8; 256];
    for (value, slot) in lut.iter_mut().enumerate() {
        let normalized = channel_transform(
            settings,
            channel,
            value as f32 / 255.0,
            &tangents,
            &rgb_tangents,
        );
        *slot = (normalized.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    lut
}

/// The continuous output range that each source level covers after adjustment.
fn channel_intervals(settings: &EditSettings, channel: usize) -> [(f32, f32); 256] {
    let (_, curve) = channel_parts(settings, channel);
    let tangents = curve.tangents();
    let rgb_tangents = settings.curves.rgb.tangents();
    let mut intervals = [(0.0, 0.0); 256];
    for (value, slot) in intervals.iter_mut().enumerate() {
        let low = ((value as f32 - 0.5) / 255.0).clamp(0.0, 1.0);
        let high = ((value as f32 + 0.5) / 255.0).clamp(0.0, 1.0);
        let first = channel_transform(settings, channel, low, &tangents, &rgb_tangents);
        let second = channel_transform(settings, channel, high, &tangents, &rgb_tangents);
        *slot = if first <= second {
            (first, second)
        } else {
            (second, first)
        };
    }
    intervals
}

fn channel_parts(settings: &EditSettings, channel: usize) -> (ChannelLevels, &ToneCurve) {
    let levels = match channel {
        0 => settings.levels.red,
        1 => settings.levels.green,
        _ => settings.levels.blue,
    };
    let curve = match channel {
        0 => &settings.curves.red,
        1 => &settings.curves.green,
        _ => &settings.curves.blue,
    };
    (levels, curve)
}

fn channel_transform(
    settings: &EditSettings,
    channel: usize,
    mut normalized: f32,
    curve_tangents: &[f32],
    rgb_tangents: &[f32],
) -> f32 {
    if settings.invert {
        if let Some(base) = settings.film_base {
            let base = base[channel] as f32 / 255.0;
            normalized = ((normalized - base) / (1.0 - base).max(1e-4)).clamp(0.0, 1.0);
        }
        normalized = 1.0 - normalized;
    }
    // Three stops of slope keeps every slider step visually even, including
    // at the ends of the range.
    let contrast = 2f32.powf(3.0 * settings.contrast);
    normalized = (normalized - 0.5) * contrast + 0.5;
    normalized *= 2f32.powf(settings.exposure);
    let (levels, curve) = channel_parts(settings, channel);
    normalized = levels.apply(normalized);
    normalized = curve.evaluate_with(normalized, curve_tangents);
    settings.curves.rgb.evaluate_with(normalized, rgb_tangents)
}

fn gray_transform(levels: &ChannelLevels, curve: &ToneCurve, value: f32, tangents: &[f32]) -> f32 {
    curve.evaluate_with(levels.apply(value), tangents)
}

fn gray_lut(levels: &ChannelLevels, curve: &ToneCurve) -> [u8; 256] {
    let tangents = curve.tangents();
    let mut lut = [0u8; 256];
    for (value, slot) in lut.iter_mut().enumerate() {
        let normalized = gray_transform(levels, curve, value as f32 / 255.0, &tangents);
        *slot = (normalized.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    lut
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use super::*;

    /// One display frame at 60 Hz.
    const FRAME_BUDGET: Duration = Duration::from_micros(16_667);

    fn patterned_image(width: u32, height: u32) -> RgbaImage {
        RgbaImage::from_fn(width, height, |x, y| {
            let noise = x.wrapping_mul(31) ^ y.wrapping_mul(17);
            [noise as u8, (noise >> 8) as u8, (noise >> 16) as u8, 255]
        })
    }

    fn representative_settings(contrast: f32, exposure: f32) -> EditSettings {
        EditSettings {
            invert: true,
            monochrome: true,
            film_base: Some([190, 140, 100]),
            exposure,
            contrast,
            levels: Levels {
                red: ChannelLevels {
                    black: 0.03,
                    white: 0.92,
                    gamma: 1.05,
                },
                ..Levels::default()
            },
            ..EditSettings::default()
        }
    }

    fn best_of(samples: usize, mut run: impl FnMut()) -> Duration {
        run();
        (0..samples)
            .map(|_| {
                let start = Instant::now();
                run();
                start.elapsed()
            })
            .min()
            .unwrap_or_default()
    }

    #[test]
    #[ignore = "performance test; run with `just perf`"]
    fn preview_tone_adjustments_render_within_a_frame() {
        let image = patterned_image(3072, 2048);
        let settings = representative_settings(0.3, 0.35);
        let best = best_of(6, || {
            black_box(settings.develop_preview(&image, 3072));
        });
        assert!(
            best < FRAME_BUDGET,
            "tone preview took {best:?}, above the {FRAME_BUDGET:?} frame budget"
        );
    }

    #[test]
    fn a_fine_exposure_step_changes_a_pixel_by_at_most_one_level() {
        let image = RgbaImage::filled(1, 1, [128, 128, 128, 255]);
        let base = EditSettings::default().develop(&image).pixel(0, 0)[0];
        let nudged = EditSettings {
            exposure: 0.01,
            ..EditSettings::default()
        }
        .develop(&image)
        .pixel(0, 0)[0];
        assert!(base.abs_diff(nudged) <= 1);
    }

    #[test]
    fn a_fine_contrast_step_stays_near_the_original() {
        let image = RgbaImage::from_fn(256, 1, |x, _| [x as u8, x as u8, x as u8, 255]);
        let base = EditSettings {
            contrast: 1.0,
            ..EditSettings::default()
        }
        .develop(&image);
        let nudged = EditSettings {
            contrast: 0.995,
            ..EditSettings::default()
        }
        .develop(&image);
        let worst = base
            .pixels()
            .iter()
            .zip(nudged.pixels())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(worst <= 8, "contrast step moved a channel by {worst}");
    }

    #[test]
    fn a_fine_levels_step_changes_a_pixel_by_at_most_one_level() {
        let image = RgbaImage::from_fn(256, 1, |x, _| [x as u8, x as u8, x as u8, 255]);
        for (before, after) in [
            (
                ChannelLevels {
                    black: 0.2,
                    ..ChannelLevels::default()
                },
                ChannelLevels {
                    black: 0.201,
                    ..ChannelLevels::default()
                },
            ),
            (
                ChannelLevels {
                    white: 0.8,
                    ..ChannelLevels::default()
                },
                ChannelLevels {
                    white: 0.799,
                    ..ChannelLevels::default()
                },
            ),
            (
                ChannelLevels {
                    gamma: 1.0,
                    ..ChannelLevels::default()
                },
                ChannelLevels {
                    gamma: 1.001,
                    ..ChannelLevels::default()
                },
            ),
        ] {
            let first = EditSettings {
                levels: Levels {
                    red: before,
                    ..Levels::default()
                },
                ..EditSettings::default()
            }
            .develop(&image);
            let second = EditSettings {
                levels: Levels {
                    red: after,
                    ..Levels::default()
                },
                ..EditSettings::default()
            }
            .develop(&image);
            let worst = first
                .pixels()
                .iter()
                .zip(second.pixels())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            assert!(worst <= 1, "levels step moved a channel by {worst}");
        }
    }

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
    fn rotation_happens_before_the_crop() {
        let image = RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        });
        let settings = EditSettings {
            rotation: 90.0,
            crop: Some(CropRect {
                x: 0,
                y: 1,
                width: 1,
                height: 1,
            }),
            ..EditSettings::default()
        };
        let developed = settings.develop(&image);
        assert_eq!((developed.width(), developed.height()), (1, 1));
        assert_eq!(developed.pixel(0, 0), [0, 0, 255, 255]);
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

    #[test]
    fn rotation_is_normalized_and_finite() {
        assert_eq!(normalized_rotation(0.0), 0.0);
        assert_eq!(normalized_rotation(360.0), 0.0);
        assert_eq!(normalized_rotation(-90.0), 270.0);
        assert_eq!(normalized_rotation(450.0), 90.0);
        assert_eq!(normalized_rotation(f32::NAN), 0.0);
        assert!(rotation_is_identity(720.0));
        assert!(!rotation_is_identity(0.1));
    }

    #[test]
    fn default_curve_is_identity() {
        let curve = ToneCurve::default();
        assert!(curve.is_identity());
        for value in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!((curve.evaluate(value) - value).abs() < 1e-5);
        }
    }

    #[test]
    fn curve_inverts_values() {
        let mut curve = ToneCurve::default();
        curve.set_points(vec![
            CurvePoint { x: 0.0, y: 1.0 },
            CurvePoint { x: 1.0, y: 0.0 },
        ]);
        assert!((curve.evaluate(0.0) - 1.0).abs() < 1e-5);
        assert!((curve.evaluate(1.0) - 0.0).abs() < 1e-5);
        assert!((curve.evaluate(0.5) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn curve_midpoint_lifts_the_midtones() {
        let mut curve = ToneCurve::default();
        curve.set_points(vec![
            CurvePoint { x: 0.0, y: 0.0 },
            CurvePoint { x: 0.5, y: 0.75 },
            CurvePoint { x: 1.0, y: 1.0 },
        ]);
        assert!((curve.evaluate(0.5) - 0.75).abs() < 1e-5);
        assert!(curve.evaluate(0.25) > 0.25 && curve.evaluate(0.25) < 0.75);
    }

    #[test]
    fn curve_points_are_sorted_and_clamped() {
        let mut curve = ToneCurve::default();
        curve.set_points(vec![
            CurvePoint { x: 2.0, y: -1.0 },
            CurvePoint { x: -1.0, y: 2.0 },
            CurvePoint { x: 0.5, y: 0.5 },
        ]);
        let points = curve.points();
        assert!(points.windows(2).all(|pair| pair[0].x <= pair[1].x));
        assert!(points.iter().all(|point| (0.0..=1.0).contains(&point.x)));
        assert!(points.iter().all(|point| (0.0..=1.0).contains(&point.y)));
    }

    #[test]
    fn curve_changes_the_developed_pixel() {
        let image = RgbaImage::filled(1, 1, [128, 128, 128, 255]);
        let mut curves = Curves::default();
        curves.rgb.set_points(vec![
            CurvePoint { x: 0.0, y: 0.0 },
            CurvePoint { x: 0.5, y: 0.9 },
            CurvePoint { x: 1.0, y: 1.0 },
        ]);
        let settings = EditSettings {
            curves,
            ..EditSettings::default()
        };
        let pixel = settings.develop(&image).pixel(0, 0);
        assert!(pixel[0] > 200);
    }

    #[test]
    fn identity_histogram_counts_the_image_levels() {
        let image = RgbaImage::filled(10, 10, [12, 34, 56, 255]);
        let histogram = EditSettings::default().histogram(&image);
        assert_eq!(histogram.red[12], 100.0);
        assert_eq!(histogram.green[34], 100.0);
        assert_eq!(histogram.blue[56], 100.0);
    }

    #[test]
    fn adjusted_histogram_has_no_empty_output_levels() {
        let image = RgbaImage::from_fn(256, 1, |x, _| [x as u8, x as u8, x as u8, 255]);
        let settings = EditSettings {
            exposure: 0.4,
            contrast: 0.3,
            ..EditSettings::default()
        };
        let histogram = settings.histogram(&image);
        assert!(
            histogram.red.iter().all(|count| *count > 0.0),
            "adjusted histogram still has empty bins"
        );
    }

    #[test]
    fn adjusted_histogram_preserves_the_sample_count() {
        let image = RgbaImage::from_fn(256, 1, |x, _| [x as u8, x as u8, x as u8, 255]);
        let settings = EditSettings {
            exposure: -0.7,
            contrast: -0.4,
            ..EditSettings::default()
        };
        let histogram = settings.histogram(&image);
        let total: f32 = histogram.red.iter().sum();
        assert!((total - 256.0).abs() < 0.01, "count was {total}");
    }

    #[test]
    fn monochrome_histogram_has_equal_channels() {
        let image = RgbaImage::from_fn(256, 1, |x, _| [x as u8, 255 - x as u8, x as u8 / 2, 255]);
        let settings = EditSettings {
            monochrome: true,
            exposure: 0.5,
            contrast: 0.2,
            ..EditSettings::default()
        };
        let histogram = settings.histogram(&image);
        assert_eq!(histogram.red, histogram.green);
        assert_eq!(histogram.green, histogram.blue);
    }
}
