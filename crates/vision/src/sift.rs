//! A dependency-free implementation of David Lowe's SIFT detector/descriptor.
//!
//! The implementation follows the classic four stages:
//!
//! 1. Gaussian scale space + difference-of-Gaussians,
//! 2. scale-space extrema with quadratic refinement and edge rejection,
//! 3. gradient-orientation assignment (possibly multiple per keypoint),
//! 4. a 128-bin rotation-normalized descriptor.
//!
//! It is intentionally an adapter: the application only sees the
//! [`FeatureExtractor`] port.

use std::f32::consts::PI;

use rayon::prelude::*;
use stitch_domain::{
    CoreError, CoreResult, Descriptor, Feature, FeatureExtractor, FeatureSet, GrayImage, Keypoint,
};

const TAU: f32 = 2.0 * PI;

/// SIFT parameters. Defaults are tuned for document/photo scans.
#[derive(Clone, Debug)]
pub struct SiftConfig {
    /// Scales per octave (Lowe uses 3).
    pub scales_per_octave: usize,
    /// Base blur sigma.
    pub sigma: f32,
    /// Blur already present in the input image.
    pub assumed_blur: f32,
    /// DoG contrast threshold on images normalized to 0..=1.
    pub contrast_threshold: f32,
    /// Edge response threshold (`r` in Lowe's paper, typically 10).
    pub edge_threshold: f32,
    /// Maximum number of retained keypoints.
    pub max_features: usize,
    /// Maximum number of octaves.
    pub max_octaves: usize,
}

impl Default for SiftConfig {
    fn default() -> Self {
        Self {
            scales_per_octave: 3,
            sigma: 1.6,
            assumed_blur: 0.5,
            contrast_threshold: 0.01,
            edge_threshold: 10.0,
            max_features: 6000,
            max_octaves: 6,
        }
    }
}

impl SiftConfig {
    fn sanitized(mut self) -> Self {
        self.scales_per_octave = self.scales_per_octave.clamp(1, 6);
        self.sigma = self.sigma.clamp(0.8, 4.0);
        self.assumed_blur = self.assumed_blur.clamp(0.0, self.sigma - 0.1);
        self.contrast_threshold = self.contrast_threshold.clamp(1e-5, 0.5);
        self.edge_threshold = self.edge_threshold.clamp(2.0, 50.0);
        self.max_features = self.max_features.clamp(32, 50_000);
        self.max_octaves = self.max_octaves.clamp(1, 12);
        self
    }
}

/// Production SIFT extractor.
#[derive(Clone, Debug, Default)]
pub struct SiftExtractor {
    config: SiftConfig,
}

impl SiftExtractor {
    pub fn new(config: SiftConfig) -> Self {
        Self {
            config: config.sanitized(),
        }
    }

    pub fn config(&self) -> &SiftConfig {
        &self.config
    }
}

impl FeatureExtractor for SiftExtractor {
    fn extract(&self, image: &GrayImage) -> CoreResult<FeatureSet> {
        if image.width() < 16 || image.height() < 16 {
            return Err(CoreError::FeatureExtraction(
                "SIFT needs an image of at least 16x16".to_owned(),
            ));
        }
        Ok(detect_and_describe(image, &self.config))
    }
}

#[derive(Clone)]
struct Plane {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl Plane {
    fn from_gray(image: &GrayImage) -> Self {
        Self {
            width: image.width() as usize,
            height: image.height() as usize,
            data: image.data().to_vec(),
        }
    }

    fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    fn at_clamped(&self, x: i32, y: i32) -> f32 {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        self.at(x, y)
    }

    fn gaussian_blur(&self, sigma: f32) -> Self {
        if sigma <= 0.05 || self.width < 3 || self.height < 3 {
            return self.clone();
        }
        let radius = (3.0 * sigma).ceil() as i32;
        if radius < 1 {
            return self.clone();
        }
        let mut kernel = Vec::with_capacity((radius * 2 + 1) as usize);
        let mut sum = 0.0f32;
        for i in -radius..=radius {
            let value = (-(i * i) as f32 / (2.0 * sigma * sigma)).exp();
            kernel.push(value);
            sum += value;
        }
        for value in &mut kernel {
            *value /= sum;
        }

        // Horizontal pass, one row at a time.
        let mut horizontal = vec![0.0f32; self.data.len()];
        horizontal
            .par_chunks_mut(self.width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, slot) in row.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for (k, weight) in kernel.iter().enumerate() {
                        let sx = x as i32 + k as i32 - radius;
                        acc += self.at_clamped(sx, y as i32) * weight;
                    }
                    *slot = acc;
                }
            });

        // Vertical pass.
        let mut output = vec![0.0f32; self.data.len()];
        output
            .par_chunks_mut(self.width)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..self.width {
                    let mut acc = 0.0;
                    for (k, weight) in kernel.iter().enumerate() {
                        let sy = y as i32 + k as i32 - radius;
                        let sy = sy.clamp(0, self.height as i32 - 1) as usize;
                        acc += horizontal[sy * self.width + x] * weight;
                    }
                    row[x] = acc;
                }
            });

        Self {
            width: self.width,
            height: self.height,
            data: output,
        }
    }

    /// Averages 2x2 blocks and halves the resolution.
    fn downsample(&self) -> Self {
        let width = self.width / 2;
        let height = self.height / 2;
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let x0 = x * 2;
                let y0 = y * 2;
                data.push(
                    (self.at(x0, y0)
                        + self.at(x0 + 1, y0)
                        + self.at(x0, y0 + 1)
                        + self.at(x0 + 1, y0 + 1))
                        * 0.25,
                );
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    fn difference(&self, other: &Self) -> Self {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self {
            width: self.width,
            height: self.height,
            data,
        }
    }
}

struct Octave {
    gaussian: Vec<Plane>,
    /// Difference-of-Gaussians between consecutive Gaussian planes.
    dog: Vec<Plane>,
    octave: usize,
}

fn build_scale_space(image: &GrayImage, config: &SiftConfig) -> Vec<Octave> {
    let s = config.scales_per_octave;
    let base_sigma = (config.sigma * config.sigma - config.assumed_blur * config.assumed_blur)
        .max(0.01)
        .sqrt();
    let mut base = Plane::from_gray(image).gaussian_blur(base_sigma);
    let mut octaves = Vec::new();

    for octave_index in 0..config.max_octaves {
        if base.width < 16 || base.height < 16 {
            break;
        }
        let mut gaussian = Vec::with_capacity(s + 3);
        gaussian.push(base.clone());
        for level in 1..(s + 3) {
            let previous_sigma = config.sigma * 2.0f32.powf((level - 1) as f32 / s as f32);
            let current_sigma = config.sigma * 2.0f32.powf(level as f32 / s as f32);
            let incremental = (current_sigma * current_sigma - previous_sigma * previous_sigma)
                .max(0.0)
                .sqrt();
            let next = gaussian[level - 1].gaussian_blur(incremental);
            gaussian.push(next);
        }
        let dog: Vec<Plane> = gaussian
            .windows(2)
            .map(|pair| pair[0].difference(&pair[1]))
            .collect();
        let next_base = gaussian[s].downsample();
        octaves.push(Octave {
            gaussian,
            dog,
            octave: octave_index,
        });
        if next_base.width < 16 || next_base.height < 16 {
            break;
        }
        base = next_base;
    }
    octaves
}

fn detect_and_describe(image: &GrayImage, config: &SiftConfig) -> FeatureSet {
    let octaves = build_scale_space(image, config);
    let mut features = Vec::new();
    let s = config.scales_per_octave;

    for octave in &octaves {
        let scale_factor = (1usize << octave.octave) as f32;
        for layer in 1..=s {
            let below = &octave.dog[layer - 1];
            let current = &octave.dog[layer];
            let above = &octave.dog[layer + 1];
            let width = current.width;
            let height = current.height;

            for y in 1..height.saturating_sub(1) {
                for x in 1..width.saturating_sub(1) {
                    let center = current.at(x, y);
                    if center.abs() < config.contrast_threshold {
                        continue;
                    }
                    if !is_extremum(below, current, above, x, y, center) {
                        continue;
                    }
                    if is_edge_response(current, x, y, config.edge_threshold) {
                        continue;
                    }

                    let Some((rx, ry, rlayer)) =
                        refine_extremum(octave, layer, x, y, config.contrast_threshold)
                    else {
                        continue;
                    };

                    let local_sigma = config.sigma * 2.0f32.powf(rlayer / s as f32);
                    let gaussian_index =
                        (rlayer.round() as usize).clamp(1, octave.gaussian.len().saturating_sub(1));
                    let plane = &octave.gaussian[gaussian_index];
                    let xi = rx.round() as i32;
                    let yi = ry.round() as i32;
                    if xi <= 0
                        || yi <= 0
                        || xi >= plane.width as i32 - 1
                        || yi >= plane.height as i32 - 1
                    {
                        continue;
                    }

                    let responses = orientation_histogram(plane, xi, yi, local_sigma);
                    for orientation in responses {
                        let Some(descriptor) = describe(plane, xi, yi, local_sigma, orientation)
                        else {
                            continue;
                        };
                        features.push(Feature {
                            keypoint: Keypoint {
                                x: rx * scale_factor,
                                y: ry * scale_factor,
                                scale: local_sigma * scale_factor,
                                orientation,
                                response: center.abs(),
                            },
                            descriptor,
                        });
                    }
                }
            }
        }
    }

    features.sort_by(|a, b| b.keypoint.response.total_cmp(&a.keypoint.response));
    features.truncate(config.max_features);
    FeatureSet::new(features)
}

#[inline]
fn is_extremum(
    below: &Plane,
    current: &Plane,
    above: &Plane,
    x: usize,
    y: usize,
    center: f32,
) -> bool {
    let mut positive = center > 0.0;
    let mut negative = center < 0.0;
    for (di, plane) in [below, current, above].iter().enumerate() {
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                if di == 1 && dx == 0 && dy == 0 {
                    continue;
                }
                let value = plane.at_clamped(x as i32 + dx, y as i32 + dy);
                if positive && value >= center {
                    positive = false;
                }
                if negative && value <= center {
                    negative = false;
                }
                if !positive && !negative {
                    return false;
                }
            }
        }
    }
    positive || negative
}

fn is_edge_response(plane: &Plane, x: usize, y: usize, edge_threshold: f32) -> bool {
    let dxx = plane.at(x + 1, y) - 2.0 * plane.at(x, y) + plane.at(x - 1, y);
    let dyy = plane.at(x, y + 1) - 2.0 * plane.at(x, y) + plane.at(x, y - 1);
    let dxy = (plane.at(x + 1, y + 1) - plane.at(x + 1, y - 1) - plane.at(x - 1, y + 1)
        + plane.at(x - 1, y - 1))
        * 0.25;
    let trace = dxx + dyy;
    let determinant = dxx * dyy - dxy * dxy;
    if determinant <= 0.0 || !determinant.is_finite() {
        return true;
    }
    trace * trace / determinant >= (edge_threshold + 1.0) * (edge_threshold + 1.0) / edge_threshold
}

/// Fits a quadratic in (x, y, layer) and iterates until the offset is below
/// half a pixel. Returns refined (x, y, layer) in octave coordinates.
fn refine_extremum(
    octave: &Octave,
    layer: usize,
    x: usize,
    y: usize,
    contrast_threshold: f32,
) -> Option<(f32, f32, f32)> {
    let max_layer = octave.dog.len() - 2;
    let mut xi = x as i32;
    let mut yi = y as i32;
    let mut li = layer as i32;

    for _ in 0..6 {
        if li < 1 || li as usize > max_layer {
            return None;
        }
        let plane = &octave.dog[li as usize];
        if xi < 1 || yi < 1 || xi >= plane.width as i32 - 1 || yi >= plane.height as i32 - 1 {
            return None;
        }
        let below = &octave.dog[(li - 1) as usize];
        let above = &octave.dog[(li + 1) as usize];

        let center = plane.at(xi as usize, yi as usize);
        let gx = (plane.at((xi + 1) as usize, yi as usize)
            - plane.at((xi - 1) as usize, yi as usize))
            * 0.5;
        let gy = (plane.at(xi as usize, (yi + 1) as usize)
            - plane.at(xi as usize, (yi - 1) as usize))
            * 0.5;
        let gs = (above.at(xi as usize, yi as usize) - below.at(xi as usize, yi as usize)) * 0.5;

        let dxx = plane.at((xi + 1) as usize, yi as usize) - 2.0 * center
            + plane.at((xi - 1) as usize, yi as usize);
        let dyy = plane.at(xi as usize, (yi + 1) as usize) - 2.0 * center
            + plane.at(xi as usize, (yi - 1) as usize);
        let dss =
            above.at(xi as usize, yi as usize) - 2.0 * center + below.at(xi as usize, yi as usize);
        let dxy = (plane.at((xi + 1) as usize, (yi + 1) as usize)
            - plane.at((xi + 1) as usize, (yi - 1) as usize)
            - plane.at((xi - 1) as usize, (yi + 1) as usize)
            + plane.at((xi - 1) as usize, (yi - 1) as usize))
            * 0.25;
        let dxs = (above.at((xi + 1) as usize, yi as usize)
            - above.at((xi - 1) as usize, yi as usize)
            - below.at((xi + 1) as usize, yi as usize)
            + below.at((xi - 1) as usize, yi as usize))
            * 0.25;
        let dys = (above.at(xi as usize, (yi + 1) as usize)
            - above.at(xi as usize, (yi - 1) as usize)
            - below.at(xi as usize, (yi + 1) as usize)
            + below.at(xi as usize, (yi - 1) as usize))
            * 0.25;

        let det = dxx * (dyy * dss - dys * dys) - dxy * (dxy * dss - dys * dxs)
            + dxs * (dxy * dys - dyy * dxs);
        if !det.is_finite() || det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        let ox = -(inv
            * (gx * (dyy * dss - dys * dys) - dxy * (gy * dss - dys * gs)
                + dxs * (gy * dys - dyy * gs)));
        let oy = -(inv
            * (dxx * (gy * dss - dys * gs) - gx * (dxy * dss - dys * dxs)
                + dxs * (dxy * gs - gy * dxs)));
        let os = -(inv
            * (dxx * (dyy * gs - gy * dys) - dxy * (dxy * gs - gy * dxs)
                + gx * (dxy * dys - dyy * dxs)));

        if ox.abs() <= 0.5 && oy.abs() <= 0.5 && os.abs() <= 0.5 {
            let contrast = center + 0.5 * (gx * ox + gy * oy + gs * os);
            if contrast.abs() < contrast_threshold {
                return None;
            }
            return Some((xi as f32 + ox, yi as f32 + oy, li as f32 + os));
        }

        xi += ox.round() as i32;
        yi += oy.round() as i32;
        li += os.round() as i32;
    }
    None
}

fn orientation_histogram(plane: &Plane, x: i32, y: i32, sigma: f32) -> Vec<f32> {
    let radius = (3.0 * 1.5 * sigma).round().max(1.0) as i32;
    let mut histogram = [0.0f32; 36];
    let gaussian_weight = 2.0 * (1.5 * sigma) * (1.5 * sigma);

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let px = x + dx;
            let py = y + dy;
            if px <= 0 || py <= 0 || px >= plane.width as i32 - 1 || py >= plane.height as i32 - 1 {
                continue;
            }
            let gx =
                plane.at((px + 1) as usize, py as usize) - plane.at((px - 1) as usize, py as usize);
            let gy =
                plane.at(px as usize, (py + 1) as usize) - plane.at(px as usize, (py - 1) as usize);
            let magnitude = (gx * gx + gy * gy).sqrt();
            if magnitude <= 0.0 {
                continue;
            }
            let angle = gy.atan2(gx);
            let weight = (-((dx * dx + dy * dy) as f32) / gaussian_weight).exp();
            let position = (angle + PI) / TAU * 36.0;
            let bin = position.floor() as i32;
            let fraction = position - bin as f32;
            let b0 = bin.rem_euclid(36) as usize;
            let b1 = (b0 + 1) % 36;
            histogram[b0] += magnitude * weight * (1.0 - fraction);
            histogram[b1] += magnitude * weight * fraction;
        }
    }

    // Two passes of Lowe's smoothing kernel.
    for _ in 0..2 {
        let previous = histogram;
        for i in 0..36 {
            histogram[i] = (previous[(i + 35) % 36]
                + 4.0 * previous[i]
                + 6.0 * previous[(i + 1) % 36]
                + 4.0 * previous[(i + 2) % 36]
                + previous[(i + 3) % 36])
                / 16.0;
        }
    }

    let max = histogram.iter().copied().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return Vec::new();
    }
    let mut orientations = Vec::new();
    for i in 0..36 {
        let value = histogram[i];
        let previous = histogram[(i + 35) % 36];
        let next = histogram[(i + 1) % 36];
        if value > previous && value >= next && value >= 0.8 * max {
            let denominator = previous - 2.0 * value + next;
            let offset = if denominator.abs() > 1e-9 {
                (0.5 * (previous - next) / denominator).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            let angle = (i as f32 + 0.5 + offset) * TAU / 36.0 - PI;
            orientations.push(angle);
        }
    }
    orientations
}

fn describe(plane: &Plane, x: i32, y: i32, sigma: f32, orientation: f32) -> Option<Descriptor> {
    let bins = 4usize;
    let orientation_bins = 8usize;
    let hist_width = 3.0 * sigma;
    let radius = (hist_width * 2.0f32.sqrt() * (bins as f32 + 1.0) * 0.5).round() as i32;
    if radius < 1 {
        return None;
    }
    let cos = orientation.cos();
    let sin = orientation.sin();
    let mut histogram = [0.0f32; 128];

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let px = x + dx;
            let py = y + dy;
            if px <= 0 || py <= 0 || px >= plane.width as i32 - 1 || py >= plane.height as i32 - 1 {
                continue;
            }
            let rotated_col = (dx as f32 * cos + dy as f32 * sin) / hist_width;
            let rotated_row = (-dx as f32 * sin + dy as f32 * cos) / hist_width;
            let col_bin = rotated_col + bins as f32 * 0.5 - 0.5;
            let row_bin = rotated_row + bins as f32 * 0.5 - 0.5;
            if row_bin <= -1.0
                || row_bin >= bins as f32
                || col_bin <= -1.0
                || col_bin >= bins as f32
            {
                continue;
            }

            let gx =
                plane.at((px + 1) as usize, py as usize) - plane.at((px - 1) as usize, py as usize);
            let gy =
                plane.at(px as usize, (py + 1) as usize) - plane.at(px as usize, (py - 1) as usize);
            let magnitude = (gx * gx + gy * gy).sqrt();
            if magnitude <= 0.0 {
                continue;
            }
            let angle = gy.atan2(gx);
            let relative = (angle - orientation).rem_euclid(TAU);
            let orientation_bin = relative / TAU * orientation_bins as f32;
            if !orientation_bin.is_finite() {
                continue;
            }

            let row0 = row_bin.floor() as i32;
            let col0 = col_bin.floor() as i32;
            let ori0 = orientation_bin.floor() as i32;
            let fr = row_bin - row0 as f32;
            let fc = col_bin - col0 as f32;
            let fo = orientation_bin - ori0 as f32;

            for dr in 0..2 {
                let row = row0 + dr;
                if row < 0 || row >= bins as i32 {
                    continue;
                }
                let wr = if dr == 0 { 1.0 - fr } else { fr };
                for dc in 0..2 {
                    let col = col0 + dc;
                    if col < 0 || col >= bins as i32 {
                        continue;
                    }
                    let wc = if dc == 0 { 1.0 - fc } else { fc };
                    for dor in 0..2 {
                        let ori = (ori0 + dor).rem_euclid(orientation_bins as i32) as usize;
                        let wo = if dor == 0 { 1.0 - fo } else { fo };
                        let index = (row as usize * bins + col as usize) * orientation_bins + ori;
                        histogram[index] += magnitude * wr * wc * wo;
                    }
                }
            }
        }
    }

    let mut descriptor = Descriptor::from_array(histogram);
    if descriptor.normalize() <= f32::EPSILON {
        return None;
    }
    for value in &mut descriptor.values {
        *value = value.min(0.2);
    }
    descriptor.normalize();
    Some(descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homography::RansacHomographyEstimator;
    use crate::matcher::RatioMatcher;
    use stitch_domain::{
        FeatureMatcher, GrayImage, Homography, HomographyEstimator, RgbaImage, StitchOptions,
    };

    fn texture(width: u32, height: u32, seed: u64) -> RgbaImage {
        let mut state = seed;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / (1u64 << 53) as f32
        };
        // Low-frequency blobs plus fine structure gives SIFT stable features at
        // several scales.
        let mut blobs = Vec::new();
        for _ in 0..70 {
            blobs.push((
                random() * width as f32,
                random() * height as f32,
                6.0 + random() * 40.0,
                random(),
            ));
        }
        RgbaImage::from_fn(width, height, |x, y| {
            let mut value = 0.5;
            for (bx, by, radius, brightness) in &blobs {
                let dx = x as f32 - bx;
                let dy = y as f32 - by;
                let d = (dx * dx + dy * dy).sqrt();
                if d < *radius {
                    value += (brightness - 0.5) * (1.0 - d / radius) * 0.9;
                }
            }
            value += 0.08 * ((x as f32 * 0.37).sin() * (y as f32 * 0.29).cos());
            let v = (value.clamp(0.0, 1.0) * 255.0) as u8;
            [v, v, v, 255]
        })
    }

    fn warp(input: &RgbaImage, homography: &Homography, width: u32, height: u32) -> RgbaImage {
        let inverse = homography.inverse().unwrap();
        RgbaImage::from_fn(width, height, |x, y| {
            inverse
                .transform(stitch_domain::Point::new(x as f64, y as f64))
                .and_then(|p| input.sample_bilinear(p.x, p.y))
                .unwrap_or([20, 20, 20, 255])
        })
    }

    #[test]
    fn detects_and_normalizes_features() {
        let image = texture(320, 240, 7);
        let gray = image.to_gray();
        let features = SiftExtractor::default().extract(&gray).unwrap();
        assert!(features.len() > 80, "only {} features", features.len());
        for feature in features.iter() {
            assert!(feature.keypoint.x >= 0.0 && feature.keypoint.x < 320.0);
            assert!(feature.keypoint.y >= 0.0 && feature.keypoint.y < 240.0);
            assert!(feature.keypoint.scale > 0.0);
            assert!(feature.descriptor.is_normalized(1e-3));
        }
    }

    #[test]
    fn extraction_is_deterministic() {
        let gray = texture(200, 160, 11).to_gray();
        let extractor = SiftExtractor::default();
        let a = extractor.extract(&gray).unwrap();
        let b = extractor.extract(&gray).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn recovers_rotation_and_scale_transform() {
        let source = texture(360, 280, 3);
        let transform = Homography::translation(40.0, 30.0).then(&Homography::from_rows([
            [1.15, -0.22, 0.0],
            [0.22, 1.15, 0.0],
            [0.0, 0.0, 1.0],
        ]));
        let target = warp(&source, &transform, 360, 280);

        let extractor = SiftExtractor::default();
        let a = extractor.extract(&source.to_gray()).unwrap();
        let b = extractor.extract(&target.to_gray()).unwrap();
        let matches = RatioMatcher::default().match_features(&a, &b, 0.8);
        assert!(matches.len() > 20, "matches: {}", matches.len());

        // Matches map source->target; the estimator returns source->target.
        let estimate = RansacHomographyEstimator::default()
            .estimate(&a, &b, &matches, &StitchOptions::default())
            .unwrap();
        assert!(estimate.inliers.len() > 15);
        let expected = [
            stitch_domain::Point::new(30.0, 40.0),
            stitch_domain::Point::new(300.0, 60.0),
            stitch_domain::Point::new(250.0, 250.0),
        ];
        for point in expected {
            let actual = estimate.homography.transform(point).unwrap();
            let wanted = transform.transform(point).unwrap();
            assert!(
                actual.distance(wanted) < 4.0,
                "expected {wanted:?}, got {actual:?} (inliers {})",
                estimate.inliers.len()
            );
        }
    }

    #[test]
    fn too_small_images_are_rejected() {
        let image = GrayImage::from_fn(8, 8, |_, _| 0.5);
        assert!(SiftExtractor::default().extract(&image).is_err());
    }
}
