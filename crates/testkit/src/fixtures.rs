//! Deterministic synthetic images used by contract and end-to-end tests.

use stitch_domain::{
    Descriptor, Feature, FeatureSet, GrayImage, Homography, Keypoint, Point, RgbaImage,
};

pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// A photo-like texture: smooth blobs at several scales plus fine sine detail.
pub fn textured_rgba(width: u32, height: u32, seed: u64) -> RgbaImage {
    let mut rng = SplitMix64::new(seed);
    let mut blobs = Vec::new();
    for _ in 0..90 {
        blobs.push((
            rng.next_f32() * width as f32,
            rng.next_f32() * height as f32,
            4.0 + rng.next_f32() * 40.0,
            rng.next_f32(),
        ));
    }
    RgbaImage::from_fn(width, height, |x, y| {
        let mut value = 0.5;
        for (bx, by, radius, brightness) in &blobs {
            let dx = x as f32 - bx;
            let dy = y as f32 - by;
            let distance = (dx * dx + dy * dy).sqrt();
            if distance < *radius {
                value += (brightness - 0.5) * (1.0 - distance / radius);
            }
        }
        value += 0.09 * ((x as f32 * 0.41).sin() * (y as f32 * 0.33).cos());
        value += 0.04 * ((x as f32 * 1.13).cos() + (y as f32 * 0.97).sin());
        let v = (value.clamp(0.0, 1.0) * 255.0) as u8;
        [v, v, v, 255]
    })
}

pub fn textured_gray(width: u32, height: u32, seed: u64) -> GrayImage {
    textured_rgba(width, height, seed).to_gray()
}

pub fn solid_rgba(width: u32, height: u32, color: [u8; 4]) -> RgbaImage {
    RgbaImage::filled(width, height, color)
}

/// Warps `input` into a new `width`x`height` image through `homography`
/// (source -> output). Uncovered pixels use `background`.
pub fn warp_rgba(
    input: &RgbaImage,
    homography: &Homography,
    width: u32,
    height: u32,
    background: [u8; 4],
) -> RgbaImage {
    let inverse = homography
        .inverse()
        .expect("fixture transform must be invertible");
    RgbaImage::from_fn(width, height, |x, y| {
        inverse
            .transform(Point::new(x as f64, y as f64))
            .and_then(|p| input.sample_bilinear(p.x, p.y))
            .unwrap_or(background)
    })
}

/// Builds a descriptor-filled feature set at the given points. Descriptors are
/// deterministic and unit length.
pub fn features_at_points(points: &[Point], seed: u64) -> FeatureSet {
    let mut rng = SplitMix64::new(seed);
    let features = points
        .iter()
        .map(|point| {
            let mut values = [0.0f32; 128];
            for value in &mut values {
                *value = rng.next_f32() * 2.0 - 1.0;
            }
            let mut descriptor = Descriptor::from_array(values);
            descriptor.normalize();
            Feature {
                keypoint: Keypoint {
                    x: point.x as f32,
                    y: point.y as f32,
                    scale: 2.0,
                    orientation: 0.0,
                    response: 1.0,
                },
                descriptor,
            }
        })
        .collect();
    FeatureSet::new(features)
}

pub fn grid_points(columns: usize, rows: usize, spacing: f64) -> Vec<Point> {
    let mut points = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        for column in 0..columns {
            points.push(Point::new(
                20.0 + column as f64 * spacing,
                20.0 + row as f64 * spacing,
            ));
        }
    }
    points
}
