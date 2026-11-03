//! Color normalization between overlapping images.
//!
//! Camera scans of one negative can differ in exposure and white balance.
//! Matching the overlap to the registered target keeps seams invisible.

use stitch_domain::{Homography, Point, RgbaImage};

const MIN_SAMPLES: usize = 48;
const MIN_GAIN: f32 = 0.5;
const MAX_GAIN: f32 = 2.0;
const TARGET_SAMPLES_PER_SIDE: usize = 192;

/// Estimates a per-channel multiplier that brings `query` toward `target`.
///
/// The transform maps query coordinates into target coordinates. The function
/// samples a sparse grid, compares channel medians in the overlap and clamps
/// the result so a bad overlap cannot destroy the image.
pub fn estimate_gain(
    query: &RgbaImage,
    target: &RgbaImage,
    query_to_target: &Homography,
) -> [f32; 3] {
    let longest = query.width().max(query.height()) as usize;
    let step = (longest / TARGET_SAMPLES_PER_SIDE).max(1) as u32;
    let max_x = target.width().saturating_sub(1) as f64;
    let max_y = target.height().saturating_sub(1) as f64;

    let mut query_values: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut target_values: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut samples = 0usize;

    let mut y = 0;
    while y < query.height() {
        let mut x = 0;
        while x < query.width() {
            if let Some(point) = query_to_target.transform(Point::new(x as f64, y as f64)) {
                if point.x >= 1.0 && point.y >= 1.0 && point.x < max_x && point.y < max_y {
                    let target_x = point.x.round() as u32;
                    let target_y = point.y.round() as u32;
                    let from = query.pixel(x, y);
                    let to = target.pixel(target_x, target_y);
                    if from[3] > 0 && to[3] > 0 {
                        for channel in 0..3 {
                            query_values[channel].push(from[channel]);
                            target_values[channel].push(to[channel]);
                        }
                        samples += 1;
                    }
                }
            }
            x += step;
        }
        y += step;
    }

    if samples < MIN_SAMPLES {
        return [1.0; 3];
    }

    let mut gain = [1.0; 3];
    for channel in 0..3 {
        let from = median(&mut query_values[channel]);
        let to = median(&mut target_values[channel]);
        if from > 2.0 && to > 2.0 {
            gain[channel] = (to / from).clamp(MIN_GAIN, MAX_GAIN);
        }
    }
    gain
}

fn median(values: &mut [u8]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable();
    values[values.len() / 2] as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_a_linear_brightness_gain() {
        let query = RgbaImage::filled(64, 64, [100, 60, 30, 255]);
        let target = RgbaImage::filled(64, 64, [200, 120, 60, 255]);
        let gain = estimate_gain(&query, &target, &Homography::identity());
        assert!((gain[0] - 2.0).abs() < 0.01);
        assert!((gain[1] - 2.0).abs() < 0.01);
        assert!((gain[2] - 2.0).abs() < 0.01);
    }

    #[test]
    fn estimates_white_balance_per_channel() {
        let query = RgbaImage::filled(64, 64, [120, 100, 80, 255]);
        let target = RgbaImage::filled(64, 64, [120, 120, 120, 255]);
        let gain = estimate_gain(&query, &target, &Homography::identity());
        assert!(gain[0] < 1.05);
        assert!((gain[1] - 1.2).abs() < 0.01);
        assert!((gain[2] - 1.5).abs() < 0.01);
    }

    #[test]
    fn no_overlap_leaves_the_image_unchanged() {
        let query = RgbaImage::filled(64, 64, [100, 100, 100, 255]);
        let target = RgbaImage::filled(64, 64, [200, 200, 200, 255]);
        let gain = estimate_gain(&query, &target, &Homography::translation(1000.0, 0.0));
        assert_eq!(gain, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn extreme_gains_are_clamped() {
        let query = RgbaImage::filled(64, 64, [10, 10, 10, 255]);
        let target = RgbaImage::filled(64, 64, [250, 250, 250, 255]);
        let gain = estimate_gain(&query, &target, &Homography::identity());
        assert_eq!(gain, [MAX_GAIN; 3]);
    }
}
