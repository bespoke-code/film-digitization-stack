//! Color normalization between overlapping images.
//!
//! Camera scans of one negative can differ in exposure and white balance.
//! Matching the overlap to the registered target keeps seams invisible.

use stitch_domain::{ColorCurve, Homography, Point, RgbaImage};

const TARGET_SAMPLES_PER_SIDE: usize = 192;
const CALIBRATION_BINS: usize = 32;
const CALIBRATION_MIN_BIN_SAMPLES: u32 = 16;
/// Below this level the sample is more likely a film rebate or a clipped
/// shadow than usable tone. Those pixels must not steer a curve.
const CALIBRATION_DARK_FLOOR: u8 = 24;
const CALIBRATION_REGULARIZATION: f64 = 1e-2;
const CALIBRATION_KNOTS: usize = 17;
/// Fraction of the overlap evidence that the gain line is worth. The spline is
/// free to depart from the line only where the overlaps outweigh this prior.
const CALIBRATION_PRIOR: f64 = 0.3;
/// Largest deviation from the exposure line that a tone bin may have and still
/// steer the spline.
const CALIBRATION_OUTLIER: f64 = 24.0;
const MIN_GAIN: f64 = 0.5;
const MAX_GAIN: f64 = 2.0;

/// Solves one exposure gain per image and channel against every overlap.
///
/// All pairs constrain the same unknowns at once, so a gain cannot drift along
/// a chain of overlaps. The gain also anchors the tone curves: it is the part
/// of the correction that every overlap agrees on.
///
/// `overlaps` holds `(query, target, query_to_target)` triples. Image 0 stays
/// at unit gain and is the reference for the rest.
fn solve_gains(images: &[&RgbaImage], overlaps: &[(usize, usize, Homography)]) -> Vec<[f64; 3]> {
    let count = images.len();
    let mut gains = vec![[1.0f64; 3]; count];
    if count < 2 {
        return gains;
    }
    let unknowns = count - 1;

    for channel in 0..3 {
        let mut normal = vec![vec![0.0f64; unknowns]; unknowns];
        let mut rhs = vec![0.0f64; unknowns];

        for (query, target, homography) in overlaps {
            let (query, target) = (*query, *target);
            if query >= count || target >= count || query == target {
                continue;
            }
            for (value, mapped, weight) in
                overlap_tone_bins(images[query], images[target], homography, channel)
            {
                // The gain is multiplicative, so solve it in log space. Fitting
                // `gain * level` directly would let the brightest bins decide
                // the gain and leave the midtones dark.
                if value < 8.0 || mapped < 8.0 {
                    continue;
                }
                let mut terms = Vec::with_capacity(2);
                let mut equation_rhs = 0.0;
                for (image, sample, sign) in [(query, value, 1.0), (target, mapped, -1.0)] {
                    equation_rhs -= sign * sample.ln();
                    if image == 0 {
                        continue;
                    }
                    terms.push((image - 1, sign));
                }
                accumulate_equation(&mut normal, &mut rhs, &terms, equation_rhs, weight);
            }
        }

        let scale = (0..unknowns)
            .map(|index| normal[index][index])
            .fold(0.0f64, f64::max);
        if scale <= f64::EPSILON {
            continue;
        }
        // The prior pulls every log gain toward zero, which is unit gain. An
        // image without overlaps then keeps its exposure instead of drifting.
        let lambda = CALIBRATION_REGULARIZATION * scale;
        for (index, row) in normal.iter_mut().enumerate() {
            row[index] += lambda;
        }

        let solution = solve_linear(normal, rhs);
        for (image, gain) in gains.iter_mut().enumerate().skip(1) {
            gain[channel] = solution[image - 1].exp().clamp(MIN_GAIN, MAX_GAIN);
        }
    }

    gains
}

/// Solves one tone curve per image and channel against every overlap.
///
/// Each curve is a smooth spline that is pulled toward the global gain line.
/// The pull matters: a long-range pair can meet at a frame corner, where
/// vignetting looks exactly like an exposure difference, and a curve with no
/// prior would bend the whole frame to fit that corner and flatten its tones.
/// The gain line absorbs the exposure, and the spline only departs from it
/// where several overlaps agree.
pub fn calibrate_color_curves(
    images: &[&RgbaImage],
    overlaps: &[(usize, usize, Homography)],
) -> Vec<ColorCurve> {
    let count = images.len();
    let mut luts = vec![identity_lut(); count];
    if count < 2 {
        return luts.into_iter().map(ColorCurve::from_lut).collect();
    }
    let gains = solve_gains(images, overlaps);
    let unknowns = (count - 1) * CALIBRATION_KNOTS;

    let mut solutions: Vec<Option<Vec<f64>>> = Vec::with_capacity(3);
    for channel in 0..3 {
        let mut normal = vec![vec![0.0f64; unknowns]; unknowns];
        let mut rhs = vec![0.0f64; unknowns];

        for (query, target, homography) in overlaps {
            let (query, target) = (*query, *target);
            if query >= count || target >= count || query == target {
                continue;
            }
            for (value, mapped, weight) in
                overlap_tone_bins(images[query], images[target], homography, channel)
            {
                // A bin far from the exposure line is more likely a vignetted
                // corner or a strip of film rebate than a tone response, and
                // one such bin can bend a whole frame. Keep the fit near the
                // line unless the evidence is plausible.
                let predicted = mapped * gains[target][channel] / gains[query][channel];
                if (value - predicted).abs() > CALIBRATION_OUTLIER {
                    continue;
                }
                let mut terms = Vec::with_capacity(4);
                let mut equation_rhs = 0.0;
                for (image, sample, sign) in [(query, value, 1.0), (target, mapped, -1.0)] {
                    if image == 0 {
                        equation_rhs -= sign * sample;
                        continue;
                    }
                    let (knots, coefficients) = interpolation_weights(sample);
                    for (knot, coefficient) in knots.into_iter().zip(coefficients) {
                        let index = (image - 1) * CALIBRATION_KNOTS + knot;
                        terms.push((index, sign * coefficient));
                    }
                }
                accumulate_equation(&mut normal, &mut rhs, &terms, equation_rhs, weight);
            }
        }

        let scale = (0..unknowns)
            .map(|index| normal[index][index])
            .fold(0.0f64, f64::max);
        if scale <= f64::EPSILON {
            solutions.push(None);
            continue;
        }
        let prior_weight = (CALIBRATION_PRIOR * scale / CALIBRATION_KNOTS as f64).sqrt();
        for (image, image_gains) in gains.iter().enumerate().skip(1) {
            let base = (image - 1) * CALIBRATION_KNOTS;
            let gain = image_gains[channel];
            for knot in 0..CALIBRATION_KNOTS {
                accumulate_equation(
                    &mut normal,
                    &mut rhs,
                    &[(base + knot, 1.0)],
                    gain * knot_level(knot),
                    prior_weight,
                );
            }
        }

        solutions.push(Some(solve_linear(normal, rhs)));
    }

    for (image, lut) in luts.iter_mut().enumerate().skip(1) {
        for (channel, channel_lut) in lut.iter_mut().enumerate() {
            let Some(solution) = solutions[channel].as_ref() else {
                continue;
            };
            let knots = &solution[(image - 1) * CALIBRATION_KNOTS..image * CALIBRATION_KNOTS];
            write_curve(channel_lut, knots);
        }
    }

    luts.into_iter().map(ColorCurve::from_lut).collect()
}

fn identity_lut() -> [[u8; 256]; 3] {
    let mut lut = [[0u8; 256]; 3];
    for channel in &mut lut {
        for (level, entry) in channel.iter_mut().enumerate() {
            *entry = level as u8;
        }
    }
    lut
}

fn knot_level(knot: usize) -> f64 {
    knot as f64 * 255.0 / (CALIBRATION_KNOTS - 1) as f64
}

/// Linear interpolation weights over the calibration knot grid.
fn interpolation_weights(value: f64) -> ([usize; 2], [f64; 2]) {
    let scaled = value.clamp(0.0, 255.0) * (CALIBRATION_KNOTS - 1) as f64 / 255.0;
    let lower = (scaled.floor() as usize).min(CALIBRATION_KNOTS - 2);
    let fraction = scaled - lower as f64;
    ([lower, lower + 1], [1.0 - fraction, fraction])
}

fn evaluate_knots(knots: &[f64], value: f64) -> f64 {
    let (indices, weights) = interpolation_weights(value);
    knots[indices[0]] * weights[0] + knots[indices[1]] * weights[1]
}

fn write_curve(lut: &mut [u8; 256], knots: &[f64]) {
    let mut previous = 0u8;
    for (level, entry) in lut.iter_mut().enumerate() {
        let value = evaluate_knots(knots, level as f64);
        let low = MIN_GAIN * level as f64;
        let high = MAX_GAIN * level as f64;
        let mut mapped = value.clamp(low, high).round().clamp(0.0, 255.0) as u8;
        if level > 0 {
            mapped = mapped.max(previous);
        }
        *entry = mapped;
        previous = mapped;
    }
}

/// Bins the overlap by query level and returns `(query_level, target_median,
/// weight)` points for one channel. Medians keep dust and registration noise
/// from steering the calibration.
fn overlap_tone_bins(
    query: &RgbaImage,
    target: &RgbaImage,
    query_to_target: &Homography,
    channel: usize,
) -> Vec<(f64, f64, f64)> {
    let longest = query.width().max(query.height()) as usize;
    let step = (longest / TARGET_SAMPLES_PER_SIDE).max(1) as u32;
    let max_x = target.width().saturating_sub(1) as f64;
    let max_y = target.height().saturating_sub(1) as f64;
    let mut histogram = vec![[0u32; 256]; CALIBRATION_BINS];
    let mut counts = [0u32; CALIBRATION_BINS];

    let mut y = 0;
    while y < query.height() {
        let mut x = 0;
        while x < query.width() {
            if let Some(point) = query_to_target.transform(Point::new(x as f64, y as f64)) {
                if point.x >= 1.0 && point.y >= 1.0 && point.x < max_x && point.y < max_y {
                    let from = query.pixel(x, y);
                    let to = target.pixel(point.x.round() as u32, point.y.round() as u32);
                    if from[3] > 0
                        && to[3] > 0
                        && from[channel] >= CALIBRATION_DARK_FLOOR
                        && to[channel] >= CALIBRATION_DARK_FLOOR
                    {
                        let bin = from[channel] as usize * CALIBRATION_BINS / 256;
                        histogram[bin][to[channel] as usize] += 1;
                        counts[bin] += 1;
                    }
                }
            }
            x += step;
        }
        y += step;
    }

    let mut points = Vec::with_capacity(CALIBRATION_BINS);
    for bin in 0..CALIBRATION_BINS {
        if counts[bin] < CALIBRATION_MIN_BIN_SAMPLES {
            continue;
        }
        let Some(median) = histogram_median(&histogram[bin], counts[bin]) else {
            continue;
        };
        let center = (bin as f64 + 0.5) * 256.0 / CALIBRATION_BINS as f64;
        points.push((center, median as f64, (counts[bin] as f64).sqrt()));
    }
    points
}

fn histogram_median(histogram: &[u32; 256], total: u32) -> Option<u8> {
    let mut seen = 0u32;
    let halfway = total / 2;
    for (level, count) in histogram.iter().copied().enumerate() {
        seen += count;
        if seen > halfway {
            return Some(level as u8);
        }
    }
    None
}

fn accumulate_equation(
    normal: &mut [Vec<f64>],
    rhs: &mut [f64],
    terms: &[(usize, f64)],
    equation_rhs: f64,
    weight: f64,
) {
    let squared = weight * weight;
    for &(row, row_coefficient) in terms {
        rhs[row] += squared * row_coefficient * equation_rhs;
        for &(column, column_coefficient) in terms {
            normal[row][column] += squared * row_coefficient * column_coefficient;
        }
    }
}

/// Solves `matrix * x = rhs` with partial pivoting.
fn solve_linear(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let size = rhs.len();
    for column in 0..size {
        let mut pivot = column;
        for row in column + 1..size {
            if matrix[row][column].abs() > matrix[pivot][column].abs() {
                pivot = row;
            }
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let divisor = matrix[column][column];
        if divisor.abs() < 1e-12 {
            continue;
        }
        for entry in &mut matrix[column][column..] {
            *entry /= divisor;
        }
        rhs[column] /= divisor;
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor == 0.0 {
                continue;
            }
            for index in column..size {
                matrix[row][index] -= factor * matrix[column][index];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    rhs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(width: u32, height: u32) -> RgbaImage {
        RgbaImage::from_fn(width, height, |x, _| {
            let level = (x * 255 / (width - 1)) as u8;
            [level, level, level, 255]
        })
    }

    fn scaled(source: &RgbaImage, gain: [f64; 3]) -> RgbaImage {
        RgbaImage::from_fn(source.width(), source.height(), |x, y| {
            let pixel = source.pixel(x, y);
            let mut out = [0u8; 4];
            for channel in 0..3 {
                out[channel] = (pixel[channel] as f64 * gain[channel])
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
            out[3] = pixel[3];
            out
        })
    }

    /// Mean absolute difference between the two calibrated images over the
    /// whole overlap.
    fn residual(images: &[&RgbaImage], curves: &[ColorCurve]) -> f64 {
        let mut total = 0.0;
        let mut samples = 0.0;
        for (pixel_a, pixel_b) in images[0]
            .pixels()
            .chunks(4)
            .zip(images[1].pixels().chunks(4))
        {
            let a = curves[0].map([pixel_a[0], pixel_a[1], pixel_a[2]]);
            let b = curves[1].map([pixel_b[0], pixel_b[1], pixel_b[2]]);
            for channel in 0..3 {
                total += (a[channel] as f64 - b[channel] as f64).abs();
                samples += 1.0;
            }
        }
        total / samples
    }

    #[test]
    fn calibration_matches_a_linear_exposure_difference() {
        let target = gradient(128, 96);
        let darker = scaled(&target, [180.0 / 255.0; 3]);
        let images = [&target, &darker];
        let overlaps = [(1, 0, Homography::identity())];
        let curves = calibrate_color_curves(&images, &overlaps);
        assert!(curves[0].is_identity());
        let before = residual(&images, &[ColorCurve::identity(), ColorCurve::identity()]);
        let after = residual(&images, &curves);
        assert!(before > 20.0, "fixture is too flat: {before:.2}");
        assert!(after < 3.0, "calibrated residual was {after:.2}");
    }

    #[test]
    fn calibration_matches_white_balance_per_channel() {
        let target = gradient(128, 96);
        let tinted = scaled(&target, [0.6, 0.8, 0.9]);
        let images = [&target, &tinted];
        let overlaps = [(1, 0, Homography::identity())];
        let curves = calibrate_color_curves(&images, &overlaps);
        let mapped = curves[1].map([77, 102, 115]);
        for (channel, value) in mapped.iter().enumerate() {
            assert!(
                (*value as i32 - 128).abs() <= 4,
                "channel {channel} mapped to {value}"
            );
        }
    }

    #[test]
    fn calibration_keeps_the_tone_response_near_the_gain() {
        let target = gradient(128, 96);
        let darker = scaled(&target, [0.7, 0.7, 0.7]);
        let images = [&target, &darker];
        let overlaps = [(1, 0, Homography::identity())];
        let curves = calibrate_color_curves(&images, &overlaps);
        let lut = curves[1].lut();
        let gain = 1.0 / 0.7;
        let mut close = 0;
        let mut compared = 0;
        for (level, actual) in lut[0].iter().enumerate().take(249).skip(8) {
            let ideal = level as f64 * gain;
            if ideal > 250.0 {
                continue;
            }
            compared += 1;
            if (*actual as f64 - ideal).abs() <= 6.0 {
                close += 1;
            }
        }
        assert!(
            close * 100 >= compared * 90,
            "only {close} of {compared} levels followed the exposure gain"
        );
    }

    #[test]
    fn a_single_image_stays_identity() {
        let image = gradient(16, 16);
        let curves = calibrate_color_curves(&[&image], &[]);
        assert_eq!(curves.len(), 1);
        assert!(curves[0].is_identity());
    }

    #[test]
    fn disconnected_images_stay_identity() {
        let first = gradient(32, 32);
        let second = scaled(&first, [0.4, 0.4, 0.4]);
        let curves = calibrate_color_curves(&[&first, &second], &[]);
        assert!(curves[0].is_identity());
        assert!(curves[1].is_identity());
    }

    #[test]
    fn an_image_without_overlaps_keeps_its_tones() {
        // A rough-arranged, featureless frame has no overlap equations. It
        // must not inherit a gain from the regularization.
        let target = gradient(128, 96);
        let darker = scaled(&target, [0.8, 0.8, 0.8]);
        let isolated = gradient(64, 64);
        let images = [&target, &darker, &isolated];
        let overlaps = [(1, 0, Homography::identity())];
        let gains = solve_gains(&images, &overlaps);
        for channel in 0..3 {
            assert!(
                (gains[2][channel] - 1.0).abs() < 1e-6,
                "isolated gain {:?}",
                gains[2]
            );
            assert!(
                (gains[1][channel] - 1.25).abs() < 0.05,
                "connected gain {:?}",
                gains[1]
            );
        }
        let curves = calibrate_color_curves(&images, &overlaps);
        for level in [16u8, 64, 128, 200, 250] {
            let mapped = curves[2].map([level; 3]);
            assert!(
                mapped.iter().all(|value| value.abs_diff(level) <= 1),
                "level {level} mapped to {mapped:?}"
            );
        }
    }

    #[test]
    fn extreme_differences_are_clamped() {
        let target = gradient(64, 64);
        let dark = scaled(&target, [0.05, 0.05, 0.05]);
        let images = [&target, &dark];
        let overlaps = [(1, 0, Homography::identity())];
        let curves = calibrate_color_curves(&images, &overlaps);
        let lut = curves[1].lut();
        assert!(lut[0][64] <= (64.0 * MAX_GAIN) as u8 + 1);
    }
}
