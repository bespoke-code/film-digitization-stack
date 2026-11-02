//! RANSAC + normalized DLT homography estimation.

use std::collections::HashSet;

use nalgebra::DMatrix;
use stitch_domain::{
    CoreError, CoreResult, FeatureMatch, FeatureSet, Homography, HomographyEstimate,
    HomographyEstimator, Point, StitchOptions,
};

/// Robust estimator using a four-point minimal DLT inside RANSAC and a
/// least-squares refinement on the final inlier set.
#[derive(Clone, Debug)]
pub struct RansacHomographyEstimator {
    /// Seed for the deterministic pseudo-random sample order.
    pub seed: u64,
}

impl Default for RansacHomographyEstimator {
    fn default() -> Self {
        Self { seed: 0x5EED_51F7 }
    }
}

impl RansacHomographyEstimator {
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl HomographyEstimator for RansacHomographyEstimator {
    fn estimate(
        &self,
        query: &FeatureSet,
        train: &FeatureSet,
        matches: &[FeatureMatch],
        options: &StitchOptions,
    ) -> CoreResult<HomographyEstimate> {
        let mut source = Vec::with_capacity(matches.len());
        let mut destination = Vec::with_capacity(matches.len());
        for m in matches {
            let Some(query_feature) = query.features.get(m.query_index) else {
                continue;
            };
            let Some(train_feature) = train.features.get(m.train_index) else {
                continue;
            };
            source.push(Point::new(
                query_feature.keypoint.x as f64,
                query_feature.keypoint.y as f64,
            ));
            destination.push(Point::new(
                train_feature.keypoint.x as f64,
                train_feature.keypoint.y as f64,
            ));
        }

        if source.len() < 4 {
            return Err(CoreError::Homography(format!(
                "need at least 4 correspondences, got {}",
                source.len()
            )));
        }

        let (normalized_source, source_norm) = normalize_points(&source);
        let (normalized_destination, destination_norm) = normalize_points(&destination);
        let threshold = options.ransac_threshold.max(0.5);
        let threshold_squared = threshold * threshold;
        let sample_count = source.len();
        let iterations = options.ransac_iterations.max(100);

        let mut rng = SplitMix64::new(self.seed);
        let mut best_inliers: Vec<usize> = Vec::new();
        let mut best_homography: Option<Homography> = None;
        let mut required_iterations = iterations;
        let mut iteration = 0usize;

        while iteration < required_iterations {
            iteration += 1;
            let sample = rng.sample_distinct(4, sample_count);
            if sample.len() != 4 {
                continue;
            }
            if is_degenerate(&sample, &normalized_source)
                || is_degenerate(&sample, &normalized_destination)
            {
                continue;
            }

            let src: Vec<Point> = sample.iter().map(|&i| normalized_source[i]).collect();
            let dst: Vec<Point> = sample.iter().map(|&i| normalized_destination[i]).collect();
            let Some(h_normalized) = solve_dlt(&src, &dst) else {
                continue;
            };
            let Some(homography) = denormalize(&h_normalized, &source_norm, &destination_norm)
            else {
                continue;
            };

            let inliers = collect_inliers(&homography, &source, &destination, threshold_squared);
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                best_homography = Some(homography);

                let ratio = best_inliers.len() as f64 / sample_count as f64;
                if ratio > 0.05 {
                    let success = 0.999f64;
                    let probability = ratio.powi(4);
                    if probability < 1.0 {
                        let estimate =
                            ((1.0 - success).ln() / (1.0 - probability).ln()).ceil() as usize;
                        required_iterations = required_iterations.min(estimate.max(100)).max(50);
                    }
                }
            }
        }

        let mut homography = best_homography.ok_or_else(|| {
            CoreError::Homography("RANSAC did not find a valid transform".to_owned())
        })?;
        if best_inliers.len() < 4 {
            return Err(CoreError::Homography(
                "RANSAC found no geometrically consistent set".to_owned(),
            ));
        }

        // Refine twice on the inlier set and reclassify.
        for _ in 0..2 {
            let src: Vec<Point> = best_inliers.iter().map(|&i| normalized_source[i]).collect();
            let dst: Vec<Point> = best_inliers
                .iter()
                .map(|&i| normalized_destination[i])
                .collect();
            let Some(h_normalized) = solve_dlt(&src, &dst) else {
                break;
            };
            let Some(refined) = denormalize(&h_normalized, &source_norm, &destination_norm) else {
                break;
            };
            let inliers = collect_inliers(&refined, &source, &destination, threshold_squared);
            if inliers.len() < 4 {
                break;
            }
            homography = refined;
            if inliers.len() == best_inliers.len() {
                best_inliers = inliers;
                break;
            }
            best_inliers = inliers;
        }

        if best_inliers.len() < options.min_inliers {
            return Err(CoreError::Homography(format!(
                "only {} inliers (minimum {})",
                best_inliers.len(),
                options.min_inliers
            )));
        }
        if !homography.is_finite() || homography.inverse().is_none() {
            return Err(CoreError::Homography(
                "estimated homography is not invertible".to_owned(),
            ));
        }

        let mut squared_error = 0.0;
        for &index in &best_inliers {
            if let Some(projected) = homography.transform(source[index]) {
                let error = projected.distance(destination[index]);
                squared_error += error * error;
            }
        }
        let rms_error = (squared_error / best_inliers.len() as f64).sqrt();
        let num_matches = matches.len();

        Ok(HomographyEstimate {
            homography,
            inliers: best_inliers,
            rms_error,
            num_matches,
        })
    }
}

fn solve_dlt(source: &[Point], destination: &[Point]) -> Option<Homography> {
    if source.len() != destination.len() || source.len() < 4 {
        return None;
    }
    // Pad to at least 9 rows so the thin SVD exposes all nine right singular
    // vectors. With only four correspondences the 8x9 system has a
    // one-dimensional nullspace that would otherwise be missing from `v_t`.
    let rows = (source.len() * 2).max(9);
    let mut matrix = DMatrix::<f64>::zeros(rows, 9);
    for (index, (src, dst)) in source.iter().zip(destination).enumerate() {
        let row = index * 2;
        matrix[(row, 0)] = -src.x;
        matrix[(row, 1)] = -src.y;
        matrix[(row, 2)] = -1.0;
        matrix[(row, 6)] = dst.x * src.x;
        matrix[(row, 7)] = dst.x * src.y;
        matrix[(row, 8)] = dst.x;

        matrix[(row + 1, 3)] = -src.x;
        matrix[(row + 1, 4)] = -src.y;
        matrix[(row + 1, 5)] = -1.0;
        matrix[(row + 1, 6)] = dst.y * src.x;
        matrix[(row + 1, 7)] = dst.y * src.y;
        matrix[(row + 1, 8)] = dst.y;
    }

    let svd = matrix.svd(true, true);
    let v_t = svd.v_t?;
    if v_t.ncols() < 9 {
        return None;
    }
    let row = v_t.row(v_t.nrows() - 1);
    let mut values = [0.0f64; 9];
    for (index, value) in row.iter().enumerate() {
        values[index] = *value;
    }
    let homography = Homography::from_slice(&values);
    if !homography.is_finite() || homography.determinant().abs() < 1e-12 {
        return None;
    }
    Some(homography)
}

/// Hartley normalization: translate the centroid to the origin and scale the
/// mean distance to sqrt(2).
fn normalize_points(points: &[Point]) -> (Vec<Point>, Homography) {
    let count = points.len().max(1) as f64;
    let cx = points.iter().map(|p| p.x).sum::<f64>() / count;
    let cy = points.iter().map(|p| p.y).sum::<f64>() / count;
    let mean_distance = points
        .iter()
        .map(|p| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt())
        .sum::<f64>()
        / count;
    let scale = if mean_distance > 1e-9 {
        2.0f64.sqrt() / mean_distance
    } else {
        1.0
    };
    let transform = Homography::from_rows([
        [scale, 0.0, -scale * cx],
        [0.0, scale, -scale * cy],
        [0.0, 0.0, 1.0],
    ]);
    let normalized = points
        .iter()
        .map(|&point| transform.transform(point).unwrap_or(point))
        .collect();
    (normalized, transform)
}

fn denormalize(
    normalized: &Homography,
    source_norm: &Homography,
    destination_norm: &Homography,
) -> Option<Homography> {
    let destination_inverse = destination_norm.inverse()?;
    let homography = destination_inverse.then(normalized).then(source_norm);
    if homography.is_finite() {
        Some(homography)
    } else {
        None
    }
}

fn collect_inliers(
    homography: &Homography,
    source: &[Point],
    destination: &[Point],
    threshold_squared: f64,
) -> Vec<usize> {
    let mut inliers = Vec::new();
    for index in 0..source.len() {
        let Some(projected) = homography.transform(source[index]) else {
            continue;
        };
        let dx = projected.x - destination[index].x;
        let dy = projected.y - destination[index].y;
        if dx * dx + dy * dy <= threshold_squared {
            inliers.push(index);
        }
    }
    inliers
}

fn is_degenerate(sample: &[usize], points: &[Point]) -> bool {
    for i in 0..sample.len() {
        for j in (i + 1)..sample.len() {
            for k in (j + 1)..sample.len() {
                let a = points[sample[i]];
                let b = points[sample[j]];
                let c = points[sample[k]];
                let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
                if area.abs() < 1e-8 {
                    return true;
                }
            }
        }
    }
    false
}

/// Deterministic SplitMix64 with the tiny helpers RANSAC needs.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn sample_distinct(&mut self, count: usize, range: usize) -> Vec<usize> {
        if count > range {
            return Vec::new();
        }
        let mut seen = HashSet::with_capacity(count);
        let mut result = Vec::with_capacity(count);
        let mut attempts = 0;
        while result.len() < count && attempts < range * 8 {
            attempts += 1;
            let index = (self.next_u64() % range as u64) as usize;
            if seen.insert(index) {
                result.push(index);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stitch_domain::{Descriptor, Feature, Keypoint, StitchOptions};

    fn feature_set(points: &[Point]) -> FeatureSet {
        FeatureSet::new(
            points
                .iter()
                .map(|point| Feature {
                    keypoint: Keypoint {
                        x: point.x as f32,
                        y: point.y as f32,
                        scale: 2.0,
                        orientation: 0.0,
                        response: 1.0,
                    },
                    descriptor: Descriptor::from_slice(&[0.1; 128]).unwrap(),
                })
                .collect(),
        )
    }

    fn matches(count: usize) -> Vec<FeatureMatch> {
        (0..count).map(|i| FeatureMatch::new(i, i, 1.0)).collect()
    }

    #[test]
    fn recovers_known_transform_with_outliers() {
        let transform = Homography::from_rows([
            [1.1, -0.15, 25.0],
            [0.12, 1.05, -10.0],
            [0.0002, -0.0001, 1.0],
        ]);
        let mut source_points = Vec::new();
        let mut destination_points = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                let point = Point::new(30.0 + x as f64 * 24.0, 20.0 + y as f64 * 21.0);
                source_points.push(point);
                destination_points.push(transform.transform(point).unwrap());
            }
        }
        // Corrupt 20% of the correspondences.
        for index in (0..destination_points.len()).step_by(5) {
            destination_points[index].x += 90.0;
            destination_points[index].y -= 70.0;
        }

        let query = FeatureSet::new(
            source_points
                .iter()
                .map(|point| Feature {
                    keypoint: Keypoint {
                        x: point.x as f32,
                        y: point.y as f32,
                        scale: 2.0,
                        orientation: 0.0,
                        response: 1.0,
                    },
                    descriptor: Descriptor::from_slice(&[0.1; 128]).unwrap(),
                })
                .collect(),
        );
        let train = FeatureSet::new(
            destination_points
                .iter()
                .map(|point| Feature {
                    keypoint: Keypoint {
                        x: point.x as f32,
                        y: point.y as f32,
                        scale: 2.0,
                        orientation: 0.0,
                        response: 1.0,
                    },
                    descriptor: Descriptor::from_slice(&[0.1; 128]).unwrap(),
                })
                .collect(),
        );
        let all_matches: Vec<FeatureMatch> = (0..source_points.len())
            .map(|i| FeatureMatch::new(i, i, 1.0))
            .collect();
        let options = StitchOptions {
            min_inliers: 20,
            ..StitchOptions::default()
        };
        let estimate = RansacHomographyEstimator::default()
            .estimate(&query, &train, &all_matches, &options)
            .unwrap();
        assert!(estimate.inliers.len() > 50);
        let probe = Point::new(80.0, 60.0);
        let actual = estimate.homography.transform(probe).unwrap();
        let expected = transform.transform(probe).unwrap();
        assert!(actual.distance(expected) < 1.0);
        assert!(estimate.rms_error < 1.0);
    }

    #[test]
    fn rejects_too_few_points() {
        let set = feature_set(&[Point::new(0.0, 0.0), Point::new(1.0, 0.0)]);
        let error = RansacHomographyEstimator::default()
            .estimate(&set, &set, &matches(2), &StitchOptions::default())
            .unwrap_err();
        assert!(matches!(error, CoreError::Homography(_)));
    }

    #[test]
    fn rejects_random_correspondences() {
        // Use a deterministic PRNG instead of modular arithmetic: linear
        // congruential sequences can accidentally satisfy an exact
        // homography, which would make this test assert the wrong thing.
        let mut rng = SplitMix64::new(0xC0FF_EE12);
        let mut random_point = |scale_x: f64, scale_y: f64| {
            let x = (rng.next_u64() % 100_000) as f64 / 100_000.0;
            let y = (rng.next_u64() % 100_000) as f64 / 100_000.0;
            Point::new(x * scale_x, y * scale_y)
        };
        let source: Vec<Point> = (0..60).map(|_| random_point(311.0, 271.0)).collect();
        let destination: Vec<Point> = (0..60)
            .map(|_| {
                let point = random_point(307.0, 263.0);
                Point::new(point.x + 500.0, point.y)
            })
            .collect();
        let query = feature_set(&source);
        let train = feature_set(&destination);
        let options = StitchOptions {
            min_inliers: 6,
            ransac_threshold: 1.0,
            ..StitchOptions::default()
        };
        assert!(RansacHomographyEstimator::default()
            .estimate(&query, &train, &matches(60), &options)
            .is_err());
    }

    #[test]
    fn sample_contains_distinct_indices() {
        let mut rng = SplitMix64::new(3);
        for _ in 0..100 {
            let sample = rng.sample_distinct(4, 10);
            let set: HashSet<_> = sample.iter().copied().collect();
            assert_eq!(sample.len(), 4);
            assert_eq!(set.len(), 4);
        }
    }
}
