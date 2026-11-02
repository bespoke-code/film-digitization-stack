//! Brute-force nearest-neighbour matcher with Lowe's ratio test.

use rayon::prelude::*;
use stitch_domain::{FeatureMatch, FeatureMatcher, FeatureSet};

/// Ratio-test matcher.
///
/// A match is accepted when the best descriptor distance is below `ratio` times
/// the second-best distance. The inner loop abandons a candidate as soon as its
/// partial squared distance cannot beat the current runner-up, which keeps
/// brute force practical for a handful of multi-megapixel scans.
#[derive(Clone, Debug, Default)]
pub struct RatioMatcher {
    /// Reject matches whose absolute distance is larger than this (L2 in
    /// descriptor space). `None` disables the check.
    pub max_distance: Option<f32>,
}

impl RatioMatcher {
    pub fn with_max_distance(max_distance: f32) -> Self {
        Self {
            max_distance: Some(max_distance),
        }
    }
}

impl FeatureMatcher for RatioMatcher {
    fn match_features(
        &self,
        query: &FeatureSet,
        train: &FeatureSet,
        ratio: f32,
    ) -> Vec<FeatureMatch> {
        if query.is_empty() || train.len() < 2 {
            return Vec::new();
        }
        let ratio = ratio.clamp(0.1, 0.99);
        let max_squared = self.max_distance.map(|d| d * d);

        query
            .features
            .par_iter()
            .enumerate()
            .filter_map(|(query_index, feature)| {
                let mut best = f32::INFINITY;
                let mut second = f32::INFINITY;
                let mut best_index = 0usize;

                for (train_index, candidate) in train.features.iter().enumerate() {
                    let mut distance = 0.0f32;
                    let mut abandoned = false;
                    for (a, b) in feature
                        .descriptor
                        .values
                        .iter()
                        .zip(candidate.descriptor.values.iter())
                    {
                        let difference = a - b;
                        distance += difference * difference;
                        if distance >= second {
                            abandoned = true;
                            break;
                        }
                    }
                    if abandoned {
                        continue;
                    }
                    if distance < best {
                        second = best;
                        best = distance;
                        best_index = train_index;
                    } else if distance < second {
                        second = distance;
                    }
                }

                if !best.is_finite() || !second.is_finite() || second <= 0.0 {
                    return None;
                }
                if let Some(limit) = max_squared {
                    if best > limit {
                        return None;
                    }
                }
                let distance = best.sqrt();
                let runner_up = second.sqrt();
                if distance < ratio * runner_up {
                    Some(FeatureMatch::new(query_index, best_index, distance))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stitch_domain::{Descriptor, Feature, Keypoint};

    fn feature(value: f32, index: usize) -> Feature {
        Feature {
            keypoint: Keypoint {
                x: index as f32,
                y: index as f32,
                scale: 1.0,
                orientation: 0.0,
                response: 1.0,
            },
            descriptor: Descriptor::from_slice(&[value; 128]).unwrap(),
        }
    }

    #[test]
    fn identical_sets_match_one_to_one() {
        let a = FeatureSet::new((0..10).map(|i| feature(i as f32, i)).collect());
        let b = FeatureSet::new((0..10).map(|i| feature(i as f32, i)).collect());
        let matches = RatioMatcher::default().match_features(&a, &b, 0.8);
        assert_eq!(matches.len(), 10);
        for m in &matches {
            assert_eq!(m.query_index, m.train_index);
        }
    }

    #[test]
    fn ambiguous_descriptors_are_rejected() {
        let a = FeatureSet::new(vec![feature(1.0, 0)]);
        let b = FeatureSet::new(vec![feature(1.0, 0), feature(1.0, 1)]);
        assert!(RatioMatcher::default()
            .match_features(&a, &b, 0.8)
            .is_empty());
    }

    #[test]
    fn max_distance_filters_far_matches() {
        let a = FeatureSet::new(vec![feature(0.0, 0)]);
        let b = FeatureSet::new(vec![feature(1.0, 0), feature(2.0, 1)]);
        let matcher = RatioMatcher::with_max_distance(1.0);
        assert!(matcher.match_features(&a, &b, 0.99).is_empty());
    }

    #[test]
    fn empty_sets_are_safe() {
        let empty = FeatureSet::default();
        assert!(RatioMatcher::default()
            .match_features(&empty, &empty, 0.8)
            .is_empty());
    }
}
