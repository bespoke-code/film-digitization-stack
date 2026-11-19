//! Brute-force nearest-neighbour matcher with Lowe's ratio test.

use rayon::prelude::*;
use stitch_domain::{FeatureMatch, FeatureMatcher, FeatureSet};

/// Ratio-test matcher.
///
/// A match is accepted when the best descriptor distance is below `ratio` times
/// the second-best distance. The inner loop abandons a candidate as soon as a
/// block of its partial squared distance cannot beat the current runner-up,
/// which keeps brute force practical for a handful of multi-megapixel scans.
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

/// Squared distances to the nearest and second-nearest train descriptors.
struct Neighbours {
    best_index: usize,
    best: f32,
    second: f32,
}

/// Descriptor values summed per block before the abandon test. Independent
/// lanes let the compiler use SIMD for the squared differences.
const BLOCK: usize = 32;
const LANES: usize = 8;

impl RatioMatcher {
    fn neighbours(query: &FeatureSet, train: &FeatureSet) -> Vec<Option<Neighbours>> {
        query
            .features
            .par_iter()
            .map(|feature| {
                let mut best = f32::INFINITY;
                let mut second = f32::INFINITY;
                let mut best_index = 0usize;

                'candidates: for (train_index, candidate) in train.features.iter().enumerate() {
                    let mut distance = 0.0f32;
                    let blocks = feature.descriptor.values.as_chunks::<BLOCK>().0;
                    let candidate_blocks = candidate.descriptor.values.as_chunks::<BLOCK>().0;
                    for (a_block, b_block) in blocks.iter().zip(candidate_blocks) {
                        let mut lanes = [0.0f32; LANES];
                        for (a, b) in a_block
                            .as_chunks::<LANES>()
                            .0
                            .iter()
                            .zip(b_block.as_chunks::<LANES>().0)
                        {
                            for lane in 0..LANES {
                                let difference = a[lane] - b[lane];
                                lanes[lane] += difference * difference;
                            }
                        }
                        distance += lanes.iter().sum::<f32>();
                        if distance >= second {
                            continue 'candidates;
                        }
                    }
                    if distance < best {
                        second = best;
                        best = distance;
                        best_index = train_index;
                    } else if distance < second {
                        second = distance;
                    }
                }

                (best.is_finite() && second.is_finite() && second > 0.0).then_some(Neighbours {
                    best_index,
                    best,
                    second,
                })
            })
            .collect()
    }
}

impl FeatureMatcher for RatioMatcher {
    fn match_features(
        &self,
        query: &FeatureSet,
        train: &FeatureSet,
        ratio: f32,
    ) -> Vec<FeatureMatch> {
        self.match_features_with_ratios(query, train, &[ratio])
            .pop()
            .unwrap_or_default()
    }

    fn match_features_with_ratios(
        &self,
        query: &FeatureSet,
        train: &FeatureSet,
        ratios: &[f32],
    ) -> Vec<Vec<FeatureMatch>> {
        if query.is_empty() || train.len() < 2 {
            return vec![Vec::new(); ratios.len()];
        }
        let max_squared = self.max_distance.map(|d| d * d);
        let neighbours = Self::neighbours(query, train);
        ratios
            .iter()
            .map(|ratio| {
                let ratio = ratio.clamp(0.1, 0.99);
                neighbours
                    .iter()
                    .enumerate()
                    .filter_map(|(query_index, found)| {
                        let found = found.as_ref()?;
                        if max_squared.is_some_and(|limit| found.best > limit) {
                            return None;
                        }
                        let distance = found.best.sqrt();
                        (distance < ratio * found.second.sqrt())
                            .then(|| FeatureMatch::new(query_index, found.best_index, distance))
                    })
                    .collect()
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
    fn several_ratios_match_like_separate_calls() {
        let a = FeatureSet::new((0..12).map(|i| feature(i as f32 * 0.3, i)).collect());
        let b = FeatureSet::new(
            (0..12)
                .map(|i| feature(i as f32 * 0.31 + 0.05, i))
                .collect(),
        );
        let matcher = RatioMatcher::default();
        let together = matcher.match_features_with_ratios(&a, &b, &[0.5, 0.9]);
        assert_eq!(together[0], matcher.match_features(&a, &b, 0.5));
        assert_eq!(together[1], matcher.match_features(&a, &b, 0.9));
        assert!(together[0].len() <= together[1].len());
    }

    #[test]
    fn empty_sets_are_safe() {
        let empty = FeatureSet::default();
        assert!(RatioMatcher::default()
            .match_features(&empty, &empty, 0.8)
            .is_empty());
    }
}
