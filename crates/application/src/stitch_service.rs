//! Use case: register up to sixteen images and composite a panorama.

use std::sync::{Arc, OnceLock};
use std::time::Instant;

use stitch_domain::{
    Clock, ColorCurve, Compositor, CoreError, CoreResult, FeatureExtractor, FeatureMatcher,
    FeatureSet, Homography, HomographyEstimate, HomographyEstimator, ImageId, ImagePlacement,
    PairDiagnostics, Point, ProgressSink, RgbaImage, SourceImage, StitchDiagnostics, StitchOptions,
    StitchProgress, StitchResult, StitchStage, WarpOptions, MAX_IMAGES,
};

use crate::bundle::{refine_transforms, PairLink};
use crate::canvas::plan_canvas;
use crate::color::calibrate_color_curves;

/// Monotonic implementation of [`Clock`]. Wall-clock time can jump backwards
/// and would corrupt durations, so milliseconds count from the first call.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        static EPOCH: OnceLock<Instant> = OnceLock::new();
        EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64
    }
}

/// The registration pipeline.
///
/// The service is cheap to clone (all collaborators are `Arc`s) so it can be
/// moved into a background worker by a UI adapter.
#[derive(Clone)]
pub struct StitchService {
    extractor: Arc<dyn FeatureExtractor>,
    matcher: Arc<dyn FeatureMatcher>,
    estimator: Arc<dyn HomographyEstimator>,
    compositor: Arc<dyn Compositor>,
    clock: Arc<dyn Clock>,
}

impl StitchService {
    pub fn new(
        extractor: Arc<dyn FeatureExtractor>,
        matcher: Arc<dyn FeatureMatcher>,
        estimator: Arc<dyn HomographyEstimator>,
        compositor: Arc<dyn Compositor>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            extractor,
            matcher,
            estimator,
            compositor,
            clock,
        }
    }

    pub fn extractor(&self) -> &Arc<dyn FeatureExtractor> {
        &self.extractor
    }

    pub fn matcher(&self) -> &Arc<dyn FeatureMatcher> {
        &self.matcher
    }

    pub fn estimator(&self) -> &Arc<dyn HomographyEstimator> {
        &self.estimator
    }

    pub fn compositor(&self) -> &Arc<dyn Compositor> {
        &self.compositor
    }

    /// Runs the full pipeline. `progress` may be a closure thanks to the
    /// blanket implementation on [`ProgressSink`].
    pub fn stitch(
        &self,
        images: &[SourceImage],
        options: &StitchOptions,
        progress: &dyn ProgressSink,
    ) -> CoreResult<StitchResult> {
        self.stitch_with_manual(images, &[], options, progress)
    }

    /// Runs the pipeline with rough placements as a fallback.
    ///
    /// `manual[i]` maps image `i` into the reference frame of image 0. The
    /// service still tries automatic registration first. A manual placement is
    /// used only when no pair produces an acceptable homography, which lets a
    /// user align featureless scans without losing automatic corrections for
    /// the rest.
    pub fn stitch_with_manual(
        &self,
        images: &[SourceImage],
        manual: &[Option<Homography>],
        options: &StitchOptions,
        progress: &dyn ProgressSink,
    ) -> CoreResult<StitchResult> {
        self.stitch_with_rotations(images, &[], manual, options, progress)
    }

    /// Runs the pipeline after applying a clockwise rotation to each source.
    ///
    /// `rotations[i]` is in degrees and may be empty when no image rotates.
    /// Manual placements are expressed in the rotated image's coordinates.
    pub fn stitch_with_rotations(
        &self,
        images: &[SourceImage],
        rotations: &[f32],
        manual: &[Option<Homography>],
        options: &StitchOptions,
        progress: &dyn ProgressSink,
    ) -> CoreResult<StitchResult> {
        if !rotations.is_empty() && rotations.len() != images.len() {
            return Err(CoreError::InvalidImage(
                "every image needs exactly one rotation".to_owned(),
            ));
        }
        let rotated;
        let images = if rotations
            .iter()
            .any(|degrees| !stitch_domain::rotation_is_identity(*degrees))
        {
            rotated = rotate_images(images, rotations);
            rotated.as_slice()
        } else {
            images
        };
        self.stitch_images(images, manual, options, progress)
    }

    fn stitch_images(
        &self,
        images: &[SourceImage],
        manual: &[Option<Homography>],
        options: &StitchOptions,
        progress: &dyn ProgressSink,
    ) -> CoreResult<StitchResult> {
        if images.is_empty() {
            return Err(CoreError::EmptyProject);
        }
        if images.len() > MAX_IMAGES {
            return Err(CoreError::TooManyImages { max: MAX_IMAGES });
        }
        let options = options.clone().sanitized();
        let started = self.clock.now_ms();

        if images.len() == 1 {
            return Ok(StitchResult {
                image: images[0].image.clone(),
                placements: vec![ImagePlacement {
                    image_id: images[0].id,
                    name: images[0].name.clone(),
                    homography: Homography::identity(),
                    color: ColorCurve::identity(),
                }],
                diagnostics: StitchDiagnostics {
                    features_per_image: vec![0],
                    pairs: Vec::new(),
                    duration_ms: self.clock.now_ms().saturating_sub(started) as u128,
                    output_width: images[0].image.width(),
                    output_height: images[0].image.height(),
                    output_scale: 1.0,
                },
            });
        }

        let analyses = self.extract_all(images, &options, progress)?;
        let features: Vec<&FeatureSet> =
            analyses.iter().map(|analysis| &analysis.features).collect();

        progress.report(StitchProgress::new(
            StitchStage::EstimatingTransforms,
            0,
            images.len() - 1,
            "Registering images",
        ));

        let (transforms, colors, pairs) =
            self.register(images, &analyses, &features, manual, &options, progress)?;

        let transforms: Vec<Homography> = transforms
            .into_iter()
            .map(|transform| {
                transform
                    .ok_or_else(|| CoreError::Homography("image was not registered".to_owned()))
            })
            .collect::<CoreResult<Vec<_>>>()?;

        let mut plan = plan_canvas(images, &transforms, &options)?;
        for (layer, color) in plan.layers.iter_mut().zip(colors.iter()) {
            layer.color = color.clone();
        }
        progress.report(StitchProgress::new(
            StitchStage::Compositing,
            0,
            1,
            format!("Blending {} x {} canvas", plan.width, plan.height),
        ));

        let warp_options = WarpOptions {
            canvas_width: plan.width,
            canvas_height: plan.height,
            feather_fraction: options.feather_fraction,
        };
        let composited = self.compositor.composite(&plan.layers, &warp_options)?;

        let placements = images
            .iter()
            .zip(plan.layers.iter())
            .map(|(image, layer)| ImagePlacement {
                image_id: image.id,
                name: image.name.clone(),
                homography: layer.homography,
                color: layer.color.clone(),
            })
            .collect();

        progress.report(StitchProgress::new(
            StitchStage::Finished,
            1,
            1,
            "Stitch complete",
        ));

        Ok(StitchResult {
            image: Arc::new(composited),
            placements,
            diagnostics: StitchDiagnostics {
                features_per_image: features.iter().map(|set| set.len()).collect(),
                pairs,
                duration_ms: self.clock.now_ms().saturating_sub(started) as u128,
                output_width: plan.width,
                output_height: plan.height,
                output_scale: plan.scale,
            },
        })
    }

    fn extract_all(
        &self,
        images: &[SourceImage],
        options: &StitchOptions,
        progress: &dyn ProgressSink,
    ) -> CoreResult<Vec<AnalysisImage>> {
        let mut analyses = Vec::with_capacity(images.len());
        for (index, image) in images.iter().enumerate() {
            progress.report(StitchProgress::new(
                StitchStage::ExtractingFeatures,
                index,
                images.len(),
                format!("Detecting features in {}", image.name),
            ));

            let source = image.image.as_ref();
            let resized = source.resized_to_fit(options.max_analysis_dimension);
            let scale_x = resized.width() as f64 / source.width() as f64;
            let scale_y = resized.height() as f64 / source.height() as f64;
            let gray = resized.to_gray();
            drop(resized);
            let mut features = self.extractor.extract(&gray).map_err(|error| {
                CoreError::FeatureExtraction(format!("{}: {error}", image.name))
            })?;

            features
                .features
                .sort_by(|a, b| b.keypoint.response.total_cmp(&a.keypoint.response));
            features.features.truncate(options.max_features);
            features.features.shrink_to_fit();

            analyses.push(AnalysisImage {
                scale_x,
                scale_y,
                features,
            });
        }
        Ok(analyses)
    }

    fn register(
        &self,
        images: &[SourceImage],
        analyses: &[AnalysisImage],
        features: &[&FeatureSet],
        manual: &[Option<Homography>],
        options: &StitchOptions,
        progress: &dyn ProgressSink,
    ) -> CoreResult<Registration> {
        let mut transforms: Vec<Option<Homography>> = vec![None; images.len()];
        transforms[0] = Some(Homography::identity());
        let mut diagnostics = Vec::new();
        let mut links: Vec<PairLink> = Vec::new();
        let mut evaluated: Vec<(usize, usize)> = Vec::new();

        for index in 1..images.len() {
            progress.report(StitchProgress::new(
                StitchStage::Matching,
                index,
                images.len() - 1,
                format!("Matching {}", images[index].name),
            ));

            let mut candidates = Vec::new();
            candidates.push(index - 1);
            if index > 1 {
                candidates.push(0);
            }
            for candidate in (0..index).rev() {
                if !candidates.contains(&candidate) {
                    candidates.push(candidate);
                }
            }

            // Evaluate the two most likely neighbours first. Only fall back to
            // the remaining previous images if neither produced a transform.
            let primary_count = candidates.len().min(2);
            let mut best: Option<Candidate> = None;
            for (attempt, &target) in candidates.iter().enumerate() {
                evaluated.push(ordered_pair(index, target));
                let evaluation = self.evaluate_pair(
                    index,
                    target,
                    images,
                    analyses,
                    features,
                    options,
                    &mut diagnostics,
                );
                if let Some(candidate) = evaluation {
                    push_link(&mut links, &diagnostics, index, &candidate);
                    if best
                        .as_ref()
                        .map(|current| candidate.inliers > current.inliers)
                        .unwrap_or(true)
                    {
                        best = Some(candidate);
                    }
                }
                if attempt + 1 >= primary_count && best.is_some() {
                    break;
                }
            }

            if let Some(chosen) = best {
                if let Some(diagnostic) = diagnostics.get_mut(chosen.diagnostic_index) {
                    diagnostic.used_for_alignment = true;
                }

                let parent = transforms[chosen.target].as_ref().ok_or_else(|| {
                    CoreError::Homography(format!(
                        "{} was registered before its reference",
                        images[chosen.target].name
                    ))
                })?;
                transforms[index] = Some(parent.then(&chosen.homography));
                continue;
            }

            let placement = manual
                .get(index)
                .copied()
                .flatten()
                .filter(Homography::is_finite)
                .ok_or_else(|| CoreError::NotEnoughOverlap(images[index].name.clone()))?;
            transforms[index] = Some(placement);
        }

        // The tree only evaluates a few neighbours. Every remaining overlap
        // still reaches the compositor, so evaluate the pairs whose canvas
        // bounds intersect and give all of them to the global fit.
        let registered: Vec<Homography> = transforms
            .iter()
            .map(|transform| transform.expect("every image is registered"))
            .collect();
        for first in 0..images.len() {
            for second in (first + 1)..images.len() {
                if evaluated.contains(&ordered_pair(first, second)) {
                    continue;
                }
                if overlap_fraction(
                    &registered[first],
                    &registered[second],
                    images[first].image.as_ref(),
                    images[second].image.as_ref(),
                ) < 0.02
                {
                    continue;
                }
                evaluated.push(ordered_pair(first, second));
                if let Some(candidate) = self.evaluate_pair(
                    second,
                    first,
                    images,
                    analyses,
                    features,
                    options,
                    &mut diagnostics,
                ) {
                    push_link(&mut links, &diagnostics, second, &candidate);
                }
            }
        }

        let sizes: Vec<(f64, f64)> = images
            .iter()
            .map(|image| (image.image.width() as f64, image.image.height() as f64))
            .collect();
        let refined = refine_transforms(&registered, &sizes, &links);
        let transforms: Vec<Option<Homography>> = refined.into_iter().map(Some).collect();

        let colors = if options.normalize_colors {
            calibrate_registered_colors(images, &transforms, &diagnostics)
        } else {
            vec![ColorCurve::identity(); images.len()]
        };

        Ok((transforms, colors, diagnostics))
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_pair(
        &self,
        query_index: usize,
        target_index: usize,
        images: &[SourceImage],
        analyses: &[AnalysisImage],
        features: &[&FeatureSet],
        options: &StitchOptions,
        diagnostics: &mut Vec<PairDiagnostics>,
    ) -> Option<Candidate> {
        let query_features = features[query_index];
        let target_features = features[target_index];
        // A blurred or low-contrast frame can lose its matches under the strict
        // ratio. Retry once with a relaxed ratio before declaring no overlap.
        // Both ratios come from one neighbour search.
        let relaxed_ratio = (options.match_ratio + 0.1).min(0.95);
        let ratios: &[f32] = if relaxed_ratio > options.match_ratio {
            &[options.match_ratio, relaxed_ratio]
        } else {
            &[options.match_ratio]
        };
        let mut lists = self
            .matcher
            .match_features_with_ratios(query_features, target_features, ratios)
            .into_iter();
        let mut matches = lists.next().unwrap_or_default();
        let relaxed = lists.next();
        let mut estimate = self.estimate_pair(query_features, target_features, &matches, options);

        let weak = estimate
            .as_ref()
            .map(|estimate| estimate.inliers.len() < options.min_inliers * 2)
            .unwrap_or(true);
        if let Some(relaxed) = relaxed.filter(|_| weak) {
            if relaxed.len() > matches.len() {
                if let Some(relaxed_estimate) =
                    self.estimate_pair(query_features, target_features, &relaxed, options)
                {
                    let better = estimate
                        .as_ref()
                        .map(|current| {
                            relaxed_estimate.inliers.len() > current.inliers.len()
                                || (relaxed_estimate.inliers.len() == current.inliers.len()
                                    && relaxed_estimate.rms_error < current.rms_error)
                        })
                        .unwrap_or(true);
                    if better {
                        matches = relaxed;
                        estimate = Some(relaxed_estimate);
                    }
                }
            }
        }

        let from = images[query_index].id;
        let to = images[target_index].id;
        let mut diagnostic = PairDiagnostics {
            from,
            to,
            num_matches: matches.len(),
            num_inliers: 0,
            rms_error: 0.0,
            accepted: false,
            used_for_alignment: false,
        };

        let estimate = match estimate {
            Some(estimate) => estimate,
            None => {
                log::debug!(
                    "pair {} -> {} rejected: no valid transform",
                    images[query_index].name,
                    images[target_index].name
                );
                diagnostics.push(diagnostic);
                return None;
            }
        };

        diagnostic.num_inliers = estimate.inliers.len();
        diagnostic.rms_error = estimate.rms_error;
        diagnostic.accepted = estimate.inliers.len() >= options.min_inliers;
        let diagnostic_index = diagnostics.len();
        diagnostics.push(diagnostic);

        if estimate.inliers.len() < options.min_inliers {
            return None;
        }

        let homography =
            to_original_coordinates(&estimate, &analyses[query_index], &analyses[target_index])?;

        Some(Candidate {
            target: target_index,
            homography,
            inliers: estimate.inliers.len(),
            diagnostic_index,
        })
    }

    fn estimate_pair(
        &self,
        query: &FeatureSet,
        target: &FeatureSet,
        matches: &[stitch_domain::FeatureMatch],
        options: &StitchOptions,
    ) -> Option<HomographyEstimate> {
        if matches.len() < options.min_inliers {
            return None;
        }
        self.estimator
            .estimate(query, target, matches, options)
            .ok()
    }
}

struct AnalysisImage {
    scale_x: f64,
    scale_y: f64,
    features: FeatureSet,
}

fn rotate_images(images: &[SourceImage], rotations: &[f32]) -> Vec<SourceImage> {
    images
        .iter()
        .enumerate()
        .map(|(index, image)| {
            let degrees = rotations.get(index).copied().unwrap_or(0.0);
            if stitch_domain::rotation_is_identity(degrees) {
                image.clone()
            } else {
                SourceImage {
                    id: image.id,
                    name: image.name.clone(),
                    locator: image.locator.clone(),
                    image: Arc::new(image.image.rotated(degrees)),
                    thumbnail: image.thumbnail.clone(),
                    jpeg_quality: image.jpeg_quality,
                }
            }
        })
        .collect()
}

/// Per-image transforms, color curves and pair diagnostics.
type Registration = (
    Vec<Option<Homography>>,
    Vec<ColorCurve>,
    Vec<PairDiagnostics>,
);

struct Candidate {
    target: usize,
    homography: Homography,
    inliers: usize,
    diagnostic_index: usize,
}

/// Calibrates all images at once from every accepted overlap. Pair
/// transforms come from the final canvas transforms so the calibration
/// matches what the compositor actually blends.
///
/// The registration tree alone is a chain: a calibration fitted only along it
/// drifts at the far end, and two frames that overlap directly but sit far
/// apart in the chain then blend with a visible step. The long-range pairs
/// close those loops. The gain prior and the outlier limit in the curve fit
/// keep a vignetted corner overlap from bending a whole frame.
fn calibrate_registered_colors(
    images: &[SourceImage],
    transforms: &[Option<Homography>],
    diagnostics: &[PairDiagnostics],
) -> Vec<ColorCurve> {
    let mut overlaps = Vec::new();
    for pair in diagnostics.iter().filter(|pair| pair.accepted) {
        let Some(query) = image_index(images, pair.from) else {
            continue;
        };
        let Some(target) = image_index(images, pair.to) else {
            continue;
        };
        let (Some(query_transform), Some(target_transform)) =
            (transforms[query].as_ref(), transforms[target].as_ref())
        else {
            continue;
        };
        let Some(target_inverse) = target_transform.inverse() else {
            continue;
        };
        overlaps.push((query, target, target_inverse.then(query_transform)));
    }

    let images: Vec<&RgbaImage> = images.iter().map(|image| image.image.as_ref()).collect();
    calibrate_color_curves(&images, &overlaps)
}

fn image_index(images: &[SourceImage], id: ImageId) -> Option<usize> {
    images.iter().position(|image| image.id == id)
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn push_link(
    links: &mut Vec<PairLink>,
    diagnostics: &[PairDiagnostics],
    query: usize,
    candidate: &Candidate,
) {
    let rms = diagnostics
        .get(candidate.diagnostic_index)
        .map(|pair| pair.rms_error)
        .unwrap_or(0.0);
    links.push(PairLink {
        query,
        target: candidate.target,
        homography: candidate.homography,
        weight: candidate.inliers as f64 / (1.0 + rms),
    });
}

fn overlap_fraction(
    first: &Homography,
    second: &Homography,
    first_image: &RgbaImage,
    second_image: &RgbaImage,
) -> f64 {
    let (Some(first_quad), Some(second_quad)) = (
        first.transform_quad(first_image.width(), first_image.height()),
        second.transform_quad(second_image.width(), second_image.height()),
    ) else {
        return 0.0;
    };
    let first_box = bounds(&first_quad);
    let second_box = bounds(&second_quad);
    let width = (first_box.2.min(second_box.2) - first_box.0.max(second_box.0)).max(0.0);
    let height = (first_box.3.min(second_box.3) - first_box.1.max(second_box.1)).max(0.0);
    let intersection = width * height;
    let first_area = (first_box.2 - first_box.0) * (first_box.3 - first_box.1);
    let second_area = (second_box.2 - second_box.0) * (second_box.3 - second_box.1);
    let smaller = first_area.min(second_area);
    if smaller <= 0.0 {
        0.0
    } else {
        intersection / smaller
    }
}

fn bounds(quad: &[Point; 4]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    for point in quad {
        min_x = min_x.min(point.x);
        min_y = min_y.min(point.y);
        max_x = max_x.max(point.x);
        max_y = max_y.max(point.y);
    }
    (min_x, min_y, max_x, max_y)
}

/// Converts an analysis-coordinate transform into source-image coordinates.
fn to_original_coordinates(
    estimate: &HomographyEstimate,
    query: &AnalysisImage,
    target: &AnalysisImage,
) -> Option<Homography> {
    if query.scale_x <= 0.0
        || query.scale_y <= 0.0
        || target.scale_x <= 0.0
        || target.scale_y <= 0.0
    {
        return None;
    }
    let query_scale = Homography::scale(query.scale_x, query.scale_y);
    let target_inverse = Homography::scale(1.0 / target.scale_x, 1.0 / target.scale_y);
    Some(target_inverse.then(&estimate.homography).then(&query_scale))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use stitch_domain::{
        Descriptor, Feature, FeatureMatch, GrayImage, ImageId, ImageLocator, Keypoint, LoadedImage,
        RgbaImage,
    };

    /// Extractor that emits a fixed set of features so the orchestration can be
    /// tested without running SIFT.
    struct ScriptedExtractor {
        queue: Mutex<Vec<FeatureSet>>,
    }

    impl FeatureExtractor for ScriptedExtractor {
        fn extract(&self, _image: &GrayImage) -> CoreResult<FeatureSet> {
            self.queue
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| CoreError::FeatureExtraction("script exhausted".to_owned()))
        }
    }

    struct FixedMatcher;

    impl FeatureMatcher for FixedMatcher {
        fn match_features(
            &self,
            query: &FeatureSet,
            train: &FeatureSet,
            _ratio: f32,
        ) -> Vec<FeatureMatch> {
            let count = query.len().min(train.len());
            (0..count)
                .map(|index| FeatureMatch::new(index, index, 0.1))
                .collect()
        }
    }

    /// Returns no matches under the strict ratio and a full set under the
    /// relaxed retry ratio, which is what a low-contrast scan can look like.
    struct StrictThenRelaxedMatcher;

    impl FeatureMatcher for StrictThenRelaxedMatcher {
        fn match_features(
            &self,
            query: &FeatureSet,
            train: &FeatureSet,
            ratio: f32,
        ) -> Vec<FeatureMatch> {
            if ratio < 0.8 {
                return Vec::new();
            }
            let count = query.len().min(train.len());
            (0..count)
                .map(|index| FeatureMatch::new(index, index, 0.1))
                .collect()
        }
    }

    struct FixedEstimator;

    impl HomographyEstimator for FixedEstimator {
        fn estimate(
            &self,
            _query: &FeatureSet,
            _train: &FeatureSet,
            matches: &[FeatureMatch],
            _options: &StitchOptions,
        ) -> CoreResult<HomographyEstimate> {
            if matches.len() < 8 {
                return Err(CoreError::Homography("too few matches".to_owned()));
            }
            Ok(HomographyEstimate {
                homography: Homography::translation(20.0, 0.0),
                inliers: (0..matches.len()).collect(),
                rms_error: 0.0,
                num_matches: matches.len(),
            })
        }
    }

    struct SolidCompositor;

    impl Compositor for SolidCompositor {
        fn composite(
            &self,
            layers: &[stitch_domain::WarpLayer],
            options: &WarpOptions,
        ) -> CoreResult<RgbaImage> {
            assert!(!layers.is_empty());
            Ok(RgbaImage::filled(
                options.canvas_width,
                options.canvas_height,
                [10, 20, 30, 255],
            ))
        }
    }

    fn feature_set(count: usize) -> FeatureSet {
        FeatureSet::new(
            (0..count)
                .map(|i| Feature {
                    keypoint: Keypoint {
                        x: i as f32,
                        y: i as f32,
                        scale: 2.0,
                        orientation: 0.0,
                        response: 1.0,
                    },
                    descriptor: Descriptor::from_slice(&[0.1; 128]).unwrap(),
                })
                .collect(),
        )
    }

    fn source(id: u64) -> SourceImage {
        SourceImage {
            id: ImageId(id),
            name: format!("image-{id}"),
            locator: ImageLocator::new(format!("image-{id}")),
            image: Arc::new(RgbaImage::filled(200, 100, [0, 0, 0, 255])),
            thumbnail: Arc::new(RgbaImage::filled(20, 10, [0, 0, 0, 255])),
            jpeg_quality: None,
        }
    }

    fn service(feature_count: usize) -> StitchService {
        StitchService::new(
            Arc::new(ScriptedExtractor {
                queue: Mutex::new(vec![feature_set(feature_count), feature_set(feature_count)]),
            }),
            Arc::new(FixedMatcher),
            Arc::new(FixedEstimator),
            Arc::new(SolidCompositor),
            Arc::new(SystemClock),
        )
    }

    #[test]
    fn stitches_two_images_and_reports_progress() {
        let service = service(20);
        let images = vec![source(1), source(2)];
        let stages = Mutex::new(Vec::new());
        let result = service
            .stitch(
                &images,
                &StitchOptions::default(),
                &|progress: StitchProgress| {
                    stages.lock().unwrap().push(progress.stage);
                },
            )
            .unwrap();
        assert_eq!(result.image.width(), 220);
        assert_eq!(result.image.height(), 100);
        assert!(result
            .diagnostics
            .pairs
            .iter()
            .any(|pair| pair.used_for_alignment));
        let stages = stages.lock().unwrap();
        assert!(stages.contains(&StitchStage::Compositing));
        assert!(stages.contains(&StitchStage::Finished));
    }

    #[test]
    fn retries_a_pair_with_a_relaxed_match_ratio() {
        let service = StitchService::new(
            Arc::new(ScriptedExtractor {
                queue: Mutex::new(vec![feature_set(20), feature_set(20)]),
            }),
            Arc::new(StrictThenRelaxedMatcher),
            Arc::new(FixedEstimator),
            Arc::new(SolidCompositor),
            Arc::new(SystemClock),
        );
        let images = vec![source(1), source(2)];
        let result = service
            .stitch(
                &images,
                &StitchOptions::default(),
                &stitch_domain::NoProgress,
            )
            .unwrap();
        assert_eq!(result.image.width(), 220);
        let pair = result
            .diagnostics
            .pairs
            .iter()
            .find(|pair| pair.used_for_alignment)
            .expect("the relaxed retry must register the pair");
        assert_eq!(pair.num_matches, 20);
    }

    #[test]
    fn fails_when_no_pair_can_be_registered() {
        let service = service(2);
        let images = vec![source(1), source(2)];
        let error = service
            .stitch(
                &images,
                &StitchOptions::default(),
                &stitch_domain::NoProgress,
            )
            .unwrap_err();
        assert!(matches!(error, CoreError::NotEnoughOverlap(_)));
    }

    #[test]
    fn single_image_is_returned_unchanged() {
        let service = service(0);
        let images = vec![source(7)];
        let result = service
            .stitch(
                &images,
                &StitchOptions::default(),
                &stitch_domain::NoProgress,
            )
            .unwrap();
        assert_eq!(result.image.width(), 200);
        assert_eq!(result.placements[0].image_id, ImageId(7));
    }

    #[test]
    fn rotations_are_applied_before_stitching() {
        let service = service(0);
        let images = vec![source(7)];
        let result = service
            .stitch_with_rotations(
                &images,
                &[90.0],
                &[],
                &StitchOptions::default(),
                &stitch_domain::NoProgress,
            )
            .unwrap();
        assert_eq!(result.image.width(), 100);
        assert_eq!(result.image.height(), 200);
    }

    #[test]
    fn every_image_needs_a_rotation_when_rotations_are_given() {
        let service = service(0);
        let images = vec![source(1), source(2)];
        let error = service
            .stitch_with_rotations(
                &images,
                &[90.0],
                &[],
                &StitchOptions::default(),
                &stitch_domain::NoProgress,
            )
            .unwrap_err();
        assert!(matches!(error, CoreError::InvalidImage(_)));
    }

    #[test]
    fn export_through_repository_contract_double_is_possible() {
        // The application layer only needs the port; this test documents that
        // in-memory implementations are first-class collaborators.
        struct Repo(Mutex<Option<RgbaImage>>);
        impl stitch_domain::ImageRepository for Repo {
            fn load(&self, _locator: &ImageLocator) -> CoreResult<LoadedImage> {
                Err(CoreError::NotFound("unused".to_owned()))
            }
            fn save(&self, _locator: &ImageLocator, image: &RgbaImage) -> CoreResult<()> {
                *self.0.lock().unwrap() = Some(image.clone());
                Ok(())
            }
        }
        let repo = Arc::new(Repo(Mutex::new(None)));
        let service = crate::ProjectService::new(repo.clone());
        let image = RgbaImage::filled(3, 2, [1, 2, 3, 255]);
        service
            .export(&ImageLocator::new("out.png"), &image)
            .unwrap();
        assert_eq!(repo.0.lock().unwrap().as_ref().unwrap().width(), 3);
    }
}
