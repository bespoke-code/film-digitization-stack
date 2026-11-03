//! Use case: register 1..=6 images and composite a panorama.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use stitch_domain::{
    Clock, Compositor, CoreError, CoreResult, FeatureExtractor, FeatureMatcher, FeatureSet,
    GrayImage, Homography, HomographyEstimate, HomographyEstimator, ImagePlacement,
    PairDiagnostics, ProgressSink, SourceImage, StitchDiagnostics, StitchOptions, StitchProgress,
    StitchResult, StitchStage, WarpOptions, MAX_IMAGES,
};

use crate::canvas::plan_canvas;
use crate::color::estimate_gain;

/// Wall-clock implementation of [`Clock`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0)
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
        let features: Vec<FeatureSet> = analyses
            .iter()
            .map(|analysis| analysis.features.clone())
            .collect();

        progress.report(StitchProgress::new(
            StitchStage::EstimatingTransforms,
            0,
            images.len() - 1,
            "Registering images",
        ));

        let (transforms, gains, pairs) =
            self.register(images, &analyses, &features, &options, progress)?;

        let transforms: Vec<Homography> = transforms
            .into_iter()
            .map(|transform| {
                transform
                    .ok_or_else(|| CoreError::Homography("image was not registered".to_owned()))
            })
            .collect::<CoreResult<Vec<_>>>()?;

        let mut plan = plan_canvas(images, &transforms, &options)?;
        for (layer, gain) in plan.layers.iter_mut().zip(gains.iter()) {
            layer.gain = *gain;
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
                features_per_image: features.iter().map(FeatureSet::len).collect(),
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
            let gray = to_gray(&resized);
            let mut features = self.extractor.extract(&gray).map_err(|error| {
                CoreError::FeatureExtraction(format!("{}: {error}", image.name))
            })?;

            features
                .features
                .sort_by(|a, b| b.keypoint.response.total_cmp(&a.keypoint.response));
            features.features.truncate(options.max_features);

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
        features: &[FeatureSet],
        options: &StitchOptions,
        progress: &dyn ProgressSink,
    ) -> CoreResult<Registration> {
        let mut transforms: Vec<Option<Homography>> = vec![None; images.len()];
        transforms[0] = Some(Homography::identity());
        let mut gains = vec![[1.0f32; 3]; images.len()];
        let mut diagnostics = Vec::new();

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

            let chosen =
                best.ok_or_else(|| CoreError::NotEnoughOverlap(images[index].name.clone()))?;

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
            let parent_gain = gains[chosen.target];
            gains[index] = [
                parent_gain[0] * chosen.gain[0],
                parent_gain[1] * chosen.gain[1],
                parent_gain[2] * chosen.gain[2],
            ];
        }

        Ok((transforms, gains, diagnostics))
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_pair(
        &self,
        query_index: usize,
        target_index: usize,
        images: &[SourceImage],
        analyses: &[AnalysisImage],
        features: &[FeatureSet],
        options: &StitchOptions,
        diagnostics: &mut Vec<PairDiagnostics>,
    ) -> Option<Candidate> {
        let matches = self.matcher.match_features(
            &features[query_index],
            &features[target_index],
            options.match_ratio,
        );

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

        if matches.len() < options.min_inliers {
            diagnostics.push(diagnostic);
            return None;
        }

        let estimate = match self.estimator.estimate(
            &features[query_index],
            &features[target_index],
            &matches,
            options,
        ) {
            Ok(estimate) => estimate,
            Err(error) => {
                log::debug!(
                    "pair {} -> {} rejected: {error}",
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

        let gain = if options.normalize_colors {
            estimate_gain(
                images[query_index].image.as_ref(),
                images[target_index].image.as_ref(),
                &homography,
            )
        } else {
            [1.0; 3]
        };

        Some(Candidate {
            target: target_index,
            homography,
            gain,
            inliers: estimate.inliers.len(),
            diagnostic_index,
        })
    }
}

#[derive(Clone)]
struct AnalysisImage {
    scale_x: f64,
    scale_y: f64,
    features: FeatureSet,
}

/// Per-image transforms, color gains and pair diagnostics.
type Registration = (Vec<Option<Homography>>, Vec<[f32; 3]>, Vec<PairDiagnostics>);

struct Candidate {
    target: usize,
    homography: Homography,
    gain: [f32; 3],
    inliers: usize,
    diagnostic_index: usize,
}

fn to_gray(image: &stitch_domain::RgbaImage) -> GrayImage {
    image.to_gray()
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
        Descriptor, Feature, FeatureMatch, ImageId, ImageLocator, Keypoint, LoadedImage, RgbaImage,
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
