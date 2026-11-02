//! Reusable behavioural contracts for ports.
//!
//! Each function takes a port implementation (or a factory) and asserts only
//! properties that every adapter must satisfy. Run the same function against a
//! production adapter and an in-memory double to prove substitutability.

use std::sync::Arc;

use stitch_domain::{
    Clock, Compositor, CoreError, FeatureExtractor, FeatureMatch, FeatureMatcher, Homography,
    HomographyEstimator, ImageLocator, ImageRepository, Point, RgbaImage, StitchOptions, WarpLayer,
    WarpOptions,
};

use crate::fixtures::{features_at_points, grid_points, textured_gray, textured_rgba};

/// Contract for [`ImageRepository`].
pub fn image_repository_contract(
    repository: &dyn ImageRepository,
    locator: &dyn Fn(&str) -> ImageLocator,
) {
    let round_trip = locator("contract-round-trip.png");
    let image = RgbaImage::from_fn(37, 23, |x, y| [(x * 7) as u8, (y * 11) as u8, 128, 255]);
    repository
        .save(&round_trip, &image)
        .expect("saving a PNG must succeed");
    let loaded = repository
        .load(&round_trip)
        .expect("a just-saved image must load");
    assert_eq!(loaded.image.width(), image.width());
    assert_eq!(loaded.image.height(), image.height());
    assert_eq!(loaded.image.pixels(), image.pixels());
    assert!(!loaded.name.is_empty());
    assert!(loaded.thumbnail.width() > 0 && loaded.thumbnail.height() > 0);
    assert!(loaded.thumbnail.width() <= loaded.image.width());

    // Overwriting must replace, not append.
    let replacement = RgbaImage::filled(9, 5, [1, 2, 3, 255]);
    repository.save(&round_trip, &replacement).unwrap();
    assert_eq!(repository.load(&round_trip).unwrap().image, replacement);

    let missing = locator("contract-does-not-exist.png");
    assert!(matches!(
        repository.load(&missing),
        Err(CoreError::NotFound(_))
    ));
}

/// Contract for [`FeatureExtractor`].
pub fn feature_extractor_contract(extractor: &dyn FeatureExtractor) {
    let image = textured_gray(256, 192, 42);
    let features = extractor
        .extract(&image)
        .expect("textured images must produce features");
    assert!(!features.is_empty(), "no features detected");

    for feature in features.iter() {
        let keypoint = feature.keypoint;
        assert!(keypoint.x.is_finite() && keypoint.y.is_finite());
        assert!(keypoint.x >= -1.0 && keypoint.x <= image.width() as f32 + 1.0);
        assert!(keypoint.y >= -1.0 && keypoint.y <= image.height() as f32 + 1.0);
        assert!(keypoint.scale > 0.0 && keypoint.scale.is_finite());
        assert!(keypoint.orientation.is_finite());
        assert!(feature.descriptor.is_normalized(1e-2));
        assert!(feature.descriptor.values.iter().all(|v| v.is_finite()));
    }

    let again = extractor.extract(&image).unwrap();
    assert_eq!(features, again, "extraction must be deterministic");

    let flat = stitch_domain::GrayImage::from_fn(128, 128, |_, _| 0.5);
    let flat_features = extractor.extract(&flat).unwrap();
    assert!(
        flat_features.is_empty(),
        "flat images must not produce features"
    );
}

/// Contract for [`FeatureMatcher`].
pub fn feature_matcher_contract(matcher: &dyn FeatureMatcher) {
    let points = grid_points(8, 6, 17.0);
    let a = features_at_points(&points, 7);
    let b = features_at_points(&points, 7);
    let matches = matcher.match_features(&a, &b, 0.8);
    assert!(
        matches.len() >= points.len() - 4,
        "identical sets should match, got {}",
        matches.len()
    );
    let mut query_indices = std::collections::HashSet::new();
    let mut train_indices = std::collections::HashSet::new();
    for m in &matches {
        assert!(query_indices.insert(m.query_index));
        assert!(train_indices.insert(m.train_index));
        assert!(m.distance >= 0.0);
    }

    let unrelated = features_at_points(&points, 99);
    let unrelated_matches = matcher.match_features(&a, &unrelated, 0.75);
    assert!(
        unrelated_matches.len() < 6,
        "unrelated descriptors produced {} matches",
        unrelated_matches.len()
    );

    let empty = stitch_domain::FeatureSet::default();
    assert!(matcher.match_features(&empty, &b, 0.8).is_empty());
    assert!(matcher.match_features(&a, &empty, 0.8).is_empty());
}

/// Contract for [`HomographyEstimator`] with clean correspondences.
pub fn homography_estimator_contract(estimator: &dyn HomographyEstimator) {
    let transform = Homography::from_rows([
        [1.07, -0.11, 19.0],
        [0.09, 1.04, -7.0],
        [0.00015, -0.00009, 1.0],
    ]);
    let source_points = grid_points(6, 6, 23.0);
    let target_points: Vec<Point> = source_points
        .iter()
        .map(|&point| transform.transform(point).unwrap())
        .collect();
    let query = features_at_points(&source_points, 11);
    let train = features_at_points(&target_points, 12);
    let matches: Vec<FeatureMatch> = (0..source_points.len())
        .map(|index| FeatureMatch::new(index, index, 0.5))
        .collect();
    let options = StitchOptions {
        min_inliers: 20,
        ransac_threshold: 2.0,
        ..StitchOptions::default()
    };
    let estimate = estimator
        .estimate(&query, &train, &matches, &options)
        .expect("a clean transform must be recoverable");
    assert!(estimate.inliers.len() >= 20);
    assert!(estimate.rms_error < 1.0);
    for probe in [
        Point::new(50.0, 50.0),
        Point::new(130.0, 70.0),
        Point::new(90.0, 140.0),
    ] {
        let actual = estimate.homography.transform(probe).unwrap();
        let expected = transform.transform(probe).unwrap();
        assert!(
            actual.distance(expected) < 1.0,
            "{actual:?} vs {expected:?}"
        );
    }

    assert!(
        estimator
            .estimate(&query, &train, &matches[..3], &options)
            .is_err(),
        "fewer than four matches must fail"
    );
}

/// Contract for [`Compositor`].
pub fn compositor_contract(compositor: &dyn Compositor) {
    let image = RgbaImage::from_fn(48, 32, |x, y| [(x * 4) as u8, (y * 6) as u8, 77, 255]);
    let options = WarpOptions {
        canvas_width: 48,
        canvas_height: 32,
        feather_fraction: 0.0,
    };
    let single = compositor
        .composite(
            &[WarpLayer {
                name: "single".to_owned(),
                image: Arc::new(image.clone()),
                homography: Homography::identity(),
            }],
            &options,
        )
        .unwrap();
    assert_eq!(single.width(), 48);
    assert_eq!(single.height(), 32);
    assert_eq!(single, image);

    let red = RgbaImage::filled(100, 40, [255, 0, 0, 255]);
    let blue = RgbaImage::filled(100, 40, [0, 0, 255, 255]);
    let overlap = compositor
        .composite(
            &[
                WarpLayer {
                    name: "red".to_owned(),
                    image: Arc::new(red),
                    homography: Homography::identity(),
                },
                WarpLayer {
                    name: "blue".to_owned(),
                    image: Arc::new(blue),
                    homography: Homography::translation(50.0, 0.0),
                },
            ],
            &WarpOptions {
                canvas_width: 200,
                canvas_height: 40,
                feather_fraction: 0.0,
            },
        )
        .unwrap();
    assert_eq!((overlap.width(), overlap.height()), (200, 40));
    assert_eq!(overlap.pixel(10, 20), [255, 0, 0, 255]);
    assert_eq!(overlap.pixel(140, 20), [0, 0, 255, 255]);
    let blended = overlap.pixel(75, 20);
    assert!(blended[0] > 100 && blended[2] > 100, "{blended:?}");
    assert_eq!(blended[3], 255);
    assert_eq!(overlap.pixel(190, 20)[3], 0);

    assert!(compositor.composite(&[], &options).is_err());
}

/// Contract for [`Clock`].
pub fn clock_contract(clock: &dyn Clock) {
    let first = clock.now_ms();
    let second = clock.now_ms();
    assert!(second >= first);
}

/// Builds a small source image for application-level tests.
pub fn test_image(width: u32, height: u32) -> RgbaImage {
    textured_rgba(width, height, 5)
}
