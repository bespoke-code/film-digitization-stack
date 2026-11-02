//! Runs every reusable contract against both the production adapter and the
//! in-memory double. A failure here means an adapter is not substitutable.

use std::sync::Arc;

use stitch_application::SystemClock;
use stitch_domain::{Clock, ImageLocator};
use stitch_imaging::{BilinearCompositor, FileImageRepository};
use stitch_testkit::contracts::{
    clock_contract, compositor_contract, feature_extractor_contract, feature_matcher_contract,
    homography_estimator_contract, image_repository_contract,
};
use stitch_testkit::{
    FakeClock, InMemoryCompositor, InMemoryFeatureExtractor, InMemoryHomographyEstimator,
    InMemoryImageRepository, InMemoryMatcher,
};
use stitch_vision::{RansacHomographyEstimator, RatioMatcher, SiftExtractor};

#[test]
fn file_repository_satisfies_contract() {
    let directory = tempfile::tempdir().unwrap();
    let repository = FileImageRepository::default();
    let locator =
        |name: &str| ImageLocator::new(directory.path().join(name).to_string_lossy().into_owned());
    image_repository_contract(&repository, &locator);
}

#[test]
fn in_memory_repository_satisfies_contract() {
    let repository = InMemoryImageRepository::new();
    image_repository_contract(&repository, &|name| ImageLocator::new(name));
}

#[test]
fn sift_extractor_satisfies_contract() {
    feature_extractor_contract(&SiftExtractor::default());
}

#[test]
fn in_memory_extractor_satisfies_contract() {
    feature_extractor_contract(&InMemoryFeatureExtractor);
}

#[test]
fn ratio_matcher_satisfies_contract() {
    feature_matcher_contract(&RatioMatcher::default());
}

#[test]
fn in_memory_matcher_satisfies_contract() {
    feature_matcher_contract(&InMemoryMatcher);
}

#[test]
fn ransac_estimator_satisfies_contract() {
    homography_estimator_contract(&RansacHomographyEstimator::default());
}

#[test]
fn in_memory_estimator_satisfies_contract() {
    homography_estimator_contract(&InMemoryHomographyEstimator);
}

#[test]
fn bilinear_compositor_satisfies_contract() {
    compositor_contract(&BilinearCompositor);
}

#[test]
fn in_memory_compositor_satisfies_contract() {
    compositor_contract(&InMemoryCompositor);
}

#[test]
fn system_clock_satisfies_contract() {
    clock_contract(&SystemClock);
}

#[test]
fn fake_clock_satisfies_contract() {
    let clock = FakeClock::new(1_000);
    clock_contract(&clock);
    clock.advance(25);
    assert_eq!(clock.now_ms(), 1_025);
}

#[test]
fn shared_repository_handles_are_usable() {
    let repository = Arc::new(InMemoryImageRepository::new());
    repository.insert(
        "image.png",
        stitch_testkit::fixtures::textured_rgba(32, 24, 1),
    );
    let shared: Arc<dyn stitch_domain::ImageRepository> = repository.clone();
    let loaded = shared
        .load(&ImageLocator::new("image.png"))
        .expect("shared repository must load");
    assert_eq!((loaded.image.width(), loaded.image.height()), (32, 24));
}
