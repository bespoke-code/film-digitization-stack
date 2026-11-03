//! End-to-end stitching with the real SIFT/RANSAC/compositor adapters.

use std::sync::Arc;

use stitch_application::{ProjectService, StitchService, SystemClock};
use stitch_domain::{ImageLocator, RgbaImage};
use stitch_imaging::BilinearCompositor;
use stitch_testkit::fixtures::{textured_rgba, warp_rgba};
use stitch_testkit::InMemoryImageRepository;
use stitch_ui_core::Workbench;
use stitch_vision::{RansacHomographyEstimator, RatioMatcher, SiftExtractor};

fn production_service() -> StitchService {
    StitchService::new(
        Arc::new(SiftExtractor::default()),
        Arc::new(RatioMatcher::default()),
        Arc::new(RansacHomographyEstimator::default()),
        Arc::new(BilinearCompositor),
        Arc::new(SystemClock),
    )
}

fn crop(image: &RgbaImage, x: u32, width: u32) -> RgbaImage {
    RgbaImage::from_fn(width, image.height(), |dx, y| image.pixel(x + dx, y))
}

fn scale(image: &RgbaImage, factor: f32) -> RgbaImage {
    RgbaImage::from_fn(image.width(), image.height(), |x, y| {
        let pixel = image.pixel(x, y);
        [
            (pixel[0] as f32 * factor).round().clamp(0.0, 255.0) as u8,
            (pixel[1] as f32 * factor).round().clamp(0.0, 255.0) as u8,
            (pixel[2] as f32 * factor).round().clamp(0.0, 255.0) as u8,
            pixel[3],
        ]
    })
}

#[test]
fn stitches_overlapping_scans_through_the_workbench() {
    let base = textured_rgba(760, 440, 21);
    let left = crop(&base, 0, 480);
    let right = crop(&base, 280, 480);

    let repository = Arc::new(InMemoryImageRepository::new());
    repository.insert("left.png", left);
    repository.insert("right.png", right);

    let project_service = ProjectService::new(repository.clone());
    let mut workbench = Workbench::new();
    workbench
        .add_images(
            &project_service,
            &[
                ImageLocator::new("left.png"),
                ImageLocator::new("right.png"),
            ],
        )
        .unwrap();
    assert_eq!(workbench.project().len(), 2);

    workbench.stitch_sync(&production_service()).unwrap();
    let result = workbench.result().expect("a panorama must exist");
    assert!(
        result.image.width() >= 700,
        "expected a wide panorama, got {}x{}",
        result.image.width(),
        result.image.height()
    );
    assert_eq!(result.placements.len(), 2);
    assert!(result
        .diagnostics
        .pairs
        .iter()
        .any(|pair| pair.used_for_alignment && pair.num_inliers >= 12));

    let export = ImageLocator::new("panorama.png");
    workbench.export(&project_service, &export).unwrap();
    assert!(repository.saved().iter().any(|name| name == "panorama.png"));
    assert_eq!(
        repository.stored("panorama.png").unwrap().width(),
        result.image.width()
    );
}

#[test]
fn color_normalization_matches_exposure_between_frames() {
    let base = textured_rgba(760, 440, 21);
    let left = crop(&base, 0, 480);
    let right = scale(&crop(&base, 280, 480), 0.6);

    let repository = Arc::new(InMemoryImageRepository::new());
    repository.insert("left.png", left);
    repository.insert("right.png", right);
    let project_service = ProjectService::new(repository);
    let mut workbench = Workbench::new();
    workbench
        .add_images(
            &project_service,
            &[
                ImageLocator::new("left.png"),
                ImageLocator::new("right.png"),
            ],
        )
        .unwrap();
    workbench.stitch_sync(&production_service()).unwrap();
    let result = workbench.result().unwrap();

    let mut output = [0u64; 3];
    let mut expected = [0u64; 3];
    let mut count = 0u64;
    for y in 100..160 {
        for x in 650..720 {
            let actual = result.image.pixel(x, y);
            let wanted = base.pixel(x, y);
            for channel in 0..3 {
                output[channel] += actual[channel] as u64;
                expected[channel] += wanted[channel] as u64;
            }
            count += 1;
        }
    }
    for channel in 0..3 {
        let difference = (output[channel] as i64 - expected[channel] as i64).unsigned_abs() / count;
        assert!(
            difference <= 8,
            "channel {channel} differs by {difference} levels"
        );
    }
}

#[test]
fn manual_placement_aligns_featureless_scans() {
    let left = RgbaImage::filled(120, 80, [20, 20, 20, 255]);
    let right = RgbaImage::filled(120, 80, [20, 20, 20, 255]);
    let repository = Arc::new(InMemoryImageRepository::new());
    repository.insert("left.png", left);
    repository.insert("right.png", right);
    let project_service = ProjectService::new(repository);
    let mut workbench = Workbench::new();
    workbench
        .add_images(
            &project_service,
            &[
                ImageLocator::new("left.png"),
                ImageLocator::new("right.png"),
            ],
        )
        .unwrap();
    let right_id = workbench.project().images()[1].id;
    workbench.set_manual_placement(right_id, stitch_domain::Homography::translation(60.0, 0.0));
    workbench.stitch_sync(&production_service()).unwrap();
    let result = workbench.result().unwrap();
    assert_eq!((result.image.width(), result.image.height()), (180, 80));
    assert_eq!(result.image.pixel(170, 40), [20, 20, 20, 255]);
}

#[test]
fn automatic_registration_wins_over_a_rough_placement() {
    let base = textured_rgba(760, 440, 21);
    let left = crop(&base, 0, 480);
    let right = crop(&base, 280, 480);

    let repository = Arc::new(InMemoryImageRepository::new());
    repository.insert("left.png", left);
    repository.insert("right.png", right);
    let project_service = ProjectService::new(repository);
    let mut workbench = Workbench::new();
    workbench
        .add_images(
            &project_service,
            &[
                ImageLocator::new("left.png"),
                ImageLocator::new("right.png"),
            ],
        )
        .unwrap();
    let right_id = workbench.project().images()[1].id;
    workbench.set_manual_placement(
        right_id,
        stitch_domain::Homography::translation(3000.0, 900.0),
    );
    workbench.stitch_sync(&production_service()).unwrap();
    let result = workbench.result().unwrap();
    assert_eq!((result.image.width(), result.image.height()), (760, 440));
}

#[test]
fn handles_a_single_scan_without_stitching() {
    let repository = Arc::new(InMemoryImageRepository::new());
    repository.insert("single.png", textured_rgba(320, 240, 4));
    let project_service = ProjectService::new(repository);
    let mut workbench = Workbench::new();
    workbench
        .add_images(&project_service, &[ImageLocator::new("single.png")])
        .unwrap();
    workbench.stitch_sync(&production_service()).unwrap();
    let result = workbench.result().unwrap();
    assert_eq!((result.image.width(), result.image.height()), (320, 240));
    assert_eq!(result.placements.len(), 1);
}

#[test]
fn registers_a_rotated_scan() {
    let base = textured_rgba(640, 480, 33);
    let transform = stitch_domain::Homography::translation(70.0, 40.0).then(
        &stitch_domain::Homography::from_rows([
            [1.04, -0.09, 0.0],
            [0.09, 1.04, 0.0],
            [0.0, 0.0, 1.0],
        ]),
    );
    let warped = warp_rgba(&base, &transform, 760, 620, [18, 18, 18, 255]);

    let repository = Arc::new(InMemoryImageRepository::new());
    repository.insert("base.png", base);
    repository.insert("rotated.png", warped);
    let project_service = ProjectService::new(repository);
    let mut workbench = Workbench::new();
    workbench
        .add_images(
            &project_service,
            &[
                ImageLocator::new("base.png"),
                ImageLocator::new("rotated.png"),
            ],
        )
        .unwrap();
    workbench.stitch_sync(&production_service()).unwrap();
    let result = workbench.result().unwrap();
    assert!(result
        .diagnostics
        .pairs
        .iter()
        .any(|pair| pair.used_for_alignment && pair.num_inliers >= 12));
    assert!(result.image.width() > 640 && result.image.height() > 480);
}
