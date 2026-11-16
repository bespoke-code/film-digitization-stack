//! Hard-evidence tests against the real Nikon scan set in `docs/draft/pano-test`.
//!
//! The directory is intentionally gitignored: when the scans are absent these
//! tests report the reason and pass, so the workspace stays runnable from a
//! clean checkout.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use stitch_application::{StitchService, SystemClock};
use stitch_domain::{
    laplacian_variance, FocusMap, ImageId, ImageLocator, ImageRepository, SourceImage,
    StitchOptions, StitchResult,
};
use stitch_imaging::{BilinearCompositor, FileImageRepository};
use stitch_vision::{RansacHomographyEstimator, RatioMatcher, SiftExtractor};

const SCAN_NAMES: [&str; 6] = [
    "DSC_0376.jpg",
    "DSC_0377.jpg",
    "DSC_0378.jpg",
    "DSC_0379.jpg",
    "DSC_0380.jpg",
    "DSC_0381.jpg",
];

fn scan_paths() -> Option<Vec<PathBuf>> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/draft/pano-test");
    let paths: Vec<PathBuf> = SCAN_NAMES.iter().map(|name| directory.join(name)).collect();
    if paths.iter().any(|path| !path.is_file()) {
        eprintln!("skipping real scan test: {} not found", directory.display());
        return None;
    }
    Some(paths)
}

fn service() -> StitchService {
    StitchService::new(
        Arc::new(SiftExtractor::default()),
        Arc::new(RatioMatcher::default()),
        Arc::new(RansacHomographyEstimator::default()),
        Arc::new(BilinearCompositor),
        Arc::new(SystemClock),
    )
}

fn load(paths: &[PathBuf]) -> Vec<SourceImage> {
    let repository = FileImageRepository::default();
    paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            let loaded = repository
                .load(&ImageLocator::new(path.to_string_lossy().into_owned()))
                .unwrap();
            SourceImage {
                id: ImageId(index as u64 + 1),
                name: loaded.name,
                locator: loaded.locator,
                image: Arc::new(loaded.image),
                thumbnail: Arc::new(loaded.thumbnail),
                jpeg_quality: loaded.jpeg_quality,
            }
        })
        .collect()
}

fn stitch(images: &[SourceImage]) -> StitchResult {
    service()
        .stitch(images, &StitchOptions::default(), &|_| {})
        .unwrap()
}

/// Focus is resolution dependent, so compare everything at analysis scale.
fn focus_at_analysis_scale(image: &stitch_domain::RgbaImage) -> f64 {
    let preview = image.resized_to_fit(1280);
    laplacian_variance(&preview.to_gray())
}

fn tile_focus_at_analysis_scale(image: &stitch_domain::RgbaImage) -> f64 {
    let preview = image.resized_to_fit(1280);
    FocusMap::new(&preview.to_gray(), 8, 6).median()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted[sorted.len() / 2]
}

#[test]
fn real_scans_register_every_frame_with_low_reprojection_error() {
    let Some(paths) = scan_paths() else { return };
    let images = load(&paths);
    let result = stitch(&images);

    let used: Vec<_> = result
        .diagnostics
        .pairs
        .iter()
        .filter(|pair| pair.used_for_alignment)
        .collect();
    assert_eq!(
        used.len(),
        images.len() - 1,
        "every frame must be registered"
    );
    for pair in &used {
        assert!(
            pair.num_inliers >= 100,
            "{} -> {} only had {} inliers",
            pair.from.0,
            pair.to.0,
            pair.num_inliers
        );
        assert!(
            pair.rms_error < 3.0,
            "{} -> {} RMS {:.3} px at analysis scale",
            pair.from.0,
            pair.to.0,
            pair.rms_error
        );
    }

    // The set is a 2 x 3 grid: two distinct columns and three distinct rows.
    let columns = cluster(&result, |h| h[2], 1000.0);
    let rows = cluster(&result, |h| h[5], 1000.0);
    assert_eq!(columns, 2, "placements did not form two columns");
    assert_eq!(rows, 3, "placements did not form three rows");
}

fn cluster(result: &StitchResult, pick: impl Fn(&[f64; 9]) -> f64, tolerance: f64) -> usize {
    let mut centers: Vec<f64> = Vec::new();
    for placement in &result.placements {
        let value = pick(&placement.homography.normalized().m);
        if centers
            .iter()
            .all(|center| (value - center).abs() > tolerance)
        {
            centers.push(value);
        }
    }
    centers.len()
}

#[test]
fn real_scans_stitch_at_full_source_resolution() {
    let Some(paths) = scan_paths() else { return };
    let images = load(&paths);
    let result = stitch(&images);

    assert!(
        result.diagnostics.output_scale >= 0.99,
        "panorama was downscaled to {:.3} and lost detail",
        result.diagnostics.output_scale
    );
    assert!(
        result.image.width() >= 9000,
        "width {}",
        result.image.width()
    );
    assert!(
        result.image.height() >= 9000,
        "height {}",
        result.image.height()
    );
}

#[test]
fn real_scans_expose_soft_frames_without_losing_sharpness() {
    let Some(paths) = scan_paths() else { return };
    let images = load(&paths);
    let scores: Vec<f64> = images
        .iter()
        .map(|image| focus_at_analysis_scale(&image.image))
        .collect();
    let sharp_median = median(&scores);
    assert!(scores.iter().all(|&score| score > 0.0));
    assert!(
        scores[3] < sharp_median * 0.6,
        "DSC_0379 should be measurably softer: {scores:?}"
    );

    let source_tiles: Vec<f64> = images
        .iter()
        .map(|image| tile_focus_at_analysis_scale(&image.image))
        .collect();
    let source_tile_median = median(&source_tiles);

    let result = stitch(&images);
    let preview = result.image.resized_to_fit(1280);
    let map = FocusMap::new(&preview.to_gray(), 8, 6);
    assert!(
        map.median() > source_tile_median * 0.7,
        "stitched tile focus {:.1} is below 70% of source tile median {:.1}",
        map.median() * 1e6,
        source_tile_median * 1e6
    );
    let soft = map.soft_tile_fraction(map.median(), 0.25);
    assert!(
        soft < 0.3,
        "soft tile fraction {:.3} is too high (median focus {:.1})",
        soft,
        map.median() * 1e6
    );
}

#[test]
fn reversed_input_order_still_registers_the_scans() {
    let Some(paths) = scan_paths() else { return };
    let selected = vec![paths[0].clone(), paths[1].clone(), paths[2].clone()];
    let images = load(&selected);
    let reversed: Vec<SourceImage> = images.into_iter().rev().collect();
    let result = stitch(&reversed);

    let used = result
        .diagnostics
        .pairs
        .iter()
        .filter(|pair| pair.used_for_alignment)
        .count();
    assert_eq!(used, reversed.len() - 1);
    assert!(result.diagnostics.output_scale >= 0.99);
}
