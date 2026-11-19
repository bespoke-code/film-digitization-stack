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

/// Every overlap must agree in both geometry and tone after normalization.
///
/// A visible frame edge needs only one of the two to disagree: a misaligned
/// overlap blends different content, and a mismatched curve blends a step in
/// brightness. The test measures both properties on the final placements, so a
/// regression in registration or color calibration fails here.
#[test]
fn real_scans_blend_overlaps_without_visible_steps() {
    let Some(paths) = scan_paths() else { return };
    let images = load(&paths);
    let result = stitch(&images);

    let mut checked = 0;
    let mut located = 0;
    let mut reports = Vec::new();
    for pair in result.diagnostics.pairs.iter().filter(|pair| pair.accepted) {
        let query = pair.from.0 as usize - 1;
        let target = pair.to.0 as usize - 1;
        if query >= images.len() || target >= images.len() {
            continue;
        }
        let Some(target_inverse) = result.placements[target].homography.inverse() else {
            continue;
        };
        let query_to_target = target_inverse.then(&result.placements[query].homography);
        let report = overlap_agreement(
            images[query].image.as_ref(),
            images[target].image.as_ref(),
            &query_to_target,
            &result.placements[query].color,
            &result.placements[target].color,
        );
        if report.samples < 2000 {
            continue;
        }
        checked += 1;
        reports.push((pair.from.0, pair.to.0, report));
    }

    // The bias is what a feathered seam shows: a constant offset between the
    // two frames. A step below 13 of 255 levels stays under 5%, which the
    // feather spreads into a ramp the eye does not pick out. The residual is
    // looser because it also carries grain, the misregistration no homography
    // removes, and the lens falloff that meets at frame corners; it only
    // guards against gross mismatch.
    for (from, to, report) in &reports {
        assert!(
            report.bias.abs() <= 13.0,
            "{} -> {} blends a {:.2} level step",
            from,
            to,
            report.bias
        );
        assert!(
            report.residual < 50.0,
            "{} -> {} residual {:.2}",
            from,
            to,
            report.residual
        );
        assert!(
            report.correlation > 0.90,
            "{} -> {} overlap correlates {:.3}",
            from,
            to,
            report.correlation
        );
        // A featureless overlap, such as sky on a soft frame, has a flat
        // error surface whose minimum follows the vignetting gradient instead
        // of the content. Its shift says nothing about the registration;
        // correlation and residual above still guard it.
        if report.shift_contrast < 0.10 {
            continue;
        }
        located += 1;
        assert!(
            report.shift.0.abs() <= 5 && report.shift.1.abs() <= 5,
            "{} -> {} is off by {:?} px",
            from,
            to,
            report.shift
        );
    }
    assert!(checked >= 6, "only {checked} overlaps were verified");
    assert!(
        located >= 4,
        "only {located} of {checked} overlaps had structure to locate"
    );
}

struct Agreement {
    samples: usize,
    residual: f64,
    bias: f64,
    correlation: f64,
    shift: (i32, i32),
    /// Relative rise of the shift error eight pixels away from its minimum.
    /// A flat surface means the overlap has no structure to locate.
    shift_contrast: f64,
}

fn overlap_agreement(
    query: &stitch_domain::RgbaImage,
    target: &stitch_domain::RgbaImage,
    query_to_target: &stitch_domain::Homography,
    query_curve: &stitch_domain::ColorCurve,
    target_curve: &stitch_domain::ColorCurve,
) -> Agreement {
    let step = (query.width().max(query.height()) / 200).max(1);
    let mut mapped = Vec::new();
    let mut original = Vec::new();
    let mut points = Vec::new();
    let mut y = 0;
    while y < query.height() {
        let mut x = 0;
        while x < query.width() {
            if let Some(point) =
                query_to_target.transform(stitch_domain::Point::new(x as f64, y as f64))
            {
                if point.x >= 1.0
                    && point.y >= 1.0
                    && point.x < target.width() as f64 - 1.0
                    && point.y < target.height() as f64 - 1.0
                {
                    let from = query.pixel(x, y);
                    let to = target.pixel(point.x.round() as u32, point.y.round() as u32);
                    if from[3] > 0 && to[3] > 0 {
                        mapped.push(query_curve.map([from[0], from[1], from[2]]));
                        original.push(target_curve.map([to[0], to[1], to[2]]));
                        points.push((point.x, point.y));
                    }
                }
            }
            x += step;
        }
        y += step;
    }

    let samples = mapped.len();
    if samples == 0 {
        return Agreement {
            samples,
            residual: f64::MAX,
            bias: f64::MAX,
            correlation: 0.0,
            shift: (0, 0),
            shift_contrast: 0.0,
        };
    }

    let residual = mapped
        .iter()
        .zip(original.iter())
        .map(|(a, b)| {
            (0..3)
                .map(|channel| (a[channel] as f64 - b[channel] as f64).abs())
                .sum::<f64>()
                / 3.0
        })
        .sum::<f64>()
        / samples as f64;

    let luma_a: Vec<f64> = mapped.iter().map(|pixel| luma(*pixel)).collect();
    let luma_b: Vec<f64> = original.iter().map(|pixel| luma(*pixel)).collect();
    let mut differences: Vec<f64> = luma_a
        .iter()
        .zip(luma_b.iter())
        .map(|(a, b)| (a - b) * 255.0)
        .collect();
    differences.sort_by(|a, b| a.total_cmp(b));
    let bias = differences[differences.len() / 2];
    let mean_a = luma_a.iter().sum::<f64>() / samples as f64;
    let mean_b = luma_b.iter().sum::<f64>() / samples as f64;
    let mut covariance = 0.0;
    let mut variance_a = 0.0;
    let mut variance_b = 0.0;
    for (a, b) in luma_a.iter().zip(luma_b.iter()) {
        covariance += (a - mean_a) * (b - mean_b);
        variance_a += (a - mean_a).powi(2);
        variance_b += (b - mean_b).powi(2);
    }
    let correlation = if variance_a > 0.0 && variance_b > 0.0 {
        covariance / (variance_a.sqrt() * variance_b.sqrt())
    } else {
        0.0
    };

    let (shift, shift_contrast) = best_shift(&points, &luma_a, target, target_curve);
    Agreement {
        samples,
        residual,
        bias,
        correlation,
        shift,
        shift_contrast,
    }
}

/// Returns the shift with the lowest error and how distinct that minimum is.
fn best_shift(
    points: &[(f64, f64)],
    values: &[f64],
    target: &stitch_domain::RgbaImage,
    target_curve: &stitch_domain::ColorCurve,
) -> ((i32, i32), f64) {
    let mut best = (0i32, 0i32, f64::MAX);
    for dy in (-8..=8).step_by(2) {
        for dx in (-8..=8).step_by(2) {
            let error = shift_error(points, values, target, target_curve, dx, dy);
            if error < best.2 {
                best = (dx, dy, error);
            }
        }
    }
    let (center_x, center_y) = (best.0, best.1);
    for dy in (center_y - 2)..=(center_y + 2) {
        for dx in (center_x - 2)..=(center_x + 2) {
            let error = shift_error(points, values, target, target_curve, dx, dy);
            if error < best.2 {
                best = (dx, dy, error);
            }
        }
    }
    let ring = [
        (-8, 0),
        (8, 0),
        (0, -8),
        (0, 8),
        (-8, -8),
        (8, -8),
        (-8, 8),
        (8, 8),
    ];
    let ring_error = ring
        .iter()
        .map(|(dx, dy)| {
            shift_error(
                points,
                values,
                target,
                target_curve,
                best.0 + dx,
                best.1 + dy,
            )
        })
        .sum::<f64>()
        / ring.len() as f64;
    let contrast = if best.2 > 0.0 {
        (ring_error - best.2) / best.2
    } else {
        0.0
    };
    ((best.0, best.1), contrast)
}

fn shift_error(
    points: &[(f64, f64)],
    values: &[f64],
    target: &stitch_domain::RgbaImage,
    target_curve: &stitch_domain::ColorCurve,
    dx: i32,
    dy: i32,
) -> f64 {
    let max_x = target.width() as i32 - 1;
    let max_y = target.height() as i32 - 1;
    let mut total = 0.0;
    for (index, (px, py)) in points.iter().enumerate() {
        let x = (px.round() as i32 + dx).clamp(0, max_x) as u32;
        let y = (py.round() as i32 + dy).clamp(0, max_y) as u32;
        let pixel = target.pixel(x, y);
        let shifted = luma(target_curve.map([pixel[0], pixel[1], pixel[2]]));
        total += (values[index] - shifted).abs();
    }
    total / points.len().max(1) as f64
}

fn luma(pixel: [u8; 3]) -> f64 {
    (pixel[0] as f64 * 0.299 + pixel[1] as f64 * 0.587 + pixel[2] as f64 * 0.114) / 255.0
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
