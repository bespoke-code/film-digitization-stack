use std::sync::Arc;

use stitch_application::{ProjectService, StitchService, SystemClock};
use stitch_domain::{ImageLocator, ImageRepository, Project, StitchOptions};
use stitch_imaging::{BilinearCompositor, FileImageRepository};
use stitch_vision::{RansacHomographyEstimator, RatioMatcher, SiftExtractor};

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().expect("output path");
    let inputs: Vec<ImageLocator> = args.map(ImageLocator::new).collect();

    let repository = Arc::new(FileImageRepository::default());
    let project_service = ProjectService::new(repository.clone());
    let stitch_service = StitchService::new(
        Arc::new(SiftExtractor::default()),
        Arc::new(RatioMatcher::default()),
        Arc::new(RansacHomographyEstimator::default()),
        Arc::new(BilinearCompositor),
        Arc::new(SystemClock),
    );

    let mut project = Project::new();
    project_service.add_images(&mut project, &inputs).unwrap();
    let result = stitch_service
        .stitch(project.images(), &StitchOptions::default(), &|_| {})
        .unwrap();
    println!(
        "{}x{} placements={} pairs={:?}",
        result.image.width(),
        result.image.height(),
        result.placements.len(),
        result
            .diagnostics
            .pairs
            .iter()
            .map(|p| (p.num_matches, p.num_inliers, p.used_for_alignment))
            .collect::<Vec<_>>()
    );
    repository
        .save(&ImageLocator::new(out), &result.image)
        .unwrap();
}
