//! Native entry point: wires the production adapters into the egui shell.

use std::sync::Arc;

use stitch_application::{ProjectService, StitchService, SystemClock};
use stitch_imaging::{BilinearCompositor, FileImageRepository};
use stitch_ui_egui::ScanToImgApp;
use stitch_vision::{RansacHomographyEstimator, RatioMatcher, SiftExtractor};

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let initial_paths: Vec<std::path::PathBuf> =
        std::env::args_os().skip(1).map(Into::into).collect();

    let repository = Arc::new(FileImageRepository::default());
    let project_service = ProjectService::new(repository);

    let stitch_service = StitchService::new(
        Arc::new(SiftExtractor::default()),
        Arc::new(RatioMatcher::default()),
        Arc::new(RansacHomographyEstimator::default()),
        Arc::new(BilinearCompositor),
        Arc::new(SystemClock),
    );

    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        multisampling: 0,
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Scan to image")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([960.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Scan to image",
        options,
        Box::new(move |cc| {
            Ok(Box::new(ScanToImgApp::new(
                cc,
                project_service,
                stitch_service,
                initial_paths,
            )))
        }),
    )
}
