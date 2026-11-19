//! Native entry point: wires the production adapters into the egui shell.

use std::sync::Arc;

use stitch_application::{ProjectService, StitchService, SystemClock};
use stitch_imaging::{memory_budget, BilinearCompositor, CacheDir, FileImageRepository};
use stitch_ui_egui::ScanToImgApp;
use stitch_vision::{RansacHomographyEstimator, RatioMatcher, SiftExtractor};

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let initial_paths: Vec<std::path::PathBuf> =
        std::env::args_os().skip(1).map(Into::into).collect();

    // Held until the app exits; dropping it removes the cache directory.
    let _cache = match CacheDir::create() {
        Ok(cache) => {
            log::debug!("cache directory: {}", cache.path().display());
            Some(cache)
        }
        Err(error) => {
            log::warn!("could not create the cache directory: {error}");
            None
        }
    };

    let repository = Arc::new(FileImageRepository::default());
    let budget = memory_budget();
    log::debug!("memory budget: {} MiB", budget >> 20);
    let project_service = ProjectService::new(repository).with_memory_budget(budget);

    let stitch_service = StitchService::new(
        Arc::new(SiftExtractor::default()),
        Arc::new(RatioMatcher::default()),
        Arc::new(RansacHomographyEstimator::default()),
        Arc::new(BilinearCompositor),
        Arc::new(SystemClock),
    );

    let mut viewport = eframe::egui::ViewportBuilder::default()
        .with_title("Scan to Image")
        .with_app_id("scan-to-img")
        .with_inner_size([1440.0, 900.0])
        .with_min_inner_size([960.0, 600.0]);
    if cfg!(target_os = "macos") {
        // A unified window: the toolbar sits in the title bar next to the
        // window controls, as in Finder, Photos and Preview.
        viewport = viewport
            .with_fullsize_content_view(true)
            .with_titlebar_shown(false)
            .with_title_shown(false);
    }

    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        multisampling: 0,
        persist_window: false,
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "Scan to Image",
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
