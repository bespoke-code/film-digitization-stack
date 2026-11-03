//! egui adapter for the scan-to-img workbench.
//!
//! The adapter only renders [`Workbench`] state and forwards user intents.
//! Stitching runs on a background thread; the GPU preview goes through
//! [`stitch_renderer_wgpu::PreviewRenderer`] inside an egui-wgpu paint callback.

mod dialog;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread::JoinHandle;

use dialog::{
    folder_has_raw_images, images_in_folder, is_raw_image, is_supported_image, DialogMode,
    DialogOutcome, FileDialog,
};
use eframe::egui;
use stitch_application::{ProjectService, StitchService};
use stitch_domain::{
    CoreResult, CurvePoint, EditSettings, Homography, ImageId, ImageLocator, RgbaImage,
    SourceImage, StitchOptions, StitchProgress, StitchResult, ToneCurve,
};
use stitch_renderer_wgpu::PreviewRenderer;
use stitch_ui_core::{PreviewInfo, PreviewTarget, Workbench, WorkbenchStatus};

enum WorkerEvent {
    Progress(StitchProgress),
    Done(Box<CoreResult<StitchResult>>),
}

struct StitchWorker {
    receiver: Receiver<WorkerEvent>,
    handle: Option<JoinHandle<()>>,
}

struct Histogram {
    red: [u32; 256],
    green: [u32; 256],
    blue: [u32; 256],
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            red: [0; 256],
            green: [0; 256],
            blue: [0; 256],
        }
    }
}

impl Histogram {
    fn from_image(image: &RgbaImage) -> Self {
        let mut histogram = Self::default();
        let pixels = image.total_pixels();
        let stride = (pixels / 200_000).max(1);
        for (index, pixel) in image.pixels().as_chunks::<4>().0.iter().enumerate() {
            if index % stride != 0 {
                continue;
            }
            histogram.red[pixel[0] as usize] += 1;
            histogram.green[pixel[1] as usize] += 1;
            histogram.blue[pixel[2] as usize] += 1;
        }
        histogram
    }

    fn peak(&self) -> u32 {
        self.red
            .iter()
            .chain(self.green.iter())
            .chain(self.blue.iter())
            .copied()
            .max()
            .unwrap_or(1)
            .max(1)
    }
}

#[derive(Default)]
struct HistogramCache {
    key: Option<(u64, String)>,
    histogram: Histogram,
}

impl HistogramCache {
    fn update(&mut self, preview: &PreviewInfo) {
        let key = (preview.revision, preview.label.clone());
        if self.key.as_ref() != Some(&key) {
            self.histogram = Histogram::from_image(&preview.image);
            self.key = Some(key);
        }
    }
}

struct PreviewState {
    zoom: f32,
    pan: egui::Vec2,
    key: Option<(PreviewTarget, u32, u32)>,
}

impl Default for PreviewState {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            key: None,
        }
    }
}

struct PreviewCallback {
    image: Option<std::sync::Arc<RgbaImage>>,
    revision: u64,
}

impl egui_wgpu::CallbackTrait for PreviewCallback {
    fn prepare(
        &self,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut egui_wgpu::wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<egui_wgpu::wgpu::CommandBuffer> {
        if let Some(renderer) = resources.get_mut::<PreviewRenderer>() {
            renderer.prepare(
                device,
                queue,
                self.image.as_ref(),
                self.revision,
                [-1.0, 1.0, 1.0, -1.0],
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        pass: &mut egui_wgpu::wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(renderer) = resources.get::<PreviewRenderer>() {
            // egui-wgpu sets the viewport but not the scissor for callbacks.
            // Clip the image to the central panel so a zoomed image cannot
            // paint over the side panels.
            let clip = info.clip_rect_in_pixels();
            let screen = [info.screen_size_px[0] as i32, info.screen_size_px[1] as i32];
            let min_x = clip.left_px.max(0);
            let min_y = clip.top_px.max(0);
            let max_x = (clip.left_px + clip.width_px).min(screen[0]);
            let max_y = (clip.top_px + clip.height_px).min(screen[1]);
            if max_x > min_x && max_y > min_y {
                pass.set_scissor_rect(
                    min_x as u32,
                    min_y as u32,
                    (max_x - min_x) as u32,
                    (max_y - min_y) as u32,
                );
                renderer.paint(pass);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LevelChannel {
    Red,
    Green,
    Blue,
    Gray,
}

/// Default horizontal overlap when a rough layout starts.
const ARRANGE_OVERLAP: f32 = 0.15;

struct ArrangeItem {
    id: ImageId,
    name: String,
    /// Top-left corner in reference-image pixels.
    position: egui::Vec2,
    /// Source image size in pixels.
    size: egui::Vec2,
    texture: egui::TextureHandle,
}

struct ArrangeState {
    items: Vec<ArrangeItem>,
    scale: f32,
    scene: egui::Rect,
    drag: Option<usize>,
}

#[derive(Default)]
struct ArrangeAction {
    reset: bool,
    confirm: bool,
}

impl ArrangeState {
    fn reset_layout(&mut self) {
        self.drag = None;
        let first_height = self.items.first().map(|item| item.size.y).unwrap_or(0.0);
        let mut cursor = 0.0;
        for item in &mut self.items {
            let y = ((first_height - item.size.y) * 0.5).max(0.0);
            item.position = egui::vec2(cursor, y);
            cursor = item.position.x + item.size.x * (1.0 - ARRANGE_OVERLAP);
        }
        self.rebuild_scene();
    }

    fn rebuild_scene(&mut self) {
        let mut bounds = egui::Rect::NOTHING;
        for item in &self.items {
            let rect =
                egui::Rect::from_min_size(egui::pos2(item.position.x, item.position.y), item.size);
            bounds = bounds.union(rect);
        }
        if !bounds.is_positive() {
            bounds = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1.0, 1.0));
        }
        self.scene = bounds.expand2(egui::vec2(
            bounds.width() * 0.25 + 200.0,
            bounds.height() * 0.25 + 200.0,
        ));
    }
}

/// The egui application shell.
pub struct ScanToImgApp {
    workbench: Workbench,
    project_service: ProjectService,
    stitch_service: StitchService,
    worker: Option<StitchWorker>,
    dialog: FileDialog,
    thumbnails: HashMap<ImageId, egui::TextureHandle>,
    options: StitchOptions,
    preview: PreviewState,
    histogram: HistogramCache,
    gpu_available: bool,
    pending_paths: Vec<PathBuf>,
    picking_film_base: bool,
    level_channel: LevelChannel,
    tone_open: bool,
    curve_drag: Option<usize>,
    crop_mode: bool,
    crop_start: Option<egui::Pos2>,
    crop_end: Option<egui::Pos2>,
    arrange_open: bool,
    arrange: Option<ArrangeState>,
}

impl ScanToImgApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        project_service: ProjectService,
        stitch_service: StitchService,
        pending_paths: Vec<PathBuf>,
    ) -> Self {
        let gpu_available = if let Some(state) = cc.wgpu_render_state.as_ref() {
            let renderer = PreviewRenderer::new(&state.device, state.target_format, 1);
            state.renderer.write().callback_resources.insert(renderer);
            true
        } else {
            log::error!("wgpu render state is unavailable; the preview will be disabled");
            false
        };

        let options = StitchOptions::default();
        Self {
            workbench: Workbench::new(),
            project_service,
            stitch_service,
            worker: None,
            dialog: FileDialog::default(),
            thumbnails: HashMap::new(),
            options,
            preview: PreviewState::default(),
            histogram: HistogramCache::default(),
            gpu_available,
            pending_paths,
            picking_film_base: false,
            level_channel: LevelChannel::Red,
            tone_open: false,
            curve_drag: None,
            crop_mode: false,
            crop_start: None,
            crop_end: None,
            arrange_open: false,
            arrange: None,
        }
    }

    fn poll_worker(&mut self) {
        let mut events = Vec::new();
        let mut disconnected = false;
        if let Some(worker) = &self.worker {
            loop {
                match worker.receiver.try_recv() {
                    Ok(event) => events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }

        let mut finished = false;
        for event in events {
            match event {
                WorkerEvent::Progress(progress) => self.workbench.report_progress(progress),
                WorkerEvent::Done(result) => {
                    match *result {
                        Ok(result) => self.workbench.finish_stitch(result),
                        Err(error) => self.workbench.fail_stitch(error.to_string()),
                    }
                    finished = true;
                }
            }
        }
        if disconnected && !finished {
            self.workbench
                .fail_stitch("The stitching worker stopped unexpectedly.");
            finished = true;
        }
        if finished {
            if let Some(mut worker) = self.worker.take() {
                if let Some(handle) = worker.handle.take() {
                    let _ = handle.join();
                }
            }
        }
    }

    fn start_stitch(&mut self, ctx: &egui::Context) {
        let job = match self.workbench.begin_stitch() {
            Ok(job) => job,
            Err(error) => {
                self.workbench.fail_stitch(error.to_string());
                return;
            }
        };
        let service = self.stitch_service.clone();
        let (sender, receiver) = mpsc::channel();
        let progress_sender = sender.clone();
        let progress_context = ctx.clone();
        let finish_context = ctx.clone();
        let handle = std::thread::spawn(move || {
            let progress = move |progress: StitchProgress| {
                let _ = progress_sender.send(WorkerEvent::Progress(progress));
                progress_context.request_repaint();
            };
            let result =
                service.stitch_with_manual(&job.images, &job.manual, &job.options, &progress);
            let _ = sender.send(WorkerEvent::Done(Box::new(result)));
            finish_context.request_repaint();
        });
        self.worker = Some(StitchWorker {
            receiver,
            handle: Some(handle),
        });
    }

    fn add_paths(&mut self, paths: Vec<PathBuf>) {
        let mut images = Vec::new();
        let mut raw_files = 0usize;
        for path in paths {
            if path.is_dir() {
                images.extend(images_in_folder(&path));
            } else if is_supported_image(&path) {
                images.push(path);
            } else if is_raw_image(&path) {
                raw_files += 1;
            }
        }
        if images.is_empty() {
            if raw_files > 0 {
                self.workbench.notify(
                    "RAW files are not supported yet. Convert them to TIFF, JPEG or PNG first.",
                );
            } else {
                self.workbench
                    .notify("No supported images (PNG, JPEG, WebP, BMP, TIFF) were found.");
            }
            return;
        }
        let locators: Vec<ImageLocator> = images
            .iter()
            .map(|path| ImageLocator::from(path.as_path()))
            .collect();
        if let Err(error) = self.workbench.add_images(&self.project_service, &locators) {
            self.workbench.fail_stitch(error.to_string());
        } else {
            self.arrange = None;
            self.arrange_open = false;
        }
    }

    fn remove_selected(&mut self) {
        if let Some(id) = self.workbench.selected() {
            if let Err(error) = self.workbench.remove_image(id) {
                self.workbench.fail_stitch(error.to_string());
            } else {
                self.arrange = None;
                self.arrange_open = false;
            }
        }
    }

    fn default_export_name(&self) -> String {
        let image = match self.workbench.preview_target() {
            PreviewTarget::Image(id) => self.workbench.project().get(id),
            PreviewTarget::Stitched => self.workbench.project().images().first(),
        };
        let stem = image
            .and_then(|image| Path::new(image.locator.as_str()).file_stem())
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "panorama".to_owned());
        match self.workbench.preview_target() {
            PreviewTarget::Stitched => format!("{stem}-stitched.png"),
            PreviewTarget::Image(_) => format!("{stem}-edited.png"),
        }
    }

    fn default_export_dir(&self) -> Option<PathBuf> {
        self.workbench
            .project()
            .images()
            .first()
            .and_then(|image| Path::new(image.locator.as_str()).parent())
            .map(Path::to_path_buf)
    }

    fn handle_dialog_outcome(&mut self, outcome: DialogOutcome) {
        match outcome {
            DialogOutcome::AddImages(paths) => self.add_paths(paths),
            DialogOutcome::OpenFolder(folder) => {
                let images = images_in_folder(&folder);
                if images.is_empty() {
                    if folder_has_raw_images(&folder) {
                        self.workbench.notify(
                            "RAW files are not supported yet. Convert them to TIFF, JPEG or PNG first.",
                        );
                    } else {
                        self.workbench.fail_stitch(format!(
                            "No supported images were found in {}.",
                            folder.display()
                        ));
                    }
                } else {
                    self.add_paths(images);
                }
            }
            DialogOutcome::SaveImage(path) => {
                let locator = ImageLocator::from(path.as_path());
                match self.workbench.export(&self.project_service, &locator) {
                    Ok(()) => self
                        .workbench
                        .notify(format!("Exported to {}", path.display())),
                    Err(error) => self.workbench.fail_stitch(error.to_string()),
                }
            }
            DialogOutcome::SaveFolder(folder, format) => {
                match self
                    .workbench
                    .export_all(&self.project_service, &folder, format.extension())
                {
                    Ok(count) => self.workbench.notify(format!(
                        "Exported {count} image{} to {}.",
                        if count == 1 { "" } else { "s" },
                        folder.display()
                    )),
                    Err(error) => self.workbench.fail_stitch(error.to_string()),
                }
            }
        }
    }

    fn thumbnail(&mut self, ctx: &egui::Context, image: &SourceImage) -> egui::TextureHandle {
        if let Some(texture) = self.thumbnails.get(&image.id) {
            return texture.clone();
        }
        let thumbnail = image.thumbnail.as_ref();
        let color = egui::ColorImage::from_rgba_unmultiplied(
            [thumbnail.width() as usize, thumbnail.height() as usize],
            thumbnail.pixels(),
        );
        let texture = ctx.load_texture(
            format!("thumbnail-{}", image.id.0),
            color,
            egui::TextureOptions::LINEAR,
        );
        self.thumbnails.insert(image.id, texture.clone());
        texture
    }

    fn toolbar(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.heading("Scan to image");
            ui.separator();

            let add_label = format!("Add images ({} left)", self.workbench.remaining_slots());
            if ui
                .add_enabled(self.workbench.can_add(), egui::Button::new(add_label))
                .clicked()
            {
                self.dialog
                    .open_in(DialogMode::AddImages, None, String::new());
            }
            if ui.button("Open folder").clicked() {
                self.dialog
                    .open_in(DialogMode::OpenFolder, None, String::new());
            }
            if ui
                .add_enabled(
                    self.workbench.selected().is_some(),
                    egui::Button::new("Remove"),
                )
                .clicked()
            {
                self.remove_selected();
            }
            if ui
                .add_enabled(
                    self.workbench.selected().is_some() && !self.workbench.is_busy(),
                    egui::Button::new("Move up"),
                )
                .clicked()
            {
                if let Some(id) = self.workbench.selected() {
                    if self.workbench.move_up(id).is_ok() {
                        self.arrange = None;
                        self.arrange_open = false;
                    }
                }
            }
            if ui
                .add_enabled(
                    self.workbench.selected().is_some() && !self.workbench.is_busy(),
                    egui::Button::new("Move down"),
                )
                .clicked()
            {
                if let Some(id) = self.workbench.selected() {
                    if self.workbench.move_down(id).is_ok() {
                        self.arrange = None;
                        self.arrange_open = false;
                    }
                }
            }
            ui.separator();
            if ui
                .add_enabled(
                    self.workbench.can_stitch(),
                    egui::Button::new("Stitch panorama"),
                )
                .clicked()
            {
                self.start_stitch(ctx);
            }
            if ui
                .add_enabled(
                    self.workbench.project().len() > 1 && !self.workbench.is_busy(),
                    egui::Button::new("Rough arrange…"),
                )
                .clicked()
            {
                self.open_arrange(ctx);
            }
            if ui
                .add_enabled(self.workbench.can_export(), egui::Button::new("Export…"))
                .clicked()
            {
                self.dialog.open_in(
                    DialogMode::SaveImage,
                    self.default_export_dir(),
                    self.default_export_name(),
                );
            }
            if ui
                .add_enabled(
                    !self.workbench.project().is_empty() && !self.workbench.is_busy(),
                    egui::Button::new("Export all…"),
                )
                .clicked()
            {
                self.dialog.open_in(
                    DialogMode::SaveFolder,
                    self.default_export_dir(),
                    String::new(),
                );
            }
            if ui.button("Clear").clicked() {
                self.workbench.clear();
                self.thumbnails.clear();
                self.arrange = None;
                self.arrange_open = false;
            }
        });
    }

    fn image_list(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.heading("Images");
        ui.weak(format!(
            "{} of {} selected",
            self.workbench.project().len(),
            stitch_domain::MAX_IMAGES
        ));
        ui.separator();

        let images = self.workbench.project().images().to_vec();
        let selected = self.workbench.selected();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if images.is_empty() {
                    ui.label("Drop images here, or use Add images.");
                    return;
                }
                for image in &images {
                    let texture = self.thumbnail(ctx, image);
                    let is_selected = selected == Some(image.id);
                    ui.horizontal(|ui| {
                        let size = egui::vec2(56.0, 42.0);
                        let scaled = fit_size(texture.size_vec2(), size);
                        ui.image((texture.id(), scaled));
                        ui.vertical(|ui| {
                            if ui.selectable_label(is_selected, &image.name).clicked() {
                                let _ = self.workbench.select_image(image.id);
                                self.picking_film_base = false;
                            }
                            ui.weak(format!(
                                "{} × {}",
                                image.image.width(),
                                image.image.height()
                            ));
                        });
                    });
                    ui.separator();
                }
            });
    }

    fn diagnostics(&self, ui: &mut egui::Ui) {
        ui.heading("Diagnostics");
        match self.workbench.result() {
            Some(result) => {
                let diagnostics = &result.diagnostics;
                ui.label(format!(
                    "Output: {} × {} at {:.0}% scale",
                    diagnostics.output_width,
                    diagnostics.output_height,
                    diagnostics.output_scale * 100.0
                ));
                ui.label(format!(
                    "Duration: {:.2}s",
                    diagnostics.duration_ms as f64 / 1000.0
                ));
                ui.label(format!(
                    "Features: {}",
                    diagnostics
                        .features_per_image
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                ui.separator();
                egui::Grid::new("pair-diagnostics")
                    .num_columns(4)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Pair");
                        ui.strong("Matches");
                        ui.strong("Inliers");
                        ui.strong("Used");
                        ui.end_row();
                        for pair in &diagnostics.pairs {
                            ui.label(format!("#{} → #{}", pair.from.0, pair.to.0));
                            ui.label(pair.num_matches.to_string());
                            ui.label(format!("{} ({:.2}px)", pair.num_inliers, pair.rms_error));
                            ui.label(if pair.used_for_alignment { "yes" } else { "no" });
                            ui.end_row();
                        }
                    });
            }
            None => {
                ui.weak("Stitch a project to see registration diagnostics.");
            }
        }
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Settings");
        ui.add(egui::Slider::new(&mut self.options.match_ratio, 0.5..=0.95).text("Match ratio"));
        ui.add(egui::Slider::new(&mut self.options.min_inliers, 6..=200).text("Minimum inliers"));
        ui.add(
            egui::Slider::new(&mut self.options.ransac_threshold, 0.5..=20.0)
                .text("RANSAC threshold (px)"),
        );
        ui.add(
            egui::Slider::new(&mut self.options.max_analysis_dimension, 320..=4096)
                .text("Analysis size (px)"),
        );
        ui.add(
            egui::Slider::new(&mut self.options.max_features, 100..=8000)
                .text("Features per image"),
        );
        ui.add(egui::Slider::new(&mut self.options.feather_fraction, 0.0..=0.5).text("Feather"));
        ui.checkbox(
            &mut self.options.normalize_colors,
            "Normalize color and exposure across frames",
        );
        ui.add(
            egui::Slider::new(&mut self.options.max_canvas_dimension, 512..=16_384)
                .text("Max canvas side (px)"),
        );
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    *self.workbench.options() != self.options,
                    egui::Button::new("Apply"),
                )
                .clicked()
            {
                self.workbench.update_options(self.options.clone());
            }
            if ui.button("Reset").clicked() {
                self.options = StitchOptions::default();
            }
        });
        ui.weak("Settings changes discard the current panorama.");
    }

    fn develop(&mut self, ui: &mut egui::Ui) {
        ui.heading("Develop");
        if self.workbench.base_preview().is_none() {
            ui.weak("Select an image or stitch a panorama to edit it.");
            return;
        }

        let mut edits = self.workbench.current_edits();
        let mut changed = false;
        changed |= ui.checkbox(&mut edits.invert, "Invert negative").changed();
        changed |= ui.checkbox(&mut edits.monochrome, "Monochrome").changed();

        ui.horizontal(|ui| {
            ui.label("Film base");
            match edits.film_base {
                Some(base) => draw_swatch(ui, base),
                None => {
                    ui.weak("none");
                }
            }
            if ui
                .selectable_label(self.picking_film_base, "Pick")
                .clicked()
            {
                self.picking_film_base = !self.picking_film_base;
            }
            if ui.button("Auto").clicked() {
                if let Some(preview) = self.workbench.base_preview() {
                    edits.film_base = Some(stitch_domain::suggest_film_base(&preview.image));
                    changed = true;
                }
            }
            if edits.film_base.is_some() && ui.button("Clear").clicked() {
                edits.film_base = None;
                changed = true;
            }
        });
        if self.picking_film_base {
            ui.weak("Click the orange film base in the preview.");
        }

        changed |= ui
            .add(egui::Slider::new(&mut edits.exposure, -6.0..=6.0).text("Exposure (EV)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut edits.contrast, -0.95..=0.95).text("Contrast"))
            .changed();

        if ui.button("Levels & curves…").clicked() {
            self.tone_open = true;
        }

        if changed {
            self.workbench.update_current_edits(edits.clone());
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.crop_mode, "Crop")
                .on_hover_text("Drag a rectangle on the preview")
                .clicked()
            {
                self.crop_mode = !self.crop_mode;
                self.crop_start = None;
                self.crop_end = None;
                self.picking_film_base = false;
            }
            if edits.crop.is_some() && ui.button("Reset crop").clicked() {
                edits.crop = None;
                changed = true;
            }
        });
        if let Some(crop) = edits.crop {
            ui.weak(format!(
                "Crop {} × {} at {}, {}",
                crop.width, crop.height, crop.x, crop.y
            ));
        }
        if self.crop_mode {
            ui.weak("Drag a rectangle on the preview to crop.");
        }
        if changed {
            self.workbench.update_current_edits(edits.clone());
        }

        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    self.workbench.project().len() > 1 || self.workbench.result().is_some(),
                    egui::Button::new("Apply to all"),
                )
                .on_hover_text("Copy tone and color settings to every photo and the panorama")
                .clicked()
            {
                self.workbench.apply_edits_to_all(edits.clone());
            }
            if ui.button("Reset").clicked() {
                self.workbench.reset_current_edits();
            }
        });

        let mut show = self.workbench.show_edits();
        if ui.checkbox(&mut show, "Preview edits").changed() {
            self.workbench.set_show_edits(show);
        }
    }

    fn open_arrange(&mut self, ctx: &egui::Context) {
        let images = self.workbench.project().images().to_vec();
        if images.len() < 2 {
            self.workbench
                .notify("Add at least two images before arranging them.");
            return;
        }
        let manual = self.workbench.manual_placements();
        let first_height = images[0].image.height() as f32;
        let mut cursor = 0.0f32;
        let mut items = Vec::with_capacity(images.len());
        for (index, image) in images.iter().enumerate() {
            let size = egui::vec2(image.image.width() as f32, image.image.height() as f32);
            let position = match manual.get(index).copied().flatten() {
                Some(placement) => egui::vec2(placement.m[2] as f32, placement.m[5] as f32),
                None => egui::vec2(cursor, ((first_height - size.y) * 0.5).max(0.0)),
            };
            cursor = position.x + size.x * (1.0 - ARRANGE_OVERLAP);
            items.push(ArrangeItem {
                id: image.id,
                name: image.name.clone(),
                position,
                size,
                texture: self.thumbnail(ctx, image),
            });
        }
        let max_side = images
            .iter()
            .map(|image| image.image.width().max(image.image.height()))
            .max()
            .unwrap_or(1) as f32;
        let scale = (320.0 / max_side).clamp(0.01, 1.0);
        let mut state = ArrangeState {
            items,
            scale,
            scene: egui::Rect::NOTHING,
            drag: None,
        };
        state.rebuild_scene();
        self.arrange = Some(state);
        self.arrange_open = true;
    }

    fn apply_arrangement(&mut self, ctx: &egui::Context) {
        let Some(state) = self.arrange.take() else {
            return;
        };
        self.workbench.clear_manual_placements();
        for item in state.items.iter().skip(1) {
            self.workbench.set_manual_placement(
                item.id,
                Homography::translation(item.position.x as f64, item.position.y as f64),
            );
        }
        self.arrange_open = false;
        self.start_stitch(ctx);
    }

    fn arrange_window(&mut self, ctx: &egui::Context) {
        if !self.arrange_open {
            return;
        }
        let mut open = true;
        let mut action = ArrangeAction::default();
        egui::Window::new("Rough arrange")
            .open(&mut open)
            .default_size(egui::vec2(860.0, 580.0))
            .resizable(true)
            .show(ctx, |ui| {
                ui.weak(
                    "Drag frames until the overlap looks right. Frame 1 is the reference and stays fixed. \
                     The stitcher still refines every frame automatically and uses these placements only when no match is found.",
                );
                ui.horizontal(|ui| {
                    if ui.button("Reset layout").clicked() {
                        action.reset = true;
                    }
                    if ui.button("Apply and stitch").clicked() {
                        action.confirm = true;
                    }
                });
                ui.separator();
                if let Some(state) = self.arrange.as_mut() {
                    if action.reset {
                        state.reset_layout();
                    }
                    draw_arrange(ui, state);
                }
            });
        self.arrange_open = open && !action.confirm;
        if action.confirm {
            self.apply_arrangement(ctx);
        } else if !self.arrange_open {
            self.arrange = None;
        }
    }

    fn tone_window(&mut self, ctx: &egui::Context) {
        if !self.tone_open {
            return;
        }
        let mut open = self.tone_open;
        let mut edits = self.workbench.current_edits();
        let mut changed = false;
        egui::Window::new("Levels & curves")
            .open(&mut open)
            .default_width(320.0)
            .resizable(false)
            .show(ctx, |ui| {
                changed = self.tone_controls(ui, &mut edits);
            });
        self.tone_open = open;
        if changed {
            self.workbench.update_current_edits(edits);
        }
    }

    fn tone_controls(&mut self, ui: &mut egui::Ui, edits: &mut EditSettings) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.level_channel, LevelChannel::Red, "R");
            ui.selectable_value(&mut self.level_channel, LevelChannel::Green, "G");
            ui.selectable_value(&mut self.level_channel, LevelChannel::Blue, "B");
            ui.selectable_value(&mut self.level_channel, LevelChannel::Gray, "Gray");
        });

        let levels = match self.level_channel {
            LevelChannel::Red => &mut edits.levels.red,
            LevelChannel::Green => &mut edits.levels.green,
            LevelChannel::Blue => &mut edits.levels.blue,
            LevelChannel::Gray => &mut edits.levels.gray,
        };
        changed |= ui
            .add(
                egui::Slider::new(&mut levels.black, 0.0..=0.9)
                    .text("Black")
                    .fixed_decimals(3),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut levels.white, 0.1..=1.0)
                    .text("White")
                    .fixed_decimals(3),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut levels.gamma, 0.1..=4.0).text("Gamma"))
            .changed();

        let curve = match self.level_channel {
            LevelChannel::Red => &mut edits.curves.red,
            LevelChannel::Green => &mut edits.curves.green,
            LevelChannel::Blue => &mut edits.curves.blue,
            LevelChannel::Gray => &mut edits.curves.gray,
        };
        ui.separator();
        ui.horizontal(|ui| {
            ui.strong("Tone curve");
            if ui.button("Reset curve").clicked() {
                curve.reset();
                changed = true;
            }
        });
        let color = match self.level_channel {
            LevelChannel::Red => egui::Color32::from_rgb(230, 90, 90),
            LevelChannel::Green => egui::Color32::from_rgb(90, 220, 120),
            LevelChannel::Blue => egui::Color32::from_rgb(100, 150, 240),
            LevelChannel::Gray => egui::Color32::from_gray(210),
        };
        changed |= curve_editor(ui, curve, &mut self.curve_drag, color);
        ui.weak("Click adds a point. Drag moves it. Right-click removes an inner point.");
        changed
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        let status = self.workbench.status().clone();
        ui.horizontal(|ui| match status {
            WorkbenchStatus::Empty => {
                ui.weak("Add up to sixteen overlapping images, then stitch.");
            }
            WorkbenchStatus::Ready(message) => {
                ui.label(message);
            }
            WorkbenchStatus::Busy {
                stage,
                fraction,
                message,
            } => {
                ui.spinner();
                ui.label(format!("{}: {message}", stage.label()));
                if let Some(fraction) = fraction {
                    ui.add(
                        egui::ProgressBar::new(fraction)
                            .desired_width(220.0)
                            .show_percentage(),
                    );
                }
            }
            WorkbenchStatus::Failed(message) => {
                ui.colored_label(egui::Color32::from_rgb(230, 110, 110), message);
                if ui.small_button("Dismiss").clicked() {
                    self.workbench.acknowledge_error();
                }
            }
        });
    }

    fn preview_area(&mut self, ui: &mut egui::Ui) {
        let preview = if self.crop_mode {
            self.workbench.base_preview()
        } else {
            self.workbench.preview()
        };
        let Some(preview) = preview else {
            ui.centered_and_justified(|ui| {
                ui.weak("Add images to start. Drag and drop works too.");
            });
            return;
        };

        let image_size = egui::vec2(preview.image.width() as f32, preview.image.height() as f32);
        let key = (
            self.workbench.preview_target(),
            preview.image.width(),
            preview.image.height(),
        );
        if self.preview.key != Some(key) {
            self.preview.key = Some(key);
            self.preview.zoom = 1.0;
            self.preview.pan = egui::Vec2::ZERO;
        }

        ui.horizontal(|ui| {
            ui.label(&preview.label);
            let showing_original = !self.workbench.show_edits();
            if ui
                .selectable_label(showing_original, "Original")
                .on_hover_text("Show the unedited photo")
                .clicked()
            {
                self.workbench.set_show_edits(showing_original);
            }
            if self.workbench.result().is_some() && ui.button("Show panorama").clicked() {
                let _ = self.workbench.show_stitched();
                self.picking_film_base = false;
            }
            ui.separator();
            if ui.button("−").clicked() {
                self.preview.zoom = (self.preview.zoom / 1.25).max(0.1);
            }
            ui.label(format!("{:.0}%", self.preview.zoom * 100.0));
            if ui.button("+").clicked() {
                self.preview.zoom = (self.preview.zoom * 1.25).min(20.0);
            }
            if ui.button("Fit").clicked() {
                self.preview.zoom = 1.0;
                self.preview.pan = egui::Vec2::ZERO;
            }
            ui.weak(format!(
                "{} × {}",
                preview.image.width(),
                preview.image.height()
            ));
        });

        let available = ui.available_size();
        let (response, painter) = ui.allocate_painter(available, egui::Sense::click_and_drag());
        let viewport = response.rect;
        painter.rect_filled(viewport, 0.0, egui::Color32::from_gray(24));
        painter.rect_stroke(
            viewport,
            0.0,
            egui::Stroke::new(1.0_f32, egui::Color32::from_gray(60)),
            egui::StrokeKind::Inside,
        );

        let fit = (viewport.width() / image_size.x)
            .min(viewport.height() / image_size.y)
            .max(0.0001);
        let scale = fit * self.preview.zoom;
        let display = image_size * scale;
        let center = viewport.center() + self.preview.pan;
        let image_rect = egui::Rect::from_center_size(center, display);

        if self.crop_mode {
            if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            }
            if response.drag_started() {
                self.crop_start = response
                    .interact_pointer_pos()
                    .filter(|position| image_rect.contains(*position));
                self.crop_end = self.crop_start;
            }
            if response.dragged() && self.crop_start.is_some() {
                self.crop_end = response.interact_pointer_pos().or(self.crop_end);
            }
            if let (Some(start), Some(end)) = (self.crop_start, self.crop_end) {
                let rect = egui::Rect::from_two_pos(start, end).intersect(image_rect);
                if rect.is_positive() {
                    painter.rect_filled(rect, 0.0, egui::Color32::from_white_alpha(30));
                    painter.rect_stroke(
                        rect,
                        0.0,
                        egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(255, 200, 60)),
                        egui::StrokeKind::Inside,
                    );
                }
            }
            if response.drag_stopped() {
                let start = self.crop_start.take();
                let end = self
                    .crop_end
                    .take()
                    .or_else(|| response.interact_pointer_pos());
                if let (Some(start), Some(end)) = (start, end) {
                    if let Some(crop) = crop_from_screen(&preview.image, start, end, image_rect) {
                        let mut edits = self.workbench.current_edits();
                        edits.crop = Some(crop);
                        self.workbench.update_current_edits(edits);
                        self.crop_mode = false;
                    }
                }
            }
        } else if response.dragged() {
            self.preview.pan += response.drag_delta();
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                let factor = (scroll * 0.002).exp();
                self.preview.zoom = (self.preview.zoom * factor).clamp(0.1, 20.0);
            }
        }

        if self.picking_film_base && !self.crop_mode && response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        }
        if self.picking_film_base && !self.crop_mode && response.clicked() {
            if let Some(position) = response.interact_pointer_pos() {
                if image_rect.contains(position) {
                    let normalized_x = (position.x - image_rect.left()) / image_rect.width();
                    let normalized_y = (position.y - image_rect.top()) / image_rect.height();
                    if let Some(base) = self.workbench.base_preview() {
                        let settings = self.workbench.current_edits();
                        let crop = settings
                            .crop
                            .filter(|_| self.workbench.show_edits())
                            .and_then(|crop| crop.clamped(base.image.width(), base.image.height()));
                        let (x, y) = match crop {
                            Some(crop) => (
                                crop.x as f32 + normalized_x * crop.width as f32,
                                crop.y as f32 + normalized_y * crop.height as f32,
                            ),
                            None => (
                                normalized_x * base.image.width() as f32,
                                normalized_y * base.image.height() as f32,
                            ),
                        };
                        let x = x.clamp(0.0, base.image.width().saturating_sub(1) as f32) as u32;
                        let y = y.clamp(0.0, base.image.height().saturating_sub(1) as f32) as u32;
                        let pixel = base.image.pixel(x, y);
                        let mut edits = settings;
                        edits.film_base = Some([pixel[0], pixel[1], pixel[2]]);
                        self.workbench.update_current_edits(edits);
                        self.picking_film_base = false;
                    }
                }
            }
        }

        let visible = image_rect.intersect(viewport);
        if visible.is_positive() {
            painter.add(egui_wgpu::Callback::new_paint_callback(
                image_rect,
                PreviewCallback {
                    image: Some(preview.image.clone()),
                    revision: preview.revision,
                },
            ));
        }
    }
}

impl eframe::App for ScanToImgApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_worker();
        if !self.pending_paths.is_empty() {
            let paths = std::mem::take(&mut self.pending_paths);
            self.add_paths(paths);
        }
        let dropped: Vec<PathBuf> = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .map(|file| file.path().to_path_buf())
                .collect()
        });
        if !dropped.is_empty() {
            self.add_paths(dropped);
        }

        if let Some(outcome) = self.dialog.show(&ctx) {
            self.handle_dialog_outcome(outcome);
        }
        self.tone_window(&ctx);
        self.arrange_window(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| {
            self.toolbar(&ctx, ui);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            self.status_bar(ui);
        });
        egui::Panel::left("images")
            .resizable(true)
            .default_size(280.0)
            .size_range(220.0..=420.0)
            .show(ui, |ui| {
                self.image_list(&ctx, ui);
            });
        egui::Panel::right("inspector")
            .resizable(true)
            .default_size(320.0)
            .size_range(260.0..=460.0)
            .show(ui, |ui| {
                self.develop(ui);
                ui.separator();
                if let Some(preview) = self.workbench.preview() {
                    self.histogram.update(&preview);
                    ui.heading("Histogram");
                    draw_histogram(ui, &self.histogram.histogram);
                }
                ui.separator();
                egui::CollapsingHeader::new("Stitch settings")
                    .default_open(false)
                    .show(ui, |ui| self.settings(ui));
                egui::CollapsingHeader::new("Diagnostics")
                    .default_open(true)
                    .show(ui, |ui| self.diagnostics(ui));
            });
        egui::CentralPanel::default().show(ui, |ui| {
            if self.gpu_available {
                self.preview_area(ui);
            } else {
                ui.centered_and_justified(|ui| {
                    ui.colored_label(
                        egui::Color32::from_rgb(230, 110, 110),
                        "GPU preview unavailable. Start the app with a wgpu-capable backend.",
                    );
                });
            }
        });
    }
}

fn draw_arrange(ui: &mut egui::Ui, state: &mut ArrangeState) {
    let viewport = ui.available_size();
    let scale = state.scale;
    let scene = state.scene;
    let desired = egui::vec2(
        (scene.width() * scale).max(viewport.x.max(320.0)),
        (scene.height() * scale).max(viewport.y.max(320.0)),
    );
    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let (response, painter) = ui.allocate_painter(desired, egui::Sense::hover());
            let canvas = response.rect;
            painter.rect_filled(canvas, 0.0, egui::Color32::from_gray(22));
            let scene_px = scene.size() * scale;
            let origin =
                canvas.min + ((canvas.size() - scene_px) * 0.5) - scene.min.to_vec2() * scale;

            for index in 0..state.items.len() {
                let (id, position, size) = {
                    let item = &state.items[index];
                    (item.id, item.position, item.size)
                };
                let rect = egui::Rect::from_min_size(origin + position * scale, size * scale);
                let sense = if index == 0 {
                    egui::Sense::hover()
                } else {
                    egui::Sense::click_and_drag()
                };
                let interaction =
                    ui.interact(rect, ui.scope_id().with(("arrange-frame", id.0)), sense);
                if index > 0 {
                    if interaction.drag_started() {
                        state.drag = Some(index);
                    }
                    if interaction.dragged() && state.drag == Some(index) {
                        state.items[index].position += interaction.drag_delta() / scale;
                    }
                    if interaction.drag_stopped() && state.drag == Some(index) {
                        state.drag = None;
                    }
                    if interaction.hovered() {
                        ui.ctx().set_cursor_icon(if interaction.dragged() {
                            egui::CursorIcon::Grabbing
                        } else {
                            egui::CursorIcon::Grab
                        });
                    }
                }

                let item = &state.items[index];
                painter.image(
                    item.texture.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
                let stroke = if state.drag == Some(index) {
                    egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(255, 200, 60))
                } else if index == 0 {
                    egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(90, 170, 240))
                } else {
                    egui::Stroke::new(1.0_f32, egui::Color32::from_gray(120))
                };
                painter.rect_stroke(rect, 2.0, stroke, egui::StrokeKind::Inside);
                let caption = if index == 0 {
                    format!("1 · {} · reference", item.name)
                } else {
                    format!("{} · {}", index + 1, item.name)
                };
                painter.text(
                    rect.left_top() + egui::vec2(5.0, 4.0),
                    egui::Align2::LEFT_TOP,
                    caption,
                    egui::FontId::proportional(12.0),
                    egui::Color32::WHITE,
                );
            }
        });
}

fn curve_to_screen(point: CurvePoint, rect: egui::Rect) -> egui::Pos2 {
    egui::pos2(
        rect.left() + point.x * rect.width(),
        rect.bottom() - point.y * rect.height(),
    )
}

fn screen_to_curve(position: egui::Pos2, rect: egui::Rect) -> CurvePoint {
    CurvePoint {
        x: ((position.x - rect.left()) / rect.width()).clamp(0.0, 1.0),
        y: ((rect.bottom() - position.y) / rect.height()).clamp(0.0, 1.0),
    }
}

fn nearest_curve_point(
    points: &[CurvePoint],
    position: egui::Pos2,
    rect: egui::Rect,
) -> Option<usize> {
    points
        .iter()
        .enumerate()
        .map(|(index, point)| (index, curve_to_screen(*point, rect).distance(position)))
        .filter(|(_, distance)| *distance <= 12.0)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(index, _)| index)
}

/// A square tone-curve editor with add, drag and remove interactions.
fn curve_editor(
    ui: &mut egui::Ui,
    curve: &mut ToneCurve,
    drag: &mut Option<usize>,
    color: egui::Color32,
) -> bool {
    let side = ui.available_width().clamp(160.0, 280.0);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(18));

    for step in 0..=4 {
        let t = step as f32 / 4.0;
        let x = rect.left() + t * rect.width();
        let y = rect.bottom() - t * rect.height();
        let stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_gray(38));
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            stroke,
        );
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            stroke,
        );
    }
    painter.line_segment(
        [
            curve_to_screen(CurvePoint { x: 0.0, y: 0.0 }, rect),
            curve_to_screen(CurvePoint { x: 1.0, y: 1.0 }, rect),
        ],
        egui::Stroke::new(1.0_f32, egui::Color32::from_gray(70)),
    );

    let mut points = curve.points().to_vec();
    let mut changed = false;

    if response.drag_started() {
        if let Some(position) = response.interact_pointer_pos() {
            match nearest_curve_point(&points, position, rect) {
                Some(index) => *drag = Some(index),
                None => {
                    let point = screen_to_curve(position, rect);
                    points.push(point);
                    points.sort_by(|a, b| a.x.total_cmp(&b.x));
                    *drag = points
                        .iter()
                        .position(|candidate| (candidate.x - point.x).abs() < 1e-6);
                }
            }
            changed = true;
        }
    }

    if response.dragged() {
        if let (Some(index), Some(position)) = (*drag, response.interact_pointer_pos()) {
            let mut updated = screen_to_curve(position, rect);
            if index == 0 {
                updated.x = 0.0;
            } else if index + 1 == points.len() {
                updated.x = 1.0;
            } else {
                let low = points[index - 1].x + 0.004;
                let high = points[index + 1].x - 0.004;
                updated.x = updated.x.clamp(low, high.max(low));
            }
            if let Some(point) = points.get_mut(index) {
                *point = updated;
                changed = true;
            }
        }
    }
    if response.drag_stopped() {
        *drag = None;
    }

    if response.clicked() {
        if let Some(position) = response.interact_pointer_pos() {
            if nearest_curve_point(&points, position, rect).is_none() {
                points.push(screen_to_curve(position, rect));
                changed = true;
            }
        }
    }
    if response.secondary_clicked() {
        if let Some(position) = response.interact_pointer_pos() {
            if let Some(index) = nearest_curve_point(&points, position, rect) {
                if index > 0 && index + 1 < points.len() {
                    points.remove(index);
                    *drag = None;
                    changed = true;
                }
            }
        }
    }

    if changed {
        curve.set_points(points);
    }

    let samples: Vec<egui::Pos2> = (0..=128)
        .map(|step| {
            let x = step as f32 / 128.0;
            curve_to_screen(
                CurvePoint {
                    x,
                    y: curve.evaluate(x),
                },
                rect,
            )
        })
        .collect();
    painter.add(egui::Shape::line(
        samples,
        egui::Stroke::new(1.5_f32, color),
    ));

    for (index, point) in curve.points().iter().enumerate() {
        let position = curve_to_screen(*point, rect);
        let radius = if *drag == Some(index) { 6.0 } else { 4.5 };
        painter.circle_filled(position, radius, color);
        painter.circle_stroke(
            position,
            radius,
            egui::Stroke::new(1.0_f32, egui::Color32::WHITE),
        );
    }

    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }
    changed
}

fn crop_from_screen(
    image: &RgbaImage,
    start: egui::Pos2,
    end: egui::Pos2,
    image_rect: egui::Rect,
) -> Option<stitch_domain::CropRect> {
    let to_image = |position: egui::Pos2| {
        let x = ((position.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0);
        let y = ((position.y - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0);
        (x * image.width() as f32, y * image.height() as f32)
    };
    let (ax, ay) = to_image(start);
    let (bx, by) = to_image(end);
    let x = (ax.min(bx).round() as u32).min(image.width() - 1);
    let y = (ay.min(by).round() as u32).min(image.height() - 1);
    let width = ((ax - bx).abs().round() as u32).min(image.width() - x);
    let height = ((ay - by).abs().round() as u32).min(image.height() - y);
    (width >= 8 && height >= 8).then_some(stitch_domain::CropRect {
        x,
        y,
        width,
        height,
    })
}

fn draw_swatch(ui: &mut egui::Ui, color: [u8; 3]) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(
        rect,
        2.0,
        egui::Color32::from_rgb(color[0], color[1], color[2]),
    );
    painter.rect_stroke(
        rect,
        2.0,
        egui::Stroke::new(1.0_f32, egui::Color32::from_gray(140)),
        egui::StrokeKind::Inside,
    );
    ui.label(format!("#{:02X}{:02X}{:02X}", color[0], color[1], color[2]));
}

fn fit_size(texture: egui::Vec2, bounds: egui::Vec2) -> egui::Vec2 {
    if texture.x <= 0.0 || texture.y <= 0.0 {
        return bounds;
    }
    let scale = (bounds.x / texture.x).min(bounds.y / texture.y);
    texture * scale
}

fn draw_histogram(ui: &mut egui::Ui, histogram: &Histogram) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 110.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, egui::Color32::from_gray(18));
    let peak = histogram.peak() as f32;
    let to_screen = |bin: usize, value: u32| {
        egui::pos2(
            rect.left() + rect.width() * bin as f32 / 255.0,
            rect.bottom() - rect.height() * (value as f32 / peak).sqrt(),
        )
    };
    let channels = [
        (&histogram.red, egui::Color32::from_rgb(230, 90, 90)),
        (&histogram.green, egui::Color32::from_rgb(90, 220, 120)),
        (&histogram.blue, egui::Color32::from_rgb(100, 150, 240)),
    ];
    for (values, color) in channels {
        let points: Vec<egui::Pos2> = (0..256).map(|bin| to_screen(bin, values[bin])).collect();
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.0_f32, color)));
    }
}
