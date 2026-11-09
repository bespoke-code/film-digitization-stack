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
    normalized_rotation, rotation_is_identity, CoreResult, CurvePoint, EditSettings, Histogram,
    Homography, ImageId, ImageLocator, RgbaImage, SaveOptions, SourceImage, StitchOptions,
    StitchProgress, StitchResult, ToneCurve,
};
use stitch_renderer_wgpu::PreviewRenderer;
use stitch_ui_core::{PreviewTarget, Workbench, WorkbenchStatus};

enum WorkerEvent {
    Progress(StitchProgress),
    Done(Box<CoreResult<StitchResult>>),
}

struct StitchWorker {
    receiver: Receiver<WorkerEvent>,
    handle: Option<JoinHandle<()>>,
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
    ndc_rect: [f32; 4],
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
                self.ndc_rect,
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

/// Smallest useful adjustment for each tone control.
const EXPOSURE_RANGE: f32 = 6.0;
const EXPOSURE_STEP: f64 = 0.01;
const CONTRAST_STEP: f64 = 0.005;
const LEVEL_STEP: f64 = 0.001;
const GAMMA_STEP: f64 = 0.001;

/// Visual size of the crop handles and the thinnest usable border.
const CROP_HANDLE_RADIUS: f32 = 7.0;
const CROP_HANDLE_SIZE: f32 = 6.0;
const CROP_MIN_SIZE: f32 = 12.0;

/// A crop selection in normalized image coordinates.
type CropDraft = egui::Rect;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CropHandle {
    New,
    Move,
    NorthWest,
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
}

#[derive(Clone, Copy, Debug)]
struct CropDrag {
    handle: CropHandle,
    origin: CropDraft,
    anchor: egui::Pos2,
}

struct ArrangeItem {
    id: ImageId,
    name: String,
    /// Top-left corner of the transformed bounds in reference-image pixels.
    position: egui::Vec2,
    /// Source image size after the edit rotation, in pixels.
    source_size: egui::Vec2,
    /// Clockwise placement rotation in degrees.
    rotation: f32,
    /// Placement scale relative to the source pixels.
    zoom: f32,
    texture: egui::TextureHandle,
}

impl ArrangeItem {
    fn transformed_size(&self) -> egui::Vec2 {
        transformed_size(self.source_size, self.rotation, self.zoom)
    }

    fn homography(&self) -> Homography {
        let rotation = Homography::rotation(f64::from(self.rotation));
        let scale = Homography::scale(f64::from(self.zoom), f64::from(self.zoom));
        let linear = rotation.then(&scale);
        let width = f64::from(self.source_size.x);
        let height = f64::from(self.source_size.y);
        let (min_x, min_y) = quad_bounds(&linear, width, height);
        Homography::translation(
            f64::from(self.position.x) - min_x,
            f64::from(self.position.y) - min_y,
        )
        .then(&linear)
    }
}

fn transformed_size(source: egui::Vec2, rotation: f32, zoom: f32) -> egui::Vec2 {
    let (sin, cos) = rotation.to_radians().sin_cos();
    let width = source.x * zoom;
    let height = source.y * zoom;
    egui::vec2(
        width * cos.abs() + height * sin.abs(),
        width * sin.abs() + height * cos.abs(),
    )
}

fn quad_bounds(homography: &Homography, width: f64, height: f64) -> (f64, f64) {
    let corners = [
        stitch_domain::Point::new(0.0, 0.0),
        stitch_domain::Point::new(width, 0.0),
        stitch_domain::Point::new(width, height),
        stitch_domain::Point::new(0.0, height),
    ];
    corners
        .into_iter()
        .filter_map(|corner| homography.transform(corner))
        .fold((f64::INFINITY, f64::INFINITY), |(min_x, min_y), point| {
            (min_x.min(point.x), min_y.min(point.y))
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArrangeLayout {
    Horizontal,
    Vertical,
}

struct ArrangeState {
    items: Vec<ArrangeItem>,
    scale: f32,
    view_zoom: f32,
    scene: egui::Rect,
    drag: Option<usize>,
    selected: Option<usize>,
    layout: ArrangeLayout,
}

#[derive(Default)]
struct ArrangeAction {
    reset: Option<ArrangeLayout>,
    confirm: bool,
}

impl ArrangeState {
    fn reset_layout(&mut self) {
        self.drag = None;
        let extent = self
            .items
            .iter()
            .map(|item| match self.layout {
                ArrangeLayout::Horizontal => item.transformed_size().y,
                ArrangeLayout::Vertical => item.transformed_size().x,
            })
            .fold(0.0_f32, f32::max);
        let mut cursor = 0.0;
        for item in &mut self.items {
            let size = item.transformed_size();
            item.position = match self.layout {
                ArrangeLayout::Horizontal => egui::vec2(cursor, ((extent - size.y) * 0.5).max(0.0)),
                ArrangeLayout::Vertical => egui::vec2(((extent - size.x) * 0.5).max(0.0), cursor),
            };
            cursor += match self.layout {
                ArrangeLayout::Horizontal => size.x,
                ArrangeLayout::Vertical => size.y,
            } * (1.0 - ARRANGE_OVERLAP);
        }
        self.rebuild_scene();
    }

    fn rebuild_scene(&mut self) {
        let mut bounds = egui::Rect::NOTHING;
        for item in &self.items {
            let rect = egui::Rect::from_min_size(
                egui::pos2(item.position.x, item.position.y),
                item.transformed_size(),
            );
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
    gpu_available: bool,
    pending_paths: Vec<PathBuf>,
    picking_film_base: bool,
    level_channel: LevelChannel,
    tone_open: bool,
    curve_drag: Option<usize>,
    crop_mode: bool,
    crop_draft: Option<CropDraft>,
    crop_drag: Option<CropDrag>,
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
            gpu_available,
            pending_paths,
            picking_film_base: false,
            level_channel: LevelChannel::Red,
            tone_open: false,
            curve_drag: None,
            crop_mode: false,
            crop_draft: None,
            crop_drag: None,
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
            let result = service.stitch_with_rotations(
                &job.images,
                &job.rotations,
                &job.manual,
                &job.options,
                &progress,
            );
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
            DialogOutcome::SaveImage(path, quality) => {
                let locator = ImageLocator::from(path.as_path());
                let options = SaveOptions {
                    jpeg_quality: quality,
                };
                match self
                    .workbench
                    .export_with_options(&self.project_service, &locator, &options)
                {
                    Ok(()) => self
                        .workbench
                        .notify(format!("Exported to {}", path.display())),
                    Err(error) => self.workbench.fail_stitch(error.to_string()),
                }
            }
            DialogOutcome::SaveFolder(folder, format, quality) => {
                let options = SaveOptions {
                    jpeg_quality: quality,
                };
                match self.workbench.export_all_with_options(
                    &self.project_service,
                    &folder,
                    format.extension(),
                    &options,
                ) {
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
                self.dialog.open_in(
                    DialogMode::AddImages,
                    None,
                    String::new(),
                    self.workbench.default_export_jpeg_quality(),
                );
            }
            if ui.button("Open folder").clicked() {
                self.dialog.open_in(
                    DialogMode::OpenFolder,
                    None,
                    String::new(),
                    self.workbench.default_export_jpeg_quality(),
                );
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
                    self.workbench.default_export_jpeg_quality(),
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
                    self.workbench.default_batch_jpeg_quality(),
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
        if !self.workbench.has_preview() {
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
            .add(
                egui::Slider::new(&mut edits.exposure, -EXPOSURE_RANGE..=EXPOSURE_RANGE)
                    .text("Exposure (EV)")
                    .fixed_decimals(2)
                    .step_by(EXPOSURE_STEP)
                    .drag_value_speed(EXPOSURE_STEP),
            )
            .on_hover_text("Arrow keys move 0.01 EV. Drag or type the value for fine control.")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut edits.contrast, -1.0..=1.0)
                    .text("Contrast")
                    .fixed_decimals(3)
                    .step_by(CONTRAST_STEP)
                    .drag_value_speed(CONTRAST_STEP),
            )
            .on_hover_text("Arrow keys move 0.005. Drag or type the value for fine control.")
            .changed();

        ui.horizontal(|ui| {
            ui.label("Rotate");
            if ui
                .button("Left 90°")
                .on_hover_text("Rotate a quarter turn counter-clockwise")
                .clicked()
            {
                edits.rotation = normalized_rotation(edits.rotation - 90.0);
                changed = true;
            }
            if ui
                .button("Right 90°")
                .on_hover_text("Rotate a quarter turn clockwise")
                .clicked()
            {
                edits.rotation = normalized_rotation(edits.rotation + 90.0);
                changed = true;
            }
            if ui.button("Reset").clicked() {
                edits.rotation = 0.0;
                changed = true;
            }
        });
        changed |= ui
            .add(
                egui::DragValue::new(&mut edits.rotation)
                    .range(0.0..=359.9)
                    .speed(0.25)
                    .suffix("°"),
            )
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
                .on_hover_text("Adjust the crop on the preview, then press Apply")
                .clicked()
            {
                self.crop_mode = !self.crop_mode;
                self.crop_drag = None;
                self.picking_film_base = false;
                self.crop_draft = if self.crop_mode {
                    self.workbench.crop_preview().and_then(|preview| {
                        edits.crop.map(|crop| {
                            crop_to_normalized(crop, preview.source_width, preview.source_height)
                        })
                    })
                } else {
                    None
                };
                if self.crop_mode {
                    // Build the downscaled base before the first drag so the
                    // first click lands in an already responsive viewport.
                    let _ = self.workbench.crop_preview();
                }
            }
            if !self.crop_mode && edits.crop.is_some() && ui.button("Reset crop").clicked() {
                edits.crop = None;
                changed = true;
            }
        });
        if self.crop_mode {
            ui.horizontal(|ui| {
                if ui.button("Apply crop").clicked() {
                    self.apply_crop(&mut edits);
                }
                if ui.button("Cancel").clicked() {
                    self.crop_mode = false;
                    self.crop_draft = None;
                    self.crop_drag = None;
                }
                if ui.button("Select all").clicked() {
                    self.crop_draft = Some(full_crop());
                }
            });
            ui.weak("Drag inside to move, drag a handle to resize, then Apply.");
        }
        if let Some(crop) = edits.crop {
            ui.weak(format!(
                "Crop {} × {} at {}, {}",
                crop.width, crop.height, crop.x, crop.y
            ));
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

    fn apply_crop(&mut self, edits: &mut EditSettings) {
        if let (Some(draft), Some(preview)) = (self.crop_draft, self.workbench.crop_preview()) {
            if let Some(crop) =
                crop_from_normalized(draft, preview.source_width, preview.source_height)
            {
                edits.crop = Some(crop);
                self.workbench.update_current_edits(edits.clone());
            }
        }
        self.crop_mode = false;
        self.crop_draft = None;
        self.crop_drag = None;
    }

    fn metadata(&mut self, ui: &mut egui::Ui) {
        ui.heading("Metadata");
        if !self.workbench.has_preview() {
            ui.weak("Select an image or stitch a panorama to add metadata.");
            return;
        }

        let mut metadata = self.workbench.current_metadata();
        let mut changed = false;
        egui::Grid::new("photo-metadata")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                changed |= metadata_field(ui, "Camera", "Nikon FM2", &mut metadata.camera);
                changed |= metadata_field(ui, "Lens", "Nikkor 50mm f/1.8", &mut metadata.lens);
                changed |= metadata_field(ui, "Film", "Kodak Portra 400", &mut metadata.film);
                changed |= metadata_field(ui, "ISO", "400", &mut metadata.iso);
                changed |= metadata_field(ui, "Aperture", "f/2.8", &mut metadata.aperture);
                changed |= metadata_field(ui, "Shutter", "1/125", &mut metadata.shutter_speed);
                changed |= metadata_field(ui, "Focal length", "50 mm", &mut metadata.focal_length);
                changed |= metadata_field(ui, "Date", "YYYY-MM-DD", &mut metadata.date);
                changed |= metadata_field(ui, "Location", "", &mut metadata.location);
                ui.label("Notes");
                changed |= ui
                    .add(
                        egui::TextEdit::multiline(&mut metadata.notes)
                            .desired_rows(3)
                            .desired_width(160.0)
                            .hint_text("Development, scanner, push process…"),
                    )
                    .changed();
                ui.end_row();
            });

        let mut reset = false;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    self.workbench.project().len() > 1 || self.workbench.result().is_some(),
                    egui::Button::new("Apply to all"),
                )
                .on_hover_text("Copy these details to every photo and the panorama")
                .clicked()
            {
                self.workbench.apply_metadata_to_all(metadata.clone());
            }
            reset = ui.button("Reset").clicked();
        });
        ui.weak("Metadata is written into exported PNG, JPEG and WebP files.");

        if reset {
            self.workbench.reset_current_metadata();
        } else if changed {
            self.workbench.update_current_metadata(metadata);
        }
    }

    fn open_arrange(&mut self, ctx: &egui::Context) {
        let images = self.workbench.project().images().to_vec();
        if images.len() < 2 {
            self.workbench
                .notify("Add at least two images before arranging them.");
            return;
        }
        let rotations: Vec<f32> = images
            .iter()
            .map(|image| self.workbench.edits(image.id).rotation)
            .collect();
        let sizes: Vec<egui::Vec2> = images
            .iter()
            .zip(rotations.iter())
            .map(|(image, rotation)| {
                let (width, height) = RgbaImage::rotated_dimensions(
                    image.image.width(),
                    image.image.height(),
                    *rotation,
                );
                egui::vec2(width as f32, height as f32)
            })
            .collect();
        let manual = self.workbench.manual_placements();
        let first_height = sizes.first().map(|size| size.y).unwrap_or(0.0);
        let mut cursor = 0.0f32;
        let mut items = Vec::with_capacity(images.len());
        for (index, image) in images.iter().enumerate() {
            let source_size = sizes[index];
            let (position, rotation, zoom) = manual
                .get(index)
                .copied()
                .flatten()
                .and_then(|placement| decompose_placement(placement, source_size))
                .unwrap_or_else(|| {
                    (
                        egui::vec2(cursor, ((first_height - source_size.y) * 0.5).max(0.0)),
                        0.0,
                        1.0,
                    )
                });
            cursor = position.x
                + transformed_size(source_size, rotation, zoom).x * (1.0 - ARRANGE_OVERLAP);
            let texture = if rotation_is_identity(rotations[index]) {
                self.thumbnail(ctx, image)
            } else {
                let thumbnail = image.thumbnail.rotated(rotations[index]);
                texture_from_rgba(
                    ctx,
                    format!("arrange-{}-{}", image.id.0, rotations[index].to_bits()),
                    &thumbnail,
                )
            };
            items.push(ArrangeItem {
                id: image.id,
                name: image.name.clone(),
                position,
                source_size,
                rotation,
                zoom,
                texture,
            });
        }
        let max_side = sizes
            .iter()
            .map(|size| size.x.max(size.y))
            .fold(0.0_f32, f32::max)
            .max(1.0);
        let scale = (320.0 / max_side).clamp(0.01, 1.0);
        let mut state = ArrangeState {
            items,
            scale,
            view_zoom: 1.0,
            scene: egui::Rect::NOTHING,
            drag: None,
            selected: None,
            layout: ArrangeLayout::Horizontal,
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
            self.workbench
                .set_manual_placement(item.id, item.homography());
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
            .default_size(egui::vec2(860.0, 620.0))
            .resizable(true)
            .show(ctx, |ui| {
                ui.weak(
                    "Drag frames until the overlap looks right. Frame 1 is the reference and stays fixed. \
                     Rotate or scale a frame to fit scans that were captured at an angle. \
                     The stitcher still refines every frame automatically and uses these placements only when no match is found.",
                );
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Row").clicked() {
                        action.reset = Some(ArrangeLayout::Horizontal);
                    }
                    if ui.button("Column").clicked() {
                        action.reset = Some(ArrangeLayout::Vertical);
                    }
                    if ui.button("Apply and stitch").clicked() {
                        action.confirm = true;
                    }
                });
                if let Some(state) = self.arrange.as_mut() {
                    if let Some(layout) = action.reset {
                        state.layout = layout;
                        state.reset_layout();
                    }
                    ui.horizontal_wrapped(|ui| {
                        ui.label("View zoom");
                        if ui.button("−").clicked() {
                            state.view_zoom = (state.view_zoom / 1.25).clamp(0.1, 4.0);
                        }
                        ui.add(
                            egui::Slider::new(&mut state.view_zoom, 0.1..=4.0)
                                .logarithmic(true)
                                .show_value(false),
                        );
                        if ui.button("+").clicked() {
                            state.view_zoom = (state.view_zoom * 1.25).clamp(0.1, 4.0);
                        }
                        if ui.button("Fit").clicked() {
                            state.view_zoom = 1.0;
                        }
                    });
                    if arrange_frame_controls(ui, state) {
                        state.rebuild_scene();
                    }
                    ui.separator();
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
                    .fixed_decimals(3)
                    .step_by(LEVEL_STEP)
                    .drag_value_speed(LEVEL_STEP),
            )
            .on_hover_text("Arrow keys move 0.001. Drag or type the value for fine control.")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut levels.white, 0.1..=1.0)
                    .text("White")
                    .fixed_decimals(3)
                    .step_by(LEVEL_STEP)
                    .drag_value_speed(LEVEL_STEP),
            )
            .on_hover_text("Arrow keys move 0.001. Drag or type the value for fine control.")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut levels.gamma, 0.1..=4.0)
                    .text("Gamma")
                    .fixed_decimals(3)
                    .step_by(GAMMA_STEP)
                    .drag_value_speed(GAMMA_STEP),
            )
            .on_hover_text("Arrow keys move 0.001. Drag or type the value for fine control.")
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
        let crop_preview = if self.crop_mode {
            self.workbench.crop_preview()
        } else {
            None
        };
        let preview = if self.crop_mode {
            crop_preview.as_ref().map(|preview| preview.info())
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
            let target_changed = self.preview.key.map(|previous| previous.0) != Some(key.0);
            self.preview.key = Some(key);
            self.preview.zoom = 1.0;
            self.preview.pan = egui::Vec2::ZERO;
            if self.crop_mode && target_changed {
                self.crop_draft = crop_preview.as_ref().and_then(|preview| {
                    self.workbench.current_edits().crop.map(|crop| {
                        crop_to_normalized(crop, preview.source_width, preview.source_height)
                    })
                });
                self.crop_drag = None;
            }
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

        let visible = image_rect.intersect(viewport);
        if visible.is_positive() {
            let ndc_rect = rect_in_ndc(image_rect, visible);
            painter.add(egui_wgpu::Callback::new_paint_callback(
                visible,
                PreviewCallback {
                    image: Some(preview.image.clone()),
                    revision: preview.revision ^ u64::from(self.crop_mode) << 62,
                    ndc_rect,
                },
            ));
        }

        if self.crop_mode {
            let pointer = ui.input(|input| input.pointer.latest_pos());
            if let Some(draft) = self.crop_draft {
                if let Some(position) = pointer {
                    let screen = normalized_rect_to_screen(draft, image_rect);
                    let cursor = crop_hit(screen, position)
                        .map(crop_cursor)
                        .unwrap_or(egui::CursorIcon::Crosshair);
                    if response.hovered() {
                        ui.ctx().set_cursor_icon(cursor);
                    }
                }
                let rect = normalized_rect_to_screen(draft, image_rect);
                draw_crop_rect(&painter, rect, true);
            } else if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            }

            let pressed = response.hovered() && ui.input(|input| input.pointer.primary_pressed());
            if pressed {
                if let Some(position) = pointer.filter(|position| image_rect.contains(*position)) {
                    let handle = self
                        .crop_draft
                        .map(|draft| normalized_rect_to_screen(draft, image_rect))
                        .and_then(|screen| crop_hit(screen, position))
                        .unwrap_or(CropHandle::New);
                    let anchor = screen_to_normalized(position, image_rect);
                    self.crop_drag = Some(CropDrag {
                        handle,
                        origin: self
                            .crop_draft
                            .unwrap_or_else(|| egui::Rect::from_min_size(anchor, egui::Vec2::ZERO)),
                        anchor,
                    });
                    if handle == CropHandle::New {
                        self.crop_draft = Some(egui::Rect::from_min_size(anchor, egui::Vec2::ZERO));
                    }
                }
            }
            if let Some(drag) = self.crop_drag {
                if ui.input(|input| input.pointer.primary_down()) {
                    if let Some(position) = pointer {
                        self.crop_draft = Some(update_crop_draft(drag, position, image_rect));
                    }
                } else {
                    self.crop_drag = None;
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
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.develop(ui);
                        ui.separator();
                        self.metadata(ui);
                        ui.separator();
                        if let Some(histogram) = self.workbench.histogram() {
                            ui.heading("Histogram");
                            draw_histogram(ui, &histogram);
                        }
                        ui.separator();
                        egui::CollapsingHeader::new("Stitch settings")
                            .default_open(false)
                            .show(ui, |ui| self.settings(ui));
                        egui::CollapsingHeader::new("Diagnostics")
                            .default_open(true)
                            .show(ui, |ui| self.diagnostics(ui));
                    });
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

fn metadata_field(ui: &mut egui::Ui, label: &str, hint: &str, value: &mut String) -> bool {
    ui.label(label);
    let changed = ui
        .add(
            egui::TextEdit::singleline(value)
                .desired_width(160.0)
                .hint_text(hint),
        )
        .changed();
    ui.end_row();
    changed
}

fn decompose_placement(
    placement: Homography,
    source_size: egui::Vec2,
) -> Option<(egui::Vec2, f32, f32)> {
    let width = f64::from(source_size.x);
    let height = f64::from(source_size.y);
    if width <= 0.0 || height <= 0.0 {
        return None;
    }
    let corner = |x, y| placement.transform(stitch_domain::Point::new(x, y));
    let origin = corner(0.0, 0.0)?;
    let right = corner(width, 0.0)?;
    let bottom = corner(0.0, height)?;
    let opposite = corner(width, height)?;
    let dx = right.x - origin.x;
    let dy = right.y - origin.y;
    let length = (dx * dx + dy * dy).sqrt();
    if !length.is_finite() || length <= 1e-9 {
        return None;
    }
    let zoom = length / width;
    let rotation = normalized_rotation(dy.atan2(dx).to_degrees() as f32);
    let min_x = origin.x.min(right.x).min(bottom.x).min(opposite.x);
    let min_y = origin.y.min(right.y).min(bottom.y).min(opposite.y);
    if !zoom.is_finite() || !min_x.is_finite() || !min_y.is_finite() {
        return None;
    }
    Some((
        egui::vec2(min_x as f32, min_y as f32),
        rotation,
        zoom as f32,
    ))
}

fn texture_from_rgba(ctx: &egui::Context, name: String, image: &RgbaImage) -> egui::TextureHandle {
    let color = egui::ColorImage::from_rgba_unmultiplied(
        [image.width() as usize, image.height() as usize],
        image.pixels(),
    );
    ctx.load_texture(name, color, egui::TextureOptions::LINEAR)
}

fn arrange_quad(item: &ArrangeItem, origin: egui::Pos2, scale: f32) -> [egui::Pos2; 4] {
    let placement = item.homography();
    let width = f64::from(item.source_size.x);
    let height = f64::from(item.source_size.y);
    [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)].map(|(x, y)| {
        let point = placement
            .transform(stitch_domain::Point::new(x, y))
            .unwrap_or_default();
        origin + egui::vec2(point.x as f32, point.y as f32) * scale
    })
}

fn arrange_frame_controls(ui: &mut egui::Ui, state: &mut ArrangeState) -> bool {
    let Some(index) = state.selected else {
        ui.weak("Click a frame to rotate or scale it.");
        return false;
    };
    if index >= state.items.len() {
        state.selected = None;
        return false;
    }
    let reference = index == 0;
    let item = &mut state.items[index];
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("Frame {} · {}", index + 1, item.name));
        ui.add_enabled_ui(!reference, |ui| {
            if ui
                .button("Left 90°")
                .on_hover_text("Rotate counter-clockwise")
                .clicked()
            {
                item.rotation = normalized_rotation(item.rotation - 90.0);
                changed = true;
            }
            if ui
                .button("Right 90°")
                .on_hover_text("Rotate clockwise")
                .clicked()
            {
                item.rotation = normalized_rotation(item.rotation + 90.0);
                changed = true;
            }
            changed |= ui
                .add(
                    egui::DragValue::new(&mut item.rotation)
                        .speed(0.5)
                        .suffix("°"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut item.zoom, 0.1..=4.0)
                        .logarithmic(true)
                        .text("Scale"),
                )
                .changed();
            if ui.button("Reset frame").clicked() {
                item.rotation = 0.0;
                item.zoom = 1.0;
                changed = true;
            }
        });
        if reference {
            ui.weak("The reference frame stays fixed.");
        }
    });
    item.rotation = normalized_rotation(item.rotation);
    changed
}

fn draw_arrange(ui: &mut egui::Ui, state: &mut ArrangeState) {
    let viewport = ui.available_size();
    let scale = state.scale * state.view_zoom;
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
                let (id, quad) = {
                    let item = &state.items[index];
                    (item.id, arrange_quad(item, origin, scale))
                };
                let bounds = egui::Rect::from_points(&quad);
                let sense = if index == 0 {
                    egui::Sense::click()
                } else {
                    egui::Sense::click_and_drag()
                };
                let interaction =
                    ui.interact(bounds, ui.scope_id().with(("arrange-frame", id.0)), sense);
                if interaction.clicked() {
                    state.selected = Some(index);
                }
                if index > 0 {
                    if interaction.drag_started() {
                        state.drag = Some(index);
                        state.selected = Some(index);
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
                let quad = arrange_quad(item, origin, scale);
                let mut mesh = egui::Mesh::with_texture(item.texture.id());
                let uvs = [
                    egui::pos2(0.0, 0.0),
                    egui::pos2(1.0, 0.0),
                    egui::pos2(1.0, 1.0),
                    egui::pos2(0.0, 1.0),
                ];
                for (position, uv) in quad.into_iter().zip(uvs) {
                    mesh.vertices.push(egui::epaint::Vertex {
                        pos: position,
                        uv,
                        color: egui::Color32::WHITE,
                    });
                }
                mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
                painter.add(egui::Shape::mesh(mesh));

                let stroke = if state.drag == Some(index) || state.selected == Some(index) {
                    egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(255, 200, 60))
                } else if index == 0 {
                    egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(90, 170, 240))
                } else {
                    egui::Stroke::new(1.0_f32, egui::Color32::from_gray(120))
                };
                painter.add(egui::Shape::closed_line(quad.to_vec(), stroke));
                let caption = if index == 0 {
                    format!("1 · {} · reference", item.name)
                } else {
                    format!("{} · {}", index + 1, item.name)
                };
                painter.text(
                    egui::Rect::from_points(&quad).left_top() + egui::vec2(5.0, 4.0),
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

fn full_crop() -> CropDraft {
    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0))
}

fn crop_to_normalized(
    crop: stitch_domain::CropRect,
    source_width: u32,
    source_height: u32,
) -> CropDraft {
    let width = source_width.max(1) as f32;
    let height = source_height.max(1) as f32;
    egui::Rect::from_min_max(
        egui::pos2(crop.x as f32 / width, crop.y as f32 / height),
        egui::pos2(
            (crop.x + crop.width) as f32 / width,
            (crop.y + crop.height) as f32 / height,
        ),
    )
}

fn crop_from_normalized(
    draft: CropDraft,
    source_width: u32,
    source_height: u32,
) -> Option<stitch_domain::CropRect> {
    let width = source_width.max(1) as f32;
    let height = source_height.max(1) as f32;
    let left = (draft.min.x.clamp(0.0, 1.0) * width).round() as u32;
    let top = (draft.min.y.clamp(0.0, 1.0) * height).round() as u32;
    let right = (draft.max.x.clamp(0.0, 1.0) * width).round() as u32;
    let bottom = (draft.max.y.clamp(0.0, 1.0) * height).round() as u32;
    let x = left.min(source_width.saturating_sub(1));
    let y = top.min(source_height.saturating_sub(1));
    let width = right.saturating_sub(x).min(source_width - x);
    let height = bottom.saturating_sub(y).min(source_height - y);
    (width >= 8 && height >= 8).then_some(stitch_domain::CropRect {
        x,
        y,
        width,
        height,
    })
}

fn normalized_rect_to_screen(rect: CropDraft, image_rect: egui::Rect) -> egui::Rect {
    egui::Rect::from_min_max(
        egui::pos2(
            image_rect.left() + rect.min.x * image_rect.width(),
            image_rect.top() + rect.min.y * image_rect.height(),
        ),
        egui::pos2(
            image_rect.left() + rect.max.x * image_rect.width(),
            image_rect.top() + rect.max.y * image_rect.height(),
        ),
    )
}

fn screen_to_normalized(position: egui::Pos2, image_rect: egui::Rect) -> egui::Pos2 {
    egui::pos2(
        ((position.x - image_rect.left()) / image_rect.width().max(1.0)).clamp(0.0, 1.0),
        ((position.y - image_rect.top()) / image_rect.height().max(1.0)).clamp(0.0, 1.0),
    )
}

fn crop_handle_position(handle: CropHandle, rect: egui::Rect) -> egui::Pos2 {
    let center = rect.center();
    match handle {
        CropHandle::NorthWest => rect.left_top(),
        CropHandle::North => egui::pos2(center.x, rect.top()),
        CropHandle::NorthEast => rect.right_top(),
        CropHandle::East => egui::pos2(rect.right(), center.y),
        CropHandle::SouthEast => rect.right_bottom(),
        CropHandle::South => egui::pos2(center.x, rect.bottom()),
        CropHandle::SouthWest => rect.left_bottom(),
        CropHandle::West => egui::pos2(rect.left(), center.y),
        CropHandle::Move | CropHandle::New => center,
    }
}

fn crop_hit(rect: egui::Rect, position: egui::Pos2) -> Option<CropHandle> {
    for handle in [
        CropHandle::NorthWest,
        CropHandle::North,
        CropHandle::NorthEast,
        CropHandle::East,
        CropHandle::SouthEast,
        CropHandle::South,
        CropHandle::SouthWest,
        CropHandle::West,
    ] {
        if crop_handle_position(handle, rect).distance(position) <= CROP_HANDLE_RADIUS {
            return Some(handle);
        }
    }
    rect.contains(position).then_some(CropHandle::Move)
}

fn crop_cursor(handle: CropHandle) -> egui::CursorIcon {
    match handle {
        CropHandle::New => egui::CursorIcon::Crosshair,
        CropHandle::Move => egui::CursorIcon::Grab,
        CropHandle::NorthWest | CropHandle::SouthEast => egui::CursorIcon::ResizeNwSe,
        CropHandle::NorthEast | CropHandle::SouthWest => egui::CursorIcon::ResizeNeSw,
        CropHandle::North | CropHandle::South => egui::CursorIcon::ResizeVertical,
        CropHandle::East | CropHandle::West => egui::CursorIcon::ResizeHorizontal,
    }
}

fn update_crop_draft(drag: CropDrag, pointer: egui::Pos2, image_rect: egui::Rect) -> CropDraft {
    let pointer = screen_to_normalized(pointer, image_rect);
    let min_width = (CROP_MIN_SIZE / image_rect.width().max(1.0)).min(0.5);
    let min_height = (CROP_MIN_SIZE / image_rect.height().max(1.0)).min(0.5);
    let rect = match drag.handle {
        CropHandle::New => egui::Rect::from_two_pos(drag.anchor, pointer),
        CropHandle::Move => {
            let delta = pointer - drag.anchor;
            let dx = delta.x.clamp(-drag.origin.min.x, 1.0 - drag.origin.max.x);
            let dy = delta.y.clamp(-drag.origin.min.y, 1.0 - drag.origin.max.y);
            drag.origin.translate(egui::vec2(dx, dy))
        }
        handle => {
            let mut min = drag.origin.min;
            let mut max = drag.origin.max;
            if matches!(
                handle,
                CropHandle::NorthWest | CropHandle::West | CropHandle::SouthWest
            ) {
                min.x = pointer.x;
            }
            if matches!(
                handle,
                CropHandle::NorthEast | CropHandle::East | CropHandle::SouthEast
            ) {
                max.x = pointer.x;
            }
            if matches!(
                handle,
                CropHandle::NorthWest | CropHandle::North | CropHandle::NorthEast
            ) {
                min.y = pointer.y;
            }
            if matches!(
                handle,
                CropHandle::SouthWest | CropHandle::South | CropHandle::SouthEast
            ) {
                max.y = pointer.y;
            }
            egui::Rect::from_two_pos(min, max)
        }
    };
    let rect = egui::Rect::from_min_max(
        egui::pos2(rect.min.x.clamp(0.0, 1.0), rect.min.y.clamp(0.0, 1.0)),
        egui::pos2(rect.max.x.clamp(0.0, 1.0), rect.max.y.clamp(0.0, 1.0)),
    );
    let width = rect.width().max(min_width);
    let height = rect.height().max(min_height);
    let min_x = rect.min.x.min(1.0 - width);
    let min_y = rect.min.y.min(1.0 - height);
    egui::Rect::from_min_size(egui::pos2(min_x, min_y), egui::vec2(width, height))
}

/// Projects the visible part of the image into the callback viewport.
fn rect_in_ndc(image_rect: egui::Rect, viewport: egui::Rect) -> [f32; 4] {
    let width = viewport.width().max(1.0);
    let height = viewport.height().max(1.0);
    [
        (image_rect.left() - viewport.left()) / width * 2.0 - 1.0,
        1.0 - (image_rect.top() - viewport.top()) / height * 2.0,
        (image_rect.right() - viewport.left()) / width * 2.0 - 1.0,
        1.0 - (image_rect.bottom() - viewport.top()) / height * 2.0,
    ]
}

fn draw_crop_rect(painter: &egui::Painter, rect: egui::Rect, active: bool) {
    if !rect.is_positive() {
        return;
    }
    painter.rect_filled(rect, 0.0, egui::Color32::from_white_alpha(14));
    let color = if active {
        egui::Color32::from_rgb(255, 200, 60)
    } else {
        egui::Color32::from_rgb(230, 160, 40)
    };
    painter.rect_stroke(
        rect,
        0.0,
        egui::Stroke::new(1.0_f32, color),
        egui::StrokeKind::Inside,
    );
    if active {
        for handle in [
            CropHandle::NorthWest,
            CropHandle::North,
            CropHandle::NorthEast,
            CropHandle::East,
            CropHandle::SouthEast,
            CropHandle::South,
            CropHandle::SouthWest,
            CropHandle::West,
        ] {
            let position = crop_handle_position(handle, rect);
            painter.rect_filled(
                egui::Rect::from_center_size(
                    position,
                    egui::vec2(CROP_HANDLE_SIZE, CROP_HANDLE_SIZE),
                ),
                1.0,
                color,
            );
        }
    }
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
    let peak = histogram.peak();
    let to_screen = |bin: usize, value: f32| {
        egui::pos2(
            rect.left() + rect.width() * bin as f32 / 255.0,
            rect.bottom() - rect.height() * (value / peak).sqrt(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn developed(settings: EditSettings) -> RgbaImage {
        let image = RgbaImage::from_fn(256, 1, |x, _| [x as u8, x as u8, x as u8, 255]);
        settings.develop(&image)
    }

    fn worst_channel_change(first: &RgbaImage, second: &RgbaImage) -> u8 {
        first
            .pixels()
            .iter()
            .zip(second.pixels())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn exposure_slider_steps_are_visually_small() {
        let before = developed(EditSettings::default());
        let after = developed(EditSettings {
            exposure: EXPOSURE_STEP as f32,
            ..EditSettings::default()
        });
        assert!(
            worst_channel_change(&before, &after) <= 2,
            "one exposure step moves a channel too far"
        );
    }

    #[test]
    fn contrast_slider_steps_are_visually_small_at_the_extremes() {
        let before = developed(EditSettings {
            contrast: 1.0,
            ..EditSettings::default()
        });
        let after = developed(EditSettings {
            contrast: 1.0 - CONTRAST_STEP as f32,
            ..EditSettings::default()
        });
        assert!(
            worst_channel_change(&before, &after) <= 8,
            "one contrast step moves a channel too far"
        );
    }

    #[test]
    fn levels_and_gamma_slider_steps_are_sub_level() {
        let before = developed(EditSettings {
            levels: stitch_domain::Levels {
                red: stitch_domain::ChannelLevels {
                    black: 0.2,
                    white: 0.8,
                    gamma: 1.0,
                },
                ..stitch_domain::Levels::default()
            },
            ..EditSettings::default()
        });
        let after = developed(EditSettings {
            levels: stitch_domain::Levels {
                red: stitch_domain::ChannelLevels {
                    black: 0.2 + LEVEL_STEP as f32,
                    white: 0.8,
                    gamma: 1.0 + GAMMA_STEP as f32,
                },
                ..stitch_domain::Levels::default()
            },
            ..EditSettings::default()
        });
        assert!(
            worst_channel_change(&before, &after) <= 2,
            "one levels or gamma step moves a channel too far"
        );
    }

    #[test]
    fn crop_normalization_round_trips() {
        let crop = stitch_domain::CropRect {
            x: 100,
            y: 50,
            width: 400,
            height: 300,
        };
        let draft = crop_to_normalized(crop, 1000, 600);
        assert_eq!(crop_from_normalized(draft, 1000, 600), Some(crop));
    }

    #[test]
    fn crop_drag_stays_inside_the_image() {
        let image_rect =
            egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(200.0, 100.0));
        let origin = egui::Rect::from_min_max(egui::pos2(0.2, 0.2), egui::pos2(0.6, 0.6));
        let drag = CropDrag {
            handle: CropHandle::Move,
            origin,
            anchor: egui::pos2(0.4, 0.4),
        };
        let moved = update_crop_draft(drag, egui::pos2(1000.0, -500.0), image_rect);
        assert!(moved.min.x >= 0.0 && moved.min.y >= 0.0);
        assert!(moved.max.x <= 1.0 && moved.max.y <= 1.0);
        assert!((moved.width() - origin.width()).abs() <= f32::EPSILON);
        assert!((moved.height() - origin.height()).abs() <= f32::EPSILON);
    }

    #[test]
    fn crop_drag_can_start_from_a_click() {
        let image_rect =
            egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(200.0, 100.0));
        let anchor = screen_to_normalized(egui::pos2(60.0, 50.0), image_rect);
        let drag = CropDrag {
            handle: CropHandle::New,
            origin: egui::Rect::from_min_size(anchor, egui::Vec2::ZERO),
            anchor,
        };
        let draft = update_crop_draft(drag, egui::pos2(500.0, 500.0), image_rect);
        assert!(draft.width() > 0.0 && draft.height() > 0.0);
        assert!(draft.max.x <= 1.0 && draft.max.y <= 1.0);
    }
}
