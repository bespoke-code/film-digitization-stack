//! UI-agnostic presentation state.
//!
//! A UI adapter (egui today, Slint/Qt tomorrow) renders this state and calls the
//! methods below. No egui, windowing or GPU type appears in this crate, which is
//! what makes swapping the UI framework feasible.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use stitch_application::{ProjectService, StitchService};
use stitch_domain::{
    suggest_film_base, CoreError, CoreResult, CropRect, EditSettings, Homography, ImageId,
    ImageLocator, PhotoMetadata, Project, RgbaImage, SourceImage, StitchOptions, StitchProgress,
    StitchResult, StitchStage, MAX_IMAGES,
};

/// Longest side used for an edited preview.
const PREVIEW_DIMENSION: u32 = 3072;

/// What the preview viewport is currently showing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewTarget {
    Image(ImageId),
    Stitched,
}

/// Lifecycle of the current operation.
#[derive(Clone, Debug, PartialEq)]
pub enum WorkbenchStatus {
    Empty,
    Ready(String),
    Busy {
        stage: StitchStage,
        fraction: Option<f32>,
        message: String,
    },
    Failed(String),
}

impl WorkbenchStatus {
    pub fn is_busy(&self) -> bool {
        matches!(self, Self::Busy { .. })
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Empty => "Add up to sixteen overlapping images, then stitch.",
            Self::Ready(message) => message,
            Self::Busy { message, .. } => message,
            Self::Failed(message) => message,
        }
    }
}

/// Everything the viewport needs to draw a frame.
#[derive(Clone, Debug)]
pub struct PreviewInfo {
    pub image: Arc<RgbaImage>,
    pub label: String,
    pub revision: u64,
}

/// Preview of the rotated source before cropping, for the crop viewport.
#[derive(Clone, Debug)]
pub struct CropPreview {
    pub image: Arc<RgbaImage>,
    pub label: String,
    pub revision: u64,
    /// Full pixel size of the rotated source.
    pub source_width: u32,
    pub source_height: u32,
}

impl CropPreview {
    pub fn info(&self) -> PreviewInfo {
        PreviewInfo {
            image: self.image.clone(),
            label: self.label.clone(),
            revision: self.revision,
        }
    }
}

/// An immutable job handed to a background worker.
#[derive(Clone, Debug)]
pub struct StitchJob {
    pub images: Vec<SourceImage>,
    pub options: StitchOptions,
    /// Rough source-to-reference placements used when automatic matching fails.
    pub manual: Vec<Option<Homography>>,
    /// Clockwise rotation per image, in degrees.
    pub rotations: Vec<f32>,
    pub revision: u64,
}

#[derive(Clone, Debug)]
struct PreviewCache {
    revision: u64,
    target: PreviewTarget,
    image: Arc<RgbaImage>,
}

/// Geometry-dependent preview base, reused while only tone controls change.
#[derive(Clone, Debug)]
struct PreviewBaseCache {
    source: Arc<RgbaImage>,
    rotation: f32,
    crop: Option<CropRect>,
    image: Arc<RgbaImage>,
    target: PreviewTarget,
}

/// Downscaled, rotated source for the interactive crop viewport.
#[derive(Clone, Debug)]
struct CropPreviewCache {
    source: Arc<RgbaImage>,
    base: Arc<RgbaImage>,
    rotation: f32,
    image: Arc<RgbaImage>,
    source_width: u32,
    source_height: u32,
    target: PreviewTarget,
}

/// The UI-agnostic application state machine.
#[derive(Clone, Debug)]
pub struct Workbench {
    project: Project,
    options: StitchOptions,
    selected: Option<ImageId>,
    preview_target: PreviewTarget,
    result: Option<StitchResult>,
    revision: u64,
    status: WorkbenchStatus,
    edits: HashMap<ImageId, EditSettings>,
    result_edits: EditSettings,
    metadata: HashMap<ImageId, PhotoMetadata>,
    result_metadata: PhotoMetadata,
    manual: HashMap<ImageId, Homography>,
    show_edits: bool,
    preview_cache: RefCell<Option<PreviewCache>>,
    preview_base_cache: RefCell<Option<PreviewBaseCache>>,
    crop_preview_cache: RefCell<Option<CropPreviewCache>>,
}

impl Default for Workbench {
    fn default() -> Self {
        Self::new()
    }
}

impl Workbench {
    pub fn new() -> Self {
        Self {
            project: Project::new(),
            options: StitchOptions::default(),
            selected: None,
            preview_target: PreviewTarget::Stitched,
            result: None,
            revision: 0,
            status: WorkbenchStatus::Empty,
            edits: HashMap::new(),
            result_edits: EditSettings::default(),
            metadata: HashMap::new(),
            result_metadata: PhotoMetadata::default(),
            manual: HashMap::new(),
            show_edits: true,
            preview_cache: RefCell::new(None),
            preview_base_cache: RefCell::new(None),
            crop_preview_cache: RefCell::new(None),
        }
    }

    pub fn project(&self) -> &Project {
        &self.project
    }

    pub fn options(&self) -> &StitchOptions {
        &self.options
    }

    pub fn status(&self) -> &WorkbenchStatus {
        &self.status
    }

    pub fn result(&self) -> Option<&StitchResult> {
        self.result.as_ref()
    }

    pub fn selected(&self) -> Option<ImageId> {
        self.selected
    }

    pub fn preview_target(&self) -> PreviewTarget {
        self.preview_target
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The unedited image currently selected for preview, with rotation applied.
    pub fn base_preview(&self) -> Option<PreviewInfo> {
        let (image, label) = self.base_image()?;
        let rotation = self.current_edits().rotation;
        let image = if stitch_domain::rotation_is_identity(rotation) {
            image
        } else {
            Arc::new(image.rotated(rotation))
        };
        Some(PreviewInfo {
            image,
            label,
            revision: self.revision,
        })
    }

    /// The rotated source without cropping, downscaled for the crop viewport.
    ///
    /// The full rotated dimensions are returned so the UI can turn a
    /// normalized selection into source-pixel crop coordinates.
    pub fn crop_preview(&self) -> Option<CropPreview> {
        let (base, label) = self.base_image()?;
        let rotation = self.current_edits().rotation;
        let cached = {
            let cache = self.crop_preview_cache.borrow();
            cache
                .as_ref()
                .filter(|cache| {
                    cache.target == self.preview_target && Arc::ptr_eq(&cache.source, &base)
                })
                .map(|cache| {
                    (
                        cache.base.clone(),
                        cache.rotation,
                        cache.image.clone(),
                        cache.source_width,
                        cache.source_height,
                    )
                })
        };
        let (image, source_width, source_height) = match cached {
            Some((_, cached_rotation, cached_image, width, height))
                if cached_rotation.to_bits() == rotation.to_bits() =>
            {
                (cached_image, width, height)
            }
            Some((cached_base, _, _, _, _)) => {
                let (image, width, height) = rotated_crop_image(&cached_base, &base, rotation);
                *self.crop_preview_cache.borrow_mut() = Some(CropPreviewCache {
                    source: base.clone(),
                    base: cached_base,
                    rotation,
                    image: image.clone(),
                    source_width: width,
                    source_height: height,
                    target: self.preview_target,
                });
                (image, width, height)
            }
            None => {
                let fitted = Arc::new(base.resized_to_fit(PREVIEW_DIMENSION));
                let (image, width, height) = rotated_crop_image(&fitted, &base, rotation);
                *self.crop_preview_cache.borrow_mut() = Some(CropPreviewCache {
                    source: base.clone(),
                    base: fitted,
                    rotation,
                    image: image.clone(),
                    source_width: width,
                    source_height: height,
                    target: self.preview_target,
                });
                (image, width, height)
            }
        };
        Some(CropPreview {
            image,
            label,
            revision: self.revision,
            source_width,
            source_height,
        })
    }

    fn base_image(&self) -> Option<(Arc<RgbaImage>, String)> {
        match self.preview_target {
            PreviewTarget::Stitched => self
                .result
                .as_ref()
                .map(|result| (result.image.clone(), "Stitched panorama".to_owned())),
            PreviewTarget::Image(id) => self
                .project
                .get(id)
                .map(|image| (image.image.clone(), image.name.clone())),
        }
    }

    pub fn edits(&self, id: ImageId) -> EditSettings {
        self.edits.get(&id).cloned().unwrap_or_default()
    }

    /// Descriptive metadata for one source image.
    pub fn image_metadata(&self, id: ImageId) -> PhotoMetadata {
        self.metadata.get(&id).cloned().unwrap_or_default()
    }

    /// Metadata for the current preview: the panorama or the selected image.
    pub fn has_preview(&self) -> bool {
        match self.preview_target {
            PreviewTarget::Stitched => self.result.is_some(),
            PreviewTarget::Image(_) => !self.project.is_empty(),
        }
    }

    pub fn current_metadata(&self) -> PhotoMetadata {
        match self.preview_target {
            PreviewTarget::Stitched => self.result_metadata.clone(),
            PreviewTarget::Image(id) => self.image_metadata(id),
        }
    }

    pub fn update_image_metadata(&mut self, id: ImageId, metadata: PhotoMetadata) {
        if self.project.get(id).is_none() {
            return;
        }
        let metadata = metadata.bounded();
        if self.metadata.get(&id) != Some(&metadata) {
            self.metadata.insert(id, metadata);
        }
    }

    pub fn update_current_metadata(&mut self, metadata: PhotoMetadata) {
        match self.preview_target {
            PreviewTarget::Stitched => {
                let metadata = metadata.bounded();
                if self.result_metadata != metadata {
                    self.result_metadata = metadata;
                }
            }
            PreviewTarget::Image(id) => self.update_image_metadata(id, metadata),
        }
    }

    pub fn reset_current_metadata(&mut self) {
        self.update_current_metadata(PhotoMetadata::default());
    }

    /// Copies metadata to every image and the panorama for batch work.
    pub fn apply_metadata_to_all(&mut self, metadata: PhotoMetadata) {
        let metadata = metadata.bounded();
        let mut changed = false;
        for image in self.project.images() {
            if self.metadata.get(&image.id) != Some(&metadata) {
                self.metadata.insert(image.id, metadata.clone());
                changed = true;
            }
        }
        if self.result.is_some() && self.result_metadata != metadata {
            self.result_metadata = metadata;
            changed = true;
        }
        if changed {
            self.set_status(WorkbenchStatus::Ready(format!(
                "Applied metadata to {} image{}.",
                self.project.len(),
                if self.project.len() == 1 { "" } else { "s" }
            )));
        }
    }

    /// Rough placement of one image in the reference frame, when the user set it.
    pub fn manual_placement(&self, id: ImageId) -> Option<Homography> {
        self.manual.get(&id).copied()
    }

    /// Rough placements in project order. The reference image has no entry.
    pub fn manual_placements(&self) -> Vec<Option<Homography>> {
        self.project
            .images()
            .iter()
            .enumerate()
            .map(|(index, image)| {
                if index == 0 {
                    None
                } else {
                    self.manual.get(&image.id).copied()
                }
            })
            .collect()
    }

    /// Stores a rough placement. Non-finite transforms are ignored.
    pub fn set_manual_placement(&mut self, id: ImageId, homography: Homography) {
        if self.project.get(id).is_none() || !homography.is_finite() {
            return;
        }
        self.manual.insert(id, homography);
    }

    pub fn clear_manual_placements(&mut self) {
        self.manual.clear();
    }

    pub fn current_edits(&self) -> EditSettings {
        match self.preview_target {
            PreviewTarget::Stitched => self.result_edits.clone(),
            PreviewTarget::Image(id) => self.edits(id),
        }
    }

    pub fn show_edits(&self) -> bool {
        self.show_edits
    }

    pub fn set_show_edits(&mut self, show: bool) {
        if self.show_edits != show {
            self.show_edits = show;
            self.bump_revision();
        }
    }

    pub fn update_edits(&mut self, id: ImageId, settings: EditSettings) {
        if self.project.get(id).is_none() {
            return;
        }
        let settings = settings.sanitized();
        let rotation_changed = self
            .edits
            .get(&id)
            .map(|current| (current.rotation - settings.rotation).abs() > f32::EPSILON)
            .unwrap_or(settings.rotation != 0.0);
        if self.edits.get(&id) != Some(&settings) {
            self.edits.insert(id, settings);
            if rotation_changed {
                if self.project.position(id) == Some(0) {
                    self.manual.clear();
                } else {
                    self.manual.remove(&id);
                }
            }
            self.bump_revision();
        }
    }

    pub fn update_current_edits(&mut self, settings: EditSettings) {
        match self.preview_target {
            PreviewTarget::Stitched => {
                let settings = settings.sanitized();
                if self.result_edits != settings {
                    self.result_edits = settings;
                    self.bump_revision();
                }
            }
            PreviewTarget::Image(id) => self.update_edits(id, settings),
        }
    }

    /// Copies the given adjustments to every loaded image for batch work.
    ///
    /// The crop is intentionally left alone because it is defined in each
    /// image's own pixel coordinates. The stitched panorama receives the same
    /// tone and color settings.
    pub fn apply_edits_to_all(&mut self, settings: EditSettings) {
        let mut settings = settings.sanitized();
        settings.crop = None;
        let mut changed = false;
        for image in self.project.images() {
            let mut target = settings.clone();
            target.rotation = self
                .edits
                .get(&image.id)
                .map(|current| current.rotation)
                .unwrap_or(0.0);
            if self.edits.get(&image.id) != Some(&target) {
                self.edits.insert(image.id, target);
                changed = true;
            }
        }
        if self.result.is_some() {
            let mut target = settings.clone();
            target.rotation = self.result_edits.rotation;
            if self.result_edits != target {
                self.result_edits = target;
                changed = true;
            }
        }
        if changed {
            self.bump_revision();
            let panorama = if self.result.is_some() {
                " and the panorama"
            } else {
                ""
            };
            self.set_status(WorkbenchStatus::Ready(format!(
                "Applied adjustments to {} image{}{}.",
                self.project.len(),
                if self.project.len() == 1 { "" } else { "s" },
                panorama
            )));
        }
    }

    pub fn reset_current_edits(&mut self) {
        self.update_current_edits(EditSettings::default());
    }

    /// Suggests a film base from the selected preview and stores it.
    pub fn auto_film_base(&mut self) -> CoreResult<[u8; 3]> {
        let (image, _) = self.base_image().ok_or(CoreError::NoStitchResult)?;
        let base = suggest_film_base(&image);
        let mut settings = self.current_edits();
        settings.film_base = Some(base);
        self.update_current_edits(settings);
        Ok(base)
    }

    pub fn is_busy(&self) -> bool {
        self.status.is_busy()
    }

    pub fn can_add(&self) -> bool {
        !self.project.is_full()
    }

    pub fn can_stitch(&self) -> bool {
        !self.project.is_empty() && !self.is_busy()
    }

    pub fn can_export(&self) -> bool {
        !self.is_busy() && (self.result.is_some() || self.selected.is_some())
    }

    pub fn remaining_slots(&self) -> usize {
        self.project.remaining_slots()
    }

    pub fn update_options(&mut self, options: StitchOptions) {
        let options = options.sanitized();
        if options != self.options {
            self.options = options;
            self.invalidate_result();
            self.set_status(WorkbenchStatus::Ready(
                "Settings changed; stitch again to apply them.".to_owned(),
            ));
        }
    }

    /// Loads and appends images. The project is unchanged if loading fails.
    pub fn add_images(
        &mut self,
        service: &ProjectService,
        locators: &[ImageLocator],
    ) -> CoreResult<Vec<ImageId>> {
        if locators.is_empty() {
            return Ok(Vec::new());
        }
        if self.project.is_full() {
            return Err(CoreError::TooManyImages { max: MAX_IMAGES });
        }
        let added = service.add_images_with_metadata(&mut self.project, locators)?;
        let mut ids = Vec::with_capacity(added.len());
        for (id, metadata) in added {
            self.edits.insert(id, EditSettings::default());
            self.metadata.insert(id, metadata);
            ids.push(id);
        }
        if let Some(&id) = ids.first() {
            self.selected = Some(id);
        }
        self.invalidate_result();
        self.preview_target = self
            .selected
            .map(PreviewTarget::Image)
            .unwrap_or(PreviewTarget::Stitched);
        self.set_status(WorkbenchStatus::Ready(format!(
            "Added {} image{} ({} of {}).",
            ids.len(),
            if ids.len() == 1 { "" } else { "s" },
            self.project.len(),
            MAX_IMAGES
        )));
        Ok(ids)
    }

    pub fn remove_image(&mut self, id: ImageId) -> CoreResult<()> {
        self.project.remove(id)?;
        self.edits.remove(&id);
        self.metadata.remove(&id);
        self.manual.clear();
        if self.selected == Some(id) {
            self.selected = self.project.images().first().map(|image| image.id);
        }
        self.invalidate_result();
        self.preview_target = self
            .selected
            .map(PreviewTarget::Image)
            .unwrap_or(PreviewTarget::Stitched);
        if self.project.is_empty() {
            self.set_status(WorkbenchStatus::Empty);
        } else {
            self.set_status(WorkbenchStatus::Ready(format!(
                "Removed image; {} remaining.",
                self.project.len()
            )));
        }
        Ok(())
    }

    pub fn move_up(&mut self, id: ImageId) -> CoreResult<()> {
        self.project.move_up(id)?;
        self.manual.clear();
        self.invalidate_result();
        self.set_status(WorkbenchStatus::Ready("Image order updated.".to_owned()));
        Ok(())
    }

    pub fn move_down(&mut self, id: ImageId) -> CoreResult<()> {
        self.project.move_down(id)?;
        self.manual.clear();
        self.invalidate_result();
        self.set_status(WorkbenchStatus::Ready("Image order updated.".to_owned()));
        Ok(())
    }

    pub fn clear(&mut self) {
        self.project.clear();
        self.selected = None;
        self.result = None;
        self.edits.clear();
        self.metadata.clear();
        self.manual.clear();
        self.result_edits = EditSettings::default();
        self.result_metadata = PhotoMetadata::default();
        self.preview_target = PreviewTarget::Stitched;
        self.bump_revision();
        self.set_status(WorkbenchStatus::Empty);
    }

    pub fn select_image(&mut self, id: ImageId) -> CoreResult<()> {
        if self.project.get(id).is_none() {
            return Err(CoreError::UnknownImage(id.0));
        }
        self.selected = Some(id);
        self.preview_target = PreviewTarget::Image(id);
        self.bump_revision();
        Ok(())
    }

    pub fn show_stitched(&mut self) -> CoreResult<()> {
        if self.result.is_none() {
            return Err(CoreError::NoStitchResult);
        }
        self.preview_target = PreviewTarget::Stitched;
        self.bump_revision();
        Ok(())
    }

    pub fn preview(&self) -> Option<PreviewInfo> {
        let (base, label) = self.base_image()?;
        let settings = self.current_edits();
        let image = if self.show_edits && !settings.is_identity() {
            if let Some(cache) = self.preview_cache.borrow().as_ref() {
                if cache.revision == self.revision && cache.target == self.preview_target {
                    return Some(PreviewInfo {
                        image: cache.image.clone(),
                        label,
                        revision: self.revision,
                    });
                }
            }
            let prepared = self.preview_base(&base, &settings);
            let developed = Arc::new(settings.develop_tone(&prepared));
            *self.preview_cache.borrow_mut() = Some(PreviewCache {
                revision: self.revision,
                target: self.preview_target,
                image: developed.clone(),
            });
            developed
        } else {
            base
        };
        Some(PreviewInfo {
            image,
            label,
            revision: self.revision,
        })
    }

    /// Rotation, crop and fitting are expensive but do not change while a tone
    /// slider moves, so the result is cached per source image and geometry.
    fn preview_base(&self, base: &Arc<RgbaImage>, settings: &EditSettings) -> Arc<RgbaImage> {
        if let Some(cache) = self.preview_base_cache.borrow().as_ref() {
            if cache.target == self.preview_target
                && Arc::ptr_eq(&cache.source, base)
                && cache.rotation.to_bits() == settings.rotation.to_bits()
                && cache.crop == settings.crop
            {
                return cache.image.clone();
            }
        }
        let prepared = settings.prepared(base);
        let fitted = Arc::new(prepared.as_ref().resized_to_fit(PREVIEW_DIMENSION));
        *self.preview_base_cache.borrow_mut() = Some(PreviewBaseCache {
            source: base.clone(),
            rotation: settings.rotation,
            crop: settings.crop,
            image: fitted.clone(),
            target: self.preview_target,
        });
        fitted
    }

    /// Creates a snapshot for a background worker and marks the workbench busy.
    pub fn begin_stitch(&mut self) -> CoreResult<StitchJob> {
        if self.project.is_empty() {
            return Err(CoreError::EmptyProject);
        }
        if self.is_busy() {
            return Err(CoreError::InvalidImage(
                "a stitch operation is already running".to_owned(),
            ));
        }
        self.set_status(WorkbenchStatus::Busy {
            stage: StitchStage::ExtractingFeatures,
            fraction: Some(0.0),
            message: "Preparing…".to_owned(),
        });
        Ok(StitchJob {
            images: self.project.images().to_vec(),
            options: self.options.clone(),
            manual: self.manual_placements(),
            rotations: self
                .project
                .images()
                .iter()
                .map(|image| self.edits(image.id).rotation)
                .collect(),
            revision: self.revision,
        })
    }

    pub fn report_progress(&mut self, progress: StitchProgress) {
        self.set_status(WorkbenchStatus::Busy {
            stage: progress.stage,
            fraction: progress.fraction(),
            message: progress.message,
        });
    }

    pub fn report_fraction(
        &mut self,
        stage: StitchStage,
        fraction: f32,
        message: impl Into<String>,
    ) {
        self.set_status(WorkbenchStatus::Busy {
            stage,
            fraction: Some(fraction.clamp(0.0, 1.0)),
            message: message.into(),
        });
    }

    pub fn finish_stitch(&mut self, result: StitchResult) {
        let summary = format!(
            "Panorama {}×{} from {} images in {:.1}s ({} matches, {} inliers).",
            result.diagnostics.output_width,
            result.diagnostics.output_height,
            result.placements.len(),
            result.diagnostics.duration_ms as f64 / 1000.0,
            result
                .diagnostics
                .pairs
                .iter()
                .map(|pair| pair.num_matches)
                .sum::<usize>(),
            result
                .diagnostics
                .pairs
                .iter()
                .filter(|pair| pair.used_for_alignment)
                .map(|pair| pair.num_inliers)
                .sum::<usize>(),
        );
        let size_changed = self.result.as_ref().is_some_and(|previous| {
            previous.image.width() != result.image.width()
                || previous.image.height() != result.image.height()
        });
        if self.result_edits.is_identity() {
            if let Some(image) = self.project.images().first() {
                let mut inherited = self.edits(image.id);
                inherited.crop = None;
                inherited.rotation = 0.0;
                self.result_edits = inherited;
            }
        } else if size_changed {
            self.result_edits.crop = None;
        }
        if self.result.is_none() && self.result_metadata.is_empty() {
            if let Some(image) = self.project.images().first() {
                self.result_metadata = self.image_metadata(image.id);
            }
        }
        self.result = Some(result);
        self.preview_target = PreviewTarget::Stitched;
        self.bump_revision();
        self.set_status(WorkbenchStatus::Ready(summary));
    }

    pub fn fail_stitch(&mut self, message: impl Into<String>) {
        self.set_status(WorkbenchStatus::Failed(message.into()));
    }

    pub fn acknowledge_error(&mut self) {
        if matches!(self.status, WorkbenchStatus::Failed(_)) {
            self.set_status(WorkbenchStatus::Ready(
                "Fix the problem and try again.".to_owned(),
            ));
        }
    }

    /// Shows a short informational message without changing the project.
    pub fn notify(&mut self, message: impl Into<String>) {
        if !self.is_busy() {
            self.set_status(WorkbenchStatus::Ready(message.into()));
        }
    }

    pub fn export(&self, service: &ProjectService, locator: &ImageLocator) -> CoreResult<()> {
        let (base, _) = self.base_image().ok_or(CoreError::NoStitchResult)?;
        let image = self.current_edits().develop(&base);
        service.export_with_metadata(locator, &image, Some(&self.current_metadata()))
    }

    /// Writes every loaded image and, when present, the panorama to a folder.
    pub fn export_all(
        &self,
        service: &ProjectService,
        folder: &Path,
        extension: &str,
    ) -> CoreResult<usize> {
        if self.project.is_empty() {
            return Err(CoreError::EmptyProject);
        }
        let mut written = 0;
        for image in self.project.images() {
            let stem = export_stem(&image.locator, image.id.0);
            let path = folder.join(format!("{stem}-edited.{extension}"));
            let developed = self.edits(image.id).develop(&image.image);
            service.export_with_metadata(
                &ImageLocator::from(path.as_path()),
                &developed,
                Some(&self.image_metadata(image.id)),
            )?;
            written += 1;
        }
        if let Some(result) = &self.result {
            let stem = self
                .project
                .images()
                .first()
                .map(|image| export_stem(&image.locator, image.id.0))
                .unwrap_or_else(|| "panorama".to_owned());
            let path = folder.join(format!("{stem}-stitched.{extension}"));
            let developed = self.result_edits.develop(&result.image);
            service.export_with_metadata(
                &ImageLocator::from(path.as_path()),
                &developed,
                Some(&self.result_metadata),
            )?;
            written += 1;
        }
        Ok(written)
    }

    /// Runs the use case synchronously. Kept for tests, headless use and simple
    /// integrations; the egui adapter uses [`Self::begin_stitch`] on a worker.
    pub fn stitch_sync(&mut self, service: &StitchService) -> CoreResult<()> {
        let job = self.begin_stitch()?;
        match service.stitch_with_rotations(
            &job.images,
            &job.rotations,
            &job.manual,
            &job.options,
            &stitch_domain::NoProgress,
        ) {
            Ok(result) => {
                self.finish_stitch(result);
                Ok(())
            }
            Err(error) => {
                self.fail_stitch(error.to_string());
                Err(error)
            }
        }
    }

    fn invalidate_result(&mut self) {
        if self.result.take().is_some() {
            self.preview_target = self
                .selected
                .map(PreviewTarget::Image)
                .unwrap_or(PreviewTarget::Stitched);
            self.bump_revision();
        }
    }

    fn set_status(&mut self, status: WorkbenchStatus) {
        self.status = status;
    }

    fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }
}

fn rotated_crop_image(
    fitted: &Arc<RgbaImage>,
    full: &RgbaImage,
    rotation: f32,
) -> (Arc<RgbaImage>, u32, u32) {
    let (source_width, source_height) =
        RgbaImage::rotated_dimensions(full.width(), full.height(), rotation);
    let image = if stitch_domain::rotation_is_identity(rotation) {
        fitted.clone()
    } else {
        Arc::new(fitted.rotated(rotation))
    };
    (image, source_width, source_height)
}

fn export_stem(locator: &ImageLocator, fallback: u64) -> String {
    Path::new(locator.as_str())
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| format!("image-{fallback}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::hint::black_box;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    use stitch_domain::{ImageRepository, LoadedImage};

    /// One display frame at 60 Hz.
    const FRAME_BUDGET: Duration = Duration::from_micros(16_667);

    fn patterned_image(width: u32, height: u32) -> RgbaImage {
        RgbaImage::from_fn(width, height, |x, y| {
            let noise = x.wrapping_mul(31) ^ y.wrapping_mul(17);
            [noise as u8, (noise >> 8) as u8, (noise >> 16) as u8, 255]
        })
    }

    #[test]
    fn crop_preview_uses_rotated_dimensions_without_crop() {
        let repo = Arc::new(MemoryRepo::default());
        let name = "crop.png".to_owned();
        repo.images.lock().unwrap().insert(
            name.clone(),
            LoadedImage {
                locator: ImageLocator::new(name.clone()),
                name,
                image: patterned_image(400, 200),
                thumbnail: RgbaImage::filled(20, 10, [0, 0, 0, 255]),
                metadata: PhotoMetadata::default(),
            },
        );
        let mut workbench = Workbench::new();
        let service = ProjectService::new(repo);
        workbench
            .add_images(&service, &[ImageLocator::new("crop.png")])
            .unwrap();
        let id = workbench.selected().unwrap();
        workbench.update_edits(
            id,
            EditSettings {
                rotation: 90.0,
                crop: Some(CropRect {
                    x: 10,
                    y: 10,
                    width: 50,
                    height: 50,
                }),
                exposure: 1.0,
                ..EditSettings::default()
            },
        );

        let preview = workbench.crop_preview().unwrap();
        assert_eq!((preview.source_width, preview.source_height), (200, 400));
        assert_eq!((preview.image.width(), preview.image.height()), (200, 400));
    }

    #[test]
    #[ignore = "performance test; run with `just perf`"]
    fn repeated_tone_slider_changes_render_within_a_frame() {
        let repo = Arc::new(MemoryRepo::default());
        let name = "large.png".to_owned();
        repo.images.lock().unwrap().insert(
            name.clone(),
            LoadedImage {
                locator: ImageLocator::new(name.clone()),
                name,
                image: patterned_image(3840, 2560),
                thumbnail: RgbaImage::filled(20, 15, [0, 0, 0, 255]),
                metadata: PhotoMetadata::default(),
            },
        );
        let mut workbench = Workbench::new();
        let service = ProjectService::new(repo);
        workbench
            .add_images(&service, &[ImageLocator::new("large.png")])
            .unwrap();
        let id = workbench.selected().unwrap();

        let render = |workbench: &mut Workbench, exposure: f32| {
            workbench.update_edits(
                id,
                EditSettings {
                    invert: true,
                    film_base: Some([190, 140, 100]),
                    exposure,
                    contrast: 0.3,
                    levels: stitch_domain::Levels {
                        red: stitch_domain::ChannelLevels {
                            black: 0.03,
                            white: 0.92,
                            gamma: 1.05,
                        },
                        ..stitch_domain::Levels::default()
                    },
                    ..EditSettings::default()
                },
            );
            black_box(workbench.preview());
        };

        // The first render includes the one-off rotation, crop and downscale.
        render(&mut workbench, 0.35);
        let best = (0..6)
            .map(|sample| {
                let start = Instant::now();
                render(&mut workbench, 0.35 + sample as f32 * 0.01);
                start.elapsed()
            })
            .min()
            .unwrap_or_default();
        assert!(
            best < FRAME_BUDGET,
            "tone preview took {best:?}, above the {FRAME_BUDGET:?} frame budget"
        );
    }

    #[test]
    #[ignore = "performance test; run with `just perf`"]
    fn repeated_crop_preview_frames_render_within_a_frame() {
        let repo = Arc::new(MemoryRepo::default());
        let name = "crop-large.png".to_owned();
        repo.images.lock().unwrap().insert(
            name.clone(),
            LoadedImage {
                locator: ImageLocator::new(name.clone()),
                name,
                image: patterned_image(6000, 4000),
                thumbnail: RgbaImage::filled(20, 15, [0, 0, 0, 255]),
                metadata: PhotoMetadata::default(),
            },
        );
        let mut workbench = Workbench::new();
        let service = ProjectService::new(repo);
        workbench
            .add_images(&service, &[ImageLocator::new("crop-large.png")])
            .unwrap();
        let id = workbench.selected().unwrap();
        workbench.update_edits(
            id,
            EditSettings {
                rotation: 3.0,
                ..EditSettings::default()
            },
        );

        // The first frame builds the downscaled base; later frames must be
        // immediate while the user drags crop handles.
        black_box(workbench.crop_preview());
        let best = (0..6)
            .map(|_| {
                let start = Instant::now();
                black_box(workbench.crop_preview());
                start.elapsed()
            })
            .min()
            .unwrap_or_default();
        assert!(
            best < FRAME_BUDGET,
            "crop preview took {best:?}, above the {FRAME_BUDGET:?} frame budget"
        );
    }

    #[derive(Default)]
    struct MemoryRepo {
        images: Mutex<HashMap<String, LoadedImage>>,
        saved: Mutex<Vec<(String, RgbaImage)>>,
        saved_metadata: Mutex<Vec<(String, Option<PhotoMetadata>)>>,
    }

    impl ImageRepository for MemoryRepo {
        fn load(&self, locator: &ImageLocator) -> CoreResult<LoadedImage> {
            self.images
                .lock()
                .unwrap()
                .get(locator.as_str())
                .cloned()
                .ok_or_else(|| CoreError::NotFound(locator.as_str().to_owned()))
        }

        fn save(&self, locator: &ImageLocator, image: &RgbaImage) -> CoreResult<()> {
            self.saved
                .lock()
                .unwrap()
                .push((locator.as_str().to_owned(), image.clone()));
            Ok(())
        }

        fn save_with_metadata(
            &self,
            locator: &ImageLocator,
            image: &RgbaImage,
            metadata: Option<&PhotoMetadata>,
        ) -> CoreResult<()> {
            self.saved_metadata
                .lock()
                .unwrap()
                .push((locator.as_str().to_owned(), metadata.cloned()));
            self.save(locator, image)
        }
    }

    fn repo_with(count: usize) -> Arc<MemoryRepo> {
        let repo = Arc::new(MemoryRepo::default());
        let mut map = repo.images.lock().unwrap();
        for index in 0..count {
            let name = format!("image-{index}.png");
            map.insert(
                name.clone(),
                LoadedImage {
                    locator: ImageLocator::new(name.clone()),
                    name,
                    image: RgbaImage::filled(40, 30, [index as u8, 0, 0, 255]),
                    thumbnail: RgbaImage::filled(20, 15, [0, 0, 0, 255]),
                    metadata: PhotoMetadata::default(),
                },
            );
        }
        drop(map);
        repo
    }

    fn add(workbench: &mut Workbench, repo: &Arc<MemoryRepo>, count: usize) {
        let service = ProjectService::new(repo.clone());
        let locators: Vec<ImageLocator> = (0..count)
            .map(|index| ImageLocator::new(format!("image-{index}.png")))
            .collect();
        workbench.add_images(&service, &locators).unwrap();
    }

    #[test]
    fn add_select_remove_flow() {
        let repo = repo_with(3);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 3);
        assert_eq!(workbench.project().len(), 3);
        assert_eq!(workbench.preview_target(), PreviewTarget::Image(ImageId(1)));
        let id = workbench.project().images()[2].id;
        workbench.select_image(id).unwrap();
        assert_eq!(workbench.preview_target(), PreviewTarget::Image(id));
        workbench.remove_image(id).unwrap();
        assert_eq!(workbench.project().len(), 2);
        assert!(workbench.preview().is_some());
    }

    #[test]
    fn cannot_add_more_than_sixteen() {
        let repo = repo_with(17);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 16);
        let service = ProjectService::new(repo.clone());
        let error = workbench
            .add_images(&service, &[ImageLocator::new("image-16.png")])
            .unwrap_err();
        assert!(matches!(error, CoreError::TooManyImages { max: 16 }));
    }

    #[test]
    fn failed_load_does_not_modify_project() {
        let repo = repo_with(1);
        let mut workbench = Workbench::new();
        let service = ProjectService::new(repo.clone());
        let result = workbench.add_images(
            &service,
            &[
                ImageLocator::new("image-0.png"),
                ImageLocator::new("missing.png"),
            ],
        );
        assert!(result.is_err());
        assert!(workbench.project().is_empty());
    }

    #[test]
    fn reordering_invalidates_result_and_updates_status() {
        let repo = repo_with(2);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 2);
        let first = workbench.project().images()[0].id;
        workbench.move_down(first).unwrap();
        assert_eq!(workbench.project().images()[1].id, first);
        assert!(matches!(workbench.status(), WorkbenchStatus::Ready(_)));
    }

    #[test]
    fn export_without_any_image_fails() {
        let repo = repo_with(0);
        let workbench = Workbench::new();
        let service = ProjectService::new(repo);
        assert_eq!(
            workbench
                .export(&service, &ImageLocator::new("out.png"))
                .unwrap_err(),
            CoreError::NoStitchResult
        );
    }

    #[test]
    fn inversion_applies_to_preview_and_export() {
        let repo = repo_with(1);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 1);
        let id = workbench.selected().unwrap();
        let edits = EditSettings {
            invert: true,
            ..EditSettings::default()
        };
        workbench.update_edits(id, edits);
        let preview = workbench.preview().unwrap();
        assert_eq!(preview.image.pixel(0, 0)[0], 255);

        let service = ProjectService::new(repo.clone());
        workbench
            .export(&service, &ImageLocator::new("out.png"))
            .unwrap();
        let saved = repo.saved.lock().unwrap();
        assert_eq!(saved[0].1.pixel(0, 0)[0], 255);
    }

    #[test]
    fn batch_edits_reach_every_image() {
        let repo = repo_with(3);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 3);
        let edits = EditSettings {
            invert: true,
            monochrome: true,
            ..EditSettings::default()
        };
        workbench.apply_edits_to_all(edits.clone());
        for image in workbench.project().images() {
            assert_eq!(workbench.edits(image.id), edits);
        }
    }

    #[test]
    fn batch_edits_reach_the_panorama_without_copying_crops() {
        let repo = repo_with(2);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 2);
        workbench.finish_stitch(StitchResult {
            image: Arc::new(RgbaImage::filled(8, 8, [10, 20, 30, 255])),
            placements: Vec::new(),
            diagnostics: stitch_domain::StitchDiagnostics::default(),
        });
        let edits = EditSettings {
            invert: true,
            crop: Some(stitch_domain::CropRect {
                x: 1,
                y: 1,
                width: 4,
                height: 4,
            }),
            ..EditSettings::default()
        };
        workbench.apply_edits_to_all(edits);
        workbench.show_stitched().unwrap();
        let panorama = workbench.current_edits();
        assert!(panorama.invert);
        assert!(panorama.crop.is_none());
        for image in workbench.project().images() {
            assert!(workbench.edits(image.id).invert);
            assert!(workbench.edits(image.id).crop.is_none());
        }
    }

    #[test]
    fn batch_export_writes_every_image() {
        let repo = repo_with(2);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 2);
        workbench.update_edits(
            workbench.selected().unwrap(),
            EditSettings {
                invert: true,
                ..EditSettings::default()
            },
        );
        let service = ProjectService::new(repo.clone());
        let count = workbench
            .export_all(&service, Path::new("/tmp"), "png")
            .unwrap();
        assert_eq!(count, 2);
        let saved = repo.saved.lock().unwrap();
        assert!(saved[0].0.ends_with("image-0-edited.png"));
        assert!(saved[1].0.ends_with("image-1-edited.png"));
        assert_eq!(saved[0].1.pixel(0, 0)[0], 255);
        assert_eq!(saved[1].1.pixel(0, 0)[0], 1);
    }

    #[test]
    fn hiding_edits_shows_the_original() {
        let repo = repo_with(1);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 1);
        let id = workbench.selected().unwrap();
        workbench.update_edits(
            id,
            EditSettings {
                invert: true,
                ..EditSettings::default()
            },
        );
        workbench.set_show_edits(false);
        assert_eq!(workbench.preview().unwrap().image.pixel(0, 0)[0], 0);
        workbench.set_show_edits(true);
        assert_eq!(workbench.preview().unwrap().image.pixel(0, 0)[0], 255);
    }

    #[test]
    fn busy_state_blocks_a_second_job() {
        let repo = repo_with(1);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 1);
        let job = workbench.begin_stitch().unwrap();
        assert_eq!(job.images.len(), 1);
        assert!(workbench.is_busy());
        assert!(workbench.begin_stitch().is_err());
        workbench.fail_stitch("cancelled");
        assert!(!workbench.is_busy());
    }

    #[test]
    fn panorama_keeps_tone_edits_across_restitches() {
        let repo = repo_with(1);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 1);
        let id = workbench.selected().unwrap();
        workbench.update_edits(
            id,
            EditSettings {
                invert: true,
                crop: Some(stitch_domain::CropRect {
                    x: 1,
                    y: 1,
                    width: 4,
                    height: 4,
                }),
                ..EditSettings::default()
            },
        );
        let result = |width, height| StitchResult {
            image: Arc::new(RgbaImage::filled(width, height, [10, 20, 30, 255])),
            placements: Vec::new(),
            diagnostics: stitch_domain::StitchDiagnostics::default(),
        };
        workbench.finish_stitch(result(8, 8));
        let panorama = workbench.current_edits();
        assert!(panorama.invert);
        assert!(panorama.crop.is_none());

        workbench.update_current_edits(EditSettings {
            invert: true,
            crop: Some(stitch_domain::CropRect {
                x: 2,
                y: 2,
                width: 4,
                height: 4,
            }),
            ..EditSettings::default()
        });
        workbench.finish_stitch(result(8, 8));
        assert!(workbench.current_edits().crop.is_some());

        workbench.finish_stitch(result(12, 8));
        let panorama = workbench.current_edits();
        assert!(panorama.invert);
        assert!(panorama.crop.is_none());
    }

    #[test]
    fn rough_placements_travel_with_the_job() {
        let repo = repo_with(2);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 2);
        let second = workbench.project().images()[1].id;
        let placement = Homography::translation(12.0, 4.0);
        workbench.set_manual_placement(second, placement);
        let job = workbench.begin_stitch().unwrap();
        assert_eq!(job.manual, vec![None, Some(placement)]);
        workbench.fail_stitch("cancelled");

        workbench.move_down(second).unwrap();
        assert_eq!(workbench.manual_placements(), vec![None, None]);
    }

    #[test]
    fn rotation_reaches_the_job_and_the_base_preview() {
        let repo = repo_with(2);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 2);
        let first = workbench.project().images()[0].id;
        workbench.update_edits(
            first,
            EditSettings {
                rotation: 90.0,
                ..EditSettings::default()
            },
        );
        let preview = workbench.base_preview().unwrap();
        assert_eq!((preview.image.width(), preview.image.height()), (30, 40));
        let job = workbench.begin_stitch().unwrap();
        assert_eq!(job.rotations, vec![90.0, 0.0]);
    }

    #[test]
    fn batch_edits_preserve_each_image_rotation() {
        let repo = repo_with(2);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 2);
        let first = workbench.project().images()[0].id;
        let second = workbench.project().images()[1].id;
        workbench.update_edits(
            first,
            EditSettings {
                rotation: 90.0,
                ..EditSettings::default()
            },
        );
        workbench.update_edits(
            second,
            EditSettings {
                rotation: 180.0,
                ..EditSettings::default()
            },
        );
        workbench.apply_edits_to_all(EditSettings {
            invert: true,
            rotation: 45.0,
            ..EditSettings::default()
        });
        assert!(workbench.edits(first).invert);
        assert!(workbench.edits(second).invert);
        assert_eq!(workbench.edits(first).rotation, 90.0);
        assert_eq!(workbench.edits(second).rotation, 180.0);
    }

    #[test]
    fn changing_a_rotation_drops_the_stale_rough_placement() {
        let repo = repo_with(2);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 2);
        let second = workbench.project().images()[1].id;
        workbench.set_manual_placement(second, Homography::translation(12.0, 4.0));
        workbench.update_edits(
            second,
            EditSettings {
                rotation: 90.0,
                ..EditSettings::default()
            },
        );
        assert_eq!(workbench.manual_placement(second), None);
    }

    #[test]
    fn panorama_inherits_and_exports_the_source_metadata() {
        let repo = repo_with(1);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 1);
        let id = workbench.selected().unwrap();
        let metadata = PhotoMetadata {
            camera: "Nikon FM2".to_owned(),
            film: "Kodak Portra 400".to_owned(),
            ..PhotoMetadata::default()
        };
        workbench.update_image_metadata(id, metadata.clone());
        workbench.finish_stitch(StitchResult {
            image: Arc::new(RgbaImage::filled(8, 8, [10, 20, 30, 255])),
            placements: Vec::new(),
            diagnostics: stitch_domain::StitchDiagnostics::default(),
        });
        workbench.show_stitched().unwrap();
        assert_eq!(workbench.current_metadata(), metadata);

        let service = ProjectService::new(repo.clone());
        workbench
            .export(&service, &ImageLocator::new("panorama.png"))
            .unwrap();
        let saved = repo.saved_metadata.lock().unwrap();
        assert_eq!(saved.last().unwrap().1, Some(metadata));
    }

    #[test]
    fn metadata_can_be_applied_to_every_image() {
        let repo = repo_with(3);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 3);
        let metadata = PhotoMetadata {
            camera: "Nikon FM2".to_owned(),
            ..PhotoMetadata::default()
        };
        workbench.apply_metadata_to_all(metadata.clone());
        for image in workbench.project().images() {
            assert_eq!(workbench.image_metadata(image.id), metadata);
        }
    }

    #[test]
    fn loaded_metadata_reaches_the_editor() {
        let repo = Arc::new(MemoryRepo::default());
        let name = "image-0.png".to_owned();
        let metadata = PhotoMetadata {
            camera: "Nikon FM2".to_owned(),
            iso: "400".to_owned(),
            ..PhotoMetadata::default()
        };
        repo.images.lock().unwrap().insert(
            name.clone(),
            LoadedImage {
                locator: ImageLocator::new(name.clone()),
                name,
                image: RgbaImage::filled(40, 30, [1, 2, 3, 255]),
                thumbnail: RgbaImage::filled(20, 15, [0, 0, 0, 255]),
                metadata: metadata.clone(),
            },
        );

        let mut workbench = Workbench::new();
        let service = ProjectService::new(repo.clone());
        workbench
            .add_images(&service, &[ImageLocator::new("image-0.png")])
            .unwrap();

        let id = workbench.selected().unwrap();
        assert_eq!(workbench.image_metadata(id), metadata);
        assert_eq!(workbench.current_metadata(), metadata);
    }

    #[test]
    fn panorama_does_not_inherit_the_source_rotation() {
        let repo = repo_with(1);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 1);
        let id = workbench.selected().unwrap();
        workbench.update_edits(
            id,
            EditSettings {
                invert: true,
                rotation: 90.0,
                ..EditSettings::default()
            },
        );
        workbench.finish_stitch(StitchResult {
            image: Arc::new(RgbaImage::filled(10, 20, [1, 2, 3, 255])),
            placements: Vec::new(),
            diagnostics: stitch_domain::StitchDiagnostics::default(),
        });
        let panorama = workbench.current_edits();
        assert!(panorama.invert);
        assert_eq!(panorama.rotation, 0.0);
    }
}
