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
    suggest_film_base, CoreError, CoreResult, EditSettings, ImageId, ImageLocator, Project,
    RgbaImage, SourceImage, StitchOptions, StitchProgress, StitchResult, StitchStage, MAX_IMAGES,
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

/// An immutable job handed to a background worker.
#[derive(Clone, Debug)]
pub struct StitchJob {
    pub images: Vec<SourceImage>,
    pub options: StitchOptions,
    pub revision: u64,
}

#[derive(Clone, Debug)]
struct PreviewCache {
    revision: u64,
    target: PreviewTarget,
    image: Arc<RgbaImage>,
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
    show_edits: bool,
    preview_cache: RefCell<Option<PreviewCache>>,
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
            show_edits: true,
            preview_cache: RefCell::new(None),
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

    /// The unedited image currently selected for preview.
    pub fn base_preview(&self) -> Option<PreviewInfo> {
        let (image, label) = self.base_image()?;
        Some(PreviewInfo {
            image,
            label,
            revision: self.revision,
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
        if self.edits.get(&id) != Some(&settings) {
            self.edits.insert(id, settings);
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
    pub fn apply_edits_to_all(&mut self, settings: EditSettings) {
        let settings = settings.sanitized();
        let mut changed = false;
        for image in self.project.images() {
            if self.edits.get(&image.id) != Some(&settings) {
                self.edits.insert(image.id, settings.clone());
                changed = true;
            }
        }
        if changed {
            self.bump_revision();
            self.set_status(WorkbenchStatus::Ready(format!(
                "Applied adjustments to {} images.",
                self.project.len()
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
        let ids = service.add_images(&mut self.project, locators)?;
        for &id in &ids {
            self.edits.insert(id, EditSettings::default());
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
        self.invalidate_result();
        self.set_status(WorkbenchStatus::Ready("Image order updated.".to_owned()));
        Ok(())
    }

    pub fn move_down(&mut self, id: ImageId) -> CoreResult<()> {
        self.project.move_down(id)?;
        self.invalidate_result();
        self.set_status(WorkbenchStatus::Ready("Image order updated.".to_owned()));
        Ok(())
    }

    pub fn clear(&mut self) {
        self.project.clear();
        self.selected = None;
        self.result = None;
        self.edits.clear();
        self.result_edits = EditSettings::default();
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
            let developed = Arc::new(settings.develop_preview(&base, PREVIEW_DIMENSION));
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
        self.result = Some(result);
        self.result_edits = EditSettings::default();
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
        service.export(locator, &image)
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
            service.export(&ImageLocator::from(path.as_path()), &developed)?;
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
            service.export(&ImageLocator::from(path.as_path()), &developed)?;
            written += 1;
        }
        Ok(written)
    }

    /// Runs the use case synchronously. Kept for tests, headless use and simple
    /// integrations; the egui adapter uses [`Self::begin_stitch`] on a worker.
    pub fn stitch_sync(&mut self, service: &StitchService) -> CoreResult<()> {
        let job = self.begin_stitch()?;
        match service.stitch(&job.images, &job.options, &stitch_domain::NoProgress) {
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
    use std::sync::Mutex;
    use stitch_domain::{ImageRepository, LoadedImage};

    #[derive(Default)]
    struct MemoryRepo {
        images: Mutex<HashMap<String, LoadedImage>>,
        saved: Mutex<Vec<(String, RgbaImage)>>,
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
}
