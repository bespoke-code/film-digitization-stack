//! UI-agnostic presentation state.
//!
//! A UI adapter (egui today, Slint/Qt tomorrow) renders this state and calls the
//! methods below. No egui, windowing or GPU type appears in this crate, which is
//! what makes swapping the UI framework feasible.

use std::sync::Arc;

use stitch_application::{ProjectService, StitchService};
use stitch_domain::{
    CoreError, CoreResult, ImageId, ImageLocator, Project, RgbaImage, SourceImage, StitchOptions,
    StitchProgress, StitchResult, StitchStage, MAX_IMAGES,
};

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
        self.result.is_some() && !self.is_busy()
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
        match self.preview_target {
            PreviewTarget::Stitched => self.result.as_ref().map(|result| PreviewInfo {
                image: result.image.clone(),
                label: "Stitched panorama".to_owned(),
                revision: self.revision,
            }),
            PreviewTarget::Image(id) => self.project.get(id).map(|image| PreviewInfo {
                image: image.image.clone(),
                label: image.name.clone(),
                revision: self.revision,
            }),
        }
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
        let result = self.result.as_ref().ok_or(CoreError::NoStitchResult)?;
        service.export(locator, &result.image)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use stitch_domain::{ImageRepository, LoadedImage};

    #[derive(Default)]
    struct MemoryRepo {
        images: Mutex<HashMap<String, LoadedImage>>,
        saved: Mutex<Vec<String>>,
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

        fn save(&self, locator: &ImageLocator, _image: &RgbaImage) -> CoreResult<()> {
            self.saved.lock().unwrap().push(locator.as_str().to_owned());
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
    fn export_without_result_fails() {
        let repo = repo_with(1);
        let mut workbench = Workbench::new();
        add(&mut workbench, &repo, 1);
        let service = ProjectService::new(repo);
        assert_eq!(
            workbench
                .export(&service, &ImageLocator::new("out.png"))
                .unwrap_err(),
            CoreError::NoStitchResult
        );
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
