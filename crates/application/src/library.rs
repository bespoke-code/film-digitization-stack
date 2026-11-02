//! Use case: load images into a project and export results.

use std::sync::Arc;

use stitch_domain::{
    CoreError, CoreResult, ImageId, ImageLocator, ImageRepository, LoadedImage, Project, RgbaImage,
    MAX_IMAGES,
};

/// Application service around image persistence.
#[derive(Clone)]
pub struct ProjectService {
    repository: Arc<dyn ImageRepository>,
}

impl ProjectService {
    pub fn new(repository: Arc<dyn ImageRepository>) -> Self {
        Self { repository }
    }

    pub fn repository(&self) -> &Arc<dyn ImageRepository> {
        &self.repository
    }

    /// Loads all locators and appends them to the project.
    ///
    /// Loading is transactional per call: if any image fails to load, the
    /// project is left untouched. Capacity is checked before decoding.
    pub fn add_images(
        &self,
        project: &mut Project,
        locators: &[ImageLocator],
    ) -> CoreResult<Vec<ImageId>> {
        if locators.is_empty() {
            return Ok(Vec::new());
        }
        if locators.len() > project.remaining_slots() {
            return Err(CoreError::TooManyImages { max: MAX_IMAGES });
        }

        let loaded: Vec<LoadedImage> = locators
            .iter()
            .map(|locator| self.repository.load(locator))
            .collect::<CoreResult<_>>()?;

        let mut ids = Vec::with_capacity(loaded.len());
        for image in loaded {
            ids.push(project.add_loaded(image)?);
        }
        Ok(ids)
    }

    pub fn export(&self, locator: &ImageLocator, image: &RgbaImage) -> CoreResult<()> {
        self.repository.save(locator, image)
    }
}
