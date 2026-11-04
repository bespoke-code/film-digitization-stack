//! Use case: load images into a project and export results.

use std::sync::Arc;

use stitch_domain::{
    CoreError, CoreResult, ImageId, ImageLocator, ImageRepository, LoadedImage, PhotoMetadata,
    Project, RgbaImage, MAX_IMAGES,
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
        let added = self.add_images_with_metadata(project, locators)?;
        Ok(added.into_iter().map(|(id, _)| id).collect())
    }

    /// Loads images and returns the metadata read from each file.
    pub fn add_images_with_metadata(
        &self,
        project: &mut Project,
        locators: &[ImageLocator],
    ) -> CoreResult<Vec<(ImageId, PhotoMetadata)>> {
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

        let mut added = Vec::with_capacity(loaded.len());
        for image in loaded {
            let metadata = image.metadata.clone();
            let id = project.add_loaded(image)?;
            added.push((id, metadata));
        }
        Ok(added)
    }

    pub fn export(&self, locator: &ImageLocator, image: &RgbaImage) -> CoreResult<()> {
        self.repository.save(locator, image)
    }

    /// Exports an image and embeds the given descriptive metadata when possible.
    pub fn export_with_metadata(
        &self,
        locator: &ImageLocator,
        image: &RgbaImage,
        metadata: Option<&PhotoMetadata>,
    ) -> CoreResult<()> {
        self.repository.save_with_metadata(locator, image, metadata)
    }
}
