//! Project aggregate: an ordered collection of at most sixteen source images.

use std::path::Path;
use std::sync::Arc;

use crate::error::{CoreError, CoreResult};
use crate::image::{GrayImage, RgbaImage};
use crate::ports::LoadedImage;

pub const MAX_IMAGES: usize = 16;

/// Stable identifier for an image inside a project.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImageId(pub u64);

impl std::fmt::Display for ImageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// A UI- and platform-neutral reference to an image.
///
/// Native builds use filesystem paths, the web build uses URLs and tests use
/// arbitrary keys. The domain only treats it as an opaque string.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageLocator(String);

impl ImageLocator {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// A short name suitable for display in a list.
    pub fn display_name(&self) -> String {
        Path::new(&self.0)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| self.0.clone())
    }
}

impl From<&str> for ImageLocator {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for ImageLocator {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&std::path::Path> for ImageLocator {
    fn from(value: &std::path::Path) -> Self {
        Self::new(value.to_string_lossy())
    }
}

/// An image loaded into a project.
#[derive(Clone, Debug)]
pub struct SourceImage {
    pub id: ImageId,
    pub name: String,
    pub locator: ImageLocator,
    pub image: Arc<RgbaImage>,
    pub thumbnail: Arc<RgbaImage>,
}

impl SourceImage {
    pub fn to_gray(&self) -> GrayImage {
        self.image.to_gray()
    }
}

/// Ordered project state.
#[derive(Clone, Debug)]
pub struct Project {
    images: Vec<SourceImage>,
    next_id: u64,
}

impl Default for Project {
    fn default() -> Self {
        Self::new()
    }
}

impl Project {
    pub fn new() -> Self {
        Self {
            images: Vec::new(),
            next_id: 1,
        }
    }

    pub fn images(&self) -> &[SourceImage] {
        &self.images
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.images.len() >= MAX_IMAGES
    }

    pub fn remaining_slots(&self) -> usize {
        MAX_IMAGES.saturating_sub(self.images.len())
    }

    pub fn get(&self, id: ImageId) -> Option<&SourceImage> {
        self.images.iter().find(|image| image.id == id)
    }

    pub fn position(&self, id: ImageId) -> Option<usize> {
        self.images.iter().position(|image| image.id == id)
    }

    /// Adds a loaded image and assigns it a stable id.
    pub fn add_loaded(&mut self, loaded: LoadedImage) -> CoreResult<ImageId> {
        if self.is_full() {
            return Err(CoreError::TooManyImages { max: MAX_IMAGES });
        }
        let id = ImageId(self.next_id);
        self.next_id += 1;
        self.images.push(SourceImage {
            id,
            name: loaded.name,
            locator: loaded.locator,
            image: Arc::new(loaded.image),
            thumbnail: Arc::new(loaded.thumbnail),
        });
        Ok(id)
    }

    pub fn remove(&mut self, id: ImageId) -> CoreResult<SourceImage> {
        let index = self.position(id).ok_or(CoreError::UnknownImage(id.0))?;
        Ok(self.images.remove(index))
    }

    pub fn clear(&mut self) {
        self.images.clear();
    }

    pub fn move_up(&mut self, id: ImageId) -> CoreResult<()> {
        let index = self.position(id).ok_or(CoreError::UnknownImage(id.0))?;
        if index > 0 {
            self.images.swap(index - 1, index);
        }
        Ok(())
    }

    pub fn move_down(&mut self, id: ImageId) -> CoreResult<()> {
        let index = self.position(id).ok_or(CoreError::UnknownImage(id.0))?;
        if index + 1 < self.images.len() {
            self.images.swap(index, index + 1);
        }
        Ok(())
    }

    pub fn move_to(&mut self, id: ImageId, target: usize) -> CoreResult<()> {
        let index = self.position(id).ok_or(CoreError::UnknownImage(id.0))?;
        let target = target.min(self.images.len().saturating_sub(1));
        if index != target {
            let image = self.images.remove(index);
            self.images.insert(target, image);
        }
        Ok(())
    }

    /// Replaces a source image's pixels (used by tests and future crop tools).
    pub fn replace_image(&mut self, id: ImageId, image: RgbaImage) -> CoreResult<()> {
        let source = self
            .images
            .iter_mut()
            .find(|image| image.id == id)
            .ok_or(CoreError::UnknownImage(id.0))?;
        source.thumbnail = Arc::new(image.thumbnail(320));
        source.image = Arc::new(image);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::PhotoMetadata;

    fn loaded(name: &str) -> LoadedImage {
        LoadedImage {
            locator: ImageLocator::new(name),
            name: name.to_owned(),
            thumbnail: RgbaImage::filled(4, 4, [0, 0, 0, 255]),
            image: RgbaImage::filled(8, 8, [0, 0, 0, 255]),
            metadata: PhotoMetadata::default(),
        }
    }

    #[test]
    fn add_move_and_remove() {
        let mut project = Project::new();
        let a = project.add_loaded(loaded("a")).unwrap();
        let b = project.add_loaded(loaded("b")).unwrap();
        assert_eq!(project.images()[0].id, a);
        project.move_down(a).unwrap();
        assert_eq!(project.images()[0].id, b);
        project.move_up(a).unwrap();
        assert_eq!(project.images()[0].id, a);
        project.remove(a).unwrap();
        assert_eq!(project.len(), 1);
    }

    #[test]
    fn enforces_sixteen_image_limit() {
        let mut project = Project::new();
        for i in 0..MAX_IMAGES {
            project.add_loaded(loaded(&format!("{i}"))).unwrap();
        }
        assert!(project.is_full());
        assert_eq!(
            project.add_loaded(loaded("overflow")),
            Err(CoreError::TooManyImages { max: MAX_IMAGES })
        );
    }

    #[test]
    fn unknown_image_is_reported() {
        let mut project = Project::new();
        assert!(matches!(
            project.remove(ImageId(99)),
            Err(CoreError::UnknownImage(99))
        ));
    }
}
