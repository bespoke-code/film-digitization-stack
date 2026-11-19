//! Use case: load images into a project and export results.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use stitch_domain::{
    CoreError, CoreResult, ImageId, ImageLocator, ImageRepository, LoadEstimate, LoadedImage,
    PhotoMetadata, Project, RgbaImage, SaveOptions, MAX_IMAGES,
};

/// Memory the service may use for loaded images when no budget is given.
pub const DEFAULT_MEMORY_BUDGET: u64 = 4 << 30;

/// Application service around image persistence.
#[derive(Clone)]
pub struct ProjectService {
    repository: Arc<dyn ImageRepository>,
    memory_budget: u64,
}

impl ProjectService {
    pub fn new(repository: Arc<dyn ImageRepository>) -> Self {
        Self {
            repository,
            memory_budget: DEFAULT_MEMORY_BUDGET,
        }
    }

    /// Limits the memory that the project images and the decodes in flight
    /// may use together. Fewer images decode at once on a smaller budget.
    pub fn with_memory_budget(mut self, bytes: u64) -> Self {
        self.memory_budget = bytes;
        self
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

        let loaded = self.load_all(project, locators)?;

        let mut added = Vec::with_capacity(loaded.len());
        for image in loaded {
            let metadata = image.metadata.clone();
            let id = project.add_loaded(image)?;
            added.push((id, metadata));
        }
        Ok(added)
    }

    /// Decodes the locators on parallel workers and returns them in order.
    ///
    /// Decoding a large scan is single-threaded and takes most of the load
    /// time, so a few workers cut opening a folder by several times.
    fn load_all(
        &self,
        project: &Project,
        locators: &[ImageLocator],
    ) -> CoreResult<Vec<LoadedImage>> {
        let workers = self.load_workers(project, locators);
        if workers <= 1 {
            return locators
                .iter()
                .map(|locator| self.repository.load(locator))
                .collect();
        }

        let next = AtomicUsize::new(0);
        let mut slots: Vec<Option<CoreResult<LoadedImage>>> =
            locators.iter().map(|_| None).collect();
        thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some(locator) = locators.get(index) else {
                                break;
                            };
                            done.push((index, self.repository.load(locator)));
                        }
                        done
                    })
                })
                .collect();
            for handle in handles {
                let done = handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                for (index, result) in done {
                    slots[index] = Some(result);
                }
            }
        });
        slots
            .into_iter()
            .map(|slot| slot.expect("every locator has a result"))
            .collect()
    }

    /// How many decodes fit in the memory budget at once.
    ///
    /// The project images and the new images stay in memory regardless. Each
    /// decode in flight needs its transient buffers on top, so the rest of the
    /// budget divided by the largest transient need gives the worker count.
    /// One worker always runs, even when the images alone exceed the budget.
    fn load_workers(&self, project: &Project, locators: &[ImageLocator]) -> usize {
        let cores = thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        let estimates: Vec<LoadEstimate> = locators
            .iter()
            .filter_map(|locator| self.repository.estimate_load(locator))
            .collect();
        let transient = estimates
            .iter()
            .map(|estimate| estimate.peak.saturating_sub(estimate.retained))
            .max()
            .unwrap_or(0);
        let resident: u64 = project
            .images()
            .iter()
            .map(|image| image_bytes(&image.image) + image_bytes(&image.thumbnail))
            .chain(estimates.iter().map(|estimate| estimate.retained))
            .sum();
        let fitting = match transient {
            0 => cores,
            _ => (self.memory_budget.saturating_sub(resident) / transient)
                .try_into()
                .unwrap_or(usize::MAX),
        };
        fitting.clamp(1, cores).min(locators.len().max(1))
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

    /// Exports an image with metadata and explicit encoder settings.
    pub fn export_with_metadata_and_options(
        &self,
        locator: &ImageLocator,
        image: &RgbaImage,
        metadata: Option<&PhotoMetadata>,
        options: &SaveOptions,
    ) -> CoreResult<()> {
        self.repository
            .save_with_metadata_and_options(locator, image, metadata, options)
    }
}

fn image_bytes(image: &RgbaImage) -> u64 {
    image.pixels().len() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Records how many loads overlap and fails one named locator.
    #[derive(Default)]
    struct SlowRepository {
        active: AtomicUsize,
        peak: AtomicUsize,
        estimate: Option<LoadEstimate>,
    }

    impl ImageRepository for SlowRepository {
        fn load(&self, locator: &ImageLocator) -> CoreResult<LoadedImage> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(active, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(30));
            self.active.fetch_sub(1, Ordering::SeqCst);
            if locator.as_str() == "broken" {
                return Err(CoreError::NotFound("broken".to_owned()));
            }
            let index: u8 = locator.as_str().parse().unwrap();
            Ok(LoadedImage {
                locator: locator.clone(),
                name: locator.as_str().to_owned(),
                image: RgbaImage::filled(2, 2, [index, 0, 0, 255]),
                thumbnail: RgbaImage::filled(1, 1, [index, 0, 0, 255]),
                metadata: PhotoMetadata::default(),
                jpeg_quality: None,
            })
        }

        fn estimate_load(&self, _locator: &ImageLocator) -> Option<LoadEstimate> {
            self.estimate
        }

        fn save(&self, _locator: &ImageLocator, _image: &RgbaImage) -> CoreResult<()> {
            Ok(())
        }
    }

    #[test]
    fn the_memory_budget_limits_parallel_loads() {
        let estimate = LoadEstimate {
            retained: 1_000,
            peak: 3_000,
        };
        let repository = Arc::new(SlowRepository {
            estimate: Some(estimate),
            ..SlowRepository::default()
        });
        // Eight images keep 8,000 bytes; 4,000 more fit two transient decodes.
        let service = ProjectService::new(repository.clone()).with_memory_budget(12_000);
        let locators: Vec<ImageLocator> =
            (0..8).map(|i| ImageLocator::new(i.to_string())).collect();
        let mut project = Project::new();
        assert_eq!(
            service.load_workers(&project, &locators),
            2.min(
                thread::available_parallelism()
                    .map(|c| c.get())
                    .unwrap_or(1)
            )
        );
        service.add_images(&mut project, &locators).unwrap();
        assert!(repository.peak.load(Ordering::SeqCst) <= 2);

        let tight = ProjectService::new(repository).with_memory_budget(1_000);
        assert_eq!(tight.load_workers(&Project::new(), &locators), 1);
    }

    #[test]
    fn images_load_in_parallel_and_keep_their_order() {
        let repository = Arc::new(SlowRepository::default());
        let service = ProjectService::new(repository.clone());
        let locators: Vec<ImageLocator> =
            (0..8).map(|i| ImageLocator::new(i.to_string())).collect();
        let mut project = Project::new();
        service.add_images(&mut project, &locators).unwrap();

        let order: Vec<u8> = project
            .images()
            .iter()
            .map(|image| image.image.pixel(0, 0)[0])
            .collect();
        assert_eq!(order, (0..8).collect::<Vec<u8>>());
        let cores = thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        if cores > 1 {
            assert!(
                repository.peak.load(Ordering::SeqCst) > 1,
                "loads ran one at a time"
            );
        }
        assert!(repository.peak.load(Ordering::SeqCst) <= cores);
    }

    #[test]
    fn one_failed_load_leaves_the_project_unchanged() {
        let service = ProjectService::new(Arc::new(SlowRepository::default()));
        let locators = ["0", "1", "broken", "3"].map(ImageLocator::new);
        let mut project = Project::new();
        assert!(matches!(
            service.add_images(&mut project, &locators),
            Err(CoreError::NotFound(_))
        ));
        assert!(project.is_empty());
    }
}
