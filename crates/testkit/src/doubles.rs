//! In-memory and scripted implementations of every port.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use nalgebra::DMatrix;
use stitch_domain::{
    Clock, Compositor, CoreError, CoreResult, Descriptor, Feature, FeatureExtractor, FeatureMatch,
    FeatureMatcher, FeatureSet, GrayImage, Homography, HomographyEstimate, HomographyEstimator,
    ImageLocator, ImageRepository, Keypoint, LoadedImage, PhotoMetadata, Point, Renderer,
    RgbaImage, SourceImage, StitchOptions, WarpLayer, WarpOptions,
};

use crate::fixtures::SplitMix64;

/// Memory-backed image repository.
#[derive(Default)]
pub struct InMemoryImageRepository {
    images: Mutex<HashMap<String, LoadedImage>>,
    saved: Mutex<Vec<String>>,
    metadata: Mutex<HashMap<String, PhotoMetadata>>,
}

impl InMemoryImageRepository {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, name: impl Into<String>, image: RgbaImage) {
        let name = name.into();
        let thumbnail = image.thumbnail(320);
        self.images.lock().unwrap().insert(
            name.clone(),
            LoadedImage {
                locator: ImageLocator::new(name.clone()),
                name,
                image,
                thumbnail,
                metadata: PhotoMetadata::default(),
            },
        );
    }

    pub fn insert_loaded(&self, loaded: LoadedImage) {
        self.images
            .lock()
            .unwrap()
            .insert(loaded.locator.as_str().to_owned(), loaded);
    }

    pub fn contains(&self, name: &str) -> bool {
        self.images.lock().unwrap().contains_key(name)
    }

    pub fn saved(&self) -> Vec<String> {
        self.saved.lock().unwrap().clone()
    }

    pub fn stored(&self, name: &str) -> Option<RgbaImage> {
        self.images
            .lock()
            .unwrap()
            .get(name)
            .map(|loaded| loaded.image.clone())
    }

    pub fn stored_metadata(&self, name: &str) -> Option<PhotoMetadata> {
        self.metadata.lock().unwrap().get(name).cloned()
    }
}

impl ImageRepository for InMemoryImageRepository {
    fn load(&self, locator: &ImageLocator) -> CoreResult<LoadedImage> {
        let mut loaded = self
            .images
            .lock()
            .unwrap()
            .get(locator.as_str())
            .cloned()
            .ok_or_else(|| CoreError::NotFound(locator.as_str().to_owned()))?;
        if let Some(metadata) = self.metadata.lock().unwrap().get(locator.as_str()) {
            loaded.metadata = metadata.clone();
        }
        Ok(loaded)
    }

    fn save(&self, locator: &ImageLocator, image: &RgbaImage) -> CoreResult<()> {
        let name = locator.as_str().to_owned();
        self.saved.lock().unwrap().push(name.clone());
        let thumbnail = image.thumbnail(320);
        self.images.lock().unwrap().insert(
            name.clone(),
            LoadedImage {
                locator: locator.clone(),
                name,
                image: image.clone(),
                thumbnail,
                metadata: PhotoMetadata::default(),
            },
        );
        Ok(())
    }

    fn save_with_metadata(
        &self,
        locator: &ImageLocator,
        image: &RgbaImage,
        metadata: Option<&PhotoMetadata>,
    ) -> CoreResult<()> {
        let name = locator.as_str().to_owned();
        let mut stored = self.metadata.lock().unwrap();
        match metadata {
            Some(metadata) => {
                stored.insert(name, metadata.clone());
            }
            None => {
                stored.remove(&name);
            }
        }
        drop(stored);
        self.save(locator, image)
    }
}

/// A deterministic feature extractor for contract and orchestration tests.
///
/// It is *not* scale/rotation invariant; use the real SIFT adapter when the test
/// is about registration quality. It produces well-formed, normalized features
/// at a regular grid and nothing on flat images.
#[derive(Clone, Debug, Default)]
pub struct InMemoryFeatureExtractor;

impl FeatureExtractor for InMemoryFeatureExtractor {
    fn extract(&self, image: &GrayImage) -> CoreResult<FeatureSet> {
        if image.variance() < 1e-7 {
            return Ok(FeatureSet::default());
        }
        let step = (image.width().min(image.height()) / 10).max(8) as i64;
        let mut features = Vec::new();
        let mut y = step / 2;
        while y < image.height() as i64 {
            let mut x = step / 2;
            while x < image.width() as i64 {
                if let Some(feature) = describe_grid_point(image, x, y) {
                    features.push(feature);
                }
                x += step;
            }
            y += step;
        }
        Ok(FeatureSet::new(features))
    }
}

fn describe_grid_point(image: &GrayImage, x: i64, y: i64) -> Option<Feature> {
    let mut values = [0.0f32; 128];
    let mut sum = 0.0f32;
    for row in 0..8 {
        for column in 0..16 {
            let sx = x as f64 - 7.5 + column as f64;
            let sy = y as f64 - 3.5 + row as f64;
            let value = image.sample(sx, sy)?;
            let index = row * 16 + column;
            values[index] = value;
            sum += value;
        }
    }
    let mean = sum / 128.0;
    let mut energy = 0.0f32;
    for value in &mut values {
        *value -= mean;
        energy += *value * *value;
    }
    if energy < 1e-7 {
        return None;
    }
    let mut descriptor = Descriptor::from_array(values);
    descriptor.normalize();
    Some(Feature {
        keypoint: Keypoint {
            x: x as f32,
            y: y as f32,
            scale: 2.0,
            orientation: 0.0,
            response: energy.sqrt(),
        },
        descriptor,
    })
}

/// Simple sequential ratio-test matcher.
#[derive(Clone, Copy, Debug, Default)]
pub struct InMemoryMatcher;

impl FeatureMatcher for InMemoryMatcher {
    fn match_features(
        &self,
        query: &FeatureSet,
        train: &FeatureSet,
        ratio: f32,
    ) -> Vec<FeatureMatch> {
        let mut matches = Vec::new();
        if train.len() < 2 {
            return matches;
        }
        for (query_index, feature) in query.iter().enumerate() {
            let mut best = f32::INFINITY;
            let mut second = f32::INFINITY;
            let mut best_index = 0;
            for (train_index, candidate) in train.iter().enumerate() {
                let distance = feature.descriptor.squared_distance(&candidate.descriptor);
                if distance < best {
                    second = best;
                    best = distance;
                    best_index = train_index;
                } else if distance < second {
                    second = distance;
                }
            }
            if best.is_finite() && second.is_finite() && second > 0.0 {
                let best = best.sqrt();
                let second = second.sqrt();
                if best < ratio * second {
                    matches.push(FeatureMatch::new(query_index, best_index, best));
                }
            }
        }
        matches
    }
}

/// Exact DLT estimator without RANSAC.
///
/// This is a genuine (if simple) implementation suitable for clean
/// correspondences. Outlier robustness is covered by the production adapter's
/// extra tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct InMemoryHomographyEstimator;

impl HomographyEstimator for InMemoryHomographyEstimator {
    fn estimate(
        &self,
        query: &FeatureSet,
        train: &FeatureSet,
        matches: &[FeatureMatch],
        options: &StitchOptions,
    ) -> CoreResult<HomographyEstimate> {
        if matches.len() < 4 {
            return Err(CoreError::Homography(
                "at least four correspondences are required".to_owned(),
            ));
        }
        let mut source_points = Vec::with_capacity(matches.len());
        let mut target_points = Vec::with_capacity(matches.len());
        let mut valid = Vec::with_capacity(matches.len());
        for (match_index, m) in matches.iter().enumerate() {
            let (Some(a), Some(b)) = (
                query.features.get(m.query_index),
                train.features.get(m.train_index),
            ) else {
                continue;
            };
            source_points.push(Point::new(a.keypoint.x as f64, a.keypoint.y as f64));
            target_points.push(Point::new(b.keypoint.x as f64, b.keypoint.y as f64));
            valid.push(match_index);
        }
        if source_points.len() < 4 {
            return Err(CoreError::Homography(
                "at least four valid correspondences are required".to_owned(),
            ));
        }

        let (normalized_source, source_norm) = normalize(&source_points);
        let (normalized_target, target_norm) = normalize(&target_points);
        let h_normalized = solve(&normalized_source, &normalized_target)?;
        let homography = target_norm
            .inverse()
            .ok_or_else(|| CoreError::Homography("singular normalization".to_owned()))?
            .then(&h_normalized)
            .then(&source_norm);
        if !homography.is_finite() {
            return Err(CoreError::Homography("non-finite transform".to_owned()));
        }

        let threshold = options.ransac_threshold.max(0.5);
        let threshold_squared = threshold * threshold;
        let mut inliers = Vec::new();
        let mut squared_error = 0.0;
        for index in 0..source_points.len() {
            if let Some(projected) = homography.transform(source_points[index]) {
                let error = projected.distance(target_points[index]);
                if error * error <= threshold_squared {
                    inliers.push(valid[index]);
                    squared_error += error * error;
                }
            }
        }
        if inliers.len() < options.min_inliers {
            return Err(CoreError::Homography(format!(
                "only {} inliers",
                inliers.len()
            )));
        }
        let rms_error = (squared_error / inliers.len() as f64).sqrt();
        Ok(HomographyEstimate {
            homography,
            inliers,
            rms_error,
            num_matches: matches.len(),
        })
    }
}

fn solve(source: &[Point], target: &[Point]) -> CoreResult<Homography> {
    // A four-point system is 8x9; pad it so the thin SVD exposes all nine
    // right singular vectors, including the nullspace direction.
    let mut matrix = DMatrix::<f64>::zeros((source.len() * 2).max(9), 9);
    for (index, (a, b)) in source.iter().zip(target).enumerate() {
        let row = index * 2;
        matrix[(row, 0)] = -a.x;
        matrix[(row, 1)] = -a.y;
        matrix[(row, 2)] = -1.0;
        matrix[(row, 6)] = b.x * a.x;
        matrix[(row, 7)] = b.x * a.y;
        matrix[(row, 8)] = b.x;
        matrix[(row + 1, 3)] = -a.x;
        matrix[(row + 1, 4)] = -a.y;
        matrix[(row + 1, 5)] = -1.0;
        matrix[(row + 1, 6)] = b.y * a.x;
        matrix[(row + 1, 7)] = b.y * a.y;
        matrix[(row + 1, 8)] = b.y;
    }
    let svd = matrix.svd(true, true);
    let v_t = svd
        .v_t
        .ok_or_else(|| CoreError::Homography("SVD failed".to_owned()))?;
    if v_t.ncols() < 9 {
        return Err(CoreError::Homography(
            "SVD produced too few columns".to_owned(),
        ));
    }
    let row = v_t.row(v_t.nrows() - 1);
    let mut values = [0.0f64; 9];
    for (index, value) in row.iter().enumerate() {
        values[index] = *value;
    }
    let homography = Homography::from_slice(&values);
    if homography.is_finite() {
        Ok(homography)
    } else {
        Err(CoreError::Homography("non-finite transform".to_owned()))
    }
}

fn normalize(points: &[Point]) -> (Vec<Point>, Homography) {
    let count = points.len().max(1) as f64;
    let cx = points.iter().map(|p| p.x).sum::<f64>() / count;
    let cy = points.iter().map(|p| p.y).sum::<f64>() / count;
    let mean = points
        .iter()
        .map(|p| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt())
        .sum::<f64>()
        / count;
    let scale = if mean > 1e-9 {
        2.0f64.sqrt() / mean
    } else {
        1.0
    };
    let transform = Homography::from_rows([
        [scale, 0.0, -scale * cx],
        [0.0, scale, -scale * cy],
        [0.0, 0.0, 1.0],
    ]);
    let normalized = points
        .iter()
        .map(|&point| transform.transform(point).unwrap_or(point))
        .collect();
    (normalized, transform)
}

/// Sequential compositor with the same semantics as the production adapter but
/// without rayon or feathering.
#[derive(Clone, Copy, Debug, Default)]
pub struct InMemoryCompositor;

impl Compositor for InMemoryCompositor {
    fn composite(&self, layers: &[WarpLayer], options: &WarpOptions) -> CoreResult<RgbaImage> {
        if layers.is_empty() {
            return Err(CoreError::EmptyProject);
        }
        let width = options.canvas_width.max(1);
        let height = options.canvas_height.max(1);
        let mut accum = vec![0.0f32; width as usize * height as usize * 3];
        let mut weights = vec![0.0f32; width as usize * height as usize];

        for layer in layers {
            let inverse = layer
                .homography
                .inverse()
                .ok_or_else(|| CoreError::Compositing(format!("'{}' is singular", layer.name)))?;
            for y in 0..height {
                for x in 0..width {
                    let Some(source) = inverse.transform(Point::new(x as f64, y as f64)) else {
                        continue;
                    };
                    let Some(color) = layer.image.sample_bilinear(source.x, source.y) else {
                        continue;
                    };
                    let alpha = color[3] as f32 / 255.0;
                    if alpha <= 0.0 {
                        continue;
                    }
                    let index = y as usize * width as usize + x as usize;
                    accum[index * 3] += color[0] as f32 * alpha;
                    accum[index * 3 + 1] += color[1] as f32 * alpha;
                    accum[index * 3 + 2] += color[2] as f32 * alpha;
                    weights[index] += alpha;
                }
            }
        }

        let mut pixels = vec![0u8; width as usize * height as usize * 4];
        for index in 0..width as usize * height as usize {
            let weight = weights[index];
            if weight <= 1e-8 {
                continue;
            }
            pixels[index * 4] = (accum[index * 3] / weight).round().clamp(0.0, 255.0) as u8;
            pixels[index * 4 + 1] = (accum[index * 3 + 1] / weight).round().clamp(0.0, 255.0) as u8;
            pixels[index * 4 + 2] = (accum[index * 3 + 2] / weight).round().clamp(0.0, 255.0) as u8;
            pixels[index * 4 + 3] = 255;
        }
        RgbaImage::new(width, height, pixels)
    }
}

/// Deterministic clock with manual advancing.
#[derive(Clone, Debug, Default)]
pub struct FakeClock {
    now: Arc<Mutex<u64>>,
}

impl FakeClock {
    pub fn new(start: u64) -> Self {
        Self {
            now: Arc::new(Mutex::new(start)),
        }
    }

    pub fn advance(&self, milliseconds: u64) {
        *self.now.lock().unwrap() += milliseconds;
    }
}

impl Clock for FakeClock {
    fn now_ms(&self) -> u64 {
        *self.now.lock().unwrap()
    }
}

/// Renderer double that records every call.
type RendererCalls = Arc<Mutex<Vec<Option<(u64, RgbaImage)>>>>;

#[derive(Clone, Debug, Default)]
pub struct RecordingRenderer {
    calls: RendererCalls,
}

impl RecordingRenderer {
    pub fn calls(&self) -> Vec<Option<(u64, RgbaImage)>> {
        self.calls.lock().unwrap().clone()
    }
}

impl Renderer for RecordingRenderer {
    fn set_image(&mut self, image: Arc<RgbaImage>, revision: u64) -> CoreResult<()> {
        self.calls
            .lock()
            .unwrap()
            .push(Some((revision, image.as_ref().clone())));
        Ok(())
    }

    fn clear(&mut self) {
        self.calls.lock().unwrap().push(None);
    }
}

/// Extractor that returns queued feature sets in FIFO order.
#[derive(Clone, Debug, Default)]
pub struct ScriptedExtractor {
    queue: Arc<Mutex<VecDeque<FeatureSet>>>,
    calls: Arc<AtomicUsize>,
}

impl ScriptedExtractor {
    pub fn new(script: Vec<FeatureSet>) -> Self {
        Self {
            queue: Arc::new(Mutex::new(script.into())),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl FeatureExtractor for ScriptedExtractor {
    fn extract(&self, _image: &GrayImage) -> CoreResult<FeatureSet> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.queue
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| CoreError::FeatureExtraction("script exhausted".to_owned()))
    }
}

/// Matcher that always returns a fixed list.
#[derive(Clone, Debug, Default)]
pub struct ScriptedMatcher {
    pub matches: Vec<FeatureMatch>,
}

impl ScriptedMatcher {
    pub fn new(matches: Vec<FeatureMatch>) -> Self {
        Self { matches }
    }
}

impl FeatureMatcher for ScriptedMatcher {
    fn match_features(
        &self,
        _query: &FeatureSet,
        _train: &FeatureSet,
        _ratio: f32,
    ) -> Vec<FeatureMatch> {
        self.matches.clone()
    }
}

/// Estimator that returns queued results.
#[derive(Clone, Debug, Default)]
pub struct ScriptedEstimator {
    queue: Arc<Mutex<VecDeque<CoreResult<HomographyEstimate>>>>,
}

impl ScriptedEstimator {
    pub fn new(script: Vec<CoreResult<HomographyEstimate>>) -> Self {
        Self {
            queue: Arc::new(Mutex::new(script.into())),
        }
    }
}

impl HomographyEstimator for ScriptedEstimator {
    fn estimate(
        &self,
        _query: &FeatureSet,
        _train: &FeatureSet,
        _matches: &[FeatureMatch],
        _options: &StitchOptions,
    ) -> CoreResult<HomographyEstimate> {
        self.queue.lock().unwrap().pop_front().unwrap_or_else(|| {
            Err(CoreError::Homography(
                "estimator script exhausted".to_owned(),
            ))
        })
    }
}

/// Compositor that returns a fixed image.
#[derive(Clone, Debug)]
pub struct ScriptedCompositor {
    pub output: RgbaImage,
    calls: Arc<AtomicUsize>,
}

impl ScriptedCompositor {
    pub fn new(output: RgbaImage) -> Self {
        Self {
            output,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Compositor for ScriptedCompositor {
    fn composite(&self, _layers: &[WarpLayer], _options: &WarpOptions) -> CoreResult<RgbaImage> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.output.clone())
    }
}

/// Generates a feature set at a grid, used by scripted application tests.
pub fn scripted_features(count: usize, seed: u64) -> FeatureSet {
    let mut rng = SplitMix64::new(seed);
    let features = (0..count)
        .map(|index| {
            let mut values = [0.0f32; 128];
            for value in &mut values {
                *value = rng.next_f32() * 2.0 - 1.0;
            }
            let mut descriptor = Descriptor::from_array(values);
            descriptor.normalize();
            Feature {
                keypoint: Keypoint {
                    x: index as f32,
                    y: index as f32,
                    scale: 2.0,
                    orientation: 0.0,
                    response: 1.0,
                },
                descriptor,
            }
        })
        .collect();
    FeatureSet::new(features)
}

/// Convenience for building a source image in tests.
pub fn source_image(id: u64, width: u32, height: u32) -> SourceImage {
    use stitch_domain::ImageId;
    let image = RgbaImage::filled(width, height, [10, 20, 30, 255]);
    SourceImage {
        id: ImageId(id),
        name: format!("image-{id}"),
        locator: ImageLocator::new(format!("image-{id}")),
        thumbnail: Arc::new(image.thumbnail(64)),
        image: Arc::new(image),
    }
}
