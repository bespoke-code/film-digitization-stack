//! Canvas planning: turns source->reference homographies into a bounded,
//! memory-safe canvas and a set of source->canvas homographies.

use stitch_domain::{
    CoreError, CoreResult, Homography, Point, SourceImage, StitchOptions, WarpLayer,
};

/// Planned canvas geometry.
#[derive(Clone, Debug)]
pub struct CanvasPlan {
    pub width: u32,
    pub height: u32,
    /// Uniform downscale applied to the source-resolution plan (1.0 = none).
    pub scale: f64,
    /// Layers with transforms that map directly into canvas coordinates.
    pub layers: Vec<WarpLayer>,
}

/// Computes the union of all transformed image quads, translates it to the
/// origin and downscales it if it exceeds the configured limits.
pub fn plan_canvas(
    images: &[SourceImage],
    transforms: &[Homography],
    options: &StitchOptions,
) -> CoreResult<CanvasPlan> {
    if images.is_empty() {
        return Err(CoreError::EmptyProject);
    }
    if images.len() != transforms.len() {
        return Err(CoreError::Homography(
            "every image needs exactly one transform".to_owned(),
        ));
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for (image, transform) in images.iter().zip(transforms) {
        let quad = transform
            .transform_quad(image.image.width(), image.image.height())
            .ok_or_else(|| CoreError::Homography(format!("'{}' maps to infinity", image.name)))?;
        for point in quad {
            if !point.is_finite() {
                return Err(CoreError::Homography(format!(
                    "'{}' produced a non-finite corner",
                    image.name
                )));
            }
            min_x = min_x.min(point.x);
            min_y = min_y.min(point.y);
            max_x = max_x.max(point.x);
            max_y = max_y.max(point.y);
        }
    }

    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        return Err(CoreError::Homography(
            "the panorama has no finite bounds".to_owned(),
        ));
    }

    let span_x = (max_x - min_x).max(1.0);
    let span_y = (max_y - min_y).max(1.0);
    let max_dimension = options.max_canvas_dimension as f64;
    let max_pixels = options.max_canvas_pixels as f64;

    let scale = (max_dimension / span_x)
        .min(max_dimension / span_y)
        .min((max_pixels / (span_x * span_y)).sqrt())
        .min(1.0);

    if !scale.is_finite() || scale <= 0.0 {
        return Err(CoreError::Compositing(
            "the computed canvas scale is invalid".to_owned(),
        ));
    }

    let mut width = ((span_x + 1.0) * scale).round().max(1.0) as u32;
    let mut height = ((span_y + 1.0) * scale).round().max(1.0) as u32;
    width = width.min(options.max_canvas_dimension).max(1);
    height = height.min(options.max_canvas_dimension).max(1);

    // Rounding can push the total slightly over the pixel budget; shrink the
    // longer side until it fits.
    while width as u64 * height as u64 > options.max_canvas_pixels {
        if width >= height {
            width = (width - 1).max(1);
        } else {
            height = (height - 1).max(1);
        }
    }

    let translation = Homography::translation(-min_x, -min_y);
    let scaling = Homography::scale(scale, scale);
    let normalize = scaling.then(&translation);

    let layers = images
        .iter()
        .zip(transforms)
        .map(|(image, transform)| WarpLayer {
            name: image.name.clone(),
            image: image.image.clone(),
            homography: normalize.then(transform),
            gain: [1.0, 1.0, 1.0],
        })
        .collect();

    Ok(CanvasPlan {
        width,
        height,
        scale,
        layers,
    })
}

/// Convenience helper for tests and diagnostics: projects a point through a
/// planned transform and returns `None` for points at infinity.
pub fn project_point(transform: &Homography, point: Point) -> Option<Point> {
    transform.transform(point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use stitch_domain::{ImageId, ImageLocator, RgbaImage};

    fn source(id: u64, width: u32, height: u32) -> SourceImage {
        SourceImage {
            id: ImageId(id),
            name: format!("image-{id}"),
            locator: ImageLocator::new(format!("image-{id}")),
            image: Arc::new(RgbaImage::filled(width, height, [0, 0, 0, 255])),
            thumbnail: Arc::new(RgbaImage::filled(8, 8, [0, 0, 0, 255])),
        }
    }

    #[test]
    fn identity_single_image_keeps_size() {
        let images = vec![source(1, 100, 50)];
        let plan = plan_canvas(
            &images,
            &[Homography::identity()],
            &StitchOptions::default(),
        )
        .unwrap();
        assert_eq!((plan.width, plan.height), (100, 50));
        assert_eq!(plan.scale, 1.0);
    }

    #[test]
    fn translation_expands_canvas() {
        let images = vec![source(1, 100, 50), source(2, 100, 50)];
        let transforms = [Homography::identity(), Homography::translation(100.0, 0.0)];
        let plan = plan_canvas(&images, &transforms, &StitchOptions::default()).unwrap();
        assert_eq!(plan.width, 200);
        assert_eq!(plan.height, 50);
        let p = plan.layers[1]
            .homography
            .transform(Point::new(0.0, 0.0))
            .unwrap();
        assert!((p.x - 100.0).abs() < 1e-9);
    }

    #[test]
    fn oversized_canvas_is_scaled_down() {
        let images = vec![source(1, 4000, 4000)];
        let options = StitchOptions {
            max_canvas_dimension: 1000,
            max_canvas_pixels: 500_000,
            ..StitchOptions::default()
        };
        let plan = plan_canvas(&images, &[Homography::identity()], &options).unwrap();
        assert!(plan.width <= 1000 && plan.height <= 1000);
        assert!(plan.width as u64 * plan.height as u64 <= 500_000);
        assert!(plan.scale < 1.0);
    }

    #[test]
    fn mismatched_transform_count_fails() {
        let images = vec![source(1, 10, 10), source(2, 10, 10)];
        assert!(plan_canvas(
            &images,
            &[Homography::identity()],
            &StitchOptions::default()
        )
        .is_err());
    }
}
