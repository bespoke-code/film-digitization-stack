//! CPU compositor with inverse warping, bilinear sampling and focus-aware
//! feather blending.

use std::sync::Arc;

use rayon::prelude::*;
use stitch_domain::{
    Compositor, CoreError, CoreResult, FocusMap, Homography, Point, RgbaImage, WarpLayer,
    WarpOptions,
};

/// Rows accumulated at once. Bounded band memory lets the compositor work at
/// full scan resolution without allocating a canvas-sized float accumulator.
const BAND_ROWS: usize = 128;
const ANALYSIS_LIMIT: u32 = 1024;
const FOCUS_TILE_COLUMNS: usize = 8;
const FOCUS_TILE_ROWS: usize = 8;
const MIN_FOCUS_WEIGHT: f64 = 0.15;

/// Warps all layers into one RGBA canvas.
///
/// Each layer is sampled through the inverse homography. Overlapping pixels
/// are averaged with a smooth edge weight and a local focus weight, so a soft
/// frame cannot smear detail over a sharp one. Rows are processed in bands,
/// keeping memory independent of the canvas area.
#[derive(Clone, Copy, Debug, Default)]
pub struct BilinearCompositor;

impl BilinearCompositor {
    pub fn new() -> Self {
        Self
    }
}

struct PreparedLayer {
    image: Arc<RgbaImage>,
    inverse: Homography,
    lut: [[u8; 256]; 3],
    min_x: i64,
    min_y: i64,
    max_x: i64,
    max_y: i64,
    feather: f64,
    focus: FocusMap,
    focus_weight: f64,
}

impl Compositor for BilinearCompositor {
    fn composite(&self, layers: &[WarpLayer], options: &WarpOptions) -> CoreResult<RgbaImage> {
        if layers.is_empty() {
            return Err(CoreError::EmptyProject);
        }
        let width = options.canvas_width.max(1) as usize;
        let height = options.canvas_height.max(1) as usize;
        if width == 0 || height == 0 {
            return Err(CoreError::Compositing(
                "the output canvas is empty".to_owned(),
            ));
        }

        let mut prepared = prepare_layers(layers, options)?;
        let reference_focus = prepared
            .iter()
            .map(|layer| layer.focus.median())
            .fold(0.0f64, f64::max);
        for layer in &mut prepared {
            if reference_focus > 0.0 && layer.focus.median() > 0.0 {
                layer.focus_weight =
                    (layer.focus.median() / reference_focus).clamp(MIN_FOCUS_WEIGHT, 1.0);
            }
        }

        let mut output = vec![0u8; width * height * 4];
        for band_start in (0..height).step_by(BAND_ROWS) {
            let band_end = (band_start + BAND_ROWS).min(height);
            let band_height = band_end - band_start;
            let mut accum_rgb = vec![0.0f32; band_height * width * 3];
            let mut accum_weight = vec![0.0f32; band_height * width];

            accum_rgb
                .par_chunks_mut(width * 3)
                .zip(accum_weight.par_chunks_mut(width))
                .enumerate()
                .for_each(|(local_y, (rgb_row, weight_row))| {
                    let y = band_start + local_y;
                    for layer in &prepared {
                        if (y as i64) < layer.min_y || (y as i64) > layer.max_y {
                            continue;
                        }
                        for x in layer.min_x..=layer.max_x {
                            let Some((sample_x, sample_y)) =
                                inverse_point(&layer.inverse, x as f64, y as f64)
                            else {
                                continue;
                            };
                            let Some(color) = layer.image.sample_bilinear(sample_x, sample_y)
                            else {
                                continue;
                            };
                            let alpha = color[3] as f32 / 255.0;
                            if alpha <= 0.0 {
                                continue;
                            }
                            let distance = sample_x
                                .min(sample_y)
                                .min(layer.image.width() as f64 - 1.0 - sample_x)
                                .min(layer.image.height() as f64 - 1.0 - sample_y);
                            let local_focus = layer.focus.at(
                                sample_x,
                                sample_y,
                                layer.image.width(),
                                layer.image.height(),
                            );
                            let sharpness = if reference_focus <= 0.0 || local_focus <= 0.0 {
                                layer.focus_weight
                            } else {
                                (local_focus / reference_focus)
                                    .clamp(MIN_FOCUS_WEIGHT, 1.0)
                                    .min(layer.focus_weight)
                            };
                            let weight = (feather_weight(distance, layer.feather) as f64
                                * sharpness
                                * alpha as f64) as f32;
                            if weight <= 0.0 {
                                continue;
                            }
                            let base = x as usize * 3;
                            rgb_row[base] += layer.lut[0][color[0] as usize] as f32 * weight;
                            rgb_row[base + 1] += layer.lut[1][color[1] as usize] as f32 * weight;
                            rgb_row[base + 2] += layer.lut[2][color[2] as usize] as f32 * weight;
                            weight_row[x as usize] += weight;
                        }
                    }
                });

            let band_output = &mut output[band_start * width * 4..band_end * width * 4];
            band_output
                .par_chunks_mut(width * 4)
                .zip(accum_rgb.par_chunks(width * 3))
                .zip(accum_weight.par_chunks(width))
                .for_each(|((out_row, rgb_row), weight_row)| {
                    for (x, weight) in weight_row.iter().copied().enumerate() {
                        if weight <= 1e-8 {
                            continue;
                        }
                        let base_in = x * 3;
                        let base_out = x * 4;
                        out_row[base_out] =
                            (rgb_row[base_in] / weight).round().clamp(0.0, 255.0) as u8;
                        out_row[base_out + 1] =
                            (rgb_row[base_in + 1] / weight).round().clamp(0.0, 255.0) as u8;
                        out_row[base_out + 2] =
                            (rgb_row[base_in + 2] / weight).round().clamp(0.0, 255.0) as u8;
                        out_row[base_out + 3] = 255;
                    }
                });
        }

        RgbaImage::new(width as u32, height as u32, output)
    }
}

fn prepare_layers(layers: &[WarpLayer], options: &WarpOptions) -> CoreResult<Vec<PreparedLayer>> {
    let mut prepared = Vec::with_capacity(layers.len());
    for layer in layers {
        let inverse = layer.homography.inverse().ok_or_else(|| {
            CoreError::Compositing(format!("'{}' has a singular transform", layer.name))
        })?;
        let quad = layer
            .homography
            .transform_quad(layer.image.width(), layer.image.height())
            .ok_or_else(|| CoreError::Compositing(format!("'{}' maps to infinity", layer.name)))?;
        let min_x = quad
            .iter()
            .map(|p| p.x)
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as i64;
        let min_y = quad
            .iter()
            .map(|p| p.y)
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as i64;
        let max_x = quad
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            .min(options.canvas_width as f64 - 1.0) as i64;
        let max_y = quad
            .iter()
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            .min(options.canvas_height as f64 - 1.0) as i64;

        let preview = layer.image.resized_to_fit(ANALYSIS_LIMIT);
        let gray = preview.to_gray();
        let focus = FocusMap::new(&gray, FOCUS_TILE_COLUMNS, FOCUS_TILE_ROWS);
        let feather = (options.feather_fraction.max(0.0) as f64)
            * layer.image.width().min(layer.image.height()) as f64;
        prepared.push(PreparedLayer {
            image: layer.image.clone(),
            inverse,
            lut: *layer.color.lut(),
            min_x,
            min_y,
            max_x,
            max_y,
            feather,
            focus,
            focus_weight: 1.0,
        });
    }
    Ok(prepared)
}

fn inverse_point(inverse: &Homography, x: f64, y: f64) -> Option<(f64, f64)> {
    let m = &inverse.m;
    let w = m[6] * x + m[7] * y + m[8];
    if !w.is_finite() || w.abs() < 1e-12 {
        return None;
    }
    let sx = (m[0] * x + m[1] * y + m[2]) / w;
    let sy = (m[3] * x + m[4] * y + m[5]) / w;
    if sx.is_finite() && sy.is_finite() {
        Some((sx, sy))
    } else {
        None
    }
}

/// A tiny floor keeps the outermost valid pixels from disappearing entirely
/// while retaining a smooth transition toward the image boundary.
fn feather_weight(distance: f64, feather: f64) -> f32 {
    if feather <= 0.5 {
        return 1.0;
    }
    let t = (distance / feather).clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    (0.02 + 0.98 * smooth) as f32
}

/// Maps a canvas point back into a source image. Exposed for tests of the
/// compositor geometry.
pub fn canvas_to_source(homography: &Homography, canvas: Point) -> Option<Point> {
    homography.inverse()?.transform(canvas)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(image: RgbaImage, transform: Homography) -> WarpLayer {
        WarpLayer {
            name: "layer".to_owned(),
            image: Arc::new(image),
            homography: transform,
            color: stitch_domain::ColorCurve::identity(),
        }
    }

    fn options(width: u32, height: u32) -> WarpOptions {
        WarpOptions {
            canvas_width: width,
            canvas_height: height,
            feather_fraction: 0.1,
        }
    }

    #[test]
    fn identity_composite_preserves_pixels() {
        let image = RgbaImage::from_fn(32, 24, |x, y| [x as u8 * 5, y as u8 * 7, 99, 255]);
        let output = BilinearCompositor
            .composite(
                &[layer(image.clone(), Homography::identity())],
                &options(32, 24),
            )
            .unwrap();
        for y in 0..24 {
            for x in 0..32 {
                assert_eq!(output.pixel(x, y), image.pixel(x, y), "at {x},{y}");
            }
        }
    }

    #[test]
    fn overlap_is_blended_and_uncovered_area_is_transparent() {
        let red = RgbaImage::filled(100, 50, [255, 0, 0, 255]);
        let blue = RgbaImage::filled(100, 50, [0, 0, 255, 255]);
        let layers = vec![
            layer(red, Homography::identity()),
            layer(blue, Homography::translation(50.0, 0.0)),
        ];
        let output = BilinearCompositor
            .composite(&layers, &options(200, 50))
            .unwrap();
        assert_eq!(output.width(), 200);
        assert_eq!(output.pixel(10, 25), [255, 0, 0, 255]);
        assert_eq!(output.pixel(75, 25), [128, 0, 128, 255]);
        assert_eq!(output.pixel(120, 25), [0, 0, 255, 255]);
        assert_eq!(output.pixel(180, 25), [0, 0, 0, 0]);
    }

    #[test]
    fn projection_is_used_not_forward_mapping() {
        let image = RgbaImage::from_fn(20, 20, |x, _| [x as u8 * 10, 0, 0, 255]);
        let transform = Homography::from_rows([[1.0, 0.2, 3.0], [0.1, 1.0, 2.0], [0.0, 0.0, 1.0]]);
        let output = BilinearCompositor
            .composite(&[layer(image, transform)], &options(64, 64))
            .unwrap();
        assert!(output.pixel(20, 15)[3] > 0);
    }

    #[test]
    fn singular_transform_is_reported() {
        let image = RgbaImage::filled(10, 10, [0, 0, 0, 255]);
        let singular = Homography::from_rows([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]);
        let error = BilinearCompositor
            .composite(&[layer(image, singular)], &options(20, 20))
            .unwrap_err();
        assert!(matches!(error, CoreError::Compositing(_)));
    }

    #[test]
    fn empty_layer_list_is_rejected() {
        assert!(BilinearCompositor.composite(&[], &options(10, 10)).is_err());
    }

    #[test]
    fn layer_color_curve_is_applied_before_blending() {
        let dim = RgbaImage::filled(4, 4, [100, 50, 25, 255]);
        let mut warp = layer(dim, Homography::identity());
        warp.color = stitch_domain::ColorCurve::from_gain([2.0, 0.5, 1.0]);
        let output = BilinearCompositor
            .composite(&[warp], &options(4, 4))
            .unwrap();
        assert_eq!(output.pixel(0, 0), [200, 25, 25, 255]);
    }

    #[test]
    fn soft_layer_does_not_average_away_sharp_detail() {
        use stitch_domain::laplacian_variance;

        let sharp = RgbaImage::from_fn(128, 128, |x, y| {
            let value = if (x / 4 + y / 4).is_multiple_of(2) {
                255
            } else {
                0
            };
            [value, value, value, 255]
        });
        let sharp_gray = sharp.to_gray();
        let blurred = RgbaImage::from_fn(128, 128, |x, y| {
            let mut sum = 0.0;
            let mut count = 0;
            for dy in -4i64..=4 {
                for dx in -4i64..=4 {
                    if let Some(value) = sharp_gray.at_signed(x as i64 + dx, y as i64 + dy) {
                        sum += value;
                        count += 1;
                    }
                }
            }
            let value = (sum / count as f32 * 255.0).round().clamp(0.0, 255.0) as u8;
            [value, value, value, 255]
        });
        assert!(
            laplacian_variance(&sharp_gray) > laplacian_variance(&blurred.to_gray()) * 4.0,
            "the synthetic soft layer must be measurably softer"
        );

        let layers = vec![
            layer(sharp.clone(), Homography::identity()),
            layer(blurred.clone(), Homography::translation(8.0, 0.0)),
        ];
        let output = BilinearCompositor
            .composite(
                &layers,
                &WarpOptions {
                    canvas_width: 136,
                    canvas_height: 128,
                    feather_fraction: 0.0,
                },
            )
            .unwrap();

        let (x, y) = (64, 64);
        let actual = output.pixel(x, y)[0] as f64;
        let sharp_value = sharp.pixel(x, y)[0] as f64;
        let soft_value = blurred.sample_bilinear(x as f64 - 8.0, y as f64).unwrap()[0] as f64;
        assert!(
            (actual - sharp_value).abs() < (actual - soft_value).abs() * 0.5,
            "output {actual:.1} should follow sharp {sharp_value:.1}, not soft {soft_value:.1}"
        );
    }

    #[test]
    fn bands_cover_the_whole_canvas() {
        let image = RgbaImage::from_fn(20, 300, |x, y| [x as u8, y as u8, 40, 255]);
        let output = BilinearCompositor
            .composite(
                &[layer(image.clone(), Homography::identity())],
                &options(20, 300),
            )
            .unwrap();
        assert_eq!(output.height(), 300);
        for y in (0..300).step_by(37) {
            assert_eq!(output.pixel(3, y), image.pixel(3, y), "row {y}");
        }
    }
}
