//! No-reference focus measurements used to keep panoramas sharp.
//!
//! Laplacian variance is a standard focus score: an in-focus patch contains
//! strong second derivatives, while a defocused or motion-blurred patch does
//! not. The score is deliberately cheap enough to run on a small preview of a
//! full-resolution source before compositing.

use crate::image::GrayImage;

/// Variance of the discrete Laplacian of a grayscale image.
///
/// Input samples are normalized luminance in `0..=1`. The returned value is in
/// squared intensity units; larger means sharper. A uniform image scores zero.
pub fn laplacian_variance(gray: &GrayImage) -> f64 {
    let width = gray.width() as usize;
    let height = gray.height() as usize;
    if width < 3 || height < 3 {
        return 0.0;
    }
    let data = gray.data();
    let mut sum = 0.0f64;
    let mut sum_squared = 0.0f64;
    let mut count = 0usize;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = data[y * width + x];
            let laplacian = 4.0 * center
                - data[y * width + x - 1]
                - data[y * width + x + 1]
                - data[(y - 1) * width + x]
                - data[(y + 1) * width + x];
            let value = laplacian as f64;
            sum += value;
            sum_squared += value * value;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    let mean = sum / count as f64;
    (sum_squared / count as f64 - mean * mean).max(0.0)
}

/// Focus scores over a regular grid of image tiles.
#[derive(Clone, Debug, PartialEq)]
pub struct FocusMap {
    columns: usize,
    rows: usize,
    values: Vec<f64>,
}

impl FocusMap {
    /// Measures `columns * rows` equally sized interior tiles. Tiles smaller
    /// than three pixels get a zero score.
    pub fn new(gray: &GrayImage, columns: usize, rows: usize) -> Self {
        let columns = columns.max(1);
        let rows = rows.max(1);
        let tile_width = gray.width() as usize / columns;
        let tile_height = gray.height() as usize / rows;
        let values = (0..rows)
            .flat_map(|row| {
                (0..columns).map(move |column| {
                    let tile = GrayImage::from_fn(
                        tile_width.max(1) as u32,
                        tile_height.max(1) as u32,
                        |x, y| {
                            gray.at(
                                (column * tile_width + x as usize).min(gray.width() as usize - 1)
                                    as u32,
                                (row * tile_height + y as usize).min(gray.height() as usize - 1)
                                    as u32,
                            )
                        },
                    );
                    laplacian_variance(&tile)
                })
            })
            .collect();
        Self {
            columns,
            rows,
            values,
        }
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Score of the tile containing the given pixel coordinate.
    #[inline]
    pub fn at(&self, x: f64, y: f64, width: u32, height: u32) -> f64 {
        if width == 0 || height == 0 || !x.is_finite() || !y.is_finite() {
            return 0.0;
        }
        let column = ((x / width as f64) * self.columns as f64)
            .floor()
            .clamp(0.0, (self.columns - 1) as f64) as usize;
        let row = ((y / height as f64) * self.rows as f64)
            .floor()
            .clamp(0.0, (self.rows - 1) as f64) as usize;
        self.values[row * self.columns + column]
    }

    /// Median tile score; robust to small uniform borders.
    pub fn median(&self) -> f64 {
        let mut sorted = self.values.clone();
        if sorted.is_empty() {
            return 0.0;
        }
        sorted.sort_by(|a, b| a.total_cmp(b));
        sorted[sorted.len() / 2]
    }

    pub fn min(&self) -> f64 {
        self.values.iter().copied().fold(f64::INFINITY, f64::min)
    }

    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Fraction of tiles whose score falls below `reference * ratio`.
    pub fn soft_tile_fraction(&self, reference: f64, ratio: f64) -> f64 {
        if self.values.is_empty() || reference <= 0.0 {
            return 0.0;
        }
        let threshold = reference * ratio;
        self.values
            .iter()
            .filter(|&&value| value < threshold)
            .count() as f64
            / self.values.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(size: u32, square: u32) -> GrayImage {
        GrayImage::from_fn(size, size, |x, y| {
            if (x / square + y / square).is_multiple_of(2) {
                1.0
            } else {
                0.0
            }
        })
    }

    fn box_blur(image: &GrayImage, radius: u32) -> GrayImage {
        GrayImage::from_fn(image.width(), image.height(), |x, y| {
            let mut sum = 0.0;
            let mut count = 0;
            for dy in -(radius as i32)..=radius as i32 {
                for dx in -(radius as i32)..=radius as i32 {
                    if let Some(value) =
                        image.at_signed((x as i64) + dx as i64, (y as i64) + dy as i64)
                    {
                        sum += value;
                        count += 1;
                    }
                }
            }
            sum / count as f32
        })
    }

    #[test]
    fn uniform_image_has_no_focus() {
        let flat = GrayImage::from_fn(32, 32, |_, _| 0.5);
        assert_eq!(laplacian_variance(&flat), 0.0);
    }

    #[test]
    fn sharp_detail_scores_higher_than_blurred_detail() {
        let sharp = checkerboard(64, 4);
        let blurred = box_blur(&sharp, 3);
        let sharp_score = laplacian_variance(&sharp);
        let blurred_score = laplacian_variance(&blurred);
        assert!(sharp_score > 0.0);
        assert!(
            blurred_score < sharp_score * 0.25,
            "sharp={sharp_score} blurred={blurred_score}"
        );
    }

    #[test]
    fn focus_map_localizes_a_blurred_half() {
        let sharp = checkerboard(128, 4);
        let blurred = box_blur(&sharp, 4);
        let half = GrayImage::from_fn(128, 128, |x, y| {
            if x < 64 {
                blurred.at(x, y)
            } else {
                sharp.at(x, y)
            }
        });
        let map = FocusMap::new(&half, 4, 2);
        let top_left = map.at(16.0, 32.0, 128, 128);
        let top_right = map.at(96.0, 32.0, 128, 128);
        assert!(
            top_left < top_right * 0.2,
            "left={top_left} right={top_right}"
        );
        assert!(map.median() > 0.0);
        assert!(map.soft_tile_fraction(map.max(), 0.5) >= 0.25);
    }
}
