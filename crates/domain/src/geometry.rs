//! 2D projective geometry.

/// A point in image coordinates. Coordinates are in pixels, with the origin at
/// the top-left corner and `y` growing downwards.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

/// A 3x3 projective transform stored in row-major order.
///
/// `H * [x, y, 1]^T` maps a point from the source image into the destination
/// image. Homographies are only defined up to a non-zero scale factor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Homography {
    pub m: [f64; 9],
}

impl Default for Homography {
    fn default() -> Self {
        Self::identity()
    }
}

impl Homography {
    pub const fn identity() -> Self {
        Self {
            m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    pub const fn translation(tx: f64, ty: f64) -> Self {
        Self {
            m: [1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0],
        }
    }

    pub fn scale(sx: f64, sy: f64) -> Self {
        Self {
            m: [sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0],
        }
    }

    pub fn from_rows(rows: [[f64; 3]; 3]) -> Self {
        Self {
            m: [
                rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
                rows[2][1], rows[2][2],
            ],
        }
    }

    pub fn from_slice(values: &[f64; 9]) -> Self {
        Self { m: *values }
    }

    pub fn determinant(&self) -> f64 {
        let m = &self.m;
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
    }

    /// Returns the inverse transform, or `None` for a singular matrix.
    pub fn inverse(&self) -> Option<Self> {
        let m = &self.m;
        let c0 = m[4] * m[8] - m[5] * m[7];
        let c1 = m[5] * m[6] - m[3] * m[8];
        let c2 = m[3] * m[7] - m[4] * m[6];
        let det = m[0] * c0 + m[1] * c1 + m[2] * c2;
        if !det.is_finite() || det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self {
            m: [
                c0 * inv_det,
                (m[2] * m[7] - m[1] * m[8]) * inv_det,
                (m[1] * m[5] - m[2] * m[4]) * inv_det,
                c1 * inv_det,
                (m[0] * m[8] - m[2] * m[6]) * inv_det,
                (m[2] * m[3] - m[0] * m[5]) * inv_det,
                c2 * inv_det,
                (m[1] * m[6] - m[0] * m[7]) * inv_det,
                (m[0] * m[4] - m[1] * m[3]) * inv_det,
            ],
        })
    }

    /// Matrix product `self * rhs`. Applying the result is equivalent to
    /// applying `rhs` first and then `self`.
    pub fn then(&self, rhs: &Self) -> Self {
        let mut out = [0.0; 9];
        for row in 0..3 {
            for col in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += self.m[row * 3 + k] * rhs.m[k * 3 + col];
                }
                out[row * 3 + col] = sum;
            }
        }
        Self { m: out }
    }

    /// Applies the transform. Returns `None` if the point maps to or near the
    /// line at infinity.
    pub fn transform(&self, point: Point) -> Option<Point> {
        let m = &self.m;
        let w = m[6] * point.x + m[7] * point.y + m[8];
        if !w.is_finite() || w.abs() < 1e-12 {
            return None;
        }
        let x = (m[0] * point.x + m[1] * point.y + m[2]) / w;
        let y = (m[3] * point.x + m[4] * point.y + m[5]) / w;
        if x.is_finite() && y.is_finite() {
            Some(Point::new(x, y))
        } else {
            None
        }
    }

    /// Maps the four image corners (clockwise from top-left).
    pub fn transform_quad(&self, width: u32, height: u32) -> Option<[Point; 4]> {
        let w = width.saturating_sub(1) as f64;
        let h = height.saturating_sub(1) as f64;
        Some([
            self.transform(Point::new(0.0, 0.0))?,
            self.transform(Point::new(w, 0.0))?,
            self.transform(Point::new(w, h))?,
            self.transform(Point::new(0.0, h))?,
        ])
    }

    /// Rescales the matrix so that the bottom-right element is 1 when possible.
    /// Useful for diagnostics and stable comparisons; does not change the map.
    pub fn normalized(&self) -> Self {
        let s = self.m[8];
        if s.is_finite() && s.abs() > 1e-12 {
            let mut m = self.m;
            for value in &mut m {
                *value /= s;
            }
            Self { m }
        } else {
            *self
        }
    }

    pub fn is_finite(&self) -> bool {
        self.m.iter().all(|v| v.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: Point, b: Point, eps: f64) {
        assert!((a.x - b.x).abs() < eps, "{a:?} != {b:?}");
        assert!((a.y - b.y).abs() < eps, "{a:?} != {b:?}");
    }

    #[test]
    fn inverse_round_trip() {
        let h =
            Homography::from_rows([[1.2, 0.1, 10.0], [-0.05, 0.9, -4.0], [0.0002, 0.0001, 1.0]]);
        let inv = h.inverse().unwrap();
        let p = Point::new(123.0, 456.0);
        assert_close(inv.transform(h.transform(p).unwrap()).unwrap(), p, 1e-9);
    }

    #[test]
    fn composition_order_is_rhs_first() {
        let translate = Homography::translation(10.0, 20.0);
        let scale = Homography::scale(2.0, 2.0);
        let combined = translate.then(&scale);
        assert_close(
            combined.transform(Point::new(3.0, 4.0)).unwrap(),
            Point::new(16.0, 28.0),
            1e-9,
        );
    }

    #[test]
    fn quad_is_clockwise() {
        let quad = Homography::translation(5.0, 7.0)
            .transform_quad(100, 50)
            .unwrap();
        assert_eq!(quad[0], Point::new(5.0, 7.0));
        assert_eq!(quad[2], Point::new(104.0, 56.0));
    }

    #[test]
    fn singular_matrix_has_no_inverse() {
        let h = Homography::from_rows([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]);
        assert!(h.inverse().is_none());
    }
}
