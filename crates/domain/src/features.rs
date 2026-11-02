//! Scale-space features used for image registration.

use crate::error::{CoreError, CoreResult};

pub const DESCRIPTOR_LEN: usize = 128;

/// A localized interest point in source image coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keypoint {
    /// Horizontal position in pixels.
    pub x: f32,
    /// Vertical position in pixels.
    pub y: f32,
    /// Detection scale (sigma in pixels in the image where it was found).
    pub scale: f32,
    /// Dominant gradient orientation in radians, in `-PI..=PI`.
    pub orientation: f32,
    /// Detector response. Larger is stronger.
    pub response: f32,
}

/// A 128-dimensional SIFT descriptor. Vectors are L2-normalized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Descriptor {
    pub values: [f32; DESCRIPTOR_LEN],
}

impl Descriptor {
    pub fn from_array(values: [f32; DESCRIPTOR_LEN]) -> Self {
        Self { values }
    }

    pub fn from_slice(values: &[f32]) -> CoreResult<Self> {
        if values.len() != DESCRIPTOR_LEN {
            return Err(CoreError::FeatureExtraction(format!(
                "a descriptor must contain {DESCRIPTOR_LEN} values, got {}",
                values.len()
            )));
        }
        let mut array = [0.0; DESCRIPTOR_LEN];
        array.copy_from_slice(values);
        Ok(Self { values: array })
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// L2 normalization is the SIFT contract. Returns the norm before
    /// normalization so callers can reject degenerate descriptors.
    pub fn normalize(&mut self) -> f32 {
        let norm = self.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for value in &mut self.values {
                *value /= norm;
            }
        }
        norm
    }

    pub fn squared_distance(&self, other: &Self) -> f32 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum()
    }

    pub fn is_normalized(&self, tolerance: f32) -> bool {
        let norm = self.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        (norm - 1.0).abs() <= tolerance || norm <= f32::EPSILON
    }
}

/// A keypoint together with its descriptor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Feature {
    pub keypoint: Keypoint,
    pub descriptor: Descriptor,
}

/// The features detected in one image.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeatureSet {
    pub features: Vec<Feature>,
}

impl FeatureSet {
    pub fn new(features: Vec<Feature>) -> Self {
        Self { features }
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Feature> {
        self.features.iter()
    }

    pub fn descriptor_slice(&self) -> Vec<&Descriptor> {
        self.features.iter().map(|f| &f.descriptor).collect()
    }
}

impl From<Vec<Feature>> for FeatureSet {
    fn from(features: Vec<Feature>) -> Self {
        Self { features }
    }
}

/// A tentative correspondence produced by a feature matcher.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureMatch {
    /// Index into the query feature set.
    pub query_index: usize,
    /// Index into the train feature set.
    pub train_index: usize,
    /// Descriptor distance (L2).
    pub distance: f32,
}

impl FeatureMatch {
    pub const fn new(query_index: usize, train_index: usize, distance: f32) -> Self {
        Self {
            query_index,
            train_index,
            distance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_is_unit_length() {
        let mut descriptor = Descriptor::from_slice(&[3.0; DESCRIPTOR_LEN]).unwrap();
        descriptor.normalize();
        assert!(descriptor.is_normalized(1e-5));
    }

    #[test]
    fn distance_of_identical_descriptors_is_zero() {
        let d = Descriptor::from_slice(&[0.5; DESCRIPTOR_LEN]).unwrap();
        assert_eq!(d.squared_distance(&d), 0.0);
    }

    #[test]
    fn rejects_wrong_descriptor_size() {
        assert!(Descriptor::from_slice(&[0.0; 12]).is_err());
    }
}
