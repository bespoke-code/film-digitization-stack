//! Descriptive metadata for a photo or panorama.
//!
//! The fields are free text because photographers write ISO, shutter speed and
//! focal length in many ways. Adapters that need structured EXIF values parse
//! the text and keep the original wording in the description.

const MAX_FIELD_LEN: usize = 512;

/// Camera, lens, film and exposure details attached to an image.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhotoMetadata {
    pub camera: String,
    pub lens: String,
    pub film: String,
    pub iso: String,
    pub aperture: String,
    pub shutter_speed: String,
    pub focal_length: String,
    pub date: String,
    pub location: String,
    pub notes: String,
}

impl PhotoMetadata {
    /// Trims every field and limits its length.
    pub fn sanitized(self) -> Self {
        let mut metadata = self.bounded();
        metadata.camera = metadata.camera.trim().to_owned();
        metadata.lens = metadata.lens.trim().to_owned();
        metadata.film = metadata.film.trim().to_owned();
        metadata.iso = metadata.iso.trim().to_owned();
        metadata.aperture = metadata.aperture.trim().to_owned();
        metadata.shutter_speed = metadata.shutter_speed.trim().to_owned();
        metadata.focal_length = metadata.focal_length.trim().to_owned();
        metadata.date = metadata.date.trim().to_owned();
        metadata.location = metadata.location.trim().to_owned();
        metadata.notes = metadata.notes.trim().to_owned();
        metadata
    }

    /// Limits the length without trimming so text fields stay editable.
    pub fn bounded(mut self) -> Self {
        self.camera = bound(&self.camera);
        self.lens = bound(&self.lens);
        self.film = bound(&self.film);
        self.iso = bound(&self.iso);
        self.aperture = bound(&self.aperture);
        self.shutter_speed = bound(&self.shutter_speed);
        self.focal_length = bound(&self.focal_length);
        self.date = bound(&self.date);
        self.location = bound(&self.location);
        self.notes = bound(&self.notes);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.camera.trim().is_empty()
            && self.lens.trim().is_empty()
            && self.film.trim().is_empty()
            && self.iso.trim().is_empty()
            && self.aperture.trim().is_empty()
            && self.shutter_speed.trim().is_empty()
            && self.focal_length.trim().is_empty()
            && self.date.trim().is_empty()
            && self.location.trim().is_empty()
            && self.notes.trim().is_empty()
    }

    /// A single human-readable line used in EXIF descriptions and tooltips.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        push_labeled(&mut parts, "Camera", &self.camera);
        push_labeled(&mut parts, "Lens", &self.lens);
        push_labeled(&mut parts, "Film", &self.film);
        push_labeled(&mut parts, "ISO", &self.iso);
        push_aperture(&mut parts, &self.aperture);
        push_exposure(&mut parts, &self.shutter_speed);
        push_value(&mut parts, &self.focal_length, "mm");
        push_value(&mut parts, &self.date, "");
        push_value(&mut parts, &self.location, "");
        push_value(&mut parts, &self.notes, "");
        parts.join(" · ")
    }
}

fn bound(value: &str) -> String {
    value.chars().take(MAX_FIELD_LEN).collect()
}

fn push_labeled(parts: &mut Vec<String>, label: &str, value: &str) {
    let value = value.trim();
    if !value.is_empty() {
        parts.push(format!("{label}: {value}"));
    }
}

fn push_value(parts: &mut Vec<String>, value: &str, unit: &str) {
    let value = value.trim();
    if value.is_empty() {
        return;
    }
    if unit.is_empty() || value.ends_with(unit) {
        parts.push(value.to_owned());
    } else {
        parts.push(format!("{value} {unit}"));
    }
}

fn push_aperture(parts: &mut Vec<String>, value: &str) {
    let value = value.trim();
    if value.is_empty() {
        return;
    }
    if value.starts_with("f/") {
        parts.push(value.to_owned());
    } else {
        parts.push(format!("f/{value}"));
    }
}

fn push_exposure(parts: &mut Vec<String>, value: &str) {
    let value = value.trim();
    if value.is_empty() {
        return;
    }
    if value.ends_with('s') {
        parts.push(value.to_owned());
    } else {
        parts.push(format!("{value} s"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_and_limits_fields() {
        let long = "x".repeat(MAX_FIELD_LEN + 40);
        let metadata = PhotoMetadata {
            camera: "  Nikon FM2  ".to_owned(),
            notes: long.clone(),
            ..PhotoMetadata::default()
        }
        .sanitized();
        assert_eq!(metadata.camera, "Nikon FM2");
        assert_eq!(metadata.notes.chars().count(), MAX_FIELD_LEN);
    }

    #[test]
    fn empty_ignores_whitespace() {
        let metadata = PhotoMetadata {
            camera: "   ".to_owned(),
            ..PhotoMetadata::default()
        };
        assert!(metadata.is_empty());
    }

    #[test]
    fn bounded_keeps_trailing_spaces_for_editing() {
        let metadata = PhotoMetadata {
            camera: "Nikon ".to_owned(),
            ..PhotoMetadata::default()
        }
        .bounded();
        assert_eq!(metadata.camera, "Nikon ");
        assert!(!metadata.is_empty());
    }

    #[test]
    fn summary_joins_known_fields() {
        let metadata = PhotoMetadata {
            camera: "Nikon FM2".to_owned(),
            film: "Kodak Portra 400".to_owned(),
            iso: "400".to_owned(),
            aperture: "f/2.8".to_owned(),
            shutter_speed: "1/125".to_owned(),
            focal_length: "50".to_owned(),
            ..PhotoMetadata::default()
        };
        let summary = metadata.summary();
        assert!(summary.contains("Camera: Nikon FM2"));
        assert!(summary.contains("Film: Kodak Portra 400"));
        assert!(summary.contains("ISO: 400"));
        assert!(summary.contains("f/2.8"));
        assert!(summary.contains("1/125 s"));
        assert!(summary.contains("50 mm"));
    }
}
