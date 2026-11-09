//! Reads the encoder quality from a JPEG without decoding its pixels.

/// Luminance quantization table from ITU-T T.81, Annex K.1.
#[rustfmt::skip]
const STANDARD_LUMA: [u8; 64] = [
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99,
];

/// Maps a zigzag position to its coefficient index.
#[rustfmt::skip]
const UNZIGZAG: [u8; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// Estimates the quality of a baseline JPEG from its luminance quantization
/// table. Returns `None` for non-JPEG data, 12-bit tables or custom tables that
/// do not resemble the standard scaling curve.
pub(crate) fn jpeg_quality(bytes: &[u8]) -> Option<u8> {
    let table = find_luma_table(bytes)?;
    let mut best = (0u8, u32::MAX);
    for quality in 1..=100u8 {
        let candidate = scaled_luma(quality);
        let score: u32 = table
            .iter()
            .zip(candidate)
            .map(|(actual, expected)| u32::from(actual.abs_diff(expected)))
            .sum();
        if score <= best.1 {
            best = (quality, score);
        }
    }
    (best.1 <= 64 * 4).then_some(best.0)
}

fn scaled_luma(quality: u8) -> [u8; 64] {
    let quality = u32::from(quality.clamp(1, 100));
    let scale = if quality < 50 {
        5000 / quality
    } else {
        200 - quality * 2
    };
    let mut table = [0u8; 64];
    for (value, standard) in table.iter_mut().zip(STANDARD_LUMA) {
        *value = (((u32::from(standard) * scale + 50) / 100).clamp(1, 255)) as u8;
    }
    table
}

fn find_luma_table(bytes: &[u8]) -> Option<[u8; 64]> {
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    let mut position = 2;
    while position + 1 < bytes.len() {
        if bytes[position] != 0xFF {
            return None;
        }
        let mut marker = position + 1;
        while marker < bytes.len() && bytes[marker] == 0xFF {
            marker += 1;
        }
        let kind = *bytes.get(marker)?;
        if kind == 0xDA || kind == 0xD9 {
            return None;
        }
        if kind == 0x01 || kind == 0xD8 || (0xD0..=0xD7).contains(&kind) {
            position = marker + 1;
            continue;
        }
        let length = usize::from(u16::from_be_bytes(
            bytes.get(marker + 1..marker + 3)?.try_into().ok()?,
        ));
        if length < 2 {
            return None;
        }
        let payload_start = marker + 3;
        let payload_end = marker.checked_add(1 + length)?;
        let payload = bytes.get(payload_start..payload_end)?;
        if kind == 0xDB {
            if let Some(table) = luma_table(payload) {
                return Some(table);
            }
        }
        position = payload_end;
    }
    None
}

fn luma_table(payload: &[u8]) -> Option<[u8; 64]> {
    let mut position = 0;
    while position + 65 <= payload.len() {
        let precision = payload[position] >> 4;
        let destination = payload[position] & 0x0F;
        position += 1;
        if precision == 0 {
            if destination == 0 {
                let zigzag: [u8; 64] = payload.get(position..position + 64)?.try_into().ok()?;
                let mut table = [0u8; 64];
                for (index, value) in UNZIGZAG.iter().zip(zigzag) {
                    table[*index as usize] = value;
                }
                return Some(table);
            }
            position += 64;
        } else if precision == 1 {
            position += 128;
        } else {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::jpeg::JpegEncoder;
    use image::{ExtendedColorType, RgbImage};

    fn encoded(quality: u8) -> Vec<u8> {
        let image = RgbImage::from_fn(32, 24, |x, y| {
            image::Rgb([(x * 5) as u8, (y * 7) as u8, 90])
        });
        let mut bytes = Vec::new();
        JpegEncoder::new_with_quality(&mut bytes, quality)
            .encode(image.as_raw(), 32, 24, ExtendedColorType::Rgb8)
            .unwrap();
        bytes
    }

    #[test]
    fn reads_quality_from_standard_tables() {
        for quality in [10, 40, 75, 90, 92, 100] {
            assert_eq!(jpeg_quality(&encoded(quality)), Some(quality));
        }
    }

    #[test]
    fn ignores_non_jpeg_data() {
        assert_eq!(jpeg_quality(b"not a jpeg"), None);
    }

    #[test]
    fn reads_example_input_quality() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/example-input.jpg");
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(jpeg_quality(&bytes), Some(92));
    }
}
