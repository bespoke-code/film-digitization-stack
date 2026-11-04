//! Writes descriptive metadata as EXIF without an extra crate.
//!
//! PNG stores the TIFF block in an `eXIf` chunk, JPEG in an APP1 segment and
//! WebP in an `EXIF` chunk. TIFF and BMP are left untouched because their
//! metadata layout needs a different writer.

use image::ImageFormat;
use stitch_domain::PhotoMetadata;

const TYPE_ASCII: u16 = 2;
const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;
const TYPE_RATIONAL: u16 = 5;
const TYPE_UNDEFINED: u16 = 7;
const TYPE_SRATIONAL: u16 = 10;

const TAG_IMAGE_DESCRIPTION: u16 = 0x010E;
const TAG_MAKE: u16 = 0x010F;
const TAG_MODEL: u16 = 0x0110;
const TAG_SOFTWARE: u16 = 0x0131;
const TAG_DATETIME: u16 = 0x0132;
const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_EXPOSURE_TIME: u16 = 0x829A;
const TAG_FNUMBER: u16 = 0x829D;
const TAG_ISO: u16 = 0x8827;
const TAG_DATETIME_ORIGINAL: u16 = 0x9003;
const TAG_DATETIME_DIGITIZED: u16 = 0x9004;
const TAG_FOCAL_LENGTH: u16 = 0x920A;
const TAG_USER_COMMENT: u16 = 0x9286;
const TAG_RECOMMENDED_ISO: u16 = 0x8832;
const TAG_ISO_SPEED: u16 = 0x8833;
const TAG_LENS_SPECIFICATION: u16 = 0xA432;
const TAG_LENS_MODEL: u16 = 0xA434;

struct Entry {
    tag: u16,
    kind: u16,
    count: u32,
    data: Vec<u8>,
}

/// Adds an EXIF block to an encoded image. Returns `None` when the format is
/// unsupported or the file is malformed.
pub(crate) fn embed_metadata(
    bytes: &[u8],
    format: ImageFormat,
    metadata: &PhotoMetadata,
) -> Option<Vec<u8>> {
    let tiff = build_tiff(metadata);
    match format {
        ImageFormat::Png => insert_png_exif(bytes, &tiff),
        ImageFormat::Jpeg => insert_jpeg_exif(bytes, &tiff),
        ImageFormat::WebP => insert_webp_exif(bytes, &tiff),
        _ => None,
    }
}

fn build_tiff(metadata: &PhotoMetadata) -> Vec<u8> {
    let summary = metadata.summary();
    let datetime = exif_datetime(&metadata.date);
    let mut ifd0 = Vec::new();
    push_entry(&mut ifd0, ascii(TAG_IMAGE_DESCRIPTION, &summary));
    let camera = metadata.camera.trim();
    if !camera.is_empty() {
        let make = camera.split_whitespace().next().unwrap_or(camera);
        push_entry(&mut ifd0, ascii(TAG_MAKE, make));
        push_entry(&mut ifd0, ascii(TAG_MODEL, camera));
    }
    push_entry(&mut ifd0, ascii(TAG_SOFTWARE, "scan-to-img"));
    if let Some(datetime) = datetime.as_deref() {
        push_entry(&mut ifd0, ascii(TAG_DATETIME, datetime));
    }

    let mut exif = Vec::new();
    if let Some((numerator, denominator)) = parse_exposure(&metadata.shutter_speed) {
        exif.push(rational(TAG_EXPOSURE_TIME, numerator, denominator));
    }
    if let Some((numerator, denominator)) = parse_aperture(&metadata.aperture) {
        exif.push(rational(TAG_FNUMBER, numerator, denominator));
    }
    if let Some(iso) = parse_iso(&metadata.iso) {
        exif.push(short(TAG_ISO, iso));
    }
    if let Some(datetime) = datetime.as_deref() {
        push_entry(&mut exif, ascii(TAG_DATETIME_ORIGINAL, datetime));
    }
    if let Some((numerator, denominator)) = parse_focal_length(&metadata.focal_length) {
        exif.push(rational(TAG_FOCAL_LENGTH, numerator, denominator));
    }
    push_entry(&mut exif, ascii(TAG_LENS_MODEL, &metadata.lens));
    push_entry(&mut exif, user_comment(&labeled_comment(metadata)));

    ifd0.sort_by_key(|entry| entry.tag);
    exif.sort_by_key(|entry| entry.tag);

    let ifd0_count = ifd0.len() as u32 + u32::from(!exif.is_empty());
    let exif_ifd_offset = 8 + 2 + 12 * ifd0_count + 4;
    if !exif.is_empty() {
        ifd0.push(long(TAG_EXIF_IFD, exif_ifd_offset));
        ifd0.sort_by_key(|entry| entry.tag);
    }

    let data_start = if exif.is_empty() {
        exif_ifd_offset
    } else {
        exif_ifd_offset + 2 + 12 * exif.len() as u32 + 4
    };
    let mut next = data_start;
    let ifd0_offsets = assign_offsets(&ifd0, &mut next);
    let exif_offsets = assign_offsets(&exif, &mut next);

    let mut out = Vec::new();
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42u16.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());
    write_ifd(&mut out, &ifd0, &ifd0_offsets);
    if !exif.is_empty() {
        write_ifd(&mut out, &exif, &exif_offsets);
    }
    write_data(&mut out, &ifd0, &ifd0_offsets);
    write_data(&mut out, &exif, &exif_offsets);
    out
}

fn push_entry(entries: &mut Vec<Entry>, entry: Option<Entry>) {
    if let Some(entry) = entry {
        entries.push(entry);
    }
}

fn ascii(tag: u16, value: &str) -> Option<Entry> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let mut data = value.as_bytes().to_vec();
    data.push(0);
    Some(Entry {
        tag,
        kind: TYPE_ASCII,
        count: data.len() as u32,
        data,
    })
}

fn short(tag: u16, value: u32) -> Entry {
    let value = value.min(u32::from(u16::MAX)) as u16;
    Entry {
        tag,
        kind: TYPE_SHORT,
        count: 1,
        data: value.to_le_bytes().to_vec(),
    }
}

fn long(tag: u16, value: u32) -> Entry {
    Entry {
        tag,
        kind: TYPE_LONG,
        count: 1,
        data: value.to_le_bytes().to_vec(),
    }
}

fn rational(tag: u16, numerator: u32, denominator: u32) -> Entry {
    let (numerator, denominator) = reduce(numerator, denominator.max(1));
    let mut data = Vec::with_capacity(8);
    data.extend_from_slice(&numerator.to_le_bytes());
    data.extend_from_slice(&denominator.to_le_bytes());
    Entry {
        tag,
        kind: TYPE_RATIONAL,
        count: 1,
        data,
    }
}

fn user_comment(value: &str) -> Option<Entry> {
    if value.is_empty() {
        return None;
    }
    let mut data = b"ASCII\0\0\0".to_vec();
    data.extend_from_slice(value.as_bytes());
    Some(Entry {
        tag: TAG_USER_COMMENT,
        kind: TYPE_UNDEFINED,
        count: data.len() as u32,
        data,
    })
}

fn labeled_comment(metadata: &PhotoMetadata) -> String {
    let mut lines = Vec::new();
    push_line(&mut lines, "Camera", &metadata.camera);
    push_line(&mut lines, "Lens", &metadata.lens);
    push_line(&mut lines, "Film", &metadata.film);
    push_line(&mut lines, "ISO", &metadata.iso);
    push_line(&mut lines, "Aperture", &metadata.aperture);
    push_line(&mut lines, "Shutter speed", &metadata.shutter_speed);
    push_line(&mut lines, "Focal length", &metadata.focal_length);
    push_line(&mut lines, "Date", &metadata.date);
    push_line(&mut lines, "Location", &metadata.location);
    push_line(&mut lines, "Notes", &metadata.notes);
    lines.join("\n")
}

fn push_line(lines: &mut Vec<String>, label: &str, value: &str) {
    let value = value.trim();
    if !value.is_empty() {
        lines.push(format!("{label}: {value}"));
    }
}

fn assign_offsets(entries: &[Entry], next: &mut u32) -> Vec<Option<u32>> {
    entries
        .iter()
        .map(|entry| {
            if entry.data.len() <= 4 {
                return None;
            }
            if *next % 2 == 1 {
                *next += 1;
            }
            let offset = *next;
            *next += entry.data.len() as u32;
            Some(offset)
        })
        .collect()
}

fn write_ifd(out: &mut Vec<u8>, entries: &[Entry], offsets: &[Option<u32>]) {
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (entry, offset) in entries.iter().zip(offsets) {
        out.extend_from_slice(&entry.tag.to_le_bytes());
        out.extend_from_slice(&entry.kind.to_le_bytes());
        out.extend_from_slice(&entry.count.to_le_bytes());
        match offset {
            Some(offset) => out.extend_from_slice(&offset.to_le_bytes()),
            None => {
                let start = out.len();
                out.extend_from_slice(&entry.data);
                out.resize(start + 4, 0);
            }
        }
    }
    out.extend_from_slice(&0u32.to_le_bytes());
}

fn write_data(out: &mut Vec<u8>, entries: &[Entry], offsets: &[Option<u32>]) {
    for (entry, offset) in entries.iter().zip(offsets) {
        let Some(offset) = offset else {
            continue;
        };
        let offset = *offset as usize;
        if out.len() < offset {
            out.resize(offset, 0);
        }
        out.extend_from_slice(&entry.data);
    }
}

fn exif_datetime(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let normalized = raw.replace('-', ":");
    let (date, time) = match normalized.split_once([' ', 'T']) {
        Some((date, time)) => (date, time.trim()),
        None => (normalized.as_str(), ""),
    };
    let date_valid = date
        .chars()
        .enumerate()
        .all(|(index, character)| match index {
            4 | 7 => character == ':',
            _ => character.is_ascii_digit(),
        });
    if date.len() != 10 || !date_valid {
        return None;
    }
    let time = if valid_time(time) {
        if time.len() == 5 {
            format!("{time}:00")
        } else {
            time.to_owned()
        }
    } else {
        "00:00:00".to_owned()
    };
    Some(format!("{date} {time}"))
}

fn valid_time(value: &str) -> bool {
    if value.len() != 5 && value.len() != 8 {
        return false;
    }
    value
        .chars()
        .enumerate()
        .all(|(index, character)| match index {
            2 | 5 => character == ':',
            _ => character.is_ascii_digit(),
        })
}

fn parse_iso(raw: &str) -> Option<u32> {
    let value = raw.trim();
    let value = value
        .strip_prefix("ISO")
        .or_else(|| value.strip_prefix("iso"))
        .unwrap_or(value)
        .trim();
    value.parse().ok()
}

fn parse_exposure(raw: &str) -> Option<(u32, u32)> {
    let value = raw.trim().trim_end_matches('s').trim();
    if value.is_empty() {
        return None;
    }
    if let Some((numerator, denominator)) = value.split_once('/') {
        let numerator = numerator.trim().parse::<u32>().ok()?;
        let denominator = denominator.trim().parse::<u32>().ok()?;
        if numerator == 0 || denominator == 0 {
            return None;
        }
        return Some(reduce(numerator, denominator));
    }
    let seconds = value.parse::<f64>().ok()?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    if seconds >= 1.0 {
        let denominator = 1_000_000_u32;
        let numerator = (seconds * f64::from(denominator)).round();
        if numerator <= 0.0 || numerator > f64::from(u32::MAX) {
            return None;
        }
        Some(reduce(numerator as u32, denominator))
    } else {
        let denominator = (1.0 / seconds).round();
        if denominator < 1.0 || denominator > f64::from(u32::MAX) {
            return None;
        }
        Some(reduce(1, denominator as u32))
    }
}

fn parse_aperture(raw: &str) -> Option<(u32, u32)> {
    let value = raw.trim();
    let value = value
        .strip_prefix("f/")
        .or_else(|| value.strip_prefix("F/"))
        .unwrap_or(value)
        .trim();
    let value = value.parse::<f64>().ok()?;
    scaled_rational(value, 10)
}

fn parse_focal_length(raw: &str) -> Option<(u32, u32)> {
    let value = raw.trim().trim_end_matches("mm").trim();
    let value = value.parse::<f64>().ok()?;
    scaled_rational(value, 100)
}

fn scaled_rational(value: f64, scale: u32) -> Option<(u32, u32)> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let numerator = (value * f64::from(scale)).round();
    if numerator <= 0.0 || numerator > f64::from(u32::MAX) {
        return None;
    }
    Some(reduce(numerator as u32, scale))
}

fn reduce(numerator: u32, denominator: u32) -> (u32, u32) {
    let divisor = gcd(numerator, denominator);
    match (
        numerator.checked_div(divisor),
        denominator.checked_div(divisor),
    ) {
        (Some(numerator), Some(denominator)) => (numerator, denominator),
        _ => (0, 1),
    }
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn insert_png_exif(bytes: &[u8], tiff: &[u8]) -> Option<Vec<u8>> {
    const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() < 8 || &bytes[..8] != SIGNATURE {
        return None;
    }
    let mut position = 8;
    let mut insert_at = None;
    while position + 12 <= bytes.len() {
        let kind = &bytes[position + 4..position + 8];
        let length = u32::from_be_bytes(bytes[position..position + 4].try_into().ok()?) as usize;
        let chunk_end = position.checked_add(12)?.checked_add(length)?;
        if chunk_end > bytes.len() {
            return None;
        }
        if kind == b"eXIf" {
            return Some(bytes.to_vec());
        }
        if insert_at.is_none() && (kind == b"IDAT" || kind == b"IEND") {
            insert_at = Some(position);
        }
        if kind == b"IEND" {
            break;
        }
        position = chunk_end;
    }
    let end = insert_at?;
    let mut out = Vec::with_capacity(bytes.len() + 12 + tiff.len());
    out.extend_from_slice(&bytes[..end]);
    out.extend_from_slice(&(tiff.len() as u32).to_be_bytes());
    out.extend_from_slice(b"eXIf");
    out.extend_from_slice(tiff);
    let mut crc = Crc32::new();
    crc.update(b"eXIf");
    crc.update(tiff);
    out.extend_from_slice(&crc.finish().to_be_bytes());
    out.extend_from_slice(&bytes[end..]);
    Some(out)
}

fn insert_jpeg_exif(bytes: &[u8], tiff: &[u8]) -> Option<Vec<u8>> {
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let segment_length = 2 + 6 + tiff.len();
    if segment_length > usize::from(u16::MAX) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() + 2 + segment_length);
    out.extend_from_slice(&bytes[..2]);
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&(segment_length as u16).to_be_bytes());
    out.extend_from_slice(b"Exif\0\0");
    out.extend_from_slice(tiff);
    out.extend_from_slice(&bytes[2..]);
    Some(out)
}

fn insert_webp_exif(bytes: &[u8], tiff: &[u8]) -> Option<Vec<u8>> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return None;
    }
    let mut chunks: Vec<([u8; 4], &[u8])> = Vec::new();
    let mut position = 12;
    while position + 8 <= bytes.len() {
        let fourcc: [u8; 4] = bytes[position..position + 4].try_into().ok()?;
        let length =
            u32::from_le_bytes(bytes[position + 4..position + 8].try_into().ok()?) as usize;
        let data_start = position + 8;
        let data_end = data_start.checked_add(length)?;
        if data_end > bytes.len() {
            return None;
        }
        if &fourcc != b"EXIF" {
            chunks.push((fourcc, &bytes[data_start..data_end]));
        }
        position = data_end + (length & 1);
    }

    let has_vp8x = chunks.iter().any(|(fourcc, _)| fourcc == b"VP8X");
    let canvas = if has_vp8x {
        None
    } else {
        Some(simple_webp_canvas(&chunks)?)
    };

    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(b"WEBP");
    if let Some((width, height, alpha)) = canvas {
        let mut flags = 0x08;
        if alpha {
            flags |= 0x10;
        }
        out.extend_from_slice(b"VP8X");
        out.extend_from_slice(&10u32.to_le_bytes());
        out.push(flags);
        out.extend_from_slice(&[0, 0, 0]);
        out.extend_from_slice(&(width - 1).to_le_bytes()[..3]);
        out.extend_from_slice(&(height - 1).to_le_bytes()[..3]);
    }
    for (fourcc, data) in &chunks {
        out.extend_from_slice(fourcc);
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        if data.len() & 1 != 0 {
            out.push(0);
        }
    }
    out.extend_from_slice(b"EXIF");
    out.extend_from_slice(&(tiff.len() as u32).to_le_bytes());
    out.extend_from_slice(tiff);
    if tiff.len() & 1 != 0 {
        out.push(0);
    }
    if has_vp8x {
        let mut position = 12;
        while position + 8 <= out.len() {
            let length =
                u32::from_le_bytes(out[position + 4..position + 8].try_into().ok()?) as usize;
            if &out[position..position + 4] == b"VP8X" {
                out[position + 8] |= 0x08;
                break;
            }
            position = position + 8 + length + (length & 1);
        }
    }
    let riff_size = (out.len() - 8) as u32;
    out[4..8].copy_from_slice(&riff_size.to_le_bytes());
    Some(out)
}

/// Reads the canvas size of a simple WebP file that has VP8 or VP8L data only.
fn simple_webp_canvas(chunks: &[([u8; 4], &[u8])]) -> Option<(u32, u32, bool)> {
    for (fourcc, data) in chunks {
        if fourcc == b"VP8L" {
            let bits = u32::from_le_bytes(data.get(1..5)?.try_into().ok()?);
            if data.first() != Some(&0x2F) {
                return None;
            }
            let width = 1 + (bits & 0x3FFF);
            let height = 1 + ((bits >> 14) & 0x3FFF);
            let alpha = (bits >> 28) & 1 != 0;
            return Some((width, height, alpha));
        }
        if fourcc == b"VP8 " {
            if data.get(3..6)? != b"\x9d\x01\x2a" {
                return None;
            }
            let width = u32::from(u16::from_le_bytes(data.get(6..8)?.try_into().ok()?) & 0x3FFF);
            let height = u32::from(u16::from_le_bytes(data.get(8..10)?.try_into().ok()?) & 0x3FFF);
            return Some((width, height, false));
        }
    }
    None
}

/// Reads descriptive metadata from PNG, JPEG, WebP or TIFF bytes.
pub(crate) fn extract_metadata(bytes: &[u8]) -> PhotoMetadata {
    exif_tiff(bytes)
        .and_then(Tiff::new)
        .and_then(|tiff| tiff.metadata())
        .unwrap_or_default()
}

fn exif_tiff(bytes: &[u8]) -> Option<&[u8]> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        png_exif(bytes)
    } else if bytes.starts_with(&[0xFF, 0xD8]) {
        jpeg_exif(bytes)
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(&b"WEBP"[..]) {
        webp_exif(bytes)
    } else if bytes.starts_with(b"II*\0") || bytes.starts_with(b"MM\0*") {
        Some(bytes)
    } else {
        None
    }
}

fn png_exif(bytes: &[u8]) -> Option<&[u8]> {
    let mut position = 8;
    while position + 12 <= bytes.len() {
        let length = u32::from_be_bytes(bytes[position..position + 4].try_into().ok()?) as usize;
        let data_start = position + 8;
        let data_end = data_start.checked_add(length)?;
        if data_end > bytes.len() {
            return None;
        }
        if &bytes[position + 4..position + 8] == b"eXIf" {
            return Some(&bytes[data_start..data_end]);
        }
        position = data_end.checked_add(4)?;
    }
    None
}

fn jpeg_exif(bytes: &[u8]) -> Option<&[u8]> {
    let mut position = 2;
    while position + 1 < bytes.len() {
        if bytes[position] != 0xFF {
            return None;
        }
        let mut marker = position + 1;
        while marker < bytes.len() && bytes[marker] == 0xFF {
            marker += 1;
        }
        let marker_kind = *bytes.get(marker)?;
        if marker_kind == 0xDA || marker_kind == 0xD9 {
            return None;
        }
        if marker_kind == 0x01 || (0xD0..=0xD7).contains(&marker_kind) {
            position = marker + 1;
            continue;
        }
        let length_start = marker + 1;
        let length = usize::from(u16::from_be_bytes(
            bytes.get(length_start..length_start + 2)?.try_into().ok()?,
        ));
        if length < 2 {
            return None;
        }
        let payload_start = length_start + 2;
        let payload_end = length_start.checked_add(length)?;
        if payload_end > bytes.len() {
            return None;
        }
        if marker_kind == 0xE1 && bytes[payload_start..payload_end].starts_with(b"Exif\0\0") {
            return Some(&bytes[payload_start + 6..payload_end]);
        }
        position = payload_end;
    }
    None
}

fn webp_exif(bytes: &[u8]) -> Option<&[u8]> {
    let mut position = 12;
    while position + 8 <= bytes.len() {
        let fourcc = &bytes[position..position + 4];
        let length =
            u32::from_le_bytes(bytes[position + 4..position + 8].try_into().ok()?) as usize;
        let data_start = position + 8;
        let data_end = data_start.checked_add(length)?;
        if data_end > bytes.len() {
            return None;
        }
        if fourcc == b"EXIF" {
            let data = &bytes[data_start..data_end];
            return Some(data.strip_prefix(b"Exif\0\0").unwrap_or(data));
        }
        position = data_end.checked_add(length & 1)?;
    }
    None
}

#[derive(Clone, Copy)]
enum Endian {
    Little,
    Big,
}

struct Tiff<'a> {
    bytes: &'a [u8],
    endian: Endian,
}

struct TiffEntry {
    tag: u16,
    kind: u16,
    count: u32,
    total: usize,
    data_offset: usize,
}

impl<'a> Tiff<'a> {
    fn new(bytes: &'a [u8]) -> Option<Self> {
        let endian = if bytes.starts_with(b"II") {
            Endian::Little
        } else if bytes.starts_with(b"MM") {
            Endian::Big
        } else {
            return None;
        };
        let tiff = Self { bytes, endian };
        (tiff.u16_at(2)? == 42).then_some(tiff)
    }

    fn u16_at(&self, offset: usize) -> Option<u16> {
        let bytes: [u8; 2] = self
            .bytes
            .get(offset..offset.checked_add(2)?)?
            .try_into()
            .ok()?;
        Some(match self.endian {
            Endian::Little => u16::from_le_bytes(bytes),
            Endian::Big => u16::from_be_bytes(bytes),
        })
    }

    fn u32_at(&self, offset: usize) -> Option<u32> {
        let bytes: [u8; 4] = self
            .bytes
            .get(offset..offset.checked_add(4)?)?
            .try_into()
            .ok()?;
        Some(match self.endian {
            Endian::Little => u32::from_le_bytes(bytes),
            Endian::Big => u32::from_be_bytes(bytes),
        })
    }

    fn ifd(&self, offset: usize) -> Option<Vec<TiffEntry>> {
        let count = usize::from(self.u16_at(offset)?);
        let start = offset.checked_add(2)?;
        let end = start.checked_add(count.checked_mul(12)?)?.checked_add(4)?;
        if end > self.bytes.len() {
            return None;
        }
        let mut entries = Vec::with_capacity(count);
        for index in 0..count {
            let position = start + index * 12;
            let tag = self.u16_at(position)?;
            let kind = self.u16_at(position + 2)?;
            let count = self.u32_at(position + 4)?;
            let Some(size) = type_size(kind) else {
                continue;
            };
            let Some(total) = size.checked_mul(u64::from(count)) else {
                continue;
            };
            let Ok(total) = usize::try_from(total) else {
                continue;
            };
            let data_offset = if total <= 4 {
                position + 8
            } else {
                let offset = self.u32_at(position + 8)? as usize;
                let Some(data_end) = offset.checked_add(total) else {
                    continue;
                };
                if data_end > self.bytes.len() {
                    continue;
                }
                offset
            };
            entries.push(TiffEntry {
                tag,
                kind,
                count,
                total,
                data_offset,
            });
        }
        Some(entries)
    }

    fn entry_bytes(&self, entry: &TiffEntry) -> Option<&'a [u8]> {
        let end = entry.data_offset.checked_add(entry.total)?;
        self.bytes.get(entry.data_offset..end)
    }

    fn ascii(&self, entry: &TiffEntry) -> Option<String> {
        if entry.kind != TYPE_ASCII {
            return None;
        }
        let bytes = self.entry_bytes(entry)?;
        let end = bytes
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(bytes.len());
        let text = String::from_utf8_lossy(&bytes[..end]).trim().to_owned();
        (!text.is_empty()).then_some(text)
    }

    fn short_values(&self, entry: &TiffEntry) -> Option<Vec<u16>> {
        if entry.kind != TYPE_SHORT {
            return None;
        }
        (0..entry.count as usize)
            .map(|index| self.u16_at(entry.data_offset + index * 2))
            .collect()
    }

    fn long_values(&self, entry: &TiffEntry) -> Option<Vec<u32>> {
        if entry.kind != TYPE_LONG {
            return None;
        }
        (0..entry.count as usize)
            .map(|index| self.u32_at(entry.data_offset + index * 4))
            .collect()
    }

    fn rational_values(&self, entry: &TiffEntry) -> Option<Vec<(u32, u32)>> {
        if entry.kind != TYPE_RATIONAL && entry.kind != TYPE_SRATIONAL {
            return None;
        }
        (0..entry.count as usize)
            .map(|index| {
                let offset = entry.data_offset + index * 8;
                Some((self.u32_at(offset)?, self.u32_at(offset + 4)?))
            })
            .collect()
    }

    fn integer(&self, entry: &TiffEntry) -> Option<u32> {
        match entry.kind {
            TYPE_SHORT => self.short_values(entry)?.first().copied().map(u32::from),
            TYPE_LONG => self.long_values(entry)?.first().copied(),
            _ => None,
        }
    }

    fn user_comment(&self, entry: &TiffEntry) -> Option<String> {
        if entry.kind != TYPE_UNDEFINED {
            return None;
        }
        let bytes = self.entry_bytes(entry)?;
        let text = if let Some(rest) = bytes.strip_prefix(b"ASCII\0\0\0") {
            String::from_utf8_lossy(rest)
                .trim_matches(|character: char| character == '\0' || character.is_whitespace())
                .to_owned()
        } else if let Some(rest) = bytes.strip_prefix(b"UNICODE\0") {
            let (chunks, _) = rest.as_chunks::<2>();
            let units: Vec<u16> = chunks
                .iter()
                .map(|chunk| match self.endian {
                    Endian::Little => u16::from_le_bytes(*chunk),
                    Endian::Big => u16::from_be_bytes(*chunk),
                })
                .take_while(|unit| *unit != 0)
                .collect();
            String::from_utf16_lossy(&units).trim().to_owned()
        } else {
            String::from_utf8_lossy(bytes)
                .trim_matches(|character: char| character == '\0' || character.is_whitespace())
                .to_owned()
        };
        (!text.is_empty()).then_some(text)
    }

    fn metadata(&self) -> Option<PhotoMetadata> {
        let ifd0_offset = self.u32_at(4)? as usize;
        let ifd0 = self.ifd(ifd0_offset)?;
        let exif = find(&ifd0, TAG_EXIF_IFD)
            .and_then(|entry| self.integer(entry))
            .and_then(|offset| self.ifd(offset as usize))
            .unwrap_or_default();

        let comment = find(&exif, TAG_USER_COMMENT).and_then(|entry| self.user_comment(entry));
        let description = find(&ifd0, TAG_IMAGE_DESCRIPTION).and_then(|entry| self.ascii(entry));
        let mut metadata = match comment {
            Some(comment) => {
                let parsed = parse_labeled_comment(&comment);
                if parsed.is_empty() {
                    PhotoMetadata {
                        notes: comment,
                        ..PhotoMetadata::default()
                    }
                } else {
                    parsed
                }
            }
            None => PhotoMetadata {
                notes: description.unwrap_or_default(),
                ..PhotoMetadata::default()
            },
        };

        if let Some(camera) = self.camera(&ifd0) {
            metadata.camera = camera;
        }
        if let Some(lens) = self.lens(&exif) {
            metadata.lens = lens;
        }
        if let Some(iso) = self.iso(&exif) {
            metadata.iso = iso;
        }
        if let Some(aperture) = self.aperture(&exif) {
            metadata.aperture = aperture;
        }
        if let Some(shutter) = self.shutter(&exif) {
            metadata.shutter_speed = shutter;
        }
        if let Some(focal_length) = self.focal_length(&exif) {
            metadata.focal_length = focal_length;
        }
        if let Some(date) = self.date(&ifd0, &exif) {
            metadata.date = date;
        }
        Some(metadata)
    }

    fn camera(&self, ifd0: &[TiffEntry]) -> Option<String> {
        let make = find(ifd0, TAG_MAKE).and_then(|entry| self.ascii(entry));
        let model = find(ifd0, TAG_MODEL).and_then(|entry| self.ascii(entry));
        match (make, model) {
            (Some(make), Some(model)) => {
                let first_make = make.split_whitespace().next().unwrap_or_default();
                if !first_make.is_empty()
                    && model
                        .to_ascii_lowercase()
                        .starts_with(&first_make.to_ascii_lowercase())
                {
                    Some(model)
                } else {
                    Some(format!("{make} {model}"))
                }
            }
            (Some(make), None) => Some(make),
            (None, Some(model)) => Some(model),
            (None, None) => None,
        }
    }

    fn lens(&self, exif: &[TiffEntry]) -> Option<String> {
        if let Some(lens) = find(exif, TAG_LENS_MODEL).and_then(|entry| self.ascii(entry)) {
            return Some(lens);
        }
        let specification =
            find(exif, TAG_LENS_SPECIFICATION).and_then(|entry| self.rational_values(entry))?;
        if specification.len() < 2 {
            return None;
        }
        let minimum = format_decimal(specification[0].0, specification[0].1);
        let maximum = format_decimal(specification[1].0, specification[1].1);
        let focal_length = if minimum == maximum {
            format!("{minimum} mm")
        } else {
            format!("{minimum}-{maximum} mm")
        };
        let aperture = match specification.get(3) {
            Some((numerator, denominator)) if *numerator != 0 && *denominator != 0 => {
                Some(format!("f/{}", format_decimal(*numerator, *denominator)))
            }
            _ => None,
        };
        Some(match aperture {
            Some(aperture) => format!("{focal_length} {aperture}"),
            None => focal_length,
        })
    }

    fn iso(&self, exif: &[TiffEntry]) -> Option<String> {
        for tag in [TAG_ISO, TAG_ISO_SPEED, TAG_RECOMMENDED_ISO] {
            if let Some(value) = find(exif, tag).and_then(|entry| self.integer(entry)) {
                if value > 0 {
                    return Some(value.to_string());
                }
            }
        }
        None
    }

    fn aperture(&self, exif: &[TiffEntry]) -> Option<String> {
        let (numerator, denominator) = find(exif, TAG_FNUMBER)
            .and_then(|entry| self.rational_values(entry))?
            .first()
            .copied()?;
        if numerator == 0 || denominator == 0 {
            return None;
        }
        Some(format!("f/{}", format_decimal(numerator, denominator)))
    }

    fn shutter(&self, exif: &[TiffEntry]) -> Option<String> {
        let (numerator, denominator) = find(exif, TAG_EXPOSURE_TIME)
            .and_then(|entry| self.rational_values(entry))?
            .first()
            .copied()?;
        if numerator == 0 || denominator == 0 {
            return None;
        }
        Some(format_exposure(numerator, denominator))
    }

    fn focal_length(&self, exif: &[TiffEntry]) -> Option<String> {
        let (numerator, denominator) = find(exif, TAG_FOCAL_LENGTH)
            .and_then(|entry| self.rational_values(entry))?
            .first()
            .copied()?;
        if numerator == 0 || denominator == 0 {
            return None;
        }
        Some(format!("{} mm", format_decimal(numerator, denominator)))
    }

    fn date(&self, ifd0: &[TiffEntry], exif: &[TiffEntry]) -> Option<String> {
        for entries in [exif, ifd0] {
            for tag in [TAG_DATETIME_ORIGINAL, TAG_DATETIME_DIGITIZED, TAG_DATETIME] {
                if let Some(raw) = find(entries, tag).and_then(|entry| self.ascii(entry)) {
                    if let Some(date) = normalize_date(&raw) {
                        return Some(date);
                    }
                }
            }
        }
        None
    }
}

fn find(entries: &[TiffEntry], tag: u16) -> Option<&TiffEntry> {
    entries.iter().find(|entry| entry.tag == tag)
}

fn type_size(kind: u16) -> Option<u64> {
    Some(match kind {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 => 8,
        _ => return None,
    })
}

fn parse_labeled_comment(text: &str) -> PhotoMetadata {
    let mut metadata = PhotoMetadata::default();
    let mut in_notes = false;
    for line in text.lines() {
        let trimmed = line.trim();
        let Some((label, value)) = trimmed.split_once(':') else {
            if in_notes && !trimmed.is_empty() {
                if !metadata.notes.is_empty() {
                    metadata.notes.push('\n');
                }
                metadata.notes.push_str(line);
            }
            continue;
        };
        let label = label.trim().to_ascii_lowercase();
        let value = value.trim();
        in_notes = false;
        match label.as_str() {
            "camera" => metadata.camera = value.to_owned(),
            "lens" => metadata.lens = value.to_owned(),
            "film" => metadata.film = value.to_owned(),
            "iso" => metadata.iso = value.to_owned(),
            "aperture" => metadata.aperture = value.to_owned(),
            "shutter" | "shutter speed" => metadata.shutter_speed = value.to_owned(),
            "focal length" => metadata.focal_length = value.to_owned(),
            "date" => metadata.date = value.to_owned(),
            "location" => metadata.location = value.to_owned(),
            "notes" => {
                metadata.notes = value.to_owned();
                in_notes = true;
            }
            _ => {}
        }
    }
    metadata
}

fn normalize_date(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let normalized = raw.replace('T', " ");
    let (date, time) = match normalized.split_once(' ') {
        Some((date, time)) => (date, time.trim()),
        None => (normalized.as_str(), ""),
    };
    let date = date.replace(':', "-");
    let valid = date.len() == 10
        && date
            .chars()
            .enumerate()
            .all(|(index, character)| match index {
                4 | 7 => character == '-',
                _ => character.is_ascii_digit(),
            });
    if !valid {
        return Some(raw.to_owned());
    }
    if time.is_empty() {
        Some(date)
    } else {
        Some(format!("{date} {time}"))
    }
}

fn format_decimal(numerator: u32, denominator: u32) -> String {
    if denominator == 0 {
        return String::new();
    }
    let value = f64::from(numerator) / f64::from(denominator);
    let mut text = format!("{value:.4}");
    while text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}

fn format_exposure(numerator: u32, denominator: u32) -> String {
    let (numerator, denominator) = reduce(numerator, denominator.max(1));
    if denominator == 1 {
        numerator.to_string()
    } else if numerator == 1 {
        format!("1/{denominator}")
    } else {
        format_decimal(numerator, denominator)
    }
}

struct Crc32(u32);

impl Crc32 {
    fn new() -> Self {
        Self(0xFFFF_FFFF)
    }

    fn update(&mut self, data: &[u8]) {
        for &byte in data {
            let mut value = (self.0 ^ u32::from(byte)) & 0xFF;
            for _ in 0..8 {
                value = if value & 1 != 0 {
                    0xEDB8_8320 ^ (value >> 1)
                } else {
                    value >> 1
                };
            }
            self.0 = (self.0 >> 8) ^ value;
        }
    }

    fn finish(self) -> u32 {
        self.0 ^ 0xFFFF_FFFF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> PhotoMetadata {
        PhotoMetadata {
            camera: "Nikon FM2".to_owned(),
            lens: "Nikkor 50mm f/1.8".to_owned(),
            film: "Kodak Portra 400".to_owned(),
            iso: "400".to_owned(),
            aperture: "f/2.8".to_owned(),
            shutter_speed: "1/125".to_owned(),
            focal_length: "50 mm".to_owned(),
            date: "2024-05-01".to_owned(),
            location: "Berlin".to_owned(),
            notes: "Scanned with a Fujifilm X-T5".to_owned(),
        }
    }

    #[test]
    fn tiff_contains_standard_and_descriptive_tags() {
        let tiff = build_tiff(&metadata());
        assert_eq!(&tiff[..4], b"II*\0");
        let text = String::from_utf8_lossy(&tiff);
        assert!(text.contains("Nikon FM2"));
        assert!(text.contains("Kodak Portra 400"));
        assert!(text.contains("2024:05:01 00:00:00"));
        assert!(text.contains("scan-to-img"));
    }

    #[test]
    fn png_receives_an_exif_chunk() {
        let image = image::RgbaImage::new(8, 8);
        let mut encoded = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut encoded), ImageFormat::Png)
            .unwrap();
        let tagged = embed_metadata(&encoded, ImageFormat::Png, &metadata()).unwrap();
        let exif = tagged.windows(4).position(|window| window == b"eXIf");
        let idat = tagged.windows(4).position(|window| window == b"IDAT");
        assert!(exif.is_some() && exif < idat);
        assert!(tagged.len() > encoded.len());
        image::load_from_memory(&tagged).expect("tagged PNG must still decode");
    }

    #[test]
    fn jpeg_receives_an_app1_segment() {
        let image = image::RgbImage::new(8, 8);
        let mut encoded = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut encoded), ImageFormat::Jpeg)
            .unwrap();
        let tagged = embed_metadata(&encoded, ImageFormat::Jpeg, &metadata()).unwrap();
        assert_eq!(&tagged[..2], &[0xFF, 0xD8]);
        assert_eq!(&tagged[2..4], &[0xFF, 0xE1]);
        assert_eq!(&tagged[6..12], b"Exif\0\0");
        image::load_from_memory(&tagged).expect("tagged JPEG must still decode");
    }

    #[test]
    fn webp_receives_an_exif_chunk() {
        let image = image::RgbaImage::new(8, 8);
        let mut encoded = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut encoded), ImageFormat::WebP)
            .unwrap();
        let tagged = embed_metadata(&encoded, ImageFormat::WebP, &metadata()).unwrap();
        assert_eq!(&tagged[12..16], b"VP8X");
        assert_eq!(tagged[20] & 0x08, 0x08);
        assert!(tagged.windows(4).any(|window| window == b"EXIF"));
        image::load_from_memory(&tagged).expect("tagged WebP must still decode");
    }

    #[test]
    fn generated_files_round_trip_their_metadata() {
        let metadata = metadata();
        for format in [ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::WebP] {
            let mut encoded = Vec::new();
            if format == ImageFormat::Jpeg {
                image::RgbImage::new(8, 8)
                    .write_to(&mut std::io::Cursor::new(&mut encoded), format)
                    .unwrap();
            } else {
                image::RgbaImage::new(8, 8)
                    .write_to(&mut std::io::Cursor::new(&mut encoded), format)
                    .unwrap();
            }
            let tagged = embed_metadata(&encoded, format, &metadata).unwrap();
            let extracted = extract_metadata(&tagged);
            assert_eq!(extracted.camera, metadata.camera);
            assert_eq!(extracted.lens, metadata.lens);
            assert_eq!(extracted.film, metadata.film);
            assert_eq!(extracted.iso, metadata.iso);
            assert_eq!(extracted.aperture, metadata.aperture);
            assert_eq!(extracted.shutter_speed, metadata.shutter_speed);
            assert_eq!(extracted.focal_length, metadata.focal_length);
            assert!(extracted.date.starts_with("2024-05-01"));
            assert_eq!(extracted.location, metadata.location);
            assert_eq!(extracted.notes, metadata.notes);
        }
    }

    #[test]
    fn tiff_bytes_are_read_directly() {
        let tiff = build_tiff(&metadata());
        let extracted = extract_metadata(&tiff);
        assert_eq!(extracted.camera, "Nikon FM2");
        assert_eq!(extracted.film, "Kodak Portra 400");
        assert_eq!(extracted.location, "Berlin");
    }

    #[test]
    fn labeled_comments_keep_multiline_notes() {
        let parsed = parse_labeled_comment("Camera: Nikon FM2\nNotes: first line\nsecond line");
        assert_eq!(parsed.camera, "Nikon FM2");
        assert_eq!(parsed.notes, "first line\nsecond line");
    }

    #[test]
    fn dates_are_normalized_for_the_editor() {
        assert_eq!(
            normalize_date("2024:05:01 12:30:00").as_deref(),
            Some("2024-05-01 12:30:00")
        );
        assert_eq!(normalize_date("2024-05-01").as_deref(), Some("2024-05-01"));
    }

    #[test]
    fn rational_values_are_reduced() {
        assert_eq!(parse_exposure("1/125"), Some((1, 125)));
        assert_eq!(parse_exposure("0.5"), Some((1, 2)));
        assert_eq!(parse_exposure("2"), Some((2, 1)));
        assert_eq!(parse_aperture("f/2.8"), Some((14, 5)));
        assert_eq!(parse_focal_length("50 mm"), Some((50, 1)));
    }

    #[test]
    fn dates_accept_dashes_or_colons() {
        assert_eq!(
            exif_datetime("2024-05-01"),
            Some("2024:05:01 00:00:00".to_owned())
        );
        assert_eq!(
            exif_datetime("2024:05:01 12:30"),
            Some("2024:05:01 12:30:00".to_owned())
        );
        assert_eq!(exif_datetime("May 2024"), None);
    }
}
