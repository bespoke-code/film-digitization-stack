//! Minimal dependency-free file browser and save dialog.
//!
//! A native picker would pull in GTK/portal bindings. This dialog only needs
//! `std::fs`, so the app keeps working on Linux and macOS without extra system
//! libraries.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use egui::{Context, RichText};

use stitch_domain::DEFAULT_JPEG_QUALITY;

pub(crate) const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"];

/// Camera RAW extensions. They are listed here only to explain why they are
/// skipped, because this build has no RAW decoder.
pub(crate) const RAW_EXTENSIONS: &[&str] = &[
    "3fr", "ari", "arw", "bay", "cr2", "cr3", "crw", "dcr", "dng", "erf", "fff", "iiq", "k25",
    "kdc", "mef", "mos", "mrw", "nef", "nrw", "orf", "pef", "raf", "raw", "rw2", "rwl", "sr2",
    "srf", "srw", "x3f",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DialogMode {
    AddImages,
    OpenFolder,
    SaveImage,
    SaveFolder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SaveFormat {
    Png,
    Jpeg,
    Tiff,
    WebP,
    Bmp,
}

impl SaveFormat {
    const ALL: [Self; 5] = [Self::Png, Self::Jpeg, Self::Tiff, Self::WebP, Self::Bmp];

    pub(crate) fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Tiff => "tiff",
            Self::WebP => "webp",
            Self::Bmp => "bmp",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Tiff => "TIFF",
            Self::WebP => "WebP",
            Self::Bmp => "BMP",
        }
    }
}

pub(crate) enum DialogOutcome {
    AddImages(Vec<PathBuf>),
    OpenFolder(PathBuf),
    SaveImage(PathBuf, u8),
    SaveFolder(PathBuf, SaveFormat, u8),
}

struct Entry {
    path: PathBuf,
    name: String,
    is_dir: bool,
}

pub(crate) struct FileDialog {
    open: bool,
    mode: DialogMode,
    current: PathBuf,
    entries: Vec<Entry>,
    selected: BTreeSet<PathBuf>,
    filename: String,
    format: SaveFormat,
    quality: u8,
    error: Option<String>,
    needs_reload: bool,
}

impl Default for FileDialog {
    fn default() -> Self {
        Self {
            open: false,
            mode: DialogMode::AddImages,
            current: std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            entries: Vec::new(),
            selected: BTreeSet::new(),
            filename: String::new(),
            format: SaveFormat::Png,
            quality: DEFAULT_JPEG_QUALITY,
            error: None,
            needs_reload: true,
        }
    }
}

impl FileDialog {
    pub(crate) fn open_in(
        &mut self,
        mode: DialogMode,
        start: Option<PathBuf>,
        filename: String,
        quality: u8,
    ) {
        self.mode = mode;
        self.selected.clear();
        self.error = None;
        self.filename = filename;
        self.quality = quality.clamp(1, 100);
        if let Some(start) = start {
            let start = if start.is_dir() {
                start
            } else {
                start
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| self.current.clone())
            };
            self.current = start;
        }
        self.open = true;
        self.needs_reload = true;
    }

    pub(crate) fn show(&mut self, ctx: &Context) -> Option<DialogOutcome> {
        if !self.open {
            return None;
        }
        if self.needs_reload {
            self.reload();
        }

        let title = match self.mode {
            DialogMode::AddImages => "Add images",
            DialogMode::OpenFolder => "Open folder",
            DialogMode::SaveImage => "Export image",
            DialogMode::SaveFolder => "Export all images",
        };

        let mut open = self.open;
        let mut outcome = None;
        egui::Window::new(title)
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size([620.0, 460.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                self.draw(ui, &mut outcome);
            });

        if !open {
            self.open = false;
        }
        if outcome.is_some() {
            self.open = false;
        }
        outcome
    }

    fn draw(&mut self, ui: &mut egui::Ui, outcome: &mut Option<DialogOutcome>) {
        ui.horizontal(|ui| {
            if ui.button("Up").clicked() {
                if let Some(parent) = self.current.parent() {
                    self.current = parent.to_path_buf();
                    self.selected.clear();
                    self.needs_reload = true;
                }
            }
            if ui.button("Home").clicked() {
                if let Some(home) = std::env::var_os("HOME") {
                    self.current = PathBuf::from(home);
                    self.selected.clear();
                    self.needs_reload = true;
                }
            }
            if ui.button("Refresh").clicked() {
                self.needs_reload = true;
            }
            ui.add(
                egui::Label::new(RichText::new(self.current.display().to_string()).monospace())
                    .truncate(),
            );
        });

        ui.separator();

        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(230, 100, 100), error);
        }

        let mut go_to = None;
        egui::ScrollArea::vertical()
            .max_height(300.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if self.entries.is_empty() {
                    ui.weak("This folder contains no supported entries.");
                }
                for entry in &self.entries {
                    if entry.is_dir {
                        if file_row(ui, true, &entry.name, false, true).clicked() {
                            go_to = Some(entry.path.clone());
                        }
                    } else if self.mode == DialogMode::AddImages {
                        let selected = self.selected.contains(&entry.path);
                        let response = file_row(ui, false, &entry.name, selected, true);
                        if response.clicked() {
                            if selected {
                                self.selected.remove(&entry.path);
                            } else {
                                self.selected.insert(entry.path.clone());
                            }
                        }
                        if response.double_clicked() {
                            *outcome = Some(DialogOutcome::AddImages(vec![entry.path.clone()]));
                        }
                    } else {
                        file_row(ui, false, &entry.name, false, false);
                    }
                }
            });
        if let Some(path) = go_to {
            self.current = path;
            self.selected.clear();
            self.needs_reload = true;
        }

        ui.separator();
        match self.mode {
            DialogMode::AddImages => {
                ui.horizontal(|ui| {
                    ui.label(format!("{} selected", self.selected.len()));
                    if ui.button("Select all").clicked() {
                        for entry in &self.entries {
                            if !entry.is_dir {
                                self.selected.insert(entry.path.clone());
                            }
                        }
                    }
                    if ui.button("Clear selection").clicked() {
                        self.selected.clear();
                    }
                    if ui
                        .add_enabled(!self.selected.is_empty(), egui::Button::new("Add selected"))
                        .clicked()
                    {
                        *outcome = Some(DialogOutcome::AddImages(
                            self.selected.iter().cloned().collect(),
                        ));
                    }
                    if ui.button("Cancel").clicked() {
                        self.open = false;
                    }
                });
            }
            DialogMode::OpenFolder => {
                ui.horizontal(|ui| {
                    if ui.button("Open this folder").clicked() {
                        *outcome = Some(DialogOutcome::OpenFolder(self.current.clone()));
                    }
                    if ui.button("Cancel").clicked() {
                        self.open = false;
                    }
                });
            }
            DialogMode::SaveImage => {
                ui.horizontal(|ui| {
                    ui.label("File name");
                    ui.add(egui::TextEdit::singleline(&mut self.filename).desired_width(260.0));
                    egui::ComboBox::from_id_salt("save-format")
                        .selected_text(self.format.label())
                        .show_ui(ui, |ui| {
                            for format in SaveFormat::ALL {
                                ui.selectable_value(&mut self.format, format, format.label());
                            }
                        });
                });
                self.jpeg_quality(ui);
                ui.horizontal(|ui| {
                    if ui.button("Export").clicked() {
                        match self.save_path() {
                            Ok(path) => {
                                *outcome = Some(DialogOutcome::SaveImage(path, self.quality))
                            }
                            Err(error) => self.error = Some(error),
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        self.open = false;
                    }
                    ui.weak("The original files are never overwritten.");
                });
            }
            DialogMode::SaveFolder => {
                ui.horizontal(|ui| {
                    ui.label("Format");
                    egui::ComboBox::from_id_salt("batch-format")
                        .selected_text(self.format.label())
                        .show_ui(ui, |ui| {
                            for format in SaveFormat::ALL {
                                ui.selectable_value(&mut self.format, format, format.label());
                            }
                        });
                    if ui.button("Export all here").clicked() {
                        *outcome = Some(DialogOutcome::SaveFolder(
                            self.current.clone(),
                            self.format,
                            self.quality,
                        ));
                    }
                    if ui.button("Cancel").clicked() {
                        self.open = false;
                    }
                });
                self.jpeg_quality(ui);
                ui.weak("Each image is written as <name>-edited.<format> next to the originals.");
            }
        }
    }

    fn jpeg_quality(&mut self, ui: &mut egui::Ui) {
        if self.format != SaveFormat::Jpeg {
            return;
        }
        ui.horizontal(|ui| {
            ui.label("JPEG quality");
            ui.add(egui::Slider::new(&mut self.quality, 1..=100));
            ui.weak("Defaults to the source JPEG quality.");
        });
    }

    fn save_path(&self) -> Result<PathBuf, String> {
        let name = self.filename.trim();
        if name.is_empty() {
            return Err("Enter a file name.".to_owned());
        }
        let file_name = Path::new(name)
            .file_name()
            .ok_or_else(|| "The file name is not valid.".to_owned())?;
        let mut path = self.current.join(file_name);
        path.set_extension(self.format.extension());
        Ok(path)
    }

    fn reload(&mut self) {
        self.needs_reload = false;
        self.error = None;
        let mut entries = Vec::new();
        match std::fs::read_dir(&self.current) {
            Ok(reader) => {
                for entry in reader.flatten() {
                    let path = entry.path();
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.starts_with('.') {
                        continue;
                    }
                    let is_dir = path.is_dir();
                    if is_dir || is_supported_image(&path) {
                        entries.push(Entry { path, name, is_dir });
                    }
                }
            }
            Err(error) => {
                self.error = Some(format!(
                    "Could not read {}: {error}",
                    self.current.display()
                ));
            }
        }
        entries.sort_by(|a, b| {
            b.is_dir
                .cmp(&a.is_dir)
                .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
        });
        self.entries = entries;
    }
}

fn file_row(
    ui: &mut egui::Ui,
    is_dir: bool,
    name: &str,
    selected: bool,
    enabled: bool,
) -> egui::Response {
    let height = ui.spacing().interact_size.y.max(20.0);
    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), sense);
    let visuals = ui.visuals().clone();
    let painter = ui.painter();
    if selected {
        painter.rect_filled(rect, 2.0, visuals.selection.bg_fill);
    } else if enabled && response.hovered() {
        painter.rect_filled(rect, 2.0, visuals.widgets.hovered.bg_fill);
    }
    let icon_center = egui::pos2(rect.left() + 11.0, rect.center().y);
    draw_file_icon(painter, icon_center, is_dir);
    let color = if selected {
        visuals.selection.stroke.color
    } else if enabled {
        visuals.text_color()
    } else {
        visuals.weak_text_color()
    };
    painter.text(
        egui::pos2(rect.left() + 24.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        name,
        egui::FontId::proportional(14.0),
        color,
    );
    response
}

/// Draws a small folder or photo icon so rows look like a file browser.
fn draw_file_icon(painter: &egui::Painter, center: egui::Pos2, is_dir: bool) {
    if is_dir {
        let tab = egui::Rect::from_min_size(center + egui::vec2(-7.0, -7.0), egui::vec2(7.0, 4.0));
        let body =
            egui::Rect::from_min_size(center + egui::vec2(-7.0, -4.0), egui::vec2(14.0, 10.0));
        painter.rect_filled(tab, 1.0, egui::Color32::from_rgb(224, 184, 84));
        painter.rect_filled(body, 1.5, egui::Color32::from_rgb(238, 198, 100));
        painter.rect_stroke(
            body,
            1.5,
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(150, 112, 36)),
            egui::StrokeKind::Inside,
        );
    } else {
        let frame = egui::Rect::from_center_size(center, egui::vec2(15.0, 12.0));
        painter.rect_filled(frame, 1.5, egui::Color32::from_rgb(70, 112, 160));
        painter.rect_stroke(
            frame,
            1.5,
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(190, 210, 230)),
            egui::StrokeKind::Inside,
        );
        painter.circle_filled(
            frame.left_top() + egui::vec2(4.0, 3.5),
            1.6,
            egui::Color32::from_rgb(245, 240, 200),
        );
        painter.add(egui::Shape::convex_polygon(
            vec![
                frame.left_bottom() + egui::vec2(1.5, -1.5),
                frame.left_bottom() + egui::vec2(6.0, -6.0),
                frame.right_bottom() + egui::vec2(-1.5, -1.5),
            ],
            egui::Color32::from_rgb(225, 235, 245),
            egui::Stroke::NONE,
        ));
    }
}

pub(crate) fn is_supported_image(path: &Path) -> bool {
    extension_in(path, IMAGE_EXTENSIONS)
}

pub(crate) fn is_raw_image(path: &Path) -> bool {
    extension_in(path, RAW_EXTENSIONS)
}

fn extension_in(path: &Path, extensions: &[&str]) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            let extension = extension.to_ascii_lowercase();
            extensions.contains(&extension.as_str())
        })
        .unwrap_or(false)
}

pub(crate) fn folder_has_raw_images(path: &Path) -> bool {
    std::fs::read_dir(path)
        .map(|reader| {
            reader
                .flatten()
                .any(|entry| entry.path().is_file() && is_raw_image(&entry.path()))
        })
        .unwrap_or(false)
}

/// Collects the supported images of a folder, sorted by name.
pub(crate) fn images_in_folder(path: &Path) -> Vec<PathBuf> {
    let mut images: Vec<PathBuf> = std::fs::read_dir(path)
        .map(|reader| {
            reader
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.is_file() && is_supported_image(path))
                .collect()
        })
        .unwrap_or_default();
    images.sort_by_key(|path| {
        path.file_name()
            .map(|name| name.to_string_lossy().to_lowercase())
            .unwrap_or_default()
    });
    images
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_raw_from_supported_extensions() {
        assert!(is_supported_image(Path::new("scan.JPG")));
        assert!(is_supported_image(Path::new("scan.tiff")));
        assert!(!is_supported_image(Path::new("scan.CR3")));
        assert!(is_raw_image(Path::new("scan.CR3")));
        assert!(is_raw_image(Path::new("scan.nef")));
        assert!(!is_raw_image(Path::new("scan.png")));
    }

    #[test]
    fn finds_raw_files_in_a_folder() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("frame.CR2"), b"raw").unwrap();
        assert!(folder_has_raw_images(dir.path()));
        std::fs::write(dir.path().join("frame.png"), b"png").unwrap();
        assert!(is_supported_image(&dir.path().join("frame.png")));
    }
}
