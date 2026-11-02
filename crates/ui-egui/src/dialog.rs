//! Minimal dependency-free file browser and save dialog.
//!
//! A native picker would pull in GTK/portal bindings. This dialog only needs
//! `std::fs`, so the app keeps working on Linux and macOS without extra system
//! libraries.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use egui::{Context, RichText};

pub(crate) const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DialogMode {
    AddImages,
    OpenFolder,
    SaveImage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SaveFormat {
    Png,
    Jpeg,
    Tiff,
    WebP,
    Bmp,
}

impl SaveFormat {
    const ALL: [Self; 5] = [Self::Png, Self::Jpeg, Self::Tiff, Self::WebP, Self::Bmp];

    fn extension(self) -> &'static str {
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
    SaveImage(PathBuf),
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
            error: None,
            needs_reload: true,
        }
    }
}

impl FileDialog {
    pub(crate) fn open_in(&mut self, mode: DialogMode, start: Option<PathBuf>, filename: String) {
        self.mode = mode;
        self.selected.clear();
        self.error = None;
        self.filename = filename;
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
            DialogMode::SaveImage => "Export panorama",
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
                        ui.horizontal(|ui| {
                            if ui
                                .add(egui::Button::new(format!("DIR  {}", entry.name)).frame(false))
                                .clicked()
                            {
                                go_to = Some(entry.path.clone());
                            }
                        });
                    } else if self.mode == DialogMode::AddImages {
                        let selected = self.selected.contains(&entry.path);
                        let response =
                            ui.selectable_label(selected, format!("IMG  {}", entry.name));
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
                        ui.weak(format!("IMG  {}", entry.name));
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
                ui.horizontal(|ui| {
                    if ui.button("Export").clicked() {
                        match self.save_path() {
                            Ok(path) => *outcome = Some(DialogOutcome::SaveImage(path)),
                            Err(error) => self.error = Some(error),
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        self.open = false;
                    }
                    ui.weak("The original files are never overwritten.");
                });
            }
        }
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

pub(crate) fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            let extension = extension.to_ascii_lowercase();
            IMAGE_EXTENSIONS.contains(&extension.as_str())
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
