//! File panels modelled on the macOS Open and Save panels.
//!
//! macOS shows the system panels (see `native`). Elsewhere this dependency-free
//! panel takes their place: a sidebar of places, a path menu with back and
//! forward, a sortable Name, Date Modified, Size and Kind list, and Cancel and
//! the default button in the bottom-right corner. It only needs `std::fs`, so
//! Linux needs no GTK or portal libraries.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::SystemTime;

use egui::{Align2, Color32, Context, FontId, RichText, Sense, Stroke};

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

impl DialogMode {
    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::AddImages => "Add Images",
            Self::OpenFolder => "Open Folder",
            Self::SaveImage => "Export Image",
            Self::SaveFolder => "Export All Images",
        }
    }

    fn default_button(self) -> &'static str {
        match self {
            Self::AddImages => "Add",
            Self::OpenFolder => "Open",
            Self::SaveImage | Self::SaveFolder => "Export",
        }
    }

    fn picks_folder(self) -> bool {
        matches!(self, Self::OpenFolder | Self::SaveFolder)
    }

    fn saves(self) -> bool {
        matches!(self, Self::SaveImage | Self::SaveFolder)
    }
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
    pub(crate) const ALL: [Self; 5] = [Self::Png, Self::Jpeg, Self::Tiff, Self::WebP, Self::Bmp];

    pub(crate) fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Tiff => "tiff",
            Self::WebP => "webp",
            Self::Bmp => "bmp",
        }
    }

    /// Extensions accepted for this format, preferred one first.
    pub(crate) fn extensions(self) -> &'static [&'static str] {
        match self {
            Self::Png => &["png"],
            Self::Jpeg => &["jpg", "jpeg"],
            Self::Tiff => &["tiff", "tif"],
            Self::WebP => &["webp"],
            Self::Bmp => &["bmp"],
        }
    }

    pub(crate) fn from_path(path: &Path) -> Option<Self> {
        let extension = path.extension()?.to_str()?.to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|format| format.extensions().contains(&extension.as_str()))
    }

    pub(crate) fn label(self) -> &'static str {
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
    SaveFolder(PathBuf),
}

/// Replaces the extension of `name` with the preferred one of `format`.
pub(crate) fn with_format_extension(name: &str, format: SaveFormat) -> String {
    let path = Path::new(name.trim());
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    format!("{stem}.{}", format.extension())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SortColumn {
    Name,
    Modified,
    Size,
    Kind,
}

struct Entry {
    path: PathBuf,
    name: String,
    is_dir: bool,
    enabled: bool,
    modified: Option<SystemTime>,
    size: u64,
    kind: String,
}

struct Place {
    name: String,
    path: PathBuf,
}

pub(crate) struct FileDialog {
    open: bool,
    mode: DialogMode,
    current: PathBuf,
    back: Vec<PathBuf>,
    forward: Vec<PathBuf>,
    entries: Vec<Entry>,
    selected: BTreeSet<PathBuf>,
    anchor: Option<usize>,
    filename: String,
    format: SaveFormat,
    search: String,
    sort: SortColumn,
    ascending: bool,
    new_folder: Option<String>,
    error: Option<String>,
    needs_reload: bool,
    favorites: Vec<Place>,
    locations: Vec<Place>,
}

impl Default for FileDialog {
    fn default() -> Self {
        Self {
            open: false,
            mode: DialogMode::AddImages,
            current: std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            back: Vec::new(),
            forward: Vec::new(),
            entries: Vec::new(),
            selected: BTreeSet::new(),
            anchor: None,
            filename: String::new(),
            format: SaveFormat::Png,
            search: String::new(),
            sort: SortColumn::Name,
            ascending: true,
            new_folder: None,
            error: None,
            needs_reload: true,
            favorites: Vec::new(),
            locations: Vec::new(),
        }
    }
}

const PANEL_SIZE: egui::Vec2 = egui::vec2(760.0, 470.0);
const SIDEBAR_WIDTH: f32 = 160.0;
const ROW_HEIGHT: f32 = 22.0;

impl FileDialog {
    pub(crate) fn open_in(&mut self, mode: DialogMode, start: Option<PathBuf>, filename: String) {
        self.mode = mode;
        self.selected.clear();
        self.anchor = None;
        self.search.clear();
        self.new_folder = None;
        self.error = None;
        self.format = SaveFormat::from_path(Path::new(&filename)).unwrap_or(SaveFormat::Png);
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
            if start != self.current {
                self.current = start;
                self.back.clear();
                self.forward.clear();
            }
        }
        self.favorites = favorite_places();
        self.locations = location_places();
        self.open = true;
        self.needs_reload = true;
    }

    pub(crate) fn is_open(&self) -> bool {
        self.open
    }

    pub(crate) fn show(&mut self, ctx: &Context) -> Option<DialogOutcome> {
        if !self.open {
            return None;
        }
        if self.needs_reload {
            self.reload();
        }

        let mut outcome = None;
        let mut cancel = false;
        let response = egui::Modal::new(egui::Id::unique("file-panel")).show(ctx, |ui| {
            ui.set_width(PANEL_SIZE.x);
            ui.set_height(PANEL_SIZE.y);
            self.draw(ui, &mut outcome, &mut cancel);
        });
        if (response.should_close() && self.new_folder.is_none()) || cancel || outcome.is_some() {
            self.open = false;
        }
        outcome
    }

    fn draw(&mut self, ui: &mut egui::Ui, outcome: &mut Option<DialogOutcome>, cancel: &mut bool) {
        ui.vertical_centered(|ui| {
            ui.label(RichText::new(self.mode.title()).strong());
        });
        if self.mode == DialogMode::SaveImage {
            self.save_as_row(ui);
        }
        ui.add_space(4.0);
        self.navigation_bar(ui);
        ui.add_space(4.0);

        let footer_height = if self.mode == DialogMode::SaveImage {
            78.0
        } else {
            40.0
        };
        let body_height = (ui.available_height() - footer_height).max(120.0);
        ui.horizontal(|ui| {
            ui.set_height(body_height);
            self.sidebar(ui, body_height);
            ui.separator();
            ui.vertical(|ui| {
                ui.set_height(body_height);
                self.file_list(ui, outcome);
            });
        });

        ui.separator();
        if self.mode == DialogMode::SaveImage {
            ui.horizontal(|ui| {
                ui.add_space((ui.available_width() - 200.0).max(0.0) / 2.0);
                ui.label("Format:");
                let before = self.format;
                egui::ComboBox::from_id_salt("panel-format")
                    .width(120.0)
                    .selected_text(self.format.label())
                    .show_ui(ui, |ui| {
                        for format in SaveFormat::ALL {
                            ui.selectable_value(&mut self.format, format, format.label());
                        }
                    });
                if self.format != before {
                    self.filename = with_format_extension(&self.filename, self.format);
                }
            });
            ui.add_space(2.0);
        }
        self.footer(ui, outcome, cancel);
    }

    fn save_as_row(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("panel-save-as")
            .num_columns(2)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label("Save As:");
                });
                let field =
                    ui.add(egui::TextEdit::singleline(&mut self.filename).desired_width(320.0));
                if field.changed() {
                    if let Some(format) = SaveFormat::from_path(Path::new(&self.filename)) {
                        self.format = format;
                    }
                }
                ui.end_row();
            });
    }

    fn navigation_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !self.back.is_empty(),
                    egui::Button::new("‹").min_size([26.0, 22.0].into()),
                )
                .on_hover_text("Back")
                .clicked()
            {
                if let Some(previous) = self.back.pop() {
                    self.forward
                        .push(std::mem::replace(&mut self.current, previous));
                    self.after_navigation();
                }
            }
            if ui
                .add_enabled(
                    !self.forward.is_empty(),
                    egui::Button::new("›").min_size([26.0, 22.0].into()),
                )
                .on_hover_text("Forward")
                .clicked()
            {
                if let Some(next) = self.forward.pop() {
                    self.back.push(std::mem::replace(&mut self.current, next));
                    self.after_navigation();
                }
            }

            let mut go_to = None;
            egui::ComboBox::from_id_salt("panel-path")
                .width(220.0)
                .selected_text(display_name(&self.current))
                .show_ui(ui, |ui| {
                    for ancestor in self.current.ancestors() {
                        if ui
                            .selectable_label(ancestor == self.current, display_name(ancestor))
                            .clicked()
                        {
                            go_to = Some(ancestor.to_path_buf());
                        }
                    }
                });
            if let Some(path) = go_to {
                self.navigate(path);
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.search)
                        .hint_text("Search")
                        .desired_width(170.0),
                );
            });
        });
    }

    fn sidebar(&mut self, ui: &mut egui::Ui, height: f32) {
        let mut go_to = None;
        ui.vertical(|ui| {
            ui.set_width(SIDEBAR_WIDTH);
            ui.set_height(height);
            egui::ScrollArea::vertical()
                .id_salt("panel-sidebar")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for (title, places) in [
                        ("Favorites", &self.favorites),
                        ("Locations", &self.locations),
                    ] {
                        if places.is_empty() {
                            continue;
                        }
                        ui.add_space(4.0);
                        ui.label(RichText::new(title).small().weak().strong());
                        for place in places {
                            let active = place.path == self.current;
                            if sidebar_row(ui, &place.name, active).clicked() {
                                go_to = Some(place.path.clone());
                            }
                        }
                    }
                });
        });
        if let Some(path) = go_to {
            self.navigate(path);
        }
    }

    fn file_list(&mut self, ui: &mut egui::Ui, outcome: &mut Option<DialogOutcome>) {
        if let Some(error) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        let widths = column_widths(ui.available_width());
        self.header(ui, widths);

        let needle = self.search.trim().to_lowercase();
        let visible: Vec<usize> = (0..self.entries.len())
            .filter(|index| {
                needle.is_empty() || self.entries[*index].name.to_lowercase().contains(&needle)
            })
            .collect();

        let mut go_to = None;
        let mut confirm = None;
        let modifiers = ui.input(|input| input.modifiers);
        egui::ScrollArea::vertical()
            .id_salt("panel-list")
            .auto_shrink([false, false])
            .show_rows(ui, ROW_HEIGHT, visible.len(), |ui, range| {
                for row in range {
                    let index = visible[row];
                    let entry = &self.entries[index];
                    let selected = self.selected.contains(&entry.path);
                    let response = entry_row(ui, entry, selected, row % 2 == 1, widths);
                    if !entry.enabled {
                        continue;
                    }
                    if response.double_clicked() {
                        if entry.is_dir {
                            go_to = Some(entry.path.clone());
                        } else {
                            confirm = Some(entry.path.clone());
                        }
                    } else if response.clicked() {
                        let path = entry.path.clone();
                        let multiple = self.mode == DialogMode::AddImages;
                        if multiple && modifiers.command {
                            if !self.selected.remove(&path) {
                                self.selected.insert(path);
                            }
                            self.anchor = Some(row);
                        } else if multiple && modifiers.shift && self.anchor.is_some() {
                            let anchor = self.anchor.unwrap_or(row);
                            let (low, high) = (anchor.min(row), anchor.max(row));
                            self.selected.clear();
                            for &other in &visible[low..=high] {
                                let entry = &self.entries[other];
                                if entry.enabled && !entry.is_dir {
                                    self.selected.insert(entry.path.clone());
                                }
                            }
                        } else {
                            self.selected.clear();
                            self.selected.insert(path);
                            self.anchor = Some(row);
                        }
                        if self.mode == DialogMode::SaveImage && !entry.is_dir {
                            self.filename = with_format_extension(&entry.name, self.format);
                        }
                    }
                }
                if visible.is_empty() {
                    ui.add_space(24.0);
                    ui.vertical_centered(|ui| {
                        ui.weak(if needle.is_empty() {
                            "This folder is empty."
                        } else {
                            "No matching items."
                        });
                    });
                }
            });

        if let Some(path) = go_to {
            self.navigate(path);
        } else if let Some(path) = confirm {
            match self.mode {
                DialogMode::AddImages => *outcome = Some(DialogOutcome::AddImages(vec![path])),
                DialogMode::SaveImage => {
                    if let Some(name) = path.file_name() {
                        self.filename = name.to_string_lossy().into_owned();
                        self.confirm(outcome);
                    }
                }
                DialogMode::OpenFolder | DialogMode::SaveFolder => {}
            }
        }
    }

    fn header(&mut self, ui: &mut egui::Ui, widths: [f32; 4]) {
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW_HEIGHT), Sense::hover());
        let mut x = rect.left();
        for (column, label, width) in [
            (SortColumn::Name, "Name", widths[0]),
            (SortColumn::Modified, "Date Modified", widths[1]),
            (SortColumn::Size, "Size", widths[2]),
            (SortColumn::Kind, "Kind", widths[3]),
        ] {
            let cell =
                egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2(width, ROW_HEIGHT));
            let response = ui.interact(
                cell,
                ui.scope_id().with(("panel-sort", label)),
                Sense::click(),
            );
            if response.clicked() {
                if self.sort == column {
                    self.ascending = !self.ascending;
                } else {
                    self.sort = column;
                    self.ascending = true;
                }
                self.sort_entries();
            }
            let arrow = if self.sort == column {
                if self.ascending {
                    "  ⌃"
                } else {
                    "  ⌄"
                }
            } else {
                ""
            };
            ui.painter().text(
                egui::pos2(cell.left() + 8.0, cell.center().y),
                Align2::LEFT_CENTER,
                format!("{label}{arrow}"),
                FontId::proportional(11.5),
                ui.visuals().weak_text_color(),
            );
            if x > rect.left() {
                ui.painter().vline(
                    x,
                    cell.y_range().shrink(4.0),
                    Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color),
                );
            }
            x += width;
        }
        ui.painter().hline(
            rect.x_range(),
            rect.bottom(),
            Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color),
        );
    }

    fn footer(
        &mut self,
        ui: &mut egui::Ui,
        outcome: &mut Option<DialogOutcome>,
        cancel: &mut bool,
    ) {
        ui.horizontal(|ui| {
            if self.mode.saves() {
                if let Some(name) = self.new_folder.as_mut() {
                    let field = ui.add(egui::TextEdit::singleline(name).desired_width(160.0));
                    let submitted =
                        field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
                    if ui.button("Create").clicked() || submitted {
                        self.create_folder();
                    } else if ui.button("Cancel").clicked() {
                        self.new_folder = None;
                    } else {
                        field.request_focus();
                    }
                } else if ui.button("New Folder").clicked() {
                    self.new_folder = Some("untitled folder".to_owned());
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let enabled = self.can_confirm();
                let primary = egui::Button::new(
                    RichText::new(self.mode.default_button()).color(Color32::WHITE),
                )
                .fill(ui.visuals().selection.bg_fill)
                .min_size([82.0, 24.0].into());
                let clicked = ui.add_enabled(enabled, primary).clicked();
                let enter = self.new_folder.is_none()
                    && ui.input(|input| input.key_pressed(egui::Key::Enter));
                if enabled && (clicked || enter) {
                    self.confirm(outcome);
                }
                if ui
                    .add(egui::Button::new("Cancel").min_size([82.0, 24.0].into()))
                    .clicked()
                {
                    *cancel = true;
                }
                if self.mode == DialogMode::SaveFolder {
                    ui.weak("Each image is saved as <name>-edited next to its original.");
                }
            });
        });
    }

    fn can_confirm(&self) -> bool {
        match self.mode {
            DialogMode::AddImages => !self.selected.is_empty(),
            DialogMode::OpenFolder | DialogMode::SaveFolder => true,
            DialogMode::SaveImage => !self.filename.trim().is_empty(),
        }
    }

    fn confirm(&mut self, outcome: &mut Option<DialogOutcome>) {
        match self.mode {
            DialogMode::AddImages => {
                let files: Vec<PathBuf> = self
                    .selected
                    .iter()
                    .filter(|path| !path.is_dir())
                    .cloned()
                    .collect();
                if let Some(folder) = self.selected.iter().find(|path| path.is_dir()).cloned() {
                    self.navigate(folder);
                } else if !files.is_empty() {
                    *outcome = Some(DialogOutcome::AddImages(files));
                }
            }
            DialogMode::OpenFolder => {
                *outcome = Some(DialogOutcome::OpenFolder(self.chosen_folder()));
            }
            DialogMode::SaveFolder => {
                *outcome = Some(DialogOutcome::SaveFolder(self.chosen_folder()));
            }
            DialogMode::SaveImage => match self.save_path() {
                Ok(path) => *outcome = Some(DialogOutcome::SaveImage(path)),
                Err(error) => self.error = Some(error),
            },
        }
    }

    /// The highlighted folder, or the open one when nothing is highlighted.
    fn chosen_folder(&self) -> PathBuf {
        self.selected
            .iter()
            .find(|path| path.is_dir())
            .cloned()
            .unwrap_or_else(|| self.current.clone())
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
        if SaveFormat::from_path(&path) != Some(self.format) {
            path.set_extension(self.format.extension());
        }
        Ok(path)
    }

    fn create_folder(&mut self) {
        let Some(name) = self.new_folder.take() else {
            return;
        };
        let name = name.trim();
        if name.is_empty() || Path::new(name).file_name().is_none() {
            self.error = Some("The folder name is not valid.".to_owned());
            return;
        }
        let path = self.current.join(name);
        match std::fs::create_dir(&path) {
            Ok(()) => self.navigate(path),
            Err(error) => self.error = Some(format!("Could not create {name}: {error}")),
        }
    }

    fn navigate(&mut self, path: PathBuf) {
        if path == self.current {
            return;
        }
        self.back.push(std::mem::replace(&mut self.current, path));
        self.forward.clear();
        self.after_navigation();
    }

    fn after_navigation(&mut self) {
        self.selected.clear();
        self.anchor = None;
        self.search.clear();
        self.needs_reload = true;
    }

    fn reload(&mut self) {
        self.needs_reload = false;
        self.error = None;
        let mut entries = Vec::new();
        match std::fs::read_dir(&self.current) {
            Ok(reader) => {
                for entry in reader.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.starts_with('.') {
                        continue;
                    }
                    let path = entry.path();
                    let metadata = std::fs::metadata(&path).ok();
                    let is_dir = metadata.as_ref().is_some_and(|metadata| metadata.is_dir());
                    let supported = is_supported_image(&path);
                    let enabled = is_dir
                        || (!self.mode.picks_folder()
                            && (supported || self.mode == DialogMode::SaveImage));
                    entries.push(Entry {
                        kind: kind_label(&path, is_dir),
                        modified: metadata
                            .as_ref()
                            .and_then(|metadata| metadata.modified().ok()),
                        size: metadata.as_ref().map_or(0, |metadata| metadata.len()),
                        path,
                        name,
                        is_dir,
                        enabled,
                    });
                }
            }
            Err(error) => {
                self.error = Some(format!(
                    "Could not read {}: {error}",
                    self.current.display()
                ));
            }
        }
        self.entries = entries;
        self.sort_entries();
    }

    fn sort_entries(&mut self) {
        let column = self.sort;
        let ascending = self.ascending;
        self.entries.sort_by(|a, b| {
            let order = match column {
                SortColumn::Name => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                SortColumn::Modified => a.modified.cmp(&b.modified),
                SortColumn::Size => (!a.is_dir, a.size).cmp(&(!b.is_dir, b.size)),
                SortColumn::Kind => a.kind.cmp(&b.kind),
            };
            let order = if ascending { order } else { order.reverse() };
            order.then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
        });
    }
}

fn column_widths(total: f32) -> [f32; 4] {
    let date = 150.0;
    let size = 80.0;
    let kind = 100.0;
    [(total - date - size - kind).max(140.0), date, size, kind]
}

fn sidebar_row(ui: &mut egui::Ui, name: &str, active: bool) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW_HEIGHT), Sense::click());
    let visuals = ui.visuals();
    if active {
        ui.painter()
            .rect_filled(rect, 5.0, visuals.widgets.active.weak_bg_fill);
    } else if response.hovered() {
        ui.painter()
            .rect_filled(rect, 5.0, visuals.widgets.hovered.weak_bg_fill);
    }
    draw_file_icon(
        ui.painter(),
        egui::pos2(rect.left() + 13.0, rect.center().y),
        true,
    );
    ui.painter().text(
        egui::pos2(rect.left() + 28.0, rect.center().y),
        Align2::LEFT_CENTER,
        name,
        FontId::proportional(13.0),
        visuals.text_color(),
    );
    response
}

fn entry_row(
    ui: &mut egui::Ui,
    entry: &Entry,
    selected: bool,
    striped: bool,
    widths: [f32; 4],
) -> egui::Response {
    let sense = if entry.enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW_HEIGHT), sense);
    let visuals = ui.visuals();
    let painter = ui.painter_at(rect);
    if selected {
        painter.rect_filled(rect, 5.0, visuals.selection.bg_fill);
    } else if striped {
        painter.rect_filled(rect, 0.0, visuals.faint_bg_color);
    }
    let color = if selected {
        Color32::WHITE
    } else if entry.enabled {
        visuals.text_color()
    } else {
        visuals.weak_text_color().gamma_multiply(0.7)
    };
    let secondary = if selected {
        Color32::WHITE
    } else {
        visuals.weak_text_color()
    };
    draw_file_icon(
        &painter,
        egui::pos2(rect.left() + 14.0, rect.center().y),
        entry.is_dir,
    );

    let font = FontId::proportional(13.0);
    let mut x = rect.left();
    let cells = [
        (entry.name.clone(), color, widths[0], 28.0),
        (
            entry
                .modified
                .map(format_date)
                .unwrap_or_else(|| "--".to_owned()),
            secondary,
            widths[1],
            8.0,
        ),
        (
            if entry.is_dir {
                "--".to_owned()
            } else {
                format_size(entry.size)
            },
            secondary,
            widths[2],
            8.0,
        ),
        (entry.kind.clone(), secondary, widths[3], 8.0),
    ];
    for (text, color, width, inset) in cells {
        let cell =
            egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2(width, ROW_HEIGHT));
        let galley = painter.layout(text, font.clone(), color, f32::INFINITY);
        let clip = cell.shrink2(egui::vec2(4.0, 0.0));
        painter.with_clip_rect(clip).galley(
            egui::pos2(cell.left() + inset, cell.center().y - galley.size().y / 2.0),
            galley,
            color,
        );
        x += width;
    }
    response
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

fn kind_label(path: &Path, is_dir: bool) -> String {
    if is_dir {
        return "Folder".to_owned();
    }
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) => {
            let upper = extension.to_ascii_uppercase();
            if is_supported_image(path) || is_raw_image(path) {
                format!("{upper} image")
            } else {
                format!("{upper} file")
            }
        }
        None => "Document".to_owned(),
    }
}

fn favorite_places() -> Vec<Place> {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return Vec::new();
    };
    let mut places = vec![Place {
        name: display_name(&home),
        path: home.clone(),
    }];
    for folder in ["Desktop", "Documents", "Downloads", "Pictures"] {
        let path = home.join(folder);
        if path.is_dir() {
            places.push(Place {
                name: folder.to_owned(),
                path,
            });
        }
    }
    places
}

fn location_places() -> Vec<Place> {
    let mut places = vec![Place {
        name: "Computer".to_owned(),
        path: PathBuf::from("/"),
    }];
    let user = std::env::var("USER").unwrap_or_default();
    let roots = [
        PathBuf::from("/media").join(&user),
        PathBuf::from("/run/media").join(&user),
        PathBuf::from("/mnt"),
        PathBuf::from("/Volumes"),
    ];
    for root in roots {
        let Ok(reader) = std::fs::read_dir(&root) else {
            continue;
        };
        for entry in reader.flatten() {
            let path = entry.path();
            if path.is_dir() {
                places.push(Place {
                    name: display_name(&path),
                    path,
                });
            }
        }
    }
    places
}

/// Finder-style decimal size, such as "10.2 MB".
fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    if bytes < 1000 {
        return format!("{bytes} bytes");
    }
    let mut value = bytes as f64 / 1000.0;
    let mut unit = 0;
    while value >= 999.95 && unit + 1 < UNITS.len() {
        value /= 1000.0;
        unit += 1;
    }
    if value < 10.0 {
        format!("{value:.1} {}", UNITS[unit])
    } else {
        format!("{value:.0} {}", UNITS[unit])
    }
}

fn format_date(time: SystemTime) -> String {
    let seconds = match time.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(duration) => duration.as_secs() as i64,
        Err(_) => return "--".to_owned(),
    };
    format_local_seconds(seconds + utc_offset_seconds())
}

/// Formats seconds since the epoch, already shifted to local time, the way
/// Finder lists dates: "18 Nov 2026 at 12:24".
fn format_local_seconds(seconds: i64) -> String {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let days = seconds.div_euclid(86_400);
    let of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{day} {} {year} at {:02}:{:02}",
        MONTHS[(month - 1) as usize],
        of_day / 3600,
        (of_day % 3600) / 60
    )
}

/// Days since 1970-01-01 to a proleptic Gregorian date (H. Hinnant's method).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// The local UTC offset, read once from `date`. The standard library has no
/// time zone support and the panel only needs it for display.
fn utc_offset_seconds() -> i64 {
    static OFFSET: OnceLock<i64> = OnceLock::new();
    *OFFSET.get_or_init(|| {
        std::process::Command::new("date")
            .arg("+%z")
            .output()
            .ok()
            .and_then(|output| parse_utc_offset(String::from_utf8_lossy(&output.stdout).trim()))
            .unwrap_or(0)
    })
}

fn parse_utc_offset(text: &str) -> Option<i64> {
    let (sign, digits) = match text.as_bytes().first()? {
        b'+' => (1, &text[1..]),
        b'-' => (-1, &text[1..]),
        _ => return None,
    };
    if digits.len() != 4 {
        return None;
    }
    let hours: i64 = digits[..2].parse().ok()?;
    let minutes: i64 = digits[2..].parse().ok()?;
    Some(sign * (hours * 3600 + minutes * 60))
}

/// Draws a small folder or photo icon so rows look like a file browser.
fn draw_file_icon(painter: &egui::Painter, center: egui::Pos2, is_dir: bool) {
    if is_dir {
        let tab = egui::Rect::from_min_size(center + egui::vec2(-7.0, -6.0), egui::vec2(6.0, 3.0));
        let body =
            egui::Rect::from_min_size(center + egui::vec2(-7.0, -4.0), egui::vec2(14.0, 10.0));
        painter.rect_filled(tab, 1.0, Color32::from_rgb(84, 170, 240));
        painter.rect_filled(body, 2.0, Color32::from_rgb(110, 188, 248));
    } else {
        let frame = egui::Rect::from_center_size(center, egui::vec2(12.0, 15.0));
        painter.rect_filled(frame, 1.5, Color32::from_rgb(246, 246, 246));
        painter.rect_stroke(
            frame,
            1.5,
            Stroke::new(1.0_f32, Color32::from_rgb(170, 170, 170)),
            egui::StrokeKind::Inside,
        );
        let photo = frame.shrink(2.5);
        painter.add(egui::Shape::convex_polygon(
            vec![
                photo.left_bottom(),
                photo.left_bottom() + egui::vec2(3.0, -4.0),
                photo.right_bottom(),
            ],
            Color32::from_rgb(90, 160, 110),
            Stroke::NONE,
        ));
        painter.circle_filled(
            photo.right_top() + egui::vec2(-2.0, 2.0),
            1.4,
            Color32::from_rgb(240, 190, 60),
        );
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
    fn formats_follow_the_extension() {
        assert_eq!(
            SaveFormat::from_path(Path::new("a.JPEG")),
            Some(SaveFormat::Jpeg)
        );
        assert_eq!(
            SaveFormat::from_path(Path::new("a.tif")),
            Some(SaveFormat::Tiff)
        );
        assert_eq!(SaveFormat::from_path(Path::new("a.cr3")), None);
        assert_eq!(
            with_format_extension("scan-stitched.png", SaveFormat::Jpeg),
            "scan-stitched.jpg"
        );
        assert_eq!(with_format_extension("scan", SaveFormat::WebP), "scan.webp");
    }

    #[test]
    fn sizes_and_dates_read_like_finder() {
        assert_eq!(format_size(512), "512 bytes");
        assert_eq!(format_size(10_153_071), "10 MB");
        assert_eq!(format_size(2_500_000), "2.5 MB");
        assert_eq!(format_local_seconds(0), "1 Jan 1970 at 00:00");
        assert_eq!(format_local_seconds(1_795_609_440), "25 Nov 2026 at 12:24");
        assert_eq!(parse_utc_offset("+0130"), Some(5400));
        assert_eq!(parse_utc_offset("-0800"), Some(-28_800));
        assert_eq!(parse_utc_offset("UTC"), None);
    }

    #[test]
    fn save_path_keeps_or_replaces_the_extension() {
        let mut dialog = FileDialog {
            current: PathBuf::from("/tmp"),
            filename: "scan.jpeg".to_owned(),
            format: SaveFormat::Jpeg,
            ..FileDialog::default()
        };
        assert_eq!(dialog.save_path().unwrap(), PathBuf::from("/tmp/scan.jpeg"));
        dialog.format = SaveFormat::Png;
        assert_eq!(dialog.save_path().unwrap(), PathBuf::from("/tmp/scan.png"));
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
