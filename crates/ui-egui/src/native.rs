//! macOS system Open and Save panels.

use std::path::{Path, PathBuf};

use crate::dialog::{DialogMode, DialogOutcome, SaveFormat, IMAGE_EXTENSIONS};

/// Runs the system panel for `mode` and returns the user's choice. The panel
/// is application-modal, so the frame resumes after it closes.
pub(crate) fn run(
    mode: DialogMode,
    start: Option<PathBuf>,
    filename: String,
) -> Option<DialogOutcome> {
    let mut panel = rfd::FileDialog::new().set_title(mode.title());
    if let Some(start) = start.filter(|start| start.is_dir()) {
        panel = panel.set_directory(start);
    }
    match mode {
        DialogMode::AddImages => panel
            .add_filter("Images", IMAGE_EXTENSIONS)
            .pick_files()
            .map(DialogOutcome::AddImages),
        DialogMode::OpenFolder => panel.pick_folder().map(DialogOutcome::OpenFolder),
        DialogMode::SaveFolder => panel
            .set_can_create_directories(true)
            .pick_folder()
            .map(DialogOutcome::SaveFolder),
        DialogMode::SaveImage => {
            // The first filter is the one the Format menu starts on.
            let preferred = SaveFormat::from_path(Path::new(&filename)).unwrap_or(SaveFormat::Png);
            let formats = std::iter::once(preferred).chain(
                SaveFormat::ALL
                    .into_iter()
                    .filter(|format| *format != preferred),
            );
            for format in formats {
                panel = panel.add_filter(format.label(), format.extensions());
            }
            panel
                .set_file_name(filename)
                .set_can_create_directories(true)
                .set_format_label("Format:")
                .save_file()
                .map(DialogOutcome::SaveImage)
        }
    }
}
