//! The macOS menu bar. Each item maps to a [`Command`]; standard items use the
//! system implementations.

use std::sync::mpsc::{self, Receiver};
use std::sync::Mutex;

use eframe::egui;
use muda::accelerator::{Accelerator, Code, Modifiers};
use muda::{AboutMetadata, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};

use crate::Command;

pub(crate) struct MenuBar {
    _menu: Menu,
    items: Vec<(Command, MenuItem, bool)>,
    events: Receiver<Command>,
}

impl MenuBar {
    pub(crate) fn install(ctx: &egui::Context) -> Option<Self> {
        let (sender, events) = mpsc::channel();
        let sender = Mutex::new(sender);
        let repaint = ctx.clone();
        MenuEvent::set_event_handler(Some(move |event: MenuEvent| {
            if let Some(command) = Command::from_id(event.id().as_ref()) {
                if let Ok(sender) = sender.lock() {
                    let _ = sender.send(command);
                }
                repaint.request_repaint();
            }
        }));

        let mut items = Vec::new();
        let mut item = |command: Command| {
            let accelerator = shortcut(command);
            let entry = MenuItem::with_id(command.id(), command.menu_title(), true, accelerator);
            items.push((command, entry.clone(), true));
            entry
        };
        let separator = PredefinedMenuItem::separator;

        let app = Submenu::with_items(
            "Scan to Image",
            true,
            &[
                &PredefinedMenuItem::about(
                    Some("About Scan to Image"),
                    Some(AboutMetadata {
                        name: Some("Scan to Image".to_owned()),
                        version: Some(env!("CARGO_PKG_VERSION").to_owned()),
                        ..Default::default()
                    }),
                ),
                &separator(),
                &PredefinedMenuItem::services(None),
                &separator(),
                &PredefinedMenuItem::hide(Some("Hide Scan to Image")),
                &PredefinedMenuItem::hide_others(None),
                &PredefinedMenuItem::show_all(None),
                &separator(),
                &item(Command::Quit),
            ],
        )
        .ok()?;
        let file = Submenu::with_items(
            "File",
            true,
            &[
                &item(Command::AddImages),
                &item(Command::OpenFolder),
                &separator(),
                &PredefinedMenuItem::close_window(None),
                &separator(),
                &item(Command::Export),
                &item(Command::ExportAll),
                &separator(),
                &item(Command::ClearProject),
            ],
        )
        .ok()?;
        let edit = Submenu::with_items(
            "Edit",
            true,
            &[
                &PredefinedMenuItem::cut(None),
                &PredefinedMenuItem::copy(None),
                &PredefinedMenuItem::paste(None),
                &PredefinedMenuItem::select_all(None),
            ],
        )
        .ok()?;
        let image = Submenu::with_items(
            "Image",
            true,
            &[
                &item(Command::Stitch),
                &item(Command::RoughArrange),
                &separator(),
                &item(Command::MoveUp),
                &item(Command::MoveDown),
                &item(Command::RemoveImage),
                &separator(),
                &item(Command::LevelsAndCurves),
                &item(Command::Crop),
            ],
        )
        .ok()?;
        let view = Submenu::with_items(
            "View",
            true,
            &[
                &item(Command::ShowOriginal),
                &item(Command::ShowPanorama),
                &separator(),
                &item(Command::ZoomIn),
                &item(Command::ZoomOut),
                &item(Command::ZoomToFit),
                &separator(),
                &PredefinedMenuItem::fullscreen(None),
            ],
        )
        .ok()?;
        let window = Submenu::with_items(
            "Window",
            true,
            &[
                &PredefinedMenuItem::minimize(None),
                &PredefinedMenuItem::maximize(Some("Zoom")),
                &separator(),
                &PredefinedMenuItem::bring_all_to_front(None),
            ],
        )
        .ok()?;

        let menu = Menu::with_items(&[&app, &file, &edit, &image, &view, &window]).ok()?;
        menu.init_for_nsapp();
        window.set_as_windows_menu_for_nsapp();
        Some(Self {
            _menu: menu,
            items,
            events,
        })
    }

    /// Commands chosen since the last frame, by mouse or key equivalent.
    pub(crate) fn take_commands(&self) -> Vec<Command> {
        self.events.try_iter().collect()
    }

    /// Greys out items that cannot run now. Only changed items reach AppKit.
    pub(crate) fn update(&mut self, enabled: impl Fn(Command) -> bool) {
        for (command, item, current) in &mut self.items {
            let next = enabled(*command);
            if next != *current {
                item.set_enabled(next);
                *current = next;
            }
        }
    }
}

fn shortcut(command: Command) -> Option<Accelerator> {
    let (modifiers, code) = command.shortcut()?;
    let mut mods = Modifiers::META;
    if modifiers.shift {
        mods |= Modifiers::SHIFT;
    }
    if modifiers.alt {
        mods |= Modifiers::ALT;
    }
    let code = match code {
        egui::Key::O => Code::KeyO,
        egui::Key::E => Code::KeyE,
        egui::Key::Q => Code::KeyQ,
        egui::Key::R => Code::KeyR,
        egui::Key::L => Code::KeyL,
        egui::Key::K => Code::KeyK,
        egui::Key::P => Code::KeyP,
        egui::Key::Backspace => Code::Backspace,
        egui::Key::ArrowUp => Code::ArrowUp,
        egui::Key::ArrowDown => Code::ArrowDown,
        egui::Key::Equals => Code::Equal,
        egui::Key::Minus => Code::Minus,
        egui::Key::Num9 => Code::Digit9,
        _ => return None,
    };
    Some(Accelerator::new(mods, code))
}
