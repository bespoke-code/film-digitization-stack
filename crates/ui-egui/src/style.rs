//! Appearance modelled on macOS: 13 pt system text, 22 pt controls, rounded
//! corners, the system accent blue, and light and dark palettes that follow
//! the system setting.

use eframe::egui::{self, Color32, CornerRadius, FontFamily, FontId, Stroke, TextStyle, Theme};

pub(crate) fn apply(ctx: &egui::Context) {
    install_system_font(ctx);
    // Command-plus and Command-minus zoom the photo, not the whole interface.
    ctx.options_mut(|options| options.zoom_with_keyboard = false);
    ctx.set_theme(egui::ThemePreference::System);
    ctx.style_mut_of(Theme::Light, |style| configure(style, &Palette::LIGHT));
    ctx.style_mut_of(Theme::Dark, |style| configure(style, &Palette::DARK));
}

struct Palette {
    dark: bool,
    window: Color32,
    panel: Color32,
    field: Color32,
    control: Color32,
    control_hover: Color32,
    control_active: Color32,
    separator: Color32,
    text: Color32,
    secondary: Color32,
    accent: Color32,
    stripe: Color32,
}

impl Palette {
    const LIGHT: Self = Self {
        dark: false,
        window: Color32::from_rgb(236, 236, 236),
        panel: Color32::from_rgb(246, 246, 246),
        field: Color32::WHITE,
        control: Color32::WHITE,
        control_hover: Color32::from_rgb(245, 245, 245),
        control_active: Color32::from_rgb(222, 222, 222),
        separator: Color32::from_rgb(212, 212, 212),
        text: Color32::from_rgb(29, 29, 31),
        secondary: Color32::from_rgb(110, 110, 115),
        accent: Color32::from_rgb(0, 122, 255),
        stripe: Color32::from_rgb(244, 245, 245),
    };

    const DARK: Self = Self {
        dark: true,
        window: Color32::from_rgb(30, 30, 30),
        panel: Color32::from_rgb(40, 40, 40),
        field: Color32::from_rgb(30, 30, 30),
        control: Color32::from_rgb(70, 70, 72),
        control_hover: Color32::from_rgb(84, 84, 86),
        control_active: Color32::from_rgb(100, 100, 102),
        separator: Color32::from_rgb(58, 58, 58),
        text: Color32::from_rgb(242, 242, 247),
        secondary: Color32::from_rgb(152, 152, 157),
        accent: Color32::from_rgb(10, 132, 255),
        stripe: Color32::from_rgb(46, 46, 46),
    };
}

fn configure(style: &mut egui::Style, palette: &Palette) {
    style.text_styles = [
        (
            TextStyle::Small,
            FontId::new(11.0, FontFamily::Proportional),
        ),
        (TextStyle::Body, FontId::new(13.0, FontFamily::Proportional)),
        (
            TextStyle::Button,
            FontId::new(13.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Heading,
            FontId::new(15.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Monospace,
            FontId::new(12.0, FontFamily::Monospace),
        ),
    ]
    .into();

    let spacing = &mut style.spacing;
    spacing.item_spacing = egui::vec2(8.0, 6.0);
    spacing.button_padding = egui::vec2(10.0, 3.0);
    spacing.interact_size.y = 22.0;
    spacing.window_margin = egui::Margin::same(14);
    spacing.menu_margin = egui::Margin::same(6);
    spacing.slider_width = 150.0;
    spacing.combo_width = 140.0;

    let visuals = &mut style.visuals;
    *visuals = if palette.dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    visuals.window_fill = palette.window;
    visuals.panel_fill = palette.panel;
    visuals.extreme_bg_color = palette.field;
    visuals.text_edit_bg_color = Some(palette.field);
    visuals.faint_bg_color = palette.stripe;
    visuals.override_text_color = None;
    visuals.hyperlink_color = palette.accent;
    visuals.selection.bg_fill = palette.accent;
    visuals.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.window_corner_radius = CornerRadius::same(10);
    visuals.menu_corner_radius = CornerRadius::same(8);
    visuals.window_stroke = Stroke::new(1.0, palette.separator);
    visuals.slider_trailing_fill = true;
    visuals.striped = true;

    let control_stroke = Stroke::new(1.0, palette.separator);
    let widgets = &mut visuals.widgets;
    widgets.noninteractive.bg_fill = palette.panel;
    widgets.noninteractive.weak_bg_fill = palette.panel;
    widgets.noninteractive.bg_stroke = Stroke::new(1.0, palette.separator);
    widgets.noninteractive.fg_stroke = Stroke::new(1.0, palette.text);
    widgets.noninteractive.corner_radius = CornerRadius::same(6);
    for (state, fill) in [
        (&mut widgets.inactive, palette.control),
        (&mut widgets.hovered, palette.control_hover),
        (&mut widgets.active, palette.control_active),
        (&mut widgets.open, palette.control_active),
    ] {
        state.bg_fill = fill;
        state.weak_bg_fill = fill;
        state.bg_stroke = control_stroke;
        state.fg_stroke = Stroke::new(1.0, palette.text);
        state.corner_radius = CornerRadius::same(6);
        state.expansion = 0.0;
    }
    widgets.inactive.fg_stroke = Stroke::new(1.0, palette.text);
    visuals.weak_text_color = Some(palette.secondary);
}

/// Uses San Francisco, the macOS system font, when it is installed. Other
/// systems keep egui's bundled fonts.
fn install_system_font(ctx: &egui::Context) {
    if !cfg!(target_os = "macos") {
        return;
    }
    let Ok(bytes) = std::fs::read("/System/Library/Fonts/SFNS.ttf") else {
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "system".to_owned(),
        egui::FontData::from_owned(bytes).into(),
    );
    if let Some(family) = fonts.families.get_mut(&FontFamily::Proportional) {
        family.insert(0, "system".to_owned());
    }
    ctx.set_fonts(fonts);
}
