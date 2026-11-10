# Scan to image

Turn overlapping digital-camera scans of film negatives into one clean positive
image. Everything runs locally in a single native Rust binary: no cloud service,
no OpenCV, no Python and no web view.

## Example

Three overlapping scans of one negative:

![Input frames](docs/example-input.jpg)

Stitched by the app:

![Stitched output](docs/example-output.jpg)

## Install and run

### Packages

`just release` builds the uploadable artifacts for the host in `dist/` and
writes `SHA256SUMS`. Run it on each system and architecture that you publish.
Ubuntu needs `dpkg-deb`; Omarchy/Arch needs `makepkg` from `base-devel`.

| System | Artifact | Install |
| --- | --- | --- |
| Ubuntu/Debian | `scan-to-img_<version>_<arch>.deb` | `sudo apt install ./dist/...deb` |
| Omarchy/Arch | `scan-to-img-<version>-1-<arch>.pkg.tar.zst` | `sudo pacman -U dist/...pkg.tar.zst` |
| macOS | `scan-to-img-<version>-macos-<arch>.dmg` | Open, then drag to Applications |

The same run also writes portable `tar.gz` archives and a macOS `zip`. The
macOS app is ad-hoc signed by default. If Gatekeeper blocks the first launch,
right-click the app and choose Open. Set `CODESIGN_IDENTITY` to sign with a
Developer ID.

### From source

Rust 1.95 or newer is required. Linux needs the usual X11/Wayland development
packages and a working Vulkan or OpenGL driver. macOS uses Metal.

```sh
git clone <repository-url>
cd scan-to-img
just install
```

`just install` puts the binary in `~/.local/bin` and adds a desktop entry on
Linux. On macOS it installs `Scan to image.app` in `~/Applications` and links
the binary into `~/.local/bin`. Pass a prefix to override the binary location:
`just install /usr/local`.

Without [`just`](https://github.com/casey/just):

```sh
cargo build --release
./target/release/scan-to-img
```

Pass image paths to open them at launch:

```sh
scan-to-img scans/*.jpg
```

## Use

1. Open a folder, add up to 16 images, or drag files onto the window.
2. Put overlapping frames in order and press **Stitch panorama**. Use the rotate
   controls in the Develop pane when a scan is sideways.
3. If a pair has no usable detail, press **Rough arrange…**, then drag, rotate or
   scale the frames and apply. Row and Column give a fresh starting layout.
4. Inspect the result in the zoomable preview and rotate the panorama if needed.
   Use **Crop** to drag a selection, move or resize it, then press **Apply crop**.
5. Existing EXIF details appear automatically. Add or change camera, lens, film and
   exposure details in the **Metadata** pane. PNG, JPEG and WebP exports embed them
   as EXIF; TIFF and BMP do not.
6. Export as PNG, JPEG, TIFF, WebP or BMP. JPEG quality defaults to the source
   JPEG quality (the lowest source in a panorama) and can be changed in the
   export dialog.

Source files are never modified.

Exports are written atomically, so an interrupted save never leaves a partial
image. The app keeps temporary files in an app-owned directory that is removed
when the app exits or a panic unwinds. At startup it removes only its own stale
cache directories from earlier crashes, leaving other applications' files alone.

RAW files are not decoded yet. Convert them to TIFF, JPEG or PNG before
opening them in the app.

## Development

```sh
just check    # formatting, clippy, tests and real-time preview budgets
just perf     # performance tests only
just package  # installers and archives in dist/
just release  # formatting, clippy, tests and package
just build    # release binary
```

The workspace keeps the domain and UI-independent state free of GUI, GPU,
filesystem and vision dependencies so the UI and renderer can be replaced.
