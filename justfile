set shell := ["bash", "-uc"]
set positional-arguments

version := `awk -F'"' '/^version = / {print $2; exit}' Cargo.toml`

default: check

# Format all code.
fmt:
    cargo fmt --all

# Check formatting without changing files.
fmt-check:
    cargo fmt --all -- --check

# Lint every target.
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Run all tests.
test:
    cargo test --workspace

# Enforce the real-time preview budgets with release optimization.
perf:
    cargo test --release -p stitch-domain -p stitch-ui-core --lib -- --ignored --test-threads=1 --nocapture

# Build the release binary for the host.
build:
    cargo build --release --locked -p scan-to-img

# Build the release binary for an explicit Rust target triple.
build-target target:
    cargo build --release --locked --target {{ target }} -p scan-to-img

# Run the application.
run:
    cargo run --release -p scan-to-img

# Full release pipeline: deterministic checks and host installers.
release: fmt-check lint test package

# Create the installers and archives for the host in dist/.
package:
    #!/usr/bin/env bash
    set -euo pipefail
    just clean-dist
    case "$(uname -s)" in
        Darwin)
            just package-macos
            ;;
        Linux)
            made=0
            if command -v dpkg-deb >/dev/null 2>&1; then
                just package-deb
                made=1
            fi
            if command -v makepkg >/dev/null 2>&1; then
                just package-arch
                made=1
            fi
            if [ "$made" -eq 0 ]; then
                echo "error: install dpkg-deb (Ubuntu) or makepkg (Arch/Omarchy)" >&2
                exit 1
            fi
            ;;
        *)
            echo "error: unsupported host: $(uname -s)" >&2
            exit 1
            ;;
    esac
    just package-portable
    rm -rf dist/deb dist/arch dist/portable dist/macos
    just checksums

# Build an Ubuntu/Debian package.
package-deb:
    #!/usr/bin/env bash
    set -euo pipefail
    umask 022
    command -v dpkg-deb >/dev/null 2>&1 || { echo "error: dpkg-deb is required" >&2; exit 1; }
    just build
    version="{{ version }}"
    arch="$(dpkg --print-architecture)"
    root="dist/deb/scan-to-img_${version}_${arch}"
    rm -rf "$root"
    install -Dm755 target/release/scan-to-img "$root/usr/bin/scan-to-img"
    mkdir -p "$root/usr/share/applications" "$root/usr/share/icons/hicolor/scalable/apps" "$root/usr/share/doc/scan-to-img" "$root/DEBIAN"
    sed 's|@EXEC@|scan-to-img|g' packaging/scan-to-img.desktop > "$root/usr/share/applications/scan-to-img.desktop"
    install -m644 packaging/scan-to-img.svg "$root/usr/share/icons/hicolor/scalable/apps/scan-to-img.svg"
    install -m644 LICENSE.md "$root/usr/share/doc/scan-to-img/copyright"
    sed -e "s|@VERSION@|$version|g" -e "s|@ARCH@|$arch|g" packaging/debian/control.in > "$root/DEBIAN/control"
    dpkg-deb --build --root-owner-group "$root" "dist/scan-to-img_${version}_${arch}.deb"

# Build an Arch/Omarchy package.
package-arch:
    #!/usr/bin/env bash
    set -euo pipefail
    umask 022
    command -v makepkg >/dev/null 2>&1 || { echo "error: makepkg is required" >&2; exit 1; }
    [ "$(id -u)" -ne 0 ] || { echo "error: makepkg must not run as root" >&2; exit 1; }
    just build
    version="{{ version }}"
    arch="$(uname -m)"
    work="dist/arch/scan-to-img"
    rm -rf "$work"
    mkdir -p "$work"
    cp target/release/scan-to-img "$work/scan-to-img"
    sed 's|@EXEC@|scan-to-img|g' packaging/scan-to-img.desktop > "$work/scan-to-img.desktop"
    cp packaging/scan-to-img.svg "$work/scan-to-img.svg"
    cp LICENSE.md "$work/LICENSE.md"
    sed -e "s|@VERSION@|$version|g" -e "s|@ARCH@|$arch|g" packaging/arch/PKGBUILD.in > "$work/PKGBUILD"
    ( cd "$work" && makepkg --force --nodeps --noconfirm )
    cp "$work"/scan-to-img-${version}-1-${arch}.pkg.tar.* dist/

# Build a macOS application bundle, disk image and zip archive.
package-macos:
    #!/usr/bin/env bash
    set -euo pipefail
    umask 022
    [ "$(uname -s)" = Darwin ] || { echo "error: macOS is required" >&2; exit 1; }
    just build
    version="{{ version }}"
    arch="$(uname -m)"
    app="dist/macos/Scan to image.app"
    rm -rf "dist/macos"
    mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
    cp target/release/scan-to-img "$app/Contents/MacOS/scan-to-img"
    sed "s|@VERSION@|$version|g" packaging/macos/Info.plist.in > "$app/Contents/Info.plist"
    cp LICENSE.md "$app/Contents/Resources/LICENSE.md"
    if [ -n "${CODESIGN_IDENTITY:-}" ]; then
        codesign --force --options runtime --timestamp --sign "$CODESIGN_IDENTITY" "$app"
    else
        codesign --force --sign - "$app"
    fi
    staging="dist/macos/dmg"
    mkdir -p "$staging"
    cp -R "$app" "$staging/"
    ln -s /Applications "$staging/Applications"
    hdiutil create -volname "Scan to image" -srcfolder "$staging" -ov -format UDZO "dist/scan-to-img-${version}-macos-${arch}.dmg"
    ditto -c -k --keepParent "$app" "dist/scan-to-img-${version}-macos-${arch}.zip"

# Build a portable archive with the raw binary.
package-portable:
    #!/usr/bin/env bash
    set -euo pipefail
    umask 022
    just build
    version="{{ version }}"
    case "$(uname -s)" in
        Darwin) os="macos" ;;
        Linux) os="linux" ;;
        *) os="$(uname -s | tr '[:upper:]' '[:lower:]')" ;;
    esac
    arch="$(uname -m)"
    name="scan-to-img-${version}-${os}-${arch}"
    stage="dist/portable/$name"
    rm -rf "$stage"
    mkdir -p "$stage"
    cp target/release/scan-to-img "$stage/"
    cp README.md LICENSE.md "$stage/"
    tar -C "$stage" -czf "dist/$name.tar.gz" .

# Write SHA256SUMS for every release artifact in dist/.
checksums:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p dist
    cd dist
    rm -f SHA256SUMS
    if command -v sha256sum >/dev/null 2>&1; then
        find . -maxdepth 1 -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS
    else
        find . -maxdepth 1 -type f ! -name SHA256SUMS -exec shasum -a 256 {} + | sort -k2 > SHA256SUMS
    fi

# Install for the current user.
install prefix="":
    #!/usr/bin/env bash
    set -euo pipefail
    umask 022
    just build
    prefix="${1:-${PREFIX:-$HOME/.local}}"
    case "$(uname -s)" in
        Darwin)
            app="$HOME/Applications/Scan to image.app"
            rm -rf "$app"
            mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" "$prefix/bin"
            cp target/release/scan-to-img "$app/Contents/MacOS/scan-to-img"
            sed "s|@VERSION@|{{ version }}|g" packaging/macos/Info.plist.in > "$app/Contents/Info.plist"
            cp LICENSE.md "$app/Contents/Resources/LICENSE.md"
            codesign --force --sign - "$app"
            ln -sf "$app/Contents/MacOS/scan-to-img" "$prefix/bin/scan-to-img"
            echo "Installed $app"
            echo "Linked $prefix/bin/scan-to-img"
            ;;
        Linux)
            bin="$prefix/bin"
            install -Dm755 target/release/scan-to-img "$bin/scan-to-img"
            mkdir -p "$prefix/share/applications" "$prefix/share/icons/hicolor/scalable/apps" "$prefix/share/doc/scan-to-img"
            sed "s|@EXEC@|\"$bin/scan-to-img\"|g" packaging/scan-to-img.desktop > "$prefix/share/applications/scan-to-img.desktop"
            install -m644 packaging/scan-to-img.svg "$prefix/share/icons/hicolor/scalable/apps/scan-to-img.svg"
            install -m644 LICENSE.md "$prefix/share/doc/scan-to-img/LICENSE.md"
            command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$prefix/share/applications" >/dev/null 2>&1 || true
            command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -f -t "$prefix/share/icons/hicolor" >/dev/null 2>&1 || true
            echo "Installed $bin/scan-to-img"
            ;;
        *)
            echo "error: unsupported host: $(uname -s)" >&2
            exit 1
            ;;
    esac

# Remove the current user installation.
uninstall prefix="":
    #!/usr/bin/env bash
    set -euo pipefail
    prefix="${1:-${PREFIX:-$HOME/.local}}"
    case "$(uname -s)" in
        Darwin)
            rm -rf "$HOME/Applications/Scan to image.app"
            rm -f "$prefix/bin/scan-to-img"
            ;;
        Linux)
            rm -f "$prefix/bin/scan-to-img"
            rm -f "$prefix/share/applications/scan-to-img.desktop"
            rm -f "$prefix/share/icons/hicolor/scalable/apps/scan-to-img.svg"
            rm -f "$prefix/share/doc/scan-to-img/LICENSE.md"
            rmdir "$prefix/share/doc/scan-to-img" 2>/dev/null || true
            ;;
        *)
            echo "error: unsupported host: $(uname -s)" >&2
            exit 1
            ;;
    esac

# Remove generated release artifacts.
clean-dist:
    rm -rf dist

# Full local pipeline.
check: fmt-check lint test perf
