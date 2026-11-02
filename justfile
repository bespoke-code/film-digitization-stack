set shell := ["bash", "-uc"]

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

# Build the release binary.
build:
    cargo build --release

# Run the application.
run:
    cargo run --release -p scan-to-img

# Full local pipeline.
check: fmt-check lint test
