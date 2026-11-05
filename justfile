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

# Enforce the real-time preview budgets with release optimization.
perf:
    cargo test --release -p stitch-domain -p stitch-ui-core --lib -- --ignored --test-threads=1 --nocapture

# Build the release binary.
build:
    cargo build --release

# Run the application.
run:
    cargo run --release -p scan-to-img

# Full local pipeline.
check: fmt-check lint test perf
