//! Test doubles and contract suites.
//!
//! Contract suites describe the observable behaviour every adapter of a port
//! must provide. They are run against the production adapters *and* against the
//! in-memory doubles in this crate, which is exactly what keeps the hexagonal
//! boundaries honest: swapping an adapter must not change the contract.

pub mod contracts;
pub mod doubles;
pub mod fixtures;

pub use doubles::{
    FakeClock, InMemoryCompositor, InMemoryFeatureExtractor, InMemoryHomographyEstimator,
    InMemoryImageRepository, InMemoryMatcher, RecordingRenderer, ScriptedCompositor,
    ScriptedEstimator, ScriptedExtractor, ScriptedMatcher,
};
