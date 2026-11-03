//! Application use cases.
//!
//! This layer knows *what* has to happen (load, register, plan canvas,
//! composite, export) and in which order. It never knows *how* a feature is
//! detected, how a file is decoded or how a pixel reaches the GPU. Those are
//! ports implemented by adapters.

pub mod canvas;
pub mod color;
pub mod library;
pub mod stitch_service;

pub use canvas::{plan_canvas, CanvasPlan};
pub use color::estimate_gain;
pub use library::ProjectService;
pub use stitch_service::{StitchService, SystemClock};
