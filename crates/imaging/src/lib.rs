//! Imaging infrastructure adapters: filesystem repository and CPU compositor.

pub mod cache;
pub mod compositor;
mod jpeg;
mod memory;
mod metadata;
pub mod repository;

pub use cache::CacheDir;
pub use compositor::BilinearCompositor;
pub use memory::{memory_budget, MAX_MEMORY_BUDGET};
pub use repository::FileImageRepository;
