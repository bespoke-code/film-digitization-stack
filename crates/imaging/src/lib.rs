//! Imaging infrastructure adapters: filesystem repository and CPU compositor.

pub mod cache;
pub mod compositor;
mod jpeg;
mod metadata;
pub mod repository;

pub use cache::CacheDir;
pub use compositor::BilinearCompositor;
pub use repository::FileImageRepository;
