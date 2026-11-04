//! Imaging infrastructure adapters: filesystem repository and CPU compositor.

pub mod compositor;
mod metadata;
pub mod repository;

pub use compositor::BilinearCompositor;
pub use repository::FileImageRepository;
