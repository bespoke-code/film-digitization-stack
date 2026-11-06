//! Process-owned temporary directory with panic-safe cleanup.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};

static CACHE_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// A temporary cache directory owned by the running process.
///
/// The directory and everything inside it are removed when the value is
/// dropped. A panic hook also removes it so an unhandled error does not leave
/// preview data behind.
#[derive(Debug)]
pub struct CacheDir {
    path: PathBuf,
}

impl CacheDir {
    pub fn create() -> std::io::Result<Self> {
        install_panic_cleanup();
        let base = std::env::temp_dir();
        let pid = std::process::id();
        for attempt in 0..1000 {
            let path = base.join(format!("scan-to-img-{pid}-{attempt}"));
            match std::fs::create_dir(&path) {
                Ok(()) => {
                    if let Ok(mut slot) = CACHE_PATH.lock() {
                        *slot = Some(path.clone());
                    }
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not create a unique cache directory",
        ))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for CacheDir {
    fn drop(&mut self) {
        remove_cache_dir(&self.path);
    }
}

fn install_panic_cleanup() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Ok(mut slot) = CACHE_PATH.lock() {
                if let Some(path) = slot.take() {
                    let _ = std::fs::remove_dir_all(path);
                }
            }
            previous(info);
        }));
    });
}

fn remove_cache_dir(path: &Path) {
    let _ = std::fs::remove_dir_all(path);
    if let Ok(mut slot) = CACHE_PATH.lock() {
        if slot.as_deref() == Some(path) {
            *slot = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_the_directory_on_drop_and_panic() {
        let dir = CacheDir::create().unwrap();
        let path = dir.path().to_path_buf();
        std::fs::write(path.join("preview.bin"), b"cached").unwrap();
        drop(dir);
        assert!(!path.exists());

        let dir = CacheDir::create().unwrap();
        let path = dir.path().to_path_buf();
        std::fs::write(path.join("preview.bin"), b"cached").unwrap();
        let result = std::panic::catch_unwind(move || {
            let _dir = dir;
            panic!("simulated unhandled error");
        });
        assert!(result.is_err());
        assert!(!path.exists());
    }
}
