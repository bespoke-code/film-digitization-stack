//! Process-owned temporary directory with startup, panic and drop cleanup.

use std::fs::{File, OpenOptions, TryLockError};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, Once};

/// Prefix shared by every cache directory this application creates.
const PREFIX: &str = "scan-to-img-";

static CACHE_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// A temporary cache directory owned by the running process.
///
/// The directory and its lock file are removed when the value is dropped. A
/// panic hook also removes them so an unhandled error does not leave data
/// behind. At startup, [`CacheDir::create`] removes only stale directories
/// that match this application's naming scheme and are no longer locked.
#[derive(Debug)]
pub struct CacheDir {
    path: PathBuf,
    _lock: File,
}

impl CacheDir {
    pub fn create() -> io::Result<Self> {
        install_panic_cleanup();
        clean_stale_dirs();
        let base = std::env::temp_dir();
        let pid = std::process::id();
        for attempt in 0..1000 {
            let path = base.join(format!("{PREFIX}{pid}-{attempt}"));
            let lock_path = lock_path_for(&path);
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&lock_path)?;
            // Lock before the directory becomes visible, so a concurrent
            // startup never mistakes this directory for a stale one.
            match file.try_lock() {
                Ok(()) => {}
                Err(TryLockError::WouldBlock) => continue,
                Err(TryLockError::Error(error)) => return Err(error),
            }
            match std::fs::create_dir(&path) {
                Ok(()) => {
                    if let Ok(mut slot) = CACHE_PATH.lock() {
                        *slot = Some(path.clone());
                    }
                    return Ok(Self { path, _lock: file });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    let _ = std::fs::remove_file(&lock_path);
                    continue;
                }
                Err(error) => {
                    let _ = std::fs::remove_file(&lock_path);
                    return Err(error);
                }
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not create a unique cache directory",
        ))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for CacheDir {
    fn drop(&mut self) {
        remove_cache_files(&self.path);
        if let Ok(mut slot) = CACHE_PATH.lock() {
            if slot.as_deref() == Some(self.path.as_path()) {
                *slot = None;
            }
        }
    }
}

fn install_panic_cleanup() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Some(path) = cache_slot().take() {
                remove_cache_files(&path);
            }
            previous(info);
        }));
    });
}

fn cache_slot() -> MutexGuard<'static, Option<PathBuf>> {
    CACHE_PATH
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Removes cache directories left behind by processes that exited abruptly.
///
/// Only directories named `scan-to-img-<pid>-<attempt>` are considered. A live
/// process holds an exclusive lock on the matching `.lock` file, so its
/// directory is left untouched.
fn clean_stale_dirs() {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !is_cache_name(name) {
            continue;
        }
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            remove_if_unlocked(&entry.path());
        }
    }
}

fn remove_if_unlocked(path: &Path) {
    let lock_path = lock_path_for(path);
    match OpenOptions::new().read(true).write(true).open(&lock_path) {
        Ok(file) => {
            if file.try_lock().is_ok() {
                let _ = std::fs::remove_dir_all(path);
                let _ = std::fs::remove_file(&lock_path);
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let _ = std::fs::remove_dir_all(path);
        }
        Err(_) => {}
    }
}

fn remove_cache_files(path: &Path) {
    let _ = std::fs::remove_dir_all(path);
    let _ = std::fs::remove_file(lock_path_for(path));
}

fn lock_path_for(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    name.push(".lock");
    path.with_file_name(name)
}

fn is_cache_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix(PREFIX) else {
        return false;
    };
    let mut parts = rest.split('-');
    let (Some(pid), Some(attempt), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    !pid.is_empty()
        && !attempt.is_empty()
        && pid.bytes().all(|byte| byte.is_ascii_digit())
        && attempt.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn serialized() -> MutexGuard<'static, ()> {
        TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn creates_a_directory_and_removes_it_on_drop() {
        let _guard = serialized();
        let dir = CacheDir::create().unwrap();
        let path = dir.path().to_path_buf();
        assert!(path.is_dir());
        assert!(is_cache_name(path.file_name().unwrap().to_str().unwrap()));
        std::fs::write(path.join("preview.bin"), b"cached").unwrap();
        drop(dir);
        assert!(!path.exists());
        assert!(!lock_path_for(&path).exists());
    }

    #[test]
    fn removes_the_directory_on_panic() {
        let _guard = serialized();
        let dir = CacheDir::create().unwrap();
        let path = dir.path().to_path_buf();
        std::fs::write(path.join("preview.bin"), b"cached").unwrap();
        let result = std::panic::catch_unwind(move || {
            let _dir = dir;
            panic!("simulated unhandled error");
        });
        assert!(result.is_err());
        assert!(!path.exists());
        assert!(!lock_path_for(&path).exists());
    }

    #[test]
    fn startup_cleanup_removes_only_stale_cache_directories() {
        let _guard = serialized();
        let base = std::env::temp_dir();

        // A pid above every platform's limit cannot be running.
        let stale = base.join(format!("{PREFIX}4294967295-0"));
        let _ = std::fs::remove_dir_all(&stale);
        std::fs::create_dir_all(&stale).unwrap();
        std::fs::write(lock_path_for(&stale), b"").unwrap();

        // Directories from versions without a lock file are stale too.
        let older = base.join(format!("{PREFIX}4294967294-1"));
        let _ = std::fs::remove_dir_all(&older);
        std::fs::create_dir_all(&older).unwrap();

        let unrelated = base.join("scan-to-img-not-a-cache");
        let _ = std::fs::remove_dir_all(&unrelated);
        std::fs::create_dir_all(&unrelated).unwrap();

        let live = CacheDir::create().unwrap();
        let fresh = CacheDir::create().unwrap();

        assert!(!stale.exists());
        assert!(!lock_path_for(&stale).exists());
        assert!(!older.exists());
        assert!(unrelated.exists());
        assert!(live.path().exists());
        assert!(fresh.path().exists());

        drop(live);
        drop(fresh);
        let _ = std::fs::remove_dir_all(&unrelated);
    }
}
