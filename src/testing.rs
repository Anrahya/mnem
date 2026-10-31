//! Helpers shared by the unit tests.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

/// A temporary directory path, removed when dropped, even when the test
/// fails. The directory itself is not created.
pub(crate) struct Scratch {
    pub(crate) path: PathBuf,
}

impl Scratch {
    pub(crate) fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("mnem-unit-{}-{n}", process::id()));
        if path.exists() {
            fs::remove_dir_all(&path).expect("stale scratch folder is removable");
        }
        Self { path }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Drop cannot report an error; at worst one temp folder is left.
        drop(fs::remove_dir_all(&self.path));
    }
}
