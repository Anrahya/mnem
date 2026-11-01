//! Small file operations shared by the files Mnem keeps beside the
//! notebook: MIND.md, the watermark, the fired ledger and the locks.

use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::fs::TryLockError;
use std::io;
use std::io::Write;
use std::path::Path;

use crate::notebook::NotebookError;

/// Replaces `path` with `contents` all at once: readers see the old file
/// or the new one, never part of either.
pub(crate) fn replace(path: &Path, contents: &str) -> Result<(), NotebookError> {
    let mut name = path.file_name().unwrap_or_default().to_owned();
    name.push(".tmp");
    let temp = path.with_file_name(name);
    let written = File::create(&temp)
        .and_then(|mut file| {
            file.write_all(contents.as_bytes())?;
            file.sync_all()
        })
        .and_then(|()| fs::rename(&temp, path));
    written.map_err(|source| {
        drop(fs::remove_file(&temp));
        io_error(path, source)
    })?;
    // Make the rename itself durable, so after a power cut MIND.md is never
    // older than the watermark written after it.
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| io_error(parent, source))
}

/// Removes `path`; a file that is already gone is fine.
pub(crate) fn remove(path: &Path) -> Result<(), NotebookError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(io_error(path, source)),
    }
}

/// An exclusive lock on a file, released when dropped.
///
/// Dropping unlocks before closing. A child process forked meanwhile shares
/// the open file until it execs (Rust opens every file close-on-exec), and
/// closing alone would leave the lock held by the child's copy until then.
#[derive(Debug)]
pub(crate) struct Lock {
    file: File,
}

impl Lock {
    /// Waits for the lock on `path`, creating the file if it is missing.
    pub(crate) fn wait(path: &Path) -> Result<Self, NotebookError> {
        let file = open_lock(path)?;
        loop {
            match file.lock() {
                Ok(()) => return Ok(Self { file }),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(source) => return Err(io_error(path, source)),
            }
        }
    }

    /// Takes the lock on `path` unless someone else holds it.
    pub(crate) fn try_take(path: &Path) -> Result<Option<Self>, NotebookError> {
        let file = open_lock(path)?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { file })),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(source)) => Err(io_error(path, source)),
        }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        // Closing the file right after releases the lock anyway, unless a
        // forked child still shares it; nothing better can be done here.
        drop(self.file.unlock());
    }
}

/// Opens a lock file for append, which never touches its contents.
fn open_lock(path: &Path) -> Result<File, NotebookError> {
    OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .map_err(|source| io_error(path, source))
}

fn io_error(path: &Path, source: io::Error) -> NotebookError {
    NotebookError::Io {
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::Lock;
    use crate::testing::Scratch;

    #[test]
    fn dropping_a_lock_frees_it_while_a_copy_stays_open() {
        let scratch = Scratch::new();
        std::fs::create_dir_all(&scratch.path).expect("scratch folder");
        let path = scratch.path.join(".lock");
        let lock = Lock::wait(&path).expect("taken");
        // A forked child shares the open file the same way a cloned handle does.
        let copy = lock.file.try_clone().expect("a copy of the handle");
        drop(lock);
        assert!(
            Lock::try_take(&path).expect("tried").is_some(),
            "the lock is free although a copy of its file is still open"
        );
        drop(copy);
    }
}
