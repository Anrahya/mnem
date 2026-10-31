//! Small file operations shared by the files Mnem keeps beside the
//! notebook: MIND.md, the watermark and the fired ledger.

use std::fs;
use std::fs::File;
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

fn io_error(path: &Path, source: io::Error) -> NotebookError {
    NotebookError::Io {
        path: path.to_path_buf(),
        source,
    }
}
