//! The notebook on disk: OptMem's `LOG.txt` and `TREE/<size>` files.
//!
//! Notes are appended to `LOG.txt` and never changed. Summaries live in one
//! file per block size, `TREE/<size>`, where block `k` sits at
//! `k * TREE_REC`; a level file grows in block order and only shrinks when
//! [`Notebook::forget`] cuts it back. Writers hold an exclusive advisory lock
//! on `.lock`, the same lock `memo` takes, so Mnem and `memo` can write one
//! notebook at the same time. Readers take no lock.
//!
//! Two reads are deliberately stricter than `memo`: a partial summary record
//! (a write cut short by a crash) reads as not built rather than as the cut
//! text, and the `config` file is read once, when the notebook is opened.

use std::error::Error;
use std::fmt;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::io::BufReader;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use crate::config::Config;
use crate::config::ConfigError;
use crate::day::Day;
use crate::record;
use crate::record::LOG_REC;
use crate::record::Note;
use crate::record::RecordError;
use crate::record::TREE_REC;
use crate::span::Span;
use crate::text::Text;

/// Blocks of at most this many notes are summarized from the notes
/// themselves; larger blocks from the summaries of their two halves.
pub const RAW_MAX: u64 = 16;

/// An open notebook directory.
#[derive(Debug)]
pub struct Notebook {
    dir: PathBuf,
    config: Config,
}

/// What [`Notebook::put_summary`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PutOutcome {
    /// The summary was appended.
    Saved,
    /// The block is not the next one its level file expects (another writer
    /// got there first, or it was forgotten meanwhile); nothing was written.
    NotNext,
}

/// Why a notebook operation failed.
#[derive(Debug)]
pub enum NotebookError {
    /// The directory does not exist. Only [`Notebook::init`] creates one, so
    /// a mistyped path cannot silently start an empty notebook.
    Missing { dir: PathBuf },
    /// A file operation failed.
    Io { path: PathBuf, source: io::Error },
    /// A record could not be decoded.
    Corrupt {
        path: PathBuf,
        offset: u64,
        source: RecordError,
    },
    /// The `config` file is invalid.
    Config(ConfigError),
    /// Note `id` was asked for, but the log holds only `len` notes.
    OutOfRange { id: u64, len: u64 },
    /// A range of notes was asked for with its start after its end.
    BackwardRange { lo: u64, hi: u64 },
    /// A summary was offered for a block that reaches past the log.
    BeyondLog { block: Span, len: u64 },
    /// A summary was offered for a single note, which has none.
    NotABlock { note: Span },
    /// A position or size does not fit in 64 bits.
    Overflow,
}

impl Notebook {
    /// Creates the notebook at `dir` if it does not exist, and opens it.
    /// Safe to run on an existing notebook: it only creates what is missing.
    pub fn init(dir: &Path) -> Result<Self, NotebookError> {
        let tree = dir.join("TREE");
        fs::create_dir_all(&tree).map_err(|source| io_error(&tree, source))?;
        Self::open(dir)
    }

    /// Opens an existing notebook, creating `TREE/` and an empty `LOG.txt`
    /// inside it only if they are missing; an existing read-only log is fine.
    pub fn open(dir: &Path) -> Result<Self, NotebookError> {
        if !dir.is_dir() {
            return Err(NotebookError::Missing {
                dir: dir.to_path_buf(),
            });
        }
        let tree = dir.join("TREE");
        if !tree.is_dir() {
            fs::create_dir_all(&tree).map_err(|source| io_error(&tree, source))?;
        }
        let log = dir.join("LOG.txt");
        match OpenOptions::new().write(true).create_new(true).open(&log) {
            Ok(_) => sync_dir(dir)?,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(source) => return Err(io_error(&log, source)),
        }
        let config = Config::read(dir).map_err(NotebookError::Config)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            config,
        })
    }

    /// The notebook's size settings, as read when it was opened.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The notebook's directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// How many whole notes the log holds. A partial record left by a crash
    /// is not counted.
    pub fn len(&self) -> Result<u64, NotebookError> {
        records_in(&self.log_path(), LOG_REC)
    }

    /// Whether the log holds no whole note.
    pub fn is_empty(&self) -> Result<bool, NotebookError> {
        Ok(self.len()? == 0)
    }

    /// Appends notes in order and returns the id of the first. Ids are taken
    /// under the lock, so concurrent writers never share one, and the records
    /// are on disk before this returns.
    ///
    /// A batch is not atomic: if the process dies mid-write, the notes
    /// written whole stay and only the cut-off one is dropped, so retrying
    /// the whole batch would repeat them.
    pub fn append(&self, notes: &[(Day, Text)]) -> Result<u64, NotebookError> {
        let _lock = self.lock()?;
        let path = self.log_path();
        repair(&path, LOG_REC)?;
        let first = records_in(&path, LOG_REC)?;
        let mut bytes = Vec::new();
        for (id, (day, text)) in (first..).zip(notes) {
            bytes.extend(record::encode_note(id, day, text));
        }
        append_synced(&self.dir, &path, &bytes)?;
        Ok(first)
    }

    /// Note `id`.
    pub fn note(&self, id: u64) -> Result<Note, NotebookError> {
        let end = id.checked_add(1).ok_or(NotebookError::Overflow)?;
        let len = self.len()?;
        let mut notes = self.notes(id, end)?;
        notes.pop().ok_or(NotebookError::OutOfRange { id, len })
    }

    /// Notes `[lo, hi)`.
    pub fn notes(&self, lo: u64, hi: u64) -> Result<Vec<Note>, NotebookError> {
        if lo > hi {
            return Err(NotebookError::BackwardRange { lo, hi });
        }
        let len = self.len()?;
        if hi > len {
            return Err(NotebookError::OutOfRange {
                id: lo.max(len),
                len,
            });
        }
        let path = self.log_path();
        let mut file = File::open(&path).map_err(|source| io_error(&path, source))?;
        let mut notes = Vec::new();
        for id in lo..hi {
            let offset = id.checked_mul(LOG_REC).ok_or(NotebookError::Overflow)?;
            let bytes = read_record_at(&mut file, offset, LOG_REC)
                .map_err(|source| io_error(&path, source))?
                .ok_or(NotebookError::OutOfRange { id, len })?;
            notes.push(decode_at(&path, offset, &bytes)?);
        }
        Ok(notes)
    }

    /// Every note in order, read in a stream so the whole log is never held
    /// in memory. A partial record at the end, left by a crash, ends the
    /// stream.
    pub fn scan(&self) -> Result<NoteScan, NotebookError> {
        self.scan_from(0)
    }

    /// Like [`Notebook::scan`], starting at note `first`; past the end, the
    /// stream is empty.
    pub fn scan_from(&self, first: u64) -> Result<NoteScan, NotebookError> {
        let path = self.log_path();
        let mut file = File::open(&path).map_err(|source| io_error(&path, source))?;
        let offset = first.checked_mul(LOG_REC).ok_or(NotebookError::Overflow)?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|source| io_error(&path, source))?;
        let width = usize::try_from(LOG_REC).map_err(|_too_wide| NotebookError::Overflow)?;
        Ok(NoteScan {
            reader: BufReader::with_capacity(1 << 16, file),
            width,
            path,
            offset,
            done: false,
        })
    }

    /// The summary of `block`, or `None` if it is not built yet, blank, or
    /// cut short by a crash. A single note has no summary.
    pub fn summary(&self, block: Span) -> Result<Option<String>, NotebookError> {
        let Some(offset) = block.tree_offset() else {
            return Ok(None);
        };
        let path = self.level_path(block);
        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(io_error(&path, source)),
        };
        let Some(bytes) = read_record_at(&mut file, offset, TREE_REC)
            .map_err(|source| io_error(&path, source))?
        else {
            return Ok(None);
        };
        record::decode_summary(&bytes).map_err(|source| NotebookError::Corrupt {
            path,
            offset,
            source,
        })
    }

    /// Appends the summary of `block`. The block must lie inside the log and
    /// be the next one its level file expects; blocks are built in order.
    pub fn put_summary(&self, block: Span, text: &Text) -> Result<PutOutcome, NotebookError> {
        self.put_summary_over(block, text, &[])
    }

    /// Like [`Notebook::put_summary`], but only while each block in
    /// `sources` still has the summary given with it. A summary written from
    /// summaries that were forgotten or rebuilt while it was being written
    /// would outlive them, so it is refused as [`PutOutcome::NotNext`].
    pub fn put_summary_over(
        &self,
        block: Span,
        text: &Text,
        sources: &[(Span, String)],
    ) -> Result<PutOutcome, NotebookError> {
        let Some(offset) = block.tree_offset() else {
            return Err(NotebookError::NotABlock { note: block });
        };
        let _lock = self.lock()?;
        let len = self.len()?;
        if block.hi() > len {
            return Err(NotebookError::BeyondLog { block, len });
        }
        let path = self.level_path(block);
        repair(&path, TREE_REC)?;
        if records_in(&path, TREE_REC)? != offset / TREE_REC {
            return Ok(PutOutcome::NotNext);
        }
        for (source, expected) in sources {
            if self.summary(*source)?.as_ref() != Some(expected) {
                return Ok(PutOutcome::NotNext);
            }
        }
        append_synced(&self.dir.join("TREE"), &path, &record::encode_summary(text))?;
        Ok(PutOutcome::Saved)
    }

    /// Drops the summary of `block` together with the later blocks on its
    /// level and on every larger level from that point on, since those were
    /// built from it or after it. The dropped blocks become pending again;
    /// the log is untouched. Returns them smallest level first.
    ///
    /// The largest level is cut first, so a crash part-way leaves only
    /// summaries whose sources still exist, and calling `forget` again
    /// finishes the job.
    pub fn forget(&self, block: Span) -> Result<Vec<Span>, NotebookError> {
        if block.is_note() {
            return Ok(Vec::new());
        }
        let _lock = self.lock()?;
        let notes = self.len()?;
        let mut sizes = Vec::new();
        let mut size = block.size();
        while size <= notes {
            sizes.push(size);
            size = size.checked_mul(2).ok_or(NotebookError::Overflow)?;
        }
        let mut dropped_by_level = Vec::new();
        for &size in sizes.iter().rev() {
            let path = self.dir.join("TREE").join(size.to_string());
            let keep = block.lo() / size;
            let have = records_in(&path, TREE_REC)?;
            let mut dropped = Vec::new();
            if have > keep {
                for k in keep..have {
                    dropped.push(block_at(k, size)?);
                }
                let length = keep.checked_mul(TREE_REC).ok_or(NotebookError::Overflow)?;
                truncate_synced(&path, length)?;
            }
            dropped_by_level.push(dropped);
        }
        let mut all = Vec::new();
        for level in dropped_by_level.into_iter().rev() {
            all.extend(level);
        }
        Ok(all)
    }

    /// The blocks of a log of `notes` notes that can be built but are not
    /// yet: smaller blocks first, and in log order within one size. With
    /// `Some(limit)`, at most that many. Costs one file-length check per
    /// block size, because each level file holds exactly its first blocks.
    pub fn pending(&self, notes: u64, limit: Option<usize>) -> Result<Vec<Span>, NotebookError> {
        let mut due = Vec::new();
        if limit == Some(0) {
            return Ok(due);
        }
        let mut size: u64 = 2;
        while size <= notes {
            let have = records_in(&self.dir.join("TREE").join(size.to_string()), TREE_REC)?;
            for k in have..notes / size {
                due.push(block_at(k, size)?);
                if limit.is_some_and(|limit| due.len() >= limit) {
                    return Ok(due);
                }
            }
            size = size.checked_mul(2).ok_or(NotebookError::Overflow)?;
        }
        Ok(due)
    }

    /// How many blocks [`Notebook::pending`] would list, without building
    /// the list. `notes` may be an older count than the log's current one,
    /// in which case a level can already hold every block that count needs
    /// and more; it then contributes nothing.
    pub fn pending_count(&self, notes: u64) -> Result<u64, NotebookError> {
        let mut count: u64 = 0;
        let mut size: u64 = 2;
        while size <= notes {
            let have = records_in(&self.dir.join("TREE").join(size.to_string()), TREE_REC)?;
            count = count.saturating_add((notes / size).saturating_sub(have));
            size = size.checked_mul(2).ok_or(NotebookError::Overflow)?;
        }
        Ok(count)
    }

    fn log_path(&self) -> PathBuf {
        self.dir.join("LOG.txt")
    }

    fn level_path(&self, block: Span) -> PathBuf {
        self.dir.join("TREE").join(block.size().to_string())
    }

    /// Takes the notebook's exclusive write lock, held until the returned
    /// file is dropped. Opening for append leaves the file's contents as they
    /// are, so opening it never touches a lock someone else holds.
    fn lock(&self) -> Result<File, NotebookError> {
        let path = self.dir.join(".lock");
        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .map_err(|source| io_error(&path, source))?;
        loop {
            match file.lock() {
                Ok(()) => return Ok(file),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(source) => return Err(io_error(&path, source)),
            }
        }
    }
}

/// Every note of a log, in order; see [`Notebook::scan`].
#[derive(Debug)]
pub struct NoteScan {
    reader: BufReader<File>,
    width: usize,
    path: PathBuf,
    offset: u64,
    done: bool,
}

impl Iterator for NoteScan {
    type Item = Result<Note, NotebookError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let mut bytes = Vec::with_capacity(self.width);
        let limit = u64::try_from(self.width).unwrap_or(u64::MAX);
        let read = match (&mut self.reader).take(limit).read_to_end(&mut bytes) {
            Ok(read) => read,
            Err(source) => {
                self.done = true;
                return Some(Err(io_error(&self.path, source)));
            }
        };
        if read < self.width {
            // The end, or a partial record left by a crash: stop for good, so
            // records appended later are never read from a shifted offset.
            self.done = true;
            return None;
        }
        let offset = self.offset;
        self.offset = self.offset.saturating_add(LOG_REC);
        Some(decode_at(&self.path, offset, &bytes))
    }
}

/// Reads the whole record of `width` bytes at `offset`, or `None` if the file
/// ends before the record does.
fn read_record_at(file: &mut File, offset: u64, width: u64) -> io::Result<Option<Vec<u8>>> {
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = Vec::new();
    let read = file.take(width).read_to_end(&mut bytes)?;
    let whole = u64::try_from(read).is_ok_and(|read| read == width);
    Ok(whole.then_some(bytes))
}

/// Block `k` of size `size`: notes `[k * size, (k + 1) * size)`.
fn block_at(k: u64, size: u64) -> Result<Span, NotebookError> {
    let lo = k.checked_mul(size).ok_or(NotebookError::Overflow)?;
    let hi = lo.checked_add(size).ok_or(NotebookError::Overflow)?;
    // A multiple of `size` spanning `size` notes is aligned by construction.
    Ok(Span::aligned(lo, hi))
}

/// How many whole records of `width` bytes the file holds; 0 if it does not
/// exist.
fn records_in(path: &Path, width: u64) -> Result<u64, NotebookError> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.len() / width),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(source) => Err(io_error(path, source)),
    }
}

/// Drops trailing bytes that do not make a whole record. Only a write cut
/// short by a crash leaves them, and that write was never reported as done;
/// keeping them would shift the next record off its position. The caller
/// holds the lock.
fn repair(path: &Path, width: u64) -> Result<(), NotebookError> {
    let length = match fs::metadata(path) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(source) => return Err(io_error(path, source)),
    };
    let partial = length % width;
    if partial != 0 {
        truncate_synced(path, length - partial)?;
    }
    Ok(())
}

/// Cuts the file to `length` bytes and waits until that is on disk.
fn truncate_synced(path: &Path, length: u64) -> Result<(), NotebookError> {
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|source| io_error(path, source))?;
    file.set_len(length)
        .map_err(|source| io_error(path, source))?;
    file.sync_all().map_err(|source| io_error(path, source))
}

/// Appends `bytes` to the file in `dir` and waits until they are on disk.
/// When the append creates the file, the directory entry is synced too, so
/// the new file survives a power loss.
fn append_synced(dir: &Path, path: &Path, bytes: &[u8]) -> Result<(), NotebookError> {
    let created = !path.exists();
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .map_err(|source| io_error(path, source))?;
    file.write_all(bytes)
        .map_err(|source| io_error(path, source))?;
    file.sync_all().map_err(|source| io_error(path, source))?;
    if created {
        sync_dir(dir)?;
    }
    Ok(())
}

/// Flushes a directory's entries to disk, so files just created in it stay.
fn sync_dir(dir: &Path) -> Result<(), NotebookError> {
    File::open(dir)
        .and_then(|handle| handle.sync_all())
        .map_err(|source| io_error(dir, source))
}

fn decode_at(path: &Path, offset: u64, bytes: &[u8]) -> Result<Note, NotebookError> {
    record::decode_note(bytes).map_err(|source| NotebookError::Corrupt {
        path: path.to_path_buf(),
        offset,
        source,
    })
}

fn io_error(path: &Path, source: io::Error) -> NotebookError {
    NotebookError::Io {
        path: path.to_path_buf(),
        source,
    }
}

impl fmt::Display for NotebookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { dir } => write!(
                f,
                "no notebook at {}; create one with init, or point at an existing one",
                dir.display()
            ),
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Corrupt {
                path,
                offset,
                source,
            } => write!(f, "{} at byte {offset}: {source}", path.display()),
            Self::Config(error) => write!(f, "{error}"),
            Self::OutOfRange { id, len } => {
                write!(f, "there is no note {id}: the log holds {len} notes")
            }
            Self::BackwardRange { lo, hi } => {
                write!(
                    f,
                    "notes {lo} to {hi} is backwards: the start is after the end"
                )
            }
            Self::BeyondLog { block, len } => {
                write!(
                    f,
                    "block {block} reaches past the log, which holds {len} notes"
                )
            }
            Self::NotABlock { note } => write!(f, "note {note} is a single note, not a block"),
            Self::Overflow => write!(f, "a position in the notebook does not fit in 64 bits"),
        }
    }
}

impl Error for NotebookError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Corrupt { source, .. } => Some(source),
            Self::Config(error) => Some(error),
            Self::Missing { .. }
            | Self::OutOfRange { .. }
            | Self::BackwardRange { .. }
            | Self::BeyondLog { .. }
            | Self::NotABlock { .. }
            | Self::Overflow => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::Path;
    use std::thread;

    use super::Notebook;
    use super::NotebookError;
    use super::PutOutcome;
    use crate::day::Day;
    use crate::span::Span;
    use crate::testing::Scratch;
    use crate::text::TEXT_MAX;
    use crate::text::Text;

    fn day() -> Day {
        Day::try_new("2026-10-31").expect("a real date")
    }

    fn text(raw: &str) -> Text {
        Text::try_new(raw, TEXT_MAX).expect("valid test text")
    }

    fn notebook_with(scratch: &Scratch, count: u64) -> Notebook {
        let notebook = Notebook::init(&scratch.path).expect("init succeeds");
        for i in 0..count {
            notebook
                .append(&[(day(), text(&format!("note {i}")))])
                .expect("append succeeds");
        }
        notebook
    }

    fn block(raw: &str) -> Span {
        Span::parse_block(raw).expect("a valid block id")
    }

    fn append_raw(path: &Path, bytes: &[u8]) {
        let mut file = OpenOptions::new()
            .append(true)
            .open(path)
            .expect("file exists");
        file.write_all(bytes).expect("write succeeds");
    }

    #[test]
    fn open_requires_an_existing_directory() {
        let scratch = Scratch::new();
        assert!(
            matches!(
                Notebook::open(&scratch.path),
                Err(NotebookError::Missing { .. })
            ),
            "open must not create a notebook"
        );
        Notebook::init(&scratch.path).expect("init creates it");
        Notebook::init(&scratch.path).expect("init again is harmless");
        let notebook = Notebook::open(&scratch.path).expect("now it opens");
        assert_eq!(notebook.len().expect("len"), 0, "a new notebook is empty");
        assert!(scratch.path.join("TREE").is_dir(), "TREE/ exists");
        assert!(scratch.path.join("LOG.txt").is_file(), "LOG.txt exists");
    }

    #[test]
    fn append_assigns_ids_and_reads_back() {
        let scratch = Scratch::new();
        let notebook = notebook_with(&scratch, 3);
        let first = notebook
            .append(&[(day(), text("four")), (day(), text("five"))])
            .expect("batch append");
        assert_eq!(first, 3, "the batch starts after the existing notes");
        assert_eq!(notebook.len().expect("len"), 5, "five notes");
        assert_eq!(notebook.note(4).expect("note 4").text, "five", "read by id");
        let middle = notebook.notes(1, 3).expect("a range");
        assert_eq!(middle.len(), 2, "two notes in [1, 3)");
        assert_eq!(middle.first().map(|n| n.id), Some(1), "range starts at 1");
        let all: Vec<String> = notebook
            .scan()
            .expect("scan opens")
            .map(|note| note.expect("valid record").text)
            .collect();
        assert_eq!(
            all,
            ["note 0", "note 1", "note 2", "four", "five"],
            "scan order"
        );
        assert!(
            matches!(notebook.note(5), Err(NotebookError::OutOfRange { .. })),
            "reading past the end is an error"
        );
    }

    #[test]
    fn a_partial_record_from_a_crash_is_ignored_then_repaired() {
        let scratch = Scratch::new();
        let notebook = notebook_with(&scratch, 2);
        append_raw(&scratch.path.join("LOG.txt"), b"#2 2026-10-31 half-writ");
        assert_eq!(
            notebook.len().expect("len"),
            2,
            "the partial record is not counted"
        );
        assert_eq!(notebook.scan().expect("scan").count(), 2, "scan skips it");
        let id = notebook
            .append(&[(day(), text("after the crash"))])
            .expect("append");
        assert_eq!(id, 2, "the partial record is cut off before appending");
        let size = fs::metadata(scratch.path.join("LOG.txt"))
            .expect("log")
            .len();
        assert_eq!(size, 3 * 320, "three whole records, nothing else");
        assert_eq!(
            notebook.note(2).expect("note 2").text,
            "after the crash",
            "new note"
        );
    }

    #[test]
    fn summaries_are_built_in_order_and_read_back() {
        let scratch = Scratch::new();
        let notebook = notebook_with(&scratch, 4);
        assert_eq!(
            notebook.summary(block("0-1")).expect("read"),
            None,
            "not built yet"
        );
        assert_eq!(
            notebook
                .put_summary(block("2-3"), &text("too early"))
                .expect("put"),
            PutOutcome::NotNext,
            "2-3 cannot come before 0-1"
        );
        assert_eq!(
            notebook
                .put_summary(block("0-1"), &text("first pair"))
                .expect("put"),
            PutOutcome::Saved,
            "0-1 is next"
        );
        assert_eq!(
            notebook
                .put_summary(block("0-1"), &text("again"))
                .expect("put"),
            PutOutcome::NotNext,
            "0-1 is already built"
        );
        assert_eq!(
            notebook.summary(block("0-1")).expect("read"),
            Some("first pair".to_owned()),
            "read back"
        );
        let note = Span::try_note(0).expect("note span");
        assert_eq!(
            notebook.summary(note).expect("read"),
            None,
            "a note has no summary"
        );
    }

    #[test]
    fn pending_lists_due_blocks_smallest_first() {
        let scratch = Scratch::new();
        let notebook = notebook_with(&scratch, 5);
        let due = notebook.pending(5, None).expect("pending");
        assert_eq!(
            due,
            [block("0-1"), block("2-3"), block("0-3")],
            "all due blocks"
        );
        assert_eq!(
            notebook.pending(5, Some(1)).expect("pending"),
            [block("0-1")],
            "limit"
        );
        assert_eq!(
            notebook.pending_count(5).expect("count"),
            3,
            "count matches"
        );
        notebook
            .put_summary(block("0-1"), &text("pair"))
            .expect("put");
        assert_eq!(
            notebook.pending(5, None).expect("pending"),
            [block("2-3"), block("0-3")],
            "built blocks drop out"
        );
        assert_eq!(
            notebook.pending_count(1).expect("count"),
            0,
            "an older snapshot"
        );
    }

    #[test]
    fn forget_cuts_every_level_from_the_block_up() {
        let scratch = Scratch::new();
        let notebook = notebook_with(&scratch, 8);
        for due in notebook.pending(8, None).expect("pending") {
            let outcome = notebook
                .put_summary(due, &text(&format!("summary {due}")))
                .expect("put");
            assert_eq!(outcome, PutOutcome::Saved, "{due} is built in order");
        }
        let dropped = notebook.forget(block("2-3")).expect("forget");
        let expected = ["2-3", "4-5", "6-7", "0-3", "4-7", "0-7"].map(block);
        assert_eq!(
            dropped, expected,
            "the block, its later neighbours and its ancestors"
        );
        assert_eq!(
            notebook.summary(block("0-1")).expect("read"),
            Some("summary 0-1".to_owned()),
            "earlier blocks stay"
        );
        assert_eq!(
            notebook.pending(8, None).expect("pending"),
            expected,
            "everything dropped is due again"
        );
        assert_eq!(notebook.len().expect("len"), 8, "the log is untouched");
    }

    #[test]
    fn out_of_range_reads_name_the_first_missing_note() {
        let scratch = Scratch::new();
        let notebook = notebook_with(&scratch, 5);
        assert!(
            matches!(
                notebook.note(5),
                Err(NotebookError::OutOfRange { id: 5, len: 5 })
            ),
            "one past the end"
        );
        assert!(
            matches!(
                notebook.notes(3, 9),
                Err(NotebookError::OutOfRange { id: 5, len: 5 })
            ),
            "a range that runs past the end names note 5"
        );
        assert!(
            matches!(
                notebook.notes(7, 9),
                Err(NotebookError::OutOfRange { id: 7, len: 5 })
            ),
            "a range wholly past the end names its start"
        );
        assert!(
            matches!(
                notebook.notes(3, 1),
                Err(NotebookError::BackwardRange { lo: 3, hi: 1 })
            ),
            "a backwards range has its own error"
        );
        assert!(
            notebook.notes(2, 2).expect("empty range").is_empty(),
            "empty range"
        );
    }

    #[test]
    fn a_zero_limit_lists_nothing() {
        let scratch = Scratch::new();
        let notebook = notebook_with(&scratch, 4);
        assert!(
            notebook.pending(4, Some(0)).expect("pending").is_empty(),
            "limit 0"
        );
    }

    #[test]
    fn put_summary_rejects_notes_and_blocks_past_the_log() {
        let scratch = Scratch::new();
        let notebook = notebook_with(&scratch, 4);
        let note = Span::try_note(0).expect("note span");
        assert!(
            matches!(
                notebook.put_summary(note, &text("x")),
                Err(NotebookError::NotABlock { .. })
            ),
            "a single note"
        );
        assert!(
            matches!(
                notebook.put_summary(block("0-7"), &text("x")),
                Err(NotebookError::BeyondLog { len: 4, .. })
            ),
            "the log holds only 4 notes"
        );
        assert!(
            !scratch.path.join("TREE").join("8").exists(),
            "nothing was written"
        );
    }

    #[test]
    fn a_partial_summary_reads_as_unbuilt_and_is_repaired() {
        let scratch = Scratch::new();
        let notebook = notebook_with(&scratch, 4);
        notebook
            .put_summary(block("0-1"), &text("first pair"))
            .expect("put");
        let level = scratch.path.join("TREE").join("2");
        append_raw(&level, b"second pa");
        assert_eq!(
            notebook.summary(block("2-3")).expect("read"),
            None,
            "a cut-off record is not a summary"
        );
        assert_eq!(
            notebook
                .put_summary(block("2-3"), &text("second pair"))
                .expect("put"),
            PutOutcome::Saved,
            "the cut-off record is dropped first"
        );
        assert_eq!(
            fs::metadata(&level).expect("level").len(),
            2 * 288,
            "two whole records"
        );
        assert_eq!(
            notebook.summary(block("2-3")).expect("read"),
            Some("second pair".to_owned()),
            "read back"
        );
    }

    #[test]
    fn forget_of_unbuilt_blocks_and_notes_drops_nothing() {
        let scratch = Scratch::new();
        let notebook = notebook_with(&scratch, 8);
        assert!(
            notebook.forget(block("0-3")).expect("forget").is_empty(),
            "nothing built"
        );
        let note = Span::try_note(3).expect("note span");
        assert!(
            notebook.forget(note).expect("forget").is_empty(),
            "a note has no summary"
        );
        notebook
            .put_summary(block("0-1"), &text("pair"))
            .expect("put");
        assert!(
            notebook.forget(block("2-3")).expect("forget").is_empty(),
            "2-3 was never built"
        );
        assert_eq!(
            notebook.summary(block("0-1")).expect("read"),
            Some("pair".to_owned()),
            "0-1 stays"
        );
    }

    #[test]
    fn a_record_that_is_not_utf8_is_corrupt() {
        let scratch = Scratch::new();
        let notebook = notebook_with(&scratch, 1);
        let mut record = b"#1 2026-10-31 \xff".to_vec();
        record.resize(319, b' ');
        record.push(b'\n');
        append_raw(&scratch.path.join("LOG.txt"), &record);
        assert!(
            matches!(
                notebook.note(1),
                Err(NotebookError::Corrupt { offset: 320, .. })
            ),
            "the bad record is reported with its offset"
        );
        let results: Vec<bool> = notebook.scan().expect("scan").map(|r| r.is_ok()).collect();
        assert_eq!(results, [true, false], "scan reports it too");
    }

    #[test]
    fn a_bad_config_stops_open() {
        let scratch = Scratch::new();
        Notebook::init(&scratch.path).expect("init");
        fs::write(scratch.path.join("config"), "COLOR = 3\n").expect("write config");
        assert!(
            matches!(Notebook::open(&scratch.path), Err(NotebookError::Config(_))),
            "an unknown name is an error"
        );
    }

    #[test]
    fn a_read_only_log_still_opens() {
        let scratch = Scratch::new();
        let notebook = notebook_with(&scratch, 2);
        let log = scratch.path.join("LOG.txt");
        let mut permissions = fs::metadata(&log).expect("log").permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&log, permissions).expect("make read-only");
        let reopened = Notebook::open(&scratch.path).expect("a read-only log opens");
        assert_eq!(reopened.len().expect("len"), 2, "and reads");
        drop(notebook);
    }

    #[test]
    fn concurrent_appends_get_distinct_ids() {
        let scratch = Scratch::new();
        Notebook::init(&scratch.path).expect("init");
        let mut writers = Vec::new();
        for writer in 0..8 {
            let dir = scratch.path.clone();
            writers.push(thread::spawn(move || {
                let notebook = Notebook::open(&dir).expect("open");
                for i in 0..25 {
                    notebook
                        .append(&[(day(), text(&format!("writer {writer} note {i}")))])
                        .expect("append");
                }
            }));
        }
        for handle in writers {
            handle.join().expect("writer thread finishes");
        }
        let notebook = Notebook::open(&scratch.path).expect("open");
        let mut ids = Vec::new();
        for note in notebook.scan().expect("scan") {
            ids.push(note.expect("every record is whole").id);
        }
        let expected: Vec<u64> = (0..200).collect();
        assert_eq!(ids, expected, "200 notes with ids in position order");
    }
}
