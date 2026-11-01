//! Mnem as a library, for a harness that embeds it instead of running the
//! `mnem` command: Renoa's `renoa.memory` plugin, first.
//!
//! [`Memory`] is one notebook. Every call is synchronous and blocking, takes
//! the current time as an argument when it needs one, and never calls a
//! model. Work that needs a model comes in two halves: a job (a prompt and
//! the id of what it is for), which the harness sends to its model however
//! it likes, and an apply, which checks the answer before anything touches
//! disk. A stale job is refused with [`MemoryError::Conflict`], and an apply
//! retried after it succeeded settles without writing twice. While a dream
//! runs, another is refused with [`MemoryError::Busy`]; ask again later.
//!
//! Notes carry their author as a leading `[by <author>]`, so several agents
//! can share one notebook and the notebook format stays OptMem's.
//!
//! One driver per notebook runs the summarizer: this library or the `mnem`
//! command's background runs, not both. Two drivers stay correct, since
//! every summary is saved under the notebook's lock, but they can waste
//! model calls on the same block.

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use jiff::civil::DateTime;
use regex::RegexBuilder;

use crate::compactor;
use crate::compactor::SummaryId;
use crate::compactor::SummaryJob;
use crate::config::Config;
use crate::cover;
use crate::day::Day;
use crate::dream;
use crate::dream::DreamId;
use crate::dream::DreamJob;
use crate::dream::Prepared;
use crate::dream::Trigger;
use crate::fired;
use crate::heartbeat::Agenda;
use crate::index::TagIndex;
use crate::mind;
use crate::mind::Mind;
use crate::mind::Nudge;
use crate::model::Answer;
use crate::notebook::Notebook;
use crate::notebook::NotebookError;
use crate::record::Note;
use crate::simhash;
use crate::span::Span;
use crate::tags;
use crate::text;
use crate::text::TEXT_MAX;
use crate::text::Text;
use crate::text::TextError;

/// How large a compiled `recall` pattern may grow, which bounds the time
/// compiling it takes. `\w` covers all of Unicode, so `\w{50}` fits and
/// `\w{100}` does not; ordinary patterns are far smaller.
pub(crate) const RECALL_REGEX_BYTES: usize = 4 << 20;

/// How many of the latest notes a new note is compared with to spot a
/// repeat, so writing a note stays fast however long the log grows.
pub(crate) const REPEAT_WINDOW: u64 = 4096;

/// The longest author name, in bytes.
pub const AUTHOR_MAX: usize = 32;

/// The longest note a new notebook takes, in bytes, its `[by …]` prefix
/// included. A notebook's `config` may lower it; [`Memory::entry_chars`]
/// gives the limit in force.
pub const ENTRY_CHARS: usize = TEXT_MAX;

/// One notebook, opened for a harness.
#[derive(Debug)]
pub struct Memory {
    notebook: Notebook,
    /// False when opened with [`Memory::open_read_only`].
    writable: bool,
}

/// Why a call failed.
#[derive(Debug)]
pub enum MemoryError {
    /// The call or its input was wrong; `message` says what, and what a
    /// correct call looks like. Meant for the agent.
    Invalid { message: String },
    /// The job is stale: the notebook moved on since it was made. Ask for
    /// the current job.
    Conflict { message: String },
    /// Another process is doing the same work on this notebook, such as a
    /// dream; ask again later. Nothing was changed.
    Busy { message: String },
    /// A file in the notebook cannot be read as Mnem wrote it.
    Corrupt { path: PathBuf, detail: String },
    /// A file operation failed.
    Io { path: PathBuf, source: io::Error },
}

/// What writing a note did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoteOutcome {
    /// The new note's id.
    pub id: u64,
    /// Things the agent should know, such as a note it nearly repeats.
    pub warnings: Vec<String>,
    /// How many summaries are due now; when above zero, ask for
    /// [`Memory::compaction_job`].
    pub summaries_due: u64,
}

/// A note as the harness sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoteView {
    /// The note's id.
    pub id: u64,
    /// The day it was written, `YYYY-MM-DD`.
    pub written: String,
    /// The day it says it happened, from an `[on …]` tag.
    pub happened: Option<String>,
    /// The agent that wrote it, from its `[by …]` prefix.
    pub author: Option<String>,
    /// Its text, without the author prefix.
    pub text: String,
    /// The newest note that corrects it, if any.
    pub superseded_by: Option<u64>,
}

/// One line of a zoomed block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entry {
    /// A single note.
    Note(NoteView),
    /// A block of notes, by its summary; `None` while it is not written yet.
    Summary { block: Span, text: Option<String> },
}

/// MIND.md, read for a session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MindView {
    /// A digest of the file, in hex: it changes only when the file does.
    pub revision: String,
    /// The file as written.
    pub text: String,
    /// The file parsed.
    pub mind: Mind,
}

/// A rendering of MIND.md and its revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MindText {
    /// A digest of `text`, in hex: it changes only when `text` does.
    pub revision: String,
    /// The rendered text.
    pub text: String,
}

impl MindView {
    /// MIND.md without its Nudges section, for agents that talk with the
    /// user but do not act on nudges. Its revision is its own, so a change to
    /// the nudges alone leaves it as it was.
    pub fn without_nudges(&self) -> MindText {
        let text = mind::render_without_nudges(&self.mind);
        MindText {
            revision: revision(&text),
            text,
        }
    }
}

/// What the heartbeat should schedule, as of a time. Fired and stale items
/// are left out, and two items for one intent never both appear.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AgendaView {
    /// Items due at a time, earliest first, overdue ones included. Key a
    /// schedule on `(id, due)`.
    pub timed: Vec<Nudge>,
    /// Items for the start of the next session.
    pub next_session: Vec<Nudge>,
    /// Items waiting for a named event.
    pub events: Vec<Nudge>,
    /// What could not be read, such as a MIND.md that does not parse.
    pub problems: Vec<String>,
}

impl Memory {
    /// Creates the notebook at `dir` if it does not exist, and opens it.
    pub fn create(dir: &Path) -> Result<Self, MemoryError> {
        Ok(Self {
            notebook: Notebook::init(dir).map_err(|error| opening(dir, error))?,
            writable: true,
        })
    }

    /// Writes the first note to the notebook at `dir`, creating the notebook
    /// only once the note is known to be valid: a refused note leaves no
    /// notebook behind. On an existing notebook it is [`Memory::note`]. An
    /// I/O failure after the notebook was created can leave it empty.
    pub fn create_with_note(
        dir: &Path,
        author: &str,
        text: &str,
        now: DateTime,
    ) -> Result<(Self, NoteOutcome), MemoryError> {
        if !Notebook::exists(dir) {
            // A `config` placed before the first note sets the limit.
            let config =
                Config::read(dir).map_err(|error| opening(dir, NotebookError::Config(error)))?;
            checked_note(author, text, config.entry_chars(), 0, now)?;
        }
        let memory = Self::create(dir)?;
        let outcome = memory.note(author, text, now)?;
        Ok((memory, outcome))
    }

    /// Opens the existing notebook at `dir`.
    pub fn open(dir: &Path) -> Result<Self, MemoryError> {
        Ok(Self {
            notebook: Notebook::open(dir).map_err(|error| opening(dir, error))?,
            writable: true,
        })
    }

    /// Opens the existing notebook at `dir` for reading only: opening
    /// creates nothing, and calls that write are refused with
    /// [`MemoryError::Invalid`].
    pub fn open_read_only(dir: &Path) -> Result<Self, MemoryError> {
        Ok(Self {
            notebook: Notebook::open_read_only(dir).map_err(|error| opening(dir, error))?,
            writable: false,
        })
    }

    /// The notebook's directory.
    pub fn dir(&self) -> &Path {
        self.notebook.dir()
    }

    /// The longest note this notebook takes, in bytes, its `[by …]` prefix
    /// included.
    pub fn entry_chars(&self) -> usize {
        self.notebook.config().entry_chars()
    }

    /// Writes `text` as a note by `author`, dated `now`. Its tags are checked
    /// strictly; the author prefix counts toward the note's length limit.
    pub fn note(
        &self,
        author: &str,
        text: &str,
        now: DateTime,
    ) -> Result<NoteOutcome, MemoryError> {
        self.writable()?;
        let (text, found, day) =
            checked_note(author, text, self.entry_chars(), self.notebook.len()?, now)?;
        let print = simhash::fingerprint(strip_author(text.as_str()).1);
        let id = self.notebook.append(&[(day, text)])?;
        let mut warnings = Vec::new();
        if let Some(print) = print
            && let Some(similar) = repeat_of(&self.notebook, id, print, &found.updates)?
        {
            warnings.push(format!(
                "It nearly repeats #{}: {}. Before noting, recall shows what is already there.",
                similar.id, similar.text
            ));
        }
        let notes = id.checked_add(1).ok_or(NotebookError::Overflow)?;
        Ok(NoteOutcome {
            id,
            warnings,
            summaries_due: self.notebook.pending_count(notes)?,
        })
    }

    /// The latest `limit` notes whose `#<id> <day> <text>` line matches
    /// `pattern` (Rust regex syntax, case-insensitive), oldest first.
    pub fn recall(&self, pattern: &str, limit: usize) -> Result<Vec<NoteView>, MemoryError> {
        let regex = RegexBuilder::new(pattern)
            .case_insensitive(true)
            .size_limit(RECALL_REGEX_BYTES)
            .build()
            .map_err(|error| {
                invalid(format!(
                    "Bad pattern: {error}. Patterns use Rust regex syntax, which has no lookaround or backreferences."
                ))
            })?;
        let index = TagIndex::build(&self.notebook, 0, u64::MAX)?;
        let mut kept: VecDeque<Note> = VecDeque::new();
        for note in self.notebook.scan()? {
            let note = note?;
            if !regex.is_match(&note.line()) {
                continue;
            }
            kept.push_back(note);
            if kept.len() > limit {
                kept.pop_front();
            }
        }
        let mut found = Vec::new();
        for note in kept {
            found.push(view(&index, note));
        }
        Ok(found)
    }

    /// The two halves of `block`: notes, or summaries of smaller blocks.
    pub fn zoom(&self, block: Span) -> Result<Vec<Entry>, MemoryError> {
        let notes = self.notebook.len()?;
        if block.lo() >= notes {
            return Err(invalid(format!(
                "#{block} starts past the end of the notebook, which holds {notes} notes."
            )));
        }
        let index = TagIndex::build(&self.notebook, block.lo(), notes)?;
        let mut entries = Vec::new();
        if let Some((first, second)) = block.halves() {
            for half in [first, second] {
                if half.lo() >= notes {
                    continue;
                }
                if half.is_note() {
                    entries.push(Entry::Note(view(&index, self.notebook.note(half.lo())?)));
                } else {
                    entries.push(Entry::Summary {
                        block: half,
                        text: self.notebook.summary(half)?,
                    });
                }
            }
        }
        Ok(entries)
    }

    /// The whole notebook in a bounded view, oldest first: summaries for
    /// older blocks, the latest notes in full. A summary not written yet is
    /// marked so; [`Memory::compaction_job`] writes it.
    pub fn wake(&self) -> Result<String, MemoryError> {
        let notes = self.notebook.len()?;
        if notes == 0 {
            return Ok("The notebook is empty.".to_owned());
        }
        let spans = cover::cover(notes, self.notebook.config().wake_lines())
            .map_err(|error| invalid(error.to_string()))?;
        let mut first_note = notes;
        for span in &spans {
            if span.is_note() {
                first_note = first_note.min(span.lo());
            }
        }
        let index = TagIndex::build(&self.notebook, first_note, notes)?;
        let mut lines = Vec::new();
        for span in spans {
            if span.is_note() {
                lines.push(index.shown(&self.notebook.note(span.lo())?));
                continue;
            }
            lines.push(match self.notebook.summary(span)? {
                Some(summary) => format!("#{span} {summary}"),
                None => format!("#{span} (not summarized yet; zoom in to read it)"),
            });
        }
        Ok(lines.join("\n"))
    }

    /// MIND.md, if Dream has written one.
    pub fn mind(&self) -> Result<Option<MindView>, MemoryError> {
        let Some(text) = dream::read_mind(self.dir())? else {
            return Ok(None);
        };
        let (mind, _) = mind::parse(&text).map_err(|missing| MemoryError::Corrupt {
            path: self.dir().join(mind::FILE),
            detail: missing.to_string(),
        })?;
        Ok(Some(MindView {
            revision: revision(&text),
            text,
            mind,
        }))
    }

    /// The next summary due, if any: send its prompt to a model, then pass
    /// the answer to [`Memory::apply_compaction`].
    pub fn compaction_job(&self) -> Result<Option<SummaryJob>, MemoryError> {
        self.writable()?;
        match compactor::next_job(&self.notebook)? {
            compactor::Next::Done => Ok(None),
            compactor::Next::Job(job) => Ok(Some(job)),
            compactor::Next::Blank { block, half } => Err(MemoryError::Corrupt {
                path: self.dir().join("TREE"),
                detail: format!(
                    "#{block} is built from #{half}, whose summary record is empty; forget #{half} to have it written again"
                ),
            }),
        }
    }

    /// Checks the model's `answer` to the summary job `id` and saves it, or
    /// returns the job to send next when the line was too long.
    pub fn apply_compaction(
        &self,
        id: &SummaryId,
        answer: Answer,
    ) -> Result<compactor::Applied, MemoryError> {
        self.writable()?;
        compactor::apply(&self.notebook, id, answer).map_err(|error| match error {
            compactor::ApplyError::Stale(message) => MemoryError::Conflict { message },
            compactor::ApplyError::Invalid(message) => MemoryError::Invalid { message },
            compactor::ApplyError::Notebook(error) => error.into(),
        })
    }

    /// The dream due as of `now` for `trigger`, or `None` when too few notes
    /// are new; then MIND.md is only re-checked, without a model. `user` is
    /// the user's own profile (USER.md), which the dream reads but never
    /// repeats.
    pub fn dream_job(
        &self,
        trigger: Trigger,
        user: Option<&str>,
        now: DateTime,
    ) -> Result<Option<DreamJob>, MemoryError> {
        self.writable()?;
        match dream::job(&self.notebook, trigger, user, now)? {
            Prepared::Tidied { .. } => Ok(None),
            Prepared::Job(job) => Ok(Some(job)),
            Prepared::Busy => Err(dreaming()),
        }
    }

    /// Checks the model's `answer` to the dream `id`, and if it holds up,
    /// replaces MIND.md. A rejected or declined answer counts toward
    /// Dream's give-up rule and changes nothing else, so pass
    /// [`Answer::Declined`] only when the model declined; when it could not
    /// be reached, do not apply at all.
    pub fn apply_dream(
        &self,
        id: &DreamId,
        answer: Answer,
        now: DateTime,
    ) -> Result<dream::Report, MemoryError> {
        self.writable()?;
        dream::apply(&self.notebook, id, answer, now).map_err(|error| match error {
            dream::ApplyError::Stale(message) => MemoryError::Conflict { message },
            dream::ApplyError::Busy => dreaming(),
            dream::ApplyError::Notebook(error) => error.into(),
        })
    }

    /// Forgets the summary of `block` and every summary built on it, so they
    /// are written again. Returns the blocks forgotten. Notes are never
    /// forgotten.
    pub fn forget(&self, block: Span) -> Result<Vec<Span>, MemoryError> {
        self.writable()?;
        Ok(self.notebook.forget(block)?)
    }

    /// Everything the heartbeat should schedule as of `now`.
    pub fn agenda(&self, now: DateTime) -> Result<AgendaView, MemoryError> {
        let agenda = Agenda::read(&self.notebook)?;
        let owned = |items: Vec<&Nudge>| items.into_iter().cloned().collect::<Vec<Nudge>>();
        let timed = owned(agenda.timed(now));
        let next_session = owned(agenda.for_session(now));
        let events = owned(agenda.waiting(now));
        Ok(AgendaView {
            timed,
            next_session,
            events,
            problems: agenda.problems,
        })
    }

    /// Records that the item `id`, due at `due` if it is timed, was handed to
    /// the agent at `at`, so it does not fire again. Recording it twice is
    /// harmless.
    pub fn record_fired(
        &self,
        id: &str,
        due: Option<DateTime>,
        at: DateTime,
    ) -> Result<(), MemoryError> {
        self.writable()?;
        if id.is_empty() || id.contains(char::is_whitespace) {
            return Err(invalid(format!(
                "{id:?} is not an item id: use the id from the agenda, such as \"call-mom\" or \"r12\"."
            )));
        }
        // The ledger keeps minutes; compare and record at that precision.
        let due = due.map(minute);
        let at = minute(at);
        let held = fired::hold(self.dir())?;
        let ledger = fired::read(self.dir())?;
        if ledger
            .iter()
            .any(|entry| entry.id == id && entry.due == due)
        {
            return Ok(());
        }
        let agenda = Agenda::read(&self.notebook)?;
        fired::record(&held, self.dir(), &[(id.to_owned(), due)], at, |entry| {
            agenda.still_needed(entry, at)
        })?;
        Ok(())
    }

    fn writable(&self) -> Result<(), MemoryError> {
        if self.writable {
            return Ok(());
        }
        Err(invalid(format!(
            "{} was opened read-only; open it with Memory::open to write.",
            self.dir().display()
        )))
    }
}

/// A note by `author` checked for a notebook that takes `limit` bytes and
/// holds `before` notes: its full text, its tags and its day.
fn checked_note(
    author: &str,
    text: &str,
    limit: usize,
    before: u64,
    now: DateTime,
) -> Result<(Text, tags::Tags, Day), MemoryError> {
    let author = checked_author(author)?;
    if text.trim_matches(text::is_python_space).is_empty() {
        return Err(invalid(note_problem(&TextError::Empty, limit)));
    }
    let full = format!("[by {author}] {text}");
    let text = Text::try_new(&full, limit).map_err(|error| invalid(note_problem(&error, limit)))?;
    let found = checked_tags(text.as_str(), before).map_err(invalid)?;
    let day = Day::try_new(&now.date().to_string())
        .map_err(|_not_a_day| invalid(format!("{now} cannot date a note")))?;
    Ok((text, found, day))
}

fn revision(text: &str) -> String {
    format!("{:016x}", simhash::fnv1a(text))
}

/// `text` split into its author and the rest, when it starts `[by <author>] `.
pub fn strip_author(text: &str) -> (Option<&str>, &str) {
    let Some(rest) = text.strip_prefix("[by ") else {
        return (None, text);
    };
    match rest.split_once("] ") {
        Some((author, rest)) if checked_author(author).is_ok() => (Some(author), rest),
        _ => (None, text),
    }
}

fn checked_author(author: &str) -> Result<&str, MemoryError> {
    let bad = author.is_empty()
        || author.len() > AUTHOR_MAX
        || author.trim() != author
        || author.contains(['[', ']'])
        || author.contains(char::is_control);
    if bad {
        return Err(invalid(format!(
            "{author:?} cannot be an author: use the agent's name, 1 to {AUTHOR_MAX} bytes, without brackets or line breaks."
        )));
    }
    Ok(author)
}

/// An error opening the notebook at `dir`, with the config file's full path.
fn opening(dir: &Path, error: NotebookError) -> MemoryError {
    if matches!(error, NotebookError::Config(_)) {
        return MemoryError::Corrupt {
            path: dir.join("config"),
            detail: error.to_string(),
        };
    }
    error.into()
}

fn view(index: &TagIndex, note: Note) -> NoteView {
    let tags = tags::parse_lenient(&note.text);
    let (author, text) = strip_author(&note.text);
    NoteView {
        id: note.id,
        happened: tags.happened.map(|day| day.as_str().to_owned()),
        author: author.map(str::to_owned),
        text: text.to_owned(),
        superseded_by: index.superseded_by(note.id),
        written: note.day,
    }
}

fn note_problem(error: &TextError, limit: usize) -> String {
    match error {
        TextError::Empty => "The note is empty; write one line.".to_owned(),
        TextError::MultiLine { .. } => "A note is one line; remove the line breaks.".to_owned(),
        TextError::TooLong { .. } => format!(
            "The note, with its [by …] prefix, is over {limit} bytes; shorten it or split it into two notes."
        ),
    }
}

fn minute(time: DateTime) -> DateTime {
    time.with()
        .second(0)
        .subsec_nanosecond(0)
        .build()
        .unwrap_or(time)
}

fn dreaming() -> MemoryError {
    MemoryError::Busy {
        message: "another dream is running on this notebook; ask again when it ends".to_owned(),
    }
}

fn invalid(message: impl Into<String>) -> MemoryError {
    MemoryError::Invalid {
        message: message.into(),
    }
}

/// The tags of a note about to be written when `before` notes exist, or why
/// they are wrong.
pub(crate) fn checked_tags(text: &str, before: u64) -> Result<tags::Tags, String> {
    let found = tags::parse(text).map_err(|error| error.to_string())?;
    for target in &found.updates {
        if *target >= before {
            let held = if before == 1 {
                "1 note".to_owned()
            } else {
                format!("{before} notes")
            };
            return Err(format!(
                "[updates #{target}] names a note that is not written yet; the notebook holds {held} before this one."
            ));
        }
    }
    Ok(found)
}

/// The newest of the [`REPEAT_WINDOW`] notes before `id` that note `id`,
/// whose fingerprint is `print`, nearly repeats. Notes it corrects, and
/// notes already superseded, do not count: saying again what was
/// corrected away is news.
pub(crate) fn repeat_of(
    notebook: &Notebook,
    id: u64,
    print: u64,
    corrects: &[u64],
) -> Result<Option<Note>, NotebookError> {
    let mut index = TagIndex::default();
    let mut near = Vec::new();
    for note in notebook.scan_from(id.saturating_sub(REPEAT_WINDOW))? {
        let note = note?;
        if note.id >= id {
            break;
        }
        index.add(&note);
        if corrects.contains(&note.id) {
            continue;
        }
        if let Some(other) = simhash::fingerprint(strip_author(&note.text).1)
            && simhash::distance(print, other) <= simhash::NEAR
        {
            near.push(note);
        }
    }
    Ok(near
        .into_iter()
        .rev()
        .find(|note| index.superseded_by(note.id).is_none()))
}

impl From<NotebookError> for MemoryError {
    fn from(error: NotebookError) -> Self {
        match error {
            NotebookError::Io { path, source } => Self::Io { path, source },
            NotebookError::Missing { dir } => Self::Io {
                source: io::Error::new(io::ErrorKind::NotFound, "no notebook here"),
                path: dir,
            },
            NotebookError::Corrupt { ref path, .. } | NotebookError::TooLarge { ref path, .. } => {
                Self::Corrupt {
                    path: path.clone(),
                    detail: error.to_string(),
                }
            }
            NotebookError::Config(_) => Self::Corrupt {
                path: PathBuf::from("config"),
                detail: error.to_string(),
            },
            NotebookError::OutOfRange { .. }
            | NotebookError::BackwardRange { .. }
            | NotebookError::BeyondLog { .. }
            | NotebookError::NotABlock { .. }
            | NotebookError::Overflow => Self::Invalid {
                message: error.to_string(),
            },
        }
    }
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { message } | Self::Conflict { message } | Self::Busy { message } => {
                f.write_str(message)
            }
            Self::Corrupt { path, detail } => write!(f, "{}: {detail}", path.display()),
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
        }
    }
}

impl Error for MemoryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Invalid { .. }
            | Self::Conflict { .. }
            | Self::Busy { .. }
            | Self::Corrupt { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use jiff::civil::DateTime;

    use super::Entry;
    use super::Memory;
    use super::MemoryError;
    use crate::compactor::Applied;
    use crate::dream;
    use crate::dream::Outcome;
    use crate::dream::Trigger;
    use crate::mind::When;
    use crate::model::Answer;
    use crate::span::Span;
    use crate::tags;
    use crate::testing::Scratch;

    fn at(raw: &str) -> DateTime {
        tags::date_time(raw).expect("a valid time")
    }

    fn block(raw: &str) -> Span {
        Span::parse_block(raw).expect("a valid block")
    }

    fn reply(text: &str) -> Answer {
        Answer::Reply(text.to_owned())
    }

    /// A notebook with `notes`, each `(time, author, text)`.
    fn memory(scratch: &Scratch, notes: &[(&str, &str, &str)]) -> Memory {
        let memory = Memory::create(&scratch.path).expect("a new notebook");
        for (time, author, text) in notes {
            memory
                .note(author, text, at(time))
                .expect("the note is saved");
        }
        memory
    }

    #[test]
    fn notes_carry_their_author_and_are_checked() {
        let scratch = Scratch::new();
        let memory = memory(&scratch, &[]);
        let first = memory
            .note(
                "Arcee",
                "Lives in Mumbai [on 2026-10-01]",
                at("2026-10-28T09:00"),
            )
            .expect("saved");
        assert_eq!(first.id, 0, "the first note");
        let second = memory
            .note(
                "Scout",
                "[updates #0] Moved to Delhi",
                at("2026-10-30T09:00"),
            )
            .expect("saved");
        assert_eq!(
            second.summaries_due, 1,
            "two notes make a block to summarize"
        );
        let found = memory.recall("mumbai|delhi", 10).expect("recall");
        let first_view = found.first().expect("the first note");
        assert_eq!(
            first_view.author.as_deref(),
            Some("Arcee"),
            "author parsed back"
        );
        assert_eq!(
            first_view.text, "Lives in Mumbai [on 2026-10-01]",
            "text without the prefix"
        );
        assert_eq!(first_view.written, "2026-10-28", "dated by now");
        assert_eq!(
            first_view.happened.as_deref(),
            Some("2026-10-01"),
            "happened from the tag"
        );
        assert_eq!(first_view.superseded_by, Some(1), "corrected by #1");
        assert_eq!(
            memory.recall(".", 1).expect("recall").len(),
            1,
            "limit keeps the latest"
        );
        for (author, text) in [("", "x"), ("a]b", "x"), ("Arcee", "[updates #9] x")] {
            assert!(
                matches!(
                    memory.note(author, text, at("2026-10-30T09:00")),
                    Err(MemoryError::Invalid { .. })
                ),
                "{author:?} / {text:?} is refused"
            );
        }
        let long = "x".repeat(275);
        assert!(
            matches!(
                memory.note("Arcee", &long, at("2026-10-30T09:00")),
                Err(MemoryError::Invalid { .. })
            ),
            "the prefix counts toward the limit"
        );
    }

    #[test]
    fn a_near_repeat_by_another_agent_is_flagged() {
        let scratch = Scratch::new();
        let memory = memory(
            &scratch,
            &[(
                "2026-10-28T09:00",
                "Arcee",
                "The dentist appointment is on Friday morning",
            )],
        );
        let outcome = memory
            .note(
                "Scout",
                "The dentist appointment is on Friday morning",
                at("2026-10-28T10:00"),
            )
            .expect("saved");
        assert!(
            outcome
                .warnings
                .first()
                .is_some_and(|warning| warning.contains("nearly repeats #0")),
            "authors do not hide a repeat: {:?}",
            outcome.warnings
        );
    }

    #[test]
    fn zoom_and_wake_show_blocks() {
        let scratch = Scratch::new();
        let memory = memory(
            &scratch,
            &[
                ("2026-10-28T09:00", "Arcee", "one"),
                ("2026-10-28T09:01", "Arcee", "two"),
            ],
        );
        let entries = memory.zoom(block("0-1")).expect("zoom");
        assert!(
            matches!(entries.as_slice(), [Entry::Note(a), Entry::Note(b)] if a.text == "one" && b.text == "two"),
            "a pair zooms to its notes: {entries:?}"
        );
        assert!(
            matches!(memory.zoom(block("4-5")), Err(MemoryError::Invalid { .. })),
            "past the end"
        );
        assert!(
            memory
                .wake()
                .expect("wake")
                .contains("#1 2026-10-28 [by Arcee] two"),
            "recent notes in full"
        );
    }

    #[test]
    fn a_summary_job_retries_across_calls_and_settles_once() {
        let scratch = Scratch::new();
        let memory = memory(
            &scratch,
            &[
                ("2026-10-28T09:00", "Arcee", "one"),
                ("2026-10-28T09:01", "Arcee", "two"),
            ],
        );
        let job = memory.compaction_job().expect("job").expect("one is due");
        assert_eq!(job.id.block, block("0-1"), "the pair");
        let Applied::Retry(retry) = memory
            .apply_compaction(&job.id, reply(&"y".repeat(300)))
            .expect("applied")
        else {
            panic!("a line too long is asked for again");
        };
        assert!(
            retry.prompt.contains("Your last line was 300 bytes"),
            "the retry says why"
        );
        let reopened = Memory::open(&scratch.path).expect("open");
        assert_eq!(
            reopened
                .compaction_job()
                .expect("job")
                .map(|again| again.id),
            Some(retry.id),
            "the try survives a new process"
        );
        assert!(
            matches!(
                reopened.apply_compaction(&job.id, reply("late")),
                Err(MemoryError::Conflict { .. })
            ),
            "an answer to the first try is stale"
        );
        assert_eq!(
            reopened
                .apply_compaction(&retry.id, reply("one and two"))
                .expect("applied"),
            Applied::Saved(crate::compactor::Written::ByModel),
            "saved"
        );
        assert_eq!(
            reopened
                .apply_compaction(&retry.id, reply("again"))
                .expect("applied"),
            Applied::Settled,
            "a retried apply does not write twice"
        );
        assert_eq!(reopened.compaction_job().expect("job"), None, "nothing due");
    }

    #[test]
    fn a_summary_job_over_changed_input_is_stale() {
        let scratch = Scratch::new();
        let memory = Memory::create(&scratch.path).expect("a new notebook");
        for n in 0..32 {
            memory
                .note("Arcee", &format!("note {n}"), at("2026-10-28T09:00"))
                .expect("saved");
        }
        // Blocks up to 16 notes are summarized from notes; 0-31 is built
        // from the summaries of its halves.
        let parent = loop {
            let job = memory.compaction_job().expect("job").expect("one is due");
            if job.id.block == block("0-31") {
                break job;
            }
            memory
                .apply_compaction(&job.id, reply(&format!("summary of {}", job.id.block)))
                .expect("saved");
        };
        memory.forget(block("16-31")).expect("forget");
        let half = memory.compaction_job().expect("job").expect("16-31 again");
        assert_eq!(
            half.id.block,
            block("16-31"),
            "the forgotten half comes first"
        );
        memory
            .apply_compaction(&half.id, reply("a different summary"))
            .expect("saved");
        let again = memory.compaction_job().expect("job").expect("0-31 again");
        assert_eq!(again.id.block, parent.id.block, "the same block is due");
        assert_ne!(again.id.digest, parent.id.digest, "with new input");
        assert!(
            matches!(
                memory.apply_compaction(&parent.id, reply("whole")),
                Err(MemoryError::Conflict { message }) if message.contains("input")
            ),
            "an answer about the old halves is refused"
        );
        assert_eq!(
            memory
                .apply_compaction(&again.id, reply("whole"))
                .expect("applied"),
            Applied::Saved(crate::compactor::Written::ByModel),
            "the current job saves"
        );
    }

    #[test]
    fn fired_times_are_kept_to_the_minute() {
        let scratch = Scratch::new();
        let memory = memory(&scratch, &[]);
        for raw in ["2026-10-31T08:00:15", "2026-10-31T08:00:45"] {
            let time: DateTime = raw.parse().expect("a time with seconds");
            memory
                .record_fired("tea", Some(time), time)
                .expect("recorded");
        }
        let ledger = crate::fired::read(&scratch.path).expect("ledger");
        assert_eq!(ledger.len(), 1, "seconds do not make a second entry");
    }

    #[test]
    fn a_dream_never_waits_for_another() {
        let scratch = Scratch::new();
        let memory = memory(&scratch, &NOTES);
        let now = at("2026-10-31T03:00");
        let job = memory
            .dream_job(Trigger::Nightly, None, now)
            .expect("job")
            .expect("five new notes");
        let held = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(scratch.path.join(".dream.lock"))
            .expect("lock file");
        held.lock().expect("lock");
        assert!(
            matches!(
                memory.dream_job(Trigger::Nightly, None, now),
                Err(MemoryError::Busy { .. })
            ),
            "a held lock makes a new job busy, not a wait"
        );
        assert!(
            matches!(
                memory.apply_dream(&job.id, reply(&draft("eggs", "Omelette")), now),
                Err(MemoryError::Busy { .. })
            ),
            "a held lock makes an apply busy, not a conflict"
        );
        held.unlock().expect("unlock");
        assert!(
            memory
                .apply_dream(&job.id, reply(&draft("eggs", "Omelette")), now)
                .is_ok(),
            "the same apply goes through once the lock is free"
        );
    }

    #[test]
    fn an_oversized_mind_is_corrupt_and_not_read() {
        let scratch = Scratch::new();
        let memory = memory(&scratch, &NOTES);
        let path = scratch.path.join(crate::mind::FILE);
        std::fs::write(&path, "x".repeat(crate::mind::FILE_MAX + 1)).expect("written");
        assert!(
            matches!(memory.mind(), Err(MemoryError::Corrupt { .. })),
            "a MIND.md larger than Mnem writes is refused"
        );
        let agenda = memory
            .agenda(at("2026-10-31T03:00"))
            .expect("the agenda still reads");
        assert_eq!(agenda.problems.len(), 1, "and reports it: {agenda:?}");
    }

    #[test]
    fn recall_patterns_are_bounded() {
        let scratch = Scratch::new();
        let memory = memory(&scratch, &NOTES);
        assert!(
            memory.recall(r"\w{50}", 10).is_ok(),
            "an ordinary long pattern compiles"
        );
        assert!(
            matches!(
                memory.recall(r"(\w{50}){40}", 10),
                Err(MemoryError::Invalid { .. })
            ),
            "a pattern that compiles huge is refused"
        );
    }

    #[test]
    fn a_blank_note_is_refused() {
        let scratch = Scratch::new();
        let memory = memory(&scratch, &[]);
        for text in ["", "   ", "\u{1c}\u{1f}"] {
            assert!(
                matches!(
                    memory.note("Arcee", text, at("2026-10-31T09:00")),
                    Err(MemoryError::Invalid { .. })
                ),
                "the author prefix alone is not a note: {text:?}"
            );
        }
        assert_eq!(
            memory.recall(".", 10).expect("recall").len(),
            0,
            "nothing was written"
        );
    }

    #[test]
    fn a_refused_first_note_creates_no_notebook() {
        let scratch = Scratch::new();
        let dir = scratch.path.join("person").join("mnem");
        let long = "x".repeat(super::ENTRY_CHARS);
        for text in ["", long.as_str(), "[updates #0] nothing to update"] {
            assert!(
                matches!(
                    Memory::create_with_note(&dir, "Arcee", text, at("2026-10-31T09:00")),
                    Err(MemoryError::Invalid { .. })
                ),
                "refused: {text:?}"
            );
            assert!(!dir.exists(), "no directory left behind for {text:?}");
        }
        let (memory, outcome) =
            Memory::create_with_note(&dir, "Arcee", "Lives in Mumbai", at("2026-10-31T09:00"))
                .expect("created");
        assert_eq!(outcome.id, 0, "the first note");
        assert_eq!(memory.entry_chars(), super::ENTRY_CHARS, "default limit");
        let (_, again) =
            Memory::create_with_note(&dir, "Scout", "Likes tea", at("2026-10-31T09:05"))
                .expect("an existing notebook takes the note");
        assert_eq!(again.id, 1, "appended, not recreated");

        let seeded = scratch.path.join("seeded");
        std::fs::create_dir_all(&seeded).expect("dir");
        std::fs::write(seeded.join("config"), "ENTRY_CHARS=20\n").expect("config");
        assert!(
            matches!(
                Memory::create_with_note(&seeded, "Arcee", &"x".repeat(30), at("2026-10-31T09:00")),
                Err(MemoryError::Invalid { .. })
            ),
            "a seeded config's limit applies to the first note"
        );
        assert!(
            !crate::notebook::Notebook::exists(&seeded) && !seeded.join("TREE").exists(),
            "the refused note created no notebook in the seeded dir"
        );
    }

    #[test]
    fn a_read_only_open_creates_nothing_and_writes_nothing() {
        let scratch = Scratch::new();
        let missing = scratch.path.join("nobody");
        assert!(
            matches!(
                Memory::open_read_only(&missing),
                Err(MemoryError::Io { .. })
            ),
            "no notebook to open"
        );
        assert!(!missing.exists(), "opening created nothing");
        drop(memory(&scratch, &NOTES));
        std::fs::remove_dir_all(scratch.path.join("TREE")).expect("remove TREE");
        let reader = Memory::open_read_only(&scratch.path).expect("opened");
        assert_eq!(
            reader.recall("breakfast", 10).expect("recall").len(),
            2,
            "reads work"
        );
        assert!(reader.wake().is_ok(), "wake reads without TREE");
        assert!(reader.zoom(block("0-1")).is_ok(), "zoom reads without TREE");
        assert!(
            matches!(
                reader.note("Arcee", "x", at("2026-10-31T09:00")),
                Err(MemoryError::Invalid { .. })
            ),
            "writes are refused"
        );
        assert!(
            matches!(reader.compaction_job(), Err(MemoryError::Invalid { .. })),
            "summary jobs are refused"
        );
        assert!(
            !scratch.path.join("TREE").exists(),
            "reading did not recreate TREE"
        );
    }

    #[test]
    fn mind_without_nudges_keeps_its_revision_across_nudge_changes() {
        let scratch = Scratch::new();
        let memory = memory(&scratch, &NOTES);
        let now = at("2026-10-31T03:00");
        let job = memory
            .dream_job(Trigger::Nightly, None, now)
            .expect("job")
            .expect("five new notes");
        memory
            .apply_dream(&job.id, reply(&draft("eggs", "Suggest an omelette")), now)
            .expect("applied");
        let first = memory.mind().expect("read").expect("written");
        let shared = first.without_nudges();
        assert!(
            !shared.text.contains("## Nudges") && !shared.text.contains("omelette"),
            "no nudges: {}",
            shared.text
        );
        assert!(shared.text.contains("Lives in Delhi"), "the rest stays");
        let written =
            std::fs::read_to_string(scratch.path.join(crate::mind::FILE)).expect("MIND.md");
        let changed = written.replace("Suggest an omelette", "Suggest a frittata");
        std::fs::write(scratch.path.join(crate::mind::FILE), changed).expect("rewrite");
        let second = memory.mind().expect("read").expect("written");
        assert_ne!(first.revision, second.revision, "the whole file changed");
        assert_eq!(
            shared.revision,
            second.without_nudges().revision,
            "a nudge-only change keeps the shared revision"
        );
    }

    const NOTES: [(&str, &str, &str); 5] = [
        ("2026-10-28T09:00", "Arcee", "Lives in Mumbai"),
        (
            "2026-10-29T09:00",
            "Arcee",
            "Skipped breakfast, early meeting",
        ),
        ("2026-10-30T09:00", "Scout", "Skipped breakfast again"),
        ("2026-10-30T09:05", "Arcee", "[updates #0] Moved to Delhi"),
        ("2026-10-30T09:10", "Arcee", "Bought eggs and spinach"),
    ];

    fn draft(id: &str, action: &str) -> String {
        format!(
            "<mind>\n## Durable\n- Lives in Delhi (#3)\n## Active\n- Busy mornings (#1, #2)\n## Hunches\n## Nudges\n\
             - {id} | 2026-10-31T08:00 | 10:00 | {action} | Skipped breakfast; bought eggs (#2, #4) | notify\n</mind>\n"
        )
    }

    #[test]
    fn a_dream_job_applies_once_and_keeps_nudge_ids() {
        let scratch = Scratch::new();
        let memory = memory(&scratch, &NOTES);
        let now = at("2026-10-31T03:00");
        let job = memory
            .dream_job(Trigger::Nightly, Some("Name: Asha <not a tag>"), now)
            .expect("job")
            .expect("five new notes");
        assert!(
            job.prompt
                .contains("<user-profile>\nName: Asha ‹not a tag›\n</user-profile>"),
            "the profile goes in as data"
        );
        let report = memory
            .apply_dream(&job.id, reply(&draft("eggs", "Suggest an omelette")), now)
            .expect("applied");
        assert!(
            matches!(
                report.outcome,
                Outcome::Dreamed {
                    first: 0,
                    end: 5,
                    ..
                }
            ),
            "dreamed: {report:?}"
        );
        let mind = memory.mind().expect("read").expect("written");
        assert_eq!(
            memory
                .apply_dream(&job.id, reply(&draft("x", "y")), now)
                .expect("applied")
                .outcome,
            Outcome::Settled,
            "a retried apply does not write twice"
        );
        assert_eq!(
            memory.mind().expect("read").map(|again| again.revision),
            Some(mind.revision),
            "MIND.md is unchanged"
        );
        let timed = memory.agenda(now).expect("agenda").timed;
        assert_eq!(
            timed.first().map(|item| item.id.as_str()),
            Some("eggs"),
            "the nudge is scheduled"
        );
        memory
            .record_fired("eggs", Some(at("2026-10-31T08:00")), at("2026-10-31T08:00"))
            .expect("recorded");
        memory
            .record_fired("eggs", Some(at("2026-10-31T08:00")), at("2026-10-31T08:01"))
            .expect("recorded twice");
        assert_eq!(
            crate::fired::read(&scratch.path).expect("ledger").len(),
            1,
            "recording twice writes once"
        );
        memory
            .note(
                "Arcee",
                "Skipped breakfast once more today",
                at("2026-10-31T09:00"),
            )
            .expect("saved");
        let later = at("2026-10-31T09:30");
        let job = memory
            .dream_job(Trigger::Compact, None, later)
            .expect("job")
            .expect("one note is enough before compaction");
        let report = memory
            .apply_dream(
                &job.id,
                reply(&draft("omelette", "Suggest an omelette")),
                later,
            )
            .expect("applied");
        assert!(
            report
                .dropped
                .iter()
                .any(|dropped| dropped.line.starts_with("eggs |")),
            "renamed, it keeps its old id and is dropped as fired: {report:?}"
        );
        assert!(
            memory.agenda(later).expect("agenda").timed.is_empty(),
            "it does not fire twice"
        );
    }

    #[test]
    fn a_stale_dream_answer_is_refused() {
        let scratch = Scratch::new();
        let memory = memory(&scratch, &NOTES);
        let now = at("2026-10-31T03:00");
        let old = memory
            .dream_job(Trigger::Nightly, None, now)
            .expect("job")
            .expect("due");
        // Another dream reads only notes 0 and 1 first.
        let shorter = dream::DreamId {
            first: 0,
            end: 2,
            trigger: Trigger::Nightly,
        };
        let early = "<mind>\n## Durable\n- Lives in Mumbai (#0)\n## Active\n- Busy mornings (#1)\n## Hunches\n## Nudges\n</mind>\n";
        let report = memory
            .apply_dream(&shorter, reply(early), now)
            .expect("applied");
        assert!(
            matches!(report.outcome, Outcome::Dreamed { end: 2, .. }),
            "the shorter dream lands: {report:?}"
        );
        assert!(
            matches!(
                memory.apply_dream(&old.id, reply(&draft("eggs", "x")), now),
                Err(MemoryError::Conflict { .. })
            ),
            "an answer about notes 0 to 4 no longer fits a watermark at 2"
        );
        let next = memory
            .dream_job(Trigger::Nightly, None, now)
            .expect("job")
            .map(|job| job.id.first);
        assert_eq!(
            next, None,
            "three new notes are too few for a nightly dream"
        );
    }

    #[test]
    fn agenda_lists_sessions_and_events() {
        let scratch = Scratch::new();
        let memory = memory(&scratch, &NOTES);
        let now = at("2026-10-31T03:00");
        let job = memory
            .dream_job(Trigger::Nightly, None, now)
            .expect("job")
            .expect("due");
        let text = "<mind>\n## Durable\n## Active\n## Hunches\n## Nudges\n\
            - ask | next session | - | Ask about the move | Moved to Delhi (#3) | notify\n\
            - shop | event:arrived-home | - | Suggest an omelette | Bought eggs (#4) | notify\n</mind>\n";
        memory
            .apply_dream(&job.id, reply(text), now)
            .expect("applied");
        let agenda = memory.agenda(now).expect("agenda");
        assert_eq!(
            agenda.next_session.first().map(|item| &item.when),
            Some(&When::NextSession),
            "next session"
        );
        assert!(
            matches!(agenda.events.first().map(|item| &item.when), Some(When::Event(name)) if name == "arrived-home"),
            "waiting for an event"
        );
        assert!(
            matches!(
                memory.record_fired("two words", None, now),
                Err(MemoryError::Invalid { .. })
            ),
            "ids have no spaces"
        );
    }
}
