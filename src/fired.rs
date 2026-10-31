//! The fired-nudge ledger, `fired.log`: one `<id> <fired at>` line, with
//! the time it was due after it when it had one, for each nudge or reminder
//! handed to the agent, so each fires once across sessions and dreams. Dream reads it to drop or re-time nudges
//! that already fired. Entries older than [`KEEP_DAYS`] are pruned.

use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::path::Path;

use jiff::Span;
use jiff::civil::DateTime;

use crate::files;
use crate::notebook::NotebookError;
use crate::tags;

/// The ledger's file name in the notebook directory.
pub const FILE: &str = "fired.log";

/// Entries older than this many days are pruned, unless the caller says
/// the item they record could still fire.
pub const KEEP_DAYS: i64 = 30;

const LOCK: &str = ".fired.lock";

/// The ledger's lock, held from reading what is due to recording what was
/// handed over, so two briefs at once cannot both hand over one item.
/// Dropping it releases the lock.
#[derive(Debug)]
pub struct Held {
    _file: File,
}

/// Waits for the ledger's lock and takes it.
pub fn hold(dir: &Path) -> Result<Held, NotebookError> {
    let path = dir.join(LOCK);
    let io_error = |source| NotebookError::Io {
        path: path.clone(),
        source,
    };
    let file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(&path)
        .map_err(io_error)?;
    file.lock().map_err(io_error)?;
    Ok(Held { _file: file })
}

/// Records that each of `handed` (an id, and the time it was due if it was
/// timed) fired at `now`. Entries older than [`KEEP_DAYS`] are pruned unless
/// `still_needed` says their item could fire again without them.
pub fn record(
    _held: &Held,
    dir: &Path,
    handed: &[(String, Option<DateTime>)],
    now: DateTime,
    still_needed: impl Fn(&Fired) -> bool,
) -> Result<(), NotebookError> {
    if handed.is_empty() {
        return Ok(());
    }
    let since = now.checked_sub(Span::new().days(KEEP_DAYS)).unwrap_or(now);
    let mut text = String::new();
    for entry in read(dir)? {
        if entry.at >= since || still_needed(&entry) {
            text.push_str(&line(&entry));
            text.push('\n');
        }
    }
    for (id, due) in handed {
        text.push_str(&line(&Fired {
            id: id.clone(),
            at: now,
            due: *due,
        }));
        text.push('\n');
    }
    // The ledger is small, so it is rewritten whole: a crash leaves the old
    // ledger or the new one, never a torn line.
    files::replace(&dir.join(FILE), &text)
}

/// One fired nudge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fired {
    /// The nudge's id, or `r<note id>` for a reminder.
    pub id: String,
    /// When it fired.
    pub at: DateTime,
    /// When it was due, for a timed nudge.
    pub due: Option<DateTime>,
}

/// Every entry of the ledger in `dir`, oldest first. A missing ledger is
/// empty; a line that does not parse is skipped, since nothing but Mnem
/// writes the file and a torn last line must not hide the rest.
pub fn read(dir: &Path) -> Result<Vec<Fired>, NotebookError> {
    let path = dir.join(FILE);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => return Err(NotebookError::Io { path, source }),
    };
    let mut entries = Vec::new();
    for line in text.lines() {
        if let Some(entry) = parse_line(line) {
            entries.push(entry);
        }
    }
    Ok(entries)
}

/// Whether `id`, due at `due` if it is timed, has fired. A timed nudge
/// that Dream moved to a new time is new again: an entry counts only for
/// the time it was due.
pub fn has_fired(ledger: &[Fired], id: &str, due: Option<DateTime>) -> bool {
    for entry in ledger {
        if entry.id != id {
            continue;
        }
        let same_time = match (due, entry.due) {
            (None, _) => true,
            (Some(due), Some(fired_for)) => due == fired_for,
            // Fired without a due time: it covers what was due by then.
            (Some(due), None) => entry.at >= due,
        };
        if same_time {
            return true;
        }
    }
    false
}

/// The ledger line for `entry`.
pub fn line(entry: &Fired) -> String {
    match entry.due {
        Some(due) => format!("{} {} {}", entry.id, minutes(entry.at), minutes(due)),
        None => format!("{} {}", entry.id, minutes(entry.at)),
    }
}

/// `at` written `YYYY-MM-DDTHH:MM`, the form tags and MIND.md use.
pub fn minutes(at: DateTime) -> String {
    at.strftime("%Y-%m-%dT%H:%M").to_string()
}

fn parse_line(line: &str) -> Option<Fired> {
    let mut fields = line.split(' ');
    let id = fields.next().filter(|id| !id.is_empty())?;
    let at = tags::date_time(fields.next()?)?;
    let due = match fields.next() {
        Some(due) => Some(tags::date_time(due)?),
        None => None,
    };
    if fields.next().is_some() {
        return None;
    }
    Some(Fired {
        id: id.to_owned(),
        at,
        due,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::fired;
    use crate::tags;
    use crate::testing::Scratch;

    #[test]
    fn recording_appends_and_prunes() {
        let scratch = Scratch::new();
        fs::create_dir_all(&scratch.path).expect("dir");
        let old = tags::date_time("2026-09-01T08:00").expect("a time");
        let now = tags::date_time("2026-10-31T08:00").expect("a time");
        let held = fired::hold(&scratch.path).expect("the lock");
        let handed = [("old".to_owned(), None), ("kept".to_owned(), None)];
        fired::record(&held, &scratch.path, &handed, old, |_| false).expect("recorded");
        let due = tags::date_time("2026-10-31T07:30").expect("a time");
        fired::record(
            &held,
            &scratch.path,
            &[("a".to_owned(), None), ("r7".to_owned(), Some(due))],
            now,
            |entry| entry.id == "kept",
        )
        .expect("recorded");
        assert_eq!(
            fs::read_to_string(scratch.path.join(fired::FILE)).expect("the ledger"),
            "kept 2026-09-01T08:00\na 2026-10-31T08:00\nr7 2026-10-31T08:00 2026-10-31T07:30\n",
            "entries past 30 days are pruned, unless their item could still fire"
        );
        let ledger = fired::read(&scratch.path).expect("readable");
        assert!(
            fired::has_fired(&ledger, "r7", Some(due)),
            "fired for its time"
        );
        assert!(
            !fired::has_fired(&ledger, "r7", Some(now)),
            "moved to another time, it is new again"
        );
    }

    #[test]
    fn reads_the_ledger_and_skips_torn_lines() {
        let scratch = Scratch::new();
        fs::create_dir_all(&scratch.path).expect("dir");
        assert!(
            fired::read(&scratch.path).is_ok_and(|entries| entries.is_empty()),
            "no ledger yet"
        );
        let at = tags::date_time("2026-10-30T08:00").expect("a time");
        fs::write(
            scratch.path.join(fired::FILE),
            format!(
                "{}\nr12 2026-10-30T09:\n",
                fired::line(&fired::Fired {
                    id: "tea".to_owned(),
                    at,
                    due: None
                })
            ),
        )
        .expect("write");
        let ledger = fired::read(&scratch.path).expect("readable");
        assert_eq!(ledger.len(), 1, "the torn line is skipped");
        assert!(fired::has_fired(&ledger, "tea", None), "fired at all");
        assert!(
            fired::has_fired(&ledger, "tea", Some(at)),
            "fired at its time"
        );
        let later = tags::date_time("2026-10-31T08:00").expect("a time");
        assert!(
            !fired::has_fired(&ledger, "tea", Some(later)),
            "a nudge moved later has not fired yet"
        );
    }
}
