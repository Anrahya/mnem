//! The fired-nudge ledger, `fired.log`: one `<id> <YYYY-MM-DDTHH:MM>` line
//! for each nudge or reminder handed to the agent, so each fires once
//! across sessions and dreams. Dream reads it to drop or re-time nudges
//! that already fired.

use std::fs;
use std::io;
use std::path::Path;

use jiff::civil::DateTime;

use crate::notebook::NotebookError;
use crate::tags;

/// The ledger's file name in the notebook directory.
pub const FILE: &str = "fired.log";

/// One fired nudge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fired {
    /// The nudge's id, or `r<note id>` for a reminder.
    pub id: String,
    /// When it fired.
    pub at: DateTime,
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

/// Whether `id` fired at or after `since` (at any time when `since` is
/// `None`).
pub fn has_fired(ledger: &[Fired], id: &str, since: Option<DateTime>) -> bool {
    for entry in ledger {
        if entry.id == id && since.is_none_or(|since| entry.at >= since) {
            return true;
        }
    }
    false
}

/// The ledger line for `id` firing at `at`.
pub fn line(id: &str, at: DateTime) -> String {
    format!("{id} {}", minutes(at))
}

/// `at` written `YYYY-MM-DDTHH:MM`, the form tags and MIND.md use.
pub fn minutes(at: DateTime) -> String {
    at.strftime("%Y-%m-%dT%H:%M").to_string()
}

fn parse_line(line: &str) -> Option<Fired> {
    let (id, at) = line.split_once(' ')?;
    if id.is_empty() {
        return None;
    }
    Some(Fired {
        id: id.to_owned(),
        at: tags::date_time(at.trim())?,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::fired;
    use crate::tags;
    use crate::testing::Scratch;

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
            format!("{}\nr12 2026-10-30T09:\n", fired::line("tea", at)),
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
