//! What the tags in the log say, gathered in one scan.
//!
//! Nothing here is stored: every command that needs the index rebuilds it
//! from `LOG.txt`, so it can never disagree with the notes. One scan of a
//! year's notes (about 16 MB) takes milliseconds.

use std::collections::HashMap;

use crate::day::Day;
use crate::notebook::Notebook;
use crate::notebook::NotebookError;
use crate::record::Note;
use crate::tags;
use crate::tags::Reminder;

/// The tags of every note in a log.
#[derive(Clone, Debug, Default)]
pub struct TagIndex {
    /// Each corrected note, and the latest note that names it in
    /// `[updates #id]`. Only links to earlier notes count.
    corrected_by: HashMap<u64, u64>,
    /// Notes whose `[on …]` day differs from the day they were written.
    happened: Vec<(Day, u64)>,
    /// Every note's reminder, in log order.
    reminders: Vec<(u64, Reminder)>,
}

impl TagIndex {
    /// Reads every note's tags.
    pub fn build(notebook: &Notebook) -> Result<Self, NotebookError> {
        let mut index = Self::default();
        for note in notebook.scan()? {
            index.add(&note?);
        }
        Ok(index)
    }

    /// Adds one note, which must come after every note added before it.
    pub fn add(&mut self, note: &Note) {
        let found = tags::parse_lenient(&note.text);
        for target in found.updates {
            if target < note.id {
                self.corrected_by.insert(target, note.id);
            }
        }
        if let Some(day) = found.happened
            && day.as_str() != note.day
        {
            self.happened.push((day, note.id));
        }
        if let Some(reminder) = found.reminder {
            self.reminders.push((note.id, reminder));
        }
    }

    /// The newest note that corrects `id`, following chains of corrections:
    /// if 902 corrects 600, which corrects 455, the answer for 455 is 902.
    pub fn superseded_by(&self, id: u64) -> Option<u64> {
        let mut newest = *self.corrected_by.get(&id)?;
        // Each link points to a later note, so the chain ends.
        while let Some(&next) = self.corrected_by.get(&newest) {
            newest = next;
        }
        Some(newest)
    }

    /// The notes that say they happened on `day`, though written on another.
    pub fn happened_on(&self, day: &Day) -> Vec<u64> {
        let mut ids = Vec::new();
        for (happened, id) in &self.happened {
            if happened == day {
                ids.push(*id);
            }
        }
        ids
    }

    /// Every reminder, with the note that set it, in log order.
    pub fn reminders(&self) -> &[(u64, Reminder)] {
        &self.reminders
    }

    /// The note as `wake`, `recall` and `zoom` show it: [`Note::line`], with
    /// `[superseded by #id]` before the text when a later note corrects it.
    pub fn shown(&self, note: &Note) -> String {
        match self.superseded_by(note.id) {
            Some(newer) => format!(
                "#{} {} [superseded by #{newer}] {}",
                note.id, note.day, note.text
            ),
            None => note.line(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TagIndex;
    use crate::day::Day;
    use crate::record::Note;

    fn note(id: u64, day: &str, text: &str) -> Note {
        Note {
            id,
            day: day.to_owned(),
            text: text.to_owned(),
        }
    }

    fn index(notes: &[Note]) -> TagIndex {
        let mut index = TagIndex::default();
        for note in notes {
            index.add(note);
        }
        index
    }

    #[test]
    fn corrections_chain_to_the_newest() {
        let index = index(&[
            note(0, "2026-10-01", "Lives in Mumbai"),
            note(1, "2026-10-02", "[updates #0] Lives in Pune"),
            note(2, "2026-10-03", "[updates #1] Lives in Delhi"),
            note(3, "2026-10-04", "[updates #9] a link forward is ignored"),
        ]);
        assert_eq!(index.superseded_by(0), Some(2), "follows the chain");
        assert_eq!(index.superseded_by(1), Some(2), "direct");
        assert_eq!(index.superseded_by(2), None, "the newest stands");
        assert_eq!(
            index.superseded_by(9),
            None,
            "only earlier notes can be corrected"
        );
        assert_eq!(
            index.shown(&note(0, "2026-10-01", "Lives in Mumbai")),
            "#0 2026-10-01 [superseded by #2] Lives in Mumbai",
            "the mark goes before the text"
        );
        assert_eq!(
            index.shown(&note(2, "2026-10-03", "[updates #1] Lives in Delhi")),
            "#2 2026-10-03 [updates #1] Lives in Delhi",
            "unmarked notes are shown as memo shows them"
        );
    }

    #[test]
    fn happened_dates_and_reminders_are_listed() {
        let index = index(&[
            note(0, "2026-10-01", "Party [on 2026-09-28]"),
            note(1, "2026-10-01", "Lunch [on 2026-10-01]"),
            note(2, "2026-10-02", "Call mum [at 2026-10-03T18:00]"),
        ]);
        let day = Day::try_new("2026-09-28").expect("a date");
        assert_eq!(
            index.happened_on(&day),
            [0],
            "written later than it happened"
        );
        let same = Day::try_new("2026-10-01").expect("a date");
        assert!(
            index.happened_on(&same).is_empty(),
            "same-day tags add nothing"
        );
        assert_eq!(index.reminders().len(), 1, "one reminder");
        assert_eq!(
            index.reminders().first().map(|(id, _)| *id),
            Some(2),
            "set by #2"
        );
    }
}
