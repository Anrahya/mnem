//! Mnem's inline tags, written inside a note's text so OptMem's record format
//! does not change:
//!
//! - `[on 2026-10-23]`: the day the event happened, when it is not the day
//!   the note was written (the two-date model of Supermemory);
//! - `[updates #455]`: this note corrects note 455 (Supermemory and
//!   Graphiti's update links); a note may correct several;
//! - `[at 2026-10-30T14:00]`, optionally `[at 2026-10-30T14:00 until 16:00]`
//!   or `until 2026-10-31T09:00`: a reminder at that local time.
//!
//! A bracket is a tag only when it looks like one: the keyword, a space, and
//! then a digit (`on`, `at`) or `#` (`updates`). Other brackets, such as
//! "[on the bus]", are ordinary text. `note` and `import` check tags strictly
//! with [`parse`]; everything that reads the log uses [`parse_lenient`],
//! which skips malformed tags, because notes written by `memo` were never
//! checked.

use std::error::Error;
use std::fmt;

use jiff::civil::Date;
use jiff::civil::DateTime;
use jiff::civil::Time;

use crate::day::Day;

/// The tags found in one note.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tags {
    /// When the event happened, if the note says.
    pub happened: Option<Day>,
    /// The notes this one corrects, in the order written.
    pub updates: Vec<u64>,
    /// The reminder the note asks for, if any.
    pub reminder: Option<Reminder>,
}

/// A reminder at a local time, optionally good only until a later time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reminder {
    /// When it is due.
    pub at: DateTime,
    /// After this it is stale and is not given to the agent.
    pub until: Option<DateTime>,
}

/// Why a tag is malformed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TagError {
    /// `[on …]` does not hold a real `YYYY-MM-DD` date.
    BadDay { tag: String },
    /// `[updates …]` does not hold `#` and a note id.
    BadId { tag: String },
    /// `[at …]` does not hold `YYYY-MM-DDTHH:MM`, an optional `until` with
    /// `HH:MM` or a full time, or the `until` is not after the `at`.
    BadTime { tag: String },
    /// A note may have only one `[on …]` and one `[at …]`.
    Repeated { tag: String },
}

/// The tags in `text`, rejecting any that are malformed or repeated.
pub fn parse(text: &str) -> Result<Tags, TagError> {
    let mut tags = Tags::default();
    for (tag, keyword, value) in candidates(text) {
        match keyword {
            Keyword::On => {
                let day = Day::try_new(value).map_err(|_not_a_day| TagError::BadDay {
                    tag: tag.to_owned(),
                })?;
                if tags.happened.replace(day).is_some() {
                    return Err(TagError::Repeated {
                        tag: tag.to_owned(),
                    });
                }
            }
            Keyword::Updates => {
                let id = note_id(value).ok_or_else(|| TagError::BadId {
                    tag: tag.to_owned(),
                })?;
                tags.updates.push(id);
            }
            Keyword::At => {
                let reminder = reminder(value).ok_or_else(|| TagError::BadTime {
                    tag: tag.to_owned(),
                })?;
                if tags.reminder.replace(reminder).is_some() {
                    return Err(TagError::Repeated {
                        tag: tag.to_owned(),
                    });
                }
            }
        }
    }
    Ok(tags)
}

/// The well-formed tags in `text`; malformed ones are skipped, and of a
/// repeated `[on …]` or `[at …]` the first counts.
pub fn parse_lenient(text: &str) -> Tags {
    let mut tags = Tags::default();
    for (_, keyword, value) in candidates(text) {
        match keyword {
            Keyword::On => {
                if tags.happened.is_none() {
                    tags.happened = Day::try_new(value).ok();
                }
            }
            Keyword::Updates => tags.updates.extend(note_id(value)),
            Keyword::At => {
                if tags.reminder.is_none() {
                    tags.reminder = reminder(value);
                }
            }
        }
    }
    tags
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Keyword {
    On,
    Updates,
    At,
}

/// Every bracket in `text` that looks like a tag: the whole tag, its
/// keyword, and the value after the keyword, trimmed.
fn candidates(text: &str) -> Vec<(&str, Keyword, &str)> {
    let mut found = Vec::new();
    let mut rest = text;
    while let Some((_, after_open)) = rest.split_once('[') {
        let Some((inside, after_close)) = after_open.split_once(']') else {
            break;
        };
        rest = after_close;
        let Some((word, value)) = inside.split_once(' ') else {
            continue;
        };
        let value = value.trim();
        let keyword = match word {
            "on" if value.starts_with(|c: char| c.is_ascii_digit()) => Keyword::On,
            "updates" if value.starts_with('#') => Keyword::Updates,
            "at" if value.starts_with(|c: char| c.is_ascii_digit()) => Keyword::At,
            _ => continue,
        };
        found.push((inside, keyword, value));
    }
    found
}

/// `#` and a note id in ASCII digits.
fn note_id(value: &str) -> Option<u64> {
    let digits = value.strip_prefix('#')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok()
}

/// `YYYY-MM-DDTHH:MM`, optionally followed by ` until ` and `HH:MM` (the
/// same day) or a full `YYYY-MM-DDTHH:MM`, which must be later.
fn reminder(value: &str) -> Option<Reminder> {
    let (at, until) = match value.split_once(" until ") {
        Some((at, until)) => (at.trim(), Some(until.trim())),
        None => (value, None),
    };
    let at = date_time(at)?;
    let until = match until {
        None => None,
        Some(until) => {
            let until = match clock(until) {
                Some(time) => DateTime::from_parts(at.date(), time),
                None => date_time(until)?,
            };
            if until <= at {
                return None;
            }
            Some(until)
        }
    };
    Some(Reminder { at, until })
}

/// A local time written `YYYY-MM-DDTHH:MM`.
pub fn date_time(value: &str) -> Option<DateTime> {
    let (day, time) = value.split_once('T')?;
    let date = date(&Day::try_new(day).ok()?)?;
    Some(DateTime::from_parts(date, clock(time)?))
}

/// A wall-clock time written `HH:MM`.
fn clock(value: &str) -> Option<Time> {
    let (hour, minute) = value.split_once(':')?;
    let hour = two_digits(hour)?;
    let minute = two_digits(minute)?;
    Time::new(hour, minute, 0, 0).ok()
}

fn two_digits(value: &str) -> Option<i8> {
    if value.len() != 2 || !value.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    value.parse::<i8>().ok()
}

/// The calendar date of a [`Day`].
pub fn date(day: &Day) -> Option<Date> {
    let mut parts = day.as_str().split('-');
    let year = parts.next()?.parse::<i16>().ok()?;
    let month = parts.next()?.parse::<i8>().ok()?;
    let day = parts.next()?.parse::<i8>().ok()?;
    Date::new(year, month, day).ok()
}

impl fmt::Display for TagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadDay { tag } => {
                write!(f, "[{tag}] is not a real date; write it as [on YYYY-MM-DD]")
            }
            Self::BadId { tag } => {
                write!(f, "[{tag}] does not name a note; write it as [updates #id]")
            }
            Self::BadTime { tag } => write!(
                f,
                "[{tag}] is not a reminder time; write it as [at YYYY-MM-DDTHH:MM], optionally followed by until HH:MM or a later YYYY-MM-DDTHH:MM"
            ),
            Self::Repeated { tag } => write!(
                f,
                "[{tag}] repeats a tag a note may hold once; keep one [on …] and one [at …] per note"
            ),
        }
    }
}

impl Error for TagError {}

#[cfg(test)]
mod tests {
    use jiff::civil::DateTime;

    use super::Reminder;
    use super::TagError;
    use super::Tags;
    use crate::day::Day;
    use crate::tags;

    fn at(value: &str) -> DateTime {
        tags::date_time(value).expect("a valid time")
    }

    #[test]
    fn reads_every_kind_of_tag() {
        let parsed = tags::parse(
            "[updates #4] [updates #9] Moved the dentist [on 2026-10-23] to Friday [at 2026-10-30T14:00 until 16:00]",
        )
        .expect("valid tags");
        assert_eq!(
            parsed,
            Tags {
                happened: Some(Day::try_new("2026-10-23").expect("a date")),
                updates: vec![4, 9],
                reminder: Some(Reminder {
                    at: at("2026-10-30T14:00"),
                    until: Some(at("2026-10-30T16:00")),
                }),
            },
            "all three kinds"
        );
        let later = tags::parse("[at 2026-10-30T22:00 until 2026-10-31T08:00]").expect("valid");
        assert_eq!(
            later.reminder.and_then(|reminder| reminder.until),
            Some(at("2026-10-31T08:00")),
            "until may be another day"
        );
    }

    #[test]
    fn ordinary_brackets_are_text() {
        for text in [
            "met her [on the bus]",
            "[at home] all day",
            "[updates] later",
            "[sic]",
            "unclosed [on 2026-10-01",
        ] {
            assert_eq!(
                tags::parse(text),
                Ok(Tags::default()),
                "{text:?} has no tags"
            );
        }
    }

    #[test]
    fn malformed_tags_are_rejected_when_written() {
        let cases = [
            ("[on 2026-02-30]", "BadDay"),
            ("[updates #four]", "BadId"),
            ("[updates #]", "BadId"),
            ("[at 2026-10-30 14:00]", "BadTime"),
            ("[at 2026-10-30T25:00]", "BadTime"),
            ("[at 2026-10-30T14:00 until 13:00]", "BadTime"),
            ("[on 2026-10-01] [on 2026-10-02]", "Repeated"),
            ("[at 2026-10-30T14:00] [at 2026-10-30T15:00]", "Repeated"),
        ];
        for (text, kind) in cases {
            let error = tags::parse(text).expect_err("malformed");
            let matched = match error {
                TagError::BadDay { .. } => "BadDay",
                TagError::BadId { .. } => "BadId",
                TagError::BadTime { .. } => "BadTime",
                TagError::Repeated { .. } => "Repeated",
            };
            assert_eq!(matched, kind, "{text:?}");
        }
    }

    #[test]
    fn reading_skips_what_writing_would_reject() {
        let parsed = tags::parse_lenient("[on 2026-02-30] [updates #7] [on 2026-10-01] [at nope]");
        assert_eq!(
            parsed.happened,
            Some(Day::try_new("2026-10-01").expect("a date")),
            "the first well-formed date"
        );
        assert_eq!(parsed.updates, [7], "good links are kept");
        assert_eq!(parsed.reminder, None, "a bad reminder is skipped");
    }
}
