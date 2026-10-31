//! OptMem's fixed-width records: one note per `LOG.txt` record, one summary
//! per `TREE/<size>` record.
//!
//! A log record is the text `#<id> <YYYY-MM-DD> <text>`, padded with spaces
//! to `LOG_REC - 1` bytes and ended with a newline. A tree record is the
//! summary text padded the same way to `TREE_REC` bytes. Fixed widths make a
//! record's position its identity: note `i` starts at byte `i * LOG_REC`.

use std::error::Error;
use std::fmt;
use std::str;

use crate::day::Day;
use crate::text;
use crate::text::TEXT_MAX;
use crate::text::Text;

/// The width of one note record in `LOG.txt`, in bytes.
pub const LOG_REC: u64 = 320;

/// The width of one summary record in a `TREE/<size>` file, in bytes.
pub const TREE_REC: u64 = 288;

/// The longest `#<id> <YYYY-MM-DD> ` prefix: a `u64` id has at most 20 digits.
const NOTE_PREFIX_MAX: u64 = 1 + 20 + 1 + 10 + 1;

// Proves at compile time that encode_note and encode_summary never overflow
// a record: the longest text plus its prefix leaves room for the newline.
#[expect(
    clippy::as_conversions,
    reason = "TEXT_MAX is 280; u64 holds it exactly"
)]
const _: () = {
    assert!(
        NOTE_PREFIX_MAX + (TEXT_MAX as u64) < LOG_REC,
        "the longest note fits"
    );
    assert!((TEXT_MAX as u64) < TREE_REC, "the longest summary fits");
};

/// A note as read back from `LOG.txt`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Note {
    /// The id written in the record; it equals the record's position in a
    /// healthy log.
    pub id: u64,
    /// The day written in the record, as stored.
    pub day: String,
    /// The note text.
    pub text: String,
}

/// Why a record could not be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordError {
    /// The slice is not exactly one record wide.
    WrongWidth { expected: u64, actual: usize },
    /// The record is not valid UTF-8.
    NotUtf8,
    /// A log record does not start with `#<digits>`.
    BadId,
}

/// Encodes note `id` as one `LOG.txt` record of exactly [`LOG_REC`] bytes.
pub fn encode_note(id: u64, day: &Day, text: &Text) -> Vec<u8> {
    // Fits: see the compile-time check on NOTE_PREFIX_MAX.
    let line = format!("#{id} {} {}", day.as_str(), text.as_str());
    pad(line.into_bytes(), LOG_REC)
}

/// Decodes one `LOG.txt` record.
pub fn decode_note(record: &[u8]) -> Result<Note, RecordError> {
    let line = decode_line(record, LOG_REC)?;
    // Python's str.partition: a missing separator leaves the rest empty.
    let (head, rest) = line.split_once(' ').unwrap_or((line, ""));
    let (day, text) = rest.split_once(' ').unwrap_or((rest, ""));
    let digits = head.strip_prefix('#').ok_or(RecordError::BadId)?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(RecordError::BadId);
    }
    let id = digits
        .parse::<u64>()
        .map_err(|_overflow: std::num::ParseIntError| RecordError::BadId)?;
    Ok(Note {
        id,
        day: day.to_owned(),
        text: text.to_owned(),
    })
}

/// Encodes a summary as one `TREE/<size>` record of exactly [`TREE_REC`]
/// bytes.
pub fn encode_summary(text: &Text) -> Vec<u8> {
    // Fits: see the compile-time check on NOTE_PREFIX_MAX.
    pad(text.as_str().as_bytes().to_vec(), TREE_REC)
}

/// Decodes one `TREE/<size>` record. A blank record, which only a corrupt
/// write leaves, decodes to `None`. A summary not built yet has no record at
/// all; telling that apart is the caller's job, from the file's length.
pub fn decode_summary(record: &[u8]) -> Result<Option<String>, RecordError> {
    let line = decode_line(record, TREE_REC)?;
    Ok((!line.is_empty()).then(|| line.to_owned()))
}

/// Pads `bytes` with spaces to `width - 1` bytes and ends it with a newline.
fn pad(mut bytes: Vec<u8>, width: u64) -> Vec<u8> {
    let width = usize::try_from(width).unwrap_or(usize::MAX);
    debug_assert!(
        bytes.len() < width,
        "callers only pass text that fits the record"
    );
    bytes.resize(width.saturating_sub(1), b' ');
    bytes.push(b'\n');
    bytes
}

/// The record's text with the padding and newline trimmed, as Python's
/// `rstrip` trims it.
fn decode_line(record: &[u8], width: u64) -> Result<&str, RecordError> {
    let fits = u64::try_from(record.len()).is_ok_and(|len| len == width);
    if !fits {
        return Err(RecordError::WrongWidth {
            expected: width,
            actual: record.len(),
        });
    }
    let line = str::from_utf8(record).map_err(|_invalid: str::Utf8Error| RecordError::NotUtf8)?;
    Ok(line.trim_end_matches(text::is_python_space))
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongWidth { expected, actual } => {
                write!(f, "record is {actual} bytes, expected {expected}")
            }
            Self::NotUtf8 => write!(f, "record is not valid UTF-8"),
            Self::BadId => write!(f, "record does not start with #<id>"),
        }
    }
}

impl Error for RecordError {}

#[cfg(test)]
mod tests {
    use super::LOG_REC;
    use super::RecordError;
    use super::TREE_REC;
    use super::decode_note;
    use super::decode_summary;
    use super::encode_note;
    use super::encode_summary;
    use crate::day::Day;
    use crate::text::TEXT_MAX;
    use crate::text::Text;

    fn text(raw: &str) -> Text {
        Text::try_new(raw, TEXT_MAX).expect("valid test text")
    }

    #[test]
    fn note_record_layout() {
        let day = Day::try_new("2026-10-31").expect("a real date");
        let record = encode_note(7, &day, &text("ship it"));
        assert_eq!(u64::try_from(record.len()), Ok(LOG_REC));
        let prefix = b"#7 2026-10-31 ship it ";
        assert!(record.starts_with(prefix), "text comes first, then padding");
        assert_eq!(record.last(), Some(&b'\n'));
        assert_eq!(record.iter().filter(|&&b| b == b'\n').count(), 1);
    }

    #[test]
    fn note_round_trips_at_full_width() {
        let day = Day::try_new("2026-10-31").expect("a real date");
        let long = "é".repeat(140);
        let record = encode_note(u64::MAX, &day, &text(&long));
        let note = decode_note(&record).expect("a well-formed record");
        assert_eq!(note.id, u64::MAX);
        assert_eq!(note.day, "2026-10-31");
        assert_eq!(note.text, long);
    }

    #[test]
    fn decode_rejects_bad_records() {
        assert_eq!(
            decode_note(b"#1 2026-10-31 short\n"),
            Err(RecordError::WrongWidth {
                expected: LOG_REC,
                actual: 20
            })
        );
        let mut bad = encode_note(1, &Day::try_new("2026-10-31").expect("date"), &text("x"));
        if let Some(first) = bad.first_mut() {
            *first = b'1';
        }
        assert_eq!(decode_note(&bad), Err(RecordError::BadId));
        let mut invalid = vec![b' '; 319];
        invalid.push(b'\n');
        if let Some(first) = invalid.first_mut() {
            *first = 0xff;
        }
        assert_eq!(decode_note(&invalid), Err(RecordError::NotUtf8));
    }

    #[test]
    fn summary_round_trips_and_blank_is_none() {
        let record = encode_summary(&text("merged line"));
        assert_eq!(u64::try_from(record.len()), Ok(TREE_REC));
        assert_eq!(decode_summary(&record), Ok(Some("merged line".to_owned())));
        let mut blank = vec![b' '; 287];
        blank.push(b'\n');
        assert_eq!(decode_summary(&blank), Ok(None));
    }
}
