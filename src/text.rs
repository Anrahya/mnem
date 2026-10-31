//! Note and summary text, validated the way OptMem's `check` validates it.

use std::error::Error;
use std::fmt;

/// The longest text a record can hold, in bytes. OptMem derives it as
/// `min(TREE_REC - 8, LOG_REC - 40)`.
pub const TEXT_MAX: usize = 280;

/// One line of note or summary text: trimmed, non-empty, without line
/// breaks, and at most [`TEXT_MAX`] bytes of UTF-8.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Text(String);

/// Why text was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextError {
    /// Nothing is left after trimming.
    Empty,
    /// The text contains `\n` or `\r`; `lines` counts it as OptMem does.
    MultiLine { lines: usize },
    /// The trimmed text is longer than the limit, in bytes.
    TooLong { bytes: usize, limit: usize },
}

impl Text {
    /// Validates `raw` as OptMem does: trims whitespace from both ends, then
    /// rejects empty text, any `\n` or `\r`, and text longer than `limit`
    /// bytes. A `limit` above [`TEXT_MAX`] is lowered to it, because longer
    /// text does not fit a record.
    pub fn try_new(raw: &str, limit: usize) -> Result<Self, TextError> {
        let limit = limit.min(TEXT_MAX);
        let trimmed = raw.trim_matches(is_python_space);
        if trimmed.is_empty() {
            return Err(TextError::Empty);
        }
        if trimmed.contains(['\n', '\r']) {
            let lines = trimmed.matches('\n').count() + 1;
            return Err(TextError::MultiLine { lines });
        }
        let bytes = trimmed.len();
        if bytes > limit {
            return Err(TextError::TooLong { bytes, limit });
        }
        Ok(Self(trimmed.to_owned()))
    }

    /// The validated text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// `raw` made to fit as one line: line breaks become spaces, the ends
    /// are trimmed, and the text is cut at a character boundary to at most
    /// `limit` bytes (and never more than [`TEXT_MAX`]). Text that leaves
    /// nothing becomes `-`, so `limit` must be at least 1.
    pub(crate) fn cut(raw: &str, limit: usize) -> Self {
        let limit = limit.min(TEXT_MAX);
        let flat = raw.replace(['\n', '\r'], " ");
        let mut kept = String::new();
        for c in flat.trim_matches(is_python_space).chars() {
            if kept.len().saturating_add(c.len_utf8()) > limit {
                break;
            }
            kept.push(c);
        }
        let kept = kept.trim_end_matches(is_python_space);
        if kept.is_empty() {
            debug_assert!(limit >= 1, "config allows no ENTRY_CHARS below 1");
            return Self("-".to_owned());
        }
        Self(kept.to_owned())
    }
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "nothing to save: the text is blank"),
            Self::MultiLine { .. } => {
                write!(f, "the text has a line break; notes are a single line")
            }
            Self::TooLong { bytes, limit } => {
                write!(
                    f,
                    "the text is {bytes} bytes; the most a note holds is {limit}"
                )
            }
        }
    }
}

impl Error for TextError {}

/// Whether Python's `str.isspace` is true for `c`. OptMem trims with Python's
/// `strip`, which also treats U+001C to U+001F as whitespace; Rust's
/// `char::is_whitespace` does not, so the set is spelled out here.
pub(crate) fn is_python_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n'
            | '\u{0B}'
            | '\u{0C}'
            | '\r'
            | '\u{1C}'..='\u{1F}'
            | ' '
            | '\u{85}'
            | '\u{A0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}'
    )
}

#[cfg(test)]
mod tests {
    use super::TEXT_MAX;
    use super::Text;
    use super::TextError;

    #[test]
    fn trims_python_whitespace_including_separators() {
        let text = Text::try_new("\u{1C} hello \u{3000}", TEXT_MAX).expect("valid text");
        assert_eq!(text.as_str(), "hello");
    }

    #[test]
    fn rejects_empty_and_whitespace_only() {
        assert_eq!(Text::try_new("", TEXT_MAX), Err(TextError::Empty));
        assert_eq!(Text::try_new(" \t\u{A0}", TEXT_MAX), Err(TextError::Empty));
    }

    #[test]
    fn rejects_line_breaks_counting_newlines() {
        assert_eq!(
            Text::try_new("a\nb\nc", TEXT_MAX),
            Err(TextError::MultiLine { lines: 3 })
        );
        assert_eq!(
            Text::try_new("a\rb", TEXT_MAX),
            Err(TextError::MultiLine { lines: 1 })
        );
    }

    #[test]
    fn limit_counts_utf8_bytes() {
        let two_byte = "é".repeat(140);
        assert!(Text::try_new(&two_byte, TEXT_MAX).is_ok(), "280 bytes fits");
        let over = format!("{two_byte}a");
        assert_eq!(
            Text::try_new(&over, TEXT_MAX),
            Err(TextError::TooLong {
                bytes: 281,
                limit: 280
            })
        );
    }

    #[test]
    fn limit_above_record_capacity_is_lowered() {
        let long = "a".repeat(281);
        assert_eq!(
            Text::try_new(&long, 1000),
            Err(TextError::TooLong {
                bytes: 281,
                limit: 280
            })
        );
    }

    #[test]
    fn cut_makes_one_line_that_fits() {
        assert_eq!(
            Text::cut(" one\ntwo\r\nthree ", 280).as_str(),
            "one two  three",
            "breaks become spaces"
        );
        assert_eq!(
            Text::cut("ab\u{1F680}", 5).as_str(),
            "ab",
            "never splits a character"
        );
        assert_eq!(
            Text::cut("abc   def", 5).as_str(),
            "abc",
            "no trailing space"
        );
        assert_eq!(Text::cut("\u{1F680}", 2).as_str(), "-", "nothing fits");
        assert_eq!(
            Text::cut(&"z".repeat(400), 1000).as_str().len(),
            TEXT_MAX,
            "never past TEXT_MAX"
        );
    }
}
