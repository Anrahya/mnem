//! Aligned power-of-two ranges of notes: the nodes of OptMem's summary tree.

use std::error::Error;
use std::fmt;

use crate::record::TREE_REC;

/// The notes `[lo, hi)`, where `hi - lo` is a power of two and `lo` is a
/// multiple of it. A span of one note is a leaf; a longer span is a block,
/// stored as one summary line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    lo: u64,
    hi: u64,
}

/// Why a span or block id was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpanError {
    /// Not shaped `<lo>-<hi>` with ASCII digits.
    Format,
    /// The range is empty, not a power of two, or not aligned to its size.
    NotAligned,
    /// A block id names a single note, which has no summary.
    SingleNote,
    /// A bound does not fit in 64 bits.
    Overflow,
}

impl Span {
    /// The span `[lo, hi)`, if it is non-empty, its size is a power of two,
    /// and `lo` is a multiple of its size.
    pub fn try_new(lo: u64, hi: u64) -> Result<Self, SpanError> {
        let size = hi.checked_sub(lo).ok_or(SpanError::NotAligned)?;
        if !size.is_power_of_two() || !lo.is_multiple_of(size) {
            return Err(SpanError::NotAligned);
        }
        Ok(Self { lo, hi })
    }

    /// The single note `id`, or `None` if `id` is `u64::MAX`.
    pub fn try_note(id: u64) -> Option<Self> {
        let hi = id.checked_add(1)?;
        Some(Self { lo: id, hi })
    }

    /// Parses a block id the way `memo` prints it: `<lo>-<hi>`, inclusive at
    /// both ends, naming a block of at least two notes.
    ///
    /// Stricter than `memo`, which also takes non-ASCII digits and bounds
    /// past 64 bits; neither names a block that can exist in a real log.
    pub fn parse_block(raw: &str) -> Result<Self, SpanError> {
        let (lo, last) = raw.split_once('-').ok_or(SpanError::Format)?;
        let lo = number(lo)?;
        let hi = number(last)?.checked_add(1).ok_or(SpanError::Overflow)?;
        let span = Self::try_new(lo, hi)?;
        if span.is_note() {
            return Err(SpanError::SingleNote);
        }
        Ok(span)
    }

    /// The first note in the span.
    pub fn lo(self) -> u64 {
        self.lo
    }

    /// One past the last note in the span.
    pub fn hi(self) -> u64 {
        self.hi
    }

    /// How many notes the span covers; always a power of two.
    pub fn size(self) -> u64 {
        self.hi - self.lo
    }

    /// Whether the span is a single note rather than a summary block.
    pub fn is_note(self) -> bool {
        self.size() == 1
    }

    /// The two spans this one merges, or `None` for a single note.
    pub fn halves(self) -> Option<(Self, Self)> {
        if self.is_note() {
            return None;
        }
        let mid = self.lo + self.size() / 2;
        Some((
            Self {
                lo: self.lo,
                hi: mid,
            },
            Self {
                lo: mid,
                hi: self.hi,
            },
        ))
    }

    /// Where this block's summary starts in `TREE/<size>`, in bytes. `None`
    /// for a single note, which has no summary, or if the offset does not fit
    /// in 64 bits.
    pub fn tree_offset(self) -> Option<u64> {
        if self.is_note() {
            return None;
        }
        (self.lo / self.size()).checked_mul(TREE_REC)
    }

    /// A span the caller has already proven aligned.
    pub(crate) fn aligned(lo: u64, hi: u64) -> Self {
        debug_assert!(
            Self::try_new(lo, hi).is_ok(),
            "callers build spans by halving an aligned root, so they stay aligned"
        );
        Self { lo, hi }
    }
}

/// Prints a block as its id, `<lo>-<last>` inclusive, and a single note as
/// its number.
impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_note() {
            return write!(f, "{}", self.lo);
        }
        write!(f, "{}-{}", self.lo, self.hi - 1)
    }
}

impl fmt::Display for SpanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format => write!(f, "a block id is two numbers joined by a dash"),
            Self::NotAligned => write!(
                f,
                "not a block: its size must be a power of two and its start a multiple of it"
            ),
            Self::SingleNote => write!(f, "a single note is not a block"),
            Self::Overflow => write!(f, "block id is out of range"),
        }
    }
}

impl Error for SpanError {}

fn number(field: &str) -> Result<u64, SpanError> {
    if field.is_empty() || !field.bytes().all(|b| b.is_ascii_digit()) {
        return Err(SpanError::Format);
    }
    field
        .parse::<u64>()
        .map_err(|_overflow: std::num::ParseIntError| SpanError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::Span;
    use super::SpanError;

    #[test]
    fn parses_blocks_inclusively() {
        let span = Span::parse_block("16-31").expect("an aligned block");
        assert_eq!((span.lo(), span.hi(), span.size()), (16, 32, 16));
        assert_eq!(span.to_string(), "16-31");
    }

    #[test]
    fn accepts_aligned_and_rejects_unaligned() {
        assert_eq!(
            Span::parse_block("4-5"),
            Span::try_new(4, 6),
            "4-5 is aligned"
        );
        assert_eq!(
            Span::parse_block("5-6"),
            Err(SpanError::NotAligned),
            "starts mid-pair"
        );
        assert_eq!(
            Span::parse_block("0-2"),
            Err(SpanError::NotAligned),
            "three notes"
        );
        assert_eq!(
            Span::parse_block("8-3"),
            Err(SpanError::NotAligned),
            "backwards"
        );
    }

    #[test]
    fn rejects_single_notes_and_bad_shapes() {
        assert_eq!(Span::parse_block("7-7"), Err(SpanError::SingleNote));
        for raw in ["", "4", "4-", "-5", "a-b", "4-5-6", "+4-5", " 4-5"] {
            assert_eq!(Span::parse_block(raw), Err(SpanError::Format), "{raw:?}");
        }
        assert_eq!(
            Span::parse_block("0-18446744073709551615"),
            Err(SpanError::Overflow)
        );
    }

    #[test]
    fn halves_and_tree_offset() {
        let span = Span::parse_block("8-15").expect("an aligned block");
        let older = Span::try_new(8, 12).expect("aligned");
        let newer = Span::try_new(12, 16).expect("aligned");
        assert_eq!(span.halves(), Some((older, newer)), "8-15 splits at 12");
        assert_eq!(span.tree_offset(), Some(288), "second block of size 8");
        let leaf = Span::try_note(3).expect("small id");
        assert_eq!(leaf.halves(), None, "a note has no halves");
        assert_eq!(leaf.tree_offset(), None, "a note has no summary");
        assert_eq!(leaf.to_string(), "3", "a note prints as its number");
    }
}
