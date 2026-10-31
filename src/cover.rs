//! Which notes and summaries `wake` shows for a log.
//!
//! `wake` must fit a whole log into a fixed number of lines. It does that by
//! tiling the log with aligned power-of-two spans: the newest notes appear
//! one per line, and older stretches appear as ever larger summary blocks.
//! The result must equal OptMem's choice exactly, including where its
//! floating-point search lands, so that both tools print the same lines for
//! the same notebook; `tests/optmem_compat.rs` checks this against `memo`.

use std::error::Error;
use std::fmt;

use crate::span::Span;

/// The largest log the cover handles. Every count below it is exactly
/// representable as `f64`, which the search for a tiling depends on.
pub const COVER_MAX_NOTES: u64 = 1 << 52;

/// Why a cover could not be computed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoverError {
    /// The log holds [`COVER_MAX_NOTES`] notes or more.
    TooManyNotes { notes: u64 },
}

/// The spans `wake` prints for a log of `notes` notes, oldest first.
///
/// When the log has no more notes than `budget`, every note gets its own
/// line. Otherwise the result aims for `budget` lines. It can exceed a very
/// small budget: a log can never be tiled with fewer spans than the number
/// of 1 bits in `notes`, so `cover(5, 1)` returns two spans.
pub fn cover(notes: u64, budget: usize) -> Result<Vec<Span>, CoverError> {
    if notes >= COVER_MAX_NOTES {
        return Err(CoverError::TooManyNotes { notes });
    }
    let fits_whole = u64::try_from(budget).map_or(true, |budget| notes <= budget);
    if fits_whole {
        let mut lines = Vec::new();
        for id in 0..notes {
            lines.push(Span::aligned(id, id + 1));
        }
        return Ok(lines);
    }
    // Search for the finest coarseness whose tiling still fits the budget.
    // The same 60 halvings of [0, 1] as OptMem, so both land on the same value.
    let (mut fits, mut too_fine) = (1.0_f64, 0.0_f64);
    for _ in 0..60 {
        let middle = (too_fine + fits) / 2.0;
        if tile(notes, middle).len() > budget {
            too_fine = middle;
        } else {
            fits = middle;
        }
    }
    let mut lines = tile(notes, fits);
    // Coarseness moves span sizes in powers of two, so the tiling can come in
    // under budget. Spend any spare lines by opening the newest block that is
    // still a summary, one at a time.
    while lines.len() < budget {
        let Some(newest_block) = lines.iter().rposition(|span| !span.is_note()) else {
            break;
        };
        let Some((older, newer)) = lines.get(newest_block).and_then(|span| span.halves()) else {
            break;
        };
        lines.splice(newest_block..=newest_block, [older, newer]);
    }
    Ok(lines)
}

/// Tiles the log `[0, notes)` for one coarseness value. A span is kept as one
/// line when it ends inside the log and its size is no more than
/// `coarseness` times its distance from the newest note; otherwise it is
/// split in two. Higher coarseness means bigger blocks and fewer lines.
/// `notes` must be below [`COVER_MAX_NOTES`].
fn tile(notes: u64, coarseness: f64) -> Vec<Span> {
    let mut whole_tree: u64 = 1;
    while whole_tree < notes {
        whole_tree *= 2;
    }
    let mut lines = Vec::new();
    let mut to_visit = vec![Span::aligned(0, whole_tree)];
    while let Some(span) = to_visit.pop() {
        if span.lo() >= notes {
            continue;
        }
        let age = exact_f64(notes - span.lo());
        let split = span.hi() > notes || exact_f64(span.size()) > coarseness * age;
        match span.halves() {
            Some((older, newer)) if split => {
                to_visit.push(newer);
                to_visit.push(older);
            }
            Some(_) | None => lines.push(span),
        }
    }
    lines.sort();
    lines
}

/// `n` as `f64`. Exact for every `n` below 2^53, so for every count this
/// module sees; a rounded conversion would move the search's comparisons.
fn exact_f64(n: u64) -> f64 {
    // Each half is below 2^32 by construction, so the conversions cannot fail.
    let high = u32::try_from(n >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(n & 0xffff_ffff).unwrap_or(u32::MAX);
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
}

impl fmt::Display for CoverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyNotes { notes } => {
                write!(f, "{notes} notes is more than wake can cover exactly")
            }
        }
    }
}

impl Error for CoverError {}

#[cfg(test)]
mod tests {
    use crate::span::Span;

    fn spans(notes: u64, budget: usize) -> Vec<(u64, u64)> {
        let lines = super::cover(notes, budget).expect("small logs are covered");
        let mut pairs = Vec::new();
        for span in lines {
            pairs.push((span.lo(), span.hi()));
        }
        pairs
    }

    #[test]
    fn small_logs_are_all_single_notes() {
        assert_eq!(spans(0, 96), Vec::<(u64, u64)>::new(), "empty log");
        assert_eq!(spans(3, 96), vec![(0, 1), (1, 2), (2, 3)], "three notes");
    }

    #[test]
    fn tiny_budgets_stop_at_one_span_per_set_bit() {
        assert_eq!(
            spans(5, 1),
            vec![(0, 4), (4, 5)],
            "5 = 0b101 needs two spans"
        );
        assert_eq!(
            spans(5, 0),
            vec![(0, 4), (4, 5)],
            "a zero budget gives the same"
        );
        assert_eq!(spans(7, 2).len(), 3, "7 = 0b111 needs three spans");
    }

    #[test]
    fn cover_tiles_the_log_within_budget() {
        for notes in [97_u64, 100, 1000, 4097, 50_000] {
            let lines = super::cover(notes, 96).expect("covered");
            assert!(lines.len() <= 96, "notes={notes}: {} lines", lines.len());
            let mut next = 0;
            for span in &lines {
                assert_eq!(span.lo(), next, "notes={notes}: spans must be contiguous");
                assert_eq!(
                    Span::try_new(span.lo(), span.hi()),
                    Ok(*span),
                    "notes={notes}: spans must be aligned"
                );
                next = span.hi();
            }
            assert_eq!(
                next, notes,
                "notes={notes}: spans must end at the log's end"
            );
            assert!(
                lines.last().is_some_and(|span| span.is_note()),
                "notes={notes}: the newest note stays single"
            );
        }
    }

    #[test]
    fn rejects_logs_too_large_to_cover_exactly() {
        assert!(
            super::cover(super::COVER_MAX_NOTES, 96).is_err(),
            "2^52 notes is past the exact range"
        );
        assert!(
            super::cover(super::COVER_MAX_NOTES - 1, 96).is_ok(),
            "one below the limit is covered"
        );
    }
}
