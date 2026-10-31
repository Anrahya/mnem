//! The calendar day a note is dated with, as OptMem writes it.

use std::error::Error;
use std::fmt;

/// A real calendar date written as `YYYY-MM-DD`, year 0001 to 9999. OptMem
/// dates notes to the day, with no time.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Day(String);

/// Why a day was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DayError {
    /// Not shaped `YYYY-MM-DD` with ASCII digits.
    Format,
    /// Shaped right, but not a date on the calendar (for example
    /// `2026-02-30` or year `0000`).
    NotADate,
}

impl Day {
    /// Validates `raw` as a `YYYY-MM-DD` date that exists on the Gregorian
    /// calendar, as Python's `strptime(raw, "%Y-%m-%d")` does for the
    /// zero-padded form OptMem writes.
    ///
    /// Stricter than `memo import`, which also takes a year written in
    /// non-ASCII digits such as `２０２６`. Mnem only writes ASCII dates.
    pub fn try_new(raw: &str) -> Result<Self, DayError> {
        let mut parts = raw.split('-');
        let (Some(year), Some(month), Some(day), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(DayError::Format);
        };
        let year = digits(year, 4).ok_or(DayError::Format)?;
        let month = digits(month, 2).ok_or(DayError::Format)?;
        let day = digits(day, 2).ok_or(DayError::Format)?;
        let valid =
            year >= 1 && (1..=12).contains(&month) && (1..=days_in(year, month)).contains(&day);
        if !valid {
            return Err(DayError::NotADate);
        }
        Ok(Self(raw.to_owned()))
    }

    /// The day as `YYYY-MM-DD`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format => write!(f, "expected a date shaped YYYY-MM-DD"),
            Self::NotADate => write!(f, "not a real calendar date"),
        }
    }
}

impl Error for DayError {}

/// The value of `field` if it is exactly `width` ASCII digits.
fn digits(field: &str, width: usize) -> Option<u32> {
    if field.len() != width || !field.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // At most four ASCII digits always parse as u32, so no error is lost here.
    field.parse::<u32>().ok()
}

fn days_in(year: u32, month: u32) -> u32 {
    let leap = (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400);
    match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

#[cfg(test)]
mod tests {
    use super::Day;
    use super::DayError;

    #[test]
    fn accepts_real_dates() {
        for raw in ["2026-10-31", "2000-02-29", "0001-01-01", "9999-12-31"] {
            let day = Day::try_new(raw).expect("a real date");
            assert_eq!(day.as_str(), raw);
        }
    }

    #[test]
    fn rejects_impossible_dates() {
        for raw in [
            "1900-02-29",
            "2026-02-30",
            "2026-13-01",
            "2026-00-10",
            "0000-01-01",
        ] {
            assert_eq!(Day::try_new(raw), Err(DayError::NotADate), "{raw}");
        }
    }

    #[test]
    fn rejects_other_shapes() {
        for raw in [
            "2026-1-01",
            "26-10-31",
            "2026/10/31",
            "2026-10-31-",
            "２０２６-10-31",
            "",
        ] {
            assert_eq!(Day::try_new(raw), Err(DayError::Format), "{raw}");
        }
    }
}
