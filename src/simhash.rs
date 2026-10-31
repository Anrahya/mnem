//! Near-duplicate detection for notes: a 64-bit SimHash of each note's
//! words, compared by Hamming distance.

/// Notes whose fingerprints differ in at most this many bits are treated as
/// repeats.
pub const NEAR: u32 = 3;

/// Notes with fewer words get no fingerprint: a single changed word flips
/// too many bits of a short one, and texts with no words would all match.
pub const MIN_WORDS: usize = 4;

/// The SimHash of `text`'s words, ignoring case and punctuation, or `None`
/// when it has fewer than [`MIN_WORDS`] words. Each word is hashed with
/// 64-bit FNV-1a. Words are split at anything that is not a letter or digit,
/// so text written without spaces, such as Chinese, counts as one word.
pub fn fingerprint(text: &str) -> Option<u64> {
    let mut votes = [0_i64; 64];
    let mut words: usize = 0;
    let lower = text.to_lowercase();
    for word in lower.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        words = words.saturating_add(1);
        let hash = fnv1a(word);
        for (bit, vote) in (0_u32..).zip(votes.iter_mut()) {
            if hash >> bit & 1 == 1 {
                *vote += 1;
            } else {
                *vote -= 1;
            }
        }
    }
    if words < MIN_WORDS {
        return None;
    }
    let mut print = 0_u64;
    for (bit, vote) in (0_u32..).zip(votes) {
        if vote > 0 {
            print |= 1 << bit;
        }
    }
    Some(print)
}

/// How many bits two fingerprints differ in.
pub fn distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

fn fnv1a(word: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in word.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use crate::simhash;

    fn distance(a: &str, b: &str) -> u32 {
        let print = |text| simhash::fingerprint(text).expect("enough words");
        simhash::distance(print(a), print(b))
    }

    #[test]
    fn repeats_are_near_and_different_notes_are_not() {
        assert_eq!(
            distance(
                "Dentist moved to Friday 3pm",
                "dentist moved to friday, 3pm!"
            ),
            0,
            "case and punctuation are ignored"
        );
        assert!(
            distance(
                "The owner prefers green tea in the morning, no sugar, and black coffee after lunch on workdays",
                "The owner prefers green tea in the morning, no sugar, and black coffee after lunch on weekdays"
            ) <= simhash::NEAR,
            "one word changed in a long note is a repeat"
        );
        assert!(
            distance(
                "Dentist moved to Friday 3pm",
                "Bought eggs and spinach for breakfast"
            ) > simhash::NEAR,
            "unrelated notes are far apart"
        );
    }

    #[test]
    fn short_notes_have_no_fingerprint() {
        for text in ["", "👍", "✅ done", "bought eggs today"] {
            assert_eq!(
                simhash::fingerprint(text),
                None,
                "{text:?} has too few words to compare"
            );
        }
    }
}
