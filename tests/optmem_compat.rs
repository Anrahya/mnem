//! Byte-for-byte compatibility with OptMem's `memo` script.
//!
//! These tests run the real Python tool, so they are ignored by default. Run
//! them with the path to a checkout's `memo`:
//!
//! ```bash
//! OPTMEM_MEMO=/path/to/OptMem/memo cargo test --test optmem_compat -- --ignored
//! ```

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;

use mnem::cover;
use mnem::day::Day;
use mnem::record;
use mnem::record::LOG_REC;
use mnem::record::TREE_REC;
use mnem::text::TEXT_MAX;
use mnem::text::Text;

/// Helpers that run the Python tool. They sit in a `cfg(test)` module so
/// Clippy's test allowances cover their `expect` calls.
#[cfg(test)]
mod harness {
    use std::env;
    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;
    use std::process::Command;
    use std::process::Output;

    use mnem::text::TEXT_MAX;
    use mnem::text::Text;

    /// Loads `memo` as a module, so scripts can call its functions directly.
    const LOAD_MEMO: &str = "import sys\nfrom importlib.machinery import SourceFileLoader\nfrom importlib.util import module_from_spec, spec_from_file_location\nspec = spec_from_file_location('memo_cli', sys.argv[1], loader=SourceFileLoader('memo_cli', sys.argv[1]))\nm = module_from_spec(spec)\nspec.loader.exec_module(m)\n";

    pub(crate) fn memo_path() -> PathBuf {
        PathBuf::from(env::var_os("OPTMEM_MEMO").expect("OPTMEM_MEMO must point at OptMem's memo"))
    }

    /// Runs a Python script with `memo` loaded as `m`; returns its stdout.
    pub(crate) fn python(script: &str) -> String {
        let output = Command::new("python3")
            .arg("-I")
            .arg("-c")
            .arg(format!("{LOAD_MEMO}{script}"))
            .arg(memo_path())
            .output()
            .expect("python3 runs");
        assert!(
            output.status.success(),
            "python failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("python prints UTF-8")
    }

    /// Runs `memo <args>` against the memory in `dir`.
    pub(crate) fn run_memo(dir: &Path, args: &[&str]) -> Output {
        Command::new("python3")
            .arg(memo_path())
            .args(args)
            .env("MEMORY_DIR", dir)
            .output()
            .expect("memo runs")
    }

    /// A fresh, empty directory for one test.
    pub(crate) fn scratch(name: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!("mnem-compat-{}-{name}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).expect("old scratch directory is removable");
        }
        fs::create_dir_all(&dir).expect("scratch directory is creatable");
        dir
    }

    pub(crate) fn text(raw: &str) -> Text {
        Text::try_new(raw, TEXT_MAX).expect("test input is valid text")
    }
}

#[test]
#[ignore = "runs OptMem's Python memo; set OPTMEM_MEMO"]
fn cover_matches_memo() {
    let script = "Ts = list(range(0, 2100)) + [4095, 4096, 4097, 10000, 123457, 1000000, 2**31 + 5, 2**45 + 12345, 2**52 - 1]\nfor B in (0, 1, 2, 3, 4, 5, 7, 8, 96, 300):\n    for T in Ts:\n        print(T, B, ' '.join('%d:%d' % b for b in m.cover(T, B)))\n";
    let expected = harness::python(script);
    let mut lines = 0;
    for line in expected.lines() {
        let mut fields = line.splitn(3, ' ');
        let t: u64 = fields.next().and_then(|f| f.parse().ok()).expect("T");
        let budget: usize = fields.next().and_then(|f| f.parse().ok()).expect("B");
        let want = fields.next().unwrap_or("");
        let mut got = String::new();
        for span in cover::cover(t, budget).expect("covered") {
            if !got.is_empty() {
                got.push(' ');
            }
            write!(got, "{}:{}", span.lo(), span.hi()).expect("writing to a String");
        }
        assert_eq!(got, want, "cover({t}, {budget})");
        lines += 1;
    }
    assert_eq!(lines, 10 * 2109, "every case was compared");
}

#[test]
#[ignore = "runs OptMem's Python memo; set OPTMEM_MEMO"]
fn records_match_memo() {
    let dir = harness::scratch("records");
    let memory = dir.join("memory");
    assert!(
        harness::run_memo(&memory, &["init"]).status.success(),
        "memo init"
    );

    let two_byte = "é".repeat(140);
    let three_byte = format!("{}a", "€".repeat(93));
    let four_byte = "🚀".repeat(70);
    let inputs = [
        "plain note",
        "  padded on both sides  ",
        "\u{1C}separators trimmed\u{1F}",
        two_byte.as_str(),
        three_byte.as_str(),
        four_byte.as_str(),
        "emoji 🚀 and ünïcode",
        "a\ttab inside",
        "#9 2026-01-01 text that looks like a record",
    ];
    for input in inputs {
        let out = harness::run_memo(&memory, &["note", input]);
        assert!(out.status.success(), "memo note {input:?}");
    }

    let log = fs::read(memory.join("LOG.txt")).expect("LOG.txt exists");
    let width = usize::try_from(LOG_REC).expect("record width fits usize");
    assert_eq!(log.len(), width * inputs.len(), "one record per note");
    for (position, (bytes, input)) in (0_u64..).zip(log.chunks(width).zip(inputs)) {
        let note = record::decode_note(bytes).expect("memo writes valid records");
        assert_eq!(note.id, position, "ids follow positions");
        assert_eq!(
            note.text,
            harness::text(input).as_str(),
            "text of {input:?}"
        );
        let day = Day::try_new(&note.day).expect("memo writes real dates");
        assert_eq!(
            record::encode_note(note.id, &day, &harness::text(input)),
            bytes,
            "bytes of {input:?}"
        );
    }

    let summary = "  merged: é and 🚀  ";
    let out = harness::run_memo(&memory, &["nap", "0-1", summary]);
    assert!(out.status.success(), "memo nap 0-1");
    let tree = fs::read(memory.join("TREE").join("2")).expect("TREE/2 exists");
    let tree_width = usize::try_from(TREE_REC).expect("record width fits usize");
    let first = tree.get(..tree_width).expect("one summary record");
    assert_eq!(
        record::encode_summary(&harness::text(summary)),
        first,
        "summary bytes"
    );
    assert_eq!(
        record::decode_summary(first),
        Ok(Some(harness::text(summary).as_str().to_owned()))
    );

    let rejected = [
        String::new(),
        "   ".to_owned(),
        "two\nlines".to_owned(),
        "a".repeat(281),
        "é".repeat(141),
    ];
    for input in &rejected {
        let out = harness::run_memo(&memory, &["note", input]);
        assert!(!out.status.success(), "memo rejects {input:?}");
        assert!(
            Text::try_new(input, TEXT_MAX).is_err(),
            "Mnem rejects {input:?}"
        );
    }
    let after = fs::metadata(memory.join("LOG.txt")).expect("LOG.txt exists");
    assert_eq!(
        after.len(),
        u64::try_from(log.len()).expect("fits"),
        "rejected notes leave no record"
    );
    fs::remove_dir_all(&dir).expect("scratch directory is removable");
}

#[test]
#[ignore = "runs OptMem's Python memo; set OPTMEM_MEMO"]
fn trimming_matches_python_strip() {
    let listed =
        harness::python("print(' '.join(str(c) for c in range(0x110000) if chr(c).isspace()))");
    let spaces: HashSet<u32> = listed
        .split_whitespace()
        .map(|c| c.parse().expect("a code point"))
        .collect();
    assert!(spaces.contains(&0x1C), "Python treats U+001C as space");
    for code in 0..=0x10FFFF_u32 {
        let Some(c) = char::from_u32(code) else {
            continue;
        };
        let raw = format!("{c}x");
        let trimmed = Text::try_new(&raw, TEXT_MAX).expect("at least 'x' remains");
        let is_space = spaces.contains(&code);
        assert_eq!(trimmed.as_str() == "x", is_space, "U+{code:04X}");
    }
}

#[test]
#[ignore = "runs OptMem's Python memo; set OPTMEM_MEMO"]
fn days_match_python_strptime() {
    let mut candidates = Vec::new();
    for year in [0, 1, 1900, 2000, 2024, 2026, 9999] {
        for month in 0..=13 {
            for day in 0..=32 {
                candidates.push(format!("{year:04}-{month:02}-{day:02}"));
            }
        }
    }
    let script = format!(
        "import datetime\nfor s in {:?}:\n    try:\n        datetime.datetime.strptime(s, '%Y-%m-%d')\n        print(s)\n    except ValueError:\n        pass\n",
        candidates
    );
    let valid: HashSet<String> = harness::python(&script)
        .lines()
        .map(str::to_owned)
        .collect();
    assert!(valid.contains("2024-02-29"), "Python accepts a leap day");
    for candidate in &candidates {
        assert_eq!(
            Day::try_new(candidate).is_ok(),
            valid.contains(candidate),
            "{candidate}"
        );
    }
}
