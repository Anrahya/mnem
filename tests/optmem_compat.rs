//! Byte-for-byte compatibility with OptMem's `memo` script.
//!
//! These tests run the real Python tool, so they are ignored by default. Run
//! them with the path to a checkout's `memo`:
//!
//! ```bash
//! OPTMEM_MEMO=/path/to/OptMem/memo cargo test --test optmem_compat -- --ignored
//! ```

use std::collections::HashSet;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::fs;
use std::fs::OpenOptions;
use std::io::BufRead as _;
use std::io::BufReader;
use std::io::Write as _;
use std::path::Path;
use std::process::Stdio;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use mnem::cover;
use mnem::day::Day;
use mnem::notebook::Notebook;
use mnem::notebook::PutOutcome;
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
    use std::ffi::OsStr;
    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;
    use std::process::Command;
    use std::process::Output;

    use mnem::day::Day;
    use mnem::span::Span;
    use mnem::text::TEXT_MAX;
    use mnem::text::Text;

    /// Loads `memo` as a module, so scripts can call its functions directly.
    const LOAD_MEMO: &str = "import sys\nfrom importlib.machinery import SourceFileLoader\nfrom importlib.util import module_from_spec, spec_from_file_location\nspec = spec_from_file_location('memo_cli', sys.argv[1], loader=SourceFileLoader('memo_cli', sys.argv[1]))\nm = module_from_spec(spec)\nspec.loader.exec_module(m)\n";

    pub(crate) fn memo_path() -> PathBuf {
        PathBuf::from(env::var_os("OPTMEM_MEMO").expect("OPTMEM_MEMO must point at OptMem's memo"))
    }

    /// Runs a Python script with `memo` loaded as `m`; returns its stdout.
    pub(crate) fn python(script: &str) -> String {
        python_with(script, &[])
    }

    /// The command that runs a Python script with `memo` loaded as `m`;
    /// `args` follow the memo path in `sys.argv`, so the first is
    /// `sys.argv[2]`.
    pub(crate) fn python_command(script: &str, args: &[&OsStr]) -> Command {
        let mut command = Command::new("python3");
        command
            .arg("-I")
            .arg("-c")
            .arg(format!("{LOAD_MEMO}{script}"))
            .arg(memo_path())
            .args(args);
        command
    }

    /// Like [`python`], with extra arguments.
    pub(crate) fn python_with(script: &str, args: &[&OsStr]) -> String {
        let output = python_command(script, args).output().expect("python3 runs");
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

    /// A fresh, empty directory for one test, deleted when the value is
    /// dropped: after the test passes, and also when an assertion fails.
    pub(crate) struct Scratch {
        path: PathBuf,
    }

    impl Scratch {
        pub(crate) fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            // Drop cannot report an error, and panicking here during a failed
            // test would abort the run; at worst one temp folder is left.
            drop(fs::remove_dir_all(&self.path));
        }
    }

    pub(crate) fn scratch(name: &str) -> Scratch {
        let path = env::temp_dir().join(format!("mnem-compat-{}-{name}", std::process::id()));
        if path.exists() {
            fs::remove_dir_all(&path).expect("old scratch directory is removable");
        }
        fs::create_dir_all(&path).expect("scratch directory is creatable");
        Scratch { path }
    }

    pub(crate) fn text(raw: &str) -> Text {
        Text::try_new(raw, TEXT_MAX).expect("test input is valid text")
    }

    /// Copies a notebook: its files and `TREE/` level files.
    pub(crate) fn copy_notebook(from: &Path, to: &Path) {
        fs::create_dir_all(to.join("TREE")).expect("target is creatable");
        for name in ["LOG.txt", ".lock", "config"] {
            if from.join(name).exists() {
                fs::copy(from.join(name), to.join(name)).expect("file copies");
            }
        }
        for entry in fs::read_dir(from.join("TREE")).expect("TREE/ is readable") {
            let entry = entry.expect("entry is readable");
            fs::copy(entry.path(), to.join("TREE").join(entry.file_name())).expect("level copies");
        }
    }

    pub(crate) fn day(raw: &str) -> Day {
        Day::try_new(raw).expect("a real date")
    }

    pub(crate) fn block(lo: u64, hi: u64) -> Span {
        Span::try_new(lo, hi).expect("an aligned block")
    }

    /// Parses Python's printed `lo hi` lines into blocks.
    pub(crate) fn blocks_from(printed: &str) -> Vec<Span> {
        let mut blocks = Vec::new();
        for line in printed.lines() {
            let (lo, hi) = line.split_once(' ').expect("a 'lo hi' line");
            blocks.push(block(
                lo.parse::<u64>().expect("lo"),
                hi.parse::<u64>().expect("hi"),
            ));
        }
        blocks
    }

    /// Every file under `TREE/`, by name, with its bytes.
    pub(crate) fn tree_files(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut files = Vec::new();
        for entry in fs::read_dir(dir.join("TREE")).expect("TREE/ is readable") {
            let entry = entry.expect("entry is readable");
            let name = entry.file_name().to_string_lossy().into_owned();
            files.push((name, fs::read(entry.path()).expect("level is readable")));
        }
        files.sort();
        files
    }
}

/// Python that makes notebook `sys.argv[2]` hold `sys.argv[3]` notes, then
/// builds the first `sys.argv[4]` due summaries with deterministic text.
const PY_BUILD: &str = "import os\nd = sys.argv[2]\nos.makedirs(os.path.join(d, 'TREE'), exist_ok=True)\nopen(os.path.join(d, 'LOG.txt'), 'a').close()\nm.log_append(d, [('2026-10-%02d' % (1 + i % 28), 'py note %d é' % i) for i in range(int(sys.argv[3]))])\nT = m.log_len(d)\nfor lo, hi in m.pending(d, T)[:int(sys.argv[4])]:\n    m.tree_put(d, lo, hi, 'py summary %d-%d' % (lo, hi - 1))\n";

/// Builds a notebook at `dir` with Python: `notes` notes, `built` summaries.
fn py_build(dir: &Path, notes: u64, built: usize) {
    let notes = notes.to_string();
    let built = built.to_string();
    let args = [dir.as_os_str(), OsStr::new(&notes), OsStr::new(&built)];
    harness::python_with(PY_BUILD, &args);
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
    let memory = dir.path().join("memory");
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

#[test]
#[ignore = "runs OptMem's Python memo; set OPTMEM_MEMO"]
fn memo_reads_what_mnem_writes() {
    let scratch = harness::scratch("mnem-writes");
    let dir = scratch.path().join("nb");
    let notebook = Notebook::init(&dir).expect("init");
    for i in 0..150_u64 {
        let note = harness::text(&format!("mnem note {i} \u{1F680}"));
        notebook
            .append(&[(harness::day("2026-10-31"), note)])
            .expect("append");
    }
    for due in notebook.pending(150, None).expect("pending") {
        let summary = harness::text(&format!("mnem summary {due}"));
        assert_eq!(
            notebook.put_summary(due, &summary).expect("put"),
            PutOutcome::Saved,
            "{due}"
        );
    }

    let script = "d = sys.argv[2]\nT = m.log_len(d)\nprint(T, len(m.pending(d, T)))\nfor i in range(T):\n    print('#%d %s %s' % m.log_get(d, i))\nsize = 2\nwhile size <= T:\n    for k in range(T // size):\n        print('%d-%d %s' % (k * size, (k + 1) * size - 1, m.tree_get(d, k * size, (k + 1) * size)))\n    size *= 2\n";
    let printed = harness::python_with(script, &[dir.as_os_str()]);
    let mut lines = printed.lines();
    assert_eq!(
        lines.next(),
        Some("150 0"),
        "memo sees 150 notes and nothing pending"
    );
    for i in 0..150 {
        let want = format!("#{i} 2026-10-31 mnem note {i} \u{1F680}");
        assert_eq!(lines.next(), Some(want.as_str()), "memo reads note {i}");
    }
    let mut summaries = 0;
    for line in lines {
        let (id, summary) = line.split_once(' ').expect("an 'id summary' line");
        assert_eq!(
            summary,
            format!("mnem summary {id}"),
            "memo reads summary {id}"
        );
        summaries += 1;
    }
    assert_eq!(
        summaries,
        75 + 37 + 18 + 9 + 4 + 2 + 1,
        "every summary of 150 notes was compared"
    );

    let wake = harness::run_memo(&dir, &["wake"]);
    let output = String::from_utf8_lossy(&wake.stdout);
    assert!(wake.status.success(), "memo wake succeeds");
    assert!(
        output.contains("You are awake."),
        "memo wake reads the whole notebook"
    );
    assert!(
        output.contains("mnem summary"),
        "150 notes overflow wake's budget, so memo reads Mnem's summaries"
    );
}

#[test]
#[ignore = "runs OptMem's Python memo; set OPTMEM_MEMO"]
fn mnem_reads_what_memo_writes() {
    let scratch = harness::scratch("memo-writes");
    let dir = scratch.path().join("nb");
    py_build(&dir, 45, 30);
    let pending = harness::python_with(
        "d = sys.argv[2]\nfor lo, hi in m.pending(d, m.log_len(d)):\n    print(lo, hi)\n",
        &[dir.as_os_str()],
    );

    let notebook = Notebook::open(&dir).expect("open");
    assert_eq!(notebook.len().expect("len"), 45, "45 notes");
    for (i, note) in (0_u64..).zip(notebook.scan().expect("scan")) {
        let note = note.expect("memo's records decode");
        assert_eq!(note.id, i, "id {i}");
        assert_eq!(note.text, format!("py note {i} \u{e9}"), "text of note {i}");
    }
    let due = notebook.pending(45, None).expect("pending");
    assert!(!due.is_empty(), "some blocks are left unbuilt");
    assert_eq!(due, harness::blocks_from(&pending), "same blocks pending");
    assert_eq!(
        notebook.pending_count(45).expect("count"),
        u64::try_from(due.len()).expect("fits"),
        "count matches the list"
    );
    let built = harness::python_with(
        "d = sys.argv[2]\nT = m.log_len(d)\nfor size in (2, 4, 8, 16, 32):\n    for k in range(T // size):\n        print(k * size, (k + 1) * size, m.tree_get(d, k * size, (k + 1) * size))\n",
        &[dir.as_os_str()],
    );
    for line in built.lines() {
        let mut fields = line.splitn(3, ' ');
        let lo = fields
            .next()
            .and_then(|f| f.parse::<u64>().ok())
            .expect("lo");
        let hi = fields
            .next()
            .and_then(|f| f.parse::<u64>().ok())
            .expect("hi");
        let want = fields.next().expect("summary or None");
        let got = notebook.summary(harness::block(lo, hi)).expect("read");
        assert_eq!(
            got.as_deref().unwrap_or("None"),
            want,
            "summary of {lo}-{}",
            hi - 1
        );
    }
}

#[test]
#[ignore = "runs OptMem's Python memo; set OPTMEM_MEMO"]
fn forget_matches_memo() {
    let scratch = harness::scratch("forget");
    let by_memo = scratch.path().join("by-memo");
    let by_mnem = scratch.path().join("by-mnem");
    py_build(&by_memo, 40, 1000);
    harness::copy_notebook(&by_memo, &by_mnem);
    let log_before = fs::read(by_mnem.join("LOG.txt")).expect("log");

    let gone = harness::python_with(
        "for lo, hi in m.tree_drop(sys.argv[2], 8, 16):\n    print(lo, hi)\n",
        &[by_memo.as_os_str()],
    );
    let dropped = Notebook::open(&by_mnem)
        .expect("open")
        .forget(harness::block(8, 16))
        .expect("forget");
    assert!(
        !dropped.is_empty(),
        "forgetting a built block drops something"
    );
    assert_eq!(
        dropped,
        harness::blocks_from(&gone),
        "same blocks dropped, same order"
    );
    assert_eq!(
        harness::tree_files(&by_mnem),
        harness::tree_files(&by_memo),
        "identical TREE files"
    );
    assert_eq!(
        fs::read(by_mnem.join("LOG.txt")).expect("log"),
        log_before,
        "forget leaves the log alone"
    );
}

#[test]
#[ignore = "runs OptMem's Python memo; set OPTMEM_MEMO"]
fn crash_repair_matches_memo() {
    let scratch = harness::scratch("repair");
    let by_memo = scratch.path().join("by-memo");
    let by_mnem = scratch.path().join("by-mnem");
    py_build(&by_memo, 3, 0);
    let mut log = OpenOptions::new()
        .append(true)
        .open(by_memo.join("LOG.txt"))
        .expect("log opens");
    log.write_all(b"#3 2026-10-31 cut off mid-wri")
        .expect("partial write");
    drop(log);
    harness::copy_notebook(&by_memo, &by_mnem);

    harness::python_with(
        "m.log_append(sys.argv[2], [('2026-10-31', 'after the crash')])\n",
        &[by_memo.as_os_str()],
    );
    let id = Notebook::open(&by_mnem)
        .expect("open")
        .append(&[(harness::day("2026-10-31"), harness::text("after the crash"))])
        .expect("append");
    assert_eq!(id, 3, "the partial record is cut off first");
    assert_eq!(
        fs::read(by_mnem.join("LOG.txt")).expect("log"),
        fs::read(by_memo.join("LOG.txt")).expect("log"),
        "identical logs after repair"
    );
}

#[test]
#[ignore = "runs OptMem's Python memo; set OPTMEM_MEMO"]
fn mnem_and_memo_share_the_write_lock() {
    let scratch = harness::scratch("lock");
    let dir = scratch.path().join("nb");
    Notebook::init(&dir).expect("init");
    let mut memo_writer = harness::python_command(
        "for i in range(100):\n    m.log_append(sys.argv[2], [('2026-10-31', 'memo writer %d' % i)])\n",
        &[dir.as_os_str()],
    )
    .spawn()
    .expect("python starts");
    let mut writers = Vec::new();
    for writer in 0..4 {
        let dir = dir.clone();
        writers.push(thread::spawn(move || {
            let notebook = Notebook::open(&dir).expect("open");
            for i in 0..50 {
                let note = harness::text(&format!("mnem writer {writer} note {i}"));
                notebook
                    .append(&[(harness::day("2026-10-31"), note)])
                    .expect("append");
            }
        }));
    }
    for handle in writers {
        handle.join().expect("writer thread finishes");
    }
    assert!(
        memo_writer.wait().expect("python exits").success(),
        "memo writer succeeds"
    );

    let notebook = Notebook::open(&dir).expect("open");
    let mut count = 0_u64;
    for (position, note) in (0_u64..).zip(notebook.scan().expect("scan")) {
        assert_eq!(
            note.expect("every record is whole").id,
            position,
            "ids are unique and in order"
        );
        count += 1;
    }
    assert_eq!(count, 300, "every note from both tools is there");
    assert_eq!(
        fs::metadata(dir.join("LOG.txt")).expect("log").len(),
        300 * LOG_REC,
        "no torn writes"
    );
}

#[test]
#[ignore = "runs OptMem's Python memo; set OPTMEM_MEMO"]
fn mnem_waits_while_memo_holds_the_lock() {
    let scratch = harness::scratch("lock-held");
    let dir = scratch.path().join("nb");
    Notebook::init(&dir).expect("init");
    let mut holder = harness::python_command(
        "lock = m.locked(sys.argv[2])\nprint('held', flush=True)\nsys.stdin.readline()\nlock.close()\n",
        &[dir.as_os_str()],
    )
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .spawn()
    .expect("python starts");
    let mut said = String::new();
    BufReader::new(holder.stdout.take().expect("piped stdout"))
        .read_line(&mut said)
        .expect("python reports the lock");
    assert_eq!(said.trim(), "held", "memo holds the lock");

    let (done, finished) = mpsc::channel();
    let writer_dir = dir.clone();
    let writer = thread::spawn(move || {
        let id = Notebook::open(&writer_dir)
            .expect("open")
            .append(&[(harness::day("2026-10-31"), harness::text("waited"))])
            .expect("append");
        done.send(id).expect("the test is listening");
    });
    assert!(
        finished.recv_timeout(Duration::from_millis(500)).is_err(),
        "append must wait while memo holds the lock"
    );
    assert_eq!(
        fs::metadata(dir.join("LOG.txt")).expect("log").len(),
        0,
        "nothing was written meanwhile"
    );

    let mut stdin = holder.stdin.take().expect("piped stdin");
    stdin.write_all(b"\n").expect("python is listening");
    drop(stdin);
    assert_eq!(
        finished
            .recv_timeout(Duration::from_secs(10))
            .expect("append finishes once the lock is released"),
        0,
        "the first note"
    );
    writer.join().expect("writer thread finishes");
    assert!(
        holder.wait().expect("python exits").success(),
        "memo releases cleanly"
    );
}
