//! The `mnem` binary with a summarizer: `note` returns at once, and the
//! summaries are written afterwards by a detached `mnem compact`.

use std::time::Duration;
use std::time::Instant;

/// Helpers in a `cfg(test)` module, so Clippy's test allowances cover their
/// `expect` calls.
#[cfg(test)]
mod harness {
    use std::env;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::path::PathBuf;
    use std::process;
    use std::process::Command;
    use std::process::Output;

    /// A fresh directory for one test, deleted when dropped.
    pub(crate) struct Scratch {
        pub(crate) path: PathBuf,
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            // Drop cannot report an error; at worst one temp folder is left.
            drop(fs::remove_dir_all(&self.path));
        }
    }

    pub(crate) fn scratch(name: &str) -> Scratch {
        let path = env::temp_dir().join(format!("mnem-summarizer-{}-{name}", process::id()));
        if path.exists() {
            fs::remove_dir_all(&path).expect("stale scratch folder is removable");
        }
        fs::create_dir_all(&path).expect("scratch folder");
        Scratch { path }
    }

    /// An executable shell script at `path`.
    pub(crate) fn script(path: &Path, body: &str) {
        fs::write(path, format!("#!/bin/sh\n{body}\n")).expect("write script");
        let mut permissions = fs::metadata(path).expect("script").permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).expect("make executable");
    }

    /// Runs `mnem <args>` on the notebook in `dir` with `summarizer`.
    pub(crate) fn mnem(dir: &Path, summarizer: &Path, args: &[&str]) -> Output {
        let output = Command::new(env!("CARGO_BIN_EXE_mnem"))
            .args(args)
            .env("MEMORY_DIR", dir)
            .env("MNEM_MODEL", summarizer)
            .output()
            .expect("mnem runs");
        assert!(
            output.status.success(),
            "mnem {args:?} succeeds: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output
    }

    pub(crate) fn level_len(dir: &Path, size: u64) -> u64 {
        fs::metadata(dir.join("TREE").join(size.to_string())).map_or(0, |metadata| metadata.len())
    }
}

#[test]
fn notes_return_at_once_and_summaries_follow() {
    let scratch = harness::scratch("background");
    let dir = scratch.path.join("nb");
    let summarizer = scratch.path.join("summarizer.sh");
    // A slow model: each summary takes half a second.
    harness::script(
        &summarizer,
        "cat > /dev/null; sleep 0.5; echo \"summary by $MNEM_JOB\"",
    );
    harness::mnem(&dir, &summarizer, &["init"]);
    for i in 0..4 {
        let output = harness::mnem(&dir, &summarizer, &["note", &format!("note {i}")]);
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            format!("Noted as #{i}.\n"),
            "the agent is never asked for a summary"
        );
    }
    assert_eq!(
        harness::level_len(&dir, 4),
        0,
        "note returned before the slow summarizer could finish"
    );
    let deadline = Instant::now() + Duration::from_secs(20);
    while harness::level_len(&dir, 4) < 288 {
        assert!(
            Instant::now() < deadline,
            "the background summarizer writes 0-1, 2-3 and 0-3 within 20 s"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(harness::level_len(&dir, 2), 2 * 288, "both pairs written");
    let zoom = harness::mnem(&dir, &summarizer, &["zoom", "0-3"]);
    assert_eq!(
        String::from_utf8_lossy(&zoom.stdout),
        "#0-1 summary by summary\n#2-3 summary by summary\n",
        "the summaries are the model's lines"
    );
    // The background run may still hold its lock for a moment after the
    // last summary lands; a foreground run then reports it as busy.
    loop {
        let report = harness::mnem(&dir, &summarizer, &["compact"]);
        let said = String::from_utf8_lossy(&report.stdout).into_owned();
        if said == "Nothing left to summarize.\n" {
            break;
        }
        assert!(
            said.contains("already running"),
            "only busy, never anything else: {said}"
        );
        assert!(Instant::now() < deadline, "the background run finishes");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        !dir.join(".compact.failed").exists(),
        "no failure was recorded"
    );
}

#[test]
fn session_compact_dreams_in_the_background() {
    let scratch = harness::scratch("dream");
    let dir = scratch.path.join("nb");
    let model = scratch.path.join("model.sh");
    harness::script(
        &model,
        "cat > /dev/null\n\
         if [ \"$MNEM_JOB\" = dream ]; then\n\
         sleep 1\n\
         printf '<reflection>Tea twice.</reflection>\\n<mind>\\n## Durable\\n- Drinks green tea (#0, #1)\\n## Active\\n## Hunches\\n## Nudges\\n</mind>\\n'\n\
         else echo \"a summary\"; fi",
    );
    harness::mnem(&dir, &model, &["init"]);
    harness::mnem(&dir, &model, &["note", "Had green tea in the morning"]);
    harness::mnem(&dir, &model, &["note", "Green tea again after lunch"]);
    let output = harness::mnem(&dir, &model, &["session", "compact"]);
    assert!(
        String::from_utf8_lossy(&output.stdout).starts_with("Dreaming in the background"),
        "the hook says it is dreaming"
    );
    assert!(
        !dir.join("dream.watermark").exists(),
        "and returns before the one-second dream is done"
    );
    let deadline = Instant::now() + Duration::from_secs(20);
    while !dir.join("dream.watermark").exists() {
        assert!(
            Instant::now() < deadline,
            "the background dream finishes within 20 s"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    let start = harness::mnem(&dir, &model, &["session", "start"]);
    let said = String::from_utf8_lossy(&start.stdout);
    assert!(
        said.contains("Durable:\n- Drinks green tea (#0, #1)\n"),
        "the next session starts with what the dream learned: {said}"
    );
    assert!(said.ends_with("You are awake.\n"), "and then wakes: {said}");
    // Let every background run finish, so the scratch folder can go.
    for (args, done) in [
        (["dream", "compact"], "No dream"),
        (["compact", ""], "Nothing left to summarize."),
    ] {
        let args: Vec<&str> = args.into_iter().filter(|arg| !arg.is_empty()).collect();
        loop {
            let report = harness::mnem(&dir, &model, &args);
            let said = String::from_utf8_lossy(&report.stdout).into_owned();
            if said.starts_with(done) {
                break;
            }
            assert!(said.contains("already running"), "only busy: {said}");
            assert!(Instant::now() < deadline, "the background runs finish");
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}
