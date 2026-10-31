//! Dream: rewrites MIND.md from what the notebook learned since the last
//! dream, with one model call, and checks everything the model writes
//! before it is saved.
//!
//! A run reads the notes after the watermark (at most [`NOTES_MAX`]), the
//! previous MIND.md, a wake-sized overview of the whole notebook, the
//! corrections among the new notes and the fired-nudge ledger. The model
//! answers with a `<reflection>`, which goes to `dream.log`, and a `<mind>`,
//! the draft. Code applies [`mind::check`], replaces MIND.md atomically, and
//! only then advances the watermark. If anything fails, the previous MIND.md
//! and watermark stay, and the next run tries again.
//!
//! When too few notes are new, or no model is set, a run only re-checks the
//! existing MIND.md, which drops faded hunches and stale or fired nudges
//! without a model call.

use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::fs::TryLockError;
use std::io;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;

use jiff::civil::DateTime;

use crate::cover;
use crate::fired;
use crate::fired::Fired;
use crate::mind;
use crate::mind::Dropped;
use crate::mind::Facts;
use crate::mind::Mind;
use crate::model::Answer;
use crate::model::Job;
use crate::model::Model;
use crate::notebook::Notebook;
use crate::notebook::NotebookError;
use crate::tags;

/// Dream's log in the notebook directory: each run's reflection, and the
/// lines code dropped or flagged.
pub const LOG: &str = "dream.log";

/// The number of notes the last dream read, in the notebook directory.
pub const WATERMARK: &str = "dream.watermark";

const LOCK: &str = ".dream.lock";

/// Nightly and idle dreams call the model only when at least this many
/// notes are new.
pub const MIN_NEW: u64 = 4;

/// The most new notes one dream reads; the next one reads on from there.
pub const NOTES_MAX: u64 = 400;

/// When `dream.log` grows past this many bytes, its oldest entries go.
const LOG_MAX: u64 = 1 << 20;

/// How much of `dream.log` is kept when it is trimmed.
const LOG_KEEP: u64 = 1 << 19;

/// A line is flagged in the log, though kept, when fewer than one in this
/// many of its longer words appear in the notes it cites.
const OVERLAP_ONE_IN: usize = 5;

/// What started a dream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// The nightly consolidation.
    Nightly,
    /// The agent has been idle after a session.
    Idle,
    /// The agent's context is about to be compacted.
    Compact,
}

impl Trigger {
    /// The trigger named `name`: `nightly`, `idle` or `compact`.
    pub fn named(name: &str) -> Option<Self> {
        match name {
            "nightly" => Some(Self::Nightly),
            "idle" => Some(Self::Idle),
            "compact" => Some(Self::Compact),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Nightly => "nightly",
            Self::Idle => "idle",
            Self::Compact => "compact",
        }
    }

    /// How many new notes it takes to call the model. Before compaction
    /// any new note counts, since the session is about to be forgotten.
    pub fn min_new(self) -> u64 {
        match self {
            Self::Nightly | Self::Idle => MIN_NEW,
            Self::Compact => 1,
        }
    }
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Another dream holds the lock; this one did nothing.
    Busy,
    /// The model wrote a new MIND.md from notes `first..end`.
    Dreamed {
        first: u64,
        end: u64,
        dropped: usize,
        flagged: usize,
    },
    /// No model call: too few notes are new (`new`), or no model is set.
    /// `dropped` lines expired from MIND.md.
    Tidied { new: u64, dropped: usize },
    /// The model declined; nothing changed.
    Declined,
    /// The model's draft was unusable; nothing changed.
    Rejected(String),
    /// The model could not be reached; nothing changed.
    Unavailable(String),
}

/// Runs one dream on `notebook` as of `now`. Without a model, it only
/// tidies.
pub fn run(
    notebook: &Notebook,
    model: Option<&mut dyn Model>,
    trigger: Trigger,
    now: DateTime,
) -> Result<Outcome, NotebookError> {
    let dir = notebook.dir();
    let lock_path = dir.join(LOCK);
    let lock = OpenOptions::new()
        .append(true)
        .create(true)
        .open(&lock_path)
        .map_err(|source| io_error(&lock_path, source))?;
    match lock.try_lock() {
        Ok(()) => {}
        // A dream is running; it, or the next one, reads these notes.
        Err(TryLockError::WouldBlock) => return Ok(Outcome::Busy),
        Err(TryLockError::Error(source)) => return Err(io_error(&lock_path, source)),
    }
    let outcome = locked(notebook, model, trigger, now);
    lock.unlock()
        .map_err(|source| io_error(&lock_path, source))?;
    outcome
}

fn locked(
    notebook: &Notebook,
    model: Option<&mut dyn Model>,
    trigger: Trigger,
    now: DateTime,
) -> Result<Outcome, NotebookError> {
    let dir = notebook.dir();
    let mut facts = Facts::default();
    for note in notebook.scan()? {
        facts.add(&note?);
    }
    let len = facts.len();
    let watermark = read_watermark(dir)?.min(len);
    let new = len.saturating_sub(watermark);
    let old_text = read_mind(dir)?;
    let ledger = fired::read(dir)?;
    let mut entry = vec![format!("== {} {}", fired::minutes(now), trigger.name())];
    let model = match model {
        Some(model) if new >= trigger.min_new() => model,
        _ => {
            let dropped = tidy(dir, old_text.as_deref(), &facts, &ledger, now, &mut entry)?;
            if dropped > 0 {
                append_log(dir, &entry)?;
            }
            return Ok(Outcome::Tidied { new, dropped });
        }
    };
    let end = len.min(watermark.saturating_add(NOTES_MAX));
    let prompt = prompt(
        notebook,
        &facts,
        &ledger,
        old_text.as_deref(),
        watermark,
        end,
        now,
    )?;
    let reply = match model.ask(Job::Dream, &prompt) {
        Ok(Answer::Reply(reply)) => reply,
        Ok(Answer::Declined) => {
            entry.push(format!("the model declined notes #{watermark} to #{end}"));
            append_log(dir, &entry)?;
            return Ok(Outcome::Declined);
        }
        Err(unavailable) => {
            entry.push(format!("the model is unavailable: {}", unavailable.0));
            append_log(dir, &entry)?;
            return Ok(Outcome::Unavailable(unavailable.0));
        }
    };
    let draft = match read_reply(&reply, old_text.as_deref()) {
        Ok(draft) => draft,
        Err(why) => {
            entry.push(format!("rejected the draft: {why}"));
            append_log(dir, &entry)?;
            return Ok(Outcome::Rejected(why));
        }
    };
    entry.push(format!("dreamed over notes #{watermark} to #{end}"));
    if let Some(reflection) = &draft.reflection {
        entry.push("reflection:".to_owned());
        for line in reflection.lines() {
            entry.push(format!("  {line}"));
        }
    }
    let (kept, more) = mind::check(draft.mind, &facts, &ledger, now);
    let mut dropped = draft.dropped;
    dropped.extend(more);
    log_dropped(&mut entry, &dropped);
    let flagged = weakly_cited(notebook, &kept)?;
    for line in &flagged {
        entry.push(format!(
            "flagged: {line} (few words in common with the notes it cites)"
        ));
    }
    replace(&dir.join(mind::FILE), &mind::render(&kept))?;
    replace(&dir.join(WATERMARK), &format!("{end}\n"))?;
    append_log(dir, &entry)?;
    Ok(Outcome::Dreamed {
        first: watermark,
        end,
        dropped: dropped.len(),
        flagged: flagged.len(),
    })
}

/// Re-checks the existing MIND.md and saves it if any line expired.
/// Returns how many lines were dropped.
fn tidy(
    dir: &Path,
    old_text: Option<&str>,
    facts: &Facts,
    ledger: &[Fired],
    now: DateTime,
    entry: &mut Vec<String>,
) -> Result<usize, NotebookError> {
    let Some(Ok((old, _))) = old_text.map(mind::parse) else {
        // No MIND.md yet, or one that is not Dream's: leave it alone.
        return Ok(0);
    };
    let (kept, dropped) = mind::check(old, facts, ledger, now);
    if dropped.is_empty() {
        return Ok(0);
    }
    entry.push("expired lines without a model call".to_owned());
    log_dropped(entry, &dropped);
    replace(&dir.join(mind::FILE), &mind::render(&kept))?;
    Ok(dropped.len())
}

fn log_dropped(entry: &mut Vec<String>, dropped: &[Dropped]) {
    for item in dropped {
        entry.push(format!("dropped: {} ({})", item.line, item.why));
    }
}

/// A parsed reply.
struct Draft {
    reflection: Option<String>,
    mind: Mind,
    dropped: Vec<Dropped>,
}

/// The draft in a reply, or why it cannot be used.
fn read_reply(reply: &str, old_text: Option<&str>) -> Result<Draft, String> {
    if reply.contains("<previous-mind>") {
        return Err("the reply repeats the prompt".to_owned());
    }
    let mind_text =
        between(reply, "<mind>", "</mind>").ok_or("the reply has no <mind> … </mind> part")?;
    let (mind, dropped) = mind::parse(mind_text).map_err(|missing| missing.to_string())?;
    let had_lines = old_text
        .and_then(|text| mind::parse(text).ok())
        .is_some_and(|(old, _)| old != Mind::default());
    if had_lines && mind == Mind::default() && dropped.is_empty() {
        return Err("the draft is empty and would erase MIND.md".to_owned());
    }
    Ok(Draft {
        reflection: between(reply, "<reflection>", "</reflection>")
            .map(|text| text.trim().to_owned()),
        mind,
        dropped,
    })
}

/// The text between the last `open` in `text` and the `close` after it.
fn between<'a>(text: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let (_, after) = text.rsplit_once(open)?;
    let (inside, _) = after.split_once(close)?;
    Some(inside)
}

/// The lines of `mind` whose longer words barely appear in the notes they
/// cite: a sign the model wrote something the notes do not say.
fn weakly_cited(notebook: &Notebook, mind: &Mind) -> Result<Vec<String>, NotebookError> {
    let mut lines: Vec<String> = mind.durable.iter().chain(&mind.active).cloned().collect();
    for hunch in &mind.hunches {
        lines.push(hunch.text.clone());
    }
    for nudge in &mind.nudges {
        lines.push(format!("{} {}", nudge.action, nudge.why));
    }
    let mut flagged = Vec::new();
    for line in lines {
        let own = words(&line);
        if own.len() < 3 {
            continue;
        }
        let mut cited = String::new();
        for id in mind::cites(&line) {
            cited.push_str(&notebook.note(id)?.text);
            cited.push(' ');
        }
        let theirs = words(&cited);
        let shared = own.iter().filter(|word| theirs.contains(*word)).count();
        if shared.saturating_mul(OVERLAP_ONE_IN) < own.len() {
            flagged.push(line);
        }
    }
    Ok(flagged)
}

/// The distinct lowercase words of four letters or more in `text`.
fn words(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    for word in text.split(|c: char| !c.is_alphabetic()) {
        if word.chars().count() < 4 {
            continue;
        }
        let word = word.to_lowercase();
        if !found.contains(&word) {
            found.push(word);
        }
    }
    found
}

fn prompt(
    notebook: &Notebook,
    facts: &Facts,
    ledger: &[Fired],
    old_text: Option<&str>,
    first: u64,
    end: u64,
    now: DateTime,
) -> Result<String, NotebookError> {
    let index = facts.index();
    let mut overview = Vec::new();
    let spans = cover::cover(facts.len(), notebook.config().wake_lines())
        .map_err(|_too_large| NotebookError::Overflow)?;
    for span in spans {
        if span.is_note() {
            overview.push(index.shown(&notebook.note(span.lo())?));
        } else if let Some(summary) = notebook.summary(span)? {
            overview.push(format!("#{span} {summary}"));
        }
    }
    let mut new_notes = Vec::new();
    let mut corrections = Vec::new();
    for note in notebook.notes(first, end)? {
        for target in tags::parse_lenient(&note.text).updates {
            if target < note.id {
                corrections.push(format!(
                    "#{} corrects {}",
                    note.id,
                    notebook.note(target)?.line()
                ));
            }
        }
        new_notes.push(index.shown(&note));
    }
    let mut fired_lines = Vec::new();
    for entry in ledger {
        fired_lines.push(fired::line(&entry.id, entry.at));
    }
    let none = "(none)".to_owned();
    let section = |lines: &[String]| {
        if lines.is_empty() {
            none.clone()
        } else {
            lines.join("\n")
        }
    };
    Ok(format!(
        "Task: you are the reflective \"Dream\" pass of a personal assistant's memory. \
         Read the material below and write the assistant's new MIND.md: a short, cited \
         picture of what holds, what is going on, what keeps happening, and what the \
         assistant should do next for its user. Call no tools; answer from this material alone.\n\n\
         Now: {now} (the user's local time).\n\n\
         Answer in two parts. First, between <reflection> and </reflection>, think in plain \
         prose; it is logged, not shown. Pair each correction with the note it corrects; \
         for each hunch, count the new notes that support it; mark open items that are \
         resolved; then say what changed, what keeps happening, and what should happen \
         next. \"Nothing\" is a fine answer.\n\n\
         Then, between <mind> and </mind>, the new MIND.md: these four headings, in this \
         order, each followed by lines that start with \"- \".\n\
         ## Durable: lasting facts and preferences, at most {durable} lines. Keep a line \
         until a newer note contradicts it; then rewrite it, citing the newer note.\n\
         ## Active: current work and recent events, at most {active} lines.\n\
         ## Hunches: patterns across days, not yet confirmed, at most {hunches} lines. \
         Start each with [daily] or [weekly], and cite notes from at least two different days.\n\
         ## Nudges: things worth doing, at most {nudges} lines, each written\n\
         - <id> | <when> | <until> | <what to do> | <why, with citations> | <notify or quiet>\n\
         where <id> is a few lowercase letters, digits or hyphens (keep the id of a nudge \
         you carry over); <when> is YYYY-MM-DDTHH:MM, \"next session\", or \"event:<name>\"; \
         <until> is \"-\", HH:MM or YYYY-MM-DDTHH:MM, after which the nudge is stale; \
         notify means tell the user, quiet means act or prepare without asking.\n\n\
         Rules:\n\
         - Every line cites the notes it rests on, like (#12, #40). Cite note ids only, \
         never summary ranges such as #0-15. Lines without valid citations are deleted.\n\
         - A note marked [superseded by #n] is out of date; rely on #n.\n\
         - Do not repeat a nudge listed under fired unless you give it a new, later time.\n\
         - A quiet night is a good result: carry MIND.md over unchanged, or with fewer \
         lines. Never invent nudges to seem useful.\n\n\
         <previous-mind>\n{old}\n</previous-mind>\n\n\
         <notebook>\nAn overview of the whole notebook, oldest first; #lo-hi lines \
         summarize notes lo to hi.\n{overview}\n</notebook>\n\n\
         <new-notes>\nNotes #{first} to #{last}, written since the last dream.\n{new}\n</new-notes>\n\n\
         <corrections>\n{corrections}\n</corrections>\n\n\
         <fired>\nNudges and reminders already given to the assistant.\n{fired}\n</fired>\n",
        now = fired::minutes(now),
        durable = mind::DURABLE_MAX,
        active = mind::ACTIVE_MAX,
        hunches = mind::HUNCHES_MAX,
        nudges = mind::NUDGES_MAX,
        old = old_text.map_or_else(|| none.clone(), |text| text.trim().to_owned()),
        overview = section(&overview),
        last = end.saturating_sub(1),
        new = section(&new_notes),
        corrections = section(&corrections),
        fired = section(&fired_lines),
    ))
}

/// The number of notes the last dream read; 0 before the first dream.
pub fn read_watermark(dir: &Path) -> Result<u64, NotebookError> {
    let path = dir.join(WATERMARK);
    match fs::read_to_string(&path) {
        Ok(text) => text.trim().parse::<u64>().map_err(|_not_a_number| {
            io_error(
                &path,
                io::Error::new(io::ErrorKind::InvalidData, "not a note count"),
            )
        }),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(source) => Err(io_error(&path, source)),
    }
}

/// MIND.md's text, if there is one.
pub fn read_mind(dir: &Path) -> Result<Option<String>, NotebookError> {
    let path = dir.join(mind::FILE);
    match fs::read_to_string(&path) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(io_error(&path, source)),
    }
}

/// Replaces `path` with `contents` all at once: readers see the old file
/// or the new one, never part of either.
pub(crate) fn replace(path: &Path, contents: &str) -> Result<(), NotebookError> {
    let mut name = path.file_name().unwrap_or_default().to_owned();
    name.push(".tmp");
    let temp = path.with_file_name(name);
    let written = File::create(&temp)
        .and_then(|mut file| {
            file.write_all(contents.as_bytes())?;
            file.sync_all()
        })
        .and_then(|()| fs::rename(&temp, path));
    written.map_err(|source| {
        drop(fs::remove_file(&temp));
        io_error(path, source)
    })
}

/// Appends `entry` to the log, dropping its oldest entries when it is
/// too large.
fn append_log(dir: &Path, entry: &[String]) -> Result<(), NotebookError> {
    let path = dir.join(LOG);
    let mut text = entry.join("\n");
    text.push_str("\n\n");
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(&path)
        .map_err(|source| io_error(&path, source))?;
    file.write_all(text.as_bytes())
        .map_err(|source| io_error(&path, source))?;
    let size = file
        .metadata()
        .map_err(|source| io_error(&path, source))?
        .len();
    if size <= LOG_MAX {
        return Ok(());
    }
    let mut file = File::open(&path).map_err(|source| io_error(&path, source))?;
    file.seek(SeekFrom::Start(size.saturating_sub(LOG_KEEP)))
        .map_err(|source| io_error(&path, source))?;
    let mut tail = Vec::new();
    file.read_to_end(&mut tail)
        .map_err(|source| io_error(&path, source))?;
    let tail = String::from_utf8_lossy(&tail);
    // Start at a whole entry.
    let kept = tail.split_once("\n\n").map_or(&*tail, |(_, rest)| rest);
    replace(&path, kept)
}

fn io_error(path: &Path, source: io::Error) -> NotebookError {
    NotebookError::Io {
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::fs::OpenOptions;

    use jiff::civil::DateTime;

    use super::Outcome;
    use super::Trigger;
    use crate::day::Day;
    use crate::dream;
    use crate::mind;
    use crate::model::Answer;
    use crate::model::Job;
    use crate::model::Model;
    use crate::model::Unavailable;
    use crate::notebook::Notebook;
    use crate::tags;
    use crate::testing::Scratch;
    use crate::text::Text;

    /// A model that gives one scripted answer and keeps the prompts.
    struct Scripted {
        answer: Result<Answer, Unavailable>,
        prompts: Vec<String>,
    }

    impl Scripted {
        fn replying(reply: &str) -> Self {
            Self {
                answer: Ok(Answer::Reply(reply.to_owned())),
                prompts: Vec::new(),
            }
        }
    }

    impl Model for Scripted {
        fn ask(&mut self, job: Job, prompt: &str) -> Result<Answer, Unavailable> {
            assert_eq!(job, Job::Dream, "Dream asks only for dreams");
            self.prompts.push(prompt.to_owned());
            self.answer.clone()
        }
    }

    fn now() -> DateTime {
        tags::date_time("2026-10-31T03:00").expect("a valid time")
    }

    fn notebook(scratch: &Scratch, notes: &[(&str, &str)]) -> Notebook {
        let notebook = Notebook::init(&scratch.path).expect("a new notebook");
        add(&notebook, notes);
        notebook
    }

    fn add(notebook: &Notebook, notes: &[(&str, &str)]) {
        let mut batch = Vec::new();
        for (day, text) in notes {
            batch.push((
                Day::try_new(day).expect("a valid day"),
                Text::try_new(text, 280).expect("a valid note"),
            ));
        }
        notebook.append(&batch).expect("notes are saved");
    }

    const NOTES: [(&str, &str); 5] = [
        ("2026-10-28", "Lives in Mumbai"),
        ("2026-10-29", "Skipped breakfast, early meeting"),
        ("2026-10-30", "Skipped breakfast again"),
        ("2026-10-30", "[updates #0] Moved to Delhi"),
        ("2026-10-30", "Bought eggs and spinach"),
    ];

    const REPLY: &str = "<reflection>\n#3 corrects #0. Breakfast keeps being skipped.\n</reflection>\n\
        <mind>\n## Durable\n- Lives in Delhi (#3, replaces #0)\n- Has a dog (#99)\n\
        ## Active\n- Busy mornings this week (#1, #2)\n\
        ## Hunches\n- [daily] Skips breakfast when meetings start early (#1, #2)\n\
        - [daily] Likes spinach (#4)\n\
        ## Nudges\n- eggs | 2026-10-31T08:00 | 10:00 | Suggest an omelette before work | Skipped breakfast twice; bought eggs and spinach (#2, #4) | notify\n\
        </mind>\n";

    fn read(scratch: &Scratch, file: &str) -> String {
        fs::read_to_string(scratch.path.join(file)).unwrap_or_default()
    }

    #[test]
    fn a_dream_rewrites_mind_from_checked_lines() {
        let scratch = Scratch::new();
        let notebook = notebook(&scratch, &NOTES);
        let mut model = Scripted::replying(REPLY);
        let outcome = dream::run(&notebook, Some(&mut model), Trigger::Nightly, now())
            .expect("the dream runs");
        assert_eq!(
            outcome,
            Outcome::Dreamed {
                first: 0,
                end: 5,
                dropped: 2,
                flagged: 1
            },
            "two lines fail the checks, and one shares no words with its notes"
        );
        let prompt = model.prompts.first().expect("one call");
        for part in [
            "Now: 2026-10-31T03:00",
            "Notes #0 to #4, written since the last dream.",
            "#0 2026-10-28 [superseded by #3] Lives in Mumbai",
            "#3 corrects #0 2026-10-28 Lives in Mumbai",
            "<previous-mind>\n(none)\n</previous-mind>",
        ] {
            assert!(prompt.contains(part), "the prompt has {part:?}");
        }
        let written = read(&scratch, mind::FILE);
        let (mind, _) = mind::parse(&written).expect("Dream writes valid MIND.md");
        assert_eq!(
            mind.durable,
            ["Lives in Delhi (#3, replaces #0)"],
            "durable"
        );
        assert_eq!(mind.hunches.len(), 1, "a one-day hunch is dropped");
        assert_eq!(mind.nudges.len(), 1, "the nudge stands");
        assert_eq!(
            read(&scratch, dream::WATERMARK),
            "5\n",
            "the watermark moves"
        );
        let log = read(&scratch, dream::LOG);
        for part in [
            "== 2026-10-31T03:00 nightly",
            "  #3 corrects #0. Breakfast keeps being skipped.",
            "dropped: Has a dog (#99) (it cites #99, which does not exist)",
            "dropped: [daily] Likes spinach (#4) (a hunch needs notes from two different days)",
            "flagged: Busy mornings this week (#1, #2)",
        ] {
            assert!(log.contains(part), "the log has {part:?}: {log}");
        }
    }

    #[test]
    fn the_next_dream_reads_on_from_the_watermark() {
        let scratch = Scratch::new();
        let notebook = notebook(&scratch, &NOTES);
        let mut model = Scripted::replying(REPLY);
        dream::run(&notebook, Some(&mut model), Trigger::Nightly, now()).expect("first");
        add(&notebook, &[("2026-10-31", "Ate the omelette")]);
        let tidied = dream::run(&notebook, Some(&mut model), Trigger::Idle, now()).expect("idle");
        assert_eq!(
            tidied,
            Outcome::Tidied { new: 1, dropped: 0 },
            "one new note is too few for an idle dream"
        );
        assert_eq!(model.prompts.len(), 1, "no model call");
        let outcome =
            dream::run(&notebook, Some(&mut model), Trigger::Compact, now()).expect("compact");
        assert!(
            matches!(
                outcome,
                Outcome::Dreamed {
                    first: 5,
                    end: 6,
                    ..
                }
            ),
            "before compaction one note is enough: {outcome:?}"
        );
        let prompt = model.prompts.last().expect("a second call");
        assert!(
            prompt.contains("Notes #5 to #5,")
                && prompt.contains("- Lives in Delhi (#3, replaces #0)"),
            "only the new note, with the previous MIND.md"
        );
    }

    #[test]
    fn a_failed_dream_changes_nothing() {
        let scratch = Scratch::new();
        let notebook = notebook(&scratch, &NOTES);
        dream::run(
            &notebook,
            Some(&mut Scripted::replying(REPLY)),
            Trigger::Nightly,
            now(),
        )
        .expect("first");
        let before = read(&scratch, mind::FILE);
        add(&notebook, &NOTES);
        let cases = [
            (Scripted::replying("I would rather not."), "no <mind>"),
            (
                Scripted::replying("<mind>\n## Durable\n</mind>"),
                "no \"## Active\"",
            ),
            (
                Scripted::replying("<mind>\n## Durable\n## Active\n## Hunches\n## Nudges\n</mind>"),
                "would erase MIND.md",
            ),
            (
                Scripted::replying("<previous-mind>x</previous-mind><mind></mind>"),
                "repeats the prompt",
            ),
        ];
        for (mut model, why) in cases {
            let outcome = dream::run(&notebook, Some(&mut model), Trigger::Nightly, now())
                .expect("the run completes");
            assert!(
                matches!(&outcome, Outcome::Rejected(said) if said.contains(why)),
                "{outcome:?} says {why:?}"
            );
        }
        let mut unavailable = Scripted {
            answer: Err(Unavailable("timed out".to_owned())),
            prompts: Vec::new(),
        };
        assert_eq!(
            dream::run(&notebook, Some(&mut unavailable), Trigger::Nightly, now()).ok(),
            Some(Outcome::Unavailable("timed out".to_owned())),
            "an unreachable model"
        );
        assert_eq!(read(&scratch, mind::FILE), before, "MIND.md is kept");
        assert_eq!(
            read(&scratch, dream::WATERMARK),
            "5\n",
            "the notes wait for the next dream"
        );
    }

    #[test]
    fn without_a_model_old_lines_still_expire() {
        let scratch = Scratch::new();
        let notebook = notebook(&scratch, &NOTES);
        dream::run(
            &notebook,
            Some(&mut Scripted::replying(REPLY)),
            Trigger::Nightly,
            now(),
        )
        .expect("first");
        let later = tags::date_time("2026-10-31T11:00").expect("a valid time");
        assert_eq!(
            dream::run(&notebook, None, Trigger::Nightly, later).ok(),
            Some(Outcome::Tidied { new: 0, dropped: 1 }),
            "the omelette nudge went stale at 10:00"
        );
        let (mind, _) = mind::parse(&read(&scratch, mind::FILE)).expect("valid");
        assert!(mind.nudges.is_empty(), "the stale nudge is gone");
        assert_eq!(mind.durable.len(), 1, "the rest stays");
    }

    #[test]
    fn one_dream_at_a_time() {
        let scratch = Scratch::new();
        let notebook = notebook(&scratch, &NOTES);
        let held = OpenOptions::new()
            .append(true)
            .create(true)
            .open(scratch.path.join(".dream.lock"))
            .expect("the lock file opens");
        held.lock().expect("the lock is free");
        let mut model = Scripted::replying(REPLY);
        assert_eq!(
            dream::run(&notebook, Some(&mut model), Trigger::Nightly, now()).ok(),
            Some(Outcome::Busy),
            "a second dream leaves at once"
        );
        assert!(model.prompts.is_empty(), "without calling the model");
    }
}
