//! The `mnem` command line: OptMem's commands over a [`Notebook`].
//!
//! Commands return their output as a [`Reply`] instead of printing it, so the
//! binary's one output function does all the writing and tests read replies
//! directly. The wording of messages is Mnem's own; the lines that show
//! notes and summaries (`#id day text` and `#lo-hi summary`) match `memo`
//! byte for byte, because agents and tools read them.
//!
//! This is the stock OptMem flow: when a summary is due, `note`, `wake` and
//! `nap` print a prompt asking the agent to write it.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use regex::RegexBuilder;

use crate::compactor;
use crate::compactor::Input;
use crate::compactor::Run;
use crate::compactor::Stop;
use crate::compactor::Written;
use crate::config::ConfigError;
use crate::config::Knob;
use crate::cover;
use crate::day::Day;
use crate::day::DayError;
use crate::model::CommandModel;
use crate::notebook::Notebook;
use crate::notebook::NotebookError;
use crate::notebook::PutOutcome;
use crate::span::Span;
use crate::span::SpanError;
use crate::text;
use crate::text::Text;
use crate::text::TextError;

/// How large a compiled `recall` pattern may grow. Patterns such as
/// `\w{1000}` expand far past the regex crate's default limit, because `\w`
/// covers all of Unicode.
const RECALL_REGEX_BYTES: usize = 256 << 20;

/// What a command needs from outside the notebook.
#[derive(Clone, Debug)]
pub struct Context {
    /// The notebook directory.
    pub dir: PathBuf,
    /// How messages name this program, in a form the agent can run.
    pub me: String,
    /// The user's home directory, used to shorten paths in messages.
    pub home: Option<PathBuf>,
    /// The day new notes are dated with.
    pub today: Day,
    /// The program that writes summaries, if the harness provides one.
    /// Without it the agent is asked to write them, as in stock OptMem.
    pub summarizer: Option<CommandModel>,
}

/// A command's output and exit status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reply {
    /// What goes to standard output.
    pub stdout: String,
    /// What goes to standard error.
    pub stderr: String,
    /// The exit status.
    pub code: u8,
    /// Whether summaries are due and the summarizer should start in the
    /// background once the reply is out.
    pub compact: bool,
}

impl Reply {
    /// A failure before any command ran, such as an unreadable environment.
    pub fn failed(message: &str) -> Self {
        Self {
            stdout: String::new(),
            stderr: format!("{message}\n"),
            code: 1,
            compact: false,
        }
    }

    fn done(lines: &[String]) -> Self {
        Self {
            stdout: joined(lines),
            stderr: String::new(),
            code: 0,
            compact: false,
        }
    }

    /// Output the agent must act on before the command can succeed, such as
    /// `wake` asking for a missing summary.
    fn refused(lines: &[String]) -> Self {
        Self {
            stdout: joined(lines),
            stderr: String::new(),
            code: 1,
            compact: false,
        }
    }
}

/// Why a command stopped.
#[derive(Debug)]
enum Failure {
    Notebook(NotebookError),
    Said(String),
}

impl From<NotebookError> for Failure {
    fn from(error: NotebookError) -> Self {
        Self::Notebook(error)
    }
}

impl From<ConfigError> for Failure {
    fn from(error: ConfigError) -> Self {
        Self::Said(error.to_string())
    }
}

type Outcome = Result<Reply, Failure>;

/// Runs one command. `args` are the words after the program name.
pub fn run(context: &Context, args: &[OsString]) -> Reply {
    let mut words = Vec::new();
    for arg in args {
        let Some(word) = arg.to_str() else {
            return Reply::failed("Arguments must be UTF-8 text.");
        };
        words.push(word);
    }
    let Some((&command, rest)) = words.split_first() else {
        return Reply::done(&[usage(context)]);
    };
    let outcome = match command {
        "help" | "-h" | "--help" => Ok(Reply::done(&[usage(context)])),
        // The one command that may create the notebook directory.
        "init" => init(context, rest),
        _ => match command_named(command) {
            Some(command) => Notebook::open(&context.dir)
                .map_err(Failure::Notebook)
                .and_then(|notebook| command(context, &notebook, rest)),
            None => Ok(Reply {
                compact: false,
                stdout: String::new(),
                stderr: format!("No such command: {command}\n\n{}\n", usage(context)),
                code: 1,
            }),
        },
    };
    match outcome {
        Ok(reply) => reply,
        Err(failure) => Reply::failed(&explain(context, failure)),
    }
}

/// A command that runs on an existing notebook.
type Command = fn(&Context, &Notebook, &[&str]) -> Outcome;

fn command_named(name: &str) -> Option<Command> {
    let command: Command = match name {
        "wake" => wake,
        "note" => note,
        "nap" => nap,
        "recall" => recall,
        "zoom" => zoom,
        "forget" => forget,
        "config" => config,
        "import" => import,
        "compact" => compact,
        _ => return None,
    };
    Some(command)
}

fn usage(context: &Context) -> String {
    let me = &context.me;
    format!(
        "mnem: a notebook for agents, compatible with OptMem.\n\n\
         Usage: {me} <command> [arguments]\n\n\
         Reading\n\
         \x20 wake [part [T]]          print the notebook; the first command of every session\n\
         \x20 recall <pattern>         search all notes (Rust regex syntax, case-insensitive)\n\
         \x20 zoom <lo>-<hi>           show the two halves of a summary block\n\n\
         Writing\n\
         \x20 note \"<line>\"            add a note of one line\n\
         \x20 nap [<lo>-<hi> \"<line>\"] show or write the next summary that is due\n\
         \x20 forget <lo>-<hi>         discard a summary and every summary built on it\n\
         \x20 compact                  write every due summary with the summarizer\n\n\
         Setup\n\
         \x20 init                     create the notebook, then print instructions for your agent\n\
         \x20 config [NAME=VALUE ...]  list the notebook's size settings, or set them\n\
         \x20 import <file>            bulk-add old notes, one \"YYYY-MM-DD text\" per line\n\n\
         Notebook folder: $MEMORY_DIR, or ~/.optmem/memory when that is unset.\n\
         Summarizer: the program named by $MNEM_SUMMARIZER; without one, the agent\n\
         is asked to write summaries itself."
    )
}

fn explain(context: &Context, failure: Failure) -> String {
    match failure {
        Failure::Notebook(NotebookError::Missing { dir }) => format!(
            "There is no notebook at {}.\nCreate one with: {} init\nOr set MEMORY_DIR to the folder of an existing one.",
            pretty(&dir, context.home.as_deref()),
            context.me
        ),
        Failure::Notebook(error) => error.to_string(),
        Failure::Said(message) => message,
    }
}

fn usage_error(context: &Context, form: &str) -> Failure {
    Failure::Said(format!("usage: {} {form}", context.me))
}

fn init(context: &Context, args: &[&str]) -> Outcome {
    if !args.is_empty() {
        return Err(usage_error(context, "init"));
    }
    let fresh = !context.dir.is_dir();
    let notebook = Notebook::init(&context.dir)?;
    if !context.dir.join("config").exists() {
        notebook.config().write(&context.dir, &context.me)?;
    }
    let shown = pretty(&context.dir, context.home.as_deref());
    let mut out = Vec::new();
    if fresh {
        out.push(format!("Created {shown}: a new, empty notebook."));
    } else {
        out.push(format!(
            "Found {shown}: {}.",
            count(notebook.len()?, "note", "notes")
        ));
    }
    out.push(format!(
        "Its sizes are in {shown}/config; the defaults suit most agents."
    ));
    out.push(String::new());
    out.push(
        "Add this to your agent's standing instructions (its system prompt, or a file such as SOUL.md):".to_owned(),
    );
    out.push(String::new());
    out.push(setup_block(
        &context.me,
        &shown,
        notebook.config().entry_chars(),
    ));
    Ok(Reply::done(&out))
}

fn setup_block(me: &str, dir: &str, chars: usize) -> String {
    format!(
        "## Memory (Mnem)\n\n\
         You keep a persistent notebook with `{me}`. Its files are in `{dir}`,\n\
         and only the tool changes them.\n\n\
         1. Begin each session with `{me} wake`. Read all it prints and follow\n\
         \x20  its instructions, part by part, until a part ends with \"You are awake.\"\n\
         2. Record what is worth remembering across sessions (decisions, facts about\n\
         \x20  the user or the project, results of substantial work, mistakes to avoid)\n\
         \x20  with `{me} note \"<single line, at most {chars} bytes>\"`. Do not repeat\n\
         \x20  what is already recorded. If the reply requests a summary, write it at once.\n\
         3. If two notes conflict, trust the newer one.\n\
         4. For older details, search with `{me} recall <pattern>`, or expand any\n\
         \x20  `#lo-hi` summary with `{me} zoom <lo>-<hi>`.\n\
         5. Only you write to the notebook. Subagents must not use `{me}`; say so\n\
         \x20  whenever you delegate to one."
    )
}

/// Who writes the summaries that are due.
enum Summaries {
    /// The agent, as in stock OptMem: no summarizer is set.
    ByAgent,
    /// The summarizer, in the background.
    BySummarizer,
    /// The agent for now, because the summarizer's last run failed; it is
    /// still started each time, so it takes over again once it works.
    Failing(String),
}

impl Summaries {
    fn of(context: &Context, notebook: &Notebook) -> Self {
        if context.summarizer.is_none() {
            return Self::ByAgent;
        }
        match compactor::failure(notebook.dir()) {
            Some(why) => Self::Failing(why),
            None => Self::BySummarizer,
        }
    }

    /// Whether the summarizer should start.
    fn starts(&self) -> bool {
        !matches!(self, Self::ByAgent)
    }

    /// The line that tells the agent why summaries fall to it, if they
    /// normally would not.
    fn excuse(&self) -> Option<String> {
        match self {
            Self::Failing(why) => Some(format!(
                "The summarizer is failing ({why}), so summaries fall to you until it works again."
            )),
            Self::ByAgent | Self::BySummarizer => None,
        }
    }
}

/// The reply `lines`, followed by the prompt for the next summary due among
/// the first `notes` notes when the agent is the one to write it. With a
/// summarizer the reply starts it instead.
fn with_next_summary(
    context: &Context,
    notebook: &Notebook,
    mut lines: Vec<String>,
    notes: u64,
) -> Outcome {
    let summaries = Summaries::of(context, notebook);
    let due = notebook.pending_count(notes)? > 0;
    if !matches!(summaries, Summaries::BySummarizer)
        && let Some(prompt) = next_nap(context, notebook, notes)?
    {
        lines.push(String::new());
        if let Some(excuse) = summaries.excuse() {
            lines.push(excuse);
        }
        lines.push(prompt);
    }
    let mut reply = Reply::done(&lines);
    reply.compact = due && summaries.starts();
    Ok(reply)
}

fn wake(context: &Context, notebook: &Notebook, args: &[&str]) -> Outcome {
    let form = "wake [part [T]]";
    if args.len() > 2 {
        return Err(usage_error(context, form));
    }
    let now = notebook.len()?;
    let part = match args.first() {
        Some(raw) => whole_number(raw).ok_or_else(|| usage_error(context, form))?,
        None => 1,
    };
    let snapshot = match args.get(1) {
        Some(raw) => whole_number(raw).ok_or_else(|| usage_error(context, form))?,
        None => now,
    };
    let me = &context.me;
    if snapshot > now {
        return Err(Failure::Said(format!(
            "The notebook holds only {}, fewer than T={snapshot}. Start again with: {me} wake",
            count(now, "note", "notes")
        )));
    }
    if snapshot == 0 {
        return Ok(Reply::done(&[
            format!("The notebook is empty. Save the first note with: {me} note \"<one line>\""),
            "You are awake.".to_owned(),
        ]));
    }
    let lines = match document(context, notebook, snapshot)? {
        Document::Lines(lines) => lines,
        Document::Refused(refusal) => {
            // The agent is asked as in stock OptMem; a summarizer, if there
            // is one, starts on the same work, and whichever writes first wins.
            let mut reply = Reply::refused(&refusal);
            reply.compact = Summaries::of(context, notebook).starts();
            return Ok(reply);
        }
    };
    let config = notebook.config();
    let parts = paginate(lines, config.part_lines(), config.part_chars());
    let total = u64::try_from(parts.len()).unwrap_or(u64::MAX);
    let chosen = part
        .checked_sub(1)
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| parts.get(index));
    let Some(chosen) = chosen else {
        return Err(Failure::Said(format!(
            "No part {part}: the notebook prints in {}. Run: {me} wake",
            count(total, "part", "parts")
        )));
    };
    let mut out = Vec::new();
    if total > 1 {
        out.push(format!(
            "Your notebook, part {part} of {total}, oldest first ({}).",
            count(snapshot, "note", "notes")
        ));
    }
    out.extend(chosen.iter().cloned());
    if part < total {
        // Harnesses that shorten long output keep its end, so the last line
        // carries the instruction to continue.
        out.push(format!(
            "Not finished: part {part} of {total} read. Continue with: {me} wake {} {snapshot}",
            part.saturating_add(1)
        ));
    } else {
        out.push("You are awake.".to_owned());
        return with_next_summary(context, notebook, out, snapshot);
    }
    Ok(Reply::done(&out))
}

/// The lines `wake` shows for the first `notes` notes, or the reply that
/// asks for the summary it is missing.
enum Document {
    Lines(Vec<String>),
    Refused(Vec<String>),
}

fn document(context: &Context, notebook: &Notebook, notes: u64) -> Result<Document, Failure> {
    let spans = cover::cover(notes, notebook.config().wake_lines())
        .map_err(|error| Failure::Said(error.to_string()))?;
    let mut lines = Vec::new();
    for span in spans {
        if span.is_note() {
            lines.push(notebook.note(span.lo())?.line());
            continue;
        }
        let summary = match notebook.summary(span)? {
            Some(summary) => summary,
            None => {
                if let Some(prompt) = next_nap(context, notebook, notes)? {
                    let due = notebook.pending_count(notes)?;
                    return Ok(Document::Refused(vec![
                        format!(
                            "Cannot wake yet: the notebook needs #{span}, which is not summarized."
                        ),
                        format!(
                            "Write the pending summaries ({due} in all) in order, starting with the one below, then run {} wake again.",
                            context.me
                        ),
                        String::new(),
                        prompt,
                    ]));
                }
                // Nothing is pending, yet the record reads as empty: either a
                // concurrent nap has just written it, or it is stored blank.
                notebook
                    .summary(span)?
                    .ok_or_else(|| blank(context, span))?
            }
        };
        lines.push(format!("#{span} {summary}"));
    }
    Ok(Document::Lines(lines))
}

/// Splits lines into parts of at most `max_lines` lines and `max_bytes`
/// bytes, counting each line's newline. A line longer than `max_bytes` gets
/// a part of its own.
fn paginate(lines: Vec<String>, max_lines: usize, max_bytes: usize) -> Vec<Vec<String>> {
    let mut parts = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut size: usize = 0;
    for line in lines {
        let bytes = line.len().saturating_add(1);
        if !current.is_empty()
            && (current.len() >= max_lines || size.saturating_add(bytes) > max_bytes)
        {
            parts.push(current);
            current = Vec::new();
            size = 0;
        }
        current.push(line);
        size = size.saturating_add(bytes);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

fn note(context: &Context, notebook: &Notebook, args: &[&str]) -> Outcome {
    let limit = notebook.config().entry_chars();
    let [raw] = args else {
        return Err(usage_error(
            context,
            &format!("note \"<one line, at most {limit} bytes>\""),
        ));
    };
    let text =
        Text::try_new(raw, limit).map_err(|error| Failure::Said(text_problem(&error, "note")))?;
    let id = notebook.append(&[(context.today.clone(), text)])?;
    let notes = id.checked_add(1).ok_or(NotebookError::Overflow)?;
    with_next_summary(context, notebook, vec![format!("Noted as #{id}.")], notes)
}

fn nap(context: &Context, notebook: &Notebook, args: &[&str]) -> Outcome {
    let notes = notebook.len()?;
    let mut out = Vec::new();
    match args {
        [] => {}
        [id, line] => {
            let block = parse_block(id)?;
            let Some(&next) = notebook.pending(notes, Some(1))?.first() else {
                return Ok(Reply::done(&["Nothing left to summarize.".to_owned()]));
            };
            if block == next {
                let text = Text::try_new(line, notebook.config().entry_chars())
                    .map_err(|error| Failure::Said(text_problem(&error, "summary")))?;
                out.push(match notebook.put_summary(block, &text)? {
                    PutOutcome::Saved => format!("Summary of {block} written."),
                    PutOutcome::NotNext => {
                        format!("{block} changed while you wrote it, by another writer; nothing was saved.")
                    }
                });
            } else if notebook.summary(block)?.is_some() {
                out.push(format!(
                    "{block} already has a summary; it was left as it is."
                ));
            } else {
                return Err(Failure::Said(format!(
                    "{id} is not next: summaries are built in order, and {next} comes first. See its prompt with: {} nap",
                    context.me
                )));
            }
        }
        _ => return Err(usage_error(context, "nap <lo>-<hi> \"<one line>\"")),
    }
    match next_nap(context, notebook, notes)? {
        None => out.push("Nothing left to summarize.".to_owned()),
        Some(prompt) => {
            if !args.is_empty() {
                out.push(String::new());
            }
            out.push(prompt);
        }
    }
    Ok(Reply::done(&out))
}

/// The prompt for the first summary due among the first `notes` notes.
fn next_nap(context: &Context, notebook: &Notebook, notes: u64) -> Result<Option<String>, Failure> {
    let Some(&block) = notebook.pending(notes, Some(1))?.first() else {
        return Ok(None);
    };
    let left = notebook.pending_count(notes)?.saturating_sub(1);
    nap_prompt(context, notebook, block, left).map(Some)
}

fn nap_prompt(
    context: &Context,
    notebook: &Notebook,
    block: Span,
    left: u64,
) -> Result<String, Failure> {
    let mut lines = vec![
        format!(
            "Condense notes #{block} into a single line of at most {} bytes.",
            notebook.config().entry_chars()
        ),
        "Favour facts and decisions that will matter later, and say nothing the notes do not."
            .to_owned(),
        String::new(),
    ];
    let input = Input::read(notebook, block)?.map_err(|half| blank(context, half))?;
    for line in input.lines {
        lines.push(format!("  {line}"));
    }
    lines.push(String::new());
    if left > 0 {
        lines.push(format!(
            "{} due after this one.",
            count(left, "more summary is", "more summaries are")
        ));
    }
    lines.push(format!("Run: {} nap {block} \"<your line>\"", context.me));
    Ok(lines.join("\n"))
}

fn recall(context: &Context, notebook: &Notebook, args: &[&str]) -> Outcome {
    let [pattern] = args else {
        return Err(usage_error(context, "recall <pattern>"));
    };
    let regex = RegexBuilder::new(pattern)
        .case_insensitive(true)
        .size_limit(RECALL_REGEX_BYTES)
        .build()
        .map_err(|error| {
            Failure::Said(format!(
                "Bad pattern: {error}\nPatterns use Rust regex syntax, which has no lookaround or backreferences."
            ))
        })?;
    // Matches beyond the output cap push out the oldest ones kept, so memory
    // stays bounded however loose the pattern is.
    let limit = notebook.config().part_chars();
    let mut kept: VecDeque<String> = VecDeque::new();
    let mut size: usize = 0;
    let mut hits: u64 = 0;
    for note in notebook.scan()? {
        let line = note?.line();
        if !regex.is_match(&line) {
            continue;
        }
        hits = hits.saturating_add(1);
        size = size.saturating_add(line.len().saturating_add(1));
        kept.push_back(line);
        while size > limit {
            let Some(oldest) = kept.pop_front() else {
                break;
            };
            size = size.saturating_sub(oldest.len().saturating_add(1));
        }
    }
    if hits == 0 {
        return Ok(Reply::done(&["Nothing matches.".to_owned()]));
    }
    let shown = u64::try_from(kept.len()).unwrap_or(u64::MAX);
    let mut out: Vec<String> = kept.into_iter().collect();
    if out.is_empty() {
        // Not even one match fits the cap. `memo` prints an empty line here;
        // keep the output the same shape.
        out.push(String::new());
    }
    if shown < hits {
        out.push(format!(
            "Showing the latest {shown} of {}; a narrower pattern shows the rest.",
            count(hits, "match", "matches")
        ));
    } else {
        out.push(format!("{} found.", count(hits, "match", "matches")));
    }
    Ok(Reply::done(&out))
}

fn zoom(context: &Context, notebook: &Notebook, args: &[&str]) -> Outcome {
    let [id] = args else {
        return Err(usage_error(context, "zoom <lo>-<hi>"));
    };
    let block = parse_block(id)?;
    let notes = notebook.len()?;
    if block.lo() >= notes {
        return Err(Failure::Said(format!(
            "#{id} starts past the end of the notebook, which holds {}. See what is there with: {} wake",
            count(notes, "note", "notes"),
            context.me
        )));
    }
    let mut out = Vec::new();
    if let Some((first, second)) = block.halves() {
        for half in [first, second] {
            if half.lo() >= notes {
                // Notes not written yet.
                continue;
            }
            if half.is_note() {
                out.push(notebook.note(half.lo())?.line());
            } else {
                let summary = notebook
                    .summary(half)?
                    .unwrap_or_else(|| "not summarized yet".to_owned());
                out.push(format!("#{half} {summary}"));
            }
        }
    }
    Ok(Reply::done(&out))
}

fn forget(context: &Context, notebook: &Notebook, args: &[&str]) -> Outcome {
    let [id] = args else {
        return Err(usage_error(context, "forget <lo>-<hi>"));
    };
    let block = parse_block(id)?;
    let gone = notebook.forget(block)?;
    let Some(first) = gone.first() else {
        return Err(Failure::Said(format!("{id} has no summary to forget.")));
    };
    let dropped = count(
        u64::try_from(gone.len()).unwrap_or(u64::MAX),
        "summary",
        "summaries",
    );
    let summaries = Summaries::of(context, notebook);
    let line = match summaries {
        Summaries::BySummarizer => {
            format!("Dropped {dropped}, starting at {first}; the summarizer is rewriting them.")
        }
        Summaries::ByAgent | Summaries::Failing(_) => format!(
            "Dropped {dropped}, starting at {first}; they are due again. Rewrite them with: {} nap",
            context.me
        ),
    };
    let mut reply = Reply::done(&[line]);
    reply.compact = summaries.starts();
    Ok(reply)
}

fn config(context: &Context, notebook: &Notebook, args: &[&str]) -> Outcome {
    let mut config = *notebook.config();
    for arg in args {
        let setting = arg.split_once('=').and_then(|(name, value)| {
            Knob::from_name(name.trim_matches(text::is_python_space)).map(|knob| (knob, value))
        });
        let Some((knob, value)) = setting else {
            let names: Vec<&str> = Knob::ALL.iter().map(|knob| knob.name()).collect();
            return Err(usage_error(
                context,
                &format!(
                    "config [NAME=VALUE ...]   (NAME is one of {})",
                    names.join(", ")
                ),
            ));
        };
        let value = value.trim_matches(text::is_python_space);
        if value.is_empty() {
            config.set(knob, None);
        } else {
            let size = knob
                .check(value)
                .map_err(|error| Failure::Said(format!("{error}.")))?;
            config.set(knob, Some(size));
        }
    }
    if !args.is_empty() {
        config.write(&context.dir, &context.me)?;
    }
    let mut out = Vec::new();
    for knob in Knob::ALL {
        let default = match config.overridden(knob) {
            Some(_) => format!(" (default {})", knob.default_value()),
            None => String::new(),
        };
        out.push(format!(
            "{:<12} {:<7} {}{default}",
            knob.name(),
            config.get(knob),
            knob.meaning()
        ));
    }
    Ok(Reply::done(&out))
}

fn import(context: &Context, notebook: &Notebook, args: &[&str]) -> Outcome {
    let [file] = args else {
        return Err(usage_error(
            context,
            "import <file>   (lines of 'YYYY-MM-DD <text>')",
        ));
    };
    let shown = pretty(Path::new(file), context.home.as_deref());
    let bytes = fs::read(file).map_err(|error| Failure::Said(format!("{shown}: {error}")))?;
    let contents = String::from_utf8(bytes).map_err(|_not_utf8| {
        Failure::Said(format!(
            "{shown} is not UTF-8 text. Convert it, then import again."
        ))
    })?;
    let limit = notebook.config().entry_chars();
    let notes = notebook.len()?;
    let mut last = match notes.checked_sub(1) {
        Some(id) => Some(notebook.note(id)?.day),
        None => None,
    };
    let contents = contents.replace("\r\n", "\n").replace('\r', "\n");
    let mut batch = Vec::new();
    for (number, line) in (1_usize..).zip(contents.split('\n')) {
        if line.trim_matches(text::is_python_space).is_empty() {
            continue;
        }
        let (date, rest) = line.split_once(' ').unwrap_or((line, ""));
        let day = Day::try_new(date).map_err(|error| {
            Failure::Said(match error {
                DayError::Format => {
                    format!("line {number}: expected 'YYYY-MM-DD <text>', got: {line}")
                }
                DayError::NotADate => format!("line {number}: {date} is not a real date."),
            })
        })?;
        if let Some(previous) = last.as_deref()
            && day.as_str() < previous
        {
            return Err(Failure::Said(format!(
                "line {number}: {date} comes before the note above it ({previous})."
            )));
        }
        let text = Text::try_new(rest, limit).map_err(|error| {
            Failure::Said(format!("line {number}: {}", text_problem(&error, "note")))
        })?;
        last = Some(day.as_str().to_owned());
        batch.push((day, text));
    }
    if batch.is_empty() {
        return Err(Failure::Said(format!("{shown} has no notes.")));
    }
    let added = u64::try_from(batch.len()).unwrap_or(u64::MAX);
    let first = notebook.append(&batch)?;
    let mut out = vec![format!(
        "Added {} as #{first} to #{}.",
        count(added, "note", "notes"),
        first.saturating_add(added).saturating_sub(1)
    )];
    let due = notebook.pending_count(notebook.len()?)?;
    let summaries = Summaries::of(context, notebook);
    if due > 0 {
        let due = count(due, "summary is", "summaries are");
        out.push(match summaries {
            Summaries::BySummarizer => format!("{due} now due; the summarizer is writing them."),
            Summaries::ByAgent | Summaries::Failing(_) => {
                format!("{due} now due. Write them with: {} nap", context.me)
            }
        });
    }
    let mut reply = Reply::done(&out);
    reply.compact = due > 0 && summaries.starts();
    Ok(reply)
}

fn compact(context: &Context, notebook: &Notebook, args: &[&str]) -> Outcome {
    if !args.is_empty() {
        return Err(usage_error(context, "compact"));
    }
    let Some(summarizer) = &context.summarizer else {
        return Err(Failure::Said(format!(
            "No summarizer is set. Name the program in MNEM_SUMMARIZER, or write the summaries with: {} nap",
            context.me
        )));
    };
    let mut model = summarizer.clone();
    let report = match compactor::run(notebook, &mut model)? {
        Run::Busy => {
            return Ok(Reply::done(&[
                "A summarizer is already running on this notebook.".to_owned(),
            ]));
        }
        Run::Ran(report) => report,
    };
    let mut out = Vec::new();
    for (block, how) in &report.written {
        out.push(match how {
            Written::ByModel => format!("Wrote {block}."),
            Written::CutReply => format!(
                "Wrote {block} from the model's shortest line, cut to fit: every try was too long."
            ),
            Written::CutInput => {
                format!("Wrote {block} from its input, cut to fit: the model declined.")
            }
        });
    }
    let failed = match &report.stopped {
        None => false,
        Some(Stop::Unavailable(why)) => {
            out.push(format!("Stopped: the summarizer could not run: {why}"));
            true
        }
        Some(Stop::BlankHalf { block, half }) => {
            out.push(format!(
                "Stopped: {block} is built from #{half}, whose summary record is empty. Drop it with: {} forget {half}",
                context.me
            ));
            true
        }
    };
    let due = notebook.pending_count(notebook.len()?)?;
    if due == 0 {
        out.push("Nothing left to summarize.".to_owned());
    } else {
        out.push(format!(
            "{} still due.",
            count(due, "summary is", "summaries are")
        ));
    }
    Ok(if failed {
        Reply::refused(&out)
    } else {
        Reply::done(&out)
    })
}

fn parse_block(raw: &str) -> Result<Span, Failure> {
    Span::parse_block(raw).map_err(|error| {
        Failure::Said(match error {
            SpanError::Format | SpanError::Overflow => {
                format!(
                    "'{raw}' is not a block id. Copy one from the prompt or from wake, like 16-31."
                )
            }
            SpanError::NotAligned | SpanError::SingleNote => {
                format!("{raw} is not a block. Copy one from the prompt or from wake, like 16-31.")
            }
        })
    })
}

fn blank(context: &Context, block: Span) -> Failure {
    Failure::Said(format!(
        "#{block} has an empty summary record. Drop it with: {} forget {block}",
        context.me
    ))
}

fn text_problem(error: &TextError, what: &str) -> String {
    match error {
        TextError::Empty => format!("Nothing to save: the {what} is blank."),
        TextError::MultiLine { lines } => format!(
            "The {what} has {lines} lines, and it must be one. Join them, or save them separately."
        ),
        TextError::TooLong { bytes, limit } => format!(
            "Too long: {bytes} bytes, and a {what} holds at most {limit}. Accented letters take 2 bytes and emoji 4. Shorten it."
        ),
    }
}

/// A count of ASCII digits, saturating at `u64::MAX`; `None` for anything
/// else.
fn whole_number(raw: &str) -> Option<u64> {
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(raw.parse::<u64>().unwrap_or(u64::MAX))
}

fn count(n: u64, one: &str, many: &str) -> String {
    if n == 1 {
        format!("1 {one}")
    } else {
        format!("{n} {many}")
    }
}

fn joined(lines: &[String]) -> String {
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

/// `path` as a person would type it, with the home directory written `~`.
pub fn pretty(path: &Path, home: Option<&Path>) -> String {
    if let Some(rest) = home.and_then(|home| path.strip_prefix(home).ok())
        && !rest.as_os_str().is_empty()
    {
        return format!("~/{}", rest.display());
    }
    path.display().to_string()
}

/// How messages should name this program so the agent can run what they
/// print: the bare name when it was found on `PATH`, otherwise its full
/// path. `program` is the first command-line argument.
pub fn invoked_as(program: Option<&Path>, cwd: Option<&Path>, home: Option<&Path>) -> String {
    let Some(program) = program else {
        return "mnem".to_owned();
    };
    if program.components().count() == 1 && !program.is_absolute() {
        return program.display().to_string();
    }
    let full: PathBuf = match cwd {
        Some(cwd) if program.is_relative() => cwd.join(program).components().collect(),
        _ => program.to_path_buf(),
    };
    pretty(&full, home)
}

/// The notebook directory: `$MEMORY_DIR` when set and not empty, with a
/// leading `~` meaning the home directory, otherwise `~/.optmem/memory`, the
/// directory `memo` uses. A relative path is taken from `cwd`. `None` when
/// the home directory is needed but unknown.
pub fn memory_dir(
    variable: Option<OsString>,
    home: Option<&Path>,
    cwd: Option<&Path>,
) -> Option<PathBuf> {
    let dir = match variable.filter(|value| !value.is_empty()) {
        Some(value) => {
            let path = PathBuf::from(value);
            match path.strip_prefix("~") {
                Ok(rest) => home?.join(rest),
                Err(_not_home_relative) => path,
            }
        }
        None => home?.join(".optmem").join("memory"),
    };
    match cwd {
        Some(cwd) if dir.is_relative() => Some(cwd.join(dir)),
        _ => Some(dir),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::fs;
    use std::os::unix::ffi::OsStringExt;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::time::Duration;

    use super::Context;
    use super::Reply;
    use crate::cli;
    use crate::config::Config;
    use crate::day::Day;
    use crate::model::CommandModel;
    use crate::testing::Scratch;

    fn context(scratch: &Scratch) -> Context {
        Context {
            dir: scratch.path.clone(),
            me: "mnem".to_owned(),
            home: None,
            today: Day::try_new("2026-10-31").expect("a real date"),
            summarizer: None,
        }
    }

    fn run(context: &Context, words: &[&str]) -> Reply {
        let args: Vec<OsString> = words.iter().map(OsString::from).collect();
        cli::run(context, &args)
    }

    /// Runs a command that must succeed and returns its standard output.
    fn ok(context: &Context, words: &[&str]) -> String {
        let reply = run(context, words);
        assert_eq!(reply.code, 0, "{words:?} succeeds: {}", reply.stderr);
        reply.stdout
    }

    /// Runs a command that must fail and returns its error message.
    fn fails(context: &Context, words: &[&str]) -> String {
        let reply = run(context, words);
        assert_eq!(reply.code, 1, "{words:?} fails: {}", reply.stdout);
        reply.stderr
    }

    fn notebook_with(scratch: &Scratch, notes: u64) -> Context {
        let context = context(scratch);
        ok(&context, &["init"]);
        for i in 0..notes {
            ok(&context, &["note", &format!("note {i}")]);
        }
        context
    }

    /// Writes every pending summary, as an agent answering the prompts would.
    fn nap_all(context: &Context) {
        loop {
            let prompt = ok(context, &["nap"]);
            let Some(run_line) = prompt
                .lines()
                .find(|line| line.starts_with("Run: mnem nap "))
            else {
                break;
            };
            let block = run_line
                .split(' ')
                .nth(3)
                .expect("the run line names the block");
            ok(context, &["nap", block, &format!("summary {block}")]);
        }
    }

    #[test]
    fn commands_other_than_init_need_a_notebook() {
        let scratch = Scratch::new();
        let context = context(&scratch);
        let error = fails(&context, &["wake"]);
        assert!(
            error.contains("There is no notebook at"),
            "names the problem: {error}"
        );
        assert!(error.contains("mnem init"), "says how to fix it: {error}");
        assert!(!scratch.path.exists(), "wake must not create the notebook");
    }

    #[test]
    fn init_creates_the_notebook_and_is_safe_to_repeat() {
        let scratch = Scratch::new();
        let context = context(&scratch);
        let first = ok(&context, &["init"]);
        assert!(first.starts_with("Created "), "a new notebook: {first}");
        assert!(
            first.contains("You are awake."),
            "the setup block explains wake"
        );
        let config = fs::read_to_string(scratch.path.join("config")).expect("config written");
        assert_eq!(
            Config::parse(&config).expect("it parses"),
            Config::default(),
            "every size is at its default"
        );
        ok(&context, &["note", "kept"]);
        let again = ok(&context, &["init"]);
        assert!(again.starts_with("Found "), "an existing notebook: {again}");
        assert!(again.contains("1 note."), "and its size: {again}");
        assert!(
            fails(&context, &["init", "extra"]).contains("usage"),
            "no arguments"
        );
    }

    #[test]
    fn note_saves_one_line_and_asks_for_due_summaries() {
        let scratch = Scratch::new();
        let context = notebook_with(&scratch, 1);
        let reply = ok(&context, &["note", "  second  "]);
        assert!(reply.starts_with("Noted as #1."), "the new id: {reply}");
        assert!(
            reply.contains("  #0 2026-10-31 note 0"),
            "the prompt shows the notes"
        );
        assert!(
            reply.contains("  #1 2026-10-31 second"),
            "trimmed, with today's date"
        );
        assert!(
            reply.contains("Run: mnem nap 0-1"),
            "and how to answer: {reply}"
        );
        assert!(fails(&context, &["note", "  "]).contains("blank"), "blank");
        assert!(
            fails(&context, &["note", "one\ntwo"]).contains("2 lines"),
            "multi-line"
        );
        let long = "x".repeat(281);
        assert!(
            fails(&context, &["note", &long]).contains("281 bytes"),
            "too long"
        );
        assert!(fails(&context, &["note"]).contains("usage"), "no text");
        let log = fs::metadata(scratch.path.join("LOG.txt"))
            .expect("log")
            .len();
        assert_eq!(log, 2 * 320, "rejected notes are not saved");
    }

    #[test]
    fn nap_writes_summaries_in_order() {
        let scratch = Scratch::new();
        let context = notebook_with(&scratch, 4);
        assert!(
            ok(&context, &["nap"]).contains("Run: mnem nap 0-1"),
            "0-1 is first"
        );
        assert!(
            fails(&context, &["nap", "2-3", "early"]).contains("0-1 comes first"),
            "out of order"
        );
        assert!(
            fails(&context, &["nap", "0-2", "x"]).contains("not a block"),
            "unaligned"
        );
        assert!(
            fails(&context, &["nap", "zero", "x"]).contains("not a block id"),
            "not an id"
        );
        let saved = ok(&context, &["nap", "0-1", "first pair"]);
        assert!(
            saved.starts_with("Summary of 0-1 written.\n\n"),
            "saved: {saved}"
        );
        assert!(saved.contains("Run: mnem nap 2-3"), "then the next prompt");
        assert!(
            ok(&context, &["nap", "0-1", "again"]).starts_with("0-1 already has a summary"),
            "a settled block is reported, not rewritten"
        );
        nap_all(&context);
        assert_eq!(
            ok(&context, &["nap"]),
            "Nothing left to summarize.\n",
            "all built"
        );
        let zoomed = ok(&context, &["zoom", "0-3"]);
        assert_eq!(
            zoomed, "#0-1 first pair\n#2-3 summary 2-3\n",
            "zoom shows both halves"
        );
    }

    #[test]
    fn large_blocks_are_summarized_from_their_halves() {
        let scratch = Scratch::new();
        let context = notebook_with(&scratch, 32);
        let mut prompt = ok(&context, &["nap"]);
        while !prompt.contains("nap 0-31") {
            let block = prompt
                .lines()
                .find_map(|line| line.strip_prefix("Run: mnem nap "))
                .and_then(|rest| rest.split(' ').next())
                .expect("a prompt")
                .to_owned();
            prompt = ok(&context, &["nap", &block, &format!("summary {block}")]);
        }
        assert!(
            prompt.contains("  #0-15 summary 0-15\n  #16-31 summary 16-31\n"),
            "a 32-note block is built from its halves: {prompt}"
        );
    }

    #[test]
    fn wake_prints_the_notebook_in_parts() {
        let scratch = Scratch::new();
        let context = notebook_with(&scratch, 6);
        nap_all(&context);
        let whole = ok(&context, &["wake"]);
        assert!(
            whole.starts_with("#0 2026-10-31 note 0\n"),
            "notes verbatim"
        );
        assert!(
            whole.ends_with("#5 2026-10-31 note 5\nYou are awake.\n"),
            "{whole}"
        );
        ok(&context, &["config", "WAKE_LINES=3", "PART_LINES=2"]);
        let first = ok(&context, &["wake"]);
        assert_eq!(
            first,
            "Your notebook, part 1 of 2, oldest first (6 notes).\n#0-3 summary 0-3\n#4 2026-10-31 note 4\nNot finished: part 1 of 2 read. Continue with: mnem wake 2 6\n",
            "older notes are summarized and the output is paged"
        );
        ok(&context, &["note", "seventh"]);
        let second = ok(&context, &["wake", "2", "6"]);
        assert!(
            second.ends_with("#5 2026-10-31 note 5\nYou are awake.\n"),
            "part 2 is read as of 6 notes, not 7: {second}"
        );
        assert!(
            fails(&context, &["wake", "3", "6"]).contains("No part 3"),
            "no part 3"
        );
        assert!(
            fails(&context, &["wake", "0"]).contains("No part 0"),
            "parts count from 1"
        );
        assert!(
            fails(&context, &["wake", "1", "9"]).contains("fewer than T=9"),
            "T past the log"
        );
        assert!(
            fails(&context, &["wake", "x"]).contains("usage"),
            "not a number"
        );
    }

    #[test]
    fn wake_refuses_only_when_it_needs_a_missing_summary() {
        let scratch = Scratch::new();
        let context = notebook_with(&scratch, 4);
        let reply = run(&context, &["wake"]);
        assert_eq!(reply.code, 0, "four notes fit without summaries");
        assert!(
            reply.stdout.contains("You are awake.\n\nCondense"),
            "then a prompt"
        );
        ok(&context, &["config", "WAKE_LINES=1"]);
        let refused = run(&context, &["wake"]);
        assert_eq!(refused.code, 1, "one line needs the 0-3 summary");
        assert!(
            refused
                .stdout
                .starts_with("Cannot wake yet: the notebook needs #0-3"),
            "{}",
            refused.stdout
        );
        assert!(
            refused.stdout.contains("Run: mnem nap 0-1"),
            "the first summary due"
        );
    }

    #[test]
    fn an_empty_notebook_wakes_at_once() {
        let scratch = Scratch::new();
        let context = notebook_with(&scratch, 0);
        assert!(
            ok(&context, &["wake"]).ends_with("You are awake.\n"),
            "nothing to read"
        );
    }

    #[test]
    fn recall_keeps_the_newest_matches_that_fit() {
        let scratch = Scratch::new();
        let context = notebook_with(&scratch, 5);
        assert_eq!(
            ok(&context, &["recall", "NOTE [13]"]),
            "#1 2026-10-31 note 1\n#3 2026-10-31 note 3\n2 matches found.\n",
            "case is ignored"
        );
        assert_eq!(
            ok(&context, &["recall", "^#4 "]),
            "#4 2026-10-31 note 4\n1 match found.\n",
            "the id is searchable"
        );
        assert_eq!(
            ok(&context, &["recall", "absent"]),
            "Nothing matches.\n",
            "no match"
        );
        ok(&context, &["config", "PART_CHARS=50"]);
        assert_eq!(
            ok(&context, &["recall", "note"]),
            "#3 2026-10-31 note 3\n#4 2026-10-31 note 4\nShowing the latest 2 of 5 matches; a narrower pattern shows the rest.\n",
            "two 21-byte lines fit in 50 bytes"
        );
        assert!(
            fails(&context, &["recall", "("]).contains("Bad pattern"),
            "bad regex"
        );
    }

    #[test]
    fn zoom_opens_a_block_down_to_notes() {
        let scratch = Scratch::new();
        let context = notebook_with(&scratch, 3);
        assert_eq!(
            ok(&context, &["zoom", "0-1"]),
            "#0 2026-10-31 note 0\n#1 2026-10-31 note 1\n",
            "a pair opens into notes"
        );
        assert_eq!(
            ok(&context, &["zoom", "0-3"]),
            "#0-1 not summarized yet\n#2-3 not summarized yet\n",
            "halves without summaries, the second only partly written"
        );
        assert_eq!(
            ok(&context, &["zoom", "2-3"]),
            "#2 2026-10-31 note 2\n",
            "a note not written yet is left out"
        );
        assert!(
            fails(&context, &["zoom", "4-7"]).contains("past the end"),
            "past the log"
        );
    }

    #[test]
    fn forget_drops_a_summary_and_what_was_built_on_it() {
        let scratch = Scratch::new();
        let context = notebook_with(&scratch, 4);
        nap_all(&context);
        assert_eq!(
            ok(&context, &["forget", "2-3"]),
            "Dropped 2 summaries, starting at 2-3; they are due again. Rewrite them with: mnem nap\n",
            "2-3 and 0-3"
        );
        assert!(
            ok(&context, &["nap"]).contains("nap 2-3"),
            "2-3 is due again"
        );
        assert!(
            fails(&context, &["forget", "2-3"]).contains("2-3 has no summary to forget"),
            "gone"
        );
    }

    #[test]
    fn config_shows_sets_and_resets_sizes() {
        let scratch = Scratch::new();
        let context = notebook_with(&scratch, 0);
        assert!(
            ok(&context, &["config"]).starts_with("WAKE_LINES   96      how many lines wake shows"),
            "defaults"
        );
        let set = ok(&context, &["config", " wake_lines = 120 "]);
        assert!(set.starts_with("WAKE_LINES   120     "), "set: {set}");
        assert!(
            set.lines()
                .next()
                .is_some_and(|line| line.ends_with("(default 96)")),
            "marked: {set}"
        );
        assert!(
            fails(&context, &["config", "ENTRY_CHARS=281"]).contains("at most 280"),
            "too large"
        );
        assert!(
            fails(&context, &["config", "COLOR=3"]).contains("usage"),
            "unknown name"
        );
        assert!(
            fails(&context, &["config", "WAKE_LINES"]).contains("usage"),
            "no value"
        );
        let file = fs::read_to_string(scratch.path.join("config")).expect("config");
        assert_eq!(
            Config::parse(&file).expect("parses").wake_lines(),
            120,
            "failed edits leave the file alone"
        );
        let reset = ok(&context, &["config", "WAKE_LINES="]);
        assert!(
            reset.starts_with("WAKE_LINES   96      "),
            "an empty value resets: {reset}"
        );
    }

    #[test]
    fn import_loads_dated_lines_in_order() {
        let scratch = Scratch::new();
        let context = notebook_with(&scratch, 0);
        let file = scratch.path.join("import.txt");
        let file_arg = file.to_str().expect("a UTF-8 path");
        fs::write(
            &file,
            "2026-01-02 first\r\n\n2026-01-02 second\r2026-03-04  third ",
        )
        .expect("write");
        assert_eq!(
            ok(&context, &["import", file_arg]),
            "Added 3 notes as #0 to #2.\n1 summary is now due. Write them with: mnem nap\n",
            "every line ending is accepted and blank lines skipped"
        );
        assert_eq!(
            ok(&context, &["recall", "third"]),
            "#2 2026-03-04 third\n1 match found.\n",
            "the text is trimmed and keeps its own date"
        );
        let cases = [
            ("2026-01-01 too early\n", "comes before"),
            ("2026-02-30 no such day\n", "not a real date"),
            ("yesterday text\n", "expected 'YYYY-MM-DD <text>'"),
            ("2026-05-01\n", "blank"),
            ("\n\n", "has no notes"),
        ];
        for (contents, problem) in cases {
            fs::write(&file, contents).expect("write");
            let error = fails(&context, &["import", file_arg]);
            assert!(
                error.contains(problem),
                "{contents:?} gives {problem:?}: {error}"
            );
        }
        fs::write(&file, b"2026-05-01 \xff\n").expect("write");
        assert!(
            fails(&context, &["import", file_arg]).contains("not UTF-8"),
            "binary input"
        );
        assert!(
            ok(&context, &["wake"]).contains("#2 2026-03-04 third"),
            "failed imports add nothing"
        );
    }

    #[test]
    fn unknown_commands_and_bad_arguments_fail() {
        let scratch = Scratch::new();
        let context = notebook_with(&scratch, 0);
        let reply = run(&context, &["bogus"]);
        assert_eq!(reply.code, 1, "unknown command");
        assert!(
            reply.stderr.starts_with("No such command: bogus"),
            "{}",
            reply.stderr
        );
        assert!(
            ok(&context, &[]).contains("Usage: mnem <command>"),
            "no command shows usage"
        );
        let not_utf8 = vec![OsString::from("note"), OsString::from_vec(vec![0xff])];
        assert_eq!(
            cli::run(&context, &not_utf8).code,
            1,
            "arguments must be UTF-8"
        );
    }

    /// A context whose summarizer is a shell script answering `line`.
    fn with_summarizer(scratch: &Scratch, line: &str) -> Context {
        let context = notebook_with(scratch, 0);
        let program = scratch.path.join("summarizer.sh");
        fs::write(
            &program,
            format!("#!/bin/sh\ncat > /dev/null\necho '{line}'\n"),
        )
        .expect("script");
        let mut permissions = fs::metadata(&program).expect("script").permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&program, permissions).expect("executable");
        Context {
            summarizer: Some(CommandModel::new(program, Duration::from_secs(30))),
            ..context
        }
    }

    #[test]
    fn with_a_summarizer_the_agent_is_not_asked_for_summaries() {
        let scratch = Scratch::new();
        let context = with_summarizer(&scratch, "a pair");
        let first = run(&context, &["note", "one"]);
        assert_eq!(first.stdout, "Noted as #0.\n", "nothing due yet");
        assert!(!first.compact, "so no summarizer");
        let second = run(&context, &["note", "two"]);
        assert_eq!(second.stdout, "Noted as #1.\n", "no prompt for the agent");
        assert!(second.compact, "the summarizer starts instead");
        let woken = run(&context, &["wake"]);
        assert!(
            woken.stdout.ends_with("You are awake.\n"),
            "no prompt after wake"
        );
        assert!(woken.compact, "wake starts it too while summaries are due");
        assert_eq!(
            ok(&context, &["compact"]),
            "Wrote 0-1.\nNothing left to summarize.\n",
            "compact writes them"
        );
        assert!(!run(&context, &["wake"]).compact, "nothing due any more");
        assert_eq!(
            ok(&context, &["zoom", "0-1"]),
            "#0 2026-10-31 one\n#1 2026-10-31 two\n",
            "notes kept"
        );
        assert!(
            ok(&context, &["wake"]).contains("#0 2026-10-31 one"),
            "wake shows notes"
        );
    }

    #[test]
    fn compact_reports_an_unreachable_summarizer() {
        let scratch = Scratch::new();
        let context = with_summarizer(&scratch, "unused");
        ok(&context, &["note", "one"]);
        ok(&context, &["note", "two"]);
        let broken = Context {
            summarizer: Some(CommandModel::new(
                scratch.path.join("absent"),
                Duration::from_secs(1),
            )),
            ..context.clone()
        };
        let reply = run(&broken, &["compact"]);
        assert_eq!(reply.code, 1, "a failed run");
        assert!(
            reply
                .stdout
                .contains("Stopped: the summarizer could not run"),
            "{}",
            reply.stdout
        );
        assert!(
            reply.stdout.ends_with("1 summary is still due.\n"),
            "{}",
            reply.stdout
        );
        let noted = run(&broken, &["note", "three"]);
        assert!(
            noted.stdout.contains("The summarizer is failing (")
                && noted.stdout.contains("Run: mnem nap 0-1"),
            "while it fails, the agent is told why and asked instead: {}",
            noted.stdout
        );
        assert!(noted.compact, "and the summarizer is still tried");
        assert!(
            run(&broken, &["wake"])
                .stdout
                .contains("The summarizer is failing ("),
            "wake says so too"
        );
        assert_eq!(
            ok(&context, &["compact"]),
            "Wrote 0-1.\nNothing left to summarize.\n",
            "a working summarizer catches up"
        );
        assert_eq!(
            run(&context, &["note", "four"]).stdout,
            "Noted as #3.\n",
            "and the agent is left alone again"
        );
        let without = Context {
            summarizer: None,
            ..context
        };
        assert!(
            fails(&without, &["compact"]).contains("No summarizer is set"),
            "none set"
        );
    }

    #[test]
    fn import_and_forget_hand_their_work_to_the_summarizer() {
        let scratch = Scratch::new();
        let context = with_summarizer(&scratch, "a summary");
        let file = scratch.path.join("import.txt");
        fs::write(&file, "2026-01-01 one\n2026-01-01 two\n").expect("write");
        let imported = run(&context, &["import", file.to_str().expect("UTF-8 path")]);
        assert_eq!(
            imported.stdout,
            "Added 2 notes as #0 to #1.\n1 summary is now due; the summarizer is writing them.\n",
            "import leaves the summary to the summarizer"
        );
        assert!(imported.compact, "and starts it");
        ok(&context, &["compact"]);
        let forgot = run(&context, &["forget", "0-1"]);
        assert_eq!(
            forgot.stdout,
            "Dropped 1 summary, starting at 0-1; the summarizer is rewriting them.\n",
            "forget too"
        );
        assert!(forgot.compact, "and starts it");
    }

    #[test]
    fn paths_and_names_are_shown_runnable() {
        let home = Path::new("/home/ada");
        assert_eq!(
            cli::pretty(Path::new("/home/ada/notes"), Some(home)),
            "~/notes",
            "home is folded"
        );
        assert_eq!(
            cli::pretty(Path::new("/srv/notes"), Some(home)),
            "/srv/notes",
            "elsewhere"
        );
        let cwd = Path::new("/home/ada/bin");
        assert_eq!(
            cli::invoked_as(Some(Path::new("mnem")), Some(cwd), Some(home)),
            "mnem",
            "found on PATH"
        );
        assert_eq!(
            cli::invoked_as(Some(Path::new("./mnem")), Some(cwd), Some(home)),
            "~/bin/mnem",
            "a relative path is made absolute"
        );
        assert_eq!(
            cli::memory_dir(None, Some(home), Some(cwd)),
            Some(home.join(".optmem").join("memory")),
            "memo's default"
        );
        assert_eq!(
            cli::memory_dir(Some(OsString::from("~/mine")), Some(home), Some(cwd)),
            Some(home.join("mine")),
            "a leading ~ is home"
        );
        assert_eq!(
            cli::memory_dir(Some(OsString::from("rel")), Some(home), Some(cwd)),
            Some(cwd.join("rel")),
            "relative to the working directory"
        );
        assert_eq!(
            cli::memory_dir(Some(OsString::new()), None, Some(cwd)),
            None,
            "an empty variable is unset, and there is no home"
        );
    }
}
