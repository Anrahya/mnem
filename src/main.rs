//! The `mnem` binary: reads the environment and the clock, runs one command,
//! writes its reply, and starts the summarizer when summaries are due.

use std::env;
use std::ffi::OsString;
use std::io;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;
use std::process::ExitCode;
use std::process::Stdio;
use std::time::Duration;

use jiff::Zoned;
use jiff::civil::Date;
use mnem::cli;
use mnem::cli::Context;
use mnem::cli::Reply;
use mnem::day::Day;
use mnem::model::CommandModel;

/// How long one model call may take before Mnem gives up on it. Dream's
/// call is the long one.
const MODEL_TIMEOUT: Duration = Duration::from_secs(600);

fn main() -> ExitCode {
    let mut args = env::args_os();
    let program = args.next().map(PathBuf::from);
    let rest: Vec<OsString> = args.collect();
    let context = match context(program) {
        Ok(context) => context,
        Err(message) => return emit(&Reply::failed(&message)),
    };
    let reply = cli::run(&context, &rest);
    let code = emit(&reply);
    if reply.compact {
        start_background(&context, &["compact"]);
    }
    if reply.dream {
        start_background(&context, &["dream", "compact"]);
    }
    code
}

fn context(program: Option<PathBuf>) -> Result<Context, String> {
    let home = env::home_dir();
    let cwd = env::current_dir().ok();
    let me = cli::invoked_as(program.as_deref(), cwd.as_deref(), home.as_deref());
    let dir = cli::memory_dir(env::var_os("MEMORY_DIR"), home.as_deref(), cwd.as_deref())
        .ok_or("Cannot tell where the notebook is: set MEMORY_DIR, or HOME.")?;
    let model = env::var_os("MNEM_MODEL")
        .filter(|program| !program.is_empty())
        .map(|program| CommandModel::new(PathBuf::from(program), MODEL_TIMEOUT));
    // Read the clock once, so the day and the time always agree.
    let now = Zoned::now().datetime();
    Ok(Context {
        dir,
        me,
        home,
        today: today(now.date())?,
        now,
        model,
    })
}

/// Today in the local time zone: the date `memo` would give a note written
/// now. Without a known time zone this falls back to UTC.
fn today(date: Date) -> Result<Day, String> {
    Day::try_new(&date.to_string())
        .map_err(|error| format!("Today's date, {date}, cannot date a note: {error}."))
}

/// Starts `mnem <args>` (the summarizer, or a dream) on the same notebook
/// and does not wait for it. The agent's command returns at once; the work
/// lands when the model is done. If the start fails, nothing is lost: due
/// summaries stay due, and `wake` asks the agent for any it needs, and the
/// next dream reads the notes this one would have.
fn start_background(context: &Context, args: &[&str]) {
    let Ok(program) = env::current_exe() else {
        return;
    };
    let mut command = Command::new(program);
    command
        .args(args)
        .env("MEMORY_DIR", &context.dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // Its own process group, so a terminal closing behind the agent's
    // command does not stop it.
    #[cfg(unix)]
    command.process_group(0);
    drop(command.spawn());
}

/// The only place Mnem writes to the terminal. It writes through `io::Write`
/// rather than `print!`, so a closed pipe (`mnem recall x | head`) becomes a
/// failed exit instead of a panic.
fn emit(reply: &Reply) -> ExitCode {
    let mut stdout = io::stdout().lock();
    let wrote = stdout
        .write_all(reply.stdout.as_bytes())
        .and_then(|()| stdout.flush());
    let warned = io::stderr().lock().write_all(reply.stderr.as_bytes());
    if wrote.is_err() || warned.is_err() {
        return ExitCode::FAILURE;
    }
    ExitCode::from(reply.code)
}
