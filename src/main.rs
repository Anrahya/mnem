//! The `mnem` binary: reads the environment and the clock, runs one command,
//! and writes its reply.

use std::env;
use std::ffi::OsString;
use std::io;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use jiff::Zoned;
use mnem::cli;
use mnem::cli::Context;
use mnem::cli::Reply;
use mnem::day::Day;

fn main() -> ExitCode {
    let mut args = env::args_os();
    let program = args.next().map(PathBuf::from);
    let rest: Vec<OsString> = args.collect();
    let reply = match context(program) {
        Ok(context) => cli::run(&context, &rest),
        Err(message) => Reply::failed(&message),
    };
    emit(&reply)
}

fn context(program: Option<PathBuf>) -> Result<Context, String> {
    let home = env::home_dir();
    let cwd = env::current_dir().ok();
    let me = cli::invoked_as(program.as_deref(), cwd.as_deref(), home.as_deref());
    let dir = cli::memory_dir(env::var_os("MEMORY_DIR"), home.as_deref(), cwd.as_deref())
        .ok_or("Cannot tell where the notebook is: set MEMORY_DIR, or HOME.")?;
    Ok(Context {
        dir,
        me,
        home,
        today: today()?,
    })
}

/// Today in the local time zone: the date `memo` would give a note written
/// now. Without a known time zone this falls back to UTC.
fn today() -> Result<Day, String> {
    let date = Zoned::now().date();
    Day::try_new(&date.to_string())
        .map_err(|error| format!("Today's date, {date}, cannot date a note: {error}."))
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
