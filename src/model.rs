//! The seam through which Mnem asks a model for text.
//!
//! Mnem has no model of its own: the agent's harness answers. The harness
//! names a program; Mnem runs it with the prompt on standard input and reads
//! the reply from standard output. The program learns what the call is for
//! from `MNEM_JOB` (`summary` or `dream`) and `MNEM_EFFORT`, so it can pick
//! a cheap, fast setting for summaries and a stronger one for Dream. It does not get `MEMORY_DIR` or `MNEM_MODEL`, so a
//! program that is itself an agent cannot write to the notebook.
//!
//! The program's exit tells Mnem how the call went:
//!
//! - exit 0 with text: the reply;
//! - exit [`DECLINED_EXIT`]: the model declined, for example a refusal or a
//!   content filter;
//! - anything else, exit 0 with no text included, or no exit before the
//!   timeout: the model could not be reached, and the work is left for later.
//!
//! The prompt and the program's output pass through temporary files rather
//! than pipes, so Mnem needs no threads to serve them, and a process the
//! program leaves behind cannot hold the reply up. The program runs in its
//! own process group, and a timeout kills the whole group.

use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::io::Read;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::path::PathBuf;
use std::process;
use std::process::Child;
use std::process::Command;
use std::process::ExitStatus;
use std::process::Stdio;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use std::time::Instant;

/// The exit status a program uses to say the model declined.
pub const DECLINED_EXIT: i32 = 3;

/// The most of a reply Mnem reads; anything past it is dropped.
const REPLY_MAX: u64 = 64 * 1024;

/// The most of the program's error output kept for a message.
const ERRORS_MAX: u64 = 2 * 1024;

/// How often a running program is checked for its exit.
const POLL: Duration = Duration::from_millis(10);

/// How many times a program that is being written (`ETXTBSY`) is started
/// again before giving up.
const BUSY_TRIES: u32 = 5;

/// What a call is for. The harness may choose a model and effort by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Job {
    /// One summary line for a block of the notebook: small and mechanical.
    Summary,
    /// A new MIND.md from what the notebook learned: the hard one.
    Dream,
}

impl Job {
    /// The job's name, passed as `MNEM_JOB`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::Dream => "dream",
        }
    }

    /// How much thought the job needs, passed as `MNEM_EFFORT`.
    pub fn effort(self) -> &'static str {
        match self {
            Self::Summary => "low",
            Self::Dream => "high",
        }
    }
}

/// A model's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answer {
    /// The reply, exactly as written.
    Reply(String),
    /// The model declined.
    Declined,
}

/// Why the model could not be asked. The work stays undone, to be tried
/// again later.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unavailable(pub String);

/// Something that answers prompts.
pub trait Model {
    /// Asks for `job` with `prompt`.
    fn ask(&mut self, job: Job, prompt: &str) -> Result<Answer, Unavailable>;
}

/// A model reached by running a program; see the module documentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandModel {
    program: PathBuf,
    timeout: Duration,
}

impl CommandModel {
    /// Runs `program` for each call, giving up after `timeout`.
    pub fn new(program: PathBuf, timeout: Duration) -> Self {
        Self { program, timeout }
    }

    /// The program run for each call.
    pub fn program(&self) -> &Path {
        &self.program
    }

    fn start(&self, job: Job, exchange: &Exchange) -> io::Result<Child> {
        let mut tries = 0;
        loop {
            let mut command = Command::new(&self.program);
            command
                .env("MNEM_JOB", job.name())
                .env("MNEM_EFFORT", job.effort())
                .env_remove("MEMORY_DIR")
                .env_remove("MNEM_MODEL")
                .stdin(Stdio::from(File::open(&exchange.input)?))
                .stdout(Stdio::from(rewrite(&exchange.output)?))
                .stderr(Stdio::from(rewrite(&exchange.errors)?));
            #[cfg(unix)]
            command.process_group(0);
            match command.spawn() {
                // Another thread may hold the program open for writing for a
                // moment (a harness rewriting it, or a test making it).
                Err(error)
                    if error.kind() == io::ErrorKind::ExecutableFileBusy && tries < BUSY_TRIES =>
                {
                    tries += 1;
                    thread::sleep(POLL);
                }
                started => return started,
            }
        }
    }
}

impl Model for CommandModel {
    fn ask(&mut self, job: Job, prompt: &str) -> Result<Answer, Unavailable> {
        let shown = self.program.display().to_string();
        let exchange = Exchange::new(job, prompt).map_err(|error| {
            Unavailable(format!("cannot stage the prompt for {shown}: {error}"))
        })?;
        let mut child = self
            .start(job, &exchange)
            .map_err(|error| Unavailable(format!("cannot start {shown}: {error}")))?;
        let deadline = Instant::now().checked_add(self.timeout);
        let status = wait(&mut child, deadline)
            .map_err(|problem| Unavailable(format!("{shown} {problem}")))?;
        if status.code() == Some(DECLINED_EXIT) {
            return Ok(Answer::Declined);
        }
        if !status.success() {
            let said = read_capped(&exchange.errors, ERRORS_MAX).unwrap_or_default();
            let said = String::from_utf8_lossy(&said);
            let said = said.trim();
            let detail = if said.is_empty() {
                String::new()
            } else {
                format!(": {said}")
            };
            return Err(Unavailable(format!("{shown} failed ({status}){detail}")));
        }
        let reply = read_capped(&exchange.output, REPLY_MAX)
            .map_err(|error| Unavailable(format!("cannot read the reply of {shown}: {error}")))?;
        let reply = String::from_utf8_lossy(&reply).into_owned();
        if reply.trim().is_empty() {
            return Err(Unavailable(format!(
                "{shown} printed nothing; a model that declines exits with {DECLINED_EXIT}"
            )));
        }
        Ok(Answer::Reply(reply))
    }
}

/// The temporary files of one call, removed when it ends.
struct Exchange {
    input: PathBuf,
    output: PathBuf,
    errors: PathBuf,
}

impl Exchange {
    fn new(job: Job, prompt: &str) -> io::Result<Self> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let stem = format!("mnem-{}-{}-{n}", job.name(), process::id());
        let dir = env::temp_dir();
        let exchange = Self {
            input: dir.join(format!("{stem}.prompt")),
            output: dir.join(format!("{stem}.reply")),
            errors: dir.join(format!("{stem}.errors")),
        };
        // Each file is new, so a link planted at its name is refused, and
        // only this user can read the notebook material in it.
        private(&exchange.input)?.write_all(prompt.as_bytes())?;
        private(&exchange.output)?;
        private(&exchange.errors)?;
        Ok(exchange)
    }
}

/// Creates `path`, which must not exist, readable only by this user.
fn private(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    options.open(path)
}

/// Opens a file [`private`] made, emptied for another try.
fn rewrite(path: &Path) -> io::Result<File> {
    OpenOptions::new().write(true).truncate(true).open(path)
}

impl Drop for Exchange {
    fn drop(&mut self) {
        // Drop cannot report an error; a file that will not go is left in
        // the temporary directory, which the system clears.
        for path in [&self.input, &self.output, &self.errors] {
            drop(fs::remove_file(path));
        }
    }
}

/// Waits for the program to exit, killing its process group at `deadline`.
fn wait(child: &mut Child, deadline: Option<Instant>) -> Result<ExitStatus, String> {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {}
            Err(error) => return Err(format!("cannot be watched: {error}")),
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            kill_group(child);
            return Err("did not answer in time".to_owned());
        }
        thread::sleep(POLL);
    }
}

/// Kills the program and everything it started, then reaps it. Each step can
/// only fail if the processes are already gone, which is the goal anyway.
fn kill_group(child: &mut Child) {
    #[cfg(unix)]
    drop(
        Command::new("kill")
            .args(["-KILL", "--", &format!("-{}", child.id())])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status(),
    );
    drop(child.kill());
    drop(child.wait());
}

fn read_capped(path: &Path, limit: u64) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?.take(limit).read_to_end(&mut bytes)?;
    Ok(bytes)
}

impl fmt::Display for Unavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for Unavailable {}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::process::Command;
    use std::thread;
    use std::time::Duration;
    use std::time::Instant;

    use super::Answer;
    use super::CommandModel;
    use super::Job;
    use super::Model;
    use crate::testing::Scratch;

    /// A shell script in `scratch` with `body`, made executable.
    fn script(scratch: &Scratch, body: &str) -> PathBuf {
        fs::create_dir_all(&scratch.path).expect("scratch folder");
        let path = scratch.path.join("model.sh");
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write script");
        let mut permissions = fs::metadata(&path).expect("script").permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).expect("make executable");
        path
    }

    fn ask(program: PathBuf, timeout: Duration, prompt: &str) -> Result<Answer, String> {
        CommandModel::new(program, timeout)
            .ask(Job::Summary, prompt)
            .map_err(|unavailable| unavailable.0)
    }

    const PATIENT: Duration = Duration::from_secs(30);

    #[test]
    fn the_reply_is_what_the_program_prints() {
        let scratch = Scratch::new();
        let program = script(
            &scratch,
            r#"read -r first; echo "$MNEM_JOB $MNEM_EFFORT ${MEMORY_DIR:-no-dir} ${MNEM_MODEL:-none}: $first""#,
        );
        assert_eq!(
            ask(program, PATIENT, "hello\nmore"),
            Ok(Answer::Reply("summary low no-dir none: hello\n".to_owned())),
            "the prompt arrives on stdin, the job in the environment, and the notebook's variables do not"
        );
    }

    #[test]
    fn exit_three_declines_and_silence_is_a_failure() {
        let scratch = Scratch::new();
        let declines = script(&scratch, "cat > /dev/null; exit 3");
        assert_eq!(ask(declines, PATIENT, "x"), Ok(Answer::Declined), "exit 3");
        let silent = script(&scratch, "cat > /dev/null; echo '  '");
        let why = ask(silent, PATIENT, "x").expect_err("nothing printed");
        assert!(why.contains("printed nothing"), "{why}");
    }

    #[test]
    fn a_failed_exit_is_unavailable_and_says_why() {
        let scratch = Scratch::new();
        let program = script(&scratch, "echo 'no key set' >&2; exit 1");
        let why = ask(program, PATIENT, "x").expect_err("a failed exit");
        assert!(
            why.contains("no key set"),
            "the program's error is kept: {why}"
        );
    }

    #[test]
    fn a_missing_program_is_unavailable() {
        let scratch = Scratch::new();
        let why = ask(scratch.path.join("absent"), PATIENT, "x").expect_err("no program");
        assert!(why.contains("cannot start"), "{why}");
    }

    #[test]
    fn a_program_that_hangs_is_stopped_with_everything_it_started() {
        let scratch = Scratch::new();
        let program = script(&scratch, "sleep 30 &\necho $! > \"$0.child\"\nwait");
        let why = ask(program.clone(), Duration::from_secs(2), "x").expect_err("timed out");
        assert!(why.contains("did not answer in time"), "{why}");
        let child = fs::read_to_string(program.with_extension("sh.child")).expect("child pid");
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let alive = Command::new("kill")
                .args(["-0", child.trim()])
                .status()
                .expect("kill runs")
                .success();
            if !alive {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the program's own child was killed too"
            );
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn large_prompts_and_lingering_children_do_not_block() {
        let scratch = Scratch::new();
        let prompt = "p".repeat(1 << 20);
        let early = script(&scratch, "echo early; cat > /dev/null");
        assert_eq!(
            ask(early, PATIENT, &prompt),
            Ok(Answer::Reply("early\n".to_owned())),
            "a reply written before the input is read"
        );
        let lingering = script(&scratch, "echo answered; (sleep 2; echo late) &");
        let started = Instant::now();
        assert_eq!(
            ask(lingering, PATIENT, "x"),
            Ok(Answer::Reply("answered\n".to_owned())),
            "a process left behind does not hold the reply"
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "nor make it wait"
        );
    }
}
