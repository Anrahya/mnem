//! A notebook's size settings, read from its `config` file.
//!
//! The file holds `NAME = value` lines; `#` starts a comment, and a name not
//! set keeps its default, so a notebook without the file uses the defaults.
//! Lines may end in `\n`, `\r\n` or a lone `\r`. Values are written in
//! ASCII digits; other digit characters are rejected.

use std::error::Error;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process;

use crate::text;
use crate::text::TEXT_MAX;

/// A size a notebook may set for itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Knob {
    /// How many lines `wake` prints.
    WakeLines,
    /// The longest a note or summary may be, in bytes; at most [`TEXT_MAX`].
    EntryChars,
    /// The largest part of paged output, in bytes.
    PartChars,
    /// The largest part of paged output, in lines.
    PartLines,
}

impl Knob {
    /// Every knob, in the order the file and `config` list them.
    pub const ALL: [Self; 4] = [
        Self::WakeLines,
        Self::EntryChars,
        Self::PartChars,
        Self::PartLines,
    ];

    /// The knob's name in the file.
    pub fn name(self) -> &'static str {
        match self {
            Self::WakeLines => "WAKE_LINES",
            Self::EntryChars => "ENTRY_CHARS",
            Self::PartChars => "PART_CHARS",
            Self::PartLines => "PART_LINES",
        }
    }

    /// The knob named `name`, written in any case.
    pub fn from_name(name: &str) -> Option<Self> {
        let name = name.to_uppercase();
        let mut found = None;
        for knob in Self::ALL {
            if knob.name() == name {
                found = Some(knob);
            }
        }
        found
    }

    /// The value used when the file does not set one.
    pub fn default_value(self) -> usize {
        match self {
            Self::WakeLines => 96,
            Self::EntryChars => TEXT_MAX,
            Self::PartChars => 20_000,
            Self::PartLines => 500,
        }
    }

    /// What the knob controls, in a few words.
    pub fn meaning(self) -> &'static str {
        match self {
            Self::WakeLines => "how many lines wake shows",
            Self::EntryChars => "byte limit for one note or summary",
            Self::PartChars => "byte limit for one page of output",
            Self::PartLines => "line limit for one page of output",
        }
    }

    /// Validates `value` for this knob: a whole number of at least 1 in
    /// ASCII digits, and for `ENTRY_CHARS` at most [`TEXT_MAX`]. A number too
    /// large for `usize` means "no limit" and becomes `usize::MAX`.
    pub fn check(self, value: &str) -> Result<usize, ValueError> {
        let not_positive = || ValueError::NotPositive {
            knob: self,
            value: value.to_owned(),
        };
        if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
            return Err(not_positive());
        }
        let size = value.parse::<usize>().unwrap_or(usize::MAX);
        if size == 0 {
            return Err(not_positive());
        }
        if self == Self::EntryChars && size > TEXT_MAX {
            return Err(ValueError::EntryTooLarge { value: size });
        }
        Ok(size)
    }
}

/// The sizes a notebook sets for itself; a knob it does not set uses its
/// default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Config {
    wake_lines: Option<usize>,
    entry_chars: Option<usize>,
    part_chars: Option<usize>,
    part_lines: Option<usize>,
}

/// Why a value was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueError {
    /// Not a whole number of at least 1.
    NotPositive { knob: Knob, value: String },
    /// `ENTRY_CHARS` is larger than a record can hold.
    EntryTooLarge { value: usize },
}

/// A line of a `config` file that cannot be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineError {
    /// The line number, from 1.
    pub line: usize,
    /// What is wrong with it.
    pub problem: LineProblem,
}

/// What is wrong with a line of a `config` file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LineProblem {
    /// The line names something that is not a size.
    UnknownName(String),
    /// The line sets a size to a value it cannot take.
    BadValue(ValueError),
}

/// Why a `config` file could not be used.
#[derive(Debug)]
pub enum ConfigError {
    /// The file exists but could not be read or written.
    Io { path: PathBuf, source: io::Error },
    /// A line of the file is invalid. Every command stops on it, `config`
    /// included, so the message names the file to fix by hand.
    Invalid { path: PathBuf, error: LineError },
}

impl Config {
    /// Reads `<dir>/config`. A missing file means every size keeps its
    /// default.
    pub fn read(dir: &Path) -> Result<Self, ConfigError> {
        let path = dir.join("config");
        let contents = match fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(source) => return Err(ConfigError::Io { path, source }),
        };
        Self::parse(&contents).map_err(|error| ConfigError::Invalid { path, error })
    }

    /// Parses the contents of a `config` file.
    pub fn parse(contents: &str) -> Result<Self, LineError> {
        let mut config = Self::default();
        let contents = contents.replace("\r\n", "\n").replace('\r', "\n");
        for (line, raw) in (1_usize..).zip(contents.split('\n')) {
            let setting = raw
                .split('#')
                .next()
                .unwrap_or("")
                .trim_matches(text::is_python_space);
            let Some((name, value)) = setting.split_once('=') else {
                continue;
            };
            let name = name.trim_matches(text::is_python_space);
            let Some(knob) = Knob::from_name(name) else {
                return Err(LineError {
                    line,
                    problem: LineProblem::UnknownName(name.to_uppercase()),
                });
            };
            let size = knob
                .check(value.trim_matches(text::is_python_space))
                .map_err(|error| LineError {
                    line,
                    problem: LineProblem::BadValue(error),
                })?;
            config.set(knob, Some(size));
        }
        Ok(config)
    }

    /// The value in force for `knob`.
    pub fn get(&self, knob: Knob) -> usize {
        self.overridden(knob)
            .unwrap_or_else(|| knob.default_value())
    }

    /// The value the file sets for `knob`, if it sets one.
    pub fn overridden(&self, knob: Knob) -> Option<usize> {
        match knob {
            Knob::WakeLines => self.wake_lines,
            Knob::EntryChars => self.entry_chars,
            Knob::PartChars => self.part_chars,
            Knob::PartLines => self.part_lines,
        }
    }

    /// Sets `knob`, or with `None` returns it to its default.
    pub fn set(&mut self, knob: Knob, value: Option<usize>) {
        let slot = match knob {
            Knob::WakeLines => &mut self.wake_lines,
            Knob::EntryChars => &mut self.entry_chars,
            Knob::PartChars => &mut self.part_chars,
            Knob::PartLines => &mut self.part_lines,
        };
        *slot = value;
    }

    /// How many lines `wake` prints.
    pub fn wake_lines(&self) -> usize {
        self.get(Knob::WakeLines)
    }

    /// The longest a note or summary may be, in bytes.
    pub fn entry_chars(&self) -> usize {
        self.get(Knob::EntryChars)
    }

    /// The largest part of paged output, in bytes.
    pub fn part_chars(&self) -> usize {
        self.get(Knob::PartChars)
    }

    /// The largest part of paged output, in lines.
    pub fn part_lines(&self) -> usize {
        self.get(Knob::PartLines)
    }

    /// The file's contents: every knob on its own line, commented out when
    /// it keeps its default. `me` is the command that edits it.
    pub fn render(&self, me: &str) -> String {
        let mut out =
            String::from("# Sizes for this notebook. A line that starts with # keeps the\n");
        out.push_str(&format!(
            "# default. Change one with `{me} config NAME=VALUE`.\n\n"
        ));
        for knob in Knob::ALL {
            let (mark, value) = match self.overridden(knob) {
                Some(value) => ("", value),
                None => ("# ", knob.default_value()),
            };
            out.push_str(&format!(
                "{mark}{:<12} = {value:<6} # {}\n",
                knob.name(),
                knob.meaning()
            ));
        }
        out
    }

    /// Replaces `<dir>/config` with [`Config::render`]'s text. The new file is
    /// written beside the old one and renamed over it, so a reader sees the
    /// old file or the new one, never half of one.
    pub fn write(&self, dir: &Path, me: &str) -> Result<(), ConfigError> {
        let path = dir.join("config");
        let staging = dir.join(format!(".config.{}", process::id()));
        let io_error = |path: &Path, source| ConfigError::Io {
            path: path.to_path_buf(),
            source,
        };
        let result = File::create(&staging)
            .and_then(|mut file| {
                file.write_all(self.render(me).as_bytes())?;
                file.sync_all()
            })
            .map_err(|source| io_error(&staging, source));
        if let Err(error) = result {
            // The staging file is ours alone; if removing it fails too, the
            // write error is the one worth reporting.
            drop(fs::remove_file(&staging));
            return Err(error);
        }
        fs::rename(&staging, &path).map_err(|source| io_error(&path, source))?;
        File::open(dir)
            .and_then(|handle| handle.sync_all())
            .map_err(|source| io_error(dir, source))
    }
}

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositive { knob, value } => write!(
                f,
                "{} needs a whole number of at least 1, not '{value}'",
                knob.name()
            ),
            Self::EntryTooLarge { value } => write!(
                f,
                "ENTRY_CHARS is {value}, but a record holds at most {TEXT_MAX}"
            ),
        }
    }
}

impl Error for ValueError {}

impl fmt::Display for LineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let line = self.line;
        match &self.problem {
            LineProblem::UnknownName(name) => write!(
                f,
                "line {line}: {name} is not a size; delete the line or use one of WAKE_LINES, ENTRY_CHARS, PART_CHARS, PART_LINES"
            ),
            LineProblem::BadValue(error) => write!(f, "line {line}: {error}"),
        }
    }
}

impl Error for LineError {}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Invalid { path, error } => write!(f, "{} {error}", path.display()),
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Invalid { error, .. } => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Config;
    use super::Knob;
    use super::LineError;
    use super::LineProblem;
    use super::ValueError;

    #[test]
    fn empty_and_commented_files_keep_defaults() {
        let config = Config::parse("# a comment\n# WAKE_LINES = 300\n\nno equals sign\n")
            .expect("comments and blank lines are ignored");
        assert_eq!(config, Config::default(), "nothing was set");
        assert_eq!(config.wake_lines(), 96, "the default");
    }

    #[test]
    fn reads_settings_with_spaces_case_and_trailing_comments() {
        let config = Config::parse("  wake_lines = 300   # more context\nENTRY_CHARS=200\n")
            .expect("valid settings");
        assert_eq!(config.wake_lines(), 300, "lower-case name is accepted");
        assert_eq!(config.entry_chars(), 200, "no spaces around = is accepted");
        assert_eq!(config.part_chars(), 20_000, "unset sizes keep defaults");
        assert_eq!(config.overridden(Knob::PartChars), None, "and say so");
    }

    #[test]
    fn rejects_unknown_names_and_bad_values() {
        assert!(
            matches!(
                Config::parse("COLOR = 3"),
                Err(LineError {
                    line: 1,
                    problem: LineProblem::UnknownName(_)
                })
            ),
            "unknown name"
        );
        for bad in [
            "WAKE_LINES = 0",
            "WAKE_LINES = -5",
            "WAKE_LINES = ten",
            "WAKE_LINES =",
            "WAKE_LINES = \u{0663}",
        ] {
            assert!(
                matches!(
                    Config::parse(bad),
                    Err(LineError {
                        line: 1,
                        problem: LineProblem::BadValue(ValueError::NotPositive { .. })
                    })
                ),
                "{bad:?} is not a positive number in ASCII digits"
            );
        }
        assert!(
            matches!(
                Config::parse("\nENTRY_CHARS = 281"),
                Err(LineError {
                    line: 2,
                    problem: LineProblem::BadValue(ValueError::EntryTooLarge { value: 281 })
                })
            ),
            "ENTRY_CHARS above the record limit"
        );
    }

    #[test]
    fn a_lone_carriage_return_ends_a_line() {
        assert!(
            matches!(
                Config::parse("WAKE_LINES = 7\rCOLOR = 3\r\n"),
                Err(LineError {
                    line: 2,
                    problem: LineProblem::UnknownName(_)
                })
            ),
            "the second line is read on its own"
        );
        let config = Config::parse("WAKE_LINES = 7\r\nPART_LINES = 9\r").expect("valid settings");
        assert_eq!(config.wake_lines(), 7, "first line");
        assert_eq!(config.part_lines(), 9, "second line");
    }

    #[test]
    fn huge_values_saturate() {
        let config =
            Config::parse("PART_CHARS = 99999999999999999999999999").expect("a huge size is valid");
        assert_eq!(config.part_chars(), usize::MAX, "saturated");
    }

    #[test]
    fn rendered_files_read_back() {
        let mut config = Config::default();
        config.set(Knob::WakeLines, Some(120));
        let text = config.render("mnem");
        assert_eq!(
            Config::parse(&text).expect("the rendered file parses"),
            config,
            "only the overridden knob is active"
        );
        config.set(Knob::WakeLines, None);
        assert_eq!(
            Config::parse(&config.render("mnem")).expect("parses"),
            Config::default(),
            "unset again"
        );
    }
}
