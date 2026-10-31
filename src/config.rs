//! A notebook's size settings, read from its `config` file.
//!
//! The file holds `NAME = value` lines; `#` starts a comment, and a name not
//! set keeps its default, so a notebook without the file uses the defaults.
//! Lines may end in `\n`, `\r\n` or a lone `\r`. Values are written in
//! ASCII digits; other digit characters are rejected.

use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use crate::text;
use crate::text::TEXT_MAX;

/// The sizes a notebook may set for itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// How many lines `wake` prints.
    pub wake_lines: usize,
    /// The longest a note or summary may be, in bytes; at most [`TEXT_MAX`].
    pub entry_chars: usize,
    /// The largest part of paged output, in bytes.
    pub part_chars: usize,
    /// The largest part of paged output, in lines.
    pub part_lines: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            wake_lines: 96,
            entry_chars: TEXT_MAX,
            part_chars: 20_000,
            part_lines: 500,
        }
    }
}

/// Why a `config` file could not be used.
#[derive(Debug)]
pub enum ConfigError {
    /// The file exists but could not be read.
    Io { path: PathBuf, source: io::Error },
    /// A line names something that is not a size.
    UnknownName { line: usize, name: String },
    /// A value is not a whole number of at least 1.
    NotPositive {
        line: usize,
        name: String,
        value: String,
    },
    /// `ENTRY_CHARS` is larger than a record can hold.
    EntryTooLarge { line: usize, value: usize },
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
        Self::parse(&contents)
    }

    /// Parses the contents of a `config` file.
    pub fn parse(contents: &str) -> Result<Self, ConfigError> {
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
            let name = name.trim_matches(text::is_python_space).to_uppercase();
            let value = value.trim_matches(text::is_python_space);
            let slot = match name.as_str() {
                "WAKE_LINES" => &mut config.wake_lines,
                "ENTRY_CHARS" => &mut config.entry_chars,
                "PART_CHARS" => &mut config.part_chars,
                "PART_LINES" => &mut config.part_lines,
                _ => return Err(ConfigError::UnknownName { line, name }),
            };
            let size = positive(value).ok_or_else(|| ConfigError::NotPositive {
                line,
                name: name.clone(),
                value: value.to_owned(),
            })?;
            if name == "ENTRY_CHARS" && size > TEXT_MAX {
                return Err(ConfigError::EntryTooLarge { line, value: size });
            }
            *slot = size;
        }
        Ok(config)
    }
}

/// `value` as a whole number of at least 1, written in ASCII digits. A
/// number too large for `usize` means "no limit" and becomes `usize::MAX`.
fn positive(value: &str) -> Option<usize> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let size = value.parse::<usize>().unwrap_or(usize::MAX);
    (size >= 1).then_some(size)
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "cannot read {}: {source}", path.display()),
            Self::UnknownName { line, name } => {
                write!(f, "config line {line}: {name} is not a size setting")
            }
            Self::NotPositive { line, name, value } => write!(
                f,
                "config line {line}: {name} needs a whole number of at least 1, not '{value}'"
            ),
            Self::EntryTooLarge { line, value } => write!(
                f,
                "config line {line}: ENTRY_CHARS is {value}, but a record holds at most {TEXT_MAX}"
            ),
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::UnknownName { .. } | Self::NotPositive { .. } | Self::EntryTooLarge { .. } => {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Config;
    use super::ConfigError;

    #[test]
    fn empty_and_commented_files_keep_defaults() {
        let config = Config::parse("# a comment\n# WAKE_LINES = 300\n\nno equals sign\n")
            .expect("comments and blank lines are ignored");
        assert_eq!(config, Config::default(), "nothing was set");
    }

    #[test]
    fn reads_settings_with_spaces_case_and_trailing_comments() {
        let config = Config::parse("  wake_lines = 300   # more context\nENTRY_CHARS=200\n")
            .expect("valid settings");
        assert_eq!(config.wake_lines, 300, "lower-case name is accepted");
        assert_eq!(config.entry_chars, 200, "no spaces around = is accepted");
        assert_eq!(config.part_chars, 20_000, "unset sizes keep defaults");
    }

    #[test]
    fn rejects_unknown_names_and_bad_values() {
        assert!(
            matches!(
                Config::parse("COLOR = 3"),
                Err(ConfigError::UnknownName { line: 1, .. })
            ),
            "unknown name"
        );
        for bad in [
            "WAKE_LINES = 0",
            "WAKE_LINES = -5",
            "WAKE_LINES = ten",
            "WAKE_LINES =",
        ] {
            assert!(
                matches!(
                    Config::parse(bad),
                    Err(ConfigError::NotPositive { line: 1, .. })
                ),
                "{bad:?} is not a positive number"
            );
        }
        assert!(
            matches!(
                Config::parse("\nENTRY_CHARS = 281"),
                Err(ConfigError::EntryTooLarge {
                    line: 2,
                    value: 281
                })
            ),
            "ENTRY_CHARS above the record limit"
        );
        assert!(
            matches!(
                Config::parse("WAKE_LINES = \u{0663}"),
                Err(ConfigError::NotPositive { line: 1, .. })
            ),
            "non-ASCII digits are rejected"
        );
    }

    #[test]
    fn a_lone_carriage_return_ends_a_line() {
        assert!(
            matches!(
                Config::parse("WAKE_LINES = 7\rCOLOR = 3\r\n"),
                Err(ConfigError::UnknownName { line: 2, .. })
            ),
            "the second line is read on its own"
        );
        let config = Config::parse("WAKE_LINES = 7\r\nPART_LINES = 9\r").expect("valid settings");
        assert_eq!(config.wake_lines, 7, "first line");
        assert_eq!(config.part_lines, 9, "second line");
    }

    #[test]
    fn huge_values_saturate() {
        let config =
            Config::parse("PART_CHARS = 99999999999999999999999999").expect("a huge size is valid");
        assert_eq!(config.part_chars, usize::MAX, "saturated");
    }
}
