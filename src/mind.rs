//! MIND.md: Dream's short, cited picture of what holds, what is going on,
//! what keeps happening and what to do next. Dream is its only writer; the
//! agent reads it at session start and the heartbeat reads its nudges.
//!
//! The file has four sections, each a list of `- ` lines:
//!
//! ```text
//! ## Durable
//! - Lives in Delhi (#902)
//! ## Active
//! - Planning a trip to Goa in December (#940, #951)
//! ## Hunches
//! - [daily] Skips breakfast on workdays (#930, #944)
//! ## Nudges
//! - breakfast | 2026-11-01T08:00 | 10:00 | Suggest an omelette | Eggs bought, breakfast skipped twice (#944, #951) | notify
//! ```
//!
//! A nudge's fields are its id, when it is due (`YYYY-MM-DDTHH:MM`,
//! `next session`, or `event:<name>`), when it goes stale (`-`, `HH:MM` on
//! the same day, or `YYYY-MM-DDTHH:MM`), what to do, why (with citations),
//! and `notify` or `quiet`: whether to tell the user or act without asking.
//!
//! [`check`] applies the rules code enforces on every draft and again on
//! every expiry pass: citations must name real notes that are not all
//! superseded, hunches need support from two days and fade without new
//! support, fired and stale nudges go, and each section has a hard cap.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use jiff::civil::Date;
use jiff::civil::DateTime;

use crate::fired;
use crate::fired::Fired;
use crate::index::TagIndex;
use crate::record::Note;
use crate::tags;

/// MIND.md's file name in the notebook directory.
pub const FILE: &str = "MIND.md";

/// The most lines each section keeps; extra lines are cut, last first.
pub const DURABLE_MAX: usize = 40;
/// See [`DURABLE_MAX`].
pub const ACTIVE_MAX: usize = 20;
/// See [`DURABLE_MAX`].
pub const HUNCHES_MAX: usize = 10;
/// See [`DURABLE_MAX`].
pub const NUDGES_MAX: usize = 5;

/// A hunch fades after this many active periods (days or ISO weeks with
/// notes) pass without a note that supports it.
pub const FADE_AFTER: usize = 2;

const ID_MAX: usize = 32;

/// The longest line, in bytes, any section keeps: MIND.md is read at every
/// session start and fed back into every dream.
pub const LINE_MAX: usize = 400;

/// The contents of MIND.md.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Mind {
    /// Slow-changing facts and preferences, corrections included.
    pub durable: Vec<String>,
    /// Current work and recent events.
    pub active: Vec<String>,
    /// Patterns across days, not yet confirmed.
    pub hunches: Vec<Hunch>,
    /// Actions with a time or event and a reason.
    pub nudges: Vec<Nudge>,
}

/// A pattern across days.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hunch {
    /// How quickly it fades without support.
    pub cadence: Cadence,
    /// The pattern, with its citations.
    pub text: String,
}

/// How quickly a hunch fades.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cadence {
    /// After [`FADE_AFTER`] days with notes and no support.
    Daily,
    /// After [`FADE_AFTER`] ISO weeks with notes and no support.
    Weekly,
}

/// Something the agent should do, and when.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nudge {
    /// Kept by Dream when it carries the nudge over, so the fired ledger
    /// can recognize it.
    pub id: String,
    /// When it is due.
    pub when: When,
    /// After this it is stale and is dropped unfired.
    pub until: Option<DateTime>,
    /// What to do.
    pub action: String,
    /// Why, citing notes.
    pub why: String,
    /// Whether to tell the user, rather than act or prepare quietly.
    pub notify: bool,
}

/// When a nudge is due.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum When {
    /// At a local time.
    At(DateTime),
    /// At the start of the next session.
    NextSession,
    /// When the harness reports the named event.
    Event(String),
}

/// A line taken out of a draft, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dropped {
    /// The line as written.
    pub line: String,
    /// Why it was taken out.
    pub why: String,
}

/// Why a draft cannot be read at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MissingSection(pub &'static str);

/// The section headings, in order.
pub const SECTIONS: [&str; 4] = ["Durable", "Active", "Hunches", "Nudges"];

/// Reads a MIND.md draft. Lines that do not parse are returned as dropped;
/// a draft without all four section headings is rejected whole, since it
/// is not MIND.md at all.
pub fn parse(text: &str) -> Result<(Mind, Vec<Dropped>), MissingSection> {
    let mut mind = Mind::default();
    let mut dropped = Vec::new();
    let mut seen = [false; 4];
    let mut section: Option<usize> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(heading) = line.strip_prefix("## ") {
            section = SECTIONS
                .iter()
                .position(|name| name.eq_ignore_ascii_case(heading.trim()));
            if let Some(found) = section.and_then(|index| seen.get_mut(index)) {
                *found = true;
            }
            continue;
        }
        let Some(item) = line.strip_prefix("- ") else {
            // Titles and prose between items carry nothing.
            continue;
        };
        let item = item.trim();
        let parsed = match section {
            Some(0) => {
                mind.durable.push(item.to_owned());
                Ok(())
            }
            Some(1) => {
                mind.active.push(item.to_owned());
                Ok(())
            }
            Some(2) => parse_hunch(item).map(|hunch| mind.hunches.push(hunch)),
            Some(3) => parse_nudge(item).map(|nudge| mind.nudges.push(nudge)),
            _ => Err("it is outside the four sections".to_owned()),
        };
        if let Err(why) = parsed {
            dropped.push(Dropped {
                line: item.to_owned(),
                why,
            });
        }
    }
    for (name, found) in SECTIONS.iter().zip(seen) {
        if !found {
            return Err(MissingSection(name));
        }
    }
    Ok((mind, dropped))
}

fn parse_hunch(item: &str) -> Result<Hunch, String> {
    let (cadence, text) = if let Some(text) = item.strip_prefix("[daily]") {
        (Cadence::Daily, text)
    } else if let Some(text) = item.strip_prefix("[weekly]") {
        (Cadence::Weekly, text)
    } else {
        return Err("a hunch starts with [daily] or [weekly]".to_owned());
    };
    let text = text.trim();
    if text.is_empty() {
        return Err("the hunch is empty".to_owned());
    }
    Ok(Hunch {
        cadence,
        text: text.to_owned(),
    })
}

fn parse_nudge(item: &str) -> Result<Nudge, String> {
    let fields: Vec<&str> = item.split('|').map(str::trim).collect();
    let [id, when, until, action, why, notify] = fields.as_slice() else {
        return Err(format!(
            "a nudge has 6 fields separated by |, and this has {}",
            fields.len()
        ));
    };
    let id_ok = !id.is_empty()
        && id.len() <= ID_MAX
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    if !id_ok {
        return Err(format!(
            "the id {id:?} is not 1 to {ID_MAX} lowercase letters, digits or hyphens"
        ));
    }
    if is_reminder_id(id) {
        return Err(format!("the id {id} is kept for note reminders"));
    }
    let when = if *when == "next session" {
        When::NextSession
    } else if let Some(name) = when.strip_prefix("event:") {
        let name_ok = !name.is_empty()
            && name.len() <= ID_MAX
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'));
        if !name_ok {
            return Err(format!("{when:?} does not name an event"));
        }
        When::Event(name.to_owned())
    } else {
        When::At(tags::date_time(when).ok_or_else(|| {
            format!("{when:?} is not YYYY-MM-DDTHH:MM, \"next session\" or event:<name>")
        })?)
    };
    let until = match (*until, &when) {
        ("-", _) => None,
        (raw, When::At(at)) => {
            let until = tags::until_after(*at, raw)
                .ok_or_else(|| format!("{raw:?} is not a time after the nudge is due"))?;
            Some(until)
        }
        (raw, _) => {
            Some(tags::date_time(raw).ok_or_else(|| format!("{raw:?} is not YYYY-MM-DDTHH:MM"))?)
        }
    };
    if action.is_empty() || why.is_empty() {
        return Err("a nudge needs both what to do and why".to_owned());
    }
    let notify = match *notify {
        "notify" => true,
        "quiet" => false,
        other => return Err(format!("{other:?} is neither notify nor quiet")),
    };
    Ok(Nudge {
        id: (*id).to_owned(),
        when,
        until,
        action: (*action).to_owned(),
        why: (*why).to_owned(),
        notify,
    })
}

/// Whether `id` has the form `r<digits>`, which names a note's reminder.
pub fn is_reminder_id(id: &str) -> bool {
    id.strip_prefix('r')
        .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}

/// MIND.md's text for `mind`. [`parse`] reads it back unchanged.
pub fn render(mind: &Mind) -> String {
    let mut out = String::from(
        "# MIND\n\n\
         Written by Mnem's Dream from the notebook. Each line cites the notes it\n\
         rests on; the next dream replaces this file, so do not edit it.\n",
    );
    out.push_str("\n## Durable\n");
    for line in &mind.durable {
        out.push_str(&format!("- {line}\n"));
    }
    out.push_str("\n## Active\n");
    for line in &mind.active {
        out.push_str(&format!("- {line}\n"));
    }
    out.push_str("\n## Hunches\n");
    for hunch in &mind.hunches {
        out.push_str(&format!("- {}\n", hunch_line(hunch)));
    }
    out.push_str("\n## Nudges\n");
    for nudge in &mind.nudges {
        out.push_str(&format!("- {}\n", nudge_line(nudge)));
    }
    out
}

fn hunch_line(hunch: &Hunch) -> String {
    let cadence = match hunch.cadence {
        Cadence::Daily => "daily",
        Cadence::Weekly => "weekly",
    };
    format!("[{cadence}] {}", hunch.text)
}

/// A nudge as MIND.md writes it.
pub fn nudge_line(nudge: &Nudge) -> String {
    let until = nudge.until.map_or_else(|| "-".to_owned(), fired::minutes);
    let notify = if nudge.notify { "notify" } else { "quiet" };
    format!(
        "{} | {} | {until} | {} | {} | {notify}",
        nudge.id,
        when_text(&nudge.when),
        nudge.action,
        nudge.why
    )
}

/// How a nudge's due time is written.
pub fn when_text(when: &When) -> String {
    match when {
        When::At(at) => fired::minutes(*at),
        When::NextSession => "next session".to_owned(),
        When::Event(name) => format!("event:{name}"),
    }
}

/// The note ids `text` cites: every `#` followed by digits, except summary
/// ranges such as `#0-15`, which are not notes.
pub fn cites(text: &str) -> Vec<u64> {
    let mut ids = Vec::new();
    let mut rest = text;
    while let Some((_, after)) = rest.split_once('#') {
        let end = after
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after.len());
        let Some((digits, tail)) = after.split_at_checked(end) else {
            break;
        };
        rest = tail;
        let range = tail
            .strip_prefix('-')
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()));
        if digits.is_empty() || range {
            continue;
        }
        if let Ok(id) = digits.parse::<u64>() {
            ids.push(id);
        }
    }
    ids
}

/// Gives each nudge of `draft` that Dream renamed the id it had in `old`,
/// so the fired ledger and the harness's schedule still recognize it. A
/// nudge with a new id takes the id of an old nudge not carried over that is
/// due at the same time and cites a note in common with it, or that asks
/// for the same thing in the same words.
pub fn keep_ids(draft: &mut Mind, old: &Mind) {
    let mut taken: HashSet<String> = HashSet::new();
    for nudge in &draft.nudges {
        taken.insert(nudge.id.clone());
    }
    for nudge in &mut draft.nudges {
        if old.nudges.iter().any(|previous| previous.id == nudge.id) {
            continue;
        }
        let cited: HashSet<u64> = cites(&nudge.why).into_iter().collect();
        let action = plain(&nudge.action);
        for previous in &old.nudges {
            if taken.contains(&previous.id) {
                continue;
            }
            let same_time = previous.when == nudge.when
                && cites(&previous.why).iter().any(|id| cited.contains(id));
            if same_time || plain(&previous.action) == action {
                taken.insert(previous.id.clone());
                nudge.id.clone_from(&previous.id);
                break;
            }
        }
    }
}

/// `text` in lowercase with its words separated by single spaces.
fn plain(text: &str) -> String {
    let words: Vec<String> = text.split_whitespace().map(str::to_lowercase).collect();
    words.join(" ")
}

/// What [`check`] needs to know about the notebook.
#[derive(Clone, Debug, Default)]
pub struct Facts {
    /// The day each note was written, by id.
    days: Vec<Option<Date>>,
    index: TagIndex,
}

impl Facts {
    /// Adds the next note; notes must be added in order from #0.
    pub fn add(&mut self, note: &Note) {
        self.days.push(note.day.parse::<Date>().ok());
        self.index.add(note);
    }

    /// How many notes have been added.
    pub fn len(&self) -> u64 {
        u64::try_from(self.days.len()).unwrap_or(u64::MAX)
    }

    /// Whether no notes have been added.
    pub fn is_empty(&self) -> bool {
        self.days.is_empty()
    }

    /// The notebook's tag index.
    pub fn index(&self) -> &TagIndex {
        &self.index
    }

    fn day(&self, id: u64) -> Option<Date> {
        let index = usize::try_from(id).ok()?;
        self.days.get(index).copied().flatten()
    }

    /// Whether `cadence` periods with notes have passed since note `newest`
    /// without it being supported again.
    fn faded(&self, cadence: Cadence, newest: u64) -> bool {
        let Some(support) = self.day(newest) else {
            return false;
        };
        let period = |date: Date| match cadence {
            Cadence::Daily => (date.year(), date.day_of_year()),
            Cadence::Weekly => {
                let week = date.iso_week_date();
                (week.year(), i16::from(week.week()))
            }
        };
        let mut later = HashSet::new();
        let first = usize::try_from(newest.saturating_add(1)).unwrap_or(usize::MAX);
        for date in self.days.iter().skip(first).flatten() {
            if period(*date) > period(support) {
                later.insert(period(*date));
                if later.len() >= FADE_AFTER {
                    return true;
                }
            }
        }
        false
    }
}

/// Applies the rules code enforces to `mind`, as of `now`, and returns what
/// is kept and what was dropped.
pub fn check(mind: Mind, facts: &Facts, ledger: &[Fired], now: DateTime) -> (Mind, Vec<Dropped>) {
    let mut dropped = Vec::new();
    let mut kept = Mind::default();
    for line in mind.durable {
        match cited(&line, facts) {
            Ok(_) => kept.durable.push(line),
            Err(why) => dropped.push(Dropped { line, why }),
        }
    }
    for line in mind.active {
        match cited(&line, facts) {
            Ok(_) => kept.active.push(line),
            Err(why) => dropped.push(Dropped { line, why }),
        }
    }
    for hunch in mind.hunches {
        match check_hunch(&hunch, facts) {
            Ok(()) => kept.hunches.push(hunch),
            Err(why) => dropped.push(Dropped {
                line: hunch_line(&hunch),
                why,
            }),
        }
    }
    let mut ids = HashSet::new();
    for nudge in mind.nudges {
        let checked = check_nudge(&nudge, facts, ledger, now).and_then(|()| {
            if ids.insert(nudge.id.clone()) {
                Ok(())
            } else {
                Err(format!("another nudge already has the id {}", nudge.id))
            }
        });
        match checked {
            Ok(()) => kept.nudges.push(nudge),
            Err(why) => dropped.push(Dropped {
                line: nudge_line(&nudge),
                why,
            }),
        }
    }
    cap(&mut kept.durable, DURABLE_MAX, &mut dropped, String::clone);
    cap(&mut kept.active, ACTIVE_MAX, &mut dropped, String::clone);
    cap(&mut kept.hunches, HUNCHES_MAX, &mut dropped, hunch_line);
    cap(&mut kept.nudges, NUDGES_MAX, &mut dropped, nudge_line);
    (kept, dropped)
}

fn cap<T>(items: &mut Vec<T>, max: usize, dropped: &mut Vec<Dropped>, line: fn(&T) -> String) {
    for item in items.drain(max.min(items.len())..) {
        dropped.push(Dropped {
            line: line(&item),
            why: format!("the section keeps at most {max} lines"),
        });
    }
}

/// The ids `text` cites, if they are valid: at least one, all of them real
/// notes, and not all superseded. Also checks the line's length.
fn cited(text: &str, facts: &Facts) -> Result<Vec<u64>, String> {
    if text.len() > LINE_MAX {
        return Err(format!("it is longer than {LINE_MAX} bytes"));
    }
    let ids = cites(text);
    if ids.is_empty() {
        return Err("it cites no note".to_owned());
    }
    for id in &ids {
        if *id >= facts.len() {
            return Err(format!("it cites #{id}, which does not exist"));
        }
    }
    if ids
        .iter()
        .all(|id| facts.index.superseded_by(*id).is_some())
    {
        return Err("every note it cites is superseded".to_owned());
    }
    Ok(ids)
}

fn check_hunch(hunch: &Hunch, facts: &Facts) -> Result<(), String> {
    let ids = cited(&hunch.text, facts)?;
    let mut days = HashSet::new();
    for id in &ids {
        if let Some(day) = facts.day(*id) {
            days.insert(day);
        }
    }
    if days.len() < 2 {
        return Err("a hunch needs notes from two different days".to_owned());
    }
    let newest = ids.iter().copied().max().unwrap_or_default();
    if facts.faded(hunch.cadence, newest) {
        return Err("it faded: nothing new supports it".to_owned());
    }
    Ok(())
}

fn check_nudge(
    nudge: &Nudge,
    facts: &Facts,
    ledger: &[Fired],
    now: DateTime,
) -> Result<(), String> {
    if nudge_line(nudge).len() > LINE_MAX {
        return Err(format!("it is longer than {LINE_MAX} bytes"));
    }
    cited(&nudge.why, facts)?;
    if nudge.until.is_some_and(|until| until < now) {
        return Err("it is stale: its until time has passed".to_owned());
    }
    let since = match nudge.when {
        When::At(at) => Some(at),
        When::NextSession | When::Event(_) => None,
    };
    if fired::has_fired(ledger, &nudge.id, since) {
        return Err("it already fired".to_owned());
    }
    Ok(())
}

impl fmt::Display for MissingSection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the draft has no \"## {}\" section", self.0)
    }
}

impl Error for MissingSection {}

#[cfg(test)]
mod tests {
    use jiff::civil::DateTime;

    use super::Cadence;
    use super::Facts;
    use super::Hunch;
    use super::Mind;
    use super::MissingSection;
    use super::Nudge;
    use super::When;
    use crate::fired;
    use crate::fired::Fired;
    use crate::mind;
    use crate::record::Note;
    use crate::tags;

    fn at(value: &str) -> DateTime {
        tags::date_time(value).expect("a valid time")
    }

    fn facts(notes: &[(&str, &str)]) -> Facts {
        let mut facts = Facts::default();
        for (id, (day, text)) in (0_u64..).zip(notes) {
            facts.add(&Note {
                id,
                day: (*day).to_owned(),
                text: (*text).to_owned(),
            });
        }
        facts
    }

    fn nudge(id: &str, when: When, until: Option<DateTime>) -> Nudge {
        Nudge {
            id: id.to_owned(),
            when,
            until,
            action: "Suggest an omelette".to_owned(),
            why: "Eggs bought (#0)".to_owned(),
            notify: true,
        }
    }

    #[test]
    fn render_and_parse_agree() {
        let mind = Mind {
            durable: vec!["Lives in Delhi (#2)".to_owned()],
            active: vec!["Planning a trip (#1, #2)".to_owned()],
            hunches: vec![Hunch {
                cadence: Cadence::Weekly,
                text: "Calls home on Sundays (#0, #2)".to_owned(),
            }],
            nudges: vec![
                nudge(
                    "eggs",
                    When::At(at("2026-11-01T08:00")),
                    Some(at("2026-11-01T10:00")),
                ),
                nudge("ask", When::NextSession, None),
                nudge("ci", When::Event("build-failed".to_owned()), None),
            ],
        };
        let text = mind::render(&mind);
        assert!(
            text.contains("- eggs | 2026-11-01T08:00 | 2026-11-01T10:00 | Suggest an omelette | Eggs bought (#0) | notify\n"),
            "nudges are pipe fields: {text}"
        );
        assert_eq!(mind::parse(&text), Ok((mind, Vec::new())), "a round trip");
        assert_eq!(
            mind::parse(&mind::render(&Mind::default())),
            Ok((Mind::default(), Vec::new())),
            "empty sections are still sections"
        );
    }

    #[test]
    fn lines_that_do_not_parse_are_dropped_and_a_draft_without_sections_is_rejected() {
        assert_eq!(
            mind::parse("## Durable\n- x (#0)\n## Active\n## Hunches\n"),
            Err(MissingSection("Nudges")),
            "all four sections are required"
        );
        let draft = "- before any section (#0)\n## Durable\n## Active\n## Hunches\n\
                     - no cadence (#0, #1)\n\
                     ## Nudges\n\
                     - only | three | fields\n\
                     - Bad_Id | next session | - | do | why (#0) | notify\n\
                     - r12 | next session | - | do | why (#0) | notify\n\
                     - x | tomorrow | - | do | why (#0) | notify\n\
                     - x | 2026-11-01T08:00 | 07:00 | do | why (#0) | notify\n\
                     - x | next session | - | do | why (#0) | loudly\n\
                     - ok | 2026-11-01T08:00 | 09:30 | do | why (#0) | quiet\n";
        let (mind, dropped) = mind::parse(draft).expect("all sections present");
        assert_eq!(mind.nudges.len(), 1, "one good nudge");
        assert_eq!(
            mind.nudges.first().and_then(|nudge| nudge.until),
            Some(at("2026-11-01T09:30")),
            "HH:MM is on the day the nudge is due"
        );
        let reasons: Vec<&str> = dropped.iter().map(|item| item.why.as_str()).collect();
        assert_eq!(dropped.len(), 8, "every bad line is reported: {reasons:?}");
        for (reason, expected) in reasons.iter().zip([
            "outside the four sections",
            "[daily] or [weekly]",
            "6 fields",
            "lowercase letters",
            "note reminders",
            "is not YYYY-MM-DDTHH:MM",
            "after the nudge is due",
            "neither notify nor quiet",
        ]) {
            assert!(reason.contains(expected), "{reason:?} says {expected:?}");
        }
    }

    #[test]
    fn citations_name_notes_not_summaries() {
        assert_eq!(
            mind::cites("Moved (#902, replaces #455), see #0-15, #7, #5-#6 and #12-ish."),
            [902, 455, 7, 5, 6, 12],
            "summary ranges are not citations"
        );
        assert!(mind::cites("C# and # alone").is_empty(), "no digits, no id");
    }

    #[test]
    fn lines_must_cite_real_notes_that_still_stand() {
        let facts = facts(&[
            ("2026-10-01", "Lives in Mumbai"),
            ("2026-10-02", "[updates #0] Lives in Delhi"),
        ]);
        let mind = Mind {
            durable: vec![
                "Lives in Delhi (#1, replaces #0)".to_owned(),
                "Lives in Mumbai (#0)".to_owned(),
                "Has a cat".to_owned(),
                "Has a dog (#5)".to_owned(),
            ],
            ..Mind::default()
        };
        let (kept, dropped) = mind::check(mind, &facts, &[], at("2026-10-03T09:00"));
        assert_eq!(
            kept.durable,
            ["Lives in Delhi (#1, replaces #0)"],
            "citing the old note beside the new one is fine"
        );
        let reasons: Vec<&str> = dropped.iter().map(|item| item.why.as_str()).collect();
        assert_eq!(
            reasons,
            [
                "every note it cites is superseded",
                "it cites no note",
                "it cites #5, which does not exist"
            ],
            "each bad line says why"
        );
    }

    #[test]
    fn hunches_need_two_days_and_fade_without_support() {
        let facts = facts(&[
            ("2026-10-05", "Skipped breakfast"),
            ("2026-10-05", "Skipped breakfast again"),
            ("2026-10-06", "Skipped breakfast, late meeting"),
            ("2026-10-07", "Gym"),
            ("2026-10-08", "Gym"),
            ("2026-10-19", "Groceries"),
        ]);
        let hunch = |cadence, text: &str| Hunch {
            cadence,
            text: text.to_owned(),
        };
        let mind = Mind {
            hunches: vec![
                hunch(Cadence::Daily, "Skips breakfast (#0, #1)"),
                hunch(Cadence::Daily, "Skips breakfast (#0, #2)"),
                hunch(Cadence::Daily, "Goes to the gym (#3, #4)"),
                hunch(Cadence::Weekly, "Skips breakfast (#0, #2)"),
                hunch(Cadence::Weekly, "Goes to the gym (#3, #4)"),
            ],
            ..Mind::default()
        };
        let (kept, dropped) = mind::check(mind, &facts, &[], at("2026-10-20T09:00"));
        assert_eq!(
            kept.hunches,
            [
                hunch(Cadence::Daily, "Goes to the gym (#3, #4)"),
                hunch(Cadence::Weekly, "Skips breakfast (#0, #2)"),
                hunch(Cadence::Weekly, "Goes to the gym (#3, #4)"),
            ],
            "only one week with notes has passed since, so weekly hunches stay"
        );
        let reasons: Vec<&str> = dropped.iter().map(|item| item.why.as_str()).collect();
        assert_eq!(
            reasons,
            [
                "a hunch needs notes from two different days",
                "it faded: nothing new supports it"
            ],
            "the daily breakfast hunch saw 3 later days with notes"
        );
    }

    #[test]
    fn nudges_go_when_stale_fired_or_repeated() {
        let facts = facts(&[("2026-10-30", "Bought eggs")]);
        let now = at("2026-10-31T09:00");
        let ledger = [
            Fired {
                id: "fired".to_owned(),
                at: at("2026-10-31T08:00"),
                due: None,
            },
            Fired {
                id: "moved".to_owned(),
                at: at("2026-10-30T08:00"),
                due: None,
            },
            Fired {
                id: "asked".to_owned(),
                at: at("2026-10-30T08:00"),
                due: None,
            },
        ];
        let mind = Mind {
            nudges: vec![
                nudge(
                    "stale",
                    When::At(at("2026-10-31T07:00")),
                    Some(at("2026-10-31T08:00")),
                ),
                nudge("late", When::At(at("2026-10-31T07:00")), None),
                nudge("fired", When::At(at("2026-10-31T08:00")), None),
                nudge("moved", When::At(at("2026-11-01T08:00")), None),
                nudge("asked", When::NextSession, None),
                nudge("late", When::NextSession, None),
            ],
            ..Mind::default()
        };
        let (kept, dropped) = mind::check(mind, &facts, &ledger, now);
        let ids: Vec<&str> = kept.nudges.iter().map(|nudge| nudge.id.as_str()).collect();
        assert_eq!(
            ids,
            ["late", "moved"],
            "an overdue unfired nudge stays for the heartbeat, and a fired one moved later is new"
        );
        let reasons: Vec<&str> = dropped.iter().map(|item| item.why.as_str()).collect();
        assert_eq!(
            reasons,
            [
                "it is stale: its until time has passed",
                "it already fired",
                "it already fired",
                "another nudge already has the id late"
            ],
            "each dropped nudge says why"
        );
        assert_eq!(
            fired::line(&Fired {
                id: "x".to_owned(),
                at: now,
                due: None
            }),
            "x 2026-10-31T09:00",
            "the ledger uses the same minutes"
        );
    }

    #[test]
    fn long_lines_are_dropped() {
        let facts = facts(&[("2026-10-30", "A note")]);
        let long = format!("{} (#0)", "word ".repeat(mind::LINE_MAX / 5));
        let mind = Mind {
            durable: vec![long],
            nudges: vec![Nudge {
                action: "x".repeat(mind::LINE_MAX),
                ..nudge("long", When::NextSession, None)
            }],
            ..Mind::default()
        };
        let (kept, dropped) = mind::check(mind, &facts, &[], at("2026-10-31T09:00"));
        assert_eq!(kept, Mind::default(), "neither line fits");
        assert_eq!(dropped.len(), 2, "both are reported");
    }

    #[test]
    fn sections_are_capped_from_the_end() {
        let facts = facts(&[("2026-10-30", "A note")]);
        let mut lines = Vec::new();
        for n in 0..mind::ACTIVE_MAX + 2 {
            lines.push(format!("Item {n} (#0)"));
        }
        let mind = Mind {
            active: lines,
            ..Mind::default()
        };
        let (kept, dropped) = mind::check(mind, &facts, &[], at("2026-10-31T09:00"));
        assert_eq!(kept.active.len(), mind::ACTIVE_MAX, "the cap holds");
        assert_eq!(
            dropped.first().map(|item| item.line.as_str()),
            Some(format!("Item {} (#0)", mind::ACTIVE_MAX).as_str()),
            "the last lines go"
        );
    }

    #[test]
    fn renamed_nudges_keep_their_ids() {
        let nudge = |id: &str, when: When, action: &str, why: &str| Nudge {
            id: id.to_owned(),
            when,
            until: None,
            action: action.to_owned(),
            why: why.to_owned(),
            notify: true,
        };
        let morning = When::At(at("2026-11-01T08:00"));
        let old = Mind {
            nudges: vec![
                nudge(
                    "eggs",
                    morning.clone(),
                    "Suggest an omelette",
                    "Bought eggs (#4)",
                ),
                nudge(
                    "call",
                    When::NextSession,
                    "Call the bank",
                    "Card blocked (#7)",
                ),
                nudge(
                    "kept",
                    When::NextSession,
                    "Water the plants",
                    "Dry soil (#9)",
                ),
            ],
            ..Mind::default()
        };
        let mut draft = Mind {
            nudges: vec![
                nudge(
                    "omelette",
                    morning,
                    "Offer to cook eggs",
                    "Eggs in the fridge (#4, #5)",
                ),
                nudge(
                    "bank",
                    When::Event("home".to_owned()),
                    "call  the BANK",
                    "Still blocked (#8)",
                ),
                nudge(
                    "kept",
                    When::NextSession,
                    "Water the plants",
                    "Dry soil (#9)",
                ),
                nudge(
                    "new",
                    When::NextSession,
                    "Water the plants",
                    "Dry again (#10)",
                ),
            ],
            ..Mind::default()
        };
        mind::keep_ids(&mut draft, &old);
        let ids: Vec<&str> = draft.nudges.iter().map(|item| item.id.as_str()).collect();
        assert_eq!(
            ids,
            ["eggs", "call", "kept", "new"],
            "same time and a shared cite, or the same words, keep the old id; an id carried over is not taken twice"
        );
    }
}
