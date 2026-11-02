//! Mnem: memory for proactive agents.
//!
//! An OptMem-compatible notebook, a background compactor, a Dream pass that
//! writes MIND.md, a heartbeat that wakes the agent when a nudge is due, and
//! a note pass that catches what the agent forgot to note.
//! The design is in the spec linked from README.md.

pub mod cli;
pub mod compactor;
pub mod config;
pub mod cover;
pub mod day;
pub mod dream;
mod files;
pub mod fired;
pub mod heartbeat;
pub mod index;
pub mod memory;
pub mod mind;
pub mod model;
pub mod notebook;
pub mod pass;
pub mod record;
pub mod simhash;
pub mod span;
pub mod tags;
pub mod text;

#[cfg(test)]
mod testing;
