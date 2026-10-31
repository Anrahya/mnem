//! Mnem: memory for proactive agents.
//!
//! An OptMem-compatible notebook, a background compactor, a Dream pass that
//! writes MIND.md, and a heartbeat that wakes the agent when a nudge is due.
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
pub mod mind;
pub mod model;
pub mod notebook;
pub mod record;
pub mod simhash;
pub mod span;
pub mod tags;
pub mod text;

#[cfg(test)]
mod testing;
