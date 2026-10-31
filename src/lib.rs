//! Mnem: memory for proactive agents.
//!
//! An OptMem-compatible notebook, a background compactor, a Dream pass that
//! writes MIND.md, and a heartbeat that wakes the agent when a nudge is due.
//! The design is in the spec linked from README.md.

pub mod config;
pub mod cover;
pub mod day;
pub mod notebook;
pub mod record;
pub mod span;
pub mod text;
