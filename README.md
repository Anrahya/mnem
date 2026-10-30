# Mnem

Memory for proactive agents. Mnem keeps an agent's past in an append-only
notebook (compatible with [OptMem](https://github.com/VictorTaelin/OptMem)),
lets a nightly "Dream" pass turn it into a short, cited picture of what is
going on and what to do next (`MIND.md`), and wakes the agent only when one of
Dream's nudges is due.

Status: early development; nothing usable yet.

Design: [Mnem: A Memory System for Proactive Agents](https://claude.ai/artifact/5VHSbkX5WpDaiS3PHd2xJT)
(draft 3).

## Build

Requires Rust 1.98 or newer.

```bash
cargo build
cargo test
```
