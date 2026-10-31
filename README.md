# Mnem

Memory for proactive agents. Mnem keeps an agent's past in an append-only
notebook (compatible with [OptMem](https://github.com/VictorTaelin/OptMem)),
lets a nightly "Dream" pass turn it into a short, cited picture of what is
going on and what to do next (`MIND.md`), and wakes the agent only when one of
Dream's nudges is due.

Status: early development. The notebook, Dream, MIND.md, the heartbeat's
context and the session entry points work and are tested; no harness is wired
to them yet.

Design: [Mnem: A Memory System for Proactive Agents](https://claude.ai/artifact/5VHSbkX5WpDaiS3PHd2xJT)
(draft 4).

## Wiring it to a harness

Mnem has no model and runs no daemon. The agent's harness supplies both:

| When | The harness runs |
|---|---|
| A session starts | `mnem session start`, and gives its output to the agent |
| The context is about to be compacted | `mnem session compact` (dreams in the background) |
| The agent has been idle 45–60 minutes | `mnem dream idle` |
| Once a night | `mnem dream nightly` |
| Its timer | `mnem next` says when to wake the agent; then `mnem brief heartbeat` |
| An event it saw | `mnem brief event <name>` |

Model calls go to the program named in `MNEM_MODEL`: the prompt arrives on
standard input, `MNEM_JOB` is `summary` or `dream`, the reply goes to standard
output, and exit status 3 means the model declined. `mnem init` prints the
instructions to add to the agent's system prompt.

## Build

Requires Rust 1.98 or newer.

```bash
cargo build
cargo test
```

## Licence

MIT; see `LICENSE`. The compactor's summary prompt is adapted from
UniiChat and stays under Apache-2.0 (`LICENSE-APACHE`); `NOTICE` has the
details.
