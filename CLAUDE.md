# Mnem

Memory for proactive agents, in Rust. The design is the spec at
https://claude.ai/artifact/5VHSbkX5WpDaiS3PHd2xJT (draft 3). Read the part
you are building before writing code; when code and spec disagree, raise it
instead of silently picking one.

## Checks

Run all of these before calling a change done. They must pass with no warnings:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --locked
```

Any change to the notebook format also runs the OptMem compatibility tests,
which need a checkout of OptMem and `python3`:

```bash
OPTMEM_MEMO=/path/to/OptMem/memo cargo test --test optmem_compat -- --ignored
```

## Disk space

This machine is short on disk. Build only into this repo's `target/`; never
set `CARGO_TARGET_DIR` to a second location, and tell subagents the same.
Delete scratch builds and temporary folders as soon as their work is done.
`cargo clean` reclaims `target/` when it grows.

## How to write Rust here

Follow the `rust-anti-slop` skill. Its lint policy is in `Cargo.toml` and
`clippy.toml`. In particular, production code has no `unwrap`, `expect` or
`panic`.

On top of that skill, these rules come from Google's Rust style guide:

- Name fallible constructors `try_new`, use the `as_`/`to_`/`into_` method
  prefixes as std does, and avoid abbreviations.
- Call free functions through their module (`log::append()`); import types
  and use them unqualified. One module per `use` line.
- Write the type on the result of `.parse()`, `.into()` and `.collect()`.
- Prefer a plain `for` loop to a long iterator chain; avoid `for_each`.
- Give every `assert!` a message that says why it should hold.
- Never call `std::panic::catch_unwind`.

## Mnem-specific rules

- **OptMem compatibility.** `LOG.txt` and `TREE/<size>` must stay
  byte-compatible, and `config` readable both ways, with the stock `memo` tool
  (github.com/VictorTaelin/OptMem). That repo has no license: re-implement its
  format from its documented behaviour and never copy its code. Test
  compatibility by running the Python `memo` and Mnem on the same inputs and
  comparing the files.
- **UniiChat** (github.com/VictorTaelin/uniichat) is Apache-2.0. Ideas or
  code taken from it get an attribution comment and a line in `NOTICE`.
- **No async.** Mnem is single-threaded. Lock files with `std::fs::File::lock`.
- **Time is injected.** Code that depends on the clock (heartbeat, `until`,
  hunch fade) takes the current time as an argument, so tests never sleep.
- **One output boundary.** Commands return their output as a `Reply`; only
  `emit` in `src/main.rs` writes to the terminal. It uses `io::Write`, not
  `print!`, so a closed pipe is a failed exit rather than a panic.
- **Model calls** go through the `Model` seam in `src/model.rs`: the harness
  names a program (`MNEM_SUMMARIZER`; prompt on stdin, reply on stdout, exit
  3 to decline), and Mnem has no model of its own or any provider's API. They happen only in the compactor and Dream, and code
  validates everything they return before it touches disk.
