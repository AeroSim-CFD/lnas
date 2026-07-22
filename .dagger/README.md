# LNAS CI (Dagger)

lnas's CI *checks* are defined as code with [Dagger](https://dagger.io) (Apache-2.0).
The same pipeline runs on your laptop and on any remote runner - there is no
platform-specific YAML, and every step runs inside a container, so a green run
locally is a green run everywhere. This mirrors the Dagger migration already done
for `nassu` and `cfdmod`.

Release/publish automation (`rust_build`, `release_artifacts`, `publish_pypi`)
stays in GitHub Actions - it is driven by GitHub `release` events and secrets, so
it is out of scope for this check pipeline.

## One-time setup

Install the Dagger CLI (Docker must be running):

```bash
curl -fsSL https://dl.dagger.io/dagger/install.sh | sh
# then generate the SDK bindings for this module:
dagger develop
```

`dagger develop` populates `.dagger/sdk/` (git-ignored) and reconciles the
`engineVersion` in `dagger.json` with your installed CLI.

## Running checks

From the repository root, `--source=.` mounts the working tree:

```bash
dagger call --source=. check              # every check, concurrently, with a summary
dagger call --source=. lint               # a single check (also: rust-test, rust-fmt)
dagger call --source=. test --python=3.11 # pytest on one interpreter
```

`check` runs every check below concurrently and fails if any one fails:

| Function    | Mirrors                  | What it does                                              |
| ----------- | ------------------------ | --------------------------------------------------------- |
| `lint`      | `linting.yml`            | `isort --check-only`, `black --check`, `ruff check`       |
| `test`      | `testing.yml`            | `pytest` on py3.10/3.11/3.12, with `stl2lnas` built on PATH |
| `rust-test` | `rust_testing.yml`       | `cargo test` in `stl2lnas/`                               |
| `rust-fmt`  | `check_rust_format.yml`  | `cargo fmt --check` in `stl2lnas/`                        |

The `test` check builds the `stl2lnas` Rust binary with `cargo install` and puts
it on `PATH`, because the `testStl2Lnas` parity suite shells out to that binary.

## How the CI server runs this

This repo owns only its CI *definition* (the checks above). Scheduling, checkout,
and running many repos' pipelines are the CI server's job (shared across repos),
not this repo's. From the server's point of view the entrypoint is one command
run against a checkout:

```bash
dagger call --source=. check
```
