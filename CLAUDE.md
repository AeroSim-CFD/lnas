# LNAS - Claude Guide

## Project Overview
`lnas` is the Python API for `.lnas` (Lagrangian Nassu) files - the surface-geometry format
consumed by the AeroSim CFD stack (the `nassu` solver and the `aerosim` web app). It loads,
saves, transforms, and inspects triangular surface meshes. A companion Rust CLI, `stl2lnas`,
converts STL meshes into `.lnas` and lives in this same repo under `stl2lnas/`.

This is a **library**, not an application. Keep the public API small and stable; downstream
`nassu` and `aerosim` depend on it.

## Tech Stack
- **Language**: Python 3.10+ (library) and Rust (the `stl2lnas` CLI)
- **Core deps**: `numpy>=1.24`, `ruamel-yaml` (the geometry math is all NumPy)
- **Package manager**: `uv` (Python), `cargo` (Rust). Never use `pip` directly for dev.
- **Build backend**: `hatchling` (only the `lnas/` package is shipped to PyPI as `aerosim-lnas`)
- **Testing**: `pytest` (Python), `cargo test` (Rust)
- **Formatters/Linters**: Black, isort, Ruff (Python); `cargo fmt` (Rust). Line length 99.

## Package Layout
| Path | Purpose |
|---|---|
| `lnas/geometry.py` | `LnasGeometry`: vertices/triangles, normals, areas, transforms, STL export |
| `lnas/fmt.py` | `LnasFormat`: versioned `.lnas` container (geometry + named surfaces); `from_stl` |
| `lnas/transformations.py` | `Transformations` / `TransformationsMatrix`: translate/rotate/scale matrices |
| `lnas/stl.py` | Binary STL read/write helpers |
| `lnas/utils.py`, `lnas/exceptions.py` | YAML helpers; typed exceptions |
| `stl2lnas/` | Rust CLI that converts STL -> `.lnas`; the source of truth `from_stl` mirrors |
| `tests/` | pytest suite (files are `test*.py`) |
| `fixture/` | Test fixtures (e.g. `cylinder.lnas`) |

Public API is re-exported from `lnas/__init__.py`: `LnasGeometry`, `LnasFormat`,
`LagrangianReader`, `Transformations`, `TransformationsMatrix`. Imports in `__init__.py` are in
**dependency order** - keep that order if you add to it (transformations -> geometry -> fmt).

## Development Commands
Run from the repo root (where `pyproject.toml` and `tests/` live).

```bash
# Python tests
uv run pytest                       # full suite
uv run pytest tests/testLnasGeometry.py -q   # single file
uv run pytest -k normals -q         # by keyword

# Formatting & linting (must pass CI - see linting.yml)
uv run isort --check-only .
uv run black --config ./pyproject.toml --check .
uv run ruff check .
# Apply fixes:
uv run isort . && uv run black . && uv run ruff check --fix .

# Rust CLI (cwd must be stl2lnas/)
cd stl2lnas && cargo test
cd stl2lnas && cargo fmt --check     # CI checks this; run `cargo fmt` to fix
cargo install --path stl2lnas        # build + install the `stl2lnas` binary on PATH
```

**stl2lnas dependency for tests:** Some tests (the `from_stl` vs Rust parity suite in
`tests/testStl2Lnas.py`) shell out to the installed `stl2lnas` binary. If those error with
"command not found", run `cargo install --path stl2lnas` first. The pure-Python geometry/format
tests do not need it.

## Geometry Numerics
- Vertices and triangles are stored as NumPy arrays: `vertices` is `(Np, 3)` `float32`,
  `triangles` is `(Nt, 3)` `uint32`. The `.lnas` on-disk format is base64-encoded
  little-endian buffers of exactly these dtypes (`from_dct` / `to_dct`) - do not change the
  serialized dtypes without a format version bump.
- Derived quantities (`normals`, `areas`, `vertices_normals`, `triangle_vertices`) are cached on
  the instance and lazily computed via properties. Any mutation of `vertices`/`triangles` must go
  through `_full_update()` (or one of the `apply_transformation*` / `join` methods that call it),
  which recomputes them in the right order. Order matters: triangles_vertices -> normals -> areas
  -> vertices_normals.
- **Prefer vectorized NumPy over per-element Python loops.** This code is on the hot path of every
  transform and every nassu body refinement; a per-triangle Python loop is `O(n)` interpreted and
  has caused worker timeouts on large meshes (see the `_update_vertices_normals` vectorization,
  issue #18). When touching geometry math, reach for `np.add.at`, boolean masks, and broadcasting.
- Float precision: `vertices` are `float32`, so an ill-conditioned transform (large translate then
  small scale) can drive triangle cross-products toward zero and produce NaN normals on a
  geometrically valid mesh (issue #9). When that matters, do the geometric math (cross product,
  normalization) in `float64` and only cast the stored result back.

## Code Style
- **Line length**: 99 (Black + Ruff + isort all configured to 99 in `pyproject.toml`)
- **Python**: 3.10+; `from __future__ import annotations` at the top of modules; use `X | Y`
  unions, not `Optional[X]`. Add type hints (no mypy gate, but they document intent).
- **isort**: `profile = "black"`; `__init__.py` files are skipped to avoid circular-import
  reordering - don't fight that.
- Match the surrounding code's style and naming when editing a file.

## Workflow & Branch Rules
- **Never commit directly to `main`.** Always branch off `main` first.
  Branch naming: `type/kebab-case-description` (`feature/`, `fix/`, `refactor/`, `chore/`).
- **Open a PR for every change** and let the human merge it - don't merge PRs yourself or push to
  `main`. PRs target `main` (this repo has no `staging` branch).
- **Link the issue:** when a PR resolves a GitHub issue, put a closing keyword (`closes #N`,
  `fixes #N`) in the PR body so it auto-closes on merge.
- **Ship tests with code changes.** New behavior or a regression fix should come with a pytest
  case in the same PR. For numeric changes that preserve behavior, add a parity test comparing
  against the previous implementation (see `test_vertices_normals_parity_with_loop`).
- Issues and PRs are tracked in `AeroSim-CFD/lnas`; use the `gh` CLI to read/create/comment.

## Changelog (`CHANGELOG.md`)
- New entries go under a `## Unreleased` heading at the top (create it above the latest released
  version if absent). **Do not bump the version in `pyproject.toml` per change** - the version
  bump + renaming `## Unreleased` to `## X.Y.Z` is a separate release step.
- Entries are `*` bullets, past tense, one line each, focused on the user-visible what/why.
  Reference the issue or PR number inline, e.g. `... on large meshes (#18)`.
- Bug fixes can go under a `Bugs:` sub-section within the version block (see older entries).

## Versioning & Format
- Package version lives in `pyproject.toml` (`version = "X.Y.Z"`). Released as `aerosim-lnas`
  on PyPI via `.github/workflows/publish_pypi.yml`.
- The `.lnas` **file** format has its own version string in `lnas/fmt.py`
  (`_CURRENT_VERSION`, `_SUPPORTED_MAJOR_VERSIONS`) - independent of the package version. Bump it
  only when the on-disk layout changes, and update the supported-versions tuple accordingly.

## CI
Pull requests must pass (all triggered on `pull_request` and pushes to `main`):
- `testing.yml` - pytest across Python 3.10 / 3.11 / 3.12, with `stl2lnas` built via cargo
- `linting.yml` - `isort --check-only`, `black --check`, `ruff check`
- `rust_testing.yml` - `cargo test` in `stl2lnas/`
- `check_rust_format.yml` - `cargo fmt --check` in `stl2lnas/`

Run the matching local command before pushing so CI is green on the first try.

## Output Formatting Rules
- Never use a Unicode arrow; write `->` (hyphen + greater-than).
- Never use a Unicode ellipsis; write `...` (three dots).
- Never use an em dash; write `-` or `--`.
- No superscript/subscript Unicode (`²`, `₀`); write `x^2`, `CO2`.
