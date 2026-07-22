"""LNAS CI pipeline, defined as code with Dagger.

The exact same pipeline runs on a laptop and on any remote runner - there is no
platform-specific YAML. Everything executes inside containers, so a green run
locally means a green run everywhere.

Quick start (see ``.dagger/README.md`` for details)::

    dagger call --source=. check          # every check, concurrently, with a summary
    dagger call --source=. lint           # a single check (also: rust-test, rust-fmt)
    dagger call --source=. test --python=3.11   # pytest on one interpreter

The checks mirror the retired GitHub Actions *check* workflows:

- ``lint``      -> ``linting.yml``        (isort + black + ruff)
- ``test``      -> ``testing.yml``        (pytest, with the ``stl2lnas`` binary on PATH)
- ``rust-test`` -> ``rust_testing.yml``   (``cargo test`` in ``stl2lnas/``)
- ``rust-fmt``  -> ``check_rust_format.yml`` (``cargo fmt --check`` in ``stl2lnas/``)

Release/publish automation (``rust_build``, ``release_artifacts``, ``publish_pypi``)
stays in GitHub Actions - it is driven by GitHub ``release`` events and secrets.
"""

import asyncio
from typing import Annotated

import dagger
from dagger import Doc, dag, function, object_type

# uv provisions its own interpreters, so the base image version is not the test
# matrix; ``test`` selects the interpreter via ``uv --python``.
PYTHON_IMAGE = "python:3.12-slim-bookworm"
RUST_IMAGE = "rust:1-slim-bookworm"

# The Python versions the suite is tested against (mirrors testing.yml).
PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]

# System packages needed to compile the stl2lnas Rust binary and fetch sources.
BUILD_APT = ["build-essential", "git", "curl", "ca-certificates", "pkg-config"]


@object_type
class Lnas:
    """CI pipeline for the lnas library and the stl2lnas Rust CLI."""

    source: Annotated[dagger.Directory, Doc("The lnas repository root")]

    def _uv_python(self) -> dagger.Container:
        """Python container with uv installed and the working tree mounted.

        Layers are content-addressed, so every Python check reuses this build.
        """
        uv_cache = dag.cache_volume("lnas-uv-cache")
        return (
            dag.container()
            .from_(PYTHON_IMAGE)
            .with_exec(["pip", "install", "--no-cache-dir", "uv"])
            .with_env_variable("UV_LINK_MODE", "copy")
            .with_mounted_cache("/root/.cache/uv", uv_cache)
            .with_directory("/src", self.source)
            .with_workdir("/src")
        )

    def _rust(self) -> dagger.Container:
        """Rust container with the cargo registry cached, workdir at stl2lnas/."""
        registry = dag.cache_volume("lnas-cargo-registry")
        return (
            dag.container()
            .from_(RUST_IMAGE)
            .with_mounted_cache("/usr/local/cargo/registry", registry)
            .with_directory("/src", self.source)
            .with_workdir("/src/stl2lnas")
        )

    @function
    async def lint(self) -> str:
        """isort, black and ruff checks on the Python sources (line length 99)."""
        return await (
            self._uv_python()
            .with_exec(["uv", "sync"])
            .with_exec(["uv", "run", "isort", "--check-only", "."])
            .with_exec(["uv", "run", "black", "--config", "./pyproject.toml", "--check", "."])
            .with_exec(["uv", "run", "ruff", "check", "."])
            .stdout()
        )

    @function
    async def test(
        self,
        python: Annotated[str, Doc("Python version, e.g. 3.10")] = "3.12",
    ) -> str:
        """Run pytest on one interpreter, with the stl2lnas binary built and on PATH.

        The ``testStl2Lnas`` parity suite shells out to the ``stl2lnas`` binary, so
        the container needs a Rust toolchain and the compiled binary on PATH -
        exactly what ``cargo install --path stl2lnas`` provides.
        """
        registry = dag.cache_volume("lnas-cargo-registry")
        apt = "apt-get update && apt-get install -y --no-install-recommends " + " ".join(BUILD_APT)
        return await (
            self._uv_python()
            .with_exec(["sh", "-c", apt])
            .with_mounted_cache("/root/.cargo/registry", registry)
            .with_env_variable("PATH", "/root/.cargo/bin:$PATH", expand=True)
            .with_exec(["sh", "-c", "curl -LsSf https://sh.rustup.rs | sh -s -- -y --profile minimal"])
            .with_exec(["cargo", "install", "--path", "stl2lnas"])
            .with_exec(["uv", "sync", "--python", python])
            .with_exec(["uv", "run", "--python", python, "pytest"])
            .stdout()
        )

    @function
    async def rust_test(self) -> str:
        """``cargo test`` for the stl2lnas CLI."""
        return await self._rust().with_exec(["cargo", "test"]).stdout()

    @function
    async def rust_fmt(self) -> str:
        """``cargo fmt --check`` for the stl2lnas CLI."""
        return await (
            self._rust()
            .with_exec(["rustup", "component", "add", "rustfmt"])
            .with_exec(["cargo", "fmt", "--check"])
            .stdout()
        )

    @function
    async def check(self) -> str:
        """Run every check concurrently and report a pass/fail summary.

        Raises if any check fails, so this is the single command a runner calls.
        """
        checks: dict[str, "asyncio.Future[str]"] = {
            "lint": self.lint(),
            "rust-test": self.rust_test(),
            "rust-fmt": self.rust_fmt(),
        }
        for version in PYTHON_VERSIONS:
            checks[f"test-py{version}"] = self.test(version)

        names = list(checks)
        results = await asyncio.gather(*checks.values(), return_exceptions=True)

        lines: list[str] = []
        failed = False
        for name, result in zip(names, results):
            if isinstance(result, BaseException):
                failed = True
                lines.append(f"FAIL  {name}\n{result}")
            else:
                lines.append(f"ok    {name}")
        report = "\n".join(lines)
        if failed:
            raise RuntimeError("CI checks failed:\n" + report)
        return "All checks passed:\n" + report
