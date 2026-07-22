# GitHub Actions workflows

The CI **checks** (linting, Python tests, Rust tests, Rust formatting) were
migrated off GitHub Actions to a [Dagger](https://dagger.io) pipeline defined as
code, which runs identically locally and on any remote runner. See
[`.dagger/README.md`](../../.dagger/README.md) for setup and usage
(`dagger call --source=. check`).

The retired check workflows (`linting`, `testing`, `rust_testing`,
`check_rust_format`) can be restored from git history if needed:

```bash
git log --all --oneline -- .github/workflows/
```

The **release/publish** workflows below stay in GitHub Actions - they are driven
by GitHub `release` events and repository secrets:

- `rust_build.yml` - build stl2lnas release binaries and upload them to the release
- `release_artifacts.yml` - build the Python sdist/wheel and attach them to the release
- `publish_pypi.yml` - publish `aerosim-lnas` to PyPI
