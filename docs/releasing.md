# How to release pg-phenotype

A tag `vX.Y.Z` on `main` runs `.github/workflows/publish.yml`.  It builds
five abi3 wheels (Linux x86_64 and aarch64, macOS x86_64 and arm64, Windows
x86_64) and the sdist, and runs the whole test suite on each wheel on its own
platform and on an install from the sdist.  It builds the R source tarball
and checks it offline with `R CMD check --as-cran`.  If everything passes,
it publishes the wheels and sdist to PyPI and creates a GitHub release with
the R tarball attached.

## Set up once

1. On PyPI, add a pending trusted publisher (Account settings, Publishing)
   with project name `pg-phenotype`, owner `rwaples`, repository
   `pg-phenotype`, workflow `publish.yml` and environment `pypi`.
2. The GitHub environment `pypi` exists in the repository settings.  To
   require an approval before each upload, add yourself as a required
   reviewer there.

## Release a version

1. Set the version in three places: `[workspace.package]` in `Cargo.toml`,
   `Version:` in `r/DESCRIPTION`, and `version` in `r/src/rust/Cargo.toml`.
   The workflow refuses to run when they disagree.
2. Rename the `CHANGELOG.md` section for the release to `## vX.Y.Z`.  The
   GitHub release notes are that section.
3. Commit, push to `main`, and wait for CI to pass.
4. Run the pipeline without publishing, and wait for it to pass:

        gh workflow run publish.yml --ref main

5. Run [pg-phenotype-validate](https://github.com/rwaples/pg-phenotype-validate)
   against the candidate, as its README's "Checking a local pg-phenotype
   checkout" describes: point its `pg-phenotype` entry at this checkout,
   build the R tarball with `pixi run -e r tools/r_build_tarball.sh`, and
   run `PGP_VALIDATE_R_TARBALL=<tarball> pixi run harness` there.  Don't tag
   while a row is FAIL or ERROR.  A row whose status changed needs its
   differences.md entry updated there, after the release, against the
   published pair.
6. Tag the commit and push the tag:

        git tag vX.Y.Z
        git push origin vX.Y.Z

7. To submit the R package to CRAN, upload the release's
   `pgphenotype_X.Y.Z.tar.gz` at <https://cran.r-project.org/submit.html>.
   The workflow does not submit to CRAN.
