import tomllib
from pathlib import Path

import pg_phenotype
from pg_phenotype import _native

REPO = Path(__file__).resolve().parents[1]


def test_distribution_version_is_the_cargo_workspace_version():
    assert pg_phenotype.__version__ == _native.core_version()


def test_pg_core_rev_is_a_full_git_sha():
    rev = pg_phenotype.pg_core_rev()
    assert len(rev) == 40
    assert all(c in "0123456789abcdef" for c in rev)


def test_r_binding_crate_carries_the_workspace_lints():
    # The R crate is its own workspace, so it repeats the lint table.
    workspace = tomllib.loads((REPO / "Cargo.toml").read_text())["workspace"]["lints"]["clippy"]
    binding = tomllib.loads((REPO / "r" / "src" / "rust" / "Cargo.toml").read_text())["lints"]["clippy"]
    assert binding == workspace
