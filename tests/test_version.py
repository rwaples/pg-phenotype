import pg_phenotype
from pg_phenotype import _native


def test_distribution_version_is_the_cargo_workspace_version():
    assert pg_phenotype.__version__ == _native.core_version()


def test_pg_core_rev_is_a_full_git_sha():
    rev = pg_phenotype.pg_core_rev()
    assert len(rev) == 40
    assert all(c in "0123456789abcdef" for c in rev)
