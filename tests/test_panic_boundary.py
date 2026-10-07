"""A Rust panic reaches Python as an exception, not an abort."""

import os

import pytest

from pg_phenotype import _native


def test_panic_is_an_exception_not_an_abort():
    probe = getattr(_native, "_panic_for_test", None)
    if probe is None:
        if os.environ.get("PG_PHENOTYPE_REQUIRE_TEST_HOOKS") == "1":
            pytest.fail("the extension was built without the test-hooks feature")
        pytest.skip("installed wheel without test hooks")
    with pytest.raises(BaseException, match="deliberate panic") as err:
        probe()
    assert type(err.value).__name__ == "PanicException"
