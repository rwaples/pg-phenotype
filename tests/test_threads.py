"""The package thread budget's rules, each in a fresh process (the budget is process-global)."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code: str, env_value: str | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "PG_PHENOTYPE_THREADS"}
    if env_value is not None:
        env["PG_PHENOTYPE_THREADS"] = env_value
    return subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {ROOT!r})\n" + code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_the_budget_defaults_to_one():
    assert _run("import pg_phenotype as p; print(p.thread_budget())") == "1"


def test_the_environment_sets_the_budget():
    assert _run("import pg_phenotype as p; print(p.thread_budget())", "3") == "3"


def test_configure_wins_over_the_environment_and_commits_on_first_use():
    code = (
        "import pg_phenotype as p\n"
        "p.configure_threads(5)\n"
        "p.configure_threads(2)\n"
        "print(p.thread_budget())\n"
        "p.configure_threads(2)\n"
        "try:\n"
        "    p.configure_threads(4)\n"
        "except RuntimeError as e:\n"
        "    print('RuntimeError', 'committed thread budget is 2' in str(e))\n"
    )
    assert _run(code, "3").split("\n") == ["2", "RuntimeError True"]


def test_a_native_call_commits_the_budget():
    code = (
        "import pg_phenotype as p\n"
        "from pg_phenotype import pafgrs\n"
        "pafgrs.prepare({'id': [1, 2], 'mother': [-1, -1], 'father': [-1, -1]})\n"
        "try:\n"
        "    p.configure_threads(4)\n"
        "except RuntimeError:\n"
        "    print('conflict')\n"
    )
    assert _run(code, "2") == "conflict"


@pytest.mark.parametrize("n", [0, -1, 1.5, True, "2", 2**31, 2**70])
def test_a_bad_budget_is_a_value_error(n):
    from pg_phenotype import configure_threads

    with pytest.raises(ValueError, match="configure_threads"):
        configure_threads(n)


@pytest.mark.parametrize("raw", ["", "two", "0", "+3", "3000000000"])
def test_a_bad_environment_budget_is_a_value_error(raw):
    code = (
        "import pg_phenotype as p\n"
        "try:\n"
        "    p.thread_budget()\n"
        "except ValueError as e:\n"
        "    print('ValueError', 'PG_PHENOTYPE_THREADS' in str(e))\n"
    )
    assert _run(code, raw) == "ValueError True"
