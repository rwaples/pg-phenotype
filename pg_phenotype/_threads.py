"""Package-wide thread budget.

One process-global budget covers every parallel code path.  The
budget resolves as ``configure_threads(n)``
> the ``PG_PHENOTYPE_THREADS`` environment variable > ``1``, and it is
*committed* the first time :func:`thread_budget` resolves it.  Reconfiguring to a
different value after that is an error, so work already dispatched under the
committed budget cannot be invalidated behind its own back (pedigree-graph
ADR 0007, which this follows).  The budget sizes this package's own Rayon
pool; pedigree-graph's pool, in a process that imports both, is separate.

The state lives in the Rust core (``pg_phenotype_core::threads``), which the
R package reads too, so both hosts follow one set of rules.
"""

from __future__ import annotations

__all__ = ["configure_threads", "thread_budget"]

from pg_phenotype import _native


def configure_threads(n: int) -> None:
    """Set the package-wide thread budget.

    Args:
        n: Thread budget, an ``int`` from 1 to 2**31 - 1.  ``bool`` is rejected.

    Raises:
        ValueError: If ``n`` is not such an ``int``.
        RuntimeError: If the budget is already committed to a different value.
            The native pool raises the same class for the same reason, so a
            budget that reaches it late fails the same way.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"configure_threads(n) requires an int >= 1, got {n!r}")
    try:
        _native.configure_threads(n)
    except OverflowError:
        raise ValueError(f"configure_threads(n) requires an int from 1 to 2**31 - 1, got {n!r}") from None


def thread_budget() -> int:
    """Return the package-wide thread budget, committing it on the first call.

    The budget is the value handed to :func:`configure_threads` if there was one,
    else ``PG_PHENOTYPE_THREADS`` parsed as a decimal integer >= 1, else ``1``.
    Later calls return the committed value even if the environment changes.

    Returns:
        The committed thread budget, an ``int`` >= 1.

    Raises:
        ValueError: If ``PG_PHENOTYPE_THREADS`` is set to anything but a
            decimal integer from 1 to 2**31 - 1.
    """
    return _native.thread_budget()


def _reset_thread_state() -> None:
    """Clear the configured and committed budget.  For tests only.

    The native Rayon pool is built once per process from the first committed
    budget and has no reset, so a test that resets here, commits a
    *different* budget, and then reaches a native call gets ``RuntimeError``
    from the pool.  Cross-budget tests therefore run in a child process.
    """
    _native._reset_thread_budget()
