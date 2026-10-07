"""Package-wide thread budget.

One process-global budget covers every parallel code path.  The
budget resolves as ``configure_threads(n)``
> the ``PG_PHENOTYPE_THREADS`` environment variable > ``1``, and it is
*committed* the first time :func:`thread_budget` resolves it.  Reconfiguring to a
different value after that is an error, so work already dispatched under the
committed budget cannot be invalidated behind its own back (pedigree-graph
ADR 0007, which this follows).  The budget sizes this package's own Rayon
pool; pedigree-graph's pool, in a process that imports both, is separate.
"""

from __future__ import annotations

__all__ = ["configure_threads", "thread_budget"]

import os
from dataclasses import dataclass

_ENV_VAR = "PG_PHENOTYPE_THREADS"
_DEFAULT_THREADS = 1


@dataclass(slots=True)
class _ThreadState:
    """Resolution state of the package-wide thread budget.

    Attributes:
        configured: Value handed to :func:`configure_threads`, or ``None`` if it
            was never called.
        committed: Budget resolved by the first :func:`thread_budget` call, or
            ``None`` while the budget is still open to configuration.
    """

    configured: int | None = None
    committed: int | None = None


_STATE = _ThreadState()


def configure_threads(n: int) -> None:
    """Set the package-wide thread budget.

    Args:
        n: Thread budget, an ``int`` >= 1.  ``bool`` is rejected.

    Raises:
        ValueError: If ``n`` is not an ``int`` >= 1.
        RuntimeError: If the budget is already committed to a different value.
            The native pool raises the same class for the same reason, so a
            budget that reaches it late fails the same way.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"configure_threads(n) requires an int >= 1, got {n!r}")
    committed = _STATE.committed
    if committed is not None:
        if n != committed:
            raise RuntimeError(
                f"the committed thread budget is {committed} and cannot be changed to {n}; "
                "call configure_threads() before the first thread_budget() call"
            )
        return
    _STATE.configured = n


def thread_budget() -> int:
    """Return the package-wide thread budget, committing it on the first call.

    The budget is the value handed to :func:`configure_threads` if there was one,
    else ``PG_PHENOTYPE_THREADS`` parsed as a decimal integer >= 1, else ``1``.
    Later calls return the committed value even if the environment changes.

    Returns:
        The committed thread budget, an ``int`` >= 1.

    Raises:
        ValueError: If ``PG_PHENOTYPE_THREADS`` is set to anything but a
            decimal integer >= 1.
    """
    committed = _STATE.committed
    if committed is None:
        configured = _STATE.configured
        committed = configured if configured is not None else _budget_from_env()
        _STATE.committed = committed
    return committed


def _budget_from_env() -> int:
    """Return the budget requested by ``PG_PHENOTYPE_THREADS``.

    Returns:
        The parsed environment value, or ``1`` when the variable is unset.

    Raises:
        ValueError: If the variable is set to anything but a decimal integer >= 1.
    """
    raw = os.environ.get(_ENV_VAR)
    if raw is None:
        return _DEFAULT_THREADS
    if not (raw.isascii() and raw.isdigit()) or int(raw) < 1:
        raise ValueError(f"{_ENV_VAR} must be a decimal integer >= 1, got {raw!r}")
    return int(raw)


def _reset_thread_state() -> None:
    """Clear the configured and committed budget.  For tests only.

    This resets the Python budget alone.  The native Rayon pool is built once
    per process from the first committed budget and has no reset, so a test
    that resets here, commits a *different* budget, and then reaches a native
    relationship call gets ``RuntimeError`` from the pool.  Cross-budget tests
    therefore run in a child process.
    """
    _STATE.configured = None
    _STATE.committed = None
