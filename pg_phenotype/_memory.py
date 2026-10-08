"""Hand the host's free heap memory back to the system before native work.

The native computation runs on the package's pool threads.  glibc keeps the
memory a thread frees in that thread's arena, so memory the caller freed on
its own thread (while parsing the input, say) stays resident while the pool
allocates afresh in its own arenas.  ``malloc_trim(0)`` returns those free
pages first.  Elsewhere (macOS, Windows, musl) this is a no-op.
"""

from __future__ import annotations

__all__ = ["release_free_memory"]

import ctypes
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def _malloc_trim() -> Callable[[int], int] | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(None)
    except OSError:
        return None
    # musl has no malloc_trim.
    return getattr(libc, "malloc_trim", None)


_MALLOC_TRIM = _malloc_trim()


def release_free_memory() -> None:
    """Return the process's free heap pages to the system (glibc only)."""
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)
