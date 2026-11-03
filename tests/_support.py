"""Hypothesis example budgets, scaled by ``HYPOTHESIS_PROFILE``.

A property test that needs a budget other than the default sets it through
``property_budget``, and ``HYPOTHESIS_PROFILE`` scales every one: ``fast``
(the default) runs it as written, ``thorough`` ten times over, for a release
or a property under suspicion (``pixi run test-thorough``).  The rest of the
family uses the same two names.  The profiles cover every test without a
budget: Hypothesis's default 100 examples, or ten times that.
"""

from __future__ import annotations

import os
from typing import Any

from hypothesis import settings

_PROFILE_SCALE = {"fast": 1, "thorough": 10}
_PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "fast")
if _PROFILE not in _PROFILE_SCALE:
    raise ValueError(f"HYPOTHESIS_PROFILE must be one of {sorted(_PROFILE_SCALE)}, got {_PROFILE!r}")

settings.register_profile("fast", deadline=None)
settings.register_profile("thorough", deadline=None, max_examples=1000)
settings.load_profile(_PROFILE)


def property_budget(max_examples: int, **kwargs: Any) -> settings:
    """Hypothesis settings for *max_examples* examples, scaled by ``HYPOTHESIS_PROFILE``, with no deadline.

    No deadline, because a per-example deadline flakes under pytest-xdist.
    *kwargs* pass through to :class:`hypothesis.settings`.
    """
    return settings(deadline=None, max_examples=max_examples * _PROFILE_SCALE[_PROFILE], **kwargs)
