"""The private seam for eligibility logic.

Which rules exist, and the questions that surface them, are public — they are
transcribed from published protocol text and there is nothing to protect there.

*How* answers are weighed is not. The reference matcher below is deliberately
transparent and deliberately simple: it applies each rule literally and reports
what it found. A private package can register something better — thresholds,
precedence, ambiguity handling, a learned model — without this repository
changing, because the call site only ever asks `load_matcher()` for an object
with an `evaluate(criteria, answers)` method.

That mirrors `bone_viewer.solvers` and `bone_viewer.backends` in the other
project: a public register, implementations supplied through entry points.
"""

from __future__ import annotations

import os
from importlib.metadata import entry_points

ENTRY_POINT_GROUP = "clinical_trials.matchers"


class ReferenceMatcher:
    """Applies every rule literally. No weights, no precedence, no model."""

    name = "reference"

    def evaluate(self, criteria: dict, answers: dict):
        # Imported here so a private matcher can be used without the reference
        # implementation being importable at all.
        from .engine import evaluate as _evaluate
        return _evaluate(criteria, answers)


def load_matcher():
    """Return the first registered matcher, or the reference implementation.

    Selection is by environment variable first (`CT_MATCHER`) so a deployment can
    pin one explicitly, then by entry point, then by falling back. A broken
    third-party matcher must not take the endpoint down, hence the broad except.
    """
    wanted = os.environ.get("CT_MATCHER")
    try:
        candidates = list(entry_points(group=ENTRY_POINT_GROUP))
    except Exception:  # noqa: BLE001 - entry point machinery varies by version
        candidates = []

    for ep in candidates:
        if wanted and ep.name != wanted:
            continue
        try:
            loaded = ep.load()
            return loaded() if isinstance(loaded, type) else loaded
        except Exception as exc:  # noqa: BLE001
            print(f"matcher {ep.name!r} failed to load: {exc}")
            continue

    return ReferenceMatcher()


def describe_matcher():
    m = load_matcher()
    return {"name": getattr(m, "name", type(m).__name__),
            "entry_point_group": ENTRY_POINT_GROUP,
            "private": getattr(m, "name", "reference") != "reference"}
