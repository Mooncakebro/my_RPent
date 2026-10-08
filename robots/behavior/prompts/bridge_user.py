"""User prompt sections for a bridged evaluator episode."""

from __future__ import annotations

from robots.behavior.prompts import user as base

MODE = """BEHAVIOR bridge evaluation; one evaluator-owned episode per planner
invocation. The bridge cannot observe official predicate state. Use the active
public bridge tools and finish honestly; evaluator scoring is authoritative."""

CELL = base.CELL
BEGIN = base.BEGIN

__all__ = ["BEGIN", "CELL", "MODE"]
