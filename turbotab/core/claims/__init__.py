"""The claims registry (CLAIMS_REGISTRY.md): one home for each statement the app relies on to set,
rank, refuse or explain a decision, its number, its two sentences, its grade per part, the regime
its evidence covers, and the tests that hold the code to it.

* **The schema** (``_schema``): :class:`Claim`, its :class:`Part` s each graded on its own with a
  :class:`Basis`, its :class:`Regime` in words and as predicates, its :class:`Param` s, and its
  known-truth tests, :class:`Fires` and :class:`Silent`, each declaring where its truth comes from.
  A claim is checked against the vocabulary when it registers, as a method contract is.
* **Its origin** (R1_PLAN.md R1-G): every claim names one of :data:`ORIGINS`, engine, audit or
  swarm. A swarm claim enters proposed with every part unsupported and renders nowhere; a part
  leaves unsupported only as graded by someone other than its author, and above judgment only
  through the two-referee gate (:func:`referee_gate`, its :class:`Referee` s), its grade's own
  requirement and a basis audit (an :class:`Audit`, ``Part.audited_by``), meta-test 7. Each
  referee and audit names the part's fingerprint as it read it (:func:`checked`), and counts only
  while the part stands so. Every grade change is a line of the history with its date, grader and
  reason, which the generator writes, and meta-test 8 refuses a grade that differs from its last
  line, and a swarm claim's author, grader or referee edited after the history recorded it; a part
  its sentence no longer asserts is :data:`RETIRED` through such a line, never dropped.
* **The registry** (``_registry``): :class:`Registry`, and the default one, :data:`REGISTRY`, which
  :func:`registry` fills by scanning this package. Each public module here, ``claims/<area>.py``,
  declares one area's claims with :func:`register`; a private module (``_name``) is the registry's
  own. No list of the claims modules exists, so two packages adding claims never share one.
* **Status is derived** (§6): a claim is built when every ``enforced_by`` resolves and every test
  it names exists (and is not marked to skip or xfail unconditionally), read from the source and
  never imported; otherwise it is proposed. Retracted is the only declared status. The shipped app
  holds no tests, so the generator freezes each claim's status and content hash in
  ``status.json`` here (:data:`FROZEN`), and rendering reads that file, not the source.
* **Sentences** (:func:`card`, :func:`methods`) render only for a claim the frozen file lists as
  built with its current hash, and only inside its regime: strict mode (``strict=True``, or
  ``TURBOTAB_CLAIMS_STRICT=1``, which the core tests set) raises outside it, and also refuses
  where the frozen status is not the one the source now derives; otherwise the named fallback
  claim renders. :func:`value` reads a parameter, so a module constant can take its value from its
  claim.
* **The meta-tests** (``_meta``, :data:`META_TESTS`, eight of them) and **the digit gate's
  allowlist** (``digit_allowlist.json``).

The registry stands alone: importing it, and loading its claims modules, loads no other
``turbotab.core`` module. Predicates and enforcers are named as ``"module:function"`` strings and
resolved only when they are checked. ``python -m turbotab.core.reference.claims`` writes
``docs/turbotab-next/reference/CLAIMS_REFERENCE.md`` and ``status.json``, and keeps every claim's
versions and every part's grade changes in ``history.jsonl``.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from turbotab.core.claims._digits import (ALLOWLIST, CITATIONS, TAGS, citation_keys,
                                          citation_spans, load_allowlist, stale_allowlist)
from turbotab.core.claims._meta import (GRADE_LINE, META_TESTS, digit_gate, grades_met,
                                        grades_recorded, no_conflicts, no_self_pinning,
                                        none_proposed_renders, swarm_gate, tests_exist)
from turbotab.core.claims._registry import (FROZEN, GENERATOR, HISTORY, REPO, STRICT_ENV,
                                            ClaimError, NotRenderable, OutsideRegime, Registry,
                                            frozen_records, frozen_text, strict_mode)
from turbotab.core.claims._schema import (ABOVE_JUDGMENT, CHECKED_DIGITS, GRADED_FORM, GRADES,
                                          KINDS, ORIGINS, PURPOSES, REFEREE_VERDICTS, RETIRED,
                                          RUNGS, TRUTHS, VERDICTS, Audit, Basis, Claim, Fires,
                                          HeldOut, Param, Part, Referee, Regime, Silent, checked,
                                          content, content_hash, dated, fill, grader,
                                          referee_gate, same_person)

REGISTRY = Registry()
_loaded = False


def scan(package: str = __name__) -> list[str]:
    """Import every public module of ``package`` (each registers its claims), in name order."""
    pkg = importlib.import_module(package)
    names = sorted(f"{package}.{m.name}" for m in pkgutil.iter_modules(pkg.__path__)
                   if not m.name.startswith("_"))
    for name in names:
        importlib.import_module(name)
    return names


def registry() -> Registry:
    """The default registry, with every claims module in this package loaded."""
    global _loaded
    if not _loaded:
        scan()
        _loaded = True
    return REGISTRY


def register(c: Claim) -> Claim:
    """Declare a claim in the default registry (a claims module calls this at import)."""
    return REGISTRY.register(c)


def card(id: str, ctx: Any = None, strict: bool | None = None) -> str:
    return registry().card(id, ctx, strict)


def methods(id: str, ctx: Any = None, strict: bool | None = None) -> str:
    return registry().methods(id, ctx, strict)


def value(id: str, param: str) -> int | float:
    return registry().value(id, param)


def status(id: str) -> str:
    """The claim's status derived from the source, which only a complete checkout holds (the
    generator's and the meta-tests'); a sentence renders from the frozen one instead."""
    return registry().status(id)


__all__ = [
    "ABOVE_JUDGMENT", "ALLOWLIST", "Audit", "Basis", "CHECKED_DIGITS", "CITATIONS", "Claim",
    "ClaimError", "FROZEN", "Fires", "GENERATOR", "GRADED_FORM", "GRADES", "GRADE_LINE", "HISTORY",
    "HeldOut", "KINDS", "META_TESTS", "NotRenderable", "ORIGINS", "OutsideRegime", "PURPOSES",
    "Param", "Part", "REFEREE_VERDICTS", "REGISTRY", "REPO", "RETIRED", "RUNGS", "Referee",
    "Regime", "Registry", "STRICT_ENV", "Silent", "TAGS", "TRUTHS", "VERDICTS", "card", "checked",
    "citation_keys", "citation_spans", "content",
    "content_hash", "dated", "digit_gate", "fill", "frozen_records", "frozen_text", "grader",
    "grades_met", "grades_recorded", "load_allowlist", "methods", "no_conflicts",
    "no_self_pinning", "none_proposed_renders", "referee_gate", "register", "registry",
    "same_person", "scan", "stale_allowlist", "status", "strict_mode", "swarm_gate",
    "tests_exist", "value",
]
