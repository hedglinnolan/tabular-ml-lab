"""Planted claims and a planted repository for the claims registry's tests. No real claim is here.

A planted root is a temporary directory laid out as the repository is: an engine module with an
enforcer and a constant, a test file whose two functions exist, and a derivation note. A claim
built against it is ``built``; one that names a missing piece is ``proposed``. :func:`registry`
also writes the root's frozen status file, as the generator does where the source is complete, so
its claims render (``test_claims_frozen.py``), and :func:`write_history` writes its history as the
generator does, so meta-test 8 has the grade lines it reads. The predicates below are real
functions, named as ``"module:function"`` strings, so a regime check imports them as it would an
engine predicate. :func:`swarm` is a claim the research swarm proposed, as it enters: every part
unsupported. :func:`bound` binds each referee and basis audit a planted claim names to its part as
it stands, as a promotion records them.

This module holds no test; the claims tests import it.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from turbotab.core.claims import (Audit, Basis, Claim, Fires, Param, Part, Regime, Registry,
                                  Silent, checked, frozen_text)

HERE = "turbotab.core.tests.test_claims_fixtures"

ENGINE = '''"""A planted engine module."""
FEW = 10


def act(frame):
    return frame


class Guard:
    def check(self):
        return True


def outer():
    def inner():
        return None
    return inner
'''

TESTS = '''"""Planted known-truth tests: nine clusters are fewer than ten, forty are not."""


def test_nine_clusters_name_the_concern():
    assert 9 < 10


def test_forty_clusters_say_nothing():
    assert not 40 < 10


class TestGrouped:
    def test_in_a_class(self):
        assert True
'''

FIRES = Fires("planted/test_rule.py::test_nine_clusters_name_the_concern", truth="construction")
SILENT = Silent("planted/test_rule.py::test_forty_clusters_say_nothing", truth="construction")
DERIVATION = "notes/derivation.md"


def outcome_is_binary(ctx: Any) -> bool:
    return ctx.get("outcome") == "binary"


def any_design(ctx: Any) -> bool:
    return True


def not_a_bool(ctx: Any) -> Any:
    return "yes"


def planted_root(tmp_path: Path) -> Path:
    """A temporary repository: ``planted/engine.py``, ``planted/test_rule.py`` and a note."""
    (tmp_path / "planted").mkdir(exist_ok=True)
    (tmp_path / "planted" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "planted" / "engine.py").write_text(ENGINE, encoding="utf-8")
    (tmp_path / "planted" / "test_rule.py").write_text(TESTS, encoding="utf-8")
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / DERIVATION).write_text("# A planted derivation\n", encoding="utf-8")
    return tmp_path


def rule(**over: Any) -> Claim:
    """A conforming rule claim against a planted root, with any field replaced."""
    base = Claim(
        id="planted.few_clusters",
        kind="rule",
        card="Fewer than {few} clusters: intervals can still be too narrow.",
        methods="With fewer than {few} clusters, the concern is named.",
        goal=("inference",),
        regime=Regime("a yes/no outcome, GEE with a small-sample correction",
                      requires=(f"{HERE}:outcome_is_binary",)),
        decision="planted.correction",
        options=("gee_corrected", "mixed_kr"),
        verdict="noted",
        parts=(Part("value", grade="judgment", basis=Basis(reason="a planted judgment")),),
        params={"few": Param(10, unit="clusters", constant="planted.engine:FEW")},
        tests=(FIRES, SILENT),
        enforced_by=("planted.engine:act",),
        author="the R0a builder",
        origin="engine",
    )
    return dataclasses.replace(base, **over)


def teaching(**over: Any) -> Claim:
    """A conforming teaching claim with no parameter, test or enforcer."""
    base = Claim(
        id="planted.your_call",
        kind="teaching",
        card="Your call: here is what each choice does.",
        methods="The analyst chose between the options.",
        regime=Regime("any design"),
        parts=(Part("scope", grade="judgment", basis=Basis(reason="a planted judgment")),),
        author="the R0a builder",
        origin="engine",
    )
    return dataclasses.replace(base, **over)


def swarm(**over: Any) -> Claim:
    """A rule the research swarm proposed, as it enters the registry (R1_PLAN.md R1-G): its code
    and tests are the planted root's, and its one part is unsupported."""
    base = rule(id="planted.swarm_found", origin="swarm", author="theorist A",
                parts=(Part("value", grade="unsupported"),))
    return dataclasses.replace(base, **over)


def bound(c: Claim) -> Claim:
    """``c`` with each referee that names no grade given its part's, and each referee and basis
    audit that names no fingerprint given its part's as it stands (``claims.checked``), as a
    promotion records them; a grade or fingerprint already named is kept, so a planted claim can
    name a stale one."""
    parts = []
    for p in c.parts:
        now = checked(c, p)
        referees = tuple(dataclasses.replace(r, grade=r.grade or p.grade, checked=r.checked or now)
                         for r in p.referees)
        audit = p.audited_by
        if isinstance(audit, Audit) and not audit.checked:
            audit = dataclasses.replace(audit, checked=now)
        parts.append(dataclasses.replace(p, referees=referees, audited_by=audit))
    return dataclasses.replace(c, parts=tuple(parts))


def write_frozen(reg: Registry) -> Registry:
    """Freeze each claim's derived status and hash in the registry's frozen file, as the generator
    does."""
    reg.frozen_path.parent.mkdir(parents=True, exist_ok=True)
    reg.frozen_path.write_text(frozen_text(reg), encoding="utf-8")
    return reg


def registry(tmp_path: Path, *claims: Claim, freeze: bool = True) -> Registry:
    """A registry of ``claims`` on a planted root, with its status frozen unless ``freeze`` is
    False."""
    reg = Registry(claims, root=planted_root(tmp_path))
    return write_frozen(reg) if freeze else reg


def write_history(reg: Registry, today: str = "2026-10-18") -> Path:
    """Append the history the generator would for ``reg``, at the registry's own history path, with
    ``today`` as the day a line is recorded on where the source names none; the reference it also
    writes lands beside the history."""
    from turbotab.core.reference import claims as reference

    path = reg.history_path
    path.parent.mkdir(parents=True, exist_ok=True)
    code = reference.main(["--out", str(path.with_name("CLAIMS_REFERENCE.md")), "--history",
                           str(path), "--date", today], registry=reg)
    assert code == 0, "the generator refused the planted registry"
    return path
