"""Planted claims and a planted repository for the claims registry's tests. No real claim is here.

A planted root is a temporary directory laid out as the repository is: an engine module with an
enforcer and a constant, a test file whose two functions exist, and a derivation note. A claim
built against it is ``built``; one that names a missing piece is ``proposed``. The predicates
below are real functions, named as ``"module:function"`` strings, so a regime check imports them
as it would an engine predicate.

This module holds no test; the claims tests import it.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from turbotab.core.claims import Basis, Claim, Fires, Param, Part, Regime, Registry, Silent

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
    )
    return dataclasses.replace(base, **over)


def registry(tmp_path: Path, *claims: Claim) -> Registry:
    return Registry(claims, root=planted_root(tmp_path))
