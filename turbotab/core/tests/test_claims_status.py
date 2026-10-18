"""A claim's status is derived, never declared, except a retraction (CLAIMS_REGISTRY.md §6;
R0_PLAN.md R0a acceptance 3).

A claim is built when every ``enforced_by`` resolves and every test it names exists; otherwise it
is proposed. An enforcer resolves to any name its module binds at top level; a test exists only as
a function (or a method of a class) its file defines, and not one marked to skip or xfail
unconditionally, which never asserts (a ``skipif`` or ``xfail`` on a condition is a platform guard,
and counts). Both are read from the source with ``ast``, never imported: a test file that would
fail on import still defines its functions, and nothing a status check reads is executed. Each
case is planted in a temporary repository whose files the test writes, so the truth is the
construction; which planted tests never assert is also checked against pytest's own run of them.

A claim the research swarm proposed is held proposed while any part it asserts is unsupported,
whatever its code and tests (R1_PLAN.md R1-G): it enters so, every part unsupported, and so
renders nowhere until its parts are graded.
"""
from __future__ import annotations

import os
import subprocess
import sys
from xml.etree import ElementTree

import pytest

from turbotab.core.claims import Basis, Fires, NotRenderable, Part, Silent
from turbotab.core.tests.test_claims_fixtures import (FIRES, SILENT, registry, rule, swarm,
                                                       teaching)

ID = "planted.few_clusters"


def test_a_claim_whose_enforcers_resolve_and_whose_tests_exist_is_built(tmp_path):
    assert registry(tmp_path, rule()).status(ID) == "built"


@pytest.mark.parametrize("enforcer", [
    "planted.nowhere:act",  # no such module
    "planted.engine:not_written_yet",  # no such function
    "planted.engine:Guard.not_written_yet",  # no such method
    "planted.engine:inner",  # defined only inside another function
])
def test_an_enforcer_that_does_not_resolve_leaves_the_claim_proposed(tmp_path, enforcer):
    reg = registry(tmp_path, rule(enforced_by=("planted.engine:act", enforcer)))
    assert reg.status(ID) == "proposed"


@pytest.mark.parametrize("enforcer", [
    "planted.engine:act", "planted.engine:Guard", "planted.engine:Guard.check",
    "planted.engine:FEW",
])
def test_a_function_a_class_a_method_or_a_name_assigned_at_top_level_resolves(tmp_path, enforcer):
    assert registry(tmp_path, rule(enforced_by=(enforcer,))).status(ID) == "built"


@pytest.mark.parametrize("test", [
    "planted/test_missing.py::test_nine_clusters_name_the_concern",  # no such file
    "planted/test_rule.py::test_not_written_yet",  # no such function
    "planted/test_rule.py::TestGrouped::test_not_written_yet",  # no such method
    "planted/test_rule.py::TestMissing::test_in_a_class",  # no such class
])
def test_a_test_that_does_not_exist_leaves_the_claim_proposed(tmp_path, test):
    reg = registry(tmp_path, rule(tests=(FIRES, SILENT, Fires(test, "construction"))))
    assert reg.status(ID) == "proposed"


@pytest.mark.parametrize("test", [
    "planted/test_rule.py::test_forty_clusters_say_nothing[forty]",
    "planted/test_rule.py::TestGrouped::test_in_a_class",
])
def test_a_parametrized_or_class_test_exists_by_its_function(tmp_path, test):
    reg = registry(tmp_path, rule(tests=(FIRES, Silent(test, "construction"))))
    assert reg.status(ID) == "built"


ODD = '''test_var = None
test_annotated: object = None
from os import path as test_imported


class test_klass:
    pass


class TestGroupedOnly:
    def test_in_it(self):
        pass


async def test_awaited():
    pass


def test_rebound():
    pass


test_rebound = None
test_redefined = None


def test_redefined():
    pass
'''


def _odd(tmp_path, test):
    reg = registry(tmp_path, rule(tests=(FIRES, SILENT, Fires(test, "construction"))))
    (reg.root / "planted" / "test_odd.py").write_text(ODD, encoding="utf-8")
    return reg


@pytest.mark.parametrize("name", [
    "test_var", "test_annotated", "test_imported",  # a name bound to a value or an import
    "test_klass", "TestGroupedOnly",  # a class, which holds tests but is not one
    "test_rebound",  # a function the file then binds to a value
])
def test_a_test_named_by_anything_but_a_function_leaves_the_claim_proposed(tmp_path, name):
    """pytest runs a function; a variable, an import or a class of that name is not a test."""
    assert _odd(tmp_path, f"planted/test_odd.py::{name}").status(ID) == "proposed"


@pytest.mark.parametrize("name", ["test_awaited", "TestGroupedOnly::test_in_it", "test_redefined"])
def test_an_async_function_a_method_or_a_function_bound_last_is_a_test(tmp_path, name):
    assert _odd(tmp_path, f"planted/test_odd.py::{name}").status(ID) == "built"


def test_a_test_file_is_read_not_imported(tmp_path):
    reg = registry(tmp_path, rule())
    (reg.root / "planted" / "test_rule.py").write_text(
        'raise RuntimeError("importing this file fails")\n\n\n'
        "def test_nine_clusters_name_the_concern():\n    pass\n\n\n"
        "def test_forty_clusters_say_nothing():\n    pass\n", encoding="utf-8")
    assert reg.status(ID) == "built"


def test_a_test_file_that_does_not_parse_defines_nothing(tmp_path):
    reg = registry(tmp_path, rule())
    (reg.root / "planted" / "test_rule.py").write_text(
        "def test_nine_clusters_name_the_concern(:\n", encoding="utf-8")
    assert reg.status(ID) == "proposed"


def test_a_test_written_after_the_claim_registers_makes_it_built(tmp_path):
    """The status follows the files: no stale cache keeps a claim proposed."""
    late = Fires("planted/test_late.py::test_written_later", "construction")
    reg = registry(tmp_path, rule(tests=(FIRES, SILENT, late)))
    assert reg.status(ID) == "proposed"
    (reg.root / "planted" / "test_late.py").write_text("def test_written_later():\n    pass\n",
                                                       encoding="utf-8")
    assert reg.status(ID) == "built"


def test_a_retraction_is_the_one_declared_status_and_wins(tmp_path):
    reg = registry(tmp_path, rule(retracted="2026-10-16: a planted retraction"))
    assert reg.status(ID) == "retracted"


def test_a_teaching_claim_with_no_tests_or_enforcers_is_built(tmp_path):
    assert registry(tmp_path, teaching()).status("planted.your_call") == "built"


def test_an_unknown_claim_has_no_status(tmp_path):
    with pytest.raises(KeyError, match="planted.nothing"):
        registry(tmp_path).status("planted.nothing")


# ── a test marked to skip or xfail unconditionally never asserts, so it is no test for status ──

MARKED = '''"""Planted tests under skip and xfail marks: the unconditional ones never assert."""
import sys

import pytest
from pytest import mark


@pytest.mark.skip
def test_skip_bare():
    assert False


@pytest.mark.skip()
def test_skip_called():
    assert False


@pytest.mark.skip(reason="not yet")
def test_skip_with_a_reason():
    assert False


@pytest.mark.skip("not yet")
def test_skip_with_a_positional_reason():
    assert False


@mark.skip
def test_skip_from_mark():
    assert False


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.skip(reason="not yet")
def test_skip_under_parametrize(n):
    assert False


@pytest.mark.xfail
def test_xfail_bare():
    assert False


@pytest.mark.xfail()
def test_xfail_called():
    assert False


@pytest.mark.xfail(reason="not yet", strict=True)
def test_xfail_with_a_reason():
    assert False


@pytest.mark.xfail(True, reason="always")
def test_xfail_on_true():
    assert False


@pytest.mark.skipif(True, reason="always")
def test_skipif_on_true():
    assert False


@pytest.mark.skipif(reason="no condition, so it always skips")
def test_skipif_with_no_condition():
    assert False


@pytest.mark.skip
class TestSkippedClass:
    def test_in_it(self):
        assert False


class TestMarkedClass:
    pytestmark = pytest.mark.xfail(reason="not yet")

    def test_in_it(self):
        assert False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX paths")
def test_skipif_on_a_platform():
    assert True


@pytest.mark.xfail(sys.platform == "win32", reason="POSIX paths")
def test_xfail_on_a_platform():
    assert True


@pytest.mark.xfail(condition=sys.version_info < (3, 9), reason="an old Python")
def test_xfail_on_a_named_condition():
    assert True


@pytest.mark.skipif("sys.platform == 'win32'", reason="a condition as a string")
def test_skipif_on_a_string_condition():
    assert True


@pytest.mark.xfail(False, reason="never")
def test_xfail_on_false():
    assert True


@pytest.mark.slow
@pytest.mark.parametrize("n", [1, 2])
def test_under_other_marks(n):
    assert True


class TestPlainClass:
    def test_in_it(self):
        assert True
'''
MODULE_SKIPPED = ('import pytest\n\npytestmark = [pytest.mark.slow, pytest.mark.skip(reason="not '
                  'yet")]\n\n\ndef test_in_it():\n    assert False\n')
MODULE_GUARDED = ('import sys\n\nimport pytest\n\npytestmark = pytest.mark.skipif(sys.platform == '
                  '"win32", reason="POSIX paths")\n\n\ndef test_in_it():\n    assert True\n')
NEVER_RUN = [
    "test_marks.py::test_skip_bare", "test_marks.py::test_skip_called",
    "test_marks.py::test_skip_with_a_reason", "test_marks.py::test_skip_with_a_positional_reason",
    "test_marks.py::test_skip_from_mark", "test_marks.py::test_skip_under_parametrize[1]",
    "test_marks.py::test_xfail_bare", "test_marks.py::test_xfail_called",
    "test_marks.py::test_xfail_with_a_reason", "test_marks.py::test_xfail_on_true",
    "test_marks.py::test_skipif_on_true", "test_marks.py::test_skipif_with_no_condition",
    "test_marks.py::TestSkippedClass::test_in_it", "test_marks.py::TestMarkedClass::test_in_it",
    "test_module_skipped.py::test_in_it",
]
RUN = [
    "test_marks.py::test_skipif_on_a_platform", "test_marks.py::test_xfail_on_a_platform",
    "test_marks.py::test_xfail_on_a_named_condition",
    "test_marks.py::test_skipif_on_a_string_condition", "test_marks.py::test_xfail_on_false",
    "test_marks.py::test_under_other_marks[2]", "test_marks.py::TestPlainClass::test_in_it",
    "test_module_guarded.py::test_in_it",
]


def _marked(tmp_path, test):
    reg = registry(tmp_path, rule(tests=(FIRES, SILENT, Fires(f"planted/{test}", "construction"))))
    for name, text in (("test_marks.py", MARKED), ("test_module_skipped.py", MODULE_SKIPPED),
                       ("test_module_guarded.py", MODULE_GUARDED)):
        (reg.root / "planted" / name).write_text(text, encoding="utf-8")
    return reg


@pytest.mark.parametrize("test", NEVER_RUN)
def test_a_test_marked_to_skip_or_xfail_unconditionally_leaves_the_claim_proposed(tmp_path, test):
    """A bare or called ``skip`` (with a reason or not), an ``xfail`` or ``skipif`` with no
    condition or a literal true one, on the function, its class or its module's ``pytestmark``."""
    assert _marked(tmp_path, test).status(ID) == "proposed"


@pytest.mark.parametrize("test", RUN)
def test_a_test_under_a_platform_guard_or_another_mark_counts(tmp_path, test):
    """``skipif`` and ``xfail`` on a condition are platform guards: the test runs elsewhere."""
    assert _marked(tmp_path, test).status(ID) == "built"


def test_which_planted_tests_never_assert_is_what_pytest_itself_reports(tmp_path):
    """pytest runs the planted files on their own: the tests it skips or xfails here are the ones
    the registry counts as never asserting, and the ones it passes are the ones it counts."""
    reg = _marked(tmp_path, NEVER_RUN[0])
    root = reg.root / "planted"
    (root / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: planted\n", encoding="utf-8")
    report = tmp_path / "report.xml"
    files = ["test_marks.py", "test_module_skipped.py", "test_module_guarded.py"]
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c",
                    str(root / "pytest.ini"), "--rootdir", str(root), f"--junitxml={report}",
                    *files], cwd=root, env=env, capture_output=True, text=True, timeout=120,
                   check=True)
    ran, never = set(), set()
    for case in ElementTree.parse(report).iter("testcase"):
        module, _, cls = case.get("classname").partition(".")
        test = f"{module}.py::" + (f"{cls}::" if cls else "") + case.get("name")
        (never if case.find("skipped") is not None else ran).add(test)
    assert never >= set(NEVER_RUN) and ran >= set(RUN)
    for test in sorted(ran | never):
        expected = "proposed" if test in never else "built"
        assert _marked(tmp_path, test).status(ID) == expected, test


# ── a swarm claim enters proposed, every part unsupported (R1_PLAN.md R1-G) ───────────────────

SWARM = "planted.swarm_found"
JUDGED = Part("value", grade="judgment", basis=Basis(reason="a planted judgment"),
              graded_by="the orchestrator, 2026-10-18")


def test_a_swarm_claim_with_an_unsupported_part_is_proposed_though_its_code_and_tests_exist(
        tmp_path):
    reg = registry(tmp_path, swarm(), swarm(id="planted.swarm_half",
                                            parts=(JUDGED, Part("scope", grade="unsupported"))))
    assert reg.resolves("planted.engine:act") and reg.test_exists(FIRES.test)
    assert reg.status(SWARM) == "proposed"
    assert reg.status("planted.swarm_half") == "proposed"


def test_a_swarm_claim_whose_parts_are_all_graded_is_built(tmp_path):
    assert registry(tmp_path, swarm(parts=(JUDGED,))).status(SWARM) == "built"


def test_a_claim_of_another_origin_with_an_unsupported_part_is_built(tmp_path):
    """The hold is the swarm's alone: an engine or audit claim with an unsupported part renders as
    a labeled convention (CLAIMS_REGISTRY.md §11, ruling 2)."""
    for origin in ("engine", "audit"):
        reg = registry(tmp_path, swarm(origin=origin))
        assert reg.status(SWARM) == "built", origin


def test_a_swarm_claim_as_it_enters_renders_nowhere(tmp_path):
    reg = registry(tmp_path, swarm())
    for render in (reg.card, reg.methods):
        with pytest.raises(NotRenderable, match="proposed"):
            render(SWARM, strict=False)
    assert registry(tmp_path, swarm(parts=(JUDGED,))).card(SWARM, strict=False).startswith(
        "Fewer than 10 clusters")
