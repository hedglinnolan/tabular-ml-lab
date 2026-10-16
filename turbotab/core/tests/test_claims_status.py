"""A claim's status is derived, never declared, except a retraction (CLAIMS_REGISTRY.md §6;
R0_PLAN.md R0a acceptance 3).

A claim is built when every ``enforced_by`` resolves and every test it names exists; otherwise it
is proposed. An enforcer resolves to any name its module binds at top level; a test exists only as
a function (or a method of a class) its file defines. Both are read from the source with ``ast``,
never imported: a test file that would
fail on import still defines its functions, and nothing a status check reads is executed. Each
case is planted in a temporary repository whose files the test writes, so the truth is the
construction.
"""
from __future__ import annotations

import pytest

from turbotab.core.claims import Fires, Silent
from turbotab.core.tests.test_claims_fixtures import FIRES, SILENT, registry, rule, teaching

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
