"""Provenance must record the target trim the page actually set.

`pages/06_Train_and_Compare.py` builds a `SplitConfig` and writes the user's
target-trimming choice onto it as `target_trim_enabled` / `target_trim_lower` /
`target_trim_upper`. The provenance call read `trim_target` / `trim_lower` /
`trim_upper` — names that exist nowhere — so `getattr(..., default)` swallowed
the miss and `record_split` was *always* told trimming was off with bounds
[0.0, 1.0].

The consequence is not cosmetic. The on-screen caption and the methodology log
use the correct fields, so the screen says rows were removed and the Methods
section says they were not; whichever channel the reader trusts decides whether
a published paper discloses that rows were dropped from the outcome
distribution before splitting. An undisclosed exclusion criterion.

Findings: CONTRACT-009, STATE-007, STATE-016.

The test reads the page's own source rather than a copy of it: it extracts the
three argument expressions from the real `record_split(...)` call and evaluates
them against a real `SplitConfig`. Renaming the attributes back — or reaching
for any other name that does not exist on `SplitConfig` — puts the defaults
back and fails here.
"""
from __future__ import annotations

import ast
import os

import pytest

from utils.session_state import SplitConfig
from utils.workflow_provenance import WorkflowProvenance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "pages", "06_Train_and_Compare.py")

TRIM_KWARGS = ("target_trim_enabled", "target_trim_lower", "target_trim_upper")


def _record_split_call() -> ast.Call:
    """The one `record_split(...)` call in the Train & Compare page."""
    with open(PAGE, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=PAGE)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_split"
    ]
    assert len(calls) == 1, (
        f"expected exactly one record_split call in {PAGE}, found {len(calls)}")
    return calls[0]


def _trim_arguments() -> dict[str, ast.expr]:
    call = _record_split_call()
    args = {kw.arg: kw.value for kw in call.keywords if kw.arg in TRIM_KWARGS}
    missing = [name for name in TRIM_KWARGS if name not in args]
    assert not missing, (
        "the split provenance call no longer passes " + ", ".join(missing)
        + " — the Methods section cannot state an exclusion it is never told about")
    return args


def _trimmed_config() -> SplitConfig:
    cfg = SplitConfig(train_size=0.7, val_size=0.15, test_size=0.15)
    cfg.target_trim_enabled = True
    cfg.target_trim_lower = 0.05
    cfg.target_trim_upper = 0.95
    return cfg


def _evaluate(expr: ast.expr, split_config: SplitConfig):
    """Evaluate one argument expression from the page against a real config."""
    return eval(  # noqa: S307 - the expression comes from our own source tree
        compile(ast.Expression(body=expr), PAGE, "eval"),
        {"__builtins__": __builtins__},
        {"split_config": split_config},
    )


# ── the three names the page reads must be the three names it wrote ──────

@pytest.mark.parametrize("kwarg, expected", [
    ("target_trim_enabled", True),
    ("target_trim_lower", 0.05),
    ("target_trim_upper", 0.95),
])
def test_the_split_provenance_reads_the_trim_the_page_set(kwarg, expected):
    cfg = _trimmed_config()
    got = _evaluate(_trim_arguments()[kwarg], cfg)
    assert got == expected, (
        f"record_split was told {kwarg}={got!r} while the page set "
        f"{kwarg}={expected!r} on the SplitConfig — the screen and the "
        f"manuscript disagree about whether rows were removed")


def test_the_recorded_split_states_the_exclusion_that_happened():
    """End to end: the page's own expressions, into a real provenance record."""
    cfg = _trimmed_config()
    args = {name: _evaluate(expr, cfg) for name, expr in _trim_arguments().items()}

    prov = WorkflowProvenance()
    prov.record_split(strategy="random", train_n=70, val_n=15, test_n=15, **args)

    assert prov.split.target_trim_enabled is True, (
        "provenance recorded trimming as disabled while it was on — an "
        "undisclosed exclusion criterion")
    assert prov.split.target_trim_lower == pytest.approx(0.05)
    assert prov.split.target_trim_upper == pytest.approx(0.95)


def test_trimming_left_off_is_still_recorded_as_off():
    """The fix must not invert the bug: an untrimmed run stays untrimmed."""
    cfg = SplitConfig()
    args = {name: _evaluate(expr, cfg) for name, expr in _trim_arguments().items()}

    prov = WorkflowProvenance()
    prov.record_split(strategy="random", train_n=70, val_n=15, test_n=15, **args)

    assert prov.split.target_trim_enabled is False
    assert prov.split.target_trim_lower == pytest.approx(0.0)
    assert prov.split.target_trim_upper == pytest.approx(1.0)


def test_no_getattr_default_hides_a_renamed_split_config_field():
    """The class fix: every attribute the call reads must exist on SplitConfig.

    `getattr(obj, 'literal', default)` against a typed config turns a rename
    into a silent wrong answer. Any name read off `split_config` in the
    provenance call has to be a real field.
    """
    call = _record_split_call()
    fields = set(SplitConfig.__dataclass_fields__)

    unknown = []
    for node in ast.walk(call):
        # getattr(split_config, 'name', default)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "split_config"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value not in fields):
            unknown.append(node.args[1].value)
        # split_config.name
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "split_config"
                and node.attr not in fields):
            unknown.append(node.attr)

    assert not unknown, (
        "the split provenance call reads names that are not SplitConfig "
        f"fields: {sorted(set(unknown))} — getattr's default will be recorded "
        "as if the user had chosen it")
