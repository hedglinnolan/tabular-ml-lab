"""A test that reads a source file must say what encoding it is in.

Python's `open()` and `Path.read_text()` default to the platform's preferred
encoding. On Linux and macOS that is UTF-8 and nothing goes wrong. On Windows
it is cp1252, which cannot decode an em dash — and this repository's source is
full of em dashes, because its comments are written in prose.

So a test that reads a page and asserts something about its text does not fail
on Windows. It raises `UnicodeDecodeError` while opening the file, before it
reaches its own assertion, and reports red for a reason that has nothing to do
with what it checks. Twenty-four of them were in that state:

    tests/test_drive8_explainability.py ………………… 12 assertions, none reached
    tests/test_working_table_is_accounted_for.py … 7
    tests/test_paper_risk_report.py ……………………………… 4
    tests/test_paper_risk_split_identity.py ……………… 1

Every one of them passed the moment it could read its file, so nothing had
drifted underneath them. That is the good outcome and it is not the point: for
however long they were red, the invariants they guard — the export recipe, the
explainability denominators, the working-table accounting, the split's row
identity — had a check that could not have noticed a violation. A guard that
cannot fail for the right reason is not a guard.

This is the fourth angle in the family `test_a_fixed_row_names_a_test_that_actually_runs`
names, one row down from its own:

    principle-locality      fails across SPACE        stated here, violated there
    expiring guarantees     fails across TIME         true then, false now
    an untriggered check    fails across OCCASION     correct always, run never
    an unreadable check     fails across ENVIRONMENT  green there, unreachable here

The rule is one keyword. Say the encoding.
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Where a test's own helpers live. `tools/` and `scripts/` are swept too
#: because the same helper shape appears there and the same platform reads it.
_SWEPT = ("tests", "tools", "scripts")


def _text_reads_in(source: str):
    """Every text-mode `open()` / `read_text()` in `source` that names none.

    Takes SOURCE, not a path, so the self-test below can exercise it without
    writing a file. That is not only tidier: `test_no_test_writes_a_path_git_tracks`
    counts write destinations it cannot statically resolve and bounds them, and
    a `write_text` into a `TemporaryDirectory` is exactly one of those. A guard
    that has to weaken another guard to test itself is paying for its coverage
    with someone else's.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        named = {k.arg for k in node.keywords if k.arg}
        if "encoding" in named:
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            # Binary mode has no encoding to declare.
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value or "")
            if "b" in mode:
                continue
            out.append((node.lineno, "open()"))
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "read_text":
            out.append((node.lineno, "read_text()"))
    # `ast.walk` is breadth-first, so the offender list would otherwise come out
    # in an order that has nothing to do with the file a reader is about to open.
    return sorted(out)


def _text_reads_without_an_encoding(path: pathlib.Path):
    return _text_reads_in(path.read_text(encoding="utf-8"))


def _swept_files():
    for directory in _SWEPT:
        base = ROOT / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def test_every_text_read_in_the_suite_names_its_encoding():
    """The whole rule, in one assertion.

    It is deliberately not scoped to files that read *source* — a test that
    reads a fixture, a manifest or its own temporary file has the same problem
    the moment that file gains a character cp1252 cannot spell, and the fix is
    the same keyword either way.
    """
    offenders = []
    for path in _swept_files():
        for line, kind in _text_reads_without_an_encoding(path):
            offenders.append(f"{path.relative_to(ROOT).as_posix()}:{line} {kind}")
    assert not offenders, (
        "these reads use the platform default encoding, so they raise "
        "UnicodeDecodeError on Windows instead of checking what they check:\n  "
        + "\n  ".join(offenders)
        + "\nPass encoding=\"utf-8\"."
    )


def test_a_path_compared_against_a_written_one_is_posix():
    """`str(Path)` is `pages\\10_Report_Export.py` on Windows.

    `test_the_module_has_a_consumer_outside_its_own_tests` built its importer
    list with `str(path.relative_to(ROOT))` and compared it against
    `"pages/10_Report_Export.py"` — a literal, written by a person, with
    forward slashes. It reported that a module had no consumer at the site its
    ledger row names, while the consumer was right there. Two other files in
    this suite already normalize with `.replace(os.sep, "/")`; this one did not.
    """
    src = (ROOT / "tests" / "integration"
           / "test_the_sample_size_bullet_is_a_claim_about_this_study.py"
           ).read_text(encoding="utf-8")
    assert "str(path.relative_to(ROOT))" not in src, (
        "the importer list is built with str(), which yields backslashes on "
        "Windows and cannot match the forward-slash page names it is compared "
        "against")
    assert "relative_to(ROOT).as_posix()" in src


def test_this_guard_would_have_caught_the_reads_it_was_written_for():
    """A meta-test that cannot fail is the thing it exists to prevent.

    The detector is run against a synthetic module holding one instance of each
    shape — the two that were actually found, plus a binary read and an
    encoded read that must NOT be reported.
    """
    sample = (
        'import pathlib\n'
        'a = open("pages/01_Upload_and_Audit.py").read()\n'          # line 2
        'b = (pathlib.Path("x") / "y.py").read_text()\n'             # line 3
        'c = open("z.bin", "rb").read()\n'                           # ok
        'd = open("w.py", encoding="utf-8").read()\n'                # ok
        'e = pathlib.Path("v.py").read_text(encoding="utf-8")\n'     # ok
    )
    found = _text_reads_in(sample)
    assert [line for line, _ in found] == [2, 3], found
    assert {kind for _, kind in found} == {"open()", "read_text()"}
