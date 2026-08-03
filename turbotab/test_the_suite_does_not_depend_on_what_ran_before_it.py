"""`TEST-041` — a test file's result must not depend on which files ran first.

Found while fixing `TEST-040`, and it is the same property from the other side.
`TEST-040` is a count that moves with machine **load**; this is a count that
moves with **ordering**. `TEST-030` already names the second axis for
`tests/workflow/`; this is the `turbotab/` instance, and it was live.

**The mechanism.** `turbotab/figures.py` declares `REGISTRY = {}` and nothing
in that module fills it. It is filled as an **import side effect** of
`turbotab/figure_specs.py`, whose module body is a run of `register(...)`
calls. So a file that reads `FIG.REGISTRY` without importing `figure_specs`
somewhere is reading whatever *an earlier file in the session* happened to
load.

`test_the_companion_rule_reaches_the_document.py` did exactly that. Its first
test asserts ``len(declaring) >= 4`` over the registry; it imported
`turbotab.api` only *inside* two helper functions, and the first test takes no
fixture, so nothing had run. Alone the file reported::

    AssertionError: {}
    assert 0 >= 4

In the full suite it passed — because some earlier file had imported the
specs. **That is a false pass**: the assertion was made true by a different
file's import, not by the code under test. The full-suite green and the
isolated red are both honest runs of one tree.

**Why this is a static check and not a driven one.** The honest test of the
property is *run every file alone*, and that is ~40 pytest sessions. The claim
this file makes is deliberately narrower and genuinely about the file — trap 5
says reserve grep for exactly that — so it checks the one thing that makes the
isolated run differ: does a file that reads the registry import the thing that
fills it?

**The root fix is not here.** As long as a module-level dict is filled by
somebody else's import, this is a convention with a guard rather than a
structure. `TEST-041`'s `act` records the accessor refactor that would remove
the hazard; this check holds the line until then.
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Importing any of these runs `figure_specs`'s module body, which is the
#: thing that fills `figures.REGISTRY`. `api` and `figure_bundle` both import
#: it transitively; naming all three keeps the check from failing a file that
#: is genuinely self-sufficient by a different route.
POPULATORS = ("figure_specs", "figure_bundle", "turbotab.api")

#: The registry names whose emptiness is silent. A read of one of these is the
#: thing that needs a populator in the same file.
LAZY_REGISTRIES = ("REGISTRY", "PENDING")


def _reads_a_lazy_registry(src: str) -> bool:
    """Attribute access or bare name, via AST rather than substring.

    `"REGISTRY" in src` also matches the word inside a docstring — including
    this one — so the whole file would flag itself.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:                                      # pragma: no cover
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in LAZY_REGISTRIES:
            return True
        if isinstance(node, ast.Name) and node.id in LAZY_REGISTRIES:
            return True
    return False


def _is_populator_import(node) -> bool:
    if isinstance(node, ast.Import):
        return any(any(p in a.name for p in POPULATORS) for a in node.names)
    if isinstance(node, ast.ImportFrom):
        joined = f"{node.module or ''}." + ".".join(a.name for a in node.names)
        return any(p in joined for p in POPULATORS)
    return False


def _unsafe_reads(src: str):
    """Registry reads that may execute before this file has run a populator.

    **The first version of this was wrong and a revert probe caught it.** It
    asked *does the file import a populator anywhere*, and answered yes for
    the companion file on the strength of a `from turbotab import api` sitting
    inside two helper functions that the failing test never calls. So removing
    the real import came back `GREEN — NOT LOAD-BEARING`: the guard passed the
    exact file it was written for. Trap 2, in the guard rather than in the app.

    The rule that actually holds: a read is safe if a populator import runs
    before it. Statically that is —

    - a populator imported at **module scope** covers every read in the file,
      because the module body runs at collection; otherwise
    - a read inside a function is safe only if **that same function** imports
      a populator, which is what `test_a_stand_in_resolves_in_the_real_registry`
      does and is why a module-scope-only rule would wrongly fail it.

    Anything else is reported. This under-approximates — a helper that imports
    and is called first would be flagged — and that direction is the safe one:
    a false positive costs one explicit import, a false negative costs a count
    that moves with ordering.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:                                      # pragma: no cover
        return []
    if any(_is_populator_import(n) for n in tree.body):
        return []                                    # module scope covers all

    unsafe = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        reads = any(
            (isinstance(k, ast.Attribute) and k.attr in LAZY_REGISTRIES)
            or (isinstance(k, ast.Name) and k.id in LAZY_REGISTRIES)
            for k in ast.walk(node))
        if not reads:
            continue
        if not any(_is_populator_import(k) for k in ast.walk(node)):
            unsafe.append(node.name)
    return unsafe


def _registry_readers():
    for path in sorted(ROOT.glob("turbotab/test_*.py")):
        src = path.read_text(encoding="utf-8", errors="ignore")
        if _reads_a_lazy_registry(src):
            yield path, src


def test_every_registry_reader_imports_the_thing_that_fills_it():
    """The standing check.

    A file that reads `figures.REGISTRY` and imports no populator passes or
    fails depending on what ran before it, which means its verdict is not
    about the code.
    """
    offenders = {str(p.relative_to(ROOT)): _unsafe_reads(src)
                 for p, src in _registry_readers() if _unsafe_reads(src)}
    assert not offenders, (
        f"these read a lazily-populated registry with no populator imported "
        f"first: {offenders}. `figures.REGISTRY` is filled by importing "
        f"`turbotab.figure_specs`; without it the file reads whatever an "
        f"earlier test file loaded. Add the import at MODULE scope — function "
        f"scope only counts inside the same function as the read, because the "
        f"first test in a file runs before any fixture.")


def test_the_sweep_finds_the_readers_it_is_sweeping():
    """The positive control. Everything above is an absence claim, so a walk
    that matched nothing would report a clean tree it never read."""
    readers = [p.name for p, _ in _registry_readers()]
    assert len(readers) >= 8, (
        f"the sweep found only {len(readers)} registry readers; there were ten "
        f"when this was written, so the detector has stopped detecting: "
        f"{readers}")
    assert "test_the_companion_rule_reaches_the_document.py" in readers, (
        "the file this check was written for is not in the sweep's own "
        "denominator, so a regression in it would go unseen")


def test_the_detector_rejects_a_reader_with_no_populator():
    """And the negative control — the detector must actually say no.

    Built from source text rather than by deleting an import from a real file,
    because a check that only ever answers *yes* is the shape of trap 2.
    """
    bad = ("from turbotab import figures as FIG\n"
           "def test_x():\n"
           "    assert len(FIG.REGISTRY) >= 4\n")
    good = bad.replace("from turbotab import figures as FIG\n",
                       "from turbotab import figures as FIG\n"
                       "from turbotab import figure_specs\n")
    assert _reads_a_lazy_registry(bad), "the reader detector missed a read"
    assert _unsafe_reads(bad) == ["test_x"], (
        f"the detector cleared a file that imports no populator: "
        f"{_unsafe_reads(bad)}")
    assert _unsafe_reads(good) == [], (
        "the detector flagged a file whose module scope imports one")

    # AND THE CASE THAT BROKE THE FIRST VERSION: a populator imported inside a
    # helper the failing test never calls. `ast.walk` over the whole file says
    # "there is an import"; the read still runs first.
    elsewhere = ("from turbotab import figures as FIG\n"
                 "def _helper():\n"
                 "    from turbotab import api\n"
                 "    return api\n"
                 "def test_x():\n"
                 "    assert len(FIG.REGISTRY) >= 4\n")
    assert _unsafe_reads(elsewhere) == ["test_x"], (
        "a populator imported in a function the read never calls was accepted "
        "as covering the read — this is the exact hole a revert probe found "
        "in the first version of this check")

    # And the shape that is genuinely fine: import and read in one function.
    same_fn = ("from turbotab import figures as FIG\n"
               "def test_x():\n"
               "    import turbotab.figure_specs\n"
               "    assert len(FIG.REGISTRY) >= 4\n")
    assert _unsafe_reads(same_fn) == [], (
        "a file that imports the populator in the same function as the read "
        "is correct and must not be flagged")


def test_a_docstring_mentioning_the_word_is_not_a_read():
    """Why the reader detector is an AST walk and not a substring test: this
    file's own docstring says `REGISTRY` several times, and a substring check
    would flag the guard as its own offender."""
    prose = '"""A docstring about REGISTRY and PENDING."""\nx = 1\n'
    assert not _reads_a_lazy_registry(prose), (
        "prose mentioning the name reads as a registry access, so the sweep "
        "would flag every file that documents the hazard")
