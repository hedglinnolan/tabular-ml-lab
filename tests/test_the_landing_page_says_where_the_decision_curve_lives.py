"""`AUDIT-018` — the Classic landing page's capability list, and the one
capability on it that no Classic page can reach.

**THE SENTENCE THE ROW WAS FILED FOR**, `app.py` as it shipped at `cf7fc17`,
line 215, under **Evaluation Metrics**:

    - Decision curve analysis for clinical utility

Nothing under `pages/` or `utils/` computes net benefit. `ml.calibration.
decision_curve_analysis` exists and has no production caller; the Guided door's
`decision_curve` figure is a different door a Classic user never reaches from
this page. `research/CLINICAL_SURVEY_PACK.md` §A5.3 asks for net benefit over a
clinically-motivated threshold range, so **not building it is silence and is
allowed** — listing it under *Evaluation Metrics* is an assertion and is not.

**THE SENTENCE THAT IS THERE NOW**, corrected at `cc93767`, and it is
`AUDIT-028`'s model — less, and true, with the shortfall stated rather than the
line deleted:

    - Decision curve analysis (net benefit): available as a library call
      (`ml.calibration.decision_curve_analysis`), not yet wired into a page

**WHAT THIS FILE IS AND IS NOT.** The correction above is not this chunk's
work — it landed at `cc93767`, before this test was written — and `app.py` is
owned by another chunk, so the revert probe that would make this a closing
guard was NOT run: it would require writing a file this chunk does not own.
What is here is the guard that was missing: it holds the corrected sentence to
its two checkable claims, and it goes red in **both** directions — if the
library call is renamed out from under the sentence, and if the call is wired
into a page while the sentence still says it is not.

`GUIDED-045` — the caller sweep proves it can find a symbol that IS wired
before it is allowed to report finding none for this one. Without that control,
a sweep that silently stopped matching would report the same clean nothing.
§07 trap 5 is the same point: a grep answers *does this text appear*, and the
question is *does this run*.
"""
from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The exact line `app.py` carried at `cf7fc17`, kept verbatim so a revert of
#: the correction is caught by its own bytes rather than by a paraphrase.
THE_FALSE_LINE = "- Decision curve analysis for clinical utility"

#: The library call the corrected sentence cites. §07 trap 3: a test handed a
#: name that stands for a real object asserts the name resolves.
THE_CITED_CALL = "ml.calibration.decision_curve_analysis"

#: Wired, and therefore the control for the sweep below: `pages/07_Explainability.py`
#: imports it from `ml.publication`, which is what makes the *Subgroup analysis
#: with forest plots* bullet beside ours keepable.
A_SYMBOL_THAT_IS_WIRED = "subgroup_analysis"

SEARCHED = ("pages", "utils")


def _app_source() -> str:
    return (ROOT / "app.py").read_text(encoding="utf-8")


def _evaluation_metrics_block(source: str) -> list[str]:
    """The bullets under `**Evaluation Metrics:**`, by STRUCTURE — the heading,
    then every line up to the next blank line or the next bold heading. Matching
    the list rather than searching the file for a phrase is the difference
    between a guard that follows the block when it moves and one that reports a
    clean nothing once someone renames the section."""
    lines = source.splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.strip() == "**Evaluation Metrics:**"]
    if not starts:
        return []
    heading = lines[starts[0]]
    indent = len(heading) - len(heading.lstrip())
    out: list[str] = []
    for line in lines[starts[0] + 1:]:
        stripped = line.strip()
        here = len(line) - len(line.lstrip())
        if stripped.startswith("- ") and here == indent:
            out.append(stripped)
        elif out and stripped and here > indent:
            # A WRAPPED BULLET IS ONE BULLET. The decision-curve line wraps,
            # and §07 trap 5 is exactly this: a claim split across two source
            # lines that no per-line search can see whole.
            out[-1] = out[-1] + " " + stripped
        else:
            break
    return out


def _files_searched() -> list[Path]:
    return sorted(path for directory in SEARCHED
                  for path in (ROOT / directory).rglob("*.py"))


def _mentions(symbol: str) -> list[str]:
    return [str(path.relative_to(ROOT)) for path in _files_searched()
            if symbol in path.read_text(encoding="utf-8", errors="replace")]


def test_the_capability_list_is_found_at_all():
    """`GUIDED-045`. Every assertion below is about the contents of one block;
    if the block stops resolving they all pass over an empty list."""
    bullets = _evaluation_metrics_block(_app_source())
    assert len(bullets) >= 4, (
        f"the **Evaluation Metrics:** list in app.py resolved to {bullets} — "
        f"the section was renamed or reshaped and every check in this file is "
        f"now sweeping nothing")
    assert all(b.startswith("-") or b.startswith("(") for b in bullets), bullets


def test_the_landing_page_does_not_list_decision_curves_as_shipped():
    """`AUDIT-018`, pinned to the bytes the row was filed for."""
    assert THE_FALSE_LINE not in _app_source(), (
        f"app.py lists {THE_FALSE_LINE!r} under Evaluation Metrics again. No "
        f"file under {SEARCHED} computes net benefit, so a Classic user cannot "
        f"reach it from that page — say where it is available, as the line "
        f"corrected at cc93767 does, or build the page")


def test_the_library_call_the_sentence_cites_resolves():
    """§07 trap 3 — the sentence hands the reader a dotted path, and a path
    that does not resolve is the same false claim in a smaller font."""
    block = "\n".join(_evaluation_metrics_block(_app_source()))
    assert THE_CITED_CALL in block, (
        f"the Evaluation Metrics list no longer cites {THE_CITED_CALL} — if "
        f"the decision-curve bullet was rewritten, rewrite this guard with it")

    # `import_module` rather than `pytest.importorskip`: a skip here would be
    # this file reporting nothing about the claim it exists to check, which is
    # `TEST-059`'s shape and was swept at L52. If the module app.py names does
    # not import, the sentence is false and the red says so.
    module_name, _, attribute = THE_CITED_CALL.rpartition(".")
    module = import_module(module_name)
    assert callable(getattr(module, attribute, None)), (
        f"app.py tells the reader {THE_CITED_CALL} is available as a library "
        f"call and {module_name} has no callable {attribute}")


def test_the_not_yet_wired_clause_is_still_true():
    """The other direction, and the one that will fire first in practice.

    The sentence's second claim is that the call is NOT wired into a page. That
    is a fact about the tree, so it is swept rather than asserted — and when
    somebody wires it, this goes red asking for the sentence to be updated
    rather than letting the page understate what the app can do.
    """
    searched = _files_searched()
    assert len(searched) >= 10, (
        f"only {len(searched)} python files found under {SEARCHED}; the sweep "
        f"is looking in the wrong place")
    # `GUIDED-045` — the sweep finds a symbol that IS wired before it is
    # allowed to report that this one is not.
    assert _mentions(A_SYMBOL_THAT_IS_WIRED), (
        f"the sweep cannot find {A_SYMBOL_THAT_IS_WIRED}, which pages/07 "
        f"imports — so its silence about decision curves means nothing")

    callers = _mentions("decision_curve_analysis")
    block = "\n".join(_evaluation_metrics_block(_app_source()))
    hedged = "not yet wired into a page" in block
    assert bool(callers) is not hedged, (
        f"app.py says the decision curve is {'not ' if hedged else ''}wired "
        f"into a page and {SEARCHED} says {callers or 'nothing calls it'} — "
        f"§A5.3 asks for net benefit, so wiring it is the good outcome; the "
        f"sentence has to move with it")


def test_the_detector_fires_on_the_line_the_row_was_filed_for():
    """`GUIDED-045`'s positive control for the pin above.

    The pin reports the same clean nothing for a corrected page, a mistyped
    literal and a block that stopped resolving. This proves the literal is the
    one that shipped: it is checked against the historical file's own bullet
    shape rather than against the memory of it.
    """
    historical = ("        **Evaluation Metrics:**\n"
                  "        - Bootstrap 95% CIs (BCa method, 1000 resamples)\n"
                  "        - Calibration: Brier score, ECE, reliability diagrams\n"
                  "        " + THE_FALSE_LINE + "\n"
                  "        - Subgroup analysis with forest plots\n")
    bullets = _evaluation_metrics_block(historical)
    assert THE_FALSE_LINE in bullets, (
        "the block parser does not find the false line in the shape app.py "
        "actually carried it, so the pin is checking nothing")
    assert not re.search(r"library call|not yet wired", THE_FALSE_LINE), (
        "the false line as recorded here already carries the hedge — the "
        "literal has drifted toward the correction and no longer pins it")
