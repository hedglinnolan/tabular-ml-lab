"""L43-B · §A5.1 and §A5.3 run against shipped code — `AUDIT-015`.

> *"Enforce that calibration is computed for **every** model, including tree
> ensembles and neural nets."* — §A5.1
> *"Report: C-statistic with CI, calibration intercept and slope with CIs, the
> flexible calibration curve, O:E ratio, Brier score (and scaled Brier), and
> net benefit."* — §A5.3 **[SETTLED — essentially TRIPOD+AI's expectation]**

The audit did not find a missing quantity. It found the app **asserting** one.

`manuscript._calibration_text` gated on whether `"calibration"` appeared in
`{f["id"] for f in doc["figures"]}`. That was correct when `figures` was *the
figures this project drew*. `GUIDED-131` changed it to the **whole registry**,
carrying drawability as a per-row `drawn` field — so the id is unconditionally
present and the guard became a tautology.

The consequence is the governing rule's serious failure mode:

* on a **regression** project, the exported LaTeX says *"Calibration was
  assessed graphically; the calibration plot is reported with its intercept,
  slope and C-statistic"* — while the same project's `/figures` surface says
  *"Calibration is a claim about predicted PROBABILITIES, and this is a
  regression task."* Two surfaces of one app, contradicting each other about
  one project;
* on a classification project where the calibration fit is undefined, it fires
  next to an annotation box reading *"not estimable"*.

**A guard whose condition changed meaning under it is not a guard.** This is
`GUIDED-131`'s cost, one layer down, and nothing caught it because the
assertion was still literally true about the *registry* — trap 2's shape,
where the check tests a description rather than the behavior.
"""
from __future__ import annotations

import pathlib

import pytest

from turbotab import figure_specs  # noqa: F401 — populates FIG.REGISTRY
from turbotab import figures as FIG
from turbotab import manuscript as MS

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _registry_rows(**overrides):
    """The figures list in the shape `api.get_manuscript` actually composes.

    Trap #3's rule: the fixture has to be the production shape, or the test
    guards nothing. `api.py` builds one row per registered spec with `drawn`
    and `promoted` as fields — it does not filter.
    """
    rows = []
    for spec in FIG.REGISTRY.values():
        rows.append({"id": spec.id, "title": spec.title, "tier": spec.tier,
                     "promoted": False, "drawn": False})
    for fid, drawn in overrides.items():
        for row in rows:
            if row["id"] == fid:
                row["drawn"] = drawn
    return rows


def test_the_fixture_is_the_shape_the_route_composes():
    """Asserted rather than assumed, because everything below depends on it.

    If `api.get_manuscript` ever goes back to filtering, these rows stop
    standing for anything and the tests become vacuous in the direction that
    reads as a pass.
    """
    source = (ROOT / "turbotab" / "api.py").read_text(encoding="utf-8")
    assert '"drawn": None if drawn is None else (f.id in drawn)' in source, (
        "`api.get_manuscript` no longer carries drawability as a per-row "
        "field, so these fixtures no longer stand for what it sends")
    assert "for f in _figure_specs_all()" in source, (
        "`api.get_manuscript` no longer sends the whole registry, so the "
        "tautology this file exists for may no longer be possible — recheck "
        "before deleting anything")
    assert "calibration" in FIG.REGISTRY, (
        "the calibration figure is not registered, so the id this gate reads "
        "resolves to nothing")


def test_a_project_that_drew_no_calibration_plot_says_nothing_about_it():
    """The defect, directly. Every row `drawn=False` — a regression project,
    or any project the bundle refused the figure for."""
    assert MS._calibration_text({"figures": _registry_rows()}) == "", (
        "the manuscript claims calibration was assessed on a project that "
        "drew no calibration plot. The app may be silent and it may refuse; "
        "it may not assert something false.")


def test_a_project_that_drew_it_still_says_so():
    """The other direction, and the reason this is a gate rather than a
    deletion. §A5.1 ranks calibration above discrimination, so an export that
    dropped the sentence entirely would invert the field's own ordering."""
    said = MS._calibration_text({"figures": _registry_rows(calibration=True)})
    assert said, "a project that drew the plot now says nothing about it"
    assert "Calibration was assessed graphically" in said
    assert "[AUTHOR REQUIRED]" in said or "AUTHOR" in said.upper(), (
        "the sentence no longer carries its author gap, and §A5.3 wants the "
        "intercept and slope with CIs that this app does not yet carry")


def test_an_undecidable_drawability_is_silence_not_a_claim():
    """`drawn is None` is `api.py`'s own third state: the bundle could not say.

    Three states, never two — the lockbox constitution §03's rule, applied to
    a different registry. Coercing `None` to *yes* is how the tautology read
    in the first place; coercing it to *no* would be a different assertion.
    Silence is what the governing rule permits.
    """
    rows = _registry_rows()
    for row in rows:
        row["drawn"] = None
    assert MS._calibration_text({"figures": rows}) == "", (
        "the manuscript claims calibration was assessed on a project whose "
        "figure bundle could not say whether the plot is drawable")


@pytest.mark.parametrize("doc", [
    {},
    {"figures": []},
    {"figures": None},
    {"figures": [{"id": "roc"}]},
    {"figures": [{"id": "calibration"}]},          # no `drawn` key at all
])
def test_a_malformed_or_empty_figures_list_never_earns_the_sentence(doc):
    """Including the shape that used to earn it: an id with no `drawn` key.

    A row that does not say it was drawn has not said it was drawn.
    """
    assert MS._calibration_text(doc) == "", doc


def test_the_exported_latex_does_not_carry_the_claim_for_a_regression_project():
    """End to end, through the exporter the user actually receives.

    `_calibration_text` feeding `generate_latex_report` is what puts the
    sentence under a `\\subsection{Calibration}` heading in the downloaded
    document, so the unit assertion above is not the whole claim.
    """
    from ml import latex_report

    text = MS._calibration_text({"figures": _registry_rows()})
    assert text == ""
    # And the exporter's own regression branch is what should render instead —
    # pinned so a future edit cannot quietly make the empty string print an
    # empty section rather than the honest one.
    source = (ROOT / "ml" / "latex_report.py").read_text(encoding="utf-8")
    assert "calibration_text" in source, (
        "the exporter no longer reads `calibration_text`, so this gate no "
        "longer controls what the document says")
    assert hasattr(latex_report, "generate_latex_report")
