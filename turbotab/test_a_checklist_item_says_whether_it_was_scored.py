"""L64-B. `GUIDED-238` — 26 of 27 items PASS and most could not have failed.

Driven on `clinical_risk.csv` fitted with logreg/lda/rf: the bundle admits five
figures and serves a wall of green. A reader takes that as *this figure meets
twenty-six publication requirements*. For some of them the app never checked
anything — the producer pins the value the predicate reads — so the badge
certifies a check nobody performed. That is the governing rule's
assert-something-false branch on the surface whose whole job is to certify.

## The partition in the row does not reproduce, and that is this part's result

`GUIDED-238` says **43 / 42 / 16 of 85**; the prompt re-derived **52 / 29 / 5
of 86**. The count is 86 — confirmed twice, by an `ast` walk of
`figure_specs.py` and by a runtime walk of `figures.REGISTRY`. The PARTITION
was measured three more ways at `L64` and came back three more answers:

| instrument | question it answers | result |
|---|---|---|
| mutation probe, 9 sentinels × every top-level key, all 17 specs | can ANY payload falsify it? | **85 of 86 can** |
| static walk: does the item read only keys that appear as literals? | is a key it reads pinned somewhere? | 19 pinned / 42 computed / 22 mixed / 3 unresolved |
| empty-payload probe | does it pass when handed nothing at all? | **5 of 86** |

Three instruments, three answers, none of them 52. **The three-way partition is
not a measurement — it is a human judgment about what a producer "can" emit,
and it is not reproducible.** So this file does not assert one. It asserts the
two properties that ARE mechanical, and both can fail.

The row's `act` is separately falsified and that is recorded rather than
worked around: it asks for scored and declared items to be separated *"so a
constant-reading item cannot contribute to a compliance count"*, and **there is
no compliance count over figure checklist items.** The rendered aggregate the
row imagines exists one registry over, on `ml/manuscript_validator.py`, and is
audited under `MISC-029`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import eventfixture                                # noqa: E402
from turbotab import figure_bundle as FB                         # noqa: E402
from turbotab import figures                                     # noqa: E402
from turbotab import figure_specs as FS                          # noqa: E402
from turbotab.project import AnalysisProject                     # noqa: E402
from turbotab import training as T                               # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")

#: Items that pass when handed `{}`. Measured, not chosen — see
#: `test_an_item_that_passes_on_an_empty_payload_is_named`. Each is here
#: because a guard clause is vacuously satisfied by ABSENCE, which is a
#: sharper definition of vacuous than "no sentinel flipped it": it needs no
#: static analysis, it covers all 86, and it costs one line.
PASSES_ON_NOTHING = {
    # `not p.get("labeled_features")` — and this one is now DECLARED, which is
    # the fix rather than the symptom.
    "volcano.no_label_without_an_msi_level",
    # `all(...)` over an empty dict of curves.
    "roc.c_statistic_with_interval",
    # `not p.get("ratio_measure")` — a linear axis needs no log axis.
    "forest.log_axis_for_ratios",
    # `n_models_with_coefficients <= 1` — ADDED AT L63, by me, and it lands on
    # this list. One candidate needs no disambiguation, so the guard clause is
    # correct and is vacuous on nothing.
    "forest.names_the_model_it_is_about",
    # `all(...)` over an empty list of panels.
    "item_panel.n_per_panel",
}


def _items():
    for spec_id, spec in figures.REGISTRY.items():
        for item in spec.checklist:
            yield spec_id, item


def test_the_population_is_eighty_seven_items_across_seventeen_specs(capsys):
    """The count every number below is a fraction of, re-derived here.

    `GUIDED-238` says 85. It was **86** when this file was written, and it is
    **87** at the end of the same loop: `L64-D` added
    `decision_curve.models_accounted_for` for `GUIDED-247`. That is recorded
    rather than absorbed, because this assertion caught its own loop's addition
    and a number that moves silently is the decay this file exists to stop.

    An equality rather than a floor, deliberately: a new checklist item is a
    new claim the app makes to a reader, and it should arrive here.
    """
    specs = figures.REGISTRY
    items = list(_items())
    assert len(specs) == 17, sorted(specs)
    assert len(items) == 87, len(items)
    with capsys.disabled():
        print(f"\n  {len(items)} items across {len(specs)} specs")


def test_an_item_that_passes_on_an_empty_payload_is_named(capsys):
    """The cheap probe, and it is the one definition that reproduces.

    *Passes when handed no data whatsoever* needs no static analysis, covers
    all 86, and costs one line. An item on this list is not automatically
    wrong — four of the five are guard clauses that are correctly satisfied by
    absence — but a NEW one is a claim nobody asked for, and it arrives here
    rather than in a wall of green.
    """
    vacuous = set()
    for spec_id, item in _items():
        try:
            if bool(item.check({})):
                vacuous.add(f"{spec_id}.{item.id}")
        except Exception:
            pass

    # THE CONTROL: an empty result would look identical to a clean sweep, and
    # `AGENT_ONBOARD.md` §07 trap 5c is about exactly this assertion shape.
    assert vacuous, (
        "no item passes on an empty payload, which would be a real improvement "
        "and is not what was measured — check the enumeration before believing "
        "it")
    assert vacuous == PASSES_ON_NOTHING, (
        f"the set of items that pass on nothing has moved.\n"
        f"  new: {sorted(vacuous - PASSES_ON_NOTHING)}\n"
        f"  gone: {sorted(PASSES_ON_NOTHING - vacuous)}\n"
        f"A new one is a checklist item that certifies a figure it was never "
        f"shown. If it is a correct guard clause, add it here with the reason.")
    with capsys.disabled():
        print(f"\n  {len(vacuous)} of 86 pass on an empty payload")


def test_every_declarable_item_states_why_it_cannot_be_scored():
    """`scored_when` with no reason is the claim without the evidence.

    Enforced in `ChecklistItem.__post_init__` so it cannot be forgotten; this
    asserts the enforcement is live rather than trusting the constructor.
    """
    declarable = [(s, i) for s, i in _items() if i.scored_when is not None]
    assert declarable, "nothing declares itself unscorable; this probe is inert"
    for spec_id, item in declarable:
        assert item.declared_because.strip(), f"{spec_id}.{item.id}"
        assert len(item.declared_because) > 40, (
            f"{spec_id}.{item.id}'s reason is too short to be an argument")

    with pytest.raises(figures.FigureError):
        figures.ChecklistItem("x", "x", "x", lambda p: True,
                              scored_when=lambda p: False)


def test_a_declared_item_is_reported_as_declared_and_not_as_a_pass():
    """The badge, which is the whole deliverable.

    A rule the app repeated without checking must not render in the same ink as
    a check it performed.
    """
    spec = figures.REGISTRY["volcano"]
    item = next(i for i in spec.checklist
                if i.id == "no_label_without_an_msi_level")
    # Unlabeled — nothing to score. The predicate still passes; the BADGE is
    # what changes.
    scored = spec.score({"labeled_features": [],
                         "labels_require_msi_level": True})
    row = next(r for r in scored if r["id"] == item.id)
    assert row["passed"] is True
    assert row["scored"] is False
    assert row["declared_because"], "declared with no reason reaches the page"

    # And the moment a label exists it becomes live, because `scored_when` is a
    # predicate on the payload rather than a flag on the item.
    live = spec.score({"labeled_features": ["mz_001"],
                       "labels_require_msi_level": False})
    row = next(r for r in live if r["id"] == item.id)
    assert row["scored"] is True
    assert row["passed"] is False, (
        "a volcano that labels a compound without an MSI level is exactly what "
        "this item is for, and it passed")
    assert row["declared_because"] == ""


def test_both_producers_of_passed_carry_the_third_state():
    """`figures.py:194` is the exception branch and it is a SECOND producer.

    A third state expressed at only one of them serves a raising item in the
    old shape — a `passed: False` with no `scored` key at all, which the page
    would render as a plain FAIL.
    """
    def explodes(_payload):
        raise RuntimeError("boom")

    spec = figures.FigureSpec(
        id="probe_spec", title="probe",
        when_applicable=lambda s: True, layers=(),
        annotations=(),
        checklist=(figures.ChecklistItem(
            "raises", "an item whose predicate raises", "because",
            explodes, scored_when=lambda p: False,
            declared_because="a stand-in built to reach the exception branch "
                             "of FigureSpec.score, which is the second "
                             "producer of the passed flag"),),
        caption=lambda p: "", tier=figures.EXPLORATORY,
        evidence={"status": "SETTLED", "source": "probe"})

    row = spec.score({})[0]
    assert row["passed"] is False
    assert "scored" in row, (
        "the exception branch emits no `scored` key, so a raising item is "
        "served in the old two-state shape")
    assert row["scored"] is False
    assert row["declared_because"]


# ═══════════ the three scope_stated items — a matcher firing on prose ═══════

def _resampled():
    """A real instability payload, so the scope claim is about a real draw."""
    from turbotab import instability

    df = pd.read_csv(os.path.join(DATA, "clinical_risk.csv"))
    df = df[df["readmit_30d"].notna()].copy()
    project = AnalysisProject.from_dataframe(df, "clinical_risk.csv")
    project.target, project.task_type = "readmit_30d", "classification"
    project.set_grain("not_sure")
    project.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(project.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.25))]
    project.seal_lockbox(labels, fraction=len(labels) / len(project.df))
    eventfixture.choose_event(project, required=True)
    project.training_run = T.train(project, ["logreg"])
    return project, instability.run(project, "logreg", b=6, seed=42)


def test_the_scope_item_scores_the_scope_and_not_a_substring(capsys):
    """`GUIDED-238`'s sharpest instance, and no accounting fix reaches it.

    The predicate was `"held-out" in str(p.get("scored_on", ""))` and the only
    live `scored_on` is a sentence reading *"training rows with an outcome (the
    held-out rows are **not** resampled and not predicted…)"*. **The item passed
    because the sentence says the opposite of what the item claims.** Measured
    at HEAD, the old predicate's truth table:

    | `scored_on` | old check |
    |---|---|
    | the live sentence | True |
    | "the held-out rows were resampled and predicted" | **True** |
    | "training rows with an outcome" (honest) | **False** |
    | "all rows including the sealed lockbox" (the danger) | **False** |

    It passed the lie and failed two honest sentences. The quantity the item is
    about does not exist in prose, so the producer now measures it.
    """
    project, payload = _resampled()
    item = next(i for i in figures.REGISTRY["prediction_instability"].checklist
                if i.id == "scope_stated")

    assert payload["held_out_rows_resampled"] == 0
    assert payload["rows_resampled_from"] == len(project.analysis_rows)
    assert item.check(payload) is True

    # THE SENTENCE THE OLD PREDICATE PASSED. It is the condition the item's own
    # `because` says it exists to catch, and it must now fail.
    lying = dict(payload, scored_on="the held-out rows were resampled and "
                                    "predicted")
    assert item.check(lying) is True, (
        "the sentence is still stated, so the item is about the DRAW rather "
        "than about the wording — this asserts the predicate stopped reading "
        "prose at all")

    # And the state that matters: a draw that touched a sealed row.
    breached = dict(payload, held_out_rows_resampled=1)
    assert item.check(breached) is False, (
        "a resample that drew a sealed row leaves this item green, which is "
        "the seal dissolving under a figure that looks identical")
    assert item.check(dict(payload, scored_on="")) is False
    with capsys.disabled():
        print(f"\n  resampled {payload['rows_resampled_from']} rows, "
              f"{payload['held_out_rows_resampled']} of them sealed")


def test_the_counter_would_notice_a_sealed_row_in_a_draw():
    """The measurement's own control.

    `held_out_rows_resampled == 0` proves nothing unless the counter can reach
    a non-zero. This drives it by sealing rows the resampler is given.
    """
    from turbotab import instability

    project, _ = _resampled()
    # Hand the resampler the WHOLE table, sealed rows included — the state the
    # item exists to catch, produced rather than asserted.
    original = AnalysisProject.analysis_rows
    try:
        AnalysisProject.analysis_rows = property(
            lambda self: self.df[self.df[str(self.target)].notna()])
        payload = instability.run(project, "logreg", b=4, seed=7)
    finally:
        AnalysisProject.analysis_rows = original

    assert payload["held_out_rows_resampled"] > 0, (
        "the resampler drew from the whole table and the counter still says "
        "no sealed row was touched, so the counter is not counting")
    item = next(i for i in figures.REGISTRY["prediction_instability"].checklist
                if i.id == "scope_stated")
    assert item.check(payload) is False
