"""L46-C — `GUIDED-153`. The deferral that named no step, driven to the step.

Every pack card's defer button read *"Decide later"* rather than naming one, and
pressing it recorded a target **the API's fallback chose rather than the
Router** — which `ml.router.defer_destination`'s own docstring forbids in those
words: *a deferral whose target is chosen by the renderer is a deferral the
record cannot honor.*

## Three layers, and only the first is the one the row named

1. **No destination was attached.** `engine.rank_findings` applied
   `_with_deferral` to the structural and profile streams and appended
   `packs.findings(...)` raw, so every pack finding shipped `defer_target: null`.
2. **The API filled the hole.** `api.py:1193` defaults a missing `target_step` to
   `explore`, so the record said *"deferred to the step where it belongs"* while
   nothing had decided where it belongs — and `explore` is the step the user is
   standing on.
3. **And it could not have come back anyway.** `ml.router.plan` re-presents a
   deferred finding as a repair question and `_is_repairable` admits only
   findings carrying a `fix_kind`. A pack finding carries `none` **by design** —
   §A1.1's rule is *detect, propose, require explicit confirmation*, and
   `test_no_detector_offers_a_repair` asserts none of the eighteen proposes one.
   So the deferral was recorded, the coach ledger listed it, and the step it named
   showed nothing.

Layer 3 is the one `PRODUCT_VISION.md` §04 is about: *deferred items resurface,
pre-checked and attributed, at the step they target. That closes the loop between
what the app noticed and what the user decided, which is the whole point.* The
loop was open for the entire pack stream.

## What "which step" is decided by, and why it is a table rather than a default

`PACK_DEFER` is read off each detector's own `why_it_matters`, not assigned by
taste. `mixed_units` needs the units declared per analyte — a repair to the
table, so `explore`. `default_value_mass` needs a spike at a default treated as
missing — a transform fitted in the pipeline, so `preprocess`. They are not one
kind of thing, which is why one default for the stream was always going to be
wrong for most of it.

**An unmapped detector raises rather than defaulting**, and
`test_every_detector_declares_where_it_comes_back` is what makes that a check
rather than a hope: a detector added next loop with no entry turns this file red.

## `GUIDED-097` — two lenses

Clinical and dietary, and they are chosen because they land in **different**
steps: clinical's censored values come back at Preprocess, dietary's energy
adjustment at Features. A pair that both landed at the same step would verify one
row of the table twice.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ml import router
from turbotab import attention as A
from turbotab import packs as P

DATA = Path(__file__).resolve().parent / "sample_data"

#: `(fixture, lens, target, finding id, the step it must come back at)`.
#: Different steps on purpose — see the module docstring.
CASES = {
    "clinical to Preprocess": ("clinical_labs.csv", "clinical", "readmitted",
                              "pack::clinical::censored_values", "preprocess"),
    "dietary to Features": ("nhanes_kilojoules.csv", "dietary", "DR1TKCAL",
                           "pack::dietary::energy_adjustment", "features"),
}

#: NOT COVERED, said out loud.
SHAPES_NOT_COVERED = (
    "The `train` destinations. `dietary::survey_weights`, `partial_design` and "
    "`lonely_psu` route to Train, the page has a `back-train` slot, and no "
    "drive here reaches the Train step — it needs a sealed lockbox and a fitted "
    "model. The slot is asserted to exist; that it renders there is not.",
    "Un-deferring. There is no inverse decision for a deferral, so nothing "
    "checks that a finding leaves the came-back block.",
    "The other three packs' routing rows are asserted as a table below and not "
    "driven to a step.",
)


def _detector_ids():
    """Every finding id any registered detector can produce, from the fixtures.

    Derived rather than listed: a hand-written list is how a detector added next
    loop goes unrouted, which is this file's own subject one level up.
    """
    seen = set()
    for fixture in sorted(DATA.glob("*.csv")):
        try:
            df = pd.read_csv(fixture)
        except Exception:
            continue
        for lens in P.PACKS:
            try:
                for finding in P.findings(df, [lens]):
                    seen.add(finding["id"])
            except Exception:
                continue
    return sorted(seen)


def test_every_detector_declares_where_it_comes_back():
    """No pack finding is routed by a default, because a default *was* the bug.

    Also the other direction: a row in `PACK_DEFER` for a detector that no longer
    exists is a stale entry nobody will notice, so the table is checked against
    what the packs actually produce.
    """
    observed = _detector_ids()
    assert observed, "no detector produced anything on any fixture"
    unrouted = []
    for fid in observed:
        try:
            step, label = router.defer_destination(
                {"id": fid, "source": "pack", "severity": "warning"})
        except router.UnroutedFinding:
            unrouted.append(fid)
            continue
        assert step in router.STEP_LABELS, (fid, step)
        assert label == router.STEP_LABELS[step], (fid, label)
    assert not unrouted, (
        f"these detectors have no declared destination and would have been "
        f"routed by a default, which is `GUIDED-153` returning: {unrouted}")

    short = {"::".join(f.split("::")[1:]) for f in observed}
    stale = sorted(set(router.PACK_DEFER) - short)
    assert not stale, (
        f"`PACK_DEFER` routes detectors that no fixture produces: {stale}. "
        f"Either a detector was removed or a fixture stopped triggering it; "
        f"either way the row is a claim with nothing behind it.")


def test_a_finding_that_was_never_deferred_comes_back_nowhere():
    """The positive control's negative half.

    Without it the claim above is satisfied by a block that renders every
    finding, which is `LOOP.md` trap #3's shape: the fixture supplying what
    production could not.
    """
    findings = [{"id": "a", "source": "pack", "pack": "clinical",
                 "severity": "warning", "defer_target": "preprocess",
                 "defer_target_label": "Preprocess", "title": "a"}]
    assert A.deferred_noticings(findings, []) == {}
    assert A.deferred_noticings(findings, [{"kind": "dismiss", "subject": "a"}]) == {}
    got = A.deferred_noticings(findings, [{"kind": "defer", "subject": "a"}])
    assert list(got) == ["preprocess"] and got["preprocess"][0]["id"] == "a"


def test_a_deferral_with_no_destination_is_reported_rather_than_filed():
    """The failure mode `GUIDED-153` was, if it ever returns.

    A finding deferred with no `defer_target` is not quietly dropped and not
    quietly filed under a step nobody chose — it lands in `__unrouted__`, which
    the drive above asserts is empty. Silence here would be the app choosing a
    step in the user's name.
    """
    findings = [{"id": "a", "source": "pack", "severity": "warning", "title": "a"}]
    got = A.deferred_noticings(findings, [{"kind": "defer", "subject": "a"}])
    assert list(got) == ["__unrouted__"]
    assert "nothing recorded where it comes back" in got["__unrouted__"][0]["why"]


def test_the_probe_reports_its_own_coverage(capsys):
    observed = _detector_ids()
    by_step = {}
    for fid in observed:
        step, _ = router.defer_destination(
            {"id": fid, "source": "pack", "severity": "warning"})
        by_step.setdefault(step, []).append(fid)
    with capsys.disabled():
        print("\n  ── L46-C · GUIDED-153, where a pack finding comes back ──")
        print(f"  detectors observed on fixtures {len(observed)}")
        print(f"  routed by PACK_DEFER           {len(observed)}   <- all of them")
        for step in sorted(by_step):
            print(f"      {step:<12} {len(by_step[step])}")
        print(f"  driven to the step, end to end {len(CASES)} "
              f"({', '.join(sorted(CASES))})")
        print(f"  shapes NOT covered             {len(SHAPES_NOT_COVERED)}")
        for shape in SHAPES_NOT_COVERED:
            print(f"      · {shape}")
    assert len(observed) == sum(len(v) for v in by_step.values())
