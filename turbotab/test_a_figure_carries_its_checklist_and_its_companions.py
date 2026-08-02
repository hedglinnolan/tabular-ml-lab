"""`GUIDED-051` — the figure spec, and the two figures that had to survive it.

`DOMAIN_SCIENCE.md` §02. The app has seven geometries and nineteen EDA actions
and none of them knows what field it is looking at. The research says that is
the wrong axis:

> Every pack specified its signature figures as a **checklist**, and the
> checklist items are overwhelmingly about **annotation rather than geometry**.

So the figure layer is a caption-and-annotation engine wrapped around a plotting
library, not the other way round — and the spec has five fields, of which
`companions` has no analogue in the app today and is the load-bearing one.

## Two figures, deliberately

Calibration and PCA scores were chosen to be maximally different: confirmatory
against exploratory, needs-a-fitted-model against needs-a-numeric-block,
do-not-truncate against aspect-proportional-to-variance, has-a-companion against
makes-no-claim, not-promotable against promotable. A third is not built until
the spec has survived both.

## What the seams turned out to be

**The checklist found a real gap on its first run.** `annotation_box` requires
the calibration intercept and slope, and `ml/calibration.py` computed neither —
it had the hierarchy's first and third rungs and not the second, which the
clinical pack calls mandatory and which is the single most useful number on the
figure. The item failed against a real render, which is exactly what a checklist
scored against a render is for.

**A checklist item can be about what is NOT done.** *"Do not truncate the axis"*
is scored by comparing the drawn range against the observed one, so the payload
has to carry both. An earlier draft carried only what it drew, and the item was
unscoreable — `GUIDED-045`'s axis one layer into the figure layer.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import cohort_findings as CF                            # noqa: E402
from turbotab import figure_specs as FS                               # noqa: E402
from turbotab import figures as FIG                                   # noqa: E402


def _calibrated(n=600, extreme=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    truth = 1.0 / (1.0 + np.exp(-x))
    y = rng.binomial(1, truth)
    p = 1.0 / (1.0 + np.exp(-extreme * x))
    return y, p


def _assay(seed=1, n=60, p=25):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.lognormal(0, 1, (n, p)),
                      columns=[f"mz_{i:03d}" for i in range(p)])
    df["group"] = ["case"] * (n // 2) + ["control"] * (n - n // 2)
    return df, [i % 10 == 0 for i in range(n)]


# ── the spec ────────────────────────────────────────────────────────────────

def test_a_confirmatory_figure_without_its_companion_is_not_admitted():
    """**The rule with no analogue in the app today**, and the one that kills
    the circular-figure family.

    Not a warning beside the figure, not a caption caveat — both are things a
    reader skips, and the circular-figure family survives precisely by being
    skippable. It is admissibility: the bundle does not contain it.
    """
    y, p = _calibrated()
    payload = FS.calibration_payload(y, p)
    alone = FIG.bundle({"calibration": payload})
    assert alone["n_admitted"] == 0 and alone["n_held"] == 1
    held = alone["held"][0]
    assert held["missing_companions"] == ["roc"]
    assert "not in this bundle" in held["why_held"]

    # **AND THE COMPANION IS A FIGURE THAT EXISTS** (`GUIDED-128`). Until L40
    # this line paired calibration with `{"discrimination": {}}` — an id that
    # was never registered — so the test proved the bundle honors an id string
    # and could not notice that no project would ever supply one. The ROC
    # curve is the discrimination figure §A4.4 specifies, and it exists now.
    with_roc = FIG.bundle({"calibration": payload,
                           "roc": FS.roc_payload(y, {"m": p})})
    assert {r["id"] for r in with_roc["admitted"]} == {"calibration"}
    # AND THE ROC IS ITSELF HELD, because §A4.4 ranks it below BOTH calibration
    # and the decision curve and it declares both. The rule composes: adding
    # one figure to satisfy another's companion does not exempt the new one.
    assert {r["id"] for r in with_roc["held"]} == {"roc"}

    y_flag = np.asarray(y, dtype=float)
    whole = FIG.bundle({
        "calibration": payload,
        "roc": FS.roc_payload(y, {"m": p}),
        "decision_curve": FS.decision_curve_payload(y_flag, {"m": np.asarray(p)}),
    })
    assert whole["n_admitted"] == 3 and whole["n_held"] == 0


def test_an_exploratory_figure_needs_no_companion_and_may_not_declare_one():
    """Companions exist because a confirmatory claim needs its validation
    beside it. Requiring one of a figure that makes no claim is ceremony."""
    assert FS.PCA_SCORES.companions == ()
    ok, missing = FS.PCA_SCORES.admissible([])
    assert ok and not missing
    with pytest.raises(FIG.FigureError, match="no companions"):
        FIG.FigureSpec(
            id="x", title="x", tier=FIG.EXPLORATORY,
            when_applicable=lambda s: True, layers=(), annotations=(),
            checklist=(), caption=lambda p: "", companions=("y",),
            evidence=FS.PCA_EVIDENCE)


def test_every_figure_states_where_the_field_stands():
    """A checklist is a set of claims about what a reviewer expects, and an
    unbadged claim is the uniform confidence §01.1 exists to end."""
    assert FIG.REGISTRY, "no figures registered"
    for spec in FIG.REGISTRY.values():
        assert spec.evidence is not None, spec.id
        assert spec.evidence.source.startswith("research/"), spec.id
        served = spec.to_dict()
        assert served["evidence_status"] and served["source"]


def test_a_figure_may_not_be_registered_twice():
    with pytest.raises(FIG.FigureError, match="already registered"):
        FIG.register(FS.CALIBRATION)


# ── the calibration plot ────────────────────────────────────────────────────

def test_the_calibration_checklist_passes_on_a_real_render():
    y, p = _calibrated()
    scored = FS.CALIBRATION.score(FS.calibration_payload(y, p))
    failed = [r for r in scored if not r["passed"]]
    assert not failed, [(r["id"], r["because"]) for r in failed]
    assert len(scored) == 5


def test_the_risk_distribution_is_present_and_split_by_outcome():
    """The item the research singles out: *without it the reader cannot tell
    whether the curve's behavior at 0.8 rests on 3 patients or 300.*"""
    y, p = _calibrated()
    dist = FS.calibration_payload(y, p)["risk_distribution"]
    assert sum(dist["events"]) == int((np.asarray(y) == 1).sum())
    assert sum(dist["non_events"]) == int((np.asarray(y) == 0).sum())
    assert len(dist["edges"]) == len(dist["events"]) + 1


def test_the_truncation_item_can_actually_fail():
    """`GUIDED-045`'s axis, checked on a checklist item.

    An item scored against a payload that only carries what it drew has a pass
    set as wide as "the figure exists". So the payload carries both ranges, and
    this asserts the item goes red when the drawn range hides observed data.
    """
    y, p = _calibrated()
    payload = FS.calibration_payload(y, p)
    item = next(i for i in FS.CALIBRATION.checklist if i.id == "no_truncation")
    assert item.check(payload) is True
    truncated = dict(payload, x_range_drawn=[0.2, 0.8],
                     x_range_observed=[0.01, 0.99])
    assert item.check(truncated) is False


def test_the_annotation_box_names_the_six_numbers_a_reviewer_wants():
    y, p = _calibrated()
    payload = FS.calibration_payload(y, p)
    for key in ("calibration_intercept", "calibration_slope", "c_statistic",
                "e_avg", "e_max", "n", "events"):
        assert payload.get(key) is not None, key
    caption = FS.CALIBRATION.caption(payload)
    assert "not truncated" in caption and "events" in caption


def test_the_weak_calibration_numbers_are_the_engines_and_are_right():
    """The gap the checklist found, closed in the engine rather than here.

    Perfect calibration is intercept 0 and slope 1; predictions twice too
    extreme give a slope near 0.5, which is the reading that makes the number
    worth printing.
    """
    from ml.calibration import c_statistic, weak_calibration

    y, p = _calibrated(n=4000)
    intercept, slope = weak_calibration(y, p)
    assert abs(intercept) < 0.15 and abs(slope - 1.0) < 0.15

    _, too_extreme = _calibrated(n=4000, extreme=2.0)
    _, slope2 = weak_calibration(y, too_extreme)
    assert 0.35 < slope2 < 0.65, slope2

    assert 0.5 < c_statistic(y, p) < 1.0


def test_an_undefined_fit_reports_nothing_rather_than_perfection():
    """`(0.0, 1.0)` are the values of PERFECT calibration. Returning them for
    'could not compute' would be the app reporting an ideal result where it has
    none — the governing rule's own failure, in two floats."""
    from ml.calibration import c_statistic, weak_calibration

    assert weak_calibration(np.ones(20), np.full(20, 0.5)) == (None, None)
    assert weak_calibration(np.r_[np.ones(10), np.zeros(10)],
                            np.full(20, 0.3)) == (None, None)
    assert c_statistic(np.ones(10), np.linspace(0, 1, 10)) is None


# ── the PCA scores plot ─────────────────────────────────────────────────────

def test_the_pca_checklist_passes_on_a_real_render():
    df, qc = _assay()
    payload = FS.pca_scores_payload(df, group_col="group", qc_mask=qc,
                                    scaling="Pareto")
    failed = [r for r in FS.PCA_SCORES.score(payload) if not r["passed"]]
    assert not failed, [(r["id"], r["because"]) for r in failed]


def test_the_axis_labels_carry_the_percent_variance():
    """*Omitting these is the single most common defect and reviewers ask for
    it*, so it is built into the payload rather than left to a renderer."""
    df, qc = _assay()
    payload = FS.pca_scores_payload(df, group_col="group", qc_mask=qc)
    assert len(payload["axis_labels"]) == 2
    for i, label in enumerate(payload["axis_labels"]):
        assert label.startswith(f"PC{i + 1} (") and label.endswith("%)")


def test_the_aspect_ratio_is_proportional_to_variance_explained():
    """Stretching PC2 to fill the panel visually exaggerates separation, which
    is the claim this figure is most often misused to make."""
    df, qc = _assay()
    payload = FS.pca_scores_payload(df, group_col="group", qc_mask=qc)
    r = payload["explained_variance_ratio"]
    assert payload["aspect_ratio"] == pytest.approx(r[1] / r[0])
    assert payload["aspect_ratio"] <= 1.0


def test_the_two_ellipses_are_different_objects():
    """*Papers routinely mislabel one as the other.* The T² ellipse is a single
    outlier boundary over all samples; group ellipses describe where each group
    lies. Rendered differently and labeled explicitly."""
    df, qc = _assay()
    payload = FS.pca_scores_payload(df, group_col="group", qc_mask=qc)
    t2, groups = payload["hotelling_t2"], payload["group_ellipses"]
    assert t2["single"] is True and "T²" in t2["label"]
    assert t2["style"] != groups["style"]
    assert t2["kind"] == "outlier_boundary" and groups["kind"] == "group_confidence"
    caption = FS.PCA_SCORES.caption(payload)
    assert "is not a group confidence region" in caption


def test_the_qcs_are_overlaid_and_never_dropped():
    """*Their tight central cluster IS part of the result.*"""
    df, qc = _assay()
    payload = FS.pca_scores_payload(df, group_col="group", qc_mask=qc)
    assert payload["n_qc"] == sum(qc) > 0
    assert len(payload["qc"]) == len(df)
    item = next(i for i in FS.PCA_SCORES.checklist if i.id == "qc_overlaid")
    assert item.check(payload) is True
    assert item.check(dict(payload, n_qc=0)) is False


def test_the_caption_says_pca_does_not_test_a_group_difference():
    """The pack's coaching, in the artifact that travels: *separation in a PCA
    is not a result.*"""
    df, qc = _assay()
    caption = FS.PCA_SCORES.caption(
        FS.pca_scores_payload(df, group_col="group", qc_mask=qc))
    assert "does not use the group labels" in caption
    assert "not a test of a group difference" in caption


# ── Part C · promotability ──────────────────────────────────────────────────

def test_every_figure_states_whether_its_content_may_become_model_input():
    """`PRODUCT_VISION.md`, artifact promotion. The rule is **not**
    label-blindness — that was wrong — it is **re-executability**: an artifact
    is promotable when the app can re-run its computation inside every fold."""
    for spec in FIG.REGISTRY.values():
        served = spec.to_dict()
        assert "promotable" in served, spec.id
        if spec.promotable:
            assert len(spec.promotable_because) > 80, spec.id


def test_pca_is_promotable_and_calibration_is_not():
    """The two verdicts, and neither follows from the figure being exploratory.

    A PCA fit is a deterministic function of the rows it sees, so it refits per
    fold and nothing crosses the split. A calibration curve is a property of a
    model already fitted to these rows — re-running it inside a fold would need
    the fold's own model, which is the thing being evaluated.
    """
    assert FS.PCA_SCORES.promotable is True
    assert "refitted inside every training fold" in FS.PCA_SCORES.promotable_because
    assert "never the component values" in FS.PCA_SCORES.promotable_because
    assert FS.CALIBRATION.promotable is False


def test_promotable_true_must_carry_its_argument():
    """`True` with no argument is the claim without the evidence."""
    with pytest.raises(FIG.FigureError, match="names why"):
        FIG.FigureSpec(
            id="z", title="z", tier=FIG.EXPLORATORY,
            when_applicable=lambda s: True, layers=(), annotations=(),
            checklist=(), caption=lambda p: "", evidence=FS.PCA_EVIDENCE,
            promotable=True)


def test_the_bundle_carries_promotability_to_the_reader():
    """Built, correct and unreachable is the same as not built — `DRIVE-001`."""
    df, qc = _assay()
    payload = FS.pca_scores_payload(df, group_col="group", qc_mask=qc)
    row = FIG.bundle({"pca_scores": payload})["admitted"][0]
    assert row["promotable"] is True and row["promotable_because"]


# ── Part D · the finding with no column ─────────────────────────────────────

def test_a_finding_about_the_cohort_states_its_scope_rather_than_nothing():
    """An empty chip row inside a full card frame reads as a card that failed
    to load. An absence is read as a missing name, never as 'there is no
    name'."""
    finding = {"id": "profile_sample_size_0", "severity": "warning",
               "title": "Small sample", "fix_kind": "none", "params": {}}
    shape = CF.render_shape(finding)
    assert shape["scope"] == CF.COHORT
    assert shape["has_chips"] is False
    assert shape["subject_line"], "the card would render an empty subject"
    assert "study as a whole" in shape["subject_line"]


def test_the_scope_is_derived_and_not_a_field_a_producer_must_remember():
    """Two of the producers are frozen modules that know nothing about scopes,
    and a finding whose scope was never set would default to `columns` and
    render the empty chip row this exists to prevent."""
    assert CF.scope_of({"affected_columns": ["age"]}) == CF.COLUMNS
    assert CF.scope_of({"params": {"columns": ["age"]}}) == CF.COLUMNS
    assert CF.scope_of({"params": {"rows": [1, 2, 3]}}) == CF.ROWS
    assert CF.scope_of({"params": {}}) == CF.COHORT
    # An explicit scope wins, so a producer that DOES know can say so.
    assert CF.scope_of({"scope": CF.COHORT, "affected_columns": ["age"]}) == CF.COHORT


def test_a_real_fixture_already_produces_a_cohort_finding():
    """This is not a hypothetical shape. `clinic_visits.csv` produces a
    sample-size finding whose subject is the study, and it was rendering an
    empty chip row before this landed."""
    from fastapi.testclient import TestClient

    from turbotab import api
    client = TestClient(api.app)
    fixture = Path(__file__).parent / "sample_data" / "clinic_visits.csv"
    with open(fixture, "rb") as fh:
        project = client.post("/project", files={
            "file": ("c.csv", fh, "text/csv")}).json()

    findings = project["findings"]
    assert findings, "no findings on this fixture"
    for f in findings:
        assert "shape" in f, f"{f['id']} reaches the page with no subject shape"
        assert f["shape"]["subject_line"], f["id"]
    cohort = [f for f in findings if f["shape"]["scope"] == CF.COHORT]
    assert cohort, (
        "no cohort-scoped finding on this fixture, so the shape is untested "
        "against real data")


def test_the_page_branches_on_the_shape_and_never_renders_an_empty_subject():
    page = (Path(__file__).resolve().parents[1] / "turbotab" / "web" /
            "index.html").read_text(encoding="utf-8")
    assert len(page) > 20_000 and "renderAll" in page      # positive control
    assert "shape.has_chips === false" in page, (
        "the card does not branch on the subject shape")
    assert "chip scope" in page and ".chip.scope{" in page
    # It is a STATEMENT, so it is sans and not mono — §03's three-voice rule
    # doing the work of saying "this is not a list of columns with the columns
    # missing".
    style = page[page.index(".chip.scope{"):page.index(".chip.scope{") + 240]
    assert "var(--sans)" in style


def test_a_repair_whose_columns_are_gone_is_withdrawn_rather_than_offered():
    """`ml/import_doctor.py:954` filters a finding's columns against the frame,
    and an empty intersection drops nothing and reports having dropped nothing —
    a repair that silently succeeds at doing nothing.

    That module is frozen, so the Guided door does the reading BEFORE offering
    the repair, and the refusal says which columns are gone rather than
    rendering a button that will no-op.
    """
    finding = {"id": "drop_columns", "fix_kind": "drop_columns",
               "affected_columns": ["ghost_a", "ghost_b"], "params": {}}
    refusal = CF.check_subject_survives(finding, ["age", "sex"])
    # On the distinctive CLAIM, not on a phrase that happens to be nearby —
    # the first draft asserted "not in the table any more" and the sentence
    # reads "none of those columns are in the table any more".
    assert refusal
    assert "withdrawn rather than offered" in refusal
    assert "change nothing and report success" in refusal
    assert "`ghost_a`" in refusal, "the refusal does not name what is gone"
    # A finding that still has a subject is offered normally.
    assert CF.check_subject_survives(finding, ["ghost_a"]) is None
    # And a cohort finding has no columns to lose.
    assert CF.check_subject_survives({"params": {}}, []) is None
