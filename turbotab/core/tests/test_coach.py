"""The coach (M2_CONTRACT §6): data-grounded notes on the stage's pictures and one line per card.

Budgets and the never-names-an-option rule run over every fixture in ``test_word_budgets.py``;
these pin what the notes say on known data, and that a note never reads a sealed row (Tier A,
leakage: the notes are computed on the preview's own pool).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core import coach, consequences, evidence, fact_previews, row_previews  # noqa: F401
from turbotab.core import decisions as d
from turbotab.core.consequences import (
    MAX_COACH, PreviewContext, RowFlowView, RowStep, TableFocusView, plan,
)
from turbotab.core.datastore import DataStore, ingest
from turbotab.core.decisions import ProjectState
from turbotab.core.models import previews  # noqa: F401 - the energy builder
from turbotab.core.stages.proposals import build_proposals
from turbotab.core.tests.stage_harness import SAMPLES
from turbotab.core.tests.test_row_previews import SpyStore

ROLES = {"participant_id": "identifier", "recall_number": "time", "age": "covariate", "sex": "covariate",
         "bmi": "covariate", "energy_kcal": "energy", "protein_g": "exposure", "fat_g": "exposure",
         "carbohydrate_g": "exposure", "sodium_mg": "exposure"}


@pytest.fixture(scope="module")
def recalls(tmp_path_factory):
    dest = tmp_path_factory.mktemp("coach") / "recalls.parquet"
    ingest(SAMPLES / "dietary_recalls.csv", dest)
    with DataStore(dest, 2 << 30) as store:
        yield store


@pytest.fixture(scope="module")
def blanks(tmp_path_factory):
    """400 people: `meds_hbp` asked of a quarter of them (a yes/no blank means not asked)."""
    rng = np.random.default_rng(0)
    n = 400
    meds = np.where(rng.random(n) < 0.25, rng.choice(["yes", "no"], n), None)
    frame = pd.DataFrame({"glucose": rng.normal(100, 15, n), "age": rng.integers(20, 80, n),
                          "meds_hbp": meds, "bmi": rng.normal(27, 4, n)})
    frame.loc[:9, "age"] = np.nan
    src = tmp_path_factory.mktemp("blanks") / "blanks.csv"
    frame.to_csv(src, index=False)
    dest = src.with_suffix(".parquet")
    ingest(src, dest)
    with DataStore(dest, 2 << 30) as store:
        yield store


def ctx_for(store, state, *, artifacts=None, training=None, sealed=None):
    artifacts = artifacts or {}
    return PreviewContext(project_id="coach", state=state, datastore=store,
                          artifact=lambda s: artifacts.get(s), training_row_ids=training,
                          cohort_row_ids=None, sealed_row_ids=sealed)


def notes(result):
    return [(v.kind, n.text, n.anchor.kind, n.anchor.ref) for v in result.views for n in v.coach]


# ── what the notes say ───────────────────────────────────────────────────────

def test_the_exclusions_cut_names_its_tails_and_who_leaves(recalls):
    state = ProjectState(lens=["dietary"], target="hba1c", roles=ROLES)
    rule = d.ExclusionRule(column="energy_kcal", low=500, high=5000, reason="implausible intakes")
    result = plan(d.SetExclusions(rules=[rule]), ctx_for(recalls, state), basis="")
    said = notes(result)
    frame = recalls.materialize(["energy_kcal", "hba1c"])
    measured = frame[frame["hba1c"].notna()]
    below = int((measured["energy_kcal"] < 500).sum())
    above = int((measured["energy_kcal"] > 5000).sum())
    assert ("distribution", f"`{below}` rows below `500` kcal: likely under-reporting.", "range",
            [said[1][3][0], 500.0]) in said
    assert any(t == f"`{above}` rows above `5,000` kcal: likely over-reporting." for _, t, _, _ in said)
    flow = [n for n in said if n[0] == "row_flow"]
    assert flow and flow[0][1].startswith("Excluded rows' median `bmi` is ") and flow[0][2] == "step"
    assert all(len(v.coach) <= MAX_COACH for v in result.views)


def test_the_coach_withholds_the_outcome_at_the_eligibility_question(recalls):
    """A rule on the outcome itself: the cut is drawn by the rows builder, but the coach adds
    nothing about the outcome's values (lockbox constitution §04)."""
    state = ProjectState(lens=["dietary"], target="hba1c", roles=ROLES)
    rule = d.ExclusionRule(column="hba1c", low=5, reason="a range of the outcome")
    result = plan(d.SetExclusions(rules=[rule]), ctx_for(recalls, state), basis="")
    assert not [n for n in notes(result) if n[0] == "distribution"]


def test_blanks_that_mean_not_asked_are_said_on_the_cells(blanks):
    state = ProjectState(target="glucose", roles={"age": "covariate", "meds_hbp": "covariate",
                                                  "bmi": "covariate"})
    columns = [c.to_dict() for c in blanks.info().columns]
    proposals = build_proposals(blanks.materialize(), columns, lens=None, target="glucose",
                                roles=state.roles)
    assert proposals["coach"]["missing"]["text"].startswith("`meds_hbp` blank on `7")
    ctx = ctx_for(blanks, state, artifacts={"proposals": proposals})
    result = plan(d.SetMissing(strategy="complete_case"), ctx, basis="")
    said = notes(result)
    n_blank = int(blanks.materialize(["meds_hbp"])["meds_hbp"].isna().sum())
    assert ("row_flow", f"`meds_hbp` is blank on `{n_blank}` of these rows.", "step",
            "complete_cases") in said
    table = [n for n in said if n[0] == "table_focus"]
    assert table and table[0][1].endswith("likely a question not asked.") and table[0][3] == "meds_hbp"


def test_the_energy_notes_read_the_relationship_on_training_rows(recalls):
    state = ProjectState(lens=["dietary"], target="hba1c", roles=ROLES)
    training = np.arange(0, int(recalls.n_rows), 2)
    nutrients = ["protein_g", "fat_g", "carbohydrate_g"]
    for method, second in (("residual", "With this method r `0.00`: what is left is composition."),
                           ("standard", "enters the model beside it."),
                           ("none", "Unadjusted, ")):
        decision = d.SetEnergyAdjustment(method=method, energy_column="energy_kcal", nutrients=nutrients)
        result = plan(decision, ctx_for(recalls, state, training=training), basis="")
        rel = next(v for v in result.views if v.kind == "relationship")
        first, other = (n.text for n in rel.coach)
        assert first.startswith(f"`{rel.y_label_before}` tracks `energy_kcal` at r `0.")
        assert second in other, other


def test_the_aggregation_notes_follow_what_repeats(recalls):
    """The aggregation preview is the sequence's; the coach writes on whatever its primary view is."""
    base = ProjectState(lens=["dietary"], target="hba1c",
                        grain={"grain": "repeated", "id_column": "participant_id"})

    def primary():
        steps = [RowStep(key="loaded", label="Rows", n=600), RowStep(key="combined", label="People", n=300)]
        return RowFlowView(title="One row per person", caption="Rows combine.", before=steps[:1],
                           after=steps)

    view = primary()
    replicates = base.model_copy(update={"repeat_kind": d.RepeatSpec(repeat_kind="repeats")})
    coach.annotate(d.SetAggregation(method="mean"), [view], ctx_for(recalls, replicates))
    texts = [n.text for n in view.coach]
    assert texts == ["Averaging `2` replicates cuts within-person variance `2`-fold."]
    assert view.coach[0].anchor.kind == "step" and view.coach[0].anchor.ref == "combined"
    # hba1c is measured once per person here, so nothing is said about which outcome to keep.

    view = primary()
    visits = base.model_copy(update={"repeat_kind": d.RepeatSpec(repeat_kind="time_points")})
    coach.annotate(d.SetAggregation(method="mean", outcome="mean"), [view], ctx_for(recalls, visits))
    assert [n.text for n in view.coach] == ["Averaging `2` time points erases change between them."]

    view = primary()
    coach.annotate(d.SetAggregation(method="change", outcome="last"), [view], ctx_for(recalls, visits))
    assert [n.text for n in view.coach] == ["No time column is named, so order is file order."]


def test_an_outcome_that_varies_within_a_unit_is_said(tmp_path):
    """Clinic visits: the outcome is measured at every visit, so combining needs "which outcome"."""
    dest = tmp_path / "visits.parquet"
    ingest(SAMPLES / "clinical_longitudinal.csv", dest)
    frame = pd.read_csv(SAMPLES / "clinical_longitudinal.csv")
    varies = int((frame.groupby("subject_id")["hba1c"].nunique() > 1).sum())
    state = ProjectState(target="hba1c", grain={"grain": "repeated", "id_column": "subject_id"},
                         repeat_kind={"repeat_kind": "time_points", "time_column": "visit_date"})
    view = RowFlowView(title="t", caption="c", before=[], after=[RowStep(key="k", label="l", n=1)])
    with DataStore(dest, 2 << 30) as store:
        coach.annotate(d.SetAggregation(method="first"), [view], ctx_for(store, state))
    assert [n.text for n in view.coach] == [
        f"`hba1c` differs within `{varies}` of `200` `subject_id` values."]
    assert view.coach[0].anchor.model_dump() == {"kind": "column", "ref": "hba1c"}


def test_the_cards_get_one_line_each_and_never_the_energy_card(recalls):
    columns = [c.to_dict() for c in recalls.info().columns]
    proposals = build_proposals(recalls.materialize(), columns, lens=["dietary"], target="hba1c",
                                roles=ROLES)
    frame = recalls.materialize(["energy_kcal", "hba1c"])
    below = int(((frame["energy_kcal"] < 500) & frame["hba1c"].notna()).sum())
    assert proposals["coach"]["exclusions"] == {
        "text": f"`{below}` rows below `500` kcal: likely under-reporting.",
        "anchor": {"kind": "column", "ref": "energy_kcal"}}
    # A modeling choice made after the seal: the proposals read every row, so no line there.
    assert "energy_adjustment" not in proposals["coach"]


def test_implausible_intake_evidence_points_at_both_tails(recalls):
    finding = {"id": "pack::dietary::implausible_intake", "affected_columns": ["energy_kcal"],
               "summary": "x"}
    result = evidence.evidence(finding, evidence.EvidenceContext(state=ProjectState(), datastore=recalls))
    texts = [n.text for v in result.views for n in v.coach]
    assert any(t.endswith("below `500` kcal: likely under-reporting.") for t in texts)
    assert any(t.endswith("above `5,000` kcal: likely over-reporting.") for t in texts)


def test_a_failing_annotator_never_fails_the_preview(recalls):
    def broken(decision, views, ctx):
        raise RuntimeError("boom")

    coach.register_coach("set_purpose", broken)
    try:
        result = plan(d.SetPurpose(purpose="prediction"), ctx_for(recalls, ProjectState()), basis="")
        assert result.note and result.note.startswith("With prediction")
    finally:
        coach._ANNOTATORS["set_purpose"].remove(broken)


# ── Tier A: a note never reads a sealed row ──────────────────────────────────

@pytest.mark.parametrize("decision", [
    d.SetExclusions(rules=[d.ExclusionRule(column="energy_kcal", low=800, high=4000, reason="tight")]),
    d.SetMissing(strategy="complete_case"),
    d.SetAggregation(method="mean"),
])
def test_no_coach_note_reads_a_held_out_row(recalls, decision):
    sealed = np.arange(0, int(recalls.n_rows), 5, dtype=np.int64)
    spy = SpyStore(recalls)
    state = ProjectState(lens=["dietary"], target="hba1c", roles=ROLES,
                         grain={"grain": "repeated", "id_column": "participant_id"},
                         repeat_kind={"repeat_kind": "repeats"})
    ctx = ctx_for(spy, state, sealed=sealed)
    views = [RowFlowView(title="t", caption="c", before=[], after=[RowStep(key="k", label="l", n=1)]),
             TableFocusView(title="t", caption="c", columns_before=["sodium_mg"], columns_after=[],
                            rows=[], changed=[], n_affected_columns=1)]
    coach.annotate(decision, views, ctx)
    result = plan(decision, ctx, basis="") if decision.kind != "set_aggregation" else None
    assert not spy.read_everything
    assert not spy.read & set(sealed.tolist())
    assert result is None or all(len(v.coach) <= MAX_COACH for v in result.views)


# ── outcome units ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("column, values, unit, proposed", [
    ("glucose", [99, 101, 110], (None, None), "mg/dL"),
    ("glucose", [5.2, 5.9, 6.1], (None, None), "mmol/L"),
    ("glucose_mgdl", None, ("mg/dL", "name"), None),
    ("ldl_mmol_l", None, ("mmol/L", "name"), None),
    ("hba1c", [5.4, 5.9], (None, None), "%"),
    ("bp_sys", None, (None, None), "mmHg"),
    ("bmi", None, (None, None), "kg/m²"),
    ("progressed", [0, 1], (None, None), None),
    ("score", [1, 2, 3], (None, None), None),
])
def test_the_outcome_unit_is_stated_from_the_name_and_proposed_from_the_pack(column, values, unit,
                                                                               proposed):
    """Audit IN-05: a unit is stated only when the name spells it out (or a decision records it);
    the clinical pack's reading is a proposal, never a statement."""
    from turbotab.core.units import from_pack, outcome_unit

    assert outcome_unit(column, values) == unit
    assert outcome_unit(column, values, recorded="mg/dL") == ("mg/dL", "decision")
    if unit[0] is None:
        assert from_pack(column, values) == proposed


def test_the_target_preview_shows_the_outcome_with_its_unit(tmp_path):
    rng = np.random.default_rng(1)
    src = tmp_path / "labs.csv"
    pd.DataFrame({"glucose_mg_dl": rng.normal(100, 12, 300), "glucose": rng.normal(100, 12, 300),
                  "diabetes": rng.choice(["no", "yes"], 300),
                  "age": rng.integers(20, 80, 300)}).to_csv(src, index=False)
    ingest(src, tmp_path / "labs.parquet")
    with DataStore(tmp_path / "labs.parquet", 2 << 30) as store:
        ctx = ctx_for(store, ProjectState())
        result = plan(d.SetTarget(column="glucose_mg_dl"), ctx, basis="")
        dist = next(v for v in result.views if v.kind == "distribution")
        assert " mg/dL, middle half " in dist.caption and dist.before_label.endswith("(mg/dL)")
        # A name that does not spell its unit out states none (audit IN-05: never guessed).
        result = plan(d.SetTarget(column="glucose"), ctx_for(store, ProjectState()), basis="")
        dist = next(v for v in result.views if v.kind == "distribution")
        assert "mg/dL" not in dist.caption and "mmol" not in dist.caption
        result = plan(d.SetTarget(column="diabetes"), ctx_for(store, ProjectState()), basis="")
        assert result.note.startswith("Two levels: `")
        assert result.note.endswith("the next question asks which is the event.")
