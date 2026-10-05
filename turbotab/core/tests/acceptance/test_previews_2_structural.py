"""PREVIEWS (2) · no number-changing choice goes without a picture.

The registry is enumerated from the decision union itself, so a kind added later fails here until
it either registers a consequence preview or is put on ``consequences.UNPREVIEWED`` with the reason
nothing on the canvas changes. The reasons are then held to the code: a kind on the list must not
feed a number. Its slot is read by no stage, or, where stages read it, what they compute from it is
shown here to be unchanged (a deferred or dismissed finding applies no repair; the outcome's unit
changes the words, never the numbers).

Every kind with a builder also previews on a fixture (``test_previews_kinds.py`` holds the
required eighteen; the rest are planned below on a small table), so a builder registered and never
reaching a view would fail too.
"""
from __future__ import annotations

from typing import get_args

import pytest

from turbotab.core import consequences, decisions as d
from turbotab.core.consequences import UNPREVIEWED, registered_kinds, words
from turbotab.core.tests.acceptance import preview_fixtures as F
from turbotab.core.tests.acceptance.preview_harness import Project

# V2 definition of done §1 and this package's spec: the kinds that must preview, by name.
REQUIRED = {
    "set_estimand", "set_adjustment", "set_follow_up", "set_clusters", "set_exposure_form",
    "set_model_sequence", "set_multiplicity", "set_scales", "set_usual_intake", "set_causal",
    "set_time_varying", "set_survey", "set_measurement_error", "set_batch", "set_outcome_scale",
    "set_column_unit", "join_files", "import_codebook",
}


def every_kind() -> set[str]:
    union = get_args(get_args(d.Decision)[0])
    return {model.model_fields["kind"].default for model in union}


def test_2_every_decision_kind_previews_or_says_why_nothing_changes():
    kinds = every_kind()
    previewed = registered_kinds()
    missing = sorted(kinds - previewed - set(UNPREVIEWED))
    assert not missing, f"decision kinds with neither a preview nor a reason: {missing}"
    both = sorted(previewed & set(UNPREVIEWED))
    assert not both, f"on the allow-list but previewed: {both}"
    stale = sorted(set(UNPREVIEWED) - kinds)
    assert not stale, f"allow-listed kinds that no longer exist: {stale}"
    assert REQUIRED <= previewed
    for kind, reason in UNPREVIEWED.items():
        assert 8 <= words(reason) <= 60, (kind, reason)


def test_2_every_preview_builder_is_registered_wherever_a_preview_is_planned():
    """``plan`` loads every builder module, so a server that imports only some of them still
    previews every kind: the registry is whole after one call."""
    for name in consequences.BUILDER_MODULES:
        __import__(name)
    assert registered_kinds() == set(consequences._BUILDERS) | set(consequences._TRANSFORMS)


def test_2_a_previews_state_is_the_logs_fold_with_the_answer():
    """A builder reads the state the answer would leave (``consequences.after_state``). Without the
    log at hand it is ``decisions.fold_onto``: for every representative decision of every kind, the
    same state as folding the log with the answer appended, but where the answer moves the slot a
    conditional answer reads (a new outcome, whose task the log's fold re-reads)."""
    from turbotab.core.tests.test_word_budgets import representative_decisions

    at = "2026-10-05T00:00:00Z"
    base = [d.SetLens(lenses=["dietary"]), d.SetTarget(column="hba1c"),
            d.SetTask(column="hba1c", task="regression"), d.SetPurpose(purpose="inference"),
            d.SetRoles(roles={"participant_id": "identifier", "energy_kcal": "energy",
                              "protein_g": "exposure", "fat_g": "exposure", "age": "covariate"})]
    records = [d.DecisionRecord(id=f"{i:032x}", seq=i, at=at, decision=x)
               for i, x in enumerate(base, start=1)]
    state = d.fold(records)
    conditional = set(d._HOLDS)
    checked = set()
    for decision in representative_decisions():
        if decision.kind == "revert":
            continue
        probe = d.DecisionRecord(id="f" * 32, seq=len(records) + 1, at=at, decision=decision)
        onto, folded = d.fold_onto(state, decision), d.fold([*records, probe])
        if onto != folded:
            # The outcome moved (``ln_hba1c`` is the log scale's): the log's fold re-reads the
            # conditional answers against it; every other slot is the same.
            assert decision.kind in ("set_target", "set_outcome_scale"), decision.kind
            assert onto.target == folded.target != state.target
            differ = {k for k in d.ProjectState.model_fields
                      if getattr(onto, k) != getattr(folded, k)}
            assert {d.SLOTS[k] for k in conditional} >= differ, decision.kind
        checked.add(decision.kind)
    assert checked == every_kind() - {"revert"}


def test_2_an_allow_listed_kinds_slot_feeds_no_stage_but_as_the_reason_says():
    """The stages that read each allow-listed kind's slot (``Stage.reads``): none for the plan lock,
    the re-seal and the censoring answer; the findings' dispositions for a deferral or a dismissal
    (tested below: only an applied repair changes anything); the outcome's unit for the words of
    the target's card, the substitution curve's labels and the explanations' axes."""
    from turbotab.core.stages import build_graph

    graph = build_graph()
    readers = {kind: sorted(s.name for s in graph.order() if d.SLOTS[kind] in s.reads)
               for kind in UNPREVIEWED}
    assert readers["lock_plan"] == readers["reseal"] == readers["set_censoring"] == []
    assert readers["set_outcome_unit"] == ["explain", "substitution", "target_info"]
    assert d.SLOTS["defer_finding"] == d.SLOTS["dismiss_finding"] == "findings"


def test_2_a_deferred_or_dismissed_finding_applies_nothing():
    """Every consumer of the findings slot reads it through the applied repairs alone
    (``repairs._applied``): the rows a repair excludes, the columns it sets aside, the values it
    rewrites. The same finding deferred or dismissed leaves each of them as no answer does, while
    applied it changes all three."""
    from turbotab.core import repairs

    finding = "pack::clinical::impossible_vs_extreme"
    params = {"bands": {"bp_di": [20, 200]}}
    nothing = d.ProjectState()

    def effects(state: d.ProjectState) -> tuple:
        return (repairs.exclusion_rules(state), repairs.unusable_columns(state),
                repairs.column_expressions(None, state))

    for action in ("deferred", "dismissed"):
        disposed = d.ProjectState(findings={finding: d.FindingDisposition(
            action=action, option="exclude_rows", params=params)})
        assert effects(disposed) == effects(nothing) == ([], [], {})
    for option, at in (("exclude_rows", 0), ("unusable", 1), ("set_missing", 2)):
        applied = d.ProjectState(findings={finding: d.FindingDisposition(
            action="applied", option=option, params=params)})
        assert effects(applied)[at], option


def test_2_the_outcomes_unit_changes_the_words_and_no_number(tmp_path):
    """The target stage with and without the outcome's unit: every number it reports (the
    histogram, the classes, the task) is the same; only the unit and its source differ."""
    from turbotab.core.tests.stage_harness import Ingested
    from turbotab.core.stages.target import target_info_stage

    frame = F.dietary_table(n=200)
    path = tmp_path / "t.csv"
    frame.to_csv(path, index=False)
    ing = Ingested(path, tmp_path)
    base = d.ProjectState(lens=["dietary"], target="tg", task="regression")
    plain = ing.run(target_info_stage, base)
    named = ing.run(target_info_stage, base.model_copy(update={"outcome_unit": "mg/dL"}))
    differ = {k for k in plain if plain[k] != named[k]}
    assert differ <= {"unit", "unit_source", "proposed_unit", "unit_candidates", "reason"}
    assert named["unit"] == "mg/dL"
    assert plain["histogram"] == named["histogram"]


# ── the kinds beyond the eighteen, each planned on a small table ─────────────

@pytest.fixture(scope="module")
def diet(tmp_path_factory):
    from turbotab.core.tests import modeling_fixtures as mf

    roles = {k: v for k, v in mf.NHANES_ROLES.items() if k != "triglycerides"}
    state = d.ProjectState(
        lens=["dietary"], target="glucose", task="regression", purpose="prediction",
        roles=roles, role_confirmations=dict(roles),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="SEQN"),
        shape_confirmations={"code_or_count:age": "amount",
                             "code_or_count:cycle_begin_year": "amount"},
        exclusions=[], missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.2, seed=0, folds=5), models=["linear"])
    project = Project(F.dietary_table(), tmp_path_factory.mktemp("previews_more"), state,
                      upto=["design", "cohort", "split", "proposals"])
    yield project
    project.close()


MORE = [
    d.SetTask(column="glucose", task="regression"),
    d.SetCategorical(columns=["age"]),
    d.SetSensitivity(analyses=[d.SensitivityAnalysis(label="500–3,500 kcal", rules=[
        d.ExclusionRule(column="kcal", low=500, high=3500, reason="implausible intakes")])]),
    d.ConfirmReading(reading="unit", column="kcal", value="kcal"),
    d.ConfirmReadings(items=[d.ReadingItem(reading="code_or_count", column="age", value="code")]),
    d.ConfirmRole(column="weight", role="excluded"),
    d.SetSubstitution(donor="fat_total", recipient="carb", step_kcal=100),
    d.SetExplain(curves="partial_dependence"),
    d.SetExplain(curves="ale"),
]


@pytest.mark.parametrize("decision", MORE, ids=lambda x: x.kind)
def test_2_each_other_number_changing_kind_previews(diet, decision):
    result, _ = diet.preview(decision)
    consequences.PreviewResult.model_validate(result.model_dump())
    assert 1 <= len(result.views) <= consequences.MAX_VIEWS, result.note
    for v in result.views:
        assert words(v.caption) <= consequences.CAPTION_WORDS, v.caption
        assert words(v.title) <= consequences.TITLE_WORDS, v.title


def test_2_the_substitution_preview_is_the_swap_on_real_rows(diet):
    """100 kcal from fat to carbohydrate: 100/9 g of fat out and 100/4 g of carbohydrate in on each
    row shown, by Atwater's factors (pandas here), total energy unchanged."""
    result, _ = diet.preview(d.SetSubstitution(donor="fat_total", recipient="carb", step_kcal=100))
    table = result.views[0]
    assert table.caption == ("One step: 11.1 of `fat_total` out, 25 of `carb` in, total energy "
                             "unchanged.")
    frame = F.dietary_table()
    for row in table.rows:
        assert row.after["fat_total"] == pytest.approx(frame.loc[row.row_id, "fat_total"] - 100 / 9)
        assert row.after["carb"] == pytest.approx(frame.loc[row.row_id, "carb"] + 25)


def test_2_explanations_preview_where_their_curve_reads_the_model(diet):
    """Partial dependence reads every row at every grid value: grid × rows points, many where no
    row is (the exposure tracks energy); accumulated local effects move each row within its bin."""
    pd_, _ = diet.preview(d.SetExplain(curves="partial_dependence"))
    ale, _ = diet.preview(d.SetExplain(curves="ale"))
    a, b = pd_.views[0], ale.views[0]
    assert a.x_label == b.x_label and a.r_before == pytest.approx(b.r_before)
    assert a.r_before > 0.7  # protein tracks energy in this table
    assert "combinations no row has" in a.caption and "its own bin" in b.caption


def test_2_a_revert_previews_the_answer_it_restores(diet):
    """With the log at hand (the server passes it), a revert previews the earlier answer it
    restores: here the earlier exposure form, a spline, in place of quintiles."""
    at = "2026-10-05T00:00:00Z"
    first = d.DecisionRecord(id="a" * 32, seq=1, at=at,
                             decision=d.SetExposureForm(column="protein", form="spline", knots=4))
    second = d.DecisionRecord(id="b" * 32, seq=2, at=at,
                              decision=d.SetExposureForm(column="protein", form="quintiles"))
    state = d.fold([first, second])
    ctx = diet.context(state=diet.state.model_copy(update={"exposure_forms": state.exposure_forms}))
    ctx.settings["records"] = [first, second]
    result = consequences.plan(d.Revert(decision_id=second.id), ctx, basis="")
    rel = result.views[0]
    assert rel.kind == "relationship" and "restricted cubic spline" in rel.caption
    alone = diet.context()
    alone.settings["records"] = [second]
    lone = consequences.plan(d.Revert(decision_id=second.id), alone, basis="")
    assert lone.views == [] and lone.note == ("Undoing it leaves the question unanswered, so it is "
                                              "asked again.")
