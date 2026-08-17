"""`GUIDED-230` — the three recorded answers reach the model shelf.

The lens was consumed by thirteen non-test modules and stopped at exactly one
place: `grep -c lens` was **0** in `turbotab/models.py` and **0** in
`ml/model_registry.py`, and `AnalysisProject.model_shelf_ranked` passed none of
`purpose`, `repeat_kind` or `unit_of_analysis` — three answers the opening
sequence deliberately asks for, and which the app then declined to read where a
user picks a model.

Three rules govern the carry-in, and the file is organized around them because
they are what stop it becoming a filter.

1. **The shelf is never shortened.** Asserted as set equality against the
   registry, under **every combination of the three answers** rather than
   against one — a filter that only fires on `inference` would pass a check run
   under `prediction`. That is the failure condition and it is checked first.
2. **The sentence is the deliverable.** Every clause is compared **character for
   character** against `Decision.text`, the string the record actually kept, and
   never against a sentence composed here — which would be this file agreeing
   with itself. `L36` ruled this for the methods section, `L53-B` for the
   checklist and `L54-B` for the deck card; this is the fourth surface under one
   rule.
3. **An unanswered question changes nothing.** Asserted as identity of the
   whole ordered key list, not as "similar".

## The capability this rests on is MEASURED, not declared

`ModelCapabilities.exposes_coefficients` is the only new fact in the registry
and the ordering is entirely downstream of it, so it is checked against a real
fit: every model is fitted and asked `hasattr(est, "coef_")` — the same question
`turbotab.figure_bundle._coefficients_for` asks to decide whether a coefficient
figure can be drawn. A hand-declared capability agreeing with a hand-written
clause is trap #3 with two hands.

## `GUIDED-097` — two fixtures of different target shape

Every claim about the ordering runs against `clinical_longitudinal.csv`
(**binary numeric** `progressed`, 200 people × 3 visits, so the repeat questions
are answerable) and `multiclass_stage.csv` (**multiclass string**
`disease_stage`, one row per person, so the repeat questions are *not* asked and
the shelf sees a design with one answer instead of three). **The shape not
covered is said at the bottom of this file.**
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model_registry import get_registry                          # noqa: E402
from turbotab import eligibility as E, engine, grain as G           # noqa: E402
from turbotab import models as M                                    # noqa: E402
from turbotab.project import AnalysisProject                        # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: `(fixture, target, task, group column or None)`. The second has no group
#: column at all, which is what makes it the shape where two of the three
#: questions are never asked.
SHAPES: Dict[str, tuple] = {
    "binary_numeric_repeated": ("clinical_longitudinal.csv", "progressed",
                                "classification", "subject_id"),
    "multiclass_string_flat": ("multiclass_stage.csv", "disease_stage",
                               "classification", None),
}


def _project(shape: str, purpose: Optional[str] = None,
             repeat_kind: str = "time_points",
             unit: str = M.UNIT_RECORD) -> AnalysisProject:
    """A sealed project carrying exactly the answers it was asked for.

    **The repeat answers are not optional and that is the constitution, not a
    convenience here.** `seal_lockbox` refuses while `repeat_chain_gap()` is
    non-empty, so a table where people repeat cannot reach the shelf without
    questions 4 and 5 answered. The only one of the three that can genuinely be
    absent at the shelf is the purpose — which is what "unanswered" means below,
    and it is the app's own shape rather than one chosen for the test.
    """
    fixture, target, task, group_col = SHAPES[shape]
    df = pd.read_csv(DATA / fixture)
    p = AnalysisProject.from_dataframe(df, fixture)
    p.set_target(target, task, "high", [])
    if purpose is not None:
        p.set_purpose(purpose)
    if group_col:
        p.set_grain(G.PEOPLE_REPEAT, group_col=group_col)
        p.set_repeat_kind(repeat_kind)
        p.set_unit_of_analysis(unit)
        # ONE ROW PER PERSON MEANS THE ROWS ARE ACTUALLY COMBINED, and the seal
        # refuses until they are — Decision A's identity barrier, because a seal
        # drawn beforehand names rows that no longer exist. Answering question 5
        # `person` without question 6 is not a state the app lets a project
        # reach, so the fixture cannot pretend otherwise either.
        if unit == M.UNIT_PERSON:
            p.set_aggregation("mean" if repeat_kind == "repeats" else "last")
    else:
        p.set_grain(G.ONE_ROW_PER_PERSON)
    p.set_eligibility(E.EVERYONE)
    drawn = engine.draw_holdout(p.df, target, task, p.grain)
    p.seal_lockbox(drawn["labels"], **drawn["disclosure"])
    return p


def _project_with_blank_outcomes(blank_from: int = 400) -> AnalysisProject:
    """The shape `DRIVE-050` needs and NO fixture in this repository has.

    Surveyed at L63: every one of the four fixtures with a genuine
    people-repeat roster — `clinical_labs`, `clinical_longitudinal`,
    `dietary_recalls`, `longitudinal_visits` — has a 100% complete target, and
    two of them contain no NaN in any column at all. The one file that looks
    like it has both (`metabolomics_paired_logged.csv`: 36 subjects × 2, eight
    blank `responder`) is a false positive — its eight blank outcomes are
    exactly its eight blank-`subject_id` QC rows, so the blanks belong to rows
    with no person rather than to people whose visits lack an outcome.

    **Without blank outcomes `analysis_rows == training_rows` and both the
    correct and the incorrect implementation satisfy every assertion**, which
    is why this defect survived `DRIVE-045` in the same file, thirty-three
    lines from the fix. So the frame is derived from the real longitudinal
    fixture with its outcome blanked from a row onward — a real roster, a real
    seal, and the two populations genuinely different.
    """
    fixture, target, task, group_col = SHAPES["binary_numeric_repeated"]
    df = pd.read_csv(DATA / fixture)
    df.loc[df.index[blank_from:], target] = None
    p = AnalysisProject.from_dataframe(df, fixture)
    p.set_target(target, task, "high", [])
    p.set_purpose(M.INFERENCE)
    p.set_grain(G.PEOPLE_REPEAT, group_col=group_col)
    p.set_repeat_kind("time_points")
    p.set_unit_of_analysis(M.UNIT_RECORD)
    p.set_eligibility(E.EVERYONE)
    drawn = engine.draw_holdout(p.df, target, task, p.grain)
    p.seal_lockbox(drawn["labels"], **drawn["disclosure"])
    return p


def test_the_shelf_says_which_rows_it_ranked_and_how_many_people(capsys):
    """`DRIVE-050`. The sentence describing the ranking counted a wider
    population than the ranking, and both numbers reached a reader.

    `recorded_design()` set `rows = self.training_rows`; thirty-three lines
    below, `model_shelf_ranked` ranks on `self.analysis_rows`. Driven on a
    900-row fixture the ONE `/models` response served `n_rows_seen: 225` and,
    in the same payload, *"This order was computed from 825 rows, which are 275
    people."*

    **The people count is the one a repeated-measures reader acts on**, and no
    row had named it before `L63`. It is derived from the same frame, so it was
    wrong by the same mistake.
    """
    project = _project_with_blank_outcomes()

    # THE CONTROL THE FILE LACKED, and on the axis that matters. The existing
    # control is `people < rows`, which is about the grain and is satisfied by
    # both implementations.
    n_training = len(project.training_rows)
    n_analysis = len(project.analysis_rows)
    assert n_training > n_analysis, (
        f"the fixture has no rows whose outcome is blank ({n_training} "
        f"training, {n_analysis} analysis), so it cannot tell the two "
        f"implementations apart and this test proves nothing")

    design = project.recorded_design()
    assert design.n_rows == n_analysis, (
        f"the recorded design says the order was computed from "
        f"{design.n_rows} rows; the shelf ranked on {n_analysis}")
    assert design.n_people == project.analysis_rows["subject_id"].nunique()

    entries, ranked_on = project.model_shelf_ranked()
    assert design.n_rows == int(ranked_on.n_rows), (
        f"the sentence says {design.n_rows} and the profile the shelf actually "
        f"ranked on holds {ranked_on.n_rows} — one response, two numbers, "
        f"thirty-three lines of source apart")

    # THE SENTENCE, not the field. `DRIVE-050`'s effect string contains no `n=`
    # token at all, which is why `L63`-era regexes scoped to `model["concern"]`
    # could never have seen this.
    statements = {s["answer"]: s for s in M.design_statement(design)}
    effect = statements["repeat_kind"]["effect"]
    assert f"{n_analysis:,} rows" in effect, (
        f"the statement does not say the count the order was computed from: "
        f"{effect!r}")
    assert f"{n_training:,} rows" not in effect, (
        f"the statement still cites the training-row count: {effect!r}")
    people = project.analysis_rows["subject_id"].nunique()
    assert f"{people:,} people" in effect, effect
    assert f"{project.training_rows['subject_id'].nunique():,} people" \
        not in effect or people == project.training_rows["subject_id"].nunique()
    with capsys.disabled():
        print(f"\n  training {n_training} rows / "
              f"{project.training_rows['subject_id'].nunique()} people · "
              f"ranked on {n_analysis} rows / {people} people")


def _keys(entries) -> List[str]:
    return [e.key for e in entries]


def _last_text(project: AnalysisProject, kind: str) -> str:
    """The record's own sentence for a decision kind. Never recomposed."""
    hits = [d.text for d in project.decisions if d.kind == kind]
    assert hits, f"no {kind} decision was recorded, so there is nothing to quote"
    return hits[-1]


# ═══════════ 1 · the failure condition, checked first and everywhere ═════════

@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_no_answer_removes_disables_or_hides_a_model(shape):
    """**The shelf is never shortened**, under every combination of answers.

    Set equality against the registry filtered on capability, not a count: a
    check on length would pass a shelf that dropped the two the design disliked
    and gained nothing. And every combination, not one, because a filter that
    fires on a single answer is exactly what a single-answer check misses.
    """
    fixture, target, task, group_col = SHAPES[shape]
    can_fit = {k for k, spec in get_registry().items()
               if getattr(spec.capabilities, f"supports_{task}", False)}
    assert can_fit, "no registry model can fit this task; the check has no subject"

    combos = []
    for purpose in (None, M.PREDICTION, M.INFERENCE):
        if not group_col:
            combos.append((purpose, "time_points", M.UNIT_RECORD))
            continue
        for repeat_kind in ("repeats", "time_points"):
            for unit in (M.UNIT_PERSON, M.UNIT_RECORD, M.UNIT_NOT_DESCRIBED):
                combos.append((purpose, repeat_kind, unit))

    for purpose, repeat_kind, unit in combos:
        project = _project(shape, purpose, repeat_kind, unit)
        entries = project.model_shelf()
        on_shelf = {e.key for e in entries}
        assert on_shelf == can_fit, (
            f"{shape} with purpose={purpose!r} repeat_kind={repeat_kind!r} "
            f"unit={unit!r}: the shelf is {sorted(on_shelf)} and the registry "
            f"offers {sorted(can_fit)}. Missing: {sorted(can_fit - on_shelf)}. "
            f"The recorded design changes the ORDER and never what is "
            f"available — judgment is rendered as order, not as absence.")
        # NOT DISABLED AND NOT HIDDEN EITHER, which is a different claim from
        # "present": every entry must still be selectable, and a lowered one
        # must still carry the engine's own concern rather than having it
        # replaced by the design clause.
        for entry in entries:
            assert entry.concern, (
                f"{entry.key} lost its shape concern under "
                f"purpose={purpose!r}; the design clause is an ADDITION")


# ═══════════ 2 · an unanswered question changes nothing ══════════════════════

@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_an_unanswered_question_leaves_the_order_exactly_as_it_was(shape):
    """Identity of the ordered key list, not similarity.

    Saying nothing must produce the same list as the code before `L55-B`: no
    design object at all, an empty one, and a real project where the purpose was
    never asked — including, on the repeated fixture, one where the repeat
    answers ARE recorded, which is the stronger version of the claim.
    """
    project = _project(shape)
    assert project.recorded_design().purpose is None, (
        f"{shape}: the fixture recorded a purpose, so 'unanswered' below is "
        "not about an unanswered question")
    profile = engine.profile(project.training_rows, project.target,
                             project.task_type)
    task = project.task_type or "regression"

    without = _keys(M.shelf(profile, task))
    empty = _keys(M.shelf(profile, task, design=M.NO_DESIGN))
    unasked = _keys(project.model_shelf())

    assert without == empty == unasked, (
        f"{shape}: an unanswered design moved the shelf.\n"
        f"  no design object : {without}\n"
        f"  empty design     : {empty}\n"
        f"  purpose unasked  : {unasked}\n"
        "Absence of an answer is not an answer.")
    assert not any(e.design_notes for e in project.model_shelf()), (
        f"{shape}: an entry carries a design clause with no purpose recorded")

    statements = M.design_statement(project.recorded_design())
    assert not [s for s in statements if s["answer"] == "purpose"], (
        f"{shape}: the shelf made a statement about a purpose nobody recorded")
    assert M.design_statement(M.NO_DESIGN) == [], (
        "an empty design produced a statement")


def test_an_answer_the_app_cannot_quote_reorders_nothing():
    """The second half of rule 3, and it is the one a reader would miss.

    An answer whose recorded SENTENCE is missing is treated as unanswered. This
    is not defensive coding — it is the rule that the clause is the deliverable,
    made structural: the shelf will not move a model on grounds it cannot state.
    """
    project = _project("binary_numeric_repeated", M.INFERENCE)
    quotable = project.recorded_design()
    assert quotable.answered("purpose"), (
        "the fixture recorded no quotable purpose, so the negative below "
        "would be true for the wrong reason")

    profile = engine.profile(project.training_rows, project.target,
                             project.task_type)
    task = project.task_type or "regression"
    mute = M.RecordedDesign(purpose=M.INFERENCE, purpose_said=None)

    assert not mute.answered("purpose")
    assert _keys(M.shelf(profile, task, design=mute)) == \
        _keys(M.shelf(profile, task)), (
        "a purpose with no recorded sentence beside it reordered the shelf. "
        "The clause quotes the record; an answer the record cannot supply a "
        "sentence for is one the shelf cannot explain, and an unexplained "
        "reorder is a black box wearing a lens.")
    assert M.design_statement(mute) == []


# ═══════════ 3 · the answer that does move something, and its sentence ══════

@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_recording_inference_ranks_the_coefficientless_models_lower(shape):
    """The consequence, observed as a change in position rather than a flag.

    Compared against the SAME project answered `prediction`, so the only thing
    that differs between the two orders is the recorded answer.
    """
    predicting = _keys(_project(shape, M.PREDICTION).model_shelf())
    estimating = _keys(_project(shape, M.INFERENCE).model_shelf())

    assert predicting != estimating, (
        f"{shape}: recording an inference objective changed nothing about the "
        f"order. Both read {predicting}. The answer is being carried and not "
        f"read, which is this codebase's oldest habit.")
    assert sorted(predicting) == sorted(estimating), (
        f"{shape}: the two orders hold different models, so something was "
        f"added or removed rather than reordered")

    registry = get_registry()
    entries = _project(shape, M.INFERENCE).model_shelf()
    by_bucket: Dict[str, List] = {}
    for entry in entries:
        by_bucket.setdefault(entry.bucket, []).append(entry)

    moved = 0
    for bucket, group in by_bucket.items():
        exposes = [registry[e.key].capabilities.exposes_coefficients
                   for e in group]
        # WITHIN THE BUCKET, because that is the whole of what the design is
        # allowed to do: every model that exposes coefficients precedes every
        # model that does not, and the coach's verdict about the shape is
        # untouched.
        seen_without = False
        for entry, has in zip(group, exposes):
            if has is False:
                seen_without = True
                moved += 1
            elif has is True:
                assert not seen_without, (
                    f"{shape}/{bucket}: {entry.key} exposes coefficients and "
                    f"sits below a model that does not, under a recorded "
                    f"inference objective. Order: {[e.key for e in group]}")
    assert moved, (
        f"{shape}: no model was ranked lower, so this fixture cannot tell a "
        f"shelf that reads the answer from one that ignores it")


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_clause_quotes_the_recorded_decision_character_for_character(shape):
    """Rule 2. The record's own string, not a sentence composed here.

    `set_purpose` composes one sentence and hands it to `record`; nothing copies
    it back onto the project field. So a surface that wants to quote it has to
    go to the record — and this asserts it did, by equality with
    `Decision.text` rather than by a substring or a keyword.
    """
    project = _project(shape, M.INFERENCE)
    recorded = _last_text(project, "set_purpose")

    lowered = [e for e in project.model_shelf() if e.design_rank]
    assert lowered, f"{shape}: nothing was lowered, so nothing carries a clause"

    for entry in lowered:
        assert entry.design_notes, (
            f"{shape}: {entry.key} was ranked lower and says nothing about "
            f"why. A reordered shelf with no account of why is a black box.")
        for note in entry.design_notes:
            assert note.quote == recorded, (
                f"{shape}: {entry.key}'s clause quotes\n  {note.quote!r}\n"
                f"and the record says\n  {recorded!r}\n"
                "The recorded decision is quoted, never re-composed.")
            assert note.answer == "purpose"
            assert note.question == "What is this model for?"
            assert entry.name in note.clause, (
                f"{shape}: {entry.key}'s clause does not name the model it is "
                f"about: {note.clause!r}")

    # AND THE OTHER SIDE OF IT: an entry that did not move carries no clause,
    # so a clause means exactly one thing — this model was moved, by this
    # answer. A caveat on every row would make a real concern and a routine one
    # read identically.
    for entry in project.model_shelf():
        if not entry.design_rank:
            assert not entry.design_notes, (
                f"{shape}: {entry.key} did not move and carries a clause")


def test_the_shelf_level_statement_quotes_both_repeat_answers_and_says_the_number():
    """The answers that bear on every model equally, reported once.

    Repeated measures do not reorder anything and the statement says so: every
    model here treats rows as independent, so demoting all of them would move
    nothing while reading as a judgment. What is said instead is the number —
    rows against people — which nothing else on this surface says.
    """
    project = _project("binary_numeric_repeated", M.INFERENCE,
                       "time_points", M.UNIT_RECORD)
    design = project.recorded_design()
    statements = {s["answer"]: s for s in M.design_statement(design)}

    assert set(statements) == {"purpose", "repeat_kind"}, (
        f"expected a statement for the purpose and for the repeat design, got "
        f"{sorted(statements)}")

    repeats = statements["repeat_kind"]
    assert _last_text(project, "set_repeat_kind") in repeats["quote"], (
        "the repeat statement does not carry the recorded sentence")
    assert _last_text(project, "set_unit_of_analysis") in repeats["quote"], (
        "the repeat statement does not carry the recorded unit sentence — both "
        "answers are recorded and both are quoted, because the unit is what "
        "decides whether the repeats were resolved before the seal")

    # `analysis_rows`, NOT `training_rows` — `DRIVE-050`. This read
    # `len(project.training_rows)` and passed either way, because
    # `binary_numeric_repeated` has no blank outcome: the two populations are
    # the same frame here, so the assertion could not tell the implementations
    # apart. The falsifying fixture is
    # `test_the_shelf_says_which_rows_it_ranked` below; this stays on the
    # correct axis so the two agree about what the sentence means.
    rows = len(project.analysis_rows)
    people = project.analysis_rows["subject_id"].nunique()
    assert people < rows, (
        "the fixture's training rows are already one per person, so the "
        "rows-versus-people claim has nothing to distinguish")
    assert f"{rows:,} rows" in repeats["effect"], (
        f"the statement does not say how many rows the order was computed "
        f"from: {repeats['effect']!r}")
    assert f"{people:,} people" in repeats["effect"], (
        f"the statement does not say how many people those rows are: "
        f"{repeats['effect']!r}")

    # THE ORDER IS UNCHANGED BY THIS ANSWER, and that is asserted rather than
    # described: the same project without the two repeat answers must produce
    # the same order.
    with_repeats = _keys(project.model_shelf())
    without = _keys(_project("binary_numeric_repeated", M.INFERENCE).model_shelf())
    assert with_repeats == without, (
        "recording a repeated-measures design moved the shelf. It must not: "
        "every model here treats rows as independent, so the concern is "
        "uniform, and a uniform demotion is a no-op dressed as a judgment.")


def test_a_prediction_objective_is_reported_as_read_and_unchanged():
    """The recorded-absence rule, one level in.

    An answer that changed nothing and an answer nobody read look identical
    from outside, so the shelf says which one happened.
    """
    project = _project("binary_numeric_repeated", M.PREDICTION)
    statements = {s["answer"]: s for s in
                  M.design_statement(project.recorded_design())}
    assert "purpose" in statements, (
        f"a recorded prediction objective produced no statement at all; got "
        f"{sorted(statements)}")
    said = statements["purpose"]
    assert said["quote"] == _last_text(project, "set_purpose")
    assert "unchanged" in said["effect"], (
        f"a prediction objective produced no statement that the order is "
        f"unchanged: {said['effect']!r}")
    assert not any(e.design_notes for e in project.model_shelf()), (
        "a prediction objective moved a model; nothing about predicting "
        "separates one model on this shelf from another")


# ═══════════ 4 · the capability the whole ordering rests on ═════════════════

def test_every_registry_model_declares_whether_it_exposes_coefficients():
    """`None` is undeclared and takes no part in the ordering — so it is caught.

    The behavior on `None` is silence rather than a false claim, which is the
    right failure. This is the check that stops silence becoming the norm.
    """
    undeclared = sorted(k for k, spec in get_registry().items()
                        if spec.capabilities.exposes_coefficients is None)
    assert not undeclared, (
        f"{undeclared} do not declare `exposes_coefficients`. The shelf will "
        f"say nothing about them under a recorded inference objective, which "
        f"is correct and is not the same as being right. Fit the model and ask "
        f"`hasattr(est, 'coef_')` — the test below is the measurement.")


def test_the_declared_capability_matches_a_real_fit():
    """Measured, not asserted. The one fact the ordering is downstream of.

    `hasattr(est, "coef_")` is the same question `figure_bundle
    ._coefficients_for` asks before drawing the coefficient figure, so the
    declaration and the figure cannot disagree about which models have an
    estimate to show.

    **`nn` is excluded, named, and the exclusion is ASSERTED rather than
    skipped.** `torch` is deliberately absent here (`TEST-038`), so the model
    cannot be fitted at all; a `pytest.skip` would take the whole check with it,
    which is `TEST-059`'s shape. The set actually measured is asserted to be
    everything else.
    """
    import numpy as np

    registry = get_registry()
    unmeasurable = {"nn"}
    assert unmeasurable < set(registry), (
        "the excluded key is not in the registry; this exclusion has gone stale")

    rng = np.random.default_rng(11)
    n = 60
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
    y_reg = pd.Series(X["x0"] * 2 + rng.normal(size=n))
    y_clf = pd.Series((X["x0"] > 0).astype(int))

    measured: Dict[str, bool] = {}
    for key, spec in registry.items():
        if key in unmeasurable:
            continue
        caps = spec.capabilities
        task = "classification" if caps.supports_classification else "regression"
        estimator = spec.factory(task, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X.values,
                          (y_clf if task == "classification" else y_reg).values)
        measured[key] = hasattr(estimator, "coef_")

    assert set(measured) == set(registry) - unmeasurable, (
        f"measured {len(measured)} of {len(registry) - len(unmeasurable)} "
        f"fittable models; a model that raised was silently dropped")

    wrong = {k: (registry[k].capabilities.exposes_coefficients, v)
             for k, v in measured.items()
             if bool(registry[k].capabilities.exposes_coefficients) != v}
    assert not wrong, (
        f"declared vs measured `coef_` disagree: {wrong} (declared, measured). "
        f"The shelf's clause under a recorded inference objective says whether "
        f"there is an association estimate to read off the model, so a wrong "
        f"declaration is the app asserting something false about a model at "
        f"the moment a user picks one.")


# ═══════════ 5 · the page renders it ════════════════════════════════════════

def test_the_page_renders_the_clause_and_the_statement():
    """`AUDIT-008`: a server that composes a sentence nobody renders.

    Driven through the page's own controller rather than read out of the
    payload, because the payload carrying it is what was already true of
    `SHELF.priors` — served since `L32` and rendered nowhere.
    """
    from fastapi.testclient import TestClient

    from turbotab import api, pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    fixture, target, _task, group_col = SHAPES["binary_numeric_repeated"]
    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    for kind, payload in [
            ("set_target", {"column": target}),
            ("set_purpose", {"answer": "inference"}),
            ("set_grain", {"answer": "people_repeat", "group_col": group_col}),
            ("set_repeat_kind", {"kind": "time_points"}),
            ("set_unit_of_analysis", {"unit": "record"}),
            ("set_eligibility", {"answer": "everyone"}),
            ("seal", {"fraction": 0.25})]:
        got = client.post(f"/project/{pid}/decision",
                          json={"kind": kind, "payload": payload})
        assert got.status_code == 200, (kind, got.text[:300])

    served = client.get(f"/project/{pid}/models").json()
    statements = served.get("design") or []
    assert statements, "the route served no design statement to render"
    lowered = [m for g in served.get("groups", []) for m in g.get("models", [])
               if m.get("ranked_lower_by_design")]
    assert lowered, "no served entry was ranked lower, so no clause is on the wire"

    routes = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "interview?step=preprocess",
                 "capabilities", "features", "recipes", "preprocess", "figures",
                 "draft", "manuscript", "models", "training", "instability",
                 "explain", "sensitivity", "checklist",
                 "evidence/plausibility", "evidence/missingness"):
        got = client.get(f"/project/{pid}/{path}")
        routes[f"/project/{pid}/{path}"] = (got.json() if got.status_code == 200
                                            else {})

    shown = PH.run(
        "for (var i = 0; i < 12; i++) "
        "  await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({box: __harness.html('shelfBox') || '',"
        "        statements: __harness.html('shelfDesign') || '',"
        "        list: __harness.html('shelfList') || '',"
        "        calls: __harness.calls().map(function(c){ return c.path; })});",
        routes=routes, search=f"?project={pid}")

    box, region, listing = shown["box"], shown["statements"], shown["list"]
    # POSITIVE CONTROL — the page fetched the shelf at all. Without this an
    # empty region reads as "the clause is missing" when the truth is "the step
    # never rendered", which is a different defect with a different fix.
    assert any(c.endswith("/models") for c in shown["calls"]), (
        f"the page never fetched the shelf; nothing here is about rendering. "
        f"Calls: {shown['calls']}")
    assert box.strip(), "the shelf region rendered empty"

    # THE TWO REGIONS ARE READ SEPARATELY, AND A PROBE IS WHY. Asserting the
    # statement appears anywhere in `shelfBox` passed with the shelf-level
    # renderer reverted, because the SAME recorded sentence is quoted again in
    # every per-entry clause below it. One region satisfying another region's
    # claim is trap #3's shape at the level of the assertion rather than the
    # fixture, and it is only visible under a revert.
    for statement in statements:
        assert statement["quote"][:60] in region, (
            f"the shelf-level statement for {statement['answer']!r} is served "
            f"and not rendered. This is the class `GUIDED-080` measured at six "
            f"surfaces: the server composes a user-facing string and the "
            f"interface never shows it.\nRegion: {region[:1200]}")
        assert statement["effect"][:60] in region, (
            f"the statement's effect clause is not rendered: "
            f"{statement['effect'][:120]!r}\nRegion: {region[:1200]}")

    note = (lowered[0].get("design_notes") or [{}])[0]
    assert note.get("clause"), "a lowered entry was served with no clause"
    assert note["clause"][:60] in listing, (
        f"the clause naming which answer moved {lowered[0]['key']} is served "
        f"and not rendered beside the model it is about.\n"
        f"List: {listing[:1200]}")
    assert note["quote"][:60] in listing, (
        "the recorded sentence the clause quotes is not rendered beside the "
        "model, so the page shows a judgment without the answer it rests on")


# ═══════════ the shape this file does NOT cover ═════════════════════════════
#
# `GUIDED-097`'s obligation, stated rather than left to be noticed.
#
# **Regression is not covered.** Both fixtures are classification, because the
# repeat questions need a group column and the two fixtures that have one are
# both classification tables. The ordering rule reads
# `exposes_coefficients`, which is task-independent, and the regression half of
# the registry declares it the same way — but *the ordering has not been
# observed on a continuous target*, and `ridge`/`lasso`/`elasticnet`/`huber` are
# regression-only, so the regression shelf is a different set of models than
# anything asserted above.
#
# **`not_described` as a unit is exercised only in the never-shortened sweep.**
# Its recorded sentence is the longest of the three and carries an
# `[AUTHOR REQUIRED]` gap; nothing here asserts how that reads inside the
# shelf statement.
#
# **A project restored from an archive is not driven.** `archive.py` restores
# the three fields and the decisions separately, and the "answer without a
# quotable sentence" case above is constructed by hand rather than reached
# through a real restore.
