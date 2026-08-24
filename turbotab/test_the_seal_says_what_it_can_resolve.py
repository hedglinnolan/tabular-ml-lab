"""`GUIDED-102` — the seal states what a holdout this size can resolve.

The shipped step contradicted its own specification. `engine.draw_holdout`
takes `fraction=0.15` with no *n* term; on `metabolomics_untargeted.csv`
(n=80, `PRODUCT_VISION.md`'s own worked case) that is 11 rows, and the seal
said *"11 rows (15%) are held out"* and nothing else. §04 specifies the other
half: **state the instrument's resolution and let the researcher judge their
claim against it.**

The four rules are tested as four groups below, because three of them are
about what the app must NOT say and a module like this fails by growing a
verdict rather than by computing a wrong number.

`GUIDED-097` — THE FIXTURE RULE. Every claim about a journey step runs against
at least two fixtures of different target shape, and the shapes not covered are
named in the file.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from turbotab import draft as draft_mod
from turbotab import resolution as R

#: `GUIDED-097`. Two target shapes, both real fixtures, both driven end to end.
TARGET_SHAPES = {
    "binary classification": ("metabolomics_untargeted.csv", "responder",
                              "classification"),
    # `survey_instrument.csv` rather than `clinical_longitudinal.csv`, and the
    # reason is worth recording: the longitudinal fixture has ~3 rows per
    # `subject_id`, so `set_grain("one_row_per_person")` raises
    # `GrainContradiction` — correctly. Sealing it honestly means walking the
    # repeat chain, which is a different step's test. This fixture is
    # cross-sectional, so the seal here exercises the seal.
    "continuous regression": ("survey_instrument.csv", "age", "regression"),
}

#: What is NOT covered, said out loud rather than left to be discovered.
#:
#: MULTICLASS is the one that matters and it is a real gap: `_push_because`'s
#: first condition reasons about the distance from a coin flip to a perfect
#: classifier, which is 0.5 for a binary discrimination statistic and is NOT
#: 0.5 for a k-class problem — chance is 1/k, so the trigger would fire late.
#: The count of held-out events also becomes k counts rather than one. No
#: fixture in this repository has a multiclass target, so this is filed rather
#: than guessed at: the module reports on multiclass (the arithmetic on rows is
#: unchanged) and its PUSH trigger is calibrated for binary only.
#:
#: SURVIVAL is not covered because the app has no survival task type; the
#: relevant denominator there is events rather than rows, and `n_test` would be
#: the wrong number to reason from entirely.
SHAPES_NOT_COVERED = [
    "multiclass classification — the chance-to-perfect distance is 1/k, not "
    "0.5, so the push trigger is calibrated for binary targets only",
    "survival / time-to-event — no task type exists, and the resolving "
    "quantity would be events rather than rows",
]


def _frame(name, target):
    df = pd.read_csv(f"turbotab/sample_data/{name}")
    return df[df[target].notna()].copy()


def _split(df, fraction=0.15, seed=42):
    """The labels `engine.draw_holdout` would produce at this fraction — the
    same arithmetic, so the counts under test are the shipped ones."""
    rng = np.random.default_rng(seed)
    idx = list(df.index)
    rng.shuffle(idx)
    return idx[:max(1, int(round(len(idx) * fraction)))]


# ═══════════ RULE 2 · THE INSTRUMENT, NEVER A VERDICT ═══════════

#: Every way this module could put on the nicer suit. `METABOLOMICS_PACK.md`
#: §10 names post-hoc power as an anti-pattern flatly, and an app that computed
#: one while presenting as the tool that catches them would be worse than one
#: that stayed silent.
FORBIDDEN = [
    "underpowered", "under-powered", "power", "powered",
    "adequate", "inadequate", "insufficient", "sufficient",
    "too small", "too few", "not enough", "should", "must",
    "recommend", "advise", "consider increasing", "unreliable",
    "invalid", "meaningless", "worthless", "cannot be trusted",
]


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_it_states_the_instrument_and_never_a_verdict(shape):
    """No prose this module emits may judge the study.

    This is the test that would catch the module drifting into the thing it
    exists to avoid, and it reads EVERY string rather than the headline,
    because the anti-pattern would arrive in a helpful aside.
    """
    name, target, task = TARGET_SHAPES[shape]
    df = _frame(name, target)
    s = R.statement(df, target, task, _split(df))

    prose = " ".join(str(s.get(k) or "") for k in
                     ("headline", "because", "sentence")).lower()
    for word in FORBIDDEN:
        assert word not in prose, (
            f"{shape}: the resolution statement says '{word}'. This module "
            f"reports what the instrument can resolve; a judgment on the "
            f"study is post-hoc power in a nicer suit "
            f"(METABOLOMICS_PACK.md §10). Said: {prose!r}")

    # And it says so itself, in a sentence the user reads.
    assert "not about your study" in s["not_a_verdict"]
    assert "cannot say whether this design is adequate" in s["not_a_verdict"]


# ═══════════ RULE 1 · IT NEVER SAYS "DON'T" ═══════════

def test_the_seal_is_never_refused_or_gated_by_this():
    """A researcher who wants a holdout at n=80 gets one.

    Driven against the real project rather than the module, because the rule
    is about the SEAL and a module that returns a peaceable dict while the
    seal raises would satisfy the unit and fail the requirement.
    """
    from turbotab.project import AnalysisProject

    name, target, task = TARGET_SHAPES["binary classification"]
    df = _frame(name, target)
    p = AnalysisProject.from_dataframe(df, name)
    p.target, p.task_type = target, task
    p.set_grain("one_row_per_person")
    p.set_eligibility("everyone")

    labels = _split(df)
    decision = p.seal_lockbox(              # must not raise
        labels, fraction=len(labels) / len(df))

    assert p.barrier_raised
    assert p.lockbox["n_test"] == len(labels)
    res = p.lockbox["resolution"]
    assert res is not None, (
        "the seal succeeded and recorded no resolution statement")
    assert res["push"] is True, (
        "n=72 with 11 held out is PRODUCT_VISION §04's own worked case")
    # The seal's basis is untouched: a statement BESIDE it, not a fifth value.
    assert p.lockbox["seal_basis"] == "cross_sectional"
    # The seal's own sentence is unchanged: the arithmetic rides in the
    # payload, and the recorded text still says what it always said.
    assert "sealed before exploration" in decision.text
    assert "interval" not in decision.text.lower()
    assert decision.payload["resolution"] == res


# ═══════════ RULE 3 · IT FIRES ONLY WHEN STARK ═══════════

#: Every fixture in the repository, with the split the app would draw. The
#: point of running ALL of them is that "not wallpaper" is a claim about the
#: population of datasets, which one fixture cannot check.
ALL_FIXTURES = [
    ("metabolomics_untargeted.csv", "responder", "classification"),
    ("clinic_visits.csv", "outcome", "classification"),
    ("clinic_visits.csv", "hba1c", "regression"),
    ("leaky_sepsis.csv", "sepsis", "classification"),
    ("clinical_longitudinal.csv", "sbp", "regression"),
    ("survey_instrument.csv", "age", "regression"),
]


def test_the_card_is_not_wallpaper():
    """It fires on a minority of the fixtures, and on the right one.

    A card that appears on every dataset is wallpaper (`PRODUCT_VISION.md` §04,
    *push the notable*), and a trigger that never fires is the shipped defect
    unfixed. Both directions are asserted.
    """
    fired = []
    for name, target, task in ALL_FIXTURES:
        df = _frame(name, target)
        if R.statement(df, target, task, _split(df))["push"]:
            fired.append(f"{name}:{target}")

    assert fired, "the trigger never fires; GUIDED-102 would be unfixed"
    assert len(fired) < len(ALL_FIXTURES) / 2, (
        f"fired on {len(fired)} of {len(ALL_FIXTURES)} fixtures: {fired}. "
        f"A card that appears on every dataset is wallpaper.")
    assert fired == ["metabolomics_untargeted.csv:responder"], (
        f"expected exactly PRODUCT_VISION §04's worked case (n=80 → 11 rows); "
        f"got {fired}")


def test_the_trigger_is_derived_from_arithmetic_not_from_a_round_number():
    """The threshold is a consequence, not a choice.

    The rule is *the widest 95% interval this holdout can produce is wider than
    the whole distance from a coin flip to a perfect classifier*. Every term
    comes from somewhere: 0.5 is the metric's own scale, 1.96 is the interval
    named, 0.5 again is the worst-case SD of a proportion. The boundary falls
    where the algebra puts it — 2·1.96·0.5/√n > 0.5 ⟺ n < 15.37 — and this
    test recomputes that from the constants rather than hard-coding 15, so
    changing the interval to 90% moves the boundary here too.
    """
    boundary = (2 * R.Z95 * R.WORST_CASE_SD / 0.5) ** 2
    assert 15 < boundary < 16, boundary

    df = _frame(*TARGET_SHAPES["binary classification"][:2])
    below = math.floor(boundary)
    assert R.statement(df, "responder", "classification",
                       _split(df, fraction=below / len(df)))["push"] is True
    above = math.ceil(boundary) + 1
    assert R.statement(df, "responder", "classification",
                       _split(df, fraction=above / len(df)))["push"] is False


def test_a_class_missing_from_the_holdout_is_undefined_not_imprecise():
    """The second condition, and it is a different kind of fact.

    With fewer than two of a class in the holdout, sensitivity does not have a
    wide interval — it has no value. Constructed rather than drawn from a
    fixture, because no fixture is imbalanced enough and inventing one to make
    a trigger fire would be tuning the trigger.
    """
    df = pd.DataFrame({
        "y": [1] * 3 + [0] * 97,
        "x": list(range(100)),
        "g": ["a", "b"] * 50,
    })
    # 40 held out, so the interval condition cannot be what fires.
    labels = [i for i in range(100) if i >= 60]
    assert 1 not in df.loc[labels, "y"].unique() or True
    s = R.statement(df, "y", "classification", labels)
    assert s["n_test"] == 40
    assert s["widest_interval"] < 0.5, "the interval condition must not fire here"
    assert s["push"] is True
    assert "undefined rather than imprecise" in s["because"]


# ═══════════ RULE 4 · IT IS RECORDED, AND IT REACHES THE MANUSCRIPT ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_sentence_reaches_the_methods_section(shape):
    """Recorded at the seal, quoted in the draft — including when not pushed.

    `push` decides what interrupts a user mid-journey; a methods section
    interrupts nobody. Suppressing the line on the comfortable studies would
    make its presence a verdict.
    """
    name, target, task = TARGET_SHAPES[shape]
    df = _frame(name, target)
    labels = _split(df)
    s = R.statement(df, target, task, labels)

    d = {"kind": "seal_lockbox", "subject": target,
         "text": "A test set of N rows was sealed before exploration.",
         "payload": {"resolution": s}}
    line = draft_mod._sentence_for(d)

    assert s["sentence"] in line, (
        f"{shape}: the resolution sentence did not reach the methods section")
    assert "sealed before exploration" in line, (
        "the seal's own sentence carries the BASIS, which the arithmetic "
        "cannot supply; both belong")
    assert str(s["n_test"]) in line and str(s["parameters"]["total"]) in line

    for word in FORBIDDEN:
        assert word not in line.lower(), f"{shape}: methods line says '{word}'"


def test_the_manuscript_line_appears_even_when_the_card_is_quiet():
    """The asymmetry, asserted directly on the fixture that does not push."""
    name, target, task = TARGET_SHAPES["continuous regression"]
    df = _frame(name, target)
    s = R.statement(df, target, task, _split(df))
    assert s["push"] is False
    line = draft_mod._sentence_for(
        {"kind": "seal_lockbox", "subject": target, "text": "Sealed.",
         "payload": {"resolution": s}})
    assert s["sentence"] in line


def test_the_statement_travels_with_the_lockbox_not_the_current_frame():
    """Recomputed later it would describe a table that has since changed.

    The seal is the moment the cohort stops changing, so the statement is
    stored on the lockbox. This drives the actual invalidation: trim rows after
    the seal and the recorded numbers must not move.
    """
    from turbotab.project import AnalysisProject

    name, target, task = TARGET_SHAPES["binary classification"]
    df = _frame(name, target)
    p = AnalysisProject.from_dataframe(df, name)
    p.target, p.task_type = target, task
    p.set_grain("one_row_per_person")
    p.set_eligibility("everyone")
    p.seal_lockbox(labels_ := _split(df),
                   fraction=len(labels_) / len(p.df))
    before = dict(p.lockbox["resolution"])

    survivors = [i for i in p.df.index][:40]
    p.df = p.df.loc[survivors]

    assert p.lockbox["resolution"] == before, (
        "the recorded resolution changed when the frame did; it describes the "
        "sealed cohort, not the current one")


# ═══════════ THE ARITHMETIC ITSELF ═══════════

def test_parameters_are_counted_not_variables():
    """`CLINICAL_SURVEY_PACK.md` §A5.4, and it is the count people get wrong.

    A 5-level factor is 4 parameters, not 1 column. The count is over the
    columns this app will actually hand the model, so it describes the fit.
    """
    df = pd.DataFrame({
        "y": [0, 1] * 10,
        "num": range(20),
        "five_level": ["a", "b", "c", "d", "e"] * 4,
        "binary": ["p", "q"] * 10,
    })
    p = R.candidate_parameters(df, "y")
    assert p["numeric"] == 1
    assert p["from_categorical"] == 4 + 1
    assert p["total"] == 6, (
        "three feature columns, six parameters — counting variables would "
        "give three and understate the sample size the study needs")
    assert p["evidence_status"] == "SETTLED"
    assert "CLINICAL_SURVEY_PACK" in p["source"]


def test_a_holdout_of_zero_returns_nothing_rather_than_a_number():
    """Return nothing rather than a wrong value. An interval computed from no
    rows is a number asserting precision it does not have."""
    df = pd.DataFrame({"y": [0, 1] * 10, "x": range(20)})
    s = R.statement(df, "y", "classification", [])
    assert s["n_test"] == 0
    assert s["widest_interval"] is None
    assert s["step_per_row"] is None
    assert "interval" not in s["sentence"]


def test_the_counts_describe_the_split_that_was_drawn():
    """Not the fraction that was requested — the same reason `draw_holdout`
    reports the achieved row fraction (`IMPORT-255`)."""
    df = _frame("metabolomics_untargeted.csv", "responder")
    labels = list(df.index)[:7]
    s = R.statement(df, "responder", "classification", labels)
    assert s["n_test"] == 7
    assert s["n_train"] == len(df) - 7
    assert s["events_held_out"] + s["non_events_held_out"] == 7


# ═══════════ THE PAGE, DRIVEN ═══════════
#
# The claim *the card reaches the reader* is a claim about behavior, so it is
# driven rather than asserted against the source. `GUIDED-080`'s class is
# exactly this failure — the server composes a user-facing string and the
# interface never renders it — and this module would land in it silently: the
# statement would be on the record, in the manuscript, and invisible.

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import pageharness as H                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

_needs_js = pytest.mark.skipif(not H.available(),
                               reason="no JS engine on this machine")


def _sealed_project(name, target, task, fraction=0.15):
    """A real API project, sealed, so `/project/{id}` serves the real
    `_disclosures`. The seal is set up directly rather than by walking the
    interview because the subject under test is the disclosure band, not the
    route to it."""
    from fastapi.testclient import TestClient

    from turbotab import api
    client = TestClient(api.app)
    with open(DATA / name, "rb") as fh:
        created = client.post("/project",
                              files={"file": (name, fh, "text/csv")}).json()
    p = api.STORE.get(created["id"])
    p.df = p.df[p.df[target].notna()].copy()
    p.target, p.task_type = target, task
    p.set_grain("one_row_per_person")
    p.set_eligibility("everyone")
    sealed = _split(p.df, fraction=fraction)
    p.seal_lockbox(sealed, fraction=len(sealed) / len(p.df))
    return client, client.get(f"/project/{created['id']}").json()


def _routes(client, project):
    pid = project["id"]
    return {
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": [], "steps": []},
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": {"columns": []},
        f"/project/{pid}/draft": {"paragraphs": []},
        f"/project/{pid}/gaps": {"gaps": []},
    }


@_needs_js
def test_the_card_reaches_the_reader():
    """GUIDED-080's class, checked rather than assumed."""
    name, target, task = TARGET_SHAPES["binary classification"]
    client, project = _sealed_project(name, target, task)
    assert project["disclosures"]["resolution"]["push"] is True, (
        "the server did not serve a pushed resolution; the page claim below "
        "would be vacuous")

    out = H.run("__emit(__harness.html('disc-seal'));",
                routes=_routes(client, project),
                search=f"?project={project['id']}")
    assert out, "the seal disclosure row did not render at all"
    assert "disc-res" in out, (
        "the server composed a resolution statement and the page rendered "
        "none of it — GUIDED-080 exactly")
    res = project["disclosures"]["resolution"]
    assert res["headline"] in out
    assert res["not_a_verdict"] in out, (
        "the card rendered its number without the sentence that says it is "
        "not a verdict on the study")
    assert str(res["parameters"]["total"]) in out


@_needs_js
def test_a_quiet_holdout_renders_no_card_at_all():
    """The positive control for the assertion above, and rule 3 on the page.

    Absent, not blank: a labeled empty region would read as an answer of
    nothing, and a card on every dataset is the wallpaper the trigger exists
    to prevent.
    """
    name, target, task = TARGET_SHAPES["continuous regression"]
    client, project = _sealed_project(name, target, task)
    assert project["disclosures"]["resolution"]["push"] is False

    out = H.run("__emit(__harness.html('disc-seal'));",
                routes=_routes(client, project),
                search=f"?project={project['id']}")
    assert out, "the seal disclosure row did not render at all"
    assert "sealed" in out, "the seal itself must still render"
    assert "disc-res" not in out, (
        "a holdout of 90 rows raised the card; it fires only when stark")


# ═══════════ L40-A1 · `GUIDED-125` — THE TRIGGER LEARNS ITS ARITY ═══════════
#
# **A DELIBERATE EXCEPTION to *never move a threshold in the same loop as the
# change that pressured it*.** `LOOP.md` §06.2 prescribes exactly this run: the
# quantity being gated was wrong, the correction is made on a PASSING run, and
# the reasoning is recorded before it is load-bearing. After a breach the same
# correction would be indistinguishable from relaxing a gate under pressure.

def test_the_informative_range_is_derived_from_the_arity():
    """`1 − 1/k`, generalized rather than special-cased at k = 3.

    A classifier that guesses is right `1/k` of the time, so the distance
    between chance and perfect is `1 − 1/k`. The original derivation used a
    constant 0.5, which IS `1 − 1/k` at k = 2 — it was right for every fixture
    that existed when it was written, and wrong the moment one had three
    classes.
    """
    assert R.informative_range(2) == 0.5
    assert round(R.informative_range(3), 4) == 0.6667
    assert round(R.informative_range(4), 4) == 0.75
    # UNKNOWN ARITY IS THE WIDEST RANGE, so it can only make the card fire
    # LATER. An unknown that made a warning more likely would be the app
    # asserting something about a study it could not see.
    assert R.informative_range(None) == 1.0
    assert R.informative_range(0) == 1.0
    assert R.informative_range(1) == 1.0


def test_the_boundary_moves_down_with_k_and_the_numbers_are_stated():
    """Recomputed from the module's own constants, so changing the interval to
    90% moves every boundary here too.

    **8.6 at k = 3, not 9.5.** The L39 report rounded it loosely and the L40
    prompt quoted that back; the arithmetic is written out in
    `_push_because`'s docstring so the number cannot drift again.
    """
    def boundary(k):
        return (2 * R.Z95 * R.WORST_CASE_SD / R.informative_range(k)) ** 2

    assert round(boundary(2), 1) == 15.4
    assert round(boundary(3), 1) == 8.6
    assert round(boundary(4), 1) == 6.8
    assert boundary(3) < boundary(2), (
        "a wider informative range must move the boundary DOWN: a three-class "
        "model has more room between chance and perfect, so a given interval "
        "width says relatively less")

    # AND THE FIRING POINTS, which are what a reader actually experiences.
    # STRATIFIED, so condition B (a class with fewer than two held-out rows)
    # cannot fire and mask the interval condition this test is about.
    df = pd.read_csv("turbotab/sample_data/multiclass_stage.csv")
    by_class = {k: list(g.index) for k, g in df.groupby("disease_stage")}
    for n_test, expected in ((8, True), (9, False)):
        labels, i = [], 0
        while len(labels) < n_test:
            for level in sorted(by_class):
                if len(labels) < n_test and i < len(by_class[level]):
                    labels.append(by_class[level][i])
            i += 1
        s = R.statement(df, "disease_stage", "classification", labels)
        assert s["n_classes"] == 3
        assert s["push"] is expected, (
            f"k=3 with {n_test} held-out rows: push={s['push']}, expected "
            f"{expected} — the boundary is 8.6")


def test_the_card_states_k_where_a_reader_can_see_it():
    """*Say the number.* A threshold that changes with the outcome's arity and
    does not say so is a threshold nobody can check."""
    df = pd.read_csv("turbotab/sample_data/multiclass_stage.csv")
    s = R.statement(df, "disease_stage", "classification", list(df.index)[:8])

    assert s["n_classes"] == 3
    assert s["chance"] == 0.3333
    assert round(s["informative_range"], 4) == 0.6667
    for text in (s["headline"], s["because"], s["sentence"]):
        assert "3 classes" in text, (
            f"the arity is not stated where the reader meets the number: "
            f"{text!r}")
    assert "33%" in s["headline"]
    # AND IT IS STILL NOT A VERDICT.
    for word in FORBIDDEN:
        assert word not in (s["headline"] + s["because"] + s["sentence"]).lower()


def test_the_binary_case_is_unchanged():
    """The correction must not move a boundary that was already right. `1 − 1/2`
    is 0.5, so every binary project behaves exactly as it did before."""
    df = _frame(*TARGET_SHAPES["binary classification"][:2])
    for n_test, expected in ((15, True), (16, False)):
        s = R.statement(df, "responder", "classification",
                        list(df.index)[:n_test])
        assert s["n_classes"] == 2
        assert s["push"] is expected, f"binary at {n_test}: {s['push']}"


def test_a_class_count_travels_and_a_class_label_does_not():
    """`GUIDED-102`'s own correction still holds: the archive guard refuses a
    serialized project carrying a cell value, so what travels is HOW MANY
    classes rather than which."""
    df = pd.read_csv("turbotab/sample_data/multiclass_stage.csv")
    s = R.statement(df, "disease_stage", "classification", list(df.index)[:20])
    blob = repr(s)
    for label in df["disease_stage"].unique():
        assert str(label) not in blob, (
            f"the class label {label!r} is in the resolution statement, which "
            f"`archive.assert_no_participant_data` refuses")


def test_a_middle_class_missing_from_the_holdout_is_caught():
    """Condition B, generalized. The first version checked the minority class
    and its complement — every class when k = 2, and two of three when k = 3 —
    so a three-class holdout missing its MIDDLE class passed silently."""
    df = pd.read_csv("turbotab/sample_data/multiclass_stage.csv")
    middle = df["disease_stage"].value_counts().index[1]
    # A holdout large enough that the interval condition cannot fire, and
    # deliberately missing one class.
    labels = [i for i in df.index if df.at[i, "disease_stage"] != middle][:40]

    s = R.statement(df, "disease_stage", "classification", labels)
    assert s["n_classes"] == 3
    assert s["widest_interval"] < s["informative_range"], (
        "the interval condition fires here, so this cannot show condition B")
    assert s["classes_with_fewer_than_two_held_out"] >= 1
    assert s["push"] is True
    assert "outcome classes" in s["because"]
    assert "undefined rather than imprecise" in s["because"]
