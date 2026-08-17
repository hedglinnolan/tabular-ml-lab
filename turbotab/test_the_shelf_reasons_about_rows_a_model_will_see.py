"""`DRIVE-045` — capacity advice derived from rows no model will be fitted on.

Run 5 uploaded a 21,849 × 29 NHANES table. `meds_hbp` is blank on **15,552** of
those rows. The shelf said *"Neural Network — n=20,904 supports the capacity"*
and *"SVC — slow at n=20,904"*; 20,904 is 21,849 − 945, the seal removed and
nothing else. The fit, on the same page, reported **5,352 trained on** —
correctly.

**This is not copy quoting the wrong number.** It is a recommendation *derived*
from it, on the surface a user picks models from. A neural network at n=5,352
with 770 minority events is a different recommendation from one at n=20,904,
and the regression path was right (18,572) only because `bp_sys` has no
missing values — which is to say the defect is invisible on a complete column.

## The two masks, and why there are two

`training_mask` answers *which rows may inform a decision* — the leakage
question, `GUIDED-088` — and dropping the sealed rows answers it. `analysis_mask`
answers *which rows will the model see*, and `training.train` says what that is:
`features[has_y & ~is_test]`. Both are correct about their own question. What
was wrong was one surface asking the second question and reading the first
answer.

**The sweep is the deliverable here rather than the two fixes.** `A3` asked who
else reads the wrong population; the answer was three surfaces, not the two the
row named, and `/recipes` was the one nobody had looked at.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, eventfixture                          # noqa: E402

N = 900
N_LABELED = 300          # two thirds have no outcome, as run 5's table did


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({
        "age": rng.normal(50, 12, N).round(1),
        "bmi": rng.normal(27, 5, N).round(1),
        "sbp": rng.normal(128, 16, N).round(1),
    })
    outcome = pd.Series(rng.choice(["yes", "no"], N, p=[0.7, 0.3]),
                        dtype=object)
    outcome.iloc[N_LABELED:] = None
    frame["treated"] = outcome
    return frame


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


@pytest.fixture(scope="module")
def sealed(client):
    pid = client.post("/project", files={
        "file": ("p.csv", _frame().to_csv(index=False).encode(), "text/csv")}
    ).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])

    decide("set_target", column="treated")
    eventfixture.choose_event_over_http(client, pid, "treated", required=True)
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=0.25)
    return pid, api.STORE.get(pid)


def _trainable(project) -> int:
    """What `training.train` will actually fit on, from pandas."""
    table = project.working_table
    target = str(project.target)
    has_y = table[target].notna()
    sealed = set(project.lockbox["labels"])
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    return int((has_y & ~is_test).sum())


def test_the_fixture_has_rows_no_model_can_be_fitted_on(sealed):
    """**The control, first.** On a complete outcome column the two masks agree
    and every assertion below passes against either. Run 5's regression path
    was right for exactly that reason."""
    _, project = sealed
    assert len(project.training_rows) > _trainable(project), (
        "this fixture has no unlabeled rows, so it cannot tell "
        "`training_rows` from `analysis_rows` and proves nothing")


def test_the_masks_answer_different_questions(sealed):
    """`analysis_mask` is a subset of `training_mask` and the difference is
    exactly the rows with no outcome."""
    _, project = sealed
    target = str(project.target)
    blank = project.df[target].isna()
    assert (project.analysis_mask & ~project.training_mask).sum() == 0, (
        "a row is in the analysis population and not in the training one, "
        "which would mean a held-out row reaching a decision")
    lost = project.training_mask & ~project.analysis_mask
    assert int(lost.sum()) == int((project.training_mask & blank).sum())
    assert len(project.analysis_rows) == _trainable(project)


def test_the_shelf_is_ranked_on_the_rows_a_model_will_see(client, sealed):
    """`DRIVE-045` itself, over the route."""
    pid, project = sealed
    shelf = client.get(f"/project/{pid}/models").json()
    assert shelf["n_rows_seen"] == _trainable(project), (
        f"the shelf ranked on {shelf['n_rows_seen']} rows and the models will "
        f"be fitted on {_trainable(project)}")


def test_the_recipes_divergence_is_measured_on_the_same_rows(client, sealed):
    """**The third instance, and the row named two.**

    `/recipes` measures how two scalings would rescale the columns relative to
    one another, and uses that to decide whether to ASK. The scaler is fitted
    on the analysis population; measuring the divergence over rows with no
    outcome raises or suppresses a question about a fit that will not include
    them.
    """
    pid, project = sealed
    recipes = client.get(f"/project/{pid}/recipes").json()
    assert recipes["n_rows_seen"] == _trainable(project), (
        f"/recipes measured on {recipes['n_rows_seen']} rows; the fit will see "
        f"{_trainable(project)}")


def test_the_two_reasons_a_row_is_excluded_are_reported_apart(client, sealed):
    """**And the fix must not commit the defect it repairs.**

    Narrowing `n_rows_seen` made `n_rows_withheld` — which meant *sealed* —
    silently start meaning *sealed or unusable*. Two counts, each saying which
    it is, because a reader told only the total cannot tell a large seal from a
    column that is mostly empty.
    """
    pid, project = sealed
    for route in ("models", "recipes"):
        body = client.get(f"/project/{pid}/{route}").json()
        assert body["n_rows_withheld"] == len(project.lockbox["labels"]), (
            f"/{route} reports {body['n_rows_withheld']} withheld; the seal "
            f"holds {len(project.lockbox['labels'])}")
        assert body["n_rows_without_an_outcome"] == N - N_LABELED, (
            f"/{route} reports {body['n_rows_without_an_outcome']} rows with "
            f"no outcome; the file has {N - N_LABELED}")
        # `DRIVE-051`. THE PARTITION IS NOW OVER THE DISJOINT FIELD, and this
        # assertion is an IDENTITY rather than a check — see
        # `test_the_overlap_between_the_seal_and_the_blanks_is_served`, which
        # is the one that can still go red. It is kept because a breakdown
        # presented as a breakdown should be asserted to be one, and because it
        # would catch a producer that stopped deriving these from one frame.
        assert (body["n_rows_seen"] + body["n_rows_withheld"]
                + body["n_rows_available_without_an_outcome"]
                ) == len(project.df), (
            "the three counts do not partition the table, so at least one of "
            "them is about a population the others are not")
        # This fixture has no sealed row without an outcome — the seal draws
        # only from outcome-bearing rows — so here the two blank counts agree.
        assert body["n_rows_withheld_without_an_outcome"] == 0
        assert (body["n_rows_available_without_an_outcome"]
                == body["n_rows_without_an_outcome"])


@pytest.fixture(scope="module")
def sealed_then_blanked(client):
    """A project where a SEALED row has lost its outcome. `DRIVE-051`.

    The `sealed` fixture above cannot produce this state and no fixture can:
    `engine.draw_holdout` opens with `eligible = df.index[y.notna()]`, so at
    seal time the intersection is zero by construction. **The overlap requires
    a row to LOSE its outcome after being sealed**, which is a mechanism rather
    than an accident, and there are two live paths to it. This drives the
    ordinary one — a post-seal data-quality repair — because a probe written
    around the other (changing the target after the seal) invites the reading
    that this is an edge case. It is not.

    `set_impossible_missing` writes NaN into whatever column it is pointed at,
    including the target, and it is not barrier-gated:
    `PRE_BARRIER_ONLY_FIXES` is only `{promote_header, melt_repeated}`.
    """
    n_rows = 600
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({
        "age": rng.normal(50, 12, n_rows).round(1),
        "bmi": rng.normal(27, 5, n_rows).round(1),
    })
    # Sixty physiologically impossible systolic readings, scattered, so the
    # repair takes some sealed rows and some not.
    values = list(rng.normal(128, 16, n_rows - 60).round(1)) + [5.0] * 60
    frame["sbp"] = rng.choice(values, n_rows, replace=False)
    pid = client.post("/project", files={
        "file": ("impossible.csv", frame.to_csv(index=False).encode(),
                 "text/csv")}).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])

    decide("set_target", column="sbp")
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=0.25)
    project = api.STORE.get(pid)
    assert project.barrier_raised
    assert project.n_rows_held_out_without_an_outcome == 0, (
        "the seal already took a row with no outcome, which `draw_holdout` "
        "makes impossible — this fixture's premise has moved")
    project.set_impossible_missing("sbp")
    return pid, project


def test_the_overlap_between_the_seal_and_the_blanks_is_served(
        client, sealed_then_blanked):
    """`DRIVE-051`, and this is the assertion that replaces the tautology.

    The obvious repair — narrowing `n_rows_without_an_outcome` to the training
    population — makes `seen + withheld + that == len(df)` true for every
    dataset, so the guard could never fail again. That is the same loss as
    relaxing it to `<=`, arriving by the other door.

    So the disjoint field is served AND the overlap is served, and what is
    asserted is that the three blank-outcome counts agree with each other.
    They are three independently computed properties over three different
    masks; nothing makes them agree except being right.
    """
    pid, project = sealed_then_blanked
    for route in ("models", "recipes"):
        body = client.get(f"/project/{pid}/{route}").json()

        # THE CONTROL. Without a real overlap every assertion below is
        # satisfied by the pre-fix implementation too.
        overlap = body["n_rows_withheld_without_an_outcome"]
        assert overlap > 0, (
            f"/{route} reports no sealed row without an outcome, so this "
            f"fixture no longer reaches the state `DRIVE-051` is about")

        blank_total = body["n_rows_without_an_outcome"]
        available = body["n_rows_available_without_an_outcome"]
        assert available + overlap == blank_total, (
            f"/{route}: {available} unsealed blanks + {overlap} sealed blanks "
            f"= {available + overlap}, and the table reports {blank_total} "
            f"blanks in total. Three properties over three masks, disagreeing")

        # And the pre-fix sum is still WRONG, which is what makes the disjoint
        # field necessary rather than cosmetic. Asserted so a later edit cannot
        # quietly restore the old field to the breakdown.
        naive = (body["n_rows_seen"] + body["n_rows_withheld"] + blank_total)
        assert naive == len(project.df) + overlap, (
            f"the old three counts sum to {naive} against {len(project.df)} "
            f"rows; the overshoot should be exactly the {overlap} sealed rows "
            f"with no outcome and nothing else")
        assert (body["n_rows_seen"] + body["n_rows_withheld"] + available
                ) == len(project.df)

        # Against pandas, not against another server-side derivation.
        target = str(project.target)
        blank = project.df[target].isna()
        assert overlap == int(((~project.training_mask) & blank).sum())
        assert available == int((project.training_mask & blank).sum())


def test_a_post_seal_repair_cannot_drop_a_sealed_row_unnoticed(client):
    """`TEST-104`. `assert_identity_intact` had ONE call site and `apply_fix`
    was not it.

    Driven at HEAD: sealed 75 rows, dropped 10 of them through `apply_fix`,
    nothing raised, `n_rows_held_out` fell to 65 while `lockbox['labels']`
    stayed 75, and `/models` served `withheld=65`. **In that state the three
    counts still summed to `len(df)`**, so the partition guard beside it stayed
    green while `withheld == len(lockbox["labels"])` was false by 10 — the
    guard was not merely fixture-kind, it was satisfiable by a state that is
    wrong in a different way.

    The check now lives in `_install`, the one door every frame write goes
    through, so a method added next loop inherits it.
    """
    pid, project = _sealed_project(client)
    sealed_labels = list(project.lockbox["labels"])
    assert len(sealed_labels) > 10

    keep = project.df.drop(index=sealed_labels[:10])
    with pytest.raises(Exception) as caught:
        project.apply_fix(keep, "made-up-finding", True, "drop", "dropped")
    assert "no longer in the table" in str(caught.value), (
        f"the drop was refused for the wrong reason: {caught.value}")

    # The refusal is not a blanket ban on dropping rows after the seal —
    # `drop_rows` and `drop_empty_rows` are deliberately allowed on either
    # side. Only sealed rows are protected.
    unsealed = [i for i in project.df.index if i not in set(sealed_labels)]
    project.apply_fix(project.df.drop(index=unsealed[:5]), "f2", True,
                      "drop_rows", "dropped five unsealed rows")
    assert project.n_rows_held_out == len(project.lockbox["labels"])


def _sealed_project(client):
    n_rows = 400
    rng = np.random.default_rng(21)
    frame = pd.DataFrame({
        "age": rng.normal(50, 12, n_rows).round(1),
        "bmi": rng.normal(27, 5, n_rows).round(1),
        "sbp": rng.normal(128, 16, n_rows).round(1),
    })
    pid = client.post("/project", files={
        "file": ("identity.csv", frame.to_csv(index=False).encode(),
                 "text/csv")}).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])

    decide("set_target", column="sbp")
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=0.25)
    return pid, api.STORE.get(pid)


def test_the_capacity_clauses_cite_the_same_n_the_shelf_ranked_on(client, sealed):
    """The clauses are the thing a user reads, and they are composed from the
    profile rather than from the number served beside them — so this asserts
    the sentence, not the field. `DRIVE-045` was a wrong number *inside a
    recommendation*, and a served count that agreed while the prose did not
    would be the same defect with a passing test over it."""
    import re

    pid, project = sealed
    shelf = client.get(f"/project/{pid}/models").json()
    trainable = _trainable(project)
    stale = len(project.training_rows)
    for group in shelf["groups"]:
        for model in group["models"]:
            for cited in re.findall(r"n\s*=\s*([\d,]+)", model["concern"]):
                seen = int(cited.replace(",", ""))
                assert seen != stale, (
                    f"{model['key']}'s concern cites n={seen}, which is the "
                    f"pre-fix count of rows including {stale - trainable} with "
                    f"no outcome: {model['concern']}")
                assert seen == trainable, (
                    f"{model['key']}'s concern cites n={seen}; the fit will "
                    f"see {trainable}: {model['concern']}")
