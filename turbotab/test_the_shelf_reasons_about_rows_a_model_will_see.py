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
        assert (body["n_rows_seen"] + body["n_rows_withheld"]
                + body["n_rows_without_an_outcome"]) == len(project.df), (
            "the three counts do not partition the table, so at least one of "
            "them is about a population the others are not")


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
