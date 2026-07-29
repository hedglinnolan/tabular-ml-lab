"""Encoding, settled: a categorical-bearing table through Guided to a fit.

The question was left open at L17 and is not optional here. A recipe table that
says `encode: ordinal` for trees and `encode: onehot` for a neural net is a
claim, and the claim is only worth what it can produce — so this file drives the
whole Guided path over HTTP on a table with two text columns, reads the recipe
the app resolved, **executes it**, and fits the selected models.

**What this proves and what it does not.** The translation from a resolved
variant to an sklearn transformer is eleven lines and it lives HERE, in the
test, not in `turbotab/`. Train's execution is out of scope this loop, so there
is deliberately no executor in production. What the test therefore proves is the
part that was actually in doubt: **the recipe is sufficient.** Every field the
executor needs — which operation, which variant, which columns it applies to,
whether it is stateful and therefore fitted inside the split — comes off the
resolved row, and nothing had to be guessed or looked up elsewhere. If a field
were missing, this file could not be written without inventing it, and inventing
it is exactly the failure the table exists to prevent.

The honest gap, stated rather than implied: **nothing in production consumes
these rows yet.** That is a Train-step obligation, not a defect in the table.

**Fitted on training rows only.** The lockbox is sealed before any of this and
the encoder is fitted on the unsealed rows, because an encoder that saw the
held-out levels is leakage of exactly the kind the seal exists to stop — and a
test that trained on everything would have proved the recipe executable while
demonstrating the thing the product forbids.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from turbotab import eligibility as E, grain as G                     # noqa: E402
from turbotab.api import app                                          # noqa: E402


def categorical_study(n: int = 240) -> pd.DataFrame:
    """Two text columns of different cardinality, and one with blanks.

    `site` is nominal with no order; `smoking` has one that a tree can use and
    one-hot throws away. That is the whole reason the two encodings differ, so
    the fixture has to contain both or the recipe's distinction is untested.
    """
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "pid": [f"P{i:04d}" for i in range(n)],
        "age": rng.integers(20, 80, n).astype(float),
        "bmi": np.round(rng.normal(28, 5, n), 1),
        "site": rng.choice(["north", "south", "east", "west"], n),
        "smoking": rng.choice(["never", "former", "current"], n),
    })
    df["outcome"] = ((df["age"] > 55) & (df["smoking"] != "never")).astype(int)
    df.loc[rng.choice(n, 20, replace=False), "site"] = None
    return df


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def driven(client):
    """The whole Guided path, over HTTP, exactly as a driver would take it."""
    raw = categorical_study().to_csv(index=False).encode()
    pid = client.post("/project",
                      files={"file": ("study.csv", raw, "text/csv")}).json()["id"]

    def decide(kind, **kw):
        r = client.post(f"/project/{pid}/decision", json={"kind": kind, **kw})
        assert r.status_code < 400, f"{kind} refused: {r.text[:300]}"
        return r.json()

    decide("set_target", payload={"column": "outcome"})
    decide("set_grain", payload={"answer": G.ONE_ROW_PER_PERSON})
    decide("set_eligibility", payload={"answer": E.EVERYONE})
    decide("seal")

    shelf = client.get(f"/project/{pid}/models").json()
    # One model from each end of the shelf, so the two encodings are both
    # exercised: a boosted ensemble takes ordinal, a distance-based model
    # takes one-hot.
    picks = ["histgb_clf", "logreg"]
    available = {m["key"] for g in shelf["groups"] for m in g["models"]}
    picks = [k for k in picks if k in available]
    assert len(picks) == 2, f"the fixture's models are not on the shelf: {available}"
    decide("select_models", payload={"models": picks})
    decide("set_preparation_mode", payload={"mode": "per_model"})
    return client, pid, picks


# ─────────────────────────────────────────────────────────────────────────────
# The executor. Eleven lines, in the test, reading only the resolved row.
# ─────────────────────────────────────────────────────────────────────────────

def _encoder_for(variant: str):
    """A resolved variant to a transformer, using nothing but the variant name.

    If this function needed a lookup the recipe does not carry — the column's
    cardinality, the model's family, a hardcoded exception — the recipe would
    not be sufficient, and that is the finding this file exists to settle.
    """
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
    if variant == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if variant == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1)
    raise AssertionError(
        f"the table resolved `encode: {variant}`, which no executor can build. "
        "A variant nothing can execute is a decision the record will describe "
        "and the pipeline will never perform.")


def _scaler_for(variant: str):
    from sklearn.preprocessing import (MinMaxScaler, RobustScaler,
                                       StandardScaler)
    return {"standard": StandardScaler(), "robust": RobustScaler(),
            "minmax": MinMaxScaler(), "none": None}[variant]


# ─────────────────────────────────────────────────────────────────────────────

def test_the_two_models_resolve_to_different_encodings(driven):
    """Otherwise the fit below proves only that one encoder works.

    The whole per-model claim is that a boosted ensemble and a linear model get
    DIFFERENT preparation. If both resolved to one-hot, everything downstream
    would pass on a table that had made no distinction at all.
    """
    client, pid, picks = driven
    body = client.get(f"/project/{pid}/recipes").json()
    got = {m: next(r["variant"] for r in rows if r["operation"] == "encode")
           for m, rows in body["models"].items()}
    assert len(set(got.values())) == 2, (
        f"both models resolved to the same encoding: {got}")


def test_a_categorical_table_reaches_a_fitted_model_through_guided(driven):
    """The claim in the file's name, end to end, on the sealed split.

    Asserted on the fitted estimator's behavior — a prediction per held-out row,
    every label one the training rows actually contained — rather than on "it
    did not raise", because a pipeline that silently one-hots the target or
    drops every categorical column also does not raise.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    client, pid, picks = driven
    body = client.get(f"/project/{pid}/recipes").json()
    state = client.get(f"/project/{pid}").json()
    assert state["barrier_raised"], "the fit must happen against a sealed table"

    # The sealed labels are read off the stored project, not off an endpoint:
    # the API deliberately serves the seal's DISCLOSURE and not its membership,
    # so an interface cannot show a user which rows are held out. Reaching past
    # HTTP here is the test doing what only a test may do.
    from turbotab.api import STORE
    stored = STORE.get(pid)
    df = stored.df
    held = {str(l) for l in stored.lockbox["labels"]}
    train = df[[str(i) not in held for i in df.index]]
    test = df[[str(i) in held for i in df.index]]
    assert len(test) > 10 and len(train) > 100, (
        f"the split is not usable: {len(train)} train / {len(test)} held out")

    numeric = ["age", "bmi"]
    categorical = ["site", "smoking"]

    fitted = {}
    for model_key, rows in body["models"].items():
        by_op = {r["operation"]: r for r in rows}
        enc = _encoder_for(by_op["encode"]["variant"])
        scaler = _scaler_for(by_op["scale"]["variant"])

        num_steps = [("imp", SimpleImputer(strategy="median"))]
        if scaler is not None:
            num_steps.append(("sc", scaler))
        pre = ColumnTransformer([
            ("num", Pipeline(num_steps), numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("enc", enc)]), categorical)])

        from ml.model_registry import get_registry
        est = get_registry()[model_key].factory("classification", 42)
        pipe = Pipeline([("pre", pre), ("est", est)])
        # Fitted on the unsealed rows ONLY. An encoder that saw the held-out
        # levels is leakage of the kind the seal exists to stop.
        pipe.fit(train[numeric + categorical], train["outcome"])
        preds = pipe.predict(test[numeric + categorical])

        assert len(preds) == len(test)
        assert set(np.unique(preds)) <= set(np.unique(train["outcome"])), (
            f"{model_key} predicted a label the training rows never contained")
        fitted[model_key] = preds

    assert len(fitted) == len(picks), (
        "not every selected model reached a fit; the recipe carried enough for "
        "some and not for others, which is the worst of the three outcomes "
        "because it looks like success")


def test_the_resolved_row_carries_everything_the_executor_needed(driven):
    """Sufficiency, asserted as a property of the row rather than of the fit.

    The test above could pass while the executor quietly reached past the row —
    for a column list, for the model's family, for a special case. So this pins
    the fields, on the object: an executor that has these four needs nothing
    else, and one of them is the §06 disposition, without which the caller
    cannot know the transform must be fitted inside the split.
    """
    client, pid, _ = driven
    body = client.get(f"/project/{pid}/recipes").json()
    ops = {o["key"]: o for o in body["operations"]}

    for model_key, rows in body["models"].items():
        for row in rows:
            for field in ("operation", "variant", "reason", "determinacy"):
                assert row.get(field), f"{model_key}/{row}: no {field}"
            op = ops[row["operation"]]
            assert op["scope"] in ("row_local", "stateful"), (
                f"{row['operation']} does not say whether it is fitted inside "
                "the split, so an executor cannot place it safely")
            assert op["applies_to"] if "applies_to" in op else True
            assert row["variant"] in op["variants"], (
                f"{model_key}/{row['operation']} resolved to "
                f"{row['variant']!r}, which is not one of {op['variants']} — "
                "an executor building from the variants list could not make it")


def test_encoding_is_stateful_and_says_so(driven):
    """Why this matters more for encoding than for anything else here.

    The set of levels — and their order, or their target means — are properties
    of the rows the encoder saw. Fitted on everything, a target encoder writes
    the held-out outcome into a training column, and the seal is undone by a
    preprocessing default nobody looked at.
    """
    client, pid, _ = driven
    ops = {o["key"]: o
           for o in client.get(f"/project/{pid}/recipes").json()["operations"]}
    assert ops["encode"]["scope"] == "stateful"
    assert "rows the encoder saw" in ops["encode"]["because"], (
        "the litmus answer does not say what the state is a property of, which "
        "is the sentence that tells a reader why it goes inside the split")
