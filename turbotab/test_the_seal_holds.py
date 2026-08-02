"""The lockbox constitution, **tested rather than asserted**.

Every held-out number this app will ever print rests on one claim: the test
rows were never seen. `IMPORT-020` is the proof that the claim can be false
while a lock icon renders cleanly — which is why the seal states its basis in
three sentences rather than one, and why `undetermined` is first-class.

**Until L34-C the Guided door had never computed a held-out number**, so the
claim had never been tested here. It had been asserted: by the disclosure, by
the constitution, by the devchecks that read the lockbox's own fields. Asserting
a claim from inside the thing that makes it is how `IMPORT-020` happened.

This file probes it from outside, and adversarially. Five families, chosen
because they fail in five different ways:

1. **Parameters.** `DOMAIN_SCIENCE.md` §05 — *every parameter estimated from
   data must be estimated inside the resampling loop* — and the family is
   larger than the model's coefficients. The imputer's medians, the scaler's
   means and standard deviations, the encoder's level set are each a fact about
   the rows they saw, and a fact learned from a held-out row has leaked whether
   or not anyone predicts with it. Probed by MOVING the held-out rows and
   watching the fitted parameters not move.
2. **Grouping.** A grouped seal is honored end to end on a table where one
   participant contributes several rows.
3. **Identity.** Decision A's barrier: no post-seal operation changes the index
   of a surviving row, so the labels the lockbox holds still name the same
   rows.
4. **Honesty under uncertainty.** An `undetermined` seal produces numbers, and
   they arrive labeled exploratory — not a clean lock, and not a locked door.
5. **The choices AROUND the fit**, which is where it found something.

## What it found

**Seventeen probes, and the fit itself is clean.** No held-out row moves any
fitted parameter — asserted as bitwise equality after replacing the held-out
rows with values no training row could produce, so a scaler that saw one of
them fails. No participant spans a grouped split. The labels still name the
same rows after training, and training does not write to the table.

**The leak was one level out, in a decision rather than in a fit.**
`GUIDED-088`: `model_shelf` profiled the whole table, so the ranking a user
picks a model from was computed with the held-out rows in view. Mild, and still
a decision informed by rows the seal exists to keep out — and `select_models`
states the requirement in its own refusal, *the shape it reads must be the
shape the models will actually be fitted on*. `api.selection_evidence` already
masked to the training rows. Two paths in one app, one consulting the seal and
one not.

That is worth saying plainly: **the probe aimed at the fit and the finding was
beside it.** Everything that computes a number was careful; the thing that
ranked the options was not, because nobody had thought of the ranking as a
parameter estimated from data. It is one.

The count is the answer to *how hard did it look* — a probe that finds nothing
and does not say how many ways it tried is indistinguishable from one that did
not run.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, training                                    # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _sealed_project(client, frame: pd.DataFrame, target: str, *,
                    grain: str = "one_row_per_person",
                    group_col: str = None, fraction: float = 0.25):
    pid = client.post("/project", files={
        "file": ("s.csv", frame.to_csv(index=False).encode(),
                 "text/csv")}).json()["id"]

    def decide(what, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": what, "payload": payload})
        assert r.status_code == 200, (what, r.text[:250])

    decide("set_target", column=target)
    if group_col:
        decide("set_grain", answer=grain, group_col=group_col)
        decide("set_repeat_kind", kind="repeats")
        decide("set_unit_of_analysis", unit="record")
    else:
        decide("set_grain", answer=grain)
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=fraction)
    return pid, api.STORE.get(pid)


def _repeated_frame(n_people: int = 60, per: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    people = np.repeat(np.arange(n_people), per)
    effect = rng.normal(0, 3, n_people)[people]
    return pd.DataFrame({
        "pid": people,
        "x1": effect + rng.normal(0, 0.3, len(people)),
        "x2": rng.normal(0, 1, len(people)),
        "y": effect + rng.normal(0, 1, len(people)),
    })


def _flat_frame(n: int = 160, seed: int = 1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    return pd.DataFrame({
        "x1": x1,
        "x2": rng.normal(0, 1, n),
        "y": 2 * x1 + rng.normal(0, 1, n),
    })


# ── 1 · parameters, and the family is larger than the coefficients ─────────

def _fitted_prep(project, model_key="ridge"):
    """Every preprocessing parameter a run actually fitted.

    Read off the pipeline rather than recomputed, because a recomputation here
    would be a second implementation agreeing with itself.

    **Collected by walking the fitted pipeline rather than by naming three
    attributes.** It used to reach for `prep.named_transformers_["num"]` and
    two known steps inside it, which was fine while the trainer built one
    hard-coded ColumnTransformer — and would have quietly stopped covering
    anything the moment the pipeline was composed from the user's own plan
    (`GUIDED-095`), because the imputer a user chose could be in a block this
    never opened. Walking finds every fitted array wherever the plan put it, so
    a step added next loop is probed without anybody remembering to add it.
    """
    table = project.working_table
    target = str(project.target)
    sealed = set(project.lockbox["labels"])
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    has_y = table[target].notna()
    features = training._feature_frame(table, target,
                                       (project.grain or {}).get("group_col"))
    X_train = features[has_y & ~is_test]
    y_train = table.loc[X_train.index, target]

    from ml.model_registry import get_registry
    spec = get_registry()[model_key]
    pipe = training._pipeline(spec.factory("regression", 42), features,
                              needs_scaling=True, project=project,
                              model_key=model_key)
    pipe.fit(X_train, y_train)
    out = {"n_train": int(len(X_train)),
           "coef": np.asarray(pipe.named_steps["model"].coef_, dtype=float)}
    for name, array in sorted(_fitted_arrays(pipe.named_steps["prep"])
                              + _fitted_arrays(pipe.named_steps["shape"])):
        out[name] = array
    assert len(out) > 2, (
        "no fitted preprocessing parameter was found at all, so the "
        "comparison below is between two dictionaries of nothing")
    return out


#: Attributes scikit-learn fits from data. Named rather than sniffed by the
#: trailing underscore, because that convention also covers `n_features_in_`
#: and `feature_names_in_`, which are facts about the SHAPE and move whenever
#: the column list moves — a probe that compared those would report a leak
#: every time the plan changed.
_FITTED_ATTRIBUTES = ("statistics_", "mean_", "scale_", "var_", "center_",
                      "deviation_", "lower_", "upper_", "lambdas_",
                      "data_min_", "data_max_", "bin_edges_",
                      "components_", "categories_", "encodings_")


def _fitted_arrays(node, prefix=""):
    """Every fitted parameter under a fitted transformer, named by its path."""
    found = []
    for attribute in _FITTED_ATTRIBUTES:
        if not hasattr(node, attribute):
            continue
        try:
            value = np.asarray(getattr(node, attribute), dtype=float)
        except (TypeError, ValueError):
            value = np.asarray(
                [str(v) for v in np.ravel(np.asarray(
                    getattr(node, attribute), dtype=object))])
        found.append((f"{prefix}{type(node).__name__}.{attribute}", value))
    for entry in (getattr(node, "transformers_", None) or []):
        child_name, child = entry[0], entry[1]
        if hasattr(child, "fit"):
            found += _fitted_arrays(child, f"{prefix}{child_name}/")
    for child_name, child in (getattr(node, "named_steps", None) or {}).items():
        found += _fitted_arrays(child, f"{prefix}{child_name}/")
    return found


def test_no_held_out_row_moves_any_fitted_parameter(client):
    """**The probe, and it is a moving one.** The held-out rows are replaced
    with wildly different values and every fitted parameter must be unchanged
    to the last bit — the model's coefficients AND the imputer's medians AND
    the scaler's means and standard deviations.

    A leak that a metric comparison would call "about the same" this asserts
    exactly, so a scaler that saw one held-out row fails it.
    """
    frame = _flat_frame()
    pid, project = _sealed_project(client, frame, "y")
    before = _fitted_prep(project)

    sealed = set(project.lockbox["labels"])
    assert sealed, "nothing was held out"                            # control
    poisoned = project.df.copy()
    mask = poisoned.index.isin(list(sealed))
    assert mask.sum() > 0
    # Not noise: values no training row could produce, so anything that saw
    # them moves visibly.
    poisoned.loc[mask, "x1"] = 10_000.0
    poisoned.loc[mask, "x2"] = -10_000.0
    poisoned.loc[mask, "y"] = 99_999.0
    project.df = poisoned

    after = _fitted_prep(project)
    assert after["n_train"] == before["n_train"]
    assert set(before) == set(after), (
        "the fitted pipeline changed shape when only the held-out rows did")
    for key in sorted(set(before) - {"n_train"}):
        assert np.array_equal(before[key], after[key]), (
            f"{key} moved when only the HELD-OUT rows changed, so the held-out "
            f"rows are inside the fit")


def test_the_probe_can_fail(client):
    """The positive control, and this file is worthless without it.

    The same comparison against a fit that DOES see the held-out rows must
    detect it. A probe that cannot fail is a probe that proves nothing, and
    every assertion above is an equality between two numbers that could both be
    computed wrong in the same way.
    """
    frame = _flat_frame()
    pid, project = _sealed_project(client, frame, "y")

    def fitted_on_everything(table):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        numeric = table[["x1", "x2"]]
        imp = SimpleImputer(strategy="median").fit(numeric)
        sc = StandardScaler().fit(imp.transform(numeric))
        return np.asarray(sc.mean_, dtype=float)

    before = fitted_on_everything(project.df)
    poisoned = project.df.copy()
    sealed = list(project.lockbox["labels"])
    poisoned.loc[poisoned.index.isin(sealed), "x1"] = 10_000.0
    after = fitted_on_everything(poisoned)
    assert not np.array_equal(before, after), (
        "the comparison cannot see a leak it was pointed straight at, so the "
        "test above proves nothing")


def test_the_preprocessing_lives_inside_the_estimator(client):
    """Structural, because it is the property that makes the numeric one hold
    for every model rather than for the one this file fitted."""
    frame = _flat_frame()
    pid, project = _sealed_project(client, frame, "y")
    features = training._feature_frame(project.working_table, "y", None)
    from ml.model_registry import get_registry
    pipe = training._pipeline(get_registry()["ridge"].factory("regression", 0),
                              features, needs_scaling=True, project=project,
                              model_key="ridge")
    names = list(pipe.named_steps)
    assert names[0] == "prep" and names[-1] == "model", (
        "preprocessing is not inside the estimator, so `fit` is no longer the "
        "only place a parameter can come from")
    # EVERY step between them is a transformer, so nothing can smuggle a fit
    # in outside the estimator by being named something else.
    for name in names[1:-1]:
        assert hasattr(pipe.named_steps[name], "fit_transform"), (
            f"the step {name!r} sits inside the pipeline and is not a "
            "transformer")


def test_the_outcome_is_not_among_the_features(client):
    """The leak that needs no scaler. Trivial to check and catastrophic to
    miss, which is exactly the kind this project keeps finding."""
    frame = _flat_frame()
    pid, project = _sealed_project(client, frame, "y")
    run = training.train(project, ["ridge"])
    assert "y" not in run.features
    assert run.features, "every feature was dropped"                 # control


# ── 2 · grouping, honored end to end ───────────────────────────────────────

def test_no_participant_spans_a_grouped_split(client):
    """What a grouped seal PROMISES, recomputed against the frame rather than
    read off the disclosure. `IMPORT-020` is what a clean-looking lock over a
    real leak costs."""
    frame = _repeated_frame()
    pid, project = _sealed_project(client, frame, "y",
                                   grain="people_repeat", group_col="pid")
    assert project.lockbox["seal_basis"] == "grouped", (
        "this project did not produce a grouped seal, so the claim is not "
        "being tested")
    sealed = set(project.lockbox["labels"])
    table = project.df
    test_people = set(table.loc[table.index.isin(sealed), "pid"])
    train_people = set(table.loc[~table.index.isin(sealed), "pid"])
    overlap = test_people & train_people
    assert not overlap, (
        f"{len(overlap)} participant(s) appear on both sides of a grouped "
        f"split: {sorted(overlap)[:5]}")
    assert test_people and train_people                              # control


def test_the_grouping_column_is_not_handed_to_the_model(client):
    """A participant id is an identifier, not a feature. Given one, a model can
    memorize who rather than learn what — and on a grouped split every test
    row's id is a level the model has never seen."""
    frame = _repeated_frame()
    pid, project = _sealed_project(client, frame, "y",
                                   grain="people_repeat", group_col="pid")
    run = training.train(project, ["ridge"])
    assert "pid" not in run.features
    assert "x1" in run.features                                      # control


def test_a_grouped_run_still_produces_a_number(client):
    """A guard that made every score `nan` would satisfy every assertion above
    and be worthless."""
    frame = _repeated_frame()
    pid, project = _sealed_project(client, frame, "y",
                                   grain="people_repeat", group_col="pid")
    run = training.train(project, ["ridge"])
    scored = [r for r in run.results if r.metrics]
    assert scored, [r.error for r in run.results]
    assert np.isfinite(list(scored[0].metrics.values())[0])


# ── 3 · identity, which is Decision A ──────────────────────────────────────

def test_the_sealed_labels_still_name_the_same_rows_after_training(client):
    """Decision A's barrier. The lockbox holds row LABELS, so an operation that
    reindexes turns the seal into a set of labels pointing at other rows —
    silently, and afterwards there is no way to recover which rows it meant."""
    frame = _flat_frame()
    pid, project = _sealed_project(client, frame, "y")
    before = list(project.lockbox["labels"])
    rows_before = project.df.loc[project.df.index.isin(before)].copy()

    training.train(project, ["ridge", "knn_reg"])

    after = list(project.lockbox["labels"])
    assert after == before, "the seal's labels changed while training ran"
    rows_after = project.df.loc[project.df.index.isin(after)]
    pd.testing.assert_frame_equal(rows_before, rows_after)


def test_training_does_not_touch_the_working_table(client):
    """Training READS. A step that quietly wrote to the table would move the
    rows the seal names without anything saying so."""
    frame = _flat_frame()
    pid, project = _sealed_project(client, frame, "y")
    before = project.working_table.copy()
    training.train(project, ["ridge"])
    pd.testing.assert_frame_equal(before, project.working_table)


# ── 4 · honesty under uncertainty ──────────────────────────────────────────

def test_an_undetermined_seal_produces_numbers_and_labels_them_exploratory(
        client):
    """Not a clean lock, and not a locked door. The constitution's §03 says
    `undetermined` is first-class: the analysis continues and every number it
    produces carries what it rests on."""
    frame = _repeated_frame()
    pid, project = _sealed_project(client, frame, "y", grain="not_sure")
    assert project.lockbox["seal_basis"] == "undetermined"
    run = training.train(project, ["ridge"])

    scored = [r for r in run.results if r.metrics]
    assert scored, "an undetermined seal produced no number at all, which is a "\
                   "locked door rather than an honest one"
    assert run.exploratory is True
    assert any("exploratory" in n for n in run.notes), (
        "the numbers are exploratory and the run does not say so, so they read "
        "as a clean holdout")
    assert any("read better than the models are" in n for n in run.notes)


def test_an_attested_contradiction_makes_the_run_exploratory_too(client):
    """The basis is `cross_sectional` and honest — it is what the user said —
    so a run deriving exploratory from the basis alone would call this clean.
    It is not: the split rests on a disagreement that is on the record."""
    frame = _repeated_frame()
    pid = client.post("/project", files={
        "file": ("s.csv", frame.to_csv(index=False).encode(),
                 "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "y"}})
    clash = client.post(f"/project/{pid}/decision", json={
        "kind": "set_grain", "payload": {"answer": "one_row_per_person"}})
    assert clash.status_code == 409                                  # control
    for what, payload in [
            ("set_grain", {"answer": "one_row_per_person",
                           "acknowledge_contradiction": True}),
            ("set_eligibility", {"answer": "everyone"}),
            ("seal", {})]:
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": what, "payload": payload})
        assert r.status_code == 200, (what, r.text[:200])

    project = api.STORE.get(pid)
    assert project.lockbox["seal_basis"] == "cross_sectional"
    run = training.train(project, ["ridge"])
    assert run.exploratory is True, (
        "a split resting on an attested disagreement was reported as clean")


def test_a_run_before_the_seal_is_refused_rather_than_computed(client):
    """The order, enforced where the number is produced. A score computed
    before the seal is a score on rows the model may have been fitted on, and
    afterwards there is no way to tell which it was."""
    frame = _flat_frame()
    pid = client.post("/project", files={
        "file": ("s.csv", frame.to_csv(index=False).encode(),
                 "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "y"}})
    project = api.STORE.get(pid)
    with pytest.raises(training.TrainingRefusal, match="not sealed"):
        training.train(project, ["ridge"])


def test_a_holdout_too_small_to_mean_anything_is_refused(client):
    """The other honest silence. A metric from four rows is noise with a
    decimal point, and printing it would be the app asserting a precision it
    does not have."""
    frame = _flat_frame(n=24)
    pid, project = _sealed_project(client, frame, "y", fraction=0.15)
    if len(project.lockbox["labels"]) >= training.MIN_TEST_ROWS:
        pytest.skip("this split is large enough; the refusal is tested by the "
                    "unit below")
    with pytest.raises(training.TrainingRefusal, match="too few"):
        training.train(project, ["ridge"])


def test_the_metric_is_computed_on_the_held_out_rows_and_only_those(client):
    """The count, checked against the seal rather than against the run's own
    bookkeeping — a run that reported `n_test` from the same variable it
    predicted with would agree with itself no matter what it did."""
    frame = _flat_frame()
    pid, project = _sealed_project(client, frame, "y")
    run = training.train(project, ["ridge"])
    sealed = set(project.lockbox["labels"])
    table = project.working_table
    expected = int((table.index.isin(list(sealed)) & table["y"].notna()).sum())
    assert run.n_test == expected
    assert run.n_train == int(len(table)) - expected
    scored = [r for r in run.results if r.predictions]
    assert scored and len(scored[0].predictions) == expected, (
        "the number of predictions does not match the number of held-out rows")


# ── 5 · the choices AROUND the fit, which is where this probe found something ─

def test_the_model_shelf_ranks_on_the_training_rows(client):
    """**What the probe found.** `GUIDED-088`.

    `model_shelf` profiled the WHOLE table, so the order a user picks a model
    from was computed with the held-out rows in view — and `select_models`
    states the requirement in its own refusal: *the shape it reads must be the
    shape the models will actually be fitted on.* It was not.

    Mild as leaks go, and still one: choosing a model is a decision, and a
    decision informed by the held-out rows is what the seal exists to prevent.
    `api.selection_evidence` already masked to the training rows whenever a
    lockbox existed — two paths in one app, one consulting the seal and one
    not, which is `AUDIT-008` exactly.

    Asserted on the CITED NUMBER rather than on the order: an order can agree
    by luck on any particular table, and `p/n` cannot.
    """
    from turbotab import engine, models as _models

    frame = _flat_frame(n=120)
    pid, project = _sealed_project(client, frame, "y", fraction=0.3)
    sealed = set(project.lockbox["labels"])
    train = project.df.loc[[i not in sealed for i in project.df.index]]
    assert len(train) < len(project.df)                              # control

    on_train = _models.shelf(engine.profile(train, project.target,
                                            project.task_type),
                             project.task_type or "regression")
    served = project.model_shelf()
    assert [e.key for e in served] == [e.key for e in on_train]
    assert {e.concern for e in served} == {e.concern for e in on_train}, (
        "the shelf's clauses cite numbers from a different set of rows than "
        "the models will be fitted on")

    whole = _models.shelf(engine.profile(project.df, project.target,
                                         project.task_type),
                          project.task_type or "regression")
    assert {e.concern for e in served} != {e.concern for e in whole}, (
        "this fixture cannot tell the two profiles apart, so the assertion "
        "above would pass either way — pick a split that changes p/n")


def test_the_seal_is_consulted_wherever_a_choice_is_ranked(client):
    """The class, and this test used to be the reason `GUIDED-092` exists.

    It said in its own docstring that it was the class rather than the
    instance, and every assertion in it was about `/selection/evidence` — the
    path that was **already correct**. Its only reference to the shelf was
    `assert served` as a non-empty control. So it PASSED against the reverted
    shelf, and it enumerated nothing, so a third ranking added next loop was
    not covered either. A class guarded by one example is guarded by nothing.

    What it asserts now is that the enumeration exists and is not a subset of
    the two surfaces that were already right. The probes themselves live in
    `turbotab/test_every_ranking_is_computed_on_the_training_rows.py`, which
    iterates the registry — this is the seal file's pointer to them, kept here
    because §5 of this file is where a reader looks for the class.
    """
    from turbotab import rankings

    frame = _flat_frame(n=120)
    pid, project = _sealed_project(client, frame, "y", fraction=0.3)

    evidence = client.get(f"/project/{pid}/selection/evidence").json()
    assert evidence["n_rows_seen"] < len(project.df), (
        "the selection ranking saw every row, so the columns it prefers were "
        "chosen with the held-out rows in view")
    assert "train" in evidence["scope"].lower(), evidence["scope"]

    # THE PART THAT WAS MISSING. The enumeration has to reach past the two
    # surfaces that were already masked, or it is this test's old shape wearing
    # a registry.
    keys = set(rankings.keys())
    assert {"model_shelf", "selection_evidence"} < keys, (
        "the ranking registry covers only the surfaces that were already "
        "correct, which is exactly what this test used to do")
    for surface in rankings.training_scoped():
        assert surface.computes, f"{surface.key} names no primitive"
    assert rankings.exemptions(), (
        "every surface claims to be masked, and two of them measurably are "
        "not — an enumeration that cannot say so is an enumeration nobody "
        "checked against the code")


def test_the_run_records_what_it_read(client):
    """**This test asserted a defect and the defect is closed** (`GUIDED-089`).

    It used to require the run to say that the recorded preprocessing plan was
    NOT the one fitted — the honest form of a real divergence, and a
    placeholder with a deadline. The plan is composed from the record now, so
    the run says what it READ, with counts a reader can check against the
    Preprocess receipt.

    The full property is in
    `turbotab/test_the_fitted_pipeline_is_the_recorded_plan.py`. What is here
    is the seal file's own stake: a run that goes back to fitting defaults
    would put this sentence back, and this is where a reader of the seal probes
    would look for it.
    """
    frame = _flat_frame()
    pid, project = _sealed_project(client, frame, "y")
    run = training.train(project, ["ridge"])
    assert not any("not the recorded" in n or "not yet what gets fitted" in n
                   for n in run.notes), (
        "the run still says the recorded plan was not the one fitted")
    assert any("composed from the recorded plan" in n for n in run.notes), (
        "the run does not say where its pipeline came from, so a reader "
        "cannot tell a fit from the record from a fit from the defaults")
    assert run.results[0].plan["steps"], (
        "the result carries no plan, so nothing says what was fitted")
