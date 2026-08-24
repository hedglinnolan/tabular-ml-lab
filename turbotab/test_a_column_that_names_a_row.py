"""`GUIDED-108` — a per-row identifier is not a predictor.

Found at L37 while rejecting a trigger for the resolution card, and the
measurement is the argument: **`respondent_id` is 299 of
`survey_instrument.csv`'s 344 candidate parameters.** `training._feature_frame`
dropped the target and the grain's group column and nothing else, so every one
reached the one-hot encoder and then the model.

The load-bearing claim is **driven, not asserted**: an identifier can separate
the outcome perfectly and explain nothing, and
`test_an_identifier_can_separate_the_outcome_perfectly` shows it happening.

`GUIDED-097` — THE FIXTURE RULE. Two target shapes, and the shapes not covered
are named below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab import identifiers as ID
from turbotab import training as T
from turbotab.project import AnalysisProject, ProjectError

#: `GUIDED-097`. Both fixtures carry a real identifier column.
TARGET_SHAPES = {
    "continuous regression": ("survey_instrument.csv", "age", "regression",
                              "respondent_id", 299),
    "binary classification": ("leaky_sepsis.csv", "sepsis", "classification",
                              "admission_id", 159),
}

#: NOT COVERED, said out loud.
#:
#: NEAR-UNIQUE COLUMNS — a column at 0.95 distinct-per-row is *nearly* an
#: identifier and the app does not know whether the 5% are real repeats or a
#: typo, so it says nothing rather than guessing. `UNIQUE_PER_ROW` is 1.0 and
#: is deliberately not lowered; `GUIDED-121` files the case, unbuilt.
#:
#: COMPOSITE IDENTIFIERS — two columns that are unique only in combination
#: (site + local record number) are each non-unique and neither is flagged. The
#: arithmetic here is per column.
#:
#: MULTICLASS — the exclusion does not depend on the target's shape at all, so
#: the behavior is expected to hold; it is simply not driven here.
SHAPES_NOT_COVERED = [
    "near-unique columns — the rule is exactly one level per row, and the "
    "in-between case is filed as GUIDED-121 rather than guessed at",
    "composite identifiers — unique only in combination, and the arithmetic "
    "here is per column",
    "multiclass classification — the rule is target-independent, but undriven",
]


def _sealed(name, target, task):
    df = pd.read_csv(f"turbotab/sample_data/{name}")
    df = df[df[target].notna()].copy()
    p = AnalysisProject.from_dataframe(df, name)
    p.target, p.task_type = target, task
    p.set_grain("not_sure")
    p.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(p.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.25))]
    p.seal_lockbox(labels, fraction=len(labels) / len(p.df))
    return p


# ═══════════ THE CLAIM, DRIVEN ═══════════

def test_an_identifier_can_separate_the_outcome_perfectly():
    """**The reason this is a defect and not an inefficiency.**

    A level that appears exactly once is a row's name, so a model with capacity
    can fit the training outcome exactly through it and has learned nothing
    that transfers. Constructed rather than taken from a fixture, because the
    point is the mechanism and a fixture would leave it arguable.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    n = 120
    rng = np.random.default_rng(3)
    # THE OUTCOME IS A COIN FLIP. Nothing in this table predicts it.
    y = rng.integers(0, 2, size=n)
    X = pd.DataFrame({"record_id": [f"R{i:04d}" for i in range(n)]})

    pipe = Pipeline([("enc", OneHotEncoder(handle_unknown="ignore")),
                     ("m", DecisionTreeClassifier(random_state=0))])
    pipe.fit(X, y)
    apparent = float((pipe.predict(X) == y).mean())

    assert apparent == 1.0, (
        f"the identifier did not separate a coin flip perfectly ({apparent}); "
        f"if this ever fails the mechanism has changed and the argument in "
        f"GUIDED-108 needs re-making")
    # And it transfers nothing: unseen ids are all-zero rows.
    unseen = pd.DataFrame({"record_id": [f"Z{i:04d}" for i in range(n)]})
    assert len(set(pipe.predict(unseen))) == 1, (
        "an unseen identifier should collapse to one constant prediction")


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_identifier_is_found_and_its_cost_is_counted(shape):
    name, target, task, column, parameters = TARGET_SHAPES[shape]
    df = pd.read_csv(f"turbotab/sample_data/{name}")
    found = {row["column"]: row for row in ID.detect(df, target)}

    assert column in found, f"{column} was not detected on {name}"
    row = found[column]
    assert row["n_levels"] == row["n_rows"]
    assert row["distinct_per_row"] == 1.0
    assert row["parameters"] == parameters, (
        f"{column} costs {row['parameters']} parameters, and GUIDED-108's "
        f"evidence says {parameters}")
    assert column in row["sentence"] and str(parameters) in row["sentence"], (
        "the sentence states the conclusion without the arithmetic it rests on")


# ═══════════ THE FALSE POSITIVE THAT DRIVING FOUND ═══════════

def test_a_continuous_measurement_is_not_an_identifier():
    """**The defect the first version had, and it would have been severe.**

    Unique-per-row alone flagged `sample_id` plus about NINETY `mz_*` columns
    on `metabolomics_untargeted.csv` — the study's actual predictors. A
    continuous measurement is unique per row because it is continuous; every
    float differs. Excluding them would have deleted the analysis in order to
    protect it.

    The distinction is what the model can DO with the values: a numeric column
    is used for its ORDER, and an order over n distinct values is exactly what
    a continuous predictor is. A text column with one level per row has no
    order to use.
    """
    df = pd.read_csv("turbotab/sample_data/metabolomics_untargeted.csv")
    found = {row["column"] for row in ID.detect(df, "responder")}

    assay = [c for c in df.columns if str(c).startswith("mz_")]
    unique_assay = [c for c in assay if df[c].dropna().nunique() == len(df)]
    assert unique_assay, (
        "no assay column is unique per row, so this fixture cannot exercise "
        "the false positive and the test proves nothing")
    assert not (found & set(assay)), (
        f"{len(found & set(assay))} assay column(s) were treated as "
        f"identifiers: {sorted(found & set(assay))[:5]}")
    assert "sample_id" in found, "the real identifier stopped being detected"


def test_the_grouping_column_is_not_reported_as_a_defect():
    """It is already dropped from the feature frame, and it is the answer to a
    question the user was asked. Naming it here would report a recorded
    decision back to them as a problem."""
    df = pd.read_csv("turbotab/sample_data/survey_instrument.csv")
    found = {row["column"] for row in
             ID.detect(df, "age", group_col="respondent_id")}
    assert "respondent_id" not in found


def test_a_column_with_blanks_is_not_flagged():
    """A column with blanks cannot have one level per ROW, and treating it as
    unique-per-present-value would flag any sparse column."""
    df = pd.DataFrame({"y": range(20),
                       "sparse": [f"v{i}" if i % 2 else None for i in range(20)]})
    assert ID.detect(df, "y") == []


# ═══════════ IT REACHES THE FIT ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_model_is_not_fed_the_identifier(shape):
    """`AUDIT-008`'s shape closed: the exclusion has to reach the frame the
    model is actually handed, not only a receipt beside it."""
    name, target, task, column, _ = TARGET_SHAPES[shape]
    p = _sealed(name, target, task)

    loose = T._feature_frame(p.working_table, target, None)
    fed = T.feature_frame(p)
    assert column in loose.columns, "the fixture does not carry the identifier"
    assert column not in fed.columns, (
        f"{column} still reaches the model")
    assert set(fed.columns) == set(loose.columns) - {column}, (
        "the exclusion removed more than the identifier")


def test_every_path_that_feeds_a_model_goes_through_one_door():
    """`_feature_frame` takes three loose arguments and four call sites passed
    them, which is four places that have to remember the same rule.
    `GUIDED-108` is what happened when one forgot, and the prediction that
    *somebody will pass the loose one next time* is one this codebase has been
    right about repeatedly."""
    import ast
    import pathlib

    # THE POSITIVE CONTROL. An absence assertion over a tree gets easier to
    # satisfy as the tree empties (`GUIDED-045`), so this checks the scan can
    # SEE the thing it is asserting is absent before asserting it.
    scanned = 0
    seen_the_door = False
    offenders = []
    for path in sorted(pathlib.Path("turbotab").glob("*.py")):
        if path.name.startswith("test_") or path.name == "training.py":
            continue
        tree = ast.parse(path.read_text())
        scanned += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "_feature_frame":
                offenders.append(path.name)
            if isinstance(node, ast.Attribute) and node.attr == "feature_frame":
                seen_the_door = True
    assert scanned > 20, f"the scan read {scanned} modules; it is not reading the package"
    assert seen_the_door, (
        "the scan found no call to `training.feature_frame` anywhere, so it "
        "cannot see the name it is asserting the absence of a sibling for — "
        "the assertion below would pass against an empty package")
    assert not offenders, (
        f"{sorted(set(offenders))} call `_feature_frame` directly instead of "
        f"`training.feature_frame(project)`, so identifier exclusion — and "
        f"whatever rule is added next — does not apply there")


# ═══════════ THE SHELF IS NEVER SHORTENED ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_exclusion_comes_with_a_receipt(shape):
    """A column dropped without a receipt is indistinguishable from a column
    the app never saw."""
    name, target, task, column, parameters = TARGET_SHAPES[shape]
    p = _sealed(name, target, task)
    receipt = ID.receipt(p)

    assert receipt is not None
    assert receipt["excluded"] == [column]
    # COUNTED OVER THE TRAINING ROWS, which is the cost the model actually
    # pays and is smaller than the whole-table figure in `GUIDED-108`'s
    # evidence. The detector reads the rows a decision is entitled to be
    # informed by, per `GUIDED-088` — `test_the_identifier_is_found_and_its_
    # cost_is_counted` above checks the finding's own number on the raw table.
    n_train = len(p.training_rows)
    assert receipt["parameters_saved"] == n_train - 1
    assert receipt["parameters_saved"] < parameters
    assert "left out of the models" in receipt["headline"]
    # THE RULE IS STATED, so the user can disagree with it rather than only
    # with its result. *Say the number.*
    assert "100%" in receipt["rule"] and "different" in receipt["rule"]
    assert "named" in receipt["rule"], (
        "the receipt does not say that the column's NAME played no part, "
        "which is the objection a reader will have first")


def test_the_user_can_put_it_back_and_the_model_sees_it_again():
    name, target, task, column, _ = TARGET_SHAPES["continuous regression"]
    p = _sealed(name, target, task)
    assert column not in T.feature_frame(p).columns

    decision = p.keep_identifier(column)
    assert p.kept_identifiers == [column]
    assert column in T.feature_frame(p).columns, (
        "the user overturned the exclusion and the model still does not see "
        "the column")
    assert "over the app's objection" in decision.text
    assert ID.receipt(p)["kept"] == [column]

    p.keep_identifier(column, keep=False)
    assert column not in T.feature_frame(p).columns


def test_putting_back_a_column_that_was_never_excluded_is_refused():
    """An allow-list entry for a column nobody excluded is a decision about
    nothing that outlives the reason it was made."""
    name, target, task, _, _ = TARGET_SHAPES["continuous regression"]
    p = _sealed(name, target, task)
    with pytest.raises(ProjectError, match="never excluded"):
        p.keep_identifier("age")


def test_no_identifiers_means_no_card_rather_than_an_empty_one():
    """Absent, not blank: a table with no identifier columns has not had any
    excluded, and a labeled empty region would read as a finding of nothing."""
    p = _sealed("clinical_longitudinal.csv", "sbp", "regression")
    assert ID.detect(p.df, "sbp") == []
    assert ID.receipt(p) is None


def test_the_receipt_reaches_the_features_payload():
    """`LOOP.md` §05: ships with its consumer."""
    from fastapi.testclient import TestClient

    from turbotab import api

    name, target, task, column, _ = TARGET_SHAPES["continuous regression"]
    p = _sealed(name, target, task)
    api.STORE.add(p)
    client = TestClient(api.app)

    body = client.get(f"/project/{p.id}/features").json()
    assert body["identifiers"], "the Features payload carries no receipt"
    assert body["identifiers"]["excluded"] == [column]

    ok = client.post(f"/project/{p.id}/decision", json={
        "kind": "keep_identifier", "subject": column,
        "payload": {"column": column, "kept": True}})
    assert ok.status_code == 200, ok.text
    after = client.get(f"/project/{p.id}/features").json()
    assert after["identifiers"]["kept"] == [column]
    assert after["identifiers"]["excluded"] == []

    bad = client.post(f"/project/{p.id}/decision", json={
        "kind": "keep_identifier", "subject": "age",
        "payload": {"column": "age", "kept": True}})
    assert bad.status_code == 400, "a column nobody excluded was accepted"


# ═══════════ L40-A2 · `GUIDED-120` — ONE CORE, NO FORKS ═══════════

def test_the_core_and_this_module_give_the_same_answer():
    """**The private copy is gone, and this is what would notice it returning.**

    `turbotab/identifiers.py` used to read `ml/dataset_profile` for numeric
    columns and apply its own arithmetic to text ones, because the core's rule
    required an integer dtype and answered `False` for every string identifier.
    That was the honest interim; the product owner ruled at L40 that the fix
    belongs in core, and it is there.

    So the two must now agree on every column of every fixture, and
    `core_says_id_like` stops being a compensation and becomes the receipt.
    """
    import pathlib

    from ml.dataset_profile import compute_feature_profile

    checked = 0
    for path in sorted(pathlib.Path("turbotab/sample_data").glob("*.csv")):
        df = pd.read_csv(path)
        flagged = {r["column"] for r in ID.detect(df, target=None)}
        for column in df.columns:
            core = compute_feature_profile(df, column, len(df))
            checked += 1
            assert bool(core.is_id_like) == (str(column) in flagged), (
                f"{path.name}:{column} — core says "
                f"{core.is_id_like}, this module says "
                f"{str(column) in flagged}. One of them holds a private copy "
                f"of the rule, which is what ROADMAP's One core, no forks "
                f"forbids.")
    assert checked > 200, f"only {checked} columns compared"

    for row in ID.detect(pd.read_csv(
            "turbotab/sample_data/survey_instrument.csv"), "age"):
        assert row["core_says_id_like"] is True, (
            "a flagged column the core disagrees with means the private copy "
            "is back")


def test_the_widened_core_rule_catches_string_identifiers():
    """The ruling's own argument: *a string identifier is an identifier*, and
    widening makes Classic MORE correct rather than different.

    Before L40 all four of these answered `False` in core.
    """
    from ml.dataset_profile import compute_feature_profile

    for name, column in [("survey_instrument.csv", "respondent_id"),
                         ("leaky_sepsis.csv", "admission_id"),
                         ("clinic_visits.csv", "patient_id"),
                         ("metabolomics_untargeted.csv", "sample_id")]:
        df = pd.read_csv(f"turbotab/sample_data/{name}")
        assert df[column].dtype == object, f"{column} is no longer a string"
        assert compute_feature_profile(df, column, len(df)).is_id_like, (
            f"core does not flag {column}; the L40 widening regressed")


def test_the_widened_rule_still_spares_continuous_measurements():
    """The half that must not be lost. Dropping the numeric dtype test would
    have flagged 88 assay columns — the study's own predictors."""
    from ml.dataset_profile import compute_feature_profile

    df = pd.read_csv("turbotab/sample_data/metabolomics_untargeted.csv")
    assay = [c for c in df.columns if str(c).startswith("mz_")]
    unique = [c for c in assay if df[c].dropna().nunique() == len(df)]
    assert unique, "the fixture no longer exercises this"
    flagged = [c for c in unique
               if compute_feature_profile(df, c, len(df)).is_id_like]
    assert not flagged, f"{len(flagged)} continuous assay columns flagged"
