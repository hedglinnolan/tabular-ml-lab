"""Clause §06: declaration and execution are separate.

    Does this transform's output for row *i* depend on any other row?

    No  → structural repair. Executes immediately, posts a receipt.
    Yes → statistical transform. Recorded now, fitted inside training folds.
    Unsure → defer.

The tests below pin the litmus as a **precondition**, not a convention: `apply`
refuses anything stateful, so a caller cannot materialize a
distribution-dependent transform on the working table even by asking for it
directly. That refusal is the clause; everything else is bookkeeping around it.

Feature selection is the sharpest case and gets its own section. *"Feature
selection on the full dataset"* is a named leak in Kapoor & Narayanan's
taxonomy, and it is subtle enough to ship: no held-out row is copied anywhere,
and yet the selected SET encodes test signal.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from turbotab import (engine, eligibility as E, features as F,           # noqa: E402
                      grain as G, selection as S)
from turbotab.project import AnalysisProject, ProjectError               # noqa: E402


def study(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "record_id": [f"R{i:04d}" for i in range(n)],
        "age": rng.integers(20, 80, n).astype(float),
        "weight_kg": rng.normal(78, 12, n),
        "height_m": rng.normal(1.7, 0.1, n),
        "chol": rng.normal(190, 25, n),
        "outcome": rng.integers(0, 2, n),
    })


def _sealed_project(df: pd.DataFrame | None = None) -> AnalysisProject:
    p = AnalysisProject.from_dataframe(df if df is not None else study(), "t")
    p.set_target("outcome", "classification", "high", [])
    p.set_grain(G.ONE_ROW_PER_PERSON)
    p.set_eligibility(E.EVERYONE)          # clause 01: grain -> eligibility -> SEAL
    d = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(d["labels"], **d["disclosure"])
    return p


# ── the litmus, as a precondition ────────────────────────────────────────────

def test_every_catalogue_entry_declares_a_scope_and_a_reason():
    """A transform whose classification is unstated is a transform nobody
    applied the litmus to."""
    for key, t in F.CATALOGUE.items():
        assert t.scope in (F.ROW_LOCAL, F.STATEFUL), f"{key} scope {t.scope!r}"
        assert len(t.because) > 40, (
            f"{key} states a scope without saying why. The litmus answer is "
            f"what makes the classification checkable by a reader.")
        assert t.sentence, f"{key} has no methods sentence"


def test_a_stateful_transform_is_recorded_not_materialized():
    """THE clause. `apply` refuses; `declare` records and computes nothing.

    Named in `tests/test_every_clause_is_tracked.py` as lockbox-06's coverage.
    """
    df = study()
    before = list(df.columns)

    for key in F.deferred_keys():
        with pytest.raises(F.FeatureRefusal, match="distribution|learns"):
            F.apply(df, key, ["age"], {"n_bins": 3, "n_components": 2})

    assert list(df.columns) == before, "a refused transform still changed the frame"

    spec = F.declare("bin_quantile", ["age"], {"n_bins": 3})
    assert spec["scope"] == F.STATEFUL
    assert spec["fit_on"] == "training folds only"
    assert "within each training fold" in spec["sentence"], (
        "the decision sentence must carry the TIMING — it is simultaneously "
        "the receipt, the schedule and the manuscript line")


def test_a_row_local_transform_executes_and_posts_a_receipt():
    df = study()
    out = F.apply(df, "ratio", ["weight_kg", "height_m"])
    assert "weight_kg_per_height_m" in out["frame"].columns
    r = out["receipt"]
    assert r["scope"] == F.ROW_LOCAL
    assert r["inputs"] == ["weight_kg", "height_m"]
    assert r["sentence"], "a structural repair executes immediately AND posts a receipt"
    assert "weight_kg" not in [c for c in df.columns if c.endswith("_per_height_m")], (
        "apply() mutated the caller's frame instead of returning a new one")


def test_declaring_a_row_local_transform_is_refused():
    """The refusal runs both ways. A row-local transform that got 'declared'
    would sit in the deferred list and never execute, which is the opposite
    failure and just as silent."""
    with pytest.raises(F.FeatureRefusal, match="row-local"):
        F.declare("ratio", ["weight_kg", "height_m"])


def test_a_row_local_transform_gives_the_same_answer_row_by_row():
    """The litmus, verified rather than asserted.

    If a transform is genuinely row-local, computing it on the whole frame and
    computing it on one row in isolation must agree. This is what "does row i
    depend on any other row" MEANS, and it is cheap to check.
    """
    df = study()
    for key in F.row_local_keys():
        t = F.get(key)
        if t.needs:                          # needs params; covered separately
            continue
        cols = ["weight_kg", "height_m"][:t.n_inputs]
        whole = F.apply(df, key, cols)["frame"]
        name = F.new_column_name(key, cols)
        for label in list(df.index[:5]):
            one = F.apply(df.loc[[label]], key, cols)["frame"]
            a, b = whole.loc[label, name], one.loc[label, name]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b), (
                f"{key} on {label} differs between the whole frame ({a}) and "
                f"that row alone ({b}) — it is not row-local")


def test_a_stateful_transform_would_give_a_different_answer_row_by_row():
    """The control. Without it the test above could pass for a catalogue that
    classified everything as row-local."""
    df = study()
    q_whole = pd.qcut(df["age"], 3, labels=False, duplicates="drop")
    q_half = pd.qcut(df["age"].iloc[:40], 3, labels=False, duplicates="drop")
    assert not (q_whole.iloc[:40] == q_half).all(), (
        "quantile bins came out the same on a subset, so this fixture does not "
        "demonstrate why quantile binning defers")


# ── binning and encoding split rather than resolve ───────────────────────────

def test_binning_splits_by_where_the_edges_come_from():
    assert F.classify("bin_fixed") == F.ROW_LOCAL
    for k in ("bin_quantile", "bin_uniform", "bin_kmeans"):
        assert F.classify(k) == F.STATEFUL, f"{k} should defer"


def test_uniform_binning_defers_even_though_it_looks_fixed():
    """The subtle one. Equal-WIDTH bins look like fixed cut-points until you
    notice the range comes from the data, so one extreme value moves every
    other row's bin."""
    assert F.classify("bin_uniform") == F.STATEFUL
    assert "minimum and maximum" in F.get("bin_uniform").because


def test_encoding_splits_by_where_the_order_comes_from():
    assert F.classify("ordinal_declared") == F.ROW_LOCAL
    assert F.classify("ordinal_frequency") == F.STATEFUL


def test_binning_by_supplied_edges_refuses_without_the_edges():
    """Without edges it would have to derive them, which is a different
    transform. Refusing is better than silently becoming the stateful one."""
    df = study()
    with pytest.raises(F.FeatureRefusal, match="at least two edges"):
        F.apply(df, "bin_fixed", ["age"])


def test_encoding_in_a_stated_order_refuses_without_the_order():
    df = study()
    df["severity"] = (["mild", "moderate", "severe"] * 40)
    with pytest.raises(F.FeatureRefusal, match="needs the order"):
        F.apply(df, "ordinal_declared", ["severity"])


# ── the preview is the real computation ──────────────────────────────────────

def test_a_preview_computes_the_real_values_and_does_not_persist():
    df = study()
    pv = F.preview(df, "log", ["chol"])
    assert pv["applied"] is False
    assert pv["rows"], "a preview with no rows is a description, not a preview"
    for row in pv["rows"]:
        assert row["after"] == pytest.approx(float(np.log(row["before"])), rel=1e-3)
    assert "log_chol" not in df.columns


def test_a_deferred_transforms_preview_is_labeled_and_shows_no_values():
    """Clause §06 permits a read-only preview 'not persisted to the modeling
    table' — and a preview of a distribution-dependent transform fitted on the
    whole column would be showing the researcher a picture of their held-out
    data, which is the leak arriving through the preview instead.

    Clause: `lockbox-06`
    """
    pv = F.preview(study(), "bin_quantile", ["age"], {"n_bins": 3})
    assert pv["preview_not_applied"] is True
    assert pv["rows"] == []
    assert pv["fit_on"] == "training folds only"


def test_the_domain_of_a_transform_is_reported_before_it_is_applied():
    """log of a non-positive value is undefined. The count of rows that will
    become missing is a fact the user needs BEFORE choosing, not after."""
    df = study()
    df.loc[df.index[:7], "chol"] = -1.0
    pv = F.preview(df, "log", ["chol"])
    assert pv["n_undefined"] == 7, (
        f"{pv['n_undefined']} undefined reported for 7 non-positive values")


# ── the project records it, and the cascade fires ────────────────────────────

def test_adding_a_feature_records_a_receipt_and_marks_downstream_stale():
    p = _sealed_project()
    p.findings_stale = False
    p.add_feature("ratio", ["weight_kg", "height_m"])
    assert p.engineered[-1]["column"] == "weight_kg_per_height_m"
    assert p.findings_stale is True, (
        "a new column changes the modeling problem; results computed under the "
        "old one are stale and the cascade must say so")
    assert p.decisions[-1].kind == "add_feature"
    assert p.decisions[-1].text, "the receipt is the transcript line"


def test_the_cascade_names_its_cause_and_is_not_cleared_by_recompute():
    """DESIGN_LANGUAGE §10 wants the cascade VISIBLE. A boolean that a
    re-diagnosis clears would say the cascade had been dealt with when nothing
    downstream had been recomputed — models and metrics are not refreshed by
    re-running the doctor."""
    p = _sealed_project()
    p.add_feature("ratio", ["weight_kg", "height_m"])
    p.add_feature("log", ["chol"])
    whys = [s["why"] for s in p.stale_downstream]
    assert len(whys) == 2, whys
    assert "weight_kg_per_height_m" in whys[0] and "log_chol" in whys[1], (
        f"the cascade did not name what changed: {whys}")

    # a re-diagnosis refreshes findings and must NOT claim the cascade is over
    p.set_findings([], None)
    assert p.findings_stale is False, "findings really were recomputed"
    assert len(p.stale_downstream) == 2, (
        "recomputing findings cleared the downstream cascade, which nothing "
        "downstream was recomputed by")


def test_the_selection_decision_also_cascades():
    p = _sealed_project()
    spec = S.declare("lasso", "outcome", ["age", "chol"])
    p.set_selection(spec)
    assert p.stale_downstream and "selection" in p.stale_downstream[-1]["why"]


def test_the_feature_work_survives_the_save_file():
    """`archive.py` is an explicit whitelist, so a field added to the project
    and not to the archive is dropped on save — the same gap the seal's basis
    fell through at L13."""
    from turbotab import archive
    p = _sealed_project()
    p.add_feature("ratio", ["weight_kg", "height_m"])
    p.defer_feature("bin_quantile", ["age"], {"n_bins": 3})
    p.set_selection(S.declare("mutual_info", "outcome", ["age", "chol"],
                              n_features=1))
    p.settle_features()

    back = archive.from_bytes(archive.to_bytes(p))
    assert [e["column"] for e in back.engineered] == ["weight_kg_per_height_m"]
    assert back.deferred_transforms[-1]["key"] == "bin_quantile"
    assert back.selection_spec["selected"] is None
    assert back.features_settled is True
    assert "weight_kg_per_height_m" in back.df.columns, (
        "the engineered column itself did not survive the round trip")


def test_removing_a_feature_is_possible_and_also_cascades():
    p = _sealed_project()
    p.add_feature("ratio", ["weight_kg", "height_m"])
    p.findings_stale = False
    p.remove_feature("weight_kg_per_height_m")
    assert "weight_kg_per_height_m" not in p.df.columns
    assert not p.engineered
    assert p.findings_stale is True


def test_only_engineered_columns_can_be_removed():
    p = _sealed_project()
    with pytest.raises(ProjectError, match="not created here"):
        p.remove_feature("age")


def test_a_feature_that_would_overwrite_an_existing_column_is_refused():
    p = _sealed_project()
    p.add_feature("log", ["chol"])
    with pytest.raises(ProjectError, match="already exists"):
        p.add_feature("log", ["chol"])


def test_a_deferred_transform_never_reaches_the_working_table():
    p = _sealed_project()
    before = list(p.df.columns)
    p.defer_feature("bin_quantile", ["age"], {"n_bins": 3})
    assert list(p.df.columns) == before, "a deferred transform added a column"
    assert p.deferred_transforms[-1]["key"] == "bin_quantile"
    assert p.decisions[-1].kind == "defer_feature"


def test_skipping_the_step_is_recorded():
    """A silent skip is indistinguishable from a step nobody reached."""
    p = _sealed_project()
    p.settle_features(skipped=True)
    assert p.features_settled is True
    assert p.decisions[-1].kind == "settle_features"
    assert "skipped" in p.decisions[-1].text.lower()


# ── selection: recorded, never performed ─────────────────────────────────────

def test_a_selection_spec_never_carries_a_selected_set():
    spec = S.declare("mutual_info", "outcome", ["age", "chol"], n_features=1)
    assert spec["selected"] is None, (
        "a spec carrying a chosen set is a selection that already ran, using "
        "the held-out rows")
    assert spec["fit_on"] == "training folds only"


def test_the_selection_sentence_carries_the_timing_as_methods_prose():
    spec = S.declare("mutual_info", "outcome", ["age", "chol", "weight_kg"],
                     n_features=2)
    assert spec["sentence"] == (
        "The top 2 features by mutual information with `outcome` will be "
        "selected within each training fold.")


def test_a_project_refuses_a_spec_that_already_chose():
    p = _sealed_project()
    spec = S.declare("lasso", "outcome", ["age", "chol"])
    spec["selected"] = ["age"]
    with pytest.raises(ProjectError, match="already-chosen"):
        p.set_selection(spec)


def test_the_outcome_cannot_be_a_candidate_feature():
    with pytest.raises(S.SelectionRefusal, match="outcome"):
        S.declare("lasso", "outcome", ["age", "outcome"])


def test_there_is_no_scope_that_fits_on_the_whole_table():
    """The API has two scopes and neither is 'everything'. A third option would
    be the leak with a name."""
    with pytest.raises(S.SelectionRefusal, match="no third option"):
        S.declare("lasso", "outcome", ["age"], scope="all_rows")


def test_the_weaker_scope_says_so_in_its_own_sentence():
    """Classic selects once over all training rows. That is better than the
    full table and worse than per-fold, and a project inheriting it must be
    able to SAY which happened rather than imply the stronger claim."""
    folds = S.declare("univariate", "outcome", ["age", "chol"], n_features=1)
    rows = S.declare("univariate", "outcome", ["age", "chol"], n_features=1,
                     scope=S.TRAIN_ROWS)
    assert "within each training fold" in folds["sentence"]
    assert "once over the training rows" in rows["sentence"]
    assert folds["sentence"] != rows["sentence"]


def test_selection_evidence_is_ranked_on_training_rows_only():
    """Guided must not regress from `pages/04:126-135`, which masks to training
    rows and says so. Here the mask is applied and the response records how
    many rows it saw."""
    p = _sealed_project()
    sealed = set(p.lockbox["labels"])
    mask = pd.Series([i not in sealed for i in p.df.index], index=p.df.index)
    ev = S.evidence(p.df, "outcome", ["age", "chol", "weight_kg"], mask)

    assert ev["preview_not_applied"] is True
    assert ev["n_rows_seen"] == len(p.df) - len(sealed), (
        "the ranking saw rows it should not have")
    assert ev["scope"] == S.TRAIN_ROWS
    assert ev["ranked"], "no ranking returned"


def test_selection_evidence_without_a_mask_says_it_saw_everything():
    """Silence about scope would be the regression. If no mask is supplied the
    response must say so rather than implying training-only."""
    p = _sealed_project()
    ev = S.evidence(p.df, "outcome", ["age", "chol"])
    assert ev["scope"] == "all rows"
    assert "exploratory" in ev["note"]


# ── drivable over HTTP: the whole step, as a browser can do it ───────────────

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from turbotab.api import app
    return TestClient(app)


def _drive_to_sealed(client) -> str:
    df = study()
    pid = client.post("/project", files={
        "file": ("study.csv", df.to_csv(index=False).encode(), "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_grain",
                      "payload": {"answer": G.ONE_ROW_PER_PERSON}})
    # Clause 01's sequence, over HTTP: grain -> eligibility -> SEAL. "Everyone"
    # is a recorded answer, not a skip.
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_eligibility",
                          "payload": {"answer": E.EVERYONE}})
    assert r.status_code == 200, r.text
    r = client.post(f"/project/{pid}/decision", json={"kind": "seal"})
    assert r.status_code == 200, r.text
    return pid


def test_a_driver_reaches_the_end_of_feature_work_without_leaving_guided(client):
    pid = _drive_to_sealed(client)

    # the interview asks, and both questions state their consumer
    iv = client.get(f"/project/{pid}/interview?step=features").json()
    keys = {q["key"]: q for q in iv["questions"]}
    assert "choose_features" in keys and "choose_selection" in keys
    for q in keys.values():
        assert q["consumer"], f"{q['key']} is a FACT with no stated consumer"

    # the catalogue arrives split, and every entry says why
    cat = client.get(f"/project/{pid}/features").json()
    assert {t["key"] for t in cat["row_local"]} >= {"ratio", "log", "product"}
    assert {t["key"] for t in cat["deferred"]} >= {"bin_quantile", "pca"}
    for t in cat["row_local"] + cat["deferred"]:
        assert t["because"], f"{t['key']} does not say why it is {t['scope']}"

    # a CHOICE gets a real before/after
    pv = client.get(f"/project/{pid}/feature/preview",
                    params={"transform": "ratio",
                            "columns": "weight_kg,height_m"}).json()
    assert pv["rows"] and pv["applied"] is False
    assert pv["new_column"] == "weight_kg_per_height_m"

    # applying it lands the column and cascades
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "add_feature",
                          "payload": {"transform": "ratio",
                                      "columns": ["weight_kg", "height_m"]}})
    assert r.status_code == 200, r.text
    stale = r.json()["stale_downstream"]
    assert stale and "weight_kg_per_height_m" in stale[-1]["why"], (
        f"the stale cascade did not fire, or did not name its cause: {stale}")
    assert any(e["column"] == "weight_kg_per_height_m"
               for e in r.json()["engineered"])

    # a stateful one is refused as an application and accepted as a decision
    bad = client.post(f"/project/{pid}/decision",
                      json={"kind": "add_feature",
                            "payload": {"transform": "bin_quantile",
                                        "columns": ["age"],
                                        "params": {"n_bins": 3}}})
    assert bad.status_code == 400
    assert "distribution" in bad.json()["detail"]

    ok = client.post(f"/project/{pid}/decision",
                     json={"kind": "defer_feature",
                           "payload": {"transform": "bin_quantile",
                                       "columns": ["age"],
                                       "params": {"n_bins": 3}}})
    assert ok.status_code == 200
    assert ok.json()["deferred_transforms"][-1]["fit_on"] == "training folds only"
    assert "bin_quantile" not in str(ok.json()["columns"])

    # selection: evidence on training rows, then a recorded spec
    ev = client.get(f"/project/{pid}/selection/evidence").json()
    assert ev["preview_not_applied"] is True
    assert ev["n_rows_seen"] < len(study()), "the ranking saw the sealed rows"

    sel = client.post(f"/project/{pid}/decision",
                      json={"kind": "set_selection",
                            "payload": {"method": "mutual_info",
                                        "candidates": ["age", "chol", "weight_kg"],
                                        "n_features": 2}})
    assert sel.status_code == 200, sel.text
    spec = sel.json()["selection_spec"]
    assert spec["selected"] is None
    assert "within each training fold" in spec["sentence"]

    # and the step ends, recorded
    done = client.post(f"/project/{pid}/decision",
                       json={"kind": "settle_features"})
    assert done.json()["features_settled"] is True
    iv = client.get(f"/project/{pid}/interview?step=features").json()
    assert "choose_features" not in [q["key"] for q in iv["questions"]]


def test_a_driver_can_skip_the_whole_step(client):
    pid = _drive_to_sealed(client)
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "settle_features", "payload": {"skipped": True}})
    assert r.status_code == 200
    assert r.json()["features_settled"] is True
    assert not r.json()["engineered"]


def test_selecting_every_column_is_a_recorded_answer_not_an_absence(client):
    """"Use everything" and "nobody answered" must not look the same."""
    pid = _drive_to_sealed(client)
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_selection", "payload": {}})
    assert r.status_code == 200
    assert r.json()["selection_spec"] is None
    kinds = [d["kind"] for d in r.json()["decisions"]]
    assert "set_selection" in kinds, (
        "choosing every column left no trace, so it is indistinguishable from "
        "never having been asked")


# ── the gap that became routing ──────────────────────────────────────────────

def test_asking_for_the_polynomial_basis_gets_the_routing_answer_not_unknown_key():
    """`feat-polynomial` is `classic-only`, and the register row carries two
    arguments for that. A user who reaches for it must meet both plus somewhere
    to go — "not in the catalogue" reads as an omission and teaches nothing.

    The specific thing pinned: the refusal names a MODEL as the route. Trees get
    interactions free, so mass generation is a model choice at the modeling step
    rather than a feature choice here, and that is the sentence worth more than
    the transform would have been.
    """
    for spelling in ("polynomial", "poly", "PolynomialFeatures", "interactions"):
        with pytest.raises(F.FeatureRefusal) as exc:
            F.get(spelling)
        said = str(exc.value)
        # Ordered most-diagnostic first, deliberately. A probe reads the FIRST
        # assertion to fire, so "routing did not happen at all" has to come
        # before the four that inspect the message it would have produced —
        # otherwise every routing failure reports as a missing substring and the
        # probe verifies the wrong reason.
        assert "not in the transform catalogue" not in said, (
            f"{spelling} fell through to the unknown-key message")
        # The breakdown, not the bare count: "55" appears in both arguments, so
        # asserting on it alone survived deleting the first one.
        assert "10 squares and 45 pairwise" in said, (
            f"{spelling}: the expansion is not quantified")
        assert "0.39" in said, f"{spelling}: the p/n argument is missing"
        # Both of these were weaker on the first draft and the probe caught it:
        # `"model" in said` survived deleting the route sentence, because "45
        # pairwise products" and an earlier "a model" both remain; so each now
        # asserts the CLAIM rather than a word that happens to be nearby.
        assert "not a feature choice" in said, f"{spelling}: no route is offered"
        assert "`product`" in said, (
            f"{spelling}: the one-interaction route is missing")

    # And the routing is not a catch-all: a genuine typo still says so, or the
    # message would become noise attached to every mistake.
    with pytest.raises(F.FeatureRefusal, match="not in the transform catalogue"):
        F.get("lorgarithm")
