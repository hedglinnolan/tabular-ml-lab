"""EXPLORE 1 and 2 · Explore after the seal, outcome views recorded as looked at, and every lever
offered first as an in-fold rule (MODELING_SEQUENCE §0 ruling 3; §1 rows 1 and 5; §4 "Outcome views
in Explore"; TRIPOD+AI 7).

Expected values: the definitions by hand (Harrell's rule and his spline basis written out, caret's
near-zero-variance rule, the binned means), R's ``Hmisc::rcspline.eval`` (the knots each training
fold's rule places), the data's own held-out rows (Explore never reads them under prediction), and
the methods sentences asserted verbatim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core import contracts as C
from turbotab.core import decisions as d
from turbotab.core import provenance, voice
from turbotab.core.decisions import Refusal
from turbotab.core.methods import levers as L
from turbotab.core.tests.acceptance import explore_references as ref
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r
from turbotab.core.tests.graph_runner import GraphRun


def _table(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fiber = rng.gamma(4, 4, n)
    sugar = rng.gamma(3, 20, n)
    sex = rng.integers(1, 3, n)
    age = rng.uniform(20, 80, n)
    flat = np.where(rng.random(n) < 0.97, 0.0, 1.0)  # near-zero variance
    y = 100 - 0.08 * (fiber - 15) ** 2 + 0.05 * sugar + 2 * (sex == 2) + rng.normal(0, 5, n)
    blank = rng.random(n) < np.where(sex == 2, 0.10, 0.02)
    sugar = np.where(blank, np.nan, sugar)
    return pd.DataFrame({"pid": np.arange(n), "fiber_g": fiber, "sugar_g": sugar, "sex": sex,
                         "age": age, "flag_x": flat, "ldl": y})


def _state(purpose: str, holdout: float = 0.2, **extra) -> d.ProjectState:
    roles = {"pid": "identifier", "fiber_g": "exposure", "sugar_g": "covariate",
             "sex": "covariate", "age": "covariate", "flag_x": "covariate"}
    return d.ProjectState(
        lens=["dietary"], target="ldl", task="regression", purpose=purpose, roles=roles,
        missing=d.MissingSpec(strategy="impute"), categorical=["sex"],
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=holdout, seed=5, folds=5), models=["linear"], **extra)


def _explore(frame: pd.DataFrame, state: d.ProjectState, folder):
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "t.csv", index=False)
    run = GraphRun(folder / "t.csv", folder / "p")
    out = run.run(state, upto=["explore"])
    run.close()
    return out


# ── which rows Explore reads ─────────────────────────────────────────────────


def test_1_explore_reads_the_training_rows_under_prediction_and_every_analyzed_row_under_inference(
        tmp_path):
    """Ruling 3 and BLUEPRINT §12 ruling 3: under prediction Explore reads the training rows only, so
    rewriting every held-out row's outcome changes nothing it shows; under inference it reads every
    analyzed row, and the same rewrite moves its outcome views. The binned means are checked by hand
    (the outcome's mean in each tenth of the predictor, type-7 deciles)."""
    frame = _table()
    held_out = None
    shown = {}
    for purpose in ("prediction", "inference"):
        for tag, data in (("as_is", frame), ("rewritten", None)):
            if data is None:
                data = frame.copy()
                data.loc[data["pid"].isin(held_out), "ldl"] = 1e4
            out = _explore(data, _state(purpose), tmp_path / f"{purpose}_{tag}")
            assignment = out["split"].frames["assignment"]
            held = assignment.loc[~assignment["partition"].astype(str).str.lower()
                                  .isin(["train", "training"]), "row_id"].to_numpy()
            if held_out is None:
                held_out = held
            assert sorted(held) == sorted(held_out)  # the seal is the same draw
            shown[(purpose, tag)] = out["explore"].data
    p_as, p_re = shown[("prediction", "as_is")], shown[("prediction", "rewritten")]
    i_as, i_re = shown[("inference", "as_is")], shown[("inference", "rewritten")]
    assert p_as["rows"] == "training" and p_as["n_rows"] == len(frame) - len(held_out)
    assert p_as["n_holdout"] == len(held_out)
    assert i_as["rows"] == "analyzed" and i_as["n_rows"] == len(frame)
    assert p_as["findings"] == p_re["findings"]
    fiber = lambda art: next(f for f in art["findings"] if f["id"] == "explore::relationship::fiber_g")
    assert fiber(i_as)["points"] != fiber(i_re)["points"]
    train = frame[~frame["pid"].isin(held_out)]
    x, y = train["fiber_g"].to_numpy(), train["ldl"].to_numpy()
    cuts = np.unique(np.quantile(x, np.linspace(0.1, 0.9, 9)))
    group = np.searchsorted(cuts, x, side="left")
    want = [float(y[group == g].mean()) for g in range(len(cuts) + 1)]
    assert np.allclose([p["y"] for p in fiber(p_as)["points"]], want, atol=1e-10)


def test_1_each_finding_is_tied_to_a_lever_and_each_outcome_view_carries_its_record(tmp_path):
    """BLUEPRINT §11 rule 7: "a finding is a one-line claim plus its lever". The stack: the outcome's
    relationship with each continuous predictor and its distribution (outcome views, each with the
    ``view_outcome`` record a client posts on opening it), near-zero variance (caret's rule, by
    hand), and TRIPOD+AI 7's data quality across sociodemographic groups (each group's share of rows
    with a predictor blank, by hand)."""
    frame = _table()
    out = _explore(frame, _state("prediction"), tmp_path)
    art = out["explore"].data
    kinds = {f["kind"] for f in art["findings"]}
    assert {"outcome_relationship", "outcome_distribution", "low_variance",
            "quality_by_group"} <= kinds
    for f in art["findings"]:
        assert f["lever"] is not None and f["lever"]["options"], f["id"]
        assert len(f["summary"].split()) <= 20, f["summary"]
        if f["outcome_view"]:
            assert f["record"]["kind"] == "view_outcome" and not f["viewed"]
    flat = next(f for f in art["findings"] if f["kind"] == "low_variance")
    train_ids = out["explore"].data["n_rows"]
    assert flat["columns"] == ["flag_x"]
    assignment = out["split"].frames["assignment"]
    train = frame.set_index("pid").loc[assignment.loc[assignment["partition"].astype(str).str.lower()
                                                      .isin(["train", "training"]), "row_id"]]
    assert ref.caret_near_zero(train["flag_x"]) and not ref.caret_near_zero(train["fiber_g"])
    assert len(train) == train_ids
    quality = next(f for f in art["findings"] if f["id"] == "explore::quality::sex")
    blank = train[["fiber_g", "sugar_g", "age", "flag_x"]].isna().any(axis=1)
    for g in quality["groups"]:
        rows = train["sex"].astype(int).astype(str) == g["group"]
        assert g["n"] == int(rows.sum())
        assert abs(g["missing_share"] - float(blank[rows].mean())) < 1e-12
    assert quality["summary"] == ("Missing predictors differ by "
                                  f"{abs(quality['groups'][0]['missing_share'] - quality['groups'][1]['missing_share']):.0%}"
                                  " across `sex`'s groups")
    assert "sex" in art["proposed_subgroups"] and "age" in art["proposed_subgroups"]


# ── outcome views recorded as looked at, and disclosed ───────────────────────


def _log(tmp_path, purpose: str, holdout: float):
    log = d.DecisionLog(tmp_path / f"{purpose}_{holdout}.jsonl")
    split = {"n_train": 2400, "n_holdout": 600} if holdout else {"n_train": 3000, "n_holdout": 0}

    def add(decision):
        ctx = {"state": log.state()}
        if decision.kind == "view_outcome":  # the split stage's counts, as the server serves them
            ctx["artifact"] = lambda stage: split if stage == "split" else None
        decision = d.validate(decision, ctx)
        log.append(decision, sentence=lambda dec, st: voice.sentence_for(dec, st))
        return decision

    add(d.SetTarget(column="ldl"))
    add(d.SetPurpose(purpose=purpose))
    add(d.SetSplit(holdout=holdout, seed=1, folds=5))
    return log, add


@pytest.mark.parametrize("purpose,holdout", [("prediction", 0.0), ("prediction", 0.2),
                                             ("inference", 0.0)])
def test_1_outcome_views_are_recorded_and_disclosed_in_the_methods_text(tmp_path, purpose, holdout):
    """Ruling 3 and §4 ("record; … disclose hand-chosen levers; never block"): opening an outcome
    view records ``view_outcome`` (the server fills the rows and each column's lever answers at the
    first look); a later look keeps the first look's answers; the methods text says what was viewed
    on which rows and, restated on the answers as they stand, which lever was set after the view.
    Under prediction that lever is outside the corrected score, and a holdout covers it; under
    inference it is disclosed as made after the view. Verbatim."""
    log, add = _log(tmp_path, purpose, holdout)
    view = add(d.ViewOutcome(view="relationship", columns=["fiber_g"]))
    assert view.target == "ldl" and view.levers == {"fiber_g": {"form": "linear", "role": "none",
                                                                "kept": "yes"}}
    rows = "analyzed" if purpose == "inference" else "training"
    count = "3,000" if purpose == "inference" or not holdout else "2,400"
    said = (f"The outcome's relationship with `fiber_g` was viewed in Explore on the `{count}` "
            f"{rows} rows and recorded as looked at (Gelman & Loken 2013: a choice made after it "
            f"is disclosed).")
    assert log.records()[-1].sentence == said
    add(d.SetExposureForm(column="fiber_g", form="spline", knots=4))
    again = add(d.ViewOutcome(view="relationship", columns=["fiber_g"]))
    assert again.levers["fiber_g"]["form"] == "linear"  # the first look's answers carried forward
    text = provenance.methods_text(log.records()).text
    if purpose == "inference":
        tail = ("`fiber_g`'s form was changed from linear to a restricted cubic spline with 4 knots "
                "after its relationship with the outcome was viewed (forking paths).")
    else:
        tail = ("`fiber_g`'s form was set by hand from linear to a restricted cubic spline with 4 "
                "knots, chosen after viewing outcome relationships on the rows that validate the "
                "model; its optimism is not in the corrected score"
                + ("; the held-out rows, never viewed, cover it." if holdout else "."))
    assert f"{said} {tail[0].upper()}{tail[1:]}" in text
    # never blocked: a view is accepted under both purposes, before or after any lever
    assert d.validate(d.ViewOutcome(view="distribution"), {"state": log.state()}).view == "distribution"


def test_1_a_view_needs_the_outcome_and_names_real_columns():
    with pytest.raises(Refusal) as caught:
        d.validate(d.ViewOutcome(view="relationship", columns=["x"]), {"state": d.ProjectState()})
    assert caught.value.code == "no_outcome"
    with pytest.raises(Refusal) as caught:
        d.validate(d.ViewOutcome(view="relationship", columns=["nope"]),
                   {"state": d.ProjectState(target="y"), "columns": ["y", "x"]})
    assert caught.value.code == "unknown_column"


def test_1_the_explore_sentence_discloses_the_views_and_the_hand_levers(tmp_path):
    frame = _table()
    view = d.OutcomeViewSpec(view="relationship", column="fiber_g", target="ldl", rows="training",
                             n_rows=240, levers={"form": "linear", "role": "exposure", "kept": "yes"})
    state = _state("prediction", holdout=0.2, outcome_views={"relationship:fiber_g": view},
                   exposure_forms={"fiber_g": d.ExposureFormSpec(form="spline", knots=4)})
    art = _explore(frame, state, tmp_path)["explore"].data
    assert art["sentence"] == (
        "Explore read the 240 training rows only (the 60 held-out rows were never read), and every "
        "lever was offered first as an in-fold rule the resampling repeats; the outcome's "
        "relationship with `fiber_g` was viewed and recorded as looked at (Gelman & Loken 2013). "
        "`fiber_g`'s form was set by hand from linear to a restricted cubic spline with 4 knots, "
        "chosen after viewing outcome relationships on the rows that validate the model; its "
        "optimism is not in the corrected score; the held-out rows, never viewed, cover it.")
    finding = next(f for f in art["findings"] if f["id"] == "explore::relationship::fiber_g")
    assert finding["viewed"] is True


# ── each lever first as an in-fold rule ─────────────────────────────────────


def test_2_every_lever_is_offered_first_as_an_in_fold_rule(tmp_path):
    """Ruling 3: "each Explore lever is offered first as an in-fold rule the resampling repeats";
    a lever applied by hand comes after, its cost stated (and the holdout's cover when rows are held
    out)."""
    frame = _table()
    for holdout in (0.0, 0.2):
        art = _explore(frame, _state("prediction", holdout=holdout), tmp_path / str(holdout))["explore"].data
        for f in art["findings"]:
            options = f["lever"]["options"]
            hand = [i for i, o in enumerate(options) if o["key"] == "by_hand"]
            rules = [i for i, o in enumerate(options) if o["in_fold"]]
            if hand and rules:
                assert max(rules) < min(hand), f["id"]
        form = next(f for f in art["findings"] if f["id"] == "explore::relationship::fiber_g")
        keys = [o["key"] for o in form["lever"]["options"]]
        assert keys == ["rule", "inner_cv", "by_hand"]
        rule = form["lever"]["options"][0]
        assert rule["decision"] == {"kind": "set_levers", "forms": "rule", "variance_filter": "none",
                                    "keep": None, "imbalance": "none"}
        hand = form["lever"]["options"][2]
        assert hand["sound"] == ("Outside the corrected score: chosen after viewing this relationship"
                                 + ("; the held-out rows cover it" if holdout else ""))
        flat = next(f for f in art["findings"] if f["kind"] == "low_variance")
        assert flat["lever"]["options"][0]["decision"]["variance_filter"] == "near_zero"


@needs_r
@pytest.mark.parametrize("n,k", [(25, 3), (80, 4), (240, 5)])
def test_2_the_spline_rule_places_harrells_knots_on_each_training_fold(tmp_path, n, k):
    """Harrell (RMS §2.4.6): 3 knots below 30 rows, 5 above 100, else 4, on the fitting rows; the
    knots where R's ``Hmisc::rcspline.eval(x, nk = k, knots.only = TRUE)`` places them on the same
    rows, to 10⁻¹⁰; the basis as RMS eq. 2.25 writes it, by hand."""
    rng = np.random.default_rng(n)
    X = pd.DataFrame({"x": rng.gamma(3, 2, n), "z": rng.normal(size=n)})
    step = L.RuleSplines(["x", "z"], "regression").fit(X)
    assert step.k_ == k == ref.harrell_k(n)
    r = run_r(f"""
        suppressPackageStartupMessages(library(Hmisc))
        d <- read.csv(k_csv)
        out(list(x = rcspline.eval(d$x, nk = {k}, knots.only = TRUE),
                 z = rcspline.eval(d$z, nk = {k}, knots.only = TRUE)))
    """, {"k": X}, tmp_path)
    for c in ("x", "z"):
        assert np.max(np.abs(step.knots_[c] - np.asarray(r[c]))) < 1e-10
    out = step.transform(X)
    want = ref.rcs_columns(X["x"].to_numpy(), step.knots_["x"])
    names = [c for c in out.columns if c.startswith("x")]
    assert np.max(np.abs(out[names].to_numpy() - want)) < 1e-10


def test_2_the_inner_cv_form_choice_matches_an_independent_loop_on_the_training_fold():
    """Nonlinearity by inner cross-validation: on the training fold, each candidate is linear or a
    spline (k by the rule) as the inner folds' mean squared error says, the knots placed on each
    inner training fold. Reference: the same loop by hand (``numpy.linalg.lstsq`` on the basis
    written out from RMS eq. 2.25, knots at Harrell's type-7 percentiles), its losses to 10⁻¹⁰; and a
    known truth: the curved predictor is bent, the straight one is not."""
    rng = np.random.default_rng(21)
    n = 400
    X = pd.DataFrame({"curved": rng.uniform(-2, 2, n), "straight": rng.normal(size=n)})
    y = np.sin(2 * X["curved"]) + 0.5 * X["straight"] + rng.normal(0, 0.3, n)
    order = np.random.default_rng(3).permutation(n)
    splits = [(np.sort(np.setdiff1d(order, order[f::5])), np.sort(order[f::5])) for f in range(5)]
    step = L.InnerCVForms(["curved", "straight"], "regression", cv=splits).fit(X, y)
    assert set(step.knots_) == {"curved"}
    k = ref.harrell_k(n)
    for c in ("curved", "straight"):
        lin, spl = [], []
        for train, test in splits:
            other = [o for o in X.columns if o != c]
            base_tr = np.column_stack([np.ones(len(train)), X.iloc[train][X.columns].to_numpy()])
            base_te = np.column_stack([np.ones(len(test)), X.iloc[test][X.columns].to_numpy()])
            b = np.linalg.lstsq(base_tr, y.iloc[train], rcond=None)[0]
            lin.append((len(test), float(np.mean((y.iloc[test] - base_te @ b) ** 2))))
            knots = ref.harrell_knots(X.iloc[train][c].to_numpy(), k)
            s_tr = np.column_stack([np.ones(len(train)), ref.rcs_columns(X.iloc[train][c].to_numpy(), knots),
                                    X.iloc[train][other].to_numpy()])
            s_te = np.column_stack([np.ones(len(test)), ref.rcs_columns(X.iloc[test][c].to_numpy(), knots),
                                    X.iloc[test][other].to_numpy()])
            b = np.linalg.lstsq(s_tr, y.iloc[train], rcond=None)[0]
            spl.append((len(test), float(np.mean((y.iloc[test] - s_te @ b) ** 2))))
        mean = lambda pairs: sum(w * v for w, v in pairs) / sum(w for w, _ in pairs)
        got_lin, got_spl = step.losses_[c]
        assert abs(got_lin - mean(lin)) < 1e-10 and abs(got_spl - mean(spl)) < 1e-10


def test_2_the_variance_filter_follows_carets_rule_on_the_training_fold():
    """caret's ``nearZeroVar`` (Kuhn 2008) with its defaults, by hand, on the fold's own rows; the
    ``top`` filter keeps the most variable candidates of the fold."""
    rng = np.random.default_rng(22)
    n = 300
    X = pd.DataFrame({"a": np.where(rng.random(n) < 0.97, 0.0, rng.normal(size=n)),
                      "b": rng.normal(size=n), "c": np.round(rng.normal(size=n), 0),
                      "d": np.ones(n), "e": rng.normal(0, 3, n)})
    step = L.VarianceFilter("near_zero").fit(X)
    assert step.dropped_ == [c for c in X.columns if ref.caret_near_zero(X[c])]
    top = L.VarianceFilter("top", keep=2, columns=list(X.columns)).fit(X)
    variances = {c: X[c].var(ddof=1) for c in X.columns
                 if X[c].nunique() >= L.MIN_DISTINCT}
    assert sorted(top.kept_) == sorted(sorted(variances, key=lambda c: -variances[c])[:2]
                                       + [c for c in X.columns if c not in variances])


def test_2_the_levers_scopes_are_the_ones_lockbox_constitution_06s_test_observes():
    """``contracts.observed_scope``: the spline rule and the variance filter learn from the fold's
    rows but never its outcome (``training_fold``); the inner-CV form choice and the imbalance
    correction read the outcome (``model``). Each matches its contract."""
    rng = np.random.default_rng(23)
    n = 150
    frame = pd.DataFrame({"x": rng.gamma(3, 2, n), "z": rng.normal(size=n),
                          "w": rng.normal(size=n)})
    frame["w"] *= frame["z"].std() / frame["w"].std()  # two equally spread candidates
    y = np.sin(frame["x"]) + rng.normal(0, 0.5, n)
    none = np.zeros(n, dtype=bool)

    def spline(f, r, yy):
        return L.RuleSplines(["x"], "regression").fit(f).transform(f)

    def filt(f, r, yy):
        step = L.VarianceFilter("top", keep=1, columns=["z", "w"]).fit(f)
        return pd.DataFrame({c: (f[c] if c in step.kept_ else 0.0) for c in f.columns})

    def inner(f, r, yy):  # a fixed set of columns whichever form is chosen
        out = L.InnerCVForms(["x"], "regression", cv=5, seed=1).fit(f, yy).transform(f)
        return out.reindex(columns=["x", "x'", "x''", "x'''", "z", "w"], fill_value=0.0)

    assert C.observed_scope(spline, frame, none, y, 3) == "training_fold" == C.scope_of("spline_rule")
    assert C.observed_scope(filt, frame, none, y, 3) == "training_fold" == C.scope_of("variance_filter")
    assert C.observed_scope(inner, frame, none, y.to_numpy(), 3) == "model" == C.scope_of("inner_cv_form")


def test_2_the_levers_run_in_every_fold_of_the_fit_and_say_so(tmp_path):
    """``set_levers`` puts its rules in each family's pipeline (construction, then filters, then the
    model; MODELING_SEQUENCE §1.1), so cross-validation and the bootstrap repeat them; the design's
    step list names them and the record's sentence says they were fitted in each training fold."""
    frame = _table()
    state = _state("prediction", holdout=0.0,
                   levers=d.LeverSpec(forms="rule", variance_filter="near_zero"))
    frame.to_csv(tmp_path / "t.csv", index=False)
    run = GraphRun(tmp_path / "t.csv", tmp_path / "p")
    out = run.run(state, upto=["design"])
    steps = [n for n, _ in out["design"].objects["pipelines"]["linear"].steps]
    assert steps[-3:] == ["lever_forms", "lever_filter", "model"]
    described = [s["key"] for s in out["design"].data["models"][0]["steps"]]
    assert "lever_forms" in described and "lever_filter" in described
    run.close()
    assert voice.sentence_for(d.SetLevers(forms="rule", variance_filter="near_zero")) == (
        "Within each training fold, every continuous predictor entered as a restricted cubic spline "
        "with its number of knots set by Harrell's rule (3 below an effective size of 30, 5 above "
        "100, else 4) and its knots at Harrell's percentiles; near-zero-variance predictors were "
        "dropped (caret's nearZeroVar rule), so the resampling repeated each rule and its optimism "
        "is in the corrected score.")


def test_2_the_levers_are_refused_under_inference_with_ways_forward():
    """Under inference a form is declared before the estimates, never chosen by the rows it is
    reported from; the refusal's exits are the functional-form question and prediction."""
    state = d.ProjectState(purpose="inference", target="y", task="regression")
    with pytest.raises(Refusal) as caught:
        d.validate(d.SetLevers(forms="inner_cv"), {"state": state})
    assert caught.value.code == "levers_not_inference"
    assert [e["label"] for e in caught.value.exits] == [
        "Declare the form of what you study (the functional-form question)", "Make the purpose prediction"]
    assert d.validate(d.SetLevers(), {"state": state}).forms == "none"
