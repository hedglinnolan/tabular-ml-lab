"""Wave 2a's seams: what holds only once ESTIMAND, CAUSAL, TIMEVARY and EXPLAIN run on one engine
beside wave 1 (DATAIN, SURVEY, OMICS, SCALES, NCI) and the routing gate.

Each package's own acceptance file tests its method against its references. This file tests the
places where two of them meet, each a rule one package wrote that the other's method must obey:

* **One registry** (BLUEPRINT §13). Three of the four packages were built before wave 1's registry
  landed and wrote their own contract shapes (EXPLAIN a copy of OMICS's retired module); every
  method now enters through the one registry (``turbotab.core.contracts``) with its vocabulary for
  slots, scopes and rungs, each tagged with the package that declared it.
* **One multiplicity method for a family** (ESTIMAND and MS7). ESTIMAND asks an exposure family's
  multiplicity on the estimand card; OMICS records it as ``set_multiplicity``. Both now write one
  slot, so the latest answer holds for the caption, the methods statement, the fit's table and the
  effects stage alike; and each question's leash is both packages' (unadjusted p-values are blocked
  and recorded for an omics family at any size, and beyond a few declared hypotheses).
* **The causal lane's sensitivity is ESTIMAND's** (CAUSAL and ESTIMAND; MODELING_SEQUENCE §0 ruling
  10: "required in the causal lane"). The lane's one call site (``causal.sensitivity_for``) runs
  ESTIMAND's ``unmeasured_confounding``: the E-value of a difference from the outcome's standard
  deviation, of a yes/no outcome's marginal risk ratio, and post-double selection's robustness value
  with each selected covariate a named benchmark; where an analysis is undefined it says why.
* **A time-varying exposure is the g-methods' lane** (CAUSAL and TIMEVARY). When the
  ``time_varying`` stage reads the declared exposure as changing within units, the point-exposure
  causal lane is not applicable and a ``set_causal`` estimator is refused (relation
  ``time_varying_exposure``): a confounder that earlier exposure changed biases these learners'
  estimate as it biases standard regression's. The time-varying lane's E-values are ESTIMAND's one
  implementation.
* **A spline exposure is the exposure** (ESTIMAND and WP12a). The Table 2 display shows the
  exposure's rows only; a restricted cubic spline's terms (``x``, ``x'``, ``x''``, as ``rms`` labels
  them) are all the exposure's.

The expected numbers come from an independent path (statsmodels' least squares, R's ``EValue`` and
``sensemakr``); the rules are read from the packages' own constants.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d
from turbotab.core.decisions import Refusal
from turbotab.core.tests.acceptance import estimand_fixtures as ef
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

AT = "2026-10-05T00:00:00Z"


def _fold(*decisions: d.BaseModel) -> d.ProjectState:
    return d.fold([d.DecisionRecord(id=f"r{i}", seq=i, at=AT, decision=x)
                   for i, x in enumerate(decisions, start=1)])


# ── one registry ─────────────────────────────────────────────────────────────

WAVE2A = {"ESTIMAND": ("effect_measure", "g_computation", "model_sequence", "diagnostics",
                       "unmeasured_confounding", "exposure_family", "plan_export"),
          "CAUSAL": ("dml_plr", "dml_irm", "tmle", "pds_lasso"),
          "TIMEVARY": ("time_varying",),
          "EXPLAIN": ("explain",)}


def test_every_wave2a_method_enters_through_the_one_registry():
    """BLUEPRINT §13: slot, data scope, needs, routing (question; options labeled customary and
    sound for each purpose, a rung for each), storyboard, sentence and relations, in one registry
    with one vocabulary; a conflict is refused or blocked and recorded, never silent; the run order
    agrees with every *precedes* relation; no package keeps a registry of its own."""
    from turbotab.core import contracts as C

    registry = C.contracts()
    for package, keys in WAVE2A.items():
        assert {k for k, c in registry.items() if c.package == package} == set(keys), package
    for key in (k for keys in WAVE2A.values() for k in keys):
        c = registry[key]
        assert c.slot in C.SLOTS and c.scope in C.SCOPES, key
        assert c.needs and c.question and c.storyboard and c.options and c.sentence, key
        for o in c.options:
            assert o.label and o.customary, (key, o.key)
            for purpose in C.PURPOSES:
                assert o.sound[purpose] and o.rung[purpose] in C.RUNGS, (key, o.key, purpose)
        for r in c.relations:
            if r.kind == "conflicts":
                assert r.rung in ("refused", "block_and_record"), (key, r.name)
        if isinstance(c.sentence, str):
            module, name = c.sentence.split(":")
            assert callable(getattr(importlib.import_module(module), name)), c.sentence
    C.run_order(list(registry))  # raises on a precedes relation the order breaks
    assert not hasattr(C, "register")  # ESTIMAND's own registry function is gone
    with pytest.raises(ModuleNotFoundError):  # EXPLAIN's copy of the retired OMICS registry
        importlib.import_module("turbotab.core.methods.contract")
    from turbotab.core.estimand import ESTIMATE_STAGES

    # Each package's estimates wait for the plan and lock it when first shown.
    assert {"effects", "causal", "time_varying", "explain"} <= set(ESTIMATE_STAGES)


# ── one multiplicity method for a family ─────────────────────────────────────

MEMBERS = [f"n{i}" for i in range(1, 7)]


def _family_state(members: list[str], lens: list[str] | None = None) -> d.ProjectState:
    roles = {"pid": "identifier", "age": "covariate", "sex": "covariate", "smoking": "covariate",
             **{m: "exposure" for m in members}}
    answers = {c: ef.CONFOUNDER for c in ("age", "sex", "smoking")}
    return ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                    answers=answers, family=True, multiplicity=None, models=["featurewise"],
                    lens=lens)


def test_the_estimand_card_and_set_multiplicity_write_one_slot_and_the_latest_holds():
    """``set_estimand`` declaring a family with its method writes the family's multiplicity slot,
    which ``set_multiplicity`` (MS7) also writes; the latest answer, from either question, is what
    the caption and the methods statement say, and what the fit's table applies. A single
    exposure's estimand has no family, so it clears the slot."""
    from turbotab.core import estimand
    from turbotab.core.methods import omics

    family = d.SetEstimand(family=True, measure="mean_difference", multiplicity="count_stated")
    st = _fold(family)
    assert st.multiplicity == d.MultiplicitySpec(method="stated_count")
    assert omics.multiplicity_policy(st)["method"] == "stated_count"
    st = _fold(family, d.SetMultiplicity(method="bh"))
    assert omics.multiplicity_policy(st)["method"] == "bh"
    assert estimand.family_multiplicity(st) == ("fdr_bh", False)
    st = _fold(d.SetMultiplicity(method="bh"), family)
    assert estimand.family_multiplicity(st) == ("count_stated", False)
    st = _fold(family, d.SetEstimand(exposure="n1", measure="mean_difference"))
    assert st.multiplicity is None

    # A state built without the fold (the estimand alone) reads the estimand's method.
    built = _family_state(MEMBERS).model_copy(update={"estimand": d.EstimandSpec(
        family=True, measure="mean_difference", multiplicity="count_stated")})
    assert omics.multiplicity_policy(built) == {"method": "stated_count", "acknowledged": False,
                                                "recorded": True}
    caption = estimand.caption(built)
    assert "with unadjusted p-values, with the number of tests stated (6 tests)" in caption
    recorded = built.model_copy(update={"multiplicity": d.MultiplicitySpec(method="bh")})
    assert "with Benjamini–Hochberg q-values across the family (6 tests)" in estimand.caption(recorded)
    assert estimand.multiplicity_statement(recorded, recorded.estimand, 6).startswith(
        "6 exposures were tested, each in turn; Benjamini–Hochberg q-values control")


def test_unadjusted_p_values_are_blocked_by_both_packages_rules_on_both_questions():
    """MS7 blocks unadjusted p-values beyond a few declared hypotheses (more than
    ``MULTIPLICITY_SMALL`` tests); ESTIMAND blocks them for an omics family at any size. Each
    question now applies both, each with its own exits (Benjamini–Hochberg, or the attestation)."""
    from turbotab.core.methods.omics import MULTIPLICITY_SMALL

    many = [f"n{i}" for i in range(1, MULTIPLICITY_SMALL + 3)]
    ctx = {"state": _family_state(many), "task": "regression"}
    body = {"kind": "set_estimand", "family": True, "measure": "mean_difference",
            "multiplicity": "count_stated"}
    with pytest.raises(Refusal) as refused:
        d.validate(body, ctx)
    assert refused.value.code == "family_without_fdr"
    assert f"not {len(many):,}" in refused.value.message
    first, second = refused.value.exits
    assert first["decision"]["multiplicity"] == "fdr_bh"
    d.validate(second["decision"], ctx)  # kept as recorded
    d.validate(body, {"state": _family_state(MEMBERS), "task": "regression"})  # a few: customary

    omics_few = {"state": _family_state(MEMBERS, ["metabolomics"]), "task": "regression"}
    with pytest.raises(Refusal) as refused:
        d.validate(d.SetMultiplicity(method="stated_count"), omics_few)
    assert refused.value.code == "family_without_multiplicity"
    assert refused.value.message.startswith("Unadjusted p-values for an omics family of 6 tests")
    d.validate(d.SetMultiplicity(method="stated_count"),
               {"state": _family_state(MEMBERS), "task": "regression"})
    question = __import__("turbotab.core.estimand", fromlist=["x"]).multiplicity_question(
        _family_state(many), len(many))
    unadjusted = question["options"][1]
    assert unadjusted["sound"]["verdict"] == "unsound" and question["customary_first"] == "fdr_bh"


def _family_frame(n: int = 500, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1_000)
    frame = ef.cohort(n, seed=seed)
    for j, name in enumerate(MEMBERS):
        frame[name] = rng.normal(10 + j, 2, n) + 0.5 * frame["smoking"]
    frame["glucose"] = frame["glucose"] - 1.2 * frame["n1"] + 0.9 * frame["n4"]
    return frame


def test_a_recorded_method_reaches_the_fit_and_the_effects_stage_alike(tmp_path):
    """With unadjusted p-values recorded by ``set_multiplicity`` after the estimand declared
    Benjamini–Hochberg, the fit's table and every model of the effects stage carry no q column,
    each member's estimate and p-value still statsmodels' least squares of the outcome on it and the
    covariates, and the statement says the p-values are not adjusted."""
    frame = _family_frame()
    st = _family_state(MEMBERS).model_copy(update={
        "estimand": d.EstimandSpec(family=True, measure="mean_difference", multiplicity="fdr_bh"),
        "multiplicity": d.MultiplicitySpec(method="stated_count")})
    run = ef.run(frame, tmp_path / "count", st)
    rows = {r["feature"]: r for r in run["fit"]["models"][0]["coefficients"]}
    assert list(rows) == MEMBERS
    for m in MEMBERS:
        X = ef.design_matrix(frame, [m, "age", "sex", "smoking"])
        ref = sm.OLS(frame["glucose"].to_numpy(float), X).fit()
        assert rows[m]["estimate"] == pytest.approx(ref.params[m], rel=1e-8)
        assert rows[m]["p"] == pytest.approx(ref.pvalues[m], rel=1e-6)
        assert rows[m]["q"] is None
    for model in run["effects"]["families"][0]["sequence"]:
        assert [r["feature"] for r in model["effects"]] == MEMBERS, model["key"]
        assert all(r["q"] is None for r in model["effects"]), model["key"]
    assert run["effects"]["multiplicity"].startswith(
        "6 exposures were tested, each in turn; p-values are not adjusted for multiplicity")
    # A later set_multiplicity recomputes both: each stage reads the one slot.
    from turbotab.core.graph import load_graph
    from turbotab.core.stages import GRAPH_FACTORY

    graph = load_graph(GRAPH_FACTORY)
    assert "multiplicity" in graph["fit"].reads and "multiplicity" in graph["effects"].reads


# ── the causal lane's sensitivity is ESTIMAND's ──────────────────────────────

SENSITIVITY_R = """
suppressPackageStartupMessages({library(EValue); library(sensemakr)})
d <- read.csv(rows_csv)
m <- lm(sbp ~ fiber + age + smoker, data = d)
s <- sensemakr(m, treatment = "fiber", kd = 1,
               benchmark_covariates = list(age = "age", smoker = "smoker"))
st <- s$sensitivity_stats; b <- s$bounds
e <- suppressMessages(evalues.OLS(est = 0.42, se = 0.11, sd = 2.3))
r <- suppressMessages(evalues.RR(est = 0.81, lo = 0.70, hi = 0.94))
out(list(rv_q = st$rv_q, rv_qa = st$rv_qa, labels = b$bound_label, est = b$adjusted_estimate,
         lo = b$adjusted_lower_CI, hi = b$adjusted_upper_CI,
         ols_point = e[2, 1], ols_limit = if (is.na(e[2, 2])) e[2, 3] else e[2, 2],
         rr_point = r[2, 1], rr_limit = if (is.na(r[2, 2])) r[2, 3] else r[2, 2]))
"""


@needs_r
def test_the_causal_lane_s_required_sensitivity_is_estimand_s_against_r(tmp_path):
    """``causal.sensitivity_for`` against R: the E-value of a difference (``EValue::evalues.OLS``),
    of a marginal risk ratio (``evalues.RR``), and post-double selection's robustness values and
    named benchmarks (``sensemakr`` on the same least-squares fit), to 1e-8. Cross-fitted
    estimators carry no robustness value, and a risk difference with no ratio no E-value, each said
    with the reason; the analysis is never a pass or a fail."""
    from turbotab.core import causal

    rng = np.random.default_rng(31)
    n = 600
    age = rng.normal(50, 10, n)
    smoker = (rng.uniform(size=n) < 0.3).astype(float)
    fiber = 15 + 0.05 * (age - 50) - 2 * smoker + rng.normal(0, 4, n)
    sbp = 120 + 0.4 * (age - 50) + 5 * smoker - 0.5 * fiber + rng.normal(0, 8, n)
    rows = pd.DataFrame({"fiber": fiber, "age": age, "smoker": smoker, "sbp": sbp})
    r = run_r(SENSITIVITY_R, {"rows": rows}, tmp_path)

    difference = {"measure": "mean_difference", "estimate": 0.42, "se": 0.11, "ci_low": 0.20,
                  "ci_high": 0.64}
    pds = causal.sensitivity_for(
        [difference], method="pds_lasso", exposure="fiber", outcome="sbp", outcome_sd=2.3,
        matrix=rows[["fiber", "age", "smoker"]], y=sbp, exposure_column="fiber",
        benchmarks={"age": ["age"], "smoker": ["smoker"]})
    assert pds["required"] and pds["computed"] and pds["not_computed"] is None
    assert pds["methods"] == ["robustness_value", "e_value"]  # ranked: the robustness value first
    assert pds["robustness"]["rv"] == pytest.approx(r["rv_q"], rel=1e-8)
    assert pds["robustness"]["rv_alpha"] == pytest.approx(r["rv_qa"], rel=1e-8)
    bench = {b["covariate"]: b for b in pds["robustness"]["benchmarks"]}
    for label, est, lo, hi in zip(r["labels"], r["est"], r["lo"], r["hi"]):
        name = label.split(" ")[-1]
        assert (bench[name]["estimate"], bench[name]["ci_low"], bench[name]["ci_high"]) == \
            pytest.approx((est, lo, hi), rel=1e-8)
    assert pds["e_value"]["point"] == pytest.approx(r["ols_point"], rel=1e-8)
    assert pds["e_value"]["limit"] == pytest.approx(r["ols_limit"], rel=1e-8)
    assert "pass" not in pds["reading"].replace("never as a pass or a fail", "")

    dml = causal.sensitivity_for([difference], method="dml_plr", exposure="fiber", outcome="sbp",
                                 outcome_sd=2.3)
    assert dml["methods"] == ["e_value"] and dml["computed"]
    assert dml["e_value"]["point"] == pytest.approx(r["ols_point"], rel=1e-8)
    assert dml["not_computed"].startswith("No robustness value: it is defined for one least-squares")
    assert causal.sensitivity_sentence(dml) == (
        " Sensitivity to unmeasured confounding is reported by the E-value for the estimate and for "
        "the confidence limit nearer the null, never as a pass or a fail.")

    risk = [{"measure": "risk_difference", "estimate": -0.04, "se": 0.012, "ci_low": -0.064,
             "ci_high": -0.016},
            {"measure": "risk_ratio", "estimate": 0.81, "se": None, "ci_low": 0.70, "ci_high": 0.94}]
    tmle = causal.sensitivity_for(risk, method="tmle", exposure="heavy", outcome="dm",
                                  outcome_sd=None)
    assert tmle["methods"] == ["e_value"]
    assert tmle["e_value"]["point"] == pytest.approx(r["rr_point"], rel=1e-8)
    assert tmle["e_value"]["limit"] == pytest.approx(r["rr_limit"], rel=1e-8)
    assert "E-value for the marginal risk ratio" in tmle["reading"]

    irm = causal.sensitivity_for(risk[:1], method="dml_irm", exposure="heavy", outcome="dm",
                                 outcome_sd=None)
    assert irm["computed"] is False and irm["methods"] == []
    assert irm["not_computed"] == (
        "No robustness value: it is defined for one least-squares coefficient (Cinelli & Hazlett "
        "2020, J R Stat Soc B 82:39), and double/debiased machine learning in the interactive "
        "model fits its nuisance models by cross-fitted learners; no E-value: a risk difference "
        "with no risk ratio beside it carries no risks to form one from.")
    assert causal.sensitivity_sentence(irm).startswith(
        " Sensitivity to unmeasured confounding, required in the causal lane, could not be computed "
        "for this estimate: no robustness value")


# ── a time-varying exposure is the g-methods' lane ───────────────────────────


def test_a_time_varying_exposure_is_the_g_methods_lane_not_the_causal_lane(tmp_path):
    """The ``time_varying`` stage, run by the real graph on a long cohort whose diet switches
    between visits (counted by pandas), reads the exposure as changing within units; the causal
    question is then not applicable and an estimator is refused with its exits (the time-varying
    question, or the primary model only), while ``method = none`` is not refused by it. The same
    cohort with the diet fixed within each person leaves the causal lane to its own gate."""
    from turbotab.core import causal
    from turbotab.core.models import time_varying as tv
    from turbotab.core.tests.acceptance.test_timevary import feedback_state, run_stage
    from turbotab.core.tests.acceptance.timevary_fixtures import feedback_cohort

    frame = feedback_cohort(n=600, visits=4, seed=5)
    switches = int((frame.groupby("pid")["dash"].nunique() > 1).sum())
    assert switches > 0
    state = feedback_state()
    art = run_stage(frame, state, tmp_path, "varies")
    assert art["setting"]["exposure_varies"] is True
    assert art["setting"]["exposure_changers"] == switches
    gate = causal.causal_gate(state, None, art)
    assert gate == ("not_applicable", "`dash` changes over time within units, so its effect is "
                                      "estimated by the time-varying lane's g-methods; the causal "
                                      "lane estimates a point exposure's effect.")
    ctx = {"state": state, "task": "binary",
           "artifact": lambda name: art if name == "time_varying" else None}
    with pytest.raises(Refusal) as refused:
        causal._point_exposure_only(d.SetCausal(exposure="dash", method="tmle",
                                                assumptions=list(causal.ASSUMPTIONS)), ctx)
    assert refused.value.code == "time_varying_exposure"
    assert [e["decision"] for e in refused.value.exits] == [
        None, d.SetCausal(exposure="dash", method="none").model_dump(mode="json")]
    causal._point_exposure_only(d.SetCausal(exposure="dash", method="none"), ctx)
    assert causal.RELATIONS["time_varying_exposure"].rung == "refused"

    fixed = frame.copy()
    fixed["dash"] = fixed.groupby("pid")["dash"].transform("first")
    still = run_stage(fixed, state, tmp_path, "fixed")
    assert still["setting"]["exposure_varies"] is False
    assert causal.causal_gate(state, None, still) != gate
    causal._point_exposure_only(d.SetCausal(exposure="dash", method="tmle"),
                                {**ctx, "artifact": lambda name: still})

    # One E-value implementation: the lane's wrappers are ESTIMAND's ``e_values``.
    from turbotab.core.models.effects import e_values

    assert tv.e_value_or(1.8, 1.2, 2.6, rare=False)["point"] == e_values(
        1.8, 1.2, 2.6, measure="OR", rare=False)["point"]


# ── a spline exposure is the exposure ────────────────────────────────────────


def test_a_spline_exposure_s_terms_are_all_the_exposure_s_rows():
    """``rms`` labels a restricted cubic spline's k − 1 columns ``x``, ``x'``, ``x''``
    (``methods.exposure_form.spline_names``); the Table 2 display keeps every one of them among the
    exposure's rows, and a longer predictor's name is still that predictor's."""
    from turbotab.core.estimand import primary_features
    from turbotab.core.methods.exposure_form import spline_names

    names = spline_names("fiber", 5)
    assert names == ["fiber", "fiber'", "fiber''", "fiber'''"]
    fitted = [*names, "fiber_soluble", "age", "kcal"]
    assert primary_features(fitted, "fiber", ["fiber", "fiber_soluble", "age", "kcal"]) == names
