"""Wave 2a's seams: what holds only once ESTIMAND, CAUSAL, TIMEVARY and EXPLAIN run on one engine
beside wave 1 (DATAIN, SURVEY, OMICS, SCALES, NCI) and the routing gate.

Each package's own acceptance file tests its method against its references. This file tests the
places where two of them meet, each a rule one package wrote that the other's method must obey:

* **One registry** (BLUEPRINT §13). Three of the four packages were built before wave 1's registry
  landed and wrote their own contract shapes; every method now enters through the one registry
  (``turbotab.core.contracts``) with its vocabulary for slots, scopes and rungs.
* **One multiplicity method for a family** (ESTIMAND and MS7). ESTIMAND asks an exposure family's
  multiplicity on the estimand card; OMICS records it as ``set_multiplicity``. Both now write one
  slot, so the latest answer holds for the caption, the methods statement, the fit's table and the
  effects stage alike; and each question's leash is both packages' (unadjusted p-values are blocked
  and recorded for an omics family at any size, and beyond a few declared hypotheses).
* **A spline exposure is the exposure** (ESTIMAND and WP12a). The Table 2 display shows the
  exposure's rows only; a restricted cubic spline's terms (``x``, ``x'``, ``x''``, as ``rms`` labels
  them) are all the exposure's.

The expected numbers come from an independent path (statsmodels' least squares); the rules are read
from the packages' own constants.
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

AT = "2026-10-05T00:00:00Z"


def _fold(*decisions: d.BaseModel) -> d.ProjectState:
    return d.fold([d.DecisionRecord(id=f"r{i}", seq=i, at=AT, decision=x)
                   for i, x in enumerate(decisions, start=1)])


# ── one registry ─────────────────────────────────────────────────────────────

WAVE2A = {"ESTIMAND": ("effect_measure", "g_computation", "model_sequence", "diagnostics",
                       "unmeasured_confounding", "exposure_family", "plan_export")}


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
