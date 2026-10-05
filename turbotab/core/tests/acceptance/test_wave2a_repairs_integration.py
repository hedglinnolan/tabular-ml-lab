"""The wave 2a repairs' seams: what holds only once REPAIR-ESTIMAND, REPAIR-CAUSAL, REPAIR-TIMEVARY and
REPAIR-EXPLAIN run on wave 1b and the owner's rulings 13 and 14 (MODELING_SEQUENCE §0).

Each repair's own acceptance file tests its package against its references. This file tests the
places where a repair meets a rule another package wrote:

* **Model 2 is the fit's primary, imputed as the fit imputes** (REPAIR-ESTIMAND and wave 1b's MS2).
  The repair fits the unadjusted model, Model 1 and Model 2 on the primary's own completed copies
  and Model 3 on copies imputed with its columns; wave 1b put the survey design and the clustering
  in the declared models' imputation model. Under multiple imputation and the surveyed-population
  answer both hold: Model 2 is the fit's primary number for number, and Model 3's imputation holds
  the design too.
* **One SD for a difference's E-value** (ruling 14; REPAIR-ESTIMAND and REPAIR-CAUSAL). Under the
  surveyed-population answer the effects stage and the causal lane standardize a difference by the
  population's SD of the outcome (R survey's ``svyvar``), by one function
  (``models.effects.outcome_sd``).
* **The data's own imputed copies** (wave 1b's MS3 and ESTIMAND, CAUSAL). The fit pools its table
  over the copies by Rubin's rules; Table 2 and the causal lane are not built over them, so they are
  withheld with the reason (relations ``sequence-supplied-copies`` and ``supplied_copies``), never
  fit on the copies stacked, where each participant would count once per copy.

The causal lane's robustness value beside an interval that is not classical (ESTIMAND's relation
``rv-interval-classical``: post-double selection's HC3, the partially linear model's estimating
equation) is asserted in ``test_causal_lane.py`` against computations by hand and R's sensemakr.

Expected numbers come from an independent path: R's ``survey`` and ``EValue``, pandas counts, and
the fit stage's own table where the claim is that two stages agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.tests.acceptance import estimand_fixtures as ef
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r


def _surveyed(n: int = 900, seed: int = 23) -> pd.DataFrame:
    """The estimand cohort drawn by a survey: 15 strata of two PSUs, weights that over-represent
    smokers and the least and most active (so the population's SD of glucose is not the rows')."""
    frame = ef.cohort(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    frame["stratum"] = np.repeat(np.arange(1, 16), n // 15)[:n]
    frame["psu"] = frame["stratum"] * 10 + rng.integers(1, 3, n)
    frame["w"] = np.round(rng.uniform(0.5, 1.5, n) * np.exp(0.6 * np.abs(frame["activity"] - 3.5))
                          * (1 + 2 * frame["smoking"]) * 1000, 1)
    return frame


SURVEY_ROLES = {**ef.ROLES, "w": "design", "stratum": "design", "psu": "design"}
SURVEY_AMOUNTS = {**ef.AMOUNTS, "code_or_count:stratum": "code", "code_or_count:psu": "code"}
SURVEY = d.SurveySpec(estimand="population", weight="w", strata="stratum", psu="psu")


# ── Model 2 is the fit's primary, imputed as the fit imputes ─────────────────


def test_under_multiple_imputation_and_the_population_answer_model_2_is_the_fits_primary(
        tmp_path):
    """Ruling 3 (REPAIR-ESTIMAND) on wave 1b's MS2: with `age` and `bmi` (a possible mediator,
    Model 3's column) partly missing, multiple imputation and the surveyed-population answer, Model
    2's estimate and standard error are the fit's primary's to 1e-12 (the same completed copies,
    imputed with the design's variables), and Model 3's imputation names the design and takes its
    df as Rubin's complete-data df: PSUs minus strata, counted by pandas."""
    frame = _surveyed(600, seed=29)
    rng = np.random.default_rng(3)
    frame.loc[rng.random(len(frame)) < 0.12, "age"] = np.nan
    frame.loc[rng.random(len(frame)) < 0.10, "bmi"] = np.nan
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  roles=SURVEY_ROLES, survey=SURVEY, amounts=SURVEY_AMOUNTS,
                  missing={"strategy": "multiple_imputation", "m": 20})
    run = ef.run(frame, tmp_path, st)
    seq = ef.sequence(run["effects"])
    assert list(seq) == ["crude", "model_2", "model_3"]
    primary = next(r for r in run["fit_raw"]["models"][0]["coefficients"]
                   if r["feature"] == "fiber")
    [row] = [r for r in seq["model_2"]["effects"] if r["feature"] == "fiber"]
    assert row["estimate"] == pytest.approx(primary["estimate"], rel=1e-12)
    assert row["se"] == pytest.approx(primary["se"], rel=1e-12)
    design_df = frame.groupby(["stratum", "psu"]).ngroups - frame["stratum"].nunique()
    for key in ("model_2", "model_3"):
        record = seq[key]["inference"]["missing"]
        assert seq[key]["inference"]["covariance"] == "design", key
        assert (record["design_strata"], record["design_psu"], record["design_weight"]) == \
            ("stratum", "psu", "w"), key
        assert record["df_com"] == design_df, key
    assert seq["model_3"]["n_rows"] == len(frame)  # imputed, so every analyzed row


# ── one SD for a difference's E-value (ruling 14) ────────────────────────────

SVYVAR_EVALUE_R = """
suppressPackageStartupMessages({library(survey); library(EValue)})
d <- read.csv(rows_csv)
des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE, data = d)
sd <- sqrt(coef(svyvar(~glucose, des))[[1]])
est <- read.csv(estimates_csv)
ev <- lapply(seq_len(nrow(est)), function(i) {
  e <- evalues.OLS(est$estimate[i], se = est$se[i], sd = sd, delta = 1)
  lim <- e["E-values", c("lower", "upper")]
  c(e["E-values", "point"], lim[!is.na(lim)][[1]])
})
out(list(sd = sd, sample_sd = sd(d$glucose), point = sapply(ev, `[`, 1),
         limit = sapply(ev, `[`, 2)))
"""


@needs_r
def test_the_effects_stage_and_the_causal_lane_standardize_by_the_populations_sd(tmp_path):
    """MODELING_SEQUENCE §0 ruling 14: under the surveyed-population answer the E-value of a
    difference standardizes it by the design-weighted SD of the outcome, in the effects stage and in
    the causal lane alike. Reference: R ``survey::svyvar`` on the design over the analyzed rows,
    and ``EValue::evalues.OLS`` of each stage's own estimate and standard error with that SD; 1e-8.
    The weights make that SD more than 1% from the rows' (checked), far beyond the tolerance."""
    from turbotab.core import causal
    from turbotab.core.stages.causal import causal_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = _surveyed()
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  roles=SURVEY_ROLES, survey=SURVEY, amounts=SURVEY_AMOUNTS)
    run = ef.run(frame, tmp_path / "effects", st, fit=False)
    [sens] = run["effects"]["families"][0]["sensitivity"]
    [two] = [r for r in ef.sequence(run["effects"])["model_2"]["effects"]
             if r["feature"] == "fiber"]

    lane = st.model_copy(update={"causal": d.CausalSpec(
        exposure="fiber", method="dml_plr", learner="linear", folds=5, repetitions=3,
        assumptions=list(causal.ASSUMPTIONS))})
    paths = mf.ingest_frame(frame, tmp_path / "causal")
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=lane.split.seed)
    art = causal_stage(mf.context(lane, {"split": split, "target_info": mf.target_info(
        "regression", "glucose")}, paths)).data
    assert art["withheld"] is None and art["n"] == len(frame)
    [plr] = art["estimates"]

    estimates = pd.DataFrame({"estimate": [two["estimate"], plr["estimate"]],
                              "se": [two["se"], plr["se"]]})
    r = run_r(SVYVAR_EVALUE_R, {"rows": frame, "estimates": estimates}, tmp_path / "r")
    assert abs(r["sd"] / r["sample_sd"] - 1) > 0.01
    for found, i in ((sens, 0), (art["sensitivity"], 1)):
        assert found["e_value"]["measure"] == "OLS"
        assert found["e_value"]["point"] == pytest.approx(r["point"][i], rel=1e-8), i
        assert found["e_value"]["limit"] == pytest.approx(r["limit"][i], rel=1e-8), i


# ── the data's own imputed copies ────────────────────────────────────────────

COPIES_TABLE2 = (
    "The rows are the data's own imputed copies (numbered by `_MULT_`). The fit's table analyzes "
    "each copy with its own outcome and pools them by Rubin's rules, so its primary model is the "
    "estimate. The declared models beside it (the unadjusted model, Model 1 and Model 3), the "
    "marginal risks, the diagnostics and the sensitivity analyses are not built over the copies, so "
    "none is shown: fit on the copies stacked, each participant would count once per copy and every "
    "interval would leave out the variation between the copies.")
COPIES_CAUSAL = (
    "The rows are the data's own imputed copies (numbered by `_MULT_`). The causal lane's "
    "estimators are not pooled over imputed copies in v2, and fit on the copies stacked each "
    "participant would count once per copy; the fit's table analyzes each copy with its own outcome "
    "and pools them by Rubin's rules.")


def test_the_data_s_own_imputed_copies_withhold_table_2_and_the_causal_lane(tmp_path):
    """NHANES DXA's shape (each SEQN five times, ``_MULT_`` 1–5, kept as rows): the fit pools its
    table over the five copies (wave 1b's MS3), while Table 2 and the causal lane, which are not
    built over them, are withheld with the reason, never fit on the 1,200 stacked rows as if they
    were 1,200 participants (before this, Model 2's standard error there was the stacked rows'
    cluster-robust one, below the pooled table's). The card is not offered; a causal answer already
    recorded is withheld with its way forward, the primary model only. Each is the contract's
    relation."""
    from turbotab.core import causal, contracts
    from turbotab.core.decisions import GrainSpec, RepeatSpec
    from turbotab.core.stages.causal import causal_design_stage, causal_stage
    from turbotab.core.tests import modeling_fixtures as mf
    from turbotab.core.tests.acceptance.test_ms1_ms3_multiple_imputation import dxa_frame

    frame = dxa_frame()
    st = ef.state(target="DXDTOPF", task="regression", exposure="fiber_g",
                  measure="mean_difference",
                  roles={"RIDAGEYR": "covariate", "fiber_g": "exposure", "SEQN": "identifier"},
                  answers={"RIDAGEYR": ef.CONFOUNDER}, amounts={"code_or_count:_MULT_": "code"},
                  grain=GrainSpec(grain="repeated", id_column="SEQN"),
                  repeat_kind=RepeatSpec(repeat_kind="imputed_copies", implicate_column="_MULT_"),
                  unit="row",
                  causal=d.CausalSpec(exposure="fiber_g", method="dml_plr", learner="linear",
                                      assumptions=list(causal.ASSUMPTIONS)))
    run = ef.run(frame, tmp_path / "run", st)
    assert run["fit_raw"]["models"][0]["inference"]["missing"]["model"] == "supplied"
    effects = run["effects"]
    assert effects["families"] == [] and effects["methods"] == COPIES_TABLE2

    paths = mf.ingest_frame(frame, tmp_path / "causal")
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, groups=frame["SEQN"].to_numpy(),
                            grouped_by="SEQN")
    ctx = mf.context(st, {"split": split, "target_info": mf.target_info("regression", "DXDTOPF")},
                     paths)
    card = causal_design_stage(ctx).data
    assert card["offered"] is False and card["reason"] == COPIES_CAUSAL
    lane = causal_stage(ctx).data
    assert lane["withheld"] == COPIES_CAUSAL and lane["estimates"] == []
    [way] = lane["exits"]
    assert way["label"] == "Keep the primary model only, pooled over the copies by the fit"
    assert way["decision"]["kind"] == "set_causal" and way["decision"]["method"] == "none"

    sequence = contracts.contracts()["model_sequence"]
    [relation] = [r for r in sequence.relations if r.id == "sequence-supplied-copies"]
    assert relation.kind == "disables"
    supplied = causal.RELATIONS["supplied_copies"]
    assert supplied.kind == "conflicts" and supplied.rung == "refused" and supplied.exits
    for key in causal.METHODS:
        assert "supplied_copies" in {r.name for r in contracts.contracts()[key].relations}, key
