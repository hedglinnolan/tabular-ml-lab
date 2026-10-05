"""REPAIR-SCALES: the independent verifier's findings on MS8, each a permanent acceptance test
(docs/turbotab-next/MODELING_SEQUENCE.md §0 rulings 7–8, §2, §4; BLUEPRINT §13, §14.3).

    (1) Codes counted as answers, everywhere the answer reads values (§4 and the contract's
        "conflicts" relation; §14.3): a repeat administration recorded as one score column must lie
        in the range that score can take, and a calibration substudy's reference measure must have
        a settled amount reading and hold no missing-value code. Each refusal's exits work, and a
        code that reaches the values after the answer (a repair undone) blocks the correction at
        the stage.
    (2) Repeated units or clusters imply cluster-aware intervals (§2; ruling 7, "a bootstrap over
        the whole chain, resampling PSUs within strata, or clusters"): the uncorrected interval
        beside the correction is the coefficient table's CR2 interval, the correction's bootstrap
        resamples whole clusters, and too few clusters block the correction with the table's exits.
    (3) §2 *invalidates*: an answer that takes a scale's item out of the predictors, or puts its
        repeat administration or reference into the model, is refused until the scales answer
        changes, with exits that work; the project is never left to fail in the pipeline.
    (4) §2 "multivariate calibration when several intakes are error-prone" (ruling 7, Rosner):
        several corrected scores are calibrated jointly.
    (5) The leash too tight: a formative index whose components are continuous scores (the
        HEI-2015's) is declared and corrected from its repeat; a blank item is named as blank.

**References, each independent of the code under test:** R's ``psych::omega``, ``psych::ICC``,
``mecor::mecor`` and ``clubSandwich::coef_test`` (CR2, Satterthwaite) run as a subprocess on CSVs
written here (skipped without R); NumPy by hand for the calibrations (the Schur complement of the
covariance matrix; the multivariate formula E[X | W, Z] = fitted + (S − D) S⁻¹ r), the score's
range (k answers on the response scale) and the documented bootstrap draws; statsmodels for least
squares; and simulations with a known truth (continuous congeneric items, whose true score and
coefficient are known exactly).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d
from turbotab.core.methods import scales as S
from turbotab.core.tests.acceptance.scales_fixtures import (DQ_ITEMS, DQ_RETEST, HEI, HEI_RETEST,
                                                            PSS_ITEMS, PSS_REVERSE, SAT,
                                                            SAT_REVERSE, calibrate_by_hand,
                                                            hei_table, household_scale_table,
                                                            keyed, linear_scale_table, needs_r,
                                                            omega_script, run_r,
                                                            survey_scale_table)
from turbotab.core.tests.acceptance.server_drive import Truth, local_server, open_project

SAT_SCALE = {"name": "sat_score", "items": SAT, "reverse": SAT_REVERSE, "low": 1, "high": 5,
             "kind": "reflective", "correction": "regression_calibration", "n_boot": 50}
# WP17 asks the adjustment set column by column, before the scales answer (not yet one of the
# Router's questions): every predictor here causes the exposure the card leads with and the
# outcome, so all of them stay in the model.
CONFOUNDS = "yes,yes,no"


def _drive(client, path: Path, roles: dict[str, str], target: str, task: str,
           readings: dict[str, str] | None = None, *, grouped: bool = False):
    """A project through the real server to the scales answer: inference, every row analyzed,
    complete cases, the roles as their author gives them. ``grouped``: each row is still a
    different person, in households the cluster question names (the grain answer acknowledged
    over the repeating grouping)."""
    predictors = [c for c, r in roles.items() if r in ("covariate", "exposure")]
    drive = open_project(client, path, Truth({**{f"adjust:{c}": CONFOUNDS for c in predictors},
                                              **(readings or {})}, fixture=path.name))
    drive.decide({"kind": "set_lens", "lenses": ["survey"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": target})
    drive.answer("task", {"kind": "set_task", "column": target, "task": task})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "inference"})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit",
                           "acknowledged": grouped})
    drive.reach("roles")
    drive.decide_roles(roles)
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0})
    return drive


def _store(frame: pd.DataFrame, tmp_path: Path, name: str):
    from turbotab.core.datastore import DataStore, ingest

    src = tmp_path / f"{name}.csv"
    frame.to_csv(src, index=False)
    dest = src.with_suffix(".parquet")
    ingest(src, dest)
    return DataStore(dest, 2 << 30)


def _refused(body: dict, ctx: dict) -> d.Refusal:
    with pytest.raises(d.Refusal) as refused:
        d.validate(body, ctx)
    return refused.value


# ── (1) codes counted as answers, beyond the items ────────────────────────────

DQ_SCALE = {"name": "dq_score", "items": DQ_ITEMS, "low": 0, "high": 10, "kind": "formative",
            "reliability": "test_retest", "retest": ["dq_t2_score"],
            "correction": "regression_calibration", "n_boot": 50}


def _dq_state() -> d.ProjectState:
    roles = {"participant_id": "identifier", "age": "covariate", "sex": "covariate",
             **{c: "covariate" for c in PSS_ITEMS + DQ_ITEMS},
             **{c: "excluded" for c in [*DQ_RETEST, "dq_t2_score"]}}
    return d.ProjectState(purpose="inference", target="wellbeing", roles=roles)


def test_1_a_repeat_score_holding_one_code_is_refused_with_exits_that_work(tmp_path):
    """The verifier's S7: one code of 77 among the repeats of a 0–10 × 6 diet score, recorded as
    one score column, was read as a score (an ICC of 0.00 printed). The score's range is the
    instrument's own, by hand: 6 answers on 0–10, summed, 0 to 60, so 77 is outside it and the
    answer is refused, naming both. The recurrence rule keeps the findings from reading a single
    code, so no repair is offered; the exits are the repeat's own items (found by their suffix)
    and the uncorrected estimate, and each passes the leash."""
    frame = survey_scale_table(n=600)
    frame["dq_t2_score"] = frame[DQ_RETEST].sum(axis=1, skipna=False)
    row = frame.index[frame["dq_t2_score"].notna()][3]
    frame.loc[row, "dq_t2_score"] = 77
    with _store(frame, tmp_path, "dq") as store:
        ctx = {"state": _dq_state(), "columns": list(frame.columns), "store": store}
        error = _refused({"kind": "set_scales", "scales": [DQ_SCALE]}, ctx)
        x = frame["dq_t2_score"]
        assert error.code == "retest_outside_the_score"
        assert error.message == (
            f"`dq_t2_score`, the repeat administration of `dq_score`, holds values from "
            f"{x.min():g} to {x.max():g}, outside the {6 * 0:g}–{6 * 10:g} its score can take (6 "
            f"answers on the 0–10 response scale, summed). A value outside it is a code ('don't "
            f"know', 'refused', not administered); counted as a score it moves the test–retest ICC "
            f"and the correction. Recode the codes to missing first.")
        items, uncorrected = error.exits
        assert items["label"] == "Score the repeat administration from its items (`dq_1_t2` and " \
                                 "`dq_2_t2` …)"
        assert items["decision"]["scales"][0]["retest"] == DQ_RETEST
        assert uncorrected["decision"]["scales"][0]["correction"] == "none"
        assert uncorrected["decision"]["scales"][0]["retest"] == []
        for exit_ in error.exits:
            d.validate(exit_["decision"], ctx)
    # The same score recorded as the mean of the answers can take 0–10: 77/6 lies outside it.
    averaged = frame.assign(dq_t2_score=frame[DQ_RETEST].mean(axis=1, skipna=False))
    averaged.loc[row, "dq_t2_score"] = 77 / 6
    with _store(averaged, tmp_path, "dq_mean") as store:
        ctx = {"state": _dq_state(), "columns": list(averaged.columns), "store": store}
        error = _refused({"kind": "set_scales", "scales": [{**DQ_SCALE, "scoring": "mean"}]}, ctx)
        assert error.code == "retest_outside_the_score"
        assert "outside the 0–10 its score can take (6 answers on the 0–10 response scale, " \
               "averaged)" in error.message
    clean = frame.assign(dq_t2_score=frame[DQ_RETEST].sum(axis=1, skipna=False))
    with _store(clean, tmp_path, "dq_clean") as store:
        d.validate({"kind": "set_scales", "scales": [DQ_SCALE]},
                   {"state": _dq_state(), "columns": list(clean.columns), "store": store})


@needs_r
def test_1_a_repeated_code_takes_the_findings_repair_and_the_icc_is_psychs_without_it(tmp_path):
    """999 on four repeats of the diet score's one-column repeat administration: the findings read
    it as a missing-value code, and the refusal's first exit is that repair. Posted, it blanks the
    codes in the working table; the same answer is then recorded, and the test–retest ICC and the
    error variance are psych::ICC's and var(W₁ − W₂)/2 by hand on the CSV with the codes as NA
    (to 1e-9). Counted as scores, the codes would have given another ICC entirely."""
    frame = survey_scale_table(n=600)
    frame["dq_t2_score"] = frame[DQ_RETEST].sum(axis=1, skipna=False)
    rows = frame.index[frame["dq_t2_score"].notna()][:4]
    frame.loc[rows, "dq_t2_score"] = 999
    frame.to_csv(tmp_path / "dq999.csv", index=False)
    roles = {"participant_id": "identifier", "age": "covariate", "sex": "covariate",
             **{c: "covariate" for c in DQ_ITEMS},
             **{c: "excluded" for c in [*PSS_ITEMS, *DQ_RETEST, "dq_t2_score"]}}
    with local_server(tmp_path / "srv") as client:
        drive = _drive(client, tmp_path / "dq999.csv", roles, "wellbeing", "ordinal",
                       {"code_or_count:age": "amount"})
        drive.artifact("findings")
        r = drive.post({"kind": "set_scales", "scales": [DQ_SCALE]})
        assert r.status_code == 409, r.text
        error = r.json()["error"]
        assert error["code"] == "retest_outside_the_score"
        repair = error["exits"][0]["decision"]
        assert repair == {"kind": "apply_repair", "finding_id": "sentinel_missing__dq_t2_score",
                          "option": "set_missing", "params": {}}
        drive.decide(repair)
        drive.artifact("working")
        drive.decide({"kind": "set_scales", "scales": [DQ_SCALE]})
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["proportional_odds"]})
        (dq,) = drive.artifact("scales")["scales"]
    W1 = frame[DQ_ITEMS].sum(axis=1)
    W2 = frame["dq_t2_score"].where(frame["dq_t2_score"] != 999)
    pd.DataFrame({"W1": W1, "W2": W2}).to_csv(tmp_path / "rt.csv", index=False)
    ref = run_r("""
suppressMessages({library(psych); library(jsonlite)})
rt <- read.csv("rt.csv"); rt <- rt[!is.na(rt$W2), ]
icc <- suppressWarnings(suppressMessages(ICC(rt, lmer = FALSE)))
coded <- read.csv("rt.csv")
writeLines(toJSON(list(icc3 = icc$results["Single_fixed_raters", "ICC"], n = nrow(rt)),
                  digits = NA, auto_unbox = TRUE), "out.json")
""", tmp_path)
    both = W2.notna()
    assert dq["reliability"]["value"] == pytest.approx(ref["icc3"], abs=1e-9)
    assert dq["reliability"]["n"] == ref["n"] == int(both.sum())
    assert dq["correction"]["error_variance"] == pytest.approx(
        float(np.var((W1 - W2)[both], ddof=1) / 2), rel=1e-10)
    # What the codes would have done: the ICC with 999 counted as four scores, by hand.
    coded = frame["dq_t2_score"].notna()
    X = np.column_stack([W1[coded], frame.loc[coded, "dq_t2_score"]])
    n = len(X)
    grand = X.mean()
    ms_r = 2 * ((X.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_e = (((X - grand) ** 2).sum() - 2 * ((X.mean(axis=1) - grand) ** 2).sum()
            - n * ((X.mean(axis=0) - grand) ** 2).sum()) / (n - 1)
    assert abs((ms_r - ms_e) / (ms_r + ms_e) - ref["icc3"]) > 0.3


REF_SCALE = {**SAT_SCALE, "reliability": "calibration_substudy", "reference": "sat_ref"}
LIN_ROLES = {"pid": "identifier", "age": "covariate", "bmi": "covariate", "sat_ref": "excluded",
             **{c: "covariate" for c in SAT}}


def _lin_ctx(frame: pd.DataFrame, store, **state) -> dict:
    return {"state": d.ProjectState(purpose="inference", target="sbp", roles=dict(LIN_ROLES),
                                    **state),
            "columns": list(frame.columns), "store": store}


def test_1_a_reference_with_one_code_or_an_unsettled_reading_is_refused_with_exits(tmp_path):
    """BLUEPRINT §14.3: the calibration reads the substudy's reference measure as an amount.
    (a) One 999 beyond a reference that runs about 5–45 (by hand: the gap is more than ten
    spacings and half the range) is refused, naming it; its exits are another reference, ω
    instead, or the uncorrected estimate, and each passes the leash. (b) A reference recorded in
    whole numbers is not settled as an amount by its values (the readings ledger's rule), so it
    is asked: the exit confirms it, after which the answer passes. (c) A reference confirmed as
    codes is refused."""
    frame = linear_scale_table(n=600)
    one = frame.copy()
    one.loc[one.index[one["sat_ref"].notna()][5], "sat_ref"] = 999
    rest = one["sat_ref"][(one["sat_ref"] != 999) & one["sat_ref"].notna()]
    gaps = np.diff(np.sort(rest.unique()))
    assert 999 - rest.max() >= max(10 * np.median(gaps[gaps > 0]), 0.5 * (rest.max() - rest.min()))
    with _store(one, tmp_path, "one") as store:
        ctx = _lin_ctx(one, store)
        error = _refused({"kind": "set_scales", "scales": [REF_SCALE]}, ctx)
        assert error.code == "reference_codes"
        assert error.message == (
            "`sat_ref`, the reference measure of `sat_score`, holds 999, beyond every other value "
            "with a wide gap between: that is how a missing-value code ('don't know', not "
            "measured) looks, and one such value moves the calibration's slope. Recode it to "
            "missing first.")
        assert [e["label"] for e in error.exits] == [
            "Name another reference measure",
            "Correct `sat_score` from its internal consistency (ω) instead",
            "Report `sat_score`'s uncorrected estimate only"]
        for exit_ in error.exits[1:]:
            d.validate(exit_["decision"], ctx)
    whole = frame.assign(sat_ref=frame["sat_ref"].round())
    with _store(whole, tmp_path, "whole") as store:
        ctx = _lin_ctx(whole, store)
        error = _refused({"kind": "set_scales", "scales": [REF_SCALE]}, ctx)
        assert error.code == "reference_unsettled"
        confirm = error.exits[0]["decision"]
        assert confirm == {"kind": "confirm_reading", "reading": "code_or_count",
                           "column": "sat_ref", "value": "amount"}
        settled = _lin_ctx(whole, store,
                           reading_confirmations={"code_or_count:sat_ref": "amount"})
        d.validate({"kind": "set_scales", "scales": [REF_SCALE]}, settled)
        coded = _lin_ctx(whole, store, reading_confirmations={"code_or_count:sat_ref": "code"})
        assert _refused({"kind": "set_scales", "scales": [REF_SCALE]}, coded).code == \
            "reference_is_codes"
    with _store(frame, tmp_path, "clean") as store:
        d.validate({"kind": "set_scales", "scales": [REF_SCALE]}, _lin_ctx(frame, store))


@needs_r
def test_1_a_repeated_reference_code_takes_the_repair_and_a_code_that_returns_blocks_the_stage(
        tmp_path):
    """Three 999s in the reference: the findings read them, the refusal's first exit is the
    repair; once applied the substudy answer is recorded and its corrected coefficient is mecor's
    internal-validation calibration on the CSV with the codes as NA (to 1e-6). The repair then
    undone, the codes reach the reference again: the stage blocks the correction (it never reads
    them as amounts), says why, and its exit (the uncorrected estimate) is a decision that is
    accepted."""
    frame = linear_scale_table(n=600)
    rows = frame.index[frame["sat_ref"].notna()][:3]
    frame.loc[rows, "sat_ref"] = 999
    frame.to_csv(tmp_path / "ref999.csv", index=False)
    with local_server(tmp_path / "srv") as client:
        drive = _drive(client, tmp_path / "ref999.csv", LIN_ROLES, "sbp", "regression",
                       {"code_or_count:age": "amount", "code_or_count:bmi": "amount"})
        drive.artifact("findings")
        r = drive.post({"kind": "set_scales", "scales": [REF_SCALE]})
        assert r.status_code == 409, r.text
        error = r.json()["error"]
        assert error["code"] == "reference_codes"
        repair = error["exits"][0]["decision"]
        assert repair["finding_id"] == "sentinel_missing__sat_ref"
        drive.decide(repair)
        repaired = drive.view()["decisions"][-1]["id"]
        drive.artifact("working")
        drive.decide({"kind": "set_scales", "scales": [REF_SCALE]})
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["linear"]})
        (scale,) = drive.artifact("scales")["scales"]
        drive.decide({"kind": "revert", "decision_id": repaired})
        (returned,) = drive.artifact("scales")["scales"]
        exit_ = returned["exits"][-1]["decision"]
        drive.decide(exit_)
        (after,) = drive.artifact("scales")["scales"]
    W = keyed(frame, SAT, SAT_REVERSE, 1, 5).sum(axis=1)
    frame.assign(W=W, sat_ref=frame["sat_ref"].where(frame["sat_ref"] != 999))[
        ["sbp", "W", "age", "bmi", "sat_ref"]].to_csv(tmp_path / "sub.csv", index=False)
    ref = run_r("""
suppressMessages({library(mecor); library(jsonlite)})
dd <- read.csv("sub.csv")
m <- mecor(sbp ~ MeasError(substitute = W, reference = sat_ref) + age + bmi, data = dd)
writeLines(toJSON(list(mecor = unname(m$corfit$coef["sat_ref"])), digits = NA,
                  auto_unbox = TRUE), "out.json")
""", tmp_path)
    assert scale["correction"]["estimate"] == pytest.approx(ref["mecor"], rel=1e-6)
    assert scale["reliability"]["n"] == int(frame["sat_ref"].notna().sum()) - 3
    assert returned["correction"] is None
    assert returned["not_corrected"] == (
        "`sat_ref`, the reference measure, holds 999, beyond every other value with a wide gap: a "
        "missing-value code, which would move the calibration's slope, so the correction is not "
        "computed.")
    assert returned["methods"].endswith(returned["not_corrected"])
    assert exit_["scales"][0]["correction"] == "none" and exit_["scales"][0]["reference"] is None
    assert after["correction"] is None and after["not_corrected"] is None


# ── (2) cluster-aware intervals ──────────────────────────────────────────────


def _cluster_draws(codes: np.ndarray, B: int, seed: int) -> list[np.ndarray]:
    """The documented draws, written here from the docstring: ``default_rng(seed)``, per replicate
    ``integers(0, G, G)`` clusters, each drawn cluster's rows in table order."""
    rng = np.random.default_rng(seed)
    G = int(codes.max()) + 1
    members = [np.flatnonzero(codes == g) for g in range(G)]
    return [np.concatenate([members[g] for g in rng.integers(0, G, G)]) for _ in range(B)]


def _omega_by_replicate(items: np.ndarray, draws: list[np.ndarray], tmp_path: Path) -> list[float]:
    """The score's ω-total on each replicate's rows, by psych::omega (one factor)."""
    width = max(len(r) for r in draws)
    pd.DataFrame(items, columns=SAT).to_csv(tmp_path / "items.csv", index=False)
    pd.DataFrame([np.pad(r + 1, (0, width - len(r))) for r in draws]).to_csv(
        tmp_path / "draws.csv", index=False)
    return run_r("""
suppressMessages({library(psych); library(jsonlite)})
x <- read.csv("items.csv"); dr <- as.matrix(read.csv("draws.csv"))
om <- sapply(seq_len(nrow(dr)), function(b) {
  rows <- dr[b, ]; xb <- x[rows[rows > 0], ]
  o <- suppressWarnings(suppressMessages(omega(xb, nfactors = 1, plot = FALSE, flip = FALSE)))
  s <- apply(xb, 2, sd); 1 - sum(s^2 * o$schmid$sl[, "u2"]) / sum(cov(xb))
})
writeLines(toJSON(list(omega = om), digits = NA), "out.json")
""", tmp_path)["omega"]


def _by_hand(items: np.ndarray, W: np.ndarray, Z: np.ndarray, y: np.ndarray,
             draws: list[np.ndarray], omegas: list[float]) -> list[float]:
    out = []
    for rows, om in zip(draws, omegas):
        Wb, Zb, yb = W[rows], Z[rows], y[rows]
        X_hat, _ = calibrate_by_hand(Wb, Zb, (1 - om) * np.var(Wb, ddof=1))
        out.append(sm.OLS(yb, sm.add_constant(np.column_stack([X_hat, Zb]))).fit().params[1])
    return out


@needs_r
def test_2_the_bootstrap_resamples_whole_clusters_as_documented(tmp_path):
    """Ruling 7: with each row's cluster given, a replicate draws G clusters with replacement and
    keeps every row of each. The draws written here from the docstring, R's psych::omega on each
    replicate's rows, and NumPy's calibration by hand give the app's 40 replicate estimates and
    their standard error to 1e-5."""
    frame = household_scale_table(households=60)
    items = keyed(frame, SAT, SAT_REVERSE, 1, 5).to_numpy()
    W, Z, y = items.sum(axis=1), frame[["age", "bmi"]].to_numpy(float), frame["sbp"].to_numpy()
    codes = pd.factorize(frame["hh"])[0]
    draws = _cluster_draws(codes, 40, seed=3)
    by_hand = _by_hand(items, W, Z, y, draws, _omega_by_replicate(items, draws, tmp_path))
    source = S.Source("internal_consistency",
                      reliability=S.internal_consistency(items, "sum", "unidimensional", 1))
    got = S.correct(W, Z, y, 0, source, n_boot=40, seed=3, groups=codes)
    assert got.n_boot == got.n_boot_ok == 40 and got.n_clusters == 60
    np.testing.assert_allclose(got.boot, by_hand, rtol=1e-5)
    assert got.se == pytest.approx(np.std(by_hand, ddof=1), rel=1e-5)


def clustered_congeneric(rng: np.random.Generator, G: int, m: int):
    """Known truth: G clusters of m rows; a true score F (70% of its variance shared within a
    cluster) measured by five continuous congeneric items, a covariate z, and an outcome 1.5 per
    unit of the true score T = (Σλ) F whose error (SD 8) is also 70% shared within a cluster."""
    lam, psi = [0.9, 0.8, 0.7, 0.8, 0.8], [0.7, 0.9, 1.0, 0.8, 0.8]
    n, g = G * m, np.repeat(np.arange(G), m)
    F = math.sqrt(0.7) * rng.standard_normal(G)[g] + math.sqrt(0.3) * rng.standard_normal(n)
    z = 0.5 * F + rng.standard_normal(n)
    X = np.column_stack([l * F + math.sqrt(p) * rng.standard_normal(n) for l, p in zip(lam, psi)])
    y = (1.5 * sum(lam) * F + z + 8 * math.sqrt(0.7) * rng.standard_normal(G)[g]
         + 8 * math.sqrt(0.3) * rng.standard_normal(n))
    return X, z, y, g


def test_2_by_simulation_the_cluster_bootstrap_covers_where_the_row_bootstrap_does_not():
    """Known truth (1.5 per unit of true score; the score and the outcome's error each 70% shared
    within clusters of 10). Over 120 datasets of 60 clusters, the 95% interval from 80 replicates
    of whole clusters covers 1.5 in at least 89% of them (the binomial Monte Carlo error at 0.95
    is 0.02; observed 0.95); the same correction's interval from rows resampled as independent
    covers it in fewer than 85% (observed 0.76, its standard error about 1.7 times smaller, as in
    the verifier's S5a, 0.0038 against 0.0069)."""
    rng = np.random.default_rng(404)
    cluster_hits = row_hits = 0
    reps = 120
    for r in range(reps):
        X, z, y, g = clustered_congeneric(rng, 60, 10)
        source = S.Source("internal_consistency",
                          reliability=S.internal_consistency(X, "sum", "unidimensional", 1))
        W = X.sum(axis=1)
        whole = S.correct(W, z[:, None], y, 0, source, n_boot=80, seed=r, groups=g)
        rows = S.correct(W, z[:, None], y, 0, source, n_boot=80, seed=r)
        cluster_hits += whole.ci_low <= 1.5 <= whole.ci_high
        row_hits += rows.ci_low <= 1.5 <= rows.ci_high
    print(f"\ncoverage: clusters {cluster_hits / reps:.3f} | rows {row_hits / reps:.3f}")
    assert cluster_hits / reps >= 0.89
    assert row_hits / reps < 0.85


HH_ROLES = {"pid": "identifier", "hh": "cluster", "age": "covariate", "bmi": "covariate",
            **{c: "covariate" for c in SAT}}


def _households(client, path: Path):
    drive = _drive(client, path, HH_ROLES, "sbp", "regression",
                   {"code_or_count:age": "amount", "code_or_count:bmi": "amount"}, grouped=True)
    drive.decide({"kind": "set_scales", "scales": [SAT_SCALE]})
    drive.reach("models")
    drive.decide({"kind": "select_models", "models": ["linear"]})
    return drive


@needs_r
def test_2_through_the_server_both_intervals_keep_each_households_rows_together(tmp_path):
    """The verifier's S5a: 80 households of 10, the grouping answered (``set_clusters`` hh,
    intervals only). (a) The uncorrected interval beside the correction is the fit table's own
    row (to 1e-10), which is R's clubSandwich CR2 interval with Satterthwaite degrees of freedom
    (to 1e-6), and wider than the interval on independent rows (statsmodels OLS). (b) The
    correction's 50 replicates resample whole households: the documented draws over the analyzed
    rows, psych::omega on each, and NumPy's calibration by hand give its standard error (to 1e-5).
    (c) The labels and the methods sentence say so, verbatim; no concern says the rows are treated
    as independent."""
    frame = household_scale_table()
    frame.to_csv(tmp_path / "hh.csv", index=False)
    with local_server(tmp_path / "srv") as client:
        drive = _households(client, tmp_path / "hh.csv")
        state = drive.view()["state"]
        scales = drive.artifact("scales")
        fit = drive.artifact("fit")
    assert state["clusters"]["column"] == "hh"
    (scale,) = scales["scales"]
    corr = scale["correction"]
    row = next(c for c in fit["models"][0]["coefficients"] if c["feature"] == "sat_score")
    items = keyed(frame, SAT, SAT_REVERSE, 1, 5)
    W = items.sum(axis=1).to_numpy()
    Z = frame[["age", "bmi"]].to_numpy(dtype=float)
    y = frame["sbp"].to_numpy()
    frame.assign(W=W)[["sbp", "W", "age", "bmi", "hh"]].to_csv(tmp_path / "lin.csv", index=False)
    ref = run_r("""
suppressMessages({library(clubSandwich); library(jsonlite)})
dd <- read.csv("lin.csv")
fit <- lm(sbp ~ W + age + bmi, data = dd)
ct <- coef_test(fit, vcov = "CR2", cluster = dd$hh, test = "Satterthwaite")
df <- if ("df_Satt" %in% names(ct)) ct$df_Satt[2] else ct$df[2]
se <- ct$SE[2]; b <- unname(coef(fit)["W"])
writeLines(toJSON(list(lo = b - qt(0.975, df) * se, hi = b + qt(0.975, df) * se, b = b),
                  digits = NA, auto_unbox = TRUE), "out.json")
""", tmp_path)
    assert corr["naive"] == pytest.approx(row["estimate"], rel=1e-10)
    assert (corr["naive_ci_low"], corr["naive_ci_high"]) == (pytest.approx(row["ci_low"], rel=1e-10),
                                                             pytest.approx(row["ci_high"], rel=1e-10))
    assert corr["naive_ci_low"] == pytest.approx(ref["lo"], rel=1e-6)
    assert corr["naive_ci_high"] == pytest.approx(ref["hi"], rel=1e-6)
    lo, hi = sm.OLS(y, sm.add_constant(np.column_stack([W, Z]))).fit().conf_int()[1]
    assert corr["naive_ci_high"] - corr["naive_ci_low"] > 1.3 * (hi - lo)
    # (b) the replicates: seed 0 (the split's), the households coded in the analyzed rows' order
    codes = pd.factorize(frame["hh"])[0]
    draws = _cluster_draws(codes, 50, seed=0)
    by_hand = _by_hand(items.to_numpy(), W, Z, y, draws,
                       _omega_by_replicate(items.to_numpy(), draws, tmp_path))
    assert corr["se"] == pytest.approx(np.std(by_hand, ddof=1), rel=1e-5)
    assert (corr["clustered_by"], corr["n_clusters"], corr["n_boot_ok"]) == ("hh", 80, 50)
    # (c) the words
    assert ("Both intervals keep each `hh`'s rows together: the uncorrected one is the coefficient "
            "table's cluster-robust interval (CR2), and the corrected one's bootstrap resamples "
            "whole `hh` clusters (80 of them).") in corr["labels"]
    rel = scale["reliability"]
    assert scale["methods"] == (
        "`sat_score` was the sum of its 8 items, `sat_3` and `sat_6` reverse-coded on the 1–5 "
        f"response scale. Its reliability was ω-total = {rel['value']:.2f} for the score "
        "(McDonald's ω; minimum-residual factor analysis of the item correlations, one factor; "
        f"Cronbach's α = {rel['alpha']:.2f}, reported as customary). The corrected coefficient was "
        "obtained by regression calibration including all model covariates, with bootstrap CIs "
        "re-estimating the reliability (50 replicates, resampling whole `hh` clusters); "
        "uncorrected and corrected estimates are both reported. Both intervals are clustered by "
        "`hh`. A reliability from internal consistency omits transient error, so the correction "
        "under-corrects.")
    assert not any("independent" in c for c in scale["concerns"])


def test_2_too_few_clusters_block_the_correction_with_the_tables_own_exits(tmp_path):
    """Six households, under the eight units TurboTab requires for cluster-robust intervals: the
    fit table reports no interval and says why; the correction, whose interval would rest on the
    same six, is blocked and recorded with that reason and those exits (the reliability, which
    describes the items, stays)."""
    frame = household_scale_table(households=6, size=100)
    frame.to_csv(tmp_path / "few.csv", index=False)
    with local_server(tmp_path / "srv") as client:
        drive = _households(client, tmp_path / "few.csv")
        (scale,) = drive.artifact("scales")["scales"]
        info = drive.artifact("fit")["models"][0]["inference"]
    assert info["refused"] and scale["correction"] is None
    assert scale["not_corrected"] == info["refused"]
    assert scale["exits"] == info["exits"]
    assert scale["reliability"]["value"] is not None


# ── (3) an item leaving the predictors invalidates the scales answer ──────────


@needs_r
def test_3_an_item_excluded_after_the_scales_answer_reopens_it_and_its_exit_recomputes(tmp_path):
    """The verifier's S6a: excluding an item after the scales answer was accepted, the answer kept
    silently and the design stage failed. Now the exclusion is refused (§2 *invalidates*), saying
    why, with three exits: take the item out of the scale, stop scoring it, or keep the answers.
    Posting the first, then the exclusion, both are recorded; the score is the sum of the seven
    items left, its ω psych::omega's on them (to 1e-5), and the fit's table holds the score. A
    role change of a column outside the scale is not refused. Making the substudy's reference a
    predictor, the outcome an item, or reverting the roles are refused too."""
    frame = linear_scale_table(n=600)
    frame.to_csv(tmp_path / "lin.csv", index=False)
    scale = {**SAT_SCALE, "correction": "none"}
    with local_server(tmp_path / "srv") as client:
        drive = _drive(client, tmp_path / "lin.csv", LIN_ROLES, "sbp", "regression",
                       {"code_or_count:age": "amount", "code_or_count:bmi": "amount"})
        roles_record = next(r["id"] for r in reversed(drive.view()["decisions"])
                            if r["decision"]["kind"] == "set_roles")
        drive.decide({"kind": "set_scales", "scales": [scale]})
        r = drive.post({"kind": "set_roles", "roles": {**LIN_ROLES, "sat_3": "excluded"}})
        assert r.status_code == 409, r.text
        error = r.json()["error"]
        assert error["code"] == "scale_invalidated"
        assert error["message"] == (
            "`sat_3` is an item of `sat_score`, whose score replaces its items in the models. This "
            "answer would take it out of the predictors (no longer a confirmed exposure or "
            "covariate, left out, or the outcome), so the score could not be formed as declared. "
            "The scales answer rests on it, so it changes first and is asked again, never kept "
            "silently: take the column out of the scale or stop scoring the scale, then record "
            "this answer.")
        assert [e["label"] for e in error["exits"]] == [
            "Take `sat_3` out of `sat_score` (7 items remain), then record this answer",
            "Stop scoring `sat_score` (its items enter the models on their own), then record this "
            "answer", "Keep the answers as they are"]
        taken = error["exits"][0]["decision"]
        assert taken["scales"][0]["items"] == [c for c in SAT if c != "sat_3"]
        assert taken["scales"][0]["reverse"] == ["sat_6"]
        target = drive.post({"kind": "set_target", "column": "sat_1"})
        revert = drive.post({"kind": "revert", "decision_id": roles_record})
        drive.decide(taken)
        drive.decide_roles({**LIN_ROLES, "sat_3": "excluded"})
        drive.decide_roles({**LIN_ROLES, "sat_3": "excluded", "bmi": "excluded"})
        drive.decide_roles({**LIN_ROLES, "sat_3": "excluded"})
        drive.decide({"kind": "set_scales", "scales": [
            {**taken["scales"][0], "correction": "regression_calibration",
             "reliability": "calibration_substudy", "reference": "sat_ref"}]})
        ref_in = drive.post({"kind": "set_roles", "roles": {**LIN_ROLES, "sat_3": "excluded",
                                                            "sat_ref": "covariate"}})
        drive.decide({"kind": "set_scales", "scales": [taken["scales"][0]]})
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["linear"]})
        (scored,) = drive.artifact("scales")["scales"]
        table = drive.artifact("fit")["models"][0]["coefficients"]
    assert target.status_code == 409 and target.json()["error"]["code"] == "scale_invalidated"
    assert revert.status_code == 409 and revert.json()["error"]["code"] == "scale_invalidated"
    assert ref_in.status_code == 409 and ref_in.json()["error"]["code"] == "scale_invalidated"
    stop = ref_in.json()["error"]["exits"][0]
    assert stop["label"] == ("Stop reading `sat_ref` as `sat_score`'s reliability, then record "
                             "this answer")
    assert stop["decision"]["scales"][0]["reference"] is None
    seven = [c for c in SAT if c != "sat_3"]
    keyed(frame, seven, ["sat_6"], 1, 5).to_csv(tmp_path / "items.csv", index=False)
    ref = run_r(omega_script("items.csv", 1), tmp_path)
    assert scored["items"] == seven
    assert scored["reliability"]["value"] == pytest.approx(ref["score_omega_tot"], abs=1e-5)
    assert scored["methods"].startswith(
        "`sat_score` was the sum of its 7 items, `sat_6` reverse-coded on the 1–5 response scale.")
    features = {c["feature"] for c in table}
    assert "sat_score" in features and not set(SAT) & features


# ── (4) several corrected scores: multivariate regression calibration ────────


def test_4_joint_calibration_is_rosners_formula_by_hand():
    """Three error-prone scores, two with a known error variance and one with a calibration
    substudy, and two exact covariates. NumPy by hand: the residual covariance S of the scores
    given (1, Z), D = diag of the known error variances, each known-error score's calibration
    fitted + (S − D)_j S⁻¹ r and the substudy score's reference regressed on (1, W, Z) in its rows;
    the app's calibrated scores agree to 1e-10, and each known-error score's is also its univariate
    calibration given every other column (the other scores' raw values among them). The corrected
    coefficients are the least-squares refit with all three calibrated (statsmodels, 1e-10)."""
    rng = np.random.default_rng(12)
    n = 900
    Z = rng.standard_normal((n, 2))
    T = rng.multivariate_normal([0, 0, 0], [[1, .5, .3], [.5, 1, .4], [.3, .4, 1]], n) + Z @ [
        [.3, .1, .2], [.2, .3, .1]]
    W = T + rng.standard_normal((n, 3)) * [0.6, 0.7, 0.5]
    errors = [0.36, 0.49, None]
    reference = np.where(rng.random(n) < 0.3, T[:, 2] + rng.normal(0, 0.2, n), np.nan)
    y = T @ [1.0, 0.5, -0.4] + Z @ [0.3, 0.2] + rng.standard_normal(n)
    got = S.calibrate_jointly(W, Z, errors, [None, None, reference])
    D1 = np.column_stack([np.ones(n), Z])
    F = D1 @ np.linalg.lstsq(D1, W, rcond=None)[0]
    R = W - F
    Sm = R.T @ R / (n - 1)
    D = np.diag([0.36, 0.49, 0.0])
    G = (Sm - D) @ np.linalg.inv(Sm)
    by_hand = F + R @ G.T
    have = np.isfinite(reference)
    D3 = np.column_stack([np.ones(n), W, Z])
    by_hand[:, 2] = D3 @ np.linalg.lstsq(D3[have], reference[have], rcond=None)[0]
    np.testing.assert_allclose(got.calibrated, by_hand, rtol=0, atol=1e-10)
    for j in (0, 1):
        others = np.column_stack([np.delete(W, j, axis=1), Z])
        X_hat, lam = calibrate_by_hand(W[:, j], others, errors[j])
        np.testing.assert_allclose(got.calibrated[:, j], X_hat, rtol=0, atol=1e-10)
        assert got.attenuation[j] == pytest.approx(lam, rel=1e-10)
    sources = [S.Source("internal_consistency", reliability=lambda rows, v=v: (0.0, v))
               for v in errors[:2]] + [S.Source("calibration_substudy", reference=reference)]
    out = S.correct_jointly(W, Z, y, [0, 1, 2], sources, n_boot=0)
    refit = sm.OLS(y, sm.add_constant(np.column_stack([by_hand, Z]))).fit().params[1:4]
    np.testing.assert_allclose([c.estimate for c in out], refit, rtol=1e-10)


def two_scales(rng: np.random.Generator, n: int):
    """Known truth: two true scores (correlated 0.6, both with a covariate z), each measured by five
    continuous congeneric items; the outcome 1.0 per unit of the first true score and 0.5 per unit
    of the second."""
    lam, psi = [0.9, 0.8, 0.7, 0.8, 0.8], [0.7, 0.9, 1.0, 0.8, 0.8]
    z = rng.standard_normal(n)
    F = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], n) + 0.4 * z[:, None]
    items = [np.column_stack([l * F[:, k] + math.sqrt(p) * rng.standard_normal(n)
                              for l, p in zip(lam, psi)]) for k in (0, 1)]
    T = F * sum(lam)
    y = 1.0 * T[:, 0] + 0.5 * T[:, 1] + z + rng.normal(0, 2, n)
    return items, z, y


def test_4_by_simulation_joint_calibration_recovers_both_coefficients_where_separate_ones_do_not():
    """Freedman et al. 2011: with two error-prone exposures, estimates "may become attenuated,
    inflated, or can even change direction". Known truth (1.0 and 0.5 per unit of true score),
    40 datasets of 3,000: calibrated jointly, each coefficient averages within 2% of its truth;
    corrected each on its own (the other score read as exact), the second is more than 10% off."""
    rng = np.random.default_rng(2024)
    joint, separate = [], []
    for _ in range(40):
        (A, B), z, y = two_scales(rng, 3000)
        WA, WB = A.sum(axis=1), B.sum(axis=1)
        src = [S.Source("internal_consistency",
                        reliability=S.internal_consistency(M, "sum", "unidimensional", 1))
               for M in (A, B)]
        both = S.correct_jointly(np.column_stack([WA, WB]), z[:, None], y, [0, 1], src, n_boot=0)
        joint.append([c.estimate for c in both])
        separate.append([S.correct(WA, np.column_stack([WB, z]), y, 0, src[0], n_boot=0).estimate,
                         S.correct(WB, np.column_stack([WA, z]), y, 0, src[1], n_boot=0).estimate])
    joint_mean, separate_mean = np.mean(joint, axis=0), np.mean(separate, axis=0)
    print(f"\njoint {joint_mean.round(4)} | separate {separate_mean.round(4)} | truth [1.0, 0.5]")
    assert abs(joint_mean[0] / 1.0 - 1) < 0.02 and abs(joint_mean[1] / 0.5 - 1) < 0.02
    assert abs(separate_mean[1] / 0.5 - 1) > 0.10


# ── (5) the leash loosened where it was too tight ─────────────────────────────

HEI_SCALE = {"name": "hei_total", "items": HEI, "low": 0, "high": 10, "kind": "formative",
             "instrument": "HEI-2015", "reliability": "test_retest", "retest": HEI_RETEST,
             "correction": "regression_calibration", "n_boot": 50}
HEI_ROLES = {"pid": "identifier", "age": "covariate", **{c: "covariate" for c in HEI},
             **{c: "excluded" for c in HEI_RETEST}}


@needs_r
def test_5_a_diet_quality_index_of_continuous_component_scores_is_declared_and_corrected(tmp_path):
    """The verifier's S4b: the HEI-2015's components are prorated scores with decimals (0–5 or
    0–10), and a formative index of them was refused as "not whole", with no way forward. Now it
    is declared, and its disattenuation from a repeat recall runs: the score is pandas' sum of the
    thirteen components, its uncorrected coefficient statsmodels' (1e-10), its test–retest ICC
    psych::ICC's (1e-9), and its corrected coefficient mecor's MeasErrorRandom with the repeat's
    error variance var(W₁ − W₂)/2 (1e-6)."""
    frame = hei_table()
    assert not np.allclose(frame[HEI].to_numpy() % 1, 0)
    frame.to_csv(tmp_path / "hei.csv", index=False)
    with local_server(tmp_path / "srv") as client:
        drive = _drive(client, tmp_path / "hei.csv", HEI_ROLES, "sbp", "regression",
                       {"code_or_count:age": "amount"})
        drive.decide({"kind": "set_scales", "scales": [HEI_SCALE]})
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["linear"]})
        (hei,) = drive.artifact("scales")["scales"]
    W1 = frame[HEI].sum(axis=1)
    W2 = frame[HEI_RETEST].sum(axis=1, skipna=False)
    both = W2.notna()
    s2u = float(np.var((W1 - W2)[both], ddof=1) / 2)
    frame.assign(W1=W1, W2=W2)[["sbp", "W1", "W2", "age"]].to_csv(tmp_path / "rt.csv", index=False)
    ref = run_r(f"""
suppressMessages({{library(psych); library(mecor); library(jsonlite)}})
dd <- read.csv("rt.csv")
icc <- suppressWarnings(suppressMessages(ICC(dd[!is.na(dd$W2), c("W1", "W2")], lmer = FALSE)))
m <- mecor(sbp ~ MeasErrorRandom(substitute = W1, variance = {s2u!r}) + age, data = dd,
           method = "standard")
writeLines(toJSON(list(icc3 = icc$results["Single_fixed_raters", "ICC"],
                       mecor = unname(m$corfit$coef["cor_W1"])), digits = NA, auto_unbox = TRUE),
           "out.json")
""", tmp_path)
    naive = sm.OLS(frame["sbp"], sm.add_constant(np.column_stack([W1, frame["age"]]))).fit()
    assert hei["correction"]["naive"] == pytest.approx(naive.params.iloc[1], rel=1e-10)
    assert hei["reliability"]["value"] == pytest.approx(ref["icc3"], abs=1e-9)
    assert hei["correction"]["error_variance"] == pytest.approx(s2u, rel=1e-10)
    assert hei["correction"]["estimate"] == pytest.approx(ref["mecor"], rel=1e-6)
    assert hei["methods"].startswith(
        "`hei_total` was the sum of the HEI-2015's 13 components. Reliability was estimated as the "
        f"test–retest ICC from a repeat administration (ICC(3,1) = {ref['icc3']:.2f}, "
        f"{int(both.sum()):,} participants with both); the corrected coefficient was obtained by "
        "regression calibration including all model covariates")


def test_5_decimals_in_a_reflective_scale_ask_whether_it_is_an_index_and_a_blank_item_is_named(
        tmp_path):
    """A reflective scale's answers are whole numbers on its response scale, so decimals there are
    still refused, now with a way forward that works: declaring it a formative index of continuous
    component scores (its ω-based correction then off, ruling 8). An item blank on every row is
    said to be blank (the verifier's S6b: it was said to hold text), with the exit that scores the
    scale from its other items."""
    frame = hei_table(n=300)
    roles = {"pid": "identifier", "age": "covariate", **{c: "covariate" for c in HEI}}
    reflective = {**HEI_SCALE, "kind": "reflective", "reliability": "internal_consistency",
                  "retest": []}
    with _store(frame, tmp_path, "hei") as store:
        ctx = {"state": d.ProjectState(purpose="inference", target="sbp", roles=roles),
               "columns": list(frame.columns), "store": store}
        error = _refused({"kind": "set_scales", "scales": [reflective]}, ctx)
        assert error.code == "items_not_whole"
        index = error.exits[0]
        assert index["label"] == "`hei_total` is a formative index of continuous component scores"
        assert (index["decision"]["scales"][0]["kind"],
                index["decision"]["scales"][0]["correction"]) == ("formative", "none")
        d.validate(index["decision"], ctx)
    blank = linear_scale_table(n=300).assign(sat_8=np.nan)
    with _store(blank, tmp_path, "blank") as store:
        ctx = _lin_ctx(blank, store)
        error = _refused({"kind": "set_scales", "scales": [SAT_SCALE]}, ctx)
        assert error.code == "items_blank"
        assert error.message == "`sat_8` has no answer on any row, so it cannot be scored into " \
                                "`sat_score`."
        assert error.exits[0]["label"] == "Score `sat_score` from its other 7 items"
        d.validate(error.exits[0]["decision"], ctx)


def test_5_the_contract_names_the_relations_this_repair_enforces():
    """BLUEPRINT §13: each relation the repair adds is declared on the scales contract and names
    the live function that enforces it."""
    import importlib

    from turbotab.core.scales import CONTRACT

    declared = {(r.kind, r.target): r.enforced_by for r in CONTRACT.relations}
    for key, target in {
        ("invalidates", "the scales answer when an item leaves the predictors"):
            "turbotab.core.scales:_answers_keep_the_scales_whole",
        ("implies", "the reference measure read as an amount"):
            "turbotab.core.scales:_reference_reads_as_an_amount",
        ("implies", "multivariate calibration when several scores are corrected"):
            "turbotab.core.methods.scales:calibrate_jointly",
        ("implies", "cluster-aware intervals on grouped rows"):
            "turbotab.core.methods.scales:bootstrap_draws",
    }.items():
        assert declared[key] == target
        module, name = target.split(":")
        assert callable(getattr(importlib.import_module(module), name))
