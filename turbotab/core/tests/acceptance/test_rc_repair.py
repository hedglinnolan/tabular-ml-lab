"""REPAIR-RC · the independent verifier's findings on MS5 (regression calibration), each closed in
its layer with a permanent test (docs/turbotab-next/MODELING_SEQUENCE.md §0 ruling 7, §2, §4).

1. **The uncorrected estimate beside it is the primary's** (the open item, MS5 test 5's claim):
   under multiple imputation, 30 of 400 participants have energy blank on every recall day. The
   primary imputes them; they have no replicate, so they are left out of the calibration. The
   uncorrected estimate, interval and test beside the calibrated one are the primary's on all 400
   (NumPy's least squares on each copy, and the fit stage's pooled table), the refit on the 370
   is given beside them, and a concern and the methods text say so.
2. **The cluster floor** (§2, repeated units "with refusal below a floor"): four households refuse
   the calibration as the fit refuses their intervals, with an exit.
3. **A lonely PSU** (a stratum with a single PSU): the bootstrap draws it twice or not at all, so
   its variance is the one R survey's ``lonely.psu = "adjust"`` gives the primary (R's ``svyglm``
   to 6%), and the calibrated interval covers where keeping it whole did not (simulation truth).
4. **Block and record, the record half**: a declared calibration the stage blocked is said to be
   blocked in the record's sentence in force and in the exported methods (HTTP), restoring MS4's
   verified sentence (also restored in MS4's own test 4); one whose adjustment set changed is said
   to be re-asked, and the export refuses until it is declared again (HTTP).
5. **The declared secondary reaches the export**: its table (the primary's uncorrected estimate,
   the calibrated one, the refit on the same people), its methods paragraph and its label; through
   the server, and ``python -m turbotab.replay`` reproduces its numbers in a fresh home.
6. **Every refusal of the data's carries an exit** ("Record no calibration").
7. **The preview's λ is the stage's** under the surveyed population (weighted, on the domain); under
   multiple imputation with blanks it is not drawn.

The verifier's note on resamples dropped for a true-intake covariance that is not positive (kept,
disclosed) is now said with what dropping does: those are the resamples where the correction is
largest, so the interval's end away from zero may be too near (MS5's test 4, word for word).

**References, each independent of the code under test:** NumPy by hand from the CSV's recall days
(the closed form of ``test_ms5_regression_calibration.by_hand``; least squares on each imputed
copy); WP12c's NumPy and statsmodels reference from the CSV (``test_wp12c_calibration.reference``);
R ``survey`` 4.5's ``svyglm`` with ``options(survey.lonely.psu = "adjust")`` (skipped without R) and
its sandwich by hand in NumPy; simulations with a known truth (slope 0.5).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.decisions import ProjectState, SetMeasurementError, validate
from turbotab.core.methods import calibration as RC
from turbotab.core.tests.acceptance.rc_fixtures import chain2_table, needs_r, run_r
from turbotab.core.tests.acceptance.test_ms5_regression_calibration import (CHAIN_ROLES, LABEL,
                                                                            by_hand, chain_state,
                                                                            ctx_of, declared,
                                                                            kcal_days)
from turbotab.core.tests.graph_runner import GraphRun

NO_CALIBRATION = {"label": "Record no calibration",
                  "decision": {"kind": "set_measurement_error", "method": "none"}}
HEAD = ("Regression calibration of every intake the recalls measure from the repeated recalls was "
        "declared as a secondary analysis beside the uncorrected estimate")


def rel(a: Any, b: Any) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


# ── (1) the uncorrected estimate beside it is the primary's ──────────────────


@pytest.fixture(scope="module")
def no_recall_day(tmp_path_factory) -> dict:
    """Chain 2's table (400 people, two recalls each, BMI blank for some) with energy blank on
    every recall day of 30 people, the sample answer, multiple imputation (m at least 20); the
    graph runs once with a spy on ``methods.calibration.correct`` that keeps each call's input."""
    folder = tmp_path_factory.mktemp("rc_no_recall_day")
    frame, _ = chain2_table(seed=2, per_psu=25)
    people = np.sort(frame["seqn"].unique())
    gone = np.sort(np.random.default_rng(11).choice(people, 30, replace=False))
    frame.loc[frame["seqn"].isin(gone), "energy_kcal"] = np.nan
    path = folder / "chain.csv"
    frame.to_csv(path, index=False)
    run = GraphRun(path, folder / "project")
    state = chain_state(missing={"strategy": "multiple_imputation", "m": 20})
    out = run.run(state, upto=["findings"])
    state = declared(state, frame, out, n_boot=50)
    calls: list[dict] = []
    real = RC.correct

    def spy(X, columns, rec, y, *, fit=RC.ols_fit, weights=None):
        calls.append({"X": np.array(X, copy=True), "columns": list(columns), "y": np.array(y)})
        return real(X, columns, rec, y, fit=fit, weights=weights)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RC, "correct", spy)
        out = run.public(run.run(state, upto=["calibration", "fit"]))
    yield {"frame": frame, "gone": gone, "people": people, "state": state, "out": out,
           "calls": calls}
    run.close()


def _by_hand_on_copies(case: dict) -> dict[str, np.ndarray]:
    """Over the analysis's own imputed copies (the stage's first m calls): the calibrated and the
    uncorrected coefficients by hand on the 370 people with a recall day of every source (the
    closed form from the CSV's recall days), and the uncorrected ones on all 400 (NumPy's least
    squares on the copy's matrix): each averaged over the copies."""
    cal = case["out"]["calibration"]
    m = cal["imputations"]
    first = case["calls"][:m]
    frame, gone, people = case["frame"], case["gone"], case["people"]
    has = ~np.isin(people, gone)
    D, names, kept = kcal_days(frame[~frame["seqn"].isin(gone)])
    assert names == cal["calibrated"]
    y = kept["sbp"].to_numpy(dtype=float)
    calibrated, naive, primary = [], [], []
    for call in first:
        X, J = call["X"], call["columns"]
        others = [c for c in range(X.shape[1]) if c not in J]
        hand = by_hand(D, X[has][:, others], y)
        calibrated.append(hand["calibrated"])
        naive.append(hand["naive"])
        full = np.column_stack([np.ones(len(X)), X])
        coef = np.linalg.lstsq(full, call["y"], rcond=None)[0][1:]
        primary.append(coef[J])
    return {"calibrated": np.mean(calibrated, axis=0), "naive": np.mean(naive, axis=0),
            "primary": np.mean(primary, axis=0), "names": names}


def test_r1_the_uncorrected_estimate_beside_the_calibration_is_the_primarys_on_every_participant(
        no_recall_day):
    """The verifier's open item: the stage calibrated 370 people silently and its "uncorrected"
    row was a 370-person refit labeled the primary's. Now, by hand on the analysis's own
    imputed copies:

    * the calibrated coefficients are the closed form on the 370 people with a recall day of every
      source, averaged over the copies, to 1e-8;
    * the uncorrected estimate beside each is the primary's on all 400: NumPy's least squares on
      each copy averaged (1e-8), and the fit stage's pooled estimate, interval and test (1e-9);
    * the refit on the same 370 (``naive_refit``) is the closed form's uncorrected coefficient
      (1e-8), and differs from the primary's;
    * the artifact counts both (370 calibrated, 400 analyzed), the concern says so word for word,
      and so does the methods text."""
    cal, fit = no_recall_day["out"]["calibration"], no_recall_day["out"]["fit"]
    assert cal["applies"] and cal["imputations"] >= 20
    assert (cal["n_persons"], cal["n_primary"]) == (370, 400)
    hand = _by_hand_on_copies(no_recall_day)
    table = {c["feature"]: c for c in fit["models"][0]["coefficients"]}
    moved = 0.0
    for e in cal["exposures"]:
        j = hand["names"].index(e["feature"])
        primary = table[e["feature"]]
        assert e["estimate"] == pytest.approx(hand["calibrated"][j], rel=1e-8)
        assert e["naive_refit"] == pytest.approx(hand["naive"][j], rel=1e-8)
        assert e["naive"] == pytest.approx(hand["primary"][j], rel=1e-8)
        assert (e["naive"], e["naive_ci_low"], e["naive_ci_high"], e["p"]) == pytest.approx(
            (primary["estimate"], primary["ci_low"], primary["ci_high"], primary["p"]), rel=1e-9)
        assert e["n_persons"] == 370
        moved = max(moved, abs(e["naive"] - e["naive_refit"]) / abs(e["naive"]))
    assert moved > 1e-3  # the refit on 370 is not the primary's on 400
    (swap,) = cal["contrasts"]
    rows = {e["feature"]: e for e in cal["exposures"]}
    assert swap["naive"] == pytest.approx(
        100 * (rows["kcal_from_carbohydrate_g"]["naive"] - rows["kcal_from_fat_g"]["naive"]),
        rel=1e-12)
    assert ("30 of the 400 participants the primary analyzes have no recall day with every "
            "calibrated intake recorded (the primary fills their intakes in each imputed copy), so "
            "they are not in the calibration: the calibrated estimate describes the other 370, "
            "while the uncorrected estimate beside it, with its interval and test, is the "
            "primary's on all 400. Refit on the same 370, the uncorrected coefficient is given "
            "beside it.") in cal["concerns"]
    assert (" Of the 400 participants the primary analyzes, 30 had no recall day with every "
            "calibrated intake recorded and were left out of the calibration; the uncorrected "
            "estimate beside it is the primary's, on all 400.") in cal["methods"]
    assert cal["test"].startswith("The test of no association is the uncorrected model's")


# ── (2) the cluster floor ────────────────────────────────────────────────────


def test_r2_below_the_cluster_floor_the_calibration_is_refused_as_the_fits_intervals_are(tmp_path):
    """MODELING_SEQUENCE §2: repeated units imply cluster-aware intervals "with refusal below a
    floor". Chain 2's people in four households (a grouping the user confirmed): the fit reports
    no interval for them ("fewer than the 8 TurboTab requires"), and the calibration, whose
    interval would come from a bootstrap redrawing four households, is refused with the reason,
    an exit that is a decision the validators accept, and its block in the methods' words. At the
    floor itself (8 clusters) it is not refused; at 7 it is."""
    from turbotab.core.models.inference import Clusters
    from turbotab.core.stages.calibration import cluster_floor

    frame, _ = chain2_table(seed=6, per_psu=20, missing_share=0.0, household=True)
    frame["household"] = 50_000 + frame["seqn"] % 4
    path = tmp_path / "households.csv"
    frame.to_csv(path, index=False)
    run = GraphRun(path, tmp_path / "project")
    try:
        state = chain_state(roles={**CHAIN_ROLES, "household": "cluster"},
                            reading_confirmations={"cluster:household": "yes"})
        out = run.run(state, upto=["findings"])
        state = declared(state, frame, out, n_boot=50)
        out = run.public(run.run(state, upto=["calibration", "fit"]))
    finally:
        run.close()
    cal, fit = out["calibration"], out["fit"]
    model = fit["models"][0]
    assert all(c.get("ci_low") is None for c in model["coefficients"])
    assert [s for s in _strings(model) if "fewer than the 8 TurboTab requires" in s]
    assert cal["applies"] is False and cal["exposures"] == []
    assert cal["reason"] == (
        "The analyzed participants lie in 4 `household` units, fewer than the 8 TurboTab requires "
        "for cluster-robust intervals (the fit reports none for the uncorrected estimate either). "
        "A calibrated coefficient's interval would come from a bootstrap redrawing 4 whole "
        "clusters, which rests on a handful of units, so regression calibration is refused here.")
    assert cal["exits"] == [NO_CALIBRATION]
    validate(cal["exits"][0]["decision"], ctx_of(state, frame, out))
    assert cal["blocked"] == ("the participants lay in 4 `household` units, fewer than the 8 that "
                              "cluster-robust intervals require, too few for a bootstrap of whole "
                              "clusters to carry its interval")
    assert cal["methods"] == (
        "Regression calibration was declared as a secondary analysis, but the participants lay in "
        "4 `household` units, fewer than the 8 that cluster-robust intervals require, too few for "
        "a bootstrap of whole clusters to carry its interval; it was blocked and recorded, and the "
        "estimates are uncorrected.")
    for G, refused in ((7, True), (8, False)):
        codes = np.arange(80) % G
        found = cluster_floor(Clusters(column="site", codes=codes, n_clusters=G), np.arange(80))
        assert (found is not None) == refused


# ── (3) a lonely PSU ─────────────────────────────────────────────────────────


def _lonely_design(seed: int = 31):
    """Six strata: three of two PSUs (25 people), one of three (20), two of a single PSU (40 each);
    weights by PSU, and PSU effects in both the covariate and the outcome."""
    rng = np.random.default_rng(seed)
    sizes = [25, 25] * 3 + [20, 20, 20] + [40, 40]
    stratum_of = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 5])
    psu = np.repeat(np.arange(len(sizes)), sizes)
    n = len(psu)
    w = np.exp(rng.normal(0, 0.4, len(sizes)))[psu] * rng.uniform(0.8, 1.2, n)
    x = rng.normal(0, 1, n) + rng.normal(0, 0.8, len(sizes))[psu]
    y = 1 + 0.5 * x + rng.normal(0, 1, n) + rng.normal(0, 1.0, len(sizes))[psu]
    return stratum_of, psu, w, x, y


def _scores(w, x, y):
    """The weighted least-squares slope's bread and each person's score w x̃ e, by hand."""
    D = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(D * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    U = D * (w * (y - D @ beta))[:, None]
    return np.linalg.inv(D.T @ (D * w[:, None])), U, beta


def _replicate_variance(stratum_of, psu, U, B, draws: int = 4000, seed: int = 5) -> float:
    """The variance over ``draws`` of ``psu_draw``'s replicates of the slope's linearization,
    B Σ_i (m_i − 1) u_i, m_i each person's weight multiplier in the replicate (how many times its
    PSU was drawn, times the stratum's rescaling)."""
    rng = np.random.default_rng(seed)
    stratum = stratum_of[psu]
    out = []
    for _ in range(draws):
        dr = RC.psu_draw(stratum, psu, rng)
        mult = np.bincount(dr.rows, weights=dr.factor, minlength=len(psu))
        out.append((B @ (U * (mult - 1)[:, None]).sum(axis=0))[1])
    return float(np.var(out, ddof=1))


def test_r3_a_lonely_psu_adds_its_score_total_squared_as_the_primarys_rule_does():
    """The verifier: single-PSU strata were kept whole in every replicate, adding no variance,
    while the primary centers them by R survey's lonely.psu "adjust" (``models.survey``). Now the
    bootstrap draws a lonely PSU twice or not at all: the variance of its replicates of a weighted
    slope's linearization is the "adjust" sandwich by hand, B [Σ_h n_h/(n_h − 1) Σ_j (z_hj − z̄_h)²
    + Σ_lonely (z_h − z̄)²] B, z̄ the grand mean of the PSU score totals (zero: the scores sum to
    zero), to 6% (Monte Carlo error about 2%); the lonely strata's part is about 8% of it, and
    the old rule (kept whole) would have left it out."""
    stratum_of, psu, w, x, y = _lonely_design()
    B, U, _ = _scores(w, x, y)
    G = len(stratum_of)
    Z = np.column_stack([np.bincount(psu, weights=U[:, j], minlength=G) for j in range(2)])
    grand = Z.sum(axis=0) / G
    meat, lonely = np.zeros((2, 2)), np.zeros((2, 2))
    for h in np.unique(stratum_of):
        zs = Z[stratum_of == h]
        if len(zs) > 1:
            dz = zs - zs.mean(axis=0)
            meat += len(zs) / (len(zs) - 1) * dz.T @ dz
        else:
            dz = zs - grand
            lonely += dz.T @ dz
    adjust = float((B @ (meat + lonely) @ B)[1, 1])
    assert float((B @ lonely @ B)[1, 1]) / adjust > 0.05
    assert _replicate_variance(stratum_of, psu, U, B) == pytest.approx(adjust, rel=0.06)


@needs_r
def test_r3_the_lonely_rule_is_r_survey_adjust(tmp_path):
    """The same replicate variance against R survey 4.5's ``svyglm`` with
    ``options(survey.lonely.psu = "adjust")`` on the CSV, to 6%."""
    stratum_of, psu, w, x, y = _lonely_design()
    pd.DataFrame({"y": y, "x": x, "w": w, "psu": psu, "stratum": stratum_of[psu]}).to_csv(
        tmp_path / "design.csv", index=False)
    ref = run_r("""
suppressMessages({library(survey); library(jsonlite)})
options(survey.lonely.psu = "adjust")
d <- read.csv("design.csv")
des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, data = d)
m <- svyglm(y ~ x, design = des)
writeLines(toJSON(list(v = unname(vcov(m)["x", "x"]), b = unname(coef(m)["x"])),
                  digits = NA, auto_unbox = TRUE), "out.json")
""", tmp_path)
    B, U, beta = _scores(w, x, y)
    assert beta[1] == pytest.approx(ref["b"], rel=1e-10)
    assert _replicate_variance(stratum_of, psu, U, B) == pytest.approx(ref["v"], rel=0.06)


def test_r3_by_simulation_the_calibrated_interval_covers_with_lonely_strata():
    """Known truth: slope 0.5 per unit of true intake. Seven of eight strata hold a single PSU (45
    people each; the eighth two), PSU effects in the intake and the outcome, two recalls each,
    PSU weights. Over 200 datasets, the weighted calibration (``correct``) with its whole-chain
    bootstrap by PSU within strata (100 resamples) covers 0.5 in at least 93% (observed about
    0.98: the "adjust" rule is conservative); kept whole, as before the repair, the lonely strata
    added no variance and it covered about a quarter of the time."""
    def kept_whole(stratum, psu, rng):
        rows, factor = [], []
        for h in np.unique(stratum):
            units = np.unique(psu[stratum == h])
            n_h = len(units)
            drawn = rng.choice(units, n_h - 1) if n_h > 1 else units
            for u in drawn:
                members = np.flatnonzero(psu == u)
                rows.append(members)
                factor.append(np.full(len(members), n_h / (n_h - 1) if n_h > 1 else 1.0))
        rows_, factor_ = np.concatenate(rows), np.concatenate(factor)
        return RC.Draw(rows_, factor_, np.zeros(len(rows_), dtype=np.int64),
                       np.zeros(len(rows_), dtype=np.int64))

    reps, covered, before = 200, 0, 0
    for r in range(reps):
        rng = np.random.default_rng(500 + r)
        stratum_of = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7])
        psu = np.repeat(np.arange(9), 45)
        stratum = stratum_of[psu]
        n = len(psu)
        w = np.exp(rng.normal(0, 0.3, 9))[psu]
        age = rng.normal(50, 10, n)
        x = 10 + 0.05 * age + rng.normal(0, 1.5, 9)[psu] + rng.normal(0, 2, n)
        person = np.repeat(np.arange(n), 2)
        W = x[person] + rng.normal(0, 2.8, person.size)
        y = 2 + 0.5 * x + 0.03 * age + rng.normal(0, 1.5, 9)[psu] + rng.normal(0, 2, n)
        rec = RC.Recalls.of(W, person, n)
        X = np.column_stack([rec.means()[:, 0], age])

        def estimate(draw, b):
            idx = draw.rows
            got = RC.correct(X[idx], [0], rec.take(idx), y[idx], weights=w[idx] * draw.factor)
            return np.array([got.estimate[0]])

        res = RC.Resampling("psu_within_strata", n, stratum=stratum, psu=psu,
                            design_psus={int(u): int(s) for u, s in enumerate(stratum_of)})
        low, high = RC.whole_chain(estimate, res, 100, seed=r).interval()
        covered += low[0] <= 0.5 <= high[0]

        class Old(RC.Resampling):
            def draw(self, rng):  # type: ignore[override]
                return kept_whole(stratum, psu, rng)

        low, high = RC.whole_chain(estimate, Old("persons", n), 100, seed=r).interval()
        before += low[0] <= 0.5 <= high[0]
    print(f"\ncoverage with the lonely rule {covered / reps:.3f} | kept whole {before / reps:.3f}")
    assert covered / reps >= 0.93
    assert before / reps < 0.6


# ── the surveyed population on WP12c's recalls, some strata lonely ───────────


def _population_frame(lonely: bool) -> pd.DataFrame:
    """WP12c's two recalls of protein and energy for 400 people, with a stratified design (six
    strata, two PSUs each, weights by person; strata 5 and 6 a single PSU when ``lonely``)."""
    from turbotab.core.tests.acceptance.test_wp12c_calibration import recall_table

    frame = recall_table(n=400)
    person = frame["participant_id"].str[1:].astype(int).to_numpy()
    weight = np.round(np.random.default_rng(3).uniform(2000, 40000, person.max() + 1), 1)
    frame["SDMVSTRA"], frame["SDMVPSU"] = 1 + person % 6, 1 + (person // 6) % 2
    if lonely:
        frame.loc[frame["SDMVSTRA"] >= 5, "SDMVPSU"] = 1
    frame["WTDRD1"] = weight[person]
    return frame


def _population_state(**slots: Any) -> ProjectState:
    from turbotab.core.tests.acceptance.test_wp12c_calibration import ROLES

    roles = {**ROLES, "WTDRD1": "design", "SDMVSTRA": "design", "SDMVPSU": "design"}
    base: dict[str, Any] = dict(
        lens=["dietary"], target="ldl", task="regression", purpose="inference",
        roles=roles, role_confirmations=dict(roles),
        grain=d.GrainSpec(grain="repeated", id_column="participant_id"),
        repeat_kind=d.RepeatSpec(repeat_kind="repeats"), unit="unit",
        aggregation=d.AggregationSpec(method="mean"), temporal=d.TemporalSpec(temporal=False),
        shape_confirmations={"code_or_count:age": "amount", "code_or_count:SDMVSTRA": "code",
                             "code_or_count:SDMVPSU": "code"},
        column_units={"energy_kcal": d.ColumnUnitSpec(unit="kcal", days=1)},
        exclusions=[], missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"],
        energy_adjustment=d.EnergyAdjustment(method="residual", energy_column="energy_kcal",
                                             nutrients=["protein_g"]),
        survey=d.SurveySpec(estimand="population", weight="WTDRD1", strata="SDMVSTRA",
                            psu="SDMVPSU"))
    base.update(slots)
    return ProjectState(**base)


CALIBRATE = SetMeasurementError(method="regression_calibration", n_boot=50)


@pytest.fixture(scope="module")
def lonely_population(tmp_path_factory):
    """The preview harness's project (the graph run upto the design) on the lonely design under
    the population answer; the calibration's preview, then the graph with it recorded."""
    from turbotab.core.tests.acceptance.preview_harness import Project

    project = Project(_population_frame(lonely=True), tmp_path_factory.mktemp("rc_lonely"),
                      _population_state(), upto=["design", "cohort", "split"])
    result, ctx = project.preview(CALIBRATE)
    state, out = project.after(CALIBRATE, upto=["calibration", "fit"])
    yield {"project": project, "preview": result, "ctx": ctx, "state": state,
           "out": {k: getattr(v, "data", v) for k, v in out.items()}}
    project.close()


def test_r3_through_the_stage_lonely_strata_are_resampled_and_said_as_the_primary_says_them(
        lonely_population):
    """Under the surveyed population with two of six strata holding a single PSU, the calibration
    runs design-based (weighted, PSUs resampled within strata), and its concern names the lonely
    strata and the rule word for word, as the primary's design-based table (the fit stage's) names
    them: two rules that agree, where before the calibrated interval kept them whole."""
    cal, fit = lonely_population["out"]["calibration"], lonely_population["out"]["fit"]
    assert cal["applies"] and (cal["resampling"], cal["weighted"]) == ("psu_within_strata", True)
    assert ("2 strata have a single PSU, each PSU drawn twice or not at all, each with chance 1/2, "
            "so its PSU total varies about zero as R survey's lonely.psu \"adjust\" centers it in "
            "the primary's table: this overstates rather than understates the variance."
            ) in cal["concerns"]
    assert [s for s in _strings(fit["models"][0])
            if s.startswith("2 strata have a single PSU") and "lonely.psu \"adjust\"" in s]


def _strings(obj: Any) -> list[str]:
    """Every string inside an artifact (its concerns and notes, wherever they sit)."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _strings(v)]
    return []


# ── (7) the preview's λ is the stage's ───────────────────────────────────────


def test_r7_under_the_surveyed_population_the_previews_lambda_is_the_stages(lonely_population):
    """Wave 2c's invariant ("headline numbers equal to its stage's once recorded"): the
    ``set_measurement_error`` preview's λ was computed unweighted on every row, while the stage
    calibrates the survey domain survey-weighted. It now reads the design as the stage does, so
    λ is the stage's to 1e-12. The weights matter here: the same preview under the sample answer
    (unweighted, what the preview used to show) differs from it in the third decimal."""
    ctx = lonely_population["ctx"]
    (exposure,) = lonely_population["out"]["calibration"]["exposures"]
    assert ctx.read["calibration"] == {"protein_g_adj": pytest.approx(exposure["attenuation"],
                                                                      rel=1e-12)}
    project = lonely_population["project"]
    sample = _population_state(survey=d.SurveySpec(estimand="sample"))
    _, unweighted = project.preview(CALIBRATE, ctx=project.context(state=sample))
    assert abs(unweighted.read["calibration"]["protein_g_adj"] - exposure["attenuation"]) > 1e-3


def test_r7_under_multiple_imputation_with_blanks_no_lambda_is_drawn(tmp_path):
    """With BMI blank for some people under multiple imputation, the stage calibrates inside each
    imputed copy; the preview, which fills nothing, draws no λ and says why."""
    from turbotab.core.method_previews import IMPUTED_LAMBDA
    from turbotab.core.tests.acceptance.preview_harness import Project

    frame, _ = chain2_table(seed=2, per_psu=12)
    project = Project(frame, tmp_path, chain_state(missing={"strategy": "multiple_imputation",
                                                            "m": 20}),
                      upto=["design", "cohort", "split"])
    try:
        result, ctx = project.preview(CALIBRATE)
    finally:
        project.close()
    assert "calibration" not in ctx.read
    assert ctx.read["note"] == IMPUTED_LAMBDA and result.note == IMPUTED_LAMBDA


# ── (4) block and record: the record half ────────────────────────────────────


def test_r4_a_blocked_calibration_is_said_blocked_in_the_record_in_force(tmp_path):
    """MS4's verified block-and-record sentence, reopened by MS5: through the server, with one PSU
    in every stratum under the surveyed population, the calibration stage blocks (no bootstrap by
    PSU within strata), and the Record's line in force for ``set_measurement_error`` now says it
    was blocked and recorded, and the estimates are uncorrected, word for word. The plan locked
    (the primary's estimates shown), the export is served with that sentence in its methods and no
    calibrated table. The no-correction exit, taken after the estimates were seen, says the intakes
    were not corrected, with that lead."""
    from turbotab.core.stages.calibration import POPULATION as CALIBRATION_BLOCK
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project
    from turbotab.core.tests.acceptance.test_wp12c_calibration import (RESIDUAL, ROLES,
                                                                       recall_truth)

    frame = _population_frame(lonely=False).assign(SDMVPSU=1)
    path = tmp_path / "lonely.csv"
    frame.to_csv(path, index=False)
    truth = recall_truth()
    truth.update({"code_or_count:SDMVSTRA": "code", "code_or_count:SDMVPSU": "code"})
    with local_server(tmp_path / "home") as client:
        def force(dr) -> str:
            lines = client.get(f"/api/projects/{dr.pid}/methods").json()["lines"]
            (line,) = [x for x in lines if x["kind"] == "set_measurement_error" and x["in_force"]]
            return line["sentence"]

        dr = open_project(client, path, truth)
        dr.decide({"kind": "set_lens", "lenses": ["dietary"]})
        dr.reach("target")
        dr.decide({"kind": "set_target", "column": "ldl"})
        dr.answer("task", {"kind": "set_task", "column": "ldl", "task": "regression"})
        dr.reach("purpose")
        dr.decide({"kind": "set_purpose", "purpose": "inference"})
        dr.reach("grain")
        dr.decide({"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
        dr.answer("repeat_kind", {"kind": "set_repeat_kind", "repeat_kind": "repeats"})
        dr.answer("unit", {"kind": "set_unit", "unit": "unit"})
        dr.answer("aggregation", {"kind": "set_aggregation", "method": "mean"})
        dr.answer("temporal", {"kind": "set_temporal", "temporal": False})
        dr.reach("roles")
        dr.decide_roles({**ROLES, "WTDRD1": "design", "SDMVSTRA": "design", "SDMVPSU": "design"})
        dr.reach("survey")
        population = dr.artifact("proposals")["survey"]["options"][0]["decision"]
        assert population["estimand"] == "population"
        dr.decide(population)
        dr.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        dr.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        dr.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        dr.answer("energy_adjustment", RESIDUAL)
        dr.reach("models")
        dr.decide({"kind": "select_models", "models": ["linear"]})
        dr.decide({"kind": "set_measurement_error", "method": "regression_calibration",
                   "n_boot": 50})
        blocked = dr.artifact("calibration")
        said_blocked = force(dr)
        fit = dr.artifact("fit")  # the primary's estimates shown: the plan locks
        _wait_fresh(dr, ("cohort", "design", "fit", "effects", "calibration"))
        exported = client.get(f"/api/projects/{dr.pid}/export")
        dr.decide(blocked["exits"][1]["decision"])  # no correction
        dr.artifact("calibration")
        said_none = force(dr)
    assert (blocked["applies"], blocked["reason"]) == (False, CALIBRATION_BLOCK)
    assert blocked["blocked"] == ("under the surveyed population no stratum held two PSUs with "
                                  "analyzed participants, leaving no bootstrap by PSU within "
                                  "strata for its interval")
    assert said_blocked == (
        f"{HEAD}, but under the surveyed population no stratum held two PSUs with analyzed "
        "participants, leaving no bootstrap by PSU within strata for its interval; it was blocked "
        "and recorded, and the estimates are uncorrected.")
    # The export is served with the block said: the record's sentence in its methods, and no
    # calibrated table, since none was run.
    assert any(c.get("estimate") is not None for c in fit["models"][0]["coefficients"])
    assert exported.status_code == 200, exported.text[:900]
    files = _files(exported.content)
    assert said_blocked in files["methods.md"].decode()
    assert "results/calibration.csv" not in files
    assert said_none == ("After the estimates were seen, intakes were not corrected for day-to-day "
                         "error in the recalls.")


def _log(tmp_path, decisions_: list[Any]) -> Any:
    from turbotab.core import voice
    from turbotab.core.decisions import DecisionLog

    log = DecisionLog(tmp_path / "decisions.jsonl")
    for decision in decisions_:
        log.append(decision, sentence=lambda dd, st: voice.sentence_for(dd, st, None))
    return log


def test_r4_a_changed_adjustment_set_is_said_re_asked_and_the_export_waits_for_it():
    """MODELING_SEQUENCE §2: "a change to the adjustment set invalidates it". The answer records
    the set it was declared under (age, BMI, energy and sex); BMI then leaves the model. The
    record's sentence, restated on the answers as they stand (``voice.restate``, which the methods
    text and the export's read), says it is re-asked and was not run, word for word, and the
    export refuses with the stage's own exits (declare it again, or record none), each a decision
    the validators accept; declared again, the export no longer waits for it."""
    from turbotab.core import voice
    from turbotab.core.export import gate

    answer = SetMeasurementError(method="regression_calibration", n_boot=50,
                                 adjustment=["age", "bmi", "energy_kcal", "sex"])
    then = chain_state()
    said = voice.sentence_for(answer, then)
    assert said == (f"{HEAD}, with intervals from `50` bootstrap resamples of the whole chain.")
    moved = chain_state(roles={**CHAIN_ROLES, "bmi": "excluded"})
    assert voice.restate(answer, said, then, moved) == (
        f"{HEAD} when the model adjusted for `age`, `bmi`, `energy_kcal` and `sex`; the "
        "adjustment set has changed since, so the declaration is re-asked and was not run, and the "
        "estimates are uncorrected.")
    state = moved.model_copy(update={"measurement_error": d.MeasurementErrorSpec(
        **answer.model_dump(exclude={"kind"}))})
    source = type("S", (), {"inputs": [], "interview": [], "state": state, "purpose": "inference",
                            "artifact": staticmethod(lambda stage: None),
                            "status": staticmethod(lambda stage: None)})()
    assert "calibration" in gate.result_stages(state)
    (reasked,) = [m for m in gate.missing(source) if m.message.startswith("Regression calibration")]
    assert reasked.code == "unanswered_questions"
    assert [e["label"] for e in reasked.exits] == [
        "Declare the calibration again under the current adjustment set", "Record no calibration"]
    for e in reasked.exits:
        validate(e["decision"], {"state": state})
    again = state.model_copy(update={"measurement_error": state.measurement_error.model_copy(
        update={"adjustment": ["age", "energy_kcal", "sex"]})})
    source.state = again
    assert not [m for m in gate.missing(source) if m.message.startswith("Regression calibration")]


# ── (5) the declared secondary reaches the export ────────────────────────────


def test_r5_the_calibration_its_label_and_its_paragraph_reach_the_export(no_recall_day, tmp_path):
    """The verifier: the bundle had no calibrated estimate, no label and no chain-2 sentence.
    Now, from the stage's artifact as served (the MI chain with 30 people left out of the
    calibration):

    * the export waits for the calibration when one is declared (it is a result stage);
    * ``results/calibration.csv`` holds each intake's uncorrected estimate, interval and test (the
      primary's: NumPy's least squares on each copy, and the fit stage's pooled table), the
      calibrated estimate (the closed form by hand) with its bootstrap interval, the refit on the
      same 370 (by hand), and the substitution; its caption carries the label and both counts;
    * the methods carry the calibration's own paragraph (chain 2's reviewers' sentence first);
    * a blocked calibration adds no table: the record's sentence says it was blocked (the
      restatement, through the methods text the export reads)."""
    from turbotab.core.export import gate, methods, tables
    from turbotab.core.provenance import methods_text
    from turbotab.core.stages.calibration import record_facts

    cal, fit = no_recall_day["out"]["calibration"], no_recall_day["out"]["fit"]
    assert gate.result_stages(no_recall_day["state"]) == (
        "cohort", "design", "fit", "effects", "calibration")
    hand = _by_hand_on_copies(no_recall_day)
    (table,) = tables.calibration_table(cal)
    assert table.name == "calibration"
    pooled = {c["feature"]: c for c in fit["models"][0]["coefficients"]}
    rows = {r["key"]: r for r in table.rows}
    for e in cal["exposures"]:
        j = hand["names"].index(e["feature"])
        row = rows[e["feature"]]
        assert row["uncorrected"] == pytest.approx(hand["primary"][j], rel=1e-8)
        assert (row["uncorrected_low"], row["uncorrected_high"], row["p"]) == pytest.approx(
            tuple(pooled[e["feature"]][k] for k in ("ci_low", "ci_high", "p")), rel=1e-9)
        assert row["calibrated"] == pytest.approx(hand["calibrated"][j], rel=1e-8)
        assert row["uncorrected_same"] == pytest.approx(hand["naive"][j], rel=1e-8)
        assert row["ci_low"] < row["calibrated"] < row["ci_high"]
        assert (row["n_primary"], row["n"]) == (400, 370)
    assert "contrast/fat_g/carbohydrate_g" in rows
    assert table.caption.endswith(f"A declared secondary analysis: it {LABEL}.")
    assert "on its 400 participants" in table.caption and "(370 participants;" in table.caption
    assert "Refit on the same 370 participants" in table.caption
    csv = tables.to_csv(table)
    assert csv.splitlines()[0].startswith("key,term,intake,n_primary,n,uncorrected,")
    source = type("S", (), {"purpose": "inference", "artifact": staticmethod(
        lambda stage: {"calibration": cal, "fit": fit}.get(stage))})()
    paragraphs = [p.text for p in methods.analysis_paragraphs(source)]
    assert [p for p in paragraphs if p.startswith("Usual intakes of all energy sources were "
                                                  "calibrated jointly")] == [cal["methods"]]
    # A blocked calibration: no table, no paragraph of its own; the record says it.
    blocked = {**cal, "applies": False, "exposures": [], "contrasts": [],
               "blocked": "a question it rests on was not settled"}
    assert tables.calibration_table(blocked) == []
    source = type("S", (), {"purpose": "inference", "artifact": staticmethod(
        lambda stage: {"calibration": blocked}.get(stage))})()
    assert methods.analysis_paragraphs(source) == []
    answer = SetMeasurementError(method="regression_calibration", n_boot=50)
    log = _log(tmp_path, [d.SetPurpose(purpose="inference"), answer])
    text = methods_text(log.records(), {"counts": record_facts(blocked)})
    (line,) = [x for x in text.lines if x.kind == "set_measurement_error"]
    assert line.sentence == (f"{HEAD}, but a question it rests on was not settled; it was blocked "
                             "and recorded, and the estimates are uncorrected.")


def _files(data: bytes) -> dict[str, bytes]:
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return {n.split("/", 1)[1]: z.read(n) for n in z.namelist() if not n.endswith("/")}


def _wait_fresh(drive: Any, stages: tuple[str, ...], timeout: float = 900.0) -> None:
    """Wait until ``stages`` are computed, reading only their status."""
    import time

    end = time.monotonic() + timeout
    while True:
        statuses = drive.view()["stages"]
        if all(statuses[s]["status"] == "fresh" for s in stages):
            return
        for s in stages:
            assert statuses[s]["status"] != "error", statuses[s]
        assert time.monotonic() < end, {s: statuses[s]["status"] for s in stages}
        time.sleep(0.1)


def test_r5_through_the_server_the_bundle_carries_the_calibration_and_its_replay_reproduces_it(
        tmp_path):
    """The verifier checked the export through the server; so does this. WP12c's two-recall table
    (400 people, the sample, complete cases, nothing held out), regression calibration declared
    with 50 whole-chain resamples:

    * the export waits for the calibration, then serves ``results/calibration.csv``: λ, the
      calibrated coefficient and the uncorrected estimate, interval and test are WP12c's NumPy and
      statsmodels reference from the CSV (``reference``: λ and the calibrated coefficient to 1e-6,
      the uncorrected estimate to 1e-8, its HC3 interval to 1e-7); its bootstrap interval is the
      artifact's as served, and its caption carries the label;
    * ``methods.md`` carries the calibration's own paragraph as the stage wrote it, and the
      record's declaration;
    * ``python -m turbotab.replay``, in a fresh home, reproduces every estimate of the bundle, the
      calibration's among them (its seeded whole-chain bootstrap included);
    * the adjustment set then changes (sex leaves the model): the Record's line in force says the
      declaration is re-asked and was not run, word for word, and the export refuses with the
      stage's exits; declared again, the calibration runs and the export is served."""
    import os
    import subprocess
    import sys

    from turbotab.core.tests.acceptance.server_drive import every_row, local_server, open_project
    from turbotab.core.tests.acceptance.test_wp12c_calibration import (RESIDUAL, ROLES,
                                                                       recall_table, recall_truth,
                                                                       reference)
    from turbotab.core.tests.stage_harness import REPO

    frame = recall_table(n=400)
    path = tmp_path / "recalls.csv"
    frame.to_csv(path, index=False)
    stages = ("cohort", "design", "fit", "effects", "calibration")
    with local_server(tmp_path / "home") as client:
        dr = open_project(client, path, recall_truth())

        def force() -> str:
            lines = client.get(f"/api/projects/{dr.pid}/methods").json()["lines"]
            (line,) = [x for x in lines if x["kind"] == "set_measurement_error" and x["in_force"]]
            return line["sentence"]

        dr.decide({"kind": "set_lens", "lenses": ["dietary"]})
        dr.reach("target")
        dr.decide({"kind": "set_target", "column": "ldl"})
        dr.answer("task", {"kind": "set_task", "column": "ldl", "task": "regression"})
        dr.reach("purpose")
        dr.decide({"kind": "set_purpose", "purpose": "inference"})
        dr.reach("grain")
        dr.decide({"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
        dr.answer("repeat_kind", {"kind": "set_repeat_kind", "repeat_kind": "repeats"})
        dr.answer("unit", {"kind": "set_unit", "unit": "unit"})
        dr.answer("aggregation", {"kind": "set_aggregation", "method": "mean"})
        dr.answer("temporal", {"kind": "set_temporal", "temporal": False})
        dr.reach("roles")
        dr.decide_roles(ROLES)
        dr.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        dr.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        dr.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        dr.answer("energy_adjustment", RESIDUAL)
        dr.reach("models")
        dr.decide({"kind": "select_models", "models": ["linear"]})
        dr.decide({"kind": "set_measurement_error", "method": "regression_calibration",
                   "n_boot": 50})
        cal = dr.artifact("calibration")  # an estimate shown: the plan locks
        fit = dr.artifact("fit")
        _wait_fresh(dr, stages)
        exported = client.get(f"/api/projects/{dr.pid}/export")
        declared_said = force()
        recorded_set = dr.view()["state"]["measurement_error"]["adjustment"]
        # the adjustment set changes: sex leaves the model
        dr.decide_roles({**ROLES, "sex": "excluded"})
        stale = dr.artifact("calibration")
        reasked_said = force()
        refused = client.get(f"/api/projects/{dr.pid}/export")
        dr.decide(stale["exits"][0]["decision"])  # declared again under the current set
        redone = dr.artifact("calibration")
        _wait_fresh(dr, stages)
        again = client.get(f"/api/projects/{dr.pid}/export")
    assert exported.status_code == 200, exported.text[:900]
    files = _files(exported.content)
    assert {"results/calibration.csv", "results/calibration.md"} <= set(files)
    table = pd.read_csv(__import__("io").BytesIO(files["results/calibration.csv"]),
                        float_precision="round_trip").set_index("key")
    features = {c["feature"] for c in every_row(fit["models"][0])}
    ref = reference(frame, energy_in_model="energy_kcal" in features)
    (exposure,) = cal["exposures"]
    row = table.loc[exposure["feature"]]
    assert (row["intake"], int(row["n"]), int(row["n_primary"])) == ("protein_g", ref["n"], ref["n"])
    assert row["attenuation"] == pytest.approx(ref["lam"], rel=1e-6)
    assert row["calibrated"] == pytest.approx(ref["estimate"], rel=1e-6)
    assert row["uncorrected"] == pytest.approx(ref["naive"], rel=1e-8)
    assert (row["uncorrected_low"], row["uncorrected_high"]) == pytest.approx(ref["naive_ci"],
                                                                              rel=1e-7)
    assert row["p"] == pytest.approx(ref["p"], rel=1e-6)
    assert np.isnan(row["uncorrected_same"])  # the same people: no refit beside it
    assert (row["ci_low"], row["ci_high"]) == (exposure["ci_low"], exposure["ci_high"])
    assert row["ci_low"] < row["calibrated"] < row["ci_high"]
    caption = files["results/calibration.md"].decode()
    assert LABEL in caption and "on its 400 participants" in caption
    methods_md = files["methods.md"].decode()
    assert cal["methods"] in methods_md
    assert declared_said == f"{HEAD}, with intervals from `50` bootstrap resamples of the whole chain."
    assert declared_said in methods_md
    # the replay, in a fresh home, as a separate process
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(exported.content)
    env = {**os.environ, "TURBOTAB_WORKERS": "2", "OMP_NUM_THREADS": "2", "PYTHONPATH": str(REPO)}
    done = subprocess.run([sys.executable, "-m", "turbotab.replay", str(bundle), "--data", str(path),
                           "--home", str(tmp_path / "replay_home"), "--json"],
                          capture_output=True, text=True, cwd=str(REPO), env=env, timeout=1800)
    import json

    report = json.loads(done.stdout[done.stdout.index("{"):])
    recorded = json.loads(files["provenance.json"])["estimates"]
    ours = [k for k in recorded if k.startswith("calibration|")]
    assert f"calibration|{exposure['feature']}|calibrated" in ours
    assert f"calibration|{exposure['feature']}|ci_low" in ours
    assert done.returncode == 0 and report["reproduced"], done.stdout[-2000:]
    assert not report["estimates"]["mismatched"] and not report["estimates"]["missing"]
    # the adjustment set changed after the declaration
    assert recorded_set == ["age", "energy_kcal", "sex"]
    assert stale["applies"] is False and [e["label"] for e in stale["exits"]] == [
        "Declare the calibration again under the current adjustment set", "Record no calibration"]
    assert reasked_said == (
        f"{HEAD} when the model adjusted for `age`, `energy_kcal` and `sex`; the adjustment set has "
        "changed since, so the declaration is re-asked and was not run, and the estimates are "
        "uncorrected.")
    assert refused.status_code == 409
    error = refused.json()["error"]
    assert stale["reason"] in error["message"]
    assert [e for e in error["exits"] if e["label"] == "Record no calibration"]
    assert redone["applies"] and again.status_code == 200, again.text[:900]
    assert "results/calibration.csv" in _files(again.content)


# ── (6) every refusal of the data's carries an exit ──────────────────────────


def test_r6_when_the_data_cannot_carry_the_calibration_no_correction_is_the_exit(tmp_path):
    """The verifier: "no one has two recalls", "no more people than columns" and "the true
    intakes' covariance not positive" reached the artifact with no exit. Each now carries "Record
    no calibration", a decision the validators accept; through the stage (two recall days of
    protein and energy, energy blank on every second day, so no one has two recalls of both), the
    artifact says it, with its block in the methods' words."""
    rng = np.random.default_rng(3)
    one = RC.Recalls.of(rng.normal(size=(40, 1)), np.arange(40), 40)  # nobody has two recalls
    narrow = RC.Recalls.of(rng.normal(size=(8, 1)), np.repeat(np.arange(4), 2), 4)  # 4 people, 1+3
    a = rng.normal(size=60)  # two days a and 1 − a: every person's mean is 1/2, all spread is daily
    flat = RC.Recalls.of(np.column_stack([a, 1 - a]).reshape(-1, 1), np.repeat(np.arange(60), 2),
                         60)
    for rec, Z in ((one, None), (narrow, rng.normal(size=(4, 3))), (flat, None)):
        with pytest.raises(RC.CalibrationRefused) as refused:
            RC.calibrate(rec, Z)
        assert refused.value.exits == [NO_CALIBRATION]
    from turbotab.core.tests.acceptance.rc_fixtures import protein_recalls
    from turbotab.core.tests.acceptance.test_ms5_regression_calibration import protein_state

    frame = protein_recalls(seed=8, n=200)
    frame.loc[frame["day"] == 2, "energy_kcal"] = np.nan
    path = tmp_path / "recalls.csv"
    frame.to_csv(path, index=False)
    run = GraphRun(path, tmp_path / "project")
    try:
        state = protein_state("residual")
        out = run.run(state, upto=["findings"])
        state = declared(state, frame, out, n_boot=50)
        cal = run.public(run.run(state, upto=["calibration"]))["calibration"]
    finally:
        run.close()
    assert cal["applies"] is False and cal["exits"] == [NO_CALIBRATION]
    validate(cal["exits"][0]["decision"], ctx_of(state, frame, {"ingest": out["ingest"]}))
    assert cal["reason"] == ("No one has two or more recalls with a value, so the day-to-day "
                             "variance cannot be estimated.")
    assert cal["methods"] == (
        "Regression calibration was declared as a secondary analysis, but no one has two or more "
        "recalls with a value, so the day-to-day variance cannot be estimated; it was blocked and "
        "recorded, and the estimates are uncorrected.")


# ── the contract: the new relations, with live code behind each ──────────────


def test_rc_the_contract_holds_the_repairs_relations_and_each_fires_in_a_chain():
    """BLUEPRINT §13: the repairs enter the one registry as relations of the calibration's
    contract, each naming live code: the lonely PSU's rule, the cluster floor (refused, with its
    exit), the primary's estimate beside the calibrated one, and the export; each fires in the
    chain that holds its condition (``contracts.fired``)."""
    import importlib

    from turbotab.core.contracts import contract, fired

    rc = contract("regression_calibration")
    by = {r.name: r for r in rc.relations}
    assert {"lonely_psu", "cluster_floor", "secondary_beside_uncorrected", "in_the_export",
            "adjustment_set_invalidates"} <= set(by)
    assert (by["cluster_floor"].kind, by["cluster_floor"].rung) == ("conflicts", "refused")
    assert by["cluster_floor"].exits
    for r in rc.relations:
        module, name = r.enforced_by.split(":")
        assert callable(getattr(importlib.import_module(module), name)), r.enforced_by
    chains = {"lonely_psu": ["a lonely PSU resampled as the primary centers it"],
              "cluster_floor": ["fewer clusters than the floor"],
              "in_the_export": ["the calibration in the manuscript bundle"]}
    for name, consequences in chains.items():
        firing = fired({"regression_calibration": "multivariate"}, "inference", consequences)
        assert [f.relation.name for f in firing] == [name]
        assert not fired({"regression_calibration": "multivariate"}, "prediction", consequences)
