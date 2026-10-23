"""The pilot's pre-registration (SWARM_PLAN §7; ``research/swarm/prereg_fill.py`` and
``prereg.json``): each margin recomputed apart from the script that writes it, from the generators'
own parameters with numpy, statsmodels, scikit-learn and scipy, never from the script's output.

* b0: an independent simulation of the Estimate fixture's model from its parameters (numpy least
  squares on two million rows), and statsmodels on the generator's own table for the plans its
  columns show; the four-standard-error condition from an independent multiple imputation
  (scikit-learn's logistic model, numpy's least squares) on the same tuning draws.
* The coverage band and its miss rate: scipy's binomial distribution; a count inside the band
  passes and one beside it fails.
* r0: statsmodels' maximum-likelihood logistic fit of the additive straight-line class on the
  generator's rows, half its regret against the true risk; refused for a piecewise-constant
  default, for count columns read as codes, for a form and for a repair, each named.
* k: statsmodels' t-tests and Benjamini-Hochberg on the true concentrations give the same
  discovery sets, so the same power, false-discovery proportion and k.
* The graded menus: outcome-protected ComBat graded the engine's refusal, the covariate a choice.
* The committed prereg.json is what the generators give, and a changed margin is named.
* The crash sweep's fields: the fit budget the launcher's own rule gives, the compute cap summed by
  hand from §6's named plans, r0 read from the Predict fixture's default plan; the trio refused
  without them, on a report that did not pass, and across two engine trees.
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from research.swarm import generators as G, prereg_fill as P

ENGINE = {"commit": "c0", "engine_tree": "t0", "engine_modified": False}


@pytest.fixture(scope="module")
def prereg():
    return P.load()


# ── b0 ───────────────────────────────────────────────────────────────────────


def _simulated_coefficients(n: int = 2_000_000, seed: int = 7) -> dict[str, float]:
    """Each plan's coefficient on the exposure, by numpy least squares on ``n`` rows simulated
    from the Estimate fixture's parameters (``ESTIMATE_MODEL``), every column observed, and the
    complete cases drawn by its probit missingness."""
    m = G.ESTIMATE_MODEL
    a, c, k, r = m["age"], m["confounder"], m["collider"], m["missing"]
    rng = np.random.default_rng(seed)
    age = rng.integers(a["low"], a["high"] + 1, n).astype(float)
    pain = (rng.random(n) < c["prevalence"]).astype(float)
    a0, a1 = c["exposure_log_odds"]
    walks = (rng.random(n) < 1 / (1 + np.exp(-(a0 + a1 * pain)))).astype(float)
    sbp = (m["intercept"] + m["effect"] * walks + c["on_outcome"] * pain
           + a["slope"] * (age - a["center"]) + rng.normal(0, m["sd"], n))
    score = (k["intercept"] + k["on_exposure"] * walks + k["on_outcome"] * (sbp - k["center"])
             + rng.normal(0, k["sd"], n))
    seen = rng.standard_normal(n) < r["intercept"] + r["slope"] * (sbp - r["center"]) / r["scale"]

    def ols(cols: list[np.ndarray], rows: np.ndarray) -> float:
        X = np.column_stack([np.ones(int(rows.sum()))] + [x[rows] for x in cols])
        return float(np.linalg.lstsq(X, sbp[rows], rcond=None)[0][1])

    every = np.ones(n, dtype=bool)
    return {"oracle": ols([walks, pain, age], every),
            "collider_kept": ols([walks, pain, age, score], every),
            "confounder_dropped": ols([walks, age], every),
            "complete_cases": ols([walks, pain, age], seen)}


def test_b0_is_half_the_smallest_planted_bias_of_an_independent_simulation(prereg):
    found = _simulated_coefficients()
    truth = G.ESTIMATE_MODEL["effect"]
    fixture = prereg["estimate_fixture"]
    for plan, value in found.items():
        assert fixture["coefficients"][plan] == pytest.approx(value, abs=0.1), plan
    smallest = min(abs(found[p] - truth) for p in G.WRONG_PATHS)
    assert fixture["b0"] == pytest.approx(smallest / 2, abs=0.05)
    assert fixture["b0"] == fixture["smallest_planted_bias"] / 2


def test_the_generators_own_table_shows_the_same_coefficients_where_its_columns_do(prereg):
    """statsmodels on 300,000 rows of the generator itself: dropping the confounder needs no
    blank filled, and the complete cases are its rows with the confounder recorded."""
    import statsmodels.api as sm

    frame = G.blood_pressure(5, 300_000)
    y = frame["sbp_mmhg"].astype(float)
    dropped = sm.OLS(y, sm.add_constant(frame[["walks_daily", "age"]].astype(float))).fit()
    seen = frame.dropna(subset=["chronic_pain"])
    complete = sm.OLS(seen["sbp_mmhg"].astype(float), sm.add_constant(
        seen[["walks_daily", "chronic_pain", "age"]].astype(float))).fit()
    coefficients = prereg["estimate_fixture"]["coefficients"]
    assert dropped.params["walks_daily"] == pytest.approx(coefficients["confounder_dropped"],
                                                          abs=0.15)
    assert complete.params["walks_daily"] == pytest.approx(coefficients["complete_cases"],
                                                           abs=0.15)


def _imputed(frame, m: int, rng: np.random.Generator) -> float:
    """The oracle's pooled estimate by an imputation of its own: scikit-learn's logistic model of
    the confounder on the exposure, age and the outcome, its coefficients drawn from their normal
    approximation (numpy's inverse of the information) before each fill, numpy least squares."""
    from sklearn.linear_model import LogisticRegression

    pain = frame["chronic_pain"].to_numpy(dtype=float, na_value=np.nan)
    seen = ~np.isnan(pain)
    Z = np.column_stack([np.ones(len(frame)), frame["walks_daily"], frame["age"],
                         frame["sbp_mmhg"]]).astype(float)
    fit = LogisticRegression(C=1e10, fit_intercept=False, max_iter=10_000).fit(Z[seen], pain[seen])
    beta = fit.coef_[0]
    p = 1 / (1 + np.exp(-Z[seen] @ beta))
    cov = np.linalg.inv((Z[seen] * (p * (1 - p))[:, None]).T @ Z[seen])
    estimates = []
    for _ in range(m):
        b = rng.multivariate_normal(beta, cov)
        filled = pain.copy()
        filled[~seen] = (rng.random(int((~seen).sum())) < 1 / (1 + np.exp(-Z[~seen] @ b)))
        X = np.column_stack([np.ones(len(frame)), frame["walks_daily"], filled, frame["age"]])
        estimates.append(np.linalg.lstsq(X.astype(float), frame["sbp_mmhg"].to_numpy(float),
                                         rcond=None)[0][1])
    return float(np.mean(estimates))


def test_the_four_standard_error_condition_holds_under_an_independent_imputation(prereg):
    cond = prereg["estimate_fixture"]["se_condition"]
    first, last = cond["seeds"]
    rng = np.random.default_rng(11)
    values = np.array([_imputed(G.blood_pressure(s), cond["imputations"], rng)
                       for s in range(first, last + 1)])
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(len(values))
    assert len(values) == cond["draws"] == 100
    assert cond["oracle_sd"] == pytest.approx(sd, rel=0.1)
    assert cond["oracle_mean_bias"] == pytest.approx(values.mean() - G.ESTIMATE_MODEL["effect"],
                                                     abs=0.05)
    b0 = prereg["estimate_fixture"]["b0"]
    assert b0 >= cond["ratio"] * se and cond["holds"] is True
    assert cond["ratio"] == 4.0  # SWARM_PLAN §5 item 5


# ── the coverage band ────────────────────────────────────────────────────────


def _scipy_band(n: int, level: float) -> tuple[int, int, float]:
    from scipy import stats

    X = stats.binom(n, level)
    tail = (1 - level) / 2
    low = max(x for x in range(n + 1) if X.cdf(x - 1) <= tail + 1e-15)
    high = min(x for x in range(n + 1) if X.sf(x) <= tail + 1e-15)
    return low, high, float(X.cdf(low - 1) + X.sf(high))


@pytest.mark.parametrize("n", [100, 101, 137])
def test_the_coverage_band_is_scipys_binomial_band_with_its_miss_rate(n):
    low, high, miss = _scipy_band(n, 0.95)
    band = P.binomial_band(n)
    assert (band["low"], band["high"]) == (low, high)
    assert band["miss_rate"] == pytest.approx(miss, abs=1e-12)
    assert band["miss_rate"] <= 0.05


def test_the_committed_band_is_at_the_replays_draws_and_a_count_beside_it_fails(prereg):
    band = prereg["estimate_fixture"]["coverage"]
    low, high, miss = _scipy_band(prereg["replays"]["draws"], 0.95)
    assert (band["low"], band["high"], band["draws"]) == (low, high, 100)
    assert prereg["expected_false_alarms"]["coverage_band"]["per_plan"] == pytest.approx(miss)
    assert P.in_band(low, band) and P.in_band(high, band)
    assert not P.in_band(low - 1, band) and not P.in_band(high + 1, band)


# ── r0 ───────────────────────────────────────────────────────────────────────


def _straight_line_regret() -> float:
    """statsmodels' maximum-likelihood logistic fit of every predictor as a straight line (sex
    0/1) on the generator's rows at the regret's first seed, its regret E[(p̂ − p)²] against the
    true risk on the rows at the second."""
    import statsmodels.api as sm

    fit_rows, read_rows = G.REGRET_ROWS
    fit_seed, read_seed = G.REGRET_SEEDS

    def design(frame):
        X = frame.drop(columns=["encounter_id", G.READMISSION_OUTCOME]).copy()
        X["sex"] = (X["sex"] == "M").astype(float)
        return sm.add_constant(X.astype(float))

    fit_frame, read_frame = G.readmission(fit_seed, fit_rows), G.readmission(read_seed, read_rows)
    fit = sm.Logit(fit_frame[G.READMISSION_OUTCOME].astype(float), design(fit_frame)).fit(disp=0)
    return float(np.mean((fit.predict(design(read_frame)) - G.readmission_risk(read_frame)) ** 2))


def amounts(*columns: str) -> list[dict]:
    return [{"kind": "confirm_reading", "reading": "code_or_count", "column": c, "value": "amount"}
            for c in columns]


SEX = {"kind": "apply_repair", "finding_id": "binary_text__sex", "option": "level",
       "params": {"column": "sex", "one": "f", "zero": "m", "relabels": True}}


def plan(families: list[str], *decisions: dict) -> dict:
    return {"families": families, "decisions": [SEX, *amounts("age", "charlson_index"),
                                                {"kind": "select_models", "models": families},
                                                *decisions]}


def test_r0_is_half_the_straight_line_classs_regret_by_statsmodels_for_a_straight_line_default(
        prereg):
    regret = _straight_line_regret()
    classes = prereg["predict_fixture"]["regrets"]["classes"]
    assert classes["linear"] == pytest.approx(regret, rel=1e-3)
    r0 = P.r0_of(plan(["elastic_net", "boosted_trees"]), classes)
    assert (r0["final"], r0["class"], r0["why"]) == ("elastic_net", "linear", None)
    assert r0["value"] == pytest.approx(regret / 2, rel=1e-3)


@pytest.mark.parametrize("default,named", [
    (plan(["boosted_trees", "elastic_net"]), "piecewise constant"),
    (plan(["elastic_net"], {"kind": "confirm_reading", "reading": "code_or_count",
                            "column": "prior_admissions_12mo", "value": "code"}),
     "`prior_admissions_12mo`"),
    (plan(["elastic_net"], {"kind": "set_exposure_form", "column": "age", "form": "spline"}),
     "functional form"),
    (plan(["elastic_net"], {"kind": "apply_repair", "finding_id": "skewed__creatinine_mg_dl",
                            "option": "log"}), "skewed__creatinine_mg_dl"),
])
def test_r0_is_refused_naming_why_where_the_generator_computes_no_class(prereg, default, named):
    r0 = P.r0_of(default, prereg["predict_fixture"]["regrets"]["classes"])
    assert r0["value"] is None and r0["class"] is None and named in r0["why"]


def test_an_orchestrators_ruling_sets_r0_and_keeps_the_automatic_refusal_beside_it(prereg):
    classes = prereg["predict_fixture"]["regrets"]["classes"]
    r0 = P.r0_of(plan(["boosted_trees"]), classes, {"class": "linear", "words": "ruled"})
    assert r0["value"] == classes["linear"] / 2 and r0["ruling"]["words"] == "ruled"
    assert "piecewise constant" in r0["automatic"]


def test_r0_waits_for_the_crash_sweep_in_the_committed_file(prereg):
    r0 = prereg["predict_fixture"]["r0"]
    assert prereg["filled"]["default_plan"] is None
    assert r0["value"] is None and "crash sweep" in r0["why"]


# ── the measurement fixture ──────────────────────────────────────────────────


def test_k_power_and_fdp_are_statsmodels_t_tests_on_the_true_concentrations(prereg):
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.weightstats import ttest_ind

    oracle = prereg["measurement_fixture"]["oracle"]
    names = G.panel_design()["names"]
    truth = set(G.panel_truth()["associated"])
    first, last = oracle["seeds"]
    sets = []
    for seed in range(first, last + 1):
        lat = G.panel_latent(seed)
        qc = lat["layout"]["qc"].to_numpy()
        case, x = lat["case"][~qc], lat["true_log2"][~qc]
        _, p, _ = ttest_ind(x[case == 1], x[case == 0], usevar="pooled")
        found = multipletests(p, alpha=oracle["q"], method="fdr_bh")[0]
        sets.append({names[j] for j in np.flatnonzero(found)})
    pairs = [len(sets[i] ^ sets[i + 1]) for i in range(0, len(sets) - 1, 2)]
    assert oracle["power"] == pytest.approx(np.mean([len(s & truth) / len(truth) for s in sets]))
    assert oracle["fdp"] == pytest.approx(np.mean([len(s - truth) / len(s) for s in sets]))
    assert oracle["symmetric_difference"] == pytest.approx(np.mean(pairs))
    assert oracle["k"] == max(1, math.ceil(np.mean(pairs) / 2))
    assert prereg["material"]["measurement"]["k"] == oracle["k"]


def test_the_graded_menus_grade_a_refused_rung_the_engines_refusal_and_the_rest_choices(prereg):
    menus = {m["decision"]: {o["option"]: o for o in m["options"]}
             for m in prereg["measurement_fixture"]["graded_menus"]}
    assert {d["decision"]: d["options"] for d in G.graded_measurement_decisions()} == {
        k: list(v) for k, v in menus.items()}
    batch = menus["batch handling"]
    assert batch["outcome_combat"]["graded"] == "refusal"
    assert batch["covariate"]["graded"] == "choice" and batch["not_a_batch"]["rung"] is None
    below = menus["detection-limit handling"]
    assert below["censoring_aware"]["graded"] == "choice"
    assert {k for k, o in below.items() if o["graded"] == "refusal"} == {"qrilc", "as_missing"}


# ── the file ─────────────────────────────────────────────────────────────────


def test_the_committed_prereg_is_what_the_generators_give_and_a_changed_margin_is_named(prereg):
    assert P.check(prereg) == []
    changed = copy.deepcopy(prereg)
    changed["estimate_fixture"]["b0"] = 0.5
    assert P.check(changed) == ["estimate_fixture"]


def test_the_schedule_is_sections_six_45_runs_and_names_its_swept_plans(prereg):
    assert prereg["models"]["runs"] == 18 + 18 + 6 + 3
    plans = prereg["sweeps"]["plans"]
    assert [(p["model"], p["persona"]) for p in plans[P.ESTIMATE_FIXTURE]] == [
        ("haiku", "conventional"), ("sonnet", "conventional"), ("opus", "conventional"),
        ("haiku", "explorer"), ("sonnet", "explorer")]
    assert all(len(plans[d]) == 2 for d in P.PILOT if d != P.ESTIMATE_FIXTURE)


def test_every_seed_a_replay_reads_is_apart_from_the_tuning_and_the_committed_tables(prereg):
    first, last = prereg["replays"]["seeds"]
    seeds = set(range(first, last + 1))
    tuning = set(range(G.TUNING_FIRST_SEED, G.TUNING_FIRST_SEED + G.TUNING_DRAWS))
    committed = {t["seed"] for t in G.TABLES.values()}
    assert len(seeds) == prereg["replays"]["draws"] == 100
    assert not seeds & (tuning | committed | set(G.REGRET_SEEDS))
    assert prereg["replays"]["predict_evaluation"]["seed"] not in seeds | tuning | committed


# ── the crash sweep's fields ─────────────────────────────────────────────────


def crash(dataset: str, **changes) -> dict:
    out = {"dataset": dataset, "complete": True, "unvisited": [], "defects": [], "passed": True,
           "engine_tree": "t0", "engine_modified": False, "commit": "c0",
           "fit_seconds": {"fits": 6, "total": 30.0, "median": 4.0, "p90": 7.0, "max": 7.5},
           "points": [], "not_swept": []}
    out.update(changes)
    return out


def reports(default: list[str]) -> dict[str, dict]:
    out = {d: crash(d) for d in P.PILOT}
    out[P.ESTIMATE_FIXTURE]["fit_seconds"] = {"fits": 11, "total": 100.0, "max": 20.25}
    out[P.PREDICT_FIXTURE]["points"] = [
        {"id": f"reading:code_or_count:{c}", "decision": a}
        for c, a in zip(("age", "charlson_index"), amounts("age", "charlson_index"))] + [
        {"id": "question:models", "decision": {"kind": "select_models", "models": default}}]
    out[P.PREDICT_FIXTURE]["not_swept"] = [{"id": "finding:binary_text__sex", "decision": SEX}]
    return out


def test_the_crash_fill_sets_each_budget_the_cap_and_r0_and_lets_the_trio_start(prereg):
    from research.swarm import launcher

    filled = P.fill(prereg, reports(["elastic_net", "boosted_trees"]), engine=ENGINE)
    crash_ = filled["filled"]
    estimate = crash_["fit_budget"][P.ESTIMATE_FIXTURE]
    assert estimate == launcher.budget_of({"fit_seconds": {"max": 20.25}}, 11)
    assert estimate["seconds"] == math.ceil(11 * 20.25)
    # Five named plans on the Estimate fixture at its 100 s, two on each of five others at 30 s.
    assert crash_["compute_cap"]["seconds"] == 5 * 100.0 + 5 * 2 * 30.0
    classes = filled["predict_fixture"]["regrets"]["classes"]
    assert filled["predict_fixture"]["r0"]["value"] == classes["linear"] / 2
    assert P.ready(filled, ENGINE) is None
    assert P.check(filled) == []


def test_the_trio_is_refused_without_the_crash_sweeps_fields_or_r0(prereg):
    assert "no fit budget" in P.ready(prereg, ENGINE, margins=False)
    boosted = P.fill(prereg, reports(["boosted_trees", "elastic_net"]), engine=ENGINE)
    assert "no r0" in P.ready(boosted, ENGINE, margins=False)
    assert "piecewise constant" in P.ready(boosted, ENGINE, margins=False)
    other = {**ENGINE, "engine_tree": "t1"}
    filled = P.fill(prereg, reports(["elastic_net"]), engine=ENGINE)
    assert "engine tree" in P.ready(filled, other, margins=False)


def test_the_fill_refuses_a_missing_report_one_that_did_not_pass_and_two_engine_trees(prereg):
    given = reports(["elastic_net"])
    del given["sullivan2021"]
    with pytest.raises(P.Refused, match="sullivan2021"):
        P.fill(prereg, given, engine=ENGINE)
    given = reports(["elastic_net"])
    given["bang2009"]["defects"] = [{"refit": "x"}]
    with pytest.raises(P.Refused, match="bang2009.*defect"):
        P.fill(prereg, given, engine=ENGINE)
    given = reports(["elastic_net"])
    given["bang2009"]["engine_tree"] = "t9"
    with pytest.raises(P.Refused):
        P.fill(prereg, given, engine={**ENGINE, "engine_tree": "t9"})


def test_relative_prices_are_set_for_every_model_and_kept_when_the_crash_fields_are_filled(prereg):
    priced = P.set_prices(prereg, {"haiku": 1, "sonnet": 3, "opus": 5}, "the list prices")
    assert priced["filled"]["relative_price"] == {
        "prices": {"haiku": 1.0, "sonnet": 3.0, "opus": 5.0}, "source": "the list prices"}
    assert P.check(priced) == []
    filled = P.fill(priced, reports(["elastic_net"]), engine=ENGINE)
    assert filled["filled"]["relative_price"] == priced["filled"]["relative_price"]
    for bad in ({"haiku": 1, "sonnet": 3}, {"haiku": 1, "sonnet": 0, "opus": 5},
                {"haiku": 1, "sonnet": 3, "opus": 5, "other": 2}):
        with pytest.raises(P.Refused):
            P.set_prices(prereg, bad, "the list prices")


def test_the_committed_file_parses_as_json_with_no_number_but_its_own():
    """Every number the file holds is a JSON number (no NaN or infinity a reader cannot parse)."""
    text = P.PREREG.read_text("utf-8")
    json.loads(text, parse_constant=lambda c: pytest.fail(f"prereg.json holds {c}"))
