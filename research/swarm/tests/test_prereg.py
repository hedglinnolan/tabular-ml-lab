"""The pilot's pre-registration (SWARM_PLAN §7; ``research/swarm/prereg_fill.py`` and
``prereg.json``): each margin recomputed apart from the script that writes it, from the generators'
own parameters with numpy, statsmodels, scikit-learn and scipy, never from the script's output.

* b0: an independent simulation of the Estimate fixture's model from its parameters (numpy least
  squares on two million rows), and statsmodels on the generator's own table for the plans its
  columns show; the four-standard-error condition from an independent multiple imputation
  (scikit-learn's logistic model, numpy's least squares) on the same tuning draws.
* The coverage band and its miss rate: scipy's binomial distribution; a count inside the band
  passes and one beside it fails.
* r0 (the orchestrator's ruling of 2026-10-23): half the default plan's mean regret on the
  pre-registered draws and rows, numpy's mean of a constructed replay's regrets; the fill refuses a
  record that is not the default path's exported plan with its final family on the frozen engine
  (a crash sweep's commit, or git's tree of turbotab/ at its commit the crash sweeps'), and a
  replay that is not that export on the Predict fixture on those draws and rows on the frozen
  engine, or one refused on a draw; the Predict fixture's launch waits for it, and for one
  measured on the frozen engine. A crash sweep whose default plan declares another final than the ruling's family is
  refused, naming both. Its reference, half the additive straight-line class's regret, is
  statsmodels' maximum-likelihood logistic fit on the generator's rows against the true risk.
* k (the orchestrator's ruling of 2026-10-23): statsmodels' t-tests and Benjamini-Hochberg on the
  true concentrations give the same discovery sets, so the same power, false-discovery proportion
  and mean draw-to-draw change, whose rounding up is k.
* The graded menus: outcome-protected ComBat graded the engine's refusal, the covariate a choice.
* The committed prereg.json is what the generators give with its fills, and a changed margin is
  named. Every other test starts from what the generators give with nothing filled
  (``prereg_fill.build``), each fill constructed, so it holds whatever the pilot's real fills are.
* The crash sweep's fields: the fit budget the launcher's own rule gives, the compute cap summed by
  hand from §6's named plans, the Predict fixture's default plan; the pilot refused without them,
  on a report that did not pass, across two engine trees, and on a report whose table is not the
  one its dataset's manifest names (another table's SHA-256, or none).
* The staged pre-registration (the orchestrator's ruling of 2026-10-28): a crash fill of any
  pilot datasets adds to what is filled, each entry with when it was filled; a dataset with its
  own budget may launch while another without one is refused in words; r0 missing refuses the
  Predict fixture only; a report on another tree, or the engine now on another, refuses; a fill
  after a graded run on that dataset (a launch of it) refuses, and one on another dataset does
  not; the compute cap waits for every pilot dataset. Bang 2009 is no pilot dataset.
* The margins fixed at the first graded run: once the pilot has one, a crash fill, the r0 fill,
  the prices and the plain write go through while the generators give what prereg.json holds,
  and each is refused, naming the part and the run, once a rule has changed (§7's unit's
  minimum, which the unit's part alone states); with no graded run the changed rule is written.
  With a graded run and no prereg.json, the plain write writes none; with none, it writes one.
  A launch kept in a repository's main checkout refuses a fill made in a worktree of it
  (``git worktree add``), as in the main checkout; where git cannot place that folder (a ``.git``
  file naming no repository), the plain write is refused in the launcher's words.
"""
from __future__ import annotations

import copy
import json
import math
import re

import numpy as np
import pytest

from research.swarm import generators as G, prereg_fill as P

ENGINE = {"commit": "c0", "engine_tree": "t0", "engine_modified": False}


@pytest.fixture(scope="module")
def prereg():
    """What the generators give with nothing filled, the content of the plain write with no file
    to keep fills from (``prereg_fill.build``): the base every fill here starts from, so each
    test holds whatever the pilot's real fills in the committed file are."""
    return P.build()


@pytest.fixture(scope="module")
def committed():
    """The committed prereg.json, with whatever fills it holds."""
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


def test_r0s_reference_is_half_the_straight_line_classs_regret_by_statsmodels(prereg):
    regret = _straight_line_regret()
    classes = prereg["predict_fixture"]["regrets"]["classes"]
    assert classes["linear"] == pytest.approx(regret, rel=1e-3)
    reference = prereg["predict_fixture"]["r0"]["reference"]
    assert reference["class"] == "linear" and reference["value"] == classes["linear"] / 2
    assert reference["value"] == pytest.approx(regret / 2, rel=1e-3)
    assert "not r0" in reference["label"]


def test_r0_waits_for_the_crash_sweep_and_its_measure_in_the_committed_file(prereg, committed):
    """With nothing filled, r0 waits for the Predict fixture's crash sweep; once it is filled,
    for its measure. The committed file, whatever its fills, holds a value only once r0 is
    filled, and until then says which of the two it waits for; its ruling is the one the
    generators give."""
    r0 = prereg["predict_fixture"]["r0"]
    assert prereg["filled"]["default_plan"] is None and prereg["filled"]["r0"] is None
    assert r0["value"] is None and "crash sweep" in r0["why"]
    crashed = P.fill(prereg, reports(["boosted_trees", "elastic_net"]), engine=ENGINE)
    waiting = crashed["predict_fixture"]["r0"]
    assert waiting["value"] is None and "has not been measured" in waiting["why"]
    held = committed["predict_fixture"]["r0"]
    if committed["filled"]["r0"] is None:
        assert held["value"] is None
        assert ("crash sweep" if committed["filled"]["default_plan"] is None
                else "has not been measured") in held["why"]
    else:
        assert held["value"] is not None
    assert held["ruling"] == r0["ruling"]
    ruling = r0["ruling"]
    assert (ruling["date"], ruling["by"]) == ("2026-10-23", "the orchestrator, the methods lead")
    assert "measured" in ruling["words"] and "seeds 1001-1100" in ruling["words"]
    assert "seed 2001 x 20,000" in ruling["words"]
    # The family the ruling's words name is the one the fill holds it to (the orchestrator's
    # ruling of 2026-10-23 names boosted_trees).
    assert P.RULED_FINAL == "boosted_trees"
    assert ruling["words"].startswith(
        f"The default path on the Predict fixture declares {P.RULED_FINAL} final, ")


def test_a_crash_sweep_whose_default_plan_declares_another_final_is_refused_naming_both(prereg):
    """The ruling holds for the family it was made for: a default plan declaring another final is
    refused at the crash fill and at the r0 fill, naming both, so the orchestrator rules again;
    one declaring the ruling's family is filled."""
    with pytest.raises(P.Refused, match="`elastic_net` final.*made for `boosted_trees`"):
        P.fill(prereg, reports(["elastic_net", "boosted_trees"]), engine=ENGINE)
    crashed = P.fill(prereg, reports(["boosted_trees", "elastic_net"]), engine=ENGINE)
    assert crashed["predict_fixture"]["r0"]["final"] == "boosted_trees"
    other = P.build(crashed["filled"] | {"default_plan": crashed["filled"]["default_plan"]
                                          | {"families": ["elastic_net"]}})
    data = default_record("elastic_net")
    with pytest.raises(P.Refused, match="`elastic_net` final.*made for `boosted_trees`"):
        P.fill_r0(other, data, default_replay(other, data))


def default_record(family: str | None = "boosted_trees", **run) -> dict:
    """The default path's record on the Predict fixture (``arms.run_arm``' model), its final
    family ``family``, its export's bundle the constructed one, made at the crash sweeps' commit
    (:func:`crash`) with no file modified."""
    from research.swarm.tests import constructed as C

    data = C.predict_record()
    data["run"].update({"id": "run_default", "analyst_model": "mechanical-default",
                        "persona": None, "engine_commit": ENGINE["commit"],
                        "engine_modified": False} | run)
    data["dataset"]["id"] = P.PREDICT_FIXTURE
    data["result"]["family"] = family
    return data


def regrets_of(seeds: list[int]) -> list[float]:
    return [0.004 + 0.001 * ((7 * s) % 5) for s in seeds]


def default_replay(prereg, data: dict, **changes) -> dict:
    """``data``'s replay as ``score_pilot.make_replay`` writes one, on the pre-registered draws
    and evaluation rows on the crash sweeps' engine, each draw's regret :func:`regrets_of` its
    seed."""
    first, last = prereg["replays"]["seeds"]
    seeds = list(range(first, last + 1))
    fx = prereg["predict_fixture"]
    out = {"run": data["run"]["id"], "dataset": P.PREDICT_FIXTURE, "generator": fx["generator"],
           "recorded_seed": fx["recorded_seed"],
           "target": {"kind": "predict", "family": data["result"]["family"]},
           "evaluation": dict(prereg["replays"]["predict_evaluation"]),
           "bundle_sha256": data["export"]["sha256"], "engine": dict(ENGINE),
           "draws": [{"seed": s, "refused": None, "regret": r}
                     for s, r in zip(seeds, regrets_of(seeds))]}
    out.update(changes)
    return out


def measured(prereg) -> dict:
    """``prereg`` with its crash fields filled (the default plan's final family boosted_trees)
    and r0 measured on :func:`default_replay`."""
    crashed = P.fill(prereg, reports(["boosted_trees", "elastic_net"]), engine=ENGINE)
    data = default_record()
    return P.fill_r0(crashed, data, default_replay(crashed, data))


def test_r0_is_half_the_default_plans_mean_regret_on_the_preregistered_draws_and_rows(prereg):
    crashed = P.fill(prereg, reports(["boosted_trees", "elastic_net"]), engine=ENGINE)
    why = P.ready(crashed, ENGINE, margins=False)
    assert "no r0" in why and "prereg_fill r0" in why
    filled = measured(prereg)
    first, last = prereg["replays"]["seeds"]
    expected = float(np.mean(regrets_of(list(range(first, last + 1))))) / 2
    r0 = filled["predict_fixture"]["r0"]
    assert r0["value"] == pytest.approx(expected, rel=1e-12)
    assert (r0["final"], r0["why"]) == ("boosted_trees", None)
    assert filled["filled"]["r0"]["run"] == "run_default"
    assert filled["filled"]["r0"]["draws"] == 100
    assert (filled["filled"]["r0"]["engine_tree"], filled["filled"]["r0"]["replay_engine"]) == (
        ENGINE["engine_tree"], ENGINE)
    assert P.ready(filled, ENGINE) is None
    assert P.check(filled) == []
    again = P.fill(filled, reports(["boosted_trees", "elastic_net"]), engine=ENGINE)
    assert again["filled"]["r0"] is None and again["predict_fixture"]["r0"]["value"] is None


def _short(prereg, data):
    replay = default_replay(prereg, data)
    replay["draws"] = replay["draws"][:-1]
    return replay


def _refused_draw(prereg, data):
    replay = default_replay(prereg, data)
    replay["draws"][4] = {"seed": replay["draws"][4]["seed"], "refused": "a level unseen"}
    return replay


def _unexported() -> dict:
    data = default_record()
    data["export"].update(reached=False, sha256=None, methods=None, checklist=None,
                          refused={"code": "result_not_ready", "message": "no fit"})
    return data


def _result_refused() -> dict:
    data = default_record()
    data["result"].update(scores=None, intervals=None, intervals_why="none served",
                          refused="The opening recorded no held-out scores.",
                          refused_kind="not_served")
    return data


@pytest.mark.parametrize("record,replay,named", [
    (lambda: default_record(analyst_model="claude-haiku-5-5"), None, "default path"),
    (lambda: default_record() | {"dataset": default_record()["dataset"] | {"id": "sullivan2021"}},
     None, "default path"),
    (lambda: default_record("elastic_net"), None, "boosted_trees"),
    (default_record, lambda p, d: default_replay(p, d, run="run_other"), "run_other"),
    (default_record, lambda p, d: default_replay(p, d, bundle_sha256="ef" * 32), "bundle"),
    (default_record, lambda p, d: default_replay(p, d, evaluation={"seed": 2001, "rows": 1000}),
     "evaluation rows"),
    (default_record, lambda p, d: default_replay(p, d, generator="tests:small"), "generator"),
    (default_record, lambda p, d: default_replay(p, d, target={"kind": "predict",
                                                               "family": "linear"}), "family"),
    (default_record, _short, "pre-registered"),
    (default_record, _refused_draw, "refused"),
    # The record: exported, its result reported, a family declared final (each guard's words).
    (_unexported, None, "no exported plan with its final family: no fit"),
    (_result_refused, None, "no exported plan with its final family: The opening recorded"),
    (lambda: default_record(None), None, "no exported plan.*no family was declared final"),
    # The replay: the fixture's recorded seed and dataset.
    (default_record, lambda p, d: default_replay(p, d, recorded_seed=7), "recorded seed is 7"),
    (default_record, lambda p, d: default_replay(p, d, dataset=P.ESTIMATE_FIXTURE),
     "dataset is 'walking_bp_estimate'"),
    # The frozen engine: the record's commit and files, the replay's tree and files.
    (lambda: default_record(engine_commit="some-other-commit"), None,
     "commit some-other-commit .*not the frozen engine"),
    (lambda: default_record(engine_modified=True), None, "tracked files modified"),
    (lambda: default_record(engine_modified=None), None, "not known to be unmodified"),
    (default_record, lambda p, d: default_replay(p, d, engine={**ENGINE, "engine_tree": "t1"}),
     "engine tree t1 \\(no file modified\\); the frozen engine"),
    (default_record, lambda p, d: default_replay(p, d, engine={**ENGINE, "engine_modified": True}),
     "engine tree t0 \\(tracked files modified\\)"),
    (default_record, lambda p, d: default_replay(p, d, engine=None),
     "engine tree None \\(not known to be unmodified\\)"),
])
def test_the_r0_fill_refuses_a_replay_not_the_default_plans_on_the_preregistered_draws(
        prereg, record, replay, named):
    crashed = P.fill(prereg, reports(["boosted_trees", "elastic_net"]), engine=ENGINE)
    data = record()
    given = replay(crashed, data) if replay else default_replay(crashed, data)
    with pytest.raises(P.Refused, match=named):
        P.fill_r0(crashed, data, given)


def test_the_r0_fill_takes_a_record_whose_commit_has_the_crash_sweeps_engine_tree(prereg):
    """The frozen engine is a tree of turbotab/, the same at every commit that leaves it
    unchanged (launcher.py): a record made at another commit whose tree git resolves to the crash
    sweeps' is on it, and one at a commit git does not resolve is not."""
    import subprocess

    def git(*args):
        out = subprocess.run(["git", "-C", str(P.REPO), *args], capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None

    head, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD:turbotab")
    if not head or not tree:
        pytest.skip("not run from a git checkout")
    engine = {**ENGINE, "engine_tree": tree}
    given = reports(["boosted_trees", "elastic_net"])
    for r in given.values():
        r["engine_tree"] = tree
    crashed = P.fill(prereg, given, engine=engine)
    data = default_record(engine_commit=head)
    assert crashed["filled"]["reports"][P.PREDICT_FIXTURE]["commit"] == "c0" != head
    filled = P.fill_r0(crashed, data, default_replay(crashed, data, engine=engine))
    assert filled["filled"]["r0"]["engine_tree"] == tree
    assert P.ready(filled, engine, margins=False) is None
    data = default_record(engine_commit="0" * 40)
    with pytest.raises(P.Refused, match="not the frozen engine"):
        P.fill_r0(crashed, data, default_replay(crashed, data, engine=engine))


def test_the_pilot_is_refused_where_r0_was_measured_on_another_engine(prereg):
    """ready() reads the engine r0 was measured on: a filled r0 whose record's tree or replay's
    tree is not the crash sweeps', or with a file modified, holds the pilot whole and the Predict
    fixture; the measured one does not."""
    filled = measured(prereg)
    assert P.ready(filled, ENGINE, margins=False) is None
    entry = filled["filled"]["r0"]
    for changed in ({"engine_tree": "t1"}, {"engine_modified": True},
                    {"replay_engine": {**ENGINE, "engine_tree": "t1"}},
                    {"replay_engine": {**ENGINE, "engine_modified": True}}):
        other = P.build(filled["filled"] | {"r0": entry | changed})
        assert other["predict_fixture"]["r0"]["value"] is not None
        assert "measure it again" in P.ready(other, ENGINE, margins=False), changed


def test_r0_is_measured_only_after_the_crash_fill(prereg):
    data = default_record()
    with pytest.raises(P.Refused, match="crash"):
        P.fill_r0(prereg, data, default_replay(prereg, data))


def test_the_r0_command_writes_the_measure_it_reads(prereg, tmp_path):
    crashed = P.fill(prereg, reports(["boosted_trees", "elastic_net"]), engine=ENGINE)
    path = P.write(crashed, tmp_path / "prereg.json")
    data = default_record()
    (tmp_path / "record.json").write_text(json.dumps(data), "utf-8")
    (tmp_path / "replay.json").write_text(json.dumps(default_replay(crashed, data)), "utf-8")
    assert P.main(["--prereg", str(path), "r0", "--record", str(tmp_path / "record.json"),
                   "--replay", str(tmp_path / "replay.json")]) == 0
    written = P.load(path)
    assert written["predict_fixture"]["r0"]["value"] == measured(prereg)["predict_fixture"][
        "r0"]["value"]
    assert written["filled"]["r0"]["replay_sha256"] == P._sha256(tmp_path / "replay.json")


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
    assert oracle["k"] == max(1, math.ceil(np.mean(pairs))) == 2
    assert prereg["material"]["measurement"]["k"] == oracle["k"]
    ruling = prereg["material"]["measurement"]["ruling"]
    assert (ruling["date"], ruling["by"]) == ("2026-10-23", "the orchestrator, the methods lead")
    assert f"({oracle['k']}, from {oracle['symmetric_difference']:g})" in ruling["words"]


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


def test_the_committed_prereg_is_what_the_generators_give_and_a_changed_margin_is_named(committed):
    assert P.check(committed) == []
    changed = copy.deepcopy(committed)
    changed["estimate_fixture"]["b0"] = 0.5
    assert P.check(changed) == ["estimate_fixture"]


def test_the_schedule_is_sections_six_39_runs_on_five_datasets_and_names_its_swept_plans(prereg):
    """R1 (the owner and the orchestrator, 2026-10-28): Bang 2009 leaves the pilot, so Haiku and
    Sonnet each run three personas on five datasets (15 runs each), beside the six Haiku
    replicates and the three Opus runs on the Estimate fixture."""
    assert "bang2009" not in P.PILOT and len(P.PILOT) == 5
    assert prereg["datasets"]["pilot"] == list(P.PILOT)
    assert prereg["models"]["runs"] == 3 * 5 + 3 * 5 + 6 + 3
    assert "bang2009" not in prereg["sweeps"]["plans"]


def test_the_plan_states_the_pilots_datasets_and_runs_as_prereg_json_holds_them(prereg):
    """SWARM_PLAN §6's count of runs and of datasets is prereg.json's (the plan is prose, so this
    holds it to the schedule the code writes)."""
    import re

    text = (P.REPO / "docs" / "turbotab-next" / "plans" / "SWARM_PLAN.md").read_text("utf-8")
    words = {5: "five", 6: "six"}
    runs = prereg["models"]["runs"]
    assert f"**The pilot: {words[len(P.PILOT)]} datasets, {runs} agent runs.**" in text
    assert f"**Agent runs** ({runs}):" in text
    per = len(P.PERSONAS) * len(P.PILOT)
    rows = re.findall(rf"\| (Haiku|Sonnet) 5\.5, high effort, each persona \| all "
                      rf"{words[len(P.PILOT)]} \| (\d+) \|", text)
    assert rows == [("Haiku", str(per)), ("Sonnet", str(per))]
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


def table_of(dataset: str) -> dict:
    """The table ``dataset``'s manifest names, as a crash sweep's report names the table it ran on
    (``sweep.crash``'s ``data``): its file and SHA-256, read from the manifest itself."""
    from research.swarm import launcher

    table = json.loads((launcher.DATASETS / dataset / "manifest.json").read_text("utf-8"))["table"]
    return {"path": table["file"], "sha256": table["sha256"]}


def crash(dataset: str, **changes) -> dict:
    """A crash sweep's report on ``dataset`` that passed, in ``sweep.crash``'s shape: every decision
    of the default plan swept (``only`` None) on the table its manifest names."""
    out = {"dataset": dataset, "complete": True, "unvisited": [], "defects": [], "passed": True,
           "engine_tree": "t0", "engine_modified": False, "commit": "c0",
           "fit_seconds": {"fits": 6, "total": 30.0, "median": 4.0, "p90": 7.0, "max": 7.5},
           "points": [], "not_swept": [], "only": None, "data": table_of(dataset)}
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


def test_the_crash_fill_sets_each_budget_the_cap_and_the_default_plans_final_family(prereg):
    from research.swarm import launcher

    filled = P.fill(prereg, reports(["boosted_trees", "elastic_net"]), engine=ENGINE)
    crash_ = filled["filled"]
    estimate = crash_["fit_budget"][P.ESTIMATE_FIXTURE]
    assert estimate == launcher.budget_of({"fit_seconds": {"max": 20.25}}, 11)
    assert estimate["seconds"] == math.ceil(11 * 20.25)
    # Five named plans on the Estimate fixture at its 100 s, two on each of four others at 30 s.
    assert crash_["compute_cap"]["seconds"] == 5 * 100.0 + 4 * 2 * 30.0
    r0 = filled["predict_fixture"]["r0"]
    assert (r0["final"], r0["value"]) == ("boosted_trees", None)
    assert "not been measured" in r0["why"]
    assert P.check(filled) == []


def test_the_pilot_is_refused_without_the_crash_sweeps_fields_or_r0(prereg):
    assert "no fit budget" in P.ready(prereg, ENGINE, margins=False)
    boosted = P.fill(prereg, reports(["boosted_trees", "elastic_net"]), engine=ENGINE)
    assert "no r0" in P.ready(boosted, ENGINE, margins=False)
    other = {**ENGINE, "engine_tree": "t1"}
    assert "engine tree" in P.ready(measured(prereg), other, margins=False)


def test_the_fill_refuses_a_dataset_off_the_pilot_one_that_did_not_pass_and_two_engine_trees(
        prereg):
    """Bang 2009, deferred, is no pilot dataset: its crash sweep fills nothing. (No report at all
    is refused too; a subset of the pilot's is filled, below.)"""
    with pytest.raises(P.Refused, match="bang2009 is not one of the pilot's datasets"):
        P.fill(prereg, {"bang2009": crash("bang2009")}, engine=ENGINE)
    with pytest.raises(P.Refused, match="No crash sweep report"):
        P.fill(prereg, {}, engine=ENGINE)
    given = reports(["elastic_net"])
    given["nhanes_glucose_predict"]["defects"] = [{"refit": "x"}]
    with pytest.raises(P.Refused, match="nhanes_glucose_predict.*defect"):
        P.fill(prereg, given, engine=ENGINE)
    given = reports(["elastic_net"])
    given["nhanes_glucose_predict"]["engine_tree"] = "t9"
    with pytest.raises(P.Refused):
        P.fill(prereg, given, engine={**ENGINE, "engine_tree": "t9"})


def test_the_fill_refuses_a_report_whose_table_is_not_the_manifests(prereg):
    """The reports on the tables the manifests name fill the crash fields; one that ran on another
    table (here sullivan2021's, its manifest's SHA-256), or names no table's SHA-256, does not."""
    assert P.check(P.fill(prereg, reports(["boosted_trees", "elastic_net"]), engine=ENGINE)) == []
    other = table_of("sullivan2021")
    glucose = "nhanes_glucose_predict"
    assert other["sha256"] != table_of(glucose)["sha256"]  # the construction: another table
    for data in (other, {"path": other["path"], "sha256": None}, None):
        given = reports(["boosted_trees", "elastic_net"])
        given[glucose]["data"] = data
        with pytest.raises(P.Refused, match=f"{glucose}: .*not the manifest's"):
            P.fill(prereg, given, engine=ENGINE)
    given = reports(["boosted_trees", "elastic_net"])
    del given[glucose]["data"]
    with pytest.raises(P.Refused, match=f"{glucose}: .*not the manifest's"):
        P.fill(prereg, given, engine=ENGINE)


def test_relative_prices_are_set_for_every_model_and_kept_when_the_crash_fields_are_filled(prereg):
    priced = P.set_prices(prereg, {"haiku": 1, "sonnet": 3, "opus": 5}, "the list prices")
    assert priced["filled"]["relative_price"] == {
        "prices": {"haiku": 1.0, "sonnet": 3.0, "opus": 5.0}, "source": "the list prices"}
    assert P.check(priced) == []
    filled = P.fill(priced, reports(["boosted_trees"]), engine=ENGINE)
    assert filled["filled"]["relative_price"] == priced["filled"]["relative_price"]
    for bad in ({"haiku": 1, "sonnet": 3}, {"haiku": 1, "sonnet": 0, "opus": 5},
                {"haiku": 1, "sonnet": 3, "opus": 5, "other": 2}):
        with pytest.raises(P.Refused):
            P.set_prices(prereg, bad, "the list prices")


def test_the_committed_file_parses_as_json_with_no_number_but_its_own():
    """Every number the file holds is a JSON number (no NaN or infinity a reader cannot parse)."""
    text = P.PREREG.read_text("utf-8")
    json.loads(text, parse_constant=lambda c: pytest.fail(f"prereg.json holds {c}"))


# ── the staged pre-registration (R2, 2026-10-28) ─────────────────────────────

ESTIMATE, SULLIVAN, GLUCOSE = P.ESTIMATE_FIXTURE, "sullivan2021", "nhanes_glucose_predict"


def only(*datasets: str) -> dict[str, dict]:
    """:func:`reports`' crash reports of ``datasets`` alone (the Predict fixture's default plan
    declaring the ruling's family)."""
    given = reports(["boosted_trees", "elastic_net"])
    return {d: given[d] for d in datasets}


def launched(folder, run: str, dataset: str):
    """A launch file as the launcher writes one (``launcher.launch``): its run and dataset."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{run}.json").write_text(json.dumps({"run": run, "dataset": {"id": dataset},
                                                    "design": 2}), "utf-8")
    return folder


def test_a_dataset_with_its_own_budget_launches_while_another_without_one_is_refused(prereg):
    """Only the Estimate fixture's crash sweep is filled: it may launch (its budget the launcher's
    rule over its own report), and Sullivan, whose sweep is not filled, is refused in words naming
    it; the pilot whole is not filled, and Bang 2009 is no pilot dataset."""
    from research.swarm import launcher

    staged = P.fill(prereg, only(ESTIMATE), engine=ENGINE)
    assert P.ready(staged, ENGINE, dataset=ESTIMATE) is None  # its margins recomputed too
    assert staged["filled"]["fit_budget"] == {
        ESTIMATE: launcher.budget_of({"fit_seconds": {"max": 20.25}}, 11)}
    why = P.ready(staged, ENGINE, dataset=SULLIVAN, margins=False)
    assert why == (f"prereg.json holds no fit budget for {SULLIVAN}: fill it from its own crash "
                   f"sweep (python -m research.swarm.prereg_fill crash …).")
    assert "no fit budget for" in P.ready(staged, ENGINE, margins=False)
    assert "bang2009 is not one of the pilot's datasets" in P.ready(
        staged, ENGINE, dataset="bang2009", margins=False)
    assert staged["filled"]["compute_cap"] is None  # needed only before the sweep step


def test_r0_missing_refuses_the_predict_fixture_only(prereg):
    """With every pilot dataset's crash sweep filled and r0 not yet measured, the Predict fixture
    is refused for its r0 and every other dataset may launch; the pilot whole waits for r0, and
    once r0 is measured nothing stops either."""
    crashed = P.fill(prereg, only(*P.PILOT), engine=ENGINE)
    assert crashed["filled"]["compute_cap"] is not None
    for d in P.PILOT:
        why = P.ready(crashed, ENGINE, dataset=d, margins=False)
        if d == P.PREDICT_FIXTURE:
            assert why.startswith("prereg.json holds no r0: ")
        else:
            assert why is None, d
    assert P.ready(crashed, ENGINE, margins=False).startswith("prereg.json holds no r0")
    filled = measured(prereg)
    assert P.ready(filled, ENGINE, dataset=P.PREDICT_FIXTURE, margins=False) is None
    assert P.ready(filled, ENGINE, margins=False) is None


def test_a_crash_fill_adds_to_what_is_filled_with_when_each_was_filled(prereg):
    """Two fills, each of its datasets: the second keeps the first's entries and when they were
    filled; a fill of a dataset again replaces its own entry alone; r0 stays through a fill of
    another dataset and is cleared by the Predict fixture's; the compute cap is the hand sum once
    the fifth dataset is in (five plans at 100 s on the Estimate fixture, two at 30 s on each of
    the four others)."""
    first = P.fill(prereg, only(ESTIMATE), engine=ENGINE, at="2026-10-28T10:00:00+00:00")
    second = P.fill(first, only(SULLIVAN, GLUCOSE), engine=ENGINE, at="2026-10-30T09:00:00+00:00")
    held = second["filled"]["reports"]
    assert set(held) == {ESTIMATE, SULLIVAN, GLUCOSE}
    assert held[ESTIMATE] == first["filled"]["reports"][ESTIMATE]
    assert held[ESTIMATE]["filled_at"] == "2026-10-28T10:00:00+00:00"
    assert held[SULLIVAN]["filled_at"] == held[GLUCOSE]["filled_at"] == "2026-10-30T09:00:00+00:00"
    assert set(second["filled"]["fit_budget"]) == {ESTIMATE, SULLIVAN, GLUCOSE}
    again = P.fill(second, only(SULLIVAN), engine=ENGINE, at="2026-10-31T09:00:00+00:00")
    assert again["filled"]["reports"][GLUCOSE] == held[GLUCOSE]
    assert again["filled"]["reports"][SULLIVAN]["filled_at"] == "2026-10-31T09:00:00+00:00"
    assert P.check(again) == []
    filled = measured(prereg)
    kept = P.fill(filled, only(SULLIVAN), engine=ENGINE)
    assert kept["filled"]["r0"] == filled["filled"]["r0"] and kept["filled"]["r0"]["filled_at"]
    assert kept["predict_fixture"]["r0"]["value"] == filled["predict_fixture"]["r0"]["value"]
    cleared = P.fill(filled, only(P.PREDICT_FIXTURE), engine=ENGINE)
    assert cleared["filled"]["r0"] is None and cleared["predict_fixture"]["r0"]["value"] is None
    whole = P.fill(again, only(P.PREDICT_FIXTURE, P.MEASUREMENT_FIXTURE), engine=ENGINE)
    assert whole["filled"]["compute_cap"]["seconds"] == 5 * 100.0 + 4 * 2 * 30.0


def test_a_report_on_another_tree_refuses_and_so_does_the_engine_now_on_another(prereg):
    staged = P.fill(prereg, only(ESTIMATE), engine=ENGINE)
    other = {**ENGINE, "engine_tree": "t1"}
    moved = only(SULLIVAN)
    moved[SULLIVAN]["engine_tree"] = "t1"
    with pytest.raises(P.Refused, match="different engine trees: .*on t0.*on t1|on t1.*on t0"):
        P.fill(staged, moved, engine=other)
    why = P.ready(staged, other, dataset=ESTIMATE, margins=False)
    assert "engine now is t1" in why
    assert "with modified files" in P.ready(staged, {**ENGINE, "engine_modified": True},
                                            dataset=ESTIMATE, margins=False)
    assert P.ready(staged, ENGINE, dataset=ESTIMATE, margins=False) is None


def test_a_fill_after_a_graded_run_on_that_dataset_refuses_and_on_another_does_not(prereg,
                                                                                   tmp_path):
    """A launch of Sullivan (a graded run) freezes Sullivan's fill: a crash fill of it is refused,
    naming the run; a fill of the Estimate fixture beside it is not. A launch on the Predict
    fixture refuses its r0 fill the same way; with no launch, r0 is filled."""
    launches = launched(tmp_path / "launches", "run_01-" + "a" * 32, SULLIVAN)
    assert P.graded_runs(SULLIVAN, launches) == ["run_01-" + "a" * 32]
    assert P.graded_runs(ESTIMATE, launches) == []
    with pytest.raises(P.Refused, match=f"{SULLIVAN} has 1 graded run.*run_01-a+.*not filled"):
        P.fill(prereg, only(SULLIVAN), engine=ENGINE, launches=launches)
    with pytest.raises(P.Refused, match=f"{SULLIVAN} has 1 graded run"):
        P.fill(prereg, only(ESTIMATE, SULLIVAN), engine=ENGINE, launches=launches)
    assert ESTIMATE in P.fill(prereg, only(ESTIMATE), engine=ENGINE,
                              launches=launches)["filled"]["reports"]
    crashed = P.fill(prereg, only(*P.PILOT), engine=ENGINE, launches=tmp_path / "none")
    data = default_record()
    launched(launches, "run_02-" + "b" * 32, P.PREDICT_FIXTURE)
    with pytest.raises(P.Refused, match=f"{P.PREDICT_FIXTURE} has 1 graded run"):
        P.fill_r0(crashed, data, default_replay(crashed, data), launches=launches)
    got = P.fill_r0(crashed, data, default_replay(crashed, data), launches=tmp_path / "none")
    assert got["predict_fixture"]["r0"]["value"] is not None


RUN = "run_01-" + "a" * 32
PRICES = {"haiku": 1, "sonnet": 3, "opus": 5}
# A fill refused after a graded run, once §7's unit's minimum (UNIT_MIN, which the unit's part
# alone states, compute()) has changed since prereg.json was written: it names that part and RUN.
FROZEN = (rf"prereg\.json's unit is no longer what the generators give, and the pilot has 1 "
          rf"graded run already \({RUN}\): .*cannot be rewritten")


@pytest.fixture
def once(monkeypatch):
    """The generators' slow analyses that every fill recomputes (b0's tuning run, the regret
    classes, the latent oracle) each computed once in the test and handed back as fresh copies:
    each reads only the generators' constants, which these tests leave as they are, so every fill
    reads the values a fresh computation gives, and a test of many fills stays short."""
    import functools

    for owner, name in ((G, "b0_tuning"), (G, "readmission_regrets"), (P, "measurement_oracle")):
        cached = functools.cache(getattr(owner, name))
        monkeypatch.setattr(owner, name,
                            lambda *a, _c=cached, **k: copy.deepcopy(_c(*a, **k)))


def test_after_a_graded_run_a_fill_goes_through_while_the_rules_are_unchanged(prereg, tmp_path,
                                                                              once):
    """A launch of the Estimate fixture is a graded run. While the generators give what
    prereg.json holds, a crash fill of another dataset, the r0 fill and the prices still go
    through, and the Estimate fixture may launch."""
    crashed = P.fill(prereg, only(*(d for d in P.PILOT if d != SULLIVAN)), engine=ENGINE,
                     launches=tmp_path / "none")
    launches = launched(tmp_path / "launches", RUN, ESTIMATE)
    filled = P.fill(crashed, only(SULLIVAN), engine=ENGINE, launches=launches)
    assert SULLIVAN in filled["filled"]["reports"]
    data = default_record()
    got = P.fill_r0(filled, data, default_replay(filled, data), launches=launches)
    assert got["predict_fixture"]["r0"]["value"] is not None
    priced = P.set_prices(got, PRICES, "the list prices", launches=launches)
    assert priced["filled"]["relative_price"]["prices"] == {m: float(p) for m, p in PRICES.items()}
    assert P.ready(priced, ENGINE, dataset=ESTIMATE, launches=launches) is None


def test_after_a_graded_run_a_changed_rule_refuses_every_write_naming_the_part_and_the_run(
        prereg, tmp_path, monkeypatch, capsys, once):
    """Once the Estimate fixture has a graded run, a rule changed since prereg.json was written
    refuses the crash fill of another dataset, the r0 fill, the prices and the plain write, each
    naming the part that changed and the graded run, and leaves the file as it was; ``ready``
    says the margins cannot be rewritten. With no graded run the changed rule is written again,
    and ``ready`` says to write it before any graded run."""
    crashed = P.fill(prereg, only(*P.PILOT), engine=ENGINE, launches=tmp_path / "none")
    launches = launched(tmp_path / "launches", RUN, ESTIMATE)
    monkeypatch.setattr(P, "UNIT_MIN", P.UNIT_MIN + 1)
    with pytest.raises(P.Refused, match=FROZEN):
        P.fill(crashed, only(SULLIVAN), engine=ENGINE, launches=launches)
    data = default_record()
    with pytest.raises(P.Refused, match=FROZEN):
        P.fill_r0(crashed, data, default_replay(crashed, data), launches=launches)
    with pytest.raises(P.Refused, match=FROZEN):
        P.set_prices(crashed, PRICES, "the list prices", launches=launches)
    assert re.search(FROZEN, P.ready(crashed, ENGINE, dataset=ESTIMATE, launches=launches))
    # The plain write reads the graded runs where the launcher keeps them (here the test's).
    monkeypatch.setattr(P, "GRADED", launches)
    path = P.write(crashed, tmp_path / "prereg.json")
    before = path.read_text("utf-8")
    assert P.main(["--prereg", str(path)]) == 1
    assert re.search(FROZEN, json.loads(capsys.readouterr().out)["refused"])
    assert path.read_text("utf-8") == before
    # The other way: with no graded run the changed rule is written.
    none = tmp_path / "none"
    again = P.fill(crashed, only(SULLIVAN), engine=ENGINE, launches=none)
    assert again["unit"]["minimum"] == P.UNIT_MIN == 3
    assert P.ready(crashed, ENGINE, dataset=ESTIMATE, launches=none) == (
        "prereg.json's unit is no longer what the generators give: write it again before any "
        "graded run.")


def checkouts(tmp_path):
    """Two checkouts of one repository: its main checkout (``git init``, one commit) and a
    worktree added from it (``git worktree add``)."""
    import subprocess

    main, other = tmp_path / "main", tmp_path / "other"
    main.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(main), "-c", "core.hooksPath=/dev/null", "-c",
                        "commit.gpgsign=false", "-c", "user.name=test", "-c",
                        "user.email=research@example.org", *args], check=True,
                       capture_output=True, timeout=60)

    git("init", "-q")
    (main / "README").write_text("x\n", "utf-8")
    git("add", "README")
    git("commit", "-q", "-m", "x")
    git("worktree", "add", "-q", "--detach", str(other))
    return main.resolve(), other.resolve()


def test_a_launch_in_one_checkout_refuses_a_fill_made_in_another_of_the_same_repository(
        prereg, tmp_path, monkeypatch):
    """Every checkout of the repository reads the launches its main checkout keeps: a launch of
    Sullivan written there refuses a crash fill of Sullivan made in a worktree, as one made in the
    main checkout; a dataset with no launch is filled in either."""
    from research.swarm import launcher as L

    main, other = checkouts(tmp_path)
    shared = main / "research" / "corpus" / "launches"
    assert L.launches_of(main) == L.launches_of(other) == shared
    launched(shared, RUN, SULLIVAN)
    monkeypatch.setattr(P, "GRADED", None)  # the launcher's folder, not the tests' empty one
    for checkout in (main, other):
        monkeypatch.setattr(L, "REPO", checkout)
        with pytest.raises(P.Refused, match=f"{SULLIVAN} has 1 graded run.*{RUN}"):
            P.fill(prereg, only(SULLIVAN), engine=ENGINE)
    assert ESTIMATE in P.fill(prereg, only(ESTIMATE), engine=ENGINE)["filled"]["reports"]


def test_with_a_graded_run_and_no_prereg_json_the_plain_write_refuses_and_writes_none(
        tmp_path, monkeypatch, capsys, once):
    """The plain write reads the graded runs where the launcher keeps them (here the test's). With
    a launch of the Estimate fixture and no prereg.json, it is refused, naming the run, and no
    file is written; with an empty launches folder, it writes prereg.json, what the generators
    give with nothing filled."""
    path = tmp_path / "prereg.json"
    monkeypatch.setattr(P, "GRADED", launched(tmp_path / "launches", RUN, ESTIMATE))
    assert P.main(["--prereg", str(path)]) == 1
    assert json.loads(capsys.readouterr().out) == {"refused": (
        f"There is no prereg.json, and the pilot has 1 graded run already ({RUN}): its margins "
        f"and rules are fixed before the first graded run, so none is written now.")}
    assert not path.exists()
    # The other way: with no graded run, one is written.
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(P, "GRADED", tmp_path / "empty")
    assert P.main(["--prereg", str(path)]) == 0
    assert capsys.readouterr().out == f"wrote {path}\n"
    written = P.load(path)
    assert written["filled"] == P.EMPTY_FILLED and P.check(written) == []


def test_the_plain_write_is_refused_where_git_cannot_place_the_launches_and_not_where_it_can(
        tmp_path, monkeypatch, capsys, once):
    """With no launches folder named, the plain write reads the one the main checkout keeps
    (``launcher.launches_of``). In a checkout whose ``.git`` file names no repository, git cannot
    place it: the write is refused in the launcher's words and no file is written. In a worktree
    of a repository whose main checkout holds no launch, prereg.json is written."""
    from research.swarm import launcher as L

    monkeypatch.setattr(P, "GRADED", None)  # the launcher's folder, not the tests' empty one
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / ".git").write_text(f"gitdir: {tmp_path / 'nowhere'}\n", "utf-8")
    monkeypatch.setattr(L, "REPO", broken)
    path = tmp_path / "prereg.json"
    assert P.main(["--prereg", str(path)]) == 1
    assert json.loads(capsys.readouterr().out) == {"refused": (
        f"git cannot tell which checkout is the main one for {broken.resolve()} (its .git is a "
        f"file git does not follow), so the launches every checkout shares cannot be found.")}
    assert not path.exists()
    # The other way: a worktree whose main checkout git places, with no launch there.
    main, other = checkouts(tmp_path)
    monkeypatch.setattr(L, "REPO", other)
    assert not L.launches_of(other).exists()
    assert P.main(["--prereg", str(path)]) == 0
    assert capsys.readouterr().out == f"wrote {path}\n" and P.check(P.load(path)) == []


def test_the_commands_fill_one_dataset_and_say_whether_it_may_launch(prereg, tmp_path, capsys,
                                                                     monkeypatch):
    """``prereg_fill crash`` takes one dataset's report, and ``ready --dataset`` says that dataset
    may launch and another may not, in ``ready``'s words."""
    from research.swarm import launcher

    monkeypatch.setattr(launcher, "engine_now", lambda: dict(ENGINE))
    path = P.write(prereg, tmp_path / "prereg.json")
    (tmp_path / "crash.json").write_text(json.dumps(only(ESTIMATE)[ESTIMATE]), "utf-8")
    assert P.main(["--prereg", str(path), "crash", str(tmp_path / "crash.json")]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["datasets"] == [ESTIMATE] and printed["ready"][ESTIMATE] is None
    assert printed["ready"][SULLIVAN].startswith("prereg.json holds no fit budget for")
    written = P.load(path)
    assert written["filled"]["reports"][ESTIMATE]["sha256"] == P._sha256(tmp_path / "crash.json")
    assert P.main(["--prereg", str(path), "ready", "--dataset", ESTIMATE]) == 0
    assert json.loads(capsys.readouterr().out) == {"dataset": ESTIMATE, "ready": True,
                                                   "why": None}
    assert P.main(["--prereg", str(path), "ready", "--dataset", SULLIVAN]) == 1
    said = json.loads(capsys.readouterr().out)
    assert said["why"] == P.ready(written, ENGINE, dataset=SULLIVAN, margins=False)
