"""The swarm's graded fixtures (``research/swarm/generators.py``; SWARM_PLAN §5 items 4–6, §7): each
generator's planted truth, by construction, by an independent library or by a derivation checked
against a simulation of the generator itself.

* The hardened Predict fixture draws its columns exactly as ``make_fixtures.clinical_risk`` does
  (the same stream gives the same columns), its linear part is that fixture's model (the intake's
  own implementation of it), and its outcome is a Bernoulli draw of the true risk from a stream of
  its own. Its regret ladder (additive lines, additive splines, splines with the interaction) is
  strictly decreasing, so a path's choices change its regret, and the lines' regret is reproduced
  by statsmodels' maximum likelihood, an independent fit.
* The Estimate fixture's closed-form population coefficients agree with statsmodels' least squares
  on 400,000 rows of the generator within four standard errors, plan by plan; the oracle's is the
  planted effect; b0 is half the smallest planted bias, and every wrong path's bias is at least
  twice it. The tuning run over 100 fresh draws shows b0 at least four standard errors of the
  oracle's mean bias, the oracle within b0, and every wrong path beyond it.
* The measurement fixture's layout meets the engine's QC-RLSC requirements (its MIN_QC and MAX_GAP),
  its blanks are exactly the values below the detection limit, its batch offsets go up and down in
  run order for every feature, its drift starts again in each batch, and its true concentrations
  regressed on case status (statsmodels, pooled over draws) give back the planted log2 fold
  changes; its graded decisions are the engine's menus.
* Each committed table is its generator's bytes at its seed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.swarm import generators as G  # noqa: E402

DATASETS = REPO / "research" / "datasets"


def answers(id_: str) -> dict:
    data = json.loads((DATASETS / id_ / "answers.json").read_text("utf-8"))
    return {t["name"]: t["value"] for t in data["truths"]}


@pytest.mark.parametrize("name", list(G.TABLES))
def test_each_committed_table_is_its_generators_bytes_and_a_seed_draws_its_own(name):
    assert (G.FIXTURES / name).read_bytes() == G.table_bytes(name)
    seed = G.TABLES[name]["seed"]
    assert G.table_bytes(name, seed + 1) != G.table_bytes(name, seed)


# ── the hardened Predict fixture ─────────────────────────────────────────────


def test_the_readmission_columns_are_clinical_risks_draws():
    """``make_fixtures.clinical_risk`` seeds default_rng(20260802) and draws its columns in this
    order before the outcome: the same stream gives the same columns."""
    from turbotab.sample_data import make_fixtures

    ours = G.readmission_columns(np.random.default_rng(20260802), 480)
    theirs = make_fixtures.clinical_risk()
    for column in ours.columns:
        if column == "encounter_id":
            continue
        pd.testing.assert_series_equal(ours[column], theirs[column], check_names=False)


def test_the_readmission_outcome_is_a_draw_of_the_true_risk_from_its_own_stream():
    frame = G.readmission(11, 3000)
    _, outcome = G.streams(11, 2)
    risk = G.readmission_risk(frame)
    assert np.array_equal(frame[G.READMISSION_OUTCOME].to_numpy(),
                          (outcome.random(3000) < risk).astype(int))


def test_the_linear_part_is_clinical_risks_model_and_the_rest_is_the_u_shape_and_interaction():
    from research.swarm import intake

    frame = G.readmission(12, 500)
    linear = G.readmission_log_odds(frame, linear_only=True)
    assert np.allclose(linear, intake.linear_predictor(intake.CLINICAL_RISK_MODEL, frame),
                       rtol=0, atol=1e-12)
    na = frame["sodium_mmol_l"].to_numpy(dtype=float)
    age, alb = frame["age"].to_numpy(dtype=float), frame["albumin_g_dl"].to_numpy(dtype=float)
    by_hand = linear + 0.5 * (((na - 138) / 4.0) ** 2 - 1) - 0.4 * ((age - 68) / 13) * (
        (alb - 3.6) / 0.45)
    assert np.allclose(G.readmission_log_odds(frame), by_hand, rtol=0, atol=1e-12)


@pytest.fixture(scope="module")
def regrets():
    return G.readmission_regrets()


def test_the_regret_ladder_falls_with_each_term_a_path_can_add(regrets):
    """Additive lines (the default path's class) miss the U-shape and the interaction; splines
    catch the U-shape; the interaction leaves only the fit's own noise."""
    assert regrets["linear"] > 4 * regrets["spline"] > 0
    assert regrets["spline"] > 50 * regrets["interaction"]
    committed = answers("inpatient_readmission_predict")
    ladder = next(v for k, v in committed.items() if k.startswith("each model class's"))
    assert ladder["oracle"] == 0.0
    for kind in ("linear", "spline", "interaction"):
        assert ladder[kind] == pytest.approx(regrets[kind], rel=1e-6, abs=1e-9)


def test_the_lines_regret_is_statsmodels_maximum_likelihoods(regrets):
    """The same fit by statsmodels' Logit (Newton), an independent implementation."""
    import statsmodels.api as sm

    fit = G.readmission(G.REGRET_SEEDS[0], G.REGRET_ROWS[0])
    read = G.readmission(G.REGRET_SEEDS[1], G.REGRET_ROWS[1])

    def design(frame):
        X = frame.drop(columns=["encounter_id", G.READMISSION_OUTCOME]).copy()
        X["sex"] = (X["sex"] == "M").astype(float)
        return sm.add_constant(X.astype(float))

    model = sm.Logit(fit[G.READMISSION_OUTCOME].to_numpy(), design(fit)).fit(disp=0)
    p_hat = np.asarray(model.predict(design(read)))
    ours = float(np.mean((p_hat - G.readmission_risk(read)) ** 2))
    assert ours == pytest.approx(regrets["linear"], rel=1e-3)


# ── the Estimate fixture ─────────────────────────────────────────────────────


def test_the_estimate_fixture_is_its_structural_equations_drawn_from_five_streams():
    m = G.ESTIMATE_MODEL
    n = 2000
    frame = G.blood_pressure(13, n)
    people, exposure, noise, concern, blanks = G.streams(13, 5)
    age = people.integers(40, 80, n)
    pain = (people.random(n) < 0.35).astype(int)
    height = np.round(people.normal(170.0, 9.0, n), 1)
    walks = (exposure.random(n) < 1 / (1 + np.exp(-(0.2 - 1.4 * pain)))).astype(int)
    sbp = np.round(118 - 5 * walks + 8 * pain + 0.5 * (age - 60) + noise.normal(0, 12, n), 1)
    score = np.round(50 + 6 * walks + 0.6 * (sbp - 118) + concern.normal(0, 8, n), 1)
    seen = blanks.standard_normal(n) < 0.6 - 1.0 * (sbp - 118) / 12
    assert np.array_equal(frame["age"].to_numpy(), age)
    assert np.array_equal(frame["walks_daily"].to_numpy(), walks)
    assert np.array_equal(frame["height_cm"].to_numpy(), height)
    assert np.array_equal(frame["sbp_mmhg"].to_numpy(), sbp)
    assert np.array_equal(frame["health_concern_score"].to_numpy(), score)
    assert np.array_equal(frame["chronic_pain"].isna().to_numpy(), ~seen)
    assert np.array_equal(frame.loc[seen, "chronic_pain"].to_numpy(dtype=int), pain[seen])
    assert m["effect"] == -5.0 and m["exposure"] == "walks_daily"


def test_the_closed_form_population_coefficients_are_least_squares_on_the_generator():
    """Each plan's coefficient on walks_daily by statsmodels' OLS on 400,000 rows of the generator
    (the confounder's true values from the same stream where the plan reads every row), within
    four of its standard errors of the closed form."""
    import statsmodels.api as sm

    n, seed = 400_000, 14
    frame = G.blood_pressure(seed, n)
    people = G.streams(seed, 5)[0]
    people.integers(40, 80, n)
    full = frame.assign(chronic_pain=(people.random(n) < 0.35).astype(int))
    closed = G.population_effects()
    for name, plan in G.ESTIMATE_PLANS.items():
        rows = frame.dropna(subset=["chronic_pain"]) if plan["rows"] == "complete" else full
        fit = sm.OLS(rows["sbp_mmhg"].astype(float),
                     sm.add_constant(rows[plan["columns"]].astype(float))).fit()
        assert abs(fit.params["walks_daily"] - closed[name]) < 4 * fit.bse["walks_daily"], name


def test_b0_is_half_the_smallest_planted_bias_and_the_oracle_has_none():
    p = G.planted_biases()
    assert p["coefficients"]["oracle"] == pytest.approx(-5.0, abs=1e-9)
    sizes = [abs(p["bias"][w]) for w in G.WRONG_PATHS]
    assert p["b0"] == min(sizes) / 2 and all(s >= 2 * p["b0"] for s in sizes)
    assert p["b0"] > 0.5  # mmHg: the plan's margin is not a rounding error


def test_b0_is_at_least_four_standard_errors_of_the_oracles_mean_bias_over_100_draws():
    """SWARM_PLAN §5 item 5, by the tuning run itself: the claims, not its digits (a solver may
    move the last of them on another platform)."""
    run = G.b0_tuning()
    assert run["draws"] == 100 and run["holds"] is True
    assert run["b0"] >= G.B0_SE_RATIO * run["plans"]["oracle"]["se_of_mean_bias"]
    assert run["oracle_within_b0"] is True
    assert all(run["wrong_paths_beyond_b0"].values())
    # The oracle's 95% intervals cover in the binomial band at 100 draws (2.5% each side).
    assert 0.90 <= run["oracle_coverage"] <= 0.99
    committed = answers("walking_bp_estimate")
    tuned = next(v for k, v in committed.items() if k.startswith("b0 against"))
    assert tuned["holds"] is True and tuned["b0"] == pytest.approx(run["b0"], rel=1e-12)


def test_the_tuning_condition_fails_where_the_margin_is_too_small():
    """The condition is not silent by construction: at 150 rows a draw (an eighth of the
    fixture's) the oracle's spread is about √8 times wider, and b0 is no longer four of its
    standard errors."""
    run = G.b0_tuning(draws=100, m=5, rows=150)
    assert run["holds"] is False
    assert run["b0_over_oracle_se"] < G.B0_SE_RATIO


# ── the measurement fixture ──────────────────────────────────────────────────


def test_the_panel_layout_meets_the_engines_qc_rlsc_requirements():
    from turbotab.core.methods.qc_drift import MAX_GAP, MIN_QC

    lay = G.panel_layout()
    for _, batch in lay.groupby("batch"):
        qc = batch.loc[batch["qc"], "place"].to_numpy()
        assert len(qc) >= MIN_QC
        assert qc[0] == batch["place"].min() and qc[-1] == batch["place"].max()
        assert np.diff(qc).max() <= MAX_GAP


def test_the_panels_blanks_are_exactly_the_values_below_the_detection_limit():
    lat = G.panel_latent(15)
    frame = G.metabolite_panel(15)
    names = G.panel_design()["names"]
    below = lat["measured_log2"] < G.PANEL["detection_limit_log2"]
    assert np.array_equal(frame[names].isna().to_numpy(), below)
    assert below.any() and not below.all()
    qc = frame["sample_type"] == "pooled_qc"
    assert frame.loc[qc, G.PANEL_OUTCOME].isna().all()
    assert frame.loc[~qc, G.PANEL_OUTCOME].notna().all()


def test_the_batch_offsets_go_up_and_down_in_run_order_and_the_drift_starts_again():
    d = G.panel_design()
    steps = np.sign(np.diff(d["batch_offset"], axis=1))
    assert (np.abs(np.diff(steps, axis=1)) > 0).all()  # every feature turns in run order
    assert (d["batch_offset"][:, 0] > 0).any() and (d["batch_offset"][:, 0] < 0).any()
    lat = G.panel_latent(16)
    lay = lat["layout"]
    for _, batch in lay.groupby("batch"):
        rows = batch.index.to_numpy()
        assert np.allclose(lat["drift_log2"][rows], lat["drift_log2"][rows[:len(batch)]])
    first = lay.index[lay["batch"] == 1].to_numpy()
    second = lay.index[lay["batch"] == 2].to_numpy()
    assert np.allclose(lat["drift_log2"][first], lat["drift_log2"][second])


def test_the_true_concentrations_give_back_the_planted_fold_changes():
    """statsmodels' OLS of each feature's true log2 concentration on case status, pooled over 30
    draws' participants: every coefficient within four standard errors of its planted fold change
    (0 for the features outside the associated set)."""
    import statsmodels.api as sm

    truth = G.panel_truth()
    names = G.panel_design()["names"]
    ys, cases = [], []
    for seed in range(30):
        lat = G.panel_latent(1000 + seed)
        keep = ~lat["layout"]["qc"].to_numpy()
        ys.append(lat["true_log2"][keep])
        cases.append(lat["case"][keep])
    Y, x = np.vstack(ys), np.concatenate(cases)
    X = sm.add_constant(x.astype(float))
    for j, name in enumerate(names):
        fit = sm.OLS(Y[:, j], X).fit()
        planted = truth["log2_fold_change"].get(name, 0.0)
        assert abs(fit.params[1] - planted) < 4 * fit.bse[1], name
    assert len(truth["associated"]) == G.PANEL["associated"] == len(truth["log2_fold_change"])


def test_the_graded_measurement_decisions_are_the_engines_menus():
    from typing import get_args

    from turbotab.core import decisions
    from turbotab.core.methods import omics, qc_drift

    graded = {d["records"]: d["options"] for d in G.graded_measurement_decisions()}
    assert graded == {
        "apply_repair on the omics_scale finding": list(omics.INTENSITY_METHODS),
        "set_batch": list(get_args(decisions.BatchMethod)),
        f"apply_repair on {qc_drift.FINDING}": list(qc_drift.OPTIONS),
        "set_missing below_detection": list(get_args(decisions.BelowDetection)),
    }


# ── where package 6b's prereg reads them: the datasets' answers and manifests ─

NEW = {"inpatient_readmission_predict": G.READMISSION_FILE,
       "walking_bp_estimate": G.ESTIMATE_FILE, "plasma_panel_estimate": G.PANEL_FILE}


@pytest.mark.parametrize("id_", list(NEW))
def test_each_manifest_names_the_generator_and_seed_that_write_its_table(id_):
    import hashlib

    manifest = json.loads((DATASETS / id_ / "manifest.json").read_text("utf-8"))
    gen = manifest["generator"]
    assert gen["module"] == "research/swarm/generators.py"
    frame = getattr(G, gen["function"])(gen["seed"])
    data = frame.to_csv(index=False).encode()
    assert len(frame) == gen["rows"] == manifest["table"]["rows"]
    assert hashlib.sha256(data).hexdigest() == manifest["table"]["sha256"] \
        == manifest["files"][0]["sha256"]
    assert manifest["files"][0]["file"] == f"research/swarm/fixtures/{NEW[id_]}"


def test_the_answers_carry_each_generators_planted_values():
    readmit = answers("inpatient_readmission_predict")
    assert readmit["the outcome's generating model"] == json.loads(json.dumps(G.READMISSION_MODEL))
    frame = pd.read_csv(G.FIXTURES / G.READMISSION_FILE)
    risk = G.readmission_risk(frame)
    y = frame[G.READMISSION_OUTCOME].to_numpy()
    # The C-statistic as a Mann-Whitney count over every event-nonevent pair (ties count half).
    pos, neg = risk[y == 1], risk[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    c = readmit["the C-statistic of ranking every row by its true risk"]
    assert c == pytest.approx(wins / (len(pos) * len(neg)), abs=1e-12)

    walk = answers("walking_bp_estimate")
    planted = next(v for k, v in walk.items() if k.startswith("each plan's population"))
    assert planted["b0"] == pytest.approx(G.planted_biases()["b0"], rel=1e-12)
    assert walk["the effect of walks_daily on sbp_mmhg, a difference in means"] == -5.0

    panel = answers("plasma_panel_estimate")
    truth = next(v for k, v in panel.items() if k.startswith("the associated features"))
    assert truth == json.loads(json.dumps(G.panel_truth()))
    graded = panel["the measurement decisions graded: the engine's own menus"]
    assert graded == G.graded_measurement_decisions()
