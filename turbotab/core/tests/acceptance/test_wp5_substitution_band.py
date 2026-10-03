"""WP5 · Substitution band and support (docs/turbotab-next/audit/AUDIT_REPORT.md §5).

Closes MA-12 (the substitution curve's "95%" band is the wrong width), MA-13 ("within the observed
range" checks each moved column, not the diet the swap creates), and the minors B16 (a band drawn
from the refits that succeeded but labeled with the number attempted) and B17 (the rows each point
of the curve averages over change with k). One test per acceptance test in §5, numbered as there.

Every reference is computed by a path independent of the code under test: statsmodels' OLS
covariance of the substitution contrast (the analytic half-width), the data-generating
coefficients (the truth a band must cover), statsmodels' exact t interval on the same datasets,
and shares of energy counted by hand with numpy from the fixture's own columns.

Fixtures are the audit's (``repro.tar.gz``): ``A/r4_band.py`` (the auditor's) and
``A-skeptic/s5_band.py`` (the skeptic's) for the band; ``B/exp5_support.py`` and
``B-skeptic/s10_support.py`` for the support.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from turbotab.core.decisions import SubstitutionSpec
from turbotab.core.methods import substitution as methods
from turbotab.core.methods.substitution import (
    MIN_REFIT_SHARE,
    PERCENTILE_MIN_REFITS,
    Shift,
    refit_band,
    substitution_curve,
)
from turbotab.core.models.artifacts import SubstitutionArtifact
from turbotab.core.stages import modeling
from turbotab.core.stages.modeling import (
    BAND_BOOT,
    BAND_ROWS,
    design_stage,
    fit_stage,
    substitution_stage,
)
from turbotab.core.tests import modeling_fixtures as mf

REPO = Path(__file__).resolve().parents[4]
RESULTS_TSX = REPO / "turbotab" / "frontend" / "src" / "components" / "stage" / "results" / "Results.tsx"
KPU = {"fat_g": 9.0, "carb_g": 4.0}


# ── fixtures ─────────────────────────────────────────────────────────────────


def _auditor(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """A/r4_band.py: fat 0.03 per g, carbohydrate none, energy 0.001 per kcal; error SD 3."""
    energy = rng.normal(2000, 400, n).clip(800, None)
    fat = (energy * 0.35 / 9) * rng.lognormal(0, 0.25, n)
    carb = (energy * 0.50 / 4) * rng.lognormal(0, 0.2, n)
    protein = (energy * 0.15 / 4) * rng.lognormal(0, 0.2, n)
    frame = pd.DataFrame({"fat_g": fat, "carb_g": carb, "protein_g": protein, "energy_kcal": energy})
    frame["y"] = 5 + 0.03 * fat + 0.0 * carb + 0.001 * energy + rng.normal(0, 3, n)
    return frame


def _skeptic(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """A-skeptic/s5_band.py: fat 0.02 per g, carbohydrate 0.01 per g, energy 0.002; error SD 4."""
    energy = rng.normal(2100, 450, n).clip(900, None)
    fat = energy * 0.33 / 9 * rng.lognormal(0, 0.3, n)
    carb = energy * 0.5 / 4 * rng.lognormal(0, 0.25, n)
    frame = pd.DataFrame({"fat_g": fat, "carb_g": carb, "energy_kcal": energy})
    frame["y"] = 3 + 0.02 * fat + 0.01 * carb + 0.002 * energy + rng.normal(0, 4, n)
    return frame


BAND_FIXTURES = {"auditor": _auditor, "skeptic": _skeptic}
# The data-generating change in y when 100 kcal move from fat to carbohydrate.
TRUTH = {"auditor": 100 * (0.0 / 4 - 0.03 / 9), "skeptic": 100 * (0.01 / 4 - 0.02 / 9)}


def _support_auditor() -> tuple[pd.DataFrame, str]:
    """B/exp5_support.py: n = 5,000; fat and protein shares of energy drawn, carbohydrate the rest."""
    rng = np.random.default_rng(3)
    n = 5000
    energy = np.exp(rng.normal(np.log(2000), 0.3, n))
    fat_sh = np.clip(rng.normal(0.34, 0.05, n), 0.15, 0.55)
    prot_sh = np.clip(rng.normal(0.16, 0.03, n), 0.08, 0.3)
    carb_sh = 1 - fat_sh - prot_sh
    return pd.DataFrame({"fat_g": fat_sh * energy / 9, "carb_g": carb_sh * energy / 4,
                         "protein_g": prot_sh * energy / 4, "energy_kcal": energy}), "carb_g"


def _support_skeptic() -> tuple[pd.DataFrame, str]:
    """B-skeptic/s10_support.py: n = 5,000; fat share drawn, protein 16% of energy."""
    rng = np.random.default_rng(8)
    n = 5000
    energy = np.exp(rng.normal(np.log(2100), 0.35, n))
    fs = np.clip(rng.normal(0.34, 0.05, n), 0.18, 0.5)
    cs = 1 - fs - 0.16
    return pd.DataFrame({"fat_g": fs * energy / 9, "carbohydrate_g": cs * energy / 4,
                         "protein_g": 0.16 * energy / 4, "energy_kcal": energy}), "carbohydrate_g"


SUPPORT_FIXTURES = {"auditor": _support_auditor, "skeptic": _support_skeptic}


def _contrast(columns: list[str], recipient: str = "carb_g") -> np.ndarray:
    """100 kcal from fat to the recipient, as a contrast on [const, *columns] coefficients."""
    c = np.zeros(len(columns) + 1)
    c[1 + columns.index("fat_g")] = -100 / 9
    c[1 + columns.index(recipient)] = 100 / 4
    return c


def _analytic(frame: pd.DataFrame) -> tuple[float, float, float]:
    """(estimate, standard error, residual df) of the 100-kcal contrast from statsmodels' OLS."""
    X = frame.drop(columns="y")
    result = sm.OLS(frame["y"].to_numpy(), sm.add_constant(X)).fit()
    c = _contrast(list(X.columns))
    se = float(np.sqrt(c @ result.cov_params().to_numpy() @ c))
    return float(c @ result.params.to_numpy()), se, float(result.df_resid)


def _ols(Xb: pd.DataFrame, yb: np.ndarray):
    """Least squares with an intercept (the linear family's estimator), as a refit for the band."""
    A = np.column_stack([np.ones(len(Xb)), Xb.to_numpy(dtype=float)])
    beta = np.linalg.lstsq(A, yb, rcond=None)[0]
    return lambda frame: beta[0] + frame.to_numpy(dtype=float) @ beta[1:]


def _stage(frame: pd.DataFrame, folder: Path, *, n_boot: int, recipient: str = "carb_g",
           models: tuple[str, ...] = ("linear",)) -> dict:
    """Ingest ``frame`` and run design, fit and substitution (fat_g -> recipient), every row training."""
    paths = mf.ingest_frame(frame, folder)
    roles = {c: "exposure" for c in frame.columns if c.endswith("_g")}
    roles["energy_kcal"] = "energy"
    st = mf.state(roles=roles, target="y", models=list(models),
                  substitution=SubstitutionSpec(donor="fat_g", recipient=recipient, step_kcal=100.0,
                                                n_boot=n_boot),
                  column_units=mf.grams(*[c for c in frame.columns if c.endswith("_g")]))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    ti = mf.target_info("regression", "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    sub = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
    SubstitutionArtifact.model_validate(sub)
    return sub


def _by_hand(X: pd.DataFrame, recipient: str, k: float) -> tuple[np.ndarray, np.ndarray]:
    """(the amount check, the share-of-energy check) at k, counted with numpy alone."""
    energy = X["energy_kcal"].to_numpy()
    fat, carb = X["fat_g"].to_numpy(), X[recipient].to_numpy()
    new_fat, new_carb = fat - k / 9, carb + k / 4
    amount = ((new_fat >= fat.min()) & (new_fat <= fat.max()) & (new_fat >= 0)
              & (new_carb >= carb.min()) & (new_carb <= carb.max()) & (new_carb >= 0))
    fat_share, carb_share = 9 * fat / energy, 4 * carb / energy
    new_fat_share, new_carb_share = 9 * new_fat / energy, 4 * new_carb / energy
    share = ((new_fat_share >= fat_share.min()) & (new_fat_share <= fat_share.max())
             & (new_carb_share >= carb_share.min()) & (new_carb_share <= carb_share.max()))
    return amount, share


# ── 1 · the band's width at N = 10,000 ───────────────────────────────────────


@pytest.mark.parametrize("band_rows", [BAND_ROWS, 2_000], ids=["every-row", "2000-rows-rescaled"])
@pytest.mark.parametrize("fixture", ["auditor", "skeptic"])
def test_1_at_n_10000_the_band_half_width_is_within_15_percent_of_the_analytic_one(
        fixture, band_rows, tmp_path, monkeypatch):
    """§5 WP5 test 1: "Linear fat → carbohydrate fixture at N = 10,000: band half-width within 15%
    of the analytic 0.042–0.044 (today 0.087–0.106)."

    Reference: 1.96 × the OLS standard error of 100·(β_carb/4 − β_fat/9) from statsmodels' own
    covariance matrix, on the same 10,000 rows: 0.0415 (auditor's fixture) and 0.0442 (skeptic's),
    the audit's 0.042–0.044. Measured: the band the stage draws by default (``BAND_BOOT`` refits)
    at k = 100, through ingest, design, fit and the substitution stage.

    Two paths: every training row resampled (N ≤ ``BAND_ROWS``, the ordinary bootstrap), and
    the defect's own setting, refits on 2,000 of the 10,000 rows, now rescaled by √(2,000/10,000)
    (an m-out-of-n bootstrap). Without that rescaling the 2,000-row spread is √5 = 2.24 times
    the analytic width, which the test also checks, so the 15% bound discriminates.

    Monte Carlo error: with 200 refits the standard error's relative error is about
    1/√(2·199) = 5%, so the 15% bound is three of them.
    """
    frame = BAND_FIXTURES[fixture](10_000, np.random.default_rng(42))
    estimate, se, _ = _analytic(frame)
    analytic = 1.959963984540054 * se
    assert 0.040 < analytic < 0.046  # the audit's 0.042-0.044, reproduced
    monkeypatch.setattr(modeling, "BAND_ROWS", band_rows)
    sub = _stage(frame, tmp_path, n_boot=BAND_BOOT)
    curve = sub["models"][0]
    i = sub["ks"].index(100.0)
    assert curve["delta"][i] == pytest.approx(estimate, rel=1e-6)  # the curve is the OLS contrast
    half_width = (curve["ci_high"][i] - curve["ci_low"][i]) / 2
    assert abs(half_width / analytic - 1) <= 0.15, (half_width, analytic)
    band = sub["band"]
    drawn = min(10_000, band_rows)
    assert band["n_rows"] == 10_000 and band["n_units"] == 10_000
    assert band["resample_units"] == drawn
    assert band["scale"] == pytest.approx(math.sqrt(drawn / 10_000))
    if drawn < 10_000:  # the defect: the same refits, not rescaled, are about √5 times too wide
        assert half_width / band["scale"] / analytic > 1.8


# ── 2 · coverage at N = 1,500 ────────────────────────────────────────────────


def test_2_at_n_1500_the_band_covers_the_truth_in_at_least_93_percent_of_300_datasets():
    """§5 WP5 test 2: "At N = 1,500 the band's coverage over 300 datasets is ≥ 0.93 (today
    0.89–0.91 at B = 50)."

    The skeptic's fixture and datasets (A-skeptic/s5_band.py: N = 1,500, seeds 5000..5299). The
    band is the stage's default: ``BAND_BOOT`` refits of the linear model on resamples of every
    row, the interval the stage chooses for that many, placed around the curve. The truth is the
    data-generating contrast, 100·(0.01/4 − 0.02/9) = 0.0278.

    Monte Carlo error: over 300 datasets a coverage near 0.95 has a standard error of 0.013, so
    the 0.93 bound sits 1.5 of them below nominal and even an exact interval misses it on some
    sets of 300 datasets. (On the auditor's fixture, seeds 5000..5299, statsmodels' exact t
    interval covers 0.920; over 600 fresh datasets the band and the exact interval both cover
    0.943.) So two more checks go with the bound. Paired: on the same 300 datasets the band's
    coverage is within 0.02 of the exact t interval's, which removes the datasets' own luck.
    Positive control: the band as it was (50 refits, the percentile interval) misses 0.93 on the
    same datasets, so the bound discriminates.
    """
    from turbotab.core.stages.modeling import BAND_ROWS as rows_cap

    truth = TRUTH["skeptic"]
    interval = "percentile" if BAND_BOOT >= PERCENTILE_MIN_REFITS else "normal"  # as the stage
    n = 300
    band_hits = exact_hits = before_hits = 0
    for s in range(n):
        frame = _skeptic(1500, np.random.default_rng(5000 + s))
        estimate, se, df = _analytic(frame)
        exact_hits += abs(estimate - truth) <= stats.t.ppf(0.975, df) * se
        X, y = frame.drop(columns="y"), frame["y"].to_numpy()
        predict = _ols(X, y)
        curve = substitution_curve(predict, X, donor="fat_g", recipient="carb_g", kcal_per_unit=KPU,
                                   ks=[0.0, 100.0], total="energy_kcal")
        shift = Shift(X, donor="fat_g", recipient="carb_g", kcal_per_unit=KPU, total="energy_kcal")
        band = refit_band(_ols, X, y, shift=shift, ks=[0.0, 100.0], live=curve["live"],
                          n_boot=BAND_BOOT, center=curve["delta"], max_rows=rows_cap,
                          interval=interval, random_state=0)
        band_hits += band["ci_low"][1] <= truth <= band["ci_high"][1]
        before = refit_band(_ols, X, y, shift=shift, ks=[0.0, 100.0], live=curve["live"],
                            n_boot=50, center=curve["delta"], interval="percentile", random_state=0)
        before_hits += before["ci_low"][1] <= truth <= before["ci_high"][1]
    coverage, exact, pre_fix = band_hits / n, exact_hits / n, before_hits / n
    assert coverage >= 0.93, (coverage, exact, pre_fix)
    assert coverage >= exact - 0.02, (coverage, exact)
    assert pre_fix < 0.93, pre_fix


# ── 3 · what the saved caption says, and when a band is not drawn ────────────


def test_3_the_saved_caption_states_rows_refits_and_successes(tmp_path):
    """§5 WP5 test 3, first half: "The saved caption states rows used, B and the number of
    successful refits."

    The server composes the caption (``band.caption``) and the saved figure carries it verbatim.
    Source check, ``turbotab/frontend/src/components/stage/results/Results.tsx``, the SaveMenu's
    figure: ``caption: `${sub.estimand ?? ""} ${hasBand(sub) ? (sub.band?.caption ?? ...) : ""}
    Hatched: where the models disagree.```.
    """
    frame = _skeptic(1500, np.random.default_rng(11))
    sub = _stage(frame, tmp_path, n_boot=30, models=("linear", "elastic_net"))
    band = sub["band"]
    assert {k: band[k] for k in ("n_boot", "n_rows", "n_units", "resample_units", "failed",
                                 "interval", "min_ok_share")} == {
        "n_boot": 30, "n_rows": 1500, "n_units": 1500, "resample_units": 1500, "failed": 0,
        "interval": "normal", "min_ok_share": MIN_REFIT_SHARE}
    assert band["scale"] == 1.0 and band["level"] == 0.95
    caption = band["caption"]
    assert caption.startswith("Shaded bands: 95% intervals, each curve ± 1.96 bootstrap standard "
                              "errors from 30 refits of its model, on bootstrap resamples of all "
                              "1,500 training rows.")
    assert "Refits that succeeded: Linear model 30 of 30, Elastic net 30 of 30." in caption
    assert [m["band_ok"] for m in sub["models"]] == [30, 30]
    assert "The shaded bands are 95% intervals" in sub["note"]
    source = RESULTS_TSX.read_text("utf-8")
    saved = re.search(r"curvesFigure\(sub, target, \{(.*?)\}\)", source, re.S)
    assert saved and "caption:" in saved.group(1) and "sub.band?.caption" in saved.group(1)


def test_3_a_band_needs_a_stated_minimum_share_of_successful_refits(tmp_path, monkeypatch):
    """§5 WP5 test 3, second half: "a band needs a stated minimum share of successful refits"
    (B16: a band was drawn from 2 of 50 refits and labeled with the 50).

    The minimum is ``MIN_REFIT_SHARE`` = 90% (a practitioner convention, stated as one). Failed
    refits are injected by count, so the share is known exactly: 45 of 50 succeed and the band is
    drawn; 44 of 50 and it is not, and the refusal says how many succeeded. Through the stage,
    the caption names the family that has no band and why.
    """
    frame = _skeptic(400, np.random.default_rng(3))
    X, y = frame.drop(columns="y"), frame["y"].to_numpy()
    curve = substitution_curve(_ols(X, y), X, donor="fat_g", recipient="carb_g", kcal_per_unit=KPU,
                               ks=[0.0, 100.0], total="energy_kcal")
    shift = Shift(X, donor="fat_g", recipient="carb_g", kcal_per_unit=KPU, total="energy_kcal")

    def failing(n_fail: int):
        calls = {"n": 0}

        def fit(Xb, yb):
            calls["n"] += 1
            if calls["n"] <= n_fail:
                raise ValueError("a resample held one class")
            return _ols(Xb, yb)
        return fit

    common = dict(shift=shift, ks=[0.0, 100.0], live=curve["live"], n_boot=50, center=curve["delta"])
    drawn = refit_band(failing(5), X, y, **common)
    assert drawn["n_ok"] == 45 and drawn["failed"] == 5 and drawn["refused"] is None
    assert drawn["ci_low"][1] < curve["delta"][1] < drawn["ci_high"][1]
    refused = refit_band(failing(6), X, y, **common)
    assert refused["n_ok"] == 44 and refused["ci_low"] == [None, None]
    assert refused["refused"] == "44 of 50 refits succeeded; a band needs at least 90% of them"

    real = methods.refit_band

    def one_in_five_fails(fit, *args, **kwargs):  # every fifth refit of the stage's band fails
        calls = {"n": 0}

        def flaky(Xb, yb):
            calls["n"] += 1
            if calls["n"] % 5 == 0:
                raise ValueError("a resample held one class")
            return fit(Xb, yb)
        return real(flaky, *args, **kwargs)

    monkeypatch.setattr(methods, "refit_band", one_in_five_fails)
    sub = _stage(frame, tmp_path, n_boot=30)
    model = sub["models"][0]
    assert model["band_ok"] == 24 and all(v is None for v in model["ci_low"] + model["ci_high"])
    assert sub["band"]["failed"] == 6
    assert "Refits that succeeded: Linear model 24 of 30." in sub["band"]["caption"]
    assert sub["band"]["caption"].endswith(
        "No band for Linear model: a band needs at least 90% of its refits to succeed.")


# ── 4 · support: the diet the swap creates ──────────────────────────────────


@pytest.mark.parametrize("fixture", ["auditor", "skeptic"])
def test_4_at_300_kcal_rows_whose_shifted_share_of_energy_leaves_its_range_are_excluded(fixture):
    """§5 WP5 test 4, first half: "At k = 300 kcal, rows whose shifted share of energy falls
    outside the observed range are excluded (today 621–1,398 such rows are counted on-support),
    with the count per check reported."

    Reference: the checks counted by hand with numpy (shares of energy = kcal from the nutrient
    over total energy; observed ranges from the same rows). Today's counts reproduce first: on
    the auditor's fixture 621 rows pass the amount check with a fat share below the lowest
    observed, and on the skeptic's 1,398 rows pass it with a fat or carbohydrate share outside its
    range. (On the auditor's fixture the composition check removes 743: its 621 counted the fat
    share alone, and 122 more rows have a carbohydrate share above the highest observed.)
    """
    X, recipient = SUPPORT_FIXTURES[fixture]()
    kpu = {"fat_g": 9.0, recipient: 4.0}
    amount, share = _by_hand(X, recipient, 300.0)
    energy = X["energy_kcal"].to_numpy()
    if fixture == "auditor":
        fat_share = 9 * X["fat_g"].to_numpy() / energy
        new_fat_share = 9 * (X["fat_g"].to_numpy() - 300 / 9) / energy
        today = amount & ((new_fat_share < fat_share.min()) | (new_fat_share > fat_share.max()))
        assert int(today.sum()) == 621
    else:
        assert int((amount & ~share).sum()) == 1398

    def predict(frame: pd.DataFrame) -> np.ndarray:  # exp5's tree-like surface: only fat-rich diets respond
        fat_share = frame["fat_g"].to_numpy() * 9 / frame["energy_kcal"].to_numpy()
        return 500.0 * np.where(fat_share > 0.30, fat_share - 0.30, 0.0)

    ks = [0.0, 100.0, 200.0, 300.0, 400.0]
    curve = substitution_curve(predict, X, donor="fat_g", recipient=recipient, kcal_per_unit=kpu,
                               ks=ks, total="energy_kcal")
    i = ks.index(300.0)
    assert curve["total"] == "energy_kcal" and curve["n_not_recorded"] == 0
    assert curve["n_off_amount"][i] == int((~amount).sum())
    assert curve["n_off_share"][i] == int((amount & ~share).sum()) > 0
    assert curve["n_on_support"][i] == int((amount & share).sum())
    _, mask = Shift(X, donor="fat_g", recipient=recipient, kcal_per_unit=kpu,
                    total="energy_kcal").apply(X, 300.0)
    assert np.array_equal(mask, amount & share)  # exactly the rows kept by hand
    for j, k in enumerate(ks):  # every k: each row is counted by exactly one check, or kept
        a, s = _by_hand(X, recipient, k)
        assert (curve["n_off_amount"][j], curve["n_off_share"][j], curve["n_on_support"][j]) == (
            int((~a).sum()), int((a & ~s).sum()), int((a & s).sum()))
    assert "share of energy" in curve["note"] and "shares of energy" in curve["effect_sentence"]


def test_4_a_fixed_population_curve_averages_the_same_rows_at_every_k():
    """§5 WP5 test 4, second half: "a fixed-population curve is available" (B17: the rows each
    point averages over change with k, mixing the effect with which rows remain).

    Reference: by hand, the rows on support (both checks) at every k the curve reaches, and the
    mean change in prediction over exactly those rows at each k. On exp5's tree-like surface the
    per-k curve and the fixed-population curve differ, which is the selection B17 names.
    """
    X, recipient = _support_auditor()
    kpu = {"fat_g": 9.0, recipient: 4.0}

    def predict(frame: pd.DataFrame) -> np.ndarray:
        fat_share = frame["fat_g"].to_numpy() * 9 / frame["energy_kcal"].to_numpy()
        return 500.0 * np.where(fat_share > 0.30, fat_share - 0.30, 0.0)

    ks = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    curve = substitution_curve(predict, X, donor="fat_g", recipient=recipient, kcal_per_unit=kpu,
                               ks=ks, total="energy_kcal")
    masks = {k: np.logical_and(*_by_hand(X, recipient, k)) for k in ks}
    live = [k for k in ks if masks[k].mean() >= 0.5 and all(masks[j].mean() >= 0.5 for j in ks if j < k)]
    fixed = np.logical_and.reduce([masks[k] for k in live])
    base = predict(X)
    population = curve["fixed_population"]
    assert population["n_rows"] == int(fixed.sum()) and population["through"] == max(live)
    for k, per_k, same_rows in zip(ks, curve["delta"], population["delta"]):
        if k not in live:
            assert per_k is None and same_rows is None
            continue
        moved = X.copy()
        moved["fat_g"] = X["fat_g"] - k / 9
        moved[recipient] = X[recipient] + k / 4
        change = predict(moved) - base
        assert same_rows == pytest.approx(change[fixed].mean(), rel=1e-12, abs=1e-12)
        assert per_k == pytest.approx(change[masks[k]].mean(), rel=1e-12, abs=1e-12)
    gaps = [abs(a - b) for a, b in zip(curve["delta"], population["delta"]) if a is not None]
    assert max(gaps) > 0.01  # the per-k curve mixes the effect with which rows remain


def test_4_the_stage_reports_the_counts_per_check_and_a_fixed_population_curve(tmp_path):
    """§5 WP5 test 4, through the app: the substitution artifact carries the support's counts per
    check at every k (``support``) and each model's fixed-population curve (``fixed_delta``), with
    the total-energy column the shares were taken of. Reference: the same counts by hand over
    the stage's curve rows (every row of the skeptic's support fixture, as training rows)."""
    X, recipient = _support_skeptic()
    frame = X.drop(columns="protein_g")  # protein is 16% of energy there: collinear with energy
    rng = np.random.default_rng(21)
    frame["y"] = (3 + 0.02 * frame["fat_g"] + 0.01 * frame[recipient] + 0.002 * frame["energy_kcal"]
                  + rng.normal(0, 4, len(frame)))
    sub = _stage(frame, tmp_path, n_boot=0, recipient=recipient)
    support = sub["support"]
    assert support["total"] == "energy_kcal" and support["n_rows"] == len(frame)
    for j, k in enumerate(sub["ks"]):
        amount, share = _by_hand(frame, recipient, k)
        assert support["off_amount"][j] == int((~amount).sum())
        assert support["off_share"][j] == int((amount & ~share).sum())
    assert support["off_share"][sub["ks"].index(300.0)] == 1398  # the skeptic's count, now excluded
    model = sub["models"][0]
    live = [k for k, d in zip(sub["ks"], model["delta"]) if d is not None]
    assert support["fixed_through"] == max(live)
    fixed = np.logical_and.reduce([np.logical_and(*_by_hand(frame, recipient, k)) for k in live])
    assert support["fixed_rows"] == int(fixed.sum())
    assert [d is None for d in model["fixed_delta"]] == [d is None for d in model["delta"]]
    assert "share of energy from either (of energy_kcal)" in sub["note"]
