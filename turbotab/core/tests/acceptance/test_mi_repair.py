"""MI repair · the wave-1 verifier's open items on MS1–MS3 (docs/turbotab-next/MODELING_SEQUENCE.md §0
rulings 5 and 12, §1.1, §2, §4, §5; BLUEPRINT §13, §14.1–§14.3), each a permanent test here:

3. **The energy identity** (MS1). Two blank sources on a row with a recorded total held each other
   above it forever on the log scale (each one's bound read the other's random start, found no room
   and kept its own): the rest of energy reached −840 to −1,715 kcal in 2–3 of 20 copies, with no
   concern and a sentence saying it never goes negative. Now every chain starts each row inside its
   total jointly (``smcfcs._Chain.feasible_start``), so every truncated draw keeps it there; the
   rows whose recorded sources alone reach the total are counted, said and raised as a concern.
   Reference: the copies' own columns, Σ fⱼNⱼ against the total, by NumPy; a random-table property
   run; the design and fit stages with grams recorded.
4. **Knots fixed across copies in Table 2's models** (MS1, the wave-2a seam). The effects stage fit
   each copy with a plain clone: 35 knot sets over 35 copies, a median imputer in each, pooled as
   one basis. Now every stage that fits copies fits them as the fit stage does
   (``missing.copy_pipeline``). Reference: Harrell's knot percentiles of the observed values by
   NumPy (``np.quantile``, R's type 7, as ``Hmisc::rcspline.eval``); each copy's least squares on
   Harrell's spline basis written out here, averaged by hand (Rubin's point estimate).
5. **The survey design in SMC-FCS** (MS2). The design entered the covariate models only, so the
   imputation assumed the outcome independent of the design given the analysis model's terms, and
   an outcome that differs by stratum biased the pooled design-based estimates (the verifier: z
   −0.06, about 7 Monte Carlo SE). Now the design's terms are in the substantive model too.
   Reference: simulation truth, each replicate's full-data design-based estimate on the same knots,
   with a control arm (the design in the covariate models only) that must show the bias.
6. **Time-invariance is a ledger reading** (ruling 12; BLUEPRINT §14.1, §14.3). It was inferred from
   the values, which cannot tell a characteristic that never changes from one asked once (an
   education recorded only at baseline), so a baseline-only column was imputed row by row and its
   structural blanks inflated m to 89. Now it is the ``time_invariant`` kind, settled only by the
   user: unasked, the fit asks; "yes" carries the recorded value to the unit's blank rows (not
   imputed, not counted for m) and imputes once per unit where no row records it; "no" imputes row
   by row. Reference: pandas over the data and the copies.
7. **Sentences**: passive imputation with no derived term; the substitution note under the standard
   energy model; a refusal's nested backticks; ``m = `20```; the stale "(m = 20)" labels; "raise
   m" for the data's own copies.
8. **The Cox SMC-FCS's baseline hazard** is smcfcs's: ``survival::basehaz`` of the Efron-ties fit
   holding the drawn coefficients (Efron's tie-corrected increments, not Breslow's). Reference: R.

**Independent references.** R 4.6.1 (``Rscript`` on CSV files written here; ``survival``), skipped
where R is absent; NumPy and pandas arithmetic written out here; simulation truth with its Monte
Carlo SE stated beside each bound. Never the engine's own code for an expected value.

**Source checks.** Bartlett JW, Seaman SR, White IR, Carpenter JR. *Stat Methods Med Res*
2015;24:462 (SMC-FCS: an auxiliary variable in the covariate models "must be conditionally
independent of the outcome given the covariates in the substantive model"; ``smcfcs``'s manual:
auxiliary variables are "assumed to be conditionally independent of the outcome"). Reiter JP,
Raghunathan TE, Kinney SK. *Survey Methodology* 2006;32:143: "the safest course of action is to
include design variables in the specification of imputation models". Harrell, *Regression Modeling
Strategies* §2.4.6 (knots at the 0.05, 0.35, 0.65, 0.95 quantiles for k = 4). Lüdtke, Robitzsch &
Grund 2017 (clustered imputation). Therneau, ``survival::survfit.coxph`` (``ctype`` 2 with Efron
ties).
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d
from turbotab.core.decisions import (EnergyAdjustment, ExposureFormSpec, GrainSpec, MissingSpec,
                                     ProjectState, Refusal, SetMissing, SplitSpec,
                                     SubstitutionSpec, validate)
from turbotab.core.methods.missing import (copy_template, engine_substantive, fixed_forms,
                                           impute_for_inference, imputation_plan, mi_concerns,
                                           multiple_imputation_info)
from turbotab.core.models import get_family
from turbotab.core.models.inference import INDEPENDENT, Outcome
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.pipeline import build_pipeline, design_spec
from turbotab.core.tests import modeling_fixtures as mf

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R (Rscript) is not installed: the R reference "
                                                     "is skipped")
LINEAR = get_family("linear")
ATWATER = {"protein_g": 4.0, "fat_g": 9.0, "carb_g": 4.0}


def run_r(folder: Path, script: str, **frames: pd.DataFrame) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(folder / f"{name}.csv", index=False, na_rep="NA")
    (folder / "reference.R").write_text(script)
    done = subprocess.run([RSCRIPT, "--no-save", "--no-restore", "reference.R"], cwd=folder,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    return folder


def _rec(i: int, decision: Any) -> d.DecisionRecord:
    return d.DecisionRecord(id=f"r{i}", seq=i, at="2026-10-05T00:00:00Z", decision=decision)


def confirmed(*decisions: Any) -> ProjectState:
    """The state the decision log folds to (the write path the server uses)."""
    return d.fold([_rec(i, dec) for i, dec in enumerate(decisions, start=1)])


# ═════════════════════════════════════════════════════════════════════════════
# 3 · the energy identity: the rest of energy is never negative in a copy
# ═════════════════════════════════════════════════════════════════════════════


def macro_frame(seed: int, n: int = 300, blank: tuple[str, ...] = ("protein_g", "fat_g"),
                share: float = 0.12) -> tuple[pd.DataFrame, np.ndarray]:
    """The verifier's fixture: grams of protein, fat and carbohydrate and total energy (kcal) that
    is their Atwater sum plus 1–6% from other sources, recorded on every row; ``blank`` sources
    blank together on ``share`` of the rows (completely at random)."""
    rng = np.random.default_rng(seed)
    size = rng.lognormal(np.log(2000), 0.3, n)
    p = size * rng.uniform(0.12, 0.20, n) / 4
    f = size * rng.uniform(0.30, 0.40, n) / 9
    c = size * rng.uniform(0.40, 0.50, n) / 4
    kcal = 4 * p + 9 * f + 4 * c + size * rng.uniform(0.01, 0.06, n)
    age = rng.uniform(20, 80, n)
    y = 1 + 0.01 * p + 0.002 * age + rng.normal(0, 1, n)
    rows = rng.random(n) < share
    frame = pd.DataFrame({"protein_g": p, "fat_g": f, "carb_g": c, "kcal": kcal, "age": age})
    for col in blank:
        frame.loc[rows, col] = np.nan
    return frame, y


def identity_spec(frame: pd.DataFrame, log: bool) -> Any:
    roles = {"protein_g": "exposure", "fat_g": "exposure", "carb_g": "exposure", "kcal": "energy",
             "age": "covariate"}
    st = ProjectState(target="y", task="regression", purpose="inference", roles=roles,
                      missing=MissingSpec(strategy="multiple_imputation"),
                      energy_adjustment=EnergyAdjustment(
                          method="residual" if log else "all_components", energy_column="kcal",
                          nutrients=list(ATWATER), log_transform=log))
    return design_spec(st, frame, list(roles))


def identity_copies(frame: pd.DataFrame, y: np.ndarray, log: bool, seed: int) -> tuple[Any, Any]:
    spec = identity_spec(frame, log)
    return impute_for_inference(spec, frame, y, "regression", seed=seed, factors=ATWATER), spec


def rest_of_energy(copy: pd.DataFrame) -> np.ndarray:
    """E − Σ fⱼ Nⱼ, the copy's own columns, by NumPy."""
    return copy["kcal"].to_numpy(dtype=float) - sum(f * copy[s].to_numpy(dtype=float)
                                                    for s, f in ATWATER.items())


@pytest.mark.parametrize("log", [True, False], ids=["log_residual_smcfcs", "all_components_fcs"])
def test_3_two_blank_sources_under_a_recorded_total_never_take_the_rest_below_zero(log):
    """The verifier's minimal reproduction (protein and fat blank together on 12% of the rows,
    total energy recorded on every row; seeds 1–3): on the log scale (SMC-FCS) it gave the rest
    −1,169 to −1,389 kcal in 2 of 20 copies per seed. Now no copy of any seed holds a negative rest
    (NumPy over every copy's own columns), every recorded value comes back as recorded, every drawn
    source is positive, no row is counted infeasible (none is: the recorded carbohydrate alone is
    under 55% of each total), and the record's sentence says the identity as it holds."""
    for seed in (1, 2, 3):
        frame, y = macro_frame(seed)
        imputations, spec = identity_copies(frame, y, log, seed)
        assert imputations.method == ("smcfcs" if log else "chained_equations")
        assert imputations.plan["identity"]["sources"] == list(ATWATER)
        both = frame["protein_g"].isna().to_numpy()
        assert both.sum() > 20
        for copy in imputations.frames:
            assert float(rest_of_energy(copy).min()) >= 0.0, (seed, float(rest_of_energy(copy).min()))
            assert float(copy.loc[both, ["protein_g", "fat_g"]].min().min()) > 0.0
            kept = frame.notna()
            assert np.array_equal(copy[kept].to_numpy(dtype=float)[kept.to_numpy()],
                                  frame[kept].to_numpy(dtype=float)[kept.to_numpy()])
        assert imputations.identity_infeasible == 0
        record = multiple_imputation_info(imputations, spec, len(frame))
        assert record["identity_infeasible"] == 0
        assert ("; the energy sources (`protein_g`, `fat_g` and `carb_g`) and the rest of energy were "
                "imputed and total energy (`kcal`) derived as their sum, so the rest is never "
                "negative;") in record["sentence"]


def drawn_against_truth(blank: tuple[str, ...], log: bool, seed: int) -> float:
    """(the copies' mean imputed protein on its blank rows − the true mean of those rows) over the
    truth's simulation SE (their SD over √n): simulation truth, the values before they were
    blanked."""
    frame, y = macro_frame(seed, n=600, blank=blank, share=0.2)
    truth, _ = macro_frame(seed, n=600, blank=(), share=0.0)
    imputations, _ = identity_copies(frame, y, log, seed)
    rows = frame["protein_g"].isna().to_numpy()
    true = truth.loc[rows, "protein_g"].to_numpy()
    drawn = float(np.mean([c.loc[rows, "protein_g"].mean() for c in imputations.frames]))
    return (drawn - float(true.mean())) / float(np.std(true, ddof=1) / math.sqrt(len(true)))


def test_3_the_drawn_sources_stay_near_the_truth_they_replace(monkeypatch):
    """What the identity's draws condition on (simulation truth; seeds 5–7, n = 600, 20% of rows
    blank, protein alone or protein and fat together, total energy recorded): the copies' mean
    imputed protein on its blank rows lies within 3 simulation SE of the true mean, on the log
    scale (SMC-FCS) and the raw scale (chained equations). A source's covariate model reads the
    recorded total, never "other" (on such a row other is E − Σ fⱼNⱼ, the source's own last value);
    the control arm, with "other" as the predictor as the engine had it, puts the two-source log
    case below −3 SE (−4 to −5.5 in the repair's run), so the check can fail."""
    from turbotab.core.methods import smcfcs

    for blank in (("protein_g",), ("protein_g", "fat_g")):
        for log in (True, False):
            for seed in (5, 6, 7):
                z = drawn_against_truth(blank, log, seed)
                assert abs(z) < 3, (blank, log, seed, z)
    monkeypatch.setattr(smcfcs._Chain, "energy_terms", lambda self, target: None)
    control = [drawn_against_truth(("protein_g", "fat_g"), True, seed) for seed in (5, 6, 7)]
    assert max(control) < -3, control


def random_identity_table(seed: int) -> tuple[pd.DataFrame, np.ndarray, int]:
    """A random table for the property run: 80–200 rows; two or three sources blank together on
    5–30% of rows under a recorded total; some totals blank; two rows whose recorded carbohydrate
    alone exceeds the total while protein is blank (infeasible, by construction). Returns the
    table, the outcome and the infeasible rows NumPy counts."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(80, 201))
    frame, y = macro_frame(seed, n=n, blank=(), share=0.0)
    k = int(rng.integers(2, 4))
    sources = list(rng.permutation(list(ATWATER))[:k])
    rows = rng.random(n) < rng.uniform(0.05, 0.30)
    for s in sources:
        frame.loc[rows, s] = np.nan
    frame.loc[(rng.random(n) < 0.05) & ~rows, "kcal"] = np.nan
    bad = rng.choice(np.flatnonzero(~rows & frame["kcal"].notna().to_numpy()), 2, replace=False)
    frame.loc[bad, "protein_g"] = np.nan
    frame.loc[bad, "kcal"] = 4.0 * frame.loc[bad, "carb_g"] * 0.9
    E = frame["kcal"].to_numpy(dtype=float)
    known = sum(np.nan_to_num(f * frame[s].to_numpy(dtype=float)) for s, f in ATWATER.items())
    blank = frame[list(ATWATER)].isna().any(axis=1).to_numpy()
    infeasible = int(np.sum(np.isfinite(E) & blank & (E - known <= 1e-6 * np.abs(E))))
    return frame, y, infeasible


def test_3_property_run_over_random_tables_the_rest_is_negative_only_where_it_must_be():
    """Twenty random tables (the verifier's property run found 705 violating copy-rows over 20
    configurations), each on the log scale (SMC-FCS) and on the raw scale (chained equations),
    drawn as the inference table's imputation plans them (``imputation_plan``) with 5 copies each
    (the property is each copy's, whatever m): every copy's rest of energy is non-negative on every
    row but the infeasible ones (whose recorded sources alone reach the recorded total), the
    sampler counts exactly those (NumPy), and where the total was blank it is derived as the
    sources plus the rest, which is positive."""
    from turbotab.core.methods.smcfcs import impute

    for seed in range(20):
        frame, y, infeasible = random_identity_table(100 + seed)
        bad = frame["kcal"].notna() & frame[list(ATWATER)].isna().any(axis=1)
        E = frame["kcal"].to_numpy(dtype=float)
        known = sum(np.nan_to_num(f * frame[s].to_numpy(dtype=float)) for s, f in ATWATER.items())
        hopeless = (bad & (E - known <= 1e-6 * np.abs(E))).to_numpy()
        assert infeasible == 2 and int(hopeless.sum()) == 2
        for log in (True, False):
            plan = imputation_plan(identity_spec(frame, log), frame, y, "regression",
                                   factors=ATWATER)
            imputations = impute(plan.data, plan.variables, mode=plan.mode,
                                 substantive=plan.substantive, identity=plan.identity, m=5,
                                 seed=seed, kinds=plan.kinds)
            assert imputations.identity_infeasible == infeasible, (seed, log)
            derived = frame["kcal"].isna().to_numpy()
            for copy in imputations.frames:
                rest = rest_of_energy(copy)
                assert float(rest[~hopeless].min()) >= -1e-9, (seed, log, float(rest[~hopeless].min()))
                assert float(rest[derived].min()) > 0.0


def test_3_through_the_design_and_fit_stages_with_grams_recorded(tmp_path):
    """The verifier's stage run (log residual of fiber with energy kept, a spline of fiber, protein
    and fat blank together on 12% of rows, total energy recorded, units recorded in grams): it gave
    3 of 20 copies a negative rest (−838 to −1,069 kcal) with no concern. Now none (NumPy over the
    fit's own copies), no row is infeasible, no concern names the rest of energy, and the record's
    sentence says it as it holds."""
    from turbotab.core.stages.modeling import design_stage, fit_stage

    rng = np.random.default_rng(22)
    n = 500
    size = rng.lognormal(np.log(2000), 0.3, n)
    p = size * rng.uniform(0.12, 0.2, n) / 4
    f = size * rng.uniform(0.3, 0.4, n) / 9
    c = size * rng.uniform(0.4, 0.5, n) / 4
    kcal = 4 * p + 9 * f + 4 * c + size * rng.uniform(0.01, 0.06, n)
    fib = rng.lognormal(np.log(15), 0.4, n) * (size / 2000) ** 0.7
    age = rng.uniform(20, 80, n)
    y = 1 + 0.5 * np.log(fib) + 0.002 * age + rng.normal(0, 1, n)
    both = rng.random(n) < 0.12
    frame = pd.DataFrame({"fiber_g": fib, "protein_g": np.where(both, np.nan, p),
                          "fat_g": np.where(both, np.nan, f), "carb_g": c, "kcal": kcal,
                          "age": age, "y": y})
    roles = {"fiber_g": "exposure", "protein_g": "covariate", "fat_g": "covariate",
             "carb_g": "covariate", "kcal": "energy", "age": "covariate"}
    paths = mf.ingest_frame(frame, tmp_path)
    st = mf.state(roles=roles, target="y", task="regression", models=["linear"],
                  purpose="inference",
                  energy_adjustment=EnergyAdjustment(method="residual", energy_column="kcal",
                                                     nutrients=["fiber_g", *ATWATER],
                                                     log_transform=True),
                  exposure_forms={"fiber_g": ExposureFormSpec(form="spline", knots=4)},
                  missing=MissingSpec(strategy="multiple_imputation"),
                  split=SplitSpec(holdout=0.0, seed=0, folds=5),
                  column_units=mf.grams(*ATWATER))
    split = mf.split_bundle(np.arange(n), holdout=0.0)
    ti = mf.target_info("regression", "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    model = fit.data["models"][0]
    record = model["inference"]["missing"]
    assert record["model"] == "smcfcs" and record["identity_energy"] == "kcal"
    assert record["identity_infeasible"] == 0
    copies = fit.objects["imputations"]["frames"]
    assert len(copies) == record["m"]
    for copy in copies:
        assert float(rest_of_energy(copy).min()) >= 0.0
    assert not [c for c in model["concerns"] if "rest of energy" in c]
    assert ("the rest of energy were imputed and total energy (`kcal`) derived as their sum, so "
            "the rest is never negative;") in record["sentence"]


def test_3_rows_whose_recorded_sources_reach_the_total_are_counted_said_and_raised():
    """Three rows whose recorded carbohydrate alone exceeds the recorded total while protein and
    fat are blank (NumPy counts them): no draw of the blank sources can keep the rest above zero
    there, so the record counts them, the sentence says the rest is never negative except on those
    rows, and the pooled table raises a concern naming the total."""
    frame, y = macro_frame(9, n=300)
    rows = np.flatnonzero(frame["protein_g"].isna().to_numpy())[:3]
    frame.loc[rows, "kcal"] = 4.0 * frame.loc[rows, "carb_g"] * 0.8
    E = frame["kcal"].to_numpy(dtype=float)
    known = sum(np.nan_to_num(f * frame[s].to_numpy(dtype=float)) for s, f in ATWATER.items())
    expected = int(np.sum(frame[list(ATWATER)].isna().any(axis=1).to_numpy() & (E - known <= 0)))
    assert expected == 3
    imputations, spec = identity_copies(frame, y, True, 9)
    assert imputations.identity_infeasible == expected
    record = multiple_imputation_info(imputations, spec, len(frame))
    assert ("derived as their sum, so the rest is never negative except on the 3 rows whose "
            "recorded sources alone already reach the recorded total") in record["sentence"]
    concerns = mi_concerns(imputations, [], len(frame))
    assert ("On 3 rows the recorded energy sources alone already reach the recorded total energy "
            "(`kcal`), so no value of the blank sources leaves the rest of energy above zero "
            "there; check those rows' units and totals before reading the energy terms.") in concerns
    others = np.setdiff1d(np.arange(len(frame)), rows)
    for copy in imputations.frames:
        assert float(rest_of_energy(copy)[others].min()) >= 0.0


def test_3_a_source_imputed_once_per_unit_stays_inside_every_rows_total():
    """A source the user confirmed as one value per unit (protein, from a baseline food-frequency
    questionnaire, blank for whole persons) under a total recorded on every visit: one value per
    person in every copy (pandas), and the rest of energy non-negative on every row of that person
    (NumPy), the unit's bound being the tightest of its rows'."""
    from turbotab.core.methods.missing import time_invariant_columns
    from turbotab.core.models.inference import Clusters

    rng = np.random.default_rng(31)
    units, visits = 120, 3
    pid = np.repeat(np.arange(units), visits)
    p = np.repeat(rng.uniform(60, 100, units), visits)
    f = rng.uniform(60, 90, len(pid))
    c = rng.uniform(200, 300, len(pid))
    kcal = 4 * p + 9 * f + 4 * c + rng.uniform(20, 150, len(pid))
    frame = pd.DataFrame({"protein_g": p, "fat_g": f, "carb_g": c, "kcal": kcal,
                          "age": np.repeat(rng.uniform(20, 80, units), visits)})
    frame.loc[np.repeat(rng.random(units) < 0.25, visits), "protein_g"] = np.nan
    frame.loc[rng.random(len(pid)) < 0.15, "fat_g"] = np.nan
    y = 1 + 0.01 * p + rng.normal(0, 1, len(pid))
    roles = {"protein_g": "exposure", "fat_g": "exposure", "carb_g": "exposure", "kcal": "energy",
             "age": "covariate"}
    st = confirmed(d.ConfirmReading(reading="time_invariant", column="protein_g", value="yes"))
    st = st.model_copy(update=dict(
        target="y", task="regression", purpose="inference", roles=roles,
        missing=MissingSpec(strategy="multiple_imputation"),
        energy_adjustment=EnergyAdjustment(method="all_components", energy_column="kcal",
                                           nutrients=list(ATWATER))))
    spec = design_spec(st, frame, list(roles))
    codes = pid.copy()
    once, ask = time_invariant_columns(st, frame, codes, ["protein_g", "fat_g"])
    assert once == ["protein_g"] and ask == []  # fat differs within persons: row by row, unasked
    imputations = impute_for_inference(
        spec, frame, y, "regression", seed=4, factors=ATWATER, time_invariant=once,
        clusters=Clusters(column="pid", codes=codes, n_clusters=units))
    assert imputations.plan["unit_imputed"] == ["protein_g"]
    for copy in imputations.frames:
        assert int(copy.assign(pid=pid).groupby("pid")["protein_g"].nunique().max()) == 1
        assert float(rest_of_energy(copy).min()) >= 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 4 · Table 2's models hold the knots placed once, in every copy
# ═════════════════════════════════════════════════════════════════════════════


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell's restricted cubic spline (norm = 2, ``Hmisc::rcspline.eval``'s default), written
    out: x, then for j = 1…k−2, (x − tⱼ)₊³ − (x − t_{k−1})₊³ (t_k − tⱼ)/(t_k − t_{k−1})
    + (x − t_k)₊³ (t_{k−1} − tⱼ)/(t_k − t_{k−1}), each over (t_k − t₁)^{2/3}."""
    t = np.asarray(knots, dtype=float)
    k = len(t)
    kd = (t[-1] - t[0]) ** (2 / 3)
    cols = [x]
    for j in range(k - 2):
        cols.append(np.maximum((x - t[j]) / kd, 0) ** 3
                    + ((t[-2] - t[j]) * np.maximum((x - t[-1]) / kd, 0) ** 3
                       - (t[-1] - t[j]) * np.maximum((x - t[-2]) / kd, 0) ** 3) / (t[-1] - t[-2]))
    return np.column_stack(cols)


@pytest.fixture(scope="module")
def table2_run(tmp_path_factory):
    """The verifier's fixture (a cohort of 700, fiber blank on 30% of rows, age on 10%, a declared
    4-knot spline of fiber, multiple imputation): the design and fit stages, then the effects stage
    (Table 2's crude model, Model 1, the primary and Model 3) with every pipeline it fits recorded."""
    import turbotab.core.models.inner_cv as inner
    from turbotab.core.stages.effects import effects_stage
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(700, seed=31)
    rng = np.random.default_rng(4)
    frame.loc[rng.random(len(frame)) < 0.3, "fiber"] = np.nan
    frame.loc[rng.random(len(frame)) < 0.1, "age"] = np.nan
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  missing={"strategy": "multiple_imputation"},
                  exposure_forms={"fiber": ExposureFormSpec(form="spline", knots=4)})
    paths = mf.ingest_frame(frame, tmp_path_factory.mktemp("table2"))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    ti = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    inputs = {"design": design, "split": split, "target_info": ti}
    fit = fit_stage(mf.context(st, inputs, paths))
    seen: list[Any] = []
    original = inner.fit_pipeline

    def spy(model: Any, X: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        out = original(model, X, y, *args, **kwargs)
        seen.append(out)
        return out

    inner.fit_pipeline = spy
    try:
        effects = effects_stage(mf.context(st, inputs, paths)).data
    finally:
        inner.fit_pipeline = original
    return {"frame": frame, "fit": fit, "effects": effects, "seen": seen,
            "sequence": ef.sequence(effects)}


def test_4_every_copy_of_table_2s_models_holds_the_knots_placed_once(table2_run):
    """Every pipeline the effects stage fits under multiple imputation (the crude model, Model 1 and
    the primary in each of the fit's copies; Model 3 in its own) holds a form step with the knots
    Harrell's rule places on fiber's observed values (NumPy: the 0.05, 0.35, 0.65 and 0.95
    quantiles, R's type 7, 489 observed values so no small-sample rule), and an impute step that
    refuses a blank instead of filling the median (MODELING_SEQUENCE §4: median fill is
    block-and-record for the inference table). The verifier saw 35 knot sets over 35 copies."""
    from turbotab.core.methods.missing import CopyGuard, FixedForms

    observed = table2_run["frame"]["fiber"].dropna().to_numpy()
    assert len(observed) > 100
    knots = np.quantile(observed, [0.05, 0.35, 0.65, 0.95])
    seen = table2_run["seen"]
    m = table2_run["fit"].data["models"][0]["inference"]["missing"]["m"]
    m3 = table2_run["sequence"]["model_3"]["inference"]["missing"]["m"]
    assert len(seen) == m + m3
    for fitted in seen:
        steps = dict(fitted.steps)
        assert isinstance(steps["form"], FixedForms) and isinstance(steps["impute"], CopyGuard)
        np.testing.assert_allclose(steps["form"].knots_["fiber"], knots, rtol=1e-12)
    for key, model in table2_run["sequence"].items():
        record = model["inference"]["missing"]
        np.testing.assert_allclose(record["knots"]["fiber"], knots, rtol=1e-12)
        said = ", ".join(f"{k:.4g}" for k in knots)
        assert (f"the knots of `fiber` ({said}) were placed once on its observed values and held "
                f"in every copy") in record["sentence"], key


def test_4_the_primary_is_the_fits_primary_and_each_model_pools_one_basis(table2_run):
    """Table 2's primary model is the fit's pooled table, row for row (it is fit in the fit's own
    copies: the primary's imputation model, its seed); and each model's pooled spline rows are
    Rubin's point estimate written out here: in each of the fit's copies, least squares (NumPy) on
    an intercept, Harrell's basis of fiber at the observed knots and the model's covariates (sex as
    its male indicator), averaged over the copies. Before the repair Model 2's fiber'' was −1.386
    against the fit's −1.012."""
    fit = table2_run["fit"]
    frame = table2_run["frame"]
    copies = fit.objects["imputations"]["frames"]
    y = frame["glucose"].to_numpy(dtype=float)
    knots = np.quantile(frame["fiber"].dropna().to_numpy(), [0.05, 0.35, 0.65, 0.95])
    fit_rows = {r["feature"]: r for r in fit.data["models"][0]["coefficients"]}
    sequence = table2_run["sequence"]
    primary = {r["feature"]: r for r in sequence["model_2"]["effects"]}
    for name in ("fiber", "fiber'", "fiber''"):
        assert primary[name]["estimate"] == pytest.approx(fit_rows[name]["estimate"], rel=1e-12)
        assert primary[name]["se"] == pytest.approx(fit_rows[name]["se"], rel=1e-12)
    covariates = {"crude": [], "model_2": ["age", "sex", "smoking", "activity"]}
    assert sequence["model_2"]["adjusted_for"] == covariates["model_2"]
    for key, columns in covariates.items():
        coefs = []
        for copy in copies:
            parts = [np.ones(len(copy)), rcs_basis(copy["fiber"].to_numpy(dtype=float), knots)]
            for c in columns:
                parts.append((copy[c] == "male").to_numpy(dtype=float) if c == "sex"
                             else copy[c].to_numpy(dtype=float))
            X = np.column_stack(parts)
            coefs.append(np.linalg.lstsq(X, y, rcond=None)[0][1:4])
        pooled = np.mean(coefs, axis=0)
        got = {r["feature"]: r["estimate"] for r in sequence[key]["effects"]}
        np.testing.assert_allclose([got["fiber"], got["fiber'"], got["fiber''"]], pooled,
                                   rtol=1e-7, atol=1e-10)


def test_4_a_scales_correction_fits_each_copy_as_the_fit_does(tmp_path, monkeypatch):
    """The scales stage's regression calibration under multiple imputation (MS8's item-level copies)
    fits each copy with the impute step that refuses a blank, never the median imputer, as the fit
    stage does (``copy_pipeline``); every copy is fit once per correction."""
    import turbotab.core.models.inner_cv as inner
    from turbotab.core.decisions import ScaleSpec
    from turbotab.core.methods.missing import CopyGuard
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.scales import scales_stage
    from turbotab.core.tests.acceptance.scales_fixtures import linear_scale_table

    items = [f"sat_{j}" for j in range(1, 9)]
    frame = linear_scale_table(n=300, missing=0.1)
    roles = {"age": "covariate", "bmi": "covariate", **{c: "covariate" for c in items}}
    st = mf.state(purpose="inference", target="sbp", task="regression", models=["linear"],
                  roles=roles, role_confirmations=dict(roles), lens=["clinical"],
                  missing=MissingSpec(strategy="multiple_imputation"),
                  split=SplitSpec(holdout=0.0, seed=0, folds=5), shape_confirmations={},
                  scales=[ScaleSpec(name="sat_score", items=items, reverse=["sat_3", "sat_6"],
                                    low=1, high=5, kind="reflective",
                                    correction="regression_calibration", n_boot=50)])
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    ti = mf.target_info("regression", "sbp")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    seen: list[Any] = []
    original = inner.fit_pipeline

    def spy(model: Any, X: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        out = original(model, X, y, *args, **kwargs)
        seen.append(out)
        return out

    monkeypatch.setattr(inner, "fit_pipeline", spy)
    out = scales_stage(mf.context(st, {"design": design, "split": split, "target_info": ti},
                                  paths)).data
    (scale,) = out["scales"]
    m = scale["imputation"]["m"]
    assert m >= 20 and scale["correction"] is not None and scale["correction"]["copies"] == m
    assert len(seen) == m
    assert all(isinstance(dict(f.steps)["impute"], CopyGuard) for f in seen)


# ═════════════════════════════════════════════════════════════════════════════
# 5 · the survey design is in SMC-FCS's substantive model
# ═════════════════════════════════════════════════════════════════════════════


def survey_rep(seed: int, stratum_effect: float = 1.0, n: int = 800) -> dict[str, Any]:
    """The verifier's design: eight strata of two PSUs, weights rising with the stratum; x depends
    on z and the stratum, y on a spline of x, z and the stratum; x blank at random given z."""
    from turbotab.core.models.survey import build_design

    rng = np.random.default_rng(seed)
    s = rng.integers(0, 8, n)
    psu = rng.integers(1, 3, n)
    w = (40 + 30 * s) * rng.uniform(0.8, 1.25, n)
    z = rng.normal(size=n)
    x = 0.4 * z + 0.35 * s + rng.normal(size=n)
    y = 1 + 0.5 * x + 0.4 * z + stratum_effect * s + rng.normal(size=n)
    frame = pd.DataFrame({"x": x, "z": z, "s": s, "psu": psu, "w": w})
    design = build_design(frame, w, weight_column="w", strata_column="s", psu_column="psu")
    blank = rng.random(n) < 1 / (1 + np.exp(-(-0.9 + 0.8 * z)))
    return {"frame": frame, "y": y, "design": design, "blank": blank}


SURVEY_STATE = ProjectState(target="y", task="regression", purpose="inference",
                            roles={"x": "exposure", "z": "covariate"},
                            missing=MissingSpec(strategy="multiple_imputation"),
                            exposure_forms={"x": ExposureFormSpec(form="spline", knots=3)})


def test_5_the_substantive_model_holds_the_designs_terms_beside_the_analysis_models():
    """The SMC-FCS substantive model's matrix is the analysis model's terms (x's spline basis and z,
    as the pipeline's steps make them) followed by the same design terms the covariate models
    read: one indicator per stratum label but the first, one per PSU-within-stratum label but the
    first, and the weight standardized over every row, each written out here with pandas from the
    imputation data; those labels are one-to-one with the design's strata and its PSUs within
    them (pandas); without a survey design, the analysis terms alone."""
    rep = survey_rep(9001)
    X = rep["frame"][["x", "z"]].copy()
    X.loc[rep["blank"], "x"] = np.nan
    spec = design_spec(SURVEY_STATE, X, ["x", "z"])
    plan = imputation_plan(spec, X, rep["y"], "regression", survey=rep["design"])
    assert plan.mode == "smcfcs" and plan.design["columns"] == ["__stratum", "__psu", "__weight"]
    data = plan.data
    frame = rep["frame"]
    pairs = pd.DataFrame({"s": frame["s"].to_numpy(), "psu": frame["psu"].to_numpy(),
                          "S": data["__stratum"].to_numpy(), "P": data["__psu"].to_numpy()})
    assert pairs.groupby("S")["s"].nunique().max() == 1
    assert pairs.groupby("s")["S"].nunique().max() == 1
    assert pairs.groupby(["S", "P"])["psu"].nunique().max() == 1
    assert pairs.groupby(["s", "psu"])["P"].nunique().max() == 1
    raw = data.copy()
    raw["x"] = raw["x"].fillna(float(np.nanmean(raw["x"])))
    M = plan.substantive.design(raw)
    alone = engine_substantive(spec, "regression", rep["y"], plan.forms).design(raw)
    assert alone.shape[1] == 3  # x, x', z
    np.testing.assert_array_equal(M[:, :3], alone)
    expected = []
    for c in ("__stratum", "__psu"):
        levels = sorted(data[c].unique(), key=str)
        expected += [(data[c] == lv).to_numpy(dtype=float) for lv in levels[1:]]
    w = data["__weight"].to_numpy(dtype=float)
    expected.append((w - w.mean()) / w.std())
    np.testing.assert_allclose(M[:, 3:], np.column_stack(expected), rtol=1e-12, atol=1e-12)
    bare = imputation_plan(spec, X, rep["y"], "regression")
    assert bare.substantive.design(raw.drop(columns=["__stratum", "__psu", "__weight"])).shape[1] == 3


def test_5_an_outcome_that_differs_by_stratum_leaves_the_pooled_design_based_estimates_unbiased():
    """Simulation truth (24 replicates, seeds 9000–9023, n = 800, m = 10, a 3-knot spline of x):
    each replicate's truth is the design-based estimate on the full data, on the same knots. With
    the design's terms in the substantive model (the engine), the pooled estimates of x, x' and z
    are unbiased within 3 Monte Carlo SE; with them in the covariate models only (the control arm,
    the engine's model before the repair) z's bias is beyond 3 Monte Carlo SE, so the check can
    fail. The verifier measured z −0.055 (MCSE 0.009) for the old model over 30 replicates."""
    from turbotab.core.methods.smcfcs import impute
    from turbotab.core.stages.modeling import _inference_table, pooled_table

    arms: dict[str, list[dict[str, float]]] = {"engine": [], "covariates_only": []}
    for rep_i in range(24):
        rep = survey_rep(9000 + rep_i)
        full = rep["frame"][["x", "z"]]
        y = rep["y"]
        spec = design_spec(SURVEY_STATE, full, ["x", "z"])
        pipe = build_pipeline(spec, LINEAR, "regression", "inference", len(full), 2)
        forms = fixed_forms(spec, full)
        truth_fit = fit_pipeline(copy_template(pipe, {"forms": forms}), full, y)
        truth = {r["feature"]: r["estimate"] for r in _inference_table(
            LINEAR, truth_fit, full, y, task="regression", clusters=INDEPENDENT,
            outcome=Outcome(name="y"), rows="all", survey=rep["design"]).rows}
        X = full.copy()
        X.loc[rep["blank"], "x"] = np.nan
        spec_x = design_spec(SURVEY_STATE, X, ["x", "z"])
        plan = imputation_plan(spec_x, X, y, "regression", survey=rep["design"])
        for arm in arms:
            design = (plan.data, plan.design["columns"]) if arm == "engine" else None
            substantive = engine_substantive(spec_x, "regression", y, forms, design=design)
            imputations = impute(plan.data, plan.variables, mode=plan.mode,
                                 substantive=substantive, m=10, seed=rep_i, kinds=plan.kinds)
            imputations.frames = [f[["x", "z"]] for f in imputations.frames]
            imputations.plan = {"forms": forms, "levels": []}
            _, rows, _, _ = pooled_table(LINEAR, pipe, imputations, y, task="regression",
                                         clusters=INDEPENDENT, outcome=Outcome(name="y"),
                                         survey=rep["design"], design=None, spec=spec_x,
                                         rows="all", fit=lambda mdl, Xk: fit_pipeline(mdl, Xk, y))
            pooled = {r["feature"]: r["estimate"] for r in rows}
            arms[arm].append({f: pooled[f] - truth[f] for f in ("x", "x'", "z")})
    engine = pd.DataFrame(arms["engine"])
    control = pd.DataFrame(arms["covariates_only"])
    mcse = engine.std(ddof=1) / math.sqrt(len(engine))
    for f in ("x", "x'", "z"):
        assert abs(engine[f].mean()) < 3 * mcse[f], (f, engine[f].mean(), mcse[f])
    control_mcse = control["z"].std(ddof=1) / math.sqrt(len(control))
    assert control["z"].mean() < -3 * control_mcse, (control["z"].mean(), control_mcse)


def test_5_the_sentence_says_where_the_design_entered():
    """The record's sentence names the design's columns in the covariate models and, under SMC-FCS,
    beside the analysis model's terms in its outcome model; under chained equations (the outcome
    a predictor in every covariate model) in the imputation model."""
    rep = survey_rep(9002)
    X = rep["frame"][["x", "z"]].copy()
    X.loc[rep["blank"], "x"] = np.nan
    for forms, smc in (({"x": ExposureFormSpec(form="spline", knots=3)}, True), ({}, False)):
        st = SURVEY_STATE.model_copy(update={"exposure_forms": forms or None})
        spec = design_spec(st, X, ["x", "z"])
        imputations = impute_for_inference(spec, X, rep["y"], "regression", seed=1,
                                           survey=rep["design"])
        imputations.plan["design"].update(strata="s", psu="psu", weight="w")
        record = multiple_imputation_info(imputations, spec, len(X))
        tail = (": in each covariate model and, beside the analysis model's terms, in its outcome "
                "model;" if smc else ";")
        assert (f"; the survey strata (`s`), PSU (`psu`) and weight (`w`) were in the imputation "
                f"model{tail}") in record["sentence"], smc


# ═════════════════════════════════════════════════════════════════════════════
# 6 · time-invariance under clustered imputation is a reading the user settles
# ═════════════════════════════════════════════════════════════════════════════


def visits_frame(seed: int = 606, units: int = 250) -> pd.DataFrame:
    """The verifier's layout: 1–8 visits a person (singletons among them); sex recorded on every
    visit, blank for whole persons (15%); education recorded only at the first visit (and blank on
    10% of those); x varies by visit around the person's level, blank on 30% of visits."""
    rng = np.random.default_rng(seed)
    size = rng.integers(1, 9, units)
    pid = np.repeat(np.arange(units), size)
    sex = np.repeat(rng.integers(0, 2, units).astype(float), size)
    edu = np.repeat(rng.normal(12, 3, units), size)
    level = np.repeat(rng.normal(0, 0.9, units), size)
    x = 0.1 * (edu - 12) + level + rng.normal(0, 1, len(pid))
    y = (1 + 0.6 * x + 0.3 * sex + 0.05 * edu + np.repeat(rng.normal(0, 1, units), size)
         + rng.normal(size=len(pid)))
    first = np.r_[True, pid[1:] != pid[:-1]]
    frame = pd.DataFrame({"id": pid, "x": x, "sex": sex, "edu": edu, "y": y})
    frame.loc[np.repeat(rng.random(units) < 0.15, size), "sex"] = np.nan
    frame.loc[~first, "edu"] = np.nan
    frame.loc[first & (rng.random(len(pid)) < 0.1), "edu"] = np.nan
    frame.loc[rng.random(len(pid)) < 0.3, "x"] = np.nan
    return frame


def visits_fit(frame: pd.DataFrame, folder: Path, answers: dict[str, str]) -> Any:
    """The split, design and fit stages over the visits, rows clustered by ``id`` (the grain
    answer), under multiple imputation, with each time-invariance answer confirmed through the
    decision fold."""
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.stages.rows import split_stage

    roles = {"id": "identifier", "x": "exposure", "sex": "covariate", "edu": "covariate"}
    folded = confirmed(*[d.ConfirmReading(reading="time_invariant", column=c, value=v)
                         for c, v in answers.items()]) if answers else None
    st = ProjectState(lens=["clinical"], target="y", task="regression", purpose="inference",
                      roles=roles, role_confirmations=dict(roles),
                      missing=MissingSpec(strategy="multiple_imputation"),
                      split=SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"],
                      grain=GrainSpec(grain="repeated", id_column="id"), unit="row",
                      shape_confirmations={"code_or_count:edu": "amount"},
                      reading_confirmations=folded.reading_confirmations if folded else None)
    paths = mf.ingest_frame(frame, folder)
    ti = mf.target_info("regression", "y")
    cohort = mf.cohort_bundle(np.arange(len(frame)), ["x", "sex", "edu"])
    split = split_stage(mf.context(st, {"cohort": cohort, "target_info": ti}, paths))
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    return fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))


@pytest.fixture(scope="module")
def visits_runs(tmp_path_factory):
    frame = visits_frame()
    base = tmp_path_factory.mktemp("visits")
    return {"frame": frame,
            "unasked": visits_fit(frame, base / "unasked", {}),
            "yes": visits_fit(frame, base / "yes", {"edu": "yes", "sex": "yes"}),
            "no": visits_fit(frame, base / "no", {"edu": "no", "sex": "yes"})}


def test_6_the_kind_is_settled_only_by_the_user_and_its_consumer_reads_the_ledger():
    """BLUEPRINT §14.3: a kind is settled by its values only through a test that rejects each of
    its alternatives. A baseline-only covariate agrees within every unit exactly as a value that
    never changes does, so ``time_invariant`` declares its alternatives and no value test; the
    user's confirmation settles it, one value per record ("yes" or "no"); and the clustered
    imputation's consumer is registered as asking, reading this kind."""
    from turbotab.core import readings as R

    rule = R.KIND_RULES["time_invariant"]
    assert rule.test is None and not rule.value_settleable and len(rule.alternatives) == 2
    assert "time_invariant" in R.CONFIRMABLE and R.VALUES["time_invariant"] == ("yes", "no")
    (consumer,) = [c for c in R.CONSUMERS if c.where.endswith("missing:time_invariant_columns")]
    assert consumer.changes and consumer.path == R.ASK and consumer.kinds == ("time_invariant",)
    validate(d.ConfirmReading(reading="time_invariant", column="edu", value="yes"),
             {"columns": ["edu", "y"], "target": "y"})
    with pytest.raises(Refusal) as refused:
        validate(d.ConfirmReading(reading="time_invariant", column="edu", value="maybe"),
                 {"columns": ["edu", "y"], "target": "y"})
    assert refused.value.code == "not_a_value"
    frame = visits_frame()
    guess = R.time_invariance(ProjectState(), "edu", frame["edu"], frame["id"])
    assert guess.value == "yes" and not guess.settled and guess.state == "proposed"
    assert not R.time_invariance(ProjectState(), "x", frame["x"], frame["id"]).settled


def test_6_unasked_the_fit_asks_before_any_copy_is_drawn(visits_runs):
    """With rows clustered by `id` and nothing confirmed, the two columns whose recorded values agree
    within every person (pandas: one distinct value per person; sex and edu, not x) are asked,
    their guess and evidence in the question, before any copy is drawn: the table has no
    coefficients, and its exits confirm both as shown in one block, or each either way."""
    frame = visits_runs["frame"]
    per = frame.groupby("id")
    agree = [c for c in ("x", "sex", "edu") if int(per[c].nunique().max()) <= 1]
    assert agree == ["sex", "edu"]
    fit = visits_runs["unasked"]
    model = fit.data["models"][0]
    info = model["inference"]
    assert not model.get("coefficients") and fit.objects.get("imputations") is None
    reason = info.get("refused") or ""
    assert reason.startswith("The rows repeat by `id`, and the recorded values of `sex` and `edu` "
                             "agree within every `id`.")
    assert "the values cannot tell the two apart" in reason
    once = int((frame.groupby("id")["edu"].count() == 1).sum())
    assert f"{once:,} on one row" in reason  # the evidence: education recorded once a person
    exits = [e["decision"] for e in info["exits"]]
    assert exits[0]["kind"] == "confirm_readings"
    assert {(i["column"], i["value"]) for i in exits[0]["items"]} == {("sex", "yes"), ("edu", "yes")}
    singles = {(e["column"], e["value"]) for e in exits[1:]}
    assert singles == {("sex", "yes"), ("sex", "no"), ("edu", "yes"), ("edu", "no")}


def test_6_confirmed_one_value_per_person_education_is_carried_and_m_counts_only_real_blanks(
        visits_runs):
    """Education confirmed as one value per person: in every copy each person holds one education,
    the recorded baseline value wherever the baseline records it (pandas); the blanks carried from
    that row are not imputed, so m is max(20, ⌈100 × the rows still blank / n⌉) with "still blank"
    counted by pandas (x blank, sex blank for the whole person, or education blank for a person
    who never records it). The verifier saw m = 89 from structural blanks. The record and the
    sentence say what was carried and what imputed."""
    frame = visits_runs["frame"]
    fit = visits_runs["yes"]
    record = fit.data["models"][0]["inference"]["missing"]
    copies = fit.objects["imputations"]["frames"]
    per = frame.groupby("id")
    nobody = per["edu"].transform("count").eq(0)
    still = frame["x"].isna() | frame["sex"].isna() | (frame["edu"].isna() & nobody)
    m = max(20, math.ceil(100 * still.mean() - 1e-9))
    assert record["m"] == m == len(copies) and m < 89
    assert record["percent_incomplete"] == round(100 * still.mean(), 1)
    carried = int((frame["edu"].isna() & ~nobody).sum())
    assert record["unit_carried"] == {"sex": 0, "edu": carried}
    assert sorted(record["unit_level"]) == ["edu", "sex"]
    baseline = per["edu"].first()
    for copy in copies:
        held = copy.assign(id=frame["id"].to_numpy()).groupby("id")
        assert int(held["edu"].nunique().max()) == 1 and int(held["sex"].nunique().max()) == 1
        assert np.allclose(held["edu"].first()[baseline.notna()], baseline.dropna())
    assert (f"; with rows clustered by `id`, `sex` and `edu` were confirmed as one value per `id`, "
            f"carried from the rows that record them to the `id`'s blank rows ({carried:,} values) "
            f"and imputed once per `id` where no row records them; each row-level variable was "
            f"imputed with the `id` means of the others and its own mean over the `id`'s other "
            f"rows;") in record["sentence"]


def test_6_confirmed_as_changing_education_is_imputed_row_by_row(visits_runs):
    """The other answer produces the other behavior (BLUEPRINT §14.3, every confirmation is honored):
    education confirmed as able to change between visits is imputed row by row, so some person's
    copies hold several educations (pandas), it leaves the record's one-value-per-unit list, and m
    counts every blank education row again."""
    frame = visits_runs["frame"]
    fit = visits_runs["no"]
    record = fit.data["models"][0]["inference"]["missing"]
    assert record["unit_level"] == ["sex"] and "edu" in record["row_level"]
    copy = fit.objects["imputations"]["frames"][0]
    assert int(copy.assign(id=frame["id"].to_numpy()).groupby("id")["edu"].nunique().max()) > 1
    still = frame[["x", "sex", "edu"]].isna().any(axis=1)
    assert record["m"] == max(20, math.ceil(100 * still.mean() - 1e-9))


def test_6_a_confirmed_column_whose_records_differ_keeps_them_and_says_so():
    """A column the user confirmed as one value per unit although its recorded values differ within
    some units (a data-entry slip): its recorded values are kept, each such unit's blank rows take
    its most frequent recorded value (pandas' mode, the first on a tie), and a concern names the
    column and how many units differ."""
    from turbotab.core.methods.missing import carry_within_units
    from turbotab.core.models.inference import Clusters

    values = pd.Series([1.0, 1.0, 2.0, np.nan, 5.0, np.nan, np.nan, 3.0, 4.0, np.nan])
    units = np.array([0, 0, 0, 0, 1, 1, 2, 3, 3, 3])
    carried, n, differ = carry_within_units(values, units)
    assert carried.tolist()[:4] == [1.0, 1.0, 2.0, 1.0] and carried[5] == 5.0
    assert np.isnan(carried[6]) and carried[9] == 3.0  # unit 3: 3 and 4 tie, the first kept
    assert (n, differ) == (3, 2)
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"g": np.repeat(np.arange(60), 3), "x": rng.normal(size=180)})
    frame["w"] = np.repeat(rng.integers(0, 3, 60).astype(float), 3)
    frame.loc[[0, 3], "w"] = [7.0, 8.0]
    frame.loc[[1, 5, 10, 11, 12], "w"] = np.nan
    frame.loc[rng.random(180) < 0.2, "x"] = np.nan
    st = confirmed(d.ConfirmReading(reading="time_invariant", column="w", value="yes"))
    st = st.model_copy(update=dict(target="y", task="regression", purpose="inference",
                                   roles={"x": "exposure", "w": "covariate"},
                                   missing=MissingSpec(strategy="multiple_imputation")))
    X = frame[["x", "w"]]
    spec = design_spec(st, X, ["x", "w"])
    y = rng.normal(size=180)
    codes = frame["g"].to_numpy()
    imputations = impute_for_inference(spec, X, y, "regression", seed=1, time_invariant=["w"],
                                       clusters=Clusters(column="g", codes=codes, n_clusters=60))
    assert imputations.plan["unit_differ"] == {"w": 2}
    concerns = mi_concerns(imputations, [], len(X))
    assert ("Confirmed as one value per `g`, `w` (2) has recorded values that differ within some "
            "`g`s (the count in brackets): their recorded values were kept, and each such `g`'s "
            "blank rows took its most frequent recorded value.") in concerns
    for copy in imputations.frames:
        assert copy.loc[0, "w"] == 7.0 and copy.loc[3, "w"] == 8.0


# ═════════════════════════════════════════════════════════════════════════════
# 7 · the sentences the verifier found wrong
# ═════════════════════════════════════════════════════════════════════════════


def plain_frame(seed: int = 3, n: int = 400) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = 0.5 * z + rng.normal(size=n)
    y = 1 + x + z + rng.normal(size=n)
    return pd.DataFrame({"x": np.where(rng.random(n) < 0.3, np.nan, x), "z": z}), y


def test_7_passive_imputation_with_no_derived_term_is_said_as_what_it_is():
    """Passive chained equations chosen for a linear model with no nonlinear term are exactly the
    compatible chained equations, and the sentence says so (it said "with derived in each copy
    (passive imputation), kept as a recorded limitation"); for a logistic model with no term they
    are the customary approximation, said as one."""
    frame, y = plain_frame()
    for task, outcome in (("regression", y), ("binary", (y > 1).astype(float))):
        st = ProjectState(target="y", task=task, purpose="inference",
                          roles={"x": "exposure", "z": "covariate"},
                          missing=MissingSpec(strategy="multiple_imputation",
                                              imputation_model="passive"))
        spec = design_spec(st, frame, ["x", "z"])
        imputations = impute_for_inference(spec, frame, outcome, task, seed=0)
        record = multiple_imputation_info(imputations, spec, len(frame))
        pct = record["percent_incomplete"]
        start = (f"Missing values were multiply imputed by chained equations with the outcome in "
                 f"the imputation model, ")
        if task == "regression":
            assert record["compatible"] is True and record["approximate"] is False
            assert record["sentence"].startswith(
                start + f"compatible with the analysis model, linear in the imputed variables; m = "
                        f"{record['m']} imputations, at least 20 and at least the {pct:g}% of rows "
                        f"with an imputed value (White, Royston & Wood 2011);")
        else:
            assert record["approximate"] is True
            assert record["sentence"].startswith(
                start + f"as the answer chose; they match the analysis model, a logistic model, "
                        f"only approximately, which SMC-FCS would match exactly (Bartlett et al. "
                        f"2015); m = {record['m']} imputations")
        assert "derived in each copy" not in record["sentence"]


def test_7_the_substitution_note_under_the_standard_energy_model_names_no_other_model(tmp_path):
    """Under the standard energy model (nutrients and total energy, linear) the pooled curve is the
    exact contrast of the pooled coefficients; the note says why by the property it rests on, never
    "for the linear all-components model" (the model it is not)."""
    from turbotab.core.stages.modeling import design_stage, fit_stage, substitution_stage

    rng = np.random.default_rng(909)
    n = 600
    age = rng.normal(45, 13, n)
    size = 2000 + 500 * rng.normal(size=n) - 6 * (age - 45)
    kp, kf = size * np.clip(rng.normal(0.17, 0.03, n), 0.05, 0.4), size * np.clip(
        rng.normal(0.33, 0.05, n), 0.1, 0.6)
    kc, ko = size * np.clip(rng.normal(0.43, 0.05, n), 0.1, 0.7), size * rng.uniform(0.02, 0.1, n)
    y = 0.012 * kp + 0.003 * kc - 0.002 * kf + 0.15 * (age - 45) + rng.normal(0, 4, n)
    frame = pd.DataFrame({"protein_g": kp / 4, "carb_g": kc / 4, "fat_g": kf / 9,
                          "kcal": kp + kf + kc + ko, "age": age, "y": y})
    frame.loc[rng.random(n) < 0.2, "fat_g"] = np.nan
    roles = {"protein_g": "exposure", "carb_g": "exposure", "fat_g": "exposure", "kcal": "energy",
             "age": "covariate"}
    paths = mf.ingest_frame(frame, tmp_path)
    st = mf.state(roles=roles, target="y", task="regression", models=["linear"],
                  purpose="inference",
                  energy_adjustment=EnergyAdjustment(method="standard", energy_column="kcal",
                                                     nutrients=list(ATWATER)),
                  missing=MissingSpec(strategy="multiple_imputation"),
                  substitution=SubstitutionSpec(donor="fat_g", recipient="protein_g",
                                                step_kcal=100, n_boot=0),
                  split=SplitSpec(holdout=0.0, seed=0, folds=5), column_units=mf.grams(*ATWATER))
    split = mf.split_bundle(np.arange(n), holdout=0.0)
    ti = mf.target_info("regression", "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    sub = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
    assert sub["models"][0]["pooled"] == "contrast"
    m = len(fit.objects["imputations"]["frames"])
    assert (f"The curve is pooled over the {m} imputations: the move changes every row's model "
            f"terms by the same amount, so the curve is the exact contrast of the pooled "
            f"coefficients, with Rubin's total variance and Barnard–Rubin degrees of freedom.") \
        in sub["note"]
    assert "all-components" not in sub["note"]


def test_7_the_passive_refusal_ticks_each_column_once():
    """The refusal of passive imputation with a declared spline names the term with its column
    ticked once (it read "`a restricted cubic spline of `fiber``")."""
    st = ProjectState(target="y", purpose="inference", roles={"fiber": "exposure"},
                      exposure_forms={"fiber": ExposureFormSpec(form="spline", knots=4)})
    with pytest.raises(Refusal) as refused:
        validate(SetMissing(strategy="multiple_imputation", imputation_model="passive"),
                 {"state": st})
    assert refused.value.message.startswith(
        "The analysis model holds a restricted cubic spline of `fiber`. Under inference passive "
        "imputation draws")
    assert "``" not in refused.value.message


def test_7_the_answers_sentence_and_labels_state_m_by_its_rule():
    """The missing-values answer's sentence writes m as a number, not a data value (it read
    "m = `20`"); the method's label and the single-fill refusal's exit no longer fix m at 20, which
    the rule raises with the share of incomplete rows."""
    from turbotab.core.methods.missing import METHODS, MI_EXIT_LABEL, missing_block
    from turbotab.core.voice import sentence_for

    said = sentence_for(SetMissing(strategy="multiple_imputation"),
                        ProjectState(target="y", purpose="inference", roles={"x": "exposure"}))
    assert "m = 20 or the percentage of rows with an imputed value if that is larger" in said
    assert "`20`" not in said
    label = METHODS["multiple_imputation"].label
    assert label == "Multiple imputation (m at least 20, more when more rows are incomplete)"
    assert MI_EXIT_LABEL == ("Multiple imputation with the outcome and energy (m at least 20, "
                             "more when more rows are incomplete)")
    with pytest.raises(Refusal) as refused:
        validate(SetMissing(strategy="impute"),
                 {"state": ProjectState(target="y", purpose="inference", roles={"x": "exposure"})})
    assert refused.value.exits[0]["label"] == MI_EXIT_LABEL
    held = missing_block({"strategy": "impute"}, "inference")
    assert held is not None and held[1][0]["label"] == MI_EXIT_LABEL
    assert all("m = 20" not in e["label"] for e in [*refused.value.exits, *held[1]])


def test_7_the_datas_own_copies_never_ask_to_raise_m():
    """For the data's own imputed copies (NHANES DXA, m fixed by NCHS) a Monte Carlo error above 10%
    of a standard error is said as the provider's m, never "raise m"; for the engine's own copies
    it still says to raise m. The ratio is the row's own (0.15, NumPy)."""
    from turbotab.core.methods.imputation import Imputations

    rows = [{"feature": "fiber", "estimate": 1.0, "se": 0.2, "mc_se": 0.03, "fmi": 0.1}]
    supplied = Imputations(frames=[], m=5, iterations=0, imputed={}, n_incomplete_rows=0,
                           variables=[], kinds={}, method="supplied")
    said = mi_concerns(supplied, rows, 100)
    assert said == ["The Monte Carlo error exceeds 10% of the standard error for `fiber` (15%) "
                    "(White, Royston & Wood 2011): the data's provider drew only 5 copies, which "
                    "cannot be added to here, so these estimates would move with another set of "
                    "the provider's copies; read their last digits with that in mind."]
    engine = Imputations(frames=[], m=20, iterations=10, imputed={}, n_incomplete_rows=0,
                         variables=[], kinds={}, method="smcfcs")
    assert mi_concerns(engine, rows, 100)[-1].endswith("another set of m imputations would move "
                                                       "it; raise m.")


# ═════════════════════════════════════════════════════════════════════════════
# 8 · the Cox SMC-FCS's baseline hazard is smcfcs's
# ═════════════════════════════════════════════════════════════════════════════


@needs_r
@pytest.mark.parametrize("entry", [False, True], ids=["from_zero", "delayed_entry"])
def test_8_the_cox_draws_baseline_hazard_is_basehaz_of_the_efron_fit_at_the_draw(tmp_path, entry):
    """smcfcs (2.0.2, ``smcfcs.core``) replaces the Efron-ties ``coxph`` fit's coefficients by the
    draw and takes ``survival::basehaz(ymod, centered = FALSE)``: the Efron-corrected cumulative
    hazard at the drawn coefficients. With yearly follow-up (heavy ties) the engine's draw's H₀ at
    each row's time (less H₀ at its entry, with delayed entry) equals R's to 1e-10; Breslow's,
    which the engine used before, differs by more than 5% here (the verifier: up to 5.2%)."""
    from turbotab.core.methods.smcfcs import Outcome as SMOutcome
    from turbotab.core.methods.smcfcs import _cox_draw, breslow

    rng = np.random.default_rng(12)
    n = 400
    x = rng.normal(size=n)
    z = rng.binomial(1, 0.5, n).astype(float)
    t = np.minimum(np.ceil(rng.exponential(1 / (0.12 * np.exp(0.5 * x + 0.3 * z)))), 10.0)
    start = (np.floor(rng.uniform(0, 3, n)) * (t > 3)) if entry else np.zeros(n)
    d_ = ((t < 10) | (rng.random(n) < 0.3)).astype(float)
    M = np.column_stack([x, z])
    fit = _cox_draw(M, SMOutcome("cox", time=t, event=d_, entry=start), np.random.default_rng(3))
    surv = "Surv(s, t, d)" if entry else "Surv(t, d)"
    folder = run_r(tmp_path, f"""
suppressMessages(library(survival))
dat <- read.csv("data.csv"); b <- read.csv("beta.csv")$beta
ymod <- coxph({surv} ~ x + z, dat, control = coxph.control(timefix = FALSE), model = TRUE)
ymod$coefficients <- b
bh <- basehaz(ymod, centered = FALSE)
H0 <- function(u) {{ k <- findInterval(u, bh$time); ifelse(k == 0, 0, bh$hazard[pmax(k, 1)]) }}
write.csv(data.frame(H = H0(dat$t) - H0(dat$s)), "H.csv", row.names = FALSE)
""", data=pd.DataFrame({"s": start, "t": t, "d": d_, "x": x, "z": z}),
                   beta=pd.DataFrame({"beta": fit.beta}))
    H = pd.read_csv(folder / "H.csv")["H"].to_numpy()
    np.testing.assert_allclose(fit.H, H, rtol=1e-10, atol=1e-12)
    old = breslow(M @ fit.beta, t, d_, start)
    B = old(t) - old(start)
    assert float(np.max(np.abs(B - H) / np.where(H > 0, H, 1.0))) > 0.05
