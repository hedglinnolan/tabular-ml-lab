"""The sensitivity stage names every exposure matrix column it reports as the fit's coefficient
table names it (PP2 N1: under Predict the stage failed in glucose analyst runs 36 and 37 and on
every step-4 refit of glucose's plans, a ``ValidationError`` on ``exposures.29 ... Input should be
a valid string, input_value=None``).

The premise checked: no pipeline step makes a matrix column without a name. The stage read its
exposures from ``lineage.trace``, which collapses a plan of more than ``COLLAPSE_AT`` raw columns
for display: the columns no step changed fold into one count node per role and operation, and a
count node has no column. Under Predict every predictor is an exposure, so each folded exposure
group was read as an exposure named None; glucose's plan has 49 inputs. The stage now reads the
lineage unfolded, so every matrix column it returns is one the coefficient table names; a plan of
no more than ``COLLAPSE_AT`` raw columns, which the lineage never folds, is read as before.

Expected values come from construction (the exposures are the columns made from `x..` and `grp`,
the covariate `age` the only other input) and from statsmodels (each named column's estimate).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d
from turbotab.core.decisions import SplitSpec
from turbotab.core.models.lineage import COLLAPSE_AT

N = 300
KEEP = 200  # the sensitivity analysis keeps the first 200 rows


def _table(width: int) -> pd.DataFrame:
    """``width`` numeric exposures `x00`, `x01`, ..., a three-level exposure `grp`, the covariate
    `age`, an identifier and a `visit` column that numbers the rows (excluded: only a rule reads
    it), and the outcome `y`."""
    rng = np.random.default_rng(11)
    xs = {f"x{j:02d}": rng.normal(size=N) for j in range(width)}
    frame = pd.DataFrame({"pid": np.arange(N), **xs, "grp": rng.choice(["a", "b", "c"], N),
                          "age": rng.normal(50, 10, N), "visit": np.arange(N, dtype=float)})
    signal = sum((j % 3 - 1) * 0.5 * frame[c] for j, c in enumerate(xs))
    frame["y"] = (signal + 1.5 * (frame["grp"] == "b") + 0.1 * frame["age"]
                  + rng.normal(size=N))
    return frame


def _stage(tmp_path: Path, width: int, purpose: str) -> tuple[dict, list[str]]:
    """The sensitivity stage's artifact (the linear family, every row the primary's, an analysis
    on the first ``KEEP``) and the design's inputs."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.models.pipeline import DesignSpec
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.sensitivity import sensitivity_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = _table(width)
    roles = {"pid": "identifier", "visit": "excluded", "age": "covariate", "grp": "exposure",
             **{c: "exposure" for c in frame.columns if c.startswith("x")}}
    rule = d.ExclusionRule(column="visit", low=0, high=KEEP - 1, reason=f"the first {KEEP}")
    st = mf.state(roles=roles, role_confirmations=dict(roles), target="y", task="regression",
                  purpose=purpose, models=["linear"], lens=["clinical"],
                  split=SplitSpec(holdout=0.0, seed=3, folds=5), shape_confirmations={},
                  grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
                  sensitivity=[d.SensitivityAnalysis(label="first", rules=[rule])])
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(N), holdout=0.0, seed=3)
    inputs = {"split": split, "target_info": mf.target_info("regression", "y")}
    design = design_stage(mf.context(st, inputs, paths))
    with DataStore(paths["data"], 1 << 30) as store:
        ingest = store.info().to_dict()
    out = sensitivity_stage(mf.context(st, {**inputs, "design": design, "ingest": ingest}, paths))
    return out.data, list(DesignSpec.from_dict(design.objects["spec"]).inputs)


def _ols(frame: pd.DataFrame, width: int) -> pd.Series:
    """statsmodels' least squares of `y` on the exposures and `age`, `grp` coded against `a`."""
    X = frame[[f"x{j:02d}" for j in range(width)] + ["age"]].assign(
        grp_b=(frame["grp"] == "b").astype(float), grp_c=(frame["grp"] == "c").astype(float))
    return sm.OLS(frame["y"], sm.add_constant(X)).fit().params


# The design's inputs: the `x..` columns, `grp` and `age`. COLLAPSE_AT of them the lineage never
# folds; one more and it folds the columns no step changed.
NARROW = COLLAPSE_AT - 2
WIDE = COLLAPSE_AT - 1


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
@pytest.mark.parametrize("width", [WIDE, NARROW], ids=["folded", "never-folded"])
def test_the_stage_names_each_exposure_column_as_the_coefficient_table_does(tmp_path, width,
                                                                            purpose):
    """On the base, the plan one raw column wider than the lineage shows unfolded failed the stage
    with pydantic's "Input should be a valid string" (the folded `x..` columns read as one exposure
    named None); the plan at the lineage's width ran. Both run here: the exposures are the
    coefficient table's columns made from `x..` and `grp`, in its order, never `age`'s, and each
    one's change across the analyses starts at the primary's estimate, statsmodels' on every row."""
    out, inputs = _stage(tmp_path, width, purpose)
    assert (len(inputs) > COLLAPSE_AT) == (width == WIDE)
    [linear] = out["families"]
    primary, first = linear["fits"]
    assert (primary["label"], primary["n_rows"], first["n_rows"]) == ("Primary", N, KEEP)
    features = [r["feature"] for r in primary["coefficients"]]
    made_from_exposures = [f for f in features if f.startswith(("x", "grp_"))]
    # construction: every `x..` column and two of `grp`'s three levels, beside `age` and the
    # intercept, which are no exposure's
    assert set(features) - set(made_from_exposures) == {"(intercept)", "age"}
    assert sorted(f for f in made_from_exposures if f.startswith("x")) == [
        f"x{j:02d}" for j in range(width)]
    assert sorted(f for f in made_from_exposures if f.startswith("grp_")) == ["grp_b", "grp_c"]
    assert out["exposures"] == made_from_exposures
    changes = {c["feature"]: c for c in out["changes"]["linear"]}
    assert list(changes) == made_from_exposures
    ols = _ols(_table(width), width)
    for feature in made_from_exposures:
        assert changes[feature]["primary"] == pytest.approx(ols[feature], rel=1e-8, abs=1e-12)


def test_a_change_reads_each_fits_first_row_of_the_name_and_none_where_a_fit_lacks_it():
    """``changes_for`` reads each fit's rows once (a search per exposure grew as their square on a
    plan this wide): a name's first row in a fit is its estimate there, as before, a fit without
    the name gives no interval for it, and a name no fit estimates has no change, written out
    here."""
    from turbotab.core.stages.sensitivity import changes_for

    fits = [{"coefficients": [{"feature": "x", "estimate": 2.0, "ci_low": 1.0, "ci_high": 3.0},
                              {"feature": "x", "estimate": -9.0, "ci_low": -10.0, "ci_high": -8.0},
                              {"feature": "z", "estimate": 1.0}]},
            {"coefficients": [{"feature": "x", "estimate": -1.0, "ci_low": -2.0, "ci_high": 0.5}]},
            {"coefficients": None}]
    [x, z] = changes_for(fits, ["x", "z", "w"])
    assert x == {"feature": "x", "primary": 2.0, "lowest": -1.0, "highest": 2.0,
                 "sign_changes": True, "excludes_zero": [True, False, None]}
    assert z == {"feature": "z", "primary": 1.0, "lowest": 1.0, "highest": 1.0,
                 "sign_changes": False, "excludes_zero": [None, None, None]}
