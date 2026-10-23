"""The comparison substrate's repeats by the training units (the owner's ruling of 2026-10-23,
option 3; ``models/folds.py``): under prediction the families and the no-predictor baseline are
compared on repeated k-fold of at least 5 repeats of K folds where the training units number 5,000
or more, and of at least 10 below, the split's own repeats first and kept where they are more.

Every count expected here is the ruling's, written out (never read from the code): 4,999 training
units keep 10 repeats and 5,000 take 5, counted in units, not rows, where a unit's rows repeat. A
fit on 5,000 training rows pairs its families over the 25 folds it ran, by the corrected repeated
k-fold t written out from Nadeau & Bengio's formula (``references_ms6``) on the artifact's own fold
scores and fold sizes, and says so in its chain; a fit on 4,999 pairs them over 50. On a true null
built here (two exchangeable least-squares models on 5,000 rows, 5 × 5 folds) the interval calls
one ahead in at most the nominal share of tables, as it does at 10 × 5 on small ones
(``test_rf1_sentences``). No test here names the substrate's constants or its claim by its id
(meta-test 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core import voice
from turbotab.core.decisions import (DecisionLog, GrainSpec, ProjectState, SelectModels, SetGrain,
                                    SetLens, SetMissing, SetPurpose, SetRoles, SetSplit, SetTarget,
                                    SetTask, SetValidation, SplitSpec)
from turbotab.core.models.folds import comparison_folds
from turbotab.core.models.validation import corrected_t, validation_plan
from turbotab.core.provenance import methods_text
from turbotab.core.stages.modeling import design_stage, fit_stage, shelf_stage, substrate_repeats
from turbotab.core.stages.rows import cohort_stage, split_stage
from turbotab.core.stages.seal import seal_plan_stage
from turbotab.core.stages.target import target_info_stage
from turbotab.core.tests.acceptance import references_ms6 as ref
from turbotab.core.tests.stage_harness import Ingested


def _headline(n: int, seed: int, folds: int = 5) -> np.ndarray:
    """One balanced random partition of ``n`` rows into ``folds`` folds."""
    return np.random.default_rng(seed).permutation(np.arange(n) % folds)


def _pairs_of_rows(units: int) -> np.ndarray:
    """Each of ``units`` units holds two rows, in order: 0, 0, 1, 1, …"""
    return np.repeat(np.arange(units), 2)


def _whole_units(columns: list[np.ndarray]) -> bool:
    """Every column puts both rows of each unit (rows 2u and 2u + 1) in one fold."""
    return all(np.array_equal(c[0::2], c[1::2]) for c in columns)


# ── the rule, on comparison_folds ─────────────────────────────────────────────


@pytest.mark.parametrize("validation", ["kfold", "bootstrap"])
def test_from_5000_training_units_the_substrate_takes_5_repeats(validation):
    """5,000 training rows, one row each, and 5,000 units of two rows each: 5 repeats of the 5
    folds, the split's own run first; the units' rows stay together in every repeat."""
    rows = _headline(5_000, 1)
    columns, shared, note = comparison_folds([rows], validation=validation, scheme="random",
                                             n=5_000, folds=5, seed=4)
    assert (len(columns), shared, note) == (5, 1, None)
    assert np.array_equal(columns[0], rows)
    assert all(sorted(np.bincount(c).tolist()) == [1_000] * 5 for c in columns)
    assert len({c.tobytes() for c in columns}) == 5  # five partitions, none drawn twice
    groups = _pairs_of_rows(5_000)
    own = np.repeat(_headline(5_000, 2), 2)
    columns, shared, _ = comparison_folds([own], validation=validation, scheme="random",
                                          n=10_000, groups=groups, folds=5, seed=4)
    assert (len(columns), shared) == (5, 1) and _whole_units(columns)


def test_below_5000_training_units_the_substrate_keeps_10_repeats():
    """4,999 training rows, one row each: 10 repeats. 4,999 units of two rows each, 9,998 rows: the
    units are counted, so 10 repeats too, though the rows number more than 5,000."""
    columns, shared, _ = comparison_folds([_headline(4_999, 1)], validation="kfold",
                                          scheme="random", n=4_999, folds=5, seed=4)
    assert (len(columns), shared) == (10, 1)
    groups = _pairs_of_rows(4_999)
    columns, shared, _ = comparison_folds([np.repeat(_headline(4_999, 2), 2)], validation="kfold",
                                          scheme="random", n=9_998, groups=groups, folds=5, seed=4)
    assert (len(columns), shared) == (10, 1) and _whole_units(columns)


@pytest.mark.parametrize("n,own,want", [
    (5_000, 3, 5), (5_000, 7, 7), (4_999, 3, 10), (4_999, 12, 12)])
def test_the_splits_own_repeats_come_first_and_stay_where_they_are_more(n, own, want):
    """Repeated k-fold: the split's own repeats are the substrate's first, and the substrate runs
    at least the rule's repeats (5 from 5,000 training units, 10 below), more where the split
    drew more."""
    headline = [_headline(n, s) for s in range(own)]
    columns, shared, _ = comparison_folds(headline, validation="repeated_kfold", scheme="random",
                                          n=n, folds=5, seed=4)
    assert (len(columns), shared) == (want, own)
    assert all(np.array_equal(a, b) for a, b in zip(columns, headline))


@pytest.mark.parametrize("n,want", [(5_000, 5), (4_999, 10)])
def test_beside_internal_external_validation_the_substrate_is_drawn_afresh_by_the_rule(n, want):
    """Internal–external validation's folds are a grouping's levels, so none of the substrate's
    repeats is the headline's."""
    sites = np.arange(n) % 4
    columns, shared, _ = comparison_folds([sites], validation="internal_external",
                                          scheme="random", n=n, folds=5, seed=4)
    assert (len(columns), shared) == (want, 0)
    assert all(sorted(np.unique(c).tolist()) == [0, 1, 2, 3, 4] for c in columns)


@pytest.mark.parametrize("n", [5_000, 4_999])
def test_time_ordered_folds_run_once_at_any_size(n):
    columns, shared, note = comparison_folds([_headline(n, 1)], validation="kfold",
                                             scheme="time_ordered", n=n, folds=5, seed=4)
    assert (len(columns), shared) == (1, 1) and note


# ── the validation plan counts the same refits ───────────────────────────────


def test_the_plans_cost_counts_25_comparison_refits_from_5000_units_and_says_when_50():
    """The split question's costs, counted before any holdout is chosen, over every analyzed unit:
    from 5,000 the comparisons' 5 × 5 folds, 10 × 5 should a holdout leave fewer than 5,000 to
    train; below 5,000 the comparisons' 10 × 5, whatever is held out. Repeated k-fold's own 10
    repeats are at least either, so its cost is 50 at both sizes; with 3 of its own on 6,000
    units, 25, and 50 should a holdout leave fewer than 5,000 to train."""
    large = {o.validation: o.cost for o in validation_plan("prediction", 13_921).options}
    assert large["bootstrap"] == ("about 500 refits of each family, besides 25 for the "
                                  "comparisons' 5 × 5-fold repeats (50, 10 × 5, if fewer than "
                                  "5,000 rows train)")
    assert large["kfold"] == ("25 refits of each family: the score's 5-fold run is the first of "
                              "the comparisons' 5 × 5-fold repeats (50, 10 × 5, if fewer than "
                              "5,000 rows train)")
    assert large["repeated_kfold"] == "50 refits of each family"
    small = {o.validation: o.cost for o in validation_plan("prediction", 4_999).options}
    assert small["bootstrap"] == ("about 500 refits of each family, besides 50 for the "
                                  "comparisons' 10 × 5-fold repeats")
    assert small["kfold"] == ("50 refits of each family: the score's 5-fold run is the first of "
                              "the comparisons' 10 × 5-fold repeats")
    assert small["repeated_kfold"] == "50 refits of each family"
    three = {o.validation: o.cost for o in validation_plan("prediction", 6_000, repeats=3).options}
    assert three["repeated_kfold"] == ("25 refits of each family (50, 10 × 5, if fewer than 5,000 "
                                       "rows train)")


# ── the fit compares over the folds it ran ───────────────────────────────────


def _regression(n: int, seed: int) -> pd.DataFrame:
    """y = 1 + 0.8·x1 − 0.5·x2 + noise, x3 noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    frame["pid"] = np.arange(n)
    frame["y"] = 1.0 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=n)
    return frame


def _fit(frame: pd.DataFrame, folder) -> tuple[pd.DataFrame, object]:
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "table.csv"
    frame.to_csv(source, index=False)
    state = ProjectState(
        lens=["clinical"], target="y", task="regression", purpose="prediction",
        roles={"pid": "identifier", "x1": "covariate", "x2": "covariate", "x3": "covariate"},
        missing="complete_case", grain=GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=SplitSpec(holdout=0.0, seed=5, folds=5, validation="kfold"),
        models=["linear", "ridge"])
    table = Ingested(source, folder / "i")
    info = table.run(target_info_stage, state)
    cohort = table.run(cohort_stage, state, {"target_info": info})
    split = table.run(split_stage, state, {"cohort": cohort, "target_info": info})
    design = table.run(design_stage, state, {"split": split, "target_info": info})
    fit = table.run(fit_stage, state, {"design": design, "split": split, "target_info": info})
    return split.frames["assignment"], fit


@pytest.mark.parametrize("n,repeats", [(5_000, 5), (4_999, 10)])
def test_a_fit_pairs_its_families_over_the_folds_it_ran_by_the_corrected_t(tmp_path, n, repeats):
    """No rows held out, one 5-fold run as the headline: the fit compares linear and ridge
    regression, and each with the baseline, over 5 repeats of the 5 folds on 5,000 training rows
    (25 paired folds, 24 df) and over 10 on 4,999 (50, 49 df). The t, its SE and df are Nadeau &
    Bengio's formula written out on the artifact's fold scores, with every fold's rows trained and
    tested counted on the substrate's own folds, to 10⁻¹⁰; the chain says the repeats run."""
    frame = _regression(n, 8)
    assignment, fit = _fit(frame, tmp_path)
    data = fit.data
    rk = 5 * repeats
    assert (data["comparison"]["folds"], data["comparison"]["repeats"],
            data["comparison"]["shared"]) == (5, repeats, 1)
    models = {m["family"]: m for m in data["models"]}
    for m in models.values():
        assert len(m["compared_on"]["folds"]) == rk and m["compared_on"]["repeats"] == repeats
        assert m["versus_baseline"]["df"] == rk - 1
    cmp = data["comparisons"][0]
    assert (cmp["a"], cmp["b"], cmp["df"], cmp["folds"], cmp["repeats"]) == (
        "linear", "ridge", rk - 1, rk, repeats)
    a = assignment.sort_values("row_id").reset_index(drop=True)
    columns, _, _ = comparison_folds([a["fold"].to_numpy()], validation="kfold", scheme="random",
                                     n=n, folds=5, seed=5)
    n_train = [int((c != k).sum()) for c in columns for k in range(5)]
    n_test = [int((c == k).sum()) for c in columns for k in range(5)]
    t, df, se = ref.corrected_repeated_t(models["linear"]["compared_on"]["folds"],
                                         models["ridge"]["compared_on"]["folds"], n_train, n_test,
                                         lower_is_better=True)
    assert df == rk - 1
    assert abs(cmp["t"] - t) < 1e-10 and abs(cmp["se"] - se) < 1e-10
    assert f"{rk - 1} df, {repeats} × 5 folds" in cmp["sentence"]
    link = next(c for c in data["chain"] if c["relation"] == "families_compared_on_substrate")
    assert link["then"] == ("every family and the no-predictor baseline were compared on 5-fold "
                            f"cross-validation repeated {repeats} times ({rk} paired folds, "
                            f"{rk - 1} degrees of freedom)")


# ── the corrected t at 5 × 5 on a large table, under a true null ────────────


def _null_cv(rng: np.random.Generator, n: int, r: int, K: int = 5):
    """Two least-squares models, each on one noise column, so neither predicts better: their
    fold scores (negative mean squared error) over r repeats of K folds."""
    x = rng.normal(size=(n, 2))
    y = rng.normal(size=n)
    out = {0: [], 1: []}
    for _ in range(r):
        folds = rng.permutation(n) % K
        for k in range(K):
            test, train = folds == k, folds != k
            for j in (0, 1):
                X = np.column_stack([np.ones(train.sum()), x[train, j]])
                beta = np.linalg.lstsq(X, y[train], rcond=None)[0]
                pred = beta[0] + beta[1] * x[test, j]
                out[j].append(-float(np.mean((y[test] - pred) ** 2)))
    return out[0], out[1]


# ── the split's record sentence says the repeats the fit runs ───────────────
#
# The sentence is written when the split is answered, before any row is drawn. Where nothing is
# held out every analyzed unit trains, so the seal plan counts them; where a holdout is drawn, at
# most the analyzed rows train, so fewer than 5,000 analyzed rows take 10 repeats, and otherwise
# the draw decides and the sentence states the rule.

COMPARED = ("models were compared with each other and with the no-predictor baseline on `5`-fold "
            "cross-validation repeated ")


def _project(units: int, per_unit: int, split: SplitSpec, folder, *,
             purpose: str = "prediction", blank: int = 0) -> tuple[ProjectState, Ingested, dict]:
    """``units`` units of ``per_unit`` rows each (y = 1 + 0.8·x1 + noise), their stages through the
    split, and the seal plan the split's sentence reads. ``blank``: x2 is blank on the first so
    many rows, which complete cases leave out of the analysis."""
    rng = np.random.default_rng(units + per_unit)
    n = units * per_unit
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                          "pid": np.repeat(np.arange(units), per_unit)})
    frame["y"] = 1.0 + 0.8 * frame["x1"] + rng.normal(size=n)
    frame.loc[: blank - 1, "x2"] = np.nan
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "table.csv", index=False)
    repeated = per_unit > 1
    state = ProjectState(
        lens=["clinical"], target="y", task="regression", purpose=purpose,
        roles={"pid": "identifier", "x1": "covariate", "x2": "covariate"},
        missing="complete_case", split=split, models=["linear"],
        grain=GrainSpec(grain="repeated" if repeated else "one_row_per_unit", id_column="pid"),
        **({"repeat_kind": {"repeat_kind": "repeats"}} if repeated else {}))
    table = Ingested(folder / "table.csv", folder / "i")
    info = table.run(target_info_stage, state)
    cohort = table.run(cohort_stage, state, {"target_info": info})
    plan = table.run(seal_plan_stage, state, {"cohort": cohort, "target_info": info})
    split_ = table.run(split_stage, state, {"cohort": cohort, "target_info": info})
    return state, table, {"target_info": info, "cohort": cohort, "split": split_, "plan": plan}


def _said(state: ProjectState, ctx: dict) -> str:
    return voice.sentence_for(SetSplit(**state.split.model_dump()), state, ctx)


def _fit_repeats(state: ProjectState, table: Ingested, run: dict) -> int:
    design = table.run(design_stage, state,
                       {"split": run["split"], "target_info": run["target_info"]})
    fit = table.run(fit_stage, state, {"design": design, "split": run["split"],
                                       "target_info": run["target_info"]})
    return fit.data["comparison"]["repeats"]


@pytest.mark.parametrize("units,per_unit", [(5_000, 1), (5_000, 2)])
def test_with_nothing_held_out_the_split_sentence_says_5_repeats_from_5000_units(
        tmp_path, units, per_unit):
    """No rows held out, one 5-fold run: 5,000 rows of one unit each, and 5,000 units of two rows
    each, take 5 repeats, and the sentence says the 5 the fit runs, with no rule beside it."""
    state, table, run = _project(units, per_unit, SplitSpec(holdout=0.0, seed=5, folds=5),
                                 tmp_path)
    said = _said(state, {"seal_plan": run["plan"]})
    assert f"{COMPARED}`5` times, by the corrected repeated k-fold t" in said, said
    if per_unit > 1:
        assert "keeping each `pid`'s rows together" in said
    assert _fit_repeats(state, table, run) == 5


@pytest.mark.parametrize("units,per_unit", [(4_999, 1), (4_999, 2)])
def test_with_nothing_held_out_the_split_sentence_keeps_10_repeats_below_5000_units(
        tmp_path, units, per_unit):
    """4,999 rows, and 4,999 units of two rows each (9,998 rows: the units are counted): 10
    repeats, said as the fit runs them, with no rule beside them."""
    state, table, run = _project(units, per_unit, SplitSpec(holdout=0.0, seed=5, folds=5),
                                 tmp_path)
    said = _said(state, {"seal_plan": run["plan"]})
    assert f"{COMPARED}`10` times, by the corrected repeated k-fold t" in said, said
    assert _fit_repeats(state, table, run) == 10


def test_where_a_holdout_decides_how_many_train_the_split_sentence_states_the_rule(tmp_path):
    """A 20% holdout drawn over 10,000 rows with the outcome recorded, of which 6,500 are analyzed
    (x2 blank on 3,500, complete cases): it holds out at most 2,000, which could leave 4,500 to
    train, so the sentence, written before the draw, states the rule (10 repeats, 5 from 5,000
    training rows); the draw leaves 5,000 or more and the fit runs 5, which the methods text says
    once the split is drawn. On 4,999 rows at most 4,999 train, so 10, with no rule. 5,000 units of
    two rows each with a holdout: the rule counts units."""
    state, table, run = _project(10_000, 1, SplitSpec(holdout=0.2, seed=5, folds=5),
                                 tmp_path / "a", blank=3_500)
    assert (run["plan"]["n_measured"], run["plan"]["n_analyzed"]) == (10_000, 6_500)
    said = _said(state, {"seal_plan": run["plan"]})
    rule = "`10` times (`5` where `5,000` or more rows train)"
    assert f"{COMPARED}{rule}, by the corrected repeated k-fold t" in said, said
    assert int(run["split"].data["n_train"]) >= 5_000 and _fit_repeats(state, table, run) == 5
    with table.store() as store:
        drawn = substrate_repeats(run["split"], store)
    now = voice.restate(SetSplit(**state.split.model_dump()), said, state, state,
                        {"counts": {"set_split": {"comparison_repeats": drawn}}})
    assert now == said.replace(rule, "`5` times"), now
    state, _, run = _project(4_999, 1, SplitSpec(holdout=0.2, seed=5, folds=5), tmp_path / "b")
    said = _said(state, {"seal_plan": run["plan"]})
    assert f"{COMPARED}`10` times, by the corrected repeated k-fold t" in said, said
    state, _, run = _project(5_000, 2, SplitSpec(holdout=0.2, seed=5, folds=5), tmp_path / "c")
    said = _said(state, {"seal_plan": run["plan"]})
    assert (f"{COMPARED}`10` times (`5` where `5,000` or more units train), by the corrected "
            f"repeated k-fold t") in said, said


@pytest.mark.parametrize("n,says,repeats", [
    (21_849, "`5` times", 5), (6_250, "`5` times", 5),
    (6_249, "`10` times (`5` where `5,000` or more rows train)", 10)])
def test_where_the_most_a_share_holds_out_leaves_5000_rows_to_train_the_sentence_says_5(
        tmp_path, n, says, repeats):
    """A 20% holdout by row with every row analyzed holds out at most ⌈0.2 × n⌉ rows, so at least
    n − ⌈0.2 × n⌉ train: 17,479 of 21,849 (the NHANES glucose table's rows) and 5,000 of 6,250,
    so the sentence says the 5 repeats the fit runs, with no rule; and so does a changed scheme
    recorded after the draw. 6,249 rows leave 4,999, so the rule, and the fit runs 10."""
    state, table, run = _project(n, 1, SplitSpec(holdout=0.2, seed=5, folds=5), tmp_path)
    assert int(run["split"].data["n_train"]) == n - int(np.ceil(0.2 * n))
    said = _said(state, {"seal_plan": run["plan"]})
    assert f"{COMPARED}{says}, by the corrected repeated k-fold t" in said, said
    changed = voice.sentence_for(SetValidation(folds=5), state, {"seal_plan": run["plan"]})
    assert f"{COMPARED}{says}, by the corrected repeated k-fold t" in changed, changed
    assert _fit_repeats(state, table, run) == repeats


@pytest.mark.parametrize("n,own,holdout,says", [
    (6_250, 3, 0.0, "`5` times"), (6_250, 3, 0.2, "`5` times"), (6_250, 7, 0.2, "`7` times"),
    (6_249, 3, 0.2, "`10` times (`5` where `5,000` or more rows train)"),
    (6_249, 7, 0.2, "`10` times (`7` where `5,000` or more rows train)"),
    (6_249, 12, 0.2, "`12` times")])
def test_the_splits_own_repeats_count_in_the_sentence_as_in_the_fit(tmp_path, n, own, holdout,
                                                                   says):
    """Repeated k-fold: the substrate runs the split's own repeats where they are more than the
    rule's, so 12 of its own are said with no rule, whatever is held out; on 6,250 rows, nothing
    held out or a 20% holdout (which leaves at least 5,000 to train), 3 of its own take 5 and 7
    stay 7; on 6,249 with a 20% holdout (at least 4,999 train) the rule says them."""
    split = SplitSpec(holdout=holdout, seed=5, folds=5, validation="repeated_kfold", repeats=own)
    state, _, run = _project(n, 1, split, tmp_path)
    said = _said(state, {"seal_plan": run["plan"]})
    assert f"cross-validation repeated `{own}` times, each score the mean over the repeats" in said
    assert f"{COMPARED}{says}, by the corrected repeated k-fold t" in said, said


def test_without_the_seal_plan_the_sentence_counts_what_it_can():
    """No seal plan: 4,999 rows entering the split train fewer than 5,000, so 10; 5,000 rows may
    group into fewer units, so the rule; with no count at all, the rule."""
    state = ProjectState(target="y", task="regression", purpose="prediction",
                         split=SplitSpec(holdout=0.0, seed=5, folds=5))
    assert f"{COMPARED}`10` times, by" in _said(state, {"n_cohort": 4_999})
    rule = f"{COMPARED}`10` times (`5` where `5,000` or more rows train), by"
    assert rule in _said(state, {"n_cohort": 5_000})
    assert rule in _said(state, {})


def test_under_inference_the_split_sentence_says_no_comparison(tmp_path):
    """Under inference the scores describe fit: no comparisons, so no repeats, at any size."""
    state, _, run = _project(5_000, 1, SplitSpec(holdout=0.0, seed=5, folds=5), tmp_path,
                             purpose="inference")
    said = _said(state, {"seal_plan": run["plan"]})
    assert "compared with each other" not in said and "repeated" not in said, said


# ── the methods text says the repeats the fit runs, as the rows stand ─────────
#
# The split's sentence is said on the rows as they stand when it is recorded. A later answer can
# move the training units across 5,000 (here a role that brings in, or leaves out, a column blank
# on 300 of 5,200 rows, under complete cases), and with them the repeats the fit runs, so the
# methods text restates the repeats: those the drawn split gives (``substrate_repeats``, the fit's
# own count), or their rule while no split is drawn on the answers as they stand.


def _roles(x2: str) -> dict[str, str]:
    return {"pid": "identifier", "x1": "covariate", "x2": x2}


def _walk(x2_then: str, x2_now: str, folder, *, scheme: SetValidation | None = None) -> dict:
    """A decision log on 5,200 rows, one per ``pid`` (y = 1 + 0.8·x1 + noise), x2 blank on 300,
    under complete cases: the split (nothing held out, 5 folds) answered while x2 is ``x2_then``,
    its sentence said on the seal plan then, as the server says it; ``scheme``, a changed
    validation scheme recorded next; then x2 made ``x2_now``. The stages run on the answers as
    they stand, and the methods text is built with the repeats the drawn split gives."""
    rng = np.random.default_rng(52)
    frame = pd.DataFrame({"pid": np.arange(5_200), "x1": rng.normal(size=5_200),
                          "x2": rng.normal(size=5_200)})
    frame["y"] = 1.0 + 0.8 * frame["x1"] + rng.normal(size=5_200)
    frame.loc[:299, "x2"] = np.nan
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "table.csv", index=False)
    table = Ingested(folder / "table.csv", folder / "i")

    def on_the_plan(decision, state):
        info = table.run(target_info_stage, state)
        cohort = table.run(cohort_stage, state, {"target_info": info})
        plan = table.run(seal_plan_stage, state, {"cohort": cohort, "target_info": info})
        return voice.sentence_for(decision, state, {"seal_plan": plan})

    log = DecisionLog(folder / "decisions.jsonl")
    for decision in (SetLens(lenses=["clinical"]), SetTarget(column="y"),
                     SetTask(column="y", task="regression"), SetPurpose(purpose="prediction"),
                     SetGrain(grain="one_row_per_unit", id_column="pid"),
                     SetRoles(roles=_roles(x2_then)), SetMissing(strategy="complete_case"),
                     SelectModels(models=["linear"])):
        log.append(decision)
    log.append(SetSplit(holdout=0.0, seed=5, folds=5), sentence=on_the_plan)
    if scheme is not None:
        log.append(scheme, sentence=on_the_plan)
    if x2_now != x2_then:
        log.append(SetRoles(roles=_roles(x2_now)))
    state = log.state()
    info = table.run(target_info_stage, state)
    cohort = table.run(cohort_stage, state, {"target_info": info})
    split = table.run(split_stage, state, {"cohort": cohort, "target_info": info})
    with table.store() as store:
        drawn = substrate_repeats(split, store)
    facts = {"comparison_repeats": drawn}
    text = methods_text(log.records(), {"counts": {"set_split": facts, "set_validation": facts}})
    return {"records": {r.decision.kind: r.sentence for r in log.records() if r.sentence},
            "methods": {line.kind: line.sentence for line in text.lines}, "drawn": drawn,
            "n_train": int(split.data["n_train"]),
            "fit": _fit_repeats(state, table, {"split": split, "target_info": info})}


@pytest.mark.parametrize("then,now,said,ran", [("excluded", "covariate", 5, 10),
                                               ("covariate", "excluded", 10, 5)])
def test_a_later_answer_that_moves_the_training_units_across_5000_restates_the_repeats(
        tmp_path, then, now, said, ran):
    """x2 left out when the split is answered: 5,200 rows train, and the sentence says 5 repeats;
    x2 then made a covariate, complete cases leave 4,900, and the fit runs 10, which the methods
    text says. The reverse: 4,900 when answered (10), 5,200 after (5). The Record keeps what was
    said."""
    walk = _walk(then, now, tmp_path)
    assert walk["n_train"] == {"excluded": 5_200, "covariate": 4_900}[now]
    assert walk["fit"] == walk["drawn"] == ran
    assert f"{COMPARED}`{said}` times, by" in walk["records"]["set_split"]
    assert walk["methods"]["set_split"] == walk["records"]["set_split"].replace(
        f"{COMPARED}`{said}` times, by", f"{COMPARED}`{ran}` times, by")


@pytest.mark.parametrize("x2,ran", [("excluded", 5), ("covariate", 10)])
def test_with_no_later_answer_the_methods_text_keeps_the_split_sentence_as_said(tmp_path, x2, ran):
    """No answer after the split's moves its rows: the fit runs the repeats the sentence said (5 on
    5,200 training rows, 10 on 4,900), and the methods text keeps the sentence as said."""
    walk = _walk(x2, x2, tmp_path)
    assert walk["fit"] == walk["drawn"] == ran
    assert f"{COMPARED}`{ran}` times, by" in walk["records"]["set_split"]
    assert walk["methods"]["set_split"] == walk["records"]["set_split"]


def test_a_changed_scheme_is_restated_with_the_repeats_the_fit_runs(tmp_path):
    """Repeated k-fold of 3 repeats recorded after the split, on 5,200 training rows: 5 repeats,
    said; x2 then a covariate, 4,900 train, and the fit runs 10, which the methods text says of
    every sentence that states the comparisons."""
    walk = _walk("excluded", "covariate", tmp_path,
                 scheme=SetValidation(folds=5, validation="repeated_kfold", repeats=3))
    assert walk["fit"] == walk["drawn"] == 10
    assert f"{COMPARED}`5` times, by" in walk["records"]["set_validation"]
    stated = [s for s in walk["methods"].values() if COMPARED in s]
    assert stated and all(f"{COMPARED}`10` times, by" in s for s in stated), walk["methods"]


def _said_on(log: DecisionLog, split: SetSplit) -> str:
    """``split`` recorded on ``log`` and said with 4,999 rows entering it (no seal plan)."""
    return log.append(split, sentence=lambda d, s: voice.sentence_for(d, s, {"n_cohort": 4_999})
                      ).sentence


@pytest.mark.parametrize("facts,says", [
    ({"comparison_repeats": None, "grouped": False},
     "`10` times (`5` where `5,000` or more rows train)"),
    ({"comparison_repeats": None, "n_analyzed": 5_000, "grouped": False},
     "`10` times (`5` where `5,000` or more rows train)"),
    ({"comparison_repeats": None, "n_analyzed": 9_000, "grouped": True},
     "`10` times (`5` where `5,000` or more units train)"),
    ({"comparison_repeats": None, "n_analyzed": 4_999, "grouped": False}, "`10` times"),
    ({"comparison_repeats": 5}, "`5` times"), (None, "`10` times"), ({}, "`10` times")])
def test_while_no_split_is_drawn_on_the_answers_as_they_stand_the_rule_is_said(tmp_path, facts,
                                                                              says):
    """A split said with its count (10, on 4,999 rows entering it): while no split is drawn on the
    answers as they stand the methods text states the rule, by unit where the folds keep units
    whole, unless fewer than 5,000 rows are analyzed now (at most they train: 10); with a count,
    the count; with nothing known, the sentence as said."""
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.append(SetTarget(column="y"))
    said = _said_on(log, SetSplit(holdout=0.0, seed=5, folds=5))
    assert f"{COMPARED}`10` times, by" in said
    counts = {"counts": {"set_split": facts}} if facts is not None else None
    [line] = [x for x in methods_text(log.records(), counts).lines if x.kind == "set_split"]
    assert line.sentence == said.replace(f"{COMPARED}`10` times", f"{COMPARED}{says}"), line


@pytest.mark.parametrize("own,says", [(12, "`12` times"),
                                      (7, "`10` times (`7` where `5,000` or more rows train)")])
def test_the_rule_said_in_the_methods_counts_the_splits_own_repeats(tmp_path, own, says):
    """Repeated k-fold of the split's own 12 repeats: more than either count, so 12 is said, as
    said and in the methods text; 7: more than 5 and fewer than 10, so the rule says 7 from
    5,000 training rows."""
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.append(SetTarget(column="y"))
    said = _said_on(log, SetSplit(holdout=0.0, seed=5, folds=5, validation="repeated_kfold",
                                  repeats=own))
    assert f"{COMPARED}`{max(own, 10)}` times, by" in said
    counts = {"counts": {"set_split": {"comparison_repeats": None, "grouped": False}}}
    [line] = [x for x in methods_text(log.records(), counts).lines if x.kind == "set_split"]
    assert line.sentence == said.replace(f"{COMPARED}`{max(own, 10)}` times",
                                         f"{COMPARED}{says}"), line


# ── the shelf's estimate counts the comparisons' refits the fit will make ──────


@pytest.mark.parametrize("units,per_unit,holdout,refits", [
    (5_000, 1, 0.0, 25), (4_999, 1, 0.0, 50), (5_000, 2, 0.0, 25), (4_999, 2, 0.0, 50),
    (6_250, 1, 0.2, 25)])
def test_the_shelfs_estimate_counts_the_comparisons_refits_by_the_training_units(
        tmp_path, monkeypatch, units, per_unit, holdout, refits):
    """One 5-fold run as the headline, so every fit the estimate counts is the comparisons': 5 × 5
    from 5,000 training units (5,000 rows, 5,000 units of two rows, or 6,250 rows less a 20%
    holdout), 10 × 5 below (4,999 rows, or 4,999 units of two rows), counted on the training rows
    the shelf times, as the fit stage counts them."""
    import turbotab.core.models.cost as cost

    state, table, run = _project(units, per_unit, SplitSpec(holdout=holdout, seed=5, folds=5),
                                 tmp_path)
    seen: dict[str, int] = {}
    real = cost.estimate_fits

    def spy(store, state, task, train_ids, families, folds, **kw):
        seen["folds"] = folds
        return real(store, state, task, train_ids, families, folds, **kw)

    monkeypatch.setattr(cost, "estimate_fits", spy)
    table.run(shelf_stage, state, {"cohort": run["cohort"], "target_info": run["target_info"],
                                   "split": run["split"]})
    assert seen["folds"] == refits


# ── the corrected t at 5 × 5 on a large table, under a true null ────────────


def test_at_5_by_5_folds_on_5000_rows_a_true_null_rarely_calls_a_family_ahead():
    """Two exchangeable models on 5,000 rows, 5 repeats of 5 folds, 300 tables: the corrected
    interval (written out here: (1/(rK) + 1/(K − 1))·s², the t quantile on rK − 1 df from scipy,
    checked against ``corrected_t`` on the first table) excludes zero in at most 5% of them."""
    rng = np.random.default_rng(23)
    r, K = 5, 5
    ahead = 0
    for i in range(300):
        a, b = _null_cv(rng, 5_000, r)
        d = np.asarray(a) - np.asarray(b)
        se = np.sqrt((1 / (r * K) + 1 / (K - 1)) * d.var(ddof=1))
        half = stats.t.ppf(0.975, r * K - 1) * se
        lo, hi = d.mean() - half, d.mean() + half
        if i == 0:
            got = corrected_t(a, b, test_share=None, repeats=r)
            assert got.df == r * K - 1
            assert (got.ci_low, got.ci_high) == pytest.approx((lo, hi), rel=1e-12)
        ahead += int(lo > 0 or hi < 0)
    assert ahead / 300 <= 0.05
