"""Acceptance: EXPLAIN (V2 definition of done §2, "Explainability"; MODELING_SEQUENCE §5, wave 2).

Every expected value comes from a path independent of ``turbotab/core/models/explain.py``:

* SHAP: the shap package's ``TreeExplainer`` (path-dependent) and ``LinearExplainer``, and the
  closed form ``β (z − E[z])`` computed here from the fitted model's own coefficients, with those
  coefficients checked against R's ``lm``;
* interactions: Friedman and Popescu's H² and its root-mean-square size computed here by loops over
  the rows, each partial dependence from the fitted pipeline's own ``predict``;
* curves: Apley and Zhu's ALE computed here by loops from the definition (the grid from NumPy's
  inverse-CDF quantiles), and an additive function whose ALE is known in closed form;
* the architecture lane: R's ``lm`` (the equation) and R's ``glmnet`` (the shrinkage path, with
  scikit-learn's penalty mapped to glmnet's: glmnet scales a Gaussian response by its SD, so
  ``λ_g = α (ρ + (1 − ρ) s)`` and ``α_g = α ρ / λ_g``), and the tree read straight off the fitted
  model's nodes;
* stability: SciPy's Spearman correlation of the reported importances, and a simulation with a
  known ranking of true effects.

Tests that call R are skipped where ``Rscript`` is not installed.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core import decisions as d
from turbotab.core.decisions import ColumnUnitSpec, EstimandSpec, ExplainSpec, GrainSpec, RepeatSpec
from turbotab.core.models import explain as E
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.graph_runner import GraphRun

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R (Rscript) is not installed")
ADJUST = ("detect", "normalize", "impute", "energy")


# ── runs through the real stage graph ────────────────────────────────────────


@dataclass
class Run:
    graph: GraphRun
    out: dict[str, Any]
    state: Any

    @property
    def art(self) -> dict[str, Any]:
        return self.out["explain"].data

    def family(self, key: str) -> dict[str, Any]:
        return next(f for f in self.art["families"] if f["family"] == key)

    def fitted(self, key: str) -> Any:
        objects = self.out["fit"].objects
        if self.state.purpose == "inference":
            return objects["every_row"][key]
        return objects["fitted"][key]

    def rows(self, ids: Any) -> pd.DataFrame:
        """The raw inputs of these rows, read as every modeling stage reads them."""
        from turbotab.core.datastore import DataStore
        from turbotab.core.models.pipeline import DesignSpec, modeling_frame

        spec = DesignSpec.from_dict(self.out["design"].objects["spec"])
        with DataStore(self.graph.raw, 1 << 30) as store:
            frame = modeling_frame(store, spec.inputs, np.asarray(ids, dtype=np.int64))
        return frame.loc[list(ids), spec.inputs]

    def fit_ids(self) -> np.ndarray:
        if self.state.purpose == "inference":
            return np.asarray(self.out["fit"].objects["every_row_ids"], dtype=np.int64)
        from turbotab.core.stages.modeling import row_ids_of

        return row_ids_of(self.out["design"].frames["training"])

    def y_fit(self) -> np.ndarray:
        from turbotab.core.datastore import DataStore

        with DataStore(self.graph.raw, 1 << 30) as store:
            frame = store.materialize([self.state.target], self.fit_ids())
        return frame[self.state.target].to_numpy(dtype=float)


def _graph(tmp: Path, frame: pd.DataFrame, **slots: Any) -> Run:
    tmp.mkdir(parents=True, exist_ok=True)
    source = tmp / "table.csv"
    frame.to_csv(source, index=False)
    graph = GraphRun(source, tmp / "project")
    state = mf.state(**slots)
    return Run(graph, graph.run(state, upto=["explain"]), state)


@pytest.fixture(scope="module")
def regression(tmp_path_factory: Any) -> Run:
    """NHANES-shaped, residual energy model, the outcome's unit and age's unit recorded."""
    return _graph(tmp_path_factory.mktemp("regression"), mf.nhanes_like(600, seed=1),
                  energy_adjustment=mf.energy("residual"), explain=ExplainSpec(),
                  outcome_unit="mg/dL",
                  column_units={"age": ColumnUnitSpec(unit="years"),
                                "kcal": ColumnUnitSpec(unit="kcal", days=1)})


@pytest.fixture(scope="module")
def binary(tmp_path_factory: Any) -> Run:
    frame = mf.nhanes_like(700, seed=4)
    frame["high"] = np.where(frame["glucose"] > frame["glucose"].median(), "yes", "no")
    frame = frame.drop(columns=["glucose"])
    return _graph(tmp_path_factory.mktemp("binary"), frame, target="high", event="yes",
                  energy_adjustment=mf.energy("residual"), explain=ExplainSpec(reseeds=2))


@pytest.fixture(scope="module")
def noise(tmp_path_factory: Any) -> Run:
    """An outcome no predictor carries: no family beats the outcome's average, and one of the
    elastic net's refits shrinks every coefficient to zero."""
    frame = mf.nhanes_like(400, seed=5)
    frame["glucose"] = np.random.default_rng(55).normal(100, 10, len(frame))
    return _graph(tmp_path_factory.mktemp("noise"), frame, explain=ExplainSpec(reseeds=2))


PLANTED = 0.5  # mg/dL per year of age over 50, in men only: the planted age × sex interaction


@pytest.fixture(scope="module")
def sorted_by_sex(tmp_path_factory: Any) -> Run:
    """An export sorted by sex, then age, as exports often are, with an interaction planted on the
    sort key (simulation truth): glucose rises ``PLANTED`` mg/dL a year faster with age in men."""
    frame = mf.nhanes_like(600, seed=6)
    men = (frame["gender"] == "male").to_numpy()
    frame["glucose"] = frame["glucose"] + PLANTED * (frame["age"] - 50) * men
    frame = frame.sort_values(["gender", "age"], kind="stable").reset_index(drop=True)
    return _graph(tmp_path_factory.mktemp("sorted"), frame,
                  energy_adjustment=mf.energy("residual"), explain=ExplainSpec(reseeds=2))


@pytest.fixture(scope="module")
def grouped(tmp_path_factory: Any) -> Run:
    """Two rows per participant (`SEQN`), the seal and the refits by participant."""
    return _graph(tmp_path_factory.mktemp("grouped"), mf.nhanes_like(600, seed=2, repeats=2),
                  energy_adjustment=mf.energy("residual"), explain=ExplainSpec(reseeds=2),
                  grain=GrainSpec(grain="repeated", id_column="SEQN"),
                  repeat_kind=RepeatSpec(repeat_kind="repeats"), unit="row")


@pytest.fixture(scope="module")
def inference(tmp_path_factory: Any) -> Run:
    return _graph(tmp_path_factory.mktemp("inference"), mf.nhanes_like(500, seed=3),
                  purpose="inference", energy_adjustment=mf.energy("residual"),
                  explain=ExplainSpec(reseeds=2),
                  estimand=EstimandSpec(exposure="protein", measure="mean_difference",
                                        contrast="substitution"))


# ── by hand ──────────────────────────────────────────────────────────────────


def split(pipeline: Any) -> tuple[Any, Any]:
    """The fitted pipeline's adjusted-lane steps and the rest, by its own step names."""
    k = 0
    while pipeline.steps[k][0] in ADJUST:
        k += 1
    return (pipeline[:k] if k else None), pipeline[k:]


def inputs_of(pipeline: Any, X: pd.DataFrame) -> pd.DataFrame:
    adjust, _ = split(pipeline)
    return X if adjust is None else adjust.transform(X)


def owner(column: str, inputs: list[str]) -> str:
    """The input a matrix column came from: itself, or the categorical input it indicates."""
    if column in inputs:
        return column
    return max((a for a in inputs if column.startswith(f"{a}_")), key=len)


def by_input(phi: np.ndarray, columns: list[str], inputs: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=range(phi.shape[0]), columns=list(dict.fromkeys(
        owner(c, inputs) for c in columns)))
    for j, c in enumerate(columns):
        out[owner(c, inputs)] += phi[:, j]
    return out


def raw_scores(pipeline: Any, X: pd.DataFrame, task: str) -> np.ndarray:
    if task == "binary":
        return np.asarray(pipeline.decision_function(X), dtype=float)
    return np.asarray(pipeline.predict(X), dtype=float)


def listed_phi(family: dict[str, Any]) -> pd.DataFrame:
    obs = family["observations"]
    return pd.DataFrame(obs["phi"], columns=obs["inputs"])


def three_places(value: float) -> str:
    """A number as the methods paragraph prints it, written here: three decimals, a true minus."""
    return f"{value:.3f}".replace("-", "−")


def like_a_random_sample(share: float, k: int, p: float, N: int) -> bool:
    """Whether ``share`` of ``k`` rows drawn from ``N`` (a share ``p`` of them in a group) is within
    four standard deviations of ``p`` under drawing without replacement (hypergeometric)."""
    sd = math.sqrt(p * (1 - p) / k * (N - k) / (N - 1))
    return abs(share - p) <= 4 * sd


def interaction_truth(a: np.ndarray, b: np.ndarray, coefficient: float) -> float:
    """Simulation truth: the size of the interaction of ``coefficient · a · b`` at these rows.
    For f = c·a·b + (terms in one input), Friedman & Popescu's F_ab − F_a − F_b at row i is
    c (a_i − ā)(b_i − b̄) less its mean, so its root mean square is this."""
    part = coefficient * (a - a.mean()) * (b - b.mean())
    return float(np.sqrt(np.mean((part - part.mean()) ** 2)))


# ═════════════════════════════════════════════════════════════════════════════
# (1) SHAP
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("run_name, key", [("regression", "linear"), ("regression", "elastic_net"),
                                           ("binary", "linear"), ("binary", "elastic_net")])
def test_1a_linear_shap_is_the_closed_form_by_hand(run_name: str, key: str,
                                                   request: Any) -> None:
    """φ_j(x) = β_j (z_j − E[z_j]) with E over the rows the model was fit on (Lundberg & Lee
    2017), computed here from the fitted model's coefficients and matrix, summed over a
    category's indicators: agreement to 1e-10. shap's LinearExplainer gives the same."""
    import shap

    run: Run = request.getfixturevalue(run_name)
    family = run.family(key)
    pipe = run.fitted(key)
    ids = run.art["row_ids"][:len(family["observations"]["row_ids"])]
    assert family["observations"]["row_ids"] == ids
    X_obs, X_fit = run.rows(ids), run.rows(run.fit_ids())
    Z_obs, Z_fit = pipe[:-1].transform(X_obs), pipe[:-1].transform(X_fit)
    beta = np.ravel(pipe[-1].coef_)
    hand = (Z_obs.to_numpy() - Z_fit.to_numpy().mean(axis=0)) * beta
    inputs = list(inputs_of(pipe, X_obs).columns)
    expected = by_input(hand, [str(c) for c in Z_obs.columns], inputs)
    got = listed_phi(family)
    listed = family["observations"]["inputs"]
    assert np.max(np.abs(got.to_numpy() - expected[listed].to_numpy())) < 1e-10
    rest = expected.drop(columns=listed).sum(axis=1).to_numpy()
    assert np.max(np.abs(np.asarray(family["observations"]["rest"]) - rest)) < 1e-10
    assert abs(family["base"] - (np.ravel(pipe[-1].intercept_)[0]
                                 + Z_fit.to_numpy().mean(axis=0) @ beta)) < 1e-10
    explainer = shap.LinearExplainer((beta, float(np.ravel(pipe[-1].intercept_)[0])),
                                     shap.maskers.Independent(Z_fit.to_numpy(),
                                                              max_samples=len(Z_fit)))
    library = by_input(np.asarray(explainer.shap_values(Z_obs.to_numpy())),
                       [str(c) for c in Z_obs.columns], inputs)
    assert np.max(np.abs(got.to_numpy() - library[listed].to_numpy())) < 1e-10


@pytest.mark.parametrize("run_name", ["regression", "binary"])
def test_1b_tree_shap_agrees_with_shap_tree_explainer(run_name: str, request: Any) -> None:
    """Path-dependent TreeSHAP of the fitted boosted trees against shap's TreeExplainer on the
    same matrix: agreement to 1e-6 (the acceptance bound), the expected value too."""
    import shap

    run: Run = request.getfixturevalue(run_name)
    family = run.family("boosted_trees")
    pipe = run.fitted("boosted_trees")
    X_obs = run.rows(family["observations"]["row_ids"])
    Z = pipe[:-1].transform(X_obs)
    explainer = shap.TreeExplainer(pipe[-1])
    reference = by_input(np.asarray(explainer.shap_values(Z.to_numpy())),
                         [str(c) for c in Z.columns], list(inputs_of(pipe, X_obs).columns))
    listed = family["observations"]["inputs"]
    assert np.max(np.abs(listed_phi(family).to_numpy() - reference[listed].to_numpy())) < 1e-6
    assert abs(family["base"] - float(np.ravel(explainer.expected_value)[0])) < 1e-6


def test_1c_tree_shap_routes_blanks_as_the_trees_do_and_agrees_with_shap() -> None:
    """The engine's TreeSHAP on trees grown with blanks in their inputs (scikit-learn sends a
    blank where ``missing_go_to_left`` says), a regressor and a classifier, against shap."""
    import shap
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    rng = np.random.default_rng(3)
    X = rng.normal(size=(1200, 5))
    X[rng.random(X.shape) < 0.08] = np.nan
    f = np.nan_to_num(X[:, 0]) * 2 + np.sin(np.nan_to_num(X[:, 1])) \
        + np.nan_to_num(X[:, 2] * X[:, 3])
    y = f + rng.normal(size=len(X))
    for model, target in ((HistGradientBoostingRegressor(random_state=0), y),
                          (HistGradientBoostingClassifier(random_state=0), (y > 0.3).astype(int))):
        model.fit(X, target)
        phi, expected = E.tree_shap(E.hgb_ensemble(model), X[:500])
        explainer = shap.TreeExplainer(model)
        reference = np.asarray(explainer.shap_values(X[:500])).reshape(500, 5)
        assert np.max(np.abs(phi[:, :, 0] - reference)) < 1e-6
        assert abs(expected[0] - float(np.ravel(explainer.expected_value)[0])) < 1e-6


@pytest.mark.parametrize("run_name", ["regression", "binary"])
def test_1d_every_listed_row_adds_up_to_its_prediction(run_name: str, request: Any) -> None:
    """base + Σ φ + rest equals the fitted pipeline's own prediction (the log-odds for a yes/no
    outcome) for every listed row of every family, to 1e-8."""
    run: Run = request.getfixturevalue(run_name)
    for family in run.art["families"]:
        obs = family["observations"]
        X_obs = run.rows(obs["row_ids"])
        total = obs["base"] + np.sum(obs["phi"], axis=1) + np.asarray(obs["rest"])
        want = raw_scores(run.fitted(family["family"]), X_obs, run.art["task"])
        assert np.max(np.abs(total - want)) < 1e-8, family["family"]
        assert np.max(np.abs(np.asarray(obs["prediction"]) - want)) < 1e-8


def test_1e_beeswarm_points_are_the_rows_attributions_with_their_values_and_colors(regression):
    """Each input's beeswarm: the first rows' SHAP values (as the per-row list has them), the
    input's value, and its percentile among the explained rows (SciPy's average ranks)."""
    family = regression.family("boosted_trees")
    phi = listed_phi(family)
    pipe = regression.fitted("boosted_trees")
    ids = regression.art["row_ids"][:len(family["beeswarm"][0]["phi"])]
    A = inputs_of(pipe, regression.rows(ids))
    for point in family["beeswarm"]:
        a = point["input"]
        n = min(len(point["phi"]), len(phi))
        assert np.allclose(point["phi"][:n], phi[a].to_numpy()[:n], atol=1e-12, rtol=0)
        if pd.api.types.is_numeric_dtype(A[a]):
            x = A[a].to_numpy(dtype=float)
            assert np.allclose(point["value"], x, atol=1e-12)
            want = (stats.rankdata(x) - 1) / (len(x) - 1)
            assert np.allclose(point["color"], want, atol=1e-12)
        else:
            assert point["level"] == [str(v) for v in A[a]]
    # The inputs are listed by mean |SHAP|, largest first.
    order = [i["mean_abs"] for i in family["importance"]]
    assert order == sorted(order, reverse=True)


@pytest.mark.parametrize("key", ["linear", "elastic_net", "boosted_trees"])
def test_1f_stability_over_five_reseeds_is_spearman_of_the_reported_importances(regression, key):
    """Five refits on bootstrap resamples, each with its own seed; every reported ρ is SciPy's
    Spearman correlation of the reported importances (between refits, and against the fit)."""
    family = regression.family(key)
    st = family["stability"]
    assert st["reseeds"] == 5 and st["resampled_by"] == "row"
    imp = np.asarray(st["importance"])
    assert imp.shape == (5, len(st["inputs"]))
    k = 0
    for i in range(5):
        for j in range(i + 1, 5):
            assert abs(st["pairwise"][k] - stats.spearmanr(imp[i], imp[j]).statistic) < 1e-12
            k += 1
    assert abs(st["rho_mean"] - np.mean(st["pairwise"])) < 1e-12
    assert abs(st["rho_min"] - np.min(st["pairwise"])) < 1e-12
    main = {i["input"]: i["mean_abs"] for i in family["importance"]}
    fit = np.asarray([main[a] for a in st["inputs"]])
    for r in range(5):
        assert abs(st["versus_fit"][r] - stats.spearmanr(fit, imp[r]).statistic) < 1e-12
        ranks = stats.rankdata(-imp[r], method="ordinal")
        for a, rank in zip(st["inputs"], ranks):
            row = next(i for i in family["importance"] if i["input"] == a)
            assert row["reseed_ranks"][r] == int(rank)


def test_1g_a_known_ranking_of_true_effects_holds_across_the_reseeds():
    """Simulation truth: y = 3·x1 + 2·x2 + 1·x3 + noise with two inputs that carry nothing, the
    inputs independent and of unit variance. The mean |SHAP| of a linear effect is |β| E|x − x̄|,
    so the truth ranks x1, x2, x3 first, second and third; every refit keeps that, and with only
    the two null inputs free to swap, every Spearman ρ between refits is at least 0.9."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(11)
    n = 800
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=["x1", "x2", "x3", "n1", "n2"],
                     index=pd.RangeIndex(1000, 1000 + n))
    y = 3 * X["x1"] + 2 * X["x2"] + X["x3"] + rng.normal(size=n)
    pipe = Pipeline([("model", LinearRegression())]).set_output(transform="pandas")

    def refit(model: Any, X_b: pd.DataFrame, y_b: Any, units: Any) -> Any:
        return model.fit(X_b, y_b)

    fam = E.FamilyFit(key="linear", label="Linear model", fitted=refit(pipe, X, y.to_numpy(), None),
                      unfitted=Pipeline([("model", LinearRegression())]),
                      versus={"verdict": "better"}, score=0.9, baseline=0.0)
    art = E.explain([fam], E.Setting(task="regression", purpose="prediction", target="y",
                                     event=None, X=X, y=y.to_numpy()), refit)
    family = art.families[0]
    ranks = {i.input: i.reseed_ranks for i in family.importance}
    assert ranks["x1"] == [1] * 5 and ranks["x2"] == [2] * 5 and ranks["x3"] == [3] * 5
    assert family.stability.rho_min >= 0.9 - 1e-12


def test_1h_a_bootstrap_by_unit_takes_every_row_of_each_drawn_unit():
    """MODELING_SEQUENCE §2 (a bootstrap by unit): as many units drawn as there are, each drawn
    unit's rows all taken, each time it is drawn."""
    units = np.repeat(np.arange(40), np.random.default_rng(0).integers(1, 4, 40))
    idx = E.resample(len(units), 7, units)
    drawn = pd.Series(units[idx]).value_counts()
    sizes = pd.Series(units).value_counts()
    for unit, rows in drawn.items():
        assert rows % sizes[unit] == 0
    assert int((drawn / sizes[drawn.index]).sum()) == 40


def test_1i_rows_that_repeat_are_resampled_by_their_unit(grouped):
    for family in grouped.art["families"]:
        assert family["stability"]["resampled_by"] == "`SEQN`"
    assert E.UNITS_SAYS in grouped.art["relations"]


def test_1j_a_correlation_defined_for_only_some_pairs_of_refits_says_so():
    """Spearman's ρ has no value when a refit ranks nothing (every input's importance equal). A
    family whose ρ is defined for some pairs of refits reports the mean over those and says how
    many; one with none says it is undefined (never "n/a"), and the paragraph says why."""
    def family(key: str, label: str, importance: list[list[float]]) -> Any:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # SciPy warns of a constant input, and returns nan
            rho = [stats.spearmanr(importance[i], importance[j]).statistic
                   for i in range(3) for j in range(i + 1, 3)]
        pairwise = [None if np.isnan(r) else float(r) for r in rho]
        known = [p for p in pairwise if p is not None]
        return E.FamilyExplanation(
            family=key, label=label, explained=True, method="exact linear SHAP",
            floor=E.Floor(passed=True, verdict="better", metric="MSE", model=0.6, baseline=1.0,
                          reason=None),
            stability=E.Stability(reseeds=3, resampled_by="row", inputs=["a", "b", "c"],
                                  importance=importance, pairwise=pairwise, versus_fit=[],
                                  rho_mean=float(np.mean(known)) if known else None,
                                  rho_min=float(np.min(known)) if known else None, top=3))

    # The linear model's second refit predicts one value for every row; every elastic net refit
    # shrinks every coefficient to zero.
    linear = family("linear", "Linear model", [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [3.0, 2.0, 1.0]])
    assert linear.stability.pairwise == [None, -1.0, None]
    art = E.ExplainArtifact(
        purpose="prediction", task="regression", describes=E.DESCRIBES, rows=90, rows_of=90,
        rows_basis="", curve_method="ale", curves=[], methods="",
        families=[linear, family("elastic_net", "Elastic net", [[0.0] * 3] * 3)])
    setting = E.Setting(task="regression", purpose="prediction", target="y", event=None,
                        X=pd.DataFrame(), y=np.zeros(0))
    assert E.methods_sentence(art, setting) == (
        "SHAP values were computed for all 90 training rows on each model's own scale: exact "
        "linear SHAP values for the linear model and the elastic net. Their stability was measured "
        "over 3 refits of each model on bootstrap resamples of rows, each with its own seed: the "
        "mean Spearman correlation of the inputs' mean absolute SHAP values between refits was "
        "−1.000 for the linear model (over the 1 of its 3 pairs of refits for which it is "
        "defined) and undefined for the elastic net. A pair of refits has no such correlation "
        "when either gives every input the same mean absolute SHAP value, as a refit that "
        "predicts one value for every row does: there is no ranking to compare. These "
        "explanations describe each model's predictions, not causal effects.")


# ═════════════════════════════════════════════════════════════════════════════
# (2) interaction ranking
# ═════════════════════════════════════════════════════════════════════════════


def _h_by_hand(score: Any, A: pd.DataFrame, a: str, b: str) -> tuple[float, float]:
    """Friedman & Popescu (2008): F_S(x_i) = mean_l f(x_iS, x_l\\S), centered; H² and the RMS of
    F_ab − F_a − F_b; one row of the grid at a time."""
    m = len(A)

    def pd_at(columns: list[str]) -> np.ndarray:
        out = np.empty(m)
        for i in range(m):
            block = A.copy()
            for c in columns:
                block[c] = A[c].iloc[i]
            out[i] = np.mean(score(block))
        return out - out.mean()

    fa, fb, fab = pd_at([a]), pd_at([b]), pd_at([a, b])
    part = fab - fa - fb
    return float(np.sum(part ** 2) / np.sum(fab ** 2)), float(np.sqrt(np.mean(part ** 2)))


def test_2a_h_statistic_agrees_with_a_numpy_hand_computation(regression):
    """The boosted trees' top pairs: H² and the interaction's size from loops over the rows, the
    model scored by its own pipeline: agreement to 1e-8."""
    family = regression.family("boosted_trees")
    it = family["interactions"]
    pipe = regression.fitted("boosted_trees")
    _, rest = split(pipe)
    ids = regression.art["row_ids"][:it["rows"]]
    A = inputs_of(pipe, regression.rows(ids)).reset_index(drop=True)
    for pair in it["pairs"][:4]:
        h2, size = _h_by_hand(lambda frame: rest.predict(frame), A, pair["a"], pair["b"])
        assert abs(pair["h2"] - h2) < 1e-8 and abs(pair["strength"] - size) < 1e-8


def test_2b_h_statistic_of_known_functions():
    """f = x1·x2 + x3: the pair (x1, x2) interacts and the others do not (H² = 0 to rounding);
    the engine's values equal the loops' to 1e-12. An additive function interacts nowhere."""
    rng = np.random.default_rng(2)
    A = pd.DataFrame(rng.uniform(-1, 1, size=(60, 3)), columns=["x1", "x2", "x3"])

    def f(frame: pd.DataFrame) -> np.ndarray:
        return (frame["x1"] * frame["x2"] + frame["x3"]).to_numpy()

    found = {(h.a, h.b): h for h in E.h_statistics(f, A, ["x1", "x2", "x3"])}
    for (a, b), h in found.items():
        h2, size = _h_by_hand(f, A, a, b)
        assert abs(h.h2 - h2) < 1e-12 and abs(h.strength - size) < 1e-12
    assert found[("x1", "x2")].h2 > 0.5
    assert found[("x1", "x3")].strength < 1e-12 and found[("x2", "x3")].strength < 1e-12


def test_2c_the_top_interactions_are_ranked_with_their_stability(regression):
    """Ranked by size, largest first; each pair's size in each refit; how often a pair stays in
    the first three; ρ is SciPy's Spearman of the sizes, each refit against the fit."""
    it = regression.family("boosted_trees")["interactions"]
    sizes = [p["strength"] for p in it["pairs"]]
    assert sizes == sorted(sizes, reverse=True) and [p["rank"] for p in it["pairs"]] == \
        list(range(1, len(sizes) + 1))
    assert len(it["pairs"]) == len(it["inputs"]) * (len(it["inputs"]) - 1) // 2
    again = np.asarray([p["reseed_strength"] for p in it["pairs"]])
    assert again.shape == (len(sizes), 5)
    rhos = [stats.spearmanr(sizes, again[:, r]).statistic for r in range(5)]
    assert abs(it["rho_mean"] - np.mean(rhos)) < 1e-12
    for i, p in enumerate(it["pairs"]):
        within = sum(int(stats.rankdata(-again[:, r], method="ordinal")[i]) <= E.TOP_PAIRS
                     for r in range(5))
        assert p["in_top"] == within


def test_2d_a_model_that_adds_its_inputs_reports_no_interaction(regression):
    """The linear model and the elastic net have no product terms: every pair's interaction is
    zero, so none is ranked, and the methods say so."""
    for key in ("linear", "elastic_net"):
        it = regression.family(key)["interactions"]
        assert it["additive"] is True and it["pairs"] == []
    assert ("The linear model and elastic net add their inputs' effects, so no interaction was "
            "found to rank.") in regression.art["methods"]


def test_2e_an_export_sorted_by_the_interacting_input_keeps_its_interaction(sorted_by_sex):
    """Through the stage graph, a file sorted by sex then age with an age × sex interaction
    planted (simulation truth). The training rows in file order begin with H_ROWS women, so rows
    taken from the top of the file would hold one sex and no age × sex interaction at all. The
    H statistic's rows, the beeswarm's and the per-row list's are a random sample of the
    explained rows: each one's share of men is within four hypergeometric SDs of the training
    rows'. The boosted trees rank the planted pair first and keep it in the first three in every
    refit; its size is the fitted pipeline's own H at those rows (loops, to 1e-8) and at least
    40% of the true function's there (trees fit to 480 rows, the outcome's noise SD 8 mg/dL,
    recover about half; rows of one sex give 0)."""
    run = sorted_by_sex
    every = run.fit_ids()
    sexes = run.rows(every)["gender"]
    assert (sexes.iloc[:E.H_ROWS] == "female").all()  # the file's own order
    p, N = float((sexes == "male").mean()), len(every)
    family = run.family("boosted_trees")
    it = family["interactions"]
    order = run.art["row_ids"]
    assert sorted(order) == sorted(int(i) for i in every)
    listed = family["observations"]["row_ids"]
    for k in (it["rows"], len(family["beeswarm"][0]["phi"]), len(listed)):
        share = float((run.rows(order[:k])["gender"] == "male").mean())
        assert like_a_random_sample(share, k, p, N), (k, share, p)
    assert listed == order[:len(listed)]
    top = it["pairs"][0]
    assert {top["a"], top["b"]} == {"age", "gender"} and top["rank"] == 1
    assert top["in_top"] == 2
    raw = run.rows(order[:it["rows"]])
    truth = interaction_truth(raw["age"].to_numpy(dtype=float),
                              (raw["gender"] == "male").to_numpy(dtype=float), PLANTED)
    assert top["strength"] >= 0.4 * truth
    pipe = run.fitted("boosted_trees")
    _, rest = split(pipe)
    A = inputs_of(pipe, raw).reset_index(drop=True)
    h2, size = _h_by_hand(lambda frame: rest.predict(frame), A, top["a"], top["b"])
    assert abs(top["h2"] - h2) < 1e-8 and abs(top["strength"] - size) < 1e-8


def test_2f_the_interaction_ranking_does_not_depend_on_the_files_order():
    """Simulation truth: y = x + 2·x·s + 0.5·z + noise (SD 0.3), s a 0/1 indicator, 800 rows; the
    one interaction is x × s, whose size at any rows is ``interaction_truth(x, s, 2)``. The same
    rows explained from a shuffled file and from one sorted by s then x: in both, the H
    statistic's rows hold s = 1 in a share within four hypergeometric SDs of the file's, x × s
    ranks first with its size within 15% of the truth at those rows, and every other pair's size
    is under a fifth of it."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(12)
    n = 800
    X = pd.DataFrame({"x": rng.normal(size=n), "s": rng.integers(0, 2, n).astype(float),
                      "z": rng.normal(size=n)})
    y = (X["x"] + 2 * X["x"] * X["s"] + 0.5 * X["z"] + rng.normal(scale=0.3, size=n)).to_numpy()

    def refit(model: Any, X_b: pd.DataFrame, y_b: Any, units: Any) -> Any:
        return model.fit(X_b, y_b)

    def trees() -> Any:
        return Pipeline([("model", HistGradientBoostingRegressor(random_state=0))])

    p = float(X["s"].mean())
    for order in (rng.permutation(n), np.lexsort((X["x"].to_numpy(), X["s"].to_numpy()))):
        X_o = X.iloc[order].set_index(pd.RangeIndex(100, 100 + n))
        y_o = y[order]
        fam = E.FamilyFit(key="boosted_trees", label="Boosted trees",
                          fitted=refit(trees(), X_o, y_o, None), unfitted=trees(),
                          versus={"verdict": "better"}, score=0.9, baseline=0.0)
        art = E.explain([fam], E.Setting(task="regression", purpose="prediction", target="y",
                                         event=None, X=X_o, y=y_o, reseeds=2), refit)
        it = art.families[0].interactions
        rows = X_o.loc[art.row_ids[:it.rows]]
        assert like_a_random_sample(float(rows["s"].mean()), it.rows, p, n)
        top = it.pairs[0]
        assert {top.a, top.b} == {"x", "s"} and top.in_top == 2
        truth = interaction_truth(rows["x"].to_numpy(), rows["s"].to_numpy(), 2.0)
        assert abs(top.strength - truth) <= 0.15 * truth, (top.strength, truth)
        assert all(q.strength < 0.2 * top.strength for q in it.pairs[1:])


# ═════════════════════════════════════════════════════════════════════════════
# (3) inductive-bias curves
# ═════════════════════════════════════════════════════════════════════════════


def _grid_by_hand(x: np.ndarray, K: int = 20) -> np.ndarray:
    x = np.sort(x[np.isfinite(x)])
    distinct = np.unique(x)
    if len(distinct) <= K + 1:
        return distinct
    q = np.quantile(x, np.linspace(0, 1, K + 1), method="inverted_cdf")
    width = (distinct[-1] - distinct[0]) / K
    gaps = [i for i in range(len(distinct) - 1) if distinct[i + 1] - distinct[i] > width]
    return np.unique(np.concatenate([q, distinct[gaps], distinct[[g + 1 for g in gaps]]]))


def _ale_by_hand(score: Any, A: pd.DataFrame, column: str, z: np.ndarray) -> tuple[np.ndarray,
                                                                                    np.ndarray]:
    """Apley & Zhu (2020): row i belongs to the interval (z_{k−1}, z_k] holding it (z_0 to the
    first); the local effect of interval k is the mean of f(z_k, x_i\\j) − f(z_{k−1}, x_i\\j) over
    its rows; the curve accumulates them from z_0 and is centered by the mean over the rows of each
    row's accumulated value. Returns the curve at the edges and the rows per interval."""
    x = A[column].to_numpy(dtype=float)
    K = len(z) - 1
    k_of = np.empty(len(x), dtype=int)
    for i, v in enumerate(x):
        k_of[i] = 1 if v <= z[1] else next(k for k in range(1, K + 1) if z[k - 1] < v <= z[k])
    local = np.zeros(K + 1)
    counts = np.zeros(K + 1, dtype=int)
    for k in range(1, K + 1):
        rows = A.iloc[np.flatnonzero(k_of == k)]
        counts[k] = len(rows)
        if len(rows):
            up, down = rows.copy(), rows.copy()
            up[column], down[column] = z[k], z[k - 1]
            local[k] = np.mean(score(up) - score(down))
    g = np.cumsum(local)
    return g - np.mean(g[k_of]), counts[1:]


@pytest.mark.parametrize("run_name", ["regression", "binary"])
def test_3a_ale_agrees_with_a_numpy_hand_implementation(run_name: str, request: Any) -> None:
    """Every drawn curve, for every family, from the definition by loops: the grid (NumPy's
    inverse-CDF quantiles, gaps split) exactly, the counts exactly, the curve to 1e-10 wherever
    it is drawn, and no curve at an edge with no supported interval beside it."""
    run: Run = request.getfixturevalue(run_name)
    assert run.art["curves"], "no curve was drawn"
    for curve in run.art["curves"]:
        pipe0 = run.fitted(curve["curves"][0]["family"])
        A0 = inputs_of(pipe0, run.rows(run.art["row_ids"]))
        z = _grid_by_hand(A0[curve["input"]].to_numpy(dtype=float))
        assert np.array_equal(np.asarray(curve["grid"]), z)
        for drawn in curve["curves"]:
            if not drawn["drawn"]:
                continue
            pipe = run.fitted(drawn["family"])
            _, rest = split(pipe)
            A = inputs_of(pipe, run.rows(run.art["row_ids"]))
            values, counts = _ale_by_hand(lambda f: raw_scores(rest, f, run.art["task"]), A,
                                          curve["input"], z)
            assert counts.tolist() == curve["counts"]
            supported = counts >= E.MIN_SEGMENT_ROWS
            assert supported.tolist() == curve["supported"]
            for i, (got, want) in enumerate(zip(drawn["values"], values)):
                touches = (i > 0 and supported[i - 1]) or (i < len(counts) and supported[i])
                if touches:
                    assert abs(got - want) < 1e-10, (curve["input"], drawn["family"], i)
                else:
                    assert got is None


def test_3b_ale_of_a_known_additive_function_is_its_own_shape():
    """For f = 2·x1 + sin(x2) the ALE of x1 at the edges is 2·(z_k − mean_i z_{k(i)}) and of x2
    sin(z_k) − mean_i sin(z_{k(i)}), whatever the other input does (x2 rises with x1 here); the
    partial dependence of an additive function is the same curve."""
    rng = np.random.default_rng(9)
    x1 = rng.normal(size=500)
    A = pd.DataFrame({"x1": x1, "x2": 0.8 * x1 + rng.normal(scale=0.6, size=500)})

    def f(frame: pd.DataFrame) -> np.ndarray:
        return (2 * frame["x1"] + np.sin(frame["x2"])).to_numpy()

    for column, g in (("x1", lambda z: 2 * z), ("x2", np.sin)):
        grid = E.ale_grid(A[column])
        seg = grid.segment_of(A[column].to_numpy())
        truth = g(grid.edges) - np.mean(g(grid.edges)[seg])
        assert np.max(np.abs(E.ale_curve(f, A, column, grid) - truth)) < 1e-12
        assert np.max(np.abs(E.pd_curve(f, A, column, grid) - truth)) < 1e-12


def test_3c_curves_share_one_grid_per_input_and_mask_where_data_are_absent():
    """One grid per input for every family (shared axes). A gap in the data (no value between 3
    and 9 here) is split out, so no segment spans it, and a stretch with fewer than five rows
    draws no curve."""
    x = np.concatenate([np.linspace(0, 3, 200), np.linspace(9, 10, 200), [14.0, 15.0]])
    grid = E.ale_grid(x)
    assert 3.0 in grid.edges and 9.0 in grid.edges
    gap = int(np.flatnonzero(grid.edges == 9.0)[0])
    assert grid.counts[gap - 1] == 1 and not grid.supported[gap - 1]  # (3, 9]: only 9 itself
    curve = E.masked(np.arange(len(grid.edges), dtype=float), grid)
    assert curve[-1] is None  # 14 and 15: two rows, past a gap
    assert all(v is not None for v in curve[:gap])


def test_3d_a_family_below_the_floor_draws_no_curve_and_says_why(noise):
    """No family beats the outcome's average on an outcome no predictor carries: every curve is
    withheld with its family's reason, quoted verbatim, and no interaction is ranked. The whole
    methods paragraph, verbatim: it says no curve was drawn (never that one was drawn for each
    family), and the elastic net, one of whose refits shrinks every coefficient to zero, has an
    undefined correlation between refits, said in words. The correlations are SciPy's Spearman
    of the reported importances, the training rows the design's own count."""
    import warnings

    assert noise.art["curves"]
    for curve in noise.art["curves"]:
        for drawn in curve["curves"]:
            assert drawn["drawn"] is False and drawn["values"] == []
            family = noise.family(drawn["family"])
            assert drawn["reason"] == family["floor"]["reason"]
            assert family["floor"]["passed"] is False
    linear = noise.family("linear")["floor"]
    how = "is worse than" if linear["verdict"] == "worse" else "is not shown to beat"
    # MS6 (wave 1b): a regression's primary is the MSE, a strictly proper score, lower better.
    assert linear["metric"] == "MSE"
    if linear["verdict"] == "worse":
        assert linear["model"] > linear["baseline"]
    assert linear["reason"] == (
        f"Linear model draws no curve: its cross-validated MSE of "
        f"{three_places(linear['model'])} {how} the MSE of the outcome's average, "
        f"{three_places(linear['baseline'])}, so a curve would describe noise.")
    assert all(f["interactions"] is None for f in noise.art["families"])
    rho = {}
    for family in noise.art["families"]:
        imp = np.asarray(family["stability"]["importance"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # SciPy warns of a constant input, and returns nan
            rho[family["family"]] = float(stats.spearmanr(imp[0], imp[1]).statistic)
    # One of the elastic net's two refits shrinks every coefficient to zero, so predicts one value
    # for every row: all its importances are zero, and the one pair of refits ranks nothing.
    assert any(not row.any() for row in np.asarray(
        noise.family("elastic_net")["stability"]["importance"]))
    assert np.isnan(rho["elastic_net"]) and not np.isnan(rho["linear"]) \
        and not np.isnan(rho["boosted_trees"])
    rows = len(noise.fit_ids())
    assert noise.art["methods"] == (
        f"SHAP values were computed for all {rows} training rows on each model's own scale: "
        "exact linear SHAP values for the linear model and the elastic net; path-dependent "
        "TreeSHAP values for the boosted trees. Their stability was measured over 2 refits of "
        "each model on bootstrap resamples of rows, each with its own seed: the mean Spearman "
        "correlation of the inputs' mean absolute SHAP values between refits was "
        f"{three_places(rho['linear'])} for the linear model, undefined for the elastic net and "
        f"{three_places(rho['boosted_trees'])} for the boosted trees. A pair of refits has no "
        "such correlation when either gives every input the same mean absolute SHAP value, as a "
        "refit that predicts one value for every row does: there is no ranking to compare. No "
        "curve was drawn and no interaction was ranked for the linear model, the elastic net and "
        "the boosted trees, whose cross-validated MSE was not shown to beat the outcome's average. "
        "These explanations describe each model's predictions, not causal effects.")
    assert "n/a" not in noise.art["methods"] and "drawn for" not in noise.art["methods"]


def test_3e_the_floor_reads_the_fits_verdict_against_the_baseline(regression):
    """The floor is the fit stage's own paired comparison with the baseline on the same folds."""
    by_key = {m["family"]: m for m in regression.out["fit"].data["models"]}
    for family in regression.art["families"]:
        verdict = by_key[family["family"]]["versus_baseline"]["verdict"]
        assert family["floor"]["verdict"] == verdict
        assert family["floor"]["passed"] == (verdict == "better")


def test_3f_the_top_exposures_are_the_exposures_the_best_family_leans_on(regression):
    """Under prediction the curves are of the exposures (by role), on their final scale, ordered
    by mean |SHAP| in the family with the best cross-validated score."""
    fit = regression.out["fit"].data
    passing = [m for m in fit["models"] if m["versus_baseline"]["verdict"] == "better"]
    best = max(passing, key=lambda m: m["cv"][fit["primary_metric"]]["estimate"])
    family = regression.family(best["family"])
    weights = {i["input"]: i["mean_abs"] for i in family["importance"]}
    shown = [c["input"] for c in regression.art["curves"]]
    exposures = [f"{n}_adj" if n in ("protein", "carb", "fat_total") else n for n in mf.NUTRIENTS]
    want = sorted(exposures, key=lambda a: -weights.get(a, 0.0))[:E.TOP_EXPOSURES]
    assert shown == want
    assert E.SCALE_SAYS in regression.art["relations"]


def _two_families(purpose: str):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(21)
    n = 300
    X = pd.DataFrame({"protein": rng.normal(70, 15, n), "age": rng.uniform(20, 80, n)},
                     index=pd.RangeIndex(5000, 5000 + n))
    y = (0.3 * X["protein"] + 0.2 * X["age"] + rng.normal(0, 5, n)).to_numpy()
    linear = Pipeline([("model", LinearRegression())]).fit(X, y)
    trees = Pipeline([("model", HistGradientBoostingRegressor(random_state=0))]).fit(X, y)
    families = [
        E.FamilyFit(key="linear", label="Linear model", fitted=linear, unfitted=linear,
                    versus={"verdict": "better"}, score=0.62, baseline=0.0),
        E.FamilyFit(key="boosted_trees", label="Boosted trees", fitted=trees, unfitted=trees,
                    versus={"verdict": "no_better"}, score=24.92, baseline=25.01)]
    return E.explain(families, E.Setting(
        task="regression", purpose=purpose, target="glucose", event=None, X=X, y=y,
        declared=["protein"] if purpose == "inference" else [], exposures=["protein", "age"],
        reseeds=0, metric_label="MSE", baseline_label="the outcome's average"), lambda *a: None)


def test_3g_curves_drawn_by_some_families_name_only_those():
    """Under prediction, a linear model above the floor and boosted trees not shown to beat it
    (MSE 24.92 against 25.01, MS6's primary for a regression: below the baseline's number, lower
    being better, yet not shown to beat it): the paragraph names the linear model as the one that
    drew curves and says the trees drew none and ranked no interaction. Whole paragraph, verbatim;
    the trees' curves are withheld with their reason."""
    art = _two_families("prediction")
    assert [(c.input, c.role) for c in art.curves] == [("protein", "predictor"),
                                                        ("age", "predictor")]
    for curve in art.curves:
        drawn = {k.family: k for k in curve.curves}
        assert drawn["linear"].drawn and not drawn["boosted_trees"].drawn
        assert drawn["boosted_trees"].reason == (
            "Boosted trees draws no curve: its cross-validated MSE of 24.920 is not shown to beat "
            "the MSE of the outcome's average, 25.010, so a curve would describe noise.")
    assert art.methods == (
        "SHAP values were computed for all 300 training rows on each model's own scale: exact "
        "linear SHAP values for the linear model; path-dependent TreeSHAP values for the boosted "
        "trees. The linear model adds its inputs' effects, so no interaction was found to rank. "
        "Accumulated local effects (Apley and Zhu 2020) of `protein` and `age` were drawn for the "
        "linear model on one grid of the inputs' quantiles, with no curve where an interval held "
        "fewer than 5 rows. No curve was drawn and no interaction was ranked for the boosted "
        "trees, whose cross-validated MSE was not shown to beat the outcome's average. These "
        "explanations describe each model's predictions, not causal effects.")


def test_3g_under_inference_no_floor_reads_or_quotes_a_cross_validated_score():
    """MODELING_SEQUENCE ruling 13 and §1 row 11 (EXPLORE repair): under inference no
    cross-validated score is shown, so the explanation reads none: no family carries a floor, the
    declared exposure's curve is drawn for every family whatever its score would have been, the
    floor's relation does not fire, and the paragraph never mentions a cross-validated score.
    Whole paragraph, verbatim."""
    art = _two_families("inference")
    assert all(f.floor is None for f in art.families)
    assert E.FLOOR_SAYS not in art.relations
    assert [(c.input, c.role) for c in art.curves] == [("protein", "exposure"),
                                                        ("age", "adjustment")]
    for curve in art.curves:
        assert all(k.drawn and k.reason is None for k in curve.curves)
    assert "cross-validated" not in art.methods
    assert art.methods == (
        "SHAP values were computed for all 300 analyzed rows on each model's own scale: exact "
        "linear SHAP values for the linear model; path-dependent TreeSHAP values for the boosted "
        "trees. Pairwise interactions among each model's 2 most important inputs were ranked by "
        "the root mean square of their interaction part, with Friedman and Popescu's H² beside it, "
        "on a seeded random sample of 150 of the 300 rows. The linear model adds its inputs' "
        "effects, so no interaction "
        "was found to rank. Accumulated local effects (Apley and Zhu 2020) of `protein` and `age` "
        "were drawn for each family on one grid of the inputs' quantiles, with no curve where an "
        "interval held fewer than 5 rows. `age` is an adjustment term, not an effect estimate: its "
        "curve describes the models. These explanations describe each model's predictions, not "
        "causal effects. Under inference they were not used as effect estimates.")
    from turbotab.core.models.selection import explained_in

    assert explained_in(art.model_dump(mode="json")) == []


# ═════════════════════════════════════════════════════════════════════════════
# (4) the architecture lane
# ═════════════════════════════════════════════════════════════════════════════


def run_r(script: str, frames: dict[str, pd.DataFrame], folder: Path) -> Any:
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
    (folder / "reference.R").write_text(
        "suppressMessages({library(jsonlite); library(glmnet)})\n" + script)
    done = subprocess.run([RSCRIPT, "--vanilla", "reference.R"], cwd=folder, capture_output=True,
                          text=True, timeout=600)
    if done.returncode:
        raise RuntimeError(done.stderr[-3000:])
    return json.loads(done.stdout.strip().splitlines()[-1])


@needs_r
def test_4a_the_linear_equation_is_r_lm_with_settled_units(regression, tmp_path):
    """Every coefficient of the fitted equation, and its intercept, against R's lm on the same
    matrix: agreement to 1e-8. Units appear only where recorded (age in years, the outcome in
    mg/dL); every other column is quoted as its header."""
    family = regression.family("linear")
    eq = family["architecture"]["equation"]
    pipe = regression.fitted("linear")
    Z = pipe[:-1].transform(regression.rows(regression.fit_ids()))
    names = [f"v{i}" for i in range(Z.shape[1])]
    frame = pd.DataFrame(Z.to_numpy(), columns=names)
    frame["y"] = regression.y_fit()
    r = run_r("d <- read.csv('m.csv'); f <- lm(y ~ ., data = d)\n"
              "cat(toJSON(as.numeric(coef(f)), digits = NA))", {"m": frame}, tmp_path)
    coef = dict(zip([str(c) for c in Z.columns], r[1:]))
    assert abs(eq["intercept"] - r[0]) < 1e-8 * max(1.0, abs(r[0]))
    for term in eq["terms"]:
        assert abs(term["coefficient"] - coef[term["column"]]) < 1e-8, term["column"]
    age = next(t for t in eq["terms"] if t["column"] == "age")
    assert age["unit"] == "years" and age["coefficient_unit"] == "mg/dL per year"
    assert eq["outcome_unit"] == "mg/dL"
    assert eq["text"].startswith("`glucose` (mg/dL) = ")
    assert "× `age` (years)" in eq["text"]
    assert all(t["unit"] is None for t in eq["terms"] if t["column"] not in ("age", "kcal"))
    gender = next(t for t in eq["terms"] if t["input"] == "gender")
    assert gender["kind"] == "indicator" and "[`gender` = `male`]" in eq["text"]


def test_4b_the_tree_structure_is_read_off_the_fitted_trees(regression):
    """The first tree's top three levels and the split counts near the root, against the fitted
    model's own nodes."""
    trees = regression.family("boosted_trees")["architecture"]["trees"]
    pipe = regression.fitted("boosted_trees")
    model = pipe[-1]
    columns = [str(c) for c in model.feature_names_in_]
    nodes = model._predictors[0][0].nodes
    assert trees["n_trees"] == len(model._predictors)
    for node in trees["first_tree"]:
        raw = nodes[node["id"]]
        assert node["depth"] == int(raw["depth"]) < 3 and node["n"] == int(raw["count"])
        if bool(raw["is_leaf"]):
            assert node["value"] == pytest.approx(float(raw["value"]), abs=0)
        else:
            assert node["column"] == columns[int(raw["feature_idx"])]
            assert node["threshold"] == float(raw["num_threshold"])
            assert node["blanks"] == ("left" if raw["missing_go_to_left"] else "right")
    assert {n["id"] for n in trees["first_tree"]} == {
        i for i in range(len(nodes)) if int(nodes[i]["depth"]) < 3}
    roots: dict[str, int] = {}
    for predictor in model._predictors:
        root = predictor[0].nodes[0]
        name = columns[int(root["feature_idx"])]
        name = "gender" if name.startswith("gender_") else name
        roots[name] = roots.get(name, 0) + 1
    assert {s["input"]: s["root"] for s in trees["splits"] if s["root"]} == roots


@needs_r
def test_4c_the_elastic_net_path_is_glmnets(regression, tmp_path):
    """The shrinkage path at the chosen mix, on the standardized columns the elastic net was fit
    on, against R's glmnet at every penalty: agreement to 1e-6. glmnet scales a Gaussian response
    by its SD s, so scikit-learn's (α, ρ) is glmnet's λ = α(ρ + (1 − ρ)s), alpha = αρ/λ. The chosen
    penalty is marked and its coefficients are the fitted model's, to the fit's own tolerance."""
    path = regression.family("elastic_net")["architecture"]["path"]
    pipe = regression.fitted("elastic_net")
    Z = pipe[:-1].transform(regression.rows(regression.fit_ids()))
    y = regression.y_fit()
    s, rho = float(np.std(y)), path["l1_ratio"]
    names = [f"v{i}" for i in range(Z.shape[1])]
    frame = pd.DataFrame(Z.to_numpy(), columns=names)
    frame["y"] = y
    lam = [a * (rho + (1 - rho) * s) for a in path["penalties"]]
    alpha = [a * rho / lg for a, lg in zip(path["penalties"], lam)]
    script = ("d <- read.csv('m.csv'); x <- as.matrix(d[, setdiff(names(d), 'y')]); y <- d$y\n"
              f"lam <- c({','.join(repr(float(v)) for v in lam)})\n"
              f"alp <- c({','.join(repr(float(v)) for v in alpha)})\n"
              "out <- sapply(seq_along(lam), function(i) as.numeric(as.matrix(glmnet(x, y, "
              "family = 'gaussian', alpha = alp[i], lambda = lam[i], standardize = FALSE, "
              "intercept = TRUE, control = list(thresh = 1e-22, maxit = 1e8))$beta)))\n"
              "cat(toJSON(out, digits = NA))")
    beta = np.asarray(run_r(script, {"m": frame}, tmp_path))  # columns × penalties
    columns = [str(c) for c in Z.columns]
    for line in path["lines"]:
        want = beta[columns.index(line["column"])]
        assert np.max(np.abs(np.asarray(line["coefficients"]) - want)) < 1e-6, line["column"]
    assert path["chosen"] == pytest.approx(float(pipe[-1].alpha_), abs=0)
    at = path["chosen_index"]
    assert path["penalties"][at] == path["chosen"]
    fitted = dict(zip(columns, np.ravel(pipe[-1].coef_)))
    for line in path["lines"]:
        assert line["coefficients"][at] == pytest.approx(fitted[line["column"]], abs=1e-3)
    assert path["nonzero"] == [int(v) for v in (np.abs(beta) > 0).sum(axis=0)]


@needs_r
def test_4c_the_penalized_logistic_path_is_glmnets(binary, tmp_path):
    """A yes/no outcome's penalized logistic path against R's glmnet (binomial, which does not
    scale the response): scikit-learn's C on n rows is glmnet's λ = 1/(C n) at the same mix;
    agreement to 1e-6 on the log-odds scale."""
    path = binary.family("elastic_net")["architecture"]["path"]
    assert path["penalty_name"] == "C"
    pipe = binary.fitted("elastic_net")
    Z = pipe[:-1].transform(binary.rows(binary.fit_ids()))
    from turbotab.core.stages.modeling import coded_outcome

    from turbotab.core.datastore import DataStore

    with DataStore(binary.graph.raw, 1 << 30) as store:
        raw = store.materialize(["high"], binary.fit_ids())["high"].to_numpy()
    y = np.asarray(coded_outcome("binary", raw, "yes"), dtype=float)
    frame = pd.DataFrame(Z.to_numpy(), columns=[f"v{i}" for i in range(Z.shape[1])])
    frame["y"] = y
    lam = [1.0 / (c * len(y)) for c in path["penalties"]]
    script = ("d <- read.csv('m.csv'); x <- as.matrix(d[, setdiff(names(d), 'y')]); y <- d$y\n"
              f"lam <- c({','.join(repr(float(v)) for v in lam)})\n"
              "out <- sapply(lam, function(l) as.numeric(as.matrix(glmnet(x, y, "
              f"family = 'binomial', alpha = {path['l1_ratio']!r}, lambda = l, "
              "standardize = FALSE, control = list(thresh = 1e-22, maxit = 1e8))$beta)))\n"
              "cat(toJSON(out, digits = NA))")
    beta = np.asarray(run_r(script, {"m": frame}, tmp_path))
    columns = [str(c) for c in Z.columns]
    for line in path["lines"]:
        want = beta[columns.index(line["column"])]
        assert np.max(np.abs(np.asarray(line["coefficients"]) - want)) < 1e-6, line["column"]


def test_4e_a_wide_table_is_explained_within_its_bounds():
    """5,000 columns: the explained rows are capped so rows × inputs stays within CELLS, and the
    shrinkage path runs from the strongest penalty down to the chosen one."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(8)
    n, p = 800, 5000
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"g{i}" for i in range(p)])
    y = X.iloc[:, :5].to_numpy() @ np.array([2.0, -1.0, 1.0, 0.5, 0.5]) + rng.normal(size=n)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", ElasticNetCV(l1_ratio=[0.5, 1.0], cv=3, max_iter=5000))]
                    ).set_output(transform="pandas").fit(X, y)
    fam = E.FamilyFit(key="elastic_net", label="Elastic net", fitted=pipe, unfitted=pipe,
                      versus={"verdict": "better"}, score=0.8, baseline=0.0)
    art = E.explain([fam], E.Setting(task="regression", purpose="prediction", target="y",
                                     event=None, X=X, y=y, reseeds=0, candidates=["g0", "g1"]),
                    lambda *a: None)
    assert art.rows == E.CELLS // p == 400
    path = art.families[0].architecture.path
    assert all(v >= path.chosen for v in path.penalties) and path.penalties[-1] == path.chosen
    assert [c.input for c in art.curves] == ["g0", "g1"]


def test_4d_the_elastic_nets_equation_lists_what_it_kept(regression):
    eq = regression.family("elastic_net")["architecture"]["equation"]
    pipe = regression.fitted("elastic_net")
    zeros = int(np.sum(np.ravel(pipe[-1].coef_) == 0))
    assert eq["zeros"] == zeros and all(t["coefficient"] != 0 for t in eq["terms"])
    if zeros:
        assert eq["text"].endswith(f"; {zeros} coefficient{'s' if zeros != 1 else ''} shrunk "
                                   f"to zero")


# ═════════════════════════════════════════════════════════════════════════════
# (5) the leash
# ═════════════════════════════════════════════════════════════════════════════


def test_5a_every_explanation_says_it_describes_the_model_not_an_effect(regression, binary):
    for run in (regression, binary):
        assert run.art["describes"] == ("These explanations describe how each fitted model turns "
                                        "its inputs into predictions. They are not causal effects.")
        assert run.art["methods"].endswith(
            "These explanations describe each model's predictions, not causal effects.")
        assert run.art["under_inference"] is None


def test_5b_under_inference_they_are_never_offered_as_effect_estimates(inference):
    """The declared exposure's curve only; every covariate an adjustment term; the effect stays
    the coefficient table's, untouched by the explanations."""
    art = inference.art
    assert art["under_inference"] == (
        "Under inference the effect is the declared estimate in the coefficient table. These "
        "explanations describe each model and are never offered as effect estimates.")
    assert art["adjustment_terms"] == "adjustment term, not an effect estimate"
    assert [c["input"] for c in art["curves"]] == ["protein_adj"]
    assert art["curves"][0]["role"] == "exposure"
    for family in art["families"]:
        for item in family["importance"]:
            assert item["role"] == ("exposure" if item["input"] == "protein_adj" else "adjustment")
        assert "effect" not in family and "estimate" not in family
    assert art["methods"].endswith("Under inference they were not used as effect estimates.")
    for says in (E.LOCK_SAYS, E.TERMS_SAYS):
        assert says in art["relations"]
    # The fit's coefficient table is the fit stage's own: the explanations add nothing to it.
    assert all("explain" not in m for m in inference.out["fit"].data["models"])


def test_5c_reporting_an_explanation_as_an_effect_is_refused_under_both_purposes():
    for purpose, code_exit in (("inference", "Report the declared estimate from the coefficient "
                                             "table"),
                               ("prediction", "Declare an inference analysis with an exposure and "
                                              "its effect")):
        state = mf.state(purpose=purpose)
        with pytest.raises(d.Refusal) as refused:
            d.validate({"kind": "set_explain", "as_effect": True}, {"state": state})
        assert refused.value.code == "explanation_is_not_an_effect"
        labels = [e["label"] for e in refused.value.exits]
        assert labels == ["Keep the explanations, described as the models'", code_exit]
        keep = refused.value.exits[0]["decision"]
        assert keep["kind"] == "set_explain" and keep["as_effect"] is False
        d.validate(keep, {"state": state})  # the way forward is accepted
    inference_message = (
        "Under inference the effect is the declared estimate in the coefficient table; an "
        "explanation describes the fitted model's predictions and is never reported as an effect "
        "estimate (Molnar et al. 2022, xxAI, LNCS 13200:39–68: \"making unjustified causal "
        "interpretations\").")
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_explain", "as_effect": True},
                   {"state": mf.state(purpose="inference")})
    assert refused.value.message == inference_message


def test_5d_under_inference_explanations_wait_for_the_plan_and_lock_it(inference):
    """An estimate stage: withheld (every family and curve removed, the reason first) while a
    question the estimate rests on is open; served, it puts what the models learned in view."""
    from turbotab.core import estimand, plan_lock

    assert "explain" in estimand.ESTIMATE_STAGES and "explain" in plan_lock.ESTIMATE_STAGES
    art = dict(inference.art)
    assert plan_lock.shows_estimates("explain", art)
    gate = {"question": "estimand", "reason": "No estimate is shown until the exposure is answered."}
    held = estimand.withhold("explain", art, gate)
    assert held["withheld"] == gate["reason"] and held["families"] == [] and held["curves"] == []
    assert not plan_lock.shows_estimates("explain", held)
    assert "explain" in plan_lock.plan_slots()


def test_5e_multiple_imputation_under_inference_is_said_and_estimates_nothing():
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["protein", "age"])
    y = X["protein"] + rng.normal(size=200)
    pipe = Pipeline([("model", LinearRegression())]).fit(X, y)
    fam = E.FamilyFit(key="linear", label="Linear model", fitted=pipe, unfitted=pipe,
                      versus={"verdict": "better"}, score=0.5, baseline=0.0)
    art = E.explain([fam], E.Setting(task="regression", purpose="inference", target="y",
                                     event=None, X=X, y=y.to_numpy(), declared=["protein"],
                                     reseeds=0, multiple_imputation=True),
                    lambda *a: None)
    assert art.single_fill == (
        "They describe each model as fitted on a single in-fold fill of the missing values, not "
        "the analysis pooled over the multiple imputations, and estimate nothing.")
    assert E.FILL_SAYS in art.relations


# ═════════════════════════════════════════════════════════════════════════════
# the method contract (BLUEPRINT §13), the sentences, and the chain
# ═════════════════════════════════════════════════════════════════════════════


def test_6a_the_contract_declares_every_part_of_section_13():
    from turbotab.core.contracts import CONTRACTS, options_for

    c = CONTRACTS["explain"]
    assert c.slot == "evaluation" and c.scope == "model" and c.needs and c.question
    assert c.storyboard and c.sources and c.clause({"explain_methods": "x."}) == "x."
    for purpose in ("prediction", "inference"):
        options = options_for("explain", purpose)
        assert [o["key"] for o in options] == ["ale", "partial_dependence", "as_effect"]
        assert [o["rung"] for o in options] == ["recommended", "rank_lower", "refused"]
        assert all(o["customary"] and o["sound"] for o in options)
    kinds = {(r.kind, r.target) for r in c.relations}
    assert ("conflicts", "effect_estimate") in kinds and ("implies", "performance_floor") in kinds


def test_6b_the_declared_scope_is_the_one_the_scope_test_observes():
    """Lockbox constitution §06, by perturbation: row i's attributions move with the outcome, so
    the scope is the model's."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    from turbotab.core.contracts import observed_scope

    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    y = frame["a"] * 2 + rng.normal(size=80)

    def fit_transform(f: pd.DataFrame, reference: Any, yy: Any) -> pd.DataFrame:
        pipe = Pipeline([("model", LinearRegression())]).set_output(transform="pandas")
        pipe.fit(f, yy)
        anat = E.anatomy(pipe, list(f.columns), "regression")
        return E.attributions(anat, f, f)[0]

    assert observed_scope(fit_transform, frame, np.zeros(80, dtype=bool), y.to_numpy(),
                          row=3) == "model"


def test_6c_the_decision_sentence_is_verbatim():
    from turbotab.core.voice import sentence_for

    assert sentence_for(d.SetExplain(), mf.state()) == (
        "The fitted models were described by their SHAP values, with their stability over 5 refits "
        "on bootstrap resamples, by a ranking of pairwise interactions (Friedman and Popescu's H "
        "statistic), and by accumulated local effects curves (Apley and Zhu 2020) of the top "
        "exposures, drawn only for families whose cross-validated score beats the no-predictor "
        "baseline; these describe the models' predictions, not causal effects.")
    assert sentence_for(d.SetExplain(curves="partial_dependence", exposures=["protein"],
                                     reseeds=0), mf.state(purpose="inference")) == (
        "The fitted models were described by their SHAP values, without a check of their "
        "stability across refits, by a ranking of pairwise interactions (Friedman and Popescu's H "
        "statistic), and by partial dependence curves (Friedman 2001), the customary choice, "
        "which average over combinations of the inputs the data may not contain, of `protein`, "
        "drawn for every fitted family; these describe the models' predictions, not causal "
        "effects, and are not effect estimates.")


def test_6d_one_refit_or_a_column_outside_the_model_is_refused_with_ways_forward():
    state = mf.state()
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_explain", "reseeds": 1}, {"state": state})
    assert refused.value.code == "one_refit"
    assert [e["decision"]["reseeds"] for e in refused.value.exits] == [5, 0]
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_explain", "exposures": ["SEQN"]}, {"state": state})
    assert refused.value.code == "not_a_predictor"
    assert refused.value.exits[0]["decision"]["exposures"] == []


def test_6e_the_chain_fires_every_relation_it_declares(grouped, inference):
    """MODELING_SEQUENCE §6: the relations the contract declares fire where the run shows them.
    Repeated units with the residual energy model under prediction: the floor, the resampling by
    unit and the transformed scale; under inference also the wait-and-lock and the adjustment
    terms. Each says what the artifact shows."""
    from turbotab.core.contracts import fired

    present = {"performance_floor", "unit_resampling", "domain_transform"}
    says = [f.says for f in fired({"explain": "ale"}, "prediction", sorted(present))]
    assert says == grouped.art["relations"]
    assert all(f["stability"]["resampled_by"] == "`SEQN`" for f in grouped.art["families"])
    # The transformed scale fired because a curve is of an energy-adjusted intake: made from the
    # raw nutrient and total energy, drawn on the scale the model reads.
    adjusted = [c for c in grouped.art["curves"] if c["input"].endswith("_adj")]
    assert adjusted and all(set(c["sources"]) == {c["input"][:-4], "kcal"} and c["unit"] is None
                            for c in adjusted)
    later = {"performance_floor", "domain_transform", "plan_lock", "adjustment_terms"}
    says = [f.says for f in fired({"explain": "ale"}, "inference", sorted(later))]
    assert says == inference.art["relations"]
    assert grouped.art["methods"] == (
        "SHAP values were computed for all 480 training rows on each model's own scale: exact "
        "linear SHAP values for the linear model and the elastic net; path-dependent TreeSHAP "
        "values for the boosted trees. Their stability was measured over 2 refits of each model on "
        "bootstrap resamples of whole `SEQN` units, each with its own seed: the mean Spearman "
        "correlation of the inputs' mean absolute SHAP values between refits was "
        + E._listing([f"{E._fmt(f['stability']['rho_mean'])} for the {f['label'].lower()}"
                      for f in grouped.art["families"]])
        + ". Pairwise interactions among each model's 6 most important inputs were ranked by the "
        "root mean square of their interaction part, with Friedman and Popescu's H² beside it, on "
        "a seeded random sample of 150 of the 480 rows. The linear model and elastic net add their "
        "inputs' effects, so no interaction was found to rank. Accumulated local effects (Apley "
        "and Zhu 2020) of "
        + E._listing([f"`{c['input']}`" for c in grouped.art["curves"]])
        + " were drawn for each family on one grid of the inputs' quantiles, with no curve where "
        "an interval held fewer than 5 rows. These explanations describe each model's "
        "predictions, not causal effects.")
