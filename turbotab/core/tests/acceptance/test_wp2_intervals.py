"""WP2 · Intervals that match how the data were sampled (docs/turbotab-next/audit/AUDIT_REPORT.md
§5; closes MA-01, MA-06, MA-07, MA-08 and the minor A18).

The six acceptance tests of the package, in its order. Every reference is computed by a path
independent of ``turbotab/core/models/inference.py``: statsmodels' own estimators, closed forms from
the literature, or a direct implementation of the published definition (``references.py``).
Monte Carlo runs are seeded, so they are deterministic; each states its Monte Carlo standard error
against the bound it checks.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core.decisions import GrainSpec, ProjectState, SplitSpec
from turbotab.core.models.inference import (
    FEW_CLUSTERS,
    INDEPENDENT,
    Clusters,
    inference_table,
    min_clusters,
    resolve_clusters,
)
from turbotab.core.stages.modeling import design_stage, fit_stage
from turbotab.core.stages.rows import split_stage
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import references as ref
from turbotab.core.tests.acceptance.server_drive import release, served

T975 = stats.t.ppf(0.975, 9)  # 2.2622


def _state(grain: str | None, *, id_column: str | None = None, target: str = "y",
           task: str = "regression", roles: dict[str, str] | None = None, holdout: float = 0.2,
           folds: int = 5) -> ProjectState:
    return ProjectState(
        lens=["clinical"], target=target, task=task, purpose="inference",
        roles=roles or {"pid": "identifier", "x": "exposure"}, missing="complete_case",
        split=SplitSpec(holdout=holdout, seed=0, folds=folds), models=["linear"],
        grain=None if grain is None else GrainSpec(grain=grain, id_column=id_column),
    )


def _stages(st: ProjectState, paths: dict[str, str], ids: Any, predictors: list[str],
            task: str = "regression") -> tuple[Any, Any]:
    """The real split, design and fit stages over these rows, as a worker runs them."""
    ti = mf.target_info(task, st.target)
    cohort = mf.cohort_bundle(ids, predictors)
    split = split_stage(mf.context(st, {"cohort": cohort, "target_info": ti}, paths))
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    return split, fit


def _row(model: dict[str, Any], feature: str) -> dict[str, Any]:
    return next(c for c in model["coefficients"] if c["feature"] == feature)


def _covers(row: dict[str, Any], truth: float = 0.0) -> bool:
    return row["ci_low"] is not None and row["ci_low"] <= truth <= row["ci_high"]


# ── 1 · repeated rows ────────────────────────────────────────────────────────

REPEATED_REPS = 1000  # the package asks for 400; 1,000 tightens the Monte Carlo SE to 0.007
PEOPLE, VISITS = 300, 3


@pytest.fixture(scope="module")
def repeated(tmp_path_factory):
    """1,000 replicates of 300 people × 3 rows, stacked: a person effect on the outcome and on the
    exposure (habitual intake plus day-to-day variation), and no effect of the exposure
    (the audit's fixture, A/r17_undetermined.py)."""
    rng = np.random.default_rng(20261002)
    blocks = []
    for r in range(REPEATED_REPS):
        g = np.repeat(np.arange(PEOPLE), VISITS)
        x = rng.normal(size=PEOPLE)[g] + 0.5 * rng.normal(size=PEOPLE * VISITS)
        y = rng.normal(size=PEOPLE)[g] + 0.7 * rng.normal(size=PEOPLE * VISITS)
        blocks.append(pd.DataFrame({"rep": r, "pid": g + 100_000 * r, "x": x, "y": y}))
    table = pd.concat(blocks, ignore_index=True)
    return table, mf.ingest_frame(table, tmp_path_factory.mktemp("wp2_repeated"))


@pytest.mark.parametrize("grain,basis", [("unknown", "undetermined"), ("one_row_per_unit", "abandoned")])
def test_1_repeated_rows_are_covered_however_the_seal_was_drawn(repeated, grain, basis):
    """AUDIT_REPORT §5 WP2 test 1. Grain "I don't know" (the seal's basis is ``undetermined``) and
    grain "one row each" while the identifier repeats (``abandoned``): the seal hands on no
    grouping in either case, yet over 1,000 replicates (the package asks for 400) the reported 95%
    intervals cover the truth at least 0.93 of the time. Reference: plain OLS 0.83–0.84, clustered
    0.947–0.948.

    Every replicate runs the real split, design and fit stages. The reference intervals are
    statsmodels' own on the same training rows: independent-row OLS (what the table reported
    before), and CR1 with t(G − 1) (Stata's clustered interval). Monte Carlo SE at 0.95 is 0.007.
    """
    import statsmodels.api as sm

    table, paths = repeated
    st = _state(grain, folds=2)
    reps = table["rep"].to_numpy()
    hits = plain_hits = cr1_hits = 0
    for r in range(REPEATED_REPS):
        ids = np.flatnonzero(reps == r)
        split, fit = _stages(st, paths, ids, ["x"])
        assert split.data["basis"]["state"] == basis
        assert split.data["grouped_by"] is None  # the seal grouped nothing
        model = fit.data["models"][0]
        info = model["inference"]
        assert info["covariance"] == "CR2" and info["grouped_by"] == "pid", info
        hits += _covers(_row(model, "x"))
        a = split.frames["assignment"]
        train = table.loc[a.loc[a["partition"] == "train", "row_id"].to_numpy()]
        X = sm.add_constant(train[["x"]])
        lo, hi = sm.OLS(train["y"], X).fit().conf_int(0.05).loc["x"]
        plain_hits += lo <= 0 <= hi
        lo, hi = sm.OLS(train["y"], X).fit(cov_type="cluster", cov_kwds={"groups": train["pid"]},
                                           use_t=True).conf_int(0.05).loc["x"]
        cr1_hits += lo <= 0 <= hi
    coverage, plain, cr1 = (h / REPEATED_REPS for h in (hits, plain_hits, cr1_hits))
    print(f"\n[{basis}] coverage reported {coverage:.3f} | plain OLS {plain:.3f} | CR1 t(G-1) {cr1:.3f}")
    assert coverage >= 0.93
    assert plain <= 0.88  # the reference 0.83–0.84: the bound tells the two apart
    assert cr1 >= 0.93  # the reference 0.947–0.948


def test_1_the_concern_says_why_the_intervals_cluster_against_the_answer(repeated):
    """The table says it clusters although the grain answer said otherwise, and why."""
    table, paths = repeated
    ids = np.flatnonzero(table["rep"].to_numpy() == 0)
    for grain, said in (("unknown", "answered as not known"), ("one_row_per_unit", "said to be a different unit")):
        _, fit = _stages(_state(grain), paths, ids, ["x"])
        concerns = fit.data["models"][0]["concerns"]
        assert any(said in c and "cluster-robust by `pid`" in c for c in concerns), concerns


# ── 2 · few units ────────────────────────────────────────────────────────────


def _six_by_forty() -> pd.DataFrame:
    """The audit's server scenario (E/scenA.py, seed 11): 6 participants × 40 visits, strong
    participant intercepts, sodium varying mostly between participants, and no sodium effect."""
    rng = np.random.default_rng(11)
    units, per = 6, 40
    pid = np.repeat(np.arange(units) + 1001, per)
    u_x = rng.normal(0, 1, units)
    u_y = rng.normal(0, 2, units)
    x = np.repeat(u_x, per) + rng.normal(0, 0.3, units * per)
    age = np.repeat(rng.integers(30, 70, units), per).astype(float)
    y = 5 + np.repeat(u_y, per) + rng.normal(0, 1, units * per)
    return pd.DataFrame({"participant_id": pid, "sodium_mg": x * 500 + 3000, "age": age,
                         "sbp": y * 3 + 110})


# These fixtures' truth (BLUEPRINT §14.3): age is drawn in whole years, an amount.
def _age_truth() -> Any:
    from turbotab.core.tests.truths import Truth

    # WP17: age is drawn before the exposure and causes the outcome.
    return Truth({"code_or_count:age": "amount", "adjust:age": "yes,yes,no"},
                 fixture="the audit's server scenario")


class _Drive:
    """Answers the opening sequence through the real HTTP API, in the Router's order."""

    def __init__(self, client: Any, pid: str, truth: Any = None):
        from turbotab.core.tests.truths import Truth

        self.c, self.pid = client, pid
        # The fixture's declared truth answers the readings asked (BLUEPRINT §14.3); the roles
        # the test records are its truth for the role readings.
        self.truth = truth if truth is not None else Truth(fixture="this fixture")

    def view(self) -> dict[str, Any]:
        return self.c.get(f"/api/projects/{self.pid}").json()

    def decide(self, body: dict[str, Any]) -> None:
        from turbotab.core.tests.acceptance.server_drive import settle_post

        if body.get("kind") == "set_roles":
            for column, role in body["roles"].items():
                self.truth.setdefault(f"role:{column}", role)
        r = settle_post(self.c, self.pid, body, self.truth)  # the readings asked, from the truth
        assert r.status_code == 200, (body["kind"], r.text[:600])

    def reach(self, key: str, timeout: float = 120.0) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            steps = self.view()["interview"]
            step = next(s for s in steps if s["key"] == key)
            first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
            ready = step["status"] not in ("open", "waiting") or first is None or first["key"] == key
            if ready and not (step["status"] == "waiting" and step.get("waiting_on")):
                return step
            assert time.monotonic() < end, f"{key} held behind {first}"
            time.sleep(0.05)

    def artifact(self, stage: str, timeout: float = 120.0) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            status = self.view()["stages"][stage]
            release(self.c, self.pid, status)
            if status["status"] == "fresh":
                return served(self.c, self.pid, stage)
            assert status["status"] != "error", status
            assert time.monotonic() < end, f"{stage} never fresh: {status}"
            time.sleep(0.05)


def test_2_six_units_refuse_intervals_with_exits(tmp_path):
    """AUDIT_REPORT §5 WP2 test 2: the 6 × 40 null-sodium server scenario no longer prints
    p = 4 × 10⁻¹⁴ silently. Below the unit floor, cluster-robust intervals are refused with exits
    (combine to one row per person; a mixed model, once WP12 adds it). Reference: a random-
    intercept model gives p = 0.85 (statsmodels MixedLM on the same rows); independent-row OLS,
    the table before, p ≈ 4 × 10⁻¹⁴ (statsmodels OLS).

    Driven through the real server (TestClient, real job workers), as the audit drove it.
    """
    import statsmodels.formula.api as smf
    from fastapi.testclient import TestClient

    from turbotab.core.config import Settings
    from turbotab.server.app import create_app

    frame = _six_by_forty()
    path = tmp_path / "six_by_forty.csv"
    frame.to_csv(path, index=False)
    settings = Settings(home=tmp_path / "home", mode="local", workers=2, memory_budget_bytes=1 << 30)
    with TestClient(create_app(settings, frontend_dist=tmp_path / "none"),
                    base_url="http://127.0.0.1") as client:
        r = client.post("/api/projects", json={"path": str(path)})
        assert r.status_code == 200, r.text
        d = _Drive(client, r.json()["id"], _age_truth())
        d.artifact("ingest")
        d.decide({"kind": "set_lens", "lenses": ["clinical"]})
        d.reach("target")
        d.decide({"kind": "set_target", "column": "sbp"})
        if d.reach("task")["status"] in ("open", "waiting"):
            d.decide({"kind": "set_task", "column": "sbp", "task": "regression"})
        d.reach("purpose")
        d.decide({"kind": "set_purpose", "purpose": "inference"})
        d.reach("grain")
        d.decide({"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
        answers = {"repeat_kind": {"kind": "set_repeat_kind", "repeat_kind": "repeats"},
                   "unit": {"kind": "set_unit", "unit": "row"},
                   "temporal": {"kind": "set_temporal", "temporal": False}}
        for key in ("repeat_kind", "unit", "aggregation", "temporal"):
            if d.reach(key)["status"] in ("open", "waiting"):
                d.decide(answers[key])
        d.reach("roles")
        d.decide({"kind": "set_roles", "roles": {"participant_id": "identifier",
                                                 "sodium_mg": "exposure", "age": "covariate"}})
        from turbotab.core.tests.acceptance.server_drive import answer_plan

        d.reach("exclusions")
        d.decide({"kind": "set_exclusions", "rules": []})
        d.reach("missing")
        d.decide({"kind": "set_missing", "strategy": "complete_case"})
        d.reach("split")
        d.decide({"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        # WP17, after the split (MODELING_SEQUENCE §1 steps 2–3): the exposure, its effect and the
        # adjustment set
        answer_plan(d, "sodium_mg")
        if d.reach("energy_adjustment")["status"] in ("open", "waiting"):
            d.decide({"kind": "set_energy_adjustment", "method": "none"})
        d.reach("models")
        d.decide({"kind": "select_models", "models": ["linear"]})
        split = d.artifact("split")
        fit = d.artifact("fit")
    assert split["basis"]["state"] == "abandoned" and split["grouped_by"] is None
    model = fit["models"][0]
    info = model["inference"]
    assert info["covariance"] == "none" and info["grouped_by"] == "participant_id"
    assert info["n_clusters"] == 6 and min_clusters() == 8
    assert "6 units" in info["refused"] and f"the {min_clusters()}" in info["refused"]
    labels = [e["label"] for e in info["exits"]]
    assert any("Combine each `participant_id`'s rows into one" in label for label in labels), labels
    assert any("mixed model" in label for label in labels), labels
    assert info["refused"] in model["concerns"]
    sodium = _row(model, "sodium_mg")
    assert sodium["estimate"] is not None
    assert all(c["p"] is None and c["ci_low"] is None and c["ci_high"] is None
               for c in model["coefficients"])
    # The references, on the same 240 rows (holdout 0): the random-intercept model's p, and the
    # independent-row p the table printed before.
    mixed = smf.mixedlm("sbp ~ sodium_mg + age", frame, groups=frame["participant_id"]).fit()
    naive = smf.ols("sbp ~ sodium_mg + age", frame).fit()
    print(f"\nmixed p {mixed.pvalues['sodium_mg']:.3f}; independent-row OLS p {naive.pvalues['sodium_mg']:.2g}")
    assert mixed.pvalues["sodium_mg"] == pytest.approx(0.85, abs=0.01)
    assert naive.pvalues["sodium_mg"] < 1e-12


def test_2_rows_said_to_repeat_with_no_unit_column_get_no_intervals():
    """The other way the seal abandons grouping: the rows were said to repeat, but no column names
    the unit (or the named one is not in the table). Which rows belong together is unknown, so no
    interval is reported, and the exit is to name the unit. Rows already combined into one per
    unit are not refused."""
    rng = np.random.default_rng(5)
    M = pd.DataFrame({"x": rng.normal(size=120)})
    y = rng.normal(size=120)
    frame = pd.DataFrame(index=np.arange(120))
    for grain in (GrainSpec(grain="repeated"), GrainSpec(grain="repeated", id_column="person")):
        st = _state(None).model_copy(update={"grain": grain})
        clusters = resolve_clusters(st, frame)
        table = inference_table("regression", M, y, None, clusters)
        assert all(r["ci_low"] is None and r["p"] is None for r in table.rows)
        assert table.info["covariance"] == "none" and "said to repeat" in table.info["refused"]
        assert [e["label"] for e in table.info["exits"]] == [
            "Name the column that identifies the unit (the grain question)"]
    from turbotab.core.decisions import AggregationSpec

    combined = _state(None).model_copy(update={
        "grain": GrainSpec(grain="repeated", id_column="person"), "unit": "unit",
        "aggregation": AggregationSpec(method="mean")})
    assert resolve_clusters(combined, frame) is INDEPENDENT


# ── 3 · the small-G correction ───────────────────────────────────────────────


def _balanced_binary(G0: int, G1: int, m: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    G = G0 + G1
    codes = np.repeat(np.arange(G), m)
    treated = (np.arange(G) >= G0).astype(float)[codes]
    y = rng.normal(size=G)[codes] + rng.normal(size=G * m)
    return pd.DataFrame({"arm": treated}), y, codes


def test_3_ten_clusters_get_t_intervals_on_small_sample_degrees_of_freedom():
    """AUDIT_REPORT §5 WP2 test 3, first part, at G = 10.

    Source check (Cameron & Miller 2015, *J Hum Resour* 50:317–372): "at a minimum one should
    use the T(G − 1) distribution rather than the standard normal", and "The best methods use the
    CR2VE and T(v*)". The intervals are CR2 with Bell–McCaffrey degrees of freedom (the T(v*)
    above), so ``use_t`` holds: every interval's half-width over its SE is the t quantile at that
    row's df. On a balanced cluster-level binary exposure, 5 clusters against 5, the df has a
    closed form (Imbens & Kolesár 2016): ν = 8, so half-width/SE = t₀.₉₇₅(8) = 2.306, wider than
    t(9)'s 2.262; and the CR2 SE equals the Neyman variance of the cluster means. Deviation from
    the package text: it names t(9) for G = 10, the minimum; the engine uses the sounder T(v*).
    """
    M, y, codes = _balanced_binary(5, 5, 20, seed=3)
    clusters = Clusters(column="pid", codes=codes, n_clusters=10)
    table = inference_table("regression", M, y, None, clusters)
    arm = next(r for r in table.rows if r["feature"] == "arm")
    assert arm["df"] == pytest.approx(ref.bm_df_binary(5, 5), rel=1e-10) == pytest.approx(8.0)
    half = (arm["ci_high"] - arm["ci_low"]) / 2
    assert half / arm["se"] == pytest.approx(stats.t.ppf(0.975, 8), rel=1e-10)
    assert half / arm["se"] >= T975
    means = pd.Series(y).groupby(codes).mean().to_numpy()
    arm_of = np.arange(10) >= 5
    neyman = np.sqrt(means[~arm_of].var(ddof=1) / 5 + means[arm_of].var(ddof=1) / 5)
    assert arm["se"] == pytest.approx(neyman, rel=1e-10)
    caption = table.info["caption"]
    assert "cluster-robust" in caption and "G = 10" in caption and "Bell–McCaffrey" in caption
    assert table.info["covariance"] == "CR2" and table.info["n_clusters"] == 10


def test_3_cr2_equals_its_definition_on_an_unbalanced_design():
    """CR2 SEs and Bell–McCaffrey df, every row, equal the definition written out with n × n
    matrices (``references.cr2_by_definition``), on 10 unbalanced clusters with a within-cluster
    covariate and a rare binary one; every interval is the t quantile at its df."""
    import statsmodels.api as sm

    rng = np.random.default_rng(8)
    sizes = rng.integers(2, 15, 10)
    codes = np.repeat(np.arange(10), sizes)
    n = len(codes)
    M = pd.DataFrame({"x": rng.normal(size=10)[codes] + 0.4 * rng.normal(size=n),
                      "z": rng.normal(size=n), "rare": (rng.random(n) < 0.15).astype(float)})
    y = rng.normal(size=10)[codes] + rng.normal(size=n)
    table = inference_table("regression", M, y, None, Clusters(column="pid", codes=codes, n_clusters=10))
    X = sm.add_constant(M).to_numpy()
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    V, df = ref.cr2_by_definition(X, e, codes)
    for j, row in enumerate(table.rows):
        assert row["se"] == pytest.approx(np.sqrt(V[j, j]), rel=1e-9)
        assert row["df"] == pytest.approx(df[j], rel=1e-9)
        half = (row["ci_high"] - row["ci_low"]) / 2
        assert half == pytest.approx(stats.t.ppf(0.975, df[j]) * np.sqrt(V[j, j]), rel=1e-9)
        t = row["estimate"] / row["se"]
        assert row["p"] == pytest.approx(2 * stats.t.sf(abs(t), df[j]), rel=1e-9)


def test_3_past_the_cost_budget_the_df_fall_back_to_t_of_g_minus_one_and_say_so(monkeypatch):
    """When the Bell–McCaffrey df would cost too much (G·P³ past the budget), the CR2 standard
    errors stand and the reference falls back to the minimum Cameron & Miller name, t(G − 1),
    and the caption says so."""
    import statsmodels.api as sm

    import turbotab.core.models.inference as inference

    rng = np.random.default_rng(9)
    M, y, codes = _few_clusters(12, rng, "regression")
    monkeypatch.setattr(inference, "BM_BUDGET", 1.0)
    table = inference_table("regression", M, y, None, Clusters(column="pid", codes=codes, n_clusters=12))
    X = sm.add_constant(M).to_numpy()
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    V, _ = ref.cr2_by_definition(X, e, codes)
    for j, row in enumerate(table.rows):
        assert row["df"] == 11
        assert row["se"] == pytest.approx(np.sqrt(V[j, j]), rel=1e-9)
        assert (row["ci_high"] - row["ci_low"]) / 2 == pytest.approx(stats.t.ppf(0.975, 11) * row["se"], rel=1e-9)
    assert "t(11)" in table.info["caption"] and "too costly" in table.info["caption"]


def _few_clusters(G: int, rng: np.random.Generator, task: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The audit's few-cluster fixture (A-skeptic/s3_cluster.py): 20 rows per cluster, an exposure
    that varies mostly between clusters, a cluster effect on the outcome, no exposure effect."""
    codes = np.repeat(np.arange(G), 20)
    x = rng.normal(size=G)[codes] + 0.3 * rng.normal(size=G * 20)
    u = rng.normal(size=G)[codes]
    if task == "regression":
        y = u + rng.normal(size=G * 20)
    else:
        y = (rng.random(G * 20) < 1 / (1 + np.exp(-(-0.3 + 1.5 * u)))).astype(int)
    return pd.DataFrame({"x": x}), y, codes


@pytest.mark.parametrize("task,reps", [("regression", 1000), ("binary", 600)])
def test_3_type_one_error_at_twelve_clusters(task, reps):
    """AUDIT_REPORT §5 WP2 test 3, second part: with CR2, Monte Carlo type-I error at G = 12 is
    at most 0.07. Reference with z: 0.10–0.20 (statsmodels' clustered fit with its default normal
    reference, what the table reported before, computed on the same replicates). Monte Carlo SE
    at 0.05: 0.007 (1,000 replicates) and 0.009 (600)."""
    import statsmodels.api as sm

    rng = np.random.default_rng(12_000 + reps)
    rejected = z_rejected = done = 0
    state = _state("repeated", id_column="pid")
    for _ in range(reps):
        M, y, codes = _few_clusters(12, rng, task)
        if task == "binary" and len(np.unique(y)) < 2:
            continue
        frame = pd.DataFrame({"pid": codes}, index=np.arange(len(codes)))
        clusters = resolve_clusters(state, frame)
        assert clusters.n_clusters == 12
        table = inference_table(task, M, y, [0, 1] if task == "binary" else None, clusters)
        row = next(r for r in table.rows if r["feature"] == "x")
        exog = sm.add_constant(M)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y, exog) if task == "regression" else sm.Logit(y, exog)
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": codes},
                            **({} if task == "regression" else {"disp": 0}))
        done += 1
        rejected += row["p"] < 0.05
        z_rejected += fit.pvalues["x"] < 0.05
    rate, z_rate = rejected / done, z_rejected / done
    print(f"\n[{task}] G = 12 type-I: CR2 with Bell–McCaffrey df {rate:.3f}; CR1 with z {z_rate:.3f}")
    assert rate <= 0.07
    assert 0.09 <= z_rate <= 0.22  # the reference 0.10–0.20: the bound tells the two apart


def test_3_logistic_and_multinomial_working_models_are_statsmodels_own_sandwich():
    """CR2 for a logistic or multinomial model is CR2 on the model's working linear model
    (McCaffrey & Bell 2006). With the CR2 adjustment off, that working model must give exactly
    statsmodels' clustered sandwich divided by statsmodels' small-sample factor
    G/(G − 1)·(n − 1)/(n − k), and with one row per cluster its HC0 sandwich; its information must
    be statsmodels' model covariance, in statsmodels' coefficient order."""
    import statsmodels.api as sm

    from turbotab.core.models.inference import bread_of, cr2, logistic_working, multinomial_working

    rng = np.random.default_rng(2)
    G = 15
    codes = np.repeat(np.arange(G), rng.integers(3, 12, G))
    n = len(codes)
    X = np.column_stack([np.ones(n), rng.normal(size=G)[codes] + rng.normal(size=n), rng.normal(size=n)])
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.3 * X[:, 1] + rng.normal(size=G)[codes])))).astype(float)
    fit = sm.Logit(y, X).fit(disp=0)
    Xw, ew = logistic_working(X, y, fit.predict(X))
    V0, _ = cr2(Xw, ew, codes, bread_of(Xw), adjust=False)
    clustered = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": codes})
    np.testing.assert_allclose(V0 * G / (G - 1) * (n - 1) / (n - 3), clustered.cov_params(), rtol=1e-9)
    singletons, _ = cr2(Xw, ew, np.arange(n), bread_of(Xw), adjust=False)
    np.testing.assert_allclose(singletons, sm.Logit(y, X).fit(disp=0, cov_type="HC0").cov_params(),
                               rtol=1e-9)

    y3 = np.digitize(X[:, 1] + rng.normal(size=G)[codes] + rng.normal(size=n), [-0.7, 0.7])
    mfit = sm.MNLogit(y3, X).fit(disp=0)
    Xw, ew = multinomial_working(X, y3, mfit.predict(X))
    np.testing.assert_allclose(np.sqrt(np.diag(bread_of(Xw))), np.asarray(mfit.bse).ravel(order="F"),
                               rtol=1e-9)
    V0, _ = cr2(Xw, ew, np.repeat(codes, 2), bread_of(Xw), adjust=False)
    clustered = sm.MNLogit(y3, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": codes})
    np.testing.assert_allclose(np.sqrt(np.diag(V0) * G / (G - 1) * (n - 1) / (n - 6)),
                               np.asarray(clustered.bse).ravel(order="F"), rtol=1e-9)


def test_3_multinomial_type_one_error_at_twelve_clusters():
    """The same bound for a three-class outcome: at G = 12 the CR2 row for the exposure (first
    non-reference class) rejects a true null at most 0.07 of the time; statsmodels' clustered fit
    with a normal reference, on the same replicates, is the reference. Monte Carlo SE 0.011."""
    import statsmodels.api as sm

    rng = np.random.default_rng(31)
    rejected = z_rejected = done = 0
    while done < 400:
        codes = np.repeat(np.arange(12), 20)
        x = rng.normal(size=12)[codes] + 0.3 * rng.normal(size=240)
        y = np.digitize(rng.normal(size=12)[codes] + rng.logistic(size=240), [-0.8, 0.8])
        if len(np.unique(y)) < 3:
            continue
        clusters = Clusters(column="pid", codes=codes, n_clusters=12)
        table = inference_table("multiclass", pd.DataFrame({"x": x}), y, [0, 1, 2], clusters)
        row = next(r for r in table.rows if r["feature"] == "x [1]")
        z = sm.MNLogit(y, sm.add_constant(x)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": codes})
        done += 1
        rejected += row["p"] < 0.05
        z_rejected += np.asarray(z.pvalues)[1, 0] < 0.05
    rate, z_rate = rejected / done, z_rejected / done
    print(f"\n[multiclass] G = 12 type-I: CR2 {rate:.3f}; CR1 with z {z_rate:.3f}")
    assert rate <= 0.07
    assert z_rate >= 0.09


@pytest.mark.parametrize("G", [12, FEW_CLUSTERS - 1, FEW_CLUSTERS])
def test_3_a_concern_names_the_number_of_clusters_below_thirty(G):
    """AUDIT_REPORT §5 WP2 test 3, third part: a concern names G whenever G < 30."""
    rng = np.random.default_rng(G)
    M, y, codes = _few_clusters(G, rng, "regression")
    frame = pd.DataFrame({"pid": codes}, index=np.arange(len(codes)))
    table = inference_table("regression", M, y, None, resolve_clusters(_state("repeated", id_column="pid"), frame))
    few = [c for c in table.concerns if f"Only {G} `pid` clusters" in c]
    assert bool(few) == (G < FEW_CLUSTERS), table.concerns
    assert f"G = {G}" in table.info["caption"]


# ── 4 · heteroskedasticity ───────────────────────────────────────────────────


def _intake(n: int, rng: np.random.Generator, spread: str) -> tuple[pd.DataFrame, np.ndarray]:
    """A skewed intake (lognormal) with no effect on the outcome; the error SD grows in proportion
    to intake (``spread="sd"``, the audit's fixture A/r9_coef.py) or is constant."""
    x = rng.lognormal(0, 0.6, n)
    noise = rng.normal(size=n)
    y = 1 + 0.0 * x + (noise * x if spread == "sd" else noise)
    return pd.DataFrame({"x": x}), y


def test_4_hc3_intervals_cover_when_the_spread_grows_with_intake():
    """AUDIT_REPORT §5 WP2 test 4: null slope on lognormal intake with error SD proportional to
    intake, n = 200: the reported intervals (HC3, t(n − 2)) cover at least 0.92. Reference:
    classical 0.61, HC3 0.93 (statsmodels' classical and HC3 intervals on the same replicates).
    Monte Carlo SE at 0.93: 0.004 (4,000 replicates)."""
    import statsmodels.api as sm

    rng = np.random.default_rng(2026)
    reps = 4000
    hits = classical = sm_hc3 = 0
    for _ in range(reps):
        M, y = _intake(200, rng, "sd")
        table = inference_table("regression", M, y, None, INDEPENDENT)
        hits += _covers(next(r for r in table.rows if r["feature"] == "x"))
        fit = sm.OLS(y, sm.add_constant(M)).fit()
        lo, hi = fit.conf_int(0.05).loc["x"]
        classical += lo <= 0 <= hi
        lo, hi = sm.OLS(y, sm.add_constant(M)).fit(cov_type="HC3").conf_int(0.05).loc["x"]
        sm_hc3 += lo <= 0 <= hi
    coverage, plain, reference = hits / reps, classical / reps, sm_hc3 / reps
    print(f"\nn = 200 coverage: reported {coverage:.3f}; classical {plain:.3f}; statsmodels HC3 (z) {reference:.3f}")
    assert coverage >= 0.92
    assert plain <= 0.70  # the reference 0.61
    assert table.info["covariance"] == "HC3" and "HC3" in table.info["caption"]


def test_4_hc3_equals_its_definition():
    """The HC3 SEs equal MacKinnon & White's formula written out, and the intervals use t(n − p)."""
    import statsmodels.api as sm

    rng = np.random.default_rng(4)
    M, y = _intake(150, rng, "sd")
    M["z"] = rng.normal(size=150)
    table = inference_table("regression", M, y, None, INDEPENDENT)
    X = sm.add_constant(M).to_numpy()
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    V = ref.hc3_by_definition(X, e)
    for j, row in enumerate(table.rows):
        assert row["se"] == pytest.approx(np.sqrt(V[j, j]), rel=1e-9)
        assert row["df"] == 150 - 3
        half = (row["ci_high"] - row["ci_low"]) / 2
        assert half == pytest.approx(stats.t.ppf(0.975, 147) * np.sqrt(V[j, j]), rel=1e-9)
    assert "t(147)" in table.info["caption"]


def test_4_the_residual_spread_check_speaks_on_that_fixture_and_not_on_a_clean_one():
    """AUDIT_REPORT §5 WP2 test 4, the check: a residual-variance concern on the heteroskedastic
    fixture, silence on a homoskedastic one. Over 500 replicates each: it speaks on at least 95%
    of the heteroskedastic fixtures and on at most 2% of the homoskedastic ones (its level is 1%;
    Monte Carlo SE 0.0045). The p-value the check uses is Koenker's studentized Breusch–Pagan
    statistic, recomputed by hand (``references.koenker_bp``)."""
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan

    def speaks(table) -> bool:
        return any("residual spread is not constant" in c for c in table.concerns)

    rng = np.random.default_rng(44)
    fired = {"sd": 0, "constant": 0}
    for spread in fired:
        for _ in range(500):
            M, y = _intake(200, rng, spread)
            fired[spread] += speaks(inference_table("regression", M, y, None, INDEPENDENT))
    print(f"\nthe check spoke on {fired['sd']}/500 heteroskedastic and {fired['constant']}/500 clean fixtures")
    assert fired["sd"] >= 475 and fired["constant"] <= 10
    M, y = _intake(200, np.random.default_rng(7), "sd")
    hetero = inference_table("regression", M, y, None, INDEPENDENT)
    M0, y0 = _intake(200, np.random.default_rng(7), "constant")
    clean = inference_table("regression", M0, y0, None, INDEPENDENT)
    assert speaks(hetero) and not speaks(clean)
    X = sm.add_constant(M).to_numpy()
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    assert het_breuschpagan(e, X, robust=True)[1] == pytest.approx(ref.koenker_bp(e, X), rel=1e-9)


# ── 5 · separation ───────────────────────────────────────────────────────────


def _separated(seed: int, n: int = 300) -> tuple[pd.DataFrame, np.ndarray]:
    """The audit's fixture (A-skeptic/s14b_sep.py): an 8% exposure, every exposed row an event."""
    rng = np.random.default_rng(seed)
    exposed = (rng.random(n) < 0.08).astype(float)
    age = rng.normal(50, 10, n)
    base = (rng.random(n) < 1 / (1 + np.exp(-(-1 + 0.02 * (age - 50))))).astype(int)
    return pd.DataFrame({"exposed": exposed, "age": age}), np.where(exposed == 1, 1, base)


def test_5_separation_is_named_and_firth_replaces_the_wald_row():
    """AUDIT_REPORT §5 WP2 test 5, on the audit's 20 seeds: a concern names separation and the
    column, the Wald interval and p-value are not reported for that row (the table is Firth's),
    and the Firth estimate is finite. Reference: the maximum-likelihood Wald row the table printed
    before (statsmodels Logit: a log-odds of 20 or more with p near 1, or a singular matrix), and
    the Firth estimate recomputed by a general-purpose optimizer of the penalized likelihood."""
    import statsmodels.api as sm

    wald_meaningless = 0
    for seed in range(20):
        M, y = _separated(seed)
        table = inference_table("binary", M, y, [0, 1], INDEPENDENT)
        assert table.info["separated"] == ["exposed"], seed
        assert table.info["estimator"] == "Firth-penalized logistic regression"
        assert any("`exposed` separates the outcome" in c for c in table.concerns)
        assert not any("`age`" in c for c in table.concerns)
        row = next(r for r in table.rows if r["feature"] == "exposed")
        assert np.isfinite(row["estimate"]) and 0 < row["estimate"] < 10
        assert row["df"] is None  # a likelihood interval, not a Wald t or z interval
        X = sm.add_constant(M).to_numpy()
        beta, _ = ref.firth_by_optimizer(X, y.astype(float))
        np.testing.assert_allclose([r["estimate"] for r in table.rows], beta, rtol=1e-5, atol=1e-7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ml = sm.Logit(y, sm.add_constant(M)).fit(disp=0, maxiter=200)
                wald_meaningless += ml.params["exposed"] > 15 and ml.pvalues["exposed"] > 0.9
            except Exception:  # noqa: BLE001 - "Singular matrix": the other way it failed
                wald_meaningless += 1
    assert wald_meaningless == 20


def test_5_firth_equals_the_haldane_table_and_its_profile_interval_is_the_definition():
    """Firth's estimate for one binary covariate with an intercept is the log odds ratio of the
    2 × 2 table with ½ added to each cell: the Jeffreys penalty of a saturated two-group model is
    ½ log μ(1 − μ) per group, so each group's fitted risk is (events + ½)/(n + 1). The profile
    penalized-likelihood interval and likelihood-ratio p-value (Heinze & Schemper 2002) equal the
    definition recomputed with a general-purpose optimizer."""
    from scipy.optimize import brentq

    import statsmodels.api as sm

    rng = np.random.default_rng(3)
    n = 200
    x = (rng.random(n) < 0.1).astype(float)
    y = np.where(x == 1, 1, rng.random(n) < 0.3).astype(int)
    table = inference_table("binary", pd.DataFrame({"x": x}), y, [0, 1], INDEPENDENT)
    row = next(r for r in table.rows if r["feature"] == "x")
    assert row["estimate"] == pytest.approx(ref.haldane_log_odds_ratio(x, y), rel=1e-9)

    z = rng.normal(size=n)
    M = pd.DataFrame({"x": x, "z": z})
    table = inference_table("binary", M, y, [0, 1], INDEPENDENT)
    X = sm.add_constant(M).to_numpy()
    beta, top = ref.firth_by_optimizer(X, y.astype(float))
    crit = stats.chi2.ppf(0.95, 1) / 2
    for j, row in enumerate(table.rows):
        def drop(b: float, j: int = j) -> float:
            return top - ref.firth_by_optimizer(X, y.astype(float), fixed={j: b}, start=beta)[1]

        low = brentq(lambda b: drop(b) - crit, beta[j] - 15, beta[j])
        high = brentq(lambda b: drop(b) - crit, beta[j], beta[j] + 15)
        assert row["ci_low"] == pytest.approx(low, abs=1e-5)
        assert row["ci_high"] == pytest.approx(high, abs=1e-5)
        assert row["p"] == pytest.approx(stats.chi2.sf(2 * drop(0.0), 1), rel=1e-4, abs=1e-12)


def test_5_separation_does_not_fire_on_data_that_are_not_separated():
    """Must-not-fire: a rare exposure whose rows hold both outcomes is not separation; the table
    stays maximum likelihood with Wald intervals equal to statsmodels' own."""
    import statsmodels.api as sm

    for seed in range(50):
        rng = np.random.default_rng(1000 + seed)
        n = 300
        exposed = (rng.random(n) < 0.08).astype(float)
        exposed[:2] = 1.0  # at least one exposed event and one exposed non-event
        age = rng.normal(50, 10, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(-1 + 1.0 * exposed + 0.02 * (age - 50))))).astype(int)
        y[0], y[1] = 1, 0
        M = pd.DataFrame({"exposed": exposed, "age": age})
        table = inference_table("binary", M, y, [0, 1], INDEPENDENT)
        assert table.info["separated"] == [] and table.info["covariance"] == "model", seed
        direct = sm.Logit(y, sm.add_constant(M)).fit(disp=0)
        lo, hi = direct.conf_int(0.05).loc["exposed"]
        row = next(r for r in table.rows if r["feature"] == "exposed")
        assert row["ci_low"] == pytest.approx(lo, rel=1e-6) and row["ci_high"] == pytest.approx(hi, rel=1e-6)


def test_5_separation_with_repeated_rows_keeps_the_firth_estimate_and_reports_no_interval():
    """Firth's profile intervals assume independent rows. When the rows also repeat, the table
    keeps the finite Firth estimates, names the separation, and reports no interval or p-value."""
    M, y = _separated(4)
    codes = np.arange(len(y)) // 15  # 20 clusters of 15
    clusters = Clusters(column="pid", codes=codes, n_clusters=20)
    table = inference_table("binary", M, y, [0, 1], clusters)
    assert table.info["separated"] == ["exposed"] and table.info["covariance"] == "none"
    assert "rows repeat by `pid`" in table.info["refused"]
    assert any("`exposed` separates the outcome" in c for c in table.concerns)
    assert all(r["ci_low"] is None and r["p"] is None and np.isfinite(r["estimate"]) for r in table.rows)


def test_5_the_fit_stage_reports_separation_on_its_model(tmp_path):
    """Through the real stages: the separation concern and the Firth table reach the fit."""
    M, y = _separated(1)
    frame = M.assign(event=np.where(y == 1, "yes", "no"))
    paths = mf.ingest_frame(frame, tmp_path)
    st = _state("one_row_per_unit", target="event", task="binary",
                roles={"exposed": "exposure", "age": "covariate"}, holdout=0.0)
    st = st.model_copy(update={"event": "yes"})
    _, fit = _stages(st, paths, np.arange(len(frame)), ["exposed", "age"], task="binary")
    model = fit.data["models"][0]
    assert model["inference"]["separated"] == ["exposed"]
    assert model["inference"]["covariance"] == "profile"
    assert any("`exposed` separates the outcome" in c for c in model["concerns"])
    row = _row(model, "exposed")
    assert row["ci_low"] is not None and row["ci_high"] is not None and row["ci_low"] > 0


# ── 6 · missing identifiers ──────────────────────────────────────────────────


def test_6_missing_identifiers_are_units_of_their_own_as_the_split_maps_them(tmp_path):
    """AUDIT_REPORT §5 WP2 test 6 (closes A18): rows with no identifier are mapped to one unit per
    row before clustering, the mapping the split uses (``seal.seal_inputs``). Before, a missing
    identifier was factorized to one shared code, so every such row fell in one giant cluster."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.seal import seal_inputs

    rng = np.random.default_rng(6)
    people, visits = 60, 3
    pid = np.repeat(np.arange(people) + 500, visits).astype(float)
    gone = rng.choice(len(pid), 12, replace=False)
    pid[gone] = np.nan
    x = rng.normal(size=len(pid))
    frame = pd.DataFrame({"pid": pid, "x": x, "y": x * 0.1 + rng.normal(size=len(pid))})
    paths = mf.ingest_frame(frame, tmp_path)
    st = _state("repeated", id_column="pid", holdout=0.0)
    ids = np.arange(len(frame))

    with DataStore(Path(paths["data"]), 1 << 30) as store:
        draw = seal_inputs(st, ids, store, "regression", holdout=0.0, seed=0)
        held = store.materialize(["pid"], ids)
    clusters = resolve_clusters(st, held)
    expected_units = frame["pid"].nunique() + len(gone)  # the units by hand: 60 named, 12 alone
    assert clusters.n_clusters == expected_units == len(set(draw.groups))
    assert clusters.n_missing == len(gone)
    # The same partition of the rows: a bijection between the split's groups and the codes.
    pairs = set(zip(draw.groups, clusters.codes.tolist()))
    assert len(pairs) == len(set(draw.groups)) == len(set(clusters.codes.tolist()))
    # The old mapping, for the record: every missing identifier shared one code.
    assert len(set(pd.factorize(frame["pid"])[0].tolist())) == frame["pid"].nunique() + 1

    _, fit = _stages(st, paths, ids, ["x"])
    info = fit.data["models"][0]["inference"]
    assert info["n_clusters"] == expected_units and info["n_missing_ids"] == len(gone)
    assert "12 rows with no `pid` counted as units of their own" in info["caption"]
