"""WP16 acceptance tests 1 and 2: the seal holds at its edges (docs/turbotab-next/audit/AUDIT_REPORT.md §5).

They close RO-01 and RO-02, the two one-validator guards taken early with the math layer
(BLUEPRINT §12, "Fixing order"):

1. **Membership by identity.** "After the split, setting three impossible outcome values to missing
   moves **0** held-out rows (today 119–130); ``set_target`` and ``set_task`` after the split are
   refused with the re-seal exit or recorded as "re-sealed" and stated; the Router orders the
   impossibility and code repairs before the split."
2. **Outcome eligibility.** "``bmi 18.5–30`` with ``bmi`` as the outcome is refused (409) with exits,
   under both purposes; the exclusions preview no longer draws the outcome's histogram."

Every project is driven through the real server (the FastAPI app over its job runner), as the
auditors drove it. The two tables replay the routing skeptic's generator (``repro.tar.gz``:
``I-skeptic/mkfx.py``, ``numpy.random.default_rng(2026)``) draw for draw, so the 1,000-row ``sbp``
table and the 4,000-row ``bmi`` table are the audit's own; the ``sbp`` table gains one predictor
column, drawn from its own stream after the audit's, so that a repair to a predictor can be shown
to pass the guard.

References come from paths independent of the code under test: the held-out rows are read from the
split's parquet on disk and compared as sets; the draw an unguarded repair would have made is
recomputed with scikit-learn's ``train_test_split``; the bias an outcome rule causes is recomputed
by ordinary least squares in NumPy.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core import seal  # noqa: F401 - registers the seal's validators
from turbotab.core.graph import artifact_dir
from turbotab.server.tests.conftest import make_client, prepare, wait_for

SBP_IMPOSSIBLE_POSITIONS = (7, 333, 801)  # mkfx.py: sbp[[7, 333, 801]] = [999, 0, 1200]
TRUE_FIBER_SLOPE = -0.25  # mkfx.py: bmi = 27 - 0.25 * (fiber - 20) + ...


# ── the audit's fixtures, replayed ───────────────────────────────────────────


def audit_tables(folder: Path) -> dict[str, Path]:
    """I-skeptic/mkfx.py, draw for draw from one stream; only the two tables used here are written."""
    rng = np.random.default_rng(2026)
    # I1: BMI outcome; fiber exposure, true slope -0.25 per g
    n = 4000
    age = rng.normal(50, 10, n).round(0)
    fiber = rng.gamma(4, 5, n).round(1)
    bmi = (27 - 0.25 * (fiber - 20) + 0.03 * (age - 50) + rng.normal(0, 5, n)).round(1)
    bmi_table = pd.DataFrame({"participant_id": [f"P{i:05d}" for i in range(n)], "age": age,
                              "fiber_g": fiber, "bmi": bmi})
    # I3, I4, I5, I9, I10: drawn as the generator draws them, so that I2 gets the audit's numbers.
    n = 3000
    entry = rng.uniform(0, 14, n)
    _ = (15 + 1.0 * entry + rng.normal(0, 4, n)).clip(1)
    age3 = rng.normal(55, 8, n)
    _ = rng.exponential(1 / (0.03 * np.exp(0.04 * (age3 - 55))))
    m = 2400
    grp = rng.random(m) < 0.5
    _ = rng.integers(1, 16, m), rng.integers(1, 3, m)
    _ = rng.normal(2100, 500, m)
    fib = rng.gamma(4, 5, m)
    _ = 3 + np.where(grp, -0.10, 0.02) * (fib - 20) + 0.4 * grp + rng.normal(0, 1, m)
    n = 700
    s = rng.integers(0, 10, n)
    se = rng.normal(0, 1, 10)[s]
    x = rng.normal(0, 1, n) + 0.8 * se
    _ = 0 * x + 1.5 * se + rng.normal(0, 1, n)
    for _i in range(300):
        base, slope = rng.normal(130, 12), rng.normal(2, 3)
        for v in (1, 2, 3):
            _ = round(base + slope * (v - 1) * 3 + rng.normal(0, 4), 1), round(rng.normal(120, 20), 1)
    n = 800
    _ = rng.normal(55, 8, n).round(0), rng.gamma(4, 5, n).round(1)
    _ = rng.normal(0, 1, n)
    # I2: impossible outcome values
    n = 1000
    age = rng.normal(55, 10, n).round(0)
    na = rng.normal(3, 1, n).round(2)
    sbp = (110 + 0.5 * (age - 55) + 3 * na + rng.normal(0, 10, n)).round(0)
    sbp[list(SBP_IMPOSSIBLE_POSITIONS)] = [999, 0, 1200]
    # One predictor more, from its own stream: two impossible values, on rows the outcome's are not.
    extra = np.random.default_rng(16).normal(27, 4, n).round(1)
    extra[[50, 600]] = [3.0, 250.0]
    sbp_table = pd.DataFrame({"participant_id": [f"B{i:04d}" for i in range(n)], "age": age,
                              "sodium_g": na, "bmi": extra, "sbp": sbp})
    out = {"bmi": folder / "bmi.csv", "sbp": folder / "sbp_imp.csv"}
    bmi_table.to_csv(out["bmi"], index=False)
    sbp_table.to_csv(out["sbp"], index=False)
    return out


def score_table(folder: Path) -> Path:
    """A five-level numeric outcome, which reads as regression or as multiclass."""
    rng = np.random.default_rng(7)
    n = 800
    x = rng.normal(0, 1, n)
    score = np.clip(np.round(3 + 0.8 * x + rng.normal(0, 1, n)), 1, 5).astype(int)
    path = folder / "score.csv"
    pd.DataFrame({"participant_id": [f"S{i:04d}" for i in range(n)], "x": x.round(3),
                  "score": score}).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("wp16_tables")
    return {**audit_tables(folder), "score": score_table(folder)}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with make_client(tmp_path_factory.mktemp("wp16_home"), "local", 2, "http://127.0.0.1") as c:
        yield c


# ── driving the server ───────────────────────────────────────────────────────


def post(client, pid: str, decision: dict) -> tuple[int, dict]:
    response = client.post(f"/api/projects/{pid}/decisions", json=decision)
    return response.status_code, response.json()


def accepted(client, pid: str, decision: dict) -> dict:
    code, body = post(client, pid, decision)
    assert code == 200, body
    return body


def refused(client, pid: str, decision: dict, code: str) -> dict:
    status, body = post(client, pid, decision)
    assert status == 409, body
    assert body["error"]["code"] == code, body
    assert body["error"]["exits"], "a refusal always offers a way forward"
    return body["error"]


def open_project(client, path: Path, lens: str, target: str, purpose: str) -> str:
    response = client.post("/api/projects", json={"path": str(path)})
    assert response.status_code == 200, response.text
    pid = response.json()["id"]
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"}, timeout=120)
    accepted(client, pid, {"kind": "set_lens", "lenses": [lens]})
    accepted(client, pid, {"kind": "set_target", "column": target})
    prepare(client, pid, {"kind": "set_purpose", "purpose": purpose})
    accepted(client, pid, {"kind": "set_purpose", "purpose": purpose})
    return pid


SPLIT = {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}


def up_to_the_split(client, pid: str) -> None:
    prepare(client, pid, SPLIT)
    wait_for(client, pid, {"findings": "fresh"}, timeout=120)  # the split question needs them


def sealed_rows(client, pid: str) -> set[int]:
    """The held-out rows of the current split, read from its parquet on disk."""
    view = wait_for(client, pid, {"split": "fresh"}, timeout=120)
    cache = client.app.state.service.workspace.cache_dir(pid)
    frame = pd.read_parquet(artifact_dir(cache, "split", view["stages"]["split"]["key"])
                            / "frames" / "sealed.parquet")
    return set(frame["row_id"].astype(int))


def assignment(client, pid: str) -> pd.DataFrame:
    view = wait_for(client, pid, {"split": "fresh"}, timeout=120)
    cache = client.app.state.service.workspace.cache_dir(pid)
    return pd.read_parquet(artifact_dir(cache, "split", view["stages"]["split"]["key"])
                           / "frames" / "assignment.parquet")


def records(client, pid: str) -> list[dict]:
    return client.get(f"/api/projects/{pid}").json()["decisions"]


def writer(client, pid: str, kind: str) -> str:
    return [r for r in records(client, pid) if r["decision"]["kind"] == kind][-1]["id"]


def finding(client, pid: str, prefix: str) -> dict:
    found = client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]["findings"]
    return next(f for f in found if f["id"].startswith(prefix))


def option(f: dict, key: str) -> dict:
    return next(o for o in f["repairs"] if o["key"] == key)


def outcome_values(client, pid: str, column: str) -> pd.Series:
    """The column by row id, as the working table holds it."""
    store = client.app.state.service.store(pid)
    return store.materialize([column], None)[column]


def independent_draw(row_ids: set[int], holdout: float = 0.2, seed: int = 0) -> set[int]:
    """The held-out rows a plain random draw takes over these rows, by scikit-learn directly: the
    rows sorted by id, positions split by ``train_test_split`` (no strata, no groups)."""
    from sklearn.model_selection import train_test_split

    ids = np.sort(np.fromiter(row_ids, dtype=np.int64))
    _, test = train_test_split(np.arange(len(ids)), test_size=holdout, random_state=seed)
    return set(ids[test].tolist())


# ── 1 · membership by identity ───────────────────────────────────────────────


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_after_the_split_outcome_repairs_move_no_held_out_row(client, tables, purpose):
    """Three impossible ``sbp`` values (999, 0, 1,200) on the audit's table, found after the split.

    Before this guard the set-to-missing repair was accepted and re-drew the seal: 119–130 of 200
    held-out rows crossed into training, and as many training rows, whose outcomes had trained the
    fits already seen, became "sealed" (RO-02; Dwork et al. 2015, Science 349:636: "Reusing a
    holdout set adaptively multiple times can easily lead to overfitting to the holdout set
    itself"). That number is recomputed here with scikit-learn directly (130).

    Reference: **0** held-out rows move, read from the split's parquet before and after. The
    set-to-missing repair is refused after the split (BLUEPRINT §12: "refuse outcome repairs after
    the split") with two exits: exclude those rows, which takes the three rows out of the analysis
    and moves no held-out row (measured), or the re-seal, a recorded withdraw-repair-redraw.
    """
    pid = open_project(client, tables["sbp"], "clinical", "sbp", purpose)
    up_to_the_split(client, pid)
    impossible = finding(client, pid, "pack::clinical::impossible_vs_extreme")
    # The user keeps the values at first (the Router's order is tested on its own below).
    accepted(client, pid, refused(client, pid, SPLIT, "settle_first")["exits"][-1]["decision"])
    accepted(client, pid, SPLIT)
    prepare(client, pid, {"kind": "select_models", "models": ["linear"]})
    accepted(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    before = sealed_rows(client, pid)
    split_writer = writer(client, pid, "set_split")
    reseal = {"kind": "revert", "decision_id": split_writer}

    # The independent path agrees with the app's draw, and shows what an unguarded repair would do.
    sbp = outcome_values(client, pid, "sbp")
    universe = set(sbp.index[sbp.notna()].astype(int))
    assert len(before) == 200 and before == independent_draw(universe)
    bad = set(sbp.index[(sbp < 40) | (sbp > 300)].astype(int))
    assert len(bad) == 3
    would_move = len(before - independent_draw(universe - bad))
    assert 119 <= would_move <= 130, would_move  # the audit's 119–130: the scenario discriminates

    # A repair to a predictor still passes: the draw does not read `bmi`.
    both = option(impossible, "set_missing")["decision"]
    predictor_only = {**both, "params": {"bands": {"bmi": both["params"]["bands"]["bmi"]}}}
    accepted(client, pid, predictor_only)
    assert sealed_rows(client, pid) == before

    # Setting the outcome's impossible values to missing is refused, with two exits: exclude those
    # rows (the draw stays as it is), or re-seal.
    for decision in (both, {**both, "params": {"bands": {"sbp": both["params"]["bands"]["sbp"]}}}):
        error = refused(client, pid, decision, "sealed")
        assert "`sbp`" in error["message"]
        assert error["exits"][-1]["decision"] == reseal
        assert error["exits"][0]["decision"]["option"] == "exclude_rows"
        assert sealed_rows(client, pid) == before
    accepted(client, pid, error["exits"][0]["decision"])
    after = sealed_rows(client, pid)
    moved = len(before - after) + len(after - before)
    assert moved == 0, moved
    # ...and the three impossible outcomes left the analysis.
    assert not bad & set(assignment(client, pid)["row_id"].astype(int))

    # set_target after the split: refused, with the re-seal as the exit; reverting the answer too.
    error = refused(client, pid, {"kind": "set_target", "column": "age"}, "sealed")
    assert error["exits"] == [{"label": error["exits"][0]["label"], "decision": reseal}]
    refused(client, pid, {"kind": "revert", "decision_id": writer(client, pid, "set_target")}, "sealed")
    assert sealed_rows(client, pid) == before

    # The exit is the re-seal path, and it works: withdraw, repair, draw again — each one recorded.
    accepted(client, pid, reseal)
    accepted(client, pid, both)
    up_to_the_split(client, pid)
    accepted(client, pid, SPLIT)
    redrawn = sealed_rows(client, pid)
    assert not bad & redrawn and len(redrawn) == math.ceil(0.2 * len(universe - bad))
    kinds = [r["decision"]["kind"] for r in records(client, pid)]
    assert kinds[-3:] == ["revert", "apply_repair", "set_split"]


def test_set_task_after_the_split_waits_for_a_reseal(client, tables):
    """A five-level numeric outcome reads as regression or as multiclass; the held-out rows are
    stratified by its classes only under multiclass, so a changed task draws them again. The same
    task answered again changes nothing and is accepted. Reference: the held-out rows are the same
    set before and after, read from disk."""
    pid = open_project(client, tables["score"], "clinical", "score", "prediction")
    up_to_the_split(client, pid)
    accepted(client, pid, SPLIT)
    before = sealed_rows(client, pid)
    info = client.get(f"/api/projects/{pid}/stages/target_info").json()["artifact"]
    detected = info["task"]
    assert detected in ("regression", "multiclass")
    other = "multiclass" if detected == "regression" else "regression"
    error = refused(client, pid, {"kind": "set_task", "column": "score", "task": other}, "sealed")
    assert error["exits"][-1]["decision"] == {"kind": "revert",
                                              "decision_id": writer(client, pid, "set_split")}
    accepted(client, pid, {"kind": "set_task", "column": "score", "task": detected})
    assert sealed_rows(client, pid) == before
    # Withdrawing the seal lets the task change, and the seal is then drawn again for it.
    accepted(client, pid, error["exits"][-1]["decision"])
    accepted(client, pid, {"kind": "set_task", "column": "score", "task": other})
    up_to_the_split(client, pid)
    accepted(client, pid, SPLIT)
    redrawn = sealed_rows(client, pid)
    assert len(redrawn) == 160 and redrawn != before  # what the guard kept from happening silently


def test_the_router_orders_impossibility_and_code_repairs_before_the_split(client, tables):
    """The split question waits for the table's checks, and a finding whose repair would rewrite the
    outcome is settled first: repaired, or kept as it is, by a recorded answer. Impossible values
    (``sbp`` on the audit's table) and missing-value codes (``item_14``'s ``9`` on the survey
    sample) alike. Reference: 409 ``settle_first`` naming the outcome, with the finding's own
    repairs and "keep the values" as exits; once repaired, the seal is drawn over the repaired
    values and none of the three impossible rows can be held out."""
    from turbotab.core.decisions import ProjectState
    from turbotab.core.interview import route

    # The Router: while the findings are being computed, the split question waits on them.
    state = ProjectState(lens=["clinical"], target="sbp", purpose="prediction",
                         grain={"grain": "one_row_per_unit"}, roles={"age": "covariate"},
                         exclusions=[], missing={"strategy": "complete_case"})
    steps = {s.key: s for s in route(state, {"findings": {"status": "running"}}, {}, [])}
    assert steps["split"].status == "waiting" and "findings" in steps["split"].waiting_on

    pid = open_project(client, tables["sbp"], "clinical", "sbp", "prediction")
    up_to_the_split(client, pid)
    error = refused(client, pid, SPLIT, "settle_first")
    assert "`sbp`" in error["message"]
    keys = [e["decision"].get("option") or e["decision"]["kind"] for e in error["exits"]]
    # Not "mark `bmi` unusable": that option is about a predictor, not what the draw reads.
    assert keys == ["set_missing", "exclude_rows", "dismiss_finding"]
    accepted(client, pid, error["exits"][0]["decision"])  # set the impossible values to missing
    up_to_the_split(client, pid)
    accepted(client, pid, SPLIT)
    sbp = outcome_values(client, pid, "sbp")
    assert int(sbp.isna().sum()) == 3
    sealed = sealed_rows(client, pid)
    assert len(sealed) == math.ceil(0.2 * 997) and not set(sbp.index[sbp.isna()].astype(int)) & sealed

    from turbotab.server.tests.conftest import SAMPLES

    pid = open_project(client, SAMPLES / "survey_sentinels.csv", "survey", "item_14", "prediction")
    up_to_the_split(client, pid)
    error = refused(client, pid, SPLIT, "settle_first")
    assert "`item_14`" in error["message"]
    assert error["exits"][-1]["decision"]["kind"] == "dismiss_finding"
    repair = next(e["decision"] for e in error["exits"] if e["decision"]["kind"] == "apply_repair")
    assert 9 in [int(v) for v in repair["params"]["values"]["item_14"]]


def test_every_answer_the_draw_reads_waits_for_a_reseal():
    """The guard by kind, on a project's state alone (no server): after a split that holds rows
    out, the outcome, its task, a chronological request and repairs to the unit column wait for a
    re-seal; under cross-validation alone nothing is held out and nothing waits."""
    from turbotab.core.decisions import ProjectState, Refusal, validate

    base = dict(lens=["clinical"], target="sbp", purpose="prediction", task="regression",
                grain={"grain": "repeated", "id_column": "pid"}, unit="row",
                repeat_kind={"repeat_kind": "time_points", "time_column": "visit"},
                temporal={"temporal": False}, roles={"age": "covariate", "pid": "identifier"})
    columns = ["pid", "visit", "age", "sbp"]
    info = {c: {"dtype": "numeric", "n_unique": 500, "n_missing": 0} for c in columns}
    info["sbp"]["n_unique"] = 12  # reads as regression or as multiclass

    def codes(column: str) -> dict:
        """A missing-value code finding on ``column``, as the findings stage serves one."""
        fid = f"sentinel_missing__{column}"
        decision = {"kind": "apply_repair", "finding_id": fid, "option": "set_missing",
                    "params": {"values": {column: [999.0]}}}
        return {"id": fid, "affected_columns": [column],
                "repairs": [{"key": "set_missing", "label": "Set to missing", "effect": "values",
                             "decision": decision}]}

    findings = {"findings": [codes("pid"), codes("age")]}

    def ctx(**split) -> dict:
        state = ProjectState(**base, split={"holdout": 0.2, "seed": 0, "folds": 5, **split})
        return {"columns": columns, "column_info": info, "state": state, "target": "sbp",
                "task": "regression", "artifact": lambda s: findings if s == "findings" else None}

    for decision in ({"kind": "set_target", "column": "age"},
                     {"kind": "set_task", "column": "sbp", "task": "multiclass"},
                     {"kind": "set_temporal", "temporal": True, "time_column": "visit"},
                     codes("pid")["repairs"][0]["decision"]):
        with pytest.raises(Refusal) as caught:
            validate(decision, ctx())
        assert caught.value.code == "sealed", decision
        validate(decision, ctx(holdout=0.0))  # cross-validation alone: nothing held out
    validate({"kind": "set_task", "column": "sbp", "task": "regression"}, ctx())  # the same task
    # The same outcome again keeps its task answer, even one that overrode the detection.
    answered = {**ctx(), "task": "multiclass",
                "state": ctx()["state"].model_copy(update={"task": "multiclass"}),
                "artifact": lambda s: {"column": "sbp", "task": "regression"} if s == "target_info" else None}
    validate({"kind": "set_target", "column": "sbp"}, answered)
    validate(codes("age")["repairs"][0]["decision"], ctx())  # a predictor: the draw never reads it


# ── 2 · outcome eligibility ──────────────────────────────────────────────────

BMI_RULE = {"column": "bmi", "low": 18.5, "high": 30, "reason": "normal range"}


def ols_slope(frame: pd.DataFrame) -> float:
    """The fiber coefficient of bmi ~ fiber + age, by least squares in NumPy."""
    X = np.column_stack([np.ones(len(frame)), frame["fiber_g"], frame["age"]])
    beta, *_ = np.linalg.lstsq(X, frame["bmi"].to_numpy(), rcond=None)
    return float(beta[1])


@pytest.mark.parametrize("purpose", ["inference", "prediction"])
def test_an_eligibility_rule_on_the_outcome_is_refused_under_both_purposes(client, tables, purpose):
    """``bmi 18.5–30`` with ``bmi`` as the outcome. Today it was accepted, and the fiber slope (true
    −0.25) came out −0.092 (−0.104, −0.080): a tight interval around a truncated answer (RO-01).
    The stakes, recomputed by least squares on the same table: the full table recovers the true
    slope, the truncated one does not. Reference: 409 ``rule_on_outcome`` with exits, from the
    decision and from its preview, under each purpose."""
    frame = pd.read_csv(tables["bmi"])
    kept = frame[(frame["bmi"] >= 18.5) & (frame["bmi"] <= 30)]
    assert abs(ols_slope(frame) - TRUE_FIBER_SLOPE) < 0.02
    assert ols_slope(kept) - TRUE_FIBER_SLOPE > 0.1  # biased toward zero by more than 40%

    pid = open_project(client, tables["bmi"], "clinical", "bmi", purpose)
    rule = {"kind": "set_exclusions", "rules": [BMI_RULE]}
    prepare(client, pid, rule)
    wait_for(client, pid, {"findings": "fresh"}, timeout=120)
    error = refused(client, pid, rule, "rule_on_outcome")
    preview = client.post(f"/api/projects/{pid}/preview", json=rule)
    assert preview.status_code == 409 and preview.json()["error"]["code"] == "rule_on_outcome"
    assert "`bmi` is the outcome" in error["message"]
    exits = error["exits"]
    assert exits[0]["decision"] == {"kind": "set_exclusions", "rules": []}
    # The impossible-values repair for the outcome's impossible values (bmi below 8 here).
    repairs = [e["decision"]["option"] for e in exits if (e["decision"] or {}).get("kind") == "apply_repair"]
    assert repairs == ["set_missing", "exclude_rows"]
    assert exits[-1]["decision"] is None  # restrict by a variable measured before the outcome

    # Mixed with a rule on a baseline variable, the exit keeps that one; a range set by the
    # outcome's level is a rule on the outcome too.
    age = {"column": "age", "low": 30, "high": 70, "reason": "adults of working age"}
    error = refused(client, pid, {"kind": "set_exclusions", "rules": [age, BMI_RULE]}, "rule_on_outcome")
    assert error["exits"][0]["decision"]["rules"] == [{**age, "kind": "range", "by": None}]
    by_outcome = {"column": "fiber_g", "high": 60, "reason": "plausible",
                  "by": {"column": "bmi", "ranges": {"30": [None, 50]}}}
    refused(client, pid, {"kind": "set_exclusions", "rules": [by_outcome]}, "rule_on_outcome")

    # A rule on a baseline variable is accepted, and its preview never draws the outcome.
    preview = client.post(f"/api/projects/{pid}/preview", json={"kind": "set_exclusions", "rules": [age]})
    assert preview.status_code == 200, preview.text
    views = preview.json()["views"]
    assert any(v.get("column") == "age" for v in views)
    assert all(v.get("column") != "bmi" for v in views)
    accepted(client, pid, {"kind": "set_exclusions", "rules": [age]})


def test_the_exclusions_preview_never_draws_the_outcome(client, tables):
    """A rule on the outcome recorded before the guard existed (planted in the log as an older
    project would hold it) still draws no histogram of the outcome: the preview's views name no
    ``bmi`` column. Reference: no view whose column is the outcome."""
    from turbotab.core.decisions import SetExclusions

    pid = open_project(client, tables["bmi"], "clinical", "bmi", "inference")
    prepare(client, pid, {"kind": "set_exclusions", "rules": []})
    service = client.app.state.service
    service.log(pid).append(SetExclusions(rules=[BMI_RULE]))  # the validator is bypassed: an old log
    service.engine.on_decision(pid)
    wait_for(client, pid, {"cohort": "fresh"}, timeout=120)
    for rules in ([], [{"column": "age", "low": 30, "reason": "adults"}]):
        preview = client.post(f"/api/projects/{pid}/preview", json={"kind": "set_exclusions", "rules": rules})
        assert preview.status_code == 200, preview.text
        assert all(v.get("column") != "bmi" for v in preview.json()["views"]), preview.json()["views"]


def test_a_rule_cannot_become_a_rule_on_the_outcome(client, tables):
    """The outcome moving onto a column a rule reads is the same selection: choosing it as the
    outcome, or undoing the answer that moved the outcome away from it, is refused with the way
    out. Reference: 409 ``rule_on_outcome``."""
    pid = open_project(client, tables["bmi"], "clinical", "fiber_g", "prediction")
    accepted(client, pid, {"kind": "set_target", "column": "bmi"})
    moved_away = writer(client, pid, "set_target")  # the answer that moved the outcome off fiber_g
    rule = {"kind": "set_exclusions",
            "rules": [{"column": "fiber_g", "high": 80, "reason": "a plausible daily intake"}]}
    prepare(client, pid, rule)
    accepted(client, pid, rule)
    error = refused(client, pid, {"kind": "set_target", "column": "fiber_g"}, "rule_on_outcome")
    assert error["exits"][0]["decision"] == {"kind": "set_exclusions", "rules": []}
    refused(client, pid, {"kind": "revert", "decision_id": moved_away}, "rule_on_outcome")
