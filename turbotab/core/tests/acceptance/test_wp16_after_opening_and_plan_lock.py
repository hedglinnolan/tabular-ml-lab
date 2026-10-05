"""WP16 acceptance tests 3 and 4: the seal holds at its edges (docs/turbotab-next/audit/AUDIT_REPORT.md
§5), and the methods text built from the log.

They close RO-05 and RO-12:

3. **After opening.** "The at-opening scores are stored and reported; a new seed after opening
   requires a recorded re-seal; a new outcome starts its own seal with scores withheld."
4. **Inference plan lock.** "Changes to exposure, adjustment set, exclusions or missing-data plan
   after the first coefficient is shown are recorded 'after the estimates were seen' and appear in
   the methods sentence. Source check: Gelman & Loken 2013."

And the package's last item: a methods text built from the append-only log folds out superseded
decisions. The intelligence fix run saw "`cvd_event` was analyzed as a time to event" survive a
later binary answer while the fit was logistic.

Every project is driven through the real server (the FastAPI app over its job runner), as the
auditors drove it (``I/j3_seal_reuse.py`` opened ``clinical_risk.csv`` and re-drew seeds 1–5, each
served at once). The references come from paths independent of the code under test:

* the held-out AUC at the opening is recomputed from the table: an unpenalized logistic regression
  fit by statsmodels on the split's training rows (read from its parquet on disk), scored on its
  held-out rows by a rank (Mann–Whitney) AUC in NumPy;
* the scores a fit holds back are read from its sealed frame on disk with pandas, and must appear
  in no response until their seal is opened;
* the plan lock's hash is recomputed with ``hashlib`` over the recorded plan's canonical JSON.

**Source check (test 4).** Gelman A, Loken E. "The garden of forking paths: Why multiple comparisons
can be a problem, even when there is no 'fishing expedition' or 'p-hacking' and the research
hypothesis was posited ahead of time." 14 Nov 2013 (sites.stat.columbia.edu/gelman/research/
unpublished/p_hacking.pdf, read 2026-10-03). Abstract: "Researcher degrees of freedom can lead to a
multiple comparisons problem, even in settings where researchers perform only a single analysis on
their data. The problem is there can be a large number of potential comparisons when the details of
data analysis are highly contingent on data, without the researcher having to perform any conscious
procedure of fishing or examining multiple p-values." §1.2 names the forks: "choices of control
variables in a regression, transformations, and data coding and excluding rules, as well as the
decision of which main effect or interaction to focus on." Each is a slot the lock records (the
roles, the exposure form, the repairs and confirmed readings, the exclusions), and test 4 changes
each kind it names after the estimates are displayed. The paper also asks, of a protocol said to be
decided ahead of time, "why not preregister it?": a lock in the software registers nothing outside
it, so it is never called prespecified or preregistered (MODELING_SEQUENCE §1 row 12).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core.graph import artifact_dir
from turbotab.core.tests.acceptance.test_wp16_seal_guards import (accepted, assignment, open_project,
                                                                   post, records, refused,
                                                                   sealed_rows, writer)
from turbotab.server.tests.conftest import SAMPLES, declare, make_client, prepare, wait_for

CLINICAL = SAMPLES / "clinical_risk.csv"
SPLIT = {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}
FAMILIES = ["linear", "boosted_trees"]
NEVER = ("prespecified", "pre-specified", "preregistered", "pre-registered", "registered")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with make_client(tmp_path_factory.mktemp("wp16_after"), "local", 2, "http://127.0.0.1") as c:
        yield c


# ── reading what the server holds ────────────────────────────────────────────


def view_of(client, pid: str) -> dict:
    return client.get(f"/api/projects/{pid}").json()


def served_fit(client, pid: str) -> dict:
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    return client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]


def disk_scores(client, pid: str) -> dict[str, dict[str, float]]:
    """The held-out scores the fresh fit holds back, read from its sealed frame on disk."""
    view = wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    cache = client.app.state.service.workspace.cache_dir(pid)
    frame = pd.read_parquet(artifact_dir(cache, "fit", view["stages"]["fit"]["key"]) / "frames"
                            / "sealed_scores.parquet")
    out: dict[str, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        out.setdefault(row.family, {})[row.metric] = float(row.value)
    return out


def numbers_in(obj: Any) -> set[float]:
    if isinstance(obj, bool):
        return set()
    if isinstance(obj, float):
        return {obj}
    if isinstance(obj, dict):
        return set().union(*(numbers_in(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(numbers_in(v) for v in obj)) if obj else set()
    return set()


def assert_withheld(bodies: list[Any], hidden: dict[str, dict[str, float]]) -> None:
    """None of ``hidden``'s scores, as a number or as printed to three or four places."""
    values = [v for scores in hidden.values() for v in scores.values() if np.isfinite(v)]
    assert values
    for body in bodies:
        seen, text = numbers_in(body), json.dumps(body, ensure_ascii=False)
        for v in values:
            assert v not in seen, (v, text[:300])
            for shown in (f"{v:.4f}", f"{v:.3f}"):
                assert re.search(rf"(?<![\d.]){re.escape(shown)}(?!\d)", text) is None, (shown, text[:300])


def responses(client, pid: str) -> list[Any]:
    """What a client reads of the project: the view, every stage, the methods text."""
    out = [view_of(client, pid), client.get(f"/api/projects/{pid}/methods").json()]
    for stage in client.app.state.service.engine.graph.stages():
        out.append(client.get(f"/api/projects/{pid}/stages/{stage.name}").json())
    return out


def opened_project(client, target: str = "readmit_30d") -> str:
    """``clinical_risk.csv`` under prediction, 20% held out (seed 0), two families fitted."""
    pid = open_project(client, CLINICAL, "clinical", target, "prediction")
    models = {"kind": "select_models", "models": FAMILIES}
    prepare(client, pid, models)
    accepted(client, pid, models)
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    return pid


def rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    """The Mann–Whitney AUC: the share of (event, non-event) pairs ordered right, ties one half."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):  # average ranks over ties
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic_holdout_auc(client, pid: str) -> float:
    """The held-out AUC of an unpenalized logistic regression on the project's predictors, fit by
    statsmodels on the split's training rows: the table's values, the recorded roles and event,
    the split read from disk."""
    import statsmodels.api as sm

    state = view_of(client, pid)["state"]
    target = state["target"]
    predictors = [c for c, r in state["roles"].items()
                  if r in ("exposure", "covariate") and c != target]
    store = client.app.state.service.store(pid)
    data = store.materialize([*predictors, target], None)
    X = sm.add_constant(pd.get_dummies(data[predictors], drop_first=True, dtype=float))
    y = (data[target].astype(str) == str(state["event"])).astype(float)
    split = assignment(client, pid)
    train = split.loc[split["partition"] == "train", "row_id"].to_numpy()
    held = split.loc[split["partition"] != "train", "row_id"].to_numpy()
    fit = sm.Logit(y.loc[train], X.loc[train]).fit(disp=0, method="newton", maxiter=200)
    return rank_auc(y.loc[held].to_numpy(), fit.predict(X.loc[held]).to_numpy())


# ── 3 · after opening ────────────────────────────────────────────────────────


def test_3_the_scores_at_the_opening_are_kept_in_the_record_and_reported(client):
    """Opened once, then changed: the at-opening scores stay the reported result.

    Reference: the held-out AUC of the declared logistic model, recomputed by statsmodels on the
    split's own training rows and scored on its held-out rows by a rank AUC. Before the fix nothing
    stored the opened score while the screen said it "is then fixed in the record" (RO-05). Now the
    opening record keeps every family's held-out scores (equal to the sealed frame on disk), its
    sentence states the declared family's, and the served fit reports them as ``at_opening`` after
    a post-seal change has moved the held-out score shown, saying the shown one is not
    independent."""
    pid = opened_project(client)
    before = served_fit(client, pid)
    assert before["holdout_sealed"] is True and before["at_opening"] is None
    at_disk = disk_scores(client, pid)
    reference = logistic_holdout_auc(client, pid)
    assert abs(at_disk["linear"]["auc"] - reference) < 1e-6, (at_disk, reference)

    view = accepted(client, pid, {"kind": "open_seal", "family": "linear"})
    opening = view["decisions"][-1]
    assert opening["decision"]["kind"] == "open_seal" and opening["decision"]["target"] == "readmit_30d"
    assert opening["decision"]["scores"] == at_disk  # kept in the append-only record
    assert opening["decision"]["metric"] == "auc" and opening["decision"]["n_holdout"] == 96
    assert f"held-out AUC `{reference:.3f}`" in opening["sentence"]
    fit = served_fit(client, pid)
    assert fit["at_opening"]["scores"] == at_disk and fit["at_opening"]["current"] is True
    assert fit["at_opening"]["seq"] == opening["seq"]
    assert {m["family"]: m["holdout"] for m in fit["models"]} == at_disk

    # A post-seal change moves the held-out score shown; the reported result does not move.
    rule = {"kind": "set_exclusions",
            "rules": [{"column": "age", "low": 40, "reason": "adults over 40"}]}
    changed = accepted(client, pid, rule)["decisions"][-1]
    assert changed["post_seal"] is True
    fit = served_fit(client, pid)
    shown = {m["family"]: m["holdout"]["auc"] for m in fit["models"]}
    assert shown["linear"] != pytest.approx(reference, abs=1e-6)  # the scenario discriminates
    assert fit["changed_after_seal"] is True
    at = fit["at_opening"]
    assert at["scores"]["linear"]["auc"] == pytest.approx(reference, abs=1e-6)
    assert at["current"] is False and at["family"] == "linear"
    assert "post-seal and not an independent test" in at["note"]
    assert fit["final_note"] == at["note"]  # the reported result is the opening's, not the shown
    # ...and the opening's sentence, in the methods text, carries the score.
    text = client.get(f"/api/projects/{pid}/methods").json()["text"]
    assert f"held-out AUC `{reference:.3f}`" in text
    # Under prediction, displaying the fit locks no inference plan.
    assert "lock_plan" not in [r["decision"]["kind"] for r in records(client, pid)]


def test_3_a_new_seed_after_opening_needs_a_recorded_reseal(client):
    """The audit re-drew seeds 1–5 after opening and was served each new held-out score at once.

    Reference: the held-out rows, read from the split's parquet, do not move until a re-seal is
    recorded: a new seed is refused (409 ``seal_opened``) with the re-seal as its exit, and so is
    the same new seed after withdrawing the split (the guard reads the state at the opening, not
    the last answer); the opened seed again is accepted. After the re-seal the new draw's scores
    (read from disk) reach no response until the new seal is opened, its opening says it is not an
    independent test, and the first opening's scores stay the reported result."""
    pid = opened_project(client)
    accepted(client, pid, {"kind": "open_seal", "family": "linear"})
    first = disk_scores(client, pid)
    rows = sealed_rows(client, pid)
    reseed = {**SPLIT, "seed": 1}

    error = refused(client, pid, reseed, "seal_opened")
    assert error["exits"][0]["decision"]["kind"] == "reseal"
    assert "seed 1 in place of seed 0" in error["message"]
    assert sealed_rows(client, pid) == rows
    # Withdrawing the split does not open a way round: the new seed is refused all the same.
    accepted(client, pid, {"kind": "revert", "decision_id": writer(client, pid, "set_split")})
    refused(client, pid, reseed, "seal_opened")
    accepted(client, pid, SPLIT)  # the opened draw again: the same rows, still open
    assert sealed_rows(client, pid) == rows
    assert served_fit(client, pid)["holdout_sealed"] is False

    # The recorded re-seal, then the new seed.
    view = accepted(client, pid, error["exits"][0]["decision"])
    reseal = view["decisions"][-1]
    assert reseal["decision"] == {"kind": "reseal", "reason": None, "target": "readmit_30d"}
    assert reseal["post_seal"] is True and view["state"]["seal_opened"] is None
    assert reseal["sentence"].startswith("After the held-out rows were opened, the seal was withdrawn")
    accepted(client, pid, reseed)
    assert sealed_rows(client, pid) != rows
    # The re-seal stays while the rows drawn after it are unopened: withdrawing it would count
    # them as opened at the first opening.
    refused(client, pid, {"kind": "revert", "decision_id": reseal["id"]}, "seal_opened")
    second = disk_scores(client, pid)
    fit = served_fit(client, pid)
    assert fit["holdout_sealed"] is True and all(m["holdout"] is None for m in fit["models"])
    assert fit["at_opening"]["scores"] == first and fit["at_opening"]["current"] is False
    assert_withheld(responses(client, pid), {f: {k: v for k, v in s.items() if v != first[f].get(k)}
                                             for f, s in second.items()})
    steps = {s["key"]: s for s in view_of(client, pid)["interview"]}
    assert steps["open_seal"]["status"] == "open"  # the new seal is opened in its turn

    view = accepted(client, pid, {"kind": "open_seal", "family": "linear"})
    reopening = view["decisions"][-1]
    assert "not an independent test" in reopening["sentence"]
    assert reopening["decision"]["scores"] == second
    fit = served_fit(client, pid)
    assert {m["family"]: m["holdout"] for m in fit["models"]} == second
    assert fit["at_opening"]["scores"] == first and fit["at_opening"]["seq"] < reopening["seq"]
    assert "not an independent test" in fit["final_note"]
    assert post(client, pid, {"kind": "open_seal", "family": "linear"})[0] == 409  # opened once


def test_3_a_new_outcome_starts_its_own_seal_with_scores_withheld(client):
    """After ``readmit_30d``'s seal was opened, ``length_of_stay_days`` becomes the outcome.

    Reference: the new outcome's held-out scores, read from the fit's sealed frame on disk, reach
    no response (the audit was served a new outcome's held-out R² with no seal ever drawn for it);
    the Router asks to open its seal; opening it serves exactly the frame's scores, and they are its
    own reported result. Every answer after the first opening stays marked post-seal. Going back to
    ``readmit_30d`` with its opened draw finds that seal open, with its own opening's scores."""
    pid = opened_project(client)
    view = accepted(client, pid, {"kind": "open_seal", "family": "linear"})
    opened_first = view["decisions"][-1]
    first = disk_scores(client, pid)
    error = refused(client, pid, {"kind": "set_target", "column": "length_of_stay_days"}, "sealed")
    accepted(client, pid, error["exits"][-1]["decision"])  # withdraw the seal (RO-02's exit)
    view = accepted(client, pid, {"kind": "set_target", "column": "length_of_stay_days"})
    moved = view["decisions"][-1]
    assert moved["post_seal"] is True and moved["sentence"].startswith("After the held-out rows were opened")
    assert view["state"]["seal_opened"] is None
    prepare(client, pid, SPLIT)
    accepted(client, pid, SPLIT)  # the new outcome's own seal: no re-seal needed
    hidden = disk_scores(client, pid)
    fit = served_fit(client, pid)
    assert fit["holdout_sealed"] is True and fit["at_opening"] is None
    assert all(m["holdout"] is None for m in fit["models"])
    assert_withheld(responses(client, pid), hidden)
    steps = {s["key"]: s for s in view_of(client, pid)["interview"]}
    assert steps["open_seal"]["status"] == "open"
    refused(client, pid, {"kind": "reseal"}, "nothing_to_reseal")  # nothing opened to re-seal

    view = accepted(client, pid, {"kind": "open_seal", "family": "linear"})
    own = view["decisions"][-1]
    assert own["decision"]["target"] == "length_of_stay_days" and own["decision"]["scores"] == hidden
    fit = served_fit(client, pid)
    assert {m["family"]: m["holdout"] for m in fit["models"]} == hidden
    assert fit["at_opening"]["seq"] == own["seq"] and fit["at_opening"]["current"] is True

    # Back to the first outcome, with the draw that was opened: its own seal, open, its own scores.
    accepted(client, pid, {"kind": "revert", "decision_id": writer(client, pid, "set_split")})
    accepted(client, pid, {"kind": "set_target", "column": "readmit_30d"})
    prepare(client, pid, SPLIT)
    view = accepted(client, pid, SPLIT)
    assert view["state"]["seal_opened"] is True
    fit = served_fit(client, pid)
    assert fit["at_opening"]["seq"] == opened_first["seq"] and fit["at_opening"]["scores"] == first


# ── 4 · the inference analysis-plan lock ─────────────────────────────────────


def plan_table(folder: Path) -> Path:
    """800 people: BMI on fiber (true slope −0.2 per g), age and sodium; 6% of ages blank."""
    rng = np.random.default_rng(1616)
    n = 800
    age = rng.normal(50, 10, n).round(0)
    fiber = rng.gamma(4, 5, n).round(1)
    sodium = rng.normal(3.4, 0.8, n).round(2)
    bmi = (27 - 0.2 * (fiber - 20) + 0.05 * (age - 50) + 0.6 * (sodium - 3.4)
           + rng.normal(0, 4, n)).round(1)
    age[rng.random(n) < 0.06] = np.nan
    path = folder / "plan_lock.csv"
    pd.DataFrame({"participant_id": [f"Q{i:04d}" for i in range(n)], "age": age, "fiber_g": fiber,
                  "sodium_g": sodium, "bmi": bmi}).to_csv(path, index=False)
    return path


ROLES = {"participant_id": "identifier", "fiber_g": "exposure", "age": "covariate",
         "sodium_g": "covariate"}


def canonical_sha256(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def test_4_changes_after_the_first_coefficient_is_shown_are_recorded_after_the_estimates_were_seen(
        client, tmp_path):
    """Under inference, coefficients and p-values were live while the plan changed (RO-12).

    Reference: the record's own order. Every decision before the first coefficient is served to a
    client is unmarked, including one made after the fit was computed but before it was displayed;
    serving it records the lock once, with the plan then in force and its SHA-256 (recomputed here
    with hashlib); every later decision is marked ``after_estimates`` and its sentence begins
    "After the estimates were seen". The four changes test 4 names (the exposure, the adjustment
    set, the exclusions and the missing-data plan) and a transformation (Gelman & Loken's forks) are
    made after it, and each sentence appears in the methods text, beside the plan as declared. An
    exclusion answer replaced before anything was displayed is folded out of the methods text. The
    lock is never undone, never recorded twice, and never called prespecified or preregistered."""
    path = plan_table(tmp_path)
    response = client.post("/api/projects", json={"path": str(path)})
    pid = response.json()["id"]
    declare(pid, {"code_or_count:age": "amount"}, fixture=path.name)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"}, timeout=120)
    accepted(client, pid, {"kind": "set_lens", "lenses": ["clinical"]})
    accepted(client, pid, {"kind": "set_target", "column": "bmi"})
    for decision in ({"kind": "set_purpose", "purpose": "inference"},
                     {"kind": "set_roles", "roles": ROLES},
                     {"kind": "set_exclusions",
                      "rules": [{"column": "fiber_g", "high": 60, "reason": "a plausible intake"}]},
                     {"kind": "set_exclusions", "rules": []},  # replaced before any estimate
                     {"kind": "set_missing", "strategy": "complete_case"},
                     {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5},
                     {"kind": "select_models", "models": ["linear"]}):
        prepare(client, pid, decision)
        accepted(client, pid, decision)
    replaced = [r for r in records(client, pid) if r["decision"]["kind"] == "set_exclusions"][0]
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    # Computed, not yet displayed: a change now is made before the estimates were seen.
    declared_rule = {"kind": "set_exclusions",
                     "rules": [{"column": "fiber_g", "high": 80, "reason": "a plausible intake"}]}
    accepted(client, pid, declared_rule)
    assert not any(r["after_estimates"] for r in records(client, pid))
    assert "lock_plan" not in [r["decision"]["kind"] for r in records(client, pid)]

    fit = served_fit(client, pid)  # the first coefficient is shown
    assert any(c["feature"] == "fiber_g" for c in fit["models"][0]["coefficients"])
    log = records(client, pid)
    lock = log[-1]
    assert lock["decision"]["kind"] == "lock_plan" and not lock["after_estimates"]
    plan = lock["decision"]["plan"]
    state = view_of(client, pid)["state"]
    for slot in ("target", "purpose", "roles", "exclusions", "missing", "models", "split"):
        assert plan[slot] == state[slot], slot
    assert lock["decision"]["digest"] == canonical_sha256(plan)
    assert lock["sentence"] == (
        "The analysis plan recorded above was declared in TurboTab before any estimate was "
        f"displayed (SHA-256 `{canonical_sha256(plan)[:12]}`); every later change is marked as made "
        "after the estimates were seen.")
    served_fit(client, pid)  # displayed again: no second lock
    assert [r["decision"]["kind"] for r in records(client, pid)].count("lock_plan") == 1
    assert post(client, pid, {"kind": "lock_plan"})[1]["error"]["code"] == "plan_already_locked"
    refused_undo = post(client, pid, {"kind": "revert", "decision_id": lock["id"]})
    assert refused_undo[0] == 409 and refused_undo[1]["error"]["code"] == "plan_stays_locked"

    # The forks, each after the estimates were seen.
    changes = [
        ("exposure", {"kind": "set_roles", "roles": {**ROLES, "sodium_g": "exposure"}}),
        ("adjustment set", {"kind": "set_roles", "roles": {**ROLES, "sodium_g": "exposure",
                                                            "age": "excluded"}}),
        ("exclusions", {"kind": "set_exclusions",
                        "rules": [{"column": "fiber_g", "high": 60, "reason": "a plausible intake"}]}),
        ("missing-data plan", {"kind": "set_missing", "strategy": "multiple_imputation"}),
        ("transformation", {"kind": "set_exposure_form", "column": "fiber_g", "form": "spline",
                            "knots": 4}),
    ]
    marked: dict[str, dict] = {}
    for name, decision in changes:
        seq = max(r["seq"] for r in records(client, pid))
        accepted(client, pid, decision)
        made = [r for r in records(client, pid) if r["seq"] > seq]
        assert made and all(r["after_estimates"] for r in made), name
        own = next(r for r in made if r["decision"]["kind"] == decision["kind"])
        assert own["sentence"].startswith("After the estimates were seen, "), (name, own["sentence"])
        marked[name] = own
    assert view_of(client, pid)["state"]["roles"]["age"] == "excluded"

    methods = client.get(f"/api/projects/{pid}/methods").json()
    text = methods["text"]
    assert methods["seen_from"] == lock["seq"]
    for name, record in marked.items():
        assert record["sentence"] in text, name
    # The plan as declared stays beside the changes: the roles and the complete cases it held.
    declared = {r["decision"]["kind"]: r for r in log if r["seq"] < lock["seq"]}
    for kind in ("set_roles", "set_missing", "set_exclusions"):
        line = next(x for x in methods["lines"] if x["record_id"] == declared[kind]["id"])
        assert line["in_force"] is False and not line["after_estimates"], kind
    assert lock["sentence"] in text
    # Superseded before anything was seen: folded out (its rule comes back later, after the lock,
    # as a change of its own).
    assert replaced["id"] not in [x["record_id"] for x in methods["lines"]]
    lowered = text.lower()
    assert not any(word in lowered for word in NEVER), [w for w in NEVER if w in lowered]
    every = " ".join(r["sentence"] or "" for r in records(client, pid)).lower()
    assert not any(word in every for word in NEVER)


# ── the methods text folds out what was superseded before anything was seen ──


def test_the_methods_text_drops_a_time_to_event_sentence_a_later_task_answer_superseded(client, tmp_path):
    """The intelligence fix run: after ``set_task`` binary, "`cvd_event` was analyzed as a time to
    event" stayed in a methods text built from the log. Reference: the Record keeps both answers
    (it is append-only), the methods text keeps only the one in force: the binary task, whose
    sentence says the follow-up is not used, and no sentence claiming a time to event."""
    from turbotab.core.tests.acceptance.test_wp12b_cox_mixed_gee import _staggered_entry_cohort

    path = tmp_path / "cohort_tte_null.csv"
    _staggered_entry_cohort().to_csv(path, index=False)
    response = client.post("/api/projects", json={"path": str(path)})
    pid = response.json()["id"]
    declare(pid, {}, fixture=path.name)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"}, timeout=120)
    accepted(client, pid, {"kind": "set_lens", "lenses": ["clinical"]})
    accepted(client, pid, {"kind": "set_target", "column": "cvd_event"})
    wait_for(client, pid, {"target_info": "fresh"}, timeout=120)
    accepted(client, pid, {"kind": "set_task", "column": "cvd_event", "task": "time_to_event"})
    accepted(client, pid, {"kind": "set_follow_up", "column": "cvd_event",
                           "time_column": "followup_years"})
    accepted(client, pid, {"kind": "set_task", "column": "cvd_event", "task": "binary"})
    said = [r["sentence"] for r in records(client, pid)]
    assert any("was analyzed as a time to event" in s for s in said)  # the Record keeps it
    methods = client.get(f"/api/projects/{pid}/methods").json()
    assert methods["seen_from"] is None
    assert "was analyzed as a time to event" not in methods["text"]
    assert "`cvd_event` was modeled as a `binary` task" in methods["text"]
    assert "no longer analyzed as a time to event" in methods["text"]
    assert [x["kind"] for x in methods["lines"]].count("set_task") == 1
    assert "set_follow_up" not in [x["kind"] for x in methods["lines"]]
