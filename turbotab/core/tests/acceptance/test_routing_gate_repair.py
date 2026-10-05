"""The routing gate's repair round (WP16–WP18 variants and the readings ledger's residue).

The routing verifier reopened eight items, each on a fresh fixture of its own. Each test below
replays one through the real server, on a generator written here (own seeds), and asserts the
behavior the rule asks for, by the rule rather than by the names it was found on:

1. **WP16-1 · an outcome written as text.** Its read-numbers repair is offered where the task
   question refuses it, and once read as numbers the outcome's impossible values are found by the
   plausibility check and settled before the seal is drawn (audit RO-02's ordering guard), so the
   working table holds none of them.
2. **WP16-2 · a rule on a time-to-event outcome's follow-up time.** It is a rule on the outcome
   (RO-01: a time to event is its time and its event), refused under both purposes with the
   outcome's own answers as exits: follow-up counted from a landmark (the customary "first years
   excluded"), or ended at a horizon. Taken, each matches lifelines with delayed entry or
   administrative censoring on the same rows.
3. **WP16-4 · the purpose switched after coefficients were shown under prediction.** The plan is
   locked at the switch and says estimates were seen; every later answer is marked so; a lock
   posted by hand is refused, so no answer made before any estimate is marked as made after one.
4. **WP17-2 · NHANES linked mortality.** ``MORTSTAT`` beside ``PERMTH_INT``: under the clinical
   lens the follow-up question is asked whatever the names; ``PERMTH_INT`` is read as the follow-up
   time and proposed "time"; the fit is Cox, matching lifelines, and no log-odds is served.
5. **WP17-3 · a direct effect.** It asks whether a covariate the criterion leaves out is a common
   cause of a mediator and the outcome, and whether the exposure's effect differs with a mediator's
   level (block and record); the caption names the controlled direct effect and the mediators held
   fixed; the estimate is least squares' on the identifying set.
6. **WP18-0 · the roles the estimand reads.** Under inference the estimand question carries the ask
   card for every role that rode along unconfirmed, before the exposure is chosen.
7. **The ledger's residue · the energy model's kcal per unit.** The all-components model reads each
   source's kcal per unit from its settled unit, never its name: an `alcohol_g` holding US standard
   drinks is asked its unit, and once recorded the relative effect is NumPy's at 98 kcal per drink.
8. **Gate 6 re-run · the cluster reading after the grouping question.** Each confirmation, and the
   grouping question's "they group nothing", is honored: CR2 by the household, or HC3, each equal to
   its definition written out.

**Source checks.** (5) Valeri L, VanderWeele TJ. Mediation analysis allowing for exposure–mediator
interactions and causal interpretation. *Psychol Methods* 2013;18:137–150 (PMC3659198, read on
2026-10-05): "In order to ensure identifiability of controlled direct effect, two assumptions are
needed: namely those of (i) no unmeasured confounding of the treatment-outcome relationship and
(ii) no unmeasured confounding of mediator-outcome relationship"; with an exposure–mediator
interaction the controlled direct effect is "(θ₁+θ₃m)(a−a*)". (3) Gelman A, Loken E. The garden of
forking paths (2013), abstract: "Researcher degrees of freedom can lead to a multiple comparisons
problem, even in settings where researchers perform only a single analysis on their data."
(2) Moons et al. 2019, PROBAST explanation, item 4.6, as quoted in ``estimand.py``.
"""
from __future__ import annotations

import inspect
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core import decisions as d, estimand
from turbotab.core.decisions import ProjectState
from turbotab.core.tests.acceptance import references as ref
from turbotab.core.tests.acceptance.server_drive import local_server, open_project
from turbotab.core.tests.truths import Truth

VALERI_QUOTE = ("In order to ensure identifiability of controlled direct effect, two assumptions are "
                "needed: namely those of (i) no unmeasured confounding of the treatment-outcome "
                "relationship and (ii) no unmeasured confounding of mediator-outcome relationship")
GELMAN_LOKEN = ("Researcher degrees of freedom can lead to a multiple comparisons problem, even in "
                "settings where researchers perform only a single analysis on their data")


def _csv(frame: pd.DataFrame, tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def _flat(text: Any) -> str:
    return " ".join(str(text).split())


def _coef(fit: dict[str, Any], family: str, feature: str) -> dict[str, Any]:
    model = next(m for m in fit["models"] if m["family"] == family)
    return next(r for r in model["coefficients"] or [] if r["feature"] == feature)


def _error(response: Any, code: str) -> dict[str, Any]:
    assert response.status_code == 409, response.text[:600]
    error = response.json()["error"]
    assert error["code"] == code, error
    return error


def _records(drive: Any) -> list[dict[str, Any]]:
    return drive.view()["decisions"]


def _until(read: Any, done: Any, timeout: float = 120.0) -> Any:
    """``read()`` again until ``done`` holds of it (a stage recomputing after an answer)."""
    import time

    end = time.monotonic() + timeout
    while True:
        value = read()
        if done(value):
            return value
        assert time.monotonic() < end, value
        time.sleep(0.1)


def _served(drive: Any, stage: str) -> dict[str, Any]:
    drive.artifact(stage)
    return drive.c.get(f"/api/projects/{drive.pid}/stages/{stage}").json()["artifact"]


# ═════════════════════════════════════════════════════════════════════════════
# 1 · an outcome written as text: read as numbers, then checked before the seal
# ═════════════════════════════════════════════════════════════════════════════

IMPOSSIBLE_ROWS = [7, 222, 640]
IMPOSSIBLE_VALUES = [999, 0, 1300]


def _text_sbp(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(5151)
    age = rng.normal(57, 9, n).round(0)
    salt = rng.normal(8.5, 2.5, n).round(1)
    sbp = (112 + 0.55 * (age - 57) + 1.4 * salt + rng.normal(0, 11, n)).round(0).astype(object)
    sbp[IMPOSSIBLE_ROWS] = IMPOSSIBLE_VALUES
    for i in range(4, n, 53):  # SAS's "." for missing, about 2% of the rows: the column is text
        sbp[i] = "."
    return pd.DataFrame({"pid": [f"S{i:04d}" for i in range(n)], "age": age, "salt_g": salt,
                         "sbp": sbp})


def test_1_an_outcome_written_as_text_is_read_and_checked_before_the_seal(tmp_path):
    """The verifier's p05: once read as numbers, the outcome's 999, 0 and 1300 were never checked,
    and the split drew the seal over them. Reference: pandas reads the file's column as numbers
    (the "." blank) and the three planted impossible values are exactly what the settled repair
    blanks; every other value is as written."""
    frame = _text_sbp()
    path = _csv(frame, tmp_path, "sbp_text")
    parsed = pd.to_numeric(frame["sbp"], errors="coerce")
    assert parsed.isna().sum() == len(range(4, len(frame), 53))
    truth = Truth({"code_or_count:age": "amount", "code_or_count:salt_g": "amount",
                   "code_or_count:sbp": "amount"}, fixture="the text sbp table")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "sbp"})
        drive.artifact("findings")
        # The task question's refusal offers the read-numbers repair, not only another outcome.
        error = _error(drive.post({"kind": "set_task", "column": "sbp", "task": "regression"}),
                       "task_mismatch")
        read = [e for e in error["exits"] if (e["decision"] or {}).get("kind") == "apply_repair"]
        assert [e["decision"]["finding_id"] for e in read] == ["text_numbers__sbp"]
        assert read[0]["decision"]["option"] == "read_numbers"
        drive.decide(read[0]["decision"])
        drive.artifact("target_info")
        drive.answer("task", {"kind": "set_task", "column": "sbp", "task": "regression"})
        drive.decide({"kind": "set_purpose", "purpose": "prediction"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles({"pid": "identifier", "age": "covariate", "salt_g": "exposure"})
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.reach("split")
        findings = drive.artifact("findings")["findings"]
        impossible = [f for f in findings if f["id"].startswith("pack::clinical::impossible")
                      and "sbp" in f["affected_columns"]]
        assert len(impossible) == 1, [f["id"] for f in findings]
        # The lab pack's "no control yet" duplicate is gone beside the app's own lever.
        assert not [f for f in findings if f["id"].startswith("pack::clinical::text_numeric")
                    and f["affected_columns"] == ["sbp"]]
        split = {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}
        error = _error(drive.post(split), "settle_first")
        assert "`sbp`" in error["message"]
        blank = next(e["decision"] for e in error["exits"]
                     if (e["decision"] or {}).get("option") == "set_missing")
        drive.decide(blank)
        drive.decide(split)
        drive.artifact("split")
        store = client.app.state.service.store(drive.pid)
        working = pd.to_numeric(store.materialize(["sbp"], None)["sbp"], errors="coerce")
    expected = parsed.copy()
    expected.iloc[IMPOSSIBLE_ROWS] = np.nan
    assert not working.isin(IMPOSSIBLE_VALUES).any()
    assert int(working.notna().sum()) == int(expected.notna().sum())
    assert np.allclose(np.sort(working.dropna().to_numpy()), np.sort(expected.dropna().to_numpy()))


# ═════════════════════════════════════════════════════════════════════════════
# 2 · a rule on a time to event's follow-up time is a rule on the outcome
# ═════════════════════════════════════════════════════════════════════════════


def _staggered(n: int = 2400) -> pd.DataFrame:
    """Staggered entry over 12 years, closing at year 13; fiber rises with the entry year and has
    no effect on the hazard (the verifier's p12 design, own seed)."""
    rng = np.random.default_rng(2121)
    entry = rng.uniform(0, 12, n)
    fiber = (14 + 0.8 * entry + rng.normal(0, 4, n)).clip(1)
    age = rng.normal(56, 9, n)
    t_event = rng.exponential(1 / (0.035 * np.exp(0.045 * (age - 56))))
    closes = 13 - entry
    return pd.DataFrame({"participant_id": [f"T{i:05d}" for i in range(n)], "age": age.round(1),
                         "fiber_g": fiber.round(2),
                         "followup_years": np.minimum(t_event, closes).round(3),
                         "chd": (t_event <= closes).astype(int)})


def _cox(frame: pd.DataFrame, entry: str | None = None) -> Any:
    from lifelines import CoxPHFitter

    columns = ["fiber_g", "age", "followup_years", "chd"] + ([entry] if entry else [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CoxPHFitter().fit(frame[columns], "followup_years", "chd", entry_col=entry,
                                 fit_options={"precision": 1e-12, "max_steps": 200})


def test_2_a_rule_on_the_follow_up_time_is_refused_with_the_landmark_and_horizon(tmp_path):
    """The verifier's p12: `followup_years >= 3` was accepted as eligibility ("exclude early
    events") and its preview drew the follow-up's distribution. Reference: lifelines' Cox on the
    same rows: at the landmark, the rows followed past 3 years entering at 3 (delayed entry); at the
    horizon too, every event after year 10 censored at 10."""
    frame = _staggered()
    path = _csv(frame, tmp_path, "tte_rule")
    truth = Truth({"code_or_count:age": "amount", "adjust:age": "no,yes,no"},
                  fixture="the staggered cohort")
    roles = {"participant_id": "identifier", "fiber_g": "exposure", "age": "covariate",
             "followup_years": "time"}
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "chd"})
        drive.answer("event", {"kind": "set_event", "column": "chd", "level": "1"})
        drive.decide({"kind": "set_task", "column": "chd", "task": "time_to_event"})
        drive.decide({"kind": "set_follow_up", "column": "chd", "time_column": "followup_years"})
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles(roles)
        drive.reach("exclusions")
        rule = {"kind": "set_exclusions", "rules": [
            {"column": "followup_years", "low": 3, "reason": "exclude early events"}]}
        error = _error(drive.post(rule), "rule_on_outcome")
        # Its preview is refused too: no picture of the follow-up's distribution with cuts on it.
        _error(client.post(f"/api/projects/{drive.pid}/preview", json=rule), "rule_on_outcome")
        assert "follow-up time of `chd`" in error["message"]
        assert error["exits"][0]["decision"] == {"kind": "set_exclusions", "rules": []}
        landmark = next(e["decision"] for e in error["exits"]
                        if (e["decision"] or {}).get("kind") == "set_follow_up")
        assert landmark["landmark"] == 3 and landmark["horizon"] is None
        drive.decide(landmark)
        assert "a landmark" in _records(drive)[-1]["sentence"]
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.answer_plan("fiber_g")
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["cox"]})
        at_landmark = _coef(drive.artifact("fit"), "cox", "fiber_g")
        steps = {s["key"]: s for s in drive.artifact("cohort")["steps"]}
        # A ceiling on follow-up is the horizon, never rows dropped by how long they were followed.
        error = _error(drive.post({"kind": "set_exclusions", "rules": [
            {"column": "followup_years", "high": 10, "reason": "the first decade"}]}),
            "rule_on_outcome")
        horizon = next(e["decision"] for e in error["exits"]
                       if (e["decision"] or {}).get("kind") == "set_follow_up")
        assert horizon["horizon"] == 10 and horizon["landmark"] == 3
        drive.decide(horizon)
        at_horizon = _coef(drive.artifact("fit"), "cox", "fiber_g")
    followed = frame[frame["followup_years"] > 3].assign(entry=3.0)
    assert steps["landmark"]["dropped"] == int((frame["followup_years"] <= 3).sum())
    assert steps["landmark"]["n"] == len(followed)
    reference = _cox(followed, entry="entry")
    assert at_landmark["estimate"] == pytest.approx(reference.params_["fiber_g"], abs=1e-6)
    censored = followed.assign(chd=np.where(followed["followup_years"] > 10, 0, followed["chd"]),
                               followup_years=followed["followup_years"].clip(upper=10))
    reference = _cox(censored, entry="entry")
    assert at_horizon["estimate"] == pytest.approx(reference.params_["fiber_g"], abs=1e-6)
    # Under prediction the rule is refused as well (no one's follow-up is known in advance).
    state = ProjectState(target="chd", task="time_to_event", purpose="prediction",
                         follow_up={"time_column": "followup_years"})
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_exclusions", "rules": [
            {"column": "followup_years", "low": 3, "reason": "early events"}]},
            {"state": state, "target": "chd"})
    assert refused.value.code == "rule_on_outcome"


# ═════════════════════════════════════════════════════════════════════════════
# 3 · coefficients seen under prediction, then the purpose made inference
# ═════════════════════════════════════════════════════════════════════════════


def _sodium(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(9393)
    age = rng.normal(48, 11, n).round(0)
    sodium = rng.normal(3.3, 0.9, n).round(2)
    potassium = (2.8 + 0.25 * (sodium - 3.3) + rng.normal(0, 0.6, n)).round(2)
    sbp = (118 + 0.45 * (age - 48) + 2.2 * sodium - 1.5 * potassium
           + rng.normal(0, 9, n)).round(1)
    return pd.DataFrame({"pid": [f"Z{i:04d}" for i in range(n)], "age": age, "sodium_g": sodium,
                         "potassium_g": potassium, "sbp": sbp})


SODIUM_TRUTH = {"code_or_count:age": "amount", "adjust:age": "no,yes,no",
                "adjust:potassium_g": "unknown,yes,no", "exposure:sbp": "sodium_g"}
SODIUM_ROLES = {"pid": "identifier", "age": "covariate", "sodium_g": "exposure",
                "potassium_g": "covariate"}


def _to_models(drive: Any, purpose: str) -> None:
    drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "sbp"})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": purpose})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    drive.reach("roles")
    drive.decide_roles(SODIUM_ROLES)


def test_3_estimates_seen_under_prediction_lock_the_plan_at_the_switch(tmp_path):
    """The verifier's p02 and p14. Reference: least squares on the fixture's columns (NumPy) for
    the coefficient shown; the record's own flags for what was seen when."""
    from turbotab.core import plan_lock

    assert GELMAN_LOKEN in _flat(inspect.getsource(plan_lock))
    frame = _sodium()
    path = _csv(frame, tmp_path, "purpose_switch")
    X = np.column_stack([np.ones(len(frame)), frame[["age", "sodium_g", "potassium_g"]]])
    beta = np.linalg.lstsq(X, frame["sbp"].to_numpy(float), rcond=None)[0]
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, Truth(SODIUM_TRUTH, fixture="the sodium table"))
        _to_models(drive, "prediction")
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["linear"]})
        shown = _coef(_served(drive, "fit"), "linear", "sodium_g")
        assert shown["estimate"] == pytest.approx(beta[2], abs=1e-8)
        assert "lock_plan" not in [r["decision"]["kind"] for r in _records(drive)]
        before = len(_records(drive))
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        records = _records(drive)
        switch, lock = records[before], records[before + 1]
        assert switch["decision"] == {"kind": "set_purpose", "purpose": "inference"}
        assert lock["decision"]["kind"] == "lock_plan" and lock["decision"]["seen"] == "prediction"
        assert lock["decision"]["seen_target"] == "sbp"
        assert "had been displayed under prediction" in lock["sentence"]
        assert "after estimates were seen, not before" in lock["sentence"]
        assert "before any estimate was displayed" not in lock["sentence"]
        drive.answer_plan("sodium_g")
        later = _records(drive)[before + 2:]
        assert later and all(r["after_estimates"] for r in later)
        assert all(r["sentence"].startswith("After the estimates were seen") for r in later)
        fit = _served(drive, "fit")
        assert _coef(fit, "linear", "sodium_g")["estimate"] == pytest.approx(beta[2], abs=1e-8)
        assert [r["decision"]["kind"] for r in _records(drive)].count("lock_plan") == 1
        text = client.get(f"/api/projects/{drive.pid}/methods").json()["text"]
        assert "declared for `prediction`" in text and "had been displayed under prediction" in text

    # A lock posted by hand, under inference before the exclusions (p14), is refused: the plan
    # locks itself when an estimate is first displayed, so no earlier answer is marked as made
    # after one; the lock it records then says nothing was displayed before it, truly.
    with local_server(tmp_path / "home2") as client:
        drive = open_project(client, path, Truth(SODIUM_TRUTH, fixture="the sodium table"))
        _to_models(drive, "inference")
        drive.reach("exclusions")
        _error(drive.post({"kind": "lock_plan"}), "plan_locks_itself")
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.answer_plan("sodium_g")
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["linear"]})
        assert not any(r["after_estimates"] for r in _records(drive))
        _served(drive, "fit")
        locks = [r for r in _records(drive) if r["decision"]["kind"] == "lock_plan"]
        assert len(locks) == 1 and locks[0]["decision"]["seen"] == "inference"
        assert "before any estimate was displayed" in locks[0]["sentence"]
        _error(drive.post({"kind": "lock_plan"}), "plan_already_locked")


# ═════════════════════════════════════════════════════════════════════════════
# 4 · NHANES linked mortality: MORTSTAT beside PERMTH_INT
# ═════════════════════════════════════════════════════════════════════════════


def _linked_mortality(n: int = 2500) -> pd.DataFrame:
    """Staggered enrolment over 13 years, linkage closing at year 14; fiber rises with the
    enrolment year and has no effect on the hazard (the verifier's p01 design, own seed)."""
    rng = np.random.default_rng(4343)
    entry = rng.uniform(0, 13, n)
    fiber = (12 + 0.9 * entry + rng.normal(0, 4, n)).clip(0.5)
    age = rng.normal(52, 10, n)
    t_event = rng.exponential(1 / (0.025 * np.exp(0.05 * (age - 52))))
    t = np.minimum(t_event, 14 - entry)
    return pd.DataFrame({"SEQN": np.arange(30001, 30001 + n), "RIDAGEYR": age.round(0),
                         "DR1TFIBE": fiber.round(1), "PERMTH_INT": np.ceil(12 * t).astype(int),
                         "MORTSTAT": (t_event <= 14 - entry).astype(int)})


FOLLOW_UP_NAMES = ("time_in_study", "years_followed", "person_years", "duration", "obs_years",
                   "observation_time", "years_to_cvd", "tstop", "time_at_risk", "pyrs",
                   "length_of_follow_up", "PERMTH_INT", "PERMTH_EXM")


def test_4_linked_mortality_asks_the_follow_up_and_fits_cox(tmp_path):
    """The verifier's p01: the follow-up was skipped by a name test and the logistic log-odds
    served. Reference: lifelines' Cox (Efron ties) on the file's own columns."""
    from lifelines import CoxPHFitter

    assert all(estimand.reads_as_follow_up(n) for n in FOLLOW_UP_NAMES)
    # Whether it is asked never rests on a name under the clinical or dietary lens (or none):
    # a yes/no outcome there is asked even beside no column a name reads; under an assay lens it
    # is a status at sampling, asked only beside one.
    no_name = {"column": "y", "task": "binary", "confidence": "high", "follow_up": []}
    for lens, asked in ((["clinical"], True), (["dietary"], True), ([], True),
                        (["metabolomics"], False), (["survey"], False)):
        gate = estimand.follow_up_gate(ProjectState(lens=lens, target="y"), no_name)
        assert (gate is None) is asked, (lens, gate)

    frame = _linked_mortality()
    path = _csv(frame, tmp_path, "nhanes_mortality")
    truth = Truth({"code_or_count:RIDAGEYR": "amount", "code_or_count:PERMTH_INT": "amount",
                   "adjust:RIDAGEYR": "no,yes,no"}, fixture="the linked-mortality table")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "MORTSTAT"})
        info = drive.artifact("target_info")
        assert [c["column"] for c in info["follow_up"]] == ["PERMTH_INT"]
        assert info["follow_up_options"][0] == "PERMTH_INT"
        drive.answer("event", {"kind": "set_event", "column": "MORTSTAT", "level": "1"})
        assert drive.reach("follow_up")["status"] == "open"
        error = _error(drive.post({"kind": "set_censoring", "column": "MORTSTAT"}),
                       "follow_up_varies")
        assert error["exits"][0]["decision"] == {"kind": "set_task", "column": "MORTSTAT",
                                                 "task": "time_to_event"}
        drive.decide(error["exits"][0]["decision"])
        drive.decide({"kind": "set_follow_up", "column": "MORTSTAT", "time_column": "PERMTH_INT"})
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        proposed = {c["column"]: c for c in drive.artifact("roles")["columns"]}
        assert proposed["PERMTH_INT"]["proposed"] == "time"
        assert proposed["PERMTH_INT"]["confidence"] == "high"
        wrong = {"SEQN": "identifier", "RIDAGEYR": "covariate", "DR1TFIBE": "exposure",
                 "PERMTH_INT": "exposure"}
        error = _error(drive.post({"kind": "set_roles", "roles": wrong}), "follow_up_as_predictor")
        assert error["exits"][0]["decision"]["roles"]["PERMTH_INT"] == "time"
        drive.decide_roles({**wrong, "PERMTH_INT": "time"})
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.answer_plan("DR1TFIBE")
        drive.reach("models")
        assert [f["key"] for f in drive.artifact("shelf")["families"]] == ["cox"]
        drive.decide({"kind": "select_models", "models": ["cox"]})
        fit = drive.artifact("fit")
    assert fit["task"] == "time_to_event"
    fiber = _coef(fit, "cox", "DR1TFIBE")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = CoxPHFitter().fit(frame[["DR1TFIBE", "RIDAGEYR", "PERMTH_INT", "MORTSTAT"]],
                                      "PERMTH_INT", "MORTSTAT",
                                      fit_options={"precision": 1e-12, "max_steps": 200})
    assert fiber["estimate"] == pytest.approx(reference.params_["DR1TFIBE"], abs=1e-6)
    assert "hazard ratio" in fit["estimand"]["caption"]


# ═════════════════════════════════════════════════════════════════════════════
# 5 · a direct effect asks for its mediator–outcome confounders and interactions
# ═════════════════════════════════════════════════════════════════════════════


def _mediation(n: int = 1500) -> pd.DataFrame:
    """Sugar → bmi → glucose, with age confounding sugar and glucose, and sleep a common cause of
    bmi and glucose that does not cause sugar: the controlled direct effect of sugar (bmi held
    fixed) is 0.2, identified only when sleep is adjusted too (bmi is a collider of sugar and
    sleep)."""
    rng = np.random.default_rng(5353)
    age = rng.normal(50, 10, n)
    sleep = rng.normal(7, 1, n)
    sugar = 40 + 0.3 * (age - 50) + rng.normal(0, 8, n)
    bmi = 22 + 0.15 * (sugar - 40) - 2.5 * (sleep - 7) + rng.normal(0, 2, n)
    glucose = (90 + 0.2 * (sugar - 40) + 0.8 * (bmi - 22) + 0.3 * (age - 50)
               - 4.0 * (sleep - 7) + rng.normal(0, 5, n))
    return pd.DataFrame({"pid": [f"M{i:05d}" for i in range(n)], "age": age.round(2),
                         "sleep": sleep.round(3), "sugar": sugar.round(3), "bmi": bmi.round(3),
                         "glucose": glucose.round(3)})


def test_5_a_direct_effect_asks_its_mediator_confounders_and_the_interaction(tmp_path):
    """The verifier's WP17-3 variant: a direct effect asked the same five questions as a total
    effect, named no mediator–outcome confounder and no interaction. Reference: least squares
    (NumPy) of glucose on sugar, bmi, age and sleep: the identifying set by Valeri & VanderWeele's
    two assumptions; leaving sleep out (the old answer for a "cause of neither") is biased."""
    assert VALERI_QUOTE in _flat(inspect.getsource(estimand._a_direct_effect_asks_its_questions))
    frame = _mediation()
    path = _csv(frame, tmp_path, "mediation")

    def ols(columns: list[str]) -> float:
        X = np.column_stack([np.ones(len(frame)), frame[columns]])
        return float(np.linalg.lstsq(X, frame["glucose"].to_numpy(float), rcond=None)[0][1])

    identified = ols(["sugar", "bmi", "age", "sleep"])
    assert abs(identified - 0.2) < 0.05 and abs(ols(["sugar", "bmi", "age"]) - identified) > 0.05
    truth = Truth({"code_or_count:age": "amount", "exposure:glucose": "sugar",
                   "adjust:age": "yes,yes,no",
                   "adjust:bmi": "no,yes,yes,interacts=no",
                   "adjust:sleep": "no,no,no,confounds_mediator=yes"},
                  fixture="the mediation table")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "glucose"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles({"pid": "identifier", "sugar": "exposure", "age": "covariate",
                            "bmi": "covariate", "sleep": "covariate"})
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.reach("estimand")
        drive.decide({"kind": "set_estimand", "exposure": "sugar", "effect": "direct",
                      "measure": "mean_difference"})
        drive.reach("adjustment")
        plain = {"age": {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"},
                 "bmi": {"causes_exposure": "no", "causes_outcome": "yes", "after_exposure": "yes"},
                 "sleep": {"causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"}}
        drive.decide({"kind": "set_adjustment", "exposure": "sugar", "answers": plain})
        # The five questions are not enough: the card asks each the direct effect's own question.
        card = _until(lambda: drive.artifact("proposals")["adjustment"],
                      lambda c: bool(c and c.get("direct_questions")))
        assert {q["column"]: q["fields"] for q in card["direct_questions"]} == {
            "bmi": ["interacts"], "sleep": ["confounds_mediator"]}
        assert set(card["questions"]) >= {"confounds_mediator", "interacts"}
        assert drive.reach("adjustment")["status"] == "open"
        # A possible exposure–mediator interaction is block and record, with its ways forward.
        maybe = {**plain["bmi"], "interacts": "unknown"}
        error = _error(drive.post({"kind": "set_adjustment", "exposure": "sugar",
                                   "answers": {"bmi": maybe}}), "exposure_mediator_interaction")
        exits = [e["decision"] for e in error["exits"]]
        assert exits[0]["kind"] == "set_estimand" and exits[0]["effect"] == "total"
        assert exits[1]["answers"]["bmi"]["interacts"] == "no"
        assert exits[2]["answers"]["bmi"]["interaction_attested"] is True
        drive.decide({"kind": "set_adjustment", "exposure": "sugar", "answers": {
            "bmi": {**plain["bmi"], "interacts": "no"},
            "sleep": {**plain["sleep"], "confounds_mediator": "yes"}}})
        assert drive.reach("adjustment")["status"] == "answered"
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["linear"]})
        fit = drive.artifact("fit")
    caption = fit["estimand"]["caption"]
    assert "The controlled direct effect of `sugar` on `glucose`" in caption
    assert "the mediator `bmi` held fixed" in caption and "conditional on `age` and `sleep`" in caption
    assert "no exposure–mediator interaction" in caption
    assert "no unmeasured common cause of a mediator and the outcome" in caption
    assert _coef(fit, "linear", "sugar")["estimate"] == pytest.approx(identified, abs=1e-8)
    # With no mediator among the answers, a direct effect is refused with the total effect.
    state = d.fold([d.DecisionRecord(id=f"r{i}", seq=i, at="2026-10-05T00:00:00Z", decision=x)
                    for i, x in enumerate([
                        d.SetTarget(column="glucose"), d.SetPurpose(purpose="inference"),
                        d.SetRoles(roles={"sugar": "exposure", "age": "covariate"}),
                        d.SetEstimand(exposure="sugar", effect="direct",
                                      measure="mean_difference")], start=1)])
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_adjustment", "exposure": "sugar", "answers": {"age": plain["age"]}},
                   {"state": state})
    assert refused.value.code == "no_mediator"
    assert refused.value.exits[0]["decision"]["effect"] == "total"


# ═════════════════════════════════════════════════════════════════════════════
# 6 · under inference the roles the estimand reads are asked at the estimand
# ═════════════════════════════════════════════════════════════════════════════


def _rode_along(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(6363)
    f = pd.DataFrame({"participant_id": [f"S{i:04d}" for i in range(n)]})
    f["age"] = rng.normal(50, 12, n).round(1)
    f["alcohol_flag"] = rng.choice([0, 1], n, p=[0.35, 0.65])
    f["alcohol"] = np.where(f["alcohol_flag"] == 1, rng.gamma(2.0, 8.0, n).round(1), np.nan)
    f["fiber_g"] = rng.gamma(4, 5, n).round(1)
    f["sbp"] = (120 + 0.3 * (f["age"] - 50) + 4.0 * f["alcohol_flag"]
                + 0.25 * np.nan_to_num(f["alcohol"]) - 0.2 * f["fiber_g"]
                + rng.normal(0, 8, n)).round(1)
    return f


def test_6_the_estimand_question_asks_the_roles_that_rode_along(tmp_path):
    """The verifier's p16: the roles answered in bulk, the exposure question opened with no ask
    card and offered only `age`. Reference: the roles the server itself recorded unconfirmed."""
    path = _csv(_rode_along(), tmp_path, "estimand_card")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, Truth({"code_or_count:age": "amount"},
                                                 fixture="the rode-along table"))
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "sbp"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        roles = {c["column"]: c["proposed"] for c in drive.artifact("roles")["columns"]}
        r = drive.post({"kind": "set_roles", "roles": roles})  # the bulk tap, "use these"
        assert r.status_code == 200, r.text
        waiting = r.json()["decisions"][-1]["decision"]["unconfirmed"]
        assert "fiber_g" in waiting
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        step = drive.reach("estimand")
        assert step["status"] == "open" and step["ask"] is not None
        card = step["ask"]
        assert card["consumer"] == "the exposure and its effect"
        assert {c for g in card["groups"] for c in g["columns"]} == set(waiting)
        confirms = {(e["decision"]["column"], e["decision"]["value"])
                    for e in card["exits"] if (e["decision"] or {}).get("kind") == "confirm_reading"}
        assert {(c, roles[c]) for c in waiting} <= confirms
        offered = drive.artifact("proposals")["estimand"]
        assert "fiber_g" in {u["column"] for u in offered["unconfirmed"]}
        assert "fiber_g" not in {e["column"] for e in offered["exposures"]}
        # Confirmed one by one, as their author reads them, the exposure is offered and chosen.
        drive.decide({"kind": "confirm_readings", "items": [
            {"reading": "role", "column": c, "value": roles[c]} for c in waiting]})
        _until(lambda: drive.artifact("proposals")["estimand"],
               lambda o: "fiber_g" in {e["column"] for e in o["exposures"]})
        assert drive.reach("estimand")["ask"] is None
        assert drive.reach("estimand")["status"] == "open"


# ═════════════════════════════════════════════════════════════════════════════
# 7 · the energy model's kcal per unit is the settled unit's, never the name's
# ═════════════════════════════════════════════════════════════════════════════


def _drinks(n: int = 900) -> pd.DataFrame:
    """Alcohol recorded in US standard drinks (14 g, 98 kcal) but named `alcohol_g`; energy built
    from 4P + 4C + 9F + 98 drinks (the verifier's p08 design, own seed)."""
    rng = np.random.default_rng(8181)
    f = pd.DataFrame({"participant_id": [f"A{i:04d}" for i in range(n)]})
    f["age"] = rng.normal(50, 12, n).round(1)
    P = rng.normal(80, 20, n).clip(20)
    C = rng.normal(250, 60, n).clip(50)
    F = rng.normal(75, 20, n).clip(15)
    drinks = np.where(rng.random(n) < 0.45, 0, rng.gamma(1.5, 1.0, n)).round(1)
    f["protein_g"], f["fat_g"], f["carbohydrate_g"], f["alcohol_g"] = (
        P.round(1), F.round(1), C.round(1), drinks)
    f["energy_kcal"] = ((4 * P + 4 * C + 9 * F + 98 * drinks) * rng.normal(1, 0.03, n)).round(0)
    f["sbp"] = (118 + 0.4 * (f["age"] - 50) + 1.5 * drinks + 0.004 * f["energy_kcal"]
                + rng.normal(0, 9, n)).round(1)
    return f


def _relative_per_unit(f: pd.DataFrame, kcal_per_drink: float) -> float:
    """Tomova et al.'s average relative effect of alcohol, per drink, by least squares on the kcal
    terms written out: θ = β_A − Σ_k w_k β_k, w_k each other source's share of the remaining
    energy, times the kcal one drink carries."""
    kc = {"P": 4 * f["protein_g"], "F": 9 * f["fat_g"], "C": 4 * f["carbohydrate_g"],
          "A": kcal_per_drink * f["alcohol_g"]}
    kc["O"] = f["energy_kcal"] - sum(kc.values())
    X = np.column_stack([np.ones(len(f)), f["age"], *kc.values()])
    beta = dict(zip(kc, np.linalg.lstsq(X, f["sbp"].to_numpy(float), rcond=None)[0][2:]))
    others = [k for k in kc if k != "A"]
    total = sum(float(kc[k].mean()) for k in others)
    return (beta["A"] - sum(float(kc[k].mean()) / total * beta[k] for k in others)) * kcal_per_drink


def test_7_the_all_components_model_reads_the_settled_unit_not_the_name(tmp_path):
    """The verifier's p08 and gate6r q2. Reference: NumPy's relative effect at 98 kcal per drink
    (the truth) against 7 kcal per gram (the name's): the served one is the first."""
    frame = _drinks()
    at_drink, at_gram = _relative_per_unit(frame, 98.0), _relative_per_unit(frame, 7.0)
    assert abs(at_drink - at_gram) > 0.3  # the unit decides the number
    path = _csv(frame, tmp_path, "alcohol_named_g")
    truth = Truth({"code_or_count:age": "amount", "unit:energy_kcal": "kcal",
                   "day_count:energy_kcal": "1", "code_or_count:energy_kcal": "amount",
                   "unit:alcohol_g": "drinks:14", "exposure:sbp": "alcohol_g",
                   "contrast:alcohol_g": "addition", "adjust:age": "no,yes,no",
                   **{f"adjust:{c}": "no,yes,no" for c in ("protein_g", "fat_g", "carbohydrate_g")}},
                  fixture="the drinks table")
    nutrients = ["protein_g", "fat_g", "carbohydrate_g", "alcohol_g"]
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "sbp"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles({"participant_id": "identifier", "age": "covariate",
                            **{c: "exposure" for c in nutrients}, "energy_kcal": "energy"})
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.answer_plan("alcohol_g")
        drive.reach("energy_adjustment")
        body = {"kind": "set_energy_adjustment", "method": "all_components",
                "energy_column": "energy_kcal", "nutrients": nutrients}
        error = _error(drive.post(body), "reading_unsettled")
        assert "_g suffix" not in error["message"] and "`alcohol_g`" in error["message"]
        units = {e["decision"]["value"] for e in error["exits"]
                 if (e["decision"] or {}).get("kind") == "confirm_reading"
                 and e["decision"]["column"] == "alcohol_g"}
        assert {"g", "drinks:14"} <= units
        assert not any((e["decision"] or {}).get("column") in ("protein_g", "fat_g",
                                                               "carbohydrate_g")
                       for e in error["exits"])  # read in grams by the Atwater test
        drive.decide(body)  # the truth answers the unit it asks: drinks of 14 g
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["linear"]})
        fit = drive.artifact("fit")
        design = drive.artifact("design")
    row = _coef(fit, "linear", "alcohol_g_relative")
    assert row["estimate"] == pytest.approx(at_drink, abs=1e-6)
    assert row["meaning"].endswith("per standard drink of 14 g")
    formula = next(n for n in design["lineage"]["nodes"] if n.get("column") == "kcal_from_alcohol_g")
    assert "98 × alcohol_g" in _flat(formula)

    # Gate 6's q2: `alcohol` with no suffix, its unit recorded in drinks, is no longer refused for
    # its name; the macronutrients' grams recorded, nothing else is asked.
    roles = {"energy_kcal": "energy", "protein_g": "exposure", "fat_g": "exposure",
             "carbohydrate_g": "exposure", "alcohol": "exposure"}
    records = [d.SetTarget(column="sbp"), d.SetRoles(roles=roles),
               d.ConfirmReadings(items=[d.ReadingItem(reading="unit", column=c, value=v) for c, v in
                                        (("protein_g", "g"), ("fat_g", "g"),
                                         ("carbohydrate_g", "g"), ("alcohol", "drinks:14"))])]
    state = d.fold([d.DecisionRecord(id=f"r{i}", seq=i, at="2026-10-05T00:00:00Z", decision=x)
                    for i, x in enumerate(records, start=1)])
    accepted = d.validate({"kind": "set_energy_adjustment", "method": "all_components",
                           "energy_column": "energy_kcal",
                           "nutrients": ["protein_g", "fat_g", "carbohydrate_g", "alcohol"]},
                          {"state": state, "target": "sbp", "columns": [*roles, "sbp"]})
    assert accepted.method == "all_components"
    # The contrast's exits are whole answers: an answer naming no energy column ("none") takes the
    # settled ones, so taking an exit never lands on the energy-role refusal.
    state = state.model_copy(update={"purpose": "inference", "estimand": d.EstimandSpec(
        exposure="alcohol", contrast="substitution", measure="mean_difference")})
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_energy_adjustment", "method": "none"},
                   {"state": state, "target": "sbp", "columns": [*roles, "sbp"]})
    assert refused.value.code == "contrast_mismatch"
    for e in refused.value.exits:
        decision = e["decision"]
        if decision and decision["kind"] == "set_energy_adjustment":
            assert decision["energy_column"] == "energy_kcal"
            assert set(decision["nutrients"]) == {"protein_g", "fat_g", "carbohydrate_g", "alcohol"}


# ═════════════════════════════════════════════════════════════════════════════
# 8 · the cluster reading and the grouping question agree, each honored
# ═════════════════════════════════════════════════════════════════════════════


def _households() -> pd.DataFrame:
    rng = np.random.default_rng(1818)
    sizes = rng.integers(1, 5, 240)
    hh = np.repeat(np.arange(len(sizes)), sizes)
    n = len(hh)
    f = pd.DataFrame({"participant_id": [f"H{i:05d}" for i in range(n)],
                      "hhid": [f"HH{h:04d}" for h in hh]})
    f["x"] = (rng.normal(0, 1, len(sizes))[hh] + rng.normal(0, 0.5, n)).round(3)
    f["age"] = rng.normal(45, 12, n).round(1)
    f["y"] = (2 + 0.5 * f["x"] + 0.02 * f["age"] + rng.normal(0, 1.5, len(sizes))[hh]
              + rng.normal(0, 0.7, n)).round(3)
    return f


def _intervals(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """x's 95% interval by HC3 on t(n − p) and by CR2 by household on Bell–McCaffrey df, each
    written out by definition (``references``)."""
    X = np.column_stack([np.ones(len(frame)), frame["x"], frame["age"]])
    y = frame["y"].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    hc3 = ref.hc3_by_definition(X, e)
    q = stats.t.ppf(0.975, len(frame) - X.shape[1])
    out = {"HC3": (beta[1] - q * math.sqrt(hc3[1, 1]), beta[1] + q * math.sqrt(hc3[1, 1]))}
    V, df = ref.cr2_by_definition(X, e, pd.factorize(frame["hhid"])[0])
    q = stats.t.ppf(0.975, df[1])
    out["CR2"] = (beta[1] - q * math.sqrt(V[1, 1]), beta[1] + q * math.sqrt(V[1, 1]))
    return out


def test_8_every_cluster_answer_is_honored_by_the_fit(tmp_path):
    """Gate 6's q7 re-run and the verifier's p19/p19b: after the grouping question named the
    household, a "no" confirmation was ignored; after "they group nothing", the fit still clustered
    by it. Reference: HC3 and CR2 written out by definition on the file's own columns."""
    frame = _households()
    expected = _intervals(frame)
    path = _csv(frame, tmp_path, "households")
    truth = Truth({"code_or_count:age": "amount", "adjust:age": "no,yes,no", "exposure:y": "x"},
                  fixture="the household table")
    roles = {"participant_id": "identifier", "hhid": "cluster", "x": "exposure", "age": "covariate"}

    def to_clusters(drive: Any) -> None:
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "y"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles(roles)
        assert drive.reach("clusters")["status"] == "open"

    def to_fit(drive: Any) -> None:
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.answer_plan("x")
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["linear"]})

    def served(drive: Any) -> tuple[str, tuple[float, float]]:
        fit = _served(drive, "fit")
        row = _coef(fit, "linear", "x")
        return fit["models"][0]["inference"]["covariance"], (row["ci_low"], row["ci_high"])

    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        to_clusters(drive)
        drive.decide({"kind": "set_clusters", "column": "hhid", "adjust": "cluster_only"})
        to_fit(drive)
        seen = {"first": served(drive)}
        for value in ("no", "yes"):
            drive.decide({"kind": "confirm_reading", "reading": "cluster", "column": "hhid",
                          "value": value})
            seen[value] = served(drive)
    for answer, kind in (("first", "CR2"), ("no", "HC3"), ("yes", "CR2")):
        covariance, (low, high) = seen[answer]
        assert covariance == kind, (answer, covariance)
        assert (low, high) == pytest.approx(expected[kind], abs=1e-6), answer

    with local_server(tmp_path / "home2") as client:
        drive = open_project(client, path, truth)
        to_clusters(drive)
        error = _error(drive.post({"kind": "set_clusters", "column": None}), "grouping_reads")
        nothing = next(e["decision"] for e in error["exits"]
                       if e["label"] == "They group nothing; record that")
        drive.decide(nothing)
        record = _records(drive)[-1]
        assert record["decision"]["none_of"] == ["hhid"]
        assert "`hhid`" in record["sentence"] and "do not cluster by it" in record["sentence"]
        assert drive.view()["state"]["reading_confirmations"]["cluster:hhid"] == "no"
        to_fit(drive)
        covariance, interval = served(drive)
    assert covariance == "HC3" and interval == pytest.approx(expected["HC3"], abs=1e-6)
