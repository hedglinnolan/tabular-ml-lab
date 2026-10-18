"""Corpus records built by hand for the schema's and the checks' tests: a Predict run and an
Estimate run, each constructed to pass every check, so a test changes one thing and names what
must fire. Every number here is constructed, none is an engine's output."""
from __future__ import annotations

import copy
from typing import Any

T0 = "2026-10-18T10:00:00+00:00"


def at(minute: int) -> str:
    return f"2026-10-18T10:{minute:02d}:00+00:00"


def movement(family: str, against: str, value: float) -> dict[str, Any]:
    return {"family": family, "against": against, "why": None,
            "movement": {"instrument": "paired_folds", "regime": "realized", "value": value,
                         "band": 1, "calibrated": True, "changes_question": False,
                         "not_measurable": False, "crosses": False,
                         "words": f"For `{family}`, what differs moved the score by {value}.",
                         "label": f"|mean Δscore| / SD of Δ by fold = {value}"}}


def decision(seq: int, kind: str, option: dict[str, Any], answers: list[tuple[str, Any]], *,
             note: str | None = "a reason", source: str = "analyst", post_seal: bool = False,
             after_estimates: bool = False, examined: list[dict[str, Any]] | None = None,
             materiality: dict[str, Any] | None = None, minute: int | None = None
             ) -> dict[str, Any]:
    phase = "after_seal" if post_seal else "after_estimates" if after_estimates else "before"
    return {"id": f"r{seq}", "seq": seq, "at": at(minute if minute is not None else seq),
            "kind": kind, "option": {"kind": kind, **option},
            "answers": [{"question": q, "value": v} for q, v in answers], "reason": note,
            "source": source, "phase": phase, "post_seal": post_seal,
            "after_estimates": after_estimates, "early": None, "examined": examined or [],
            "materiality": materiality or {"source": None, "why": "constructed: none measured"}}


def attempt(seq: int, decisions: list[int], changed: list[int], *, minute: int,
            notes: dict[int, str] | None = None, purpose: str = "prediction",
            families: list[str] | None = None, scores: dict[str, Any] | None = None,
            effects: dict[str, Any] | None = None, moved: list[dict[str, Any]] | None = None,
            movement_why: str | None = None, at_opening: dict[str, Any] | None = None,
            changed_after_seal: bool = False, plan_digest: str | None = None,
            holdout: dict[str, Any] | None = None,
            withdrawn: list[tuple[int, int | None]] | None = None,
            reinstated: list[tuple[int, int | None]] | None = None) -> dict[str, Any]:
    notes = notes or {s: f"reason for record #{s}" for s in changed}
    out = {"id": f"fit{seq}", "seq": seq, "at": at(minute), "purpose": purpose, "target": "y",
           "fit_key": f"key{seq}", "plan_digest": plan_digest or f"digest{seq}",
           "decisions": decisions, "changed": changed,
           "reasons": [{"seq": s, "note": n} for s, n in notes.items()],
           "press": None, "families": families or ["linear"], "seconds": 1.5, "workers": 2,
           "primary_metric": "auc" if purpose == "prediction" else None, "folds": "folds",
           "seed": 0, "held_out": "rows", "scores": scores or {}, "holdout": holdout,
           "holdout_why": None if holdout is not None else "constructed: none served",
           "changed_after_seal": changed_after_seal,
           "at_opening": at_opening, "effects": effects or {}, "movement": moved or [],
           "movement_why": movement_why, "withheld": None}
    out["withdrawn"] = [{"seq": s, "by": b} for s, b in withdrawn or ()]
    out["reinstated"] = [{"seq": s, "by": b} for s, b in reinstated or ()]
    return out


def run(run_id: str = "run-1") -> dict[str, Any]:
    return {"id": run_id, "design": 2, "engine_commit": "c0ffee", "engine_modified": False,
            "engine_version": "2.0.0.dev0", "analyst_model": "claude-haiku-5-5", "effort": "high",
            "prompt_version": "v1", "started": T0, "ended": at(59),
            "spend": {"fresh": 1000, "cached": 50000, "messages": 12}, "spend_why": None,
            "fits": 2, "fit_seconds": 3.0, "fits_unseen": 0, "fit_budget": {"count": 10},
            "help": [], "server": {"pid": 1, "port": 8765, "mode": "local", "workers": 2}}


def surroundings(control: str | None = None) -> dict[str, Any]:
    return {
        "dataset": {"id": "fixture", "control": control, "fixture": "table.csv", "known": None,
                    "files": [{"role": "table", "name": "table.csv", "sha256": "ab" * 32}],
                    "codebook": None, "card": None},
        "claims": [{"at": at(5), "id": "designs.observational_default", "register": "card",
                    "status": "built", "grade": "B", "fallback": None, "version": 1,
                    "hash": "h", "refused": None}],
        "surfacings": {"refusals": [], "concerns": []},
        "export": {"reached": True, "sha256": "cd" * 32, "methods": "# Methods", "checklist": {},
                   "refused": None},
        "replication": None,
    }


SCORES = {"linear": {"auc": 0.71, "log_loss": 0.52}, "elastic_net": {"auc": 0.70,
                                                                      "log_loss": 0.53}}
CV = {"linear": {"auc": {"estimate": 0.74, "ci_low": 0.69, "ci_high": 0.79,
                         "folds": [0.7, 0.72, 0.75, 0.76, 0.77]}}}


def predict_record(control: str | None = None) -> dict[str, Any]:
    """Two fits under Predict: the first with the linear model, the second after the analyst
    added the elastic net (record #4, which replaced record #2; examined by the first fit); then
    the opening of the seal on the second, which serves the opening's held-out scores. Its result
    is the opening's scores, read by records #1, #3 and #4."""
    decisions = [
        decision(1, "set_target", {"column": "y"}, [("set_target", {"column": "y"})]),
        decision(2, "select_models", {"models": ["linear"]},
                 [("select_models", {"models": ["linear"]})]),
        decision(3, "confirm_reading", {"reading": "code_or_count", "column": "age",
                                        "value": "amount"}, [("code_or_count:age", "amount")]),
        decision(4, "select_models", {"models": ["linear", "elastic_net"]},
                 [("select_models", {"models": ["linear", "elastic_net"]})],
                 examined=[{"question": "select_models", "value": {"models": ["linear"]},
                            "how": "fit", "at": at(3), "seq": 2, "attempt": "fit1",
                            "alternative": True}],
                 materiality={"source": "fits", "attempt": "fit2",
                              "movement": [movement("linear", "fit1", 0.4)]}),
        decision(5, "open_seal", {"family": "linear", "target": "y", "metric": "auc",
                                  "n_holdout": 96, "scores": SCORES},
                 [("open_seal", {"family": "linear"})], minute=8),
    ]
    attempts = [
        attempt(1, [1, 2, 3], [1, 2, 3], minute=3, scores=CV,
                movement_why="No earlier fit is recorded to compare this one with."),
        attempt(2, [1, 3, 4], [4], minute=6, families=["linear", "elastic_net"], scores=CV,
                holdout=copy.deepcopy(SCORES), withdrawn=[(2, 4)],
                moved=[movement("linear", "fit1", 0.4),
                       {"family": "elastic_net", "against": "fit1", "movement": None,
                        "why": "`elastic_net` was not fitted in the previous fit."}],
                at_opening={"seq": 5, "family": "linear", "metric": "auc", "n_holdout": 96,
                            "scores": SCORES, "current": True, "note": "the reported result"}),
    ]
    result = {"purpose": "prediction", "attempt": "fit2", "plan_digest": "digest2",
              "decisions": [1, 3, 4], "opening": 5, "family": "linear", "metric": "auc",
              "n_holdout": 96, "scores": copy.deepcopy(SCORES),
              "intervals": {"auc": {"estimate": 0.71, "ci_low": 0.45, "ci_high": 0.97,
                                    "level": 0.95, "method": "DeLong"}},
              "intervals_why": None, "lock": None, "effects": None, "refused": None}
    return {"version": 1, "run": run(), **surroundings(control),
            "question": {"purpose": "prediction", "target": "y", "estimand": None,
                         "intended_use": None, "stated": None, "paper": None},
            "regime": {"n_rows": 480, "n_analyzed": 480, "task": "binary", "event": "1",
                       "events": 112, "missing": {}, "design": {}, "lenses": ["clinical"]},
            "decisions": decisions, "attempts": attempts, "result": result}


EFFECT = [{"feature": "x", "estimate": 0.30, "ci_low": 0.10, "ci_high": 0.50, "p": 0.003}]


def estimate_record(control: str | None = None) -> dict[str, Any]:
    """Under Estimate: the plan locked at record #4 (TurboTab's own), the locked fit (its plan
    the lock's) read records #1–#4, and a secondary analysis after the estimates were seen (record
    #5, a second fit)."""
    decisions = [
        decision(1, "set_purpose", {"purpose": "inference"},
                 [("set_purpose", {"purpose": "inference"})]),
        decision(2, "set_estimand", {"exposure": "x"}, [("set_estimand", {"exposure": "x"})]),
        decision(3, "set_adjustment", {"exposure": "x", "answers": {}},
                 [("set_adjustment:x", {"answers": {}})],
                 examined=[{"question": "set_adjustment:x", "value": {"answers": {"z": "yes"}},
                            "how": "preview", "at": at(2), "refused": None,
                            "alternative": True}]),
        decision(4, "lock_plan", {"digest": "lockdigest", "plan": {}}, [("lock_plan", {})],
                 note=None, source="app"),
        decision(5, "set_exclusions", {"rules": []}, [("set_exclusions", {"rules": []})],
                 after_estimates=True, minute=9),
    ]
    attempts = [
        attempt(1, [1, 2, 3, 4], [1, 2, 3, 4], minute=6, purpose="inference",
                notes={1: "r1", 2: "r2", 3: "r3"}, effects={"linear": copy.deepcopy(EFFECT)},
                plan_digest="lockdigest",
                movement_why="This fit is under Estimate."),
        attempt(2, [1, 2, 3, 4, 5], [5], minute=10, purpose="inference",
                effects={"linear": [{**EFFECT[0], "estimate": 0.31}]},
                movement_why="This fit is under Estimate."),
    ]
    result = {"purpose": "inference", "attempt": "fit1", "plan_digest": "lockdigest",
              "decisions": [1, 2, 3, 4], "opening": None, "family": None, "metric": None,
              "n_holdout": None, "scores": None, "intervals": None, "intervals_why": None,
              "lock": 4, "effects": {"linear": copy.deepcopy(EFFECT)}, "refused": None}
    return {"version": 1, "run": run("run-2"), **surroundings(control),
            "question": {"purpose": "inference", "target": "y", "estimand": {"exposure": "x"},
                         "intended_use": None, "stated": None, "paper": None},
            "regime": {"n_rows": 700, "n_analyzed": 690, "task": "regression", "event": None,
                       "events": None, "missing": {"z": 10}, "design": {}, "lenses": []},
            "decisions": decisions, "attempts": attempts, "result": result}


REVERT_NOTE = "Back to `a` as a predictor: excluding it lowered the AUC, so it carries signal."


def revert_record() -> dict[str, Any]:
    """Four fits under Predict and no opening: `a` excluded (record #3, fit 2), then `b` (record
    #4, fit 3), then record #3 reverted (record #5), the only change before fit 4. Its result is
    refused, since the seal was never opened."""
    def fitted(n: int, value: float) -> dict[str, Any]:
        return {"source": "fits", "attempt": f"fit{n}",
                "movement": [movement("linear", f"fit{n - 1}", value)]}

    decisions = [
        decision(1, "set_target", {"column": "y"}, [("set_target", {"column": "y"})]),
        decision(2, "select_models", {"models": ["linear"]},
                 [("select_models", {"models": ["linear"]})]),
        decision(3, "confirm_role", {"column": "a", "role": "excluded"},
                 [("role:a", "excluded")], minute=4, materiality=fitted(2, 0.4)),
        decision(4, "confirm_role", {"column": "b", "role": "excluded"},
                 [("role:b", "excluded")], minute=6, materiality=fitted(3, 0.2)),
        decision(5, "revert", {"decision_id": "r3"}, [("revert", {"decision_id": "r3"})],
                 note=REVERT_NOTE, minute=8, materiality=fitted(4, 0.4)),
    ]
    first = "No earlier fit is recorded to compare this one with."
    attempts = [
        attempt(1, [1, 2], [1, 2], minute=3, scores=CV, movement_why=first),
        attempt(2, [1, 2, 3], [3], minute=5, scores=CV, moved=[movement("linear", "fit1", 0.4)]),
        attempt(3, [1, 2, 3, 4], [4], minute=7, scores=CV,
                moved=[movement("linear", "fit2", 0.2)]),
        attempt(4, [1, 2, 4], [5], minute=9, scores=CV, notes={5: REVERT_NOTE},
                withdrawn=[(3, 5)], moved=[movement("linear", "fit3", 0.4)]),
    ]
    result = {"purpose": "prediction", "attempt": None, "plan_digest": None, "decisions": None,
              "refused": "The held-out rows were never opened."}
    return {"version": 1, "run": {**run("run-3"), "fits": 4, "fit_seconds": 6.0},
            **surroundings(),
            "question": {"purpose": "prediction", "target": "y", "estimand": None,
                         "intended_use": None, "stated": None, "paper": None},
            "regime": {"n_rows": 480, "n_analyzed": 480, "task": "binary", "event": "1",
                       "events": 112, "missing": {}, "design": {}, "lenses": []},
            "decisions": decisions, "attempts": attempts, "result": result}
