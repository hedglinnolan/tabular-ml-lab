"""The pilot's scoring (SWARM_PLAN §7): ``prereg.json`` applied mechanically, by this script and
never by the orchestrator, to the corpus's records, the final plans' replays on fresh draws, the
sweeps, the referees' verdicts and the blind reader's classes of the stuck runs.

    python -m research.swarm.score_pilot score --corpus DIR --replays DIR --sweeps FILE … \\
        [--referee FILE] [--classes FILE] [--budget-fresh N] --out DIR
    python -m research.swarm.score_pilot replay --record FILE --bundle ZIP --out FILE

**The inputs.** The records (``research/swarm/record.py``; a run whose record failed its schema
is in the corpus's ``failed/``). One replay per final plan on a fixture (:func:`make_replay`, or
any writer of the same shape): its run, its dataset, and each draw's seed with the plan's numbers
on it, or the reason the draw was refused: on the Estimate fixture the reported estimate and its
interval, on the Predict fixture the Brier regret against the true risk on the pre-registered
evaluation rows, on the measurement fixture each feature's estimate and q-value and the scale the
plan analyzed. The sweeps (``research/swarm/sweep.py``'s ``sweep.json`` and ``crash.json``). The
referees' verdicts: a list of ``{run, seq, planted, faithful}``, one per audited attempt
(``referee.py`` draws them; the referees run in the pilot). The blind reader's classes: run id →
``engine_defect``, ``driver_defect``, ``capability_gap`` or ``analyst_failure``.

**What it applies,** each rule in prereg.json's own numbers:

* the regression checks on every run (:func:`regression_checks`);
* each replay against its fixture's margins (:func:`score_replay`): the pre-registered draws or
  none, a draw refused making the plan a refused plan; b0 and the coverage band, r0, or (the
  measurement fixture) its numbers reported;
* the unit per model and fixture (:func:`unit`);
* materiality on every sweep with prereg.json's margins (:func:`sweep_verdicts`), beside the
  sweep's own verdict, each disagreement listed, never silently; and the runs whose final plans
  the sweeps go around, by when each started (:func:`named_plans`);
* reach (:func:`reach_of`), the decision rule (:func:`decision`), the persona rule
  (:func:`persona_rule`), early stopping (:func:`early_stopping`), the comparisons with the
  mechanical arms, refusals as their own rate, cost per useful result and the expected false
  alarms (:func:`score`).

**The output**: ``score.json``, every number computed, and ``score.md``, whose every number is
interpolated from it (:func:`report_md`).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from research.swarm import prereg_fill as P

CLASSES = ("engine_defect", "driver_defect", "capability_gap", "analyst_failure")
SCORED = ("within", "outside")


# ── who and where ────────────────────────────────────────────────────────────


def model_of(rec: Any, prereg: Mapping[str, Any]) -> str | None:
    """The record's model key (prereg.json's ``models``): a mechanical arm's ``mechanical-default``
    or ``mechanical-random``, else the key its ``analyst_model`` names, else None."""
    name = str(rec.run.analyst_model).lower()
    mechanical = prereg["models"]["mechanical"]
    if name.startswith(mechanical):
        return f"{mechanical}random" if "random" in name else f"{mechanical}default"
    found = [m for m in prereg["models"]["keys"] if m in name]
    return found[0] if len(found) == 1 else None


def fixture_of(prereg: Mapping[str, Any], dataset: Any) -> str | None:
    d = prereg["datasets"]
    return {d["estimate_fixture"]: "estimate", d["predict_fixture"]: "predict",
            d["measurement_fixture"]: "measurement"}.get(dataset)


def _agents(prereg: Mapping[str, Any]) -> list[str]:
    return list(prereg["models"]["keys"])


# ── the regression checks ────────────────────────────────────────────────────


def regression_checks(rec: Any) -> list[str]:
    """What fails of prereg.json's regression checks on a record that validated: the seal check,
    and a reason on every decision the analyst recorded and every option it previewed."""
    from research.swarm import checks

    problems = []
    seal = checks.seal_integrity(rec)
    if seal.outcome == "fail":
        problems += [f"seal: {p}" for p in seal.problems]
    for d in rec.decisions:
        if d.source == "analyst" and not (d.reason or "").strip():
            problems.append(f"Record #{d.seq} ({d.kind}) holds no reason.")
        for e in d.examined:
            if e.how == "preview" and not (e.reason or "").strip():
                problems.append(f"A preview of {e.question} beside record #{d.seq} holds no "
                                f"reason.")
    return problems


# ── a replay ─────────────────────────────────────────────────────────────────


def _refused(out: dict[str, Any], why: str, **more: Any) -> dict[str, Any]:
    out.update(verdict="refused", why=why, **more)
    return out


def score_replay(replay: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """One final plan's replay against its fixture's margins (the module docstring): its numbers
    and its verdict, ``within``, ``outside``, ``refused`` (a draw refused, or a number a draw does
    not hold), ``invalid`` (not the pre-registered draws), ``unjudged`` (a margin prereg.json does
    not fix) or ``reported`` (the measurement fixture)."""
    kind = fixture_of(prereg, replay.get("dataset"))
    out: dict[str, Any] = {"run": replay.get("run"), "dataset": replay.get("dataset"),
                           "kind": kind, "verdict": None, "why": None}
    first, last = prereg["replays"]["seeds"]
    expected = list(range(first, last + 1))
    draws = list(replay.get("draws") or [])
    if kind is None:
        out.update(verdict="invalid", why=f"{replay.get('dataset')!r} is not a fixture.")
        return out
    if [d.get("seed") for d in draws] != expected:
        out.update(verdict="invalid",
                   why=(f"The replay holds {len(draws)} draws, not the {len(expected)} "
                        f"pre-registered ones, seeds {first} to {last} in order."))
        return out
    out["draws"] = len(draws)
    refused = [d for d in draws if d.get("refused")]
    out["refused_draws"] = len(refused)
    if refused:
        reasons: dict[str, int] = {}
        for d in refused:
            reasons[str(d["refused"])] = reasons.get(str(d["refused"]), 0) + 1
        return _refused(out, f"The plan was refused on {len(refused)} of its {len(draws)} draws.",
                        reasons=reasons)
    if kind == "estimate":
        return _estimate(out, draws, prereg["estimate_fixture"])
    if kind == "predict":
        return _predict(out, draws, prereg["predict_fixture"])
    return _measurement(out, draws, prereg["measurement_fixture"])


def _estimate(out: dict[str, Any], draws: list[Mapping[str, Any]], fx: Mapping[str, Any]
              ) -> dict[str, Any]:
    lacking = [d["seed"] for d in draws
               if any(d.get(k) is None for k in ("estimate", "ci_low", "ci_high"))]
    if lacking:
        return _refused(out, f"{len(lacking)} draw(s) hold no estimate with its interval "
                             f"(first seed {lacking[0]}).")
    effect, b0, band = float(fx["effect"]), float(fx["b0"]), fx["coverage"]
    est = np.array([float(d["estimate"]) for d in draws])
    covered = sum(1 for d in draws if float(d["ci_low"]) <= effect <= float(d["ci_high"]))
    sd = float(est.std(ddof=1))
    bias = float(est.mean() - effect)
    bias_ok, cover_ok = abs(bias) < b0, P.in_band(covered, band)
    out.update(mean_bias=bias, sd=sd, se=sd / math.sqrt(len(est)), covered=covered,
               band=[band["low"], band["high"]], b0=b0, bias_within=bias_ok,
               coverage_within=cover_ok, verdict="within" if bias_ok and cover_ok else "outside")
    return out


def _predict(out: dict[str, Any], draws: list[Mapping[str, Any]], fx: Mapping[str, Any]
             ) -> dict[str, Any]:
    lacking = [d["seed"] for d in draws if d.get("regret") is None]
    if lacking:
        return _refused(out, f"{len(lacking)} draw(s) hold no regret (first seed {lacking[0]}).")
    regret = float(np.mean([float(d["regret"]) for d in draws]))
    r0 = fx["r0"].get("value")
    out.update(regret=regret, r0=r0)
    if r0 is None:
        out.update(verdict="unjudged", why=f"prereg.json holds no r0: {fx['r0'].get('why')}")
    else:
        out["verdict"] = "within" if regret < float(r0) else "outside"
    return out


def _measurement(out: dict[str, Any], draws: list[Mapping[str, Any]], fx: Mapping[str, Any]
                 ) -> dict[str, Any]:
    from research.swarm import generators as G

    q, truth = float(fx["q"]), set(fx["associated"])
    change = G.panel_truth()["log2_fold_change"]
    lacking = [d["seed"] for d in draws if not d.get("features")
               or any((v or {}).get("q") is None for v in d["features"].values())]
    if lacking:
        return _refused(out, f"{len(lacking)} draw(s) hold no q-value for some feature (first "
                             f"seed {lacking[0]}).")
    fdp, power, signed, away = [], [], [], []
    log2 = all(d.get("scale") == "log2" for d in draws)
    for d in draws:
        found = {f for f, v in d["features"].items() if float(v["q"]) < q}
        fdp.append(len(found - truth) / len(found) if found else 0.0)
        power.append(len(found & truth) / len(truth))
        if log2:
            for f in sorted(found & truth):
                error = float(d["features"][f]["estimate"]) - change[f]
                signed.append(error)
                away.append(math.copysign(1.0, change[f]) * error)
    oracle = float(fx["oracle"]["power"])
    out.update(fdp=float(np.mean(fdp)), nominal=q, power=float(np.mean(power)),
               oracle_power=oracle, power_share=float(np.mean(power)) / oracle,
               effect_error=float(np.mean(signed)) if signed else None,
               effect_error_away_from_zero=float(np.mean(away)) if away else None,
               effect_why=None if log2 else "The plan did not analyze every draw on the log2 "
                                            "scale, so no effect error is read on it.",
               verdict="reported")
    return out


# ── the unit ─────────────────────────────────────────────────────────────────


def unit(scored: Sequence[Mapping[str, Any]], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """§7's unit over one model's plans on one fixture: ``within``, ``outside`` or
    ``not_judged`` (fewer than the minimum of non-refused plans, or a plan without a verdict)."""
    share = Fraction(*prereg["unit"]["share"])
    minimum = int(prereg["unit"]["minimum"])
    verdicts = [s.get("verdict") for s in scored]
    judged = [v for v in verdicts if v in SCORED]
    within = verdicts.count("within")
    out: dict[str, Any] = {"plans": len(verdicts), "refused": verdicts.count("refused"),
                           "judged": len(judged), "within": within, "verdict": "not_judged",
                           "why": None}
    open_ = [s for s in scored if s.get("verdict") not in (*SCORED, "refused")]
    if open_:
        out["why"] = (f"{len(open_)} plan(s) have no verdict: "
                      + "; ".join(f"{s.get('run')}: {s.get('why')}" for s in open_[:3]))
    elif len(judged) < minimum:
        out["why"] = f"{len(judged)} non-refused plan(s), fewer than {minimum}."
    else:
        out["verdict"] = "within" if within >= share * len(judged) else "outside"
    return out


# ── the sweeps ───────────────────────────────────────────────────────────────


def point_of(kind: str, body: Mapping[str, Any]) -> str | None:
    """The sweep's point a decision answers (``sweep.points_of``' ids), or None."""
    from research.swarm import declarations as decl, sweep

    if kind in sweep.NOT_A_POINT:
        return None
    if kind in sweep.READING_KINDS:
        if kind == "confirm_reading" and body.get("reading") == "role":
            return None
        reading = body.get("reading") if kind == "confirm_reading" else "unit"
        return f"reading:{reading}:{body.get('column')}"
    if kind in sweep.FINDING_KINDS:
        return f"finding:{body.get('finding_id')}"
    if kind in decl.kinds():
        return f"declaration:{kind}"
    key = sweep.question_key(kind)
    return f"question:{key}" if key else None


def predict_material(change: Mapping[str, Any], half: float) -> bool | None:
    diff, width, excludes = (change.get(k) for k in ("difference", "half_width", "excludes_zero"))
    if diff is None or width is None or excludes is None:
        return None
    return bool(excludes) and abs(float(diff)) >= half * float(width)


def estimate_material(row: Mapping[str, Any], half: float) -> bool | None:
    shift, width = row.get("shift"), row.get("half_width")
    if shift is None or width is None:
        return None
    return abs(float(shift)) >= half * float(width)


def discoveries(rows: Sequence[Mapping[str, Any]] | None, q: float
                ) -> tuple[set[str] | None, str | None]:
    if not rows:
        return None, "No estimates are held."
    if any(r.get("q") is None for r in rows):
        return None, "The sweep's estimates carry no q-value, so no discovery set is read."
    return {str(r.get("feature")) for r in rows if float(r["q"]) < q}, None


def _any(values: Iterable[bool | None]) -> bool | None:
    known = [v for v in values if v is not None]
    return any(known) if known else None


def sweep_verdicts(sweep: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """prereg.json's materiality on each scored refit of ``sweep`` and on each decision, beside
    the sweep's own, every disagreement listed."""
    half = float(prereg["material"]["half"])
    measure = prereg["material"]["measurement"]
    measurement = sweep.get("dataset") == prereg["datasets"]["measurement_fixture"]
    prediction = sweep.get("purpose") == "prediction"
    base = sweep.get("baseline") or {}
    refits = []
    for r in sweep.get("refits") or []:
        if r.get("kind") != "scored":
            continue
        why = None
        if measurement:
            after, why_a = discoveries(r.get("estimates"), float(measure["q"]))
            before, why_b = discoveries(base.get("estimates"), float(measure["q"]))
            verdict = None if after is None or before is None else \
                len(after ^ before) >= int(measure["k"])
            why = why_a or why_b
        elif prediction:
            verdict = _any(predict_material(c, half) for c in r.get("change") or [])
        else:
            verdict = _any(estimate_material(c, half) for c in r.get("change") or [])
        if verdict is None and why is None:
            why = r.get("change_why") or "The refit's change holds no number."
        refits.append({"refit": r.get("id"), "point": r.get("point"), "option": r.get("option"),
                       "sweep": r.get("material"), "prereg": verdict, "why": why})
    points = [d.get("point") for d in sweep.get("decisions") or []]
    decisions = []
    for d in sweep.get("decisions") or []:
        mine = [x["prereg"] for x in refits if x["point"] == d.get("point")]
        decisions.append({"point": d.get("point"), "sweep": d.get("material"),
                          "prereg": _any(mine), "refused_share": d.get("refused_share"),
                          "refused_codes": d.get("refused_codes") or {}})
    return {"dataset": sweep.get("dataset"), "plan": sweep.get("plan"),
            "purpose": sweep.get("purpose"), "points": points, "refits": refits,
            "decisions": decisions,
            "material": sorted(str(d["point"]) for d in decisions if d["prereg"]),
            "disagreements": [x for x in refits if x["sweep"] != x["prereg"]],
            "decision_disagreements": [d for d in decisions if d["sweep"] != d["prereg"]],
            "fit_seconds": (sweep.get("fit_seconds") or {}).get("total")}


# ── reach ────────────────────────────────────────────────────────────────────


def examined_points(rec: Any) -> set[str]:
    """The sweep points at which the run examined an option other than its own: previewed with a
    reason, or fitted (§3)."""
    out = set()
    for d in rec.decisions:
        point = point_of(d.kind, d.option)
        if point and any(e.alternative and (e.how == "fit" or (e.reason or "").strip())
                         for e in d.examined):
            out.add(point)
    return out


def reach_of(rec: Any, union: set[str], prereg: Mapping[str, Any]) -> float | None:
    """The share of ``union`` (its dataset's material decisions) the run examined, or None where
    the union holds fewer than prereg.json's minimum."""
    if len(union) < int(prereg["reach"]["minimum_union"]):
        return None
    return len(examined_points(rec) & union) / len(union)


def named_plans(recs: Sequence[Any], prereg: Mapping[str, Any]) -> dict[str, list[dict]]:
    """The runs whose final plans are swept beyond the default (prereg.json's ``sweeps``): for
    each named model and persona on each pilot dataset, its run that started first, or None
    where it has none yet. Chosen by when the runs started, never by their results."""
    out: dict[str, list[dict]] = {}
    for dataset, wanted in prereg["sweeps"]["plans"].items():
        out[dataset] = []
        for w in wanted:
            runs = sorted((r for r in recs if r.dataset.id == dataset
                           and model_of(r, prereg) == w["model"] and r.run.persona == w["persona"]),
                          key=lambda r: (r.run.started, r.run.id))
            out[dataset].append({"model": w["model"], "persona": w["persona"],
                                 "run": runs[0].run.id if runs else None})
    return out


# ── the referees, robustness and cost ────────────────────────────────────────


def faithfulness(verdicts: Sequence[Mapping[str, Any]], runs: set[str], prereg: Mapping[str, Any]
                 ) -> dict[str, Any]:
    """The referees' verdicts on one model's runs: the reasons judged faithful among the real ones
    judged, against prereg.json's share, and the planted reasons caught."""
    need = Fraction(*prereg["decision_rule"]["faithful"])
    mine = [v for v in verdicts if v.get("run") in runs and v.get("faithful") is not None]
    real = [v for v in mine if not v.get("planted")]
    planted = [v for v in mine if v.get("planted")]
    faithful = sum(1 for v in real if v["faithful"])
    caught = sum(1 for v in planted if not v["faithful"])
    return {"judged": len(real), "faithful": faithful,
            "rate": faithful / len(real) if real else None,
            "meets": (faithful >= need * len(real)) if real else None,
            "planted": len(planted), "caught": caught,
            "catch_rate": caught / len(planted) if planted else None}


def robustness(recs: Sequence[Any], classes: Mapping[str, str], prereg: Mapping[str, Any]
               ) -> dict[str, Any]:
    """Runs that reached the export with no help, against those plus the stuck runs a blind
    reader classed analyst failures; None where a stuck run is not classed."""
    need = Fraction(*prereg["decision_rule"]["exported"])
    reached = [r for r in recs if r.export.reached and not r.run.help]
    stuck = [r for r in recs if r not in reached]
    classed = {r.run.id: classes.get(r.run.id) for r in stuck}
    unclassed = sorted(k for k, v in classed.items() if v not in CLASSES)
    failures = sum(1 for v in classed.values() if v == "analyst_failure")
    counted = len(reached) + failures
    out = {"runs": len(recs), "reached": len(reached), "stuck": len(stuck),
           "analyst_failures": failures, "unclassed": unclassed, "rate": None, "meets": None}
    if not unclassed and counted:
        out.update(rate=len(reached) / counted, meets=len(reached) >= need * counted)
    return out


# ── the scoring ──────────────────────────────────────────────────────────────


def _mean(values: Iterable[float | None]) -> float | None:
    known = [float(v) for v in values if v is not None]
    return float(np.mean(known)) if known else None


def decision(models: Mapping[str, Mapping[str, Any]], prereg: Mapping[str, Any]
             ) -> dict[str, Any]:
    """The decision rule (prereg.json's) on each candidate's unit on the Estimate fixture, its
    referees' share and its export rate; the production analyst, the cheaper of the acceptable by
    fresh tokens per useful result at its relative price."""
    rule = prereg["decision_rule"]
    out: dict[str, Any] = {"models": {}, "production": None, "why": None}
    for m in rule["candidates"]:
        s = models.get(m) or {}
        parts = {"estimate_unit": (s.get("units") or {}).get("estimate", {}).get("verdict"),
                 "faithful": (s.get("faithfulness") or {}).get("meets"),
                 "exported": (s.get("robustness") or {}).get("meets")}
        values = [parts["estimate_unit"] == "within" if parts["estimate_unit"] in SCORED
                  else None, parts["faithful"], parts["exported"]]
        acceptable = False if False in values else (None if None in values else True)
        out["models"][m] = {**parts, "acceptable": acceptable,
                            "cost_per_useful": (s.get("cost") or {}).get("cost_per_useful")}
    ref = ((models.get(rule["reference"]) or {}).get("units") or {}).get("estimate", {})
    if ref.get("verdict") == "outside":
        out["why"] = rule["reference_rule"]
        return out
    undecided = [m for m, v in out["models"].items() if v["acceptable"] is None]
    if undecided:
        out["why"] = (f"Whether {', '.join(undecided)} is acceptable cannot be said yet, so no "
                      f"production analyst is named.")
        return out
    acceptable = [m for m, v in out["models"].items() if v["acceptable"]]
    if not acceptable:
        out["why"] = "No model is acceptable: the plan returns to Nolan."
        return out
    costs = {m: out["models"][m]["cost_per_useful"] for m in acceptable}
    if any(c is None for c in costs.values()):
        out["why"] = ("The acceptable models' cost per useful result is not known (no relative "
                      "prices, or a run with no spend), so no production analyst is named.")
        return out
    cheapest = min(costs.values())
    named = sorted(m for m, c in costs.items() if c == cheapest)
    out["production"] = named[0] if len(named) == 1 else None
    out["why"] = None if len(named) == 1 else f"{' and '.join(named)} cost the same."
    return out


def persona_rule(recs: Sequence[Any], reach: Mapping[str, float | None],
                 prereg: Mapping[str, Any]) -> dict[str, Any]:
    """prereg.json's persona rule: each other persona's mean reach over its Haiku runs on the
    Estimate fixture against the conventional persona's, beside the larger of their ranges."""
    rule = prereg["persona_rule"]
    by: dict[str, list[float | None]] = {p: [] for p in prereg["models"]["personas"]}
    for r in recs:
        if (model_of(r, prereg) == rule["model"] and r.dataset.id == rule["dataset"]
                and r.run.persona in by):
            by[r.run.persona].append(reach.get(r.run.id))

    def stats(values: list[float | None]) -> dict[str, Any]:
        known = [v for v in values if v is not None]
        return {"runs": len(values), "with_reach": len(known),
                "mean": float(np.mean(known)) if known else None,
                "range": (max(known) - min(known)) if known else None}

    base = stats(by["conventional"])
    out: dict[str, Any] = {"conventional": base}
    for p in [p for p in by if p != "conventional"]:
        s = stats(by[p])
        if min(s["with_reach"], base["with_reach"]) < 2:
            s.update(kept=None, why="Fewer than two runs with a reach: deferred.")
        else:
            s.update(kept=abs(s["mean"] - base["mean"]) > max(s["range"], base["range"]),
                     why=None)
        out[p] = s
    return out


def early_stopping(units: Mapping[str, Mapping[str, Any]], recs: Sequence[Any],
                   plans: Sequence[Mapping[str, Any]], prereg: Mapping[str, Any],
                   budget_fresh: float | None) -> dict[str, Any]:
    """prereg.json's early stopping: each model outside the margins on the Estimate fixture once
    its plans there under every persona it runs are scored; and the spend projected over the
    remaining scheduled runs, against ``budget_fresh``."""
    estimate = prereg["datasets"]["estimate_fixture"]
    stopped = []
    for m in _agents(prereg):
        personas = {p for s in prereg["models"]["schedule"] if s["model"] == m
                    and estimate in s["datasets"] for p in s["personas"]}
        seen = {s.get("persona") for s in plans if s.get("model") == m
                and s.get("dataset") == estimate and s.get("verdict") in (*SCORED, "refused")}
        if personas and personas <= seen and units.get(m, {}).get("verdict") == "outside":
            stopped.append(m)
    scheduled = {m: sum(len(s["personas"]) * len(s["datasets"]) * s["runs"]
                        for s in prereg["models"]["schedule"] if s["model"] == m)
                 for m in _agents(prereg)}
    spent, remaining, means = 0.0, {}, {}
    projected: float | None = 0.0
    why = None
    for m in _agents(prereg):
        mine = [r for r in recs if model_of(r, prereg) == m]
        fresh = [r.run.spend.fresh for r in mine if r.run.spend is not None]
        spent += sum(fresh)
        remaining[m] = max(0, scheduled[m] - len(mine))
        means[m] = float(np.mean(fresh)) if fresh else None
        if remaining[m] and means[m] is None:
            projected, why = None, f"{m} has scheduled runs left and no spend measured yet."
        elif projected is not None:
            projected += remaining[m] * (means[m] or 0.0)
    if projected is not None:
        projected += spent
    stop = None if projected is None or budget_fresh is None else projected > budget_fresh
    if budget_fresh is None and why is None:
        why = "No budget in fresh tokens was given."
    return {"stopped": stopped, "spent": spent, "remaining": remaining,
            "mean_fresh_per_run": means, "projected": projected, "budget": budget_fresh,
            "stop_waves": stop, "why": why}


def score(records: Sequence[Any], replays: Sequence[Mapping[str, Any]],
          sweeps: Sequence[Mapping[str, Any]], prereg: Mapping[str, Any], *,
          referee: Sequence[Mapping[str, Any]] = (), classes: Mapping[str, str] | None = None,
          failed: Sequence[str] = (), budget_fresh: float | None = None) -> dict[str, Any]:
    """prereg.json applied to the pilot (the module docstring): every number, as data."""
    from research.swarm import checks

    classes = classes or {}
    by_run = {r.run.id: r for r in records}
    plans = []
    for rp in replays:
        s = score_replay(rp, prereg)
        rec = by_run.get(str(rp.get("run")))
        s.update(model=model_of(rec, prereg) if rec else None,
                 persona=rec.run.persona if rec else None)
        plans.append(s)
    replayed = {p["run"] for p in plans}
    for rec in records:
        kind = fixture_of(prereg, rec.dataset.id)
        if kind and rec.export.sha256 and rec.run.id not in replayed:
            plans.append({"run": rec.run.id, "dataset": rec.dataset.id, "kind": kind,
                          "model": model_of(rec, prereg), "persona": rec.run.persona,
                          "verdict": "unscored", "why": "No replay of this plan is given."})
    verdict_of = {p["run"]: p for p in plans}
    swept = [sweep_verdicts(s, prereg) for s in sweeps]
    union: dict[str, set[str]] = {}
    for s in swept:
        union.setdefault(str(s["dataset"]), set()).update(s["material"])
    reach = {r.run.id: reach_of(r, union.get(r.dataset.id, set()), prereg) for r in records}

    models: dict[str, dict[str, Any]] = {}
    prices = ((prereg.get("filled") or {}).get("relative_price") or {}).get("prices") or {}
    groups = sorted({g for g in (model_of(r, prereg) for r in records) if g})
    for m in groups:
        mine = [r for r in records if model_of(r, prereg) == m]
        runs = {r.run.id for r in mine}
        units = {kind: unit([p for p in plans if p.get("model") == m and p["kind"] == kind],
                            prereg) for kind in ("estimate", "predict")}
        useful = [r for r in mine if r.export.reached and not r.run.help and (
            fixture_of(prereg, r.dataset.id) not in ("estimate", "predict")
            or (verdict_of.get(r.run.id) or {}).get("verdict") == "within")]
        fresh = [r.run.spend.fresh if r.run.spend is not None else None for r in mine]
        total = None if None in fresh else float(sum(fresh))
        per_useful = total / len(useful) if total is not None and useful else None
        price = prices.get(m)
        mine_plans = [p for p in plans if p.get("model") == m]
        models[m] = {
            "runs": len(mine), "units": units,
            "faithfulness": faithfulness(referee, runs, prereg),
            "robustness": robustness(mine, classes, prereg),
            "refusals": {"plans": len(mine_plans),
                         "refused": sum(1 for p in mine_plans if p["verdict"] == "refused"),
                         "share": (sum(1 for p in mine_plans if p["verdict"] == "refused")
                                   / len(mine_plans)) if mine_plans else None,
                         "met_per_run": _mean(len(r.surfacings.refusals) for r in mine),
                         "fits_refused_per_run": _mean(
                             sum(1 for f in r.surfacings.failed_fits if f.kind == "refused")
                             for r in mine)},
            "cost": {"fresh": total, "useful": len(useful), "fresh_per_useful": per_useful,
                     "relative_price": price,
                     "cost_per_useful": (per_useful * float(price)
                                         if per_useful is not None and price is not None
                                         else None)},
            "reach_per_million_fresh": _mean(
                reach[r.run.id] / (r.run.spend.fresh / 1e6) for r in mine
                if reach.get(r.run.id) is not None and r.run.spend and r.run.spend.fresh),
        }
    estimate_units = {m: v["units"]["estimate"] for m, v in models.items()}
    decoys = [r for r in records if r.dataset.control == "null_decoy"]
    false_alarms = prereg["expected_false_alarms"]
    # The coverage band's misses are false alarms only for a plan unbiased by the margin.
    est_plans = [p for p in plans if p["kind"] == "estimate" and p["verdict"] in SCORED
                 and p.get("bias_within")]
    finding = []
    for s in swept:
        u = union.get(str(s["dataset"]), set())
        share = (len(set(s["material"]) & u) / len(u)
                 if len(u) >= prereg["reach"]["minimum_union"] else None)
        hours = s["fit_seconds"] / 3600 if s["fit_seconds"] else None
        finding.append({"dataset": s["dataset"], "plan": s["plan"], "reach": share,
                        "fit_hours": hours,
                        "reach_per_fit_hour": share / hours if share is not None and hours
                        else None})
    return {
        "prereg": {"version": prereg.get("version"),
                   "generators_sha256": prereg.get("generators_sha256")},
        "regression": {"failed_schema": list(failed),
                       "runs": {r.run.id: regression_checks(r) for r in records}},
        "plans": plans,
        "models": models,
        "decision": decision(models, prereg),
        "persona": persona_rule(records, reach, prereg),
        "early_stopping": early_stopping(estimate_units, records, plans, prereg, budget_fresh),
        "sweeps": swept,
        "named_plans": named_plans(records, prereg),
        "material": {d: sorted(u) for d, u in union.items()},
        "reach": reach,
        "choosing_well": {
            kind: {g: {"plans": len(ps), "within": sum(1 for p in ps if p["verdict"] == "within"),
                       "refused": sum(1 for p in ps if p["verdict"] == "refused"),
                       "mean_bias": _mean(p.get("mean_bias") for p in ps),
                       "covered": _mean(p.get("covered") for p in ps),
                       "regret": _mean(p.get("regret") for p in ps),
                       "fdp": _mean(p.get("fdp") for p in ps),
                       "power": _mean(p.get("power") for p in ps)}
                   for g in sorted({p.get("model") or "unknown" for p in plans
                                    if p["kind"] == kind})
                   for ps in [[p for p in plans if (p.get("model") or "unknown") == g
                               and p["kind"] == kind]]}
            for kind in ("estimate", "predict", "measurement")}
            | {"real": {r.run.id: {"dataset": r.dataset.id, "model": model_of(r, prereg),
                                   "family": r.result.family, "metric": r.result.metric,
                                   "scores": r.result.scores,
                                   "estimates": [e.model_dump() for e in r.result.estimates or []],
                                   "refused": r.result.refused}
                        for r in records if fixture_of(prereg, r.dataset.id) is None
                        and r.dataset.control is None}},
        "finding_what_matters": {
            "agents": {m: v["reach_per_million_fresh"] for m, v in models.items()},
            "sweeps": finding},
        "false_alarms": {
            "decoy": {"runs": len(decoys),
                      "expected": len(decoys) * float(false_alarms["decoy"]["per_run"]),
                      "observed": sum(1 for r in decoys
                                      if checks.null_decoy(r).outcome == "fail")},
            "coverage": {"plans": len(est_plans),
                         "expected": len(est_plans)
                         * float(false_alarms["coverage_band"]["per_plan"]),
                         "observed": sum(1 for p in est_plans if not p["coverage_within"])}},
    }


# ── the report ───────────────────────────────────────────────────────────────


def _n(x: Any, digits: int = 4) -> str:
    if x is None:
        return "none"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{float(x):.{digits}g}"


def report_md(report: Mapping[str, Any], prereg: Mapping[str, Any]) -> str:
    """``report`` in sentences, every number interpolated from it or from prereg.json."""
    fx = prereg["estimate_fixture"]
    lines = ["# The pilot's scoring", "",
             f"prereg.json version {report['prereg']['version']}; b0 = {_n(fx['b0'])}, the "
             f"coverage band {fx['coverage']['low']} to {fx['coverage']['high']} of "
             f"{fx['coverage']['draws']}, r0 = "
             f"{_n(prereg['predict_fixture']['r0'].get('value'))}, k = "
             f"{prereg['material']['measurement']['k']}.", ""]
    d = report["decision"]
    lines += ["## The decision", "",
              (f"The production analyst is {d['production']}." if d["production"]
               else f"No production analyst is named: {d['why']}"), ""]
    lines += ["| Model | Estimate unit | Faithful | Exported | Acceptable | Cost per useful |",
              "|---|---|---|---|---|---|"]
    for m, v in d["models"].items():
        lines.append(f"| {m} | {v['estimate_unit'] or 'none'} | {_n(v['faithful'])} | "
                     f"{_n(v['exported'])} | {_n(v['acceptable'])} | "
                     f"{_n(v['cost_per_useful'])} |")
    lines += ["", "## The plans", "",
              "| Run | Model | Fixture | Verdict | Mean bias | Covered | Regret | Why |",
              "|---|---|---|---|---|---|---|---|"]
    for p in report["plans"]:
        lines.append(f"| {p['run']} | {p.get('model') or 'none'} | {p['kind']} | {p['verdict']} "
                     f"| {_n(p.get('mean_bias'))} | {_n(p.get('covered'))} | "
                     f"{_n(p.get('regret'))} | {p.get('why') or ''} |")
    lines += ["", "## Materiality: prereg.json against each sweep's own verdict", ""]
    for s in report["sweeps"]:
        lines.append(f"- {s['dataset']} around the {s['plan']} plan: "
                     f"{len(s['material'])} material decision(s); "
                     f"{len(s['disagreements'])} refit(s) where the two verdicts differ"
                     + (": " + "; ".join(f"`{x['refit']}` (sweep {_n(x['sweep'])}, prereg.json "
                                         f"{_n(x['prereg'])})" for x in s["disagreements"][:10])
                        if s["disagreements"] else "") + ".")
    persona = report["persona"]
    lines += ["", "## The personas", ""]
    for p, v in persona.items():
        if p == "conventional":
            continue
        base = persona["conventional"]
        lines.append(f"- {p}: mean reach {_n(v['mean'])} over {v['with_reach']} run(s), range "
                     f"{_n(v['range'])}, against the conventional persona's {_n(base['mean'])} "
                     f"(range {_n(base['range'])}): "
                     + ("kept." if v["kept"] else "not kept." if v["kept"] is False
                        else v["why"]))
    e = report["early_stopping"]
    lines += ["", "## Early stopping", "",
              f"Stopped: {', '.join(e['stopped']) or 'none'}. Spent {_n(e['spent'])} fresh "
              f"tokens; projected {_n(e['projected'])} against a budget of {_n(e['budget'])}: "
              + ("the waves stop." if e["stop_waves"] else "the waves go on."
                 if e["stop_waves"] is False else f"not judged ({e['why']})"), ""]
    f = report["false_alarms"]
    lines += ["## Expected false alarms", "",
              f"Decoy runs: {f['decoy']['runs']}, {_n(f['decoy']['expected'])} false alarm(s) "
              f"expected, {f['decoy']['observed']} observed. Estimate plans whose mean bias is "
              f"within b0: {f['coverage']['plans']}, {_n(f['coverage']['expected'])} coverage "
              f"miss(es) expected, {f['coverage']['observed']} observed.", ""]
    bad = {k: v for k, v in report["regression"]["runs"].items() if v}
    lines += ["## Regression checks", "",
              f"{len(report['regression']['failed_schema'])} record(s) failed the schema; "
              f"{len(bad)} run(s) failed another check"
              + (": " + "; ".join(f"{k}: {v[0]}" for k, v in list(bad.items())[:10]) if bad
                 else "") + "."]
    return "\n".join(lines) + "\n"


# ── the replay of a final plan on fresh draws ────────────────────────────────

VALUE_COLUMNS = {"estimate": ("ci_low", "ci_high"), "ratio": ("ratio_low", "ratio_high"),
                 "rd": ("rd_low", "rd_high"), "rr": ("rr_low", "rr_high")}


def locate(estimates: Mapping[str, float | None], family: str, labels: Sequence[str],
           value: float) -> tuple[str, str, str] | None:
    """The one row of a bundle's reported numbers (``provenance.json``' ``estimates``, keyed
    ``<table>|<row>|<column>``) whose family is ``family``, whose row ends in one of ``labels``
    and whose value column holds ``value``: ``(table, row, column)``, or None where none or
    several do."""
    hits = []
    for key, v in estimates.items():
        table, _, rest = key.partition("|")
        row, _, column = rest.rpartition("|")
        if column not in VALUE_COLUMNS or v is None or not row.startswith(f"{family}/"):
            continue
        if row.rsplit("/", 1)[-1] in labels and math.isclose(float(v), float(value),
                                                            rel_tol=1e-9, abs_tol=1e-12):
            hits.append((table, row, column))
    return hits[0] if len(hits) == 1 else None


def _bundle_estimates(bundle: Path) -> dict[str, float | None]:
    import zipfile

    with zipfile.ZipFile(bundle) as z:
        name = next(n for n in z.namelist() if n.endswith("/provenance.json"))
        return dict(json.loads(z.read(name)).get("estimates") or {})


def target_of(rec: Any, prereg: Mapping[str, Any]) -> dict[str, Any]:
    """What a replay of the record's final plan reads on each draw: its kind, and its reported
    family with (Estimate) the exposure's row and value, (measurement) every feature's."""
    kind = fixture_of(prereg, rec.dataset.id)
    if kind == "predict":
        return {"kind": kind, "family": rec.result.family}
    rows = [e for e in rec.result.estimates or [] if e.estimate is not None]
    out = {"kind": kind, "family": rows[0].family if rows else None, "rows": [
        {"feature": e.feature, "setting": e.setting, "estimate": e.estimate} for e in rows]}
    if kind == "measurement":
        from turbotab.core.methods import omics

        logged = [d for d in rec.decisions if d.kind == "apply_repair"
                  and d.option.get("option") in omics.INTENSITY_METHODS
                  and str(d.option.get("option")).endswith("log2")]
        out["scale"] = "log2" if logged else None
    return out


def make_replay(bundle: Path, target: Mapping[str, Any], dataset: str, run: str,
                prereg: Mapping[str, Any], *, generator: Any = None,
                seeds: Sequence[int] | None = None, workers: int = 2,
                evaluation: Any = None, keep: Path | None = None) -> dict[str, Any]:
    """The final plan in ``bundle`` replayed on each pre-registered fresh draw of its fixture's
    generator (``core/export/replay.py``), read as ``target`` says (:func:`target_of`); a draw the
    replay or the scorer refuses is kept with its reason. ``generator``, ``seeds`` and
    ``evaluation`` (the Predict fixture's rows) stand in for prereg.json's in a test; ``keep``
    keeps each Predict draw's rebuilt home in ``keep/<seed>`` (its ``project`` in the draw),
    which is otherwise removed once scored."""
    from turbotab.core.export import replay, scorer

    from research.swarm import generators as G

    kind = target["kind"]
    fx = prereg[{"estimate": "estimate_fixture", "predict": "predict_fixture",
                 "measurement": "measurement_fixture"}[kind]]
    first, last = prereg["replays"]["seeds"]
    seeds = list(seeds) if seeds is not None else list(range(first, last + 1))
    generator = generator or fx["generator"]
    out: dict[str, Any] = {"run": run, "dataset": dataset, "generator": str(fx["generator"]),
                           "recorded_seed": fx["recorded_seed"], "target": dict(target),
                           "draws": []}
    where: tuple[str, str, str] | None = None
    if kind != "predict":
        rows = target.get("rows") or []
        if not rows:
            return out | {"refused": "The record reports no estimate to replay."}
        first_row = rows[0]
        where = locate(_bundle_estimates(Path(bundle)), str(target["family"]),
                       [str(first_row["feature"]), str(first_row["setting"])],
                       float(first_row["estimate"]))
        if where is None:
            return out | {"refused": ("The bundle's reported numbers hold no one row equal to "
                                      "the record's estimate, so a draw's estimate cannot be "
                                      "read as the record's.")}
    if kind == "predict":
        ev = evaluation if evaluation is not None else G.readmission(
            prereg["replays"]["predict_evaluation"]["seed"],
            prereg["replays"]["predict_evaluation"]["rows"])
        risk = G.readmission_risk(ev)
    for seed in seeds:
        fresh = replay.FreshDraw(generator, int(fx["recorded_seed"]), int(seed))
        draw: dict[str, Any] = {"seed": int(seed), "refused": None}
        if kind == "predict":
            home = (Path(keep) / str(seed)) if keep is not None else Path(
                tempfile.mkdtemp(prefix="turbotab-pilot-replay-"))
            try:
                rep = replay.run(bundle, fresh=fresh, cut=True, keep=True,
                                 home=str(home / "home"), workers=workers)
                if rep.refused or not rep.finished:
                    draw["refused"] = rep.refused or "; ".join(rep.messages) or "not finished"
                else:
                    if keep is not None:
                        draw["project"] = rep.project
                    try:
                        got = scorer.predict_in(rep.home, rep.project, ev,
                                                family=target.get("family"), workers=workers)
                        draw["regret"] = scorer.brier_regret(got.predictions, risk)
                    except scorer.ScoreRefused as exc:
                        draw["refused"] = exc.message
            finally:
                if keep is None:
                    shutil.rmtree(home, ignore_errors=True)
        else:
            rep = replay.run(bundle, fresh=fresh, workers=workers)
            if rep.refused or not rep.finished:
                draw["refused"] = rep.refused or "; ".join(rep.messages) or "not finished"
            else:
                again = rep.estimates_again or {}
                table, row, column = where  # type: ignore[misc]
                low, high = VALUE_COLUMNS[column]
                if kind == "estimate":
                    draw.update({k: again.get(f"{table}|{row}|{c}") for k, c in
                                 (("estimate", column), ("ci_low", low), ("ci_high", high),
                                  ("se", "se"), ("df", "df"))})
                else:
                    model = row.rsplit("/", 1)[0]
                    draw["scale"] = target.get("scale")
                    draw["features"] = {
                        str(r["feature"]): {"estimate": again.get(
                            f"{table}|{model}/{r['feature']}|{column}"),
                            "q": again.get(f"{table}|{model}/{r['feature']}|q")}
                        for r in target["rows"]}
        out["draws"].append(draw)
    return out


# ── the command ──────────────────────────────────────────────────────────────


def _records(folder: Path) -> tuple[list[Any], list[str]]:
    from research.swarm.record import validate

    records = [validate(json.loads(p.read_text("utf-8"))) for p in sorted(folder.glob("*.json"))]
    failed = [str(p) for p in sorted((folder / "failed").glob("*.json"))]
    return records, failed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.score_pilot",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--prereg", type=Path, default=P.PREREG)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="apply prereg.json to the pilot")
    s.add_argument("--corpus", type=Path, required=True)
    s.add_argument("--replays", type=Path, required=True)
    s.add_argument("--sweeps", type=Path, nargs="*", default=[])
    s.add_argument("--referee", type=Path)
    s.add_argument("--classes", type=Path)
    s.add_argument("--budget-fresh", type=float)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("replay", help="replay one final plan on the pre-registered draws")
    r.add_argument("--record", type=Path, required=True)
    r.add_argument("--bundle", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    prereg = P.load(args.prereg)
    if args.command == "replay":
        from research.swarm.record import validate

        rec = validate(json.loads(args.record.read_text("utf-8")))
        if fixture_of(prereg, rec.dataset.id) is None:
            print(json.dumps({"refused": f"{rec.dataset.id} is not a fixture."}))
            return 1
        out = make_replay(args.bundle, target_of(rec, prereg), rec.dataset.id, rec.run.id,
                          prereg, workers=args.workers)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n", "utf-8")
        print(json.dumps({"replay": str(args.out), "draws": len(out["draws"]),
                          "refused": sum(1 for d in out["draws"] if d.get("refused"))}))
        return 0
    records, failed = _records(args.corpus)
    replays = [json.loads(p.read_text("utf-8")) for p in sorted(args.replays.glob("*.json"))]
    sweeps = [json.loads(p.read_text("utf-8")) for p in args.sweeps]
    referee = json.loads(args.referee.read_text("utf-8")) if args.referee else []
    classes = json.loads(args.classes.read_text("utf-8")) if args.classes else {}
    report = score(records, replays, sweeps, prereg, referee=referee, classes=classes,
                   failed=failed, budget_fresh=args.budget_fresh)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "score.json").write_text(json.dumps(report, indent=1, default=str) + "\n",
                                         "utf-8")
    (args.out / "score.md").write_text(report_md(report, prereg), "utf-8")
    print(json.dumps({"score": str(args.out / "score.json"),
                      "production": report["decision"]["production"],
                      "why": report["decision"]["why"]}))
    return 0


__all__ = ["decision", "early_stopping", "examined_points", "faithfulness", "fixture_of",
           "locate", "main", "make_replay", "model_of", "named_plans", "persona_rule",
           "point_of", "reach_of", "regression_checks", "report_md", "robustness", "score",
           "score_replay", "sweep_verdicts", "target_of", "unit"]


if __name__ == "__main__":
    sys.exit(main())
