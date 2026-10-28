"""The pilot's scoring (SWARM_PLAN §7): ``prereg.json`` applied mechanically, by this script and
never by the orchestrator, to the corpus's records, the final plans' replays on fresh draws, the
sweeps, the referees' verdicts and the blind reader's classes of the stuck runs.

    python -m research.swarm.score_pilot score --corpus DIR --replays DIR --sweeps FILE … \\
        [--referee FILE] [--classes FILE] [--budget-fresh N] --out DIR
    python -m research.swarm.score_pilot replay --record FILE --bundle ZIP --out FILE

``replay`` replays the record's final plan from its own export: a bundle whose SHA-256 is not the
record's export's is refused before any draw.

**The inputs.** The records (``research/swarm/record.py``; a run whose record failed its schema
is in the corpus's ``failed/``). One replay per final plan on a fixture (:func:`make_replay`, or
any writer of the same shape): its run, its dataset, the bundle's SHA-256, what it reads
(``target``, :func:`target_of`), the generator and recorded seed it drew from, on the Predict
fixture the evaluation rows, the engine it ran on (``launcher.engine_now``), and each draw's seed
with the plan's numbers on it, or the reason the draw was refused: on the Estimate fixture the
reported estimate and its interval, on the Predict fixture the Brier regret against the true risk
on the pre-registered evaluation rows, on the measurement fixture each feature's estimate and
q-value and the scale the plan analyzed; or, with no draws, why at its top: ``refused`` (the run
reported no result) or ``unjudged`` (the record cannot say what the run reported, or the bundle
does not hold its reported number once). The sweeps (``research/swarm/sweep.py``'s ``sweep.json``
and ``crash.json``). The referees' verdicts: a list of ``{run, seq, planted, faithful}``, one per
audited attempt (``referee.py`` draws them; the referees run in the pilot). The blind reader's
classes: run id → ``engine_defect``, ``driver_defect``, ``capability_gap`` or
``analyst_failure``.

**What it applies,** each rule in prereg.json's own numbers:

* first, the engine each record was made on (:func:`engine_refusal`): a record whose engine
  commit's ``turbotab/`` tree is not the frozen one (the crash sweeps', ``prereg_fill.
  frozen_engine``), or whose ``engine_modified`` is not False, is refused, listed with why
  (``refused_records``), and scored nowhere (its replay is no plan's); with no crash fill the
  frozen engine is not known, and every record is refused so. Its tokens were spent all the
  same: they count against the budget in early stopping, where its run is still one to do, and
  in no model's cost per useful result, which is of the runs on the frozen engine;
* the regression checks on every run (:func:`regression_checks`);
* each replay judged only as its record's plan (:func:`paired`: its run, dataset, bundle and
  reading the record's; a replay with no record, or one of two of a run, judges nothing), on the
  pre-registered draws and rows on the frozen engine (:func:`provenance`: the fixture's generator
  and recorded seed, the Predict fixture's evaluation rows, the crash sweeps' engine tree with no
  file modified; with no crash fill the frozen engine is not known, and nothing is judged);
* each replay against its fixture's margins (:func:`score_replay`): the pre-registered draws or
  none, a draw refused making the plan a refused plan, as a replay refused as a whole does; b0
  and the coverage band, r0, or (the measurement fixture) its numbers reported. An exported
  record on a fixture whose result is refused is judged by its record whatever replay is given
  (:func:`result_refusal`): a refused plan where the run reported no result or the engine served
  none, unjudged where the record cannot say. A run whose export was refused is no plan, except
  where its engine served no result (:func:`unexported_refusal`, as under Estimate for a family
  with no coefficient table): that is a refused plan, the mechanical arms' and the analysts'
  alike;
* the unit per model and fixture (:func:`unit`);
* materiality on every sweep with prereg.json's margins (:func:`sweep_verdicts`), but a sweep
  not run on the frozen engine (:func:`sweep_refusal`: its report says the engine changed while
  it ran, or does not say, or it ended on another ``turbotab/`` tree than the crash sweeps', or
  with tracked files modified or not known to be unmodified; with no crash fill the frozen engine
  is not known), which is refused, listed with why naming its tree and the frozen one
  (``refused_sweeps``), and read for nothing; beside the
  sweep's own verdict, each disagreement listed, never silently; a refit scored as the plan in
  every one of the same folds not material (its change zero, though with no spread it has no
  interval: ``sweep.unchanged_of``); a decision none of whose refits
  is read counted apart, as not read, with why, and a dataset whose sweeps prereg.json's rule
  read no refit of holding no material set (the measurement fixture's k rule reads the q-values
  the sweep's estimates carry); and the runs whose final plans the sweeps go around, by when each
  started (:func:`named_plans`);
* reach (:func:`reach_of`), the decision rule (:func:`decision`; no analyst named until the
  reference is within), the persona rule (:func:`persona_rule`), early stopping
  (:func:`early_stopping`; once a model has a record under every persona, a stuck run's
  included), the comparisons with the mechanical arms, refusals as their own rate, cost per
  useful result and the expected false alarms (:func:`score`; the coverage band's bounded over
  the plans judged on coverage, a count outside the band reported as that, never as a false
  alarm).

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
FIXTURE_KEY = {"estimate": "estimate_fixture", "predict": "predict_fixture",
               "measurement": "measurement_fixture"}


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


# A record's result with no numbers (``record.RefusedKind``): the run reported none, or the engine
# served none, is a refused plan; a record that cannot say what the run reported is not judged,
# as the record's own checks judge it ("does not judge one it cannot know").
NO_RESULT = {"not_reported": ("refused", "none reported"), "not_served": ("refused", "none served"),
             "not_known": ("unjudged", "the record cannot say what the run reported")}


def result_refusal(kind: str | None, words: str) -> tuple[str, str]:
    """The verdict and why of a plan whose record's result is refused (:data:`NO_RESULT`)."""
    verdict, said = NO_RESULT.get(str(kind), ("unjudged", f"of an unknown kind {kind!r}"))
    return verdict, f"The record reports no result ({said}, {kind}): {words}"


def unexported_refusal(rec: Any) -> bool:
    """Whether ``rec``, a run whose export the server refused, is a refused plan all the same
    (SWARM_PLAN §7: "A refused plan counts as neither, and goes into the refusal rate", for every
    arm and model alike): its engine served no result for the plan the run locked or opened
    (``not_served``), as under Estimate for a family with no coefficient table, whose effects
    stage refuses, so its export is refused and the run stops there (the random arm's draw of
    one, ``arms.run_arm``). A run never exported that reported no result (stuck before its lock
    or its opening, ``not_reported``) or whose record cannot say (``not_known``) is no plan."""
    return not rec.export.sha256 and rec.result.refused_kind == "not_served"


def _same(a: Any, b: Any) -> bool:
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def provenance(replay: Mapping[str, Any], prereg: Mapping[str, Any]) -> tuple[str, str] | None:
    """Why ``replay`` (on a fixture) is not on the pre-registered draws and rows on the frozen
    engine, as ``(verdict, why)``: ``invalid`` where it names another generator or recorded seed
    than its fixture's, other evaluation rows than prereg.json's (the Predict fixture), or an
    engine other than the crash sweeps' (``prereg_fill.frozen_engine``: their ``turbotab/`` tree,
    no tracked file modified); ``unjudged`` where prereg.json holds no crash fill, so the frozen
    engine is not known; else None."""
    kind = fixture_of(prereg, replay.get("dataset"))
    fx = prereg[FIXTURE_KEY[str(kind)]]
    wanted = [("generator", "generator", fx["generator"]),
              ("recorded_seed", "recorded seed", fx["recorded_seed"])]
    if kind == "predict":
        wanted.append(("evaluation", "evaluation rows", prereg["replays"]["predict_evaluation"]))
    for key, said, want in wanted:
        if not _same(replay.get(key), want):
            return "invalid", (f"The replay's {said} is {replay.get(key)!r}, not the "
                               f"pre-registered {want!r}.")
    frozen = P.frozen_engine(prereg)
    if frozen is None:
        return "unjudged", ("prereg.json holds no crash fill, so the frozen engine a replay must "
                            "run on is not known.")
    engine = replay.get("engine") or {}
    if engine.get("engine_tree") != frozen["engine_tree"] or engine.get("engine_modified") \
            is not False:
        state = ("no file modified" if engine.get("engine_modified") is False else
                 "tracked files modified" if engine.get("engine_modified") else
                 "not known to be unmodified")
        return "invalid", (f"The replay ran on engine tree {engine.get('engine_tree')} ({state}); "
                           f"the frozen engine the crash sweeps passed on is tree "
                           f"{frozen['engine_tree']} with no file modified.")
    return None


def paired(replay: Mapping[str, Any], rec: Any, prereg: Mapping[str, Any]) -> str | None:
    """Why ``replay`` is not ``rec``'s final plan replayed, or None: its run, its dataset, its
    bundle (the record's export, by SHA-256) and what it reads (:func:`target_of` the record) must
    each be the record's."""
    if str(replay.get("run")) != str(rec.run.id):
        return f"The replay's run is {replay.get('run')!r}, not its record's {rec.run.id!r}."
    if replay.get("dataset") != rec.dataset.id:
        return (f"The replay's dataset is {replay.get('dataset')!r}, not its record's "
                f"{rec.dataset.id!r}.")
    if not rec.export.sha256 or replay.get("bundle_sha256") != rec.export.sha256:
        return (f"The replay's bundle is {replay.get('bundle_sha256')!r}, not its record's "
                f"export {rec.export.sha256!r}.")
    target, want = replay.get("target") or {}, target_of(rec, prereg)
    if target.get("family") != want.get("family"):
        return (f"The replay reads the family `{target.get('family')}`, not its record's final "
                f"`{want.get('family')}`.")
    if not _same(target, want):
        return (f"The replay reads {json.dumps(target)}, not what its record reported "
                f"({json.dumps(want)}).")
    return None


def score_replay(replay: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """One final plan's replay against its fixture's margins (the module docstring): its numbers
    and its verdict, ``within``, ``outside``, ``refused`` (a draw refused, a number a draw does
    not hold, or the replay refused as a whole: its run reported no result), ``invalid`` (not the
    pre-registered draws and rows on the frozen engine, :func:`provenance`), ``unjudged`` (a
    margin prereg.json does not fix, no frozen engine known, or the replay not read as a whole,
    with why) or ``reported`` (the measurement fixture). Whether it is its record's plan is
    :func:`paired`'s, in :func:`score`."""
    kind = fixture_of(prereg, replay.get("dataset"))
    out: dict[str, Any] = {"run": replay.get("run"), "dataset": replay.get("dataset"),
                           "kind": kind, "verdict": None, "why": None}
    first, last = prereg["replays"]["seeds"]
    expected = list(range(first, last + 1))
    draws = list(replay.get("draws") or [])
    if kind is None:
        out.update(verdict="invalid", why=f"{replay.get('dataset')!r} is not a fixture.")
        return out
    if replay.get("refused"):
        return _refused(out, str(replay["refused"]))
    if replay.get("unjudged"):
        out.update(verdict="unjudged", why=str(replay["unjudged"]))
        return out
    off = provenance(replay, prereg)
    if off is not None:
        out.update(verdict=off[0], why=off[1])
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


def predict_material(change: Mapping[str, Any], half: float, unchanged: bool = False
                     ) -> bool | None:
    """prereg.json's Predict rule on one family's change; not material where the refit scored as
    the plan in every one of the same folds (``unchanged``, ``sweep.unchanged_of``: the change is
    zero, though with no spread it has no interval); None where it has no number otherwise."""
    diff, width, excludes = (change.get(k) for k in ("difference", "half_width", "excludes_zero"))
    if diff is None and unchanged:
        return False
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


def rule_of(sweep: Mapping[str, Any], prereg: Mapping[str, Any]) -> str:
    """The name of prereg.json's materiality rule on ``sweep``, in words."""
    if sweep.get("dataset") == prereg["datasets"]["measurement_fixture"]:
        return (f"discovery-set rule (k = {prereg['material']['measurement']['k']} features at "
                f"q < {prereg['material']['measurement']['q']:g})")
    return "Predict rule" if sweep.get("purpose") == "prediction" else "Estimate rule"


NO_FROZEN_SWEEP = ("prereg.json holds no crash fill, so the frozen engine a sweep must run on is "
                   "not known.")


def sweep_refusal(sweep: Mapping[str, Any], frozen: Mapping[str, Any] | None) -> str | None:
    """Why ``sweep`` (a sweep's report, ``sweep.json`` or ``crash.json``) is refused for the engine
    it ran on, in words naming its ``turbotab/`` tree and the frozen one: its report says the
    engine changed while it ran (``sweep.engine_changed``, its words), or does not say whether it
    did, or it ended on another tree than the crash sweeps' (``prereg_fill.frozen_engine``), or
    with tracked files modified or not known to be unmodified; with no crash fill the frozen
    engine is not known (:data:`NO_FROZEN_SWEEP`). None where it ran on the frozen engine from its
    start to its end."""
    if frozen is None:
        return NO_FROZEN_SWEEP
    tree, modified = sweep.get("engine_tree"), sweep.get("engine_modified")
    state = ("no file modified" if modified is False else
             "tracked files modified" if modified else "not known to be unmodified")
    on = f"The sweep ran on engine tree {tree} ({state})"
    want = (f"the frozen engine the crash sweeps passed on is tree {frozen['engine_tree']} with no "
            f"file modified")
    if "engine_changed" not in sweep:
        return (f"{on}, and its report does not say whether the engine changed while it ran; "
                f"{want}.")
    if sweep["engine_changed"]:
        return (f"{on}, and its report says the engine changed while it ran: "
                f"{str(sweep['engine_changed']).rstrip('.')}; {want}.")
    if tree != frozen["engine_tree"] or modified is not False:
        return f"{on}; {want}."
    return None


def sweep_verdicts(sweep: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """prereg.json's materiality on each scored refit of ``sweep`` and on each decision, beside
    the sweep's own, every disagreement listed; each decision none of whose refits is read
    counted apart, as not read, with why; and whether the rule read any refit at all
    (``applied``), with why where it read none. A sweep not run on the frozen engine
    (:func:`sweep_refusal`) is refused, its why in ``refused``, and nothing of it is read."""
    refused = sweep_refusal(sweep, P.frozen_engine(prereg))
    if refused is not None:
        return {"dataset": sweep.get("dataset"), "plan": sweep.get("plan"),
                "purpose": sweep.get("purpose"), "rule": rule_of(sweep, prereg),
                "refused": refused, "engine_tree": sweep.get("engine_tree"), "points": [],
                "refits": [], "decisions": [], "applied": False, "not_applied": refused,
                "material": [], "not_read": [], "disagreements": [],
                "decision_disagreements": [], "fit_seconds": None}
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
            from research.swarm.sweep import unchanged_of

            after = (r.get("scores") or {}).get("families")
            before = (base.get("scores") or {}).get("families")
            verdict = _any(predict_material(c, half, unchanged_of(c, after, before))
                           for c in r.get("change") or [])
        else:
            verdict = _any(estimate_material(c, half) for c in r.get("change") or [])
        if verdict is None and why is None:
            why = r.get("change_why") or " ".join(dict.fromkeys(
                str(c["why"]) for c in r.get("change") or [] if c.get("why"))) or \
                "The refit's change holds no number."
        refits.append({"refit": r.get("id"), "point": r.get("point"), "option": r.get("option"),
                       "sweep": r.get("material"), "prereg": verdict, "why": why})
    points = [d.get("point") for d in sweep.get("decisions") or []]
    decisions = []
    for d in sweep.get("decisions") or []:
        mine = [x for x in refits if x["point"] == d.get("point")]
        verdict = _any(x["prereg"] for x in mine)
        why = None
        if verdict is None:
            why = " ".join(dict.fromkeys(str(x["why"]) for x in mine if x["why"])) or \
                "No refit of it was scored."
        decisions.append({"point": d.get("point"), "sweep": d.get("material"),
                          "prereg": verdict, "why": why,
                          "refused_share": d.get("refused_share"),
                          "refused_codes": d.get("refused_codes") or {}})
    applied = any(x["prereg"] is not None for x in refits)
    not_applied = None
    if not applied:
        not_applied = " ".join(dict.fromkeys(str(x["why"]) for x in refits if x["why"])) or \
            "The sweep holds no scored refit."
    return {"dataset": sweep.get("dataset"), "plan": sweep.get("plan"),
            "purpose": sweep.get("purpose"), "rule": rule_of(sweep, prereg), "refused": None,
            "engine_tree": sweep.get("engine_tree"), "points": points,
            "refits": refits, "decisions": decisions, "applied": applied,
            "not_applied": not_applied,
            "material": sorted(str(d["point"]) for d in decisions if d["prereg"]),
            "not_read": [{"point": d["point"], "why": d["why"]} for d in decisions
                         if d["prereg"] is None],
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
    fresh tokens per useful result at its relative price, named only once the reference's own
    unit there is within (its ``reference_rule``)."""
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
    name = rule["reference"]
    ref = models.get(name)
    unit_ = ((ref or {}).get("units") or {}).get("estimate") or {}
    if unit_.get("verdict") != "within":
        state = ("has no run" if ref is None else "is outside the margins"
                 if unit_.get("verdict") == "outside" else
                 f"is not judged ({str(unit_.get('why') or 'no verdict').rstrip('.')})")
        out["why"] = f"The reference, {name}, {state} on the Estimate fixture. " \
                     f"{rule['reference_rule']}"
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
                   prereg: Mapping[str, Any], budget_fresh: float | None,
                   refused: Sequence[Any] = ()) -> dict[str, Any]:
    """prereg.json's early stopping: each model whose unit on the Estimate fixture (``units``) is
    outside once it has a record there under every persona it runs (a run that ended, one stuck
    before its export and so with no plan included); and the spend projected over the remaining
    scheduled runs, against ``budget_fresh``. ``recs`` are the records scored, ``refused`` those
    refused for their engine (:func:`engine_refusal`): a refused run's tokens were spent, so they
    are in the spend (``spent``, and apart in ``refused_fresh``) and in its model's mean per run,
    each record's once, and the run is still to do, since it is none of the pilot's runs (no
    persona has run by it, and its scheduled run stays among the ``remaining``)."""
    estimate = prereg["datasets"]["estimate_fixture"]
    stopped = []
    for m in _agents(prereg):
        personas = {p for s in prereg["models"]["schedule"] if s["model"] == m
                    and estimate in s["datasets"] for p in s["personas"]}
        ran = {r.run.persona for r in recs if model_of(r, prereg) == m
               and r.dataset.id == estimate}
        if personas and personas <= ran and units.get(m, {}).get("verdict") == "outside":
            stopped.append(m)
    scheduled = {m: sum(len(s["personas"]) * len(s["datasets"]) * s["runs"]
                        for s in prereg["models"]["schedule"] if s["model"] == m)
                 for m in _agents(prereg)}
    spent, remaining, means = 0.0, {}, {}
    projected: float | None = 0.0
    why = None
    for m in _agents(prereg):
        mine = [r for r in recs if model_of(r, prereg) == m]
        fresh = [r.run.spend.fresh for r in [*mine, *(r for r in refused
                                                     if model_of(r, prereg) == m)]
                 if r.run.spend is not None]
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
    refused_fresh = [r.run.spend.fresh for r in refused if r.run.spend is not None]
    return {"stopped": stopped, "spent": spent, "refused_fresh": float(sum(refused_fresh)),
            "refused_runs": len(refused), "remaining": remaining,
            "mean_fresh_per_run": means, "projected": projected, "budget": budget_fresh,
            "stop_waves": stop, "why": why}


NO_FROZEN_ENGINE = ("prereg.json holds no crash fill, so the frozen engine a record must be made "
                    "on is not known.")


def engine_refusal(rec: Any, frozen: Mapping[str, Any] | None) -> str | None:
    """Why ``rec`` is refused for the engine it was made on (the module docstring): not the
    frozen one (``prereg_fill.record_engine_why``: its commit's ``turbotab/`` tree another than
    the crash sweeps', or tracked files modified, or not known to be unmodified), or no frozen
    engine known (:data:`NO_FROZEN_ENGINE`); None where it was made on the frozen engine."""
    if frozen is None:
        return NO_FROZEN_ENGINE
    return P.record_engine_why(rec, frozen)[1]


def score(records: Sequence[Any], replays: Sequence[Mapping[str, Any]],
          sweeps: Sequence[Mapping[str, Any]], prereg: Mapping[str, Any], *,
          referee: Sequence[Mapping[str, Any]] = (), classes: Mapping[str, str] | None = None,
          failed: Sequence[str] = (), budget_fresh: float | None = None) -> dict[str, Any]:
    """prereg.json applied to the pilot (the module docstring): every number, as data. A record
    not made on the frozen engine is refused before anything is scored (:func:`engine_refusal`),
    listed with why (``refused_records``), and scored nowhere; only its spend counts, against the
    budget (:func:`early_stopping`)."""
    from research.swarm import checks

    classes = classes or {}
    frozen = P.frozen_engine(prereg)
    refused_records, kept, refused = [], [], []
    for rec in records:
        why = engine_refusal(rec, frozen)
        if why is None:
            kept.append(rec)
        else:
            refused.append(rec)
            refused_records.append({"run": rec.run.id, "dataset": rec.dataset.id,
                                    "engine_commit": rec.run.engine_commit,
                                    "engine_modified": rec.run.engine_modified, "why": why})
    records = kept
    refused_runs = {r["run"]: r["why"] for r in refused_records}
    by_run = {r.run.id: r for r in records}

    def judged_by_record(rec: Any) -> dict[str, Any] | None:
        # A plan on a fixture whose result is refused: no replay judges it. A run never exported
        # has one only where :func:`unexported_refusal` says so.
        kind = fixture_of(prereg, rec.dataset.id)
        if not kind or rec.result.refused is None or not (
                rec.export.sha256 or unexported_refusal(rec)):
            return None
        verdict, why = result_refusal(rec.result.refused_kind, rec.result.refused)
        return {"run": rec.run.id, "dataset": rec.dataset.id, "kind": kind, "verdict": verdict,
                "why": why}

    def plan_of(rec: Any, given: list[Mapping[str, Any]]) -> dict[str, Any]:
        # The record's plan as its replays judge it: one replay, its own (:func:`paired`).
        mine = {"run": rec.run.id, "dataset": rec.dataset.id,
                "kind": fixture_of(prereg, rec.dataset.id), "verdict": None, "why": None}
        if len(given) > 1:
            return mine | {"verdict": "unjudged",
                           "why": (f"{len(given)} replays are given for this run, so which is "
                                   f"its plan's cannot be said.")}
        why = paired(given[0], rec, prereg)
        if why is not None:
            return mine | {"verdict": "invalid", "why": why}
        return score_replay(given[0], prereg)

    given_for: dict[str, list[Mapping[str, Any]]] = {}
    for rp in replays:
        given_for.setdefault(str(rp.get("run")), []).append(rp)
    plans = []
    for run, given in given_for.items():
        rec = by_run.get(run)
        if rec is None:
            plans.append({"run": run, "dataset": given[0].get("dataset"),
                          "kind": fixture_of(prereg, given[0].get("dataset")),
                          "verdict": "invalid", "model": None, "persona": None,
                          "why": (f"The record of this run is refused: {refused_runs[run]}"
                                  if run in refused_runs else
                                  "No record of this run is given, so its replay is no plan's.")})
            continue
        s = judged_by_record(rec) or plan_of(rec, given)
        s.update(model=model_of(rec, prereg), persona=rec.run.persona)
        plans.append(s)
    replayed = set(given_for)
    for rec in records:
        kind = fixture_of(prereg, rec.dataset.id)
        if kind and (rec.export.sha256 or unexported_refusal(rec)) and rec.run.id not in replayed:
            plans.append((judged_by_record(rec) or {
                "run": rec.run.id, "dataset": rec.dataset.id, "kind": kind,
                "verdict": "unscored", "why": "No replay of this plan is given."})
                | {"model": model_of(rec, prereg), "persona": rec.run.persona})
    verdict_of = {p["run"]: p for p in plans}
    # A sweep not run on the frozen engine is refused (:func:`sweep_refusal`), listed with why,
    # and read for nothing.
    swept = [sweep_verdicts(s, prereg) for s in sweeps]
    refused_sweeps = [{k: s[k] for k in ("dataset", "plan", "engine_tree")} | {"why": s["refused"]}
                      for s in swept if s["refused"]]
    swept = [s for s in swept if not s["refused"]]
    # A dataset's material decisions: the union over its sweeps; none where prereg.json's rule
    # read no refit of any of them, so no decision there is counted material or not.
    union: dict[str, set[str] | None] = {}
    unread: dict[str, str] = {}
    for d in dict.fromkeys(str(s["dataset"]) for s in swept):
        mine = [s for s in swept if str(s["dataset"]) == d]
        if any(s["applied"] for s in mine):
            union[d] = set().union(*(s["material"] for s in mine))
        else:
            union[d] = None
            unread[d] = (f"prereg.json's {mine[0]['rule']} read no refit of its sweeps: "
                         + " ".join(dict.fromkeys(str(s["not_applied"]) for s in mine)))
    reach = {r.run.id: reach_of(r, union.get(r.dataset.id) or set(), prereg) for r in records}

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
    # The band's miss rate is a false alarm's chance for a plan covering at exactly 95% only, so
    # over the plans judged on coverage it bounds the false alarms expected, and a count outside
    # the band is reported as that.
    est_plans = [p for p in plans if p["kind"] == "estimate" and p["verdict"] in SCORED]
    finding = []
    for s in swept:
        u = union.get(str(s["dataset"])) or set()
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
        "refused_records": refused_records,
        "plans": plans,
        "models": models,
        "decision": decision(models, prereg),
        "persona": persona_rule(records, reach, prereg),
        "early_stopping": early_stopping(estimate_units, records, prereg, budget_fresh,
                                         refused),
        "sweeps": swept,
        "refused_sweeps": refused_sweeps,
        "named_plans": named_plans(records, prereg),
        "material": {d: (sorted(u) if u is not None else None) for d, u in union.items()},
        "material_unread": unread,
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
                         "expected_at_most": len(est_plans)
                         * float(false_alarms["coverage_band"]["per_plan"]),
                         "outside_band": sum(1 for p in est_plans if not p["coverage_within"])}},
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


def _read(x: bool | None) -> str:
    """A materiality verdict in words: ``yes``, ``no``, or ``not read`` where none was read."""
    return "not read" if x is None else "yes" if x else "no"


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
    if report.get("refused_records"):
        lines += ["", "## Records refused", "",
                  f"{len(report['refused_records'])} record(s) are not known to be made on the "
                  f"frozen engine, so none of them is scored:", ""]
        lines += [f"- {r['run']}: {r['why']}" for r in report["refused_records"]]
    lines += ["", "## The plans", "",
              "| Run | Model | Fixture | Verdict | Mean bias | Covered | Regret | Why |",
              "|---|---|---|---|---|---|---|---|"]
    for p in report["plans"]:
        lines.append(f"| {p['run']} | {p.get('model') or 'none'} | {p['kind']} | {p['verdict']} "
                     f"| {_n(p.get('mean_bias'))} | {_n(p.get('covered'))} | "
                     f"{_n(p.get('regret'))} | {p.get('why') or ''} |")
    lines += ["", "## Materiality: prereg.json against each sweep's own verdict", ""]
    for s in report["sweeps"]:
        if s["applied"]:
            counted = (f"{len(s['material'])} material, {len(s['not_read'])} not read"
                       + (" (" + "; ".join(f"`{x['point']}`: {str(x['why']).rstrip('.')}"
                                           for x in s["not_read"][:10]) + ")"
                          if s["not_read"] else ""))
        else:
            counted = (f"prereg.json's {s['rule']} was applied to no refit "
                       f"({str(s['not_applied']).rstrip('.')}), so no decision is counted "
                       f"material or not")
        lines.append(f"- {s['dataset']} around the {s['plan']} plan: {counted}; "
                     f"{len(s['disagreements'])} refit(s) where the two verdicts differ"
                     + (": " + "; ".join(f"`{x['refit']}` (sweep {_read(x['sweep'])}, "
                                         f"prereg.json {_read(x['prereg'])})"
                                         for x in s["disagreements"][:10])
                        if s["disagreements"] else "") + ".")
    if report.get("refused_sweeps"):
        lines += ["", "## Sweeps refused", "",
                  f"{len(report['refused_sweeps'])} sweep(s) are not known to have run on the "
                  f"frozen engine, so none of them is read:", ""]
        lines += [f"- {s['dataset']} around the {s['plan']} plan: {s['why']}"
                  for s in report["refused_sweeps"]]
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
              f"tokens"
              + (f", {_n(e['refused_fresh'])} of them on {e['refused_runs']} record(s) refused "
                 f"for their engine" if e.get("refused_runs") else "")
              + f"; projected {_n(e['projected'])} against a budget of {_n(e['budget'])}: "
              + ("the waves stop." if e["stop_waves"] else "the waves go on."
                 if e["stop_waves"] is False else f"not judged ({e['why']})"), ""]
    f = report["false_alarms"]
    lines += ["## Expected false alarms", "",
              f"Decoy runs: {f['decoy']['runs']}, {_n(f['decoy']['expected'])} false alarm(s) "
              f"expected, {f['decoy']['observed']} observed. {f['coverage']['plans']} Estimate "
              f"plan(s) were judged on coverage. If each covered at exactly "
              f"{fx['coverage']['level']:.0%}, {_n(f['coverage']['expected_at_most'])} of them "
              f"would be expected to fall outside the band {fx['coverage']['low']} to "
              f"{fx['coverage']['high']} by chance (false alarms); "
              f"{f['coverage']['outside_band']} fell outside it.", ""]
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
    family with (Estimate) the exposure's row and value, (measurement) every feature's; where the
    record's result is refused, its kind and words (``result_refused``) and nothing to read."""
    kind = fixture_of(prereg, rec.dataset.id)
    if rec.result.refused is not None:
        return {"kind": kind, "family": rec.result.family, "rows": [],
                "result_refused": {"kind": rec.result.refused_kind,
                                   "words": rec.result.refused}}
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
                evaluation: Any = None, keep: Path | None = None,
                engine: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The final plan in ``bundle`` replayed on each pre-registered fresh draw of its fixture's
    generator (``core/export/replay.py``), read as ``target`` says (:func:`target_of`); a draw the
    replay or the scorer refuses is kept with its reason. Before any draw, a record whose result
    is refused is replayed as a whole as :func:`result_refusal` classes it (``refused`` or
    ``unjudged``, with its words), and a reported number the bundle does not hold once as
    ``unjudged``. The replay names its bundle's SHA-256, the generator and (Predict) the
    evaluation rows it read, and the engine it runs on (``engine``, else
    ``launcher.engine_now()``: the commit, the ``turbotab/`` tree, whether a tracked file under it
    is modified). ``generator``, ``seeds`` and ``evaluation`` (the Predict fixture's
    rows) stand in for prereg.json's in a test, and are named so; ``keep`` keeps each Predict
    draw's rebuilt home in ``keep/<seed>`` (its ``project`` in the draw), which is otherwise
    removed once scored."""
    from turbotab.core.export import replay, scorer

    from research.swarm import generators as G

    from research.swarm import launcher

    kind = target["kind"]
    fx = prereg[FIXTURE_KEY[kind]]
    first, last = prereg["replays"]["seeds"]
    seeds = list(seeds) if seeds is not None else list(range(first, last + 1))
    named = str(fx["generator"]) if generator is None else (
        f"given: {getattr(generator, '__module__', '?')}:"
        f"{getattr(generator, '__qualname__', repr(generator))}")
    generator = generator or fx["generator"]
    out: dict[str, Any] = {"run": run, "dataset": dataset, "generator": named,
                           "recorded_seed": fx["recorded_seed"], "target": dict(target),
                           "bundle_sha256": P._sha256(Path(bundle)),
                           "engine": dict(engine if engine is not None else launcher.engine_now()),
                           "draws": []}
    if kind == "predict":
        out["evaluation"] = (dict(prereg["replays"]["predict_evaluation"]) if evaluation is None
                             else {"given": True, "rows": len(evaluation)})
    no_result = target.get("result_refused")
    if no_result:
        verdict, why = result_refusal(no_result.get("kind"), str(no_result.get("words")))
        return out | {verdict: why}
    where: tuple[str, str, str] | None = None
    if kind != "predict":
        rows = target.get("rows") or []
        if not rows:
            return out | {"unjudged": "The record reports no estimate row to replay."}
        first_row = rows[0]
        where = locate(_bundle_estimates(Path(bundle)), str(target["family"]),
                       [str(first_row["feature"]), str(first_row["setting"])],
                       float(first_row["estimate"]))
        if where is None:
            return out | {"unjudged": ("The bundle's reported numbers hold no one row equal to "
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
        sha = P._sha256(args.bundle)
        if not rec.export.sha256 or sha != rec.export.sha256:
            print(json.dumps({"refused": (f"The bundle's SHA-256 is {sha}, not the record's "
                                          f"export {rec.export.sha256}: it is not this run's "
                                          f"plan.")}))
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
           "locate", "main", "make_replay", "model_of", "named_plans", "paired", "persona_rule",
           "point_of", "provenance", "reach_of", "regression_checks", "report_md", "robustness",
           "rule_of", "score", "score_replay", "sweep_verdicts", "target_of", "unit"]


if __name__ == "__main__":
    sys.exit(main())
