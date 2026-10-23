"""The pilot's pre-registration (SWARM_PLAN §7): ``research/swarm/prereg.json``, written by this
script and applied by ``research/swarm/score_pilot.py``, never by the orchestrator and never by
hand.

    python -m research.swarm.prereg_fill            # write prereg.json from the generators
    python -m research.swarm.prereg_fill --check    # fail where prereg.json is not what they give
    python -m research.swarm.prereg_fill crash <crash.json> … [--r0-class C --ruling WORDS]
    python -m research.swarm.prereg_fill prices haiku=P sonnet=P opus=P --source WORDS
    python -m research.swarm.prereg_fill ready      # whether the trio may start

**Every margin is computed from the generators** (``research/swarm/generators.py``), never from a
draw a plan is graded on:

* **b0** (:func:`compute`'s ``estimate_fixture``): half the smallest population bias the Estimate
  generator plants (``generators.planted_biases``, in closed form), with the four-standard-error
  condition's inputs from the generator's own tuning run (``generators.b0_tuning``: an analysis
  independent of the engine on seeds 1 to 100, none a graded draw);
* **the coverage band** (:func:`binomial_band`): the counts of 95% intervals covering the truth
  that a plan with exactly 95% coverage reaches with each tail's chance at most 2.5%, at the
  number of draws a plan is replayed on, with the band's own miss rate, in exact arithmetic;
* **r0** (:func:`r0_of`): half the population regret (``generators.readmission_regrets``) of the
  model class the default path's final plan falls in on the Predict fixture. The default path
  (``research/swarm/arms.py``) fits the shelf's first two predicting families there and declares
  final the first, so the rule reads that family, the one §7's replay grades ("each final plan
  is replayed"). Which family it is, and which readings and forms the plan holds, is read from
  the crash sweep's default plan on the frozen engine (``crash``); r0 waits for it. A class is
  read only where the generator computes it: a straight-line logistic family (its declared
  ``curve_shape`` and its yes/no estimator), every predictor read as an amount, no form recorded
  and no repair but a two-level text's 0/1 coding is the additive straight-line class; any other
  plan's r0 is refused with why, unless the orchestrator rules a class (``--r0-class``, with its
  words), which is then recorded as a ruling;
* **the measurement fixture's materiality k** (:func:`measurement_oracle`): half the discovery
  set's own draw-to-draw change, as §7 makes a change material at half the result's own interval
  half-width: the mean size of the symmetric difference between the discovery sets (q < the
  feature-wise family's FDR) of the analysis of the true concentrations on two independent
  draws, over the tuning seeds in pairs, half of it rounded up, at least 1. That analysis (each
  feature's Student t-test of its true log2 concentration between cases and controls, the
  participants only, Benjamini–Hochberg) is also the oracle the plans' power is reported against;
* **the graded menus** (:func:`graded_menus`): the engine's own menus
  (``generators.graded_measurement_decisions``), each option graded a choice, or the engine's
  refusal where its contract's rung under inference is ``refused`` (outcome-protected ComBat,
  QRILC and missing-at-random below the detection limit).

**Left to the crash sweep** (:func:`fill`, from package 3's ``crash.json`` on every pilot
dataset, each passed on one engine tree: ``launcher.crash_passed``):

* **each dataset's fit budget**: as many fits as the crash sweep made there (every option of the
  default plan fitted once), and as many seconds as those fits take at its slowest fit
  (``launcher.budget_of``);
* **the sweep's compute cap**: per dataset, the plans §6 names to sweep beyond the default (whose
  fits the crash sweep already paid) times the crash sweep's own fit seconds there, summed;
* the default plan on the Predict fixture, from which r0 is read.

**Set before the scoring** (:func:`set_prices`): each model's list price relative to the others',
a fact rather than a choice, which the decision rule's "the cheaper" reads. The ``filled`` section
holds these inputs, and ``--check`` reads them as held.

:func:`ready` says why the trio may not start (a pilot dataset without its fit budget, no compute
cap, no r0, reports on different engines or not the engine now, or margins that are no longer the
generators'), or None; the launcher calls it before the trio.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.swarm import generators as G

HERE = Path(__file__).resolve().parent
PREREG = HERE / "prereg.json"
VERSION = 1

# SWARM_PLAN §5's six datasets, the three fixtures among them, and the decoy (mechanical only).
ESTIMATE_FIXTURE = "walking_bp_estimate"
PREDICT_FIXTURE = "inpatient_readmission_predict"
MEASUREMENT_FIXTURE = "plasma_panel_estimate"
PILOT = ("bang2009", "sullivan2021", "nhanes_glucose_predict", PREDICT_FIXTURE, ESTIMATE_FIXTURE,
         MEASUREMENT_FIXTURE)
DECOY = "outpatient_admission_predict"
PERSONAS = ("conventional", "skeptical", "explorer")
# §6's agent runs: a record's model is the key its ``run.analyst_model`` names (lower case); a
# mechanical arm's model starts with ``mechanical-`` (``arms.run_arm``).
MODELS = ("haiku", "sonnet", "opus")
MECHANICAL = "mechanical-"
SCHEDULE: tuple[dict[str, Any], ...] = (
    {"model": "haiku", "personas": list(PERSONAS), "datasets": list(PILOT), "runs": 1},
    {"model": "sonnet", "personas": list(PERSONAS), "datasets": list(PILOT), "runs": 1},
    {"model": "haiku", "personas": list(PERSONAS), "datasets": [ESTIMATE_FIXTURE], "runs": 2,
     "replicates": True},
    {"model": "opus", "personas": ["conventional"], "datasets": [ESTIMATE_FIXTURE], "runs": 3,
     "reference": True},
)

# The replays (§7): every final plan on the same fresh draws of its fixture's generator, none a
# tuning draw (seeds 1 to 100), a regret draw (1, 2) or a committed table's seed.
REPLAY_DRAWS = 100
REPLAY_FIRST_SEED = 1001
# The Predict fixture's regret is read on these rows of its generator (their true risk known).
EVALUATION_SEED, EVALUATION_ROWS = 2001, 20_000
LEVEL = Fraction(19, 20)  # §7: a 95% interval
HALF = Fraction(1, 2)  # §7: half the planted bias, half the default's regret, half a half-width
B0_SE_RATIO = G.B0_SE_RATIO
UNIT_SHARE = Fraction(2, 3)  # §7's unit: at least two-thirds of the non-refused plans
UNIT_MIN = 2  # ... and at least two of them
FAITHFUL = Fraction(4, 5)  # §7's decision rule
EXPORTED = Fraction(9, 10)
REACH_MIN = 3  # §3: reach is reported only where the union holds at least three decisions

# Each graded measurement decision's method contract in the engine (``core/contracts.py``).
CONTRACT_OF = {"normalization and the log scale": "omics_normalization",
               "batch handling": "batch",
               "drift and batch correction from the pooled QCs": "qc_rlsc",
               "detection-limit handling": "detection_limit"}
# The kinds that record a predictor's functional form (``core/decisions.py``).
FORM_KINDS = ("set_exposure_form", "set_forms")
CODING_REPAIR = "binary_text__"  # a two-level text recoded 0/1, as the linear class codes sex


def _f(x: Fraction | float) -> float:
    return float(x)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── the coverage band ────────────────────────────────────────────────────────


def binomial_band(n: int, level: Fraction = LEVEL) -> dict[str, Any]:
    """The counts of ``n`` intervals of nominal ``level`` covering the truth that a plan with
    exactly that coverage reaches with each tail's chance at most (1 − level)/2: the lowest count
    ``low`` with P(X < low) ≤ the tail, the highest ``high`` with P(X > high) ≤ the tail, X
    binomial(n, level), and the band's miss rate P(X < low) + P(X > high), each in exact
    arithmetic."""
    p = Fraction(level)
    tail = (1 - p) / 2
    pmf = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
    below = [sum(pmf[:x], Fraction(0)) for x in range(n + 2)]  # below[x] = P(X < x)
    low = max(x for x in range(n + 1) if below[x] <= tail)
    high = min(x for x in range(n + 1) if 1 - below[x + 1] <= tail)
    miss = below[low] + (1 - below[high + 1])
    return {"draws": n, "level": _f(p), "tail": _f(tail), "low": low, "high": high,
            "miss_rate": _f(miss)}


def in_band(covered: int, band: Mapping[str, Any]) -> bool:
    return int(band["low"]) <= int(covered) <= int(band["high"])


# ── the measurement fixture ──────────────────────────────────────────────────


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (q-values), in ``p``'s order."""
    m = len(p)
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def latent_discoveries(seed: int, q: float) -> set[str]:
    """The features the analysis of the true concentrations discovers on the panel's draw at
    ``seed``: each feature's Student t-test (pooled variance) of the participants' true log2
    concentration between cases and controls, Benjamini–Hochberg q-values below ``q``."""
    from scipy import stats

    names = G.panel_design()["names"]
    lat = G.panel_latent(seed)
    qc = lat["layout"]["qc"].to_numpy()
    case = lat["case"][~qc]
    x = lat["true_log2"][~qc]
    a, b = x[case == 1], x[case == 0]
    na, nb = len(a), len(b)
    pooled = ((na - 1) * a.var(axis=0, ddof=1) + (nb - 1) * b.var(axis=0, ddof=1)) / (na + nb - 2)
    t = (a.mean(axis=0) - b.mean(axis=0)) / np.sqrt(pooled * (1 / na + 1 / nb))
    p = 2 * stats.t.sf(np.abs(t), na + nb - 2)
    return {names[j] for j in np.flatnonzero(_bh(p) < q)}


def measurement_oracle(draws: int = G.TUNING_DRAWS, first_seed: int = G.TUNING_FIRST_SEED
                       ) -> dict[str, Any]:
    """The analysis of the true concentrations on the tuning draws (:func:`latent_discoveries`):
    its mean power, false-discovery proportion and discoveries, the mean symmetric difference of
    its discovery sets on the draws in pairs (first and second, third and fourth, …), and k, half
    that rounded up, at least 1 (the module docstring)."""
    from turbotab.core.models.featurewise import FDR

    truth = set(G.panel_truth()["associated"])
    sets = [latent_discoveries(first_seed + i, FDR) for i in range(draws)]
    power = [len(s & truth) / len(truth) for s in sets]
    fdp = [len(s - truth) / len(s) if s else 0.0 for s in sets]
    pairs = [len(sets[i] ^ sets[i + 1]) for i in range(0, draws - 1, 2)]
    change = float(np.mean(pairs))
    return {"analysis": ("each feature's Student t-test of the participants' true log2 "
                         "concentration between cases and controls, Benjamini-Hochberg"),
            "q": FDR, "draws": draws, "seeds": [first_seed, first_seed + draws - 1],
            "associated": len(truth), "power": float(np.mean(power)),
            "fdp": float(np.mean(fdp)), "discoveries": float(np.mean([len(s) for s in sets])),
            "pairs": len(pairs), "symmetric_difference": change,
            "k": max(1, math.ceil(_f(HALF) * change))}


def graded_menus() -> list[dict[str, Any]]:
    """The measurement fixture's graded decisions (``generators.graded_measurement_decisions``),
    each option graded a choice, or the engine's refusal where its contract's rung under inference
    is ``refused``; an option its contract does not list is a choice with no rung."""
    from turbotab.core import contracts

    out = []
    for d in G.graded_measurement_decisions():
        contract = contracts.contract(CONTRACT_OF[d["decision"]])
        listed = {o.key: o for o in contract.options}
        options = []
        for key in d["options"]:
            rung = listed[key].rung["inference"] if key in listed else None
            options.append({"option": key, "rung": rung,
                            "graded": "refusal" if rung == "refused" else "choice"})
        out.append({**d, "contract": contract.key, "purpose": "inference", "options": options})
    return out


# ── r0 ───────────────────────────────────────────────────────────────────────


def default_plan_of(report: Mapping[str, Any]) -> dict[str, Any]:
    """The default plan a crash sweep's report holds (its points and the decisions it did not
    sweep): the families its models answer fits, in order, and every decision body it records."""
    decisions = [dict(p["decision"]) for p in [*(report.get("points") or []),
                                               *(report.get("not_swept") or [])]
                 if isinstance(p.get("decision"), Mapping)]
    models = next((d.get("models") for d in decisions if d.get("kind") == "select_models"), None)
    return {"dataset": report.get("dataset"), "families": list(models or []),
            "decisions": decisions}


def _straight_logistic(family: str) -> tuple[bool, str]:
    from sklearn.linear_model import LogisticRegression

    from turbotab.core.models.base import get_family

    fam = get_family(family)
    if fam.curve_shape != "straight":
        return False, (f"`{family}` is a {fam.curve_shape.replace('_', ' ')} learner, a class the "
                       f"generator computes no regret for")
    estimator = fam.build("binary", "prediction", G.READMISSION_ROWS, 10)
    if not isinstance(estimator, LogisticRegression):
        return False, f"`{family}`'s yes/no model is not a logistic regression"
    return True, f"`{family}` is a straight-line logistic regression"


def r0_of(default_plan: Mapping[str, Any] | None, regrets: Mapping[str, float],
          ruling: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """r0 from the default plan on the Predict fixture (the module docstring): its final family,
    the generator's class it falls in, and half that class's regret; refused (``value`` None)
    with why where no class is read; a ruling's class where the orchestrator ruled one."""
    out: dict[str, Any] = {"final": None, "class": None, "value": None, "why": None,
                           "ruling": None}
    reasons: list[str] = []
    if default_plan is None:
        reasons.append("No crash sweep of the Predict fixture has been filled in, so the default "
                       "plan's final family is not known.")
    elif not default_plan.get("families"):
        reasons.append("The crash sweep's default plan records no models answer.")
    else:
        final = str(default_plan["families"][0])
        out["final"] = final
        ok, said = _straight_logistic(final)
        if not ok:
            reasons.append(f"The default plan declares `{final}` final: {said}.")
        decisions = default_plan.get("decisions") or []
        numeric = set(G.SPLINE_COLUMNS) | {t["column"] for t in G.READMISSION_MODEL["terms"]}
        coded = sorted({str(i.get("column")) for d in decisions
                        for i in ([d] if d.get("kind") == "confirm_reading" else
                                  d.get("items") or [] if d.get("kind") == "confirm_readings"
                                  else [])
                        if i.get("reading") == "code_or_count" and i.get("value") == "code"
                        and i.get("column") in numeric})
        if coded:
            reasons.append(f"The default plan reads {', '.join(f'`{c}`' for c in coded)} as "
                           f"codes, which the generator's classes read as amounts.")
        forms = [d for d in decisions if d.get("kind") in FORM_KINDS]
        if forms:
            reasons.append(f"The default plan records {len(forms)} functional form answer(s), "
                           f"outside the generator's additive straight-line class.")
        repairs = sorted({str(d.get("finding_id")) for d in decisions
                          if d.get("kind") == "apply_repair"
                          and not str(d.get("finding_id")).startswith(CODING_REPAIR)})
        if repairs:
            reasons.append(f"The default plan repairs {', '.join(repairs)}, which the generator's "
                           f"classes do not.")
        if not reasons:
            out["class"] = "linear"
    if ruling is not None:
        cls = str(ruling["class"])
        out.update({"class": cls, "ruling": {"class": cls, "words": str(ruling["words"])},
                    "automatic": " ".join(reasons) or None})
        reasons = []
    if out["class"] is not None:
        out["value"] = _f(HALF) * float(regrets[out["class"]])
    else:
        out["why"] = " ".join(reasons)
    return out


# ── the pre-registration ─────────────────────────────────────────────────────


def replay_seeds() -> list[int]:
    return list(range(REPLAY_FIRST_SEED, REPLAY_FIRST_SEED + REPLAY_DRAWS))


def swept_plans() -> dict[str, list[dict[str, str]]]:
    """The final plans §6 names to sweep on each pilot dataset beyond the default plan: each
    model's first conventional-persona run there, and on the Estimate fixture each model's first
    explorer run too, by the order the runs started."""
    out: dict[str, list[dict[str, str]]] = {d: [] for d in PILOT}
    for d in PILOT:
        wanted = ["conventional"] + (["explorer"] if d == ESTIMATE_FIXTURE else [])
        for persona in wanted:
            for model in MODELS:
                runs = any(model == s["model"] and persona in s["personas"] and d in s["datasets"]
                           for s in SCHEDULE)
                if runs:
                    out[d].append({"model": model, "persona": persona, "run": "first started"})
    return out


def compute(filled: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every margin and rule of §7 from the generators (the module docstring); r0 from the default
    plan and the ruling the ``filled`` section holds."""
    from turbotab.core.models.featurewise import FDR

    from research.swarm import sweep

    planted = G.planted_biases()
    tuning = G.b0_tuning()
    band = binomial_band(REPLAY_DRAWS)
    regrets = G.readmission_regrets()
    oracle = measurement_oracle()
    filled = filled or {}
    r0 = r0_of(filled.get("default_plan"), regrets, filled.get("ruling"))
    seeds = replay_seeds()
    tuned = tuning["plans"]["oracle"]
    runs = sum(len(s["personas"]) * len(s["datasets"]) * s["runs"] for s in SCHEDULE)
    plans = swept_plans()
    return {
        "about": ("SWARM_PLAN §7's pre-registration, written by `python -m "
                  "research.swarm.prereg_fill` from research/swarm/generators.py and, in its "
                  "filled section, from package 3's crash sweep reports and the inputs its "
                  "commands record; applied by "
                  "research/swarm/score_pilot.py. Never edited by hand: `--check` fails where it "
                  "is not what the generators give."),
        "version": VERSION,
        "plan": "docs/turbotab-next/plans/SWARM_PLAN.md, version 5.1, §6 and §7",
        "generators_sha256": _sha256(Path(G.__file__)),
        "datasets": {"pilot": list(PILOT), "estimate_fixture": ESTIMATE_FIXTURE,
                     "predict_fixture": PREDICT_FIXTURE,
                     "measurement_fixture": MEASUREMENT_FIXTURE, "decoy": DECOY},
        "models": {"keys": list(MODELS), "match": ("a record's model is the key its "
                                                   "run.analyst_model names, in lower case"),
                   "mechanical": MECHANICAL, "personas": list(PERSONAS),
                   "schedule": [dict(s) for s in SCHEDULE], "runs": runs},
        "regression_checks": {
            "checks": ["schema", "seal_integrity", "reasons"],
            "rule": ("On every run: its record validates (research/swarm/record.py), the seal "
                     "check does not fail (research/swarm/checks.py), and every decision the "
                     "analyst recorded and every option it previewed carries a reason. They hold "
                     "by construction, so a run that fails one is a defect, reported by name.")},
        "material": {
            "half": _f(HALF),
            "predict": {"folds": sweep.FOLDS, "repeats": sweep.REPEATS,
                        "rule": (f"A refit is material where some family's signed change in the "
                                 f"primary score has a corrected repeated k-fold interval "
                                 f"excluding zero and is at least {_f(HALF):g} of the plan's own "
                                 f"interval half-width; paired over the same folds, else "
                                 f"unpaired with each arm's Nadeau-Bengio inflation, under the "
                                 f"same size rule.")},
            "estimate": {"rule": (f"A refit is material where some estimate shifts from the "
                                  f"plan's by at least {_f(HALF):g} of the plan's interval "
                                  f"half-width.")},
            "measurement": {"q": FDR, "k": oracle["k"],
                            "rule": (f"On the measurement fixture a refit is material where its "
                                     f"discovery set at q < {FDR:g} differs from the plan's by "
                                     f"at least {oracle['k']} feature(s) (the symmetric "
                                     f"difference), k being half the discovery set's own "
                                     f"draw-to-draw change rounded up "
                                     f"({oracle['symmetric_difference']:g} features)."),
                            "derivation": "measurement_fixture.oracle"},
            "decision": "A decision is material where any of its refits is.",
            "refused_refits": ("A refit the engine refuses has no score: it is reported beside "
                               "its decision's materiality as the sweep's refused share, kept "
                               "apart from the analysts' refusal rate.")},
        "replays": {
            "draws": REPLAY_DRAWS, "seeds": [seeds[0], seeds[-1]],
            "rule": (f"Every final plan on a fixture is replayed on the same {REPLAY_DRAWS} fresh "
                     f"draws of its generator, seeds {seeds[0]} to {seeds[-1]} "
                     f"(core/export/replay.py's fresh draw). A plan refused on any draw is a "
                     f"refused plan: it counts as neither within nor outside the margins, and goes "
                     f"into the refusal rate with each draw's reason."),
            "predict_evaluation": {"seed": EVALUATION_SEED, "rows": EVALUATION_ROWS}},
        "estimate_fixture": {
            "dataset": ESTIMATE_FIXTURE, "generator": "research.swarm.generators:blood_pressure",
            "recorded_seed": G.ESTIMATE_SEED, "effect": planted["effect"],
            "coefficients": planted["coefficients"], "bias": planted["bias"],
            "wrong_paths": list(G.WRONG_PATHS),
            "smallest_planted_bias": planted["smallest_planted_bias"], "b0": planted["b0"],
            "se_condition": {
                "ratio": B0_SE_RATIO, "draws": tuning["draws"], "seeds": tuning["seeds"],
                "imputations": tuning["imputations"], "oracle_mean_bias": tuned["mean_bias"],
                "oracle_sd": tuned["sd"], "oracle_se": tuned["se_of_mean_bias"],
                "b0_over_oracle_se": tuning["b0_over_oracle_se"], "holds": tuning["holds"],
                "oracle_within_b0": tuning["oracle_within_b0"],
                "wrong_paths_beyond_b0": tuning["wrong_paths_beyond_b0"],
                "oracle_coverage": tuning["oracle_coverage"],
                "oracle_covered_in_band": in_band(round(tuning["oracle_coverage"]
                                                        * tuning["draws"]),
                                                  binomial_band(tuning["draws"]))},
            "coverage": band,
            "rule": (f"A plan is within the margins where its mean bias over the draws is less "
                     f"than b0 = {planted['b0']:.4f} in size and the count of its 95% intervals "
                     f"holding the planted effect ({planted['effect']:g}) lies in the band "
                     f"{band['low']} to {band['high']} of {band['draws']}.")},
        "predict_fixture": {
            "dataset": PREDICT_FIXTURE, "generator": "research.swarm.generators:readmission",
            "recorded_seed": G.READMISSION_SEED,
            "regrets": {"classes": regrets, "rows": list(G.REGRET_ROWS),
                        "seeds": list(G.REGRET_SEEDS)},
            "r0": r0,
            "rule": ("A plan is within the margin where its mean Brier regret against the true "
                     "risk over the draws (its predictions on the evaluation rows) is less than "
                     "r0, half the population regret of the class the default path's final plan "
                     "falls in.")},
        "measurement_fixture": {
            "dataset": MEASUREMENT_FIXTURE,
            "generator": "research.swarm.generators:metabolite_panel",
            "recorded_seed": G.PANEL_SEED, "q": FDR, "associated": G.panel_truth()["associated"],
            "graded_menus": graded_menus(), "oracle": oracle,
            "reported": ["the false-discovery proportion against the nominal rate",
                         "power against the oracle analysis's",
                         "the discovered associated features' effect error on the log2 scale"],
            "rule": "Reported only: the measurement fixture decides nothing in the pilot."},
        "unit": {
            "share": [UNIT_SHARE.numerator, UNIT_SHARE.denominator], "minimum": UNIT_MIN,
            "rule": (f"A model is within the margins on a fixture where at least "
                     f"{UNIT_SHARE.numerator} in {UNIT_SHARE.denominator} of its non-refused plans "
                     f"there are within them, with at least {UNIT_MIN} non-refused plans; outside "
                     f"them with at least {UNIT_MIN} non-refused plans and fewer within; not "
                     f"judged with fewer, or while any of its plans there has no verdict (no "
                     f"replay, not the pre-registered draws, or a margin not fixed). A refused "
                     f"plan counts as neither.")},
        "comparisons": {
            "choosing_well": ("Agent plans against the default and random plans: mean bias and "
                              "coverage on the Estimate fixture, regret on the Predict fixture, "
                              "discovery quality on the measurement fixture, and the reported "
                              "held-out score or estimate on the real datasets."),
            "finding_what_matters": ("Each agent run's reach per million fresh tokens, against "
                                     "each sweep's reach per fit-hour.")},
        "reach": {"minimum_union": REACH_MIN,
                  "rule": (f"A run's reach is the share of its dataset's material decisions "
                           f"(the union over every sweep of the dataset) that it examined: an "
                           f"option other than its own previewed with a reason or fitted. Reported "
                           f"only where the union holds at least {REACH_MIN}.")},
        "statistics": {
            "refusals": ("Each model's refused plans as a share of its plans, and the refusals "
                         "the driver met per run, apart from the sweep's own refused share."),
            "cost_per_useful_result": ("Each model's fresh tokens over its useful results: runs "
                                       "that reached the export with no help and, on the "
                                       "Estimate or Predict fixture, a plan within the margins."),
            "cost": {"measure": ("fresh tokens per useful result (record run.spend.fresh), at "
                                 "the model's relative price"),
                     "relative_price": "filled.relative_price",
                     "why": ("A record's fresh tokens join input, cache creation and output, so "
                             "no price is computed from it; each model's fresh token is weighted "
                             "by its list price relative to the others' once `prereg_fill prices` "
                             "sets them, and until then no model is named the cheaper.")}},
        "decision_rule": {
            "candidates": ["haiku", "sonnet"], "reference": "opus",
            "reference_rule": ("Where the reference is outside the margins on the Estimate "
                               "fixture, the fixture is re-examined, not the models: no "
                               "production analyst is named."),
            "faithful": [FAITHFUL.numerator, FAITHFUL.denominator],
            "exported": [EXPORTED.numerator, EXPORTED.denominator],
            "rule": (f"A model is acceptable where it is within the margins on the Estimate "
                     f"fixture, the referees find its reasons faithful at least "
                     f"{FAITHFUL.numerator} times in {FAITHFUL.denominator}, and at least "
                     f"{EXPORTED.numerator} of its runs in {EXPORTED.denominator} reach the "
                     f"export with no help, a stuck run counted against it only where a blind "
                     f"reader classed it an analyst failure. Of acceptable models the cheaper, by "
                     f"fresh tokens per useful result at its relative price, is the production "
                     f"analyst; with none acceptable the plan returns to Nolan.")},
        "persona_rule": {
            "rule": ("A persona is kept where the mean reach of its Haiku runs on the Estimate "
                     "fixture differs from the conventional persona's by more than the larger of "
                     "the two personas' ranges over those runs (the first and its replicates); "
                     "deferred where either has fewer than two runs with a reach."),
            "model": "haiku", "dataset": ESTIMATE_FIXTURE},
        "expected_false_alarms": {
            "decoy": {"per_run": _f((1 - LEVEL) / 2), "level": _f(LEVEL),
                      "why": ("A decoy run fails where its held-out score's 95% interval lies "
                              "above the score of no skill (checks.null_decoy): one tail of the "
                              "interval.")},
            "coverage_band": {"per_plan": band["miss_rate"], "draws": band["draws"],
                              "among": ("the Estimate fixture's plans whose mean bias is within "
                                        "b0, for whom a coverage outside the band is a false "
                                        "alarm")},
            "rule": ("A miss on a control is settled by fresh seeds, never by editing an "
                     "expectation.")},
        "early_stopping": {
            "rule": ("A model outside the margins on the Estimate fixture once its runs there "
                     "under every persona are scored stops; a spend projected past the budget "
                     "(each model's remaining scheduled runs at its mean fresh tokens per run so "
                     "far) stops the waves.")},
        "sweeps": {
            "plans": plans,
            "rule": ("Beyond the default plan (whose fits the crash sweep paid), each dataset's "
                     "sweeps are around the plans named here, chosen by when their runs started, "
                     "never by their results.")},
    }



# The inputs no generator gives, each written by its command and kept when the file is written
# again:
# the crash sweeps' reports and what is read from them (``crash``), the orchestrator's ruling of
# r0's class, and each model's list price relative to the others' (``prices``).
EMPTY_FILLED: dict[str, Any] = {"reports": {}, "fit_budget": None, "compute_cap": None,
                                "default_plan": None, "ruling": None, "relative_price": None}


def build(filled: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The whole pre-registration: :func:`compute` and its ``filled`` section."""
    section = copy.deepcopy(EMPTY_FILLED) | copy.deepcopy(dict(filled or {}))
    out = compute(section)
    out["filled"] = section
    return out


def check(prereg: Mapping[str, Any]) -> list[str]:
    """Where ``prereg`` is not what the generators give (its ``filled`` inputs read as held):
    each top-level part that differs, by name."""
    again = build(prereg.get("filled"))
    return [k for k in sorted(set(again) | set(prereg)) if k != "filled"
            and json.loads(json.dumps(again.get(k))) != json.loads(json.dumps(prereg.get(k)))]


# ── the crash sweep's fields ─────────────────────────────────────────────────


class Refused(Exception):
    """The fill does not fill: why, in words."""


def fill(prereg: Mapping[str, Any], reports: Mapping[str, Mapping[str, Any]], *,
         paths: Mapping[str, str] | None = None, engine: Mapping[str, Any] | None = None,
         ruling: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``prereg`` with the crash sweep's fields filled from ``reports`` (dataset → its crash.json),
    every pilot dataset's, each passed (``launcher.crash_passed`` on ``engine``) on one engine tree
    (the module docstring); ``paths`` names each report's file, for its SHA-256."""
    from research.swarm import launcher

    missing = [d for d in PILOT if d not in reports]
    if missing:
        raise Refused(f"No crash sweep report is given for {', '.join(missing)}.")
    trees = {}
    for d in PILOT:
        why = launcher.crash_passed(reports[d], d, engine)
        if why is not None:
            raise Refused(f"{d}: {why}")
        trees[d] = reports[d].get("engine_tree")
    if len(set(trees.values())) != 1:
        raise Refused("The crash sweeps ran on different engine trees: "
                      + ", ".join(f"{d} on {t}" for d, t in trees.items()) + ".")
    entries, budget, cap = {}, {}, {}
    plans = swept_plans()
    for d in PILOT:
        r = reports[d]
        timed = r.get("fit_seconds") or {}
        fits = int(timed.get("fits") or 0)
        try:
            budget[d] = launcher.budget_of(r, fits)
        except launcher.Refused as exc:
            raise Refused(f"{d}: {exc}") from None
        total = float(timed.get("total") or 0.0)
        cap[d] = {"plans": len(plans[d]), "crash_seconds": total,
                  "seconds": len(plans[d]) * total}
        entries[d] = {"commit": r.get("commit"), "engine_tree": r.get("engine_tree"),
                      "fits": fits, "total_seconds": total, "slowest_fit": timed.get("max"),
                      "sha256": (_sha256(Path(paths[d])) if paths and d in paths else None)}
    section = dict(prereg.get("filled") or {}) | {
        "reports": entries, "fit_budget": budget,
        "compute_cap": {"seconds": sum(c["seconds"] for c in cap.values()), "by_dataset": cap,
                        "rule": ("Per dataset, the plans named to sweep beyond the default times "
                                 "the crash sweep's own fit seconds there, summed; the sweeps of "
                                 "a dataset stop once they have spent its share.")},
        "default_plan": default_plan_of(reports[PREDICT_FIXTURE]),
        "ruling": dict(ruling) if ruling else None}
    return build(section)


def set_prices(prereg: Mapping[str, Any], prices: Mapping[str, float], source: str
               ) -> dict[str, Any]:
    """``prereg`` with each model's list price relative to the others' (a fact, not a choice:
    set before the scoring, from ``source``), every model's, each above zero."""
    missing = [m for m in MODELS if m not in prices]
    if missing or any(not float(prices[m]) > 0 for m in MODELS) or set(prices) - set(MODELS):
        raise Refused(f"A relative price is given for each of {', '.join(MODELS)}, each above "
                      f"zero, and for no other model.")
    section = dict(prereg.get("filled") or {}) | {
        "relative_price": {"prices": {m: float(prices[m]) for m in MODELS},
                           "source": str(source)}}
    return build(section)


def ready(prereg: Mapping[str, Any], engine: Mapping[str, Any] | None = None, *,
          margins: bool = True) -> str | None:
    """Why the trio may not start on ``prereg`` (the module docstring), or None. ``engine``: the
    engine now (``launcher.engine_now`` where None); ``margins``: also recompute the margins."""
    filled = prereg.get("filled") or {}
    budget = filled.get("fit_budget") or {}
    missing = [d for d in PILOT if d not in budget]
    if missing:
        return (f"prereg.json holds no fit budget for {', '.join(missing)}: fill it from the "
                f"crash sweeps (python -m research.swarm.prereg_fill crash …).")
    if not filled.get("compute_cap"):
        return "prereg.json holds no compute cap for the sweeps: fill it from the crash sweeps."
    r0 = (prereg.get("predict_fixture") or {}).get("r0") or {}
    if r0.get("value") is None:
        return f"prereg.json holds no r0: {r0.get('why')}"
    trees = {e.get("engine_tree") for e in (filled.get("reports") or {}).values()}
    if engine is None:
        from research.swarm import launcher

        engine = launcher.engine_now()
    if trees != {engine.get("engine_tree")} or engine.get("engine_modified"):
        return (f"The crash sweeps prereg.json was filled from ran on engine tree(s) "
                f"{', '.join(sorted(map(str, trees)))}, and the engine now is "
                f"{engine.get('engine_tree')}"
                + (" with modified files" if engine.get("engine_modified") else "") + ".")
    if margins:
        changed = check(prereg)
        if changed:
            return (f"prereg.json's {', '.join(changed)} are no longer what the generators give: "
                    f"write it again before the trio.")
    return None


# ── the files ────────────────────────────────────────────────────────────────


def load(path: Path = PREREG) -> dict[str, Any]:
    return json.loads(Path(path).read_text("utf-8"))


def write(prereg: Mapping[str, Any], path: Path = PREREG) -> Path:
    Path(path).write_text(json.dumps(prereg, indent=1, ensure_ascii=False) + "\n", "utf-8")
    return Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.prereg_fill",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--prereg", type=Path, default=PREREG)
    parser.add_argument("--check", action="store_true", help="fail where it is not the generators'")
    sub = parser.add_subparsers(dest="command")
    c = sub.add_parser("crash", help="fill the crash sweep's fields from its reports")
    c.add_argument("reports", type=Path, nargs="+", help="each pilot dataset's crash.json")
    c.add_argument("--r0-class", choices=("linear", "spline", "interaction"))
    c.add_argument("--ruling", help="the orchestrator's words for the r0 class it rules")
    p = sub.add_parser("prices", help="each model's list price relative to the others'")
    p.add_argument("prices", nargs="+", help="model=price, for each of " + ", ".join(MODELS))
    p.add_argument("--source", required=True, help="where the list prices were read")
    sub.add_parser("ready", help="whether the trio may start")
    args = parser.parse_args(argv)
    try:
        if args.command == "crash":
            if (args.r0_class is None) != (args.ruling is None):
                parser.error("--r0-class and --ruling go together")
            reports, paths = {}, {}
            for path in args.reports:
                r = json.loads(path.read_text("utf-8"))
                reports[str(r.get("dataset"))], paths[str(r.get("dataset"))] = r, str(path)
            ruling = {"class": args.r0_class, "words": args.ruling} if args.r0_class else None
            out = fill(load(args.prereg), reports, paths=paths, ruling=ruling)
            write(out, args.prereg)
            print(json.dumps({"filled": str(args.prereg), "r0": out["predict_fixture"]["r0"],
                              "ready": ready(out)}))
            return 0
        if args.command == "prices":
            given = dict(item.split("=", 1) for item in args.prices)
            out = set_prices(load(args.prereg), {k: float(v) for k, v in given.items()},
                             args.source)
            write(out, args.prereg)
            print(json.dumps({"filled": str(args.prereg),
                              "relative_price": out["filled"]["relative_price"]}))
            return 0
    except (Refused, ValueError) as exc:
        print(json.dumps({"refused": str(exc)}))
        return 1
    if args.command == "ready":
        why = ready(load(args.prereg))
        print(json.dumps({"ready": why is None, "why": why}))
        return 0 if why is None else 1
    if args.check:
        changed = check(load(args.prereg))
        for k in changed:
            print(f"prereg.json's {k} is not what the generators give")
        return 1 if changed else 0
    filled = load(args.prereg).get("filled") if args.prereg.exists() else None
    print(f"wrote {write(build(filled), args.prereg)}")
    return 0


__all__ = ["PREREG", "Refused", "binomial_band", "build", "check", "compute", "default_plan_of",
           "fill", "graded_menus", "in_band", "latent_discoveries", "load", "main",
           "measurement_oracle", "r0_of", "ready", "replay_seeds", "set_prices", "swept_plans",
           "write"]


if __name__ == "__main__":
    sys.exit(main())
