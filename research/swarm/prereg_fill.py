"""The pilot's pre-registration (SWARM_PLAN §7): ``research/swarm/prereg.json``, written by this
script and applied by ``research/swarm/score_pilot.py``, never by the orchestrator and never by
hand.

    python -m research.swarm.prereg_fill            # write prereg.json from the generators
    python -m research.swarm.prereg_fill --check    # fail where prereg.json is not what they give
    python -m research.swarm.prereg_fill crash <crash.json> …   # any pilot datasets' reports
    python -m research.swarm.prereg_fill r0 --record <default.json> --replay <replay.json>
    python -m research.swarm.prereg_fill prices haiku=P sonnet=P opus=P --source WORDS
    python -m research.swarm.prereg_fill ready [--dataset ID]   # whether it may launch

**Every margin is computed from the generators** (``research/swarm/generators.py``), never from a
draw a plan is graded on, but r0, which the orchestrator's ruling measures on the very draws and
rows every plan's regret is read on:

* **b0** (:func:`compute`'s ``estimate_fixture``): half the smallest population bias the Estimate
  generator plants (``generators.planted_biases``, in closed form), with the four-standard-error
  condition's inputs from the generator's own tuning run (``generators.b0_tuning``: an analysis
  independent of the engine on seeds 1 to 100, none a graded draw);
* **the coverage band** (:func:`binomial_band`): the counts of 95% intervals covering the truth
  that a plan with exactly 95% coverage reaches with each tail's chance at most 2.5%, at the
  number of draws a plan is replayed on, with the band's own miss rate, in exact arithmetic;
* **r0** (:func:`r0_of`, the orchestrator's ruling of 2026-10-23, :func:`r0_ruling`): measured,
  half the default plan's mean Brier regret against the true risk over the pre-registered replay
  draws on the pre-registered evaluation rows. The default path (``research/swarm/arms.py``) fits
  the shelf's first two predicting families on the Predict fixture and declares final the first,
  which §7's replay grades ("each final plan is replayed"); a family with no class the generator
  computes a regret for (a tree ensemble) leaves r0 nothing to read but the replay. So the default
  path's exported plan on the frozen engine is replayed by ``score_pilot replay`` (its
  ``make_replay``) on the draws and rows every plan is replayed on, read by ``score_pilot``'s own
  scorer (``score_replay``), and filled after the crash fill (``r0``, :func:`fill_r0`): r0 and
  every plan's regret come from the same draws, rows and scorer, and ``score_pilot`` judges no
  replay that is not on them, on that engine. The ruling names the family it was made for
  (:data:`RULED_FINAL`); a crash sweep whose default plan declares another final is refused, so the
  orchestrator rules again. Half the additive straight-line class's population regret
  (``generators.readmission_regrets``) is kept beside it as a reference, labeled so;
* **the measurement fixture's materiality k** (:func:`measurement_oracle`, the orchestrator's
  ruling of 2026-10-23, :func:`k_ruling`): the discovery set's own draw-to-draw change, rounded
  up: the mean size of the symmetric difference between the discovery sets (q < the feature-wise
  family's FDR) of the analysis of the true concentrations on two independent draws, over the
  tuning seeds in pairs, rounded up, at least 1. A change smaller than two independent draws of
  the same truth differ by is within noise. That analysis (each feature's Student t-test of its
  true log2 concentration between cases and controls, the participants only, Benjamini–Hochberg)
  is also the oracle the plans' power is reported against;
* **the graded menus** (:func:`graded_menus`): the engine's own menus
  (``generators.graded_measurement_decisions``), each option graded a choice, or the engine's
  refusal where its contract's rung under inference is ``refused`` (outcome-protected ComBat,
  QRILC and missing-at-random below the detection limit).

**Left to the crash sweep, dataset by dataset** (:func:`fill`, from package 3's ``crash.json`` on
any pilot datasets, each passed: ``launcher.crash_passed``, on the engine now; the orchestrator's
ruling of 2026-10-28, a staged pre-registration). Each fill is mechanical (a sweep's fits and
seconds), reads no graded run, adds to what is filled without dropping an earlier dataset's, and
is recorded with when it was filled (``filled_at``); every report filled is on one ``turbotab/``
tree; and a fill for a dataset that already has a graded run (a launch of it, ``launcher.
LAUNCHES``: :func:`graded_runs`) is refused, since what it fixes is fixed before that dataset's
first graded run:

* **each dataset's fit budget**: as many fits as the crash sweep made there (every option of the
  default plan fitted once), and as many seconds as those fits take at its slowest fit
  (``launcher.budget_of``);
* **the sweep's compute cap**, once every pilot dataset's is filled (it is needed only before the
  pilot's sweep step): per dataset, the plans §6 names to sweep beyond the default (whose fits the
  crash sweep already paid) times the crash sweep's own fit seconds there, summed;
* the default plan on the Predict fixture, from its own crash sweep: its final family, which the
  replay r0 is measured on must read, and which must be the family r0's ruling was made for
  (:data:`RULED_FINAL`).

**Measured after the Predict fixture's crash fill** (:func:`fill_r0`, refused once the Predict
fixture has a graded run): r0, from the default path's record on the
Predict fixture and its replay, each checked: the record the default path's
(``mechanical-default``) on the Predict fixture, exported, its final family the crash sweep's
default plan's, made on the frozen engine (a crash sweep's commit, or a commit whose ``turbotab/``
tree is the crash sweeps', :func:`engine_tree_of`; no tracked file modified); the replay of that
run's bundle (its SHA-256 the record's export's), reading that family, from the fixture's
generator at its recorded seed, on the pre-registered draws and evaluation rows, run on the frozen
engine (``score_pilot.provenance``), refused on no draw. A crash fill of the Predict fixture
clears it, since it reads that crash sweep's default plan; a fill of another dataset keeps it.

**Set before the scoring** (:func:`set_prices`): each model's list price relative to the others',
a fact rather than a choice, which the decision rule's "the cheaper" reads. The ``filled`` section
holds these inputs, and ``--check`` reads them as held.

:func:`ready` with a dataset says why that dataset may not launch, or None: it is not a pilot
dataset; its own fit budget is not filled from its own crash sweep; on the Predict fixture, r0 is
not measured, or not on the crash sweeps' engine; the reports filled are not on one engine tree,
or not on the engine now with no file modified; or the margins and rules (b0, the coverage band,
k, the r0 rule, the prices' rule, the materiality rules) are no longer the generators'. With no
dataset it says why the pilot is not filled whole, as its sweep step needs: every pilot dataset's
fit budget, the sweeps' compute cap and r0, on the same terms. The launcher asks it of each
dataset before each launch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.swarm import generators as G

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PREREG = HERE / "prereg.json"
VERSION = 2  # 2: Bang 2009 deferred, and the pre-registration filled dataset by dataset

# SWARM_PLAN §5's five datasets, the three fixtures among them, and the decoy (mechanical only).
# Bang 2009 is deferred to the next wave (the owner and the orchestrator, 2026-10-28: its default
# plan's fit is too slow for the pilot's daytime windows); its intake stays, and the glucose
# benchmark still reads its published score.
ESTIMATE_FIXTURE = "walking_bp_estimate"
PREDICT_FIXTURE = "inpatient_readmission_predict"
MEASUREMENT_FIXTURE = "plasma_panel_estimate"
PILOT = ("sullivan2021", "nhanes_glucose_predict", PREDICT_FIXTURE, ESTIMATE_FIXTURE,
         MEASUREMENT_FIXTURE)
DECOY = "outpatient_admission_predict"
PERSONAS = ("conventional", "skeptical", "explorer")
# §6's agent runs: a record's model is the key its ``run.analyst_model`` names (lower case); a
# mechanical arm's model starts with ``mechanical-`` (``arms.run_arm``).
MODELS = ("haiku", "sonnet", "opus")
MECHANICAL = "mechanical-"
DEFAULT_MODEL = MECHANICAL + "default"  # the default path's run (``arms.run_arm``)
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
# The orchestrator's rulings, each recorded in prereg.json with its words and date for the owner
# to review.
RULED_BY = "the orchestrator, the methods lead"
RULED_ON = "2026-10-23"
REVIEW = "for the owner to review"
# The family r0's ruling was made for: the default plan's final on the Predict fixture, a family
# with no class the generator computes a regret for. A crash sweep whose default plan declares
# another is refused (:func:`fill`, :func:`fill_r0`), since the ruling's premise is then not known
# to hold.
RULED_FINAL = "boosted_trees"


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
    its discovery sets on the draws in pairs (first and second, third and fourth, …), and k, that
    rounded up, at least 1 (the module docstring, :func:`k_ruling`)."""
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
            "k": max(1, math.ceil(change))}


def k_ruling(oracle: Mapping[str, Any]) -> dict[str, str]:
    """The orchestrator's ruling of k (the module docstring), its numbers ``oracle``'s."""
    return {"by": RULED_BY, "date": RULED_ON, "review": REVIEW,
            "words": (f"The measurement fixture's materiality k is the mean draw-to-draw "
                      f"symmetric difference of the latent oracle's discovery set, rounded up "
                      f"({oracle['k']}, from {oracle['symmetric_difference']:g}): a change "
                      f"smaller than two independent draws of the same truth differ by is within "
                      f"noise.")}


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


def r0_ruling() -> dict[str, str]:
    """The orchestrator's ruling of r0 (the module docstring), its numbers the pre-registered
    draws' and rows'."""
    seeds = replay_seeds()
    return {"by": RULED_BY, "date": RULED_ON, "review": REVIEW,
            "words": (f"The default path on the Predict fixture declares {RULED_FINAL} final, "
                      f"which has no generator class, so r0 is measured: half the default plan's "
                      f"mean Brier regret against the generator's oracle over the pre-registered "
                      f"replay draws (seeds {seeds[0]}-{seeds[-1]}) on the pre-registered "
                      f"evaluation rows (seed {EVALUATION_SEED} x {EVALUATION_ROWS:,}), computed "
                      f"by score_pilot's own replay producer after the Predict fixture's crash "
                      f"fill and before its first graded run (staged, 2026-10-28), so r0 and "
                      f"every plan's regret come from the same draws, rows and "
                      f"scorer. The linear class's value stays in prereg.json as a reference, "
                      f"labeled so.")}


def r0_of(default_plan: Mapping[str, Any] | None, measured: Mapping[str, Any] | None,
          regrets: Mapping[str, float]) -> dict[str, Any]:
    """r0 (the module docstring): half the default plan's mean regret ``measured`` (the filled
    section's, :func:`fill_r0`), the final family the crash sweep's ``default_plan`` declares,
    and the straight-line class's reference; with no value, why."""
    out: dict[str, Any] = {
        "final": (list((default_plan or {}).get("families") or []) or [None])[0],
        "value": None, "default_regret": None, "why": None,
        "reference": {"class": "linear", "value": _f(HALF) * float(regrets["linear"]),
                      "label": ("A reference only, not r0: half the population regret of the "
                                "generator's additive straight-line class "
                                "(generators.readmission_regrets), whatever the default plan's "
                                "final family.")},
        "ruling": r0_ruling()}
    if default_plan is None:
        out["why"] = ("No crash sweep of the Predict fixture has been filled in, so the default "
                      "plan is not known; r0 is measured after it (python -m "
                      "research.swarm.prereg_fill r0 …).")
    elif measured is None:
        out["why"] = ("The default plan's regret on the pre-registered draws has not been "
                      "measured: replay its export (python -m research.swarm.score_pilot replay "
                      "…) and fill it (python -m research.swarm.prereg_fill r0 …).")
    else:
        regret = float(measured["mean_regret"])
        out.update(default_regret=regret, value=_f(HALF) * regret)
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
    plan and its measured regret the ``filled`` section holds."""
    from turbotab.core.models.featurewise import FDR

    from research.swarm import sweep

    planted = G.planted_biases()
    tuning = G.b0_tuning()
    band = binomial_band(REPLAY_DRAWS)
    regrets = G.readmission_regrets()
    oracle = measurement_oracle()
    filled = filled or {}
    r0 = r0_of(filled.get("default_plan"), filled.get("r0"), regrets)
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
        "plan": "docs/turbotab-next/plans/SWARM_PLAN.md, version 5.2, §6 and §7",
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
                                     f"difference), k being the discovery set's own mean "
                                     f"draw-to-draw change rounded up "
                                     f"({oracle['symmetric_difference']:g} features), each set "
                                     f"read from the q-values the sweep's estimates carry; where "
                                     f"they carry none the rule is not applied, and no decision "
                                     f"there is counted material or not."),
                            "derivation": "measurement_fixture.oracle",
                            "ruling": k_ruling(oracle)},
            "decision": ("A decision is material where any of its refits read is, not "
                         "material where none of them is, and not read where none of its refits "
                         "is read (with why), counted apart."),
            "refused_refits": ("A refit the engine refuses has no score: it is reported beside "
                               "its decision's materiality as the sweep's refused share, kept "
                               "apart from the analysts' refusal rate.")},
        "replays": {
            "draws": REPLAY_DRAWS, "seeds": [seeds[0], seeds[-1]],
            "rule": (f"Every final plan on a fixture is replayed on the same {REPLAY_DRAWS} fresh "
                     f"draws of its generator, seeds {seeds[0]} to {seeds[-1]} "
                     f"(core/export/replay.py's fresh draw). A replay is its plan's only where "
                     f"its dataset, its bundle (the record's export, by SHA-256) and what it "
                     f"reads are its record's, it names the fixture's generator and recorded seed "
                     f"and, on the Predict fixture, the evaluation rows below, and it ran on the "
                     f"frozen engine (the crash sweeps' turbotab/ tree, no tracked file "
                     f"modified); else it is invalid, and two replays of one run leave its plan "
                     f"not judged. A plan refused on any draw is a "
                     f"refused plan, and so is a plan whose run reported no result (its record's "
                     f"result refused: none reported, or none served): it counts as neither "
                     f"within nor outside the margins, and goes into the refusal rate with its "
                     f"reasons. A plan whose record cannot say what the run reported, or whose "
                     f"reported number its bundle does not hold once, is not judged."),
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
                     "r0, half the default plan's mean regret on the same draws and rows.")},
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
                     f"replay, or two of one run; a replay not its record's plan, not on the "
                     f"pre-registered draws and rows, or not on the frozen engine; a record "
                     f"that cannot say what the run reported, a reported number its bundle does "
                     f"not hold once, or a margin not fixed). A refused plan counts as "
                     f"neither.")},
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
            "reference_rule": ("The reference checks the fixture itself, so no production "
                               "analyst is named until it is within the margins on the "
                               "Estimate fixture: outside them, the fixture is re-examined, not "
                               "the models; not judged there, or with no run, the check has not "
                               "run."),
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
                              "coverage": band["level"],
                              "among": (f"a plan whose intervals cover the planted effect at "
                                        f"exactly {band['level']:.0%}, for which a count outside "
                                        f"the band is a false alarm at this rate. A plan covering "
                                        f"more or less than that is not falsely alarmed when its "
                                        f"count falls outside the band, so among the plans judged "
                                        f"on coverage at most their number times this rate are "
                                        f"expected; a count outside the band is reported as "
                                        f"that, never as a false alarm.")},
            "rule": ("A miss on a control is settled by fresh seeds, never by editing an "
                     "expectation.")},
        "early_stopping": {
            "rule": ("A model outside the margins on the Estimate fixture once it has a record "
                     "there under every persona it runs (a run that ended, one stuck before its "
                     "export included) stops; a spend projected past the budget (each model's "
                     "remaining scheduled runs at its mean fresh tokens per run so far) stops the "
                     "waves.")},
        "sweeps": {
            "plans": plans,
            "rule": ("Beyond the default plan (whose fits the crash sweep paid), each dataset's "
                     "sweeps are around the plans named here, chosen by when their runs started, "
                     "never by their results.")},
    }



# The inputs no generator gives, each written by its command and kept when the file is written
# again: the crash sweeps' reports and what is read from them (``crash``), the default plan's
# regret on the pre-registered draws (``r0``), and each model's list price relative to the
# others' (``prices``).
EMPTY_FILLED: dict[str, Any] = {"reports": {}, "fit_budget": None, "compute_cap": None,
                                "default_plan": None, "r0": None, "relative_price": None}


def build(filled: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The whole pre-registration: :func:`compute` and its ``filled`` section (its inputs only,
    each :data:`EMPTY_FILLED` names)."""
    given = dict(filled or {})
    section = {k: copy.deepcopy(given.get(k, v)) for k, v in EMPTY_FILLED.items()}
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


def ruled_final_why(final: str | None) -> str:
    """Why a default plan whose final family is ``final`` is refused (:data:`RULED_FINAL`)."""
    return (f"The crash sweep's default plan on the Predict fixture declares `{final}` final, and "
            f"r0's ruling ({RULED_BY}, {RULED_ON}) was made for `{RULED_FINAL}`: its premise is "
            f"not known to hold for this plan, so the orchestrator rules again before this is "
            f"filled.")


def frozen_engine(prereg: Mapping[str, Any]) -> dict[str, Any] | None:
    """The engine the crash sweeps passed on (the module docstring): their one ``turbotab/`` tree
    and the commits they ran at, or None where none is filled in."""
    reports = ((prereg.get("filled") or {}).get("reports") or {}).values()
    trees = {e.get("engine_tree") for e in reports}
    if len(trees) != 1 or None in trees:
        return None
    return {"engine_tree": next(iter(trees)),
            "commits": sorted({str(e.get("commit")) for e in reports if e.get("commit")})}


def engine_tree_of(commit: str | None) -> str | None:
    """The tree id of ``turbotab/`` at ``commit`` in this checkout (``git rev-parse
    <commit>:turbotab``), or None where git does not resolve it."""
    if not commit:
        return None
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "--quiet",
                              f"{commit}:turbotab"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def record_engine_why(rec: Any, frozen: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """The ``turbotab/`` tree of the engine ``rec`` was made on, and why it is not the
    ``frozen`` one (a commit neither a crash sweep's nor one whose tree is theirs, or tracked
    files modified, or not known to be unmodified), else None."""
    commit = rec.run.engine_commit
    tree = frozen["engine_tree"] if commit in frozen["commits"] else engine_tree_of(commit)
    if tree != frozen["engine_tree"]:
        return tree, (f"The record was made on engine commit {commit} (its turbotab/ tree "
                      f"{tree}), and the crash sweeps passed on commit(s) "
                      f"{', '.join(frozen['commits'])}, tree {frozen['engine_tree']}: not the "
                      f"frozen engine.")
    if rec.run.engine_modified is not False:
        return tree, (f"The record was made on engine commit {commit} with tracked files "
                      + ("modified" if rec.run.engine_modified else "not known to be unmodified")
                      + ", so not on the frozen engine.")
    return tree, None


def now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# The launches folder a graded run is read from (``launcher.LAUNCHES``, where None); a test's
# conftest points it at an empty folder of its own, so no test reads the pilot's launches.
GRADED: Path | None = None


def graded_runs(dataset: str, launches: Path | None = None) -> list[str]:
    """The graded runs ``dataset`` has: each launch the launcher wrote of it (``launches``, else
    :data:`GRADED`, else ``launcher.LAUNCHES``: ``<run id>.json``, its ``dataset.id``), by run id.
    Only which dataset each launch is of is read, never what its run did."""
    from research.swarm import launcher

    folder = Path(launches if launches is not None else GRADED or launcher.LAUNCHES)
    out = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
        try:
            launch = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            out.append(path.stem)  # a launch that cannot be read may be of any dataset
            continue
        if not isinstance(launch, Mapping) or (launch.get("dataset") or {}).get("id") == dataset:
            out.append(str(launch.get("run") if isinstance(launch, Mapping) else path.stem))
    return out


def graded_why(dataset: str, runs: Sequence[str]) -> str:
    return (f"{dataset} has {len(runs)} graded run(s) already ({', '.join(runs[:3])}"
            f"{', ...' if len(runs) > 3 else ''}): what its fill fixes is fixed before its first "
            f"graded run, so it is not filled again.")


CAP_RULE = ("Per dataset, the plans named to sweep beyond the default times the crash sweep's own "
            "fit seconds there, summed; the sweeps of a dataset stop once they have spent its "
            "share.")


def compute_cap_of(entries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """The sweeps' compute cap (the module docstring) from the filled reports ``entries``, or None
    until every pilot dataset's is filled."""
    if any(d not in entries for d in PILOT):
        return None
    plans = swept_plans()
    cap = {d: {"plans": len(plans[d]), "crash_seconds": float(entries[d]["total_seconds"]),
               "seconds": len(plans[d]) * float(entries[d]["total_seconds"])} for d in PILOT}
    return {"seconds": sum(c["seconds"] for c in cap.values()), "by_dataset": cap,
            "rule": CAP_RULE}


def fill(prereg: Mapping[str, Any], reports: Mapping[str, Mapping[str, Any]], *,
         paths: Mapping[str, str] | None = None, engine: Mapping[str, Any] | None = None,
         launches: Path | None = None, at: str | None = None) -> dict[str, Any]:
    """``prereg`` with the crash sweep's fields filled from ``reports`` (dataset → its crash.json),
    any pilot datasets', each passed (``launcher.crash_passed`` on ``engine``) on the one engine
    tree every report filled is on, none of a dataset with a graded run (:func:`graded_runs` in
    ``launches``), each recorded with when it was filled (``at``, else now), added to what is
    filled (the module docstring); the compute cap once every pilot dataset's is filled; and,
    where the Predict fixture's report is among them, its default plan set and r0's measure
    cleared (:func:`fill_r0` reads that plan). ``paths`` names each report's file, for its
    SHA-256."""
    from research.swarm import launcher

    if not reports:
        raise Refused("No crash sweep report is given.")
    other = [d for d in reports if d not in PILOT]
    if other:
        raise Refused(f"{', '.join(map(str, other))} is not one of the pilot's datasets "
                      f"({', '.join(PILOT)}), so its crash sweep fills nothing.")
    for d in reports:
        runs = graded_runs(d, launches)
        if runs:
            raise Refused(graded_why(d, runs))
        why = launcher.crash_passed(reports[d], d, engine)
        if why is not None:
            raise Refused(f"{d}: {why}")
    filled = dict(prereg.get("filled") or {})
    held = {d: dict(e) for d, e in (filled.get("reports") or {}).items() if d not in reports}
    trees = {d: e.get("engine_tree") for d, e in held.items()} | {
        d: r.get("engine_tree") for d, r in reports.items()}
    if len(set(trees.values())) != 1:
        raise Refused("The crash sweeps ran on different engine trees: "
                      + ", ".join(f"{d} on {t}" for d, t in trees.items()) + ".")
    default, r0 = filled.get("default_plan"), filled.get("r0")
    if PREDICT_FIXTURE in reports:
        default, r0 = default_plan_of(reports[PREDICT_FIXTURE]), None
        final = (default["families"] or [None])[0]
        if final != RULED_FINAL:
            raise Refused(ruled_final_why(final))
    budget = {d: b for d, b in (filled.get("fit_budget") or {}).items() if d not in reports}
    entries = dict(held)
    when = at or now()
    for d, r in reports.items():
        timed = r.get("fit_seconds") or {}
        fits = int(timed.get("fits") or 0)
        try:
            budget[d] = launcher.budget_of(r, fits)
        except launcher.Refused as exc:
            raise Refused(f"{d}: {exc}") from None
        entries[d] = {"commit": r.get("commit"), "engine_tree": r.get("engine_tree"),
                      "fits": fits, "total_seconds": float(timed.get("total") or 0.0),
                      "slowest_fit": timed.get("max"),
                      "sha256": (_sha256(Path(paths[d])) if paths and d in paths else None),
                      "filled_at": when}
    section = filled | {"reports": entries, "fit_budget": budget,
                        "compute_cap": compute_cap_of(entries), "default_plan": default,
                        "r0": r0}
    return build(section)


def _same(a: Any, b: Any) -> bool:
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def fill_r0(prereg: Mapping[str, Any], record: Any, replay: Mapping[str, Any], *,
            paths: Mapping[str, str] | None = None, launches: Path | None = None,
            at: str | None = None) -> dict[str, Any]:
    """``prereg`` with r0 measured (the module docstring): ``replay`` (``score_pilot.make_replay``'s
    shape) of ``record`` (a record's JSON, or the record), the default path's run on the Predict
    fixture, read by ``score_pilot.score_replay``, after the Predict fixture's crash fill and
    before its first graded run (:func:`graded_runs` in ``launches``), recorded with when it was
    filled (``at``, else now); each input checked, and refused with why where one fails.
    ``paths``: the ``record`` and ``replay`` files, for their SHA-256."""
    from research.swarm import score_pilot as S
    from research.swarm.record import validate

    filled = dict(prereg.get("filled") or {})
    plan = filled.get("default_plan")
    frozen = frozen_engine(prereg)
    if not plan or PREDICT_FIXTURE not in (filled.get("fit_budget") or {}) or frozen is None:
        raise Refused("r0 is measured after the Predict fixture's crash fill: fill its crash "
                      "sweep's fields first (python -m research.swarm.prereg_fill crash …).")
    runs = graded_runs(PREDICT_FIXTURE, launches)
    if runs:
        raise Refused(graded_why(PREDICT_FIXTURE, runs))
    final = (list(plan.get("families") or []) or [None])[0]
    if final != RULED_FINAL:
        raise Refused(ruled_final_why(final))
    rec = validate(dict(record)) if isinstance(record, Mapping) else record
    if rec.run.analyst_model != DEFAULT_MODEL or rec.dataset.id != PREDICT_FIXTURE:
        raise Refused(f"The record is not the default path's on the Predict fixture "
                      f"({DEFAULT_MODEL} on {PREDICT_FIXTURE}): its model is "
                      f"{rec.run.analyst_model}, its dataset {rec.dataset.id}.")
    if rec.export.sha256 is None or rec.result.refused is not None or not rec.result.family:
        why = (rec.result.refused or (rec.export.refused or {}).get("message")
               or "no family was declared final")
        raise Refused(f"The default path's record holds no exported plan with its final family: "
                      f"{why}.")
    if rec.result.family != final:
        raise Refused(f"The default path's record declares `{rec.result.family}` final, and the "
                      f"crash sweep's default plan `{final}`: they are not one plan.")
    tree, why = record_engine_why(rec, frozen)
    if why is not None:
        raise Refused(why)
    why = S.paired(replay, rec, prereg)
    if why is not None:
        raise Refused(why)
    scored = S.score_replay(replay, prereg)
    if scored["verdict"] in ("invalid", "refused") or scored.get("regret") is None:
        reasons = scored.get("reasons") or {}
        raise Refused(f"r0 is not measured on this replay: {scored['why']}"
                      + ("" if not reasons else " " + "; ".join(
                          f"{why} ({n})" for why, n in reasons.items())))
    engine = dict(replay.get("engine") or {})
    entry = {"run": rec.run.id, "engine_commit": rec.run.engine_commit,
             "engine_modified": rec.run.engine_modified, "engine_tree": tree,
             "replay_engine": engine, "family": rec.result.family,
             "bundle_sha256": rec.export.sha256, "draws": scored["draws"],
             "seeds": list(prereg["replays"]["seeds"]),
             "evaluation": dict(prereg["replays"]["predict_evaluation"]),
             "mean_regret": scored["regret"],
             "record_sha256": _sha256(Path(paths["record"])) if paths and "record" in paths
             else None,
             "replay_sha256": _sha256(Path(paths["replay"])) if paths and "replay" in paths
             else None, "filled_at": at or now()}
    return build(filled | {"r0": entry})


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
          dataset: str | None = None, margins: bool = True) -> str | None:
    """Why ``dataset`` may not launch on ``prereg``, or with no dataset why the pilot is not
    filled whole, as its sweep step needs (the module docstring); None where nothing stops it.
    ``engine``: the engine now (``launcher.engine_now`` where None); ``margins``: also recompute
    the margins."""
    filled = prereg.get("filled") or {}
    budget = filled.get("fit_budget") or {}
    reports = filled.get("reports") or {}
    if dataset is not None and dataset not in PILOT:
        return (f"{dataset} is not one of the pilot's datasets ({', '.join(PILOT)}), so "
                f"prereg.json pre-registers no run of it.")
    wanted = list(PILOT) if dataset is None else [dataset]
    missing = [d for d in wanted if d not in budget or d not in reports]
    if missing:
        return (f"prereg.json holds no fit budget for {', '.join(missing)}: fill it from "
                f"{'its' if len(missing) == 1 else 'their'} own crash sweep "
                f"(python -m research.swarm.prereg_fill crash …).")
    if dataset is None and not filled.get("compute_cap"):
        return "prereg.json holds no compute cap for the sweeps: fill it from the crash sweeps."
    trees = {e.get("engine_tree") for e in reports.values()}
    if len(trees) != 1:
        return (f"The crash sweeps prereg.json was filled from ran on more than one engine tree "
                f"({', '.join(sorted(map(str, trees)))}): fill them again on one.")
    if dataset in (None, PREDICT_FIXTURE):
        r0 = (prereg.get("predict_fixture") or {}).get("r0") or {}
        if r0.get("value") is None:
            return f"prereg.json holds no r0: {r0.get('why')}"
        measured = filled.get("r0") or {}
        on = (measured.get("engine_tree"),
              (measured.get("replay_engine") or {}).get("engine_tree"))
        if set(on) != trees or measured.get("engine_modified") is not False \
                or (measured.get("replay_engine") or {}).get("engine_modified") is not False:
            return (f"prereg.json's r0 was measured on a record made on engine tree {on[0]} and a "
                    f"replay run on {on[1]}, not both on the crash sweeps' "
                    f"{', '.join(sorted(map(str, trees)))} with no file modified: measure it "
                    f"again (python -m research.swarm.prereg_fill r0 …).")
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
                    f"write it again before any graded run.")
    return None


def readiness(prereg: Mapping[str, Any], engine: Mapping[str, Any] | None = None
              ) -> dict[str, str | None]:
    """:func:`ready` for each pilot dataset, and for the pilot whole (``"all"``), on a
    pre-registration just written from the generators (so its margins are theirs)."""
    if engine is None:
        from research.swarm import launcher

        engine = launcher.engine_now()
    out: dict[str, str | None] = {d: ready(prereg, engine, dataset=d, margins=False)
                                  for d in PILOT}
    out["all"] = ready(prereg, engine, margins=False)
    return out


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
    c.add_argument("reports", type=Path, nargs="+",
                   help="the crash.json of each pilot dataset filled now (any of them)")
    r = sub.add_parser("r0", help="measure r0 on the default plan's replay, after the Predict "
                                  "fixture's crash fill")
    r.add_argument("--record", type=Path, required=True,
                   help="the default path's record on the Predict fixture")
    r.add_argument("--replay", type=Path, required=True,
                   help="its replay (python -m research.swarm.score_pilot replay)")
    p = sub.add_parser("prices", help="each model's list price relative to the others'")
    p.add_argument("prices", nargs="+", help="model=price, for each of " + ", ".join(MODELS))
    p.add_argument("--source", required=True, help="where the list prices were read")
    y = sub.add_parser("ready", help="whether a dataset may launch, or the pilot is filled whole")
    y.add_argument("--dataset", help="a pilot dataset (none: every one, and the compute cap)")
    args = parser.parse_args(argv)
    try:
        if args.command == "crash":
            reports, paths = {}, {}
            for path in args.reports:
                r = json.loads(path.read_text("utf-8"))
                reports[str(r.get("dataset"))], paths[str(r.get("dataset"))] = r, str(path)
            out = fill(load(args.prereg), reports, paths=paths)
            write(out, args.prereg)
            print(json.dumps({"filled": str(args.prereg), "datasets": sorted(reports),
                              "r0": out["predict_fixture"]["r0"], "ready": readiness(out)}))
            return 0
        if args.command == "r0":
            out = fill_r0(load(args.prereg), json.loads(args.record.read_text("utf-8")),
                          json.loads(args.replay.read_text("utf-8")),
                          paths={"record": str(args.record), "replay": str(args.replay)})
            write(out, args.prereg)
            print(json.dumps({"filled": str(args.prereg), "r0": out["predict_fixture"]["r0"],
                              "ready": readiness(out)}))
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
        why = ready(load(args.prereg), dataset=args.dataset)
        print(json.dumps({"dataset": args.dataset, "ready": why is None, "why": why}))
        return 0 if why is None else 1
    if args.check:
        changed = check(load(args.prereg))
        for k in changed:
            print(f"prereg.json's {k} is not what the generators give")
        return 1 if changed else 0
    filled = load(args.prereg).get("filled") if args.prereg.exists() else None
    print(f"wrote {write(build(filled), args.prereg)}")
    return 0


__all__ = ["PREREG", "RULED_FINAL", "Refused", "binomial_band", "build", "check", "compute",
           "compute_cap_of", "default_plan_of", "engine_tree_of", "fill", "fill_r0",
           "frozen_engine", "graded_menus", "graded_runs", "in_band", "k_ruling",
           "latent_discoveries", "load", "main", "measurement_oracle", "r0_of", "r0_ruling",
           "readiness", "ready", "record_engine_why", "replay_seeds", "ruled_final_why",
           "set_prices", "swept_plans", "write"]


if __name__ == "__main__":
    sys.exit(main())
