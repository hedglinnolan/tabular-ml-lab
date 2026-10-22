"""The driver's dry run (DRIVER_PLAN D6, the scripted analyst): a scripted run on each of the pilot's
fixtures, through the driver's own commands, to its record and the six mechanical checks.

    python -m research.swarm.dry_run [--runs DIR] [--out research/swarm/dryrun] [name ...]

Each run (:data:`RUNS`) starts a scripted run (``start --scripted``): the driver's persistent
local server in analyst mode with two workers, its data file rebuilt here from the intake's
manifest and held to the manifest's SHA-256. A scripted analyst then answers each open question
through ``ask``, ``preview``, ``answer``, ``revisit``, ``fit``, ``results``, ``seal`` and
``export``, as ``python -m research.swarm.driver`` subprocesses, from journeys.py's answer source
(``turbotab/core/reference/journeys.py``, its module docstring): the run's own answer where its
spec gives one (the card's outcome, purpose and exposure, the journey's roles); else the app's
own first-ranked option as the driver offers it; else the acceptance drivers' answer for the
grouping, the exposure, the adjustment set, the time-varying exposure and the forms; every reading
the server asks from the fixture's declared truth (``truths.FIXTURE_TRUTHS``) with the journey's
readings beside it, else the app's own guess. The analyst's choices are limited to what the
driver offers, and it answers only the Router's questions, the readings, the role lines and the
findings, as journeys.py does: it makes no declaration (the open declaration lines ``ask`` lists,
the survey journey's scale among them), so each run's construction is what it was before the
driver offered them. Each answer is tagged with a marker of the controlled vocabulary by the
question it answers (:data:`MARKER_OF`; record version 2's ``Tag``), and its one preview says why.
One run explores (a preview of another roles answer, two revisits, three presses of Fit,
the last for answers already fitted, so two fits); the others pass once. Each run's id is the one
``start`` makes itself (``run_`` and a counter): no name of the dry run's reaches its analyst.

With the server still up the record is assembled (``research.swarm.assemble``) into ``--out``, the
checks run on it (``research.swarm.checks``), the server is stopped by its own PID, and each
check's result is compared with :data:`EXPECTED`: each run's expected result per check, derived
from the run's construction and the dataset's declared truth before any run, with why
(:data:`WHY`). A check whose result differs is a defect to fix in the check, the assembler or the
driver, never an expectation to edit. ``comparison.json`` and ``comparison.md`` in ``--out`` hold
the table. No analyst command may be refused as reading what a run is judged against
(``driver.analyst_sealed``); the dry run says so where one is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "research" / "datasets"
OUT = REPO / "research" / "swarm" / "dryrun"
COMMAND_SECONDS = 300.0
RUN_SECONDS = 600.0  # each run stays under ten minutes (the task's bound)
BUDGET_FITS, BUDGET_SECONDS = 4, 600.0
PROMPT_VERSION = "dry-run-2"  # record version 2: each answer tagged, each preview with a reason
# The scripted analyst's tag on each answer, by the question it answers: a marker of the
# vocabulary (``research/swarm/vocabulary.py``) the question's answer rests on; a reading, a role
# line, a finding or a way forward takes the default's.
MARKER_OF = {
    "lens": "purpose", "orientation": "design", "target": "outcome_type", "event": "event_share",
    "task": "outcome_type", "follow_up": "time_order", "design": "design", "purpose": "purpose",
    "grain": "repeated_units", "repeat_kind": "repeated_measurement", "unit": "repeated_units",
    "aggregation": "repeated_measurement", "temporal": "time_order", "roles": "predictors_by_role",
    "clusters": "design", "survey": "design", "exclusions": "rows_analyzed",
    "missing": "missingness", "split": "rows_analyzed", "estimand": "estimand",
    "adjustment": "estimand", "time_varying": "time_order", "energy_adjustment": "units",
    "form": "predictors_per_row", "causal": "estimand", "models": "events_per_parameter",
    "substitution": "units"}
DEFAULT_MARKER = "design"

# ── each run's construction ─────────────────────────────────────────────────


@dataclass
class Spec:
    """One scripted run: the dataset, the journey's answers and readings, and whether it explores."""

    name: str
    dataset: str
    target: str
    event: str
    purpose: str
    lenses: list[str]
    readings: dict[str, str]
    roles: dict[str, str] | None = None  # the journey's roles, else the proposals confirmed
    exposure: str | None = None
    repairs: bool = False  # each finding takes its first repair (journeys.apply_first_repair(".")
    # The run's own answers, by question, as a journey's spec gives them (``Journey.answers``):
    # each an option the driver offers and its settings.
    answers: dict[str, tuple[list[str], list[str]]] = field(default_factory=dict)
    explore: bool = False
    known: dict[str, Any] | None = None  # the checks' known answer, from the declared truth


def _truth(key: str) -> dict[str, str]:
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    return dict(FIXTURE_TRUTHS[key])


def specs() -> dict[str, Spec]:
    """The runs. The survey run asks the card's question, age's association with having sought
    support, with journeys.py's ``survey-inference`` journey's roles and readings: the ten items
    the journey scores as its scale (item_01 to item_10) enter one by one as covariates beside age,
    sex and education, and the other thirty are excluded, since the scripted analyst makes no
    declaration, a scale among them (the card names none, so which columns enter is the analyst's
    decision). Age's
    form is declared a straight line (a reading journeys takes in place of the form card's
    proposal), so the known answer is one row; its declared truth is that age has no effect, on
    every scale (``answers.json``), so the known answer is no effect (``null``): 0 on the
    difference scale of the marginal risk difference the estimand declares (the card's exposure,
    its first-ranked measure), 1 on a ratio's; the run gives the linear family as its own models
    answer, as the journey's capture answered it (without the scale, which the scripted analyst
    does not declare, the shelf ranks the elastic net first, which reports no effect: the export refuses
    it). The Predict fixture and the decoy have no journey:
    their specs are the card's outcome and purpose, the clinical lens, the fixture's declared
    truth (none for the decoy), and each finding's first repair, as journeys' clinical journey takes
    them. The Predict fixture declares the generating model and the C-statistic of its true risk,
    not a range a fitted model's held-out score must fall in, so its result has no known answer.
    Both Predict runs give the split as their own answer, a fifth of the rows held out (the plan's
    first option holds none out, so the seal's way forward would draw them and refit): the seal
    then opens on the fit it was pressed for."""
    from turbotab.core.reference.journeys import SURVEY_INFERENCE_ROLES, SURVEY_READINGS

    return {s.name: s for s in (
        Spec("clinical_explore", "clinical_readmission_predict", target="readmit_30d", event="1",
             purpose="prediction", lenses=["clinical"], readings=_truth("clinical_risk.csv"),
             repairs=True, answers={"split": (["0.2"], [])}, explore=True),
        Spec("survey_single", "survey_support_estimate", target="sought_support", event="1",
             purpose="inference", lenses=["survey"],
             readings={**_truth("survey_instrument.csv"), **SURVEY_READINGS, "form:age": "linear"},
             roles=dict(SURVEY_INFERENCE_ROLES), exposure="age",
             answers={"models": (["linear"], [])},
             known={"family": "linear", "feature": "age", "null": True}),
        Spec("decoy_single", "outpatient_admission_predict", target="admitted_2y", event="1",
             purpose="prediction", lenses=["clinical"], readings={}, repairs=True,
             answers={"split": (["0.2"], [])}),
    )}


RUNS = ("clinical_explore", "survey_single", "decoy_single")
CHECKS = ("seal_integrity", "logging", "single_pass", "known_answer", "null_decoy", "robustness")

# Each check's expected result per run, from the construction above and the dataset's declared
# truth, written before any run (``single_pass`` "fail" is the check flagging a single pass).
EXPECTED: dict[str, dict[str, str]] = {
    "clinical_explore": {"seal_integrity": "pass", "logging": "pass", "single_pass": "pass",
                         "known_answer": "not_checked", "null_decoy": "not_checked",
                         "robustness": "pass"},
    "survey_single": {"seal_integrity": "pass", "logging": "pass", "single_pass": "fail",
                      "known_answer": "pass", "null_decoy": "not_checked", "robustness": "pass"},
    "decoy_single": {"seal_integrity": "pass", "logging": "pass", "single_pass": "fail",
                     "known_answer": "not_checked", "null_decoy": "pass", "robustness": "pass"},
}
WHY: dict[str, dict[str, str]] = {
    "clinical_explore": {
        "seal_integrity": "One opening, no re-seal; the driver refuses any change after it, so "
                          "no later record reaches the result.",
        "logging": "Every answer the script posts carries its reason (the driver requires one), "
                   "TurboTab's own records need none, and each fit's family has the engine's "
                   "movement or its why.",
        "single_pass": "Before the opening the roles answer had another option previewed (sodium "
                       "left out), and the run fits more than once.",
        "known_answer": "A known-answer control whose fact readings are answered from its declared "
                        "truth (no problem), with no known answer for the held-out score.",
        "null_decoy": "Not a decoy.",
        "robustness": "The script's own export returns the bundle, and no person helps."},
    "survey_single": {
        "seal_integrity": "Under Estimate no seal; the result is the fit the lock was recorded on, "
                          "and no record after the lock is made.",
        "logging": "Every answer the script posts carries its reason; the split and the lock are "
                   "TurboTab's own.",
        "single_pass": "No preview, no revisit, and one fit: flagged as a single pass.",
        "known_answer": "Fact readings answered from the declared truth; age has no effect by "
                        "construction, so its marginal risk difference, the number the app "
                        "reports for the estimand, is 0: its 95% interval holds 0 (in 1 run in 20 "
                        "it would not by chance; that would be checked against an independent fit "
                        "before calling it a defect), or the effects stage refuses it, as it does "
                        "under multiple imputation (the app's first-ranked missing-values answer "
                        "here): right or refused.",
        "null_decoy": "Not a decoy.",
        "robustness": "The script's own export returns the bundle, and no person helps."},
    "decoy_single": {
        "seal_integrity": "One opening, no re-seal, no change after it.",
        "logging": "Every answer the script posts carries its reason.",
        "single_pass": "No preview, no revisit, and one fit: flagged as a single pass.",
        "known_answer": "Not a known-answer control.",
        "null_decoy": "No column enters the outcome (decoys.py), so the held-out AUC's interval "
                      "lies above 0.5 in about 1 run in 40 by chance only: none expected; a model "
                      "that gives everyone the same score has an AUC of 0.5, no skill itself, and "
                      "its interval refused.",
        "robustness": "The script's own export returns the bundle, and no person helps."},
}


# ── the data file, from its manifest ─────────────────────────────────────────


def table_of(dataset: str, folder: Path) -> Path:
    """The dataset's table, written into ``folder`` from what its manifest names (a repository
    fixture copied; the decoy generated from its seed) and held to the manifest's SHA-256."""
    manifest = json.loads((DATASETS / dataset / "manifest.json").read_text("utf-8"))
    table = manifest["table"]
    dest = folder / Path(table["file"]).name
    folder.mkdir(parents=True, exist_ok=True)
    if manifest.get("files"):
        shutil.copyfile(REPO / manifest["files"][0]["file"], dest)
    else:
        from research.swarm import decoys

        gen = manifest["generator"]
        decoys.generate(int(gen["seed"]), int(gen["rows"])).to_csv(dest, index=False)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != table["sha256"]:
        raise RuntimeError(f"{dest} is not the manifest's table ({digest} != {table['sha256']})")
    return dest


# ── the scripted analyst, through the driver's commands ──────────────────────


class Refused(RuntimeError):
    pass


@dataclass
class Analyst:
    spec: Spec
    runs: Path
    run: str = ""
    log: list[dict[str, Any]] = field(default_factory=list)
    proposed: dict[str, str] = field(default_factory=dict)
    families: list[str] = field(default_factory=list)
    n: int = 0
    started: float = field(default_factory=time.monotonic)

    def drv(self, *args: str) -> tuple[int, dict[str, Any]]:
        env = {**os.environ, "TURBOTAB_SWARM_RUNS": str(self.runs), "OMP_NUM_THREADS": "2",
               "TURBOTAB_WORKERS": "2"}
        r = subprocess.run([sys.executable, "-m", "research.swarm.driver", *args], cwd=REPO,
                           env=env, capture_output=True, text=True, timeout=COMMAND_SECONDS)
        try:
            payload = json.loads(r.stdout)
        except ValueError:
            raise RuntimeError(f"{args[:2]}: no JSON: {r.stdout[-800:]} {r.stderr[-800:]}") from None
        refused = payload.get("refused") or {}
        if refused.get("code") == "not_for_the_analyst":
            raise RuntimeError(f"an analyst command read what a run is judged against: {args}")
        self.log.append({"args": list(args[:3]), "exit": r.returncode,
                         "refused": refused.get("code")})
        if time.monotonic() - self.started > RUN_SECONDS:
            raise RuntimeError(f"the run took over {RUN_SECONDS:g} seconds")
        return r.returncode, payload

    def cmd(self, name: str, *args: str) -> tuple[int, dict[str, Any]]:
        return self.drv(name, "--run", self.run, *args)

    def reason(self, what: str) -> str:
        self.n += 1
        return f"Scripted analyst, step {self.n}: {what}"

    def answer(self, ids: list[str], sets: list[str], why: str,
               marker: str = DEFAULT_MARKER) -> dict[str, Any]:
        args = [*ids, *[a for s in sets for a in ("--set", s)], "--reason", self.reason(why),
                "--marker", marker]
        code, out = self.cmd("answer", *args)
        if code != 0:
            raise Refused(json.dumps(out)[:1500])
        return out

    # what journeys answers from the readings

    def reading(self, reading: str, column: str, groups: list[dict[str, Any]]) -> str | None:
        """The declared truth's value for a reading, else the app's guess (its ask card's group;
        journeys' ``GuessingTruth``)."""
        key = f"{reading}:{column}"
        if key in self.spec.readings:
            return str(self.spec.readings[key])
        for g in groups:
            if g.get("kind") == reading and column in (g.get("columns") or []) \
                    and g.get("guess") is not None:
                return str(g["guess"])
        return None

    def reading_way(self, q: dict[str, Any]) -> str | None:
        """A way forward confirming one reading the server asks at the value journeys gives it."""
        readings = q.get("readings") or {}
        ways = [*(readings.get("exits") or []), *(q.get("last_refusal_exits") or []),
                *(w for line in q.get("also_open") or [] for item in line.get("items") or []
                  for w in item.get("options") or [])]
        for way in ways:
            d = way.get("decision") or {}
            if d.get("kind") == "confirm_reading" and d.get("reading") != "role" and way.get("id"):
                if self.reading(d["reading"], d["column"], readings.get("groups") or []) == \
                        str(d.get("value")):
                    return way["id"]
            if d.get("kind") == "set_column_unit" and way.get("id"):
                unit = self.spec.readings.get(f"unit:{d.get('column')}")
                if unit is not None and unit == str(d.get("unit")):
                    return way["id"]
        return None

    def role_way(self, q: dict[str, Any]) -> str | None:
        """A role line's press confirming the role the roles answer gave (``confirm_each``)."""
        given = {**self.proposed, **(self.spec.roles or {})}
        for line in q.get("also_open") or []:
            for item in line.get("items") or []:
                for way in item.get("options") or []:
                    d = way.get("decision") or {}
                    if d.get("kind") == "confirm_role" and given.get(d.get("column")) == d.get("role"):
                        return way["id"]
                    if d.get("kind") == "confirm_readings" and d.get("items") and all(
                            i.get("reading") == "role" and given.get(i.get("column")) == i.get("value")
                            for i in d["items"]):
                        return way["id"]
        return None

    def finding_way(self, q: dict[str, Any]) -> str | None:
        """A finding's first repair (journeys' ``apply_first_repair``), where the run takes them."""
        if not self.spec.repairs:
            return None
        for line in q.get("also_open") or []:
            if line.get("source") != "finding":
                continue
            way = next((w for w in line.get("options") or []
                        if (w.get("decision") or {}).get("kind") == "apply_repair"), None)
            if way is not None:
                return way["id"]
        return None

    # what journeys answers each question with, as the driver offers it

    def artifact(self, stage: str) -> dict[str, Any]:
        """The stage's artifact as the run's server serves it (journeys reads the same)."""
        import urllib.request

        meta = json.loads((self.runs / self.run / "driver.json").read_text("utf-8"))
        url = f"http://127.0.0.1:{meta['port']}/api/projects/{meta['project']}/stages/{stage}"
        with urllib.request.urlopen(url, timeout=120) as r:
            return (json.loads(r.read() or b"null") or {}).get("artifact") or {}

    def journeys_body(self, key: str) -> dict[str, Any] | None:
        """journeys.py's own answer to ``key`` where it reads the app's first-ranked option from an
        artifact (``first_exclusions``, ``first_missing``, ``first_split``, ``first_energy``,
        ``first_models``), over this run's server; None for any other question."""
        from types import SimpleNamespace

        from turbotab.core.reference import journeys as J
        from turbotab.server import app  # noqa: F401 - every family the server registers

        run = SimpleNamespace(drive=SimpleNamespace(artifact=lambda s, timeout=None: self.artifact(s)),
                              note=lambda text: None)
        firsts = {"exclusions": J.first_exclusions, "missing": J.first_missing,
                  "split": J.first_split, "energy_adjustment": J.first_energy}
        if key in firsts:
            return firsts[key](run)
        if key == "models":
            return J.first_models(run, {"state": {"purpose": self.spec.purpose}})
        return None

    def offered_as(self, body: dict[str, Any], offered: list[dict[str, Any]]) -> str | None:
        """The offered option that records ``body``, by its id."""
        from research.swarm.options import canonical

        return next((str(o["id"]) for o in offered if o.get("decision") is not None
                     and canonical(o["decision"]) == canonical(body)), None)

    def choose(self, key: str, q: dict[str, Any]) -> tuple[list[str], list[str], str]:
        offered = q.get("options") or []
        ids = [str(o["id"]) for o in offered if o.get("id") is not None]
        data = q.get("data") or {}
        first = ids[0] if ids else None
        spec = self.spec
        if key in spec.answers:
            given, sets = spec.answers[key]
            missing = [i for i in given if i not in ids]
            if missing:
                raise RuntimeError(f"the run's own {key} answer {missing} is not offered: {ids}")
            if key == "models":
                self.families = list(given)
            return list(given), list(sets), "the run's own answer"
        body = self.journeys_body(key)
        if key == "models" and body is not None:
            self.families = list(body["models"])
            return list(body["models"]), [], "the shelf's first-ranked families (journeys)"
        if key == "split" and body is not None:
            sets = [] if body["validation"] == ((q.get("fields") or [{}])[0].get("initial")) \
                else [f"validation={body['validation']}"]
            return [format(float(body["holdout"]), "g")], sets, \
                "the seal plan's first option (journeys)"
        if body is not None:
            pick = self.offered_as(body, offered)
            if pick is None:
                raise RuntimeError(f"journeys answers {key} with {body}, which the driver does not "
                                   f"offer: {[o.get('decision') for o in offered]}")
            return [pick], [], "the app's first-ranked option (journeys)"
        if key == "lens":
            return list(spec.lenses), [], "the journey's lens"
        if key == "target":
            return [spec.target], [], "the card's outcome"
        if key == "event":
            return [spec.event], [], "the card's outcome level"
        if key == "purpose":
            return [spec.purpose], [], "the card's purpose"
        if key == "follow_up":
            return ["same"], [], "one period for everyone, the acceptance drivers' answer"
        if key == "grain":
            return ["one_row_per_unit"], [], "one row per unit, the card's population"
        if key == "task":
            detected = data.get("detected_task")
            pick = next((o["id"] for o in offered
                         if (o.get("decision") or {}).get("task") == detected), first)
            return [pick], [], "the app's detection"
        if key == "roles":
            self.proposed = {c["column"]: c["proposed"] for c in data.get("columns") or []}
            sets = [f"{c}={r}" for c, r in (spec.roles or {}).items() if self.proposed.get(c) != r]
            return ["proposed"], sets, ("the journey's roles" if spec.roles else
                                        "the app's proposed roles, confirmed")
        if key == "clusters":
            named = [g["column"] for g in data.get("groupings") or []]
            pick = next((o["id"] for o in offered if o.get("column") in named[:1]
                         and (spec.purpose == "prediction" or str(o["id"]).startswith(
                             "cluster_only:"))), "none")
            return [pick], [], "the roles' named grouping, else none (the acceptance drivers)"
        if key == "estimand":
            exposure = spec.exposure or spec.readings.get(f"exposure:{spec.target}") or first
            return [exposure], [], "the card's exposure, its first-ranked measure"
        if key == "adjustment":
            sets = []
            for o in offered:
                for column in o.get("columns") or []:
                    said = spec.readings.get(f"adjust:{column}")
                    guess = o.get("guess") or {}
                    if said is None and guess:
                        said = ",".join(str(guess.get(k)) for k in ("causes_exposure",
                                                                    "causes_outcome",
                                                                    "after_exposure"))
                    if said is not None and f"{column}=" not in "".join(sets):
                        sets.append(f"{column}={said}")
            return ["answers"], sets, "each covariate's causal place from the readings"
        if key == "time_varying":
            return ["standard"], ["ordering=exposure_precedes_outcome"], \
                "standard regression, the exposure before the outcome (the acceptance drivers)"
        if key == "form":
            sets = [f"{f['name']}={spec.readings[f'form:' + f['name']]}"
                    for f in q.get("fields") or [] if f"form:{f['name']}" in spec.readings]
            return ["proposed"], sets, "the form card's proposal, unless the readings declare one"
        if key == "causal":
            return ["none"], [], "the primary model alone"
        if key == "survey":
            return [first], [], "the survey card's first-ranked option"
        if key == "substitution":
            return [first], ["step_kcal=100"], "the first pair the design offers"
        if first is None:
            raise RuntimeError(f"no option the driver offers for {key!r}: {json.dumps(q)[:800]}")
        return [first], [], "the app's first-ranked option"

    # the run

    def ask(self) -> dict[str, Any]:
        code, q = self.cmd("ask")
        if code != 0:
            raise RuntimeError(f"ask refused: {json.dumps(q)[:1000]}")
        return q

    def settle(self, q: dict[str, Any]) -> bool:
        """Answer one reading, role line or finding the open question rests on. True if one was."""
        for way, why in ((self.reading_way(q), "a reading the server asked, from the declared "
                                               "truth or the app's guess"),
                         (self.role_way(q), "a role confirmed as the roles answer gave it"),
                         (self.finding_way(q), "a finding acted on with its first repair")):
            if way:
                self.answer([way], [], why)
                return True
        return False

    def until(self, stop: Any, explore: dict[str, Any] | None = None) -> dict[str, Any]:
        """Answer the Router until ``stop(q)``; returns that ask."""
        explore = explore or {}
        waits = 0
        for _ in range(300):
            q = self.ask()
            if self.settle(q):
                continue
            if stop(q):
                return q
            question = q.get("question")
            if question is None or q.get("fit_reached") or question.get("key") == "open_seal":
                return q  # the seal's question is opened by ``seal``, never answered
            if question["status"] != "open":
                if q.get("failed"):
                    raise RuntimeError(f"the question waits on failed stages: {q['failed']}")
                waits += 1
                if waits > 20:
                    raise RuntimeError(f"the {question['key']} question kept waiting")
                continue
            waits = 0
            key = question["key"]
            if key in explore:
                explore.pop(key)(q)
            ids, sets, source = self.choose(key, q)
            try:
                self.answer(ids, sets, f"{key} answered from {source}.",
                            MARKER_OF.get(key, DEFAULT_MARKER))
            except Refused as exc:
                out = json.loads(str(exc)) if str(exc).startswith("{") else {}
                ways = [w for w in (out.get("refused") or {}).get("exits") or []
                        if w.get("decision") and w.get("id")]
                if not ways:
                    raise
                # a refusal that asks for no reading: its first way forward (journeys.post_answer)
                q2 = self.ask()
                if not self.settle(q2):
                    self.answer([ways[0]["id"]], [], f"the refusal's first way forward, "
                                                     f"{ways[0].get('label')}")
        raise RuntimeError("the Router never stopped")

    def press(self, why: str) -> dict[str, Any]:
        code, out = self.cmd("fit", "--wait", "240")
        if code != 0:
            raise RuntimeError(f"fit refused ({why}): {json.dumps(out)[:1000]}")
        return out

    def results(self) -> dict[str, Any]:
        code, out = self.cmd("results", "--wait", "240")
        if code != 0:
            raise RuntimeError(f"results refused: {json.dumps(out)[:1000]}")
        return out

    def cv(self, out: dict[str, Any]) -> str:
        words = []
        for f in (out.get("results") or {}).get("families") or []:
            auc = ((f.get("cv") or {}).get("auc") or {}).get("mean")
            if auc is not None:
                words.append(f"{f['family']} {auc:.3f}")
        return ", ".join(words) or "no score served"

    def last(self, kind: str) -> str:
        q = self.ask()
        return next(d["id"] for d in reversed(q.get("decisions") or []) if d.get("kind") == kind)

    def revisit(self, record: str, why: str) -> None:
        code, out = self.cmd("revisit", record, "--reason", self.reason(why))
        if code != 0:
            raise RuntimeError(f"revisit refused: {json.dumps(out)[:1000]}")

    def reopen(self, key: str, kind: str, why: str) -> dict[str, Any]:
        """Revisit the last record of ``kind`` until its question is open again (TurboTab may
        record it again itself, as the roles from one-by-one confirmations): its ask."""
        self.revisit(self.last(kind), why)
        for _ in range(3):
            q = self.until(lambda q: (q.get("question") or {}).get("key") == key
                           and q["question"]["status"] == "open")
            if (q.get("question") or {}).get("key") == key:
                return q
            self.revisit(self.last(kind), f"{why} (TurboTab had recorded it again)")
        raise RuntimeError(f"the {key} question did not open again")

    def seal(self, family: str, why: str) -> dict[str, Any]:
        reason = self.reason(why)
        for _ in range(6):
            code, out = self.cmd("seal", family, "--reason", reason)
            if code == 0:
                return out
            refused = out.get("refused") or {}
            ways = [w for w in refused.get("exits") or [] if w.get("decision") and w.get("id")]
            sweep = next((w for w in ways if w["decision"].get("kind") == "confirm_sweep"), None)
            act = next((w for w in ways if w["decision"].get("kind") != "open_seal"), None)
            way = sweep or act
            if way is None:
                raise RuntimeError(f"seal refused: {json.dumps(out)[:1000]}")
            self.answer([way["id"]], [], f"the opening's way forward, {way.get('label')}")
        raise RuntimeError("the seal never opened")


def drive(a: Analyst) -> None:
    """The scripted analyst's run (the module docstring)."""
    spec = a.spec

    def preview_roles(q: dict[str, Any]) -> None:
        column = "sodium_mmol_l"
        code, out = a.cmd("preview", "proposed", "--set", f"{column}=excluded", "--reason",
                          a.reason(f"what the roles would record with {column} left out"))
        if code != 0:
            raise RuntimeError(f"preview refused: {json.dumps(out)[:800]}")

    a.until(lambda q: False, {"roles": preview_roles} if spec.explore else {})
    a.press("the Router reached Fit")
    first = a.results()
    if spec.explore:
        q = a.reopen("roles", "set_roles", f"Revisit the roles: testing whether sodium, which no "
                                           f"card reason ties to readmission, carries signal. "
                                           f"CV AUC with it: {a.cv(first)}.")
        ids, sets, _ = a.choose("roles", q)
        a.answer(ids, [*sets, "sodium_mmol_l=excluded"],
                 "roles with sodium left out, to test whether it carries signal.")
        a.until(lambda q: False)
        a.press("the roles changed")
        second = a.results()
        # A revert: the revisit's own revert withdrawn, which brings the first roles answer back
        # under the second, a later answer to the same question: the plan, and so its fit, stay
        # the second's (no new fit; Fit is pressed for answers already fitted).
        a.revisit(a.last("revert"), f"Withdraw the revisit's revert: the roles without sodium "
                                    f"stand, CV AUC {a.cv(second)} against {a.cv(first)} with it.")
        a.until(lambda q: False)
        a.press("the answers already fitted")
        a.results()
    if spec.purpose == "prediction":
        q = a.ask()
        families = [str(o["id"]) for o in q.get("options") or []] if (
            (q.get("question") or {}).get("key") == "open_seal") else []
        family = families[0] if families else (a.families[0] if a.families else None)
        if family is None:
            raise RuntimeError(f"no family to declare final: {json.dumps(q)[:800]}")
        a.seal(family, f"{family} is the shelf's first-ranked family fitted; its held-out score "
                       f"is the result.")
        a.results()
    code, out = a.cmd("export", "--wait", "240")
    if code != 0:
        raise RuntimeError(f"export refused: {json.dumps(out)[:1000]}")


# ── one run, to its record and checks ────────────────────────────────────────


def dataset_meta(spec: Spec) -> dict[str, Any]:
    answers = json.loads((DATASETS / spec.dataset / "answers.json").read_text("utf-8"))
    card = json.loads((DATASETS / spec.dataset / "card.json").read_text("utf-8"))
    control = {"known_answer": "known_answer", "decoy": "null_decoy"}.get(answers.get("kind"))
    return {"id": spec.dataset, "control": control, "fixture": answers.get("fixture_key"),
            "known": spec.known, "codebook": None, "card": card}


def one(spec: Spec, runs: Path, out: Path) -> dict[str, Any]:
    from research.swarm import assemble as A
    from research.swarm import checks as C
    from research.swarm.record import validate

    data = table_of(spec.dataset, runs / "data" / spec.dataset)
    a = Analyst(spec, runs)
    code, started = a.drv("start", "--data", str(data), "--budget-fits", str(BUDGET_FITS),
                          "--budget-seconds", str(BUDGET_SECONDS), "--scripted")
    if code != 0:
        raise RuntimeError(f"start refused: {json.dumps(started)[:1000]}")
    a.run = started["started"]["run"]
    error = None
    try:
        try:
            drive(a)
        except Exception as exc:  # noqa: BLE001 - the run is assembled as it stands, and says why
            error = f"{type(exc).__name__}: {exc}"[:2000]
        meta = {"design": 2, "analyst_model": "scripted", "effort": None,
                "prompt_version": PROMPT_VERSION, "dataset": dataset_meta(spec), "help": []}
        path, ok = A.assemble(runs / a.run, meta=meta, corpus=out)
    finally:
        a.drv("stop", "--run", a.run)
    got: dict[str, Any] = {}
    if ok:
        rec = validate(json.loads(path.read_text("utf-8")))
        got = {o.check: o for o in C.run_checks([rec])}
    rows = []
    for check in CHECKS:
        o = got.get(check)
        rows.append({"check": check, "expected": EXPECTED[spec.name][check],
                     "got": o.outcome if o else "failed run",
                     "agrees": o is not None and o.outcome == EXPECTED[spec.name][check],
                     "why": WHY[spec.name][check],
                     "problems": o.problems if o else [], "unchecked": o.unchecked if o else [],
                     "notes": o.notes if o else []})
    return {"run": spec.name, "run_id": a.run, "dataset": spec.dataset, "record": str(
        path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path), "valid": ok,
            "error": error, "seconds": round(time.monotonic() - a.started, 1),
            "commands": len(a.log), "refused": sorted({e["refused"] for e in a.log
                                                       if e["refused"]}),
            "checks": rows}


def table(results: list[dict[str, Any]]) -> str:
    lines = ["# The driver's dry run: each check against its construction", "",
             "Written by `python -m research.swarm.dry_run` (DRIVER_PLAN D6): each run's expected "
             "result per check, written in `dry_run.py` before the run, beside the check's "
             "result on the run's record.", "",
             "| Run | Dataset | Check | Expected | Got | Agrees |", "|---|---|---|---|---|---|"]
    for r in results:
        for c in r["checks"]:
            lines.append(f"| {r['run']} | {r['dataset']} | {c['check']} | {c['expected']} | "
                         f"{c['got']} | {'yes' if c['agrees'] else 'NO'} |")
    lines.append("")
    for r in results:
        lines.append(f"- `{r['run']}`: record `{r['record']}`, {r['commands']} driver commands, "
                     f"{r['seconds']} s" + (f"; stopped: {r['error']}" if r["error"] else "") + ".")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.dry_run",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", default=list(RUNS))
    parser.add_argument("--runs", type=Path, default=REPO / "research" / "runs" / "dryrun")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    all_specs = specs()
    args.out.mkdir(parents=True, exist_ok=True)
    results = [one(all_specs[n], args.runs.absolute(), args.out.absolute()) for n in args.names]
    # Each run is one command (under ten minutes): the table keeps the runs made before it.
    kept = args.out / "comparison.json"
    earlier = json.loads(kept.read_text("utf-8")) if kept.exists() else []
    now = {r["run"]: r for r in [*earlier, *results]}
    every = [now[n] for n in RUNS if n in now]
    kept.write_text(json.dumps(every, indent=1) + "\n", "utf-8")
    (args.out / "comparison.md").write_text(table(every), "utf-8")
    print(json.dumps([{k: r[k] for k in ("run", "valid", "error", "seconds")}
                      | {"agrees": all(c["agrees"] for c in r["checks"])} for r in results]))
    return 0 if all(r["valid"] and all(c["agrees"] for c in r["checks"]) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
