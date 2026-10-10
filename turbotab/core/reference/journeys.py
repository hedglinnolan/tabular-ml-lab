"""The review packets' reference journeys, run headlessly to the export (V2 definition of done §1
and §4).

Each journey opens a fixture in a local-mode server with two workers over a fresh home (the
acceptance harness's ``server_drive.local_server``), follows the Router question by question, and
ends in the export bundle (``GET /projects/{pid}/export``, ``turbotab.core.export``). What it keeps
is a **capture**: the bundle's methods section exactly as written (``methods.md``), its checklist
(STROBE-nut under inference, TRIPOD+AI under prediction) item by item, every answer the journey
posted and where that answer came from, and what stopped it, if anything. A packet quotes the
capture; nothing in a capture is written by hand.

**How a journey answers.** The open question takes, in this order: the journey's own answer when
its spec gives one (the research question: the lens, the outcome, the purpose, the exposure, the
roles where the fixture's author knows them); else the app's own first-ranked option, read from
the artifact that offers it (the proposals' labeled options for the screens, the missing values
and the energy model; the seal plan for the split; the shelf for the models; the survey card; the
estimand card's first-ranked measure; the form card's proposal for each column); else the
acceptance drivers' answer (``server_drive.answer_wp17``) for the grouping, the time-varying
exposure and the adjustment set, each covariate from the journey's readings. Each decision's
capture says which of these gave it, kind by kind (:data:`WP17_SOURCES`).

**The readings.** A reading the server asks about, and each covariate's causal place, is answered
from the journey's readings: the fixture's declared truth (``truths.FIXTURE_TRUTHS``, BLUEPRINT
§14.3) with the journey's own readings beside or in place of it (the causal assumptions of its
research question, and the units and nesting a fixture without a declared truth needs). The capture
lists both apart (``readings``), so a packet shows the reviewer every assumption the journey
injected. Where neither declares a reading, the app's own best guess from the ask card is
confirmed, and the capture lists each such reading (``guessed``). Where neither declares an
effect measure or a form, the app's first-ranked one is taken, and the capture lists it too
(``app_ranked``). A refusal that asks for something other than readings takes its first exit with a
decision, as a person pressing it would; the capture records the refusal and the exit taken.

    python -m turbotab.core.reference.journeys --list
    python -m turbotab.core.reference.journeys dietary-inference [more names]

writes ``docs/turbotab-next/review-packets/captures/<name>.json`` (and keeps each bundle's zip in
``--bundles``, default a temporary folder). Run each journey once, with ``TURBOTAB_WORKERS=2`` and
``OMP_NUM_THREADS=2``: the journeys are the packets' only fits.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[3]
CAPTURES = REPO / "docs" / "turbotab-next" / "review-packets" / "captures"
SAMPLES = REPO / "turbotab" / "sample_data"
OPEN = ("open", "waiting")
# Readings and units: the refusals the truth answers (``truths.ASKING``), never logged as exits.
ASKING = ("reading_unsettled", "role_unconfirmed", "energy_unit_unconfirmed")


# ── the fixture's truth, with the app's guess where the fixture declares none ─


def _truth_class() -> Any:
    from turbotab.core.tests.truths import Truth

    class GuessingTruth(Truth):
        """A journey's readings; a reading they declare nothing for takes the app's own best guess
        (the ask card's, or the adjustment card's), recorded in ``guessed``; an effect measure or a
        form they declare nothing for (``measure:<exposure>``, ``form:<column>``, which the drivers
        read with ``get``) takes the app's first-ranked one, recorded in ``app_ranked``."""

        def __init__(self, readings: dict[str, Any] | None = None, *, fixture: str = "") -> None:
            super().__init__(readings, fixture=fixture)
            self.guess: Callable[[str, str], str | None] | None = None
            self.guessed: list[dict[str, str]] = []
            self.ranked: Callable[[str, str], str | None] | None = None
            self.app_ranked: list[dict[str, str]] = []

        def get(self, key: Any, default: Any = None) -> Any:
            if key in self or self.ranked is None or not isinstance(key, str):
                return super().get(key, default)
            what, _, column = key.partition(":")
            if what not in RANKED:
                return super().get(key, default)
            value = self.ranked(what, column)
            if value is None:
                return default
            self[key] = value
            self.app_ranked.append({"reading": what, "column": column, "value": value})
            return value

        def answer(self, reading: str, column: str) -> str:
            key = f"{reading}:{column}"
            if key in self:
                return str(self[key])
            value = self.guess(reading, column) if self.guess is not None else None
            if value is None:
                return super().answer(reading, column)  # raises, naming the reading
            self[key] = value
            self.guessed.append({"reading": reading, "column": column, "value": value})
            return value

    return GuessingTruth


def truth(readings: dict[str, Any] | None = None, *, fixture: str) -> Any:
    return _truth_class()(readings, fixture=fixture)


# What the app ranks for the drivers when the journey's readings say nothing: the estimand card's
# first-ranked measure, and the form card's proposal (its form; k is completed by the same rule).
RANKED = ("measure", "form")


def _server_ranked(client: Any, pid: str) -> Callable[[str, str], str | None]:
    """The app's first-ranked answer for ``measure:<exposure>`` (the estimand card's first fitted
    measure, marginal ones included: ESTIMAND ranks them first for a common event) and
    ``form:<column>`` (the form card's proposal for the column)."""

    def artifact(stage: str) -> dict[str, Any]:
        return (client.get(f"/api/projects/{pid}/stages/{stage}").json() or {}).get("artifact") or {}

    def ranked(what: str, column: str) -> str | None:
        if what == "measure":
            card = artifact("proposals").get("estimand") or {}
            first = next((m for m in card.get("measures") or [] if m.get("fitted")), None)
            return str(first["measure"]) if first else None
        need = next((n for n in artifact("forms").get("needs") or [] if n.get("column") == column),
                    None)
        form = ((need or {}).get("proposal") or {}).get("form")
        return str(form) if form else None

    return ranked


def _server_guess(client: Any, pid: str) -> Callable[[str, str], str | None]:
    """The app's best guess for a reading: the open question's ask card, or for an adjustment
    answer (``adjust``), the adjustment card's guess for the group holding the column."""

    def guess(reading: str, column: str) -> str | None:
        view = client.get(f"/api/projects/{pid}").json()
        if reading == "adjust":
            r = client.get(f"/api/projects/{pid}/stages/proposals").json()
            card = ((r or {}).get("artifact") or {}).get("adjustment") or {}
            for group in card.get("groups") or []:
                g = group.get("guess")
                if column in (group.get("columns") or []) and g:
                    return ",".join(str(g.get(k)) for k in
                                    ("causes_exposure", "causes_outcome", "after_exposure"))
            return None
        for step in view.get("interview") or []:
            for group in ((step.get("ask") or {}).get("groups") or []):
                if group.get("kind") == reading and column in (group.get("columns") or []) \
                        and group.get("guess") is not None:
                    return str(group["guess"])
        return None

    return guess


# ── the recording client ─────────────────────────────────────────────────────


class Recording:
    """The acceptance harness's client, with every decision posted through it recorded."""

    def __init__(self, client: Any, run: "Run"):
        self.c = client
        self.run = run

    @property
    def app(self) -> Any:
        return self.c.app

    def get(self, url: str, **kw: Any) -> Any:
        return self.c.get(url, **kw)

    def post(self, url: str, json: Any = None, **kw: Any) -> Any:  # noqa: A002
        r = self.c.post(url, json=json, **kw)
        if url.endswith("/decisions"):
            self.run.recorded(json, r)
        return r


@dataclass
class Journey:
    """One reference journey: the research question, the fixture, and the answers its author gives."""

    name: str
    lens: str  # the packet it belongs to
    purpose: str
    question: str  # the research question, in words
    fixture: str  # the file as the packet names it
    why: str  # why this fixture serves the lens
    path: Callable[[], Path]
    truth: Callable[[], Any]
    target: str
    lenses: tuple[str, ...]
    event: str | None = None
    exposure: str | None = None  # a column, or "family" for every exposure in turn
    roles: dict[str, str] | None = None
    answers: dict[str, Any] = field(default_factory=dict)
    before: dict[str, Callable[["Run", dict[str, Any]], None]] = field(default_factory=dict)
    needs: tuple[str, ...] = ()  # what must exist for it to run (a data file)
    timeout: float = 5400.0
    fixture_key: str = ""  # the fixture's entry in ``truths.FIXTURE_TRUTHS`` (its declared truth)


@dataclass
class Run:
    spec: Journey
    client: Any = None
    drive: Any = None
    pid: str = ""
    source: str = ""  # where the answer being posted came from
    log: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    beside: dict[str, Any] = field(default_factory=dict)  # what a stage holds that the bundle does not

    def note(self, text: str) -> None:
        self.notes.append(text)
        print(f"  [{self.spec.name}] {text}", flush=True)

    def recorded(self, decision: Any, response: Any) -> None:
        kind = (decision or {}).get("kind") if isinstance(decision, dict) else None
        entry: dict[str, Any] = {"kind": kind, "status": response.status_code,
                                 "summary": summarize(decision), "source": self.source_of(kind)}
        if response.status_code == 200:
            records = (response.json() or {}).get("decisions") or []
            if records:
                entry["seq"] = records[-1].get("seq")
                entry["sentence"] = records[-1].get("sentence")
        else:
            try:
                error = (response.json() or {}).get("error") or {}
            except ValueError:
                error = {"code": "unparsed", "message": response.text[:300]}
            if error.get("code") == "not_yet":
                return  # the Router holding a question behind one still computing
            entry["refusal"] = {"code": error.get("code"), "message": error.get("message"),
                                "exits": [e.get("label") for e in error.get("exits") or []]}
        last = self.log[-1] if self.log else None
        if last and last.get("refusal") and entry.get("refusal") and last["kind"] == kind \
                and last["summary"] == entry["summary"] \
                and last["refusal"]["code"] == entry["refusal"]["code"]:
            return  # the same refusal again, while the driver waits
        self.log.append(entry)

    def source_of(self, kind: str | None) -> str:
        if kind in READING_KINDS:
            return READINGS_SOURCE
        # The drivers answer the plan's questions on the way to another (``answer_wp17_before``),
        # whatever answer was being posted: each such kind says where its own answer came from.
        if kind in WP17_SOURCES and not self.source.startswith(OWN_EXITS):
            return WP17_SOURCES[kind]
        return self.source or "the journey"


READING_KINDS = ("confirm_readings", "confirm_reading", "confirm_role", "set_column_unit")
READINGS_SOURCE = ("a reading the server asked about, answered from the journey's readings (the "
                   "fixture's declared truth and the journey's own, listed below), else the app's "
                   "guess (listed below)")
WP17_SOURCE = "the acceptance drivers (server_drive.answer_wp17)"
# Where each answer the drivers post came from (``server_drive.answer_wp17``, with the journey's
# truth and its app-ranked hooks: ``GuessingTruth.get``).
WP17_SOURCES: dict[str, str] = {
    "set_estimand": "the journey's exposure, else the fixture's declared one, else the estimand "
                    "card's first column; the estimand card's first-ranked measure unless the "
                    "readings declare one (listed below); an energy-bearing exposure's contrast "
                    "from the readings, else the substitution",
    "set_adjustment": "each covariate's causal place from the journey's readings (listed below), "
                      "grouped as the adjustment card groups them",
    "set_forms": "the form card's proposal for each column it asks about (a spline, k by the "
                 "card's rule; non-consumers apart for an exposure with a mass at zero), unless "
                 "the readings declare a form (listed below)",
    "set_clusters": "the acceptance drivers: the roles' named grouping, its intervals only under "
                    "inference; none, acknowledged, when nothing reads as one",
    "set_time_varying": "the acceptance drivers: standard regression, the exposure declared to "
                        "precede the outcome (its attestation exit where the question blocks it)",
    "set_censoring": "the same follow-up for everyone: the yes/no outcome counted over one "
                     "period (the journey's answer and the drivers' alike)",
}
# A source the journey set itself while taking a refusal's way forward is kept as it is.
OWN_EXITS = ("the refusal's way forward", "the result's own refusal", "the export's first exit")


def summarize(decision: Any, limit: int = 260) -> str:
    """A decision's payload in a line: a roles answer counted by role, a block confirmation by
    reading, anything else as its JSON without the kind, cut at ``limit`` characters."""
    if not isinstance(decision, dict):
        return ""
    kind = decision.get("kind")
    if kind == "set_roles":
        by: dict[str, list[str]] = {}
        for column, role in (decision.get("roles") or {}).items():
            by.setdefault(str(role), []).append(str(column))
        parts = []
        for role, columns in by.items():
            shown = ", ".join(columns[:6]) + (f" and {len(columns) - 6:,} more" if len(columns) > 6
                                              else "")
            parts.append(f"{role}: {shown}")
        return "; ".join(parts)
    if kind == "confirm_readings":
        items = decision.get("items") or []
        kinds: dict[str, int] = {}
        for i in items:
            kinds[str(i.get("reading"))] = kinds.get(str(i.get("reading")), 0) + 1
        shown = ", ".join(f"{i.get('reading')}:{i.get('column')}={i.get('value')}"
                          for i in items[:4])
        more = f" and {len(items) - 4:,} more" if len(items) > 4 else ""
        noun = "reading" if len(items) == 1 else "readings"
        return f"{len(items):,} {noun} ({', '.join(f'{k} {v}' for k, v in kinds.items())}): " \
               f"{shown}{more}"
    body = {k: v for k, v in decision.items() if k != "kind"}
    text = json.dumps(body, ensure_ascii=False, separators=(", ", ": "))
    return text if len(text) <= limit else text[:limit - 1] + "…"


# ── the app's first-ranked answers ───────────────────────────────────────────


def _artifact(run: Run, stage: str, timeout: float = 1800.0) -> dict[str, Any]:
    return run.drive.artifact(stage, timeout=timeout)


MISSING_BODIES: dict[str, dict[str, Any]] = {
    "multiple_imputation": {"strategy": "multiple_imputation"},
    "complete_case": {"strategy": "complete_case"},
    "impute": {"strategy": "impute"},
    "indicators": {"strategy": "impute", "indicators": True},
    "missing_category": {"strategy": "impute", "categorical": "missing_category"},
}


def first_missing(run: Run) -> dict[str, Any]:
    labels = (_artifact(run, "proposals").get("labels") or {}).get("missing") or {}
    key = ((labels.get("options") or [{}])[0]).get("key") or "complete_case"
    return {"kind": "set_missing", **MISSING_BODIES.get(key, {"strategy": "complete_case"})}


def first_exclusions(run: Run) -> dict[str, Any]:
    proposals = _artifact(run, "proposals")
    labels = (proposals.get("labels") or {}).get("exclusions") or {}
    offered = {e.get("key"): e for e in proposals.get("exclusions") or []}
    for option in labels.get("options") or []:
        key = option.get("key")
        if key == "keep_every_row":
            return {"kind": "set_exclusions", "rules": []}
        offer = offered.get(key)
        # An offer held back only until a reading is confirmed is posted all the same: the server
        # asks for the reading, and the fixture's truth answers it (``post_answer``).
        if offer is not None and offer.get("rule"):
            if offer.get("refused"):
                run.note(f"the {key} screen was offered pending a confirmation: "
                         f"{offer['refused']}")
            return {"kind": "set_exclusions", "rules": [offer["rule"]]}
    return {"kind": "set_exclusions", "rules": []}


def first_energy(run: Run) -> dict[str, Any]:
    proposals = _artifact(run, "proposals")
    reading = proposals.get("energy") or {}
    labels = (proposals.get("labels") or {}).get("energy_adjustment") or {}
    applicability = reading.get("applicability") or {}
    method = "none"
    for option in labels.get("options") or []:
        ok = (applicability.get(option.get("key")) or {}).get("ok", True)
        if ok:
            method = option["key"]
            break
    body: dict[str, Any] = {"kind": "set_energy_adjustment", "method": method}
    if method != "none":
        body.update({"energy_column": reading.get("energy_column"),
                     "nutrients": reading.get("nutrients") or []})
    return body


def first_split(run: Run) -> dict[str, Any]:
    plan = _artifact(run, "seal_plan")
    options = plan.get("options") or []
    holdout = float(options[0]["holdout"]) if options else 0.0
    validation = ((plan.get("validation") or {}).get("options") or [{}])[0]
    body: dict[str, Any] = {"kind": "set_split", "holdout": holdout, "seed": 0, "folds": 5,
                            "validation": validation.get("validation") or "kfold"}
    if body["validation"] == "internal_external":
        body["cluster"] = validation.get("cluster")
    return body


def first_models(run: Run, view: dict[str, Any]) -> dict[str, Any]:
    """The shelf's first-ranked family; under prediction its first two that predict, compared."""
    from turbotab.core.models.base import get_family

    shelf = _artifact(run, "shelf")
    ranked = sorted(shelf.get("families") or [], key=lambda f: f.get("rank", 0))
    if view["state"].get("purpose") == "prediction":
        predicting = [f["key"] for f in ranked if getattr(get_family(f["key"]), "predicts", True)]
        return {"kind": "select_models", "models": predicting[:2] or [ranked[0]["key"]]}
    return {"kind": "select_models", "models": [ranked[0]["key"]]}


def answer_for(run: Run, key: str, step: dict[str, Any],
               view: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The answer to the open question ``key`` and where it came from (module docstring)."""
    from turbotab.core.tests.acceptance.server_drive import WP17_QUESTIONS, answer_wp17

    spec, d, state = run.spec, run.drive, view["state"]
    target = state.get("target")
    given = spec.answers.get(key)
    if callable(given):
        return given(run, step, view), "the journey"
    if given is not None:
        return given, "the journey"
    if key == "lens":
        return {"kind": "set_lens", "lenses": list(spec.lenses)}, "the journey"
    if key == "orientation":
        return {"kind": "set_orientation", "orientation": "sample_major"}, "the fixture's layout"
    if key == "target":
        return {"kind": "set_target", "column": spec.target}, "the journey"
    if key == "event":
        return {"kind": "set_event", "column": target, "level": str(spec.event)}, "the journey"
    if key == "task":
        if step.get("followup"):
            run.source = "the fixture's truth, else the usual (the original scale; the words' order)"
            d.task_followups()
            return None, ""
        ti = _artifact(run, "target_info")
        return {"kind": "set_task", "column": target, "task": ti["task"]}, "the app's detection"
    if key == "follow_up":
        task = state.get("task") or _artifact(run, "target_info").get("task")
        if task == "time_to_event":
            raise RuntimeError("a time-to-event journey names its follow-up in its spec")
        return {"kind": "set_censoring", "column": target}, WP17_SOURCES["set_censoring"]
    if key == "purpose":
        return {"kind": "set_purpose", "purpose": spec.purpose}, "the journey"
    if key == "grain":
        return {"kind": "set_grain", "grain": "one_row_per_unit"}, "the fixture's layout"
    if key == "repeat_kind":
        structure = _artifact(run, "structure")
        reading = (structure.get("repeats") or {}).get("reading") or "repeats"
        return ({"kind": "set_repeat_kind", "repeat_kind": reading,
                 "time_column": structure.get("time_column") if reading == "time_points" else None},
                "the app's reading of the structure")
    if key == "unit":
        return {"kind": "set_unit", "unit": "unit"}, "the journey"
    if key == "aggregation":
        menu = _artifact(run, "structure").get("aggregation") or {}
        return ({"kind": "set_aggregation", "method": menu.get("recommended") or "mean"},
                "the app's first-ranked option")
    if key == "temporal":
        return {"kind": "set_temporal", "temporal": False}, "the journey"
    if key == "roles":
        roles = spec.roles
        run.source = "the journey (the fixture's author)" if roles else \
            "the app's proposed roles, confirmed"
        if roles is None:
            proposals = _artifact(run, "roles")
            roles = {c["column"]: c["proposed"] for c in proposals["columns"]}
        d.decide_roles(dict(roles))
        return None, ""
    if key in WP17_QUESTIONS:
        run.source = WP17_SOURCE  # each kind it posts says where its answer came from
        answer_wp17(d, key, exposure=spec.exposure)
        return None, ""
    if key == "survey":
        survey = _artifact(run, "proposals").get("survey") or {}
        option = (survey.get("options") or [{}])[0].get("decision")
        return (option or {"kind": "set_survey", "estimand": "sample"},
                "the app's first-ranked option")
    if key == "exclusions":
        return first_exclusions(run), "the app's first-ranked option"
    if key == "missing":
        return first_missing(run), "the app's first-ranked option"
    if key == "split":
        return first_split(run), "the app's first-ranked option (the seal plan)"
    if key == "energy_adjustment":
        return first_energy(run), "the app's first-ranked applicable option"
    if key == "causal":
        return ({"kind": "set_causal", "exposure": (state.get("estimand") or {}).get("exposure"),
                 "method": "none"}, "the primary model alone (the question's stated default)")
    if key == "models":
        return first_models(run, view), "the shelf's first-ranked families"
    if key == "substitution":
        pairs = _artifact(run, "design").get("substitution_pairs") or []
        if not pairs:
            raise RuntimeError("the substitution question is open with no pair to draw")
        pair = pairs[0]
        return ({"kind": "set_substitution", "donor": pair["donor"],
                 "recipient": pair["recipient"], "step_kcal": 100},
                "the first pair the design offers, 100 kcal a step")
    if key == "open_seal":
        selected = state.get("models") or []
        family = selected[0] if selected else None
        return ({"kind": "open_seal", "family": family},
                "the shelf's first-ranked family, declared final")
    raise RuntimeError(f"no answer for the open question {key!r}")


# ── following the Router ─────────────────────────────────────────────────────


def _settle(run: Run, body: dict[str, Any]) -> Any:
    from turbotab.core.tests.acceptance.server_drive import settle_post

    d = run.drive
    return settle_post(d.c, d.pid, body, d.truth, unblock=lambda: d.answer_wp17_before(body))


def _exits(r: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    error = (r.json() or {}).get("error") or {} if r.status_code == 409 else {}
    return error, [e for e in error.get("exits") or [] if e.get("decision")]


def post_answer(run: Run, body: dict[str, Any], depth: int = 0) -> None:
    """Post ``body``; readings are answered from the truth (``settle_post``). Any other refusal
    is answered as a person working through it would: its ways forward are tried in the order it
    offers them, and the first one accepted is taken; when every one is refused in turn, the
    first one's own ways forward are followed the same way. Each exit taken is noted."""
    r = _settle(run, body)
    if r.status_code == 200:
        return
    error, exits = _exits(r)
    if not exits or depth >= 3:
        raise RuntimeError(f"{body.get('kind')} refused ({error.get('code')}): "
                           f"{error.get('message') or r.text[:600]}")
    for i, e in enumerate(exits):
        run.source = f"the refusal's way forward “{e['label']}”"
        if _settle(run, e["decision"]).status_code == 200:
            run.note(f"{body.get('kind')} refused ({error.get('code')}); took its "
                     f"{'first' if i == 0 else f'way forward {i + 1}'} “{e['label']}”")
            return
    run.note(f"{body.get('kind')} refused ({error.get('code')}), and each way forward was "
             f"refused in turn; following the first one's own")
    run.source = f"the refusal's way forward “{exits[0]['label']}”"
    post_answer(run, exits[0]["decision"], depth + 1)


def follow(run: Run) -> bool:
    """Answer the Router's open questions until none is open or waiting. True when it got there."""
    spec, d = run.spec, run.drive
    end = run.started + spec.timeout
    done_before: set[str] = set()
    while time.monotonic() < end:
        view = d.view()
        first = next((s for s in view["interview"] if s["status"] in OPEN), None)
        if first is None:
            return True
        key = first["key"]
        if first["status"] == "waiting":
            failed = [w for w in first.get("waiting_on") or []
                      if view["stages"].get(w, {}).get("status") == "error"]
            if failed:
                errors = {w: view["stages"][w].get("error") for w in failed}
                run.note(f"stopped: the {key} question waits on failed stages {errors}")
                return False
            time.sleep(0.3)
            continue
        if key in spec.before and key not in done_before:
            done_before.add(key)
            run.source = "the journey (declared before this question)"
            spec.before[key](run, view)
            continue
        body, source = answer_for(run, key, first, view)
        if body is not None:
            run.source = source
            print(f"  [{spec.name}] {key}: {summarize(body, 120)}", flush=True)
            post_answer(run, body)
    run.note("stopped: the journey ran out of time")
    return False


def wait_results(run: Run, timeout: float = 3600.0) -> bool:
    from turbotab.core.export.gate import result_stages
    from turbotab.core.decisions import ProjectState

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        view = run.drive.view()
        stages = result_stages(ProjectState(**view["state"]))
        statuses = {s: view["stages"][s]["status"] for s in stages}
        failed = {s: view["stages"][s].get("error") for s, v in statuses.items() if v == "error"}
        if failed:
            run.note(f"stopped: a result stage failed: {failed}")
            return False
        if all(v == "fresh" for v in statuses.values()):
            return True
        if any(v == "blocked" for v in statuses.values()):
            blocked = {s: view["stages"][s].get("missing") for s, v in statuses.items()
                       if v == "blocked"}
            run.note(f"a result stage is blocked: {blocked}")
            return False
        time.sleep(0.5)
    run.note("stopped: the results never finished computing")
    return False


def refusals_in(artifact: Any) -> list[tuple[str, list[dict[str, Any]]]]:
    """Every refusal a served artifact carries inside it (``refused`` with ``exits``), with the
    ways forward that carry a decision, in the artifact's order."""
    out: list[tuple[str, list[dict[str, Any]]]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            exits = [e for e in node.get("exits") or [] if isinstance(e, dict) and e.get("decision")]
            if node.get("refused") and exits:
                out.append((str(node["refused"]), exits))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(artifact)
    return out


def export(run: Run) -> Any:
    """The bundle, taking the export's own ways forward: under inference Fit pressed, which locks
    the plan (SIZING P0.8); under prediction Fit pressed, and the final model declared and the
    held-out rows opened (its first exit)."""
    d = run.drive
    url = f"/api/projects/{run.pid}/export"
    r = d.c.get(url)
    presses = 0
    for _ in range(6):
        if r.status_code == 200:
            return r
        error = (r.json() or {}).get("error") or {}
        code = error.get("code")
        if presses < 2 and (code in ("plan_open", "estimates_withheld")
                            and any("Press Fit" in str(e.get("label")) for e in
                                    error.get("exits") or [])):
            # Pressed again after a way forward a refused result offered: that change to the plan
            # withdrew the lock no estimate had been shown under (calm/FOUNDATION §7).
            presses += 1
            taken = d.c.post(f"/api/projects/{run.pid}/fit")
            run.note(f"the export waited for Fit: pressed ({taken.status_code})")
            if taken.status_code == 200 and d.view()["state"].get("purpose") == "inference":
                from turbotab.core.estimand import ESTIMATE_STAGES
                from turbotab.core.plan_lock import shows_estimates

                view = d.view()
                refusals, shown = [], False
                for stage in ("effects", *ESTIMATE_STAGES):
                    if view["stages"].get(stage, {}).get("status") == "fresh":
                        served = d.c.get(f"/api/projects/{run.pid}/stages/{stage}").json() or {}
                        refusals += refusals_in(served.get("artifact"))
                        shown = shown or shows_estimates(stage, served.get("artifact"))
                if refusals and not shown:
                    # No estimate to show: a result refused inside its artifact, with its ways
                    # forward (a family that cannot pool imputations), tried in order until one
                    # leaves every result computed.
                    message, exits = refusals[0]
                    run.note(f"a result was refused inside its artifact (“{message}”)")
                    for e in exits:
                        run.source = f"the result's own refusal, its way forward “{e['label']}”"
                        post_answer(run, e["decision"])
                        if wait_results(run):
                            run.note(f"took the result's way forward “{e['label']}”")
                            break
                        run.note(f"the way forward “{e['label']}” left a result uncomputed; "
                                 f"trying the next")
        elif code == "plan_open" and any(e.get("decision") for e in error.get("exits") or []):
            chosen = next(e for e in error["exits"] if e.get("decision"))
            run.note(f"the export waited for the final model: took “{chosen['label']}”")
            run.source = "the export's first exit"
            post_answer(run, chosen["decision"])
            wait_results(run)
        elif code == "result_not_ready":
            time.sleep(2.0)
        else:
            run.note(f"the export refused ({code}): {error.get('message')}")
            return r
        r = d.c.get(url)
    return r


# ── the capture ──────────────────────────────────────────────────────────────


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=10", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _scrub(text: str, *folders: Path) -> str:
    for folder in folders:
        text = text.replace(str(folder), "<folder>")
    return text


def capture_of(run: Run, response: Any, home: Path, bundles: Path | None) -> dict[str, Any]:
    spec = run.spec
    out: dict[str, Any] = {
        "journey": {"name": spec.name, "lens": spec.lens, "purpose": spec.purpose,
                    "question": spec.question, "fixture": spec.fixture, "why": spec.why,
                    "lenses": list(spec.lenses), "target": spec.target,
                    "exposure": spec.exposure},
        "captured": {"commit": _git_head(), "date": time.strftime("%Y-%m-%d"),
                     "seconds": round(time.monotonic() - run.started),
                     "workers": 2},
        "answers": run.log,
        "readings": journey_readings(spec),
        "guessed": list(getattr(run.drive.truth, "guessed", []) if run.drive else []),
        "app_ranked": list(getattr(run.drive.truth, "app_ranked", []) if run.drive else []),
        "notes": run.notes,
        "export": {"status": getattr(response, "status_code", None)},
    }
    if run.beside:
        out["beside_the_bundle"] = run.beside
    if response is None:
        return out
    if response.status_code != 200:
        try:
            error = (response.json() or {}).get("error") or {}
        except ValueError:
            error = {"message": response.text[:600]}
        out["export"]["refusal"] = {"code": error.get("code"), "message": error.get("message"),
                                    "exits": [e.get("label") for e in error.get("exits") or []]}
        return out
    if bundles is not None:
        bundles.mkdir(parents=True, exist_ok=True)
        (bundles / f"{spec.name}.zip").write_bytes(response.content)
    try:
        out["export"].update(bundle_parts(response.content, home))
    except Exception as exc:  # noqa: BLE001 - the run's answers are kept whatever the bundle holds
        run.note(f"the bundle could not be read: {type(exc).__name__}: {exc}")
    return out


def journey_readings(spec: Journey) -> dict[str, Any]:
    """The readings a journey answers from, apart: those of the fixture's declared truth it uses
    (``truths.FIXTURE_TRUTHS``), the journey's own (beside or in place of them: the causal
    assumptions of its research question, units and nesting), and the declared ones it sets aside."""
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    declared = {k: str(v) for k, v in FIXTURE_TRUTHS.get(spec.fixture_key, {}).items()}
    used = {k: str(v) for k, v in dict(spec.truth()).items()}
    return {"fixture": spec.fixture_key,
            "from_fixture": {k: v for k, v in used.items() if declared.get(k) == v},
            "journey_own": {k: v for k, v in used.items() if declared.get(k) != v},
            "set_aside": sorted(k for k in declared if k not in used)}


def beside_the_bundle(run: Run) -> dict[str, Any]:
    """What a stage reports that the export bundle does not hold, kept so the packet can show it:
    the scales stage's reliability and corrected coefficient of each declared scale."""
    out: dict[str, Any] = {}
    if not (run.drive.view()["state"].get("scales") or []):
        return out
    served = run.drive.c.get(f"/api/projects/{run.pid}/stages/scales").json() or {}
    art = served.get("artifact") or {}
    if not art:
        return out
    keep_r = ("coefficient", "label", "value", "alpha", "alpha_label", "n", "reason")
    keep_c = ("feature", "scale", "naive", "naive_ci_low", "naive_ci_high", "estimate", "ci_low",
              "ci_high", "naive_ratio", "ratio", "ratio_low", "ratio_high", "attenuation", "n",
              "n_boot", "copies")
    out["scales"] = {
        "methods": art.get("methods"),
        "scales": [{"name": sc.get("name"), "role": sc.get("role"),
                    "reliability": {k: (sc.get("reliability") or {}).get(k) for k in keep_r},
                    "correction": ({k: sc["correction"].get(k) for k in keep_c}
                                   if sc.get("correction") else None),
                    "not_corrected": sc.get("not_corrected"), "methods": sc.get("methods")}
                   for sc in art.get("scales") or []]}
    return out


def _table2_rows(files: dict[str, bytes]) -> list[dict[str, Any]]:
    """Each declared model of Table 2: the model, what it is adjusted for, and its terms (the
    exposure's rows: one, a spline's basis, or each member of an exposure family), in order."""
    import csv

    raw = files.get("results/table2.csv")
    if raw is None:
        return []
    out: dict[tuple[str, str], list[str]] = {}
    for r in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
        terms = out.setdefault((r.get("model", ""), r.get("adjusted_for", "")), [])
        if r.get("term", "") not in terms:
            terms.append(r.get("term", ""))
    return [{"model": m, "adjusted_for": a, "terms": t} for (m, a), t in out.items()]


def bundle_parts(data: bytes, home: Path) -> dict[str, Any]:
    """What a capture keeps of an export bundle: the engine, the file list, the methods section
    verbatim, the checklist item by item (its quoted item texts left to the bundle), the model
    matrix's columns and each declared model's row of Table 2 (what the packet checks the methods'
    adjustment set against)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        files = {n.split("/", 1)[1]: z.read(n) for n in z.namelist() if not n.endswith("/")}
    provenance = json.loads(files["provenance.json"])
    stem = next(n for n in files if n.startswith("checklist/") and n.endswith(".json"))
    report = json.loads(files[stem])
    return {
        "engine": (provenance.get("engine") or {}).get("turbotab"),
        "files": sorted(files),
        "model_matrix": list((provenance.get("model_matrix") or {}).get("columns") or []),
        "table2": _table2_rows(files),
        "methods_md": _scrub(files["methods.md"].decode("utf-8"), home),
        "checklist": {
            "checklist": report["checklist"], "title": report.get("title"),
            "citation": report.get("citation"), "counts": report["counts"],
            "items": [{"id": i["id"], "section": i["section"], "topic": i["topic"],
                       "kind": i.get("kind"), "scope": i.get("scope"), "status": i["status"],
                       "where": [{"source": w["source"], "file": w["file"], "kind": w.get("kind"),
                                  "seq": w.get("seq")} for w in i.get("where") or []],
                       "owed": i.get("owed")} for i in report["items"]],
            "unanswered": report.get("unanswered") or [],
        },
    }


def run_journey(spec: Journey, bundles: Path | None = None,
                out_dir: Path = CAPTURES) -> dict[str, Any]:
    """Run ``spec`` once in a fresh home and write its capture; returns the capture."""
    from turbotab.core.tests.acceptance.server_drive import Drive, local_server

    run = Run(spec)
    print(f"── {spec.name}: {spec.question}", flush=True)
    response = None
    with tempfile.TemporaryDirectory(prefix=f"tt-ref-{spec.name}-") as folder:
        home = Path(folder) / "home"
        missing = [n for n in spec.needs if not Path(n).exists()]
        if missing:
            run.note(f"not run: {', '.join(missing)} is not here")
        else:
            try:
                path = spec.path()
                with local_server(home) as client:
                    run.client = Recording(client, run)
                    r = client.post("/api/projects", json={"path": str(path)})
                    r.raise_for_status()
                    run.pid = r.json()["id"]
                    t = spec.truth()
                    t.guess = _server_guess(client, run.pid)
                    t.ranked = _server_ranked(client, run.pid)
                    run.drive = Drive(run.client, run.pid, t)
                    run.drive.exposure = spec.exposure
                    run.drive.artifact("ingest", timeout=900)
                    if follow(run) and wait_results(run):
                        run.source = "the export"
                        response = export(run)
                        if getattr(response, "status_code", None) == 200:
                            run.beside = beside_the_bundle(run)
                    else:
                        response = run.client.get(f"/api/projects/{run.pid}/export")
            except Exception as exc:  # noqa: BLE001 - a journey that stops is kept with its reason
                run.note(f"stopped by {type(exc).__name__}: {str(exc)[:900]}")
                traceback.print_exc()
        out = capture_of(run, response, Path(folder), bundles)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path} (export {out['export'].get('status')}, {out['captured']['seconds']} s)",
          flush=True)
    return out


def load_capture(name: str, folder: Path = CAPTURES) -> dict[str, Any] | None:
    path = folder / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# ── the journeys ─────────────────────────────────────────────────────────────

NUTRIENTS = ["protein", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
NHANES_KEY = "_tt_tmp_nhanes.csv"  # the NHANES export's entry in ``truths.FIXTURE_TRUTHS``
NHANES_FLAGS = ["imputed_weight", "imputed_height", "imputed_bmi", "imputed_waist",
                "imputed_bp_sys", "imputed_bp_di"]


def nhanes_path() -> Path:
    """The NHANES export: the tracked fixture, decompressed (``stage_harness.NHANES``)."""
    from turbotab.core.tests.stage_harness import NHANES

    return NHANES


def nhanes_readings() -> dict[str, str]:
    """The NHANES export's declared truth (``truths.FIXTURE_TRUTHS``), with the units the NHANES
    documentation states (every DR1T* macronutrient and sugar total in grams) and the nesting of
    sugars in carbohydrate and the fatty acids in total fat."""
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    return {**FIXTURE_TRUTHS["_tt_tmp_nhanes.csv"],
            **{f"unit:{c}": "g" for c in ("sugar", *NUTRIENTS)},
            "nested_in:sugar": "carb",
            **{f"nested_in:{c}": "fat_total" for c in ("fat_sat", "fat_mon", "fat_poly")}}


def nhanes_dietary_truth() -> Any:
    return truth(nhanes_readings(), fixture="_tt_tmp_nhanes.csv")


def nhanes_clinical_truth() -> Any:
    """Waist circumference and fasting glucose: age, sex and the survey cycle come before body
    shape and cause both (confounders); blood pressure, HDL, triglycerides and their medications
    follow central adiposity and track glucose (downstream, not adjusted)."""
    readings = {k: v for k, v in nhanes_readings().items() if not k.startswith(("adjust:",
                                                                               "exposure:",
                                                                               "contrast:"))}
    readings.update({"exposure:glucose": "waist",
                     **{f"adjust:{c}": "yes,yes,no" for c in ("age", "gender",
                                                              "cycle_begin_year")},
                     **{f"adjust:{c}": "no,yes,yes" for c in ("bp_sys", "bp_di", "hdl",
                                                              "triglycerides", "meds_hbp",
                                                              "meds_chol")}})
    return truth(readings, fixture="_tt_tmp_nhanes.csv (clinical question)")


NHANES_DIETARY_ROLES = {
    "SEQN": "identifier", "sugar": "exposure", "kcal": "energy",
    **{c: "exposure" for c in NUTRIENTS},
    **{c: "covariate" for c in ("age", "gender", "cycle_begin_year", "weight", "height", "bmi",
                                "waist", "bp_sys", "bp_di", "hdl", "triglycerides", "meds_hbp",
                                "meds_chol")},
    **{c: "flag" for c in NHANES_FLAGS},
}
NHANES_CLINICAL_ROLES = {
    "SEQN": "identifier", "waist": "exposure",
    **{c: "covariate" for c in ("age", "gender", "cycle_begin_year", "bp_sys", "bp_di", "hdl",
                                "triglycerides", "meds_hbp", "meds_chol")},
    **{c: "excluded" for c in ("weight", "height", "bmi", "kcal", "sugar", *NUTRIENTS)},
    **{c: "flag" for c in NHANES_FLAGS},
}


def apply_first_repair(pattern: str) -> Callable[[Run, dict[str, Any]], None]:
    """Before the outcome question: apply the first repair option of every finding whose id
    matches ``pattern`` (the repairs come before the outcome, OPENING_SEQUENCE §01)."""

    def go(run: Run, view: dict[str, Any]) -> None:
        findings = _artifact(run, "findings")["findings"]
        with_repairs = [f for f in findings if f.get("repairs")]
        run.note("findings with repairs: " + (", ".join(f["id"] for f in with_repairs) or "none"))
        for f in with_repairs:
            if not re.search(pattern, f["id"], re.I):
                continue
            option = f["repairs"][0]
            run.source = "the finding's first repair option"
            post_answer(run, {"kind": "apply_repair", "finding_id": f["id"],
                              "option": option["key"]})
    return go


def declare_model_sequence(run: Run, view: dict[str, Any]) -> None:
    """Under inference, before the models: the declared model sequence the proposals card offers
    (Table 2: the crude model, Model 1, the primary), as a user taking the card's offer does."""
    end = time.monotonic() + 600
    card = None
    while time.monotonic() < end:
        card = _artifact(run, "proposals").get("model_sequence")
        if card is not None:
            break
        time.sleep(0.5)
    if card is None or not card.get("decision"):
        run.note("no model sequence was proposed")
        return
    run.source = "the app's proposed model sequence"
    post_answer(run, card["decision"])


def declare(*bodies: dict[str, Any]) -> Callable[[Run, dict[str, Any]], None]:
    """A hook that declares what no Router question asks (a scale, an explanation)."""

    def go(run: Run, view: dict[str, Any]) -> None:
        for body in bodies:
            run.source = "the journey (declared, not asked)"
            try:
                post_answer(run, body)
            except Exception as exc:  # noqa: BLE001 - noted; the journey goes on
                run.note(f"{body['kind']} not recorded: {str(exc)[:400]}")
    return go


def declare_batch(column: str) -> Callable[[Run, dict[str, Any]], None]:
    """Batch handling, which no Router question asks: the batch contract's first-ranked option for
    the declared purpose (``contracts.contract("batch")``), as a user taking the app's order does."""

    def go(run: Run, view: dict[str, Any]) -> None:
        from turbotab.core.contracts import contract

        purpose = view["state"].get("purpose") or run.spec.purpose
        method = contract("batch").options_for(purpose)[0]["key"]
        run.source = "the batch contract's first-ranked option for the purpose (declared, not asked)"
        try:
            post_answer(run, {"kind": "set_batch", "column": column, "method": method})
        except Exception as exc:  # noqa: BLE001 - noted; the journey goes on
            run.note(f"set_batch not recorded: {str(exc)[:400]}")
    return go


def chain(*hooks: Callable[[Run, dict[str, Any]], None]) -> Callable[[Run, dict[str, Any]], None]:
    def go(run: Run, view: dict[str, Any]) -> None:
        for hook in hooks:
            hook(run, run.drive.view())
    return go


def sample(name: str) -> Callable[[], Path]:
    return lambda: SAMPLES / name


def sample_truth(name: str, extra: dict[str, str] | None = None) -> Callable[[], Any]:
    def make() -> Any:
        from turbotab.core.tests.truths import FIXTURE_TRUTHS

        return truth({**FIXTURE_TRUTHS.get(name, {}), **(extra or {})}, fixture=name)
    return make


METABOLOMICS_READINGS = {"code_or_count:age": "amount", "code_or_count:run_order": "amount",
                         "code_or_count:batch": "code", "code_or_count:responder": "code",
                         "adjust:age": "yes,yes,no", "adjust:sex": "yes,yes,no",
                         "adjust:bmi": "unknown,yes,unknown", "exposure:responder": "family",
                         # the injection order and the batch move the measured intensities, not
                         # who responds: causes of the exposure only
                         "adjust:run_order": "yes,no,no", "adjust:batch": "yes,no,no"}
# Library size varies with batch, and the cases are balanced within batch (the data card): batch
# moves the measured counts, not case status.
GENOMICS_READINGS = {"code_or_count:age": "amount", "code_or_count:batch": "code",
                     "adjust:age": "yes,yes,no", "adjust:sex": "yes,yes,no",
                     "adjust:batch": "yes,no,no", "exposure:condition": "family"}
METABOLOMICS_REPAIRS = r"pooled_qc|^omics_scale$|drift"
QC_TOO_SPARSE = ("Its pooled QCs are too sparse for drift correction (four per batch, and nine "
                 "study injections after the last QC), so the app refuses QC-RLSC both per batch "
                 "and as one curve, and the journey sets the QC rows aside uncorrected: the "
                 "drift-correction sentence is not exercised here.")
GENOMICS_REPAIRS = r"^omics_scale$"
SURVEY_ITEMS = [f"item_{i:02d}" for i in range(1, 11)]
SURVEY_SCALE = {"name": "support_scale", "items": SURVEY_ITEMS, "reverse": ["item_05"],
                "low": 1, "high": 5, "kind": "reflective"}
# The plan's exposure is the estimand card's first column, age (the survey inference journey): no
# respondent's characteristic and no item of the scale can cause it, and none comes after it in
# time; each is read as a cause of the outcome (the instrument measures a trait that moves who
# seeks support), so the disjunctive cause criterion adjusts for it.
SURVEY_READINGS = {"code_or_count:education": "code",
                   **{f"adjust:{c}": "no,yes,no" for c in ("sex", "education", *SURVEY_ITEMS)}}
# Under inference the instrument's first ten items are the scale's, every other item is left out,
# and the respondents' characteristics are covariates (MODELING_SEQUENCE §6 chain 4, as its
# acceptance test drives it: test_ms8_scales.chain4).
SURVEY_INFERENCE_ROLES = {"respondent_id": "identifier",
                          **{c: "covariate" for c in ("age", "sex", "education", *SURVEY_ITEMS)},
                          **{f"item_{i:02d}": "excluded" for i in range(11, 41)}}

NHANES_FILE = ("_tt_tmp_nhanes.csv (the NHANES export, tracked gzipped as "
               "turbotab/core/tests/fixtures/nhanes.csv.gz)")
NHANES_WHY = ("V2_DEFINITION_OF_DONE §1: the NHANES export runs the dietary and clinical reference "
              "journeys.")

JOURNEYS: dict[str, Journey] = {j.name: j for j in (
    Journey(
        "dietary-inference", "dietary", "inference",
        "Is a higher sugar intake, at the same total energy, associated with fasting glucose?",
        NHANES_FILE, NHANES_WHY,
        nhanes_path, nhanes_dietary_truth, target="glucose", lenses=("dietary",),
        exposure="sugar", roles=NHANES_DIETARY_ROLES,
        before={"target": apply_first_repair(r"^sas_zeros"), "models": declare_model_sequence},
        needs=(str(nhanes_path()),), fixture_key=NHANES_KEY),
    Journey(
        "dietary-prediction", "dietary", "prediction",
        "How well do diet and body measures predict fasting glucose?",
        NHANES_FILE, NHANES_WHY,
        nhanes_path, nhanes_dietary_truth, target="glucose", lenses=("dietary",),
        before={"target": apply_first_repair(r"^sas_zeros")},
        needs=(str(nhanes_path()),), fixture_key=NHANES_KEY),
    Journey(
        "clinical-inference", "clinical", "inference",
        "Is a larger waist circumference associated with fasting glucose?",
        NHANES_FILE, NHANES_WHY,
        nhanes_path, nhanes_clinical_truth, target="glucose", lenses=("clinical",),
        exposure="waist", roles=NHANES_CLINICAL_ROLES,
        before={"target": apply_first_repair(r"^sas_zeros"), "models": declare_model_sequence},
        needs=(str(nhanes_path()),), fixture_key=NHANES_KEY),
    Journey(
        "clinical-prediction", "clinical", "prediction",
        "Which patients' disease will have progressed at a visit, from their visits so far?",
        "turbotab/sample_data/clinical_longitudinal.csv",
        "The clinical rows of V2_DEFINITION_OF_DONE §2 that the NHANES export cannot exercise: "
        "impossible values repaired, visits kept as rows, and the temporal seal (the data card's "
        "200 people × 3 scheduled visits).",
        sample("clinical_longitudinal.csv"), sample_truth("clinical_longitudinal.csv"),
        target="progressed", lenses=("clinical",), event="1",
        answers={"grain": {"kind": "set_grain", "grain": "repeated", "id_column": "subject_id"},
                 "unit": {"kind": "set_unit", "unit": "row"},
                 "temporal": {"kind": "set_temporal", "temporal": True,
                              "time_column": "visit_date"}},
        before={"target": apply_first_repair(r".")}, fixture_key="clinical_longitudinal.csv"),
    Journey(
        "metabolomics-prediction", "metabolomics", "prediction",
        "How well does an untargeted metabolite panel predict who responds?",
        "turbotab/sample_data/metabolomics_untargeted.csv",
        "MODELING_SEQUENCE §6 chain 3: pooled QCs, run-order drift and detection-limit blanks "
        "(72 participants, 8 pooled QCs, 392 features). " + QC_TOO_SPARSE,
        sample("metabolomics_untargeted.csv"),
        lambda: truth(METABOLOMICS_READINGS, fixture="metabolomics_untargeted.csv"),
        target="responder", lenses=("metabolomics",), event="1",
        before={"target": apply_first_repair(METABOLOMICS_REPAIRS),
                "models": declare_batch("batch")}, fixture_key="metabolomics_untargeted.csv"),
    Journey(
        "metabolomics-inference", "metabolomics", "inference",
        "Which metabolites differ between responders and non-responders?",
        "turbotab/sample_data/metabolomics_untargeted.csv",
        "MODELING_SEQUENCE §6 chain 3's inference twin: every feature tested on its own, the "
        "false-discovery rate held across them. " + QC_TOO_SPARSE,
        sample("metabolomics_untargeted.csv"),
        lambda: truth(METABOLOMICS_READINGS, fixture="metabolomics_untargeted.csv"),
        target="responder", lenses=("metabolomics",), event="1", exposure="family",
        before={"target": apply_first_repair(METABOLOMICS_REPAIRS),
                "models": declare_batch("batch")}, fixture_key="metabolomics_untargeted.csv"),
    Journey(
        "genomics-prediction", "genomics", "prediction",
        "How well does an expression count matrix predict case status?",
        "turbotab/sample_data/genomics_expression.csv",
        "MODELING_SEQUENCE §6 chain 5: p ≫ n raw counts in three batches (60 samples, 495 "
        "genes).",
        sample("genomics_expression.csv"),
        lambda: truth(GENOMICS_READINGS, fixture="genomics_expression.csv"),
        target="condition", lenses=("genomics",), event="case",
        before={"target": apply_first_repair(GENOMICS_REPAIRS),
                "models": declare_batch("batch")}, fixture_key="genomics_expression.csv"),
    Journey(
        "genomics-inference", "genomics", "inference",
        "Which genes are differentially expressed between cases and controls?",
        "turbotab/sample_data/genomics_expression.csv",
        "MODELING_SEQUENCE §6 chain 5's inference twin: feature-wise models with "
        "Benjamini–Hochberg.",
        sample("genomics_expression.csv"),
        lambda: truth(GENOMICS_READINGS, fixture="genomics_expression.csv"),
        target="condition", lenses=("genomics",), event="case", exposure="family",
        before={"target": apply_first_repair(GENOMICS_REPAIRS),
                "models": declare_batch("batch")}, fixture_key="genomics_expression.csv"),
    Journey(
        "survey-prediction", "survey", "prediction",
        "How well do a support scale and its respondents' characteristics predict who sought "
        "support?",
        "turbotab/sample_data/survey_instrument.csv",
        "MODELING_SEQUENCE §6 chain 4: a reflective scale scored from its items (the "
        "instrument's key reverse-codes item_05), its reliability stated (300 respondents, 40 "
        "five-point items).",
        sample("survey_instrument.csv"), sample_truth("survey_instrument.csv", SURVEY_READINGS),
        target="sought_support", lenses=("survey",), event="1",
        before={"models": declare({"kind": "set_scales",
                                   "scales": [{**SURVEY_SCALE, "role": "covariate"}]})},
        fixture_key="survey_instrument.csv"),
    Journey(
        "survey-inference", "survey", "inference",
        "Is age associated with having sought support, with the support scale scored from its "
        "items in the model beside it?",
        "turbotab/sample_data/survey_instrument.csv",
        "MODELING_SEQUENCE §6 chain 4 under inference, as its acceptance test drives it "
        "(test_ms8_scales.chain4): the scale scored from its ten items (the instrument's key "
        "reverse-codes item_05) and declared an exposure, its reliability stated, and its "
        "coefficient corrected for measurement error by the scales stage. The app cannot make a "
        "declared scale the estimand's exposure (§1.5, `scale_as_exposure`): the estimand card "
        "offers the table's columns, so the plan's exposure is its first, `age`, and the scale "
        "enters the model beside it. The scales stage's corrected coefficient is not in the "
        "export bundle; this packet shows it beside the bundle.",
        sample("survey_instrument.csv"), sample_truth("survey_instrument.csv", SURVEY_READINGS),
        target="sought_support", lenses=("survey",), event="1",
        roles=SURVEY_INFERENCE_ROLES,
        before={"estimand": declare({"kind": "set_scales", "scales": [{
            **SURVEY_SCALE, "role": "exposure", "correction": "regression_calibration",
            "n_boot": 100}]})},
        fixture_key="survey_instrument.csv"),
)}


def for_lens(lens: str) -> list[Journey]:
    return [j for j in JOURNEYS.values() if j.lens == lens]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="journeys to run (default: every journey)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--bundles", type=Path, default=None,
                        help="keep each export bundle here (default: not kept)")
    parser.add_argument("--out", type=Path, default=CAPTURES)
    args = parser.parse_args(argv)
    if args.list:
        for j in JOURNEYS.values():
            print(f"{j.name:26} {j.purpose:10} {j.fixture}")
        return 0
    os.environ.setdefault("TURBOTAB_WORKERS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    names = args.names or list(JOURNEYS)
    unknown = [n for n in names if n not in JOURNEYS]
    if unknown:
        parser.error(f"unknown journeys: {', '.join(unknown)}")
    for name in names:
        run_journey(JOURNEYS[name], bundles=args.bundles, out_dir=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
