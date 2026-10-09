"""Driving a project through the real server (FastAPI app over its job runner), as the audit did.

Shared by the WP12 acceptance tests: :class:`Drive` answers the opening sequence in the Router's
order and waits for a stage's artifact; :func:`local_server` is a local-mode server with two
workers over a fresh home. A reading the server asks about is answered from the fixture's declared
:class:`Truth`, never a constant (BLUEPRINT §14.3).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd

from turbotab.core.graph import artifact_dir
from turbotab.core.tests.truths import Truth, answer_refusal, asked  # noqa: F401 - re-exported


def press_fit(client: Any, pid: str) -> bool:
    """Fit, pressed as the user does on the analysis flowchart (SIZING P0.8): no estimate stage is
    served before it (under Estimate and Describe it locks the plan). True when it was taken; False
    when it was refused because there is nothing to fit yet or a question the estimates rest on is
    open, so the estimates stay withheld, as WP17 withholds them."""
    r = client.post(f"/api/projects/{pid}/fit")
    if r.status_code == 409 and r.json()["error"]["code"] == "fit_not_yet":
        return False
    assert r.status_code == 200, r.text[:600]
    return True


def estimate_stage(stage: str) -> bool:
    from turbotab.core.estimand import ESTIMATE_STAGES

    return stage in ESTIMATE_STAGES


def release(client: Any, pid: str, status: dict[str, Any]) -> None:
    """A stage the scheduler holds for Fit (its estimate exceeds about 2 minutes) is released as
    the user releases it: by pressing Fit (RECIPES_AND_TUNING §4.4)."""
    if status.get("held") == "fit":
        press_fit(client, pid)


def served(client: Any, pid: str, stage: str) -> Any:
    """A stage's artifact as the user reads it: an estimate stage's after Fit is pressed, as the
    user opens Results (P0.8)."""
    if estimate_stage(stage):
        press_fit(client, pid)
    return client.get(f"/api/projects/{pid}/stages/{stage}").json()["artifact"]


@contextmanager
def local_server(home: Path) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from turbotab.core.config import Settings
    from turbotab.server.app import create_app

    settings = Settings(home=home, mode="local", workers=2, memory_budget_bytes=1 << 30)
    with TestClient(create_app(settings, frontend_dist=home / "none"),
                    base_url="http://127.0.0.1") as client:
        yield client


def open_project(client: Any, path: Path, truth: "Truth | None" = None) -> "Drive":
    r = client.post("/api/projects", json={"path": str(path)})
    assert r.status_code == 200, r.text
    drive = Drive(client, r.json()["id"], truth)
    drive.artifact("ingest")
    return drive


def settle_post(client: Any, pid: str, body: dict[str, Any], truth: Truth | None = None,
                unblock: Callable[[], bool] | None = None) -> Any:
    """Post ``body``; while it is refused only for readings to settle, answer every reading it
    asks about from the fixture's ``truth`` (one block confirmation, ``confirm_readings``, which
    settles exactly the readings it lists; an energy column's unit and days as declared), then
    post it again. ``unblock``: answers an earlier question that holds ``body`` (WP17: a reading
    confirmed here can settle a covariate the adjustment question then asks about)."""
    url = f"/api/projects/{pid}/decisions"
    truth = truth if truth is not None else Truth(fixture="the test (no truth given)")
    response = _post_when_reached(client, url, body, unblock=unblock)
    return answer_refusal(lambda d: client.post(url, json=d), response, truth,
                          lambda: _post_when_reached(client, url, body, unblock=unblock))


def _post_when_reached(client: Any, url: str, body: dict[str, Any], timeout: float = 240.0,
                       unblock: Callable[[], bool] | None = None) -> Any:
    """Post ``body``; while the Router holds its question behind an earlier one that a reshaped
    table is recomputing (a code-or-amount answer rebuilds the working table), wait and post
    again. While it holds it behind an earlier open question ``unblock`` answers (True when it
    answered one), post again at once."""
    end = time.monotonic() + timeout
    while True:
        response = client.post(url, json=body)
        if response.status_code != 409 or time.monotonic() > end:
            return response
        if (response.json().get("error") or {}).get("code") != "not_yet":
            return response
        if unblock is not None and unblock():
            continue
        time.sleep(0.1)


def every_row(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Every coefficient a served model carries. Under inference with a declared exposure the fit is
    served as the Table 2 display (ESTIMAND; Westreich & Greenland 2013): ``coefficients`` holds the
    exposure's rows and every other row sits in ``adjustment_terms``, the appendix titled
    "adjustment terms, not effect estimates". A test that checks a covariate's coefficient (that a
    code reached the fit as indicators, that age was adjusted for) reads both."""
    return [*(model.get("coefficients") or []), *(model.get("adjustment_terms") or [])]


def answer_plan(drive: Any, exposure: str, *, effect: str = "total", contrast: str | None = None,
                timeout: float = 240.0) -> list[dict[str, Any]]:
    """Under inference, the exposure and its effect (WP17, MODELING_SEQUENCE §1 steps 2–3), then the
    adjustment set, each covariate answered from the fixture's declared truth
    (``truths.answer_adjustment``: a group the pack guesses alike in one tap). The measure is the
    one the proposals' card says the engine fits; an energy-bearing exposure's contrast is the
    analyst's choice: ``contrast``, else the truth's ``contrast:<exposure>``, else the
    substitution (the field's energy-adjusted estimand, NUTRITION_PACK §04). ``drive`` is any of
    the tests' drives (``reach``, ``artifact``, ``decide``, ``c``, ``pid``, ``truth``). Returns the
    adjustment decisions posted."""
    from turbotab.core.tests.truths import answer_adjustment

    if drive.reach("estimand")["status"] in ("open", "waiting"):
        answer_estimand(drive, exposure, effect=effect, contrast=contrast)
    if drive.reach("adjustment")["status"] not in ("open", "waiting"):
        return []
    end = time.monotonic() + timeout
    while True:
        card = drive.artifact("proposals").get("adjustment")
        if card and card.get("exposure") == exposure:
            break
        assert time.monotonic() < end, (
            f"the adjustment card never named the exposure {exposure!r}: {card!r}; the state's "
            f"estimand {drive.view()['state'].get('estimand')!r}")
        time.sleep(0.05)
    posted = answer_adjustment(
        lambda d: drive.c.post(f"/api/projects/{drive.pid}/decisions", json=d), card, drive.truth)
    settle_forms(drive, timeout=timeout)
    return posted


def settle_forms(drive: Any, timeout: float = 240.0) -> None:
    """FORM (MODELING_SEQUENCE §1 row 5): after the plan's questions, the form question, when the
    Router holds it next (the domain transforms answered or not applicable), as a drive written
    before it would answer it (:func:`answer_forms`); nothing when another question comes first."""
    end = time.monotonic() + timeout
    while True:
        view = drive.c.get(f"/api/projects/{drive.pid}").json()
        card = (view.get("stages") or {}).get("forms") or {}
        if card.get("status") in ("idle", "queued", "running", "stale") and not card.get("cancelled"):
            # The card is being read for these answers (a new covariate may need a form): the
            # Router says what it asks once it is.
            assert time.monotonic() < end, f"the form card never computed: {card}"
            time.sleep(0.05)
            continue
        steps = view["interview"]
        first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
        if first is None:
            return
        if first["key"] == "form":
            if first["status"] == "open":
                answer_forms(drive)
                return
        elif not (first["key"] in ("estimand", "adjustment") and first["status"] == "waiting"):
            return  # another question first (the adjustment's card recomputing aside)
        assert time.monotonic() < end, f"the form question never opened: {first}"
        time.sleep(0.05)


def answer_estimand(drive: Any, exposure: str, *, effect: str = "total",
                    contrast: str | None = None) -> None:
    """The exposure and its effect, on the scale the engine fits; an energy-bearing exposure's
    contrast is ``contrast``, else the truth's ``contrast:<exposure>``, else the substitution. An
    exposure whose role rode along unconfirmed is not offered yet: the answer is refused with the
    confirmation it needs, which the fixture's truth answers (``settle_post``)."""
    card = drive.artifact("proposals")["estimand"]
    option = next((e for e in card["exposures"] if e["column"] == exposure), None)
    # The outcome model's own measure (ESTIMAND: a yes/no outcome's card also fits the marginal
    # risk difference and ratio, ranked first when the event is common), unless the fixture's
    # author declares another (``measure:<exposure>``).
    fitted = drive.truth.get(f"measure:{exposure}") or next(
        m["measure"] for m in card["measures"]
        if m["fitted"] and m.get("conditioning", "conditional") != "marginal")
    body: dict[str, Any] = {"kind": "set_estimand", "exposure": exposure, "effect": effect,
                            "measure": fitted}
    chosen = contrast or drive.truth.get(f"contrast:{exposure}") or "substitution"
    if option is not None and option["energy_contrast"]:
        body["contrast"] = chosen
    r = settle_post(drive.c, drive.pid, body, drive.truth)
    if r.status_code == 409 and r.json()["error"]["code"] == "which_contrast":
        r = settle_post(drive.c, drive.pid, {**body, "contrast": chosen}, drive.truth)
    assert r.status_code == 200, ("set_estimand", r.text[:900])


# With the V2 causal row's time-varying exposure question (``turbotab/core/time_varying.py``).
WP17_QUESTIONS = ("follow_up", "clusters", "estimand", "adjustment", "time_varying", "form")


def answer_forms(drive: Any) -> None:
    """FORM (MODELING_SEQUENCE §1 row 5): the form question, as a drive written before it would
    have to answer it: each column the card asks about takes the fixture's declared form
    (``form:<column>``: a ``set_exposure_form`` body without its kind and column), else the form
    the drive declared for it earlier (on the present scale), else a straight line, the form every
    model had before the question was asked."""
    from turbotab.core.tests.truths import answers, asked, forms_answer

    # BLUEPRINT §14.2 (the FORM repair): a column whose code-or-amount reading is unsettled is
    # asked first, in the question's ask card; the drive answers it from the fixture's truth, as a
    # user who knows the table does, then the card is read again for the answers.
    for _ in range(4):
        step = drive.reach("form", timeout=300)
        ask = step.get("ask") or {}
        if step["status"] != "open" or not asked(ask.get("exits") or []):
            break
        if hasattr(drive, "form_asks"):
            # Whether a fit had been computed by then (not served: Fit waits for the form).
            fit = drive.c.get(f"/api/projects/{drive.pid}/stages/fit").json()
            drive.form_asks.append({"ask": ask, "fit_artifact": bool(fit.get("artifact"))})
        for decision in answers({"exits": ask["exits"]}, drive.truth):
            r = drive.c.post(f"/api/projects/{drive.pid}/decisions", json=decision)
            assert r.status_code == 200, r.text[:600]
    if drive.reach("form", timeout=300)["status"] not in ("open", "waiting"):
        return
    card = drive.artifact("forms")
    declared = drive.view()["state"].get("exposure_forms") or {}
    body = forms_answer(card, declared, drive.truth)
    if body is None:  # the card asks about nothing: the question does not apply
        return
    r = drive.c.post(f"/api/projects/{drive.pid}/decisions", json=body)
    assert r.status_code == 200, r.text[:600]


def answer_wp17(drive: Any, key: str, *, exposure: str | None = None,
                contrast: str = "substitution") -> bool:
    """Answer one of WP17's questions when the Router asks it, as a drive written before them
    would have to (``key`` among :data:`WP17_QUESTIONS`; False for any other key):

    * the follow-up: the test's yes/no outcome counted over one period (``set_censoring``);
    * the grouping: the roles' named grouping, its intervals only under inference (what a confirmed
      cluster role did before the question existed), none when nothing reads as one;
    * the exposure and its effect: ``exposure``, else the fixture's declared ``exposure:<outcome>``
      (a column, or ``family``), else the card's first; the total effect on the scale the engine
      fits, an energy-bearing exposure's ``contrast``;
    * the adjustment set: each covariate from the fixture's declared causal truth
      (:func:`answer_plan`);
    * the time-varying exposure (V2 causal row): standard regression, the analysis every drive
      written before the question runs, with the exposure declared to precede the outcome; where
      a confounder affected by prior exposure holds it (block and record), its attestation exit."""
    if key not in WP17_QUESTIONS:
        return False
    step = drive.reach(key, timeout=300)
    if step["status"] not in ("open", "waiting"):
        return True
    view = drive.view()
    state = view["state"]
    post = lambda d: drive.c.post(f"/api/projects/{drive.pid}/decisions", json=d)  # noqa: E731
    if key == "follow_up":
        r = post({"kind": "set_censoring", "column": state["target"]})
        assert r.status_code == 200, r.text[:600]
    elif key == "clusters":
        from turbotab.core.decisions import ProjectState
        from turbotab.core.estimand import cluster_candidates

        found = cluster_candidates(ProjectState(**state), drive.artifact("roles"))
        inference = state.get("purpose") == "inference"
        r = post({"kind": "set_clusters", "column": found[0] if found else None,
                  "adjust": "cluster_only" if inference and found else None,
                  "acknowledged": not found})
        assert r.status_code == 200, r.text[:600]
    elif key == "estimand":
        card = drive.artifact("proposals")["estimand"]
        # The author's question, where the fixture declares it (``exposure:<outcome>``: a column,
        # or ``family`` for every exposure in turn); else the card's first exposure.
        chosen = exposure or drive.truth.get(f"exposure:{state['target']}") \
            or card["exposures"][0]["column"]
        if chosen == "family":
            family = card["family"]
            fitted = next(m["measure"] for m in family["measures"] if m["fitted"])
            drive.decide({"kind": "set_estimand", "family": True, "measure": fitted,
                          "contrast": contrast if family["energy_contrast"] else None})
            return True
        answer_estimand(drive, chosen, contrast=drive.truth.get(f"contrast:{chosen}") or contrast)
    elif key == "time_varying":
        exposed = (state.get("estimand") or {}).get("exposure")
        lane = {"kind": "set_time_varying", "exposure": exposed, "method": "standard",
                "ordering": "exposure_precedes_outcome"}
        r = post(lane)
        if r.status_code == 409 and r.json()["error"]["code"] == "affected_confounder":
            r = post(next(e["decision"] for e in r.json()["error"]["exits"]
                          if (e["decision"] or {}).get("acknowledged")))
        assert r.status_code == 200, r.text[:600]
    elif key == "form":
        answer_forms(drive)
    else:
        from turbotab.core.decisions import EXPOSURE_FAMILY

        spec = state.get("estimand") or {}
        answer_plan(drive, EXPOSURE_FAMILY if spec.get("family")
                    else spec.get("exposure") or exposure or "")
    return True


class Drive:
    """Answers the opening sequence through the real HTTP API, in the Router's order."""

    def __init__(self, client: Any, pid: str, truth: Truth | None = None):
        self.c, self.pid = client, pid
        self.truth = truth if truth is not None else Truth()
        # WP17: a question of the declared purpose's that holds the one a test reaches for is
        # answered on the way (``answer_wp17``): the exposure named here (the card's first when
        # None), the adjustment set from the fixture's declared causal truth.
        self.exposure: str | None = None
        # FORM: each ask the form question made before the drive answered it (its ask card and
        # whether a fit had been computed then), so a test can check what was asked, and when.
        self.form_asks: list[dict[str, Any]] = []

    def view(self) -> dict[str, Any]:
        return self.c.get(f"/api/projects/{self.pid}").json()

    def post(self, body: dict[str, Any]) -> Any:
        return self.c.post(f"/api/projects/{self.pid}/decisions", json=body)

    def decide(self, body: dict[str, Any]) -> None:
        if body.get("kind") == "set_roles":  # the author's roles are the truth for their readings
            for column, role in body["roles"].items():
                self.truth.setdefault(f"role:{column}", role)
        self.answer_wp17_before(body)
        r = settle_post(self.c, self.pid, body, self.truth,
                        unblock=lambda: self.answer_wp17_before(body))
        assert r.status_code == 200, (body["kind"], r.text[:900])

    def answer_wp17_before(self, body: dict[str, Any]) -> bool:
        """WP17: answer, on the way (``answer_wp17``), the declared purpose's questions that hold the
        question ``body`` answers; nothing else is waited for here. True when one was answered."""
        from turbotab.core.sequence import ANSWERED_ANY_TIME, OUTCOME_QUESTIONS, question_of

        question = question_of(str(body.get("kind")))
        # (FORM: one column's form is declared whenever the user sees it, holding nothing back)
        if question is None or question in WP17_QUESTIONS or body.get("kind") in ANSWERED_ANY_TIME:
            return False
        answered = False
        for _ in range(len(WP17_QUESTIONS) + 1):
            steps = self.view()["interview"]
            first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
            # A WP17 question waiting on its stage (the proposals' cards) is waited for and
            # answered (``answer_wp17`` reaches it first).
            if first is None or first["key"] == question or first["key"] not in WP17_QUESTIONS:
                return answered
            # The event, the task and the follow-up describe one outcome and do not hold each
            # other back (``sequence.OUTCOME_QUESTIONS``): the time-to-event exit is its own answer.
            if question in OUTCOME_QUESTIONS and first["key"] in OUTCOME_QUESTIONS:
                return answered
            answer_wp17(self, first["key"], exposure=self.exposure)
            answered = True
        return answered

    def decide_roles(self, roles: dict[str, str]) -> None:
        """Record the roles as their author answers them (BLUEPRINT §14, recognition's leash):
        the ``set_roles`` answer, then every role the server recorded unconfirmed (proposed below
        high confidence) confirmed in one block, each with the role the author gave it, as a user
        who knows the table does. A bulk ``set_roles`` alone confirms none of them. The author's
        roles are the fixture's truth for its role readings."""
        for column, role in roles.items():
            self.truth.setdefault(f"role:{column}", role)
        self.decide({"kind": "set_roles", "roles": roles})
        record = self.view()["decisions"][-1]
        waiting = record["decision"].get("unconfirmed") or []
        if waiting:
            self.decide({"kind": "confirm_readings", "items": [
                {"reading": "role", "column": c, "value": roles[c]} for c in waiting]})

    def reach(self, key: str, timeout: float = 120.0) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            steps = self.view()["interview"]
            step = next(s for s in steps if s["key"] == key)
            first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
            ready = step["status"] not in ("open", "waiting") or first is None or first["key"] == key
            if ready and not (step["status"] == "waiting" and step.get("waiting_on")):
                return step
            if (key != "task" and first is not None and first["key"] == "task"
                    and first["status"] == "open" and first.get("followup")):
                # WP18: a driver passing the task question answers what it still asks, as usual.
                self.task_followups()
                continue
            if (first is not None and first["key"] in WP17_QUESTIONS and first["key"] != key
                    and first["status"] == "open"):
                answer_wp17(self, first["key"], exposure=self.exposure)
                continue
            assert time.monotonic() < end, f"{key} held behind {first}"
            time.sleep(0.05)

    def answer(self, key: str, body: dict[str, Any]) -> None:
        if self.reach(key)["status"] in ("open", "waiting"):
            self.decide(body)
        if key == "task":
            self.task_followups()

    def task_followups(self) -> None:
        """WP18 (audit RO-10): what the task question still asks after its task, answered as the
        fixture declares (``outcome_scale:<column>``), else as usual: the original scale, and an
        ordinal text outcome's levels in the order their words propose."""
        for _ in range(3):
            step = self.reach("task")
            follow = step.get("followup")
            if step["status"] not in ("open", "waiting") or follow is None:
                return
            target = self.view()["state"]["target"]
            if follow == "scale":
                scale = self.truth.get(f"outcome_scale:{target}", "original")
                self.decide({"kind": "set_outcome_scale", "column": target, "scale": scale})
            else:
                question = self.artifact("target_info").get("order_question") or {}
                self.decide({"kind": "set_outcome_order", "column": target,
                             "levels": question.get("proposed_order") or question.get("levels")})

    def press_fit(self) -> bool:
        return press_fit(self.c, self.pid)

    def artifact(self, stage: str, timeout: float = 240.0) -> dict[str, Any]:
        """The stage's artifact once fresh. An estimate stage's is served only after Fit (P0.8),
        so Fit is pressed first, as the user opens Results."""
        if estimate_stage(stage):
            self.press_fit()
        end = time.monotonic() + timeout
        while True:
            status = self.view()["stages"][stage]
            release(self.c, self.pid, status)
            if status["status"] == "fresh":
                return self.c.get(f"/api/projects/{self.pid}/stages/{stage}").json()["artifact"]
            assert status["status"] != "error", status
            assert time.monotonic() < end, f"{stage} never fresh: {status}"
            time.sleep(0.05)

    def answer_plan(self, exposure: str, *, effect: str = "total") -> list[dict[str, Any]]:
        return answer_plan(self, exposure, effect=effect)

    def sealed(self) -> set[int]:
        self.artifact("split")
        key = self.view()["stages"]["split"]["key"]
        cache = self.c.app.state.service.workspace.cache_dir(self.pid)
        frame = pd.read_parquet(artifact_dir(cache, "split", key) / "frames" / "sealed.parquet")
        return set(frame["row_id"].astype(int))
