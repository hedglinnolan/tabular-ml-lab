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
    return answer_adjustment(
        lambda d: drive.c.post(f"/api/projects/{drive.pid}/decisions", json=d), card, drive.truth)


def answer_estimand(drive: Any, exposure: str, *, effect: str = "total",
                    contrast: str | None = None) -> None:
    """The exposure and its effect, on the scale the engine fits; an energy-bearing exposure's
    contrast is ``contrast``, else the truth's ``contrast:<exposure>``, else the substitution. An
    exposure whose role rode along unconfirmed is not offered yet: the answer is refused with the
    confirmation it needs, which the fixture's truth answers (``settle_post``)."""
    card = drive.artifact("proposals")["estimand"]
    option = next((e for e in card["exposures"] if e["column"] == exposure), None)
    fitted = next(m["measure"] for m in card["measures"] if m["fitted"])
    body: dict[str, Any] = {"kind": "set_estimand", "exposure": exposure, "effect": effect,
                            "measure": fitted}
    chosen = contrast or drive.truth.get(f"contrast:{exposure}") or "substitution"
    if option is not None and option["energy_contrast"]:
        body["contrast"] = chosen
    r = settle_post(drive.c, drive.pid, body, drive.truth)
    if r.status_code == 409 and r.json()["error"]["code"] == "which_contrast":
        r = settle_post(drive.c, drive.pid, {**body, "contrast": chosen}, drive.truth)
    assert r.status_code == 200, ("set_estimand", r.text[:900])


WP17_QUESTIONS = ("follow_up", "clusters", "estimand", "adjustment")


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
      (:func:`answer_plan`)."""
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
        from turbotab.core.sequence import OUTCOME_QUESTIONS, question_of

        question = question_of(str(body.get("kind")))
        if question is None or question in WP17_QUESTIONS:
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
            if (first is not None and first["key"] in WP17_QUESTIONS and first["key"] != key
                    and first["status"] == "open"):
                answer_wp17(self, first["key"], exposure=self.exposure)
                continue
            assert time.monotonic() < end, f"{key} held behind {first}"
            time.sleep(0.05)

    def answer(self, key: str, body: dict[str, Any]) -> None:
        if self.reach(key)["status"] in ("open", "waiting"):
            self.decide(body)

    def artifact(self, stage: str, timeout: float = 240.0) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            status = self.view()["stages"][stage]
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
