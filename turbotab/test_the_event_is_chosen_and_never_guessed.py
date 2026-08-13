"""`DRIVE-032` — the app asked the right question, refused to guess when asked
directly, and then guessed anyway on a path that never consulted it.

Two defects, measured one fit per shape before anything changed:

**One — it fired on the detector, not on the target.** `positive_class_finding`
built its plan with `read_as_binary_plan`, which opens `if not _is_texty(s):
return None`. So `case`/`control`, `died`/`survived` and `Yes`/`No` all raised
*"Which of these is the event you are predicting?"* and **numeric `0`/`1` did
not** — which is the tester's `meds_hbp`, and why two human drives saw nothing.

**Two — an unanswered question was overruled.** On a 132-case / 1,068-control
study the finding was still open at seal time, the seal succeeded, and the fit
recorded `positive_label = 'control'` — the larger group, the non-event — which
`/figures` served as `"event": "control"`. `training.py` set
`result.positive_label = classes_[1]`, sklearn's sorted-second level.

So the app raised the right question, `api.py` refused to guess when asked
directly, and a different path answered it anyway.

## Why the fit refuses rather than the seal

Both were defensible and the measurement chose. `engine.draw_holdout` does not
stratify by class — it draws from `df.index[y.notna()]` and never looks at
levels — so **the split is identical whichever level is the event.** Gating the
seal would refuse a step that cannot use the answer, and the pre-seal questions
are exactly the ones the split depends on. Where the answer IS consumed is at
scoring and at `figure_bundle.predictions_for`, so the refusal lands there,
beside `api.py`'s existing one.

## What "chosen" means

Applying the repair rewrites the column so the chosen level is `1`. A target
whose event was chosen is therefore `0`/`1` with the event as `1`, and
`classes_[1]` is then genuinely correct. **The encoding was never the defect —
running at all with the question open was.** So the DECISION is consulted, not
the dtype: a `0`/`1` column can be one a user chose or one that arrived that
way, and only the record tells them apart.
"""
from __future__ import annotations

import io
import json
import re
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

N_ROWS = 1200


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _frame(labels: Dict[int, Any]) -> pd.DataFrame:
    """A 12%-event study with real signal, so the two levels are asymmetric and
    calling the wrong one the event is a visible mistake."""
    rng = np.random.default_rng(11)
    event = rng.choice([0, 1], size=N_ROWS, p=[0.88, 0.12])
    x = rng.normal(0, 1, N_ROWS) + event * 2.2
    return pd.DataFrame({"x1": x.round(3),
                         "x2": rng.normal(0, 1, N_ROWS).round(3),
                         "outcome": [labels[e] for e in event]})


def _upload(client, df: pd.DataFrame) -> str:
    buf = io.BytesIO(df.to_csv(index=False).encode())
    return client.post("/project", files={
        "file": ("t.csv", buf, "text/csv")}).json()["id"]


def _target(client, pid, column="outcome"):
    resp = client.post(f"/project/{pid}/decision",
                       json={"kind": "set_target", "payload": {"column": column}})
    assert resp.status_code == 200, resp.text[:200]
    return resp.json()


def _seal(client, pid):
    for kind, payload in (("set_grain", {"answer": "one_row_per_person"}),
                          ("set_eligibility", {"answer": "everyone"}),
                          ("seal", {})):
        resp = client.post(f"/project/{pid}/decision",
                           json={"kind": kind, "payload": payload})
        assert resp.status_code == 200, (kind, resp.text[:200])


def _fit(client, pid) -> Dict[str, Any]:
    shelf = client.get(f"/project/{pid}/models").json()
    keys = [m["key"] for g in shelf.get("groups", []) for m in (g.get("models") or [])]
    pick = [k for k in keys if k == "logreg"] or keys[:1]
    client.post(f"/project/{pid}/decision",
                json={"kind": "select_models", "payload": {"models": pick}})
    started = client.post(f"/project/{pid}/train", json={"models": pick})
    if started.status_code != 200:
        return {"status": started.status_code,
                "detail": str(started.json().get("detail", ""))}
    jid = started.json()["id"]
    for _ in range(240):
        job = client.get(f"/job/{jid}").json()
        if job.get("terminal"):
            break
        time.sleep(0.25)
    run = (client.get(f"/project/{pid}/training").json().get("run") or {})
    results = run.get("results") or []
    figures = client.get(f"/project/{pid}/figures")
    body = json.dumps(figures.json()) if figures.status_code == 200 else ""
    return {"status": 200,
            "positive_label": results[0].get("positive_label") if results else None,
            "figure_events": sorted(set(re.findall(
                r'"(?:event|positive_label)"\s*:\s*"([^"]+)"', body)))}


def _finding_ids(client, pid):
    return [f.get("id") for f in client.get(f"/project/{pid}").json().get("findings", [])]


# ── defect one · the question exists wherever two levels do ──────────────────

@pytest.mark.parametrize("name,labels", [
    ("text case/control", {0: "control", 1: "case"}),
    ("text died/survived", {0: "survived", 1: "died"}),
    ("text Yes/No", {0: "No", 1: "Yes"}),
    ("numeric 0/1", {0: 0, 1: 1}),
])
def test_every_two_level_target_is_asked_which_level_is_the_event(name, labels, capsys):
    """`numeric 0/1` is the case that was missing, and the other three are the
    positive control — a test that drove only the numeric shape would pass over
    a change that broke the text ones."""
    client = _client()
    pid = _upload(client, _frame(labels))
    _target(client, pid)
    assert "positive_class__outcome" in _finding_ids(client, pid), (
        f"{name}: a two-level outcome is not asked which level is the event. "
        f"The question rode `read_as_binary_plan`, which refuses a column that "
        f"is not text — DRIVE-032's first defect.")
    with capsys.disabled():
        print(f"\n  {name:<22} asked")


def test_a_three_level_target_is_not_asked(capsys):
    """The boundary, so the widening did not become "ask about everything".

    A three-level outcome has no single event to name and `two_level_plan` must
    return `None` for it — otherwise the question would appear on a multiclass
    problem where it is meaningless.
    """
    client = _client()
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x1": rng.normal(0, 1, N_ROWS).round(3),
                       "outcome": rng.choice(["mild", "moderate", "severe"], N_ROWS)})
    pid = _upload(client, df)
    _target(client, pid)
    assert "positive_class__outcome" not in _finding_ids(client, pid)
    with capsys.disabled():
        print("\n  three-level outcome     not asked")


# ── defect two · the answer is never invented ────────────────────────────────

@pytest.mark.parametrize("name,labels", [
    ("text case/control", {0: "control", 1: "case"}),
    # THE TESTER'S OWN SHAPE, and the one a text-only version of this test would
    # have missed. A revert probe that swapped the record check for a dtype
    # check passed while this file held only the text case — the gap was found
    # by probing, not by reading.
    ("numeric 0/1", {0: 0, 1: 1}),
])
def test_an_unanswered_event_cannot_reach_a_fitted_label(name, labels, capsys):
    """The load-bearing claim, and it is §00's measurement verbatim.

    Before this, the same drive produced `positive_label = 'control'` on a
    132-case / 1,068-control study and `/figures` served `"event": "control"`.
    """
    client = _client()
    pid = _upload(client, _frame(labels))
    _target(client, pid)
    assert "positive_class__outcome" in _finding_ids(client, pid)
    _seal(client, pid)

    got = _fit(client, pid)
    assert got["status"] == 400, (
        f"the fit ran with the event unrecorded and called "
        f"{got.get('positive_label')!r} the event")
    assert "has not been recorded" in got["detail"], got["detail"]
    # The sentence `api.py` already used for the same refusal, so the two doors
    # say the same thing about the same question.
    assert "research question" in got["detail"]
    with capsys.disabled():
        print(f"\n  {name:<22} unanswered: {got['status']} — "
              f"{got['detail'][:44]}…")


def test_the_recorded_event_is_the_one_the_user_named(capsys):
    """`GUIDED-093` is what this protects — *a picture of the complementary
    event, drawn confidently.*"""
    client = _client()
    pid = _upload(client, _frame({0: "control", 1: "case"}))
    _target(client, pid)

    answered = client.post(f"/project/{pid}/decision",
                           json={"kind": "apply",
                                 "subject": "positive_class__outcome",
                                 "payload": {"choice": "case"}})
    assert answered.status_code == 200, answered.text[:300]
    said = [d["text"] for d in answered.json()["decisions"] if d["kind"] == "apply"][-1]
    assert "case as the event" in said, said

    _seal(client, pid)
    got = _fit(client, pid)
    assert got["status"] == 200, got
    # The column is re-encoded with the chosen level as 1, so the event IS 1 in
    # the data — asserted as "the positive level", not as the word the user
    # typed, because the word is no longer in the frame.
    assert str(got["positive_label"]).startswith("1"), got
    assert got["figure_events"] and all(
        e.startswith("1") for e in got["figure_events"]), got
    with capsys.disabled():
        print(f"\n  answered 'case' → label {got['positive_label']!r}, "
              f"figures {got['figure_events']}")


def test_answering_it_does_not_ask_it_again(capsys):
    """The repair rewrites the column to `0`/`1`, which `two_level_plan` now
    recognizes — so the question would re-raise if the record did not settle it.

    It does: `apply` folds `repair::<id>` into `answered` and the Router drops
    it. Asserted, because the widening in defect one is what made this possible.
    """
    client = _client()
    pid = _upload(client, _frame({0: "control", 1: "case"}))
    _target(client, pid)
    client.post(f"/project/{pid}/decision",
                json={"kind": "apply", "subject": "positive_class__outcome",
                      "payload": {"choice": "case"}})
    plan = client.get(f"/project/{pid}/interview?step=data").json()
    asked = [q["key"] for q in plan["questions"]
             if q["status"] == "asked" and "positive_class" in q["key"]]
    assert not asked, f"the event question came back after being answered: {asked}"
    with capsys.disabled():
        print("\n  answered once, not re-asked")


def test_a_regression_target_is_untouched(capsys):
    """The refusal is scoped to a two-level classification outcome. A
    regression fit must not acquire a question that means nothing for it."""
    client = _client()
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, N_ROWS)
    df = pd.DataFrame({"x1": x.round(3), "outcome": (x * 3 + 50).round(2)})
    pid = _upload(client, df)
    _target(client, pid)
    _seal(client, pid)
    got = _fit(client, pid)
    assert got["status"] == 200, got
    assert got["positive_label"] is None
    with capsys.disabled():
        print(f"\n  regression fits, no event recorded")


def test_applying_the_repair_without_a_choice_is_still_refused(capsys):
    """`api.py`'s existing refusal, pinned — it is the sentence the new one
    quotes, and a fix that widened the question must not have loosened it."""
    client = _client()
    pid = _upload(client, _frame({0: "control", 1: "case"}))
    _target(client, pid)
    resp = client.post(f"/project/{pid}/decision",
                       json={"kind": "apply",
                             "subject": "positive_class__outcome", "payload": {}})
    assert resp.status_code == 400
    assert "There is no default" in resp.json()["detail"]
    with capsys.disabled():
        print("\n  apply with no choice: 400, unchanged")


# ── A3 · the refusal `figure_bundle` was written to make ─────────────────────

def test_the_figure_layer_refuses_a_run_that_names_no_event(capsys):
    """`figure_bundle.py:432`, driven at the function rather than reasoned about.

    **Said plainly, because it is the honest form of this claim:** with the fit
    refused, no API path can now produce a scored run whose `positive_label` is
    `None` — `positive_label` is set inside the same branch that fills
    `probabilities`, and the error path clears both. So this branch is
    unreachable *through the app*, and that is the point: it existed to catch a
    state the app should never be in, and `training.py` used to guarantee the
    app was never in it by guessing instead.

    Driving it directly asserts the guard is correct and would fire, rather than
    asserting a path that no longer exists. A branch nothing can execute is
    `TEST-077`'s class, and the answer here is that the branch is right and its
    trigger has been removed upstream — which is recorded rather than implied.
    """
    from turbotab import figure_bundle, training

    class _Run:
        task_type = "classification"

        def __init__(self, results):
            self.results = results

    class _Project:
        """Enough of a project for `predictions_for`, and no more."""
        working_table = pd.DataFrame({"outcome": [0, 1] * 5})
        target = "outcome"
        lockbox = {"labels": list(range(10))}

        def __init__(self, run):
            self.training_run = run

    def _result(label):
        got = training.ModelResult(key="logreg", name="Logistic regression",
                                   concern="", bucket="")
        got.probabilities = [0.1 * i for i in range(10)]
        got.positive_label = label
        return got

    # THE POSITIVE CONTROL FIRST. With an event named, the same project yields a
    # bundle — so `None` below is the guard firing rather than the fixture
    # failing to reach it.
    named = figure_bundle.predictions_for(_Project(_Run([_result(1)])))
    assert named is not None, (
        "the fixture produces no bundle even with an event named, so a refusal "
        "below would prove nothing")
    binary, proba, name, event = named
    assert event == 1 and len(binary) == 10

    # AND THE BRANCH. Same project, same probabilities, no event recorded.
    refused = figure_bundle.predictions_for(_Project(_Run([_result(None)])))
    assert refused is None, (
        "the figure layer drew a curve for a run that names no event — the "
        "branch at figure_bundle.py:432 did not fire")
    with capsys.disabled():
        print("\n  figure refusal fires; with an event it draws")
