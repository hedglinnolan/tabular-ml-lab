"""`DRIVE-043` — one field, four renderers, and three of them said the wrong noun.

**What run 5 found, and why 2,607 green tests could not.** `resolution.statement`
counted held-out events against `counts.index[-1]` — the **least frequent**
level, never the recorded decision. When the event is the minority that is
accidentally right, and the minority *is* the event in the ordinary clinical
case and in every fixture this repository had. It is wrong only when a user
names the **majority** as the event, which is what the fifth human drive did:
`meds_hbp` is 87.77% `True`, the tester chose `True`, and the Methods section
printed **116** while every figure payload printed **829** on the same 945
rows. 945 − 829 = 116 — the non-event count under an events label, in the
artifact that leaves the building.

**The quantity was deliberate and the label was not.** `resolution.py` and
`web/index.html` carry the *same* documented reason for holding arithmetic
rather than a class value: `archive.assert_no_participant_data` rejects a
serialized class label, and it is right to. So the count stays a count. What
changed is that the count is now of the **recorded** event where one exists,
and every renderer says which of the two things it is looking at.

## What this file guards, and it is the class rather than the sentence

A test that only checked the new Methods sentence would leave the defect's
shape intact: **two implementations of one field's meaning, free to diverge.**
The page said *"of the less common outcome"* and the Methods section said
*"carrying the outcome"* about the same integer, and both had been shipping for
as long as both existed. So the assertions below run the Python renderers and
the JavaScript one **against each other on the same payload**, and fail if they
disagree — which is the only form that a later edit to one cannot slip past.

`resolution.event_noun` is the single source. The page reads
`res.event_count_noun` off the wire rather than deciding for itself.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, eventfixture, pageharness as PH   # noqa: E402
from turbotab import resolution as R, training as T         # noqa: E402

#: Run 5's shape in miniature: a heavy majority, so counting the minority and
#: counting the event give different answers. **A fixture whose event is the
#: minority cannot fail the old code**, which is exactly why this defect
#: survived every sweep.
N = 800
N_LABELED = 400
MAJORITY_SHARE = 0.88


def _frame(seed: int = 5, n_labeled: int = N_LABELED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "age": rng.normal(50, 12, N).round(1),
        "bmi": rng.normal(27, 5, N).round(1),
        "sbp": rng.normal(128, 16, N).round(1),
    })
    outcome = pd.Series(
        rng.choice(["yes", "no"], N, p=[MAJORITY_SHARE, 1 - MAJORITY_SHARE]),
        dtype=object)
    # Rows with no outcome, because run 5's table had 15,552 of them and they
    # are what `analysis_mask` is about one row over.
    outcome.iloc[n_labeled:] = None
    frame["treated"] = outcome
    return frame


def _sealed(client, *, event: str | None = "yes",
            fraction: float = 0.25, n_labeled: int = N_LABELED):
    """A sealed project, with the event answered through the real route.

    `event=None` seals without answering, which is a reachable state:
    `set_positive_class` is not in `PRE_BARRIER_ONLY_FIXES`, so a user may seal
    first and answer afterwards.
    """
    pid = client.post("/project", files={
        "file": ("p.csv",
                 _frame(n_labeled=n_labeled).to_csv(index=False).encode(),
                 "text/csv")}).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])

    decide("set_target", column="treated")
    if event is not None:
        eventfixture.choose_event_over_http(client, pid, "treated",
                                            level=event, required=True)
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=fraction)
    return pid, api.STORE.get(pid)


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _truth(project):
    """Ground truth from pandas, not from the app."""
    table = project.working_table
    target = str(project.target)
    has_y = table[target].notna()
    sealed = set(project.lockbox["labels"])
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    held = table.loc[has_y & is_test, target]
    return {"n_test": int(len(held)),
            "events": int((held == T.EVENT_VALUE).sum()),
            "non_events": int((held != T.EVENT_VALUE).sum())}


# ── the count ───────────────────────────────────────────────────────────────

def test_the_majority_event_is_counted_and_not_its_complement(client):
    """**The defect, driven on the shape that exposes it.**

    The user names the majority level. The count must be the event's, not the
    minority's, and the two must differ — asserted, or this fixture is the
    accidentally-right case and proves nothing.
    """
    _, project = _sealed(client, event="yes")
    truth = _truth(project)
    assert truth["events"] > truth["non_events"], (
        "this fixture's event is not the majority, so the old code would have "
        "been accidentally right and this test asserts nothing")

    res = project.lockbox["resolution"]
    assert res["events_held_out"] == truth["events"], (
        f"the resolution counted {res['events_held_out']} where the recorded "
        f"event has {truth['events']} of {truth['n_test']} held-out rows; "
        f"{truth['non_events']} is the complement, which is what "
        f"`counts.index[-1]` returns")
    assert res["non_events_held_out"] == truth["non_events"]
    assert res["event_is_recorded"] is True


def test_the_minority_event_still_counts_the_minority(client):
    """The other direction. Naming the minority must not now be wrong — the
    old code's answer was right in this case and must stay right."""
    _, project = _sealed(client, event="no")
    truth = _truth(project)
    res = project.lockbox["resolution"]
    assert res["events_held_out"] == truth["events"]
    assert truth["events"] < truth["non_events"], (
        "the minority arm is not the minority; check the fixture")


def test_with_no_event_recorded_it_counts_the_minority_and_says_so(client):
    """**The branch that keeps the fallback honest.**

    Sealing before answering is reachable. There is then no event, and the
    honest form is the count the app can actually make — the least frequent
    level — under a noun that says that. Calling it an event would be inventing
    a decision nobody made.
    """
    _, project = _sealed(client, event=None)
    res = project.lockbox["resolution"]
    assert res["event_is_recorded"] is False
    assert res["event_count_noun"] == R.MINORITY_NOUN
    assert "the event" not in res["sentence"], res["sentence"]
    assert R.MINORITY_NOUN in res["sentence"], res["sentence"]


# ── the noun, in every sentence that carries it ─────────────────────────────

@pytest.mark.parametrize("event,noun", [("yes", R.EVENT_NOUN),
                                        (None, R.MINORITY_NOUN)])
def test_every_python_renderer_uses_the_same_noun(client, event, noun):
    """**Three sentences in one module disagreed with a fourth in the page.**

    `_headline`, `_sentence` and `_push_because` each named this count, and all
    three said *"the outcome"* whichever level was counted. They read one
    function now.
    """
    _, project = _sealed(client, event=event)
    res = project.lockbox["resolution"]
    assert res["event_count_noun"] == noun
    for key in ("headline", "sentence"):
        said = res[key]
        assert noun in said, f"{key} does not use {noun!r}: {said}"
        other = R.MINORITY_NOUN if noun == R.EVENT_NOUN else R.EVENT_NOUN
        assert other not in said, (
            f"{key} uses BOTH nouns, so a reader cannot tell which count it "
            f"is: {said}")


def test_the_push_sentence_names_the_statistic_the_event_decides():
    """`_push_because` was worse than a wrong noun and it is worth its own test.

    It said *"sensitivity is undefined"* when the count of the LEAST FREQUENT
    level fell below two. Sensitivity is the rate **of the event**. With the
    event as the majority, that branch announced the statistic that was fine
    and stayed silent about the one that was not — so reading the recorded
    event corrects which statistic is named, not only the noun.
    """
    thin = R._push_because("classification", 40, 1, 39, 2, 0, True)
    assert thin and "sensitivity" in thin and R.EVENT_NOUN in thin, thin
    fat = R._push_because("classification", 40, 39, 1, 2, 0, True)
    assert fat and "specificity" in fat, fat


# ── the two implementations, against each other ─────────────────────────────

def _routes(client, pid):
    """Every route the controller asks for, answered from the TestClient.

    Pre-seeded and then iterated: the first render happens before its routes
    exist and the controller throws on an interview payload it has not fetched.
    """
    routes = {}
    for step in ("data", "explore", "preprocess", "features", "train",
                 "explain", "report"):
        path = f"/project/{pid}/interview?step={step}"
        got = client.get(path)
        if got.status_code == 200:
            routes[path] = got.json()
    for path in (f"/project/{pid}", f"/project/{pid}/findings",
                 f"/project/{pid}/figures", f"/project/{pid}/features",
                 f"/project/{pid}/recipes", f"/project/{pid}/preprocess",
                 f"/project/{pid}/training", "/capabilities", "/dev/status"):
        got = client.get(path)
        if got.status_code == 200:
            try:
                routes[path] = got.json()
            except ValueError:
                pass
    return routes


def _render_resolution(client, pid, resolution=None):
    """What the PAGE makes of this resolution, **through its own bootstrap**.

    `resolutionHTML` lives inside the controller's closure and the harness
    cannot call it — which is right, and is the same reason `SHELF` is not
    reachable either. So the page is booted on the real project and the seal
    disclosure is read out of the DOM. `resolution` doctors the served payload
    where a test needs to.
    """
    routes = _routes(client, pid)
    if resolution is not None:
        served = {**routes[f"/project/{pid}"]}
        served["disclosures"] = {**served["disclosures"],
                                 "resolution": resolution}
        routes[f"/project/{pid}"] = served
    out = PH.run("__emit({html: __harness.html('disclosuresBox')});",
                 routes=routes, search=f"?project={pid}")
    return out["html"]


def test_the_page_and_the_methods_section_agree_about_what_was_counted(client):
    """**`A1b`, and it is the assertion the row asked for.**

    Two renderers of one field disagreed for as long as both existed, and only
    the one that leaves the building was false. Checking the corrected sentence
    alone would leave that class open, so this drives the PAGE and compares
    what it renders against what the Python composed — same project, same
    payload, both nouns read off one server-side function.
    """
    if not PH.available():
        pytest.skip("no JS engine on this machine")

    # **A STARK holdout, and the reason is itself a finding.**
    # `resolutionHTML` returns "" unless `res.push` — so the page's honest line
    # renders only when the card is pushed, while the Methods sentence renders
    # unconditionally. The false renderer was the one always on screen.
    pid, project = _sealed(client, event="yes", n_labeled=60, fraction=0.2)
    res = project.lockbox["resolution"]
    assert res["push"] is True, (
        "this seal does not push, so the page renders no resolution card and "
        "the comparison below would be between a sentence and an empty string")
    served = client.get(f"/project/{pid}").json()
    disclosed = (served.get("disclosures") or {}).get("resolution")
    assert disclosed, f"the payload carries no resolution: {sorted(served)}"

    rendered = _render_resolution(client, pid)

    assert str(res["events_held_out"]) in rendered, (
        f"the page did not render the count at all: {rendered[:300]}")
    assert res["event_count_noun"] in rendered, (
        f"the page calls this count something the server does not: it says "
        f"{rendered[:300]!r} and the record says {res['event_count_noun']!r}")
    # AND THE DISAGREEMENT ITSELF, asserted directly rather than implied by the
    # two checks above passing.
    wrong = (R.MINORITY_NOUN if res["event_is_recorded"] else R.EVENT_NOUN)
    assert wrong not in rendered, (
        f"the page uses {wrong!r} while the Methods section uses "
        f"{res['event_count_noun']!r} — one field, two renderers, disagreeing")


def test_the_noun_comes_from_the_server_and_not_from_the_page(client):
    """The mechanism that makes the agreement structural rather than a
    coincidence two files currently share.

    If the page ever decides the noun for itself again, this fails: the served
    payload is doctored to an implausible noun and the render must follow it.
    """
    if not PH.available():
        pytest.skip("no JS engine on this machine")

    pid, project = _sealed(client, event="yes", n_labeled=60, fraction=0.2)
    served = client.get(f"/project/{pid}").json()
    disclosed = (served.get("disclosures") or {})["resolution"]
    rendered = _render_resolution(
        client, pid, {**disclosed, "event_count_noun": "SENTINEL-NOUN"})
    assert "SENTINEL-NOUN" in rendered, (
        "the page ignored the server's noun and composed its own, which is the "
        "divergence this file exists to prevent")
