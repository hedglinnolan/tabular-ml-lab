"""L48-E — `GUIDED-183`, and it was bigger than the row.

The row says: `purpose.INDICATOR_EXITS[0]` carries no `retry`, so `showRefusal`
emits `disabled` and **the safe way out renders greyed out beside a live
attest** — the inversion §09 exists to prevent. `GUIDED-087` is the same shape
and `missingness.blocker_exits` is its build, so the row reads as *do that here
too*.

**Two of the row's premises did not survive being driven.**

1. *"the way every other RESOLVE exit carries one"* — **one of five does.**
   `grain._RESOLVE` and `packs._LENS_RESOLVE` are `revise`-shaped: the way to
   take them is to go back and answer the question differently, and there is no
   request to re-post. They carry no payload because they honestly have none,
   and the page greys them out for it. That is a real defect and it needs a page
   mechanism the server describes — filed, not fixed here.
   `clinical.substitution_blocker`'s `keep_censored` has no consumer anywhere
   outside a test, so there is no refused request to retry at all — also filed.

2. **The one exit that DOES carry a retry does not open.** Driven: from the
   Explore door, taking `blocker_exits`' resolve the way `showRefusal` takes it
   — merging `retry.payload` into the request that was refused — produced a
   **second 409 naming the same refused strategy**. `api.py:729` reads
   `card_option` in preference to `strategy`, the Explore door posts
   `card_option`, and a retry carrying only `strategy` is shadowed by the
   refused option still sitting in the payload. That is `GUIDED-072`'s defect —
   *an exit that renders as a way through and opens nothing* — alive inside the
   fix built for it, on the door the product owner uses.

So the fix is two lines of payload in two modules and one new inverse lookup,
and the test that matters is not *does the exit carry a retry* but **does the
retry open**.

## What is NOT covered, said out loud

- **`informative` + `inference` + `indicator`.** The purpose blocker's resolve
  is `impute_median`, and on an informative mechanism clause §07's own blocker
  refuses exactly that — so the retry returns a **different** 409. Both rules
  are right and they collide. Driven and reported below rather than papered
  over; filed as its own row.
- **Whether the enabled button is on screen.** Nothing without layout can tell.
"""
from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "sample_data"

#: Every `resolve` exit this app composes, and what taking it means.
#: `takeable` is whether a client holding only the payload could post the retry.
RESOLVE_EXITS = {
    "missingness.blocker_exits": "takeable — re-posts a different strategy",
    "purpose.INDICATOR_EXITS[0]": "takeable — re-posts a training-fold median",
    "grain._RESOLVE": "REVISE — go back to the question; no request to re-post",
    "packs._LENS_RESOLVE": "REVISE — go back to the question; no request to re-post",
    "clinical.substitution_blocker": "no consumer anywhere; nothing refuses, so "
                                     "there is no request to retry",
}


def _project(fixture: str, target: str, purpose: str | None = None):
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    if purpose:
        client.post(f"/project/{pid}/decision",
                    json={"kind": "set_purpose", "payload": {"answer": purpose}})
    return client, pid


def _take_the_resolve(client, pid, request):
    """Post `request`, take its resolve exit the way `showRefusal` takes it.

    The merge is the page's: `LAST_REFUSAL.request` with `exit.retry.payload`
    merged in, posted to the same endpoint. Reproducing it here rather than
    asserting on the payload's shape is the whole point — a payload that looks
    right and is shadowed by a key already in the request is what this found.
    """
    first = client.post(f"/project/{pid}/decision", json=request)
    assert first.status_code == 409, (
        f"expected a blocker and got {first.status_code}: "
        f"{str(first.json())[:200]}")
    detail = first.json()["detail"]
    resolves = [e for e in detail["exits"] if e.get("kind") == "resolve"]
    assert resolves, f"the blocker offers no way out at all: {detail!r}"
    exit_row = resolves[0]
    retry = (exit_row.get("retry") or {}).get("payload")
    assert retry, (
        f"the resolve exit {exit_row['id']!r} carries no retry payload, so "
        f"`showRefusal` renders it `disabled` — the safe way out greyed out "
        f"beside a live attest, which is the inversion §09 forbids")
    merged = dict(request)
    merged["payload"] = dict(request["payload"])
    merged["payload"].update(retry)
    return exit_row, client.post(f"/project/{pid}/decision", json=merged)


@pytest.mark.parametrize("fixture,target,column,option", [
    ("clinic_visits.csv", "outcome", "notes", "impute_mode"),
    ("metabolomics_untargeted.csv", "bmi", "mz_0022", "impute_median"),
], ids=["classification · categorical column", "regression · numeric column"])
def test_clause_07s_resolve_opens_from_the_door_that_posts_card_options(
        fixture, target, column, option):
    """`GUIDED-087`'s build, driven end to end for the first time.

    Two fixtures of different target shape (`GUIDED-097`). Before L48 both
    returned a second 409 naming the strategy that had just been refused.
    """
    client, pid = _project(fixture, target)
    exit_row, took = _take_the_resolve(client, pid, {
        "kind": "route_missingness", "subject": column,
        "payload": {"column": column, "card_option": option,
                    "mechanism": "informative"}})
    assert took.status_code == 200, (
        f"taking the resolve exit {exit_row['id']!r} was refused "
        f"({took.status_code}): {str(took.json())[:300]}. The retry payload "
        f"names a strategy and the request still carries the refused "
        f"`card_option`, which `api.py` reads first")
    assert exit_row["retry"]["payload"].get("card_option") == exit_row["id"], (
        "the retry does not name the card option, so it is shadowed by the "
        "refused one from the Explore door")


@pytest.mark.parametrize("mechanism", ["not_informative", "not_sure"])
def test_the_purpose_contraindications_resolve_opens(mechanism):
    """`GUIDED-183` itself. The row's own exit, taken.

    `blocks_indicator` fires on the recorded purpose alone, so this reproduces
    on any mechanism — and on `informative` it collides with clause §07's own
    blocker, which the next test drives and reports rather than hides.
    """
    client, pid = _project("clinic_visits.csv", "outcome", purpose="inference")
    exit_row, took = _take_the_resolve(client, pid, {
        "kind": "route_missingness", "subject": "Unnamed: 0",
        "payload": {"column": "Unnamed: 0", "card_option": "indicator",
                    "mechanism": mechanism}})
    assert exit_row["id"] == "impute_median"
    assert took.status_code == 200, (
        f"the purpose contraindication's SAFE exit does not open on a "
        f"{mechanism} mechanism ({took.status_code}): "
        f"{str(took.json())[:300]}")


def test_the_two_blockers_collide_and_this_says_so(capsys):
    """NOT a fix — the measurement, published.

    On `informative` + `inference` + `indicator`, both constitutional rules
    fire: the purpose blocker offers `impute_median` as the way out, and clause
    §07 refuses exactly that because the user has said a blank means something.
    Each rule is right on its own. The user's way through is neither exit.

    Asserted as the CURRENT behavior so the collision cannot be quietly
    resolved in one direction without this failing and someone deciding it.
    """
    client, pid = _project("clinic_visits.csv", "outcome", purpose="inference")
    request = {"kind": "route_missingness", "subject": "Unnamed: 0",
               "payload": {"column": "Unnamed: 0", "card_option": "indicator",
                           "mechanism": "informative"}}
    exit_row, took = _take_the_resolve(client, pid, request)
    with capsys.disabled():
        print("\n  ── L48-E · the two blockers that collide ──")
        print(f"  first refusal   409 · indicator_under_inference")
        print(f"  its resolve     {exit_row['id']}")
        print(f"  taking it       {took.status_code} · "
              f"{took.json().get('detail', {}).get('kind') if took.status_code != 200 else 'accepted'}")
        print("  Both rules are correct. Filed, not resolved here.")
    assert took.status_code == 409, (
        "the collision has been resolved in one direction. That is a "
        "constitutional decision and it needs a row, not a passing test")
    assert took.json()["detail"]["kind"] == "blocker", (
        "the second refusal is no longer clause §07's mechanism blocker")


def test_the_page_renders_a_resolve_with_a_retry_as_live(capsys):
    """The consumer. `showRefusal` enables an exit that carries a payload.

    `GUIDED-183`'s visible half: the row is about a button rendering `disabled`,
    so a test that only checked the payload would be asserting the server's side
    of a defect that lives at the boundary.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")
    from turbotab import purpose as _purpose

    blocker = _purpose.indicator_blocker("hs_crp")
    client, pid = _project("clinic_visits.csv", "outcome", purpose="inference")
    routes = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "capabilities", "features",
                 "recipes", "preprocess", "figures", "draft", "manuscript",
                 "models", "training", "instability", "explain", "sensitivity",
                 "evidence/plausibility", "evidence/missingness"):
        resp = client.get(f"/project/{pid}/{path}")
        routes[f"/project/{pid}/{path}"] = (resp.json() if resp.status_code == 200
                                            else {})
    routes[f"POST /project/{pid}/decision"] = {"__status": 409,
                                               "body": {"detail": blocker}}

    out = PH.run(
        "__harness.dispatch('click', __harness.target("
        "  {'data-task': 'regression', 'data-ac': 'task'}));\n"
        "for (var i = 0; i < 8; i++) await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({band: __harness.html('refusal')});",
        routes=routes, search=f"?project={pid}")

    band = out["band"] or ""
    assert "Impute instead" in band, (
        f"the purpose blocker's resolve did not render at all: {band[:300]!r}")
    resolve_button = band[:band.index("Impute instead")]
    assert "disabled" not in resolve_button.rsplit("<button", 1)[-1], (
        "the SAFE way out still renders `disabled`, beside a live "
        f"'keep the indicator': {band[:400]!r}")


def test_the_sweep_names_every_resolve_exit_and_what_it_can_do(capsys):
    """Coverage, and the two this loop did not fix.

    `LOOP.md` §10: a sweep that reports only what it fixed has not reported its
    coverage. Five resolve exits, two fixed, three named with reasons.
    """
    from turbotab import clinical, exits, grain, missingness, packs, purpose

    # COUNTED AS COMPOSED OBJECTS, NOT AS A SOURCE PATTERN — and the change is
    # `TEST-048`'s lesson at a much smaller scale. This counted
    # `"kind": "resolve"` with a regex over five modules, which is a grep
    # answering *does this text appear* when the question is *how many resolve
    # exits does this app compose* (trap #5). L49-E moved two of them from
    # literal dicts to `exits.revise()` calls and the count dropped from five
    # to three with nothing about the app having changed — a sweep reporting a
    # refactor as a disappearance.
    composed = [("grain._RESOLVE", grain._RESOLVE),
                ("packs._LENS_RESOLVE", packs._LENS_RESOLVE),
                ("purpose.INDICATOR_EXITS[0]", purpose.INDICATOR_EXITS[0]),
                ("clinical.substitution_blocker",
                 [e for e in clinical.substitution_blocker("hs_crp", 0.19)["exits"]
                  if e["kind"] == exits.RESOLVE][0]),
                ("missingness.blocker_exits",
                 [e for e in missingness.blocker_exits("categorical")
                  if e["kind"] == exits.RESOLVE][0])]
    for where, row in composed:
        assert row["kind"] == exits.RESOLVE, where
    found = len(composed)

    with capsys.disabled():
        print("\n  ── L48-E · every resolve exit in the app ──")
        print(f"  resolve exits composed              {found}")
        for where, what in RESOLVE_EXITS.items():
            print(f"      {where:<32} {what}")
        print("  fixed this loop: missingness.blocker_exits (the retry was")
        print("  shadowed) and purpose.INDICATOR_EXITS[0] (there was none).")
        print("  NOT fixed: the two REVISE-shaped exits render `disabled`")
        print("  because they honestly have no payload — the page needs a")
        print("  mechanism the server describes, which is a design decision.")

    assert found == len(RESOLVE_EXITS), (
        f"{found} resolve exits are composed and {len(RESOLVE_EXITS)} are "
        f"named here. A sweep whose list has drifted from the code is worse "
        f"than no list")
    assert {w for w, _ in composed} == set(RESOLVE_EXITS), (
        "the composed exits and the named ones are the same COUNT and not the "
        "same SET, which is the drift this assertion was meant to catch "
        "passing on arithmetic")
    # The two revise-shaped ones are asserted as they are, so that giving one a
    # payload without revisiting this table fails here rather than silently.
    assert not grain._RESOLVE.get("retry"), (
        "grain's revise exit now carries a retry — a revise exit is not a "
        "request, and inventing one would re-post what was just refused")
    assert not packs._LENS_RESOLVE.get("retry")
    # L49-E: both now describe how they ARE taken, which is what let the page
    # stop greying them out. `GUIDED-184`.
    for where, row in composed:
        if not (row.get("retry") or {}).get("payload"):
            assert (row.get("takes") or {}).get("action") or \
                   where == "clinical.substitution_blocker", (
                f"{where} carries neither a retry payload nor a described "
                f"action, so `showRefusal` renders it disabled")
    assert exits.is_actionable(grain._RESOLVE), (
        "`is_actionable` and the page disagree again about the same exit")
