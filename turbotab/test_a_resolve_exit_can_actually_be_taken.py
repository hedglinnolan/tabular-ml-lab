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
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

from pathlib import Path


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
