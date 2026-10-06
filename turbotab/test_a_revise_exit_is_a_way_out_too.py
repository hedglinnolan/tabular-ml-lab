"""L49-E — `GUIDED-184` and `GUIDED-187`, which are one decision.

`GUIDED-183` closed at L48 and its report said it was bigger than the row: of
five resolve exits, one carried a retry, one was given one, one has no consumer
at all, and **two are not requests.** `grain._RESOLVE` and `packs._LENS_RESOLVE`
both mean *go back to the question and answer it differently* — there is no
request to re-post, so they carried no `retry.payload`, and `showRefusal`
enables on `retry.payload`.

So the **safe** way out rendered `disabled` beside a live *"continue anyway"*,
on three more surfaces than `GUIDED-183` named. That is §09's choice inverted:
a consequence resolves or is attested, and only one of the two was pressable.

## Why the page could not fix this alone, and why that was right

The page's own stated rule is that **it will not invent a mechanism for a way
out the server did not describe**, which is why L48 filed this rather than
patching `showRefusal` to enable every resolve. The rule is correct and it was
the obstacle. So the server describes it: `exits.revise` builds an exit with
`takes.action = "revise"` and a `how` sentence, and the page implements exactly
that one verb. An action it does not recognize stays disabled — the same
refusal as before for anything undescribed.

## `GUIDED-187` is the same decision seen from the predicate

`exits.is_actionable` returned `True` for **every** non-attest exit, on the
stated grounds that *a resolve exit sends the user back to the question and
needs nothing*. That premise stopped being the page's the day `showRefusal`
began reading the payload rather than the kind — its own comment records the
change — so the unifying test for `GUIDED-064` and `GUIDED-072` was passing on
exits nobody could take. It reads what the page reads now, in the same words.

**Deciding them together was the point.** Changing the predicate before the
mechanism existed would only have moved the wrong answer.

## What is NOT covered

- **Whether the enabled button is on screen.** Nothing without layout can say.
- **`clinical.substitution_blocker`'s `keep_censored`** — the fifth resolve
  exit, and `GUIDED-185`: it has no consumer anywhere outside a test, so there
  is no refusal to render and nothing to take. Left alone deliberately.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

from pathlib import Path


DATA = Path(__file__).resolve().parent / "sample_data"


def test_both_revise_exits_describe_how_they_are_taken():
    """The server's half. A hand-written dict is what left them mute."""
    from turbotab import exits, grain, packs

    for name, exit_row in (("grain._RESOLVE", grain._RESOLVE),
                           ("packs._LENS_RESOLVE", packs._LENS_RESOLVE)):
        assert exit_row["kind"] == exits.RESOLVE, name
        takes = exit_row.get("takes") or {}
        assert takes.get("action") == exits.REVISE, (
            f"{name} carries no way to be taken, so `showRefusal` renders it "
            f"`disabled` — the safe exit greyed out beside a live attestation")
        assert takes.get("how"), (
            f"{name} names an action and does not say what taking it does. The "
            f"page implements the verb; the SENTENCE is what a person reads")
        assert not (exit_row.get("retry") or {}).get("payload"), (
            f"{name} grew a retry payload. It is not a request — inventing one "
            f"would re-post the request that was just refused, which is "
            f"`GUIDED-183`'s own finding")


def test_the_predicate_agrees_with_the_page_now():
    """`GUIDED-187`. Two answers to one question was the defect."""
    from turbotab import exits, grain, missingness, packs

    assert exits.is_actionable(grain._RESOLVE)
    assert exits.is_actionable(packs._LENS_RESOLVE)
    # The retry-carrying kind still passes, unchanged.
    resolve = [e for e in missingness.blocker_exits("categorical")
               if e["kind"] == exits.RESOLVE][0]
    assert exits.is_actionable(resolve)
    # AND IT STILL SAYS NO. A predicate that answers yes to everything is the
    # defect with the sign flipped.
    assert not exits.is_actionable(
        {"id": "x", "kind": "resolve", "label": "Mute", "detail": ""}), (
        "a resolve exit carrying neither a retry payload nor a described "
        "action is called actionable, so the predicate is back to answering "
        "from `kind` — which is the premise the page stopped holding")
    assert not exits.is_actionable(
        {"id": "y", "kind": "resolve", "takes": {"action": "teleport"}}), (
        "an action the page does not implement is called actionable")


def test_the_grain_contradiction_is_the_same_shape():
    """The second surface, asserted on the composed object.

    Driven end to end for the lens above; the grain contradiction needs a
    repeated-measures journey to reach, which is a different fixture chain —
    said here rather than left as uncounted coverage.
    """
    from turbotab import exits, grain

    for exit_row in grain._EXITS_STATED_UNIQUE + grain._EXITS_STATED_REPEATS:
        assert exits.is_actionable(exit_row), (
            f"{exit_row.get('id')} on a grain contradiction cannot be taken by "
            f"a client holding only the payload")
    resolves = [e for e in grain._EXITS_STATED_UNIQUE
                if e["kind"] == exits.RESOLVE]
    assert resolves and (resolves[0].get("takes") or {}).get("action") == "revise"
