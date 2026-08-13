"""Answer *which level is the event*, the way a user does — one place.

**`DRIVE-041`, and it is the consumer side of `DRIVE-032`.** `L60-A` made the
fit refuse while nobody has said which level of a two-level outcome is the
event. Seventy-four tests across twenty files fitted a classification without
ever choosing one and went red — **not because the fix broke them, but because
they were passing by virtue of the defect.** They are the population that
depended on the wrong default, and `LOOP.md` §05's rule is that a capability
ships with its consumer. Here the fixtures ARE the consumer.

## Why this is not a fixture helper that writes to the project

`AGENT_ONBOARD.md` trap #3: **a guard that manufactures the thing whose absence
is the defect.** A fixture that appended a decision, or wrote `0`/`1` into the
column itself, would satisfy `training.check()` while the door it stands in for
stayed shut — and seventy-four tests would go green over a control no user can
reach. That is `DRIVE-017` written into the test suite.

So both entry points below end in `engine.record_fix`, which is the function
`api.py`'s `apply` branch calls. **A fixture's answer travels the same code as
a person's.** If the engine stops raising `positive_class__<target>`, if the
identity barrier starts refusing, if the repair stops encoding — these fixtures
find out, and they find out for the real reason.

## Which level a fixture picks, and why that is not a research question

A fixture has no research question, so it takes **the level the engine itself
names as conventional**, and where the engine names none, the last level in the
engine's own sorted order. Deterministic, and declared here rather than left to
be inferred from a fixture's numbers.

Two consequences worth knowing before reading a diff that uses this:

* on a target already stored as `0`/`1` the convention is `1`, so the encoding
  is the identity and no fixture's numbers move;
* on `responder` / `non-responder` the convention is `responder`, and the
  column becomes `0`/`1` — because **recording the answer IS the encoding.**
  After `L60-A` there is no way to fit a two-level target with the level names
  still in it, which is `DRIVE-040`'s whole subject and why the figure now
  carries the name separately.

Where a test's claim depends on WHICH level was chosen, it passes `level` and
says why in the same breath.
"""
from __future__ import annotations

from typing import Any, Optional


def question_id(target: str) -> str:
    """The finding id the engine raises for a two-level outcome."""
    return f"positive_class__{target}"


def offered_levels(project: Any) -> Optional[list]:
    """The levels the engine offers for this project's outcome, or `None`.

    `None` means there is no question to answer — a regression target, a
    three-level one, or a column the engine sees no two-level plan in. Asking
    is how a caller tells *nothing to record* from *nothing recorded*.
    """
    from turbotab import engine

    target = str(project.target or "")
    if not target:
        return None
    for finding in engine.diagnose(project.df, target=target):
        if finding.id == question_id(target):
            return list((finding.params or {}).get("levels") or [])
    return None


def _pick(project: Any, level: Optional[str]) -> Optional[str]:
    from turbotab import engine

    target = str(project.target or "")
    if not target:
        return None
    for finding in engine.diagnose(project.df, target=target):
        if finding.id != question_id(target):
            continue
        params = finding.params or {}
        levels = list(params.get("levels") or [])
        if not levels:
            return None
        if level is not None:
            assert level in levels, (
                f"{level!r} is not one of {levels!r} for {target!r}")
            return level
        return str(params.get("suggested") or levels[-1])
    return None


def choose_event(project: Any, *, level: Optional[str] = None,
                 required: bool = False) -> Optional[str]:
    """Record the event on an in-process project. Returns the level, or `None`.

    `required=True` asserts the question was actually raised, for the fixtures
    whose whole point is a two-level outcome — without it, a fixture whose
    target silently stopped being two-level would pass by answering nothing.
    """
    from turbotab import engine

    chosen = _pick(project, level)
    if chosen is None:
        assert not required, (
            f"the engine raised no {question_id(str(project.target))!r} for "
            f"this project, so there is no event question to answer — check "
            f"the target actually has two levels")
        return None
    engine.record_fix(project, question_id(str(project.target)), choice=chosen)
    return chosen


def choose_event_over_http(client: Any, pid: str, target: str, *,
                           level: Optional[str] = None,
                           required: bool = False) -> Optional[str]:
    """The same answer, posted as a decision — for fixtures that hold a client.

    It reads the levels off `GET /project/{id}/findings` rather than off the
    engine, so the fixture is answering the question **the page was shown**.
    A fixture that diagnosed for itself and posted the answer would agree with
    the route by construction and could not notice it drifting.
    """
    subject = question_id(target)
    served = client.get(f"/project/{pid}/findings")
    assert served.status_code == 200, served.text[:300]
    live = next((f for f in served.json().get("findings", [])
                 if f.get("id") == subject), None)
    if live is None:
        assert not required, (
            f"no {subject!r} finding is served for this project, so there is "
            f"no event question to answer")
        return None
    params = live.get("params") or {}
    levels = list(params.get("levels") or [])
    assert levels, f"{subject} is served with no levels to choose between"
    if level is not None:
        assert level in levels, f"{level!r} is not one of {levels!r}"
        chosen = level
    else:
        chosen = str(params.get("suggested") or levels[-1])

    answered = client.post(f"/project/{pid}/decision",
                           json={"kind": "apply", "subject": subject,
                                 "payload": {"choice": chosen}})
    assert answered.status_code == 200, answered.text[:300]
    return chosen
