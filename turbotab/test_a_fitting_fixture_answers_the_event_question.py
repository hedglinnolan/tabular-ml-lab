"""How a fixture is allowed to answer the event question — `DRIVE-041`, `A3`.

**What happened, in one paragraph.** `L60-A` made the fit refuse while nobody
has said which level of a two-level outcome is the event. Before building it,
the loop measured the blast radius by grepping test files for `POST "/train"`,
found three, and called it small. The suite does not reach the fit that way:
nine files call `training.train()` in process, and the sweep came back with
**seventy-four** red tests across twenty files. *"The grep answered 'does this
text appear' when the question was 'does this run'"* — trap #5, in the one
measurement the part's scope rested on.

**This file guards the repair rather than the count.** Fixing seventy-four
tests means seventy-four chances to take the cheap way out, and the cheap way
out has a name: `AGENT_ONBOARD.md` trap #3, *a guard that manufactures the
thing whose absence is the defect.* A fixture that appended an `apply` decision
itself, or wrote `0`/`1` into the outcome column by hand, would satisfy
`training.check()` while the door it stands in for stayed shut — and the suite
would go green over a control no user can reach. That is `DRIVE-017` written
into the test suite, and it would pass every sweep.

So the rule is narrow and checkable: **a test may name
`positive_class__<target>` only where it is going through the real answering
path** — `turbotab.eventfixture`, which posts the decision the page posts and
ends in `engine.record_fix`, the function `api.py`'s `apply` branch calls. A
file that names the subject and rides neither is constructing the record, and
that is the one shape no sweep can see.

**What this file does NOT claim.** It is not an inventory of which fixtures
must answer — that depends on each parametrization's target shape, and a
static guess at it is the kind of reasoning-about-behavior this project keeps
paying for. A file that fits a classification and forgets to answer simply goes
red in the suite, loudly, with the refusal quoted. The inventory below is
printed rather than asserted for the same reason.

The positive control at the bottom is not optional: a sweep that finds nothing
because its pattern is wrong reports a clean tree it never looked at, and the
pattern here is the one that was already got wrong once.
"""
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent

SUBJECT = "positive_class__"

#: The legitimate ways to answer, all of which end in `engine.record_fix`.
#:
#: **Four spellings, not one, and that is the lesson rather than an
#: inconvenience.** The twenty files this loop touched import the helper as
#: `eventfixture`, as `_EF` and as `eventfixture as _EF`, and the calls read
#: `choose_event` or `choose_event_over_http`. A detector that looked for the
#: module name alone would report every aliased file as an offender.
_ANSWERING_PATHS = ("eventfixture", "choose_event", "engine.record_fix")

#: Files allowed to name the subject without importing the helper, each with
#: the reason. **Both entries are checked below**, so a stale exemption fails
#: rather than lingering.
NAMES_THE_SUBJECT_DIRECTLY = {
    "test_the_event_is_chosen_and_never_guessed.py":
        "The acceptance test for the refusal itself. It posts the `apply` "
        "decision by hand BECAUSE that is the behavior under test — the route, "
        "its 400 with no choice, and the sentence it records. Going through "
        "the helper would test the helper.",
    "eventfixture.py":
        "The helper itself. It composes the subject, posts the decision and "
        "calls `engine.record_fix`, so naming the subject is the whole of what "
        "it is for — exempting it is not a concession, it is the definition.",
}


def _modules():
    return sorted(p for p in ROOT.glob("*.py")
                  if p.name != pathlib.Path(__file__).name)


def _names_the_subject():
    out = []
    for path in _modules():
        source = path.read_text(encoding="utf-8")
        if SUBJECT in source:
            out.append((path, source))
    return out


# ── the rule ────────────────────────────────────────────────────────────────

def test_no_fixture_constructs_the_event_record_for_itself():
    """A test that names the subject goes through the real answering path.

    The failure this prevents is not a red test — it is a GREEN one. A fixture
    that records the decision itself fits happily, and the door a user would
    have to walk through is never touched by anything.
    """
    offenders = []
    for path, source in _names_the_subject():
        if not path.name.startswith("test_"):
            continue
        if path.name in NAMES_THE_SUBJECT_DIRECTLY:
            continue
        if any(marker in source for marker in _ANSWERING_PATHS):
            continue
        offenders.append(path.name)
    assert not offenders, (
        f"these test modules name {SUBJECT!r} and reach neither "
        f"`turbotab.eventfixture` nor `engine.record_fix`: {offenders}. "
        f"Constructing the record directly satisfies the fit gate without "
        f"touching the control a user would use, which is a green test over a "
        f"shut door. Answer through the helper, or add the file to "
        f"NAMES_THE_SUBJECT_DIRECTLY with the reason it is the exception.")


def test_no_fixture_encodes_the_outcome_by_hand_to_dodge_the_gate():
    """The other way out, and the quieter one.

    The gate reads the DECISION rather than the dtype precisely so a `0`/`1`
    column cannot pass as an answered question — `training.event_not_chosen`
    says so in its own docstring. A fixture that mapped its outcome to `0`/`1`
    before fitting would still be refused, so the failure mode is not that this
    works; it is that somebody tries it, reads the refusal, and reaches for the
    record next. This asserts the property that makes the first attempt fail.
    """
    from turbotab import training as _training
    from turbotab.project import AnalysisProject
    import pandas as _pd

    rng = __import__("numpy").random.default_rng(3)
    frame = _pd.DataFrame({"x": rng.normal(0, 1, 200),
                           "y": rng.integers(0, 2, 200)})
    project = AnalysisProject.from_dataframe(frame, "hand_encoded.csv")
    project.set_target("y", "classification", "high", [])
    assert _training.event_not_chosen(project), (
        "a 0/1 outcome with nothing recorded is not being refused, so the gate "
        "has started reading the dtype and a hand-encoded fixture would pass")


def test_no_exemption_here_has_outlived_its_reason():
    """A file listed as naming the subject directly that no longer names it is
    an exemption describing a state that does not exist."""
    named = {path.name for path, _ in _names_the_subject()}
    stale = [name for name in NAMES_THE_SUBJECT_DIRECTLY if name not in named]
    assert not stale, f"{stale} no longer name {SUBJECT!r} at all"
    for name, reason in NAMES_THE_SUBJECT_DIRECTLY.items():
        assert len(reason) > 80, f"{name}: the reason is a shrug"


# ── the controls ────────────────────────────────────────────────────────────

def test_the_sweep_can_see_the_shape_it_is_looking_for(capsys):
    """**The positive control.** Both assertions above are absences.

    It also prints the inventory, because the number is the thing `L60`
    reported wrong and the next loop should not have to re-derive it.
    """
    named = [path.name for path, _ in _names_the_subject()]
    # **Counted separately from the set above, and the reason is the point.**
    # `eventfixture` composes the subject with an f-string, so a file that
    # answers correctly does NOT contain the literal — a literal search for it
    # finds the helper and the acceptance test and nothing else. That is trap
    # #5 one layer in, in this file's own instrument.
    answering = [path.name for path in _modules()
                 if path.name.startswith("test_")
                 and any(m in path.read_text(encoding="utf-8")
                         for m in _ANSWERING_PATHS)]
    assert len(named) >= 2, (
        f"only {named} name the subject at all; the helper and the acceptance "
        f"test both do, so a number this small means the pattern stopped "
        f"matching")
    assert len(answering) >= 10, (
        f"only {len(answering)} test module(s) reach the answering path; "
        f"twenty files were wired to it at `L61`, so a number this small "
        f"means fixtures have started answering some other way")
    with capsys.disabled():
        print(f"\n  {len(named)} module(s) name {SUBJECT!r} literally; "
              f"{len(answering)} test module(s) answer through the helper")


@pytest.mark.parametrize("spelling", [
    "eventfixture.choose_event(p)",
    "_EF.choose_event_over_http(client, pid, target)",
    "from turbotab import eventfixture",
    "engine.record_fix(project, subject, choice=level)",
])
def test_the_detector_recognizes_every_way_the_helper_is_reached(spelling):
    """The negative control's negative control. The import is aliased three
    different ways across the twenty files this loop touched, and a detector
    that saw only one of them would report offenders that are fine."""
    assert any(marker in spelling for marker in _ANSWERING_PATHS), spelling
