"""`GUIDED-039` — six lens options, one hover sentence between them.

Hovering *Metabolomics or proteomics* said exactly what hovering *Dietary
intake* said: `effectOf("set_lens", null)`, the question-level effect, six
times. Picking a lens is a bet on what the app will then notice, and the one
moment where the app could say what it will notice said nothing.

`DOMAIN_PACKS.md` §02 is explicit that the payoff is a **finding a generic tool
would never raise** — left-censored missingness, run-order drift, pooled QC
rows. Those exist, in the detectors, and the control that offers them named
none of them.

## The shape of the fix, and why it is a registry

The obvious repair is six strings in the page. That is the failure mode this
project has a corollary for: a description written beside the thing it
describes drifts from it, and nothing says so. `FEATURE_PARITY.md`'s "two
specific things to watch" names the exact case —

> `utils/theory_anchors.py` and `utils/theory_demos.py` are a 19-key registry
> pair with **no test asserting the keys match** … the most fragile intelligent
> feature in the app and the most likely to quietly not survive a rewrite.

So `Pack.looks_for` binds each phrase to the **source** that will produce it,
and this file is the key-match test that pair never got. It asserts in both
directions:

* every detector a pack declares emits a finding id that some `looks_for` names
  — a detector nobody promised is a surprise;
* every `pack::`-sourced `looks_for` names an id a detector actually emits — a
  promise nobody keeps is the worse half, because it is the app asserting it
  will look for something it will not.

The ids are checked by **running the detectors against the fixture that pack
was built for**, not by reading a list. A list is the thing that goes stale,
which is `test_the_page_renders_every_question_the_router_can_serve`'s own
argument reused.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402
from turbotab import packs as P                                       # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

# The fixtures each pack was built against — `OPENING_SEQUENCE.md` §04's table.
# `other` has no pack and no fixture, which is the point of it.
#
# **A TUPLE, and the dietary pack is why** (`GUIDED-058`). It was one name per
# pack until the nutrition module's four detectors were wired in, and two of
# them have preconditions that are exact negations of each other:
# `partial_design` needs a weight with no strata or PSU, `lonely_psu` needs both
# present and a stratum holding one. **No single table can exercise both**, so a
# one-fixture registry could not promise both, and a promise the registry cannot
# make is a capability the user cannot know they are buying — which is the half
# of the key match this file calls the worse one.
FIXTURE = {
    P.METABOLOMICS: ("metabolomics_untargeted",),
    P.GENOMICS: ("genomics_expression",),
    P.DIETARY: ("dietary_recalls", "nhanes_dietary", "nhanes_partial_design",
                "nhanes_kilojoules"),
    P.SURVEY: ("survey_instrument",),
    P.CLINICAL: ("clinic_visits",),
}


def _emitted(pack_key: str) -> set:
    """The finding ids this pack's detectors actually produce, run for real.

    Unioned across the pack's fixtures. A detector still has to fire on one of
    them — the union is what lets mutually exclusive preconditions both be
    promised, not a way to promise something nothing triggers.
    """
    out = set()
    for name in FIXTURE[pack_key]:
        df = pd.read_csv(DATA / f"{name}.csv")
        for detector in P.PACKS[pack_key].detectors:
            found = detector(df)
            if found:
                out.add(found["id"])
    return out


@pytest.mark.parametrize("key", sorted(FIXTURE))
def test_every_promise_a_pack_makes_is_one_a_detector_keeps(key):
    """A `pack::` phrase whose detector emits nothing is the app announcing it
    will look for something it will not — the assert-something-false branch, on
    a control the user reads before answering."""
    promised = {lf.source for lf in P.PACKS[key].looks_for
                if lf.source.startswith("pack::")}
    emitted = _emitted(key)
    orphans = sorted(promised - emitted)
    assert not orphans, (
        f"the {key} pack's hover promises these and no detector emits them on "
        f"any of {list(FIXTURE[key])}: {orphans}\n  emitted: {sorted(emitted)}")


@pytest.mark.parametrize("key", sorted(FIXTURE))
def test_every_detector_a_pack_runs_is_one_its_hover_names(key):
    """The other direction. A detector added without a phrase is a capability
    the user cannot know they are buying, which is how a lens question stops
    being answerable on its merits."""
    named = {lf.source for lf in P.PACKS[key].looks_for}
    unnamed = sorted(_emitted(key) - named)
    assert not unnamed, (
        f"the {key} pack fires these on its own fixture and its hover names "
        f"none of them: {unnamed}. Add a `LooksFor` beside the detector.")


@pytest.mark.parametrize("key", sorted(FIXTURE))
def test_the_key_match_is_not_passing_on_an_empty_set(key):
    """*A check nothing triggers is a check that does not exist.*

    Both directions above compare `_emitted(key)` against the declared sources,
    and both pass trivially when `_emitted` is empty — which it is for a pack
    with no detectors. So the packs that DO declare detectors are asserted to
    fire them on the fixture the pack was built for, and `clinical`'s emptiness
    is asserted as the deliberate thing it is rather than left to look like the
    same silence.
    """
    emitted = _emitted(key)
    if P.PACKS[key].detectors:
        assert emitted, (
            f"the {key} pack declares {len(P.PACKS[key].detectors)} detector(s) "
            f"and none fires on any of {list(FIXTURE[key])}, so both key-match "
            f"assertions above are comparing against nothing")
    else:
        assert key == P.CLINICAL and not emitted, (
            f"{key} has no detectors; if that is now false the fixture table "
            f"above is stale")


@pytest.mark.parametrize("key", sorted(FIXTURE))
def test_a_prior_sourced_phrase_names_a_prior_that_exists(key):
    """The `prior::` half of the same key match.

    `clinical` and `genomics` promise things no detector emits — a prior, and a
    considered refusal to set a default. Both are legitimate, and both are one
    typo away from being a promise bound to nothing at all.
    """
    declared = {p.question for p in P.PACKS[key].priors}
    promised = {lf.source.split("::", 1)[1] for lf in P.PACKS[key].looks_for
                if lf.source.startswith("prior::")}
    assert promised <= declared, (
        f"the {key} pack's hover cites priors it does not set: "
        f"{sorted(promised - declared)}\n  sets: {sorted(declared)}")


def test_every_lens_option_says_something_different_from_every_other():
    """The defect itself, stated as the property it violated.

    Asserted on DISTINCTNESS rather than on any particular wording, because a
    test naming the metabolomics sentence would pass with the other five still
    identical — which is precisely the state that shipped.
    """
    notes = {k: P.option_note(k) for k in P.LENS_KEYS}
    assert len(set(notes.values())) == len(P.LENS_KEYS), (
        "two lens options carry the same hover, so the control cannot be read "
        f"as a choice between them:\n" +
        "\n".join(f"  {k}: {v[:70]}…" for k, v in notes.items()))
    for k, note in notes.items():
        assert len(note) > 80, f"{k}'s note says nothing: {note!r}"


def test_the_hover_names_what_is_looked_for_and_never_what_was_found():
    """The rule that keeps this honest.

    *"Your missing values cluster in the lowest-abundance features"* is the
    finding, and it is true only of a table that has been read. On a hover,
    before the question is answered, it would assert that of every table. So the
    phrases are noun phrases, and the finding stays the only thing that reports.

    Checked as the ABSENCE of a category — second-person possessives and
    past-tense report verbs — because `FEATURE_PARITY.md` prefers asserting the
    absence of a whole class where the guarantee is a subtraction.
    """
    banned = ("your ", "we found", "was found", "were found", "this table has",
              "there is a run-order column")
    for key in P.LENS_KEYS:
        low = P.option_note(key).lower()
        hits = [b for b in banned if b in low]
        assert not hits, (
            f"{key}'s hover claims something about a table nobody has read "
            f"yet: {hits} in {P.option_note(key)!r}")


def test_the_notes_reach_the_question_the_router_serves():
    """A phrase the interview does not carry is a phrase nobody reads.

    `DRIVE-001`'s lesson applied to content rather than to a question: built,
    correct, and not on the wire is the same as not built.
    """
    plan = router.plan([], target=None, detection=None, step="data",
                       deferred={}, answered=[], recommendations=[],
                       signals=None, missing_columns=[])
    lens = next(q.to_dict() for q in plan if q.key == "state_lens")
    assert len(lens["option_notes"]) == len(lens["options"])
    assert lens["option_notes"] == [P.option_note(k) for k in P.LENS_KEYS]


def test_the_page_shows_the_note_on_the_option_it_belongs_to():
    """Read back off the render, not off a grep.

    The tailored sentence has to land on the button the user hovers, and
    `option_notes` is positional — an off-by-one here would put the
    metabolomics sentence on the genomics option, which is worse than the shared
    string it replaced.
    """
    from turbotab import pageharness as H
    if not H.available():
        pytest.skip("no JS engine on this machine")

    from fastapi.testclient import TestClient

    from turbotab import api
    client = TestClient(api.app)
    with open(DATA / "metabolomics_untargeted.csv", "rb") as fh:
        project = client.post("/project", files={
            "file": ("m.csv", fh, "text/csv")}).json()
    pid = project["id"]
    html = H.run("__emit(__harness.html('askedQuestions'));", routes={
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/evidence/missingness": {"cards": []},
    }, search=f"?project={pid}")

    shown = {b["data-answer-value"]: b.get("data-tip", "")
             for b in H.elements(html)
             if b.get("data-answer-key") == "state_lens"}
    assert shown, "the lens card rendered no options"
    for key in P.LENS_KEYS:
        assert shown[key] == P.option_note(key), (
            f"the {key} option carries the wrong sentence:\n"
            f"  shown:    {shown[key][:90]!r}\n"
            f"  expected: {P.option_note(key)[:90]!r}")
