"""`GUIDED-159` — the analysis map, driven, and the fourth state it never had.

Two defects in one rail, and they are separable.

**(a) `now` was never computed.** Of the eleven `setMap` call sites in
`web/index.html`, exactly two passed `now` on purpose: `renderTarget`'s
unanswered branch, and `renderEda` — which passed it unconditionally, on every
render, for the whole life of the project. Every other `now` on the map came
from `reveal()`, which set it on any section it uncovered; and `renderAll`
uncovers every downstream section the moment a target exists. Driven on
`clinic_visits.csv` past the seal, one render left **six** steps wearing `now`
at once — `eda`, `features`, `preprocess`, `train`, `explain`, `report` — with
the last writer decided by which fetch resolved last. §02 gives `--accent` one
meaning and one budget, *the current position, one accent moment per viewport*.
Six of them is not a highlight, and Train never had the dot on purpose.

**(b) The fourth state was specified and never built.** `DESIGN_LANGUAGE.md`
§04 names four — `done · now · waiting · stale` — and the CSS defined three.
A step nobody has reached wore the bare `.map-step` default: `--faint` ink and
a hollow ring, which the product owner read as a verdict about availability
rather than as *not yet*. That is §09's recorded-absence rule one surface over:
**an absence must be legible as an absence, never as a verdict.**

## What is driven, and what is only read

`test_the_map_defines_the_four_states_its_design_language_names` is a claim
about the FILE — whether a CSS rule exists — so it greps, and that is trap 5
used the right way round.

Everything else is a drive. The map is read off the rendered dots at four
points in a real journey, so `now` is observed where the user is rather than
inferred from a call site.

**The load-bearing assertion is a TRANSITION, not a state** (trap 3). The
markup now seeds the seven downstream dots as `waiting`, so an assertion that
Train reads `waiting` before the seal could be satisfied by the fixture alone.
The claim this file is really making is that Train reads `waiting` at one stage
and `now` at the next, and nothing but the fold can produce the second.

Two fixtures of different target shape (`GUIDED-097`): `clinic_visits.csv` has
a two-level string outcome and drives a classification journey,
`metabolomics_untargeted.csv` a continuous `bmi` and drives a regression one.
The shapes not covered are named in `SHAPES_NOT_COVERED`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "sample_data"
PAGE = Path(__file__).resolve().parent / "web" / "index.html"

#: NOT COVERED, said out loud.
SHAPES_NOT_COVERED = (
    "Whether the dot is on screen, or what color it actually paints. Nothing "
    "without layout can tell, and `pageharness.py` says so in its own "
    "docstring. This reads the state the rail was put into, never a pixel.",
    "A multiclass target. Both fixtures here are binary-vs-continuous, which "
    "is the fork the map's arithmetic could plausibly care about; the number "
    "of classes is not on any path this rail reads.",
    "A project whose journey has been REOPENED after Report was written — the "
    "`stale` state is exercised only where a step records it, never by "
    "un-recording a downstream step and watching the front retreat.",
)

#: The eight steps, in the order the rail draws them.
STEPS = ("data", "target", "eda", "features", "preprocess", "train",
         "explain", "report")

#: fixture -> the target that reaches the journey, and its shape.
TARGETS = {"clinic_visits.csv": "outcome",           # two-level string
           "metabolomics_untargeted.csv": "bmi"}     # continuous

_PATHS = ("interview?step=data", "interview?step=explore",
          "interview?step=features", "capabilities", "features", "recipes",
          "preprocess", "figures", "draft", "manuscript", "models", "training",
          "instability", "explain", "sensitivity", "evidence/plausibility",
          "evidence/missingness")


def _routes(client, pid):
    """The eighteen answers one render asks for, captured from the real API."""
    out = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in _PATHS:
        resp = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = (resp.json() if resp.status_code == 200
                                         else {})
    return out


def _decide(client, pid, kind, **payload):
    resp = client.post(f"/project/{pid}/decision",
                       json={"kind": kind, "payload": payload})
    assert resp.status_code < 400, (
        f"{kind} refused ({resp.status_code}): {resp.text[:300]} — this drive "
        f"is pinned to a journey the server accepts")
    return resp.json()


def _read_map(client, pid):
    """Drive one render and report the state of all eight dots.

    Reads the class the rail actually put on each dot AND the `data-map-state`
    attribute beside it, because trap 7 is that the machine-readable form ends
    up lossier than the treatment; a caller here can see both disagree.
    """
    from turbotab import pageharness as PH

    routes = _routes(client, pid)
    out = PH.run(
        "var STEPS = " + json.dumps(list(STEPS)) + ";\n"
        "var got = {};\n"
        "STEPS.forEach(function(k){\n"
        "  var el = __harness.el('map-' + k);\n"
        "  got[k] = el ? {cls: el.className,\n"
        "                 attr: el.getAttribute('data-map-state'),\n"
        "                 cur: el.getAttribute('aria-current')} : null;\n"
        "});\n"
        "__emit({map: got, calls: __harness.calls().map(function(c){\n"
        "  return c.method + ' ' + c.path; })});",
        routes=routes, search=f"?project={pid}")

    # THE RENDER HAS TO BE THE APP'S. The bootstrap must have read the record,
    # or the map below is a static markup dump rather than something the
    # controller painted. Which DOWNSTREAM routes get fetched is stage-
    # dependent by design — `renderTrainStep` returns early before the barrier
    # is raised — so this asserts the boot and not a count.
    asked = {c[4:] for c in out["calls"] if c.startswith("GET ")}
    assert f"/project/{pid}" in asked, (
        f"the controller never read the record; it fetched {sorted(asked)}")
    # Two families are deliberately NOT answered, named rather than silently
    # allowed: `/dev/status` (the dev banner) and
    # `/evidence/histogram/<column>` (one per skew candidate, so the set is not
    # knowable before the drive). Neither is on any path the rail reads.
    stray = [c[4:] for c in out["calls"]
             if c.startswith("GET ") and c[4:] not in routes
             and not c[4:].startswith("/dev/")
             and "/evidence/histogram/" not in c[4:]]
    assert not stray, (
        f"the render asked for routes this drive did not answer, so it read "
        f"empty bodies where the app reads real ones: {stray}")

    states = {}
    for key in STEPS:
        node = out["map"][key]
        assert node is not None, (
            f"the rail exposes no readable dot for `{key}`. The analysis map "
            f"writes its state through `document.querySelector('.map-step"
            f"[data-map=...]')` onto buttons that carry no `id`, so no reader "
            f"— this harness, a test, or anything else — can observe which "
            f"step the app says the user is on. That is defect (a) in its "
            f"most complete form: the state is unobservable, so the six-way "
            f"`now` below could not have been caught by anything.")
        worn = [c for c in node["cls"].split()
                if c in ("done", "now", "waiting", "stale")]
        assert len(worn) == 1, (
            f"the `{key}` dot wears {worn or 'no'} state class; a dot carries "
            f"exactly one of the four §04 names")
        assert node["attr"] == worn[0], (
            f"the `{key}` dot's class says `{worn[0]}` and its "
            f"`data-map-state` says `{node['attr']}` — trap 7, the "
            f"machine-readable form disagreeing with the treatment beside it")
        states[key] = worn[0]
    return states, out["map"]


def _journey(fixture):
    from fastapi.testclient import TestClient

    from turbotab import api, eligibility as E, grain as G

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    stages = {}
    stages["no target"] = _read_map(client, pid)[0]

    _decide(client, pid, "set_target", column=TARGETS[fixture])
    stages["target set"] = _read_map(client, pid)[0]

    _decide(client, pid, "set_grain", answer=G.ONE_ROW_PER_PERSON)
    _decide(client, pid, "set_eligibility", answer=E.EVERYONE)
    _decide(client, pid, "seal")
    stages["sealed"] = _read_map(client, pid)[0]

    _decide(client, pid, "settle_features")
    stages["features settled"] = _read_map(client, pid)[0]

    _decide(client, pid, "settle_preprocess")
    stages["preprocess settled"] = _read_map(client, pid)[0]
    return stages


# ─────────────────────────────────────────────────────────────────────────────
# (b) · the claim that is genuinely about the file
# ─────────────────────────────────────────────────────────────────────────────

def test_the_map_defines_the_four_states_its_design_language_names():
    """§04 specifies `done · now · waiting · stale`; the CSS defined three.

    A grep, deliberately, because *does this rule exist in the stylesheet* is a
    question about the file. What it cannot tell is what the rule looks like —
    that is a pixel claim and nothing here makes it.
    """
    css = PAGE.read_text(encoding="utf-8")
    # THE POSITIVE CONTROL COMES FIRST (`GUIDED-045`). "Nothing is missing" is
    # most true of an empty file, so the component has to be there before its
    # states can be counted.
    assert ".map-step{" in css, (
        "the page carries no `.map-step` rule at all, so the four-state claim "
        "below would be a statement about a page that has no analysis map")
    built = [state for state in ("done", "now", "waiting", "stale")
             if f".map-step.{state}" in css]
    assert len(built) == 4, (
        f"`DESIGN_LANGUAGE.md` §04 gives the analysis map four states and the "
        f"page styles {len(built)} — {built}. The one it does not style renders "
        f"in the bare `.map-step` default, `--faint` ink and a hollow ring, "
        f"which is what a disabled control looks like. §09: an absence must be "
        f"legible as an absence, never as a verdict.")


def test_a_step_not_reached_and_a_step_that_is_next_do_not_render_alike():
    """The consequence of the missing state, on the dot itself.

    `waiting` and `now` must be distinguishable without reading the label, so
    the rule for one may not be the rule for the other.
    """
    css = PAGE.read_text(encoding="utf-8")
    rules = {}
    for state in ("done", "now", "waiting", "stale"):
        cut = css.find(f".map-step.{state} .md{{")
        assert cut != -1, (
            f"the `{state}` dot has no rule of its own, so it draws whatever "
            f"the base `.map-step .md` draws — which is exactly what a step "
            f"nobody has reached draws")
        rules[state] = css[cut:css.index("}", cut)]
    assert len(set(rules.values())) == 4, (
        f"two of the four map states draw the same dot: {rules}")


# ─────────────────────────────────────────────────────────────────────────────
# (a) · the drive
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", sorted(TARGETS), ids=[
    "classification · two-level string target",
    "regression · continuous target"])
def test_exactly_one_step_is_now_at_every_point_of_the_journey(fixture):
    """Six dots wore `now` at once, and `--accent` gets one per viewport."""
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    for stage, states in _journey(fixture).items():
        now = [k for k, v in states.items() if v == "now"]
        assert len(now) == 1, (
            f"at `{stage}` the rail marks {len(now)} steps as the current "
            f"position ({now or 'none'}); the whole map reads: {states}")


@pytest.mark.parametrize("fixture", sorted(TARGETS), ids=[
    "classification · two-level string target",
    "regression · continuous target"])
def test_the_map_marks_train_as_now_once_preprocess_is_settled(fixture):
    """The row's headline, asserted as a TRANSITION so no fixture can supply it.

    Train reads `waiting` while Preprocess is open and `now` the moment
    Preprocess records itself. The markup seeds `waiting`, so only the first
    half could be handed over by a fixture; nothing but the rail's own fold
    over the record can produce the second.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    stages = _journey(fixture)
    assert stages["sealed"]["train"] == "waiting", (
        f"with Preprocess still open the rail says Train is "
        f"`{stages['sealed']['train']}`: {stages['sealed']}")
    assert stages["preprocess settled"]["train"] == "now", (
        f"Preprocess is settled and the rail still does not put the user at "
        f"Train — it says `{stages['preprocess settled']['train']}`. The whole "
        f"map reads: {stages['preprocess settled']}")


@pytest.mark.parametrize("fixture", sorted(TARGETS), ids=[
    "classification · two-level string target",
    "regression · continuous target"])
def test_a_step_the_journey_has_not_reached_says_waiting(fixture):
    """Not-yet-reached is a state the rail enters, not a state it omits."""
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    stages = _journey(fixture)
    fresh = stages["no target"]
    assert fresh["target"] == "now", (
        f"a fresh upload puts the user at Target; the rail says "
        f"`{fresh['target']}`: {fresh}")
    ahead = [k for k in ("eda", "features", "preprocess", "train", "explain",
                         "report") if fresh[k] != "waiting"]
    assert not ahead, (
        f"before a target is chosen these steps are unreachable and the rail "
        f"does not say `waiting` for them: "
        f"{ {k: fresh[k] for k in ahead} }")
    # And the state moves, so `waiting` is a state the rail LEAVES rather than
    # a class the markup happens to hold.
    assert stages["target set"]["eda"] == "now", (
        f"with a target recorded the user is at Explore; the rail says "
        f"`{stages['target set']['eda']}`: {stages['target set']}")


def test_the_current_step_says_so_in_the_accessibility_tree_too():
    """Trap 7 — `now` is a treatment and it is also a landmark.

    Driven rather than grepped: `aria-current` has to be on the dot the fold
    chose, which is not knowable from the markup.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / "clinic_visits.csv").open("rb") as handle:
        pid = client.post("/project", files={
            "file": ("clinic_visits.csv", handle, "text/csv")}).json()["id"]
    _decide(client, pid, "set_target", column="outcome")
    states, nodes = _read_map(client, pid)

    current = [k for k, v in states.items() if v == "now"]
    marked = [k for k in STEPS if nodes[k]["cur"] == "step"]
    assert marked == current, (
        f"the rail paints {current} as the current step and the "
        f"accessibility tree says {marked}")
