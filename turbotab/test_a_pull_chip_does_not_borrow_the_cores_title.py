"""`GUIDED-168` · a Guided chip is titled by what the Guided endpoint returns.

## What was measured

Driven on `clinic_visits.csv` at the Explore step: **seven pull chips, five of
them built, and three of the five carried a title the server's own capability
table disagreed with.**

| chip | title on the chip | `/capabilities` label |
|---|---|---|
| `look::r1_plausibility` | Physiologic Plausibility Check | Physiologic plausibility |
| `look::r2_missingness` | **Missingness Pattern Analysis** | **Missingness by feature** |
| `look::r8_collinearity` | Collinearity Heatmap | Correlation matrix |

The middle one is the finding and the other two are the same mechanism with
less at stake. `ml/router.py:399` titles a palette entry `get("title") or rid`
straight off `ml.eda_recommender`'s card, and the core's title describes the
CORE's analysis: `r2_missingness` is *Missingness Pattern Analysis*, and its
stated deliverable is three items, the third being *"Patterns suggesting MCAR
… vs MAR … vs MNAR …"*. The Guided endpoint behind the chip returns per-column
blank rates and a question — `Could a blank in 'notes' mean something?` — so it
delivers the first item, ASKS the second, and computes nothing for the third.

## Unrouted, absent, and delivered are three different things

`MISC-014` is the row that exists because a design document said a capability
was *absent from the app entirely* when it shipped in Classic. The same
distinction has to be drawn here item by item, because the borrowed title spans
all three states at once:

* the **rate per column** is delivered by this door;
* the **association-with-target test** is **built in the core and unrouted
  here** — `ml/eda_actions.py:217` runs a two-sample location test or a
  categorical association test per high-missing column, reached from
  `pages/02_EDA.py:1771`. Guided asks the user instead of computing it;
* the **MCAR/MAR/MNAR reading** is **absent from both doors**. Nothing under
  `ml/` computes one.

So the palette entry is not deleted and the title is not softened into
vagueness. **The shelf is never shortened**: the chip keeps its place, takes
the label that is true of it, and names what the borrowed title promised and
where each promise actually lives.

## Fixture shapes — `GUIDED-097`

`SHAPES` below is two targets over the one shipped fixture that has columns
above the recommender's threshold. `SHAPES_NOT_COVERED` names the rest —
**including multiclass, which this file used to cover and no longer can.**
`GUIDED-189` made the chip read `ml/missingness_plan.HIGH_MISSING_SHARE`
(0.20) instead of holding its own 0.05, and `multiclass_stage.csv`'s worst
column is 10.0% — it sat between the two thresholds, which is what that row
was about. The coverage is genuinely narrower and it is written down rather
than papered over with a fixture built to clear a number.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api                          # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: The chip this row is about.
KEY = "look::r2_missingness"

#: `GUIDED-097`. Only a fixture with a column over the recommender's missing
#: threshold raises `r2_missingness` at all. That threshold is
#: `ml/missingness_plan.HIGH_MISSING_SHARE` = 0.20 since `GUIDED-189`, and
#: exactly one shipped CSV with a usable target clears it.
SHAPES = {
    "continuous": ("clinic_visits.csv", "hba1c"),
    "binary_string": ("clinic_visits.csv", "outcome"),
}

SHAPES_NOT_COVERED = {
    "multiclass": (
        "COVERED UNTIL L51 AND NOT ANY MORE, which is why it is spelled out. "
        "`multiclass_stage.csv`'s worst column is 10.0% missing — above the "
        "chip's old private 0.05 and below the 0.20 that fills the panel it "
        "opens onto. `GUIDED-189` deleted the duplicate threshold, so the "
        "chip is correctly not raised here now and this file lost a shape. "
        "No shipped CSV has both a genuine multiclass target and a column "
        "over 0.20: `clinic_visits.csv` gets close with `notes`, which is 5 "
        "levels and 25% missing, but it is a free-text remark column and "
        "using it as a target to keep a number in a table is the fixture "
        "manufacturing its own result (trap #3). The honest form is this "
        "sentence."),
    "binary_numeric": (
        "`leaky_sepsis.csv` is the 0/1 fixture and it has no missing values at "
        "all, so it raises no `r2_missingness` recommendation and there is no "
        "chip to check. The gap is the fixture's, not this claim's."),
    "wide": (
        "`metabolomics_untargeted.csv` has 213 columns over the threshold and "
        "400 columns overall. It would exercise the same one title through the "
        "same one composer; what it would add is runtime."),
}


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _core_card():
    """The core recommendation this chip is built from, as a live object.

    Read from `ml.eda_recommender` by running it, not by quoting it: a test
    that restated the core's `what_you_learn` would keep passing after the core
    changed, which is the drift this whole row is about.
    """
    import pandas as pd
    from ml.eda_recommender import compute_dataset_signals, recommend_eda

    df = pd.read_csv(DATA / "clinic_visits.csv")
    signals = compute_dataset_signals(df, "hba1c", "regression",
                                      "cross_sectional", None)
    recs = {r.id: r for r in recommend_eda(signals)}
    assert "r2_missingness" in recs, (                                # control
        "the core no longer raises r2_missingness on this fixture, so every "
        "claim below is about a card that is not there")
    return recs["r2_missingness"]


def _explore(client, shape):
    fixture, target = SHAPES[shape]
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_target", "payload": {"column": target}})
    assert r.status_code == 200, r.text[:250]
    body = client.get(f"/project/{pid}/interview?step=explore").json()
    chips = {q["key"]: q for q in body["questions"] if q["mode"] == "pull"}
    return pid, chips


# ── 1 · the borrowed title is gone, on every shape ───────────────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_missingness_chip_is_not_titled_with_the_cores_pattern_analysis(
        client, shape):
    """**The finding.** The core's title is read off the core, so this cannot
    pass by agreeing with a string that has since changed."""
    core = _core_card()
    assert core.title == "Missingness Pattern Analysis", (            # control
        f"the core's title moved to {core.title!r}; this file's premise needs "
        f"re-reading before its assertions mean anything")

    pid, chips = _explore(client, shape)
    assert KEY in chips, (
        f"{shape} raised no missingness chip, so nothing here was checked")
    chip = chips[KEY]

    assert chip["title"] != core.title, (
        f"the Guided chip is titled {chip['title']!r}, which is the core "
        f"recommender's title for an analysis this endpoint does not run")
    caps = client.get(f"/project/{pid}/capabilities").json()["pulls"]
    assert chip["title"] == caps[KEY]["title"] == "Missingness by feature", (
        chip["title"], caps[KEY]["title"])
    # The borrowed title is kept on the record rather than dropped, so the
    # substitution is auditable rather than invisible.
    assert chip["core_title"] == core.title


# ── 2 · the required claim: the Guided surface does not promise the core's
#        deliverable, and accounts for every item of it ──────────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_chip_accounts_for_every_deliverable_the_borrowed_title_promised(
        client, shape):
    """**Each of the core's stated deliverables is placed, and none is
    claimed.**

    The three items come out of the live `EDARecommendation`, so a fourth added
    upstream turns this red rather than passing silently — which is exactly how
    the third one, the MCAR/MAR/MNAR reading, arrived on a Guided chip in the
    first place.
    """
    core = _core_card()
    learn = list(core.what_you_learn)
    assert len(learn) == 3, learn                                     # control

    _, chips = _explore(client, shape)
    instead = chips[KEY]["instead_of"]
    assert instead["core_title"] == core.title

    placed = (list(instead["delivered_here"])
              + list(instead["asked_here_not_computed"])
              + list(instead["absent_from_both_doors"]))
    assert sorted(placed) == sorted(learn), (
        "the chip's account of the borrowed title does not match the core's "
        f"stated deliverables.\n  core: {learn}\n  placed: {placed}")

    # THE MCAR ITEM IS THE ONE THAT MATTERS AND IT IS FILED AS ABSENT — not as
    # delivered, and not as merely unrouted. Nothing in `ml/` computes it, so
    # "the core has it and Guided is not wired to it" would be the other
    # direction of `MISC-014`'s error.
    mcar = [i for i in learn if "MCAR" in i]
    assert len(mcar) == 1, learn
    assert mcar[0] in instead["absent_from_both_doors"], (
        "the MCAR/MAR/MNAR reading is filed somewhere other than "
        "absent-from-both-doors")
    assert mcar[0] not in instead["delivered_here"]
    assert mcar[0] not in instead["asked_here_not_computed"]

    # AND THE UNROUTED HALF IS NAMED WITH ITS MODULE, because "unrouted" is a
    # claim about a file that exists and has to resolve.
    unrouted = " ".join(instead["built_in_core_unrouted_here"])
    assert "ml/eda_actions.py" in unrouted, unrouted
    root = Path(__file__).resolve().parents[1]
    assert (root / "ml" / "eda_actions.py").exists()
    source = (root / "ml" / "eda_actions.py").read_text(encoding="utf-8")
    assert "def missingness_scan" in source, (
        "the chip says the association test is built in the core and names "
        "`ml/eda_actions.py`; the function it names is not there")


# ── 3 · the difference is on the chip, not only in the payload ───────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_difference_rides_in_the_field_the_page_draws_as_the_tooltip(
        client, shape):
    """Trap #6: the server composes a sentence and the interface never renders
    it. The page builds a chip's `data-tip` from `why`, falling back to
    `title`, so the correction goes there — and the evidence the core supplied
    is kept in front of it rather than replaced.
    """
    _, chips = _explore(client, shape)
    chip = chips[KEY]
    sentence = chip["instead_of"]["sentence"]
    assert sentence in chip["why"], (
        f"the sentence is in the payload and not in the tooltip: "
        f"{chip['why']!r}")
    assert "columns with >5% missing" in chip["why"], (
        "the core's own evidence was dropped when the sentence was added")

    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    assert 'b.setAttribute("data-tip", q.why || q.title);' in page, (
        "the page no longer renders a chip's `why`, so this sentence reaches "
        "nobody and the assertion above is checking a field nothing draws")


# ── 4 · what the endpoint the chip opens actually returns ────────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_endpoint_behind_the_chip_asks_the_association_question_it_does_not_answer(
        client, shape):
    """The other half of the title's promise, driven rather than described.

    *Whether missingness is associated with target* is filed as **asked, not
    computed**, and this presses the endpoint the chip opens to check that the
    filing is true: every card carries a `mechanism_question` with options for
    the user, and no card carries a test statistic or a p-value.
    """
    pid, chips = _explore(client, shape)
    endpoint = chips[KEY]["endpoint"]
    assert endpoint == "missingness", endpoint
    cards = client.get(f"/project/{pid}/evidence/{endpoint}").json()["cards"]
    if not cards:
        # NOT AN ASSERTION HERE, because it is a different defect and it has
        # its own test below. `multiclass_stage.csv` raises the chip and opens
        # onto nothing — see `test_a_built_chip_can_still_open_onto_no_cards`.
        pytest.skip(
            f"{shape} produces 0 cards behind a built chip; that is the "
            f"threshold mismatch filed separately, not this claim")

    for card in cards:
        q = card.get("mechanism_question") or {}
        assert q.get("options"), (
            f"{card['column']!r} has no question to answer, so the "
            f"association item is neither asked nor computed")
        blob = str(card).lower()
        for word in ("p-value", "p_value", "mann-whitney", "chi-square",
                     "statistic"):
            assert word not in blob, (
                f"{card['column']!r} carries {word!r}, so the endpoint does "
                f"compute the association and the chip's account of itself is "
                f"the thing that is now wrong")


# ── 4b · found while driving row 4, filed rather than fixed ──────────────────

@pytest.mark.xfail(strict=True, reason=(
    "FOUND WHILE DRIVING GUIDED-168 AND NOT FIXED — it is a threshold, and a "
    "threshold does not move in the same loop as the change that pressured "
    "it (AGENT_ONBOARD.md section 08, check 2). `ml/eda_recommender.py:120` "
    "raises the missingness recommendation at rate > 0.05, and "
    "`ml/missingness_plan.py:49 HIGH_MISSING_SHARE = 0.20` gates the cards "
    "the chip opens onto. Any table whose worst column sits between the two "
    "gets a live-styled chip that opens onto an empty panel — GUIDED-006's "
    "own sentence, `a control that silently does nothing`, arriving through a "
    "threshold mismatch instead of a missing endpoint. STRICT so closing it "
    "without removing this marker is itself reported."))
def test_a_built_chip_can_still_open_onto_no_cards(client):
    """Measured, not asserted from the source: `multiclass_stage.csv` has
    `crp` at 10.0% and `bmi` at 7.1% missing. The chip is built, its own
    tooltip says *2 columns with >5% missing values*, and the endpoint returns
    zero cards.

    The missingness SURVEY reports both columns, so Preprocess still asks about
    them. It is the Explore panel alone that is empty, which is why nothing
    downstream is wrong and the cost is again a chip that asserts something
    false about itself.
    """
    pid, chips = _explore(client, "multiclass")
    chip = chips[KEY]
    assert chip["built"] is True                                      # control
    assert "2 columns with >5% missing" in chip["why"]                # control
    cards = client.get(f"/project/{pid}/evidence/missingness").json()["cards"]
    assert cards, (
        "a built chip opened onto zero cards: the recommender's 5% gate and "
        "the card composer's 20% gate disagree")


# ── 5 · the sweep, one surface over ──────────────────────────────────────────

def test_no_built_chip_anywhere_keeps_a_title_the_capability_table_contradicts(
        client):
    """`AGENT_ONBOARD.md` §08's fifth check: *ask what the same lens would find
    one surface over.*

    It found two more — `Physiologic Plausibility Check` and `Collinearity
    Heatmap`, against `Physiologic plausibility` and `Correlation matrix`. The
    second is the interesting one: the page's own evidence panel for that
    endpoint is headed `Correlation matrix`, so the chip and the panel it
    opened carried two names for one thing. This sweeps every built chip rather
    than the one the row was filed against.
    """
    pid, chips = _explore(client, "continuous")
    caps = client.get(f"/project/{pid}/capabilities").json()["pulls"]
    built = {k: q for k, q in chips.items() if q.get("built")}
    assert len(built) >= 3, sorted(built)                             # control

    wrong = {k: (q["title"], caps[k]["title"]) for k, q in built.items()
             if k in caps and q["title"] != caps[k]["title"]}
    assert not wrong, (
        "these chips are titled by the core recommendation rather than by the "
        f"capability table that describes what they open: {wrong}")

    # AND THE SUBSTITUTION IS REAL RATHER THAN VACUOUS: at least one chip's
    # recorded `core_title` differs from the title it now shows. Without this
    # the claim above would also hold on a build where nothing was borrowed.
    swapped = {k: q["core_title"] for k, q in built.items()
               if q.get("core_title")}
    assert len(swapped) >= 3, (
        f"only {len(swapped)} chips had a borrowed title replaced; the "
        f"measurement behind this file found three")
