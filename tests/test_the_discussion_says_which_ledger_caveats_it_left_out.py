"""`AUDIT-032`'s sibling, found one surface over — the Discussion's own truncation.

## How this was found

`AUDIT-032` is that running the leakage diagnostic marks the leakage blocker
*resolved*, the report calls it *addressed*, and the manuscript drops the caveat
the app itself authored — while the column is still a model feature. That
resolution lives in `pages/02_EDA.py` and `utils/insight_ledger.py`, neither of
which this chunk owns.

`AGENT_ONBOARD.md` §08 item 5 asks what the same lens finds one surface over.
One surface over is `ml/narrative_engine.py._gen_discussion`, the module that
renders `discussion_points_for_manuscript()` into the exported Discussion, and
it had the same failure by a different mechanism:

    strength_strs = strengths[:3]
    limitation_strs = limitations[:5]

printed under headings that read **"Limitations (auto-generated from analysis
ledger)"** — a heading that claims the list came from the ledger, over a list
that is silently the first five of it. On a busy ledger the manuscript asserted
a complete set of caveats and printed five, and nothing on the page said so.

**And the exploratory-mode caveat was appended AFTER the ledger's list**, so
`[:5]` removed it first. That caveat — *the held-out test set was not
quarantined … performance should not be presented as validated held-out
performance* — is the strongest disclosure this app makes, it is about the whole
study rather than about one column, and it vanished from exactly the manuscripts
with the most to disclose. That is `AUDIT-032`'s shape precisely: a caveat the
app wrote, suppressed in the artifact that leaves the building.

## What changed

The cap stays — a Discussion is not a log. The exploratory caveat is put FIRST
so a cap can never reach it, and whatever the cap does cut is **counted and
disclosed** as an `[AUTHOR REQUIRED]` gap rather than dropped.

## `GUIDED-097` — two target shapes

The Discussion is composed per study, so every claim below runs against a
**continuous** target (`glucose`, regression) and a **binary** target
(`readmit_30d`, classification). **The shape not covered is named at the bottom
of this file.**
"""
from __future__ import annotations

import pytest

from ml.narrative_engine import NarrativeEngine
from utils.insight_ledger import Insight, InsightLedger
from utils.workflow_provenance import WorkflowProvenance

#: `GUIDED-097`. `(target, task_type, features)`.
SHAPES = {
    "continuous": ("glucose", "regression", ["age", "bmi", "insulin", "bp"]),
    "binary": ("readmit_30d", "classification", ["age", "bmi", "los", "crp"]),
}

#: The exploratory-mode caveat, by the clause a reader would look for. Matched
#: on its substance rather than on the whole paragraph, so a rewording of the
#: sentence does not silently turn this guard off.
EXPLORATORY_CLAUSE = "was not quarantined from feature engineering"

#: How many unresolved ledger limitations the fixture seeds. Chosen larger than
#: the module's cap and NOT imported from it: importing the constant would make
#: a revert raise `ImportError` and the probe would be red for the wrong reason.
N_LEDGER_LIMITATIONS = 8


def _provenance(shape):
    target, task, features = SHAPES[shape]
    prov = WorkflowProvenance()
    prov.record_upload(target, task, list(features), 900)
    prov.record_split(strategy="stratified", train_n=630, val_n=135,
                      test_n=135, random_seed=42)
    prov.record_training(
        models_trained=["ridge" if task == "regression" else "logreg"],
        primary_model="ridge" if task == "regression" else "logreg",
        selection_criteria="held-out score",
        use_cv=False, cv_folds=0, hyperparameters={},
        metrics_by_model={},
    )
    return prov


def _ledger(n_limitations):
    """A ledger holding `n_limitations` UNRESOLVED, narrative-worthy insights.

    §07 trap 3: the fixture must produce what production produces. These are
    real `Insight` objects with `manuscript_text` set, which is the field
    `discussion_points_for_manuscript` prefers, and the count is asserted off
    the ledger's own method below rather than assumed from this loop.
    """
    ledger = InsightLedger()
    for i in range(n_limitations):
        ledger.add(Insight(
            id=f"eda_skew_col_{i:02d}",
            source_page="02_EDA",
            category="data_quality",
            severity="warning",
            finding=f"col_{i:02d} is heavily right-skewed",
            implication="Linear models may be influenced by the tail",
            manuscript_text=(f"the distribution of col_{i:02d} was strongly "
                             f"right-skewed and was modeled untransformed"),
        ))
    return ledger


def _discussion(shape, n_limitations, exploratory=True):
    engine = NarrativeEngine(
        _provenance(shape), _ledger(n_limitations),
        manuscript_context={"exploratory_mode": True} if exploratory else None)
    return engine


# ═══════════ 1 · the caveat a cap must never reach ═══════════

@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_exploratory_caveat_survives_a_ledger_bigger_than_the_cap(shape):
    """The load-bearing claim. Appended last, it was the first thing cut."""
    engine = _discussion(shape, N_LEDGER_LIMITATIONS)

    # POSITIVE CONTROL (`GUIDED-045`) — the ledger really does hand the
    # Discussion more limitations than it prints, and the caveat really is
    # produced on a SMALL ledger. Without both, an absence below would be a
    # statement about the fixture.
    points = engine.ledger.discussion_points_for_manuscript()
    assert len(points["limitations"]) == N_LEDGER_LIMITATIONS, (
        f"{shape}: the ledger yielded {len(points['limitations'])} limitations, "
        f"not {N_LEDGER_LIMITATIONS}; this fixture is not exercising the cap")
    small = _discussion(shape, 1)._gen_discussion()
    assert EXPLORATORY_CLAUSE in small, (
        f"{shape}: the exploratory caveat is not produced even on a one-entry "
        f"ledger, so its absence on a large one would not be the cap's doing")

    discussion = engine._gen_discussion()
    assert EXPLORATORY_CLAUSE in discussion, (
        f"{shape}: the exploratory-mode caveat is missing from the exported "
        f"Discussion on a ledger of {N_LEDGER_LIMITATIONS} limitations. It is "
        f"the app's strongest disclosure — the held-out set was not quarantined "
        f"— and it was appended after the ledger's own list, so the cap cut it "
        f"first. AUDIT-032's sibling.")


# ═══════════ 2 · and what the cap does cut is counted ═══════════

@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_discussion_states_how_many_limitations_it_did_not_print(shape):
    """*Auto-generated from analysis ledger* over five of nine is a heading
    that asserts a complete list. The omission is stated, with its count."""
    engine = _discussion(shape, N_LEDGER_LIMITATIONS)
    discussion = engine._gen_discussion()

    # POSITIVE CONTROL — the Limitations block is rendered at all.
    assert "**Limitations (auto-generated from analysis ledger):**" in discussion, (
        f"{shape}: no Limitations block was rendered, so a claim about what it "
        f"discloses is a claim about nothing")

    assert "not\nprinted here" in discussion or "not printed here" in discussion, (
        f"{shape}: the Discussion printed a subset of the ledger's "
        f"{N_LEDGER_LIMITATIONS + 1} limitations under a heading that says the "
        f"list came from the ledger, and said nothing about the remainder. "
        f"AUDIT-032's sibling. The Discussion read: "
        f"{discussion[discussion.find('**Limitations'):][:600]!r}")
    assert "[AUTHOR REQUIRED" in discussion, (
        f"{shape}: the omission is disclosed without handing the decision back "
        f"to the author; the app does not get to decide which caveats matter")


def test_a_ledger_under_the_cap_says_nothing_about_omissions():
    """The disclosure is not wallpaper. With everything printed there is
    nothing to disclose, and a note that fired anyway would be a second false
    claim in the other direction."""
    engine = _discussion("continuous", 2)
    discussion = engine._gen_discussion()

    # POSITIVE CONTROL — the block rendered, so the absence below is about the
    # note rather than about a missing section.
    assert "**Limitations (auto-generated from analysis ledger):**" in discussion
    assert "not printed here" not in discussion.replace("\n", " "), (
        "a 2-entry ledger fits under the cap, so the Discussion claimed an "
        "omission that did not happen")


#: NOT COVERED, said out loud — `GUIDED-097`'s second clause.
#:
#: A MULTICLASS TARGET. `_gen_discussion`'s Strengths-and-Limitations block
#: reads the ledger and `exploratory_mode` only; it branches on neither the task
#: type nor the number of classes, so a multiclass study exercises no path the
#: two shapes above do not. Named rather than assumed.
#:
#: THE GUIDED DOOR. `turbotab/manuscript.py` composes its Limitations from
#: `draft.py`'s decision fold and never calls this module, so it has no cap and
#: is not covered here. `AUDIT-032` itself — the leakage blocker marked resolved
#: by a diagnostic that removes nothing — is in `pages/02_EDA.py` and
#: `utils/insight_ledger.py` and remains open; this file closes only the
#: downstream truncation that would have hidden the caveat even if the ledger
#: had kept it.
