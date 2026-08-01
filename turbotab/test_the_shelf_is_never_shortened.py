"""`PRODUCT_VISION.md`, *"the shelf is never shortened"* — as a check.

> The shape of the data changes how models are **ranked**. It never changes
> which models are **available.**

The failure this guards against is not hypothetical and not subtle: Classic's
Train page already had `ml.model_coach` bucketing every model into good / ok /
poor with an evidence-bearing clause, and rendered the result as badges beside a
taxonomy-ordered list. The judgment existed and whispered. So the claims here
are about ORDER and PROMINENCE, and about the one thing that must never happen —
a model disappearing because the engine dislikes it for this data.

**Structure, not prose substrings** (the L17 rule). The claim "the shelf is not
filtered" is asserted against the registry's own key set rather than against a
count or a sentence, because a count drifts with the registry and a sentence is
a wildcard wearing an assertion's clothes. Where prose IS the deliverable — the
concern that travels with a poor-fit choice — the assertion is that the ENGINE'S
OWN clause arrives whole, compared against `model_coach`'s output object.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml import model_coach                                            # noqa: E402
from ml.model_registry import get_registry                            # noqa: E402
from turbotab import eligibility as E, engine, grain as G             # noqa: E402
from turbotab import models as M                                      # noqa: E402
from turbotab.project import AnalysisProject, ProjectError            # noqa: E402


def wide_and_short(n: int = 60, p: int = 40) -> pd.DataFrame:
    """p/n = 0.67 — the regime where the coach has something to say.

    Deliberately a shape several models are genuinely wrong for, because a
    fixture nothing is wrong for cannot tell "the shelf is never shortened"
    from "nothing wanted to shorten it".
    """
    rng = np.random.default_rng(4)
    df = pd.DataFrame({f"x{i:02d}": rng.normal(0, 1, n) for i in range(p)})
    df["outcome"] = (df["x00"] + rng.normal(0, 0.5, n) > 0).astype(int)
    return df


def _sealed(df: pd.DataFrame | None = None) -> AnalysisProject:
    p = AnalysisProject.from_dataframe(wide_and_short() if df is None else df, "s")
    p.set_target("outcome", "classification", "high", [])
    p.set_grain(G.ONE_ROW_PER_PERSON)
    p.set_eligibility(E.EVERYONE)
    d = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(d["labels"], **d["disclosure"])
    return p


def _shelf(p: AnalysisProject):
    return p.model_shelf()


# ── the claim the file is named after ────────────────────────────────────────

def test_every_model_the_task_can_use_is_on_the_shelf():
    """Not "enough models". EVERY model, keyed against the registry itself.

    The comparison is set-equality with the registry filtered on capability,
    which is the only assertion that stays true as models are added. A test
    asserting `len(shelf) >= 8` would pass on a shelf that dropped the two the
    coach disliked most and gained three others.
    """
    p = _sealed()
    on_shelf = {e.key for e in _shelf(p)}
    can_classify = {k for k, spec in get_registry().items()
                    if spec.capabilities.supports_classification}

    missing = sorted(can_classify - on_shelf)
    assert not missing, (
        f"{missing} can fit a classification task and are not on the shelf. "
        "The shape of the data changes the ORDER; it never changes what is "
        "available. A model the engine dislikes appears last, with its concern, "
        "not nowhere.")


def test_a_model_the_coach_calls_poor_is_still_there_and_still_selectable():
    """The specific case, because the general check would pass on an empty coach.

    If `model_viability` returned nothing at all, every model would land in
    `worth_trying` and the set-equality above would still hold. So this asserts
    that the fixture actually produced a `not_recommended` verdict AND that the
    model carrying it survived to a completed selection.
    """
    p = _sealed()
    entries = _shelf(p)
    poor = [e for e in entries if e.bucket == M.NOT_RECOMMENDED]
    assert poor, (
        "the wide-and-short fixture produced no poor verdicts, so this file "
        "cannot tell a shelf that keeps them from a shelf with none to keep. "
        "Widen the fixture rather than weakening the claim.")

    # The refusal is caught rather than allowed to propagate: a raise and a
    # wrong answer are the same defect here — the app declining a legitimate
    # choice — and the failure should say so either way rather than showing a
    # traceback the reader has to interpret.
    try:
        p.select_models([poor[0].key])
    except ProjectError as exc:
        pytest.fail(
            f"a model the coach ranked last was refused: {exc} — model choice "
            "is the ladder's BOTTOM rung, rank and state the concern, because "
            "the test for the top rung is whether a competent researcher could "
            "have a reason, and for these models they routinely do")
    assert p.selected_models == [poor[0].key], (
        "a model the coach ranked last was refused, silently — it was dropped "
        "from the selection rather than rejected, which is the shelf being "
        "shortened one step later than usual")


def test_the_order_carries_the_judgment_rather_than_a_badge():
    """Recommended before worth-trying before not-recommended, without exception.

    This is the whole mechanism. If the ordering were by group or by name with
    the bucket rendered beside it, that is Classic's Train page again: the
    ranking exists, and the eye never reaches it.
    """
    entries = _shelf(_sealed())
    ranks = [M._BUCKET_ORDER[e.bucket] for e in entries]
    assert ranks == sorted(ranks), (
        "the shelf is not ordered by bucket. Judgment rendered as a badge "
        "beside an alphabetical list is judgment nobody reads — that is the "
        "exact failure this replaces.")


def test_the_three_groups_are_always_returned_including_the_empty_ones():
    """"No model is recommended for this data" is a state, and it has to be sayable.

    A renderer handed two groups cannot draw the difference between "nothing
    was recommended" and "the recommended section is above the fold". Asserted
    on the returned structure, not on a rendered string.
    """
    groups = M.grouped(_shelf(_sealed()))
    assert [g["bucket"] for g in groups] == [
        M.RECOMMENDED, M.WORTH_TRYING, M.NOT_RECOMMENDED]

    # And with a coach that says nothing at all, so two of the three are empty.
    empty = M.grouped([e for e in _shelf(_sealed())
                       if e.bucket == M.WORTH_TRYING])
    assert len(empty) == 3, "an empty group was dropped rather than returned"
    assert empty[0]["models"] == [] and empty[2]["models"] == []


def test_the_concern_arrives_whole_rather_than_summarized():
    """The engine's own clause, character for character, against its source.

    A concern shortened to fit a badge is the failure this module exists to
    correct, so the assertion compares `ShelfEntry.concern` to what
    `model_coach.model_viability` returned — not to a substring, and not to a
    length threshold, either of which a truncating renderer would survive.

    **Profiled on the TRAINING rows**, changed at L34-D. This built its
    expectation from the whole table, which is what `model_shelf` used to read;
    the clause cites `p/n`, so the two disagreed the moment the shelf started
    ranking on the rows the models are actually fitted on (0.67 against 0.77).
    The test was pinning the leak, not the paraphrase it is about.
    """
    p = _sealed()
    sealed = set((p.lockbox or {}).get("labels") or [])
    train = p.df.loc[[i not in sealed for i in p.df.index]]
    assert len(train) < len(p.df), (
        "nothing is held out, so this is not testing the training profile")
    prof = engine.profile(train, p.target, p.task_type)
    verdicts = model_coach.model_viability(prof, None)
    assert verdicts, "the coach produced no verdicts; the check would be vacuous"

    entries = {e.key: e for e in _shelf(p)}
    for key, (_verdict, clause) in verdicts.items():
        if key in entries and clause:
            assert entries[key].concern == clause, (
                f"{key}'s concern was altered on the way to the shelf. The "
                "clause cites this dataset's numbers; a paraphrase of it is "
                "the app deciding how much of its own reasoning the user gets.")


def test_a_model_that_cannot_fit_the_task_at_all_is_the_only_exclusion():
    """The one filter, and it is structural rather than a judgment.

    A classifier cannot fit a continuous outcome — offering it would be
    offering something that raises, not something that fits poorly. That is the
    ladder's TOP rung (no legitimate use exists), and it is the only rung this
    module uses.
    """
    reg = get_registry()
    classification = {e.key for e in M.shelf(
        engine.profile(_sealed().df, "outcome", "classification"),
        "classification")}
    regression_only = {k for k, s in reg.items()
                       if s.capabilities.supports_regression
                       and not s.capabilities.supports_classification}
    assert regression_only, "no regression-only model exists; check is vacuous"
    assert not (regression_only & classification), (
        f"{sorted(regression_only & classification)} were offered for a "
        "classification task and cannot perform one.")


def test_the_disclosure_says_the_list_is_not_filtered():
    """Prose IS the deliverable here, so the assertion is on the claim.

    A user who believes a model is unavailable will not go looking for it, so
    the page has to say the list is complete. Asserted on the two distinctive
    claims — availability and the reason for the order — rather than on a word.
    """
    said = M.SHELF_DISCLOSURE
    assert "Every model is available" in said
    assert "about your data" in said, (
        "the disclosure does not say what the ORDER is about, so a reader can "
        "still take a low position as a verdict on the model itself")


# ── selection is recorded, and the concern travels with it ───────────────────

def test_selecting_a_low_ranked_model_records_the_concern_for_the_methods_section():
    """Not a warning dialog. A sentence in the record.

    The choice is legitimate and already made; what must not happen is the
    concern staying on a screen the reader of the results never saw.
    """
    p = _sealed()
    entries = _shelf(p)
    poor = [e for e in entries if e.bucket == M.NOT_RECOMMENDED]
    good = [e for e in entries if e.bucket == M.RECOMMENDED]
    assert poor and good, "fixture must produce both buckets"

    d = p.select_models([good[0].key, poor[0].key])
    note = d.payload["concern_note"]
    assert note, "a stated concern was selected and nothing recorded it"
    assert poor[0].concern in note, (
        "the record carries a summary rather than the engine's clause; the "
        "reader of the manuscript gets the paraphrase and never the numbers")

    clean = p.select_models([good[0].key])
    assert clean.payload["concern_note"] is None, (
        "a selection with no stated concern produced a concern note, which "
        "makes the note unreadable as a signal")


def test_an_unknown_model_is_refused_and_an_empty_selection_is_refused():
    """The only two refusals, and neither is about fit."""
    p = _sealed()
    with pytest.raises(ProjectError):
        p.select_models(["definitely_not_a_model"])
    with pytest.raises(ProjectError):
        p.select_models([])


def test_models_cannot_be_chosen_before_the_seal():
    """The shelf is ordered by the shape of the data the models will see.

    Chosen before the seal, the ordering reads a table that still includes rows
    the trim will remove and columns Features has not built — a ranking of the
    wrong dataset, presented with the confidence of a ranking of the right one.
    """
    p = AnalysisProject.from_dataframe(wide_and_short(), "s")
    p.set_target("outcome", "classification", "high", [])
    with pytest.raises(ProjectError):
        p.select_models(["rf"])


def test_deselecting_a_model_drops_the_recipe_that_described_it():
    """A recipe for a model nobody trains is a decision about nothing.

    Left behind, it travels into the archive and then into the record as a
    configured preprocessing step for a model that was never fitted.
    """
    p = _sealed()
    p.select_models(["rf", "logreg"])
    p.set_model_recipe("rf", "power", "log1p")
    assert "rf" in p.model_recipes

    p.select_models(["logreg"])
    assert "rf" not in p.model_recipes, (
        "the recipe for a deselected model survived the change; the record "
        "would then describe preprocessing applied to a model nobody trained")
