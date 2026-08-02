"""Every field on the project survives the archive. Structurally, not by memory.

`turbotab/archive.py` writes an explicit **whitelist** — `build_members` names
each field it saves, and the lockbox member names each key. That is the right
design: a serializer that walks whatever it finds writes whatever it finds, and
"never persist participant data" then depends on nobody adding the wrong field.

But a whitelist has one failure mode and this one has hit it **twice**:

* **L13** — the seal gained `seal_basis` and `basis_source` (constitution §03).
  `lockbox.json` did not, so a restored seal came back unable to say what it
  rested on: exactly the `group_col: None` ambiguity §03 exists to remove.
* **L14** — the project gained `engineered`, `deferred_transforms`,
  `selection_spec` and `features_settled`. `config.json` did not, so a restored
  project had the engineered COLUMNS (they are in the parquet) and no record of
  how they got there, and the deferred specs — which have no other home, because
  nothing has executed them — were gone outright.

Both were caught by a test written for that feature, which means both were
caught by remembering. The third one will be caught by this file instead.

**Two levels, because the whitelist has two.**

1. Every dataclass field on `AnalysisProject` is classified — `PERSISTED` or
   `REGENERATED` with a written reason — and a field in neither fails. Then every
   `PERSISTED` field is populated with a **non-default** value, round-tripped,
   and compared. A field dropped from `build_members` comes back as its default
   and the comparison fails.
2. Every KEY of the nested dicts that have their own whitelists — the lockbox
   and the cohort — round-trips too. Level 1 cannot see inside those: the
   lockbox is one field, and it survives level 1 while losing half its contents,
   which is precisely what happened at L13.

This is deliberately about EXISTENCE, like the clause check. It does not judge
whether a field *should* be persisted; it refuses to let that question go
unanswered.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from turbotab import archive, eligibility as E, engine, grain as G   # noqa: E402
from turbotab.project import AnalysisProject                         # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# The classification. Adding a field to AnalysisProject and not to one of these
# is a failure — that is the point of the file.
# ─────────────────────────────────────────────────────────────────────────────

# Fields the archive must carry. The value is how the round-tripped project is
# read back, because a few land under a different name or shape.
PERSISTED = {
    "id", "name", "created_at", "target", "task_type", "task_confidence",
    # `promoted_figures` is a DECISION about the manuscript, not a derivative
    # of one: nothing recomputes which figures the author placed in the
    # results, and a restored project that lost it would render a manuscript
    # missing the figures its own transcript says were promoted (`GUIDED-107`).
    "promoted_figures",
    # `orientation` is the sharpest field in this set after `aggregation`, for
    # the same reason: answering it TRANSPOSED the frame, and the parquet in the
    # archive holds the turned-around table. A restored project that lost the
    # record would hold sample-rows with nothing saying they were ever columns —
    # the receipt for an irreversible operation, missing.
    # `purpose` travels because the DOWNSTREAM DEFAULTS are read from it — a
    # restored project that lost it would silently revert to the app's old
    # assumption (prediction) and change what a dozen later questions default
    # to, with nothing saying so.
    "lens", "orientation", "purpose", "repeat_kind", "unit_of_analysis",
    "aggregation",
    "temporal_prediction",
    "task_overridden", "workflow_mode", "pipeline_specs", "grain",
    "eligibility", "obligations", "missingness", "preprocess_settled",
    "selected_models", "preparation_mode", "model_recipes",
    "engineered",
    "deferred_transforms", "selection_spec", "features_settled",
    "stale_downstream", "lockbox", "cohort", "decisions",
}

# Fields deliberately not carried, each with the reason. `_NEVER_PERSIST` is a
# drop-list, not a prohibition: derivatives are cheap to rebuild and unsafe to
# pickle, and a restored project that claims fresh findings would be lying.
REGENERATED = {
    "df": "The table travels as parquet and is restored from it; the field is a "
          "handle, and serializing a handle would serialize the frame twice.",
    "findings": "A derivative. `turbotab.cascade` names it a result key, and a "
                "restored project sets it empty and `findings_stale` true, "
                "which is the honest state to restore into.",
    "profile": "Same: a derivative of the frame, recomputed by the engine. "
               "`session_manager` says so in as many words — regenerated from "
               "raw_data.",
    "findings_stale": "Not restored but SET, unconditionally true. Restoring "
                      "the saved value would let a stale-false travel with a "
                      "project whose findings were dropped on the way in.",
    "task_reasons": "The detection's own words, recomputed with the detection. "
                    "Carried on the set_target decision, which IS persisted, so "
                    "the record still says what the engine thought.",
    "_history": "Whole dataframes, one per applied fix, kept so fixes are "
                "reversible in the session. Persisting them would write N "
                "copies of the table into a file whose whole contract is that "
                "the table appears once.",
}


def _fields():
    return [f.name for f in dataclasses.fields(AnalysisProject)]


def test_every_field_on_the_project_is_classified():
    """A new field is a decision, and this is where the decision gets made.

    Neither answer is assumed. A field nobody classified is not "probably a
    derivative" — it is a field whose persistence nobody thought about, which is
    how the seal lost its basis and the features step lost its record.
    """
    unclassified = [f for f in _fields()
                    if f not in PERSISTED and f not in REGENERATED]
    assert not unclassified, (
        f"these fields on AnalysisProject are in neither list: {unclassified}.\n"
        "Add each to PERSISTED (and to `archive.build_members`), or to "
        "REGENERATED with the reason it is safe to drop. A field in neither is "
        "a field whose persistence nobody decided.")

    stale = [f for f in list(PERSISTED) + list(REGENERATED) if f not in _fields()]
    assert not stale, (
        f"these are classified here and no longer exist on the project: {stale}. "
        "Renamed, or removed?")

    thin = [f for f, why in REGENERATED.items() if len(why) < 40]
    assert not thin, (
        f"{thin} claim to be regenerated without a real reason. Dropping a "
        "field is an argument, not a keyword.")


def _fully_populated() -> AnalysisProject:
    """A project with every PERSISTED field set to something non-default.

    Non-default is the whole mechanism. A field dropped from the whitelist comes
    back as its default, so a test that populated nothing would pass on an
    archive that saved nothing.
    """
    rng = np.random.default_rng(11)
    n = 60
    df = pd.DataFrame({
        "SUBJ": [f"S{i // 3:03d}" for i in range(n)],
        "age": rng.integers(20, 80, n).astype(float),
        "glucose": rng.normal(95, 12, n),
        "outcome": rng.integers(0, 2, n),
    })
    p = AnalysisProject.from_dataframe(df, "everything")
    # Detected regression, overridden to classification — so `task_overridden`
    # is True rather than its default, which is the only way this fixture can
    # tell "the archive carried it" from "the archive dropped a False".
    p.set_target("outcome", "regression", "low", ["low cardinality"])
    p.override_task_type("classification")

    # Features: a row-local one applied, a stateful one deferred, a selection
    # spec recorded, the step settled. All four are the L14 omission.
    p.add_feature("log", ["glucose"])
    p.defer_feature("standardize", ["age"])
    from turbotab import repeats as R, selection as S
    p.set_selection(S.declare("mutual_info", "outcome", ["age", "glucose"],
                              n_features=2))
    p.settle_features()

    # Grain and the seal, with a grouped basis so `group_col`, `n_groups`,
    # `n_test_groups` and `group_noun` are all non-null — the L13 omission.
    p.set_lens(["dietary", "clinical"])
    # Question 1.5, answered the way that does NOT rewrite the table — this
    # fixture is upright, and turning it around here would invalidate every
    # column name the rest of this drive names. The record still has to carry
    # it: "the table was confirmed as one row per sample" is a claim, and a
    # restored project that lost it could not tell a table that was checked
    # from one nobody looked at.
    p.set_orientation("rows_are_samples")
    # Question 2.5. `inference` deliberately, because it is the answer that
    # CHANGES things: a project restored as `prediction` by default would pass
    # a round trip that never exercised the field.
    p.set_purpose("inference")
    p.set_grain(G.PEOPLE_REPEAT, "SUBJ")
    # Questions 4 to 7. The unit is the RECORD rather than the person, so the
    # rows survive and the seal below still has 60 of them to draw from — and
    # `temporal_prediction` is reachable, which it is not under aggregation.
    p.set_repeat_kind(R.TIME_POINTS, overturned=True)
    p.set_unit_of_analysis(R.UNIT_RECORD)
    p.set_temporal_prediction(True)
    p.set_eligibility(E.RESTRICTED, column="age", minimum=25,
                      reason="The study is about adults over 25.")
    drawn = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(drawn["labels"], **drawn["disclosure"])

    # A post-seal robustness trim, which arms a §05 obligation. The obligation
    # has to survive the archive because it spans two steps and a save is the
    # longest form of that gap.
    p.trim_training_rows("age", minimum=25, maximum=70,
                         reason="The cohort of clinical interest is 25 to 70.")

    # Preprocess: this step is almost entirely DECLARATIONS, so the archive is
    # the only place they live — nothing has executed them.
    p.df = p.df.copy()
    p.df.loc[p.df.index[:5], "glucose"] = None
    from turbotab import missingness as MISS
    p.route_missingness("glucose", MISS.NOT_INFORMATIVE, MISS.IMPUTE_MICE,
                        uses_columns=["age"])

    # Model selection and per-model preparation. These three are the L18
    # omission-in-waiting, and they are the same shape as L14's: the recipes
    # describe transforms that HAVE NOT RUN, so the archive is the only place
    # they exist. A restored project that lost them would show a settled
    # Preprocess step with no record of what it settled.
    p.select_models(["rf", "logreg"])
    p.set_preparation_mode("uniform")
    p.set_model_recipe("rf", "power", "log1p")
    p.settle_preprocess()

    # The cohort filter, which has its own whitelist.
    keep = [l for l in p.df.index if p.df.loc[l, "age"] < 60]
    p.set_cohort("age", "under 60", keep, label="age under 60", position=1, of=2,
                 order=["under 60", "60 and over"], dropped_features=["SUBJ"])

    p.workflow_mode = "advanced"
    p.pipeline_specs = {"glm": {"impute": "median", "scale": True}}
    return p


# Two of the persisted fields CANNOT both be non-default on one project, and
# that is a property of the constitution rather than a gap in the fixture:
# question 6 (aggregation) fires only when the unit of analysis is the person,
# question 7 (temporal prediction) only when it is the record. A single fixture
# populating both would be a project the app refuses to build.
#
# So there is a second fixture, and it is named rather than worked around — a
# test that quietly skipped the assertion for these two would be asserting
# nothing about them at all.
EXCLUSIVE_FIXTURE = {"aggregation"}


def _aggregated() -> AnalysisProject:
    """The other branch: one row per person, rows combined before the seal."""
    from turbotab import repeats as R
    rng = np.random.default_rng(12)
    n = 60
    df = pd.DataFrame({
        "SUBJ": [f"S{i // 3:03d}" for i in range(n)],
        "visit": [1 + i % 3 for i in range(n)],
        "age": rng.integers(20, 80, n).astype(float),
        "glucose": rng.normal(95, 12, n),
        "outcome": rng.integers(0, 2, n),
    })
    p = AnalysisProject.from_dataframe(df, "aggregated")
    p.set_target("outcome", "classification", "high", ["two levels"])
    p.set_grain(G.PEOPLE_REPEAT, "SUBJ")
    p.set_repeat_kind(R.REPEATS)
    p.set_unit_of_analysis(R.UNIT_PERSON)
    p.set_aggregation(R.MEAN)
    return p


@pytest.fixture(scope="module")
def pair():
    before = _fully_populated()
    after = archive.from_bytes(archive.to_bytes(before))
    return before, after


@pytest.fixture(scope="module")
def exclusive_pair():
    before = _aggregated()
    after = archive.from_bytes(archive.to_bytes(before))
    return before, after


@pytest.mark.parametrize("field", sorted(PERSISTED))
def test_the_field_survives_the_archive(field, pair, exclusive_pair):
    """One test per persisted field, so a failure names the field that was lost.

    Parametrized rather than looped for exactly that reason: a single test
    asserting eighteen fields reports "the archive lost something", and the
    useful sentence is "the archive lost `selection_spec`".
    """
    before, after = exclusive_pair if field in EXCLUSIVE_FIXTURE else pair
    was, now = getattr(before, field), getattr(after, field)

    if field == "decisions":
        assert [d.to_dict() for d in now] == [d.to_dict() for d in was]
        return

    if field in ("lockbox", "cohort"):
        # These have their own key-level whitelist, and it MATERIALIZES
        # DEFAULTS: a lockbox with no `strata` comes back with `strata: []`,
        # because `lockbox.json` writes every key it knows about. So the
        # restored dict is a superset, and strict equality would fail on an
        # archive that lost nothing. The claim here is only that the field
        # arrived at all; `test_every_key_of_the_nested_whitelists_survives`
        # makes the key-level claim, which is the one that caught L13.
        assert now, f"`{field}` is empty after the round trip"
        return

    # Every persisted field must have been NON-DEFAULT going in, or this test
    # would pass on an archive that dropped it. Asserted rather than assumed.
    default = next(f for f in dataclasses.fields(AnalysisProject)
                   if f.name == field)
    if default.default is not dataclasses.MISSING:
        assert was != default.default, (
            f"`{field}` was left at its default in `_fully_populated`, so this "
            "test would pass on an archive that never wrote it. Populate it.")

    assert now == was, (
        f"`{field}` did not survive the archive.\n  before: {was!r}\n  after:  {now!r}\n"
        "If it should be carried, add it to `archive.build_members` AND to the "
        "reader in `archive.from_bytes`. If it should not, move it to "
        "REGENERATED with the reason.")


@pytest.mark.parametrize("member", ["lockbox", "cohort"])
def test_every_key_of_the_nested_whitelists_survives(member, pair):
    """The level the field check cannot reach.

    `lockbox` is one field and it survives level 1 while losing half its keys —
    that is L13 exactly. `lockbox.json` and `cohort.json` each name their keys
    one at a time, so each key needs the same guarantee the field does.
    """
    before, after = pair
    was, now = getattr(before, member), getattr(after, member)
    assert was, f"the fixture built no {member}; the check would be vacuous"

    lost = sorted(k for k in was if k not in (now or {}))
    assert not lost, (
        f"`{member}` came back missing {lost}. The {member}.json member in "
        "`archive.build_members` is an explicit whitelist — a key added to the "
        f"{member} and not added there is dropped on save, silently, and the "
        "restored project cannot tell it is incomplete.")

    changed = sorted(k for k in was
                     if k in (now or {}) and now[k] != was[k]
                     and not (isinstance(was[k], float) and abs(now[k] - was[k]) < 1e-9))
    assert not changed, f"`{member}` came back with {changed} altered in transit."


def test_the_seal_still_states_its_basis_after_a_round_trip(pair):
    """L13's own regression, kept by name as well as by the general check.

    The general check would catch it. This one says what it means: a seal that
    cannot state its basis is `group_col: None` again, and §03 exists because a
    consumer cannot tell that from a verified cross-sectional seal.
    """
    _, after = pair
    assert after.lockbox["seal_basis"] == "grouped"
    assert after.lockbox["basis_source"] == "user_stated"
    assert after.lockbox["group_col"] == "SUBJ"
    assert after.grain["answer"] == G.PEOPLE_REPEAT


def test_the_features_record_survives_even_though_the_columns_would_anyway(pair):
    """L14's own regression, and the reason it was easy to miss.

    The engineered COLUMN is in the parquet, so a restored project looks right:
    `log_glucose` is there. What was gone was the RECORD of how it got there,
    and the deferred specs, which have no other home because nothing executed
    them. A check that only compared frames would have passed.
    """
    before, after = pair
    assert "log_glucose" in after.df.columns, "the column is in the parquet"
    assert after.engineered == before.engineered, "and its receipt is not"
    assert after.deferred_transforms, (
        "the deferred specs are gone, and nothing else holds them — they "
        "describe transforms that have not run")
    assert after.selection_spec["method"] == "mutual_info"
    assert after.selection_spec["selected"] is None, (
        "a restored spec carrying a selected set would be a selection that ran")
