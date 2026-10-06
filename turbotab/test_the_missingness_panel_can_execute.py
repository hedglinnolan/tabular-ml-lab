"""`DRIVE-008` — the panel showed what would change and changed nothing.

> The product owner endorsed the spirit of the panel and named two gaps: it
> should show a snippet of the actual data for the transformation alongside the
> synopsis of what is affected, and it should let the transformation be executed
> when the step is one the user is allowed to add.

## What was wrong, and it was worse than "not wired"

Pressing *"Record this"* posted a `note` — a free-text sentence carrying the
column and the option in its payload and **no routing behind it**. So the
transcript gained a sentence describing work that never happened. That is not a
missing feature; it is the record asserting something false, which is the one
thing the governing rule forbids, in the place a manuscript reads from.

Three things had to be true before the button could be honest:

1. **One vocabulary.** `ml/missingness_plan.py` names the card's options
   `explicit_missing`, `indicator_and_impute`; `turbotab/missingness.py` names
   the declarations `explicit_category`, `indicator`. The `note` was bridging
   two vocabularies by writing prose. `CARD_STRATEGY` is the join, it lives in
   the Guided door's module because the engine builds a card for both doors, and
   an option with no declaration behind it is **refused** rather than defaulted.

2. **The timing had to be true.** *"Make Missing its own level"* was labeled
   `in_pipeline` — *fitted inside each model's pipeline, on training folds only*
   — while `project.route_missingness` has always executed it immediately,
   because a blank becoming the literal level `Missing` consults nothing but
   that row's own cell. The card was stating a timing the server contradicted,
   on the one clause that is about timing. Two options are genuinely compound —
   indicator now, fill in the fold — and they get a third timing that says both,
   because understating what already happened to the table and overstating it
   are both wrong.

3. **`drop_rows` is not a missingness strategy at all.** Clause §04: dropping
   every row with no value changes who the study is about, so it is an
   eligibility criterion reported in participant flow. Routing it through
   `declare` would file an exclusion as a preprocessing decision and lose it
   from the flow diagram. It is refused, with that reason.

## What the snippet is for

Not decoration. The question is *"could a blank here mean something?"*, and it
was asked while showing a count and a share and none of the data. The snippet
carries blank rows **and** present rows with a few neighboring columns, because
what the user needs is what distinguishes the two — a list of only the blanks
answers "how many" a second time.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import missingness_plan as MP                                 # noqa: E402
from turbotab import missingness as M                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


def _project(client, name="clinic_visits", target="outcome"):
    with open(DATA / f"{name}.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": (f"{name}.csv", fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    return pid


def _cards(client, pid):
    return client.get(f"/project/{pid}/evidence/missingness").json()["cards"]


# ── the timing the card states ───────────────────────────────────────────────

def test_the_card_states_the_timing_the_engine_actually_performs():
    """The agreement that was false, made into a check.

    `explicit_missing` was labeled *fitted inside each model's pipeline* while
    `route_missingness` executed it immediately. This asserts the card's timing
    against the door's own row-local classification, in both directions, for
    every option any fixture produces.
    """
    seen = set()
    for name in ("clinic_visits", "clinical_longitudinal", "dietary_recalls",
                 "metabolomics_untargeted"):
        df = pd.read_csv(DATA / f"{name}.csv")
        for card in MP.missingness_cards(df):
            for option in card["options"]:
                key, timing = option["key"], option["timing"]
                seen.add(key)
                if key in M.NOT_A_STRATEGY:
                    continue
                strategy = M.CARD_STRATEGY[key]
                row_local = strategy in M.ROW_LOCAL_STRATEGIES
                if timing == MP.TIMING_IMMEDIATE:
                    assert row_local, (
                        f"{key} says it is applied to the working table now, "
                        f"and `{strategy}` is not row-local — the card "
                        f"promises an immediate change clause 06 forbids")
                elif timing == MP.TIMING_IN_PIPELINE:
                    assert not row_local, (
                        f"{key} says it is fitted on training folds, and "
                        f"`{strategy}` is row-local and executes immediately — "
                        f"the card understates what happens to the table")
    assert len(seen) >= 6, f"only {len(seen)} options exercised: {sorted(seen)}"


def test_every_card_option_maps_to_a_declaration_or_is_refused_with_a_reason():
    """A key-match test across the two vocabularies.

    An option with no entry would be recorded as a sentence and executed as
    nothing, which is the defect this whole finding is about.
    """
    for name in ("clinic_visits", "clinical_longitudinal", "dietary_recalls"):
        df = pd.read_csv(DATA / f"{name}.csv")
        for card in MP.missingness_cards(df):
            for option in card["options"]:
                key = option["key"]
                if key in M.NOT_A_STRATEGY:
                    with pytest.raises(M.MissingnessRefusal):
                        M.strategy_for_card_option(key)
                    assert len(M.NOT_A_STRATEGY[key]) > 80, (
                        f"{key} is excluded with no argument behind it")
                    continue
                assert M.strategy_for_card_option(key) in M.STRATEGIES_ALL, key


def test_an_unknown_option_is_refused_rather_than_defaulted():
    with pytest.raises(M.MissingnessRefusal, match="not an option this record"):
        M.strategy_for_card_option("impute_with_vibes")
