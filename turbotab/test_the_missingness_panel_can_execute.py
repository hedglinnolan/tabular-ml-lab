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


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _project(client, name="clinic_visits", target="outcome"):
    with open(DATA / f"{name}.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": (f"{name}.csv", fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    return pid


def _frame(pid):
    from turbotab.api import STORE
    return STORE.get(pid).df


def _cards(client, pid):
    return client.get(f"/project/{pid}/evidence/missingness").json()["cards"]


# ── the snippet ──────────────────────────────────────────────────────────────

def test_the_card_carries_real_rows_from_this_table():
    """The data, not a second statement of the count."""
    client = _client()
    pid = _project(client)
    cards = _cards(client, pid)
    assert cards, "no missingness cards on this fixture"
    df = _frame(pid)

    for card in cards:
        snip = card["snippet"]
        assert snip["rows"], f"{card['column']} carries no rows"
        assert snip["n_blank_shown"] >= 1, (
            f"{card['column']}: a missingness card showing no blank rows is a "
            f"count with extra steps")
        for row in snip["rows"]:
            # THE ROWS ARE REAL. Checked against the frame the project holds,
            # not against the card's own description of them.
            assert row["row"] in df.index, (
                f"{card['column']}: row {row['row']!r} is not in the table")
            actual = df.at[row["row"], card["column"]]
            assert bool(pd.isna(actual)) == row["missing"], (
                f"{card['column']} row {row['row']}: the snippet says "
                f"missing={row['missing']} and the frame disagrees")


def test_the_snippet_shows_what_a_value_looks_like_beside_what_a_blank_does():
    """Both sides, because the question is what distinguishes them.

    A snippet of only the blanks would answer "how many" a second time, and the
    user is being asked what the absence MEANS.
    """
    client = _client()
    pid = _project(client)
    both = [c for c in _cards(client, pid)
            if c["snippet"]["n_present_shown"] > 0]
    assert both, (
        "no card shows a present row; on a column that is not entirely blank "
        "that is the half the question needs")
    card = both[0]
    assert card["snippet"]["neighbors"], (
        "the snippet shows the column alone, which is a list of the word "
        "'missing'")
    assert len(card["snippet"]["neighbors"]) <= MP.SNIPPET_COLUMNS


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


# ── the apply path ───────────────────────────────────────────────────────────

def test_a_row_local_choice_changes_the_working_table_now():
    """**The read-back.** Not *a decision was recorded* — *the column changed.*

    The whole defect was a button that wrote a sentence and did nothing, so a
    test that read the transcript would have passed on the broken version.
    """
    client = _client()
    pid = _project(client)
    card = next(c for c in _cards(client, pid)
                if any(o["key"] == "explicit_category" for o in c["options"]))
    column = card["column"]

    before = _frame(pid)
    n_blank = int(before[column].isna().sum())
    assert n_blank, f"{column} has no blanks; this test proves nothing"

    r = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness", "subject": column,
        "payload": {"column": column, "card_option": "explicit_missing",
                    "mechanism": "informative"}})
    assert r.status_code == 200, r.text

    after = _frame(pid)
    assert int(after[column].isna().sum()) == 0, (
        f"{column} still has blanks, so the panel recorded a decision and "
        f"changed nothing — which is the finding")
    assert int((after[column] == M.MISSING_LEVEL).sum()) == n_blank, (
        f"the blanks did not become the level {M.MISSING_LEVEL!r}")
    assert after.index.equals(before.index), (
        "a row-local strategy renumbered the rows")


def test_a_stateful_choice_is_recorded_and_the_table_is_untouched():
    """Clause §06's other half, and the more important one.

    Materializing an imputation on the working table before the split is the
    canonical preprocessing leak. The decision lands; the frame does not move.
    """
    client = _client()
    pid = _project(client)
    card = next(c for c in _cards(client, pid)
                if any(o["key"] == "impute_median" for o in c["options"]))
    column = card["column"]
    before = _frame(pid).copy()

    r = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness", "subject": column,
        "payload": {"column": column, "card_option": "impute_median",
                    "mechanism": "not_informative"}})
    assert r.status_code == 200, r.text

    pd.testing.assert_frame_equal(_frame(pid), before)
    said = next(d for d in client.get(f"/project/{pid}").json()["decisions"]
                if d["kind"] == "route_missingness")
    assert "will be filled" in said["text"], (
        f"the sentence does not carry the timing: {said['text']!r}")
    assert said["payload"]["fit_on"] == "training folds only"


def test_the_panel_no_longer_records_a_note_that_routes_nothing():
    """The specific defect, watched on the wire.

    A `note` carrying the column and the option is a sentence about work that
    did not happen, and in the transcript it is indistinguishable from work that
    did. The first version of this test POSTed to the API directly and asserted
    the record — which a revert probe showed proved nothing about the button,
    because the page was never run. So this one dispatches at the real control
    and reads the body its real `decide()` composes.
    """
    from turbotab import pageharness as H
    if not H.available():
        pytest.skip("no JS engine on this machine")

    client = _client()
    pid = _project(client)
    project = client.get(f"/project/{pid}").json()
    cards = client.get(f"/project/{pid}/evidence/missingness").json()

    body = H.run(
        """
        // §07's ORDER, and it now binds on this door too (`GUIDED-091`): the
        // strategies are not on screen until the mechanism is answered, so the
        // drive answers it exactly as a user would before it can press one.
        var mech = /data-miss-mech-for="([^"]+)"/.exec(__harness.html('missBox'));
        if (!mech) throw new Error('no mechanism question rendered');
        __harness.dispatch('click', __harness.target(
          {'data-miss-mech-for': mech[1], 'data-miss-mech-value': 'not_sure'},
          ['pill']));
        var html = __harness.html('missBox');
        var m = /data-miss-choose="([^"]+)"[^>]*data-miss-opt="([^"]+)"/.exec(html);
        if (!m) throw new Error('no missingness control rendered');
        __harness.dispatch('click', __harness.target(
          {'data-miss-choose': m[1], 'data-miss-opt': m[2],
           'data-miss-mech': 'not_sure'}, ['cbtn']));
        var posts = __harness.posts();
        __emit(posts.length ? posts[posts.length - 1] : null);
        """,
        routes={
            f"/project/{pid}": project,
            f"/project/{pid}/interview?step=data":
                client.get(f"/project/{pid}/interview?step=data").json(),
            f"/project/{pid}/interview?step=explore": {"questions": []},
            f"/project/{pid}/evidence/missingness": cards,
            f"/project/{pid}/evidence/plausibility": {"columns": []},
        }, search=f"?project={pid}")

    assert body, "pressing the control sent nothing at all"
    assert body["body"]["kind"] == "route_missingness", (
        f"the panel still records a {body['body']['kind']!r} — a sentence with "
        f"no routing behind it")
    assert body["body"]["payload"]["card_option"], (
        "the request carries no option, so the server cannot know which "
        "strategy was chosen")

    # And the server accepts exactly that body.
    replay = client.post(f"/project/{pid}/decision", json=body["body"])
    assert replay.status_code in (200, 409), replay.text


def test_dropping_the_rows_is_refused_as_a_missingness_strategy():
    """Clause §04. A complete-case analysis changes who the study is about, so
    it is an eligibility criterion reported in participant flow — not a way of
    handling a blank. Refused with that reason rather than quietly filed as
    preprocessing."""
    client = _client()
    pid = _project(client)
    card = next(c for c in _cards(client, pid)
                if any(o["key"] == "drop_rows" for o in c["options"]))
    r = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness", "subject": card["column"],
        "payload": {"column": card["column"], "card_option": "drop_rows",
                    "mechanism": "not_sure"}})
    assert r.status_code == 400
    assert "participant flow" in r.text


# ── what the driver presses ──────────────────────────────────────────────────

def test_the_button_says_which_of_the_two_things_it_will_do():
    """Clause §06 read back off the control.

    *"Record this"* and *"Apply this now"* describe different events, and a
    single label for both is the panel being vague about the one distinction
    the clause exists to draw.
    """
    from turbotab import pageharness as H
    if not H.available():
        pytest.skip("no JS engine on this machine")

    client = _client()
    pid = _project(client)
    project = client.get(f"/project/{pid}").json()
    html = H.run(
        """
        // The mechanism first, because §07's fork now binds on this door too
        // (`GUIDED-091`) — no strategy is on screen until it is answered.
        var mech = /data-miss-mech-for="([^"]+)"/.exec(__harness.html('missBox'));
        if (!mech) throw new Error('no mechanism question rendered');
        __harness.dispatch('click', __harness.target(
          {'data-miss-mech-for': mech[1], 'data-miss-mech-value': 'not_sure'},
          ['pill']));
        __emit(__harness.html('missBox'));
        """, routes={
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/evidence/missingness":
            client.get(f"/project/{pid}/evidence/missingness").json(),
        f"/project/{pid}/evidence/plausibility": {"columns": []},
    }, search=f"?project={pid}")

    assert "data-miss-choose" in html, "the missingness panel did not render"
    assert "Apply this now" in html, (
        "no option offers to apply now, and `explicit_category` is row-local")
    assert "Record this" in html, (
        "no option is recorded for the fold, and every imputation is")
    assert "Show me these rows" in html, "the data snippet did not render"
