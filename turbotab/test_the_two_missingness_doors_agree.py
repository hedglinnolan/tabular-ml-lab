"""`GUIDED-090` / `GUIDED-091` / `GUIDED-098` — one question, two doors, three
disagreements.

The Explore card and the Preprocess panel both ask what to do about a blank.
Both surfaces are legitimate and the vocabularies were never the defect —
`PRODUCT_VISION.md` §04 already names their relationship: deferral is a
first-class disposition and a noticing resurfaces at the step it targets. What
they may not do is answer the same question differently.

They did, three times.

**`GUIDED-090` — the shelf was shortened.** Measured on `clinic_visits.csv`, a
numeric column: the Explore card offered four strategies, `/preprocess` offered
five, and the missing one was `leave`. `_numeric_options` never emitted it while
`_binary_options` did, and `CARD_STRATEGY` already knew it. That is the product
owner's own ruling at a surface nobody had compared — *judgment renders as
ranking, never as absence* — and it was missing on exactly the column where the
absence carries signal.

**`GUIDED-091` — the mechanism was never asked.** The card carried no
`mechanism` field at all, so the page's `c.mechanism || "not_sure"` was
unconditional and every column routed from that door recorded `not_sure`.
`blocks()` fires only on `informative`, so **§07's blocker was unreachable from
that door by any user on any column** — not bypassed, unreachable.

**`GUIDED-098`, found while fixing those two — one click, two methods
sentences.** The card's `indicator_and_impute` promised *"imputed with the
training-fold median and a missingness indicator was retained"* and
`CARD_STRATEGY` mapped it to `INDICATOR`, whose recorded sentence is *"the
underlying value is left blank."* The binary branch's `indicator` did the same.
After `GUIDED-095` the pipeline honors the record, so the fill the card promised
does not happen — the contradiction stopped being about prose and became about
the fit.

## The fix, in one sentence each

**One table decides what both doors offer.** `_options_for` iterates
`missingness.STRATEGIES_BY_BRANCH`; `GUIDED-086` made the CHECK read it and this
is the half that OFFERS. **One composer writes the sentence.**
`missingness.sentence_for` is asked by the card and by `declare`. **The card
asks the mechanism**, with the same copy and the same order Preprocess uses.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import missingness_plan as MP                                 # noqa: E402
from turbotab import api                                              # noqa: E402
from turbotab import missingness as M                                 # noqa: E402
from turbotab import pageharness as H                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: Enough shapes that a branch cannot be right by accident: a continuous
#: column, a 0/1 numeric one (which the two doors used to route DIFFERENTLY),
#: and a text one.
FIXTURES = ("clinic_visits.csv", "survey_instrument.csv",
            "metabolomics_untargeted.csv")


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _project(client, name="clinic_visits.csv"):
    with open(DATA / name, "rb") as fh:
        pid = client.post("/project", files={
            "file": (name, fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "hba1c"}})
    return pid


# ── one table decides what both doors offer ──────────────────────────────────

@pytest.mark.parametrize("fixture", FIXTURES)
def test_both_doors_offer_the_same_strategies_for_the_same_column(fixture):
    """`GUIDED-090`. The offer, not the check.

    Every strategy the Explore card offers must be one the record permits on
    that column's branch, AND every strategy the branch permits must be
    offered. A shorter shelf on one door is the app making a decision in the
    user's name at the surface where they are looking.
    """
    # `threshold=0.0`: this claim is about what the card OFFERS, not about
    # which columns are notable enough to earn one, so every column with a
    # blank is exercised rather than only the loud ones.
    frame = pd.read_csv(DATA / fixture)
    cards = MP.missingness_cards(frame, threshold=0.0)
    assert cards, f"{fixture} produced no card, so this asserts nothing"
    for card in cards:
        permitted = set(M.STRATEGIES_BY_BRANCH[card["branch"]])
        offered = {o["key"] for o in card["options"] if o.get("is_strategy")}
        assert offered == permitted, (
            f"{fixture}:{card['column']} ({card['branch']}) — the card offers "
            f"{sorted(offered)} and the record permits {sorted(permitted)}; "
            f"missing from the card: {sorted(permitted - offered)}")


def test_leave_is_on_the_numeric_card_where_the_absence_carries_signal():
    """The instance, named, because a set comparison passing does not say
    WHICH option came back."""
    frame = pd.read_csv(DATA / "clinic_visits.csv")
    numeric = [c for c in MP.missingness_cards(frame)
               if c["branch"] == "numeric"]
    assert numeric, "clinic_visits has no numeric card"
    for card in numeric:
        keys = {o["key"] for o in card["options"]}
        assert "leave" in keys, (
            f"{card['column']} cannot be left alone from the Explore door, and "
            "it is the option that matters most where a blank means something")


def test_the_card_and_the_record_route_a_column_to_the_same_branch():
    """The other half of the same divergence, and it was invisible.

    `_kind_of` called a 0/1 numeric column `binary` and offered `impute_mode`;
    `declare` calls it `numeric` by dtype and REFUSES `impute_mode` there
    (`GUIDED-086`), so the card offered a route the record would reject. The
    card keeps its three-way `dtype_route` for how it words the question and
    routes the OFFER on the record's branch.
    """
    frame = pd.DataFrame({
        "zero_one": [1, 0, None, 1, None, 0, 1, None, 0, 1] * 4,
        "text": ["a", "b", None, "c", None, "a", "b", None, "c", "a"] * 4,
    })
    cards = {c["column"]: c for c in MP.missingness_cards(frame)}
    assert cards["zero_one"]["dtype_route"] == "binary"
    assert cards["zero_one"]["branch"] == "numeric", (
        "the card routes a 0/1 numeric column to a branch the record does not")
    for card in cards.values():
        for option in card["options"]:
            if not option.get("is_strategy"):
                continue
            strategy = M.strategy_for_card_option(option["key"])
            M.declare(card["column"], card["branch"], "not_sure", strategy)


def test_the_one_option_that_is_not_a_strategy_is_still_offered_with_its_reason():
    """**Do not settle this by deleting the card.** `drop_rows` is an
    eligibility criterion wearing a missingness costume, and the right answer
    is the argument plus somewhere to go — a gap that becomes routing is worth
    more than a transform."""
    frame = pd.read_csv(DATA / "clinic_visits.csv")
    for card in MP.missingness_cards(frame):
        drop = [o for o in card["options"] if o["key"] == "drop_rows"]
        assert drop, f"{card['column']} offers no route for dropping the rows"
        assert drop[0]["is_strategy"] is False
        assert "participant flow" in drop[0]["consequence"]
        with pytest.raises(M.MissingnessRefusal):
            M.strategy_for_card_option("drop_rows")


# ── one composer writes the sentence ─────────────────────────────────────────

@pytest.mark.parametrize("fixture", FIXTURES)
def test_the_card_and_the_record_write_the_same_sentence(fixture):
    """`GUIDED-098`, and it is asserted as EQUALITY of the served strings
    rather than as identity, because the card is composed on one request and
    the record on another — two processes, one composer.

    The pairing that was wrong is the one that matters: an option promising a
    training-fold median mapped to a declaration whose sentence says the value
    is left blank. One click, two methods sentences, opposite claims.
    """
    frame = pd.read_csv(DATA / fixture)
    for card in MP.missingness_cards(frame, threshold=0.0):
        for option in card["options"]:
            if not option.get("is_strategy"):
                continue
            strategy = M.strategy_for_card_option(option["key"])
            recorded = M.declare(card["column"], card["branch"], "not_sure",
                                 strategy)
            assert option["decision_sentence"] == recorded["sentence"], (
                f"{fixture}:{card['column']} option {option['key']!r} — the "
                f"card says {option['decision_sentence']!r} and the record "
                f"says {recorded['sentence']!r}")


def test_the_compound_strategy_says_it_fills_and_then_fills(client):
    """`indicator_and_impute` is genuinely both halves of clause §06, and
    modeling it as one is what produced the contradiction: the indicator lands
    now, the fill is fitted in the fold."""
    from turbotab import pipeline_plan, training

    pid = _project(client, "clinic_visits.csv")
    for kind, payload in [("set_purpose", {"answer": "prediction"}),
                          ("set_grain", {"answer": "one_row_per_person"}),
                          ("set_eligibility", {"answer": "everyone"}),
                          ("seal", {"fraction": 0.25})]:
        client.post(f"/project/{pid}/decision",
                    json={"kind": kind, "payload": payload})
    project = api.STORE.get(pid)
    column = next(c["column"] for c in project.missingness_survey()
                  if c["branch"] == "categorical")
    n_blank = int(project.df[column].isna().sum())

    r = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness",
        "payload": {"column": column, "mechanism": "informative",
                    "strategy": "indicator_and_impute"}})
    assert r.status_code == 200, r.text[:250]

    record = [d for d in project.missingness if d["column"] == column][0]
    assert "indicator is added" in record["sentence"]
    assert "filled with" in record["sentence"], (
        "the compound strategy's sentence promises no fill, which is the "
        "sentence that used to contradict the card")

    # THE ROW-LOCAL HALF LANDED.
    assert M.indicator_column(column) in project.df.columns
    assert int(project.df[column].isna().sum()) == n_blank, (
        "the fill was materialized on the working table, which is the leak "
        "clause §06 defers to avoid")

    # AND THE STATEFUL HALF IS IN THE FITTED PIPELINE.
    features = training._feature_frame(project.working_table, "hba1c", None)
    plan = pipeline_plan.compose(project, "histgb_reg", features)
    step = plan.step_for(column)
    assert step.sentence is record["sentence"]
    filled = [name for name, _, cols in plan._blocks[0].transformers
              if column in list(cols)]
    assert filled == ["fill_impute_mode"], (
        f"the compound strategy's fill did not reach the pipeline: {filled}")


# ── the card asks the mechanism ──────────────────────────────────────────────

def test_the_card_carries_the_mechanism_question_and_no_answer(client):
    """`GUIDED-091`. `mechanism` is `None` — *not yet asked* — and the question
    travels with the card so the door can put it."""
    pid = _project(client)
    cards = client.get(f"/project/{pid}/evidence/missingness").json()["cards"]
    assert cards, "no card to check"
    for card in cards:
        assert card["mechanism"] is None, (
            "the card ships an answer to a question the user was never asked")
        question = card["mechanism_question"]
        assert card["column"] in question["question"]
        assert question["values"] == list(M.MECHANISMS)
        assert question["why"] == M.MECHANISM_WHY, (
            "the Explore door asks §07's question in its own words, so the two "
            "doors can drift about what the question means")


@pytest.mark.skipif(not H.available(), reason="no JS engine on this machine")
def test_no_strategy_is_on_screen_until_the_mechanism_is_answered(client):
    """**§07's fork asserted as a property of the surface**, which is how the
    Preprocess panel already carries it. A list of fills beside an unanswered
    question is an invitation to pick one without answering it — and here it
    was worse: the page supplied `not_sure` on the user's behalf.
    """
    pid = _project(client)
    project = client.get(f"/project/{pid}").json()
    routes = {
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/evidence/missingness":
            client.get(f"/project/{pid}/evidence/missingness").json(),
        f"/project/{pid}/evidence/plausibility": {"columns": []},
    }
    out = H.run(
        """
        var before = __harness.html('missBox');
        var mech = /data-miss-mech-for="([^"]+)"/.exec(before);
        if (!mech) throw new Error('no mechanism question rendered');
        __harness.dispatch('click', __harness.target(
          {'data-miss-mech-for': mech[1], 'data-miss-mech-value': 'informative'},
          ['pill']));
        __emit({before: before, after: __harness.html('missBox'),
                column: mech[1]});
        """, routes=routes, search=f"?project={pid}")

    assert "data-miss-choose" not in out["before"], (
        "the strategies are on screen before the mechanism is answered, so a "
        "user can pick a fill without saying what a blank means")
    assert "data-miss-choose" in out["after"], (
        "answering the mechanism did not bring the strategies out, so the "
        "card is now a dead end")
    assert 'data-miss-mech="informative"' in out["after"], (
        "the strategy button carries a mechanism other than the one the user "
        "chose, which is the unconditional `not_sure` all over again")
    assert 'data-miss-mech="not_sure"' not in out["after"]


@pytest.mark.skipif(not H.available(), reason="no JS engine on this machine")
def test_clause_07s_blocker_is_reachable_from_the_explore_door(client):
    """**The consequence, and it is not cosmetic.**

    `blocks()` fires only when the mechanism is `informative` — deliberately,
    and the reasoning in its docstring is right: turning an admission of
    uncertainty into a wall teaches people to stop admitting it. But that
    reasoning assumes `not_sure` was ANSWERED. Here it was supplied, so the
    interruption §07 exists to raise, with its typed acknowledgment and its
    recorded stability assumption, could not be reached from this door by any
    user on any column.
    """
    pid = _project(client)
    project = api.STORE.get(pid)
    column = next(c["column"] for c in project.missingness_survey()
                  if c["branch"] == "numeric")

    # The page composes the body; the record is what answers it.
    served = client.get(f"/project/{pid}").json()
    out = H.run(
        """
        function settle(n){
          var p = Promise.resolve();
          for (var i = 0; i < n; i++) { p = p.then(function(){}); }
          return p;
        }
        settle(10).then(function(){
          __harness.dispatch('click', __harness.target(
            {'data-miss-mech-for': COLUMN, 'data-miss-mech-value': 'informative'},
            ['pill']));
          return settle(6);
        }).then(function(){
          // THE BUTTON THE PAGE RENDERED, attributes and all. `target()` builds
          // an element from what it is handed, so composing the mechanism here
          // would be this test supplying the very thing GUIDED-091 is about.
          // It is read off the render instead.
          //
          // PARSED, NOT PATTERN-MATCHED ON ADJACENCY. This read a single regex
          // requiring `data-miss-choose`, `data-miss-opt` and `data-miss-mech`
          // to be adjacent in that order, and L48-A1 inserting a `data-ac`
          // between the first two turned it red — a true claim broken by an
          // unrelated attribute. The whole button is parsed now and every
          // attribute travels, which is also more faithful to a press.
          var html = __harness.html('missBox') || '';
          var re = /<button\\b([^>]*)>/g, hit = null, mm;
          while ((mm = re.exec(html))){
            var attrs = {}, a = /([a-zA-Z-]+)="([^"]*)"/g, kv;
            while ((kv = a.exec(mm[1]))) attrs[kv[1]] = kv[2];
            if (attrs['data-miss-choose'] === COLUMN &&
                attrs['data-miss-opt'] === 'impute_median'){ hit = attrs; break; }
          }
          if (!hit) throw new Error('no impute_median control rendered');
          __harness.dispatch('click', __harness.target(hit, ['cbtn']));
          return settle(10);
        }).then(function(){
          var posts = __harness.posts();
          __emit(posts.length ? posts[posts.length - 1] : null);
        });
        """.replace("COLUMN", f'"{column}"'),
        routes={
            f"/project/{pid}": served,
            f"/project/{pid}/interview?step=data":
                client.get(f"/project/{pid}/interview?step=data").json(),
            f"/project/{pid}/interview?step=explore": {"questions": []},
            f"/project/{pid}/evidence/missingness":
                client.get(f"/project/{pid}/evidence/missingness").json(),
            f"/project/{pid}/evidence/plausibility": {"columns": []},
            f"POST /project/{pid}/decision": served,
        }, search=f"?project={pid}")

    assert out, "the press produced no request at all"
    body = out["body"] if isinstance(out["body"], dict) else None
    if body is None:
        import json as _json
        body = _json.loads(out["body"])
    assert body["payload"]["mechanism"] == "informative", (
        f"the page posted {body['payload']['mechanism']!r}, which is the "
        "unconditional fallback GUIDED-091 is about")

    # AND THE RECORD RAISES THE BLOCKER against exactly that body.
    refused = client.post(f"/project/{pid}/decision", json=body)
    assert refused.status_code == 409, (
        "filling an informatively-missing column from the Explore door was "
        "accepted, so §07's blocker is still unreachable from this door")
    detail = refused.json()["detail"]
    assert detail["acknowledgment_kind"] == "typed"
    assert detail["exits"], "the blocker offers no way through"
