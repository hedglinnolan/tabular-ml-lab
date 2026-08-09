"""`AUDIT-011` · the purpose card's list of who reads the answer.

## The finding, and the half this file closes

`DOMAIN_SCIENCE.md` §01.3 names five decisions whose advice **inverts** on the
prediction/inference answer. `AUDIT-011` is that none of them can read the
answer on the Streamlit workflow, which has no purpose field. `L52-B` corrected
`resolution.py`'s seal inventory and named `AnalysisProject.purpose` as the
authoritative record; the row stayed open on *recording the purpose where the
Streamlit workflow records its other answers*.

This file closes a **different** false claim the same finding produced, and it
is the one a user actually reads. `purpose.CONSUMER` is the `consumer` line on
question 2.5 — the sentence the app shows **before** the user answers, telling
them what their answer will decide. It said:

    Three more places read it: whether a value below the limit of detection may
    be substituted, whether the outcome may sit inside the imputation model,
    and whether class weighting is contraindicated. **Those four are the whole
    list today.**

Two of the four read nothing:

* **the limit-of-detection substitution** — `clinical.blocks_substitution` is
  written and has no production caller at all (`GUIDED-138` already names it as
  a capability without a consumer);
* **class weighting** — `ml.imbalance_advice.advice` has exactly one production
  call site in the repository, `pages/06_Train_and_Compare.py`, on the *other*
  workflow, and it passes `session_state["model_purpose"]` — a key three sites
  read and none writes. Nothing on the Guided door hands it this answer, so it
  can only ever return the `UNANSWERED` branch.

So the card promised four consumers and has two. Corrected to the two, **with
the two that do not read it stated** rather than quietly dropped: a list
shortened from four to two would have said less and still not told the user
their answer stops there.

## Why the sentence assertion is not `AGENT_ONBOARD` §07 trap 2

`test_the_card_says_two_read_it_and_names_the_two_that_do_not` reads a served
string, which is the shape of *a guard testing its own description*. It is
admissible only because the two tests above it **drive** what the sentence now
claims: the two named readers are observed inverting on the recorded purpose,
and the two disclaimed ones are measured over the call graph with a positive
control. The sentence is checked after the facts in it have been observed, and
that order is stated here rather than left to be noticed.

## `GUIDED-097` — two fixtures of different target shape

Question 2.5 is a journey step (`step="data"`, gated on the target), so every
claim about the card runs against `metabolomics_untargeted.csv` (`responder`,
**binary numeric**) and `multiclass_stage.csv` (`disease_stage`, **multiclass
string**). Both also carry numeric missingness, which is what the two named
readers need to be driven at all.

**The shape not covered is said out loud at the bottom of this file.**
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api                                          # noqa: E402
from turbotab import missingness as _miss                         # noqa: E402
from turbotab import purpose as _purpose                          # noqa: E402
from turbotab.project import AnalysisProject                      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(__file__).resolve().parent / "sample_data"

#: `GUIDED-097`. Two target shapes, each with a numeric column that has blanks.
#: `(fixture, target, a numeric column with missing values)`.
SHAPES = {
    "binary_numeric": ("metabolomics_untargeted.csv", "responder", "bmi"),
    "multiclass_string": ("multiclass_stage.csv", "disease_stage", "bmi"),
}

#: The clause the row was filed against, verbatim enough to survive a revert.
FALSE_CLAUSES = (
    "Three more places read it",
    "Those four are the whole list today",
    "and whether class weighting is contraindicated",
)

#: Production trees. Test files are excluded: a fixture calling `advice()` is
#: not the workflow reaching it, and counting one would be §07 trap 3 exactly.
_PRODUCTION = ("pages", "ml", "utils", "turbotab")


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _card(client, shape):
    """The served question 2.5, through the route the interface calls."""
    fixture, target, _ = SHAPES[shape]
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_target", "payload": {"column": target}})
    assert r.status_code == 200, r.text[:300]
    plan = client.get(f"/project/{pid}/interview").json()
    cards = [q for q in (plan.get("questions") or [])
             if q.get("key") == "state_purpose"]
    return pid, cards


def _project(shape, answer=None):
    """A project at the point the missingness route is reached."""
    fixture, target, _ = SHAPES[shape]
    df = pd.read_csv(DATA / fixture)
    p = AnalysisProject.from_dataframe(df, fixture)
    p.target = target
    if answer is not None:
        p.set_purpose(answer)
    return p


def _production_call_sites(function_name):
    """Every production call of `function_name`, by AST rather than by grep.

    §07 trap 5: a grep answers *does this text appear*. The question here is
    *does anything call it*, so the call nodes are matched — bare or attribute —
    and a mention inside a docstring or a comment is not one.
    """
    hits = []
    for tree in _PRODUCTION:
        for path in sorted((ROOT / tree).rglob("*.py")):
            if path.name.startswith("test_") or "test" in path.parts:
                continue
            try:
                module = ast.parse(path.read_text(encoding="utf-8",
                                                  errors="ignore"))
            except SyntaxError:                       # pragma: no cover
                continue
            for node in ast.walk(module):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (func.attr if isinstance(func, ast.Attribute)
                        else func.id if isinstance(func, ast.Name) else None)
                if name == function_name:
                    hits.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return hits


# ═══════════ 1 · the two the card names really do read the answer ═══════════

@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_missing_indicator_route_inverts_on_the_recorded_purpose(shape):
    """§01.3's first inversion, driven. This is claim one of the corrected
    sentence, observed rather than described."""
    _, _, column = SHAPES[shape]

    # POSITIVE CONTROL (`GUIDED-045`) — the same call succeeds under the other
    # answer, so the refusal below is the purpose's doing and not the column's.
    predicting = _project(shape, "prediction")
    predicting.route_missingness(column, "not_sure", "indicator")

    inferring = _project(shape, "inference")
    with pytest.raises(Exception) as caught:
        inferring.route_missingness(column, "not_sure", "indicator")
    assert "association" in str(caught.value).lower(), (
        f"{shape}: the missing-indicator route refused for some reason other "
        f"than the recorded purpose: {str(caught.value)[:200]}")


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_outcome_in_the_imputation_scope_inverts_on_it_too(shape):
    """§01.3's second inversion, and claim two of the corrected sentence.

    `AUDIT-005`: under prediction the outcome inside a MICE scope is a hard
    blocker; under inference it is the correct specification and the note
    travels with the record.
    """
    _, target, _ = SHAPES[shape]

    refused = _miss.outcome_in_scope(target, "prediction")
    allowed = _miss.outcome_in_scope(target, "inference")
    # POSITIVE CONTROL — the two answers are genuinely different verdicts.
    assert refused["refuse"] is True and allowed["refuse"] is False, (
        f"{shape}: the recorded purpose changes nothing here, so naming this "
        f"a reader of the answer would be the card describing a fork that "
        f"does not fork: {refused!r} / {allowed!r}")

    unanswered = _miss.outcome_in_scope(target, None)
    assert unanswered["refuse"] is True, unanswered


# ═══════════ 2 · and the two it disclaims read nothing ═══════════

def test_nothing_on_this_door_hands_the_answer_to_the_other_two():
    """The corrected sentence's negative half, measured over the call graph.

    **Positive control first** (`GUIDED-045`): the same sweep must find the two
    readers the card DOES claim. An all-absence result from a sweep that finds
    nothing is a statement about the sweep.
    """
    indicator = _production_call_sites("blocks_indicator")
    outcome = _production_call_sites("outcome_in_scope")
    assert indicator and outcome, (
        f"the sweep lost its subject: it found no production call of "
        f"`blocks_indicator` ({indicator}) or `outcome_in_scope` ({outcome}), "
        f"so a zero below would say nothing about routing")

    lod = _production_call_sites("blocks_substitution")
    assert not lod, (
        f"`clinical.blocks_substitution` is now called at {lod} — the "
        f"limit-of-detection fork reads the purpose, and question 2.5's card "
        f"must stop saying nothing calls it. AUDIT-011.")

    weighting = _production_call_sites("advice")
    # POSITIVE CONTROL — the advisory exists and is reached from somewhere, so
    # "not from this door" is about routing rather than about dead code.
    assert weighting, (
        "`ml.imbalance_advice.advice` has no production caller at all; the "
        "card's claim about which door reaches it is then the wrong claim")
    on_this_door = [h for h in weighting if h.startswith("turbotab/")]
    assert not on_this_door, (
        f"the Guided door now reaches the class-weighting advisory at "
        f"{on_this_door}; question 2.5's card says it does not. AUDIT-011.")


# ═══════════ 3 · the sentence the user reads, after the facts in it ═══════════

@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_card_says_two_read_it_and_names_the_two_that_do_not(client, shape):
    """The corrected claim, on the route the interface calls.

    See this file's docstring on why reading a served string is admissible
    here: everything it asserts has been driven above.
    """
    _, cards = _card(client, shape)

    # POSITIVE CONTROL — question 2.5 is served at all on this shape.
    assert len(cards) == 1, (
        f"{shape}: expected exactly one `state_purpose` card, got "
        f"{len(cards)}; the sentence under test never reaches a user")
    consumer = cards[0].get("consumer") or ""
    assert consumer.strip(), f"{shape}: the purpose card carries no consumer line"

    for clause in FALSE_CLAUSES:
        assert clause not in consumer, (
            f"{shape}: question 2.5 still tells the user {clause!r}. Two of "
            f"the four named decisions read the answer: "
            f"`clinical.blocks_substitution` has no production caller and "
            f"`ml.imbalance_advice.advice` is reached only from "
            f"`pages/06_Train_and_Compare.py`, which passes a session key "
            f"nothing writes. AUDIT-011. The card read: {consumer!r}")

    # The shelf is not shortened — the corrected sentence still says what the
    # answer decides, and now also says where it stops.
    assert "Those two are the whole list today" in consumer, (
        f"{shape}: the card no longer states how many decisions read the "
        f"answer. AUDIT-028's model is a claim corrected, not deleted. The "
        f"card read: {consumer!r}")
    assert "limit of detection" in consumer and "nothing calls it" in consumer, (
        f"{shape}: the card dropped the limit-of-detection fork instead of "
        f"saying nothing calls it yet. The card read: {consumer!r}")
    assert "class-weighting advisory" in consumer, (
        f"{shape}: the card dropped the class-weighting fork instead of "
        f"saying this door does not reach it. The card read: {consumer!r}")


def test_the_streamlit_schema_declares_the_purpose_is_not_one_of_its_answers():
    """The other door's half of the same claim, in the file that declares it.

    `utils/session_state.py` is the Streamlit workflow's statement of what it
    records. Its initializer said *"Initialize all session state variables with
    defaults"* — 50 keys are declared there and 128 more are read from
    `st.session_state` and initialized nowhere, so the word *all* was false, and
    the one it was falsest about is the purpose. The schema now states the
    absence instead of implying completeness.

    The docstring is read only after the schema itself is measured: the defaults
    really carry no purpose key, and a production site really reads one.
    """
    src = (ROOT / "utils" / "session_state.py").read_text(encoding="utf-8")
    module = ast.parse(src)
    declared = None
    for node in ast.walk(module):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "defaults"
                and isinstance(node.value, ast.Dict)):
            declared = [k.value for k in node.value.keys
                        if isinstance(k, ast.Constant)]
            break

    # POSITIVE CONTROL — the schema was found and is not empty, so "no purpose
    # key in it" is a fact about the schema and not about a failed parse.
    assert declared, (
        "`init_session_state` no longer declares a `defaults` dict this test "
        "can read; the sweep lost its subject")
    assert not [k for k in declared if "purpose" in k], (
        f"the Streamlit workflow now declares a purpose field: "
        f"{[k for k in declared if 'purpose' in k]}. AUDIT-011's other half "
        f"has landed and every sentence calling `AnalysisProject.purpose` the "
        f"only record must be re-read.")

    initializer = next(n for n in module.body
                       if isinstance(n, ast.FunctionDef)
                       and n.name == "init_session_state")
    doc = ast.get_docstring(initializer) or ""
    # ANCHORED TO THE SUMMARY LINE, not to the docstring as a whole. The body
    # QUOTES the false sentence in AUDIT-021's before/after style, so a matcher
    # over the whole text would fire on the correction itself and its silence
    # would mean nothing.
    summary = (doc.strip().splitlines() or [""])[0]
    assert "all session state variables" not in summary, (
        f"`init_session_state` still claims to initialize ALL session state "
        f"variables. It declares {len(declared)} keys and the workflow reads "
        f"many more that it never initializes — the purpose among them. "
        f"AUDIT-011. The summary line read: {summary!r}")
    assert "AnalysisProject.purpose" in doc, (
        f"the Streamlit schema does not say where the purpose IS recorded, so "
        f"its absence here reads as an oversight rather than as the stated "
        f"divergence. AUDIT-011. The docstring read: {doc[:200]!r}")


def test_the_module_constant_and_the_served_card_are_one_string(client):
    """`ml/router.py` builds the card from `purpose.question()`. If it ever
    composed its own copy, the correction above would be true of the constant
    and false on the screen — §07 trap 6, one layer up."""
    _, cards = _card(client, "binary_numeric")
    assert cards, "no purpose card served"
    assert cards[0].get("consumer") == _purpose.CONSUMER, (
        "the served card's consumer line is not `purpose.CONSUMER`; a "
        "correction to the constant would not reach the screen")


#: NOT COVERED, said out loud — `GUIDED-097`'s second clause.
#:
#: A CONTINUOUS TARGET. Question 2.5 is gated on `target` alone and reads
#: neither the task type nor the target's dtype, and both inversions driven
#: above are functions of `(purpose, column)` rather than of the outcome — so a
#: regression target exercises no branch these two do not. Named rather than
#: assumed.
#:
#: THE STREAMLIT DOOR. There is no card and no `AnalysisProject` to drive over
#: there — that absence IS `AUDIT-011`, and it is what the corrected sentence's
#: last clause is about. It is measured here over the call graph instead,
#: because there is no served object to interrogate. Recording the purpose
#: where that workflow records its other answers is still open.
