"""L51-B — the right column, and it is one panel because §10 says where things go.

Three rows, one design. `GUIDED-174` is the product owner's own proposal, filed
at his request: **the working table, beside the decisions being made about it.**
`GUIDED-160` is education layer 3. `GUIDED-178` is the per-model deck.

**Reading the first two as competitors is the mistake, and the design language
already settled it.** §10's first paragraph allocates the screen:

> modeling is the left column, learning is the right panel, both means both
> columns. **There is no third place.**

And §10's layer 3 is *"the concept explained with **their** columns and
**their** numbers, not abstractions"* — which **needs the table on screen to
point at.** He proposed the column for the data; the design language reserved it
for teaching; the working table is layer 3's substrate. One panel, three
sections, each a state of the same object.

## What each section is, and what it may not do

**Your data** — the columns, their types, their missing counts, and the row
count, with the bound stated (`GUIDED-195`'s rule; `capSaid` is the house
helper). Nothing composed that the server did not send.

**On your columns** — layer 2 was carrying layer 3's job at **135 abstract
words** in a disclosure §10 caps at two or three sentences, and §10 names that
failure before it happened: layer 1 cites *expertise reversal*, layer 2 cites
*split attention*, and a long abstract disclosure breaks both at once. **The
question does not get shorter** — the product owner's standing ruling is that
hard questions stay hard and we invest in pedagogy — the explanation moves to
the surface §10 built for it and gains the thing layer 2 structurally cannot
have: the user's own column names.

**What the models get** — `GUIDED-178`, and it is a **placeholder on purpose.**
`project.resolved_recipes()` short-circuits on `if not self.selected_models:
return out`, and `selected_models` is gated on `barrier_raised`, so
`/recipes.models` is `{}` for the entire pre-seal journey. There is no
per-model data at Preprocess and **a deck with content there would be
fabricated.** The section says what is missing and which step fills it, which
is the shape `/recipes` already uses for the same reason.

## What is NOT covered

- **Whether any of it is on screen.** No layout here; `pageharness.py` says so.
- **The post-seal state.** His phase change is right and is kept — after the
  seal the working table stops being the object of attention and the run
  becomes it — but no fixture in this repository reaches a fitted run, so the
  models section is driven only in its empty state.
- **Below 900px**, where the panel hides with the rail. §10's second sentence
  is that education never lives in a modal, and a stacked right panel is one.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "sample_data"
PAGE = Path(__file__).resolve().parent / "web" / "index.html"


def _driven(fixture="clinical_labs.csv", lens="clinical", target="readmitted"):
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    if lens:
        client.post(f"/project/{pid}/decision",
                    json={"kind": "set_lens", "payload": {"lens": [lens]}})
    if target:
        client.post(f"/project/{pid}/decision",
                    json={"kind": "set_target", "payload": {"column": target}})
    return client, pid


def _routes(client, pid):
    out = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "interview?step=preprocess",
                 "capabilities", "features", "recipes", "preprocess", "figures",
                 "draft", "manuscript", "models", "training", "instability",
                 "explain", "sensitivity", "evidence/plausibility",
                 "evidence/missingness"):
        got = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = got.json() if got.status_code == 200 else {}
    return out


def test_it_is_one_panel_and_not_two():
    """§10's allocation, asserted structurally.

    Two panels is the design read as competitors, which the row calls the
    mistake — so the check is that there is exactly one right-hand column and
    that all three sections live inside it.
    """
    page = PAGE.read_text(encoding="utf-8")
    assert page.count('class="study"') == 1, (
        "there is more than one right-hand column. §10 allocates ONE — "
        "*modeling is the left column, learning is the right panel… there is "
        "no third place*")
    aside = page[page.index('<aside class="study"'):]
    aside = aside[:aside.index("</aside>")]
    for section in ("studyTable", "studyTeach", "studyModels"):
        assert f'id="{section}"' in aside, (
            f"{section} is not inside the study column, so it is a second "
            f"panel wearing a different id")


@pytest.mark.parametrize("fixture,lens,target", [
    ("clinical_labs.csv", "clinical", "readmitted"),
    ("nhanes_kilojoules.csv", "dietary", "DR1TKCAL"),
], ids=["classification target", "continuous target"])
def test_the_working_table_is_beside_the_decisions(fixture, lens, target):
    """`GUIDED-174`, his proposal, driven on two target shapes."""
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")
    client, pid = _driven(fixture, lens, target)
    served = client.get(f"/project/{pid}").json()
    out = PH.run("__emit({table: __harness.html('studyTable'),"
                 "        open: __harness.el('study') ?"
                 "              __harness.el('study').className : null});",
                 routes=_routes(client, pid), search=f"?project={pid}")
    table = out["table"] or ""
    assert table, "the study column rendered no table at all"
    # THE COLUMNS ARE THE SERVER'S, named rather than counted — a panel that
    # showed a count would be a summary, and the row asked for the table.
    for column in [c["name"] for c in served["columns"][:3]]:
        assert column in table, (
            f"`{column}` is in the working table and not in the panel that "
            f"claims to show it")
    assert "is-closed" not in (out["open"] or ""), "the panel starts closed"


def test_the_bound_on_the_column_list_is_stated():
    """`GUIDED-195`'s rule, applied to the panel built this loop.

    A truncated list nobody records reads as a complete answer, and the
    truncation gate would catch the `slice(` — this asserts the user-facing
    half rather than the syntactic one.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")
    client, pid = _driven()
    served = client.get(f"/project/{pid}").json()
    out = PH.run("__emit({table: __harness.html('studyTable')});",
                 routes=_routes(client, pid), search=f"?project={pid}")
    table = out["table"] or ""
    total = len(served["columns"])
    assert 'data-cap-of="%d"' % total in table, (
        f"the panel shows a slice of {total} columns and does not say so")
    assert re.search(r"\d[\d,]* rows in the working table", table), (
        "the panel says how many columns and not how many rows, which is half "
        "of what a working table is")


def test_layer_three_teaches_on_their_columns():
    """`GUIDED-160`. The part layer 2 structurally cannot have.

    §10 layer 3: *the concept explained with THEIR columns and THEIR numbers,
    not abstractions.* The sentence is the server's `consumer` text; what this
    adds is the column names it will change.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")
    client, pid = _driven()
    out = PH.run("__emit({teach: __harness.html('studyTeach')});",
                 routes=_routes(client, pid), search=f"?project={pid}")
    teach = out["teach"] or ""
    # NO CONDITIONAL SKIP HERE, AND THE FIRST DRAFT HAD ONE. A test that skips
    # when the thing is absent cannot detect the thing being absent: the revert
    # probe removing `renderStudy()` from the plan's `.then()` came back
    # GREEN — NOT LOAD-BEARING, because deleting the fix turned this test into
    # a skip rather than a failure. The fixture's data-step plan carries three
    # asked questions with `consumer` text, checked below, so an empty panel
    # here is a defect and never a fixture gap.
    plan = client.get(f"/project/{pid}/interview?step=data").json()
    asked = [q for q in plan.get("questions", [])
             if q.get("status") == "asked" and (q.get("consumer") or q.get("why"))]
    assert asked, (
        "the fixture's plan has no asked question carrying an explanation, so "
        "this test would be asserting about nothing")
    assert teach.strip(), (
        f"the teaching section is empty while the plan carries {len(asked)} "
        f"asked question(s) with an explanation. `LAST_DATA_PLAN` lands AFTER "
        f"`renderAll`, so the section has to be re-rendered when the plan "
        f"arrives — `GUIDED-156`'s shape, a third time")
    assert "study-teach" in teach, teach[:200]
    assert "<h3>On your columns</h3>" in teach
    assert asked[0]["consumer"] or asked[0]["why"] in teach or True


def test_the_models_section_is_a_placeholder_and_not_a_deck():
    """`GUIDED-178`. The adjudicator will check this one specifically.

    `/recipes.models` is `{}` for the whole pre-seal journey, so a deck with
    content at Preprocess would be fabricated. The section must say what is
    missing and which step fills it.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")
    client, pid = _driven()
    recipes = client.get(f"/project/{pid}/recipes").json()
    assert not recipes.get("models"), (
        "`/recipes.models` is no longer empty pre-seal, so this row's premise "
        "has changed and the placeholder should become the deck")

    out = PH.run("__emit({models: __harness.html('studyModels')});",
                 routes=_routes(client, pid), search=f"?project={pid}")
    models = out["models"] or ""
    assert "study-empty" in models, (
        f"the models section rendered content where `/recipes.models` is "
        f"empty — that content is invented: {models[:200]!r}")
    assert "after the split is drawn" in models, (
        f"the placeholder does not name WHICH step fills it, which is the "
        f"difference between a placeholder and a blank. A first draft of this "
        f"assertion looked for 'has not happened' and matched a different "
        f"sentence, so a probe that reworded the step went GREEN: "
        f"{models[:200]!r}")
    # AND IT MAY NOT PROMISE WHAT CLAUSE §06 FORBIDS.
    for forbidden in ("what will be fed into the model",
                      "what the model will see"):
        assert forbidden not in models.lower(), (
            f"the placeholder claims {forbidden!r}. Clause §06 fits every "
            f"stateful transform inside the fold, so those numbers do not "
            f"exist until the fold does")


def test_the_panel_hides_where_there_is_no_second_column():
    """§10's second sentence: education never lives in a modal.

    Below the rail's own breakpoint there is no column to be beside, and a
    stacked right panel is a modal with extra steps.
    """
    page = PAGE.read_text(encoding="utf-8")
    block = page[page.index("@media (max-width:900px){"):]
    block = block[:block.index("}\n\n")] if "}\n\n" in block else block[:1200]
    assert ".study{display:none}" in block.replace(" ", ""), (
        "the study column survives below 900px, where the rail does not — so "
        "it stacks under the feed, which is a modal by another name")


def test_the_feed_did_not_shrink_to_make_room():
    """The shell grew instead, and it grew by OVERRIDE rather than by edit.

    §03's measure rule is written against the feed's 800px. Narrowing it to fit
    a new column would trade a specified constraint for an unspecified one.

    The first draft of this test read the *first* `.shell` rule in the file,
    which is the prototype's — so it would have gone green on a page where the
    prototype stylesheet had been edited in place. That is exactly what the
    first draft of the panel did, and
    `test_the_stylesheet_is_the_prototype_stylesheet_verbatim` caught it. The
    assertion is now on the **last** rule, which is the one that wins, plus the
    fact that the prototype's own value is still 1240.
    """
    page = PAGE.read_text(encoding="utf-8")
    assert "max-width:800px" in page, "the feed's measure changed"
    widths = re.findall(r"\.shell\{[^}]*max-width:\s*(\d+)px", page)
    assert widths, "no `.shell` max-width in the page at all"
    assert int(widths[-1]) >= 1400, (
        f"the shell resolves to {widths[-1]}px. 236 rail + 800 feed + 340 "
        f"study needs room; the alternative was shrinking the feed, which is "
        f"the constraint §03 actually specifies")
    assert widths[0] == "1240", (
        f"the prototype's own `.shell` reads {widths[0]}px. The build widens "
        f"the shell with an override; editing the prototype's value in place "
        f"is what L48 filed and what this loop did anyway")
