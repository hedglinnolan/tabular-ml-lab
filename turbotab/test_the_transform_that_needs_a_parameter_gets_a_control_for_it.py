"""`GUIDED-198` — six of eighteen transforms 400'd because the page never read `needs`.

## The reproduction, re-derived rather than quoted

Driven on `clinical_labs.csv` (target `readmitted`, classification) and on
`clinic_visits.csv` (target `hba1c`, regression), sending exactly what the page
sent — the preview with no `params` at all and the decision with a literal
`params: {}`:

    before   preview 200: 12/18   decision 200: 12/18     (both fixtures)

and the six that could not be satisfied were the same six on both, for the same
reasons, at both doors:

| transform | `needs` | the sentence the user got |
|---|---|---|
| `bin_fixed` | `edges` | Binning by supplied cut-points needs at least two edges… |
| `ordinal_declared` | `order` | Encoding in a stated order needs the order… |
| `bin_quantile` / `bin_uniform` / `bin_kmeans` | `n_bins` | …cannot be described yet: `n_bins` has not been supplied… |
| `pca` | `n_components` | …cannot be described yet: `n_components` has not been supplied… |

`n_bins`, `edges` and `n_components` appeared **zero** times in `index.html`.
`featPickerHTML` read `row.n_inputs`; nothing read `row.needs`.

    after    preview 200: 18/18   decision 200: 18/18     (both fixtures)

`test_every_transform_the_catalogue_offers_can_be_satisfied` re-derives both
numbers on every run, so the "before" is a record rather than a claim in a
report.

## What was fixed, and what was deliberately not

**The server describes the parameter; the page renders what it is told.** `needs`
grew from a tuple of names into `features.Parameter` — `name`, a `kind` the page
knows how to render, a `label`, the `because` saying why the app cannot derive
the value, and the bound. A `<select>` offering 2 to 10 bins written into
`index.html` would have been a second copy of a rule that lives in
`features.py`, which is this project's most-repeated defect.

For `edges` and `order` the `because` is `_compute`'s own `FeatureRefusal`,
hoisted to `features.EDGES_REFUSAL` and `features.ORDER_REFUSAL` so there is one
copy — the sentence a user reads *before* filling the control and the sentence
they read if they do not are the same words.
`test_the_control_carries_the_engines_own_refusal_rather_than_a_paraphrase`
pins that.

**The buttons are NOT gated on the parameters, and that is a decision.** Gating
them would have made the row's own refusal unreachable, and that refusal is
`GUIDED-176`'s evidence — *Show me what it does* on `bin_fixed` with no
cut-points is a 400 that has to land at the control, and
`test_show_me_what_it_does_says_why_it_cannot.py` drives it. The shelf is never
shortened: the press stays available, it now CAN succeed, and where it cannot the
server still says why.

## `order`, which is the one that tests whether the rule generalizes

The legitimate values of `order` are the CHOSEN COLUMN's distinct levels, so
they do not exist until a column is picked. The brief allows two answers — serve
the levels, or state the precondition and leave it refusing with a failing test
naming the missing consumer. **This serves the levels.** A control that states a
precondition nothing can meet has moved the defect, not fixed it, and the levels
are one `unique()` away: `/features` now carries `column_levels`, and
`features.column_levels` gives every column either its levels or the sentence
saying why an order cannot be stated over it.

Two consequences, both driven below:

* The picker for `ordinal_declared` used to offer the NUMERIC columns only —
  too narrow for a transform that encodes categories, and unsatisfiable on every
  column it listed. It now offers every column, which is a widening.
* A column with 96 distinct values, or with one, renders the server's reason in
  place of a control rather than 96 empty dropdowns.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbotab import features as F

DATA = Path(__file__).resolve().parent / "sample_data"

#: NOT COVERED, said out loud (`GUIDED-097`, `LOOP.md` §10 rule 4).
SHAPES_NOT_COVERED = (
    "Whether any of it is ON SCREEN. `pageharness.py` has no layout and says "
    "so in its own docstring; every claim here is that a control was rendered "
    "into an addressable node and that a press carried what it held.",
    "A MULTICLASS target. Both fixtures are two-shaped — `readmitted` is 0/1 "
    "and `hba1c` is continuous — and `multiclass_stage.csv` is not driven. The "
    "features step does not read the target's shape, which is why it was not "
    "the axis chosen, but it is a shape this file does not cover.",
    "A table with NO orderable column at all. Both fixtures have at least one, "
    "so the branch where `ordinal_declared` can be offered a column and none "
    "of them has a statable order is reasoned about and not driven.",
    "The browser's own enforcement of `min` on the number input. The attribute "
    "is asserted to be on the control and the server's refusal is asserted to "
    "fire; what a real browser does with it is outside a DOM shim.",
    "Typing into the `edges` box character by character. The page reads it on "
    "`change`, so this drives `change`; a per-keystroke `input` handler is not "
    "built and is not claimed.",
)

#: `GUIDED-097`. Two fixtures of different target shape.
FIXTURES = {"clinical_labs.csv": "readmitted", "clinic_visits.csv": "hba1c"}

#: The six the row is about, ordered HARDEST FIRST — by what is most likely to
#: break the abstraction rather than by effort. `ordinal_declared` is first
#: because its parameter is defined over a column that has not been chosen yet,
#: which is the only one of the four kinds that cannot be described statically.
PARAMETERIZED = ["ordinal_declared", "bin_fixed", "pca",
                 "bin_kmeans", "bin_uniform", "bin_quantile"]

#: What a user types into a `numbers` box. Not derivable from the descriptor —
#: the descriptor says how many and in what order, never which values, because
#: which cut-points matter is the researcher's knowledge and is the whole reason
#: the parameter exists.
TYPED_EDGES = "10, 20, 30"


# ── the drive ────────────────────────────────────────────────────────────────

#: EVERY PRESS AND EVERY FILL IS BUILT FROM THE RENDER — trap #3.
#:
#: A hand-written `{'data-feat-param': 'pca', 'data-feat-pname': 'n_components'}`
#: would supply the attributes whose absence IS this defect, and the revert probe
#: would report `GREEN — NOT LOAD-BEARING`. Every control below is found by
#: scanning the row the page rendered, and every attribute dispatched is read off
#: that control. The one value not taken from the render is the text typed into a
#: `numbers` box, which is what a user supplies and the page cannot.
_HELPERS = r"""
function tags(html, name){
  var re = new RegExp("<" + name + "\\b([^>]*)>", "g"), m, out = [];
  while ((m = re.exec(html))){
    var attrs = {}, a = /([a-zA-Z-]+)="([^"]*)"/g, k;
    while ((k = a.exec(m[1]))) attrs[k[1]] = k[2];
    out.push(attrs);
  }
  return out;
}
function selects(html){
  var re = /<select\b([^>]*)>([\s\S]*?)<\/select>/g, m, out = [];
  while ((m = re.exec(html))){
    var attrs = {}, a = /([a-zA-Z-]+)="([^"]*)"/g, k;
    while ((k = a.exec(m[1]))) attrs[k[1]] = k[2];
    var opts = [], o = /<option value="([^"]*)"/g, p;
    while ((p = o.exec(m[2]))) opts.push(p[1]);
    out.push({attrs: attrs, options: opts});
  }
  return out;
}
async function settle(n){
  for (var i = 0; i < (n || 6); i++) await new Promise(function(r){ setTimeout(r, 0); });
}
function fill(attrs, value){
  var el = __harness.target(attrs);
  el.value = value;
  __harness.dispatch('change', el);
}
"""

#: Pick the columns, fill whatever parameter controls the row rendered, press
#: both buttons, and report the requests. `%(col)s` is `null` to take the
#: picker's first real option, or a column name a user would choose.
_DRIVE = r"""
var KEY = %(key)s, WANT = %(col)s, TYPED = %(typed)s;
function row(){ return __harness.html('featbody-' + KEY) || ''; }

selects(row()).filter(function(s){ return s.attrs['data-feat-col']; })
  .forEach(function(s){
    var opts = s.options.filter(function(v){ return v !== ''; });
    var slot = Number(s.attrs['data-feat-slot'] || 0);
    fill(s.attrs, WANT === null ? opts[slot] : WANT);
  });
await settle(6);

var filled = [];
selects(row()).filter(function(s){ return s.attrs['data-feat-param']; })
  .forEach(function(s){
    var opts = s.options.filter(function(v){ return v !== ''; });
    var slot = Number(s.attrs['data-feat-pslot'] || 0);
    fill(s.attrs, opts[slot]);
    filled.push([s.attrs['data-feat-pname'], opts[slot]]);
  });
await settle(6);
tags(row(), 'input').filter(function(a){ return a['data-feat-param']; })
  .forEach(function(a){
    /* The BOUND comes off the control the page rendered, which the server put
       there. A literal 4 here would be this test inventing the number the
       finding is about. */
    var v = a['data-feat-pkind'] === 'integer' ? a.min : TYPED;
    fill(a, v);
    filled.push([a['data-feat-pname'], v]);
  });
await settle(6);

var before = row();
var pv = tags(before, 'button').filter(function(b){ return b['data-feat-preview']; })[0];
if (pv) __harness.dispatch('click', __harness.target(pv));
await settle(8);
var add = tags(row(), 'button').filter(function(b){ return b['data-feat-add']; })[0];
if (add) __harness.dispatch('click', __harness.target(add));
await settle(8);

__emit({row: before, filled: filled,
        params: selects(before).filter(function(s){ return s.attrs['data-feat-param']; })
                  .map(function(s){ return {attrs: s.attrs, options: s.options}; }),
        inputs: tags(before, 'input').filter(function(a){ return a['data-feat-param']; }),
        pickers: selects(before).filter(function(s){ return s.attrs['data-feat-col']; })
                   .map(function(s){ return {attrs: s.attrs, options: s.options}; }),
        preview_button: pv || null, add_button: add || null,
        calls: __harness.calls().filter(function(c){
          return c.path.indexOf('feature/preview') !== -1 ||
                 (c.method === 'POST' && c.path.indexOf('/decision') !== -1); })});
"""


def _project(fixture: str, target: str):
    """A project driven as far as the Features step, over HTTP."""
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    live = client.post(f"/project/{pid}/decision", json={
        "kind": "set_target", "payload": {"column": target}})
    assert live.status_code == 200, (
        f"{fixture} no longer accepts `{target}` as a target "
        f"({live.status_code}), so nothing below is driving the Features step")
    return client, pid


def _routes(client, pid):
    """Every response one render of this page asks for."""
    out = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "capabilities", "features",
                 "recipes", "preprocess", "figures", "draft", "manuscript",
                 "models", "training", "instability", "explain", "sensitivity",
                 "evidence/plausibility", "evidence/missingness"):
        resp = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = (resp.json() if resp.status_code == 200
                                         else {})
    # The press posts a decision. The RESPONSE is not what is asserted — the
    # request is — but the page renders whatever comes back, so it is answered
    # with a real project rather than left to render `{}`.
    out[f"POST /project/{pid}/decision"] = out[f"/project/{pid}"]
    return out


def _drive(client, pid, key, column=None, typed=TYPED_EDGES):
    from turbotab import pageharness as PH

    return PH.run(
        _HELPERS + (_DRIVE % {"key": json.dumps(key),
                              "col": json.dumps(column),
                              "typed": json.dumps(typed)}),
        routes=_routes(client, pid), search=f"?project={pid}")


def _orderable(features):
    """The columns `/features` says an order can be stated over."""
    return [r for r in features["column_levels"] if r.get("levels")]


def _params_from_descriptors(features, row, column_levels_row):
    """One legitimate value per served descriptor, built from the SERVED form.

    `minimum` comes off the descriptor and the levels come off `column_levels`,
    so the only thing this test writes is the `numbers` list — which is what a
    user types and the app has no way to derive.
    """
    out = {}
    for param in row["needs"]:
        if param["from_column"]:
            out[param["name"]] = list(column_levels_row["levels"])
        elif param["kind"] == "integer":
            out[param["name"]] = int(param["minimum"])
        else:
            out[param["name"]] = [float(v) for v in TYPED_EDGES.split(",")]
    return out


# ── the counts ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", sorted(FIXTURES),
                         ids=["classification target", "regression target"])
def test_every_transform_the_catalogue_offers_can_be_satisfied(fixture, capsys):
    """Eighteen of eighteen, at the preview and at the decision — and the six.

    Both halves are asserted, because *twelve worked* was true before the fix
    too. The reproduction is re-derived here so the number in the report is a
    record: with what the page used to send, exactly six refuse, and they are
    the six the row names.
    """
    client, pid = _project(fixture, FIXTURES[fixture])
    served = client.get(f"/project/{pid}/features").json()
    rows = [(r, "add_feature") for r in served["row_local"]] + \
           [(r, "defer_feature") for r in served["deferred"]]
    assert len(rows) == 18, (
        f"the catalogue serves {len(rows)} transforms, not 18; the counts this "
        f"file reports are against a shape that has changed")

    numeric = served["numeric_columns"]
    orderable = _orderable(served)
    assert orderable, (
        f"{fixture} serves no column an order can be stated over, so "
        f"`ordinal_declared` cannot be satisfied on it and this fixture cannot "
        f"carry the claim below")

    before_ok, after_ok, refused = 0, 0, {}
    for row, kind in rows:
        by_column = [p for p in row["needs"] if p["from_column"]]
        columns = ([orderable[0]["column"]] if by_column
                   else numeric[:row.get("n_inputs", 1)])
        after = _params_from_descriptors(served, row, orderable[0])

        for params, tally in ((None, "before"), (after, "after")):
            query = {"transform": row["key"], "columns": ",".join(columns)}
            if params:
                query["params"] = json.dumps(params)
            preview = client.get(f"/project/{pid}/feature/preview", params=query)
            fresh_client, fresh_pid = _project(fixture, FIXTURES[fixture])
            decision = fresh_client.post(f"/project/{fresh_pid}/decision", json={
                "kind": kind, "subject": row["key"],
                "payload": {"transform": row["key"], "columns": columns,
                            "params": params or {}}})
            both = preview.status_code == 200 and decision.status_code == 200
            if tally == "before":
                before_ok += both
                if not both:
                    refused[row["key"]] = str(preview.json().get("detail"))
            else:
                after_ok += both
                assert both, (
                    f"`{row['key']}` still cannot be satisfied with the "
                    f"parameters the server itself describes: preview "
                    f"{preview.status_code} {preview.text[:200]}; decision "
                    f"{decision.status_code} {decision.text[:200]}")

    with capsys.disabled():
        print(f"\n  ── GUIDED-198 · {fixture} ({FIXTURES[fixture]}) ──")
        print(f"  satisfiable with what the page USED to send   {before_ok}/18")
        print(f"  satisfiable with the served descriptors       {after_ok}/18")
        for key in sorted(refused):
            print(f"      {key}: {refused[key][:96]}")

    assert after_ok == 18, f"{after_ok} of 18"
    assert before_ok == 12, (
        f"{before_ok} of 18 transforms used to be satisfiable with no "
        f"parameters, not 12 — the reproduction this file records has moved")
    assert set(refused) == {"bin_fixed", "ordinal_declared", "bin_quantile",
                            "bin_uniform", "bin_kmeans", "pca"}, sorted(refused)


# ── the page ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", sorted(FIXTURES),
                         ids=["classification target", "regression target"])
@pytest.mark.parametrize("key", PARAMETERIZED)
def test_the_press_carries_the_parameter_the_row_rendered_a_control_for(
        key, fixture):
    """Fill the row's `data-feat-param` control, press, and replay the request.

    The press is on `data-feat-preview` and `data-feat-add`, both read off the
    buttons the page emitted. What is asserted is the REQUEST — the URL the
    preview built and the body the decision posted — and then that request is
    replayed against the real API, so the claim is *the server accepts what the
    page sends* rather than *the page contains a string*.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _project(fixture, FIXTURES[fixture])
    served = client.get(f"/project/{pid}/features").json()
    row = next((r for r in served["row_local"] + served["deferred"]
                if r["key"] == key), None)
    assert row is not None and row["needs"], (
        f"`{key}` is no longer a transform this catalogue says needs a "
        f"parameter, so this drive is pressing something else")

    # `ordinal_declared` is driven on a column an order CAN be stated over,
    # because a user picking one is the case the fix is about. Which column is
    # the server's answer, not this file's.
    wanted = (_orderable(served)[0]["column"]
              if any(p["from_column"] for p in row["needs"]) else None)
    out = _drive(client, pid, key, column=wanted)

    assert out["preview_button"] and out["add_button"], (
        f"the `{key}` row rendered no press after a column was picked, so "
        f"nothing was driven: {out['row'][:400]!r}")
    # The two presses are this row's own, read off the buttons the page emitted.
    assert out["preview_button"]["data-feat-preview"] == key, out["preview_button"]
    assert out["add_button"]["data-feat-add"] == key, out["add_button"]

    controls = out["params"] + out["inputs"]
    assert controls, (
        f"the `{key}` row rendered no control for {[p['name'] for p in row['needs']]}"
        f" — the page is still not reading `needs`, which is the finding. Row: "
        f"{out['row'][:600]!r}")
    assert out["filled"], (
        f"the `{key}` row's controls were found and none of them took a value: "
        f"{controls!r}")
    for control in controls:
        attrs = control["attrs"] if "attrs" in control else control
        assert attrs["data-feat-param"] == key, (
            f"a parameter control in the `{key}` row is addressed to "
            f"`{attrs.get('data-feat-param')}`, so its value would be held "
            f"against a different transform: {attrs!r}")
        assert attrs["data-feat-pname"] in {p["name"] for p in row["needs"]}, attrs
        assert attrs["data-feat-pkind"] in {"integer", "numbers", "levels"}, attrs

    # EVERY FIELD OF THE DESCRIPTOR HAS A READER, pinned rather than assumed.
    # `/features` is one of the payloads `fieldsweep` declares NOT SWEPT, so a
    # field the server composes and the page drops would be invisible there.
    for param in row["needs"]:
        assert param["because"][:50] in out["row"], (
            f"the row renders no reason for `{param['name']}`, so the control "
            f"arrives as a demand rather than as a question")
        assert param["label"] in out["row"], (
            f"the row renders no label for `{param['name']}`: {out['row'][:600]!r}")
        if param["kind"] == "integer":
            assert any(str(int(param["minimum"])) == a.get("min")
                       for a in out["inputs"]), (
                f"the control publishes no `min`, so the bound the server "
                f"states reaches nobody: {out['inputs']!r}")
        if param["kind"] == "numbers":
            assert any(a.get("placeholder") == param["hint"]
                       for a in out["inputs"]), (
                f"the served `hint` is not on the control, so the only "
                f"statement of the format is one the page would have to "
                f"invent: {out['inputs']!r}")

    gets = [c for c in out["calls"] if c["method"] == "GET"]
    posts = [c for c in out["calls"] if c["method"] == "POST"]
    assert gets and posts, out["calls"]

    for param in row["needs"]:
        assert f"{param['name']}" in gets[0]["path"], (
            f"the preview URL carries no `{param['name']}`, so the press asks "
            f"the server a question it has already refused: {gets[0]['path']}")
        assert param["name"] in (posts[0]["body"]["payload"]["params"] or {}), (
            f"the decision posted no `{param['name']}`: "
            f"{posts[0]['body']['payload']}")

    # AND THE SERVER TAKES IT. Replayed against the real API rather than against
    # the harness's canned routes, which answer 200 to anything.
    replay_get = client.get(gets[0]["path"])
    assert replay_get.status_code == 200, (
        f"the preview the page built is still refused: "
        f"{replay_get.status_code} {replay_get.text[:300]}")
    fresh_client, fresh_pid = _project(fixture, FIXTURES[fixture])
    replay_post = fresh_client.post(f"/project/{fresh_pid}/decision",
                                    json=posts[0]["body"])
    assert replay_post.status_code == 200, (
        f"the decision the page posted is still refused: "
        f"{replay_post.status_code} {replay_post.text[:300]}")


@pytest.mark.parametrize("fixture", sorted(FIXTURES),
                         ids=["classification target", "regression target"])
def test_the_order_offered_is_the_chosen_columns_own_levels(fixture):
    """The `order` control is the column's levels, not a list written in the page.

    Every option in every `data-feat-pslot` select is compared against
    `/features`'s `column_levels` for the column that was picked, because a page
    holding its own idea of what the levels are would be the second copy this
    whole change exists to avoid.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _project(fixture, FIXTURES[fixture])
    served = client.get(f"/project/{pid}/features").json()
    entry = _orderable(served)[0]

    out = _drive(client, pid, "ordinal_declared", column=entry["column"])
    controls = out["params"]
    assert len(controls) == len(entry["levels"]), (
        f"`{entry['column']}` has {len(entry['levels'])} levels and the row "
        f"rendered {len(controls)} position(s) to order them into")
    for i, control in enumerate(controls):
        assert control["attrs"]["data-feat-pname"] == "order"
        assert int(control["attrs"]["data-feat-pslot"]) == i
        offered = [v for v in control["options"] if v != ""]
        assert offered == list(entry["levels"]), (
            f"position {i + 1} offers {offered}, and the server says the "
            f"levels of `{entry['column']}` are {entry['levels']}")

    posted = [c for c in out["calls"] if c["method"] == "POST"]
    assert posted, "the order was filled and nothing was posted"
    assert posted[0]["body"]["payload"]["params"]["order"] == list(entry["levels"]), (
        f"the order that reached the wire is not the one the controls held: "
        f"{posted[0]['body']['payload']['params']}")


@pytest.mark.parametrize("fixture", sorted(FIXTURES),
                         ids=["classification target", "regression target"])
def test_a_column_with_no_statable_order_says_so_instead_of_offering_a_control(
        fixture):
    """The refusal branch, which is what keeps the widened picker honest.

    The picker now offers `ordinal_declared` every column, so it offers columns
    an order cannot be stated over — an identifier, or a constant. Those render
    the server's sentence in place of the control, which is *state the
    precondition* where the precondition genuinely cannot be met, sitting beside
    the case where it can.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _project(fixture, FIXTURES[fixture])
    served = client.get(f"/project/{pid}/features").json()
    unorderable = [r for r in served["column_levels"] if r.get("refusal")]
    assert unorderable, (
        f"{fixture} serves no column an order cannot be stated over, so this "
        f"branch is not reachable on it and the assertion below is vacuous")
    entry = unorderable[0]

    out = _drive(client, pid, "ordinal_declared", column=entry["column"])
    assert not out["params"] and not out["inputs"], (
        f"`{entry['column']}` has {entry['n_levels']} distinct values and the "
        f"row still offered a control to order them into")
    assert entry["refusal"][:60] in out["row"], (
        f"the row says nothing about why `{entry['column']}` has no statable "
        f"order; a control that is simply absent is silence where a reason "
        f"belongs. Rendered: {out['row'][-600:]!r}")


@pytest.mark.parametrize("fixture", sorted(FIXTURES),
                         ids=["classification target", "regression target"])
def test_the_picker_for_an_order_offers_more_columns_than_the_numeric_ones(
        fixture):
    """The widening. `ordinal_declared` encodes categories and was offered none.

    `data-feat-col` used to be filled from `numeric_columns` for every entry in
    the catalogue, so the one transform whose parameter is a list of category
    levels could only be pointed at columns that are not categories.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _project(fixture, FIXTURES[fixture])
    served = client.get(f"/project/{pid}/features").json()
    out = _drive(client, pid, "ordinal_declared",
                 column=_orderable(served)[0]["column"])

    assert len(out["pickers"]) == 1, out["pickers"]
    picker = out["pickers"][0]
    assert picker["attrs"]["data-feat-col"] == "ordinal_declared", picker["attrs"]
    offered = [v for v in picker["options"] if v != ""]
    assert offered == [r["column"] for r in served["column_levels"]], (
        f"the picker is not offering the columns the server served levels for: "
        f"{offered}")
    categorical = [c for c in offered if c not in served["numeric_columns"]]
    assert categorical, (
        "the picker still offers only numeric columns, so the transform that "
        "encodes categories is pointed at nothing that is one")


# ── the descriptors themselves ───────────────────────────────────────────────

def test_the_control_carries_the_engines_own_refusal_rather_than_a_paraphrase():
    """One copy of each sentence, asserted against what `_compute` actually raises.

    A `because` that merely *reads like* the refusal is the same defect the
    `because` field was added to prevent one level up: two statements of one
    rule, free to drift, with nothing that notices.
    """
    import pandas as pd

    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["a", "b", "a"]})

    with pytest.raises(F.FeatureRefusal) as caught:
        F.preview(frame, "bin_fixed", ["x"], {})
    assert str(caught.value) == F.PARAMETERS["edges"].because == F.EDGES_REFUSAL

    with pytest.raises(F.FeatureRefusal) as caught:
        F.preview(frame, "ordinal_declared", ["g"], {})
    assert str(caught.value) == F.PARAMETERS["order"].because == F.ORDER_REFUSAL


def test_every_name_in_needs_resolves_to_a_descriptor_a_page_can_render():
    """Trap #3, applied to the catalogue: a `needs` entry stands for a control.

    A transform declaring a parameter with no descriptor would ship a button
    that cannot be satisfied, which is this row restated. The `kind` is checked
    against the vocabulary the page branches on, because a descriptor the page
    has no arm for renders nothing at all.
    """
    renderable = {"integer", "numbers", "levels"}
    page = (Path(__file__).resolve().parent / "web" / "index.html"
            ).read_text(encoding="utf-8")
    for key, transform in sorted(F.CATALOGUE.items()):
        for name in transform.needs:
            param = F.PARAMETERS.get(name)
            assert param is not None, f"`{key}` needs `{name}` and nothing describes it"
            assert param.kind in renderable, f"{key}/{name}: {param.kind}"
            assert param.because.strip(), f"{key}/{name} has no reason"
            assert f'"{param.kind}"' in page or f"'{param.kind}'" in page, (
                f"the page has no arm for a `{param.kind}` parameter, so "
                f"`{key}` renders a reason and no control")
    assert sorted(F.PARAMETERS) == sorted(
        {n for t in F.CATALOGUE.values() for n in t.needs}), (
        "a descriptor exists for a parameter no transform needs, or the other "
        "way round")


def test_a_bound_the_control_publishes_is_a_bound_the_engine_keeps():
    """The other half of the fix, and it only became reachable with the first.

    Once the page can send a parameter it can send a wrong one, and every bound
    the descriptor publishes is enforced where the rule lives. `pd.cut` is the
    sharp one: `bins=[30, 10]` raises a bare `ValueError` no handler catches, so
    two cut-points typed backwards would have been a 500 where a sentence
    belongs.
    """
    import pandas as pd

    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", "b", "a", "c"]})

    with pytest.raises(F.FeatureRefusal) as caught:
        F.declare("bin_quantile", ["x"], {"n_bins": 1})
    assert "at least 2" in str(caught.value)

    with pytest.raises(F.FeatureRefusal) as caught:
        F.declare("pca", ["x"], {"n_components": 0})
    assert "at least 1" in str(caught.value)

    with pytest.raises(F.FeatureRefusal) as caught:
        F.declare("bin_kmeans", ["x"], {"n_bins": 2.5})
    assert "whole number" in str(caught.value)

    with pytest.raises(F.FeatureRefusal) as caught:
        F.preview(frame, "bin_fixed", ["x"], {"edges": [30.0, 10.0]})
    assert "increase" in str(caught.value)

    with pytest.raises(F.FeatureRefusal) as caught:
        F.preview(frame, "bin_fixed", ["x"], {"edges": ["a", "b"]})
    assert "not one" in str(caught.value)

    with pytest.raises(F.FeatureRefusal) as caught:
        F.preview(frame, "ordinal_declared", ["g"], {"order": ["a", "b", "a"]})
    assert "more than once" in str(caught.value)

    # AND IT LETS THROUGH WHAT IT SHOULD. A validator nothing satisfies is the
    # same defect wearing the fix's clothes.
    assert F.declare("bin_quantile", ["x"], {"n_bins": 2})["params"]["n_bins"] == 2
    assert F.preview(frame, "bin_fixed", ["x"],
                     {"edges": [0.0, 2.0, 5.0]})["n_rows"] == 4
    assert F.preview(frame, "ordinal_declared", ["g"],
                     {"order": ["a", "b", "c"]})["n_rows"] == 4


def test_the_absence_of_a_parameter_still_refuses_in_the_words_guided_175_settled():
    """The negative control for the check above: absence is not this one's job.

    `_check_params` skips a parameter that was not supplied, so `_compute` and
    `_sentence` keep answering that case. Two sentences for one condition is the
    thing `GUIDED-175` decided against, and a validator that ran first would
    have quietly replaced both.
    """
    import pandas as pd

    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(F.FeatureRefusal) as caught:
        F.declare("bin_kmeans", ["x"], {})
    assert "cannot be described yet" in str(caught.value)
    assert "`n_bins`" in str(caught.value)
    assert "`x`" in str(caught.value), (
        "the refusal lost the column the user HAD chosen, which is the second "
        "half of `GUIDED-175`")


def test_column_levels_answers_for_every_column_and_never_guesses_an_order():
    """The service behind `order`, on a frame built to hit all three branches.

    Sorted alphabetically and deliberately not semantically: the premise of
    `ordinal_declared` is that the app does NOT know the order, so a list the
    app arranged would be an assertion where a list belongs.
    """
    import pandas as pd

    frame = pd.DataFrame({
        "grade": ["severe", "mild", "moderate", "mild"],
        "constant": ["x", "x", "x", "x"],
        "ident": [f"p{i}" for i in range(4)],
        "outcome": [0, 1, 0, 1],
    })
    rows = {r["column"]: r for r in F.column_levels(frame, exclude=["outcome"])}
    assert set(rows) == {"grade", "constant", "ident"}, sorted(rows)

    assert rows["grade"]["levels"] == ["mild", "moderate", "severe"]
    assert "refusal" not in rows["grade"]
    assert rows["constant"]["n_levels"] == 1
    assert "no order to state" in rows["constant"]["refusal"]
    assert "levels" not in rows["constant"]

    wide = pd.DataFrame({"c": [str(i) for i in range(F.ORDER_MAX_LEVELS + 1)]})
    only = F.column_levels(wide)[0]
    assert "levels" not in only
    assert "how common each value is" in only["refusal"], (
        "the refusal for too many levels names no route, so a user with a "
        "40-level column is told no and nothing else")
