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
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

from pathlib import Path

import pytest

from turbotab import features as F

DATA = Path(__file__).resolve().parent / "sample_data"


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
