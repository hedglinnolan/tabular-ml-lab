"""`GUIDED-177` · the feature-selection preview is evidence for the method chosen.

## What was measured before the fix

Driven through the real API on `clinic_visits.csv` with both of its targets:
**five methods are offered** — `mutual_info`, `lasso`, `rfe`, `univariate`,
`stability` — and all five produced **one** ranking, with **one** measure
string. `selection.evidence` took no method argument at all; it computed
`abs(pearson)` unconditionally and labeled every row *absolute correlation
with the outcome*, under a recorded sentence reading *the top 5 features by
mutual information with `glucose`*.

Second half of the same measurement, and it is the worse one: on the
**classification** shape — `outcome`, a string label — every one of the seven
numeric candidates came back `score: None` with the measure *not numeric — not
ranked here*. The columns are floats. It was the OUTCOME that a correlation
could not read, and the sentence blamed the feature.

After: **six distinct measure strings across the six requests** on each shape,
and the two methods that have a per-column statistic compute it with the same
sklearn scorer `pipeline_plan._selector` fits inside the fold.

## The line this file asserts

A method's preview is possible when its score is a property of ONE column
against the outcome. `mutual_info` and `univariate` are; `lasso`, `rfe` and
`stability` rank by what survives a fit over all candidates at once, and
getting that means running the selector, which is a selection. **Those three
stay offered** — the shelf is never shortened — and say what was not computed
instead of borrowing another method's number.

## Fixture shapes — `GUIDED-097`

`TARGET_SHAPES` runs the load-bearing claims against a continuous target, a
binary-string target and a three-level target. `SHAPES_NOT_COVERED` names the
rest, with the reason, in the file rather than in a report.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, pageharness as H     # noqa: E402
from turbotab import selection as _sel         # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: `GUIDED-097`. Three target shapes, and the multiclass one is here because a
#: three-level outcome is the case where `mutual_info_classif` and `f_classif`
#: diverge most from anything Pearson could report.
TARGET_SHAPES = {
    "continuous": ("clinic_visits.csv", "hba1c"),
    "binary_string": ("clinic_visits.csv", "outcome"),
    "multiclass": ("multiclass_stage.csv", "disease_stage"),
}

#: And the ones this file does NOT cover.
SHAPES_NOT_COVERED = {
    "binary_numeric": (
        "`leaky_sepsis.csv` has a 0/1 target and no missing values. It is the "
        "shape where the OLD behavior was least visibly wrong — a correlation "
        "against 0/1 is a real if crude statistic — so it is the weakest of "
        "the four for this claim, and it is the one dropped."),
    "survival": (
        "No fixture carries a time-and-event pair, and neither sklearn scorer "
        "used here accepts one. A survival outcome would need a third branch "
        "in `_preview_measure`, not a third fixture."),
    "wide": (
        "`metabolomics_untargeted.csv` has 396 numeric columns. The preview "
        "caps at `top=12` rows but scores every candidate first, so the wide "
        "case is a timing question rather than a correctness one, and it is "
        "not driven here."),
}

#: The one string the old code put on every row of every method.
PEARSON = "absolute correlation with the outcome"

#: What `/features` offers, read from the module rather than restated.
OFFERED = sorted(_sel.METHODS)

#: The three that rank by a fit rather than by a per-column statistic.
NO_SCORE = sorted(_sel._NO_PER_FEATURE_SCORE)

#: The two that have a per-column statistic and therefore a real preview.
PREVIEWABLE = sorted(set(OFFERED) - set(NO_SCORE))


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _sealed(client, shape):
    fixture, target = TARGET_SHAPES[shape]
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])
        return r

    decide("set_target", column=target)
    decide("set_purpose", answer="prediction")
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=0.25)
    return pid, decide


def _evidence(client, pid, method=None):
    url = f"/project/{pid}/selection/evidence"
    if method:
        url += f"?method={method}"
    r = client.get(url)
    assert r.status_code == 200, (method, r.text[:250])
    return r.json()


# ── 1 · the defect itself: one measure for five methods ──────────────────────

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES), ids=sorted(TARGET_SHAPES))
def test_no_method_is_previewed_with_the_pearson_ranking_it_did_not_ask_for(
        client, shape):
    """**The finding, stated as a condition on what the route serves.**

    Every offered method is requested and the served table is read: no
    method's heading may be the correlation string, and the five headings may
    not collapse to one. Before the fix all five were `PEARSON` and the count
    of distinct headings was 1.
    """
    pid, _ = _sealed(client, shape)
    offered = [m["key"] for m in
               client.get(f"/project/{pid}/features").json()["selection_methods"]]
    assert sorted(offered) == OFFERED, (
        "the route offers a different set of methods than this file sweeps",
        offered)
    assert len(offered) == 5, offered                                # control

    measures = {}
    for key in offered:
        body = _evidence(client, pid, key)
        measures[key] = body["measure"]
        # THE FINDING ITSELF, FIRST, so a revert probe fails here rather than
        # on the bookkeeping assertion below it.
        assert PEARSON not in body["measure"], (
            f"'{key}' is previewed with the measure '{PEARSON}', which is a "
            f"different method's statistic wearing this one's question")
        for row in body["ranked"]:
            assert PEARSON not in (row["measure"] or ""), (
                f"'{key}' row {row['feature']!r} carries the correlation "
                f"measure")
        assert body["method"] == key, (
            "the response does not say which method it is evidence for")

    assert len(set(measures.values())) >= 3, (
        f"five methods produced {len(set(measures.values()))} distinct "
        f"measures: {measures}")


# ── 2 · the two that CAN be previewed compute their own statistic ────────────

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES), ids=sorted(TARGET_SHAPES))
@pytest.mark.parametrize("key", PREVIEWABLE)
def test_a_previewable_method_scores_with_the_estimator_the_fold_will_use(
        client, shape, key):
    """Not just a different label — a different NUMBER, from the scorer
    `pipeline_plan._selector` fits inside the fold.

    The note names the sklearn function by name, so a reader can check the
    claim against the module that does the selecting.
    """
    from turbotab import pipeline_plan

    pid, _ = _sealed(client, shape)
    body = _evidence(client, pid, key)
    assert body["is_ranked"] is True, body["measure"]
    scores = [r["score"] for r in body["ranked"] if r["score"] is not None]
    assert scores, (
        f"'{key}' produced no score at all on {shape}; a previewable method "
        f"that computes nothing is the defect in a new costume")
    assert all(s >= 0 for s in scores), scores

    fn = ("mutual_info" if key == "mutual_info" else "f_")
    assert fn in body["note"], (
        f"the note does not name the estimator: {body['note']!r}")
    # THE ESTIMATOR NAMED IS ONE THE FOLD ACTUALLY FITS. A note naming a
    # function nobody calls is trap #2 with a citation on it.
    named = [w.strip(".,") for w in body["note"].split()
             if w.startswith(("mutual_info_", "f_classif", "f_regression"))]
    assert named, body["note"]
    source = Path(pipeline_plan.__file__).read_text(encoding="utf-8")
    for n in named:
        assert n in source, (
            f"the preview says it scored with {n}, and `pipeline_plan` — the "
            f"module that does the selecting — never mentions it")


# ── 3 · the three that CANNOT are still on the shelf, and say so ─────────────

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES), ids=sorted(TARGET_SHAPES))
@pytest.mark.parametrize("key", NO_SCORE)
def test_a_method_with_no_per_column_score_refuses_and_remains_offered(
        client, shape, key):
    """**The shelf is never shortened.**

    The honest answer for LASSO, RFE and stability is `score: None` with a
    measure naming what was not computed — the vocabulary `selection.evidence`
    already had for a non-numeric column. What must NOT happen is the method
    disappearing from the palette, or being previewed with somebody else's
    number.
    """
    pid, _ = _sealed(client, shape)
    body = _evidence(client, pid, key)

    assert body["is_ranked"] is False, (
        f"'{key}' claims a ranking it cannot compute")
    assert all(r["score"] is None for r in body["ranked"]), (
        f"'{key}' returned a number: "
        f"{[(r['feature'], r['score']) for r in body['ranked']][:3]}")
    assert _sel.METHODS[key].label in body["measure"], body["measure"]
    assert "nothing here is ranked" in body["note"].lower(), body["note"]
    assert "rather than a ranking" in body["note"], body["note"]
    assert "still offered" in body["note"], (
        "the refusal does not say the method remains available, so a reader "
        "cannot tell 'no preview' from 'no method'")

    # STILL ON THE SHELF, asked of the route that composes the palette.
    offered = [m["key"] for m in
               client.get(f"/project/{pid}/features").json()["selection_methods"]]
    assert key in offered, (
        f"'{key}' was dropped from the palette rather than given an honest "
        f"preview — the shelf was shortened")

    # AND STILL RECORDABLE. A method that previews nothing must still be a
    # choice the record accepts, or the refusal above has quietly removed it.
    cands = client.get(f"/project/{pid}/features").json()["numeric_columns"]
    r = client.post(f"/project/{pid}/decision", json={
        "kind": "set_selection",
        "payload": {"method": key, "candidates": cands}})
    assert r.status_code == 200, (key, r.text[:250])
    assert r.json()["selection_spec"]["method"] == key


# ── 4 · the sharpest one: MI sees what a correlation cannot ──────────────────

def test_mutual_information_ranks_the_column_correlation_ranks_last():
    """**Why the substitution mattered most for the method it was paired with.**

    Mutual information is chosen precisely because it reads non-monotone
    association. On a column with `y = (x - 0.5)**2` the Pearson correlation is
    near zero by construction and the MI is the largest in the table — so the
    correlation preview put the single most informative column LAST under a
    sentence promising a mutual-information selection.

    Direct call rather than a fixture upload, because no shipped CSV contains a
    deliberately non-monotone pair and inventing one in `sample_data` to make a
    statistical point would be a fixture manufacturing its own result.
    """
    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(0, 1, n)
    noise = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "curved": x,
        "noise": noise,
        "linear": rng.uniform(0, 1, n),
    })
    df["y"] = (df["curved"] - 0.5) ** 2 + 0.02 * rng.normal(0, 1, n)
    df["linear"] = df["y"] * 0.6 + 0.4 * rng.uniform(0, 1, n)
    cands = ["curved", "noise", "linear"]

    corr = _sel.evidence(df, "y", cands)
    assert corr["measure"] == PEARSON                                # control
    corr_order = [r["feature"] for r in corr["ranked"]]
    assert corr_order[-1] == "curved", (
        "this fixture cannot tell the two measures apart: the correlation "
        f"ranking is {corr_order}, and `curved` is meant to be last in it")

    mi = _sel.evidence(df, "y", cands, method="mutual_info",
                       task_type="regression")
    mi_order = [r["feature"] for r in mi["ranked"]]
    assert mi["measure"] == "mutual information with the outcome"
    assert mi_order[0] == "curved", (
        f"mutual information ranked {mi_order}; the non-monotone column is "
        f"not first, so this preview is not reading what MI reads")
    assert mi_order != corr_order, (
        "the two measures produced the same ordering, so this file cannot "
        "tell a method-aware preview from the old one")
    # AND THE NUMBERS ARE NOT THE CORRELATION'S. Same ordering by accident
    # would still be caught above; same values would mean the branch is dead.
    assert ([r["score"] for r in mi["ranked"]]
            != [r["score"] for r in corr["ranked"]])


# ── 5 · the string-outcome half: the reason named the wrong column ───────────

def test_a_label_outcome_is_named_as_the_reason_a_correlation_was_not_computed():
    """Every numeric candidate came back *not numeric — not ranked here* on a
    project whose target is a string, which is false of a column of floats.

    The refusal is the outcome's, so the sentence is about the outcome.
    """
    df = pd.DataFrame({"age": [40.0, 50.0, 60.0, 70.0, 55.0],
                       "chol": [1.0, 2.0, 3.0, 4.0, 5.0],
                       "status": ["died", "lived", "died", "lived", "died"]})
    body = _sel.evidence(df, "status", ["age", "chol"])

    assert body["is_ranked"] is False
    assert "status" in body["measure"], body["measure"]
    for row in body["ranked"]:
        assert "not numeric — not ranked here" != row["measure"], (
            f"{row['feature']!r} is a float column and the preview says it is "
            f"not numeric")

    # And a method that CAN read a label outcome does read it.
    mi = _sel.evidence(df, "status", ["age", "chol"],
                       method="mutual_info", task_type="classification")
    assert mi["is_ranked"] is True
    assert any(r["score"] is not None for r in mi["ranked"]), mi["ranked"]


# ── 6 · a method the module does not have is refused, not substituted ────────

def test_an_unknown_method_is_refused_rather_than_ranked_by_correlation(client):
    """The failure mode this fix could most easily reintroduce: an unrecognized
    method falling through to the old default and being served as a ranking."""
    pid, _ = _sealed(client, "continuous")
    r = client.get(f"/project/{pid}/selection/evidence?method=pearson")
    assert r.status_code == 400, r.text[:250]
    assert "not a selection method" in r.json()["detail"]


# ── 7 · pressing the button twice gives the same numbers ─────────────────────

def test_two_identical_requests_return_identical_mutual_information(client):
    """`mutual_info_*` adds noise to break nearest-neighbour ties, and an
    unseeded preview returns different numbers on each press — which a reader
    can only read as the data having changed. `PREVIEW_SEED` is why.
    """
    pid, _ = _sealed(client, "multiclass")
    first = _evidence(client, pid, "mutual_info")
    second = _evidence(client, pid, "mutual_info")
    assert [r["score"] for r in first["ranked"]] == [
        r["score"] for r in second["ranked"]]
    assert any(r["score"] for r in first["ranked"]), (
        "every score is zero or null, so this claim would hold over a dead "
        "branch")                                                     # control


# ── 8 · THE CONSUMER, and it is not there yet ────────────────────────────────

@pytest.mark.skipif(not H.available(), reason="node is not installed")
# This carried `xfail(strict=True)` while it was written, naming the exact edit
# it lacked: the page is serialized through one writer and the agent that built
# the server half could not make it. The edit landed in the same commit and the
# marker came off with it — which is what STRICT is for, since it would have
# XPASSed and failed if anyone had left it on.
def test_the_rank_button_sends_the_method_sitting_in_the_dropdown(client):
    """`DRIVE-011`'s shape for this control.

    The page's real click handler is run under node and the URL its real
    `fetch` composes is read. The claim is about the REQUEST, because that is
    the whole of the defect: a server that can answer per method and a button
    that never asks per method leaves the table exactly as wrong as before.
    """
    pid, _ = _sealed(client, "continuous")
    served = client.get(f"/project/{pid}").json()
    routes = {
        f"/project/{pid}": served,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/interview?step=features":
            client.get(f"/project/{pid}/interview?step=features").json(),
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": {"columns": []},
        f"/project/{pid}/features":
            client.get(f"/project/{pid}/features").json(),
        f"/project/{pid}/figures": client.get(f"/project/{pid}/figures").json(),
        f"/project/{pid}/preprocess":
            client.get(f"/project/{pid}/preprocess").json(),
        f"/project/{pid}/recipes": client.get(f"/project/{pid}/recipes").json(),
        f"/project/{pid}/models": client.get(f"/project/{pid}/models").json(),
        f"/project/{pid}/selection/evidence":
            _evidence(client, pid),
        f"/project/{pid}/selection/evidence?method=mutual_info":
            _evidence(client, pid, "mutual_info"),
    }
    out = H.run(
        """
        function settle(n){
          var p = Promise.resolve();
          for (var i = 0; i < n; i++) { p = p.then(function(){}); }
          return p;
        }
        settle(24).then(function(){
          __harness.drainRaf();
          var pick = __harness.target({'data-sel': 'method'}, []);
          pick.value = 'mutual_info';
          __harness.dispatch('change', pick);
          return settle(6);
        }).then(function(){
          __harness.dispatch('click', __harness.target({'data-sel-rank': '1'},
                                                       ['pill']));
          return settle(8);
        }).then(function(){
          __emit({calls: __harness.calls(),
                  evidence: __harness.html('selEvidence')});
        });
        """, routes=routes, search=f"?project={pid}")

    ranks = [c["path"] for c in out["calls"]
             if "selection/evidence" in c["path"]]
    assert ranks, "the press fetched no ranking at all"
    assert any("method=mutual_info" in p for p in ranks), (
        f"the Rank button asked for {ranks[-1]!r} with `mutual_info` chosen "
        f"in the dropdown beside it, so the table it fills is evidence for a "
        f"method the user did not pick")
