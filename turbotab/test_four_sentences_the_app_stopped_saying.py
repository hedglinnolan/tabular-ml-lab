"""`DRIVE-044`, `046`, `048`, `049` — four sentences run 5 read that were not true.

Each is small and each is the same family: **a statement placed where a reader
takes it as being about their own numbers, while it is about something else.**

* **`DRIVE-044`** — the calibration caption appended *"(a slope below 1
  indicates predictions that are too extreme)"* as literal text, unconditional
  on the slope printed immediately before it. Run 5 read it beside slope
  **1.141**, which is the opposite problem. The irony is that it sits inside
  the one caption run 5 praised for disclosing its own inadequacy.
* **`DRIVE-046`** — Table 1's Overall column pooled the whole uploaded table,
  so path 1's manuscript failed its own validator: *"Expected analysis N=6297,
  Table 1 overall N=21849."* The strata were right; only Overall pooled.
* **`DRIVE-048`** — the imbalance finding says *"Accuracy can be misleading"*
  and the held-out table then leads with Accuracy 0.88, sitting **at** the
  87.77% base rate. Both true, three cards apart.
* **`DRIVE-049`** — *"No model is selected"* under a button reading *"Fit 2
  model(s)"*. Not stale: the panel was answering about the RECORD's selection
  while the button answered about the page's, and the page never records one.

**The app caught `DRIVE-046` itself** — `passed: false`, rather than exporting
silently — which is the refusal apparatus working and is why that one is medium
rather than high.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, engine, eventfixture, manuscript      # noqa: E402
from turbotab import figure_specs as FS                         # noqa: E402
from turbotab.project import AnalysisProject                    # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


# ── DRIVE-044 · the calibration parenthetical ───────────────────────────────

@pytest.mark.parametrize("slope,expected,forbidden", [
    (0.720, "too extreme", "too conservative"),
    (1.141, "too conservative", "too extreme"),      # run 5's slope
    (1.000, "a slope of 1 is ideal", "too extreme"),
])
def test_the_slope_reading_matches_the_slope(slope, expected, forbidden):
    said = FS._slope_reading(slope)
    assert expected in said, (slope, said)
    assert forbidden not in said, (slope, said)


@pytest.mark.parametrize("absent", [None, float("nan")])
def test_no_reading_is_offered_for_a_slope_nobody_computed(absent):
    """`weak_calibration` returns `(None, None)` on one outcome class, constant
    predictions or complete separation. A reading attached to a quantity that
    does not exist would be the sharpest form of this defect, not a milder
    one."""
    assert FS._slope_reading(absent) == ""


def test_the_caption_carries_the_reading_for_its_own_slope():
    """Through the caption, not only the helper — the defect was a caption."""
    payload = {"model_name": "logreg", "n": 945, "events": 829,
               "calibration_intercept": 0.1, "calibration_slope": 1.141,
               "c_statistic": 0.7, "e_avg": 0.02,
               "curve": {"predicted": [0.1] * 5}, "event_named": False}
    caption = FS.CALIBRATION.caption(payload)
    assert "too conservative" in caption, caption
    assert "too extreme" not in caption, caption


# ── DRIVE-046 · Table 1's Overall column ────────────────────────────────────

def _partly_labeled(n=400, labeled=250):
    rng = np.random.default_rng(4)
    outcome = pd.Series(rng.choice(["case", "ctl"], n, p=[0.8, 0.2]),
                        dtype=object)
    outcome.iloc[labeled:] = None
    frame = pd.DataFrame({"x": rng.normal(0, 1, n).round(2),
                          "z": rng.normal(0, 1, n).round(2),
                          "y": outcome})
    project = AnalysisProject.from_dataframe(frame, "p.csv")
    project.set_target("y", "classification", "high", [])
    engine.record_fix(project, "positive_class__y", choice="case")
    project.set_grain("one_row_per_person")
    project.set_eligibility("everyone")
    return project, n, labeled


def test_table_one_describes_the_cohort_and_not_the_upload():
    project, uploaded, labeled = _partly_labeled()
    table, _ = manuscript.table_one(project)
    overall = [str(c) for c in table.columns if str(c).startswith("Overall")]
    assert overall, list(table.columns)
    assert f"N={labeled}" in overall[0], (
        f"Table 1's Overall column says {overall[0]!r}; the analysis cohort is "
        f"{labeled} rows and {uploaded} is the whole upload")


def test_the_strata_were_already_right_and_still_are():
    """The control. Rows with no outcome are in no level of the stratifier, so
    only Overall ever pooled — and a fix that moved the strata would be
    changing something that was correct."""
    project, _, labeled = _partly_labeled()
    table, _ = manuscript.table_one(project)
    strata = [str(c) for c in table.columns if " (n=" in str(c)]
    total = sum(int(h.split("(n=")[1].rstrip(")")) for h in strata)
    assert total == labeled, (strata, labeled)


# ── DRIVE-048 · the score every metric must beat ────────────────────────────

def test_the_run_carries_what_a_majority_answer_would_score(client):
    """Not a reordering and not a removal — the shelf is never shortened and
    that applies to metrics. What changes is that the number Accuracy has to
    beat travels with it."""
    from turbotab import training as T

    rng = np.random.default_rng(8)
    n = 400
    frame = pd.DataFrame({"x": rng.normal(0, 1, n), "z": rng.normal(0, 1, n),
                          "y": rng.choice(["a", "b"], n, p=[0.88, 0.12])})
    project = AnalysisProject.from_dataframe(frame, "p.csv")
    project.set_target("y", "classification", "high", [])
    engine.record_fix(project, "positive_class__y", choice="a")
    project.set_grain("one_row_per_person")
    project.set_eligibility("everyone")
    idx = list(project.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.25))]
    project.seal_lockbox(labels, fraction=len(labels) / len(project.df))
    run = T.train(project, ["logreg"])

    rate = run.majority_class_rate
    assert rate is not None and 0.5 <= rate <= 1.0, rate
    # Against the held-out rows themselves, which is what every metric beside
    # it is computed on.
    table = project.working_table
    held = table.loc[[i for i in table.index if i in set(labels)], "y"].dropna()
    assert abs(rate - held.value_counts(normalize=True).iloc[0]) < 1e-9
    assert run.to_dict()["majority_class_rate"] == rate


def test_a_regression_run_reports_no_base_rate():
    """`None`, never `0.0` — a base rate on a continuous outcome is not a
    quantity, and zero is a score."""
    from turbotab import training as T

    rng = np.random.default_rng(9)
    n = 300
    frame = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
    project = AnalysisProject.from_dataframe(frame, "p.csv")
    project.set_target("y", "regression", "high", [])
    project.set_grain("one_row_per_person")
    project.set_eligibility("everyone")
    idx = list(project.df.index)
    project.seal_lockbox(idx[:75], fraction=0.25)
    assert T.train(project, ["ridge"]).majority_class_rate is None


# ── DRIVE-049 · the panel and the button ────────────────────────────────────

def _sealed_over_http(client):
    rng = np.random.default_rng(6)
    n = 300
    frame = pd.DataFrame({"x": rng.normal(0, 1, n), "z": rng.normal(0, 1, n),
                          "y": rng.choice(["a", "b"], n, p=[0.7, 0.3])})
    pid = client.post("/project", files={
        "file": ("p.csv", frame.to_csv(index=False).encode(), "text/csv")}
    ).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "y"}})
    eventfixture.choose_event_over_http(client, pid, "y", required=True)
    for kind, payload in (("set_grain", {"answer": "one_row_per_person"}),
                          ("set_eligibility", {"answer": "everyone"}),
                          ("seal", {})):
        client.post(f"/project/{pid}/decision",
                    json={"kind": kind, "payload": payload})
    shelf = client.get(f"/project/{pid}/models").json()
    keys = [m["key"] for g in shelf["groups"] for m in g["models"]][:2]
    return pid, keys


def test_the_panel_answers_about_the_selection_it_was_asked_about(client):
    pid, keys = _sealed_over_http(client)
    assert keys, "the shelf offered nothing, so this asserts nothing"

    unasked = client.get(f"/project/{pid}/training").json()
    assert "No model is selected" in (unasked["blocked_by"] or ""), unasked

    asked = client.get(
        f"/project/{pid}/training?models={','.join(keys)}").json()
    assert asked["blocked_by"] is None, (
        f"the panel still reports {asked['blocked_by']!r} about a selection of "
        f"{keys} — which is the sentence run 5 read beside a button offering "
        f"to fit them")


def test_an_empty_selection_still_says_so(client):
    """The other direction. Naming nothing must not now read as ready."""
    pid, _ = _sealed_over_http(client)
    empty = client.get(f"/project/{pid}/training?models=").json()
    assert "No model is selected" in (empty["blocked_by"] or ""), empty


def test_the_server_still_owns_the_rule(client):
    """**The page names the state; it does not decide the answer.**

    Asking about a model that is not on the shelf must be refused by the
    server, not accepted because the page said so — otherwise this parameter
    would have moved `check()`'s order into the interface, which is the
    duplication this codebase has paid for twice.
    """
    pid, _ = _sealed_over_http(client)
    invented = client.get(f"/project/{pid}/training?models=not_a_model").json()
    assert invented["blocked_by"], invented
    assert "not_a_model" in invented["blocked_by"], invented
