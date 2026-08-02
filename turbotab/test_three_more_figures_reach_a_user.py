"""`DRIVE-009` — the volcano, the spline and the diverging bar, each reachable.

`LOOP.md` §02, as amended: *stop metering once it has stopped bending*, and
**order a batch hardest-first** — five instances built easiest-first are five
castings of a shape nobody stress-tested. So the volcano went first, and it bent
the spec exactly where the adjudicator predicted.

Every test here goes through HTTP. `GUIDED-058` is the finding that says why: a
figure that renders only from a hand-built payload is a specification, and from
inside the loop that built it it looks finished.

## The three data paths, checked before each figure was built

| Figure | Lens | Needs | Fixture that reaches it |
|---|---|---|---|
| volcano | metabolomics or genomics | a binary target, ≥30 numeric columns | `metabolomics_untargeted.csv` |
| dose–response spline | dietary | a continuous target, ≥40 rows | `dietary_recalls.csv` |
| diverging stacked bar | survey | a declared Likert block | `survey_instrument.csv` |

Kaplan–Meier and the forest plot were checked the same way and are **not** built:
the project has no notion of a time column or an event indicator, and model
coefficients hit `GUIDED-065`'s wall. Five more unreachable specifications is
the mistake `GUIDED-058` just closed.
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

from turbotab import api, figure_bundle, figures                     # noqa: E402
from turbotab import figure_specs as FS                              # noqa: E402
from turbotab import packs as P                                      # noqa: E402

FIXTURES = Path(__file__).parent / "sample_data"


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _drive(client, fixture: str, decisions) -> str:
    with open(FIXTURES / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    for what, payload in decisions:
        response = client.post(f"/project/{pid}/decision",
                               json={"kind": what, "payload": payload})
        assert response.status_code == 200, (what, response.text)
    return pid


@pytest.fixture(scope="module")
def assay(client):
    return _drive(client, "metabolomics_untargeted.csv", [
        ("set_lens", {"lens": ["metabolomics"]}),
        ("set_target", {"column": "responder"})])


@pytest.fixture(scope="module")
def dietary(client):
    return _drive(client, "dietary_recalls.csv", [
        ("set_lens", {"lens": ["dietary"]}),
        ("set_target", {"column": "hba1c"}),
        ("set_grain", {"answer": "people_repeat",
                       "group_col": "participant_id"}),
        ("set_repeat_kind", {"kind": "repeats"})])


@pytest.fixture(scope="module")
def survey(client):
    return _drive(client, "survey_instrument.csv", [
        ("set_lens", {"lens": ["survey"]}),
        ("set_target", {"column": "sought_support"})])


def _row(client, pid: str, figure_id: str):
    body = client.get(f"/project/{pid}/figures").json()
    for row in body["admitted"] + body["held"]:
        if row["id"] == figure_id:
            return row
    raise AssertionError(
        f"{figure_id} was not drawn: "
        f"unavailable={[(r['id'], r['why']) for r in body['unavailable']]} "
        f"not_drawn={[(r['id'], r['why']) for r in body['not_drawn']]}")


# ── all three reach a user ──────────────────────────────────────────────────

@pytest.mark.parametrize("figure_id,fixture", [
    ("volcano", "assay"),
    ("dose_response_spline", "dietary"),
    ("diverging_stacked_bar", "survey"),
])
def test_each_new_figure_reaches_a_user_from_an_upload(client, figure_id,
                                                       fixture, request):
    pid = request.getfixturevalue(fixture)
    row = _row(client, pid, figure_id)
    assert row["caption"], "the figure arrived without its caption"
    assert row["annotations"], "the figure arrived without its annotation box"
    assert row["evidence_status"] in P.EVIDENCE_STATUSES
    assert row["source"].startswith("research/")
    assert row["payload"], "the figure arrived without its numbers"


#: Figures that CANNOT reach a user from an upload alone, and why. Every entry
#: is gated on something a fresh project does not have yet, which is a
#: different state from `GUIDED-058`'s — registered and unreachable because
#: nothing fetched it. A figure absent from this list and absent from the
#: served bundle is that defect returning.
NEEDS_MORE_THAN_AN_UPLOAD = {
    "calibration": "needs a fitted model's predictions",
    "prediction_instability": "needs a bootstrap resampling run (L38-B), "
                              "which is a job the user starts",
    "calibration_instability": "same, and classification only",
    # L40-C. Four more that need a fit, and two more that need the resampling
    # job on top of it. Each names what it waits for rather than being absent.
    "decision_curve": "needs a fitted model's predicted risks, and a binary "
                      "target — net benefit is defined for one predicted risk",
    "roc": "needs a fitted model's predicted risks, and a binary target",
    "forest": "needs a fitted model that exposes coefficients; a tree "
              "ensemble has none, which is a different sentence from "
              "'you have not trained yet'",
    "classification_instability": "needs the resampling job, and a binary "
                                  "target",
    "decision_curve_instability": "needs the resampling job, and a binary "
                                  "target",
}


def test_every_registered_figure_either_reaches_a_user_or_says_what_it_needs(
        client, assay, dietary, survey):
    """`GUIDED-058` closed on three figures with one unreachable, and the rule
    it established is the one asserted here: a figure that does not arrive must
    be gated on something NAMED, not merely missing.

    Written as a partition rather than as a count. The count version said
    `len(REGISTRY) == 6` and broke the day a figure was added — which is a test
    that has to be edited to stay true, and an edit is where the thing it
    guards gets waved through.
    """
    drawn = set()
    for pid in (assay, dietary, survey):
        body = client.get(f"/project/{pid}/figures").json()
        drawn |= {row["id"] for row in body["admitted"] + body["held"]}

    registered = set(figures.REGISTRY)
    assert drawn <= registered, drawn - registered
    unreachable = registered - drawn
    assert unreachable == set(NEEDS_MORE_THAN_AN_UPLOAD), (
        f"unreachable from an upload: {sorted(unreachable)}. Every one must "
        f"be listed in NEEDS_MORE_THAN_AN_UPLOAD with what it is waiting for, "
        f"or it is GUIDED-058 again — registered, and reached by nothing.")
    assert drawn, "no figure reached a user at all"


# ── the volcano · the one that bent the spec ────────────────────────────────

def test_the_volcano_states_the_correction_and_the_counts(client, assay):
    row = _row(client, assay, "volcano")
    payload = row["payload"]
    assert payload["y_axis"] == "neg_log10_q"
    assert payload["correction"] == "Benjamini-Hochberg FDR"
    assert payload["n_features"] > 100
    assert payload["n_significant"] == (payload["n_significant_up"]
                                        + payload["n_significant_down"])
    assert all(item["passed"] for item in row["checklist"]), [
        i["id"] for i in row["checklist"] if not i["passed"]]


def test_the_caption_puts_the_uncorrected_count_beside_chance(client, assay):
    """The research's own coaching: *"At an uncorrected p < 0.05 you'd expect
    about 150 by chance and you have 187 — which is to say, your uncorrected
    result is consistent with nothing happening."*"""
    row = _row(client, assay, "volcano")
    assert "by chance alone" in row["caption"]
    assert "corrected count above is the result" in row["caption"]
    payload = row["payload"]
    assert payload["expected_by_chance"] > 0
    assert payload["n_uncorrected_significant"] >= 0


def test_a_fold_change_is_refused_on_autoscaled_data():
    """**The bend.** `when_applicable` answers *does this apply*; it cannot
    answer *are these data in a state where this figure would tell the truth*.
    After autoscaling a fold change is a fold change in z-units."""
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        rng.normal(0, 1, size=(60, 40)),
        columns=[f"mz_{i:03d}" for i in range(40)])
    frame["group"] = ["a"] * 30 + ["b"] * 30
    with pytest.raises(FS.FigureRefusal) as caught:
        FS.volcano_payload(frame, group_col="group")
    message = str(caught.value)
    assert "autoscaled" in message
    assert "z-units and is meaningless" in message
    # It is a refusal and not an error: it carries a badge and an offer.
    payload = caught.value.to_dict()
    assert payload["evidence_status"] == "SETTLED"
    assert figures.resolve(payload["offer"]["draw"])["status"] == "registered"


def test_a_fold_change_is_refused_where_a_ratio_is_undefined():
    """Negatives in an abundance matrix mean the values have already been
    transformed, and a ratio between them is not a fold change."""
    rng = np.random.default_rng(6)
    frame = pd.DataFrame(rng.normal(50, 400, size=(60, 40)),
                         columns=[f"mz_{i:03d}" for i in range(40)])
    frame["group"] = ["a"] * 30 + ["b"] * 30
    with pytest.raises(FS.FigureRefusal, match="negative"):
        FS.volcano_payload(frame, group_col="group")


def test_the_refusal_reaches_the_user_through_the_endpoint(client):
    """A precondition that only the payload builder can express still has to
    arrive somewhere a user stands. `figure_bundle.render` surfaces it under
    `unavailable`, carrying the refusal's own words — the path the shrinkage
    plot's one-recall refusal already uses, which is the evidence it is a shape
    and not a patch."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(0, 1, size=(60, 40)),
                         columns=[f"mz_{i:03d}" for i in range(40)])
    frame["responder"] = [0] * 30 + [1] * 30
    path = FIXTURES / "_autoscaled_tmp.csv"
    frame.to_csv(path, index=False)
    try:
        pid = _drive(client, path.name, [
            ("set_lens", {"lens": ["metabolomics"]}),
            ("set_target", {"column": "responder"})])
    finally:
        path.unlink()
    body = client.get(f"/project/{pid}/figures").json()
    entry = next(r for r in body["unavailable"] if r["id"] == "volcano")
    assert "z-units" in entry["why"]
    assert entry["evidence_status"] == "SETTLED"
    assert entry["offer"]["draw"] == "pca_scores"
    assert "volcano" not in {r["id"] for r in body["admitted"]}


def test_a_three_level_target_is_refused_rather_than_paired_for_you():
    frame = pd.DataFrame(np.abs(np.random.default_rng(8).normal(
        1000, 200, size=(60, 40))),
        columns=[f"mz_{i:03d}" for i in range(40)])
    frame["group"] = ["a", "b", "c"] * 20
    with pytest.raises(FS.FigureRefusal, match="choosing your contrast"):
        FS.volcano_payload(frame, group_col="group")


# ── the spline · the one that bent nothing ─────────────────────────────────

def test_the_spline_reports_a_p_for_nonlinearity_and_shows_the_exposure(
        client, dietary):
    row = _row(client, dietary, "dose_response_spline")
    payload = row["payload"]
    assert 0.0 <= payload["p_nonlinearity"] <= 1.0
    assert sum(payload["exposure_distribution"]["counts"]) > 0
    assert payload["knot_percentiles"] == [10.0, 50.0, 90.0]
    assert all(item["passed"] for item in row["checklist"]), [
        i["id"] for i in row["checklist"] if not i["passed"]]


def test_the_axis_is_truncated_and_the_item_is_scored_against_the_render(
        client, dietary):
    """The calibration plot's both-ranges trick, arriving unchanged on a figure
    from another field — and inverted, because here truncation is REQUIRED."""
    payload = _row(client, dietary, "dose_response_spline")["payload"]
    drawn, observed = payload["x_range_drawn"], payload["x_range_observed"]
    assert observed[0] < drawn[0] and drawn[1] < observed[1]


def test_repeated_recalls_are_fitted_one_row_per_participant(client, dietary):
    """600 rows from 300 people. A p computed across a person's repeated days
    under independence is too small, and too small in the direction a reader
    would act on. §03 licenses the mean of available days for ranking."""
    payload = _row(client, dietary, "dose_response_spline")["payload"]
    assert payload["unit_of_analysis"] == "participant"
    assert payload["n"] == 300 and payload["n_rows_supplied"] == 600
    caption = _row(client, dietary, "dose_response_spline")["caption"]
    assert "one row per participant" in caption
    assert "would be too small" in caption


def test_the_basis_puts_the_linear_term_first_so_the_p_is_a_block_test():
    """Harrell's parameterization, written out rather than taken from
    `patsy.cr`, and the reason is this: *"is the association non-linear"* is an
    F-test on a contiguous block of columns rather than a comparison of fits."""
    x = np.linspace(0.0, 10.0, 200)
    basis = FS._rcs_basis(x, [1.0, 5.0, 9.0])
    assert basis.shape == (200, 2)
    assert np.allclose(basis[:, 0], x), "the first column is not the linear term"


def test_the_p_for_nonlinearity_is_calibrated_rather_than_merely_computed():
    """`LOOP.md` §06.3: *does new numerical code have its own tests?* A p that
    is computed is not a p that is right, and the way to tell is to run it
    where the answer is known.

    Under a truly linear association the p should be uniform, so about one draw
    in twenty falls below 0.05 — and a real curve should be nowhere near that.
    The first draft of this test used a NOISELESS line and got p = 1e-28: with
    zero residual variance the F denominator is machine epsilon and every
    coefficient is significant. A degenerate fixture would have hidden a real
    error as easily as it produced a false one.
    """
    rng = np.random.default_rng(0)
    p_values = []
    for _ in range(60):
        x = rng.uniform(0.0, 10.0, 300)
        y = 3.0 * x + 1.0 + rng.normal(0.0, 2.0, 300)
        p_values.append(FS.spline_payload(
            pd.DataFrame({"e": x, "y": y}),
            exposure="e", outcome="y")["p_nonlinearity"])
    p_values = np.asarray(p_values)
    assert 0.3 < p_values.mean() < 0.7, p_values.mean()
    assert float((p_values < 0.05).mean()) < 0.20, float((p_values < 0.05).mean())

    curved = rng.uniform(0.0, 10.0, 300)
    payload = FS.spline_payload(
        pd.DataFrame({"e": curved,
                      "y": 2.0 * (curved - 5.0) ** 2 + rng.normal(0, 2, 300)}),
        exposure="e", outcome="y")
    assert payload["p_nonlinearity"] < 1e-20, payload["p_nonlinearity"]


def test_a_spline_over_too_few_points_refuses(client):
    frame = pd.DataFrame({"e": np.linspace(1, 10, 20),
                          "y": np.linspace(2, 5, 20)})
    with pytest.raises(FS.FigureRefusal, match="handful of points"):
        FS.spline_payload(frame, exposure="e", outcome="y")


# ── the diverging bar · the item the app can never pass ────────────────────

def test_the_bar_is_sorted_by_net_agreement_and_says_so(client, survey):
    row = _row(client, survey, "diverging_stacked_bar")
    payload = row["payload"]
    nets = [item["net_agreement"] for item in payload["items"]]
    assert nets == sorted(nets, reverse=True)
    assert "ordered by net agreement" in row["caption"]
    assert payload["n_items"] == 40


def test_the_legend_says_the_anchors_are_missing_rather_than_printing_codes(
        client, survey):
    """Requirement 7 is the anchors verbatim, and the app has only the numeric
    codes. The item FAILS — the figure genuinely is not publication-grade
    without them — and the legend says so rather than printing `1 … 5` as if
    those were the words."""
    row = _row(client, survey, "diverging_stacked_bar")
    payload = row["payload"]
    assert payload["anchors"] is None
    assert "not recoverable from the data" in payload["anchors_absent_because"]
    item = next(i for i in row["checklist"]
                if i["id"] == "anchors_verbatim_in_the_legend")
    assert item["passed"] is False
    assert "codes are not the question" in item["because"]
    # And the absence renders as an absence rather than as a blank cell.
    anchors = next(a for a in row["annotations"] if a["key"] == "anchors")
    assert anchors["value"] == figures.NOT_ESTIMABLE


def test_the_disputed_neutral_treatment_is_stated_in_the_caption(client,
                                                                 survey):
    """*"How to treat the neutral midpoint is disputed … TurboTab defaults to
    splitting and states the choice in the caption."*"""
    row = _row(client, survey, "diverging_stacked_bar")
    assert row["payload"]["neutral_treatment"] == FS.NEUTRAL_SPLIT
    assert row["payload"]["neutral_treatment_status"] == P.DISPUTED
    assert "split across the zero line" in row["caption"]
    assert "that choice is disputed" in row["caption"]


def test_the_percentages_say_what_they_are_of(client, survey):
    """With item-level missingness the two bases differ, and the difference is
    invisible on the bar. The fixture has both — n runs 288 to 300."""
    payload = _row(client, survey, "diverging_stacked_bar")["payload"]
    assert payload["percentage_basis"] == "respondents_answering_the_item"
    assert payload["n_min"] < payload["n_max"]
    assert "respondents answering each item" in \
        _row(client, survey, "diverging_stacked_bar")["caption"]


# ── the lens decides what is drawn ─────────────────────────────────────────

@pytest.mark.parametrize("figure_id,fixture,phrase", [
    ("volcano", "metabolomics_untargeted.csv", "metabolomics or genomics"),
    ("dose_response_spline", "dietary_recalls.csv", "dietary intake"),
    ("diverging_stacked_bar", "survey_instrument.csv", "survey or"),
])
def test_the_same_table_without_its_lens_draws_none_of_them(
        client, figure_id, fixture, phrase):
    """`DOMAIN_PACKS.md` §08 and `DRIVE-009`'s own act field: per-domain figure
    selection through the pack mechanism. Same table, no lens, no figure — and
    the reason names the lens rather than the data."""
    pid = _drive(client, fixture, [("set_lens", {"lens": ["other"]})])
    body = client.get(f"/project/{pid}/figures").json()
    assert figure_id not in {r["id"] for r in body["admitted"]}
    entry = next(r for r in body["not_drawn"] if r["id"] == figure_id)
    assert phrase in entry["why"], entry["why"]


def test_every_registered_figure_is_still_accounted_for(client, dietary):
    """The recorded-absence rule does not get weaker as the registry grows."""
    body = client.get(f"/project/{dietary}/figures").json()
    named = {r["id"] for r in body["admitted"] + body["held"]
             + body["unavailable"] + body["not_drawn"]}
    assert named == set(figures.REGISTRY)


def test_guided_066_did_not_reproduce_on_any_of_the_three():
    """**The count, stated, because the instruction was conditional on it.**

    `GUIDED-066` is a checklist item scoring a requirement from a domain the
    table is not in. Three more figures with domain-specific checklists
    produced ZERO new instances, and the reason is structural rather than
    lucky: each of the three is gated by its lens in `when_applicable`, so its
    checklist only ever meets a table from its own field. The PCA scores plot
    is the only figure whose `when_applicable` is domain-free — two numeric
    columns and ten rows — so it is the only one whose checklist can be scored
    against a table from another discipline.

    Which means `GUIDED-066` is not a `ChecklistItem` problem and a third state
    on that enum would not have been the fix.
    """
    gated_by = ("has_assay_lens", "has_dietary_lens", "has_survey_lens",
                "has_predictions")
    # GATED ON STATE, NOT ON DOMAIN — a distinction L38-B forced into the open.
    # The instability plots apply to any project that has run a resampling job,
    # so they are not domain-gated and this scan reads them as domain-free. The
    # question `GUIDED-066` actually asks is narrower: can a figure's CHECKLIST
    # score a requirement from a field the table is not in? Theirs cannot —
    # every item is about B, the reference line, the alpha, the units of the
    # error and whether the scope is stated, none of which belongs to a
    # discipline. So they are excluded here by their gate being named, and the
    # claim below still holds.
    # L40-C added four more state gates. `has_coefficients` is the one worth
    # naming: a forest plot is about coefficients, so a project whose only
    # fitted models are trees is not a project the figure DOES NOT APPLY to —
    # it applies and cannot be drawn, which is why the gate is on the state
    # rather than on the domain.
    state_gated = ("has_instability_run", "has_predictions", "n_classes",
                   "has_coefficients")
    domain_free = [
        spec.id for spec in figures.REGISTRY.values()
        if not any(key in spec.when_applicable.__code__.co_consts
                   for key in gated_by + state_gated)]
    assert domain_free == ["pca_scores"], domain_free
    # And every checklist item that can score a table from another field still
    # belongs to that one figure.
    exposed = {spec.id for spec in figures.REGISTRY.values()
               for item in spec.checklist if "QC" in item.text}
    assert exposed == {"pca_scores"}, exposed
