"""Under Predict with rows held out, the served fit's declared result once the seal is opened
(candidate 10; audit WP16, RO-05; MODELING_SEQUENCE §1 row 12).

Before the opening the fit declares that the final model's held-out score will be the result, and
states no number. Once the held-out rows are opened with a final family named, the result is the
held-out score the first opening recorded for that family on its metric (``at_opening``), said in
the past tense with the number in it. Its fields carry the final family's held-out interval only
while the scores served beside it are the opening's (``at_opening.current``): after a fit changed
after the opening, a re-seal, or rows drawn and opened again, the score stays and no interval is
given. The sentence states no interval: it is Table 2's caption, beside a result row that gives
none (``export.tables.performance``).

Every truth here is constructed: a served fit with two families' held-out scores and intervals, the
first in fit order not the final one, and an ``open_seal`` record naming the second with scores of
its own. The expected estimate and interval are the constructed numbers, and the expected sentence
prints them by the results table's rule written out in ``test_sentence_precision.table``.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone

from turbotab.core import decisions as d
from turbotab.core import seal
from turbotab.core.decisions import DecisionRecord
from turbotab.core.export import tables
from turbotab.core.models.selection import declared_result, mark_final, result_at_opening
from turbotab.core.tests.test_sentence_precision import table

N_HOLDOUT = 150
# What the opening recorded: elastic net is the final model, linear comes first in fit order.
AT_OPENING = {"linear": {"mse": 343.2, "r2": 0.11767},
              "elastic_net": {"mse": 342.068, "r2": 0.1203}}
INTERVAL = {"estimate": 342.068, "se": 15.6, "ci_low": 311.42, "ci_high": 372.7, "level": 0.95,
            "method": "influence function"}
FUTURE = ("The held-out rows score the final model, declared on cross-validation before they are "
          "opened; its held-out MSE will be the result.")


def record(seq: int, decision, rid: str | None = None, post_seal: bool = False) -> DecisionRecord:
    return DecisionRecord(id=rid or f"r{seq}", seq=seq, at=datetime.now(timezone.utc),
                          decision=decision, post_seal=post_seal)


def opened_log(family: str | None = "elastic_net") -> list[DecisionRecord]:
    return [
        record(1, d.SetTarget(column="hba1c")),
        record(2, d.SetPurpose(purpose="prediction")),
        record(3, d.SetSplit(holdout=0.25, seed=0, folds=5)),
        record(4, d.SelectModels(models=["linear", "elastic_net"])),
        record(5, d.OpenSeal(family=family, target="hba1c", n_holdout=N_HOLDOUT, metric="mse",
                             scores=copy.deepcopy(AT_OPENING)), "open"),
    ]


def model(family: str, label: str, scores: dict | None, interval: dict | None) -> dict:
    return {"family": family, "label": label, "cv": {"mse": {"estimate": 400.0}},
            "holdout": scores,
            "holdout_detail": None if interval is None else {"intervals": {"mse": interval}}}


def served_fit(holdout: dict | None = None, *, opened: bool = True,
               family: str | None = "elastic_net") -> dict:
    """The fit as ``ProjectService._served_fit`` has it before the result is filled: the fit's own
    declared result (``declared_result`` with rows held out), its held-out scores and intervals as
    served, and the final family marked."""
    holdout = copy.deepcopy(AT_OPENING) if holdout is None else holdout
    result = declared_result("regression", "mse", n_holdout=N_HOLDOUT,
                             families=["linear", "elastic_net"], labels={}, summaries={},
                             selection=None).model_dump(mode="json")
    assert result == {**result, "basis": "holdout", "estimate": None, "sentence": FUTURE}
    other = {**INTERVAL, "estimate": 343.2, "ci_low": 312.9, "ci_high": 373.5}
    out = {"task": "regression", "primary_metric": "mse", "metric_labels": {"mse": "MSE"},
           "n_holdout": N_HOLDOUT, "result": result,
           "models": [model("linear", "Linear", holdout.get("linear") if opened else None,
                            other if opened else None),
                      model("elastic_net", "Elastic net",
                            holdout.get("elastic_net") if opened else None,
                            INTERVAL if opened else None)]}
    return mark_final(out, opened=opened, family=family if opened else None)


def opened_sentence(rest: str) -> str:
    return (f"Elastic net was declared the final model on cross-validation before the held-out "
            f"rows were opened, so its held-out MSE at the opening (decision #5) is the result: "
            f"{table(342.068)} on {N_HOLDOUT:,} held-out rows{rest}")


def test_after_the_opening_the_result_is_the_score_the_opening_recorded_for_the_final_family():
    records = opened_log()
    at = seal.reported_result(records, opened=True, changed=False)
    assert at["current"] is True and at["family"] == "elastic_net"
    out = result_at_opening(served_fit(), at)
    result = out["result"]
    assert result["basis"] == "holdout" and result["family"] == "elastic_net"
    assert result["name"] == "Elastic net" and result["metric"] == "mse"
    # The final family's recorded score, not the first family's in fit order.
    assert result["estimate"] == 342.068 and result["estimate"] != AT_OPENING["linear"]["mse"]
    assert (result["ci_low"], result["ci_high"]) == (311.42, 372.7)
    assert result["sentence"] == opened_sentence(".")
    assert "will be" not in result["sentence"]
    assert out["final_model"] == "elastic_net"  # the other fields stand as mark_final left them


def test_table_2s_caption_states_no_interval_beside_the_result_row_that_gives_none():
    """P1: the performance table's caption is the declared result's sentence, and its row labeled
    the result gives the opening's score with no interval (``export.tables.performance``), so the
    sentence states none, opening current or not. The final family's own held-out row beside it
    still gives the constructed interval while the opening is current."""
    current = seal.reported_result(opened_log(), opened=True, changed=False)
    changed = seal.reported_result(
        [*opened_log(), record(6, d.SetSplit(holdout=0.25, seed=0, folds=10), post_seal=True)],
        opened=True, changed=True)
    assert current["current"] is True and changed["current"] is False
    for at in (current, changed):
        fit = result_at_opening({**served_fit(), "at_opening": at}, at)
        [t] = tables.performance(fit)
        assert t.caption == fit["result"]["sentence"]
        [row] = [r for r in t.rows if r["role"] == tables.RESULT]
        assert row["key"] == "result/holdout/mse" and row["estimate"] == 342.068
        assert row["ci_low"] is None and row["ci_high"] is None
        for low, high in ((311.42, 372.7), (312.9, 373.5)):
            assert f"interval {table(low)} to {table(high)}" not in t.caption
        assert "% interval" not in t.caption
        if at["current"]:
            assert (fit["result"]["ci_low"], fit["result"]["ci_high"]) == (311.42, 372.7)
            own = next(r for r in t.rows if r["key"] == "elastic_net/holdout/mse")
            assert (own["ci_low"], own["ci_high"]) == (311.42, 372.7)


def test_after_a_change_after_the_opening_the_score_stays_and_no_interval_is_given():
    """The fit changed after the opening: its held-out scores and intervals shown now are its own,
    so the result keeps the opening's score and gives no interval."""
    records = [*opened_log(), record(6, d.SetSplit(holdout=0.25, seed=0, folds=10), post_seal=True)]
    at = seal.reported_result(records, opened=True, changed=True)
    assert at["current"] is False
    moved = {"linear": {"mse": 351.0, "r2": 0.1}, "elastic_net": {"mse": 349.5, "r2": 0.11}}
    result = result_at_opening(served_fit(moved), at)["result"]
    assert result["estimate"] == 342.068 and result["estimate"] != moved["elastic_net"]["mse"]
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["sentence"] == opened_sentence("; the opening recorded no interval on it.")


def test_after_a_reseal_the_score_at_the_opening_stays_without_an_interval():
    """Re-sealed and not yet opened again (nothing held out is served), then opened again with
    the other family declared: the first opening's score of its own final family stays the result."""
    resealed = [*opened_log(), record(6, d.Reseal(target="hba1c"), post_seal=True)]
    at = seal.reported_result(resealed, opened=False, changed=False)
    assert at["current"] is False
    out = result_at_opening(served_fit(opened=False), at)
    assert out["final_model"] is None  # sealed again: nothing is marked final
    result = out["result"]
    assert result["family"] == "elastic_net" and result["estimate"] == 342.068
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["sentence"] == opened_sentence("; the opening recorded no interval on it.")

    second = {"linear": {"mse": 330.0, "r2": 0.15}, "elastic_net": {"mse": 335.0, "r2": 0.14}}
    reopened = [*resealed, record(7, d.OpenSeal(family="linear", target="hba1c",
                                                n_holdout=N_HOLDOUT, metric="mse",
                                                scores=second), "open-2", post_seal=True)]
    at = seal.reported_result(reopened, opened=True, changed=False)
    assert at["current"] is False and at["family"] == "elastic_net"
    result = result_at_opening(served_fit(second, family="linear"), at)["result"]
    assert result["family"] == "elastic_net" and result["estimate"] == 342.068
    assert result["ci_low"] is None and result["ci_high"] is None


def test_the_interval_is_given_only_beside_the_score_it_is_an_interval_of():
    """Both ways: while the opening is current the interval is the final family's served one when
    its served score is the recorded score (the test above), and none when it is not."""
    at = seal.reported_result(opened_log(), opened=True, changed=False)
    assert at["current"] is True
    other = {**AT_OPENING, "elastic_net": {"mse": 342.5, "r2": 0.1203}}
    result = result_at_opening(served_fit(other), at)["result"]
    assert result["estimate"] == 342.068
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["sentence"] == opened_sentence(".")


def test_a_final_family_with_no_score_at_the_opening_reports_no_result():
    scores = copy.deepcopy(AT_OPENING)
    scores["elastic_net"]["mse"] = None
    records = opened_log()
    records[-1] = record(5, d.OpenSeal(family="elastic_net", target="hba1c", n_holdout=N_HOLDOUT,
                                       metric="mse", scores=scores), "open")
    at = seal.reported_result(records, opened=True, changed=False)
    result = result_at_opening(served_fit(scores), at)["result"]
    assert result["family"] == "elastic_net" and result["estimate"] is None
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["sentence"] == (
        "Elastic net was declared the final model on cross-validation before the held-out rows "
        "were opened, but the opening (decision #5) recorded no held-out MSE for it, so no result "
        "is reported.")


def test_before_the_opening_or_with_no_final_family_named_the_result_is_the_fits_own():
    """Silent where it should be: nothing opened (no ``at_opening``), an opening from before a
    final family had to be named, and a result on any other basis."""
    sealed = served_fit(opened=False)
    assert result_at_opening(copy.deepcopy(sealed), None) == sealed
    assert sealed["result"]["sentence"] == FUTURE and sealed["result"]["estimate"] is None

    at = seal.reported_result(opened_log(family=None), opened=True, changed=False)
    assert at is not None and at["family"] is None
    unnamed = served_fit(family=None)
    assert result_at_opening(copy.deepcopy(unnamed), at) == unnamed

    at = seal.reported_result(opened_log(), opened=True, changed=False)
    for basis in ("own_score", "selection_corrected", "not_declared"):
        other = served_fit()
        other["result"] = {**other["result"], "basis": basis, "family": "linear",
                           "estimate": 400.0, "sentence": "S."}
        assert result_at_opening(copy.deepcopy(other), at) == other
