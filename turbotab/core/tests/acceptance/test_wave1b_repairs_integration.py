"""The wave 1b repairs' seams: what holds only once REPAIR-MI and REPAIR-VALID run on the wave 2a
repairs (MODELING_SEQUENCE §0 rulings 4 and 12, §1 row 12 (b), §4; BLUEPRINT §14.1–§14.2).

Each repair's own acceptance file tests its package against its references (``test_mi_repair.py``,
``test_ms6_prediction_validation.py``). This file tests the places where a repair meets a rule
another package wrote:

* **Time-invariance in Table 2 and the secondary model** (REPAIR-MI's ruling 12; REPAIR-ESTIMAND's
  own copies; WP17's "further adjusted for" model). Under
  clustered multiple imputation a column whose recorded values agree within every unit is asked
  about before any copy is drawn. Table 2's models are imputed as the fit imputes, so the primary
  asks the fit's question with the fit's exits; once it is answered, Model 2 is the fit's primary
  number for number, and the record counts the carried cells as pandas counts them. Model 3's own
  copies hold its added column, so Model 3 asks about that column itself, carries the question's
  exits (a question with no answer beside it is a dead end, BLUEPRINT §14.2), and the methods text
  lists only the models reported. The secondary stage's further-adjusted model asks the same, and
  its sentence reports the primary alone.
* **The explanations show the comparison too** (REPAIR-VALID and wave 2a's EXPLAIN). The record of
  scores shown (``scores_seen.json``) is what refuses the winner's own score with no rows held out.
  Each explained family's floor quotes its cross-validated score against the baseline's, so an
  explanation served before the fit is a score seen: the record keeps it, and dropping a family
  whose score it showed is refused, with the exit that keeps it.

Expected numbers come from an independent path: pandas counts over the data, and the fit stage's
own table where the claim is that two stages agree.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.tests.acceptance import estimand_fixtures as ef


def _rec(i: int, decision: Any) -> d.DecisionRecord:
    return d.DecisionRecord(id=f"r{i}", seq=i, at="2026-10-05T00:00:00Z", decision=decision)


def _visits(seed: int = 10) -> pd.DataFrame:
    """Two visits a person (200 people): `age` and `bmi` are the person's own (one value each),
    `fiber` and `glucose` change between visits. `age` is asked only at the first visit (blank on 10%
    of those), `bmi` is blank on 10% of visits and `fiber` on 20%, at random."""
    one = ef.cohort(200, seed=seed)
    rng = np.random.default_rng(seed + 1)
    two = one.assign(glucose=one["glucose"] + rng.normal(0, 5, 200),
                     fiber=one["fiber"] + rng.normal(0, 2, 200))
    frame = pd.concat([one, two], ignore_index=True)
    first = np.r_[np.ones(200, bool), np.zeros(200, bool)]
    frame.loc[~first, "age"] = np.nan
    frame.loc[first & (rng.random(400) < 0.1), "age"] = np.nan
    frame.loc[rng.random(400) < 0.2, "fiber"] = np.nan
    frame.loc[rng.random(400) < 0.1, "bmi"] = np.nan
    return frame


def _carried(frame: pd.DataFrame, column: str) -> int:
    """pandas: the blank cells of ``column`` on rows whose person records it on another row."""
    recorded = frame.groupby("pid")[column].transform(lambda v: v.notna().any())
    return int((frame[column].isna() & recorded).sum())


def _table2(frame: pd.DataFrame, folder: Any, answers: dict[str, str]) -> dict[str, Any]:
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  missing={"strategy": "multiple_imputation", "m": 20},
                  grain=d.GrainSpec(grain="repeated", id_column="pid"))
    if answers:
        folded = d.fold([_rec(i, d.ConfirmReading(reading="time_invariant", column=c, value=v))
                         for i, (c, v) in enumerate(answers.items(), start=1)])
        st = st.model_copy(update={"reading_confirmations": folded.reading_confirmations})
    out = ef.run(frame, folder, st)
    # the declared "further adjusted for" model beside the primary (WP17's secondary stage), on the
    # same ingested rows
    from turbotab.core.stages.secondary import secondary_stage
    from turbotab.core.tests import modeling_fixtures as mf

    paths = mf.ingest_frame(frame, folder / "secondary")
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    inputs = {"design": out["design"], "split": split,
              "target_info": mf.target_info(st.task, st.target)}
    out["secondary"] = secondary_stage(mf.context(st, inputs, paths)).data
    return out


@pytest.fixture(scope="module")
def visits(tmp_path_factory):
    frame = _visits()
    base = tmp_path_factory.mktemp("visits")
    return {"frame": frame,
            "unasked": _table2(frame, base / "unasked", {}),
            "age": _table2(frame, base / "age", {"age": "yes"}),
            "both": _table2(frame, base / "both", {"age": "yes", "bmi": "yes"})}


def _exits(inference: dict[str, Any]) -> list[dict[str, Any]]:
    return [e["decision"] for e in inference.get("exits") or []]


def test_unasked_table_2_asks_the_fits_question_with_its_exits(visits):
    """Ruling 12 (REPAIR-MI) on REPAIR-ESTIMAND: the people's `age` agrees within every person
    (pandas: at most one distinct value each) and nothing is confirmed, so the fit asks before any
    copy is drawn, and Table 2, imputed as the fit imputes, asks the same question with the same
    exits instead of an estimate; no estimate is reported, and the methods text says why."""
    frame = visits["frame"]
    assert int(frame.groupby("pid")["age"].nunique().max()) == 1
    fit = visits["unasked"]["fit_raw"]["models"][0]["inference"]
    effects = visits["unasked"]["effects"]
    [primary] = effects["families"][0]["sequence"]
    assert primary["key"] == "model_2" and primary["effects"] is None
    reason = fit["refused"]
    assert reason.startswith("The rows repeat by `pid`, and the recorded values of `age` agree "
                             "within every `pid`.")
    assert primary["inference"]["refused"] == reason
    assert _exits(primary["inference"]) == _exits(fit) and len(_exits(fit)) == 2
    assert {(e["column"], e["value"]) for e in _exits(fit)} == {("age", "yes"), ("age", "no")}
    assert effects["methods"] == f"No estimate of `fiber` is reported: {reason}"


def test_answered_model_2_is_the_fits_primary_and_model_3_asks_its_own_question(visits):
    """With `age` confirmed as one value a person: Model 2's estimate and SE are the fit's primary's
    to 1e-12 (the same copies, imputed once per person); both records carry `age` to the blank rows
    of the people who record it, as many cells as pandas counts, and impute it once per person
    where no row does. Model 3 adds `bmi`, which also agrees within every person and is unanswered,
    so Model 3 alone asks, with its exits; the other models stand, and the methods sentence lists
    only the models reported (asserted verbatim up to the display rule)."""
    frame = visits["frame"]
    run = visits["age"]
    fit = run["fit_raw"]["models"][0]
    seq = ef.sequence(run["effects"])
    assert list(seq) == ["crude", "model_2", "model_3"]
    primary = next(r for r in fit["coefficients"] if r["feature"] == "fiber")
    [row] = [r for r in seq["model_2"]["effects"] if r["feature"] == "fiber"]
    assert row["estimate"] == pytest.approx(primary["estimate"], rel=1e-12)
    assert row["se"] == pytest.approx(primary["se"], rel=1e-12)
    carried = _carried(frame, "age")
    assert carried == 181
    for record in (fit["inference"]["missing"], seq["model_2"]["inference"]["missing"],
                   seq["crude"]["inference"]["missing"]):
        assert record["unit"] == "pid" and record["unit_level"] == ["age"]
        assert record["unit_carried"] == {"age": carried} and record["unit_imputed"] == ["age"]
        assert record["row_level"] == ["fiber"]
        assert record["m"] == fit["inference"]["missing"]["m"]
    three = seq["model_3"]
    assert three["effects"] is None and three["adjusted_for"][-1] == "bmi"
    reason = three["inference"]["refused"]
    assert reason.startswith("The rows repeat by `pid`, and the recorded values of `bmi` agree "
                             "within every `pid`.")
    assert three["concerns"] == [f"It could not be fit: {reason}"]
    assert {(e["column"], e["value"]) for e in _exits(three["inference"])} == {
        ("bmi", "yes"), ("bmi", "no")}
    assert run["effects"]["methods"].startswith(
        "The estimate of `fiber` is reported across a declared sequence of models fit on all 400 "
        "analyzed rows: unadjusted; Model 2, the primary, adjusted for `age`, `sex`, `smoking` and "
        "`activity`. Model 3, further adjusted for `bmi`, could not be fit and is not reported. "
        "Only the exposure's estimates are shown as effects;")
    # The secondary stage's further-adjusted model holds `bmi` too: it asks with the same exits,
    # and its methods sentence reports the primary alone.
    primary_fit, further = run["secondary"]["families"][0]["fits"]
    assert primary_fit["coefficients"][0]["estimate"] == pytest.approx(primary["estimate"],
                                                                       rel=1e-12)
    assert not further["coefficients"] and further["inference"]["refused"] == reason
    assert _exits(further["inference"]) == _exits(three["inference"])
    assert run["secondary"]["methods"] == (
        "Beside the primary model, the same model further adjusted for `bmi` was declared before "
        "the estimates were shown, but it could not be fit, so the estimate of `fiber` is reported "
        "from the primary model alone, fit on every analyzed row.")


def test_answered_twice_model_3_is_fit_in_copies_that_carry_both(visits):
    """With `bmi` confirmed too, Model 3 is fit in its own copies, which carry and impute both
    columns once per person (pandas counts the carried cells); Model 2 is still the fit's primary,
    and the methods sentence names Model 3 among the models reported."""
    frame = visits["frame"]
    run = visits["both"]
    seq = ef.sequence(run["effects"])
    primary = next(r for r in run["fit_raw"]["models"][0]["coefficients"]
                   if r["feature"] == "fiber")
    [row] = [r for r in seq["model_2"]["effects"] if r["feature"] == "fiber"]
    assert row["estimate"] == pytest.approx(primary["estimate"], rel=1e-12)
    three = seq["model_3"]
    assert three["effects"] and three["inference"]["refused"] is None
    record = three["inference"]["missing"]
    assert record["unit_level"] == ["age", "bmi"]
    assert record["unit_carried"] == {"age": _carried(frame, "age"), "bmi": _carried(frame, "bmi")}
    assert run["effects"]["methods"].startswith(
        "The estimate of `fiber` is reported across a declared sequence of models fit on all 400 "
        "analyzed rows: unadjusted; Model 2, the primary, adjusted for `age`, `sex`, `smoking` and "
        "`activity`; Model 3, further adjusted for `bmi`, a possible mediator, so not a total "
        "effect. Only the exposure's estimates are shown as effects;")
    fits = run["secondary"]["families"][0]["fits"]
    assert all(f["coefficients"] for f in fits)
    assert fits[1]["coefficients"][0]["estimate"] == pytest.approx(
        next(r for r in three["effects"] if r["feature"] == "fiber")["estimate"], rel=1e-12)
    assert run["secondary"]["methods"] == (
        "Beside the primary model, the same model further adjusted for `bmi` was declared before "
        "the estimates were shown and fit on every analyzed row; the estimate of `fiber` is "
        "reported from each.")


# ── the explanations show the comparison too (REPAIR-VALID × EXPLAIN) ─────────


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from turbotab.server.tests.conftest import make_client

    with make_client(tmp_path_factory.mktemp("wave1b_server"), "local", 2,
                     "http://127.0.0.1") as c:
        yield c


def test_an_explanation_served_before_the_fit_is_a_score_seen(client, tmp_path):
    """Through the real server: two families fitted with no rows held out, and only the
    explanations served (never the fit). Each family's floor quotes its cross-validated log loss
    against the baseline's, so the record of scores shown now names both (read back from
    ``scores_seen.json``), and dropping one is refused as it is after the fit was served, with the
    exit that keeps the compared families (the message as REPAIR-VALID wrote it)."""
    from turbotab.core.models.selection import read_seen
    from turbotab.core.tests.acceptance.test_ms6_prediction_validation import (_accepted, _opened,
                                                                               _refused)
    from turbotab.server.tests.conftest import wait_for

    rng = np.random.default_rng(404)
    n = 160
    X = rng.normal(size=(n, 3))
    risk = 1 / (1 + np.exp(-(-0.2 + X @ np.array([0.9, -0.6, 0.0]))))
    frame = pd.DataFrame(X.round(4), columns=["x1", "x2", "x3"]).assign(
        pid=[f"P{i:04d}" for i in range(n)], y=np.where(rng.random(n) < risk, "yes", "no"))
    pid = _opened(client, tmp_path, frame, "explained.csv")
    _accepted(client, pid, {"kind": "set_split", "holdout": 0.0, "seed": 1, "folds": 5})
    _accepted(client, pid, {"kind": "select_models", "models": ["linear", "elastic_net"]})
    _accepted(client, pid, {"kind": "set_explain", "reseeds": 0})
    wait_for(client, pid, {"explain": "fresh"}, timeout=600)
    folder = client.app.state.service.workspace.project_dir(pid)
    assert read_seen(folder) == {}  # nothing served yet
    explained = client.get(f"/api/projects/{pid}/stages/explain").json()["artifact"]
    floors = {f["family"]: f["floor"] for f in explained["families"]}
    assert set(floors) == {"linear", "elastic_net"}
    assert all(f["model"] is not None and f["metric"] == "Log loss" for f in floors.values())
    assert read_seen(folder) == {"y": ["linear", "elastic_net"]}
    error = _refused(client, pid, {"kind": "select_models", "models": ["elastic_net"]},
                     "compared_families_stay")
    assert error["message"] == (
        "These families' cross-validated scores were shown for `y` with no rows held out: "
        "`linear`. Dropping them would make the remaining family's own score the result, which "
        "flatters the choice (Tsamardinos et al. 2018). Keep them: the result is then the "
        "selection-corrected estimate, and the best family is still the one deployed.")
    assert error["exits"][0]["decision"]["models"] == ["elastic_net", "linear"]
