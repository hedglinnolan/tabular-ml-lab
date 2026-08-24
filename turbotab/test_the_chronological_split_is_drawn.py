"""L43-C · `GUIDED-143`'s second half — the draw exists.

L42 made the record honest: the seal stopped describing a chronology nobody
drew and started saying `chronological_requested_not_drawn`. **This draws it.**

> *"TRIPOD+AI treats temporal validation as its own validation type"* — so this
> is wanted rather than tolerated, and a manuscript should be able to say which
> kind of validation it did.

**The shape.** `draw_holdout` is told which column is time; people are ordered
by their **last** observation; the tail is held out with **whole people kept
together**. That last clause is not a nicety — a person split across the
boundary is the grain violation the seal exists to prevent, so a chronological
split that broke grouping would trade one leak for another.

**Ordering by `max`, not `min` or the mean.** The held-out people are the ones
whose *last* visit is latest. Order by first visit and a person enrolled early
with follow-up running past the training data lands in training, and the model
trains on rows from after the rows it is scored on — the exact defect
`GUIDED-143` was filed for.

**The row ranges still overlap and that is correct.** A held-out person's
*early* visits legitimately predate the boundary; what cannot happen is a
held-out person's *last* visit preceding a training person's. L41's measurement
of overlapping ranges was the right alarm on a random draw and is the wrong
test for this one, so the invariant here is stated over last-observations.

**Two things it must not do, and the second took a correction.**

1. It must not resequence anything pre-seal. The lockbox constitution §01 is
   fixed and nothing here touches it.
2. It must not *silently* fall back to `grouped`. The first version of this
   refused whenever `temporal` was set and no time column was recorded, which
   is stronger than the rule asks and is wrong: L42's three-state disclosure
   for that case is a **loud** fallback, and it was accepted. Refusing would
   delete the path a user with no clean date column needs — the shelf being
   shortened. So the line is: **named-and-unusable refuses; unnamed
   discloses.**
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from turbotab import api
from turbotab import engine
from turbotab import repeats as R

FIXTURE = "clinical_longitudinal.csv"
DATA = __import__("pathlib").Path(__file__).resolve().parent / "sample_data"


def _frame():
    return pd.read_csv(DATA / FIXTURE)


GRAIN = {"basis": "grouped", "group_col": "subject_id"}


# ═══════════ THE DRAW ═══════════

def test_the_held_out_people_are_the_latest_ones():
    """The claim the record makes, now made by the draw.

    Stated over LAST observations, because that is what the ordering is over
    and it is the only form of the claim that survives keeping people whole.
    """
    df = _frame()
    got = engine.draw_holdout(df, "sbp", "regression", GRAIN, fraction=0.25,
                              time_col="visit_date", temporal=True)
    held_rows = set(got["labels"])
    assert held_rows, "nothing was held out"

    when = pd.to_datetime(df["visit_date"])
    last = when.groupby(df["subject_id"]).max()
    held_people = set(df.loc[sorted(held_rows), "subject_id"])
    train_people = set(df["subject_id"]) - held_people
    assert held_people and train_people

    earliest_held = last[last.index.isin(held_people)].min()
    latest_train = last[last.index.isin(train_people)].max()
    assert earliest_held >= latest_train, (
        f"a training person's last visit ({latest_train.date()}) falls after a "
        f"held-out person's last visit ({earliest_held.date()}), so the model "
        f"trains on rows from after the rows it is scored on — which is the "
        f"defect GUIDED-143 was filed for")


def test_whole_people_are_kept_together():
    """The grain violation a chronological split would otherwise trade for.

    Not a secondary property: `IMPORT-020`'s asymmetry says leaking behind a
    lock icon is worse than refusing, and a person on both sides of a sealed
    boundary is precisely that.
    """
    df = _frame()
    got = engine.draw_holdout(df, "sbp", "regression", GRAIN, fraction=0.25,
                              time_col="visit_date", temporal=True)
    held = set(got["labels"])
    on_both = {p for p in df["subject_id"].unique()
               if any(i in held for i in df.index[df.subject_id == p])
               and any(i not in held for i in df.index[df.subject_id == p])}
    assert not on_both, (
        f"{len(on_both)} people have rows on both sides of the seal: "
        f"{sorted(on_both)[:5]}")


def test_ordering_is_by_last_observation_and_not_by_first():
    """Pinned with a constructed frame, because the two orderings agree on
    most real data and disagree exactly where it matters.

    `EARLY` enrolls first and is followed longest; `LATE` enrolls last and is
    seen once. Ordering by FIRST visit holds out `LATE` and trains on `EARLY`'s
    2024 rows to predict `LATE`'s 2023 row.
    """
    df = pd.DataFrame({
        "pid": ["EARLY"] * 3 + ["MID"] * 3 + ["LATE"] * 3,
        "t": ["2023-01-01", "2023-06-01", "2024-06-01",
              "2023-03-01", "2023-07-01", "2023-09-01",
              "2023-05-01", "2023-05-15", "2023-06-01"],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] * 1,
    })
    df = pd.concat([df] * 4, ignore_index=True)          # enough rows to seal
    got = engine.draw_holdout(df, "y", "regression",
                              {"basis": "grouped", "group_col": "pid"},
                              fraction=0.34, time_col="t", temporal=True)
    held_people = set(df.loc[sorted(set(got["labels"])), "pid"])
    assert held_people == {"EARLY"}, (
        f"held out {held_people}; ordering by last observation makes EARLY the "
        f"latest person because its follow-up runs to 2024. Ordering by first "
        f"visit would hold out LATE and train on rows from after it.")


def test_the_disclosure_says_what_was_drawn_rather_than_what_was_asked():
    """The seal states its own basis, and the basis comes from the draw.

    The two disagreeing is the whole of `GUIDED-143`: the record said
    `chronological_grouped` because the question was answered yes, while
    `draw_holdout` never read the answer.
    """
    df = _frame()
    got = engine.draw_holdout(df, "sbp", "regression", GRAIN, fraction=0.25,
                              time_col="visit_date", temporal=True)
    d = got["disclosure"]
    assert d["chronological"] is True
    assert d["time_col"] == "visit_date"
    assert d["boundary"], "the seal cannot say where the split falls"
    assert d["n_test_groups"] and d["n_test_groups"] > 1
    assert d["exploratory"] is False

    random_draw = engine.draw_holdout(df, "sbp", "regression", GRAIN,
                                      fraction=0.25)["disclosure"]
    assert random_draw["chronological"] is False, (
        "a draw that was not chronological reports that it was")
    assert random_draw["time_col"] is None


# ═══════════ IT REFUSES RATHER THAN FALLING BACK ═══════════

@pytest.mark.parametrize("time_col,grain,fragment", [
    ("nope",       GRAIN, "No column named 'nope'"),
    ("sex",        GRAIN, "none of its values parse as a date"),
    ("visit_date", {"basis": "cross_sectional", "group_col": None},
     "needs to know who a row belongs to"),
])
def test_a_named_time_column_it_cannot_use_is_refused(time_col, grain, fragment):
    """**Named and unusable refuses.** The user told the app which column is
    time; drawing at random anyway and reporting a chronology would be the
    false assertion L42 removed, arriving through the splitter."""
    with pytest.raises(engine.EngineRefusal) as caught:
        engine.draw_holdout(_frame(), "sbp", "regression", grain,
                            fraction=0.25, time_col=time_col, temporal=True)
    assert fragment in str(caught.value), str(caught.value)


def test_mostly_unreadable_dates_are_refused_rather_than_sorted_around():
    """A column that parses for a few rows is worse than one that parses for
    none: ordering people by their last observation would put the undated ones
    somewhere arbitrary and the seal would report a chronology it did not draw.
    """
    df = _frame()
    df.loc[df.index[: int(len(df) * 0.4)], "visit_date"] = "not a date"
    with pytest.raises(engine.EngineRefusal) as caught:
        engine.draw_holdout(df, "sbp", "regression", GRAIN, fraction=0.25,
                            time_col="visit_date", temporal=True)
    assert "no readable date" in str(caught.value)


def test_no_time_column_named_discloses_rather_than_refusing():
    """**The correction, and it is the load-bearing one in this file.**

    An earlier version refused here. That is stronger than the rule asks and
    it shortens the shelf: L42 built the honest three-state disclosure for
    exactly this case, it is a *loud* fallback rather than a silent one, and
    it was accepted. A user with no clean date column must still be able to
    seal.
    """
    df = _frame()
    got = engine.draw_holdout(df, "sbp", "regression", GRAIN, fraction=0.25,
                              temporal=True)                # no time_col
    assert got["labels"], "the seal refused where it should have disclosed"
    assert got["disclosure"]["chronological"] is False, (
        "a random draw reports itself chronological, which is the silent "
        "fallback this row exists to prevent")

    basis = R.split_strategy(True, R.UNIT_RECORD, time_col=None)
    assert basis["strategy"] == R.CHRONOLOGICAL_NOT_DRAWN
    assert basis["honored"] is False
    assert "not drawn that way" in basis["sentence"]


# ═══════════ DRIVEN, END TO END, THROUGH THE REAL ROUTES ═══════════

@pytest.fixture(scope="module")
def sealed():
    """The whole journey through the API, twice: once with a time column and
    once without. Driven rather than constructed — trap 5."""
    out = {}
    for label, with_column in (("with a time column", True),
                               ("without one", False)):
        client = TestClient(api.app)
        with open(DATA / FIXTURE, "rb") as fh:
            pid = client.post("/project", files={
                "file": (FIXTURE, fh, "text/csv")}).json()["id"]

        def decide(kind, payload):
            return client.post(f"/project/{pid}/decision",
                               json={"kind": kind, "payload": payload})

        for kind, payload in (
                ("set_target", {"column": "sbp"}),
                ("set_purpose", {"answer": "prediction"}),
                ("set_grain", {"answer": "people_repeat",
                               "group_col": "subject_id"}),
                ("set_repeat_kind", {"kind": "time_points"}),
                ("set_unit_of_analysis", {"unit": "record"})):
            r = decide(kind, payload)
            assert r.status_code == 200, (kind, r.text[:250])
        if with_column:
            r = decide("set_time_column", {"column": "visit_date"})
            assert r.status_code == 200, r.text[:250]
        assert decide("set_temporal_prediction", {"temporal": True}) \
            .status_code == 200
        assert decide("set_eligibility", {"answer": "everyone"}) \
            .status_code == 200
        sealing = decide("seal", {"fraction": 0.25})
        out[label] = (client, pid, sealing)
    return out


def test_the_recorded_objective_reaches_the_draw(sealed):
    """`GUIDED-143`'s mechanism, closed. `repeats.split_strategy` had exactly
    one caller — the setter that wrote the sentence — and `draw_holdout` never
    took the answer as an argument."""
    client, pid, sealing = sealed["with a time column"]
    assert sealing.status_code == 200, sealing.text[:300]
    lockbox = client.get(f"/project/{pid}").json()["lockbox"]
    assert lockbox["temporal_basis"] == R.CHRONOLOGICAL_GROUPED
    assert lockbox["temporal_honored"] is True
    assert lockbox["n_test_groups"], "no groups were held out"


def test_without_a_time_column_the_seal_still_happens_and_says_so(sealed):
    """The shelf is not shortened by the build."""
    client, pid, sealing = sealed["without one"]
    assert sealing.status_code == 200, sealing.text[:300]
    lockbox = client.get(f"/project/{pid}").json()["lockbox"]
    assert lockbox["temporal_basis"] == R.CHRONOLOGICAL_NOT_DRAWN
    assert lockbox["temporal_honored"] is False
    assert "not drawn that way" in (lockbox.get("temporal_sentence") or "")


def test_the_time_column_is_asked_and_never_inferred(sealed):
    """The constitution's own answer for grain, applied here.

    `repeats._date_columns` can tell which columns parse as dates — it is used
    to OFFER candidates — but this table carries one date column and a real one
    could carry an enrollment date, a visit date and a lab-draw date. Which one
    the outcome comes *after* is a domain fact.
    """
    client, pid, _ = sealed["with a time column"]
    decisions = client.get(f"/project/{pid}").json()["decisions"]
    kinds = [d["kind"] for d in decisions]
    assert "set_time_column" in kinds, (
        "the time column reached the draw without being recorded as a "
        "decision, so nothing in the transcript says the user chose it")
    recorded = next(d for d in decisions if d["kind"] == "set_time_column")
    assert recorded["subject"] == "visit_date"


def test_a_date_column_that_does_not_parse_is_refused_at_the_question():
    """The refusal lands where the user can act on it — at the question —
    rather than four steps later at the seal.

    On a FRESH project deliberately: the sealed fixtures refuse for the
    barrier instead, and a test that accepted either message would pass on
    the wrong refusal.
    """
    client = TestClient(api.app)
    with open(DATA / FIXTURE, "rb") as fh:
        pid = client.post("/project", files={
            "file": (FIXTURE, fh, "text/csv")}).json()["id"]
    r = client.post(f"/project/{pid}/decision", json={
        "kind": "set_time_column", "payload": {"column": "sex"}})
    assert r.status_code == 400
    assert "parse as a date" in r.json()["detail"], r.json()["detail"][:200]

    missing = client.post(f"/project/{pid}/decision", json={
        "kind": "set_time_column", "payload": {"column": "no_such_column"}})
    assert missing.status_code == 400
    assert "No column named" in missing.json()["detail"]


def test_it_is_refused_after_the_seal(sealed):
    """`set_temporal_prediction`'s own precedent: after the seal the split was
    drawn under what was recorded then, and changing this would describe a
    chronology that was not drawn."""
    client, pid, _ = sealed["with a time column"]
    r = client.post(f"/project/{pid}/decision", json={
        "kind": "set_time_column", "payload": {"column": "visit_date"}})
    assert r.status_code == 400
    assert "already sealed" in r.json()["detail"]


def test_the_pre_seal_order_is_untouched(sealed):
    """Constitution §01 is fixed, and a build that reordered the opening
    sequence to fit a new question would be the kind of change this project
    refuses. The time-column question sits inside the repeated-measures chain
    that already exists; it adds no step before the seal that was not already
    reachable."""
    client, pid, _ = sealed["with a time column"]
    kinds = [d["kind"] for d in client.get(f"/project/{pid}").json()["decisions"]]
    for earlier, later in (("set_target", "set_grain"),
                           ("set_grain", "set_repeat_kind"),
                           ("set_repeat_kind", "set_unit_of_analysis"),
                           ("set_eligibility", "seal_lockbox")):
        assert kinds.index(earlier) < kinds.index(later), (
            f"{earlier} no longer precedes {later}")
    assert kinds.index("set_time_column") < kinds.index("seal_lockbox")
    assert kinds.index("set_time_column") > kinds.index("set_unit_of_analysis"), (
        "the time-column question moved ahead of the chain that decides "
        "whether it applies at all")


def test_the_seal_reports_the_draw_and_not_the_answer(sealed):
    """**A revert probe found this, and it is the sharpest thing in the file.**

    L42's seal wrote `temporal_basis` from `temporal_prediction` — the record
    of what was *asked* — because at the time nothing else could say. When
    L43-C removed the `temporal=` argument from the seal's call to
    `draw_holdout` as a probe, the draw went back to random and **the lockbox
    still reported `chronological_grouped`**.

    That is `GUIDED-143`'s own defect, reintroduced by the fix for it, and the
    check for it came back `GREEN — NOT LOAD-BEARING` because it was reading
    the answer rather than the draw.

    So the disclosure outranks the answer, and this pins it: a lockbox that
    says a chronological split was drawn must have a draw that says so too.
    """
    client, pid, _ = sealed["with a time column"]
    lockbox = client.get(f"/project/{pid}").json()["lockbox"]
    assert lockbox["temporal_drawn"] is True, (
        "the seal reports a chronological basis and the draw did not report "
        "drawing one — the record and the draw have come apart again")
    assert lockbox["temporal_basis"] == R.CHRONOLOGICAL_GROUPED

    without, wpid, _ = sealed["without one"]
    lb2 = without.get(f"/project/{wpid}").json()["lockbox"]
    assert lb2["temporal_drawn"] is False
    assert lb2["temporal_basis"] == R.CHRONOLOGICAL_NOT_DRAWN
    assert lb2["temporal_honored"] is False


def test_an_honored_answer_over_a_random_draw_is_corrected_to_not_drawn():
    """The reconciliation itself, at the unit, so the rule is visible rather
    than only observable through a journey.

    If the answer says honored and the draw says it drew at random, the seal
    reports **not drawn**. It never reports the answer's optimistic version:
    the seal's job is to state what it rests on.
    """
    from turbotab.project import AnalysisProject

    df = _frame()
    project = AnalysisProject.from_dataframe(df, "reconcile")
    project.set_target("sbp", "regression", "high", ["numeric"])
    project.set_grain("people_repeat", "subject_id")
    project.set_repeat_kind("time_points")
    project.set_unit_of_analysis("record")
    project.set_time_column("visit_date")
    project.set_temporal_prediction(True)
    assert project.temporal_prediction["honored"] is True, (
        "the fixture does not reach the state this test is about")
    project.set_eligibility("everyone")

    project.seal_lockbox(list(df.index[:100]), fraction=0.16,
                         fraction_requested=0.16, seed=42, n_total=600,
                         n_test_groups=30, exploratory=False,
                         chronological=False, time_col=None,
                         n_undated_groups=0, boundary=None)
    assert project.lockbox["temporal_basis"] == R.CHRONOLOGICAL_NOT_DRAWN
    assert project.lockbox["temporal_honored"] is False
    assert "not drawn that way" in project.lockbox["temporal_sentence"]
