"""The sweep's arithmetic and its classing (SWARM_PLAN §6, §7; ``research/swarm/sweep.py``), each
from construction: two fit records built here with numpy, their 5 × 5 fold scores known, so the
refit's materiality is checked against package 1's signed change (``signed.change``) and against a
recomputation of the Nadeau–Bengio corrected interval by numpy and scipy, paired and unpaired,
material and not, and constructed so each of §7's two conditions and its half are needed; the
Estimate shift by arithmetic; each refit's class both ways, among them a fit whose record the
server serves only on a later read (waited for, scored) or never (unreached after the wait, on a
clock run by hand), and what the walk meets after an option (a stage in error, typed or not, and
an answer refused with no way forward) on a session built here; a refit's wait, both ways, on a
session built here and a clock run by hand: a fit still running when the wait runs out, or one
done whose other stages are not by then, cancelled and unreached in words that name what it
waited on and for how long, each job the copy still holds cancelled where it does not go idle,
and a fit blocked or held for a press never cancelled; and after a fitted refit, an estimate
stage in error untyped (an engine defect) or typed (refused, with its code), and none in error
(scored), the plan's own refit read the same; the crash sweep's previews after
an option, of every later decision's plan answer (a question's, a finding's, a reading's, a
declaration's) and of a question opened again at its default, which decisions they reach and
that an HTTP 500 induced on one fails the refit as a defect, while one on a decision before the
option does not; a revisit the engine refuses until
the answers that read the column are withdrawn, those answers given again in the plan's order on a
log built here and folded; a refusal's way forward after an option, in its answer or served on the
asks after it, taken only where the plan or the option names it; the decisions a plan offers the
sweep and those it does not, from a decision log built here; what the report counts as not visited
and what it says it swept; the readings graded against a declared truth; the run times' summary by
hand; and the options that take fields of their own: the card's rules read back from the intake's
own sentences and from the pilot cards, each answer resolved by the driver's composer, each value
built from a table here checked against the table, every builder both ways, none on the outcome
or the declared exposure, and the crash sweep's answer of one, named by its source and superseding
the unreached line of a report finished before. A revisit refused because the split is drawn
(``sealed``) taken through the way its refusal names, both splits withdrawn and drawn again in the
plan's order on a log built here and folded, visited and refused where that way is refused, and
unlisted where it names none or none that is a split in force. Around an Estimate plan, named or
the crash sweep's, each column of the plan's adjustment answers in force (its records merged by
column on a log built here and folded by the engine, a column of an earlier record than its
latest among them) and each the question offers that the plan answered, toggled in or out by the
modified disjunctive cause criterion with the plan's other answers in force fixed, previewed and
posted as composed; none under Predict, for another question or without the plan's answers, and
none of a column the plan never answered or the question no longer takes. The engine recorded as the sweep starts and ends:
one engine (a commit that leaves the tree as it was among them) can pass, another tree or
modified files at the end refuse the report in words, as does a resumed run on another engine
or progress holding work of a run that recorded none. A refit scored as the plan in every fold
is not material, and one whose folds all moved by the same amount, or on other folds, is not read.
The sweep on a real engine is ``test_sweep_engine.py``'s."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from research.swarm import arms, signed as S, sweep as W

N_TRAIN = 240  # five folds of 48: n_test/n_train is 48/192 = 1/4 in every fold
SPLIT = {"n_train": N_TRAIN, "folds": 5, "repeats": 5, "fold_scheme": "random",
         "validation": "repeated_kfold", "grouped_by": None}
METRIC = "log_loss"  # a loss: lower is better


def record(folds, *, digest: str, fid: str, low: float, high: float) -> dict:
    """A fit record as ``GET …/fits`` serves it, one family scored on 5 × 5 folds."""
    return {"id": fid, "purpose": "prediction", "primary_metric": METRIC, "folds": digest,
            "scores": {"linear": {METRIC: {"folds": list(map(float, folds)), "repeats": 5,
                                           "estimator": "fold_mean", "mean": float(np.mean(folds)),
                                           "ci_low": low, "ci_high": high}}}}


def scored(point: str, rec: dict, split=SPLIT) -> W.Refit:
    return W.Refit(point, "x", {}, kind="scored", record=rec, split=dict(split))


def draws(shift: float, seed: int = 11, noise: float = 0.01):
    """The plan's 25 fold scores and a refit's, the refit's each the plan's plus ``shift`` and
    noise of its own (numpy, ``seed``)."""
    rng = np.random.default_rng(seed)
    before = rng.normal(0.65, 0.03, 25)
    return before, before + shift + rng.normal(0.0, noise, 25)


def by_numpy_paired(after, before, share: float, level: float = 0.95):
    d = np.asarray(after) - np.asarray(before)
    j = d.size
    se = math.sqrt((1 / j + share) * d.var(ddof=1))
    q = stats.t.ppf(0.5 + level / 2, j - 1)
    return d.mean(), d.mean() - q * se, d.mean() + q * se


def by_numpy_welch(after, before, share: float, level: float = 0.95):
    a, b = np.asarray(after), np.asarray(before)
    va, vb = (1 / a.size + share) * a.var(ddof=1), (1 / b.size + share) * b.var(ddof=1)
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    q = stats.t.ppf(0.5 + level / 2, df)
    diff = a.mean() - b.mean()
    return diff, diff - q * math.sqrt(va + vb), diff + q * math.sqrt(va + vb)


SHARE = 48 / 192  # each fold's rows scored over the rows it is fit on, by construction


@pytest.mark.parametrize("shift, material", [(0.04, True), (0.0005, False)])
def test_a_paired_refit_is_package_1s_signed_change_and_the_corrected_interval_by_numpy(
        shift, material):
    """Same folds (one digest): the paired change; material where its interval excludes zero and
    it is at least half the plan's half-width (0.02 here: the plan's interval 0.63 to 0.67)."""
    before, after = draws(shift)
    plan = scored(W.PLAN, record(before, digest="d0", fid="f0", low=0.63, high=0.67))
    refit = scored("question:missing", record(after, digest="d0", fid="f1", low=0.0, high=1.0))
    W.materiality(refit, plan, "prediction")
    (change,) = refit.change
    package_1 = S.change(refit.record["scores"]["linear"][METRIC],
                         plan.record["scores"]["linear"][METRIC], same_folds=True,
                         higher_is_better=False, metric=METRIC, after_split=SPLIT)
    for key in ("difference", "se", "df", "ci_low", "ci_high", "excludes_zero", "half_width",
                "better"):
        assert change[key] == package_1[key], key
    diff, low, high = by_numpy_paired(after, before, SHARE)
    assert change["paired"] is True and change["test_share"] == pytest.approx([SHARE, SHARE])
    assert (change["difference"], change["ci_low"], change["ci_high"]) == pytest.approx(
        (diff, low, high), rel=1e-12, abs=1e-15)
    assert change["half_width"] == pytest.approx(0.02)
    by_rule = bool((low > 0 or high < 0) and abs(diff) >= 0.5 * 0.02)
    assert by_rule is material and change["material"] is material and refit.material is material


@pytest.mark.parametrize("shift, material", [(0.1, True), (0.005, False)])
def test_a_refit_on_other_rows_is_the_unpaired_change_with_each_arms_inflation_by_numpy(
        shift, material):
    """Other folds (an exclusion: another digest, 200 training rows): Welch's interval over the
    two arms, each arm's variance inflated by its own share (40/160 and 48/192, both 1/4);
    material both ways, by the rule recomputed here."""
    before, after = draws(shift, seed=3, noise=0.02)
    other = dict(SPLIT, n_train=200)
    plan = scored(W.PLAN, record(before, digest="d0", fid="f0", low=0.63, high=0.67))
    refit = scored("question:exclusions", record(after, digest="d1", fid="f1", low=0.0,
                                                 high=1.0), split=other)
    W.materiality(refit, plan, "prediction")
    (change,) = refit.change
    package_1 = S.change(refit.record["scores"]["linear"][METRIC],
                         plan.record["scores"]["linear"][METRIC], same_folds=False,
                         higher_is_better=False, metric=METRIC, after_split=other,
                         before_split=SPLIT)
    assert change["paired"] is False and change["difference"] == package_1["difference"]
    diff, low, high = by_numpy_welch(after, before, 0.25)
    assert (change["difference"], change["ci_low"], change["ci_high"]) == pytest.approx(
        (diff, low, high), rel=1e-12)
    by_rule = bool((low > 0 or high < 0) and abs(diff) >= 0.5 * 0.02)
    assert by_rule is material and change["material"] is material


def test_a_refit_scored_the_same_as_the_plan_in_every_fold_is_not_material():
    """The same scores in every fold, on the same folds (an answer the fit does not read): no
    spread, so package 1's change has no number and says why, but the change is zero by
    construction, so §7 reads it not material (F15), and so does the decision."""
    before, _ = draws(0.0)
    plan = scored(W.PLAN, record(before, digest="d0", fid="f0", low=0.63, high=0.67))
    refit = scored("question:missing", record(before, digest="d0", fid="f1", low=0.63,
                                              high=0.67))
    W.materiality(refit, plan, "prediction")
    (change,) = refit.change
    assert change["difference"] is None and "no fold-to-fold spread" in change["why"]
    assert change["material"] is False and refit.material is False
    (decision,) = W.decisions_of([W.Point("question", "missing")], [refit])
    assert decision["material"] is False and decision["material_options"] == []


@pytest.mark.parametrize("before, after, digest", [
    (draws(0.0)[0], draws(0.0)[0] + 0.01, "d0"),
    (np.full(25, 0.65), np.full(25, 0.65), "d1")], ids=["shifted_every_fold", "other_folds"])
def test_a_change_with_no_number_that_is_not_the_plans_scores_again_is_not_read(before, after,
                                                                                 digest):
    """The guard's other side: every fold moved by the same 0.01 (a change, with no spread), or
    the plan's very scores on other folds (another digest: an exclusion's, no fold the plan's;
    both arms one value, so no spread), has no number, and §7 reads none: neither is the plan
    scored again on its own folds."""
    plan = scored(W.PLAN, record(before, digest="d0", fid="f0", low=0.63, high=0.67))
    refit = scored("question:missing", record(after, digest=digest, fid="f1", low=0.63,
                                              high=0.67))
    W.materiality(refit, plan, "prediction")
    (change,) = refit.change
    assert change["difference"] is None and change["why"]
    assert change["material"] is None and refit.material is None


def test_a_refit_of_the_scheme_moves_no_model_and_is_read_so():
    before, after = draws(0.04)
    plan = scored(W.PLAN, record(before, digest="d0", fid="f0", low=0.63, high=0.67))
    refit = scored("declaration:set_validation", record(after, digest="d9", fid="f1", low=0,
                                                        high=1))
    W.materiality(refit, plan, "prediction")
    assert refit.change == [] and refit.change_why == W.SCHEME_IS_THE_SWEEPS
    assert refit.material is None


def estimate(value: float, low: float, high: float, family: str = "linear") -> dict:
    return {"family": family, "feature": "walks", "setting": None, "estimate": value,
            "ci_low": low, "ci_high": high, "why": None}


@pytest.mark.parametrize("value, material", [(-4.2, True), (-5.3, False)])
def test_an_estimates_shift_is_material_at_half_the_plans_half_width(value, material):
    """The plan's estimate −5.0 on [−6.0, −4.0]: half-width 1.0, so a shift of 0.8 is material
    (at least 0.5) and one of 0.3 is not."""
    plan = W.Refit(W.PLAN, None, None, kind="scored", estimates=[estimate(-5.0, -6.0, -4.0)])
    refit = W.Refit("question:missing", "x", {}, kind="scored",
                    estimates=[estimate(value, value - 1, value + 1)])
    W.materiality(refit, plan, "inference")
    (row,) = refit.change
    assert row["shift"] == pytest.approx(value + 5.0) and row["half_width"] == pytest.approx(1.0)
    assert row["material"] is material and refit.material is material


def test_an_estimate_the_plan_does_not_report_has_no_shift():
    plan = W.Refit(W.PLAN, None, None, kind="scored", estimates=[estimate(-5.0, -6.0, -4.0)])
    refit = W.Refit("question:models", "x", {}, kind="scored",
                    estimates=[estimate(-4.0, -5.0, -3.0, family="poisson")])
    W.materiality(refit, plan, "inference")
    assert refit.change[0]["shift"] is None and refit.change[0]["why"]
    assert refit.material is None


# ── each refit classed, both ways ────────────────────────────────────────────

FRESH = {"pressed": {}, "fit": {"status": "fresh", "key": "k1"}, "seconds": 1.5}
CRASHED = {"pressed": {}, "fit": {"status": "error", "key": "k2",
                                  "error": "ValueError: at least one array is required"}}
TYPED = {"pressed": {}, "fit": {"status": "error", "key": "k3", "error": "ValueError: none kept",
                                "refusal": {"stage": "fit"}},
         "refused": {"by": "engine", "stage": "fit", "code": "nothing_kept",
                     "message": "none kept",
                     "exits": [{"id": "x-1", "label": "No selection",
                                "decision": {"kind": "set_selection", "method": "none"}}]}}
HELD = {"pressed": {}, "fit": {"status": "queued", "key": "k4", "held": "fit"}}


@pytest.mark.parametrize("code, payload, kind", [
    (0, FRESH, "scored"), (0, CRASHED, "failed"), (1, TYPED, "refused"), (0, HELD, "unreached"),
    (1, {"refused": {"by": "server", "status": 500, "code": "http_500", "message": "boom"}},
     "failed"),
    (1, {"refused": {"by": "server", "status": 409, "code": "fit_not_yet", "message": "w",
                     "exits": []}}, "refused"),
], ids=["fresh", "untyped_error", "typed_refusal", "held", "http_500", "press_refused"])
def test_each_press_of_fit_is_classed_by_what_the_engine_typed(code, payload, kind):
    said = arms.said_of(code, payload)
    got, detail = W.classed(said)
    assert got == kind
    if kind == "refused" and payload.get("refused", {}).get("by") == "engine":
        assert detail["code"] == "nothing_kept" and detail["at"] == "fit"
        assert detail["exits"] == [{"label": "No selection",
                                    "decision": {"kind": "set_selection", "method": "none"}}]
    if kind == "failed":
        assert detail["kind"] == "failed"


class Clock:
    """The sweep's clock, by hand: ``sleep`` moves it on and nothing else does."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Recorded:
    """A session over a fit made at ``k1`` whose record ``GET …/fits`` serves from its ``late``-th
    read on (never, where ``late`` is None), another fit's record before it; no split served."""

    RECORD = {"id": "f2", "fit_key": "k1", "seconds": 1.2, "primary_metric": METRIC, "folds": "d",
              "scores": {"linear": {METRIC: {"mean": 0.5}}}}

    def __init__(self, late):
        self.late, self.reads = late, 0

    def read(self, path):
        if path != "fits":
            return None
        self.reads += 1
        other = {"id": "f1", "fit_key": "k0", "seconds": 0.9}
        come = self.late is not None and self.reads >= self.late
        return {"records": [other, self.RECORD] if come else [other]}

    def view(self):
        return {"stages": {}}


def test_a_fit_record_the_server_writes_after_the_fit_turns_fresh_is_waited_for(monkeypatch):
    """The server writes a fit's record on its recorder's thread after the fit's key turns fresh
    (``service._fit_event``), so the sweep's first read can come before it: the record served on
    the fourth read is the refit's, scored, read as soon as it is served."""
    clock = Clock()
    monkeypatch.setattr(W, "time", clock)
    session, refit = Recorded(late=4), W.Refit("question:missing", "impute", None)
    W.read_refit(session, refit, arms.said_of(0, FRESH), "prediction")
    assert refit.kind == "scored" and refit.why is None
    assert (refit.fit_key, refit.fit_record, refit.fit_seconds) == ("k1", "f2", 1.2)
    assert refit.scores == {"metric": METRIC, "folds": "d",
                            "families": {"linear": {METRIC: {"mean": 0.5}}}}
    assert session.reads == 4 and 0 < clock.now < W.RECORD_WAIT


def test_a_fit_record_that_never_comes_leaves_the_refit_unreached_after_the_wait(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(W, "time", clock)
    session, refit = Recorded(late=None), W.Refit("question:missing", "impute", None)
    W.read_refit(session, refit, arms.said_of(0, FRESH), "prediction")
    assert refit.kind == "unreached" and refit.fit_record is None and refit.scores is None
    assert W.RECORD_WAIT <= clock.now < W.RECORD_WAIT + 1 and session.reads > 2
    assert refit.why == (f"No fit record holds the fit's key: none was served in the "
                         f"{W.RECORD_WAIT:g} s the sweep waited after the fit turned fresh.")


def test_an_ask_whose_fit_line_says_not_pressed_is_no_fit_at_all():
    """The ask's own ``fit`` (pressed, locked) is not a press of Fit: an ask is ok."""
    said = arms.said_of(0, {"question": None, "fit": {"pressed": False, "locked": False}})
    assert said.ok


# ── what is swept, and what is not ───────────────────────────────────────────


def view_of(decisions: list[tuple[str, dict]]) -> dict:
    at = datetime(2026, 10, 22, tzinfo=timezone.utc).isoformat()
    out = []
    for i, (by, d) in enumerate(decisions, start=1):
        out.append({"id": f"r{i}", "seq": i, "at": at, "recorded_by": by, "decision": d})
    return {"decisions": out}


LOG = [
    ("you", {"kind": "set_lens", "lenses": ["other"]}),
    ("you", {"kind": "set_target", "column": "y"}),
    ("you", {"kind": "set_purpose", "purpose": "prediction"}),
    ("you", {"kind": "set_roles", "roles": {"x": "covariate", "age": "covariate"}}),
    ("you", {"kind": "confirm_reading", "reading": "code_or_count", "column": "age",
             "value": "amount"}),
    ("you", {"kind": "set_exclusions", "rules": []}),
    ("you", {"kind": "set_missing", "strategy": "complete_case"}),
    ("you", {"kind": "set_missing", "strategy": "impute"}),  # the latest stands
    ("you", {"kind": "set_split", "holdout": 0.2}),
    ("you", {"kind": "set_selection", "method": "none"}),
    ("you", {"kind": "select_models", "models": ["linear"]}),
    ("you", {"kind": "set_validation", "validation": "repeated_kfold", "folds": 5,
             "repeats": 5}),
]
ASK = {"also_open": [{"source": "declaration", "key": "set_intended_use", "status": "open"},
                     {"source": "finding", "key": "f1", "status": "open"}]}


def test_the_plans_own_answers_are_swept_and_the_cards_and_turbotabs_are_not():
    points, skipped = W.points_of(view_of(LOG), ASK, purpose="prediction", crash=False)
    assert [p.id for p in points] == [
        "question:lens", "question:roles", "question:exclusions", "question:missing",
        "question:split", "declaration:set_selection", "question:models",
        "declaration:set_intended_use"]
    missing = next(p for p in points if p.id == "question:missing")
    assert missing.seq == 8 and missing.decision["strategy"] == "impute"
    why = {s.get("id") or s.get("kind"): s["why"] for s in skipped}
    assert why["question:target"].startswith("The card fixes it")
    assert why["question:purpose"].startswith("The card fixes it")
    assert why["reading:code_or_count:age"].startswith("A reading with one true answer")
    assert why["declaration:set_validation"] == W.SCHEME_IS_THE_SWEEPS


def test_the_crash_sweep_takes_the_readings_and_the_scheme_too():
    points, skipped = W.points_of(view_of(LOG), ASK, purpose="prediction", crash=True)
    ids = [p.id for p in points]
    assert "reading:code_or_count:age" in ids and "declaration:set_validation" in ids
    assert "question:target" not in ids


def test_a_record_turbotab_made_itself_is_not_swept():
    log = [*LOG[:3], ("turbotab", {"kind": "set_split", "holdout": 0.0})]
    points, skipped = W.points_of(view_of(log), {}, purpose="inference", crash=True)
    assert "question:split" not in [p.id for p in points]
    assert any(s.get("id") == "question:split" and s["why"] == "TurboTab recorded it itself."
               for s in skipped)


def test_under_estimate_the_estimand_is_the_cards():
    log = [*LOG[:3], ("you", {"kind": "set_estimand", "exposure": "x", "measure":
                              "mean_difference"})]
    points, skipped = W.points_of(view_of(log), {}, purpose="inference", crash=False)
    assert "question:estimand" not in [p.id for p in points]


def test_the_readings_are_graded_against_the_declared_truth_and_none_otherwise():
    log = [("you", {"kind": "confirm_reading", "reading": "code_or_count", "column": "age",
                    "value": "amount"}),
           ("you", {"kind": "confirm_reading", "reading": "code_or_count", "column": "sex",
                    "value": "amount"}),
           ("you", {"kind": "confirm_reading", "reading": "code_or_count", "column": "bmi",
                    "value": "amount"}),
           ("you", {"kind": "confirm_reading", "reading": "role", "column": "x",
                    "value": "covariate"})]
    truth = {"code_or_count:age": "amount", "code_or_count:sex": "code"}
    got = {g["reading"]: g["right"] for g in W.graded(view_of(log), truth)}
    assert got == {"code_or_count:age": True, "code_or_count:sex": False,
                   "code_or_count:bmi": None}


def test_a_datasets_declared_readings_are_read_from_its_answers_file():
    import json

    answers = json.loads((arms.DATASETS / "clinical_readmission_predict" / "answers.json")
                         .read_text("utf-8"))
    declared = next(t["value"] for t in answers["truths"]
                    if t["name"].startswith("the readings"))
    assert W.readings_truth("clinical_readmission_predict") == {k: str(v)
                                                                 for k, v in declared.items()}
    assert W.readings_truth("no_such_dataset") == {}


def test_the_run_times_are_summed_with_their_median_ninetieth_percentile_and_largest():
    got = W.seconds_of([3.0, 1.0, 2.0, 10.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    # Nearest rank: the 9th of 10 sorted (ceil(0.9 · 10) = 9) is 9.
    assert got == {"fits": 10, "total": 55.0, "median": 5.5, "p90": 9.0, "max": 10.0}
    assert W.seconds_of([]) == {"fits": 0, "total": 0.0, "median": None, "p90": None,
                                "max": None}


@pytest.mark.parametrize("refit, expect", [
    (W.Refit("p", "o", {}, kind="scored", seconds=0.01, fit_seconds=7.0), 7.0),
    (W.Refit("p", "o", {}, kind="refused", seconds=2.5), 2.5),
    (W.Refit("p", "o", {}, kind="duplicate", seconds=None), None),
    (W.Refit("p", "o", {}, kind="refused", seconds=None), None),
])
def test_a_fits_run_time_is_the_servers_where_it_kept_one(refit, expect):
    assert W.timed_of(refit) == expect


def test_each_decisions_refused_share_and_codes_sit_beside_its_materiality():
    p = W.Point("question", "missing")
    refits = [W.Refit(p.id, "a", {}, kind="scored", material=True),
              W.Refit(p.id, "b", {}, kind="refused", refused={"code": "c1"}),
              W.Refit(p.id, "c", {}, kind="refused", refused={"code": "c1"}),
              W.Refit(p.id, "d", {}, kind="duplicate"),
              W.Refit(p.id, "e", None, kind="unreached")]
    (got,) = W.decisions_of([p], refits)
    assert got["refused_share"] == pytest.approx(2 / 3)  # of the three refitted
    assert got["refused_codes"] == {"c1": 2} and got["refused_options"] == ["b", "c"]
    assert got["material"] is True and got["material_options"] == ["a"]
    assert got["classes"] == {"scored": 1, "refused": 2, "failed": 0, "unreached": 1,
                              "duplicate": 1}


# ── the scoring, both ways ───────────────────────────────────────────────────


class ScoringSession:
    """A session whose split is ``split`` and whose split stage serves ``facts``, fresh."""

    def __init__(self, split, facts):
        self.split, self.facts, self.decided = split, facts, []

    def view(self):
        return {"state": {"split": self.split}, "stages": {"split": {"key": "s1"}}}

    def read(self, path):
        return {"fresh": True, "key": "s1", "artifact": self.facts} if path == "stages/split" \
            else None

    def decide(self, decision, reason):
        self.decided.append(dict(decision))
        return arms.Said("ok", {})


def scoring(split, facts):
    s = ScoringSession(split, facts)
    sweep = W.Sweep(Path("/nonexistent/sweep"), dataset=None, question=arms.Question(
        target="y", purpose="prediction"), crash=False, plan="p")
    return sweep._scoring(s), s.decided


def test_the_scoring_answers_a_cut_points_folds_order_random():
    got, decided = scoring({"holdout": 0.0, "cut_point": {"column": "cycle", "at": 2017}},
                           dict(SPLIT, n_train=180))
    assert decided == [{**W.SCORING, "fold_order": "random"}]
    assert got["set"] is True and "why" not in got


def test_the_scoring_says_where_the_held_out_draw_orders_the_folds_by_time():
    got, decided = scoring({"holdout": 0.2}, dict(SPLIT, fold_scheme="time_ordered"))
    assert decided == [dict(W.SCORING)]  # no folds' order beside a share
    assert got["set"] is True and "orders the folds by time" in got["why"]


# ── §7's rule, each of its conditions and its half pinned ────────────────────


def centered(shift: float, spread: float, noise: float, seed: int = 5):
    """The plan's 25 fold scores (spread ``spread`` about 0.65) and a refit's, each the plan's
    plus ``shift`` and noise of its own centered on zero (numpy, ``seed``), so the mean change is
    ``shift`` by construction."""
    rng = np.random.default_rng(seed)
    before = rng.normal(0.65, spread, 25)
    e = rng.normal(0.0, noise, 25)
    return before, before + shift + (e - e.mean())


# The plan's interval is 0.63 to 0.67, a half-width of 0.02, so half of it is 0.01.
SEPARATING = {
    # The interval excludes zero, the change is under half the half-width: not material.
    "excludes_zero_under_half": (0.0095, 0.001, 0.0005, False),
    # The change is at least half the half-width, the interval includes zero: not material.
    "half_or_more_includes_zero": (0.015, 0.03, 0.05, False),
    # The interval excludes zero and the change is just over half the half-width, under the whole
    # of it: material at half, as it would not be at the whole half-width.
    "excludes_zero_just_over_half": (0.0105, 0.001, 0.0005, True),
}


@pytest.mark.parametrize("paired", [True, False], ids=["paired", "unpaired"])
@pytest.mark.parametrize("case", list(SEPARATING))
def test_material_needs_both_the_interval_and_half_the_plans_half_width(case, paired):
    """Each case meets one of §7's two conditions and not the other, or meets both a little past
    half the plan's half-width: its premise checked by numpy and scipy, its reading by the rule."""
    shift, spread, noise, material = SEPARATING[case]
    before, after = centered(shift, spread, noise)
    other = SPLIT if paired else dict(SPLIT, n_train=200)
    plan = scored(W.PLAN, record(before, digest="d0", fid="f0", low=0.63, high=0.67))
    refit = scored("question:missing" if paired else "question:exclusions",
                   record(after, digest="d0" if paired else "d1", fid="f1", low=0.0, high=1.0),
                   split=other)
    W.materiality(refit, plan, "prediction")
    (change,) = refit.change
    diff, low, high = by_numpy_paired(after, before, SHARE) if paired else \
        by_numpy_welch(after, before, 0.25)
    assert change["paired"] is paired and change["half_width"] == pytest.approx(0.02)
    assert (change["difference"], change["ci_low"], change["ci_high"]) == pytest.approx(
        (diff, low, high), rel=1e-12, abs=1e-15)
    excludes = bool(low > 0 or high < 0)
    half = 0.5 * 0.02
    if case == "excludes_zero_under_half":
        assert excludes and abs(diff) < half
    elif case == "half_or_more_includes_zero":
        assert not excludes and abs(diff) >= half
    else:
        assert excludes and half <= abs(diff) < 2 * half
    assert change["material"] is material and refit.material is material


# ── the crash sweep's report: what was not visited, what was swept ───────────


def a_sweep(tmp_path, *, only=None):
    """A sweep around a plan with no decision, its start recorded on the engine as it is (a
    report read without a run of it would hold no start, and pass nothing)."""
    sweep = W.Sweep(tmp_path / "sweep", dataset="tiny", question=arms.Question(
        target="y", purpose="prediction"), crash=True, plan="default", only=only)
    sweep.base = {"project": "p0", "digest": "d0", "seq": 9, "scoring": {"set": False},
                  "points": [], "skipped": []}
    sweep.engines.append({**W.engine_version(), "at": W.now()})
    return sweep


ANSWERED = W.Point("question", "missing", 4, "r4", {"kind": "set_missing",
                                                     "strategy": "impute"})
READING = W.Point("reading", "unit:age", 5, "r5", {"kind": "set_column_unit", "column": "age",
                                                    "unit": "years"})
OPEN_LINE = W.Point("declaration", "set_intended_use")  # a line the plan left open


def listed(point, options=(), why=None):
    return {"point": point.id, "why": why,
            "options": [{"id": o, "label": o, "decision": {"kind": "k"}, "needs": None}
                        for o in options]}


def test_a_decision_whose_options_could_not_be_listed_is_not_visited(tmp_path):
    sweep = a_sweep(tmp_path)
    sweep.points = [ANSWERED, READING, OPEN_LINE]
    sweep.offered = {
        ANSWERED.id: listed(ANSWERED, why="The missing question did not open again."),
        READING.id: listed(READING),  # none offered, though the plan answered it
        OPEN_LINE.id: listed(OPEN_LINE)}  # none offered, and none answered: nothing to visit
    report = sweep.report(W.Refit(W.PLAN, None, None, kind="scored"), 0.0)
    assert report["unvisited"] == [ANSWERED.id, READING.id] and report["complete"] is False
    assert [u["point"] for u in report["unlisted"]] == [ANSWERED.id, READING.id]
    assert report["unlisted"][0]["why"] == "The missing question did not open again."
    assert W.passed(report) is False


def test_a_sweep_that_listed_and_visited_every_option_is_complete(tmp_path):
    sweep = a_sweep(tmp_path)
    sweep.points = [ANSWERED, READING, OPEN_LINE]
    sweep.offered = {ANSWERED.id: listed(ANSWERED, ["impute", "complete_case"]),
                     READING.id: listed(READING, ["x-1"]), OPEN_LINE.id: listed(OPEN_LINE)}
    sweep.done = {f"{p}={o}": W.Refit(p, o, {}, kind="scored") for p, o in [
        (ANSWERED.id, "impute"), (ANSWERED.id, "complete_case"), (READING.id, "x-1")]}
    report = sweep.report(W.Refit(W.PLAN, None, None, kind="scored"), 0.0)
    assert report["unvisited"] == [] and report["unlisted"] == [] and report["complete"] is True
    assert report["only"] is None and W.passed(report) is True
    # Not yet listed: not visited.
    sweep.offered.pop(READING.id)
    assert sweep.report(W.Refit(W.PLAN, None, None, kind="scored"), 0.0)["unvisited"] == [
        READING.id]


def test_a_sweep_of_some_decisions_says_so_and_does_not_pass(tmp_path):
    sweep = a_sweep(tmp_path, only=[ANSWERED.id])
    sweep.points = [ANSWERED]
    sweep.offered = {ANSWERED.id: listed(ANSWERED, ["impute"])}
    sweep.done = {f"{ANSWERED.id}=impute": W.Refit(ANSWERED.id, "impute", {}, kind="scored")}
    report = sweep.report(W.Refit(W.PLAN, None, None, kind="scored"), 0.0)
    assert report["only"] == [ANSWERED.id] and report["complete"] is True
    assert W.passed(report) is False
    assert "swept only question:missing" in W.table_md(report | {"passed": False})
    assert W.passed({k: v for k, v in report.items() if k != "only"} | {"complete": True}) \
        is False  # a report that does not say what it swept
    assert W.passed(report | {"only": None, "defects": [{"refit": "x"}]}) is False


# ── the engine a sweep ran on, recorded when it starts and when it ends (F10) ──

ENGINE_1 = {"commit": "c1", "engine_tree": "t1", "engine_modified": False}


def served_engines(monkeypatch, *engines):
    """``sweep.engine_version`` serving ``engines`` in turn (the last one again once they are
    served): what git would read at the sweep's start, then at its end."""
    left = [dict(e) for e in engines]
    monkeypatch.setattr(W, "engine_version", lambda: dict(left.pop(0) if len(left) > 1
                                                          else left[0]))


def ran(tmp_path, monkeypatch, *engines, folder="sweep"):
    """A sweep around a plan with no decision, its plan already scored, run once with the engine
    read as ``engines`` serve it: its report."""
    served_engines(monkeypatch, *engines)
    sweep = W.Sweep(tmp_path / folder, dataset="tiny", question=arms.Question(
        target="y", purpose="prediction"), crash=True, plan="default")
    if sweep.base is None:  # the template and the plan's refit as a run would have kept them
        sweep.base = {"project": "p0", "digest": "d0", "seq": 9, "scoring": {"set": False},
                      "points": [], "skipped": []}
    sweep.done.setdefault(W.PLAN, W.Refit(W.PLAN, None, None, kind="scored"))
    return sweep.run(object(), lambda s: "p0")


@pytest.mark.parametrize("end", [
    ENGINE_1,
    {**ENGINE_1, "commit": "c2"},  # another commit that leaves turbotab/ as it was: one engine
], ids=["same", "same_tree_other_commit"])
def test_a_sweep_that_starts_and_ends_on_one_engine_names_both_and_can_pass(tmp_path,
                                                                            monkeypatch, end):
    report = ran(tmp_path, monkeypatch, ENGINE_1, end)
    assert report["engine_changed"] is None and W.passed(report) is True
    (start,) = report["engine_starts"]
    assert {k: start[k] for k in ENGINE_1} == ENGINE_1 and start["at"]
    assert {k: report[k] for k in ENGINE_1} == end  # the engine at the end, as before
    assert "## The engine changed" not in W.table_md(report)


@pytest.mark.parametrize("end, said", [
    ({**ENGINE_1, "commit": "c2", "engine_tree": "t2"}, "ended on engine tree t2 (commit c2)"),
    ({**ENGINE_1, "engine_modified": True},
     "ended on engine tree t1 (commit c1), its tracked files modified"),
], ids=["another_tree", "modified_files"])
def test_a_sweep_whose_engine_changed_while_it_ran_is_refused_and_says_so(tmp_path, monkeypatch,
                                                                         end, said):
    report = ran(tmp_path, monkeypatch, ENGINE_1, end)
    why = report["engine_changed"]
    assert why is not None and W.passed(report) is False
    assert "started on engine tree t1 (commit c1)" in why and said in why
    assert "refused" in why
    md = W.table_md(report | {"passed": W.passed(report)})
    assert "passed: False" in md and f"## The engine changed while the sweep ran\n\n{why}" in md


def test_a_resumed_sweep_holds_each_runs_start_and_is_refused_where_one_is_another_engine(
        tmp_path, monkeypatch):
    """A first run on tree t1, interrupted after its plan; the second run on tree t2 from its
    start to its end: the results are of two engines, so the report is refused. Resumed on the
    first run's engine, it can pass."""
    ran(tmp_path, monkeypatch, ENGINE_1, ENGINE_1, folder="a")
    other = {**ENGINE_1, "commit": "c2", "engine_tree": "t2"}
    report = ran(tmp_path, monkeypatch, other, other, folder="a")
    assert [s["engine_tree"] for s in report["engine_starts"]] == ["t1", "t2"]
    assert "started on engine tree t1" in report["engine_changed"]
    assert W.passed(report) is False
    ran(tmp_path, monkeypatch, ENGINE_1, ENGINE_1, folder="b")
    again = ran(tmp_path, monkeypatch, ENGINE_1, ENGINE_1, folder="b")
    assert again["engine_changed"] is None and W.passed(again) is True


def test_a_sweep_holding_work_from_a_run_that_recorded_no_engine_is_refused(tmp_path,
                                                                            monkeypatch):
    """Progress written by a run that recorded no start (before the sweep recorded one): the
    engine that work ran on is not known, so the report is refused; with a start, it is not."""
    folder = tmp_path / "old"
    folder.mkdir()
    base = {"project": "p0", "digest": "d0", "seq": 9, "scoring": {"set": False}, "points": [],
            "skipped": []}
    (folder / "progress.jsonl").write_text(
        json.dumps({"template": base}) + "\n"
        + json.dumps({"refit": W.Refit(W.PLAN, None, None, kind="scored").json()}) + "\n")
    report = ran(tmp_path, monkeypatch, ENGINE_1, ENGINE_1, folder="old")
    assert "recorded no engine when it started" in report["engine_changed"]
    assert W.passed(report) is False
    fresh = ran(tmp_path, monkeypatch, ENGINE_1, ENGINE_1, folder="new")
    assert fresh["engine_changed"] is None


# ── an engine outcome the walk meets after an option, classed ────────────────


class Walked:
    """A session for one refit on a copy: the option's answer said ``said``, then the walk served
    ``asks`` in turn, the project's stages ``stages`` until the option is answered and
    ``after`` once it is."""

    def __init__(self, asks, *, said=None, after=None):
        self.errors, self.asks, self.said = [], list(asks), said or arms.Said("ok", {})
        self.stages, self.after = {"working": {"status": "fresh", "key": "w1"}}, after
        self.pid, self.answered, self.dropped = None, [], []

    def copy(self, pid):
        return "c1"

    def use(self, pid):
        self.pid = pid

    def drop(self, pid, made):
        made.discard(pid)
        self.dropped.append(pid)
        return True

    def view(self):
        return {"decisions": [], "stages": self.stages, "state": {}}

    def read(self, path):
        return None

    def ask(self):
        return self.asks.pop(0)

    def preview_of(self, decision):
        return arms.Said("ok", {})

    def preview(self, ids, sets, reason):
        return arms.Said("ok", {})

    def decide(self, decision, reason):
        self.answered.append(dict(decision))
        if self.after is not None:
            self.stages = {**self.stages, **self.after}
        return self.said

    def answer(self, ids, sets, reason, marker=None):
        self.answered.append(list(ids))
        return arms.Said("ok", {})

    def fit(self, wait=0):  # pragma: no cover - no refit here reaches Fit
        raise AssertionError("Fit was pressed")


POINT = W.Point("declaration", "set_selection")
OPTION = {"id": "set_selection:vip", "decision": {"kind": "set_selection", "method": "vip"}}
CRASH = {"shelf": {"status": "error", "key": "s2", "error": "RuntimeError: boom",
                   "refusal": None}}
TYPED_STAGE = {"cohort": {"status": "error", "key": "c2",
                          "error": "ValueError: The answers recorded leave 3 of the 60 rows.",
                          "refusal": {"stage": "cohort", "code": "too_few_rows",
                                      "message": "The answers recorded leave 3 of the 60 rows.",
                                      "exits": [{"id": "x-1", "label": "Fill the blanks",
                                                 "decision": {"kind": "set_missing",
                                                              "strategy": "impute"}}]}}}


def waiting(stages, *names):
    from research.swarm import driver

    view = {"stages": stages}
    step = {"key": "models", "status": "waiting", "waiting_on": list(names)}
    return {"question": {"key": "models", "status": "waiting", "waiting_on": list(names)},
            "failed": driver.failed_on(view, step),
            "fit_failed": driver.failures(view, [*names, "fit"])}


def refit_on(tmp_path, session):
    sweep = a_sweep(tmp_path)
    return sweep, sweep.refit(session, POINT, OPTION, [], W.Refit(W.PLAN, None, None,
                                                                   kind="scored"))


def test_a_stage_crash_the_walk_meets_after_an_option_is_a_failed_refit_and_a_defect(tmp_path):
    session = Walked([waiting(CRASH, "shelf")])
    sweep, refit = refit_on(tmp_path, session)
    assert refit.kind == "failed" and refit.failed["code"] == "stage_failed"
    assert refit.failed["stage"] == "shelf" and refit.failed["message"] == "RuntimeError: boom"
    assert refit.failed["option"] == "set_selection:vip"
    (defect,) = sweep.defects
    kept = json.loads(Path(defect["path"]).read_text())
    assert kept["failed"]["stage"] == "shelf" and "read the shelf stage" in kept["reproduce"]
    assert session.dropped == []  # the copy is kept for the defect's reproduction


def test_a_stage_refusal_the_walk_meets_after_an_option_is_a_refused_refit(tmp_path):
    session = Walked([waiting(TYPED_STAGE, "cohort")])
    sweep, refit = refit_on(tmp_path, session)
    assert refit.kind == "refused" and sweep.defects == []
    r = refit.refused
    assert (r["by"], r["at"], r["stage"], r["code"]) == ("engine", "stage", "cohort",
                                                         "too_few_rows")
    assert r["message"] == "The answers recorded leave 3 of the 60 rows."
    assert r["exits"] == [{"label": "Fill the blanks",
                           "decision": {"kind": "set_missing", "strategy": "impute"}}]
    assert r["option"] == "set_selection:vip"
    (summary,) = W.decisions_of([POINT], [refit])
    assert summary["refused_share"] == 1.0 and summary["refused_codes"] == {"too_few_rows": 1}


def test_an_answer_the_walk_gives_refused_with_no_way_forward_is_a_refused_refit(tmp_path):
    from research.swarm.tests.test_arms import ask, opt

    missing = ask("missing", [opt("impute", {"kind": "set_missing"})])

    class Refusing(Walked):
        def answer(self, ids, sets, reason, marker=None):
            self.answered.append(list(ids))
            return arms.Said("refused", {}, by="server", code="imputation_with_the_outcome",
                             message="Not under prediction.")

    sweep, refit = refit_on(tmp_path, Refusing([missing, missing]))
    assert refit.kind == "refused" and sweep.defects == []
    assert (refit.refused["code"], refit.refused["at"], refit.refused["question"]) == (
        "imputation_with_the_outcome", "walk", "missing")
    assert refit.refused["option"] == "set_selection:vip"


@pytest.mark.parametrize("after, kind", [(CRASH, "failed"), (TYPED_STAGE, "refused"),
                                         (None, "unreached")],
                         ids=["stage_crashed", "stage_refused", "no_stage_failed"])
def test_a_walk_stopped_at_its_own_limit_is_classed_by_the_stages_that_failed_on_the_way(
        tmp_path, after, kind):
    """The walk stops where it cannot answer (every option takes a field). Where a stage of the
    copy failed after the option (a question opened over it), the refit is classed by it; where
    none did, it is unreached, and no defect is said."""
    from research.swarm.tests.test_arms import ask, opt

    stuck = ask("grain_like", [opt("needs", None, needs=["column"])])
    sweep, refit = refit_on(tmp_path, Walked([stuck], after=after))
    assert refit.kind == kind
    assert "records anything by itself" in refit.why
    assert len(sweep.defects) == (1 if kind == "failed" else 0)
    if kind == "refused":
        assert refit.refused["stage"] == "cohort" and refit.refused["code"] == "too_few_rows"


def test_a_stage_in_error_before_the_option_does_not_class_the_refit(tmp_path):
    from research.swarm.tests.test_arms import ask, opt

    session = Walked([ask("grain_like", [opt("needs", None, needs=["column"])])])
    session.stages = {**session.stages, **CRASH}  # in error on the template too
    sweep, refit = refit_on(tmp_path, session)
    assert refit.kind == "unreached" and sweep.defects == []


def test_the_sweeps_scoring_refused_after_an_option_is_a_refused_refit(tmp_path):
    """The sweep's own folds refused on the option's plan (§7: the sweep's share is its own)."""
    sweep = a_sweep(tmp_path)
    sweep.base["scoring"] = {"set": True}

    class Scheme(Walked):
        def view(self):
            return {"decisions": [], "stages": self.stages,
                    "state": {"split": {"holdout": 0.2, "validation": "kfold"}}}

        def decide(self, decision, reason):
            self.answered.append(dict(decision))
            if decision.get("kind") == "set_validation":
                return arms.Said("refused", {}, by="server", code="scheme_refused",
                                 message="Not on these rows.")
            return arms.Said("ok", {})

    session = Scheme([{"question": None, "fit_reached": True}])
    refit = sweep.refit(session, POINT, OPTION, [], W.Refit(W.PLAN, None, None, kind="scored"))
    assert refit.kind == "refused" and refit.refused["at"] == "scoring"
    assert refit.refused["code"] == "scheme_refused" and refit.refused["option"] == OPTION["id"]


# ── a refit's wait, and its estimate stages after the fit ────────────────────

RUNNING = {"pressed": {}, "fit": {"status": "running", "key": "k1", "progress": 0.4},
           "seconds": 3600.2}
BLOCKED = {"pressed": {}, "fit": {"status": "blocked"}, "seconds": 0.3}
FIT_FRESH = {"fit": {"status": "fresh", "key": "k1"}}
IN_FLIGHT = {"fit": {"status": "running", "key": "k1", "job_id": "j1"},
             "effects": {"status": "queued", "key": "e1", "job_id": "j2"}}
EFFECTS_CRASH = {"effects": {"status": "error", "key": "e2", "refusal": None,
                             "error": "ValueError: no effect can be reported here"}}
EFFECTS_TYPED = {"effects": {"status": "error", "key": "e3",
                             "error": "ValueError: No row is left to report the effect on.",
                             "refusal": {"stage": "effects", "code": "no_rows_left",
                                         "message": "No row is left to report the effect on.",
                                         "exits": [{"id": "x-2", "label": "Fill the blanks",
                                                    "decision": {"kind": "set_missing",
                                                                 "strategy": "impute"}}]}}}
ENDED = "; its jobs ended and the copy was removed."


class Pressed(Walked):
    """A copy whose walk reaches Fit at once: its press of Fit says ``fit`` (the driver's payload,
    the clock moved on by its ``seconds``, at most the wait), the copy's stages are ``stages``, each
    ask of whether it went idle answers the next of ``idle``, and a cancel answers ``cancelled``;
    each wait asked for (``waits``), each copy cancelled (``cancels``) and removed is kept. The
    fit's record is served on the first read."""

    RECORD = {"id": "f2", "fit_key": "k1", "seconds": 1.2, "primary_metric": METRIC,
              "folds": "d", "scores": {"linear": {METRIC: {"mean": 0.5}}}}

    def __init__(self, clock, fit, *, stages, idle=(True,), cancelled=None, removed=True):
        super().__init__([{"question": None, "fit_reached": True}])
        self.clock, self.fitted, self.removed = clock, fit, removed
        self.stages = {**self.stages, **stages}
        self.idles, self.waits, self.cancels = list(idle), [], []
        self.cancelled = cancelled or {"fit": 200, "jobs": [], "idle": True, "wait": 60.0}

    def fit(self, wait=0):
        self.waits.append(("fit", wait))
        self.clock.sleep(min(float(self.fitted.get("seconds") or 0.0), wait))
        return arms.said_of(0, self.fitted)

    def idle(self, pid=None, wait=0.0):
        self.waits.append(("idle", wait))
        return self.idles.pop(0)

    def cancel(self, pid, wait=None):
        self.cancels.append(pid)
        return dict(self.cancelled)

    def drop(self, pid, made):
        self.dropped.append(pid)
        if self.removed:
            made.discard(pid)
        return self.removed

    def read(self, path):
        return {"records": [self.RECORD]} if path == "fits" else None


def pressed(tmp_path, monkeypatch, fit, **kw):
    clock = Clock()
    monkeypatch.setattr(W, "time", clock)
    session = Pressed(clock, fit, **kw)
    sweep, refit = refit_on(tmp_path, session)
    return session, sweep, refit


def test_a_refit_whose_fit_is_not_done_when_its_wait_runs_out_is_cancelled_and_unreached(
        tmp_path, monkeypatch):
    """The fit still running when the refit's wait from the press runs out: the sweep cancels the
    copy's fit, waits for its jobs to end (the cancel's own wait), records the refit unreached
    with what it waited on and for how long, and removes the copy; no other wait is asked."""
    session, sweep, refit = pressed(tmp_path, monkeypatch, RUNNING, stages=IN_FLIGHT)
    assert session.waits == [("fit", W.REFIT_WAIT)]
    assert session.cancels == ["c1"] and session.dropped == ["c1"]
    assert refit.kind == "unreached" and refit.project is None and sweep.defects == []
    assert refit.why == (f"The fit was running {W.REFIT_WAIT:g} s after the sweep pressed Fit, "
                         f"as long as it waits for a refit (still computing: fit running, "
                         f"effects queued). The sweep cancelled the copy's fit{ENDED}")
    assert refit.cancelled == {"after": W.REFIT_WAIT, "fit": 200, "jobs": [], "idle": True,
                               "wait": 60.0,
                               "waiting_on": [{"stage": "fit", "status": "running"},
                                              {"stage": "effects", "status": "queued"}]}


@pytest.mark.parametrize("payload", [BLOCKED, HELD], ids=["blocked", "held_for_a_press"])
def test_a_fit_that_returns_before_the_wait_runs_out_is_not_cancelled(tmp_path, monkeypatch,
                                                                       payload):
    """Both ways: a fit that is blocked, or held for a press of Fit, returns at once; it is
    unreached in the driver's words as before, and nothing is cancelled."""
    session, sweep, refit = pressed(tmp_path, monkeypatch, payload, stages={})
    assert session.cancels == [] and refit.cancelled is None
    assert refit.kind == "unreached" and refit.why == arms.said_of(0, payload).message
    assert session.dropped == ["c1"]


def test_a_fitted_refit_whose_stages_are_not_done_in_the_rest_of_its_wait_is_cancelled(
        tmp_path, monkeypatch):
    """The fit fresh 1.5 s after the press, the sweep waits the rest of the refit's wait for the
    copy's other stages; the effects stage still running then, the refit is cancelled, unreached
    with what it waited on, and its copy removed."""
    stages = {**FIT_FRESH, "effects": {"status": "running", "key": "e1", "job_id": "j2"}}
    session, sweep, refit = pressed(tmp_path, monkeypatch, FRESH, stages=stages, idle=[False])
    assert session.waits == [("fit", W.REFIT_WAIT), ("idle", W.REFIT_WAIT - 1.5)]
    assert session.cancels == ["c1"] and session.dropped == ["c1"]
    assert refit.kind == "unreached" and refit.material is None
    assert refit.why == (f"The fit was done, but {W.REFIT_WAIT:g} s after the sweep pressed Fit, "
                         f"as long as it waits for a refit, its other stages were not (still "
                         f"computing: effects running). The sweep cancelled the copy's "
                         f"fit{ENDED}")
    assert refit.cancelled["waiting_on"] == [{"stage": "effects", "status": "running"}]


DESIGN_JOB = [{"stage": "design", "job": "j3", "status": 200}]


@pytest.mark.parametrize("idle, removed, end, dropped", [
    (True, True, f"; its stages still computing 60 s later, it cancelled their jobs too "
                 f"(design){ENDED}", ["c1"]),
    (True, False, "; its stages still computing 60 s later, it cancelled their jobs too "
                  "(design); its jobs ended, but the copy could not be removed and is kept.",
     ["c1"]),
    (False, True, "; its stages still computing 60 s later, it cancelled their jobs too "
                  "(design); the copy was still computing and is kept.", []),
], ids=["went_idle", "not_removed", "still_computing"])
def test_a_copy_that_does_not_go_idle_after_its_cancel_has_its_jobs_cancelled_and_says_so(
        tmp_path, monkeypatch, idle, removed, end, dropped):
    cancelled = {"fit": 200, "jobs": DESIGN_JOB, "idle": idle, "wait": 60.0}
    session, sweep, refit = pressed(tmp_path, monkeypatch, RUNNING, stages=IN_FLIGHT,
                                    cancelled=cancelled, removed=removed)
    assert refit.kind == "unreached" and refit.cancelled["jobs"] == DESIGN_JOB
    assert refit.why.endswith(f"The sweep cancelled the copy's fit{end}"), refit.why
    assert session.dropped == dropped
    assert (refit.project is None) is (idle and removed)


def test_an_untyped_error_in_an_estimate_stage_after_a_fitted_refit_is_an_engine_defect(
        tmp_path, monkeypatch):
    """The fit fresh and the copy idle, the effects stage (one the fit does not wait on) is in
    error the engine did not type: the refit is failed at that stage in its own words, reported
    as a defect with its reproduction, and its copy kept."""
    stages = {**FIT_FRESH, **EFFECTS_CRASH}
    session, sweep, refit = pressed(tmp_path, monkeypatch, FRESH, stages=stages)
    assert session.waits == [("fit", W.REFIT_WAIT), ("idle", W.REFIT_WAIT - 1.5)]
    assert refit.kind == "failed" and refit.failed["code"] == "stage_failed"
    assert refit.failed["stage"] == "effects" and refit.failed["at"] == "stage"
    assert refit.failed["message"] == "ValueError: no effect can be reported here"
    assert refit.failed["option"] == OPTION["id"]
    (defect,) = sweep.defects
    kept = json.loads(Path(defect["path"]).read_text())
    assert kept["failed"]["stage"] == "effects" and "read the effects stage" in kept["reproduce"]
    assert session.dropped == [] and session.cancels == []


def test_a_typed_refusal_in_an_estimate_stage_after_a_fitted_refit_is_a_refused_refit(
        tmp_path, monkeypatch):
    stages = {**FIT_FRESH, **EFFECTS_TYPED}
    session, sweep, refit = pressed(tmp_path, monkeypatch, FRESH, stages=stages)
    assert refit.kind == "refused" and sweep.defects == []
    r = refit.refused
    assert (r["by"], r["at"], r["stage"], r["code"]) == ("engine", "stage", "effects",
                                                         "no_rows_left")
    assert r["message"] == "No row is left to report the effect on."
    assert r["exits"] == [{"label": "Fill the blanks",
                           "decision": {"kind": "set_missing", "strategy": "impute"}}]
    (summary,) = W.decisions_of([POINT], [refit])
    assert summary["refused_codes"] == {"no_rows_left": 1}
    assert session.dropped == ["c1"]


def test_a_fitted_refit_whose_estimate_stages_are_none_in_error_is_scored(tmp_path, monkeypatch):
    """Both ways: every estimate stage fresh or blocked after the fit, the refit is scored."""
    stages = {**FIT_FRESH, "effects": {"status": "fresh", "key": "e1"},
              "secondary": {"status": "blocked", "key": None}}
    session, sweep, refit = pressed(tmp_path, monkeypatch, FRESH, stages=stages)
    assert refit.kind == "scored" and refit.cancelled is None and sweep.defects == []
    assert refit.fit_record == "f2" and session.dropped == ["c1"] and session.cancels == []


@pytest.mark.parametrize("payload, stages, kind", [
    (RUNNING, IN_FLIGHT, "unreached"), (FRESH, {**FIT_FRESH, **EFFECTS_CRASH}, "failed"),
], ids=["wait_ran_out", "estimate_stage_crashed"])
def test_the_plans_own_refit_is_read_as_every_refit_and_stops_the_sweep_unscored(
        tmp_path, monkeypatch, payload, stages, kind):
    """The plan's own refit is cancelled where its wait runs out, its copy removed, and read for
    its estimate stages after its fit, an untyped error there a defect with its reproduction; a
    plan not scored stops the sweep, in words."""
    clock = Clock()
    monkeypatch.setattr(W, "time", clock)
    session = Pressed(clock, payload, stages=stages)
    sweep = a_sweep(tmp_path)
    with pytest.raises(arms.Stuck, match=f"The plan itself was not scored: {kind}: "):
        sweep.plan_refit(session)
    if kind == "unreached":
        assert session.cancels == ["c1"] and session.dropped == ["c1"] and sweep.defects == []
    else:
        (defect,) = sweep.defects
        assert defect["refit"] == W.PLAN and defect["failed"]["stage"] == "effects"
        assert session.dropped == []


# ── the crash sweep's previews of every later decision's default ─────────────

EARLY = W.Point("question", "exclusions", 3, "r3", {"kind": "set_exclusions", "rules": []})
MISSING = W.Point("question", "missing", 4, "r4", {"kind": "set_missing",
                                                   "strategy": "complete_case"})
MODELS = W.Point("question", "models", 6, "r6", {"kind": "select_models", "models": ["linear"]})
REPAIR = W.Point("finding", "f1", 7, "r7", {"kind": "apply_repair", "finding_id": "f1",
                                            "repair": "recode"})
UNIT = W.Point("reading", "unit:age", 8, "r8", {"kind": "set_column_unit", "column": "age",
                                                "unit": "years"})
SELECTION = W.Point("declaration", "set_selection", 9, "r9", {"kind": "set_selection",
                                                              "method": "none"})
LEFT_OPEN = W.Point("declaration", "set_explain")  # a line the plan left open: no answer of its own
PLAN_POINTS = [EARLY, MISSING, MODELS, REPAIR, UNIT, SELECTION, LEFT_OPEN]
IMPUTE = {"kind": "set_missing", "strategy": "impute"}
LASSO = {"kind": "set_selection", "method": "lasso"}


class Previewed(Walked):
    """A crash sweep's session over :data:`PLAN_POINTS`: a revisit opens its question again, an
    answer reaches Fit, every log the same as the plan's (so each refit is the plan's duplicate and
    no Fit is pressed); each preview kept in ``calls`` in turn with each answer, and the preview of
    the decision ``fails`` failing as an HTTP 500 does (kept in ``errors``, as ``arms.Local``
    keeps it)."""

    def __init__(self, *, fails=None, asks=()):
        super().__init__(list(asks))
        self.fails, self.calls, self.reopened = fails, [], None

    def _fail(self, body):
        self.errors.append({"method": "POST", "path": "/projects/c1/preview", "status": 500,
                            "body": json.dumps(body), "said": "Internal Server Error"})
        return arms.Said("failed", {}, by="server", code="http_500",
                         message="Internal Server Error", status=500)

    def copy(self, pid):
        self.calls.append(("copy", pid))  # each refit's start
        return "c1"

    def revisit(self, record, reason):
        self.reopened = next(p.key for p in PLAN_POINTS if p.record == record)
        return arms.Said("ok", {})

    def ask(self):
        from research.swarm.tests.test_arms import ask, opt

        if self.reopened is not None:
            return ask(self.reopened, [opt("impute", IMPUTE),
                                       opt("complete_case", MISSING.decision)])
        return self.asks.pop(0) if self.asks else {"question": None, "fit_reached": True}

    def preview_of(self, decision):
        self.calls.append(("preview_of", dict(decision)))
        return self._fail(decision) if dict(decision) == self.fails else arms.Said("ok", {})

    def preview(self, ids, sets, reason):
        self.calls.append(("preview", list(ids)))
        return self._fail(ids) if list(ids) == self.fails else arms.Said("ok", {})

    def answer(self, ids, sets, reason, marker=None):
        self.reopened = None
        self.calls.append(("answer", list(ids)))
        return arms.Said("ok", {})

    def decide(self, decision, reason):
        self.calls.append(("decide", dict(decision)))
        return arms.Said("ok", {})


def swept_over(tmp_path, session, *, crash=True):
    """The crash sweep's run over :data:`PLAN_POINTS`, its options listed here: impute for the
    missing question, lasso for the selection line, none for the others."""
    sweep = W.Sweep(tmp_path / "sweep", dataset="tiny", question=arms.Question(
        target="y", purpose="prediction"), crash=crash, plan="default")
    sweep.base = {"project": "p0", "digest": "d0", "seq": 9, "scoring": {"set": False},
                  "points": [p.json() for p in PLAN_POINTS], "skipped": []}
    sweep.done[W.PLAN] = W.Refit(W.PLAN, None, None, kind="scored",
                                 digest=W.plan_digest(session.view()))
    options = {MISSING.id: [{"id": "impute", "label": "impute", "decision": IMPUTE}],
               SELECTION.id: [{"id": "set_selection:lasso", "label": "lasso",
                               "decision": LASSO}]}
    sweep.offered = {p.id: {"point": p.id, "why": None, "options": options.get(p.id, [])}
                     for p in PLAN_POINTS}
    report = sweep.run(session, lambda s: "p0")
    return sweep, {r["id"]: r for r in report["refits"]}


def previewed_after(calls, answer):
    """The decisions previewed by their body after ``answer``, until the next answer or refit."""
    i = calls.index(answer)
    out = []
    for call, body in calls[i + 1:]:
        if call in ("answer", "decide", "copy"):
            break
        if call == "preview_of":
            out.append(body)
    return out


def test_the_crash_sweep_previews_every_later_decisions_plan_answer_after_an_option(tmp_path):
    """After a question's option, each decision after it in the plan's order with an answer of its
    own (a question, a finding, a reading, a declaration) is previewed at the plan's answer; one
    before it, the question itself and a line left open are not. After a declaration's option
    (the Router does not order it among its questions), every other decision is."""
    session = Previewed()
    sweep, refits = swept_over(tmp_path, session)
    assert previewed_after(session.calls, ("answer", ["impute"])) == [
        MODELS.decision, REPAIR.decision, UNIT.decision, SELECTION.decision]
    assert previewed_after(session.calls, ("decide", LASSO)) == [
        EARLY.decision, MISSING.decision, MODELS.decision, REPAIR.decision, UNIT.decision]
    assert [p["of"] for p in refits[f"{MISSING.id}=impute"]["previews"]] == [
        f"{MISSING.id}=impute", *(f"{p.id} (the plan's answer)" for p in
                                  (MODELS, REPAIR, UNIT, SELECTION))]
    assert {r["kind"] for r in refits.values()} == {"duplicate"} and sweep.defects == []


@pytest.mark.parametrize("later", [MODELS, REPAIR, UNIT, SELECTION],
                         ids=["question", "finding", "reading", "declaration"])
def test_a_500_on_a_later_decisions_plan_answer_previewed_after_an_option_is_a_defect(
        tmp_path, later):
    """The preview of the plan's answer to a decision after the changed one fails as an HTTP 500:
    the refit of that option is failed and an engine defect, with the request that failed."""
    session = Previewed(fails=later.decision)
    sweep, refits = swept_over(tmp_path, session)
    refit = refits[f"{MISSING.id}=impute"]
    assert refit["kind"] == "failed" and refit["failed"]["code"] == "http_500"
    assert {"of": f"{later.id} (the plan's answer)", "kind": "failed", "by": "server",
            "code": "http_500", "message": "Internal Server Error", "status": 500} in \
        refit["previews"]
    defect = next(d for d in sweep.defects if d["refit"] == refit["id"])
    kept = json.loads(Path(defect["path"]).read_text())
    assert [json.loads(r["body"]) for r in kept["requests"]] == [later.decision]


def test_a_500_on_an_earlier_decisions_plan_answer_fails_only_the_options_it_is_later_than(
        tmp_path):
    """The plan's answer to a question before the changed one is not previewed after it, so its
    500 fails no refit of the later question; a declaration's option previews it, and fails."""
    session = Previewed(fails=EARLY.decision)
    sweep, refits = swept_over(tmp_path, session)
    assert refits[f"{MISSING.id}=impute"]["kind"] == "duplicate"
    assert refits[f"{SELECTION.id}=set_selection:lasso"]["kind"] == "failed"
    assert [d["refit"] for d in sweep.defects] == [f"{SELECTION.id}=set_selection:lasso"]


def test_the_sweep_around_a_named_plan_previews_no_later_default(tmp_path):
    """Only the crash sweep previews: the sweep around a named plan meets no preview's 500."""
    session = Previewed(fails=MODELS.decision)
    sweep, refits = swept_over(tmp_path, session, crash=False)
    assert not [c for c in session.calls if c[0].startswith("preview")]
    assert refits[f"{MISSING.id}=impute"]["kind"] == "duplicate" and sweep.defects == []


@pytest.mark.parametrize("fails", [None, ["false"]], ids=["previews", "preview_500"])
def test_a_question_the_option_opens_again_is_previewed_at_its_default_before_it_is_answered(
        tmp_path, fails):
    """After the option, the walk meets the temporal question open again: the crash sweep previews
    the default path's answer to it (``false``, ``arms.Default``) before the walk answers it, and
    a 500 on that preview fails the refit as an engine defect; with none it is the plan's
    duplicate."""
    from research.swarm.tests.test_arms import ask, opt

    temporal = ask("temporal", [opt("true", {"kind": "set_temporal", "temporal": True}),
                                opt("false", {"kind": "set_temporal", "temporal": False})])
    session = Previewed(fails=fails, asks=[temporal])
    sweep = a_sweep(tmp_path)
    plan = W.Refit(W.PLAN, None, None, kind="scored", digest=W.plan_digest(session.view()))
    refit = sweep.refit(session, POINT, OPTION, [], plan)
    walked = session.calls[session.calls.index(("decide", OPTION["decision"])) + 1:]
    assert walked == [("preview", ["false"]), ("answer", ["false"])]
    assert refit.previews[-1]["of"] == "question:temporal (opened again, its default)"
    if fails is None:
        assert refit.kind == "duplicate" and sweep.defects == []
    else:
        assert refit.kind == "failed" and refit.failed["code"] == "http_500"
        assert refit.previews[-1]["code"] == "http_500"
        assert [d["refit"] for d in sweep.defects] == [refit.id]


# ── a revisit the engine refuses until the answer that reads the column is withdrawn ─────────

STAGE_CODE = {"kind": "confirm_reading", "reading": "code_or_count", "column": "stage",
              "value": "code"}
STAGE_COUNT = {**STAGE_CODE, "value": "count"}
TWO_MODELS = {"kind": "select_models", "models": ["elastic_net", "screened_elastic_net"]}
# The engine's words for it, as ``decisions._answers_keep_settled_readings`` writes them.
READS_STAGE = ("The recorded models answer reads `stage`, which would then rest on a reading "
               "nobody settled. Change the models answer first, or keep the confirmation.")
STAGE = W.Point("reading", "code_or_count:stage", 13, "r13", STAGE_CODE)


class Reads(Walked):
    """A copy of a plan whose models answer (record 14) reads `stage`, read as a code by record 13:
    reverting the reading is refused (``code``, the engine's ``words``) while the models answer
    stands; reverting the models answer withdraws it, unless ``refuse`` (``"withdraw"``) refuses
    that revert or ``fail`` fails it as an HTTP 500; then the reading's two ways are asked. The
    models answer posted again is refused where ``refuse`` is ``"again"``. Every log is the plan's,
    so each refit is its duplicate and no Fit is pressed."""

    def __init__(self, *, refuse=None, fail=None, code="role_unconfirmed", words=READS_STAGE):
        super().__init__([])
        self.calls, self.refuse, self.fail, self.code, self.words = [], refuse, fail, code, words
        self.withdrawn = self.reopened = False

    def copy(self, pid):
        self.withdrawn = self.reopened = False  # a fresh copy of the template
        self.calls.append(("copy",))
        return "c1"

    def view(self):
        at = datetime(2026, 10, 23, tzinfo=timezone.utc).isoformat()
        return {"decisions": [{"id": f"r{seq}", "seq": seq, "at": at, "recorded_by": "you",
                               "decision": d} for seq, d in ((13, STAGE_CODE), (14, TWO_MODELS))],
                "stages": self.stages, "state": {}}

    def revisit(self, record, reason):
        self.calls.append(("revisit", record))
        if record == "r14":
            if self.fail == "withdraw":
                self.errors.append({"method": "POST", "path": "/projects/c1/decisions",
                                    "status": 500, "body": "{}", "said": "Internal Server Error"})
                return arms.Said("failed", {}, by="server", code="http_500",
                                 message="Internal Server Error", status=500)
            if self.refuse == "withdraw":
                return arms.Said("refused", {}, by="server", code="models_read_later",
                                 message="The recorded levers read the models answer.",
                                 exits=[{"label": "Keep the answers as they are",
                                         "decision": None}])
            self.withdrawn = True
            return arms.Said("ok", {})
        if record == "r13" and not self.withdrawn:
            return arms.Said("refused", {}, by="server", code=self.code, message=self.words,
                             exits=[{"label": "Keep the answers as they are", "decision": None}])
        self.reopened = True
        return arms.Said("ok", {})

    def ask(self):
        from research.swarm.tests.test_arms import opt

        if self.reopened:
            return {"question": {"key": "models", "status": "open"},
                    "readings": {"exits": [opt("x-code", STAGE_CODE),
                                           opt("x-count", STAGE_COUNT)]}}
        return {"question": None, "fit_reached": True}

    def preview(self, ids, sets, reason):
        self.calls.append(("preview", list(ids)))
        return arms.Said("ok", {})

    def answer(self, ids, sets, reason, marker=None):
        self.reopened = False
        self.calls.append(("answer", list(ids)))
        return arms.Said("ok", {})

    def decide(self, decision, reason):
        self.calls.append(("decide", dict(decision)))
        if self.refuse == "again" and decision.get("kind") == "select_models":
            return arms.Said("refused", {}, by="server", code="models_refused",
                             message="Not with `stage` read as a count.")
        return arms.Said("ok", {})


def swept_reads(tmp_path, session):
    sweep = W.Sweep(tmp_path / "sweep", dataset="tiny", question=arms.Question(
        target="y", purpose="prediction"), crash=True, plan="default")
    sweep.base = {"project": "p0", "digest": "d0", "seq": 14, "scoring": {"set": False},
                  "points": [STAGE.json()], "skipped": []}
    sweep.done[W.PLAN] = W.Refit(W.PLAN, None, None, kind="scored",
                                 digest=W.plan_digest(session.view()))
    report = sweep.run(session, lambda s: "p0")
    return sweep, report, {r["id"]: r for r in report["refits"]}


def test_a_reading_the_models_answer_reads_is_revisited_by_the_way_the_engine_names(tmp_path):
    """The orchestrator's ruling on V2-1: the models answer withdrawn first, the reading's
    options listed and each answered, the models answer answered again as the plan had it; the
    reading visited, and the crash sweep passes with no defect."""
    session = Reads()
    sweep, report, refits = swept_reads(tmp_path, session)
    assert report["unlisted"] == [] and report["unvisited"] == [] and report["defects"] == []
    assert report["complete"] is True and W.passed(report) is True
    (offered,) = report["offered"]
    assert [o["id"] for o in offered["options"]] == ["x-code", "x-count"]
    assert offered["withdrawn"] == [{"what": "models answer", "column": "stage", "seq": 14,
                                     "record": "r14"}]
    count = refits[f"{STAGE.id}=x-count"]
    start = max(i for i, c in enumerate(session.calls) if c == ("copy",))
    assert session.calls[start + 1:] == [("revisit", "r13"), ("revisit", "r14"),
                                         ("revisit", "r13"), ("preview", ["x-count"]),
                                         ("answer", ["x-count"]), ("decide", TWO_MODELS)]
    assert [w["decision"] for w in count["withdrawn"]] == [TWO_MODELS]
    assert "the plan's models answer again (record 14)" in [a["answer"]
                                                             for a in count["answered"]]
    assert {r["kind"] for r in refits.values()} == {"duplicate"}
    assert "the models answer (record 14), which reads `stage`" in W.table_md(report)


def test_where_the_way_the_engine_names_is_refused_the_reading_is_visited_and_refused(tmp_path):
    sweep, report, refits = swept_reads(tmp_path, Reads(refuse="withdraw"))
    revisit = refits[f"{STAGE.id}={W.REVISIT}"]
    assert revisit["kind"] == "refused" and revisit["refused"]["at"] == "revisit"
    assert revisit["refused"]["code"] == "models_read_later"
    assert revisit["refused"]["message"] == "The recorded levers read the models answer."
    assert [w["what"] for w in revisit["refused"]["withdrawn"]] == ["models answer"]
    assert report["unlisted"] == [] and report["complete"] is True and report["defects"] == []
    (decision,) = report["decisions"]
    assert decision["refused_share"] == 1.0 and decision["refused_codes"] == {
        "models_read_later": 1}


def test_where_the_plans_answer_is_refused_again_after_the_option_the_option_is_refused(
        tmp_path):
    sweep, report, refits = swept_reads(tmp_path, Reads(refuse="again"))
    for option in ("x-code", "x-count"):
        r = refits[f"{STAGE.id}={option}"]
        assert r["kind"] == "refused", r
        assert (r["refused"]["at"], r["refused"]["code"], r["refused"]["option"]) == (
            "answer_again", "models_refused", option)
        assert r["refused"]["decision"] == TWO_MODELS
    assert report["complete"] is True and report["defects"] == []


@pytest.mark.parametrize("code, words", [
    ("some_other_refusal", READS_STAGE),
    ("role_unconfirmed", READS_STAGE.replace("models answer", "levers answer"))],
    ids=["another_code", "an_answer_it_cannot_name"])
def test_a_revisit_refused_otherwise_is_unlisted_with_its_words_and_withdraws_nothing(
        tmp_path, code, words):
    session = Reads(code=code, words=words)
    sweep, report, refits = swept_reads(tmp_path, session)
    (unlisted,) = report["unlisted"]
    assert unlisted["point"] == STAGE.id and code in unlisted["why"]
    assert ("revisit", "r14") not in session.calls and report["complete"] is False


def test_a_500_on_the_way_the_engine_names_is_an_engine_defect(tmp_path):
    sweep, report, refits = swept_reads(tmp_path, Reads(fail="withdraw"))
    revisit = refits[f"{STAGE.id}={W.REVISIT}"]
    assert revisit["kind"] == "failed" and revisit["failed"]["code"] == "http_500"
    assert [d["refit"] for d in report["defects"]] == [revisit["id"]]
    assert W.passed(report) is False


# ── a revisit refused because the split is drawn (F1): the way the engine names ──

GRAIN_ONE = {"kind": "set_grain", "grain": "one_row_per_unit"}
GRAIN_UNKNOWN = {"kind": "set_grain", "grain": "unknown"}
SPLIT_A = {"kind": "set_split", "holdout": 0.0, "seed": 0}
SPLIT_B = {"kind": "set_split", "holdout": 0.0, "seed": 1}
GRAIN = W.Point("question", "grain", 6, "r6", GRAIN_ONE)
SEALED_WORDS = ("The cross-validation folds are drawn, and changing whether a unit can appear in "
                "more than one row changes what a row is: the folds name rows as they were. "
                "Withdraw the split first, then draw it again.")


class Sealed(Walked):
    """A copy of a plan whose grain answer (record 6) was given before two split answers (records
    7 and 8, neither reverted, so 8's draw is in force: ``decisions.fold``). As the engine's
    ``seal._decision_a_waits_for_a_reseal`` holds it, reverting the grain is refused ``sealed``
    while a split is in force, its one way forward the revert of the latest split in force
    (``seal._reseal_exit``: "Re-draw: withdraw the folds, change this, then draw them again");
    ``refuse`` ``"withdraw"`` refuses that revert, ``"no_way"`` offers none; ``names`` makes the
    way forward revert that record instead (one that is not a split in force). Each post is a
    record of the log, so what a refit leaves in force is the fold of what it posted; while the
    grain is reverted the ask opens the grain question, else Fit is reached."""

    def __init__(self, refuse=None, names=None):
        super().__init__([])
        self.refuse, self.names, self.calls = refuse, names, []
        self.template = [(6, GRAIN_ONE), (7, SPLIT_A), (8, SPLIT_B)]
        self.log = list(self.template)

    def copy(self, pid):
        self.log = list(self.template)  # a fresh copy of the template
        return "c1"

    def view(self):
        return {**view_at(self.log), "stages": self.stages, "state": {}}

    def _post(self, decision):
        self.log.append((max(seq for seq, _ in self.log) + 1, dict(decision)))
        return arms.Said("ok", {})

    def _splits(self):
        return [r for r in W.live(self.view()) if r.decision.kind == "set_split"]

    def revisit(self, record, reason):
        self.calls.append(("revisit", record))
        if record == "r6" and self._splits():
            exits = [{"id": None, "label": "Keep the folds as they are", "decision": None}]
            if self.refuse != "no_way":
                exits = [{"id": "x-redraw",
                          "label": "Re-draw: withdraw the folds, change this, then draw them "
                                   "again",
                          "decision": {"kind": "revert",
                                       "decision_id": self.names or self._splits()[-1].id}}]
            return arms.Said("refused", {}, by="server", code="sealed", message=SEALED_WORDS,
                             exits=exits)
        if record in ("r7", "r8") and self.refuse == "withdraw":
            return arms.Said("refused", {}, by="server", code="split_read_later",
                             message="The recorded scheme reads this split.",
                             exits=[{"id": None, "label": "Keep the split", "decision": None}])
        return self._post({"kind": "revert", "decision_id": record})

    def ask(self):
        from research.swarm.tests.test_arms import opt

        if any(r.decision.kind == "set_grain" for r in W.live(self.view())):
            return {"question": None, "fit_reached": True}
        return {"question": {"key": "grain", "status": "open"},
                "options": [opt("one_row_per_unit", GRAIN_ONE), opt("unknown", GRAIN_UNKNOWN)]}

    def answer(self, ids, sets, reason, marker=None):
        self.calls.append(("answer", list(ids)))
        return self._post({"one_row_per_unit": GRAIN_ONE, "unknown": GRAIN_UNKNOWN}[ids[0]])

    def decide(self, decision, reason):
        self.calls.append(("decide", dict(decision)))
        return self._post(decision)

    def fit(self, wait=0):
        return arms.Said("refused", {"fit": {}}, by="driver", code="not_here",
                         message="No fit is made here.")


def swept_sealed(tmp_path, session):
    sweep = W.Sweep(tmp_path / "sweep", dataset="tiny", question=arms.Question(
        target="y", purpose="prediction"), crash=True, plan="default")
    sweep.base = {"project": "p0", "digest": "d0", "seq": 8, "scoring": {"set": False},
                  "points": [GRAIN.json()], "skipped": []}
    sweep.done[W.PLAN] = W.Refit(W.PLAN, None, None, kind="scored",
                                 digest=W.plan_digest(session.view()))
    report = sweep.run(session, lambda s: "p0")
    return report, {r["id"]: r for r in report["refits"]}


def test_a_revisit_refused_as_the_split_is_drawn_takes_the_way_the_engine_names(tmp_path):
    """F1: the grain's revisit is refused while the folds are drawn; the sweep withdraws the
    split by the revert the refusal names (the latest in force first, then the one the next
    refusal names), revisits the grain, answers each option, and draws the split again as the plan
    had it, in the plan's order: the plan's latest draw is again the one in force, so the refit is
    the plan with the option and nothing else (its digest that plan's, built here); the grain is
    visited and the crash sweep passes."""
    from turbotab.core import decisions as D

    session = Sealed()
    report, refits = swept_sealed(tmp_path, session)
    assert report["unlisted"] == [] and report["unvisited"] == [] and report["defects"] == []
    assert report["complete"] is True and W.passed(report) is True
    (offered,) = report["offered"]
    assert [o["id"] for o in offered["options"]] == ["one_row_per_unit", "unknown"]
    assert offered["withdrawn"] == [{"what": "split", "column": None, "seq": 8, "record": "r8"},
                                    {"what": "split", "column": None, "seq": 7, "record": "r7"}]
    unknown = refits[f"{GRAIN.id}=unknown"]
    assert [w["decision"]["seed"] for w in unknown["withdrawn"]] == [1, 0]  # in the order withdrawn
    start = max(i for i, c in enumerate(session.calls) if c == ("revisit", "r6")) - 4
    assert [c[0] for c in session.calls[start:start + 5]] == ["revisit"] * 5
    assert [c[1] for c in session.calls[start:start + 5]] == ["r6", "r8", "r6", "r7", "r6"]
    again = [c[1] for c in session.calls if c[0] == "decide"][-2:]
    assert [d["seed"] for d in again] == [0, 1]  # drawn again in the plan's order
    assert "the plan's split again (record 7)" in [a["answer"] for a in unknown["answered"]]
    state = D.fold(W.records_of(session.view()))
    assert (state.grain.grain, state.split.seed) == ("unknown", 1)
    with_option = view_at([(6, GRAIN_UNKNOWN), (7, SPLIT_A), (8, SPLIT_B)])
    assert unknown["digest"] == W.plan_digest(with_option) != report["template"]["digest"]
    assert refits[f"{GRAIN.id}=one_row_per_unit"]["kind"] == "duplicate"
    assert "the split (record 8), the split (record 7); answered again" in W.table_md(report)


def test_where_the_split_cannot_be_withdrawn_the_decision_is_visited_and_refused(tmp_path):
    report, refits = swept_sealed(tmp_path, Sealed(refuse="withdraw"))
    revisit = refits[f"{GRAIN.id}={W.REVISIT}"]
    assert revisit["kind"] == "refused" and revisit["refused"]["at"] == "revisit"
    assert (revisit["refused"]["code"], revisit["refused"]["message"]) == (
        "split_read_later", "The recorded scheme reads this split.")
    assert [w["what"] for w in revisit["refused"]["withdrawn"]] == ["split"]
    assert report["unlisted"] == [] and report["complete"] is True and report["defects"] == []
    (decision,) = report["decisions"]
    assert decision["refused_share"] == 1.0 and decision["refused_codes"] == {
        "split_read_later": 1}


def test_a_sealed_revisit_that_names_no_withdrawal_is_unlisted_and_withdraws_nothing(tmp_path):
    """The guard's other side: a refusal ``sealed`` whose ways forward revert nothing names no
    way to take, so the decision is unlisted in the engine's words, and nothing is withdrawn."""
    session = Sealed(refuse="no_way")
    report, refits = swept_sealed(tmp_path, session)
    (unlisted,) = report["unlisted"]
    assert unlisted["point"] == GRAIN.id and "sealed" in unlisted["why"]
    assert SEALED_WORDS in unlisted["why"]
    assert [c for c in session.calls if c[0] == "revisit" and c[1] != "r6"] == []
    assert report["complete"] is False and W.passed(report) is False


@pytest.mark.parametrize("names", ["r99", "r6", "r5", "r4"],
                         ids=["no_record", "the_grain_itself", "a_split_reverted",
                              "another_answer_in_force"])
def test_a_sealed_way_forward_that_names_no_split_in_force_is_unlisted_and_withdraws_nothing(
        tmp_path, names):
    """The guard's other side, where the refusal ``sealed`` does name a revert: of a record the
    log does not hold (r99), of the grain's own record (r6), of a split no longer in force (r5, a
    draw a later revert withdrew), or of an answer in force that is no split (r4, the purpose),
    none of them a split in force. The sweep withdraws nothing,
    and the decision is unlisted in the engine's words and in those that say why no way was
    taken."""
    session = Sealed(names=names)
    if names == "r5":  # an earlier draw, withdrawn in the plan's own log
        session.template = [(5, SPLIT_A), (6, GRAIN_ONE), (7, SPLIT_A), (8, SPLIT_B),
                            (9, {"kind": "revert", "decision_id": "r5"})]
        session.log = list(session.template)
        assert "r5" not in {r.id for r in session._splits()}
    if names == "r4":  # an answer in force that is no split
        session.template = [(4, {"kind": "set_purpose", "purpose": "prediction"}),
                            (6, GRAIN_ONE), (7, SPLIT_A), (8, SPLIT_B)]
        session.log = list(session.template)
        assert "r4" in {r.id for r in W.live(session.view())}
    report, refits = swept_sealed(tmp_path, session)
    (unlisted,) = report["unlisted"]
    assert unlisted["point"] == GRAIN.id and SEALED_WORDS in unlisted["why"]
    assert unlisted["why"].endswith(f"Its way forward withdraws record {names}, which the log "
                                    f"does not hold in force as a split.")
    assert [c for c in session.calls if c[0] == "revisit" and c[1] != "r6"] == []
    assert [c for c in session.calls if c[0] in ("decide", "answer")] == []
    assert report["complete"] is False and W.passed(report) is False


LINEAR = {"kind": "select_models", "models": ["linear"]}


class Logged(Walked):
    """A copy of a plan whose log answers the models slot twice, neither answer reverted: record
    12 (linear) before record 13 reads `stage` as a code, and record 14 (the two families) after
    it, so the latest, 14, is the one in force (``decisions.fold``). The engine's guard as
    ``decisions._answers_keep_settled_readings`` holds it: reverting the reading is refused while
    any models answer is in force. Each post is a record of the log, so what the refit leaves in
    force is the fold of what it posted; while the reading is reverted, the ask asks it."""

    def __init__(self):
        super().__init__([])
        self.log = [(12, LINEAR), (13, STAGE_CODE), (14, TWO_MODELS)]

    def view(self):
        return {**view_at(self.log), "stages": self.stages, "state": {}}

    def _post(self, decision):
        self.log.append((max(seq for seq, _ in self.log) + 1, dict(decision)))
        return arms.Said("ok", {})

    def revisit(self, record, reason):
        if record == "r13" and any(r.decision.kind == "select_models"
                                   for r in W.live(self.view())):
            return arms.Said("refused", {}, by="server", code="role_unconfirmed",
                             message=READS_STAGE,
                             exits=[{"label": "Keep the answers as they are", "decision": None}])
        return self._post({"kind": "revert", "decision_id": record})

    def ask(self):
        from research.swarm.tests.test_arms import opt

        if any(r.decision.kind == "confirm_reading" for r in W.live(self.view())):
            return {"question": None, "fit_reached": True}
        return {"question": {"key": "models", "status": "open"},
                "readings": {"exits": [opt("x-code", STAGE_CODE), opt("x-count", STAGE_COUNT)]}}

    def answer(self, ids, sets, reason, marker=None):
        self.answered.append(list(ids))
        return self._post({"x-code": STAGE_CODE, "x-count": STAGE_COUNT}[ids[0]])

    def decide(self, decision, reason):
        self.answered.append(dict(decision))
        return self._post(decision)

    def fit(self, wait=0):
        return arms.Said("refused", {"fit": {}}, by="driver", code="not_here",
                         message="No fit is made here.")


def view_at(log):
    at = datetime(2026, 10, 23, tzinfo=timezone.utc).isoformat()
    return {"decisions": [{"id": f"r{seq}", "seq": seq, "at": at, "recorded_by": "you",
                           "decision": d} for seq, d in log]}


def test_the_answers_withdrawn_first_are_answered_again_in_the_plans_order(tmp_path):
    """P3-F1: both models answers are withdrawn before the reading can be revisited, the latest
    first; answered again in the plan's order, the plan's latest is the one in force again, so
    the refit is the plan with the option and nothing else (its digest that plan's, built here)."""
    from turbotab.core import decisions as D

    session = Logged()
    sweep = a_sweep(tmp_path)
    plan = W.Refit(W.PLAN, None, None, kind="scored", digest=W.plan_digest(session.view()))
    refit = sweep.refit(session, STAGE, {"id": "x-count", "decision": STAGE_COUNT}, [], plan)
    assert [w["seq"] for w in refit.withdrawn] == [14, 12]  # in the order withdrawn
    assert session.answered == [["x-count"], LINEAR, TWO_MODELS]  # again in the plan's order
    state = D.fold(W.records_of(session.view()))
    assert state.models == TWO_MODELS["models"]
    with_option = view_at([(12, LINEAR), (13, STAGE_COUNT), (14, TWO_MODELS)])
    assert refit.digest == W.plan_digest(with_option) != plan.digest


# ── a refusal's way forward after an option: only one the plan or the option names ─────────

PLAN_MISSING = {"kind": "set_missing", "strategy": "complete_case"}
UNNAMED = {"kind": "set_split", "holdout": 0.3}


class WalkedOnPlan(Walked):
    """The walk after an option meets a question whose first option is refused with ``exits``,
    on a plan whose log holds :data:`PLAN_MISSING`."""

    def __init__(self, exits):
        from research.swarm.tests.test_arms import ask, opt

        q = ask("grain_like", [opt("a", {"kind": "set_grain_like", "v": "a"}),
                               opt("b", {"kind": "set_grain_like", "v": "b"})])
        super().__init__([q, q, q, {"question": None, "fit_reached": True}])
        self.exits = [{"id": f"x-{i}", "label": f"way {i}", "decision": d}
                      for i, d in enumerate(exits, start=1)]

    def view(self):
        return {**view_of([("you", PLAN_MISSING)]), "stages": self.stages, "state": {}}

    def answer(self, ids, sets, reason, marker=None):
        self.answered.append(list(ids))
        if list(ids) == ["a"]:
            return arms.Said("refused", {}, by="server", code="not_here", message="Not here.",
                             exits=self.exits)
        self.asks = [{"question": None, "fit_reached": True}]  # answered: Fit is reached
        return arms.Said("ok", {})


def walked_refit(tmp_path, session):
    sweep = a_sweep(tmp_path)
    plan = W.Refit(W.PLAN, None, None, kind="scored", digest=W.plan_digest(session.view()))
    return sweep.refit(session, POINT, OPTION, [], plan)


@pytest.mark.parametrize("exits, taken, by", [
    ([UNNAMED, PLAN_MISSING], "x-2", "the plan"),
    ([UNNAMED, OPTION["decision"]], "x-2", "the option")], ids=["the_plan", "the_option"])
def test_after_an_option_the_walk_takes_the_way_forward_the_plan_or_the_option_names(
        tmp_path, exits, taken, by):
    session = WalkedOnPlan(exits)
    refit = walked_refit(tmp_path, session)
    assert session.answered[1:] == [["a"], [taken]]  # after the option's own (``decide``)
    assert refit.ways == [{"id": taken, "label": "way 2", "decision": exits[1], "named_by": by}]
    assert refit.kind == "duplicate"


def test_after_an_option_the_walk_takes_no_way_forward_the_plan_and_the_option_never_chose(
        tmp_path):
    session = WalkedOnPlan([UNNAMED])
    refit = walked_refit(tmp_path, session)
    assert ["x-1"] not in session.answered and refit.ways == []
    assert session.answered[1:] == [["a"], ["b"]]  # the question's next option, as with none
    assert refit.kind == "duplicate"


STAGE_AMOUNT = {**STAGE_CODE, "value": "amount"}


class ServedAfter(Walked):
    """The walk after an option meets a question whose first option, ``a``, is refused with no
    way forward in its answer until the reading ``way`` is recorded, on a plan whose log holds
    ``plan``. The refusal's way is served on the asks after it as the driver serves a last
    refusal's (``last_refusal_exits``) until an answer is recorded, which clears it as the
    driver's ``answer`` does; or, with ``card``, the ask card asks the reading itself
    (``readings.exits``) until it is recorded."""

    def __init__(self, way, *, card=False, plan=PLAN_MISSING):
        from research.swarm.tests.test_arms import ask, opt

        super().__init__([])
        self.q = ask("grain_like", [opt("a", {"kind": "set_grain_like", "v": "a"}),
                                    opt("b", {"kind": "set_grain_like", "v": "b"})])
        self.way = {"id": "x-1", "label": "way 1", "decision": way}
        self.card, self.plan = card, plan
        self.kept, self.refused, self.read, self.done = [], False, False, False

    def view(self):
        return {**view_of([("you", self.plan)]), "stages": self.stages, "state": {}}

    def ask(self):
        if self.done:
            return {"question": None, "fit_reached": True}
        q = dict(self.q)
        if self.kept:
            q["last_refusal_exits"] = list(self.kept)
        if self.card and self.refused and not self.read:
            q["readings"] = {"exits": [dict(self.way)]}
        return q

    def answer(self, ids, sets, reason, marker=None):
        self.answered.append(list(ids))
        if list(ids) == ["a"] and not self.read:
            self.refused = True
            self.kept = [] if self.card else [dict(self.way)]
            return arms.Said("refused", {}, by="server", code="reading_first",
                             message="Read `stage` first.", exits=[])
        self.kept = []  # an answer recorded: no last refusal's way is kept
        if list(ids) == ["x-1"]:
            self.read = True
        else:
            self.done = True
        return arms.Said("ok", {})


def test_after_an_option_a_refusals_way_served_later_is_not_taken_where_none_names_it(
        tmp_path):
    """P3-F3: the way forward a refusal leaves for the asks after it (a reading neither the plan
    nor the option records) is not taken; the walk goes on as with no way forward."""
    session = ServedAfter(STAGE_AMOUNT)
    refit = walked_refit(tmp_path, session)
    assert ["x-1"] not in session.answered and refit.ways == []
    assert session.answered[1:] == [["a"], ["b"]]  # the question's next option, as with none
    assert refit.kind == "duplicate"


def test_after_an_option_a_refusals_way_served_later_is_taken_where_the_plan_names_it(
        tmp_path):
    session = ServedAfter(STAGE_AMOUNT, plan=STAGE_AMOUNT)
    refit = walked_refit(tmp_path, session)
    assert session.answered[1:3] == [["a"], ["x-1"]]
    assert refit.ways == [{"id": "x-1", "label": "way 1", "decision": STAGE_AMOUNT,
                           "named_by": "the plan"}]


def test_after_an_option_a_reading_the_ask_card_asks_itself_is_answered_as_the_default_path_does(
        tmp_path):
    """The reading the card asks is no refusal's way: the default path's policy answers it,
    named or not, and it is not recorded as a way forward taken."""
    session = ServedAfter(STAGE_AMOUNT, card=True)
    refit = walked_refit(tmp_path, session)
    assert ["x-1"] in session.answered and refit.ways == []


def test_a_named_way_needs_every_field_the_way_records():
    assert W.names_way({"kind": "set_missing", "strategy": "impute", "m": 20},
                       {"kind": "set_missing", "strategy": "impute"}) is True
    assert W.names_way({"kind": "set_missing", "strategy": "complete_case"},
                       {"kind": "set_missing", "strategy": "impute"}) is False
    assert W.names_way({"kind": "set_missing"}, {}) is False


# ── the q-value beside each estimate read from the locked fit ────────────────


def test_an_estimate_read_from_the_locked_fit_keeps_its_rows_q_value():
    rows = [{"family": "featurewise", "feature": f, "setting": None, "estimate": e,
             "ci_low": e - 1, "ci_high": e + 1, "why": None}
            for f, e in (("mz_0001", 1.0), ("mz_0002", 0.5))]
    rows.append({"family": "linear", "feature": "mz_0001", "setting": None, "estimate": 2.0,
                 "ci_low": 1.0, "ci_high": 3.0, "why": None})
    got = {"source": "fits", "attempt": "f2", "estimates": rows}
    fits = [{"id": "f1", "effects": {"featurewise": [{"feature": "mz_0001", "q": 0.9}]}},
            {"id": "f2", "effects": {"featurewise": [{"feature": "mz_0001", "q": 0.01},
                                                     {"feature": "mz_0002", "q": None}],
                                     "linear": [{"feature": "mz_0001", "estimate": 2.0}]}}]
    kept = W.with_q(got, fits)
    assert [r.get("q", "none served") for r in kept] == [0.01, None, "none served"]
    assert [{k: v for k, v in r.items() if k != "q"} for r in kept] == rows
    assert W.with_q(got | {"source": "results"}, fits) == rows  # the results' rows: as held


# ── the compute cap of a dataset's sweeps around named plans ─────────────────

THREE = [{"id": o, "label": o, "decision": {"kind": "set_missing", "strategy": o},
          "needs": None} for o in ("impute", "multiple_imputation", "drop")]


def capped(tmp_path, monkeypatch, *, cap, spent):
    """A sweep around a named plan whose plan's fit took 2 s and each refit's 3 s, with three
    options of the missing values left: the refits made, and the report."""
    sweep = W.Sweep(tmp_path / "sweep", dataset="tiny", question=arms.Question(
        target="y", purpose="prediction"), crash=False, plan="named", cap=cap, spent=spent)
    sweep.base = {"project": "p0", "digest": "d0", "seq": 9, "scoring": {"set": False},
                  "points": [MISSING.json()], "skipped": []}
    sweep.done[W.PLAN] = W.Refit(W.PLAN, None, None, kind="scored", fit_seconds=2.0)
    sweep.offered = {MISSING.id: {"point": MISSING.id, "why": None, "options": THREE}}
    fitted = []

    def refit(session, point, option, later, plan):
        fitted.append(option["id"])
        return W.Refit(point.id, option["id"], option["decision"], kind="scored",
                       fit_seconds=3.0)

    monkeypatch.setattr(sweep, "refit", refit)
    return fitted, sweep.run(Walked([]), lambda s: "p0")


def test_the_sweep_stops_before_a_fit_once_the_datasets_compute_cap_is_spent(tmp_path,
                                                                            monkeypatch):
    # 4 s before and 2 s the plan's: 6 < 10; a refit: 9 < 10; another: 12, and none after it.
    fitted, report = capped(tmp_path, monkeypatch, cap=10.0, spent=4.0)
    assert fitted == ["impute", "multiple_imputation"]
    assert report["stopped"]["cap"] == 10.0 and report["stopped"]["spent"] == 12.0
    assert report["stopped"]["why"] == (
        "The sweeps of tiny stopped at its compute cap: 12 of 10 fit seconds spent "
        "(prereg.json's compute_cap), before the next fit.")
    assert report["cap"] == {"seconds": 10.0, "spent": 12.0, "spent_before": 4.0}
    assert report["complete"] is False and report["unvisited"] == [f"{MISSING.id}=drop"]
    assert "## Stopped at the compute cap" in W.table_md(report)


@pytest.mark.parametrize("cap", [None, 15.1], ids=["no_cap", "a_cap_not_reached"])
def test_a_cap_not_reached_stops_no_fit(tmp_path, monkeypatch, cap):
    fitted, report = capped(tmp_path, monkeypatch, cap=cap, spent=4.0)  # 15 s at the end
    assert fitted == ["impute", "multiple_imputation", "drop"] and report["stopped"] is None
    assert report["complete"] is True


def written_sweep(folder, name, *, dataset, crash, seconds):
    folder = folder / name
    folder.mkdir(parents=True)
    lines = [{"template": {"project": "p", "dataset": dataset, "crash": crash, "points": [],
                           "skipped": []}}]
    lines += [{"refit": W.Refit(W.PLAN if i == 0 else "question:missing",
                                None if i == 0 else str(i), None, kind="scored",
                                fit_seconds=s).json()}
              for i, s in enumerate(seconds)]
    (folder / "progress.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return folder


def test_a_datasets_earlier_sweeps_spend_counts_its_named_plans_but_not_its_crash_sweep(
        tmp_path):
    root = tmp_path / "sweeps"
    written_sweep(root, "haiku", dataset="tiny", crash=False, seconds=[2.0, 3.0])
    written_sweep(root, "sonnet", dataset="tiny", crash=False, seconds=[1.5])
    written_sweep(root, "default", dataset="tiny", crash=True, seconds=[100.0])
    written_sweep(root, "elsewhere", dataset="other", crash=False, seconds=[50.0])
    me = written_sweep(root, "opus", dataset="tiny", crash=False, seconds=[9.0])
    assert W.spent_by(root, "tiny", but=me) == 6.5
    assert W.spent_by(root, "tiny") == 15.5


def test_the_cap_is_prereg_jsons_and_a_dataset_without_one_is_refused_in_words():
    prereg = {"filled": {"compute_cap": {"seconds": 30.0, "by_dataset": {
        "tiny": {"plans": 2, "crash_seconds": 12.5, "seconds": 25.0}}}}}
    assert W.compute_cap("tiny", prereg) == 25.0
    for given in ({"filled": {"compute_cap": None}}, prereg):
        with pytest.raises(arms.Stuck, match="prereg.json holds no compute cap for the sweeps "
                                             "of other"):
            W.compute_cap("other", given)


# ── the options that take fields of their own (the crash sweep) ──────────────
# The card's rules, read back from the sentences the intake writes for them (``intake.age_line``,
# ``intake.level_line``) and from the pilot cards themselves; each answered through the driver's
# own composer (``options.compose`` and ``resolve``, over a question built here) and checked
# against the rule the card states. Each value built from a table here is checked against the
# table: a range's bounds by the percentile's own definition, worked by hand; the rarest level by
# counting; the adult floor from the CDC growth charts' last age; the adjustment set from the
# roles. Each builder both ways: where nothing can be built, the option stays unreached, with why.

def intake_sentences():
    """The intake's own sentences for an age range, a level rule naming a refused and a
    don't-know code, and a level rule naming neither."""
    from types import SimpleNamespace

    from research.swarm import intake

    coded = {"1": "Yes", "2": "No", "3": "Borderline", "7": "Refused", "9": "Don't know"}
    told = intake.level_line("Not told", "DIQ010", "1", coded,
                             intake.blanks_line("DIQ010", 1, 50, "with an outcome"))
    plain = intake.level_line("Not pregnant", "RIDEXPRG", "1",
                              {"1": "Yes", "2": "No", "3": "Cannot tell"},
                              intake.blanks_line("RIDEXPRG", 20, 50, "with an outcome"))
    age = intake.age_line("Aged 20 or older", "RIDAGEYR", 20, {
        "1999-2000": {"RIDAGEYR": SimpleNamespace(codes={"85": ">= 85 years of age"})}})
    return age, told, plain


PROSE = "The table keeps every participant: the population is the analyst's, each as a rule."


def test_a_cards_rules_are_read_back_from_the_sentences_the_intake_writes():
    age, told, plain = intake_sentences()
    rules, unread = W.card_rules({"eligibility": [PROSE, age, told, plain]})
    assert unread == []
    assert [(r.place, r.kind, r.column) for r in rules] == [
        (2, "range", "RIDAGEYR"), (3, "levels", "DIQ010"), (4, "levels", "RIDEXPRG")]
    assert rules[0].low == "20" and rules[0].handlings == ()  # the age sentence names none
    assert (rules[1].exclude, rules[1].refused, rules[1].dont_know) == (("1",), ("7",), ("9",))
    assert (rules[2].exclude, rules[2].refused, rules[2].dont_know) == (("1",), (), ())
    # "leaves out a row it cannot confirm ... unless it keeps them all": both handlings named.
    assert rules[1].handlings == rules[2].handlings == ("exclude", "keep")
    assert [r.words for r in rules] == [age, told, plain]


def test_a_sentence_naming_a_rule_the_sweep_cannot_read_is_said_and_prose_is_not_a_rule():
    vague = "Pregnant women excluded: a level rule on RIDEXPRG, its levels the analyst's."
    rules, unread = W.card_rules({"eligibility": [PROSE, vague]})
    assert rules == [] and unread == [{"place": 2, "words": vague}]
    assert W.card_rules({}) == ([], []) and W.card_rules(None) == ([], [])


@pytest.mark.parametrize("dataset", ["bang2009", "nhanes_glucose_predict"])
def test_a_pilot_cards_rules_are_every_rule_it_states_in_its_order(dataset):
    card = arms.card_of(dataset)
    rules, unread = W.card_rules(card)
    stated = [i for i, s in enumerate(card["eligibility"], 1) if " rule on " in s]
    assert unread == [] and [r.place for r in rules] == stated and len(stated) == 5
    for r in rules:
        said = card["eligibility"][r.place - 1]
        if r.kind == "range":
            assert f"a range rule on {r.column} from {r.low}." in said
            continue
        (level,) = r.exclude
        assert f"a level rule on {r.column} excluding level {level} (" in said
        assert (f"naming {r.refused[0]} as its refused code and {r.dont_know[0]} as its "
                f"don't-know code" if r.refused else
                "which has no refused or don't-know code") in said
    assert [r.kind for r in rules].count("range") == 1


def exclusions_composed(columns, target, dtype="integer"):
    """The exclusions question as the driver composes it (``options.compose``) over a table of
    ``columns``, every one numeric, no screen offered."""
    from research.swarm import options as opts

    entry = {"options": [{"value": v, "label": v}
                         for v in ("none", "custom", "eligibility", "levels")]}
    served = {"proposals": {}, "oriented": {"columns": [{"name": c, "dtype": dtype}
                                                        for c in [*columns, target]]}}
    return opts.compose("exclusions", {"key": "exclusions", "status": "open"},
                        {"state": {"target": target}}, served.get, entry)


EXCLUSIONS = W.Point("question", "exclusions", 8, "r8", {"kind": "set_exclusions", "rules": []})
BANG_COLUMNS = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDEXPRG", "DIQ010", "DIQ050", "DIQ070"]


def option(oid):
    return {"id": oid, "label": oid, "decision": None, "needs": None}


def test_the_cards_rules_answer_its_range_and_each_level_rule_under_each_handling_it_names():
    """Bang's card: its age range answers the eligibility range (who the study is about), each
    level rule the level rule under each handling of a row it cannot confirm the card names, and
    its whole population one answer of its rules in its order, once under each handling; each
    answer, resolved by the driver's own composer, is the card's rule."""
    from research.swarm import options as opts

    card = arms.card_of("bang2009")
    rules, _ = W.card_rules(card)
    fields = W.Fields(arms.Question.of_card(card), card=card)
    composed = exclusions_composed(BANG_COLUMNS, "diabetes_fpg")
    (age,), why = fields.answers(EXCLUSIONS, option("eligibility"))
    assert why is None and age.source == "card" and age.ids == ["eligibility"]
    (r,) = opts.resolve(composed, age.ids, age.sets)["rules"]
    (range_rule,) = [x for x in rules if x.kind == "range"]
    assert (r["kind"], r["column"], r["low"], r["high"]) == (
        "range", range_rule.column, float(range_rule.low), None)
    assert r["missing"] == opts.CUSTOM_RANGE["missing"]  # the age sentence names no handling
    assert age.option == f"eligibility@card:{range_rule.place}:{range_rule.column}"
    levels, why = fields.answers(EXCLUSIONS, option("levels"))
    level_rules = [x for x in rules if x.kind == "levels"]
    assert why is None and [(a.source, a.ids) for a in levels] == [("card", ["levels"])] * 8
    got = [opts.resolve(composed, a.ids, a.sets)["rules"][0] for a in levels]
    assert [(g["column"], g["exclude"], g["refused"], g["dont_know"], g["missing"])
            for g in got] == [(x.column, list(x.exclude), list(x.refused), list(x.dont_know), h)
                              for x in level_rules for h in ("exclude", "keep")]
    assert levels[0].option == (f"levels@card:{level_rules[0].place}:{level_rules[0].column}"
                                f":exclude")
    # The whole population: one answer of every rule, in the card's order, under each handling.
    whole = fields.population(EXCLUSIONS, ["none", "custom", "eligibility", "levels"])
    assert [a.option for a in whole] == ["population@card:exclude", "population@card:keep"]
    for a, handling in zip(whole, ("exclude", "keep")):
        assert a.source == "card"
        assert a.ids == ["eligibility" if x.kind == "range" else "levels" for x in rules]
        resolved = opts.resolve(composed, a.ids, a.sets)["rules"]
        assert [x["column"] for x in resolved] == [x.column for x in rules]
        assert [x["missing"] for x in resolved] == [
            opts.CUSTOM_RANGE["missing"] if x.kind == "range" else handling for x in rules]
    # Where the levels are not offered, no population is answered.
    assert fields.population(EXCLUSIONS, ["none", "eligibility"]) == []


def type7(values, p):
    """The ``p``-th percentile of ``values`` by its definition (Hyndman and Fan's type 7, the
    linear interpolation between order statistics), worked by hand."""
    v = sorted(float(x) for x in values)
    h = (len(v) - 1) * p / 100
    lo = math.floor(h)
    return v[lo] + (h - lo) * (v[min(lo + 1, len(v) - 1)] - v[lo])


def built_frame(*, age=True, levels=True):
    """200 rows: an identifier first, a numeric predictor with five blanks, a text predictor
    whose levels are a (120 rows), c and b (40 each), a 1/2 code, an age in months, an outcome."""
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 200
    frame = pd.DataFrame({"SEQN": np.arange(1, n + 1), "x1": rng.normal(50, 10, n).round(2)})
    frame.loc[:4, "x1"] = np.nan
    if levels:
        frame["grp"] = ["a"] * 120 + ["c"] * 40 + ["b"] * 40
        frame["sex"] = rng.integers(1, 3, n)
    if age:
        frame["age_months"] = rng.integers(24, 960, n)
    frame["y"] = rng.integers(0, 2, n)
    roles = {"SEQN": "identifier", **{c: "exposure" for c in frame.columns
                                      if c not in ("SEQN", "y")}}
    return frame, roles


def test_a_built_range_is_a_numeric_predictors_first_and_ninety_ninth_percentiles():
    from research.swarm import options as opts

    frame, roles = built_frame()
    fields = W.Fields(arms.Question(target="y", purpose="prediction"), frame=frame, roles=roles)
    (a,), why = fields.answers(EXCLUSIONS, option("custom"))
    x1 = frame["x1"].dropna()
    low, high = type7(x1, 1), type7(x1, 99)
    assert why is None and a.source == "built" and a.option == "custom@built:x1"
    (r,) = opts.resolve(exclusions_composed(["SEQN", "x1", "sex", "age_months"], "y"),
                        a.ids, a.sets)["rules"]
    assert r["column"] == "x1" and (r["low"], r["high"]) == pytest.approx((low, high), abs=1e-12)
    assert f"{low:g}" in a.words and f"{high:g}" in a.words and "built" in a.words


def test_a_built_eligibility_range_is_the_age_columns_adult_floor_in_its_unit():
    frame, roles = built_frame()
    fields = W.Fields(arms.Question(target="y", purpose="prediction"), frame=frame, roles=roles)
    (a,), why = fields.answers(EXCLUSIONS, option("eligibility"))
    # The CDC 2000 growth charts end at 20 years: an adult is 20 or older, 240 months.
    assert why is None and a.source == "built" and a.option == "eligibility@built:age_months"
    assert a.sets == ["column=age_months", f"low={20 * 12:.1f}"]
    years = W.Fields(arms.Question(target="y", purpose="prediction"),
                     frame=frame.rename(columns={"age_months": "age"}),
                     roles={**roles, "age": "exposure"})
    (b,), _ = years.answers(EXCLUSIONS, option("eligibility"))
    assert b.sets == ["column=age", f"low={20:.1f}"]  # a bare ``age`` is read in years


def test_a_built_level_rule_excludes_the_first_categorical_predictors_rarest_level():
    frame, roles = built_frame()
    fields = W.Fields(arms.Question(target="y", purpose="prediction"), frame=frame, roles=roles)
    (a,), why = fields.answers(EXCLUSIONS, option("levels"))
    counts = frame["grp"].value_counts()
    rarest = sorted(k for k, v in counts.items() if v == counts.min())[0]  # b and c tie: b
    assert why is None and a.source == "built" and a.option == "levels@built:grp"
    assert a.sets == ["level_column=grp", f"exclude={rarest}"] and rarest == "b"


@pytest.mark.parametrize("oid, words", [
    ("eligibility", "no age column"), ("levels", "no categorical predictor"),
    ("custom", "no numeric predictor")])
def test_an_option_with_nothing_to_build_from_stays_unreached_with_why(oid, words):
    frame, roles = built_frame(age=False, levels=False)
    if oid == "custom":
        roles = {"SEQN": "identifier"}  # the roles name no predictor
    fields = W.Fields(arms.Question(target="y", purpose="prediction"), frame=frame, roles=roles)
    answers, why = fields.answers(EXCLUSIONS, option(oid))
    assert answers == [] and "records nothing by itself" in why and words in why
    # Without the table, nothing is built either, and the why says no table was read.
    answers, why = W.Fields(arms.Question(target="y", purpose="prediction")).answers(
        EXCLUSIONS, option(oid))
    assert answers == [] and "read no table" in why


def test_under_estimate_the_declared_exposure_is_never_a_built_value():
    frame, roles = built_frame()
    q = arms.Question(target="y", purpose="inference", exposure="x1")
    (a,), _ = W.Fields(q, frame=frame, roles=roles).answers(EXCLUSIONS, option("custom"))
    assert a.option == "custom@built:age_months"  # x1 the exposure; grp and sex read as levels
    q = arms.Question(target="y", purpose="inference", exposure="grp")
    (a,), _ = W.Fields(q, frame=frame, roles=roles).answers(EXCLUSIONS, option("levels"))
    assert a.option == "levels@built:sex"  # grp the exposure: sex, a 1/2 code, the next


@pytest.mark.parametrize("question, what", [
    (arms.Question(target="y", purpose="inference", exposure="age_months"),
     "the declared exposure"),
    (arms.Question(target="age_months", purpose="prediction"), "the outcome")],
    ids=["exposure", "outcome"])
def test_a_built_eligibility_range_is_never_on_the_outcome_or_the_declared_exposure(question,
                                                                                    what):
    """The table's one age column is the declared exposure, or the outcome: no adult floor is built
    on it, and the option stays unreached, saying why; an age column among the covariates is built
    on, the next age column where the first is the exposure."""
    frame, roles = built_frame()
    answers, why = W.Fields(question, frame=frame, roles=roles).answers(EXCLUSIONS,
                                                                        option("eligibility"))
    assert answers == [] and "records nothing by itself" in why
    assert f"no age column to build an adult floor on but `age_months`, {what}," in why
    # Silent where the age column is a covariate.
    covariate = W.Fields(arms.Question(target="y", purpose="inference", exposure="x1"),
                         frame=frame, roles={**roles, "age_months": "covariate"})
    (a,), why = covariate.answers(EXCLUSIONS, option("eligibility"))
    assert why is None and a.option == "eligibility@built:age_months"
    # Two age columns, the first read (its name spells its unit) the exposure: the second's floor.
    both = W.Fields(arms.Question(target="y", purpose="inference", exposure="age_months"),
                    frame=frame.assign(age=frame["age_months"] // 12),
                    roles={**roles, "age": "covariate"})
    (a,), why = both.answers(EXCLUSIONS, option("eligibility"))
    assert why is None and a.option == "eligibility@built:age"
    assert a.sets == ["column=age", f"low={20:.1f}"]  # a bare ``age`` is read in years


ADJUSTMENT = W.Point("question", "adjustment", 11, "r11", {
    "kind": "set_adjustment", "exposure": "walks", "answers": {
        c: {"causes_exposure": "unknown", "causes_outcome": "unknown", "after_exposure": "unknown"}
        for c in ("age", "pain", "height")}})


def test_an_adjustment_set_of_ones_own_is_the_covariates_the_roles_name_each_a_confounder():
    from research.swarm import options as opts

    q = arms.Question(target="sbp", purpose="inference", exposure="walks")
    roles = {"id": "identifier", "age": "covariate", "pain": "exposure", "walks": "exposure",
             "height": "covariate"}
    (a,), why = W.Fields(q, roles=roles).answers(ADJUSTMENT, option("answers"))
    assert why is None and a.source == "built" and a.option == "answers@built:covariates"
    groups = [{"key": "g", "label": "g", "columns": ["age", "pain", "height"]}]
    composed = opts.compose("adjustment", {"key": "adjustment", "status": "open"}, {"state": {}},
                            {"proposals": {"adjustment": {"exposure": "walks",
                                                          "groups": groups}}}.get, None)
    # A confounder is adjusted for; a cause of neither, not after the exposure, is left out (the
    # modified disjunctive cause criterion, VanderWeele 2019): the set is the two covariates.
    confounder = {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}
    neither = {"causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"}
    resolved = opts.resolve(composed, a.ids, a.sets)
    assert resolved == {"kind": "set_adjustment", "exposure": "walks",
                        "answers": {"age": confounder, "pain": neither, "height": confounder}}
    assert list(resolved["answers"]) == ["age", "pain", "height"]  # the plan's order
    from turbotab.core.estimand import derive

    assert [c for c, x in resolved["answers"].items() if derive(x).adjusted] == ["age", "height"]
    answers, why = W.Fields(q, roles={**roles, "age": "exposure", "height": "exposure"}).answers(
        ADJUSTMENT, option("answers"))
    assert answers == [] and "records nothing by itself" in why and "no covariate" in why


# ── the adjustment answers toggled one column at a time around an Estimate plan (F4) ──

CONFOUNDER = {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}
NEITHER = {"causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"}
UNKNOWN = {"causes_exposure": "unknown", "causes_outcome": "unknown", "after_exposure": "unknown"}
TOGGLED = W.Point("question", "adjustment", 11, "r11", {
    "kind": "set_adjustment", "exposure": "walks",
    "answers": {"age": CONFOUNDER, "pain": UNKNOWN, "height": NEITHER}})
ESTIMATE_Q = arms.Question(target="sbp", purpose="inference", exposure="walks")
GUESSED = {"id": "confounder", "label": "Confounders",
           "decision": {"kind": "set_adjustment", "exposure": "walks",
                        "answers": {"age": CONFOUNDER}}, "needs": None, "columns": ["age"]}
ANSWERS = {"id": "answers", "label": None, "decision": None, "needs": None,
           "columns": ["age", "pain", "height"]}


def toggling_sweep(tmp_path, *, crash, question=ESTIMATE_Q):
    sweep = W.Sweep(tmp_path / "sweep", dataset="tiny", question=question, crash=crash,
                    plan="default" if crash else "haiku-conventional")
    sweep.base = {"project": "p0", "digest": "d0", "seq": 11, "scoring": {"set": False},
                  "points": [TOGGLED.json()], "skipped": []}
    sweep.fields = W.Fields(question, roles={"age": "covariate", "pain": "exposure",
                                             "height": "covariate", "walks": "exposure"})
    return sweep


@pytest.mark.parametrize("crash", [False, True], ids=["named_plan", "crash"])
def test_around_an_estimate_plan_each_offered_column_is_toggled_with_the_others_fixed(
        tmp_path, crash):
    """F4: one refit per column the adjustment question offers (the answers option's columns, as
    the driver composes it), that column answered in where the plan leaves it out and out where
    the plan adjusts for it, every other column answered as the plan answered it. By the modified
    disjunctive cause criterion (VanderWeele 2019): age, a cause of the exposure and the outcome,
    is adjusted for, so it is toggled out; pain, of unknown timing, is declared without it beside
    with it (out of the primary model), and height, a cause of neither, is left out, so both are
    toggled in. In is a confounder's answers (yes, yes, no), out a cause of neither's (no, no,
    no). Around a named plan, too, where the answers option is otherwise unreached."""
    from turbotab.core.estimand import derive

    sweep = toggling_sweep(tmp_path, crash=crash)
    listed = {"point": TOGGLED.id, "why": None, "options": [GUESSED, ANSWERS]}
    planned = {p["id"]: p for p in sweep._planned(object(), TOGGLED, listed)}
    plan = TOGGLED.decision["answers"]
    expected = {"age": ("out", NEITHER), "pain": ("in", CONFOUNDER), "height": ("in", CONFOUNDER)}
    for column, (way, answer) in expected.items():
        p = planned[f"{TOGGLED.id}=answers@toggle:{column}:{way}"]
        fielded = p["fielded"]
        assert fielded.source == "toggle" and fielded.ids == ["answers"]
        assert fielded.decision == {"kind": "set_adjustment", "exposure": "walks",
                                    "answers": {**plan, column: answer}}
        assert list(fielded.decision["answers"]) == ["age", "pain", "height"]  # the plan's order
        # The engine's criterion agrees: the column flips, and no other moves.
        flipped = {c: derive(a).adjusted for c, a in fielded.decision["answers"].items()}
        assert flipped == {c: derive(a).adjusted != (c == column) for c, a in plan.items()}
        assert f"`{column}` toggled {way}" in fielded.words
    assert f"{TOGGLED.id}=confounder" in planned  # an option that records itself, as listed
    if not crash:  # around a named plan the toggles answer the answers option: none unreached
        assert not [p for p in planned.values() if p.get("why")]
        assert set(planned) == {f"{TOGGLED.id}=confounder", *(
            f"{TOGGLED.id}=answers@toggle:{c}:{w}" for c, (w, _) in expected.items())}
    else:  # beside the crash sweep's own set of one's own
        assert f"{TOGGLED.id}=answers@built:covariates" in planned


def test_no_toggle_is_made_under_predict_for_another_question_or_without_the_plans_answers(
        tmp_path):
    """The guard's other side: under Predict, for another question's options, and for an answers
    option the plan's record holds no answers for, no column is toggled: around a named plan the
    answers option stays unreached, with why."""
    listed = {"point": TOGGLED.id, "why": None, "options": [GUESSED, ANSWERS]}
    predict = toggling_sweep(tmp_path / "p", crash=False,
                             question=arms.Question(target="sbp", purpose="prediction"))
    other = W.Point("question", "missing", 11, "r11", TOGGLED.decision)
    unanswered = W.Point("question", "adjustment", 11, "r11", None)
    for sweep, point in ((predict, TOGGLED),
                         (toggling_sweep(tmp_path / "o", crash=False), other),
                         (toggling_sweep(tmp_path / "u", crash=False), unanswered)):
        planned = sweep._planned(object(), point, {**listed, "point": point.id})
        assert not [p for p in planned if "@toggle:" in p["id"]]
        (bare,) = [p for p in planned if p["option"] == "answers"]
        assert bare["why"].startswith(W.NOTHING)


class Adjusting(Walked):
    """A copy of an Estimate plan whose revisit opens the adjustment question again with its
    answers option over three columns; each preview and post kept in ``calls``; once an
    adjustment is posted, Fit is reached. Every refit is the plan's duplicate (no Fit pressed)."""

    def __init__(self):
        super().__init__([])
        self.calls, self.reopened = [], False

    def revisit(self, record, reason):
        self.reopened = True
        return arms.Said("ok", {})

    def ask(self):
        from research.swarm.tests.test_arms import ask

        if self.reopened:
            return ask("adjustment", [GUESSED, ANSWERS])
        return {"question": None, "fit_reached": True}

    def preview_of(self, decision):
        self.calls.append(("preview_of", dict(decision)))
        return arms.Said("ok", {"previewed": dict(decision)})

    def decide(self, decision, reason):
        self.reopened = False
        self.calls.append(("decide", dict(decision)))
        return arms.Said("ok", {})

    def answer(self, ids, sets, reason, marker=None):
        self.reopened = False
        self.calls.append(("answer", list(ids), list(sets)))
        return arms.Said("ok", {})


@pytest.mark.parametrize("crash", [False, True], ids=["named_plan", "crash"])
def test_a_toggle_is_previewed_and_posted_as_the_decision_it_composed(tmp_path, crash):
    """A toggle's refit posts the plan's adjustment with its one column flipped, as composed (the
    driver's answers option takes three answers a column, and the plan's may hold more), the crash
    sweep previewing it first; the refit keeps the decision and what it was answered with."""
    sweep = toggling_sweep(tmp_path, crash=crash)
    session = Adjusting()
    sweep.done[W.PLAN] = W.Refit(W.PLAN, None, None, kind="scored",
                                 digest=W.plan_digest(session.view()))
    report = sweep.run(session, lambda s: "p0")
    rid = f"{TOGGLED.id}=answers@toggle:pain:in"
    refit = {r["id"]: r for r in report["refits"]}[rid]
    decision = {"kind": "set_adjustment", "exposure": "walks",
                "answers": {**TOGGLED.decision["answers"], "pain": CONFOUNDER}}
    assert refit["decision"] == decision and refit["answered_with"]["source"] == "toggle"
    assert refit["answered_with"]["decision"] == decision
    posted = [c for c in session.calls if c[0] in ("preview_of", "decide")
              and c[1] == decision]
    assert posted == ([("preview_of", decision)] if crash else []) + [("decide", decision)]
    assert refit["answered"][0]["answer"] == rid and refit["answered"][0]["kind"] == "ok"
    assert {f["refit"] for f in report["fielded"]} >= {rid}


# The plan answered the adjustment over two records (the group taps): age, pain and height
# first, then height again and chol. Each is in force by column, height by its later answer.
CRITERION = tuple(CONFOUNDER)
OVER_TWO = [(1, {"kind": "set_purpose", "purpose": "inference"}),
            (2, {"kind": "set_target", "column": "sbp"}),
            (3, {"kind": "set_roles", "roles": {"age": "covariate", "pain": "exposure",
                                                "height": "covariate", "chol": "covariate",
                                                "walks": "exposure"}}),
            (4, {"kind": "set_estimand", "exposure": "walks", "measure": "mean_difference"}),
            (5, {"kind": "set_adjustment", "exposure": "walks",
                 "answers": {"age": CONFOUNDER, "pain": UNKNOWN, "height": CONFOUNDER}}),
            (6, {"kind": "set_adjustment", "exposure": "walks",
                 "answers": {"height": NEITHER, "chol": NEITHER}})]
LATEST = W.Point("question", "adjustment", 6, "r6", OVER_TWO[-1][1])


def criterion(answers):
    return {c: {k: a[k] for k in CRITERION} for c, a in answers.items()}


def test_the_plans_adjustment_answers_in_force_are_its_records_merged_by_column():
    """P2: the answers in force are every live set_adjustment record's, by column, as the engine
    folds them: age and pain from the earlier record, height by the later one's answer, chol
    from the later one, in the order first answered. By construction, the silent sides: the later
    record withdrawn leaves height at its earlier answer and chol unanswered; a column whose role
    a later answer leaves out of the model (excluded) is not one the question takes; and answers
    given for another exposure than the one the estimand now names are none in force."""
    got = W.adjustment_in_force(view_at(OVER_TWO))
    assert list(got) == ["age", "pain", "height", "chol"]
    assert criterion(got) == {"age": CONFOUNDER, "pain": UNKNOWN, "height": NEITHER,
                              "chol": NEITHER}
    assert all("exposure" not in a for a in got.values())
    reverted = W.adjustment_in_force(view_at([*OVER_TWO, (7, {"kind": "revert",
                                                              "decision_id": "r6"})]))
    assert criterion(reverted) == {"age": CONFOUNDER, "pain": UNKNOWN, "height": CONFOUNDER}
    roles = {**OVER_TWO[2][1]["roles"], "chol": "excluded"}
    left_out = W.adjustment_in_force(view_at([*OVER_TWO, (7, {"kind": "set_roles",
                                                              "roles": roles})]))
    assert list(left_out) == ["age", "pain", "height"]
    other = W.adjustment_in_force(view_at([*OVER_TWO, (7, {
        "kind": "set_estimand", "exposure": "pain", "measure": "mean_difference"})]))
    assert other == {}


class OnTemplate(Walked):
    """The template's log: :data:`OVER_TWO`."""

    def __init__(self):
        super().__init__([])

    def view(self):
        return {**view_at(OVER_TWO), "stages": self.stages, "state": {}}


def test_the_template_gives_the_toggles_the_plans_answers_in_force():
    fields = W.Fields.of_template(OnTemplate(), ESTIMATE_Q, None, None, "p0")
    assert fields.adjustment == W.adjustment_in_force(view_at(OVER_TWO))
    assert list(fields.adjustment) == ["age", "pain", "height", "chol"]


@pytest.mark.parametrize("crash", [False, True], ids=["named_plan", "crash"])
def test_a_column_the_plan_answered_in_an_earlier_record_is_toggled_too(tmp_path, crash):
    """P2, fires: the plan's latest adjustment record answers height and chol, an earlier one age
    and pain; the reopened question's answers option offers the columns the revisit withdrew
    (height and chol). Each of the four is toggled, age and pain among them, each composed of
    the plan's answers in force with that one column flipped by the engine's criterion, the
    other three as in force (height by its later answer), so the revisit's withdrawn columns are
    answered again with the rest. Silent: a column the option offers that the plan never answered
    (bmi), and one the latest record answers that the question no longer takes (gone, in neither
    the answers in force nor the option's columns), are neither toggled nor answered."""
    from turbotab.core.estimand import derive

    in_force = W.adjustment_in_force(view_at(OVER_TWO))
    latest = W.Point("question", "adjustment", 6, "r6", {
        **LATEST.decision, "answers": {**LATEST.decision["answers"], "gone": NEITHER}})
    sweep = toggling_sweep(tmp_path, crash=crash)
    sweep.fields = W.Fields(ESTIMATE_Q, roles=OVER_TWO[2][1]["roles"], adjustment=in_force)
    reopened = {**ANSWERS, "columns": ["height", "chol", "bmi"]}
    listed = {"point": latest.id, "why": None, "options": [reopened]}
    planned = {p["id"]: p for p in sweep._planned(object(), latest, listed)}
    toggles = {k.split("=", 1)[1]: p["fielded"] for k, p in planned.items() if "@toggle:" in k}
    expected = {"age": "out", "pain": "in", "height": "in", "chol": "in"}
    assert list(toggles) == [f"answers@toggle:{c}:{w}" for c, w in expected.items()]
    for column, way in expected.items():
        fielded = toggles[f"answers@toggle:{column}:{way}"]
        answers = fielded.decision["answers"]
        assert list(answers) == ["age", "pain", "height", "chol"]
        assert fielded.decision["exposure"] == "walks"
        assert answers[column] == (CONFOUNDER if way == "in" else NEITHER)
        assert {c: a for c, a in answers.items() if c != column} == {
            c: a for c, a in in_force.items() if c != column}
        flipped = {c: derive(a).adjusted for c, a in answers.items()}
        assert flipped == {c: derive(a).adjusted != (c == column) for c, a in in_force.items()}
    assert not [k for k in planned if "bmi" in k or "gone" in k]


def test_a_toggle_of_an_earlier_records_column_is_posted_as_composed(tmp_path):
    """The refit of a column the plan answered before its latest record posts the plan's answers
    in force with that column flipped (age, out), as composed: the revisit withdrew the latest
    record, and the engine keeps each answer under its column, so the four columns it posts are
    the plan's in force but age."""
    in_force = W.adjustment_in_force(view_at(OVER_TWO))
    sweep = toggling_sweep(tmp_path, crash=False)
    sweep.base = {**sweep.base, "seq": 6, "points": [LATEST.json()]}
    sweep.fields = W.Fields(ESTIMATE_Q, roles=OVER_TWO[2][1]["roles"], adjustment=in_force)
    session = Adjusting()
    sweep.done[W.PLAN] = W.Refit(W.PLAN, None, None, kind="scored",
                                 digest=W.plan_digest(session.view()))
    report = sweep.run(session, lambda s: "p0")
    rid = f"{LATEST.id}=answers@toggle:age:out"
    refit = {r["id"]: r for r in report["refits"]}[rid]
    decision = {"kind": "set_adjustment", "exposure": "walks",
                "answers": {**in_force, "age": NEITHER}}
    assert refit["decision"] == decision and refit["answered_with"]["decision"] == decision
    assert [c for c in session.calls if c[0] == "decide" and c[1] == decision] == [
        ("decide", decision)]


def test_a_cut_point_of_ones_own_is_the_cards_held_out_rows_and_else_unreached():
    card = arms.card_of("nhanes_glucose_predict")
    held = card["held_out"]
    split = W.Point("question", "split", 10, "r10", {"kind": "set_split", "holdout": 0.2})
    (a,), why = W.Fields(arms.Question.of_card(card), card=card).answers(split, option("cut"))
    assert why is None and a.source == "card" and a.option == f"cut@card:{held['column']}"
    assert a.sets == [f"cut_column={held['column']}", f"cut_at={held['at']}", "fold_order=random"]
    bang = arms.card_of("bang2009")
    answers, why = W.Fields(arms.Question.of_card(bang), card=bang).answers(split, option("cut"))
    assert answers == [] and "records nothing by itself" in why and "held-out" in why


class Fielding(Walked):
    """A crash sweep's session over the exclusions question: its revisit opens the question again
    with keeping every row, a range of one's own and levels of one's own; each preview and answer
    kept in ``calls`` with its fields; every refit the plan's duplicate (no Fit pressed)."""

    def __init__(self):
        super().__init__([])
        self.calls, self.reopened = [], False

    def revisit(self, record, reason):
        self.reopened = True
        return arms.Said("ok", {})

    def ask(self):
        from research.swarm.tests.test_arms import ask, opt

        if self.reopened:
            return ask("exclusions", [opt("none", {"kind": "set_exclusions", "rules": []}),
                                      opt("custom", None), opt("levels", None)])
        return {"question": None, "fit_reached": True}

    def preview(self, ids, sets, reason):
        self.calls.append(("preview", list(ids), list(sets)))
        return arms.Said("ok", {"previewed": {"kind": "set_exclusions", "rules": [{"x": 1}]}})

    def answer(self, ids, sets, reason, marker=None):
        self.reopened = False
        self.calls.append(("answer", list(ids), list(sets)))
        return arms.Said("ok", {})


def fielding_sweep(tmp_path, session, frame, roles):
    sweep = W.Sweep(tmp_path / "sweep", dataset="tiny", question=arms.Question(
        target="y", purpose="prediction"), crash=True, plan="default")
    sweep.base = {"project": "p0", "digest": "d0", "seq": 9, "scoring": {"set": False},
                  "points": [EXCLUSIONS.json()], "skipped": []}
    sweep.done[W.PLAN] = W.Refit(W.PLAN, None, None, kind="scored",
                                 digest=W.plan_digest(session.view()))
    sweep.offered = {EXCLUSIONS.id: {"point": EXCLUSIONS.id, "why": None, "options": [
        {"id": "none", "label": "none", "decision": {"kind": "set_exclusions", "rules": []},
         "needs": None}, option("custom"), option("levels")]}}
    sweep.fields = W.Fields(sweep.question, frame=frame, roles=roles)
    return sweep


def test_the_crash_sweep_previews_and_answers_a_built_value_and_names_its_source(tmp_path):
    frame, roles = built_frame(levels=False)
    session = Fielding()
    sweep = fielding_sweep(tmp_path, session, frame, roles)
    # A finished report of the crash sweep before it answered fields: each such option unreached.
    old = "It records nothing by itself: it takes a field of its own."
    for oid in ("custom", "levels"):
        sweep.done[f"{EXCLUSIONS.id}={oid}"] = W.Refit(EXCLUSIONS.id, oid, None, kind="unreached",
                                                       why=old)
    sweep.done[f"{EXCLUSIONS.id}=none"] = W.Refit(EXCLUSIONS.id, "none", {}, kind="duplicate")
    report = sweep.run(session, lambda s: "p0")
    (built,), _ = sweep.fields.answers(EXCLUSIONS, option("custom"))
    rid = f"{EXCLUSIONS.id}={built.option}"
    assert [c for c in session.calls if c[0] in ("preview", "answer")][:2] == [
        ("preview", ["custom"], built.sets), ("answer", ["custom"], built.sets)]
    refits = {r["id"]: r for r in report["refits"]}
    assert set(refits) == {f"{EXCLUSIONS.id}=none", rid, f"{EXCLUSIONS.id}=levels"}
    assert refits[rid]["answered_with"] == built.json() and refits[rid]["kind"] == "duplicate"
    assert refits[rid]["decision"] == {"kind": "set_exclusions", "rules": [{"x": 1}]}
    levels = refits[f"{EXCLUSIONS.id}=levels"]
    assert levels["kind"] == "unreached" and "no categorical predictor" in levels["why"]
    assert report["unreached"] == [{"refit": levels["id"], "why": levels["why"]}]
    assert report["complete"] is True and report["unvisited"] == []
    md = W.table_md(report | {"passed": True})
    assert "complete: True; passed: True; unreached: 1." in md
    assert f"- `{rid}` (duplicate), built: {built.words}" in md
    # The resumed run answered only the built value; another run answers nothing more.
    answered = [c for c in session.calls if c[0] == "answer"]
    assert len(answered) == 1
    again = sweep.run(session, lambda s: "p0")
    assert [c for c in session.calls if c[0] == "answer"] == answered
    assert again["refits"] == report["refits"]


def test_the_sweep_around_a_named_plan_answers_no_field_of_its_own(tmp_path):
    frame, roles = built_frame()
    session = Fielding()
    sweep = fielding_sweep(tmp_path, session, frame, roles)
    sweep.crash = False
    report = sweep.run(session, lambda s: "p0")
    assert not [c for c in session.calls if c[0] == "answer" and c[2]]
    kinds = {r["option"]: r["kind"] for r in report["refits"]}
    assert kinds["custom"] == kinds["levels"] == "unreached"


class Resolving(Fielding):
    """A :class:`Fielding` session whose preview and answer of an option are the driver's own: the
    ids and fields resolved by ``driver.chosen`` over the exclusions question as the driver
    composes it (``composed``, :func:`exclusions_composed`), and said as ``arms.Local`` says the
    driver's commands, so a field the question does not take is the driver's refusal in the
    composer's words. A decision resolved is then served by the server: ``at_preview`` and
    ``at_answer`` name a server refusal (a code) at the preview or the answer, else each is ok."""

    def __init__(self, composed, *, at_preview=None, at_answer=None):
        super().__init__()
        self.composed, self.at_preview, self.at_answer = composed, at_preview, at_answer

    def question(self, run, client, wait):
        return {"_view": {"state": {}}, "_declared": {}, "_composed": self.composed, "_ways": []}

    @staticmethod
    def served(decision, refused, posted):
        if refused is None:
            return 0, {posted: dict(decision)}
        return 1, {posted: dict(decision), "refused": {
            "by": "server", "status": 422, "code": refused, "message": "The server refuses it.",
            "exits": []}}

    def preview(self, ids, sets, reason):
        from research.swarm import driver

        self.calls.append(("preview", list(ids), list(sets)))

        def call():
            decision, _ = driver.chosen(None, None, list(ids), list(sets))
            return self.served(decision, self.at_preview, "previewed")

        return arms.Local._said(self, call)

    def answer(self, ids, sets, reason, marker=None):
        from research.swarm import driver

        self.calls.append(("answer", list(ids), list(sets)))

        def call():
            decision, _ = driver.chosen(None, None, list(ids), list(sets))
            if self.at_answer is None:
                self.reopened = False
            return self.served(decision, self.at_answer, "posted")

        return arms.Local._said(self, call)


def resolving_sweep(tmp_path, monkeypatch, columns, **served):
    """The crash sweep of :func:`fielding_sweep` on a :class:`Resolving` session whose exclusions
    question the driver composes over ``columns`` (the outcome ``y`` beside them), the table's
    numeric predictor ``x1`` the first the roles name and no categorical one; keeping every row
    already refitted and scored, so the decision's refused share has a denominator."""
    from research.swarm import driver

    frame, roles = built_frame(levels=False)
    session = Resolving(exclusions_composed(columns, "y"), **served)
    monkeypatch.setattr(driver, "the_question", session.question)
    sweep = fielding_sweep(tmp_path, session, frame, roles)
    sweep.done[f"{EXCLUSIONS.id}=none"] = W.Refit(EXCLUSIONS.id, "none", {}, kind="scored")
    (built,), _ = sweep.fields.answers(EXCLUSIONS, option("custom"))
    return session, sweep, built


def test_fields_the_driver_does_not_take_leave_the_refit_unreached_in_the_drivers_words(
        tmp_path, monkeypatch):
    """The built range is on ``x1``, which the exclusions question, as the driver composes it over
    a table whose numeric columns are others, does not offer: the driver's preview refuses it, so
    nothing is answered and the refit is unreached in the composer's own words, never a refusal
    of the engine's in the decision's refused share."""
    from research.swarm import options as opts

    session, sweep, built = resolving_sweep(tmp_path, monkeypatch, ["SEQN", "age_months"])
    with pytest.raises(opts.Unanswerable) as words:  # the driver's own composer, the truth
        opts.resolve(session.composed, built.ids, built.sets)
    report = sweep.run(session, lambda s: "p0")
    rid = f"{EXCLUSIONS.id}={built.option}"
    assert [c for c in session.calls if c[1] == built.ids] == [("preview", built.ids, built.sets)]
    refit = {r["id"]: r for r in report["refits"]}[rid]
    assert refit["kind"] == "unreached" and refit["answered"] == []
    assert refit["why"] == (f"{W.NOTHING}the driver does not take the fields the crash sweep "
                            f"answered it with: not_offered: {words.value}")
    assert refit["answered_with"] == built.json()
    (seen,) = refit["previews"]
    assert (seen["kind"], seen["by"], seen["code"], seen["message"]) == (
        "refused", "driver", "not_offered", str(words.value))
    assert {"refit": rid, "why": refit["why"]} in report["unreached"]
    assert report["defects"] == [] and report["complete"] is True
    (summary,) = report["decisions"]
    assert summary["refused_share"] == 0.0 and summary["refused_codes"] == {}
    assert summary["refused_options"] == [] and summary["classes"]["unreached"] == 2
    assert "unreached: 2." in W.table_md(report | {"passed": W.passed(report)})


@pytest.mark.parametrize("at_answer, kind", [(None, "duplicate"), ("rule_refused", "refused")],
                         ids=["answer_ok", "answer_refused"])
def test_a_server_refusal_at_a_fielded_preview_goes_on_to_the_answer_and_is_classed_by_it(
        tmp_path, monkeypatch, at_answer, kind):
    """The question takes the built range's fields; the server refuses its preview: the guard
    stays silent, the answer is posted, and the refit is classed by the answer's outcome (the
    plan's duplicate where the server records it; refused at the answer, in the decision's
    refused share, where it refuses it)."""
    session, sweep, built = resolving_sweep(tmp_path, monkeypatch, ["SEQN", "x1", "age_months"],
                                            at_preview="preview_refused", at_answer=at_answer)
    report = sweep.run(session, lambda s: "p0")
    rid = f"{EXCLUSIONS.id}={built.option}"
    assert [c for c in session.calls if c[1] == built.ids] == [
        ("preview", built.ids, built.sets), ("answer", built.ids, built.sets)]
    refit = {r["id"]: r for r in report["refits"]}[rid]
    (seen,) = [p for p in refit["previews"] if p["of"] == rid]
    assert (seen["kind"], seen["by"], seen["code"]) == ("refused", "server", "preview_refused")
    assert refit["kind"] == kind and refit["why"] is None
    assert refit["answered"][0]["answer"] == rid
    assert refit["answered"][0]["kind"] == ("ok" if at_answer is None else "refused")
    (summary,) = report["decisions"]
    if at_answer is None:
        assert summary["refused_share"] == 0.0 and summary["refused_codes"] == {}
    else:
        assert (refit["refused"]["at"], refit["refused"]["by"], refit["refused"]["code"]) == (
            "answer", "server", "rule_refused")
        assert summary["refused_share"] == 0.5 and summary["refused_codes"] == {
            "rule_refused": 1}
    assert {"refit": rid, "why": None} not in report["unreached"]
