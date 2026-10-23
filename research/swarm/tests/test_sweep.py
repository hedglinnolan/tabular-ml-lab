"""The sweep's arithmetic and its classing (SWARM_PLAN §6, §7; ``research/swarm/sweep.py``), each
from construction: two fit records built here with numpy, their 5 × 5 fold scores known, so the
refit's materiality is checked against package 1's signed change (``signed.change``) and against a
recomputation of the Nadeau–Bengio corrected interval by numpy and scipy, paired and unpaired,
material and not, and constructed so each of §7's two conditions and its half are needed; the
Estimate shift by arithmetic; each refit's class both ways, among them what the walk meets after an
option (a stage in error, typed or not, and an answer refused with no way forward) on a session
built here; the decisions a plan offers the sweep and those it does not, from a decision log built
here; what the report counts as not visited and what it says it swept; the readings graded against
a declared truth; the run times' summary by hand. The sweep on a real engine is
``test_sweep_engine.py``'s."""
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


def test_a_change_with_no_number_reads_as_no_materiality_and_says_why():
    """The same scores in every fold (an answer the fit does not read): no spread, so package
    1's change has no number, and §7 reads none."""
    before, _ = draws(0.0)
    plan = scored(W.PLAN, record(before, digest="d0", fid="f0", low=0.63, high=0.67))
    refit = scored("question:missing", record(before, digest="d0", fid="f1", low=0.63,
                                              high=0.67))
    W.materiality(refit, plan, "prediction")
    assert refit.change[0]["difference"] is None and refit.change[0]["why"]
    assert refit.change[0]["material"] is None and refit.material is None


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
    sweep = W.Sweep(tmp_path / "sweep", dataset="tiny", question=arms.Question(
        target="y", purpose="prediction"), crash=True, plan="default", only=only)
    sweep.base = {"project": "p0", "digest": "d0", "seq": 9, "scoring": {"set": False},
                  "points": [], "skipped": []}
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
