"""SURFACING_POLICY recommendation 1: materiality, its ledger, and the two-phase triage.

``M(i, s)`` is the largest movement an alternative causes in the target quantity, in units of its
own uncertainty (§2.1): predicted before the lock from outcome-blind instruments, realized after it
from refits, recorded side by side in a ledger and calibrated on the reference journeys (§2.2,
§2.6). Every number here is checked against one computed by hand in the test, never by the engine
function it checks.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions, materiality as M, plan_lock, quest, sweep
from turbotab.core.decisions import ProjectState
from turbotab.core.tests.test_confirm_sweep import Ctx, log_of, planned
from turbotab.core.tests.test_stage_registry import line, record


@pytest.fixture
def calibrated(monkeypatch):
    """``rows_smd`` calibrated at its starting conventions (0.05, 0.2) and the other proxies not:
    the grading is tested here on fixed thresholds; the committed file's own thresholds are tested
    with the cases they were calibrated on (``test_materiality_dietary_proof``)."""
    data = json.loads(json.dumps(M.calibration()))
    rows = data["instruments"]["rows_smd"]
    rows.update(band_1=0.05, band_2=0.2, calibrated=True)
    monkeypatch.setattr(M, "calibration", lambda: data)
    return data


# ── the bands, and the cap on an uncalibrated instrument ─────────────────────


def test_an_uncalibrated_instrument_never_says_it_changes_nothing():
    """§2.6: where the calibration set has no case for an instrument its band is capped at 1, so
    it may say "could bias" but never "doesn't change your numbers here"."""
    assert not M.thresholds("exposure_shift")[2]
    quiet = M.movement("exposure_shift", 0.0, "nothing moves")
    assert quiet.band == 1 and not quiet.calibrated
    assert M.disposition(quiet) == "could_bias"
    assert M.would_change(quiet) == (True, "nothing moves")


def test_an_uncalibrated_proxy_is_floored_at_could_bias_and_still_says_act_on_it():
    """The cap of §2.6 is a floor: an uncalibrated proxy never says band 0, and a movement past its
    band-2 convention is still "act on it" (§2.3: 1 − λ ≈ 0.6 is act on it; §2.5: drift beyond
    50% of features over chance is band 2)."""
    assert M.attenuation(0.38, "sugar").band == 2  # 1 − λ = 0.62
    assert M.disposition(M.attenuation(0.38, "sugar")) == "act_on_it"
    assert M.exposure_shift(pd.DataFrame({"x": [0.0, 1, 0, 1]}),
                            pd.DataFrame({"x": [17.0, 18, 17, 18]}), "x").band == 2  # W1/SD = 34
    assert M.excess_over_chance(0.64, 0.007, "features").band == 2  # 63% beyond chance
    assert M.excess_over_chance(0.03, 0.007, "features").band == 1  # below band 1: floored
    assert M.attenuation(0.95, "sugar").band == 1  # 1 − λ = 0.05: floored, never band 0


def test_a_calibrated_instrument_grades_by_its_thresholds(calibrated):
    b1, b2, cal = M.thresholds("rows_smd")
    assert cal and (b1, b2) == (0.05, 0.2)
    assert [M.band_of("rows_smd", v) for v in (0.049, 0.05, 0.199, 0.2)] == [0, 1, 1, 2]
    assert M.disposition(M.movement("rows_smd", 0.01, "w")) == "no_change"
    # Unknown is "could bias"; a changed question or a sign change is "act on it".
    assert M.band_of("rows_smd", None) == 1
    assert M.band_of("rows_smd", 0.0, not_measurable=True) == 1
    assert M.band_of("rows_smd", 0.0, changes_question=True) == 2
    assert M.band_of("sensitivity", 0.01, crosses=True) == 2
    # The realized instruments read τ₀ and τ₁ and are the reference, never capped.
    assert M.thresholds("sensitivity") == (0.1, 0.5, True)
    assert M.movement("sensitivity", 0.05, "w").band == 0


# ── the predicted instruments, by hand ───────────────────────────────────────


def test_rows_times_smd_is_the_share_leaving_times_the_largest_standardized_difference():
    frame = pd.DataFrame({"age": [30.0, 40, 50, 60, 70, 80], "sex": ["f", "f", "m", "m", "m", "f"],
                          "glucose": [1.0, 2, 3, 4, 50, 60]})
    leaving = np.array([False, False, False, False, True, True])
    # By hand: age leavers 70, 80 (mean 75, var 50); stayers 30..60 (mean 45, var 166.67).
    d_age = 30 / math.sqrt((50 + 500 / 3) / 2)
    # sex: P(f) leavers 0.5, stayers 0.5 → 0 for both levels.
    m = M.rows_smd(frame, leaving, ["age", "sex"])
    assert m.value == pytest.approx((2 / 6) * d_age, rel=1e-12)
    assert m.instrument == "rows_smd" and m.regime == "predicted"
    assert "`2` of `6` rows (33.3%)" in m.words and "`age`" in m.words
    # The outcome is never among the columns compared: it is not what the model adjusts for.
    assert "glucose" not in m.words


def test_a_category_is_compared_level_by_level():
    s = pd.Series(["a", "a", "b", "b", "b", "a", "a", "a"])
    leaving = np.array([True, True, True, False, False, False, False, False])
    p1, p0 = 2 / 3, 3 / 5  # P(a) among leavers, stayers
    want = abs(p1 - p0) / math.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    assert M.smd(s, leaving) == pytest.approx(want, rel=1e-12)


def test_reliability_is_the_one_way_random_effects_ratio():
    """λ = σ²b / (σ²b + σ²w / k): three people, two recalls each, by hand."""
    persons = ["p", "p", "q", "q", "r", "r"]
    x = [1.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    # Means 2, 5, 10; grand 17/3. SSW = 2 + 2 + 8 = 12 on 3 df → MSW 4.
    # SSB = 2 · ((2 − 17/3)² + (5 − 17/3)² + (10 − 17/3)²) = 2 · 98/3 → MSB = 98/3; n0 = 2.
    var_b = (98 / 3 - 4) / 2
    lam = var_b / (var_b + 4 / 2)
    got = M.reliability(x, persons)
    assert got is not None
    assert got[0] == pytest.approx(lam, rel=1e-12) and got[1] == pytest.approx(var_b)
    assert got[2] == pytest.approx(4.0) and got[3] == 2.0
    a = M.attenuation(got[0], "sugar")
    assert a.value == pytest.approx(1 - lam) and a.band == 1  # uncalibrated: capped at 1
    assert M.reliability([1.0, 2.0], ["p", "q"]) is None  # no second measure


def test_one_recall_day_is_not_measurable_here_and_could_bias():
    m = M.attenuation(None, "sugar", days=1)
    assert m.not_measurable and m.value is None and m.band == 1
    assert "not measurable here" in m.words and "one recall day" in m.words


def test_the_outcome_beside_another_column_waits_for_the_lock_under_estimate():
    """Nolan's ruling: under Estimate the outcome beside any other column is hidden until the
    lock; an unanswered purpose is the strictest case. Under Predict it is read on training rows."""
    estimate = ProjectState(purpose="inference")
    assert M.design_imbalance(estimate, 0.4, "run_order") is None
    assert M.design_imbalance(ProjectState(), 0.4, "run_order") is None
    locked = M.design_imbalance(estimate.model_copy(update={"plan_locked": True}), -0.4, "run")
    assert locked is not None and locked.value == pytest.approx(0.4)
    assert M.design_imbalance(ProjectState(purpose="prediction"), 0.05, "run").value == 0.05


def test_the_exposure_shift_and_the_correlation_change_by_hand():
    before = pd.DataFrame({"x": [1.0, 2, 3, 4], "z": [1.0, 2, 3, 5]})
    after = pd.DataFrame({"x": [2.0, 3, 4, 5], "z": [1.0, 2, 3, 5]})
    shift = M.exposure_shift(before, after, "x")
    assert shift.value == pytest.approx(1 / np.std([1, 2, 3, 4]))  # W1 = 1, population SD
    flipped = pd.DataFrame({"x": [4.0, 3, 2, 1], "z": [1.0, 2, 3, 5]})
    r = np.corrcoef([1, 2, 3, 4], [1, 2, 3, 5])[0, 1]
    assert M.correlation_change(before, flipped, "x", ["z"]).value == pytest.approx(2 * r)
    assert M.excess_over_chance(0.64, 0.007, "features").value == pytest.approx(0.633)


# ── the realized instruments, by hand ────────────────────────────────────────


def fit(label, n, est, lo, hi):
    return {"label": label, "n_rows": n,
            "coefficients": [{"feature": "sugar", "estimate": est, "ci_low": lo, "ci_high": hi},
                             {"feature": "age", "estimate": 9.0, "ci_low": 8.0, "ci_high": 10.0}]}


def sensitivity(primary, every):
    return {"analyses": [{"label": "Primary", "primary": True, "rules": ["goldberg"]},
                         {"label": "Every row", "primary": False, "rules": []}],
            "families": [{"family": "linear", "fits": [primary, every]}]}


def test_the_screen_is_verified_by_the_every_row_refit():
    art = sensitivity(fit("Primary", 90, -0.02, -0.03, -0.01), fit("Every row", 100, -0.023, -0.035, -0.011))
    m = M.realized_from_sensitivity(art, "sugar")
    assert m.value == pytest.approx(0.003 / 0.01) and m.band == 1 and m.regime == "realized"
    assert "every row" in m.words
    # A sign change, or an interval that stops excluding no effect, is "act on it".
    flipped = sensitivity(fit("Primary", 90, -0.02, -0.03, -0.01), fit("Every row", 100, 0.001, -0.01, 0.012))
    assert M.realized_from_sensitivity(flipped, "sugar").band == 2
    widened = sensitivity(fit("Primary", 90, -0.02, -0.03, -0.01), fit("Every row", 100, -0.019, -0.04, 0.002))
    assert M.realized_from_sensitivity(widened, "sugar").band == 2
    # Only the exposure's columns are Q: age's move is not counted.
    assert M.realized_from_sensitivity(art, "sugar").value < 1


def test_the_calibration_the_secondary_the_benchmarks_and_the_folds_by_hand():
    cal = {"applies": True, "exposures": [{"source": "sugar", "naive": 0.10, "naive_ci_low": 0.06,
                                           "naive_ci_high": 0.14, "estimate": 0.25}]}
    assert M.realized_from_calibration(cal, "sugar").value == pytest.approx(0.15 / 0.04)
    sec = {"families": [{"fits": [
        {"label": "Primary", "coefficients": [{"feature": "sugar", "estimate": 1.0, "ci_low": 0.0, "ci_high": 2.0}]},
        {"label": "Further adjusted for `bmi`", "coefficients": [{"feature": "sugar", "estimate": 0.95}]}]}]}
    assert M.realized_from_secondary(sec).value == pytest.approx(0.05) and \
        M.realized_from_secondary(sec).band == 0
    from scipy import stats

    rob = {"estimate": 2.0, "se": 0.5, "dof": 100.0,
           "benchmarks": [{"covariate": "age", "estimate": 1.6}, {"covariate": "gender", "estimate": 1.9}]}
    half = stats.t.ppf(0.975, 100) * 0.5
    b = M.realized_from_benchmarks(rob)
    assert b.value == pytest.approx(0.4 / half) and "`age`" in b.words
    e = M.realized_from_e_value({"limit": 1.0, "interval_includes_null": True})
    assert e.band == 2
    assert M.realized_from_e_value({"limit": 2.4, "interval_includes_null": False}).band == 1
    folds = M.paired_folds([0.70, 0.72, 0.71, 0.69, 0.73], [0.71, 0.74, 0.71, 0.70, 0.75], "Drift correction")
    d = np.array([0.01, 0.02, 0.0, 0.01, 0.02])
    assert folds.value == pytest.approx(d.mean() / d.std(ddof=1))


# ── the tier (§1.3) ──────────────────────────────────────────────────────────


def test_the_tier_follows_blocking_meaning_the_question_and_the_alternative():
    quiet = M.movement("sensitivity", 0.01, "w")  # realized, band 0
    act = M.movement("rows_smd", 0.5, "w")
    assert M.tier(fires=False) == "hidden"
    assert M.tier(blocker=True, has_default=True, m_alt=quiet) == quest.DECIDE
    assert M.tier(decides_by="meaning", m=act) == quest.DECIDE
    assert M.tier(decides_by="meaning", m=None) == quest.DECIDE  # unmeasured: strictest
    assert M.tier(changes_the_question=True, has_default=True) == quest.DECIDE
    assert M.tier(has_default=True, m_alt=act) == quest.CONFIRM
    assert M.tier(has_default=True, m_alt=None) == quest.CONFIRM  # unmeasured: strictest
    assert M.tier(has_default=True, m_alt=quiet) == quest.RECORD
    assert M.tier(has_default=True, m_alt=M.movement("exposure_shift", 0.0, "w")) == quest.CONFIRM
    assert M.tier(settled_by_values=True, m_alt=quiet) == quest.RECORD


# ── the Confirm sweep reads M where it is measured ───────────────────────────


def test_the_confirm_sweep_reads_the_measured_movement_in_place_of_its_hand_test():
    """E1b's would-change test says another modifier would change the effect; where M is measured
    and calibrated below τ_confirm the line is For the record with what was measured, and where the
    instrument is uncalibrated it stays a Confirm whatever it measured."""
    state = planned()
    before = log_of(state)
    assert line(before, "modification").label == quest.CONFIRM
    quiet = M.movement("sensitivity", 0.02, "Checked: another modifier moves nothing here.")
    after = log_of(state, artifacts={"materiality": {"modification": quiet}})
    got = line(after, "modification")
    assert got.label == quest.RECORD and got.changes_nothing == quiet.words
    unsure = M.movement("exposure_shift", 0.0, "Another modifier shifts nothing measured.")
    held = log_of(state, artifacts={"materiality": {"modification": unsure.model_dump()}})
    assert line(held, "modification").label == quest.CONFIRM
    assert line(held, "modification").would_change == unsure.words


# ── the triage of measured noticings ─────────────────────────────────────────


def noticing(thread, band_value, *, family="S5", answered=False, done=None, instrument="rows_smd",
             not_measurable=False, changes=False):
    if changes:
        m = M.changes_question("It changes what the estimate is.", "r = 0.67")
    elif not_measurable:
        m = M.attenuation(None, "sugar")
    else:
        m = M.movement(instrument, band_value, f"{thread} moves {band_value}.")
    return M.Noticing(thread=thread, family=family, stage="models", subject=["sugar"],
                      summary=f"{thread} noticed", decides_by="meaning", alternative="the other",
                      predicted=m, answered=answered, done=done,
                      question="energy_adjustment" if changes else None)


def test_each_measured_noticing_is_recommended_from_its_band(calibrated):
    state = planned()
    log = log_of(state)
    noticed = [noticing("energy", None, family="K5", changes=True, answered=True),
               noticing("screen", 0.1, done="The every-row analysis is declared."),
               noticing("days", None, family="K3", not_measurable=True),
               noticing("tiny", 0.01),
               noticing("open-energy", None, family="K5", changes=True)]
    t = sweep.triage(state, log, {"findings": []}, noticed=noticed)
    got = {i.id: (i.recommended, i.limitation, i.band) for i in t.items}
    assert "energy" not in got  # act on it, and its decision is answered: it drops out
    assert got == {"open-energy": ("act_on_it", False, 2),
                   "screen": ("could_bias", False, 1),  # something was done: no limitation
                   "days": ("could_bias", True, 1),  # nothing can be done: a limitation
                   "tiny": ("no_change", False, 0)}
    assert next(i for i in t.items if i.id == "open-energy").question == "energy_adjustment"
    assert [i.id for i in t.items][0] == "open-energy"  # act on it ranks first
    days = next(i for i in t.items if i.id == "days")
    assert days.label == "Could bias the estimate (not measurable here)"
    assert days.calibrated is False and "not measurable" in days.measure
    assert t.confirmable and not t.answered
    d = decisions.validate({"kind": "confirm_sweep", "stage": "models", "sweep": "noticings"},
                           Ctx(log, t))
    assert {l.key: l.value for l in d.lines} == {"open-energy": "act_on_it", "screen": "could_bias",
                                                 "days": "could_bias", "tiny": "no_change"}
    assert all(l.id.startswith("noticing:") and l.basis for l in d.lines)


def test_the_triage_shows_at_most_six_rows_grouped_by_family():
    eight = [noticing(f"n{i}", 0.1, family=("S5", "K3", "K5")[i % 3]) for i in range(8)]
    t = sweep.triage(planned(), log_of(planned()), {"findings": []}, noticed=eight)
    assert len(t.items) == 8 and len(t.rows) == 3
    assert {r.family: r.count for r in t.rows} == {"S5": 3, "K3": 3, "K5": 2}
    assert sorted(i for r in t.rows for i in r.items) == sorted(i.id for i in t.items)
    assert next(r for r in t.rows if r.family == "S5").name == "Who is in"
    few = sweep.triage(planned(), log_of(planned()), {"findings": []}, noticed=eight[:6])
    assert len(few.rows) == 6 and all(r.count == 1 for r in few.rows)
    many = M.triage_rows([(f"x{i}", f"F{i}") for i in range(9)])
    assert len(many) == 6 and many[-1].count == 4 and many[-1].name == "4 more families"


# ── verification and the ledger ──────────────────────────────────────────────


def screen_noticing(value):
    n = noticing("diet-implausible-reporters", value)
    return n.model_copy(update={"verified_by": "sensitivity"})


def test_no_realized_movement_is_read_before_the_lock():
    art = {"sensitivity": sensitivity(fit("Primary", 90, -0.02, -0.03, -0.01),
                                      fit("Every row", 100, -0.03, -0.04, -0.02))}
    for state in (planned(), planned(purpose=None)):
        book = M.ledger(state, [screen_noticing(0.01)], art)
        assert not book.locked
        row = book.rows[0]
        assert row.realized is None and row.verdict == "pending" and row.label is None


def test_a_realized_band_above_the_predicted_one_relabels_the_exhibit_openly(calibrated):
    """Predicted below noise (0.01, band 0), realized at a whole half-width (band 2): upgraded,
    labeled on the sensitivity exhibit, and a limitation drafted since nothing was done."""
    art = {"sensitivity": sensitivity(fit("Primary", 90, -0.02, -0.03, -0.01),
                                      fit("Every row", 100, -0.03, -0.04, -0.02))}
    locked = planned(plan_locked=True)
    row = M.ledger(locked, [screen_noticing(0.01)], art).rows[0]
    assert row.recommended == "no_change" and row.verdict == "upgraded"
    assert row.realized.value == pytest.approx(1.0) and row.realized.band == 2
    assert row.exhibit == "sensitivity" and row.label.startswith("Moved more than predicted")
    assert row.limitation and row.sentence.startswith("Limitation:")
    # Realized below the prediction: downgraded, "checked; it did not".
    calm = {"sensitivity": sensitivity(fit("Primary", 90, -0.02, -0.03, -0.01),
                                       fit("Every row", 100, -0.0201, -0.03, -0.01))}
    down = M.ledger(locked, [screen_noticing(0.1)], calm).rows[0]
    assert down.verdict == "downgraded" and down.label is None
    # No instrument for it: not verifiable, the disposition stands.
    gone = M.ledger(locked, [screen_noticing(0.1)], {}).rows[0]
    assert gone.verdict == "not_verifiable"


def test_limitation_sentences_stop_at_the_budget():
    many = [noticing(f"n{i}", None, not_measurable=True) for i in range(5)]
    book = M.ledger(planned(), many)
    assert book.budget == 3 and book.limitations == 3
    assert [r.limitation for r in book.rows] == [True, True, True, False, False]
    assert not book.rows[4].sentence.startswith("Limitation:")


# ── calibration (§2.6) ───────────────────────────────────────────────────────


def case(m_pre, m_post, band_post, thread="t"):
    return M.Case(journey="j", pipeline="p", thread=thread, alternative="a", instrument="rows_smd",
                  m_pre=m_pre, instrument_post="sensitivity", m_post=m_post, band_post=band_post)


def test_calibration_lowers_a_threshold_that_would_reassure_falsely():
    start = M.calibration()
    cases = [case(0.03, 0.3, 1), case(0.3, 0.02, 0), case(0.001, 0.01, 0), case(0.06, 0.2, 1),
             case(0.25, 0.7, 2)]
    out = M.calibrate(cases, start)
    rows = out["instruments"]["rows_smd"]
    assert rows["band_1"] == 0.03 and rows["calibrated"] and rows["cases"] == 5
    assert rows["confusion"]["matrix"] == [[1, 0, 0], [0, 2, 0], [1, 0, 1]]
    assert rows["confusion"]["false_reassurance"] == 0 and rows["confusion"]["cry_wolf"] == 1
    assert not out["instruments"]["attenuation"]["calibrated"]  # no case: uncalibrated
    assert out["instruments"]["sensitivity"]["calibrated"]


def test_a_proxy_says_below_noise_only_on_enough_cases_and_a_band_zero_that_held():
    """One case is not a calibration (§2.6's guard against false reassurance): a proxy is
    calibrated, and so may say "doesn't change your numbers here", only with at least the file's
    minimum number of cases and at least one case it put below noise that stayed below noise
    after the lock. Short of that it is floored at "could bias"."""
    start = M.calibration()
    assert start["min_cases"] >= 5
    one = M.calibrate([case(0.185, 0.124, 1)], start)["instruments"]["rows_smd"]
    assert not one["calibrated"] and one["cases"] == 1
    # Five cases, every one realized above noise: no evidence band 0 is ever right.
    loud = [case(v, 0.4, 1, thread=str(v)) for v in (0.004, 0.008, 0.012, 0.019, 0.03)]
    out = M.calibrate(loud, start)["instruments"]["rows_smd"]
    assert out["band_1"] == 0.004 and not out["calibrated"]
    assert out["confusion"]["false_reassurance"] == 0
    # With one that held below noise, it is calibrated below the lowest miss.
    held = M.calibrate([*loud, case(0.001, 0.01, 0)], start)["instruments"]["rows_smd"]
    assert held["calibrated"] and held["band_1"] == 0.004


def test_the_committed_calibration_reassures_falsely_on_none_of_its_cases():
    """The file is what ``calibrate`` makes of its own cases, each says where it came from, and no
    case it calls below noise moved the estimate by τ₀ or more after the lock."""
    data = M.calibration()
    cases = [M.Case.model_validate(c) for c in data["cases"]]
    assert M.calibrate(cases, data) == data
    assert all(c.pipeline and c.alternative for c in cases)
    for name, entry in data["instruments"].items():
        if entry["regime"] == "predicted" and entry.get("cases"):
            assert entry["confusion"]["false_reassurance"] == 0, name
    for c in cases:
        b1, b2, cal = M.thresholds(c.instrument)
        if cal and c.m_pre < b1:
            assert c.band_post == 0, c


# ── the lock covers the dispositions ─────────────────────────────────────────


def test_the_locks_digest_covers_the_triage_dispositions():
    state = planned()
    log = log_of(state)
    t = sweep.triage(state, log, {"findings": []}, noticed=[noticing("screen", 0.1)])
    digests = []
    for value in ("could_bias", "no_change"):
        d = decisions.validate({"kind": "confirm_sweep", "stage": "models", "sweep": "noticings",
                                "lines": [{"id": "noticing:screen", "key": "screen", "value": value}]},
                               Ctx(log, t))
        held = planned(sweeps=decisions.fold([record(1, d.model_dump(mode="json"))]).sweeps)
        plan = plan_lock.plan_of(held)
        assert plan[plan_lock.TRIAGE][0]["value"] == value
        digests.append(plan_lock.digest(plan))
    assert digests[0] != digests[1]
    assert plan_lock.TRIAGE not in plan_lock.plan_of(planned())


# ── the prediction and the verification measure the same alternative ────────


AGE_RULE = decisions.ExclusionRule(column="age", low=20, high=80, reason="adults")
KCAL_RULE = decisions.ExclusionRule(column="kcal", low=500, high=5000, reason="implausible intake")


def screened(**update):
    from turbotab.core.stages.rows import rule_label

    state = planned(roles={**ROLES_KCAL}, exclusions=[AGE_RULE, KCAL_RULE], **update)
    art = {"analyses": [{"label": "Primary", "primary": True,
                         "rules": [rule_label(AGE_RULE), rule_label(KCAL_RULE)]},
                        {"label": "Without the screen", "primary": False,
                         "rules": [rule_label(AGE_RULE)]},
                        {"label": "Every row", "primary": False, "rules": []}],
           "families": [{"family": "linear", "fits": [
               fit("Primary", 90, -0.02, -0.03, -0.01),
               fit("Without the screen", 95, -0.0205, -0.031, -0.01),  # 0.05: the screen
               fit("Every row", 100, -0.03, -0.04, -0.02)]}]}  # 1.0: the screen and age
    return state, {"sensitivity": art}


ROLES_KCAL = {"SEQN": "identifier", "sugar": "exposure", "age": "covariate", "kcal": "energy"}


def test_the_screen_is_verified_on_the_rows_without_the_screen_alone():
    """The prediction is the screen taken out of the exclusions; the verification is the analysis
    on the same rows (the other rules kept), never the every-row analysis that also drops them."""
    state, art = screened(plan_locked=True)
    row = M.ledger(state, [screen_noticing(0.1)], art).rows[0]
    assert row.realized.value == pytest.approx(0.0005 / 0.01)
    assert "without the screen" in row.realized.words
    # With no analysis on those rows the screen is not verifiable, not verified on other rows.
    only_every = {"sensitivity": {**art["sensitivity"],
                                  "analyses": [a for a in art["sensitivity"]["analyses"]
                                               if a["label"] != "Without the screen"],
                                  "families": [{"family": "linear", "fits": [
                                      art["sensitivity"]["families"][0]["fits"][0],
                                      art["sensitivity"]["families"][0]["fits"][2]]}]}}
    assert M.ledger(state, [screen_noticing(0.1)], only_every).rows[0].verdict == "not_verifiable"
    # When the screen is the only rule, the every-row analysis is the rows without it.
    alone = planned(roles={**ROLES_KCAL}, exclusions=[KCAL_RULE], plan_locked=True)
    two = sensitivity(fit("Primary", 90, -0.02, -0.03, -0.01),
                      fit("Every row", 100, -0.023, -0.035, -0.011))
    assert M.ledger(alone, [screen_noticing(0.1)], {"sensitivity": two}).rows[0].realized.value \
        == pytest.approx(0.3)


def test_the_screen_counts_as_done_only_with_an_analysis_on_the_rows_without_it():
    frame = pd.DataFrame({"SEQN": range(6), "sugar": [1.0, 2, 3, 4, 5, 6],
                          "age": [30.0, 40, 50, 60, 70, 80], "kcal": [400.0, 900, 1500, 2000, 2500, 6000]})
    kept, without = [1, 2, 3, 4], [0, 1, 2, 3, 4, 5]
    every = decisions.SensitivityAnalysis(label="Every row", rules=[])
    mine = decisions.SensitivityAnalysis(label="Without the screen", rules=[AGE_RULE])
    state = planned(roles={**ROLES_KCAL}, exclusions=[AGE_RULE, KCAL_RULE], sensitivity=[every])
    assert M.reporters_noticing(state, frame, kept, without).done is None
    state = state.model_copy(update={"sensitivity": [mine]})
    assert "without the screen" in M.reporters_noticing(state, frame, kept, without).done


def test_the_recall_days_are_read_from_the_exposures_own_units():
    frame = pd.DataFrame({"SEQN": range(4), "sugar": [1.0, 2, 3, 4], "kcal": [1.0, 2, 3, 4]})
    units = {"sugar": decisions.ColumnUnitSpec(unit="g", days=1)}
    state = planned(roles={**ROLES_KCAL}, column_units=units,
                    estimand=decisions.EstimandSpec(exposure="sugar", measure="mean_difference"))
    n = M.variance_noticing(state, frame)
    assert "one recall day" in n.predicted.words
    both = {"sugar": decisions.ColumnUnitSpec(unit="g", days=2),
            "kcal": decisions.ColumnUnitSpec(unit="kcal", days=1)}
    n = M.variance_noticing(state.model_copy(update={"column_units": both}), frame)
    assert "2 recall days averaged" in n.predicted.words


def test_a_finding_owes_a_limitation_only_when_nothing_was_done():
    """Nolan's rule (2026-10-09) holds for findings as for measured noticings: a declared
    sensitivity analysis on the rows the finding concerns counts as done."""
    from turbotab.core.tests.test_confirm_sweep import FINDINGS

    findings = {"findings": [f for f in FINDINGS["findings"] if f["severity"] != "critical"]}
    state = planned()
    t = sweep.triage(state, log_of(state, findings=findings), findings)
    age = next(i for i in t.items if i.id == "binary_text__age")
    assert age.recommended == "could_bias" and age.limitation and age.done is None
    declared = planned(sensitivity=[decisions.SensitivityAnalysis(label="Adults", rules=[AGE_RULE])])
    t = sweep.triage(declared, log_of(declared, findings=findings), findings)
    age = next(i for i in t.items if i.id == "binary_text__age")
    assert age.recommended == "could_bias" and not age.limitation and "Adults" in age.done
    assert next(i for i in t.items if i.id == "note__SEQN").limitation is False


def test_a_lock_recorded_before_the_triage_entered_the_plan_is_not_changed_by_it():
    """A lock whose plan holds no triage (recorded before the dispositions were part of it) fixed
    the answers alone: a confirmed triage in the state is no change to that plan, so a decision
    made while nothing was shown does not withdraw it. A change to an answer still does, and a
    lock that holds the triage is changed by a changed disposition."""
    from types import SimpleNamespace

    state = planned()
    log = log_of(state)
    t = sweep.triage(state, log, {"findings": []}, noticed=[noticing("screen", 0.1)])

    def held(value):
        d = decisions.validate({"kind": "confirm_sweep", "stage": "models", "sweep": "noticings",
                                "lines": [{"id": "noticing:screen", "key": "screen",
                                           "value": value}]}, Ctx(log, t))
        return planned(sweeps=decisions.fold([record(1, d.model_dump(mode="json"))]).sweeps)

    def lock_of(plan):
        return SimpleNamespace(plan=plan, digest=plan_lock.digest(plan))

    triaged = held("could_bias")
    old = lock_of({k: v for k, v in plan_lock.plan_of(triaged).items() if k != plan_lock.TRIAGE})
    assert not plan_lock.plan_changed(old, triaged)
    assert plan_lock.plan_changed(old, triaged.model_copy(update={"exclusions": [AGE_RULE]}))
    new = lock_of(plan_lock.plan_of(triaged))
    assert not plan_lock.plan_changed(new, triaged)
    assert plan_lock.plan_changed(new, held("no_change"))


def test_noisy_replicates_say_act_on_it_and_drop_out_once_the_correction_is_answered():
    """With replicate recalls λ is measured; 1 − λ past 0.5 is "act on it" even on an
    uncalibrated proxy (§2.3). The measurement-error answer is its own decision: once answered,
    the act-on-it noticing drops out of the triage, as the answered energy question does."""
    frame = pd.DataFrame({"SEQN": ["p", "p", "q", "q", "r", "r"],
                          "sugar": [1.0, 9, 2, 8, 3, 7], "kcal": [1.0, 2, 3, 4, 5, 6]})
    state = planned(roles={**ROLES_KCAL}, repeat_kind=decisions.RepeatSpec(repeat_kind="repeats"),
                    grain=decisions.GrainSpec(grain="repeated", id_column="SEQN"))
    n = M.variance_noticing(state, frame)
    assert n.predicted.value > 0.5 and n.predicted.band == 2 and not n.answered
    assert M.in_triage(n) and M.recommend(n)[0] == "act_on_it"
    declared = state.model_copy(update={"measurement_error": decisions.MeasurementErrorSpec(
        method="regression_calibration")})
    n = M.variance_noticing(declared, frame)
    assert n.answered and not M.in_triage(n) and "Regression calibration" in n.done


def test_a_lock_recorded_while_looks_were_part_of_the_plan_is_not_changed_by_them():
    """A lock recorded when the outcome's recorded looks were still part of the plan (before
    ``plan_lock.LOOKS``) holds them; the plan read now leaves them out, so the two are compared
    on the answers alone, and a decision made while nothing was shown does not withdraw it. A
    change to an answer still does."""
    from types import SimpleNamespace

    state = planned()
    looks = {"distribution:y": {"view": "distribution", "column": "y"}}
    old = dict(plan_lock.plan_of(state), outcome_views=looks)
    lock = SimpleNamespace(plan=old, digest=plan_lock.digest(old))
    assert not plan_lock.plan_changed(lock, state)
    assert plan_lock.plan_changed(lock, state.model_copy(update={"exclusions": [AGE_RULE]}))
