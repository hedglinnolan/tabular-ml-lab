"""SURFACING_POLICY §9, "The smallest end-to-end proof": the dietary NHANES Models stage with three
noticings through notice → triage → lock → verification → ledger.

The state is the ``dietary-inference`` capture's answers up to Models (``materiality_cases``:
sugar against fasting glucose at fixed energy, the Goldberg screen at PAL 1.55 on one recall day,
the every-row analysis declared beside it), run through the real stage graph on the tracked NHANES
fixture. Every movement is checked against one computed here by hand: the screen's count against
the capture's own record (5,212 of 21,849) and EFSA's procedure written out, the standardized
differences by the pooled-SD formula, the correlation by NumPy, and the realized movement by NumPy
least squares with HC3 written out by definition (``references.hc3_by_definition``).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core import decisions, materiality as M, plan_lock, quest, sweep
from turbotab.core.methods.misreporting import screen
from turbotab.core.tests import materiality_cases as C
from turbotab.core.tests.acceptance.references import hc3_by_definition
from turbotab.core.tests.acceptance.test_wp12c_goldberg import efsa_bmr_kcal
from turbotab.core.tests.test_stage_registry import record, steps_until

CAPTURED_LEAVERS = 5212  # the capture's set_exclusions sentence: "`5,212` rows … were excluded"
N_ROWS = 21849
ADJUSTED = ["age", "gender", "cycle_begin_year", "protein", "carb", "fat_total", "fat_sat",
            "fat_mon", "fat_poly", "kcal"]  # the capture's confounders, and energy (standard model)
ENERGY = "diet-energy-carries-the-nutrient"
SCREEN = "diet-implausible-reporters"
DAYS = "diet-day-to-day-variance"
# Q-d: kcal beside its parts, and the fats beside their total, make the adjustment terms nearly
# collinear; sugar is not in either dependency, so it is band 0 by theorem.
COLLINEAR = "shared-collinear-predictors"


class Ctx:
    """What the server hands a validator: the state, the quest log and the triage."""

    def __init__(self, state, log=None, triage=None):
        self.state = state
        self.quest = lambda: log
        self.triage = lambda: triage


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(C.nhanes())


@pytest.fixture(scope="module")
def proof():
    p = C.Proof()
    yield p
    p.close()


@pytest.fixture(scope="module")
def noticed(proof):
    return {n.thread: n for n in proof.noticings(C.proof_state())}


def hand_smd(values: pd.Series, leaving: np.ndarray) -> float:
    if values.dtype == object:
        out = 0.0
        for level in values.unique():
            p1, p0 = (values[leaving] == level).mean(), (values[~leaving] == level).mean()
            out = max(out, abs(p1 - p0) / math.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2))
        return out
    a, b = values[leaving].astype(float), values[~leaving].astype(float)
    return abs(a.mean() - b.mean()) / math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def ols_sugar(rows: pd.DataFrame) -> tuple[float, float, float]:
    """glucose ~ sugar + the other nutrients + kcal + age + [gender = male] + cycle, by NumPy, with
    HC3 by definition and a t(n − p) interval: sugar's (estimate, low, high)."""
    X = np.column_stack([np.ones(len(rows)), rows["sugar"], rows["protein"], rows["carb"],
                         rows["fat_total"], rows["fat_sat"], rows["fat_mon"], rows["fat_poly"],
                         rows["kcal"], rows["age"], (rows["gender"] == "male").astype(float),
                         rows["cycle_begin_year"]]).astype(float)
    y = rows["glucose"].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    se = math.sqrt(hc3_by_definition(X, y - X @ beta)[1, 1])
    t = stats.t.ppf(0.975, len(y) - X.shape[1])
    return beta[1], beta[1] - t * se, beta[1] + t * se


# ── 1 · notice (before the lock, outcome-blind) ──────────────────────────────


def test_the_three_noticings_fire_on_the_models_stage(noticed):
    assert set(noticed) == {ENERGY, SCREEN, DAYS, COLLINEAR}
    assert {t: n.family for t, n in noticed.items()} == {ENERGY: "K5", SCREEN: "S5", DAYS: "K3",
                                                         COLLINEAR: "K5"}
    assert {t: n.stage for t, n in noticed.items()} == {ENERGY: "models", SCREEN: "whos_in",
                                                         DAYS: "models", COLLINEAR: "models"}


def test_the_adjustment_terms_are_nearly_collinear_and_sugar_is_outside(noticed, frame):
    """Belsley's proportions by hand on every row of the fixture: [1, sugar, the adjustment set]
    (gender as its male indicator), each column scaled to unit length, its SVD."""
    rows = frame[["sugar", *ADJUSTED]].dropna()
    X = np.column_stack([np.ones(len(rows)), rows["sugar"],
                         *[(rows[c] == "male").astype(float) if c == "gender" else rows[c]
                           for c in ADJUSTED]]).astype(float)
    names = ["(intercept)", "sugar", *ADJUSTED]
    X = X / np.sqrt((X ** 2).sum(axis=0))
    _u, s, vt = np.linalg.svd(X, full_matrices=False)
    phi = vt.T ** 2 / s ** 2
    pi = phi / phi.sum(axis=1, keepdims=True)
    near = [k for k in range(len(s)) if s[0] / s[k] >= 30
            and sum(pi[j, k] >= 0.5 for j in range(1, len(names))) >= 2]
    inside = {names[j] for k in near for j in range(1, len(names)) if pi[j, k] >= 0.5}
    assert {"kcal", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"} <= inside
    n = noticed[COLLINEAR]
    assert n.measure == pytest.approx(sum(pi[1, k] for k in near), abs=1e-9) and n.measure < 0.5
    assert set(n.subject) == {"sugar", *inside}
    assert n.predicted.band == 0 and n.predicted.instrument == "invariance" and not n.blocker


def test_energy_carries_the_nutrient_changes_the_question(noticed, frame):
    n = noticed[ENERGY]
    r = np.corrcoef(frame["sugar"], frame["kcal"])[0, 1]
    assert n.measure == pytest.approx(r, rel=1e-12) and 0.6 < r < 0.7
    assert n.predicted.changes_question and n.predicted.value is None and n.predicted.band == 2
    assert n.answered  # the capture answered the energy model and the contrast before Models ended
    assert not M.in_triage(n)  # act on it, already answered: it drops out


def test_the_screens_rows_times_smd_is_computed_by_hand(noticed, frame):
    """The engine's count is the capture's own record; EFSA's procedure written out agrees on every
    row not within 1% of a cut-off; the leavers' SMDs are recomputed by hand on what is adjusted
    for, never on the outcome."""
    inside = screen(frame, C.GOLDBERG)["inside"].to_numpy()
    leaving = ~inside
    assert int(leaving.sum()) == CAPTURED_LEAVERS
    bmr = efsa_bmr_kcal(frame["gender"].to_numpy(), frame["age"].to_numpy(dtype=float),
                        frame["weight"].to_numpy(dtype=float), frame["height"].to_numpy(dtype=float))
    s = math.sqrt(23.0 ** 2 / 1 + 8.5 ** 2 + 15.0 ** 2)  # Black 2000, d = 1
    low, high = 1.55 * math.exp(-2 * s / 100), 1.55 * math.exp(2 * s / 100)
    ratio = frame["kcal"].to_numpy() / bmr
    longhand = (ratio >= low) & (ratio <= high)
    clear = (np.abs(ratio / low - 1) > 0.01) & (np.abs(ratio / high - 1) > 0.01)
    assert (longhand[clear] == inside[clear]).all() and clear.mean() > 0.98

    smds = {c: hand_smd(frame[c], leaving) for c in ADJUSTED}
    worst = max(smds, key=smds.get)
    want = CAPTURED_LEAVERS / N_ROWS * smds[worst]
    n = noticed[SCREEN]
    assert worst == "kcal"
    assert n.predicted.instrument == "rows_smd"
    assert n.predicted.value == pytest.approx(want, rel=1e-9)
    assert n.predicted.band == 1 and n.predicted.calibrated  # 0.185: could bias
    assert "`5,212` of `21,849` rows (23.9%)" in n.predicted.words
    assert "glucose" not in n.predicted.words
    assert n.done and "Banna" in n.done  # the every-row analysis is declared


def test_day_to_day_variance_is_not_measurable_on_one_recall_day(noticed):
    n = noticed[DAYS]
    assert n.predicted.not_measurable and n.predicted.band == 1 and n.predicted.value is None
    assert "one recall day" in n.predicted.words and n.done is None


# ── 2–5 · triage, lock, verification, ledger ─────────────────────────────────


@pytest.fixture(scope="module")
def loop(proof, noticed):
    """Triage (Confirm all, as recommended), the lock, then the sensitivity stage."""
    state = C.proof_state()
    noticings = list(noticed.values())
    log = quest.quest_log(state, [], steps_until(None))
    t = sweep.triage(state, log, {"findings": []}, noticed=noticings)
    confirm = decisions.validate({"kind": "confirm_sweep", "stage": "models",
                                  "sweep": "noticings"}, Ctx(state, log, t))
    records = [record(1, confirm.model_dump(mode="json"))]
    triaged = decisions.fold_onto(state, records[0].decision)
    lock = decisions.validate({"kind": "lock_plan"}, Ctx(triaged))
    records.append(record(2, lock.model_dump(mode="json")))
    locked = decisions.fold_onto(triaged, records[1].decision)
    return {"state": state, "triage": t, "records": records, "triaged": triaged,
            "locked": locked, "lock": lock, "noticings": noticings,
            "artifacts": proof.artifacts(locked)}


def test_the_triage_recommends_from_each_band_and_drafts_one_limitation(loop):
    t = loop["triage"]
    got = {i.id: (i.recommended, i.limitation) for i in t.items}
    assert got == {SCREEN: ("could_bias", False),  # the every-row analysis is something done
                   DAYS: ("could_bias", True),  # nothing here can check it
                   COLLINEAR: ("no_change", False)}  # exact by theorem: sugar is outside
    assert t.stage == "models" and t.confirmable and not t.blockers
    assert next(i for i in t.items if i.id == DAYS).label.endswith("(not measurable here)")


def test_the_locks_digest_covers_the_dispositions(loop):
    lock = loop["lock"]
    assert {l["key"]: l["value"] for l in lock.plan[plan_lock.TRIAGE]} == {
        SCREEN: "could_bias", DAYS: "could_bias", COLLINEAR: "no_change"}
    assert lock.digest == plan_lock.digest(lock.plan)
    assert loop["locked"].plan_locked
    # The gate has passed: the triage can no longer be confirmed.
    after = sweep.triage(loop["locked"], quest.quest_log(loop["locked"], loop["records"],
                                                         steps_until(None)),
                         {"findings": []}, loop["records"], noticed=loop["noticings"])
    assert after.passed and not after.confirmable


def test_before_the_lock_the_ledger_reads_no_refit(loop):
    book = M.ledger(loop["triaged"], loop["noticings"], loop["artifacts"])
    assert not book.locked
    assert {r.thread: r.verdict for r in book.rows} == {ENERGY: "not_graded", SCREEN: "pending",
                                                        DAYS: "pending", COLLINEAR: "not_graded"}
    assert all(r.realized is None for r in book.rows)


def test_after_the_lock_the_screen_is_verified_against_a_hand_refit(loop, frame):
    book = M.ledger(loop["locked"], loop["noticings"], loop["artifacts"])
    rows = {r.thread: r for r in book.rows}
    inside = screen(frame, C.GOLDBERG)["inside"].to_numpy()
    est, low, high = ols_sugar(frame[inside])
    every, _, _ = ols_sugar(frame)
    want = abs(every - est) / ((high - low) / 2)
    screen_row = rows[SCREEN]
    assert screen_row.recorded == "could_bias" and screen_row.recommended == "could_bias"
    assert screen_row.realized.instrument == "sensitivity"
    assert screen_row.realized.value == pytest.approx(want, rel=1e-6)
    assert screen_row.realized.band == 1 and screen_row.verdict == "confirmed"  # 0.124: band 1
    assert screen_row.label is None and not screen_row.limitation
    assert screen_row.exhibit == "sensitivity"
    assert "Checked after the plan was fixed" in screen_row.sentence
    days = rows[DAYS]
    assert days.verdict == "not_verifiable" and days.limitation
    assert days.sentence.startswith("Limitation:")
    assert rows[ENERGY].verdict == "not_graded" and rows[ENERGY].recommended is None
    assert book.locked and book.limitations == 1 <= book.budget


def test_the_ledger_replays_and_calibrates_the_committed_file(loop):
    """The ledger is derived: folding the same records onto the same answers gives the same rows.
    Its calibration case is among the committed file's, and the file is what the proof pipeline
    gives across its ladder of energy screens, labeled as that pipeline and not as the capture."""
    first = M.ledger(loop["locked"], loop["noticings"], loop["artifacts"])
    replayed = C.proof_state()
    for r in loop["records"]:
        replayed = decisions.fold_onto(replayed, r.decision)
    again = M.ledger(replayed, loop["noticings"], loop["artifacts"])
    assert again.model_dump() == first.model_dump()
    cases = M.cases_from(C.JOURNEY, first.rows, pipeline=C.PIPELINE,
                         alternative=C.GOLDBERG_OUT)
    assert [(c.thread, c.instrument, c.instrument_post, c.band_post) for c in cases] == [
        (SCREEN, "rows_smd", "sensitivity", 1)]
    committed = M.calibration()
    assert cases[0].model_dump(mode="json") in committed["cases"]
    assert "complete cases" in C.PIPELINE and "not the capture" in C.PIPELINE
    assert C.JOURNEY != "dietary-inference"  # the capture as recorded is not what was run
    rebuilt = C.build()
    assert rebuilt == committed, "run: venv/bin/python -m turbotab.core.tests.materiality_cases --write"
    assert committed["instruments"]["rows_smd"]["confusion"]["false_reassurance"] == 0


def test_a_narrow_screen_once_said_to_change_nothing_could_bias_and_is_upgraded_openly(proof, frame):
    """The verifier's case: a 500–5,000 kcal screen removes few rows that differ little on what is
    adjusted for (rows × SMD ≈ 0.019), yet the every-row refit moves sugar's coefficient by about
    0.63 of its half-width, since the extreme reports are high-leverage. Calibrated on the ladder
    of screens, the rows instrument calls it "could bias", never "doesn't change your numbers
    here"; after the lock the realized band 2 is above it, so the exhibit is relabeled openly."""
    rule = decisions.ExclusionRule(column="kcal", low=500, high=5000, reason="implausible intake")
    state = C.proof_state(exclusions=[rule])
    n = {x.thread: x for x in proof.noticings(state)}[SCREEN]
    kcal = frame["kcal"]
    inside = ((kcal >= 500) & (kcal <= 5000)).to_numpy()
    leaving = ~inside
    want_pre = leaving.mean() * max(hand_smd(frame[c], leaving) for c in ADJUSTED)
    assert n.predicted.value == pytest.approx(want_pre, rel=1e-9) and want_pre < 0.05
    assert n.predicted.band == 1 and M.recommend(n)[0] == "could_bias"
    locked = state.model_copy(update={"plan_locked": True})
    row = next(r for r in M.ledger(locked, [n], proof.artifacts(locked)).rows)
    est, low, high = ols_sugar(frame[inside])
    every, _, _ = ols_sugar(frame)
    want_post = abs(every - est) / ((high - low) / 2)
    assert row.realized.value == pytest.approx(want_post, rel=1e-6) and want_post > 0.5
    assert row.realized.band == 2 and row.verdict == "upgraded"
    assert row.label.startswith("Moved more than predicted") and "possibly bias" in row.label
