"""Q-d · the near-collinear adjustment noticing and the exact "invariance" instrument
(WAVE_C6A_PLAN §3 row Q-d, §5 quest-log items 1–2 and item (d), §7 rulings 1–2; SURFACING_POLICY
§2.6 as amended; noticing family K5, thread ``shared-collinear-predictors``).

A total beside its parts (kcal beside protein, carbohydrate, fat and alcohol) makes the adjustment
terms nearly collinear. Whether that matters for the estimate depends on where what you study
sits, read from Belsley's variance-decomposition proportions (Belsley, Kuh & Welsch 1980): outside
the dependency, the estimate and its interval do not depend on how the adjustment terms are
written (Frisch–Waugh–Lovell: they enter only through the space they span), so the noticing is
band 0, "doesn't change your numbers here", disclosed For the record with the other coefficients
labeled as adjustment terms; inside it, which terms are held fixed changes what the estimate
means (band 2, decided at the adjustment question); an exact identity is a blocker (T1).

The references are independent of the engine: the proportions from an SVD written here, and
statsmodels refits showing the theorem the band-0 claim rests on.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d
from turbotab.core import materiality as M
from turbotab.core import quest, sweep
from turbotab.core.decisions import ProjectState
from turbotab.core.tests.test_confirm_sweep import log_of

THREAD = "shared-collinear-predictors"
PARTS = ["protein", "carb", "fat", "alcohol"]
ATWATER = {"protein": 4.0, "carb": 4.0, "fat": 9.0, "alcohol": 7.0}


def table(n=400, seed=11, *, exact=False, sugar_inside=False):
    """A dietary table drawn here: kcal is the Atwater sum of its parts plus a little noise (or
    exactly, with ``exact``). ``sugar_inside``: carb is sugar plus starch, with starch also in."""
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({
        "protein": rng.gamma(9, 9, n), "carb": rng.gamma(8, 30, n), "fat": rng.gamma(6, 13, n),
        "alcohol": rng.gamma(1.2, 8, n), "age": rng.uniform(20, 80, n),
        "gender": rng.choice(["male", "female"], n)})
    f["sugar"] = 0.25 * f["carb"] + rng.gamma(3, 10, n)
    if sugar_inside:
        f["starch"] = rng.gamma(6, 25, n)
        f["carb"] = f["sugar"] + f["starch"] + rng.normal(0, 1.0, n)
    atwater = sum(w * f[c] for c, w in ATWATER.items())
    f["kcal"] = atwater if exact else atwater + rng.normal(0, 15, n)
    f["glucose"] = 90 + 0.04 * f["sugar"] + 0.2 * f["age"] + rng.normal(0, 8, n)
    return f


def state(columns, **update) -> ProjectState:
    roles = {c: "covariate" for c in columns}
    roles.update(sugar="exposure", kcal="energy")
    base = dict(lens=["dietary"], target="glucose", purpose="inference", roles=roles,
                estimand=d.EstimandSpec(exposure="sugar", measure="mean_difference"))
    return ProjectState(**{**base, **update})


ADJUST = ["kcal", *PARTS, "age", "gender"]


def hand_proportions(f: pd.DataFrame, columns: list[str]):
    """Belsley's proportions by hand: [1, X] (gender as one indicator), each column scaled to unit
    length, its SVD; π[j, k] = (v_jk² / s_k²) / Σ_k (v_jk² / s_k²); η_k = s_max / s_k."""
    X = [np.ones(len(f))]
    names = ["(intercept)"]
    for c in columns:
        if c == "gender":
            X.append((f[c] == "male").to_numpy(float))
        else:
            X.append(f[c].to_numpy(float))
        names.append(c)
    A = np.column_stack(X)
    A = A / np.sqrt((A ** 2).sum(axis=0))
    _u, s, vt = np.linalg.svd(A, full_matrices=False)
    phi = (vt.T ** 2) / s ** 2
    return names, phi / phi.sum(axis=1, keepdims=True), s[0] / s


def noticing(f, columns=ADJUST, **update):
    return M.collinear_noticing(state(columns, **update), f)


# ── outside the dependency: band 0 by theorem ────────────────────────────────


def test_sugar_outside_the_dependency_is_band_zero_and_for_the_record():
    f = table()
    n = noticing(f)
    assert n is not None and (n.thread, n.family, n.stage) == (THREAD, "K5", "models")
    names, pi, eta = hand_proportions(f, ["sugar", *ADJUST])
    near = [k for k in range(len(eta)) if eta[k] >= 30
            and sum(pi[j, k] >= 0.5 for j in range(1, len(names))) >= 2]
    assert near, "the drawn table holds a near dependency"
    want = sum(pi[names.index("sugar"), k] for k in near)
    assert want < 0.5 and n.measure == pytest.approx(want, abs=1e-10)
    # The dependency is kcal and its parts, named in plain words; sugar is not in it.
    inside = {names[j] for k in near for j in range(1, len(names)) if pi[j, k] >= 0.5}
    assert inside == {"kcal", *PARTS} and set(n.subject) == {"sugar", *inside}
    m = n.predicted
    assert (m.instrument, m.band, m.value, m.calibrated) == ("invariance", 0, 0.0, True)
    assert "Frisch–Waugh–Lovell" in (m.theorem or "") and "Frisch–Waugh–Lovell" in m.label
    assert "adjustment terms" in m.words
    disposition, reason, owed = M.recommend(n)
    assert disposition == "no_change" and not owed
    assert M.BAND_WORDS[0] == "doesn't change your numbers here"
    assert n.decides_by == "disclosure" and not n.blocker
    assert M.noticing_tier(n) == quest.RECORD


def test_the_theorem_by_statsmodels_rewriting_kcal_leaves_sugar_alone():
    """Swapping kcal for kcal − (4P + 4C + 9F + 7A) spans the same space: sugar's coefficient and
    its standard error are identical; the adjustment terms' own coefficients are not."""
    f = table()
    others = ["protein", "carb", "fat", "alcohol", "age"]
    X1 = sm.add_constant(f[["sugar", "kcal", *others]].to_numpy(float))
    rest = f["kcal"] - sum(w * f[c] for c, w in ATWATER.items())
    X2 = sm.add_constant(np.column_stack([f["sugar"], rest, f[others]]))
    y = f["glucose"].to_numpy(float)
    a, b = sm.OLS(y, X1).fit(), sm.OLS(y, X2).fit()
    assert abs(a.params[1] - b.params[1]) < 1e-10
    assert abs(a.bse[1] - b.bse[1]) < 1e-10
    assert abs(a.params[3] - b.params[3]) > 1e-3  # protein's own coefficient moves


def test_the_noticing_is_outcome_blind():
    f = table()
    shuffled = f.assign(glucose=np.random.default_rng(3).permutation(f["glucose"].to_numpy()))
    assert noticing(shuffled).model_dump() == noticing(f).model_dump()
    assert noticing(f.drop(columns=["glucose"])).model_dump() == noticing(f).model_dump()


def test_no_noticing_without_a_near_dependency_or_under_predict():
    f = table()
    assert noticing(f, columns=["age", "gender", "protein"]) is None
    assert noticing(f, purpose="prediction") is None


# ── inside the dependency: it changes the question ───────────────────────────


def test_sugar_inside_the_dependency_is_decided_at_the_adjustment_question():
    f = table(sugar_inside=True)
    columns = [*ADJUST, "starch"]
    n = noticing(f, columns=columns)
    names, pi, eta = hand_proportions(f, ["sugar", *columns])
    near = [k for k in range(len(eta)) if eta[k] >= 30
            and sum(pi[j, k] >= 0.5 for j in range(1, len(names))) >= 2]
    want = sum(pi[names.index("sugar"), k] for k in near)
    assert want >= 0.5 and n.measure == pytest.approx(want, abs=1e-10)
    assert n.predicted.band == 2 and n.predicted.changes_question
    assert n.question == "adjustment" and n.decides_by == "meaning" and not n.blocker
    assert "starch" in n.subject and "carb" in n.subject
    assert M.recommend(n)[0] == "act_on_it"
    assert M.noticing_tier(n) == quest.DECIDE


# ── an exact identity: a blocker ─────────────────────────────────────────────


def test_an_exact_atwater_identity_is_a_blocker():
    f = table(exact=True)
    n = noticing(f)
    assert n.blocker and n.predicted.band == 2
    assert set(n.subject) >= {"kcal", *PARTS}
    assert M.noticing_tier(n) == quest.DECIDE
    s = state(ADJUST)
    log = log_of(s, findings={"findings": []}, columns=list(f.columns))
    t = sweep.triage(s, log, {"findings": []}, noticed=[n])
    item = next(i for i in t.items if i.id == THREAD)
    assert item.blocker and item.recommended == "act_on_it"
    assert t.blockers == 1 and not t.confirmable and t.items[0].id == THREAD


# ── the instrument: exact by theorem, not floored, kept by calibrate ─────────


def test_an_exact_instrument_is_not_floored_and_calibrate_keeps_it():
    entry = M.calibration()["instruments"]["invariance"]
    assert "Frisch–Waugh–Lovell" in entry["theorem"] and entry["calibrated"]
    assert M.band_of("invariance", 0.0) == 0
    assert M.movement("invariance", 0.0, "w").band == 0
    # Every proxy that is not exact keeps the floor.
    assert M.band_of("exposure_shift", 0.0) == 1
    start = json.loads(json.dumps(M.calibration()))
    out = M.calibrate([], start)
    assert out["instruments"]["invariance"] == entry


def test_the_ledger_names_the_theorem():
    f = table()
    n = noticing(f)
    book = M.ledger(state(ADJUST), [n])
    row = next(r for r in book.rows if r.thread == THREAD)
    assert "Frisch–Waugh–Lovell" in (row.predicted.theorem or "")
    assert row.recommended == "no_change" and not row.limitation
    assert row.verdict == "not_graded"  # exact: there is nothing to check after the lock
