"""A spline whose k the analyst left to Harrell's rule, replayed on fresh draws through the engine
(``core/export/replay.py``, :func:`replay.fresh_splines`; the readiness audit's finding F2 and the
ruling of 2026-10-23: the record is a plan to apply the rule, so a fresh draw applies it to its
own size, under inference and prediction alike).

Four short reference journeys (``turbotab.core.reference.journeys``) on a table written here as
the Estimate fixture is (``walking_bp.csv``'s columns and truth), whose only blank cells are in
``chronic_pain``, so a complete-case analysis keeps a number of rows known by construction on
every draw (:data:`BLANKS`; a numeric outcome's effective sample size is that number). Every
expected value is read from outside the engine: k from Harrell's rule as RMS states it, the knot
percentiles from RMS's table, the estimates from statsmodels' least squares on the complete cases
with the restricted cubic spline built here (knots placed as ``Hmisc::rcspline.eval`` places
them), the plan lock's digest from hashlib.

* Under inference (complete cases, ``set_forms`` giving ``age`` a spline with no k): a draw of
  another size at the same k (120 to 130, k = 5) and one at another k (120 to 80, k = 4) are each
  rebuilt, the estimate served and the primary model's estimate statsmodels' at that k on that
  draw's complete cases; the form record, the methods text and the plan lock state that draw's k
  and size and none of the recorded draw's, the lock's digest hashlib's over its plan. A draw
  whose bytes are the recorded table's is replayed as recorded. A k the analyst gave (4) is kept
  on a draw where the rule gives 5, its record its own bytes.
* Under prediction (complete cases, ``set_exposure_form`` once the rows are read): a draw at
  another k (120 to 80, k = 4) is rebuilt at that k. A later record that changed what the size is
  read under (the outcome put on the log scale after the spline) still refuses the rebuild on a
  draw of another size, naming the record and the change.
* Under prediction the engine reads the rule when the spline is declared and keeps that k, so a
  fresh draw reads it on the rows as they stood at the spline's record. With the rows under 45
  excluded after the spline (pandas: 120 complete cases to 85, k = 5 to 4): on the recorded
  table's bytes the record is replayed as recorded, k = 5 on 120, and the cross-validated scores
  are the recorded analysis's; on the draw of 130 complete cases (92 as the log ends, k = 4) it
  is rebuilt at k = 5 on 130, the record and the cross-validated scores those of the same answers
  given on that draw directly (its own journey).
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

N, SEED = 150, 7
# The blank cells of ``chronic_pain`` on each draw: complete cases N − blanks, 120 at the recorded
# seed, 130 at the next and 80 at the one after.
BLANKS = {SEED: 30, SEED + 1: 20, SEED + 2: 70}
SAME_K, OTHER_K = SEED + 1, SEED + 2
FIXTURE = "walking_bp.csv"


def walking_bp(seed: int) -> pd.DataFrame:
    """The Estimate fixture's columns at ``seed``: walking lowers systolic pressure, pain raises
    both, age raises it on a curve, the concern score is a collider, height does nothing; exactly
    ``BLANKS[seed]`` cells of ``chronic_pain`` blank, none elsewhere."""
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 80, N)
    pain = (rng.random(N) < 0.35).astype(int)
    walks = (rng.random(N) < 1 / (1 + np.exp(-(0.3 - 1.0 * pain)))).astype(int)
    height = np.round(rng.normal(170, 9, N), 1)
    sbp = np.round(120 - 6 * walks + 8 * pain + 0.4 * (age - 55) + 0.01 * (age - 55) ** 2
                   + rng.normal(0, 8, N), 1)
    score = np.round(50 - 5 * walks + 0.5 * (sbp - 125) + rng.normal(0, 5, N), 1)
    frame = pd.DataFrame({"person_id": [f"P{i + 1:05d}" for i in range(N)],
                          "age": age.astype(int), "chronic_pain": pd.array(pain, dtype="Int64"),
                          "walks_daily": walks, "height_cm": height,
                          "health_concern_score": score, "sbp_mmhg": sbp})
    frame.loc[rng.choice(N, BLANKS[seed], replace=False), "chronic_pain"] = pd.NA
    return frame


def complete(seed: int) -> pd.DataFrame:
    frame = walking_bp(seed).dropna()
    assert len(frame) == N - BLANKS[seed]
    return frame


def harrell(n: float) -> int:
    """Harrell's rule as RMS §2.4.6 states it: 3 knots below an effective sample size of 30, 5
    from 100, else 4."""
    return 3 if n < 30 else 5 if n >= 100 else 4


# Harrell's knot placement (RMS §2.4.6, Table 2.3): the quantiles at each k, and as the sentence
# words them.
QUANTILES = {3: (0.10, 0.50, 0.90), 4: (0.05, 0.35, 0.65, 0.95),
             5: (0.05, 0.275, 0.50, 0.725, 0.95)}
PERCENTILES = {3: "10, 50 and 90", 4: "5, 35, 65 and 95", 5: "5, 27.5, 50, 72.5 and 95"}


def knots(x: np.ndarray, k: int) -> np.ndarray:
    """``rcspline.eval``'s knots: the quantiles of RMS's table (R's type 7), the outer two the 5th
    smallest and largest values below 100 values. Its tie rule (a lowest or highest value held by
    5% or more of the values) is not reached here, which the caller checks."""
    t = np.quantile(x, QUANTILES[k])
    if len(x) < 100:
        ordered = np.sort(x)
        t[0], t[-1] = ordered[4], ordered[-5]
    return t


def rcs(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The restricted cubic spline's k − 2 nonlinear terms at knots ``t`` (RMS eq. 2.25, before
    its scaling, which changes no other coefficient)."""
    k = len(t)

    def cube(u: np.ndarray) -> np.ndarray:
        return np.maximum(u, 0.0) ** 3

    return np.column_stack([cube(x - t[j]) - cube(x - t[k - 2]) * (t[k - 1] - t[j])
                            / (t[k - 1] - t[k - 2]) + cube(x - t[k - 1]) * (t[k - 2] - t[j])
                            / (t[k - 1] - t[k - 2]) for j in range(k - 2)])


def primary(seed: int, k: int) -> float:
    """The primary model's estimate of ``walks_daily`` (Model 2: adjusted for ``chronic_pain`` and
    a spline of ``age`` at ``k`` knots) on the draw's complete cases: statsmodels' least squares."""
    import statsmodels.api as sm

    frame = complete(seed)
    age = frame["age"].to_numpy(float)
    _, counts = np.unique(age, return_counts=True)
    assert counts[0] / len(age) < 0.05 and counts[-1] / len(age) < 0.05  # no tie rule
    x = np.column_stack([np.ones(len(frame)), frame["walks_daily"].to_numpy(float),
                         frame["chronic_pain"].to_numpy(float), age, rcs(age, knots(age, k))])
    return float(sm.OLS(frame["sbp_mmhg"].to_numpy(float), x).fit().params[1])


ROLES = {"person_id": "identifier", "age": "covariate", "chronic_pain": "covariate",
         "walks_daily": "exposure", "height_cm": "covariate", "health_concern_score": "covariate"}
COMPLETE_CASES = {"kind": "set_missing", "strategy": "complete_case"}


def _journey(folder: Path, name: str, purpose: str, seed: int = SEED, **more) -> Path:
    from turbotab.core.reference import journeys as J
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / FIXTURE
    walking_bp(seed).to_csv(path, index=False)
    spec = J.Journey(name, "clinical", purpose, "?", str(path), "test", lambda: path,
                     lambda: J.truth(dict(FIXTURE_TRUTHS[FIXTURE]), fixture=path.name),
                     target="sbp_mmhg", lenses=("clinical",), **more)
    capture = J.run_journey(spec, bundles=folder, out_dir=folder / "captures")
    assert capture["export"]["status"] == 200, capture["notes"]
    return folder / f"{name}.zip"


def _spline_once_the_rows_are_read(*stages: str, outcome_scale: str | None = None):
    """Under prediction: ``age`` read as an amount (as the fixture's truth reads it; a form waits
    for that reading), then, once ``stages`` are computed again (those the rule reads), a spline of
    ``age`` with no k given; then, where ``outcome_scale`` is given, the outcome put on that scale
    (before the split, where a scale is chosen)."""
    def go(run, view) -> None:
        from turbotab.core.reference import journeys as J

        J.post_answer(run, {"kind": "confirm_reading", "reading": "code_or_count",
                            "column": "age", "value": "amount"})
        for stage in stages:
            J._artifact(run, stage)
        J.post_answer(run, {"kind": "set_exposure_form", "column": "age", "form": "spline"})
        if outcome_scale is not None:
            J.post_answer(run, {"kind": "set_outcome_scale", "column": "sbp_mmhg",
                                "scale": outcome_scale})
    return go


# The rows under this age, excluded after the spline (the verifier's 60-and-over journey built on
# this table): the complete cases aged 45 and over are below 100 on the draws used here.
AGED = 45
PREDICT = {"missing": COMPLETE_CASES,
           "split": {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
           "models": {"kind": "select_models", "models": ["linear"]}}


def _spline_then_aged(run, view) -> None:
    """The spline of ``age`` once the rows are read, then an exclusion of the rows under 45."""
    from turbotab.core.reference import journeys as J

    _spline_once_the_rows_are_read("target_info", "working", "cohort", "split")(run, view)
    J.post_answer(run, {"kind": "set_exclusions", "rules": [
        {"kind": "range", "column": "age", "low": AGED, "reason": "45 and over"}]})


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    from turbotab.core.reference import journeys as J

    folder = tmp_path_factory.mktemp("spline-rule")
    estimate = dict(exposure="walks_daily", roles=dict(ROLES), answers={"missing": COMPLETE_CASES})
    predict = dict(PREDICT)
    return {
        "later": _journey(folder / "later", "later", "prediction", answers=predict,
                          before={"models": _spline_then_aged}),
        "rule": _journey(folder / "rule", "rule", "inference", **estimate, before={
            "models": J.declare({"kind": "set_forms", "forms": {"age": {"form": "spline"}}})}),
        "hand": _journey(folder / "hand", "hand", "inference", **estimate, before={
            "models": J.declare({"kind": "set_forms",
                                 "forms": {"age": {"form": "spline", "knots": 4}}})}),
        "predict": _journey(folder / "predict", "predict", "prediction", answers=predict,
                            before={"models": _spline_once_the_rows_are_read(
                                "target_info", "working", "cohort", "split")}),
        "log": _journey(folder / "log", "log", "prediction", answers=predict,
                        before={"split": _spline_once_the_rows_are_read(
                            "target_info", "working", "cohort", outcome_scale="log")}),
    }


def _files(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as z:
        return {n.split("/", 1)[1]: z.read(n) for n in z.namelist() if "/" in n}


def _records(log: bytes) -> list[dict]:
    return [json.loads(line) for line in log.splitlines() if line.strip()]


def _age_form(records: list[dict]) -> tuple[dict, dict]:
    """The last record that gives ``age`` its form, and the form."""
    found = [r for r in records if (r["decision"]["kind"] == "set_forms"
                                    and "age" in r["decision"]["forms"])
             or (r["decision"]["kind"] == "set_exposure_form" and r["decision"]["column"] == "age")]
    record = found[-1]
    d = record["decision"]
    form = d["forms"]["age"] if d["kind"] == "set_forms" else {
        k: v for k, v in d.items() if k not in ("kind", "column")}
    return record, form


def _lock(records: list[dict]) -> dict:
    [lock] = [r for r in records if r["decision"]["kind"] == "lock_plan"]
    return lock


def _sha256(plan: dict) -> str:
    """SHA-256 over the canonical JSON (keys sorted, no spaces, UTF-8), by hashlib."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _rule_words(k: int, n: int) -> str:
    return (f"with `{k}` knots, k = {k} by Harrell's rule (3 knots below an effective sample size "
            f"of 30, 5 from 100, else 4; here {n:,}, the number of analyzed rows), at the "
            f"{PERCENTILES[k]} percentiles")


def _refilled(seq: int, kind: str, k0: int, n0: int | None, k: int, n: int) -> str:
    before = (f"k = {k0} on an effective sample size of {n0:,}" if n0 is not None
              else f"k = {k0} with no effective sample size read")
    return (f"Record {seq} ({kind}) gives `age` a spline whose k is Harrell's rule on the analyzed "
            f"rows: {before} on the recorded draw, and on this draw the rule reads {n:,} (k = {k}) "
            f"on the rows as the log ends. Its k was left to the rule, so the record is replayed "
            f"at the rule's k on this draw, stating k = {k} on an effective sample size of {n:,} "
            f"in place of {before}.")


@pytest.mark.parametrize("seed", [SAME_K, OTHER_K])
def test_a_complete_case_estimate_spline_is_rebuilt_at_the_rules_k_on_each_draws_size(
        bundles, tmp_path, seed):
    """F2: the record's k = 5 by the rule on the recorded draw's 120 complete cases. A draw of 130
    (k = 5 again) and one of 80 (k = 4) are each rebuilt, not refused: the primary model's
    estimate is statsmodels' at the draw's k on its complete cases; the form record states that
    k and size, in the sentence the methods text quotes, with that k's percentiles and none of
    the recorded draw's words; the plan lock holds that form, its digest hashlib's over the plan
    it holds and quoted in its sentence and the methods text; the report's message states both
    draws' k and size."""
    from turbotab.core.export import replay

    recorded = _records(_files(bundles["rule"])["decisions.jsonl"])
    record, form = _age_form(recorded)
    n0, n = N - BLANKS[SEED], N - BLANKS[seed]
    assert (form["form"], form["knots_rule"], form["n_effective"], form["knots"]) == (
        "spline", "harrell", float(n0), harrell(n0))
    assert record["seq"] < _lock(recorded)["seq"]
    k = harrell(n)
    assert n != n0 and (k == harrell(n0)) == (seed == SAME_K)
    report = replay.run(bundles["rule"], workers=2, keep=True, home=str(tmp_path / "home"),
                        fresh=replay.FreshDraw(walking_bp, SEED, seed))
    assert report.refused is None and report.finished is True, (report.refused, report.messages)
    key = "table2|linear/model_2/walks_daily|estimate"
    assert report.estimates_again[key] == pytest.approx(primary(seed, k), abs=1e-9)
    assert abs(primary(seed, k) - primary(seed, 9 - k)) > 1e-3  # the k is what the estimate shows
    assert [m for m in report.messages if "Harrell" in m] == [
        _refilled(record["seq"], record["decision"]["kind"], harrell(n0), n0, k, n)]
    files = _files(tmp_path / "home" / "replay" / "bundle.zip")
    rebuilt = _records(files["decisions.jsonl"])
    again, form_again = _age_form(rebuilt)
    assert again["seq"] == record["seq"]
    assert form_again == {**form, "knots": k, "n_effective": float(n)}
    assert _rule_words(k, n) in again["sentence"] and f"here {n0}," not in again["sentence"]
    methods = files["methods.md"].decode()
    assert again["sentence"] in methods and f"here {n0}," not in methods
    lock, lock_again = _lock(recorded), _lock(rebuilt)
    plan = json.loads(json.dumps(lock["decision"]["plan"]))
    plan["exposure_forms"]["age"].update(knots=k, n_effective=float(n))
    assert lock_again["decision"]["plan"] == plan
    digest = _sha256(plan)
    assert lock_again["decision"]["digest"] == digest != lock["decision"]["digest"]
    assert lock_again["sentence"] == lock["sentence"].replace(lock["decision"]["digest"][:12],
                                                              digest[:12])
    assert f"SHA-256 `{digest[:12]}`" in methods
    assert lock["decision"]["digest"][:12] not in methods
    exported = json.loads(files["analysis_plan.json"])
    assert exported["plan"] == plan and exported["plan_sha256"] == digest


def test_a_draw_whose_bytes_are_the_recorded_tables_is_replayed_as_recorded(bundles):
    """The rule's size on the recorded table's bytes is the record's (120): kept, no message, the
    primary model's estimate the recorded one (statsmodels' at k = 5 on those 120 rows)."""
    from turbotab.core.export import replay

    report = replay.run(bundles["rule"], workers=2, fresh=replay.FreshDraw(
        lambda seed: walking_bp(SEED), SEED, SEED + 1))
    assert report.fresh.sha256 == report.fresh.recorded_sha256
    assert report.refused is None and report.finished is True, (report.refused, report.messages)
    assert not any("Harrell" in m for m in report.messages)
    key = "table2|linear/model_2/walks_daily|estimate"
    assert report.estimates_again[key] == pytest.approx(primary(SEED, 5), abs=1e-9)


def test_a_k_the_analyst_gave_is_kept_on_a_draw_where_the_rule_gives_another(bundles, tmp_path):
    """The analyst gave k = 4; on the draw of 130 complete cases the rule would give 5. The record
    is its own bytes in the rebuilt log, no message names the rule, and the primary model's
    estimate is statsmodels' at k = 4 on those 130 rows, not at 5."""
    from turbotab.core.export import replay

    log = _files(bundles["hand"])["decisions.jsonl"]
    record, form = _age_form(_records(log))
    assert (form["knots"], form["knots_rule"], form["n_effective"]) == (4, None, None)
    assert harrell(N - BLANKS[SAME_K]) == 5
    report = replay.run(bundles["hand"], workers=2, keep=True, home=str(tmp_path / "home"),
                        fresh=replay.FreshDraw(walking_bp, SEED, SAME_K))
    assert report.refused is None and report.finished is True, (report.refused, report.messages)
    assert not any("Harrell" in m for m in report.messages)
    key = "table2|linear/model_2/walks_daily|estimate"
    assert report.estimates_again[key] == pytest.approx(primary(SAME_K, 4), abs=1e-9)
    assert abs(primary(SAME_K, 4) - primary(SAME_K, 5)) > 1e-3
    line = next(x for x in log.splitlines() if json.loads(x)["seq"] == record["seq"])
    rebuilt = _files(tmp_path / "home" / "replay" / "bundle.zip")["decisions.jsonl"]
    assert line in rebuilt.splitlines()


def test_a_predict_spline_is_rebuilt_at_another_k_on_a_draw_of_another_size(bundles, tmp_path):
    """Under prediction the record's k = 5 by the rule on the 120 analyzed rows; the draw of 80
    (k = 4) is rebuilt at k = 4, cut before the opening, its record and message stating that k and
    size, read on the rows as they stood at the record (no later record changes them), and its
    cross-validated scores served."""
    from turbotab.core.export import replay

    record, form = _age_form(_records(_files(bundles["predict"])["decisions.jsonl"]))
    n0, n = N - BLANKS[SEED], N - BLANKS[OTHER_K]
    assert (form["knots_rule"], form["n_effective"], form["knots"]) == (
        "harrell", float(n0), harrell(n0))
    assert harrell(n) != harrell(n0)
    report = replay.run(bundles["predict"], workers=2, cut=True, keep=True,
                        home=str(tmp_path / "home"),
                        fresh=replay.FreshDraw(walking_bp, SEED, OTHER_K))
    assert report.refused is None and report.finished is True, (report.refused, report.messages)
    assert np.isfinite(report.estimates_again["performance|linear/cv/mse|estimate"])
    assert [m for m in report.messages if "Harrell" in m] == [
        _refilled_at_record(record["seq"], record["decision"]["kind"], harrell(n0), n0,
                            harrell(n), n)]
    again, form_again = _age_form(_records(
        _files(tmp_path / "home" / "replay" / "bundle.zip")["decisions.jsonl"]))
    assert form_again == {**form, "knots": harrell(n), "n_effective": float(n)}
    assert _rule_words(harrell(n), n) in again["sentence"]


def test_a_later_record_that_changed_what_the_size_is_read_under_still_refuses(bundles):
    """The outcome put on the log scale after the spline: on the draw of 130 rows the rule reads
    130 (k = 5) under `ln_sbp_mmhg`, and the record's size and words were read under `sbp_mmhg`,
    so the rebuild is refused, the record and the change named, nothing rebuilt."""
    from turbotab.core.export import replay

    recorded = _records(_files(bundles["log"])["decisions.jsonl"])
    record, form = _age_form(recorded)
    [scale] = [r["seq"] for r in recorded if r["decision"]["kind"] == "set_outcome_scale"]
    assert scale > record["seq"] and form["knots_rule"] == "harrell"
    said = (f"k = {form['knots']} on an effective sample size of {form['n_effective']:,.0f}"
            if form["n_effective"] is not None
            else f"k = {form['knots']} with no effective sample size read")
    n = N - BLANKS[SAME_K]
    report = replay.run(bundles["log"], workers=2, cut=True,
                        fresh=replay.FreshDraw(walking_bp, SEED, SAME_K))
    assert report.refused == (
        f"Record {record['seq']} (set_exposure_form) gives `age` a spline whose k is Harrell's "
        f"rule on the analyzed rows: {said} on the recorded draw, and on this draw the rule reads "
        f"{n:,} (k = {harrell(n)}) on the rows as the log ends, after record {scale} "
        f"(set_outcome_scale), which changed what the size is read under: the outcome from "
        f"`sbp_mmhg` to `ln_sbp_mmhg` and the outcome's scale from none to log. The record's size "
        f"and sentence are read under the outcome, its scale, task and event as they stood when "
        f"it was recorded, so a size read under later ones is not replayed in it, and the "
        f"rebuild is refused."), (report.refused, report.messages)
    assert report.project is None and report.finished is None


def _refilled_at_record(seq: int, kind: str, k0: int, n0: int, k: int, n: int) -> str:
    """A refill's message under prediction, where the size is read at the spline's record."""
    before = f"k = {k0} on an effective sample size of {n0:,}"
    return (f"Record {seq} ({kind}) gives `age` a spline whose k is Harrell's rule on the analyzed "
            f"rows: {before} on the recorded draw, and on this draw the rule reads {n:,} (k = {k}) "
            f"on the rows as they stood at that record. Its k was left to the rule, which the "
            f"engine reads under prediction when the spline is declared, so the record is "
            f"replayed at the rule's k on this draw's rows as they stood then, stating k = {k} on "
            f"an effective sample size of {n:,} in place of {before}.")


def _cross_validated(path: Path) -> dict:
    """The recorded analysis's cross-validated scores, from its bundle's provenance."""
    estimates = json.loads(_files(path)["provenance.json"])["estimates"]
    return {k: v for k, v in estimates.items() if "/cv/" in k}


def _same_scores(again: dict, recorded: dict) -> None:
    assert recorded and set(recorded) <= set(again)
    for key, value in recorded.items():
        assert (again[key] is None) if value is None else (
            again[key] == pytest.approx(value, abs=1e-9)), key


def test_a_predict_spline_declared_before_an_exclusion_replays_as_recorded_on_its_own_bytes(
        bundles, tmp_path):
    """B1: under prediction the engine reads the rule when the spline is declared and keeps that
    k whatever a later record does to the rows. The record's k = 5 on the 120 complete cases; the
    exclusion of the rows under 45 recorded after it leaves 85 as the log ends (pandas; k = 4).
    On a draw whose bytes are the recorded table's the record is replayed as recorded, no message
    names the rule, and the cross-validated scores are the recorded analysis's to 1e-9."""
    from turbotab.core.export import replay

    recorded = _records(_files(bundles["later"])["decisions.jsonl"])
    record, form = _age_form(recorded)
    [exclusion] = [r["seq"] for r in recorded if r["decision"]["kind"] == "set_exclusions"
                   and r["seq"] > record["seq"]]
    n0, kept = N - BLANKS[SEED], int((complete(SEED)["age"] >= AGED).sum())
    assert (form["knots_rule"], form["n_effective"], form["knots"]) == (
        "harrell", float(n0), harrell(n0))
    assert kept == 85 and harrell(kept) != harrell(n0)
    report = replay.run(bundles["later"], workers=2, cut=True, keep=True,
                        home=str(tmp_path / "home"),
                        fresh=replay.FreshDraw(lambda seed: walking_bp(SEED), SEED, SEED + 1))
    assert report.fresh.sha256 == report.fresh.recorded_sha256
    assert report.refused is None and report.finished is True, (report.refused, report.messages)
    assert not any("Harrell" in m for m in report.messages), report.messages
    again, form_again = _age_form(_records(
        _files(tmp_path / "home" / "replay" / "bundle.zip")["decisions.jsonl"]))
    assert form_again == form and again["sentence"] == record["sentence"]
    _same_scores(report.estimates_again, _cross_validated(bundles["later"]))


def test_a_predict_spline_declared_before_an_exclusion_is_rebuilt_at_the_k_read_at_its_record(
        bundles, tmp_path):
    """B1: on the draw of 130 complete cases (k = 5; 92 aged 45 and over as the log ends, k = 4,
    pandas) the spline is rebuilt at k = 5 on 130, the size at its record, not at the log end's
    k. The rebuilt record (its decision and sentence) and the cross-validated scores are those of
    the same answers given on that draw directly, its own journey, whose log holds the same
    decisions; the message states both draws' k and size at the record."""
    from turbotab.core.export import replay

    recorded = _records(_files(bundles["later"])["decisions.jsonl"])
    record, form = _age_form(recorded)
    n0, n = N - BLANKS[SEED], N - BLANKS[SAME_K]
    end = int((complete(SAME_K)["age"] >= AGED).sum())
    assert harrell(n) == 5 and end == 92 and harrell(end) == 4
    direct = _journey(tmp_path / "direct", "direct", "prediction", seed=SAME_K,
                      answers=dict(PREDICT), before={"models": _spline_then_aged})
    mine = _records(_files(direct)["decisions.jsonl"])
    assert [r["decision"]["kind"] for r in mine] == [r["decision"]["kind"] for r in recorded]
    report = replay.run(bundles["later"], workers=2, cut=True, keep=True,
                        home=str(tmp_path / "home"),
                        fresh=replay.FreshDraw(walking_bp, SEED, SAME_K))
    assert report.refused is None and report.finished is True, (report.refused, report.messages)
    assert [m for m in report.messages if "Harrell" in m] == [
        _refilled_at_record(record["seq"], record["decision"]["kind"], harrell(n0), n0,
                            harrell(n), n)]
    again, form_again = _age_form(_records(
        _files(tmp_path / "home" / "replay" / "bundle.zip")["decisions.jsonl"]))
    assert form_again == {**form, "knots": harrell(n), "n_effective": float(n)}
    assert _rule_words(harrell(n), n) in again["sentence"]
    direct_record, _ = _age_form(mine)
    assert (again["decision"], again["sentence"]) == (direct_record["decision"],
                                                      direct_record["sentence"])
    _same_scores(report.estimates_again, _cross_validated(direct))
