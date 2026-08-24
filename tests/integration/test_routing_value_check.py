"""The routing value check — the verdict, against the frozen pre-registration.

`docs/turbotab/VALUE_CHECK_PREREG.md` was committed before any Router code
existed and states every threshold. This measures Guided on the same three
datasets, over the same exploration window as the committed Classic baseline,
and reports the table beside it.

**The thresholds are read from the prereg, not restated here.** A number copied
into a test is a number that can drift from the thing it was promised against.

If any criterion misses, this test fails and `docs/turbotab/BLOCKED.md` is the
deliverable — that outcome is the check working.
"""
from __future__ import annotations

import json
import pathlib
import re

import pandas as pd
import pytest

from ml import router
from turbotab import engine, measure
from turbotab.measure import (DeferralRecord, Measurement, QuestionRecord,
                              required_decisions)

pytestmark = pytest.mark.timeout(900)

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "turbotab" / "sample_data"
BASELINE = ROOT / "docs" / "turbotab" / "data" / "routing-baseline.json"
# The Classic column is reported from the adjudicated reference, because that is
# the ground truth today; the frozen one is reported beside it so the reader can
# see the movement rather than take it on trust. See
# VALUE_CHECK_ADJUDICATION.md §"The denominator moved".
ADJUDICATED = ROOT / "docs" / "turbotab" / "data" / "routing-baseline-l9c.json"
PREREG = ROOT / "docs" / "turbotab" / "VALUE_CHECK_PREREG.md"
RESULT = ROOT / "docs" / "turbotab" / "data" / "routing-value-check.json"


def _stamp(path) -> str:
    """The commit a baseline file records itself as measured at."""
    return measure.baseline_provenance(path).get("measured_at") or "unknown"


PREREG_AT = _stamp(BASELINE)
ADJUDICATED_AT = _stamp(ADJUDICATED)


def _ratio_from(row: dict, measured_at: str) -> str:
    """`k/n @ commit` for a stored measurement, derived rather than stored.

    The baselines predate the rule and the frozen-measurement rule says the
    envelope may gain labels while the measurements may not be altered — so the
    ratio is recomputed from what the file already holds (`coverage` and the
    required-decision inventory) instead of being written into it.
    """
    n = len(row.get("required") or [])
    coverage = row["metrics"].get("coverage")
    if not n or coverage is None:
        return f"n/a @ {measured_at}"
    return f"{round(coverage * n)}/{n} @ {measured_at}"

DATASETS = [
    ("messy-clinic", DATA / "clinic_visits.csv", "outcome"),
    ("wide-assay", DATA / "wide_assay.csv", "responder"),
    ("longitudinal", DATA / "longitudinal_visits.csv", "outcome"),
]


# ── the thresholds, read from the frozen file ────────────────────────────

def _prereg_thresholds():
    """Parse the binding numbers out of the pre-registration.

    Read rather than restated so the test cannot quietly disagree with the
    document it is supposed to be bound by.
    """
    text = PREREG.read_text(encoding="utf-8")
    out = {}

    messy = re.search(r"### The contested claim.*?\n\n(.*?)\n\n", text, re.S).group(1)
    out["messy-clinic"] = {
        "surfaced_coverage": 1.0,
        "asked_coverage": 8 / 9,
        "max_questions": int(re.search(r"\*\*≤ (\d+)\*\* — at most half", messy).group(1)),
        "max_irrelevant": int(re.search(r"\*\*≤ (\d+)\*\*\s*\|?\s*$", messy, re.M).group(1)),
    }
    guards = re.search(r"### Regression guards.*?\n\n(.*?)\n\n", text, re.S).group(1)
    clean = {
        "coverage": 1.0,
        "max_questions": int(re.search(r"\*\*≤ (\d+)\*\* — clean data", guards).group(1)),
        "max_irrelevant": int(re.search(r"\*\*≤ (\d+)\*\*\s*\|?\s*$", guards, re.M).group(1)),
    }
    out["wide-assay"] = dict(clean)
    out["longitudinal"] = dict(clean)
    out["floors"] = {"findings_driven_messy": 0.5, "deferral_closes": 1.0}
    return out


# ── running Guided over the exploration window ───────────────────────────

def _run_guided(dataset: str, csv_path: pathlib.Path, target: str) -> Measurement:
    """Drive the Router across the exploration steps and record what it asked.

    The window is the baseline's window: the exploration phase. Steps `data` and
    `explore`, which is what `pages/01`–`02` cover in Classic.
    """
    df = engine.read_table(csv_path.read_bytes(), csv_path.name)
    # With the target, as `api.py::_recompute` does once one is chosen. Without
    # it the harness would measure a door that no longer exists: the outcome
    # column would be asked how to read it rather than which level is the event.
    findings = engine.rank_findings(engine.diagnose(df, target=target), None)
    detection = engine.detect_task_type(df, target)
    required = required_decisions(findings, target_chosen=False)

    questions: list[QuestionRecord] = []
    deferrals: list[DeferralRecord] = []
    answered: list[str] = []
    deferred: dict[str, str] = {}

    # ── step 1 · data ────────────────────────────────────────────────────
    at_data = router.plan(findings, target=target, detection=detection, step="data")
    router.audit(at_data)          # a rule violation is a failure, not a score

    # Defer exactly one repair, to exercise the promise deferral makes. The
    # lowest-ranked one, because deferring the most severe finding would be a
    # strange thing for an interview to suggest.
    repairs = [q for q in at_data if q.kind == "repair"]
    to_defer = repairs[-1].key if len(repairs) > 1 else None
    if to_defer:
        deferred[to_defer] = "explore"

    at_data = router.plan(findings, target=target, detection=detection,
                          step="data", deferred=deferred)
    router.audit(at_data)

    for q in at_data:
        questions.append(_record(q, dataset, required))
        if q.status == "asked":
            answered.append(q.key)
        elif q.status == "deferred":
            deferrals.append(DeferralRecord(
                finding_id=q.triggering_finding or q.key,
                deferred_at="data", target_step=q.defer_target))

    # ── step 2 · explore, where the deferral must come back ──────────────
    # The palette ships with the Explore step, so it is measured from here on.
    # If the thresholds moved when it landed, they were counting offers as
    # questions and "push the notable, pull the rest" would read as regression.
    recommendations, signals = [], None
    try:
        from ml.eda_recommender import compute_dataset_signals, recommend_eda
        signals = compute_dataset_signals(
            df, target, detection.get("detected"), "cross_sectional", None)
        recommendations = recommend_eda(signals)
    except Exception:
        recommendations, signals = [], None

    at_explore = router.plan(findings, target=target, detection=detection,
                             step="explore", deferred=deferred, answered=answered,
                             recommendations=recommendations, signals=signals)
    router.audit(at_explore)

    for q in at_explore:
        questions.append(_record(q, dataset, required))
        if q.status == "asked" and q.deferred_from:
            for d in deferrals:
                if (d.finding_id == (q.triggering_finding or q.key)
                        and d.resurfaced_at is None):
                    d.resurfaced_at = "explore"

    return Measurement(
        door="guided", dataset=dataset,
        n_rows=len(df), n_columns=len(df.columns),
        required=required, questions=questions, deferrals=deferrals,
        notes=["Scored on the baseline's window: the exploration phase, steps "
               "data and explore.",
               "Every plan passed ml.router.audit before being scored, so a "
               "Decision B violation fails rather than scoring.",
               "The Explore step's pull palette is present and counted "
               "separately: offers are reported, never scored."],
    )


def _record(q, dataset: str, required) -> QuestionRecord:
    """One Router question, in the harness's vocabulary.

    `covers` is set from the question's own key, which is the same key
    `required_decisions` builds — both derive from the engine's findings. The
    Router never imports the metric, so this is agreement, not circularity.
    """
    keys = {r.key for r in required}
    return QuestionRecord(
        key=q.key, label=q.title, door="guided", step=q.step,
        triggering_finding=q.triggering_finding,
        mode=q.mode,
        # The fourth origin, carried from the Router rather than inferred here.
        # A harness that decided for itself which questions are constitutional
        # would be marking its own homework.
        clause=q.clause,
        skipped=(q.status != "asked"),
        skip_reason=q.skip_reason or (
            f"deferred to {q.defer_target}" if q.status == "deferred" else None),
        covers=q.key if q.key in keys else None,
        # THE QUESTION SAYS WHAT IT SETTLES, and the harness reads it rather
        # than inferring from the key. `DRIVE-002`'s bulk repair question is
        # keyed `repair_bulk::<kind>` and settles the N `repair::<id>`
        # requirements it groups; an exact-key matcher sees none of them, so
        # nine questions becoming one read as coverage falling to 0.4.
        #
        # This is not circularity. `covers` is the Router declaring which
        # findings a control has taken over — the same list the interface uses
        # to stop rendering them twice — and it is checked against
        # `required_decisions`, which is built from the engine's findings and
        # not from anything the Router said.
        #
        # `covers` holds FINDING IDS, because that is what the interface needs
        # to stop rendering a finding twice. `required_decisions` keys a repair
        # as `repair::<id>`. The prefix is added here rather than stored twice:
        # a Router that carried the harness's key naming would be the app
        # shaped to its own metric.
        also_covers=tuple(k for k in (f"repair::{i}" for i in (q.covers or []))
                          if k in keys))


def _surfaced_coverage(m: Measurement) -> float:
    """Asked OR visibly deferred, which is what the prereg counts."""
    if not m.required:
        return float("nan")
    surfaced = {k for q in m.questions for k in q.covered_keys}
    return sum(1 for r in m.required if r.key in surfaced) / len(m.required)


def _asked_coverage(m: Measurement) -> float:
    if not m.required:
        return float("nan")
    asked = {k for q in m.questions if not q.skipped for k in q.covered_keys}
    return sum(1 for r in m.required if r.key in asked) / len(m.required)


# ── the check ────────────────────────────────────────────────────────────

def test_the_prereg_predates_the_router():
    """The discipline, enforced: criteria set after a result get fitted to it."""
    import subprocess

    def _first_commit(path):
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%ct", "--", path],
            cwd=ROOT, capture_output=True, text=True).stdout.split()
        return int(out[-1]) if out else None

    prereg_at = _first_commit("docs/turbotab/VALUE_CHECK_PREREG.md")
    router_at = _first_commit("ml/router.py")
    if prereg_at is None or router_at is None:
        pytest.skip("one of the files is not committed yet")
    assert prereg_at <= router_at, (
        "the pre-registration was committed after the Router — the thresholds "
        "could have been fitted to the result")


def test_the_amendment_changed_no_verdict():
    """Prereg Amendment 1 (L26), and the only evidence that makes it honest.

    The ceiling moved from `irrelevant_questions` to `irrelevant_net`. That is
    a substitution a reader has every reason to be suspicious of, because it is
    **indistinguishable in the diff** from choosing the metric that makes a
    failing run pass.

    What distinguishes it is the timing, and the timing is checkable: on the
    run that lands the amendment, every dataset passes under **both** readings.
    An amendment that changes no verdict is an amendment nobody made to change
    a verdict.

    This test is therefore not a regression guard. It is the amendment's
    receipt, and the day it goes red the amendment has started doing work — at
    which point the honest move is to say so out loud in
    `VALUE_CHECK_ADJUDICATION.md`, not to delete this.
    """
    thresholds = _prereg_thresholds()
    both = {}
    for name, path, target in DATASETS:
        gm = _run_guided(name, path, target).to_dict()["metrics"]
        ceiling = thresholds[name]["max_irrelevant"]
        both[name] = {
            "literal": gm["irrelevant_questions"],
            "net": gm["irrelevant_net"],
            "constitutional": gm["constitutional"],
            "ceiling": ceiling,
        }
    failed_old = [n for n, r in both.items() if r["literal"] > r["ceiling"]]
    failed_new = [n for n, r in both.items() if r["net"] > r["ceiling"]]
    assert not failed_old, (
        f"the amendment was made on a run that FAILS the old reading on "
        f"{failed_old} — which is the case it was written to be "
        f"distinguishable from. Record it in VALUE_CHECK_ADJUDICATION.md as a "
        f"gate that moved under pressure, or revert the amendment.\n{both}")
    assert not failed_new, both
    # And the constitutional count is what accounts for the gap, rather than
    # the two metrics happening to agree for some other reason.
    assert any(r["constitutional"] > 0 for r in both.values()), (
        "no dataset has a constitutional question, so the two readings agree "
        "for a reason that has nothing to do with the amendment")


def test_the_literal_count_is_still_reported_everywhere():
    """A substitution nobody can see is a substitution nobody can audit.

    `irrelevant_questions` is demoted from *gated* to *reported*, and reported
    means it appears in every row of the result file — not that it survives in
    a docstring.
    """
    # `AUDIT-039`. THE FILE IS COMMITTED — `git ls-files` resolves it — so the
    # skip could never fire for a fixture reason and could only fire if the
    # recorded result were DELETED, which is the drift this file exists to
    # catch. A skip counts as not-a-failure, so the guard would have gone quiet
    # exactly when the record went missing.
    assert RESULT.exists(), (
        f"{RESULT} is gone. It is a tracked file and it is the whole subject "
        f"of this test; its absence is the finding, not a reason to stand down")
    recorded = json.loads(RESULT.read_text(encoding="utf-8"))
    assert recorded.get("rows"), "the recorded result has no rows to check"
    for row in recorded["rows"]:
        guided = row["guided"]
        assert "irrelevant_questions" in guided, row["dataset"]
        assert "irrelevant_net" in guided, row["dataset"]
        assert "constitutional" in guided, row["dataset"]


def test_routing_value_check():
    """The verdict. Reports the full table, then asserts every threshold."""
    thresholds = _prereg_thresholds()
    baseline = {m["dataset"]: m for m in measure.read_baseline(ADJUDICATED)}
    frozen = {m["dataset"]: m for m in measure.read_baseline(BASELINE)}

    rows, failures, strict_failures = [], [], []
    for name, path, target in DATASETS:
        g = _run_guided(name, path, target)
        c = baseline[name]["metrics"]
        gm = g.to_dict()["metrics"]
        surfaced = _surfaced_coverage(g)
        asked_cov = _asked_coverage(g)

        # The same Guided run, scored against the pre-registration's original
        # denominator. Both readings are reported for the same reason the
        # deferral ambiguity's were: a threshold met under one denominator and
        # missed under the other is a result, not a detail.
        frozen_keys = [r["key"] for r in frozen[name]["required"]]
        raised = {k for q in g.questions for k in q.covered_keys}
        raised_asked = {k for q in g.questions if not q.skipped
                        for k in q.covered_keys}
        surfaced_frozen = sum(1 for k in frozen_keys if k in raised) / len(frozen_keys)
        asked_frozen = sum(1 for k in frozen_keys if k in raised_asked) / len(frozen_keys)

        # Coverage never travels as a bare ratio (VALUE_CHECK_ADJUDICATION.md
        # §"Coverage carries its denominator"). Classic's numerator is
        # structurally frozen, so a ratio quoted alone rises on its own.
        n_now = len(g.required)
        n_frozen = len(frozen_keys)
        rows.append({
            "dataset": name,
            "classic": {**c,
                        "coverage_ratio": _ratio_from(baseline[name], ADJUDICATED_AT)},
            "classic_frozen": {**frozen[name]["metrics"],
                               "coverage_ratio": _ratio_from(frozen[name], PREREG_AT)},
            "guided": {**gm, "surfaced_coverage": round(surfaced, 4),
                       "asked_coverage": round(asked_cov, 4),
                       "surfaced_ratio": f"{round(surfaced * n_now)}/{n_now} @ {ADJUDICATED_AT}",
                       "asked_ratio": f"{round(asked_cov * n_now)}/{n_now} @ {ADJUDICATED_AT}"},
            "guided_under_frozen_denominator": {
                "n_required": n_frozen,
                "surfaced_coverage": round(surfaced_frozen, 4),
                "asked_coverage": round(asked_frozen, 4),
                "surfaced_ratio": f"{round(surfaced_frozen * n_frozen)}/{n_frozen} @ {PREREG_AT}",
                "asked_ratio": f"{round(asked_frozen * n_frozen)}/{n_frozen} @ {PREREG_AT}",
            },
        })

        t = thresholds[name]
        if name == "messy-clinic":
            _check(failures, name, "surfaced coverage",
                   surfaced, ">=", t["surfaced_coverage"])
            _check(failures, name, "asked coverage",
                   asked_cov, ">=", t["asked_coverage"])
            _check(failures, name, "findings-driven",
                   gm["findings_driven"], ">=",
                   thresholds["floors"]["findings_driven_messy"])
        else:
            _check(failures, name, "coverage", gm["coverage"], ">=", t["coverage"])
        _check(failures, name, "questions asked",
               gm["questions_asked"], "<=", t["max_questions"])
        # ON `irrelevant_net`, SINCE PREREG AMENDMENT 1 (L26). The ceiling value
        # is unchanged — ≤ 4 on messy-clinic, ≤ 3 on each guard, read out of the
        # same table as before. What moved is which metric it is applied to.
        #
        # The prereg defines an irrelevant question as one absent from the
        # decision inventory and citing no finding, and both conjuncts hold for
        # a CONSTITUTIONAL question by construction. With two of those the
        # literal count was a fair proxy; with four it had stopped measuring
        # "questions the dataset did not call for" and started measuring "how
        # much of the constitution is explicit" — every increment came from the
        # app being MORE honest about what it must ask.
        #
        # THE TIMING IS THE JUSTIFICATION, and it is asserted below rather than
        # asserted here: this was amended on a run that passes under BOTH
        # readings, which is the only thing separating it from choosing the
        # metric that makes a failing run pass. See VALUE_CHECK_PREREG.md
        # "Amendment 1" and `test_the_amendment_changed_no_verdict`.
        _check(failures, name, "irrelevant questions (net of constitutional)",
               gm["irrelevant_net"], "<=", t["max_irrelevant"])
        # REPORTED, NEVER INSTEAD OF. The literal count stays in every row of
        # the result file and in the printed table, so the substitution is
        # visible in the output rather than buried in this function. A
        # substitution nobody can see is a substitution nobody can audit.

        # ── deferral closure, and the one ambiguity in the prereg ──────────
        #
        # The prereg says: "This is a design promise, not a score: exactly 1.0.
        # A single deferred item that fails to resurface at a step that can act
        # on it is a bug that fails the check outright."
        #
        # On wide-assay and longitudinal there is nothing to defer — one
        # required decision each, zero repairable findings — so the metric is
        # None. Two readings:
        #
        #   (A) literal: the number must be 1.0 on every dataset, so None fails.
        #   (B) as explained: of the deferrals that occur, all must close;
        #       none occurring is not "an item that failed to resurface".
        #
        # This asserts (B), for three reasons written down rather than assumed:
        # the prereg's own explanatory sentence defines the failure as a
        # *dropped* deferral; requiring one on clean data would manufacture the
        # ceremony the clean-dataset guard forbids in the same section; and the
        # prereg calls Classic's NaN "correctly" recorded, treating
        # not-applicable as legitimate rather than as a miss.
        #
        # The determination was made after seeing the result, which is the
        # hazard pre-registration exists to prevent — so BOTH verdicts are
        # computed, reported and written to the result file, and the prereg is
        # not edited.
        dc = gm["deferral_closes"]
        deferrals_possible = any(r.key.startswith("repair::") for r in g.required) \
            and len([r for r in g.required if r.key.startswith("repair::")]) > 1
        if dc is not None:
            if abs(dc - 1.0) > 1e-9:
                failures.append(
                    f"{name}: deferral_closes = {dc}, must be exactly 1.0 — a "
                    "deferred item did not resurface where it was sent")
        elif deferrals_possible:
            failures.append(
                f"{name}: deferrals were possible but none occurred, so the "
                "promise is untested on this dataset")
        else:
            strict_failures.append(
                f"{name}: deferral_closes = None (nothing deferrable on this "
                "dataset; fails only under the literal reading of the prereg)")

    # The verdict this run computed. NOT written to the recorded result: the
    # same split the baseline got (T0-PREREG-002, T0-PREREG-003). A file called
    # a permanent record and rewritten on every run is only as permanent as the
    # code that computes it, so the suite COMPARES and
    # `scripts/rerecord_routing_value_check.py` re-records with provenance.
    # `VALUE_CHECK_ADJUDICATION.md` is the authority; the JSON is evidence.
    verdict = {
        "passes": not failures,
        "failures": failures,
        # Recorded so the user can overrule the reading rather than
        # discovering it was made silently.
        "literal_reading_only_failures": strict_failures,
        "passes_under_literal_reading": not (failures or strict_failures),
    }
    _assert_matches_the_recorded_verdict(verdict, rows)

    print("\n" + "=" * 86)
    print(f"{'dataset':<15}{'door':<9}{'asked':>7}{'irrel':>7}"
          f"{'find-drv':>10}{'coverage (k/n @ commit)':>38}")
    print("-" * 86)
    for r in rows:
        c, g = r["classic"], r["guided"]
        print(f"{r['dataset']:<15}{'classic':<9}"
              f"{c['questions_asked']:>7}{c['irrelevant_questions']:>7}"
              f"{c['findings_driven']:>10}{c['coverage_ratio']:>38}")
        print(f"{'':<15}{'guided':<9}"
              f"{g['questions_asked']:>7}{g['irrelevant_questions']:>7}"
              f"{g['findings_driven']:>10}{g['surfaced_ratio']:>38}")
        print(f"{'':<15}{'  pinned:':<9}{'':>7}{'':>7}{'':>10}"
              f"{r['classic_frozen']['coverage_ratio']:>38}")
        print(f"{'':<15}{'':<9}{'':>7}{'':>7}{'':>10}"
              f"{r['guided_under_frozen_denominator']['surfaced_ratio']:>38}")
    print("=" * 86)
    print("Classic's numerator cannot grow: the import path it renders is frozen,")
    print("so it cannot learn any detector the engine gains. A widening gap is not,")
    print("by itself, evidence of better routing. See VALUE_CHECK_ADJUDICATION.md.")

    assert not failures, (
        "ROUTING VALUE CHECK FAILED — write docs/turbotab/BLOCKED.md with these "
        "numbers and stop before L9:\n  " + "\n  ".join(failures))


RESULT_DRIFT_MESSAGE = (
    "The recorded value-check result no longer matches what this run computes. "
    "Do not fix this by re-recording. Record the new result beside the old one "
    "and adjudicate — docs/turbotab/VALUE_CHECK_ADJUDICATION.md is the "
    "authority, and this file is its evidence."
)


def _assert_matches_the_recorded_verdict(verdict, rows) -> None:
    """Compare this run against the recorded result. Never overwrite it.

    `T0-PREREG-001`'s note calls `passes_under_literal_reading: false` a
    permanent record. It was recomputed on every suite run, so it was permanent
    only for as long as the code that computed it kept computing it — the same
    defect as the baseline, one file over (`T0-PREREG-003`).

    The comparison is on the verdict and on each dataset's scored metrics.
    Anything else in the file is envelope.
    """
    if not RESULT.exists():
        pytest.skip("no recorded result yet — run scripts/rerecord_routing_value_check.py")
    recorded = json.loads(RESULT.read_text(encoding="utf-8"))

    was, now = recorded.get("verdict", {}), verdict
    drift = [f"verdict.{k}: recorded {was.get(k)!r} → now {now[k]!r}"
             for k in ("passes", "passes_under_literal_reading")
             if was.get(k) != now[k]]

    by_dataset = {r["dataset"]: r for r in recorded.get("rows", [])}
    for row in rows:
        old = by_dataset.get(row["dataset"])
        if old is None:
            drift.append(f"{row['dataset']}: absent from the recorded result")
            continue
        for door in ("classic", "guided"):
            for metric in ("questions_asked", "irrelevant_questions",
                           "findings_driven", "coverage"):
                a = (old.get(door) or {}).get(metric)
                b = (row.get(door) or {}).get(metric)
                if a != b:
                    drift.append(f"{row['dataset']}.{door}.{metric}: "
                                 f"recorded {a!r} → now {b!r}")

    assert not drift, RESULT_DRIFT_MESSAGE + "\n  " + "\n  ".join(drift)


def _check(failures, dataset, label, value, op, threshold):
    if value is None:
        failures.append(f"{dataset}: {label} is unavailable")
        return
    ok = value >= threshold - 1e-9 if op == ">=" else value <= threshold + 1e-9
    if not ok:
        failures.append(f"{dataset}: {label} = {value}, needs {op} {threshold}")
