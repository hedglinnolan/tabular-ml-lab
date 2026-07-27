"""Guard the frozen Classic baseline. This file no longer measures.

`ROADMAP.md`'s routing value check is the only evidence gate the Router gets,
because the Router is the one component with no existing implementation to be
equivalent to. A baseline measured after the Router is built is a baseline you
unconsciously fit to, so it was measured at `6bfe598` and written to
`docs/turbotab/data/routing-baseline.json` as data.

**Why this file changed at L9.** Its docstring used to say the baseline "is
written now, while the Router does not exist, so it cannot be fitted to." That
was true when written and became false the moment L8 landed, and nothing
announced the expiry: `test_measure_classic_and_commit_the_baseline` kept
calling `write_baseline`, so every suite run since has re-measured Classic with
the Router present and committed the new numbers over the reference. The L9
task-3 commit shipped one such re-measurement before it was caught.

A protection that depends on "X does not exist yet" expires the moment X
exists, and nothing will tell you. That rule is recorded in
`FEATURE_PARITY.md` beside the principle-locality corollary; this file is the
first place it is enforced.

So measurement and comparison no longer share a code path. This file
**compares**: it re-measures Classic in memory and asserts the numbers still
equal the frozen ones. Writing lives in
`scripts/remeasure_routing_baseline.py`, which the suite never calls and which
refuses to overwrite.

Classic's golden path through exploration is pages 01 and 02. A *question* is an
interactive decision point the user is actually presented with: a widget that
changes what happens next. Layout, captions and read-only displays are not
questions, and neither is a widget the page renders but never acts on.

Run:  turbotab/.venv/Scripts/python -m pytest tests/integration/test_routing_baseline.py -v
"""
from __future__ import annotations

import os
import pathlib

import pandas as pd
import pytest

from turbotab import engine, measure
from turbotab.measure import (DeferralRecord, Measurement, QuestionRecord,
                              required_decisions)

pytestmark = pytest.mark.timeout(900)

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "turbotab" / "sample_data"
BASELINE = ROOT / "docs" / "turbotab" / "data" / "routing-baseline.json"
# The leaky case gets its OWN baseline file. The three original baselines are
# frozen and every threshold in VALUE_CHECK_PREREG.md is banked against them;
# injecting a leak into one of those datasets would silently invalidate the lot.
LEAKY_BASELINE = ROOT / "docs" / "turbotab" / "data" / "routing-baseline-leaky.json"
# The reference Classic is compared against *today*. It is not the frozen file:
# at L9 the binary-text detector found a required decision the engine had been
# missing, so the ground-truth denominator moved while Classic's behavior did
# not. The frozen file stays frozen and this measurement sits beside it, which
# is what `VALUE_CHECK_ADJUDICATION.md` §"The denominator moved" rules. Every
# difference between the two is enumerated there and asserted below, so a
# second drift cannot hide inside the first.
ADJUDICATED = ROOT / "docs" / "turbotab" / "data" / "routing-baseline-l9.json"

# What the adjudication permits the two references to disagree about, and by
# how much. Anything else is drift.
ADJUDICATED_DELTAS = {
    ("messy-clinic", "required_decisions"): (9, 10),
    ("messy-clinic", "irrelevant_questions"): (25, 24),
    ("messy-clinic", "coverage"): (0.1111, 0.1),
    ("longitudinal", "required_decisions"): (1, 2),
    ("longitudinal", "irrelevant_questions"): (31, 30),
    ("longitudinal", "coverage"): (1.0, 0.5),
}

# Three shapes, as the roadmap asks. Multi-file stays frozen
# (`TRANSITION_PLAN.md` §05), so the third is the messy single file.
DATASETS = [
    ("messy-clinic", DATA / "clinic_visits.csv", "outcome"),
    ("wide-assay", DATA / "wide_assay.csv", "responder"),
    ("longitudinal", DATA / "longitudinal_visits.csv", "outcome"),
]

# Widgets that are not decisions about the analysis. Excluded by name so each
# exclusion is arguable rather than silent — a metric that quietly counts the
# "Save Progress" button as a question about the data is not measuring routing.
_NOT_A_QUESTION = (
    # chrome and navigation
    "workflow_mode_selector", "nav_prev", "nav_next", "save_progress",
    "clear_session", "confirm_clear", "modify_data", "change_merge",
    "builtin_select", "add_builtin",
    # the assistant, not the analysis
    "llm_", "ollama_", "anthropic_", "openai_",
    # display and paging
    "theme", "show_advanced", "page_size", "_page", "_pval", "_smd", "_miss",
)

_NOT_A_QUESTION_LABELS = ("save progress", "home", "eda →", "← ", " →")


def _is_question(widget) -> bool:
    key = (getattr(widget, "key", "") or "").lower()
    label = (getattr(widget, "label", "") or "").strip()
    if not label:
        return False
    if any(skip in key for skip in _NOT_A_QUESTION):
        return False
    low = label.lower()
    if any(skip in low for skip in _NOT_A_QUESTION_LABELS):
        return False
    # Download and expander toggles reveal, they do not decide.
    if key.startswith(("download", "expand", "show_", "toggle_view")):
        return False
    return True


def _seed_dataset_roster(at, df, filename):
    """Give page 01 the roster it stops without.

    `inject_data_state` seeds `raw_data`, which is what pages 02+ read — but
    page 01 stops at Step 1 unless the project has a registered dataset
    (`:559`). Without this the page never reaches the audit or the target
    selection, and a baseline built from that run would report Classic as asking
    none of the questions it actually asks. Measuring the wrong path in Guided's
    favor is precisely the bias this baseline exists to avoid.
    """
    at.session_state["sp_projects"] = {
        1: {"id": 1, "name": "baseline", "description": "",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "active": True,
            # The real store creates these alongside datasets; omitting one
            # raises a KeyError deep in the sidebar rather than here.
            "merge_configs": {},
            "datasets": {
                1: {"id": 1, "project_id": 1, "name": filename,
                    "filename": filename, "file_type": "csv",
                    "shape_rows": len(df), "shape_cols": len(df.columns),
                    "columns": [str(c) for c in df.columns],
                    "column_types": {str(c): str(t) for c, t in df.dtypes.items()},
                    "is_transposed": False,
                    "upload_timestamp": "2026-01-01T00:00:00+00:00"},
            }},
    }
    at.session_state["sp_counter_project"] = 1
    at.session_state["sp_counter_dataset"] = 1
    at.session_state["datasets_registry"] = {1: df}


def _map_to_requirement(label: str, key: str, required) -> str | None:
    """Which required decision, if any, this Classic widget settles.

    The two doors name their widgets differently, so coverage cannot be computed
    by comparing key strings — that scores every door zero against every
    requirement and makes the metric say nothing.

    Classic's mapping is small and legible: the target selectbox settles
    `choose_target`, and each "Apply: <fix>" button in Suggested Actions settles
    the repair whose engine-written `fix_label` it is showing.
    """
    if key == "target_selectbox":
        return "choose_target"
    if not label.startswith("Apply:"):
        return None
    body = label[len("Apply:"):].split(":")[0].strip().lower()
    for r in required:
        if r.key.startswith("repair::") and r.fix_label:
            if body and body in r.fix_label.lower():
                return r.key
    return None


def _classic_questions(csv_path: pathlib.Path, target: str, dataset: str):
    """Count interactive decision points on Classic's exploration path."""
    from streamlit.testing.v1 import AppTest
    from tests.integration.conftest import inject_data_state

    df = pd.read_csv(csv_path)
    questions = []

    for page, step in (("pages/01_Upload_and_Audit.py", "upload"),
                       ("pages/02_EDA.py", "explore")):
        at = AppTest.from_file(page)
        inject_data_state(at, df, target_col=target,
                          task_type="classification")
        _seed_dataset_roster(at, df, csv_path.name)
        at.run(timeout=120)
        assert not at.exception, [str(e.value)[:200] for e in at.exception]

        seen = set()
        for kind in ("selectbox", "multiselect", "radio", "checkbox",
                     "slider", "number_input", "text_input", "button"):
            for w in getattr(at, kind, []):
                if not _is_question(w):
                    continue
                key = getattr(w, "key", None) or f"{kind}:{w.label}"
                if key in seen:
                    continue
                seen.add(key)
                questions.append(QuestionRecord(
                    key=str(key), label=str(w.label), door="classic", step=step,
                    # Classic's widgets are laid out by pipeline stage. A
                    # question is findings-driven only if the page raised it
                    # *because of* a finding, which none of these do — the
                    # suggested-actions list is rendered, but the questions
                    # around it exist whether or not anything was found.
                    triggering_finding=None))
    return df, questions


def _measure_classic(dataset: str, csv_path: pathlib.Path, target: str) -> Measurement:
    df, questions = _classic_questions(csv_path, target, dataset)

    findings = engine.rank_findings(engine.diagnose(df), None)
    required = required_decisions(findings, target_chosen=False)

    for q in questions:
        q.covers = _map_to_requirement(q.label, q.key, required)

    return Measurement(
        door="classic", dataset=dataset,
        n_rows=len(df), n_columns=len(df.columns),
        required=required, questions=questions,
        # Classic has no deferral mechanism at all: a noticing is acted on now
        # or forgotten. Recorded as an empty list, which scores NaN rather than
        # 0.0 — no evidence, not a perfect or a failing score.
        deferrals=[],
        notes=[
            "Classic has no deferral disposition; deferral_closes is not "
            "measurable rather than zero.",
            "Questions counted are interactive widgets on pages 01-02 that "
            "change what happens next; navigation, session controls, LLM "
            "settings and paging are excluded by name.",
            # The scope limit, stated so the comparison is arguable. Guided must
            # be measured over the SAME window or the win is an artifact of
            # where the boundary was drawn.
            "SCOPE: the measured window is the exploration phase, pages 01-02. "
            "Classic surfaces the import doctor's findings in the audit display "
            "but offers one-click repair for only a subset; the rest are acted "
            "on at Preprocess (page 05), outside this window. Coverage below 1.0 "
            "therefore means 'not raised as an actionable decision during "
            "exploration', not 'never offered anywhere'. Guided is scored on the "
            "identical window.",
        ],
    )


# ── the harness itself, checked before it is trusted ─────────────────────

def test_required_decisions_come_from_the_engine_not_the_ui():
    """The denominator must not be biased toward the door being measured."""
    df = pd.read_csv(DATA / "clinic_visits.csv")
    findings = engine.rank_findings(engine.diagnose(df), None)
    req = required_decisions(findings, target_chosen=False)

    assert req[0].key == "choose_target"
    keys = {r.key for r in req}
    # Every repairable critical/warning finding is a decision the data forces.
    for f in findings:
        if f["severity"] in ("critical", "warning") and f.get("fix_label") \
                and f.get("fix_kind") not in (None, "none"):
            assert f"repair::{f['id']}" in keys, f"{f['id']} is not counted"
    # Findings the engine refuses to repair are reports, not questions.
    for f in findings:
        if f.get("fix_kind") in (None, "none"):
            assert f"repair::{f['id']}" not in keys


def test_only_high_confidence_findings_can_make_a_question_moot():
    """Decision B, encoded in the metric before the Router can use it."""
    df = pd.read_csv(DATA / "clinic_visits.csv")
    req = required_decisions(engine.rank_findings(engine.diagnose(df), None))
    rated = [r for r in req if r.confidence]
    assert rated, "no rated requirements to check"
    for r in rated:
        assert r.can_be_skipped == (r.confidence == "high")


def test_no_deferrals_scores_as_no_evidence_not_as_perfect():
    """A door that cannot defer must not win the deferral metric by default."""
    m = Measurement(door="classic", dataset="x", n_rows=1, n_columns=1)
    assert m.deferral_closes != m.deferral_closes          # NaN
    assert m.to_dict()["metrics"]["deferral_closes"] is None


def test_a_deferral_that_resurfaces_elsewhere_is_not_closed():
    """Coming back in the wrong room is not closure."""
    right = DeferralRecord("f1", "explore", "preprocess", resurfaced_at="preprocess")
    wrong = DeferralRecord("f2", "explore", "preprocess", resurfaced_at="report")
    never = DeferralRecord("f3", "explore", "preprocess")
    assert right.closed and not wrong.closed and not never.closed


def test_coverage_stops_a_door_winning_by_asking_nothing():
    """Fewer questions is only a win if the required ones were still asked."""
    from turbotab.measure import DecisionRequirement

    silent = Measurement(
        door="guided", dataset="x", n_rows=10, n_columns=3,
        required=[DecisionRequirement("choose_target", "needed", "critical"),
                  DecisionRequirement("repair::a", "needed", "warning")],
        questions=[])
    assert silent.questions_asked == 0
    assert silent.irrelevant_questions == 0        # flattering on its own…
    assert silent.coverage == 0.0                  # …and caught here


def test_a_skipped_question_is_not_an_asked_question():
    q_asked = QuestionRecord("a", "A", "guided", "explore")
    q_skipped = QuestionRecord("b", "B", "guided", "explore", skipped=True,
                               skip_reason="high-confidence finding made it moot")
    m = Measurement(door="guided", dataset="x", n_rows=1, n_columns=1,
                    questions=[q_asked, q_skipped])
    assert m.questions_asked == 1
    # But it still counts toward coverage: it was raised and then resolved.
    assert {q.key for q in m.questions} == {"a", "b"}


# ── the baseline ─────────────────────────────────────────────────────────

DRIFT_MESSAGE = (
    "Classic's measured behavior has drifted from the pre-registered baseline. "
    "Do not fix this by re-measuring. Record the new measurement beside the old "
    "one and adjudicate."
)

# The metrics the pre-registration's own table quotes. Drift in any of these
# moves the reference the whole check is banked against.
_PREREG_METRICS = ("required_decisions", "questions_asked",
                   "irrelevant_questions", "findings_driven", "coverage")


def _git_show(commit: str, path: str):
    """A file as of one commit, or None when git cannot answer."""
    import subprocess

    try:
        out = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8")
    except (OSError, ValueError):
        return None
    return out.stdout if out.returncode == 0 else None


def test_the_frozen_baseline_is_the_one_the_prereg_names():
    """A silent swap of the reference must fail here.

    Two checks, because a self-declared stamp proves nothing on its own:

    1. the file's `measured_at` equals the commit `VALUE_CHECK_PREREG.md`
       names — parsed out of the prereg rather than restated here, so the test
       cannot quietly disagree with the document that binds it;
    2. every metric the prereg's table quotes is byte-identical to the value at
       that commit, read back out of git. Schema may grow — `c8c5f51` added
       `pull_affordances` and `mode` without moving a number, and that is the
       only kind of change this artifact may take. A measurement may not move.
    """
    import json
    import re

    prereg = (ROOT / "docs" / "turbotab" / "VALUE_CHECK_PREREG.md").read_text(
        encoding="utf-8")
    named = re.search(r"measured at `([0-9a-f]{7,40})`", prereg)
    assert named, "the pre-registration no longer names the commit it was measured at"
    commit = named.group(1)

    stamp = measure.baseline_provenance(BASELINE)
    assert stamp["measured_at"] == commit, (
        f"the baseline is stamped {stamp['measured_at']!r} and the "
        f"pre-registration names {commit!r}. " + DRIFT_MESSAGE)

    raw = _git_show(commit, "docs/turbotab/data/routing-baseline.json")
    if raw is None:
        pytest.skip(f"git cannot read {commit}; provenance check needs history")

    frozen = {m["dataset"]: m for m in json.loads(raw)["measurements"]}
    current = {m["dataset"]: m for m in measure.read_baseline(BASELINE)}
    assert set(frozen) == set(current), (
        "the baseline's datasets changed. " + DRIFT_MESSAGE)

    for name, row in frozen.items():
        for metric in _PREREG_METRICS:
            assert current[name]["metrics"][metric] == row["metrics"][metric], (
                f"{name}.{metric}: the committed baseline says "
                f"{current[name]['metrics'][metric]!r} and {commit} measured "
                f"{row['metrics'][metric]!r}. " + DRIFT_MESSAGE)
        assert ([r["key"] for r in current[name]["required"]]
                == [r["key"] for r in row["required"]]), (
            f"{name}: the required-decision inventory in the frozen file no "
            f"longer matches {commit}. " + DRIFT_MESSAGE)


def test_the_adjudicated_reference_differs_from_the_frozen_one_only_as_ruled():
    """A second drift must not be able to hide inside the first.

    The L9 measurement differs from the frozen baseline in six numbers, all
    consequences of one cause: the binary-text detector added
    `repair::binary_text__outcome` to the ground-truth inventory on two of the
    three datasets. Each difference is enumerated in the adjudication note and
    here. Any seventh difference is new drift and fails.
    """
    frozen = {m["dataset"]: m for m in measure.read_baseline(BASELINE)}
    now = {m["dataset"]: m for m in measure.read_baseline(ADJUDICATED)}
    assert set(frozen) == set(now)

    unruled = []
    for name in frozen:
        for metric in _PREREG_METRICS:
            was, is_ = frozen[name]["metrics"][metric], now[name]["metrics"][metric]
            if was == is_:
                continue
            ruled = ADJUDICATED_DELTAS.get((name, metric))
            if ruled != (was, is_):
                unruled.append(f"{name}.{metric}: {was!r} → {is_!r} "
                               f"(adjudication says {ruled!r})")
    assert not unruled, (
        "the adjudicated reference has moved beyond what was ruled. "
        + DRIFT_MESSAGE + "\n  " + "\n  ".join(unruled))

    # And the ruling's own cause, asserted rather than described.
    for name in ("messy-clinic", "longitudinal"):
        added = ({r["key"] for r in now[name]["required"]}
                 - {r["key"] for r in frozen[name]["required"]})
        assert added == {"repair::binary_text__outcome"}, (
            f"{name}: the inventory grew by {sorted(added)}, not by the one "
            "decision the adjudication attributes it to")


def test_classic_still_measures_what_the_baseline_recorded():
    """Re-measure Classic in memory and compare. Nothing is written.

    This is the drift detector, not the measurement. When it fails, the finding
    is that Classic moved — which is a result to adjudicate, not a file to
    regenerate. `docs/turbotab/VALUE_CHECK_ADJUDICATION.md` holds the precedent
    for how: frozen artifact unmodified, both readings preserved in data, ruling
    published.

    Compared against the adjudicated reference, not the frozen one. The frozen
    one is guarded by `test_the_frozen_baseline_is_the_one_the_prereg_names`,
    and the two are held to their ruled difference by the test above.
    """
    frozen = {m["dataset"]: m for m in measure.read_baseline(ADJUDICATED)}
    drift = []

    for name, path, target in DATASETS:
        assert path.exists(), f"missing dataset {path}"
        m = _measure_classic(name, path, target)

        assert m.required_decisions > 0, f"{name}: nothing required — check the engine"
        assert m.questions_asked > 0, f"{name}: Classic presented no questions"

        now = m.to_dict()["metrics"]
        was = frozen[name]["metrics"]
        for metric in _PREREG_METRICS:
            if now[metric] != was[metric]:
                drift.append(f"{name}.{metric}: baseline {was[metric]!r} → "
                             f"measured now {now[metric]!r}")
        print(f"\n{name:<14} required={now['required_decisions']:>3} "
              f"asked={now['questions_asked']:>3} "
              f"irrelevant={now['irrelevant_questions']:>3} "
              f"findings_driven={now['findings_driven']} "
              f"coverage={now['coverage']}")

    assert not drift, DRIFT_MESSAGE + "\n  " + "\n  ".join(drift)


# ── the fourth dataset · T0-ROUTE-001 ────────────────────────────────────

LEAKY = ("leaky-sepsis", DATA / "leaky_sepsis.csv", "sepsis")


def test_classic_on_the_leaky_dataset_still_matches_its_baseline():
    """The same guard, on the fourth dataset. Nothing is written.

    `T0-ROUTE-001` added a dataset containing a column recorded after the
    outcome was known. Classic was measured on it before the Router learned to
    push blockers, so the leakage comparison cannot be fitted to either — and
    that protection has the same expiry as the main baseline's, so it gets the
    same treatment.

    It keeps its own file: the three original baselines are frozen, and every
    pre-registered threshold is banked against them.
    """
    name, path, target = LEAKY
    assert path.exists(), f"missing dataset {path}"

    m = _measure_classic(name, path, target)
    assert m.required_decisions > 0
    assert m.questions_asked > 0

    was = measure.read_baseline(LEAKY_BASELINE)[0]["metrics"]
    now = m.to_dict()["metrics"]
    drift = [f"{name}.{k}: baseline {was[k]!r} → measured now {now[k]!r}"
             for k in _PREREG_METRICS if now[k] != was[k]]
    print(f"\n{name:<14} required={now['required_decisions']:>3} "
          f"asked={now['questions_asked']:>3} "
          f"irrelevant={now['irrelevant_questions']:>3} "
          f"coverage={now['coverage']}")
    assert not drift, DRIFT_MESSAGE + "\n  " + "\n  ".join(drift)


def test_classic_does_not_ask_about_the_leak():
    """The gap, measured rather than asserted.

    Classic emits a `blocker`-severity insight for a leaking column and renders
    it in the coaching summary. What it never does is put a question. This
    records that, so closing the gap has a before.
    """
    _, path, target = LEAKY
    df, questions = _classic_questions(path, target, "leaky-sepsis")
    labels = " ".join(q.label.lower() for q in questions)

    # Classic DOES offer "Run leakage detection" — that is the recommendation
    # card, rendered as an action. The gap is narrower and worse than absence:
    # nothing ever names the column that is actually leaking, so the user has to
    # already suspect a leak to go looking for one.
    assert "leakage" in labels, (
        "Classic lost its leakage recommendation card — this test is now "
        "measuring something else")
    assert "abx_escalation_score" not in labels, (
        "Classic now names the leaking column in a question — the T0-ROUTE-001 "
        "gap may be closed upstream, so re-check what this test measures")

    # And it is an offer, not a blocker gate: nothing about leaving the step
    # with a leak unresolved appears anywhere.
    assert not any("acknowledge" in q.label.lower() or "limitation" in q.label.lower()
                   for q in questions), (
        "Classic gained an acknowledgment path for an unresolved blocker")
