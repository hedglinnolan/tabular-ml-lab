"""Measure Classic, and commit the baseline — before the Router exists.

`ROADMAP.md`'s routing value check is the only evidence gate the Router gets,
because the Router is the one component with no existing implementation to be
equivalent to. A baseline measured after the Router is built is a baseline you
unconsciously fit to, so this runs first and writes its numbers to
`docs/turbotab/data/routing-baseline.json` as data.

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

def test_measure_classic_and_commit_the_baseline():
    """Run Classic on all three datasets and write the numbers as data.

    This is the artifact the routing value check will judge against. It is
    written now, while the Router does not exist, so it cannot be fitted to.
    """
    measurements = []
    for name, path, target in DATASETS:
        assert path.exists(), f"missing dataset {path}"
        m = _measure_classic(name, path, target)

        assert m.required_decisions > 0, f"{name}: nothing required — check the engine"
        assert m.questions_asked > 0, f"{name}: Classic presented no questions"
        measurements.append(m)

    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    measure.write_baseline(BASELINE, measurements)

    written = measure.read_baseline(BASELINE)
    assert len(written) == 3
    for row in written:
        assert row["door"] == "classic"
        assert row["metrics"]["deferral_closes"] is None
        print(f"\n{row['dataset']:<14} required={row['metrics']['required_decisions']:>3} "
              f"asked={row['metrics']['questions_asked']:>3} "
              f"irrelevant={row['metrics']['irrelevant_questions']:>3} "
              f"findings_driven={row['metrics']['findings_driven']} "
              f"coverage={row['metrics']['coverage']}")
