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

Run:  venv/bin/python -m pytest tests/integration/test_routing_baseline.py -v
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
#
# **`L66`: the fifth reading, and the first where CLASSIC moved rather than the
# measuring stick.** `5acd7cd` merged `main` into `TurboTab`, and main's
# `7480564` ("Keep the five diagnostics nothing else computes, and drop the
# eleven that repeat") dropped nine deep-dive buttons from `pages/02_EDA.py`
# while main's cluster explorer and page-01 confirmation checkbox added three.
# Every previous movement here was the engine finding a decision it had been
# missing; this one is Classic's own UI. Ruled in
# `VALUE_CHECK_ADJUDICATION.md` §"L66 · Classic moved, and this time it was not
# the measuring stick", enumerated key by key, and accepted.
ADJUDICATED = ROOT / "docs" / "turbotab" / "data" / "routing-baseline-l66.json"
# The measurements this one superseded, oldest first. Kept named so the chain of
# re-measurements is readable rather than implied by filenames —
# l9 → l9c → l61 → l66. A list rather than one constant per link because the
# chain is now four long and `_PRIOR_PRIOR_PRIOR` is not a name.
ADJUDICATED_SUPERSEDED = [
    ROOT / "docs" / "turbotab" / "data" / "routing-baseline-l9.json",
    ROOT / "docs" / "turbotab" / "data" / "routing-baseline-l9c.json",
    ROOT / "docs" / "turbotab" / "data" / "routing-baseline-l61.json",
]

# What the adjudication permits the two references to disagree about, and by
# how much. Anything else is drift.
#
# **`TEST-086`, `L61`. The wide-assay entries are new and they are the third
# time this denominator has moved for the same reason.** `L60-A` made the
# target's event question dtype-agnostic, so `wide_assay.csv`'s `responder` —
# `int64` `{0,1}`, and the one dataset the L9 and L9c movements did NOT touch,
# because its outcome was already numeric — now requires the decision too.
# Classic does not ask it (`target-positive-class` is `guided-only` in the
# register), so Classic covers one of two instead of one of one.
#
# Extending this table in the same loop as the change that pressured it is
# `LOOP.md` §06.2, and the exception is invoked deliberately in
# `VALUE_CHECK_ADJUDICATION.md` §"The denominator moved a third time". The
# short form: these entries encode the SAME PURPOSE as the six above rather
# than a nudged value — same cause, same deltas `(1,2)`, `(30,29)`,
# `(1.0,0.5)`, and no assertion anywhere is relaxed.
# **`L66` adds the `questions_asked` rows and moves the `irrelevant_questions`
# ones.** These are the FIRST entries in this table that are not denominator
# movements: nine deep-dive buttons left `pages/02_EDA.py` and three widgets
# arrived, so Classic asks fewer questions and fewer irrelevant ones. Nothing
# else moved — `required_decisions`, `coverage` and `findings_driven` are
# identical between `l61` and `l66` on every dataset, and so are the
# required-decision inventories key for key, which is why `ADJUDICATED_KEY_DELTAS`
# below is unchanged. The net is −6 on three datasets and −5 on longitudinal;
# the per-key enumeration is in the adjudication note, not summarized here.
ADJUDICATED_DELTAS = {
    ("messy-clinic", "required_decisions"): (9, 10),
    ("messy-clinic", "questions_asked"): (34, 28),         # L66
    ("messy-clinic", "irrelevant_questions"): (25, 18),    # L66 (was (25, 24))
    ("messy-clinic", "coverage"): (0.1111, 0.1),
    ("longitudinal", "required_decisions"): (1, 2),
    ("longitudinal", "questions_asked"): (32, 27),         # L66
    ("longitudinal", "irrelevant_questions"): (31, 25),    # L66 (was (31, 30))
    ("longitudinal", "coverage"): (1.0, 0.5),
    ("wide-assay", "required_decisions"): (1, 2),          # L61
    ("wide-assay", "questions_asked"): (31, 25),           # L66
    ("wide-assay", "irrelevant_questions"): (30, 23),      # L66 (was (30, 29))
    ("wide-assay", "coverage"): (1.0, 0.5),                # L61
}

# The inventory keys the adjudication permits to differ from the frozen one.
# A *composition* change moves no metric — L9c swapped
# `repair::binary_text__outcome` for `repair::positive_class__outcome` and every
# number stayed identical — so comparing metrics alone cannot see it. That is
# the hole this table closes.
#
# **The added key is `__responder`, not `__outcome`.** The subject is the
# TARGET COLUMN's name, and messy-clinic and longitudinal both happen to call
# theirs `outcome` while wide-assay's is `responder` and leaky-sepsis's is
# `sepsis`. Written out per dataset rather than derived, because a table that
# computed the key from the target would agree with the engine by construction
# and could not notice the subject changing.
ADJUDICATED_KEY_DELTAS = {
    "messy-clinic": {"added": {"repair::positive_class__outcome"},
                     "removed": set()},
    "wide-assay": {"added": {"repair::positive_class__responder"},   # L61
                   "removed": set()},
    "longitudinal": {"added": {"repair::positive_class__outcome"},
                     "removed": set()},
}

# ── the leaky dataset's chain, which did not have one ───────────────────────
#
# **`TEST-086`'s second half, and the two are different mechanisms.** The three
# datasets above are compared against `ADJUDICATED` and every permitted
# difference is enumerated. `leaky-sepsis` was compared against its frozen file
# DIRECTLY, with no adjudicated reference and no deltas table — so the only
# ways to absorb a ruled movement were to edit the frozen artifact or to
# hand-write a replacement, and both are what this file exists to prevent.
#
# It gets the same shape now: a new measurement beside the frozen one, the
# frozen one still guarded, and every difference between them enumerated. The
# property that matters is the one the three-dataset half already has — *a
# second drift cannot hide inside the first* — and it is worth saying plainly
# that the leaky half did not have it before `L61` and nothing said so.
LEAKY_ADJUDICATED = (ROOT / "docs" / "turbotab" / "data"
                     / "routing-baseline-leaky-l66.json")
LEAKY_SUPERSEDED = [ROOT / "docs" / "turbotab" / "data"
                    / "routing-baseline-leaky-l61.json"]

# L66 moves this dataset for the same cause as the three above — main's
# diagnostics dedup — and it loses one key the others do not: the
# recommendation panel's `rec_run_leakage_scan`, whose label was the bare word
# "Run". `test_classic_does_not_ask_about_the_leak` below is the test that
# feels that one.
LEAKY_DELTAS = {
    ("leaky-sepsis", "required_decisions"): (1, 2),
    ("leaky-sepsis", "questions_asked"): (31, 25),         # L66
    ("leaky-sepsis", "irrelevant_questions"): (30, 23),    # L66 (was (30, 29))
    ("leaky-sepsis", "coverage"): (1.0, 0.5),
}

LEAKY_KEY_DELTAS = {
    "leaky-sepsis": {"added": {"repair::positive_class__sepsis"},
                     "removed": set()},
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

    # Diagnosed WITH the target, because the engine's answer depends on it: the
    # outcome column is asked "which level is the event", a feature is asked how
    # to read it. Both doors are scored against this one inventory, so it has to
    # be the inventory the engine actually produces for a dataset whose target
    # is known — which every fixture's is.
    findings = engine.rank_findings(engine.diagnose(df, target=target), None)
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

    # And the ruling's own cause, asserted rather than described. Composition,
    # not just size: L9c swapped one required decision for another without
    # moving a single metric, so a size-only check would have missed it.
    for name in frozen:
        was = {r["key"] for r in frozen[name]["required"]}
        is_ = {r["key"] for r in now[name]["required"]}
        ruled = ADJUDICATED_KEY_DELTAS[name]
        assert is_ - was == ruled["added"], (
            f"{name}: the inventory gained {sorted(is_ - was)}; the "
            f"adjudication accounts for {sorted(ruled['added'])}. "
            + DRIFT_MESSAGE)
        assert was - is_ == ruled["removed"], (
            f"{name}: the inventory lost {sorted(was - is_)}, which the "
            "adjudication does not account for. " + DRIFT_MESSAGE)


def test_both_doors_are_scored_against_one_inventory():
    """The harness's core promise, asserted rather than assumed.

    `required_decisions` is derived from the engine so neither door's UI biases
    the measuring stick — which only holds if both doors are handed the same
    inventory. L9c made the engine's answer depend on whether a target is known
    (the outcome is asked *which level is the event*, a feature is asked how to
    read it), so the two harnesses had to start diagnosing the same way. If they
    drift apart the doors are scored against different denominators and the
    comparison is meaningless while still producing numbers.
    """
    from tests.integration.test_routing_value_check import _run_guided

    for name, path, target in DATASETS:
        classic = [r.key for r in _measure_classic(name, path, target).required]
        guided = [r.key for r in _run_guided(name, path, target).required]
        assert classic == guided, (
            f"{name}: the doors are scored against different inventories.\n"
            f"  classic only: {sorted(set(classic) - set(guided))}\n"
            f"  guided only:  {sorted(set(guided) - set(classic))}")


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

    # Compared against the ADJUDICATED reference, not the frozen one — the same
    # arrangement the three datasets above have had since L9, and the one this
    # half was missing. The frozen file is still guarded, by
    # `test_the_leaky_reference_differs_from_the_frozen_one_only_as_ruled`.
    was = measure.read_baseline(LEAKY_ADJUDICATED)[0]["metrics"]
    now = m.to_dict()["metrics"]
    drift = [f"{name}.{k}: baseline {was[k]!r} → measured now {now[k]!r}"
             for k in _PREREG_METRICS if now[k] != was[k]]
    print(f"\n{name:<14} required={now['required_decisions']:>3} "
          f"asked={now['questions_asked']:>3} "
          f"irrelevant={now['irrelevant_questions']:>3} "
          f"coverage={now['coverage']}")
    assert not drift, DRIFT_MESSAGE + "\n  " + "\n  ".join(drift)


def test_the_leaky_reference_differs_from_the_frozen_one_only_as_ruled():
    """A second drift must not hide inside the first — on this dataset too.

    **The guard the leaky half never had.** `test_the_adjudicated_reference_
    differs_from_the_frozen_one_only_as_ruled` has protected the three
    pre-registered datasets since L9; `leaky-sepsis` was compared against its
    frozen file directly, so when `L60-A` moved its denominator there was no
    enumerated allowance to extend and nothing that could tell a ruled movement
    from a new one. This is that, in the same shape, including the inventory
    keys — because a composition change moves no metric and a size-only check
    cannot see it.
    """
    frozen = {m["dataset"]: m for m in measure.read_baseline(LEAKY_BASELINE)}
    now = {m["dataset"]: m for m in measure.read_baseline(LEAKY_ADJUDICATED)}
    assert set(frozen) == set(now)

    unruled = []
    for name in frozen:
        for metric in _PREREG_METRICS:
            was, is_ = frozen[name]["metrics"][metric], now[name]["metrics"][metric]
            if was == is_:
                continue
            ruled = LEAKY_DELTAS.get((name, metric))
            if ruled != (was, is_):
                unruled.append(f"{name}.{metric}: {was!r} → {is_!r} "
                               f"(adjudication says {ruled!r})")
    assert not unruled, (
        "the leaky reference has moved beyond what was ruled. "
        + DRIFT_MESSAGE + "\n  " + "\n  ".join(unruled))

    for name in frozen:
        was = {r["key"] for r in frozen[name]["required"]}
        is_ = {r["key"] for r in now[name]["required"]}
        ruled = LEAKY_KEY_DELTAS[name]
        assert is_ - was == ruled["added"], (
            f"{name}: the inventory gained {sorted(is_ - was)}; the "
            f"adjudication accounts for {sorted(ruled['added'])}. "
            + DRIFT_MESSAGE)
        assert was - is_ == ruled["removed"], (
            f"{name}: the inventory lost {sorted(was - is_)}, which the "
            "adjudication does not account for. " + DRIFT_MESSAGE)


def test_the_chain_of_re_measurements_is_readable_rather_than_implied():
    """Every superseded reading is still on disk and still named.

    `l9 → l9c → l61 → l66`, and the leaky file's own chain beside it. The rule this
    keeps is the one `VALUE_CHECK_ADJUDICATION.md` sets: *the frozen artifact is
    not edited, both readings are preserved in data, the ruling is published.*
    A chain implied by filenames alone is one a later loop can break by writing
    a fourth file and pointing the constant at it.
    """
    import json

    # TWO chains, checked separately. They are separate artifacts with
    # separate frozen files, and L61 re-measured both AT THE SAME COMMIT — so
    # a global uniqueness check would report a collision that is not one. The
    # first draft of this test did exactly that and this comment is why it
    # does not any more.
    chains = {
        "pre-registered": [BASELINE, *ADJUDICATED_SUPERSEDED, ADJUDICATED],
        "leaky": [LEAKY_BASELINE, *LEAKY_SUPERSEDED, LEAKY_ADJUDICATED],
    }
    for label, chain in chains.items():
        for path in chain:
            assert path.exists(), (
                f"{path.name} is gone; the {label} chain is not readable")
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload.get("measurements"), f"{path.name} holds no measurement"
            assert payload.get("measured_at"), (
                f"{path.name} does not say which commit it was measured at, so "
                f"it cannot take its place in the chain")

        stamps = [measure.baseline_provenance(p)["measured_at"] for p in chain]
        assert len(set(stamps)) == len(stamps), (
            f"two readings in the {label} chain claim the same commit: "
            f"{stamps}. A re-measurement that reuses its predecessor's stamp "
            f"is indistinguishable from an edit of the predecessor.")


def _classic_eda_ledger(csv_path: pathlib.Path, target: str):
    """The insight ledger `pages/02_EDA.py` builds on Classic's own path.

    The positive control for the test below has to come from somewhere that is
    not the question list, because the question list is exactly what the test
    says is empty. This renders the same page `_classic_questions` renders, with
    the same seeding, and hands back what the page concluded.
    """
    from streamlit.testing.v1 import AppTest
    from tests.integration.conftest import inject_data_state

    df = pd.read_csv(csv_path)
    at = AppTest.from_file("pages/02_EDA.py")
    inject_data_state(at, df, target_col=target, task_type="classification")
    _seed_dataset_roster(at, df, csv_path.name)
    at.run(timeout=120)
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    return at.session_state["insight_ledger"]


def test_classic_does_not_ask_about_the_leak():
    """The gap, measured rather than asserted — and it widened at `L66`.

    Classic emits a `blocker`-severity insight for a leaking column and renders
    it in the coaching summary. What it never does is put a question. This
    records that, so closing the gap has a before.

    **What changed.** This test used to hold that Classic *does* offer "Run
    Leakage Detection" — a recommendation card, rendered as an action — and that
    the gap was the narrower, worse one of an offer that never names the column.
    Main's `7480564` delisted that button from `pages/02_EDA.py` along with ten
    other deep-dive diagnostics that re-rendered what the page already shows, so
    at `HEAD` **no widget on Classic's exploration path says the word leakage at
    all**. The gap is now plain absence, and this test measures the wider gap
    rather than being retired: the `T0-ROUTE-001` before it exists to record did
    not go away, it got worse.

    **The detection did not go with the button, and that is asserted here rather
    than assumed.** The automatic >0.95 feature-target scan
    (`ml/eda_recommender.py`, `pages/02_EDA.py`) still raises the blocker, still
    names `abx_escalation_score` in the coaching layer, and still gates sign-off.
    What was removed is a UI duplicate. If that positive control ever fails, the
    finding is a regression in the app — a silently skippable leakage scan is
    `MINE-004`'s subject — and not a stale expectation in this file.
    """
    _, path, target = LEAKY
    df, questions = _classic_questions(path, target, "leaky-sepsis")
    labels = " ".join(q.label.lower() for q in questions)

    # Positive control, on the surface that survived: the leak IS detected.
    ledger = _classic_eda_ledger(path, target)
    leak_insights = [i for i in ledger.insights if i.id.startswith("eda_leakage_")]
    assert leak_insights, (
        "Classic no longer detects the leak on leaky_sepsis.csv at all, so this "
        "test is measuring the absence of a question about a problem the app "
        "never found. That is an app regression, not a test expectation: see "
        f"MINE-004. insights: {[i.id for i in ledger.insights]}")
    assert any(i.severity == "blocker" for i in leak_insights), (
        f"the leakage finding is no longer a blocker: "
        f"{[(i.id, i.severity) for i in leak_insights]}")
    assert any("abx_escalation_score" in i.id for i in leak_insights), (
        f"the leakage blocker no longer names the leaking column: "
        f"{[i.id for i in leak_insights]}")

    # And the gap: it is told, never asked. Nothing on the path puts the leak to
    # the user as a decision — not by name, and since L66 not even generically.
    assert "leakage" not in labels, (
        "Classic gained a question that says 'leakage' — the T0-ROUTE-001 gap "
        "may be closing upstream, so re-check what this test measures. Before "
        f"L66 this was 'Run Leakage Detection', a deep-dive button. labels: {labels}")
    assert "abx_escalation_score" not in labels, (
        "Classic now names the leaking column in a question — the T0-ROUTE-001 "
        "gap may be closed upstream, so re-check what this test measures")

    # And it is an offer, not a blocker gate: nothing about leaving the step
    # with a leak unresolved appears anywhere.
    assert not any("acknowledge" in q.label.lower() or "limitation" in q.label.lower()
                   for q in questions), (
        "Classic gained an acknowledgment path for an unresolved blocker")
