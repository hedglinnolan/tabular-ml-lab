"""
The walking skeleton's test: a real CSV in, real findings out.

Written before `api.py`, against the engine and project directly, so that the
API is built to match something already known to work rather than the other way
round. API-level tests live at the bottom and assert the *same* findings arrive
over HTTP.

Run:  venv/bin/python -m pytest turbotab/test_skeleton.py -v
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from turbotab import engine
from turbotab.project import AnalysisProject, ProjectError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_CSV = Path(__file__).resolve().parent / "sample_data" / "clinic_visits.csv"
TARGET = "outcome"


@pytest.fixture(scope="module")
def raw() -> bytes:
    return DEMO_CSV.read_bytes()


@pytest.fixture(scope="module")
def df(raw: bytes) -> pd.DataFrame:
    return engine.read_table(raw, DEMO_CSV.name)


# ═══════════════════════════════════════════════════════════════════════════
# The riskiest assumption: the engine runs with no Streamlit in the process
# ═══════════════════════════════════════════════════════════════════════════

def test_engine_imports_and_runs_with_streamlit_blocked(tmp_path: Path):
    """`ARCHITECTURE.md` §01's claim, as an executable check.

    Two things make this non-vacuous, which matters because `TRANSITION_PLAN.md`
    §03 catalogues a test that passes by finding nothing:

    1. A stub `streamlit` module is put on the path first, so `streamlit` is
       genuinely importable. Without it, a machine that simply has no Streamlit
       installed would pass this test while proving nothing.
    2. The blocker is asserted to actually block *before* the engine is
       imported. The snippet printed in the architecture doc uses
       `find_module`/`load_module`, which the import system stopped consulting
       in Python 3.12 — run as written on a modern interpreter it blocks
       nothing at all.

    Then the real assertion: after importing and *running* the engine,
    `streamlit` is still absent from `sys.modules`.
    """
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "streamlit.py").write_text("MARKER = 'stub streamlit'\n")

    script = textwrap.dedent(f"""
        import importlib.util, json, sys
        sys.path.insert(0, {str(stub_dir)!r})
        sys.path.insert(0, {str(REPO_ROOT)!r})

        # (1) streamlit really is importable here, so blocking it means something.
        assert importlib.util.find_spec("streamlit") is not None, "stub not reachable"

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "streamlit" or name.startswith("streamlit."):
                    raise ImportError("BLOCKED: " + name)
                return None
        sys.meta_path.insert(0, Blocker())

        # (2) the blocker actually blocks.
        try:
            import streamlit
            raise SystemExit("blocker did not block")
        except ImportError as e:
            assert "BLOCKED" in str(e), e

        # (3) the engine imports and runs anyway.
        from turbotab import engine
        raw = open({str(DEMO_CSV)!r}, "rb").read()
        frame = engine.read_table(raw, "clinic_visits.csv")
        findings = engine.diagnose(frame)
        task = engine.detect_task_type(frame, {TARGET!r})
        prof = engine.profile(frame, {TARGET!r}, task["detected"])
        ranked = engine.rank_findings(findings, prof)

        assert "streamlit" not in sys.modules, "engine pulled streamlit in"
        print(json.dumps({{"n": len(ranked), "task": task["detected"]}}))
    """)
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["n"] > 0
    assert out["task"] == "classification"


# ═══════════════════════════════════════════════════════════════════════════
# Real CSV in, real findings out
# ═══════════════════════════════════════════════════════════════════════════

def test_demo_csv_reads_as_a_real_table(df: pd.DataFrame):
    assert len(df) == 140
    assert TARGET in df.columns


def test_findings_are_non_empty(df: pd.DataFrame):
    task = engine.detect_task_type(df, TARGET)
    prof = engine.profile(df, TARGET, task["detected"])
    ranked = engine.rank_findings(engine.diagnose(df), prof)
    assert len(ranked) > 0
    assert all(f["title"] for f in ranked), "a finding with no title says nothing"


def test_findings_match_a_direct_engine_call(df: pd.DataFrame):
    """The load-bearing assertion: the adapter reports the engine, verbatim.

    Compared field by field against `ml.import_doctor.diagnose` reached directly,
    so a future 'improvement' in `engine.py` that rewords or reorders a finding
    fails here.

    Two engine modules now feed this stream rather than one. `ml.binary_text`
    supersedes the doctor's numeric-coercion proposal on the columns it claims
    (GUIDED-001), so the assertion is: every doctor finding that was not
    superseded is reported verbatim, and every superseded one is replaced by a
    binary reading of the same column — never dropped, never shown twice.
    """
    from ml import binary_text, import_doctor      # the real thing, no adapter

    direct = import_doctor.diagnose(df)
    assert direct, "the fixture is supposed to be a messy file"

    binary = binary_text.detect_binary_text(df)
    claimed = {c for f in binary for c in f.affected_columns}
    superseded = [d for d in direct
                  if d.fix_kind == "coerce_numeric"
                  and any(str(c) in claimed for c in d.affected_columns)]
    survivors = [d for d in direct if d not in superseded]

    ranked = engine.rank_findings(engine.diagnose(df), None)
    structural = [f for f in ranked if f["source"] == "structure"]

    assert len(structural) == len(survivors) + len(binary)
    by_id = {f["id"]: f for f in structural}

    for d in survivors + binary:
        got = by_id[d.id]
        assert got["title"] == d.title
        assert got["detail"] == d.detail
        assert got["why_it_matters"] == d.why_it_matters
        assert got["severity"] == d.severity
        assert got["confidence"] == d.confidence
        assert got["fix_kind"] == d.fix_kind
        assert got["affected_columns"] == list(d.affected_columns)
        assert got["auto_suggestable"] is bool(d.auto_suggestable)

    for d in superseded:
        assert d.id not in by_id, (
            f"{d.id} was superseded by a binary reading and is still shown; two "
            "repair proposals for one column make the user settle the engine's "
            "own disagreement")
        assert any(str(c) in claimed for c in d.affected_columns)


def test_profile_matches_a_direct_engine_call(df: pd.DataFrame):
    from ml.dataset_profile import compute_dataset_profile

    direct = compute_dataset_profile(df, target_col=TARGET, task_type="classification")
    via = engine.profile_to_dict(engine.profile(df, TARGET, "classification"))
    assert via["n_rows"] == direct.n_rows
    assert via["n_features"] == direct.n_features
    assert via["data_sufficiency"] == direct.data_sufficiency.value
    assert len(via["warnings"]) == len(direct.warnings)
    assert via["target_profile"]["name"] == direct.target_profile.name


def test_task_type_matches_a_direct_engine_call(df: pd.DataFrame):
    from ml import triage
    assert engine.detect_task_type(df, TARGET) == triage.detect_task_type(df, TARGET)


def test_diagnosis_never_mutates_the_frame(df: pd.DataFrame):
    """`ARCHITECTURE.md` §02: diagnosis never mutates; fixes are explicit."""
    before = df.copy(deep=True)
    engine.diagnose(df)
    engine.profile(df, TARGET, "classification")
    pd.testing.assert_frame_equal(df, before)


# ═══════════════════════════════════════════════════════════════════════════
# The invariants the interface leans on
# ═══════════════════════════════════════════════════════════════════════════

def test_ranking_puts_critical_before_info(df: pd.DataFrame):
    prof = engine.profile(df, TARGET, "classification")
    ranked = engine.rank_findings(engine.diagnose(df), prof)
    order = [engine.SEVERITY_RANK[f["severity"]] for f in ranked]
    assert order == sorted(order), "findings are not in the engine's severity order"
    assert [f["rank"] for f in ranked] == list(range(len(ranked)))


def test_only_high_confidence_is_auto_suggestable(df: pd.DataFrame):
    """The governing rule: `high` is the only tier the UI may pre-select.

    `ARCHITECTURE.md` §02 and `PRODUCT_VISION.md` §07.1. Everything the frontend
    pre-checks reads this flag, so it is asserted at the source.
    """
    prof = engine.profile(df, TARGET, "classification")
    for f in engine.rank_findings(engine.diagnose(df), prof):
        if f["auto_suggestable"]:
            assert f["confidence"] == "high", f"{f['id']} pre-selects at {f['confidence']}"
    # Profile warnings carry no confidence at all, so none of them may pre-select.
    assert not any(f["auto_suggestable"] for f in
                   engine.rank_findings([], prof))


def test_everything_survives_strict_json(df: pd.DataFrame):
    """No NaN on the wire.

    `json.dumps` writes a bare `NaN` for a missing float, which `JSON.parse`
    rejects — the browser reports a network error for a file whose only problem
    was a blank cell. `allow_nan=False` is that failure, moved into the suite.
    """
    prof = engine.profile(df, TARGET, "classification")
    payload = {
        "findings": engine.rank_findings(engine.diagnose(df), prof),
        "profile": engine.profile_to_dict(prof),
    }
    json.dumps(payload, allow_nan=False)


def test_a_text_target_is_read_as_classification(df: pd.DataFrame):
    """T0-LIVE-004's canary. Fails the moment the pandas cap is lifted.

    `ml/triage.py:41` decides task type with `dtype in ['object','category','bool']`.
    pandas 3 makes `str` the default dtype for text columns, so a text target
    matches no branch and falls through to the fallback at `:91` — *regression*,
    low confidence, no error raised. Measured on this fixture: `classification`
    / high under 2.3.3, `regression` / low under 3.0.5.

    Both requirements files cap `pandas<3` because of that. This test is what
    makes the cap enforceable: raising the ceiling without first replacing the
    dtype-identity checks with `pd.api.types` predicates breaks the suite here
    instead of silently changing a paper's statistics.

    Measured stage by stage across both majors on this fixture:

    | call | pandas 2.3.3 | pandas 3.0.5 |
    |---|---|---|
    | `diagnose` | 10 findings | 10 findings, identical |
    | `detect_task_type` | classification / high | **regression / low** |
    | `profile(task=detected)` | ok | **TypeError: Cannot perform reduction 'mean' with string dtype** |
    | `profile(task="classification")` | ok | ok |

    So `compute_dataset_profile` is not independently broken — it is correct
    when told the truth. The damage is that one wrong answer poisons the next
    call: the profiler takes the regression branch and averages a text column.
    The exception names the string dtype, not the misdetection that caused it,
    which is the kind of error that costs an afternoon.
    """
    assert not df[TARGET].map(type).eq(float).any(), "fixture target is not text"

    task = engine.detect_task_type(df, TARGET)
    assert task["detected"] == "classification", (
        f"a text target read as {task['detected']!r} — pandas is "
        f"{pd.__version__}; if that is 3.x, the cap was lifted without the repair"
    )
    assert task["confidence"] == "high"

    # The downstream half: feeding the detected task type back in must not
    # explode. That is where the misdetection used to surface, as a TypeError
    # naming the string dtype rather than the wrong answer that caused it.
    prof = engine.profile(df, TARGET, task["detected"])
    assert prof.target_profile.task_type == "classification"


def test_the_same_answer_under_a_pandas_3_string_dtype(df: pd.DataFrame):
    """T0-LIVE-004's repair, checked without needing pandas 3 installed.

    The bug was never about a version — it was `dtype in ['object', ...]`
    answering "is this text?" by identity. pandas 3 makes `str` the default for
    text columns, so the comparison stops matching and every branch keyed on it
    takes the wrong path.

    Rather than pin the suite to one major, this converts the target to the
    dtype pandas 3 would give it (`pd.StringDtype`, which pandas 2 also has) and
    asserts the answer is unchanged. `pd.api.types.is_string_dtype` is true for
    both, which is the whole point of asking through a predicate.
    """
    as_pandas3 = df.copy()
    as_pandas3[TARGET] = as_pandas3[TARGET].astype("string")
    assert str(as_pandas3[TARGET].dtype) != "object", "the fixture did not convert"

    task = engine.detect_task_type(as_pandas3, TARGET)
    assert task["detected"] == "classification", (
        f"a string-dtype text target read as {task['detected']!r} — this is "
        "T0-LIVE-004, and it is the shape pandas 3 hands every text column")
    assert task["confidence"] == "high"

    # And the downstream call that used to explode on it.
    prof = engine.profile(as_pandas3, TARGET, task["detected"])
    assert prof.target_profile.task_type == "classification"


def test_categorical_and_boolean_targets_are_still_classification(df: pd.DataFrame):
    """The predicate must not narrow what the identity list used to accept."""
    for dtype in ("category", "string"):
        frame = df.copy()
        frame[TARGET] = frame[TARGET].astype(dtype)
        assert engine.detect_task_type(frame, TARGET)["detected"] == "classification", (
            f"a {dtype} target stopped being classification")

    boolean = df.copy()
    boolean["flag"] = (boolean.index % 2 == 0)
    task = engine.detect_task_type(boolean, "flag")
    assert task["detected"] == "classification"


def test_a_nullable_integer_id_is_still_id_like(df: pd.DataFrame):
    """`dataset_profile:189` asked for `int64`/`int32` by name, so a nullable
    `Int64` column — which pandas produces for integers with missing values —
    was not recognized as an identifier."""
    frame = df.copy()
    frame["record_no"] = pd.array(range(len(frame)), dtype="Int64")
    prof = engine.profile(frame, TARGET, "classification")
    assert "record_no" in prof.id_like_features, (
        "a nullable-integer identifier was not recognized — the dtype-identity "
        "class again, at dataset_profile.py:189")


def test_engine_refuses_a_duplicated_target_label(df: pd.DataFrame):
    doubled = df.rename(columns={"site": TARGET})
    with pytest.raises(engine.EngineRefusal):
        engine.detect_task_type(doubled, TARGET)


def test_engine_refuses_a_column_that_is_not_there(df: pd.DataFrame):
    with pytest.raises(engine.EngineRefusal):
        engine.detect_task_type(df, "no_such_column")


# ═══════════════════════════════════════════════════════════════════════════
# Row identity is labels, not positions
# ═══════════════════════════════════════════════════════════════════════════

def test_rows_are_addressed_by_label_not_position(df: pd.DataFrame):
    """`TRANSITION_PLAN.md` §02.2, pinned.

    The frame is re-indexed so labels and positions disagree. Asking for label
    500 must return the row *labeled* 500; asking for label 0 must fail rather
    than quietly return the first row, which is what `.iloc[0]` would have done.
    """
    shifted = df.copy()
    shifted.index = range(500, 500 + len(shifted))
    proj = AnalysisProject.from_dataframe(shifted, "shifted.csv")

    assert proj.row_labels[0] == 500
    assert proj.rows([500]).iloc[0]["patient_id"] == shifted.loc[500, "patient_id"]

    with pytest.raises(ProjectError):
        proj.rows([0])


def test_row_labels_survive_a_filter(df: pd.DataFrame):
    """Filtering removes rows; it must not renumber the survivors."""
    proj = AnalysisProject.from_dataframe(df, "demo.csv")
    kept = [l for l in proj.row_labels if l % 2 == 0]
    filtered = AnalysisProject.from_dataframe(proj.rows(kept), "filtered.csv")
    assert filtered.row_labels == kept
    assert filtered.row_labels[1] == 2, "labels were reset — identity was lost"


def test_duplicate_row_labels_are_refused(df: pd.DataFrame):
    doubled = pd.concat([df, df])
    with pytest.raises(ProjectError):
        AnalysisProject.from_dataframe(doubled, "doubled.csv")


# ═══════════════════════════════════════════════════════════════════════════
# Decisions accumulate; nothing is silently destroyed
# ═══════════════════════════════════════════════════════════════════════════

def test_decisions_are_append_only(df: pd.DataFrame):
    proj = AnalysisProject.from_dataframe(df, "demo.csv")
    proj.set_target(TARGET, "classification", "high", ["object dtype"])
    proj.record("defer", "Recode 999 in age as missing", subject="sentinel_missing__age")
    proj.set_target("glucose", "regression", "med", ["numeric"])

    kinds = [d.kind for d in proj.decisions]
    assert kinds == ["set_target", "defer", "set_target"]
    assert proj.decisions[0].subject == TARGET, "the first answer was rewritten"


def test_retargeting_marks_findings_stale_without_deleting_them(df: pd.DataFrame):
    proj = AnalysisProject.from_dataframe(df, "demo.csv")
    proj.set_target(TARGET, "classification", "high", [])
    proj.set_findings(engine.rank_findings(engine.diagnose(df), None))
    n = len(proj.findings)
    assert n > 0 and proj.findings_stale is False

    proj.set_target("glucose", "regression", "med", [])
    assert proj.findings_stale is True
    assert len(proj.findings) == n, "stale findings were destroyed, not marked"


def test_project_serializes_without_the_frame(df: pd.DataFrame):
    proj = AnalysisProject.from_dataframe(df, "demo.csv")
    proj.set_target(TARGET, "classification", "high", [])
    d = proj.to_dict(include_rows=True)
    json.dumps(d, allow_nan=False)
    assert "df" not in d
    assert d["row_identity"] == "index_labels"
    assert len(d["row_labels"]) == len(df)
    assert d["n_rows"] == len(df)


# ═══════════════════════════════════════════════════════════════════════════
# Preview before apply
# ═══════════════════════════════════════════════════════════════════════════

def test_preview_shows_cells_that_really_changed(df: pd.DataFrame):
    """The gate: real cells, really different, quoted from both frames."""
    from ml import import_doctor

    finding = next(f for f in import_doctor.diagnose(df)
                   if f.id == "category_variants__sex")
    prev = engine.preview_fix(df, finding)

    assert prev["applicable"] and prev["changed_cells"] > 0
    assert "sex" in prev["changed_columns"]

    # Every cell the preview marks changed must actually differ, and the values
    # shown must be the real ones from each frame — not a formatted guess.
    after, _ = import_doctor.apply_fix(df.copy(deep=True), finding)
    cols = prev["sample"]["columns"]
    marked = 0
    for row in prev["sample"]["rows"]:
        label = row["label"]
        for i, col in enumerate(cols):
            assert row["before"][i] == ("" if pd.isna(df.at[label, col])
                                        else str(df.at[label, col]))
            assert row["after"][i] == ("" if pd.isna(after.at[label, col])
                                       else str(after.at[label, col]))
            if row["changed"][i]:
                marked += 1
                assert row["before"][i] != row["after"][i]
    assert marked > 0, "the sample showed no changed cell for a fix that changes 47"


def test_preview_reports_a_fix_that_renumbers_the_rows(df: pd.DataFrame):
    """The one thing a preview must not stay quiet about.

    Four of the nine fix kinds end in `reset_index(drop=True)`. This project
    keys rows by label, so a renumbering silently repoints every stored label —
    `TRANSITION_PLAN.md` §02.2's highest-risk item. `melt_repeated` is the
    clearest case: 140 rows become 420 and no label survives.
    """
    from ml import import_doctor

    melt = next(f for f in import_doctor.diagnose(df) if f.fix_kind == "melt_repeated")
    prev = engine.preview_fix(df, melt)
    assert prev["row_identity_preserved"] is False
    assert prev["row_identity_note"] and "renumbers" in prev["row_identity_note"]
    assert prev["shape"]["after"][0] > prev["shape"]["before"][0]


def test_row_identity_check_is_not_fooled_by_a_constant_column():
    """A constant column lines up under *any* renumbering.

    So "some column still matches" cannot be the test — it would wave through
    exactly the corruption this check exists to catch. Judged on every untouched
    column instead. Rows dropped from the middle break identity; rows dropped
    from the end of a clean `RangeIndex` genuinely do not.
    """
    from ml.import_doctor import ShapeFinding

    base = pd.DataFrame({"v": range(10), "site": ["A"] * 10,
                         "note": [f"r{i}" for i in range(10)]})
    mk = lambda kind, params: ShapeFinding(
        id="x", severity="warning", title="", detail="", why_it_matters="",
        fix_label="", fix_kind=kind, params=params)

    mid = engine.preview_fix(base, mk("drop_rows", {"positions": [2, 3]}))
    assert mid["row_identity_preserved"] is False, "a mid-frame drop renumbers survivors"

    tail = engine.preview_fix(base, mk("drop_rows", {"positions": [8, 9]}))
    assert tail["row_identity_preserved"] is True, "dropping the tail renumbers nothing"

    shifted = base.copy()
    shifted.index = range(500, 510)
    off = engine.preview_fix(shifted, mk("drop_rows", {"positions": [8, 9]}))
    assert off["row_identity_preserved"] is False, "reset on a non-RangeIndex loses labels"


def test_preview_refuses_where_the_engine_refuses(df: pd.DataFrame):
    """`fix_kind='none'` is the engine declining to guess. The preview says so."""
    from ml.import_doctor import ShapeFinding

    none_fix = ShapeFinding(id="x", severity="critical", title="", detail="",
                            why_it_matters="", fix_label="", fix_kind="none")
    prev = engine.preview_fix(df, none_fix)
    assert prev["applicable"] is False
    assert "human decision" in prev["description"]
