"""
The walking skeleton's test: a real CSV in, real findings out.

Written before `api.py`, against the engine and project directly, so that the
API is built to match something already known to work rather than the other way
round. API-level tests live at the bottom and assert the *same* findings arrive
over HTTP.

Run:  turbotab/.venv/Scripts/python -m pytest turbotab/test_skeleton.py -v
"""
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
    """
    from ml import import_doctor          # the real thing, no adapter

    direct = import_doctor.diagnose(df)
    assert direct, "the fixture is supposed to be a messy file"

    ranked = engine.rank_findings(engine.diagnose(df), None)
    structural = [f for f in ranked if f["source"] == "structure"]

    assert len(structural) == len(direct)
    by_id = {f["id"]: f for f in structural}
    for d in direct:
        got = by_id[d.id]
        assert got["title"] == d.title
        assert got["detail"] == d.detail
        assert got["why_it_matters"] == d.why_it_matters
        assert got["severity"] == d.severity
        assert got["confidence"] == d.confidence
        assert got["fix_kind"] == d.fix_kind
        assert got["affected_columns"] == list(d.affected_columns)
        assert got["auto_suggestable"] is bool(d.auto_suggestable)


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
    # explode. Under pandas 3 this is where the misdetection actually surfaces.
    prof = engine.profile(df, TARGET, task["detected"])
    assert prof.target_profile.task_type == "classification"


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
    500 must return the row *labelled* 500; asking for label 0 must fail rather
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
# The same findings, over HTTP
#
# These were written after the four above and assert the API adds nothing: the
# JSON a browser receives has to carry the same findings the engine produced.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from turbotab.api import app
    return TestClient(app)


@pytest.fixture()
def uploaded(client, raw: bytes) -> str:
    r = client.post("/project", files={"file": ("clinic_visits.csv", raw, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_upload_returns_a_diagnosed_project(client, raw: bytes):
    body = client.post("/project",
                       files={"file": ("clinic_visits.csv", raw, "text/csv")}).json()
    assert body["n_rows"] == 140
    assert body["row_identity"] == "index_labels"
    assert body["findings"], "upload produced no findings"
    # The section opens already answered — no target has been chosen yet and the
    # structural diagnosis is nonetheless present.
    assert body["target"] is None
    assert body["sample"]["labels"][:3] == [0, 1, 2]


def test_http_findings_are_the_engine_findings(client, uploaded: str, df: pd.DataFrame):
    """What the browser gets is what `ml/` said. No drift in between."""
    over_http = client.get(f"/project/{uploaded}/findings").json()
    direct = engine.rank_findings(engine.diagnose(df), engine.profile(df, None, None))

    assert over_http["count"] == len(direct)
    assert [f["id"] for f in over_http["findings"]] == [f["id"] for f in direct]
    assert [f["title"] for f in over_http["findings"]] == [f["title"] for f in direct]
    assert [f["severity"] for f in over_http["findings"]] == [f["severity"] for f in direct]


def test_choosing_a_target_detects_the_task_and_records_it(client, uploaded: str):
    body = client.post(f"/project/{uploaded}/decision",
                       json={"kind": "set_target", "payload": {"column": TARGET}}).json()
    assert body["target"] == TARGET
    assert body["task_type"] == "classification"
    assert body["task_confidence"] == "high"
    assert body["task_reasons"], "a detection with no stated reason"
    assert any(d["kind"] == "set_target" for d in body["decisions"])


def test_decisions_accumulate_over_http(client, uploaded: str):
    fid = client.get(f"/project/{uploaded}/findings").json()["findings"][0]["id"]
    for kind in ("defer", "flag", "dismiss"):
        body = client.post(f"/project/{uploaded}/decision",
                           json={"kind": kind, "subject": fid}).json()
    kinds = [d["kind"] for d in body["decisions"]]
    assert kinds.count("defer") == 1 and kinds.count("flag") == 1
    assert kinds.count("dismiss") == 1
    # The sentence quotes the finding rather than paraphrasing it.
    title = client.get(f"/project/{uploaded}/findings").json()["findings"][0]["title"]
    assert any(title in d["text"] for d in body["decisions"])


def test_retarget_over_http_marks_stale_then_clears(client, uploaded: str):
    client.post(f"/project/{uploaded}/decision",
                json={"kind": "set_target", "payload": {"column": TARGET}})
    body = client.post(f"/project/{uploaded}/decision",
                       json={"kind": "set_target", "payload": {"column": "hba1c"}}).json()
    assert body["target"] == "hba1c"
    assert body["task_type"] == "regression"
    # Recompute happens in the same request, so the findings are current again.
    assert body["findings_stale"] is False
    assert len([d for d in body["decisions"] if d["kind"] == "set_target"]) == 2


def test_api_refuses_a_bad_target(client, uploaded: str):
    r = client.post(f"/project/{uploaded}/decision",
                    json={"kind": "set_target", "payload": {"column": "not_a_column"}})
    assert r.status_code == 400
    assert "not_a_column" in r.text


def test_api_refuses_an_unknown_decision_kind(client, uploaded: str):
    r = client.post(f"/project/{uploaded}/decision", json={"kind": "delete_everything"})
    assert r.status_code == 400


def test_missing_project_is_404(client):
    assert client.get("/project/deadbeef").status_code == 404
    assert client.get("/project/deadbeef/findings").status_code == 404


def test_empty_and_unreadable_uploads_are_refused(client):
    assert client.post("/project", files={"file": ("empty.csv", b"", "text/csv")}
                       ).status_code == 400
    r = client.post("/project", files={"file": ("junk.csv", b"\x00\x01\x02", "text/csv")})
    assert r.status_code == 400


def test_the_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "TurboTab walking skeleton" in r.text


def test_the_stylesheet_is_the_prototype_stylesheet_verbatim():
    """"Keep the design language exactly; only the data source changes."

    The prototype's `<style>` block is compared byte for byte. Anything that
    drifts — a colour nudged, a radius rounded — fails here rather than being
    noticed three screens later.
    """
    proto = (REPO_ROOT / "docs" / "turbotab" / "prototypes" /
             "interview-feed.html").read_text(encoding="utf-8")
    page = (REPO_ROOT / "turbotab" / "web" / "index.html").read_text(encoding="utf-8")
    css = proto[proto.index("<style>"):proto.index("</style>") + len("</style>")]
    assert css in page, "the prototype stylesheet was edited rather than carried over"


def test_the_frontend_has_no_synthetic_constants_left():
    """The rewire is the point: no invented number may survive in the page.

    Each name below is a synthetic constant from the prototype — the model
    scoreboard, the seeded RNG, the fabricated correlation matrix, the
    hard-coded 918-row dataset. If one reappears, something is being drawn from
    a literal instead of from the engine.
    """
    page = (REPO_ROOT / "turbotab" / "web" / "index.html").read_text(encoding="utf-8")
    body = page[page.index("</style>"):]
    for ghost in ("918", "CORRM", "CORRVARS", "HISTS", "RANGES",
                  "var MODELS", "PREVIEWS", "NOTES", "1664525", "0.934"):
        assert ghost not in body, f"synthetic constant {ghost!r} survived the rewire"


def test_http_responses_contain_no_nan(client, uploaded: str):
    """Starlette renders with `allow_nan=False`; a raw NaN would 500 right here."""
    for url in (f"/project/{uploaded}", f"/project/{uploaded}/findings"):
        r = client.get(url)
        assert r.status_code == 200
        assert "NaN" not in r.text
        json.loads(r.text)
