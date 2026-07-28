"""L5 gates: the project model, the barrier, and durability.

Four things have to be true before `AnalysisProject` can replace
`st.session_state`:

1. it carries what the state layer carries — per-model pipeline **specs**, the
   active cohort filter, the lockbox, row identity as labels;
2. the identity barrier is a **phase rule**, not a convention: pre-barrier
   repairs are unreachable once a lockbox exists (`T0-ID-001`);
3. a project round-trips through the archive with no loss;
4. the serialization guard holds — no cell value from the loaded frame appears
   in a serialized project.

The DAG's own gate — reproducing both existing cascade implementations — lives
in `tests/integration/test_cascade_dag_equivalence.py`, because it needs a
Streamlit runtime to call the production function for real.

Run:  turbotab/.venv/Scripts/python -m pytest turbotab/test_project_model.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab import archive, cascade, engine, readiness
from turbotab.archive import ArchiveError
from turbotab.project import PRE_BARRIER_ONLY_FIXES, AnalysisProject, ProjectError

# Sealing requires the whole pre-seal sequence first — grain (§01/§02) since
# L13, and eligibility (§01/§04) since L16, both of which `seal_lockbox`
# refuses without. These fixtures are one row per person and unrestricted; the
# tests below are about the identity barrier and the round trip, not about
# either question, so they settle both and move on.
_GRAIN_UNIQUE = "one_row_per_person"


def _settle_preseal(project) -> None:
    """Answer grain and eligibility so the seal is drawable.

    One helper rather than two lines at twelve call sites, so the next clause
    that lands pre-seal is added once. That is the L16 migration's own lesson:
    twelve call sites is where a sequence change stops being cheap.
    """
    project.set_grain(_GRAIN_UNIQUE)
    project.set_eligibility("everyone")


DEMO_CSV = Path(__file__).resolve().parent / "sample_data" / "clinic_visits.csv"


@pytest.fixture
def df() -> pd.DataFrame:
    return engine.read_table(DEMO_CSV.read_bytes(), DEMO_CSV.name)


@pytest.fixture
def project(df) -> AnalysisProject:
    p = AnalysisProject.from_dataframe(df, "clinic_visits.csv")
    p.set_target("outcome", "classification", "high", ["object dtype"])
    return p


# ═══════════════════════════════════════════════════════════════════════════
# 1 · what the project carries
# ═══════════════════════════════════════════════════════════════════════════

def test_pipeline_specs_are_specs_not_fitted_objects(project):
    """`TRANSITION_PLAN.md` §02.1: the global slot handed two models the same
    live pipeline and page 06 fitted it in place, so their fitted pipelines
    aliased one instance. Specs are serializable, so they cannot alias."""
    project.pipeline_specs = {
        "rf": {"impute": "median", "scale": "none"},
        "ridge": {"impute": "median", "scale": "standard"},
    }
    json.dumps(project.pipeline_specs)          # a fitted pipeline would raise
    assert project.pipeline_specs["rf"] != project.pipeline_specs["ridge"]
    # There is no global slot to fall back to.
    assert not hasattr(project, "preprocessing_pipeline")


def test_the_cohort_filter_is_first_class(project, df):
    """A project that models "the working table" without modeling the active
    cohort filter silently deletes the newest feature (`ARCHITECTURE.md` §02)."""
    assert len(project.working_table) == len(df)

    male = [l for l in df.index if str(df.at[l, "sex"]).lower().startswith("m")]
    project.set_cohort("sex", "male", male, label="male participants")

    assert len(project.working_table) == len(male)
    assert len(project.working_table) < len(df)
    # And the record says which subset every later number is about.
    assert any(d.kind == "set_cohort" and "male participants" in d.text
               for d in project.decisions)

    project.clear_cohort()
    assert len(project.working_table) == len(df)


def test_the_cohort_filter_selects_by_label_not_position(df):
    """Filtering removes rows; the survivors keep their identities."""
    shifted = df.copy()
    shifted.index = range(500, 500 + len(shifted))
    p = AnalysisProject.from_dataframe(shifted, "shifted.csv")
    keep = list(shifted.index[10:20])
    p.set_cohort("site", "SITE-A", keep)
    assert list(p.working_table.index) == keep
    assert p.working_table.index.min() == 510


def test_readiness_is_the_ten_predicates(project):
    r = project.readiness()
    assert len(r.completed) == 10, "the step model lost a predicate"
    assert r.is_done("upload") and r.is_done("eda")
    assert not r.is_done("train")
    assert r.next_step().key == "features"
    # Quick vs advanced is disclosure, not deletion.
    assert len(r.visible_steps()) == 7
    project.workflow_mode = "advanced"
    assert len(project.readiness().visible_steps()) == 10


# ═══════════════════════════════════════════════════════════════════════════
# 2 · the identity barrier is a phase rule · T0-ID-001
# ═══════════════════════════════════════════════════════════════════════════

def test_pre_barrier_repairs_are_allowed_before_the_lockbox(project):
    assert project.barrier_raised is False
    for kind in PRE_BARRIER_ONLY_FIXES:
        project.check_repair_allowed(kind)      # must not raise


def test_pre_barrier_repairs_are_unreachable_once_the_lockbox_exists(project, df):
    """The phase rule. Before: allowed. After: refused, with a reason."""
    _settle_preseal(project)
    project.seal_lockbox(list(df.index[:20]), fraction=0.15, seed=42)
    assert project.barrier_raised is True

    for kind in PRE_BARRIER_ONLY_FIXES:
        with pytest.raises(ProjectError, match="rebuilds what a row is"):
            project.check_repair_allowed(kind)


def test_row_removing_repairs_stay_allowed_on_either_side(project, df):
    """`drop_rows` and `drop_empty_rows` reset the index in the engine, but they
    only remove rows — survivors keep their labels. The preview already reports
    by content whether a given drop renumbered anything, so these are not
    blanket-refused."""
    _settle_preseal(project)
    project.seal_lockbox(list(df.index[:20]), fraction=0.15, seed=42)
    for kind in ("drop_rows", "drop_empty_rows", "recode_missing",
                 "coerce_numeric", "normalize_categories", "drop_columns"):
        project.check_repair_allowed(kind)      # must not raise


def test_no_post_barrier_operation_changes_a_surviving_rows_index(project, df):
    """First barrier test. Apply an allowed repair after sealing and confirm
    every sealed label still names the same row."""
    sealed = list(df.index[:20])
    _settle_preseal(project)
    project.seal_lockbox(sealed, fraction=0.15, seed=42)
    before = {l: project.df.at[l, "patient_id"] for l in sealed}

    finding = next(f for f in engine.diagnose(project.df)
                   if f.id == "category_variants__sex")
    project.check_repair_allowed(finding.fix_kind)
    new_df, desc = engine.apply_fix(project.df, finding)
    project.apply_fix(new_df, finding.id, finding.title, desc, True)

    project.assert_identity_intact()
    for label, patient in before.items():
        assert project.df.at[label, "patient_id"] == patient, (
            f"label {label} names a different row after a post-barrier repair")


def test_a_renumbered_frame_is_detected_after_the_barrier(df):
    """Second barrier test. If something renumbers anyway, say so.

    On a clean `RangeIndex` a reset is a no-op, so the frame is shifted first —
    which is also the realistic case, since a frame that has been filtered once
    no longer has a `RangeIndex`.
    """
    shifted = df.copy()
    shifted.index = range(500, 500 + len(shifted))
    p = AnalysisProject.from_dataframe(shifted, "shifted.csv")
    p.set_target("outcome", "classification", "high", [])
    _settle_preseal(p)
    p.seal_lockbox(list(shifted.index[-20:]), fraction=0.15, seed=42)
    p.assert_identity_intact()

    p.df = p.df.reset_index(drop=True)               # the forbidden operation
    with pytest.raises(ProjectError, match="renumbered the rows"):
        p.assert_identity_intact()


def test_the_lockbox_is_sealed_once(project, df):
    _settle_preseal(project)
    project.seal_lockbox(list(df.index[:20]), fraction=0.15, seed=42)
    with pytest.raises(ProjectError, match="already has a sealed test set"):
        project.seal_lockbox(list(df.index[20:40]), fraction=0.15, seed=42)


def test_sealing_a_label_that_is_not_there_is_refused(project):
    with pytest.raises(ProjectError, match="not in this table"):
        _settle_preseal(project)
        project.seal_lockbox([999999])


# ═══════════════════════════════════════════════════════════════════════════
# 3 · durability — session_manager's schema, ported
# ═══════════════════════════════════════════════════════════════════════════

def test_the_archive_uses_session_managers_schema(project, df):
    """Same members, same names, same version — so the two doors can read each
    other's archives. A private schema forks durability."""
    from utils import session_manager

    assert archive.SAVE_SCHEMA_VERSION == session_manager.SAVE_SCHEMA_VERSION
    assert archive.ACCEPTED_SCHEMA_VERSIONS == session_manager._ACCEPTED_SCHEMA_VERSIONS

    _settle_preseal(project)

    project.seal_lockbox(list(df.index[:20]), fraction=0.15, seed=42)
    project.set_cohort("site", "SITE-A", list(df.index[:100]))
    members = archive.build_members(project)

    for expected in ("manifest.json", "config.json", "lockbox.json",
                     "cohort.json", "provenance.json",
                     "data/working_table.parquet"):
        assert expected in members, f"archive is missing {expected}"


def test_a_project_round_trips_with_no_loss(project, df):
    """The gate. Everything that is a decision or an input survives."""
    _settle_preseal(project)
    project.seal_lockbox(sorted(df.index[:20].tolist()), fraction=0.15, seed=42,
                         group_col="patient_id", group_kind="subject")
    project.set_cohort("site", "SITE-A", list(df.index[:100]), label="Site A")
    project.pipeline_specs = {"rf": {"impute": "median"}}
    project.record("defer", "sex variants — deferred", subject="category_variants__sex")

    restored = archive.from_bytes(archive.to_bytes(project))

    assert restored.id == project.id
    assert restored.name == project.name
    assert restored.target == project.target
    assert restored.task_type == project.task_type
    assert restored.task_confidence == project.task_confidence
    assert restored.pipeline_specs == project.pipeline_specs
    assert restored.workflow_mode == project.workflow_mode
    # The record, in full and in order.
    assert [d.kind for d in restored.decisions] == [d.kind for d in project.decisions]
    assert [d.text for d in restored.decisions] == [d.text for d in project.decisions]
    # The sealed holdout, verbatim — a re-draw would change the test set every
    # prior result was scored on.
    assert restored.lockbox["labels"] == project.lockbox["labels"]
    assert restored.lockbox["group_col"] == "patient_id"
    assert restored.cohort["labels"] == project.cohort["labels"]
    assert restored.cohort["label"] == "Site A"
    # The table, including its row labels.
    pd.testing.assert_frame_equal(restored.df, project.df)
    assert list(restored.df.index) == list(project.df.index)


def test_row_labels_survive_the_round_trip_when_they_are_not_a_range(df):
    """The failure `_plain_label` exists to prevent: a numpy int written as a
    string no longer matches an integer index, and the lockbox silently
    quarantines nothing."""
    shifted = df.copy()
    shifted.index = range(1000, 1000 + len(shifted))
    p = AnalysisProject.from_dataframe(shifted, "shifted.csv")
    p.set_target("outcome", "classification", "high", [])
    sealed = list(shifted.index[:15])
    _settle_preseal(p)
    p.seal_lockbox(sealed, fraction=0.1, seed=1)

    restored = archive.from_bytes(archive.to_bytes(p))
    assert restored.lockbox["labels"] == sealed
    assert all(isinstance(l, int) for l in restored.lockbox["labels"])
    # And they still select rows.
    assert len(restored.df.loc[restored.lockbox["labels"]]) == 15


def test_derivatives_are_not_persisted(project, df):
    """The drop-list, derived from the graph rather than maintained by hand."""
    project.set_findings(engine.rank_findings(engine.diagnose(df), None))
    assert project.findings, "nothing to drop"

    # The manifest lists skipped keys by name, which is the honest thing for a
    # manifest to do — so it is excluded, and only the members that carry state
    # are searched.
    members = archive.build_members(project)
    text = "\n".join(v.decode("utf-8") for k, v in sorted(members.items())
                     if k.endswith(".json") and k != "manifest.json")
    for derived in cascade.all_result_keys():
        assert f'"{derived}"' not in text, (
            f"{derived} was persisted; it is a derivative and must be regenerated")

    restored = archive.from_bytes(archive.to_bytes(project))
    assert restored.findings == []
    assert restored.findings_stale is True, (
        "a restored project claimed its findings were current")


def test_an_archive_that_is_a_pickle_is_refused():
    with pytest.raises(ArchiveError, match="pickle"):
        archive.from_bytes(b"\x80\x04\x95 pickled payload")


def test_an_unknown_schema_version_is_refused(project):
    import io
    import zipfile

    raw = archive.to_bytes(project)
    buf = io.BytesIO()
    src = zipfile.ZipFile(io.BytesIO(raw))
    with zipfile.ZipFile(buf, "w") as out:
        for name in src.namelist():
            data = src.read(name)
            if name == "manifest.json":
                m = json.loads(data)
                m["schema_version"] = "99.0"
                data = json.dumps(m).encode()
            out.writestr(name, data)
    with pytest.raises(ArchiveError, match="schema"):
        archive.from_bytes(buf.getvalue())


# ═══════════════════════════════════════════════════════════════════════════
# 4 · the serialization guard
# ═══════════════════════════════════════════════════════════════════════════

def test_no_participant_data_appears_in_a_serialized_project(project, df):
    """The operational form of "never persist participant data".

    The table travels as parquet — that is the user saving their own data. The
    *decision record* must not become a second, unmanaged copy of it.
    """
    _settle_preseal(project)
    project.seal_lockbox(list(df.index[:20]), fraction=0.15, seed=42)
    project.set_cohort("site", "SITE-A", list(df.index[:100]), label="Site A")
    project.record("defer", "sex variants — deferred", subject="category_variants__sex")

    blob = archive.serialized_json(project)
    archive.assert_no_participant_data(
        blob, df,
        # A cohort filter names the value it filters on, and the record has to
        # say which group every later number is about. Exempt by name, so the
        # exemption is arguable rather than invisible.
        allow=["SITE-A", "Site A"])
    # And the sharper statement: no row is reconstructible from the record.
    archive.assert_no_row_is_recoverable(blob, df, allow=["SITE-A", "Site A"])


def test_the_guard_catches_a_leak(project, df):
    """Proving the guard can fail, rather than trusting that it would."""
    leaked = str(df.at[df.index[0], "patient_id"])
    project.record("note", f"participant {leaked} looked unusual")

    with pytest.raises(ArchiveError, match="appear in the serialized project"):
        archive.assert_no_participant_data(archive.serialized_json(project), df)


def test_findings_would_leak_which_is_why_they_are_not_persisted(project, df):
    """Why the drop-list is a privacy control and not just a size optimization.

    Engine findings quote real values — *"Found 999 (4x)"*, *"e.g. '107 kg'"* —
    so persisting them would put cell values in the record. They are
    derivatives, so they are regenerated instead. This asserts the leak is real,
    so the reason for dropping them cannot quietly stop applying.
    """
    findings = engine.rank_findings(engine.diagnose(df), None)
    blob = json.dumps(findings, default=str)
    with pytest.raises(ArchiveError):
        archive.assert_no_participant_data(blob, df)


# ═══════════════════════════════════════════════════════════════════════════
# The stretch gate — one core, two doors
# ═══════════════════════════════════════════════════════════════════════════

def test_a_project_opened_in_the_other_door_is_unchanged(project, df):
    """`ROADMAP.md` L5's stretch gate, and the proof that there is one core.

    The framing that made this look blocked was mine: switching doors
    mid-analysis needs no persistence at all. Both doors are views over one
    running core, so "opening the other door" is constructing the other view
    over the same project object. Durability — resuming tomorrow — is the
    separate question `turbotab.archive` answers.

    So this asserts the thing that actually matters: the *state* both doors read
    is identical, and reading it through either view does not change it.
    """
    _settle_preseal(project)
    project.seal_lockbox(list(df.index[:20]), fraction=0.15, seed=42)
    project.set_cohort("site", "SITE-A", list(df.index[:100]), label="Site A")
    project.pipeline_specs = {"rf": {"impute": "median"}}
    before = project.fingerprint()

    # Door 1 — Guided: the project's own view.
    guided = project.to_dict()

    # Door 2 — Classic: the same project, read through the shared readiness
    # model the Streamlit sidebar now uses.
    from turbotab import readiness as _readiness
    classic = _readiness.assess(project._readiness_state(), project.workflow_mode)

    assert guided["readiness"]["completed"] == classic.completed, (
        "the two doors disagree about which steps are done")
    assert guided["readiness"]["next"] == (
        classic.next_step().key if classic.next_step() else None)

    # Neither view changed anything.
    assert project.fingerprint() == before
    assert len(project.decisions) == len(project.to_dict()["decisions"])
    # And the analysis is still about the same rows.
    assert guided["n_working_rows"] == len(project.working_table) == 100


def test_the_two_doors_read_one_lockbox(project, df):
    """A lockbox in the project is the lockbox Classic would see.

    `utils/test_lockbox.py` stores exactly this shape in session state, so the
    same dict satisfies both. If the doors held separate lockboxes they would
    quarantine different rows and every comparison between them would be
    meaningless.
    """
    sealed = sorted(df.index[:25].tolist())
    _settle_preseal(project)
    project.seal_lockbox(sealed, fraction=0.18, seed=7, signature="sig-1")

    as_session_state = project.lockbox
    for required in ("labels", "n_total", "n_test", "target_col"):
        assert required in as_session_state, (
            f"the project's lockbox is missing {required!r}, which Classic reads")
    assert as_session_state["labels"] == sealed

    # Round-tripping through the shared archive preserves it verbatim.
    restored = archive.from_bytes(archive.to_bytes(project))
    assert restored.lockbox["labels"] == sealed
    assert restored.lockbox["seed"] == 7
