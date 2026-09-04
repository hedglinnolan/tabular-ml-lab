"""The export describes every group that was analyzed, not the one you ended on.

Run the women, run the men, download. The zip held the men's models, the men's
metrics, and a `report.md` whose "Rows: 319" was the men's N sitting under the
study's heading. The women's run was not in the file, was not mentioned in it,
and by then did not exist anywhere. Nothing in the export path had ever read
`cohort_runs_done`.

Two things are being pinned here.

The first is coverage: every banked branch reaches the bundle, the comparison
table reaches a file, and the multiplicity caveats — which say what two AUCs
side by side cannot be read as — reach `report.md` instead of living out their
lives as an `st.warning` on a page the reader never opens.

The second is the mechanism. The other branches' artifacts are built from their
snapshots by functions that do not import Streamlit, so the export cannot swap
a branch into the live keys to read it back. That matters for a reason worse
than tidiness: an export that swapped branches and then failed halfway would
leave the researcher standing in a cohort they did not choose, with the sidebar
saying so and nothing explaining why.
"""
from __future__ import annotations

import ast
import io
import json
import pathlib
import zipfile

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from turbotab.cascade import BRANCH_ARCHIVE_KEY
from utils.cohort_export import (
    add_cohort_bundles, branch_dir, branch_manifest, branch_metrics_csv,
    branch_predictions_csv, cohort_report_section, comparison_csv,
    comparison_table,
)
from utils.cohorts import (
    CohortRun, Snapshot, branch_key, comparison_caveats, plan_cohorts,
    switch_branch,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent

_WIPE = (BRANCH_ARCHIVE_KEY, "cohort_run", "cohort_runs_done", "raw_data",
         "filtered_data", "data_config", "_raw_data_fingerprint", "test_lockbox",
         "trained_models", "model_results", "X_test", "y_test",
         "preprocessing_pipelines_by_model", "_cohort_filter_broken")


@pytest.fixture(autouse=True)
def clean_state():
    def _wipe():
        for key in _WIPE:
            st.session_state.pop(key, None)
    _wipe()
    yield
    _wipe()


def study(n=400, seed=9):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sex": rng.choice(["Male", "Female"], n),
        "age": rng.integers(20, 80, n),
        "y": rng.choice([0, 1], n, p=[0.5, 0.5]),
    })


def target_for(df, label):
    plan = plan_cohorts(df, "sex", "y", "classification")
    cell = next(c for c in plan.viable if c.label == label)
    return (df, plan, cell, "y", [])


def snap_for(tag, auc=0.7, models=("ridge",)):
    """A Snapshot shaped like a finished run, without running one."""
    return Snapshot(
        keys={
            "trained_models": {m: f"<{m} fitted on {tag}>" for m in models},
            "model_results": {
                m: {"metrics": {"ROC-AUC": auc, "Accuracy": 0.6},
                    "y_test": [0, 1, 1, 0],
                    "y_test_pred": [0, 1, 0, 0]}
                for m in models
            },
            "preprocessing_pipelines_by_model": {},
        },
        run={"column": "sex", "label": tag, "n_rows": 200, "n_total": 400,
             "dropped_features": ["sex"]},
    )


def write_zip(archive, active_key, **kw):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        written = add_cohort_bundles(zf, archive, active_key, **kw)
    buf.seek(0)
    return zipfile.ZipFile(buf), written


# ── the artifacts of one branch ──────────────────────────────────────────

class TestOneBranchesArtifacts:

    def test_the_metrics_are_that_branchs(self):
        csv = branch_metrics_csv(snap_for("Female", auc=0.81))
        assert csv is not None
        row = pd.read_csv(io.StringIO(csv)).iloc[0]
        assert row["Model"] == "RIDGE"
        assert row["ROC-AUC"] == pytest.approx(0.81)

    def test_the_predictions_come_from_what_the_run_recorded(self):
        """Not from re-scoring. Re-running a model against the sealed rows to
        build an export would be another opening of the test set, uncounted,
        in a path nobody thinks of as an evaluation."""
        csv = branch_predictions_csv(snap_for("Female"), "ridge")
        got = pd.read_csv(io.StringIO(csv))
        assert list(got.columns) == ["Actual", "Predicted"]
        assert len(got) == 4

    def test_a_branch_with_no_results_yields_no_metrics_file(self):
        assert branch_metrics_csv(Snapshot()) is None

    def test_the_manifest_names_the_group_and_its_two_row_counts(self):
        m = branch_manifest(("sex", "Female"), snap_for("Female"), seal_opens=1)
        assert m["cohort_column"] == "sex" and m["cohort_label"] == "Female"
        assert m["n_rows_in_group"] == 200 and m["n_rows_in_study"] == 400
        assert m["held_out_slice_opened"] == 1
        assert m["is_whole_study"] is False

    def test_the_manifest_names_the_predictors_that_go_flat_in_this_group(self):
        """A reader comparing two folders will otherwise read a predictor that
        carries no information inside one group as a real difference."""
        m = branch_manifest(("sex", "Female"), snap_for("Female"))
        assert m["constant_in_this_group"] == ["sex"]

    def test_the_manifest_says_the_slices_are_disjoint(self):
        """Two folders each reporting held-out metrics look like one test set
        used twice unless the bundle says otherwise."""
        m = branch_manifest(("sex", "Female"), snap_for("Female"))
        assert "disjoint" in m["note"]

    def test_the_whole_study_branch_has_a_folder_and_says_what_it_is(self):
        assert branch_dir(("", "")) == "cohorts/everyone"
        assert branch_manifest(("", ""), Snapshot())["is_whole_study"] is True

    def test_a_label_with_a_slash_cannot_escape_the_folder(self):
        assert branch_dir(("site", "a/b")) == "cohorts/site=a_b"


# ── the tree ─────────────────────────────────────────────────────────────

class TestTheBundleTree:

    def test_every_branch_except_the_active_one_gets_a_folder(self):
        archive = {("sex", "Female"): snap_for("Female"),
                   ("sex", "Male"): snap_for("Male"),
                   ("", ""): snap_for("everyone")}
        zf, written = write_zip(archive, active_key=("sex", "Male"))
        names = zf.namelist()

        assert "cohorts/sex=Female/metrics.csv" in names
        assert "cohorts/everyone/metrics.csv" in names
        assert not any(n.startswith("cohorts/sex=Male/") for n in names), (
            "the active branch is the top-level bundle; a second copy under "
            "cohorts/ invites a reader to treat it as another study")
        assert {label for label, _ in written} == {"sex = Female", "Everyone"}

    def test_the_predictions_and_models_land_under_their_branch(self):
        archive = {("sex", "Female"): snap_for("Female")}
        zf, _ = write_zip(archive, active_key=("sex", "Male"),
                          model_dumper=lambda w, k: f"fake:{w}".encode())
        names = zf.namelist()
        assert "cohorts/sex=Female/predictions/ridge_predictions.csv" in names
        assert "cohorts/sex=Female/models/ridge_model.joblib" in names
        assert b"Female" in zf.read("cohorts/sex=Female/models/ridge_model.joblib")

    def test_the_export_checkboxes_are_honored_per_branch_too(self):
        archive = {("sex", "Female"): snap_for("Female")}
        zf, _ = write_zip(archive, active_key=("sex", "Male"),
                          model_dumper=lambda w, k: b"x",
                          include_models=False, include_predictions=False)
        names = zf.namelist()
        assert "cohorts/sex=Female/metrics.csv" in names
        assert not any("models/" in n for n in names)
        assert not any("predictions/" in n for n in names)

    def test_a_branch_that_produced_nothing_is_still_visible(self):
        """An empty folder with a reason. An absence reads as 'not analyzed',
        and this group WAS analyzed — it just has nothing to show."""
        archive = {("sex", "Female"): Snapshot(run={"column": "sex", "label": "Female"})}
        zf, _ = write_zip(archive, active_key=("sex", "Male"))
        manifest = json.loads(zf.read("cohorts/sex=Female/manifest.json"))
        assert "no fitted models" in manifest["note"]

    def test_a_model_that_cannot_be_serialized_costs_only_that_model(self):
        def explode(wrapper, key):
            raise RuntimeError("joblib said no")
        archive = {("sex", "Female"): snap_for("Female")}
        zf, _ = write_zip(archive, active_key=("sex", "Male"), model_dumper=explode)
        assert "cohorts/sex=Female/metrics.csv" in zf.namelist()


# ── the comparison, in a file ────────────────────────────────────────────

class TestTheComparisonReachesAFile:

    def runs(self):
        return [
            CohortRun(column="sex", label="Female", n_train=180, n_test=40,
                      completed=True, metrics={"ROC-AUC": 0.81},
                      dropped_features=["sex"], seal_opens=1),
            CohortRun(column="sex", label="Male", n_train=190, n_test=45,
                      completed=True, metrics={"ROC-AUC": 0.77}, seal_opens=1),
        ]

    def test_one_row_per_group_with_its_own_seal_count(self):
        table = comparison_table(self.runs())
        assert list(table["Group"]) == ["Female", "Male"]
        assert list(table["Held-out slice opened"]) == [1, 1], (
            "a per-group row must not carry the study-wide open count")

    def test_a_single_run_is_not_a_comparison(self):
        assert comparison_csv(self.runs()[:1]) is None

    def test_the_caveats_reach_the_report(self):
        """All four existed only as an st.warning on page 06."""
        caveats = comparison_caveats(self.runs(), "classification")
        assert caveats
        section = "\n".join(cohort_report_section(self.runs(), caveats, []))
        for c in caveats:
            assert c in section, f"caveat missing from report.md: {c[:60]}"

    def test_the_report_says_to_report_all_of_them(self):
        section = "\n".join(cohort_report_section(self.runs(), [], []))
        assert "not the one that worked" in section

    def test_the_report_points_at_the_folders_that_exist(self):
        section = "\n".join(cohort_report_section(
            self.runs(), [], [("sex = Female", "cohorts/sex=Female")]))
        assert "`cohorts/sex=Female/`" in section


# ── the mechanism ────────────────────────────────────────────────────────

class TestTheExportNeverTouchesTheLiveBranch:

    def test_writing_the_other_branches_leaves_the_active_one_alone(self):
        df = study()
        st.session_state["raw_data"] = df
        switch_branch(target_for(df, "Female"))
        st.session_state["trained_models"] = {"ridge": "<female fit>"}
        switch_branch(target_for(df, "Male"))
        st.session_state["trained_models"] = {"ridge": "<male fit>"}

        before_run = dict(st.session_state["cohort_run"])
        before_models = st.session_state["trained_models"]

        archive = st.session_state[BRANCH_ARCHIVE_KEY]
        write_zip(archive, active_key=branch_key(st.session_state["cohort_run"]))

        assert st.session_state["trained_models"] is before_models
        assert st.session_state["cohort_run"] == before_run
        assert st.session_state["cohort_run"]["label"] == "Male"

    def test_the_export_module_cannot_reach_session_state_at_all(self):
        """Structural, not a promise. A module that does not import Streamlit
        has no way to swap a branch in, so the rule cannot be broken by a
        future edit that forgets it."""
        src = (ROOT / "utils" / "cohort_export.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "streamlit" not in imported, (
            "utils/cohort_export.py imports Streamlit; it is meant to be a pure "
            "function of a Snapshot so the export cannot mutate the analysis")
        assert "session_state" not in src


# ── page 10 says whose rows it is describing ─────────────────────────────

def test_the_report_row_count_is_labeled_with_its_cohort():
    """`df` on page 10 is the ACTIVE COHORT's frame. An unlabeled "Rows: 319"
    beside a 600-person study is a number a researcher copies into an abstract."""
    src = (ROOT / "pages" / "10_Report_Export.py").read_text(encoding="utf-8")
    assert "in the whole study) |" in src, (
        "the Dataset Summary row count no longer names the cohort")


def test_the_metadata_dataset_block_names_the_cohort_and_the_study():
    src = (ROOT / "pages" / "10_Report_Export.py").read_text(encoding="utf-8")
    assert "'cohort':" in src and "'n_study':" in src


def test_the_restriction_sentence_says_where_the_other_groups_are():
    """It has always said the groups must be reported together. A reader
    holding only the manuscript had no way to find them."""
    from utils.workflow_provenance import UploadProvenance
    up = UploadProvenance()
    up.cohort_column = "sex"
    up.cohort_value = "Female"
    up.cohort_n = 319
    up.study_n = 600
    up.cohort_runs_completed = ["Female", "Male"]
    said = up.restriction_sentence()
    assert "`cohorts/`" in said, said
    assert "cohort_comparison.csv" in said, said


# ── the real page, the real zip ──────────────────────────────────────────

class TestPageTenActuallyWritesTheTree:
    """The hook on page 10 sits inside a `try/except` that logs and continues —
    an export must not be lost to a cohort bug. That makes a mistake in it
    SILENT, so it has to be proved against the real page rather than against
    the functions it calls.

    AppTest does not expose a download button's payload, so the proof is taken
    one level down: every `writestr` the page makes is recorded, which names
    the archive entries exactly and needs no zip to be read back.
    """

    def _run_page(self, df, archive, active_run, monkeypatch):
        import logging
        from streamlit.testing.v1 import AppTest
        from tests.conftest import inject_trained_state

        written = []
        real = zipfile.ZipFile.writestr

        def recording(self, name, data, *a, **kw):
            written.append(getattr(name, "filename", name))
            return real(self, name, data, *a, **kw)

        monkeypatch.setattr(zipfile.ZipFile, "writestr", recording)

        at = AppTest.from_file("pages/10_Report_Export.py", default_timeout=120)
        inject_trained_state(at.session_state, df, target_col="y")
        at.session_state["task_mode"] = "prediction"
        at.session_state[BRANCH_ARCHIVE_KEY] = archive
        if active_run is not None:
            at.session_state["cohort_run"] = active_run
        # The page's except branch logs and swallows — an export must not be
        # lost to a cohort bug. That makes a mistake there SILENT, so the log
        # is captured here directly rather than through pytest's caplog, which
        # a `-p no:logging` run would quietly remove along with the assertion.
        logged = []

        class _Catch(logging.Handler):
            def emit(self, record):
                logged.append(record.getMessage())

        handler = _Catch(level=logging.WARNING)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            at.run()
        finally:
            root.removeHandler(handler)

        assert not at.exception, [str(e.value)[:400] for e in at.exception]
        swallowed = [m for m in logged if "Could not write cohort bundles" in m]
        assert not swallowed, swallowed
        return written

    def _study(self, n=240, with_sex=True):
        rng = np.random.default_rng(4)
        cols = {"age": rng.normal(50, 12, n), "bmi": rng.normal(27, 4, n)}
        if with_sex:
            cols["sex"] = rng.choice(["Male", "Female"], n)
        df = pd.DataFrame(cols)
        df["y"] = 2.0 * df["age"] + rng.normal(0, 5, n)
        return df

    def test_the_export_writes_the_other_group_into_the_bundle(self, monkeypatch):
        df = self._study()
        archive = {("sex", "Female"): snap_for("Female", auc=0.81)}
        active = {"column": "sex", "label": "Male", "labels": list(df.index),
                  "n_rows": len(df), "n_total": len(df), "order": ["Female", "Male"],
                  "position": 2, "of": 2, "target_col": "y", "dropped_features": []}

        written = self._run_page(df, archive, active, monkeypatch)

        assert "cohorts/sex=Female/metrics.csv" in written, (
            f"the archived branch never reached the real export; wrote {written[:25]}")
        assert "cohorts/sex=Female/manifest.json" in written
        assert not any(w.startswith("cohorts/sex=Male/") for w in written), (
            "the active branch is the top-level bundle already")

    def test_a_study_with_no_cohorts_gets_no_cohorts_folder(self, monkeypatch):
        """The feature is invisible to everyone who never used it."""
        df = self._study(with_sex=False)
        written = self._run_page(df, {}, None, monkeypatch)
        assert "report.md" in written, "the ordinary bundle stopped being built"
        assert not [w for w in written if w.startswith("cohorts/")]


# ── the bundle describes ONE study, not every split ever tried ───────────

class TestTheBundleIsScopedToOnePartition:
    """Splitting by sex, then splitting by smoking status, leaves branches from
    both in the archive. They are overlapping row sets whose counts
    double-count the same people. `completed_runs()` has always scoped the
    comparison table to one grouping column — `render_next_cohort` documents
    exactly this bug on screen — and the folders beside it have to agree, or
    the bundle lists three "groups" of a two-group study under a caveat
    announcing "you fitted this model in 3 groups"."""

    def test_a_branch_from_another_grouping_variable_is_not_bundled(self):
        archive = {
            ("sex", "Female"): snap_for("Female"),
            ("smoker", "never"): snap_for("never"),
            ("", ""): snap_for("everyone"),
        }
        zf, written = write_zip(archive, active_key=("sex", "Male"))
        names = zf.namelist()
        assert "cohorts/sex=Female/metrics.csv" in names
        assert not any(n.startswith("cohorts/smoker=") for n in names), (
            "a superseded split's groups are in the bundle beside this study's")
        assert "cohorts/everyone/metrics.csv" in names, (
            "the whole study belongs to every partition")

    def test_standing_on_everyone_still_bundles_the_groups(self):
        """The whole-study branch has no column, so it must not be read as a
        partition that excludes everything else."""
        archive = {("sex", "Female"): snap_for("Female"),
                   ("sex", "Male"): snap_for("Male")}
        zf, _ = write_zip(archive, active_key=("", ""))
        names = zf.namelist()
        assert "cohorts/sex=Female/metrics.csv" in names
        assert "cohorts/sex=Male/metrics.csv" in names


class TestTheWholeStudyFolderDoesNotClaimDisjointness:

    def test_the_everyone_manifest_says_it_contains_the_others(self):
        """Its held-out set is every sealed row, so it CONTAINS each group's
        slice rather than sitting beside it. A reader pooling the folders on
        the strength of the group wording would double-count."""
        note = branch_manifest(("", ""), snap_for("everyone"))["note"]
        assert "not disjoint" in note, note
        assert "contains them" in note, note

    def test_a_group_manifest_still_claims_it(self):
        note = branch_manifest(("sex", "Female"), snap_for("Female"))["note"]
        assert "disjoint" in note and "not disjoint" not in note, note


def test_going_back_to_everyone_does_not_erase_the_comparison():
    """`completed_runs("")` matches no banked run, so the comparison table and
    every multiplicity caveat vanished from the report — on the one screen
    where the researcher has finished and is downloading. The grouping column
    has to come from the archive when no run is active."""
    src = (ROOT / "pages" / "10_Report_Export.py").read_text(encoding="utf-8")
    assert src.count("if k[0]), \"\")") >= 2, (
        "both completed_runs() call sites must fall back to the archive's "
        "grouping column when no cohort is active")
    assert "completed_runs(_col)" in src
    assert "completed_runs(_cmp_col)" in src
