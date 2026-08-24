"""Invalidation and provenance completeness: the UI forgets, so the manuscript must.

Every finding here is one shape: a reset clears what the SCREEN shows and misses
what the EXPORT reads, so the pages go blank while the Methods draft keeps
asserting numbers computed from data, models or a quarantine regime that no
longer exist. A blank page is a prompt to re-run; a surviving number is a
published claim.

  CONTRACT-034 / STATE-047  provenance sections were nulled from a hand-typed
                            list that had drifted from the record's own schema
                            (`sensitivity` and `statistical_validation` absent)
  STATE-038                 manuscript-facing result keys were in neither of the
                            resetter's two inline lists
  STATE-040                 the exploratory watermark was popped by resets that
                            deliberately KEEP the artifacts it stains
  STATE-041                 neither the quarantine mode nor its watermark
                            survived a session archive round trip
  AUDIT-042                 the "N were addressed" count and the list beneath it
                            came from different filters
  MINE-010 / STATE-044      an unfingerprintable frame meant "unchanged"
  STATE-027                 removing an engineered feature left its technique
                            standing in the log that becomes Methods text
  STATE-037 (Wave 1 handoff) `filtered_data` is now part of WHO the results
                            describe, so the reset must drop it and page 05 must
                            fire the reset when it changes
"""
from __future__ import annotations

import ast
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_session_manager import fake_session  # noqa: F401
from utils import session_manager

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def session():
    """The real streamlit session_state, emptied around each test."""
    import streamlit as st
    st.session_state.clear()
    yield st.session_state
    st.session_state.clear()


def _study(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {"age": rng.normal(50, 10, n).round(3),
         "bmi": rng.normal(27, 4, n).round(3),
         "glucose": rng.normal(100, 12, n).round(3)},
        index=pd.RangeIndex(n),
    )


def _page_namespace(page: str, names: list) -> dict:
    """Exec the named module-level defs of a Streamlit page in isolation.

    Importing a page runs the whole script; these helpers are module-level and
    self-contained, so the regression test can exercise the real source without
    a Streamlit runtime.
    """
    import streamlit as st

    src = (REPO / "pages" / page).read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            label = (node.targets[0].id if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name)
                     else getattr(node, "name", None))
            if label in names:
                wanted.append(node)
    assert len(wanted) == len(names), (
        f"{page} no longer declares {set(names) - {getattr(n, 'name', None) for n in wanted}}")
    from utils.session_state import reset_downstream_results
    ns = {"st": st, "pd": pd, "np": np,
          "reset_downstream_results": reset_downstream_results}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), page, "exec"), ns)
    return ns


# ── CONTRACT-034 / STATE-047: the section list is the schema's, not a copy ──

class TestContract034ProvenanceSectionsAreDerived:
    def test_every_declared_section_is_discovered(self):
        from utils import workflow_provenance as wp

        names = set(wp.section_names())
        declared = {f for f, spec in wp.WorkflowProvenance.__dataclass_fields__.items()
                    if str(spec.type).endswith("Provenance]")}
        assert names == declared, "a section field the resetter cannot see is a section it cannot clear"
        # The two the hand-typed list forgot.
        assert {"sensitivity", "statistical_validation"} <= names

    def test_only_upload_survives_a_downstream_reset(self):
        from utils import workflow_provenance as wp

        cleared = set(wp.downstream_sections())
        assert cleared == set(wp.section_names()) - {"upload"}
        # `upload` describes the data configuration, which the reset preserves.
        # That omission is a decision, asserted here so it reads as one.
        assert wp.RESET_PRESERVED_SECTIONS == ("upload",)

    def test_flagged_sections_follow_the_artifact_they_describe(self):
        from utils.workflow_provenance import downstream_sections

        assert "feature_engineering" not in downstream_sections(
            clear_feature_engineering=False)
        assert "feature_selection" not in downstream_sections(
            clear_feature_selection=False)

    def test_a_data_change_leaves_no_stage_reporting_done(self, session):
        """get_completeness() is what a TRIPOD checklist and a Router key off."""
        from utils.session_state import reset_downstream_results
        from utils.workflow_provenance import WorkflowProvenance, section_names

        prov = WorkflowProvenance()
        for name in section_names():
            field_type = WorkflowProvenance.__dataclass_fields__[name].type
            cls = getattr(__import__("utils.workflow_provenance", fromlist=["x"]),
                          str(field_type).split("[")[1].rstrip("]").split(".")[-1])
            setattr(prov, name, cls())
        session["workflow_provenance"] = prov

        reset_downstream_results()

        done = [k for k, v in prov.get_completeness().items() if v]
        assert done == ["upload"], f"stages still reporting done after a reset: {done}"


class TestState047StatisticalValidationIsCleared:
    def test_the_methods_context_stops_naming_deleted_tests(self, session):
        from utils.session_state import reset_downstream_results
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        prov.record_statistical_test(
            test_name="Mann-Whitney U", variable="glucose", p_value=0.004)
        session["workflow_provenance"] = prov
        session["hypothesis_test_results"] = {"glucose": {"p": 0.004}}
        assert prov.get_methods_context().get("statistical_tests")

        reset_downstream_results()

        assert "hypothesis_test_results" not in session
        assert prov.statistical_validation is None, (
            "the record kept the tests whose results the same reset deleted")
        assert "statistical_tests" not in prov.get_methods_context()
        assert prov.get_completeness()["statistical_validation"] is False

    def test_sensitivity_goes_with_its_results(self, session):
        from utils.session_state import reset_downstream_results
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        prov.record_sensitivity(seed_stability=True, feature_dropout=True)
        session["workflow_provenance"] = prov
        session["sensitivity_seed_results"] = {"cv": 0.02}

        reset_downstream_results()

        assert prov.sensitivity is None
        ctx = prov.get_methods_context()
        assert "seed_stability" not in ctx and "feature_dropout" not in ctx


# ── STATE-038: the result registry is exhaustive ──────────────────────────

class TestState038ManuscriptResultKeys:
    MANUSCRIPT_KEYS = (
        "pdp_results",                    # pages/07 → pages/10 figures
        "sensitivity_dropout_results",    # pages/08 → ml/publication "Sensitivity Analysis"
        "sensitivity_dropout_baseline",
        "manuscript_export_context",      # a stale one WINS over rebuilding
    )

    # The wave-2 sweep of "read by an export surface, cleared by nobody".
    LATER_KEYS = (
        "split_trim_record",       # pages/06 → the realized trim of ONE split
        "dataset_profile_scope",   # pages/02 → which rows the profile describes
    )

    def test_they_are_registered(self):
        from utils.session_state import _ANALYSIS_KEYS, _REPORT_KEYS, _SPLIT_KEYS

        registry = set(_ANALYSIS_KEYS) | set(_REPORT_KEYS) | set(_SPLIT_KEYS)
        missing = [k for k in self.MANUSCRIPT_KEYS + self.LATER_KEYS
                   if k not in registry]
        assert not missing, f"unregistered manuscript result keys: {missing}"

    def test_a_target_change_clears_them(self, session):
        from utils.session_state import reset_downstream_results

        for key in self.MANUSCRIPT_KEYS + self.LATER_KEYS:
            session[key] = {"from": "the model this reset destroys"}
        session["partial_dependence"] = {"age": [1, 2]}

        reset_downstream_results()

        survivors = [k for k in self.MANUSCRIPT_KEYS + self.LATER_KEYS
                     if k in session]
        assert not survivors, f"survived the reset: {survivors}"
        assert session["partial_dependence"] == {}


class TestMethodologyLogIsTheLedgersSecondProducer:
    """pages/10:996 and ml/publication read `ledger_log or methodology_log`.

    The ledger's half rolls back on a downstream reset; the session list did
    not, so the fallback re-printed the Mann-Whitney and Shapiro-Wilk rows whose
    results the same reset had just deleted.
    """

    def test_the_rolled_back_pages_lose_their_entries(self, session):
        from utils.insight_ledger import InsightLedger
        from utils.session_state import log_methodology, reset_downstream_results

        session["insight_ledger"] = InsightLedger()
        log_methodology("Upload & Audit", "Selected target y", {"target": "y"})
        log_methodology("Preprocessing", "Built pipelines", {"scaling": "standard"})
        log_methodology("Statistical Validation", "Mann-Whitney U on glucose",
                        {"test_name": "Mann-Whitney U", "p_value": 0.004})
        log_methodology("EDA", "Generated correlation matrix", {})
        assert len(session["methodology_log"]) == 4

        reset_downstream_results()

        steps = [e["step"] for e in session["methodology_log"]]
        assert steps == ["Upload & Audit"], (
            f"stale rows survived the reset that deleted their results: {steps}")
        # And the two sources agree, which is the point of pruning by the same
        # page set the ledger rolls back.
        ledger_steps = [e["step"] for e in session["insight_ledger"].get_methodology_log()]
        assert ledger_steps == steps

    def test_a_step_this_module_cannot_place_is_kept(self, session):
        from utils.session_state import reset_downstream_results

        session["methodology_log"] = [{"step": "Hand-written note", "action": "x",
                                       "details": {}}]
        reset_downstream_results()
        assert [e["step"] for e in session["methodology_log"]] == ["Hand-written note"]


# ── STATE-037 handoff: the row filter is part of who the results describe ──

class TestState037FilteredDataIsInvalidated:
    def test_the_reset_drops_it(self, session):
        from utils.session_state import reset_downstream_results

        session["filtered_data"] = _study().iloc[:40]
        reset_downstream_results()
        assert "filtered_data" not in session, (
            "get_data() masks every page by filtered_data, so a stale filter "
            "keeps shrinking the dataset across every reset")

    def test_page_05_invalidates_when_the_filter_changes(self, session):
        ns = _page_namespace("05_Preprocess.py", ["_invalidate_on_row_filter_change"])
        invalidate = ns["_invalidate_on_row_filter_change"]
        study = _study()

        # Building with the same filter must not destroy a trained model.
        session["filtered_data"] = study.iloc[:40]
        session["trained_models"] = {"rf": object()}
        invalidate(study.index[:40])
        assert session["trained_models"], "an unchanged filter is not a data change"

        # A filter that changes WHO is in must invalidate.
        invalidate(study.index[:30])
        assert session["trained_models"] == {}
        assert "filtered_data" not in session

        # …and so must dropping the filter entirely.
        session["filtered_data"] = study.iloc[:30]
        session["trained_models"] = {"rf": object()}
        invalidate(None)
        assert session["trained_models"] == {}

    def test_build_pipelines_calls_it_on_both_arms(self):
        src = (REPO / "pages" / "05_Preprocess.py").read_text(encoding="utf-8")
        write = src.index('st.session_state["filtered_data"] = filtered_df')
        pop = src.index('st.session_state.pop("filtered_data", None)\n'
                        '                # The base, not `df`')
        for site in (write, pop):
            assert "_invalidate_on_row_filter_change" in src[site - 200:site], (
                "filtered_data changes hands with no downstream invalidation")

    def test_the_filter_is_computed_from_the_unmasked_base(self, session):
        """The ratchet: get_data() is already masked by the filter being rebuilt.

        Filtering the filtered and writing the result back means every press can
        only remove more people, and WIDENING a bound restores none of them.
        Measured on a 100-row study: 78 rows stayed excluded under bounds that
        admitted all 100.
        """
        from ml.pipeline import apply_plausibility_filter
        from utils.session_state import get_data, set_data

        study = _study(100).assign(age=lambda d: d["age"].abs())
        set_data(study)
        session["df_engineered"] = study.copy()
        narrow = {"lower_bounds": [45.0, 25.0, 90.0],
                  "upper_bounds": [55.0, 29.0, 110.0]}
        wide = {"lower_bounds": [0.0, 0.0, 0.0],
                "upper_bounds": [999.0, 999.0, 999.0]}
        cols = ["age", "bmi", "glucose"]

        # Press once with the narrow bounds, as the page does.
        session["filtered_data"] = apply_plausibility_filter(
            get_data(apply_row_filter=False), cols, narrow, None)
        narrowed = len(session["filtered_data"])
        assert 0 < narrowed < len(study), "fixture must actually exclude rows"

        # The base frame is unchanged by the mask that now exists…
        assert len(get_data(apply_row_filter=False)) == len(study)
        assert len(get_data()) == narrowed
        # …so widening the bound restores exactly the rows the bounds admit.
        session["filtered_data"] = apply_plausibility_filter(
            get_data(apply_row_filter=False), cols, wide, None)
        assert len(session["filtered_data"]) == len(study), (
            "widening a plausibility bound restored nothing: the filter was "
            "computed from the already-filtered frame")

    def test_the_build_handler_filters_the_base_frame(self):
        """The page source, because the handler is not callable in isolation."""
        src = (REPO / "pages" / "05_Preprocess.py").read_text(encoding="utf-8")
        assert "df_base = get_data(apply_row_filter=False)" in src
        assert "apply_plausibility_filter(\n                    df_base," in src, (
            "the plausibility filter is being computed from the masked frame")
        assert "sample_source = df_base" in src


# ── STATE-040: the watermark outlives what it stains ──────────────────────

class TestState040ExploratoryWatermark:
    def test_a_partial_reset_keeps_the_watermark(self, session):
        """The toggle-off call keeps df_engineered — it must keep the stain."""
        from utils.session_state import reset_downstream_results

        session["exploratory_used"] = True
        session["df_engineered"] = _study()

        reset_downstream_results(clear_feature_engineering=False)

        assert session.get("df_engineered") is not None, "fixture must reproduce the survivor"
        assert session.get("exploratory_used") is True, (
            "features derived with the lockbox open survived the reset that "
            "deleted their watermark; the manuscript then claims a clean "
            "held-out evaluation")

    def test_keeping_the_feature_selection_also_keeps_it(self, session):
        from utils.session_state import reset_downstream_results

        session["exploratory_used"] = True
        session["consensus_features"] = ["age", "bmi"]
        reset_downstream_results(clear_feature_selection=False)
        assert session.get("exploratory_used") is True

    def test_the_full_downstream_reset_keeps_it_too(self, session):
        """The APPLIED selection survives every call, so the stain must.

        `selected_features` / `pre_fe_feature_cols` / `data_config.feature_cols`
        are written by page 04 and cleared by no reset here — including the
        unflagged one. The watermark was popped anyway, so a same-schema
        set_data left the Methods claiming a sealed lockbox ahead of a feature
        set chosen with it open.
        """
        from utils.session_state import DataConfig, reset_downstream_results

        session["exploratory_used"] = True
        session["selected_features"] = ["age", "bmi"]
        session["data_config"] = DataConfig(target_col="glucose",
                                            feature_cols=["age", "bmi"])

        reset_downstream_results()

        assert session["selected_features"] == ["age", "bmi"], (
            "fixture must reproduce the survivor")
        assert session.get("exploratory_used") is True

    def test_a_same_schema_data_change_keeps_it(self, session):
        """The measured route: set_data → full downstream reset → clean claim."""
        from utils.session_state import set_data

        study = _study()
        set_data(study)
        session["exploratory_used"] = True
        session["selected_features"] = ["age", "bmi"]

        set_data(study.assign(glucose=lambda d: d["glucose"] + 1))

        assert session["selected_features"] == ["age", "bmi"]
        assert session.get("exploratory_used") is True, (
            "the watermark cleared while the selection it stains stayed applied")

    def test_only_a_new_dataset_clears_it(self, session):
        """reset_data_dependent_state empties the selection, so it may."""
        from utils.session_state import reset_data_dependent_state

        session["exploratory_used"] = True
        session["selected_features"] = ["age", "bmi"]

        reset_data_dependent_state()

        assert session["selected_features"] == []
        assert "exploratory_used" not in session


# ── STATE-041: the quarantine regime survives the archive ─────────────────

class TestState041QuarantineSurvivesRestore:
    def test_the_watermark_is_persisted(self):
        from utils.session_manager import _PLAIN_KEYS, _SAFE_WIDGET_KEYS

        assert "exploratory_used" in _PLAIN_KEYS, (
            "in neither bucket, the watermark was dropped on save and the "
            "restored session's manuscript lost its exploratory disclaimer")
        # exploratory_mode stays deferred: page 01's checkbox binds to it
        # (tests/test_session_manager.py pins that contract).
        assert "exploratory_mode" in _SAFE_WIDGET_KEYS

    def test_round_trip(self, fake_session):
        fake_session["raw_data"] = _study()
        fake_session["exploratory_mode"] = True
        fake_session["exploratory_used"] = True
        fake_session["selected_features"] = ["age", "bmi"]

        archive, _ = session_manager._collect_session_data()
        fake_session.clear()
        session_manager._restore_session_data(archive)

        assert fake_session.get("exploratory_used") is True, (
            "the honesty watermark did not survive the save")
        # pages/10 computes the manuscript flag as `mode or used`, so the
        # disclaimer survives even before page 01 claims the deferred mode key.
        pending = fake_session.get("_pending_widget_state_restore", {})
        assert pending.get("exploratory_mode") is True


# ── AUDIT-042: one filter behind the count and the list ───────────────────

class TestAudit042CountMatchesList:
    def _ledger(self):
        from utils.insight_ledger import Insight, InsightLedger

        ledger = InsightLedger()
        # What _log_to_ledger writes on every button press: resolved=True,
        # auto_generated, no substantive resolution_details.
        for i, action in enumerate(("Generated correlation matrix", "Ran mutual_info")):
            ledger.upsert(Insight(
                id=f"method_bridge_{i}", source_page="02_EDA", category="methodology",
                severity="info", finding=action,
                implication="Logged methodology decision", recommended_action="",
                resolved=True, resolved_by=action, resolved_on_page="02_EDA",
                auto_generated=True,
            ))
        ledger.upsert(Insight(
            id="eda_leakage", source_page="02_EDA", category="data_quality",
            severity="blocker", finding="`outcome_date` leaks the target",
            implication="Model performance will not generalize",
            recommended_action="Drop the column",
        ))
        return ledger

    def test_button_presses_are_not_addressed_observations(self):
        ledger = self._ledger()
        text = ledger.narrative_for_report()

        assert "identified 1 " in text, (
            f"the count still includes activity records:\n{text}")
        assert "0 were addressed during the modeling workflow" in text
        assert "Addressed observations:" not in text, (
            "a count above an empty list is the assertion this row is about")

    def test_the_second_producer_reads_the_same_filter(self):
        """pages/10's strengths line printed summary()["resolved"] — the ledger
        count — beside a narrative built from _is_narrative_worthy. Measured:
        six button presses, "6 observation(s) identified and addressed", and a
        narrative one line above saying 0 were identified."""
        ledger = self._ledger()

        assert ledger.summary()["resolved"] == 2, "fixture must hold the bridge entries"
        assert ledger.narrative_summary() == {
            "total": 1, "resolved": 0, "unresolved": 1}

        page10 = (REPO / "pages" / "10_Report_Export.py").read_text(encoding="utf-8")
        strengths = page10.index('f"Systematic data quality audit: {_resolved_count}')
        assert "_resolved_count = _report_ledger.narrative_summary()" in \
            page10[strengths - 900:strengths], (
                "the strengths line counts activity records as addressed "
                "observations")

    def test_a_real_resolution_is_still_counted(self):
        from utils.insight_ledger import Insight

        ledger = self._ledger()
        ledger.upsert(Insight(
            id="eda_skew", source_page="02_EDA", category="distribution",
            severity="warning", finding="`glucose` is heavily skewed",
            implication="Linear models will be biased",
            recommended_action="Log-transform",
            resolved=True, resolved_by="Applied log1p", resolved_on_page="05_Preprocess",
            resolution_details={"action_type": "transform", "columns_affected": ["glucose"]},
        ))
        text = ledger.narrative_for_report()
        assert "identified 2 " in text
        assert "1 were addressed" in text
        assert "Addressed observations:" in text
        assert "1 were documented and accepted" in text


# ── MINE-010 / STATE-044: unknown means changed ───────────────────────────

class TestMine010UnhashableFramesAreFingerprinted:
    def _nested(self, values) -> pd.DataFrame:
        """What pd.read_parquet yields for a list-typed column: ndarray cells."""
        return pd.DataFrame({"id": [1, 2, 3],
                             "codes": [np.array(v) for v in values]})

    def test_a_frame_pandas_cannot_hash_still_gets_a_fingerprint(self):
        from utils.session_state import _content_fingerprint

        df = self._nested([[1, 2], [3], [4, 5]])
        with pytest.raises(TypeError):
            pd.util.hash_pandas_object(df, index=False)
        assert _content_fingerprint(df) is not None, (
            "None was read by set_data as 'unchanged'")

    def test_the_fingerprint_notices_a_corrected_value(self):
        from utils.session_state import _content_fingerprint

        before = _content_fingerprint(self._nested([[1, 2], [3], [4, 5]]))
        after = _content_fingerprint(self._nested([[1, 2], [3], [4, 6]]))
        assert before != after
        assert _content_fingerprint(self._nested([[1, 2], [3], [4, 5]])) == before

    def test_a_same_schema_reupload_clears_the_stale_model(self, session):
        from utils.session_state import set_data

        set_data(self._nested([[1, 2], [3], [4, 5]]))
        session["trained_models"] = {"rf": object()}
        session["pdp_results"] = {"id": [1]}

        set_data(self._nested([[1, 2], [3], [4, 6]]))

        assert session["trained_models"] == {}
        assert "pdp_results" not in session


class TestState044UnknownMeansChanged:
    def test_an_unfingerprintable_frame_takes_the_reset_branch(self, session, monkeypatch):
        import utils.session_state as ss

        monkeypatch.setattr(ss, "_content_fingerprint", lambda df: None)
        ss.set_data(_study())
        session["trained_models"] = {"rf": object()}

        ss.set_data(_study().assign(glucose=lambda d: d["glucose"] + 1))

        assert session["trained_models"] == {}, (
            "'cannot fingerprint' was read as 'nothing changed', so every "
            "model, metric and figure survived under the new dataset's name")

    def test_a_benign_rerun_is_still_a_no_op(self, session):
        """Page 01 re-sets the same working table on every visit."""
        from utils.session_state import set_data

        study = _study()
        set_data(study)
        session["trained_models"] = {"rf": object()}

        set_data(study.copy())

        assert session["trained_models"], (
            "unknown-means-changed must not turn every page visit into a reset")

    def test_a_relabeled_reupload_is_a_change(self, session):
        """The collision channel: `index=False` made the fingerprint a hash of
        the MULTISET of rows, so a re-upload of the same people in a different
        order — a RangeIndex, so every label now names someone else — matched
        exactly. Models, metrics and `test_row_labels` survived, and the sealed
        test set became 50 different people with nothing to refuse on."""
        from ml.splits import resolve_split_rows
        from utils.session_state import get_data, set_data

        study = _study(60)
        set_data(study)
        session["trained_models"] = {"rf": object()}
        session["test_row_labels"] = list(study.index[-15:])
        sealed_before = resolve_split_rows(get_data(), session["test_row_labels"], "test")

        resorted = study.sort_values("age").reset_index(drop=True)
        assert sorted(resorted["glucose"]) == sorted(study["glucose"]), (
            "fixture must be the SAME rows, reordered")
        set_data(resorted)

        assert session["trained_models"] == {}, (
            "a re-sorted re-upload renamed every row and invalidated nothing")
        assert "test_row_labels" not in session
        # The fixture is only meaningful if the relabeling really moved people.
        assert not sealed_before.reset_index(drop=True).equals(
            resorted.loc[list(study.index[-15:])].reset_index(drop=True))

    def test_a_label_faithful_reorder_is_not_a_change(self, session):
        """Sorting a frame WITHOUT resetting the index moves no one: the labels
        travel with their rows, so every recorded split still names the same
        people and the results are still about them."""
        from utils.session_state import set_data

        study = _study(60)
        set_data(study)
        session["trained_models"] = {"rf": object()}

        set_data(study.sort_values("age"))

        assert session["trained_models"], (
            "a reorder that renamed nobody destroyed the analysis")

    def test_a_restored_session_without_a_fingerprint_is_not_a_change(self, session):
        from utils.session_state import set_data

        study = _study()
        set_data(study)
        session.pop("_raw_data_fingerprint")     # what a restore leaves behind
        session["trained_models"] = {"rf": object()}

        set_data(study.copy())

        assert session["trained_models"]


# ── STATE-027: a removed feature leaves no technique standing ─────────────

class TestState027EngineeringLogFollowsTheFeatures:
    def _ns(self):
        return _page_namespace(
            "03_Feature_Engineering.py",
            ["_build_transform_map", "_entry_feature_names",
             "_surviving_log_indices", "_prune_engineering_log", "_fe_commit"])

    def _wip(self, session):
        session["fe_work_in_progress"] = {
            "X_engineered": None, "engineered_features": [],
            "engineering_log": [], "engineering_log_sources": []}
        return session["fe_work_in_progress"]

    def _session_after(self, session, steps, removed):
        """Replay real commits (each recording what it created), then a removal."""
        commit = self._ns()["_fe_commit"]
        self._wip(session)
        feats: list = []
        log: list = []
        for entry, created in steps:
            log = log + [entry]
            feats = feats + list(created)
            log = commit(None, feats, log, created=created)
        feats = [f for f in feats if f not in removed]
        return commit(None, feats, log, removed=removed)

    def test_removing_a_techniques_whole_output_drops_its_claim(self, session):
        """A count-only entry is attributable only through what it recorded."""
        kept = self._session_after(
            session,
            [("Polynomial degree 2 (full): +2 features", ["age bmi", "age^2"]),
             ("PCA: +3 features (81.0% variance)", ["PCA_1", "PCA_2", "PCA_3"])],
            ["age bmi", "age^2"])

        assert kept == ["PCA: +3 features (81.0% variance)"], (
            "the Methods section named a technique that produced nothing the "
            "model saw")

    def test_a_shared_technique_tag_is_not_an_attribution(self, session):
        """The verifier's FALSE KEEP, with the exact strings pages/03 writes.

        `Custom interaction (Square (A²))` and `Mathematical transforms` both
        produce features that tag `power`, so a tag-based pruner kept the maths
        claim on the strength of the custom interaction's survivor — after every
        maths output had been removed.
        """
        kept = self._session_after(
            session,
            [("Custom interaction (Square (A²)): age_squared", ["age_squared"]),
             ("Mathematical transforms: +1 features", ["log_bmi"])],
            ["log_bmi"])

        assert kept == ["Custom interaction (Square (A²)): age_squared"], (
            "the surviving technique lost its entry and the emptied one kept "
            "its claim — the tag inversion, both ways at once")

    def test_an_unrelated_removal_leaves_every_other_technique_standing(self, session):
        """The verifier's FALSE DROP: `age^2` tags `other`, not `polynomial`."""
        kept = self._session_after(
            session,
            [("Polynomial degree 2 (full): +1 features", ["age^2"]),
             ("Missingness indicator: bmi_has_data (1 = observed, 0 = missing; "
              "8.0% missing)", ["bmi_has_data"]),
             ("Mathematical transforms: +1 features", ["log_age"])],
            ["log_age"])

        assert kept == [
            "Polynomial degree 2 (full): +1 features",
            "Missingness indicator: bmi_has_data (1 = observed, 0 = missing; "
            "8.0% missing)",
        ], "removing one feature deleted techniques whose output is still there"

    def test_a_partial_removal_keeps_the_technique(self, session):
        kept = self._session_after(
            session,
            [("PCA: +3 features (81.0% variance)", ["PCA_1", "PCA_2", "PCA_3"])],
            ["PCA_1", "PCA_2"])
        assert kept == ["PCA: +3 features (81.0% variance)"]

    def test_a_technique_with_survivors_keeps_its_entry(self):
        prune = self._ns()["_prune_engineering_log"]
        log = ["Mathematical transforms: +2 features"]
        assert prune(log, ["log_age"], ["sqrt_bmi"],
                     [["log_age", "sqrt_bmi"]]) == log

    def test_an_entry_that_names_its_feature_follows_that_feature(self):
        prune = self._ns()["_prune_engineering_log"]
        log = ["Missingness indicator: bmi_has_data (1 = observed, 0 = missing; 8.0% missing)",
               "Conditional ordinal: sex_ordinal (0=missing, 2 categories)"]

        kept = prune(log, ["sex_ordinal"], ["bmi_has_data"])

        assert kept == [log[1]]

    def test_a_removed_name_that_is_a_substring_of_a_survivor(self):
        """Exact tokens: `age_x_bmi` in the text is not `age_x_bmi_2`."""
        prune = self._ns()["_prune_engineering_log"]
        log = ["Custom interaction (*): age_x_bmi"]
        assert prune(log, ["age_x_bmi_2"], ["age_x_bmi"]) == []
        assert prune(log, ["age_x_bmi"], ["age_x_bmi_2"]) == log

    def test_an_unparseable_entry_is_kept(self):
        """Absent is better than false, but so is a log we cannot attribute."""
        prune = self._ns()["_prune_engineering_log"]
        log = ["Domain feature added by hand"]
        assert prune(log, ["kept_feature"], ["gone"]) == log

    def test_a_count_only_entry_with_no_recorded_source_is_kept(self):
        """"Unattributable is kept" must hold on EVERY path, including the one
        where a restored session carries a log and no attribution."""
        prune = self._ns()["_prune_engineering_log"]
        log = ["Polynomial degree 2 (full): +2 features"]
        assert prune(log, ["PCA_1"], ["age bmi", "age^2"]) == log

    def test_every_write_site_goes_through_the_one_committer(self):
        src = (REPO / "pages" / "03_Feature_Engineering.py").read_text(encoding="utf-8")
        body = src.split("def _fe_commit", 1)[1].split("\n\n\n", 1)[1]
        for key in ("X_engineered", "engineered_features", "engineering_log",
                    "engineering_log_sources"):
            assert f"fe_work_in_progress['{key}'] =" not in body, (
                "parallel assignments: the failure mode is always the one "
                "you forget")
            assert f'fe_work_in_progress["{key}"] =' not in body

    def test_the_committer_writes_all_of_them(self, session):
        ns = self._ns()
        self._wip(session)
        frame = _study()

        ns["_fe_commit"](frame, ["PCA_1"], ["PCA: +1 features"], created=["PCA_1"])

        wip = session["fe_work_in_progress"]
        assert wip["X_engineered"] is frame
        assert wip["engineered_features"] == ["PCA_1"]
        assert wip["engineering_log"] == ["PCA: +1 features"]
        assert wip["engineering_log_sources"] == [["PCA_1"]], (
            "a log entry that does not know what it made can never be pruned "
            "by anything but a guess")

    def test_every_technique_tab_records_what_it_created(self):
        src = (REPO / "pages" / "03_Feature_Engineering.py").read_text(encoding="utf-8")
        body = src.split("# Initialize\ninit_session_state()", 1)[1]
        lines = body.splitlines()
        calls = [" ".join(lines[i:i + 2]) for i, line in enumerate(lines)
                 if "_fe_commit(" in line]
        assert len(calls) >= 10, "the technique tabs moved; re-point this test"
        for call in calls:
            assert "created=" in call or "removed=" in call, (
                f"a commit that records nothing it made: {call.strip()}")
