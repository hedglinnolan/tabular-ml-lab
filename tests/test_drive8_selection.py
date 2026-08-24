"""Drive 8 — the boolean-with-missing outcome, and what the app said about it.

The drive's dataset holds `meds_hbp`: an object column of Python True/False
with 15,552 blanks. The app calls it Classification (correctly — it IS a binary
outcome), scikit-learn's `type_of_target` calls it 'unknown', and nothing in
between reconciled the two. What followed:

  * `DRIVE-064` — LASSO and RFE-CV both raised, and the message on screen was
    "Unknown label type: unknown. Maybe you are trying to fit a classifier,
    which expects discrete classes on a regression target with continuous
    values" — a cause that is not the cause. Then "Only one method completed"
    when zero did, and "✅ Feature selection complete! 0 methods run."
  * `DRIVE-063` — provenance was handed the REQUESTED method list, so the
    Methods draft printed "Consensus feature selection across LASSO and RFE-CV
    retained all 27 candidate predictors" over a run that selected nothing.
  * `DRIVE-067` — the Import Doctor described the column as "Every value is a
    plain number (e.g. 'True', 'False')", offered a high-confidence repair that
    blanked all 6,297 values, and never rendered at all for a dataset that
    reached the project through the registry rather than the uploader.
  * finding 27 — the constant-columns note carried a missing-value-sentinel
    caption.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest

from ml.import_doctor import ShapeFinding, apply_fix, diagnose
from ml.triage import detect_task_type, diagnose_target_dtype, repair_boolean_target


# ── fixtures ─────────────────────────────────────────────────────────────

def boolean_target_frame(n: int = 240, n_missing: int = 80, seed: int = 7):
    """The drive's shape in miniature: a True/False outcome with blanks."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(50, 12, n),
        "bmi": rng.normal(27, 4, n),
        "sbp": rng.normal(126, 15, n),
        "chol": rng.normal(198, 35, n),
        "kcal": rng.normal(2100, 400, n),
        "gender": rng.choice(["male", "female"], n),
    })
    truth = (df["sbp"] + df["bmi"] * 2) > (df["sbp"].mean() + df["bmi"].mean() * 2)
    # An object column of real bools with NaN — exactly what read_csv returns
    # for a True/False/'' column.
    outcome = pd.Series([bool(v) for v in truth], dtype=object)
    outcome.loc[rng.choice(n, n_missing, replace=False)] = np.nan
    df["meds_hbp"] = outcome
    return df


def _state(at, key, default=None):
    """AppTest's session_state has no .get()."""
    try:
        return at.session_state[key]
    except (KeyError, AttributeError):
        return default


def _page_text(at) -> str:
    parts = []
    for attr in ("markdown", "caption", "info", "warning", "error", "success"):
        for el in getattr(at, attr, []):
            parts.append(str(getattr(el, "value", "")))
    return " ".join(parts)


def _inject_prediction_state(at, df, target="meds_hbp", task="classification"):
    from utils.session_state import DataConfig
    feature_cols = [c for c in df.columns if c != target]
    at.session_state["raw_data"] = df
    at.session_state["task_mode"] = "prediction"
    at.session_state["data_config"] = DataConfig(
        target_col=target, feature_cols=feature_cols, task_type=task)
    at.session_state["selected_features"] = feature_cols


def _registry_app(df, name="nhanes"):
    """Page 01 with a dataset that arrived through the registry, not the uploader."""
    at = AppTest.from_file("pages/01_Upload_and_Audit.py", default_timeout=90)
    at.session_state["sp_projects"] = {1: {
        "id": 1, "name": "t", "description": "", "active": True,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "datasets": {1: {
            "id": 1, "project_id": 1, "name": name, "filename": f"{name}.csv",
            "file_type": "csv", "shape_rows": df.shape[0],
            "shape_cols": df.shape[1], "columns": list(df.columns),
            "column_types": None, "is_transposed": False,
            "upload_timestamp": "2026-01-01T00:00:00+00:00",
        }},
        "merge_configs": {},
    }}
    at.session_state["sp_counter_project"] = 1
    at.session_state["sp_counter_dataset"] = 1
    at.session_state["datasets_registry"] = {1: df}
    return at


# ── DRIVE-067: what the Import Doctor says about a boolean column ────────

def test_drive067_a_boolean_column_is_not_described_as_a_plain_number():
    """"Every value is a plain number (e.g. 'True', 'False')" — it is not."""
    df = boolean_target_frame()
    findings = {f.id: f for f in diagnose(df)}
    finding = findings.get("boolean_as_text__meds_hbp")
    assert finding is not None, (
        "a True/False column with blanks must be reported as such; "
        f"got {sorted(findings)}")
    said = f"{finding.title} {finding.detail}"
    assert "plain number" not in said, said
    assert "holds numbers" not in said, said
    assert "True/False" in said
    # The condition it names must be the one that actually bites downstream.
    assert "text" in said.lower()


def test_drive067_the_boolean_repair_does_not_blank_the_column():
    """The old high-confidence 'Convert to numbers' fix blanked every value."""
    df = boolean_target_frame()
    finding = next(f for f in diagnose(df) if f.id == "boolean_as_text__meds_hbp")
    repaired, description = apply_fix(df, finding)

    before = int(df["meds_hbp"].notna().sum())
    after = int(repaired["meds_hbp"].notna().sum())
    assert after == before, (
        f"the repair blanked {before - after} of {before} outcome values; "
        f"it reported: {description}")
    assert set(repaired["meds_hbp"].dropna().unique()) == {0.0, 1.0}
    assert int(repaired["meds_hbp"].isna().sum()) == int(df["meds_hbp"].isna().sum())
    assert "True → 1" in description and "False → 0" in description


def test_drive067_the_boolean_repair_refuses_a_column_it_would_damage():
    """The recode is exact or it does not happen."""
    df = boolean_target_frame()
    df.loc[0, "meds_hbp"] = "unknown"
    finding = ShapeFinding(
        id="boolean_as_text__meds_hbp", severity="warning", title="t",
        detail="d", why_it_matters="w", fix_label="l",
        fix_kind="coerce_boolean", params={"column": "meds_hbp"})
    with pytest.raises(ValueError, match="would blank"):
        apply_fix(df, finding)


def test_drive067_registry_dataset_gets_a_structural_review_on_page_01():
    """The card ran only under the uploader, so a registry dataset never saw it."""
    at = _registry_app(boolean_target_frame()).run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    text = _page_text(at)
    assert "Structural review" in text, (
        "the working table must be reviewed whatever path the dataset arrived by")
    assert "meds_hbp" in text and "True/False" in text, text[-1500:]


def test_drive067_a_working_table_repair_is_committed_and_recorded():
    """The card is only worth rendering if pressing its button changes the table."""
    at = _registry_app(boolean_target_frame())
    at.run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    button = next(b for b in at.button
                  if "Recode 'meds_hbp'" in b.label)
    button.click().run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]

    working = at.session_state["working_table"]
    assert set(working["meds_hbp"].dropna().unique()) == {0.0, 1.0}
    actions = " ".join(e.get("action", "")
                       for e in _state(at, "methodology_log", []))
    assert "meds_hbp" in actions, actions


# ── finding 27: the caption under the constant-columns note ──────────────

def test_finding27_constant_columns_do_not_carry_the_sentinel_caption():
    df = boolean_target_frame()
    df["study_site"] = "SITE-A"
    df["protocol"] = 3
    finding = next(f for f in diagnose(df) if f.id == "constant_columns")
    note = finding.uncertainty_note
    assert note, "a low-confidence finding must say what is uncertain about IT"
    assert "mean 'missing'" not in note, note
    assert "label" in note


def test_finding27_the_ui_prefers_a_findings_own_uncertainty_note():
    """import_ui must use the per-finding note rather than the tier default."""
    from utils.import_ui import _CONFIDENCE_NOTE
    df = boolean_target_frame()
    df["study_site"] = "SITE-A"
    finding = next(f for f in diagnose(df) if f.id == "constant_columns")
    chosen = (getattr(finding, "uncertainty_note", None)
              or _CONFIDENCE_NOTE.get(finding.confidence))
    assert chosen == finding.uncertainty_note
    assert chosen != _CONFIDENCE_NOTE["low"]


# ── DRIVE-064: the target the app called classification and sklearn did not ──

def test_drive064_boolean_with_missing_target_is_diagnosed_before_it_flows():
    df = boolean_target_frame()
    dx = diagnose_target_dtype(df, "meds_hbp")
    assert dx.sklearn_type == "unknown"
    assert dx.usable is False
    assert dx.repairable is True
    # The condition must name storage, never "continuous".
    assert "continuous" not in dx.condition.lower()
    assert "True/False" in dx.condition
    # And the fix must name the repair by the label the button carries.
    finding = next(f for f in diagnose(df) if f.id == "boolean_as_text__meds_hbp")
    assert finding.fix_label in dx.fix, (finding.fix_label, dx.fix)


def test_drive064_the_repair_makes_the_target_readable_by_the_selectors():
    from ml.feature_selection import lasso_path_selection, rfe_cv_selection
    from sklearn.impute import SimpleImputer

    df = boolean_target_frame()
    df["meds_hbp"] = repair_boolean_target(df["meds_hbp"])
    assert diagnose_target_dtype(df, "meds_hbp").usable

    feats = ["age", "bmi", "sbp", "chol", "kcal"]
    mask = df["meds_hbp"].notna()
    X = SimpleImputer(strategy="median").fit_transform(df.loc[mask, feats].values)
    y = df.loc[mask, "meds_hbp"].values
    for fn in (lasso_path_selection, rfe_cv_selection):
        result = fn(X, y, feats, "classification", cv_folds=3, random_state=42)
        assert result.all_features == feats


def test_drive064_a_repaired_binary_target_is_still_confidently_classification():
    """The repair must not demote the detection it exists to make work."""
    df = boolean_target_frame()
    before = detect_task_type(df, "meds_hbp")
    df["meds_hbp"] = repair_boolean_target(df["meds_hbp"])
    after = detect_task_type(df, "meds_hbp")
    assert before["detected"] == after["detected"] == "classification"
    assert after["confidence"] == "high", after


def test_drive064_page01_repairs_the_target_and_says_so():
    at = _registry_app(boolean_target_frame())
    at.session_state["task_mode"] = "prediction"
    at.run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    at.selectbox(key="target_selectbox").select("meds_hbp").run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]

    text = _page_text(at)
    assert "recoded to" in text.lower() or "was recoded" in text.lower(), text[-2000:]
    assert "1 (True" in text and "0 (False" in text, text[-2000:]

    working = at.session_state["working_table"]
    assert set(working["meds_hbp"].dropna().unique()) == {0.0, 1.0}
    assert diagnose_target_dtype(working, "meds_hbp").usable
    # And the repair is in the record the manuscript is written from, not just
    # on the screen.
    actions = " ".join(e.get("action", "")
                       for e in _state(at, "methodology_log", []))
    assert "meds_hbp" in actions and "1/0" in actions, actions


def test_drive064_page01_refuses_a_target_it_cannot_repair():
    df = boolean_target_frame()
    # Mixed value kinds in one column: no safe recode exists.
    df["mixed_outcome"] = ([True, "yes", 3.5, False] * (len(df) // 4 + 1))[:len(df)]
    at = _registry_app(df)
    at.session_state["task_mode"] = "prediction"
    at.run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    at.selectbox(key="target_selectbox").select("mixed_outcome").run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]

    errors = " ".join(str(e.value) for e in at.error)
    assert "cannot be used as a target" in errors, _page_text(at)[-2000:]
    # It must not repeat sklearn's misdiagnosis.
    assert "regression target with continuous values" not in errors, errors
    assert "storage problem" in errors, errors
    # Refusing means the configuration is not saved for downstream pages.
    config = _state(at, "data_config")
    assert config is None or config.target_col != "mixed_outcome"


# ── page 04 end to end, on the target that broke it ──────────────────────

def _run_selection(at):
    at.run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    button = next(b for b in at.button if "Run Feature Selection" in b.label)
    button.click().run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    return at


def test_drive064_page04_names_the_real_condition_not_a_continuous_target():
    at = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=120)
    _inject_prediction_state(at, boolean_target_frame())
    _run_selection(at)

    warnings = " ".join(str(w.value) for w in at.warning)
    assert "could not run" in warnings
    assert "regression target with continuous values" not in warnings, warnings
    assert "True/False" in warnings, warnings
    # The fix is named where the failure is reported.
    assert "Recode 'meds_hbp'" in warnings, warnings


def test_drive064_page04_zero_completions_is_not_a_green_banner():
    at = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=120)
    _inject_prediction_state(at, boolean_target_frame())
    _run_selection(at)

    successes = " ".join(str(s.value) for s in at.success)
    assert "0 methods run" not in successes, successes
    assert "Feature selection complete" not in successes, successes

    infos = " ".join(str(i.value) for i in at.info)
    assert "Only one method completed" not in infos, infos

    errors = " ".join(str(e.value) for e in at.error)
    assert "did not run" in errors, errors


def test_drive063_zero_completions_records_no_selection():
    at = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=120)
    _inject_prediction_state(at, boolean_target_frame())
    _run_selection(at)

    prov = _state(at, "workflow_provenance")
    recorded = prov.feature_selection if prov is not None else None
    assert recorded is None, (
        "a run where every method raised is not a feature selection; "
        f"recorded {recorded}")
    if prov is not None:
        ctx = prov.get_methods_context()
        assert "fs_method" not in ctx and "fs_consensus_methods" not in ctx, ctx

    actions = " ".join(e.get("action", "")
                       for e in _state(at, "methodology_log", []))
    assert "No selection method completed" in actions, actions


def test_drive063_provenance_records_the_methods_that_completed(monkeypatch):
    """Three were requested; one raised. The record must name only the survivors."""
    import ml.feature_selection as fs

    def _boom(*args, **kwargs):
        raise ValueError("Unknown label type: unknown.")

    monkeypatch.setattr(fs, "univariate_screening", _boom)

    df = boolean_target_frame()
    df["meds_hbp"] = repair_boolean_target(df["meds_hbp"])
    at = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=120)
    _inject_prediction_state(at, df)
    at.run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    univariate = next(c for c in at.checkbox if "Univariate" in c.label)
    univariate.check().run()
    button = next(b for b in at.button if "Run Feature Selection" in b.label)
    button.click().run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]

    prov = _state(at, "workflow_provenance")
    assert prov is not None and prov.feature_selection is not None, (
        "two methods completed and agreed — that is a selection worth recording")
    assert prov.feature_selection.consensus_methods == ["lasso", "rfe"], (
        "provenance was given the REQUESTED list, so the Methods draft claimed "
        "a consensus across a method that raised; "
        f"got {prov.feature_selection.consensus_methods}")
    assert prov.feature_selection.n_features_after > 0

    successes = " ".join(str(s.value) for s in at.success)
    assert "2 methods run" in successes, successes


def test_drive063_a_null_run_records_no_consensus():
    """One method cannot agree with itself, so there is no consensus to record."""
    df = boolean_target_frame()
    df["meds_hbp"] = repair_boolean_target(df["meds_hbp"])
    at = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=120)
    _inject_prediction_state(at, df)
    at.run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]
    rfe = next(c for c in at.checkbox if c.label.startswith("RFE-CV"))
    rfe.uncheck().run()
    button = next(b for b in at.button if "Run Feature Selection" in b.label)
    button.click().run()
    assert not at.exception, [str(e.value)[:400] for e in at.exception]

    infos = " ".join(str(i.value) for i in at.info)
    assert "Only one method completed" in infos, infos

    prov = _state(at, "workflow_provenance")
    recorded = prov.feature_selection if prov is not None else None
    assert recorded is None, (
        "an n_after=0 record reads back through narrative_engine's "
        "`n_after_sel or n_final` as 'retained all N candidate predictors'; "
        f"recorded {recorded}")
