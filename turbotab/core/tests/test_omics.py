"""The omics scale reading and its guards (``turbotab.core.methods.omics``, AUDIT_REPORT §5 WP11).

The acceptance tests (``acceptance/test_wp11_omics.py``) check the numbers; these check that the
reading stays quiet where it should and that a log never hands a missing value to a family that
cannot take one.
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import FindingDisposition
from turbotab.core.methods import omics
from turbotab.core.stages.modeling import design_stage
from turbotab.core.tests import modeling_fixtures as mf

SAMPLES = Path(__file__).resolve().parents[2] / "sample_data"


@pytest.mark.parametrize("name,lens,kind", [
    ("genomics_expression.csv", "genomics", "counts"),
    ("genomics_cpm.csv", "genomics", None),  # already per-sample normalized
    ("genomics_tmm_cpm.csv", "genomics", None),
    ("genomics_vst.csv", "genomics", None),
    ("genomics_microarray.csv", "genomics", None),
    ("metabolomics_untargeted.csv", "metabolomics", "intensities"),
    ("metabolomics_paired_logged.csv", "metabolomics", None),  # already on a log scale
    ("nhanes_dietary.csv", "dietary", None),  # no omics lens, no reading
])
def test_the_reading_names_raw_values_and_stays_quiet_on_normalized_ones(name, lens, kind):
    frame = pd.read_csv(SAMPLES / name)
    reading = omics.scale_reading(frame, [lens])
    assert (reading.kind if reading else None) == kind


def test_genotype_dosages_and_two_level_columns_are_not_counts():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.integers(0, 3, (80, 40)), columns=[f"rs{i}" for i in range(40)])
    assert omics.scale_reading(frame, ["genomics"]) is None


def _intensities_with_zeros(folder: Path) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(np.exp(rng.normal(8, 1, (60, 30))), columns=[f"mz_{i}" for i in range(30)])
    frame.iloc[2, 4] = 0.0
    frame.iloc[9, 7] = 0.0
    frame.insert(0, "sample_id", [f"U{i}" for i in range(60)])
    frame["case"] = np.repeat([0, 1], 30)
    features = [c for c in frame.columns if c.startswith("mz_")]
    return frame, mf.ingest_frame(frame, folder), features


@pytest.mark.parametrize("missing,option,refused", [
    ("complete_case", "pqn_log2", True),
    ("complete_case", "log2", True),
    # MS7: a fill for missing values never takes a zero a log turned missing; it routes to the
    # detection-limit question first (this case was accepted before the modeling-sequence review).
    ("impute", "pqn_log2", True),
    ("complete_case", "declared_normalized", False),  # nothing is logged
])
def test_a_zero_a_log_would_turn_missing_routes_to_the_detection_limit_question(missing, option, refused):
    folder = Path(tempfile.mkdtemp())
    frame, paths, features = _intensities_with_zeros(folder)
    n_zero = int((frame[features] == 0).to_numpy().sum())
    st = mf.state(lens=["metabolomics"], target="case", event="1", models=["elastic_net"],
                  missing=missing, roles={**{c: "exposure" for c in features}, "sample_id": "identifier"},
                  findings={"omics_scale": FindingDisposition(
                      action="applied", option=option,
                      params={"kind": "intensities", "columns": features, "n_zero": n_zero,
                              "zero_columns": {"mz_4": 1, "mz_7": 1}})})
    refusal = omics.zeros_refusal(["elastic_net", "boosted_trees"], st)
    assert (refusal is not None) == refused
    if refused:
        assert refusal.code == "zeros_cannot_be_logged" and "2 zero values in 2 assay columns" in refusal.message
        assert refusal.exits[0]["label"].startswith("Zeros mean not detected")
        assert refusal.exits[1]["decision"]["option"] == "declared_normalized"
        # whatever the families: one that takes missing values would read the zero as missing too
        assert omics.zeros_refusal(["boosted_trees"], st) is not None
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, folds=5, seed=0)
    context = mf.context(st, {"split": split, "target_info": mf.target_info("binary", "case")}, paths)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if refused:
            with pytest.raises(ValueError, match="zero values in 2 assay columns cannot be logged"):
                design_stage(context)
        else:
            design_stage(context)


def test_the_library_size_check_names_the_event_and_reads_the_training_rows_only():
    """The fit stage codes the named event 1; the sentence names it and its AUC is the event's."""
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 20)
    totals = np.exp(rng.normal(0, 0.1, 40)) * np.where(y == 1, 1.5, 1.0)
    check = omics.library_size_check(totals, y, "binary", "counts", target="condition", event="tumor")
    assert check["flagged"] and check["auc"] > 0.9 and check["event"] == "tumor"
    assert "as large in `tumor`" in check["sentence"]
    quiet = omics.library_size_check(np.exp(rng.normal(0, 0.1, 40)), y, "binary", "counts")
    assert not quiet["flagged"]


def test_a_feature_wise_table_never_offers_complete_cases_that_leave_nothing_to_fit():
    """DoD §1, no dead end: on the untargeted metabolomics fixture no row holds every feature, so
    the "Complete cases" way forward the feature-wise refusal once offered left 0 rows and the
    design stage threw (“Found array with 0 sample(s)”). It is offered only when complete cases
    keep rows enough for the design, with their count; the refusal names exactly what is offered."""
    from types import SimpleNamespace

    from turbotab.core.decisions import SetMissing
    from turbotab.core.methods.exposure_form import SPLINE_MIN_VALUES

    frame = pd.read_csv(SAMPLES / "metabolomics_untargeted.csv")
    feats = [c for c in frame.columns if c.startswith("mz_")]
    X = frame[["age", "bmi", *feats]]
    assert omics.complete_rows(X) == 0
    censored = SimpleNamespace(missing=SimpleNamespace(drop_columns=[], censored_columns=feats[:3]))
    plain = SimpleNamespace(missing=SimpleNamespace(drop_columns=["bmi"], censored_columns=[]))
    for state, way in ((censored, "censoring_aware"), (plain, "single_fill")):
        reason, exits = omics.unpooled_refusal("Feature-wise regression", state, X)
        assert [e["decision"]["strategy"] for e in exits] == ["impute"], exits
        assert "Complete cases are not offered: 0 of the 80 rows hold every value" in reason
        words = omics.EXIT_WORDS[way]
        assert reason.endswith(f" {words[:1].upper()}{words[1:]}."), reason
        for e in exits:  # each way forward is an answer the decision log accepts
            SetMissing(**{k: v for k, v in e["decision"].items() if k != "kind"})
            assert e["decision"]["acknowledged"] is True
    assert exits[0]["decision"]["drop_columns"] == ["bmi"]
    # with rows enough, complete cases lead, counted; without a frame, as before
    some = X.copy()
    some.loc[some.index[:SPLINE_MIN_VALUES], :] = 1.0
    reason, exits = omics.unpooled_refusal("Feature-wise regression", censored, some)
    assert exits[0]["label"] == f"Complete cases ({SPLINE_MIN_VALUES} of the 80 rows)"
    assert [e["decision"]["strategy"] for e in exits] == ["complete_case", "impute"]
    assert reason.endswith("Choose complete cases, or fill the values below detection once, "
                           "censoring-aware, recorded as a limitation."), reason
    few = X.copy()
    few.loc[few.index[:SPLINE_MIN_VALUES - 1], :] = 1.0
    assert [e["decision"]["strategy"] for e in omics.featurewise_missing_exits(censored, few)] == \
        ["impute"]
    assert [e["label"] for e in omics.featurewise_missing_exits(censored)][0] == "Complete cases"
