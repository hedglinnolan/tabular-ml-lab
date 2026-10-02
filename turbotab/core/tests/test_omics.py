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
    ("impute", "pqn_log2", False),  # the missing-values answer fills them, and says so
    ("complete_case", "declared_normalized", False),  # nothing is logged
])
def test_a_log_never_hands_a_missing_value_to_a_family_that_cannot_take_one(missing, option, refused):
    folder = Path(tempfile.mkdtemp())
    frame, paths, features = _intensities_with_zeros(folder)
    n_zero = int((frame[features] == 0).to_numpy().sum())
    st = mf.state(lens=["metabolomics"], target="case", event="1", models=["elastic_net"],
                  missing=missing, roles={**{c: "exposure" for c in features}, "sample_id": "identifier"},
                  findings={"omics_scale": FindingDisposition(
                      action="applied", option=option,
                      params={"kind": "intensities", "columns": features, "n_zero": n_zero})})
    refusal = omics.zeros_refusal(["elastic_net", "boosted_trees"], st)
    assert (refusal is not None) == refused
    if refused:
        assert refusal.code == "zeros_cannot_be_logged" and "2 zero values" in refusal.message
        assert refusal.exits[1]["decision"]["models"] == ["boosted_trees"]
        assert omics.zeros_refusal(["boosted_trees"], st) is None  # it takes missing values
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, folds=5, seed=0)
    context = mf.context(st, {"split": split, "target_info": mf.target_info("binary", "case")}, paths)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if refused:
            with pytest.raises(ValueError, match="zero values in the assay columns cannot be logged"):
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
