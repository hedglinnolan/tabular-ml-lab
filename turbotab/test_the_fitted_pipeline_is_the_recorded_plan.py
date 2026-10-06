"""`GUIDED-095` / `GUIDED-089` — a recorded decision reaches the thing it was
recorded for.

## The defect, in one sentence

Clause §06 splits preprocessing in two: row-local operations, which may run
now, and stateful ones, *recorded now and fitted inside each training fold*.
The first class reached the model because executing rewrites `working_table`
and the trainer read `working_table`. **The second class reached nothing** —
the record was written, the receipt counted it, the sentence was composed, and
the executor did not exist.

## What this file asserts, and in what order

**Hardest first**, because that is where the abstraction bends: `leave` and
`indicator` on a model that cannot read a blank. It is the one case where
per-model divergence is forced, and if the plan survives it the rest is
fill-out.

1. A declaration that keeps the blank keeps it **in the fitted transformer** —
   read off the fitted `ColumnTransformer`, not off the payload.
2. The same declaration on a linear model **diverges, and says so**, per model,
   naming both the recorded sentence and the one now true of the fit.
3. The blank survives into a fit where it actually reaches the estimator, which
   the case in (1) does not on its own fixture — see the correction below.
4. **The sentence and the pipeline are one object**, asserted as identity: the
   plan's step carries the record's own string, not a copy that agrees today.
5. A recipe variant changes the fitted transformer AND the methods sentence.
6. A deferred transform is fitted, at last, and its sentence is the record's.
7. Where the plan cannot honor something it **refuses** rather than
   substituting — a decision the record accepts and the pipeline silently drops
   is the defect this module was written to remove, arriving from inside.

## A correction to `GUIDED-089`'s own measurement, made while building the gate

The row records: *"`metabolomics_untargeted.csv`, column `bmi`, 8 blanks of 80
… `_pipeline` puts `bmi` in the numeric block and `SimpleImputer` fills those 8
blanks with the median, 27.15."* The first half is exactly right and is the
defect. **The second half is not true on that fixture**, and it matters that
the report says so: all 8 blank-`bmi` rows are `pooled_qc` samples with no
`responder`, so `train`'s outcome mask drops them from both partitions before
the pipeline sees a row. Measured: 8 blanks in the frame, 0 in `X_train`, 0 in
`X_test`.

What was real was the **fidelity** defect, and it was real for every column
whose blanks do sit in modeled rows — `mz_0003` has 42 — and for `bmi` in the
sense that mattered: the recorded methods sentence said the value is left blank
while the pipeline placed the column in the median-impute block, so the two
disagreed about what the analysis was. That is why (3) exists beside (1).
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import pipeline_plan, training                            # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


def _partitions(project):
    table = project.working_table
    target = str(project.target)
    sealed = set(project.lockbox["labels"])
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    has_y = table[target].notna()
    features = training._feature_frame(table, target,
                                       (project.grain or {}).get("group_col"))
    X_train = features[has_y & ~is_test]
    return features, X_train, table.loc[X_train.index, target]


def _fit(project, model_key, task):
    from ml.model_registry import get_registry

    features, X_train, y_train = _partitions(project)
    plan = pipeline_plan.compose(project, model_key, features, seed=42)
    pipe = plan.build(get_registry()[model_key].factory(task, 42))
    pipe.fit(X_train, y_train)
    return plan, pipe, features


def _block_of(prep, column):
    """Which fitted block a column was routed into."""
    return [entry[0] for entry in prep.transformers_
            if column in list(entry[2])]


# ── 4 · what it refuses ──────────────────────────────────────────────────────


def _assert_every_variant_builds(_rec):
    checked = 0
    for operation in _rec.operations():
        for variant in operation.variants:
            if operation.key == "scale":
                pipeline_plan._scaler(variant)
            elif operation.key == "encode":
                pipeline_plan._encoder(variant, 0)
            elif operation.key == "power":
                pipeline_plan._power(variant)
            elif operation.key == "outliers":
                pipeline_plan._outliers(variant)
            else:                                    # a pack added an operation
                pytest.fail(
                    f"{operation.key} is in the recipe table (from "
                    f"{operation.origin}) and this module has no builder for "
                    f"it, so choosing it would be a decision the fit silently "
                    f"drops")
            assert pipeline_plan._recipe_sentence(
                operation.key, variant, "a model", 3) is not None
            checked += 1
    assert checked >= 14, checked


def test_every_stateful_transform_in_the_catalogue_has_a_fitted_form():
    """The same completeness question for the Features step. `features.py`
    marks six transforms `STATEFUL` — *recorded now, fitted in-fold* — and
    until this loop none of them had an in-fold."""
    from turbotab import features as _feat

    deferred = _feat.deferred_keys()
    assert len(deferred) >= 6, deferred
    for key in deferred:
        made = pipeline_plan._deferred_transformer(
            {"key": key, "params": {"n_bins": 3, "n_components": 2}}, 0)
        assert hasattr(made, "fit"), key
