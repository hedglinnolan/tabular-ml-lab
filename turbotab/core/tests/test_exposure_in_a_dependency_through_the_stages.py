"""An exposure in a dependency of the model matrix's columns, through the real design, fit, effects
and mattered stages: the fit's table is refused whole in the dependency's words, the effects
stage's methods sentence gives those words, and the mattered stage's refusal gives them and the
way forward, leaving one of the dependent columns out (T1V-1, 2026-10-27).

The tail tables package's first round served a dependent design's coefficient table by row, and
the fit stage names no study factor to the table (``stages.modeling._inference_table`` passes no
``study``), so with `fiber2` = 2·`fiber` beside `fiber` the fit stage served a table with `fiber`'s
row empty and every covariate's estimated. The effects stage then read "The estimate of `fiber` is
reported across a declared sequence of models fit on all 400 analyzed rows", and the mattered
stage refused because "The linear model reports no single coefficient of `fiber`", offering to
declare the columns' scales. A table whose caller names nothing studied now refuses any
dependency whole (``inference.by_identified``), as the base did; by row it is served only where
the caller names what is studied (the specification curve does, and since T5b every stage,
``effects.study_columns``), and only where no column named so is in the dependency: here
`fiber` is, so the table is refused whole (the covariate's case:
``test_stages_name_the_study_factor``).

Truth from construction: `fiber2` is twice `fiber`, so they and nothing else are the dependency,
beside the intercept; the words are written out from that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from turbotab.core import mattered as M
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import estimand_fixtures as ef

N = 400
WORDS = (f"On these {N} rows the model matrix is singular: `fiber` and `fiber2` are fixed "
         f"combinations of the other columns and the intercept, so their coefficients are not "
         f"identified. No estimate, interval or p-value is reported; leave one of the dependent "
         f"columns out.")
LEAVE = "Leave one of the dependent columns out of the model"


def _through_the_stages(tmp_path: Path, target: str, task: str, measure: str,
                        event: str | None = None) -> dict:
    """design → fit → effects → mattered on the cohort with `fiber2` = 2·`fiber` a covariate."""
    from turbotab.core.stages.effects import effects_stage
    from turbotab.core.stages.modeling import design_stage, fit_stage

    frame = ef.cohort(n=N, seed=7).drop(columns=["bmi"])
    frame["fiber2"] = 2 * frame["fiber"]
    roles = {k: v for k, v in ef.ROLES.items() if k != "bmi"} | {"fiber2": "covariate"}
    answers = {k: v for k, v in ef.ANSWERS.items() if k != "bmi"} | {"fiber2": ef.CONFOUNDER}
    st = ef.state(target=target, task=task, event=event, measure=measure, roles=roles,
                  answers=answers, plan_locked=True)
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(N), holdout=0.0, seed=st.split.seed)
    info = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    inputs = {"design": design, "split": split, "target_info": info}
    fit = fit_stage(mf.context(st, inputs, paths))
    effects = effects_stage(mf.context(st, inputs, paths)).data
    mattered = M.mattered_stage(mf.context(st, {**inputs, "fit": fit, "ingest": {}}, paths))
    return {"fit": fit.data["models"][0], "effects": effects,
            "mattered": getattr(mattered, "data", mattered)}


@pytest.mark.parametrize(("target", "task", "measure", "event"), [
    ("glucose", "regression", "mean_difference", None),
    ("dm", "binary", "odds_ratio", "yes")])
def test_the_exposure_in_a_dependency_is_refused_in_its_words_at_every_stage(
        tmp_path, target, task, measure, event):
    """The fit's table has no coefficient and is refused in the dependency's words; the effects
    methods sentence says Model 2, the primary, reports no estimate of `fiber`, in those words,
    and that the unadjusted model alone reports it (T4a's V4a-2 sentence, merged after this test
    was written: the base's assertion that it began "No estimate of `fiber` is reported" was
    stale there); the mattered stage refuses in them, its way forward to leave a dependent column
    out. On the first round the fit table served the covariates by row and both sentences said
    otherwise."""
    out = _through_the_stages(tmp_path, target, task, measure, event)
    assert out["fit"]["inference"]["refused"] == WORDS
    assert out["fit"]["coefficients"] == []
    assert out["effects"]["methods"] == (
        f"The estimate of `fiber` is reported from the unadjusted model alone, fit on all {N} "
        f"analyzed rows. Model 2, the primary, adjusted for `age`, `sex`, `smoking`, `activity` "
        f"and `fiber2`, reports none: {WORDS}")
    refused = out["mattered"]["refused"]
    assert WORDS in refused["message"]
    assert [e["label"] for e in refused["exits"]] == [LEAVE]
