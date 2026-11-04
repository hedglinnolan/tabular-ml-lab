"""Under multiple imputation the effects stage pools each declared model over the completed copies;
where one copy's table is refused, the model is refused, in that copy's words, as the fit's pooled
table is (the verifier's FWMI-R3-3).

Before this repair ``effects._Run._pool`` returned the first copy's table whenever any copy's was
refused, so a refusal in a later copy served copy 0's unpooled estimate as the model's: its df the
complete-data t(394), its caption with no multiple-imputation clause, beside a fit that refused the
same model.

Truth from construction: `activity` (20% blank) is imputed by chained equations, and in the third
copy only it is set to 2 × `fiber` + 3, so that copy's Model 2 design (`fiber` and `activity`
among its columns) is singular (numpy's rank) and the other copies' are not. The unadjusted model
holds no `activity` and pools in every copy (the guard silent), and so does Model 2 where no copy
is changed.
"""
from __future__ import annotations

import numpy as np
import pytest

from turbotab.core.tests.acceptance import estimand_fixtures as ef

DEPENDENCY = ("On these 400 rows the model matrix is singular: `activity` and `fiber` are fixed "
              "combinations of the other columns and the intercept, so their coefficients are not "
              "identified. No estimate, interval or p-value is reported; leave one of the "
              "dependent columns out.")
MODEL_2 = ["age", "sex", "smoking", "activity"]


def _run(tmp_path, monkeypatch, singular: bool) -> tuple[dict, list]:
    from turbotab.core.methods import missing as ms

    frame = ef.cohort(n=400, seed=7)
    frame.loc[np.random.default_rng(11).random(400) < 0.2, "activity"] = np.nan
    real = ms.impute_for_inference
    drawn: list = []

    def third_singular(spec, X, y, task, **kwargs):
        out = real(spec, X, y, task, **kwargs)
        if singular:
            out.frames[2]["activity"] = 2 * out.frames[2]["fiber"] + 3
        drawn.append(out)
        return out

    monkeypatch.setattr(ms, "impute_for_inference", third_singular)
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  missing={"strategy": "multiple_imputation", "m": 20})
    return ef.run(frame, tmp_path, st), drawn


@pytest.mark.parametrize("singular", [True, False], ids=["third_copy_singular", "none"])
def test_a_model_one_later_copy_refuses_is_refused_never_served_from_copy_0(tmp_path, monkeypatch,
                                                                           singular):
    out, drawn = _run(tmp_path, monkeypatch, singular)
    copies = drawn[-1].frames
    m = len(copies)
    assert m >= 20
    for k, copy in enumerate(copies):  # the premise: only the third copy's Model 2 is singular
        X = ef.design_matrix(copy, ["fiber", *MODEL_2]).to_numpy(dtype=float)
        assert (np.linalg.matrix_rank(X) < X.shape[1]) == (singular and k == 2), k
    fit = next(e for e in out["fit"]["models"] if e["family"] == "linear")
    seq = ef.sequence(out["effects"])
    crude = seq["crude"]
    assert crude["inference"]["refused"] is None
    assert crude["inference"]["caption"].startswith(f"Multiple imputation, m = {m}: ")
    primary = seq["model_2"]
    if singular:
        assert fit["inference"]["refused"] == DEPENDENCY
        assert primary["inference"]["refused"] == DEPENDENCY
        assert not primary["effects"]
        assert [e["label"] for e in primary["inference"]["exits"]] == [
            "Leave one of the dependent columns out of the model"]
        assert f"Model 2, the primary, adjusted for `age`, `sex`, `smoking` and `activity`, " \
               f"reports none: {DEPENDENCY}" in out["effects"]["methods"]
        return
    assert fit["inference"]["refused"] is None
    assert primary["inference"]["refused"] is None
    assert primary["inference"]["caption"].startswith(f"Multiple imputation, m = {m}: ")
    (effect,) = primary["effects"]
    assert effect["df"] < 400 - 7 and effect["fmi"] is not None
