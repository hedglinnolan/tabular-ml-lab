"""MC-2b-2's leftovers and ruling 14's units (WAVE_C6A_PLAN §3 MC-2b-2, §7 rulings 13 and 14).

* **The design's words** are each family's declaration (``InferenceDecl.words``), not two tables
  keyed by what the methods text calls the family. The reference is the two tables as they stood
  at ``turbotab-next`` 0b0b10ed, copied below, read the way the code read them (by
  ``methods_label``, the label itself when a table did not hold it), for every registered family
  and every task.
* **``class_predictor`` names its family**: a missing family no longer turns the multinomial
  logit's refinement off in silence.
* **``interaction.SUPPORTED``** is gone: the declaration (``product_terms``) is what is read, and
  test_mc1's check that the declaration agreed with it was circular.
* **Ruling 14:** the refits the interaction, the scales correction and the sensitivity analyses
  make are handed the units the split kept whole (``groups=``), as the fit stage's own fits are,
  so a tuned family's inner splits keep each person's rows on one side. The reference is the
  table's own person column, read by row id, and the inner splits a search draws, recorded as it
  draws them (``tuning.observing``).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.models import get_family, inner_cv, tuning
from turbotab.core.models.base import TASKS, families
from turbotab.core.tests import modeling_fixtures as mf

# ── models/survey.py's tables as they were (0b0b10ed), the reference ─────────

OLD_DESIGN_WORDS = {
    "linear regression": "least squares",
    "logistic regression": "logistic regression (pseudo-maximum likelihood)",
    "multinomial logistic regression":
        "multinomial logistic regression (pseudo-maximum likelihood)",
    "multinomial logistic regression, which ignores the levels' order":
        "multinomial logistic regression (pseudo-maximum likelihood)",
    "a proportional-odds (cumulative logit) model":
        "the proportional-odds model (pseudo-maximum likelihood)",
    "Cox proportional hazards": "Cox regression (Binder's pseudo-likelihood, Efron ties)",
}
OLD_NO_DESIGN_WORDS = {
    "a random-intercept mixed model": "the random-intercept mixed model",
    "generalized estimating equations": "the GEE model",
    "feature-wise least-squares tests with Benjamini–Hochberg false-discovery control":
        "feature-wise regression",
    "elastic net": "the elastic net",
    "gradient-boosted trees": "the gradient-boosted tree model",
}


def old_estimator_words(family: Any, task: str) -> str:
    label = family.methods_label(task)
    return OLD_DESIGN_WORDS.get(label) or label


def old_blocked_words(family: Any, task: str) -> str:
    label = family.methods_label(task)
    return OLD_NO_DESIGN_WORDS.get(label) or label


def _families() -> list[Any]:
    from turbotab.core.contracts import contracts

    contracts()  # the omics chain registers the screened elastic net
    return list(families())


def test_the_survey_words_are_the_old_tables_for_every_family_and_task():
    """Word for word, both functions, every registered family and every task (its own or not)."""
    from turbotab.core.models.survey import blocked_words, estimator_words

    pairs = [(f, t) for f in _families() for t in TASKS]
    assert len({f.key for f, _ in pairs}) >= 13
    for f, t in pairs:
        assert estimator_words(f, t) == old_estimator_words(f, t), (f.key, t)
        assert blocked_words(f, t) == old_blocked_words(f, t), (f.key, t)


def test_the_words_are_the_families_declarations_not_tables_keyed_by_label():
    """The tables are gone; each family that had words declares them, and a family that renames
    what the methods text calls it keeps its design words (the table, keyed by the old label,
    lost them in silence)."""
    from turbotab.core.models import survey
    from turbotab.core.models.ordinal import ProportionalOdds
    from turbotab.core.models.repeated import GEE

    assert not hasattr(survey, "_DESIGN_WORDS") and not hasattr(survey, "_NO_DESIGN_WORDS")
    declared = {f.key: f.inference_decl.words for f in _families()
                if f.inference_decl is not None and f.inference_decl.words}
    assert set(declared) == {"linear", "proportional_odds", "cox", "mixed", "gee",
                             "featurewise", "elastic_net", "boosted_trees"}

    renamed = type("Renamed", (ProportionalOdds,), {
        "key": "renamed_po", "methods_label": lambda self, task: "an ordered logit"})()
    assert survey.estimator_words(renamed, "ordinal") == (
        "the proportional-odds model (pseudo-maximum likelihood)")
    assert survey.design_label(renamed, "ordinal") == "the survey-weighted proportional-odds model"
    blocked = type("RenamedGEE", (GEE,), {
        "key": "renamed_gee", "methods_label": lambda self, task: "marginal models"})()
    assert survey.blocked_words(blocked, "regression") == "the GEE model"
    # A family that declares no design-based estimator is never called by design words, nor a
    # design-based one by its blocked name: each reads its words for the sentence it is in.
    po = get_family("proportional_odds")
    assert survey.blocked_words(po, "regression") == po.methods_label("regression")
    unblocked = replace(GEE.inference_decl, design_based=True)
    probe = type("Probe", (GEE,), {"key": "probe", "inference_decl": unblocked})()
    assert survey.blocked_words(probe, "regression") == probe.methods_label("regression")


# ── stages/class_substitution.py: the family is required ─────────────────────


def test_class_predictor_refuses_a_missing_family():
    import warnings

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from turbotab.core.stages.class_substitution import class_predictor

    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.normal(size=90)})
    y = np.array(["low", "mid", "high"])[np.arange(90) % 3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = Pipeline([("model", LogisticRegression(C=np.inf))]).fit(X, y)
    with pytest.raises(TypeError):
        class_predictor(pipeline, X, y, ["low", "mid", "high"])  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="needs the family"):
        class_predictor(pipeline, X, y, ["low", "mid", "high"], None)
    predict = class_predictor(pipeline, X, y, ["low", "mid", "high"], get_family("linear"))
    assert predict(X).shape == (90, 3)


# ── methods/interaction.py: the SUPPORTED shim is gone ───────────────────────


def test_the_interaction_reads_the_declaration_and_keeps_no_family_list():
    from turbotab.core.methods import interaction

    assert not hasattr(interaction, "SUPPORTED")
    assert {f.key for f in _families() if interaction.tests_product_terms(f)} == {
        "linear", "cox", "proportional_odds"}


# ── ruling 14: the refits keep people whole ──────────────────────────────────


class _Recorder:
    """Every ``inner_cv.fit_pipeline`` call (the groups handed and the rows' ids) and every inner
    split a search draws (``tuning.observing``), by the stage running."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.splits: list[Any] = []
        self.stage = "?"
        self.real = inner_cv.fit_pipeline

    def __call__(self, pipeline: Any, X: Any, y: Any, **kw: Any) -> Any:
        self.calls.append({"stage": self.stage, "groups": kw.get("groups"),
                           "tuned": type(pipeline).__name__ == "TunedPipeline",
                           "ids": np.asarray(X.index)})
        return self.real(pipeline, X, y, **kw)

    def drawn(self, found: Any) -> None:
        if found.kind == "inner":
            self.splits.append((self.stage, found))


def _assert_whole(recorder: _Recorder, stage: str, person: pd.Series, *, tuned: bool,
                  searched: bool) -> None:
    """Every refit ``stage`` made was handed each row's person (``person``, by row id) as its
    groups, and every inner split drawn there kept each person on one side. ``tuned``: a tuned
    family was among the refits; ``searched``: its search drew inner splits."""
    mine = [c for c in recorder.calls if c["stage"] == stage]
    assert mine, f"{stage} refit nothing"
    for c in mine:
        assert c["groups"] is not None, stage
        np.testing.assert_array_equal(np.asarray(c["groups"]), person.loc[c["ids"]].to_numpy())
    splits = [s for where, s in recorder.splits if where == stage]
    assert any(c["tuned"] for c in mine) == tuned
    assert bool(splits) == searched
    for found in splits:
        train = set(person.loc[np.asarray(found.train)])
        assert train and not train & set(person.loc[np.asarray(found.validation)])


def test_the_sensitivity_refits_keep_each_person_on_one_side_of_the_inner_splits(tmp_path):
    """Two rows a person, the split grouped by person: each sensitivity refit (the tuned ridge's
    search among them) is handed the person of each of its rows, and no inner split puts one of a
    person's rows in training and the other in validation."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.sensitivity import sensitivity_stage

    frame = mf.nhanes_like(240, seed=3, repeats=2)
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(240), seed=5, groups=frame["SEQN"].to_numpy(),
                            grouped_by="SEQN")
    state = mf.state(models=["linear", "ridge"], split=d.SplitSpec(holdout=0.2, seed=5),
                     sensitivity=[d.SensitivityAnalysis(label="under 70", rules=[
                         d.ExclusionRule(column="age", low=0, high=69, reason="under 70")])])
    inputs = {"split": split, "target_info": mf.target_info("regression")}
    design = design_stage(mf.context(state, inputs, paths))
    with DataStore(paths["data"], 1 << 30) as store:
        ingest = store.info().to_dict()
    recorder = _Recorder()
    recorder.stage = "sensitivity"
    with pytest.MonkeyPatch.context() as patch, tuning.observing(recorder.drawn):
        patch.setattr(inner_cv, "fit_pipeline", recorder)
        out = sensitivity_stage(mf.context(state, {**inputs, "design": design, "ingest": ingest},
                                           paths)).data
    assert out["families"]
    _assert_whole(recorder, "sensitivity", pd.Series(frame["SEQN"].to_numpy()), tuned=True,
                  searched=True)


def _probe_family() -> Any:
    from turbotab.core.tests.test_f15_every_refit import _probe_family as probe

    return probe()


def test_the_interaction_refits_keep_each_person_on_one_side_of_the_inner_splits(tmp_path):
    """Two rows a person, the split grouped by person: each refit the interaction makes (the
    linear family's, and a tuned family's with product terms, a probe) is handed the person of
    each of its rows. The probe's plan scores no candidates on these rows (it draws no inner
    split), so the groups each refit is handed are the check."""
    from turbotab.core.methods import interaction as ix
    from turbotab.core.models import base as B
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests.acceptance import estimand_fixtures as est_f

    probe = _probe_family()
    frame = est_f.cohort(n=320, seed=17)
    frame["pid"] = np.arange(320) // 2
    recorder = _Recorder()
    recorder.stage = "modification"
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(B._REGISTRY, probe.key, probe)
        st = est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio",
                         models=["linear", probe.key],
                         grain=d.GrainSpec(grain="repeated", id_column="pid"),
                         modifications={"sex": d.ModificationSpec(kind="effect_modification",
                                                                  exposure="fiber")})
        paths = mf.ingest_frame(frame, tmp_path)
        split = mf.split_bundle(np.arange(320), holdout=0.0, seed=st.split.seed,
                                groups=frame["pid"].to_numpy(), grouped_by="pid")
        info = mf.target_info(st.task, st.target)
        design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
        inputs = {"design": design, "split": split, "target_info": info}
        with tuning.observing(recorder.drawn):
            patch.setattr(inner_cv, "fit_pipeline", recorder)
            out = ix.modification_stage(mf.context(st, inputs, paths)).data
    (result,) = out["modifications"]
    assert {f["family"] for f in result["families"]} == {"linear", probe.key}
    _assert_whole(recorder, "modification", frame["pid"], tuned=True, searched=False)


def test_the_scales_correction_refits_are_handed_each_rows_person(tmp_path):
    """Two rows a person, the split grouped by person: the regression calibration's refit of the
    outcome model is handed the person of each of its rows (the linear family draws no inner
    split, so the groups it is handed are the whole check)."""
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.scales import scales_stage
    from turbotab.core.tests.acceptance.scales_fixtures import linear_scale_table

    items = [f"sat_{j}" for j in range(1, 9)]
    frame = linear_scale_table(n=300)
    frame["pid"] = np.arange(300) // 2
    roles = {"pid": "identifier", "age": "covariate", "bmi": "covariate",
             **{c: "covariate" for c in items}}
    st = mf.state(purpose="inference", target="sbp", task="regression", models=["linear"],
                  roles=roles, role_confirmations=dict(roles), lens=["clinical"],
                  grain=d.GrainSpec(grain="repeated", id_column="pid"),
                  split=d.SplitSpec(holdout=0.0, seed=7, folds=5), shape_confirmations={},
                  scales=[d.ScaleSpec(name="sat_score", items=items, reverse=["sat_3", "sat_6"],
                                      low=1, high=5, kind="reflective",
                                      correction="regression_calibration", n_boot=50)])
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(300), holdout=0.0, seed=7, groups=frame["pid"].to_numpy(),
                            grouped_by="pid")
    ti = mf.target_info("regression", "sbp")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    recorder = _Recorder()
    recorder.stage = "scales"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(inner_cv, "fit_pipeline", recorder)
        out = scales_stage(mf.context(st, {"design": design, "split": split, "target_info": ti},
                                      paths)).data
    assert out["scales"][0]["correction"] is not None
    _assert_whole(recorder, "scales", frame["pid"], tuned=False, searched=False)


def test_a_split_by_row_hands_no_groups():
    """The fit stage's reading: no grouping column, or one whose every value is its own unit (the
    rows were combined per unit), hands no groups."""
    from turbotab.core.stages.sensitivity import split_units, units_for

    frame = pd.DataFrame({"pid": [1, 2, 3], "x": [0.1, 0.2, 0.3]}, index=[10, 11, 12])
    assert split_units(frame, None) is None
    assert split_units(frame, "pid") is None
    frame["pid"] = [1, 1, 2]
    units = split_units(frame, "pid")
    assert units_for(units, frame.loc[[12, 10]]).tolist() == [2, 1]
    assert units_for(None, frame) is None
