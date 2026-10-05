"""MODELING_SEQUENCE §6, chains 3 and 5, end to end through the app's real stage graph.

BLUEPRINT §13: "A chain test asserts that every implied consequence appears in the participant
flow, the lineage and the methods sentence." Each chain here runs the real stages in-process
(``GraphRun``: ingest → oriented → findings → structure → working → cohort → split → design →
fit), records its answers through the decision validators the server runs, and asserts:

* the run order of MODELING_SEQUENCE §1.1 (before the seal on reference rows; in each training
  fold) in the working table's record, the participant flow and the design's steps;
* that every relation of §2 the chain touches fires, and where its consequence shows;
* the leash rows of §4 it touches, refused or blocked with their exits;
* the methods paragraph the run writes, verbatim, its first sentence the reviewers' own
  (``docs/turbotab-next/audit/modeling-sequence-review.json``, chains 3 and 5).

Expected counts come from the data, computed another way (pandas over the table and the working
table), never from the code under test.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import (ApplyRepair, ProjectState, Refusal, SelectModels, SetBatch,
                                     SetMissing, SetMultiplicity, validate)
from turbotab.core.methods import omics
from turbotab.core.methods import qc_drift as Q
from turbotab.core.contracts import fired, run_order
from turbotab.core.stages.modeling import read_assignment
from turbotab.core.tests.acceptance.omics_chains import genomics_batches, metabolomics_run
from turbotab.core.tests.graph_runner import GraphRun

CHAIN_3_REVIEWERS = (
    "Drift was corrected per batch by QC-RLSC fitted to pooled QCs only, which were then removed; "
    "within each training fold, PQN, detection-limit-aware imputation, log transformation and "
    "autoscaling were fitted and applied to the held-out fold; elastic-net parameters were tuned "
    "in an inner CV nested in an outer CV.")
CHAIN_5_REVIEWERS = "Batch was included as a covariate; ComBat was used for visualization only."


def data(artifact: Any) -> Any:
    return getattr(artifact, "data", artifact)


def ctx_of(state: ProjectState, frame: pd.DataFrame, out: dict) -> dict:
    """The validators' context, as the server builds it: the columns, their summaries, the state
    before the answer, and the run's fresh artifacts."""
    info = {c["name"]: c for c in data(out["ingest"])["columns"]}
    return {"state": state, "columns": list(frame.columns), "column_info": info,
            "artifact": lambda name: data(out[name]) if name in out else None}


def steps_of(design: Any, family: str) -> list[str]:
    model = next(m for m in data(design)["models"] if m["family"] == family)
    return [s["key"] for s in model["steps"]]


# ── chain 3 · metabolomics prediction ────────────────────────────────────────


@pytest.fixture(scope="module")
def chain3(tmp_path_factory):
    """The chain run once: the answers recorded through the validators, then the whole graph."""
    folder = tmp_path_factory.mktemp("chain3")
    frame, truth = metabolomics_run()
    path = folder / "run.csv"
    frame.to_csv(path, index=False)
    run = GraphRun(path, folder / "project")
    base = {"lens": ["metabolomics"], "target": "case", "purpose": "prediction", "task": "binary",
            "event": "1", "grain": {"grain": "repeated", "id_column": "participant_id"},
            "unit": "row", "repeat_kind": {"repeat_kind": "repeats"}}
    feats = [c for c in frame.columns if c.startswith("mz_")]
    roles = {c: "exposure" for c in feats}
    roles.update({"participant_id": "identifier", "sample_id": "excluded",
                  "injection_order": "excluded", "batch": "excluded", "sample_type": "excluded"})
    first = ProjectState.model_validate({**base, "roles": roles})
    out = run.run(first, upto=["findings"])
    findings = {f["id"]: f for f in data(out["findings"])["findings"]}
    ctx = ctx_of(first, frame, out)
    record: dict[str, Any] = {"refusals": {}}
    qc = next(o for o in findings[Q.FINDING]["repairs"] if o["key"] == "qc_rlsc_lc")
    scale = next(o for o in findings["omics_scale"]["repairs"] if o["key"] == "pqn_log2")
    for option in (qc, scale):
        validate(ApplyRepair(**{k: v for k, v in option["decision"].items() if k != "kind"}), ctx)
    censored = list(findings["pack::metabolomics::left_censored"]["censored_columns"])
    applied = ProjectState.model_validate({**base, "roles": roles, "findings": {
        Q.FINDING: {"action": "applied", "option": "qc_rlsc_lc", "params": qc["decision"]["params"]},
        "omics_scale": {"action": "applied", "option": "pqn_log2",
                        "params": scale["decision"]["params"]}}})
    ctx = ctx_of(applied, frame, out)
    with pytest.raises(Refusal) as refused:  # the leash: non-detections never median-filled silently
        validate(SetMissing(strategy="impute"), ctx)
    record["refusals"]["median"] = refused.value
    answer = refused.value.exits[0]["decision"]
    assert answer["below_detection"] == "censoring_aware"
    missing = SetMissing(**{k: v for k, v in answer.items() if k != "kind"})
    validate(missing, ctx)
    state = ProjectState.model_validate({
        **applied.model_dump(exclude_none=True),
        "missing": missing.model_dump(exclude={"kind"}),
        "split": {"holdout": 0.0, "folds": 5, "seed": 0, "validation": "kfold"},
        "models": ["elastic_net"]})
    validate(SelectModels(models=["elastic_net"]), ctx_of(state, frame, out))
    out = run.run(state, upto=["fit"])
    yield {"frame": frame, "truth": truth, "state": state, "out": out, "qc": qc,
           "censored": censored, "record": record, "feats": feats}
    run.close()


def test_chain_3_before_the_seal_qc_rlsc_runs_on_the_pooled_qcs_and_they_leave_the_cohort(chain3):
    """§1.1 before the seal, on reference rows only: the QC detection filter, QC-RLSC and the QC
    RSD filter, in that order, then the QC rows leave the working table as reference rows (WP18,
    RO-13), their corrected values kept beside it for quality assessment. The participant flow
    counts the 22 pooled QC injections on a line of their own, before the outcome is read (§2: "QC
    drift correction implies that the QC rows leave"); no QC row is among the rows the folds are
    drawn from (the working table's row map names each row's source row).

    References (pandas over the table, not the app): the QC rows are the ``sample_type`` = QC
    rows; the features the detection filter removes are those detected in fewer than 70% of the QC
    injections of the raw table; the RSD filter's are those whose corrected QC RSD in the working
    table is 20% or more. Drift is gone: the corrected QCs' median RSD in the working table is at
    the noise level (at most 8%), against more than twice that in the raw table."""
    frame, out = chain3["frame"], chain3["out"]
    feats = chain3["feats"]
    qc = frame["sample_type"].eq("QC").to_numpy()
    record = data(out["working"])["qc_correction"]
    assert record["steps"] == ["qc_detection_filter", "qc_rlsc", "qc_rsd_filter"]
    rate = frame.loc[qc, feats].gt(0).sum() / int(qc.sum())
    assert set(record["dropped"]["detection"]) == {c for c in feats if not rate[c] >= 0.70}
    qc_rows = pd.read_parquet(out["working"].files[Q.QC_ROWS_FILE]).set_index("__row_id")
    assert sorted(qc_rows.index) == list(np.flatnonzero(qc))
    kept = [c for c in feats if c not in set(record["dropped"]["detection"])]
    corrected_qc = qc_rows[kept].where(qc_rows[kept] > 0)
    rsd = 100 * corrected_qc.std(ddof=1) / corrected_qc.mean()
    assert set(record["dropped"]["rsd"]) == {c for c in kept if not rsd[c] < 20}
    raw_qc = frame.loc[qc, kept].where(frame.loc[qc, kept] > 0)
    raw_rsd = 100 * raw_qc.std(ddof=1) / raw_qc.mean()
    assert float(rsd.median()) <= 8.0 and float(raw_rsd.median()) > 2 * float(rsd.median())
    steps = data(out["cohort"])["steps"]
    assert [s["key"] for s in steps][:3] == ["loaded", "reference:0", "outcome_measured"]
    assert steps[0]["n"] == len(frame)
    assert steps[1]["label"] == "`sample_type` is not `QC`"
    assert steps[1]["dropped"] == int(qc.sum()) == 22
    assignment = read_assignment(out["split"])
    source = pd.read_parquet(out["working"].files["row_map.parquet"]).set_index("row_id")
    drawn = set(source.loc[assignment.index, "source_row_id"])
    assert len(drawn) == len(assignment) and not drawn & set(np.flatnonzero(qc))
    assert data(out["split"])["n_train"] == int((~qc).sum())


def test_chain_3_in_each_fold_pqn_then_the_detection_fill_then_the_log_then_autoscaling(chain3):
    """§1.1 in each training fold: PQN with a study-sample reference, detection-limit handling, log,
    scaling. The design's steps follow the contracts' run order, the folds keep each participant's
    repeat samples together (grouped folds), and the elastic net tunes its penalty by its own inner
    cross-validation inside each outer fold (nested)."""
    out, state = chain3["out"], chain3["state"]
    keys = steps_of(out["design"], "elastic_net")
    assert keys == ["normalize", "detect", "log", "impute", "scale", "model"]
    choices = omics.chain_choices(state, keys, "elastic_net")
    in_fold = [k for k in run_order(choices) if k in ("omics_normalization", "detection_limit",
                                                       "log_transform", "autoscaling")]
    assert in_fold == ["omics_normalization", "detection_limit", "log_transform", "autoscaling"]
    position = {"omics_normalization": keys.index("normalize"), "detection_limit": keys.index("detect"),
                "log_transform": keys.index("log"), "autoscaling": keys.index("scale")}
    assert [position[k] for k in in_fold] == sorted(position.values())
    split = data(out["split"])
    assert split["grouped_by"] == "participant_id"
    assignment = read_assignment(out["split"])
    # The working table's rows are numbered after the QC rows left; its row map names each source row.
    source = pd.read_parquet(out["working"].files["row_map.parquet"]).set_index("row_id")
    people = chain3["frame"]["participant_id"].reindex(
        source.loc[assignment.index, "source_row_id"].to_numpy())
    assert people.groupby(assignment["fold"].to_numpy()).apply(set).pipe(
        lambda s: all(not (a & b) for i, a in enumerate(s) for b in list(s)[i + 1:]))
    model = next(m for m in data(out["design"])["models"] if m["family"] == "elastic_net")
    assert "inner cross-validation within the training rows it is given" in model["steps"][-1]["detail"]
    pipeline = out["design"].objects["pipelines"]["elastic_net"]
    assert int(pipeline[-1].cv) >= 3
    fit = data(out["fit"])["models"][0]
    assert len(fit["cv"]["auc"]["folds"]) == 5 and 0.5 < fit["cv"]["auc"]["estimate"] <= 1.0


def test_chain_3_every_relation_it_touches_fires_and_shows(chain3):
    """§2's relations this chain touches, fired from the contracts the run used, each with where
    it shows: QC drift correction implies that the QC rows leave (the participant flow's line) and
    precedes every in-fold normalization (the working table's record before the design's steps);
    normalization precedes the detection fill, which precedes the log (the design's step order);
    a detection-limit reading implies censoring-aware handling (the recorded answer and the
    detect step). The leash row it touches: the median fill of non-detections was refused, with the
    censoring-aware fill as the first exit."""
    out, state = chain3["out"], chain3["state"]
    keys = steps_of(out["design"], "elastic_net")
    choices = omics.chain_choices(state, keys, "elastic_net")
    fires = {(f.source, f.relation.kind, f.relation.target)
             for f in fired(choices, "prediction", consequences=["qc_rows_leave", "censoring_aware"])}
    assert {("qc_rlsc", "implies", "qc_rows_leave"),
            ("qc_rlsc", "precedes", "omics_normalization"),
            ("qc_rlsc", "precedes", "detection_limit"),
            ("omics_normalization", "precedes", "detection_limit"),
            ("detection_limit", "precedes", "log_transform"),
            ("detection_limit", "implies", "censoring_aware")} <= fires
    assert "reference:0" in [s["key"] for s in data(out["cohort"])["steps"]]
    assert state.missing.below_detection == "censoring_aware"
    detail = next(m for m in data(out["design"])["models"] if m["family"] == "elastic_net")["steps"][1]
    assert detail["key"] == "detect" and "censored-normal" in detail["detail"]
    refused = chain3["record"]["refusals"]["median"]
    assert refused.code == "median_below_detection"
    named = [c for c in chain3["censored"] if (state.roles or {}).get(c) == "exposure"]
    assert refused.exits[0]["decision"]["censored_columns"] == named


def test_chain_3_writes_the_reviewers_methods_sentence(chain3):
    """The methods paragraph the run writes, verbatim. Its first sentence is the reviewers' own for
    chain 3; the details carry this run's counts, each computed here from the data: the QC
    injections, the features each QC filter removed (pandas over the raw and the working table),
    the censored features the fill covers, and the grouping of the folds."""
    out, state, frame, feats = chain3["out"], chain3["state"], chain3["frame"], chain3["feats"]
    keys = steps_of(out["design"], "elastic_net")
    said = omics.methods_paragraph(state, keys, "elastic_net", data(out["split"]),
                                   data(out["working"]))
    assert said.startswith(CHAIN_3_REVIEWERS + " ")
    qc = frame["sample_type"].eq("QC").to_numpy()
    rate = frame.loc[qc, feats].gt(0).sum() / int(qc.sum())
    detection = sum(1 for c in feats if not rate[c] >= 0.70)
    record = data(out["working"])["qc_correction"]
    rsd = len(record["dropped"]["rsd"])
    remain = len(feats) - detection - rsd - len(record["dropped"]["uncorrectable"])
    censored = len(state.missing.censored_columns)
    expected = (
        f"{CHAIN_3_REVIEWERS} QC-RLSC (Dunn et al. 2011) fitted, for each feature and each of the 2 "
        f"batches, a LOESS of degree 2 over injection order to the 22 pooled QC injections, its "
        f"span chosen by leave-one-out cross-validation; each injection was divided by the curve "
        f"and rescaled to the feature's median QC value. Features detected in fewer than 70% of QC "
        f"injections ({detection}), that could not be corrected (0), or with a QC RSD of 20% or more "
        f"after correction ({rsd}; the LC-MS criterion) were removed (Broadhurst et al. 2018); "
        f"{remain} of {len(feats)} remain. Values below detection in {censored} features were filled "
        f"after normalization and before the log, each by its expected value below the limit under "
        f"a left-censored normal fitted to the feature's logarithm on the training fold (Lubin et "
        f"al. 2004). Folds kept each `participant_id`'s rows together.")
    assert said == expected


# ── chain 5 · genomics, p ≫ n, and its inference twin ───────────────────────


@pytest.fixture(scope="module")
def chain5(tmp_path_factory):
    folder = tmp_path_factory.mktemp("chain5")
    frame, truth = genomics_batches()
    path = folder / "counts.csv"
    frame.to_csv(path, index=False)
    run = GraphRun(path, folder / "project")
    genes = [c for c in frame.columns if c.startswith("ENSG")]
    yield {"run": run, "frame": frame, "truth": truth, "genes": genes}
    run.close()


def _chain5(chain5: dict, purpose: str, batch: dict, model: str, multiplicity: dict | None = None):
    """One configuration of chain 5: the answers validated, then the graph run to the fit."""
    run, frame, genes = chain5["run"], chain5["frame"], chain5["genes"]
    base = {"lens": ["genomics"], "target": "case", "purpose": purpose, "task": "binary",
            "event": "1", "grain": {"grain": "one_row_per_unit", "id_column": "sample_id"}}
    roles = {c: "exposure" for c in genes}
    roles.update({"sample_id": "excluded",
                  "batch": "covariate" if batch["method"] == "covariate" else "excluded"})
    first = ProjectState.model_validate({**base, "roles": roles})
    out = run.run(first, upto=["findings"])
    findings = {f["id"]: f for f in data(out["findings"])["findings"]}
    scale = next(o for o in findings["omics_scale"]["repairs"] if o["key"] == "log_cpm_tmm")
    ctx = ctx_of(first, frame, out)
    validate(ApplyRepair(**{k: v for k, v in scale["decision"].items() if k != "kind"}), ctx)
    validate(SetBatch(**batch), ctx)
    slots = {**base, "roles": roles, "batch": batch,
             "findings": {"omics_scale": {"action": "applied", "option": "log_cpm_tmm",
                                          "params": scale["decision"]["params"]}},
             "missing": {"strategy": "complete_case"},
             "split": {"holdout": 0.0, "folds": 5, "seed": 0, "validation": "kfold"},
             "models": [model]}
    if multiplicity is not None:
        validate(SetMultiplicity(**multiplicity), ctx)
        slots["multiplicity"] = multiplicity
    state = ProjectState.model_validate(slots)
    validate(SelectModels(models=[model]), ctx_of(state, frame, out))
    out = run.run(state, upto=["fit"])
    return state, out, findings


def test_chain_5_the_batch_finding_and_the_leash_on_combat_with_the_outcome(chain5):
    """The cases were measured unevenly across the three batches: the findings say so, with
    Nygaard et al.'s warning and Cramér's V (scipy's ``association``, from pandas' crosstab).
    ComBat with the outcome protected is refused under both purposes, with the purpose's own exits;
    and when the batch is perfectly confounded with the outcome, every answer and every model is
    refused until the batch is said not to be a batch, and the design refuses as a backstop."""
    from scipy.stats.contingency import association

    run, frame = chain5["run"], chain5["frame"]
    for purpose in ("prediction", "inference"):
        state = ProjectState.model_validate({"lens": ["genomics"], "target": "case", "purpose": purpose})
        out = run.run(state, upto=["findings"])
        found = next(f for f in data(out["findings"])["findings"] if f["id"] == "batch_confounding__batch")
        v = association(pd.crosstab(frame["batch"], frame["case"]).to_numpy(), method="cramer",
                        correction=False)
        assert found["severity"] == "warning" and f"Cramér's V {v:.2f}" in found["detail"]
        assert "deflates group differences" in found["detail"]
        with pytest.raises(Refusal) as refused:
            validate(SetBatch(column="batch", method="outcome_combat"), ctx_of(state, frame, out))
        assert refused.value.code == {"prediction": "outcome_combat_leaks",
                                      "inference": "outcome_combat_for_testing"}[purpose]

    confounded = frame.assign(batch=np.where(frame["case"] == 1, "B1", "B2"))
    path = chain5["run"].folder.parent / "confounded.csv"
    confounded.to_csv(path, index=False)
    other = GraphRun(path, chain5["run"].folder.parent / "confounded_project")
    try:
        genes = chain5["genes"]
        roles = {**{c: "exposure" for c in genes}, "sample_id": "excluded", "batch": "excluded"}
        for purpose in ("prediction", "inference"):
            state = ProjectState.model_validate({
                "lens": ["genomics"], "target": "case", "purpose": purpose, "task": "binary",
                "event": "1", "roles": roles,
                "grain": {"grain": "one_row_per_unit", "id_column": "sample_id"}})
            out = other.run(state, upto=["findings"])
            found = next(f for f in data(out["findings"])["findings"]
                         if f["id"] == "batch_confounding__batch")
            assert found["severity"] == "critical"
            scale = next(f for f in data(out["findings"])["findings"] if f["id"] == "omics_scale")
            option = next(o for o in scale["repairs"] if o["key"] == "log_cpm_tmm")
            normalized = ProjectState.model_validate({**state.model_dump(), "findings": {
                "omics_scale": {"action": "applied", "option": "log_cpm_tmm",
                                "params": option["decision"]["params"]}}})
            ctx = ctx_of(normalized, confounded, out)
            for answer in (SetBatch(column="batch", method="covariate"),
                           SetBatch(column="batch", method="reference_combat"),
                           SelectModels(models=["elastic_net"])):
                with pytest.raises(Refusal) as refused:
                    validate(answer, ctx)
                assert refused.value.code == "batch_confounded_with_outcome"
            fitted = ProjectState.model_validate({
                **normalized.model_dump(), "missing": {"strategy": "complete_case"},
                "models": ["elastic_net"],
                "split": {"holdout": 0.0, "folds": 5, "seed": 0, "validation": "kfold"}})
            with pytest.raises(ValueError, match="perfectly confounded"):
                other.run(fitted, upto=["design"])
    finally:
        other.close()


def test_chain_5_prediction_batch_as_a_covariate_with_in_fold_screening_and_penalized_logistic(chain5):
    """Chain 5 as §6 writes it: batch as a covariate, in-fold screening, penalized logistic. The
    screen (sure independence screening, n / log n features) runs inside each training fold, after
    the normalization and the batch indicators; the batch covariate passes it. Its methods
    paragraph, verbatim."""
    state, out, _ = _chain5(chain5, "prediction", {"column": "batch", "method": "covariate"},
                            "screened_elastic_net")
    keys = steps_of(out["design"], "screened_elastic_net")
    assert keys == ["normalize", "onehot", "screen", "scale", "model"]
    pipeline = out["design"].objects["pipelines"]["screened_elastic_net"]
    names = [n for n, _ in pipeline.steps]
    assert names.index("screen") < names.index("scale") < names.index("model")
    fit = data(out["fit"])["models"][0]
    assert len(fit["cv"]["auc"]["folds"]) == 5
    said = omics.methods_paragraph(state, keys, "screened_elastic_net", data(out["split"]))
    assert said == ("Batch was included as a covariate; within each training fold, log-CPM with "
                    "TMM factors, sure independence screening and autoscaling were fitted and "
                    "applied to the held-out fold; elastic-net parameters were tuned in an inner CV "
                    "nested in an outer CV.")


def test_chain_5_prediction_reference_combat_in_fold_precedes_the_screen(chain5):
    """Under prediction the sound first option: ComBat with a reference batch, fitted on each
    training fold without the outcome, before the screen (§2: batch correction precedes in-fold
    screening). The relation fires from the contracts and shows in the design's step order; the
    methods paragraph names it in the in-fold group."""
    state, out, _ = _chain5(chain5, "prediction", {"column": "batch", "method": "reference_combat"},
                            "screened_elastic_net")
    keys = steps_of(out["design"], "screened_elastic_net")
    assert keys == ["normalize", "batch", "screen", "scale", "model"]
    choices = omics.chain_choices(state, keys, "screened_elastic_net")
    assert ("batch", "precedes", "screen") in {(f.source, f.relation.kind, f.relation.target)
                                                for f in fired(choices, "prediction")}
    step = out["design"].objects["pipelines"]["screened_elastic_net"].named_steps["batch"]
    assert step.batch == "batch" and step.drop is True
    said = omics.methods_paragraph(state, keys, "screened_elastic_net", data(out["split"]))
    assert said == ("Within each training fold, log-CPM with TMM factors, reference-batch ComBat "
                    "without the outcome, sure independence screening and autoscaling were fitted "
                    "and applied to the held-out fold; elastic-net parameters were tuned in an inner "
                    "CV nested in an outer CV.")


def test_chain_5_inference_twin_feature_wise_tests_with_bh_batch_as_a_covariate(chain5):
    """The inference twin: every gene tested in its own model with batch as a covariate, every gene
    shown, Benjamini–Hochberg q-values implied by the exposure family; ComBat for figures only. The
    first sentence is the reviewers' for chain 5. References: statsmodels' least squares of each
    gene's log-CPM on the event and the batch indicators for three genes, and Benjamini–Hochberg
    by statsmodels' ``multipletests`` over every gene's p-value. Recorded as no multiplicity
    control (with its attestation), the same tests carry no q-values and the paragraph says so."""
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    state, out, _ = _chain5(chain5, "inference", {"column": "batch", "method": "covariate",
                                                  "figures": True}, "featurewise")
    keys = steps_of(out["design"], "featurewise")
    assert keys == ["normalize", "onehot", "model"]
    fit = data(out["fit"])["models"][0]
    rows = fit["coefficients"]
    genes = chain5["genes"]
    assert [r["feature"] for r in rows] == genes  # every member of the family shown
    q_ref = multipletests([r["p"] for r in rows], method="fdr_bh")[1]
    np.testing.assert_allclose([r["q"] for r in rows], q_ref, rtol=1e-10)
    frame = chain5["frame"]
    from sklearn.base import clone

    from turbotab.core.models.linear import model_matrix

    X = frame[[*genes, "batch"]]
    fitted = clone(out["design"].objects["pipelines"]["featurewise"]).fit(X, frame["case"].to_numpy())
    matrix = model_matrix(fitted, X)
    dummies = [c for c in matrix.columns if c.startswith("batch_")]
    assert len(dummies) == 2
    event = frame["case"].to_numpy(dtype=float)
    for gene in genes[:3]:
        X = sm.add_constant(np.column_stack([matrix[dummies].to_numpy(), event]))
        reference = sm.OLS(matrix[gene].to_numpy(), X).fit()
        row = next(r for r in rows if r["feature"] == gene)
        assert row["estimate"] == pytest.approx(reference.params[-1], rel=1e-8)
        assert row["p"] == pytest.approx(reference.pvalues[-1], rel=1e-6)
    found = int(np.sum(q_ref < 0.05))
    said = omics.methods_paragraph(state, keys, "featurewise", data(out["split"]),
                                   table={"rows": rows})
    assert said == (f"{CHAIN_5_REVIEWERS} Counts were transformed to log2 counts per million on "
                    f"library sizes scaled by TMM factors (edgeR, prior count 2; Robinson and "
                    f"Oshlack 2010). Each of the {len(genes)} features was tested in its own linear "
                    f"model with the covariates; Benjamini–Hochberg q-values across the "
                    f"{len(genes)} tests ({found} below 0.05), every feature shown.")
    choices = omics.chain_choices(state, keys, "featurewise")
    assert choices["batch"] == "covariate" and choices["multiplicity"] == "bh"
    assert ("multiplicity", "implies", "bh_q_values") in {
        (f.source, f.relation.kind, f.relation.target)
        for f in fired(choices, "inference", consequences=["bh_q_values"])}

    with pytest.raises(Refusal) as refused:
        _chain5(chain5, "inference", {"column": "batch", "method": "covariate", "figures": True},
                "featurewise", multiplicity={"method": "none"})
    assert refused.value.code == "family_without_multiplicity"
    state, out, _ = _chain5(chain5, "inference", {"column": "batch", "method": "covariate",
                                                  "figures": True}, "featurewise",
                            multiplicity={"method": "none", "acknowledged": True})
    rows = data(out["fit"])["models"][0]["coefficients"]
    assert len(rows) == len(genes) and all(r["q"] is None for r in rows)
    said = omics.methods_paragraph(state, steps_of(out["design"], "featurewise"), "featurewise",
                                   data(out["split"]), table={"rows": rows})
    assert said.startswith(CHAIN_5_REVIEWERS) and said.endswith(
        f"Each of the {len(genes)} features was tested in its own linear model with the covariates; "
        f"no multiplicity control, recorded as a limitation.")
