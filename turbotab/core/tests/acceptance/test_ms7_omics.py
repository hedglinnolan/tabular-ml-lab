"""MS7 · the omics order: QC drift and batch, QC filters, PQN, values below detection, batch
handling and an exposure family's multiplicity (MODELING_SEQUENCE §1.1, §2, §4 and §5 MS7; V2
definition of done, genomics extended). The chains (§6, chains 3 and 5) are in
``test_ms7_chains.py``.

Each item below is the package's acceptance item, as written:

1. QC-RLSC (Dunn et al. 2011): per-feature, per-batch LOESS on injection order fitted to pooled QCs
   only, reference-rows scope before the seal; the QC rows then leave the cohort; on a simulated
   drift with known truth the corrected intensities recover the truth within a stated bound, and
   the smoother agrees with R ``stats::loess`` at the same span to 1e-6.
2. QC-RSD filter (< 20% LC-MS, < 30% GC-MS) and QC detection-rate filter run on reference rows; the
   D-ratio filter is training-fold scope and cannot run before the seal.
3. PQN in two variants: pooled-QC reference (reference rows) and study-sample median (training
   fold), each with its stated scope.
4. Detection-limit handling comes after normalization and before the log: half-minimum is customary
   at ≤ 10% censored; above that, QRILC or a censored-normal draw (ranks first); a zero that a log
   would turn missing routes to the detection-limit question and never to median fill (property
   test); MAR imputation of detection-limit blanks is refused under inference.
5. Batch: as a covariate first under inference; ComBat with a reference batch (the training fold)
   fitted in-fold and outcome-free under prediction — agrees with R ``sva::ComBat(ref.batch = …)``
   to 1e-6; outcome-protected ComBat refused for testing (figures only) and refused under
   prediction (leakage); batch perfectly confounded with the outcome refused under both purposes;
   batch correction precedes in-fold screening.
6. An exposure family (feature-wise) without multiplicity control is block-and-record; BH q-values
   agree with R ``p.adjust(method = "BH")`` to 1e-12.

Every expected value comes from a path independent of the code under test: R 4.6.1 (``stats``,
``sva`` 3.60, ``survival``) through ``Rscript`` on a CSV this file writes, pandas computations
written from the published definitions (``omics_references``), scipy and statsmodels, or the truth
of a simulation. Scopes are not taken from the contracts: :func:`contracts.observed_scope` changes
the outcome, the other study rows and the reference rows in turn and watches one study row.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import (FindingDisposition, MissingSpec, ProjectState, Refusal,
                                     SetBatch, SetMissing, SetMultiplicity, validate)
from turbotab.core.methods import batch as B
from turbotab.core.methods import omics
from turbotab.core.methods import qc_drift as Q
from turbotab.core.contracts import CONTRACTS, observed_scope, options_for
from turbotab.core.tests.acceptance.omics_references import (drifting_run, needs_r, pqn_by_hand,
                                                             run_r)

FEATURES = lambda frame: [c for c in frame.columns if c.startswith("m")]  # noqa: E731


def _plan(frame: pd.DataFrame, **kw) -> Q.QCPlan:
    return Q.QCPlan("sample_type", ("QC",), tuple(FEATURES(frame)), "injection_order", "batch", **kw)


# ── 1 · QC-RLSC ──────────────────────────────────────────────────────────────

LOESS_R = r"""
d <- read.csv("points.csv")
at <- read.csv("at.csv")
out <- data.frame()
for (key in unique(d$key)) {
  s <- d[d$key == key, ]
  a <- at[at$key == key, ]
  f <- loess(y ~ x, data = s, span = s$span[1], degree = s$degree[1], family = s$family[1],
             control = loess.control(surface = "direct"))
  out <- rbind(out, data.frame(key = key, x = a$x, fit = predict(f, newdata = data.frame(x = a$x))))
}
write.csv(out, "fits.csv", row.names = FALSE)
"""


@needs_r
def test_1a_the_drift_curve_is_r_loess_at_the_span_it_chose(tmp_path):
    """Each feature's drift curve, batch by batch, against R's ``loess(y ~ x, span, degree = 2,
    family = "gaussian", control = loess.control(surface = "direct"))`` fitted to the same QC
    points and predicted at every injection of the batch, at the span QC-RLSC chose for that
    feature and batch. Agreement to 1e-6 relative (measured: about 1e-13). The robust family
    (``"symmetric"``, R's four iterations) and degree 1 are held to R the same way at fixed spans."""
    frame, _ = drifting_run(seed=3, p=8)
    feats = FEATURES(frame)
    inj = Q.injections(frame, "sample_type", ["QC"], "injection_order", "batch")
    fit = Q.qc_rlsc(frame[feats].to_numpy(), inj)
    points, at, mine = [], [], {}
    for b in inj.batches:
        rows = np.flatnonzero(inj.batch == b)
        qc = rows[inj.qc[rows]]
        for j, f in enumerate(feats):
            key = f"{b}:{f}"
            span = float(fit.spans[b][j])
            points.append(pd.DataFrame({"key": key, "x": inj.order[qc], "y": frame[f].to_numpy()[qc],
                                        "span": span, "degree": 2, "family": "gaussian"}))
            at.append(pd.DataFrame({"key": key, "x": inj.order[rows]}))
            mine[key] = fit.curves[rows, j]
    rng = np.random.default_rng(5)
    x = np.sort(rng.choice(np.arange(1, 90), 15, replace=False)).astype(float)
    y = 100 + 0.4 * x + 6 * np.sin(x / 9) + rng.normal(0, 1.5, len(x))
    y[4] += 25  # an outlier the robust family down-weights
    grid = np.linspace(x[0], x[-1], 11)
    for span in (0.5, 0.75, 1.0):
        for family in ("gaussian", "symmetric"):
            for degree in (1, 2):
                key = f"generic:{span}:{family}:{degree}"
                points.append(pd.DataFrame({"key": key, "x": x, "y": y, "span": span,
                                            "degree": degree, "family": family}))
                at.append(pd.DataFrame({"key": key, "x": grid}))
                mine[key] = Q.loess(x, y, span, degree, family, at=grid)[1]
    ref = run_r(LOESS_R, {"points": pd.concat(points), "at": pd.concat(at)}, tmp_path, ("fits",))["fits"]
    worst = 0.0
    for key, values in mine.items():
        r = ref.loc[ref["key"] == key, "fit"].to_numpy()
        worst = max(worst, float(np.max(np.abs(values - r) / np.abs(r))))
    assert worst < 1e-6, worst
    assert len(mine) == 3 * 8 + 12


LOOCV_R = r"""
d <- read.csv("points.csv")
spans <- seq(0.30, 1.0001, by = 0.05)
out <- data.frame()
for (key in unique(d$key)) {
  s <- d[d$key == key, ]
  n <- nrow(s)
  best <- NA; err <- Inf
  for (sp in spans) {
    if (floor((n - 1) * sp + 1e-5) < 4) next
    e <- 0
    for (i in seq_len(n)) {
      f <- loess(y ~ x, data = s[-i, ], span = sp, degree = 2,
                 control = loess.control(surface = "direct"))
      e <- e + (s$y[i] - predict(f, newdata = data.frame(x = s$x[i])))^2
    }
    if (e / n <= err * (1 + 1e-12)) { err <- e / n; best <- sp }
  }
  out <- rbind(out, data.frame(key = key, span = best))
}
write.csv(out, "spans.csv", row.names = FALSE)
"""


@needs_r
def test_1b_the_span_is_chosen_by_leave_one_out_cross_validation_over_the_qcs(tmp_path):
    """Dunn et al. chose the smoothing parameter by cross-validation. Reference: R refits
    ``loess`` without each QC in turn, predicts it, and keeps the span with the smallest mean
    squared error over 0.30, 0.35, …, 1.00 (ties to the larger; a span that keeps fewer than four
    of the other QCs is skipped, as here). The spans agree for every feature and batch."""
    frame, _ = drifting_run(seed=11, p=6, per_batch=50)
    feats = FEATURES(frame)
    inj = Q.injections(frame, "sample_type", ["QC"], "injection_order", "batch")
    fit = Q.qc_rlsc(frame[feats].to_numpy(), inj)
    points, mine = [], {}
    for b in inj.batches:
        qc = np.flatnonzero((inj.batch == b) & inj.qc)
        for j, f in enumerate(feats):
            key = f"{b}:{f}"
            points.append(pd.DataFrame({"key": key, "x": inj.order[qc], "y": frame[f].to_numpy()[qc]}))
            mine[key] = float(fit.spans[b][j])
    ref = run_r(LOOCV_R, {"points": pd.concat(points)}, tmp_path, ("spans",))["spans"]
    chosen = dict(zip(ref["key"], ref["span"]))
    assert {k: round(v, 2) for k, v in mine.items()} == {k: round(float(v), 2) for k, v in chosen.items()}


def test_1c_on_a_simulated_drift_the_corrected_intensities_recover_the_truth():
    """Five simulated runs (three batches of 40 injections, a pooled QC every fifth and last;
    200 features, each with its own batch offsets (SD 0.3 on the log scale) and a drift falling by
    10–40% over the batch; multiplicative noise of SD 0.05). The truth is each injection's level
    × biology; a perfect correction recovers it up to one constant per feature, so the measure is
    the within-feature SD of log(corrected / truth) over the study injections.

    The stated bound: its median over features is at most 0.065, i.e. 1.3 × the noise floor
    (0.05: no correction can remove the injection's own noise; the curve adds the QCs' noise,
    averaged by the LOESS). Measured: about 0.057. Uncorrected it is about 0.24, asserted above
    0.15, so the bound discriminates. The QCs' own RSD after correction sits at the noise level:
    median at most 7% (the 5% noise CV plus the curve's error)."""
    for seed in range(5):
        frame, truth = drifting_run(seed=seed)
        result = Q.run_reference_rows(frame, _plan(frame))
        study = frame["sample_type"].eq("Sample").to_numpy()
        after = np.log(result.values[study] / truth[study]).std(axis=0, ddof=1)
        before = np.log(frame[FEATURES(frame)].to_numpy()[study] / truth[study]).std(axis=0, ddof=1)
        assert np.median(after) <= 0.065, (seed, np.median(after))
        assert np.median(before) > 0.15, (seed, np.median(before))
        qc = ~study
        rsd = pd.DataFrame(result.values[qc]).pipe(lambda d: 100 * d.std(ddof=1) / d.mean())
        assert float(rsd.median()) <= 7.0, (seed, float(rsd.median()))
        assert result.record["kept"] == 200


def _correct(frame: pd.DataFrame, reference: pd.Series, y: np.ndarray, **kw) -> pd.DataFrame:
    marked = frame.copy()
    marked["sample_type"] = np.where(reference.to_numpy(), "QC", "Sample")
    result = Q.run_reference_rows(marked, _plan(marked, **kw))
    return pd.DataFrame(result.values, index=frame.index)


def test_1d_qc_rlsc_learns_from_the_pooled_qcs_only_and_they_leave_the_cohort():
    """The scope is observed, not declared (lockbox constitution §06): a study injection's corrected
    values change when the pooled QCs change, and do not change when every other study injection or
    the outcome does. That is reference-rows scope, which the contract declares and which may run
    before the seal. The QC rows then leave as reference rows (WP18): the working table drops them
    and the participant flow counts them on a line of their own, before the outcome is read."""
    frame, _ = drifting_run(seed=2, p=30)
    reference = frame["sample_type"].eq("QC").to_numpy()
    y = np.random.default_rng(0).normal(size=len(frame))
    study_row = int(np.flatnonzero(~reference)[7])
    scope = observed_scope(lambda f, r, yy: _correct(f, r, yy), frame.drop(columns=["sample_type"]),
                           reference, y, study_row, columns=FEATURES(frame))
    assert scope == "reference_rows" == CONTRACTS["qc_rlsc"].scope
    assert CONTRACTS["qc_rlsc"].slot == "repairs"

    from turbotab.core.stages.rows import cohort_flow

    from turbotab.core.reference_rows import reference_filter, reference_rules

    rule = {"column": "sample_type", "levels": ["QC"]}
    state = ProjectState(findings={Q.FINDING: {"action": "applied", "option": "qc_rlsc_lc", "params": {
        "column": "sample_type", "qc_levels": ["QC"]}}})
    assert reference_rules(state.findings) == [{**rule, "finding": Q.FINDING}]
    assert "NOT IN ('QC')" in reference_filter([rule])
    flow = frame.assign(outcome=y).loc[~reference]  # the working table, the QC rows gone
    steps, kept = cohort_flow(flow, target="outcome", rules=[], missing=None, predictor_columns=[],
                              reference=[{**rule, "n": int(reference.sum())}])
    assert [s["key"] for s in steps][:3] == ["loaded", "reference:0", "outcome_measured"]
    assert steps[0]["n"] == len(frame)
    assert steps[1]["dropped"] == int(reference.sum()) and steps[2]["dropped"] == 0
    assert set(kept) == set(np.flatnonzero(~reference))


def test_1e_qc_rlsc_is_refused_where_its_curve_would_be_guessed():
    """A batch with fewer than five pooled QCs (Broadhurst et al. 2018) or with study injections
    before its first QC or after its last (the curve would be extrapolated, Dunn et al.) is
    refused, with its reason, and the exit sets the QC rows aside without correcting drift."""
    frame, _ = drifting_run(seed=4, p=25, per_batch=20, qc_every=5)
    few = frame.copy()
    first = few.index[few["batch"].eq("B2") & few["sample_type"].eq("QC")][:2]
    few.loc[first, "sample_type"] = "Sample"  # B2 keeps 3 QCs, and its first injection is a sample
    inj = Q.injections(few, "sample_type", ["QC"], "injection_order", "batch")
    problems = Q.sufficiency(inj)
    assert len(problems) == 1 and "Batch `B2` has 3 pooled-QC injections" in problems[0]
    with pytest.raises(Refusal) as refused:
        Q.run_reference_rows(few, _plan(few))
    assert refused.value.code == "qc_too_sparse"
    late, _ = drifting_run(seed=4, p=25, per_batch=20, qc_every=4)  # QCs at 0, 4, …, 16 and 19
    last_qc = late.index[late["batch"].eq("B1") & late["sample_type"].eq("QC")][-1]
    late.loc[last_qc, "sample_type"] = "Sample"
    said = Q.sufficiency(Q.injections(late, "sample_type", ["QC"], "injection_order", "batch"))
    assert said == ["Batch `B1` has study injections outside its QCs (3 after its last): the curve "
                    "would be extrapolated. A run must begin and end with QC injections (Dunn et "
                    "al. 2011)."]

    from turbotab.core.repairs import OfferContext, offer

    finding = {"id": Q.FINDING, "repairs": []}
    options = offer(finding, {"params": {"column": "sample_type", "qc_value": "QC"}},
                    OfferContext(frame=few, target=None))
    keys = [o.key for o in options]
    # The batch column is a reading (MS7 repair): the whole run as one batch is offered beside it.
    assert keys == [*Q.RLSC_OPTIONS, "qc_rlsc_lc", "qc_rlsc_gc", Q.ASIDE]
    assert [o.decision.params.get("batch_column") for o in options[:6]] == ["batch"] * 4 + [None] * 2
    finding["repairs"] = [o.model_dump(mode="json") for o in options]
    state = ProjectState(target="y", purpose="prediction")
    ctx = {"state": state, "columns": list(few.columns),
           "artifact": lambda name: {"findings": [finding]} if name == "findings" else None}
    from turbotab.core.decisions import ApplyRepair

    with pytest.raises(Refusal) as refused:
        validate(ApplyRepair(finding_id=Q.FINDING, option="qc_rlsc_lc",
                             params=options[0].decision.params), ctx)
    assert refused.value.code == "qc_too_sparse"
    exit_decision = refused.value.exits[0]["decision"]
    assert exit_decision["option"] == Q.ASIDE
    validate(ApplyRepair(**{k: v for k, v in exit_decision.items() if k != "kind"}), ctx)


# ── 2 · the QC filters, and the D-ratio ──────────────────────────────────────


def planted_qc_table(seed: int = 0) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """Two batches of 30 injections, a pooled QC every third and last (22 QCs), no drift. Each
    feature has a planted QC coefficient of variation and a planted QC detection rate; study
    samples vary biologically (SD 0.6 on the log scale), so a QC CV of 0.33 puts a feature's
    D-ratio near its 50% criterion."""
    rng = np.random.default_rng(seed)
    rows, order = [], 0
    for b in ("B1", "B2"):
        for k in range(30):
            order += 1
            rows.append({"injection_order": order, "batch": b,
                         "sample_type": "QC" if (k % 3 == 0 or k == 29) else "Sample"})
    frame = pd.DataFrame(rows)
    qc = frame["sample_type"].eq("QC").to_numpy()
    cvs = {"m_cv05": 0.05, "m_cv10": 0.10, "m_cv25": 0.25, "m_cv40": 0.40, "m_cv33": 0.33,
           "m_det100": 0.05, "m_det80": 0.05, "m_det65": 0.05, "m_det40": 0.05}
    detect = {"m_det100": 1.0, "m_det80": 0.8, "m_det65": 0.65, "m_det40": 0.4}
    for name, cv in cvs.items():
        level = math.exp(rng.normal(8, 1))
        values = np.where(qc, level * np.exp(rng.normal(0, cv, len(frame))),
                          level * np.exp(rng.normal(0, 0.6, len(frame))))
        if name in detect:
            qc_rows = np.flatnonzero(qc)
            lost = rng.choice(qc_rows, int(round((1 - detect[name]) * len(qc_rows))), replace=False)
            values[lost] = np.nan
        frame[name] = values
    return frame, cvs, detect


def _filter_plan(frame: pd.DataFrame, platform: str = "lc_ms") -> Q.QCPlan:
    feats = [c for c in frame.columns if c.startswith("m_")]
    return Q.QCPlan("sample_type", ("QC",), tuple(feats), "injection_order", "batch", platform=platform)


def test_2a_the_qc_rsd_filter_removes_at_20_percent_for_lc_ms_and_30_for_gc_ms():
    """QC RSD (Broadhurst et al. 2018: "acceptance criterion for RSD is typically set to < 20% …
    or < 30%"), on the corrected QC responses. Reference: pandas' ``std(ddof = 1) / mean × 100``
    over each feature's detected QC values after correction. Planted QC CVs of 5% and 10% stay
    under both criteria, 40% fails both, 25% fails LC-MS's 20% and passes GC-MS's 30%."""
    frame, _, _ = planted_qc_table()
    qc = frame["sample_type"].eq("QC").to_numpy()
    for platform, limit, dropped in (("lc_ms", 20.0, {"m_cv25", "m_cv40"}), ("gc_ms", 30.0, {"m_cv40"})):
        result = Q.run_reference_rows(frame, _filter_plan(frame, platform))
        feats = list(_filter_plan(frame).features)
        corrected = pd.DataFrame(result.values[qc], columns=feats)
        corrected = corrected.where(corrected > 0)
        rsd = 100 * corrected.std(ddof=1) / corrected.mean()
        early = set(result.record["dropped"]["detection"]) | set(result.record["dropped"]["uncorrectable"])
        expected = {f for f in feats if f not in early and not rsd[f] < limit}
        assert set(result.record["dropped"]["rsd"]) == expected
        assert {f for f in expected if f.startswith("m_cv") and f != "m_cv33"} == dropped, (
            platform, rsd.round(1).to_dict())


def test_2b_the_qc_detection_rate_filter_keeps_features_detected_in_70_percent_of_qcs():
    """Broadhurst et al. 2018: "The acceptance criterion for detection rate is typically set to >
    70%". Reference: pandas' count of each feature's detected QC values over the QC injections.
    Planted 100% and 80% stay; 65% and 40% leave, before the drift curve is fit."""
    frame, _, detect = planted_qc_table(seed=1)
    qc = frame["sample_type"].eq("QC")
    rate = frame.loc[qc, list(detect)].gt(0).sum() / int(qc.sum())
    result = Q.run_reference_rows(frame, _filter_plan(frame))
    assert set(result.record["dropped"]["detection"]) == {f for f in detect if not rate[f] >= 0.70}
    assert set(result.record["dropped"]["detection"]) == {"m_det65", "m_det40"}


def _masked(frame: pd.DataFrame, reference: pd.Series, steps: list[str]) -> pd.DataFrame:
    marked = frame.copy()
    marked["sample_type"] = np.where(reference.to_numpy(), "QC", "Sample")
    plan = _filter_plan(marked)
    result = Q.run_reference_rows(marked, plan, steps=steps)
    out = pd.DataFrame(result.values, columns=list(plan.features), index=frame.index)
    out[[c for c in out.columns if c not in set(result.kept)]] = np.nan
    return out


@pytest.mark.parametrize("step", ["qc_detection_filter", "qc_rsd_filter"])
def test_2c_the_qc_filters_read_the_pooled_qcs_only(step):
    """Observed scope of each QC filter alone: a study injection's kept values move when the QCs
    change (a feature crosses the criterion) and never when another study injection or the outcome
    does. Reference rows, as declared, so they run before the seal."""
    frame, _, _ = planted_qc_table(seed=2)
    reference = frame["sample_type"].eq("QC").to_numpy()
    y = np.random.default_rng(1).normal(size=len(frame))
    feats = [c for c in frame.columns if c.startswith("m_")]
    row = int(np.flatnonzero(~reference)[3])
    scope = observed_scope(lambda f, r, yy: _masked(f, r, [step]), frame.drop(columns=["sample_type"]),
                           reference, y, row, columns=feats, seed=7, blank=0.3)
    assert scope == "reference_rows" == CONTRACTS[step].scope


def test_2d_the_d_ratio_reads_participants_so_it_runs_in_each_training_fold_and_never_before_the_seal():
    """The D-ratio (Broadhurst et al. 2018: QC SD over the study samples' SD, "set to, at most, <
    50%") reads the study samples' spread. Observed scope: training fold (another study row moves
    it), as declared. Asked to run before the seal, the reference-rows executor refuses it; in the
    pipeline it is the first in-fold step, its denominator the fitting rows' own: reference,
    pandas' ``std(ddof = 1)`` over the training rows, its numerator the QC SD frozen before the
    seal."""
    frame, _, _ = planted_qc_table(seed=3)
    feats = [c for c in frame.columns if c.startswith("m_")]
    qc = frame["sample_type"].eq("QC").to_numpy()
    qc_sd = frame.loc[qc, feats].std(ddof=1).to_dict()
    study = frame.loc[~qc, feats].reset_index(drop=True)
    train = study.iloc[:24]
    step = Q.DRatioFilter(feats, qc_sd, 0.5).fit(train)
    expected = {c: qc_sd[c] / float(train[c].std(ddof=1)) for c in feats}
    for c in feats:
        assert step.ratio_[c] == pytest.approx(expected[c], rel=1e-12)
    assert set(step.dropped_) == {c for c in feats if not expected[c] < 0.5}
    assert "m_cv40" in step.dropped_ and "m_cv05" not in step.dropped_
    # held-out rows never move it: the same fit whatever the held-out rows hold
    again = Q.DRatioFilter(feats, qc_sd, 0.5).fit(train)
    assert again.dropped_ == step.dropped_

    def fit_transform(f: pd.DataFrame, r: pd.Series, yy: np.ndarray) -> pd.DataFrame:
        rows = f.loc[~r.to_numpy(), feats]
        fitted = Q.DRatioFilter(feats, qc_sd, 0.5).fit(rows)
        return fitted.transform(f[feats]).reindex(columns=feats)

    reference = qc
    y = np.random.default_rng(3).normal(size=len(frame))
    row = int(np.flatnonzero(~qc)[2])
    assert observed_scope(fit_transform, frame.drop(columns=["sample_type"]), reference, y, row,
                          columns=feats, seed=11, blank=0.3) == "training_fold" == CONTRACTS["d_ratio_filter"].scope
    with pytest.raises(Refusal) as refused:
        Q.run_reference_rows(frame, _filter_plan(frame),
                             steps=["qc_detection_filter", "qc_rlsc", "d_ratio_filter"])
    assert refused.value.code == "training_fold_before_seal"
    assert "cannot run before the seal" in refused.value.message

    from turbotab.core.models.pipeline import DesignSpec, shared_steps

    spec = DesignSpec(predictors=feats, inputs=feats, categorical=[], numeric=feats, energy=None,
                      impute=True, roles={c: "exposure" for c in feats},
                      d_ratio={"columns": feats, "qc_sd": qc_sd, "threshold": 0.5})
    steps = shared_steps(spec)
    assert [name for name, _ in steps][:2] == ["d_ratio", "impute"]
    from turbotab.core.models.pipeline import transformer

    held = study.iloc[24:].copy()
    held.iloc[0, 0] = np.nan  # a blank the fill still reaches after the filter drops columns
    out = transformer(steps).fit(train).transform(held)
    assert list(out.columns) == [c for c in feats if c not in set(step.dropped_)]
    assert np.isfinite(out.to_numpy()).all()


# ── 3 · PQN in two variants ──────────────────────────────────────────────────


def test_3a_pqn_against_the_pooled_qc_reference_is_reference_rows_scope():
    """PQN with a pooled-QC reference spectrum (pmp's default, Dieterle et al.'s steps), before the
    seal. Reference: :func:`omics_references.pqn_by_hand` with the QC rows as the reference set,
    applied to every row, to 1e-12. Observed scope: reference rows (a study row's result moves
    only with the QCs), as the contract declares."""
    frame, _ = drifting_run(seed=6, p=40, amplitude=(0.0, 0.0), batch_sd=0.0)
    feats = FEATURES(frame)
    dil = np.exp(np.random.default_rng(6).normal(0, 0.3, len(frame)))
    frame[feats] = frame[feats].mul(dil, axis=0)  # each injection diluted
    qc = frame["sample_type"].eq("QC").to_numpy()
    plan = _plan(frame, correct=False, pqn=True)
    result = Q.run_reference_rows(frame, plan)
    expected = pqn_by_hand(frame.loc[qc, feats], frame[feats]).to_numpy()
    np.testing.assert_allclose(result.values, expected, rtol=1e-12)
    y = np.random.default_rng(2).normal(size=len(frame))
    scope = observed_scope(
        lambda f, r, yy: pd.DataFrame(Q.run_reference_rows(
            f.assign(sample_type=np.where(r, "QC", "Sample")),
            Q.QCPlan("sample_type", ("QC",), tuple(feats), "injection_order", "batch",
                     correct=False, pqn=True)).values, index=f.index),
        frame.drop(columns=["sample_type"]), qc, y, int(np.flatnonzero(~qc)[5]), columns=feats)
    assert scope == "reference_rows" == CONTRACTS["qc_pqn"].scope


def test_3b_pqn_against_the_study_sample_median_is_training_fold_scope():
    """PQN with the study samples' median spectrum, fitted on a training fold and applied to every
    row. Reference: :func:`omics_references.pqn_by_hand` with the training rows as the reference
    set, to 1e-12. Observed scope: training fold (another study row moves the reference), the
    scope of the ``pqn_log2`` option."""
    from turbotab.core.methods.omics import QuotientLog

    frame, _ = drifting_run(seed=7, p=40, amplitude=(0.0, 0.0), batch_sd=0.0)
    feats = FEATURES(frame)
    study = frame.loc[frame["sample_type"].eq("Sample"), feats].reset_index(drop=True)
    train = study.iloc[:60]
    step = QuotientLog(feats, quotient=True, log=False).fit(train)
    np.testing.assert_allclose(step.transform(study).to_numpy(),
                               pqn_by_hand(train, study).to_numpy(), rtol=1e-12)
    reference = np.zeros(len(study), dtype=bool)
    y = np.random.default_rng(3).normal(size=len(study))

    def fit_transform(f: pd.DataFrame, r: pd.Series, yy: np.ndarray) -> pd.DataFrame:
        return QuotientLog(feats, quotient=True, log=False).fit(f).transform(f)

    assert observed_scope(fit_transform, study, reference, y, 4) == "training_fold"
    assert CONTRACTS["omics_normalization"].scope_of("pqn_log2") == "training_fold"


# ── 4 · values below detection: after the normalization, before the log ─────


def _censored_intensities(seed: int, n: int = 120, p: int = 30, share: float = 0.08,
                          censored: int = 5) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Diluted log-normal intensities; in ``censored`` columns the lowest ``share`` of values are
    blanks below a detection limit."""
    rng = np.random.default_rng(seed)
    feats = [f"mz_{j:03d}" for j in range(p)]
    values = np.exp(rng.normal(rng.normal(7, 1.5, p), 0.7, (n, p)))
    values *= np.exp(rng.normal(0, 0.3, n))[:, None]
    for j in range(censored):
        cut = np.quantile(values[:, j], share)
        values[values[:, j] < cut, j] = np.nan
    return pd.DataFrame(values, columns=feats), feats, feats[:censored]


def test_4a_values_below_detection_are_filled_after_the_normalization_and_before_the_log():
    """MODELING_SEQUENCE §1.1: "PQN with a study-sample reference … detection-limit handling …
    log". With a logged normalization over a censored column the pipeline runs normalize (PQN
    alone) → detect → log, and its output equals the by-hand chain: :func:`pqn_by_hand` with the
    training rows' reference, each censored column's blanks set to half its smallest detected
    normalized value on the training rows, then log2, to 1e-12. Filling the raw values first and
    normalizing after, the order it replaces, gives other numbers."""
    from turbotab.core.models import get_family
    from turbotab.core.models.pipeline import DesignSpec, describe_steps, shared_steps, transformer

    frame, feats, censored = _censored_intensities(seed=1)
    train, rows = frame.iloc[:90], frame
    spec = DesignSpec(predictors=feats, inputs=feats, categorical=[], numeric=feats, energy=None,
                      impute=False, roles={c: "exposure" for c in feats},
                      normalization={"method": "pqn_log2", "kind": "intensities", "columns": feats},
                      censored={"method": "half_minimum", "columns": censored},
                      lenses=["metabolomics"])
    steps = shared_steps(spec)
    assert [name for name, _ in steps] == ["normalize", "detect", "log"]
    described = describe_steps(spec, get_family("elastic_net"), "binary", "prediction")
    assert [s["key"] for s in described][:3] == ["normalize", "detect", "log"]
    assert described[0]["label"] == "Quotient normalization" and described[2]["label"] == "Log2"
    out = transformer(steps).fit(train).transform(rows).to_numpy()
    normalized = pqn_by_hand(train, rows)
    filled = normalized.copy()
    for c in censored:
        half = 0.5 * float(normalized.loc[train.index, c].min())
        filled[c] = filled[c].fillna(half)
    np.testing.assert_allclose(out, np.log2(filled.to_numpy()), rtol=1e-12)
    raw_first = frame.copy()
    for c in censored:
        raw_first[c] = raw_first[c].fillna(0.5 * float(frame.loc[train.index, c].min()))
    old = np.log2(pqn_by_hand(raw_first.loc[train.index], raw_first).to_numpy())
    assert not np.allclose(out, old)
    assert CONTRACTS["detection_limit"].relations[-1].kind == "precedes"
    assert CONTRACTS["detection_limit"].relations[-1].target == "log_transform"


def test_4b_half_minimum_is_customary_at_ten_percent_censored_and_ranks_below_qrilc_and_the_censored_normal_above():
    """The options, soundest first, at 5% and at 25% below the limit (Lubin et al. 2004: half the
    limit "can be biased unless the percentage of measurements below detection limits is small
    (5-10%)"; Wei et al. 2018: "QRILC was the favored one for left-censored MNAR"). The
    censored-normal ranks first at every share; above 10% QRILC ranks second and half the minimum
    is ranked lower under prediction and blocked and recorded under inference. Filling them as
    any other blank is refused under inference."""
    low = omics.detection_limit_options("prediction", 0.05)
    assert [o["key"] for o in low] == ["censoring_aware", "half_minimum", "qrilc", "as_missing"]
    assert low[1]["rung"] == "available" and low[1]["customary"] == "Customary: MetaboAnalyst's default"
    high = omics.detection_limit_options("prediction", 0.25)
    assert [o["key"] for o in high] == ["censoring_aware", "qrilc", "half_minimum", "as_missing"]
    assert [o["rung"] for o in high] == ["recommended", "recommended", "rank_lower", "rank_lower"]
    inference = omics.detection_limit_options("inference", 0.25)
    rungs = {o["key"]: o["rung"] for o in inference}
    assert rungs == {"censoring_aware": "recommended", "qrilc": "refused",
                     "half_minimum": "block_and_record", "as_missing": "refused"}
    assert options_for("detection_limit", "inference")[0]["key"] == "censoring_aware"


def _missing_ctx(purpose: str, columns: list[str], n_missing: int, n_rows: int = 200) -> dict:
    state = ProjectState(target="y", purpose=purpose, roles={c: "exposure" for c in columns})
    info = {c: {"dtype": "numeric", "n_unique": n_rows - n_missing, "n_missing": n_missing}
            for c in columns}
    return {"state": state, "columns": [*columns, "y"], "column_info": info,
            "artifact": lambda name: {"n_rows": n_rows} if name == "ingest" else None}


def test_4c_under_inference_values_below_detection_are_never_filled_as_missing_at_random():
    """MODELING_SEQUENCE §4: "MAR imputation of detection-limit blanks | rank lower | refuse,
    censoring-aware instead". Under inference a fill that reads them as missing at random is
    refused whatever reason the answer gives; QRILC (one draw outside the multiple imputation) is
    refused; half the minimum at 25% censored is blocked until it is recorded. Under prediction a
    reason keeps the fill, and QRILC and half the minimum are accepted."""
    cols = ["m1", "m2"]
    ctx = _missing_ctx("inference", cols, n_missing=50)
    for answer in (SetMissing(strategy="multiple_imputation", censored_columns=cols),
                   SetMissing(strategy="multiple_imputation", censored_columns=cols,
                              reason="the blanks are failed injections")):
        with pytest.raises(Refusal) as refused:
            validate(answer, ctx)
        assert refused.value.code == "median_below_detection"
        assert all(e["decision"] is not None for e in refused.value.exits)  # no "give your reason"
        assert "never filled as missing at random" in refused.value.message
    with pytest.raises(Refusal) as refused:
        validate(SetMissing(strategy="multiple_imputation", censored_columns=cols,
                            below_detection="qrilc"), ctx)
    assert refused.value.code == "qrilc_under_inference"
    half = SetMissing(strategy="multiple_imputation", censored_columns=cols, below_detection="half_minimum")
    with pytest.raises(Refusal) as refused:
        validate(half, ctx)
    assert refused.value.code == "half_minimum_above_ten_percent"
    recorded = refused.value.exits[-1]["decision"]
    assert recorded["acknowledged"] is True and recorded["below_detection"] == "half_minimum"
    validate(SetMissing(**{k: v for k, v in recorded.items() if k != "kind"}), ctx)
    validate(SetMissing(strategy="multiple_imputation", censored_columns=cols,
                        below_detection="censoring_aware"), ctx)
    validate(half, _missing_ctx("inference", cols, n_missing=10))  # 5%: customary, accepted
    prediction = _missing_ctx("prediction", cols, n_missing=50)
    validate(SetMissing(strategy="impute", censored_columns=cols, reason="failed injections"), prediction)
    validate(SetMissing(strategy="impute", censored_columns=cols, below_detection="qrilc"), prediction)
    validate(SetMissing(strategy="impute", censored_columns=cols, below_detection="half_minimum"), prediction)


def test_4d_qrilc_recovers_each_samples_left_censored_normal():
    """QRILC (Lazar et al. 2016; imputeLCMD's ``impute.QRILC``) against simulation truth: 60
    samples × 500 features, each sample's log intensities N(μᵢ, σᵢ²) with its lowest pᵢ (10–40%)
    below its detection limit. Stated bounds, set from the method's own precision (a least-squares
    line through 100 quantiles; measured here: median errors 0.035 σ in μ, 2.5% in σ):

    * the fitted mean within 0.25 σᵢ of μᵢ and the SD within 20% of σᵢ for every sample, the
      median errors within 0.06 σ and 5%;
    * every draw below the fitted detection quantile;
    * the drawn values' mean within 0.45 σᵢ of the true mean below the limit, E[X | X < qᵢ] =
      μᵢ − σᵢ φ(zᵢ)/Φ(zᵢ), median within 0.08 σ (measured 0.052), where half the minimum misses
      it by more (its median error is asserted larger).

    A row's fill reads only that row: row-local, as the contract's QRILC option declares. No R
    reference: imputeLCMD is not installed; the simulation's truth stands in."""
    from scipy import stats

    rng = np.random.default_rng(42)
    n, p = 60, 500
    mu = rng.normal(8, 0.5, n)
    sigma = rng.uniform(0.8, 1.2, n)
    share = rng.uniform(0.10, 0.40, n)
    logs = mu[:, None] + sigma[:, None] * rng.normal(size=(n, p))
    limits = np.array([np.quantile(logs[i], share[i]) for i in range(n)])
    censored = logs < limits[:, None]
    values = np.where(censored, np.nan, np.exp(logs))
    feats = [f"f{j}" for j in range(p)]
    frame = pd.DataFrame(values, columns=feats)
    filled = omics.QRILCFill(feats, feats).fit(frame).transform(frame).to_numpy()
    err_mu, err_sd, err_draw, err_half = [], [], [], []
    for i in range(n):
        observed = np.log(values[i][~censored[i]])
        m_hat, s_hat = omics.qrilc_parameters(observed, censored[i].mean())
        err_mu.append(abs(m_hat - mu[i]) / sigma[i])
        err_sd.append(abs(s_hat / sigma[i] - 1))
        drawn = np.log(filled[i][censored[i]])
        assert np.all(drawn <= m_hat + s_hat * stats.norm.ppf(censored[i].mean() + 0.001) + 1e-12)
        z = stats.norm.ppf(share[i])
        truth = mu[i] - sigma[i] * stats.norm.pdf(z) / stats.norm.cdf(z)
        err_draw.append(abs(drawn.mean() - truth) / sigma[i])
        err_half.append(abs(math.log(np.exp(observed).min() / 2) - truth) / sigma[i])
    assert max(err_mu) <= 0.25 and max(err_sd) <= 0.20 and max(err_draw) <= 0.45, (
        max(err_mu), max(err_sd), max(err_draw))
    assert np.median(err_mu) <= 0.06 and np.median(err_sd) <= 0.05 and np.median(err_draw) <= 0.08
    assert np.median(err_half) > np.median(err_draw), (np.median(err_half), np.median(err_draw))
    assert np.isfinite(filled).all()
    reference = np.zeros(n, dtype=bool)
    scope = observed_scope(lambda f, r, yy: omics.QRILCFill(feats, feats).fit(f).transform(f),
                           frame, reference, rng.normal(size=n), 3)
    # A sample QRILC can read is filled from itself alone; one too sparse to read takes half the
    # training fold's minimum (``test_ms7_omics_repair``), so the contract declares the wider scope.
    assert scope == "row_local"
    assert CONTRACTS["detection_limit"].scope_of("qrilc") == "training_fold"


SURVREG_R = r"""
suppressPackageStartupMessages(library(survival))
d <- read.csv("column.csv")
z <- log(d$m)
det <- !is.na(z)
L <- min(z[det])
zz <- ifelse(det, z, L)
f <- survreg(Surv(zz, det, type = "left") ~ 1, dist = "gaussian",
             control = survreg.control(rel.tolerance = 1e-12, maxiter = 200))
mu <- unname(coef(f)[1]); s <- f$scale
a <- (L - mu) / s
fill <- exp(mu + s^2 / 2) * pnorm(a - s) / pnorm(a)
write.csv(data.frame(mu = mu, sigma = s, fill = fill), "fit.csv", row.names = FALSE)
"""


@needs_r
def test_4e_the_censoring_aware_fill_is_the_expected_value_below_the_limit_under_rs_tobit_fit(tmp_path):
    """The prediction pipeline's censoring-aware fill: each blank becomes E[X | X < L] under a
    left-censored normal fitted to the column's logarithm (before a log the pipeline always fits
    on the log scale, so the fill is positive; elsewhere ``imputation.censoring_of`` chooses the
    scale, and here picks the log too). Reference: R ``survival::survreg(Surv(z, detected, type =
    "left") ~ 1, dist = "gaussian")`` for μ and σ, and the log-normal's truncated mean ``exp(μ +
    σ²/2) Φ(a − σ) / Φ(a)``, to 1e-6."""
    from turbotab.core.methods.missing import BelowDetectionFill

    rng = np.random.default_rng(8)
    x = np.exp(rng.normal(2.0, 0.8, 300))
    x[x < np.quantile(x, 0.3)] = np.nan
    frame = pd.DataFrame({"m": x})
    ref = run_r(SURVREG_R, {"column": frame}, tmp_path, ("fit",))["fit"].iloc[0]
    # the pipeline's fill before a log (MS7), and the general fill where the log scale fits better
    for step in (omics.LogScaleCensoredFill(["m"]).fit(frame),
                 BelowDetectionFill(["m"], "censoring_aware").fit(frame)):
        assert step.fill_["m"] == pytest.approx(float(ref["fill"]), rel=1e-6)
        assert 0 < step.fill_["m"] < step.limit_["m"]


def test_4f_property_a_zero_a_log_would_turn_missing_never_reaches_a_median_fill():
    """A seeded property sweep: 40 random intensity tables, each with zeros in some columns, a
    random logged normalization (PQN then log2, or log2), a random purpose and a random
    missing-values answer. For every one:

    * while nobody has said what the zeros are, the selection and the design are refused (code
      ``zeros_cannot_be_logged``), whatever the families and whatever fills missing values, and
      the first exit is the detection-limit question;
    * once the zeros are recoded as non-detections, a fill that reads them as missing at random
      is refused (under inference whatever the reason);
    * with a detection-limit answer, the log step never sees a zero or a blank: every former zero
      comes out of it as the log of its detection fill, so nothing is left for a median.
    """
    from turbotab.core.models import get_family
    from turbotab.core.models.pipeline import design_spec, shared_steps, transformer

    rng = np.random.default_rng(2026)
    families = [get_family(k) for k in ("linear", "elastic_net", "boosted_trees")]
    for case in range(40):
        n, p = int(rng.integers(40, 90)), int(rng.integers(20, 35))
        feats = [f"mz_{j}" for j in range(p)]
        values = np.exp(rng.normal(rng.normal(7, 1, p), 0.6, (n, p)))
        zero = rng.random((n, p)) < rng.uniform(0.01, 0.08)
        zero[int(rng.integers(n)), int(rng.integers(p))] = True
        values[zero] = 0.0
        frame = pd.DataFrame(values, columns=feats)
        purpose = str(rng.choice(["prediction", "inference"]))
        strategies = ["impute", "complete_case"] + (["multiple_imputation"] if purpose == "inference" else [])
        strategy = str(rng.choice(strategies))
        option = str(rng.choice(["pqn_log2", "log2"]))
        zero_columns = {c: int(k) for c, k in zip(feats, zero.sum(axis=0)) if k}
        scale = {"omics_scale": FindingDisposition(
            action="applied", option=option,
            params={"kind": "intensities", "columns": feats, "n_zero": int(zero.sum()),
                    "zero_columns": zero_columns})}
        base = dict(lens=["metabolomics"], target="y", purpose=purpose,
                    roles={c: "exposure" for c in feats})
        state = ProjectState(**base, missing=MissingSpec(strategy=strategy), findings=scale)
        refusal = omics.zeros_refusal(["linear", "elastic_net", "boosted_trees"], state)
        assert refusal is not None and refusal.code == "zeros_cannot_be_logged", case
        assert refusal.exits[0]["label"].startswith("Zeros mean not detected")
        assert "cannot be logged" in (omics.design_refusal(state, frame, families) or ""), case

        # The answer: the zeros are non-detections (the repair blanks them in the working table).
        cols = sorted(zero_columns)
        answered = {**scale, "pack::metabolomics::zeros_or_missing": FindingDisposition(
            action="applied", option="nondetect", params={"columns": cols})}
        blanked = frame.mask(frame == 0)
        state = ProjectState(**base, missing=MissingSpec(strategy=strategy), findings=answered)
        assert omics.unanswered_zeros(state) == {} and omics.zeros_refusal(["linear"], state) is None
        ctx = {"state": state, "columns": [*feats, "y"], "artifact": lambda name: None,
               "column_info": {c: {"dtype": "numeric", "n_missing": int(blanked[c].isna().sum())}
                               for c in feats}}
        if strategy != "complete_case":
            with pytest.raises(Refusal):
                validate(SetMissing(strategy=strategy), ctx)
            if purpose == "inference":
                with pytest.raises(Refusal):
                    validate(SetMissing(strategy=strategy, reason="a reason"), ctx)
        method = str(rng.choice(["half_minimum", "censoring_aware"]))
        missing = MissingSpec(strategy=strategy, below_detection=method, censored_columns=cols,
                              acknowledged=True)
        state = ProjectState(**base, missing=missing, findings=answered)
        spec = design_spec(state, blanked, feats)
        steps = shared_steps(spec)
        names = [name for name, _ in steps]
        assert names.index("detect") < names.index("log"), (case, names)
        cut = names.index("log") + 1
        upto_log = transformer(steps[:cut]).fit(blanked.iloc[: n * 3 // 4])
        out = upto_log.transform(blanked)
        fill = upto_log.named_steps["detect"].fill_
        assert np.isfinite(out[cols].to_numpy()).all(), case  # nothing left for a median
        for c in cols:
            rows = zero[:, feats.index(c)]
            np.testing.assert_allclose(out.loc[rows, c], np.log2(fill[c]), rtol=1e-12)


# ── 5 · batch ─────────────────────────────────────────────────────────────────


def batched(seed: int = 3, sizes: dict | None = None, p: int = 60) -> pd.DataFrame:
    """Log-scale expression in three batches of unequal size, each batch shifting and scaling every
    gene its own way; a binary outcome moves the first ten genes."""
    rng = np.random.default_rng(seed)
    sizes = sizes or {"B1": 12, "B2": 16, "B3": 9}
    batch = np.concatenate([[b] * k for b, k in sizes.items()])
    n = len(batch)
    y = rng.integers(0, 2, n)
    shift = {b: rng.normal(0, 0.8, p) for b in sizes}
    scale = {b: np.exp(rng.normal(0, 0.3, p)) for b in sizes}
    X = rng.normal(5, 1, (n, p))
    for i, b in enumerate(batch):
        X[i] = X[i] * scale[b] + shift[b] + 0.5 * y[i] * (np.arange(p) < 10)
    frame = pd.DataFrame(X, columns=[f"g{j:02d}" for j in range(p)])
    frame["batch"] = batch
    frame["y"] = y
    return frame


COMBAT_R = r"""
suppressPackageStartupMessages(library(sva))
d <- read.csv("data.csv", check.names = FALSE)
tr <- read.csv("train.csv", check.names = FALSE)
feats <- grep("^g", names(d), value = TRUE)
dat <- t(as.matrix(d[, feats]))
batch <- as.character(d$batch)
mod <- model.matrix(~ y, data = d)
w <- function(m, name) write.csv(t(m), paste0(name, ".csv"), row.names = FALSE)
sink(stderr())
out <- list(
  ref = ComBat(dat, batch, ref.batch = "B2"),
  plain = ComBat(dat, batch),
  mod = ComBat(dat, batch, mod = mod),
  modref = ComBat(dat, batch, mod = mod, ref.batch = "B1"),
  meanref = ComBat(dat, batch, mean.only = TRUE, ref.batch = "B3"),
  train_ref = ComBat(t(as.matrix(tr[, feats])), as.character(tr$batch), ref.batch = tr$ref[1])
)
sink()
for (name in names(out)) w(out[[name]], name)
"""


@needs_r
def test_5a_combat_agrees_with_sva_and_the_in_fold_step_freezes_it_for_held_out_rows(tmp_path):
    """The port against R ``sva::ComBat`` (3.60, parametric empirical Bayes), to 1e-6 (measured:
    about 1e-14): a reference batch; none; covariates (``mod = model.matrix(~ y)``); covariates and
    a reference batch; mean-only with a reference batch.

    The in-fold step (:class:`batch.ReferenceComBat`) fitted on a training fold, its largest batch
    the reference, reproduces ``ComBat(train, ref.batch = <largest>)`` on the training rows, and
    adjusts a held-out row of a batch it has seen by the same map ComBat applied to that batch's
    training rows: within a batch ComBat's adjustment is affine in each gene, so the map is read
    off R's own output (adjusted on raw, per gene, by least squares over the batch's training rows)
    and applied to the held-out values; agreement to 1e-6. A held-out row of the reference batch
    comes back unchanged. The outcome is never in the step's model."""
    frame = batched()
    feats = [c for c in frame.columns if c.startswith("g")]
    rng = np.random.default_rng(0)
    held = np.zeros(len(frame), dtype=bool)
    for b in ("B1", "B2", "B3"):
        rows = np.flatnonzero(frame["batch"].eq(b).to_numpy())
        held[rng.choice(rows, max(2, len(rows) // 4), replace=False)] = True
    train, test = frame[~held].reset_index(drop=True), frame[held].reset_index(drop=True)
    ref_level = B.reference_level(train["batch"])
    out = run_r(COMBAT_R, {"data": frame, "train": train.assign(ref=ref_level)}, tmp_path,
                ("ref", "plain", "mod", "modref", "meanref", "train_ref"))
    X, batch, y = frame[feats].to_numpy(), frame["batch"].to_numpy(), frame["y"].to_numpy()
    mod = np.column_stack([np.ones(len(y)), y])
    mine = {"ref": B.combat(X, batch, ref_batch="B2"), "plain": B.combat(X, batch),
            "mod": B.combat(X, batch, mod=mod), "modref": B.combat(X, batch, mod=mod, ref_batch="B1"),
            "meanref": B.combat(X, batch, mean_only=True, ref_batch="B3")}
    for name, values in mine.items():
        np.testing.assert_allclose(values, out[name].to_numpy(), rtol=0, atol=1e-6, err_msg=name)
    step = B.ReferenceComBat(feats, "batch", drop=True).fit(train.drop(columns=["y"]))
    assert step.reference_ == ref_level == "B2"
    np.testing.assert_allclose(step.transform(train.drop(columns=["y"]))[feats].to_numpy(),
                               out["train_ref"].to_numpy(), rtol=0, atol=1e-6)
    adjusted = step.transform(test.drop(columns=["y"]))
    assert "batch" not in adjusted.columns
    r_train = out["train_ref"].to_numpy()
    for b in ("B1", "B2", "B3"):
        tr_rows = train["batch"].eq(b).to_numpy()
        te_rows = test["batch"].eq(b).to_numpy()
        raw_tr = train.loc[tr_rows, feats].to_numpy()
        expected = np.empty((int(te_rows.sum()), len(feats)))
        for j in range(len(feats)):
            slope, intercept = np.polyfit(raw_tr[:, j], r_train[tr_rows, j], 1)
            expected[:, j] = intercept + slope * test.loc[te_rows, feats[j]].to_numpy()
        np.testing.assert_allclose(adjusted.loc[te_rows, feats].to_numpy(), expected, rtol=0,
                                   atol=1e-6, err_msg=b)
        if b == ref_level:
            np.testing.assert_allclose(adjusted.loc[te_rows, feats].to_numpy(),
                                       test.loc[te_rows, feats].to_numpy(), rtol=0, atol=0)


def test_5b_reference_combat_is_training_fold_scope_and_never_reads_the_outcome():
    """Observed scope: a study row's adjusted values move when another study row moves (the
    training fold's batch estimates), never when the outcome does: training fold, as the
    contract's reference-ComBat option declares; outcome-protected ComBat reads the outcome."""
    frame = batched(seed=5)
    feats = [c for c in frame.columns if c.startswith("g")]
    reference = np.zeros(len(frame), dtype=bool)
    y = frame["y"].to_numpy()

    def ref_combat(f: pd.DataFrame, r: pd.Series, yy: np.ndarray) -> pd.DataFrame:
        return B.ReferenceComBat(feats, "batch").fit(f).transform(f)

    def outcome_combat(f: pd.DataFrame, r: pd.Series, yy: np.ndarray) -> pd.DataFrame:
        mod = np.column_stack([np.ones(len(yy)), yy])
        return pd.DataFrame(B.combat(f[feats].to_numpy(), f["batch"].to_numpy(), mod=mod), index=f.index)

    data = frame.drop(columns=["y"])
    assert observed_scope(ref_combat, data, reference, y, 4, columns=feats) == "training_fold"
    assert CONTRACTS["batch"].scope_of("reference_combat") == "training_fold"
    assert observed_scope(outcome_combat, data, reference, y, 4, columns=feats) == "model"
    assert CONTRACTS["batch"].scope_of("outcome_combat") == "model"


def _batch_ctx(purpose: str, state_extra: dict | None = None, findings: list | None = None) -> dict:
    roles = {"g00": "exposure", "g01": "exposure", "batch": "excluded"}
    state = ProjectState(target="y", purpose=purpose, roles=roles, **(state_extra or {}))
    return {"state": state, "columns": ["g00", "g01", "batch", "y"],
            "column_info": {"batch": {"dtype": "numeric", "n_unique": 3, "n_missing": 0}},
            "artifact": lambda name: {"findings": findings or []} if name == "findings" else None}


def test_5c_batch_is_a_covariate_first_under_inference_and_reference_combat_first_under_prediction():
    """North star 5's order and the leash of §4: under inference batch as a covariate ranks first
    (Nygaard et al. 2016: "account for batch in the statistical analysis"), under prediction
    reference ComBat fitted in-fold without the outcome (Zhang et al. 2018); outcome-protected
    ComBat is refused under both. As a covariate the batch enters as categories, so an unsettled
    role or a whole-number code is asked in one block first (BLUEPRINT §14.2)."""
    inference = options_for("batch", "inference")
    prediction = options_for("batch", "prediction")
    assert [o["key"] for o in inference][:2] == ["covariate", "reference_combat"]
    assert [o["key"] for o in prediction][:2] == ["reference_combat", "covariate"]
    assert inference[0]["rung"] == "recommended" and prediction[0]["rung"] == "recommended"
    assert {o["key"]: o["rung"] for o in inference}["outcome_combat"] == "refused"
    assert {o["key"]: o["rung"] for o in prediction}["outcome_combat"] == "refused"
    with pytest.raises(Refusal) as refused:
        validate(SetBatch(column="batch", method="covariate"), _batch_ctx("inference"))
    assert refused.value.code == "batch_covariate_unsettled"
    block = refused.value.exits[0]["decision"]
    assert block["kind"] == "confirm_readings"
    assert [(i["reading"], i["value"]) for i in block["items"]] == [("role", "covariate"),
                                                                   ("code_or_count", "code")]
    settled = {"roles": {"g00": "exposure", "g01": "exposure", "batch": "covariate"},
               "shape_confirmations": {"code_or_count:batch": "code"}}
    ctx = _batch_ctx("inference")
    ctx["state"] = ProjectState(target="y", purpose="inference", **settled)
    validate(SetBatch(column="batch", method="covariate", figures=True), ctx)
    validate(SetBatch(column="batch", method="reference_combat"), _batch_ctx("prediction"))
    validate(SetBatch(column="batch", method="reference_combat"), _batch_ctx("inference"))


def test_5d_outcome_protected_combat_is_refused_for_testing_and_under_prediction():
    """Nygaard et al. 2016: protecting the outcome in the batch model "may systematically induce
    incorrect group differences in downstream analyses when groups are distributed between the
    batches in an unbalanced manner". Under inference it is refused for testing, with batch as a
    covariate (and ComBat for figures only) as the exits; under prediction it is refused as
    leakage, the exit reference ComBat in each training fold."""
    with pytest.raises(Refusal) as refused:
        validate(SetBatch(column="batch", method="outcome_combat"), _batch_ctx("inference"))
    assert refused.value.code == "outcome_combat_for_testing"
    exits = [e["decision"] for e in refused.value.exits]
    assert exits[0] == {"kind": "set_batch", "column": "batch", "method": "covariate", "figures": True}
    assert exits[1]["method"] == "covariate" and exits[1]["figures"] is False
    with pytest.raises(Refusal) as refused:
        validate(SetBatch(column="batch", method="outcome_combat"), _batch_ctx("prediction"))
    assert refused.value.code == "outcome_combat_leaks"
    assert refused.value.exits[0]["decision"]["method"] == "reference_combat"
    with pytest.raises(Refusal) as refused:
        validate(SetBatch(column="batch", method="reference_combat", figures=True),
                 _batch_ctx("prediction"))
    assert refused.value.code == "figures_under_prediction"


class _Store:
    """The two calls the design's backstop makes of a DataStore."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.columns = list(frame.columns)

    def materialize(self, columns, rows=None):
        return self.frame.loc[:, list(columns)] if rows is None else self.frame.loc[rows, list(columns)]


def test_5e_a_batch_perfectly_confounded_with_the_outcome_is_refused_under_both_purposes():
    """Every batch holding one outcome level: no model can tell batch from outcome. Reference:
    pandas' crosstab (each batch's row has one non-zero cell), and for an imbalance short of
    that, Cramér's V by scipy (``contingency.association``). The findings stage reads it over
    every row (descriptive scope); the batch answer, the model selection and the design are
    refused under prediction and inference alike, until the user says the column is not a batch."""
    from scipy.stats.contingency import association

    from turbotab.core.decisions import SelectModels

    frame = pd.DataFrame({"batch": ["P1"] * 10 + ["P2"] * 12 + ["P3"] * 8,
                          "y": ["case"] * 10 + ["control"] * 12 + ["case"] * 8,
                          "g": np.arange(30.0)})
    crosstab = pd.crosstab(frame["batch"], frame["y"])
    assert ((crosstab > 0).sum(axis=1) == 1).all()
    found = B.confounding(frame["batch"], frame["y"], "binary", "batch", "y")
    assert found.perfect
    uneven = frame.assign(y=["case"] * 7 + ["control"] * 3 + ["control"] * 10 + ["case"] * 2 + ["case"] * 8)
    v = B.confounding(uneven["batch"], uneven["y"], "binary").cramers_v
    assert v == pytest.approx(association(pd.crosstab(uneven["batch"], uneven["y"]).to_numpy(),
                                          method="cramer", correction=False), rel=1e-12)
    raised = B.batch_findings(frame, ["metabolomics"], "y")
    assert len(raised) == 1 and raised[0][1]["severity"] == "critical"
    findings = [raised[0][1]]
    for purpose in ("prediction", "inference"):
        ctx = _batch_ctx(purpose, findings=findings)
        with pytest.raises(Refusal) as refused:
            validate(SetBatch(column="batch", method="covariate"), ctx)
        assert refused.value.code == "batch_confounded_with_outcome"
        not_a_batch = refused.value.exits[1]["decision"]
        assert not_a_batch == {"kind": "set_batch", "column": "batch", "method": "not_a_batch",
                               "figures": False}
        validate(SetBatch(**{k: v for k, v in not_a_batch.items() if k != "kind"}), ctx)
        with pytest.raises(Refusal) as refused:
            B._no_model_while_confounded(SelectModels(models=["linear"]), ctx)
        assert refused.value.code == "batch_confounded_with_outcome"
        answered = _batch_ctx(purpose, {"batch": {"column": "batch", "method": "not_a_batch"}}, findings)
        B._no_model_while_confounded(SelectModels(models=["linear"]), answered)
        state = ProjectState(target="y", purpose=purpose, batch={"column": "batch", "method": "covariate"})
        assert "perfectly confounded" in B.design_refusal(state, _Store(frame), frame.index, "binary")
        assert B.design_refusal(answered["state"], _Store(frame), frame.index, "binary") is None


def test_5f_batch_correction_precedes_in_fold_screening():
    """MODELING_SEQUENCE §2: batch correction precedes in-fold screening. The batch step is a shared
    step and the screen is the screened family's own, so the pipeline runs batch before screen; and
    the order matters. Simulation (120 samples, 400 null features, 40 of them shifted by one SD in
    batch B2, where 75% of the cases were measured): screening the raw values keeps mostly the
    batch-affected features, because batch predicts the outcome; after reference ComBat, fitted on
    the same training rows without the outcome, the share kept falls to about the share planted
    (10%). Bounds: above 50% without the batch step, below 25% with it."""
    from turbotab.core.models import get_family
    from turbotab.core.models.pipeline import DesignSpec, build_pipeline

    rng = np.random.default_rng(17)
    n, p = 120, 400
    y = np.r_[np.ones(60), np.zeros(60)]
    in_b2 = np.where(y == 1, rng.random(n) < 0.75, rng.random(n) < 0.25)
    X = rng.normal(0, 1, (n, p))
    X[np.ix_(in_b2, np.arange(40))] += 1.0
    feats = [f"x{j:03d}" for j in range(p)]
    frame = pd.DataFrame(X, columns=feats)
    frame["batch"] = np.where(in_b2, "B2", "B1")
    spec = DesignSpec(predictors=feats, inputs=[*feats, "batch"], categorical=[], numeric=feats,
                      energy=None, impute=False, roles={c: "exposure" for c in feats},
                      batch={"column": "batch", "method": "reference_combat", "columns": feats,
                             "drop": True})
    family = get_family("screened_elastic_net")
    pipe = build_pipeline(spec, family, "binary", "prediction", n, p)
    names = [name for name, _ in pipe.steps]
    assert names.index("batch") < names.index("screen") < names.index("scale")
    screened = omics.UnivariateScreen(feats)
    raw = screened.fit(frame[feats], y).kept_
    corrected = B.ReferenceComBat(feats, "batch").fit(frame).transform(frame)
    after = omics.UnivariateScreen(feats).fit(corrected[feats], y).kept_
    share = lambda kept: sum(int(c[1:]) < 40 for c in kept) / len(kept)  # noqa: E731
    assert len(raw) == omics.sis_size(n) == 25
    assert share(raw) > 0.5 and share(after) < 0.25, (share(raw), share(after))
    batch_relations = {(r.kind, r.target) for r in CONTRACTS["batch"].relations}
    assert ("precedes", "screen") in batch_relations


# ── 6 · an exposure family's multiplicity ────────────────────────────────────

BH_R = r"""
d <- read.csv("p.csv")
write.csv(data.frame(q = p.adjust(d$p, method = "BH")), "q.csv", row.names = FALSE)
"""


@needs_r
def test_6a_bh_q_values_agree_with_r_p_adjust(tmp_path):
    """Benjamini–Hochberg q-values (featurewise's ``bh_adjust`` and the drift detector's
    ``benjamini_hochberg``) against R ``p.adjust(p, method = "BH")`` to 1e-12, on 500 p-values with
    ties, ones, exact zeros and blanks (R leaves a blank out of the count, as both do)."""
    from turbotab.core.detectors.assay import benjamini_hochberg
    from turbotab.core.models.featurewise import bh_adjust

    rng = np.random.default_rng(9)
    p = np.r_[rng.uniform(0, 1, 400), rng.beta(0.3, 6, 80), [0.0, 0.0, 1.0, 1.0, 0.02, 0.02,
                                                               0.02, 0.5, 0.5, 0.5]]
    p = np.round(p, 6)  # ties
    p[rng.choice(len(p), 10, replace=False)] = np.nan
    q = run_r(BH_R, {"p": pd.DataFrame({"p": p})}, tmp_path, ("q",))["q"]["q"].to_numpy()
    for mine in (bh_adjust(p), benjamini_hochberg(p)):
        np.testing.assert_allclose(mine, q, rtol=0, atol=1e-12, equal_nan=True)


def _family_ctx(n_exposures: int) -> dict:
    roles = {f"m{j}": "exposure" for j in range(n_exposures)}
    return {"state": ProjectState(target="y", purpose="inference", roles=roles),
            "columns": [*roles, "y"], "artifact": lambda name: None}


def test_6b_an_exposure_family_without_multiplicity_control_is_blocked_and_recorded():
    """MODELING_SEQUENCE §2: "An exposure family implies multiplicity control: BH q-values for
    feature-wise analyses; for a few prespecified nutrient hypotheses, the number of tests stated."
    No control at all, or unadjusted p-values for a family of more than ten tests, is refused with
    Benjamini–Hochberg and the recorded attestation as its exits, and kept only with the
    attestation (block and record). Unadjusted p-values for a few tests, with their number stated,
    are customary (Rothman 1990) and accepted."""
    ctx = _family_ctx(300)
    for method in ("none", "stated_count"):
        with pytest.raises(Refusal) as refused:
            validate(SetMultiplicity(method=method), ctx)
        assert refused.value.code == "family_without_multiplicity"
        exits = [e["decision"] for e in refused.value.exits]
        assert exits == [{"kind": "set_multiplicity", "method": "bh", "acknowledged": False},
                         {"kind": "set_multiplicity", "method": method, "acknowledged": True}]
        validate(SetMultiplicity(method=method, acknowledged=True), ctx)
    validate(SetMultiplicity(method="bh"), ctx)
    validate(SetMultiplicity(method="stated_count"), _family_ctx(5))
    assert options_for("multiplicity", "inference")[0]["key"] == "bh"


def test_6c_the_feature_wise_table_carries_the_recorded_method_and_every_member():
    """Unanswered, the family's table carries Benjamini–Hochberg q-values (implied, not asked).
    Recorded as no control (with its attestation) or as unadjusted with the number of tests stated,
    the q column is empty and the caption says so; every member is shown either way."""
    from turbotab.core.models.featurewise import featurewise_table

    rng = np.random.default_rng(4)
    n, m = 60, 120
    feats = [f"m{j}" for j in range(m)]
    matrix = pd.DataFrame(rng.normal(size=(n, m)), columns=feats)
    y = rng.normal(size=n) + 0.8 * matrix["m0"].to_numpy()
    implied = omics.multiplicity_policy(ProjectState())
    assert implied == {"method": "bh", "acknowledged": False, "recorded": False}
    table = omics.apply_multiplicity(featurewise_table(matrix, y, "regression", feats), implied)
    assert all(r["q"] is not None for r in table.rows)
    assert "Benjamini–Hochberg q < 0.05" in table.info["caption"]
    for method, said in (("none", "No multiplicity control, recorded as a limitation: 120 tests"),
                         ("stated_count", "Unadjusted p-values for 120 tests, stated as such")):
        state = ProjectState(multiplicity={"method": method, "acknowledged": True})
        table = omics.apply_multiplicity(featurewise_table(matrix, y, "regression", feats),
                                         omics.multiplicity_policy(state))
        assert len(table.rows) == m and all(r["q"] is None for r in table.rows)
        assert said in table.info["caption"] and "Benjamini–Hochberg q <" not in table.info["caption"]


# ── 7 · every method enters through its contract ─────────────────────────────

MS7_CONTRACTS = ("qc_detection_filter", "qc_rlsc", "qc_rsd_filter", "qc_pqn", "qc_rows_leave",
                 "d_ratio_filter", "omics_normalization", "detection_limit", "log_transform",
                 "batch", "screen", "autoscaling", "multiplicity")


def test_7_every_ms7_method_enters_through_its_contract_in_the_run_order_of_section_1_1():
    """BLUEPRINT §13: each method declares its slot, its data scope, what it needs, its routing
    (a question, a place in the run order, its options labeled customary and sound for each
    purpose with a rung for each), its storyboard, its sentence and its relations. The registry
    orders them as MODELING_SEQUENCE §1.1 runs them: before the seal on reference rows (the QC
    filters around QC-RLSC, PQN to the pooled QCs), the QC rows leaving, then in each training fold
    the D-ratio, the normalization, the detection-limit fill, the log, batch, screening and
    scaling; the exposure family's multiplicity at evaluation. A method that runs before the seal
    has a scope that may (row-local, reference rows); none reads a participant there."""
    from turbotab.core.contracts import (BEFORE_THE_SEAL, PRE_SEAL_SCOPES, PURPOSES, SCOPES,
                                        SLOTS, run_order)

    sentence_from_sibling = {"qc_detection_filter", "qc_rsd_filter", "qc_pqn", "qc_rows_leave",
                             "multiplicity"}  # stated in QC-RLSC's, the normalization's, the table's
    for key in MS7_CONTRACTS:
        c = CONTRACTS[key]
        assert c.slot in SLOTS and c.scope in SCOPES, key
        assert c.needs and c.question and c.storyboard and c.sources and c.options, key
        for o in c.options:
            assert o.customary and o.label
            for purpose in PURPOSES:
                assert o.sound[purpose] and o.rung[purpose] and o.order[purpose] >= 0
        assert c.short or c.clause is not None or key in sentence_from_sibling, key
        if c.slot in BEFORE_THE_SEAL:
            assert c.scope in PRE_SEAL_SCOPES, key
    assert run_order({k: None for k in MS7_CONTRACTS}) == [
        "qc_detection_filter", "qc_rlsc", "qc_rsd_filter", "qc_pqn", "qc_rows_leave",
        "d_ratio_filter", "omics_normalization", "detection_limit", "log_transform", "batch",
        "screen", "autoscaling", "multiplicity"]
    declared = {(key, r.kind, r.target, r.rung) for key in MS7_CONTRACTS for r in CONTRACTS[key].relations}
    assert {("qc_rlsc", "implies", "qc_rows_leave", None),
            ("qc_rlsc", "precedes", "omics_normalization", None),
            ("omics_normalization", "conflicts", "linear_families", "refused"),
            ("detection_limit", "implies", "censoring_aware", None),
            ("detection_limit", "conflicts", "mar_imputation", "refused"),
            ("detection_limit", "precedes", "log_transform", None),
            ("log_transform", "conflicts", "zeros", "refused"),
            ("d_ratio_filter", "conflicts", "before_the_seal", "refused"),
            ("batch", "conflicts", "perfect_confounding", "refused"),
            ("batch", "conflicts", "testing", "refused"),
            ("batch", "conflicts", "leakage", "refused"),
            ("batch", "precedes", "screen", None),
            ("multiplicity", "implies", "bh_q_values", None),
            ("multiplicity", "conflicts", "unadjusted_family", "block_and_record")} <= declared



# ── 8 · the record's sentences, verbatim ─────────────────────────────────────


def test_8_each_answer_writes_its_methods_sentence_verbatim():
    """The sentence the Record shows for each new answer (``voice.sentence_for``), word for word:
    the batch answers, the multiplicity answers, and the pooled-QC answer with its own counts."""
    from turbotab.core import voice
    from turbotab.core.decisions import ApplyRepair

    said = {
        SetBatch(column="batch", method="covariate", figures=True):
            "`batch` enters the outcome model as a covariate, so batch differences are not read as "
            "biology (Nygaard, Rødland & Hovig 2016); ComBat with the outcome protected serves the "
            "figures only, never the tests.",
        SetBatch(column="batch", method="reference_combat"):
            "Batch effects across `batch` are removed by ComBat with a reference batch, the largest "
            "batch of each training fold, fitted without the outcome; held-out rows are adjusted by "
            "the fold's estimates (Zhang et al. 2018), and a batch absent from the fold by add-on "
            "adjustment (Hornung et al. 2017).",
        SetBatch(column="plate", method="not_a_batch"):
            "`plate` is not a batch; it is read as an ordinary column.",
        SetMultiplicity(method="bh"):
            "The exposures are tested as one family with Benjamini–Hochberg q-values (Benjamini & "
            "Hochberg 1995); every member is shown.",
        SetMultiplicity(method="stated_count"):
            "The exposures' p-values are unadjusted, with the number of tests stated (Rothman "
            "1990); every member is shown.",
        SetMultiplicity(method="none", acknowledged=True):
            "The exposures' tests carry no multiplicity control, recorded as a limitation: chance "
            "findings cannot be told from discoveries; every member is shown.",
    }
    for decision, sentence in said.items():
        assert voice.sentence_for(decision, None, None) == sentence, decision
    params = {"column": "sample_type", "qc_levels": ["QC"], "order_column": "injection_order",
              "batch_column": "batch", "features": [f"m{j}" for j in range(40)], "n_qc": 27,
              "platform": "gc_ms", "d_ratio_max": 0.5, "batches": 3, "kept": 36,
              "dropped": {"detection": ["m1", "m2"], "rsd": ["m3", "m4"]}}
    assert voice.sentence_for(ApplyRepair(finding_id=Q.FINDING, option="qc_rlsc_gc_dratio",
                                          params=params), None, None) == (
        "Drift was corrected per batch by QC-RLSC (Dunn et al. 2011) fitted to the 27 pooled QC "
        "injections only, which were then removed: for each feature and each of the 3 batches of "
        "`batch`, a LOESS of degree 2 over the injection order (`injection_order`), its span chosen "
        "by leave-one-out cross-validation, divided each injection's value, rescaled to the "
        "feature's median QC value. Features detected in fewer than 70% of QC injections (2), that "
        "could not be corrected (0), or with a QC RSD of 30% or more after correction (the GC-MS "
        "criterion; 2) were removed (Broadhurst et al. 2018); 36 of 40 remain. Within each training "
        "fold, features whose D-ratio (QC over study-sample standard deviation) was 50% or more were "
        "removed.")
    one = {**params, "batch_column": None, "batches": 1, "d_ratio_max": None, "platform": "lc_ms"}
    assert voice.sentence_for(ApplyRepair(finding_id=Q.FINDING, option="qc_rlsc_lc", params=one),
                              None, None) == (
        "Drift was corrected over the whole run by QC-RLSC (Dunn et al. 2011) fitted to the 27 "
        "pooled QC injections only, which were then removed: for each feature and the whole run as "
        "one batch (no batch column), a LOESS of degree 2 over the injection order "
        "(`injection_order`), its span chosen by leave-one-out cross-validation, divided each "
        "injection's value, rescaled to the feature's median QC value. Features detected in fewer "
        "than 70% of QC injections (2), that could not be corrected (0), or with a QC RSD of 20% or "
        "more after correction (the LC-MS criterion; 2) were removed (Broadhurst et al. 2018); 36 of "
        "40 remain.")
    aside = {"column": "sample_type", "levels": ["QC"]}  # WP18's exclusion, offered beside QC-RLSC
    assert voice.sentence_for(ApplyRepair(finding_id=Q.FINDING, option=Q.ASIDE, params=aside),
                              None, None) == (
        "The rows where `sample_type` is `QC` (instrument runs, not participants) were excluded as "
        "reference rows before the held-out rows were drawn.")
