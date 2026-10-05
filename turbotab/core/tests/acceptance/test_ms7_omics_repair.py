"""MS7 repair round: what the independent verifier found open in the omics package, each closed by a
permanent test with an expected value from a path independent of the code under test (R 4.6.1 with
``sva`` through ``Rscript``, pandas written from a method's definition, or a simulation's truth).

1. QC-RLSC fits each feature to its *detected* QCs, so a feature whose last QCs in a batch fall
   below the detection limit while study samples stay above it had its degree-2 curve
   extrapolated (values inflated about 17×). Such a feature is now uncorrectable and leaves.
2. The injection order and the batch QC-RLSC reads are readings (BLUEPRINT §14): named in every
   option, a ``run`` column read as a batch, the whole run as one batch offered beside it, the
   user's choice settling them, and the ledger registering both.
3. The censored share behind half-minimum's rung counted blanks over every row: 0 when an export
   wrote non-detects as zeros, diluted by the pooled QCs. It is now the participants' share of
   blanks and recoded zeros, read from the table the analysis reads.
4. QRILC left a sample too sparse to read with blanks for the median fill.
5. ComBat tested constancy within a batch by numpy's variance == 0, which is ~1e-30 for many
   constant vectors where R's ``var`` is exactly 0, so sva and the port disagreed by up to 5 log2
   units on such a feature.
6. The assay block counted a run-order column as an intensity, and the pack said "I couldn't find
   any pooled QC samples" beside the QC rows it had found.
7. The fill's sentence in the methods paragraph counted the columns the answer named, said "on the
   training fold" under inference and "after normalization" where there was none. It now reads
   the run's records (the chains in ``test_ms7_chains.py`` read it from the fit artifact).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from turbotab.core.contracts import CONTRACTS, observed_scope
from turbotab.core.decisions import (FindingDisposition, ProjectState, Refusal, SetMissing,
                                     validate)
from turbotab.core.methods import batch as B
from turbotab.core.methods import omics
from turbotab.core.methods import qc_drift as Q
from turbotab.core.tests.acceptance.omics_references import (drifting_run, needs_r, run_r,
                                                             uncorrectable_by_hand)

FEATURES = lambda frame: [c for c in frame.columns if c.startswith("m")]  # noqa: E731


# ── 1 · QC-RLSC never extrapolates a feature's curve ─────────────────────────


def _late_nondetects(seed: int = 7, p: int = 60) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Two batches of 45 injections (a pooled QC every fifth and last), drifting. In the first
    third of the features the last three QCs of B2 fall below the detection limit while the study
    samples after them stay detected (the realistic case: downward drift); in the second third the
    study samples after them are below the limit too; the rest are untouched."""
    frame, truth = drifting_run(seed=seed, p=p, batches=2, per_batch=45)
    feats = FEATURES(frame)
    b2 = frame["batch"].eq("B2").to_numpy()
    qc = frame["sample_type"].eq("QC").to_numpy()
    order = frame["injection_order"].to_numpy()
    qcs = np.flatnonzero(b2 & qc)
    last3 = qcs[-3:]
    after = np.flatnonzero(b2 & (order > order[qcs[-4]]))  # past the last QC still detecting
    extrapolated, silent = feats[: p // 3], feats[p // 3: 2 * p // 3]
    frame.loc[last3, extrapolated + silent] = np.nan
    frame.loc[after, silent] = np.nan
    return frame, truth, extrapolated, silent


def test_1f_a_feature_whose_detected_qcs_end_before_its_detected_samples_is_not_corrected():
    """Expected (pandas, ``uncorrectable_by_hand``: the rule by its definition): the features with
    a detected study value outside the injection-order span of the QCs that detected them in some
    batch, exactly the first third here. They are listed uncorrectable with the reason, leave with
    the QC filters, and keep their values; no curve is evaluated past its last detected QC (blank
    there). The other features recover the truth: within each, the SD over study injections of
    log(corrected / truth) is at most 0.09 (noise 0.05; measured about 0.06), and the largest
    |log error| after removing each feature's constant is at most 0.3 (6 noise SDs; the verifier's
    extrapolated values missed by 2.1, about 17-fold)."""
    frame, truth, extrapolated, silent = _late_nondetects()
    feats = FEATURES(frame)
    assert uncorrectable_by_hand(frame, feats) == set(extrapolated)
    inj = Q.injections(frame, "sample_type", ["QC"], "injection_order", "batch")
    values = frame[feats].to_numpy(dtype=float)
    fit = Q.qc_rlsc(values, inj)
    assert {feats[j] for j in fit.uncorrectable} == set(extrapolated)
    for j in fit.uncorrectable:
        assert "outside its detected QCs in batch `B2`: the curve would be extrapolated" in fit.reasons[j]
        np.testing.assert_array_equal(fit.corrected[:, j], values[:, j])
    result = Q.run_reference_rows(frame, Q.QCPlan("sample_type", ("QC",), tuple(feats),
                                                  "injection_order", "batch"))
    assert set(result.record["dropped"]["uncorrectable"]) == set(extrapolated)
    assert not set(result.kept) & set(extrapolated)
    order = frame["injection_order"].to_numpy()
    b2 = frame["batch"].eq("B2").to_numpy()
    qc = frame["sample_type"].eq("QC").to_numpy()
    for c in silent:
        j = feats.index(c)
        last = order[b2 & qc & np.isfinite(values[:, j])].max()
        assert np.isnan(fit.curves[b2 & (order > last), j]).all()
    study = ~qc
    for c in [c for c in feats if c not in extrapolated]:
        j = feats.index(c)
        ok = study & np.isfinite(values[:, j])
        err = np.log(fit.corrected[ok, j] / truth[ok, j])
        assert err.std(ddof=1) <= 0.09, (c, err.std(ddof=1))
        assert np.max(np.abs(err - np.median(err))) <= 0.3, (c, np.max(np.abs(err - np.median(err))))


# ── 2 · the injection order and the batch are readings ───────────────────────


def _offers(frame: pd.DataFrame, target: str | None = None) -> list:
    from turbotab.core.repairs import OfferContext, offer

    return offer({"id": Q.FINDING, "repairs": []},
                 {"params": {"column": "sample_type", "qc_value": "QC"}},
                 OfferContext(frame=frame, target=target))


def test_2a_every_option_names_the_order_and_batch_it_reads_and_a_run_column_is_read_as_the_batch():
    """BLUEPRINT §14: an interpretation that changes every value is a reading the user sees and
    settles. Each QC-RLSC option's consequence names the injection-order column and the batch
    column (or says one curve), and its sentence names both. A batch column named ``run`` (the
    verifier's case: one curve was fitted across batches and nothing asked) is the leading reading;
    the whole run as one batch is offered beside it with the LC-MS and GC-MS filters. The
    registry declares both kinds settled only by the user, and the consumers are registered:
    the offer (words only) and the plan the working table runs (the user's recorded answer)."""
    from turbotab.core import readings as R

    frame, _ = drifting_run(seed=1, p=30)
    run = frame.rename(columns={"batch": "run"})
    options = _offers(run)
    assert [(o.key, o.decision.params["batch_column"]) for o in options[:-1]] == [
        *[(k, "run") for k in Q.RLSC_OPTIONS], ("qc_rlsc_lc", None), ("qc_rlsc_gc", None)]
    assert options[-1].key == Q.ASIDE
    for o in options[:-1]:
        assert o.decision.params["order_column"] == "injection_order"
        assert "`injection_order`" in o.consequence and "`injection_order`" in o.sentence
        if o.decision.params["batch_column"]:
            assert "per `run`" in o.consequence and "batches of `run`" in o.sentence
        else:
            assert o.consequence.startswith("One drift curve") and "no batch column" in o.sentence
    assert [o.label for o in options[4:6]] == ["One curve; LC-MS", "One curve; GC-MS"]
    for kind in ("injection_order", "batch"):
        rule = R.KIND_RULES[kind]
        assert rule.test is None and "user" in rule.settled_by and rule.alternatives
    consumers = {c.where: c for c in R.CONSUMERS}
    assert consumers["turbotab.core.methods.qc_drift:reference_plan"].path == R.USER_APPLIED
    assert consumers["turbotab.core.methods.qc_drift:rlsc_offer"].changes is False
    assert "acquisition" in consumers["turbotab.core.methods.qc_drift:reference_plan"].census


def test_2b_the_batch_reading_the_user_chooses_is_the_one_that_runs_and_per_batch_recovers_better():
    """The plan the working table runs is the chosen option's: with ``run`` as the batch, one curve
    per batch; with one curve, a single curve over the run. Simulation truth (three batches with
    offsets of SD 0.3 on the log scale): per batch the median within-feature SD of log(corrected /
    truth) is at most 0.065, and one curve over the run does worse (the verifier measured 0.058
    against 0.084): a batch's jump is no drift a smooth curve can follow."""
    frame, truth = drifting_run(seed=4, p=80, batch_sd=0.3)
    run = frame.rename(columns={"batch": "run"})
    options = _offers(run)
    study = run["sample_type"].eq("Sample").to_numpy()
    feats = FEATURES(run)
    spread = {}
    for o in (options[0], options[4]):
        params = o.decision.params
        state = ProjectState(findings={Q.FINDING: FindingDisposition(
            action="applied", option=o.key, params=params)})
        plan = Q.reference_plan(state)
        assert (plan.order_column, plan.batch_column) == ("injection_order", params["batch_column"])
        result = Q.run_reference_rows(run, plan)
        done = [feats.index(c) for c in result.kept]
        err = np.log(result.values[np.ix_(study, done)] / truth[np.ix_(study, done)])
        spread[params["batch_column"]] = float(np.median(err.std(axis=0, ddof=1)))
    assert spread["run"] <= 0.065 and spread[None] > spread["run"] + 0.01, spread


class _Store:
    """The two calls the design's backstop makes of a DataStore."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.columns = list(frame.columns)

    def materialize(self, columns, rows=None):
        return self.frame.loc[:, list(columns)] if rows is None else self.frame.loc[rows, list(columns)]


def test_2c_a_run_column_is_a_batch_for_the_confounding_check_too():
    """One reading of what names a batch (``readings.BATCH_KINDS``): a ``run`` column every level of
    which holds one outcome level (pandas' crosstab: one non-zero cell per row) is perfectly
    confounded with the outcome, a critical finding, and the design refuses under both purposes,
    as for a column named ``batch``."""
    frame = pd.DataFrame({"run": ["R1"] * 10 + ["R2"] * 12 + ["R3"] * 8,
                          "y": ["case"] * 10 + ["control"] * 12 + ["case"] * 8,
                          "g": np.arange(30.0)})
    assert ((pd.crosstab(frame["run"], frame["y"]) > 0).sum(axis=1) == 1).all()
    raised = B.batch_findings(frame, ["metabolomics"], "y")
    assert [(f["id"], f["severity"]) for _, f in raised] == [("batch_confounding__run", "critical")]
    for purpose in ("prediction", "inference"):
        state = ProjectState(target="y", purpose=purpose)
        assert "perfectly confounded" in B.design_refusal(state, _Store(frame), frame.index, "binary")


# ── 3 · the censored share is the participants', zeros recoded included ────


class _ParquetStore:
    """The server's ``store()``: a real DataStore over the table, ingested as the app ingests."""

    def __init__(self, frame: pd.DataFrame, folder):
        from turbotab.core.datastore import DataStore, ingest

        source = folder / "table.csv"
        frame.to_csv(source, index=False)
        ingest(source, folder / "table.parquet")
        self.store = DataStore(folder / "table.parquet", 1 << 28)

    def __call__(self):
        return self.store


def _zeros_table(seed: int = 3, n_study: int = 120, n_qc: int = 24) -> tuple[pd.DataFrame, list[str]]:
    """Participants and pooled QCs; in the five censored features the export wrote each non-detect
    as 0: 30–60% of the participants' values, none of the QCs' (a pooled sample sits mid-range)."""
    rng = np.random.default_rng(seed)
    n = n_study + n_qc
    frame = pd.DataFrame({"sample_type": ["Sample"] * n_study + ["QC"] * n_qc,
                          "y": np.r_[rng.normal(size=n_study), np.full(n_qc, np.nan)]})
    censored = [f"mz_{j}" for j in range(5)]
    for j, c in enumerate(censored):
        x = np.exp(rng.normal(6, 1, n))
        share = 0.30 + 0.075 * j
        x[:n_study][x[:n_study] < np.quantile(x[:n_study], share)] = 0.0
        x[n_study:] = np.exp(6 + rng.normal(0, 0.05, n_qc))
        frame[c] = x
    return frame, censored


def test_3a_half_minimum_is_blocked_above_ten_percent_when_non_detects_are_exported_as_zeros(tmp_path):
    """The verifier's case, through the server's DataStore over the table the analysis reads: under
    inference, half the minimum at up to 60% below the limit was accepted because the share was
    blanks over every row, 0 with zeros. Expected (pandas): the largest share of zeros among the
    participants (the QC rows left by the recorded exclusion), 0.60. Read from the table before
    the working table recomputed (zeros, QC rows in it) and after (blanks, QC rows gone): the same
    share, and half the minimum is blocked and recorded either way; the median refusal's
    half-minimum exit says it is customary only to 10%. Under prediction QRILC is offered beside
    the censored normal at this share (Wei et al. 2018)."""
    frame, censored = _zeros_table()
    study = frame["sample_type"].eq("Sample")
    expected = max(float((frame.loc[study, c] == 0).mean()) for c in censored)
    assert expected == pytest.approx(0.60, abs=0.01)
    findings = {"pack::metabolomics::zeros_or_missing": FindingDisposition(
                    action="applied", option="nondetect", params={"columns": censored}),
                "pack::metabolomics::pooled_qc": FindingDisposition(
                    action="applied", option="exclude_rows",
                    params={"column": "sample_type", "levels": ["QC"]})}
    stale = frame
    fresh = frame.loc[study].assign(**{c: frame.loc[study, c].replace(0.0, np.nan) for c in censored})
    for name, table in (("stale", stale), ("fresh", fresh)):
        folder = tmp_path / name
        folder.mkdir()
        store = _ParquetStore(table, folder)
        info = {c: {"dtype": "numeric", "n_missing": int(table[c].isna().sum())} for c in censored}
        for purpose in ("inference", "prediction"):
            state = ProjectState(target="y", purpose=purpose, findings=findings,
                                 roles={c: "exposure" for c in censored})
            ctx = {"state": state, "columns": list(table.columns), "column_info": info,
                   "n_rows": len(table), "store": store, "artifact": lambda name: None}
            assert omics.censored_share_of(ctx, censored) == pytest.approx(expected, abs=1e-12)
            half = SetMissing(strategy="multiple_imputation" if purpose == "inference" else "impute",
                              censored_columns=censored, below_detection="half_minimum")
            if purpose == "inference":
                with pytest.raises(Refusal) as refused:
                    validate(half, ctx)
                assert refused.value.code == "half_minimum_above_ten_percent", name
                assert "Up to 60%" in refused.value.message
                kept = refused.value.exits[-1]["decision"]
                validate(SetMissing(**{k: v for k, v in kept.items() if k != "kind"}), ctx)
            else:
                validate(half, ctx)
            with pytest.raises(Refusal) as refused:
                validate(SetMissing(strategy=half.strategy, censored_columns=censored), ctx)
            labels = [e["label"] for e in refused.value.exits]
            assert "Half the smallest detected value (customary only to 10%; up to 60% here)" in labels
            fills = [e["decision"]["below_detection"] for e in refused.value.exits if e["decision"]]
            assert fills == (["censoring_aware", "half_minimum"] if purpose == "inference"
                             else ["censoring_aware", "qrilc", "half_minimum"])
    # Without the recoding the zeros are values, and nothing is below a limit.
    plain = ProjectState(target="y", purpose="inference", findings={
        k: v for k, v in findings.items() if "zeros" not in k})
    shares = omics.censored_shares_on(frame, censored, plain)
    assert max(shares.values()) == 0.0


# ── 4 · QRILC never leaves a blank for the median ────────────────────────────


def test_4a_a_sample_too_sparse_for_qrilc_gets_half_the_training_minimum_never_a_blank():
    """QRILC fits each sample's normal from at least five detected values. A sample with fewer
    keeps no blank: each of its values below detection becomes half the column's smallest detected
    value on the fitting rows (pandas over the training rows), the customary rung, and the step
    counts them. That row reads the training rows (observed: training-fold scope), a dense row
    only itself (row-local); the contract declares the wider."""
    rng = np.random.default_rng(11)
    n, p = 40, 12
    feats = [f"f{j}" for j in range(p)]
    values = np.exp(rng.normal(5, 1, (n, p)))
    values[values < np.exp(4.3)] = np.nan
    values[0, 3:] = np.nan  # a sample with three detected values
    frame = pd.DataFrame(values, columns=feats)
    train = frame.iloc[5:]
    step = omics.QRILCFill(feats, feats).fit(train)
    out = step.transform(frame)
    assert np.isfinite(out.to_numpy()).all()
    half = {c: float(train[c].min()) / 2 for c in feats}
    for c in feats[3:]:
        assert out.loc[0, c] == pytest.approx(half[c], rel=1e-12)
    assert step.fallback_ == int(np.isnan(values[0]).sum())
    reference = np.zeros(n, dtype=bool)
    y = rng.normal(size=n)

    def qrilc(f, r, yy):
        return omics.QRILCFill(feats, feats).fit(f).transform(f)

    assert observed_scope(qrilc, frame, reference, y, 0, blank=0.0) == "training_fold"
    dense = int(np.argmax(np.isfinite(values).sum(axis=1)))
    assert observed_scope(qrilc, frame, reference, y, dense) == "row_local"
    assert CONTRACTS["detection_limit"].scope_of("qrilc") == "training_fold"


# ── 5 · ComBat leaves a feature constant within a batch as sva does ──────────

CONSTANT_R = r"""
suppressPackageStartupMessages(library(sva))
d <- read.csv("data.csv", check.names = FALSE)
feats <- grep("^g", names(d), value = TRUE)
dat <- t(as.matrix(d[, feats]))
batch <- as.character(d$batch)
mod <- model.matrix(~ y, data = d)
w <- function(m, name) write.csv(t(m), paste0(name, ".csv"), row.names = FALSE)
sink(stderr())
out <- list(
  ref = ComBat(dat, batch, ref.batch = "10"),
  plain = ComBat(dat, batch),
  mod = ComBat(dat, batch, mod = mod),
  meanref = ComBat(dat, batch, mean.only = TRUE, ref.batch = "1")
)
sink()
for (name in names(out)) w(out[[name]], name)
"""


@needs_r
def test_5a_a_feature_constant_within_a_batch_is_left_unadjusted_as_sva_leaves_it(tmp_path):
    """The verifier's case: a metabolite never detected on one plate is constant there after a
    half-minimum fill and log. sva tests ``var(x) == 0``, exactly 0 in R for a constant vector;
    numpy's variance of the same vector is ~1e-30 for many lengths and values, so the port adjusted
    the feature (and let it into the empirical-Bayes priors) where sva leaves it. Four batches
    labeled 1, 2, 10 and 7 (sorted as text, as R's factor does), p = 300 ≫ n = 34, two features
    constant in batch 2 at values whose numpy variance there is not 0 (asserted, so the case is
    exercised). Against R ``sva::ComBat`` (3.60): a reference batch, none, covariates, mean-only:
    agreement to 1e-6 everywhere, the constant features unchanged in every batch."""
    rng = np.random.default_rng(21)
    sizes = {"1": 9, "2": 7, "10": 11, "7": 7}
    batch = np.concatenate([[b] * k for b, k in sizes.items()])
    n, p = len(batch), 300
    y = rng.integers(0, 2, n)
    X = rng.normal(8, 1, (n, p))
    for b in sizes:
        rows = batch == b
        X[rows] = X[rows] * np.exp(rng.normal(0, 0.3, p)) + rng.normal(0, 0.8, p)
    in2 = batch == "2"
    # log2 of half a detection limit: the first two whose numpy variance over 7 rows is not 0.
    misread = (math.log2(0.5 * k) for k in np.arange(500.0, 5000.0, 7.3)
               if np.var(np.full(int(in2.sum()), math.log2(0.5 * k)), ddof=1) != 0.0)
    constants = [next(misread), next(misread)]
    for j, value in enumerate(constants):
        X[in2, j] = value
        assert np.var(X[in2, j], ddof=1) != 0.0  # numpy's variance of the constant is not 0
    frame = pd.DataFrame(X, columns=[f"g{j:03d}" for j in range(p)])
    frame["batch"] = batch.astype(int)
    frame["y"] = y
    out = run_r(CONSTANT_R, {"data": frame}, tmp_path, ("ref", "plain", "mod", "meanref"))
    labels = frame["batch"].to_numpy()
    mod = np.column_stack([np.ones(n), y])
    mine = {"ref": B.combat(X, labels, ref_batch=10), "plain": B.combat(X, labels),
            "mod": B.combat(X, labels, mod=mod), "meanref": B.combat(X, labels, mean_only=True,
                                                                     ref_batch=1)}
    for name, values in mine.items():
        np.testing.assert_allclose(values, out[name].to_numpy(), rtol=0, atol=1e-6, err_msg=name)
        np.testing.assert_array_equal(values[:, :2], X[:, :2])
    fit = B.combat_fit(X, labels)
    assert not fit.keep[:2].any() and fit.keep[2:].all()


# ── 7 · the fill's sentence says what the run did ───────────────────────────


def test_7a_the_fill_sentence_follows_the_design_steps_and_the_tables_missing_record():
    """The chain tests read the paragraph the fit stage writes on real runs (prediction, and the
    inference single fill); the other records it reads are held here, verbatim, each from the
    record a run leaves: the design's detect step's columns (two of the three the answer named,
    the third removed by the QC filters), the design's step order, and the inference table's
    missing-data record. Multiple imputation draws the values within each copy (the record's m);
    half the minimum is set before the copies; a design with no normalization and no log says
    neither, and names columns, not features, outside the omics lenses."""
    from turbotab.core.decisions import MissingSpec

    named = ["mz_1", "mz_2", "mz_3"]
    design = {"method": "censoring_aware", "columns": ["mz_1", "mz_2"]}

    def said(purpose, method, steps, record=None, lens=("metabolomics",)):
        state = ProjectState(lens=list(lens), purpose=purpose, missing=MissingSpec(
            strategy="impute", below_detection=method, censored_columns=named, acknowledged=True))
        return omics.detection_details(state, method, censored={**design, "method": method},
                                       purpose=purpose, missing=record, steps=steps)

    assert said("prediction", "censoring_aware", ["normalize", "detect", "log", "scale", "model"]) == (
        "Values below detection in 2 features were filled after normalization and before the log, "
        "each by its expected value below the limit under a left-censored normal fitted to the "
        "feature's logarithm on the training fold (Lubin et al. 2004).")
    mi = {"method": "multiple_imputation", "m": 20}
    assert said("inference", "censoring_aware", ["normalize", "detect", "log", "model"], mi) == (
        "Values below detection in 2 features were drawn below the limit from a censored-normal "
        "(Tobit) model within each of the 20 multiple imputations, given the outcome (Lubin et al. "
        "2004).")
    assert said("inference", "half_minimum", ["normalize", "detect", "log", "model"], mi) == (
        "Values below detection in 2 features were set to half the feature's smallest detected "
        "value before the 20 multiple imputations (Lubin et al. 2004).")
    assert said("prediction", "censoring_aware", ["detect", "impute", "model"], lens=("clinical",)) == (
        "Values below detection in 2 columns were filled, each by its expected value below the limit "
        "under a left-censored normal fitted to the column or its logarithm, whichever fits better, "
        "on the training fold (Lubin et al. 2004).")
    # Complete cases leave the rows before the design: nothing is filled, and the sentence says so.
    left = ProjectState(lens=["metabolomics"], purpose="inference", missing=MissingSpec(
        strategy="complete_case", below_detection="censoring_aware", censored_columns=named))
    assert omics.detection_details(left, "censoring_aware", censored=design, purpose="inference",
                                   missing={"method": "complete_case"},
                                   steps=["normalize", "detect", "log", "model"]) == (
        "Rows with a value below detection in any of 2 features were left out (complete cases).")


# ── 6 · the assay block and the pack's QC findings ───────────────────────────


def test_6a_a_run_order_column_is_not_an_intensity():
    """The verifier's record said "81 columns" of a table with 80 metabolites and a ``run_order``
    column. Expected (the fixture): the 80 metabolite columns. A column named as an acquisition
    column (run order, batch, plate) or numbering the rows once each is never in the assay block;
    a metabolite named ``m_run`` with intensities is."""
    frame, _ = drifting_run(seed=2, p=80)
    table = frame.rename(columns={"injection_order": "run_order"}).assign(
        sample_no=np.arange(len(frame)))
    found = omics.scale_finding(table, ["metabolomics"], None)
    columns = found[0]["params"]["columns"]
    assert columns == FEATURES(frame) and found[1]["title"].startswith("`80` columns")
    assert "run_order" not in omics.candidate_columns(table) and "sample_no" not in omics.candidate_columns(table)


def test_6b_the_pack_says_no_pooled_qcs_only_when_none_were_found():
    """With the QC label ``pooled QC`` the naming census finds no QC row, while the variance reading
    finds all of them: "I couldn't find any pooled QC samples … can't compute … drift correction"
    is false beside the critical finding. Expected (the fixture): the 24 QC rows; the pooled-QC
    finding is raised and the no-QC finding is not. Without QC rows it is still said."""
    from turbotab.core.detectors import pack_findings

    frame, _ = drifting_run(seed=6, p=40)
    labeled = frame.assign(sample_type=frame["sample_type"].replace({"QC": "pooled QC"}))
    ids = {f["id"] for f in pack_findings(labeled, ["metabolomics"])}
    assert "pack::metabolomics::pooled_qc" in ids
    assert "pack::metabolomics::no_pooled_qc" not in ids
    study = labeled[labeled["sample_type"].eq("Sample")].reset_index(drop=True)
    ids = {f["id"] for f in pack_findings(study, ["metabolomics"])}
    assert "pack::metabolomics::pooled_qc" not in ids
    assert "pack::metabolomics::no_pooled_qc" in ids
