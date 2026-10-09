"""WP11 · Omics preprocessing and inference: the acceptance tests of AUDIT_REPORT.md §5 (closes
ME-09 and ME-18, and the minor F17).

§5 WP11, as written:

1. "On the depth-confounded null counts, a linear-family fit on raw values is refused until a
   transformation is chosen or the values are declared normalized; with log-CPM/TMM the CV AUC is
   0.50 ± 0.05 (today 0.83–1.0). A label-blind library-size check flags the depth–outcome
   association."
2. "On the dilution null, PQN + log2 gives AUC 0.50 ± 0.05 (today raw 0.59–0.71)."
3. "Under inference with p ≥ n, elastic net carries "no confidence intervals"; a feature-wise
   regression family with BH-FDR holds the false-discovery rate ≤ 0.05 on a null simulation."
4. ""Do not pre-normalize" is replaced by purpose-specific text. Source check: Hornung et al. 2015."

The fixtures are the audit's own generators (``docs/turbotab-next/audit/repro.tar.gz``):
``F-skeptic/f2_make.py`` (the depth-confounded table, ``default_rng(2024)``), ``F-skeptic/f2_sim.py``
(the depth null, negative-binomial genes, cases 1.46× deeper) and ``F/sim_dilution.py`` (urine 25%
more dilute in cases), replayed draw for draw. The projects are driven through the real server, or
through the app's own design and fit stages for the Monte Carlo replicates.

Every reference comes from a path independent of the code under test: rnanorm's TMM (a separate
Python implementation, "validated to be identical to [edgeR] to at least 10 decimal places"),
edgeR's log-CPM closed form read from its C source, Dieterle's probabilistic quotient
normalization written out from its published steps, statsmodels' least squares and
Benjamini–Hochberg, scipy's Mann–Whitney test, scikit-learn's ROC AUC, and analytic values (an AUC
of 0.50 when nothing differs; a false-discovery rate of at most 0.05 for Benjamini–Hochberg at
q = 0.05). R is not installed; where edgeR is the reference, the docstring says which closed form
or implementation stands in for it. Simulations are seeded; each states its Monte Carlo error.
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import FindingDisposition
from turbotab.core.graph import artifact_dir
from turbotab.core.methods import omics
from turbotab.core.methods.omics import LogCPM, QuotientLog
from turbotab.core.models import get_family, rank
from turbotab.core.models.base import Situation
from turbotab.core.models.featurewise import featurewise_table
from turbotab.core.models.inference import resolve_clusters  # noqa: F401 - the clustered path's home
from turbotab.core.stages.modeling import design_stage, fit_stage
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance.references import cr2_by_definition
from turbotab.server.tests.conftest import make_client, prepare, wait_for
from turbotab.core.tests.acceptance.server_drive import served

FDR = 0.05


# ── the audit's fixtures, replayed ───────────────────────────────────────────


def depth_table(folder: Path) -> Path:
    """F-skeptic/f2_make.py, draw for draw: 60 samples × 300 negative-binomial genes, no gene
    differing, cases sequenced about 1.4× deeper; ``case`` is 0/1."""
    rng = np.random.default_rng(2024)
    n, p = 60, 300
    case = np.r_[np.ones(30), np.zeros(30)].astype(int)
    rng.shuffle(case)
    sf = np.exp(rng.normal(0, 0.15, n)) * np.where(case == 1, 1.4, 1.0)
    base = np.exp(rng.normal(np.log(150), 1.2, p))
    disp = 0.1
    mu = np.outer(sf, base)
    r = 1 / disp
    X = rng.negative_binomial(r, r / (r + mu))
    frame = pd.DataFrame(X, columns=[f"ENSG{100000 + i:011d}" for i in range(p)])
    frame.insert(0, "sample_id", [f"S{i:03d}" for i in range(n)])
    frame["case"] = case
    path = folder / "depth_counts.csv"
    frame.to_csv(path, index=False)
    return path


def depth_null(rng: np.random.Generator, ratio: float = 1.46) -> pd.DataFrame:
    """F-skeptic/f2_sim.py's ``counts``: 60 samples × 300 genes, NB(r = 10), no gene differing,
    cases sequenced ``ratio``× deeper."""
    n, p = 60, 300
    case = np.r_[np.ones(30), np.zeros(30)].astype(int)
    sf = np.exp(rng.normal(0, 0.15, n)) * np.where(case == 1, ratio, 1.0)
    base = np.exp(rng.normal(np.log(60), 1.2, p))
    r = 10
    mu = np.outer(sf, base)
    X = rng.negative_binomial(r, r / (r + mu))
    frame = pd.DataFrame(X, columns=[f"ENSG{100000 + i:011d}" for i in range(p)])
    frame.insert(0, "sample_id", [f"S{i:03d}" for i in range(n)])
    frame["case"] = case
    return frame


def dilution_null(rng: np.random.Generator) -> pd.DataFrame:
    """F/sim_dilution.py: 80 urine samples × 300 log-normal metabolites, none differing, cases'
    samples 25% more dilute."""
    n, p = 80, 300
    y = np.repeat([0, 1], n // 2)
    conc = np.exp(rng.normal(rng.normal(8, 2, p), 0.6, (n, p)))
    dil = np.exp(rng.normal(0, 0.4, n)) * np.where(y == 1, 0.75, 1.0)
    frame = pd.DataFrame(conc * dil[:, None], columns=[f"mz_{i:04d}" for i in range(p)])
    frame.insert(0, "sample_id", [f"U{i:03d}" for i in range(n)])
    frame["case"] = y
    return frame


# ── independent references ───────────────────────────────────────────────────


def edger_log_cpm(counts: np.ndarray, lib: np.ndarray, ave_lib: float, prior_count: float = 2.0):
    """edgeR's ``cpm(y, lib.size = lib, log = TRUE, prior.count = 2)``, written from its C source
    (edgeR 4.10.5, ``src/add_prior_count.c``, ``compute_offsets``)::

        /* compute the adjusted prior count for each library */
        for(int lib=0;lib<nlib;++lib) prior[lib]=pptr[lib]*offset[lib]/ave_lib;
        /* add it twice back to the library sizes */
        for(int lib=0;lib<nlib;++lib) offset[lib]+=2*prior[lib];

    and ``src/compute_cpm.c`` (``calc_cpm_log``): ``(log(y + prior) − log(offset) + log(1e6)) /
    log(2)``, where ``ave_lib`` is the mean library size (``ave_lib=ave_lib/nlib``). The library
    size is ``lib.size × norm.factors`` (``cpm.DGEList``: ``lib.size <- lib.size*norm.factors``).
    Out of sample, ``ave_lib`` is the training rows' mean, the only one a held-out row may see.
    """
    prior = prior_count * lib / ave_lib
    return (np.log(counts + prior[:, None]) - np.log(lib + 2 * prior)[:, None] + np.log(1e6)) / np.log(2)


def dieterle_pqn(train: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Probabilistic quotient normalization by its published steps, in Kohl et al.'s words
    (Metabolomics 2012, 8:146, describing Dieterle et al., Anal Chem 2006, 78:4281): "PQN starts,
    with an integral normalization of each spectrum, followed by the calculation of a reference
    spectrum such as a median spectrum. Next, for each variable of interest the quotient of a given
    test spectrum and reference spectrum is calculated and the median of all quotients is
    estimated. Finally, all variables of the test spectrum are divided by the median quotient."

    The integral normalization scales each training spectrum to a common total (here the training
    rows' median total; Dieterle used 100, which multiplies every result by one constant), and the
    reference is learned from the training rows alone. Written with pandas, not the app's code.
    """
    t = pd.DataFrame(train)
    totals = t.sum(axis=1)
    constant = float(totals.median())
    reference = t.mul(constant / totals, axis=0).median(axis=0)
    r = pd.DataFrame(rows)
    test_totals = r.sum(axis=1)
    integral = r.mul(constant / test_totals, axis=0)
    quotient = integral.div(reference, axis=1).median(axis=1)
    return integral.div(quotient, axis=0).to_numpy()


def mc_error(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / math.sqrt(len(values)))


# ── driving the server ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with make_client(tmp_path_factory.mktemp("wp11_home"), "local", 2, "http://127.0.0.1") as c:
        yield c


@pytest.fixture(scope="module")
def depth_csv(tmp_path_factory) -> Path:
    return depth_table(tmp_path_factory.mktemp("wp11_tables"))


def post(client, pid: str, decision: dict) -> tuple[int, dict]:
    response = client.post(f"/api/projects/{pid}/decisions", json=decision)
    return response.status_code, response.json()


def accepted(client, pid: str, decision: dict) -> dict:
    code, body = post(client, pid, decision)
    assert code == 200, body
    return body


def open_project(client, path: Path, lens: str, target: str, purpose: str) -> str:
    from turbotab.server.tests.conftest import declare

    response = client.post("/api/projects", json={"path": str(path)})
    assert response.status_code == 200, response.text
    pid = response.json()["id"]
    # WP17: an omics table's question under inference is every feature in turn, with its
    # false-discovery statement (MODELING_SEQUENCE §1 step 2: an exposure family).
    declare(pid, {f"exposure:{target}": "family"}, fixture=path.name)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"}, timeout=120)
    accepted(client, pid, {"kind": "set_lens", "lenses": [lens]})
    # The readings ledger (BLUEPRINT §14.1): under an assay lens the orientation reading is never
    # high, so the question is asked; these tables are one row per sample, as the reading proposes.
    prepare(client, pid, {"kind": "set_target", "column": target})
    accepted(client, pid, {"kind": "set_target", "column": target})
    prepare(client, pid, {"kind": "set_purpose", "purpose": purpose})
    accepted(client, pid, {"kind": "set_purpose", "purpose": purpose})
    wait_for(client, pid, {"findings": "fresh"}, timeout=120)
    return pid


def findings(client, pid: str) -> list[dict]:
    return client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]["findings"]


def artifact(client, pid: str, stage: str) -> dict:
    return served(client, pid, stage)


def training_rows(client, pid: str, *, held_out_too: bool = False) -> np.ndarray:
    """The rows the split trains on, read from its parquet on disk; with ``held_out_too``, every
    analyzed row (the held-out ones included)."""
    view = wait_for(client, pid, {"split": "fresh"}, timeout=120)
    cache = client.app.state.service.workspace.cache_dir(pid)
    frame = pd.read_parquet(artifact_dir(cache, "split", view["stages"]["split"]["key"])
                            / "frames" / "assignment.parquet")
    if held_out_too:
        return np.sort(frame["row_id"].to_numpy())
    return np.sort(frame.loc[frame["partition"] == "train", "row_id"].to_numpy())


def shelf(client, pid: str) -> dict[str, dict]:
    wait_for(client, pid, {"shelf": "fresh"}, timeout=240)
    return {f["key"]: f for f in artifact(client, pid, "shelf")["families"]}


# ── 1 · the depth-confounded null counts ─────────────────────────────────────


def test_1a_tmm_and_log_cpm_match_edger_fit_on_training_rows_only(depth_csv):
    """The normalization step against two independent paths, on the audit's table.

    * TMM factors: rnanorm 2.2.0's ``TMM`` ("validated to be identical to [edgeR] to at least 10
      decimal places"; Robinson & Oshlack 2010), fit on the training rows. On the training rows
      the factors are edgeR's ``calcNormFactors(method = "TMM")``; a held-out row's factor is taken
      against the same reference sample and the same geometric-mean scaling, which is rnanorm's
      ``fit``/``get_norm_factors`` on new rows.
    * log-CPM: edgeR's ``cpm(log = TRUE, prior.count = 2)`` closed form (:func:`edger_log_cpm`),
      on library sizes × TMM factors, with the training rows' mean library size.

    A gene that is zero in every training row is left out of the TMM comparison (edgeR drops
    all-zero genes), and a missing count stays missing.
    """
    from rnanorm import TMM

    frame = pd.read_csv(depth_csv)
    genes = [c for c in frame.columns if c.startswith("ENSG")]
    counts = frame[genes].to_numpy(dtype=float)
    counts[:, 0] = 0.0  # a gene absent everywhere
    counts[:3, 1] = 0.0  # and one with a few zeros
    data = pd.DataFrame(counts, columns=genes)
    train, test = np.arange(48), np.arange(48, 60)

    step = LogCPM(genes, tmm=True).fit(data.iloc[train])
    reference = TMM().fit(counts[train])
    np.testing.assert_allclose(step.factors(counts[train]), reference.get_norm_factors(counts[train]),
                               rtol=1e-12)
    np.testing.assert_allclose(step.factors(counts[test]), reference.get_norm_factors(counts[test]),
                               rtol=1e-12)

    lib = counts.sum(axis=1) * reference.get_norm_factors(counts)
    ave = float(lib[train].mean())
    out = step.transform(data).to_numpy()
    np.testing.assert_allclose(out, edger_log_cpm(counts, lib, ave), rtol=1e-12, atol=1e-12)

    # Without TMM: library size alone, the same closed form.
    plain = LogCPM(genes, tmm=False).fit(data.iloc[train]).transform(data).to_numpy()
    raw_lib = counts.sum(axis=1)
    np.testing.assert_allclose(plain, edger_log_cpm(counts, raw_lib, float(raw_lib[train].mean())),
                               rtol=1e-12, atol=1e-12)

    # A missing count stays missing and adds nothing to its row's library size.
    holed = data.copy()
    holed.iloc[50, 5] = np.nan
    again = step.transform(holed).to_numpy()
    assert np.isnan(again[50, 5])
    filled = counts.copy()
    filled[50, 5] = 0.0
    # rnanorm drops the genes that are zero in every row it is given, so it is given every row.
    lib50 = filled[50].sum() * float(reference.get_norm_factors(filled)[50])
    filled = filled[50]
    expect = edger_log_cpm(filled[None, :], np.array([lib50]), ave)[0]
    np.testing.assert_allclose(np.delete(again[50], 5), np.delete(expect, 5), rtol=1e-12)


def test_1b_raw_counts_are_refused_until_normalized_and_the_library_size_check_flags_depth(client, depth_csv):
    """The audit's table through the real server, under prediction.

    Reference for refusal: AUDIT_REPORT ME-09, "refuse a linear-family fit on raw values until the
    user picks a transformation or records that the values are already normalized". A selection
    with a linear family is answered 409 ``raw_assay_values``; its exits are the three answers
    (log-CPM with TMM, log-CPM, already normalized), and keeping the families that are not linear.

    Reference for the library-size check: the library sizes summed with pandas from the CSV, on the
    training rows read from the split's parquet; scipy's Mann–Whitney U test and scikit-learn's
    ROC AUC of library size against ``case``. The audit measured an AUC of 0.987 for library size
    alone; the check must flag it (p < 0.01) on every family of the shelf, and fall silent once
    the counts are normalized. The reading that raises the finding never sees ``case``.

    Reference for the fit: the analytic 0.50 of a null, ± 0.05. On this table the elastic net on
    log-CPM/TMM scores the class prior in every fold.
    """
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score

    pid = open_project(client, depth_csv, "genomics", "case", "prediction")
    found = {f["id"]: f for f in findings(client, pid)}
    scale = found["omics_scale"]
    assert [o["key"] for o in scale["repairs"]] == ["log_cpm_tmm", "log_cpm", "declared_normalized"]
    assert "raw counts" in scale["title"] and len(scale["affected_columns"]) == 300
    assert "case" not in scale["affected_columns"] and "sample_id" not in scale["affected_columns"]

    select = {"kind": "select_models", "models": ["elastic_net", "boosted_trees"]}
    prepare(client, pid, select)
    families = shelf(client, pid)

    frame = pd.read_csv(depth_csv)
    genes = [c for c in frame.columns if c.startswith("ENSG")]
    train = training_rows(client, pid)
    library = frame[genes].sum(axis=1).to_numpy(dtype=float)[train]
    y = frame["case"].to_numpy()[train]
    test = mannwhitneyu(library[y == 1], library[y == 0], alternative="two-sided")
    auc = roc_auc_score(y, library)
    assert test.pvalue < omics.CHECK_ALPHA and auc > 0.9  # the scenario discriminates
    for family in families.values():
        said = family["concerns"][0]
        assert said.startswith("Library size tracks `case` on the training rows"), family
        assert f"AUC {auc:.2f}" in said, (said, auc)

    code, body = post(client, pid, select)
    assert code == 409, body
    error = body["error"]
    assert error["code"] == "raw_assay_values" and "raw counts" in error["message"]
    exits = [e["label"] for e in error["exits"]]
    assert exits[:3] == ["Log-CPM with TMM", "Log-CPM, library size only", "Already normalized"]
    assert "Keep only the families that are not linear" in exits
    assert error["exits"][3]["decision"]["models"] == ["boosted_trees"]
    for linear in (["linear"], ["featurewise"]):
        code, body = post(client, pid, {"kind": "select_models", "models": linear})
        assert code == 409 and body["error"]["code"] == "raw_assay_values", body

    accepted(client, pid, error["exits"][0]["decision"])  # log-CPM with TMM
    prepare(client, pid, {"kind": "select_models", "models": ["elastic_net"]})
    after = shelf(client, pid)
    assert not any("Library size tracks" in c for f in after.values() for c in f["concerns"])
    accepted(client, pid, {"kind": "select_models", "models": ["elastic_net"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=300)
    fit = artifact(client, pid, "fit")
    measured = fit["models"][0]["cv"]["auc"]["estimate"]
    assert abs(measured - 0.50) <= 0.05, measured
    steps = artifact(client, pid, "design")["models"][0]["steps"]
    assert steps[0]["key"] == "normalize" and steps[0]["label"] == "Log-CPM with TMM factors"


def test_1c_declared_normalized_lets_the_values_in_as_they_are(client, depth_csv):
    """The other way past the refusal: the researcher states the values were normalized. The
    selection is accepted, the values enter unchanged (no normalize step), and the library-size
    check keeps saying what the raw totals show, because the statement does not change them."""
    pid = open_project(client, depth_csv, "genomics", "case", "prediction")
    scale = next(f for f in findings(client, pid) if f["id"] == "omics_scale")
    declared = next(o for o in scale["repairs"] if o["key"] == "declared_normalized")
    select = {"kind": "select_models", "models": ["elastic_net"]}
    prepare(client, pid, select)
    assert post(client, pid, select)[0] == 409
    accepted(client, pid, declared["decision"])
    prepare(client, pid, select)  # the repair re-reads the table: earlier answers are re-checked
    accepted(client, pid, select)
    assert shelf(client, pid)["elastic_net"]["concerns"][0].startswith("Library size tracks `case`")
    wait_for(client, pid, {"design": "fresh"}, timeout=240)
    steps = artifact(client, pid, "design")["models"][0]["steps"]
    assert "normalize" not in [s["key"] for s in steps]


def _stage_auc(frame: pd.DataFrame, lens: str, kind: str, option: str, folder: Path) -> float:
    """One replicate through the app's own design and fit stages (prediction, 5-fold CV, no
    holdout), the normalization recorded as the ``omics_scale`` answer ``option``."""
    features = [c for c in frame.columns if c not in ("sample_id", "case")]
    paths = mf.ingest_frame(frame, folder)
    st = mf.state(lens=[lens], target="case", event="1", purpose="prediction", models=["elastic_net"],
                  roles={**{c: "exposure" for c in features}, "sample_id": "identifier"},
                  findings={"omics_scale": FindingDisposition(action="applied", option=option,
                                                              params={"kind": kind, "columns": features})})
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, folds=5, seed=0)
    ti = mf.target_info("binary", "case")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    return float(fit.data["models"][0]["cv"]["auc"]["estimate"])


def _headline_only(monkeypatch, frame: pd.DataFrame, lens: str, kind: str, option: str,
                   folder: Path) -> None:
    """These replicate loops read only each fit's headline CV AUC: the split's own 5-fold run. MS6
    also fits the comparisons' 10 × 5-fold substrate under prediction, whose first repeat is that
    same run, so the headline does not depend on it. This checks it on one replicate (the full fit
    stage against the fit stage with the substrate stood down to the headline's folds, to 10⁻¹²),
    then stands the substrate down for the loop, which would otherwise take ten times as long."""
    from turbotab.core.models import folds

    full = _stage_auc(frame, lens, kind, option, folder / "full")
    monkeypatch.setattr(folds, "comparison_folds",
                        lambda headline, **kw: ([np.asarray(c) for c in headline], len(headline), None))
    assert _stage_auc(frame, lens, kind, option, folder / "headline") == pytest.approx(full, abs=1e-12)


def test_1d_log_cpm_tmm_brings_the_depth_null_to_chance(tmp_path, monkeypatch):
    """Twenty replicates of the audit's depth null (F-skeptic/f2_sim.py: no gene differs, cases
    1.46× deeper), each through the app's design and fit stages with the elastic net.

    Reference: an AUC of 0.50 (nothing differs), ± 0.05. Measured on log-CPM/TMM: the mean CV AUC
    over the replicates (0.535 here), Monte Carlo error 0.017 (one replicate's CV AUC has an SD of
    about 0.06; 80 replicates drawn apart from this test gave 0.502 ± 0.007, so log-CPM's small
    low-count depth bias does not reach the model). The bound discriminates: the same replicates
    with the raw counts declared normalized score a mean CV AUC of 0.955 ± 0.007, asserted above
    0.85 (the audit reported 0.83–1.0). About 100 seconds.
    """
    _headline_only(monkeypatch, depth_null(np.random.default_rng(1)), "genomics", "counts",
                   "log_cpm_tmm", tmp_path / "check")
    rng = np.random.default_rng(2015)
    raw, normalized = [], []
    for rep in range(20):
        frame = depth_null(rng)
        raw.append(_stage_auc(frame, "genomics", "counts", "declared_normalized", tmp_path / f"r{rep}"))
        normalized.append(_stage_auc(frame, "genomics", "counts", "log_cpm_tmm", tmp_path / f"n{rep}"))
    error = mc_error(normalized)
    assert error < 0.02, error  # precise enough for the ± 0.05 bound to mean something
    assert abs(np.mean(normalized) - 0.50) <= 0.05, (np.mean(normalized), error)
    assert np.mean(raw) > 0.85, np.mean(raw)


# ── 2 · the dilution null ────────────────────────────────────────────────────


def test_2a_pqn_matches_dieterle_fit_on_training_rows_only():
    """Probabilistic quotient normalization against its published steps (:func:`dieterle_pqn`),
    the reference spectrum learned from the training rows and applied to every row; then log2,
    where a zero cannot be logged and becomes missing."""
    rng = np.random.default_rng(9)
    frame = dilution_null(rng)
    features = [c for c in frame.columns if c.startswith("mz_")]
    data = frame[features]
    train = np.arange(64)
    step = QuotientLog(features, quotient=True, log=True).fit(data.iloc[train])
    out = step.transform(data).to_numpy()
    values = data.to_numpy()
    np.testing.assert_allclose(out, np.log2(dieterle_pqn(values[train], values)), rtol=1e-10)
    only_log = QuotientLog(features, quotient=False, log=True).fit(data.iloc[train])
    np.testing.assert_allclose(only_log.transform(data).to_numpy(), np.log2(values), rtol=1e-12)
    zeroed = data.copy()
    zeroed.iloc[70, 3] = 0.0
    assert np.isnan(step.transform(zeroed).to_numpy()[70, 3])


def test_2b_pqn_log2_brings_the_dilution_null_to_chance(tmp_path, monkeypatch):
    """Sixteen replicates of the audit's dilution null (F/sim_dilution.py, ``default_rng(9)``:
    no metabolite differs, cases' urine 25% more dilute), each through the app's design and fit
    stages with the elastic net.

    Reference: an AUC of 0.50, ± 0.05. Measured on PQN + log2: the mean CV AUC, 0.500 ± 0.011
    (Monte Carlo error). The bound discriminates: the same replicates on the raw intensities
    (declared normalized) score 0.619 ± 0.020, asserted above 0.55 (the audit reported
    0.59–0.71). About 100 seconds.
    """
    _headline_only(monkeypatch, dilution_null(np.random.default_rng(1)), "metabolomics",
                   "intensities", "pqn_log2", tmp_path / "check")
    rng = np.random.default_rng(9)
    raw, normalized = [], []
    for rep in range(16):
        frame = dilution_null(rng)
        raw.append(_stage_auc(frame, "metabolomics", "intensities", "declared_normalized",
                              tmp_path / f"r{rep}"))
        normalized.append(_stage_auc(frame, "metabolomics", "intensities", "pqn_log2",
                                     tmp_path / f"n{rep}"))
    error = mc_error(normalized)
    assert error < 0.02, error
    assert abs(np.mean(normalized) - 0.50) <= 0.05, (np.mean(normalized), error)
    assert np.mean(raw) > 0.55, np.mean(raw)


def test_2c_raw_intensities_raise_the_finding_and_name_autoscaling(tmp_path):
    """The metabolomics reading of the dilution table, label-free: ``omics_scale`` reads the 300
    columns as raw intensities and offers PQN + log2, log2 only, or already normalized. F17: the
    standardize step under the metabolomics lens names its departure from Pareto scaling, with van
    den Berg et al. (BMC Genomics 2006, 7:142): "For the explorative analysis of the validation data
    set used in this study, autoscaling and range scaling performed better than the other
    pretreatment methods.\""""
    from turbotab.core.models.pipeline import DesignSpec, describe_steps

    frame = dilution_null(np.random.default_rng(9))
    raw, finding = omics.scale_finding(frame, ["metabolomics"], target="case")
    assert raw["params"]["kind"] == "intensities" and len(finding["affected_columns"]) == 300
    unlabeled = omics.scale_finding(frame.drop(columns="case"), ["metabolomics"])
    assert unlabeled[1]["affected_columns"] == finding["affected_columns"]  # never reads the outcome

    features = finding["affected_columns"]
    spec = DesignSpec(predictors=features, inputs=features, categorical=[], numeric=features,
                      energy=None, impute=False, roles={c: "exposure" for c in features},
                      normalization={"method": "pqn_log2", "kind": "intensities", "columns": features},
                      lenses=["metabolomics"])
    steps = describe_steps(spec, get_family("elastic_net"), "binary", "prediction")
    assert [s["key"] for s in steps][:1] == ["normalize"]
    scale = next(s for s in steps if s["key"] == "scale")
    assert "Pareto scaling is customary in metabolomics" in scale["detail"]
    assert "van den Berg et al. (2006)" in scale["detail"]


# ── 3 · inference at p ≥ n ───────────────────────────────────────────────────


def test_3a_elastic_net_carries_no_confidence_intervals_under_inference_at_p_ge_n():
    """The audit's case (ME-18): n = 60, p = 497 under inference, where elastic net said "good"
    with no word about intervals. Reference: its own caveat at n = 600, p = 50, "Penalized
    coefficients are shrunk and carry no confidence intervals." It now holds at p ≥ n too, and the
    shelf, ordered by soundness for the purpose, leads with the feature-wise tests under inference
    and puts them last under prediction."""
    caveat = "Penalized coefficients are shrunk and carry no confidence intervals."
    for lenses in (("genomics",), ("metabolomics",), ()):
        inference = rank(Situation(task="binary", purpose="inference", n_rows=60, n_features=497,
                                   n_events=30, lenses=lenses))
        keys = [f.key for f, _ in inference]
        assert keys[0] == "featurewise", keys
        net = dict((f.key, a) for f, a in inference)["elastic_net"]
        assert caveat in net.concerns and net.fit != "good"
        prediction = rank(Situation(task="binary", purpose="prediction", n_rows=60, n_features=497,
                                    n_events=30, lenses=lenses))
        assert [f.key for f, _ in prediction][0] == "elastic_net"
        assert [f.key for f, _ in prediction][-1] == "featurewise"
    metabolomics = dict((f.key, a) for f, a in rank(Situation(
        task="binary", purpose="inference", n_rows=60, n_features=497, lenses=("metabolomics",))))
    assert any(c.startswith("PLS-DA is not offered") for c in metabolomics["featurewise"].concerns)


def test_3b_feature_wise_rows_match_statsmodels_and_benjamini_hochberg():
    """Each exposure's test against statsmodels' ordinary least squares, fit one exposure at a time
    with the covariates (classical standard errors, t reference), and the q-values against
    statsmodels' ``multipletests(method="fdr_bh")``. Both directions: a numeric outcome
    (``outcome ~ x_j + Z``) and a two-level one (``x_j ~ event + Z``, the limma design). With
    repeated rows, CR2 with Bell–McCaffrey df against the written-out definition
    (``references.cr2_by_definition``)."""
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    from turbotab.core.models.inference import Clusters

    rng = np.random.default_rng(11)
    n, m = 50, 80
    features = [f"g{j}" for j in range(m)]
    matrix = pd.DataFrame(rng.lognormal(0, 1, (n, m)), columns=features)
    matrix["age"] = rng.normal(50, 10, n)
    matrix["sex_male"] = (rng.random(n) < 0.5).astype(float)
    Z = sm.add_constant(matrix[["age", "sex_male"]].to_numpy())
    outcome = 0.3 * matrix["g0"].to_numpy() + 0.02 * matrix["age"].to_numpy() + rng.normal(0, 1, n)
    event = np.r_[np.ones(n // 2), np.zeros(n - n // 2)]
    matrix.loc[event == 1, "g1"] += 1.0
    for task, y in (("regression", outcome), ("binary", event)):
        table = featurewise_table(matrix, y, task, features)
        rows = {r["feature"]: r for r in table.rows}
        assert list(rows) == features
        p_ref = []
        for j, f in enumerate(features):
            x = matrix[f].to_numpy()
            if task == "regression":
                res = sm.OLS(y, np.column_stack([Z, x])).fit()
            else:
                res = sm.OLS(x, np.column_stack([Z, y])).fit()
            lo, hi = res.conf_int(0.05)[-1]
            r = rows[f]
            assert r["estimate"] == pytest.approx(res.params[-1], rel=1e-9, abs=1e-12)
            assert r["se"] == pytest.approx(res.bse[-1], rel=1e-9)
            assert r["df"] == res.df_resid
            assert r["p"] == pytest.approx(res.pvalues[-1], rel=1e-7, abs=1e-15)
            assert (r["ci_low"], r["ci_high"]) == (pytest.approx(lo, rel=1e-9), pytest.approx(hi, rel=1e-9))
            p_ref.append(res.pvalues[-1])
        q_ref = multipletests(p_ref, alpha=FDR, method="fdr_bh")[1]
        np.testing.assert_allclose([rows[f]["q"] for f in features], q_ref, rtol=1e-7)
        assert table.info["covariance"] == "model"

    codes = np.repeat(np.arange(n // 2), 2)
    clustered = Clusters(column="person", codes=codes, n_clusters=n // 2)
    small = matrix[[*features[:5], "age", "sex_male"]]
    table = featurewise_table(small, outcome, "regression", features[:5], clustered)
    for f in features[:5]:
        X = np.column_stack([Z, small[f].to_numpy()])
        beta = np.linalg.lstsq(X, outcome, rcond=None)[0]
        V, df = cr2_by_definition(X, outcome - X @ beta, codes)
        r = next(r for r in table.rows if r["feature"] == f)
        assert r["se"] == pytest.approx(math.sqrt(V[-1, -1]), rel=1e-8)
        assert r["df"] == pytest.approx(df[-1], rel=1e-8)
    assert table.info["covariance"] == "CR2"


def _hc3_p(response: np.ndarray, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """HC3 p-values (MacKinnon & White 1985) of each column of X in ``response ~ x_j + Z``, by the
    definition ``(XᵀX)⁻¹ Xᵀ diag(e²/(1 − h)²) X (XᵀX)⁻¹`` on t(n − k): the linear family's
    covariance, used here only to show the bound discriminates."""
    from scipy import stats

    n = len(response)
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        D = np.column_stack([Z, X[:, j]])
        M = np.linalg.inv(D.T @ D)
        beta = M @ D.T @ response
        e = response - D @ beta
        h = np.einsum("ij,jk,ik->i", D, M, D)
        V = M @ (D.T * (e / (1 - h)) ** 2) @ D @ M
        out[j] = 2 * stats.t.sf(abs(beta[-1]) / math.sqrt(V[-1, -1]), n - D.shape[1])
    return out


@pytest.mark.parametrize("task,exposures", [("regression", "normal"), ("regression", "lognormal"),
                                            ("binary", "normal")])
def test_3c_feature_wise_benjamini_hochberg_holds_the_false_discovery_rate_on_a_null(task, exposures):
    """The null simulation: n = 40 samples, 500 exposures (p ≥ n), one covariate, no exposure
    related to the outcome; 2,000 replicates. Under a complete null every discovery is false, so
    the false-discovery rate is the chance of any discovery, and Benjamini–Hochberg at q = 0.05
    holds it at 0.05 (Benjamini & Hochberg 1995, Theorem 1: FDR ≤ q·m₀/m).

    Reference: 0.05, up to Monte Carlo error. Assertion: the estimate is not above 0.05 by more
    than 2.58 Monte Carlo standard errors (about 0.0049 each). The bound discriminates twice: the
    unadjusted p < 0.05 rule makes a false discovery in nearly every replicate, and the linear
    family's HC3 standard errors (measured on 200 of the replicates, the regression direction)
    make the false-discovery rate 0.11 on normal exposures and above 0.5 on log-normal ones,
    which is why the feature-wise family uses classical least-squares errors (featurewise.py).
    """
    rng = np.random.default_rng(1995 if task == "regression" else 2004)
    n, m, R = 40, 500, 2000
    names = [f"f{j}" for j in range(m)]
    any_bh, any_raw, any_hc3 = [], [], []
    for rep in range(R):
        age = rng.normal(50, 10, n)
        X = rng.normal(0, 1, (n, m))
        if exposures == "lognormal":
            X = np.exp(X)
        if task == "binary":
            y = rng.permutation(np.r_[np.ones(n // 2), np.zeros(n // 2)])
        else:
            y = 0.02 * age + rng.normal(0, 1, n)
        matrix = pd.DataFrame(X, columns=names)
        matrix["age"] = age
        table = featurewise_table(matrix, y, task, names)
        q = np.array([r["q"] for r in table.rows], dtype=float)
        p = np.array([r["p"] for r in table.rows], dtype=float)
        any_bh.append(bool((q < FDR).any()))
        any_raw.append(bool((p < 0.05).any()))
        if task == "regression" and rep < 200:
            from statsmodels.stats.multitest import multipletests

            hc3 = _hc3_p(y, X, np.column_stack([np.ones(n), age]))
            any_hc3.append(bool(multipletests(hc3, alpha=FDR, method="fdr_bh")[0].any()))
    fdr = float(np.mean(any_bh))
    error = math.sqrt(FDR * (1 - FDR) / R)
    assert fdr <= FDR + 2.58 * error, (fdr, error)
    assert np.mean(any_raw) > 0.99
    if any_hc3:
        assert np.mean(any_hc3) > FDR + 2.58 * math.sqrt(FDR * (1 - FDR) / len(any_hc3)), np.mean(any_hc3)


def test_3d_with_true_signals_the_false_share_of_discoveries_stays_under_q():
    """A mixed simulation: 500 exposures of which 25 truly differ (1.5 SD) between 20 cases and 20
    controls; 1,000 replicates. Reference: Benjamini & Hochberg's bound q·m₀/m = 0.0475 for the
    expected share of false discoveries among discoveries. Measured: that share's mean, Monte
    Carlo error about 0.0013, and the bound is held with room; the unadjusted rule's share is near
    one half."""
    rng = np.random.default_rng(1300)
    n, m, k, R = 40, 500, 25, 1000
    names = [f"f{j}" for j in range(m)]
    false_share, raw_share, found = [], [], []
    for _ in range(R):
        y = rng.permutation(np.r_[np.ones(n // 2), np.zeros(n // 2)])
        X = rng.normal(0, 1, (n, m))
        X[:, :k] += 1.5 * y[:, None]
        matrix = pd.DataFrame(X, columns=names)
        matrix["age"] = rng.normal(50, 10, n)
        table = featurewise_table(matrix, y, "binary", names)
        q = np.array([r["q"] for r in table.rows], dtype=float)
        p = np.array([r["p"] for r in table.rows], dtype=float)
        hit, raw = q < FDR, p < 0.05
        false_share.append(hit[k:].sum() / max(hit.sum(), 1))
        raw_share.append(raw[k:].sum() / max(raw.sum(), 1))
        found.append(hit[:k].sum())
    error = mc_error(false_share)
    assert np.mean(false_share) <= FDR * (m - k) / m + 2.58 * error, (np.mean(false_share), error)
    assert np.mean(raw_share) > 0.3
    assert np.mean(found) > 15  # the family still finds most of the real differences


def test_3e_on_raw_counts_every_gene_differs_and_log_cpm_tmm_removes_it(depth_csv):
    """Why inference waits for a normalization too: on the audit's depth-confounded table, where no
    gene differs, the feature-wise tests on the raw counts (the limma design, cases against
    controls) call most genes different at q < 0.05, because every count is larger in a deeper
    library; on log-CPM/TMM they call none. Reference: zero true differences (the generator draws
    every gene from the same distribution in both groups)."""
    frame = pd.read_csv(depth_csv)
    genes = [c for c in frame.columns if c.startswith("ENSG")]
    y = frame["case"].to_numpy()
    raw = featurewise_table(frame[genes].astype(float), y, "binary", genes)
    normalized = LogCPM(genes, tmm=True).fit(frame[genes]).transform(frame[genes].astype(float))
    clean = featurewise_table(normalized, y, "binary", genes)
    raw_found = sum(1 for r in raw.rows if r["q"] is not None and r["q"] < FDR)
    clean_found = sum(1 for r in clean.rows if r["q"] is not None and r["q"] < FDR)
    assert raw_found > 150, raw_found
    assert clean_found == 0, clean_found


def test_3f_feature_wise_tests_run_end_to_end_under_inference(client, depth_csv):
    """Through the server under inference: the shelf leads with the feature-wise family and
    elastic net says it has no intervals; after log-CPM/TMM, the fit reports one row per gene with
    an interval, a p-value and a q-value, no cross-validated score, and no discovery on this null
    table. Reference for the rows: :func:`featurewise_table`'s statsmodels agreement (3b), here
    re-derived for one gene with statsmodels on every analyzed row, the held-out ones included,
    with the normalization fit on them too: under inference the table is estimated from every
    analyzed row (BLUEPRINT §12 ruling 3; WP8, merged after this test was written against the
    training rows)."""
    import statsmodels.api as sm

    pid = open_project(client, depth_csv, "genomics", "case", "inference")
    scale = next(f for f in findings(client, pid) if f["id"] == "omics_scale")
    tmm = next(o for o in scale["repairs"] if o["key"] == "log_cpm_tmm")
    prepare(client, pid, {"kind": "select_models", "models": ["featurewise"]})
    accepted(client, pid, tmm["decision"])
    prepare(client, pid, {"kind": "select_models", "models": ["featurewise"]})
    families = shelf(client, pid)
    assert list(families)[0] == "featurewise"
    assert "Penalized coefficients are shrunk and carry no confidence intervals." in \
        families["elastic_net"]["concerns"]
    accepted(client, pid, {"kind": "select_models", "models": ["featurewise"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=300)
    model = artifact(client, pid, "fit")["models"][0]
    assert model["family"] == "featurewise" and model["cv"] == {} and model["holdout"] is None
    rows = model["coefficients"]
    assert len(rows) == 300 and all(r["q"] is not None and r["q"] >= FDR for r in rows)
    assert model["inference"]["covariance"] == "model"
    assert "Benjamini–Hochberg q < 0.05: 0 of 300" in model["inference"]["caption"]

    frame = pd.read_csv(depth_csv)
    genes = [c for c in frame.columns if c.startswith("ENSG")]
    analyzed = training_rows(client, pid, held_out_too=True)
    assert len(analyzed) > len(training_rows(client, pid))  # a holdout was drawn
    assert model["coefficients_n"] == len(analyzed)
    counts = frame[genes]
    step = LogCPM(genes, tmm=True).fit(counts.iloc[analyzed])
    x = step.transform(counts.iloc[analyzed].astype(float))[genes[7]].to_numpy()
    y = frame["case"].to_numpy()[analyzed].astype(float)
    res = sm.OLS(x, sm.add_constant(y)).fit()
    row = next(r for r in rows if r["feature"] == genes[7])
    assert row["estimate"] == pytest.approx(res.params[-1], rel=1e-8)
    assert row["p"] == pytest.approx(res.pvalues[-1], rel=1e-6)


# ── 4 · the coaching, made purpose-specific ──────────────────────────────────


def test_4_do_not_pre_normalize_is_replaced_by_purpose_specific_text(client, depth_csv):
    """The genomics data-type finding told users "Do not pre-normalize these" (``packs.py``),
    DESeq2's advice for its own input, given to a pipeline with no library-size correction
    (ME-09). It now says what each purpose does with the counts.

    Source check — Hornung R, Bernau C, Truntzer C, Wilson R, Stadler T, Boulesteix A-L. A measure
    of the impact of CV incompleteness on prediction error estimation with application to PCA and
    normalization. BMC Med Res Methodol 2015;15:95 (read at PMC4634762). Results: "Performing
    normalization on the entire dataset before CV did not result in a noteworthy optimistic bias
    in any of the investigated cases." Conclusions: "While the investigated forms of normalization
    can be safely performed before CV, PCA has to be performed anew in each CV split to protect
    against optimistic bias." The normalizations studied were RMA and RMA with VSN, on microarray
    data, so the text says so; TurboTab fits its normalizations within each fold regardless.
    """
    pid = open_project(client, depth_csv, "genomics", "case", "prediction")
    data_type = next(f for f in findings(client, pid) if f["id"] == "pack::genomics::data_type")
    text = " ".join(str(data_type.get(k) or "") for k in ("summary", "detail", "why_it_matters"))
    assert "Do not pre-normalize" not in text
    assert "For inference," in text and "For prediction," in text
    assert re.search(r"Hornung and colleagues \(2015\)", text)
    assert '"did not result in a noteworthy optimistic bias"' in text
    assert "microarray" in text
