"""WP14 · Detectors pass must-not-fire fixtures and the null: the acceptance tests of
AUDIT_REPORT.md §5 (closes IN-03, IN-04, IN-09, IN-11, IN-12, IN-13, IN-14, IN-15, IN-16, IN-17,
IN-18, MI-01, MI-04; also H19 and F16).

§5 WP14, as written:

1. "Survey scales: balanced 6-point, skewed 7-point, 0–5 and 9-point blocks raise no code finding;
   a 1–5 block with a real 9 still does; floor-heavy PHQ-9, GAD-7 and a 0–10 rating scale are
   recognized as instruments."
2. "Clinical codes: across eight clean generators at n = 100, 300 and 1,000 (200 replicates each),
   the code-flag rate is ≤ 1% (today up to 15.5%)."
3. "Drift and redundancy: on drift-free data at n = 16–40 the drift finding fires ≤ 5% (today 40/40
   at n = 16); with one sample ×20 the redundancy count stays near 300."
4. "Plausibility: total energy is not in the physiology bands; bands carry source, cycle and
   CONVENTION; children are judged by age-specific z-scores; DBP 0 in NHANES is described per the
   BPX documentation."
5. "Orientation: log2 GEO-style and MZmine-style feature-major tables are asked about under an
   assay lens."
6. "Repeats: a daily falling time course and a crossover are asked (not stated); quarterly recalls
   under the dietary lens are read as replicates or asked."
7. "Lens hints: TPM and log2-TPM with Ensembl IDs are hinted genomics; survey blocks survey; the
   contradiction check runs."
8. "Genomics card: log2-TPM and voom log-CPM are read; thresholds are validated on public matrices;
   single-cell raises a concern naming pseudoreplication."
9. "Histograms: discrete data get resolution-aligned bins, and the two implementations agree on a
   shared test."

The fixtures are the audit's own generators (``docs/turbotab-next/audit/repro.tar.gz``), replayed
with their seeds: ``H-skeptic/h1.py`` (survey blocks, seed 11), ``F-skeptic/f3_direct.py`` (seed
7), ``H-skeptic/h17.py`` (PHQ-9, GAD-7, 0–10, hedonic; seed 12), ``H/make_fx.py`` (codes; seed 7),
``H/sentinel_fp.py`` (the eight clean generators; seed 0), ``F-skeptic/f8_sim.py`` (drift; seed 13),
``F-skeptic/f9_sim.py`` (redundancy; seed 4), ``H-skeptic/h7.py`` (children; seed 4),
``H-skeptic/h13.py`` and ``F-skeptic/f4_orient.py`` (orientation), ``H-skeptic/h14.py`` (repeats;
seed 5), ``H-skeptic/h11.py`` (lens hints; seed 2) and ``C/t5_hist.py`` (histograms; seed 3).

Every reference comes from a path independent of the code under test: the generators' own support
and seeds (a closed form: the answers a block was drawn from), scipy's Spearman correlation and
statsmodels' Benjamini–Hochberg (the pack's drift rule computed directly), statsmodels'
``DescrStatsW.quantile`` (weighted percentiles of the NHANES 2017–2018 public files), the CDC LMS
formula written out from its definition, closed-form scalings of a public count matrix (CPM, TPM,
FPKM, limma's voom log-CPM as its source writes it, edgeR's TMM through rnanorm), and pandas'
``cut``. Primary sources are quoted where a test relies on them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import ProjectState
from turbotab.core.detectors import assay, codes, genomics, lenses, plausibility, scales
from turbotab.core.detectors import orientation as orient
from turbotab.core.detectors import repeats as repeat_reading
from turbotab.core.stages.findings import findings_stage
from turbotab.core.tests.stage_harness import NHANES, SAMPLES, Ingested

HERE = Path(__file__).resolve().parent / "wp14_data"


def run_findings(frame: pd.DataFrame, folder: Path, lens: list[str], name: str = "t") -> list[dict]:
    path = folder / f"{name}.csv"
    frame.to_csv(path, index=False)
    table = Ingested(path, folder / f"{name}_ingest")
    return table.run(findings_stage, ProjectState(lens=lens, target=None))["findings"]


def ids(found: list[dict]) -> list[str]:
    return [f["id"] for f in found]


# ═════════════════════════════════════════════════════════════════════════════
# 1 · Survey scales
# ═════════════════════════════════════════════════════════════════════════════


def _blocks_h1():
    """H-skeptic/h1.py, draw for draw (seed 11, n = 300, ten items each)."""
    rng = np.random.default_rng(11)
    n = 300
    six = pd.DataFrame({f"att_{i+1}": rng.choice(range(1, 7), n, p=np.array([1, 2, 3, 3, 2, 1.2]) / 12.2)
                        for i in range(10)})
    p7 = np.array([10, 18, 22, 22, 17, 7, 4]) / 100
    seven = pd.DataFrame({f"freq_{i+1}": rng.choice(range(1, 8), n, p=p7) for i in range(10)})
    five = pd.DataFrame({f"q{i+1}": rng.choice(range(1, 6), n) for i in range(10)})
    five = five.mask(rng.random(five.shape) < 0.03, 9)
    return six, seven, five


def _blocks_f3():
    """F-skeptic/f3_direct.py's 0–5 block (seed 7; drawn after its five 12-item blocks)."""
    rng = np.random.default_rng(7)

    def make(k, n=400, items=12, probs=None):
        probs = probs if probs is not None else np.ones(k) / k
        return pd.DataFrame({f"wb_{i+1:02d}": rng.choice(np.arange(1, k + 1), size=n, p=probs)
                             for i in range(items)})
    for k, probs in ((6, None), (6, np.array([.15, .2, .2, .2, .15, .10])),
                     (7, np.array([.14, .18, .2, .2, .14, .08, .06])), (7, None), (5, None)):
        make(k, probs=probs)
    d5 = make(5)
    rng.random(d5.shape)
    zero_five = pd.DataFrame({f"q{i}": rng.choice(np.arange(0, 6), size=400) for i in range(12)})
    return zero_five


def _instruments_h17():
    """H-skeptic/h17.py, draw for draw (seed 12, n = 400): a community PHQ-9 (modal shares
    0.45–0.93), the same with 1% 9s, GAD-7, ten 0–10 ratings and a 9-point hedonic block."""
    rng = np.random.default_rng(12)
    n = 400
    probs = [[.55, .30, .10, .05], [.62, .25, .08, .05], [.50, .30, .12, .08], [.45, .33, .13, .09],
             [.60, .25, .09, .06], [.68, .20, .08, .04], [.66, .22, .08, .04], [.80, .13, .05, .02],
             [.93, .05, .015, .005]]
    phq = pd.DataFrame({f"phq_{i+1}": rng.choice(4, n, p=p) for i, p in enumerate(probs)})
    phq_coded = phq.mask(rng.random(phq.shape) < 0.01, 9)
    gad = pd.DataFrame({f"gad_{i+1}": rng.choice(4, n, p=[.45, .30, .15, .10]) for i in range(7)})
    nrs = pd.DataFrame({f"pain_{i+1}": rng.integers(0, 11, n) for i in range(10)})
    hedonic = pd.DataFrame({f"liking_{i+1}": np.clip(np.round(rng.normal(6, 1.8, n)), 1, 9).astype(int)
                            for i in range(10)})
    return phq, phq_coded, gad, nrs, hedonic


@pytest.fixture(scope="module")
def survey_blocks():
    six, seven, five_nine = _blocks_h1()
    phq, phq_coded, gad, nrs, hedonic = _instruments_h17()
    return {"six": six, "seven_skewed": seven, "zero_five": _blocks_f3(), "hedonic9": hedonic,
            "five_nine": five_nine, "phq9": phq, "phq9_coded": phq_coded, "gad7": gad, "nrs": nrs}


@pytest.mark.parametrize("name", ["six", "seven_skewed", "zero_five", "hedonic9"])
def test_1a_real_answers_at_a_scales_end_raise_no_code_finding(survey_blocks, tmp_path, name):
    """The block's own support, the union of the answers it was drawn from, is one contiguous run,
    so nothing breaks it: no survey code finding and no column code finding, through the findings
    stage under the survey lens. The block is still recognized, on the scale it was drawn from."""
    frame = survey_blocks[name]
    found = run_findings(frame, tmp_path, ["survey"], name)
    assert "pack::survey::sentinel_codes" not in ids(found)
    assert not [f for f in found if f["id"].startswith("sentinel_missing__")]
    drawn = sorted(pd.unique(frame.to_numpy().ravel()).tolist())   # the generator's support
    assert drawn == list(range(drawn[0], drawn[-1] + 1))            # one run, by construction
    declared = next(f for f in found if f["id"] == "pack::survey::ordinal_declared")
    assert declared["title"].startswith(f"{frame.shape[1]} columns share one {len(drawn)}-point")


def test_1b_a_one_to_five_block_with_a_real_nine_is_still_flagged(survey_blocks, tmp_path):
    """h1's 1–5 block with 3% of answers replaced by 9: every item holding a 9 is named, with 9 as
    its code, and the repair blanks exactly the 9s (pandas' own mask)."""
    frame = survey_blocks["five_nine"]
    found = run_findings(frame, tmp_path, ["survey"], "five_nine")
    f = next(f for f in found if f["id"] == "pack::survey::sentinel_codes")
    expected = sorted(c for c in frame.columns if (frame[c] == 9).any())
    raw = scales.sentinel_finding(frame, scales.blocks(frame))  # the served finding drops params
    flagged = {e["item"]: e["sentinel_values"] for e in raw["params"]["items"]}
    assert sorted(flagged) == expected and all(v == [9] for v in flagged.values())
    assert sorted(f["affected_columns"]) == expected
    assert f["severity"] == "critical" and "almost certainly" not in f["detail"]
    option = f["repairs"][0]
    assert option["decision"]["params"]["values"] == {c: [9.0] for c in expected}
    assert "on the user's instruction" in option["sentence"]
    assert sum(int((frame[c] == 9).sum()) for c in expected) == int((frame == 9).to_numpy().sum())


@pytest.mark.parametrize("name,items,scale,instrument", [
    ("phq9", 9, (0, 3), "PHQ-9"), ("gad7", 7, (0, 3), "GAD-7"), ("nrs", 10, (0, 10), None)])
def test_1c_floor_heavy_phq9_gad7_and_a_zero_to_ten_scale_are_instruments(
        survey_blocks, tmp_path, name, items, scale, instrument):
    """CLINICAL_SURVEY_PACK §B1.1: "PHQ-9 (9 items, 0–3), GAD-7 (7, 0–3)"; "0–10 (NRS)". The block
    is every item, on the drawn scale, named as a hypothesis where the fingerprint matches; the
    survey hint names it; the floor-heavy PHQ-9's modal shares (0.45–0.93 by construction) do not
    disqualify it."""
    frame = survey_blocks[name]
    found = scales.blocks(frame)
    assert len(found) == 1 and found[0].columns == list(frame.columns)
    assert found[0].scale == scale and found[0].instrument == instrument
    if name == "phq9":
        modal = frame.apply(lambda s: s.value_counts(normalize=True).max())
        assert modal.max() > 0.9  # the item the legacy shape rule rejected
    hint = [h for h in lenses.hints(frame) if h["lens"] == "survey"]
    assert hint and f"{items} columns share one {scale[1] - scale[0] + 1}-point" in hint[0]["because"]
    declared = next(f for f in run_findings(frame, tmp_path, ["survey"], name)
                    if f["id"] == "pack::survey::ordinal_declared")
    assert declared["affected_columns"] == list(frame.columns)[:10]
    assert scales.ordinal_finding(found)["params"]["columns"] == list(frame.columns)


def test_1d_a_coded_phq9_flags_its_nines_and_nothing_else(survey_blocks):
    frame = survey_blocks["phq9_coded"]
    f = scales.sentinel_finding(frame, scales.blocks(frame))
    expected = sorted(c for c in frame.columns if (frame[c] == 9).any())
    assert sorted(e["item"] for e in f["params"]["items"]) == expected


# ═════════════════════════════════════════════════════════════════════════════
# 2 · Clinical codes
# ═════════════════════════════════════════════════════════════════════════════


def test_2a_eight_clean_generators_are_flagged_at_most_one_percent_of_the_time():
    """H/sentinel_fp.py's eight clean generators (seed 0), n = 100, 300 and 1,000, 200 replicates
    each: the share of columns the code reader flags is ≤ 1% in every cell. The legacy reader, on
    the same draws, flags more than 1% somewhere, so the test can fail."""
    from ml.import_doctor import check_numeric_sentinels

    rng = np.random.default_rng(0)
    gens = {
        "adult DBP N(72,11)": lambda n: rng.normal(72, 11, n).round(),
        "pediatric DBP N(60,8)": lambda n: rng.normal(60, 8, n).round(),
        "heart rate N(72,11)": lambda n: rng.normal(72, 11, n).round(),
        "age older cohort N(70,7)": lambda n: rng.normal(70, 7, n).round().clip(50, 100),
        "fasting glucose LN(100,.15)": lambda n: rng.lognormal(np.log(100), 0.15, n).round(),
        "n_children Poisson(2)": lambda n: rng.poisson(2.0, n),
        "drinks/week NB": lambda n: rng.negative_binomial(1, 0.2, n),
        "prior admissions NB": lambda n: rng.negative_binomial(1, 0.45, n),
    }
    reps, rates, legacy = 200, {}, {}
    for name, draw in gens.items():
        for n in (100, 300, 1000):
            new = old = 0
            for _ in range(reps):
                column = pd.DataFrame({"x": draw(n)})
                new += codes.read(column["x"]) is not None
                old += bool(check_numeric_sentinels(column))
            rates[(name, n)], legacy[(name, n)] = new / reps, old / reps
    assert max(rates.values()) <= 0.01, {k: v for k, v in rates.items() if v > 0.01}
    assert max(legacy.values()) > 0.01  # the audit's 2–15.5%, on the same draws


def _make_fx_codes():
    """H/make_fx.py, draw for draw (seed 7, n = 300): its fixtures F (0–10 ratings with 99) and G
    (NHANES yes/no items with 7 = refused and 9 = don't know)."""
    rng = np.random.default_rng(7)
    n = 300

    def likert(k, p=None, start=1):
        return rng.choice(np.arange(start, start + k), size=n, p=p)
    rng.integers(18, 80, n)
    for _ in range(12):
        likert(6)
    p7 = np.array([0.25, 0.22, 0.18, 0.14, 0.10, 0.07, 0.04])
    for _ in range(12):
        likert(7, p7)
    for _ in range(10):
        x = likert(5)
        rng.choice(n, 12, replace=False)
    for _ in range(10):
        likert(7)
        rng.choice(n, 15, replace=False)
    p9 = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.14, 0.25, 0.25, 0.17])
    for _ in range(10):
        likert(9, p9)
    nrs = pd.DataFrame({"id": range(n)})
    for j in range(10):
        x = rng.integers(0, 11, n)
        x[rng.choice(n, 10, replace=False)] = 99
        nrs[f"nrs_{j+1:02d}"] = x
    yesno = pd.DataFrame({"SEQN": np.arange(80000, 80000 + n)})
    for name in ["DIQ010", "BPQ020", "MCQ160B", "SMQ020", "ALQ101", "PAQ605", "DBQ700", "HIQ011"]:
        x = rng.choice([1, 2], n, p=[0.3, 0.7])
        x[rng.choice(n, 6, replace=False)] = 9
        x[rng.choice(n, 3, replace=False)] = 7
        yesno[name] = x
    return nrs, yesno


def test_2b_real_codes_are_still_read_and_extremes_and_strata_are_not(tmp_path):
    """Positive controls: NHANES yes/no items' 7 and 9 (8 of 8, corroborated by the NHANES
    codebook convention, so critical and paged as one card), 99 in each 0–10 rating, and
    clinic_visits.csv's 999 ages and glucose. Must not fire: its systolic 99, the column's
    lowest reading one mmHg below the next (a real extreme), and nhanes_dietary.csv's stratum 999,
    a real stratum the design finding reads (H19)."""
    nrs, yesno = _make_fx_codes()
    found = {f["affected_columns"][0]: f for f in codes.findings(yesno)}
    items = [c for c in yesno.columns if c != "SEQN"]
    assert sorted(found) == sorted(items)
    for c in items:
        expected = sorted(float(v) for v in (7, 9) if (yesno[c] == v).sum() >= 2)
        assert found[c]["params"]["values"] == expected and found[c]["severity"] == "critical"
    stage = run_findings(yesno, tmp_path, ["survey"], "yesno")
    cards = [f for f in stage if f["id"].startswith("sentinel_missing__")]
    assert len(cards) == 8 and {f["group"] for f in cards} == {"missing_codes"}  # one paged card
    found = {f["affected_columns"][0]: f["params"]["values"] for f in codes.findings(nrs)}
    assert found == {c: [99.0] for c in nrs.columns if c != "id"}
    visits = pd.read_csv(SAMPLES / "clinic_visits.csv")
    found = {f["affected_columns"][0]: f for f in codes.findings(visits)}
    assert found["age"]["params"]["values"] == [999.0] and found["glucose"]["params"]["values"] == [999.0]
    assert "bp_2" not in found
    bp = visits["bp_2"].dropna()
    assert bp.min() == 99 and (bp == 99).sum() == 2 and (bp == 100).any()  # 99 is a real extreme
    assert found["age"]["severity"] == "warning"  # no codebook corroborates a clinic export
    strata = pd.read_csv(SAMPLES / "nhanes_dietary.csv")
    assert (strata["SDMVSTRA"] == 999).sum() == 9
    assert "SDMVSTRA" not in {f["affected_columns"][0] for f in codes.findings(strata)}


def test_2c_a_flagged_extreme_is_described_as_what_it_is():
    """The detail never claims a code lies "far outside" values it sits among (the legacy detail
    read "Found 99 (2x) — far outside the rest of the column (36 to 105)")."""
    visits = pd.read_csv(SAMPLES / "clinic_visits.csv")
    f = next(f for f in codes.findings(visits) if f["affected_columns"] == ["glucose"])
    largest = visits.loc[visits["glucose"] != 999, "glucose"].max()
    assert f"the largest is {int(largest)}" in f["detail"]


# ═════════════════════════════════════════════════════════════════════════════
# 3 · Drift and redundancy
# ═════════════════════════════════════════════════════════════════════════════


def _pack_rule(frame: pd.DataFrame) -> int:
    """METABOLOMICS_PACK §01 item 7, computed directly: features with Spearman |ρ| > 0.3 against
    the injection order and a Benjamini–Hochberg q < 0.05 (scipy, statsmodels)."""
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    order = frame["run_order"]
    rho, p = zip(*(stats.spearmanr(order, frame[c]) for c in frame.columns[1:]))
    q = multipletests(np.array(p), method="fdr_bh")[1]
    return int(((np.abs(np.array(rho)) > 0.3) & (q < 0.05)).sum())


def test_3a_drift_free_runs_fire_at_most_five_percent_and_match_the_pack_rule():
    """F-skeptic/f8_sim.py (seed 13): 200 log-normal features with no drift, 40 tables per n. The
    finding fires in at most 2 of 40 (5%) at every n (legacy: 40/40 at n = 16), and the count of
    drifting features equals the pack's rule computed with scipy and statsmodels."""
    from turbotab import packs

    rng = np.random.default_rng(13)
    for n in (16, 20, 24, 30, 40):
        fired = legacy = 0
        for _ in range(40):
            X = np.exp(rng.normal(rng.normal(8, 1.5, 200), 0.4, (n, 200)))
            frame = pd.DataFrame(X, columns=[f"mz_{i:04d}" for i in range(200)])
            frame.insert(0, "run_order", rng.permutation(np.arange(1, n + 1)))
            reading = assay.drift_reading(frame)
            assert reading["n_drifting"] == _pack_rule(frame)
            fired += assay.run_order_finding(frame) is not None
            legacy += packs._acquisition_order(frame) is not None
        assert fired <= 2, (n, fired)
        if n == 16:
            assert legacy == 40  # the audit's number, on the same draws


def test_3b_real_drift_still_fires():
    rng = np.random.default_rng(1)
    for n in (16, 40):
        order = rng.permutation(np.arange(1, n + 1))
        X = np.exp(rng.normal(rng.normal(8, 1.5, 200), 0.4, (n, 200)))
        X[:, :80] *= np.exp(0.08 * (order - n / 2))[:, None]   # 40% of features drift
        frame = pd.DataFrame(X, columns=[f"mz_{i:04d}" for i in range(200)])
        frame.insert(0, "run_order", order)
        f = assay.run_order_finding(frame)
        assert f is not None and f["params"]["n_tracking"] == _pack_rule(frame)
        assert f["params"]["chance_share_beyond_rho"] > 0  # the null share is stated beside it


def test_3c_one_concentrated_sample_leaves_three_hundred_quantities():
    """F-skeptic/f9_sim.py (seed 4): 300 independent log-normal features, n = 80, one sample ×20.
    No pair reaches a Spearman correlation of 0.9 (pandas), so the count is every column (300
    features and bmi) and the finding is silent; the legacy Pearson reading said about 40."""
    from turbotab import packs

    rng = np.random.default_rng(4)
    n, p = 80, 300
    base = np.exp(rng.normal(rng.normal(8, 1.5, p), 0.5, (n, p)))
    X = base.copy()
    X[0] *= 20
    frame = pd.DataFrame(X, columns=[f"mz_{i:04d}" for i in range(p)])
    frame.insert(0, "sample_id", [f"S{i}" for i in range(n)])
    frame["bmi"] = rng.normal(25, 3, n)
    rho = frame.drop(columns="sample_id").corr(method="spearman").to_numpy()
    np.fill_diagonal(rho, 0)
    assert np.abs(rho).max() < 0.9
    reading = assay.redundancy_reading(frame)
    assert reading["effective"] == p + 1 == reading["n_columns"]
    assert assay.redundancy_finding(frame) is None
    legacy = packs._redundancy(frame)
    assert legacy is not None and legacy["params"]["effective_features"] < 50


def test_3d_real_redundancy_is_still_counted():
    """metabolomics_redundant.csv: 100 compounds wearing 404 feature names (its companion .md)."""
    frame = pd.read_csv(SAMPLES / "metabolomics_redundant.csv")
    f = assay.redundancy_finding(frame)
    assert f is not None and 95 <= f["params"]["effective_features"] <= 115
    assert f["params"]["groups_resting_on_one_sample"] == []


# ═════════════════════════════════════════════════════════════════════════════
# 4 · Plausibility
# ═════════════════════════════════════════════════════════════════════════════


def test_4a_total_energy_is_not_a_physiology_band(tmp_path):
    """No energy name earns a band, and nine very low recalls (0–93 kcal) are left to the dietary
    screens: the clinical plausibility finding does not name kcal."""
    for name in ("kcal", "energy", "energy_kcal", "calories", "DR1TKCAL"):
        assert plausibility.variable_of(name) is None
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"kcal": np.r_[rng.normal(2100, 500, 291).round(), np.linspace(0, 93, 9)],
                          "bp_sys": rng.normal(125, 15, 300).round()})
    frame.loc[0, "bp_sys"] = 12.0
    found = run_findings(frame, tmp_path, ["clinical", "dietary"], "energy")
    f = next(f for f in found if f["id"] == "pack::clinical::impossible_vs_extreme")
    assert list(f["repairs"][0]["decision"]["params"]["bands"]) == ["bp_sys"]
    assert "kcal" not in f["detail"]


def test_4b_bands_carry_source_cycle_and_convention_and_match_nhanes():
    """Every band says where it comes from, which NHANES cycle, and that it is a CONVENTION. The
    improbable tier is recomputed from the NHANES 2017–2018 public files (adults ≥ 20, the extract
    in wp14_data, rebuilt by build_plausibility.py) with statsmodels' weighted quantile: equal.
    CLINICAL_SURVEY_PACK §A1.2 marks its limits "[CONVENTION — Kahn et al. note that plausibility
    limits are institution- and observation-specific]"."""
    from statsmodels.stats.weightstats import DescrStatsW

    reference = plausibility.reference()
    assert reference["status"] == "CONVENTION" and "kcal" not in reference["variables"]
    extract = pd.read_csv(HERE / "nhanes_2017_2018_adults.csv.gz")
    assert (extract["RIDAGEYR"] >= 20).all()
    for var, spec in reference["variables"].items():
        imp, low = spec["improbable"], spec["impossible"]
        assert imp["status"] == low["status"] == "CONVENTION", var
        assert imp["cycle"] == "2017-2018" and imp["source"].startswith("NHANES 2017-2018"), var
        assert low["source"], var
        column, weight = imp["nhanes_variable"], imp["weight"]
        ok = extract[column].notna() & extract[weight].notna() & (extract[weight] > 0)
        q = DescrStatsW(extract.loc[ok, column].to_numpy(float),
                        weights=extract.loc[ok, weight].to_numpy(float)).quantile(
            [0.01, 0.99], return_pandas=False)
        assert (imp["p01"], imp["p99"]) == (pytest.approx(q[0]), pytest.approx(q[1])), var
        assert imp["n"] == int(ok.sum())
    # The pack's own table: "SBP | 30–300 mmHg", "DBP | 10–200 mmHg".
    assert (reference["variables"]["bp_sys"]["impossible"]["low"],
            reference["variables"]["bp_sys"]["impossible"]["high"]) == (30, 300)
    assert (reference["variables"]["bp_di"]["impossible"]["low"],
            reference["variables"]["bp_di"]["impossible"]["high"]) == (10, 200)


def _cdc_modified_z(value: float, months: float, sex: int, table: str) -> float:
    """The CDC modified z-score from its definition, read straight off the LMS file: "computed by
    extrapolating one-half of the distance between 0 and +2 (or between 0 and -2) z-scores to the
    distribution's tails" (CDC, SAS Program for CDC Growth Charts)."""
    lms = pd.read_csv(plausibility.DATA / plausibility.LMS_FILES[table])
    lms = lms[pd.to_numeric(lms["Sex"], errors="coerce") == sex].astype(float)
    L, M, S = (float(np.interp(months, lms["Agemos"], lms[k])) for k in ("L", "M", "S"))
    z2 = M * (1 + 2 * L * S) ** (1 / L) if value >= M else M * (1 - 2 * L * S) ** (1 / L)
    return (value - M) / (abs(z2 - M) / 2)


def test_4c_children_are_judged_by_age_specific_z_scores(tmp_path):
    """H-skeptic/h7.py (seed 4): 300 children aged 2–10 with one 0.2 kg weight. Adult limits are
    not applied to them (the legacy reading called 279–298 of them abnormal or set the column
    aside): the 0.2 kg child is a CDC biologically implausible value (modified z below −5 for both
    sexes, computed here from the LMS table), and no real child is."""
    rng = np.random.default_rng(4)
    n = 300
    age = rng.integers(2, 11, n)
    wt = (8 + 2.6 * age + rng.normal(0, 2.5, n)).round(1)
    wt[:1] = [0.2]
    frame = pd.DataFrame({"age": age, "weight_kg": wt,
                          "height_cm": (75 + 6.5 * age + rng.normal(0, 4, n)).round(1)})
    reading = plausibility.read(frame)
    weight = next(e for e in reading["columns"] if e["column"] == "weight_kg")
    assert weight["n_adult_rows_read"] == 0 and weight["n_outside_central_98"] == 0
    assert weight["impossible_tier"] == "any_age" and weight["n_impossible"] == 1  # 0.2 kg
    months = age * 12 + 6.0
    flagged_by_hand = [i for i in range(n)
                       if all(_cdc_modified_z(wt[i], months[i], s, "weight") < -5
                              or _cdc_modified_z(wt[i], months[i], s, "weight") > 8 for s in (1, 2))]
    assert flagged_by_hand == [0] and weight["children"]["rows"] == flagged_by_hand
    for v, m, s in ((wt[5], months[5], 1), (wt[9], months[9], 2)):
        z = plausibility.modified_z(pd.Series([v]), pd.Series([m]), float(s), "weight").iloc[0]
        assert z == pytest.approx(_cdc_modified_z(v, m, s, "weight"), abs=1e-9)
    f = run_findings(frame, tmp_path, ["clinical"], "peds")
    finding = next(x for x in f if x["id"] == "pack::clinical::impossible_vs_extreme")
    assert "adult limits are not applied" in finding["detail"]
    assert "abnormal" not in finding["detail"]


def test_4d_a_diastolic_zero_is_described_per_the_nhanes_documentation(tmp_path):
    """NHANES 2017–2018 BPX_J documentation, Data Processing and Editing: "Systolic BP and maximum
    inflation level cannot be greater than 300 mmHg; … and Diastolic BP can be zero."; BPXDI1:
    "0 to 136 Range of Values". A 0 is described as a recorded reading, not an entry error, and the
    usual treatment (set to missing) is offered."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"SEQN": np.arange(1, 501), "RIDAGEYR": rng.integers(20, 80, 500),
                          "bp_sys": rng.normal(122, 16, 500).round(),
                          "bp_di": rng.normal(70, 11, 500).round()})
    frame.loc[:4, "bp_di"] = 0.0
    found = run_findings(frame, tmp_path, ["clinical"], "dbp0")
    f = next(x for x in found if x["id"] == "pack::clinical::impossible_vs_extreme")
    assert "Diastolic BP can be zero" in f["detail"] and "entry error" not in f["detail"].replace(
        "not an entry error", "")
    assert f["repairs"][0]["key"] == "set_missing"
    assert f["repairs"][0]["decision"]["params"]["bands"]["bp_di"] == [10.0, 200.0]


def test_4f_hba1c_in_ifcc_units_is_a_unit_question_not_an_impossible_value():
    """H-skeptic/h7.py's mixed column (seed 4): 480 HbA1c values in % and 20 in IFCC mmol/mol.
    NGSP master equation (ngsp.org/ifccngsp.asp): "NGSP = [0.09148 * IFCC] + 2.152". Every value
    above 30% that the equation turns into a plausible percentage is reported as a unit question,
    and the column gets no blanking repair (the legacy reading called them impossible)."""
    rng = np.random.default_rng(4)
    h = np.r_[rng.normal(5.8, 0.8, 480).round(1), rng.normal(40, 9, 20).round(0)]
    frame = pd.DataFrame({"hba1c": h, "age": rng.integers(30, 70, 500)})
    entry = next(e for e in plausibility.read(frame)["columns"] if e["column"] == "hba1c")
    as_ngsp = 0.09148 * h + 2.152
    expected = int(((h > 30) & (as_ngsp >= 2) & (as_ngsp <= 30)).sum())
    assert entry["unit_question"]["n"] == expected and entry["n_impossible"] == int((h > 30).sum()) - expected
    f = plausibility.impossible_vs_extreme_finding(frame)
    assert f["params"]["columns"] == [] and "NGSP master equation" in f["detail"]


@pytest.mark.skipif(not NHANES.is_file(), reason="the real NHANES export is untracked")
def test_4e_the_nhanes_export_gets_sourced_bands_and_no_energy_band(tmp_path):
    table = Ingested(NHANES, tmp_path / "nhanes")
    found = table.run(findings_stage, ProjectState(lens=["dietary", "clinical"], target=None))["findings"]
    f = next(x for x in found if x["id"] == "pack::clinical::impossible_vs_extreme")
    assert "kcal" not in f["repairs"][0]["decision"]["params"]["bands"]
    assert "NHANES 2017-2018" in f["detail"] and "abnormal" not in f["detail"]


# ═════════════════════════════════════════════════════════════════════════════
# 5 · Orientation
# ═════════════════════════════════════════════════════════════════════════════


def _reading(frame: pd.DataFrame, folder: Path, name: str) -> dict:
    from turbotab.core.stages.working import named_reading, orientation_reading

    path = folder / f"{name}.csv"
    frame.to_csv(path, index=False)
    table = Ingested(path, folder / name)
    return named_reading(table.parquet, table.info, orientation_reading(table.parquet, table.info))


def _asked(reading: dict, lens: list[str]) -> bool:
    from turbotab.core.interview import _orientation_gate

    return _orientation_gate(ProjectState(lens=lens), {"reading": reading}) is None


def _geo(log: bool) -> pd.DataFrame:
    """H-skeptic/h13.py (seed 8): 3,000 probe sets × 30 GSM samples, features in rows."""
    rng = np.random.default_rng(8)
    F, S = 3000, 30
    base = rng.lognormal(np.log(300), 2.2, F)
    raw = base[:, None] * rng.lognormal(0, 0.35, (F, S))
    frame = pd.DataFrame(np.log2(raw + 1) if log else raw, columns=[f"GSM{j:05d}" for j in range(S)])
    frame.insert(0, "ID_REF", [f"{i}_at" for i in range(F)])
    return frame


def _mzmine() -> pd.DataFrame:
    """F-skeptic/f4_orient.py: metabolomics_untargeted.csv turned features-in-rows, with MZmine's
    ``row ID``, ``row m/z`` and ``row retention time`` beside the samples (seed 0)."""
    df = pd.read_csv(SAMPLES / "metabolomics_untargeted.csv")
    feats = [c for c in df.select_dtypes(include=[np.number]).columns
             if c not in ("age", "bmi", "run_order", "case", "outcome", "condition", "responder")]
    t = df.set_index("sample_id")[feats].T
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({"row ID": np.arange(1, len(t) + 1), "row m/z": rng.uniform(80, 1200, len(t)),
                         "row retention time": rng.uniform(0.5, 20, len(t))})
    return pd.concat([meta, t.reset_index(drop=True)], axis=1)


@pytest.mark.parametrize("name", ["geo_log2", "geo_raw", "mzmine"])
@pytest.mark.parametrize("lens", [["genomics"], ["metabolomics"]])
def test_5a_log2_geo_and_mzmine_tables_are_asked_about_under_an_assay_lens(tmp_path, name, lens):
    frame = {"geo_log2": lambda: _geo(True), "geo_raw": lambda: _geo(False), "mzmine": _mzmine}[name]()
    reading = _reading(frame, tmp_path, name)
    assert reading["reading"] == "feature_major" and reading["confidence"] != "high"
    assert _asked(reading, lens)
    assert not _asked(reading, ["clinical"])  # no assay lens: not asked


def test_5b_the_shape_alone_reads_a_log_scale_matrix_by_its_own_spreads(tmp_path):
    """With no name to read (``feature`` F00001…, columns c00…), the log2 matrix's row means spread
    far more than its column means, computed here with numpy on the values themselves; the legacy
    statistic on log10 |mean| read it "undetermined"."""
    frame = _geo(True).rename(columns=lambda c: "c" + c[-2:] if c.startswith("GSM") else "feature")
    frame["feature"] = [f"F{i:05d}" for i in range(len(frame))]
    reading = _reading(frame, tmp_path, "plain_log2")
    block = frame.drop(columns="feature").to_numpy()
    ratio = block.mean(axis=1).std(ddof=1) / block.mean(axis=0).std(ddof=1)
    assert reading["basis"] == "shape" and reading["reading"] == "feature_major"
    assert reading["ratio"] == pytest.approx(ratio, rel=1e-3)
    from turbotab import orientation as legacy
    assert legacy.read(frame)["reading"] == "undetermined"


def test_5c_a_sample_major_assay_is_asked_with_its_reading_as_the_proposal(tmp_path):
    """Superseded by the readings ledger (BLUEPRINT §14.1, 2026-10-03): WP14 skipped the question on
    a sample-major reading, but the orientation reader is never high (its evidence is the header's
    grammar and the shape), and a consumer reads only settled readings. The reading stands as the
    question's proposal; a settled (high) one would still skip it."""
    reading = _reading(pd.read_csv(SAMPLES / "metabolomics_untargeted.csv"), tmp_path, "met")
    assert reading["reading"] == "sample_major" and reading["confidence"] != "high"
    assert _asked(reading, ["metabolomics"])
    assert not _asked({**reading, "confidence": "high"}, ["metabolomics"])


# ═════════════════════════════════════════════════════════════════════════════
# 6 · Repeats
# ═════════════════════════════════════════════════════════════════════════════


def _h14():
    """H-skeptic/h14.py, draw for draw (seed 5)."""
    rng = np.random.default_rng(5)
    rows = []
    for pid in range(40):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 60)))
        for d in range(10):
            rows.append({"pid": pid, "date": (start + pd.Timedelta(days=d)).date().isoformat(),
                         "glucose": 120 - 1.5 * d + rng.normal(0, 3)})
    daily = pd.DataFrame(rows)
    rows = []
    for pid in range(40):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 60)))
        seq = ["A", "B", "A", "B"] if pid % 2 else ["B", "A", "B", "A"]
        for k in range(4):
            rows.append({"pid": pid, "date": (start + pd.Timedelta(weeks=k)).date().isoformat(),
                         "treatment": seq[k], "sbp": 130 + (-5 if seq[k] == "B" else 0) + rng.normal(0, 4)})
    crossover = pd.DataFrame(rows)
    rows = []
    for pid in range(40):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 60)))
        for k in range(4):
            rows.append({"pid": pid, "recall": k + 1,
                         "date": (start + pd.Timedelta(days=91 * k + int(rng.integers(-5, 6)))).date().isoformat(),
                         "kcal": rng.normal(2000, 400)})
    quarterly = pd.DataFrame(rows)
    return daily, crossover, quarterly


def _gate(reading: dict) -> tuple | None:
    """The interview's repeat-kind gate on a structure artifact holding this reading."""
    from turbotab.core.decisions import GrainSpec
    from turbotab.core.interview import _repeat_kind_gate

    state = ProjectState(grain=GrainSpec(grain="repeated", id_column="pid"))
    return _repeat_kind_gate(state, {"repeats": reading, "units": {"column": "pid"}})


def test_6a_a_falling_time_course_and_a_crossover_are_asked():
    """The legacy reading stated both as repeats ("too uneven to be a visit schedule" at a gap CV
    of 0). Here glucose falls within every unit (scipy's Spearman, by hand below) and a treatment
    changes within units, so both are asked."""
    from scipy import stats

    from turbotab import repeats as legacy

    daily, crossover, _ = _h14()
    for frame in (daily, crossover):
        assert legacy.read(frame, "pid")["stated"] is True
        reading = repeat_reading.read(frame, "pid", ["clinical"])
        assert reading["stated"] is False and reading["reading"] is None
        assert _gate(reading) is None  # asked
        assert "too uneven" not in reading["sentence"]
    rhos = [stats.spearmanr(np.arange(len(b)), b["glucose"])[0] for _, b in daily.groupby("pid")]
    assert np.median(rhos) < -0.5
    assert repeat_reading.read(daily, "pid", ["clinical"])["trend"]["direction"] == "falls"
    assert repeat_reading.read(crossover, "pid", ["clinical"])["period_column"] == "treatment"


def test_6b_quarterly_recalls_under_the_dietary_lens_are_not_stated_as_time_points():
    """Four 24-hour recalls 91 days apart: the legacy reading stated "time points: averaging them
    destroys the signal". Under the dietary lens a recall is a repeated measure of usual intake, so
    the reading is replicates or a question, never stated time points."""
    from turbotab import repeats as legacy

    _, _, quarterly = _h14()
    assert legacy.read(quarterly, "pid")["reading"] == "time_points"
    reading = repeat_reading.read(quarterly, "pid", ["dietary"])
    assert reading["reading"] in (None, "repeats")
    assert not (reading["stated"] and reading["reading"] == "time_points")
    assert reading["recall_column"] == "recall"


def test_6c_the_verified_readings_still_stand():
    """Coverage §6.3: 24-hour recalls 3–14 days apart read as replicates; clinic visits about 90
    days apart read as time points. A bare visit index with no dates is asked (I9)."""
    recalls = pd.read_csv(SAMPLES / "dietary_recalls.csv")
    r = repeat_reading.read(recalls, "participant_id", ["dietary"])
    assert (r["reading"], r["stated"]) == ("repeats", True)
    visits = pd.read_csv(SAMPLES / "clinical_longitudinal.csv")
    r = repeat_reading.read(visits, "subject_id", ["clinical"])
    assert (r["reading"], r["stated"]) == ("time_points", True)
    r = repeat_reading.read(visits.drop(columns="visit_date"), "subject_id", ["clinical"])
    assert r["stated"] is False


# ═════════════════════════════════════════════════════════════════════════════
# 7 · Lens hints
# ═════════════════════════════════════════════════════════════════════════════


def _tpm_tables():
    """H-skeptic/h11.py (seed 2): 40 samples × 500 Ensembl genes, TPM (each sample sums to 10^6)
    and log2(TPM + 1)."""
    rng = np.random.default_rng(2)
    G, N = 500, 40
    base = rng.lognormal(1.5, 2.0, G)
    tpm = base[None, :] * rng.lognormal(0, .4, (N, G))
    tpm = tpm / tpm.sum(1, keepdims=True) * 1e6
    cols = [f"ENSG{100000 + i:011d}" for i in range(G)]
    out = {}
    for name, m in (("tpm", tpm), ("log2tpm", np.log2(tpm + 1))):
        frame = pd.DataFrame(m.round(3), columns=cols)
        frame.insert(0, "sample_id", [f"S{i}" for i in range(N)])
        out[name] = frame
    return out


def test_7a_tpm_and_log2_tpm_with_ensembl_ids_are_hinted_genomics(tmp_path):
    from turbotab import packs
    from turbotab.core.stages.data import profile_stage

    for name, frame in _tpm_tables().items():
        assert [h["lens"] for h in packs.suggest(frame)["hints"]] == ["metabolomics"]  # the audit's
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        hints = Ingested(path, tmp_path / name).run(profile_stage, ProjectState())["lens_hints"]
        assert [h["lens"] for h in hints] == ["genomics"], hints


def test_7b_survey_blocks_are_hinted_survey_and_never_metabolomics():
    for name in ("survey_instrument", "survey_sentinels"):
        hints = [h["lens"] for h in lenses.hints(pd.read_csv(SAMPLES / f"{name}.csv"))]
        assert hints == ["survey"], (name, hints)
    for f in ("genomics_cpm", "genomics_fpkm", "genomics_vst", "genomics_microarray",
              "genomics_tmm_cpm", "genomics_estimated_counts"):
        assert [h["lens"] for h in lenses.hints(pd.read_csv(SAMPLES / f"{f}.csv"))] == ["genomics"], f
    assert [h["lens"] for h in lenses.hints(pd.read_csv(SAMPLES / "metabolomics_untargeted.csv"))] \
        == ["metabolomics"]


def test_7c_the_contradiction_check_runs_on_the_stated_lens(tmp_path):
    """A TPM matrix described as a survey: the findings stage raises the contradiction, routed to
    the lens question; the same table under the genomics lens raises none (log2-TPM included, which
    the legacy check called "not any of the … shapes an expression matrix comes in")."""
    tables = _tpm_tables()
    found = run_findings(tables["tpm"], tmp_path, ["survey"], "tpm_survey")
    f = next(x for x in found if x["id"] == "voice::lens_contradiction")
    assert f["routes_to"] == "lens" and "assay panel" in f["detail"]
    for name in ("tpm", "log2tpm"):
        assert "voice::lens_contradiction" not in ids(run_findings(tables[name], tmp_path, ["genomics"],
                                                                   f"{name}_genomics"))


# ═════════════════════════════════════════════════════════════════════════════
# 8 · Genomics card
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def gse60450():
    """GEO GSE60450 (Fu et al. 2015, mouse mammary gland; 12 samples), the authors' gene-wise
    counts with gene lengths, as published at
    https://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60450/suppl/GSE60450_Lactation-GenewiseCounts.txt.gz;
    a seeded random 5,000 of its 27,179 genes (``wp14_data``), samples in rows."""
    raw = pd.read_csv(HERE / "GSE60450_Lactation-GenewiseCounts_5000.tsv.gz", sep="\t")
    length = raw["Length"].to_numpy(dtype=float)
    counts = raw.drop(columns=["EntrezGeneID", "Length"]).to_numpy(dtype=float).T
    genes = [f"g{e}" for e in raw["EntrezGeneID"]]
    return counts, length, genes


def _frame(matrix: np.ndarray, genes: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(np.round(matrix, 4), columns=genes)
    frame.insert(0, "sample_id", [f"S{i:02d}" for i in range(len(matrix))])
    return frame


def test_8a_log2_tpm_and_voom_log_cpm_are_read(gse60450):
    """Closed forms on the public counts: CPM = c / N × 10^6; TPM = (c / length) / Σ(c / length) ×
    10^6; and voom's log-CPM exactly as limma writes it (voom.R: ``y <- t(log2(t(counts+0.5)/
    (lib.size+1)*1e6))``, ``lib.size <- colSums(counts)``). The legacy card read all three as
    nothing."""
    from turbotab import packs

    counts, length, genes = gse60450
    lib = counts.sum(1, keepdims=True)
    cpm = counts / lib * 1e6
    rpk = counts / length[None, :] * 1e3
    tpm = rpk / rpk.sum(1, keepdims=True) * 1e6
    voom = np.log2((counts + 0.5) / (lib + 1) * 1e6)
    for name, matrix, offset in (("log2(TPM+1)", np.log2(tpm + 1), 1.0),
                                 ("log2(CPM+1)", np.log2(cpm + 1), 1.0), ("voom", voom, 0.0)):
        frame = _frame(matrix, genes)
        assert not (packs.data_type_card(frame) or {}).get("read"), name
        card = genomics.card(frame)
        assert card["read"] and card["classification"]["keys"] == [genomics.LOG_CPM], name
        assert card["classification"]["offset"] == offset, name
        f = genomics.findings(frame)[0]
        assert f["id"] == "pack::genomics::data_type" and "limma" in f["why_it_matters"]
        assert any("count model" in c for c in f["params"]["closed"])


def test_8b_the_thresholds_hold_on_the_public_matrix(gse60450):
    """The pack's thresholds were measured on fixtures generated from one synthetic table (F16).
    On GSE60450: raw counts read as raw counts; CPM and TPM as "CPM or TPM"; edgeR's TMM-scaled CPM
    (rnanorm's TMM, validated against edgeR) as composition-scaled; FPKM lists FPKM first (its
    library-size spread is under the midpoint of the two measured spreads); a binomially thinned
    copy whose largest count is under 10,000 still reads as raw counts."""
    from rnanorm import TMM

    counts, length, genes = gse60450
    lib = counts.sum(1, keepdims=True)
    keys = lambda m: genomics.card(_frame(m, genes))["classification"]["keys"]  # noqa: E731
    assert keys(counts) == ["raw_counts"]
    assert keys(counts / lib * 1e6) == ["cpm_or_tpm"]
    rpk = counts / length[None, :] * 1e3
    assert keys(rpk / rpk.sum(1, keepdims=True) * 1e6) == ["cpm_or_tpm"]
    factors = TMM().fit(counts).get_norm_factors(counts)
    tmm_cpm = counts / (lib * factors[:, None]) * 1e6
    # The pack's TMM row ("roughly but not exactly equal near 1e6", read as within 25%) fails here:
    # the lactating samples' factors are near 0.64, so their totals reach about 1.56 million.
    assert np.abs(tmm_cpm.sum(axis=1) / 1e6 - 1).max() > 0.25
    assert keys(tmm_cpm) == ["tmm_scaled_cpm"]
    assert keys(counts / length[None, :] * 1e3 / lib * 1e6)[0] == "fpkm"
    shallow = np.random.default_rng(0).binomial(counts.astype(int), 0.005).astype(float)
    assert shallow.max() < 1e4
    assert keys(shallow) == ["raw_counts"]


def test_8c_single_cell_raises_a_concern_naming_pseudoreplication(tmp_path):
    """GENOMICS_PACK §11: "Bulk pipeline on a single-cell matrix | SETTLED wrong | Zero-inflation,
    pseudoreplication". Source check (Europe PMC abstracts): Zimmerman, Espeland & Langefeld 2021,
    Nat Commun 12:738, "Cells from the same individual share common genetic and environmental
    backgrounds and are not statistically independent; therefore, they are subsamples or
    pseudoreplicates"; Squair et al. 2021, Nat Commun 12:5692, "Methods that ignore this
    inevitable variation are biased and prone to false discoveries". The remedy is disputed
    between them (pseudobulk; a random effect for the individual), and the finding says so. The
    legacy card's out-of-scope reading reached no person."""
    frame = pd.read_csv(SAMPLES / "genomics_single_cell.csv")
    found = run_findings(frame, tmp_path, ["genomics"], "sc")
    f = next(x for x in found if x["id"] == "pack::genomics::single_cell")
    assert "pseudoreplication" in f["detail"] and "pseudobulk" in f["detail"]
    assert "random effect" in f["detail"] and "disputed" in f["detail"]
    assert genomics.findings(frame)[0]["params"]["concern"] == "pseudoreplication"


# ═════════════════════════════════════════════════════════════════════════════
# 9 · Histograms
# ═════════════════════════════════════════════════════════════════════════════


def test_9_discrete_data_get_resolution_aligned_bins_and_both_histograms_agree(tmp_path):
    """C/t5_hist.py and t5b.py: whole ages 18–80, a 0–40 score, a 0.1 grid, HbA1c to 0.1, a 0.5
    grid, and a continuous column. Each bin holds the same number of grid steps (the last may hold
    fewer), so there is no sawtooth; the datastore's DuckDB histogram and the previews' numpy
    histogram give the same edges and counts; and the counts equal pandas' ``cut`` on the edges."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from turbotab.core.consequences import _histogram_pair
    from turbotab.core.datastore import DataStore, ingest

    rng = np.random.default_rng(3)
    n = 60000
    columns = {"age": (rng.integers(18, 81, n), 1.0), "score": (rng.integers(0, 41, n), 1.0),
               "x1dp": (np.round(rng.uniform(0, 3, n), 1), 0.1),
               "hba1c": (np.round(np.clip(rng.lognormal(np.log(5.6), 0.12, n), 4.0, 12.0), 1), 0.1),
               "half": (np.round(rng.normal(10, 2, n) * 2) / 2, 0.5),
               "continuous": (rng.normal(0, 1, n), None)}
    pq.write_table(pa.table({k: v for k, (v, _) in columns.items()}), tmp_path / "h.parquet")
    ingest(tmp_path / "h.parquet", tmp_path / "h.ing.parquet")
    with DataStore(tmp_path / "h.ing.parquet", 4 << 30) as store:
        for name, (values, step) in columns.items():
            h = store.histogram(name, 30)
            pair, _ = _histogram_pair(values, values)
            assert h["edges"] == pytest.approx(pair.edges) and h["counts"] == pair.counts, name
            edges = np.array(h["edges"])
            by_pandas = pd.cut(pd.Series(values), bins=edges, right=False,
                               include_lowest=True).value_counts(sort=False).to_numpy()
            by_pandas[-1] += int((values == edges[-1]).sum())   # the last bin is closed
            assert h["counts"] == by_pandas.tolist(), name
            if step is None:
                assert len(h["counts"]) == 30
                continue
            grid = np.arange(values.min(), values.max() + step / 2, step)
            per_bin = np.histogram(grid, edges)[0]
            assert len(set(per_bin[:-1].tolist())) == 1 and per_bin[-1] <= per_bin[0], name
            assert edges[0] == pytest.approx(values.min() - step / 2), name


# ═════════════════════════════════════════════════════════════════════════════
# Repair round (2026-10-03): the independent verifier's variants of each test's failure class,
# replayed draw for draw from its probes (verify-intelligence/probes/wp14_codes.py, wp14_change.py,
# wp14_repeats.py, wp14_lenses.py, api_ffq2.py, api_mri.py, wp14_geo.py).
# ═════════════════════════════════════════════════════════════════════════════


def _verifier_generators(rng: np.random.Generator) -> dict:
    """probes/wp14_codes.py's twelve clean generators: answers and counts whose top or bottom
    value is real, never a code (their support is the generator's own)."""
    def heap_sbp(n):
        x = rng.normal(128, 17, n)
        m = rng.random(n) < 0.3
        x[m] = np.round(x[m] / 10) * 10
        return np.round(x)
    return {
        "days/week 0-7 heaped at 7": lambda n: rng.choice(8, n, p=[.30, .08, .12, .15, .08, .12, .03, .12]),
        "days/week active-only 1-7": lambda n: rng.choice(np.arange(1, 8), n, p=[.10, .18, .25, .12, .17, .05, .13]),
        "integer change N(2.5,1.2)": lambda n: np.round(rng.normal(2.5, 1.2, n)),
        "integer change N(3,1.4)": lambda n: np.round(rng.normal(3, 1.4, n)),
        "Apgar 1-min": lambda n: rng.choice(np.arange(3, 11), n, p=[.01, .02, .03, .06, .12, .40, .33, .03]),
        "SBP digit preference": heap_sbp,
        "pain NRS 0-10 floor": lambda n: rng.choice(11, n, p=np.array([30, 8, 9, 9, 7, 9, 6, 7, 6, 3, 6]) / 100),
        "sleep hours 4-10": lambda n: rng.choice(np.arange(4, 11), n, p=[.04, .12, .28, .33, .17, .05, .01]),
        "cigs/day heaped": lambda n: rng.choice([0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40], n,
                                                p=[.70, .02, .02, .02, .04, .07, .03, .06, .01, .02, .01]),
        "household size 1-9": lambda n: np.minimum(1 + rng.poisson(1.8, n), 12),
        "fruit servings/day 0-9": lambda n: rng.poisson(2.2, n),
        "eGFR integer": lambda n: np.round(np.clip(rng.normal(88, 22, n), 5, 140)),
    }


def test_2d_the_verifiers_clean_generators_stay_under_one_percent():
    """Verifier, test 2: an integer change score whose only negative value is −1 had −1 flagged in
    13–69% of columns, because "a negative code in a column of non-negative values is never a real
    observation" ignored that −1 is adjacent to 0; a days-per-week item heaped at 7 had 7 flagged.
    Expected (§5's bound, at §5's sizes): every one of the verifier's twelve generators, seed
    20261003, 200 replicates, is flagged at most 1% of the time at n = 100, 300 and 1,000. At
    n = 50, below §5's grid, the heaped days item stays near 2% (see the deviations)."""
    rng = np.random.default_rng(20261003)
    rates = {}
    for name, draw in _verifier_generators(rng).items():
        for n in (50, 100, 300, 1000):
            hits = sum(codes.read(pd.Series(draw(n))) is not None for _ in range(200))
            rates[(name, n)] = hits / 200
    over = {k: v for k, v in rates.items() if k[1] >= 100 and v > 0.01}
    assert not over, over
    assert max(v for k, v in rates.items() if k[1] == 50) <= 0.03, rates


def test_2e_a_minus_one_beside_zero_is_an_answer_and_real_codes_still_read():
    """probes/wp14_change.py (``default_rng(38)``, n = 300): an integer change score holding −1 and
    0. Expected: no code; a −9 two steps or more below a non-negative column's smallest value, and
    a −1 below a 1–5 scale (0 unused between), are still read as codes (the gap is NumPy's)."""
    rng = np.random.default_rng(38)
    change = pd.Series(np.round(rng.normal(2.5, 1.2, 300)).astype(int))
    assert (change == -1).sum() >= 1 and (change == 0).any()
    assert codes.read(change) is None
    rng = np.random.default_rng(39)
    score = rng.integers(0, 101, 300).astype(float)
    score[:6] = -9
    found = codes.read(pd.Series(score))
    assert found is not None and found["values"] == {-9.0: 6}
    real = score[score != -9]
    assert real.min() - (-9) >= 2  # an unused value lies between
    likert = rng.integers(1, 6, 300).astype(float)
    likert[:5] = -1
    found = codes.read(pd.Series(likert))
    assert found is not None and found["values"] == {-1.0: 5}
    f = codes.findings(pd.DataFrame({"x": likert}))[0]
    assert "far beyond the rest" not in f["detail"]  # −1 is two steps below 1, not far


def _peds(age_name: str | None, seed: int = 4) -> pd.DataFrame:
    """H-skeptic/h7.py's 300 children aged 2–10 (``default_rng(4)``) with a 0.2 kg weight, the age
    column named as the verifier named it (or absent)."""
    rng = np.random.default_rng(seed)
    n = 300
    age = rng.integers(2, 11, n)
    wt = (8 + 2.6 * age + rng.normal(0, 2.5, n)).round(1)
    wt[:1] = [0.2]
    frame = pd.DataFrame({"weight_kg": wt})
    if age_name:
        frame[age_name] = age
    return frame


@pytest.mark.parametrize("age_name", ["child_age", "age_child", "age_at_visit", "AgeAtExam",
                                      "visit_age"])
def test_4g_an_age_named_for_its_occasion_still_judges_children_by_z_scores(age_name, tmp_path):
    """Verifier, test 4: with ``child_age``, ``age_child``, ``age_at_visit``, ``AgeAtExam`` or
    ``visit_age``, 300 children aged 2–10 were read against adult bands ("299 adult values outside
    47.2–157.4 kg … must be kept"). CLINICAL_SURVEY_PACK §A1.2: "Pediatric and growth data: never
    apply adult bounds". Expected: every row is a child (the fixture's ages, 2–10), no adult row is
    read, and the CDC modified z-scores flag exactly the rows the test's own LMS computation flags."""
    frame = _peds(age_name)
    reading = plausibility.read(frame)
    assert reading["n_children"] == len(frame)
    weight = next(e for e in reading["columns"] if e["column"] == "weight_kg")
    assert weight["n_adult_rows_read"] == 0 and weight["n_outside_central_98"] == 0
    months = frame[age_name].to_numpy() * 12 + 6.0
    wt = frame["weight_kg"].to_numpy()
    by_hand = [i for i in range(len(frame))
               if all(not -5 <= _cdc_modified_z(wt[i], months[i], s, "weight") <= 8 for s in (1, 2))]
    assert weight["children"]["rows"] == by_hand
    f = next(x for x in run_findings(frame, tmp_path, ["clinical"], age_name)
             if x["id"] == "pack::clinical::impossible_vs_extreme")
    assert "adult values" not in f["detail"] and "adult limits are not applied" in f["detail"]


def test_4h_without_an_age_column_no_child_is_called_an_unusual_adult(tmp_path):
    """Verifier, test 4: with no age column the detail said both "Set aside rather than judged …
    children are judged by age-specific z-scores, never by adult limits" and "299 adult values
    outside 47.2–157.4 kg … must be kept". Expected: the adult percentiles are not read (no adult
    row is known), the any-age limits still catch the 0.2 kg weight, and the detail does not
    contradict itself."""
    frame = _peds(None)
    reading = plausibility.read(frame)
    weight = next(e for e in reading["columns"] if e["column"] == "weight_kg")
    assert weight["n_outside_central_98"] == 0 and weight["n_adult_rows_read"] == 0
    assert weight["impossible_tier"] == "any_age" and weight["n_impossible"] == 1
    f = next(x for x in run_findings(frame, tmp_path, ["clinical"], "no_age")
             if x["id"] == "pack::clinical::impossible_vs_extreme")
    assert "adult values" not in f["detail"]
    assert "Set aside rather than judged" not in f["detail"]
    assert "Not judged by adult percentiles" in f["detail"]
    # A woman's age at diagnosis or a mother's age is not the participant's age.
    for other in ("mother_age", "age_at_diagnosis", "gestational_age"):
        assert plausibility.age_column(_peds(other)) is None, other


def _prepost(spacing: int, jitter: int, seed: int = 8) -> pd.DataFrame:
    """probes/api_drive.py's ``prepost`` table (``default_rng(8)``): 40 people, two visits
    ``spacing`` ± ``jitter`` days apart, systolic pressure 10 mmHg lower at the second (noise SD
    5)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(40):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 60)))
        for j in range(2):
            rows.append({"pid": pid,
                         "date": (start + pd.Timedelta(days=spacing * j + int(rng.integers(-jitter, jitter + 1)))).date().isoformat(),
                         "age": 50 + pid % 20, "sbp": 130 - 10 * j + rng.normal(0, 5)})
    return pd.DataFrame(rows)


def test_6d_a_two_visit_change_and_close_irregular_visits_are_asked():
    """Verifier, test 6: trend evidence needed three records per unit, so a pre/post design whose
    pressure fell 10 mmHg within every person (paired t ≈ 9) was stated "repeats" from spacing
    alone ("close together and irregular"), and the repeat-kind question was skipped; under the
    dietary lens 7 ± 3 and 14 ± 6 days were stated too. IN-12's remedy: "ask unless the evidence is
    unambiguous; treat spacing as weak evidence". Expected: asked under both lenses, the change
    named as evidence; the reference is SciPy's paired t-test and sign test on the fixture."""
    from scipy import stats

    for spacing, jitter, lens in ((10, 4, ["clinical"]), (7, 3, ["dietary"]), (14, 6, ["dietary"])):
        frame = _prepost(spacing, jitter)
        first = frame.groupby("pid")["sbp"].first().to_numpy()
        second = frame.groupby("pid")["sbp"].last().to_numpy()
        t = stats.ttest_rel(second, first)
        sign = stats.binomtest(int((second < first).sum()), len(first), 0.5)
        assert t.statistic < -5 and sign.pvalue < 0.01
        reading = repeat_reading.read(frame, "pid", lens)
        assert reading["stated"] is False and reading["reading"] in (None, "time_points"), reading
        assert not (reading["stated"] and reading["reading"] == "repeats")
        assert reading["trend"] and reading["trend"]["column"] == "sbp"
        assert reading["trend"]["direction"] == "falls"
        if not reading["stated"]:
            assert _gate(reading) is None  # the repeat-kind question is asked
    # Spacing alone, with nothing changing, is asked too; recalls under the dietary lens and
    # same-day records are still stated (6c).
    rng = np.random.default_rng(81)
    flat = _prepost(10, 4).assign(sbp=lambda f: 125 + rng.normal(0, 5, len(f)))
    reading = repeat_reading.read(flat, "pid", ["clinical"])
    assert reading["stated"] is False and reading["trend"] is None


def test_7d_wide_tables_that_are_not_assays_get_no_genomics_hint_and_no_contradiction():
    """Verifier, test 7: the genomics log-expression and scaled-CPM signatures accepted any wide
    table of floats under 25: a 130-item FFQ in servings a day was hinted "genomics … TMM- or
    median-of-ratios-scaled CPM", and under its own dietary lens drew a critical "The lens you chose
    and the table disagree"; a 300-ROI MRI thickness panel the same under the clinical lens.
    probes/api_ffq2.py (``default_rng(3)``) and api_mri.py (``default_rng(1)``), draw for draw.
    Expected: no genomics hint and no contradiction. Each FFQ sample's values sum to tens (NumPy),
    nowhere near the million a counts-per-million scale sums to."""
    rng = np.random.default_rng(3)
    n = 400
    ffq = pd.DataFrame({"participant_id": [f"P{i:04d}" for i in range(n)],
                        "age": rng.integers(40, 75, n), "sex": rng.choice(["F", "M"], n),
                        "hba1c": rng.normal(5.7, .6, n).round(1),
                        **{f"ffq_item_{i:03d}_serv_day": rng.gamma(0.8, 0.6, n).round(2)
                           for i in range(130)}})
    rng = np.random.default_rng(1)
    n = 200
    mri = pd.DataFrame({"subject": [f"S{i:03d}" for i in range(n)], "age": rng.integers(55, 85, n),
                        "sex": rng.choice(["F", "M"], n), "mmse": rng.integers(18, 31, n),
                        **{f"roi{i:03d}_thickness_mm": rng.normal(2.5, .3, n).round(3)
                           for i in range(300)}})
    totals = ffq.filter(like="ffq_item").sum(axis=1)
    assert totals.max() < 1_000
    for frame, lens in ((ffq, ["dietary"]), (mri, ["clinical"])):
        assert "genomics" not in [h["lens"] for h in lenses.hints(frame)]
        assert lenses.contradiction_finding(frame, lens) is None


@pytest.fixture(scope="module")
def gse147507():
    """GEO GSE147507 (Blanco-Melo et al. 2020; 78 human samples), the authors' raw read counts as
    published at https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147507/suppl/
    GSE147507_RawReadCounts_Human.tsv.gz (downloaded 2026-10-03, SHA-256 9a5db634…44a5); the
    verifier's seeded subset of 5,000 of its 21,797 genes (``default_rng(147507)``,
    ``choice(21797, 5000, replace=False)``, sorted), samples in rows. Its shallowest library holds
    14,694 reads."""
    raw = pd.read_csv(HERE / "GSE147507_RawReadCounts_Human_5000.tsv.gz", sep="\t", index_col=0)
    counts = raw.to_numpy(dtype=float).T
    return counts, [f"g_{g}" for g in raw.index]


def test_8d_the_genomics_card_holds_on_a_second_public_matrix(gse147507):
    """Verifier, test 8: the LOG_EXACT threshold was fitted on one matrix, and on GSE147507 voom's
    log-CPM read as "log2 TMM-scaled CPM" (medium, not asked), because one shallow library moves
    the back-transformed sum by 0.5 × genes / library size. Expected, on the closed-form scalings
    of the second matrix: raw counts, CPM, log2(CPM + 1) (offset 1), TMM-scaled CPM (rnanorm's TMM)
    and a binomially thinned copy read as before; voom's log-CPM exactly as limma writes it
    (``log2((counts+0.5)/(lib.size+1)*1e6)``), with ``lib.size`` the column sums or the column sums
    times the TMM factors, reads as voom-style log-CPM, its libraries recovered from the values."""
    from rnanorm import TMM

    counts, genes = gse147507
    lib = counts.sum(1, keepdims=True)
    assert lib.min() == 14_694
    genes_n = counts.shape[1]
    assert 0.5 * genes_n / lib.min() > 0.05  # the back-sum the old threshold read: > 5% off
    keys = lambda m: genomics.card(_frame(m, genes))["classification"]["keys"]  # noqa: E731
    assert keys(counts) == ["raw_counts"]
    assert keys(counts / lib * 1e6) == ["cpm_or_tpm"]
    log_cpm = genomics.card(_frame(np.log2(counts / lib * 1e6 + 1), genes))
    assert log_cpm["classification"]["keys"] == [genomics.LOG_CPM]
    assert log_cpm["classification"]["offset"] == 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = TMM().fit(counts).get_norm_factors(counts)
    assert np.isfinite(factors).all()
    tmm_cpm = counts / (lib * factors[:, None]) * 1e6
    assert keys(tmm_cpm) == ["tmm_scaled_cpm"]
    totals = tmm_cpm.sum(axis=1)
    assert genomics.SCALED_SUM_RANGE[0] <= totals.min() and totals.max() <= genomics.SCALED_SUM_RANGE[1]
    for size in (lib, lib * factors[:, None]):
        voom = np.log2((counts + 0.5) / (size + 1) * 1e6)
        card = genomics.card(_frame(voom, genes))
        c = card["classification"]
        assert c["keys"] == [genomics.LOG_CPM] and c["offset"] == 0.0
        assert c["label"] == "log2 CPM with a prior count (voom-style log-CPM)"
        assert c["confidence"] == "high"
        f = genomics.findings(_frame(voom, genes))[0]
        assert "voom" in f["detail"]
    shallow = np.random.default_rng(0).binomial(counts.astype(int), 0.01).astype(float)
    assert shallow.max() < 1e4
    assert keys(shallow) == ["raw_counts"]


def test_8e_the_voom_reading_is_exact_and_not_a_threshold(gse60450):
    """The voom reading recovers counts, so it says nothing on a log matrix that is not voom's:
    log2(TPM + 1) of GSE60450 keeps its offset-1 reading, and a log-normal matrix of the same
    shape (no counts behind it) is not read as voom."""
    counts, length, genes = gse60450
    rpk = counts / length[None, :] * 1e3
    tpm = rpk / rpk.sum(1, keepdims=True) * 1e6
    assert genomics.voom_reading(np.round(np.log2(tpm + 1), 4)) is None
    noise = np.random.default_rng(5).normal(5, 2, counts.shape)
    assert genomics.voom_reading(np.round(noise, 4)) is None
    lib = counts.sum(1, keepdims=True)
    voom = np.round(np.log2((counts + 0.5) / (lib + 1) * 1e6), 4)
    found = genomics.voom_reading(voom)
    assert found is not None and found["exact"]
    assert found["libraries"][0] == pytest.approx(lib.min(), rel=1e-3)
    assert found["libraries"][1] == pytest.approx(lib.max(), rel=1e-3)
