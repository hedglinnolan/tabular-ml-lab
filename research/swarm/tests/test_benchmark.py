"""The glucose project's benchmark (``research/swarm/benchmark.py``): Bang et al. 2009's points
score on a record's own held-out participants, and its AUC.

* **The score** by hand against Table 2's points on constructed rows, one item at a time: the age
  bands, male sex, family history, hypertension by each of its three routes, body size by BMI or
  by waist in inches at the paper's cut-offs for each sex, activity by either recreational item;
  refused, don't-know and blank answers are the condition absent, and so are blank measurements,
  as the printed rules say.
* **The AUC** per participant against scikit-learn's ``roc_auc_score``, ties included; weighted
  against a direct sum over every case–control pair and against scikit-learn's weighted AUC;
  DeLong's standard error against its placement values computed pair by pair; the jackknife's
  standard error against R's survey package (a JKn design, ``withReplicates``), with the
  participants a domain of a larger design.
* **The participants are the record's**: on constructed decision logs, the held-out rows are the
  rows the record's own rules keep, as pandas computes them from the rules; refused without an
  opening, with another cut-point or none, with a held-out count not the opening's, with a repair
  or a reading of text as numbers on a column the flow reads, a repair without its settings, or a
  landmark (each silent in its other case: a column the flow does not read, the settings given, a
  follow-up without a landmark); a consumers-only domain changes nothing under prediction; end to
  end on the glucose table, refused for another table, for a bundle that is not the corpus
  record's, for a held-out participant without a fasting weight or not of the test cycle.
* **Bang's reported AUC** is read from the notes' Table 3 row, and refused unless the row is there
  once; every passage the score's items quote is on its page; the activity item's adaptation is
  refused should the later cycle's file document PAQ520.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.swarm import benchmark as B  # noqa: E402
from research.swarm import intake as I  # noqa: E402

# ── the score, by hand against Table 2 ───────────────────────────────────────

BASE = {"RIDAGEYR": 30.0, "RIAGENDR": 2.0, "MCQ300C": 2.0, "BPQ020": 2.0, "BPQ050A": np.nan,
        **{f"BPXOSY{i}": 110.0 for i in range(1, 4)}, **{f"BPXODI{i}": 70.0 for i in range(1, 4)},
        "BMXBMI": 22.0, "BMXWAIST": 70.0, "PAQ650": 2.0, "PAQ665": 2.0}
# Table 2 (the notes' §5.2): age 40–49 1, 50–59 2, ≥ 60 3; male 1; family history 1; hypertension
# 1; overweight 1, obese 2, extremely obese 3; physically active −1. Each row: changes from BASE
# (a woman of 30, no history, BP 110/70, BMI 22, waist 70 cm, inactive) and its points by hand.
ROWS = [
    ({}, 0),
    ({"RIDAGEYR": 39.0}, 0), ({"RIDAGEYR": 40.0}, 1), ({"RIDAGEYR": 49.0}, 1),
    ({"RIDAGEYR": 50.0}, 2), ({"RIDAGEYR": 60.0}, 3), ({"RIDAGEYR": 80.0}, 3),
    ({"RIAGENDR": 1.0}, 1),
    ({"MCQ300C": 1.0}, 1), ({"MCQ300C": 7.0}, 0), ({"MCQ300C": 9.0}, 0), ({"MCQ300C": np.nan}, 0),
    ({"BPQ020": 1.0}, 1), ({"BPQ020": 9.0}, 0), ({"BPQ050A": 1.0}, 1),
    ({"BPXOSY1": 150.0, "BPXOSY2": 140.0, "BPXOSY3": 130.0}, 1),  # mean 140: "≥ 140"
    ({"BPXOSY1": 150.0, "BPXOSY2": 139.0, "BPXOSY3": 130.0}, 0),  # mean 139.67
    ({"BPXOSY1": 141.0, "BPXOSY2": np.nan, "BPXOSY3": np.nan}, 1),  # the one reading available
    ({"BPXODI1": 90.0, "BPXODI2": 90.0, "BPXODI3": 90.0}, 1), ({"BPXODI1": 89.0}, 0),
    ({"BMXBMI": 25.0}, 1), ({"BMXBMI": 29.9}, 1), ({"BMXBMI": 30.0}, 2), ({"BMXBMI": 40.0}, 3),
    ({"BMXBMI": np.nan, "BMXWAIST": np.nan}, 0),
    ({"BMXWAIST": 31.5 * 2.54}, 1), ({"BMXWAIST": 35 * 2.54}, 2), ({"BMXWAIST": 49 * 2.54}, 3),
    ({"RIAGENDR": 1.0, "BMXWAIST": 35 * 2.54}, 1 + 0),  # a man's waist of 35 in is not overweight
    ({"RIAGENDR": 1.0, "BMXWAIST": 37 * 2.54}, 1 + 1), ({"RIAGENDR": 1.0, "BMXWAIST": 40 * 2.54},
                                                       1 + 2),
    ({"RIAGENDR": 1.0, "BMXWAIST": 50 * 2.54}, 1 + 3),
    ({"BMXBMI": 26.0, "BMXWAIST": 49 * 2.54}, 3),  # the higher of the two categories
    ({"PAQ650": 1.0}, -1), ({"PAQ665": 1.0}, -1), ({"PAQ650": 1.0, "PAQ665": 1.0}, -1),
    ({"PAQ650": 9.0, "PAQ665": np.nan}, 0),
    ({"RIDAGEYR": 65.0, "RIAGENDR": 1.0, "MCQ300C": 1.0, "BPQ020": 1.0, "BMXBMI": 41.0}, 9),
    ({"RIDAGEYR": 45.0, "PAQ665": 1.0}, 0),
    # A blank measurement scores no point: no reading at all is not high, and a blank half leaves
    # the other to decide; with BMI blank, the waist decides (and the reverse).
    ({**{f"BPXO{m}{i}": np.nan for m in ("SY", "DI") for i in range(1, 4)}}, 0),
    ({**{f"BPXO{m}{i}": np.nan for m in ("SY", "DI") for i in range(1, 4)}, "BPQ020": 1.0}, 1),
    ({**{f"BPXOSY{i}": np.nan for i in range(1, 4)}, "BPXODI1": 95.0, "BPXODI2": np.nan,
      "BPXODI3": np.nan}, 1),  # no systolic reading; the one diastolic, 95, is high
    ({**{f"BPXODI{i}": np.nan for i in range(1, 4)}, "BPXOSY1": 145.0, "BPXOSY2": np.nan,
      "BPXOSY3": np.nan}, 1),  # no diastolic reading; the one systolic, 145, is high
    ({**{f"BPXOSY{i}": np.nan for i in range(1, 4)}}, 0),  # no systolic; diastolic 70 is not
    ({"BMXBMI": np.nan, "BMXWAIST": 35 * 2.54}, 2), ({"BMXBMI": 31.0, "BMXWAIST": np.nan}, 2),
]


def test_the_score_is_table_2s_points_item_by_item():
    frame = pd.DataFrame([{**BASE, **change} for change, _ in ROWS])
    np.testing.assert_array_equal(B.score(frame), [points for _, points in ROWS])
    assert B.POINTS == {f: p for f, _, _, _, p in I.BANG_TABLE2}
    assert sorted(set(B.POINTS.values())) == [-1, 1, 2, 3]
    # The printed rules say how a blank measurement is scored, as the rows above score it, and
    # cite the notes' open choices where the paper leaves it unsaid.
    rule = {i["item"]: i["rule"] for i in B.ITEMS}
    assert ("with both blank, not overweight or obese (no points)" in rule["body size"]
            and "§6.5" in rule["body size"])
    assert ("a measure with no reading, systolic or diastolic, is not high, so no reading at all "
            "scores no point" in rule["history of hypertension"]
            and "§6.2" in rule["history of hypertension"])
    notes = (REPO / I.BANG_NOTES).read_text(encoding="utf-8")
    choices = notes.split("## 6. Choices the paper leaves open")[1].split("\n## ")[0]
    assert "5. **Missing obesity inputs** (both BMI and waist missing)" in choices
    assert "2. **BP readings.** The mean of the available" in choices


# ── the AUC ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(20261022)
N = 400
S = RNG.integers(-1, 10, N).astype(float)  # many ties, as a points score has
Y = RNG.uniform(size=N) < 0.04 + 0.03 * (S + 1)
W = RNG.uniform(1_000, 60_000, N)


def pairwise(y, s, w):
    """The weighted AUC by its definition, every case–control pair summed."""
    cases, controls = np.flatnonzero(y), np.flatnonzero(~y)
    total = 0.0
    for i in cases:
        total += (w[i] * w[controls] * ((s[i] > s[controls]) + 0.5 * (s[i] == s[controls]))).sum()
    return total / (w[cases].sum() * w[controls].sum())


def test_the_auc_per_participant_and_weighted_against_scikit_learn_and_every_pair():
    from sklearn.metrics import roc_auc_score
    assert B.weighted_auc(Y, S, np.ones(N)) == pytest.approx(roc_auc_score(Y, S), abs=1e-12)
    assert B.weighted_auc(Y, S, W) == pytest.approx(pairwise(Y, S, W), abs=1e-12)
    assert B.weighted_auc(Y, S, W) == pytest.approx(roc_auc_score(Y, S, sample_weight=W), abs=1e-12)
    assert B.weighted_auc(Y, S, W) != pytest.approx(roc_auc_score(Y, S), abs=1e-3)
    assert np.isnan(B.weighted_auc(np.zeros(N, bool), S, W))


def test_delongs_standard_error_is_its_placement_values():
    """DeLong (1988): V10_i = the share of controls case i outscores (ties a half), V01_j = the
    share of cases that outscore control j; Var = s²(V10)/m + s²(V01)/n."""
    domain = np.ones(N, dtype=bool)
    out = B.aucs(Y, S, W, np.repeat([1, 2], N // 2), np.tile([1, 2], N // 2), domain)
    cases, controls = S[Y], S[~Y]
    v10 = np.array([((c > controls) + 0.5 * (c == controls)).mean() for c in cases])
    v01 = np.array([((cases > c) + 0.5 * (cases == c)).mean() for c in controls])
    se = np.sqrt(v10.var(ddof=1) / len(v10) + v01.var(ddof=1) / len(v01))
    per = out["per_participant"]
    assert per["estimate"] == pytest.approx(v10.mean(), abs=1e-12)
    assert per["se"] == pytest.approx(se, abs=1e-12) and per["method"] == "DeLong"
    assert per["ci_low"] == pytest.approx(per["estimate"] - 1.959963984540054 * se, abs=1e-9)


def test_the_jackknife_is_r_surveys_jkn_on_a_domain_of_a_larger_design(tmp_path):
    from turbotab.core.tests.acceptance.r_reference import RSCRIPT, run_r
    if RSCRIPT is None:
        pytest.skip("R (Rscript) is not installed")
    rng = np.random.default_rng(7)
    n = 300
    d = pd.DataFrame({"stratum": rng.integers(1, 8, n), "psu": rng.integers(1, 4, n),
                      "w": rng.uniform(500, 5000, n), "s": rng.integers(-1, 8, n)})
    d["dom"] = (rng.uniform(size=n) < 0.7).astype(int)
    d["y"] = (rng.uniform(size=n) < 0.1 + 0.05 * d["s"]).astype(int)
    theta, se, k = B.jackknife(d["y"].astype(bool), d["s"], d["w"], d["stratum"], d["psu"],
                               d["dom"].astype(bool))
    r = run_r("""
suppressPackageStartupMessages(library(survey))
d <- read.csv(rows_csv)
des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, data = d, nest = TRUE)
rep <- as.svrepdesign(des, type = "JKn", mse = TRUE)
wauc <- function(w, data) {
  w <- w * data$dom
  pos <- data$y == 1
  h <- outer(data$s[pos], data$s[!pos], function(a, b) (a > b) + 0.5 * (a == b))
  sum(outer(w[pos], w[!pos]) * h) / (sum(w[pos]) * sum(w[!pos]))
}
res <- withReplicates(rep, wauc)
out(list(theta = unname(coef(res)), se = unname(SE(res)), k = ncol(weights(rep))))
""", {"rows": d}, tmp_path)
    assert theta == pytest.approx(r["theta"], abs=1e-12)
    assert se == pytest.approx(r["se"], rel=1e-9) and k == r["k"]


def test_the_jackknife_is_refused_where_a_stratum_has_one_psu():
    strata, psus = np.array([1, 1, 2, 2]), np.array([1, 2, 1, 1])
    with pytest.raises(B.BenchmarkRefused, match="Stratum 2 has a single PSU"):
        B.jackknife(np.array([True, False, True, False]), np.arange(4.0), np.ones(4), strata,
                    psus, np.ones(4, dtype=bool))
    theta, se, k = B.jackknife(np.array([True, False, True, False]), np.array([3.0, 1, 2, 0]),
                               np.ones(4), strata, np.array([1, 2, 1, 2]), np.ones(4, dtype=bool))
    assert theta == 1.0 and k == 4


# ── Bang's reported AUC, and the items' passages ─────────────────────────────


def test_bangs_reported_auc_is_the_notes_validation_row():
    reported = B.reported_auc()
    assert (reported["auc"], reported["cycle"], reported["n"]) == (0.83, "2005–2006", 1640)
    row = "| Validation 2005–2006 (N = 1,640, **AUC 0.83**) | ≥ 6 |"
    assert B.reported_auc(f"x\n{row}\n")["auc"] == 0.83
    assert B.reported_auc(row.replace("0.83", "0.71"))["auc"] == 0.71
    for text in ("no row", f"{row}\n{row}"):
        with pytest.raises(B.BenchmarkRefused, match="validation rows with an AUC, not one"):
            B.reported_auc(text)


def item_pages() -> dict[str, str]:
    manifests = [I.load(i)["manifest"] for i in (I.GLUCOSE_ID, "bang2009")]
    pages = I.manifest_pages({"files": [e for m in manifests for e in m["files"]]})
    if B.PAQ_C not in pages or B._url("PAQ") not in pages:
        pytest.skip("the NHANES pages are not on disk (research/data/ is not tracked)")
    return pages


def test_every_passage_the_scores_items_quote_is_on_its_page_and_a_changed_one_is_named():
    pages = item_pages()
    assert B.item_problems(pages) == []
    changed = {**pages, B._url("MCQ"): pages[B._url("MCQ")].replace("close biological", "x")}
    found = B.item_problems(changed)
    assert len(found) == 1 and found[0].startswith("family history of diabetes:")
    assert "PAQ520" in I.codebook(I.NHANES / "2003-2004" / "PAQ_C.htm")
    assert "PAQ520" not in I.glucose_books("PAQ")[I.GLUCOSE_TEST]


def test_the_activity_item_is_adapted_only_while_the_later_cycles_file_lacks_paq520(monkeypatch):
    """Constructed pages holding every passage the items quote, and a later cycle's activity
    codebook without PAQ520 and then with it: silent, then the adaptation is refused."""
    pages: dict[str, str] = {}
    for item in B.ITEMS:
        for url, quote in item["passages"]:
            pages[url] = " ".join(f"{pages.get(url, '')} {quote}".split())
    book = {"PAQ650": None, "PAQ665": None}
    asked: list[str] = []

    def books(name, folder=None):
        asked.append(name)
        return {I.GLUCOSE_TEST: book}

    monkeypatch.setattr(I, "glucose_books", books)
    assert B.item_problems(pages) == [] and asked == ["PAQ"]
    book["PAQ520"] = None
    assert B.item_problems(pages) == [
        "PAQ520 is in the later cycle's activity file: the adaptation is not needed"]


# ── the participants: the record's own ───────────────────────────────────────


def rec(seq: int, decision) -> bytes:
    from turbotab.core import decisions as D
    return D.record_line(D.DecisionRecord(id=f"r{seq}", seq=seq, decision=decision,
                                          at=datetime(2026, 10, 22, tzinfo=timezone.utc)))


def log_of(*decisions) -> bytes:
    return b"".join(rec(i + 1, d) for i, d in enumerate(decisions))


def rules():
    from turbotab.core import decisions as D
    return [D.ExclusionRule(column="RIDAGEYR", low=20, reason="adults"),
            D.LevelRule(column="DIQ010", exclude=["1"], refused=["7"], dont_know=["9"],
                        reason="diagnosed diabetes"),
            D.LevelRule(column="DIQ050", exclude=["1"], refused=["7"], dont_know=["9"],
                        missing="keep", reason="insulin"),
            D.LevelRule(column="DIQ070", exclude=["1"], refused=["7"], dont_know=["9"],
                        missing="keep", reason="pills")]


def by_pandas(t: pd.DataFrame, at: float = 2017) -> np.ndarray:
    """The held-out rows the rules above keep, computed here from their words: a glucose; age 20
    or more; DIQ010 answered 2 or 3 (1 excluded; 7, 9 and blank unconfirmed); DIQ050 and DIQ070
    not 1 (their blanks, 7 and 9 kept); at or after the cut-point (2017)."""
    keep = (t["LBXGLU"].notna() & (t["RIDAGEYR"] >= 20) & t["DIQ010"].isin([2.0, 3.0])
            & (t["DIQ050"] != 1) & (t["DIQ070"] != 1) & (t[I.GLUCOSE_CYCLE] >= at))
    return np.flatnonzero(keep.to_numpy())


def log(n_holdout: int | None, *, cut_at: float = 2017, extra=(), opened: bool = True) -> bytes:
    from turbotab.core import decisions as D
    split = D.SetSplit(holdout=0.0, cut_point=D.CutPoint(column=I.GLUCOSE_CYCLE, at=cut_at),
                       fold_order="random")
    return log_of(D.SetPurpose(purpose="prediction"), D.SetTarget(column="LBXGLU"),
                  D.SetExclusions(rules=rules()), split, *extra,
                  *([D.OpenSeal(family="linear", n_holdout=n_holdout)] if opened else []))


CARD = {"outcome": {"column": "LBXGLU"}, "held_out": {"column": I.GLUCOSE_CYCLE, "at": 2017}}


def small_table() -> pd.DataFrame:
    """Every combination that matters, by construction: the cycles, ages, the three diabetes
    items' codes and blanks, a glucose or none."""
    import itertools
    rows = []
    for year, age, d10, d50, d70, glu in itertools.product(
            (2015, 2017), (19.0, 20.0), (1.0, 2.0, 3.0, 7.0, 9.0, np.nan), (1.0, 2.0, 9.0, np.nan),
            (1.0, 2.0, np.nan), (99.0, np.nan)):
        rows.append({I.GLUCOSE_CYCLE: year, "RIDAGEYR": age, "DIQ010": d10, "DIQ050": d50,
                     "DIQ070": d70, "LBXGLU": glu, "BMXWT": 70.0})
    return pd.DataFrame(rows)


def test_the_participants_are_the_rows_the_records_own_rules_keep():
    t = small_table()
    want = by_pandas(t)
    assert 0 < len(want) < len(t)
    got = B.participants(t, log(len(want)), CARD)
    np.testing.assert_array_equal(got.held_out, want)
    assert got.cut == {"column": I.GLUCOSE_CYCLE, "at": 2017.0}
    assert [s["key"] for s in got.steps][0] == "loaded"
    assert B.participants(t, log(None), CARD).held_out.tolist() == want.tolist()


def test_the_participants_are_refused_without_an_opening_or_with_another_cut_point():
    t = small_table()
    n = len(by_pandas(t))
    with pytest.raises(B.BenchmarkRefused, match="never opened its held-out rows"):
        B.participants(t, log(n, opened=False), CARD)
    with pytest.raises(B.BenchmarkRefused, match="a cut-point at 2015.0 on cycle_begin_year, not "
                                                 "the card's cut-point at 2017"):
        B.participants(t, log(n, cut_at=2015), CARD)
    from turbotab.core import decisions as D
    random = log_of(D.SetPurpose(purpose="prediction"), D.SetTarget(column="LBXGLU"),
                    D.SetSplit(holdout=0.2), D.OpenSeal(family="linear", n_holdout=n))
    with pytest.raises(B.BenchmarkRefused, match="by no cut-point"):
        B.participants(t, random, CARD)
    with pytest.raises(B.BenchmarkRefused, match="opening held out 3 rows, and its flow"):
        B.participants(t, log(3), CARD)
    other = log_of(D.SetPurpose(purpose="prediction"), D.SetTarget(column="BMXWT"),
                   D.OpenSeal(family="linear"))
    with pytest.raises(B.BenchmarkRefused, match="predicts 'BMXWT'"):
        B.participants(t, other, CARD)


def test_a_repair_on_a_column_the_flow_reads_is_refused_and_one_elsewhere_is_not():
    from turbotab.core import decisions as D
    t = small_table()
    n = len(by_pandas(t))
    on_rule = D.ApplyRepair(finding_id="sentinel_missing__DIQ010", option="set_missing",
                            params={"values": {"DIQ010": [7, 9]}})
    with pytest.raises(B.BenchmarkRefused, match="a repair rewrites DIQ010, which the flow reads"):
        B.participants(t, log(n, extra=[on_rule]), CARD)
    elsewhere = D.ApplyRepair(finding_id="sentinel_missing__BMXWT", option="set_missing",
                              params={"values": {"BMXWT": [7]}})
    assert len(B.participants(t, log(n, extra=[elsewhere]), CARD).held_out) == n
    joined = D.JoinFiles(file="f0123456789", on="SEQN", name="other.csv")
    with pytest.raises(B.BenchmarkRefused, match="joins another file to the table"):
        B.participants(t, log(n, extra=[joined]), CARD)


def test_a_repair_recorded_without_its_settings_is_refused_and_one_with_them_is_not():
    """The engine reads an applied repair's settings from the finding's offer when the log holds
    none (``repairs._params``); the bundle carries no findings, so the columns it rewrites are
    unknown. The same repair with its settings, on a column the flow does not read, stands."""
    from turbotab.core import decisions as D
    t = small_table()
    n = len(by_pandas(t))
    bare = D.ApplyRepair(finding_id="sentinel_missing__BMXWT", option="set_missing", params={})
    with pytest.raises(B.BenchmarkRefused, match="the repair of sentinel_missing__BMXWT is "
                                                 "recorded without its settings"):
        B.participants(t, log(n, extra=[bare]), CARD)
    full = D.ApplyRepair(finding_id="sentinel_missing__BMXWT", option="set_missing",
                         params={"values": {"BMXWT": [7]}})
    assert B.participants(t, log(n, extra=[full]), CARD).held_out.tolist() == by_pandas(t).tolist()


def test_a_reading_of_text_as_numbers_on_a_column_the_flow_reads_is_refused_and_elsewhere_not():
    """A text column read as numbers (``text_numbers``) is a repair whose values the working table
    rewrites, read by its settings like any other: refused on DIQ010, which a rule reads; silent
    on BMXWT, which nothing in the flow reads."""
    from turbotab.core import decisions as D
    t = small_table()
    n = len(by_pandas(t))

    def read(column):
        return D.ApplyRepair(finding_id=f"text_numbers__{column}", option="read_numbers",
                             params={"columns": {column: {"decimal": ".", "thousands": False}}})

    with pytest.raises(B.BenchmarkRefused, match="a repair rewrites DIQ010, which the flow reads"):
        B.participants(t, log(n, extra=[read("DIQ010")]), CARD)
    assert len(B.participants(t, log(n, extra=[read("BMXWT")]), CARD).held_out) == n


def test_a_landmark_is_refused_and_a_follow_up_without_one_is_not():
    """A landmark drops the rows whose follow-up ended by it, a step of the engine's flow the
    benchmark does not take: refused. The same follow-up without a landmark changes no row."""
    from turbotab.core import decisions as D
    t = small_table()
    n = len(by_pandas(t))
    task = D.SetTask(column="LBXGLU", task="time_to_event")
    marked = D.SetFollowUp(column="LBXGLU", time_column="RIDAGEYR", landmark=1.0)
    with pytest.raises(B.BenchmarkRefused, match="counts its follow-up from a landmark"):
        B.participants(t, log(n, extra=[task, marked]), CARD)
    plain = D.SetFollowUp(column="LBXGLU", time_column="RIDAGEYR", horizon=100.0)
    got = B.participants(t, log(n, extra=[task, plain]), CARD)
    assert got.held_out.tolist() == by_pandas(t).tolist()


def test_under_prediction_no_consumers_only_domain_restricts_the_rows():
    """Why the benchmark reads no domain: the engine's consumers-only domain restricts the rows
    under Estimate only (``methods.exposure_form.domain_columns``), and the benchmark's record is
    a prediction. A form recorded with that domain leaves the participants as they were."""
    from turbotab.core import decisions as D
    from turbotab.core.stages import rows as RW
    t = small_table()
    n = len(by_pandas(t))
    form = D.SetExposureForm(column="BMXWT", form="linear", domain="consumers")
    lines = log(n, extra=[form])
    from turbotab.core.export import replay
    state = D.fold([r for r, _ in replay.log_lines(lines)])
    assert state.form_domains == {"BMXWT": "consumers"} and RW.domain_of(state) == []
    assert B.participants(t, lines, CARD).held_out.tolist() == by_pandas(t).tolist()


# ── end to end on the glucose table ──────────────────────────────────────────


def glucose_table() -> pd.DataFrame:
    manifest = I.load(I.GLUCOSE_ID)["manifest"]
    path = I.REPO / manifest["table"]["file"]
    if not path.is_file() or I.sha256(path) != manifest["table"]["sha256"]:
        pytest.skip("the glucose table is not on disk (research/data/ is not tracked)")
    item_pages()
    return pd.read_parquet(path)


def bundle(tmp_path: Path, decisions: bytes, table_sha: str) -> Path:
    from turbotab.core.export.bundle import zip_bytes
    prov = {"decisions": {"sha256": hashlib.sha256(decisions).hexdigest()},
            "inputs": [{"role": "table", "name": "nhanes_glucose.parquet", "sha256": table_sha,
                        "path": str(I.GLUCOSE_TABLE)}]}
    path = tmp_path / "export.zip"
    path.write_bytes(zip_bytes({"decisions.jsonl": decisions,
                                "provenance.json": json.dumps(prov).encode()}))
    return path


def test_the_benchmark_end_to_end_on_the_glucose_table(tmp_path):
    from sklearn.metrics import roc_auc_score
    t = glucose_table()
    want = by_pandas(t)
    sha = I.load(I.GLUCOSE_ID)["manifest"]["table"]["sha256"]
    path = bundle(tmp_path, log(len(want)), sha)
    out = B.benchmark(path)
    assert out["participants"]["held_out"] == len(want) > 3000
    rows = t.iloc[want]
    y = rows["LBXGLU"].to_numpy() >= I.BANG_FPG
    s = B.score(rows)
    assert out["participants"]["cases"] == int(y.sum()) > 100
    per = out["auc"]["per_participant"]
    assert per["estimate"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert out["auc"]["weighted"]["estimate"] == pytest.approx(
        roc_auc_score(y, s, sample_weight=rows["WTSAFPRP"]), abs=1e-12)
    assert out["auc"]["weight"] == "WTSAFPRP"
    assert out["reported"]["auc"] == B.reported_auc()["auc"]
    w = out["auc"]["weighted"]
    assert w["ci_low"] < w["estimate"] < w["ci_high"] and "jackknife" in w["method"]
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"export": {"sha256": hashlib.sha256(path.read_bytes())
                                             .hexdigest()}}))
    assert B.benchmark(path, record)["participants"]["held_out"] == len(want)
    record.write_text(json.dumps({"export": {"sha256": "0" * 64}}))
    with pytest.raises(B.BenchmarkRefused, match="not the record's bundle"):
        B.benchmark(path, record)


def test_a_held_out_participant_without_a_fasting_weight_is_refused(tmp_path, monkeypatch):
    """The glucose table with one held-out participant's WTSAFPRP blanked, as the dataset's table:
    refused; the same table untouched passes (the end-to-end test)."""
    t = glucose_table()
    want = by_pandas(t)
    t.loc[t.index[want[0]], "WTSAFPRP"] = np.nan
    path = tmp_path / "nhanes_glucose.parquet"
    t.to_parquet(path, index=False)
    real = I.load

    def load(id_):
        files = real(id_)
        if id_ == I.GLUCOSE_ID:
            files["manifest"] = {**files["manifest"], "table": {
                **files["manifest"]["table"], "file": str(path), "sha256": I.sha256(path)}}
        return files

    monkeypatch.setattr(I, "load", load)
    with pytest.raises(B.BenchmarkRefused, match="has no fasting weight \\(WTSAFPRP\\)"):
        B.benchmark(bundle(tmp_path, log(len(want)), I.sha256(path)))


def test_a_held_out_participant_not_of_the_test_cycle_is_refused(tmp_path, monkeypatch):
    """A card whose cut-point (2015) lets 2015–2016's rows into the held-out set, and a record
    that holds out by it: refused, since the score is the test cycle's. The real card's cut-point
    (2017) passes (the end-to-end test)."""
    t = glucose_table()
    want = by_pandas(t, at=2015)
    assert (t[I.GLUCOSE_CYCLE].iloc[want] == 2015).any()
    real = I.load

    def load(id_):
        files = real(id_)
        if id_ == I.GLUCOSE_ID:
            files["card"] = {**files["card"], "held_out": {**files["card"]["held_out"],
                                                           "at": 2015}}
        return files

    monkeypatch.setattr(I, "load", load)
    sha = real(I.GLUCOSE_ID)["manifest"]["table"]["sha256"]
    with pytest.raises(B.BenchmarkRefused, match="A held-out participant is not of the test cycle"):
        B.benchmark(bundle(tmp_path, log(len(want), cut_at=2015), sha))


def test_the_benchmark_refuses_a_bundle_made_from_another_table(tmp_path, capsys):
    glucose_table()
    path = bundle(tmp_path, log(1), "f" * 64)
    with pytest.raises(B.BenchmarkRefused, match="does not match the record"):
        B.benchmark(path)
    assert B.main([str(path)]) == 2
    assert "does not match the record" in json.loads(capsys.readouterr().out)["refused"]
