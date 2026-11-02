"""The findings judge a column on the participants, not on the pooled-QC rows (the plasma panel's
package, findings-rows), and a two-level finding its blanks' answer settles still leaves which
level counts as 1 to the finding, a choice apart from the blanks' answer.

A pooled QC is one sample injected repeatedly: it carries no age, sex, BMI or outcome, so those are
blank on every QC row by design. Read over every row, the two-level reading sent `sex` (blank only
on the QC rows) to the missing-values question as "may mean not asked", and the left-censoring
reading named `age`, `bmi` and the outcome as non-detections. Here both read the rows the pooled-QC
reading does not name:

* the two-level reading's blank share counts over the rows outside the QC level the served pooled-QC
  finding names (``reference_rows.pooled_qc_finding``), the rows its exclusion keeps. The findings
  are read before any answer, so those rows are read from the data, whatever the answer to the QC
  finding; where the reading does not fire, every row counts as before;
* the left-censoring reading (``packs._left_censored``) reads the assay features only
  (``omics.candidate_columns``: not the outcome, a person's characteristic such as age, sex or
  BMI, an identifier or how the samples were acquired; and, as the omics_scale reading, not a
  column of two distinct values or fewer, counted on the rows it reads, so a 0/1 outcome before
  a target is chosen or a 0/1 covariate on no name list is no feature) on the rows outside those
  the served pooled-QC finding names, and names only features with a blank there; its detail
  opens with the count of those rows wherever any row is set aside. Read on the participants
  with the covariates among its columns, the covariates' complete values pulled its correlation
  off the features' own and could hold it above the gate where the features are censored, and
  their blanks, where they tracked abundance, raised it where the features' did not. The pack's own
  pooled-QC reading (``packs._pooled_qc``), whose rows it sets aside by default, reads a
  two-level label only; on a Case/Control/QC label the served reading names the QC rows and the
  pack's names none, so the app sets aside the served reading's rows, and the outcome
  (``finding_words.left_censoring_on_the_participants``).

Each truth is by construction on a plasma-like table (60 participants, 10 QC injections, 40
features censored below a detection limit, the features being the `mz_` columns it builds),
counted with pandas, or computed by scipy's Spearman correlation. The plasma fixture itself is
``research/swarm/tests/test_plasma_findings.py``'s.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab import packs
from turbotab.core import repairs, voice
from turbotab.core.decisions import (
    ApplyRepair,
    DecisionRecord,
    ProjectState,
    SetMissing,
    fold,
    validate,
)
from turbotab.core.quest import _finding_lines
from turbotab.core.stages.finding_words import SUMMARY_WORDS
from turbotab.core.stages.findings import findings_stage
from turbotab.core.tests.stage_harness import SAMPLES, Ingested

N_PART, N_QC, N_FEATURES = 60, 10, 40
QC_AT = list(range(0, N_PART + N_QC, 7))  # a QC every seventh injection, ten in all
LIMIT = 4.0  # a feature's log value below this is not detected: a blank


def panel(*, sex_blank: int = 0, qc_signal: bool = True, seed: int = 11,
          labels: tuple[str, ...] = ("participant",)) -> pd.DataFrame:
    """A plasma-like table by construction: 60 participants and 10 pooled-QC injections (every
    seventh row, ``sample_type`` = ``pooled_qc``; the participants' ``sample_type`` is drawn from
    ``labels``, ``case`` and ``control`` for a three-level label, by a generator of its own so the
    rest of the table is the same), 40 features whose log means run from 3.5 to 12,
    each blank below ``exp(4)`` (left-censored by abundance), and the participants' `age`, `bmi`,
    `sex` and `disease`, blank on every QC row. The first ``sex_blank`` participants have a blank
    `sex` too. With ``qc_signal`` the QC rows are one pooled sample (each feature's log mean, a
    technical SD of 0.05); without it they vary as the participants do (log SD 0.8), and no
    reading can tell them from participants."""
    rng = np.random.default_rng(seed)
    n = N_PART + N_QC
    qc = np.zeros(n, dtype=bool)
    qc[QC_AT] = True
    assert int(qc.sum()) == N_QC
    log_mean = np.linspace(3.5, 12.0, N_FEATURES)
    sd = np.where(qc[:, None] & qc_signal, 0.05, 0.8)
    log_value = log_mean + rng.normal(0.0, 1.0, (n, N_FEATURES)) * sd
    values = np.round(np.exp(log_value), 2)
    values[log_value < LIMIT] = np.nan
    sex = rng.choice(["F", "M"], n).astype(object)
    sex[qc] = None
    sex[np.flatnonzero(~qc)[:sex_blank]] = None
    label = np.random.default_rng(seed + 1).choice(list(labels), n)
    frame = {"sample_type": np.where(qc, "pooled_qc", label),
             "age": np.where(qc, np.nan, rng.integers(30, 75, n).astype(float)),
             "bmi": np.where(qc, np.nan, np.round(rng.normal(27.0, 4.0, n), 1)),
             "sex": sex}
    for j in range(N_FEATURES):
        frame[f"mz_{j + 1:03d}"] = values[:, j]
    out = pd.DataFrame(frame)
    out["disease"] = np.where(qc, np.nan, rng.integers(0, 2, n).astype(float))
    return out


def findings_of(tmp_path: Path, frame: pd.DataFrame, lens: list[str],
                target: str | None = None) -> dict[str, dict]:
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)
    table = Ingested(path, tmp_path / "ingested")
    artifact = table.run(findings_stage, ProjectState(lens=lens, target=target))
    return {f["id"]: f for f in artifact["findings"]}


def read_back(tmp_path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """The table as a CSV reader reads it (a blank text cell is a blank)."""
    path = tmp_path / "read.csv"
    frame.to_csv(path, index=False)
    return pd.read_csv(path)


def participants(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["sample_type"] != "pooled_qc"]


# ── the two-level reading: blanks counted over the participants ─────────────


def test_a_column_blank_only_on_the_qc_rows_reads_as_two_level_text(tmp_path):
    """`sex` is blank on the 10 QC rows and on no participant (by construction), so 0 of 60
    participants: the two-level reading, not the missing-values question. Over every row it was
    10 of 70 (14%), which routed it there."""
    frame = panel()
    df = read_back(tmp_path, frame)
    assert int(participants(df)["sex"].isna().sum()) == 0 and int(df["sex"].isna().sum()) == N_QC
    found = findings_of(tmp_path, frame, ["metabolomics"])
    assert "pack::metabolomics::pooled_qc" in found  # the reading that names the QC rows fired
    sex = found["binary_text__sex"]
    assert sex["routes_to"] is None
    assert sex["summary"].startswith("`sex` is two-level text (")
    assert sex["summary"].endswith("which level counts as 1 is not asked yet.")
    assert "blank" not in sex["summary"]


@pytest.mark.parametrize("blank", [6, 9])
def test_a_column_blank_on_a_tenth_of_the_participants_still_routes_to_missing(tmp_path, blank):
    """6 and 9 of the 60 participants blank (10% and 15%, by construction) route to the
    missing-values question, and the sentence counts over the 60 rows the pooled-QC finding's
    exclusion would keep, read from that exclusion's own column and level with pandas."""
    frame = panel(sex_blank=blank)
    df = read_back(tmp_path, frame)
    found = findings_of(tmp_path, frame, ["metabolomics"])
    exclusion = next(o for o in found["pack::metabolomics::pooled_qc"]["repairs"]
                     if o["key"] == "exclude_rows")["decision"]["params"]
    kept = df[~df[exclusion["column"]].astype(str).isin(exclusion["levels"])]
    assert len(kept) == N_PART
    n_blank = int(kept["sex"].isna().sum())
    assert n_blank == blank
    sex = found["binary_text__sex"]
    assert sex["routes_to"] == "missing"
    assert sex["summary"] == (f"`sex`, two-level text: `{n_blank}` of `{len(kept)}` non-QC rows "
                              f"blank (maybe not asked); which level counts as 1 is another "
                              f"choice.")
    assert voice.words(sex["summary"]) <= SUMMARY_WORDS


def test_under_a_tenth_of_the_participants_blank_is_the_two_level_reading(tmp_path):
    """5 of 60 participants (8%) is under the share that routes to the missing-values question,
    though with the QC rows it is 15 of 70 (21%), over it."""
    frame = panel(sex_blank=5)
    df = read_back(tmp_path, frame)
    assert int(df["sex"].isna().sum()) / len(df) >= 0.1 > 5 / N_PART
    sex = findings_of(tmp_path, frame, ["metabolomics"])["binary_text__sex"]
    assert sex["routes_to"] is None and "blank" not in sex["summary"]


@pytest.mark.parametrize("why", ["no_qc_signal", "no_assay_lens"])
def test_where_no_pooled_qc_reading_fires_every_row_counts(tmp_path, why):
    """The guard's silent side: where the QC rows vary as the participants do (no reading can
    tell them apart), or no assay lens reads them, no row is set aside, so `sex` blank on the 10
    QC rows is 10 of 70 rows, counted with pandas over every row, and routes as before."""
    frame = panel(qc_signal=why != "no_qc_signal")
    df = read_back(tmp_path, frame)
    found = findings_of(tmp_path, frame, ["clinical"] if why == "no_assay_lens" else ["metabolomics"])
    assert "pack::metabolomics::pooled_qc" not in found
    sex = found["binary_text__sex"]
    n_blank, n = int(df["sex"].isna().sum()), len(df)
    assert (n_blank, n) == (N_QC, N_PART + N_QC)
    assert sex["routes_to"] == "missing"
    assert sex["summary"] == (f"`sex`, two-level text: `{n_blank}` of `{n}` blank (maybe not "
                              f"asked); which level counts as 1 is another choice.")


# ── the left-censoring reading: the participants' features ──────────────────


def features(df: pd.DataFrame) -> list[str]:
    """The assay features, by construction: the `mz_` columns the tables here build."""
    return [c for c in df.columns if str(c).startswith("mz_")]


def _participant_reading(df: pd.DataFrame) -> tuple[set[str], float, int]:
    """By pandas and scipy: the features with a blank among the participants, and the Spearman
    correlation of each usable feature's missing rate with its log mean there."""
    part = participants(df)
    feats = features(df)
    rate = part[feats].isna().mean()
    mean = part[feats].mean()
    usable = [c for c in feats if pd.notna(mean[c]) and mean[c] > 0]
    rho = stats.spearmanr(rate[usable].to_numpy(), np.log(mean[usable].to_numpy())).statistic
    return {c for c in feats if rate[c] > 0}, float(rho), len(usable)


def test_left_censoring_names_only_columns_blank_among_the_participants(tmp_path):
    """`age`, `bmi` and the outcome are blank only on the QC rows: not non-detections, so not
    named. Every feature with a blank among the participants still is, and the count and the
    correlation are the 40 features' own on the participants (scipy). The pack's reading is handed
    the outcome here, as the app hands it (``left_censoring_on_the_participants``)."""
    df = read_back(tmp_path, panel())
    named, rho, n_usable = _participant_reading(df)
    assert named and n_usable == N_FEATURES
    found = packs._left_censored(df, target="disease")
    assert found is not None
    assert set(found["params"]["columns"]) == named
    assert not {"age", "bmi", "disease"} & set(found["params"]["columns"])
    assert found["params"]["n_with_blanks"] == len(named)
    assert found["params"]["n_features"] == N_FEATURES
    assert found["params"]["rho"] == round(rho, 3)
    # Served: the missing-values question reads these as the censored columns.
    served = findings_of(tmp_path, panel(), ["metabolomics"], target="disease")
    assert set(served["pack::metabolomics::left_censored"]["censored_columns"]) == named


def test_left_censoring_reads_every_row_where_the_pack_flags_no_qc_rows(tmp_path):
    """The silent side: the same table with QC rows no reading tells from participants is read
    over every row, as before, so a feature blank only on those rows (the most abundant one, made
    blank on the first QC row here) is named (pandas, every row). `age`, `bmi` and the outcome,
    blank on those rows too, are not features and are not named."""
    frame = panel(qc_signal=False)
    frame.loc[QC_AT[0], "mz_040"] = np.nan
    df = read_back(tmp_path, frame)
    assert packs._pooled_qc(df) is None
    every_row = {c for c in features(df) if df[c].isna().any()}
    assert "mz_040" in every_row - _participant_reading(df)[0]
    found = packs._left_censored(df, target="disease")
    assert found is not None and set(found["params"]["columns"]) == every_row


def _detail_read_on(rows: pd.DataFrame, feats: list[str] | None = None) -> str:
    """The left-censoring detail's sentence after its opening, its numbers by pandas and scipy on
    ``rows``: the usable features, the Spearman correlation of their missing rate with their log
    mean, the features with a blank and the highest missing rate among them. The features are
    ``feats``, by default the `mz_` columns."""
    feats = features(rows) if feats is None else feats
    rate, mean = rows[feats].isna().mean(), rows[feats].mean()
    usable = [c for c in feats if pd.notna(mean[c]) and mean[c] > 0]
    rho = stats.spearmanr(rate[usable].to_numpy(), np.log(mean[usable].to_numpy())).statistic
    blank = [c for c in feats if rate[c] > 0]
    return (f"{len(usable):,} features, a feature's missing rate tracks its abundance rank at a "
            f"rank correlation of {rho:.2f}. {len(blank):,} features have blanks; the highest "
            f"rate is {rate[blank].max():.0%}, on one of the least abundant.")


def test_the_left_censoring_detail_names_the_non_qc_rows_its_numbers_are_read_on(tmp_path):
    """With the 10 QC rows set aside (the pack's own reading flags them on this two-level label),
    the detail's numbers are the 60 participants' 40 features' (pandas, scipy: a correlation of
    -0.81) and it says so: on every row the highest rate differs, so a reader who checked them
    against the whole table would get other numbers. Read with `age`, `bmi` and the outcome
    among its columns, it said 43 features and -0.64. The pack's reading flags the same QC rows
    as the served one here, and the served detail is still read again, with the outcome set
    aside."""
    frame = panel()
    df = read_back(tmp_path, frame)
    assert packs._pooled_qc_rows(df).to_numpy().nonzero()[0].tolist() == QC_AT
    part = participants(df)
    assert len(part) == N_PART
    expected = f"On the {N_PART:,} non-QC rows, across {_detail_read_on(part)}"
    assert expected.startswith(f"On the 60 non-QC rows, across {N_FEATURES} features, a feature's "
                               f"missing rate tracks its abundance rank at a rank correlation of "
                               f"-0.81. ")
    assert _detail_read_on(part) != _detail_read_on(df)
    assert packs._left_censored(df, target="disease")["detail"] == expected
    served = findings_of(tmp_path, frame, ["metabolomics"], target="disease")
    assert served["pack::metabolomics::left_censored"]["detail"] == expected


def test_the_left_censoring_detail_read_on_every_row_names_no_rows(tmp_path):
    """The silent side: QC rows no reading tells from participants, so every row is read, the
    detail's numbers are every row's features' (pandas, scipy) and it names no rows."""
    frame = panel(qc_signal=False)
    df = read_back(tmp_path, frame)
    assert not packs._pooled_qc_rows(df).any()
    assert _detail_read_on(df) != _detail_read_on(participants(df))
    expected = f"Across {_detail_read_on(df)}"
    assert packs._left_censored(df, target="disease")["detail"] == expected
    served = findings_of(tmp_path, frame, ["metabolomics"], target="disease")
    assert served["pack::metabolomics::left_censored"]["detail"] == expected
    assert "non-QC" not in served["pack::metabolomics::left_censored"]["detail"]


def gate_table(*, censored: bool = True, seed: int = 1) -> pd.DataFrame:
    """The gate's table by construction: 30 features (the reading's floor) whose log means run
    from 3.5 to 6, 60 participants and 10 pooled QCs (every seventh row, a two-level label), and
    four covariates blank on every QC row and on no participant: `age`, `bmi`, `sex` (coded 1 and
    2) and the outcome `disease`. With ``censored`` a value is blank below exp(4), so a feature's
    blanks track its abundance; without it a tenth of the participants' cells are blank at random,
    whatever the feature's abundance."""
    rng = np.random.default_rng(seed)
    n, n_features = N_PART + N_QC, 30
    qc = np.zeros(n, dtype=bool)
    qc[QC_AT] = True
    log_mean = np.linspace(3.5, 6.0, n_features)
    log_value = log_mean + rng.normal(0.0, 1.0, (n, n_features)) * np.where(qc[:, None], 0.05, 0.8)
    values = np.round(np.exp(log_value), 2)
    if censored:
        values[log_value < LIMIT] = np.nan
    else:
        values[(rng.random((n, n_features)) < 0.1) & ~qc[:, None]] = np.nan
    frame = {"sample_type": np.where(qc, "pooled_qc", "participant"),
             "age": np.where(qc, np.nan, rng.integers(30, 75, n).astype(float)),
             "bmi": np.where(qc, np.nan, np.round(rng.normal(27.0, 4.0, n), 1)),
             "sex": np.where(qc, np.nan, rng.integers(1, 3, n).astype(float))}
    for j in range(n_features):
        frame[f"mz_{j + 1:03d}"] = values[:, j]
    out = pd.DataFrame(frame)
    out["disease"] = np.where(qc, np.nan, rng.integers(0, 2, n).astype(float))
    return out


def test_the_gate_reads_the_features_and_fires_on_their_censoring(tmp_path):
    """The gate's firing side: on the participants the 30 features' blanks track their abundance
    (scipy, at or below -0.5), so the finding is said, its numbers the features' own (pandas,
    scipy). Read with the four covariates among its columns, whose values are complete on the
    participants, the correlation was above -0.5 and no finding was said."""
    frame = gate_table()
    df = read_back(tmp_path, frame)
    named, rho, n_usable = _participant_reading(df)
    assert n_usable == 30 and len(named) >= 5 and rho <= -0.5
    expected = f"On the {N_PART:,} non-QC rows, across {_detail_read_on(participants(df))}"
    found = packs._left_censored(df, target="disease")
    assert found is not None and found["detail"] == expected
    assert set(found["params"]["columns"]) == named
    served = findings_of(tmp_path, frame, ["metabolomics"], target="disease")
    assert "pack::metabolomics::pooled_qc" in served
    censored = served["pack::metabolomics::left_censored"]
    assert censored["detail"] == expected and set(censored["censored_columns"]) == named


def test_the_gate_stays_silent_where_the_features_are_not_censored(tmp_path):
    """The gate's silent side: the same table with the participants' blanks placed at random, so
    a feature's missing rate does not track its abundance (scipy, above -0.5), though 30 features
    are read and 5 or more have blanks: no left-censoring finding, from the pack or served."""
    frame = gate_table(censored=False)
    df = read_back(tmp_path, frame)
    named, rho, n_usable = _participant_reading(df)
    assert n_usable == 30 and len(named) >= 5 and rho > -0.5
    assert packs._left_censored(df, target="disease") is None
    served = findings_of(tmp_path, frame, ["metabolomics"], target="disease")
    assert "pack::metabolomics::pooled_qc" in served
    assert "pack::metabolomics::left_censored" not in served


SET_ASIDE = ("age", "bmi", "sex", "visit", "day")  # the verifier's covariates (gate_silent_noblank)
MORE_TIMES = ("hour", "session")


def covariate_blank_table(*, feature_blanks: bool, seed: int = 3) -> pd.DataFrame:
    """The verifier's table by construction: 30 features (the reading's floor) whose log means run
    from 5 to 10, above every covariate's, complete; 60 participants and 10 pooled QCs (every
    seventh row, a two-level label); and the covariates omics sets aside by name (`age`, `bmi`,
    `sex` coded 0 and 1, `visit`, `day`), blank on every QC row and on 9 to 21 participants each
    (15% to 35%), drawn at random. With ``feature_blanks`` five features spread across the
    abundance range (every sixth) are blank on 3 participants each, drawn at random, so their
    blanks do not track abundance, and two more times omics sets aside by name, `hour` and
    `session`, are blank too, so that the covariates read among the features still carry the
    correlation past the gate (with the five alone, five such features took it to about -0.45)."""
    rng = np.random.default_rng(seed)
    n, n_features = N_PART + N_QC, 30
    qc = np.zeros(n, dtype=bool)
    qc[QC_AT] = True
    part = np.flatnonzero(~qc)
    log_mean = np.linspace(5.0, 10.0, n_features)
    log_value = log_mean + rng.normal(0.0, 1.0, (n, n_features)) * np.where(qc[:, None], 0.05, 0.8)
    values = np.round(np.exp(log_value), 2)
    if feature_blanks:
        for j in range(3, n_features, 6):
            values[rng.choice(part, 3, replace=False), j] = np.nan
    draw = {"age": rng.integers(30, 75, n), "bmi": np.round(rng.normal(27.0, 4.0, n), 1),
            "sex": rng.integers(0, 2, n), "visit": rng.integers(1, 5, n),
            "day": rng.integers(1, 31, n), "hour": rng.integers(0, 24, n),
            "session": rng.integers(1, 4, n)}
    covariates = SET_ASIDE + (MORE_TIMES if feature_blanks else ())
    frame: dict[str, object] = {"sample_type": np.where(qc, "pooled_qc", "participant")}
    for name, k in zip(covariates, np.linspace(9, 21, len(covariates)).astype(int)):
        v = draw[name].astype(float)
        v[qc] = np.nan
        v[rng.choice(part, k, replace=False)] = np.nan
        frame[name] = v
    for j in range(n_features):
        frame[f"mz_{j + 1:03d}"] = values[:, j]
    return pd.DataFrame(frame)


@pytest.mark.parametrize("feature_blanks", [False, True], ids=["features_complete",
                                                               "features_blank_at_random"])
def test_the_gate_stays_silent_where_only_covariates_omics_sets_aside_track_abundance(
        tmp_path, feature_blanks):
    """The features-only gate's silent side: the participants' blanks that track abundance are in
    the covariates omics sets aside by name (`age`, `bmi`, `sex`, `visit`, `day`, and `hour` and
    `session` where features have blanks), the least abundant columns. Read among the features,
    as every numeric column was read before the reading read omics' candidate columns, they
    carried the correlation to -0.5 or below (scipy) and the finding named them as
    non-detections. On the features alone it is silent, with no covariate named: where the
    features are complete none has a blank (the 5-blank floor); where five have blanks at random
    their correlation is above -0.5 (scipy), so the gate is what keeps it silent."""
    frame = covariate_blank_table(feature_blanks=feature_blanks)
    df = read_back(tmp_path, frame)
    part = participants(df)
    covariates = [c for c in df.columns if c != "sample_type" and c not in features(df)]
    every_numeric = covariates + features(df)
    rate, mean = part[every_numeric].isna().mean(), part[every_numeric].mean()
    assert (rate[covariates] > 0).all() and (mean > 0).all()
    assert stats.spearmanr(rate.to_numpy(), np.log(mean.to_numpy())).statistic <= -0.5
    blank_features = [c for c in features(df) if rate[c] > 0]
    if feature_blanks:
        named, rho, n_usable = _participant_reading(df)
        assert n_usable == 30 and len(named) == 5 and rho > -0.5
    else:
        assert not blank_features
    assert packs._left_censored(df) is None
    served = findings_of(tmp_path, frame, ["metabolomics"])
    assert "pack::metabolomics::pooled_qc" in served
    assert "pack::metabolomics::left_censored" not in served
    assert not {c for f in served.values() for c in f.get("censored_columns") or []} & set(covariates)


# ── the left-censoring reading: a two-valued column is no assay feature ─────


def test_before_an_outcome_is_chosen_a_two_valued_outcome_is_not_read_as_a_feature(tmp_path):
    """Before an outcome is chosen nothing names `disease` the outcome and no name rule sets it
    aside, but coded 0 and 1 it carries no abundance to order: it is set aside as the omics_scale
    reading sets aside a column of two distinct values or fewer. So the detail's numbers are the
    40 features' on the 60 participants (pandas, scipy: -0.81), from the pack handed no outcome
    (as ``packs.findings`` reads it) and served before a target. Read with the outcome among the
    features, it said 41 features and -0.75."""
    frame = panel()
    df = read_back(tmp_path, frame)
    part = participants(df)
    assert sorted(part["disease"].dropna().unique()) == [0.0, 1.0]
    expected = f"On the {N_PART:,} non-QC rows, across {_detail_read_on(part)}"
    assert expected.startswith(f"On the 60 non-QC rows, across {N_FEATURES} features, a feature's "
                               f"missing rate tracks its abundance rank at a rank correlation of "
                               f"-0.81. ")
    found = packs._left_censored(df)
    assert found is not None and found["detail"] == expected
    assert found["params"]["n_features"] == N_FEATURES
    served = findings_of(tmp_path, frame, ["metabolomics"])
    assert served["pack::metabolomics::left_censored"]["detail"] == expected


@pytest.mark.parametrize("target, on_the_qc_rows", [(None, None), ("disease", None),
                                                    ("disease", 9.0)])
def test_a_two_valued_covariate_on_no_name_list_is_neither_named_nor_counted(
        tmp_path, target, on_the_qc_rows):
    """`smoker`, coded 0 and 1 and on no name list, is blank on 3 participants: two values carry
    no abundance to order, so it is no feature and its blanks are no non-detections, before a
    target and with one. Its values are counted on the participants, the rows the reading reads:
    coded 9 on the QC rows, a third value only there, it is still two-valued. The finding's
    numbers and columns are the 40 features' on the participants (pandas, scipy), and the
    missing-values question is not handed `smoker` as censored."""
    frame = panel()
    qc = frame["sample_type"].eq("pooled_qc").to_numpy()
    smoker = np.random.default_rng(29).integers(0, 2, len(frame)).astype(float)
    smoker[qc] = np.nan if on_the_qc_rows is None else on_the_qc_rows
    smoker[np.flatnonzero(~qc)[:3]] = np.nan
    frame["smoker"] = smoker
    df = read_back(tmp_path, frame)
    part = participants(df)
    assert int(part["smoker"].isna().sum()) == 3
    assert sorted(part["smoker"].dropna().unique()) == [0.0, 1.0]
    named, _, n_usable = _participant_reading(df)
    assert n_usable == N_FEATURES
    expected = f"On the {N_PART:,} non-QC rows, across {_detail_read_on(part)}"
    found = packs._left_censored(df, target=target)
    assert found is not None and found["detail"] == expected
    assert set(found["params"]["columns"]) == named
    assert found["params"]["n_features"] == N_FEATURES
    served = findings_of(tmp_path, frame, ["metabolomics"], target=target)
    censored = served["pack::metabolomics::left_censored"]
    assert censored["detail"] == expected and set(censored["censored_columns"]) == named


@pytest.mark.parametrize("levels", [2, 3])
def test_a_feature_is_read_with_three_distinct_values_and_set_aside_with_two(tmp_path, levels):
    """The guard both ways at its edge: `mz_041`, the least abundant column, holds ``levels``
    distinct values among the participants (1, 2 and, with three, 3, by construction; the QC rows
    hold 2, one pooled sample) and is blank on 12 of them. With three it is read as every
    feature is, counted and named (pandas, scipy, over the 41 `mz_` columns); with two it is
    neither, and the numbers are the 40 others' own."""
    frame = panel()
    qc = frame["sample_type"].eq("pooled_qc").to_numpy()
    values = np.random.default_rng(31).integers(1, levels + 1, len(frame)).astype(float)
    values[qc] = 2.0
    values[np.flatnonzero(~qc)[:12]] = np.nan
    frame["mz_041"] = values
    df = read_back(tmp_path, frame)
    part = participants(df)
    assert part["mz_041"].nunique() == levels and int(part["mz_041"].isna().sum()) == 12
    feats = features(df) if levels == 3 else [c for c in features(df) if c != "mz_041"]
    found = packs._left_censored(df, target="disease")
    assert found is not None
    assert found["detail"] == f"On the {N_PART:,} non-QC rows, across {_detail_read_on(part, feats)}"
    assert found["params"]["n_features"] == len(feats) == N_FEATURES + levels - 2
    assert set(found["params"]["columns"]) == {c for c in feats if part[c].isna().any()}
    assert ("mz_041" in found["params"]["columns"]) == (levels == 3)


def test_left_censoring_sets_aside_the_qc_rows_of_a_case_control_qc_label(tmp_path):
    """The participants labeled `case` and `control` beside the `pooled_qc` level: the pack's own
    pooled-QC reading needs a two-level label and flags no row, while the served one (2 to 10
    levels) names the 10 QC rows by construction. The left-censoring finding sets aside those
    rows: `age`, `bmi` and the outcome, blank only there, are not named, and its counts and
    correlation are the participants' features' own (pandas, scipy)."""
    frame = panel(labels=("case", "control"))
    df = read_back(tmp_path, frame)
    assert set(df["sample_type"]) == {"case", "control", "pooled_qc"}
    assert packs._pooled_qc(df) is None  # two levels only: the pack's own reading flags no row
    named, rho, n_usable = _participant_reading(df)
    assert named and n_usable == N_FEATURES
    found = findings_of(tmp_path, frame, ["metabolomics"], target="disease")
    exclusion = next(o for o in found["pack::metabolomics::pooled_qc"]["repairs"]
                     if o["key"] == "exclude_rows")["decision"]["params"]
    at = np.flatnonzero(df[exclusion["column"]].astype(str).isin(exclusion["levels"]).to_numpy())
    assert at.tolist() == QC_AT
    censored = found["pack::metabolomics::left_censored"]
    assert set(censored["censored_columns"]) == named
    assert not {"age", "bmi", "disease"} & set(censored["censored_columns"])
    assert censored["detail"].startswith(
        f"On the {N_PART:,} non-QC rows, across {n_usable:,} features, a feature's missing rate "
        f"tracks its abundance rank at a rank correlation of {rho:.2f}. {len(named):,} features "
        f"have blanks;")
    on_the_participants = _detail_read_on(participants(df))
    assert censored["detail"] == f"On the {N_PART:,} non-QC rows, across {on_the_participants}"


def test_left_censoring_raised_only_by_the_qc_rows_blanks_is_not_said(tmp_path):
    """A Case/Control/QC table with no blank among the participants (by construction): only the
    QC rows are blank, on `age`, `bmi`, the outcome and the six least abundant features. Read
    over every row, as the pack's own reading reads this label, those six features' blanks track
    abundance among the 40 (scipy), so the pack names left censoring; on the participants there
    is no blank to read, and no left-censoring finding is said."""
    rng = np.random.default_rng(5)
    n = N_PART + N_QC
    qc = np.zeros(n, dtype=bool)
    qc[QC_AT] = True
    log_mean = np.linspace(3.5, 12.0, N_FEATURES)
    sd = np.where(qc[:, None], 0.05, 0.8)
    values = np.round(np.exp(log_mean + rng.normal(0.0, 1.0, (n, N_FEATURES)) * sd), 2)
    values[np.ix_(qc, range(6))] = np.nan
    frame = {"sample_type": np.where(qc, "pooled_qc", rng.choice(["case", "control"], n)),
             "age": np.where(qc, np.nan, rng.integers(30, 75, n).astype(float)),
             "bmi": np.where(qc, np.nan, np.round(rng.normal(27.0, 4.0, n), 1))}
    for j in range(N_FEATURES):
        frame[f"mz_{j + 1:03d}"] = values[:, j]
    frame["disease"] = np.where(qc, np.nan, rng.integers(0, 2, n).astype(float))
    df = read_back(tmp_path, pd.DataFrame(frame))
    feats = features(df)
    assert not participants(df)[feats + ["age", "bmi", "disease"]].isna().any().any()
    rate, mean = df[feats].isna().mean(), df[feats].mean()
    assert stats.spearmanr(rate.to_numpy(), np.log(mean.to_numpy())).statistic <= -0.5
    # The pack's own reading, the outcome handed to it: no QC row on three levels.
    every_row = packs._left_censored(df, target="disease")
    assert every_row is not None
    assert set(every_row["params"]["columns"]) == {f"mz_{j:03d}" for j in range(1, 7)}
    found = findings_of(tmp_path, pd.DataFrame(frame), ["metabolomics"], target="disease")
    assert "pack::metabolomics::pooled_qc" in found
    assert "pack::metabolomics::left_censored" not in found


def test_left_censoring_on_a_case_control_qc_label_reads_every_row_where_no_reading_fires(tmp_path):
    """The silent side: the same three-level label with QC rows that vary as the participants do,
    so no reading names them, and every row is read: a feature blank only on those rows (the most
    abundant one, made blank on the first QC row here) is named with the others (pandas, every
    row), and `age`, `bmi` and the outcome, blank on those rows too, are not features."""
    frame = panel(labels=("case", "control"), qc_signal=False)
    frame.loc[QC_AT[0], "mz_040"] = np.nan
    df = read_back(tmp_path, frame)
    found = findings_of(tmp_path, frame, ["metabolomics"], target="disease")
    assert "pack::metabolomics::pooled_qc" not in found
    every_row = {c for c in features(df) if df[c].isna().any()}
    assert "mz_040" in every_row - _participant_reading(df)[0]
    assert set(found["pack::metabolomics::left_censored"]["censored_columns"]) == every_row


# ── a finding its blanks' answer settles keeps its level to choose ──────────


def _record(seq: int, decision) -> DecisionRecord:
    return DecisionRecord(id=f"r{seq}", seq=seq, at=datetime.now(timezone.utc), decision=decision)


def test_the_blanks_answer_settles_the_finding_and_its_level_is_still_its_own(tmp_path):
    """`sex` blank on 12 of 70 rows (no assay lens, by construction) routes to the missing-values
    question, which settles the finding (``repairs.matching_answer``) and chooses no level. Its
    two level options stay on it, its line (the quest card's head) says which level counts as 1
    is another choice and nothing of where it is made (the app's card for an answered finding
    offers none), and one taken after the answer is recorded as the finding's own repair."""
    frame = panel(sex_blank=2, qc_signal=False)
    df = read_back(tmp_path, frame)
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)
    artifact = Ingested(path, tmp_path / "ingested").run(findings_stage, ProjectState(lens=[]))
    sex = next(f for f in artifact["findings"] if f["id"] == "binary_text__sex")
    assert int(df["sex"].isna().sum()) == 12
    assert sex["routes_to"] == "missing"
    assert sex["summary"] == ("`sex`, two-level text: `12` of `70` blank (maybe not asked); which "
                              "level counts as 1 is another choice.")
    assert [o["key"] for o in sex["repairs"]] == ["level", "level"]
    records = [_record(1, SetMissing(strategy="complete_case"))]
    state = fold(records)
    annotated = repairs.annotate(artifact, state, records)
    served = {f["id"]: f for f in annotated["findings"]}
    (line,) = [ln for _, ln in _finding_lines(state, annotated, {}, None) if ln.key == sex["id"]]
    assert (line.status, line.decision_id, line.name) == ("answered", "r1", sex["summary"])
    assert served[sex["id"]]["answered_by"] == "r1" and served[sex["id"]]["disposition"] is None
    assert [o["key"] for o in served[sex["id"]]["repairs"]] == ["level", "level"]
    level = served[sex["id"]]["repairs"][0]["decision"]
    ctx = {"artifact": lambda stage: artifact if stage == "findings" else None}
    taken = validate(ApplyRepair(**{k: v for k, v in level.items() if k != "kind"}), ctx)
    records.append(_record(2, taken))
    after = {f["id"]: f for f in repairs.annotate(artifact, fold(records), records)["findings"]}
    assert after[sex["id"]]["answered_by"] == "r2"
    assert after[sex["id"]]["disposition"]["action"] == "applied"
    assert after[sex["id"]]["disposition"]["params"] == level["params"]


def test_a_two_level_finding_with_no_level_to_choose_says_nothing_of_one(tmp_path):
    """True/False written as text offers no level (``repairs._offer_binary``: text levels only), so
    its routed sentence is as before, its counts by pandas."""
    df = pd.read_csv(SAMPLES / "binary_shapes.csv")
    table = Ingested(SAMPLES / "binary_shapes.csv", tmp_path / "shapes")
    found = {f["id"]: f for f in table.run(findings_stage, ProjectState(lens=["clinical"]))["findings"]}
    chol = found["binary_text__meds_chol"]
    assert chol["routes_to"] == "missing" and chol["repairs"] == []
    assert chol["summary"] == (f"`meds_chol` is two-level text with `{int(df['meds_chol'].isna().sum())}` "
                               f"of `{len(df)}` blank; a blank may mean not asked.")
