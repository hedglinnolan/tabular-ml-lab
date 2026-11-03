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
  column of two distinct values or fewer, counted on the rows it reads, where it is observed on
  more than two of them, so a 0/1 outcome before a target is chosen or a 0/1 covariate on no
  name list is no feature, while a feature detected on two participants or fewer is still read)
  on the rows outside those the served pooled-QC finding names, and names only features with a
  blank there; its detail opens with the count of those rows wherever any row is set aside.
  Read on the participants with the covariates among its columns, the covariates' complete
  values pulled its correlation off the features' own and could hold it above the gate where the
  features are censored, and their blanks, where they tracked abundance, raised it where the
  features' did not. The pack's own pooled-QC reading (``packs._pooled_qc``), whose rows it sets
  aside by default, reads a two-level label only; on a Case/Control/QC label the served reading
  names the QC rows and the pack's names none, so the app sets aside the served reading's rows,
  and the outcome (``finding_words.left_censoring_on_the_participants``).

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
from turbotab.core import detectors, repairs, voice
from turbotab.core.decisions import (
    ApplyRepair,
    ColumnUnitSpec,
    DecisionRecord,
    FindingDisposition,
    ProjectState,
    Refusal,
    SetMissing,
    fold,
    validate,
)
from turbotab.core.quest import _finding_lines
from turbotab.core.stages.finding_words import (
    SUMMARY_WORDS,
    FindingContext,
    left_censoring_on_the_participants,
    pooled_qc_rows,
)
from turbotab.core.stages.findings import findings_stage, pack_finding
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


def artifact_of(tmp_path: Path, frame: pd.DataFrame, lens: list[str],
                target: str | None = None) -> dict:
    """The findings stage's artifact on ``frame``, written and ingested as the app does."""
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)
    table = Ingested(path, tmp_path / "ingested")
    return table.run(findings_stage, ProjectState(lens=lens, target=target))


def findings_of(tmp_path: Path, frame: pd.DataFrame, lens: list[str],
                target: str | None = None) -> dict[str, dict]:
    return {f["id"]: f for f in artifact_of(tmp_path, frame, lens, target)["findings"]}


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
    reading sets aside a column of two distinct values or fewer (observed here on all 60
    participants, so its values repeat). So the detail's numbers are the 40 features' on the 60
    participants (pandas, scipy: -0.81), from the pack handed no outcome (as ``packs.findings``
    reads it) and served before a target. Read with the outcome among the features, it said 41
    features and -0.75."""
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


@pytest.mark.parametrize("detected, read", [((), True), ((55.0,), True), ((55.0, 61.0), True),
                                            ((55.0, 55.0, 61.0), False)],
                         ids=["on_no_participant", "on_one", "on_two_distinct",
                              "on_three_two_values"])
def test_a_feature_detected_on_two_participants_or_fewer_is_read_and_a_code_is_set_aside(
        tmp_path, detected, read):
    """The two-valued guard's edge on the rows observed. `mz_041` is detected on the first
    participants at ``detected`` (and at 58 on every QC row, one pooled sample) and blank on the
    other participants, by construction. On two participants or fewer it has no value to repeat:
    it is a feature detected rarely, the most censored kind, so it is read, counted where it has a
    mean and named with its blanks (pandas, scipy, over the 41 `mz_` columns), and the
    missing-values question is handed it as censored. On three participants with two values one
    repeats, as a code's do: it is set aside, and the numbers are the 40 others' own."""
    frame = panel()
    qc = frame["sample_type"].eq("pooled_qc").to_numpy()
    values = np.where(qc, 58.0, np.nan)
    values[np.flatnonzero(~qc)[:len(detected)]] = detected
    frame["mz_041"] = values
    df = read_back(tmp_path, frame)
    part = participants(df)
    assert part["mz_041"].dropna().tolist() == list(detected)
    assert int(part["mz_041"].isna().sum()) == N_PART - len(detected)
    feats = features(df) if read else [c for c in features(df) if c != "mz_041"]
    expected = f"On the {N_PART:,} non-QC rows, across {_detail_read_on(part, feats)}"
    found = packs._left_censored(df, target="disease")
    assert found is not None and found["detail"] == expected
    assert set(found["params"]["columns"]) == {c for c in feats if part[c].isna().any()}
    assert ("mz_041" in found["params"]["columns"]) == read
    served = findings_of(tmp_path, frame, ["metabolomics"], target="disease")
    censored = served["pack::metabolomics::left_censored"]["censored_columns"]
    assert ("mz_041" in censored) == read


SPARSE = tuple(f"mz_{j:03d}" for j in range(31, 37))


def sparse_table(seed: int = 3) -> pd.DataFrame:
    """A table whose only censoring is in features detected rarely, by construction: 30 complete
    features whose log means run from 5 to 10, and six more (``SPARSE``) detected on one or two
    participants each, alternately, at distinct values from 50 to 61 and blank on every other
    row; 60 participants and 10 pooled QCs (every seventh row, a two-level label), and the
    outcome `disease`, coded 0 and 1 and blank on the QC rows."""
    rng = np.random.default_rng(seed)
    n, n_complete = N_PART + N_QC, 30
    qc = np.zeros(n, dtype=bool)
    qc[QC_AT] = True
    part = np.flatnonzero(~qc)
    log_mean = np.linspace(5.0, 10.0, n_complete)
    log_value = log_mean + rng.normal(0.0, 1.0, (n, n_complete)) * np.where(qc[:, None], 0.05, 0.8)
    frame: dict[str, object] = {"sample_type": np.where(qc, "pooled_qc", "participant")}
    for j in range(n_complete):
        frame[f"mz_{j + 1:03d}"] = np.round(np.exp(log_value[:, j]), 2)
    for j, name in enumerate(SPARSE):
        k = 1 + j % 2
        v = np.full(n, np.nan)
        v[part[2 * j:2 * j + k]] = 50.0 + 2 * j + np.arange(k)
        frame[name] = v
    out = pd.DataFrame(frame)
    out["disease"] = np.where(qc, np.nan, rng.integers(0, 2, n).astype(float))
    return out


def test_features_detected_on_one_or_two_participants_are_non_detections_no_fill_reads_at_random(
        tmp_path):
    """The six features of ``sparse_table`` are detected on one or two participants each, at
    distinct values below every complete feature's mean (pandas): their blanks are the table's
    censoring, and the reading fires on them alone (scipy over the 36 features, at or below
    -0.5), before an outcome is chosen and with one, naming the six. The missing-values question
    reads them as non-detections: a median fill is refused, each way out names the six, and
    under inference multiple imputation is refused. Set aside as two-valued, they left the
    reading silent and both fills were accepted."""
    frame = sparse_table()
    df = read_back(tmp_path, frame)
    part = participants(df)
    sparse = list(SPARSE)
    observed = part[sparse].notna().sum()
    assert observed.tolist() == [1, 2, 1, 2, 1, 2]
    assert (part[sparse].nunique() == observed).all()
    complete = [c for c in features(df) if c not in SPARSE]
    assert not part[complete].isna().any().any()
    assert part[sparse].mean().max() < part[complete].mean().min()
    named, rho, n_usable = _participant_reading(df)
    assert named == set(sparse) and n_usable == 36 and rho <= -0.5
    expected = f"On the {N_PART:,} non-QC rows, across {_detail_read_on(part)}"
    for target in (None, "disease"):
        found = packs._left_censored(df, target=target)
        assert found is not None and found["detail"] == expected
        assert set(found["params"]["columns"]) == set(sparse)
    artifact = artifact_of(tmp_path, frame, ["metabolomics"], target="disease")
    (censored,) = [f for f in artifact["findings"]
                   if f["id"] == "pack::metabolomics::left_censored"]
    assert censored["detail"] == expected and set(censored["censored_columns"]) == set(sparse)
    for purpose, strategy in (("prediction", "impute"), ("inference", "multiple_imputation")):
        ctx = {"artifact": lambda stage: artifact if stage == "findings" else None,
               "state": ProjectState(lens=["metabolomics"], target="disease", purpose=purpose)}
        with pytest.raises(Refusal) as refused:
            validate(SetMissing(strategy=strategy), ctx)
        assert refused.value.code == "median_below_detection"
        ways = [e["decision"] for e in refused.value.exits if e["decision"] is not None]
        assert ways and all(set(d["censored_columns"]) == set(sparse) for d in ways)
        assert ("never filled as missing at random" in refused.value.message) == (
            purpose == "inference")


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


def qc_only_blanks_table() -> pd.DataFrame:
    """A Case/Control/QC table with no blank among the participants, by construction: only the 10
    QC rows are blank, on `age`, `bmi`, the outcome and the six least abundant features."""
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
    return pd.DataFrame(frame)


def test_left_censoring_raised_only_by_the_qc_rows_blanks_is_not_said(tmp_path):
    """A Case/Control/QC table with no blank among the participants (by construction): only the
    QC rows are blank, on `age`, `bmi`, the outcome and the six least abundant features. Read
    over every row, as the pack's own reading reads this label, those six features' blanks track
    abundance among the 40 (scipy), so the pack names left censoring; on the participants there
    is no blank to read, and no left-censoring finding is said."""
    frame = qc_only_blanks_table()
    df = read_back(tmp_path, frame)
    feats = features(df)
    assert not participants(df)[feats + ["age", "bmi", "disease"]].isna().any().any()
    rate, mean = df[feats].isna().mean(), df[feats].mean()
    assert stats.spearmanr(rate.to_numpy(), np.log(mean.to_numpy())).statistic <= -0.5
    # The pack's own reading, the outcome handed to it: no QC row on three levels.
    every_row = packs._left_censored(df, target="disease")
    assert every_row is not None
    assert set(every_row["params"]["columns"]) == {f"mz_{j:03d}" for j in range(1, 7)}
    found = findings_of(tmp_path, frame, ["metabolomics"], target="disease")
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


# ── the pack findings read the participants, and the outcome as the outcome ──


def test_the_pack_findings_read_left_censoring_on_the_participants_of_a_case_control_qc_label(
        tmp_path):
    """``detectors.pack_findings``, the reading the findings stage speaks for and the lens preview
    counts, reads the left-censoring finding on the rows outside those the served pooled-QC
    finding names. On a Case/Control/QC label whose QC rows alone are blank it raises none, where
    read over every row (the pack's own pooled-QC reading flags no row of a three-level label) it
    raised one; on the censored panel under that label its detail is the 60 participants' 40
    features' own (pandas, scipy)."""
    quiet = read_back(tmp_path, qc_only_blanks_table())
    assert packs._pooled_qc(quiet) is None
    assert packs._left_censored(quiet, target="disease") is not None  # every row: the QC rows' blanks
    ids = [f["id"] for f in detectors.pack_findings(quiet, ["metabolomics"])]
    assert "pack::metabolomics::pooled_qc" in ids
    assert "pack::metabolomics::left_censored" not in ids

    df = read_back(tmp_path, panel(labels=("case", "control")))
    (found,) = [f for f in detectors.pack_findings(df, ["metabolomics"])
                if f["id"] == "pack::metabolomics::left_censored"]
    assert found["detail"] == f"On the {N_PART:,} non-QC rows, across {_detail_read_on(participants(df))}"


def staged_panel() -> pd.DataFrame:
    """``panel`` with `stage`, an outcome coded 0, 1 and 2 (three values, so the two-valued rule
    does not set it aside) on no name list, blank on the QC rows and on the first 8 participants,
    by construction."""
    frame = panel()
    qc = frame["sample_type"].eq("pooled_qc").to_numpy()
    stage = np.random.default_rng(37).integers(0, 3, len(frame)).astype(float)
    stage[qc] = np.nan
    stage[np.flatnonzero(~qc)[:8]] = np.nan
    frame["stage"] = stage
    return frame


@pytest.mark.parametrize("reader", ["packs.findings", "detectors.pack_findings"])
def test_an_outcome_of_three_values_handed_to_the_pack_findings_is_not_read_as_a_feature(
        tmp_path, reader):
    """`stage`, the outcome, holds three values among the participants and no name rule sets it
    aside, so with no outcome named the reading cannot tell it from a feature: its 8 participants'
    blanks are named non-detections and the numbers are the 41 columns' (pandas, scipy), as
    before. Handed the outcome, the legacy pack findings (``packs.findings``, which the legacy
    interview reads) and the served ones (``detectors.pack_findings``) set it aside: the numbers
    are the 40 features' own and `stage` is not named."""
    df = read_back(tmp_path, staged_panel())
    part = participants(df)
    assert sorted(part["stage"].dropna().unique()) == [0.0, 1.0, 2.0]
    assert int(part["stage"].isna().sum()) == 8
    read = packs.findings if reader == "packs.findings" else detectors.pack_findings

    def censored(target):
        (found,) = [f for f in read(df, ["metabolomics"], target=target)
                    if f["id"] == "pack::metabolomics::left_censored"]
        return found

    unnamed = censored(None)
    assert unnamed["detail"] == (f"On the {N_PART:,} non-QC rows, across "
                                 f"{_detail_read_on(part, features(df) + ['stage'])}")
    assert "stage" in unnamed["params"]["columns"]
    named = censored("stage")
    assert named["detail"] == f"On the {N_PART:,} non-QC rows, across {_detail_read_on(part)}"
    assert "stage" not in named["params"]["columns"]
    assert named["params"]["n_features"] == N_FEATURES


def test_the_derived_missingness_prior_is_not_granted_on_an_outcome_handed_to_it(tmp_path):
    """``packs.prior_for_column`` (the column-scoped prior, GUIDED-027) applies the metabolomics
    pack's derived prior to the columns the left-censoring reading names. Handed the outcome,
    `stage` is not among them, so no prior reads its blanks as non-detections, while a feature
    blank among the participants keeps it; with no outcome named, `stage` gets it, as before."""
    df = read_back(tmp_path, staged_panel())
    lens = [packs.METABOLOMICS]
    feature = next(c for c in features(df) if participants(df)[c].isna().any())

    def packs_of(column, target):
        return [p["pack"] for p in packs.prior_for_column(lens, "missingness_direction", column, df,
                                                          target=target)]

    assert packs_of("stage", None) == [packs.METABOLOMICS]
    assert packs_of("stage", "stage") == []
    assert packs_of(feature, "stage") == packs_of(feature, None) == [packs.METABOLOMICS]
    (scoped,) = packs.priors(lens, "missingness_direction", df, target="stage")
    assert "stage" not in scoped["columns"] and feature in scoped["columns"]


def test_left_censoring_the_pack_list_lacks_is_inserted_with_the_participants_numbers(tmp_path):
    """The insert path of ``left_censoring_on_the_participants``: a Case/Control/QC label whose QC
    rows are blank on the six most abundant features (by construction), so read over every row
    the 40 features' blanks do not track abundance past the gate (scipy, above -0.5) and the
    pack's own findings (``packs.findings``) hold no left-censoring finding, while on the
    participants they do (scipy, -0.81). Handed a pack list with the served pooled-QC finding and
    no left-censoring one, the re-read adds it after them, with the participants' numbers; and
    the findings stage serves it so."""
    frame = panel(labels=("case", "control"))
    qc = frame["sample_type"].eq("pooled_qc").to_numpy()
    for j in range(N_FEATURES - 6, N_FEATURES):
        frame.loc[qc, f"mz_{j + 1:03d}"] = np.nan
    df = read_back(tmp_path, frame)
    feats = features(df)
    every_row = stats.spearmanr(df[feats].isna().mean(), np.log(df[feats].mean())).statistic
    assert every_row > -0.5
    assert "pack::metabolomics::left_censored" not in {
        f["id"] for f in packs.findings(df, ["metabolomics"])}
    expected = f"On the {N_PART:,} non-QC rows, across {_detail_read_on(participants(df))}"
    assert "rank correlation of -0.81." in expected
    qc_raw = next(f for f in detectors.pack_findings(df, ["metabolomics"])
                  if f["id"] == "pack::metabolomics::pooled_qc")
    legacy = [(qc_raw, pack_finding(qc_raw))]
    fc = FindingContext(frame=df, lens=("metabolomics",), target="disease",
                        reference_rows=pooled_qc_rows(df, [qc_raw]))
    assert sorted(fc.reference_rows) == QC_AT
    out = left_censoring_on_the_participants(legacy, fc)
    assert [f["id"] for _, f in out] == ["pack::metabolomics::pooled_qc",
                                         "pack::metabolomics::left_censored"]
    assert out[1][0]["detail"] == expected
    served = findings_of(tmp_path, frame, ["metabolomics"], target="disease")
    assert served["pack::metabolomics::left_censored"]["detail"] == expected


# ── the lens contradiction reads what the left-censoring finding reads ──────


#: Covariates omics sets aside by name (age and BMI under eight names, age in months among them).
COVARIATES = {"age": (30, 75), "bmi": (20, 35), "age_at_enrollment": (30, 75),
              "bmi_baseline": (20, 35), "RIDAGEMN": (360, 900), "age_at_draw": (30, 75),
              "bmi_at_draw": (20, 35), "age_years": (30, 75)}


def low_scale_panel(*, censored: bool, seed: int = 1) -> pd.DataFrame:
    """60 participants and 10 pooled QCs (every seventh row, a two-level label), 40 features whose
    log means run from -2 to 3 (values of about 0.1 to 20), the eight :data:`COVARIATES`, each
    more abundant than any feature and blank on every QC row, and the 0/1 outcome `disease`,
    blank there too, by construction. With ``censored`` the 12 least abundant features are blank
    below a detection limit; without it 12 features spread over the abundance range (every third)
    are blank at random on about a quarter of the participants."""
    rng = np.random.default_rng(seed)
    n = N_PART + N_QC
    qc = np.zeros(n, dtype=bool)
    qc[QC_AT] = True
    log_mean = np.linspace(-2.0, 3.0, N_FEATURES)
    log_value = log_mean + rng.normal(0.0, 1.0, (n, N_FEATURES)) * np.where(qc[:, None], 0.05, 0.8)
    values = np.round(np.exp(log_value), 3)
    if censored:
        below = log_value < log_mean[11] + 0.3
        values[below & (np.arange(N_FEATURES)[None, :] < 12)] = np.nan
    else:
        at_random = (rng.random((n, N_FEATURES)) < 0.25) & ~qc[:, None]
        values[at_random & (np.arange(N_FEATURES)[None, :] % 3 == 0)] = np.nan
    frame: dict[str, object] = {"sample_type": np.where(qc, "pooled_qc", "participant")}
    for j in range(N_FEATURES):
        frame[f"mz_{j + 1:03d}"] = values[:, j]
    for name, (lo, hi) in COVARIATES.items():
        frame[name] = np.where(qc, np.nan, np.round(rng.uniform(lo, hi, n), 1))
    out = pd.DataFrame(frame)
    out["disease"] = np.where(qc, np.nan, rng.integers(0, 2, n).astype(float))
    return out


def _participants_rho(df: pd.DataFrame, feats: list[str] | None = None) -> tuple[float, int, int]:
    """By scipy and pandas, on the participants: the Spearman correlation of each usable feature's
    missing rate with its log mean, the usable features and the features with a blank."""
    part = participants(df)
    feats = features(df) if feats is None else feats
    rate, mean = part[feats].isna().mean(), part[feats].mean()
    usable = [c for c in feats if pd.notna(mean[c]) and mean[c] > 0]
    rho = stats.spearmanr(rate[usable].to_numpy(), np.log(mean[usable].to_numpy())).statistic
    return float(rho), len(usable), int((rate > 0).sum())


def test_where_the_participants_features_are_censored_the_lens_is_not_contradicted(tmp_path):
    """The participants' 12 censored features track abundance (scipy: -0.65 over the 40, with 12
    blank), so the left-censoring finding is raised; read over every numeric column and every row,
    the eight covariates blank on the QC rows, more abundant than any feature, pulled the lens
    contradiction's reading above -0.3 (scipy), and the app said both that the blanks cluster in
    the lowest-abundance features and that they do not look like non-detections. The
    contradiction reads the same features on the same rows now, and says nothing."""
    df = read_back(tmp_path, low_scale_panel(censored=True))
    rho, n_usable, n_blank = _participants_rho(df)
    assert rho <= -0.5 and n_blank == 12 and n_usable == N_FEATURES
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rate, mean = df[numeric].isna().mean(), df[numeric].mean()
    assert stats.spearmanr(rate.to_numpy(), np.log(mean.to_numpy())).statistic > -0.3
    assert packs.contradiction(df, [packs.METABOLOMICS]) is None
    found = findings_of(tmp_path, low_scale_panel(censored=True), ["metabolomics"])
    assert "pack::metabolomics::left_censored" in found
    assert "voice::lens_contradiction" not in found


def test_where_the_participants_blanks_are_at_random_the_lens_is_contradicted_on_their_features(
        tmp_path):
    """The guard's other side: 12 features blank at random on the participants (scipy: their
    missing rate does not track abundance, above -0.3), so the lens is contradicted, and the
    sentence's count and correlation are the 40 features' on the 60 participants (scipy), not
    every numeric column's over every row. Handed a three-valued outcome, `stage`, the reading
    sets it aside; with no outcome named it is one more column, as before."""
    frame = low_scale_panel(censored=False)
    qc = frame["sample_type"].eq("pooled_qc").to_numpy()
    stage = np.random.default_rng(41).integers(0, 3, len(frame)).astype(float)
    stage[qc] = np.nan
    stage[np.flatnonzero(~qc)[:20]] = np.nan
    frame["stage"] = stage
    df = read_back(tmp_path, frame)
    for target, feats in ((None, features(df) + ["stage"]), ("stage", features(df))):
        rho, n_usable, n_blank = _participants_rho(df, feats)
        assert rho > -0.3 and n_blank >= 10
        clash = packs.contradiction(df, [packs.METABOLOMICS], target=target)
        assert clash is not None and clash["kind"] == "stated_assay_lens_but_blanks_are_not_censored"
        assert clash["rho"] == round(rho, 3)
        assert (f"across {n_usable:,} measurement columns the missing rate does not track "
                f"abundance (rank correlation {rho:+.2f}") in clash["message"]
    assert n_usable == N_FEATURES
    served = findings_of(tmp_path, frame, ["metabolomics"])["voice::lens_contradiction"]
    rho, n_usable, _ = _participants_rho(df, features(df) + ["stage"])
    assert f"across {n_usable:,} measurement columns" in served["detail"]


# ── the readings beside the pooled QCs count the participants ───────────────


def statin_panel(blank: int, qc_answer: str | None = None) -> pd.DataFrame:
    """``panel`` with `statin`, answered `Yes` or `No` (which the app reads as true and false),
    blank on the first ``blank`` participants and, unless ``qc_answer`` fills them, on the 10 QC
    rows, by construction."""
    frame = panel()
    qc = frame["sample_type"].eq("pooled_qc").to_numpy()
    statin = np.random.default_rng(43).choice(["Yes", "No"], len(frame)).astype(object)
    statin[qc] = qc_answer
    statin[np.flatnonzero(~qc)[:blank]] = None
    frame["statin"] = statin
    return frame


def _not_asked(found: dict) -> list[dict]:
    return [o for o in found["binary_text__statin"]["repairs"] if o["key"] == "not_asked_no"]


def test_blanks_on_the_qc_rows_do_not_make_a_question_read_as_not_asked(tmp_path):
    """`statin` is blank on 25 of the 60 participants (42%, under the not-asked share of half) and
    on the 10 QC rows, 35 of 70 rows over every row (half): a QC injection is asked nothing, so its
    blanks are no skip pattern, and "Not asked as No" is not offered; read over every row it was."""
    df = read_back(tmp_path, statin_panel(25))
    part = participants(df)
    assert int(part["statin"].isna().sum()) / len(part) < 0.5 <= int(df["statin"].isna().sum()) / len(df)
    assert _not_asked(findings_of(tmp_path, statin_panel(25), ["metabolomics"])) == []


@pytest.mark.parametrize("lens, qc_answer", [(["metabolomics"], None), (["metabolomics"], "No"),
                                             (["clinical"], None)],
                         ids=["qc_blank", "qc_answered", "no_assay_lens"])
def test_not_asked_as_no_names_every_blank_its_recode_fills(tmp_path, lens, qc_answer):
    """`statin` blank on 30 of the 60 participants (half) is offered "Not asked as No", and its
    sentences name every cell the recode changes, counted by applying its SQL to the table and
    by pandas. With the QC rows set aside (the metabolomics lens) the blanks are counted on the
    participants, the 30 people not asked, and the recode fills the 10 QC rows' blanks too, so
    both are said, where only the 30 were; with the QC rows answered the 30 are every blank it
    fills, said as they were before the QC rows were set aside. With no assay lens no row is set
    aside and every row's blank counts, as before. Each consequence keeps its word budget."""
    frame = statin_panel(30, qc_answer)
    df = read_back(tmp_path, frame)
    part = participants(df)
    on_qc = int(df["statin"].isna().sum()) - int(part["statin"].isna().sum())
    assert int(part["statin"].isna().sum()) == 30 and on_qc == (10 if qc_answer is None else 0)
    assessed = lens == ["metabolomics"]
    n_blank = 30 if assessed else int(df["statin"].isna().sum())
    yes = int((df["statin"] == "Yes").sum())
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)
    table = Ingested(path, tmp_path / "statin-ingested")
    artifact = table.run(findings_stage, ProjectState(lens=lens))
    found = {f["id"]: f for f in artifact["findings"]}
    (option,) = _not_asked(found)
    applied = {"binary_text__statin": FindingDisposition(
        action="applied", option="not_asked_no", params=option["decision"]["params"])}
    after = repairs.evaluate(table.parquet, ["statin"],
                             repairs.column_expressions(artifact, applied))["statin"]
    changed = int(df["statin"].isna().sum())  # every blank the SQL fills, by pandas
    assert not after.isna().any() and int((after == False).sum()) == changed + int(  # noqa: E712
        (df["statin"] == "No").sum())
    assert changed == n_blank + (on_qc if assessed else 0)
    if assessed and on_qc:
        assert option["consequence"] == (
            f"`{n_blank}` non-QC and `{on_qc}` pooled-QC blanks in `statin` become `False`; "
            f"`{yes}` other answers stay as given.")
        assert option["sentence"].startswith(
            f"`statin` was blank for the `{n_blank}` people not asked; those blanks, and the "
            f"`{on_qc}` on the pooled-QC rows, were recoded as `False`,")
    else:  # every blank the recode fills is among those counted
        assert option["consequence"] == (f"`{n_blank}` blanks in `statin` become `False`; its "
                                         f"`{yes}` other answers stay as given.")
        assert option["sentence"].startswith(
            f"`statin` was blank for the `{n_blank}` people not asked; those blanks were recoded "
            f"as `False`,")
    assert voice.words(option["consequence"]) <= repairs.CONSEQUENCE_WORDS
    assert option["decision"]["params"]["n_blank"] == n_blank


def dietary_with_qc(diet: pd.DataFrame, n_qc: int = 12, seed: int = 3,
                    qc_intake: float | None = None) -> pd.DataFrame:
    """``diet`` (an NHANES-shaped dietary table) with an assay panel beside it: 40 features (log
    means 3.5 to 12, blank below exp(4)) and ``n_qc`` pooled-QC rows appended, blank on every
    dietary column but `DR1TKCAL`, which holds ``qc_intake`` there (blank by default), with
    `sample_type` naming them, by construction."""
    rng = np.random.default_rng(seed)
    out = pd.concat([diet, pd.DataFrame(np.nan, index=range(n_qc), columns=diet.columns)],
                    ignore_index=True)
    qc = np.arange(len(out)) >= len(diet)
    if qc_intake is not None:
        out.loc[qc, "DR1TKCAL"] = qc_intake
    log_mean = np.linspace(3.5, 12.0, N_FEATURES)
    log_value = log_mean + rng.normal(0.0, 1.0, (len(out), N_FEATURES)) * np.where(qc[:, None], 0.05, 0.8)
    values = np.round(np.exp(log_value), 2)
    values[log_value < LIMIT] = np.nan
    for j in range(N_FEATURES):
        out[f"mz_{j + 1:03d}"] = values[:, j]
    out.insert(0, "sample_type", np.where(qc, "pooled_qc", "participant"))
    return out


@pytest.mark.parametrize("with_qc", [True, False])
def test_the_energy_finding_counts_the_rows_its_correlation_reads(tmp_path, with_qc):
    """The energy finding's correlation pairs the rows where both columns hold a value: beside 12
    QC rows blank on every intake, the 120 participants', by pandas, where it said "across all 132
    rows". With no QC rows every row holds both, and it says all of them, as before."""
    diet = pd.read_csv(SAMPLES / "nhanes_dietary.csv")
    frame = dietary_with_qc(diet) if with_qc else diet
    df = read_back(tmp_path, frame)
    both = df[["DR1TCARB", "DR1TKCAL"]].dropna()
    r = float(both["DR1TCARB"].corr(both["DR1TKCAL"]))
    lens = ["metabolomics", "dietary"] if with_qc else ["dietary"]
    found = findings_of(tmp_path, frame, lens)
    assert ("pack::metabolomics::pooled_qc" in found) == with_qc
    rows = f"`{len(both):,}` rows" if with_qc else f"all `{len(df):,}` rows"
    assert len(both) == 120 and len(df) == (132 if with_qc else 120)
    assert found["pack::dietary::energy_adjustment"]["summary"] == (
        f"`DR1TCARB` correlates {r:.2f} with `DR1TKCAL` across {rows}: nutrient effects are "
        f"tangled with total energy.")


@pytest.mark.parametrize("with_qc", [True, False])
def test_implausible_intakes_in_kilojoules_count_the_participants(tmp_path, with_qc):
    """Recorded in kJ, 3 participants' intakes below 2,092 kJ (1,000, by construction) are counted
    out of the 120 participants beside 12 QC rows blank on every intake, where they were counted
    out of all 132 rows; with no QC rows, out of every row, as before (pandas)."""
    diet = pd.read_csv(SAMPLES / "nhanes_kilojoules.csv")
    diet.loc[:2, "DR1TKCAL"] = 1000.0
    frame = dietary_with_qc(diet) if with_qc else diet
    df = read_back(tmp_path, frame)
    kj = df["DR1TKCAL"]
    n = int(((kj < 500 * 4.184) | (kj > 5000 * 4.184)).sum())
    assert n == 3 and int(kj.notna().sum()) == 120
    path = tmp_path / "kj.csv"
    frame.to_csv(path, index=False)
    table = Ingested(path, tmp_path / "ingested")
    lens = ["metabolomics", "dietary"] if with_qc else ["dietary"]
    state = ProjectState(lens=lens, column_units={"DR1TKCAL": ColumnUnitSpec(unit="kj")})
    found = {f["id"]: f for f in table.run(findings_stage, state)["findings"]}
    assert ("pack::metabolomics::pooled_qc" in found) == with_qc
    of = "`120` non-QC rows" if with_qc else "`120` rows"
    assert found["pack::dietary::implausible_intake"]["summary"] == (
        f"`{n}` of {of} report `DR1TKCAL` below `2,092` kJ or above `20,920` kJ a day.")


def _implausible(tmp_path: Path, frame: pd.DataFrame, lens: list[str], unit: ColumnUnitSpec) -> dict:
    """The implausible-intake finding the findings stage serves with `DR1TKCAL`'s unit recorded."""
    path = tmp_path / "intake.csv"
    frame.to_csv(path, index=False)
    table = Ingested(path, tmp_path / "intake-ingested")
    state = ProjectState(lens=lens, column_units={"DR1TKCAL": unit})
    found = {f["id"]: f for f in table.run(findings_stage, state)["findings"]}
    assert ("pack::metabolomics::pooled_qc" in found) == ("metabolomics" in lens)
    return found["pack::dietary::implausible_intake"]


@pytest.mark.parametrize("qc", ["intake_zero", "intake_blank", "no_qc"])
def test_implausible_intakes_in_kcal_a_day_are_counted_on_the_rows_they_are_said_out_of(
        tmp_path, qc):
    """Recorded in kcal a day, 3 participants' intakes of 300 kcal (by construction) are the
    implausible ones among the 120 participants (pandas). Beside 12 QC rows whose `DR1TKCAL`
    holds 0, the pack counts 15 over every row, and the summary said `15` of the `120` non-QC rows:
    the summary, the title and the detail count the 3 on the non-QC rows they name. With the QC
    rows blank (as exported) the count is the same 3, said of the same rows; with no QC rows the
    pack's own count over every row stands, in its own words."""
    diet = pd.read_csv(SAMPLES / "nhanes_dietary.csv")
    diet.loc[:2, "DR1TKCAL"] = 300.0
    frame = diet if qc == "no_qc" else dietary_with_qc(
        diet, qc_intake=0.0 if qc == "intake_zero" else None)
    df = read_back(tmp_path, frame)
    rows = df if qc == "no_qc" else participants(df)
    kcal, every = rows["DR1TKCAL"], df["DR1TKCAL"]
    n = int(((kcal < 500) | (kcal > 5000)).sum())
    assert n == 3 and len(rows) == 120
    assert int(((every < 500) | (every > 5000)).sum()) == (15 if qc == "intake_zero" else 3)
    lens = ["dietary"] if qc == "no_qc" else ["metabolomics", "dietary"]
    f = _implausible(tmp_path, frame, lens, ColumnUnitSpec(unit="kcal", days=1))
    of = "`120` rows" if qc == "no_qc" else "`120` non-QC rows"
    assert f["summary"] == f"`{n}` of {of} report `DR1TKCAL` below `500` or above `5000` a day."
    if qc == "no_qc":
        assert f["title"] == f"{n} records report an implausible daily intake."
        assert f["detail"].startswith(f"`DR1TKCAL` is below 500 on {n} record(s) and above 5000 on 0.")
        return
    assert f["title"] == f"{n} non-QC records report an implausible daily intake."
    low = int((kcal < 500).sum())
    assert (f"Read in kcal on the 120 non-QC rows, `DR1TKCAL` is below 500 kcal a day on {low} "
            f"records and above 5,000 kcal a day on 0. Observed range {kcal.min():,.0f} to "
            f"{kcal.max():,.0f} kcal.") in f["detail"]


@pytest.mark.parametrize("qc", ["intake_zero", "intake_blank", "no_qc"])
def test_in_kilojoules_where_no_participant_is_out_of_range_the_rows_within_are_named(
        tmp_path, qc):
    """Recorded in kJ, every participant's intake is within 2,092–20,920 kJ (pandas); the pack,
    comparing the raw values with 500–5,000, fires on every table here. Beside 12 QC rows the
    summary says every non-QC row is within the range, where it said every row, which QC rows
    holding 0 are not; with no QC rows it says every row, as before."""
    diet = pd.read_csv(SAMPLES / "nhanes_kilojoules.csv")
    frame = diet if qc == "no_qc" else dietary_with_qc(
        diet, qc_intake=0.0 if qc == "intake_zero" else None)
    df = read_back(tmp_path, frame)
    rows = df if qc == "no_qc" else participants(df)
    kj = rows["DR1TKCAL"]
    assert int(((kj < 500 * 4.184) | (kj > 5000 * 4.184)).sum()) == 0 and kj.notna().all()
    if qc == "intake_zero":
        assert int((df["DR1TKCAL"] < 500 * 4.184).sum()) == 12
    lens = ["dietary"] if qc == "no_qc" else ["metabolomics", "dietary"]
    f = _implausible(tmp_path, frame, lens, ColumnUnitSpec(unit="kj"))
    which = "every row" if qc == "no_qc" else "every non-QC row"
    assert f["summary"] == (f"Read in kJ, {which} of `DR1TKCAL` is within `2,092`–`20,920` kJ a "
                            f"day; nothing needs excluding.")
    who = "" if qc == "no_qc" else " non-QC"
    assert f["title"] == f"No{who} record reports an implausible daily intake, read in kJ."
