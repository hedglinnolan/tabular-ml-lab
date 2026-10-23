"""The Bang 2009 prep's predictors, each from rows built by hand whose level the paper's rule gives
(``bang2009.md`` §2.3, Table 2's footnote), never from the prep's own output; and the intake's
table read only where its bytes are its manifest's."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from research.replication.bang2009 import prep

NA = np.nan


def frame(**columns) -> pd.DataFrame:
    n = max(len(v) for v in columns.values())
    base = {"RIAGENDR": [1] * n, "BMXBMI": [NA] * n, "BMXWAIST": [NA] * n, "BPQ020": [NA] * n,
            "BPQ040A": [NA] * n, "BPQ050A": [NA] * n, "MCQ250A": [NA] * n, "PAQ520": [NA] * n,
            **{c: [NA] * n for c in prep.SYSTOLIC + prep.DIASTOLIC},
            **{c: [NA] * n for c in prep.CLOSE_RELATIVES}}
    base.update(columns)
    return pd.DataFrame(base)


def test_age_bands_are_under_40_then_40_49_50_59_and_60_up():
    ages = pd.Series([20, 39, 40, 49, 50, 59, 60, 85, NA])
    assert prep.age_band(ages).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, pd.NA]


def test_obesity_is_the_higher_of_the_bmi_and_the_sex_specific_waist_in_inches():
    # (sex, BMI, waist in cm) -> the level the paper's cascade gives, by hand:
    rows = [
        (1, 24.9, 93.0, 0),    # BMI under 25; 93.0 cm = 36.61 in, under the men's 37
        (1, 24.9, 94.0, 1),    # 94.0 cm = 37.01 in: overweight by the waist alone
        (2, 22.0, 79.9, 0),    # 79.9 cm = 31.46 in, under the women's 31.5
        (2, 22.0, 80.1, 1),    # 80.1 cm = 31.54 in: overweight
        (1, 25.0, NA, 1),      # BMI 25 exactly is overweight; the waist not measured
        (1, 30.0, 90.0, 2),    # BMI 30 exactly is obese
        (1, 40.0, 90.0, 3),    # BMI 40 exactly is extremely obese
        (1, 22.0, 127.1, 3),   # 127.1 cm = 50.04 in: the men's extremely obese waist
        (2, 22.0, 124.6, 3),   # 124.6 cm = 49.06 in: the women's
        (1, 22.0, 124.6, 2),   # the same waist in a man (49.06 in) is obese, under his 50
        (2, 22.0, 89.0, 2),    # 89.0 cm = 35.04 in: the women's obese waist
        (1, NA, 102.0, 2),     # BMI not measured; 102 cm = 40.16 in: obese by the waist
        (2, 41.0, 70.0, 3),    # the BMI's level wins over a lower waist
        (1, NA, NA, None),     # neither measured
    ]
    f = frame(RIAGENDR=[r[0] for r in rows], BMXBMI=[r[1] for r in rows],
              BMXWAIST=[r[2] for r in rows])
    got = prep.obesity(f).tolist()
    assert got == [pd.NA if r[3] is None else r[3] for r in rows]
    assert prep.obesity(f, missing_as_absent=True).tolist() == [
        0 if r[3] is None else r[3] for r in rows]


def test_hypertension_reads_the_history_the_medicine_and_the_mean_or_first_reading():
    f = frame(
        BPQ020=[1, 9, 2, 2, 2, 2, NA, 2, 2],
        BPQ050A=[NA, NA, 1, NA, NA, NA, NA, NA, 2],
        BPQ040A=[NA, NA, NA, NA, NA, NA, NA, NA, 1],
        BPXSY1=[NA, NA, NA, 138, 138, 120, NA, NA, NA],
        BPXSY2=[NA, NA, NA, 142, 140, 118, NA, 142, NA],
        BPXSY3=[NA, NA, NA, NA, NA, 122, NA, 130, NA],
        BPXDI1=[NA, NA, NA, 70, 70, 88, NA, NA, NA],
        BPXDI2=[NA, NA, NA, 70, 70, 92, NA, 70, NA],
        BPXDI3=[NA, NA, NA, NA, NA, NA, NA, 70, NA],
    )
    # by hand: told; don't know (absent); taking the medicine; mean SBP 140; mean SBP 139;
    # mean DBP 90; nothing recorded; mean SBP 136 (the first recorded, 142, is high); told to take
    # the medicine but not taking it.
    assert prep.hypertension(f).tolist() == [1, 0, 1, 1, 0, 1, 0, 0, 0]
    assert prep.hypertension(f, reading="first").tolist() == [1, 0, 1, 0, 0, 0, 0, 1, 0]
    assert prep.hypertension(f, medicine="BPQ040A").tolist() == [1, 0, 0, 1, 0, 1, 0, 0, 1]
    with pytest.raises(ValueError):
        prep.hypertension(f, reading="last")


def test_family_history_is_any_blood_relative_or_a_parent_or_sibling():
    f = frame(MCQ250A=[1, 2, 9, NA, 1, 1, 1, 1],
              MCQ260AA=[NA, NA, NA, NA, NA, NA, 99, 1],
              MCQ260AB=[NA] * 8,
              MCQ260AG=[NA, NA, NA, NA, NA, 7, NA, NA],
              MCQ260AH=[NA] * 8)
    assert prep.family_history(f).tolist() == [1, 0, 0, 0, 1, 1, 1, 1]
    # a relative named only by grandparents' or others' columns is not a parent or sibling; a
    # don't-know (99) is the condition absent.
    assert prep.family_history(f, close=True).tolist() == [0, 0, 0, 0, 0, 1, 0, 1]


def design(cycles, w2, w4, strata=None, psu=None) -> pd.DataFrame:
    n = len(cycles)
    return pd.DataFrame({"SDDSRVYR": cycles, "WTSAF2YR": w2, "WTSAF4YR": w4,
                         "SDMVSTRA": strata or [1, 1, 2, 2][:n] + [2] * max(0, n - 4),
                         "SDMVPSU": psu or [1, 2, 1, 2][:n] + [2] * max(0, n - 4)})


def test_the_subsample_keeps_the_rows_with_a_fasting_weight_zero_or_not():
    # rows 1 and 4 have no fasting record; row 2 did not fast (weight 0); each stratum keeps both
    # its PSUs among the rest.
    t = design([1, 1, 2, 3, 3, 3], [5000.0, NA, 0.0, NA, 4000.0, 3000.0],
               [3000.0, NA, 0.0, NA, NA, NA], strata=[1, 1, 1, 2, 2, 2], psu=[1, 2, 2, 1, 1, 2])
    kept = prep.subsample(t)
    assert kept.index.tolist() == [0, 2, 4, 5]


def test_the_subsample_is_refused_without_the_four_year_weight_or_a_stratum_s_psu():
    no_w4 = design([1, 2, 3, 3], [5000.0, 4000.0, 3000.0, 2000.0], [3000.0, NA, NA, NA])
    with pytest.raises(ValueError, match="lack WTSAF4YR"):
        prep.subsample(no_w4)
    # stratum 2's PSU 1 holds only a row outside the subsample
    lost = design([3, 3, 3, 3], [5000.0, 4000.0, NA, 2000.0], [NA] * 4)
    with pytest.raises(ValueError, match="PSUs"):
        prep.subsample(lost)


def test_every_predictor_follows_its_row_when_the_index_has_gaps():
    # The subsample leaves the intake's index with gaps; each predictor must stay on its own row
    # (the parent-or-sibling column once went blank there, by index alignment).
    f = frame(MCQ250A=[1, 1, 2], MCQ260AA=[1, NA, NA], MCQ260AB=[NA] * 3,
              MCQ260AG=[NA, 7, NA], MCQ260AH=[NA] * 3, BMXBMI=[22.0] * 3, BMXWAIST=[80.0] * 3,
              PAQ520=[1, 2, 1])
    for c in prep.KEPT:
        f[c] = [1.0, 2.0, 3.0]
    f.index = [10, 4000, 9999]
    out = prep.derive(f)
    assert out["famhx_close"].tolist() == [1, 1, 0]
    assert out["famhx"].tolist() == [1, 1, 0]
    assert out["active"].tolist() == [1, 0, 1]
    assert not out[prep.PREDICTORS].drop(columns=["obesity"]).isna().any().any()


def test_derive_keeps_every_row_and_the_codes_of_sex_and_activity():
    n = 6
    f = frame(RIAGENDR=[1, 2, 1, 2, 1, 2], PAQ520=[1, 2, 3, 7, 9, NA],
              BMXBMI=[22.0] * n, BMXWAIST=[80.0] * n)
    for c in prep.KEPT:
        f[c] = list(range(n)) if c != "diabetes_fpg" else [0.0, 1.0, NA, 0.0, 1.0, 0.0]
    f["RIDAGEYR"] = [20, 45, 55, 65, 30, 41]
    f["LBXGLU"] = [90.0] * n
    out = prep.derive(f)
    assert len(out) == n and out["SEQN"].tolist() == list(range(n))
    assert "LBXGLU" not in out.columns
    assert out["male"].tolist() == [1, 0, 1, 0, 1, 0]
    assert out["active"].tolist() == [1, 0, 0, 0, 0, 0]
    assert out["age_band"].tolist() == [0, 1, 2, 3, 0, 1]
    assert str(out["diabetes_fpg"].dtype) == "Int64"
    assert out["diabetes_fpg"].tolist() == [0, 1, pd.NA, 0, 1, 0]
    assert set(prep.PREDICTORS) <= set(out.columns)


def test_a_code_column_is_refused_unless_built_from_a_series_whose_index_it_keeps():
    # A bare array would take a fresh index 0..n-1, and its codes would land on other rows.
    with pytest.raises(TypeError, match="Series"):
        prep.codes(np.array([1, 0]))
    with pytest.raises(TypeError, match="Series"):
        prep.codes([1, 0])
    kept = prep.codes(pd.Series([1.0, 0.0, NA], index=[5, 9, 4000]))
    assert kept.index.tolist() == [5, 9, 4000]
    assert str(kept.dtype) == "Int64" and kept.tolist() == [1, 0, pd.NA]


def manifest_of(tmp_path, table: pd.DataFrame) -> tuple:
    """``table`` written under a scratch repo and a manifest holding its SHA-256 (hashlib's), never
    the shared research/data."""
    path = tmp_path / "research" / "data" / "bang2009.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    entry = {"file": "research/data/bang2009.parquet",
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"table": entry}), "utf-8")
    return path, entry, manifest


def test_the_intake_table_is_read_only_where_its_bytes_are_the_manifest_s(monkeypatch, tmp_path):
    table = pd.DataFrame({"SEQN": [1, 2, 3], "WTSAF2YR": [2631.0, NA, 0.0]})
    path, entry, manifest = manifest_of(tmp_path, table)
    monkeypatch.setattr(prep, "REPO", tmp_path)
    monkeypatch.setattr(prep, "MANIFEST", manifest)
    read, kept = prep.intake_table()
    pd.testing.assert_frame_equal(read, table)
    assert kept == entry

    # One value changed: other bytes, so not the manifest's table.
    table.assign(SEQN=[1, 2, 4]).to_parquet(path, index=False)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
    with pytest.raises(SystemExit, match="is not the manifest's table"):
        prep.intake_table()
    path.unlink()
    with pytest.raises(SystemExit, match="is not the manifest's table"):
        prep.intake_table()
