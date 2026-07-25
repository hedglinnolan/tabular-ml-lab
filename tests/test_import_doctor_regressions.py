"""Two Import Doctor defects the pre-PR audit reproduced end to end.

Both destroy a researcher's data while looking helpful, which is the failure
this module exists to prevent.

1. A corrected re-upload of the same shape silently committed the OLD file.
   The frame's identity was rows x cols x column-names, with content left out,
   so version 2 matched version 1's cached repair and version 1 is what reached
   the project, the working table, the audit and the lockbox. The preview
   showed the old numbers too, so there was nothing to notice.

2. A CRITICAL finding on clean integer columns, with a one-click button that
   recodes real measurements to missing. It fired on 13 of 40 clean
   NHANES-shaped files, and its own detail sentence — "far outside the rest of
   the column (18 to 80)" about the value 77 — was contradicted by the numeric
   summary on the same page.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from ml.import_doctor import diagnose
from utils.import_ui import _frame_signature, repaired_frame


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear()
    yield
    st.session_state.clear()


def sentinel_ids(df):
    return [f.id for f in diagnose(df) if f.id.startswith("sentinel_missing__")]


# ── 1. a corrected file is not the old file ──────────────────────────────

class TestFrameIdentityIncludesContent:

    def test_same_shape_different_numbers_is_a_different_frame(self):
        v1 = pd.DataFrame({"id": range(50), "glucose": np.full(50, 95.0)})
        v2 = pd.DataFrame({"id": range(50), "glucose": np.full(50, 195.0)})
        assert v1.shape == v2.shape and list(v1.columns) == list(v2.columns)
        assert _frame_signature(v1) != _frame_signature(v2)

    def test_the_same_frame_is_still_the_same_frame(self):
        v1 = pd.DataFrame({"id": range(50), "glucose": np.full(50, 95.0)})
        assert _frame_signature(v1) == _frame_signature(v1.copy())

    def test_a_stored_repair_is_not_served_to_a_corrected_upload(self):
        v1 = pd.DataFrame({"id": range(50), "glucose": np.full(50, 95.0)})
        v2 = pd.DataFrame({"id": range(50), "glucose": np.full(50, 195.0)})
        repaired_v1 = v1.assign(glucose=np.full(50, 99.0))
        st.session_state["_impdoc_frame_f0"] = repaired_v1
        st.session_state["_impdoc_sig_f0"] = _frame_signature(v1)

        assert repaired_frame(v1, "f0")["glucose"].mean() == 99.0   # same file
        served = repaired_frame(v2, "f0")
        assert served["glucose"].mean() == 195.0, (
            "the corrected file was served version 1's data")


# ── 2. a code sits beyond the data, never inside it ──────────────────────

class TestSentinelsMustBeOutsideTheData:

    @pytest.mark.parametrize("seed", range(12))
    def test_clean_clinical_integers_are_not_called_codes(self, seed):
        rng = np.random.default_rng(seed)
        n = 400
        df = pd.DataFrame({
            "RIDAGEYR": rng.integers(18, 81, n),
            "systolic": rng.integers(80, 181, n),
            "y": rng.integers(0, 2, n),
        })
        assert sentinel_ids(df) == [], (
            "a clean file was offered a button that deletes real measurements")

    def test_a_real_999_is_still_caught(self):
        rng = np.random.default_rng(0)
        age = rng.integers(18, 81, 400).astype(float)
        age[:12] = 999
        df = pd.DataFrame({"age": age, "y": rng.integers(0, 2, 400)})
        assert "sentinel_missing__age" in sentinel_ids(df)

    def test_a_negative_code_is_still_caught(self):
        rng = np.random.default_rng(1)
        v = rng.integers(0, 101, 400).astype(float)
        v[:15] = -9
        df = pd.DataFrame({"score": v, "y": rng.integers(0, 2, 400)})
        assert "sentinel_missing__score" in sentinel_ids(df)

    def test_a_top_block_of_codes_on_a_likert_scale_is_still_caught(self):
        rng = np.random.default_rng(2)
        vals = np.concatenate([rng.integers(1, 6, 380),
                               np.full(10, 7), np.full(10, 8), np.full(10, 9)])
        df = pd.DataFrame({"q1": vals, "y": rng.integers(0, 2, len(vals))})
        assert "sentinel_missing__q1" in sentinel_ids(df)

    def test_the_stated_range_no_longer_contradicts_itself(self):
        rng = np.random.default_rng(3)
        age = rng.integers(18, 81, 400).astype(float)
        age[:12] = 999
        df = pd.DataFrame({"age": age, "y": rng.integers(0, 2, 400)})
        finding = next(f for f in diagnose(df) if f.id == "sentinel_missing__age")
        # "far outside the rest of the column (lo to hi)" must be true of 999
        import re
        lo, hi = (float(x) for x in re.search(r"\(([\d.]+) to ([\d.]+)\)",
                                              finding.detail).groups())
        assert 999 > hi, f"999 is not outside the stated range {lo}-{hi}"
