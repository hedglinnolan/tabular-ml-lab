"""Constitution §04: a robustness trim touches the training partition only.

    | | Eligibility criterion | Robustness trim |
    | Applied to | the whole dataset, **pre-seal** | the training partition
    |            |                                 | only, **post-seal**      |
    | Changes N  | yes — reported in the flow diagram | no |
    | Test set   | obeys it | **never touched** |

And: *"'Also trim the test set to match' is permanently off the menu."*

`STATE-101` measured the violation. The page-05 plausibility filter ran over the
whole frame and wrote the result to `filtered_data`, which `get_data()` serves
to every page — so a post-seal trim silently removed sealed rows. 400 rows, 60
sealed, 7 removed, evaluation on 53 while the chip said 60.

This file pins the sealed set against the trim. It asserts on the *composition
rule* rather than driving Streamlit, because the rule is what has to hold: every
sealed label that was in the frame before the trim is in the frame after it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.pipeline import apply_plausibility_filter                 # noqa: E402


BOUNDS = {"lower_bounds": [0.0, 20.0], "upper_bounds": [120.0, 400.0]}
FEATURES = ["age", "glucose"]


def frame_with_impossible_values(n: int = 400, n_bad: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "age": rng.integers(20, 80, n).astype(float),
        "glucose": rng.normal(95, 12, n),
        "outcome": rng.integers(0, 2, n),
    })
    df.loc[rng.choice(n, n_bad, replace=False), "glucose"] = -999.0
    return df


def restore_sealed(df: pd.DataFrame, filtered: pd.DataFrame,
                   sealed: list) -> pd.DataFrame:
    """The composition pages/05 performs. Mirrored here so the RULE is what is
    tested — if the page stops doing this, `test_the_page_still_restores_them`
    fails on the page's own source."""
    present = [l for l in sealed if l in df.index]
    restored = [l for l in present if l not in filtered.index]
    if not restored:
        return filtered
    keep = set(filtered.index) | set(restored)
    return pd.concat([filtered, df.loc[restored]]).loc[
        [i for i in df.index if i in keep]]


def test_the_unrestored_trim_really_does_remove_sealed_rows():
    """The premise. Without this the guard below could pass vacuously."""
    df = frame_with_impossible_values()
    sealed = list(df.index[:60])
    filtered = apply_plausibility_filter(df, FEATURES, BOUNDS)
    lost = [l for l in sealed if l not in filtered.index]
    assert lost, (
        "the fixture no longer puts any impossible value in the sealed rows, "
        "so it cannot exercise §04")


def test_every_sealed_row_survives_a_post_seal_trim():
    """A robustness trim touches the training partition only; the test set obeys
    an eligibility criterion and is otherwise never touched.

    `STATE-101` measured the violation: the page-05 plausibility filter ran over
    the whole frame into `filtered_data`, which `get_data()` serves everywhere,
    so 7 of 60 sealed rows disappeared and evaluation ran on 53 while the chip
    still said 60.

    This discharges ONE of clause 04's three obligations — the post-seal half.
    The two pre-seal ones are the eligibility question, and until L16 nothing
    linked them, which is why the clause read as covered while half of it was
    not built. See `test_every_clause_is_tracked.py`.

    Clause: `lockbox-04`
    """
    df = frame_with_impossible_values()
    sealed = list(df.index[:60])
    filtered = apply_plausibility_filter(df, FEATURES, BOUNDS)
    out = restore_sealed(df, filtered, sealed)

    missing = [l for l in sealed if l not in out.index]
    assert not missing, (
        f"{len(missing)} sealed row(s) were trimmed away by a post-seal "
        f"robustness decision; §04 says the test set is never touched")
    assert len(out) < len(df), "the trim must still remove TRAINING rows"


def test_the_trim_still_removes_training_rows():
    """The other half. A guard that kept every row would satisfy the letter of
    §04 and delete the feature."""
    df = frame_with_impossible_values()
    sealed = list(df.index[:60])
    filtered = apply_plausibility_filter(df, FEATURES, BOUNDS)
    out = restore_sealed(df, filtered, sealed)

    train_before = [i for i in df.index if i not in set(sealed)]
    train_after = [i for i in out.index if i not in set(sealed)]
    assert len(train_after) < len(train_before), (
        "no training row was trimmed, so the robustness trim stopped working")


def test_row_order_and_identity_survive_the_restore():
    """The restore concatenates. If it did so without re-ordering, the frame's
    row order would change — and row identity in this app is the index label,
    so a reordered frame with the same labels is fine but a REINDEXED one is
    not (`T0-ID-001`)."""
    df = frame_with_impossible_values()
    sealed = list(df.index[:60])
    out = restore_sealed(df, apply_plausibility_filter(df, FEATURES, BOUNDS), sealed)

    assert out.index.is_unique, "the restore duplicated a row label"
    assert list(out.index) == [i for i in df.index if i in set(out.index)], (
        "the restore changed row order relative to the source frame")
    for lbl in sealed:
        pd.testing.assert_series_equal(out.loc[lbl], df.loc[lbl])


def test_the_page_still_restores_them():
    """The rule above is only real if page 05 applies it. Read from source, so
    deleting the restore turns this red rather than leaving the rule tested in
    the abstract and violated in the app."""
    src = open(os.path.join(PROJECT_ROOT, "pages", "05_Preprocess.py"),
               encoding="utf-8").read()
    block = src[src.index("if any_filter:"):]
    block = block[:block.index("st.session_state[\"filtered_data\"]")]
    assert "get_lockbox" in block, (
        "pages/05 applies the plausibility filter without consulting the "
        "lockbox, so a post-seal trim can remove sealed rows again")
    assert "filtered_df = pd.concat" in block or "_restored" in block, (
        "pages/05 no longer puts sealed rows back after trimming")


@pytest.mark.parametrize("n_bad", [1, 60, 200])
def test_it_holds_at_every_severity(n_bad):
    """Escalate on evidence of error, not on the size of the consequence: one
    lost sealed row is the same violation as two hundred."""
    df = frame_with_impossible_values(n_bad=n_bad)
    sealed = list(df.index[:60])
    out = restore_sealed(df, apply_plausibility_filter(df, FEATURES, BOUNDS), sealed)
    assert not [l for l in sealed if l not in out.index]
