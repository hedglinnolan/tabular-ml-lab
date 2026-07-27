"""End-to-end: the EDA page's profile counts training rows, not everyone.

The structural half of CONTRACT-017 is pinned in
`tests/test_eda_does_not_model_on_sealed_rows.py` by reading the page. This is
the half that only a real run can show: with a lockbox frozen at upload, the
`dataset_profile` the page writes — the one the model coach reads on pages 05
and 06 — must describe the unsealed rows.

Tier 2 (AppTest), because `pages/02_EDA.py`'s module body only exists inside a
Streamlit run.
"""
import numpy as np
import pytest

from tests.integration.conftest import build_test_dataframe, inject_data_state

SEALED = 30


@pytest.fixture
def eda_page():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file("pages/02_EDA.py", default_timeout=60)


def _seal(at, df, n=SEALED):
    """Freeze a lockbox exactly as page 01 would — labels are index values."""
    rng = np.random.RandomState(7)
    labels = sorted(rng.choice(df.index.values, size=n, replace=False).tolist())
    at.session_state["test_lockbox"] = {
        "labels": labels,
        "fraction": n / len(df),
        "seed": 7,
        "n_total": len(df),
        "n_test": len(labels),
        "signature": "test-fixture",
        "stratified": False,
    }
    return labels


def _profile(at):
    try:
        return at.session_state["dataset_profile"]
    except KeyError:
        return None


def test_the_profile_the_coach_reads_excludes_the_sealed_rows(eda_page):
    at = eda_page
    df = build_test_dataframe(n=200)
    inject_data_state(at, df, target_col="glucose", task_type="regression")
    sealed = _seal(at, df)

    at.run()
    assert not at.exception, f"EDA errored: {at.exception}"

    profile = _profile(at)
    assert profile is not None, "EDA must still write dataset_profile"
    assert profile.n_rows == len(df) - len(sealed), (
        f"the profile counted {profile.n_rows} rows out of {len(df)} with "
        f"{len(sealed)} sealed — the model coach is being shown held-out people")


def test_the_page_states_the_scoping_on_screen(eda_page):
    at = eda_page
    df = build_test_dataframe(n=200)
    inject_data_state(at, df, target_col="glucose", task_type="regression")
    _seal(at, df)

    at.run()
    assert not at.exception, f"EDA errored: {at.exception}"

    captions = " ".join(c.value for c in at.caption if c.value)
    assert "training rows" in captions and "held-out test rows are excluded" in captions, (
        "the page quarantines rows without saying so; a silent scoping is a "
        "number the reader cannot interpret")
    assert f"n={len(df) - SEALED}" in captions, (
        "the on-screen count does not match the rows the profile saw")


def test_without_a_lockbox_the_profile_still_covers_everyone(eda_page):
    """The fix must not quietly shrink an analysis that sealed nothing."""
    at = eda_page
    df = build_test_dataframe(n=200)
    inject_data_state(at, df, target_col="glucose", task_type="regression")

    at.run()
    assert not at.exception, f"EDA errored: {at.exception}"

    profile = _profile(at)
    assert profile is not None
    assert profile.n_rows == len(df)

    captions = " ".join(c.value for c in at.caption if c.value)
    assert "held-out test rows are excluded" not in captions, (
        "the page claimed a scoping that did not happen")
