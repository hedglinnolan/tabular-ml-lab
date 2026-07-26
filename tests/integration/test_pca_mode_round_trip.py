"""Switching PCA modes must not crash the page.

The "Fixed Components" number input and the "Variance Threshold" slider shared
one widget key. Pick Variance Threshold and the key holds 0.95; switch back and
the number input reads it as int(0.95) = 0, which is below its own min_value of
1, and Streamlit raises. The researcher had done nothing wrong — they toggled a
radio twice — and the page went to a traceback.

The default was also computed as min(int(_pn), _maxc) with no lower clamp, so a
study with two numeric predictors could reach the same place from the other
side.
"""
import pytest
from streamlit.testing.v1 import AppTest

from tests.integration.conftest import build_test_dataframe, inject_data_state


def _advanced_page(df, **state):
    """Page 05 with the per-model controls open, where the PCA block lives."""
    at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=120)
    inject_data_state(at, df)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    mode = [r for r in at.radio if r.label == "Configuration mode"][0]
    advanced = [o for o in mode.options if "smart" not in o.lower()][0]
    mode.set_value(advanced).run()
    return at


def _pca_boxes(at):
    return [c for c in at.checkbox if "PCA" in (c.label or "")]


def _mode_radios(at):
    return [r for r in at.radio if (r.label or "") == "PCA mode"]


def test_the_two_pca_modes_do_not_share_a_widget_key():
    """The structural fact, so the crash cannot come back by refactor."""
    src = open("pages/05_Preprocess.py").read()
    assert 'key=f"preprocess_{_mk}_pca_fixed_n"' in src
    assert 'key=f"preprocess_{_mk}_pca_variance"' in src
    assert ('st.number_input("Components", 1, _maxc, _defn, '
            'key=f"preprocess_{_mk}_pca_n_components")') not in src


def test_toggling_the_mode_radio_both_ways_does_not_raise():
    """The researcher's actual gesture: pick Variance Threshold, change your mind."""
    at = _advanced_page(build_test_dataframe(n=120))
    boxes = _pca_boxes(at)
    assert boxes, "the PCA control is not reachable — this test would prove nothing"
    boxes[0].check().run()
    assert not at.exception, f"turning PCA on raised: {at.exception}"

    radios = _mode_radios(at)
    assert radios, "the PCA mode radio is not reachable"
    variance = [o for o in radios[0].options if "Variance" in o][0]
    fixed = [o for o in radios[0].options if "Fixed" in o][0]

    for step in ("to variance", "back to fixed", "to variance again", "back again"):
        target = variance if "to variance" in step else fixed
        _mode_radios(at)[0].set_value(target).run()
        assert not at.exception, f"{step} raised: {at.exception}"


@pytest.mark.parametrize("stored", [0.95, 0.5, 0.99, 10, 1, 3])
def test_a_stored_value_of_either_kind_renders_without_an_exception(stored):
    """Whatever the last mode left behind, the page still draws."""
    at = _advanced_page(build_test_dataframe(n=120),
                        preprocess_ridge_use_pca=True,
                        preprocess_ridge_pca_n_components=stored)
    assert not at.exception, f"stored={stored!r}: {at.exception}"


def test_the_page_survives_a_study_with_very_few_predictors():
    """_maxc is tiny here, so an unclamped default of 10 would be out of range."""
    df = build_test_dataframe(n=120)
    at = _advanced_page(df[list(df.columns)[:3]],
                        preprocess_ridge_use_pca=True,
                        preprocess_ridge_pca_n_components=10)
    assert not at.exception, str(at.exception)
