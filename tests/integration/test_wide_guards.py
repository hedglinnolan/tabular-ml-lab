"""Wide/high-cardinality guardrails added after the 3000×34 stress test.

- Preprocess warns (and records a ledger insight) when categorical features
  would one-hot into an explosion of columns, and self-heals when they're gone.
- Train & Compare advises before training on >500 features.
- Explainability advises that permutation importance costs features × repeats
  model evaluations (measured ~141s at 3000 features × 5 repeats).
"""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tests.integration.conftest import (
    build_test_dataframe, inject_data_state, inject_trained_state,
)
from tests.integration.test_lockbox_split import _inject_pipeline, _click_button


def _wide_frame(n=34, p=800, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, p)),
                      columns=[f"feat_{i:04d}" for i in range(p)])
    df["target"] = df["feat_0001"] * 2 + rng.normal(0, 0.5, n)
    return df


class TestHighCardinalityGuard:
    def test_warns_and_records_insight(self):
        df = build_test_dataframe(n=300)
        rng = np.random.default_rng(3)
        df["patient_id"] = [f"PT{i:05d}" for i in range(len(df))]
        df["site"] = rng.choice([f"site_{i}" for i in range(80)], len(df))

        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=120)
        inject_data_state(at, df)
        at.run()

        assert not at.exception
        warnings = " | ".join(w.value for w in at.warning)
        assert "high-cardinality categorical" in warnings
        ins = at.session_state["insight_ledger"].get("preprocess_high_cardinality")
        assert ins is not None
        assert sorted(ins.affected_features) == ["patient_id", "site"]

    def test_self_heals_when_columns_gone(self):
        df = build_test_dataframe(n=300)
        df["patient_id"] = [f"PT{i:05d}" for i in range(len(df))]
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=120)
        inject_data_state(at, df)
        at.run()
        ledger = at.session_state["insight_ledger"]
        assert ledger.get("preprocess_high_cardinality") is not None

        at2 = AppTest.from_file("pages/05_Preprocess.py", default_timeout=120)
        inject_data_state(at2, build_test_dataframe(n=300))
        at2.session_state["insight_ledger"] = ledger
        at2.run()
        assert at2.session_state["insight_ledger"].get(
            "preprocess_high_cardinality") is None


class TestWideFeatureAdvisories:
    def test_train_page_advises_above_500_features(self):
        df = _wide_frame()
        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=300)
        inject_data_state(at, df, target_col="target")
        _inject_pipeline(at, df, target_col="target")
        at.session_state["preprocess_built_model_keys"] = ["ridge"]
        at.run()
        _click_button(at, "Prepare Splits")
        at.session_state["train_model_ridge"] = True
        at.run()

        assert not at.exception
        infos = " | ".join(i.value for i in at.info)
        assert "about to train on" in infos

    def test_explainability_advises_on_perm_cost(self):
        df = _wide_frame()
        at = AppTest.from_file("pages/07_Explainability.py", default_timeout=300)
        inject_data_state(at, df, target_col="target")
        inject_trained_state(at, df, target_col="target")
        at.run()

        assert not at.exception
        infos = " | ".join(i.value for i in at.info)
        assert "reach your models" in infos
