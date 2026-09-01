"""The page half of the VIF refusal, driven through a real script run.

`tests/test_a_refused_diagnostic_does_not_read_as_a_result.py` pins the engine:
`multicollinearity_vif` declines above p = min(200, n/2) instead of returning
999.0 for every feature. That refusal is only safe if the PAGE reads it, and the
page does three things on a VIF button press that a refusal must not trigger:

* it resolves every open `eda_corr_cluster_*` insight with
  `resolved_by="VIF (Multicollinearity): <first finding>"`. Nothing else in the
  app closes those, so a refusal that closed them would delete a collinearity
  limitation from the manuscript on the strength of an analysis that did not
  run — the `AUDIT-032` class the surrounding code was written to prevent;
* it writes a methodology entry reading "Ran VIF (Multicollinearity)";
* it appends the title to the provenance list of analyses that were performed.

None of that is visible to a headless test of the engine, so this file clicks
the button on a frame with more predictors than observations and checks what
the session actually holds afterwards.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit.testing.v1 import AppTest
from tests.integration.conftest import inject_data_state


def _wide_frame(n=80, p=90, seed=3):
    """More predictors than observations, and no real collinearity in any of them.

    Whatever VIF would report here is an artifact of the shape: at p >= n every
    fit is exact, so the old code returned the 999.0 sentinel for all of them
    and flagged every feature as severely multicollinear.
    """
    rng = np.random.default_rng(seed)
    cols = {f"f{i:03d}": rng.normal(0, 1, n) for i in range(p)}
    # Two columns that ARE collinear, so the page raises a cluster insight for
    # the refusal to leave alone.
    cols["f001"] = cols["f000"] * 2.9 + rng.normal(0, 0.01, n)
    cols["glucose"] = rng.normal(100, 15, n)
    return pd.DataFrame(cols)


@pytest.fixture(scope="module")
def refused():
    at = AppTest.from_file("pages/02_EDA.py", default_timeout=300)
    inject_data_state(at, _wide_frame())
    at.run()
    if at.exception:
        pytest.fail("EDA on a p > n frame raised: "
                    + "; ".join(str(e.value)[:400] for e in at.exception))
    button = [b for b in at.button if b.key == "run_multicollinearity_vif"]
    assert button, "VIF button not found"
    button[0].click().run()
    if at.exception:
        pytest.fail("running VIF on a p > n frame raised: "
                    + "; ".join(str(e.value)[:400] for e in at.exception))
    return at


def _ledger(at):
    return at.session_state["insight_ledger"]


class TestTheRefusalIsVisibleAndRecorded:

    def test_the_engine_refused(self, refused):
        result = refused.session_state["eda_results"]["multicollinearity_vif"]
        assert result.get("refused") is True
        assert result["stats"] == {}

    def test_the_user_is_told_on_screen(self, refused):
        shown = " ".join(w.value for w in refused.warning)
        assert "VIF was not computed" in shown, shown

    def test_the_refusal_is_in_the_ledger_and_unresolved(self, refused):
        entry = _ledger(refused).get("eda_cap_vif_refused")
        assert entry is not None, "the refusal never reached the record"
        assert entry.resolved is False
        assert entry.manuscript_text

    def test_it_reaches_the_discussion_as_a_limitation(self, refused):
        limitations = _ledger(refused).discussion_points_for_manuscript()["limitations"]
        assert any("variance inflation" in t for t in limitations), limitations


class TestTheRefusalCreditsItselfWithNothing:

    def test_the_collinearity_clusters_stay_open(self, refused):
        clusters = [i for i in _ledger(refused).insights
                    if i.id.startswith("eda_corr_cluster_")]
        assert clusters, "no collinearity cluster was raised, so this proves nothing"
        assert all(not i.resolved for i in clusters), (
            "a refused VIF closed the collinearity clusters it did not answer: "
            + repr([(i.id, i.resolved_by) for i in clusters])
        )

    def test_the_methodology_log_does_not_say_it_ran(self, refused):
        log = (refused.session_state["methodology_log"]
               if "methodology_log" in refused.session_state else [])
        actions = [e.get("action", "") for e in log]
        vif = [a for a in actions if "VIF" in a]
        assert vif, f"nothing was logged at all: {actions}"
        assert not any(a.startswith("Ran ") for a in vif), vif
        assert any(a.startswith("Declined") for a in vif), vif

    def test_provenance_does_not_list_it_as_performed(self, refused):
        prov = (refused.session_state["workflow_provenance"]
                if "workflow_provenance" in refused.session_state else None)
        analyses = list(getattr(getattr(prov, "eda", None), "analyses_run", []) or [])
        assert not any("VIF" in a for a in analyses), analyses

    def test_a_later_successful_run_clears_the_refusal(self):
        """The other half of an honest cap: the caveat must not outlive it.

        `upsert` keeps an insight for the life of the session, so a user who
        hit the refusal, narrowed the feature set and re-ran would still carry
        "variance inflation factors were not computed" into the Discussion of a
        paper whose VIF table is printed two pages earlier.
        """
        at = AppTest.from_file("pages/02_EDA.py", default_timeout=300)
        inject_data_state(at, _wide_frame())
        at.run()
        next(b for b in at.button if b.key == "run_multicollinearity_vif").click().run()
        assert _ledger(at).get("eda_cap_vif_refused") is not None

        # Same session, fewer predictors — now p = 10 against n = 80.
        cfg = at.session_state["data_config"]
        cfg.feature_cols = [f"f{i:03d}" for i in range(10)]
        at.session_state["selected_features"] = list(cfg.feature_cols)
        at.run()
        next(b for b in at.button if b.key == "run_multicollinearity_vif").click().run()
        if at.exception:
            pytest.fail("re-running VIF raised: "
                        + "; ".join(str(e.value)[:400] for e in at.exception))

        result = at.session_state["eda_results"]["multicollinearity_vif"]
        assert not result.get("refused"), "VIF should compute at p=10, n=80"
        assert len(result["stats"]["vif"]) == 10
        assert _ledger(at).get("eda_cap_vif_refused") is None, (
            "the 'was not computed' caveat survived the run that computed it"
        )
        limitations = _ledger(at).discussion_points_for_manuscript()["limitations"]
        assert not any("variance inflation" in t for t in limitations), limitations

    def test_no_summary_line_asserts_an_all_clear(self, refused):
        body = " ".join(m.value for m in refused.markdown)
        assert "No severe multicollinearity" not in body, (
            "the page turned a refusal into a clean bill of health"
        )

    def test_a_refusal_does_not_wear_the_previous_run_s_disclosure(self):
        """The guard was right; the plumbing around it discarded the answer.

        `_resolve_insights_from_eda_result` returns "" for a refused result, and
        the page stored the sentence only `if _disclosure:` — so an empty string
        wrote nothing and left the LAST SUCCESSFUL run's entry standing. A user
        who ran VIF on a narrow feature set, widened it and re-ran then saw
        "VIF ... IS the answer to 1 observation this page raised" rendered
        directly above "VIF was not computed".
        """
        at = AppTest.from_file("pages/02_EDA.py", default_timeout=300)
        inject_data_state(at, _wide_frame())
        at.run()

        # Narrow first, so VIF succeeds and writes its disclosure.
        narrow = [f"f{i:03d}" for i in range(10)]
        at.session_state["data_config"].feature_cols = list(narrow)
        at.session_state["selected_features"] = list(narrow)
        at.run()
        next(b for b in at.button if b.key == "run_multicollinearity_vif").click().run()
        assert not at.session_state["eda_results"]["multicollinearity_vif"].get("refused")
        stored = at.session_state["eda_diagnostic_disclosure"]["multicollinearity_vif"]
        assert "reads the data and reports" in stored, stored

        # Widen, so the same action refuses.
        wide = [c for c in _wide_frame().columns if c != "glucose"]
        at.session_state["data_config"].feature_cols = list(wide)
        at.session_state["selected_features"] = list(wide)
        at.run()
        next(b for b in at.button if b.key == "run_multicollinearity_vif").click().run()
        if at.exception:
            pytest.fail("re-running VIF raised: "
                        + "; ".join(str(e.value)[:400] for e in at.exception))

        assert at.session_state["eda_results"]["multicollinearity_vif"]["refused"] is True
        shown = " ".join(str(i.value) for i in at.info)
        assert "reads the data and reports" not in shown, (
            "a refused VIF still claimed to have answered an observation:\n" + shown)
        assert at.session_state["eda_diagnostic_disclosure"]["multicollinearity_vif"] == ""
        assert any("VIF was not computed" in str(w.value) for w in at.warning)
