"""`AUDIT-032` — running a diagnostic does not make the report say it was addressed.

`pages/02_EDA.py` flags a `blocker` per leakage candidate, with a manuscript caveat
the app itself authors (*"…raising the possibility of information leakage; results
including this predictor should be interpreted with caution"*). The recommendation
panel then offers **Run Leakage Detection**, which is `ml/eda_actions.leakage_scan` —
it re-reads `signals.leakage_candidate_cols` and returns a string. It removes nothing.

Before the fix, running it called `ledger.resolve(...)`. A resolved insight is
counted by `InsightLedger.narrative_for_report` under *"N were addressed during the
modeling workflow"*, is printed under *"Addressed observations:"*, and is skipped
outright by `discussion_points_for_manuscript` (`utils/insight_ledger.py:1233` —
`if i.resolved: continue`). So a person who pressed a button got a report asserting
the leakage had been handled and a manuscript with the caveat gone, while the column
was still a model feature. The governing rule: the app may be silent and it may
refuse, but it must never assert something false.

**Everything load-bearing here is DRIVEN through `pages/02_EDA.py` and read back out
of the ledger the page itself built** — no assertion imports anything the fix added.
That is deliberate: the first version of this file imported the new recorder at
module scope, and a total revert killed it with `ImportError` at collection, which
proves an import was added and not that the app lied
(`AGENT_ONBOARD.md` §08.1). The page and `utils/insight_ledger.py` are surfaces whose
shape did not change, so a reverted app still answers and only the content differs.

**The class, not only the instance.** All five entries in the recommendation-panel
map are read-only — `leakage_scan`, `multicollinearity_vif`, `missingness_scan`,
`target_profile`, `data_sufficiency_check`. Not one drops a column, fills a value or
transforms a variable, so not one can resolve anything.
`test_no_read_only_diagnostic_resolves_anything` sweeps the whole map rather than the
row's one key.

`GUIDED-097`: every driven claim runs against two fixtures of **different target
shape** — a continuous float outcome (`glucose`) and a binary 0/1 outcome
(`condition`). **The shape not covered is a non-numeric outcome** — a string-labeled
or multi-class target, for which `tests/integration/conftest.py` has no fixture;
`TARGET_SHAPE_NOT_COVERED` names it so the gap is a record rather than an omission.

`GUIDED-045`: every absence assertion is preceded by a positive control that the set
being swept is non-empty and that the insight under test actually fired.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit.testing.v1 import AppTest

from tests.integration.conftest import (
    build_classification_dataframe,
    build_test_dataframe,
    inject_data_state,
)
from utils.insight_ledger import Insight, InsightLedger

TARGET_SHAPE_NOT_COVERED = "non-numeric outcome (string labels / multi-class)"


def _rendered_text(at):
    """Everything a person can actually read on the rendered page."""
    parts = []
    for attr in ("markdown", "caption", "info", "warning", "error", "success"):
        for el in getattr(at, attr, []):
            parts.append(str(getattr(el, "value", "")))
    return "\n".join(parts)


def _with_leak(builder, target):
    """A predictor that is the outcome plus noise — |r| > 0.95 by construction."""
    df = builder()
    rng = np.random.default_rng(7)
    df["lab_x"] = df[target].astype(float) + rng.normal(0, 1e-6, len(df))
    return df


TARGET_SHAPES = [
    pytest.param(build_test_dataframe, "glucose", "regression", id="continuous-float"),
    pytest.param(
        build_classification_dataframe, "condition", "classification", id="binary-0-1"
    ),
]


def _eda_after_running_the_leakage_card(builder, target, task):
    """Render the EDA page, press **Run Leakage Detection**, return (at, leak_ids).

    The positive controls live here so every caller gets them: the blocker has to
    have fired and the button has to exist, or the assertions downstream sweep a
    surface that was never there.
    """
    df = _with_leak(builder, target)
    at = AppTest.from_file("pages/02_EDA.py", default_timeout=180)
    inject_data_state(at, df, target_col=target, task_type=task)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    ledger = at.session_state["insight_ledger"]
    leak_ids = [i.id for i in ledger.insights if i.id.startswith("eda_leakage_")]
    assert leak_ids, (
        "no leakage blocker fired on this fixture, so pressing the scan proves "
        f"nothing; insights present: {[i.id for i in ledger.insights]}"
    )
    buttons = [b for b in at.button if "Leakage" in str(b.label)]
    assert buttons, (
        f"no Leakage Detection button rendered; buttons: "
        f"{[str(b.label) for b in at.button]}"
    )

    buttons[0].click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at, leak_ids


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_running_the_leakage_card_leaves_the_blocker_open(builder, target, task):
    at, leak_ids = _eda_after_running_the_leakage_card(builder, target, task)

    ledger = at.session_state["insight_ledger"]
    resolved = sorted(i.id for i in ledger.insights
                      if i.id.startswith("eda_leakage_") and i.resolved)
    assert not resolved, (
        f"pressing Run Leakage Detection marked the leakage blocker(s) {resolved} "
        f"RESOLVED. ml/eda_actions.leakage_scan re-reads "
        f"signals.leakage_candidate_cols and returns a string — it removes no "
        f"column, so nothing was addressed and the column is still a feature."
    )
    # Not deleted: the diagnostic still reaches the ledger.
    runs = [i.metadata.get("diagnostics_run") or []
            for i in ledger.insights if i.id in leak_ids]
    assert any("leakage_scan" in [r.get("method") for r in hist] for hist in runs), (
        "the completed diagnostic was dropped instead of recorded against the "
        "insight — AUDIT-028's model is a weaker TRUE claim on the same "
        f"subject, never silence. histories: {runs}"
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_running_the_leakage_card_says_it_changed_nothing(builder, target, task):
    """`AGENT_ONBOARD.md` §07 trap 6: the sentence must reach a person.

    Pressing the button reruns the page, which discards everything the button
    block writes — so this reads the text of the page that came back.
    """
    at, _ = _eda_after_running_the_leakage_card(builder, target, task)

    text = _rendered_text(at)
    assert text.strip(), "the EDA page rendered no text after the click"
    lowered = text.lower()
    assert "removed, filled and transformed nothing" in lowered, (
        "the page ran the diagnostic and said nothing about what it did not do, "
        "so a green result reads as the problem having been handled"
    )
    assert "**open**" in lowered, (
        "the disclosure does not tell the user the observation is still open"
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_the_manuscript_keeps_the_caveat_the_app_wrote(builder, target, task):
    at, leak_ids = _eda_after_running_the_leakage_card(builder, target, task)
    ledger = at.session_state["insight_ledger"]

    caveats = [i.manuscript_text for i in ledger.insights
               if i.id in leak_ids and (i.manuscript_text or "").strip()]
    # GUIDED-045 positive control: there is a caveat to lose in the first place.
    assert caveats, (
        f"the leakage blockers {leak_ids} carry no manuscript_text, so the "
        f"assertion below could not fail and proves nothing"
    )

    limitations = ledger.discussion_points_for_manuscript()["limitations"]
    missing = [c for c in caveats if c not in limitations]
    assert not missing, (
        "pressing Run Leakage Detection dropped the leakage caveat from the "
        "manuscript Discussion. utils/insight_ledger.py:1233 skips resolved "
        "insights, so resolving on a read-only scan silently deletes a "
        "limitation the app itself authored while the column is still a "
        f"feature. dropped: {missing}. limitations now: {limitations}"
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_the_report_does_not_list_the_scan_under_addressed(builder, target, task):
    at, leak_ids = _eda_after_running_the_leakage_card(builder, target, task)
    ledger = at.session_state["insight_ledger"]

    findings = [i.finding for i in ledger.insights if i.id in leak_ids]
    assert findings, f"no finding text on {leak_ids} — nothing to look for"

    narrative = ledger.narrative_for_report()
    head, sep, tail = narrative.partition("Accepted/unresolved observations:")
    assert sep, (
        "the report narrative has no unresolved section at all after a "
        f"read-only scan, so the blocker is reported nowhere. narrative:\n{narrative}"
    )
    for f in findings:
        assert f not in head, (
            "the leakage blocker is reported under 'Addressed observations' / in "
            "the 'N were addressed during the modeling workflow' count after a "
            f"diagnostic that removed nothing. narrative:\n{narrative}"
        )
        assert f in tail, (
            "the leakage blocker vanished from the report — it must be reported "
            f"as open, not dropped. narrative:\n{narrative}"
        )


# ── the class sweep ──────────────────────────────────────────────────────


def _recorder():
    """The importable bridge, or a sentence saying why the sweep cannot run.

    Phrased as an assertion rather than a bare import so that a revert reads as
    a claim about the code and not as an `ImportError` at collection time.
    """
    from ml import eda_actions

    needed = ("_ACTION_TO_INSIGHT_MAP", "DIAGNOSTIC_ONLY_ACTIONS",
              "record_diagnostic_on_insights")
    missing = [n for n in needed if not hasattr(eda_actions, n)]
    assert not missing, (
        f"ml/eda_actions.py does not expose {', '.join(missing)}: the "
        f"recommendation-panel → ledger bridge is back inside pages/02_EDA.py, "
        f"where nothing can import it and where the only test of it was a "
        f"hand-copy asserting that a read-only diagnostic resolves a blocker."
    )
    return eda_actions


def test_no_read_only_diagnostic_resolves_anything():
    """Every mapped action is read-only; none of them may resolve anything."""
    ea = _recorder()
    action_map = ea._ACTION_TO_INSIGHT_MAP

    # GUIDED-045 positive control: the swept set is the real map and non-empty.
    assert len(action_map) >= 5, (
        f"the action map shrank to {sorted(action_map)}; this sweep is only "
        f"worth running over the real set"
    )
    assert set(ea.DIAGNOSTIC_ONLY_ACTIONS) == set(action_map), (
        "an action is mapped to insights but is no longer declared read-only — "
        "if it now changes the data it must not go through the recorder"
    )

    for action_id, mapping in sorted(action_map.items()):
        ids = list(mapping.get("exact", []))
        if mapping.get("prefix"):
            ids.append(mapping["prefix"] + "probe")
        assert ids, f"{action_id} maps to no insight id at all"

        ledger = InsightLedger()
        for iid in ids:
            ledger.upsert(Insight(
                id=iid, source_page="02_EDA", category="data_quality",
                severity="warning", finding=f"{iid} fired",
                implication="test", manuscript_text=f"{iid} manuscript text",
            ))

        touched = ea.record_diagnostic_on_insights(
            ledger, action_id, {"findings": ["ran"], "warnings": []}, action_id
        )
        assert sorted(touched) == sorted(ids), (
            f"{action_id} reached {touched}, expected {ids}"
        )
        for iid in ids:
            assert ledger.get(iid).resolved is False, (
                f"{action_id} is a read-only diagnostic — it computes and "
                f"reports — yet running it marked {iid} resolved, which the "
                f"report renders as 'addressed during the modeling workflow'"
            )
        limitations = ledger.discussion_points_for_manuscript()["limitations"]
        for iid in ids:
            assert f"{iid} manuscript text" in limitations, (
                f"{action_id} removed {iid}'s manuscript limitation"
            )
