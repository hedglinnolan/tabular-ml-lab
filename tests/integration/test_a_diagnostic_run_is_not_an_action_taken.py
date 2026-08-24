"""`AUDIT-032` — running a diagnostic does not make the report say it was addressed.

`pages/02_EDA.py` flags a `blocker` per leakage candidate, with a manuscript caveat
the app itself authors (*"…raising the possibility of information leakage; results
including this predictor should be interpreted with caution"*). That blocker is
produced by the **automatic** >0.95 feature-target correlation scan
(`ml/eda_recommender.py:456-481` populates `signals.leakage_candidate_cols`;
`pages/02_EDA.py:359-375` turns each candidate into a `blocker` insight), and it
still gates sign-off at `pages/02_EDA.py:768-771` and `:2540-2541`.

Before the fix, pressing any recommendation-panel diagnostic called
`ledger.resolve(...)`. A resolved insight is counted by
`InsightLedger.narrative_for_report` under *"N were addressed during the modeling
workflow"*, is printed under *"Addressed observations:"*, and is skipped outright by
`discussion_points_for_manuscript` (`utils/insight_ledger.py:1233` — `if i.resolved:
continue`). So a person who pressed a button got a report asserting the problem had
been handled and a manuscript with the caveat gone, while the column was still a
model feature. The governing rule: the app may be silent and it may refuse, but it
must never assert something false.

**MERGE NOTE — which button these tests press.** These page-drive tests used to press
**Run Leakage Detection**. Main's diagnostics dedup (`7480564`, *"Keep the five
diagnostics nothing else computes, and drop the eleven that repeat"*) delisted
`leakage_scan` from `pages/02_EDA.py` — a **UI** decision about a deep-dive button
that re-rendered what §1-5 already show. The automatic scan, the blocker, the
manuscript caveat and the sign-off gate all survive, and `leakage_scan` is still in
`_ACTION_TO_INSIGHT_MAP`, so `test_no_read_only_diagnostic_resolves_anything` below
still sweeps it. What was lost is only the *vehicle*: there is no longer a Leakage
Detection button to press. The page-drive tests therefore press the diagnostic that
did survive and is rendered for **both** target shapes — **Run VIF
(Multicollinearity)** — and the subject of the assertions is unchanged: the leakage
blocker must still be open, still caveated, and still reported as unresolved after a
diagnostic has been run.

**The VIF carve-out is asserted, not assumed.** The merge gave VIF one deliberate
resolving power: it closes `eda_corr_cluster_*`, because VIF is the answer to the
pairwise-correlation clusters the page itself raised and nothing else in the app
closes them (`pages/02_EDA.py:2280-2301`). That is the *only* thing it may close.
`test_the_vif_carve_out_closes_only_the_clusters_it_answers` states the carve-out
positively and fences it: everything the click resolves must be a cluster it answers.

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
transforms a variable, so not one may resolve anything the recorder touches.
`test_no_read_only_diagnostic_resolves_anything` sweeps the whole map rather than the
row's one key, which is where `leakage_scan` is still guarded now that it has no UI.

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


def _with_leak_and_a_collinear_pair(builder, target):
    """Two planted problems, because the click has to be able to tell them apart.

    `lab_x` is the outcome plus noise — |r| > 0.95 with the target by construction,
    so the automatic leakage scan raises a `blocker`. `bmi_repeat` is a near-copy of
    an existing predictor, so the page raises an `eda_corr_cluster_*` — the one thing
    the merge licensed VIF to close. Without the second column the carve-out test
    would sweep an empty set and prove nothing (`GUIDED-045`).
    """
    df = builder()
    rng = np.random.default_rng(7)
    df["lab_x"] = df[target].astype(float) + rng.normal(0, 1e-6, len(df))
    df["bmi_repeat"] = df["bmi"] * 1.02 + rng.normal(0, 0.05, len(df))
    return df


TARGET_SHAPES = [
    pytest.param(build_test_dataframe, "glucose", "regression", id="continuous-float"),
    pytest.param(
        build_classification_dataframe, "condition", "classification", id="binary-0-1"
    ),
]


def _eda_after_running_a_diagnostic(builder, target, task):
    """Render the EDA page, press **Run VIF (Multicollinearity)**, return the state.

    Returns `(at, leak_ids, cluster_ids, resolved_before)`.

    The positive controls live here so every caller gets them: the leakage blocker
    has to have fired, a collinearity cluster has to have fired, and the button has
    to exist, or the assertions downstream sweep a surface that was never there.

    VIF is the diagnostic used because after `7480564` it is the only one on the page
    that is both mapped into `_ACTION_TO_INSIGHT_MAP` and rendered unconditionally —
    Physiologic Plausibility needs a biomedical column match, and Residual Normality
    and Influence Diagnostics are regression-only, so neither survives the
    `GUIDED-097` sweep over both target shapes.
    """
    df = _with_leak_and_a_collinear_pair(builder, target)
    at = AppTest.from_file("pages/02_EDA.py", default_timeout=180)
    inject_data_state(at, df, target_col=target, task_type=task)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    ledger = at.session_state["insight_ledger"]
    present = [i.id for i in ledger.insights]
    leak_ids = [i.id for i in ledger.insights
                if i.id.startswith("eda_leakage_") and not i.resolved]
    assert leak_ids, (
        "no OPEN leakage blocker fired on this fixture, so pressing a diagnostic "
        f"proves nothing; insights present: {present}. The automatic >0.95 scan "
        f"(ml/eda_recommender.py) is what raises these — if it is gone, that is a "
        f"regression in the app, not in this test."
    )
    cluster_ids = [i.id for i in ledger.insights
                   if i.id.startswith("eda_corr_cluster_") and not i.resolved]
    assert cluster_ids, (
        f"no OPEN collinearity cluster fired on this fixture, so the VIF carve-out "
        f"below would sweep an empty set; insights present: {present}"
    )
    buttons = [b for b in at.button if "VIF" in str(b.label)]
    assert buttons, (
        f"no VIF (Multicollinearity) button rendered; buttons: "
        f"{[str(b.label) for b in at.button]}"
    )
    resolved_before = {i.id for i in ledger.insights if i.resolved}

    buttons[0].click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at, leak_ids, cluster_ids, resolved_before


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_running_a_diagnostic_leaves_the_leakage_blocker_open(builder, target, task):
    at, leak_ids, _, _ = _eda_after_running_a_diagnostic(builder, target, task)

    ledger = at.session_state["insight_ledger"]
    resolved = sorted(i.id for i in ledger.insights
                      if i.id.startswith("eda_leakage_") and i.resolved)
    assert not resolved, (
        f"pressing Run VIF (Multicollinearity) marked the leakage blocker(s) "
        f"{resolved} RESOLVED. VIF computes variance inflation and returns a "
        f"table — it removes no column, so the leaking predictor is still a "
        f"feature and nothing about it was addressed."
    )
    still_there = [i.id for i in ledger.insights if i.id in leak_ids]
    assert sorted(still_there) == sorted(leak_ids), (
        "the leakage blocker was deleted from the ledger instead of left open — "
        "AUDIT-028's model is a weaker TRUE claim on the same subject, never "
        f"silence. expected {leak_ids}, present {[i.id for i in ledger.insights]}"
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_the_vif_carve_out_closes_only_the_clusters_it_answers(builder, target, task):
    """The one resolving power the merge granted a diagnostic, stated and fenced.

    `pages/02_EDA.py:2280-2301` lets VIF close `eda_corr_cluster_*`, because VIF is
    the answer to the pairwise-correlation clusters the page itself raised and
    nothing else in the app closes them. This asserts that it happens (so the
    carve-out is a claim, not an assumption) and that it is the *only* thing the
    click closes — which is the AUDIT-032 contract restated for the diagnostic that
    does have an exception.
    """
    at, leak_ids, cluster_ids, resolved_before = _eda_after_running_a_diagnostic(
        builder, target, task)
    ledger = at.session_state["insight_ledger"]

    # (a) The carve-out fired, and it names the diagnostic that fired it.
    for cid in cluster_ids:
        ins = ledger.get(cid)
        assert ins is not None and ins.resolved, (
            f"{cid} is still open after VIF ran. Nothing else in the app closes "
            f"eda_corr_cluster_*, so left open it reaches the manuscript as a "
            f"limitation the user has in fact already investigated."
        )
        assert "multicollinearity_vif" in str(
            (ins.resolution_details or {}).get("method", "")
        ) or "VIF" in str(ins.resolved_by or ""), (
            f"{cid} was resolved without recording what resolved it: "
            f"resolved_by={ins.resolved_by!r} details={ins.resolution_details!r}"
        )

    # (b) Nothing else. `method_*` ids are the run's own provenance record — the
    # methodology log entry the action returns for "Ran VIF (Multicollinearity)" —
    # not a data observation about the dataset, so they are named and excluded
    # rather than quietly swept in.
    resolved_after = {i.id for i in ledger.insights if i.resolved}
    newly_resolved = resolved_after - resolved_before
    overreach = sorted(
        i for i in newly_resolved
        if not i.startswith("eda_corr_cluster_") and not i.startswith("method_")
    )
    assert not overreach, (
        f"running VIF resolved {overreach}, which it does not answer. The carve-out "
        f"is exactly eda_corr_cluster_*; anything else a read-only diagnostic marks "
        f"resolved is reported as 'addressed during the modeling workflow' while the "
        f"data is unchanged. open leakage blockers at the time of the click: "
        f"{leak_ids}"
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_running_the_diagnostic_says_what_it_did_not_do(builder, target, task):
    """`AGENT_ONBOARD.md` §07 trap 6: the sentence must reach a person.

    Pressing the button reruns the page, which discards everything the button
    block writes — so this reads the text of the page that came back.

    The exact sentence depends on how many mapped-and-still-open observations the
    diagnostic spoke to, and for VIF that is zero by construction: the carve-out
    resolves the clusters first and `record_diagnostic_on_insights` skips resolved
    insights. So what must reach the reader is the "it changes nothing" variant.
    KNOWN GAP, flagged rather than asserted away: after `7480564` no button on this
    page can produce the `n_open > 0` variant ("removed, filled and transformed
    nothing … stay **open**"), because the only mapped action rendered here is the
    one with the carve-out and the other three surviving diagnostics are not in
    `_ACTION_TO_INSIGHT_MAP` at all and so disclose nothing.
    """
    at, _, _, _ = _eda_after_running_a_diagnostic(builder, target, task)

    text = _rendered_text(at)
    assert text.strip(), "the EDA page rendered no text after the click"
    lowered = text.lower()
    assert "reads the data and reports" in lowered, (
        "the page ran the diagnostic and said nothing about what it did not do, "
        "so a green result reads as the problem having been handled"
    )
    assert "it changes nothing" in lowered or "transformed nothing" in lowered, (
        "the disclosure does not tell the user the diagnostic left the data alone"
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_the_manuscript_keeps_the_caveat_the_app_wrote(builder, target, task):
    at, leak_ids, _, _ = _eda_after_running_a_diagnostic(builder, target, task)
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
        "pressing Run VIF (Multicollinearity) dropped the leakage caveat from the "
        "manuscript Discussion. utils/insight_ledger.py:1233 skips resolved "
        "insights, so resolving on a read-only scan silently deletes a "
        "limitation the app itself authored while the column is still a "
        f"feature. dropped: {missing}. limitations now: {limitations}"
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_the_report_does_not_list_the_scan_under_addressed(builder, target, task):
    at, leak_ids, _, _ = _eda_after_running_a_diagnostic(builder, target, task)
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
