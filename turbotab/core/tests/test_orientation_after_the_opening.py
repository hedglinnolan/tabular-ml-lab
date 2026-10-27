"""Keeping the table as it is after the held-out rows are opened is no change to the opened fit
(the verifier's B3s on the swarm's C3 repair).

The repair let `sample_major` answer an unanswered orientation once the split is drawn, the seal
reading an unanswered orientation as `sample_major`: the oriented stage turns the table only on
`feature_major` (``stages.working.oriented_stage``), so the two give the same rows and the same
fit. The engine's stage keys still keep them apart (``graph.key_value``), and re-keying every
project to merge them is ruled out; so after an opening with the orientation unanswered, recording
`sample_major` gave the served fit a key other than the one the opening was keyed with, and the
Results said the fit changed after the opening: ``changed_after_seal`` true, the answer among the
post-seal decisions, and the reported result's note "from a fit changed after it, so they are
post-seal and not an independent test", though the held-out scores were the opening's.

The truth is the oriented stage's rule: the state at the opening, read with its unanswered
orientation as the answer now recorded, keys the fit the answers now key (``seal.keys_for``, the
engine's own rule), so the opened fit is the one served. Each comparison of the opened fit with
the served one reads it so: ``changed_after_seal``, for the fit served now and for the opened
fit's own record in the fits list (keyed as the answers at the opening key it), and the post-seal
decisions. A change recorded after the opening is still one: an exclusion, beside the kept layout
or without it.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from turbotab.core import decisions as d
from turbotab.core import seal
from turbotab.core.decisions import DecisionRecord
from turbotab.core.stages import build_graph

FINGERPRINT = "f" * 64
KEEP = d.SetOrientation(orientation="sample_major")
EXCLUDE = d.SetExclusions(rules=[d.ExclusionRule(column="x", low=0, high=10,
                                                 reason="implausible values")])


def record(seq: int, decision, rid: str | None = None, *, post: bool = False) -> DecisionRecord:
    return DecisionRecord(id=rid or f"r{seq}", seq=seq, at=datetime.now(timezone.utc),
                          decision=decision, post_seal=post)


def opened() -> list[DecisionRecord]:
    """A project opened with its orientation never answered, its held-out scores kept."""
    return [record(1, d.SetLens(lenses=["dietary"])), record(2, d.SetTarget(column="y")),
            record(3, d.SetPurpose(purpose="prediction")),
            record(4, d.SetRoles(roles={"x": "covariate"})),
            record(5, d.SetSplit(holdout=0.2, seed=0, folds=5), "split"),
            record(6, d.SelectModels(models=["linear"])),
            record(7, d.OpenSeal(family="linear", target="y", n_holdout=40, metric="mse",
                                 scores={"linear": {"mse": 0.7}}), "open")]


def served(graph, records) -> str:
    """The fit the engine serves for the answers now: its key under the log folded."""
    key = seal.keys_for(graph, d.fold(records), FINGERPRINT)["fit"]
    assert key is not None
    return key


def results(graph, records) -> tuple[bool, list[str], dict]:
    """What the Results read beside the held-out scores, as the service computes them: whether
    the served fit changed after the opening, the post-seal decisions, and the reported result."""
    changed = seal.changed_after_seal(graph, records, FINGERPRINT, served(graph, records))
    post = seal.post_seal_changes(records, seal.slots_read_by(graph, "fit"))
    return changed, post, seal.reported_result(records, opened=True, changed=changed)


def test_keeping_the_table_as_it_is_after_the_opening_leaves_the_opened_fit_current():
    graph = build_graph()
    records = [*opened(), record(8, KEEP, "keep", post=True)]
    assert d.fold(records[:7]).orientation is None and d.fold(records).orientation == "sample_major"
    changed, post, at = results(graph, records)
    assert changed is False
    assert post == []
    assert at["current"] is True and at["note"].endswith(": the reported result.")
    # The fits list: the opened fit's own record, keyed as the answers at the opening key it.
    opened_key = seal.keys_for(graph, seal.state_at_opening(records), FINGERPRINT)["fit"]
    assert opened_key != served(graph, records)  # the engine keys the two apart
    assert seal.changed_after_seal(graph, records, FINGERPRINT, opened_key) is False


def test_reverting_the_kept_layout_after_the_opening_is_no_change_either():
    graph = build_graph()
    records = [*opened(), record(8, KEEP, "keep", post=True),
               record(9, d.Revert(decision_id="keep"), "back", post=True)]
    changed, post, at = results(graph, records)
    assert (changed, post, at["current"]) == (False, [], True)


def test_a_fit_recorded_while_the_kept_layout_stood_is_the_opened_fit_after_its_revert():
    """Keep, Fit, then revert the keep (the verifier's T2-C-V3): the fit recorded while the keep
    stood is the opening's state with the orientation read as `sample_major`, the same rows and
    the same fit, so it did not change after the opening, read before the revert or after it. The
    fit served after the revert is the opening's own."""
    graph = build_graph()
    kept = [*opened(), record(8, KEEP, "keep", post=True)]
    recorded = served(graph, kept)  # the fits list's record of the fit made while the keep stood
    reverted = [*kept, record(9, d.Revert(decision_id="keep"), "back", post=True)]
    opened_key = seal.keys_for(graph, seal.state_at_opening(reverted), FINGERPRINT)["fit"]
    assert served(graph, reverted) == opened_key != recorded  # the engine keys the two apart
    assert seal.changed_after_seal(graph, kept, FINGERPRINT, recorded) is False
    assert seal.changed_after_seal(graph, reverted, FINGERPRINT, recorded) is False
    assert seal.fit_keys(graph, seal.state_at_opening(reverted), FINGERPRINT) == {opened_key,
                                                                                 recorded}
    # A fit recorded after an exclusion beside the kept layout is still a change, after the
    # revert of the keep as before it.
    excluded = [*kept, record(9, EXCLUDE, "exclude", post=True)]
    changed_fit = served(graph, excluded)
    back = [*excluded, record(10, d.Revert(decision_id="keep"), "back", post=True)]
    assert seal.changed_after_seal(graph, excluded, FINGERPRINT, changed_fit) is True
    assert seal.changed_after_seal(graph, back, FINGERPRINT, changed_fit) is True


@pytest.mark.parametrize("kept", [True, False], ids=["beside the kept layout", "alone"])
def test_an_exclusion_after_the_opening_is_still_a_change_to_the_fit(kept):
    graph = build_graph()
    records = opened()
    if kept:
        records.append(record(8, KEEP, "keep", post=True))
    records.append(record(9, EXCLUDE, "exclude", post=True))
    changed, post, at = results(graph, records)
    assert changed is True
    assert post == ["exclude"]
    assert at["current"] is False
    assert at["note"].endswith("the held-out scores shown now are from a fit changed after it, "
                               "so they are post-seal and not an independent test.")
    # The opened fit's own record is still the opened fit.
    opened_key = seal.keys_for(graph, seal.state_at_opening(records), FINGERPRINT)["fit"]
    assert seal.changed_after_seal(graph, records, FINGERPRINT, opened_key) is False


def test_an_orientation_answered_before_the_opening_is_read_as_answered():
    """Recorded `sample_major` before the opening: the state at the opening holds the answer, and
    nothing is read in its place."""
    graph = build_graph()
    base = opened()
    records = [*base[:5], record(5, KEEP, "keep"), *base[5:]]
    records = [r.model_copy(update={"seq": i + 1}) for i, r in enumerate(records)]
    assert seal.state_at_opening(records).orientation == "sample_major"
    changed, post, at = results(graph, records)
    assert (changed, post, at["current"]) == (False, [], True)
