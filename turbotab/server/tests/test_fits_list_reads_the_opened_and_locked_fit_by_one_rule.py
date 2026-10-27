"""The fits list compares a record's fit with the opened fit and the locked fit by the seal's one
rule (the orchestrator's ruling (c) on the swarm's tail; T2-C-V3, T2-C-open-1, T2-C-open-4): a fit
whose rows and fit equal the opening's, or the lock's, is that fit.

Keeping the table as it is (`sample_major`) answers an unanswered orientation without turning the
table: the oriented stage turns it only on `feature_major` (``stages.working.oriented_stage``), so
the two give the same rows and the same fit, though the engine keys them apart. So:

* a record whose fit artifact is no longer kept, of a fit made while the keep stood after the
  opening, is not "changed after the opening", the keep standing or reverted
  (``service._served_record``, which compared its key with the opening's alone);
* under Estimate, a fit made after keeping the table as it is after the plan's lock, or before
  the lock with the keep since reverted, is the fit the lock was recorded on
  (``ProjectService._locked_fit``, which gave one key), so the fits list marks it locked and its
  record's gate serves it as the lock's.

Each guard both ways: an exclusion recorded after the opening or the lock is still a change.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from turbotab.core import decisions as d
from turbotab.core import seal
from turbotab.core.decisions import DecisionRecord
from turbotab.core.runs import FitRecord
from turbotab.core.stages import build_graph
from turbotab.server.service import ProjectService, _served_record

FINGERPRINT = "f" * 64
KEEP = d.SetOrientation(orientation="sample_major")
EXCLUDE = d.SetExclusions(rules=[d.ExclusionRule(column="x", low=0, high=10,
                                                 reason="implausible values")])
GRAPH = build_graph()


def record(seq: int, decision, rid: str | None = None, *, post: bool = False) -> DecisionRecord:
    return DecisionRecord(id=rid or f"r{seq}", seq=seq, at=datetime.now(timezone.utc),
                          decision=decision, post_seal=post)


def fit_of(records) -> str:
    """The fit the engine serves for the answers as the log ends: its key."""
    return seal.keys_for(GRAPH, d.fold(records), FINGERPRINT)["fit"]


def fit_record(key: str, purpose: str = "prediction", decisions=(1, 2)) -> FitRecord:
    return FitRecord(id="r" * 32, seq=1, at="2026-10-27T00:00:00Z", purpose=purpose, target="y",
                     fit_key=key, plan_digest="p", decisions=list(decisions), press=None,
                     families=["linear"], seconds=1.0, workers=2, primary_metric="mse",
                     folds=None, seed=0, held_out="rows-a")


SERVICE = SimpleNamespace(engine=SimpleNamespace(graph=GRAPH), fingerprint=lambda pid: FINGERPRINT,
                          _held_out=lambda pid, key: "rows-a")


# ── after the opening: a record whose fit artifact is not kept ───────────────


def opened() -> list[DecisionRecord]:
    """A project opened with its orientation never answered (``test_orientation_after_the_opening``)."""
    return [record(1, d.SetLens(lenses=["dietary"])), record(2, d.SetTarget(column="y")),
            record(3, d.SetPurpose(purpose="prediction")),
            record(4, d.SetRoles(roles={"x": "covariate"})),
            record(5, d.SetSplit(holdout=0.2, seed=0, folds=5), "split"),
            record(6, d.SelectModels(models=["linear"])),
            record(7, d.OpenSeal(family="linear", target="y", n_holdout=40, metric="mse",
                                 scores={"linear": {"mse": 0.7}}), "open")]


def evicted(records, key: str) -> dict:
    """The fits list's record of the fit at ``key`` once its artifact is no longer kept."""
    drawn = ProjectService._opened_draw(SERVICE, "p", records)
    return _served_record(fit_record(key, decisions=(1, 7, 8)), None, opened=drawn)


def test_an_evicted_fit_made_while_the_kept_layout_stood_did_not_change_after_the_opening():
    kept = [*opened(), record(8, KEEP, "keep", post=True)]
    made = fit_of(kept)
    assert made != fit_of(opened())  # the engine keys the two apart
    assert evicted(kept, made)["changed_after_seal"] is False
    reverted = [*kept, record(9, d.Revert(decision_id="keep"), "back", post=True)]
    assert evicted(reverted, made)["changed_after_seal"] is False
    assert evicted(reverted, fit_of(reverted))["changed_after_seal"] is False  # the opened fit


def test_an_evicted_fit_made_after_an_exclusion_changed_after_the_opening():
    excluded = [*opened(), record(8, KEEP, "keep", post=True), record(9, EXCLUDE, "x", post=True)]
    assert evicted(excluded, fit_of(excluded))["changed_after_seal"] is True


# ── under Estimate: the fit the plan's lock was recorded on ──────────────────


def estimate() -> list[DecisionRecord]:
    """An inference project whose plan was locked with the orientation never answered."""
    return [record(1, d.SetLens(lenses=["dietary"])), record(2, d.SetTarget(column="y")),
            record(3, d.SetPurpose(purpose="inference")),
            record(4, d.SetRoles(roles={"x": "covariate"})),
            record(5, d.SetSplit(holdout=0.0, seed=0, folds=5), "split"),
            record(6, d.SelectModels(models=["linear"])),
            record(7, d.LockPlan(plan={}, digest="lock"), "lock")]


def locked(records) -> frozenset[str]:
    from turbotab.core import plan_lock

    return ProjectService._locked_fit(SERVICE, "p", records, plan_lock.current_lock(records))


def test_a_fit_made_after_keeping_the_table_as_it_is_after_the_lock_is_the_locked_fit():
    base = estimate()
    kept = [*base, record(8, KEEP, "keep")]
    made = fit_of(kept)
    assert made != fit_of(base)
    assert made in locked(kept) and fit_of(base) in locked(kept)
    reverted = [*kept, record(9, d.Revert(decision_id="keep"), "back")]
    assert made in locked(reverted) and fit_of(reverted) in locked(reverted)


def test_a_fit_made_with_the_kept_layout_before_the_lock_is_the_locked_fit_and_served_so():
    """Keep, Fit, revert the keep, then the lock: the fit made while the keep stood is the locked
    fit, though its plan (the orientation answered) is not the lock's and the lock is not among its
    records, so its record's gate serves it as the lock's."""
    from turbotab.core import plan_lock

    base = estimate()
    before = [*base[:6], record(7, KEEP, "keep")]
    made = fit_of(before)
    records = [*before, record(8, d.Revert(decision_id="keep"), "back"),
               record(9, d.LockPlan(plan={}, digest="lock"), "lock")]
    keys = locked(records)
    assert made in keys
    state = SimpleNamespace(purpose="inference", plan_locked=True, target="y")
    lock = plan_lock.current_lock(records)
    rec = fit_record(made, purpose="inference", decisions=(1, 2, 3, 4, 5, 6, 7))
    assert ProjectService._record_gate(SimpleNamespace(replay=False), rec, state, None, None,
                                       lock, keys) is None


def test_a_fit_made_after_an_exclusion_after_the_lock_is_not_the_locked_fit():
    excluded = [*estimate(), record(8, KEEP, "keep"), record(9, EXCLUDE, "x")]
    assert fit_of(excluded) not in locked(excluded)
    assert not locked(estimate()[:6])  # no lock, no locked fit
