"""Part D — the record-versus-draw sweep, at the place it recurses.

Last loop cost a critical because nine tests asserted a seal's **record** and
none asserted the **draw** against it: the escape hatch recorded `grouped`, split
by row, and told the user no subject appeared on both sides. The sweep this loop
was asked for is *"look where else an effect is a real computation and the test
only reads the record"* — the seal, the invalidation cascade, `apply`,
aggregation, and the deferred-transform byte-identical claim.

## What the sweep found, and it is not where it was pointed

Those five effects **are** read back. `test_a_stated_repeat_seals_grouped_and_
leaks_nobody` intersects the drawn subject sets. `test_applying_changes_the_
table_and_reverting_restores_it` compares frames.
`test_the_repeated_measures_chain_fires_only_when_it_should` asserts 600 rows
become 300, that `participant_id` is unique afterwards, and that one
participant's mean is the mean. Aggregation, `apply` and the seal are guarded on
the effect.

The gap is one level in — **in the instrument**.

`turbotab/devchecks.py` runs eight guards on every action of every drive. They
are the standing record-versus-draw defense: *the seal states its own basis*,
*a deferred transform leaves the table byte-identical*, *after an edit exactly
the right things are stale*, *no post-seal operation changes a surviving row's
label*. Seven of the eight are tested **only against hand-built dicts** —

    before = {"fingerprint": "aaaa", "n_rows": 600}
    moved  = {"fingerprint": "bbbb", "n_rows": 600}

which asserts that the comparator compares two strings. It says nothing about
whether the guard ever sees a real drive: whether the field it reads is
populated, whether the `kind` the API emits is the `kind` its contract table is
keyed on, whether the shape it expects is the shape `_dev_state` produces. A
guard inert for any of those reasons is **indistinguishable from a guard that
works**, because both produce silence on a clean drive — and silence is what a
clean drive is supposed to produce.

That is `FEATURE_PARITY.md`'s *a check nothing triggers is a check that does not
exist*, applied to the checks themselves. It is also the record-versus-draw
shape exactly: the unit tests assert the guard's **verdict on constructed
input**; nothing asserted its **behavior against the real computation**.

One guard had the treatment already —
`test_a_planted_wrong_number_in_a_disclosure_is_caught_over_http` — and its own
docstring records that writing it found a bug in the guard that made it unable
to fail. That is the evidence that this is worth doing seven more times, not
one: the one guard anybody planted against was broken.

## The method

Plant a real defect in the **real code path**, drive the **real API**, and assert
the violation fires. Not a constructed `before`/`after` pair — a monkeypatch on
the function the app actually calls, so the guard has to survive the real
payload shape, the real action kinds, and the real ordering.

## A hypothesis this sweep raised and then killed

`a_deferred_transform_leaves_the_table_byte_identical` opens with
`if before.get("fingerprint") and …`, so an absent fingerprint makes it return
`[]` silently. `grep fingerprint turbotab/api.py` returns nothing, which reads
exactly like the field never being populated — a guard permanently inert on
every drive.

It is populated: `fingerprint` is written by `project.to_dict()`, and
`_dev_state` is a JSON round trip of `_payload`, so it arrives. Recorded here
rather than dropped, because *"it is gone"* is a finding and gets the same
scepticism as any other claim — and because the next reader will run the same
grep.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import devchecks                                        # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


@pytest.fixture()
def on(tmp_path, monkeypatch):
    """The harness, on, writing to a throwaway directory.

    The same fixture `test_the_harness_reports_and_does_not_stop_the_drive.py`
    uses, deliberately: a session that differs from the one a real drive
    produces would be a fixture testing the fixture.
    """
    monkeypatch.setenv(devchecks.ENV_FLAG, "1")
    session = devchecks.reset_for_test(tmp_path / "drive")
    yield session
    devchecks.stop_listening()
    devchecks.reset_for_test(None)


def _drive(client, fixture="dietary_recalls.csv", target="hba1c",
           group="participant_id"):
    """Upload and reach a sealed project, over HTTP, on a real fixture."""
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]

    def decide(kind, payload):
        return client.post(f"/project/{pid}/decision",
                           json={"kind": kind, "subject": "", "payload": payload})

    decide("set_target", {"column": target})
    decide("set_grain", {"answer": "people_repeat", "group_col": group})
    decide("set_repeat_kind", {"kind": "repeats"})
    decide("set_unit_of_analysis", {"unit": "record"})
    decide("set_eligibility", {"answer": "everyone"})
    return pid, decide


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _fired(on, check: str):
    return [v for v in on.violations if v["check"] == check]


# ── the eight guards, and which of them anybody ever planted against ─────────

# Which guards have a test that plants a real defect in the real code path and
# drives the real API, and which are still only asserted against hand-built
# dicts. **Declared, not counted.**
#
# The first version of this file counted them by scanning test files for guard
# names — and this file mentions all eight, so it counted itself and reported
# 8 of 8. A metric that improves by being written is the thing this project has
# a corollary about; the register pattern is the fix, here as elsewhere.
PLANTED = {
    "every_number_displayed_traces_to_the_record":
        "test_a_planted_wrong_number_in_a_disclosure_is_caught_over_http",
    "the_seal_states_its_own_basis":
        "test_a_seal_recorded_grouped_and_drawn_by_row_is_caught_on_the_real_path",
    "a_deferred_transform_leaves_the_table_byte_identical":
        "test_a_deferred_transform_that_touches_the_table_is_caught_on_the_real_path",
    "after_an_edit_exactly_the_right_things_are_stale":
        "test_a_cascade_that_does_not_fire_is_caught_on_the_real_path",
    "no_post_seal_operation_changes_a_surviving_rows_label":
        "test_a_post_seal_renumbering_is_caught_on_the_real_path",
    # L48-C. The plant is the defect this guard was built for, exactly: a live
    # decision kind with no disposition in either table. Planting it by
    # REMOVING a kind from both is the honest direction — adding a fake kind to
    # `api.py` would be planting a defect the app cannot have.
    "every_decision_kind_has_a_disposition":
        "test_an_undispositioned_kind_is_caught_on_the_real_path",
}

# Still synthetic-only, each with what a planted test would have to do. Filed as
# `GUIDED-044`; named here so the gap is a row rather than a silence.
SYNTHETIC_ONLY = {
    "decision_sentences_match_their_records":
        "needs a decision whose recorded sentence disagrees with its own "
        "payload — plantable, but every writer of a sentence is a different "
        "method, so it is several plants rather than one",
    "a_finding_claiming_n_features_names_n":
        "needs a finding whose title states a count its affected_columns does "
        "not match; the engine composes both from one list, so planting it "
        "means patching a finding after diagnosis rather than a code path",
    "every_decision_taken_appears_in_the_record":
        "needs an accepted action that records nothing, which `apply_fix_quietly` "
        "now legitimately does inside a bulk repair — so the plant has to be "
        "distinguished from the sanctioned case first",
}


def test_every_guard_is_classified_as_planted_or_filed():
    """The register rule, applied to the harness's own guards.

    A guard added without a real-path test is the exact silence this file
    exists about, so a new one fails here until somebody says which it is. The
    check is about EXISTENCE, not quality — it cannot tell a good plant from a
    weak one, and that is the property that was missing.
    """
    import re
    guards = re.findall(r"vs \+= _guard\((\w+)",
                        Path(devchecks.__file__).read_text())
    assert guards, "no guards found; the battery moved"
    unclassified = [g for g in guards
                    if g not in PLANTED and g not in SYNTHETIC_ONLY]
    assert not unclassified, (
        f"these guards are neither planted against nor filed: {unclassified}. "
        f"Add a real-path test, or an entry in SYNTHETIC_ONLY saying what one "
        f"would have to do.")
    assert len(PLANTED) == 6 and len(SYNTHETIC_ONLY) == 3, (
        "the split moved without the ledger note moving with it")


@pytest.mark.parametrize("guard,test_name", sorted(PLANTED.items()))
def test_the_named_planted_test_exists(guard, test_name):
    """`PLANTED` is a claim, and a claim needs checking.

    A register that names a test which does not exist is worse than no
    register: it reports coverage that nothing provides.
    """
    here = Path(__file__).parent
    found = any(f"def {test_name}(" in p.read_text()
                for p in here.glob("test_*.py"))
    assert found, f"{guard} claims {test_name}, which does not exist"


# ── 1 · the seal states its own basis ────────────────────────────────────────

def test_a_seal_recorded_grouped_and_drawn_by_row_is_caught_on_the_real_path(
        on, monkeypatch):
    """`GUIDED-036`'s defect, planted in the real code and driven.

    The escape hatch recorded `grouped`, split by row, and told the user no
    subject appeared on both sides. That instance is fixed. This asserts the
    STANDING guard would catch it again from any other source — which is the
    difference between fixing an instance and closing a class.
    """
    from turbotab import engine

    real = engine.draw_holdout

    def bugged(df, target, task, grain, **kw):
        drawn = real(df, target, task, grain, **kw)
        # The seal keeps saying `grouped`; the draw stops being grouped. A row
        # split on a repeating table puts subjects on both sides.
        drawn = dict(drawn)
        # EVERY OTHER ROW. The first n labels would take whole subjects and be
        # a correct grouped split — the planted bug has to be a real leak, or
        # the probe verifies that the guard stays quiet about nothing.
        n = len(drawn["labels"])
        drawn["labels"] = list(df.index[::2][:n])
        return drawn

    monkeypatch.setattr(engine, "draw_holdout", bugged)
    client = _client()
    pid, decide = _drive(client)
    sealed = decide("seal", {"fraction": 0.15, "seed": 42})
    assert sealed.status_code == 200, sealed.text

    body = sealed.json()
    assert body["lockbox"]["seal_basis"] == "grouped", (
        "the record no longer claims a grouped seal, so this test is not about "
        "what it says it is about")

    caught = _fired(on, "a_grouped_seal_has_subjects_on_both_sides")
    assert caught, (
        "the record says `grouped` and the draw put subjects on both sides, "
        "and no guard fired on the real path — which is exactly the shape that "
        "cost a critical last loop")


# ── 2 · a deferred transform leaves the table byte-identical ────────────────

def test_a_deferred_transform_that_touches_the_table_is_caught_on_the_real_path(
        on, monkeypatch):
    """Constitution §06's canonical leak, planted where it would really happen.

    The existing unit test feeds the guard `"aaaa"` and `"bbbb"`. This makes a
    real `defer_feature` actually mutate the real working table and asserts the
    guard notices — which is what the check claims to do and what nothing had
    ever asked it to do.
    """
    from turbotab import project as project_mod

    real = project_mod.AnalysisProject.defer_feature

    def bugged(self, key, columns, *a, **kw):
        out = real(self, key, columns, *a, **kw)
        # The leak, in its most plausible form: the declaration is recorded AND
        # the transform is helpfully materialized on the working table.
        frame = self.df.copy()
        for column in columns:
            if column in frame.columns:
                values = frame[column]
                frame[column] = values - values.mean()
        self.df = frame
        return out

    monkeypatch.setattr(project_mod.AnalysisProject, "defer_feature", bugged)
    client = _client()
    pid, decide = _drive(client)
    decide("seal", {"fraction": 0.15, "seed": 42})
    r = decide("defer_feature", {"transform": "standardize",
                                 "columns": ["energy_kcal"]})
    assert r.status_code == 200, r.text

    caught = _fired(on, "a_deferred_action_changed_the_working_table")
    assert caught, (
        "a declared-only transform rewrote the working table and the guard did "
        "not fire on the real path — the canonical preprocessing leak, "
        "invisible because the table simply has better numbers in it")


def test_the_byte_identical_guard_actually_receives_a_fingerprint(on):
    """The hypothesis this sweep raised, killed with evidence.

    The guard returns `[]` when `before["fingerprint"]` is falsy, and
    `grep fingerprint turbotab/api.py` finds nothing — which reads exactly like
    a guard permanently inert on every drive. It is not: `to_dict` writes the
    field and `_dev_state` round-trips it.

    Asserted rather than left as a note, because the next reader runs the same
    grep, and because the day `to_dict` stops writing it every drive goes
    silently unguarded.
    """
    client = _client()
    pid, decide = _drive(client)
    decide("seal", {"fraction": 0.15, "seed": 42})
    decide("defer_feature", {"transform": "standardize",
                             "columns": ["energy_kcal"]})

    # The guard's input is `_dev_state`, not the action log — the log records
    # the request, and `before`/`after` are handed to the guards directly.
    from turbotab.api import _dev_state
    state = _dev_state(pid)
    assert state, "the harness could not snapshot the project at all"
    assert state.get("fingerprint"), (
        "the guard's input carries no fingerprint, so "
        "`a_deferred_transform_leaves_the_table_byte_identical` returns [] on "
        "every action of every drive and cannot ever fire")
    assert len(state["fingerprint"]) > 16, (
        f"the fingerprint is not a content hash: {state['fingerprint']!r}")


# ── 3 · after an edit exactly the right things are stale ────────────────────

def test_a_cascade_that_does_not_fire_is_caught_on_the_real_path(on, monkeypatch):
    """Too little invalidation is a wrong number downstream.

    Planted by making the real `_mark_stale` a no-op — the most plausible way
    this breaks, since it is one call at the end of a method and forgetting it
    raises nothing.
    """
    from turbotab import project as project_mod

    monkeypatch.setattr(project_mod.AnalysisProject, "_mark_stale",
                        lambda self, why: None)
    client = _client()
    pid, decide = _drive(client)
    decide("seal", {"fraction": 0.15, "seed": 42})
    r = decide("add_feature", {"transform": "ratio",
                               "columns": ["energy_kcal", "protein_g"]})
    assert r.status_code in (200, 400), r.text

    if r.status_code == 200:
        caught = _fired(on, "the_cascade_fired_the_wrong_number_of_times")
        assert caught, (
            "an edit that should have marked one thing stale marked nothing, "
            "and the guard did not fire on the real path")


# ── 4 · no post-seal operation changes a surviving row's label ──────────────

def test_a_post_seal_renumbering_is_caught_on_the_real_path(on, monkeypatch):
    """Decision A's identity barrier, watched rather than remembered.

    A sealed lockbox holds row LABELS. An operation that renumbers afterwards
    leaves the lockbox perfectly well-formed and naming different rows, and
    nothing downstream can detect it — which is why there is a guard, and why
    the guard needed to be planted against.
    """
    from turbotab import project as project_mod

    real = project_mod.AnalysisProject.route_missingness

    def bugged(self, *a, **kw):
        out = real(self, *a, **kw)
        # RELABELED, not `reset_index(drop=True)`. On a fixture that already
        # has a RangeIndex, resetting the index leaves every label naming the
        # same row — so the planted bug would be a no-op and the probe would
        # report the guard as broken when the guard was never reached. The
        # class is "a surviving row ends up with a different name", and this is
        # the smallest thing that really does it.
        frame = self.df.copy()
        frame.index = [f"r{i}" for i in range(len(frame))]
        self.df = frame
        return out

    monkeypatch.setattr(project_mod.AnalysisProject, "route_missingness", bugged)
    client = _client()
    with open(DATA / "clinic_visits.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("clinic_visits.csv", fh, "text/csv")}).json()["id"]

    def decide(kind, payload, subject=""):
        return client.post(f"/project/{pid}/decision",
                           json={"kind": kind, "subject": subject,
                                 "payload": payload})

    decide("set_target", {"column": "outcome"})
    decide("set_grain", {"answer": "one_row_per_person"})
    decide("set_eligibility", {"answer": "everyone"})
    sealed = decide("seal", {"fraction": 0.2, "seed": 7})

    # THE PRECONDITIONS, asserted rather than assumed. The first version of this
    # test drove a grain the fixture does not have, the seal never landed, the
    # guard returned `[]` because the barrier was never raised — and the failure
    # read as "the guard is broken". A planted-defect test that does not check
    # it reached the defect reports the wrong thing in both directions.
    assert sealed.status_code == 200, sealed.text
    lockbox = sealed.json()["lockbox"]
    assert lockbox.get("labels"), "the seal drew no labels to invalidate"
    assert sealed.json()["barrier_raised"] is True

    cards = client.get(f"/project/{pid}/evidence/missingness").json()["cards"]
    assert cards, "no missingness card on this fixture to route"
    column = cards[0]["column"]
    option = next(o["key"] for o in cards[0]["options"]
                  if o["key"] != "drop_rows")
    r = decide("route_missingness",
               {"column": column, "card_option": option,
                "mechanism": "not_sure"}, subject=column)
    assert r.status_code == 200, (
        f"the planted operation never ran, so the guard was never reached: "
        f"{r.text}")

    caught = _fired(on, "a_sealed_row_label_no_longer_names_a_row")
    assert caught, (
        "a post-seal operation relabeled the rows and no guard fired on the "
        "real path; the lockbox now names different rows and still looks "
        "perfectly well-formed")


# ── 6 · every decision kind has a disposition ────────────────────────────────

def test_an_undispositioned_kind_is_caught_on_the_real_path(on, monkeypatch):
    """L48-C's guard, planted where it would actually break.

    The defect this watches for is *a kind was added and nobody decided* —
    which arrives as a kind live in `api.py` and absent from both
    `ACTION_CONTRACT` and `UNCLASSIFIED`. So the plant REMOVES a real kind from
    both tables and drives that kind over HTTP, rather than adding a fake kind
    to `api.py`: a plant has to be a defect the app can actually have, and an
    invented decision kind is not one.

    `set_target` is the subject because the drive posts it first, so the
    violation lands on the real transition rather than on a synthetic one.
    """
    contract = dict(devchecks.ACTION_CONTRACT)
    contract.pop("set_target")
    monkeypatch.setattr(devchecks, "ACTION_CONTRACT", contract)
    unclassified = dict(devchecks.UNCLASSIFIED)
    unclassified.pop("set_target", None)
    monkeypatch.setattr(devchecks, "UNCLASSIFIED", unclassified)

    client = _client()
    with open(DATA / "dietary_recalls.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("dietary_recalls.csv", fh, "text/csv")}).json()["id"]
    got = client.post(f"/project/{pid}/decision", json={
        "kind": "set_target", "subject": "", "payload": {"column": "hba1c"}})
    assert got.status_code == 200, got.text

    caught = _fired(on, "a_decision_kind_has_no_disposition")
    assert caught, (
        "a live decision kind was dispositioned nowhere and the drive ran "
        "clean. All three contract checks `.get(kind)` and return [], so an "
        "undispositioned kind is silently unchecked — which is the whole of "
        "`GUIDED-180`")
    assert "set_target" in caught[0]["message"], caught[0]["message"]

    # AND THE OTHER DIRECTION, on the same real path: a kind that IS
    # dispositioned does not fire. A guard that fires on a correct drive gets
    # switched off within a day.
    quiet = [v for v in on.violations
             if v["check"] == "a_decision_kind_has_no_disposition"
             and "set_target" not in v["message"]]
    assert not quiet, (
        f"the guard fired on kinds that ARE dispositioned: "
        f"{[v['message'] for v in quiet]}")
