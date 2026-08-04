"""L47-B2 — `GUIDED-165`. The record said it happened and it did not happen.

The plausibility card offers *"Set these entries to missing"*. The product owner
pressed it on `bp_di` and the endpoint reported **125 flagged before and 125
after**, while the transcript carried *"Entries of bp_di outside the
impossibility band were set to missing."* `AUDIT-001`'s shape at the decision
layer, and `draft.py` escalated it into the manuscript.

The tell was the asymmetry: *"Keep as is"* recorded *"were kept as recorded"*,
which is **true**, because nothing happened either way.

## Why it could not have worked

Both buttons posted `kind="note"` with a sentence composed **in the page**, and
`api.py`'s generic tail records a note, calls no engine function and does not
`_recompute`. Nothing in `api.py` branched on the `impossible__*` subject at all.

**And `devchecks.ACTION_CONTRACT` already forbade the repair the button
promised** — `"note": _Expected(touches_table=False, ...)`, enforced by
`a_deferred_transform_leaves_the_table_byte_identical`. The contract was right
and the kind was wrong, which is why the fix is two new kinds rather than a
mutation bolted onto `note`.

## The band, which is the part that could have shipped a worse defect

`ml/preprocess_operators.PlausibilityGate` is the operator for this and it takes
its bounds rather than fetching them — so the mistake lives in the *caller*.
`ml/pipeline.py:86` builds those bounds from `get_improbability_band`, the p01/p99
pair, which is precisely what `get_impossibility_band`'s docstring forbids:

    Returns None rather than falling back to the improbability band… answering
    "unknown" with the weaker bound would silently promote improbable values to
    impossible ones and propose deleting real data.

On `clinical_longitudinal.csv` that is the difference between removing **4**
entry errors in `dbp` and removing every value outside p01–p99. `MISC-018`'s
class at a second site, and `test_the_band_is_the_impossibility_band` is what
stops it arriving.

## `GUIDED-097` — two fixtures, and the shape neither covers

`clinical_longitudinal.csv` (binary `progressed`) and `clinical_labs.csv`
(binary `readmitted`). **Not covered: a continuous target**, and a table whose
impossible tier is empty — the second is exercised synthetically below because
no shipped fixture has a numeric physiologic column with a clean impossible tier
and something else to compare against.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parent / "sample_data"

#: `(fixture, target, a column with a real impossible tier)`. Both binary — see
#: the docstring for what that does not cover.
FIXTURES = {
    "longitudinal clinic visits": ("clinical_longitudinal.csv", "progressed", "dbp"),
    "a messy multi-site lab extract": ("clinical_labs.csv", "readmitted", "sbp"),
}


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api

    return TestClient(api.app)


def _project(client, fixture, target):
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    return pid


@pytest.mark.parametrize("label", sorted(FIXTURES))
def test_the_flagged_entries_are_gone_from_the_table_afterwards(label):
    """The adjudicator's own check: drive the endpoint before and after.

    **125 before and 125 after is the failure, whatever the record says.**
    Asserted on the endpoint the card is drawn from rather than on the frame,
    because the endpoint is what the user sees re-render.
    """
    fixture, target, column = FIXTURES[label]
    client = _client()
    pid = _project(client, fixture, target)

    before = client.get(f"/project/{pid}/evidence/plausibility").json()
    block = next((b for b in before["impossible"]
                  if b["column"] == column and not b["whole_column_suspect"]), None)
    assert block, f"{label}: {column} has no repairable impossible tier any more"
    n = int(block["n_flagged"])
    assert n, f"{label}: nothing is flagged, so this proves nothing"

    posted = client.post(f"/project/{pid}/decision", json={
        "kind": "set_impossible_missing", "subject": column,
        "payload": {"column": column}})
    assert posted.status_code == 200, posted.text[:300]

    after = client.get(f"/project/{pid}/evidence/plausibility").json()
    still = [b for b in after["impossible"] if b["column"] == column]
    assert not still, (
        f"{label}: {column} is still flagged after the repair — "
        f"{still[0]['n_flagged']} entries. The record would say they were set "
        f"to missing.")
    assert after["n_impossible"] == before["n_impossible"] - n, (
        f"{label}: {before['n_impossible']} → {after['n_impossible']} for a "
        f"repair of {n}")


@pytest.mark.parametrize("label", sorted(FIXTURES))
def test_the_record_says_what_happened_and_carries_its_own_count(label):
    """The sentence, and the two devchecks it has to clear.

    The column is backticked because `devchecks.numbers_in` strips backticked
    spans before counting, so an unbackticked name carrying a digit — `bp_1`,
    `hba1c` — reads as an unsupported number. And the count is in the payload
    because a number in the sentence with no number in the payload trips the
    same check from the other side.
    """
    fixture, target, column = FIXTURES[label]
    client = _client()
    pid = _project(client, fixture, target)
    before = client.get(f"/project/{pid}/evidence/plausibility").json()
    n = next(b["n_flagged"] for b in before["impossible"] if b["column"] == column)

    body = client.post(f"/project/{pid}/decision", json={
        "kind": "set_impossible_missing", "subject": column,
        "payload": {"column": column}}).json()
    said = [d for d in body["decisions"] if d["kind"] == "set_impossible_missing"]
    assert len(said) == 1, said
    record = said[0]

    assert f"`{column}`" in record["text"], (
        f"{label}: the object is not backticked, so a digit in the column name "
        f"reads as an unsupported number: {record['text']!r}")
    assert record["payload"]["n_set"] == n, (
        f"{label}: the record claims {record['payload']['n_set']} and the card "
        f"flagged {n}")
    assert str(n) in record["text"], (
        f"{label}: the sentence states no count: {record['text']!r}")
    from turbotab import devchecks as D

    unsupported = [x for x in D.numbers_in(record["text"])
                   if x not in D.supported_numbers({"payload": record["payload"]})]
    assert not unsupported, (
        f"{label}: the sentence carries {unsupported}, which its payload does "
        f"not support")


@pytest.mark.parametrize("label", sorted(FIXTURES))
def test_keeping_them_is_a_different_kind_and_touches_nothing(label):
    """The other half.

    Both buttons used to post `kind="note"` with the same `subject`, so nothing
    machine-readable distinguished a repair from a refusal to repair and a
    consumer had to string-match prose. Now the kind carries it.
    """
    fixture, target, column = FIXTURES[label]
    client = _client()
    pid = _project(client, fixture, target)
    before = client.get(f"/project/{pid}/evidence/plausibility").json()

    posted = client.post(f"/project/{pid}/decision", json={
        "kind": "keep_impossible", "subject": column,
        "payload": {"column": column}})
    assert posted.status_code == 200, (
        f"{label}: the server rejected `keep_impossible` — {posted.text[:200]}")
    body = posted.json()
    after = client.get(f"/project/{pid}/evidence/plausibility").json()
    assert after["n_impossible"] == before["n_impossible"], (
        f"{label}: 'keep as is' changed the table")
    kinds = {d["kind"] for d in body["decisions"]}
    assert "keep_impossible" in kinds and "set_impossible_missing" not in kinds


def test_the_band_is_the_impossibility_band():
    """`MISC-018`'s class, at the site it would have arrived at next.

    `PlausibilityGate` is reusable and its band is its caller's problem:
    `ml/pipeline.py:86` hands it the **improbability** band. Reusing the operator
    and inheriting that band would turn a repair that removes a handful of entry
    errors into one that removes every value outside p01–p99.

    Asserted by comparing what the repair removed against what each band would
    remove, on a column where the two differ by an order of magnitude.
    """
    from ml import card_evidence as C
    from ml.physiology_reference import (get_impossibility_band,
                                         get_improbability_band,
                                         load_reference_bundle,
                                         match_variable_key)
    from turbotab.project import AnalysisProject

    frame = pd.read_csv(DATA / "clinical_longitudinal.csv")
    project = AnalysisProject.from_dataframe(frame, "band")
    # The `nhanes` sub-bundle, which is what `card_evidence` resolves against —
    # `load_reference_bundle()` returns a dict of bundles and passing the outer
    # one back gives `None` for every key. Copied from `card_evidence.py:382`
    # rather than rediscovered.
    reference = load_reference_bundle()["nhanes"]
    key = match_variable_key("dbp", reference)
    hard = get_impossibility_band(reference, key)
    soft = get_improbability_band(reference, key)
    assert hard and soft, "the fixture no longer exercises both tiers"
    assert hard[0] < soft[0] and hard[1] > soft[1], (
        "the tiers do not nest on this column, so the comparison below says "
        "nothing")

    values = pd.to_numeric(frame["dbp"], errors="coerce")
    would_impossible = int(((values < hard[0]) | (values > hard[1])).sum())
    would_improbable = int(((values < soft[0]) | (values > soft[1])).sum())
    assert would_improbable > would_impossible, (
        "the two bands remove the same rows here, so this cannot detect the "
        "wrong one")

    project.set_impossible_missing("dbp")
    removed = int(values.notna().sum()
                  - pd.to_numeric(project.df["dbp"], errors="coerce").notna().sum())
    assert removed == would_impossible, (
        f"the repair removed {removed} values; the impossibility band contains "
        f"{would_impossible} and the improbability band {would_improbable}. "
        f"Inheriting the weaker band is MISC-018 at a second site.")

    report = C.plausibility_report(project.working_table)
    assert any(b["column"] == "dbp" for b in report["improbable"]), (
        "every improbable value in `dbp` was deleted too, which is the failure "
        "this test exists for")


def test_a_repair_with_nothing_to_repair_refuses_rather_than_reporting_success():
    """The inverse of the defect, and it is the same defect.

    `ml/import_doctor.py`'s apply path is the precedent: when the intersection is
    empty it drops nothing and reports having dropped nothing. A repair that
    succeeds at doing nothing is what `GUIDED-165` was.
    """
    from turbotab.project import AnalysisProject, ProjectError

    frame = pd.DataFrame({"height_cm": [160.0, 170.0, 180.0],
                          "y": [0, 1, 0]})
    project = AnalysisProject.from_dataframe(frame, "clean")
    with pytest.raises(ProjectError, match="nothing to set to missing"):
        project.set_impossible_missing("height_cm")
    with pytest.raises(ProjectError, match="not a column"):
        project.set_impossible_missing("no_such_column")


def test_the_manuscript_carries_the_true_sentence_rather_than_the_promised_one():
    """The layer the defect actually mattered at.

    `GUIDED-165`'s escalation was that `draft.py` lifts the decision's text into
    the methods section, so the false sentence left the building. This asserts
    the draft carries the count and the object — i.e. that what reaches the
    manuscript is what happened.
    """
    client = _client()
    pid = _project(client, "clinical_longitudinal.csv", "progressed")
    before = client.get(f"/project/{pid}/evidence/plausibility").json()
    n = next(b["n_flagged"] for b in before["impossible"] if b["column"] == "dbp")
    client.post(f"/project/{pid}/decision", json={
        "kind": "set_impossible_missing", "subject": "dbp",
        "payload": {"column": "dbp"}})

    draft = client.get(f"/project/{pid}/draft").json()
    said = [s["text"] for section in draft["sections"]
            for s in (section.get("sentences") or [])]
    hit = [s for s in said if "impossibility band" in s]
    assert hit, (
        "the repair does not reach the draft at all, so the manuscript is "
        f"silent about a change to the table: {said[:6]}")
    assert any(f"`dbp`" in s and str(n) in s for s in hit), (
        f"the draft sentence does not name the column and the count: {hit}")
    assert not any("were set to missing" in s and str(n) not in s for s in hit), (
        f"the draft carries an uncounted claim that entries were removed: {hit}")
