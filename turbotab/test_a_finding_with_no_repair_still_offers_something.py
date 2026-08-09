"""`GUIDED-031` — the one moment a decision is due was the one place with
nothing to act on.

The product owner drove the app, clicked *"Show me what this means"* on a
finding with no proposed repair, and landed in a branch that printed
`suggested_actions` as em-dash bullets and closed with *"the engine reports this
without proposing a repair."* Honest about the engine. A dead end for the user.

And the list was already **options wearing prose**:

    — Consider winsorizing or capping
    — Tree models are robust to outliers
    — Investigate if outliers are errors or genuine

Three different decisions rendered as three paragraphs. `DESIGN_LANGUAGE.md`
§01.4 — *three attributes wearing a sentence costume* — the critique that
started this project, surviving the rewrite by moving branch.

Run:  ./venv/bin/python -m pytest \\
          turbotab/test_a_finding_with_no_repair_still_offers_something.py -q
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import actions as A, api                                # noqa: E402
from turbotab.project import ProjectError                             # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    return TestClient(api.app)


def _driven(client, fixture: str, target: str) -> str:
    with open(DATA / f"{fixture}.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": (f"{fixture}.csv", fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    return pid


def _finding(client, pid: str, category: str) -> str:
    body = client.get(f"/project/{pid}").json()
    for f in body["findings"]:
        if (f.get("params") or {}).get("category") == category:
            return f["id"]
    pytest.skip(f"no {category} warning on this fixture")


# ── the classification is complete, and stays complete ───────────────────────

def test_every_suggested_action_the_engine_can_emit_is_classified():
    """The guard that stops a new suggestion arriving as a bullet again.

    Walks `ml/dataset_profile.py` for every literal it can put in
    `suggested_actions` and requires the table to know it. A phrase this does
    not know still renders — as the prose it always was — so the failure is
    a gap rather than a disappearance, and this is what stops the gap being
    permanent.
    """
    tree = ast.parse((ROOT / "ml" / "dataset_profile.py").read_text())
    phrases = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "DataWarning"):
            continue
        for kw in node.keywords:
            if kw.arg != "suggested_actions" or not isinstance(kw.value, ast.List):
                continue
            for element in kw.value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    phrases.append(element.value)

    assert len(phrases) > 20, "the walk found nothing; the search is wrong"
    unknown = sorted({p for p in phrases if A.classify(p) is None})
    assert not unknown, (
        "the engine can emit these and the table does not classify them:\n  "
        + "\n  ".join(unknown)
        + "\n\nEach is either an OPERATION the app can do and preview, or an "
          "EARMARK that goes to a person or to a later step. Deciding which is "
          "the work; leaving it unclassified renders it as a bullet again.")


def test_the_two_kinds_are_never_confused():
    """An operation claims the app can do this. An earmark claims it cannot.

    Getting one wrong in either direction is the governing rule broken: an
    earmark rendered as an operation offers a control that does nothing, and an
    operation rendered as an earmark hides a thing the app could have done.
    """
    assert isinstance(A.classify("Consider winsorizing or capping"), A.Operation)
    assert isinstance(A.classify("Verify units and data entry"), A.Earmark)
    assert A.classify("Verify units and data entry").is_for_a_person
    assert not A.classify("Tree models are robust to outliers").is_for_a_person


def test_the_same_phrase_with_different_examples_classifies_once():
    """*"Use regularized linear models (Ridge, Lasso)"* and *"Use regularized
    models (Ridge, Lasso, ElasticNet)"* are one suggestion with two example
    lists. A table that distinguished them would classify one decision twice
    and let the two drift."""
    a = A.classify("Use regularized linear models (Ridge, Lasso)")
    b = A.classify("Use regularized models (Ridge, Lasso, ElasticNet)")
    assert a is not None and a.key == b.key == "prefer_regularized"


def test_every_operation_binds_to_a_catalogue_that_resolves_it():
    """A binding the catalogue does not carry is an option that cannot preview
    and cannot execute — a control that does nothing, which is worse than the
    bullet it replaced."""
    from turbotab import features as F, missingness as M, recipes as R, selection as S
    for phrase, found in A.known_phrases().items():
        if not isinstance(found, A.Operation):
            continue
        if found.catalogue == A.FEATURE:
            assert F.get(found.binding) is not None
        elif found.catalogue == A.MISSINGNESS:
            assert M.strategy(found.binding) is not None
        elif found.catalogue == A.RECIPE:
            op = R.operation(found.binding)
            assert found.variant in op.variants, (
                f"{found.key}: {found.variant!r} is not a variant of "
                f"{found.binding!r}")
        elif found.catalogue == A.SELECTION:
            assert found.binding in S.METHODS
        else:                                          # pragma: no cover
            pytest.fail(f"{found.key} names an unknown catalogue")


def test_every_earmark_names_a_step_that_exists_or_a_person():
    from ml.router import STEP_LABELS
    for found in A.known_phrases().values():
        if not isinstance(found, A.Earmark):
            continue
        assert found.target_step == A.YOU or found.target_step in STEP_LABELS, (
            f"{found.key} resurfaces at {found.target_step!r}, which is not a "
            f"step — an earmark with nowhere to go is a discard with manners")


# ── the two findings the product owner hit ───────────────────────────────────

def test_the_outlier_finding_offers_an_operation_and_three_earmarks(client):
    """One of the two cards the drive landed on.

    The engine's four suggestions were four paragraphs. They are one thing the
    app does, one thing only a person can decide, and two model-choice concerns
    that belong where models are chosen.
    """
    pid = _driven(client, "clinical_longitudinal", "progressed")
    fid = _finding(client, pid, "outliers")
    offers = client.get(f"/project/{pid}/finding/{fid}/offers").json()

    assert [o["key"] for o in offers["options"]] == ["winsorize"]
    assert offers["columns"], "the option has nothing to act on"

    option = offers["options"][0]
    assert option["defers"] is True, "winsorizing learns percentiles from rows"
    assert "within each training fold" in option["sentence"], (
        "clause §06's timing must be in the methods prose, not in a note about "
        "the software")

    by_key = {e["key"]: e for e in offers["earmarks"]}
    assert by_key["triage_outliers"]["is_for_a_person"] is True
    assert by_key["prefer_robust_loss"]["target_step"] == "preprocess"
    assert by_key["prefer_trees_outliers"]["target_step"] == "preprocess"
    assert offers["unclassified"] == []
    assert "1 of these the app can do" in offers["summary"]


def test_the_physiologic_finding_is_all_earmarks_and_says_so(client):
    """The other card, and the outcome the brief calls legitimate.

    Nothing here is an operation on the data — and that is a different sentence
    from a dead end, because every earmark goes somewhere and the card names
    where.
    """
    pid = _driven(client, "clinical_longitudinal", "progressed")
    fid = _finding(client, pid, "physiologic_plausibility")
    offers = client.get(f"/project/{pid}/finding/{fid}/offers").json()

    assert offers["options"] == []
    assert offers["earmarks"], "all earmarks and no earmarks is the dead end"
    assert "Nothing here is an operation on the data" in offers["summary"]

    by_key = {e["key"]: e for e in offers["earmarks"]}
    # THE ONE THAT MATTERS. Claiming the app can verify data entry would be the
    # governing rule broken in the place built to honor it.
    assert by_key["verify_units"]["is_for_a_person"] is True
    assert "Only you can" in by_key["verify_units"]["why"]
    assert by_key["review_ranges"]["target_step"] == "explore"

    # And it names the columns it is about, which the warning itself does not
    # carry — they come from the plausibility report rather than from parsing
    # the warning's prose.
    assert set(offers["columns"]) >= {"sbp", "dbp", "glucose"}


# ── the previews ─────────────────────────────────────────────────────────────

def test_a_deferred_option_previews_as_a_simulation_labeled_not_applied(client):
    """Clause §06 permits exactly one override — *a read-only preview not
    persisted to the modeling table, labeled preview, not applied.*"""
    pid = _driven(client, "clinical_longitudinal", "progressed")
    fid = _finding(client, pid, "outliers")
    before = client.get(f"/project/{pid}").json()["fingerprint"]

    r = client.get(f"/project/{pid}/finding/{fid}/offer/winsorize/preview")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applied"] is False
    # `GUIDED-162`, L47-C. This was an equality against the whole caption, and
    # the caption grew a second clause naming where the offer GOES — because the
    # non-deferring branch used to say "this is what pressing apply would do"
    # and there is no apply to press. The property this test's name carries is
    # *labeled not applied*, and that is what is asserted; the equality was
    # pinning a sentence rather than a claim.
    assert body["label_note"].startswith("preview, not applied"), body["label_note"]
    assert "training fold" in body["label_note"], (
        "a deferred option's caption no longer says where it is fitted")

    rows = body["preview"]["rows"]
    assert rows and rows[0]["column"] == "sbp"
    # The REAL computation on a copy, not a description of one.
    assert rows[0]["low"] < rows[0]["high"]
    assert rows[0]["n_touched"] > 0
    assert rows[0]["observed_max"] >= rows[0]["high"]

    # And the table is untouched, asserted against a content hash rather than
    # trusted.
    assert client.get(f"/project/{pid}").json()["fingerprint"] == before


def test_an_option_with_no_columns_refuses_rather_than_previewing_nothing(client):
    """`sample_size` is a property of the table. An option offered over "all
    400 columns because the table is small" is an option about nothing."""
    pid = _driven(client, "clinic_visits", "outcome")
    body = client.get(f"/project/{pid}").json()
    small = next((f for f in body["findings"]
                  if (f.get("params") or {}).get("category") == "sample_size"), None)
    # `AUDIT-039`. THE FIXTURE IS SHIPPED AND THE PRECONDITION IS A FACT
    # ABOUT IT, so a skip here stands down over exactly the regression the
    # test exists to catch — and pytest counts a skip as not-a-failure.
    assert small is not None, (
        "clinic_visits.csv raises no sample-size warning, and this test's whole "
        "claim is that a DATASET-level warning offers no column operation. "
        "Without one there is nothing to be right about, and a fixture that "
        "stopped raising it is the thing worth knowing")
    offers = client.get(f"/project/{pid}/finding/{small['id']}/offers").json()
    assert offers["columns"] == []
    assert offers["options"] == [], (
        "a dataset-level warning offered a column operation")


# ── an earmark goes somewhere ────────────────────────────────────────────────

def test_an_earmark_lands_in_the_record_naming_where_it_resurfaces(client):
    pid = _driven(client, "clinical_longitudinal", "progressed")
    r = client.post(f"/project/{pid}/decision", json={
        "kind": "earmark", "subject": "verify_units",
        "payload": {"key": "verify_units", "target_step": A.YOU,
                    "label": "Verify units and data entry"}})
    assert r.status_code == 200, r.text
    d = [x for x in r.json()["decisions"] if x["kind"] == "earmark"][-1]
    assert d["payload"]["for_a_person"] is True
    assert "yours" in d["text"], (
        "the record must say the app cannot do this, or it has claimed it can")

    r = client.post(f"/project/{pid}/decision", json={
        "kind": "earmark", "subject": "prefer_robust_loss",
        "payload": {"key": "prefer_robust_loss", "target_step": "preprocess",
                    "label": "Prefer a robust loss"}})
    d = [x for x in r.json()["decisions"] if x["kind"] == "earmark"][-1]
    assert d["payload"]["for_a_person"] is False
    assert "comes back at Preprocess" in d["text"]


def test_an_earmark_with_nowhere_to_go_is_refused(client):
    pid = _driven(client, "clinical_longitudinal", "progressed")
    r = client.post(f"/project/{pid}/decision", json={
        "kind": "earmark", "subject": "x",
        "payload": {"key": "x", "target_step": "nowhere", "label": "x"}})
    assert r.status_code == 400
    assert "discard with manners" in r.json()["detail"]


def test_the_branch_is_no_longer_terminal():
    """The claim in one assertion, against the page rather than the API.

    `explainOnly` used to end with a Close button and three paragraphs. It now
    renders options with previews and earmarks with destinations, and the prose
    fallback survives underneath for a phrase the table does not know — so a new
    engine suggestion degrades to the old behavior rather than vanishing.
    """
    page = (ROOT / "turbotab" / "web" / "index.html").read_text()
    assert "data-offer-preview=" in page, "no way to preview an option"
    assert "data-earmark=" in page, "no way to earmark a decision"
    assert "function offerRow(" in page and "function earmarkRow(" in page
    assert "yours to do" in page, "the page never says the app cannot do it"
    # The fallback is still there.
    assert "The engine reports this without proposing a repair" in page
