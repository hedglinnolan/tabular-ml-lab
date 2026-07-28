"""Every specification clause maps to a passing test or an open finding.

`FEATURE_PARITY.md`: *a specification clause with neither an implementation nor
a tracked row is the same silence the register exists to prevent.*

The register tracks what the doors **do**; the ledger tracks what they
**should**. Keeping them apart is right — L13 tried to file "the impossibility
pass runs before the seal" as a register row and `register.py check` refused it,
because every valid state answers *which door has this* and neither door did.
But that leaves a third possibility neither artifact covers: a clause that was
written, agreed, and then landed nowhere. No implementation, no finding, nobody
lying, and the clause simply not happening.

This file closes that gap. It reads the clause headings out of the two
constitutions in `ROADMAP.md` and out of `ASSEMBLY_SPEC.md`, and fails on any
clause that names neither a test nor an open finding.

Deliberately dumb about QUALITY and precise about EXISTENCE. It cannot tell a
good test from a weak one, and it must not try: a clause with one honest open
finding is tracked, a clause with four vague tests may not be, and counting them
would reward the wrong thing. What was missing was existence, so existence is
what it checks.

Two things it must never become:

* **Not a coverage percentage.** See above.
* **Not a reason to write a stub test.** A clause nobody has built gets an OPEN
  FINDING, which is the truthful record. Satisfying this check with an empty
  test would be the silence rewritten as a green line — the `KNOWN_GAP_`
  problem one level up.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DOCS = os.path.join(PROJECT_ROOT, "docs", "turbotab")
FINDINGS = os.path.join(DOCS, "data", "findings.json")


# ─────────────────────────────────────────────────────────────────────────────
# The manifest. One entry per clause; each names tests, findings, or an
# exemption with its reason. Adding a clause to a constitution and not to this
# table is itself a failure — that is the point.
# ─────────────────────────────────────────────────────────────────────────────

# `tests` are (path, function-name-fragment) pairs. The fragment must appear as
# a `def <fragment>` in the file, so a renamed or deleted test breaks the link
# rather than leaving it dangling.
CLAUSES: Dict[str, Dict] = {
    # ── the routing constitution (ROADMAP.md, Decision B) ────────────────────
    "routing-01-fact": dict(
        title="Fact → skippable at high confidence",
        tests=[("ml/router.py", "_skip_is_permitted"),
               ("tests/test_router.py", "test_")],
    ),
    "routing-02-choice": dict(
        title="Choice → always asked",
        tests=[("ml/router.py", "audit"), ("tests/test_router.py", "test_")],
    ),
    "routing-03-consequence": dict(
        title="Consequence → always asked, exit is a recorded decision",
        tests=[("ml/router.py", "acknowledgment_required"),
               ("tests/test_router.py", "test_")],
    ),

    # ── the lockbox constitution (ROADMAP.md §01–§08) ────────────────────────
    "lockbox-01": dict(
        title="The pre-seal sequence is fixed",
        tests=[("turbotab/test_grain_is_asked.py",
                "test_the_seal_cannot_be_drawn_before_the_grain_is_answered")],
        findings=["STATE-101"],
    ),
    "lockbox-02": dict(
        title="Grain is asked, never inferred",
        tests=[("turbotab/test_grain_is_asked.py",
                "test_the_contradiction_detector_fires_on_an_id_name_the_heuristic_misses")],
        findings=["IMPORT-020", "IMPORT-022"],
    ),
    "lockbox-03": dict(
        title="The seal states its own basis — three states, never two",
        tests=[("tests/test_the_seal_states_its_basis.py",
                "test_an_undetermined_seal_is_never_recorded_as_cross_sectional")],
    ),
    "lockbox-04": dict(
        title="Eligibility and robustness trims are different objects",
        tests=[("tests/test_the_trim_does_not_touch_the_sealed_rows.py",
                "test_every_sealed_row_survives_a_post_seal_trim")],
    ),
    "lockbox-05": dict(
        title="The extrapolation obligation fires at the report, not at the trim",
        # Nothing. Filed at L14 rather than stubbed — see the finding.
        findings=["STATE-103"],
    ),
    "lockbox-06": dict(
        title="Declaration and execution are separate, execution bound to a data scope",
        # The test arrives when L14 builds the Features step; until then the
        # open finding is what tracks it, which is the lifecycle this check is
        # designed around — never a stub test to make the line green.
        tests=[("turbotab/test_features_are_declared_and_deferred.py",
                "test_a_stateful_transform_is_recorded_not_materialized")],
        findings=["GUIDED-012"],
    ),
    "lockbox-07": dict(
        title="Missingness routes by dtype and mechanism",
        tests=[("tests/test_missingness_encoding.py", "test_")],
    ),
    "lockbox-08": dict(
        title="What this does not settle",
        exempt="Explicitly a non-clause. It enumerates what the constitution "
               "declines to decide — no source gives a missingness rate at "
               "which an indicator beats imputation, mechanism stability is "
               "unverifiable at build time. Requiring a test would be "
               "requiring a test that an open question is still open.",
    ),

    # ── ASSEMBLY_SPEC.md §01–§08 ─────────────────────────────────────────────
    "assembly-01": dict(
        title="Scope and gating",
        findings=["IMPORT-014", "IMPORT-015"],
    ),
    "assembly-02": dict(
        title="What the research established",
        exempt="A literature review, not a requirement. It records what Power "
               "Query, Tableau Prep, Alteryx and the rest do, and the ~30% "
               "spreadsheet error base rate. Its conclusions become clauses "
               "§03–§07; those carry the obligations.",
    ),
    "assembly-03": dict(
        title="The interaction — intent, grain, relationship, row-accounting receipt",
        findings=["IMPORT-201", "IMPORT-204", "IMPORT-237"],
    ),
    "assembly-04": dict(
        title="Question grammar (DESIGN_LANGUAGE §09)",
        tests=[("turbotab/test_question_grammar.py", "test_")],
    ),
    "assembly-05": dict(
        title="The grain question is shared with the lockbox",
        tests=[("turbotab/test_grain_is_asked.py",
                "test_the_basis_source_is_reachable_for_assembly")],
    ),
    "assembly-06": dict(
        title="What the dynamic surface changes",
        findings=["IMPORT-258"],
    ),
    "assembly-07": dict(
        title="Acceptance criteria — from the audit, not invented",
        findings=["IMPORT-001", "IMPORT-005", "IMPORT-006", "IMPORT-007",
                  "IMPORT-011", "IMPORT-015", "IMPORT-017"],
    ),
    "assembly-08": dict(
        title="Open questions",
        exempt="Named as open. Same reasoning as lockbox-08.",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────

def _headings() -> Dict[str, List[str]]:
    """Clause headings as they appear in the source documents."""
    roadmap = open(os.path.join(DOCS, "ROADMAP.md"), encoding="utf-8").read()
    spec = open(os.path.join(DOCS, "ASSEMBLY_SPEC.md"), encoding="utf-8").read()
    lockbox_block = roadmap[roadmap.index("## The lockbox constitution"):]
    lockbox_block = lockbox_block[:lockbox_block.index("\n## ", 1)]
    return {
        "lockbox": re.findall(r"^### (\d\d) · ", lockbox_block, re.M),
        "assembly": re.findall(r"^## (\d\d) · ", spec, re.M),
    }


def _findings_by_id() -> Dict[str, Dict]:
    data = json.load(open(FINDINGS, encoding="utf-8"))
    rows = data["findings"] if isinstance(data, dict) and "findings" in data else data
    return {r["id"]: r for r in rows}


def test_every_clause_in_the_source_documents_is_in_the_manifest():
    """A clause added to a constitution and not to this table is a failure.

    This is the half that makes the check survive: without it, a new clause
    simply would not be looked at, which is the silence again.
    """
    found = _headings()
    missing = []
    for num in found["lockbox"]:
        if f"lockbox-{num}" not in CLAUSES:
            missing.append(f"lockbox constitution §{num}")
    for num in found["assembly"]:
        if f"assembly-{num}" not in CLAUSES:
            missing.append(f"ASSEMBLY_SPEC §{num}")
    assert not missing, (
        "these clauses exist in the documents and are not tracked here:\n  "
        + "\n  ".join(missing)
        + "\nAdd each to CLAUSES with a test, a finding, or an exemption and "
          "its reason.")


def test_no_manifest_entry_points_at_a_clause_that_no_longer_exists():
    """The other direction. A clause renumbered or deleted leaves this table
    asserting about nothing, which is a stale pointer — the family this project
    already has a rule about."""
    found = _headings()
    live = ({f"lockbox-{n}" for n in found["lockbox"]}
            | {f"assembly-{n}" for n in found["assembly"]}
            | {"routing-01-fact", "routing-02-choice", "routing-03-consequence"})
    stale = [k for k in CLAUSES if k not in live]
    assert not stale, (
        f"these manifest entries name clauses that are not in the documents: "
        f"{stale}. Renumbered, or removed?")


def test_every_clause_a_router_question_claims_is_a_real_clause():
    """`Question.clause` exempts a question from `irrelevant_net`, so it needs a
    check or it is a laundering mechanism.

    A question that names `lockbox-99` — or `lockbox-2`, or a clause that was
    renumbered — would be counted as constitutional and subtracted, and nothing
    else in the codebase reads the field. The claim has to resolve against the
    documents, which is the same identity rule the manifest runs on.
    """
    from ml import router

    found = _headings()
    live = ({f"lockbox-{n}" for n in found["lockbox"]}
            | {f"assembly-{n}" for n in found["assembly"]}
            | {"routing-01-fact", "routing-02-choice", "routing-03-consequence"})

    claimed = set()
    for step in router.STEPS:
        for q in router.plan([], target="outcome", step=step,
                             detection={"detected": "classification",
                                        "confidence": "high", "reasons": []}):
            if q.clause:
                claimed.add((q.key, q.clause))

    assert claimed, (
        "no Router question names a clause, so this check is vacuous. The grain "
        "question carries `lockbox-02`; if that was removed, the constitutional "
        "category has nothing to count and `irrelevant_net` silently equals "
        "`irrelevant_questions`.")
    unknown = sorted(f"{k} claims {c}" for k, c in claimed if c not in live)
    assert not unknown, (
        f"these Router questions name clauses that are not in the documents: "
        f"{unknown}. A clause label that resolves to nothing still exempts the "
        "question from `irrelevant_net`.")


@pytest.mark.parametrize("key", sorted(CLAUSES))
def test_the_clause_names_a_test_or_an_open_finding(key):
    entry = CLAUSES[key]
    if entry.get("exempt"):
        assert len(entry["exempt"]) > 40, (
            f"{key} claims an exemption without a real reason. An exemption is "
            f"an argument, not a keyword.")
        return

    reasons: List[str] = []

    for rel, fragment in entry.get("tests", []):
        path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.exists(path):
            reasons.append(f"{rel} does not exist")
            continue
        text = open(path, encoding="utf-8").read()
        if f"def {fragment}" not in text:
            reasons.append(f"{rel} has no `def {fragment}`")
            continue
        reasons.append("")                      # a live link

    by_id = _findings_by_id()
    for fid in entry.get("findings", []):
        row = by_id.get(fid)
        if row is None:
            reasons.append(f"{fid} is not in the ledger")
            continue
        if row["status"] not in ("OPEN", "PARTIAL"):
            reasons.append(f"{fid} is {row['status']}, so it no longer tracks anything")
            continue
        reasons.append("")                      # a live link

    live = [r for r in reasons if r == ""]
    assert live, (
        f"clause {key} ({entry['title']}) names neither a passing test nor an "
        f"open finding.\n  " + "\n  ".join(r for r in reasons if r) +
        "\nFile a finding for it. Do NOT add a stub test — a clause nobody has "
        "built gets an open finding, which is the truthful record.")
