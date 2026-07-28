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

Deliberately dumb about QUALITY and precise about EXISTENCE. It cannot tell a
good test from a weak one, and it must not try: a clause with one honest open
finding is tracked, a clause with four vague tests may not be, and counting them
would reward the wrong thing.

Two things it must never become:

* **Not a coverage percentage.** See above.
* **Not a reason to write a stub test.** A clause nobody has built gets an OPEN
  FINDING, which is the truthful record. Satisfying this check with an empty
  test would be the silence rewritten as a green line.

──────────────────────────────────────────────────────────────────────────────
L16 · WHY THE MATCHER WAS REBUILT
──────────────────────────────────────────────────────────────────────────────

The first version passed clause §04 — *eligibility and robustness trims are
different objects* — while the eligibility question did not exist in either
door and no row tracked it. Three separate weaknesses let that through, and all
three are the same mistake: **the link was inferred from proximity instead of
declared.**

1. **A clause was one thing.** §04 imposes TWO requirements — a pre-seal
   eligibility criterion that changes N, and a post-seal trim that never touches
   the test set. One test covered the second. Nothing could notice that the
   first had no link, because nothing had written down that there were two. So
   clauses now enumerate their **obligations**, and each obligation carries its
   own links.

2. **A fragment could be a wildcard.** `("tests/test_router.py", "test_")`
   matched every test in the file: the clause was "covered" by the file existing.
   A fragment must now resolve to **exactly one** `def`.

3. **A link was one-ended.** The manifest asserted that a test covers a clause
   and the test said nothing, so nothing could contradict it. The named test
   must now **cite the clause key in its own source** — `lockbox-04`, an
   identifier rather than a word. That is the homonym fix: searching the ledger
   for "eligib" returns nine rows and every one is about row eligibility for the
   split, a different sense entirely, which is `hba1c_proxy` matching `hba1c`
   inside the tool built to catch missing coverage.

And one distinction the rebuild added: an **implementation** pointer is not a
test. `("ml/router.py", "_skip_is_permitted")` says the clause is implemented,
which is not evidence that it is implemented correctly — the code cannot verify
itself. Implementation links are recorded under `implements` and satisfy
nothing.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DOCS = os.path.join(PROJECT_ROOT, "docs", "turbotab")
FINDINGS = os.path.join(DOCS, "data", "findings.json")


# ─────────────────────────────────────────────────────────────────────────────
# The manifest. One entry per clause; each enumerates the clause's OBLIGATIONS,
# and each obligation names tests, findings, or an exemption with its reason.
#
# Adding a clause and not adding it here is a failure. Adding an obligation to a
# clause's text and not here is the failure this file was rebuilt for.
# ─────────────────────────────────────────────────────────────────────────────

CLAUSES: Dict[str, Dict] = {
    # ── the routing constitution (ROADMAP.md, Decision B) ────────────────────
    "routing-01-fact": dict(
        title="Fact → skippable at high confidence",
        obligations={
            "a-skip-needs-high-confidence-and-a-question-of-fact": dict(
                requires="A question may be skipped only where a high-confidence "
                         "finding makes it moot, and only when it is a question "
                         "of fact rather than of choice.",
                implements=[("ml/router.py", "_skip_is_permitted")],
                tests=[("tests/test_router.py",
                        "test_a_high_confidence_detection_may_be_stated_rather_than_asked"),
                       ("tests/test_router.py",
                        "test_a_low_confidence_detection_is_always_asked")],
            ),
            "every-skip-is-visible-and-reversible": dict(
                requires="A skip carries its reason into the transcript, so it "
                         "can be read and undone.",
                tests=[("tests/test_router.py",
                        "test_audit_rejects_a_skip_that_breaks_decision_b")],
            ),
        },
    ),
    "routing-02-choice": dict(
        title="Choice → always asked",
        obligations={
            "a-repair-is-never-skipped-at-any-confidence": dict(
                requires="Whether to apply a repair is the user's, however "
                         "confident the engine is; skipping it would apply a "
                         "change nobody saw.",
                implements=[("ml/router.py", "audit")],
                tests=[("tests/test_router.py",
                        "test_a_repair_is_never_skipped_however_confident_the_engine_is")],
            ),
        },
    ),
    "routing-03-consequence": dict(
        title="Consequence → always asked, exit is a recorded decision",
        obligations={
            "a-blocker-is-pushed-never-offered": dict(
                requires="A question of consequence is always pushed and never "
                         "merely offered; offering it beside a palette is "
                         "decoration rather than gating.",
                tests=[("tests/test_router.py",
                        "test_a_leaking_column_is_pushed_not_offered"),
                       ("tests/test_router.py",
                        "test_a_blocker_is_never_skipped_even_at_high_confidence")],
            ),
            "leaving-past-one-is-a-recorded-decision": dict(
                requires="Exiting a step with a blocker unresolved writes an "
                         "acknowledgment into the record, so the manuscript can "
                         "carry it as a stated limitation.",
                implements=[("ml/router.py", "acknowledgment_required")],
                tests=[("tests/test_router.py",
                        "test_leaving_the_step_with_a_blocker_open_requires_an_acknowledgment")],
            ),
        },
    ),

    # ── the lockbox constitution (ROADMAP.md §01–§08) ────────────────────────
    "lockbox-01": dict(
        text="ef06ad85c3ee",
        title="The pre-seal sequence is fixed",
        obligations={
            "the-grain-comes-before-the-seal": dict(
                requires="The seal cannot be drawn before the grain is answered.",
                tests=[("turbotab/test_grain_is_asked.py",
                        "test_the_seal_cannot_be_drawn_before_the_grain_is_answered")],
            ),
            "eligibility-comes-before-the-seal": dict(
                requires="Eligibility sits between grain and SEAL in the fixed "
                         "sequence, so a criterion that changes N is applied "
                         "before any row is held out.",
                tests=[("turbotab/test_eligibility_is_asked.py",
                        "test_the_seal_cannot_be_drawn_before_eligibility_is_settled"),
                       ("turbotab/test_eligibility_is_asked.py",
                        "test_eligibility_cannot_be_answered_before_the_grain")],
            ),
            "the-impossibility-pass-comes-before-the-seal": dict(
                requires="Structural repairs and the impossibility pass run "
                         "pre-seal, because a split computed over corrupted "
                         "values is a worse split and impossible entries are "
                         "normally an exclusion that changes N.",
                findings=["STATE-101"],
            ),
        },
    ),
    "lockbox-02": dict(
        text="e3225f78a7e2",
        title="Grain is asked, never inferred",
        obligations={
            "the-question-is-asked-and-the-answer-is-the-users": dict(
                requires="The grain is asked once, pre-seal, and both consumers "
                         "read the one recorded answer. The heuristics are "
                         "demoted to a suggestion.",
                tests=[("turbotab/test_grain_is_asked.py",
                        "test_stating_one_row_per_person_on_a_repeating_file_is_refused_not_warned")],
                findings=["IMPORT-020"],
            ),
            "a-disagreement-earns-an-interruption": dict(
                requires="An answer that disagrees with the data's shape is "
                         "evidence somebody is wrong, and earns an interruption "
                         "— escalating on evidence of error, not on the "
                         "magnitude of the consequence.",
                tests=[("turbotab/test_grain_is_asked.py",
                        "test_the_contradiction_detector_fires_on_an_id_name_the_heuristic_misses")],
                findings=["IMPORT-022"],
            ),
        },
    ),
    "lockbox-03": dict(
        text="9ae6f4bb0448",
        title="The seal states its own basis — three states, never two",
        obligations={
            "undetermined-is-first-class-in-the-record": dict(
                requires="`undetermined` is persisted as itself, never as "
                         "`group_col: None`, which a consumer cannot tell from a "
                         "verified cross-sectional seal.",
                tests=[("tests/test_the_seal_states_its_basis.py",
                        "test_an_undetermined_seal_is_never_recorded_as_cross_sectional")],
            ),
            "it-is-never-rendered-as-a-clean-lock": dict(
                requires="An undetermined seal reaches the user as an advisory "
                         "with exploratory labeling — leaking and disclosing is "
                         "the refuse branch; leaking behind a lock icon is the "
                         "assert-something-false branch.",
                tests=[("turbotab/test_grain_is_asked.py",
                        "test_an_undetermined_seal_says_so_in_words_the_user_reads")],
            ),
        },
    ),
    "lockbox-04": dict(
        text="2c89c23e3e39",
        title="Eligibility and robustness trims are different objects",
        obligations={
            # THE ONE THE FIRST MATCHER MISSED. Nothing linked here and the
            # clause passed anyway, because the trim obligation below carried it.
            "the-eligibility-question-is-asked-in-scientific-terms": dict(
                requires="Eligibility is asked as a question about the research "
                         "question — does it restrict the population? — with the "
                         "target's distribution WITHHELD, because a criterion "
                         "chosen from the histogram is data-driven cohort "
                         "selection, which is its own publishable bias.",
                tests=[("turbotab/test_eligibility_is_asked.py",
                        "test_the_question_withholds_the_distribution_and_says_why"),
                       ("turbotab/test_eligibility_is_asked.py",
                        "test_a_categorical_column_offers_its_values_but_not_their_counts")],
            ),
            "an-exclusion-is-pre-seal-and-changes-n": dict(
                requires="An eligibility criterion applies to the whole dataset "
                         "pre-seal and changes N, reported in participant flow "
                         "with its reason.",
                tests=[("turbotab/test_eligibility_is_asked.py",
                        "test_an_exclusion_records_its_participant_flow_numbers"),
                       ("turbotab/test_eligibility_is_asked.py",
                        "test_an_exclusion_after_the_seal_is_refused_and_says_where_to_go")],
            ),
            "a-robustness-trim-never-touches-the-sealed-rows": dict(
                requires="A robustness trim applies to the training partition "
                         "only, post-seal, and trimming the test set to match is "
                         "permanently off the menu.",
                tests=[("tests/test_the_trim_does_not_touch_the_sealed_rows.py",
                        "test_every_sealed_row_survives_a_post_seal_trim")],
            ),
        },
    ),
    "lockbox-05": dict(
        text="b39f4b6cba15",
        title="The extrapolation obligation fires at the report, not at the trim",
        obligations={
            "a-train-only-trim-arms-the-obligation": dict(
                requires="The trim is a legitimate choice and earns no blocker, "
                         "but it silently ARMS a requirement — so something has "
                         "to record that it was armed.",
                tests=[("turbotab/test_the_trim_arms_the_obligation.py",
                        "test_a_train_only_trim_arms_the_extrapolation_obligation"),
                       ("turbotab/test_the_trim_arms_the_obligation.py",
                        "test_the_obligation_carries_the_numbers_the_report_cannot_recover")],
            ),
            "the-blocker-fires-at-export": dict(
                requires="At export, the absence of a stratified in-range / "
                         "out-of-range breakdown is a blocker. The Report step "
                         "does not exist yet.",
                findings=["STATE-105"],
            ),
        },
    ),
    "lockbox-06": dict(
        text="966e1a25a95e",
        title="Declaration and execution are separate, execution bound to a data scope",
        obligations={
            "the-litmus-decides-and-it-is-a-precondition": dict(
                requires="Row-local executes immediately with a receipt; "
                         "distribution-dependent is recorded now and fitted "
                         "inside training folds only. The classification is a "
                         "precondition of executing, not a convention.",
                tests=[("turbotab/test_features_are_declared_and_deferred.py",
                        "test_a_stateful_transform_is_recorded_not_materialized")],
            ),
            "the-router-defaults-to-deferral-when-unsure": dict(
                requires="Anything unsure defers, and the only permitted "
                         "override is a read-only preview not persisted to the "
                         "modeling table, labeled preview-not-applied.",
                tests=[("turbotab/test_features_are_declared_and_deferred.py",
                        "test_a_deferred_transforms_preview_is_labeled_and_shows_no_values")],
            ),
        },
    ),
    "lockbox-07": dict(
        text="de73dd3f005a",
        title="Missingness routes by dtype and mechanism",
        obligations={
            "categorical-missingness-asks-whether-it-is-informative": dict(
                requires="Ask first whether a blank could mean something; "
                         "default to an explicit Missing category or indicator; "
                         "imputing an informatively-missing field is a blocker "
                         "with typed acknowledgment, and the stability "
                         "assumption is recorded as a methods assumption.",
                findings=["GUIDED-021"],
            ),
            "the-outcome-never-enters-the-imputation-model": dict(
                requires="Numeric imputation is fitted inside the fold and never "
                         "places the outcome in the imputation model, which is a "
                         "blocker in any configuration.",
                findings=["GUIDED-021"],
            ),
        },
    ),
    "lockbox-08": dict(
        text="cd950aee76d9",
        title="What this does not settle",
        exempt="Explicitly a non-clause. It enumerates what the constitution "
               "declines to decide — no source gives a missingness rate at "
               "which an indicator beats imputation, mechanism stability is "
               "unverifiable at build time. Requiring a test would be "
               "requiring a test that an open question is still open.",
    ),

    # ── ASSEMBLY_SPEC.md §01–§08 ─────────────────────────────────────────────
    "assembly-01": dict(
        text="b31089c05391",
        title="Scope and gating",
        obligations={
            "assembly-is-gated-behind-the-single-file-path": dict(
                requires="Multi-file assembly ships behind a freeze gate; the "
                         "single-file path is the prerequisite.",
                findings=["IMPORT-014", "IMPORT-015"],
            ),
        },
    ),
    "assembly-02": dict(
        text="ec7d4bbc0e1c",
        title="What the research established",
        exempt="A literature review, not a requirement. It records what Power "
               "Query, Tableau Prep, Alteryx and the rest do, and the ~30% "
               "spreadsheet error base rate. Its conclusions become clauses "
               "§03–§07; those carry the obligations.",
    ),
    "assembly-03": dict(
        text="08527def4959",
        title="The interaction — intent, grain, relationship, row-accounting receipt",
        obligations={
            "the-join-states-what-it-did-to-the-rows": dict(
                requires="Intent, grain, relationship and a row-accounting "
                         "receipt are each asked or reported, so a join cannot "
                         "silently change what a row is.",
                findings=["IMPORT-201", "IMPORT-204", "IMPORT-237"],
            ),
        },
    ),
    "assembly-04": dict(
        text="fb022f7363fa",
        title="Question grammar (DESIGN_LANGUAGE §09)",
        obligations={
            "every-pushed-fact-names-its-consumer": dict(
                requires="A FACT that cannot name what reads its answer and what "
                         "changes as a result is a question we have no right to "
                         "ask.",
                tests=[("turbotab/test_question_grammar.py",
                        "test_every_pushed_fact_names_who_consumes_the_answer"),
                       ("turbotab/test_question_grammar.py",
                        "test_the_audit_refuses_a_fact_with_no_consumer")],
            ),
            "the-three-kinds-are-visually-distinct-without-color": dict(
                requires="Fact, choice and consequence are distinguishable "
                         "without relying on color or typography alone.",
                tests=[("turbotab/test_question_grammar.py",
                        "test_the_grammar_survives_color_removal")],
            ),
        },
    ),
    "assembly-05": dict(
        text="addb77072ed2",
        title="The grain question is shared with the lockbox",
        obligations={
            "an-assembled-project-inherits-the-answer": dict(
                requires="A project arriving through assembly has already "
                         "answered the grain question and the seal inherits it "
                         "rather than asking again.",
                tests=[("turbotab/test_grain_is_asked.py",
                        "test_the_basis_source_is_reachable_for_assembly")],
            ),
        },
    ),
    "assembly-06": dict(
        text="30f7501eda06",
        title="What the dynamic surface changes",
        obligations={
            "the-dynamic-surface-is-tracked": dict(
                requires="The surface that changes with the number of files is "
                         "recorded rather than assumed static.",
                findings=["IMPORT-258"],
            ),
        },
    ),
    "assembly-07": dict(
        text="6aa6f843b92b",
        title="Acceptance criteria — from the audit, not invented",
        obligations={
            "each-acceptance-criterion-traces-to-an-audit-finding": dict(
                requires="The seven criteria come from measured defects, not "
                         "from invention; each is tracked as its own row.",
                tests=[("tests/test_stress_regressions.py", "TestBlowUpIsRefused"),
                       ("tests/test_stress_regressions.py",
                        "TestSamePeopleDecidesGrouping"),
                       ("tests/test_stress_regressions.py",
                        "TestBlankCellsAreToldApart"),
                       ("tests/test_stress_regressions.py",
                        "TestChangeMapMatchesReality"),
                       ("tests/test_stress_regressions.py",
                        "TestConsequencesAreAboutTheStudy")],
                findings=["IMPORT-015", "IMPORT-017"],
            ),
        },
    ),
    "assembly-08": dict(
        text="bbefe87a4824",
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


def _live_clause_keys() -> set:
    found = _headings()
    return ({f"lockbox-{n}" for n in found["lockbox"]}
            | {f"assembly-{n}" for n in found["assembly"]}
            | {"routing-01-fact", "routing-02-choice", "routing-03-consequence"})


def _findings_by_id() -> Dict[str, Dict]:
    data = json.load(open(FINDINGS, encoding="utf-8"))
    rows = data["findings"] if isinstance(data, dict) and "findings" in data else data
    return {r["id"]: r for r in rows}


def _def_source(rel: str, name: str) -> Tuple[Optional[str], Optional[str]]:
    """The source of one named `def`, and why it could not be found.

    Located by AST rather than by substring, because `def test_x` is a substring
    of `def test_x_and_y` and the whole point of this rebuild is that a link must
    resolve to exactly one thing.
    """
    path = os.path.join(PROJECT_ROOT, rel)
    if not os.path.exists(path):
        return None, f"{rel} does not exist"
    text = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:                    # pragma: no cover - a broken file
        return None, f"{rel} does not parse: {exc}"

    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    hits = [n for n in ast.walk(tree) if isinstance(n, kinds) and n.name == name]
    if not hits:
        near = sorted(n.name for n in ast.walk(tree)
                      if isinstance(n, kinds) and n.name.startswith(name))
        if near:
            return None, (f"{rel} has no `def {name}` exactly; it has "
                          f"{near[:4]}. A fragment is not a name — name one.")
        return None, f"{rel} has no `def {name}`"
    if len(hits) > 1:
        return None, f"{rel} has {len(hits)} defs named {name}; the link is ambiguous"
    return ast.get_source_segment(text, hits[0]) or "", None


# ─────────────────────────────────────────────────────────────────────────────
# The checks
# ─────────────────────────────────────────────────────────────────────────────

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
    stale = [k for k in CLAUSES if k not in _live_clause_keys()]
    assert not stale, (
        f"these manifest entries name clauses that are not in the documents: "
        f"{stale}. Renumbered, or removed?")


def test_every_clause_a_router_question_claims_is_a_real_clause():
    """`Question.clause` exempts a question from `irrelevant_net`, so it needs a
    check or it is a laundering mechanism.

    A question naming `lockbox-99` — or a clause that was renumbered — would be
    counted as constitutional and subtracted, and nothing else in the codebase
    reads the field. The claim has to resolve against the documents, which is the
    same identity rule the rest of this file runs on.
    """
    from ml import router

    live = _live_clause_keys()
    claimed = set()
    for step in router.STEPS:
        for q in router.plan([], target="outcome", step=step,
                             detection={"detected": "classification",
                                        "confidence": "high", "reasons": []}):
            if q.clause:
                claimed.add((q.key, q.clause))

    assert claimed, (
        "no Router question names a clause, so this check is vacuous. If the "
        "labels were removed, the constitutional category has nothing to count "
        "and `irrelevant_net` silently equals `irrelevant_questions`.")
    unknown = sorted(f"{k} claims {c}" for k, c in claimed if c not in live)
    assert not unknown, (
        f"these Router questions name clauses that are not in the documents: "
        f"{unknown}. A clause label that resolves to nothing still exempts the "
        "question from `irrelevant_net`.")


def test_every_clause_states_its_obligations_or_its_exemption():
    """A clause is not one thing, and pretending it is one thing is how §04
    passed with half of itself untracked.

    Enumerating the obligations is the work this check exists to force. It is
    also the only part a reader can disagree with, which is why it is written
    out rather than derived — a derived list of obligations would be a parser's
    opinion about prose.
    """
    bad = []
    for key, entry in CLAUSES.items():
        if entry.get("exempt"):
            if len(entry["exempt"]) <= 40:
                bad.append(f"{key}: exemption is a keyword, not an argument")
            if entry.get("obligations"):
                bad.append(f"{key}: claims an exemption AND lists obligations")
            continue
        obligations = entry.get("obligations")
        if not obligations:
            bad.append(f"{key}: no obligations and no exemption")
            continue
        for name, ob in obligations.items():
            if len(ob.get("requires", "")) <= 40:
                bad.append(f"{key}/{name}: `requires` does not state anything")
    assert not bad, "\n  " + "\n  ".join(bad)


def _clause_bodies() -> Dict[str, str]:
    """Each clause's text, from heading to the next heading."""
    roadmap = open(os.path.join(DOCS, "ROADMAP.md"), encoding="utf-8").read()
    spec = open(os.path.join(DOCS, "ASSEMBLY_SPEC.md"), encoding="utf-8").read()
    lockbox_block = roadmap[roadmap.index("## The lockbox constitution"):]
    lockbox_block = lockbox_block[:lockbox_block.index("\n## ", 1)]
    out: Dict[str, str] = {}
    for m in re.finditer(r"^### (\d\d) · .*?(?=^### |\Z)", lockbox_block, re.M | re.S):
        out[f"lockbox-{m.group(1)}"] = m.group(0)
    for m in re.finditer(r"^## (\d\d) · .*?(?=^## |\Z)", spec, re.M | re.S):
        out[f"assembly-{m.group(1)}"] = m.group(0)
    return out


def _text_hash(body: str) -> str:
    """Whitespace-insensitive, so rewrapping a paragraph is not a change."""
    return hashlib.sha256(" ".join(body.split()).encode()).hexdigest()[:12]


def test_no_clause_gained_an_obligation_without_being_re_enumerated():
    """The residual this check could not otherwise close, closed as far as it can be.

    Enumerating a clause's obligations is a JUDGMENT about prose, and nothing
    can verify that the enumeration is complete — a parser deriving obligations
    from English would only be substituting its opinion for the author's. So the
    honest guard is on the input: **if the clause text changes, the enumeration
    is stale until somebody says otherwise.**

    That is exactly the L16 failure looking forward rather than backward. §04 was
    written with three obligations and read as one; the next clause to gain a
    sentence would repeat it silently. Now it cannot: the hash moves, this fails,
    and re-reading the clause is the price of editing it.

    **What it still does not catch**, stated rather than glossed: deleting an
    obligation from the manifest while the clause text stands. That edit lives in
    this file, so it is at least visible in the same diff as the check it
    weakens — but nothing here fails on it, and a reader should know that.
    """
    bodies = _clause_bodies()
    drift = []
    for key, entry in CLAUSES.items():
        if key.startswith("routing-"):
            # The routing constitution is prose in Decision B rather than a
            # numbered clause block, so there is no span to hash. Its three
            # clauses are enumerated by hand and change with the document.
            continue
        body = bodies.get(key)
        if body is None:
            drift.append(f"{key}: no clause body found to hash")
            continue
        now = _text_hash(body)
        was = entry.get("text")
        if was != now:
            drift.append(
                f"{key}: text is {now}, manifest says {was}. RE-READ THE CLAUSE "
                f"and check its obligations are still the whole list, then "
                f"update `text=`. Title: {entry['title']}")
    assert not drift, (
        "a clause changed since its obligations were enumerated:\n  "
        + "\n  ".join(drift))


def _obligation_ids() -> List[Tuple[str, str]]:
    return sorted((k, name)
                  for k, e in CLAUSES.items()
                  for name in (e.get("obligations") or {}))


@pytest.mark.parametrize("clause_key,obligation", _obligation_ids())
def test_the_obligation_names_a_test_or_an_open_finding(clause_key, obligation):
    """One test per OBLIGATION, not per clause. That is the whole repair.

    A failure now names the requirement that is untracked — "clause 04's
    eligibility question has no link" — rather than reporting that a clause with
    three obligations and one link is covered.
    """
    ob = CLAUSES[clause_key]["obligations"][obligation]
    reasons: List[str] = []
    live = 0

    for rel, name in ob.get("tests", []):
        source, why = _def_source(rel, name)
        if why:
            reasons.append(why)
            continue
        if not name.startswith(("test_", "Test")):
            reasons.append(
                f"{rel}::{name} is not a test. Implementation goes under "
                "`implements`, which satisfies nothing — code cannot verify "
                "itself.")
            continue
        # THE TWO-ENDED LINK. The manifest says this test covers this clause;
        # the test has to say so too, by naming the clause key in its own
        # source. A key is an identifier, so it cannot collide the way a word
        # can — which is exactly how "eligib" matched nine rows about something
        # else.
        if clause_key not in source:
            reasons.append(
                f"{rel}::{name} does not cite `{clause_key}` in its own source, "
                "so the link is asserted at one end only. Name the clause in the "
                "docstring — if the test does not know which clause it "
                "discharges, neither does the reader.")
            continue
        live += 1

    by_id = _findings_by_id()
    for fid in ob.get("findings", []):
        row = by_id.get(fid)
        if row is None:
            reasons.append(f"{fid} is not in the ledger")
            continue
        if row["status"] not in ("OPEN", "PARTIAL"):
            reasons.append(f"{fid} is {row['status']}, so it no longer tracks anything")
            continue
        if clause_key not in json.dumps(row):
            reasons.append(
                f"{fid} does not name `{clause_key}` anywhere in its row, so "
                "nothing distinguishes 'tracks this clause' from 'is about a "
                "similar-sounding thing'.")
            continue
        live += 1

    assert live, (
        f"{clause_key} / {obligation} names neither a passing test nor an open "
        f"finding.\n  REQUIRES: {ob['requires']}\n  "
        + "\n  ".join(reasons) +
        "\nFile a finding for it. Do NOT add a stub test — an obligation nobody "
        "has built gets an open finding, which is the truthful record.")


def test_an_implementation_pointer_never_satisfies_an_obligation():
    """`implements` is documentation, and the check must not accept it as
    evidence.

    The first matcher listed `("ml/router.py", "_skip_is_permitted")` beside the
    tests for routing §01, in the same `tests` list. It was never counted as a
    test only because the fragment happened to also appear elsewhere — that is
    luck, not design. An implementation pointer says the clause is implemented,
    which is not evidence that it is implemented correctly.
    """
    for key, entry in CLAUSES.items():
        for name, ob in (entry.get("obligations") or {}).items():
            for rel, fn in ob.get("implements", []):
                source, why = _def_source(rel, fn)
                assert why is None, f"{key}/{name}: {why}"
                assert not fn.startswith(("test_", "Test")), (
                    f"{key}/{name}: `{fn}` is a test listed under `implements`, "
                    "where it counts for nothing. Move it to `tests`.")
            assert ob.get("tests") or ob.get("findings"), (
                f"{key}/{name} lists only `implements`, which satisfies nothing.")
