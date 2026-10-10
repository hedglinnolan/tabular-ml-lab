"""P0.12 · plain words on every card (V2_DEFINITION_OF_DONE: "Plain words on every card").

"Exposure", "confounder" and "estimand" are quiet terms: a card says "what you study", "what else
could explain the link" and "the comparison you want", and the technical name rides along only as
a quiet label (a term's own name). The methods prose (the voice's sentences, the manuscript, the
export) keeps the technical register.

Three scans hold the cards to it:

* the teaching content: every card-facing field of every entry, the quiet terms only as a term's
  own name;
* the card-facing source: every sentence-length string literal of the modules that write cards,
  refusals, options and previews, outside the methods register (the functions that word the methods
  sentence, caption or limitation, and the contracts' own descriptions);
* the served cards of a real journey through the server (``test_a_served_card_speaks_plainly``).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from turbotab.core import plain, teaching

ROOT = Path(__file__).resolve().parents[2]  # turbotab/

# The modules that write what a card, a refusal, an option or a preview says.
CARD_MODULES = (
    "core/teaching/content.py", "core/estimand.py", "core/time_varying.py", "core/causal.py",
    "core/covariate_guesses.py", "core/decisions.py", "core/interview.py", "core/plan_previews.py",
    "core/method_previews.py", "core/fact_previews.py", "core/usual_intake.py", "core/survey.py",
    "core/scales.py", "core/designs.py", "core/ask.py", "core/sweep.py", "core/custom_sound.py",
    "core/structural.py", "resolution.py", "clinical.py",
    "core/stages/causal.py", "core/stages/time_varying.py", "core/stages/rows.py",
    "core/stages/modeling.py", "core/stages/calibration.py", "core/stages/explore.py",
    "core/stages/__init__.py",
    "core/methods/exposure_form.py", "core/methods/interaction.py", "core/methods/levers.py",
    "core/methods/energy.py", "core/methods/substitution.py", "core/methods/missing.py",
    "core/methods/calibration.py", "core/methods/dietary_caveats.py", "core/methods/scales.py",
    "core/models/featurewise.py", "core/models/causal.py", "core/models/survey.py",
    "core/models/pipeline.py", "core/readings.py", "packs.py",
)

# The methods register: the function that words a methods sentence, a caption, a limitation or a
# statement keeps the technical terms.
METHODS_REGISTER = re.compile(r"sentence|clause|methods|caption|limitation|paragraph|statement")
# Functions that word the methods text or the plan record under a name that does not say so.
METHODS_FUNCTIONS = {
    "core/time_varying.py": {"fit_note", "_order_clause"},
    "core/causal.py": {"final_stage_robustness", "sensitivity_for"},
    "core/stages/time_varying.py": {"_msm_text", "_gformula_text", "_interval_text", "_loss_text",
                                    "_affected_text"},
    "core/methods/interaction.py": {"family_count"},
    "core/stages/rows.py": {"cohort_flow"},
}
# The contracts' own descriptions (what a method declares about itself for the registry), and the
# verbatim quotations of a source.
CONTRACT_KEYWORDS = {"relations", "storyboard", "needs", "place", "scope_note"}
CONTRACT_CALLS = {"_relation", "Relation"}
# Verbatim quotations of a source, and the names of the methods text a sensitivity analysis carries.
QUOTATIONS = re.compile(r"_QUOTE$|^(ROBUSTNESS_NAMES|ACTION_WORDS|ROLE_SINGULAR|ROLE_PLURAL"
                        r"|OUTCOME_KIND_SINGULAR|OUTCOME_KIND_PLURAL|MEASURE_WORDS|ATTESTATION)$")


def _docstrings(tree: ast.AST) -> set[int]:
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant):
                found.add(id(body[0].value))
    return found


def _register_ranges(tree: ast.AST, rel: str) -> list[tuple[int, int]]:
    """Source line ranges the scan leaves alone: the methods register, contracts and quotations."""
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                METHODS_REGISTER.search(node.name) or node.name in METHODS_FUNCTIONS.get(rel, ())):
            ranges.append((node.lineno, node.end_lineno))
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name in CONTRACT_CALLS:
                ranges.append((node.lineno, node.end_lineno))
            elif name == "term" and node.args:  # a term's own name is its quiet label
                ranges.append((node.args[0].lineno, node.args[0].end_lineno))
            elif name == "MethodContract":  # the registry's own title of the method
                ranges += [(k.value.lineno, k.value.end_lineno) for k in node.keywords
                           if k.arg == "label"]
        elif isinstance(node, ast.keyword) and node.arg in CONTRACT_KEYWORDS:
            ranges.append((node.value.lineno, node.value.end_lineno))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
                isinstance(t, ast.Name) and QUOTATIONS.search(t.id)
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target])):
            ranges.append((node.lineno, node.end_lineno))
    return ranges


def card_strings(rel: str) -> list[tuple[int, str]]:
    """The string literals of ``rel`` that a card can carry, with their lines: two words or more,
    or a capitalized word (a label such as "Exposure"); a lowercase single token is an identifier."""
    tree = ast.parse((ROOT / rel).read_text())
    docs, skip = _docstrings(tree), _register_ranges(tree, rel)
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        words = node.value.split()
        if id(node) in docs or len(words) < 2 and not re.fullmatch(r"[A-Z][a-z]+", node.value):
            continue
        if any(a <= node.lineno <= b for a, b in skip):
            continue
        found.append((node.lineno, node.value))
    return sorted(found)


# ── the word list itself ─────────────────────────────────────────────────────

def test_the_word_list_finds_the_three_terms_and_only_them():
    assert plain.forbidden_terms("Which exposure, and its Confounders and estimands?") == [
        "exposure", "confounders", "estimands"]
    for ok in ("what you study", "what else could explain the link", "the comparison you want",
               "exposed and unexposed rows", "a study factor", "confounding", "confounded by it"):
        assert plain.is_plain(ok), ok


def test_a_term_kept_for_its_meaning_is_defined_in_place_in_twelve_words():
    kept = "the estimand (the quantity a study sets out to estimate) is fixed first"
    assert plain.is_plain(kept)
    assert len("the quantity a study sets out to estimate".split()) <= plain.DEFINITION_WORDS
    long = ("the estimand (the quantity that a study sets out to estimate, stated in words so "
            "that two methods can be compared) is fixed first")
    assert plain.forbidden_terms(long) == ["estimand"]
    assert plain.forbidden_terms("the estimand is fixed first") == ["estimand"]


# ── the teaching content ─────────────────────────────────────────────────────

def _card_texts(entry) -> list[tuple[str, str]]:
    texts = [("title", entry.title), ("question", entry.question), ("one_liner", entry.one_liner),
             ("why", entry.why), ("consumer", entry.consumer)]
    texts += [(f"option {o.value} label", o.label) for o in entry.options]
    texts += [(f"option {o.value} consequence", o.consequence) for o in entry.options]
    # A term's own name is the quiet label; its definition is a card sentence.
    texts += [(f"term {t.term} definition", t.definition) for t in entry.terms]
    for s in entry.drawer.sections if entry.drawer else []:
        texts += [(f"drawer {s.heading!r} heading", s.heading), (f"drawer {s.heading!r}", s.body)]
    return texts


@pytest.mark.parametrize("entry", teaching.entries(), ids=lambda e: e.key)
def test_a_teaching_card_speaks_plainly(entry):
    used = {where: plain.forbidden_terms(text) for where, text in _card_texts(entry)}
    assert not {w: t for w, t in used.items() if t}, used


def test_the_quiet_terms_survive_as_labels():
    """The technical name still rides along: the term entries keep their names, the card its own words."""
    names = {t.term for e in teaching.entries() for t in e.terms}
    assert {"exposure", "estimand", "time-varying confounder"} <= names
    roles = teaching.entry("roles")
    assert {o.value: o.label for o in roles.options}["exposure"] == "What you study"
    assert teaching.entry("estimand").title == "What you study and its effect"


# ── the card-facing source ───────────────────────────────────────────────────

@pytest.mark.parametrize("rel", CARD_MODULES)
def test_card_facing_strings_speak_plainly(rel):
    bad = [(line, text[:90]) for line, text in card_strings(rel) if plain.forbidden_terms(text)]
    assert not bad, f"{rel}: {bad}"


def test_the_scan_sees_the_methods_register_it_leaves_alone():
    """The register that keeps the technical terms is real, and the scan skips it deliberately."""
    tree = ast.parse((ROOT / "core/estimand.py").read_text())
    skipped = _register_ranges(tree, "core/estimand.py")
    raw = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and plain.forbidden_terms(n.value) and any(a <= n.lineno <= b for a, b in skipped)]
    assert raw, "the methods register (multiplicity statement, caption) keeps its technical terms"
