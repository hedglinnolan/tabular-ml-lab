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

# The modules that write what a card, a refusal, an option, a preview or a result's reading says:
# every engine module, apart from the ones that word only the methods register (the voice's
# sentences, the export's manuscript and tables, the generated methods reference) and the word list
# itself, whose register switch names the terms. The server's
# routes and schemas belong to the server package and are walked live by
# ``tests/acceptance/test_plain_words_served.py`` instead.
METHODS_MODULES = ("core/voice.py", "core/export/", "core/reference/", "core/plain.py")


def _engine_modules() -> tuple[str, ...]:
    found = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if ("/tests/" in f"/{rel}" or rel.startswith(("frontend/", "server/"))
                or path.name.startswith("test_") or rel.startswith(METHODS_MODULES)):
            continue
        found.append(rel)
    return tuple(found)


CARD_MODULES = _engine_modules()

# The methods register: the function that words a methods sentence, a caption, a limitation or a
# statement keeps the technical terms.
# The name must END with the register's word: ``_affected_confounders_need_g_methods`` (a refusal)
# or ``applicable_methods`` (a list of options) is not the methods register.
METHODS_REGISTER = re.compile(r"(?:^|_)(?:sentence|clause|caption|limitation|paragraph|statement)$")
# Functions that word the methods text or the plan record under a name that does not say so.
METHODS_FUNCTIONS = {
    "core/time_varying.py": {"fit_note", "_order_clause"},
    "core/causal.py": {"final_stage_robustness", "sensitivity_for"},
    "core/stages/time_varying.py": {"_msm_text", "_gformula_text", "_interval_text", "_loss_text",
                                    "_affected_text"},
    "core/methods/interaction.py": {"family_count"},
    "core/stages/rows.py": {"cohort_flow"},
    "core/models/featurewise.py": {"methods_label"},
}
# A function that refuses or offers exits writes a card, whatever its name says.
CARD_CALLS = {"Refusal", "_refusal", "refusal", "InferenceExit"}
# The methods register's own fields: an artifact's ``methods=`` and a result's ``sentence=``.
METHODS_KEYWORDS = {"methods", "sentence"}
# The contracts' own descriptions (what a method declares about itself for the registry), and the
# verbatim quotations of a source.
CONTRACT_KEYWORDS = {"relations", "storyboard", "needs", "place", "scope_note"}
CONTRACT_CALLS = {"_relation", "Relation"}
# The registry's own title of a method (``MethodContract(label=...)``, or a package's wrapper of it).
CONTRACT_TITLES = {"MethodContract", "_contract"}
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


def _writes_a_card(node: ast.AST) -> bool:
    """A function that raises a refusal or builds exits: card text, whatever its name."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and getattr(sub.func, "id", getattr(sub.func, "attr", "")) in CARD_CALLS:
            return True
        if isinstance(sub, ast.keyword) and sub.arg == "exits":
            return True
    return False


def _register_ranges(tree: ast.AST, rel: str) -> list[tuple[int, int]]:
    """Source line ranges the scan leaves alone: the methods register, contracts and quotations."""
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name in METHODS_FUNCTIONS.get(rel, ())
                or METHODS_REGISTER.search(node.name) and not _writes_a_card(node)):
            ranges.append((node.lineno, node.end_lineno))
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name in CONTRACT_CALLS:
                ranges.append((node.lineno, node.end_lineno))
            elif name == "term" and node.args:  # a term's own name is its quiet label
                ranges.append((node.args[0].lineno, node.args[0].end_lineno))
            elif name in CONTRACT_TITLES:  # the registry's own title of the method
                ranges += [(k.value.lineno, k.value.end_lineno) for k in node.keywords
                           if k.arg == "label"]
        elif isinstance(node, ast.keyword) and node.arg in CONTRACT_KEYWORDS | METHODS_KEYWORDS:
            ranges.append((node.value.lineno, node.value.end_lineno))
        elif isinstance(node, ast.Assign) and any(  # found["sentence"] = ...
                isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                and t.slice.value in METHODS_KEYWORDS for t in node.targets):
            ranges.append((node.lineno, node.end_lineno))
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


def test_a_refusal_named_like_the_register_is_still_scanned():
    """``_affected_confounders_need_g_methods`` is a refusal, not the methods register."""
    tree = ast.parse((ROOT / "core/time_varying.py").read_text())
    refusal = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "_affected_confounders_need_g_methods")
    skipped = _register_ranges(tree, "core/time_varying.py")
    assert not any(a <= refusal.lineno <= b for a, b in skipped)
    assert {"core/models/explain.py", "core/stages/effects.py", "core/models/effects.py"} <= set(
        CARD_MODULES)


# ── the methods register keeps its terms ─────────────────────────────────────

def test_the_methods_register_keeps_the_technical_terms():
    """The registry's titles, an artifact's ``methods=`` and a result's ``sentence=`` are the
    methods register: they keep "exposure" and "estimand" as the manuscript reads them."""
    from turbotab.core import contracts

    titles = {k: contracts.contract(k).label for k in (
        "effect_modification", "interaction", "survey_population", "evalue_sd")}
    assert titles == {
        "effect_modification": "Effect modification (the exposure's effect across strata of a "
                               "modifier)",
        "interaction": "Interaction (the joint effect of two exposures)",
        "survey_population": "The population estimand under a survey design",
        "evalue_sd": "The E-value of a difference, standardized by the estimand's SD",
    }
    source = (ROOT / "core/methods/interaction.py").read_text()
    assert 'methods="No effect modification is estimated without a declared exposure under "' in source
    assert "as the exposure: " in source  # the waiting interaction's methods sentence


# ── the meanings the plain words keep ────────────────────────────────────────

def _options(key: str) -> dict[str, str]:
    return {o.value: o.consequence for o in teaching.entry(key).options}


def _terms(key: str) -> dict[str, str]:
    return {t.term: t.definition for t in teaching.entry(key).terms}


def test_the_direct_effect_still_obliges_the_mediators_own_common_causes():
    direct = _options("estimand")["direct"]
    assert "must" in direct and re.search(r"\b(their|mediators?'?s?)\b.*\boutcome\b", direct), direct


def test_the_time_varying_card_keeps_what_did_the_changing():
    standard = _options("time_varying")["standard"]
    assert re.search(r"earlier (study[- ]factor values|values of what you study)", standard), standard
    terms = _terms("time_varying")
    weight = terms["stabilized weight"]
    assert "later covariates" not in weight and "could explain the link" in weight, weight
    confounder = terms["time-varying confounder"]
    assert not confounder.endswith("earlier values of it may change it."), confounder
    assert re.search(r"earlier values of what you study may change it", confounder), confounder


def test_the_interaction_adjusts_for_the_second_factors_common_causes_not_every_covariate():
    why = teaching.entry("modification").why
    assert "covariates must be adjusted" not in why
    assert re.search(r"could explain the second\b.*\blink to the outcome must be adjusted", why), why


def test_the_awkward_replacements_read_as_english():
    texts = [t for e in teaching.entries() for _, t in _card_texts(e)]
    for rel in ("core/estimand.py", "clinical.py", "core/structural.py", "core/time_varying.py",
                "core/stages/explore.py"):
        texts += [t for _, t in card_strings(rel)]
    for awkward in ("different comparisons you want", "comparison you want whose",
                    "different comparison you want", "estimate of what you study on",
                    "choose the comparison you want by", "as the comparison you want names",
                    "does the comparison you want declare", "question on the comparison you want"):
        assert not [t for t in texts if awkward in t], awkward


# ── the adjustment card's words ──────────────────────────────────────────────

def test_the_role_words_are_noun_phrases():
    """``derived_words`` follows "each" on the card ("each common cause of …"), so every entry is
    a noun phrase, never a verb phrase such as "could explain the link"."""
    from turbotab.core import estimand, plan_previews

    verbs = {"could", "can", "may", "might", "is", "are", "was", "explains", "explain", "causes",
             "changed"}
    for table in (estimand.ROLE_WORDS, plan_previews.ROLE_SHORT):
        for role, phrase in table.items():
            assert phrase.split()[0] not in verbs, (role, phrase)
            assert plain.is_plain(phrase), (role, phrase)
    assert estimand.ROLE_WORDS["confounder"] == "common cause of what you study and the outcome"


def test_the_adjustment_card_keeps_confounder_as_its_quiet_label():
    terms = _terms("adjustment")
    assert "confounder" in terms
    assert plain.is_plain(terms["confounder"]) and "could explain the link" in terms["confounder"]


# ── quotations of a source stay verbatim ─────────────────────────────────────

def test_a_quotation_is_exempt_only_inside_its_quotation_marks():
    assert plain.is_plain("Brenner & Blettner 1997: “categorization of the confounder may lead”")
    assert not plain.is_plain("the confounder “may lead”")


def test_the_quoted_sources_are_verbatim():
    """A source quoted on a card is quoted word for word; the plain words go around the quotation."""
    from turbotab.core.stages import calibration

    refusal = " ".join(t for _, t in card_strings("core/methods/exposure_form.py"))
    assert ('Blettner 1997: "categorization of the confounder may often lead to serious residual '
            'confounding if the number of categories is small").') in refusal
    assert ("“the usual statistical test of the null hypothesis (no exposure effect) remains "
            "theoretically valid”") in calibration.TEST


def test_a_quotation_in_straight_quotes_is_exempt_too():
    assert plain.is_plain('(Brenner & Blettner 1997: "categorization of the confounder")')
    assert not plain.is_plain('a confounder, "categorization"')


def test_the_served_walk_skips_only_identifiers_and_quiet_labels():
    """A served ``value``, ``values`` or ``source`` can carry a sentence; the walk reads them."""
    assert not {"value", "values", "source"} & plain.QUIET_KEYS


def test_the_export_tables_put_a_plain_reading_back_in_the_technical_register():
    """A reading the card words plainly is tabulated in the manuscript's register."""
    from turbotab.core.export import tables

    effects = {"families": [{
        "family": "linear", "label": "Linear", "sequence": [],
        "appendix": [{"key": "primary", "label": "Model 2", "terms": [
            {"feature": "sex", "why": "a modifier's main effect: the effect of what you study is "
                                      "read within its levels"}]}],
        "sensitivity": [{"feature": "fiber", "not_computed": "What you study enters as a curve, and "
                         "its straight-line estimate could not be fit."}]}]}
    rows = [r for t in tables.table2(effects, None) for r in t.rows]
    said = [r.get("why") or r.get("reading") for r in rows if r.get("why") or r.get("reading")]
    assert said == ["a modifier's main effect: the effect of the exposure is read within its levels",
                    "The exposure enters as a curve, and its straight-line estimate could not be "
                    "fit."]
    assert plain.technical("each study factor in turn; a study-factor history; the comparison "
                           "you want") == "each exposure in turn; an exposure history; the estimand"
