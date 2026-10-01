"""The word-budget gate (BLUEPRINT §11.4, M1_CONTRACT §5–§6, DRIVE_RUBRIC §2.8 and §5.12).

Every teaching field within its budget; every finding summary and lever within budget and free of
the program talking (None, NaN, {placeholders}, raw markdown) on every sample CSV under every lens
that fits it; every decision kind's sentence a finished, publishable sentence.
"""
from __future__ import annotations

import re

import pytest

from turbotab.core import decisions as d
from turbotab.core import teaching, voice
from turbotab.core.decisions import ProjectState
from turbotab.core.stages.finding_words import LEVER_WORDS, NO_LEVER, SUMMARY_WORDS
from turbotab.core.stages.findings import findings_stage
from turbotab.core.tests.stage_harness import SAMPLES, Ingested

B = teaching.BUDGETS
SAMPLE_CSVS = sorted(SAMPLES.glob("*.csv"))


def over(text: str, budget: int) -> bool:
    return voice.words(text) > budget


# ── teaching ─────────────────────────────────────────────────────────────────

def test_there_is_one_teaching_entry_per_question_in_asking_order():
    assert [e.key for e in teaching.entries()] == list(teaching.QUESTION_KEYS)


@pytest.mark.parametrize("entry", teaching.entries(), ids=lambda e: e.key)
def test_every_teaching_field_is_within_its_budget(entry):
    problems = [f"{f}: {voice.words(getattr(entry, f))} words"
                for f in ("title", "question", "one_liner", "why", "consumer")
                if over(getattr(entry, f), B[f])]
    for o in entry.options:
        if over(o.label, B["option_label"]):
            problems.append(f"option {o.value} label: {o.label!r}")
        if over(o.consequence, B["option_consequence"]):
            problems.append(f"option {o.value} consequence: {voice.words(o.consequence)} words")
    for t in entry.terms:
        if over(t.definition, B["term_definition"]):
            problems.append(f"term {t.term}: {voice.words(t.definition)} words")
    assert not problems, problems


@pytest.mark.parametrize("entry", teaching.entries(), ids=lambda e: e.key)
def test_teaching_text_is_finished_and_free_of_machinery(entry):
    texts = [entry.title, entry.question, entry.one_liner, entry.why, entry.consumer]
    texts += [o.consequence for o in entry.options] + [t.definition for t in entry.terms]
    for s in (entry.drawer.sections if entry.drawer else []):
        texts += [s.heading, s.body]
    for text in texts:
        assert not voice.machinery(text), (text, voice.machinery(text))
    for text in [entry.one_liner, entry.why, entry.consumer, *[t.definition for t in entry.terms]]:
        assert text.endswith("."), text
    assert entry.question.endswith("?"), entry.question


@pytest.mark.parametrize("entry", teaching.entries(), ids=lambda e: e.key)
def test_every_drawer_claim_carries_a_badge_and_a_section(entry):
    """The packs are the source; each section says where the field stands, and where it says so."""
    assert entry.drawer and entry.drawer.sections, f"{entry.key} has no drawer"
    for s in entry.drawer.sections:
        assert s.evidence is not None, s.heading
        assert s.evidence.status in ("SETTLED", "CONVENTION", "DISPUTED")
        assert re.match(r"research/[A-Z_]+\.md#[0-9A-Z.]+ · ", s.evidence.source), s.evidence.source


def test_every_choice_has_options_and_every_option_a_consequence():
    choices = {"task", "purpose", "roles", "exclusions", "missing", "split", "energy_adjustment",
               "models", "substitution", "lens"}
    for entry in teaching.entries():
        if entry.key in choices:
            assert entry.options, entry.key
        assert len({o.value for o in entry.options}) == len(entry.options), entry.key


def test_the_options_cover_what_the_decisions_accept():
    """Every value a decision can record has its consequence written down."""
    from typing import get_args

    values = {e.key: {o.value for o in e.options} for e in teaching.entries()}
    assert set(get_args(d.Lens)) == values["lens"]
    assert set(get_args(d.Task)) == values["task"]
    assert set(get_args(d.Purpose)) == values["purpose"]
    assert set(get_args(d.Role)) == values["roles"]
    assert set(get_args(d.EnergyMethod)) == values["energy_adjustment"]
    assert set(get_args(d.MissingStrategy)) == values["missing"]
    assert {"linear", "elastic_net", "boosted_trees"} == values["models"]


def test_the_energy_teaching_states_the_equivalence_precisely():
    """M0's note: the residual and standard models agree with energy in the model or without
    covariates — not when energy leaves the model and a covariate correlates with it."""
    drawer = teaching.entry("energy_adjustment").drawer
    text = " ".join(s.body for s in drawer.sections)
    assert "only when no covariate correlates with energy" in text
    assert "identical" in text


def test_the_required_terms_define_themselves():
    terms = {t.term for e in teaching.entries() for t in e.terms}
    for required in ("estimand", "residual method", "substitution", "holdout", "cross-validation",
                     "complete cases"):
        assert required in terms, required


# ── findings on every fixture ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spoken(tmp_path_factory):
    """Every sample CSV's findings under the lenses that fit it: (file, lens, findings)."""
    out = []
    for path in SAMPLE_CSVS:
        table = Ingested(path, tmp_path_factory.mktemp(path.stem))
        lens = table.lenses_that_fit()
        artifact = table.run(findings_stage, ProjectState(lens=lens))
        out.append((path.name, lens, artifact["findings"]))
    return out


def test_every_fixture_was_read(spoken):
    assert len(spoken) == len(SAMPLE_CSVS) >= 30
    assert sum(len(f) for _, _, f in spoken) > 100


def test_every_finding_summary_and_lever_is_within_budget(spoken):
    problems = []
    for name, lens, findings in spoken:
        for f in findings:
            if over(f["summary"], SUMMARY_WORDS):
                problems.append(f"{name} {f['id']}: summary {voice.words(f['summary'])} words")
            if f["lever_label"] and over(f["lever_label"], LEVER_WORDS):
                problems.append(f"{name} {f['id']}: lever {f['lever_label']!r}")
    assert not problems, problems


def test_every_finding_text_is_free_of_machinery(spoken):
    problems = []
    for name, lens, findings in spoken:
        for f in findings:
            for key in ("title", "summary", "detail", "why_it_matters", "lever_label"):
                found = voice.machinery(f.get(key))
                if found:
                    problems.append(f"{name} {f['id']} {key}: {found}")
    assert not problems, problems


def test_every_finding_sentence_ends_once(spoken):
    for name, lens, findings in spoken:
        for f in findings:
            for key in ("title", "summary", "detail"):
                text = f[key]
                assert re.search(r"[.?!][)\"'”’]*$", text), (name, f["id"], key, text)
                assert not re.search(r"\.\.(?!\.)|[,;:]\.|\s[,.;:]", text), (name, f["id"], key, text)
            if f["lever_label"]:
                assert not f["lever_label"].endswith("."), f["lever_label"]


def test_a_finding_without_a_lever_says_so(spoken):
    """M1_CONTRACT §6: no pretending. A lever names its question; no lever is said out loud."""
    for name, lens, findings in spoken:
        for f in findings:
            if f["routes_to"] is None:
                assert f["lever_label"] is None
                says = (NO_LEVER in f["summary"] or "not asked" in f["summary"]
                        or "nothing needs" in f["summary"] or "not in this version" in f["summary"])
                assert says, (name, f["id"], f["summary"])
            else:
                assert f["routes_to"] in teaching.QUESTION_KEYS
                assert f["lever_label"], (name, f["id"])


def test_groups_page_only_same_kind_findings(spoken):
    for name, lens, findings in spoken:
        sizes: dict[str, int] = {}
        for f in findings:
            if f["group"]:
                sizes[f["group"]] = sizes.get(f["group"], 0) + 1
        assert all(n >= 2 for n in sizes.values()), (name, sizes)


# ── decision sentences ───────────────────────────────────────────────────────

def representative_decisions():
    rule = d.ExclusionRule(column="energy_kcal", low=500, high=5000, reason="implausible intakes")
    return [
        d.SetLens(lenses=["dietary", "clinical"]),
        d.SetTarget(column="hba1c"),
        d.SetTask(column="hba1c", task="regression"),
        d.SetPurpose(purpose="inference"),
        d.Revert(decision_id="0" * 32),
        d.SetRoles(roles={"participant_id": "identifier", "energy_kcal": "energy",
                          "protein_g": "exposure", "fat_g": "exposure", "age": "covariate"}),
        d.SetEnergyAdjustment(method="residual", energy_column="energy_kcal",
                              nutrients=["protein_g", "fat_g"], strata="sex"),
        d.SetExclusions(rules=[rule]),
        d.SetMissing(strategy="complete_case"),
        d.SetSplit(holdout=0.2, seed=7, folds=5),
        d.SelectModels(models=["linear", "elastic_net", "boosted_trees"]),
        d.SetSubstitution(donor="fat_g", recipient="carbohydrate_g", step_kcal=100),
    ]


# Kinds whose shapes exist (M2_CONTRACT.md) but whose sentences the M2 voice agent still owes.
# The voice agent's job is to EMPTY this set; a kind may only leave it with a representative here
# and a sentence in voice.py. Never add to it except when a milestone's contract adds kinds.
OWED_BY_M2_VOICE = {
    "set_orientation", "set_event", "set_grain", "set_repeat_kind", "set_unit",
    "set_aggregation", "set_temporal", "open_seal", "apply_repair", "defer_finding",
    "dismiss_finding",
}


def test_every_decision_kind_has_a_representative_here():
    from typing import get_args

    union = get_args(get_args(d.Decision)[0])
    kinds = {m.model_fields["kind"].default for m in union} - OWED_BY_M2_VOICE
    assert kinds == {x.kind for x in representative_decisions()} == set(voice.kinds()) - OWED_BY_M2_VOICE


@pytest.mark.parametrize("decision", representative_decisions(), ids=lambda x: x.kind)
@pytest.mark.parametrize("with_context", [False, True], ids=["bare", "context"])
def test_every_decision_sentence_is_finished_and_free_of_machinery(decision, with_context, tmp_path):
    state = ProjectState(target="hba1c", task="regression", lens=["dietary"])
    ctx = None
    if with_context:
        table = Ingested(SAMPLES / "dietary_recalls.csv", tmp_path)
        ctx = {"frame": table.frame(), "columns": table.info["columns"],
               "n_rows": table.info["n_rows"], "n_cohort": 580, "records": []}
    text = voice.sentence_for(decision, state, ctx)
    assert text and text.strip() == text
    assert text.endswith("."), text
    assert not voice.machinery(text), (text, voice.machinery(text))
    assert "[object" not in text and "{" not in text and "None" not in text
