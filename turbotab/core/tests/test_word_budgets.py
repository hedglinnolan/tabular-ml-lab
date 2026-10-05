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
from turbotab.core.consequences import MAX_COACH
from turbotab.core.decisions import ProjectState
from turbotab.core.stages.finding_words import LEVER_WORDS, NO_LEVER, SUMMARY_WORDS
from turbotab.core.stages.findings import findings_stage
from turbotab.core.tests.stage_harness import LENSES, NHANES, SAMPLES, Ingested

B = teaching.BUDGETS
SAMPLE_CSVS = sorted(SAMPLES.glob("*.csv"))


def over(text: str, budget: int) -> bool:
    return voice.words(text) > budget


# ── teaching ─────────────────────────────────────────────────────────────────

def test_there_is_one_teaching_entry_per_question_in_asking_order():
    assert [e.key for e in teaching.entries()] == list(teaching.TEACHING_KEYS)
    cards = {"repairs"}  # taught, but not a question the Router asks
    assert [k for k in teaching.TEACHING_KEYS if k not in cards] == list(teaching.QUESTION_KEYS)


def test_every_question_the_router_asks_is_taught():
    from turbotab.core import interview

    assert set(interview.QUESTION_KEYS) <= set(teaching.QUESTION_KEYS)
    order = [k for k in teaching.QUESTION_KEYS if k in set(interview.QUESTION_KEYS)]
    assert order == list(interview.QUESTION_KEYS), "the Router and the teaching disagree on order"


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
               "models", "substitution", "lens", "orientation", "repairs", "grain", "repeat_kind",
               "unit", "aggregation", "temporal", "open_seal"}
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
    from turbotab.core.models import families

    assert {f.key for f in families()} == values["models"]
    assert set(get_args(d.Orientation)) == values["orientation"]
    assert set(get_args(d.SetGrain.model_fields["grain"].annotation)) == values["grain"]
    assert set(get_args(d.RepeatKind)) == values["repeat_kind"]
    assert set(get_args(d.SetUnit.model_fields["unit"].annotation)) == values["unit"]
    assert set(get_args(d.AggregationMethod)) == values["aggregation"]
    assert {"true", "false"} == values["temporal"]
    assert {"apply", "defer", "dismiss"} == values["repairs"]


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
def tables(tmp_path_factory):
    """Every sample CSV (and the real NHANES export when present), ingested, with the lenses that
    fit it and its findings under them: (file, table, lens, findings)."""
    out = []
    for path in [*SAMPLE_CSVS, *([NHANES] if NHANES.is_file() else [])]:
        table = Ingested(path, tmp_path_factory.mktemp(path.stem))
        lens = ["dietary", "clinical"] if path == NHANES else table.lenses_that_fit()
        artifact = table.run(findings_stage, ProjectState(lens=lens))
        out.append((path.name, table, lens, artifact["findings"]))
    return out


@pytest.fixture(scope="module")
def spoken(tables):
    """Every sample CSV's findings under the lenses that fit it: (file, lens, findings)."""
    return [(name, lens, findings) for name, _, lens, findings in tables if name != NHANES.name]


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
                if f.get("repairs"):  # M2: its repairs are its lever, and it does not deny having one
                    assert NO_LEVER not in f["summary"], (name, f["id"], f["summary"])
                    continue
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
        # M2 (M2_CONTRACT.md §1, §3, §4)
        d.SetOrientation(orientation="feature_major"),
        d.SetEvent(column="hba1c", level="1"),
        d.SetGrain(grain="repeated", id_column="participant_id"),
        d.SetRepeatKind(repeat_kind="repeats"),
        d.SetUnit(unit="unit"),
        d.SetAggregation(method="mean", outcome="mean"),
        d.SetTemporal(temporal=True, time_column="recall_date"),
        d.OpenSeal(),
        d.ApplyRepair(finding_id="pack::survey::sentinel_codes", option="to_missing",
                      params={"code": 9}),
        d.DeferFinding(finding_id="voice::flag__imputed_bmi", to="missing"),
        d.DismissFinding(finding_id="pack::dietary::compositional", reason="Shares are not modeled"),
        # WP1 (audit §5): what values mean
        d.SetFeatureTable(label="metabolite", annotations=["mz", "rt"]),
        d.SetCategorical(columns=["RIDRETH3", "DMDEDUC2"]),
        # WP10 (audit §5): the survey answer
        d.SetSurvey(estimand="population", weight="WTDRD1", strata="SDMVSTRA", psu="SDMVPSU"),
        # WP12a (audit §5): an exposure's form, an ordinal outcome's order, a %E substitution
        d.SetExposureForm(column="protein_g", form="spline", knots=4),
        d.SetExposureForm(column="fat_g", form="quintiles"),
        d.SetOutcomeOrder(column="hba1c", levels=["normal", "prediabetes", "diabetes"]),
        d.SetOutcomeUnit(column="hba1c", unit="%"),
        d.SetOutcomeScale(column="hba1c", scale="log"),
        # WP13 gate repair: a unit TurboTab could only propose, recorded by the user
        d.SetColumnUnit(column="energy_kcal", unit="kj"),
        d.SetColumnUnit(column="energy_kcal", unit="kcal", days=2),
        d.SetColumnUnit(column="age", unit="months"),
        d.ConfirmRole(column="protein_g", role="exposure"),
        # BLUEPRINT §14.1: one reading of the data confirmed on its own, kind by kind
        d.ConfirmReading(reading="role", column="protein_g", value="exposure"),
        d.ConfirmReading(reading="cluster", column="participant_id", value="no"),
        d.ConfirmReading(reading="unit", column="weight", value="kg"),
        d.ConfirmReading(reading="day_count", column="energy_kcal", value="2"),
        d.ConfirmReading(reading="code_or_count", column="age", value="code"),
        d.ConfirmReading(reading="time_column", column="age", value="orders"),
        d.ConfirmReading(reading="nested_in", column="protein_g", value="energy_kcal"),
        d.ConfirmReading(reading="sex_coding", column="sex", value="female=2,male=1"),
        # BLUEPRINT §14.2: a block confirmation, each listed reading with the value it showed
        d.ConfirmReadings(items=[
            d.ReadingItem(reading="code_or_count", column="age", value="amount"),
            d.ReadingItem(reading="code_or_count", column="sodium_mg", value="amount"),
            d.ReadingItem(reading="cluster", column="participant_id", value="yes")]),
        d.SetRoles(roles={"participant_id": "identifier", "energy_kcal": "energy",
                          "protein_g": "exposure", "fiber_g": "exposure"},
                   unconfirmed=["fiber_g"]),
        d.SetSubstitution(donor="fat_g", recipient="carbohydrate_g", scale="percent_energy",
                          step_percent=5),
        # WP12: a time-to-event outcome's follow-up
        d.SetFollowUp(column="hba1c", time_column="followup_years", entry_column="enrolled_years"),
        # WP12 (audit §5): the Goldberg screen as the primary, every row and a fixed screen beside it
        d.SetSensitivity(analyses=[
            d.SensitivityAnalysis(label="Every row", rules=[]),
            d.SensitivityAnalysis(label="500–5,000 kcal", rules=[rule]),
            d.SensitivityAnalysis(label="Goldberg", rules=[d.GoldbergRule(
                column="energy_kcal", days=2, sex="sex", female=["F"], male=["M"], age="age",
                weight="weight_kg", equation="schofield", pal=1.55,
                reason="implausible energy reports")]),
        ]),
        d.SetMeasurementError(method="regression_calibration", exposures=["protein_g"]),
        # WP16 (audit §5): a re-seal after the opening, and the inference analysis-plan lock
        d.Reseal(reason="the first draw left one site out"),
        d.LockPlan(plan={"target": "hba1c"}, digest="0" * 64),
        # WP17 (audit §5): the follow-up, the grouping, the exposure and effect, the adjustment set
        d.SetCensoring(column="hba1c"),
        d.SetCensoring(column="hba1c", acknowledged=True),
        d.SetClusters(column="site", adjust="fixed_effects"),
        d.SetClusters(column="site", adjust="cluster_only"),
        d.SetClusters(column=None, acknowledged=True),
        d.SetEstimand(exposure="protein_g", effect="total", contrast="substitution",
                      measure="mean_difference"),
        d.SetEstimand(exposure="fiber_g", effect="direct", measure="odds_ratio"),
        d.SetAdjustment(exposure="protein_g", answers={
            "age": d.CovariateAnswers(causes_exposure="yes", causes_outcome="yes",
                                      after_exposure="no"),
            "sex": d.CovariateAnswers(causes_exposure="yes", causes_outcome="yes",
                                      after_exposure="no"),
            "bmi": d.CovariateAnswers(causes_exposure="unknown", causes_outcome="yes",
                                      after_exposure="unknown"),
            "ldl": d.CovariateAnswers(causes_exposure="no", causes_outcome="yes",
                                      after_exposure="yes", keep=True, acknowledged=True)}),
        # DATAIN (V2 definition of done §1): a join on a shared identifier, a codebook import
        d.JoinFiles(file="f0123456789", on="participant_id", name="labs.csv",
                    counts=d.JoinCounts(relation="one-to-one", table_rows=600, file_rows=580,
                                        matched_keys=290, table_unmatched=20, file_unmatched=0,
                                        rows=600, added_columns=3)),
        d.ImportCodebook(codebook="c0123456789", name="dictionary.csv", form="table",
                         items=[d.ReadingItem(reading="unit", column="weight", value="kg"),
                                d.ReadingItem(reading="code_or_count", column="sex",
                                              value="code"),
                                d.ReadingItem(reading="sex_coding", column="sex",
                                              value="female=2,male=1")],
                         labels={"weight": "Weight (kg)"},
                         asked=[d.CodebookConflict(column="height_cm", field="unit", says="m",
                                                   values="a median of 166, a human height only "
                                                          "in cm")],
                         n_entries=12, n_matched=10),
        # MS7: how a batch column is handled, and an exposure family's multiplicity
        d.SetBatch(column="batch", method="covariate", figures=True),
        d.SetBatch(column="batch", method="reference_combat"),
        d.SetBatch(column="batch", method="not_a_batch"),
        d.SetMultiplicity(method="bh"),
        d.SetMultiplicity(method="none", acknowledged=True),
        d.SetScales(scales=[
            d.ScaleSpec(name="stress_score", items=["pss_1", "pss_2", "pss_3", "pss_4"],
                        reverse=["pss_4"], low=0, high=4, kind="reflective",
                        correction="regression_calibration", instrument="PSS-4"),
            d.ScaleSpec(name="diet_score", items=["dq_1", "dq_2", "dq_3"], low=0, high=10,
                        kind="formative", role="covariate", correction="regression_calibration",
                        reliability="test_retest", retest=["dq_1_t2", "dq_2_t2", "dq_3_t2"]),
        ]),
        # The NCI usual-intake method: one component's distribution, with the EAR's share
        d.SetUsualIntake(nutrient="protein_g", model="amount_only", order_column="recall",
                         weekend=["weekend"], cutoff=46, cutoff_kind="EAR"),
    ]


# Kinds whose shapes exist but whose sentences a voice agent still owes. M2's voice agent emptied
# it; a kind may only leave it with a representative here and a sentence in voice.py. Never add to
# it except when a milestone's contract adds kinds.
OWED_BY_M2_VOICE: set[str] = set()


def test_every_decision_kind_has_a_representative_here():
    from typing import get_args

    union = get_args(get_args(d.Decision)[0])
    kinds = {m.model_fields["kind"].default for m in union} - OWED_BY_M2_VOICE
    assert kinds == {x.kind for x in representative_decisions()} == set(voice.kinds()) - OWED_BY_M2_VOICE


@pytest.mark.parametrize("decision", representative_decisions(), ids=lambda x: x.kind)
@pytest.mark.parametrize("with_context", [False, True], ids=["bare", "context"])
def test_every_decision_sentence_is_finished_and_free_of_machinery(decision, with_context, tmp_path):
    state = ProjectState(target="hba1c", task="regression", lens=["dietary"],
                         grain={"grain": "repeated", "id_column": "participant_id"},
                         repeat_kind={"repeat_kind": "repeats"})
    ctx = None
    if with_context:
        table = Ingested(SAMPLES / "dietary_recalls.csv", tmp_path)
        ctx = {"frame": table.frame(), "columns": table.info["columns"],
               "n_rows": table.info["n_rows"], "n_cohort": 580, "records": [], "n_holdout": 120,
               "levels": [0, 1],
               "finding": {"id": "x", "title": "5 items carry values outside the 1–5 scale",
                           "affected_columns": ["item_03", "item_14"]},
               "repair": {"key": "to_missing", "label": "Set to missing",
                          "consequence": "Sentinel codes become blanks; the missing-values answer "
                                         "handles them.", "row_local": True}}
    text = voice.sentence_for(decision, state, ctx)
    assert text and text.strip() == text
    assert text.endswith("."), text
    assert not voice.machinery(text), (text, voice.machinery(text))
    assert "[object" not in text and "{" not in text and "None" not in text
    assert "_" not in re.sub(r"`[^`]*`", "", text), f"an identifier outside a data chip: {text}"


# ── composed card text: the extended gate (M2_CONTRACT §6) ───────────────────
# Teaching entries are budgeted where they are written. Text the app composes from data onto a
# card or the stage is budgeted here, on every fixture: the proposals' notes as the card joins them,
# every option's reason, finding summaries, coach notes and lines, and preview notes. The M1 energy
# card's nested-parts note (~70 words with the line above it) would fail the first.

C = teaching.COMPOSED_BUDGETS
DIRECTIVE = re.compile(r"\b(choose|pick|select|recommend\w*|should|best|prefer\w*|use the|go with)\b",
                       re.I)


def _views_and_notes(result) -> list[tuple[str, str, str]]:
    """(where, text, budget) for one PreviewResult: its coach notes and its note."""
    out = []
    for v in result.views:
        assert len(v.coach) <= MAX_COACH, (result.kind, v.kind, len(v.coach))
        out += [(f"{result.kind} {v.kind} coach", n.text, "coach") for n in v.coach]
    if result.note:
        out.append((f"{result.kind} note", result.note, "preview_note"))
    return out


def _target_for(info: dict, roles: dict[str, str]) -> str | None:
    """A numeric column that is not an identifier, standing in for an outcome."""
    for c in info["columns"]:
        name = c["name"]
        if c["dtype"] in ("numeric", "integer") and roles.get(name) in ("covariate", "exposure") \
                and int(c["n_unique"]) > 2:
            return name
    return None


@pytest.fixture(scope="module")
def composed(tables):
    """(file, where, text, budget key) for every piece of composed text on every fixture."""
    import numpy as np

    from turbotab.core import evidence, fact_previews, row_previews  # noqa: F401 - builders
    from turbotab.core.consequences import PreviewContext, plan
    from turbotab.core.models import previews  # noqa: F401 - the energy and model builders
    from turbotab.core.stages.data import profile_stage
    from turbotab.core.stages.proposals import proposals_stage
    from turbotab.core.stages.rows import roles_stage

    out = []
    for name, table, lens, findings in tables:
        every = sorted(set(lens) | {"dietary"})  # the energy and exclusion proposals everywhere
        bare = ProjectState(lens=every)
        profile = table.run(profile_stage, bare)
        roles_artifact = table.run(roles_stage, bare, {"profile": profile})
        roles = {e["column"]: e["proposed"] for e in roles_artifact["columns"]}
        target = "glucose" if name == NHANES.name else _target_for(table.info, roles)
        roles.pop(target, None)
        state = ProjectState(lens=every, target=target, roles=roles)
        proposals = table.run(proposals_stage, state,
                              {"roles": roles_artifact, "profile": profile})
        energy = proposals["energy"]
        if energy:
            out.append((name, "energy notes", " ".join(energy["notes"]), "card_line"))
            out += [(name, f"energy {m}", v["reason"], "option_reason")
                    for m, v in energy["applicability"].items() if not v["ok"]]
            out += [(name, "not adjusted", e["reason"], "option_reason")
                    for e in energy["not_adjusted"]]
        out += [(name, "exclusion label", p["label"], "option_reason") for p in proposals["exclusions"]]
        out += [(name, "missing reason", e["reason"], "option_reason")
                for e in proposals["missing"]["columns"]]
        out += [(name, f"card coach {k}", line["text"], "coach") for k, line in proposals["coach"].items()]
        out += [(name, "finding summary", f["summary"], "finding_summary") for f in findings]

        store = table.store()
        everything = np.arange(int(store.n_rows), dtype=np.int64)

        def ctx_for(state, training=None, _store=store, _p=proposals):
            return PreviewContext(project_id=name, state=state, datastore=_store,
                                  artifact=lambda s, _p=_p: _p if s == "proposals" else None,
                                  training_row_ids=training, cohort_row_ids=None)

        # The lens preview under the lenses that fit; each single lens, and all five at once (the
        # longest wording), are previewed in test_every_lens_preview_says_what_it_adds.
        decisions = [d.SetMissing(strategy="complete_case"), d.SetMissing(strategy="impute"),
                     d.SetPurpose(purpose="prediction"), d.SetPurpose(purpose="inference"),
                     d.SetLens(lenses=lens)]
        decisions += [d.SetExclusions(rules=[p["rule"]]) for p in proposals["exclusions"]]
        if target:
            decisions.append(d.SetTarget(column=target))
        for decision in decisions:
            out += [(name, *x) for x in _views_and_notes(plan(decision, ctx_for(state), basis=""))]
        if energy and energy["energy_column"] and energy["nutrients"]:
            exposures = {**roles, energy["energy_column"]: "energy",
                         **{n: "exposure" for n in energy["nutrients"]}}
            modeled = state.model_copy(update={"roles": exposures})
            for method in ("none", "standard", "residual", "density_multivariate", "density"):
                decision = d.SetEnergyAdjustment(method=method, energy_column=energy["energy_column"],
                                                 nutrients=energy["nutrients"])
                result = plan(decision, ctx_for(modeled, everything), basis="")
                out += [(name, *x) for x in _views_and_notes(result)]
        ectx = evidence.EvidenceContext(state=state, datastore=store)
        for finding in findings:
            result = evidence.evidence(finding, ectx)
            out += [(name, *x) for x in _views_and_notes(result)]
    return out


def test_the_gate_reads_every_fixture(composed):
    files = {name for name, *_ in composed}
    assert len(files) >= len(SAMPLE_CSVS)
    coached = [t for _, where, t, key in composed if key == "coach"]
    assert len(coached) > 50, "the coach says something on most fixtures"
    assert any("under-reporting" in t for t in coached)


@pytest.mark.parametrize("lens", [*LENSES, "all five"])
def test_every_lens_preview_says_what_it_adds(lens, tables):
    """Each lens, on the fixture made for it: a lineage of the columns as read, and a note naming
    the findings its pack raises and the questions it adds — never "Nothing can be shown"."""
    from turbotab.core import fact_previews  # noqa: F401 - registers the builder
    from turbotab.core.consequences import PreviewContext, plan

    fixture = {"dietary": "dietary_recalls.csv", "clinical": "clinical_longitudinal.csv",
               "metabolomics": "metabolomics_untargeted.csv", "genomics": "genomics_expression.csv",
               "survey": "survey_sentinels.csv", "all five": "dietary_recalls.csv"}[lens]
    lenses = list(LENSES) if lens == "all five" else [lens]
    table = next(t for name, t, _, _ in tables if name == fixture)
    ctx = PreviewContext(project_id=fixture, state=ProjectState(), datastore=table.store(),
                         artifact=lambda s: None, training_row_ids=None, cohort_row_ids=None)
    result = plan(d.SetLens(lenses=lenses), ctx, basis="")
    assert [v.kind for v in result.views] == ["lineage"]
    whose = "five packs'" if lens == "all five" else f"{lens} pack's"
    assert re.search(rf"The {whose} checks raise `\d+` findings? on this table", result.note)
    assert not over(result.note, C["preview_note"]) and "Nothing" not in result.note
    assert not over(result.views[0].caption, 20), result.views[0].caption
    if "dietary" in lenses:
        assert "energy adjustment and substitution curves join the questions" in result.note


def test_composed_card_text_is_within_budget(composed):
    problems = [f"{name} · {where}: {voice.words(text)} words > {C[key]}: {text}"
                for name, where, text, key in composed if text and over(text, C[key])]
    assert not problems, "\n".join(problems)


def test_composed_card_text_is_finished_and_free_of_machinery(composed):
    for name, where, text, key in composed:
        if not text:
            continue
        assert not voice.machinery(text), (name, where, text, voice.machinery(text))
        if key in ("card_line", "coach", "preview_note"):
            assert re.search(r"[.?!][)\"'”’]*$", text), (name, where, text)


def test_coach_notes_never_name_an_option_as_the_answer(composed):
    """The coach states what the data shows; it never tells the user which option to take."""
    labels = {o.label.lower() for e in teaching.entries() for o in e.options
              if len(o.label.split()) >= 2}
    for name, where, text, key in composed:
        if key != "coach":
            continue
        assert not DIRECTIVE.search(text), (name, where, text)
        assert not any(label in text.lower() for label in labels), (name, where, text)


def test_the_m1_energy_note_would_fail_the_gate_and_its_parts_now_fit():
    """The note the M1 energy card showed above its options on NHANES, as the card composed it,
    is over the card-line budget; the same table's reading now folds it into the partition
    option's reason (within budget) and the "nested" term card."""
    import numpy as np
    import pandas as pd

    from turbotab.core.stages.proposals import build_proposals

    m1_card = (
        "`fat_total` tracks `kcal` most closely; `protein`, `sugar`, `carb`, `fat_total`, "
        "`fat_sat`, `fat_mon` and `fat_poly` are adjusted together. `sugar` is a part of `carb`: "
        "choosing them together counts carbohydrate's energy twice in a partition; a substitution "
        "moves the parts with their total. `fat_sat`, `fat_mon` and `fat_poly` are parts of "
        "`fat_total`: choosing them together counts fat's energy twice in a partition; a "
        "substitution moves the parts with their total.")
    assert voice.words(m1_card) > 60 and over(m1_card, C["card_line"])

    rng = np.random.default_rng(0)
    n = 400
    sat, mon, poly = rng.gamma(4, 6, n), rng.gamma(4, 7, n), rng.gamma(3, 4, n)
    fat = sat + mon + poly + rng.gamma(2, 2, n)
    sugar = rng.gamma(5, 15, n)
    carb = sugar + rng.gamma(8, 20, n)
    protein = rng.gamma(9, 9, n)
    frame = pd.DataFrame({"kcal": 4 * protein + 4 * carb + 9 * fat + rng.normal(0, 30, n),
                          "protein": protein, "sugar": sugar, "carb": carb, "fat_total": fat,
                          "fat_sat": sat, "fat_mon": mon, "fat_poly": poly,
                          "glucose": rng.normal(100, 15, n)})
    columns = [{"name": c, "dtype": "numeric", "n_unique": n, "n_missing": 0} for c in frame]
    reading = build_proposals(frame, columns, lens=["dietary"], target="glucose")["energy"]
    assert reading["notes"] == []
    partition = reading["applicability"]["partition"]
    assert not partition["ok"] and "nested in" in partition["reason"], partition
    assert not over(partition["reason"], C["option_reason"]), partition["reason"]
    nested = {t.term: t.definition for t in teaching.entry("energy_adjustment").terms}["nested"]
    assert "substitution moves it with its total" in nested
