"""The methods reference and the review packets stay what the code generates (V2 definition of done
§4: "a methods reference generated from the contracts"; "a per-domain review packet").

* The committed ``METHODS_REFERENCE.md`` is regenerated from the registry and must be identical:
  a contract, an option, a label, a relation or a family that changes without regenerating it
  fails here, naming the command that fixes it.
* Every committed review packet is regenerated from the registry and its stored journey captures
  (no fit runs here): a registry that moved on fails until the packet is rebuilt with ``--reuse``.
* The lens catalog covers the registry exactly: a contract or a family registered without a lens
  fails, and so does a lens entry for something no longer registered. Every packet lists every
  registered contract and family, so the catalog decides how much of a method a packet shows,
  never whether the app's offer is hidden from a lens.
* Every decision kind the decision log accepts records a contract's method, a gap's, or says why
  it records none; every gap names real code.
* Where the app ranks a method by the data, the packets say what it offers first under each
  condition (``defaults``), and the reference journeys took exactly that where they took the app's
  first-ranked option: the split, the effect measure, the form, the omics normalization, the
  missing values.
* Every capture records where each answer came from truthfully: the readings the journey injected
  are listed apart from the fixture's declared truth and shown in the packet, and a form or a
  measure no reading declares is the one the app ranked first.
* A bundle that contradicts itself (an adjustment set the model matrix does not hold) is flagged.
* Every sentence a contract names by ``module:function`` resolves.
* The v2 citation draft is Citation File Format 1.2.0 with the paper as its preferred citation, and
  the root ``CITATION.cff`` (Classic's) is not the draft.
"""
from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest

from turbotab.core.reference import catalog, defaults, journeys, methods, packet

REPO = Path(__file__).resolve().parents[3]
DRAFT = REPO / "docs" / "turbotab-next" / "release" / "CITATION.cff.draft"


def test_the_methods_reference_is_what_the_contracts_generate():
    committed = methods.OUT.read_text(encoding="utf-8") if methods.OUT.is_file() else ""
    assert committed == methods.render(), (
        f"{methods.OUT.relative_to(REPO)} is stale: run `{methods.COMMAND}`")


@pytest.mark.parametrize("lens", catalog.LENSES)
def test_each_review_packet_is_what_the_registry_and_its_captures_generate(lens):
    path = packet.OUT / f"{lens}.md"
    assert path.is_file(), f"no packet for {lens}: run `{packet.COMMAND} --lens {lens}`"
    assert path.read_text(encoding="utf-8") == packet.render(lens, packet.captures_for(lens)), (
        f"{path.relative_to(REPO)} is stale: run `{packet.COMMAND} --lens {lens} --reuse`")


def test_every_contract_and_family_has_its_lenses_and_nothing_else_does():
    from turbotab.core.contracts import contracts

    registered = set(contracts())
    assert registered, "no contracts registered: the check would pass vacuously"
    assert sorted(registered - set(catalog.CONTRACT_LENSES)) == [], \
        "contracts with no lens: add them to turbotab/core/reference/catalog.py"
    assert sorted(set(catalog.CONTRACT_LENSES) - registered) == [], \
        "lens entries for contracts that are not registered"
    families = {f.key: tuple(f.review_lenses) for f in methods.families()}
    assert sorted(k for k in families if catalog.lenses_of_family(k) != families[k]) == []
    allowed = {*catalog.LENSES, catalog.SHARED}
    for table in (catalog.CONTRACT_LENSES, families):
        for key, lenses in table.items():
            assert lenses and set(lenses) <= allowed, key
            assert not (catalog.SHARED in lenses and len(lenses) > 1), key


def test_every_gap_names_real_code_and_no_contract():
    from turbotab.core import custom_sound
    from turbotab.core.contracts import contracts
    from turbotab.core.decisions import Decision

    union = get_args(Decision)[0]
    kinds = {m.model_fields["kind"].default for m in get_args(union)}
    registered = contracts()
    keys = [g.key for g in catalog.GAPS]
    assert len(keys) == len(set(keys))
    labeled = set(get_args(custom_sound.LabeledQuestion.model_fields["question"].annotation))
    for g in catalog.GAPS:
        assert g.key not in registered, f"{g.key} has a contract: it is no longer a gap"
        assert set(g.decisions) <= kinds, (g.key, sorted(set(g.decisions) - kinds))
        for module in g.modules:
            importlib.import_module(module)
        if g.labels:
            assert g.labels.split(":", 1)[1] in labeled, g.key
        assert set(g.contracted_parts) <= set(registered), g.key
        assert set(g.lenses) <= {*catalog.LENSES, catalog.SHARED}, g.key


def test_every_sentence_a_contract_names_resolves():
    for c in methods.ordered_contracts():
        if isinstance(c.sentence, str) and re.fullmatch(r"[\w.]+:[\w.]+", c.sentence):
            assert callable(methods.resolve(c.sentence)), c.key


def test_every_lens_has_a_journey_for_each_purpose_and_every_capture_one_journey():
    for lens in catalog.LENSES:
        assert sorted(j.purpose for j in journeys.for_lens(lens)) == ["inference", "prediction"]
    stored = {p.stem for p in journeys.CAPTURES.glob("*.json")}
    assert stored <= set(journeys.JOURNEYS), sorted(stored - set(journeys.JOURNEYS))


def test_the_citation_draft_cites_the_paper_first_and_v2_as_the_software():
    yaml = pytest.importorskip("yaml")
    draft = yaml.safe_load(DRAFT.read_text(encoding="utf-8"))
    assert draft["cff-version"] == "1.2.0"
    assert draft["type"] == "software" and str(draft["version"]) == "2.0.0"
    for key in ("message", "title", "authors"):
        assert draft[key], key
    paper = draft["preferred-citation"]
    assert paper["type"] == "article" and paper["title"] and paper["authors"]
    classic = yaml.safe_load((REPO / "CITATION.cff").read_text(encoding="utf-8"))
    assert str(classic["version"]) == "1.0.0"  # the root file is still Classic's
    assert any(str(r.get("version")) == str(classic["version"]) for r in draft["references"])


# ── every method listed, every decision accounted for ───────────────────────


def _decision_kinds() -> set[str]:
    from turbotab.core.decisions import Decision

    return {m.model_fields["kind"].default for m in get_args(get_args(Decision)[0])}


@pytest.mark.parametrize("lens", catalog.LENSES)
def test_every_packet_lists_every_registered_contract_and_family(lens):
    """The shelf never shortens the list, and no contract is gated on a lens by the registry: a
    packet that left out a family or a method another lens reviews would hide what the app offers
    (the clinical and survey prediction journeys chose the screened elastic net, which their
    packets once never described)."""
    from turbotab.core.contracts import contracts

    text = packet.render(lens, packet.captures_for(lens))
    families = text[text.index("### 1.2"):text.index("### 1.3")]
    for f in methods.families():
        assert f"`{f.key}`" in families, (lens, f.key)
    listed = text[:text.index("## 2 ")]
    for key in contracts():
        assert f"`{key}`" in listed, (lens, key)
    for capture in packet.captures_for(lens).values():
        for a in (capture or {}).get("answers") or []:
            if a["kind"] == "select_models":
                for key in json.loads(a["summary"])["models"]:
                    assert f"(`{key}`)" in families, (lens, key)


def test_every_decision_kind_records_a_method_a_gap_or_says_why_not():
    """A decision the Router records and the export writes a sentence for (model updating, the
    outcome's log scale, the follow-up, the unit, the re-seal, categorical codes) is a method the
    reviewers must see; it cannot ship as neither a contract's nor a gap's."""
    from turbotab.core.contracts import contracts

    kinds = _decision_kinds()
    contracted = {c.decision for c in contracts().values() if c.decision}
    gapped = {d for g in catalog.GAPS for d in g.decisions}
    undeclared = set(catalog.UNDECLARED_DECISIONS)
    not_methods = set(catalog.NOT_METHODS)
    assert sorted(kinds - contracted - gapped - undeclared - not_methods) == []
    assert not_methods <= kinds and undeclared <= kinds
    assert not not_methods & (contracted | gapped | undeclared), "a method said to be none"
    for kind, key in catalog.UNDECLARED_DECISIONS.items():
        assert key in contracts(), (kind, key)
        assert contracts()[key].decision != kind, f"{key} declares {kind}: drop it from the map"
    from turbotab.core.export.methods import SECTION_OF

    for kind in SECTION_OF:
        assert kind in kinds, kind


# ── what the app offers first, by the data ───────────────────────────────────


def test_every_data_ranked_default_names_a_method_and_states_true_premises():
    from turbotab.core import custom_sound
    from turbotab.core.contracts import contracts

    labeled = set(get_args(custom_sound.LabeledQuestion.model_fields["question"].annotation))
    labeled |= {q for q, _ in methods.labeled_questions()}
    for key in defaults.BY_CONDITION:
        assert key in contracts() or key in labeled, key
    for key, question in defaults.THROUGH.items():
        assert key in contracts() and question in defaults.BY_CONDITION, key
        for purpose in defaults.PURPOSES:
            for case in defaults.cases(key, purpose) or []:  # a stale premise raises here
                assert case.when and case.first, (key, purpose)


def test_the_genomics_default_normalization_is_the_one_its_counts_are_offered():
    """genomics.md once gave `pqn_log2` (a metabolomics method) as the genomics lens's
    normalization: the registry's static rank 1, though on counts the app offers log-CPM with TMM
    first and never PQN."""
    counts = next(c for c in defaults.cases("omics_normalization", "prediction")
                  if c.when.startswith("raw counts"))
    assert counts.key == "log_cpm_tmm"
    text = packet.render("genomics", packet.captures_for("genomics"))
    own = text[text.index("### 3.1"):text.index("### 3.2")]
    assert "`pqn_log2`" not in own and packet.BY_DATA in own
    ranked = text[text.index("### 3.4"):text.index("## 4 ")]
    assert "raw counts (sequencing)" in ranked and "`log_cpm_tmm`" in ranked


@pytest.mark.parametrize("lens", catalog.LENSES)
def test_a_packet_never_states_a_static_default_the_app_ranks_by_the_data(lens):
    """Every packet once gave `odds_ratio` as the inference measure and `bootstrap` as the
    prediction split, the registry's rank 1, though the app ranks both on the data (a common
    event's marginal measures first; a holdout at 20,000 units; time-ordered folds): each such
    cell defers to §3.4, which lists every condition with what the app offers first."""
    text = packet.render(lens, packet.captures_for(lens))
    section = text[text.index("## 3 "):text.index("## 4 ")]
    ranked = section[section.index("### 3.4"):]
    rows = {line.split(" | ")[0].rsplit("(`", 1)[-1].rstrip("`) "): line
            for line in section[:section.index("### 3.4")].splitlines()
            if line.startswith("| ") and "(`" in line}
    assert rows, lens
    for key, row in rows.items():
        for i, purpose in enumerate(packet.PURPOSES):
            cell = row.split(" | ")[1 + i].rstrip(" |")
            cases = defaults.cases(key, purpose)
            assert cell.startswith(packet.BY_DATA) == (cases is not None), (lens, key, purpose, cell)
            for case in cases or []:
                assert f"| {purpose} | {methods.cell(case.when)} | {methods.cell(case.first)} |" \
                    in ranked, (lens, key, purpose, case.when)


def _case(key: str, purpose: str, when: str) -> defaults.Case:
    found = [c for c in defaults.cases(key, purpose) or [] if c.when.startswith(when)]
    assert len(found) == 1, (key, purpose, when, [c.when for c in defaults.cases(key, purpose)])
    return found[0]


# The condition each journey met, in the words of its case (``defaults``).
SPLIT_WHEN = {"dietary-prediction": "no time order, 20,000 units or more, the usual 20% holdout at",
              "clinical-prediction": "rows in time order (the temporal seal), the usual 20% holdout "
                                     "below",
              "metabolomics-prediction": "no time order, fewer than",
              "genomics-prediction": "no time order, fewer than",
              "survey-prediction": "no time order, fewer than"}
MEASURE_WHEN = {"dietary-inference": "a numeric outcome", "clinical-inference": "a numeric outcome",
                "survey-inference": "a yes/no outcome whose event share is above",
                "metabolomics-inference": "an exposure family (each exposure in turn), a yes/no",
                "genomics-inference": "an exposure family (each exposure in turn), a yes/no"}
EXCLUSIONS_WHEN = {"dietary-inference": "the screens can run"}
NORMALIZATION_WHEN = {"metabolomics": "raw intensities (mass spectrometry)",
                      "genomics": "raw counts (sequencing)"}


def _summaries(capture: dict[str, Any], kind: str, source: str) -> list[str]:
    return [a["summary"] for a in capture["answers"] if a["kind"] == kind
            and not a.get("refusal") and a["source"].startswith(source)]


def _answers(capture: dict[str, Any], kind: str,
             source: str | None = None) -> list[tuple[dict[str, Any], str]]:
    """Each recorded ``kind`` answer (from ``source``, a prefix, when given) as its body: the
    capture's one-line summary, which is the body's JSON unless it was cut (a refusal's way forward
    can be long; a cut one is never one these tests read)."""
    out = []
    for a in capture["answers"]:
        if a["kind"] != kind or a.get("refusal"):
            continue
        if source is not None and not a["source"].startswith(source):
            continue
        if a["summary"].endswith("…") and a["source"].startswith(journeys.OWN_EXITS):
            continue
        out.append((json.loads(a["summary"]), a["source"]))
    return out


@pytest.mark.parametrize("name", sorted(journeys.JOURNEYS))
def test_each_journey_took_what_the_packet_says_the_app_offers_first(name):
    """Where a journey took the app's first-ranked option, it is the one §3 says the app offers
    first under the journey's condition: the split at 21,849 rows is a holdout and under the
    temporal seal time-ordered folds (not the static `bootstrap`); a common event's measure is the
    marginal risk difference (not `odds_ratio`); the form is the card's spline (not a straight
    line); counts are normalized by log-CPM with TMM (not PQN)."""
    spec, capture = journeys.JOURNEYS[name], journeys.load_capture(name)
    if capture is None or (capture.get("export") or {}).get("status") != 200:
        pytest.skip(f"{name} has no capture that reached the export")
    purpose = spec.purpose
    split = _case("split", purpose, SPLIT_WHEN.get(name, "at any size"))
    splits = _answers(capture, "set_split", "the app's first-ranked")
    assert splits, name
    for body, _ in splits:
        assert (body["holdout"] > 0) == (split.key == "holdout"), (name, body, split.when)
        assert body["validation"] == ("kfold" if split.key == "holdout" else split.key), name
    if purpose == "inference":
        measure = _case("effect_measure", "inference", MEASURE_WHEN[name])
        estimands = _answers(capture, "set_estimand")
        assert estimands and all(b["measure"] == measure.key for b, _ in estimands), name
        spline = {c.key for c in defaults.cases("functional_form", "inference")}
        forms = _answers(capture, "set_forms")
        assert forms, name
        for body, source in forms:
            assert source == journeys.WP17_SOURCES["set_forms"], name
            assert {f["form"] for f in body["forms"].values()} <= spline, (name, body)
        ranked = {(r["reading"], r["column"]): r["value"] for r in capture["app_ranked"]}
        for body, _ in forms:
            for column, form in body["forms"].items():
                assert ranked.get(("form", column)) == form["form"], (name, column)
    if spec.lens in NORMALIZATION_WHEN:
        first = _case("omics_normalization", purpose, NORMALIZATION_WHEN[spec.lens]).key
        repairs = [b for b, _ in _answers(capture, "apply_repair") if b["finding_id"] == "omics_scale"]
        assert [b["option"] for b in repairs] == [first], (name, repairs)
    # (a screen's rule and the energy model's nutrients make long summaries: read their keys)
    energy = {c.key for c in defaults.cases("energy_adjustment", purpose)}
    for summary in _summaries(capture, "set_energy_adjustment", "the app's first-ranked"):
        assert re.match(r'\{"method": "(\w+)"', summary)[1] in energy, (name, summary)
    screens = _case("exclusions", purpose,
                    EXCLUSIONS_WHEN.get(name, "whatever" if purpose == "prediction" else "no screen"))
    for summary in _summaries(capture, "set_exclusions", "the app's first-ranked"):
        kinds = re.findall(r'"kind": "(\w+)"', summary)
        assert kinds == ([] if screens.key == "keep_every_row" else [screens.key.split("_")[0]]), \
            (name, summary)
    from turbotab.core import custom_sound

    missing = custom_sound.labels_for("missing", purpose).options[0].key
    for body, _ in _answers(capture, "set_missing", "the app's first-ranked option"):
        assert body == journeys.MISSING_BODIES[missing], name


# ── where each answer came from ──────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(journeys.JOURNEYS))
def test_every_capture_says_truly_where_its_answers_came_from(name):
    """The readings a journey injects (an exposure, causal places, units) are its own, not the
    fixture's declared truth: the capture keeps them apart and the packet shows each one."""
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    spec, capture = journeys.JOURNEYS[name], journeys.load_capture(name)
    assert capture is not None, name
    readings = capture.get("readings")
    assert readings is not None, f"{name}'s capture predates the readings record: rerun it"
    declared = FIXTURE_TRUTHS.get(spec.fixture_key, {})
    assert spec.fixture_key and readings["fixture"] == spec.fixture_key, name
    for key, value in readings["from_fixture"].items():
        assert str(declared.get(key)) == value, (name, key)
    for key, value in readings["journey_own"].items():
        assert str(declared.get(key)) != value, (name, key)
    for a in capture["answers"]:
        if a["kind"] in journeys.READING_KINDS:
            assert a["source"] == journeys.READINGS_SOURCE, (name, a["kind"])
        if a["kind"] in ("set_estimand", "set_adjustment", "set_forms", "set_clusters",
                         "set_time_varying") and not a["source"].startswith(journeys.OWN_EXITS):
            assert a["source"] == journeys.WP17_SOURCES[a["kind"]], (name, a["kind"])
    text = packet.render(spec.lens, packet.captures_for(spec.lens))
    for key in readings["journey_own"]:
        what, _, column = key.partition(":")
        assert f"| `{what}` | `{column}` |" in text, (name, key)


def test_a_journey_names_the_exposure_its_plan_estimates():
    """The survey inference journey once asked about the support scale while its plan estimated
    age (a declared scale cannot be the estimand's exposure): the question a packet heads a
    journey with names the exposure the plan reports, and the limit is a gap the packet lists."""
    for name, spec in journeys.JOURNEYS.items():
        capture = journeys.load_capture(name)
        for body, _ in _answers(capture or {"answers": []}, "set_estimand"):
            if not body.get("family"):
                assert re.search(rf"\b{re.escape(body['exposure'])}\b", spec.question), \
                    (name, body["exposure"], spec.question)
    gap = next(g for g in catalog.GAPS if g.key == "scale_as_exposure")
    assert "survey" in gap.lenses
    survey = packet.render("survey", packet.captures_for("survey"))
    assert "(`scale_as_exposure`)" in survey or "scale_as_exposure" in survey


# ── a bundle that contradicts itself ─────────────────────────────────────────


def test_a_bundle_whose_adjustment_set_the_model_matrix_does_not_hold_is_flagged():
    export = {"model_matrix": ["sex_M", "education_Graduate", "age", "age'", "support_scale"],
              "table2": [{"model": "Unadjusted", "adjusted_for": "nothing", "terms": ["age"]},
                         {"model": "Model 2 (primary)", "adjusted_for": "sex, education, item_01",
                          "terms": ["age", "age'"]}]}
    flags = packet.matrix_flags(export)
    assert len(flags) == 2
    assert "`item_01`" in flags[0] and "`sex`" not in flags[0]
    assert "`support_scale`" in flags[1] and "`age'`" not in flags[1]
    whole = {**export, "model_matrix": ["sex_M", "education_Graduate", "item_01", "age", "age'"]}
    assert packet.matrix_flags(whole) == []
    # a capture's Table 2 keeps one row per declared model with its terms (older ones, a row per term)
    per_term = {**export, "table2": [{"model": "Model 2 (primary)", "adjusted_for": "sex, item_01",
                                      "term": t} for t in ("age", "age'")]}
    assert packet.matrix_flags({**per_term, "model_matrix": ["sex_M", "item_01", "age", "age'"]}) == []


@pytest.mark.parametrize("name", sorted(journeys.JOURNEYS))
def test_no_reference_journey_exports_a_bundle_that_contradicts_itself(name):
    """The survey inference bundle once said Model 2 was adjusted for item_01 … item_10 while its
    model matrix held `support_scale` and none of the items (the effects stage named a scale by the
    items scored into it): every journey's bundle holds the columns its Table 2 says."""
    capture = journeys.load_capture(name)
    if capture is None or (capture.get("export") or {}).get("status") != 200:
        pytest.skip(f"{name} has no capture that reached the export")
    assert packet.matrix_flags(capture["export"]) == [], name


# ── what the app offers, where the registry's rank says otherwise ────────────


def test_the_grouping_contract_offers_what_the_grouping_card_offers():
    """Under prediction the grouping card offers ["group", "none"], and the contract had no
    "group" option, so every packet gave `none` as the prediction default: the contract's options
    for each purpose, in order and without those it does not offer, are the card's."""
    from turbotab.core.contracts import contract
    from turbotab.core.decisions import ProjectState
    from turbotab.core.estimand import grouping_card

    roles = {"columns": [{"column": "site", "proposed": "cluster"}]}
    for purpose in packet.PURPOSES:
        card = grouping_card(ProjectState.model_validate({"purpose": purpose, "target": "y"}), roles)
        offered = [o["key"] for o in contract("grouping_by_structure").options_for(purpose)
                   if o["rung"] != "not_offered"]
        assert offered == card["options"], purpose
    text = packet.render("clinical", packet.captures_for("clinical"))
    row = next(line for line in text.splitlines() if line.startswith(
        "| The grouping question, asked by structure (`grouping_by_structure`)"))
    assert row.split(" | ")[1].startswith("`group` (recommended)"), row


def _kind_contract() -> dict[str, str]:
    from turbotab.core.contracts import contracts

    out = {k: v for k, v in catalog.UNDECLARED_DECISIONS.items()}
    for key, c in contracts().items():
        if c.decision:
            out.setdefault(c.decision, key)
    return out


def test_a_method_no_question_asks_says_so_and_is_never_called_a_default():
    """No Router question, proposals card or control posts `set_batch` or `set_scales`; the
    journeys declared both themselves, yet the packets gave reference ComBat as the prediction
    default, "asked, or stated in the Record". Every decision a journey declared without being
    asked belongs to a contract the catalog says is never asked; each such claim is checked against
    the Router, the proposals and the frontend; and its packets and reference entry say so."""
    from turbotab.core.contracts import contracts
    from turbotab.core.interview import QUESTION_KEYS, SLOT_OF

    kinds = _kind_contract()
    for name in journeys.JOURNEYS:
        for a in (journeys.load_capture(name) or {}).get("answers") or []:
            if "declared, not asked" in a["source"]:
                assert kinds.get(a["kind"]) in catalog.NOT_ASKED, (name, a["kind"])
    front = REPO / "turbotab" / "frontend" / "src"
    shown_only = {"sentences.tsx", "generated.ts"}  # the record's sentences; the API's types
    files = [f for f in front.rglob("*.ts*") if "mocks" not in f.parts and f.name not in shown_only
             and ".test." not in f.name]
    assert files
    proposals = (REPO / "turbotab" / "core" / "stages" / "proposals.py").read_text(encoding="utf-8")
    reference = methods.render()
    for key, why in catalog.NOT_ASKED.items():
        c = contracts()[key]
        kind = catalog.recorded_by(c)
        assert kind and kinds[kind] == key, key
        slot = kind.removeprefix("set_")
        assert slot not in QUESTION_KEYS and slot not in SLOT_OF.values(), (key, "a question asks it")
        assert c.question not in QUESTION_KEYS, key
        assert f'"{kind}"' not in proposals, (key, "the proposals offer it")
        for f in files:
            assert f'"{kind}"' not in f.read_text(encoding="utf-8"), (key, str(f))
        entry = reference[reference.index(f"(`{key}`)\n"):]
        assert "- **Asked:** never" in entry[:entry.index("**Options**")], key
        for lens in catalog.LENSES:
            text = packet.render(lens, packet.captures_for(lens))
            section = text[text.index("## 3 "):text.index("### 3.4")]
            for line in section.splitlines():
                if line.startswith("| ") and f"(`{key}`) |" in line:
                    for cell in line.split(" | ")[1:]:
                        assert cell.startswith(("never asked", "not offered", "refused")), \
                            (lens, key, cell)


def test_the_causal_lane_is_shown_with_its_labels_its_order_and_its_stated_default():
    """The causal question's labels and order live in `causal.options`, outside the registry and
    custom_sound, so no reviewer saw them, and §3.3 gave each estimator as an "available"
    inference default though the app states the primary model alone and offers the lane one step
    away, its order set by the exposure's kind, the outcome and the candidates: every packet and
    the reference show the lane under each condition, and each estimator's cell defers to it."""
    from turbotab.core.causal import options

    ranked = dict(defaults.causal_rankings())
    assert [o.key for o in ranked["a yes/no exposure, a yes/no outcome"]] == \
        [o.key for o in options("binary", "binary", False, False)]
    assert ranked["a yes/no exposure, a yes/no outcome"][0].key == "tmle"
    assert ranked["a continuous exposure, a numeric outcome with few candidates for n"][0].key == \
        "dml_plr"
    assert ranked["a continuous exposure, a numeric outcome with many candidates for n"][0].key == \
        "pds_lasso"
    assert all(r[-1].key == "none" for r in ranked.values())
    reference = methods.render()
    section = reference[reference.index("(`causal`)\n"):reference.index("## Gaps")]
    for o in ranked["a yes/no exposure, a numeric outcome, the surveyed population"]:
        assert f"`{o.key}`: {methods.cell(o.label)}" in section
        assert methods.cell(o.sound.reason) in section
    for lens in catalog.LENSES:
        text = packet.render(lens, packet.captures_for(lens))
        assert "(`causal`)" in text[text.index("### 1.5"):text.index("## 2 ")], lens
        for key in defaults.THROUGH:
            row = next(line for line in text.splitlines() if f"(`{key}`) |" in line
                       and line.startswith("| "))
            assert row.split(" | ")[2].startswith(packet.BY_DATA), (lens, key, row)
        ranked_rows = text[text.index("### 3.4"):text.index("## 4 ")]
        for case in defaults.cases("causal", "inference"):
            assert f"| inference | {methods.cell(case.when)} | {methods.cell(case.first)} |" in \
                ranked_rows, (lens, case.when)


def test_the_usual_intake_model_is_ranked_by_the_share_of_zero_recalls():
    """The packets gave `amount_only` as the NCI default, though above 5% zero recalls the app
    ranks the two-part model first (`usual_intake.suggested_model`)."""
    from turbotab.core.usual_intake import EPISODIC_SHARE, suggested_model

    found = {c.key: c for c in defaults.cases("nci_usual_intake", "inference")}
    assert set(found) == {"amount_only", "two_part"}
    assert suggested_model(EPISODIC_SHARE + 0.01)[0] == "two_part"
    assert suggested_model(EPISODIC_SHARE)[0] == "amount_only"
    text = packet.render("dietary", packet.captures_for("dietary"))
    row = next(line for line in text.splitlines()
               if line.startswith("| Usual-intake distribution (NCI method) (`nci_usual_intake`)"))
    assert row.split(" | ")[2].startswith(packet.BY_DATA), row


@pytest.mark.parametrize("name", sorted(journeys.JOURNEYS))
def test_no_reference_journey_meets_a_dead_end(name):
    """DoD §1, no dead end: the metabolomics inference journey once took the feature-wise
    refusal's “Complete cases”, which kept 0 of 72 rows, and the design stage failed (“Found array
    with 0 sample(s)”). No journey stops on a failed stage, and no way forward it takes leaves a
    result uncomputed."""
    capture = journeys.load_capture(name)
    assert capture is not None, name
    for n in capture.get("notes") or []:
        assert not n.startswith("stopped"), (name, n)
        assert "left a result uncomputed" not in n, (name, n)
