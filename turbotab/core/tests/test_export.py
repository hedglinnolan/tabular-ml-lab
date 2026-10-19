"""The export's parts on small, hand-made inputs (Tier A where a number could go wrong; the journeys
through the server are ``acceptance/test_export.py``)."""
from __future__ import annotations

import csv
import io
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core import plan_lock, voice
from turbotab.core.decisions import DecisionLog
from turbotab.core.export import bundle, checklists, figures, gate, matrix, methods, replay, tables
from turbotab.core.export.source import InputFile, Source
from turbotab.core.provenance import methods_text


def _say(decision: Any, state: Any) -> str:
    return voice.sentence_for(decision, state, None)


def _source(records: list[Any], *, purpose: str = "inference", interview=(), statuses=None,
            artifacts=None, bundles=None, ctx: Any = None) -> Source:
    artifacts = artifacts or {}
    return Source(
        name="test", engine_version="2.0.0.test", records=records, state=d.fold(records),
        interview=list(interview),
        statuses=statuses or {s: SimpleNamespace(status="fresh", key=f"k-{s}", error=None,
                                                 missing=[])
                              for s in ("cohort", "design", "fit", "effects")},
        artifact=lambda stage: artifacts.get(stage), bundle=lambda stage: (bundles or {}).get(stage),
        methods=methods_text(records, ctx), inputs=[], decisions_jsonl=b"")


# ── the methods ──────────────────────────────────────────────────────────────


def test_every_decision_kind_declares_the_section_its_sentence_answers():
    """A kind with no section is listed under "Other decisions"; a new kind declares its own."""
    kinds = set(d.SLOTS) - {"lock_plan", "open_seal", "reseal"}
    assert kinds <= set(methods.SECTION_OF), kinds - set(methods.SECTION_OF)
    strobe = {k for k, _, _ in methods.STROBE_SECTIONS}
    tripod = {k for k, _, _ in methods.TRIPOD_SECTIONS}
    assert all(s in strobe and t in tripod for s, t in methods.SECTION_OF.values())


def test_the_methods_follow_the_guideline_and_the_lock_closes_them(tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl")
    for decision in (d.SetLens(lenses=["dietary"]), d.SetTarget(column="dm"),
                     d.SetMissing(strategy="complete_case"), d.SetPurpose(purpose="inference"),
                     d.SetExclusions(rules=[]),
                     d.SetRoles(roles={"pid": "identifier", "fiber": "exposure", "age": "covariate"})):
        log.append(decision, sentence=_say)
    log.append(d.validate({"kind": "lock_plan"}, {"state": log.state()}), sentence=_say)
    log.append(d.SetExclusions(rules=[d.ExclusionRule(column="age", low=18, reason="adults")]),
               sentence=_say)
    doc = methods.methods_section(_source(log.records()), {"engine": {"turbotab": "x"},
                                                           "inputs": [], "decisions": {"n": 8},
                                                           "analysis_plan": {"plan_sha256": "ab"}})
    keys = [s.key for s in doc.sections]
    assert keys == ["design", "participants", "variables", "measurement", "statistical", "after",
                    "reproducibility"]
    by = {s.key: [e.kind for e in s.entries] for s in doc.sections}
    assert by["design"] == ["set_purpose"] and by["participants"] == ["set_exclusions"]
    assert by["variables"] == ["set_target", "set_roles"]
    assert by["statistical"] == ["set_missing", "lock_plan"]  # the lock closes the plan
    assert by["after"] == ["set_exclusions"]
    assert doc.markdown.index("## Study design (STROBE 4)") < doc.markdown.index(
        "## Statistical methods (STROBE 12)") < doc.markdown.index(
        "## Decisions made after the estimates were seen")
    tripod = methods.methods_section(_source(log.records()[:3], purpose="prediction"),
                                     {"engine": {}, "inputs": [], "decisions": {"n": 3},
                                      "analysis_plan": {}})
    assert tripod.guideline == "STROBE-nut"  # the purpose decides: none is declared in these three


def test_a_sentence_that_counts_rows_is_restated_with_the_counts_as_they_stand(tmp_path):
    """``voice.restated_counts``: the methods text restates the counts from its context; without
    one, or for the Record, the sentence stays as said; a disclosure lead is kept."""
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.append(d.SetTarget(column="y"), sentence=_say)
    said = log.append(d.SetMissing(strategy="complete_case"), sentence=lambda dd, s: voice.sentence_for(
        dd, s, {"n_complete": 10, "n_before": 10}))
    assert said.sentence == ("A complete-case analysis was applied: no row is missing any "
                             "predictor, so all `10` rows remain.")
    plain = methods_text(log.records())
    assert [line.sentence for line in plain.lines][-1] == said.sentence
    now = methods_text(log.records(), {"counts": {"set_missing": {"n_complete": 7,
                                                                   "n_before": 10}}})
    assert now.lines[-1].sentence == ("Rows missing any predictor were dropped (a complete-case "
                                      "analysis): `7` of `10` rows remain.")
    assert log.records()[-1].sentence == said.sentence  # the Record keeps what was said


# ── the plan under prediction ────────────────────────────────────────────────


def test_a_prediction_plan_says_no_plan_was_locked_and_never_that_nothing_was_shown(tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl")
    for decision in (d.SetTarget(column="y"), d.SetPurpose(purpose="prediction"),
                     d.SelectModels(models=["linear"])):
        log.append(decision, sentence=_say)
    doc = plan_lock.plan_document(log.records())
    assert doc.status == "declared"
    assert doc.text.startswith("This is the analysis as declared in TurboTab for prediction, "
                               "through the decision recorded on ")
    assert "no estimate has been displayed" not in doc.text
    assert not any(w in doc.text.lower() for w in plan_lock.NEVER_SAID)


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_single_confirmations_alike_but_for_their_column_are_said_in_one_sentence(tmp_path,
                                                                                  purpose):
    """The re-walk's N10: the methods listed each flag's confirmation as its own near-identical
    sentence, six of them. Single confirmations of one kind whose sentences differ only in their
    column are said once, with every column named; one alone, or one whose sentence differs
    otherwise (another role), keeps its own. The merged sentence is built here by the rule, from
    the records' own sentences."""
    flags = ["imputed_weight", "imputed_height", "imputed_bmi"]
    log = DecisionLog(tmp_path / "decisions.jsonl")
    roles = {"SEQN": "identifier", "age": "covariate", "bp_sys": "covariate", "waist": "covariate",
             **{c: "flag" for c in flags}}
    for decision in (d.SetLens(lenses=["clinical"]), d.SetTarget(column="glucose"),
                     d.SetPurpose(purpose=purpose),
                     d.SetRoles(roles=roles, unconfirmed=[*flags, "bp_sys"]),
                     *(d.ConfirmRole(column=c, role="flag") for c in flags),
                     d.ConfirmRole(column="bp_sys", role="covariate")):
        log.append(decision, sentence=_say)
    records = log.records()
    said = {r.decision.column: r.sentence for r in records if r.decision.kind == "confirm_role"}
    rest = said[flags[0]].removeprefix(f"`{flags[0]}` was ")
    assert all(said[c] == f"`{c}` was {rest}" for c in flags)  # alike but for the column
    assert said["bp_sys"] != f"`bp_sys` was {rest}"
    doc = methods.methods_section(_source(records, purpose=purpose),
                                  {"engine": {}, "inputs": [], "decisions": {"n": len(records)},
                                   "analysis_plan": {}})
    texts = [e.text for s in doc.sections for e in s.entries if e.kind == "confirm_role"]
    together = "`imputed_weight`, `imputed_height` and `imputed_bmi` were each " + rest
    assert texts == [together, said["bp_sys"]]
    merged = next(e for s in doc.sections for e in s.entries if e.text == together)
    # Under the newest of its records: the manuscript's newest line is the sentence just grown.
    last = [r for r in records if r.decision.kind == "confirm_role"][len(flags) - 1]
    assert (merged.record_id, merged.seq) == (last.id, last.seq)
    # Silent where only one is alike: two flags confirmed, the third left out, one stays its own.
    alone = methods.methods_section(_source(records[:5] + records[7:], purpose=purpose),
                                    {"engine": {}, "inputs": [], "decisions": {"n": 6},
                                     "analysis_plan": {}})
    assert [e.text for s in alone.sections for e in s.entries if e.kind == "confirm_role"] == [
        said[flags[0]], said["bp_sys"]]


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_a_folded_confirmation_names_every_record_it_says_and_the_checklist_quotes_them(tmp_path,
                                                                                       purpose):
    """W1-X's cap, ruling R2 (its P1, option b): a methods entry that folds single confirmations
    names every folded record, in the order its columns are listed (``record_ids``), so the union
    of the entries' records is the methods route's records; one alone names none and is the
    route's sentence verbatim. The checklist's place for the folded sentence names them too. The
    folded records are the three flags' confirmations, in the order they were recorded."""
    flags = ["imputed_weight", "imputed_height", "imputed_bmi"]
    log = DecisionLog(tmp_path / "decisions.jsonl")
    roles = {"SEQN": "identifier", "age": "covariate", "bp_sys": "covariate", "waist": "covariate",
             **{c: "flag" for c in flags}}
    for decision in (d.SetLens(lenses=["clinical"]), d.SetTarget(column="glucose"),
                     d.SetPurpose(purpose=purpose),
                     d.SetRoles(roles=roles, unconfirmed=[*flags, "bp_sys"]),
                     *(d.ConfirmRole(column=c, role="flag") for c in flags),
                     d.ConfirmRole(column="bp_sys", role="covariate")):
        log.append(decision, sentence=_say)
    records = log.records()
    confirmed = [r for r in records if r.decision.kind == "confirm_role"]
    folded = [r.id for r in confirmed[:3]]
    source = _source(records, purpose=purpose)
    route = {line.record_id: line.sentence for line in source.methods.lines}
    doc = methods.methods_section(source, {"engine": {}, "inputs": [], "decisions":
                                           {"n": len(records)}, "analysis_plan": {}})
    said = [e for s in doc.sections for e in s.entries if e.source == "record"]
    [merged] = [e for e in said if e.record_ids]
    assert merged.record_ids == folded and merged.record_id in folded
    rest = route[folded[0]].removeprefix(f"`{flags[0]}` was ")
    assert merged.text == "`imputed_weight`, `imputed_height` and `imputed_bmi` were each " + rest
    # Silent on every entry that is one record's: it names none, and says the route's sentence.
    for e in said:
        if e is not merged:
            assert e.record_ids is None and e.text == route[e.record_id], e
    assert {i for e in said for i in (e.record_ids or [e.record_id])} == set(route)
    report = checklists.fill(checklists.checklist_of(purpose), methods=doc, records=records,
                             state=source.state, files={})
    places = [w for item in report.items for w in item.where if w.source == "record"]
    quoting = [w for w in places if w.quote == merged.text]
    assert quoting and all(w.record_ids == folded for w in quoting)
    assert all(w.record_ids is None and w.quote == route[w.record_id] for w in places
               if w.quote != merged.text)


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_the_checklist_file_credits_a_folded_sentence_to_every_decision_it_says(tmp_path, purpose):
    """The land-1 verifier's P2: the checklist's Markdown names the decision behind each sentence it
    quotes. The log is built so the three flags' confirmations are its records #5, #6 and #7 and
    `bp_sys`'s its #8. The methods fold the three flags' into one sentence, which the checklist
    credits to all three decisions, in the order its columns are listed (``Where.seqs``), never to
    #7 alone; a place that is one record's names its one decision, as before (`bp_sys`'s #8)."""
    flags = ["imputed_weight", "imputed_height", "imputed_bmi"]
    log = DecisionLog(tmp_path / "decisions.jsonl")
    roles = {"SEQN": "identifier", "age": "covariate", "bp_sys": "covariate", "waist": "covariate",
             **{c: "flag" for c in flags}}
    for decision in (d.SetLens(lenses=["clinical"]), d.SetTarget(column="glucose"),
                     d.SetPurpose(purpose=purpose),
                     d.SetRoles(roles=roles, unconfirmed=[*flags, "bp_sys"]),
                     *(d.ConfirmRole(column=c, role="flag") for c in flags),
                     d.ConfirmRole(column="bp_sys", role="covariate")):
        log.append(decision, sentence=_say)
    records = log.records()
    assert [(r.seq, r.decision.kind) for r in records[4:]] == [
        (5, "confirm_role"), (6, "confirm_role"), (7, "confirm_role"), (8, "confirm_role")]
    source = _source(records, purpose=purpose)
    doc = methods.methods_section(source, {"engine": {}, "inputs": [], "decisions":
                                           {"n": len(records)}, "analysis_plan": {}})
    report = checklists.fill(checklists.checklist_of(purpose), methods=doc, records=records,
                             state=source.state, files={})
    places = [w for item in report.items for w in item.where if w.source == "record"]
    folded = [w for w in places if w.record_ids]
    assert folded and all(w.seqs == [5, 6, 7] for w in folded)
    assert all(w.seqs is None for w in places if not w.record_ids)
    lines = [line for line in checklists.markdown(report, "test").splitlines()
             if line.startswith("- In the methods, decision")]
    together = f"- In the methods, decisions #5, #6 and #7 (`confirm_role`): “{folded[0].quote}”"
    assert lines.count(together) == len(folded)
    assert not [line for line in lines if line.startswith("- In the methods, decision #7 ")]
    # Silent on a place that is one record's: it names its one decision, `bp_sys`'s #8 among them.
    singles = [w for w in places if not w.record_ids]
    assert any(w.record_id == records[7].id for w in singles)
    for w in singles:
        assert f"- In the methods, decision #{w.seq} (`{w.kind}`): “{w.quote}”" in lines, w
    assert len(lines) == len(places)


# ── the tables ───────────────────────────────────────────────────────────────

EFFECTS = {
    "exposure": "fiber", "measure_label": "difference in the mean outcome", "rows": "all 3 rows",
    "appendix_title": "adjustment terms, not effect estimates",
    "families": [{"family": "linear", "label": "Linear model", "sequence": [
        {"key": "crude", "label": "Unadjusted", "adjusted_for": [], "n_rows": 3,
         "inference": {"covariance": "HC3", "scale": "difference"},
         "effects": [{"feature": "fiber", "estimate": -0.1 / 3, "ci_low": -0.0123456789012345,
                      "ci_high": 1e-17, "se": 0.1, "df": 1.0, "p": 0.0004}]},
        {"key": "model_2", "label": "Model 2 (primary)", "adjusted_for": ["age"], "n_rows": 3,
         "inference": {"covariance": "none", "refused": "Too few rows."}, "effects": None}],
        "appendix": [{"key": "crude", "label": "Unadjusted", "terms": [
            {"feature": "(intercept)", "estimate": 1234.5678, "ci_low": 1000.0, "ci_high": 1500.0,
             "p": 0.5, "why": "the model's baseline, not an effect"}]}],
        "sensitivity": [{"feature": "fiber", "methods": ["e_value"],
                         "e_value": {"point": 1.5, "limit": 1.1, "rr": 0.9}, "reading": "…"}]}],
}


def test_table_2_writes_every_number_at_full_precision_and_reads_for_people():
    t2, appendix, sensitivity = tables.table2(EFFECTS, "y")
    rows = list(csv.DictReader(io.StringIO(tables.to_csv(t2))))
    assert [r["key"] for r in rows] == ["linear/crude/fiber", "linear/model_2/-"]
    assert float(rows[0]["estimate"]) == -0.1 / 3 and float(rows[0]["ci_high"]) == 1e-17
    assert rows[1]["note"] == "Too few rows." and rows[1]["estimate"] == ""
    md = tables.markdown(t2, tables.headers_for(t2))
    assert "| Unadjusted | nothing | 3 | fiber | −0.0333 (−0.0123 to 1.00e−17) | < 0.001 |" in md
    est = tables.estimates([t2, appendix, sensitivity])
    assert est["table2|linear/crude/fiber|estimate"] == -0.1 / 3
    assert est["table2_appendix|linear/crude/(intercept)|estimate"] == 1234.5678
    assert est["table2_sensitivity|linear/fiber|e_value"] == 1.5
    assert "fiber" not in {r["term"] for r in appendix.rows}


def test_the_performance_table_labels_only_the_declared_result():
    fit = {"primary_metric": "log_loss", "metric_labels": {"log_loss": "log loss", "auc": "AUC"},
           "headline_metric": "auc", "headline_label": "AUC",
           "models": [{"family": "a", "label": "A", "cv": {
               "log_loss": {"estimate": 0.5, "ci_low": 0.4, "ci_high": 0.6, "se": 0.05},
               "auc": {"estimate": 0.7}}, "baseline": {"metric": "log_loss", "value": 0.69,
                                                       "label": "the class prior"}}],
           "selection": {"extras": {"auc": {"corrected": 0.68, "corrected_low": 0.6,
                                            "corrected_high": 0.75}}},
           "result": {"basis": "selection_corrected", "family": "a", "metric": "log_loss",
                      "estimate": 0.52, "ci_low": 0.41, "ci_high": 0.63, "sentence": "S."}}
    [t] = tables.performance(fit)
    roles = {r["key"]: r["role"] for r in t.rows}
    assert {k for k, v in roles.items() if v == tables.RESULT} == {
        "result/selection_corrected/log_loss", "result/selection_corrected/auc"}
    assert roles["a/cv/log_loss"] == tables.NOT_RESULT and t.caption == "S."
    assert [r["key"] for r in t.rows][:2] == ["a/cv/log_loss", "a/cv/auc"]  # primary first



def test_the_performance_table_says_why_a_held_out_interval_is_refused():
    """DRV-FIX (b): a held-out score whose interval is refused (``performance.TIED``: the model
    gives everyone the same score) shows its estimate with no interval, and the caption says why,
    naming the family and the score. The other way: a score with its interval adds nothing."""
    from turbotab.core.models.performance import TIED

    brier = {"estimate": 0.15, "se": 0.01, "ci_low": 0.13, "ci_high": 0.17, "level": 0.95,
             "method": "influence function", "refused": None}
    auc = {"estimate": 0.5, "se": None, "ci_low": None, "ci_high": None, "level": 0.95,
           "method": "DeLong", "refused": TIED}
    fit = {"primary_metric": "brier", "metric_labels": {"brier": "Brier score", "auc": "AUC"},
           "models": [{"family": "a", "label": "Elastic net", "holdout": {"auc": 0.5, "brier": 0.15},
                       "holdout_detail": {"intervals": {"auc": auc, "brier": brier}}}],
           "result": {"basis": "holdout", "sentence": "S."}}
    [t] = tables.performance(fit)
    row = next(r for r in t.rows if r["key"] == "a/holdout/auc")
    assert (row["estimate"], row["ci_low"], row["ci_high"], row["se"]) == (0.5, None, None, None)
    assert t.caption == ("S. Elastic net's held-out AUC: the model gives everyone the same score, "
                         "so it does not rank anyone, and no interval is reported.")
    fit["models"][0]["holdout_detail"]["intervals"]["auc"] = {
        **auc, "se": 0.05, "ci_low": 0.4, "ci_high": 0.6, "refused": None}
    assert tables.performance(fit)[0].caption == "S."


# ── the model matrix ─────────────────────────────────────────────────────────


def test_the_matrix_hashes_are_its_values_and_a_fixed_parquet():
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0], "g_b": [0.0, 1.0, 0.0]},
                         index=pd.Index([4, 7, 9], name="row_id"))
    a = matrix.record(frame)
    assert matrix.record(frame.copy()) == a
    assert matrix.record_file(matrix.canonical_parquet(frame)) == a  # the design's file, read back
    assert matrix.canonical_parquet(frame) == matrix.canonical_parquet(frame.copy())
    nudged = frame.copy()
    nudged.iloc[0, 0] = np.nextafter(1.0, 2.0)  # one unit in the last place
    assert matrix.record(nudged)["content_sha256"] != a["content_sha256"]
    other_nan = frame.copy()
    other_nan.iloc[1, 0] = -np.nan
    assert matrix.content_sha256(other_nan) == a["content_sha256"]  # every NaN is one NaN
    reordered = frame.iloc[[1, 0, 2]]
    assert matrix.content_sha256(reordered) != a["content_sha256"]  # rows in their order
    import pyarrow.parquet as pq

    back = pq.read_table(io.BytesIO(matrix.canonical_parquet(frame))).to_pandas()
    assert back["row_id"].tolist() == [4, 7, 9] and back.columns.tolist() == ["row_id", "x", "g_b"]
    odd = matrix.cacheable(pd.DataFrame([[True, "a"], [False, None]], columns=["f", "f"]))
    assert odd.columns.tolist() == ["f", "f#2"] and odd["f"].tolist() == [1.0, 0.0]


def test_the_zip_is_a_pure_function_of_its_files():
    files = {"b.txt": b"2", "a/x.json": b"{}"}
    assert bundle.zip_bytes(files) == bundle.zip_bytes(dict(reversed(list(files.items()))))
    assert bundle.read_zip(bundle.zip_bytes(files)) == files


# ── the refusal ──────────────────────────────────────────────────────────────


def test_the_gate_names_each_thing_missing_with_a_way_forward(tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.append(d.SetTarget(column="y"), sentence=_say)
    log.append(d.SetPurpose(purpose="inference"), sentence=_say)
    statuses = {"cohort": SimpleNamespace(status="fresh", key="k", error=None, missing=[]),
                "design": SimpleNamespace(status="running", key=None, error=None, missing=[]),
                "fit": SimpleNamespace(status="error", key=None, error="singular", missing=[]),
                "effects": SimpleNamespace(status="blocked", key=None, error=None,
                                           missing=["estimand"])}
    interview = [{"key": "missing", "status": "open"}, {"key": "substitution", "status": "open"}]
    source = _source(log.records(), interview=interview, statuses=statuses)
    found = gate.missing(source)
    assert [m.code for m in found] == ["unanswered_questions", "plan_open", "result_not_ready",
                                       "result_failed", "result_not_ready"]
    assert found[0].message == ("The missing-values question is not answered yet; the methods "
                                "would leave it out.")  # the substitution curve is optional
    assert found[3].message == "The fit failed, so its result cannot be reported: singular."
    assert found[4].message == ("Table 2 (the declared models) waits for the question on what you "
                                "study and its effect before it can run.")
    with pytest.raises(d.Refusal) as caught:
        gate.check(source)
    assert caught.value.code == "unanswered_questions"
    assert [e["label"] for e in caught.value.exits][:2] == [
        "Answer the missing-values question",
        "Press Fit on the analysis flowchart; it locks the plan"]
    changed = _source(log.records())
    changed.inputs = [InputFile(role="table", name="t.csv", path="/x/t.csv", bytes=1, sha256="a",
                                changed=True)]
    assert gate.missing(changed)[0].code == "input_changed"


# ── the checklists ───────────────────────────────────────────────────────────


def test_an_item_is_answered_only_where_the_bundle_answers_it(tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.append(d.SetMissing(strategy="complete_case"), sentence=_say)
    log.append(d.SetExclusions(rules=[d.ExclusionRule(column="kcal", low=500, reason="r")]),
               sentence=_say)
    source = _source(log.records())
    doc = methods.methods_section(source, {"engine": {}, "inputs": [], "decisions": {"n": 2},
                                           "analysis_plan": {}})
    report = checklists.fill("STROBE-nut", methods=doc, records=log.records(), state=source.state,
                             files={checklists.FLOW: "Figure 1."})
    by = {i.id: i for i in report.items}
    assert by["12c"].status == "answered" and by["12c"].where[0].kind == "set_missing"
    assert by["nut-9"].status == "unanswered"  # a range screen is not a misreporting method
    assert by["6a"].status == "partly answered"
    assert by["6a"].note == ("partly answered — the author must supply the sources and methods of "
                             "selection of participants, and of follow-up")
    assert by["13c"].status == "answered" and by["13c"].where[0].file == checklists.FLOW
    assert by["1a"].note == checklists.UNANSWERED and by["1a"].where == []
    assert report.unanswered == [i.id for i in report.items if i.status == "unanswered"]
    assert report.counts.items == 58 == sum((report.counts.answered,
                                             report.counts.partly_answered,
                                             report.counts.unanswered))


# ── the figures ──────────────────────────────────────────────────────────────


def test_the_figures_are_self_contained_greyscale_serif_svg():
    import re
    import xml.etree.ElementTree as ET

    cohort = {"n_final": 90, "steps": [
        {"key": "loaded", "label": "Rows in the table", "n": 100, "dropped": 0},
        {"key": "exclusion:0", "label": "`kcal` within 500–5000", "n": 90, "dropped": 10,
         "reason": "an implausible intake"}]}
    svg, caption = figures.participant_flow(cohort, purpose="prediction", n_train=72, n_holdout=18)
    assert caption.startswith("Figure 1. Participant flow, from the 100 rows of the table as read "
                              "to 72 for development by cross-validation and 18 held out.")
    root = ET.fromstring(svg)
    assert "serif" in root.get("font-family") and "var(" not in svg
    assert all(c[1:3] == c[3:5] == c[5:7] for c in re.findall(r"#[0-9a-f]{6}", svg))
    assert "kcal within 500–5000" in svg and "`" not in svg
    design = {"matrix": {"n_rows": 90, "n_cols": 2}, "lineage": {"nodes": [
        {"id": "raw:a", "column": "a", "lane": "raw", "label": "a"},
        {"id": "raw:b", "column": "b", "lane": "raw", "label": "b"},
        {"id": "adj:a", "column": "a", "lane": "adjusted", "label": "a"},
        {"id": "mx:a_1", "column": "a_1", "lane": "matrix", "label": "a_1"},
        {"id": "mx:a_2", "column": "a_2", "lane": "matrix", "label": "a_2"}], "links": [
        {"source": "raw:a", "target": "adj:a", "operation": "imputed"},
        {"source": "adj:a", "target": "mx:a_1", "operation": "one-hot"},
        {"source": "adj:a", "target": "mx:a_2", "operation": "one-hot"}]}}
    svg, caption = figures.lineage(design, exposures=["a"])
    assert "Line styles: dotted, missing values filled; short dashes, coded as indicator " \
           "columns." in caption
    assert "The 1 raw column left out of the model is drawn dashed." in caption
    assert re.findall(r'font-weight="700">([^<]+)<', svg).count("a_1") == 1


# A lineage over 40 raw columns (``models.lineage.COLLAPSE_AT``), which the design stage folds into
# one node per role in each lane: 30 nutrients, 12 characteristics and total energy, each kept as
# it is (no step runs, so no column is drawn alone).
WIDE_ROLES = {**{f"n{i:02d}": "exposure" for i in range(30)},
              **{f"c{i:02d}": "covariate" for i in range(12)}, "kcal": "energy"}


def _wide_design() -> dict[str, Any]:
    from turbotab.core.models.lineage import COLLAPSE_AT, trace

    assert len(WIDE_ROLES) > COLLAPSE_AT
    lineage = trace([], list(WIDE_ROLES), WIDE_ROLES).model_dump(mode="json")
    return {"matrix": {"n_rows": 90, "n_cols": len(WIDE_ROLES)}, "lineage": lineage}


def _texts(svg: str) -> list[str]:
    import xml.etree.ElementTree as ET

    return [t.text or "" for t in ET.fromstring(svg).iter("{http://www.w3.org/2000/svg}text")]


def test_under_predict_the_lineage_figure_names_its_groups_as_the_roles_card_does():
    """The ruling at W1-W's cap (M2): under Predict the bundle's lineage figure names its collapsed
    predictor groups in the words the server serves for the roles card (``readings.served_groups``),
    as the shell's Columns lineage does (LiveScenes.tsx ``inGroupWords``): in the raw lane by the
    count and the group's name for many, after it by the group's name and its count of columns, and
    a name that is one word for one or many with its count of columns in every lane. It never says
    "covariate columns" or "exposure columns". The words for a dietary project are written out here
    from the ruling of 2026-10-17; the counts are the frame's."""
    design = _wide_design()
    state = d.ProjectState(lens=["dietary"], target="y", purpose="prediction",
                           roles=dict(WIDE_ROLES))
    svg, caption = figures.lineage(design, groups=figures.predictor_words(state))
    said = _texts(svg)
    assert {"30 nutrients", "12 characteristics", "total energy: 1 column"} <= set(said)
    assert said.count("nutrients: 30 columns") == said.count("characteristics: 12 columns") == 2
    assert said.count("total energy: 1 column") == 3
    assert not any(w in t for t in said for w in ("exposure", "covariate", "energy column"))
    assert "covariate" not in caption and "exposure" not in caption
    # the clinical lens names the Predictor role's group by its measures
    clinical = d.ProjectState(lens=["clinical"], target="y", purpose="prediction")
    svg, _ = figures.lineage(design, groups=figures.predictor_words(clinical))
    assert "30 measures" in _texts(svg) and "measures: 30 columns" in _texts(svg)


@pytest.mark.parametrize("purpose", ["inference", None])
def test_under_estimate_the_lineage_figure_keeps_the_design_stages_words(purpose):
    """The other way: under Estimate (and with no goal) the bundle passes no groups, and the figure
    is the one drawn from the design stage's labels, byte for byte ("30 exposure columns")."""
    design = _wide_design()
    state = d.ProjectState(lens=["dietary"], target="y", purpose=purpose, roles=dict(WIDE_ROLES))
    assert figures.predictor_words(state) is None
    svg, caption = figures.lineage(design, groups=figures.predictor_words(state))
    assert (svg, caption) == figures.lineage(design)
    said = _texts(svg)
    assert {"30 exposure columns", "12 covariate columns", "1 energy column"} <= set(said)


@pytest.mark.parametrize("purpose, want", [("prediction", "30 nutrients"),
                                           ("inference", "30 exposure columns")])
def test_the_bundles_lineage_figure_is_worded_by_the_goal(tmp_path, purpose, want):
    """The bundle itself draws ``figures/lineage.svg`` with the goal's words, from the design
    stage's artifact over 40 raw columns."""
    log = DecisionLog(tmp_path / "decisions.jsonl")
    for x in (d.SetLens(lenses=["dietary"]), d.SetTarget(column="y"),
              d.SetPurpose(purpose=purpose), d.SetRoles(roles=dict(WIDE_ROLES))):
        log.append(x, sentence=_say)
    source = _source(log.records(), purpose=purpose, artifacts={"design": _wide_design()})
    files = bundle.contents(source, gated=False).files
    said = _texts(files["figures/lineage.svg"].decode("utf-8"))
    assert want in said


# ── the replay's own refusals ────────────────────────────────────────────────


def test_the_replay_refuses_a_file_that_differs_naming_both_hashes(tmp_path):
    import hashlib

    data = tmp_path / "t.csv"
    data.write_bytes(b"a,b\n1,2\n")
    prov = {"inputs": [{"role": "table", "name": "t.csv", "path": str(data),
                        "sha256": hashlib.sha256(b"a,b\n1,3\n").hexdigest()}]}
    with pytest.raises(replay.Refused) as caught:
        replay.verify_inputs(prov)
    actual = hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert caught.value.message.startswith(
        f"The table `t.csv` does not match the record: its SHA-256 is {actual}, the record's is "
        f"{prov['inputs'][0]['sha256']} ({data}).")
    report = replay.run(tmp_path / "not-a-bundle.zip")
    assert report.refused and "is not a TurboTab export bundle" in report.refused
    assert replay.close(1.0, 1.0 + 1e-13) and not replay.close(1.0, 1.0 + 1e-11)
    assert replay.close(1e6, 1e6 + 1e-7) and replay.close(None, None) and not replay.close(None, 0)
    json.dumps(report.model_dump())


PROVENANCE = {"url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.xpt",
              "release_date": "2019-11-01", "license": "Public domain (17 U.S.C. 105)"}


def test_an_input_echoes_where_it_came_from_only_when_given(tmp_path):
    """SWARM_ENGINE_PLAN E5: an input's URL, release date and license, as given at upload, are
    echoed into ``provenance.json``'s inputs; an input given none has exactly the keys it had."""
    import hashlib

    from turbotab.core.export.source import input_file

    data = tmp_path / "GLU_J.csv"
    data.write_bytes(b"seqn,lbxglu\n1,98\n2,104\n")
    sha = hashlib.sha256(data.read_bytes()).hexdigest()
    given = input_file("table", "GLU_J.csv", data, None, provenance=PROVENANCE).to_json()
    assert given == {"role": "table", "name": "GLU_J.csv", "path": str(data), "bytes": 23,
                     "sha256": sha, "file_id": None, **PROVENANCE}
    some = input_file("joined file", "GLU_J.csv", data, None, file_id="f0123456789",
                      provenance={"license": PROVENANCE["license"]}).to_json()
    assert some["license"] == PROVENANCE["license"] and "url" not in some
    plain = input_file("table", "GLU_J.csv", data, None).to_json()
    assert set(plain) == {"role", "name", "path", "bytes", "sha256", "file_id"}
    # The replay checks an input by its SHA-256 alone: the echoed fields change nothing there.
    [check] = replay.verify_inputs({"inputs": [given]})
    assert check.matches and check.sha256 == sha


def test_the_performance_table_keeps_the_fits_unweighted_label():
    """MS4 and EXPLORE (ruling 13): under prediction with survey weights the fit's own scores are
    unweighted and labeled so (its models' note); the bundle keeps the label on every row and says
    it under the table, and under the surveyed-population answer says the population's design-based
    scores are not in it. The numbers are the fit's, unchanged."""
    from turbotab.core.stages.modeling import PREDICTION_POPULATION_SCORES, UNWEIGHTED_SCORES

    def fit(note: str | None) -> dict[str, Any]:
        return {"primary_metric": "rmse", "metric_labels": {"rmse": "RMSE"},
                "models": [{"family": "a", "label": "A", "concerns": [note] if note else [],
                            "cv": {"rmse": {"estimate": 1.2, "ci_low": 1.0, "ci_high": 1.4,
                                            "se": 0.1}},
                            "calibration": {"slope": {"estimate": 0.9}}}],
                "result": {"basis": "own_score", "family": "a", "metric": "rmse",
                           "estimate": 1.2, "sentence": "S."}}

    [plain] = tables.performance(fit(None))
    assert plain.caption == "S." and not any("unweighted" in r["basis"] for r in plain.rows)
    [sample] = tables.performance(fit(UNWEIGHTED_SCORES))
    assert [r["basis"] for r in sample.rows] == [
        "cross-validated (k-fold), unweighted", "out-of-fold predictions, unweighted",
        "its own score, declared before any score was seen, unweighted"]
    assert sample.caption == ("S. Every score is unweighted: it describes how the models predict "
                              "these participants, not the population the survey weights stand "
                              "for.")
    [population] = tables.performance(fit(PREDICTION_POPULATION_SCORES))
    assert population.caption == sample.caption + (
        " The surveyed population's performance, by design-based cross-validation, is not in this "
        "table.")
    assert [r["estimate"] for r in population.rows] == [r["estimate"] for r in plain.rows]


def test_table_2_names_a_declared_scale_it_is_adjusted_for_not_the_items_scored_into_it():
    """The survey inference journey's Table 2 and methods said Model 2 was "adjusted for sex,
    education, item_01 … item_10" while its model matrix held `support_scale` and none of the
    items: the effects stage read each matrix column's raw sources, and a scale's are its items.
    The adjustment set names what is in the model: the scale, by its own name (MS8)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    from turbotab.core.methods.scales import ScaleScorer
    from turbotab.core.stages.effects import _Run, matrix_sources

    rng = np.random.default_rng(0)
    items = [f"item_{i:02d}" for i in range(1, 5)]
    X = pd.DataFrame({"age": rng.normal(50, 10, 60), "sex_M": rng.integers(0, 2, 60) * 1.0,
                      **{c: rng.integers(1, 6, 60) * 1.0 for c in items}})
    scale = {"name": "support_scale", "items": items, "reverse": ["item_02"], "low": 1, "high": 5,
             "scoring": "sum", "role": "exposure"}
    fitted = Pipeline([("score", ScaleScorer([scale])),
                       ("model", LinearRegression())]).fit(X, rng.normal(size=60))
    sources = matrix_sources(fitted, list(X.columns))
    assert sources["support_scale"][0] == set(items)  # its raw sources are the items
    spec = SimpleNamespace(inputs=[*X.columns, "support_scale"], scales=[scale],
                           predictors=["age", "sex_M", *items])
    run = _Run(spec=spec, exposures=["age"])
    assert run._raw(["age", "sex_M", "support_scale"], ["age"], sources) == \
        ["sex_M", "support_scale"]
    assert run._scored(spec.predictors) == ["age", "sex_M", "support_scale"]
    plain = _Run(spec=SimpleNamespace(inputs=list(X.columns), scales=[]), exposures=["age"])
    assert plain._raw(["age", "sex_M", "support_scale"], ["age"], sources) == \
        ["sex_M", *items]  # without a declared scale, the columns it came from
