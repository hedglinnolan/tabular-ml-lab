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
    assert found[4].message == ("Table 2 (the declared models) waits for the exposure and effect "
                                "question before it can run.")
    with pytest.raises(d.Refusal) as caught:
        gate.check(source)
    assert caught.value.code == "unanswered_questions"
    assert [e["label"] for e in caught.value.exits][:2] == [
        "Answer the missing-values question",
        "Show the estimates; the first one shown locks the plan"]
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
