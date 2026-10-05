"""The manuscript bundle: one zip per project (V2 definition of done §1, "export ends every journey";
§3.6).

    turbotab-export/
      README.md                 what each file is, and how to replay the record
      methods.md · methods.txt  the methods section, from the record (``methods``); plain text too
      methods.json              the same, each sentence with its decision or the stage that wrote it
      results/*.csv · *.md      the results tables (``tables``): Table 2 and its appendix under
                                inference, the performance table with the declared result under
                                prediction
      figures/participant_flow.svg · figures/lineage.svg   journal-format figures (``figures``)
      checklist/strobe_nut.md|json  or  checklist/tripod_ai.md|json   (``checklists``)
      analysis_plan.json        the analysis plan as the record holds it, with its SHA-256
                                (``plan_lock.plan_export``, byte for byte)
      provenance.json           the provenance record (``record``)
      decisions.jsonl           the decision log exactly as stored: what a replay reads
      manifest.json             every other file's SHA-256

The bundle is a pure function of the project: no clock is read and the zip's entries are written in
a fixed order with a fixed date, so the same record and the same results give the same bytes. It
refuses (``gate``) while anything it would report is not settled.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from typing import Any

ROOT = "turbotab-export/"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


@dataclass
class ExportBundle:
    files: dict[str, bytes]
    provenance: dict[str, Any]
    checklist: Any  # checklists.ChecklistReport
    methods: Any  # methods.MethodsDocument
    tables: list[Any] = field(default_factory=list)

    def zip(self) -> bytes:
        return zip_bytes(self.files)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=1, sort_keys=False) + "\n").encode("utf-8")


def zip_bytes(files: dict[str, bytes]) -> bytes:
    """A deterministic zip of ``files`` under :data:`ROOT` (sorted, fixed date and mode)."""
    sink = io.BytesIO()
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(ROOT + name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            z.writestr(info, files[name], compresslevel=9)
    return sink.getvalue()


def read_zip(data: bytes) -> dict[str, bytes]:
    """A bundle's files by their path under :data:`ROOT`."""
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            key = name[len(ROOT):] if name.startswith(ROOT) else name
            out[key] = z.read(name)
    return out


def _stage_keys(source: Any, stages: tuple[str, ...]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for stage in stages:
        status = source.status(stage)
        out[stage] = getattr(status, "key", None) if status is not None else None
    return out


def matrix_bytes(source: Any) -> bytes | None:
    """The model matrix's canonical Parquet file, as the fresh design stage wrote it."""
    from turbotab.core.export.matrix import FILE

    design = source.bundle("design")
    path = (getattr(design, "files", None) or {}).get(FILE) if design is not None else None
    return path.read_bytes() if path is not None and path.is_file() else None


def contents(source: Any, *, gated: bool = True) -> ExportBundle:
    """Everything the bundle holds (module docstring). ``gated=False`` assembles what exists now
    without the refusal: the live checklist reads it (GET …/checklist)."""
    from turbotab.core import plan_lock
    from turbotab.core.export import checklists, figures, gate, matrix, methods, record, tables

    if gated:
        gate.check(source)
    state = source.state
    purpose = getattr(state, "purpose", None)
    raw = matrix_bytes(source)
    matrix_record = matrix.record_file(raw) if raw is not None else None
    results = tables.results_tables(source)
    estimates = tables.estimates(results)
    plan_doc = plan_lock.plan_document(source.records)
    plan_bytes = plan_lock.plan_export(source.records)
    keys = _stage_keys(source, gate.result_stages(state))
    prov = record.provenance(source, plan=plan_doc, plan_bytes=plan_bytes, matrix=matrix_record,
                             estimates=estimates, stage_keys=keys)
    doc = methods.methods_section(source, prov)
    files: dict[str, bytes] = {}
    captions: dict[str, str] = {}
    for table in results:
        files[f"results/{table.name}.csv"] = tables.to_csv(table).encode("utf-8")
        files[f"results/{table.name}.md"] = tables.markdown(
            table, tables.headers_for(table)).encode("utf-8")
        captions[f"results/{table.name}.md"] = f"{table.title}. {table.caption}"
    cohort = source.artifact("cohort")
    fit = source.artifact("fit")
    design = source.artifact("design")
    if isinstance(cohort, dict):
        fitted = fit if isinstance(fit, dict) else {}
        svg, caption = figures.participant_flow(cohort, purpose=purpose,
                                                n_train=fitted.get("n_train"),
                                                n_holdout=fitted.get("n_holdout"))
        files["figures/participant_flow.svg"] = svg.encode("utf-8")
        captions[checklists.FLOW] = caption
    if isinstance(design, dict):
        estimand = getattr(state, "estimand", None)
        exposure = ((estimand.get("exposure") if isinstance(estimand, dict)
                     else getattr(estimand, "exposure", None)) if purpose == "inference" else None)
        svg, caption = figures.lineage(design, exposures=[exposure] if exposure else [])
        files["figures/lineage.svg"] = svg.encode("utf-8")
        captions[checklists.LINEAGE] = caption
    captions[checklists.PROVENANCE] = (
        "the provenance record: the decision log, each input file's SHA-256, the engine and its "
        "stages' versions, the model matrix's hashes and every reported estimate, which "
        "`python -m turbotab.replay` checks against the input files")
    which = checklists.checklist_of(purpose)
    report = checklists.fill(which, methods=doc, records=source.records, state=state,
                             files=captions)
    stem = "tripod_ai" if which == "TRIPOD+AI" else "strobe_nut"
    files[f"checklist/{stem}.md"] = checklists.markdown(report, source.name).encode("utf-8")
    files[f"checklist/{stem}.json"] = canonical(report.model_dump(mode="json"))
    files["methods.md"] = doc.markdown.encode("utf-8")
    files["methods.txt"] = methods.plain(doc.markdown).encode("utf-8")
    files["methods.json"] = canonical(doc.model_dump(mode="json"))
    files["analysis_plan.json"] = plan_bytes
    files["provenance.json"] = canonical(prov)
    files["decisions.jsonl"] = source.decisions_jsonl
    files["README.md"] = readme(source, prov, report, files).encode("utf-8")
    files["manifest.json"] = canonical({name: record.sha256(data)
                                        for name, data in sorted(files.items())})
    return ExportBundle(files=files, provenance=prov, checklist=report, methods=doc,
                        tables=results)


def readme(source: Any, prov: dict[str, Any], report: Any, files: dict[str, bytes]) -> str:
    """What the bundle holds, in a page."""
    matrix = prov.get("model_matrix") or {}
    inputs = prov.get("inputs") or []
    c = report.counts
    stem = "tripod_ai" if report.checklist == "TRIPOD+AI" else "strobe_nut"
    result_files = sorted(n for n in files if n.startswith("results/") and n.endswith(".md"))
    lines = [
        f"# TurboTab export: {source.name}", "",
        f"Purpose: {prov['project']['purpose']}. Outcome: `{prov['project']['target']}`. "
        f"Made by TurboTab {prov['engine']['turbotab']}.", "",
        "## What is here", "",
        "- `methods.md`: the methods section, each sentence the decision record's own, in the "
        f"order of {'TRIPOD+AI' if report.checklist == 'TRIPOD+AI' else 'STROBE and STROBE-nut'} "
        "(`methods.txt` is the same as plain text; `methods.json` names each sentence's decision).",
        *[f"- `{n}` (and `{n[:-3]}.csv`, every number at full precision)" for n in result_files],
        "- `figures/participant_flow.svg` and `figures/lineage.svg`: the participant flow and the "
        "column lineage, as they would be published.",
        f"- `checklist/{stem}.md` (and `.json`): the {report.checklist} checklist, every item "
        f"quoted from its source and listed with where this bundle answers it: {c.answered} "
        f"answered, {c.partly_answered} partly answered, {c.unanswered} unanswered of {c.items}.",
        f"- `analysis_plan.json`: the analysis plan as the record holds it (its own SHA-256 inside: "
        f"`{prov['analysis_plan']['sha256']}`).",
        "- `provenance.json`: the decision log, the input files' SHA-256, the engine and the "
        "environment, the model matrix's hashes and every reported estimate.",
        "- `decisions.jsonl`: the decision log exactly as stored.",
        "- `manifest.json`: every file's SHA-256.", "",
        "## The input files", "",
        *[f"- {f['role']} `{f['name']}`: {f['bytes']:,} bytes, SHA-256 `{f['sha256']}`"
          for f in inputs], "",
        "## Replaying the record", "",
        "```",
        "python -m turbotab.replay <this bundle>.zip --data <the input file>",
        "```", "",
        "The replay checks the input file against its SHA-256 and refuses one that differs, naming "
        "both hashes. Otherwise it rebuilds the project in a fresh TurboTab home from the input "
        "file and `decisions.jsonl` alone, then compares the model matrix "
        f"(Parquet SHA-256 `{matrix.get('parquet_sha256')}`, {int(matrix.get('n_rows') or 0):,} "
        f"rows by {int(matrix.get('n_cols') or 0):,} columns) byte for byte and every number of "
        f"the results tables to within {prov['replay']['tolerance']:g} (absolute for a number "
        f"below 1, relative to its size above).", "",
    ]
    return "\n".join(lines)


__all__ = ["ExportBundle", "ROOT", "canonical", "contents", "matrix_bytes", "read_zip", "readme",
           "zip_bytes"]
