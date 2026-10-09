"""The methods section of the bundle, assembled from the Record's sentences (V2 definition of done
§3.6; North star 4).

**What it holds.** The record's methods text as ``turbotab.core.provenance.methods_text`` builds it:
each decision in force, the ones superseded before anything was seen folded out, every change made
after the estimates were seen (or the held-out rows opened) kept and marked, and a sentence that
counts rows restated on the participant flow as it stands (``voice.restated_counts``; the server
gives the counts). Beside those, the
paragraphs the analysis itself wrote on these answers (the declared model sequence's, multiple
imputation's, cross-validation's), each marked with the stage that wrote it, and one sentence the
export writes: where the provenance record is and what replaying it checks
(:func:`reproducibility_sentence`).

**How it is ordered.** By the reporting guideline of the declared purpose, never by when an answer
was given: STROBE's methods items (with STROBE-nut's) under inference, TRIPOD+AI's under
prediction. Each decision kind declares the section its sentence answers (:data:`SECTION_OF`); a
kind with none is listed under "Other decisions", and a test fails until it declares one. Within a
section the record's order holds.

**What closes it.** The analysis-plan lock says "the analysis plan recorded above was declared …
before any estimate was displayed", so it comes after every section the plan holds, and the
decisions made after the estimates were seen follow it in a section of their own, in record order,
each still led by "After the estimates were seen, …". The opening of the held-out rows closes a
prediction's methods the same way. The words "prespecified" and "preregistered" are never the
export's own (MODELING_SEQUENCE §1 row 12).
"""
from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

Guideline = Literal["STROBE-nut", "TRIPOD+AI"]
EVENTS = ("lock_plan", "open_seal", "reseal")
NEVER_SAID = ("prespecified", "pre-specified", "preregistered", "pre-registered")

# (key, title, item) in the guideline's own order; a section with nothing in it is left out.
STROBE_SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("design", "Study design", "STROBE 4"),
    ("participants", "Participants", "STROBE 6"),
    ("variables", "Variables", "STROBE 7"),
    ("measurement", "Data sources and measurement", "STROBE 8"),
    ("quantitative", "Quantitative variables", "STROBE 11"),
    ("statistical", "Statistical methods", "STROBE 12"),
    ("other", "Other decisions", ""),
)
TRIPOD_SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("data", "Data", "TRIPOD+AI 5"),
    ("participants", "Participants", "TRIPOD+AI 6"),
    ("preparation", "Data preparation", "TRIPOD+AI 7"),
    ("outcome", "Outcome", "TRIPOD+AI 8"),
    ("predictors", "Predictors", "TRIPOD+AI 9"),
    ("missing", "Missing data", "TRIPOD+AI 11"),
    ("analysis", "Analytical methods", "TRIPOD+AI 12"),
    ("other", "Other decisions", ""),
)

# Each decision kind's section: (under STROBE, under TRIPOD+AI). Its sentence answers that item.
SECTION_OF: dict[str, tuple[str, str]] = {
    # what the data are and where they came from
    "set_lens": ("measurement", "data"),
    "join_files": ("measurement", "data"),
    "import_codebook": ("measurement", "predictors"),
    "set_orientation": ("measurement", "preparation"),
    "set_feature_table": ("measurement", "preparation"),
    "apply_repair": ("measurement", "preparation"),
    "dismiss_finding": ("measurement", "preparation"),
    "defer_finding": ("measurement", "preparation"),
    "set_batch": ("measurement", "preparation"),
    "set_scales": ("measurement", "predictors"),
    # the design: what a row is, and what the analysis is for
    "set_purpose": ("design", "analysis"),
    "set_design": ("design", "analysis"),
    "set_grain": ("design", "data"),
    "set_repeat_kind": ("design", "data"),
    "set_unit": ("design", "data"),
    "set_temporal": ("design", "data"),
    # who is analyzed
    "set_exclusions": ("participants", "participants"),
    # the outcome
    "set_target": ("variables", "outcome"),
    "set_task": ("variables", "outcome"),
    "set_event": ("variables", "outcome"),
    "set_outcome_order": ("variables", "outcome"),
    "set_outcome_scale": ("variables", "outcome"),
    "set_outcome_unit": ("variables", "outcome"),
    "set_follow_up": ("variables", "outcome"),
    "set_censoring": ("variables", "outcome"),
    # the other variables, and what their values mean
    "set_roles": ("variables", "predictors"),
    "confirm_role": ("variables", "predictors"),
    "confirm_reading": ("variables", "predictors"),
    "confirm_readings": ("variables", "predictors"),
    "set_column_unit": ("variables", "predictors"),
    "set_estimand": ("variables", "analysis"),
    # how quantitative variables enter
    "set_categorical": ("quantitative", "predictors"),
    "set_exposure_form": ("quantitative", "analysis"),
    "set_forms": ("quantitative", "analysis"),  # FORM's one tap (wave 2b)
    "set_levers": ("quantitative", "analysis"),  # EXPLORE's in-fold levers (wave 2b)
    # the statistical methods
    "set_aggregation": ("statistical", "preparation"),
    "set_clusters": ("statistical", "analysis"),
    "set_survey": ("statistical", "analysis"),
    "set_missing": ("statistical", "missing"),
    "set_split": ("statistical", "analysis"),
    "set_validation": ("statistical", "analysis"),
    "set_adjustment": ("statistical", "analysis"),
    "set_model_sequence": ("statistical", "analysis"),
    "set_energy_adjustment": ("statistical", "analysis"),
    "set_usual_intake": ("statistical", "analysis"),
    "set_measurement_error": ("statistical", "analysis"),
    "select_models": ("statistical", "analysis"),
    "set_multiplicity": ("statistical", "analysis"),
    "set_causal": ("statistical", "analysis"),
    "set_time_varying": ("statistical", "analysis"),
    "set_sensitivity": ("statistical", "analysis"),
    "respond_diagnostic": ("statistical", "analysis"),
    "set_substitution": ("statistical", "analysis"),
    "set_explain": ("statistical", "analysis"),
    # wave 2b: effect modification and interaction (STROBE 12b), the outcome views Explore
    # discloses (TRIPOD+AI 7), and prediction's selection, intended use and updating
    "set_modification": ("statistical", "analysis"),
    "view_outcome": ("statistical", "preparation"),
    "set_selection": ("statistical", "analysis"),
    "set_intended_use": ("statistical", "analysis"),
    "set_updating": ("statistical", "analysis"),
    "revert": ("other", "other"),
    # P0.5: "Confirm all" quotes the defaults it confirms; each default's method is its own line
    "confirm_sweep": ("other", "other"),
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class MethodsEntry(_Model):
    """One sentence or paragraph of the methods: the record's (with its decision), a paragraph the
    analysis wrote (its stage), or the export's own reproducibility sentence."""

    source: Literal["record", "analysis", "export"]
    text: str
    record_id: str | None = None
    seq: int | None = None
    kind: str | None = None  # the decision kind, or the stage that wrote the paragraph
    in_force: bool | None = None
    after_estimates: bool = False
    post_seal: bool = False


class MethodsSection(_Model):
    key: str
    title: str
    item: str
    entries: list[MethodsEntry]


class MethodsDocument(_Model):
    guideline: Guideline
    seen_from: int | None
    sections: list[MethodsSection]
    markdown: str


def guideline_of(purpose: str | None) -> Guideline:
    return "TRIPOD+AI" if purpose == "prediction" else "STROBE-nut"


def section_of(kind: str, guideline: Guideline) -> str:
    """The section a decision kind's sentence answers; "other" for a kind that declares none."""
    pair = SECTION_OF.get(kind)
    if pair is None:
        return "other"
    return pair[1] if guideline == "TRIPOD+AI" else pair[0]


def _finish(text: str) -> str:
    from turbotab.core.voice import finish

    return finish(text)


def _said(text: str) -> str:
    """A record's sentence exactly as the record holds it, with a full stop only where it has no
    terminal mark (a record from before the receipts were finished)."""
    import re

    text = str(text).strip()
    return text if re.search(r"[.?!][)\]\"'”’]*$", text) else f"{text}."


def analysis_paragraphs(source: Any) -> list[MethodsEntry]:
    """The paragraphs the analysis wrote on these answers, once each, in a fixed order: the
    declared model sequence's (inference), each family's multiple-imputation and omics-chain
    paragraphs, (prediction) what a cross-validated score and its standard error are, and
    (inference) a declared regression calibration that ran."""
    out: list[MethodsEntry] = []
    seen: set[str] = set()

    def add(stage: str, text: Any) -> None:
        if isinstance(text, str) and text.strip():
            done = _finish(text)
            if done not in seen:
                seen.add(done)
                out.append(MethodsEntry(source="analysis", kind=stage, text=done))

    effects = source.artifact("effects") if source.purpose == "inference" else None
    if isinstance(effects, dict):
        add("effects", effects.get("methods"))
    fit = source.artifact("fit")
    if isinstance(fit, dict):
        for model in fit.get("models") or []:
            missing = ((model.get("inference") or {}).get("missing") or {})
            add("fit", missing.get("sentence"))
            add("fit", model.get("methods"))
        if source.purpose == "prediction":
            add("fit", fit.get("cv_definition"))
            add("fit", fit.get("se_definition"))
    # REPAIR-RC: the declared secondary analysis's own paragraph (MODELING_SEQUENCE §6 chain 2's
    # reviewers' sentence and what it rests on). One that was blocked is said by the record's own
    # sentence, restated with the block (``voice``), so it is not said twice.
    calibration = source.artifact("calibration") if source.purpose == "inference" else None
    if isinstance(calibration, dict) and calibration.get("applies"):
        add("calibration", calibration.get("methods"))
    return out


def reproducibility_sentence(provenance: Mapping[str, Any]) -> str:
    """The export's own methods sentence: where the provenance record is and what a replay checks.
    It says what the replay does, never that a replay was run."""
    engine = provenance.get("engine") or {}
    commit = engine.get("commit")
    changes = ", with local changes" if engine.get("modified") else ""
    built = f" (source revision `{commit[:12]}`{changes})" if commit else ""
    inputs = provenance.get("inputs") or []
    if len(inputs) == 1:
        f = inputs[0]
        named = f"the input file `{f['name']}` (SHA-256 `{f['sha256'][:12]}`)"
    else:
        named = f"each of the `{len(inputs)}` input files"
    decisions = provenance.get("decisions") or {}
    plan = provenance.get("analysis_plan") or {}
    matrix = provenance.get("model_matrix")
    n = int(decisions.get("n") or 0)
    held = [f"the decision log (`{n:,}` {'record' if n == 1 else 'records'})",
            f"the SHA-256 of {named}",
            f"the analysis plan (SHA-256 `{str(plan.get('plan_sha256'))[:12]}`)"]
    if matrix:
        held.append(f"the model matrix (`{int(matrix.get('n_rows') or 0):,}` rows by "
                    f"`{int(matrix.get('n_cols') or 0):,}` columns; SHA-256 "
                    f"`{str(matrix.get('parquet_sha256'))[:12]}` as Parquet)")
    text = (
        f"The analysis was run in TurboTab {engine.get('turbotab')}{built}. The supplementary "
        f"provenance record holds {', '.join(held[:-1])} and {held[-1]}. Replaying it with "
        f"`python -m turbotab.replay` checks each input file against its hash and refuses one that "
        f"differs; otherwise it rebuilds the analysis from the input files and the decision log "
        f"alone and compares the model matrix and every reported estimate with these.")
    _never_said(text)
    return text


def _never_said(text: str) -> None:
    lowered = text.lower()
    if any(word in lowered for word in NEVER_SAID):
        raise RuntimeError("the export never calls the plan prespecified or preregistered")


def _after_title(lines: Sequence[MethodsEntry]) -> str:
    est = any(e.after_estimates for e in lines)
    seal = any(e.post_seal for e in lines)
    if est and seal:
        return "Decisions made after the estimates were seen or the held-out rows were opened"
    if seal:
        return "Decisions made after the held-out rows were opened"
    return "Decisions made after the estimates were seen"


def methods_section(source: Any, provenance: Mapping[str, Any]) -> MethodsDocument:
    """The methods section (module docstring)."""
    guideline = guideline_of(source.purpose)
    layout = TRIPOD_SECTIONS if guideline == "TRIPOD+AI" else STROBE_SECTIONS
    text = source.methods
    lines = list(getattr(text, "lines", None) or [])
    by_key: dict[str, list[MethodsEntry]] = {key: [] for key, _, _ in layout}
    closing: list[MethodsEntry] = []
    after: list[MethodsEntry] = []
    for line in lines:
        entry = MethodsEntry(source="record", text=_said(line.sentence), record_id=line.record_id,
                             seq=line.seq, kind=line.kind, in_force=line.in_force,
                             after_estimates=bool(line.after_estimates),
                             post_seal=bool(line.post_seal))
        if entry.after_estimates or entry.post_seal:
            after.append(entry)
        elif line.kind in EVENTS:
            closing.append(entry)
        else:
            by_key[section_of(line.kind, guideline)].append(entry)
    last = "analysis" if guideline == "TRIPOD+AI" else "statistical"
    by_key[last].extend(analysis_paragraphs(source))
    by_key[last].extend(closing)
    sections = [MethodsSection(key=key, title=title, item=item, entries=by_key[key])
                for key, title, item in layout if by_key[key]]
    if after:
        sections.append(MethodsSection(key="after", title=_after_title(after), item="",
                                       entries=after))
    sections.append(MethodsSection(key="reproducibility", title="Reproducibility", item="", entries=[
        MethodsEntry(source="export", kind="provenance",
                     text=reproducibility_sentence(provenance))]))
    return MethodsDocument(guideline=guideline, seen_from=getattr(text, "seen_from", None),
                           sections=sections, markdown=render(sections, source.name, guideline))


INTRO = ("Each sentence below is the decision record's own, as it stood when this bundle was "
         "exported: answers changed before anything was seen are folded out, every change made "
         "after {seen} is kept and marked, and a sentence that counts rows counts them as they "
         "stand. Sections follow {guideline}; a paragraph that is not the record's was written by "
         "the analysis on these answers.")


def render(sections: Sequence[MethodsSection], name: str, guideline: Guideline) -> str:
    """The methods as Markdown: one heading per section; consecutive record sentences as one
    paragraph, each paragraph the analysis wrote as its own, and the sentence that closes the plan
    (the lock, or the opening of the held-out rows) as its own, last."""
    order = "STROBE and STROBE-nut" if guideline == "STROBE-nut" else "TRIPOD+AI"
    seen = ("the held-out rows were opened" if guideline == "TRIPOD+AI"
            else "the estimates were seen")
    intro = INTRO.format(guideline=order, seen=seen)
    _never_said(intro)
    parts = [f"# Methods: {name}", "", f"_{intro}_", ""]
    for section in sections:
        heading = f"{section.title} ({section.item})" if section.item else section.title
        parts += [f"## {heading}", ""]
        run: list[str] = []
        for e in section.entries:
            joins = (e.source == "record" and (e.kind not in EVENTS or section.key == "after"))
            if joins:
                run.append(e.text)
                continue
            if run:
                parts += [" ".join(run), ""]
                run = []
            parts += [e.text, ""]
        if run:
            parts += [" ".join(run), ""]
    return "\n".join(parts).rstrip() + "\n"


def plain(markdown: str) -> str:
    """The methods as plain text for a manuscript: no headings' marks, no backticks."""
    out = []
    for line in markdown.splitlines():
        line = line.lstrip("#").strip() if line.startswith("#") else line
        if line.startswith("_") and line.endswith("_") and len(line) > 1:
            line = line[1:-1]
        out.append(line.replace("`", ""))
    return "\n".join(out).rstrip() + "\n"


__all__ = ["EVENTS", "MethodsDocument", "MethodsEntry", "MethodsSection", "NEVER_SAID",
           "SECTION_OF", "STROBE_SECTIONS", "TRIPOD_SECTIONS", "analysis_paragraphs",
           "guideline_of", "methods_section", "plain", "render", "reproducibility_sentence",
           "section_of"]
