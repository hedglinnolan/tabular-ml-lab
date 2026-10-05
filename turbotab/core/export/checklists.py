"""The reporting checklists, filled from the record (V2 definition of done §3.6: "auto-filled
TRIPOD+AI (prediction) or STROBE-nut (inference) checklists that list their unanswered items").

**The items** are quoted from the primary sources, never paraphrased: TRIPOD+AI's 52 items from
Collins et al. 2024 (BMJ 385:e078378, Table 2), and STROBE-nut's 24 nutrition items with the STROBE
recommendations they extend from Lachat et al. 2016 (PLoS Med 13:e1002036, Table 1). The texts live
in ``data/*.json``, built from the publishers' full texts by
``tests/acceptance/export_data/build_checklists.py``; the acceptance suite holds each one to its
source.

**How an item is answered** (:data:`STROBE_RULES`, :data:`TRIPOD_RULES`). An item is answered where
the bundle answers it: a sentence of the record (quoted, with its decision), a paragraph the
analysis wrote (quoted, with its stage), or a figure or table of the bundle (with its caption).
Each rule names the decision kinds whose sentence answers the item (with a test on the decision
where a kind's sentence answers it only sometimes: an exclusion answers STROBE-nut 9's misreporting
only when it is a Goldberg screen), and what the record cannot know, which the author still owes
(``owed``): the item is then *partly answered*, and the checklist says what remains. An item no
rule reaches, or whose rule finds nothing in this record, is listed as "unanswered — the author must
supply this". Nothing is answered by default, and the app never fills an item by guessing what a
study did (a title, a setting, an ethics approval).

**Read live** (GET …/checklist; ``bundle.live_checklist``), the checklist records nothing, so it
quotes no score a client was not shown: under prediction the performance table's caption states
the declared result, and until the fit's cross-validated scores have been shown it is quoted as
``tables.UNSEEN_RESULT`` (MODELING_SEQUENCE §4). The bundle's own checklist quotes it whole.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

DATA = Path(__file__).resolve().parent / "data"
Checklist = Literal["STROBE-nut", "TRIPOD+AI"]
Status = Literal["answered", "partly answered", "unanswered"]
UNANSWERED = "unanswered — the author must supply this"
FLOW = "figures/participant_flow.svg"
LINEAGE = "figures/lineage.svg"
TABLE2 = "results/table2.md"
MARGINAL = "results/table2_marginal.md"
SENSITIVITY = "results/table2_sensitivity.md"
PERFORMANCE = "results/performance.md"
PROVENANCE = "provenance.json"
METHODS = "methods.md"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Where(_Model):
    """Where an item is answered: a record sentence, an analysis paragraph, or a bundle file."""

    source: Literal["record", "analysis", "file"]
    file: str  # the bundle file that holds it (the methods for a sentence or paragraph)
    quote: str  # the sentence, the paragraph, or the file's caption
    record_id: str | None = None
    seq: int | None = None
    kind: str | None = None  # the decision kind, or the stage that wrote the paragraph


class ChecklistItem(_Model):
    id: str
    section: str
    topic: str
    text: str  # quoted from the source
    kind: str | None = None  # STROBE-nut's table: "STROBE" or "STROBE-nut"
    scope: str | None = None  # TRIPOD+AI's "D;E", "D" or "E"
    status: Status
    where: list[Where]
    owed: str | None = None
    note: str


class ChecklistSource(_Model):
    table: str
    pmcid: str
    doi: str
    url: str
    retrieved: str
    sha256: str
    license: str


class ChecklistCounts(_Model):
    items: int
    answered: int
    partly_answered: int
    unanswered: int


class ChecklistReport(_Model):
    """A reporting checklist, every item listed with where the bundle answers it."""

    checklist: Checklist
    title: str
    citation: str
    source: ChecklistSource
    purpose: str | None
    items: list[ChecklistItem]
    counts: ChecklistCounts
    unanswered: list[str]  # the ids of the items the author must supply
    note: str
    # What the export still waits for (``export.gate``), when the checklist is read before it is
    # ready: the live checklist (GET …/checklist) fills from what the record holds now.
    waiting: list[str] = []


@dataclass(frozen=True)
class Rule:
    """How the bundle answers one item (module docstring)."""

    kinds: tuple[str, ...] = ()
    when: Callable[[Any], bool] | None = None  # on the decision; None: every decision of the kinds
    analysis: tuple[str, ...] = ()  # stages whose methods paragraph answers it
    files: tuple[str, ...] = ()
    owed: str | Callable[[Any], str | None] | None = None  # on the state: what the author owes


# ── the decisions a rule reads only sometimes ────────────────────────────────


def _goldberg(d: Any) -> bool:
    return d.kind == "set_exclusions" and any(getattr(r, "kind", None) == "goldberg"
                                             for r in d.rules)


def _declared_sensitivity(d: Any) -> bool:
    return d.kind == "set_sensitivity" and bool(d.analyses)


def _misreporting(d: Any) -> bool:
    return (_goldberg(d) or _declared_sensitivity(d)
            or (d.kind == "set_measurement_error" and getattr(d, "method", None) not in (None, "none")))


def _categorized(d: Any) -> bool:
    return d.kind == "set_exposure_form" and d.form == "quintiles"


def _imputed_or_sensitivity(d: Any) -> bool:
    return _declared_sensitivity(d) or (d.kind == "set_missing"
                                        and d.strategy == "multiple_imputation")


def _events_owed(state: Any) -> str | None:
    if getattr(state, "task", None) in ("binary", "time_to_event") or getattr(state, "event", None):
        return "the number of participants with and without the outcome, and any follow-up time"
    return None


OUTCOME = ("set_target", "set_task", "set_event", "set_follow_up", "set_censoring",
           "set_outcome_order", "set_outcome_unit", "set_outcome_scale")
READINGS = ("confirm_readings", "confirm_reading", "confirm_role", "set_column_unit")

STROBE_RULES: dict[str, Rule] = {
    "3": Rule(kinds=("set_estimand",),
              owed="the study's objectives and hypotheses, in the author's words"),
    "4": Rule(kinds=("set_purpose", "set_grain", "set_repeat_kind", "set_unit", "set_temporal"),
              owed="the study design by its usual name (cohort, case-control or cross-sectional)"),
    "6a": Rule(kinds=("set_exclusions",),
               owed="the sources and methods of selection of participants, and of follow-up"),
    "7": Rule(kinds=(*OUTCOME, "set_estimand", "set_adjustment", "set_roles", *READINGS),
              owed="each variable's definition beyond its column, and any diagnostic criteria"),
    "nut-7.1": Rule(kinds=("set_column_unit", "import_codebook"),
                    owed="the definition of each food, food group or nutrient"),
    "nut-7.2": Rule(kinds=("set_scales",), owed="the nutritional properties of each pattern or index"),
    "8": Rule(kinds=("join_files", "import_codebook", "set_scales", "set_batch"),
              owed="each variable's source and method of assessment"),
    "nut-8.1": Rule(kinds=("set_aggregation", "set_repeat_kind", "set_usual_intake"),
                    owed=("the dietary assessment method: the instrument, portion sizes, days and "
                          "items recorded, how it was administered and how quality was assured")),
    "nut-8.3": Rule(kinds=("set_usual_intake",), owed="why these reference values apply"),
    "nut-8.6": Rule(kinds=("set_measurement_error",),
                    owed="the method's validity in a population like this one, with its coefficients"),
    "9": Rule(kinds=("set_adjustment", "set_measurement_error", "set_sensitivity", "set_causal"),
              analysis=("effects",),
              owed="any other source of bias the author addressed, and how"),
    "nut-9": Rule(kinds=("set_exclusions", "set_sensitivity", "set_measurement_error"),
                  when=_misreporting,
                  owed="any other bias in dietary assessment (changes in habit from being measured, "
                       "imputation from other sources)"),
    "11": Rule(kinds=("set_exposure_form", "set_categorical", "set_energy_adjustment"),
               owed="why each quantitative variable has the form it has"),
    "nut-11": Rule(kinds=("set_exposure_form",), when=_categorized,
                   owed="the reference category, and how nonconsumers were handled"),
    "12a": Rule(kinds=("select_models", "set_adjustment", "set_model_sequence", "set_causal",
                       "set_time_varying", "set_energy_adjustment", "set_clusters"),
                analysis=("effects",)),
    "12c": Rule(kinds=("set_missing",)),
    "12d": Rule(kinds=("set_survey", "set_follow_up", "set_censoring")),
    "12e": Rule(kinds=("set_sensitivity", "set_measurement_error", "respond_diagnostic"),
                analysis=("effects",)),
    "nut-12.1": Rule(kinds=("set_aggregation", "set_usual_intake")),
    "nut-12.2": Rule(kinds=("set_energy_adjustment", "set_usual_intake", "set_survey")),
    "nut-12.3": Rule(kinds=("set_measurement_error",),
                     when=lambda d: getattr(d, "method", None) not in (None, "none")),
    "13a": Rule(kinds=("set_exclusions", "set_missing"), files=(FLOW,),
                owed=("the numbers before the table was assembled: potentially eligible, examined "
                      "for eligibility, and completing follow-up")),
    "13b": Rule(files=(FLOW,), owed="the reasons for nonparticipation before the table was assembled"),
    "13c": Rule(files=(FLOW,)),
    "nut-13": Rule(kinds=("set_exclusions", "set_missing"), files=(FLOW,)),
    "16a": Rule(kinds=("set_adjustment",), files=(TABLE2,)),
    "16b": Rule(kinds=("set_exposure_form",), when=_categorized,
                owed="each category's boundaries"),
    "16c": Rule(files=(MARGINAL,)),
    "17": Rule(kinds=("set_sensitivity",), when=_declared_sensitivity, files=(SENSITIVITY,)),
    "nut-17": Rule(kinds=("set_sensitivity", "set_missing"), when=_imputed_or_sensitivity),
    "nut-22.2": Rule(files=(PROVENANCE,),
                     owed="the data collection tools and the data, or how they can be accessed"),
}

TRIPOD_RULES: dict[str, Rule] = {
    "4": Rule(kinds=("set_purpose",),
              owed="the study's objectives, and whether it develops or validates a model (or both)"),
    "5a": Rule(kinds=("join_files", "import_codebook"),
               owed="the sources of the data, why these data, and how representative they are"),
    "6b": Rule(kinds=("set_exclusions",),
               owed="the study's own eligibility criteria before the table was assembled"),
    "7": Rule(kinds=("apply_repair", "dismiss_finding", "defer_finding", "set_aggregation",
                     "set_orientation", "set_feature_table", "set_batch", *READINGS),
              owed="whether the checks were similar across sociodemographic groups"),
    "8a": Rule(kinds=OUTCOME,
               owed=("how and when the outcome was assessed, why it was chosen, and whether its "
                     "assessment is consistent across sociodemographic groups")),
    "9a": Rule(kinds=("set_roles",),
               owed="why these predictors, and any pre-selection of predictors before model building"),
    "9b": Rule(kinds=("set_roles", "set_categorical", "import_codebook", *READINGS),
               owed="how and when each predictor was measured"),
    "11": Rule(kinds=("set_missing",)),
    "12a": Rule(kinds=("set_purpose", "set_split", "set_survey")),
    "12b": Rule(kinds=("set_energy_adjustment", "set_exposure_form", "set_categorical", "set_scales"),
                files=(LINEAGE,)),
    "12c": Rule(kinds=("select_models", "set_split"), analysis=("fit",),
                owed="the rationale for each model family"),
    "12d": Rule(kinds=("set_clusters",)),
    "12e": Rule(analysis=("fit",), files=(PERFORMANCE,), owed="the rationale for each measure"),
    "18f": Rule(files=(PROVENANCE,),
                owed="where the bundle, and the TurboTab version it names, can be obtained"),
    "20a": Rule(files=(FLOW,), owed=_events_owed),
    "21": Rule(files=(FLOW,),
               owed="the number of participants (and outcome events) in each analysis: "
                    "development, tuning and evaluation"),
    "23a": Rule(files=(PERFORMANCE,), owed="performance in key subgroups, with intervals"),
}

RULES: dict[str, dict[str, Rule]] = {"STROBE-nut": STROBE_RULES, "TRIPOD+AI": TRIPOD_RULES}
FILES = {"STROBE-nut": "strobe_nut.json", "TRIPOD+AI": "tripod_ai.json"}


@lru_cache(maxsize=2)
def items(checklist: Checklist) -> dict[str, Any]:
    """The checklist's items as quoted from its source (``data/*.json``)."""
    return json.loads((DATA / FILES[checklist]).read_text("utf-8"))


def checklist_of(purpose: str | None) -> Checklist:
    return "TRIPOD+AI" if purpose == "prediction" else "STROBE-nut"


def fill(checklist: Checklist, *, methods: Any, records: Sequence[Any], state: Any,
         files: Mapping[str, str]) -> ChecklistReport:
    """The checklist with every item placed (module docstring). ``methods``: the bundle's
    ``MethodsDocument``; ``files``: each bundle file that can answer an item, with its caption."""
    doc = items(checklist)
    rules = RULES[checklist]
    by_id = {r.id: r for r in records}
    entries = [e for s in methods.sections for e in s.entries]
    out: list[ChecklistItem] = []
    for item in doc["items"]:
        rule = rules.get(item["id"])
        where: list[Where] = []
        if rule is not None:
            for e in entries:
                if e.source == "record" and e.kind in rule.kinds:
                    record = by_id.get(e.record_id)
                    if rule.when is not None and (record is None or not rule.when(record.decision)):
                        continue
                    where.append(Where(source="record", file=METHODS, quote=e.text,
                                       record_id=e.record_id, seq=e.seq, kind=e.kind))
                elif e.source == "analysis" and e.kind in rule.analysis:
                    where.append(Where(source="analysis", file=METHODS, quote=e.text, kind=e.kind))
            for name in rule.files:
                if name in files:
                    where.append(Where(source="file", file=name, quote=files[name]))
        owed = rule.owed(state) if rule is not None and callable(rule.owed) else (
            rule.owed if rule is not None else None)
        if not where:
            status, note, owed = "unanswered", UNANSWERED, None
        elif owed:
            status, note = "partly answered", f"partly answered — the author must supply {owed}"
        else:
            status, note = "answered", "answered in the bundle"
        out.append(ChecklistItem(id=item["id"], section=item["section"], topic=item["topic"],
                                 text=item["text"], kind=item.get("kind"), scope=item.get("scope"),
                                 status=status, where=where, owed=owed, note=note))
    counts = ChecklistCounts(
        items=len(out), answered=sum(i.status == "answered" for i in out),
        partly_answered=sum(i.status == "partly answered" for i in out),
        unanswered=sum(i.status == "unanswered" for i in out))
    return ChecklistReport(
        checklist=checklist, title=doc["title"], citation=doc["citation"],
        source=ChecklistSource(**doc["source"]), purpose=getattr(state, "purpose", None),
        items=out, counts=counts, unanswered=[i.id for i in out if i.status == "unanswered"],
        note=(f"Each item is quoted from {doc['source']['table']} of the source and listed with "
              f"where this bundle answers it: a sentence of the decision record (with its "
              f"decision), a paragraph the analysis wrote, or a figure or table. An item the "
              f"record cannot answer is listed as “{UNANSWERED}”."))


def markdown(report: ChecklistReport, name: str) -> str:
    """The checklist as Markdown: every item, its quoted text, and where it is answered."""
    c = report.counts
    lines = [f"# {report.checklist} checklist: {name}", "",
             f"{report.citation}.", "",
             report.note, "",
             f"Coverage: {c.items} items; {c.answered} answered, {c.partly_answered} partly "
             f"answered, {c.unanswered} unanswered.", ""]
    section = None
    for item in report.items:
        if item.section != section:
            section = item.section
            lines += [f"## {section}", ""]
        label = f"{item.kind} {item.id}" if item.kind == "STROBE" else item.id
        if report.checklist == "TRIPOD+AI":
            label = f"{item.id} ({item.scope})"
        lines += [f"### {label} · {item.topic}", "", f"> {item.text}", ""]
        if item.status == "unanswered":
            lines += [f"**{UNANSWERED[:1].upper()}{UNANSWERED[1:]}.**", ""]
            continue
        lines += [f"**{item.status[:1].upper()}{item.status[1:]}.**", ""]
        for w in item.where:
            if w.source == "record":
                lines.append(f"- In the methods, decision #{w.seq} (`{w.kind}`): “{w.quote}”")
            elif w.source == "analysis":
                lines.append(f"- In the methods, the analysis's paragraph (`{w.kind}`): “{w.quote}”")
            else:
                lines.append(f"- `{w.file}`: {w.quote}")
        if item.owed:
            lines += ["", f"The author must supply {item.owed}."]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["ChecklistItem", "ChecklistReport", "RULES", "Rule", "STROBE_RULES", "TRIPOD_RULES",
           "UNANSWERED", "Where", "checklist_of", "fill", "items", "markdown"]
