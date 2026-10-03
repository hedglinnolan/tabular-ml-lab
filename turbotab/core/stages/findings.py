"""The ``findings`` stage: the structural diagnosis and the lens packs, as one list.

Two legacy streams, each passed through unchanged apart from its shape:

* ``turbotab.engine.diagnose`` (``ml.import_doctor`` + ``ml.binary_text``),
  read under the lens by ``turbotab.packs.reframe``, which annotates and never
  deletes: a reframed finding keeps its id, drops in severity, and carries the
  lens's note in its detail. ``source: "structural"``.
* ``turbotab.packs.findings``: ``source: "pack"``, with the pack's evidence
  badge. Where a finding makes several claims, the badge shows the weakest
  status among them, because that is the one a reader has to act on.

The whole table is materialized under the memory budget. A table that does
not fit fails the stage with the budget's message; findings count rows, so a
sample would state wrong numbers.

M1 (M1_CONTRACT §6): every finding also carries ``summary`` (≤ 20 words),
``routes_to`` (the question that acts on it), ``lever_label`` (≤ 5 words) and
``group`` (a pager key for same-kind findings), and its text is normalized —
see :mod:`turbotab.core.stages.finding_words`, which also adds the app's own
findings (identifiers, flags, survey design, pooled cycles).

M2 (M2_CONTRACT §4): every finding carries ``repairs``, its options from the repair registry
(:mod:`turbotab.core.repairs`), and the app adds one finding of its own there: SAS transport
zeros read as ``5.4e-79``.
"""
from __future__ import annotations

from typing import Any, Iterable

from turbotab.core import repairs  # noqa: F401 - registers the repair validators, previews, key view
from turbotab.core.methods import omics  # noqa: F401 - registers the omics_scale repairs (WP11)
from turbotab.core.graph import StageContext
from turbotab.core.shape_memo import remembered
from turbotab.core.stages.data import LENSES, open_store
from turbotab.core.stages.finding_words import (
    FindingContext,
    family,
    flag_columns,
    own_findings,
    repoint_energy,
    restate_energy,
    settle_groups,
    speak,
)

SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}
_SEVERITY = {"critical": "critical", "warning": "warning", "caution": "warning", "info": "info"}
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


def _severity(value: Any) -> str:
    return _SEVERITY.get(str(value), "info")


def _columns(values: Iterable[Any] | None) -> list[str]:
    return [str(c) for c in values or []]


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def structural_finding(f: dict[str, Any]) -> dict[str, Any]:
    reframed_by = [k for k in f.get("reframed_by") or [] if k in LENSES]
    detail = str(f.get("detail") or "")
    note = _text(f.get("reframe_note"))
    if note:
        detail = f"{detail}\n\n{note}" if detail else note
    return {
        "id": str(f["id"]),
        "severity": _severity(f.get("severity")),
        "title": str(f.get("title") or ""),
        "detail": detail,
        "why_it_matters": _text(f.get("why_it_matters")),
        "affected_columns": _columns(f.get("affected_columns")),
        "source": "structural",
        "lens": reframed_by[0] if reframed_by else None,
        "evidence": None,
    }


def pack_finding(f: dict[str, Any]) -> dict[str, Any]:
    badge = f.get("evidence") or {}
    status = badge.get("weakest_status") or badge.get("evidence_status")
    evidence = (
        {"status": str(status), "source": str(badge.get("source") or "")} if status else None
    )
    lens = f.get("pack")
    return {
        "id": str(f["id"]),
        "severity": _severity(f.get("severity")),
        "title": str(f.get("title") or ""),
        "detail": str(f.get("detail") or ""),
        "why_it_matters": _text(f.get("why_it_matters")),
        "affected_columns": _columns(f.get("affected_columns")),
        "source": "pack",
        "lens": lens if lens in LENSES else None,
        "evidence": evidence,
    }


def speak_for(frame: Any, lens: list[str], target: str | None, structural: list[dict[str, Any]],
              from_packs: list[dict[str, Any]],
              units: Any = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Both legacy streams in the app's voice, plus the app's own findings, as (raw, finding).

    One column, one card: a flag column the app speaks for (``imputed_weight`` → ``weight``) is
    not also reported as "a binary variable written as true/false", and a column the binary-text
    check already reports is not reported again as "true/false stored as text".
    """
    fc = FindingContext(frame=frame, lens=tuple(lens), target=target, units=dict(units or {}))
    legacy = [(f, structural_finding(f)) for f in structural] + [(f, pack_finding(f)) for f in from_packs]
    # The pack's energy findings are about the column the app reads as energy intake (audit WP13
    # gate repair: the pack's alias matcher read a device's energy expenditure as intake).
    legacy = repoint_energy(legacy, fc)
    own = own_findings(fc, [finding for _, finding in legacy])
    flags = flag_columns(own)
    two_level = {c for _, f in legacy if family(f["id"]) == "binary_text" for c in f["affected_columns"]}
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw, finding in legacy:
        kind = family(finding["id"])
        columns = set(finding["affected_columns"])
        if kind in ("binary_text", "boolean_as_text") and columns & flags:
            continue
        if kind == "boolean_as_text" and columns and columns <= two_level:
            continue
        if kind == "pack::dietary::energy_adjustment":
            restate_energy(finding, raw, fc)
        out.append((raw, speak(finding, raw, fc)))
    out.extend(({"confidence": "high"}, finding) for finding in own)
    return out


def _lens_phrase(lens: list[str]) -> str:
    if not lens:  # "Something else, or not sure" (audit RO-11, WP18): the generic checks alone
        return "no field's lens (the generic checks only)"
    if len(lens) == 1:
        return f"the {lens[0]} lens"
    return "the " + ", ".join(lens[:-1]) + f" and {lens[-1]} lenses"


@remembered()  # the packs' shape readings once per table, not once per finding (wide data)
def findings_stage(ctx: StageContext) -> dict[str, Any]:
    lens = [k for k in ctx.state.lens or [] if k in LENSES]
    target = ctx.state.target

    with open_store(ctx) as store:
        ctx.progress(0.05, "Reading the table")
        frame = store.materialize()  # MemoryBudgetExceeded fails the stage, by design
    frame = frame.reset_index(drop=True)  # the legacy code was written for read_csv frames
    if target is not None and target not in frame.columns:
        target = None

    from turbotab import engine
    from turbotab.core import detectors

    ctx.progress(0.3, "Checking the table's structure")
    structural = [engine.shape_finding_to_dict(f) for f in engine.diagnose(frame, target)]
    # WP14 (audit IN-03, IN-04, IN-09, IN-14–IN-18): the detectors that fired on clean data are
    # read by turbotab.core.detectors in place of the legacy ones, under the same ids.
    structural = detectors.reframe(detectors.structural(structural, frame), lens, frame)
    ctx.progress(0.65, "Running the lens packs")
    units = getattr(ctx.state, "column_units", None) or {}
    codings = getattr(ctx.state, "sex_codings", None) or {}
    from_packs = detectors.pack_findings(frame, lens, units=units, codings=codings)
    ctx.progress(0.95, "Ranking the findings")

    spoken = speak_for(frame, lens, target, structural, from_packs, units=units)
    sas = repairs.sas_zero_finding(frame, target)  # M2: the app's own detector for XPT zeros
    if sas is not None:
        spoken.append(sas)
    # WP1 (audit MA-05, MA-16, MA-18): what values mean, each with its repair. A column the app's
    # own text-number finding reads is not also reported, without a lever, by the legacy checks.
    own = repairs.text_number_findings(frame)
    for found in (repairs.ambiguous_date_finding(frame), repairs.infinite_finding(frame)):
        if found is not None:
            own.append(found)
    # WP11 (audit ME-09): raw counts or intensities, read from the values alone, with the
    # normalizations a linear model needs; and the raw-counts coaching made purpose-specific.
    scale = omics.scale_finding(frame, lens, target)
    if scale is not None:
        own.append(scale)
    # WP14 (audit IN-13): the stated lens against the table, answered by changing it or recorded.
    own.extend(detectors.own_findings(frame, lens))
    for _, f in spoken:
        if family(f["id"]) == "pack::genomics::data_type":
            omics.restate_raw_counts(f)
    read = {c for _, f in own if family(f["id"]) == "text_numbers" for c in f["affected_columns"]}
    spoken = [pair for pair in spoken
              if not (family(pair[1]["id"]) in ("numeric_as_text", "text_missing")
                      and set(pair[1]["affected_columns"]) & read)]
    spoken.extend(own)
    ranked = sorted(
        spoken,
        key=lambda pair: (
            SEVERITY_RANK[pair[1]["severity"]],
            _CONFIDENCE_RANK.get(str(pair[0].get("confidence")), 1),
            pair[1]["id"],
        ),
    )
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, finding in ranked:
        base, n = finding["id"], 1
        while finding["id"] in seen:  # two streams, one id space
            n += 1
            finding["id"] = f"{base}#{n}"
        seen.add(finding["id"])
        findings.append(finding)
    settle_groups(findings)
    repairs.attach(ranked, frame, target)  # M2_CONTRACT §4: each finding's repair options

    n_rows, n_cols = frame.shape
    about = f", with {target!r} as the target" if target else ", before a target was chosen"
    basis = f"Read all {n_rows:,} rows and {n_cols:,} columns under {_lens_phrase(lens)}{about}."
    return {"findings": findings, "basis": basis}
