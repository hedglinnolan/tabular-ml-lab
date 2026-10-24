"""The headless replay of an export bundle (V2 definition of done §3.6: "Replaying the record
reproduces the model matrix and the estimates"; BLUEPRINT §9's M3 item 4).

    python -m turbotab.replay <bundle.zip> --data <the input file> [--file <id>=<path> …]
                              [--home <empty folder>] [--keep] [--json]
                              [--truncate [open_seal|lock_plan]]
                              [--fresh-draw MODULE:FUNCTION --recorded-seed S0 --seed S]

**First, the inputs** (:func:`verify_inputs`). Each input file the provenance record names is hashed
and compared with its recorded SHA-256; a file that differs is refused with both hashes named, and
nothing is computed (a replay on other data would compare nothing that matters). The table is the
file given by ``--data``, else the path the record names; a joined file is given by ``--file
<its id or name>=<path>``, else its recorded path. A codebook is not read again: every reading it
settled is in the decision log.

**Then the analysis, from the record alone** (:func:`run`). In a fresh TurboTab home (a new
temporary folder, or ``--home``, which must be empty), a project is made from the input files and
the bundle's ``decisions.jsonl`` written as its decision log, byte for byte: the state is a fold of
the log (BLUEPRINT §3), so every stage computes from the answers exactly as recorded, with no
answer given again and nothing validated a second time against a different moment. The stages the
bundle reports are waited for, and the bundle is assembled again from the rebuilt project by the
same code.

**Then the comparison** (:func:`compare`): the model matrix byte for byte (its canonical Parquet
SHA-256, and its content hash, which does not depend on the Parquet writer), every number of the
results tables to within ``1e-12`` (absolute below 1, relative above), the analysis plan's hashes,
each result stage's key (equal when the engine's stage versions are), and which other files of the
bundle come out identical. The report says what matched, and the command exits 0 when the matrix
and every estimate are reproduced, 1 when they are not, and 2 when the replay was refused.

**For the swarm's sweep and its decision quality** (SWARM_PLAN §6, §7 and §8), two flags rebuild
an analysis without comparing it with the record; the report then holds the rebuilt bundle's
estimates (``estimates_again``) and the rebuilt project's id, and the command exits 0 when the
rebuild finished:

* **The log truncated** (``--truncate``, :func:`truncate`): every record before the first live
  ``open_seal`` (under Estimate, ``lock_plan``), the kind read from the log's purpose unless
  named; the opening or the lock and every record after it are left out. A log with no such record
  is replayed whole, and the report says nothing was cut. Cut before the lock, the rebuild's
  bundle carries no estimate, since the engine serves none before the plan is locked; the sweep's
  own press of Fit locks it after its revisit.
* **A fresh draw** (``--fresh-draw MODULE:FUNCTION --recorded-seed S0 --seed S``,
  :class:`FreshDraw`): the generator, a function of a seed that returns the table, must write the
  recorded table's bytes at ``S0`` (the same SHA-256), or the replay is refused; the analysis is
  then rebuilt on its table at ``S``, which must be another seed. A record with joined files is
  refused (a draw replaces one table), and so is a log that holds a live ``open_seal`` unless it
  is cut before it (``--truncate open_seal``; :func:`refuse_live_opening`): the opening keeps the
  held-out scores of the draw it was recorded on as the plan's reported result, so replayed as
  written it would report the recorded draw's.
* **A spline's k on a fresh draw** (:func:`fresh_splines`): a spline whose k the server set by
  Harrell's rule records the effective sample size the rule read on the analyzed rows when it was
  declared, and the k it gave. The analyst who gave no k left it to the rule, a plan to apply the
  rule, so the replay applies it to the new draw, under inference and prediction alike, on the
  rows the engine reads it on (under prediction, a record of the rule's default k, declared
  before any size was read, is the exception below). Under inference the engine asks the form
  question again where the analyzed rows' size is not the one the record states
  (``exposure_form.rule_stale``), so the rule is read on the rows as the log ends; under
  prediction it reads the rule when the spline is declared and keeps that k whatever a later
  record does to the rows, so the rule is read on the rows as they stood at the spline's record
  (:func:`size_at_record`). Before the rebuild's result stages are waited for, once the stages
  the rule reads are computed, the rule is read again as the engine records it
  (:func:`rule_forms`) on the new draw's rows, and ruled (:func:`rule_verdict`):
  - where it reads the size the record states (to within ``exposure_form.rule_stale``'s 1e-9),
    or neither reads one, the record is replayed as recorded;
  - where it reads another size, the record is replayed at the k the rule gives on it (the
    record's or another) stating that size, its sentence (which the methods text quotes) authored
    again as the log authors one, and the report's messages give both draws' k and size
    (:func:`refilled_line`; refused where the record's own sentence is not the one the log authors
    for it now), so under inference the form stands, and under prediction the k is the one the
    engine keeps for these answers on this draw. A plan lock recorded after it holds the plan the
    records before it give with the refilled spline, and that plan's SHA-256, which its sentence
    quotes (:func:`relocked_line`; refused where its plan is not the one its records give now);
  - another size is refused where a later record changed what the size is read under
    (:func:`input_changes`): the outcome, its scale, its task (answered, else the one the target
    stage detects) and its event as the log ends against those at the spline's record. The
    record's size and its sentence's words are read under those at the record, so the size the
    rule reads under later ones is not replayed in them; the refusal names each later record
    whose step changed one, and each change from the record to the log's end. Under prediction,
    where the size is read at the record, any such change refuses the rebuild;
  - a draw on which the rule reads no size is refused, since the rule gives no k there and the
    record's are the recorded draw's, and so is a standing spline no live record gives its column
    where the rule reads another size than the spline states (with no record, none is replayed at
    the rule's k); where it reads the spline's own size, the spline is replayed as recorded;
  - under prediction, a spline whose size at its record is not read on this draw (below) is
    refused;
  - under prediction, a record of the rule's default k, declared before any size was read
    (``exposure_form.rule_words``: "no effective sample size was read"), is replayed as recorded
    on every fresh draw, its k and sentence unchanged, whatever the rule reads there and whatever
    a later record changed, none of the refusals above applying to it (the ruling of
    2026-10-23): the engine keeps the k a spline is declared with, and that declaration read no
    size, so it fits the record's k on any draw for these answers; the record is the plan's own
    history, and its size is not read again. Under inference such a record is ruled as any
    other, so another size is replayed at the rule's k, as the engine asks the form question
    again.
  A spline whose k the analyst gave is replayed as recorded, whatever the rows.
  Under inference a refusal, and a refilled record's message, says the size was read on the rows
  as the log ends, and names each later record whose own step changed that size on the new draw
  (:func:`later_changes`): the rule's size read on the state folded up to the spline's record and
  up to each later record whose step writes what those rows or their outcome are read under (an
  exclusion, the missing-values answer, a confirmed reading, an answered task or event; what a
  record writes read on the state folded up to the record before it; :func:`later_records`),
  every state over the same rows; none where the size at the spline's record is the one as the
  log ends. Where the sizes cannot be read that way (a later record writes the task or the
  outcome, or the table the rows are read from), each such later record is named as one that can
  change them. Under prediction the size at the spline's record is the first of those sizes
  (the size as the log ends where no later record writes what it is read on), and the message
  says it was read on the rows as they stood at that record, naming no later record; where the
  sizes cannot be read that way it is not read, and the refusal says the size was read as the
  log ends and names each later record as one that can change it. A form's cut points and knot
  places are not in its record: the form step learns them on the rows the fit reads, and declared
  categories' cut points are the analyst's.
* **The findings rule on a fresh draw** (:func:`findings_rule`): a record tied to a finding by its
  ``finding_id`` (``apply_repair``, ``dismiss_finding``, ``defer_finding``) is left out where its
  finding does not fire on the new bytes, with every revert of a record left out: a repair answers
  a finding, and with no finding there is nothing to repair. Each record left out is named in the
  report with why. Where its finding fires, the record is kept, and a repair's settings (its
  ``params``) are read against what the finding offers on the new bytes for the same option
  (``repairs.admits``, as the engine checks an answer):
  - settings the new finding offers, or a part of them, are replayed as recorded;
  - otherwise the repair is replayed with the new draw's own settings for its option, when exactly
    one form of the option is offered with the same choices, every setting that is text, yes or
    no, or empty (a column, a level, the batch column, the platform), equal to the record's. The
    settings that are numbers, lists or mappings are what the data decide (the features a QC-RLSC
    correction drops, its counts), so they are the new draw's, and the record's sentence is
    authored again from them. Each such record is named in the report with the settings that
    differ (``redrawn``);
  - otherwise (the option is not offered, or several forms have its choices) the replay is
    refused, naming the record.
  Only a repair that stands is read this way (the last of its finding's records, not withdrawn by a
  revert): one that does not stand changes nothing and is kept as recorded.
  Which findings fire is read from the findings stage on the new bytes, first under the log's
  answers that the findings stage and the stages before it read and that are not tied to a finding,
  then also under each record the rule keeps that writes a slot those stages read (an applied
  read-numbers repair writes ``numbers_read``, and the findings are checked on the numbers it
  reads), read again until those records no longer change; findings that have not settled within
  two readings more than the log has such records refuse the replay. Every finding with a repair that
  the new bytes raise and no record kept answers by its id is named (``unanswered``): one that
  fires only on the new bytes stays unanswered, as an analyst who never saw it left it.

This module drives the server's project service (the same code a client is served by), imported
when a replay runs.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from turbotab.core.export.bundle import read_zip
from turbotab.core.export.record import TOLERANCE
from turbotab.core.export.source import hash_file

COMPARED_FILES_SKIPPED = ("provenance.json", "manifest.json", "README.md")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class InputCheck(_Model):
    role: str
    name: str
    path: str
    recorded_sha256: str
    sha256: str | None
    matches: bool
    message: str


class MatrixCheck(_Model):
    recorded_parquet_sha256: str | None
    parquet_sha256: str | None
    identical: bool
    recorded_content_sha256: str | None
    content_sha256: str | None
    content_identical: bool
    n_rows: int | None
    n_cols: int | None
    writer: str | None = None


class EstimateCheck(_Model):
    n_recorded: int
    compared: int
    reproduced: int
    max_abs_diff: float
    max_rel_diff: float
    tolerance: float
    mismatched: list[str]
    missing: list[str]  # recorded, not recomputed
    extra: list[str]  # recomputed, not recorded


class Cut(_Model):
    """Where :func:`truncate` cut the log: before the first live record of ``kind`` (its ``seq``),
    or nowhere (``seq`` None) when the log holds none."""

    kind: str
    seq: int | None
    kept: int
    left: int


class FreshDrawCheck(_Model):
    """The fresh draw's table: the generator's bytes at the recorded seed against the record's, and
    the new draw's."""

    generator: str
    recorded_seed: int
    seed: int
    recorded_sha256: str
    regenerated_sha256: str
    sha256: str
    rows: int
    path: str


class LeftOut(_Model):
    """A record the findings rule leaves out of a fresh draw's log, and why."""

    seq: int
    kind: str
    finding_id: str | None
    why: str


class Redrawn(_Model):
    """A repair the findings rule replays with the new draw's settings for its option, and the
    settings whose values differ from the record's."""

    seq: int
    finding_id: str
    option: str
    settings: list[str]


class ReplayReport(_Model):
    bundle: str
    reproduced: bool
    refused: str | None = None
    home: str | None = None
    seconds: float | None = None
    inputs: list[InputCheck] = []
    decisions: int | None = None
    plan_same: bool | None = None
    matrix: MatrixCheck | None = None
    estimates: EstimateCheck | None = None
    stage_keys: dict[str, dict[str, str | None]] = {}
    stage_keys_same: bool | None = None
    files_same: list[str] = []
    files_different: list[str] = []
    messages: list[str] = []
    # A truncated or fresh-draw rebuild (module docstring): nothing is compared; ``finished`` says
    # the rebuild reached its result stages, and ``estimates_again`` is the rebuilt bundle's.
    project: str | None = None
    cut: Cut | None = None
    fresh: FreshDrawCheck | None = None
    left_out: list[LeftOut] = []
    redrawn: list[Redrawn] = []
    unanswered: list[str] = []
    finished: bool | None = None
    estimates_again: dict[str, float | None] | None = None


class Refused(Exception):
    """The replay refuses: an input differs from the record, or the bundle is not one."""

    def __init__(self, message: str, inputs: Sequence[InputCheck] = ()):
        super().__init__(message)
        self.message = message
        self.inputs = list(inputs)


# ── the bundle ───────────────────────────────────────────────────────────────


def load_bundle(path: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """The bundle's files and its provenance record; refuses a file that is not a bundle, or one
    whose decision log is not the one its provenance record hashed."""
    import hashlib
    import zipfile

    try:
        files = read_zip(Path(path).read_bytes())
    except (OSError, zipfile.BadZipFile) as exc:
        raise Refused(f"{path} is not a TurboTab export bundle: {exc}") from None
    if "provenance.json" not in files or "decisions.jsonl" not in files:
        raise Refused(f"{path} is not a TurboTab export bundle: it has no provenance record or no "
                      f"decision log.")
    prov = json.loads(files["provenance.json"])
    recorded = (prov.get("decisions") or {}).get("sha256")
    actual = hashlib.sha256(files["decisions.jsonl"]).hexdigest()
    if recorded != actual:
        raise Refused(f"The bundle's decision log is not the one its provenance record names: its "
                      f"SHA-256 is {actual}, the record's is {recorded}.")
    return files, prov


# ── the inputs ───────────────────────────────────────────────────────────────


def _given(entry: Mapping[str, Any], data: str | None, files: Mapping[str, str]) -> str:
    if entry.get("role") == "table":
        return data or str(entry.get("path"))
    for key in (entry.get("file_id"), entry.get("name")):
        if key and key in files:
            return files[key]
    return str(entry.get("path"))


def verify_inputs(prov: Mapping[str, Any], data: str | None = None,
                  files: Mapping[str, str] | None = None) -> list[InputCheck]:
    """Each input the record names, hashed where it is given, against its recorded SHA-256
    (module docstring); raises :class:`Refused` naming every mismatch, both hashes in full."""
    checks: list[InputCheck] = []
    for entry in prov.get("inputs") or []:
        if entry.get("role") == "codebook":
            continue  # its readings are in the decision log; the file is not read again
        path = _given(entry, data, files or {})
        recorded = str(entry.get("sha256") or "")
        name, role = str(entry.get("name")), str(entry.get("role"))
        if not Path(path).is_file():
            checks.append(InputCheck(role=role, name=name, path=path, recorded_sha256=recorded,
                                     sha256=None, matches=False,
                                     message=f"The {role} `{name}` is not at {path}."))
            continue
        sha, _, _ = hash_file(path)
        same = sha == recorded
        checks.append(InputCheck(
            role=role, name=name, path=path, recorded_sha256=recorded, sha256=sha, matches=same,
            message=(f"The {role} `{name}` matches the record (SHA-256 {sha})." if same else
                     f"The {role} `{name}` does not match the record: its SHA-256 is {sha}, the "
                     f"record's is {recorded} ({path}).")))
    bad = [c for c in checks if not c.matches]
    if bad:
        raise Refused(" ".join(c.message for c in bad) + " Nothing was replayed: give the file "
                      "the record was made from.", checks)
    return checks


# ── the comparison ───────────────────────────────────────────────────────────


def close(a: float | None, b: float | None, tolerance: float = TOLERANCE) -> bool:
    """Equal to within ``tolerance`` of the larger of 1 and the recorded value's size."""
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tolerance * max(1.0, abs(a))


def compare_estimates(recorded: Mapping[str, float | None], replayed: Mapping[str, float | None],
                      tolerance: float = TOLERANCE) -> EstimateCheck:
    keys = [k for k in recorded if k in replayed]
    mismatched = [k for k in keys if not close(recorded[k], replayed[k], tolerance)]
    diffs = [abs(float(recorded[k]) - float(replayed[k])) for k in keys
             if recorded[k] is not None and replayed[k] is not None]
    rels = [abs(float(recorded[k]) - float(replayed[k])) / max(1.0, abs(float(recorded[k])))
            for k in keys if recorded[k] is not None and replayed[k] is not None]
    return EstimateCheck(
        n_recorded=len(recorded), compared=len(keys), reproduced=len(keys) - len(mismatched),
        max_abs_diff=max(diffs, default=0.0), max_rel_diff=max(rels, default=0.0),
        tolerance=tolerance, mismatched=mismatched,
        missing=[k for k in recorded if k not in replayed],
        extra=[k for k in replayed if k not in recorded])


def compare(prov: Mapping[str, Any], again: Mapping[str, Any], files: Mapping[str, bytes],
            files_again: Mapping[str, bytes]) -> dict[str, Any]:
    """The recorded bundle against the replayed one (module docstring)."""
    m0, m1 = prov.get("model_matrix") or {}, again.get("model_matrix") or {}
    matrix = MatrixCheck(
        recorded_parquet_sha256=m0.get("parquet_sha256"), parquet_sha256=m1.get("parquet_sha256"),
        identical=bool(m0) and m0.get("parquet_sha256") == m1.get("parquet_sha256"),
        recorded_content_sha256=m0.get("content_sha256"), content_sha256=m1.get("content_sha256"),
        content_identical=bool(m0) and m0.get("content_sha256") == m1.get("content_sha256"),
        n_rows=m1.get("n_rows"), n_cols=m1.get("n_cols"),
        writer=(m1.get("parquet") or {}).get("writer"))
    estimates = compare_estimates(prov.get("estimates") or {}, again.get("estimates") or {},
                                  float((prov.get("replay") or {}).get("tolerance") or TOLERANCE))
    k0, k1 = prov.get("stages") or {}, again.get("stages") or {}
    keys = {s: {"recorded": k0.get(s), "replayed": k1.get(s)} for s in sorted(set(k0) | set(k1))}
    p0, p1 = prov.get("analysis_plan") or {}, again.get("analysis_plan") or {}
    names = sorted(n for n in set(files) | set(files_again) if n not in COMPARED_FILES_SKIPPED)
    return {
        "matrix": matrix, "estimates": estimates, "stage_keys": keys,
        "stage_keys_same": all(v["recorded"] == v["replayed"] for v in keys.values()),
        "plan_same": (p0.get("sha256"), p0.get("plan_sha256")) == (p1.get("sha256"),
                                                                    p1.get("plan_sha256")),
        "files_same": [n for n in names if files.get(n) == files_again.get(n)],
        "files_different": [n for n in names if files.get(n) != files_again.get(n)],
    }


# ── the log: its lines, truncation, the findings rule ───────────────────────

# The record a truncation cuts before, by the log's purpose (SWARM_PLAN §6, the sweep's refits).
CUT_KINDS = {"prediction": "open_seal", "inference": "lock_plan"}
# The records tied to a finding by their ``finding_id`` (the findings rule).
FINDING_KINDS = ("apply_repair", "dismiss_finding", "defer_finding")


def log_lines(log: bytes) -> list[tuple[Any, bytes]]:
    """Each record of a decision log with its line's bytes, in the log's order (a blank line is
    skipped, as the log's reader skips it)."""
    from turbotab.core.decisions import read_record

    out = []
    for number, line in enumerate(log.split(b"\n"), start=1):
        if not line.strip():
            continue
        out.append((read_record(json.loads(line), where=f"decisions.jsonl:{number}"), line))
    return out


def join_lines(lines: Sequence[tuple[Any, bytes]]) -> bytes:
    return b"".join(line + b"\n" for _, line in lines)


def _reverts_of(lines: Sequence[tuple[Any, bytes]], gone: set[str]) -> set[str]:
    """``gone`` with every revert of a record in it, and every revert of those, and so on."""
    gone = set(gone)
    while True:
        more = {r.id for r, _ in lines if r.decision.kind == "revert"
                and r.decision.decision_id in gone and r.id not in gone}
        if not more:
            return gone
        gone |= more


def live_record(lines: Sequence[tuple[Any, bytes]], kind: str) -> Any:
    """The first record of ``kind`` in the log's order that no revert withdrew, or None."""
    from turbotab.core.decisions import reverted

    records = [r for r, _ in lines]
    withdrawn = reverted(records)
    return next((r for r in sorted(records, key=lambda r: r.seq)
                 if r.decision.kind == kind and r.id not in withdrawn), None)


def truncate(log: bytes, before: str | None = None) -> tuple[bytes, Cut]:
    """The log up to its first live record of kind ``before`` (``open_seal`` or ``lock_plan``;
    by default the one :data:`CUT_KINDS` names for the log's purpose), that record and every
    later one left out; the lines kept are the log's own bytes. A record a revert withdrew is not
    live: a lock withdrawn before any estimate was served under it is passed over for the lock that
    holds. A log with no live record of the kind is returned whole."""
    from turbotab.core.decisions import fold

    lines = log_lines(log)
    records = [r for r, _ in lines]
    kind = before or CUT_KINDS.get(str(fold(records).purpose), "open_seal")
    if kind not in CUT_KINDS.values():
        raise Refused(f"A log is cut before {' or '.join(CUT_KINDS.values())}, not {kind!r}.")
    cut = live_record(lines, kind)
    if cut is None:
        return join_lines(lines), Cut(kind=kind, seq=None, kept=len(lines), left=0)
    kept = [(r, line) for r, line in lines if r.seq < cut.seq]
    return join_lines(kept), Cut(kind=kind, seq=cut.seq, kept=len(kept),
                                 left=len(lines) - len(kept))


@dataclasses.dataclass
class Ruled:
    """The findings rule's reading of a log against the findings a draw raises
    (:func:`findings_rule`): the lines kept (a redrawn repair's line rewritten), the records left
    out and the repairs redrawn, each with why, and the repairs whose settings the draw's finding
    does not offer in any one form, which refuse the replay (``unmatched``)."""

    kept: list[tuple[Any, bytes]]
    left_out: list[LeftOut]
    redrawn: list[Redrawn]
    unmatched: list[LeftOut]


def _choices(params: Mapping[str, Any]) -> dict[str, Any]:
    """The settings of a repair that its answer names rather than the data decide: each one that
    is text, yes or no, or empty (module docstring)."""
    return {k: v for k, v in params.items() if v is None or isinstance(v, (str, bool))}


def _offered(finding: Mapping[str, Any], option: str) -> list[tuple[dict[str, Any], dict]]:
    """Each form of ``option`` the finding offers: its settings, and the offer itself."""
    return [(dict(((o.get("decision") or {}).get("params")) or {}), dict(o))
            for o in finding.get("repairs") or [] if o.get("key") == option]


def _authored(record: Any, decision: Any, before: Sequence[Any], facts: Any) -> str | None:
    """``decision``'s sentence as the log authors one for ``record`` (``voice.sentence_for`` on the
    state before it, then led by what had been seen when it was made; ``DecisionLog.append``)."""
    from turbotab.core import voice
    from turbotab.core.decisions import decided_early, disclose, fold

    text = voice.sentence_for(decision, fold(list(before)), facts)
    if isinstance(text, str) and record.early:
        text = decided_early(text, record.early)
    text = disclose(text, post_seal=record.post_seal, after_estimates=record.after_estimates)
    return text if text and text.strip() else None


def _redrawn_line(record: Any, params: Mapping[str, Any], finding: Mapping[str, Any],
                  offer: Mapping[str, Any], before: Sequence[Any]) -> bytes:
    """``record`` with the new draw's settings, its sentence authored again as the log authors one
    (:func:`_authored`), every other field as recorded."""
    from turbotab.core.decisions import record_line

    decision = record.decision.model_copy(update={"params": dict(params)})
    text = _authored(record, decision, before, {"finding": finding, "repair": offer})
    again = record.model_copy(update={"decision": decision, "sentence": text})
    return record_line(again).rstrip(b"\n")


def findings_rule(lines: Sequence[tuple[Any, bytes]], raised: Sequence[Mapping[str, Any]]) -> Ruled:
    """The fresh draw's log by the findings rule (module docstring), against ``raised``, the
    findings the new bytes raise (each with its offered ``repairs``): each record of
    :data:`FINDING_KINDS` whose finding is not raised left out, with every revert of a record left
    out; each standing repair kept as recorded where the finding offers its settings, redrawn where
    exactly one form of its option has its choices, and otherwise unmatched; every other line as
    recorded."""
    from turbotab.core.decisions import read_record, reverted
    from turbotab.core.repairs import admits

    by_id = {str(f.get("id")): f for f in raised}
    why: dict[str, LeftOut] = {}
    for r, _ in lines:
        fid = getattr(r.decision, "finding_id", None)
        if r.decision.kind in FINDING_KINDS and fid not in by_id:
            why[r.id] = LeftOut(seq=r.seq, kind=r.decision.kind, finding_id=fid,
                                why=f"its finding `{fid}` does not fire on this draw")
    for r, _ in lines:
        if r.id in _reverts_of(lines, set(why)) and r.id not in why:
            why[r.id] = LeftOut(seq=r.seq, kind=r.decision.kind, finding_id=None,
                                why=f"it reverts a record left out (`{r.decision.decision_id}`)")
    kept = [(r, line) for r, line in lines if r.id not in why]
    withdrawn = reverted([r for r, _ in kept])
    standing: dict[str, Any] = {}  # each finding's last record not withdrawn: the one that stands
    for r, _ in kept:
        if r.decision.kind in FINDING_KINDS and r.id not in withdrawn:
            standing[str(r.decision.finding_id)] = r
    redrawn: list[Redrawn] = []
    unmatched: list[LeftOut] = []
    out: list[tuple[Any, bytes]] = []
    for i, (r, line) in enumerate(kept):
        fid = str(getattr(r.decision, "finding_id", ""))
        if r.decision.kind != "apply_repair" or standing.get(fid) is not r:
            out.append((r, line))
            continue
        given = dict(r.decision.params or {})
        forms = _offered(by_id[fid], r.decision.option)
        if any(admits(params, given) for params, _ in forms):
            out.append((r, line))
            continue
        same = [(params, offer) for params, offer in forms if _choices(params) == _choices(given)]
        if len(same) != 1:
            unmatched.append(LeftOut(
                seq=r.seq, kind=r.decision.kind, finding_id=fid,
                why=(f"its finding `{fid}` does not offer `{r.decision.option}` on this draw"
                     if not forms else
                     f"its finding `{fid}` offers `{r.decision.option}` on this draw only with "
                     f"other choices than the record's" if not same else
                     f"its finding `{fid}` offers {len(same)} forms of `{r.decision.option}` on "
                     f"this draw with the record's choices, none with its settings")))
            out.append((r, line))
            continue
        params, offer = same[0]
        changed = sorted(k for k in set(params) | set(given)
                         if k not in params or k not in given or params[k] != given[k])
        redrawn.append(Redrawn(seq=r.seq, finding_id=fid, option=str(r.decision.option),
                               settings=changed))
        line = _redrawn_line(r, params, by_id[fid], offer, [x for x, _ in kept[:i]])
        out.append((read_record(json.loads(line), where=f"record {r.seq} redrawn"), line))
    return Ruled(kept=out, left_out=sorted(why.values(), key=lambda o: o.seq), redrawn=redrawn,
                 unmatched=unmatched)


def findings_inputs(lines: Sequence[tuple[Any, bytes]], reads: set[str]
                    ) -> list[tuple[Any, bytes]]:
    """The lines whose records write a slot in ``reads`` (the slots the findings stage and the
    stages before it read), with each revert of one of them: what the findings on the new bytes are
    computed under. A record tied to a finding is among them when it writes such a slot (an
    applied read-numbers repair writes ``numbers_read``); the caller passes the lines the findings
    rule keeps (:func:`fresh_findings`). A record whose slots cannot be read is kept, since leaving
    out a slot the findings read would change them."""
    keep: set[str] = set()
    for r, _ in lines:
        if r.decision.kind != "revert" and _writes(r.decision, reads) & reads:
            keep.add(r.id)
    for r, _ in lines:
        if r.decision.kind == "revert" and r.decision.decision_id in keep:
            keep.add(r.id)
    return [(r, line) for r, line in lines if r.id in keep]


def _writes(decision: Any, reads: set[str]) -> set[str]:
    """The slots ``decision`` writes onto an empty state (``reads`` when they cannot be read)."""
    from turbotab.core.decisions import ProjectState, fold_onto

    empty = ProjectState()
    try:
        after = fold_onto(empty, decision)
    except Exception:  # noqa: BLE001 - kept when its slots cannot be read (findings_inputs)
        return set(reads)
    return {f for f in ProjectState.model_fields if getattr(after, f) != getattr(empty, f)}


def fresh_findings(service: Any, pid: str, written: Path, lines: Sequence[tuple[Any, bytes]],
                   timeout: float) -> tuple[Ruled, list[str]]:
    """The findings rule on the project ``pid`` (made from a fresh draw; its log at ``written``),
    as the module docstring says: the findings read under the probe, the rule applied, the probe
    taken again from the lines the rule keeps, until it no longer changes. Returns the rule's
    reading and the findings with a repair that no kept record answers; raises :class:`Refused`
    for findings that do not settle and for a repair whose settings are unmatched."""
    from turbotab.core.repairs import findings_of

    graph = service.engine.graph
    reads = {s for name in graph.upstream("findings") | {"findings"} for s in graph[name].reads}
    probe = findings_inputs([x for x in lines if x[0].decision.kind not in FINDING_KINDS], reads)
    rounds = 2 + sum(1 for r, _ in lines
                     if r.decision.kind in FINDING_KINDS and _writes(r.decision, reads) & reads)
    for _ in range(rounds):
        written.write_bytes(join_lines(probe))
        service.engine.on_decision(pid)
        _wait(service, pid, ("findings",), timeout)
        raised = findings_of(service.engine.get(pid, "findings").artifact)
        ruled = findings_rule(lines, raised)
        again = findings_inputs(ruled.kept, reads)
        if join_lines(again) != join_lines(probe):
            probe = again
            continue
        if ruled.unmatched:
            raise Refused(" ".join(f"Record {o.seq} ({o.kind}) cannot be replayed on this draw: "
                                   f"{o.why}." for o in ruled.unmatched)
                          + " Nothing was replayed: a repair is replayed only with settings its "
                            "finding offers on the new bytes.")
        answered = {getattr(r.decision, "finding_id", None) for r, _ in ruled.kept}
        return ruled, sorted(str(f.get("id")) for f in raised
                             if f.get("repairs") and f.get("id") not in answered)
    raise Refused(f"The findings on this draw did not settle under the record's repairs within "
                  f"{rounds} readings: nothing was replayed.")


# ── a fresh draw ─────────────────────────────────────────────────────────────


def refuse_live_opening(log: bytes) -> None:
    """Raises :class:`Refused` when ``log`` holds a live ``open_seal`` (:func:`live_record`): the
    opening keeps the held-out scores of the draw it was recorded on as the plan's reported
    result, so a fresh draw that replays it as written would report the recorded draw's."""
    from turbotab.core.voice import table_number

    opening = live_record(log_lines(log), "open_seal")
    if opening is None:
        return
    d = opening.decision
    score = ((d.scores or {}).get(d.family or "") or {}).get(d.metric or "")
    said = (f" (`{d.family}`'s held-out `{d.metric}` {table_number(score)}"
            + (f" on {d.n_holdout:,} rows" if d.n_holdout else "") + ")"
            if score is not None else "")
    raise Refused(f"Record {opening.seq} (open_seal) holds the recorded draw's held-out scores as "
                  f"the plan's reported result{said}; a fresh draw would replay it as written, and "
                  f"the rebuilt result would be the recorded draw's, not this draw's. Replayed with "
                  f"--truncate open_seal (cut=\"open_seal\"), the log is cut before it and the plan "
                  f"rebuilt on the new draw to its fit.")


def _size(n: float) -> str:
    """An effective sample size as the rule's sentence states one (``exposure_form.rule_words``)."""
    return f"{n:,.0f}" if abs(n - round(n)) < 1e-9 else f"{n:,.1f}"


# The stages a spline's rule reads its rows through, waited for before it is read on a fresh draw:
# the form card (and the stages before it), and the split, whose rows the rule reads under
# prediction (``exposure_form._n_effective``; ``ProjectService.analyzed_rows``).
RULE_STAGES = ("forms", "split")


@dataclasses.dataclass(frozen=True)
class RuleRead:
    """A standing spline whose k the server set by Harrell's rule, read again on a fresh draw
    (:func:`rule_forms`): the record that gives the column its form (its ``seq`` and ``kind``;
    None when no live record does), the k and the effective sample size it states, the rule's k
    and size on this draw's rows (``read`` None where the rule reads none), and the later records
    to name beside that size, as (seq, kind): where ``inputs`` holds what a later record changed
    of what the size is read under (:func:`input_changes`, each change from the record to the
    log's end), the records whose step changed it; else (:func:`later_changes`) ``traced``, those
    whose own step changed the size the rule reads on this draw, or each later record that writes
    what those rows are read under (:func:`later_records`), the size not read step by step. The
    rows are those as the log ends, or, where ``at_record``, those as they stood at the record
    (:func:`size_at_record`), where the engine reads the rule under prediction (``purpose``, the
    log's), no later record named."""

    column: str
    seq: int | None
    kind: str | None
    knots: int | None
    said: float | None
    k: int | None
    read: float | None
    later: tuple[tuple[int, str], ...] = ()
    traced: bool = True
    inputs: tuple[str, ...] = ()
    purpose: str | None = None
    at_record: bool = False


@dataclasses.dataclass(frozen=True)
class _Step:
    """A later record (:func:`later_records`): the record, the decision whose writes count (a
    revert's, the one it withdraws), the slots its step wrote, and the state folded up to it."""

    record: Any
    decision: Any
    written: frozenset[str]
    state: Any


def later_records(records: Sequence[Any], seq: int, reads: set[str]) -> tuple[tuple[int, str], ...]:
    """The live records after ``seq`` whose step writes a slot in ``reads`` (what the rule's size
    is read on: the rows, their outcome and its task and event; :func:`rule_forms`), each read on
    the state folded up to the record before it (an answer that stands only while its column is
    the outcome writes nothing for another column), as (seq, kind): a revert by the record it
    withdraws, when that one was recorded before ``seq`` (one withdrawing a later record leaves
    the rows as they were)."""
    return tuple((s.record.seq, s.record.decision.kind) for s in _later(records, seq, reads))


def _later(records: Sequence[Any], seq: int, reads: set[str]) -> list[_Step]:
    """:func:`later_records` as steps (:class:`_Step`): each live record after ``seq`` folded
    onto the records up to it, the slots whose value its step changed its writes."""
    from turbotab.core.decisions import ProjectState, fold, reverted

    withdrawn = reverted(records)
    by_id = {r.id: r for r in records}
    ordered = sorted(records, key=lambda r: r.seq)
    folded = [r for r in ordered if r.seq <= seq]
    state = fold(folded)
    out: list[_Step] = []
    for r in ordered:
        if r.seq <= seq or r.id in withdrawn:
            continue
        decision = r.decision
        if decision.kind == "revert":
            target = by_id.get(decision.decision_id)
            if target is None or target.seq > seq:
                continue
            decision = target.decision
        folded.append(r)
        after = fold(folded)
        written = frozenset(f for f in ProjectState.model_fields
                            if getattr(after, f) != getattr(state, f))
        state = after
        if written & reads:
            out.append(_Step(record=r, decision=decision, written=written, state=after))
    return out


# What Harrell's rule's size is read under besides the rows (``exposure_form.effective_n``), each
# with its words and whether its values are data (ticked): the outcome (a log scale makes
# ``ln_<column>`` the outcome), the outcome's scale, its task and its event.
RULE_INPUTS = (("target", "the outcome", True), ("outcome_scale", "the outcome's scale", False),
               ("task", "the task", False), ("event", "the event", True))


def _inputs(state: Any, detected: tuple[str | None, str | None]) -> dict[str, Any]:
    """:data:`RULE_INPUTS` on ``state``: the task answered, else the one detected (``detected``,
    the target stage's outcome and the task it detects there) where the outcome is that one."""
    column, task = detected
    scale = getattr(state, "outcome_scale", None)
    return {"target": state.target, "outcome_scale": scale.scale if scale is not None else None,
            "task": state.task or (task if state.target is not None and state.target == column
                                   else None),
            "event": state.event}


def input_changes(records: Sequence[Any], seq: int, reads: set[str],
                  detected: tuple[str | None, str | None]
                  ) -> tuple[tuple[tuple[int, str], ...], tuple[str, ...]]:
    """What the rule's size is read under (:data:`RULE_INPUTS`) as the log ends against what it was
    at the spline recorded at ``seq``, on the state folded up to it: ``((), ())`` where they are
    the same; else each later record (:func:`later_records`) whose step changed one, as (seq,
    kind), and each that differs, said from the record to the log's end (``the task from binary
    to regression``; where the outcome differs, its task and event, read on another outcome, are
    not said apart)."""
    from turbotab.core.decisions import fold

    steps = _later(records, seq, set(reads) | {k for k, _, _ in RULE_INPUTS})
    first = _inputs(fold([r for r in records if r.seq <= seq]), detected)
    last = _inputs(steps[-1].state, detected) if steps else first
    if first == last:
        return (), ()
    named: list[tuple[int, str]] = []
    before = first
    for s in steps:
        now = _inputs(s.state, detected)
        if now != before:
            named.append((s.record.seq, s.record.decision.kind))
        before = now

    def shown(value: Any, ticked: bool) -> str:
        return "none" if value is None else f"`{value}`" if ticked else str(value)

    moved = first["target"] != last["target"]
    words = tuple(f"{label} from {shown(first[k], ticked)} to {shown(last[k], ticked)}"
                  for k, label, ticked in RULE_INPUTS
                  if first[k] != last[k] and not (moved and k in ("task", "event")))
    return tuple(named), words


def _same_size(a: float | None, b: float | None) -> bool:
    """Two effective sample sizes alike, as the engine compares a form's with the form card's
    (``exposure_form.rule_stale``); None alike only with None."""
    from turbotab.core.methods import exposure_form as ef

    if a is None or b is None:
        return a is None and b is None
    return not ef.rule_stale({"form": "spline", "knots_rule": "harrell", "n_effective": a},
                             {"n_effective": b})


def _ingest_of(store: Any) -> dict[str, Any]:
    """The table's columns as a cohort flow reads them (``row_previews``'s)."""
    return {"columns": [c.to_dict() for c in store.info().columns]}


def _rewrites(decision: Any) -> set[str] | None:
    """The columns of the working table a decision that writes what it is built from rewrites: a
    confirmed reading's own columns (the working table reads each column as its reading says);
    None (any column, or the rows) for every other."""
    if decision.kind == "confirm_reading":
        return {decision.column}
    if decision.kind == "confirm_readings":
        return {item.column for item in decision.items}
    return None


def _flow_columns(state: Any, ingest: Mapping[str, Any]) -> set[str]:
    """The columns ``state``'s cohort flow reads (``stages.rows.cohort_inputs``): the outcome and
    the rules' columns, and under complete cases every predictor."""
    from turbotab.core.decisions import missing_strategy
    from turbotab.core.stages.rows import cohort_inputs

    needed, preds, _ = cohort_inputs(state, ingest)
    return {*needed, *(preds if missing_strategy(state) == "complete_case" else [])}


def rule_sizes(store: Any, states: Sequence[Any], task: str | None,
               rows: Any | None = None) -> list[float | None]:
    """Harrell's rule's effective sample size on each of ``states`` (``exposure_form.effective_n``),
    read on the rows its cohort keeps, every state's flow over the same rows (``rows``; None:
    every row of ``store``; ``stages.rows.cohort_flows``, as the row previews compare flows), from
    its outcome and event, by its task, else ``task`` (the one the engine reads for that outcome).
    None where it reads none."""
    from turbotab.core.methods.exposure_form import effective_n
    from turbotab.core.stages.rows import cohort_flows

    frame, _, flows = cohort_flows(store, states, _ingest_of(store), rows)
    out: list[float | None] = []
    for state, (_, kept) in zip(states, flows):
        told = state.task or task
        if not state.target or state.target not in frame.columns or told is None:
            out.append(None)
            continue
        out.append(effective_n(told, frame.loc[kept, state.target], state.event))
    return out


def later_changes(records: Sequence[Any], seq: int, reads: set[str], store: Any,
                  task: str | None, read: float | None, *, table: set[str] = frozenset(),
                  sealed: Any | None = None) -> tuple[tuple[tuple[int, str], ...], bool]:
    """The later records to name beside the size the rule reads on this draw as the log ends
    (``read``, the engine's), for the spline recorded at ``seq``, and whether they were traced:
    the rule's size (:func:`rule_sizes`) on the state folded up to ``seq`` and then on the state
    folded up to each later record in turn (:func:`later_records`), each read over every row of
    this draw, else over every row not held out (``sealed``), whichever gives ``read`` as the log
    ends. Where the first and last sizes agree, none is named; else each whose own step changed
    the size. Not traced (each later record named, as one that can change the size it reads: each
    writes what the size is read on, as a task answer does, though it changes no row) where
    neither gives ``read``, where a later record writes the task or the outcome (a state with no
    answered task is read by the task as the log ends, ``task``), or where one writes a slot in
    ``table`` (what the working table is built from, read as the log ends) other than a confirmed
    reading of columns no state's flow reads (:func:`_rewrites`)."""
    later, sizes = _sizes_along(records, seq, reads, store, task, read, table=table,
                                sealed=sealed)
    if not later:
        return (), True
    if sizes is None:
        return later, False
    if _same_size(sizes[0], sizes[-1]):
        return (), True
    return tuple(x for x, a, b in zip(later, sizes, sizes[1:]) if not _same_size(a, b)), True


def size_at_record(records: Sequence[Any], seq: int, reads: set[str], store: Any,
                   task: str | None, read: float | None, *, table: set[str] = frozenset(),
                   sealed: Any | None = None) -> tuple[float | None, bool]:
    """The rule's size on this draw on the rows as they stood at the spline recorded at ``seq``,
    where the engine reads it under prediction (``exposure_form.complete_spec`` when the spline
    is declared), and whether it was read: ``read`` (the engine's, as the log ends) where no later
    record writes what it is read on (:func:`later_records`); else the size on the state folded
    up to ``seq`` over the rows whose size as the log ends is ``read``, as :func:`later_changes`
    reads the sizes step by step; ``(None, False)`` where they are not traced that way."""
    later, sizes = _sizes_along(records, seq, reads, store, task, read, table=table,
                                sealed=sealed)
    if not later:
        return read, True
    return (sizes[0], True) if sizes is not None else (None, False)


def _sizes_along(records: Sequence[Any], seq: int, reads: set[str], store: Any,
                 task: str | None, read: float | None, *, table: set[str] = frozenset(),
                 sealed: Any | None = None
                 ) -> tuple[tuple[tuple[int, str], ...], list[float | None] | None]:
    """The later records (:func:`later_records`) and the rule's size on the state folded up to
    ``seq`` and then up to each in turn, over the rows whose last size is ``read``
    (:func:`later_changes`); None for the sizes where they are not traced, ``[read]`` where no
    later record writes what the size is read on."""
    import numpy as np

    from turbotab.core.decisions import fold

    steps = _later(records, seq, reads)
    later = tuple((s.record.seq, s.record.decision.kind) for s in steps)
    if not later:
        return (), [read]
    rewritten: set[str] = set()
    for s in steps:
        if s.written & {"task", "target"}:
            return later, None
        if s.written & set(table):
            columns = _rewrites(s.decision)
            if columns is None:
                return later, None
            rewritten |= columns
    base = [r for r in records if r.seq <= seq]
    try:
        states = [fold(base), *(s.state for s in steps)]
        ingest = _ingest_of(store)
        if rewritten & set().union(*(_flow_columns(s, ingest) for s in states)):
            return later, None
        pools: list[Any] = [None]
        if sealed is not None and len(sealed):
            pools.append(np.setdiff1d(np.arange(int(store.n_rows), dtype=np.int64),
                                      np.asarray(sealed, dtype=np.int64)))
        for rows in pools:
            sizes = rule_sizes(store, states, task, rows)
            if _same_size(sizes[-1], read):
                return later, sizes
    except Exception:  # noqa: BLE001 - a state whose rows cannot be read here: not traced
        pass
    return later, None


def _by_rule(spec: Any) -> dict[str, Any] | None:
    """A declared form as a dict, when it is a spline whose k the server set by Harrell's rule
    (``knots_rule`` "harrell"); else None."""
    from turbotab.core.methods import exposure_form as ef

    if isinstance(spec, str) or spec is None:
        return None
    given = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
    by_rule = given.get("form") in ef.SPLINE_FORMS and given.get("knots_rule") == "harrell"
    return given if by_rule else None


def rule_forms(service: Any, pid: str) -> list[RuleRead]:
    """Each standing spline of the project ``pid`` whose k the server set by Harrell's rule
    (``knots_rule`` "harrell"), read again: the rule as the engine records it now
    (``exposure_form.complete_spec`` under the project's decision context, on the rows as the log
    ends; under prediction, where the engine reads it when the spline is declared, on the rows as
    they stood at its record, :func:`size_at_record`, its k ``exposure_form.knots_by_rule``'s on
    that size), beside the k and size the record states (module docstring), and what a later
    record changed of what that size is read under (:func:`input_changes`; the task detected read
    from the target stage, as the sentence's facts read it). The caller waits for
    :data:`RULE_STAGES` first."""
    from turbotab.core.decisions import Refusal, reverted
    from turbotab.core.methods import exposure_form as ef

    records = sorted(service.log(pid).records(), key=lambda r: r.seq)
    withdrawn = reverted(records)
    graph = service.engine.graph
    # What the rule's size is read on: the rows and the outcome the stages before the form card
    # compute, and the outcome, its scale, task and event, which the rule reads itself.
    reads = {s for name in graph.upstream("forms") for s in graph[name].reads}
    reads |= {k for k, _, _ in RULE_INPUTS}
    # What the working table is built from, which the rule's sizes step by step do not re-read.
    table = {s for name in graph.upstream("working") | {"working"} for s in graph[name].reads}
    ctx = service.decision_context(pid)
    info = ctx.artifact("target_info") if callable(ctx.artifact) else None
    detected = ((info.get("column"), info.get("detected_task") or info.get("task"))
                if isinstance(info, dict) else (None, None))
    purpose = getattr(ctx.state, "purpose", None)

    def sealed() -> Any:
        return ctx.sealed() if callable(ctx.sealed) else None

    walked: dict[int, tuple[tuple[tuple[int, str], ...], bool, tuple[str, ...]]] = {}
    at: dict[int, tuple[tuple[str, ...], tuple[tuple[int, str], ...], float | None, bool]] = {}
    out: list[RuleRead] = []
    for column, spec in ef.current_forms(ctx.state).items():
        given = _by_rule(spec)
        if given is None:
            continue
        try:
            again = ef.complete_spec(column, given, ctx)
        except Refusal:  # the rule cannot read these rows: the record's size is not certified
            again = {"knots": None, "n_effective": None}
        last = [r for r in records if r.id not in withdrawn and (
            (r.decision.kind == "set_exposure_form" and r.decision.column == column)
            or (r.decision.kind == "set_forms" and column in r.decision.forms))]
        record = last[-1] if last else None
        read, said = again.get("n_effective"), given.get("n_effective")
        k = again.get("knots") if read is not None else None
        later: tuple[tuple[tuple[int, str], ...], bool, tuple[str, ...]] = ((), True, ())
        at_record = False
        if record is not None and purpose == "prediction":
            # The engine reads the rule when the spline is declared and keeps that k whatever the
            # rows later, so the size is the one on the rows as they stood at its record.
            if record.seq not in at:
                changed, inputs = input_changes(records, record.seq, reads, detected)
                if inputs:
                    at[record.seq] = (inputs, changed, None, False)
                else:
                    size, traced = size_at_record(records, record.seq, reads, ctx.store(),
                                                  ctx.task, read, table=table, sealed=sealed())
                    at[record.seq] = ((), () if traced else later_records(
                        records, record.seq, reads), size, traced)
            inputs, named, size, at_record = at[record.seq]
            later = (named, at_record or bool(inputs), inputs)
            if at_record:
                read, k = size, (ef.knots_by_rule(size) if size is not None else None)
        elif record is not None and not _same_size(said, read):  # a size a sentence will state
            if record.seq not in walked:
                changed, inputs = input_changes(records, record.seq, reads, detected)
                walked[record.seq] = ((changed, True, inputs) if inputs else (*later_changes(
                    records, record.seq, reads, ctx.store(), ctx.task, read, table=table,
                    sealed=sealed()), ()))
            later = walked[record.seq]
        out.append(RuleRead(
            column=column, seq=record.seq if record else None,
            kind=record.decision.kind if record else None, knots=given.get("knots"),
            said=said, k=k, read=read, later=later[0], traced=later[1], inputs=later[2],
            purpose=purpose, at_record=at_record))
    return out


def rule_verdict(r: RuleRead) -> tuple[str, str | None]:
    """What a fresh draw does with the spline ``r`` (module docstring), and its sentence, since the
    record left k to the rule: ``("kept", None)`` where the rule reads the size the record states
    (``exposure_form.rule_stale``), or neither states one; ``("refilled", …)`` where it reads
    another size, the record replayed at the rule's k on it (``r.k``, the record's or another)
    stating that size; ``("refused", …)`` where a later record changed what the size is read under
    (``r.inputs``), where the rule reads no size, and where no live record gives the column the
    spline and the rule reads another size than it states. Under prediction (``r.purpose``, with a
    live record) the size is the one at the record (``r.at_record``): where it is not read there,
    the spline is refused; and a record of the rule's default k, declared before any size was
    read (``r.said`` None), is kept on every draw, whatever the rule reads there and whatever a
    later record changed, since the engine fits the k a spline is declared with (the ruling of
    2026-10-23)."""
    from turbotab.core.methods import exposure_form as ef
    from turbotab.core.voice import listing

    given = {"form": "spline", "knots_rule": "harrell", "n_effective": r.said}
    other = r.read is None and r.said is not None
    predicting = r.purpose == "prediction" and r.seq is not None
    if predicting and r.said is None:
        # The rule's default k, declared before any size was read: the engine fits it on any draw.
        return "kept", None
    stale = other or ef.rule_stale(given, {"n_effective": r.read})
    if (r.at_record or not predicting) and not stale:
        return "kept", None
    where = f"Record {r.seq} ({r.kind})" if r.seq is not None else "The plan"
    before = (f"k = {r.knots} on an effective sample size of {_size(r.said)}"
              if r.said is not None else f"k = {r.knots} with no effective sample size read")
    now = f"{_size(r.read)} (k = {r.k})" if r.read is not None else "no effective sample size"
    rows = "as they stood at that record" if r.at_record else "as the log ends"
    after = ""
    if r.later:
        named = listing([f"{seq} ({kind})" for seq, kind in r.later], limit=len(r.later),
                        ticked=False)
        which = (f"which changed what the size is read under: "
                 f"{listing(list(r.inputs), limit=len(r.inputs), ticked=False)}" if r.inputs else
                 "which changed the size it reads" if r.traced else
                 "which can change the size it reads")
        after = f", after {'record' if len(r.later) == 1 else 'records'} {named}, {which}"
    said = (f"{where} gives `{r.column}` a spline whose k is Harrell's rule on the analyzed rows: "
            f"{before} on the recorded draw, and on this draw the rule reads {now} on the rows "
            f"{rows}{after}.")
    if r.inputs:
        return "refused", (f"{said} The record's size and sentence are read under the outcome, its "
                           f"scale, task and event as they stood when it was recorded, so a size "
                           f"read under later ones is not replayed in it, and the rebuild is "
                           f"refused.")
    if predicting and not r.at_record:
        return "refused", (f"{said} Under prediction the engine reads the rule when the spline is "
                           f"declared and keeps that k, and the size on the rows as they stood "
                           f"at that record is not read on this draw where a later record can "
                           f"change it, so the rebuild is refused.")
    if r.read is None or r.k is None:
        return "refused", (f"{said} With no size read, the rule gives no k on this draw, and the "
                           f"record's k and size are the recorded draw's, so the rebuild is "
                           f"refused.")
    if r.seq is None:
        return "refused", (f"{said} No live record gives `{r.column}` this spline, so none is "
                           f"replayed at the rule's k on this draw, and the rebuild is refused.")
    if predicting:
        return "refilled", (f"{said} Its k was left to the rule, which the engine reads under "
                            f"prediction when the spline is declared, so the record is replayed "
                            f"at the rule's k on this draw's rows as they stood then, stating "
                            f"k = {r.k} on an effective sample size of {_size(r.read)} in place "
                            f"of {before}.")
    return "refilled", (f"{said} Its k was left to the rule, so the record is replayed at the "
                        f"rule's k on this draw, stating k = {r.k} on an effective sample size of "
                        f"{_size(r.read)} in place of {before}.")


def refilled_line(record: Any, sizes: Mapping[str, tuple[int, float]], before: Sequence[Any],
                  facts: Any) -> bytes:
    """``record`` (a ``set_exposure_form`` or ``set_forms``) with each column of ``sizes`` stating
    its k and effective sample size, (k, size), its sentence authored again (:func:`_authored`;
    ``before``, the records before it; ``facts``, what the sentence reads, as the server's
    ``SentenceFacts``), every other field as recorded. Right or refused: raises :class:`Refused`
    where the record's own sentence is not the one authored again on its own k and size, since
    then more than they would change."""
    from turbotab.core.decisions import record_line

    d = record.decision
    if d.kind == "set_exposure_form":
        k, n = sizes[d.column]
        decision = d.model_copy(update={"knots": int(k), "n_effective": float(n)})
    else:
        forms = dict(d.forms)
        for column, (k, n) in sizes.items():
            spec, fields = forms[column], {"knots": int(k), "n_effective": float(n)}
            forms[column] = (spec.model_copy(update=fields) if hasattr(spec, "model_copy")
                             else {**spec, **fields})
        decision = d.model_copy(update={"forms": forms})
    if record.sentence is not None and _authored(record, d, before, facts) != record.sentence:
        raise Refused(f"Record {record.seq} ({d.kind})'s sentence is not the one the log authors "
                      f"for it now, so it cannot be said again stating this draw's k and size: "
                      f"the rebuild is refused.")
    text = _authored(record, decision, before, facts) if record.sentence is not None else None
    again = record.model_copy(update={"decision": decision, "sentence": text})
    return record_line(again).rstrip(b"\n")


def relocked_line(record: Any, before: Sequence[Any], again: Sequence[Any],
                  facts: Any) -> bytes | None:
    """``record`` (a ``lock_plan``) holding the plan the records before it give once a spline is
    refilled (``again``, those records as rebuilt; ``before``, as recorded; ``plan_lock.plan_of``,
    as the server records a lock's plan) and that plan's SHA-256, its sentence, which quotes it,
    authored again (:func:`_authored`), every other field as recorded; None (its line kept as its
    own bytes) where the plan is the one recorded, or the lock holds none. Right or refused: raises
    :class:`Refused` where the lock's plan or digest is not the one the records before it give
    now, or its sentence not the one the log authors for it now, since then more than the refilled
    spline would change."""
    from turbotab.core import plan_lock
    from turbotab.core.decisions import fold, record_line

    d = record.decision
    if d.plan is None:
        return None
    recorded = plan_lock.digest(plan_lock.plan_of(fold(list(before))))
    if plan_lock.digest(d.plan) != d.digest or recorded != d.digest:
        raise Refused(f"Record {record.seq} ({d.kind})'s plan is not the one the records before it "
                      f"give now, so it cannot hold the spline at this draw's k and size: the "
                      f"rebuild is refused.")
    plan = plan_lock.plan_of(fold(list(again)))
    if plan_lock.digest(plan) == d.digest:
        return None
    decision = d.model_copy(update={"plan": plan, "digest": plan_lock.digest(plan)})
    if record.sentence is not None and _authored(record, d, before, facts) != record.sentence:
        raise Refused(f"Record {record.seq} ({d.kind})'s sentence is not the one the log authors "
                      f"for it now, so it cannot be said again quoting the plan that holds this "
                      f"draw's k and size: the rebuild is refused.")
    text = _authored(record, decision, again, facts) if record.sentence is not None else None
    return record_line(record.model_copy(update={"decision": decision, "sentence": text})
                       ).rstrip(b"\n")


def fresh_splines(service: Any, pid: str, written: Path, log: bytes, timeout: float
                  ) -> tuple[bytes, list[str]]:
    """The spline rule on the project ``pid`` (made from a fresh draw; its log ``log``, at
    ``written``), as the module docstring says: once the stages the rule reads are computed
    (:data:`RULE_STAGES`), each spline the rule set is read again (:func:`rule_forms`) and ruled
    (:func:`rule_verdict`); a record whose k and size are refilled is written again
    (:func:`refilled_line`), and each plan lock after it (:func:`relocked_line`), and the engine
    told. Returns the log as written and the sentences of the refilled records; raises
    :class:`Refused` for a spline the rule refuses."""
    from turbotab.core.decisions import fold, read_record
    from turbotab.core.methods import exposure_form as ef
    from turbotab.server.service import SentenceFacts

    lines = log_lines(log)
    state = fold([r for r, _ in lines])
    if not any(_by_rule(f) for f in ef.current_forms(state).values()):
        return log, []
    _wait(service, pid, RULE_STAGES, timeout)
    refused: list[str] = []
    said: list[str] = []
    sizes: dict[int, dict[str, tuple[int, float]]] = {}
    for r in rule_forms(service, pid):
        verdict, text = rule_verdict(r)
        if verdict == "refused":
            refused.append(str(text))
        elif verdict == "refilled":
            said.append(str(text))
            sizes.setdefault(int(r.seq), {})[r.column] = (int(r.k), float(r.read))
    if refused:
        raise Refused(" ".join(refused))
    if not sizes:
        return log, []
    ctx = service.decision_context(pid)
    first = min(sizes)
    out: list[tuple[Any, bytes]] = []
    for i, (record, line) in enumerate(lines):
        before = [x for x, _ in lines[:i]]
        if record.seq in sizes:
            line = refilled_line(record, sizes[record.seq], before,
                                 SentenceFacts(ctx, record.decision, before))
            record = read_record(json.loads(line))
        elif record.decision.kind == "lock_plan" and record.seq > first:
            rebuilt = [x for x, _ in out]
            relocked = relocked_line(record, before, rebuilt,
                                     SentenceFacts(ctx, record.decision, rebuilt))
            if relocked is not None:
                line, record = relocked, read_record(json.loads(relocked))
        out.append((record, line))
    again = join_lines(out)
    written.write_bytes(again)
    service.engine.on_decision(pid)
    return again, said


@dataclasses.dataclass(frozen=True)
class FreshDraw:
    """A new draw of the generator the recorded table came from (module docstring).
    ``generator`` is a function of a seed that returns the table (a pandas DataFrame, written as
    the intake writes it: CSV, no index), or ``"module:function"`` naming one."""

    generator: Callable[[int], Any] | str
    recorded_seed: int
    seed: int

    def function(self) -> Callable[[int], Any]:
        if callable(self.generator):
            return self.generator
        module, sep, name = str(self.generator).partition(":")
        if not sep or not module or not name:
            raise Refused(f"A generator is named MODULE:FUNCTION, not {self.generator!r}.")
        try:
            return getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError) as exc:
            raise Refused(f"The generator {self.generator} cannot be loaded: {exc}") from None

    def name(self) -> str:
        if isinstance(self.generator, str):
            return self.generator
        return f"{getattr(self.generator, '__module__', '?')}:" \
               f"{getattr(self.generator, '__qualname__', repr(self.generator))}"


def write_draw(generator: Callable[[int], Any], seed: int, path: Path) -> tuple[str, int]:
    """The generator's table at ``seed`` written to ``path`` (CSV, no index); its SHA-256 and rows."""
    frame = generator(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest(), int(len(frame))


def fresh_table(prov: Mapping[str, Any], fresh: FreshDraw, folder: Path) -> FreshDrawCheck:
    """The fresh draw's table in ``folder`` (named as the recorded table), after the generator is
    shown to write the recorded table at the recorded seed; raises :class:`Refused` otherwise, for
    the recorded seed given as the new one, and for a record with joined files."""
    inputs = [e for e in prov.get("inputs") or [] if e.get("role") != "codebook"]
    joined = [e for e in inputs if e.get("role") != "table"]
    if joined:
        raise Refused(f"A fresh draw replaces the one table; this record also reads "
                      f"{len(joined)} joined file(s) ({', '.join(str(e.get('name')) for e in joined)}).")
    table = next((e for e in inputs if e.get("role") == "table"), None)
    if table is None:
        raise Refused("The record names no table for a fresh draw to replace.")
    if fresh.seed == fresh.recorded_seed:
        raise Refused(f"Seed {fresh.seed} is the recorded table's own: a fresh draw takes another "
                      f"seed.")
    generator = fresh.function()
    name = Path(str(table.get("name") or Path(str(table.get("path"))).name)).name
    recorded = str(table.get("sha256") or "")
    again, _ = write_draw(generator, fresh.recorded_seed, folder / "recorded" / name)
    if again != recorded:
        raise Refused(f"The generator {fresh.name()} does not write the recorded table at seed "
                      f"{fresh.recorded_seed}: its SHA-256 is {again}, the record's is {recorded}. "
                      f"A fresh draw is a draw of the generator the record was made from.")
    path = folder / "fresh" / name
    sha, rows = write_draw(generator, fresh.seed, path)
    return FreshDrawCheck(generator=fresh.name(), recorded_seed=fresh.recorded_seed,
                          seed=fresh.seed, recorded_sha256=recorded, regenerated_sha256=again,
                          sha256=sha, rows=rows, path=str(path))


# ── the run ──────────────────────────────────────────────────────────────────


def _fresh_home(home: str | None) -> Path:
    if home is None:
        return Path(tempfile.mkdtemp(prefix="turbotab-replay-"))
    path = Path(home).expanduser()
    if path.exists() and any(path.iterdir()):
        raise Refused(f"{path} is not empty: a replay runs in a fresh TurboTab home.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _restore_joined(project_dir: Path, entry: Mapping[str, Any], path: str) -> None:
    """A joined file, read into ``files/<its recorded id>/`` as adding it does (DATAIN)."""
    from turbotab.core.datastore import ingest

    fid = str(entry["file_id"])
    folder = project_dir / "files" / fid
    folder.mkdir(parents=True, exist_ok=True)
    info = ingest(Path(path), folder / "raw.parquet")
    meta = {"id": fid, "name": entry.get("name"), "source_kind": "path", "source_path": str(path),
            "fingerprint": info.fingerprint, "n_rows": info.n_rows, "n_cols": info.n_cols,
            "columns": [c.name for c in info.columns], "warnings": list(info.warnings)}
    (folder / "file.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")


def _wait(service: Any, pid: str, stages: Sequence[str], timeout: float) -> None:
    end = time.monotonic() + timeout
    while True:
        statuses = service.engine.status(pid)
        done = True
        for stage in stages:
            status = statuses[stage]
            if status.status == "error":
                raise RuntimeError(f"the {stage} stage failed on replay: {status.error}")
            if status.status == "blocked":
                raise RuntimeError(f"the {stage} stage is blocked on replay, waiting for "
                                   f"{', '.join(status.missing) or 'an answer'}")
            if status.status != "fresh":
                done = False
        if done:
            return
        if time.monotonic() > end:
            raise RuntimeError(f"the replay did not finish within {timeout:g} seconds")
        time.sleep(0.2)


def run(bundle: str | Path, *, data: str | None = None, files: Mapping[str, str] | None = None,
        home: str | None = None, keep: bool = False, timeout: float = 3600.0,
        workers: int | None = None, cut: bool | str = False,
        fresh: FreshDraw | None = None) -> ReplayReport:
    """Replay ``bundle`` (module docstring). Never raises for a refusal or a failed comparison:
    the report says which. ``cut``: True truncates the log by its purpose, or names the kind to cut
    before (:func:`truncate`); ``fresh``: rebuild on a fresh draw (:class:`FreshDraw`), refused
    for a log with a live opening left uncut and for a spline the rule rules out on the new draw
    (:func:`fresh_splines`). Either rebuilds without comparing (``finished`` and
    ``estimates_again``)."""
    started = time.monotonic()
    report = ReplayReport(bundle=str(bundle), reproduced=False)
    rebuild = bool(cut) or fresh is not None
    try:
        recorded_files, prov = load_bundle(bundle)
        log = recorded_files["decisions.jsonl"]
        if cut:
            log, report.cut = truncate(log, None if cut is True else str(cut))
        if fresh is None:
            report.inputs = verify_inputs(prov, data, files)
    except Refused as exc:
        report.refused = exc.message
        report.inputs = exc.inputs
        return report
    from turbotab.core.config import Settings
    from turbotab.core.datastore import fingerprint_file
    from turbotab.core.decisions import fold
    from turbotab.core.export import gate
    from turbotab.core.export.bundle import contents, matrix_bytes
    from turbotab.server.service import SOURCE_FILE, ProjectService

    try:
        path = _fresh_home(home)
        if fresh is not None:
            report.fresh = fresh_table(prov, fresh, path / "draws")
            refuse_live_opening(log)
            table = next(e for e in prov["inputs"] if e.get("role") == "table")
            report.inputs = [InputCheck(
                role="table", name=str(table.get("name")), path=report.fresh.path,
                recorded_sha256=report.fresh.recorded_sha256, sha256=report.fresh.sha256,
                matches=False,
                message=(f"The table `{table.get('name')}` is a fresh draw: {report.fresh.generator} "
                         f"writes the recorded table at seed {report.fresh.recorded_seed} (SHA-256 "
                         f"{report.fresh.recorded_sha256}), and its draw at seed "
                         f"{report.fresh.seed} is SHA-256 {report.fresh.sha256}."))]
    except Refused as exc:
        report.refused = exc.message
        if home is None and "path" in locals():
            shutil.rmtree(path, ignore_errors=True)
        return report
    report.home = str(path)
    settings = dataclasses.replace(Settings.from_env(), home=path, mode="local",
                                   **({"workers": workers} if workers else {}))
    try:
        service = ProjectService(settings, replay=True)
    except Exception as exc:  # noqa: BLE001 - said, never a trace
        report.messages.append(f"The replay could not start TurboTab in {path}: {exc}")
        return report
    try:
        table = next(c for c in report.inputs if c.role == "table")
        name = str((prov.get("project") or {}).get("name") or Path(table.path).stem)
        meta = service.workspace.create_project(name, table.path, "path",
                                                source_name=table.name)
        report.project = meta.id
        pdir = service.workspace.project_dir(meta.id)
        (pdir / SOURCE_FILE).write_text(json.dumps({"fingerprint": fingerprint_file(
            Path(table.path))}), "utf-8")
        for entry in prov.get("inputs") or []:
            if entry.get("role") == "joined file":
                check = next(c for c in report.inputs if c.role == "joined file"
                             and c.name == entry.get("name"))
                _restore_joined(pdir, entry, check.path)
        written = service.workspace.decisions_path(meta.id)
        lines = log_lines(log)
        if fresh is not None and any(r.decision.kind in FINDING_KINDS for r, _ in lines):
            # The findings rule: the findings the new bytes raise, and each finding's record read
            # against them (module docstring).
            ruled, report.unanswered = fresh_findings(service, meta.id, written, lines, timeout)
            report.left_out, report.redrawn = ruled.left_out, ruled.redrawn
            log = join_lines(ruled.kept)
        written.write_bytes(log)
        records = service.log(meta.id).records()
        report.decisions = len(records)
        service.engine.on_decision(meta.id)
        if fresh is not None:
            # The spline rule, read before the result stages are waited for (module docstring).
            log, refilled = fresh_splines(service, meta.id, written, log, timeout)
            report.messages.extend(refilled)
        _wait(service, meta.id, gate.result_stages(fold(records)), timeout)
        source = service.export_source(meta.id)
        # A rebuild is not the recorded analysis: its bundle is assembled as it stands, ungated.
        again = contents(source, gated=not rebuild)
        out = path / "replay"
        out.mkdir(exist_ok=True)
        raw = matrix_bytes(source)
        if raw is not None:
            (out / "model_matrix.parquet").write_bytes(raw)
        (out / "bundle.zip").write_bytes(again.zip())
        if rebuild:
            report.estimates_again = dict(again.provenance.get("estimates") or {})
            report.finished = True
            return report
        found = compare(prov, again.provenance, recorded_files, again.files)
        report.matrix = found["matrix"]
        report.estimates = found["estimates"]
        report.stage_keys = found["stage_keys"]
        report.stage_keys_same = found["stage_keys_same"]
        report.plan_same = found["plan_same"]
        report.files_same = found["files_same"]
        report.files_different = found["files_different"]
        report.reproduced = bool(report.matrix.identical and report.matrix.content_identical
                                 and not report.estimates.mismatched
                                 and not report.estimates.missing
                                 and not report.estimates.extra)
    except Refused as exc:  # the findings rule or a spline's rule refuses the fresh draw
        report.refused = exc.message
        report.project = None
    except Exception as exc:  # noqa: BLE001 - a replay that cannot finish says why, never a trace
        report.messages.append(f"The replay could not finish: {exc}")
        if rebuild:
            report.finished = False
    finally:
        service.close()
        report.seconds = round(time.monotonic() - started, 3)
        if not keep and home is None:
            shutil.rmtree(path, ignore_errors=True)
            report.home = None
    return report


# ── the command ──────────────────────────────────────────────────────────────


def summary(report: ReplayReport) -> str:
    """The report in a few lines, for the terminal."""
    lines = [f"Replay of {report.bundle}" + (f" in a fresh home at {report.home}"
                                             if report.home else "")]
    for c in report.inputs:
        lines.append(f"  {c.message}")
    if report.refused:
        lines.append(f"Refused: {report.refused}")
        return "\n".join(lines)
    if report.cut is not None:
        c = report.cut
        lines.append(f"  The log cut before its first live `{c.kind}` (record {c.seq}): {c.kept} "
                     f"records kept, {c.left} left out." if c.seq is not None else
                     f"  The log holds no live `{c.kind}`: nothing was cut.")
    for o in report.left_out:
        lines.append(f"  Record {o.seq} ({o.kind}) left out of the fresh draw: {o.why}.")
    for d in report.redrawn:
        lines.append(f"  Record {d.seq} (apply_repair) replayed with the fresh draw's settings for "
                     f"`{d.option}` of `{d.finding_id}`; these differ from the record's: "
                     f"{', '.join(d.settings)}.")
    if report.unanswered:
        lines.append(f"  Findings the fresh draw raises that no record answers: "
                     f"{', '.join(report.unanswered)}.")
    if report.decisions is not None:
        lines.append(f"  The decision log: {report.decisions} records, replayed as "
                     + ("rebuilt above." if report.cut or report.fresh else "recorded."))
    if report.matrix is not None:
        m = report.matrix
        lines.append(f"  The model matrix ({m.n_rows or 0:,} rows by {m.n_cols or 0:,} columns): "
                     f"Parquet "
                     f"SHA-256 {'identical' if m.identical else 'different'} "
                     f"({m.parquet_sha256}); its values "
                     f"{'identical' if m.content_identical else 'different'}.")
    if report.estimates is not None:
        e = report.estimates
        lines.append(f"  The estimates: {e.reproduced} of {e.n_recorded} reproduced (largest "
                     f"difference {e.max_abs_diff:.3g}; tolerance {e.tolerance:g})"
                     + (f"; differing: {', '.join(e.mismatched[:5])}" if e.mismatched else "")
                     + (f"; not recomputed: {', '.join(e.missing[:5])}" if e.missing else "")
                     + ".")
    if report.stage_keys_same is not None:
        same = "the same" if report.stage_keys_same else "different (another engine version)"
        lines.append(f"  The result stages' keys: {same}.")
    if report.files_different:
        lines.append(f"  Files that differ: {', '.join(report.files_different)}.")
    lines += [f"  {m}" for m in report.messages]
    if report.finished is not None:
        lines.append(f"Rebuilt ({len(report.estimates_again or {})} estimates); nothing compared."
                     if report.finished else "Not rebuilt.")
        return "\n".join(lines)
    lines.append("Reproduced." if report.reproduced else "Not reproduced.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m turbotab.replay",
        description="Replay a TurboTab export bundle: check its input files against their hashes, "
                    "rebuild the analysis from the decision log in a fresh home, and compare the "
                    "model matrix and every reported estimate.")
    parser.add_argument("bundle", help="the export bundle (.zip)")
    parser.add_argument("--data", help="the table's input file (default: the path the record names)")
    parser.add_argument("--file", action="append", default=[], metavar="ID=PATH",
                        help="a joined file, by its id or name (repeatable)")
    parser.add_argument("--home", help="an empty folder to replay in (default: a new temporary one)")
    parser.add_argument("--keep", action="store_true", help="keep the temporary home afterwards")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds to wait (default 3600)")
    parser.add_argument("--truncate", nargs="?", const=True, default=False, metavar="KIND",
                        help="rebuild from the log before its first live open_seal (under "
                             "Estimate lock_plan), or before KIND; nothing is compared")
    parser.add_argument("--fresh-draw", metavar="MODULE:FUNCTION",
                        help="rebuild on a fresh draw of this generator (a function of a seed); "
                             "nothing is compared, and a log with a live open_seal is refused "
                             "unless --truncate open_seal cuts it")
    parser.add_argument("--recorded-seed", type=int, help="the seed the recorded table was drawn at")
    parser.add_argument("--seed", type=int, help="the fresh draw's seed")
    args = parser.parse_args(argv)
    fresh = None
    if args.fresh_draw is not None:
        if args.recorded_seed is None or args.seed is None:
            parser.error("--fresh-draw takes --recorded-seed and --seed")
        fresh = FreshDraw(args.fresh_draw, args.recorded_seed, args.seed)
    given: dict[str, str] = {}
    for item in args.file:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--file takes ID=PATH, not {item!r}")
        given[key] = value
    report = run(args.bundle, data=args.data, files=given, home=args.home, keep=args.keep,
                 timeout=args.timeout, cut=args.truncate, fresh=fresh)
    if args.json:
        sys.stdout.write(report.model_dump_json(indent=1) + "\n")
    else:
        sys.stdout.write(summary(report) + "\n")
    if report.refused:
        return 2
    if report.finished is not None:
        return 0 if report.finished else 1
    return 0 if report.reproduced else 1


__all__ = ["CUT_KINDS", "Cut", "EstimateCheck", "FINDING_KINDS", "FreshDraw", "FreshDrawCheck",
           "InputCheck", "LeftOut", "MatrixCheck", "RULE_STAGES", "Redrawn", "Refused",
           "ReplayReport", "RuleRead", "Ruled", "close", "compare", "compare_estimates",
           "findings_inputs", "findings_rule", "fresh_findings", "fresh_splines", "fresh_table",
           "join_lines", "later_changes", "later_records", "live_record", "load_bundle",
           "log_lines", "main", "refilled_line", "refuse_live_opening", "relocked_line",
           "rule_forms", "rule_sizes", "rule_verdict", "run", "size_at_record", "summary",
           "truncate", "verify_inputs", "write_draw"]
