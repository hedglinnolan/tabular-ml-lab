"""A record of every fit (SWARM_ENGINE_PLAN §2 E1): one per distinct fit that completes.

A fit completes when its key turns fresh (the engine's ``fit`` stage event). The server writes one
record for each key it has not recorded, in the order the keys turn fresh, to ``fits.jsonl`` beside
the project, and never rewrites one: a press of Fit that computes nothing new, or a revert back to
a fit already recorded, writes nothing (:class:`RunLog`).

Each record holds what the fit was, read from the engine and the decision log when it completed:

* the purpose and outcome, the fit key, the plan's digest (``plan_lock.digest(plan_lock.plan_of
  (state))``, defined under either goal), the records in force (their seqs;
  ``provenance.in_force``), the press of Fit in force (``fit_press.read_press`` while it is for the
  outcome and not withdrawn), the families fitted, the seconds the server saw the fit run and the
  runner's workers;
* per family, under Predict the cross-validated summary (the fit's ``cv``, every metric), under
  Estimate the exposure's rows of the coefficient table with their intervals (``estimand.
  annotate_fit``'s rows); never a held-out score, which stays in the fit's sealed frame;
* under Predict, against the record before it in completion order, each family's movement by
  :func:`materiality.paired_folds` on the two fits' fold scores of the primary metric, where the
  two were scored on the same folds (:func:`folds_digest`: the same training rows in the same folds,
  drawn the same way); otherwise none, and why (:func:`movement`).

The server serves the records (``GET /projects/{pid}/fits``) through the fit's own serving gate, so
each number is withheld exactly where the stage route withholds it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from turbotab.core.decisions import Purpose
from turbotab.core.materiality import Movement, paired_folds

try:  # POSIX: one append at a time across processes, as the decision log does
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

RUNS_FILE = "fits.jsonl"

# Why a record carries no paired movement (each says only what :func:`movement` checked).
NO_PREVIOUS = "No earlier fit is recorded to compare this one with."
NOT_PREDICT = ("This fit is under Estimate: movement by paired folds compares the cross-validated "
               "scores of two fits under Predict.")
PREVIOUS_NOT_PREDICT = ("The previous fit is under Estimate: movement by paired folds compares the "
                        "cross-validated scores of two fits under Predict.")
FOLDS_UNREAD = "The folds a fit was scored on could not be read, so no paired movement is measured."
ANSWERS_UNREAD = ("No prefix of the decision log keys this fit, so the answers it was computed from, "
                  "and its folds, are not known.")
OTHER_FOLDS = "The previous fit's training rows or their folds differ from this one's."


def _other_target(previous: str | None, current: str | None) -> str:
    return f"The previous fit predicts `{previous}`, not `{current}`."


def _other_metric(previous: str | None, current: str | None) -> str:
    return f"The previous fit is scored by `{previous}`, this one by `{current}`."


def _other_seed(previous: int | None, current: int | None) -> str:
    return (f"The previous fit's folds were drawn with seed {previous}, this one's with seed "
            f"{current}.")


def _not_in_previous(family: str) -> str:
    return f"`{family}` was not fitted in the previous fit."


def _too_few(family: str) -> str:
    return f"Fewer than two folds scored `{family}` in both fits."


def _constant(family: str, change: float) -> str:
    return (f"Every fold scored `{family}` {change:+.4f} from the previous fit, so the change has no "
            f"fold-to-fold spread and the paired measure is not defined.")


def _what(family: str) -> str:
    """The alternative :func:`materiality.paired_folds` names: what differs, for this family."""
    return f"For `{family}`, what differs from the previous fit"


class FoldMovement(BaseModel):
    """One family's movement against the previous record: ``movement`` from
    :func:`materiality.paired_folds` (``|mean Δ| / SD(Δ)`` of the primary metric's fold scores, this
    fit's less the previous one's), or None and ``why``."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    family: str
    against: str  # the previous record's id
    movement: Movement | None = None
    why: str | None = None


class FitRecord(BaseModel):
    """One fit, as it completed (module docstring). Written once, never rewritten."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    id: str
    seq: int  # completion order, from 1
    at: datetime  # when the fit turned fresh (the engine's stage event)
    purpose: Purpose | None
    target: str | None
    fit_key: str
    # The plan's digest and the records in force (their seqs) of the decision log as it stood when
    # the fit completed, checked to key this fit; null when no record prefix keys it.
    plan_digest: str | None
    decisions: list[int] | None
    press: dict[str, Any] | None  # the press of Fit in force, as kept beside the project
    families: list[str]
    seconds: float | None  # from the fit's job running to its key turning fresh; null unseen
    workers: int  # the runner's worker processes
    primary_metric: str | None
    folds: str | None  # :func:`folds_digest` of the split the fit read; null when unread
    seed: int | None  # the split's seed
    # Predict: each family's cross-validated summary, the fit's own ``cv`` (every metric).
    scores: dict[str, dict[str, Any]] = {}
    # Estimate: each family's rows of the exposure, the fit's own coefficient rows.
    effects: dict[str, list[dict[str, Any]]] = {}
    movement: list[FoldMovement] = []
    movement_why: str | None = None


# ── what a fit holds ─────────────────────────────────────────────────────────


def folds_digest(assignment: Any, *, scheme: str | None = None,
                 validation: str | None = None) -> str:
    """The SHA-256 of the folds a fit is scored on: the training rows (by row id, in order), each
    one's fold in every repeat (``fold``, ``fold_r1``, …), its time rank where the folds follow
    time, and how the split draws and validates them (``scheme``, ``validation``). Two fits with
    the same digest scored the same rows in the same folds."""
    from turbotab.core.stages.modeling import repeat_columns

    train = assignment.loc[assignment["train"].to_numpy(dtype=bool)]
    columns = ["fold", *repeat_columns(assignment)]
    payload: dict[str, Any] = {
        "scheme": scheme, "validation": validation,
        "rows": [str(i) for i in train.index],
        "folds": {c: [None if _missing(v) else int(v) for v in train[c].tolist()] for c in columns},
    }
    if "order" in assignment.columns:
        payload["order"] = [None if _missing(v) else float(v) for v in train["order"].tolist()]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def family_numbers(fit: Mapping[str, Any], purpose: str | None
                   ) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """``(families, scores, effects)`` of a fit's public data: every family fitted, and per family
    the cross-validated summary under Predict or, under Estimate, the exposure's coefficient rows:
    those ``estimand.annotate_fit`` leaves in ``coefficients`` once a declared estimand names the
    exposure (``fit["estimand"]``). With no estimand declared no row is the exposure's: none."""
    families: list[str] = []
    scores: dict[str, dict[str, Any]] = {}
    effects: dict[str, list[dict[str, Any]]] = {}
    declared = isinstance(fit.get("estimand"), Mapping)
    for m in fit.get("models") or []:
        if not isinstance(m, Mapping) or not m.get("family"):
            continue
        family = str(m["family"])
        families.append(family)
        if purpose == "prediction":
            scores[family] = dict(m.get("cv") or {})
        elif purpose == "inference" and declared and m.get("coefficients"):
            effects[family] = [dict(r) for r in m["coefficients"]]
    return families, scores, effects


# ── the movement under Predict ───────────────────────────────────────────────


def _fold_scores(scores: Mapping[str, Any], family: str, metric: str | None) -> list[Any] | None:
    summary = (scores.get(family) or {}).get(metric) if metric else None
    folds = summary.get("folds") if isinstance(summary, Mapping) else None
    return list(folds) if isinstance(folds, list) else None


def movement(previous: FitRecord | None, *, purpose: str | None, target: str | None,
             metric: str | None, folds: str | None, seed: int | None,
             scores: Mapping[str, Any]) -> tuple[list[FoldMovement], str | None]:
    """Each family's movement against ``previous`` (the record before this one in completion
    order), with the reason there is none, as ``(movements, why)``. A movement is measured only
    between two fits under Predict, for the same outcome and primary metric, scored on the same
    folds (:func:`folds_digest`); per family, only where both fits scored it in at least two of the
    same folds. Every difference the same nonzero amount has no spread, so the paired measure is not
    defined there: refused with why, never an infinite value."""
    if purpose is None:
        return [], ANSWERS_UNREAD
    if purpose != "prediction":
        return [], NOT_PREDICT
    if previous is None:
        return [], NO_PREVIOUS
    if previous.purpose != "prediction":
        return [], PREVIOUS_NOT_PREDICT
    if previous.target != target:
        return [], _other_target(previous.target, target)
    if previous.primary_metric != metric:
        return [], _other_metric(previous.primary_metric, metric)
    if previous.folds is None or folds is None:
        return [], FOLDS_UNREAD
    if previous.folds != folds:
        if previous.seed != seed:
            return [], _other_seed(previous.seed, seed)
        return [], OTHER_FOLDS
    out: list[FoldMovement] = []
    for family in scores:
        if family not in previous.scores:
            out.append(FoldMovement(family=family, against=previous.id,
                                    why=_not_in_previous(family)))
            continue
        now = _fold_scores(scores, family, metric)
        before = _fold_scores(previous.scores, family, metric)
        pairs = [(float(b), float(a)) for b, a in zip(before or [], now or [])
                 if not _missing(a) and not _missing(b)]
        if len(pairs) < 2:
            out.append(FoldMovement(family=family, against=previous.id, why=_too_few(family)))
            continue
        moved = paired_folds([b for b, _ in pairs], [a for _, a in pairs], _what(family))
        if moved.value is not None and not math.isfinite(moved.value):
            change = pairs[0][1] - pairs[0][0]
            out.append(FoldMovement(family=family, against=previous.id,
                                    why=_constant(family, change)))
            continue
        out.append(FoldMovement(family=family, against=previous.id, movement=moved))
    return out, None


# ── the log ──────────────────────────────────────────────────────────────────


class RunLog:
    """Append-only ``fits.jsonl``: one line per fit key, in the order they were written. Safe
    across threads and processes: each append reads the log and writes one line under an exclusive
    ``flock``, so a key is recorded once however close two writers come. A torn line is skipped on
    read."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._lock = threading.Lock()

    def records(self) -> list[FitRecord]:
        with self._lock:
            return self._load()

    def _load(self) -> list[FitRecord]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        out: list[FitRecord] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                out.append(FitRecord.model_validate_json(line))
            except ValueError:
                continue  # a torn line: the record it began is not there
        return out

    def append(self, make: Callable[[list[FitRecord]], dict[str, Any] | None]) -> FitRecord | None:
        """Write the record ``make(records)`` returns (its fields but ``id`` and ``seq``), unless it
        returns None or a record of its ``fit_key`` is already written. ``make`` reads the records
        as they stand under the lock (the last is the one before it). Returns the record written."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                existing = self._load()
                fields = make(existing)
                if fields is None or any(r.fit_key == fields.get("fit_key") for r in existing):
                    return None
                record = FitRecord(id=uuid.uuid4().hex, seq=len(existing) + 1, **fields)
                data = (record.model_dump_json() + "\n").encode("utf-8")
                size = os.fstat(fd).st_size
                if size and os.pread(fd, 1, size - 1) != b"\n":
                    data = b"\n" + data  # never glue onto a torn line
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
                return record
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)


def keyed_prefix(records: Sequence[Any], key: str, key_of: Callable[[Any], str | None]
                 ) -> list[Any] | None:
    """The longest prefix of ``records`` (sorted by seq) whose folded state keys ``key``
    (``key_of(state)``), or None when none does: the answers a fit was computed from, though the
    log moved on before its record was read."""
    from turbotab.core.decisions import Refusal, fold

    ordered = sorted(records, key=lambda r: r.seq)
    for n in range(len(ordered), -1, -1):
        prefix = ordered[:n]
        try:
            if key_of(fold(prefix)) == key:
                return prefix
        except Refusal:
            continue
    return None


def appended_by(records: Sequence[Any], at: datetime | None) -> list[Any]:
    """The records appended by ``at`` (each record's own time); all of them when ``at`` is None."""
    if at is None:
        return list(records)
    when = at if at.tzinfo is not None else at.replace(tzinfo=timezone.utc)
    return [r for r in records if r.at <= when]


__all__ = ["ANSWERS_UNREAD", "FOLDS_UNREAD", "FitRecord", "FoldMovement", "NOT_PREDICT",
           "NO_PREVIOUS", "OTHER_FOLDS", "PREVIOUS_NOT_PREDICT", "RUNS_FILE", "RunLog",
           "appended_by", "family_numbers", "folds_digest", "keyed_prefix", "movement"]
