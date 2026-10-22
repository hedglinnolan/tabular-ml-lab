"""Predictions on new rows from a fitted plan (SWARM_PLAN §8: "a scorer that predicts on new rows").

A plan replayed under prediction (``core/export/replay.py``, its log truncated before the opening of
the held-out rows, on any draw) holds each family's pipeline fit on the training rows (the fit
stage's ``fitted`` objects) and the design's predictors (``spec.inputs``). :func:`predict` reads
new rows through the same steps the training rows went through and asks that pipeline: for a
yes/no outcome the probability of the event (the level the plan coded 1), for a number the
prediction. The swarm's decision quality (§7) scores these against the generator's true risk,
the regret against the oracle.

**The steps a new row goes through.** The rows are read into Parquet by the engine's own ingest
(``datastore.ingest``), each predictor as the working table computes it, by the repairs' own SQL
(``repairs.evaluate`` with ``repairs.column_expressions``, "the same SQL the working table is made
with"), and normalized as every modeling stage reads its frame (``pipeline.normalize_frame``).

**Right or refused.** A plan whose working table is not those steps cannot be scored that way, so
before any new row is read the same steps are run on the plan's own training rows, from the table
the project was made from, and each predictor compared with the working table the fit read; a
predictor that differs (one QC-RLSC corrected, say, or any step but the repairs' own SQL) is
refused by name, as is a working table whose rows are not the table's own (it combines rows, or
leaves reference rows out and numbers the rest again). Refused too: a plan under inference
(its result is an estimate, not a prediction), a task other than a yes/no outcome or a number, a
feature-major table, model updating by shrinkage (applied after the fit, which this module does not
apply), a family the fit did not keep, more than one family with none named, a fit or design not
computed, new rows without a predictor that is a column of the table (each named), and new rows
whose predictor reads as text where the plan's read numbers, or the reverse.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

TASKS = ("binary", "regression")


class ScoreRefused(Exception):
    """The plan cannot be scored on new rows; ``message`` says why."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass
class Scored:
    """The predictions on the new rows, in their order, and what they were read from."""

    family: str
    task: str
    predictions: np.ndarray
    inputs: list[str]
    event: str | None = None
    checked_rows: int = 0  # the training rows the steps were shown to reproduce
    notes: list[str] = field(default_factory=list)


def _bundle(service: Any, pid: str, stage: str, statuses: Any) -> Any:
    from turbotab.core.graph import read_artifact

    status = statuses[stage]
    if status.status != "fresh" or status.key is None:
        raise ScoreRefused(f"The {stage} stage is not computed ({status.status}): replay the plan "
                           f"to its fit first.")
    return read_artifact(service.workspace.cache_dir(pid), stage, status.key)


def _kind(series: pd.Series) -> str:
    return "number" if pd.api.types.is_numeric_dtype(series) and not \
        pd.api.types.is_bool_dtype(series) else "text"


def _same(a: pd.Series, b: pd.Series) -> bool:
    """Equal value for value, blanks equal to blanks (numbers exactly)."""
    if len(a) != len(b):
        return False
    na, nb = a.isna().to_numpy(), b.isna().to_numpy()
    if not np.array_equal(na, nb):
        return False
    av, bv = a.to_numpy()[~na], b.to_numpy()[~nb]
    if _kind(a) == "number" and _kind(b) == "number":
        return bool(np.array_equal(av.astype(float), bv.astype(float)))
    return bool(np.array_equal(av.astype(str), bv.astype(str)))


def _columns(parquet: str | Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(str(parquet)).names)


def _ingested(frame: pd.DataFrame | str | Path, folder: Path) -> Path:
    """The new rows as the engine's ingest reads a file into Parquet."""
    from turbotab.core.datastore import ingest

    source = Path(frame) if isinstance(frame, (str, Path)) else folder / "rows.csv"
    if not isinstance(frame, (str, Path)):
        frame.to_csv(source, index=False)
    out = folder / "rows.parquet"
    ingest(source, out)
    return out


def predict(service: Any, pid: str, rows: pd.DataFrame | str | Path, *,
            family: str | None = None) -> Scored:
    """The plan's predictions on ``rows`` (a frame, or a table file the engine reads), for
    ``family`` (by default the one family fitted). Raises :class:`ScoreRefused` (module docstring)."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.decisions import fold
    from turbotab.core.models.metrics import predict as predict_with
    from turbotab.core.models.pipeline import DesignSpec, normalize_frame
    from turbotab.core.repairs import column_expressions, evaluate
    from turbotab.core.stages.modeling import row_ids_of
    from turbotab.core.stages.working import ROW_MAP, TABLE
    from turbotab.core.voice import listing

    state = fold(service.log(pid).records())
    if state.purpose != "prediction":
        raise ScoreRefused(f"The plan is declared for {state.purpose or 'no purpose'}: its result is "
                           f"an estimate, not a prediction on new rows.")
    if state.orientation == "feature_major":
        raise ScoreRefused("The table is feature-major: new rows cannot be read the way its "
                           "turned working table was.")
    updating = getattr(state.updating, "method", None)
    if updating not in (None, "none"):
        raise ScoreRefused(f"The plan updates its model ({updating}) after the fit, which these "
                           f"predictions do not apply.")
    statuses = service.engine.status(pid)
    fit = _bundle(service, pid, "fit", statuses)
    design = _bundle(service, pid, "design", statuses)
    working = _bundle(service, pid, "working", statuses)
    task = str(fit.data.get("task"))
    if task not in TASKS:
        raise ScoreRefused(f"A {task} outcome is not scored here: only a yes/no outcome or a "
                           f"number.")
    fitted = dict((fit.objects or {}).get("fitted") or {})
    if family is None:
        if len(fitted) != 1:
            raise ScoreRefused(f"The fit kept {len(fitted)} families ({', '.join(sorted(fitted))}): "
                               f"name the one to score.")
        family = next(iter(fitted))
    if family not in fitted:
        raise ScoreRefused(f"The fit kept no `{family}` model (it kept "
                           f"{', '.join(sorted(fitted)) or 'none'}).")
    spec = DesignSpec.from_dict(design.objects["spec"])
    inputs = list(spec.inputs)
    files = getattr(working, "files", None) or {}
    if ROW_MAP in files:
        if (working.data or {}).get("aggregation"):
            raise ScoreRefused("The working table combines rows: a new row is not one of its "
                               "rows.")
        raise ScoreRefused("The working table leaves reference rows out and numbers the rest "
                           "again: its rows are not the table's own, so the steps a new row would "
                           "go through cannot be checked on the training rows.")
    findings = service.engine.get(pid, "findings").artifact
    sql = column_expressions(findings, state)
    raw = service.workspace.table_path(pid, state)
    train = row_ids_of(design.frames["training"])
    # Right or refused: these steps must give the working table the fit read, on its own rows.
    mine = normalize_frame(evaluate(raw, inputs, sql, row_ids=train))
    store = DataStore(Path(files[TABLE]), int(service.settings.memory_budget_bytes))
    theirs = normalize_frame(store.materialize(inputs, train))
    differ = [c for c in inputs if not _same(mine[c].reset_index(drop=True),
                                             theirs[c].reset_index(drop=True))]
    if differ:
        raise ScoreRefused(f"On the training rows the working table holds "
                           f"{', '.join(f'`{c}`' for c in differ)} otherwise than the repairs' own "
                           f"steps compute it from the table: new rows cannot be read the same "
                           f"way.")
    with tempfile.TemporaryDirectory(prefix="turbotab-score-") as folder:
        parquet = _ingested(rows, Path(folder))
        given, table = set(_columns(parquet)), set(_columns(raw))
        lacking = [c for c in inputs if c in table and c not in given]
        if lacking:
            raise ScoreRefused(f"The new rows cannot be read as the plan's predictors: they lack "
                               f"{listing(lacking)}.")
        try:
            new = normalize_frame(evaluate(parquet, inputs, sql))
        except Exception as exc:  # noqa: BLE001 - another column the steps read, said
            raise ScoreRefused(f"The new rows cannot be read as the plan's predictors: {exc}") \
                from None
    kinds = [c for c in inputs if _kind(new[c]) != _kind(theirs[c])]
    if kinds:
        raise ScoreRefused(f"In the new rows {', '.join(f'`{c}`' for c in kinds)} read as "
                           f"{'text' if _kind(new[kinds[0]]) == 'text' else 'numbers'}, where the "
                           f"plan read them otherwise.")
    model = fitted[family]
    X = new[inputs]
    if task == "regression":
        values = predict_with(task, model, X)
    else:
        proba = predict_with(task, model, X)
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 not in classes:
            raise ScoreRefused(f"The `{family}` model holds no event class (its classes are "
                               f"{classes}).")
        values = proba[:, classes.index(1)]
    return Scored(family=family, task=task, predictions=np.asarray(values, dtype=float),
                  inputs=inputs, event=state.event, checked_rows=int(len(train)))


def predict_in(home: str | Path, pid: str, rows: pd.DataFrame | str | Path, *,
               family: str | None = None, workers: int | None = None,
               timeout: float = 3600.0) -> Scored:
    """:func:`predict` on a project in a TurboTab home (a replay's kept home), opening TurboTab there
    as a replay does and waiting for its fit."""
    import dataclasses

    from turbotab.core.config import Settings
    from turbotab.core.export.replay import _wait
    from turbotab.server.service import ProjectService

    settings = dataclasses.replace(Settings.from_env(), home=Path(home), mode="local",
                                   **({"workers": workers} if workers else {}))
    service = ProjectService(settings, replay=True)
    try:
        service.engine.on_decision(pid)
        _wait(service, pid, ("design", "fit"), timeout)
        return predict(service, pid, rows, family=family)
    finally:
        service.close()


def brier_regret(predicted: Sequence[float], risk: Sequence[float]) -> float:
    """The mean of (p̂ − p)² over the rows: for a yes/no outcome of risk p, E[(p̂ − y)²] − E[(p − y)²]
    = E[(p̂ − p)²], the Brier score's excess over the oracle's (SWARM_PLAN §7's regret)."""
    p_hat, p = np.asarray(predicted, dtype=float), np.asarray(risk, dtype=float)
    if p_hat.shape != p.shape:
        raise ValueError(f"{p_hat.shape[0]} predictions for {p.shape[0]} risks")
    return float(np.mean((p_hat - p) ** 2))


__all__ = ["ScoreRefused", "Scored", "brier_regret", "predict", "predict_in"]
