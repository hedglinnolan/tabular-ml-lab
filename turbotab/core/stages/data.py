"""The ``ingest`` and ``profile`` stages."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from turbotab.core.graph import StageContext
from turbotab.core.shape_memo import remembered

LENSES = ("metabolomics", "genomics", "dietary", "clinical", "survey")
HINT_SAMPLE_ROWS = 5_000


def open_store(ctx: StageContext) -> Any:
    """The DataStore over the table this stage reads, under the memory budget the server passed down.

    The working table for a stage that depends on ``working``, the oriented table for one that
    depends on ``oriented``, else the raw file (M2_CONTRACT §2; ``stages.working.table_path``).
    """
    from turbotab.core.config import default_memory_budget
    from turbotab.core.datastore import DataStore
    from turbotab.core.stages.working import table_path

    budget = ctx.settings.get("memory_budget_bytes") or default_memory_budget()
    return DataStore(table_path(ctx), int(budget))


def gigabytes(n: int) -> str:
    value = n / 1e9
    return f"{value:.1f}" if value >= 1 else f"{value:.2f}"


def ingest_stage(ctx: StageContext) -> dict[str, Any]:
    """Read the source file once into ``data/raw.parquet``; the artifact is DatasetInfo. The files
    joined to it (``joins``, DATAIN) are joined in answer order on their identifiers, each read
    from its own ingested copy (``files/<id>/raw.parquet``)."""
    from turbotab.core.assembly import file_parquet
    from turbotab.core.datastore import JoinInput, ingest

    joins = []
    for spec in (getattr(ctx.state, "joins", None) or {}).values():
        joins.append(JoinInput(parquet=file_parquet(Path(ctx.paths["project_dir"]), spec.file),
                               name=spec.name, on=spec.on, right_on=spec.right_on, how=spec.how))
    info = ingest(Path(ctx.paths["source"]), Path(ctx.paths["data"]), progress=ctx.progress,
                  joins=joins)
    return info.to_dict()


@remembered()  # packs.suggest reads each shape twice; once is enough (wide data)
def profile_stage(ctx: StageContext) -> dict[str, Any]:
    """Column summaries over every row of the oriented table, and the packs' lens hints.

    The oriented table rather than the working one: the lens hints are what the first question
    offers, and the working table waits for the lens (through ``findings``).

    The hints need a pandas frame. The whole table is read when it fits the
    memory budget; otherwise a fixed 5,000-row sample is read instead, and the
    ``basis`` says so. A hint is a suggestion that is never pre-selected, so a
    sample is an acceptable basis for one. Findings never use a sample.
    """
    from turbotab.core.datastore import MemoryBudgetExceeded

    with open_store(ctx) as store:
        ctx.progress(0.02, "Summarizing columns")
        columns = store.summaries()
        n_rows = store.n_rows
        ctx.progress(0.7, "Reading the table for lens hints")
        try:
            frame = store.materialize()
            basis = f"Column summaries and lens hints read all {n_rows:,} rows."
        except MemoryBudgetExceeded as exc:
            why = (
                f"the whole table needs about {gigabytes(exc.estimate_bytes)} GB and "
                f"TurboTab's memory budget on this machine is {gigabytes(exc.budget_bytes)} GB"
            )
            rows = min(HINT_SAMPLE_ROWS, n_rows)
            while rows > 0 and store.estimate_bytes(n_rows=rows) > store.memory_budget_bytes:
                rows //= 2  # a very wide table: fewer rows, still a fixed sample
            if rows:
                frame = store.sample(rows, seed=0)
                basis = (
                    f"Column summaries read all {n_rows:,} rows. Lens hints read a fixed "
                    f"sample of {rows:,} rows, because {why}."
                )
            else:
                frame = None
                basis = (
                    f"Column summaries read all {n_rows:,} rows. There are no lens hints, "
                    f"because {why}, and not even one row fits."
                )
    ctx.progress(0.85, "Looking for lens hints")
    # WP14 (audit IN-13): each hint needs positive evidence, survey before width
    # (turbotab.core.detectors.lenses, in place of packs.suggest).
    from turbotab.core.detectors import lenses

    hints = lenses.hints(frame.reset_index(drop=True)) if frame is not None else []
    lens_hints = [
        {"lens": str(h["lens"]), "because": str(h["because"])}
        for h in hints
        if h.get("lens") in LENSES
    ]
    return {"columns": columns, "lens_hints": lens_hints, "basis": basis}
