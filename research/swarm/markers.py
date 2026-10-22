"""The markers the engine reads (SWARM_PLAN §4, "Markers in the pilot"), from what the server serves:
the vocabulary's live markers as they stand on a project, and the record's two post hoc ones.

:func:`live` reads every live marker of the vocabulary (``vocabulary.LIVE``) from the project's view
(``GET /projects/{pid}``: its answers and its stages' statuses) and the routes the analyst's own
commands read: the stages that are no estimate's (the ingest, the target's reading, the
participant flow, the structure, the seal plan, the shelf, the findings), each only when it is
fresh at its current key, so every marker is the one the current answers give; the readings
(``GET …/readings``); and the fit records (``GET …/fits``). No estimate stage is read
(``plan_lock.ESTIMATE_STAGES``): reading one would mark an estimate shown. Each marker holds its
value with where it was served (``basis``), or why it has none; it never raises.

The driver reads them at the commands that read the run's results (``fit``, ``results``, ``seal``,
``export``) and keeps them by the current fit's key (``markers.jsonl``), with the facts of the
split the fit is scored on (:func:`split_now`), so the record's attempts hold the live markers at
each fit and its signed change reads each fit's folds; the assembler reads them once more at the
end for the record's regime, beside the two post hoc markers (:func:`post_hoc`).

**What each live marker is, here** (its basis says the same):

* ``rows``: the ingest's rows; ``rows_analyzed``: the participant flow's final count;
* ``repeated_units``: the grain and repeats answers with the units the held-out draw groups by
  (the seal plan's basis);
* ``effective_size``: under a yes/no or a multi-level outcome, the smallest outcome class's rows
  in the outcome's own value counts (every row of the table); under a numeric outcome, the rows
  analyzed; under time to event, none (no stage the driver reads counts its events);
* ``events_per_parameter``: the effective size over the shelf's candidate predictor parameters;
* ``predictors_by_role``: the roles answer, counted by role; ``predictors_per_row``: the
  participant flow's predictors over its rows;
* ``outcome_type``: the task as answered, else as read; ``event_share``: the event level's rows
  over every row with the outcome, a yes/no outcome only;
* ``missingness``: each column's blanks over the table's rows; ``complete_case_share``: the rows
  complete cases kept over the rows before them, under complete cases only;
* ``design``: the survey, clusters, grain, repeats, unit and design answers; ``time_order``: the
  temporal answer, the structure's time column and whether the seal plan's folds follow time;
* ``units``: each unit read from the values (the readings) or recorded; ``detection_limit_share``:
  the columns the findings found values below a detection limit in (their shares are not served);
  ``batch_columns``: the columns whose names read as a batch, a run or a plate, with the batch
  answer; ``run_order``: the columns whose names read as a run order; ``imputed_copies`` and
  ``repeated_measurement``: the repeats, grain and aggregation answers; ``omics_scale``: the
  findings' scale reading under an assay lens;
* ``purpose``, ``intended_use``, ``thresholds``, ``estimand``, ``multiplicity_family``: the answers;
  ``primary_score``: the primary metric the newest fit record names.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from research.swarm import vocabulary as V

Read = Callable[[str], Any]  # path under /projects/{pid}/ -> the JSON it serves, or None
ASSAY_LENSES = ("metabolomics", "genomics", "proteomics")


def marker(name: str, value: Any = None, basis: str | None = None,
           why: str | None = None) -> dict[str, Any]:
    """A marker as the record holds it (``record.Marker``): its axis and timing the vocabulary's."""
    fixed = V.BY_NAME[name]
    if value is None and why is None:
        why = "The engine served no value for it."
    return {"name": name, "axis": fixed.axis, "timing": fixed.timing, "value": value,
            "basis": basis if value is not None else None, "why": None if value is not None else why}


class _Reads:
    """The routes a marker reads, each once: a stage's artifact only when it is fresh at its
    current key."""

    def __init__(self, view: Mapping[str, Any], read: Read):
        self.view, self._read, self._got = view, read, {}

    def get(self, path: str) -> Any:
        if path not in self._got:
            try:
                self._got[path] = self._read(path)
            except Exception:  # noqa: BLE001 - a route that does not answer serves nothing here
                self._got[path] = None
        return self._got[path]

    def fresh(self, stage: str) -> Any:
        body = self.get(f"stages/{stage}")
        if not isinstance(body, Mapping) or not body.get("fresh"):
            return None
        artifact = body.get("artifact")
        return artifact if artifact is not None else None

    @property
    def state(self) -> Mapping[str, Any]:
        return self.view.get("state") or {}


def _not_fresh(stage: str) -> str:
    return f"The {stage} stage is not computed for the current answers."


def _task(r: _Reads) -> str | None:
    info = r.fresh("target_info") or {}
    return r.state.get("task") or (info.get("task") if info.get("column") == r.state.get("target")
                                   else None)


def _classes(r: _Reads) -> list[dict[str, Any]] | None:
    info = r.fresh("target_info")
    if not isinstance(info, Mapping) or info.get("column") != r.state.get("target"):
        return None
    classes = [c for c in info.get("classes") or [] if isinstance(c, Mapping)]
    return classes or None


def _effective(r: _Reads) -> tuple[Any, str | None, str | None]:
    task = _task(r)
    if task is None:
        return None, None, "No outcome's type is read yet."
    if task in ("binary", "multiclass", "ordinal"):
        classes = _classes(r)
        if not classes:
            return None, None, "The outcome's value counts are not served."
        return (min(int(c.get("count") or 0) for c in classes),
                "the smallest outcome class's rows in the outcome's own value counts (every row "
                "of the table)", None)
    if task == "regression":
        cohort = r.fresh("cohort")
        if not isinstance(cohort, Mapping) or cohort.get("n_final") is None:
            return None, None, _not_fresh("cohort")
        return int(cohort["n_final"]), "the rows analyzed (the participant flow)", None
    return None, None, (f"No stage the driver reads counts the events of a {task} outcome.")


def _columns(r: _Reads) -> list[str] | None:
    ingest = r.fresh("ingest")
    if not isinstance(ingest, Mapping):
        return None
    return [str(c.get("name")) for c in ingest.get("columns") or [] if isinstance(c, Mapping)]


def _rows(r: _Reads) -> dict[str, Any]:
    ingest = r.fresh("ingest")
    if not isinstance(ingest, Mapping) or ingest.get("n_rows") is None:
        return marker("rows", why=_not_fresh("ingest"))
    return marker("rows", int(ingest["n_rows"]), "the table's rows (the ingest)")


def _rows_analyzed(r: _Reads) -> dict[str, Any]:
    cohort = r.fresh("cohort")
    if not isinstance(cohort, Mapping) or cohort.get("n_final") is None:
        return marker("rows_analyzed", why=_not_fresh("cohort"))
    return marker("rows_analyzed", int(cohort["n_final"]), "the participant flow's final count")


def _repeated_units(r: _Reads) -> dict[str, Any]:
    plan = r.fresh("seal_plan")
    if not isinstance(plan, Mapping):
        return marker("repeated_units", why=_not_fresh("seal_plan"))
    basis = plan.get("basis") or {}
    grain = (r.state.get("grain") or {}).get("grain")
    repeats = (r.state.get("repeat_kind") or {}).get("repeat_kind")
    return marker("repeated_units", {"grain": grain, "repeat_kind": repeats,
                                     "unit_column": basis.get("column"),
                                     "units": basis.get("n_units")},
                  "the grain and repeats answers, and the units the held-out draw groups by (the "
                  "seal plan)")


def _effective_size(r: _Reads) -> dict[str, Any]:
    value, basis, why = _effective(r)
    return marker("effective_size", value, basis, why)


def _events_per_parameter(r: _Reads) -> dict[str, Any]:
    value, _, why = _effective(r)
    if value is None:
        return marker("events_per_parameter", why=why)
    shelf = r.fresh("shelf")
    if not isinstance(shelf, Mapping):
        return marker("events_per_parameter", why=_not_fresh("shelf"))
    parameters = (shelf.get("sample_size") or {}).get("parameters")
    if not parameters:
        return marker("events_per_parameter", why="The shelf serves no count of candidate "
                                                  "predictor parameters.")
    return marker("events_per_parameter", value / int(parameters),
                  "the effective size over the shelf's candidate predictor parameters")


def _predictors_by_role(r: _Reads) -> dict[str, Any]:
    roles = r.state.get("roles")
    if not isinstance(roles, Mapping):
        return marker("predictors_by_role", why="No roles are answered yet.")
    target = r.state.get("target")
    counts: dict[str, int] = {}
    for column, role in roles.items():
        if column != target:
            counts[str(role)] = counts.get(str(role), 0) + 1
    return marker("predictors_by_role", dict(sorted(counts.items())),
                  "the roles answer, counted by role")


def _predictors_per_row(r: _Reads) -> dict[str, Any]:
    cohort = r.fresh("cohort")
    if not isinstance(cohort, Mapping) or not cohort.get("n_final"):
        return marker("predictors_per_row", why=_not_fresh("cohort"))
    predictors = cohort.get("predictors")
    if not isinstance(predictors, list):
        return marker("predictors_per_row", why="The participant flow names no predictors.")
    return marker("predictors_per_row", len(predictors) / int(cohort["n_final"]),
                  "the participant flow's predictors over its rows")


def _outcome_type(r: _Reads) -> dict[str, Any]:
    task = _task(r)
    if task is None:
        return marker("outcome_type", why="No outcome's type is read yet.")
    how = "as answered" if r.state.get("task") else "as read from its values"
    return marker("outcome_type", task, f"the outcome's type, {how}")


def _event_share(r: _Reads) -> dict[str, Any]:
    if _task(r) != "binary":
        return marker("event_share", why="Only a yes/no outcome has an event share.")
    event = r.state.get("event")
    classes = _classes(r)
    if event is None or not classes:
        return marker("event_share", why="The event level, or the outcome's value counts, are "
                                         "not served yet.")
    total = sum(int(c.get("count") or 0) for c in classes)
    hit = next((int(c.get("count") or 0) for c in classes if str(c.get("value")) == str(event)),
               None)
    if hit is None or not total:
        return marker("event_share", why="The event level is not among the outcome's values.")
    return marker("event_share", hit / total, "the event level's rows over every row with the "
                                              "outcome (its own value counts)")


def _missingness(r: _Reads) -> dict[str, Any]:
    ingest = r.fresh("ingest")
    if not isinstance(ingest, Mapping) or not ingest.get("n_rows"):
        return marker("missingness", why=_not_fresh("ingest"))
    n = int(ingest["n_rows"])
    shares = {str(c["name"]): int(c["n_missing"]) / n for c in ingest.get("columns") or []
              if isinstance(c, Mapping) and c.get("n_missing")}
    return marker("missingness", shares, "each column's blanks over the table's rows (the "
                                         "ingest); a column with none is not listed")


def _complete_case_share(r: _Reads) -> dict[str, Any]:
    missing = r.state.get("missing") or {}
    if missing.get("strategy") != "complete_case":
        return marker("complete_case_share", why="The participant flow counts the complete cases "
                                                 "only under complete cases.")
    cohort = r.fresh("cohort")
    if not isinstance(cohort, Mapping):
        return marker("complete_case_share", why=_not_fresh("cohort"))
    loss = cohort.get("complete_case_loss")
    if loss is None:
        return marker("complete_case_share", 1.0, "complete cases left out no row (the "
                                                  "participant flow)")
    before, kept = loss.get("n_before"), loss.get("n_kept")
    if not before or kept is None:
        return marker("complete_case_share", why="The participant flow serves no counts of the "
                                                 "rows complete cases kept.")
    return marker("complete_case_share", int(kept) / int(before),
                  "the rows complete cases kept over the rows before them (the participant flow)")


def _design(r: _Reads) -> dict[str, Any]:
    keys = ("survey", "clusters", "grain", "repeat_kind", "unit", "design")
    return marker("design", {k: r.state.get(k) for k in keys}, "the design answers as recorded")


def _time_order(r: _Reads) -> dict[str, Any]:
    structure = r.fresh("structure") or {}
    plan = r.fresh("seal_plan") or {}
    return marker("time_order", {"temporal": r.state.get("temporal"),
                                 "time_column": structure.get("time_column"),
                                 "time_ordered_folds": plan.get("time_ordered_folds")},
                  "the temporal answer, the structure's time column and whether the seal plan's "
                  "folds follow time (null where either stage is not computed)")


UNIT_READINGS = ("unit", "outcome_unit", "day_count")


def _units(r: _Reads) -> dict[str, Any]:
    readings = r.get("readings")
    if not isinstance(readings, Mapping):
        return marker("units", why="The readings are not served.")
    read = [{"column": x.get("column"), "kind": x.get("kind"), "value": x.get("value"),
             "basis": x.get("evidence")} for x in readings.get("read_from_data") or []
            if isinstance(x, Mapping) and x.get("kind") in UNIT_READINGS]
    recorded = [{"column": c, "kind": "unit", "value": (u or {}).get("unit"),
                 "basis": "recorded"} for c, u in (r.state.get("column_units") or {}).items()]
    outcome = r.state.get("outcome_unit")
    if outcome:
        recorded.append({"column": r.state.get("target"), "kind": "outcome_unit",
                         "value": outcome.get("unit") if isinstance(outcome, Mapping) else outcome,
                         "basis": "recorded"})
    return marker("units", read + recorded, "each unit read from the values with its evidence "
                                            "(the readings), and each recorded")


def _findings(r: _Reads) -> list[Mapping[str, Any]] | None:
    found = r.fresh("findings")
    if not isinstance(found, Mapping):
        return None
    return [f for f in found.get("findings") or [] if isinstance(f, Mapping)]


def _detection_limit_share(r: _Reads) -> dict[str, Any]:
    findings = _findings(r)
    if findings is None:
        return marker("detection_limit_share", why=_not_fresh("findings"))
    prefix = "below_detection__"
    columns = [str(f["id"])[len(prefix):] for f in findings if str(f.get("id", "")).startswith(prefix)]
    return marker("detection_limit_share", {"columns": columns, "shares": None},
                  "the columns the findings found values below a detection limit in; each "
                  "column's share is not served")


def _batch_columns(r: _Reads) -> dict[str, Any]:
    columns = _columns(r)
    if columns is None:
        return marker("batch_columns", why=_not_fresh("ingest"))
    from turbotab.core.methods.batch import batch_columns_of

    return marker("batch_columns", {"read": batch_columns_of(columns, r.state.get("target")),
                                    "answered": r.state.get("batch")},
                  "the columns whose names read as a batch, a run or a plate, and the batch answer")


def _run_order(r: _Reads) -> dict[str, Any]:
    columns = _columns(r)
    if columns is None:
        return marker("run_order", why=_not_fresh("ingest"))
    from turbotab.core.recognizers import acquisition_kind

    return marker("run_order", [c for c in columns if acquisition_kind(c) == "run_order"],
                  "the columns whose names read as a run order")


def _imputed_copies(r: _Reads) -> dict[str, Any]:
    repeats = (r.state.get("repeat_kind") or {}).get("repeat_kind")
    return marker("imputed_copies", repeats == "imputed_copies",
                  "the repeats answer: whether a unit's rows are imputed copies")


def _repeated_measurement(r: _Reads) -> dict[str, Any]:
    grain = (r.state.get("grain") or {}).get("grain")
    return marker("repeated_measurement", {
        "grain": grain, "repeat_kind": (r.state.get("repeat_kind") or {}).get("repeat_kind"),
        "aggregation": (r.state.get("aggregation") or {}).get("method")},
        "the grain, repeats and aggregation answers")


def _omics_scale(r: _Reads) -> dict[str, Any]:
    lenses = list(r.state.get("lens") or [])
    if not any(lens in ASSAY_LENSES for lens in lenses):
        return marker("omics_scale", why="No assay lens is chosen, so no scale is read.")
    findings = _findings(r)
    if findings is None:
        return marker("omics_scale", why=_not_fresh("findings"))
    from turbotab.core.methods.omics import FINDING

    found = next((f for f in findings if f.get("id") == FINDING), None)
    return marker("omics_scale", {"finding": FINDING if found else None,
                                  "title": found.get("title") if found else None},
                  "the findings' reading of raw counts or intensities (none: none read)")


def _purpose(r: _Reads) -> dict[str, Any]:
    purpose = r.state.get("purpose")
    return marker("purpose", purpose, "the purpose answer",
                  None if purpose else "No purpose is answered yet.")


def _primary_score(r: _Reads) -> dict[str, Any]:
    if r.state.get("purpose") == "inference":
        return marker("primary_score", why="Under Estimate the result is the estimate, not a "
                                           "score.")
    fits = r.get("fits")
    records = (fits or {}).get("records") if isinstance(fits, Mapping) else None
    named = [x.get("primary_metric") for x in records or [] if x.get("primary_metric")]
    if not named:
        return marker("primary_score", why="No fit has named its primary score yet.")
    return marker("primary_score", {"metric": named[-1]},
                  "the primary metric the newest fit record names")


def _intended_use(r: _Reads) -> dict[str, Any]:
    use = r.state.get("intended_use") or {}
    if not use:
        return marker("intended_use", why="No intended use is declared.")
    return marker("intended_use", {"use": use.get("use"), "subgroups": use.get("subgroups"),
                                   "fairness": use.get("fairness")}, "the intended-use answer")


def _thresholds(r: _Reads) -> dict[str, Any]:
    use = r.state.get("intended_use") or {}
    if not use:
        return marker("thresholds", why="No intended use is declared.")
    return marker("thresholds", {k: use.get(k) for k in ("threshold_low", "threshold_high",
                                                          "threshold")},
                  "the intended-use answer's thresholds (null: none declared)")


def _estimand(r: _Reads) -> dict[str, Any]:
    estimand = r.state.get("estimand")
    if not estimand:
        return marker("estimand", why="No estimand is declared.")
    return marker("estimand", {k: estimand.get(k) for k in ("exposure", "family", "effect",
                                                             "contrast", "measure")},
                  "the estimand answer")


def _multiplicity_family(r: _Reads) -> dict[str, Any]:
    estimand = r.state.get("estimand")
    if not estimand:
        return marker("multiplicity_family", why="No estimand is declared.")
    return marker("multiplicity_family", {
        "family": bool(estimand.get("family")),
        "method": (r.state.get("multiplicity") or {}).get("method")},
        "the estimand's exposure family and the multiplicity answer (null: none recorded)")


READERS: dict[str, Callable[[_Reads], dict[str, Any]]] = {
    "rows": _rows, "rows_analyzed": _rows_analyzed, "repeated_units": _repeated_units,
    "effective_size": _effective_size, "events_per_parameter": _events_per_parameter,
    "predictors_by_role": _predictors_by_role, "predictors_per_row": _predictors_per_row,
    "outcome_type": _outcome_type, "event_share": _event_share, "missingness": _missingness,
    "complete_case_share": _complete_case_share, "design": _design, "time_order": _time_order,
    "units": _units, "detection_limit_share": _detection_limit_share,
    "batch_columns": _batch_columns, "run_order": _run_order, "imputed_copies": _imputed_copies,
    "repeated_measurement": _repeated_measurement, "omics_scale": _omics_scale,
    "purpose": _purpose, "primary_score": _primary_score, "intended_use": _intended_use,
    "thresholds": _thresholds, "estimand": _estimand, "multiplicity_family": _multiplicity_family,
}


# The live markers read from the answers alone: the record reads them again from the answers a fit
# read, since an answer the fit does not read (the intended use, the multiplicity) can change while
# the fit's key stands.
STATE_ONLY = ("predictors_by_role", "design", "imputed_copies", "repeated_measurement", "purpose",
              "intended_use", "thresholds", "estimand", "multiplicity_family")


def from_answers(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The markers of :data:`STATE_ONLY`, read from ``state`` (the answers, as folded) alone."""
    reads = _Reads({"state": state}, lambda _path: None)
    return [READERS[name](reads) for name in STATE_ONLY]


def live(view: Mapping[str, Any], read: Read) -> list[dict[str, Any]]:
    """Every live marker of the vocabulary, in its order, as the project stands (the module
    docstring). A marker whose reading fails holds why, naming the failure; none raises."""
    reads = _Reads(view, read)
    out = []
    for name in V.LIVE:
        try:
            out.append(READERS[name](reads))
        except Exception as exc:  # noqa: BLE001 - a marker the driver cannot read says so
            out.append(marker(name, why=f"The driver could not read it ({type(exc).__name__})."))
    return out


def split_now(view: Mapping[str, Any], read: Read) -> dict[str, Any] | None:
    """The facts of the folds the current answers draw (``signed.SPLIT_FACTS``: the training
    rows, the folds and their repeats, their scheme, the validation and the grouping), from the
    split stage when it is fresh at its current key, or None: kept beside the markers, so the
    record's signed change reads each fit's share of rows a fold scores (``signed.test_share``)."""
    from research.swarm.signed import split_facts

    try:
        return split_facts(_Reads(view, read).fresh("split"))
    except Exception:  # noqa: BLE001 - a read that fails keeps nothing
        return None


def fit_key(view: Mapping[str, Any]) -> str | None:
    """The current fit's key, which the live markers read now belong to."""
    return ((view.get("stages") or {}).get("fit") or {}).get("key")


def _signal(scale: str, estimate: float, low: Any, high: Any) -> dict[str, Any]:
    """The ``signal`` marker of an estimate (:func:`post_hoc`): its distance from no effect in
    half-widths of its interval, on the log scale for a ratio."""
    import math

    from research.swarm.record import NULL_OF

    if low is None or high is None:
        return marker("signal", why="The estimate has no interval.")
    low, high = float(low), float(high)
    if scale == "ratio":
        if min(estimate, low, high) <= 0:
            return marker("signal", why="A ratio's interval reaches zero, where its log is not "
                                        "defined.")
        away, half = math.log(estimate), (math.log(high) - math.log(low)) / 2
        basis = ("the first estimate's distance from no effect (1) in half-widths of its "
                 "interval, on the log scale")
    else:
        away, half = estimate - NULL_OF.get(scale, 0.0), (high - low) / 2
        basis = "the first estimate's distance from no effect in half-widths of its interval"
    if not half:
        return marker("signal", why="The estimate's interval has no width.")
    return marker("signal", abs(away) / half, basis)


def post_hoc(result: Mapping[str, Any], attempts: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The two post hoc markers from the record's own result and attempts:

    * ``signal``: under Predict, the reported family's cross-validated primary score at the
      result's fit; under Estimate, the first reported estimate's distance from no effect in
      half-widths of its interval, on the scale its interval is symmetric on: |estimate| /
      ((high − low) / 2) on a difference scale, |log estimate| / ((log high − log low) / 2) on a
      ratio scale, so an odds ratio and its reciprocal lie as far from 1 (none where a ratio's
      interval reaches zero, whose log is not defined);
    * ``result``: the reported result in brief, the sealed primary score or the first estimate
      with its interval, or why there is none (the result's own refusal).
    """
    refused = result.get("refused")
    if refused:
        return [marker("signal", why=f"No result was reported: {refused}"),
                marker("result", why=str(refused))]
    if result.get("purpose") == "prediction" or result.get("scores"):
        family, metric = result.get("family"), result.get("metric")
        attempt = next((a for a in attempts if a.get("id") == result.get("attempt")), None)
        cv = (((attempt or {}).get("scores") or {}).get(family) or {}).get(metric) or {}
        signal = (marker("signal", {"family": family, "metric": metric,
                                    "estimate": cv.get("estimate"), "ci_low": cv.get("ci_low"),
                                    "ci_high": cv.get("ci_high")},
                         "the reported family's cross-validated primary score at the result's fit")
                  if cv.get("estimate") is not None else
                  marker("signal", why="The result's fit serves no cross-validated primary score "
                                       "for the reported family."))
        sealed = ((result.get("scores") or {}).get(family) or {}).get(metric)
        return [signal, marker("result", {"family": family, "metric": metric, "held_out": sealed},
                               "the held-out primary score the opening recorded")]
    estimate = next((e for e in result.get("estimates") or [] if e.get("estimate") is not None),
                    None)
    if estimate is None:
        return [marker("signal", why="No estimate was reported."),
                marker("result", why="No estimate was reported.")]
    low, high = estimate.get("ci_low"), estimate.get("ci_high")
    signal = _signal(str(result.get("scale")), float(estimate["estimate"]), low, high)
    return [signal, marker("result", {"family": estimate.get("family"),
                                      "feature": estimate.get("feature"),
                                      "measure": result.get("measure"),
                                      "estimate": estimate.get("estimate"), "ci_low": low,
                                      "ci_high": high}, "the first reported estimate")]


__all__ = ["ASSAY_LENSES", "READERS", "STATE_ONLY", "fit_key", "from_answers", "live", "marker",
           "post_hoc", "split_now"]
