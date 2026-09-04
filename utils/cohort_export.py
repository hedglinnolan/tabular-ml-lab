"""Every group's results in one bundle — not whichever one you finished on.

The export used to describe the active cohort and nothing else. Run the women,
run the men, download: the zip held the men's models, the men's metrics and a
`report.md` whose "Rows: 319" was the men's N under the study's heading. The
women's run was not in it, was not mentioned in it, and by then no longer
existed anywhere.

Now the active branch keeps the full bundle it always had, and every OTHER
banked branch gets a `cohorts/<column>=<label>/` tree beside it, plus a
top-level comparison table and — the part that matters most for a manuscript —
the multiplicity caveats, which until now existed only as an `st.warning` on
page 06 and reached no artifact at all.

**Everything here is built from a `Snapshot`, and nothing here imports
Streamlit.** That is the whole design. Writing another branch's artifacts by
swapping it into the live keys and reading them back would mean the export
mutates the analysis it is exporting — and a failure halfway through would
leave the researcher in a cohort they did not choose. These functions take a
dict and return bytes; they could not touch session state if they tried, which
is a stronger guarantee than a rule saying they must not.

What is deliberately NOT here: `manuscript.tex`. It stays single-cohort with a
pointer to this tree. A two-column Results table means rewriting the Methods
around a design the app does not yet support — the cohorts are chosen one at a
time, not declared up front — and a manuscript that presented them as a planned
comparison would be making a claim about the study design that is not true.
"""
from __future__ import annotations

import io
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd


def branch_dir(key: Tuple[str, str]) -> str:
    """`cohorts/sex=Female/`. The key IS the directory name, so a reader can
    match a folder to a row of the comparison table without a lookup."""
    column, label = key
    if not column and not label:
        return "cohorts/everyone"
    safe = f"{column}={label}".replace("/", "_").replace("\\", "_")
    return f"cohorts/{safe}"


def same_partition(key: Tuple[str, str], active: Tuple[str, str]) -> bool:
    """Whether two branch keys belong to the same split of the study.

    The whole-study branch belongs to every partition — it is the population
    the others were carved out of — so it is never excluded by this.
    """
    if not key[0] or not active[0]:
        return True
    return key[0] == active[0]


def _metrics_rows(model_results: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(model_results, dict):
        return rows
    for name, results in model_results.items():
        if not isinstance(results, dict):
            continue
        row: Dict[str, Any] = {"Model": str(name).upper()}
        metrics = results.get("metrics")
        if isinstance(metrics, dict):
            row.update(metrics)
        rows.append(row)
    return rows


def branch_metrics_csv(snap: Any) -> Optional[str]:
    """This branch's per-model metrics, in the same shape as the top-level
    `metrics.csv` so the two can be concatenated."""
    rows = _metrics_rows(getattr(snap, "keys", {}).get("model_results"))
    if not rows:
        return None
    return pd.DataFrame(rows).to_csv(index=False)


def branch_predictions_csv(snap: Any, model_key: str) -> Optional[str]:
    """Held-out actuals and predictions for one model of one branch.

    Built from what the run recorded, not by re-scoring: re-running a model
    against the sealed rows to produce an export would be another opening of
    the test set, uncounted, in a code path nobody thinks of as an evaluation.
    """
    results = (getattr(snap, "keys", {}).get("model_results") or {}).get(model_key)
    if not isinstance(results, dict):
        return None
    actual, predicted = results.get("y_test"), results.get("y_test_pred")
    if actual is None or predicted is None:
        return None
    try:
        return pd.DataFrame({"Actual": actual, "Predicted": predicted}).to_csv(index=False)
    except Exception:
        return None


def branch_manifest(key: Tuple[str, str], snap: Any,
                    seal_opens: Optional[int] = None) -> Dict[str, Any]:
    """What a reader needs to interpret this folder without the app.

    The row counts are this branch's, the seal count is this branch's slice, and
    `constant_in_this_group` names the predictors that carry no information
    inside it — which is the thing a reader comparing two folders will otherwise
    conclude is a real difference between the groups.
    """
    keys = getattr(snap, "keys", {}) or {}
    run = getattr(snap, "run", None) or {}

    def _len(name: str) -> Optional[int]:
        value = keys.get(name)
        try:
            return int(len(value)) if value is not None else None
        except TypeError:
            return None

    whole = key == ("", "")
    return {
        "cohort_column": key[0] or None,
        "cohort_label": key[1] or None,
        "is_whole_study": whole,
        "n_rows_in_group": run.get("n_rows"),
        "n_rows_in_study": run.get("n_total"),
        "n_train": _len("y_train"),
        "n_test": _len("y_test"),
        "models": sorted((keys.get("model_results") or {}).keys())
        if isinstance(keys.get("model_results"), dict) else [],
        "constant_in_this_group": list(run.get("dropped_features") or []),
        "held_out_slice_opened": seal_opens,
        "note": (
            # The whole-study folder is the one place the disjointness claim is
            # false: its held-out set is every sealed row, so it CONTAINS each
            # group's slice rather than sitting beside it. A reader pooling the
            # folders on the strength of that sentence would double-count.
            "Metrics here were computed on the WHOLE study's held-out set. "
            "Every group folder beside this one scores a subset of these same "
            "rows, so this folder is not disjoint from them — it contains them."
            if whole else
            "Metrics here were computed on this group's slice of the one "
            "held-out set drawn on the whole study before it was split. The "
            "group folders are disjoint: no row is scored in two of them."
        ),
    }


def comparison_table(runs: Sequence[Any]) -> pd.DataFrame:
    """One row per banked run — the table page 06 draws on screen and, until
    now, nowhere else. A comparison a researcher can see but not export is a
    comparison they will retype."""
    rows: List[Dict[str, Any]] = []
    metric_keys: List[str] = []
    for r in runs:
        for k in (getattr(r, "metrics", None) or {}):
            if k not in metric_keys:
                metric_keys.append(k)
    for r in runs:
        row: Dict[str, Any] = {
            "Group": getattr(r, "label", ""),
            "Trained on": getattr(r, "n_train", None),
            "Held out": getattr(r, "n_test", None),
        }
        for k in metric_keys:
            row[k] = (getattr(r, "metrics", None) or {}).get(k)
        flat = list(getattr(r, "dropped_features", None) or [])
        row["Constant in this group"] = ", ".join(flat) if flat else ""
        row["Held-out slice opened"] = getattr(r, "seal_opens", None)
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_csv(runs: Sequence[Any]) -> Optional[str]:
    if len(runs) < 2:
        return None
    return comparison_table(runs).to_csv(index=False)


def cohort_report_section(runs: Sequence[Any], caveats: Sequence[str],
                          bundled: Sequence[Tuple[str, str]]) -> List[str]:
    """The "Cohort analyzes" section of `report.md`.

    The caveats are the point. `cohorts.comparison_caveats` says what NOT to
    conclude from two AUCs side by side — different training sizes, different
    outcome rates, the multiplicity of fitting in k groups, and the fact that
    separate fits cannot test whether the difference is real. All four existed
    only as an `st.warning` on a page the reader of the export never saw.
    """
    if not runs:
        return []
    lines: List[str] = ["## Cohort analyzes", ""]
    lines.append(
        f"This study was analyzed separately in {len(runs)} groups. "
        f"**Report all {len(runs)}, not the one that worked.**")
    lines.append("")
    table = comparison_table(runs)
    lines.append(table.to_markdown(index=False))
    lines.append("")
    if bundled:
        lines.append("Full artifacts for each group are in this bundle:")
        lines.append("")
        for label, path in bundled:
            lines.append(f"- **{label}** — `{path}/`")
        lines.append("")
    if caveats:
        lines.append("### What these numbers cannot be read as")
        lines.append("")
        for c in caveats:
            lines.append(f"- {c}")
        lines.append("")
    return lines


def add_cohort_bundles(
    zip_file: Any,
    archive: Dict[Tuple[str, str], Any],
    active_key: Tuple[str, str],
    *,
    model_dumper: Optional[Callable[[Any, str], Optional[bytes]]] = None,
    include_models: bool = True,
    include_predictions: bool = True,
    seal_opens: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, str]]:
    """Write `cohorts/<column>=<label>/` for every branch except the active one.

    The active branch already IS the bundle — the report, the metadata, the
    plots and the manuscript at the top level all describe it — so duplicating
    it under `cohorts/` would put the same numbers in two places and invite a
    reader to treat one of them as a second study.

    `model_dumper` is page 10's own `export_model_artifact`, passed in rather
    than reimplemented: it knows that a neural network exports its
    sklearn-compatible wrapper rather than itself, and a second copy of that
    rule here would be the version that goes stale.

    Returns `(label, path)` for each branch written, so `report.md` can point
    at the folders that actually exist rather than the ones that should.
    """
    written: List[Tuple[str, str]] = []
    for key, snap in archive.items():
        if key == active_key:
            continue
        if not same_partition(key, active_key):
            # A branch from a DIFFERENT grouping variable. Splitting by sex,
            # then splitting by smoking status, leaves both in the archive —
            # and they are overlapping row sets whose counts double-count the
            # same people. `completed_runs()` already scopes the comparison
            # table to one column; the folders beside it have to agree, or the
            # bundle lists three "groups" of a two-group study.
            continue
        base = branch_dir(key)
        label = "Everyone" if key == ("", "") else f"{key[0]} = {key[1]}"
        wrote_anything = False

        metrics = branch_metrics_csv(snap)
        if metrics:
            zip_file.writestr(f"{base}/metrics.csv", metrics)
            wrote_anything = True

        keys = getattr(snap, "keys", {}) or {}
        if include_predictions:
            for model_key in (keys.get("model_results") or {}):
                csv = branch_predictions_csv(snap, model_key)
                if csv:
                    zip_file.writestr(
                        f"{base}/predictions/{model_key}_predictions.csv", csv)
                    wrote_anything = True

        if include_models and model_dumper is not None:
            for model_key, wrapper in (keys.get("trained_models") or {}).items():
                try:
                    blob = model_dumper(wrapper, model_key)
                except Exception:
                    blob = None
                if blob:
                    zip_file.writestr(f"{base}/models/{model_key}_model.joblib", blob)
                    wrote_anything = True

        for model_key, model_pipeline in (keys.get("preprocessing_pipelines_by_model") or {}).items():
            try:
                import joblib
                buf = io.BytesIO()
                joblib.dump(model_pipeline, buf)
            except Exception:
                continue
            zip_file.writestr(
                f"{base}/preprocessing/{model_key}_pipeline.joblib", buf.getvalue())
            wrote_anything = True

        # The manifest is written even for a branch that produced nothing, so a
        # group that was entered and abandoned appears in the bundle as an
        # empty folder with a reason rather than as an absence a reader would
        # read as "not analyzed".
        tag = "everyone" if key == ("", "") else f"{key[0]}={key[1]}"
        manifest = branch_manifest(key, snap, (seal_opens or {}).get(tag))
        if not wrote_anything:
            manifest["note"] = (
                "This group was opened but produced no fitted models. It is "
                "listed so that its absence from the results is visible.")
        zip_file.writestr(f"{base}/manifest.json",
                          json.dumps(manifest, indent=2, default=str))
        written.append((label, base))
    return written
