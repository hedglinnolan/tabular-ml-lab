"""Capture real-data fixtures for the M2 design prototype at /lab/m2 (M2_CONTRACT §8, "design").

Every number in the fixture is computed here, from the real files, by the real code: DuckDB ingest
and ``DataStore`` (turbotab.core.datastore), the aggregation as a DuckDB GROUP BY over the ingested
parquet (cross-checked against ``turbotab.repeats.aggregate``), the repeats reading and the
aggregation menu (``turbotab.repeats``), the orientation reading and the transpose
(``turbotab.orientation``), the seal draws (``turbotab.engine.draw_holdout``, the lockbox bases in
``turbotab.grain``), and the fits (the production model families and pipelines in
``turbotab.core.models``). Nothing on the prototype's canvas is typed by hand.

Run from the repository root:

    venv/bin/python docs/turbotab-next/m2/explore/capture.py

Scenarios:

  reshape      dietary_recalls.csv: "combine each person's two recalls", mean / first / last /
               change, on a window of five participants, the whole-table row strip, the column
               strip, the row flow and energy per row before and after (with coach notes)
  wide         a seeded 600 x 20,000 synthetic (300 samples x 2 replicate assays, exported stacked
               by replicate) carried through the same question: the affected columns only
  orientation  the feature-major copy of metabolomics_untargeted.csv (built the way the legacy
               orientation test builds it) turning around
  seal         four seals: grouped (the combined dietary table), chronological grouped
               (clinical_longitudinal.csv), grouping abandoned (its first seven subjects) and
               undetermined (dietary_recalls.csv by row)
  results      linear, elastic net and boosted trees on the combined dietary table: CV on the
               training people, held-out scores computed and withheld until the seal is opened,
               then refit after a post-seal change of the energy method
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from turbotab import engine, grain, orientation, repeats  # noqa: E402
from turbotab.core.datastore import DataStore, ingest  # noqa: E402
from turbotab.core.decisions import EnergyAdjustment, ProjectState  # noqa: E402
from turbotab.core.models import get_family  # noqa: E402
from turbotab.core.models.metrics import score, summarize  # noqa: E402
from turbotab.core.models.pipeline import build_pipeline, design_spec  # noqa: E402

DATA = ROOT / "turbotab/sample_data"
DIETARY = DATA / "dietary_recalls.csv"
CLINICAL = DATA / "clinical_longitudinal.csv"
METABOLOMICS = DATA / "metabolomics_untargeted.csv"
OUT = ROOT / "turbotab/frontend/src/explore/m2/fixture.json"

BUDGET = 4 * 1024**3
WINDOW_UNITS = 4          # participants shown in the reshape table
SHOWN_COLUMNS = 8         # data columns shown beside the identifier (BLUEPRINT §11.3: ≤ 12)
COACH_WORDS = 12          # M2_CONTRACT §6
SEED = 0
HOLDOUT = 0.2
FOLDS = 5
WIDE_UNITS = 300
WIDE_FEATURES = 20_000
WIDE_SEED = 20261001
TURN = 8                  # the orientation window: 8 x 8 cells

warnings.filterwarnings("ignore")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path, workdir: Path) -> tuple[pd.DataFrame, Path]:
    """Ingest the way the app does (DuckDB -> parquet with __row_id) and materialize."""
    parquet = workdir / f"{path.stem}.parquet"
    ingest(path, parquet)
    frame = DataStore(parquet, BUDGET).materialize()
    return frame, parquet


def words(text: str) -> int:
    return len(text.split())


def coach(text: str, anchor: dict[str, Any]) -> dict[str, Any]:
    if words(text) > COACH_WORDS:
        raise SystemExit(f"coach note over {COACH_WORDS} words: {text!r}")
    return {"text": text, "anchor": anchor}


def jsonable(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else round(f, 6)
    if isinstance(v, (pd.Timestamp, dt.date)):
        return str(v)[:10]
    return v


def fmt_int(n: float) -> str:
    return f"{int(round(n)):,}"


# ── reshape: dietary_recalls ─────────────────────────────────────────────────

ID = "participant_id"
ORDER = "recall_date"
INDEX = "recall_number"


def aggregate_sql(parquet: Path, method: str, numeric: list[str], other: list[str],
                  constant: list[str]) -> pd.DataFrame:
    """One row per participant, the way the `working` stage will (M2_CONTRACT §2): a DuckDB GROUP BY.

    mean/change fold the record-level columns away (a combined row is no single record); first
    and last keep a whole record, so its `recall_number` and `recall_date` stay true. Columns that
    are constant within every participant pass through under every method.
    """
    q = lambda c: f'"{c}"'  # noqa: E731
    parts = [f"min(__row_id) AS __first_row"]
    for c in numeric + other:
        if c in constant:
            parts.append(f"any_value({q(c)}) AS {q(c)}")
        elif method == "mean":
            parts.append(f"avg({q(c)}) AS {q(c)}")
        elif method == "first":
            parts.append(f"arg_min({q(c)}, {q(ORDER)}) AS {q(c)}")
        elif method == "last":
            parts.append(f"arg_max({q(c)}, {q(ORDER)}) AS {q(c)}")
        elif method == "change":
            parts.append(f"arg_max({q(c)}, {q(ORDER)}) - arg_min({q(c)}, {q(ORDER)}) AS {q(c)}")
    if method in ("first", "last"):
        pick = "arg_min" if method == "first" else "arg_max"
        parts.append(f"{pick}(__row_id, {q(ORDER)}) AS __kept_row")
    sql = (f"SELECT {q(ID)}, {', '.join(parts)} FROM read_parquet('{parquet}') "
           f"GROUP BY {q(ID)} ORDER BY __first_row")
    return duckdb.sql(sql).df()


def hist(values: np.ndarray, edges: np.ndarray) -> dict[str, Any]:
    v = values[np.isfinite(values)]
    counts, _ = np.histogram(np.clip(v, edges[0], edges[-1] - 1e-9), bins=edges)
    return {"counts": [int(c) for c in counts], "n": int(v.size),
            "mean": round(float(v.mean()), 3), "sd": round(float(v.std(ddof=1)), 3),
            "under": int((v < 500).sum()), "over": int((v > 5000).sum())}


def reshape_scenario(frame: pd.DataFrame, parquet: Path) -> dict[str, Any]:
    df = frame
    n_rows = len(df)
    numeric = [c for c in df.columns if c not in (ID, "__row_id")
               and pd.api.types.is_numeric_dtype(df[c])]
    other = [c for c in df.columns if c not in (ID, "__row_id") and c not in numeric]
    by = df.groupby(ID, sort=False)
    constant = [c for c in numeric + other if int(by[c].nunique(dropna=True).max()) <= 1]
    record = [INDEX, ORDER]
    measures = [c for c in numeric if c not in constant and c not in record]
    per_unit = by.size()
    n_units = int(per_unit.size)
    if int(per_unit.min()) != int(per_unit.max()):
        raise SystemExit("the reshape storyboard expects the same number of recalls per participant")
    k = int(per_unit.iloc[0])

    reading = repeats.read(df.drop(columns="__row_id"), ID)
    menu = repeats.menu(reading["reading"], ["dietary"])

    # Which measures to show: energy first (the pack's column), then by how much a person's two
    # days disagree relative to the column's spread — what the combination changes most.
    disagreement = {}
    for c in measures:
        spread = float(df[c].std(ddof=1)) or 1.0
        disagreement[c] = float(by[c].agg(lambda s: s.max() - s.min()).mean()) / spread
    grams = [c for c in measures if not c.endswith("_pct_kcal")]
    shares = [c for c in measures if c.endswith("_pct_kcal")]
    ranked = ["energy_kcal"] + [c for c in df.columns if c in grams and c != "energy_kcal"]
    shown_measures = ranked[: SHOWN_COLUMNS - len(record)]
    more = [c for c in measures if c not in shown_measures]

    # The window: the first WINDOW_UNITS participants in file order; their rows, wherever they are.
    units = list(dict.fromkeys(df[ID]))[:WINDOW_UNITS]
    win = df[df[ID].isin(units)].sort_values("__row_id")
    rows = []
    for _, r in win.iterrows():
        rows.append({"row": int(r["__row_id"]), "unit": r[ID],
                     "k": int(df[(df[ID] == r[ID]) & (df["__row_id"] < r["__row_id"])].shape[0]),
                     "values": {c: jsonable(r[c]) for c in record + shown_measures}})
    unit_rows = {u: [int(x) for x in win[win[ID] == u]["__row_id"]] for u in units}
    adjacent = all(b - a == 1 for rr in unit_rows.values() for a, b in zip(rr, rr[1:]))

    # The tracked identity (DESIGN_LANGUAGE §05.2: one or two, never a table): the window's
    # participant whose two days disagree most on energy.
    gap = {u: float(win[win[ID] == u]["energy_kcal"].max() - win[win[ID] == u]["energy_kcal"].min())
           for u in units}
    tracked = max(gap, key=gap.get)

    # The whole-table strip: each row's participant (in first-appearance order) and its index.
    order = {u: i for i, u in enumerate(dict.fromkeys(df[ID]))}
    strip_unit = [order[u] for u in df[ID]]
    strip_k = [int(x) for x in by.cumcount()]

    # Energy per row, before and after, on one set of edges.
    edges = np.arange(0, 8000 + 250, 250, dtype=float)
    e_before = df["energy_kcal"].to_numpy(dtype=float)
    within = float(by["energy_kcal"].var(ddof=1).mean())
    total = float(np.var(e_before, ddof=1))
    within_share = within / total

    methods: dict[str, Any] = {}
    reference = repeats.aggregate(df.drop(columns="__row_id"), ID, "mean", target="hba1c",
                                  order_col=ORDER)["frame"].set_index(ID)
    labels = {m["key"]: m["label"] for m in menu["options"]}
    for method, key in (("mean", repeats.MEAN), ("first", repeats.FIRST), ("last", repeats.LAST),
                        ("change", repeats.CHANGE)):
        out = aggregate_sql(parquet, method, numeric, other, constant).set_index(ID)
        if method == "mean":  # the SQL and the domain module agree, to the cent
            for c in measures:
                if not np.allclose(out[c].to_numpy(float), reference.loc[out.index, c].to_numpy(float)):
                    raise SystemExit(f"DuckDB and turbotab.repeats.aggregate disagree on {c}")
        folds = record if method in ("mean", "change") else []
        after_units = {}
        for u in units:
            values = {c: jsonable(out.loc[u, c]) for c in record + shown_measures if c not in folds}
            kept = int(out.loc[u, "__kept_row"]) if "__kept_row" in out.columns else None
            after_units[u] = {"values": values, "kept_row": kept}
        strip_kept = None
        if "__kept_row" in out.columns:
            kept_rows = set(int(x) for x in out["__kept_row"])
            strip_kept = [1 if int(r) in kept_rows else 0 for r in df["__row_id"]]
        energy_after = out["energy_kcal"].to_numpy(dtype=float)
        n_after = int(len(out))
        if method in ("mean", "change"):
            flow = [{"key": "loaded", "label": "Recalls in the table", "n": n_rows, "combined": 0,
                     "dropped": 0},
                    {"key": "combined", "label": f"One row per `{ID}`", "n": n_after,
                     "combined": n_rows - n_after, "dropped": 0,
                     "reason": (f"each person's {k} recalls {'averaged' if method == 'mean' else 'differenced'}"
                                f" into one row")}]
        else:
            flow = [{"key": "loaded", "label": "Recalls in the table", "n": n_rows, "combined": 0,
                     "dropped": 0},
                    {"key": "kept", "label": f"Each person's {method} recall", "n": n_after,
                     "combined": 0, "dropped": n_rows - n_after,
                     "reason": f"the other {n_rows - n_after:,} recalls leave"}]

        b = hist(e_before, edges)
        notes: list[dict[str, Any]] = []
        if method == "change":
            d_edges = np.arange(-6000, 6000 + 250, 250, dtype=float)
            a = hist(energy_after, d_edges)
            med = float(np.median(energy_after))
            signed = f"{'+' if med > 0 else '−' if med < 0 else ''}{abs(med):,.0f}"
            notes.append(coach(f"Second recall minus first: median {signed} kcal",
                               {"kind": "points", "ref": [med]}))
            energy = {"edges": [float(x) for x in d_edges], "after": a,
                      "after_label": "change in energy_kcal (last − first)", "unit_changes": True}
        else:
            a = hist(energy_after, edges)
            if method == "mean":
                notes.append(coach(
                    f"{b['under']} recalls fell under 500 kcal; no person's mean does"
                    if a["under"] == 0 else
                    f"{b['under']} recalls under 500 kcal; {a['under']} person means still are",
                    {"kind": "range", "ref": [0, 500]}))
                notes.append(coach(
                    f"Day-to-day variation was {within_share:.0%} of the variance; averaging halves it",
                    {"kind": "range", "ref": [a["mean"] - a["sd"], a["mean"] + a["sd"]]}))
            else:
                # Under-reports are a property of single days, so the note says where they went.
                notes.append(coach(
                    (f"All {b['under']} recalls under 500 kcal stay, each standing for a person"
                     if a["under"] == b["under"] else
                     f"{a['under']} of {b['under']} recalls under 500 kcal stay, each standing for a person")
                    if a["under"] else
                    f"All {b['under']} recalls under 500 kcal were {'first' if method == 'last' else 'later'} recalls; none stays",
                    {"kind": "range", "ref": [0, 500]}))
                notes.append(coach(
                    f"One day keeps all day-to-day variation: {within_share:.0%} of the variance",
                    {"kind": "range", "ref": [a["mean"] - a["sd"], a["mean"] + a["sd"]]}))
            energy = {"edges": [float(x) for x in edges], "after": a,
                      "after_label": ("person means" if method == "mean" else f"each person's {method} recall"),
                      "unit_changes": False}
        energy["before"] = b
        energy["before_label"] = "single recalls"
        energy["before_edges"] = [float(x) for x in edges]
        menu_entry = next(m for m in menu["options"] if m["key"] == key)
        methods[method] = {
            "key": method,
            "label": labels[key],
            "sentence": menu_entry["sentence"],
            "recommended": menu["recommended"] == key,
            "steps": {
                "mean": [f"Gather rows by `{ID}`", "Average each column"],
                "first": [f"Gather rows by `{ID}`", f"Keep the earliest `{ORDER}`"],
                "last": [f"Gather rows by `{ID}`", f"Keep the latest `{ORDER}`"],
                "change": [f"Gather rows by `{ID}`", "Subtract the first from the last"],
            }[method],
            "folds": folds,
            "units": after_units,
            "n_after": n_after,
            "columns_after": len(df.columns) - 1 - len(folds),  # without __row_id
            **({"strip_kept": strip_kept} if strip_kept is not None else {}),
            "flow": flow,
            "energy": energy,
            "coach": notes,
        }

    return {
        "dataset": {"name": DIETARY.name, "rows": n_rows, "cols": len(df.columns) - 1,
                    "sha256": sha256(DIETARY)},
        "id_column": ID, "order_column": ORDER, "index_column": INDEX,
        "n_units": n_units, "per_unit": k, "noun": "recall", "unit_noun": "person",
        "repeats": {"reading": reading["reading"], "spacing": {
            kk: jsonable(v) for kk, v in (reading["spacing"] or {}).items()},
            "replicate_index": reading["replicate_index"]},
        "menu": {"recommended": menu["recommended"], "reason": menu["reason"],
                 "from_pack": menu.get("from_pack"), "marker": menu["marker"]},
        "columns": {"record": record, "shown": shown_measures, "more": more,
                    "constant": constant, "all": [c for c in df.columns if c != "__row_id"],
                    "kinds": {c: ("id" if c == ID else "record" if c in record else
                                  "constant" if c in constant else "measure")
                              for c in df.columns if c != "__row_id"},
                    "n_changed": len(measures)},
        "window": {"layout": "interleaved" if adjacent else "stacked", "units": units,
                   "rows": rows, "tracked": tracked},
        "strip": {"unit": strip_unit, "k": strip_k},
        "within_share": round(within_share, 4),
        "methods": methods,
    }


# ── wide: 600 x 20,000, stacked by replicate ─────────────────────────────────


def wide_scenario() -> dict[str, Any]:
    rng = np.random.default_rng(WIDE_SEED)
    names = [f"ft_{i:05d}" for i in range(1, WIDE_FEATURES + 1)]
    mu = rng.normal(8.0, 1.6, WIDE_FEATURES)
    noise = rng.uniform(0.05, 0.7, WIDE_FEATURES)
    person = rng.normal(0.0, 0.45, (WIDE_UNITS, WIDE_FEATURES))
    rep = [np.exp(mu + person + rng.normal(0.0, noise, (WIDE_UNITS, WIDE_FEATURES))) for _ in range(2)]
    ids = [f"S{i:03d}" for i in range(1, WIDE_UNITS + 1)]
    # Exported stacked: every sample's first replicate, then every second one (instrument run order).
    x = np.round(np.vstack(rep), 1)
    meta = pd.DataFrame({"sample_id": ids + ids, "replicate": [1] * WIDE_UNITS + [2] * WIDE_UNITS})
    reading = repeats.read(meta, "sample_id")
    menu = repeats.menu(reading["reading"], [])

    first, second = x[:WIDE_UNITS], x[WIDE_UNITS:]
    mean = (first + second) / 2.0
    spread = x.std(axis=0, ddof=1)
    spread[spread == 0] = 1.0
    disagreement = np.abs(first - second).mean(axis=0) / spread
    changed = int((np.abs(first - second) > 0).any(axis=0).sum())
    top = list(np.argsort(-disagreement)[: SHOWN_COLUMNS - 1])
    shown = [names[i] for i in top]
    # How much combining changes each of the 20,000 columns (mean |assay 1 - assay 2| over the
    # column's SD): the window shows the columns at the top of this distribution.
    r_edges = np.linspace(0.0, float(disagreement.max()) * 1.0001, 41)
    r_counts, _ = np.histogram(disagreement, bins=r_edges)
    rank = {"edges": [round(float(x), 5) for x in r_edges], "counts": [int(c) for c in r_counts],
            "shown": [round(float(disagreement[i]), 5) for i in top],
            "median": round(float(np.median(disagreement)), 5)}

    units = ids[:WINDOW_UNITS]
    rows = []
    for half, k in ((0, 0), (WIDE_UNITS, 1)):
        for j in range(WINDOW_UNITS):
            r = half + j
            values = {"replicate": int(meta.loc[r, "replicate"])}
            values.update({names[i]: float(x[r, i]) for i in top})
            rows.append({"row": r, "unit": ids[j], "k": k, "values": values})
    rows.sort(key=lambda r: r["row"])
    gap = {u: abs(float(first[j, top[0]] - second[j, top[0]])) for j, u in enumerate(units)}
    tracked = max(gap, key=gap.get)
    n_rows = 2 * WIDE_UNITS
    n_cols = WIDE_FEATURES + 2

    def combined(values: np.ndarray, label: str, sentence: str, steps: list[str], verb: str) -> dict[str, Any]:
        return {
            "key": label, "label": {"mean": "Their mean", "change": "The change from the first"}[label],
            "recommended": label == menu["recommended"], "sentence": sentence, "steps": steps,
            "folds": ["replicate"],
            "units": {u: {"values": {names[i]: round(float(values[j, i]), 2) for i in top}, "kept_row": None}
                      for j, u in enumerate(units)},
            "n_after": WIDE_UNITS, "columns_after": n_cols - 1,
            "flow": [{"key": "loaded", "label": "Assays in the table", "n": n_rows, "combined": 0, "dropped": 0},
                     {"key": "combined", "label": "One row per `sample_id`", "n": WIDE_UNITS,
                      "combined": WIDE_UNITS, "dropped": 0,
                      "reason": f"each sample's 2 assays {verb} into one row"}],
            "energy": None, "coach": []}

    def kept(which: int, label: str) -> dict[str, Any]:
        base = which * WIDE_UNITS
        return {
            "key": label, "label": {"first": "The first", "last": "The last"}[label],
            "recommended": False,
            "sentence": f"Each sample's {label} assay was kept and the rest dropped.",
            "steps": ["Gather rows by `sample_id`", f"Keep the {'lowest' if which == 0 else 'highest'} `replicate`"],
            "folds": [],
            "units": {u: {"values": {"replicate": which + 1,
                                     **{names[i]: float(x[base + j, i]) for i in top}},
                          "kept_row": base + j} for j, u in enumerate(units)},
            "n_after": WIDE_UNITS, "columns_after": n_cols,
            "strip_kept": [1 if base <= r < base + WIDE_UNITS else 0 for r in range(n_rows)],
            "flow": [{"key": "loaded", "label": "Assays in the table", "n": n_rows, "combined": 0, "dropped": 0},
                     {"key": "kept", "label": f"Each sample's {label} assay", "n": WIDE_UNITS,
                      "combined": 0, "dropped": WIDE_UNITS, "reason": f"the other {WIDE_UNITS} assays leave"}],
            "energy": None, "coach": []}

    methods = {
        "mean": combined(mean, "mean", "Each sample's assays were averaged into one row.",
                         ["Gather rows by `sample_id`", "Average each column"], "averaged"),
        "first": kept(0, "first"),
        "last": kept(1, "last"),
        "change": combined(second - first, "change", "Each sample's change from its first assay was used.",
                           ["Gather rows by `sample_id`", "Subtract the first from the last"], "differenced"),
    }
    return {
        "dataset": {"name": "wide_replicates (synthetic, seeded)", "rows": n_rows, "cols": n_cols,
                    "seed": WIDE_SEED},
        "id_column": "sample_id", "order_column": None, "index_column": "replicate",
        "n_units": WIDE_UNITS, "per_unit": 2, "noun": "assay", "unit_noun": "sample",
        "repeats": {"reading": reading["reading"], "spacing": None,
                    "replicate_index": reading["replicate_index"]},
        "menu": {"recommended": menu["recommended"], "reason": menu["reason"],
                 "from_pack": menu.get("from_pack"), "marker": menu["marker"]},
        "columns": {"record": ["replicate"], "shown": shown, "more_count": changed - len(shown),
                    "more": [], "constant": [], "n_changed": changed, "n_all": n_cols,
                    "kinds": {"sample_id": "id", "replicate": "record", **{c: "measure" for c in shown}}},
        "window": {"layout": "stacked", "units": units, "rows": rows, "tracked": tracked},
        "strip": {"unit": list(range(WIDE_UNITS)) * 2, "k": [0] * WIDE_UNITS + [1] * WIDE_UNITS},
        "rank": rank,
        "methods": methods,
    }


# ── orientation: the feature-major copy of metabolomics_untargeted.csv ───────


def orientation_scenario() -> dict[str, Any]:
    src = pd.read_csv(METABOLOMICS)
    num = src.select_dtypes(include=[np.number])
    t = num.T
    t.index.name = "feature_id"
    t = t.reset_index()
    t.columns = ["feature_id"] + [f"S{i:03d}" for i in range(1, t.shape[1])]
    before = orientation.read(t)
    turned = orientation.transpose(t)
    after_df = turned["df"]
    after = orientation.read(after_df)
    if not orientation.fires(["metabolomics"], before):
        raise SystemExit("the orientation question no longer fires on the feature-major copy")

    feats = list(t["feature_id"][:TURN])
    samples = list(t.columns[1: TURN + 1])
    cells = [[jsonable(t.loc[i, s]) for s in samples] for i in range(TURN)]
    # The transposed table holds the same numbers: assert it, cell by cell.
    for i, f in enumerate(feats):
        for j, s in enumerate(samples):
            a = after_df.loc[after_df["sample_id"] == s, f].iloc[0]
            b = cells[i][j]
            if not ((pd.isna(a) and b is None) or (b is not None and abs(float(a) - b) < 1e-9)):
                raise SystemExit(f"transpose moved a value: {f} x {s}")

    def logmeans(block: pd.DataFrame, axis: int) -> list[float]:
        m = block.select_dtypes(include=[np.number]).mean(axis=axis).abs()
        m = m[m > 0]
        return [round(float(v), 4) for v in np.log10(m)]

    tracked = samples[2]
    return {
        "dataset": {"name": "metabolomics_untargeted.csv, exported feature-major",
                    "rows": int(t.shape[0]), "cols": int(t.shape[1]), "sha256": sha256(METABOLOMICS)},
        "label_column": turned["label_column"], "sample_column": turned["sample_column"],
        "features": feats, "samples": samples, "cells": cells, "tracked": tracked,
        "before": {"rows": int(t.shape[0]), "cols": int(t.shape[1]), "ratio": before["ratio"],
                   "s_rows": before["s_rows"], "s_cols": before["s_cols"],
                   "row_means": logmeans(t.drop(columns="feature_id"), 1),
                   "col_means": logmeans(t.drop(columns="feature_id"), 0),
                   "reading": before["reading"], "sentence": before["sentence"]},
        "after": {"rows": int(after_df.shape[0]), "cols": int(after_df.shape[1]),
                  "ratio": after["ratio"], "s_rows": after["s_rows"], "s_cols": after["s_cols"],
                  "row_means": logmeans(after_df.drop(columns="sample_id"), 1),
                  "col_means": logmeans(after_df.drop(columns="sample_id"), 0),
                  "reading": after["reading"]},
        "threshold": orientation.FEATURE_MAJOR_RATIO,
        "methods_sentence": orientation.methods_sentence(orientation.ROWS_ARE_FEATURES, turned),
        "kept_sentence": orientation.methods_sentence(orientation.ROWS_ARE_SAMPLES),
    }


# ── seal: four bases ─────────────────────────────────────────────────────────


def seal_variant(key: str, df: pd.DataFrame, target: str, task: str, group: str | None,
                 basis: str, *, temporal: bool = False, time_col: str | None = None,
                 dataset: str, unit_noun: str, row_noun: str, known_groups: bool) -> dict[str, Any]:
    """One seal, drawn by turbotab.engine.draw_holdout at each holdout size the question offers."""
    df = df.reset_index(drop=True)
    n = len(df)
    out: dict[str, Any] = {
        "key": key, "basis": basis, "dataset": dataset, "target": target, "task": task,
        "group_column": group if known_groups else None, "unit_noun": unit_noun, "row_noun": row_noun,
        "n_rows": n, "n_cols": int(df.shape[1]), "seed": SEED,
        "exploratory": grain.is_exploratory_basis(basis),
    }
    index: dict[Any, int] = {}
    if known_groups and group:
        units = list(dict.fromkeys(df[group]))
        index = {u: i for i, u in enumerate(units)}
        out["row_unit"] = [index[u] for u in df[group]]
        out["n_units"] = len(units)
        if temporal and time_col:
            when = pd.to_datetime(df[time_col])
            last_seen = when.groupby(df[group]).max()
            t0 = last_seen.min()
            out["unit_time"] = [int((last_seen[u] - t0).days) for u in units]
            out["time_column"] = time_col
            out["time_start"] = str(t0.date())
            out["time_end"] = str(last_seen.max().date())
    else:
        out["row_unit"] = list(range(n))
        out["n_units"] = None
    # The draw does not know who a row belongs to when the basis is undetermined. What a candidate
    # column shows is evidence, not a claim about grouping, and it is said as that.
    cand = [] if known_groups else (grain.suggestion(df).get("from_name_heuristic") or [])
    evidence_col = cand[0] if cand and cand[0] in df.columns else None

    draws: dict[str, Any] = {}
    for fraction in (0.1, 0.2, 0.3):
        drawn = engine.draw_holdout(df, target, "classification" if task == "binary" else "regression",
                                    {"basis": basis, "group_col": group}, fraction=fraction, seed=SEED,
                                    time_col=time_col, temporal=temporal)
        held = set(drawn["labels"])
        hold = [1 if i in held else 0 for i in range(n)]
        d: dict[str, Any] = {
            "fraction": fraction, "hold": hold, "n_hold_rows": sum(hold), "n_train_rows": n - sum(hold),
            "achieved": round(drawn["disclosure"]["fraction"], 4),
            "chronological": bool(drawn["disclosure"]["chronological"]),
            "boundary": drawn["disclosure"]["boundary"],
            "n_hold_units": None, "n_train_units": None, "straddle": None,
        }
        if known_groups and group:
            hold_units = {df.loc[i, group] for i in held}
            train_units = {df.loc[i, group] for i in range(n) if i not in held}
            d["n_hold_units"] = len(hold_units)
            d["n_train_units"] = len(train_units)
            d["straddle"] = len(hold_units & train_units)
            d["straddle_units"] = sorted(index[u] for u in hold_units & train_units)
            if basis == "grouped" and d["straddle"]:
                raise SystemExit(f"a grouped seal put a {unit_noun[:-1]} on both sides")
        if evidence_col:
            h = {df.loc[i, evidence_col] for i in held}
            t = {df.loc[i, evidence_col] for i in range(n) if i not in held}
            d["evidence"] = {"column": evidence_col, "both_sides": len(h & t), "held_values": len(h)}
        draws[str(fraction)] = d
    out["draws"] = draws
    out.update(draws[str(HOLDOUT)])
    return out


def seal_scenarios(diet: pd.DataFrame, clin: pd.DataFrame) -> dict[str, Any]:
    person = repeats.aggregate(diet, ID, "mean", target="hba1c", order_col=ORDER)["frame"]
    variants = {
        "grouped": seal_variant(
            "grouped", person, "hba1c", "regression", ID,
            grain.seal_basis(grain.PEOPLE_REPEAT, ID, person[ID].nunique()),
            dataset="dietary_recalls.csv, combined (mean)", unit_noun="participants",
            row_noun="rows", known_groups=True),
        "chronological": seal_variant(
            "chronological", clin, "progressed", "binary", "subject_id",
            grain.seal_basis(grain.PEOPLE_REPEAT, "subject_id", clin["subject_id"].nunique()),
            temporal=True, time_col="visit_date", dataset=CLINICAL.name,
            unit_noun="subjects", row_noun="visits", known_groups=True),
    }
    pilot_ids = list(dict.fromkeys(clin["subject_id"]))[:7]
    pilot = clin[clin["subject_id"].isin(pilot_ids)]
    variants["abandoned"] = seal_variant(
        "abandoned", pilot, "progressed", "binary", "subject_id",
        grain.seal_basis(grain.PEOPLE_REPEAT, "subject_id", len(pilot_ids)),
        dataset=f"{CLINICAL.name}, its first {len(pilot_ids)} subjects", unit_noun="subjects",
        row_noun="visits", known_groups=True)
    variants["undetermined"] = seal_variant(
        "undetermined", diet, "hba1c", "regression", None,
        grain.seal_basis(grain.NOT_SURE), dataset=DIETARY.name, unit_noun="participants",
        row_noun="recalls", known_groups=False)
    # What each holdout size can measure, from n alone (the outcome is not read): the 95% interval
    # of an RMSE estimated on n held-out units is about ±1.96/sqrt(2n) of it.
    sizes = []
    n_people = variants["grouped"]["n_rows"]
    for f in (0.0, 0.1, 0.2, 0.3):
        n = int(round(n_people * f))
        sizes.append({"fraction": f, "n": n,
                      "rmse_pm": None if n == 0 else round(1.96 / math.sqrt(2 * n), 4)})
    return {"variants": variants, "sizes": sizes, "min_groups": 8}


# ── results: the combined dietary table, sealed ──────────────────────────────

PREDICTORS = ["age", "sex", "bmi", "energy_kcal", "protein_g", "fat_g", "carbohydrate_g",
              "fiber_g", "sodium_mg"]
NUTRIENTS = ["protein_g", "fat_g", "carbohydrate_g"]
FAMILIES = ["linear", "elastic_net", "boosted_trees"]


def fit_all(train: pd.DataFrame, held: pd.DataFrame, method: str) -> dict[str, Any]:
    from sklearn.model_selection import KFold

    roles = {"age": "covariate", "sex": "covariate", "bmi": "covariate", "energy_kcal": "energy",
             **{c: "exposure" for c in ["protein_g", "fat_g", "carbohydrate_g", "fiber_g", "sodium_mg"]}}
    state = ProjectState(roles=roles, missing="complete_case", energy_adjustment=EnergyAdjustment(
        method=method, energy_column="energy_kcal", nutrients=NUTRIENTS))
    spec = design_spec(state, train, PREDICTORS)
    folds = list(KFold(FOLDS, shuffle=True, random_state=SEED).split(train))
    y = train["hba1c"].to_numpy(float)
    base = []
    for a, b in folds:
        pred = np.full(len(b), y[a].mean())
        base.append(1 - ((y[b] - pred) ** 2).sum() / ((y[b] - y[b].mean()) ** 2).sum())
    out = []
    for key in FAMILIES:
        fam = get_family(key)
        per = []
        for a, b in folds:
            m = build_pipeline(spec, fam, "regression", "prediction", len(a), len(PREDICTORS))
            m.fit(train.iloc[a][spec.inputs], train.iloc[a]["hba1c"])
            per.append(score("regression", m, train.iloc[b][spec.inputs], train.iloc[b]["hba1c"]))
        m = build_pipeline(spec, fam, "regression", "prediction", len(train), len(PREDICTORS))
        m.fit(train[spec.inputs], train["hba1c"])
        cv = summarize("regression", per)
        ho = score("regression", m, held[spec.inputs], held["hba1c"])
        out.append({"family": key, "label": fam.label,
                    "cv": {"mean": round(cv["r2"]["mean"], 4), "sd": round(cv["r2"]["sd"], 4),
                           "folds": [round(v, 4) for v in cv["r2"]["folds"]]},
                    "holdout": round(ho["r2"], 4)})
    return {"method": method, "models": out, "baseline": round(float(np.mean(base)), 4)}


def results_scenario(diet: pd.DataFrame, seal: dict[str, Any]) -> dict[str, Any]:
    person = repeats.aggregate(diet, ID, "mean", target="hba1c", order_col=ORDER)["frame"]
    hold = np.array(seal["variants"]["grouped"]["hold"], dtype=bool)
    train, held = person[~hold].reset_index(drop=True), person[hold].reset_index(drop=True)
    before = fit_all(train, held, "residual")
    after = fit_all(train, held, "density")
    return {"metric": "r2", "label": "R²", "higher_is_better": True, "n_train": int(len(train)),
            "n_holdout": int(len(held)), "folds": FOLDS, "seed": SEED,
            "target": "hba1c", "group_column": ID,
            "fits": {"residual": before, "density": after}}


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        diet_frame, diet_parquet = load(DIETARY, work)
        # DataStore.materialize indexes by the ingest's row id; name it the way the parquet does.
        diet = diet_frame.rename_axis("__row_id").reset_index()
        reshape = reshape_scenario(diet, diet_parquet)
        clin_frame, _ = load(CLINICAL, work)
        clin = clin_frame.reset_index(drop=True)
        if "__row_id" in clin.columns:
            clin = clin.drop(columns="__row_id")
        diet_plain = diet.drop(columns="__row_id")
        seal = seal_scenarios(diet_plain, clin)
        results = results_scenario(diet_plain, seal)
    fixture = {
        "meta": {
            "script": "docs/turbotab-next/m2/explore/capture.py",
            "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
            "sources": {p.name: sha256(p) for p in (DIETARY, CLINICAL, METABOLOMICS)},
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "duckdb": duckdb.__version__},
        },
        "reshape": reshape,
        "wide": wide_scenario(),
        "orientation": orientation_scenario(),
        "seal": seal,
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)} ({kb:.0f} KB)")
    m = reshape["methods"]
    print("reshape:", {k: (v["n_after"], v["columns_after"], [n["text"] for n in v["coach"]]) for k, v in m.items()})
    print("seal:", {k: (v["basis"], v["n_train_rows"], v["n_hold_rows"], v.get("straddle"), v.get("evidence"))
                    for k, v in seal["variants"].items()})
    print("results:", {k: [(x["family"], x["cv"]["mean"], x["holdout"]) for x in v["models"]]
                       for k, v in results["fits"].items()})


if __name__ == "__main__":
    main()
