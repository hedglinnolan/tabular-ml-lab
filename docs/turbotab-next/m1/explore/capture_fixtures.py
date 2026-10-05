"""Capture real-data fixtures for the consequence-preview design prototypes (BLUEPRINT §11).

Every number in ``fixtures.json`` is computed here, from the real files, by the real code:
DuckDB ingest and ``DataStore`` (turbotab.core.datastore), ``EnergyAdjuster`` and
``applicable_methods`` (turbotab.core.methods.energy), the findings stage
(turbotab.core.stages.findings, which wraps turbotab.engine.diagnose and turbotab.packs.findings),
and sklearn for the split and the model matrix. Every view is validated by the pydantic models in
turbotab.core.consequences before it is written. The exclusion cut-offs are parsed from
NUTRITION_PACK §02, not typed here.

Run from the repository root:

    venv/bin/python docs/turbotab-next/m1/explore/capture_fixtures.py [--nhanes PATH]

Scenario A: the NHANES export, dietary lens, outcome glucose (energy adjustment on training rows,
exclusions on loaded rows, findings on the whole table). Scenario B: the 60 x 500 genomics fixture
with a log2(x + 1) transform of its count columns.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from pydantic import TypeAdapter
from scipy.stats import skew, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from turbotab.core.consequences import (  # noqa: E402
    CAPTION_WORDS, MAX_VIEWS, TITLE_WORDS, ConsequenceView, DistributionView, HistogramData,
    Lineage, LineageLink, LineageNode, LineageView, Mark, PreviewContext, PreviewResult,
    RelationshipView, RowFlowView, RowStep, TableFocusView, TableRow, words,
)
from turbotab.core.datastore import DataStore, ingest  # noqa: E402
from turbotab.core.decisions import (  # noqa: E402
    ExclusionRule, ProjectState, RangeByLevel, Refusal, SetEnergyAdjustment, SetExclusions,
    SplitSpec,
)
from turbotab.core.methods.energy import (  # noqa: E402
    METHODS, EnergyAdjuster, applicable_methods, describe_method, nutrient_role,
)
from turbotab.core.stages.findings import findings_stage  # noqa: E402

DEFAULT_NHANES = Path("/Users/nhedglin/tabular-ml-lab/_tt_tmp_nhanes.csv")
GENOMICS = ROOT / "turbotab/sample_data/genomics_expression.csv"
PACK = ROOT / "docs/turbotab-next/reference/research/NUTRITION_PACK.md"
OUT = Path(__file__).resolve().parent / "fixtures.json"

BUDGET = 4 * 1024**3
BINS = 30  # DataStore.histogram's default
SCATTER_POINTS = 800
TABLE_ROWS = 8
TABLE_COLUMNS = 12

OUTCOME = "glucose"
ENERGY = "kcal"
COVARIATES = ["age", "gender", "bmi"]
NUTRIENTS = ["protein", "carb", "sugar", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
PREDICTORS = ["age", "gender", "bmi", "kcal", "protein", "carb", "sugar", "fat_total", "fat_sat",
              "fat_mon", "fat_poly"]
CATEGORICAL = ["gender"]
SEX = "gender"
TARGET_B = "condition"  # case/control: the genomics fixture's outcome, so not in the lineage

VIEWS = TypeAdapter(ConsequenceView)


# ── small helpers ────────────────────────────────────────────────────────────


def sig(value: float, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(f"{value:.{digits}g}")


def rounded(obj: Any) -> Any:
    """Round every float to 4 significant digits; ints and strings are left alone."""
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, (float, np.floating)):
        return sig(float(obj))
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): rounded(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v) for v in obj]
    return obj


def scalar(value: Any) -> Any:
    """A JSON-ready cell value."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return sig(float(value))
    if pd.isna(value):
        return None
    return value if isinstance(value, (int, str)) else str(value)


def pearson(x: Any, y: Any) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def histogram(values: Any, edges: list[float] | None = None) -> HistogramData:
    """DataStore.histogram's binning (equal width over min..max) on an in-memory column.

    With ``edges`` given, the bins are reused (so a before/after pair shares one axis).
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    x, n_missing = v[finite], int((~finite).sum())
    if edges is None:
        if x.size == 0:
            return HistogramData(edges=[], counts=[], n_missing=n_missing)
        lo, hi = float(x.min()), float(x.max())
        if hi == lo:
            edges = [lo - 0.5, lo + 0.5]
        else:
            width = (hi - lo) / BINS
            edges = [lo + i * width for i in range(BINS)] + [hi]
    start, nb = edges[0], len(edges) - 1
    width = edges[1] - edges[0]
    idx = np.clip(np.floor((x - start) / width).astype(np.int64), 0, nb - 1)
    counts = np.bincount(idx, minlength=nb)
    return HistogramData(edges=[float(e) for e in edges], counts=[int(c) for c in counts],
                         n_missing=n_missing)


def check_view(view: Any) -> dict[str, Any]:
    """Validate a view against the contract, enforce the word budgets, return it as JSON."""
    view = VIEWS.validate_python(view.model_dump())
    assert words(view.title) <= TITLE_WORDS, (view.title, words(view.title))
    assert words(view.caption) <= CAPTION_WORDS, (view.caption, words(view.caption))
    return view.model_dump(mode="json")


def check_preview(result: PreviewResult) -> dict[str, Any]:
    result = PreviewResult.model_validate(result.model_dump())
    assert len(result.views) <= MAX_VIEWS
    for view in result.views:
        check_view(view)
    return result.model_dump(mode="json")


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_num(x: float) -> str:
    if x != 0 and abs(x) < 1e-6:
        return "≈0"  # e.g. NHANES's 5.4e-79, SAS's stand-in for zero
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def fmt_r(r: float) -> str:
    text = f"{r:.2f}"
    return "0.00" if text == "-0.00" else text


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path, workdir: Path) -> tuple[pd.DataFrame, DataStore, Path]:
    """Ingest the way the app does (DuckDB -> raw.parquet with __row_id) and materialize."""
    parquet = workdir / f"{path.stem}.parquet"
    ingest(path, parquet)
    store = DataStore(parquet, BUDGET)
    return store.materialize(), store, parquet


# ── NUTRITION_PACK §02: the cut-offs, read from the pack ─────────────────────

PACK_02 = "docs/turbotab-next/reference/research/NUTRITION_PACK.md#02 · Implausible intake exclusions"


def pack_cutoffs() -> dict[str, Any]:
    text = PACK.read_text(encoding="utf-8")
    section = text.split("## 02 · Implausible intake exclusions", 1)[1].split("\n## ", 1)[0]
    section = re.sub(r"\s+", " ", section)  # the pack wraps its lines mid-phrase
    num = lambda s: float(s.replace(",", ""))  # noqa: E731
    sex = re.search(r"Willett / Nurses' Health Study \*\*\[(\w+)\]\*\*: women \*\*<([\d,]+) or "
                    r">([\d,]+) kcal/d\*\*; men \*\*<([\d,]+) or >([\d,]+) kcal/d\*\*", section)
    neutral = re.search(r"sex-neutral \*\*([\d,]+)–([\d,]+)\*\*", section)
    if not sex or not neutral:
        raise SystemExit("NUTRITION_PACK §02 no longer states the fixed kcal screens this script reads")
    quote = sex.group(0).replace("**", "")
    return {
        "sex_specific": {"status": sex.group(1), "female": (num(sex.group(2)), num(sex.group(3))),
                         "male": (num(sex.group(4)), num(sex.group(5))), "quote": quote},
        "neutral": {"status": "CONVENTION", "range": (num(neutral.group(1)), num(neutral.group(2))),
                    "quote": "Variants in circulation: " + neutral.group(0).replace("**", "")},
    }


# ── Scenario A ───────────────────────────────────────────────────────────────


def scenario_a_setup(frame: pd.DataFrame, cutoffs: dict[str, Any]) -> dict[str, Any]:
    low, high = cutoffs["neutral"]["range"]
    ids = frame.index.to_numpy()
    measured = frame[OUTCOME].notna()
    plausible = measured & frame[ENERGY].between(low, high)
    complete = plausible & frame[PREDICTORS + [OUTCOME]].notna().all(axis=1)
    cohort = ids[complete.to_numpy()]
    train, holdout = train_test_split(cohort, test_size=0.2, random_state=0, shuffle=True)
    train, holdout = np.sort(train), np.sort(holdout)
    roles: dict[str, str] = {}
    for c in frame.columns:
        if c == OUTCOME:
            continue
        roles[c] = ("identifier" if c == "SEQN" else "time" if c == "cycle_begin_year"
                    else "flag" if c.startswith("imputed_") else "energy" if c == ENERGY
                    else "exposure" if c in NUTRIENTS else "covariate" if c in COVARIATES
                    else "excluded")
    state = ProjectState(
        lens=["dietary"], target=OUTCOME, task="regression", purpose="prediction", roles=roles,
        exclusions=[ExclusionRule(column=ENERGY, low=low, high=high,
                                  reason=f"Implausible energy intake ({PACK_02})")],
        missing="complete_case", split=SplitSpec(holdout=0.2, seed=0, folds=5))
    return {"cohort": cohort, "train": train, "holdout": holdout, "roles": roles, "state": state,
            "steps": [
                {"key": "loaded", "n": int(len(frame))},
                {"key": "outcome_measured", "n": int(measured.sum())},
                {"key": "exclude_kcal", "n": int(plausible.sum()), "rule": f"{ENERGY} {low:g}–{high:g}"},
                {"key": "complete_cases", "n": int(complete.sum())},
            ]}


def short_formula(entry: dict[str, Any]) -> str | None:
    if entry["operation"] in ("pass-through", "kept"):
        return None
    return entry["formula"].split(": ")[0].split("  (")[0]


def energy_lineage(adj: EnergyAdjuster, matrix: ColumnTransformer, roles: dict[str, str],
                   method: str) -> Lineage:
    nodes = [LineageNode(id=f"raw:{c}", column=c, lane="raw", role=roles[c], label=c)
             for c in adj.feature_names_in_]
    links: list[LineageLink] = []
    for entry in adj.lineage():
        out, op = entry["output"], entry["operation"]
        role = "energy" if out == "kcal_from_other" else roles[entry["inputs"][0]]
        nodes.append(LineageNode(id=f"adj:{out}", column=out, lane="adjusted", role=role, label=out,
                                 formula=short_formula(entry)))
        label = "kept" if op in ("pass-through", "kept") else f"energy-adjusted ({method})"
        links += [LineageLink(source=f"raw:{i}", target=f"adj:{out}", operation=label)
                  for i in entry["inputs"]]
        step = matrix.named_transformers_[out]
        operation = "one-hot" if isinstance(step, OneHotEncoder) else "scaled"
        for name in step.get_feature_names_out([out]):
            nodes.append(LineageNode(id=f"mx:{name}", column=str(name), lane="matrix", role=role,
                                     label=str(name)))
            links.append(LineageLink(source=f"adj:{out}", target=f"mx:{name}", operation=operation))
    return Lineage(nodes=nodes, links=links, collapsed=False)


def model_matrix(adjusted: pd.DataFrame) -> ColumnTransformer:
    """The linear family's matrix step (one-hot, then scale), fit on the adjusted training rows."""
    steps = [(c, OneHotEncoder(drop="if_binary", sparse_output=False) if c in CATEGORICAL
              else StandardScaler(), [c]) for c in adjusted.columns]
    return ColumnTransformer(steps, verbose_feature_names_out=False).fit(adjusted)


PHRASE = {"residual": "residuals on kcal", "density": "densities per kcal"}


def lineage_caption(adj: EnergyAdjuster, method: str) -> str:
    ops = [e["operation"] for e in adj.lineage()]
    outs = [e["output"] for e in adj.lineage()]
    n_in = len(adj.feature_names_in_)
    if method == "none":
        return f"All {n_in} predictors enter the model as recorded; nothing is energy-adjusted."
    if method == "standard":
        return f"Nutrients enter unchanged, and {ENERGY} stays in the model beside them."
    if method == "partition":
        n = ops.count("partition")
        return f"{ENERGY} splits into kcal from {n} nutrients and kcal from everything else."
    n = sum(op in PHRASE for op in ops)
    kind = next(op for op in ops if op in PHRASE)
    fate = "stays in" if ENERGY in outs else "leaves"
    return f"{n} nutrients become {PHRASE[kind]}; {ENERGY} {fate} the model."


def relationship_caption(method: str, focus: str, after_col: str, r0: float, r1: float) -> str:
    if method == "none":
        return f"{focus} correlates {fmt_r(r0)} with {ENERGY}; with no adjustment it enters the model unchanged."
    if method == "standard":
        return f"{focus} keeps its {fmt_r(r0)} correlation with {ENERGY}; {ENERGY} enters the model beside it."
    if method == "partition":
        return (f"{after_col} correlates {fmt_r(r1)} with {ENERGY}, as {focus} did; "
                f"energy is split, not removed.")
    tail = "; kcal stays in the model." if method == "density_multivariate" else "."
    return f"{focus} correlates {fmt_r(r0)} with {ENERGY}; {after_col} correlates {fmt_r(r1)}{tail}"


def energy_option(method: str, nutrients: list[str], X: pd.DataFrame, setup: dict[str, Any],
                  focus: str, ctx: PreviewContext, applicability: dict[str, Any],
                  before_lineage: Lineage | None) -> dict[str, Any]:
    roles = setup["roles"]
    decision = SetEnergyAdjustment(method=method, energy_column=ENERGY, nutrients=nutrients)
    described = describe_method(method)
    verdict = applicability[method]
    option: dict[str, Any] = {
        "method": method, "label": described["label"], "decision": decision.model_dump(mode="json"),
        "applicable": verdict, "estimand": EnergyAdjuster(method=method).estimand(),
        "method_card": {k: described[k] for k in
                        ("specification", "kind", "standing", "caveats", "source")},
    }
    if not verdict["ok"]:
        exits = [{"label": describe_method(m)["label"],
                  "decision": SetEnergyAdjustment(method=m, energy_column=ENERGY, nutrients=nutrients)}
                 for m in METHODS if applicability[m]["ok"]]
        refusal = Refusal("method_not_applicable",
                          f"{described['label']} cannot be used: {verdict['reason']}", exits)
        option.update(refusal=refusal.to_dict(), preview=None, extra_views=[])
        return option

    adj = EnergyAdjuster(method=method, energy_column=ENERGY, nutrient_columns=nutrients).fit(X)
    A = adj.transform(X)
    matrix = model_matrix(A)
    lineage = energy_lineage(adj, matrix, roles, method)
    entries = adj.lineage()
    after_col = next(e["output"] for e in entries if e["inputs"][0] == focus)
    e = X[ENERGY]
    r0, r1 = pearson(X[focus], e), pearson(A[after_col], e)
    pts = ctx.sample_row_ids(n=SCATTER_POINTS, seed=0)
    n_train = len(X)

    relationship = RelationshipView(
        title=f"{focus} against {ENERGY}, before and after",
        caption=relationship_caption(method, focus, after_col, r0, r1),
        emphasis=[focus, after_col] if after_col != focus else [focus],
        x_label=ENERGY, y_label_before=focus, y_label_after=after_col,
        points_before=[(float(e[i]), float(X.at[i, focus])) for i in pts],
        points_after=[(float(e[i]), float(A.at[i, after_col])) for i in pts],
        r_before=r0, r_after=r1)
    outs = [x["output"] for x in entries]
    changed_outputs = [x["output"] for x in entries if x["operation"] not in ("pass-through", "kept")]
    lineage_view = LineageView(
        title="Columns entering the model", caption=lineage_caption(adj, method),
        emphasis=changed_outputs + [c for c in adj.dropped_columns_],
        before=before_lineage, after=lineage)
    same = after_col == focus
    m0, m1 = float(np.nanmedian(X[focus])), float(np.nanmedian(A[after_col]))
    distribution = DistributionView(
        title=f"{focus} before and after adjustment",
        caption=(f"{focus} is unchanged: median {fmt_num(m0)} on {fmt_int(n_train)} training rows."
                 if same else
                 f"Median {focus} {fmt_num(m0)}; median {after_col} {fmt_num(m1)}, "
                 f"on {fmt_int(n_train)} training rows."),
        emphasis=[after_col], column=focus, before=histogram(X[focus]), after=histogram(A[after_col]),
        before_label=focus, after_label=after_col if not same else f"{focus} (unchanged)")

    # table focus: energy and the nutrients, before; what each becomes, after
    cols_before = [ENERGY] + nutrients
    source = {x["output"]: (ENERGY if x["output"] == "kcal_from_other" else x["inputs"][0])
              for x in entries}
    cols_after = [o for o in outs if source[o] in cols_before]
    rows_ids = ctx.sample_row_ids(n=TABLE_ROWS, seed=0)
    rows, changed = [], []
    for rid in rows_ids:
        before = {c: scalar(X.at[rid, c]) for c in cols_before}
        after = {o: scalar(A.at[rid, o]) for o in cols_after}
        rows.append(TableRow(row_id=int(rid), before=before, after=after))
        for o in cols_after:
            if o != source[o] or not np.isclose(float(A.at[rid, o]), float(X.at[rid, source[o]])):
                changed.append((int(rid), o))
    touched = {source[o] for o in changed_outputs} | set(adj.dropped_columns_)
    n_aff = len([c for c in cols_before if c in touched])
    n_other = len(PREDICTORS) - len(cols_before)
    table = TableFocusView(
        title="The adjusted columns on eight training rows",
        caption=(f"No values change; {ENERGY} and the {len(nutrients)} nutrients enter as recorded."
                 if n_aff == 0 else
                 f"{n_aff} of {len(cols_before)} columns change or leave the model; "
                 f"{n_other} other predictors pass through unchanged."),
        emphasis=changed_outputs, columns_before=cols_before, columns_after=cols_after, rows=rows,
        changed=changed, n_affected_columns=n_aff)

    basis = (f"{fmt_int(n_train)} training rows of {fmt_int(len(setup['cohort']))} in the cohort; "
             f"the scatter shows {len(pts)} of them")
    preview = PreviewResult(kind="set_energy_adjustment",
                            views=[relationship, lineage_view, distribution], basis=basis)
    option.update(
        refusal=None, preview=check_preview(preview), extra_views=[check_view(table)],
        adjuster_lineage=entries, dropped_columns=list(adj.dropped_columns_),
        matrix_columns=[str(c) for c in matrix.get_feature_names_out()],
        focus_after_column=after_col)
    return option


def scenario_a_energy(frame: pd.DataFrame, setup: dict[str, Any]) -> dict[str, Any]:
    X = frame.loc[setup["train"], PREDICTORS]
    ranking = sorted(((n, pearson(X[n], X[ENERGY])) for n in NUTRIENTS), key=lambda t: -abs(t[1]))
    focus = ranking[0][0]
    ctx = PreviewContext(project_id="fixture-nhanes", state=setup["state"], datastore=None,
                         artifact=lambda name: None, training_row_ids=setup["train"],
                         cohort_row_ids=setup["cohort"])
    applicability = applicable_methods(PREDICTORS, ENERGY, NUTRIENTS)
    none = EnergyAdjuster(method="none", energy_column=ENERGY, nutrient_columns=NUTRIENTS).fit(X)
    before_lineage = energy_lineage(none, model_matrix(none.transform(X)), setup["roles"], "none")
    options = [energy_option(m, NUTRIENTS, X, setup, focus, ctx, applicability, before_lineage)
               for m in METHODS]

    # Partition refuses the full nutrient set; show it on the macronutrient totals too, chosen by
    # rule: for each Atwater role, the column with the largest training mean (the total, not a part).
    by_role: dict[str, str] = {}
    for n in NUTRIENTS:
        role = nutrient_role(n)
        if role and (role not in by_role or X[n].mean() > X[by_role[role]].mean()):
            by_role[role] = n
    subset = [by_role[r] for r in ("protein", "carbohydrate", "fat") if r in by_role]
    sub_app = applicable_methods(PREDICTORS, ENERGY, subset)
    sub_focus = max(subset, key=lambda n: abs(pearson(X[n], X[ENERGY])))
    partition_subset = energy_option("partition", subset, X, setup, sub_focus, ctx, sub_app,
                                     before_lineage)
    partition_subset["nutrients"] = subset
    partition_subset["why_this_subset"] = (
        "Partition refuses the full nutrient set; this is the same method on the energy-bearing "
        "macronutrient totals (one column per Atwater role, the one with the largest mean).")
    return {
        "energy_column": ENERGY, "nutrients": NUTRIENTS,
        "focus_nutrient": focus,
        "correlation_with_energy": [{"column": n, "r": r} for n, r in ranking],
        "applicability": applicability,
        "options": options,
        "partition_on_macronutrient_totals": partition_subset,
    }


def scenario_a_exclusions(frame: pd.DataFrame, cutoffs: dict[str, Any]) -> dict[str, Any]:
    kcal = frame[ENERGY]
    measured = frame[OUTCOME].notna()
    complete_cols = PREDICTORS + [OUTCOME]
    lo_n, hi_n = cutoffs["neutral"]["range"]
    sexc = cutoffs["sex_specific"]
    options_spec = [
        ("keep_all", "Keep every row", None, []),
        ("kcal_500_5000", f"{fmt_num(lo_n)}–{fmt_num(hi_n)} kcal", cutoffs["neutral"],
         [ExclusionRule(column=ENERGY, low=lo_n, high=hi_n,
                        reason=f"Implausible energy intake, sex-neutral {fmt_num(lo_n)}–"
                               f"{fmt_num(hi_n)} kcal/d ({PACK_02})")]),
        ("sex_specific", "Sex-specific (Willett)", sexc,
         [ExclusionRule(column=ENERGY, by=RangeByLevel(column=SEX, ranges={
             "female": sexc["female"], "male": sexc["male"]}),
             reason=f"Implausible energy intake, women {fmt_num(sexc['female'][0])}–"
                    f"{fmt_num(sexc['female'][1])} and men {fmt_num(sexc['male'][0])}–"
                    f"{fmt_num(sexc['male'][1])} kcal/d ({PACK_02})")]),
    ]
    n_loaded = len(frame)
    n_measured = int(measured.sum())
    before_steps = [
        RowStep(key="loaded", label="Rows loaded", n=n_loaded),
        RowStep(key="outcome_measured", label=f"{OUTCOME} measured", n=n_measured,
                dropped=n_loaded - n_measured, reason=f"{OUTCOME} is missing"),
    ]
    n_cc = int((measured & frame[complete_cols].notna().all(axis=1)).sum())
    before_flow = before_steps + [RowStep(key="complete_cases", label="Complete cases", n=n_cc,
                                          dropped=n_measured - n_cc,
                                          reason="a predictor or the outcome is missing")]
    base_hist = histogram(kcal[measured])
    # the in-memory binning is DataStore.histogram's: prove it on the column the store can read
    options = []
    for key, label, evidence, rules in options_spec:
        decision = SetExclusions(rules=rules)
        keep = measured.copy()
        marks: list[Mark] = []
        by_sex: dict[str, dict[str, int]] = {}
        for rule in rules:
            if rule.by is None:
                inside = kcal.between(rule.low, rule.high)
                marks += [Mark(value=rule.low, label=f"{fmt_num(rule.low)} kcal"),
                          Mark(value=rule.high, label=f"{fmt_num(rule.high)} kcal")]
                below, above = int((measured & (kcal < rule.low)).sum()), int((measured & (kcal > rule.high)).sum())
                by_sex["all"] = {"below": below, "above": above}
            else:
                inside = pd.Series(True, index=frame.index)
                for level, (lo, hi) in rule.by.ranges.items():
                    at = frame[rule.by.column] == level
                    inside &= ~at | kcal.between(lo, hi)
                    marks += [Mark(value=lo, label=f"{fmt_num(lo)} kcal", group=level),
                              Mark(value=hi, label=f"{fmt_num(hi)} kcal", group=level)]
                    by_sex[level] = {"below": int((measured & at & (kcal < lo)).sum()),
                                     "above": int((measured & at & (kcal > hi)).sum()),
                                     "n": int((measured & at).sum())}
            keep &= inside
        n_kept = int(keep.sum())
        after = list(before_steps)
        if rules:
            rng = (f"{fmt_num(rules[0].low)}–{fmt_num(rules[0].high)} kcal" if rules[0].by is None
                   else "sex-specific kcal ranges")
            after.append(RowStep(key="exclude_kcal", label=f"{ENERGY} outside {rng}", n=n_kept,
                                 dropped=n_measured - n_kept, reason=rules[0].reason))
        n_final = int((keep & frame[complete_cols].notna().all(axis=1)).sum())
        after.append(RowStep(key="complete_cases", label="Complete cases", n=n_final,
                             dropped=n_kept - n_final, reason="a predictor or the outcome is missing"))
        dropped = n_measured - n_kept
        if not rules:
            flow_caption = f"No rows are excluded; all {fmt_int(n_measured)} rows with {OUTCOME} measured remain."
            dist_caption = f"Every {ENERGY} value is kept, from {fmt_num(kcal.min())} to {fmt_num(kcal.max())}."
        else:
            pct = 100 * dropped / n_measured
            flow_caption = (f"{fmt_int(dropped)} rows ({pct:.1f}%) fall outside {rng}; "
                            f"{fmt_int(n_final)} remain.")
            if rules[0].by is None:
                c = by_sex["all"]
                dist_caption = (f"{fmt_int(c['below'])} rows below {fmt_num(rules[0].low)} and "
                                f"{fmt_int(c['above'])} above {fmt_num(rules[0].high)} kcal are cut.")
            else:
                f, m = by_sex["female"], by_sex["male"]
                (flo, fhi), (mlo, mhi) = rules[0].by.ranges["female"], rules[0].by.ranges["male"]
                dist_caption = (f"Women: {fmt_int(f['below'])} below {fmt_num(flo)}, {fmt_int(f['above'])} "
                                f"above {fmt_num(fhi)}. Men: {fmt_int(m['below'])} below {fmt_num(mlo)}, "
                                f"{fmt_int(m['above'])} above {fmt_num(mhi)}.")
        flow = RowFlowView(title="Rows kept at each step", caption=flow_caption,
                           emphasis=["exclude_kcal"] if rules else [], before=before_flow, after=after)
        dist = DistributionView(
            title=f"{ENERGY} with the cut marked" if rules else f"{ENERGY}, nothing cut",
            caption=dist_caption, emphasis=[ENERGY], column=ENERGY, before=base_hist,
            after=histogram(kcal[keep], edges=base_hist.edges), before_label=f"{ENERGY}, all rows",
            after_label=f"{ENERGY}, rows kept", marks=marks)
        preview = PreviewResult(kind="set_exclusions", views=[flow, dist],
                                basis=f"All {fmt_int(n_loaded)} loaded rows; exclusions come before the split")
        options.append({
            "key": key, "label": label, "decision": decision.model_dump(mode="json"),
            "evidence": ({"status": evidence["status"], "source": PACK_02, "quote": evidence["quote"]}
                         if evidence else None),
            "counts": {"excluded": dropped, "kept": n_kept, "final": n_final, "by_level": by_sex},
            "preview": check_preview(preview),
        })
    return {"basis_note": ("Exclusions are answered before the split, so these previews are row-"
                           "descriptive over every loaded row (consequences.py: before the split "
                           "exists, only row-descriptive previews run)."),
            "options": options}


def scenario_a_findings(parquet: Path) -> dict[str, Any]:
    ctx = SimpleNamespace(state=SimpleNamespace(lens=["dietary"], target=OUTCOME),
                          paths={"data": str(parquet)}, settings={"memory_budget_bytes": BUDGET},
                          progress=lambda fraction, message: None)
    return findings_stage(ctx)


# ── Scenario B ───────────────────────────────────────────────────────────────


def log2p1(x: Any) -> Any:
    return np.log2(np.asarray(x, dtype=float) + 1.0)


def shape_change(before: np.ndarray, after: np.ndarray) -> float:
    """Wasserstein-1 between the z-scored columns: how much the shape moved, not the scale."""
    sb, sa = before.std(), after.std()
    if sb == 0 or sa == 0:
        return 0.0
    return float(wasserstein_distance((before - before.mean()) / sb, (after - after.mean()) / sa))


def wide_lineage(frame: pd.DataFrame, roles: dict[str, str], counts: list[str],
                 transformed: bool) -> Lineage:
    group = "exposure"
    n = len(counts)
    kept = [c for c in roles if c not in counts and roles[c] != "identifier"]
    nodes = [LineageNode(id=f"raw:{c}", column=c, lane="raw", role=roles[c], label=c)
             for c in roles if c not in counts]
    nodes.append(LineageNode(id="raw:counts", column=None, lane="raw", role=group,
                             label=f"{n} count columns", group=group, count=n))
    links: list[LineageLink] = []
    adjusted = frame[kept + counts]
    steps = ([(c, OneHotEncoder(drop="if_binary", sparse_output=False)
               if not pd.api.types.is_numeric_dtype(frame[c]) else StandardScaler(), [c])
              for c in kept] + [("counts", StandardScaler(), counts)])
    matrix = ColumnTransformer(steps, verbose_feature_names_out=False).fit(
        adjusted.assign(**{c: log2p1(adjusted[c]) for c in counts}) if transformed else adjusted)
    for c in kept:
        nodes.append(LineageNode(id=f"adj:{c}", column=c, lane="adjusted", role=roles[c], label=c))
        links.append(LineageLink(source=f"raw:{c}", target=f"adj:{c}", operation="kept"))
        step = matrix.named_transformers_[c]
        op = "one-hot" if isinstance(step, OneHotEncoder) else "scaled"
        for name in step.get_feature_names_out([c]):
            nodes.append(LineageNode(id=f"mx:{name}", column=str(name), lane="matrix",
                                     role=roles[c], label=str(name)))
            links.append(LineageLink(source=f"adj:{c}", target=f"mx:{name}", operation=op))
    nodes.append(LineageNode(id="adj:counts", column=None, lane="adjusted", role=group,
                             label=f"log2 of {n} count columns" if transformed else f"{n} count columns",
                             formula="log2(x + 1)" if transformed else None, group=group, count=n))
    links.append(LineageLink(source="raw:counts", target="adj:counts",
                             operation="log2(x + 1)" if transformed else "kept"))
    n_mx = len(matrix.named_transformers_["counts"].get_feature_names_out(counts))
    nodes.append(LineageNode(id="mx:counts", column=None, lane="matrix", role=group,
                             label=f"{n_mx} scaled count columns", group=group, count=n_mx))
    links.append(LineageLink(source="adj:counts", target="mx:counts", operation="scaled"))
    return Lineage(nodes=nodes, links=links, collapsed=True)


def scenario_b(frame: pd.DataFrame) -> dict[str, Any]:
    counts = [c for c in frame.columns if c.startswith("gene_")]
    values = frame[counts].to_numpy(dtype=float)
    assert np.all(values >= 0) and np.all(values == np.round(values)), "gene_* are not counts"
    roles = {c: ("identifier" if frame[c].is_unique and not pd.api.types.is_numeric_dtype(frame[c])
                 else "exposure" if c in counts else "covariate")
             for c in frame.columns if c != TARGET_B}
    transform = ColumnTransformer([("log2p1", FunctionTransformer(log2p1, feature_names_out="one-to-one"),
                                    counts)], remainder="passthrough", verbose_feature_names_out=False)
    transform.set_output(transform="pandas")
    after = transform.fit_transform(frame)[list(frame.columns)]
    change = sorted(((c, shape_change(frame[c].to_numpy(float), after[c].to_numpy(float)))
                     for c in counts), key=lambda t: -t[1])
    shown = [c for c, _ in change[:TABLE_COLUMNS]]
    top = shown[0]
    ctx = PreviewContext(project_id="fixture-genomics", state=ProjectState(lens=["genomics"], target=TARGET_B),
                         datastore=None, artifact=lambda name: None, training_row_ids=None,
                         cohort_row_ids=None)
    row_ids = ctx.sample_row_ids(pool=frame.index.to_numpy(), n=TABLE_ROWS, seed=0)
    rows, changed = [], []
    for rid in row_ids:
        rows.append(TableRow(row_id=int(rid), before={c: scalar(frame.at[rid, c]) for c in shown},
                             after={c: scalar(after.at[rid, c]) for c in shown}))
        changed += [(int(rid), c) for c in shown if float(after.at[rid, c]) != float(frame.at[rid, c])]
    n = len(counts)
    table = TableFocusView(
        title=f"The {len(shown)} most-changed count columns",
        caption=(f"All {n} count columns become log2(x + 1); shown: the {len(shown)} whose shape "
                 f"changed most."),
        emphasis=[top], columns_before=shown, columns_after=shown, rows=rows, changed=changed,
        n_affected_columns=n)
    b, a = frame[top].to_numpy(float), after[top].to_numpy(float)
    dist = DistributionView(
        title=f"{top} before and after log2(x + 1)",
        caption=(f"Skewness of {top} goes from {skew(b):.1f} to {skew(a):.1f}; median "
                 f"{fmt_num(np.median(b))} becomes {fmt_num(np.median(a))}."),
        emphasis=[top], column=top, before=histogram(b), after=histogram(a),
        before_label=f"{top} (counts)", after_label=f"log2({top} + 1)")
    before_lineage = wide_lineage(frame, roles, counts, transformed=False)
    after_lineage = wide_lineage(frame, roles, counts, transformed=True)
    lineage = LineageView(
        title="Columns entering the model",
        caption=f"{n} count columns become log2(x + 1); the other columns pass through unchanged.",
        emphasis=["adj:counts"], before=before_lineage, after=after_lineage)
    preview = PreviewResult(
        kind="log_transform", views=[table, dist, lineage],
        basis=f"All {len(frame)} rows (a row-local transform learns nothing from the data)",
        note=None)
    return {
        "dataset": {"path": str(GENOMICS.relative_to(ROOT)), "sha256": sha256(GENOMICS),
                    "n_rows": int(len(frame)), "n_cols": int(frame.shape[1])},
        "target": TARGET_B,
        "roles": {k: v for k, v in roles.items() if k not in counts} | {"gene_*": "exposure"},
        "count_columns": {"n": n, "first": counts[0], "last": counts[-1]},
        "transform": {"kind": "log_transform", "formula": "log2(x + 1)",
                      "note": ("No M1 decision kind does this; it stands for a future preprocessing "
                               "kind, previewed through the generic-diff views.")},
        "change_metric": ("Wasserstein-1 distance between the z-scored column before and after "
                          "(shape change, blind to pure rescaling); columns ranked by it."),
        "most_changed": [{"column": c, "shape_change": v} for c, v in change[:TABLE_COLUMNS]],
        "least_changed": [{"column": c, "shape_change": v} for c, v in change[-3:]],
        "preview": check_preview(preview),
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nhanes", type=Path, default=DEFAULT_NHANES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    cutoffs = pack_cutoffs()
    with tempfile.TemporaryDirectory(prefix="tt-fixtures-") as tmp:
        work = Path(tmp)
        nhanes, store, nhanes_parquet = load(args.nhanes, work)
        # the in-memory histogram is DataStore.histogram's binning: check it on a stored column
        stored = store.histogram(ENERGY, bins=BINS)
        mine = histogram(nhanes[ENERGY])
        assert stored["counts"] == mine.counts and np.allclose(stored["edges"], mine.edges)
        store.close()
        setup = scenario_a_setup(nhanes, cutoffs)
        energy = scenario_a_energy(nhanes, setup)
        exclusions = scenario_a_exclusions(nhanes, cutoffs)
        findings = scenario_a_findings(nhanes_parquet)
        genomics, gstore, _ = load(GENOMICS, work)
        gstore.close()
        wide = scenario_b(genomics)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    fixture = {
        "meta": {
            "generated_by": str(Path(__file__).resolve().relative_to(ROOT)),
            "generated_on": dt.date.today().isoformat(),
            "base_commit": head.stdout.strip() or None,  # the commit this script was run on
            "contract": "turbotab/core/consequences.py (views), docs/turbotab-next/M1_CONTRACT.md §4",
            "rounding": "every float rounded to 4 significant digits after validation",
            "histograms": f"{BINS} equal-width bins over min..max, as DataStore.histogram",
            "split": ("sklearn train_test_split(test_size=0.2, random_state=0) on the cohort's row "
                      "ids; SEQN is unique, so no grouping (the split stage is not built yet)"),
            "matrix_lane": ("the linear family's matrix: one-hot (drop if binary) for text columns, "
                            "StandardScaler for the rest, fit on training rows"),
            "row_ids": "__row_id from the DuckDB ingest: 0..n-1 in file order",
        },
        "scenario_a": {
            "dataset": {"path": str(args.nhanes), "sha256": sha256(args.nhanes),
                        "n_rows": int(len(nhanes)), "n_cols": int(nhanes.shape[1]),
                        "columns": [str(c) for c in nhanes.columns]},
            "setup": {
                "lens": ["dietary"], "outcome": OUTCOME, "task": "regression", "predictors": PREDICTORS,
                "state": setup["state"].model_dump(mode="json"), "cohort_steps": setup["steps"],
                "n_cohort": int(len(setup["cohort"])), "n_train": int(len(setup["train"])),
                "n_holdout": int(len(setup["holdout"])),
                "exclusion_cutoffs_from_pack": {
                    "source": PACK_02,
                    "sex_specific": {"status": cutoffs["sex_specific"]["status"],
                                     "female": cutoffs["sex_specific"]["female"],
                                     "male": cutoffs["sex_specific"]["male"],
                                     "quote": cutoffs["sex_specific"]["quote"]},
                    "sex_neutral": {"range": cutoffs["neutral"]["range"],
                                    "quote": cutoffs["neutral"]["quote"]},
                },
            },
            "energy_adjustment": energy,
            "exclusions": exclusions,
            "findings": findings,
        },
        "scenario_b": wide,
    }
    fixture = rounded(fixture)
    # re-validate after rounding: the rounded views are still contract-valid
    for option in fixture["scenario_a"]["energy_adjustment"]["options"] + \
            [fixture["scenario_a"]["energy_adjustment"]["partition_on_macronutrient_totals"]]:
        if option["preview"]:
            PreviewResult.model_validate(option["preview"])
        for view in option["extra_views"]:
            VIEWS.validate_python(view)
    for option in fixture["scenario_a"]["exclusions"]["options"]:
        PreviewResult.model_validate(option["preview"])
    PreviewResult.model_validate(fixture["scenario_b"]["preview"])
    text = json.dumps(fixture, indent=1, ensure_ascii=False, allow_nan=False)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(text) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
