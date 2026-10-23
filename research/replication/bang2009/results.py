"""The Bang 2009 replication's results: the engine's runs, the R reference and the paper's numbers.

    scripts/py research/replication/bang2009/results.py           (writes results.json and the
                                                                    notes' results block)
    scripts/py research/replication/bang2009/results.py --check   (both are what the inputs give)

It reads, and writes into no input:

* ``prereg.json``: the factors, the lanes, the variants, the tolerances, the comparison rules;
* the engine's runs (``run.py``'s captures under ``research/data/derived/bang2009/engine/``): under
  the estimate lane each term's row of the export bundle's ``results/table2.csv`` (the exposure)
  and ``results/table2_appendix.csv`` (every other term) for the primary model, since the served
  fit shows the exposure's rows alone under inference; under the predict lane the fit's own
  coefficients (point estimates: no interval is made under prediction) and its AUCs;
* ``reference.json`` (``reference.R``);
* the paper's numbers from the notes (``bang2009.md`` §5.1, §5.2), parsed from their text and
  never retyped: Table 2's odds ratios, CIs and log odds ratios, the AUC in §5.2's heading, the
  counts of §5.1;
* the intake's table (its manifest's SHA-256 checked) and the prep's derived table, for the
  counts by glucose group and the rows with a positive weight.

**Apparent AUC under the estimate lane.** The engine reports no AUC under inference, so the
apparent AUC is computed here from the engine's own coefficients on the engine's own model matrix
(the design stage's, as each run read it from its cache), over the rows the estimate uses (those
with a positive weight under the population answer), by the Mann-Whitney rank sum with ties
counted half. Under the predict lane the engine reports its apparent, optimism-corrected and
cross-validated AUC itself.

``results.json`` holds every number; the notes' results block (between :data:`START` and
:data:`END` in ``bang2009.md``) is rendered from it alone, so each number a reader sees there is
interpolated from it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
PREREG =json.loads((HERE / "prereg.json").read_text("utf-8"))
NOTES = REPO / PREREG["paper"]["notes"]
ENGINE = REPO / "research" / "data" / "derived" / "bang2009" / "engine"
REFERENCE = HERE / "reference.json"
RESULTS = HERE / "results.json"
DERIVED = REPO / "research" / "data" / "derived" / "bang2009" / "derived.parquet"
START = "<!-- results.py writes from here: results.json's numbers, never typed -->"
END = "<!-- results.py writes up to here -->"
PRIMARY_MODEL = "Model 2 (primary)"
INTERCEPT = "(intercept)"


# ── the paper's numbers, from the notes ──────────────────────────────────────


def section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = re.search(r"\n#{2,3} ", text[start + len(heading):])
    return text[start:start + len(heading) + end.start()] if end else text[start:]


def number(s: str) -> float:
    return float(s.replace("−", "-").replace(",", "").strip())


def paper_table2(text: str) -> list[dict[str, Any]]:
    """§5.2's rows in the plan's factor order: each odds ratio, its CI and log odds ratio as the
    notes print them (strings, for their printed precision)."""
    rows = []
    for line in section(text, "### 5.2 ").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        m = re.fullmatch(r"([0-9.]+) \(([0-9.]+)–([0-9.]+)\)", cells[1]) if len(cells) == 5 else None
        if m:
            rows.append({"label": cells[0], "odds_ratio": m.group(1), "ci": [m.group(2), m.group(3)],
                         "p": cells[2], "log_or": cells[3].replace("−", "-"), "points": cells[4]})
    order = PREREG["model"]["factor_order"]
    if len(rows) != len(order):
        raise SystemExit(f"§5.2 holds {len(rows)} factors, the plan {len(order)}")
    return [{"factor": f, **r} for f, r in zip(order, rows)]


def paper_auc(text: str) -> str:
    m = re.search(r"### 5\.2 [^\n]*AUC = ([0-9.]+)\)", text)
    if not m:
        raise SystemExit("§5.2's heading states no AUC")
    return m.group(1)


def paper_counts(text: str) -> dict[str, int]:
    s = section(text, "### 5.1 ")
    out = {}
    for key, pattern in (("in_model", r"N = ([0-9,]+)\.\*\*"), ("normal", r"normal glucose ([0-9,]+)"),
                         ("impaired", r"impaired fasting glucose ([0-9,]+)"),
                         ("undiagnosed", r"undiagnosed diabetes \*\*([0-9,]+)\*\*"),
                         ("known_diabetes", r"Known diabetes \(not in the model\) ([0-9,]+)")):
        m = re.search(pattern, s)
        if not m:
            raise SystemExit(f"§5.1 states no {key}")
        out[key] = int(number(m.group(1)))
    return out


# ── printing as the paper prints ─────────────────────────────────────────────


def places(printed: str) -> int:
    return len(printed.split(".")[1]) if "." in printed else 0


def at(value: float, decimals: int) -> str:
    """``value`` at ``decimals`` places, halves away from zero."""
    q = Decimal(1).scaleb(-decimals)
    return str(Decimal(repr(float(value))).quantize(q, rounding=ROUND_HALF_UP))


def matches(value: float | None, printed: str) -> bool | None:
    if value is None or not math.isfinite(value):
        return None
    return at(value, places(printed)) == at(number(printed), places(printed))


# ── the engine's numbers ─────────────────────────────────────────────────────


def term_key(feature: str) -> str:
    """A model-matrix column as the plan names a term: ``column=level`` (a coded column's level, or
    1 for a yes/no column), or the intercept."""
    if feature == INTERCEPT:
        return INTERCEPT
    m = re.fullmatch(r"(.+)_(\d+)", feature)
    if m and m.group(1) in coded_columns():
        return f"{m.group(1)}={m.group(2)}"
    return f"{feature}=1"


def coded_columns() -> set[str]:
    """Every column an option can name for a predictor with more than two levels."""
    many = {k for k, p in PREREG["model"]["predictors"].items() if len(p["levels"]) > 2}
    cols = {PREREG["model"]["predictors"][k]["column"] for k in many}
    for d in PREREG["decisions"]:
        for o in d["options"].values():
            for k, c in ((o.get("apply") or {}).get("predictors") or {}).items():
                if k in many:
                    cols.add(c)
    return cols


def factor_terms(predictors: dict[str, str]) -> dict[str, str]:
    """Each factor of Table 2 as the run's term (its predictor's column, as the run names it)."""
    column_of = {p["column"]: k for k, p in PREREG["model"]["predictors"].items()}
    out = {}
    for factor, (column, level) in PREREG["model"]["factors"].items():
        out[factor] = f"{predictors[column_of[column]]}={level}"
    return out


def bundle_rows(name: str) -> list[dict[str, str]]:
    z = zipfile.ZipFile(ENGINE / "bundles" / f"{name}.zip")
    rows = []
    for member in ("results/table2.csv", "results/table2_appendix.csv"):
        path = next(n for n in z.namelist() if n.endswith(member))
        rows += list(csv.DictReader(io.StringIO(z.read(path).decode("utf-8"))))
    return [r for r in rows if r["model"] == PRIMARY_MODEL]


def capture(name: str) -> dict[str, Any]:
    c = json.loads((ENGINE / "runs" / f"{name}.json").read_text("utf-8"))
    if c.get("error"):
        raise SystemExit(f"{name} stopped: {c['error'][:300]}")
    return c


def floats(r: dict[str, Any], *keys: str) -> dict[str, float | None]:
    return {k: (None if r.get(k) in (None, "") else float(r[k])) for k in keys}


def engine_terms(lane: str, name: str, c: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if lane == "estimate":
        rows = bundle_rows(name)
        return {term_key(r["term"]): floats(r, "estimate", "se", "ci_low", "ci_high") for r in rows}
    return {term_key(r["feature"]): floats(r, "estimate", "se", "ci_low", "ci_high")
            for r in c["fit_model"]["coefficients"]}


def rank_auc(y: list[int], score: list[float]) -> float:
    """The Mann-Whitney AUC, ties counted half (average ranks)."""
    from scipy.stats import rankdata

    ranks = rankdata(score)
    n1 = sum(y)
    n0 = len(y) - n1
    s = sum(r for r, v in zip(ranks, y) if v == 1)
    return float((s - n1 * (n1 + 1) / 2) / (n1 * n0))


def positive_weight(seqn: list[int]) -> list[bool]:
    """Whether each row has a positive six-year fasting weight (2/3 WTSAF4YR for 1999-2002,
    1/3 WTSAF2YR after), read from the derived table."""
    import pandas as pd

    t = pd.read_parquet(DERIVED, columns=["SEQN", "SDDSRVYR", "WTSAF2YR", "WTSAF4YR"]).set_index("SEQN")
    t = t.loc[seqn]
    w = t["WTSAF4YR"].where(t["SDDSRVYR"].isin([1, 2]), t["WTSAF2YR"])
    return [bool(x > 0) for x in w.fillna(0)]


def apparent_auc(c: dict[str, Any], terms: dict[str, dict[str, Any]], weighted: bool) -> dict[str, Any]:
    rows = c["rows"]
    keep = positive_weight(rows["seqn"]) if weighted else [True] * rows["n"]
    eta = []
    for i in range(rows["n"]):
        if not keep[i]:
            continue
        v = terms[INTERCEPT]["estimate"]
        for col in rows["matrix_columns"]:
            v += terms[term_key(col)]["estimate"] * rows["matrix"][col][i]
        eta.append(v)
    y = [o for o, k in zip(rows["outcome"], keep) if k]
    return {"value": rank_auc(y, eta), "rows": len(y),
            "source": "computed here from the engine's coefficients on its model matrix"}


def agree(terms: dict[str, dict[str, Any]], ref_terms: dict[str, dict[str, Any]], *,
          rows_same: bool, n_analyzed: tuple[int, int], n_estimate: tuple[int, int],
          auc: tuple[float, float], tolerance: dict[str, Any], auc_tolerance: float) -> dict[str, Any]:
    """The engine's fit against the reference's, by ``prereg.json``'s tolerance: the same
    participants and counts, every log odds ratio (the intercept's included) and, where the engine
    gives one, every standard error, and the apparent AUC. A term the reference has and the engine
    lacks is a disagreement."""
    missing = sorted(set(ref_terms) - set(terms))
    d_log = {k: abs(terms[k]["estimate"] - ref_terms[k]["log_or"]) for k in ref_terms if k in terms}
    d_se = {k: abs(terms[k]["se"] - ref_terms[k]["se"]) / ref_terms[k]["se"]
            for k in d_log if terms[k].get("se") is not None}
    out = {"rows_same": bool(rows_same), "n_analyzed": list(n_analyzed), "n_estimate": list(n_estimate),
           "terms_missing": missing,
           "max_abs_log_or": max(d_log.values()) if d_log else None,
           "tolerance_log_or": tolerance["log_odds_ratio_abs"],
           "max_rel_se": max(d_se.values()) if d_se else None,
           "tolerance_se": tolerance.get("standard_error_rel"),
           "auc_abs": abs(auc[0] - auc[1]), "tolerance_auc": auc_tolerance}
    out["within"] = bool(
        rows_same and not missing and n_analyzed[0] == n_analyzed[1] and n_estimate[0] == n_estimate[1]
        and out["max_abs_log_or"] is not None and out["max_abs_log_or"] <= out["tolerance_log_or"]
        and (out["max_rel_se"] is None or out["tolerance_se"] is None
             or out["max_rel_se"] <= out["tolerance_se"])
        and out["auc_abs"] <= auc_tolerance)
    return out


def solver_check(c: dict[str, Any], ref_terms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Under the predict lane: the engine's coefficients against scikit-learn's LogisticRegression
    as the engine builds it (no penalty, newton-cholesky, its default tolerance) and run to a tight
    tolerance, each on the engine's own model matrix, and the tight fit against the reference."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rows = c["rows"]
    cols = rows["matrix_columns"]
    X = np.column_stack([rows["matrix"][k] for k in cols])
    y = np.asarray(rows["outcome"])
    engine = {term_key(r["feature"]): r["estimate"] for r in c["fit_model"]["coefficients"]}
    keys = [INTERCEPT, *(term_key(k) for k in cols)]
    e = np.array([engine[k] for k in keys])
    r = np.array([ref_terms[k]["log_or"] for k in keys])
    out = {}
    for label, tol in (("default", None), ("tight", 1e-12)):
        kw = {} if tol is None else {"tol": tol}
        m = LogisticRegression(C=np.inf, solver="newton-cholesky", max_iter=1000, **kw).fit(X, y)
        b = np.r_[m.intercept_, m.coef_.ravel()]
        out[label] = {"tol": m.get_params()["tol"], "iterations": int(m.n_iter_[0]),
                      "max_abs_from_engine": float(np.max(np.abs(b - e))),
                      "max_abs_from_reference": float(np.max(np.abs(b - r)))}
    return out


def provenance() -> dict[str, Any]:
    """The derived table every run read: its SHA-256 now, and each run's (the driver's record)."""
    now = hashlib.sha256(DERIVED.read_bytes()).hexdigest()
    seen = {}
    for p in sorted((ENGINE / "runs").glob("*.json")):
        if p.name.endswith(".walk.json"):
            continue
        c = json.loads(p.read_text("utf-8"))
        meta = json.loads((ENGINE / "homes" / c["run"] / "driver.json").read_text("utf-8"))
        seen[p.stem] = meta["data"]["sha256"]
    return {"derived_sha256": now, "every_run_read_it": all(v == now for v in seen.values()),
            "runs": len(seen)}


def predictors_against_reference() -> dict[str, Any]:
    """The prep's predictor columns beside reference.R's own, row by row on the subsample."""
    import pandas as pd

    from research.replication.bang2009 import prep

    mine = pd.read_parquet(DERIVED, columns=["SEQN", *prep.PREDICTORS]).set_index("SEQN")
    theirs = pd.read_csv(REPO / "research" / "data" / "derived" / "bang2009" / "reference"
                         / "predictors.csv").set_index("SEQN").loc[mine.index]
    differ = {}
    for col in prep.PREDICTORS:
        a, b = mine[col].astype("Float64"), theirs[col].astype("Float64")
        same = (a == b).fillna(False) | (a.isna() & b.isna())
        differ[col] = int((~same).sum())
    return {"rows": int(len(mine)), "rows_differing": differ}


def seqn_sha(seqn: list[int]) -> str:
    return hashlib.sha256("\n".join(str(int(s)) for s in sorted(seqn)).encode()).hexdigest()


# ── the counts by glucose group ──────────────────────────────────────────────


def counts_here() -> dict[str, int]:
    """§5.1's groups from the intake's table: aged 20 or older, not pregnant, a positive fasting
    weight and a glucose result; known diabetes a yes to DIQ010, DIQ050 or DIQ070; the rest by
    fasting glucose (under 100, 100 to 125, 126 or more mg/dL)."""
    import pandas as pd

    manifest = json.loads((REPO / "research" / "datasets" / "bang2009" / "manifest.json").read_text())
    entry = manifest["table"]
    path = REPO / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"{entry['file']} is not the manifest's table")
    t = pd.read_parquet(path)
    rules = PREREG["eligibility"]["rules"]
    age, fasting = rules["age"], rules["fasting"]
    base = (t["LBXGLU"].notna() & (t[age["column"]] >= age["low"])
            & ~t[rules["pregnant"]["column"]].eq(1).fillna(False)
            & (t[fasting["column"]] >= fasting["low"]).fillna(False))
    known = (t["DIQ010"].eq(1) | t["DIQ050"].eq(1) | t["DIQ070"].eq(1)).fillna(False)
    g = t["LBXGLU"]
    rest = base & ~known
    return {"known_diabetes": int((base & known).sum()), "normal": int((rest & (g < 100)).sum()),
            "impaired": int((rest & (g >= 100) & (g < 126)).sum()),
            "undiagnosed": int((rest & (g >= 126)).sum()), "eligible": int(rest.sum())}


# ── assembling ───────────────────────────────────────────────────────────────


def build() -> dict[str, Any]:
    from research.replication.bang2009 import run as R

    text = NOTES.read_text("utf-8")
    ref = json.loads(REFERENCE.read_text("utf-8"))
    tol = PREREG["agreement"]
    runs: dict[str, Any] = {}
    for full in R.variants():
        lane, variant = full.split(":")
        name = f"{lane}-{variant}"
        c = capture(name)
        s = R.settings(lane, variant)
        terms = engine_terms(lane, name, c)
        f_terms = factor_terms(s["predictors"])
        rf = ref["fits"][full]
        ref_terms = {INTERCEPT if t["column"] == INTERCEPT else f"{t['column']}={t['level']}": t
                     for t in rf["terms"]}
        weighted = lane == "estimate" and s["survey_answer"] == "population"
        model = c["fit_model"]
        if lane == "estimate":
            auc = {"apparent": apparent_auc(c, terms, weighted)}
        else:
            opt = ((model.get("optimism") or {}).get("estimates") or {}).get("auc") or {}
            auc = {"apparent": {"value": opt.get("apparent"), "source": "the engine's (Harrell's bootstrap)"},
                   "optimism_corrected": opt.get("corrected"),
                   "cross_validated": ((model.get("cv") or {}).get("auc") or {}).get("estimate")}
            solver = solver_check(c, ref_terms)
        lane_tol = tol["estimate_lane" if lane == "estimate" else "predict_lane"]
        n_estimate = (int(sum(positive_weight(c["rows"]["seqn"]))) if weighted else c["rows"]["n"])
        agreement = agree(terms, ref_terms,
                          rows_same=seqn_sha(c["rows"]["seqn"]) == rf["seqn_sha256"],
                          n_analyzed=(c["rows"]["n"], rf["n_analyzed"]),
                          n_estimate=(n_estimate, rf["n_estimate"]),
                          auc=(auc["apparent"]["value"], rf["auc_apparent"]),
                          tolerance=lane_tol, auc_tolerance=lane_tol.get("auc_abs", tol["auc_abs"]))
        factors = {}
        for factor, key in f_terms.items():
            t, r = terms[key], ref_terms[key]
            factors[factor] = {
                "term": key, "log_or": t["estimate"], "se": t.get("se"),
                "ci_low": t.get("ci_low"), "ci_high": t.get("ci_high"),
                "odds_ratio": math.exp(t["estimate"]),
                "or_low": math.exp(t["ci_low"]) if t.get("ci_low") is not None else None,
                "or_high": math.exp(t["ci_high"]) if t.get("ci_high") is not None else None,
                "reference": {"log_or": r["log_or"], "se": r["se"], "ci_low": r["ci_low"],
                              "ci_high": r["ci_high"], "se_subset": r.get("se_subset")}}
        runs[full] = {"lane": lane, "variant": variant, "chosen": s["chosen"],
                      "survey": s["survey_answer"], "n_analyzed": c["rows"]["n"],
                      "n_estimate": n_estimate, "seconds": c["seconds"], "auc": auc,
                      "reference_auc": {"apparent": rf["auc_apparent"],
                                        "apparent_weighted": rf.get("auc_apparent_weighted")},
                      "factors": factors, "agreement": agreement,
                      "estimator": (model.get("inference") or {}).get("estimator")
                      if lane == "predict" else bundle_estimator(name),
                      **({"solver": solver} if lane == "predict" else {})}
    paper = paper_table2(text)
    for r in runs.values():
        r["table2_matches"] = printed_matches(r, paper, paper_auc(text))
    return {"about": ("The Bang 2009 replication's numbers, written by results.py from the engine's "
                      "runs, reference.json and the notes' §5; the notes' results block is rendered "
                      "from this file."),
            "paper": {"table2": paper, "auc": paper_auc(text), "counts": paper_counts(text)},
            "obesity_absent_rows": added_rows(runs, "obesity_absent"),
            "counts": {"here": counts_here(), "reference": ref["counts"],
                       "engine_flow": flow_of(capture("estimate-default"))},
            "provenance": provenance(), "predictors": predictors_against_reference(),
            "reference_software": {k: ref[k] for k in ("r", "survey", "pROC", "haven")},
            "runs": runs, "table2": table2(runs, paper_table2(text)),
            "moves": moves(runs), "agreement": overall(runs)}


def bundle_estimator(name: str) -> str | None:
    z = zipfile.ZipFile(ENGINE / "bundles" / f"{name}.zip")
    path = next(n for n in z.namelist() if n.endswith("results/table2.csv"))
    rows = [r for r in csv.DictReader(io.StringIO(z.read(path).decode("utf-8")))
            if r["model"] == PRIMARY_MODEL]
    return rows[0].get("covariance") if rows else None


def flow_of(c: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"label": s["label"], "n": s["n"], "dropped": s["dropped"]} for s in c["flow"]]


def table2(runs: dict[str, Any], paper: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in paper:
        row = {"factor": p["factor"], "paper": p}
        for name in ("estimate:default", "estimate:unweighted", "predict:default"):
            f = runs[name]["factors"][p["factor"]]
            row[name] = {"odds_ratio": f["odds_ratio"], "or_low": f["or_low"], "or_high": f["or_high"],
                         "log_or": f["log_or"],
                         "matches": {"odds_ratio": matches(f["odds_ratio"], p["odds_ratio"]),
                                     "ci_low": matches(f["or_low"], p["ci"][0]),
                                     "ci_high": matches(f["or_high"], p["ci"][1]),
                                     "log_or": matches(f["log_or"], p["log_or"])}}
        out.append(row)
    return out


def printed_matches(r: dict[str, Any], paper: list[dict[str, Any]], auc: str) -> dict[str, Any]:
    """How many of Table 2's printed values the run's fit gives at the paper's precision."""
    ors = cis = logs = 0
    for p in paper:
        f = r["factors"][p["factor"]]
        ors += bool(matches(f["odds_ratio"], p["odds_ratio"]))
        cis += bool(matches(f["or_low"], p["ci"][0]) and matches(f["or_high"], p["ci"][1]))
        logs += bool(matches(f["log_or"], p["log_or"]))
    return {"odds_ratio": ors, "ci": cis, "log_or": logs, "of": len(paper),
            "auc": bool(matches(r["auc"]["apparent"]["value"], auc))}


def added_rows(runs: dict[str, Any], variant: str) -> dict[str, int]:
    """The rows a variant's fit holds that the default's does not, and the cases among them."""
    base = capture("estimate-default")["rows"]
    other = capture(f"estimate-{variant}")["rows"]
    have = set(base["seqn"])
    extra = [y for s, y in zip(other["seqn"], other["outcome"]) if s not in have]
    # the default's cases in obesity's reference level: no indicator column of the level is 1
    levels = [c for c in base["matrix_columns"] if re.fullmatch(r"obesity_\d+", c)]
    at_reference = [y for i, y in enumerate(base["outcome"])
                    if all(base["matrix"][c][i] == 0 for c in levels)]
    return {"rows": len(extra), "cases": int(sum(extra)),
            "default_cases": int(sum(base["outcome"])), "default_rows": base["n"],
            "default_reference_level_cases": int(sum(at_reference)),
            "default_reference_level_rows": len(at_reference)}


def moves(runs: dict[str, Any]) -> dict[str, Any]:
    """Each variant's change from its lane's default, per factor: the change in the log odds
    ratio; under the estimate lane whether it is material (at least half the default's 95% CI
    half-width on the log scale) and whether the odds ratio at the paper's one decimal moves."""
    out: dict[str, Any] = {}
    for full, r in runs.items():
        lane, variant = full.split(":")
        if variant == "default":
            continue
        base = runs[f"{lane}:default"]
        (decision, option), = PREREG["sweep"]["variants"][variant].items()
        per = {}
        for factor, f in r["factors"].items():
            b = base["factors"][factor]
            delta = f["log_or"] - b["log_or"]
            half = ((b["ci_high"] - b["ci_low"]) / 2) if b.get("ci_high") is not None else None
            per[factor] = {"delta_log_or": delta,
                           "material": (abs(delta) >= half / 2) if half is not None else None,
                           "printed_change": at(f["odds_ratio"], 1) != at(b["odds_ratio"], 1)}
        out[full] = {"decision": decision, "option": option, "n_analyzed": r["n_analyzed"],
                     "n_estimate": r["n_estimate"],
                     "auc_change": r["auc"]["apparent"]["value"] - base["auc"]["apparent"]["value"],
                     "factors": per}
    return out


def overall(runs: dict[str, Any]) -> dict[str, Any]:
    by_lane = {}
    for lane in ("estimate", "predict"):
        rs = [r for r in runs.values() if r["lane"] == lane]
        by_lane[lane] = {"runs": len(rs), "within": sum(r["agreement"]["within"] for r in rs),
                         "rows_same": sum(r["agreement"]["rows_same"] for r in rs),
                         "max_abs_log_or": max(r["agreement"]["max_abs_log_or"] for r in rs),
                         "max_rel_se": max((r["agreement"]["max_rel_se"] or 0) for r in rs),
                         "max_auc_abs": max(r["agreement"]["auc_abs"] for r in rs)}
    return {"by_lane": by_lane, "all_within": all(r["agreement"]["within"] for r in runs.values())}


# ── the notes' results block ─────────────────────────────────────────────────


def ci(lo: float | None, hi: float | None, d: int = 2) -> str:
    return "—" if lo is None else f"{at(lo, d)}–{at(hi, d)}"


def mark(ok: bool | None) -> str:
    return {True: "yes", False: "**no**", None: "—"}[ok]


def render(res: dict[str, Any]) -> str:
    runs, paper = res["runs"], res["paper"]
    d, u, p = runs["estimate:default"], runs["estimate:unweighted"], runs["predict:default"]
    counts, here = paper["counts"], res["counts"]["here"]
    L: list[str] = []
    theirs = res["counts"]["reference"]
    same = all(theirs.get(k) == v for k, v in here.items())
    L.append(f"Development sample (the default eligibility, counted by `results.py` from the "
             f"intake's table; {'R counts the same' if same else '**R counts otherwise**: ' + json.dumps({k: theirs.get(k) for k in here})}): "
             f"normal fasting glucose {here['normal']:,} (paper {counts['normal']:,}), impaired "
             f"{here['impaired']:,} ({counts['impaired']:,}), undiagnosed diabetes "
             f"{here['undiagnosed']:,} ({counts['undiagnosed']:,}), {here['eligible']:,} in all "
             f"({counts['in_model']:,}); known diabetes, left out, {here['known_diabetes']:,} "
             f"({counts['known_diabetes']:,}). The model's complete cases: {d['n_analyzed']:,}.")
    L.append("")
    L.append("**Table 2 against the engine.** The weighted fit is the estimate lane's default (the "
             "primary, pre-registered comparison); the unweighted fits are the estimate lane's "
             "sample answer (with model-based intervals) and the predict lane's fit (point "
             "estimates only). \"Match\" is the paper's printed precision, halves away from zero: "
             "the OR and each CI bound at one decimal, the log OR at two.")
    L.append("")
    L.append("| Factor | Paper OR (95% CI) | Paper log OR | Engine weighted OR (95% CI) | log OR | "
             "OR / CI / log OR match | Unweighted OR (95% CI) | log OR | Predict lane OR |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for row in res["table2"]:
        pp, e, w, q = row["paper"], row["estimate:default"], row["estimate:unweighted"], row["predict:default"]
        m = e["matches"]
        L.append(f"| {row['factor']} | {pp['odds_ratio']} ({pp['ci'][0]}–{pp['ci'][1]}) | "
                 f"{pp['log_or']} | {at(e['odds_ratio'], 2)} ({ci(e['or_low'], e['or_high'])}) | "
                 f"{at(e['log_or'], 3)} | {mark(m['odds_ratio'])} / {mark(m['ci_low'] and m['ci_high'])} "
                 f"/ {mark(m['log_or'])} | {at(w['odds_ratio'], 2)} ({ci(w['or_low'], w['or_high'])}) | "
                 f"{at(w['log_or'], 3)} | {at(q['odds_ratio'], 2)} |")
    n_or = sum(bool(r["estimate:default"]["matches"]["odds_ratio"]) for r in res["table2"])
    n_ci = sum(bool(r["estimate:default"]["matches"]["ci_low"] and r["estimate:default"]["matches"]["ci_high"])
               for r in res["table2"])
    n_u = sum(bool(r["estimate:unweighted"]["matches"]["odds_ratio"]) for r in res["table2"])
    L.append("")
    L.append(f"The weighted fit matches {n_or} of the {len(res['table2'])} printed odds ratios and "
             f"both CI bounds of {n_ci}; the unweighted fit matches {n_u} odds ratios.")
    L.append("")
    pa = d["auc"]["apparent"]["value"]
    L.append(f"**AUC.** The paper's development AUC is {paper['auc']}. The weighted fit's apparent AUC "
             f"on its {d['auc']['apparent']['rows']:,} rows, unweighted, computed from the engine's "
             f"coefficients: {at(pa, 3)} (R: {at(d['reference_auc']['apparent'], 3)}; weighted by the "
             f"fasting weight, R: {at(d['reference_auc']['apparent_weighted'], 3)}). The predict "
             f"lane's, as the engine reports it: apparent {at(p['auc']['apparent']['value'], 3)}, "
             f"optimism-corrected {at(p['auc']['optimism_corrected'], 3)} (Harrell's bootstrap), "
             f"cross-validated {at(p['auc']['cross_validated'], 3)} (5-fold).")
    L.append("")
    L.append("**The unstated decisions, one at a time.** Each cell is the change in the log odds "
             "ratio from the lane's default fit; a star marks a change of at least half the "
             "default's 95% CI half-width on the log scale (material, SWARM_PLAN §7), and a dagger "
             "one that moves the odds ratio at the paper's one decimal. The predict lane has no "
             "interval, so no star.")
    L.append("")
    short = {f: i + 1 for i, f in enumerate(PREREG["model"]["factor_order"])}
    L.append("Factors: " + "; ".join(f"{i} {f}" for f, i in short.items()) + ".")
    L.append("")
    head = " | ".join(str(i) for i in short.values())
    L.append(f"| Run | Option | n | ΔAUC | {head} | Table 2 matched: OR, CI, log OR |")
    L.append("|---|---|---|---|" + "---|" * len(short) + "---|")

    def matched(name: str) -> str:
        m = runs[name]["table2_matches"]
        return f"{m['odds_ratio']}, {m['ci']}, {m['log_or']} of {m['of']}"

    for lane in ("estimate", "predict"):
        name = f"{lane}:default"
        L.append(f"| {name} | the defaults | {runs[name]['n_estimate']:,} | — | "
                 + " | ".join("—" for _ in short) + f" | {matched(name)} |")
    for full, mv in res["moves"].items():
        cells = []
        for f in short:
            x = mv["factors"][f]
            cells.append(f"{x['delta_log_or']:+.3f}" + ("*" if x["material"] else "")
                         + ("†" if x["printed_change"] else ""))
        L.append(f"| {full} | {mv['decision']}: {mv['option']} | {mv['n_estimate']:,} | "
                 f"{mv['auc_change']:+.3f} | " + " | ".join(cells) + f" | {matched(full)} |")
    best = max((n for n in runs if n.startswith("estimate:")),
               key=lambda n: (runs[n]["table2_matches"]["odds_ratio"], runs[n]["table2_matches"]["log_or"]))
    bm, oa = runs[best]["table2_matches"], res["obesity_absent_rows"]
    L.append("")
    L.append(f"Of the estimate lane's fits, `{best}` gives the most printed values: {bm['odds_ratio']} "
             f"of the {bm['of']} odds ratios, both CI bounds of {bm['ci']}, {bm['log_or']} log odds "
             f"ratios, and its apparent AUC {'rounds' if bm['auc'] else 'does not round'} to the "
             f"paper's. Counting a "
             f"participant whose BMI and waist are both missing as not overweight adds "
             f"{oa['rows']:,} rows to the default's {oa['default_rows']:,}, {oa['cases']:,} of them "
             f"undiagnosed diabetes, all in obesity's reference level, which held "
             f"{oa['default_reference_level_cases']:,} of the default's {oa['default_cases']:,} cases "
             f"(in {oa['default_reference_level_rows']:,} rows).")
    material = sorted({(mv["decision"], mv["option"]) for full, mv in res["moves"].items()
                       if full.startswith("estimate:") and any(x["material"] for x in mv["factors"].values())})
    L.append("")
    L.append("Material under the estimate lane: " + (", ".join(f"{a} ({b})" for a, b in material) or "none")
             + ".")
    fa = res["moves"]["estimate:fasting_any_glucose"]
    largest = max(abs(x["delta_log_or"]) for x in fa["factors"].values())
    L.append(f"Without the fasting rule the weighted fit, predicted to move no odds ratio, moves "
             + ("none, to the last digit" if largest == 0 else f"one by {largest:.1e} on the log scale")
             + f": {fa['n_analyzed']:,} rows are analyzed and the estimate's are still "
             f"{fa['n_estimate']:,}.")
    L.append("")
    a = res["agreement"]["by_lane"]
    e_l, p_l = a["estimate"], a["predict"]
    tol = PREREG["agreement"]
    L.append(f"**The engine against the R reference** (`reference.R`: survey::svyglm and glm on rows "
             f"it builds itself from the transport files). Estimate lane: {e_l['within']} of "
             f"{e_l['runs']} fits within the pre-registered tolerance, the same participants in "
             f"{e_l['rows_same']}; the largest difference in a log odds ratio "
             f"{e_l['max_abs_log_or']:.1e} (tolerance {tol['estimate_lane']['log_odds_ratio_abs']:.0e}), "
             f"in a standard error {e_l['max_rel_se']:.1e} relative "
             f"({tol['estimate_lane']['standard_error_rel']:.0e}), in the apparent AUC "
             f"{e_l['max_auc_abs']:.1e} ({tol['auc_abs']:.0e}). Predict lane: {p_l['within']} of "
             f"{p_l['runs']} within, the same participants in {p_l['rows_same']}; the largest "
             f"difference in a log odds ratio {p_l['max_abs_log_or']:.1e} (tolerance "
             f"{tol['predict_lane']['log_odds_ratio_abs']:.0e}: the prediction fit stops at "
             f"scikit-learn's default tolerance), in the apparent AUC {p_l['max_auc_abs']:.1e} "
             f"({tol['predict_lane']['auc_abs']:.0e}).")
    outside = [n for n, r in runs.items() if not r["agreement"]["within"]]
    if outside:
        L.append("Outside it: " + "; ".join(
            f"`{n}` ({'the same participants' if runs[n]['agreement']['rows_same'] else 'other participants'}, "
            f"log odds ratios up to {runs[n]['agreement']['max_abs_log_or']:.1e} apart)" for n in outside)
                 + ".")
    worst = max((r for r in runs.values() if r["lane"] == "predict"),
                key=lambda r: r["agreement"]["max_abs_log_or"])
    sv = worst["solver"]
    L.append(f"The predict lane's distance is the solver's: on the engine's own model matrix, "
             f"scikit-learn's LogisticRegression as the engine builds it (no penalty, newton-cholesky, "
             f"tolerance {sv['default']['tol']:g}) stops after {sv['default']['iterations']} "
             f"iterations at the engine's coefficients ("
             + ("exactly" if sv["default"]["max_abs_from_engine"] == 0
                else f"within {sv['default']['max_abs_from_engine']:.1e}") + "), "
             f"and run to a tolerance of {sv['tight']['tol']:g} it takes {sv['tight']['iterations']} "
             f"and lands within {sv['tight']['max_abs_from_reference']:.1e} of R's (the run "
             f"`{worst['lane']}:{worst['variant']}`, the farthest).")
    sub = [abs(f["reference"]["se_subset"] - f["reference"]["se"]) / f["reference"]["se"]
           for f in d["factors"].values() if f["reference"].get("se_subset") is not None]
    L.append("In R the standard errors of a design over the analysis rows alone (a subset, §6 "
             "choice 7) are " + ("the domain's, to the last digit" if max(sub) == 0 else
                                 f"within {max(sub):.1e} relative of the domain's") + ".")
    pr, pv = res["predictors"], res["provenance"]
    differ = sum(pr["rows_differing"].values())
    L.append(f"The prep's ten predictor columns, set beside the ones `reference.R` derives itself "
             f"from the transport files, differ on {differ} of the {pr['rows']:,} rows' values; "
             f"every one of the {pv['runs']} runs read the same derived table "
             f"({'yes' if pv['every_run_read_it'] else '**no**'}).")
    return "\n".join(L)


def block(res: dict[str, Any]) -> str:
    return f"{START}\n\n{render(res)}\n\n{END}"


def with_block(text: str, res: dict[str, Any]) -> str:
    if START not in text or END not in text:
        raise SystemExit(f"{NOTES.name} has no results block ({START} … {END})")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return head + block(res) + tail


def dumped(res: dict[str, Any]) -> str:
    return json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    res = build()
    notes = NOTES.read_text("utf-8")
    if args.check:
        stale = [p.name for p, now in ((RESULTS, dumped(res)), (NOTES, with_block(notes, res)))
                 if p.read_text("utf-8") != now]
        print("stale: " + ", ".join(stale) if stale else "results.json and the notes' block are current")
        return 1 if stale else 0
    RESULTS.write_text(dumped(res), "utf-8")
    NOTES.write_text(with_block(notes, res), "utf-8")
    print(json.dumps(res["agreement"]))
    return 0 if res["agreement"]["all_within"] else 1


if __name__ == "__main__":
    sys.exit(main())
