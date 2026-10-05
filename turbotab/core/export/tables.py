"""The results tables of the bundle, as CSV and Markdown (V2 definition of done §3.6).

**Inference: Table 2 and its appendix** (MODELING_SEQUENCE §1 row 11; Westreich & Greenland
2013). The exposure's rows across the declared models, the unadjusted one first (STROBE 16a: "Give
unadjusted estimates and, if applicable, confounder-adjusted estimates and their precision"), as the
``effects`` stage served them; every other coefficient in an appendix titled as the stage titles it
("adjustment terms, not effect estimates"). Beside them, when the estimand declares one, the
marginal contrasts, and the sensitivity of each estimate to unmeasured confounding.

**Prediction: the performance table with the declared result** (MODELING_SEQUENCE §1 rows 11–12,
§4). Each family's own scores are listed, labeled "not the result"; the one row labeled "the
result" is what the fit declares (``selection.declared_result``): the selection-corrected estimate
when the families were compared on these rows with nothing held out (BBC-CV, Tsamardinos et al.
2018), a family's own score only when it was declared before any score was seen, or the declared
final model's held-out score once the seal is opened. The winner's own corrected score is never
the result (§4: refused).

Every number is written to the CSV at full precision (Python's shortest round-trip form), so a
replay compares what was reported, digit for digit; the Markdown rounds for reading.
"""
from __future__ import annotations

import csv
import io
import math
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

Kind = Literal["text", "int", "float", "p"]
MINUS = "−"
NUMERIC = ("int", "float", "p")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Column(_Model):
    key: str
    header: str
    kind: Kind = "float"


class Table(_Model):
    name: str  # the file stem under results/
    title: str
    caption: str
    columns: list[Column]
    rows: list[dict[str, Any]]  # each with a "key": the row's stable name
    shown: list[str]  # the Markdown's columns, combined where a header says so (:func:`markdown`)


# ── formatting ───────────────────────────────────────────────────────────────


def _finite(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x: Any) -> str:
    """A number as the Markdown prints it: three significant digits, a true minus."""
    v = _finite(x)
    if v is None:
        return "—"
    if v == 0:
        text = "0"
    elif abs(v) >= 1000:
        text = f"{v:,.0f}"
    elif abs(v) >= 100:
        text = f"{v:.1f}"
    elif abs(v) < 0.001:
        text = f"{v:.2e}"
    else:
        text = f"{v:.3g}"
    return text.replace("-", MINUS)


def fmt_p(x: Any) -> str:
    v = _finite(x)
    if v is None:
        return "—"
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def fmt_int(x: Any) -> str:
    v = _finite(x)
    return "—" if v is None else f"{int(round(v)):,}"


def interval(est: Any, low: Any, high: Any) -> str:
    if _finite(est) is None:
        return "—"
    if _finite(low) is None or _finite(high) is None:
        return fmt(est)
    return f"{fmt(est)} ({fmt(low)} to {fmt(high)})"


def csv_value(x: Any, kind: Kind) -> str:
    if kind == "text":
        return "" if x is None else str(x)
    v = _finite(x)
    if v is None:
        return ""
    if kind == "int":
        return str(int(v)) if float(v).is_integer() else repr(v)
    return repr(v)


# ── rendering ────────────────────────────────────────────────────────────────


def to_csv(table: Table) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["key", *[c.header for c in table.columns]])
    for row in table.rows:
        writer.writerow([row["key"], *[csv_value(row.get(c.key), c.kind) for c in table.columns]])
    return out.getvalue()


def _cell(row: Mapping[str, Any], spec: str, kinds: Mapping[str, Kind]) -> str:
    """One Markdown cell: ``a`` (one column) or ``est|low|high`` (an estimate with its interval)."""
    parts = spec.split("|")
    if len(parts) == 3:
        return interval(*(row.get(p) for p in parts))
    key = parts[0]
    kind = kinds.get(key, "text")
    if kind == "p":
        return fmt_p(row.get(key))
    if kind == "int":
        return fmt_int(row.get(key))
    if kind == "float":
        return fmt(row.get(key))
    value = row.get(key)
    return "—" if value in (None, "") else str(value).replace("|", "\\|").replace("\n", " ")


def markdown(table: Table, headers: Mapping[str, str]) -> str:
    kinds = {c.key: c.kind for c in table.columns}
    lines = [f"### {table.title}", "", table.caption, ""]
    if table.rows:
        lines.append("| " + " | ".join(headers[s] for s in table.shown) + " |")
        lines.append("|" + "|".join("---" for _ in table.shown) + "|")
        for row in table.rows:
            lines.append("| " + " | ".join(_cell(row, s, kinds) for s in table.shown) + " |")
    else:
        lines.append("_No rows._")
    return "\n".join(lines) + "\n"


def estimates(tables: Sequence[Table]) -> dict[str, float | None]:
    """Every reported number, by ``<table>|<row key>|<column>``: what a replay compares."""
    out: dict[str, float | None] = {}
    for table in tables:
        for row in table.rows:
            for c in table.columns:
                if c.kind in NUMERIC:
                    out[f"{table.name}|{row['key']}|{c.key}"] = _finite(row.get(c.key))
    return out


# ── inference: Table 2 ───────────────────────────────────────────────────────

COEF = ("estimate", "ci_low", "ci_high", "se", "df", "p", "ratio", "ratio_low", "ratio_high", "q",
        "fmi", "mc_se")
RATIO_SCALES = ("odds_ratio", "relative_risk_ratio", "hazard_ratio")


def _coef_row(key: str, base: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    out = {"key": key, **base, "term": row.get("feature")}
    out.update({k: row.get(k) for k in COEF})
    return out


def table2(effects: Mapping[str, Any], target: str | None) -> list[Table]:
    """Table 2, its appendix, the marginal contrasts and the sensitivity analyses (module
    docstring), from the ``effects`` stage's artifact as served."""
    seq_rows: list[dict[str, Any]] = []
    app_rows: list[dict[str, Any]] = []
    marg_rows: list[dict[str, Any]] = []
    sens_rows: list[dict[str, Any]] = []
    ratio = False
    for family in effects.get("families") or []:
        fam, label = family.get("family"), family.get("label")
        for s in family.get("sequence") or []:
            inference = s.get("inference") or {}
            ratio = ratio or inference.get("scale") in RATIO_SCALES
            base = {"family": label, "model": s.get("label"),
                    "adjusted_for": ", ".join(s.get("adjusted_for") or []) or "nothing",
                    "n": s.get("n_rows"), "covariance": inference.get("covariance"),
                    "note": s.get("note")}
            if s.get("effects"):
                for r in s["effects"]:
                    seq_rows.append(_coef_row(f"{fam}/{s.get('key')}/{r.get('feature')}", base, r))
            else:
                seq_rows.append({"key": f"{fam}/{s.get('key')}/-", **base, "term": None,
                                 "note": inference.get("refused") or s.get("note")})
            for r in s.get("comparison") or []:
                seq_rows.append(_coef_row(
                    f"{fam}/{s.get('key')}/comparison/{r.get('feature')}",
                    {**base, "model": s.get("comparison_label") or f"{s.get('label')}, comparison",
                     "note": None}, r))
        for d in family.get("diagnostics") or []:
            for r in d.get("change") or []:
                seq_rows.append(_coef_row(
                    f"{fam}/diagnostic/{d.get('check')}/{r.get('feature')}",
                    {"family": label, "model": d.get("change_label") or d.get("check"),
                     "adjusted_for": None, "n": None, "covariance": None,
                     "note": d.get("response")}, r))
        for a in family.get("appendix") or []:
            for t in a.get("terms") or []:
                app_rows.append({**_coef_row(f"{fam}/{a.get('key')}/{t.get('feature')}",
                                             {"family": label, "model": a.get("label")}, t),
                                 "why": t.get("why")})
        marginal = family.get("marginal") or {}
        for c in marginal.get("contrasts") or []:
            marg_rows.append({"key": f"{fam}/marginal/{c.get('setting')}", "family": label,
                              "setting": c.get("setting"),
                              **{k: c.get(k) for k in ("risk_low", "risk_high", "rd", "rd_low",
                                                       "rd_high", "rr", "rr_low", "rr_high",
                                                       "n_boot")}})
        for s in family.get("sensitivity") or []:
            ev = s.get("e_value") or {}
            rv = s.get("robustness") or {}
            sens_rows.append({
                "key": f"{fam}/{s.get('feature')}", "family": label, "term": s.get("feature"),
                "of": s.get("of"), "methods": ", ".join(s.get("methods") or []),
                "e_value": ev.get("point"), "e_value_limit": ev.get("limit"), "rr": ev.get("rr"),
                "rv": rv.get("rv"), "rv_alpha": rv.get("rv_alpha"),
                "partial_r2": rv.get("partial_r2"),
                "reading": s.get("reading") or s.get("not_computed")})
    exposure = effects.get("exposure")
    measure = effects.get("measure_label") or effects.get("measure") or "the declared measure"
    rows_said = effects.get("rows") or ""
    title = effects.get("appendix_title") or "adjustment terms, not effect estimates"
    coef_cols = [Column(key="estimate", header="estimate"), Column(key="ci_low", header="ci_low"),
                 Column(key="ci_high", header="ci_high"), Column(key="se", header="se"),
                 Column(key="df", header="df"), Column(key="p", header="p", kind="p"),
                 Column(key="ratio", header="ratio"), Column(key="ratio_low", header="ratio_low"),
                 Column(key="ratio_high", header="ratio_high"), Column(key="q", header="q", kind="p"),
                 Column(key="fmi", header="fmi"), Column(key="mc_se", header="mc_se")]
    shown_effect = "ratio|ratio_low|ratio_high" if ratio else "estimate|ci_low|ci_high"
    out = [Table(
        name="table2",
        title="Table 2. The exposure's estimate across the declared models",
        caption=(f"The estimate of `{exposure}` on `{target}` as a {measure}, from {rows_said}, "
                 f"unadjusted and in each declared model; Model 2 is the primary. Only the "
                 f"exposure's rows are effect estimates; every other coefficient is listed in the "
                 f"appendix ({title})."),
        columns=[Column(key="family", header="family", kind="text"),
                 Column(key="model", header="model", kind="text"),
                 Column(key="adjusted_for", header="adjusted_for", kind="text"),
                 Column(key="n", header="n", kind="int"),
                 Column(key="term", header="term", kind="text"), *coef_cols,
                 Column(key="covariance", header="covariance", kind="text"),
                 Column(key="note", header="note", kind="text")],
        rows=seq_rows, shown=["model", "adjusted_for", "n", "term", shown_effect, "p"])]
    out.append(Table(
        name="table2_appendix",
        title=f"Appendix to Table 2. {title[:1].upper()}{title[1:]}",
        caption=("Every coefficient of each declared model other than the exposure's, with why it "
                 "is not an effect estimate (Westreich & Greenland 2013)."),
        columns=[Column(key="family", header="family", kind="text"),
                 Column(key="model", header="model", kind="text"),
                 Column(key="term", header="term", kind="text"), *coef_cols,
                 Column(key="why", header="why", kind="text")],
        rows=app_rows, shown=["model", "term", shown_effect, "p", "why"]))
    if marg_rows:
        out.append(Table(
            name="table2_marginal",
            title="Table 2, continued. Marginal contrasts by standardization",
            caption=("Each marginal risk difference and ratio, by g-computation over the analyzed "
                     "rows, with percentile intervals from a bootstrap of the whole chain."),
            columns=[Column(key="family", header="family", kind="text"),
                     Column(key="setting", header="setting", kind="text"),
                     *[Column(key=k, header=k) for k in ("risk_low", "risk_high", "rd", "rd_low",
                                                          "rd_high", "rr", "rr_low", "rr_high")],
                     Column(key="n_boot", header="n_boot", kind="int")],
            rows=marg_rows, shown=["setting", "rd|rd_low|rd_high", "rr|rr_low|rr_high", "n_boot"]))
    if sens_rows:
        out.append(Table(
            name="table2_sensitivity",
            title="Sensitivity of each estimate to unmeasured confounding",
            caption=("The E-value for the estimate and for the confidence limit nearer the null "
                     "(VanderWeele & Ding 2017), and the Cinelli–Hazlett robustness value, never "
                     "read as a pass or a fail."),
            columns=[Column(key="family", header="family", kind="text"),
                     Column(key="term", header="term", kind="text"),
                     Column(key="of", header="of", kind="text"),
                     Column(key="methods", header="methods", kind="text"),
                     *[Column(key=k, header=k) for k in ("e_value", "e_value_limit", "rr", "rv",
                                                          "rv_alpha", "partial_r2")],
                     Column(key="reading", header="reading", kind="text")],
            rows=sens_rows, shown=["term", "e_value", "e_value_limit", "rv", "reading"]))
    return out


TABLE2_HEADERS = {
    "model": "Model", "adjusted_for": "Adjusted for", "n": "n", "term": "Term",
    "estimate|ci_low|ci_high": "Estimate (95% CI)", "ratio|ratio_low|ratio_high": "Ratio (95% CI)",
    "p": "p", "why": "Why it is not an effect", "setting": "Setting",
    "rd|rd_low|rd_high": "Risk difference (95% CI)", "rr|rr_low|rr_high": "Risk ratio (95% CI)",
    "n_boot": "Resamples", "e_value": "E-value", "e_value_limit": "E-value (CI limit)",
    "rv": "Robustness value", "reading": "Reading",
}


# ── prediction: the performance table ────────────────────────────────────────

RESULT = "the result"
NOT_RESULT = "not the result"
BASIS = {
    "selection_corrected": "selection-corrected (BBC-CV)",
    "own_score": "its own score, declared before any score was seen",
    "holdout": "held-out rows, declared final before they were opened",
}


def _metric_order(fit: Mapping[str, Any]) -> list[str]:
    labels = list((fit.get("metric_labels") or {}).keys())
    primary = fit.get("primary_metric")
    head = fit.get("headline_metric")
    order = [m for m in (primary, head) if m]
    return [*order, *[m for m in labels if m not in order]]


def performance(fit: Mapping[str, Any]) -> list[Table]:
    """The performance table with the declared result (module docstring), from the fit as served."""
    labels = dict(fit.get("metric_labels") or {})
    if fit.get("headline_metric") and fit.get("headline_label"):
        labels.setdefault(fit["headline_metric"], fit["headline_label"])
    primary = fit.get("primary_metric")
    order = _metric_order(fit)
    family_label = {m.get("family"): m.get("label") or m.get("family") for m in fit.get("models") or []}
    comparison = fit.get("comparison") or {}
    scheme = f"{fit.get('validation') or 'kfold'}"
    cv_basis = ("repeated k-fold" if scheme == "repeated_kfold" else
                "bootstrap" if scheme == "bootstrap" else
                "internal–external" if scheme == "internal_external" else "k-fold")
    rows: list[dict[str, Any]] = []
    for m in fit.get("models") or []:
        fam = m.get("family")
        for metric in order:
            cv = (m.get("cv") or {}).get(metric)
            if not cv:
                continue
            rows.append({"key": f"{fam}/cv/{metric}", "family": family_label[fam],
                         "score": labels.get(metric, metric),
                         "basis": f"cross-validated ({cv_basis})",
                         "estimate": cv.get("estimate"), "ci_low": cv.get("ci_low"),
                         "ci_high": cv.get("ci_high"), "se": cv.get("se"), "role": NOT_RESULT})
        on = m.get("compared_on")
        if on:
            rows.append({"key": f"{fam}/compared/{primary}", "family": family_label[fam],
                         "score": labels.get(primary, primary),
                         "basis": (f"the comparison: {comparison.get('folds')}-fold repeated "
                                   f"{comparison.get('repeats')} times") if comparison
                         else "the comparison",
                         "estimate": on.get("estimate"), "ci_low": on.get("ci_low"),
                         "ci_high": on.get("ci_high"), "se": on.get("se"), "role": NOT_RESULT})
        for metric, value in (m.get("holdout") or {}).items():
            # The reported result is the score the first opening recorded (audit WP16, RO-05),
            # its own row below; the scores served now are each family's.
            role = NOT_RESULT
            ci = ((m.get("holdout_detail") or {}).get("intervals") or {}).get(metric) or {}
            rows.append({"key": f"{fam}/holdout/{metric}", "family": family_label[fam],
                         "score": labels.get(metric, metric), "basis": "held-out rows",
                         "estimate": value, "ci_low": ci.get("ci_low"),
                         "ci_high": ci.get("ci_high"), "se": ci.get("se"), "role": role})
        cal = m.get("calibration") or {}
        for part in ("intercept", "slope"):
            c = cal.get(part)
            if isinstance(c, Mapping):
                rows.append({"key": f"{fam}/calibration/{part}", "family": family_label[fam],
                             "score": f"calibration {part}", "basis": "out-of-fold predictions",
                             "estimate": c.get("estimate"), "ci_low": c.get("ci_low"),
                             "ci_high": c.get("ci_high"), "se": c.get("se"), "role": NOT_RESULT})
    base = next((m.get("baseline") for m in fit.get("models") or [] if m.get("baseline")), None)
    if base:
        rows.append({"key": f"baseline/cv/{base.get('metric')}",
                     "family": f"No predictors ({base.get('label')})",
                     "score": labels.get(base.get("metric"), base.get("metric")),
                     "basis": f"cross-validated ({cv_basis})", "estimate": base.get("value"),
                     "ci_low": None, "ci_high": None, "se": None, "role": NOT_RESULT})
    result = fit.get("result") or {}
    basis = result.get("basis")
    if basis in ("selection_corrected", "own_score"):
        rows.append({"key": f"result/{basis}/{result.get('metric')}",
                     "family": family_label.get(result.get("family"), result.get("family")),
                     "score": labels.get(result.get("metric"), result.get("metric")),
                     "basis": BASIS[basis], "estimate": result.get("estimate"),
                     "ci_low": result.get("ci_low"), "ci_high": result.get("ci_high"), "se": None,
                     "role": RESULT})
        if basis == "selection_corrected":
            for metric, extra in ((fit.get("selection") or {}).get("extras") or {}).items():
                rows.append({"key": f"result/{basis}/{metric}",
                             "family": family_label.get(result.get("family"), result.get("family")),
                             "score": labels.get(metric, metric), "basis": BASIS[basis],
                             "estimate": extra.get("corrected"),
                             "ci_low": extra.get("corrected_low"),
                             "ci_high": extra.get("corrected_high"), "se": None, "role": RESULT})
    elif basis == "holdout":
        at = fit.get("at_opening") or {}
        final = at.get("family") or fit.get("final_model")
        score = ((at.get("scores") or {}).get(final) or {}).get(primary) if final else None
        if score is not None:
            rows.append({"key": f"result/holdout/{primary}",
                         "family": family_label.get(final, final),
                         "score": labels.get(primary, primary), "basis": BASIS["holdout"],
                         "estimate": score, "ci_low": None, "ci_high": None, "se": None,
                         "role": RESULT})
    caption = result.get("sentence") or "No result is declared."
    return [Table(
        name="performance",
        title="Table 2. Performance of each model family, and the declared result",
        caption=caption,
        columns=[Column(key="family", header="family", kind="text"),
                 Column(key="score", header="score", kind="text"),
                 Column(key="basis", header="basis", kind="text"),
                 Column(key="estimate", header="estimate"), Column(key="ci_low", header="ci_low"),
                 Column(key="ci_high", header="ci_high"), Column(key="se", header="se"),
                 Column(key="role", header="role", kind="text")],
        rows=rows, shown=["family", "score", "basis", "estimate|ci_low|ci_high", "role"])]


PERFORMANCE_HEADERS = {"family": "Family", "score": "Score", "basis": "Basis",
                       "estimate|ci_low|ci_high": "Estimate (95% CI)", "role": "Role"}


def results_tables(source: Any) -> list[Table]:
    """The bundle's results tables for the declared purpose."""
    if source.purpose == "inference":
        effects = source.artifact("effects")
        return table2(effects, getattr(source.state, "target", None)) if isinstance(effects, dict) \
            else []
    fit = source.artifact("fit")
    return performance(fit) if isinstance(fit, dict) else []


def headers_for(table: Table) -> dict[str, str]:
    return PERFORMANCE_HEADERS if table.name == "performance" else TABLE2_HEADERS


__all__ = ["BASIS", "Column", "NOT_RESULT", "PERFORMANCE_HEADERS", "RESULT", "TABLE2_HEADERS",
           "Table", "csv_value", "estimates", "fmt", "fmt_p", "headers_for", "markdown",
           "performance", "results_tables", "table2", "to_csv"]
