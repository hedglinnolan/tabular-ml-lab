"""What each finding says in one line, where its lever is, and the findings the app adds itself.

M1_CONTRACT §6 and BLUEPRINT §11.7: a finding is a one-line claim plus its lever. Each finding
gains

* ``summary`` — at most 20 words, their data first (a count, a column, a correlation);
* ``routes_to`` — the interview question that acts on it, or None;
* ``lever_label`` — at most 5 words saying what that question will do, or None;
* ``group`` — a pager key shared by same-kind findings (set only when two or more share it).

A finding with no lever in this version says so in its summary (:data:`NO_LEVER`) instead of
pretending. Routes in M1: energy adjustment → ``energy_adjustment``; implausible or impossible
intakes → ``exclusions``; identifiers, survey design and flags → ``roles``.

The legacy detectors keep their ids, details and badges; this module only speaks for them. Where a
legacy sentence is untrue of the table (``pack::dietary::energy_adjustment`` counts every numeric
column as a candidate nutrient), the detail is restated from the data.

The app's own findings (``voice::*``) cover what the drive rubric says NHANES users were never told
(DRIVE_RUBRIC §4): the respondent identifier, the ``imputed_*`` flags and the columns they mark,
the survey design that is missing, and the survey cycles pooled in one table.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cached_property
from typing import AbstractSet, Any, Callable, Mapping, Sequence

import pandas as pd

from turbotab.core.voice import count, finish, listing, number, plural, tick

SUMMARY_WORDS = 20
LEVER_WORDS = 5
NO_LEVER = "No control for this yet."

NUT01 = "research/NUTRITION_PACK.md#01 · Import and structural recognition"


@dataclass
class FindingContext:
    """What a summary may read: the table the findings were computed on, and the answers so far."""

    frame: pd.DataFrame
    lens: Sequence[str] = ()
    target: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @cached_property
    def columns(self) -> list[str]:
        return [str(c) for c in self.frame.columns]

    @cached_property
    def names(self) -> frozenset[str]:
        return frozenset(self.columns)


def family(finding_id: str) -> str:
    """``binary_text__gender`` → ``binary_text``; ``pack::dietary::atwater#2`` → the pack id."""
    base = str(finding_id).split("#", 1)[0]
    return base.split("__", 1)[0]


def _list(value: Any) -> list[Any]:
    """Params sometimes arrive as a list, sometimes as its repr; read either."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        import ast

        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def _col(entry: Any) -> str | None:
    if isinstance(entry, Mapping):
        return entry.get("column") or entry.get("analyte") or entry.get("item")
    return str(entry) if entry is not None else None


def _blank_share(fc: FindingContext, column: str | None) -> tuple[int, float]:
    if not column or column not in fc.frame.columns or not fc.n_rows:
        return 0, 0.0
    n = int(fc.frame[column].isna().sum())
    return n, n / fc.n_rows


# ── the family table ─────────────────────────────────────────────────────────

Say = Callable[[dict[str, Any], dict[str, Any], FindingContext], "Voice | None"]


@dataclass
class Voice:
    summary: str
    routes_to: str | None = None
    lever: str | None = None
    closes: bool = False  # the summary already says there is nothing to act on
    title: str | None = None  # a restated title, where the legacy one has "(s)" grammar


BLANKS = ("missing", "Choose how blanks are handled")


def _two_level_text(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    col = p.get("column") or (f["affected_columns"] or [None])[0]
    n_blank, share = _blank_share(fc, col)
    if p.get("written_as") == "bool_dtype":
        return Voice(f"{tick(col)} is already a true/false column; nothing needs repairing.",
                     closes=True)
    if share >= 0.1:
        return Voice(f"{tick(col)} is two-level text with {count(n_blank)} of {count(fc.n_rows)} "
                     f"blank; a blank may mean not asked.", *BLANKS)
    levels = [str(v) for v in p.get("levels") or []]
    if len(levels) == 2:
        return Voice(f"{tick(col)} is two-level text ({tick(levels[0])}, {tick(levels[1])}); which "
                     f"level counts as 1 is not asked yet.", closes=True)
    return Voice(f"{tick(col)} stores true and false as text.")


def _positive_class(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    # Raised for a two-level outcome: the event question answers it (M2_CONTRACT §1).
    col = p.get("column") or p.get("target") or (f["affected_columns"] or [None])[0]
    return Voice(f"{tick(col)} has two levels; which one is the event the models predict is asked, "
                 f"never guessed.", "event", "Choose the event")


def _unnamed(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    cols = [str(c) for c in p.get("columns") or f["affected_columns"]]
    n = len(cols)
    return Voice(f"{listing(cols)} {plural(n, 'has', 'have')} no name and often {plural(n, 'is', 'are')} "
                 f"a saved row index.", "roles", "Mark as excluded",
                 title=f"{count(n)} {plural(n, 'column')} {plural(n, 'has', 'have')} no name")


def _constant(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    cols = [str(c) for c in p.get("columns") or f["affected_columns"]]
    n = len(cols)
    one = plural(n, "holds", "hold")
    return Voice(f"{count(n)} {plural(n, 'column')} {one} one value on every row, such as "
                 f"{tick(cols[0])}; {plural(n, 'it', 'they')} cannot inform a model.",
                 "roles", "Mark as excluded",
                 title=f"{count(n)} {plural(n, 'column')} {one} the same value in every row")


def _wide(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    if str(f["title"]).startswith("The wide shape is expected"):
        return Voice("The wide shape is expected for this kind of data; nothing needs changing.",
                     closes=True)
    families = p.get("families") or {}
    groups = list(families.values()) if isinstance(families, Mapping) else []
    n = len(groups) or 1
    title = f"{count(n)} {plural(n, 'group')} of columns look like repeated measures"
    if groups and isinstance(groups[0], list) and groups[0]:
        more = f", and {n - 1} more {plural(n - 1, 'group')}" if n > 1 else ""
        return Voice(f"{listing(groups[0], limit=3)} look like repeated measures of one quantity{more}.",
                     title=title)
    return Voice(title + ".", title=title)


def _number_format(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    entries = _list(p.get("columns"))
    cols = [c for c in (_col(e) for e in entries) if c] or list(f["affected_columns"])
    example = next((e.get("example") for e in entries if isinstance(e, Mapping) and e.get("example")), None)
    n = len(cols)
    such = f", such as {tick(example)}" if example else ""
    verb = plural(n, "writes", "write")
    return Voice(f"{listing(cols, limit=3)} {verb} numbers in a format that does not parse{such}.",
                 title=f"{count(n)} {plural(n, 'column')} {verb} numbers in a format that does not parse")


def _censored(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    entries = _list(p.get("analytes"))
    cols = [c for c in (_col(e) for e in entries) if c] or list(f["affected_columns"])
    n = len(cols)
    return Voice(f"{listing(cols, limit=3)} {plural(n, 'carries', 'carry')} values censored at a "
                 f"detection limit.")


def _text_numeric(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    cols = [c for c in (_col(e) for e in _list(p.get("columns"))) if c] or list(f["affected_columns"])
    n = len(cols)
    return Voice(f"{listing(cols, limit=3)} arrived as text but {plural(n, 'is', 'are')} mostly numbers.")


def _impossible(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    entries = [e for e in _list(p.get("columns")) if isinstance(e, Mapping)]
    if not entries:
        return Voice(finish(f["title"]), "exclusions", "Exclude rows by range")
    e = entries[0]
    band = e.get("impossible_band") or [None, None]
    more = f" ({len(entries) - 1} more {plural(len(entries) - 1, 'column')} too)" if len(entries) > 1 else ""
    # Audit IN-09: the second count is "outside the reference sample's central 98%", not "abnormal".
    return Voice(f"{tick(e.get('column'))} has {count(e.get('n_impossible') or 0)} impossible values "
                 f"outside {tick(number(band[0]))}–{tick(number(band[1]))}; "
                 f"{count(e.get('n_abnormal_but_possible') or 0)} more are unusual but real{more}.",
                 "exclusions", "Exclude rows by range")


def _compositional(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    cols = [str(c) for c in p.get("columns") or f["affected_columns"]]
    total = number(p.get("total", 100))
    return Voice(f"{listing(cols, limit=4)} sum to {tick(total)}: each part is fixed by the others.",
                 "roles", "Leave one part out")


def energy_unit_of(fc: FindingContext, column: str | None) -> dict[str, str] | None:
    """The energy column's unit as the proposals read it (suffix, Atwater, magnitude prior)."""
    if not column or column not in fc.frame.columns:
        return None
    from turbotab.core.stages.proposals import energy_unit_reading

    return energy_unit_reading(fc.frame, column)


def restate_implausible(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> None:
    """The pack's implausible-intake count, in the unit the energy column is in (audit IN-07).

    The pack compares the raw column with 500–5,000 whatever its unit, so a kJ column had nearly
    every row "above 5000" beside the app's own finding that it is in kilojoules. Read in kJ, the
    same range is 2,092–20,920 kJ: the count then matches the kcal screens the proposals offer."""
    col = p.get("column") or (f["affected_columns"] or [None])[0]
    reading = energy_unit_of(fc, col)
    if reading is None:
        return
    f["energy_unit"] = reading["unit"]
    if reading["unit"] != "kj":
        return
    from turbotab.core.stages.proposals import KCAL_PER_KJ

    try:
        from turbotab.packs import _PLAUSIBLE_KCAL as kcal  # the detector's own range
    except ImportError:  # pragma: no cover - the legacy pack always defines it
        kcal = (500.0, 5000.0)
    low, high = kcal[0] * KCAL_PER_KJ, kcal[1] * KCAL_PER_KJ
    s = pd.to_numeric(fc.frame[col], errors="coerce")
    below, above = int((s < low).sum()), int((s > high).sum())
    n = below + above
    p.update({"minimum": low, "maximum": high, "n_flagged": n, "unit": "kJ"})
    f["title"] = (f"{n:,} {'record reports' if n == 1 else 'records report'} an implausible daily "
                  f"intake" if n else "No record reports an implausible daily intake, read in kJ")
    f["detail"] = (f"{reading['sentence']} Read in kJ, {tick(col)} is below {low:,.0f} kJ "
                   f"({kcal[0]:,.0f} kcal) on {below:,} {plural(below, 'record')} and above "
                   f"{high:,.0f} kJ ({kcal[1]:,.0f} kcal) on {above:,}. Observed range "
                   f"{float(s.min()):,.0f} to {float(s.max()):,.0f} kJ.")


def _implausible(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    restate_implausible(f, p, fc)
    col = p.get("column") or (f["affected_columns"] or ["energy"])[0]
    n = p.get("n_flagged")
    lo, hi = p.get("minimum"), p.get("maximum")
    if n is None or lo is None or hi is None:
        return Voice(finish(f["title"]), "exclusions", "Choose an exclusion rule")
    if p.get("unit") == "kJ":
        if not n:
            return Voice(f"Read in kJ, every row of {tick(col)} is within "
                         f"{tick(f'{lo:,.0f}')}–{tick(f'{hi:,.0f}')} kJ a day; nothing needs "
                         f"excluding.", closes=True)
        return Voice(f"{count(n)} of {count(fc.n_rows)} rows report {tick(col)} below "
                     f"{tick(f'{lo:,.0f}')} kJ or above {tick(f'{hi:,.0f}')} kJ a day.",
                     "exclusions", "Choose an exclusion rule")
    return Voice(f"{count(n)} of {count(fc.n_rows)} rows report {tick(col)} below {tick(number(lo))} "
                 f"or above {tick(number(hi))} a day.", "exclusions", "Choose an exclusion rule")


def _energy(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    energy = p.get("energy_column") or (f["affected_columns"] or [None])[0]
    rs = energy_correlations(fc.frame, energy, target=fc.target)
    lever = ("energy_adjustment", "Adjust for energy")
    if rs:
        best = max(rs, key=lambda c: abs(rs[c]))
        return Voice(f"{tick(best)} correlates {rs[best]:.2f} with {tick(energy)} across all "
                     f"{count(fc.n_rows)} rows: nutrient effects are tangled with total energy.",
                     *lever)
    return Voice(f"{tick(energy)} is total energy; every nutrient association is confounded by it "
                 f"until adjusted.", *lever)


def _acquisition(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    by_role = p.get("by_role") if isinstance(p.get("by_role"), Mapping) else {}
    design = [c for role in ("batch", "run_order", "plate", "injection") for c in by_role.get(role, [])]
    cols = design or [str(c) for c in (p.get("columns") or f["affected_columns"])]
    n = len(cols)
    return Voice(f"{listing(cols, limit=3)} {plural(n, 'records', 'record')} how samples were run, "
                 f"not what was measured.", "roles", "Mark design columns")


def _run_order(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    col = p.get("run_order_column")
    share = p.get("share_tracking")
    if col and isinstance(share, (int, float)):
        return Voice(f"Intensity tracks {tick(col)} in {tick(f'{share:.0%}')} of features: the signal "
                     f"drifts over the run.")
    return Voice(finish(f["title"]))


def _p_over_n(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    n_features = p.get("n_features")
    if n_features is None:
        return Voice(finish(f["title"]), "models", "Choose penalized models")
    return Voice(f"{count(n_features)} count columns against {count(fc.n_rows)} samples: an "
                 f"unpenalized model has no unique fit.", "models", "Choose penalized models")


def _repeated_subjects(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    col = p.get("group_column")
    if not col:
        return Voice(finish(f["title"]), "roles", "Mark the subject identifier")
    return Voice(f"{tick(col)} repeats: {count(p.get('n_samples') or 0)} samples from "
                 f"{count(p.get('n_subjects') or 0)} subjects, so samples are not independent.",
                 "roles", "Mark the subject identifier")


NHANES_WEIGHTING_EVIDENCE = {
    "status": "CONVENTION",
    "source": "NHANES Tutorials, Weighting Module: the least common denominator",
}


def restate_survey_weights(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> None:
    """The weight NHANES's least-common-denominator rule names for this table's variables
    (audit IN-19): the legacy finding always said the dietary day-1 weight, even beside fasting
    analytes whose subsample weight (``WTSAF2YR``) is the smaller sample. The NHANES weighting
    tutorial: "You must use the weight of the smallest subpopulation that includes all the
    variables you want to include in your analysis.\""""
    from turbotab.core.recognizers import NHANES_LCD_QUOTE, least_common_denominator

    lcd = least_common_denominator(fc.columns)
    if lcd is None or not lcd["because"] or lcd["sample"] in ("dietary day 1", "dietary days 1 and 2"):
        return
    because = [c for c in lcd["because"] if c != fc.target][:4] or lcd["because"][:4]
    if fc.target in lcd["because"]:
        because = [fc.target, *[c for c in because if c != fc.target]][:4]
    measured = (f"{listing(because, limit=4)} {plural(len(because), 'was', 'were')} measured on "
                f"the morning fasting subsample, the smallest sample among this table's variables")
    if lcd["use"]:
        p.update({"use": [lcd["use"]], "not": list(lcd["not"]), "sample": lcd["sample"]})
        f["title"] = f"Use the fasting subsample weight, {tick(lcd['use'])}"
        f["detail"] = (f"{measured}, so the analysis takes its weight, {tick(lcd['use'])}, not "
                       f"{listing(lcd['not'], limit=4)}. NCHS: {NHANES_LCD_QUOTE}")
    else:
        p.update({"use": [], "not": list(lcd["not"]), "sample": lcd["sample"],
                  "missing": lcd["missing"]})
        f["title"] = f"The fasting subsample weight is not in this table"
        f["detail"] = (f"{measured}, so the analysis needs its weight, {tick(lcd['missing'])}, "
                       f"which this table does not carry; {listing(lcd['not'], limit=4)} "
                       f"describe{'s' if len(lcd['not']) == 1 else ''} a larger sample. NCHS: "
                       f"{NHANES_LCD_QUOTE}")
    f["affected_columns"] = list(dict.fromkeys([*([lcd["use"]] if lcd["use"] else []), *because]))
    f["evidence"] = dict(NHANES_WEIGHTING_EVIDENCE)


def _survey_weights(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    restate_survey_weights(f, p, fc)
    use = [str(w) for w in p.get("use") or []]
    avoid = [str(w) for w in p.get("not") or []]
    if p.get("sample") == "fasting subsample":
        if not use:
            return Voice(f"Fasting analytes need {tick(p.get('missing'))}, which this table lacks; "
                         f"{listing(avoid)} describe a larger sample.", "roles", "Mark design columns")
        return Voice(f"Fasting analytes set the weight: {listing(use)}, not {listing(avoid)}; as "
                     f"design columns none becomes a predictor.", "roles", "Mark design columns")
    if avoid:
        text = (f"Dietary analyses use {listing(use)}, not {listing(avoid)}; as design columns "
                f"neither becomes a predictor.")
    else:
        text = (f"Dietary analyses use {listing(use)}; as a design column it does not become a "
                f"predictor.")
    return Voice(text, "roles", "Mark design columns")


def _lonely_psu(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    n = int(p.get("n") or len(p.get("strata") or []) or 1)
    title = f"{count(n)} {plural(n, 'stratum', 'strata')} {plural(n, 'has', 'have')} a single PSU"
    return Voice(f"{title}, which breaks design-based variance estimates.", title=title)


def _atwater(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    if p.get("verdict") == "energy_in_kj" and p.get("energy_column"):
        return Voice(f"{tick(p['energy_column'])} holds kilojoules, not kcal: kcal cut-offs and "
                     f"partitions must convert by 4.184.")
    return Voice(finish(f["title"]))


def _partial_design(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    missing = [str(m) for m in p.get("missing") or []]
    if not missing:
        return Voice(finish(f["title"]), "roles", "Mark design columns")
    return Voice(f"The survey design is partial: {listing(missing, ticked=False)} "
                 f"{plural(len(missing), 'is', 'are')} missing.", "roles", "Mark design columns")


def _ordinal(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    scale = p.get("scale") or []
    cols = _list(p.get("columns")) or list(f["affected_columns"])
    return Voice(f"{count(len(cols))} columns share one {len(scale)}-point response scale; ordinal "
                 f"models are not in this version.", closes=True)


def _left_censored(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    # Every column the reading names, not the eight the card shows: the missing-values question
    # refuses a median fill of their blanks and offers half-minimum or a censoring-aware fill
    # (audit ME-08; turbotab.core.methods.missing.censored_columns).
    f["censored_columns"] = [str(c) for c in p.get("columns") or f.get("affected_columns") or []]
    return Voice("Missing values cluster in the lowest-abundance features: likely below detection, "
                 "not missing at random.", "missing", "Choose how non-detections are filled")


def _no_run_order(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    return Voice("No run-order column, so drift and QC-based corrections cannot be computed.")


def _zeros(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    if p.get("n_zeros") is None:
        return Voice(finish(f["title"]))
    return Voice(f"{count(p['n_zeros'])} zeros across {count(p.get('n_features_with_zeros') or 0)} "
                 f"features; nothing has assumed whether they mean absent.")


def _redundancy(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    if p.get("n_columns") is None or p.get("effective_features") is None:
        return Voice(finish(f["title"]))
    return Voice(f"{count(p['n_columns'])} numeric columns carry about "
                 f"{count(p['effective_features'])} independent quantities.")


def _column_title(template: str) -> Say:
    def say(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
        col = p.get("column") or (f["affected_columns"] or [None])[0]
        return Voice(template.format(col=tick(col)))
    return say


def _sentinel(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    col = p.get("column") or (f["affected_columns"] or [None])[0]
    values = [number(v) for v in p.get("values") or []]
    if f.get("lens") and values:  # reframed by an assay lens: the title says these are counts
        said = ("is a count, not a missing-value code" if len(values) == 1
                else "are counts, not missing-value codes")
        # No "treat as missing" repair is offered for counts (audit F14), and the summary says so.
        return Voice(f"{tick(col)} holds low counts: {listing(values, limit=3)} {said}; nothing "
                     f"needs repairing.", closes=True)
    if values:
        return Voice(f"{tick(col)} may code missing values as {listing(values, limit=3)}.")
    return Voice(f"{tick(col)} may use numeric codes for missing values.")


def _text_missing(f: dict[str, Any], p: dict[str, Any], fc: FindingContext) -> Voice:
    col = p.get("column") or (f["affected_columns"] or [None])[0]
    values = [str(v) for v in p.get("values") or []]
    such = f", such as {listing(values, limit=3)}" if values else ""
    return Voice(f"{tick(col)} spells missing values as text{such}.")


FAMILIES: dict[str, Say] = {
    "binary_text": _two_level_text,
    "boolean_as_text": _two_level_text,
    "positive_class": _positive_class,
    "sentinel_missing": _sentinel,
    "category_variants": _column_title("{col} has categories that differ only by spacing or case."),
    "numeric_as_text": _column_title("{col} looks numeric but is stored as text."),
    "text_missing": _text_missing,
    "unnamed_columns": _unnamed,
    "constant_columns": _constant,
    "wide_repeated_measures": _wide,
    "pack::clinical::number_format": _number_format,
    "pack::clinical::censored_values": _censored,
    "pack::clinical::text_numeric": _text_numeric,
    "pack::clinical::impossible_vs_extreme": _impossible,
    "pack::dietary::compositional": _compositional,
    "pack::dietary::implausible_intake": _implausible,
    "pack::dietary::energy_adjustment": _energy,
    "pack::dietary::survey_weights": _survey_weights,
    "pack::dietary::lonely_psu": _lonely_psu,
    "pack::dietary::atwater": _atwater,
    "pack::dietary::partial_design": _partial_design,
    "pack::metabolomics::acquisition_design": _acquisition,
    "pack::metabolomics::run_order": _run_order,
    "pack::metabolomics::no_run_order": _no_run_order,
    "pack::metabolomics::zeros_or_missing": _zeros,
    "pack::metabolomics::left_censored": _left_censored,
    "pack::metabolomics::repeated_subjects": _repeated_subjects,
    "pack::metabolomics::redundancy": _redundancy,
    "pack::genomics::counts_p_over_n": _p_over_n,
    "pack::survey::ordinal_declared": _ordinal,
}

# Same-kind findings share one paged card. Families not listed page under their own name.
GROUPS: dict[str, str] = {
    "binary_text": "two_level_text",
    "boolean_as_text": "two_level_text",
    "pack::genomics::gene_id_excel_corruption": "gene_ids",
    "pack::genomics::gene_id_versions": "gene_ids",
    "pack::genomics::gene_id_duplicates": "gene_ids",
    "pack::genomics::gene_id_mixed_vocabulary": "gene_ids",
    "voice::flag": "flags",
    "voice::identifier": "identifiers",
    # One coding convention across many columns is one card, paged (audit H19, WP14).
    "sentinel_missing": "missing_codes",
}


# ── speaking for a finding ───────────────────────────────────────────────────

_QUOTED = re.compile(r"(?<![\w`])['‘]([^'’\n]{1,120})['’](?![\w`])")


def quote_columns(text: str, columns: Sequence[str] | AbstractSet[str]) -> str:
    """``'gender'`` → ``\\`gender\\``` when the quoted text is one of this table's columns."""
    names = columns if isinstance(columns, (set, frozenset)) else set(columns)

    def swap(m: re.Match[str]) -> str:
        return f"`{m.group(1)}`" if m.group(1) in names else m.group(0)

    return _QUOTED.sub(swap, str(text))


def _fallback(title: str) -> str:
    """The legacy title as a summary, cut at a clause boundary when it runs long."""
    text = finish(title)
    budget = SUMMARY_WORDS - len(NO_LEVER.split())
    if len(text.split()) <= budget:
        return text
    for sep in (": ", " — ", "; ", ", and ", ", "):
        head = text.split(sep, 1)[0]
        if 4 <= len(head.split()) <= budget:
            return finish(head)
    return finish(" ".join(text.split()[:budget]))


def speak(finding: dict[str, Any], raw: Mapping[str, Any] | None, fc: FindingContext) -> dict[str, Any]:
    """Add ``summary``, ``routes_to``, ``lever_label`` and ``group``; normalize every text field.

    ``group`` is set to the finding's kind here; :func:`settle_groups` clears it where the kind
    has only one finding.
    """
    params = dict((raw or {}).get("params") or {})
    fam = family(finding["id"])
    columns = fc.names
    for key in ("title", "detail", "why_it_matters"):
        if finding.get(key):
            finding[key] = quote_columns(finding[key], columns)
    say = FAMILIES.get(fam)
    voice = say(finding, params, fc) if say is not None else None
    if voice is None:
        voice = Voice(_fallback(finding["title"]))
    if voice.title:
        finding["title"] = voice.title
    summary = finish(quote_columns(voice.summary, columns))
    if voice.routes_to is None and not voice.closes:
        summary = f"{summary} {NO_LEVER}"
    finding["title"] = finish(finding["title"])
    finding["detail"] = finish(finding.get("detail") or "")
    finding["why_it_matters"] = finish(finding["why_it_matters"]) if finding.get("why_it_matters") else None
    finding["summary"] = summary
    finding["routes_to"] = voice.routes_to
    finding["lever_label"] = finish(voice.lever, terminal=False) if voice.routes_to and voice.lever else None
    finding["group"] = GROUPS.get(fam, fam)
    return finding


def settle_groups(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A group key only where two or more findings share it (a pager of one is not a pager)."""
    sizes: dict[str, int] = {}
    for f in findings:
        if f.get("group"):
            sizes[f["group"]] = sizes.get(f["group"], 0) + 1
    for f in findings:
        if f.get("group") and sizes.get(f["group"], 0) < 2:
            f["group"] = None
    return findings


# ── data the summaries cite ──────────────────────────────────────────────────

def energy_correlations(frame: pd.DataFrame, energy: str | None, *, target: str | None = None) -> dict[str, float]:
    """Pearson r of each energy-bearing nutrient column with the energy column."""
    from turbotab.core.stages.proposals import energy_bearing

    if not energy or energy not in frame.columns:
        return {}
    e = pd.to_numeric(frame[energy], errors="coerce")
    out: dict[str, float] = {}
    for c in frame.columns:
        c = str(c)
        if c in (energy, target) or not energy_bearing(c):
            continue
        if not pd.api.types.is_numeric_dtype(frame[c]):
            continue
        r = e.corr(pd.to_numeric(frame[c], errors="coerce"))
        if r is not None and not math.isnan(r):
            out[c] = float(r)
    return out


def restate_energy(finding: dict[str, Any], raw: Mapping[str, Any] | None, fc: FindingContext) -> None:
    """The legacy detail counts every numeric column as a candidate nutrient; say what is true."""
    params = dict((raw or {}).get("params") or {})
    energy = params.get("energy_column") or (finding["affected_columns"] or [None])[0]
    rs = energy_correlations(fc.frame, energy, target=fc.target)
    if rs:
        lo, hi = min(rs.values()), max(rs.values())
        spread = f"{lo:.2f}" if round(lo, 2) == round(hi, 2) else f"{lo:.2f} to {hi:.2f}"
        n = len(rs)
        finding["detail"] = (
            f"{tick(energy)} is total energy. {listing(list(rs), limit=6)} "
            f"{plural(n, 'carries', 'carry')} energy, and {plural(n, 'it correlates', 'each correlates')} "
            f"{spread} with {tick(energy)}.")
        finding["affected_columns"] = [energy, *rs]
    else:
        finding["detail"] = (f"{tick(energy)} is total energy; no energy-bearing nutrient column was "
                             f"recognized by name.")
    finding["why_it_matters"] = (
        "People who eat more of everything eat more of anything, so every nutrient association is "
        "confounded by total intake; that adjustment is needed is not in dispute. Which method to "
        "use is a convention: with total energy kept in the model the standard and residual "
        "methods estimate the same substitution, and the residual method's advantages are "
        "practical, not inferential.")


# ── the app's own findings ───────────────────────────────────────────────────

_FLAG =re.compile(r"^(?:imputed|flag|is_imputed)_(?P<pre>.+)$|^(?P<post>.+)_(?:imputed|flag|imp)$", re.I)
_DESIGN = re.compile(r"^(?:wt[a-z0-9]*|sdmv\w*|sddsrvyr)$|(?:^|_)(?:pweight|sampweight|sampling_weight|"
                     r"survey_weight|svy_weight|strata|stratum|psu|cluster|fpc)(?:$|_)", re.I)
_CYCLE = re.compile(r"(?:^|_)cycle(?:$|_)|^sddsrvyr$|^survey_year$|^release$", re.I)
_TRUTHY = {"true", "1", "yes", "y", "t"}
_FALSY = {"false", "0", "no", "n", "f"}


def _tokens(name: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(name))
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def is_identifier_name(name: str) -> bool:
    """The name reads as an identifier of any kind: the one recognizer every stage shares
    (:func:`turbotab.core.recognizers.is_identifier`; audit IN-06, where three recognizers
    disagreed on 11 of 46 real names)."""
    from turbotab.core.recognizers import is_identifier

    return is_identifier(name)


def nhanes_like(columns: Sequence[str]) -> bool:
    lowered = {c.lower() for c in columns}
    return bool({"seqn", "sddsrvyr", "riagendr", "ridageyr"} & lowered
                or any(c.startswith(("dr1t", "dr2t", "bmx", "sdmv", "wtdr", "wtmec")) for c in lowered)
                or any("nhanes" in c for c in lowered))


def _finding(fid: str, severity: str, title: str, summary: str, detail: str, why: str | None,
             columns: list[str], routes_to: str | None, lever: str | None,
             evidence: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "id": fid, "severity": severity, "title": title, "detail": detail, "why_it_matters": why,
        "affected_columns": columns, "source": "profile", "lens": None, "evidence": evidence,
        "summary": summary, "routes_to": routes_to, "lever_label": lever,
        "group": GROUPS.get(family(fid), family(fid)),
    }


REPEATED_SHARE = 0.2  # of units seen more than once, for a column to read as repeated measures


def identifier_findings(fc: FindingContext, taken: set[str]) -> list[dict[str, Any]]:
    out = []
    n = fc.n_rows
    for c in fc.columns:
        if c == fc.target or c in taken or not is_identifier_name(c):
            continue
        s = fc.frame[c].dropna()
        if s.empty:
            continue
        from turbotab.core.recognizers import id_kind, reads_as_measurement

        if reads_as_measurement("__values__", values=s) is not None:
            continue  # fractional values: a measurement, whatever its name (audit IN-06)
        per = s.value_counts()
        n_units = int(len(per))
        per_unit = int(per.max())
        person = id_kind(c) == "subject"
        units = "participants" if person else "distinct values"
        if n_units == len(s) and len(s) >= 0.95 * n:
            names = "a participant" if person else "its row"
            out.append(_finding(
                f"voice::identifier__{c}", "info", f"{tick(c)} names each row",
                f"{tick(c)} is unique on every row: it names {names} and is not a predictor.",
                f"{tick(c)} has {count(n_units)} different values across {count(n)} rows, one per "
                f"row. A value that appears once cannot tell a model anything about a row it has "
                f"not seen.",
                "As a predictor it would be noise at best, and a way to memorize rows at worst.",
                [c], "roles", "Mark as identifier"))
        elif (id_kind(c) == "cluster" and n_units >= 2
              and float((per > 1).mean()) >= REPEATED_SHARE):
            # A site or household groups participants: not their identifier (audit IN-06).
            out.append(_finding(
                f"voice::repeats__{c}", "warning", f"{tick(c)} groups rows",
                f"{tick(c)} groups {count(len(s))} rows into {count(n_units)} clusters, such as "
                f"sites or households; rows within one are not independent.",
                f"{tick(c)} has {count(n_units)} values across {count(len(s))} rows, at most "
                f"{count(per_unit)} rows each. It groups participants rather than naming them.",
                "Rows that share a site or household share whatever differs between sites or "
                "households, so intervals that treat them as independent are too narrow.",
                [c], "roles", "Mark as cluster"))
        elif (2 <= per_unit <= 50 and n_units >= 10
              and float((per > 1).mean()) >= REPEATED_SHARE):
            mean = len(s) / n_units
            one = "one participant" if person else "one value"
            out.append(_finding(
                f"voice::repeats__{c}", "warning", f"{tick(c)} repeats across rows",
                f"{tick(c)} repeats: {count(len(s))} rows from {count(n_units)} {units}, so rows of "
                f"{one} are not independent.",
                f"{tick(c)} has {count(n_units)} values across {count(len(s))} rows, about "
                f"{mean:.1f} rows each and at most {count(per_unit)}. Rows of {one} share "
                f"everything about it.",
                "A split that puts one participant's rows on both sides lets a model recognize "
                "people instead of learning, and every score looks better than it is. As the "
                "identifier, it keeps each participant's rows together.",
                [c], "roles", "Mark as identifier",
                {"status": "SETTLED", "source": "research/NUTRITION_PACK.md#08 · Feature selection and modeling"}))
    return out


def _flag_base(column: str, columns: Sequence[str]) -> str | None:
    m = _FLAG.match(column)
    if not m:
        return None
    base = m.group("pre") or m.group("post")
    for c in columns:
        if c.lower() == str(base).lower() and c != column:
            return c
    return None


def flag_findings(fc: FindingContext) -> list[dict[str, Any]]:
    out = []
    for c in fc.columns:
        if c == fc.target or not _FLAG.match(c):
            continue
        values = fc.frame[c].dropna()
        keys = {str(v).strip().lower() for v in values.unique()}
        if not keys or not keys <= (_TRUTHY | _FALSY):
            continue
        n_true = int(values.map(lambda v: str(v).strip().lower() in _TRUTHY).sum())
        n_false = int(len(values) - n_true)
        base = _flag_base(c, fc.columns)
        imputed = "imput" in c.lower()
        what = "imputed values" if imputed else "rows"
        if base:
            summary = (f"{tick(c)} flags {count(n_true)} {what} of {tick(base)}; it describes the "
                       f"data, not the participant.")
            detail = (f"{tick(c)} is true on {count(n_true)} rows and false on {count(n_false)}. It "
                      f"marks where {tick(base)} was {'imputed' if imputed else 'flagged'}, so it is a "
                      f"fact about how the data were prepared.")
        else:
            summary = (f"{tick(c)} flags {count(n_true)} rows; it describes the data, not the "
                       f"participant.")
            detail = f"{tick(c)} is true on {count(n_true)} rows and false on {count(n_false)}."
        why = ("As a predictor, a flag lets a model learn how values were filled rather than what "
               "they measure. As a flag it stays in the record, linked to the column it marks.")
        out.append(_finding(f"voice::flag__{c}", "info",
                            f"{tick(c)} marks {'imputed ' if imputed else ''}values"
                            + (f" of {tick(base)}" if base else ""),
                            summary, detail, why, [c, *([base] if base else [])], "roles",
                            "Mark as a flag"))
    return out


def design_findings(fc: FindingContext) -> list[dict[str, Any]]:
    if not ({"dietary", "survey", "clinical"} & set(fc.lens)) or not nhanes_like(fc.columns):
        return []
    if any(_DESIGN.search(c) for c in fc.columns):
        return []
    evidence = {"status": "SETTLED", "source": NUT01}
    return [_finding(
        "voice::survey_design_absent", "warning", "No survey weights or design columns",
        "No survey weights or design columns: results describe these participants, not the US "
        "population.",
        "This NHANES table has no survey weight (such as `WTDRD1` or `WTMEC2YR`), no strata "
        "(`SDMVSTRA`) and no primary sampling units (`SDMVPSU`). If the export dropped them, export "
        "them again: with them, an inference can describe the surveyed population.",
        "NHANES deliberately oversamples some age, race and income groups, so unweighted means are "
        "biased toward them, and standard errors that ignore the design are too small.",
        [], "roles", "Mark design columns", evidence)]


def cycle_findings(fc: FindingContext) -> list[dict[str, Any]]:
    if not nhanes_like(fc.columns):
        return []
    out = []
    for c in fc.columns:
        if c == fc.target or not _CYCLE.search(c):
            continue
        counts = fc.frame[c].dropna().value_counts()
        if not 2 <= len(counts) <= 30:
            continue
        levels = sorted(counts.index, key=lambda v: (str(type(v)), v))
        out.append(_finding(
            f"voice::pooled_cycles__{c}", "info", f"{tick(c)} pools survey cycles",
            f"{tick(c)} pools {count(len(counts))} survey cycles, {tick(number(levels[0]))} to "
            f"{tick(number(levels[-1]))}; methods may differ across them.",
            f"Each cycle holds {count(int(counts.min()))} to {count(int(counts.max()))} rows. As a "
            f"covariate, the cycle absorbs level differences between cycles; as a time column, it "
            f"stays out of the models.",
            "Combining NHANES cycles from 2001–2002 on means dividing the two-year weights by the "
            "number of cycles combined. 1999–2000 is the exception: its two-year weights and "
            "2001–2002's rest on different censuses, so the 1999–2002 rows take the four-year "
            "weight, doubled before the division (NHANES Analytic Guidelines 2011–2016 "
            "§3.1.3–3.1.4). Confirm the same dietary methodology applies across cycles.",
            [c], "roles", "Choose the cycle's role", {"status": "SETTLED", "source": NUT01}))
    return out


_PACK_ENERGY_ALIAS = "DR1TKCAL"  # a name the pack's exact-alias matcher reads as total energy


def energy_findings(fc: FindingContext, legacy: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The pack's implausible-intake and energy-adjustment findings for a total-energy column only
    the app's recognizer reads (audit IN-08: ``DR2TKCAL``, ``DRXTKCAL``, ``ENERC_KCAL``,
    ``TotalKcal``, ``total_energy`` and ``kcal_day`` produced neither, because the pack matches
    energy by exact alias). The pack's own detectors run, with the column it could not name read as
    the energy column, so their counts, text and badges are the pack's."""
    if "dietary" not in fc.lens:
        return []
    from turbotab import packs
    from turbotab.core.stages.findings import pack_finding
    from turbotab.core.stages.proposals import energy_column

    have = {family(f["id"]) for f in legacy}
    info = {c: {"dtype": "numeric" if pd.api.types.is_numeric_dtype(fc.frame[c]) else "text"}
            for c in fc.columns if c != fc.target}
    energy = energy_column(info, {})
    if energy is None or energy == _PACK_ENERGY_ALIAS or _PACK_ENERGY_ALIAS in fc.names:
        return []
    alias = fc.frame.rename(columns={energy: _PACK_ENERGY_ALIAS})
    if fc.target is not None and fc.target in alias.columns:
        alias = alias.drop(columns=[fc.target])
    out = []
    for fid, detector in (("pack::dietary::implausible_intake", packs._implausible_intake),
                          ("pack::dietary::energy_adjustment", packs._energy_adjustment)):
        if fid in have:
            continue
        try:
            raw = detector(alias)
        except Exception:  # noqa: BLE001 - the legacy detector is a second reader, never a failure
            raw = None
        if raw is None:
            continue
        raw = _renamed(raw, _PACK_ENERGY_ALIAS, energy)
        finding = pack_finding(raw)
        if fid == "pack::dietary::energy_adjustment":
            restate_energy(finding, raw, fc)
        out.append(speak(finding, raw, fc))
    return out


def _renamed(value: Any, old: str, new: str) -> Any:
    """``value`` with every mention of the column ``old`` naming ``new`` instead."""
    if isinstance(value, str):
        return new if value == old else value.replace(f"`{old}`", f"`{new}`")
    if isinstance(value, list):
        return [_renamed(v, old, new) for v in value]
    if isinstance(value, tuple):
        return tuple(_renamed(v, old, new) for v in value)
    if isinstance(value, dict):
        return {k: _renamed(v, old, new) for k, v in value.items()}
    return value


def own_findings(fc: FindingContext, legacy: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The app's own findings, skipping columns a legacy finding already speaks for."""
    taken = set()
    for f in legacy:
        if family(f["id"]) == "pack::metabolomics::repeated_subjects":
            taken.update(f.get("affected_columns") or [])
    found = identifier_findings(fc, taken) + flag_findings(fc) + design_findings(fc) + cycle_findings(fc)
    found += energy_findings(fc, legacy)
    for f in found:
        f["title"] = finish(f["title"])
        f["summary"] = finish(f["summary"])
        f["detail"] = finish(f["detail"])
        f["why_it_matters"] = finish(f["why_it_matters"]) if f["why_it_matters"] else None
    return found


def flag_columns(findings: Sequence[dict[str, Any]]) -> set[str]:
    return {f["affected_columns"][0] for f in findings
            if family(f["id"]) == "voice::flag" and f.get("affected_columns")}


__all__ = [
    "FAMILIES", "FindingContext", "GROUPS", "LEVER_WORDS", "NO_LEVER", "SUMMARY_WORDS",
    "energy_correlations", "family", "flag_columns", "own_findings", "quote_columns",
    "restate_energy", "settle_groups", "speak",
]
