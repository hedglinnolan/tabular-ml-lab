"""turbotab.packs — field awareness without a second app.

`DOMAIN_PACKS.md`, made executable. The lens question, the five packs, and the
one architectural claim that keeps breadth tractable:

> **The unit of domain knowledge is a finding.** Adding a domain means adding
> detectors and reference data. It never means adding interface.

So nothing here invents a card type. A pack emits findings in the engine's own
shape, with `fix_kind="none"` where it is reporting rather than proposing — and
`ml.router._is_repairable` already treats that as a report and not a fork, which
is why **guard #2 is structural rather than aspirational**: a pack that reports
cannot add a question, whatever it reports.

## The three guards

1. **A pack may not add interview components.** It supplies findings and
   defaults. The one exception is deliberate and narrow: `reverse_coding` is a
   real question, because reverse-coding needs a codebook the app does not have.
   It is gated on its own detector, so it exists only where it applies.
2. **A pack must not fire on non-matching data.** Every detector below reads
   SHAPE, never a label — *"the label sets priors; the data resolves them into
   findings"* (§06). `turbotab/test_a_pack_does_not_fire_on_the_wrong_data.py`
   runs every pack against every fixture and asserts the question count is
   unchanged everywhere it does not belong.
3. **Every default states its reason and is overturnable.** The confidence
   marker governs the treatment, and it is carried on the finding rather than
   implied: `derived` is pre-selected with its reason shown, `convention` is
   pre-selected and stated AS convention, `offered` is never defaulted at all.

A fourth, on voice: **conventions are stated as conventions.** *"The field
convention here is Pareto scaling"* is honest; *"you should use Pareto scaling"*
is not, because the app never speaks in the user's name — and a pack is the
easiest place in the product to break that rule.

## Why the lens is asked and not inferred

The same architecture as the grain question, for the same reason: the user knows
and the engine can only guess. Detection runs as a **suggestion** and as a
**contradiction detector**, never as the answer. A pack that fires on the wrong
data asserts something false *authoritatively*, which is harder for a user to
catch than an ordinary bug.

And "Something else, or not sure" is first-class. The app is fully functional
with no lens; a pack is an accelerator. Any design in which an unlisted field
degrades the experience has built a tool for five disciplines rather than a tool
that is unusually good at five.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── the answers ──────────────────────────────────────────────────────────────

METABOLOMICS = "metabolomics"
GENOMICS = "genomics"
DIETARY = "dietary"
CLINICAL = "clinical"
SURVEY = "survey"
OTHER = "other"

LENS_KEYS: Tuple[str, ...] = (METABOLOMICS, GENOMICS, DIETARY, CLINICAL,
                              SURVEY, OTHER)

LENS_LABELS: Dict[str, str] = {
    METABOLOMICS: "Metabolomics or proteomics",
    GENOMICS: "Genomics or transcriptomics",
    DIETARY: "Dietary intake",
    CLINICAL: "Clinical measurements and labs",
    SURVEY: "Survey or questionnaire instruments",
    OTHER: "Something else, or not sure",
}

LENS_TITLE = "What kind of measurements are in this table?"
LENS_WHY = ("Pick all that apply. This changes what we look for and what we "
            "suggest — it never limits what you can do.")
LENS_CONSUMER = (
    "The structural diagnosis reads it first, because the diagnosis is itself "
    "field-sensitive: 400 columns across 80 rows reads as malformed to a "
    "general-purpose import doctor and is the expected shape for an assay "
    "panel. After that it sets priors on missingness, on model ranking, and on "
    "which figure answers a question. It never removes an option, and every "
    "default it raises states its reason and can be overturned.")


class PackError(Exception):
    """A lens answer the app cannot honestly record."""


def normalize(keys: Sequence[str]) -> List[str]:
    """The recorded answer, validated and ordered. `other` is a real answer.

    An empty selection is refused rather than silently read as `other`: the
    difference between *"the user said none of these apply"* and *"the question
    was never answered"* is the recorded-absence rule, and a default that
    swallows the first into the second is exactly what that rule forbids.
    """
    chosen = [str(k) for k in keys or []]
    unknown = [k for k in chosen if k not in LENS_KEYS]
    if unknown:
        raise PackError(
            f"{unknown[0]!r} is not one of {list(LENS_KEYS)}.")
    if not chosen:
        raise PackError(
            "The lens question needs an answer, and 'Something else, or not "
            "sure' is one — the app is fully functional without a lens. An "
            "empty selection would be indistinguishable from never having "
            "asked.")
    seen: List[str] = []
    for k in LENS_KEYS:                       # a stable order, not click order
        if k in chosen and k not in seen:
            seen.append(k)
    if OTHER in seen and len(seen) > 1:
        # "Not sure" beside four confident answers is not an answer, it is two.
        raise PackError(
            "'Something else, or not sure' says the listed kinds do not "
            "describe this table. Selecting it beside one that does is two "
            "different answers, and the record could not say which.")
    return seen


def methods_sentence(keys: Sequence[str]) -> str:
    """What the manuscript carries. A lens the reader cannot see is unchecked.

    §01: *the answer is a recorded decision, not hidden state.* Every domain
    default downstream is licensed by this sentence, so the sentence has to say
    what was claimed AND what it was allowed to do.
    """
    chosen = list(keys)
    if chosen == [OTHER]:
        return ("The measurements in this dataset were not described as "
                "belonging to any of the offered domains, so no domain-specific "
                "defaults were applied and every preprocessing decision was "
                "made from the data alone.")
    names = [LENS_LABELS[k].lower() for k in chosen]
    joined = names[0] if len(names) == 1 else (
        ", ".join(names[:-1]) + " and " + names[-1])
    return (f"The measurements were described as {joined}. Domain conventions "
            f"for {'these fields' if len(names) > 1 else 'this field'} informed "
            f"the defaults offered below; each is stated with its reasoning and "
            f"was open to being overridden.")


# ─────────────────────────────────────────────────────────────────────────────
# Shape readings — name-blind, and the reason they are
# ─────────────────────────────────────────────────────────────────────────────

def _numeric(df: pd.DataFrame) -> List[str]:
    return [str(c) for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def _is_assay_wide(df: pd.DataFrame, minimum: int = 30) -> bool:
    """Many measurement columns. The precondition for every assay reading.

    Deliberately a floor on the COUNT rather than on p/n. A 500-row study with
    400 features is still an assay panel, and a 5-row file with 6 columns is not
    one however bad its p/n looks.
    """
    return len(_numeric(df)) >= minimum


def _finding(fid: str, severity: str, title: str, detail: str,
             why: str, *, confidence: str, pack: str, marker: str,
             columns: Sequence[str] = (), params: Optional[Dict] = None,
             fix_label: str = "", fix_kind: str = "none") -> Dict[str, Any]:
    """One pack finding, in the engine's own shape.

    `fix_kind="none"` by default, and that default is load-bearing:
    `router._is_repairable` reads it as the engine refusing to guess — a report,
    not a fork — so a reporting pack cannot add a question. Guard #2 is a
    property of the data model rather than of anybody's restraint.

    `marker` is the confidence marker from `DOMAIN_PACKS.md` §07 — `derived`,
    `convention` or `offered` — and it is carried rather than implied because it
    governs the treatment. A `convention` rendered as a `derived` fact is the
    app speaking in the user's name.
    """
    return {
        "id": fid, "severity": severity, "title": title, "detail": detail,
        "why_it_matters": why, "fix_label": fix_label, "fix_kind": fix_kind,
        "confidence": confidence, "params": dict(params or {}),
        "affected_columns": [str(c) for c in columns],
        "source": "pack", "pack": pack, "marker": marker,
    }


# ── metabolomics ─────────────────────────────────────────────────────────────

def _left_censored(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Missingness ordered by abundance is left censoring, not randomness.

    The reading is a rank correlation between a feature's missing rate and its
    mean abundance, and it is the whole finding: **the detection is derived**,
    because a detection limit is one instrument threshold and which features
    fall below it is decided by where they sit relative to it.

    Only the METHOD is a choice, and half-minimum wins on explainability over
    QRILC — *"values below the detection limit were imputed as half the minimum
    observed"* is a sentence a reader can evaluate.
    """
    cols = _numeric(df)
    if len(cols) < 30:
        return None
    rate = df[cols].isna().mean()
    with_blanks = [c for c in cols if rate[c] > 0]
    if len(with_blanks) < 5:
        return None
    abundance = df[cols].mean(numeric_only=True)
    usable = [c for c in cols if pd.notna(abundance[c]) and abundance[c] > 0]
    if len(usable) < 30:
        return None
    rho = pd.Series(rate[usable]).corr(
        pd.Series(np.log(abundance[usable])), method="spearman")
    if pd.isna(rho) or rho > -0.5:
        return None
    worst = rate[with_blanks].sort_values(ascending=False)
    return _finding(
        "pack::metabolomics::left_censored", "warning",
        "Your missing values cluster in the lowest-abundance features",
        (f"Across {len(usable):,} features, a feature's missing rate tracks its "
         f"abundance rank at a rank correlation of {rho:.2f}. "
         f"{len(with_blanks):,} features have blanks; the highest rate is "
         f"{worst.iloc[0]:.0%}, on one of the least abundant."),
        ("In metabolomics that usually means below the detection limit — "
         "left-censored rather than missing at random — and filling with a "
         "median would place non-detections in the middle of the distribution. "
         "Half the minimum observed is the convention, and it is the one a "
         "reader can check."),
        confidence="high", pack=METABOLOMICS, marker="derived",
        columns=list(worst.index[:8]),
        params={"rho": round(float(rho), 3), "n_features": len(usable),
                "n_with_blanks": len(with_blanks),
                "suggested_method": "half_minimum"})


def _acquisition_order(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """A run-order column, and intensity that tracks it.

    Name-blind: a run-order column is an integer column that is a PERMUTATION of
    the row positions. That reading costs nothing on ordinary data — a study ID
    is not a permutation of 1..n unless it happens to be, and a permutation that
    nothing correlates with is not reported.

    Detection is derived; correction is `offered` and never automatic, because
    it alters every value in the table.
    """
    cols = _numeric(df)
    if len(cols) < 30:
        return None
    n = len(df)
    order_col = None
    for c in cols:
        s = df[c].dropna()
        if len(s) != n:
            continue
        try:
            values = np.sort(s.to_numpy())
        except (TypeError, ValueError):
            continue
        if not np.all(np.equal(np.mod(values, 1), 0)):
            continue
        if np.array_equal(values, np.arange(1, n + 1)) or \
           np.array_equal(values, np.arange(0, n)):
            order_col = c
            break
    if order_col is None:
        return None

    others = [c for c in cols if c != order_col]
    order = df[order_col].to_numpy(dtype=float)
    tracked = []
    for c in others:
        s = df[c]
        filled = s.fillna(s.median())
        if filled.nunique() < 3:
            continue
        with np.errstate(all="ignore"):
            r = np.corrcoef(order, np.log1p(np.clip(filled.to_numpy(dtype=float),
                                                    0, None)))[0, 1]
        if not np.isnan(r) and abs(r) > 0.3:
            tracked.append(c)
    share = len(tracked) / max(len(others), 1)
    if share < 0.15:
        return None
    return _finding(
        "pack::metabolomics::run_order", "warning",
        f"There is a run-order column, and intensity tracks it",
        (f"`{order_col}` runs 1 to {n:,} with every position used exactly once. "
         f"{len(tracked):,} of {len(others):,} features ({share:.0%}) correlate "
         f"with it above 0.3 in absolute value."),
        ("Instrument drift is often the largest single variance component in a "
         "metabolomics run, larger than the biology. Correction is not applied "
         "here: it alters every value in the table, so it is a decision rather "
         "than a default."),
        confidence="high", pack=METABOLOMICS, marker="offered",
        columns=[order_col] + tracked[:6],
        params={"run_order_column": order_col, "n_tracking": len(tracked),
                "share_tracking": round(share, 3)})


def _pooled_qc(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Rows that are one sample injected repeatedly, not participants.

    **This is the class of error only the lens can see**, and the cheapest
    demonstration that the opening question earns its place: pooled QC rows look
    exactly like participants, must never enter a model, and are needed for
    quality assessment. A generic tool models them silently.

    Name-blind again, and the evidence is variance: a minority level of some
    categorical column whose rows are markedly *less* variable across the
    feature block than the majority's. One sample injected eight times has
    technical variation and no biological variation, and that shows.
    """
    cols = _numeric(df)
    if len(cols) < 30:
        return None
    n = len(df)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        counts = s.value_counts(dropna=True)
        if len(counts) != 2:
            continue
        minority, majority = counts.index[-1], counts.index[0]
        n_minor = int(counts.iloc[-1])
        if n_minor < 3 or n_minor > 0.3 * n:
            continue
        block = df.loc[s == minority, cols]
        rest = df.loc[s == majority, cols]
        if len(rest) < 5:
            continue
        with np.errstate(all="ignore"):
            rsd_minor = float((block.std() / block.mean().abs()).median())
            rsd_rest = float((rest.std() / rest.mean().abs()).median())
        if not np.isfinite(rsd_minor) or not np.isfinite(rsd_rest) or rsd_rest <= 0:
            continue
        if rsd_minor > 0.6 * rsd_rest:
            continue
        return _finding(
            "pack::metabolomics::pooled_qc", "critical",
            f"{n_minor:,} rows look like pooled quality-control injections",
            (f"The {n_minor:,} rows where `{c}` is {minority!r} vary far less "
             f"across the {len(cols):,} features than the {int(counts.iloc[0]):,} "
             f"rows where it is {majority!r} — a median relative standard "
             f"deviation of {rsd_minor:.0%} against {rsd_rest:.0%}. That is one "
             f"sample injected repeatedly, not {n_minor:,} different people."),
            ("They are not participants. Modeling them is an error with no "
             "legitimate reading — they would contribute rows the model can fit "
             "perfectly and a held-out set could contain them. They stay in the "
             "table for quality assessment and out of the modeling rows."),
            confidence="high", pack=METABOLOMICS, marker="derived",
            columns=[str(c)],
            params={"column": str(c), "qc_value": str(minority),
                    "n_qc": n_minor, "rsd_qc": round(rsd_minor, 3),
                    "rsd_participants": round(rsd_rest, 3)})
    return None


# ── dietary ──────────────────────────────────────────────────────────────────

def _compositional(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Columns that sum to a constant. Parts of a whole.

    Correlation between parts of a whole is negatively biased by construction,
    so this **gates the collinearity figure** rather than adding a step — a
    correlation matrix over compositional parts is not a figure with a caveat,
    it is a figure that cannot be read.

    Bounded on purpose: the subset search runs only when there are twelve or
    fewer plausible parts. A compositional test over 400 assay features is a
    different and harder problem, and a detector that pretended to do it would
    be guessing at exactly the scale where guessing is least visible.
    """
    cols = _numeric(df)
    for total in (100.0, 1.0):
        candidates = [c for c in cols
                      if float(df[c].min(skipna=True)) >= -1e-9
                      and float(df[c].max(skipna=True)) <= total * 1.02
                      and float(df[c].mean(skipna=True)) > total * 0.005]
        if not 3 <= len(candidates) <= 12:
            continue
        for size in range(3, min(len(candidates), 6) + 1):
            for subset in itertools.combinations(candidates, size):
                sums = df[list(subset)].sum(axis=1, skipna=False)
                close = (sums - total).abs() <= total * 0.005
                if float(close.mean()) >= 0.95:
                    return _finding(
                        "pack::dietary::compositional", "warning",
                        f"{len(subset)} columns sum to {total:g} on every row",
                        ("`" + "`, `".join(subset) + "` add to "
                         f"{total:g} for {float(close.mean()):.0%} of rows."),
                        ("These are compositional — parts of a whole — and "
                         "ordinary correlation between them is not "
                         "interpretable: raising one necessarily lowers "
                         "another, so the negative correlation is arithmetic "
                         "rather than dietary. The collinearity figure is drawn "
                         "on log-ratios rather than on the parts, and the parts "
                         "are not offered as independent predictors."),
                        confidence="high", pack=DIETARY, marker="derived",
                        columns=list(subset),
                        params={"columns": list(subset), "total": total,
                                "share_closing": round(float(close.mean()), 3),
                                "gates": "collinearity_figure"})
    return None


def _reference_column(df: pd.DataFrame, variable: str) -> Optional[str]:
    """A column the engine's own reference matcher recognizes as `variable`.

    Goes through `physiology_reference.match_variable_key`, which is exact
    against the key or one of its aliases after case and separators are
    stripped — never by substring, which is what let `hba1c_proxy` inherit
    HbA1c's bounds. Borrowing the vetted matcher is the opposite of adding a
    third name list.
    """
    try:
        from ml.physiology_reference import load_reference_bundle, match_variable_key
        reference = load_reference_bundle()["nhanes"]
    except Exception:                                      # pragma: no cover
        return None
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if match_variable_key(str(c), reference) == variable:
            return str(c)
    return None


# Adult daily intake outside which a 24-hour recall is treated as a reporting
# error rather than a diet. Stated as a convention with its numbers visible,
# because it is one — and it is OFFERED rather than applied, because it changes
# N and an exclusion that changes N is an eligibility criterion the user states.
_PLAUSIBLE_KCAL = (500.0, 5000.0)


def _implausible_intake(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    low, high = _PLAUSIBLE_KCAL
    col = _reference_column(df, "kcal")
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    flagged = s[(s < low) | (s > high)]
    if flagged.empty:
        return None
    return _finding(
        "pack::dietary::implausible_intake", "info",
        f"{len(flagged):,} records report an implausible daily intake",
        (f"`{col}` is below {low:g} on {int((s < low).sum()):,} record(s) and "
         f"above {high:g} on {int((s > high).sum()):,}. Observed range "
         f"{float(s.min()):,.0f} to {float(s.max()):,.0f}."),
        ("These are possible days and poor estimates: a recall of 300 kcal is "
         "under-reporting rather than starvation. Excluding them is an "
         "eligibility criterion, which changes N and is reported in participant "
         "flow — so it is offered here and never applied. Nothing is filtered "
         "unless you say so."),
        confidence="medium", pack=DIETARY, marker="offered",
        columns=[col],
        params={"column": col, "minimum": low, "maximum": high,
                "n_flagged": int(len(flagged)),
                "offers": "eligibility_criterion"})


def _energy_adjustment(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    col = _reference_column(df, "kcal")
    if col is None:
        return None
    others = [c for c in _numeric(df) if c != col]
    if len(others) < 3:
        return None
    return _finding(
        "pack::dietary::energy_adjustment", "warning",
        "Nutrient associations need energy adjustment",
        (f"`{col}` is total energy, and {len(others):,} other numeric columns "
         f"are candidate nutrients."),
        ("Every nutrient association is confounded by total intake — people who "
         "eat more of everything eat more of anything. That the adjustment is "
         "needed is not in dispute. The residual method is the default form "
         "because it decorrelates the nutrient from energy explicitly, which "
         "makes the resulting coefficient interpretable; nutrient density is "
         "offered beside it, and that choice is a convention rather than a "
         "fact."),
        confidence="high", pack=DIETARY, marker="derived",
        columns=[col],
        params={"energy_column": col, "default_form": "residual",
                "alternative": "nutrient_density"})


# ── survey ───────────────────────────────────────────────────────────────────

# The response sets an instrument declares. A block of columns sharing exactly
# one of these is a scale; anything else is a set of numbers that happen to be
# small.
_LIKERT_SETS = ({1, 2, 3, 4, 5}, {1, 2, 3, 4}, {1, 2, 3, 4, 5, 6, 7},
                {0, 1, 2, 3}, {0, 1, 2, 3, 4})


# A response category holding more than this share of answers is a count
# distribution, not a response distribution. Measured, not guessed: the survey
# fixture's busiest category holds 29% and a low-expression gene's zero bin
# holds 78%.
_MAX_MODAL_SHARE = 0.60
# Required of most of the block, not all of it. A real instrument has floor
# effects — a screening item four in five people answer "never" is normal — and
# a rule that excluded those would reject the instruments it exists to find.
_MIN_SHARE_BALANCED = 0.7


def likert_block(df: pd.DataFrame, minimum: int = 8) -> Optional[Dict[str, Any]]:
    """The largest set of columns sharing one declared response scale.

    Shared exactly, not approximately. Two columns on 1–5 and one on 1–7 are two
    instruments or one instrument and a stray, and averaging across them is the
    error the detector exists to avoid proposing.

    **And the block must look like RESPONSES, not like small counts.** This is
    the discriminator, and guard #2 found the need for it: 30 low-expression
    genes in `genomics_expression.csv` all take values in {0, 1, 2, 3}, share
    that scale exactly, and would otherwise have been read as a 30-item
    instrument — the survey pack firing on a count matrix, which is precisely
    the authoritative false assertion `DOMAIN_PACKS.md` §05 says would embarrass
    us.

    The evidence that separates them is the shape of the distribution, and it is
    stark: an instrument's categories are all used and roughly comparable — the
    survey fixture's busiest category holds 29% — while a low-expression gene
    decays, with 78% of samples at zero and one or two at the top. So a block
    needs every category of its scale used, and no category dominating, in most
    of its columns.
    """
    by_scale: Dict[Tuple[int, ...], List[str]] = {}
    for c in _numeric(df):
        s = df[c].dropna()
        if s.empty:
            continue
        try:
            values = set(int(v) for v in s.unique() if float(v).is_integer())
        except (TypeError, ValueError, OverflowError):
            continue
        if len(values) != s.nunique():
            continue                                    # non-integral values
        for scale in _LIKERT_SETS:
            if values and values <= scale and len(values) >= len(scale) - 1:
                by_scale.setdefault(tuple(sorted(scale)), []).append(str(c))
                break
    if not by_scale:
        return None
    scale, columns = max(by_scale.items(), key=lambda kv: len(kv[1]))
    if len(columns) < minimum:
        return None

    balanced = 0
    for c in columns:
        s = df[c].dropna()
        if s.empty:
            continue
        used = set(int(v) for v in s.unique())
        if used != set(scale):
            continue                                    # a category never used
        if float(s.value_counts(normalize=True).max()) <= _MAX_MODAL_SHARE:
            balanced += 1
    if balanced / len(columns) < _MIN_SHARE_BALANCED:
        return None
    return {"scale": list(scale), "columns": columns}


def _ordinal_declared(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    block = likert_block(df)
    if block is None:
        return None
    scale, columns = block["scale"], block["columns"]
    return _finding(
        "pack::survey::ordinal_declared", "info",
        f"{len(columns):,} columns share one {len(scale)}-point response scale",
        (f"Every value in them is one of {scale}. The block runs "
         f"`{columns[0]}` … `{columns[-1]}`."),
        ("The order comes from the instrument, not from the data — which makes "
         "the encoding row-local: the number for a row depends on that row's "
         "own answer and on nothing else, so it is applied now rather than "
         "fitted inside the training folds. An encoding derived from the "
         "observed frequencies would be a different object, would have to be "
         "deferred, and would silently change meaning between cohorts."),
        confidence="high", pack=SURVEY, marker="derived",
        columns=columns[:10],
        params={"scale": scale, "columns": columns, "encoding": "declared"})


# ── genomics ─────────────────────────────────────────────────────────────────

def count_matrix(df: pd.DataFrame, minimum: int = 100) -> Optional[Dict[str, Any]]:
    """A block of non-negative integer columns wide enough to be an assay.

    Integrality is the whole reading. Counts and concentrations are different
    objects and the difference decides whether a log transform is derived — it
    is, for concentrations, because they combine multiplicatively — or merely
    one option among several, which is what it is for counts.
    """
    cols = []
    for c in _numeric(df):
        s = df[c].dropna()
        if s.empty or float(s.min()) < 0:
            continue
        try:
            if not np.all(np.equal(np.mod(s.to_numpy(dtype=float), 1), 0)):
                continue
        except (TypeError, ValueError):
            continue
        cols.append(str(c))
    if len(cols) < minimum:
        return None
    return {"columns": cols, "p_over_n": len(cols) / max(len(df), 1)}


def _counts_at_p_over_n(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    block = count_matrix(df)
    if block is None or block["p_over_n"] < 2.0:
        return None
    cols = block["columns"]
    depth = df[cols].sum(axis=1)
    spread = float(depth.max() / max(depth.min(), 1))
    return _finding(
        "pack::genomics::counts_p_over_n", "warning",
        f"{len(cols):,} count columns against {len(df):,} samples",
        (f"Every value in them is a non-negative integer, p/n is "
         f"{block['p_over_n']:.1f}, and total depth per sample varies "
         f"{spread:.1f}-fold."),
        ("Two consequences, and only one of them is ours to settle. Model "
         "ranking is: at this p/n an unregularized fit is not merely optimistic "
         "but degenerate, so regularized models rank first and distance-based "
         "ones last — ordered, never filtered. Normalization is NOT: CPM, TPM "
         "and VST answer different questions and are not interchangeable, and "
         "the right one depends on the assay and on what you are asking. "
         "**No normalization default is asserted here**, and that is the "
         "considered position rather than an omission."),
        confidence="high", pack=GENOMICS, marker="derived",
        columns=cols[:8],
        params={"n_features": len(cols), "p_over_n": round(block["p_over_n"], 2),
                "depth_spread": round(spread, 2),
                "normalization_default": None,
                "model_prior": "regularized_first"})


# ─────────────────────────────────────────────────────────────────────────────
# Reframing — a pack changes the ANSWER, not the question
# ─────────────────────────────────────────────────────────────────────────────

_ASSAY_PACKS = (METABOLOMICS, GENOMICS)


def _wide_shape_note(lens: Sequence[str], df: pd.DataFrame) -> Optional[str]:
    if METABOLOMICS in lens and _is_assay_wide(df):
        return ("These are different analytes, not one analyte measured "
                "several times. An untargeted panel names its features by mass "
                "and retention time, which reads as a numbered series to a "
                "general-purpose importer. Reshaping to long format would "
                "rebuild what a row is and is not what this table needs.")
    if GENOMICS in lens and count_matrix(df) is not None:
        return ("These are different genes, not one gene measured several "
                "times. Reshaping to long format would rebuild what a row is.")
    if SURVEY in lens and likert_block(df) is not None:
        return ("These are the items of one instrument, not one quantity "
                "measured several times. Items are combined by scoring the "
                "scale, which is a decision about the instrument, not by "
                "reshaping the table.")
    return None


def reframe(findings: List[Dict[str, Any]], lens: Sequence[str],
            df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Annotate engine findings the lens reads differently. Never deletes one.

    **Annotation, not suppression**, and the distinction is the guard. A pack
    that DELETED `wide_repeated_measures` would also delete it on
    `clinic_visits.csv`, where `bp_1`/`bp_2`/`bp_3` is exactly what the finding
    is for and the reading is correct. What changes here is the ANSWER —
    severity drops to `info`, the offer is withdrawn, and the reason is carried
    on the finding so the record can say which lens said so.

    Returns a new list; the input findings are copied rather than mutated,
    because two callers reading one finding must not see each other's edits.
    """
    lens = list(lens or [])
    if not lens or lens == [OTHER]:
        return list(findings)

    out: List[Dict[str, Any]] = []
    wide_note = _wide_shape_note(lens, df)
    counts = count_matrix(df) if GENOMICS in lens else None
    count_cols = set(counts["columns"]) if counts else set()

    for raw in findings:
        f = dict(raw)
        if f.get("id") == "wide_repeated_measures" and wide_note:
            f["severity"] = "info"
            f["fix_kind"] = "none"
            f["fix_label"] = ""
            f["reframed_by"] = [k for k in lens if k in
                                (METABOLOMICS, GENOMICS, SURVEY)]
            f["reframe_note"] = wide_note
            f["title"] = "The wide shape is expected here"
        elif (f.get("id", "").startswith("sentinel_missing__")
              and count_cols
              and str((f.get("params") or {}).get("column")) in count_cols):
            f["severity"] = "info"
            f["fix_kind"] = "none"
            f["fix_label"] = ""
            f["reframed_by"] = [GENOMICS]
            f["reframe_note"] = (
                "This is a count column, and a small integer in a "
                "low-expression gene is a count rather than a missing-value "
                "code. The detector reads an integral column with few distinct "
                "values as a coded variable, which is the right reading for a "
                "survey item and the wrong one for a transcript count.")
            f["title"] = (f"`{(f.get('params') or {}).get('column')}` holds low "
                          f"counts, not missing-value codes")
        out.append(f)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The packs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Pack:
    key: str
    label: str
    detectors: Tuple[Callable[[pd.DataFrame], Optional[Dict[str, Any]]], ...] = ()
    # Priors the pack sets on questions that already exist. A prior is not a
    # finding and never becomes a question — it changes what the existing
    # question DEFAULTS to and states why.
    priors: Dict[str, Any] = field(default_factory=dict)


PACKS: Dict[str, Pack] = {
    METABOLOMICS: Pack(
        key=METABOLOMICS, label=LENS_LABELS[METABOLOMICS],
        detectors=(_left_censored, _acquisition_order, _pooled_qc),
        priors={
            "log_transform": {
                "marker": "derived",
                "reason": ("Concentrations are bounded below by zero and "
                           "combine multiplicatively, so the resulting "
                           "distribution is log-normal by construction rather "
                           "than by convention.")},
            "scaling": {
                "marker": "convention", "variant": "pareto",
                "reason": ("The field convention here is Pareto scaling. "
                           "Auto-scaling gives every feature equal weight "
                           "including noise-dominated low-abundance ones; "
                           "dividing by the square root of the standard "
                           "deviation retains some magnitude information. A "
                           "defensible compromise, not a fact — auto-scaling "
                           "is offered beside it.")},
            "missingness_direction": {
                "marker": "derived", "mechanism": "below_detection_limit",
                "reason": ("A blank here is usually a non-detection, so it "
                           "carries information about magnitude rather than "
                           "about the participant.")},
        }),
    GENOMICS: Pack(
        key=GENOMICS, label=LENS_LABELS[GENOMICS],
        detectors=(_counts_at_p_over_n,),
        priors={
            "model_ranking": {
                "marker": "derived", "prefer": "regularized",
                "reason": ("At p much greater than n an unregularized fit is "
                           "degenerate, and a distance metric over hundreds of "
                           "features is dominated by noise.")},
            # Deliberately present and deliberately empty. An ABSENT key would
            # be indistinguishable from a pack that had not thought about
            # normalization; this one has, and declined.
            "normalization": {
                "marker": "offered", "variant": None,
                "reason": ("CPM, TPM and VST are not interchangeable and the "
                           "choice depends on the assay and the question. No "
                           "default is asserted, which is a position rather "
                           "than an omission.")},
        }),
    DIETARY: Pack(
        key=DIETARY, label=LENS_LABELS[DIETARY],
        detectors=(_compositional, _implausible_intake, _energy_adjustment),
        priors={
            "repeat_treatment": {
                "marker": "derived", "treatment": "average",
                "reason": ("A single 24-hour recall is a noisy estimate of "
                           "usual intake, and that noise attenuates "
                           "diet-outcome associations toward the null. Using "
                           "the mean reduces it. This is measurement-error "
                           "reduction, not information loss.")},
            "energy_adjustment": {
                "marker": "convention", "variant": "residual",
                "reason": ("The residual method decorrelates the nutrient from "
                           "energy explicitly, which makes the resulting "
                           "coefficient interpretable. Nutrient density is "
                           "offered beside it.")},
        }),
    CLINICAL: Pack(
        key=CLINICAL, label=LENS_LABELS[CLINICAL],
        detectors=(),
        priors={
            # The whole clinical pack, and its thinness is the point:
            # physiologic bounds and unit harmonization already exist in the
            # core. This adds ONE prior, and it points the opposite way from
            # the metabolomics one.
            "missingness_direction": {
                "marker": "offered", "mechanism": "not_ordered",
                "reason": ("Missingness here often means the test was not "
                           "ordered — a clinician saw no reason to run it — "
                           "which is informative about the patient rather than "
                           "about the measurement. That is the opposite "
                           "direction from an assay, where a blank usually "
                           "means below the detection limit. The mechanism "
                           "question already asks; this supplies the prior, "
                           "not the answer.")},
        }),
    SURVEY: Pack(
        key=SURVEY, label=LENS_LABELS[SURVEY],
        detectors=(_ordinal_declared,),
        priors={
            "ordinal_encoding": {
                "marker": "derived", "source": "instrument",
                "reason": ("The order comes from the instrument, which makes "
                           "the encoding row-local rather than a distribution "
                           "the app has to learn.")},
            "reverse_coding": {
                "marker": "offered", "variant": None,
                "reason": ("Reverse-coding requires a codebook the app does "
                           "not have. Inferring it from item correlations "
                           "would be right whenever the instrument is "
                           "unidimensional and confidently wrong whenever two "
                           "subscales measure opposing constructs — and "
                           "nothing in the numbers separates those cases.")},
        }),
    OTHER: Pack(key=OTHER, label=LENS_LABELS[OTHER]),
}


def findings(df: pd.DataFrame, lens: Sequence[str]) -> List[Dict[str, Any]]:
    """What the selected packs see in this table. Empty is the common answer.

    Ordered by pack, then by the order the detectors are declared in, so the
    same table and the same lens always produce the same list — the Router's
    determinism requirement applies to anything feeding it.
    """
    out: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return out
    for key in normalize_quiet(lens):
        pack = PACKS.get(key)
        if pack is None:
            continue
        for detector in pack.detectors:
            try:
                found = detector(df)
            except Exception:
                # A detector that cannot read this table reports nothing. It
                # must not take the interview down with it, and it must not be
                # silent about having failed either.
                from turbotab import devchecks
                devchecks.swallowed(
                    f"packs.{key}::{getattr(detector, '__name__', '?')}",
                    _last_exception(),
                    "this pack detector found nothing, and would have been "
                    "indistinguishable from one that legitimately found nothing")
                continue
            if found:
                out.append(found)
    return out


def _last_exception() -> BaseException:
    import sys
    return sys.exc_info()[1] or RuntimeError("unknown")


def normalize_quiet(keys: Optional[Sequence[str]]) -> List[str]:
    """`normalize`, for callers that already hold a recorded answer."""
    return [k for k in (keys or []) if k in LENS_KEYS]


def priors(lens: Sequence[str], name: str) -> List[Dict[str, Any]]:
    """Every selected pack's prior on one question, with the pack named.

    A list rather than one value, because the lens is multi-select and two packs
    can have opposite priors on the same question — metabolomics and clinical do,
    on missingness, and that disagreement is real. Resolving it silently would
    pick one field's reading of a dataset that is both.
    """
    out = []
    for key in normalize_quiet(lens):
        prior = PACKS[key].priors.get(name)
        if prior:
            out.append({"pack": key, "label": LENS_LABELS[key], **prior})
    return out


def question(suggestion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The lens question, as the Router and the page both read it."""
    return {
        "key": "state_lens",
        "clause": "lockbox-01",
        "title": LENS_TITLE,
        "why": LENS_WHY,
        "consumer": LENS_CONSUMER,
        "multi_select": True,
        "options": [{"key": k, "label": LENS_LABELS[k]} for k in LENS_KEYS],
        "suggestion": suggestion or {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Detection — a suggestion and a contradiction detector, never the answer
# ─────────────────────────────────────────────────────────────────────────────

def suggest(df: pd.DataFrame) -> Dict[str, Any]:
    """What the shape of this table hints at, offered beside the options.

    Never pre-selected. §01: *the user's answer is the answer; detection never
    overrides it.* This exists so a user who does not know the vocabulary has
    somewhere to start, and so the contradiction detector below has a reading to
    disagree with.
    """
    hints: List[Dict[str, str]] = []
    if df is None or df.empty:
        return {"hints": hints}
    if count_matrix(df) is not None:
        hints.append({"lens": GENOMICS,
                      "because": "every one of these columns holds "
                                 "non-negative whole numbers, which is what a "
                                 "count matrix looks like"})
    if _is_assay_wide(df) and count_matrix(df) is None:
        hints.append({"lens": METABOLOMICS,
                      "because": f"there are {len(_numeric(df)):,} measurement "
                                 f"columns across {len(df):,} rows"})
    if likert_block(df) is not None:
        block = likert_block(df)
        hints.append({"lens": SURVEY,
                      "because": f"{len(block['columns']):,} columns share one "
                                 f"{len(block['scale'])}-point response scale"})
    if _reference_column(df, "kcal") is not None:
        hints.append({"lens": DIETARY,
                      "because": "there is a total-energy column"})
    return {"hints": hints}


def contradiction(df: pd.DataFrame, lens: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Evidence that the stated lens and the table disagree.

    §01, and the same escalation rule as everywhere else: *escalate on evidence
    that a reading is wrong, never on the size of the consequence.* The example
    the document gives is the one implemented — an answer of "clinical
    measurements" over a table of hundreds of assay features with a run-order
    column is a disagreement worth raising.

    Advisory, not a refusal. The user may be right and the shape unusual.
    """
    lens = normalize_quiet(lens)
    if not lens or df is None or df.empty:
        return None
    assay = _is_assay_wide(df, minimum=100)
    if assay and not any(k in lens for k in _ASSAY_PACKS):
        stated = ", ".join(LENS_LABELS[k].lower() for k in lens)
        return {
            "kind": "stated_lens_but_shape_is_an_assay",
            "message": (
                f"This table has {len(_numeric(df)):,} numeric columns across "
                f"{len(df):,} rows, which is the shape of an assay panel, and "
                f"you described it as {stated}. One of those two readings is "
                f"probably wrong, and which one changes what is looked for."),
            "n_numeric": len(_numeric(df)), "n_rows": len(df),
            "suggests": [k for k in (GENOMICS if count_matrix(df) else
                                     METABOLOMICS,)],
        }
    return None
