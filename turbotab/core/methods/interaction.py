"""Effect modification and interaction as two declared objects (MODELING_SEQUENCE §1 row 7, the
package FORM).

The review of the modeling sequence (row 7, quoting VanderWeele 2009, *Epidemiology* 20:863):
"Interaction is defined in terms of the effects of 2 interventions whereas effect modification is
defined in terms of the effect of one intervention varying across strata of a second variable."
So they are declared apart (``set_modification``):

* **Effect modification** by M: the exposure A's effect within each stratum of M, on A's own
  adjustment set (M enters beside it, with its product terms).
* **Interaction** of A with a second exposure B: their joint effect. The adjustment set is asked
  again for B (the disjunctive cause criterion's answers, with B as the exposure), and the model
  adjusts for the confounders of each (Knol & VanderWeele 2012, *Int J Epidemiol* 41:514: "List the
  A-D and B-D confounders adjusted for").

**What is reported** (Knol & VanderWeele 2012: "We strongly encourage reporting both additive and
multiplicative interaction estimates and CIs whenever interaction or effect modification is of
interest"): every combination of A (at its declared contrast, a0 → a1) and of M's strata against a
**single reference** (a0 in M's reference stratum); A's effect within each stratum; on a ratio scale
(odds, hazard or cumulative-odds ratios) the **relative excess risk due to interaction**

    RERI = RR₁₁ − RR₁₀ − RR₀₁ + 1

with Hosmer & Lemeshow's (1992, *Epidemiology* 3:452) delta-method interval (the gradient of RERI in
the coefficients, ``e^{L₁₁}c₁₁ − e^{L₁₀}c₁₀ − e^{L₀₁}c₀₁``, against the table's covariance), beside the
**multiplicative** measure, the ratio of ratios ``RR₁₁/(RR₁₀ RR₀₁)``; on a difference scale (a mean
or a risk difference) the additive interaction is the difference of differences, and no
multiplicative measure applies. A RERI from odds ratios approximates the RERI of risk ratios when the
outcome is rare (VanderWeele & Knol 2014), and says so. The heterogeneity test is the Wald test that
every product term is zero (D1 under multiple imputation).

**Declared before estimates.** A modifier declared after the estimates were seen (the analysis-plan
lock) is labeled "suggested by data inspection", and every modifier counts in the family of tests
the record states (MODELING_SEQUENCE §1 row 7: "post hoc subgroups labeled 'suggested by data
inspection' and counted in the family").

**Under multiple imputation** the product terms are part of the analysis model, so the imputation
is compatible with it: SMC-FCS with the products in the substantive model (Bartlett et al. 2015;
MODELING_SEQUENCE §1.1: "splines, interactions and logistic or Cox outcomes use SMC-FCS"), every
scalar pooled by Rubin's rules, the heterogeneity test by D1.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

POST_HOC = "suggested by data inspection"
KNOL = "Knol & VanderWeele 2012, Int J Epidemiol 41:514"
HOSMER = "Hosmer & Lemeshow 1992, Epidemiology 3:452"
VANDERWEELE_2009 = "VanderWeele 2009, Epidemiology 20:863"
RARE = ("a RERI from odds ratios approximates the RERI of risk ratios only when the outcome is "
        "rare (VanderWeele & Knol 2014)")
TIMES = "×"


def tests_product_terms(family: Any) -> bool:
    """Whether ``family`` tests product terms here: it reports coefficients and its
    ``inference_decl`` declares ``product_terms`` (MODEL_FAMILY_CONTRACT C2)."""
    from turbotab.core.models.base import reports_coefficients

    decl = family.inference_decl
    return reports_coefficients(family) and decl is not None and decl.product_terms


def weighted_products(family: Any) -> bool:
    """Whether the survey weights enter ``family``'s product terms under the surveyed population:
    it has a design-based estimator (``design_based``) and its model is a weighted sum of the
    values as given (``linear_in_values``), the least-squares and logistic fits the weighted
    interaction is built for. Any other family's interaction is fit without the weights and says
    so."""
    decl = family.inference_decl
    return decl is not None and decl.design_based and family.linear_in_values


def __getattr__(name: str) -> Any:
    """``SUPPORTED``, the families that test product terms, read from the registry: the agreement
    check in ``test_mc1_family_declarations`` still imports the name the declaration replaced."""
    if name == "SUPPORTED":
        from turbotab.core.models.base import families

        return tuple(f.key for f in families() if tests_product_terms(f))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _tick(value: Any) -> str:
    return f"`{value}`"


def _listing(items: Sequence[Any], limit: int = 6) -> str:
    from turbotab.core.voice import listing

    return listing(list(items), limit=limit)


# ── the declared objects, as the state holds them ────────────────────────────


def bound_exposure(state: Any) -> str | None:
    """The one declared exposure a modifier is about (an exposure family has none)."""
    from turbotab.core.methods.exposure_form import exposure_of

    return exposure_of(state)


def _held(state: Any) -> dict[str, Any]:
    exposure = bound_exposure(state)
    if exposure is None or getattr(state, "purpose", None) != "inference":
        return {}
    return {str(m): s for m, s in (getattr(state, "modifications", None) or {}).items()
            if s is not None and _get(s, "exposure") in (None, exposure)}


def declared(state: Any) -> dict[str, Any]:
    """Each declared modifier or second exposure that holds: declared for the present exposure
    (a new exposure asks them again, as it asks the adjustment answers), and not withdrawn."""
    return {m: s for m, s in _held(state).items() if not _get(s, "withdrawn")}


def withdrawn_after_estimates(state: Any) -> list[str]:
    """The modifiers withdrawn after the estimates were seen: no longer estimated, still counted in
    the family of tests (the analyst saw their tests)."""
    return sorted(m for m, s in _held(state).items() if _get(s, "withdrawn"))


def second_covariates(state: Any, modifier: str) -> list[str]:
    """The covariates the adjustment question asks again with B, the second exposure, as the
    exposure: every settled predictor but A and B, the grouping's fixed effects and, under the
    dietary lens, total energy (the energy question decides it)."""
    from turbotab.core import estimand as est

    exposure = bound_exposure(state)
    roles = est.predictor_roles(state)
    fe = est.fixed_effects_column(state)
    dietary = "dietary" in (getattr(state, "lens", None) or [])
    return [c for c, r in roles.items()
            if c not in (exposure, modifier) and c != fe and not (dietary and r == "energy")]


def missing_answers(state: Any, modifier: str, spec: Any) -> list[str]:
    """The covariates an interaction's second exposure still needs answers for."""
    if _get(spec, "kind") != "interaction":
        return []
    answers = _get(spec, "answers") or {}
    return [c for c in second_covariates(state, modifier) if c not in answers]


def complete_modifications(state: Any) -> dict[str, Any]:
    """The declared modifiers whose questions are all answered."""
    return {m: s for m, s in declared(state).items() if not missing_answers(state, m, s)}


def modification_gate(state: Any) -> tuple[str, str | None] | None:
    """The Router's gate: not under prediction or for an exposure family; stated (one tap away)
    while none is declared; asked while an interaction's second exposure waits for its answers."""
    from turbotab.core import estimand as est

    purpose = getattr(state, "purpose", None)
    if purpose == "prediction":
        return ("not_applicable", "Under prediction no effect is estimated, so no effect modifier "
                                  "or second study factor is declared.")
    if purpose != "inference":
        return None
    spec = est.current_estimand(state)
    if spec is None:
        # No exposure to declare: no modifier either; else it waits behind the estimand's question.
        found = est.estimand_gate(state)
        return found if found is not None and found[0] == "not_applicable" else None
    if _get(spec, "family"):
        return ("not_applicable", "A family of study factors reports each member in turn; effect "
                                  "modification is declared for one study factor.")
    if not declared(state):
        return ("skipped", "No effect modifier or second study factor is declared; one can be, before "
                           "the estimates are seen (afterwards it is labeled suggested by data "
                           "inspection).")
    return None


def modification_answer(state: Any) -> Any:
    """Answered once every declared modifier's questions are (an interaction's second exposure's
    adjustment answers among them)."""
    found = declared(state)
    if not found or len(complete_modifications(state)) < len(found):
        return None
    return found


def modification_followup(state: Any) -> str | None:
    """What the open question still asks: the adjustment answers for a second exposure."""
    found = declared(state)
    return "adjustment" if found and len(complete_modifications(state)) < len(found) else None


def family_count(state: Any, not_computed: Sequence[str] = ()) -> dict[str, Any]:
    """The family of tests the record states: the exposure's own test and one heterogeneity test
    per declared modifier, those suggested by data inspection counted and named, and those
    withdrawn after the estimates were seen still counted (the analyst saw them). ``not_computed``:
    the declared modifiers whose test the stage could not compute, named as such (a count never
    claims a test that was not run)."""
    found = declared(state)
    gone = withdrawn_after_estimates(state)
    post = sorted(m for m, s in found.items() if _get(s, "post_hoc"))
    before = sorted(m for m in found if m not in post)
    n = 1 + len(found) + len(gone)
    parts = [f"the exposure's effect"]
    if before:
        parts.append(f"{len(before)} declared heterogeneity test{'s' if len(before) > 1 else ''} "
                     f"({_listing(before)})")
    if post:
        parts.append(f"{len(post)} {POST_HOC} ({_listing(post)})")
    if gone:
        parts.append(f"{len(gone)} withdrawn after the estimates were seen ({_listing(gone)})")
    from turbotab.core.voice import listing

    statement = (f"{n} tests in the family: {listing(parts, limit=5, ticked=False)}; the p-values "
                 f"are reported unadjusted with the number of tests stated")
    missing = [m for m in not_computed if m in found]
    if missing:
        statement += (f"; the heterogeneity test of {_listing(missing)} could not be computed on "
                      f"these rows (said with {'it' if len(missing) == 1 else 'each'}) and is "
                      f"counted as declared")
    return {"n_tests": n, "declared": before, "post_hoc": post, "withdrawn": gone,
            "not_computed": missing, "statement": statement}


# ── whether each stratum can carry an effect ─────────────────────────────────


class NotEstimable(ValueError):
    """A modification the rows cannot estimate, said in plain words (any other failure is said
    without the raw exception text)."""


def strata_problems(modifier: str, exposure: str, m_values: Any, a_values: Any, events: Any,
                    task: str | None, *, categorical: bool,
                    levels: Sequence[float] | None = None) -> list[str]:
    """Why the exposure's effect cannot be estimated within some stratum of the modifier, one
    plain line per stratum: fewer than two rows, or one value of the exposure among them (no
    contrast within it); for a yes/no outcome no event or no non-event, for a time to event no
    event, for an ordered outcome one level (the stratum's effect is then not identifiable, and a
    fit returns an overflow or a bracketing failure instead of a number). A numeric modifier with
    more than two values has no strata of rows (its effect is read at stated values), so nothing
    is checked. ``events``: the event indicator (yes/no, time to event) or the outcome's codes
    (ordered); None for a numeric outcome."""
    m = pd.Series(np.asarray(m_values, dtype=object))
    a = pd.to_numeric(pd.Series(np.asarray(a_values, dtype=object)), errors="coerce").to_numpy(float)
    keep = (m.notna().to_numpy() & np.isfinite(a))
    ev = None if events is None else np.asarray(events)
    if ev is not None and ev.dtype.kind == "f":
        keep &= np.isfinite(ev)
    m, a = m[keep].reset_index(drop=True), a[keep]
    ev = ev[keep] if ev is not None else None
    if not categorical:
        numbers = pd.to_numeric(m, errors="coerce")
        distinct = np.unique(numbers.dropna().to_numpy(float))
        if len(distinct) != 2 or levels:
            return []
        m = numbers
        strata = [float(v) for v in distinct]
    else:
        strata = sorted(pd.unique(m), key=str)
    out = []
    for s in strata:
        rows = (m == s).to_numpy()
        n = int(rows.sum())
        label = f"`{modifier}` = `{_number(s) if isinstance(s, float) else s}`"
        if n < 2 or len(np.unique(a[rows])) < 2:
            out.append(f"{label} has {n:,} row{'s' if n != 1 else ''}"
                       + (f", all at one value of `{exposure}`" if n >= 2 else "")
                       + f", so `{exposure}`'s effect within it cannot be estimated")
            continue
        if ev is None:
            continue
        here = ev[rows]
        if task == "binary":
            k = int((here.astype(float) > 0).sum())
            if k == 0 or k == n:
                out.append(f"{label} has {'no event' if k == 0 else 'only events'} among its "
                           f"{n:,} rows, so the odds ratio within it cannot be estimated")
        elif task == "time_to_event":
            if int((here.astype(float) > 0).sum()) == 0:
                out.append(f"{label} has no event among its {n:,} rows, so the hazard ratio "
                           f"within it cannot be estimated")
        elif task == "ordinal" and len(np.unique(here)) < 2:
            out.append(f"{label} has one level of the outcome among its {n:,} rows, so the "
                       f"cumulative odds ratio within it cannot be estimated")
    return out


# ── the arithmetic (pure: tested against a hand computation) ──────────────────


def _critical(df: float | None, level: float = 0.95) -> float:
    from scipy import stats

    q = 0.5 + level / 2
    return float(stats.norm.ppf(q) if df is None or not math.isfinite(df) else stats.t.ppf(q, df))


def _p(z: float, df: float | None) -> float:
    from scipy import stats

    if df is None or not math.isfinite(df):
        return float(2 * stats.norm.sf(abs(z)))
    return float(2 * stats.t.sf(abs(z), df))


def linear_combination(beta: np.ndarray, cov: np.ndarray, c: np.ndarray,
                       df: float | None = None, level: float = 0.95) -> dict[str, float]:
    """c'β with its standard error √(c'Vc), interval and p (t on ``df``, else normal)."""
    est = float(c @ beta)
    var = float(c @ cov @ c)
    se = math.sqrt(max(var, 0.0))
    crit = _critical(df, level)
    return {"estimate": est, "variance": var, "se": se, "ci_low": est - crit * se,
            "ci_high": est + crit * se, "p": _p(est / se, df) if se > 0 else float("nan")}


def reri(beta: np.ndarray, cov: np.ndarray, c10: np.ndarray, c01: np.ndarray, c11: np.ndarray,
         df: float | None = None, level: float = 0.95) -> dict[str, float]:
    """RERI = e^{c₁₁'β} − e^{c₁₀'β} − e^{c₀₁'β} + 1 with Hosmer & Lemeshow's delta-method interval:
    the gradient g = e^{c₁₁'β}c₁₁ − e^{c₁₀'β}c₁₀ − e^{c₀₁'β}c₀₁, its variance g'Vg."""
    e11, e10, e01 = (math.exp(float(c @ beta)) for c in (c11, c10, c01))
    est = e11 - e10 - e01 + 1.0
    g = e11 * c11 - e10 * c10 - e01 * c01
    var = float(g @ cov @ g)
    se = math.sqrt(max(var, 0.0))
    crit = _critical(df, level)
    return {"estimate": est, "variance": var, "se": se, "ci_low": est - crit * se,
            "ci_high": est + crit * se, "p": _p(est / se, df) if se > 0 else float("nan")}


def contrast_vectors(names: Sequence[str], a_values: Mapping[str, Sequence[float]],
                     m_values: Mapping[str, Mapping[str, float]],
                     products: Sequence[tuple[str, str, str]]) -> dict[str, np.ndarray]:
    """x(a, s): each combination of A's value (``a_values``: 'a0'/'a1' → its matrix columns'
    values, in order of ``a_cols``) and M's stratum (``m_values``: stratum → {M column: value})
    as a vector over the table's coefficients ``names`` (the intercept and the covariates 0: they
    cancel in every contrast), with each product term the product of its two columns' values."""
    where = {str(n): i for i, n in enumerate(names)}
    out: dict[str, np.ndarray] = {}
    for a, avals in a_values.items():
        for s, mvals in m_values.items():
            x = np.zeros(len(names))
            for col, v in avals.items():
                if col in where:
                    x[where[col]] = v
            for col, v in mvals.items():
                if col in where:
                    x[where[col]] = v
            for a_col, m_col, name in products:
                if name in where:
                    x[where[name]] = float(avals.get(a_col, 0.0)) * float(mvals.get(m_col, 0.0))
            out[f"{a}|{s}"] = x
    return out


def measures(names: Sequence[str], beta: np.ndarray, cov: np.ndarray, *,
             a_values: Mapping[str, Mapping[str, float]],
             m_values: Mapping[str, Mapping[str, float]], reference: str,
             products: Sequence[tuple[str, str, str]], ratio: bool,
             df: float | None = None) -> dict[str, dict[str, float]]:
    """Every quantity Knol & VanderWeele ask for, on the linear predictor's scale (log ratios on a
    ratio scale), each with its variance: ``effect|s`` (A's effect within stratum s),
    ``joint|a0|s`` and ``joint|a1|s`` (each combination against the single reference a0 in the
    reference stratum), ``mult|s`` (the multiplicative interaction), and ``reri|s`` on a ratio
    scale or ``additive|s`` (the difference of differences) on a difference scale."""
    x = contrast_vectors(names, a_values, m_values, products)
    ref0 = x[f"a0|{reference}"]
    out: dict[str, dict[str, float]] = {}
    for s in m_values:
        out[f"effect|{s}"] = linear_combination(beta, cov, x[f"a1|{s}"] - x[f"a0|{s}"], df)
        out[f"joint|a1|{s}"] = linear_combination(beta, cov, x[f"a1|{s}"] - ref0, df)
        if s == reference:
            continue
        out[f"joint|a0|{s}"] = linear_combination(beta, cov, x[f"a0|{s}"] - ref0, df)
        c10 = x[f"a1|{reference}"] - ref0
        c01 = x[f"a0|{s}"] - ref0
        c11 = x[f"a1|{s}"] - ref0
        out[f"mult|{s}"] = linear_combination(beta, cov, c11 - c10 - c01, df)
        if ratio:
            out[f"reri|{s}"] = reri(beta, cov, c10, c01, c11, df)
        else:
            out[f"additive|{s}"] = linear_combination(beta, cov, c11 - c10 - c01, df)
    return out


def pool_measures(per_copy: Sequence[Mapping[str, Mapping[str, float]]],
                  df_com: float | None) -> dict[str, dict[str, float]]:
    """Each quantity pooled over the copies by Rubin's rules (one copy: its own)."""
    from turbotab.core.methods.imputation import pool_scalar

    if len(per_copy) == 1:
        return {k: dict(v) for k, v in per_copy[0].items()}
    out: dict[str, dict[str, float]] = {}
    for key in per_copy[0]:
        pooled = pool_scalar([c[key]["estimate"] for c in per_copy],
                             [c[key]["variance"] for c in per_copy], df_com)
        out[key] = {"estimate": pooled.estimate, "variance": pooled.total,
                    "se": math.sqrt(pooled.total) if pooled.total > 0 else 0.0,
                    "ci_low": pooled.ci_low, "ci_high": pooled.ci_high, "p": pooled.p,
                    "df": pooled.df}
    return out


# ── the artifact ─────────────────────────────────────────────────────────────


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Quantity(_Model):
    """One reported quantity: on a ratio scale ``estimate`` is the ratio (the RERI its own value);
    on a difference scale the difference."""

    stratum: str
    a: Literal["a0", "a1"] | None = None
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    p: float | None = None


class Heterogeneity(_Model):
    statistic: float | None
    df_num: int | None
    df_den: float | None
    distribution: Literal["F", "chi2"] | None
    p: float | None
    caption: str


class ModificationFit(_Model):
    family: str
    label: str
    measure: str  # "odds ratio", "hazard ratio", "difference in mean `y`"
    scale: Literal["ratio", "difference"]
    n_rows: int
    effects: list[Quantity] = []  # A's effect within each stratum
    joint: list[Quantity] = []  # each combination against the single reference
    multiplicative: list[Quantity] = []  # the ratio of ratios (ratio scale only)
    reri: list[Quantity] = []  # ratio scale
    additive: list[Quantity] = []  # difference scale: the difference of differences
    heterogeneity: Heterogeneity | None = None
    pooled: str | None = None  # how the copies were pooled
    # a logistic model the data separate: Firth's penalized likelihood, profile intervals and a
    # penalized likelihood-ratio heterogeneity test, the RERI without an interval
    penalized: bool = False
    concerns: list[str] = []


class ModificationResult(_Model):
    modifier: str
    kind: Literal["effect_modification", "interaction"]
    exposure: str
    status: Literal["declared", "suggested by data inspection"]
    contrast: str  # A's contrast, in words
    low: float
    high: float
    strata: list[str]
    reference: str  # M's reference stratum
    adjusted_for: list[str]
    second_adjusted_for: list[str] = []  # interaction: B's confounders, as its answers derive them
    waiting: list[str] = []  # an interaction's covariates still to be answered for B
    families: list[ModificationFit] = []
    sentence: str
    concerns: list[str] = []


class ModificationArtifact(_Model):
    purpose: str | None
    exposure: str | None
    family_count: dict[str, Any]
    modifications: list[ModificationResult] = []
    methods: str


MODIFICATION_READS = ("modifications", "estimand", "adjustment", "multiplicity", "clusters",
                      "purpose", "models", "task", "event", "target", "roles", "roles_unconfirmed",
                      "role_confirmations", "reading_confirmations", "shape_confirmations",
                      "missing", "survey", "outcome_order", "follow_up", "categorical",
                      "energy_adjustment", "grain", "exposure_forms", "lens", "findings",
                      "column_units", "split", "scales", "batch")


# ── the stage ────────────────────────────────────────────────────────────────


def _matrix_columns(sources: Mapping[str, tuple[set[str], set[str]]], receives: str,
                    raw: str) -> list[str]:
    """The model-matrix columns made of one input: those whose adjusted-lane source is it."""
    out = [c for c, (r, a) in sources.items() if a == {receives}]
    if not out:
        out = [c for c, (r, a) in sources.items() if r == {raw}]
    return out


def _values_at(fitted: Any, X: pd.DataFrame, receives: str, value: float,
               columns: Sequence[str]) -> dict[str, float]:
    """The matrix columns of one input at one value of it: its form applied to a row that holds
    the value (the fitted step's own knots or cut points), else the value itself."""
    from turbotab.core.methods.exposure_form import _before_form, form_step

    step = form_step(fitted)
    if step is not None and receives in step._plan():
        row = _before_form(fitted, X.iloc[[0]]).copy()
        row[receives] = float(value)
        out = step.transform(row)
        return {c: float(out[c].iloc[0]) for c in columns if c in out.columns}
    return {c: float(value) for c in columns}


def _before_form_values(fitted: Any, X: pd.DataFrame, receives: str,
                        matrix: pd.DataFrame) -> np.ndarray:
    from turbotab.core.methods.exposure_form import _before_form, form_step

    if form_step(fitted) is not None:
        frame = _before_form(fitted, X)
        if receives in frame.columns:
            return pd.to_numeric(frame[receives], errors="coerce").to_numpy(dtype=float)
    if receives in matrix.columns:
        return pd.to_numeric(matrix[receives], errors="coerce").to_numpy(dtype=float)
    return np.asarray([], dtype=float)


def _number(v: float) -> str:
    return f"{v:.4g}"


class _Layout:
    """How one modification sits in the matrix: A's and M's columns, A's contrast, M's strata."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def layout(fitted: Any, X: pd.DataFrame, inputs: Sequence[str], exposure: str, modifier: str,
           adjustment: Any, spec: Any, frame: pd.DataFrame) -> _Layout:
    """Read the declared contrast, the strata and the product terms off the fitted pipeline:
    A's columns (its form's, on its final scale), M's columns (indicators of a category, or its
    linear term), A's contrast a0 → a1 (declared, else the interquartile one on the final scale;
    a two-valued A its two values), and M's strata (a category's levels; a two-valued M's two
    values; a numeric M's declared values, else its 25th and 75th percentiles)."""
    from turbotab.core.methods.exposure_form import formed_name
    from turbotab.core.models.linear import model_matrix
    from turbotab.core.stages.effects import matrix_sources

    matrix = model_matrix(fitted, X)
    sources = matrix_sources(fitted, inputs)
    a_receives = formed_name(exposure, adjustment)
    m_receives = formed_name(modifier, adjustment)
    a_cols = _matrix_columns(sources, a_receives, exposure)
    m_cols = _matrix_columns(sources, m_receives, modifier)
    if not a_cols:
        raise ValueError(f"`{exposure}` has no column in the model matrix.")
    if not m_cols:
        raise ValueError(f"`{modifier}` has no column in the model matrix.")
    a_raw = _before_form_values(fitted, X, a_receives, matrix)
    a_raw = a_raw[np.isfinite(a_raw)]
    distinct = np.unique(a_raw)
    low, high = _get(spec, "low"), _get(spec, "high")
    if low is None or high is None:
        if len(distinct) == 2:
            low, high = float(distinct[0]), float(distinct[1])
            contrast = f"{_tick(exposure)} {_number(high)} against {_number(low)}"
        else:
            low, high = (float(v) for v in np.quantile(a_raw, [0.25, 0.75]))
            contrast = (f"{_tick(a_receives)} from {_number(low)} to {_number(high)} (its 25th to "
                        f"75th percentile)")
    else:
        contrast = f"{_tick(a_receives)} from {_number(float(low))} to {_number(float(high))}"
    a_values = {"a0": _values_at(fitted, X, a_receives, float(low), a_cols),
                "a1": _values_at(fitted, X, a_receives, float(high), a_cols)}
    categorical = any(c != m_receives for c in m_cols) and all(
        c.startswith(f"{m_receives}_") or c.startswith(f"{modifier}_") for c in m_cols)
    m_values: dict[str, dict[str, float]] = {}
    if categorical:
        prefix = f"{m_receives}_" if all(c.startswith(f"{m_receives}_") for c in m_cols) \
            else f"{modifier}_"
        levels = {c[len(prefix):]: c for c in m_cols}
        recorded = [str(v) for v in pd.Series(frame[modifier]).dropna().unique()]
        others = sorted(set(recorded) - set(levels), key=str)
        reference = others[0] if others else "reference"
        m_values[reference] = {c: 0.0 for c in m_cols}
        for level, col in levels.items():
            m_values[level] = {c: (1.0 if c == col else 0.0) for c in m_cols}
        product_cols = list(m_cols)
    else:
        m_raw = pd.to_numeric(frame[modifier], errors="coerce").dropna().to_numpy(dtype=float)
        stated = _get(spec, "levels")
        values = sorted(np.unique(m_raw)) if len(np.unique(m_raw)) == 2 and not stated else (
            [float(v) for v in stated] if stated else
            [float(v) for v in np.quantile(m_raw, [0.25, 0.75])])
        main = m_cols[0]
        for v in values:
            m_values[_number(v)] = _values_at(fitted, X, m_receives, v, m_cols)
        reference = _number(values[0])
        product_cols = [main]
    products = [(a, m, f"{a}{TIMES}{m}") for a in a_cols for m in product_cols]
    return _Layout(a_cols=a_cols, m_cols=m_cols, a_values=a_values, m_values=m_values,
                   reference=reference, products=products, contrast=contrast, low=float(low),
                   high=float(high), a_receives=a_receives, m_receives=m_receives)


def with_products(matrix: pd.DataFrame, products: Sequence[tuple[str, str, str]]) -> pd.DataFrame:
    """The matrix with each product term appended."""
    out = matrix.copy()
    for a, m, name in products:
        out[name] = matrix[a].to_numpy(dtype=float) * matrix[m].to_numpy(dtype=float)
    return out


RATIO_TASKS = {"binary": "odds ratio", "time_to_event": "hazard ratio",
               "ordinal": "cumulative odds ratio"}


def _registered(key: str) -> Any:
    """The registered family named ``key``, or None."""
    from turbotab.core.models.base import get_family

    try:
        return get_family(key)
    except KeyError:
        return None


def _measure(task: str, family: Any, target: str) -> tuple[str, bool]:
    """The measure each contrast is reported on: the exponentiated contrast on ``family``'s raw
    scale for ``task`` (``raw_scale``: a log hazard gives a hazard ratio, a cumulative logit's
    latent scale a cumulative odds ratio, a yes/no outcome's log-odds an odds ratio), else the
    difference in the outcome's mean. ``family`` None: no family is known, and the task decides."""
    scale = family.raw_scale.get(task) if family is not None else None
    if scale == "log_hazard" or task == "time_to_event":
        return "hazard ratio", True
    if scale == "latent" or task == "ordinal":
        return "cumulative odds ratio", True
    if task == "binary":
        return "odds ratio", True
    return f"difference in mean {_tick(target)}", False


def modification_stage(ctx: Any) -> Any:
    """The ``modification`` stage: each declared modifier's analysis (MODELING_SEQUENCE §1 row 7)."""
    from turbotab.core.graph import Bundle

    state = ctx.state
    exposure = bound_exposure(state)
    count = family_count(state)
    if state.purpose != "inference" or exposure is None:
        return Bundle(data=ModificationArtifact(
            purpose=state.purpose, exposure=exposure, family_count=count,
            methods="No effect modification is estimated without a declared exposure under "
                    "inference.").model_dump(mode="json"))
    results = []
    found = declared(state)
    for i, (modifier, spec) in enumerate(found.items()):
        ctx.progress(0.05 + 0.9 * i / max(len(found), 1), f"Effect modification by {modifier}")
        results.append(_one(ctx, state, exposure, modifier, spec))
    # The family states every declared test, and names those the rows could not compute.
    count = family_count(state, not_computed=[
        r.modifier for r in results
        if not r.waiting and not any(f.heterogeneity is not None for f in r.families)])
    artifact = ModificationArtifact(purpose="inference", exposure=exposure, family_count=count,
                                    modifications=results,
                                    methods=" ".join(r.sentence for r in results))
    ctx.progress(1.0, "Done")
    return Bundle(data=artifact.model_dump(mode="json"))


def adjusted_sets(state: Any, modifier: str, spec: Any) -> tuple[list[str], list[str], list[str]]:
    """(A's adjusted covariates, B's confounders its answers add, concerns): an interaction adjusts
    for the confounders of each exposure; a covariate adjusted for A that B's answers make a
    consequence of B is named."""
    from turbotab.core import estimand as est

    exposure = bound_exposure(state)
    a_set = [c for c, d in est.derived_roles(state).items() if d.adjusted and c != modifier]
    if _get(spec, "kind") != "interaction":
        return a_set, [], []
    derived = {c: est.derive(a, "total") for c, a in (_get(spec, "answers") or {}).items()}
    extra = [c for c, d in derived.items() if d.adjusted and c not in a_set and c != exposure]
    concerns = [f"{_tick(c)} is adjusted for {_tick(exposure)}, and the answers for "
                f"{_tick(modifier)} make it a {d.role} of {_tick(modifier)}: the joint effect of "
                f"{_tick(modifier)} is then not its total effect."
                for c, d in derived.items() if c in a_set and d.role in ("mediator", "collider")]
    return a_set, extra, concerns


def _one(ctx: Any, state: Any, exposure: str, modifier: str, spec: Any) -> ModificationResult:
    from turbotab.core.models import get_family
    from turbotab.core.models.inference import Outcome, cluster_columns, resolve_clusters
    from turbotab.core.models.pipeline import DesignSpec, design_spec, modeling_frame
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import (_survey, _task, coded_outcome, outcome_levels,
                                               read_assignment)

    kind = _get(spec, "kind")
    status = POST_HOC if _get(spec, "post_hoc") else "declared"
    waiting = missing_answers(state, modifier, spec)
    a_set, extra, concerns = adjusted_sets(state, modifier, spec)
    base = dict(modifier=modifier, kind=kind, exposure=exposure, status=status,
                adjusted_for=a_set, second_adjusted_for=extra, waiting=waiting,
                concerns=concerns)
    if waiting:
        return ModificationResult(
            **base, contrast="", low=0.0, high=0.0, strata=[], reference="",
            sentence=(f"The interaction of {_tick(exposure)} with {_tick(modifier)} waits for the "
                      f"adjustment answers with {_tick(modifier)} as the exposure: "
                      f"{_listing(waiting)}."))
    task = _task(ctx)
    target = state.target
    design = ctx.inputs["design"]
    spec_d = DesignSpec.from_dict(design.objects["spec"])
    families = [get_family(k) for k in (state.models or [])]
    families = [f for f in families if tests_product_terms(f)]
    assignment = read_assignment(ctx.inputs["split"])
    wanted = [c for c in [modifier, *extra] if c not in spec_d.inputs]
    with open_store(ctx) as store:
        unit_columns = cluster_columns(state, store.columns)
        info = {c.name: c for c in store.info().columns}
        follow = []
        if task == "time_to_event":
            from turbotab.core.models.survival import follow_up_columns

            follow = follow_up_columns(state)
        columns = list(dict.fromkeys([*spec_d.inputs, *wanted, target, *unit_columns, *follow]))
        frame = modeling_frame(store, columns, assignment.index.to_numpy(), outcome=target)
    predictors = [*spec_d.predictors, *[c for c in [modifier, *extra]
                                        if c not in spec_d.predictors and c != exposure]]
    spec_m = design_spec(state, frame[[c for c in dict.fromkeys([*spec_d.inputs, *wanted])]],
                         predictors, column_info=info)
    levels = None
    if task == "ordinal":
        from turbotab.core.models.ordinal import ordinal_outcome

        coded, levels = ordinal_outcome(frame[target].to_numpy(), state.outcome_order, column=target)
    else:
        coded = coded_outcome(task, frame[target].to_numpy(), state.event)
    if task == "time_to_event":
        from turbotab.core.models.survival import time_to_event_outcome

        coded = time_to_event_outcome(state, frame, coded)
    y = np.asarray(coded)
    outcome = Outcome(name=target, labels=outcome_levels(task, frame[target].to_numpy(),
                                                         state.event))
    clusters = resolve_clusters(state, frame[list(unit_columns)]) if unit_columns else None
    survey, _ = _survey(ctx, clusters)
    # The FORM repair: a stratum that cannot carry the exposure's effect (one row, no event) is
    # said in plain words before any fit, never as an overflow or a bracketing failure.
    if task == "time_to_event":
        events = np.asarray(y["event"], dtype=float) if getattr(y, "dtype", None) is not None \
            and y.dtype.names else None
    elif task == "binary":
        observed = pd.Series(y).dropna()
        top = sorted(pd.unique(observed), key=str)[-1] if observed.nunique() == 2 else None
        events = (pd.Series(y) == top).astype(float).to_numpy() if top is not None else None
    elif task == "ordinal":
        events = np.asarray(y)
    else:
        events = None
    problems = strata_problems(modifier, exposure, frame[modifier], frame[exposure], events, task,
                               categorical=_categorical(spec_m, frame, modifier),
                               levels=_get(spec, "levels"))
    fits = []
    lay = None
    for family in ([] if problems else families):
        try:
            fit, lay_f = _family(ctx, state, family, spec_m, frame, y, task=task, levels=levels,
                                 outcome=outcome, clusters=clusters, survey=survey,
                                 exposure=exposure, modifier=modifier, spec=spec, target=target)
            lay = lay or lay_f
            fits.append(fit)
        except Exception as exc:  # noqa: BLE001 - said in the artifact, never silent
            if ctx.cancelled():
                from turbotab.core.jobs import Cancelled

                raise Cancelled() from exc
            measure, ratio = _measure(task, family, target)
            fits.append(ModificationFit(
                family=family.key, label=family.label, measure=measure,
                scale="ratio" if ratio else "difference", n_rows=len(frame),
                concerns=[_failure_words(family, exc)]))
    if problems:
        concerns = [*concerns, f"Not estimated: {'; '.join(problems)}. Choose another modifier, "
                               f"or one whose every stratum holds rows of both kinds."]
    if not families:
        concerns = [*concerns, "Effect modification is refit for least squares, logistic, "
                               "proportional-odds and Cox models; none of them is chosen."]
    result = ModificationResult(
        **{**base, "concerns": concerns}, contrast=lay.contrast if lay else "",
        low=lay.low if lay else 0.0, high=lay.high if lay else 0.0,
        strata=list(lay.m_values) if lay else [], reference=lay.reference if lay else "",
        families=fits, sentence="")
    result.sentence = sentence(state, result, task=task,
                               not_estimated=problems or [c for f in fits for c in f.concerns
                                                          if not f.effects])
    return result


def _categorical(spec_m: Any, frame: pd.DataFrame, modifier: str) -> bool:
    """Whether the modifier enters as indicators: a category the design reads as codes, or text."""
    return modifier in set(getattr(spec_m, "categorical", None) or []) or \
        not pd.api.types.is_numeric_dtype(frame[modifier])


def _failure_words(family: Any, exc: BaseException) -> str:
    """A family that could not be refit with the product terms, in plain words: the reason when
    the code says it (:class:`NotEstimable`), else what failed, never the raw exception text."""
    if isinstance(exc, NotEstimable):
        return f"{family.label} was not refit with the product terms: {exc}"
    return (f"{family.label} could not be refit with the product terms on these rows: its "
            f"estimation did not reach finite estimates (a stratum with too few rows or events "
            f"for the terms it adds, or nearly collinear terms).")


def _family(ctx: Any, state: Any, family: Any, spec_m: Any, frame: pd.DataFrame, y: np.ndarray, *,
            task: str, levels: Any, outcome: Any, clusters: Any, survey: Any, exposure: str,
            modifier: str, spec: Any, target: str) -> tuple[ModificationFit, _Layout]:
    from sklearn.base import clone

    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.linear import model_matrix
    from turbotab.core.models.pipeline import build_pipeline
    from turbotab.core.stages.effects import matrix_table
    from turbotab.core.stages.modeling import _design_df, with_units

    concerns: list[str] = []
    X = frame[list(spec_m.inputs)]
    units = (pd.Series(clusters.codes, index=frame.index)
             if clusters is not None and clusters.clustered else None)
    pipeline = build_pipeline(spec_m, family, task, "inference", len(frame), len(spec_m.inputs) + 1)
    design = survey.design if survey is not None and getattr(survey, "answer", None) == \
        "population" else None
    if design is not None and not weighted_products(family):
        concerns.append("Fit without the survey weights: these estimates describe these "
                        "participants, not the surveyed population.")
        design = None
    first = fit_pipeline(with_units(clone(pipeline), units), X, y)
    if levels is not None:
        first[-1].level_names_ = list(levels)
    lay = layout(first, X, spec_m.inputs, exposure, modifier, spec_m.energy_adjustment(), spec,
                 frame)
    copies, template, pooled_words = [X], pipeline, None
    strategy = state.missing.strategy if state.missing is not None else None
    gaps = [c for c in spec_m.inputs if X[c].isna().any()]
    if strategy == "multiple_imputation" and gaps:
        imputations = impute_with_products(ctx, spec_m, X, y, task, lay.products, survey=design,
                                           clusters=clusters)
        from turbotab.core.methods.missing import copy_template

        copies = list(imputations.frames)
        template = copy_template(pipeline, imputations.plan)
        pooled_words = (f"Multiple imputation compatible with the product terms (SMC-FCS with "
                        f"them in the substantive model), m = {imputations.m}: each quantity "
                        f"pooled by Rubin's rules, the heterogeneity test by D1")
    measure, ratio = _measure(task, family, target)
    per_copy, Q, U, names0, info0, rows0, df = [], [], [], None, None, None, None
    df_com: float | None = None
    df_scalar: float | None = None
    for X_k in copies:
        fitted = fit_pipeline(with_units(clone(template), units), X_k[list(spec_m.inputs)], y)
        if levels is not None:
            fitted[-1].level_names_ = list(levels)
        M = with_products(model_matrix(fitted, X_k[list(spec_m.inputs)]), lay.products)
        classes = list(getattr(fitted[-1], "classes_", [])) or None
        table = matrix_table(family, M, y, task=task, classes=classes, clusters=clusters,
                             outcome=outcome, survey=design, levels=levels, event=state.event,
                             features=[])
        if table is not None and not table.info.get("refused") and table.cov is None and \
                table.info.get("covariance") == "profile":
            # A logistic model the data separate (a Firth-penalized table): every quantity with
            # its profile penalized-likelihood interval, the heterogeneity test by the penalized
            # likelihood-ratio test (the FORM repair: under separation the declared tests run).
            if len(copies) > 1:
                raise NotEstimable(
                    "the data separate the outcome, and a Firth-penalized fit gives profile "
                    "intervals, not the covariance Rubin's rules and D1 pool across imputed "
                    "copies; complete cases, or a modifier the data do not separate, are "
                    "estimable")
            return _firth_family(family, M, y, classes, lay, measure, ratio, task, len(frame),
                                 concerns, table), lay
        if table is None or table.info.get("refused") or table.cov is None:
            raise NotEstimable((table.info.get("refused") if table is not None else None)
                               or "the model gave no covariance to test the product terms on")
        names = [str(r["feature"]) for r in table.rows]
        beta = np.array([np.nan if r["estimate"] is None else r["estimate"] for r in table.rows],
                        dtype=float)
        cov = np.asarray(table.cov, dtype=float)
        df = combination_df(table.info or {}, table.rows)
        per_copy.append(measures(names, beta, cov, a_values=lay.a_values, m_values=lay.m_values,
                                 reference=lay.reference, products=lay.products, ratio=ratio,
                                 df=df))
        where = {n: i for i, n in enumerate(names)}
        idx = [where[p[2]] for p in lay.products if p[2] in where]
        Q.append(beta[idx])
        U.append(cov[np.ix_(idx, idx)])
        if names0 is None:
            names0, info0, rows0 = names, dict(table.info), list(table.rows)
            # D1 takes the design's df as ν_com (MS2); each scalar the table's own complete-data
            # df, as Rubin's rules pool a coefficient row
            df_com, df_scalar = _design_df(table), df
    pooled = pool_measures(per_copy, df_scalar)
    het = _heterogeneity(Q, U, rows0, info0, idx_names=[p[2] for p in lay.products],
                         df_com=df_com, m=len(copies))

    def q(key: str, stratum: str, a: str | None = None, exp: bool = False) -> Quantity:
        v = pooled[key]
        f = (lambda t: None if t is None or not math.isfinite(t) else
             (math.exp(t) if exp else float(t)))
        return Quantity(stratum=stratum, a=a, estimate=f(v["estimate"]), ci_low=f(v["ci_low"]),
                        ci_high=f(v["ci_high"]),
                        p=None if v.get("p") is None or not math.isfinite(v["p"]) else v["p"])

    strata = list(lay.m_values)
    others = [s for s in strata if s != lay.reference]
    effects = [q(f"effect|{s}", s, exp=ratio) for s in strata]
    joint = ([q(f"joint|a1|{lay.reference}", lay.reference, "a1", exp=ratio)]
             + [q(f"joint|{a}|{s}", s, a, exp=ratio) for s in others for a in ("a0", "a1")])
    mult = [q(f"mult|{s}", s, exp=True) for s in others] if ratio else []
    reri_rows = [q(f"reri|{s}", s) for s in others] if ratio else []
    additive = [] if ratio else [q(f"additive|{s}", s) for s in others]
    if ratio and task == "binary":
        concerns.append(f"The RERI is computed from odds ratios: {RARE}.")
    fit = ModificationFit(family=family.key, label=family.label, measure=measure,
                          scale="ratio" if ratio else "difference", n_rows=len(frame),
                          effects=effects, joint=joint, multiplicative=mult, reri=reri_rows,
                          additive=additive, heterogeneity=het, pooled=pooled_words,
                          concerns=concerns)
    return fit, lay


def firth_contrast(X: np.ndarray, y: np.ndarray, c: np.ndarray, full: Any
                   ) -> dict[str, float | None]:
    """c'β of a Firth-penalized logistic fit with its profile penalized-likelihood interval and
    likelihood-ratio p-value: the design reparametrized so that c'β is one coefficient
    (θ = Tβ, T the identity with c in the row of c's largest entry; the penalty
    ½ log |I| moves by the constant log |det T|, so the profile is unchanged), then profiled as
    each coefficient of the table is (``models.inference.firth_profile``)."""
    from turbotab.core.models.inference import firth_fit, firth_profile

    c = np.asarray(c, dtype=float)
    j = int(np.argmax(np.abs(c)))
    T = np.eye(len(c))
    T[j, :] = c
    Xr = np.asarray(X, dtype=float) @ np.linalg.inv(T)
    fit = firth_fit(Xr, y, start=T @ np.asarray(full.beta, dtype=float))
    low, high, p = firth_profile(Xr, y, fit, j)
    return {"estimate": float(fit.beta[j]), "variance": None, "se": None, "ci_low": low,
            "ci_high": high, "p": p}


def _firth_family(family: Any, M: pd.DataFrame, y: Any, classes: Any, lay: _Layout, measure: str,
                  ratio: bool, task: str, n_rows: int, concerns: list[str], table: Any
                  ) -> ModificationFit:
    """Every quantity of a separated logistic model (Firth's penalized likelihood): each effect,
    combination and ratio of odds ratios with its profile interval; the heterogeneity test by the
    penalized likelihood-ratio test; the RERI's point estimate, its delta-method interval needing
    a Wald covariance a penalized fit does not give."""
    from turbotab.core.methods.exposure_form import FIRTH_PLR, firth_design, firth_lr_test
    from turbotab.core.models.inference import firth_fit, format_p

    found = firth_design(M, y, classes)
    if found is None:
        raise NotEstimable("the separated outcome's model could not be laid out again")
    names, X, event = found
    fit = firth_fit(X, event)
    x = contrast_vectors(names, lay.a_values, lay.m_values, lay.products)
    ref0 = x[f"a0|{lay.reference}"]
    strata = list(lay.m_values)
    others = [s for s in strata if s != lay.reference]
    f = (lambda t: None if t is None or not math.isfinite(t) else
         (math.exp(t) if ratio else float(t)))

    def q(c: np.ndarray, stratum: str, a: str | None = None) -> Quantity:
        got = firth_contrast(X, event, c, fit)
        return Quantity(stratum=stratum, a=a, estimate=f(got["estimate"]), ci_low=f(got["ci_low"]),
                        ci_high=f(got["ci_high"]), p=got["p"])

    effects = [q(x[f"a1|{s}"] - x[f"a0|{s}"], s) for s in strata]
    joint = ([q(x[f"a1|{lay.reference}"] - ref0, lay.reference, "a1")]
             + [q(x[f"{a}|{s}"] - ref0, s, a) for s in others for a in ("a0", "a1")])
    c10 = x[f"a1|{lay.reference}"] - ref0
    mult, reri_rows = [], []
    beta = np.asarray(fit.beta, dtype=float)
    for s in others:
        c01, c11 = x[f"a0|{s}"] - ref0, x[f"a1|{s}"] - ref0
        mult.append(q(c11 - c10 - c01, s))
        point = math.exp(float(c11 @ beta)) - math.exp(float(c10 @ beta)) \
            - math.exp(float(c01 @ beta)) + 1.0
        reri_rows.append(Quantity(stratum=s, estimate=point if math.isfinite(point) else None,
                                  ci_low=None, ci_high=None, p=None))
    where = {n: i for i, n in enumerate(names)}
    idx = [where[p[2]] for p in lay.products if p[2] in where]
    het = firth_lr_test(X, event, idx, fit)
    k = len(idx)
    heterogeneity = None if het is None else Heterogeneity(
        statistic=het["statistic"], df_num=het["df_num"], df_den=None, distribution="chi2",
        p=het["p"], caption=(f"Penalized likelihood-ratio test that the {k} product term"
                             f"{'s are' if k > 1 else ' is'} zero ({FIRTH_PLR}): "
                             f"χ²({het['df_num']}) = {het['statistic']:.2f}, p = "
                             f"{format_p(het['p'])}."))
    words = list(concerns)
    words += [c for c in table.concerns if c not in words]
    words.append("The data separate the outcome, so the model is Firth's penalized logistic "
                 "regression: each odds ratio and ratio of odds ratios carries a profile "
                 "penalized-likelihood interval and the heterogeneity test is a penalized "
                 "likelihood-ratio test; the RERI is reported without an interval, its delta-method "
                 "interval needing a Wald covariance the penalized fit does not give.")
    if not fit.converged:
        words.append("Firth's penalized fit stopped before converging; treat these numbers with "
                      "care.")
    if task == "binary":
        words.append(f"The RERI is computed from odds ratios: {RARE}.")
    return ModificationFit(family=family.key, label=family.label, measure=measure,
                           scale="ratio" if ratio else "difference", n_rows=n_rows,
                           effects=effects, joint=joint, multiplicative=mult, reri=reri_rows,
                           heterogeneity=heterogeneity, pooled=None, concerns=words,
                           penalized=True)


def combination_df(info: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> float | None:
    """The reference distribution's df for a combination of coefficients, as the table's joint
    tests take it (``exposure_form._reference``): n − p under HC3, the design's df under a survey,
    G − 1 under a cluster-robust covariance (each coefficient's own Satterthwaite df does not
    carry to a combination), none (normal) for a model-based table."""
    covariance = info.get("covariance")
    if covariance in ("CR2", "CR1", "CR0"):
        g = info.get("n_clusters")
        return float(g - 1) if g and g > 1 else None
    if covariance == "design":
        d = (info.get("survey") or {}).get("df")
        return float(d) if d else None
    if covariance == "HC3":
        dfs = [r.get("df") for r in rows if r.get("df") is not None]
        return float(dfs[0]) if dfs else None
    return None


def _heterogeneity(Q: Sequence[np.ndarray], U: Sequence[np.ndarray], rows: Sequence[Any],
                   info: Mapping[str, Any], *, idx_names: Sequence[str], df_com: float | None,
                   m: int) -> Heterogeneity | None:
    """The Wald test that every product term is zero: on the table's covariance (F for HC3,
    cluster-robust and design-based tables, χ² for model-based ones), D1 over imputed copies."""
    from turbotab.core.methods.exposure_form import _statistic_words, wald_test
    from turbotab.core.methods.imputation import pooled_wald
    from turbotab.core.models.inference import format_p

    k = len(idx_names)
    if m == 1:
        names = [str(r["feature"]) for r in rows]
        where = {n: i for i, n in enumerate(names)}
        idx = [where[n] for n in idx_names if n in where]
        beta = np.array([np.nan if r["estimate"] is None else r["estimate"] for r in rows], float)
        cov = np.zeros((len(rows), len(rows)))
        cov[np.ix_(idx, idx)] = np.asarray(U[0])
        found = wald_test(beta, cov, idx, info, rows)
        if found is None:
            return None
        return Heterogeneity(statistic=found["statistic"], df_num=found["df_num"],
                             df_den=found["df_den"], distribution=found["distribution"],
                             p=found["p"],
                             caption=(f"Wald test that the {k} product term"
                                      f"{'s are' if k > 1 else ' is'} zero ({found['basis']}): "
                                      f"{_statistic_words(found)}."))
    result = pooled_wald(np.asarray(Q, dtype=float), np.asarray(U, dtype=float), df_com)
    if result is None:
        return None
    f_ref = result["df_den"] is not None
    stat = result["statistic"] if f_ref else result["statistic"] * result["df_num"]
    return Heterogeneity(statistic=stat, df_num=result["df_num"], df_den=result["df_den"],
                         distribution="F" if f_ref else "chi2", p=result["p"],
                         caption=(f"Pooled over {m} imputations by Li, Raghunathan & Rubin's D1: "
                                  f"the {k} product term{'s' if k > 1 else ''}, p = "
                                  f"{format_p(result['p'])}."))


def impute_with_products(ctx: Any, spec: Any, X: pd.DataFrame, y: Any, task: str,
                         products: Sequence[tuple[str, str, str]], *, survey: Any = None,
                         clusters: Any = None) -> Any:
    """The completed copies under an imputation model compatible with the product terms: SMC-FCS
    whose substantive model is the pipeline's own design with the products appended
    (MODELING_SEQUENCE §1.1; Bartlett et al. 2015)."""
    from sklearn.base import clone

    from turbotab.core.methods.imputation import OUTCOME_PREFIX
    from turbotab.core.methods.missing import (SUBSTANTIVE, FixedForms, binary01,
                                               imputation_plan)
    from turbotab.core.methods.smcfcs import Outcome, Substantive, impute
    from turbotab.core.models.pipeline import shared_steps, transformer

    if task not in SUBSTANTIVE:
        raise ValueError(f"Product terms under multiple imputation need SMC-FCS, which is built "
                         f"here for linear, logistic and Cox outcomes; this outcome is {task}.")
    clustered = clusters is not None and getattr(clusters, "clustered", False)
    seed = int(getattr(ctx.state.split, "seed", 0) or 0) if ctx.state.split is not None else 0
    plan = imputation_plan(spec, X, y, task, survey=survey,
                           clusters=clusters if clustered else None)
    steps = []
    for name, step in shared_steps(spec):
        if name in ("impute", "detect"):
            continue
        if name == "form":
            step = FixedForms(dict(step.forms or {}), dict(plan.forms))
        steps.append((name, step))
    template = transformer(steps)
    inputs = list(spec.inputs)
    holder: dict[str, Any] = {}

    def refresh(raw: pd.DataFrame) -> None:
        holder["T"] = clone(template).fit(raw[inputs])

    def design(raw: pd.DataFrame) -> np.ndarray:
        if "T" not in holder:
            refresh(raw)
        out = holder["T"].transform(raw[inputs])
        return with_products(out, products).to_numpy(dtype=float)

    kind = SUBSTANTIVE[task]
    if kind == "cox":
        values = np.asarray(y)
        outcome = Outcome("cox", time=np.asarray(values["time"], dtype=float),
                          event=np.asarray(values["event"], dtype=float),
                          entry=np.asarray(values["entry"], dtype=float))
    elif kind == "logistic":
        outcome = Outcome("logistic", y=binary01(y))
    else:
        outcome = Outcome("linear", y=np.asarray(y, dtype=float))
    data = plan.data.drop(columns=[c for c in plan.data.columns if c.startswith(OUTCOME_PREFIX)])
    out = impute(data, plan.variables, mode="smcfcs",
                 substantive=Substantive(outcome=outcome, design=design, refresh=refresh),
                 identity=plan.identity, units=plan.units, m=plan.m, seed=seed,
                 cancelled=ctx.cancelled, kinds=plan.kinds)
    frames = []
    for f in out.frames:
        done = X.copy()
        for c in X.columns:
            if c in f.columns and c not in plan.levels:
                done[c] = f[c].to_numpy() if not isinstance(f[c].dtype, pd.CategoricalDtype) \
                    else f[c]
        frames.append(done)
    out.frames = frames
    out.plan = {"mode": "smcfcs", "forms": plan.forms, "levels": list(plan.levels),
                "products": [p[2] for p in products]}
    return out


# ── the record ───────────────────────────────────────────────────────────────


def sentence(state: Any, r: ModificationResult, *, task: str | None = None,
             not_estimated: Sequence[str] = ()) -> str:
    """The methods sentence one declared modifier writes (verbatim in the record and the
    artifact). One that no family could estimate says so and why, in place of a contrast, a
    reference and scales it does not have (the FORM repair: never "`region` at " nor a difference
    of differences for an odds ratio)."""
    target = _tick(getattr(state, "target", None))
    a, m = _tick(r.exposure), _tick(r.modifier)
    when = ("declared before the estimates were seen" if r.status == "declared"
            else f"{POST_HOC} (declared after the estimates were seen)")
    estimated = next((f for f in r.families if f.effects), None)
    if estimated is None:
        task = task or getattr(state, "task", None)
        measure, _ = _measure(str(task), _registered(r.families[0].family if r.families else ""),
                              str(getattr(state, "target", None)))
        what = (f"The interaction of {a} and {m} on {target}" if r.kind == "interaction" else
                f"Effect modification of {a}'s effect on {target} by {m}")
        why = "; ".join(dict.fromkeys(str(w).rstrip(".") for w in not_estimated if w)) or \
            "no chosen family refits it"
        return (f"{what} was {when}; the {measure} within each level of {m} was not estimated: "
                f"{_after_colon(why)}.")
    fit = estimated
    measure = fit.measure
    ratio = fit.scale == "ratio"
    if r.kind == "interaction":
        head = (f"The interaction of {a} and {m} on {target} was {when}: the joint effects of "
                f"{r.contrast} and of {m} against a single reference ({a} at {_number(r.low)}, "
                f"{m} at {r.reference}), and {a}'s effect within each level of {m}")
    else:
        head = (f"Effect modification of {a}'s effect on {target} by {m} was {when}: the "
                f"{measure} for {r.contrast} within each level of {m}, and each combination "
                f"against a single reference ({a} at {_number(r.low)}, {m} at {r.reference})")
    if ratio and fit.penalized:
        scales = (f"; the data separate the outcome, so the model is Firth's penalized logistic "
                  f"regression (Heinze & Schemper 2002), each {measure} with its profile "
                  f"penalized-likelihood interval; interaction on the additive scale as the "
                  f"relative excess risk due to interaction (RERI), its point estimate only (its "
                  f"delta-method interval needs a Wald covariance the penalized fit does not "
                  f"give), and on the multiplicative scale as the ratio of {measure}s with its "
                  f"profile interval ({KNOL})")
    elif ratio:
        scales = (f"; interaction on the additive scale as the relative excess risk due to "
                  f"interaction (RERI) with a delta-method interval ({HOSMER}) and on the "
                  f"multiplicative scale as the ratio of {measure}s ({KNOL})")
    else:
        scales = (f"; interaction on the additive scale as the difference of differences ({KNOL}); "
                  f"no multiplicative measure applies to a difference")
    adjusted = (f"adjusted for {_listing(r.adjusted_for)}" if r.adjusted_for
                else "with no covariate adjusted for")
    if r.kind == "interaction":
        adjusted = (f"adjusted for the confounders of {a} ({_listing(r.adjusted_for) or 'none'}) "
                    f"and of {m} as its own answers derive them "
                    f"({_listing(r.second_adjusted_for) or 'none beyond them'})")
    n_products = (fit.heterogeneity.df_num
                  if fit is not None and fit.heterogeneity is not None else None)
    test = "penalized likelihood-ratio test" if fit.penalized else "Wald test"
    het = (f"; the heterogeneity test is the {test} that the {n_products} product term"
           f"{'s are' if (n_products or 0) > 1 else ' is'} zero" if n_products else "")
    pooled = f"; {fit.pooled[0].lower()}{fit.pooled[1:]}" if fit is not None and fit.pooled else ""
    return f"{head}{scales}, {adjusted}{het}{pooled}."


# Words that keep their capital inside a sentence (a family named for a person).
PROPER = ("Cox",)


def _after_colon(text: str) -> str:
    """``text`` continuing a sentence after a colon: its first word lowercased, unless it is a
    column (backticked) or a proper name (``Cox proportional hazards``)."""
    first = text.split(" ", 1)[0]
    if not text[:1].isupper() or first in PROPER:
        return text
    return text[0].lower() + text[1:]


def _record_sentence(d: Any, state: Any, ctx: Any) -> str:
    """The record's sentence for ``set_modification``."""
    a = _tick(d.exposure or bound_exposure(state))
    m = _tick(d.modifier)
    if d.withdraw:
        if d.post_hoc:
            return (f"The declared modifier {m} was withdrawn; it was declared when the estimates "
                    f"were seen, so its heterogeneity test still counts in the family of tests "
                    f"stated")
        return f"The declared modifier {m} was withdrawn"
    when = POST_HOC if d.post_hoc else "declared"
    if d.modification == "interaction":
        return (f"The interaction of {a} with the second exposure {m} was {when}; the adjustment "
                f"set was asked again with {m} as the exposure, and the confounders of each are "
                f"adjusted for ({KNOL})")
    return (f"Effect modification of {a}'s effect by {m} was {when}, reported within each level "
            f"of {m} against a single reference on the additive and multiplicative scales "
            f"({KNOL})")


# ── the leash: refusals and completion ───────────────────────────────────────


def _state(ctx: Any) -> Any:
    from turbotab.core.decisions import _state as state_of

    return state_of(ctx)


def _modifier_is_declarable(decision: Any, ctx: Any) -> None:
    """Under inference, for the one declared exposure; the modifier a column of the table, neither
    the outcome nor the exposure; an interaction's answers about covariates of the model."""
    from turbotab.core.decisions import ROW_ID, Refusal, _columns_of

    if decision.withdraw:
        return
    state = _state(ctx)
    if state is None:
        return
    if getattr(state, "purpose", None) == "prediction":
        raise Refusal("modification_for_inference",
                      "Under prediction no effect is estimated, so no effect modifier is declared; "
                      "the families learn what the predictors do together.",
                      exits=[{"label": "Change the purpose to inference", "decision": None}])
    exposure = bound_exposure(state)
    if exposure is None:
        raise Refusal("estimand_first",
                      "Effect modification is of one declared study factor's effect: declare "
                      "what you study and its effect first.",
                      exits=[{"label": "Answer the question on what you study and its effect", "decision": None}])
    columns = _columns_of(ctx)
    if columns is not None and (decision.modifier not in columns or decision.modifier == ROW_ID):
        raise Refusal("unknown_column", f"This dataset has no column named `{decision.modifier}`.",
                      exits=[{"label": "Choose one of the dataset's columns", "decision": None}])
    if decision.modifier in (exposure, getattr(state, "target", None)):
        raise Refusal("modifier_is_exposure_or_outcome",
                      f"`{decision.modifier}` is the {'exposure' if decision.modifier == exposure else 'outcome'};"
                      f" a modifier is another column.",
                      exits=[{"label": "Choose another column", "decision": None}])
    if decision.exposure not in (None, exposure):
        raise Refusal("not_the_exposure",
                      f"The declared study factor is `{exposure}`, not `{decision.exposure}`.",
                      exits=[{"label": f"Declare it for `{exposure}`",
                              "decision": decision.model_copy(update={"exposure": exposure})}])
    if decision.modification == "interaction" and decision.answers:
        allowed = set(second_covariates(state, decision.modifier))
        stray = [c for c in decision.answers if c not in allowed]
        if stray:
            raise Refusal("not_a_covariate",
                          f"{_listing(stray)} {'is' if len(stray) == 1 else 'are'} not a "
                          f"covariate of the model with `{decision.modifier}` as what you study.",
                          exits=[{"label": "Answer for the model's covariates", "decision":
                                  decision.model_copy(update={"answers": {
                                      c: a for c, a in decision.answers.items()
                                      if c in allowed}})}])


def _strata_hold_the_effect(decision: Any, ctx: Any) -> None:
    """A modifier whose strata cannot each carry the exposure's effect on the analyzed rows (one
    row, one exposure value, no event) is refused with the reason and its way forward, rather than
    recorded and then fit to an overflow (:func:`strata_problems`). Checked where the context
    reads the rows (the server's); the stage says the same if the rows change later."""
    from turbotab.core.decisions import Refusal, _ctx, _store_of
    from turbotab.core.readings import confirmed_codes
    from turbotab.core.stages.rows import _level_key

    if decision.withdraw:
        return
    state = _state(ctx)
    exposure = bound_exposure(state) if state is not None else None
    target = getattr(state, "target", None) if state is not None else None
    store = _store_of(ctx)
    analyzed = _ctx(ctx, "analyzed")
    if exposure is None or target is None or store is None or not callable(analyzed):
        return
    try:
        rows = analyzed()
        if rows is None or any(c not in store.columns for c in (decision.modifier, exposure,
                                                                 target)):
            return
        frame = store.materialize([decision.modifier, exposure, target], rows)
    except Exception:  # noqa: BLE001 - rows that cannot be read are checked by the stage
        return
    task = getattr(state, "task", None) or _ctx(ctx, "task")
    y = frame[target]
    frame = frame[y.notna()]
    y = y[y.notna()]
    events = None
    # (a time to event's indicator is not the target column: the stage checks its events)
    if task == "binary" and getattr(state, "event", None) is not None:
        key = _level_key(state.event)
        events = y.map(_level_key).eq(key).astype(float).to_numpy()
    elif task == "binary" and y.nunique() == 2:
        events = y.eq(sorted(pd.unique(y), key=str)[-1]).astype(float).to_numpy()
    elif task == "ordinal":
        events = y.to_numpy()
    categorical = (not pd.api.types.is_numeric_dtype(frame[decision.modifier])
                   or decision.modifier in set(confirmed_codes(state)))
    problems = strata_problems(decision.modifier, exposure, frame[decision.modifier],
                               frame[exposure], events, task, categorical=categorical,
                               levels=decision.levels)
    if problems:
        raise Refusal(
            "modifier_stratum_not_estimable",
            f"{'; '.join(problems)[0].upper()}{'; '.join(problems)[1:]}. A modifier needs every "
            f"stratum to carry the effect of what you study.",
            exits=[{"label": "Choose another modifier (one whose every stratum holds the "
                             "study factor's contrast and the outcome's both kinds)", "decision": None}])


def _modifier_records_when(decision: Any, ctx: Any) -> Any:
    """The exposure it is about, and whether it was declared after the estimates were seen: then
    it is suggested by data inspection, whatever the client said. A withdrawal records the same:
    one after the estimates were seen leaves the modifier counted in the family of tests (its test
    was seen), one before them takes it out, whatever the client said."""
    state = _state(ctx)
    if decision.withdraw:
        if state is None:
            return decision.model_copy(update={"post_hoc": False})
        held = (getattr(state, "modifications", None) or {}).get(decision.modifier)
        update = {"post_hoc": bool(getattr(state, "plan_locked", None)) and held is not None,
                  "modification": _get(held, "kind") or decision.modification,
                  "exposure": decision.exposure or _get(held, "exposure") or bound_exposure(state)}
        return decision.model_copy(update=update)
    if state is None:
        return decision
    update: dict[str, Any] = {}
    if decision.exposure is None and bound_exposure(state) is not None:
        update["exposure"] = bound_exposure(state)
    if getattr(state, "plan_locked", None) and not decision.post_hoc:
        update["post_hoc"] = True
    return decision.model_copy(update=update) if update else decision


# ── the method contracts (BLUEPRINT §13) ─────────────────────────────────────

PACKAGE = "FORM"
_SENTENCE = "turbotab.core.methods.interaction:sentence"


def _register_contracts() -> None:
    from turbotab.core.contracts import ContractOption, MethodContract, Relation, register_contract

    me = "turbotab.core.methods.interaction"

    def option(key: str, label: str, customary: str, sound: str, rung: str) -> ContractOption:
        return ContractOption(key, label, customary,
                              sound={"inference": sound,
                                     "prediction": "Not offered: a prediction estimates no effect."},
                              rung={"inference": rung, "prediction": "not_offered"})

    def relation(id: str, kind: str, target: str, condition: str, says: str, *,
                 rung: str | None = None, exits: Sequence[str] = (), by: str = "") -> Relation:
        return Relation(kind, target, says, purposes=("inference",), rung=rung,
                        exits=tuple(exits), enforced_by=by, condition=condition, id=id)

    shared = (
        relation("single-reference", "implies", "the joint effects",
                 "a declared modifier",
                 "every combination is reported against one reference category (a0 in the "
                 "modifier's reference stratum)", by=f"{me}:measures"),
        relation("both-scales", "implies", "RERI and the multiplicative measure",
                 "a ratio scale", "the RERI with a delta-method interval (Hosmer & Lemeshow 1992) "
                 "and the ratio of ratios, both with intervals", by=f"{me}:reri"),
        relation("post-hoc-labeled", "implies", "the family of tests",
                 "a modifier declared after the estimates were seen",
                 "it is labeled suggested by data inspection and counted in the family",
                 by=f"{me}:family_count"),
        relation("modification-mi-compatible", "implies", "SMC-FCS with the product terms",
                 "multiple imputation", "the imputation model holds the products; every scalar is "
                 "pooled by Rubin's rules and the heterogeneity test by D1",
                 by=f"{me}:impute_with_products"),
        relation("modification-in-plan", "implies", "the analysis-plan lock", "a declared "
                 "modifier", "it is part of the plan; one declared after the estimates were seen "
                 "is marked so", by="turbotab.core.plan_lock:plan_of"),
        relation("stratum-estimable", "conflicts", "a stratum that cannot carry the effect",
                 "a modifier stratum with one row, one study factor value, or no event (or only "
                 "events)", "refused with the reason, never fit to an overflow; if the rows "
                 "change later, the stage says it in plain words and the family names it",
                 rung="refused", exits=("another modifier",), by=f"{me}:_strata_hold_the_effect"),
        relation("withdrawn-still-counted", "implies", "the family of tests",
                 "a modifier withdrawn after the estimates were seen",
                 "it is no longer estimated and still counted in the family the record states",
                 by=f"{me}:family_count"),
        relation("separation-profile", "implies", "profile intervals and a penalized "
                 "likelihood-ratio heterogeneity test", "a logistic model the data separate",
                 "each effect, combination and ratio of odds ratios carries its profile "
                 "penalized-likelihood interval, the heterogeneity test is the penalized "
                 "likelihood-ratio test, the RERI is given without an interval",
                 by=f"{me}:firth_contrast"),
    )
    register_contract(MethodContract(
        key="effect_modification", package=PACKAGE,
        label="Effect modification (the exposure's effect across strata of a modifier)",
        slot="model", scope="model",
        scope_note="the product terms are terms of the outcome model, fit on every analyzed row",
        needs=("a declared exposure", "a modifier column"), question="modification",
        place="MODELING_SEQUENCE §1 row 7", decision="set_modification", stage="modification",
        leash={"inference": "available", "prediction": "not_offered"},
        storyboard=("the exposure's effect in each stratum", "every combination against one "
                    "reference", "RERI and the ratio of ratios", "the heterogeneity test"),
        sentence=_SENTENCE,
        options=(option("declared", "Declared before the estimates",
                        "customary: stratified estimates and a product-term p (multiplicative "
                        "only)", "Sound: both scales with intervals (Knol & VanderWeele 2012)",
                        "recommended"),
                 option("post_hoc", "Suggested by data inspection",
                        "common, rarely labeled", "Allowed, labeled and counted in the family",
                        "available")),
        relations=(*shared,
                   relation("modification-own-set", "implies", "the exposure's adjustment set",
                            "effect modification", "the model is the exposure's own, the modifier "
                            "beside it", by=f"{me}:adjusted_sets")),
        sources=(KNOL, HOSMER, VANDERWEELE_2009)))
    register_contract(MethodContract(
        key="interaction", package=PACKAGE,
        label="Interaction (the joint effect of two exposures)",
        slot="model", scope="model",
        scope_note="the product terms are terms of the outcome model, fit on every analyzed row",
        needs=("a declared exposure", "a second exposure", "its adjustment answers"),
        question="modification", place="MODELING_SEQUENCE §1 row 7",
        decision="set_modification", stage="modification",
        leash={"inference": "available", "prediction": "not_offered"},
        storyboard=("the adjustment set asked again for the second exposure", "the joint effects "
                    "against one reference", "RERI and the ratio of ratios"),
        sentence=_SENTENCE,
        options=(option("declared", "Declared before the estimates",
                        "customary: a product-term p (multiplicative only)",
                        "Sound: what could explain the link is adjusted for each study factor, both scales reported",
                        "recommended"),),
        relations=(*[r for r in shared],
                   relation("interaction-reasks-adjustment", "invalidates",
                            "the adjustment set for the second exposure", "an interaction",
                            "the disjunctive cause criterion is asked again with the second "
                            "exposure as the exposure; its confounders join the model",
                            by=f"{me}:missing_answers")),
        sources=(KNOL, HOSMER, VANDERWEELE_2009)))


def _register() -> None:
    from turbotab.core.decisions import register_completion, register_validator
    from turbotab.core.voice import register_sentence

    register_validator("set_modification", _modifier_is_declarable)
    register_validator("set_modification", _strata_hold_the_effect)
    register_completion("set_modification", _modifier_records_when)
    register_sentence("set_modification")(_record_sentence)
    _register_contracts()


_register()

__all__ = [
    "HOSMER", "KNOL", "MODIFICATION_READS", "ModificationArtifact", "ModificationFit",
    "ModificationResult", "NotEstimable", "POST_HOC", "adjusted_sets", "bound_exposure",
    "complete_modifications", "contrast_vectors", "declared", "family_count", "firth_contrast",
    "impute_with_products", "layout", "linear_combination", "measures", "missing_answers",
    "modification_answer", "modification_followup", "modification_gate", "modification_stage",
    "pool_measures", "reri", "second_covariates", "sentence", "strata_problems",
    "tests_product_terms", "weighted_products", "with_products", "withdrawn_after_estimates",
]
