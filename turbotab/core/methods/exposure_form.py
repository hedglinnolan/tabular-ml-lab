"""The form an exposure takes in the model: a straight line, a restricted cubic spline, or
quintiles (AUDIT_REPORT §5 WP12, ME-17: "Exposure–response is straight-line only").

NUTRITION_PACK §07G: "**Restricted cubic spline** of outcome vs energy-adjusted intake: 3–5 knots
at conventional percentiles (**3 knots at 10/50/90**, or 4 at 5/35/65/95) … and a reported **p for
non-linearity** … **[CONVENTION, now near-default; quintiles remain expected alongside]**", and
§08: "**p for trend using the median of each quintile as a continuous score, not the quintile
number**". The form is a pipeline step (:class:`ExposureForms`) that runs after energy
adjustment, so a spline or a quintile is of the energy-adjusted intake, and it is fit inside the
pipeline: knots, cut points and quintile medians are learned on the rows each fit sees (every
training fold under cross-validation), never on a held-out row.

**The spline is Harrell's** (``rms::rcs``, which calls ``Hmisc::rcspline.eval`` with its default
``norm = 2``). Knots: :func:`rcs_knots` follows ``rcspline.eval``'s default placement line by line,
including its two adjustments: with fewer than 100 values the outer knots move to the 5th smallest
and 5th largest values, and a value at either extreme held by at least ``fractied`` (5%) of the
rows (zero alcohol, say), while no interior value is that common, becomes a knot of its own just
inside it. Basis (:func:`rcs_basis`): for knots t₁ < … < t_k and ``kd = (t_k − t₁)^(2/3)``,

    x_j = ((x − t_j)/kd)₊³ + [(t_{k−1} − t_j)((x − t_k)/kd)₊³ − (t_k − t_j)((x − t_{k−1})/kd)₊³] / (t_k − t_{k−1})

for j = 1…k − 2, beside x itself: k − 1 columns, linear beyond the outer knots (Harrell 2015,
*Regression Modeling Strategies*, 2nd ed., §2.4.5). The columns are named as ``rms`` prints them:
``x``, ``x'``, ``x''``.

**The tests** (:func:`exposure_tests`), under inference, use the coefficient table's own
covariance (HC3, CR2 or the model's information): for a spline, a Wald test that all of its
columns are zero (any association) and that its nonlinear columns are zero (``rms::anova``'s
"Nonlinear" row); for quintiles, the trend test of the pack: the exposure scored by the median of
its quintile (medians of the fitting rows), one column in place of the four indicators, its
coefficient tested on the same covariance.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

Form = Literal["linear", "spline", "quintiles"]
FORMS: tuple[str, ...] = ("linear", "spline", "quintiles")
# Harrell's default percentiles (``rcspline.eval``: ``seq(outer, 1 − outer, length = k)`` with
# outer 0.10 for 3 knots, 0.05 for 4–6), which are the pack's "10/50/90" and "5/35/65/95".
KNOT_PERCENTILES: dict[int, tuple[float, ...]] = {
    3: (0.10, 0.50, 0.90),
    4: (0.05, 0.35, 0.65, 0.95),
    5: (0.05, 0.275, 0.50, 0.725, 0.95),
}
KNOT_CHOICES = tuple(KNOT_PERCENTILES)
DEFAULT_KNOTS = 4  # Harrell (RMS §2.4.6): four knots are "a good compromise between flexibility and loss of precision"
FRACTIED = 0.05  # ``rcspline.eval``'s and ``rms::rcs``'s default
SMALL_SAMPLE = 100  # below this many values, the outer knots are the 5th smallest and largest
QUINTILES = 5
SCORE_SUFFIX = " (quintile median)"


# ── the restricted cubic spline ──────────────────────────────────────────────


def _quantile(values: np.ndarray, p: Sequence[float]) -> np.ndarray:
    """R's ``quantile`` (type 7, its default), which is numpy's ``linear`` method."""
    return np.quantile(values, np.asarray(p, dtype=float), method="linear")


def rcs_knots(x: Any, n_knots: int = DEFAULT_KNOTS, *, fractied: float = FRACTIED
              ) -> tuple[np.ndarray, list[str]]:
    """Knot locations as ``Hmisc::rcspline.eval`` places them when only their number is given.

    Returns the knots and plain notes on any departure from the nominal percentiles. Raises
    ``ValueError`` where ``rcspline.eval`` stops: fewer than 6 values, or fewer than 3 distinct
    knots.
    """
    xx = np.asarray(x, dtype=float)
    xx = xx[np.isfinite(xx)]
    n = len(xx)
    nk = int(n_knots)
    if n < 6:
        raise ValueError(f"A spline needs at least 6 recorded values to place its knots; there "
                         f"are {n}.")
    if nk < 3:
        raise ValueError("A restricted cubic spline needs at least 3 knots.")
    notes: list[str] = []
    xu = np.unique(xx)
    nxu = len(xu)
    if nxu - 2 <= nk:
        knots = xu[1:-1]
        notes.append(f"only {nxu} distinct values, so the knots are the {nxu - 2} interior ones")
    else:
        outer = 0.05 if nk > 3 else 0.1
        if nk > 6:
            outer = 0.025
        nke = nk
        first: float | None = None
        last: float | None = None
        override_first = override_last = False
        if 0 < fractied < 1:
            _, counts = np.unique(xx, return_counts=True)
            f = counts / n
            interior = f[1:-1]
            if (interior.max() if interior.size else -np.inf) < fractied:
                if f[0] >= fractied:
                    first = float(xx[xx > xx.min()].min())
                    xx = xx[xx > first]
                    nke -= 1
                    override_first = True
                    notes.append(f"{f[0]:.0%} of the values are the lowest one, so a knot sits at "
                                 f"the next value up")
                if f[-1] >= fractied:
                    last = float(xx[xx < xx.max()].max())
                    xx = xx[xx < last]
                    nke -= 1
                    override_last = True
                    notes.append(f"{f[-1]:.0%} of the values are the highest one, so a knot sits "
                                 f"at the next value down")
        if nke == 1:
            knots = np.array([float(np.median(xx))])
        else:
            if nxu <= nke:
                knots = xu.astype(float)
            else:
                p = (np.linspace(0.5, 1.0 - outer, nke) if nke == 2
                     else np.linspace(outer, 1.0 - outer, nke))
                knots = _quantile(xx, p)
                if len(np.unique(knots)) < min(nke, 3):
                    knots = _quantile(xx, np.linspace(outer, 1.0 - outer, 2 * nke))
                    if first is not None and len(np.unique(knots)) < 3:
                        mid = (first + last) / 2.0 if last is not None else float(np.median(xx))
                        knots = np.sort(np.array([first, mid, last if last is not None
                                                  else float(_quantile(xx, [1.0 - outer])[0])]))
                    if len(np.unique(knots)) < 3:
                        raise ValueError("Fewer than 3 distinct knots can be placed: the values "
                                         "are too heavily tied.")
                    notes.append(f"the usual percentiles gave fewer than {nke} distinct knots, so "
                                 f"they were spread over {2 * nke} percentiles instead")
            knots = np.array(knots, dtype=float)
            if len(xx) < SMALL_SAMPLE:
                ordered = np.sort(xx)
                if not override_first:
                    knots[0] = ordered[4]
                if not override_last:
                    knots[nke - 1] = ordered[len(ordered) - 5]
                notes.append(f"with {len(xx)} values (fewer than {SMALL_SAMPLE}), the outer knots "
                             f"are the 5th smallest and 5th largest")
        knots = np.concatenate([[first] if first is not None else [], knots,
                                [last] if last is not None else []])
    knots = np.unique(np.asarray(knots, dtype=float))
    if len(knots) < 3:
        raise ValueError("Fewer than 3 distinct knots can be placed: the values are too heavily "
                         "tied for a spline.")
    return knots, notes


def rcs_basis(x: Any, knots: Sequence[float]) -> np.ndarray:
    """The k − 2 nonlinear columns of Harrell's restricted cubic spline (norm = 2), n × (k − 2).

    Missing values stay missing. Beyond the outer knots every column is linear in x.
    """
    t = np.asarray(knots, dtype=float)
    k = len(t)
    if k < 3:
        raise ValueError("A restricted cubic spline needs at least 3 knots.")
    x = np.asarray(x, dtype=float)
    kd = (t[-1] - t[0]) ** (2.0 / 3.0)
    last, before = t[-1], t[-2]

    def cube(c: float) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.maximum((x - c) / kd, 0.0) ** 3

    tail_last, tail_before = cube(last), cube(before)
    out = np.empty((len(x), k - 2))
    for j in range(k - 2):
        out[:, j] = cube(t[j]) + ((before - t[j]) * tail_last
                                  - (last - t[j]) * tail_before) / (last - before)
    return out


def spline_names(column: str, n_knots: int) -> list[str]:
    """``x``, ``x'``, ``x''``…: the spline's k − 1 columns, as ``rms`` labels them."""
    return [column] + [f"{column}{chr(39) * (j + 1)}" for j in range(n_knots - 2)]


# ── quintiles ────────────────────────────────────────────────────────────────


def quantile_cuts(x: Any, groups: int = QUINTILES) -> np.ndarray:
    """The groups − 1 inner cut points of ``x`` (R type 7 quantiles, as ``pandas.qcut`` takes)."""
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    return _quantile(values, np.arange(1, groups) / groups)


def quantile_group(x: Any, cuts: Sequence[float]) -> np.ndarray:
    """Each value's group, 0 for the lowest: right-closed intervals, the lowest group holding its
    cut point (``pandas.qcut``'s and R ``cut(…, include.lowest = TRUE)``'s rule). A value below
    the lowest or above the highest value seen in the fit falls in the end group; a missing value
    gets −1."""
    x = np.asarray(x, dtype=float)
    out = np.searchsorted(np.asarray(cuts, dtype=float), x, side="left").astype(np.int64)
    out[~np.isfinite(x)] = -1
    return out


def quintile_names(column: str) -> list[str]:
    """``x_Q2`` … ``x_Q5``: the indicators against the lowest quintile, ``x``'s Q1."""
    return [f"{column}_Q{g}" for g in range(2, QUINTILES + 1)]


# ── the pipeline step ────────────────────────────────────────────────────────


def _form_of(spec: Any) -> tuple[str, int | None]:
    if isinstance(spec, str):
        return spec, None
    get = spec.get if isinstance(spec, Mapping) else (lambda k, s=spec: getattr(s, k, None))
    return str(get("form")), (int(get("knots")) if get("knots") is not None else None)


class ExposureForms(TransformerMixin, BaseEstimator):
    """Each named column replaced by its form: a spline basis, or quintile indicators.

    ``forms``: column -> ``{"form": "spline", "knots": 4}`` or ``{"form": "quintiles"}`` (a
    ``linear`` form, or a column not named, passes through). Fit on the rows each fit sees; the
    knots, the cut points and each quintile's median are kept. Row-local once fit.
    """

    def __init__(self, forms: Mapping[str, Any] | None = None):
        self.forms = forms

    def _plan(self) -> dict[str, tuple[str, int | None]]:
        plan = {}
        for column, spec in (self.forms or {}).items():
            form, knots = _form_of(spec)
            if form not in FORMS:
                raise ValueError(f"Unknown exposure form {form!r} for {column}.")
            if form != "linear":
                plan[str(column)] = (form, knots or (DEFAULT_KNOTS if form == "spline" else None))
        return plan

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ExposureForms":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ExposureForms needs a pandas DataFrame with named columns.")
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.knots_: dict[str, np.ndarray] = {}
        self.knot_notes_: dict[str, list[str]] = {}
        self.cuts_: dict[str, np.ndarray] = {}
        self.medians_: dict[str, np.ndarray] = {}
        self.counts_: dict[str, list[int]] = {}
        for column, (form, knots) in self._plan().items():
            if column not in X.columns:
                raise ValueError(f"`{column}` is not among the model's inputs at this step (an "
                                 f"energy adjustment may have replaced it), so its form cannot "
                                 f"be applied.")
            if not pd.api.types.is_numeric_dtype(X[column]):
                raise ValueError(f"`{column}` is not numeric, so it has no spline or quintiles.")
            values = X[column].to_numpy(dtype=float, na_value=np.nan)
            values = values[np.isfinite(values)]
            if form == "spline":
                self.knots_[column], self.knot_notes_[column] = rcs_knots(values, knots)
                continue
            cuts = quantile_cuts(values)
            group = quantile_group(values, cuts)
            counts = np.bincount(group, minlength=QUINTILES)[:QUINTILES]
            if len(np.unique(cuts)) < QUINTILES - 1 or (counts == 0).any():
                raise ValueError(f"`{column}` has too many tied values to be cut into five "
                                 f"groups: its quintile cut points are "
                                 f"{', '.join(f'{c:g}' for c in cuts)}.")
            self.cuts_[column] = cuts
            self.medians_[column] = np.array([float(np.median(values[group == g]))
                                              for g in range(QUINTILES)])
            self.counts_[column] = [int(c) for c in counts]
        return self

    def _outputs(self, column: str) -> list[str]:
        plan = self._plan()
        if column not in plan:
            return [column]
        form, _ = plan[column]
        if form == "spline":
            return spline_names(column, len(self.knots_[column]))
        return quintile_names(column)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names: list[str] = []
        for column in self.feature_names_in_:
            names.extend(self._outputs(str(column)))
        return np.asarray(names, dtype=object)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "knots_"):
            raise ValueError("ExposureForms is not fitted yet.")
        plan = self._plan()
        parts: dict[str, Any] = {}
        for name in X.columns:
            column = str(name)
            if column not in plan:
                parts[column] = X[name]
                continue
            values = X[name].to_numpy(dtype=float, na_value=np.nan)
            form, _ = plan[column]
            if form == "spline":
                basis = rcs_basis(values, self.knots_[column])
                names = spline_names(column, len(self.knots_[column]))
                parts[names[0]] = values
                for j, out in enumerate(names[1:]):
                    parts[out] = basis[:, j]
                continue
            group = quantile_group(values, self.cuts_[column])
            for g, out in enumerate(quintile_names(column), start=1):
                parts[out] = np.where(group < 0, np.nan, (group == g).astype(float))
        return pd.DataFrame(parts, index=X.index)

    def scores(self, column: str, values: Any) -> np.ndarray:
        """The trend score: each value's quintile median (from the fitting rows)."""
        group = quantile_group(values, self.cuts_[column])
        medians = self.medians_[column]
        return np.where(group < 0, np.nan, medians[np.clip(group, 0, QUINTILES - 1)])

    def lineage(self) -> list[dict[str, Any]]:
        """``{output, inputs, operation, formula}`` per output, as the energy step's."""
        plan = self._plan()
        entries: list[dict[str, Any]] = []
        for column in (str(c) for c in self.feature_names_in_):
            if column not in plan:
                entries.append({"output": column, "inputs": [column], "operation": "kept",
                                "formula": None})
                continue
            form, _ = plan[column]
            if form == "spline":
                knots = self.knots_[column]
                op = f"restricted cubic spline, {len(knots)} knots"
                formula = knot_sentence(column, knots, self.knot_notes_[column])
                for out in spline_names(column, len(knots)):
                    entries.append({"output": out, "inputs": [column], "operation": op,
                                    "formula": formula})
                continue
            cuts = self.cuts_[column]
            for g, out in enumerate(quintile_names(column), start=1):
                low, high = cuts[g - 1], (cuts[g] if g < len(cuts) else None)
                within = f"{low:.4g} < {column}" + (f" ≤ {high:.4g}" if high is not None else "")
                entries.append({"output": out, "inputs": [column], "operation": "quintile indicator",
                                "formula": f"1 when {within} (quintile {g + 1}); quintile 1, "
                                           f"{column} ≤ {cuts[0]:.4g}, is the reference"})
        return entries


def knot_sentence(column: str, knots: Sequence[float], notes: Sequence[str] = ()) -> str:
    """``knots at 12.1, 25, 41.3 and 70.2 (the 5th, 35th, 65th and 95th percentiles)``."""
    values = [f"{k:.4g}" for k in knots]
    listed = values[0] if len(values) == 1 else f"{', '.join(values[:-1])} and {values[-1]}"
    nominal = KNOT_PERCENTILES.get(len(knots))
    where = ""
    if nominal is not None and not notes:
        pct = [f"{100 * p:g}" for p in nominal]
        named = [f"{p}{_ordinal_suffix(p)}" for p in pct]
        where = f" (the {', '.join(named[:-1])} and {named[-1]} percentiles)"
    extra = f"; {'; '.join(notes)}" if notes else ""
    return f"knots of {column} at {listed}{where}{extra}"


def _ordinal_suffix(number: str) -> str:
    if "." in number:
        return "th"
    n = int(number)
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def form_columns(step: ExposureForms) -> dict[str, list[str]]:
    """Formed column -> its output columns in the model matrix."""
    return {column: step._outputs(column) for column in step._plan()}


def formed_meanings(terms: Mapping[str, str], step: ExposureForms | None) -> dict[str, str]:
    """Each formed column's meaning carried to the matrix columns the form made of it.

    ``terms`` maps a matrix column to what its coefficient means (``methods.energy.describe_model``,
    WP6), read as if each formed column were still one straight-line term; the form step turns it
    into a spline basis or quintile indicators (WP12a), whose coefficients mean a part of that
    curve, not a per-unit effect."""
    out = dict(terms)
    if step is None:
        return out
    plan = step._plan()
    for column, outputs in form_columns(step).items():
        base = out.pop(column, None) if column not in outputs else out.get(column)
        if base is None:
            continue
        form = plan[column][0]
        for k, name in enumerate(outputs):
            if form == "spline":
                out[name] = (f"{base}: term {k + 1} of {len(outputs)} of its restricted cubic "
                             f"spline (read the curve and its tests, not one term)")
            else:
                out[name] = f"{base}: quintile {k + 2} against the lowest"
    return out


def formed_name(column: str, adjustment: Any) -> str:
    """The column the form step receives for ``column`` after the energy step: the step that runs
    first may have replaced it (``EnergyAdjuster``'s output names: the residual method's
    ``{column}_adj``, the density methods' ``{column}_per_{energy}``). The form then shapes the
    energy-adjusted intake, and its terms carry that name. Unchanged without an adjustment, under
    the standard model (energy enters beside the nutrient) and for a column it does not adjust."""
    if adjustment is None:
        return column
    get = adjustment.get if isinstance(adjustment, Mapping) else (
        lambda k, a=adjustment: getattr(a, k, None))
    method, energy = get("method"), get("energy_column")
    if column not in (get("nutrients") or []) or method in (None, "none", "standard"):
        return column
    if method == "residual":
        return f"{column}_adj"
    if method in ("density", "density_multivariate"):
        return f"{column}_per_{energy}"
    return column  # the partition replaces the nutrient by its kcal; the decision refuses a form


def adjusted_forms(forms: Mapping[str, Any], adjustment: Any) -> dict[str, Any]:
    """``forms`` keyed by the columns the form step receives (:func:`formed_name`)."""
    return {formed_name(str(column), adjustment): spec for column, spec in forms.items()}


def model_terms(predictors: Sequence[str], forms: Mapping[str, Any] | None) -> int:
    """How many columns ``predictors`` put in the model matrix once formed: a spline k − 1, a
    quintile exposure 4, any other predictor 1 (the count the shelf's per-predictor rules read)."""
    total = 0
    for column in predictors:
        spec = (forms or {}).get(column)
        form, knots = _form_of(spec) if spec is not None else ("linear", None)
        if form == "spline":
            total += (knots or DEFAULT_KNOTS) - 1
        elif form == "quintiles":
            total += QUINTILES - 1
        else:
            total += 1
    return total


# ── the options, labeled as north star 5 asks ────────────────────────────────


def options(purpose: str | None) -> list[dict[str, Any]]:
    """The exposure-form options for ``purpose``, ordered by soundness for it, each labeled
    *customary in* (with its source) and *sound for* (with the reason), independently.

    Under inference the spline leads and quintiles are tagged customary (AUDIT_REPORT §3.6 and
    the ME-17 recommendation); under prediction the spline still leads (it nests the line), and
    quintiles come last because they discard the variation within each fifth.
    """
    spline = {
        "value": "spline", "label": "Restricted cubic spline",
        "customary": "Increasingly customary in nutritional epidemiology: \"now near-default\" "
                     "(NUTRITION_PACK §07G; Desquilbet & Mariotti 2010).",
        "sound": "Sound: a smooth curve with few parameters, linear in the tails, with a test of "
                 "nonlinearity; knots at Harrell's percentiles of the fitting rows.",
        "consequence": "Adds a curve's terms per exposure and a test of whether it bends.",
    }
    linear = {
        "value": "linear", "label": "Straight line",
        "customary": "Customary: one coefficient per unit of intake.",
        "sound": "Sound when the relation is close to a line; a curve it misses biases the slope.",
        "consequence": "One coefficient per unit; no curvature.",
    }
    quintiles = {
        "value": "quintiles", "label": "Quintiles with a trend test",
        "customary": "Customary in nutritional epidemiology: \"quintiles remain expected "
                     "alongside\" the spline (NUTRITION_PACK §07G).",
        "sound": ("Weaker for inference: cutting a continuous intake into fifths discards the "
                  "variation within each fifth and assumes a step at each cut; the trend test "
                  "scores each fifth by its median." if purpose == "inference" else
                  "Weak for prediction: five steps discard the variation within each fifth."),
        "consequence": "Four indicators against the lowest fifth, and a trend across medians.",
    }
    return [spline, linear, quintiles]


# ── tests under inference ────────────────────────────────────────────────────


def _index(names: Sequence[str]) -> dict[str, int]:
    return {str(n): i for i, n in enumerate(names)}


def _reference(info: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
               idx: Sequence[int]) -> tuple[str, float | None, str] | None:
    """The joint test's reference distribution for this table: (kind, denominator df, words)."""
    covariance = info.get("covariance")
    if covariance == "HC3":
        dfs = [rows[i].get("df") for i in idx if rows[i].get("df") is not None]
        if not dfs:
            return None
        return "F", float(dfs[0]), "HC3 covariance"
    if covariance in ("CR2", "CR1", "CR0"):  # CR0: the Cox model's Lin–Wei sandwich
        g = info.get("n_clusters")
        if not g or g < 2:
            return None
        return "F", float(g - 1), (f"{covariance} cluster-robust covariance by "
                                   f"`{info.get('grouped_by')}` (G = {g:,}), on G − 1 "
                                   f"denominator degrees of freedom as Stata does")
    if covariance == "model":
        return "chi2", None, "the model's information"
    if covariance == "design":
        # MS4: a design-based covariance rests on the design's degrees of freedom; the adjusted
        # Wald F (``models.survey.adjusted_wald``) allows for that.
        d = (info.get("survey") or {}).get("df")
        if not d or d < 1:
            return None
        return "F_design", float(d), (f"the Taylor-linearized covariance over the survey design, "
                                      f"an adjusted Wald F for its {int(d):,} design degrees of "
                                      f"freedom, Korn & Graubard 1990")
    return None


def wald_test(estimates: np.ndarray, cov: np.ndarray, idx: Sequence[int], info: Mapping[str, Any],
              rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The Wald test that the coefficients ``idx`` are all zero, on the table's covariance:
    ``W = bᵀ V⁻¹ b`` over q = rank(V) coefficients; F = W/q on (q, df) for HC3 and cluster-robust
    tables, χ²(q) for model-based ones. None when the table has no covariance to test on."""
    from scipy import stats

    ref = _reference(info, rows, idx)
    if ref is None or cov is None:
        return None
    b = np.asarray(estimates, dtype=float)[list(idx)]
    V = np.asarray(cov, dtype=float)[np.ix_(list(idx), list(idx))]
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(V))):
        return None
    q = int(np.linalg.matrix_rank(V))
    if q == 0:
        return None
    W = float(b @ np.linalg.pinv(V) @ b)
    kind, den, words = ref
    if kind == "F_design":
        from turbotab.core.models.survey import adjusted_wald

        adjusted = adjusted_wald(W, q, int(den))
        if adjusted is None:
            return None
        F, den_df, p = adjusted
        return {"statistic": F, "df_num": q, "df_den": den_df, "distribution": "F", "p": p,
                "basis": words}
    if kind == "F":
        F = W / q
        return {"statistic": F, "df_num": q, "df_den": den, "distribution": "F",
                "p": float(stats.f.sf(F, q, den)), "basis": words}
    return {"statistic": W, "df_num": q, "df_den": None, "distribution": "chi2",
            "p": float(stats.chi2.sf(W, q)), "basis": words}


def _statistic_words(test: Mapping[str, Any]) -> str:
    from turbotab.core.models.inference import format_p

    if test["distribution"] == "F":
        stat = f"F({test['df_num']}, {test['df_den']:,.0f}) = {test['statistic']:.2f}"
    else:
        stat = f"χ²({test['df_num']}) = {test['statistic']:.2f}"
    return f"{stat}, p = {format_p(test['p'])}"


def form_step(pipeline: Any) -> ExposureForms | None:
    named = getattr(pipeline, "named_steps", {}) or {}
    step = named.get("form")
    return step if isinstance(step, ExposureForms) else None


def _before_form(pipeline: Any, X: pd.DataFrame) -> pd.DataFrame:
    """What the form step saw: X through every step before it."""
    names = [name for name, _ in pipeline.steps]
    at = names.index("form")
    return X if at == 0 else pipeline[:at].transform(X)


def exposure_tests(family: Any, pipeline: Any, X: pd.DataFrame, y: Any, *, task: str,
                   clusters: Any, table: Any, outcome: Any = None,
                   survey: Any = None) -> tuple[list[dict[str, Any]], list[str]]:
    """The tests each formed exposure carries under inference, and any concerns about them.

    ``table`` is the family's inference table on ``pipeline`` (rows, ``info``, ``cov``). A spline
    gets two Wald tests (every term; the nonlinear terms); quintiles get the trend test, a refit
    of the family's inference table with the exposure scored by its quintile medians. The refit
    takes the table's outcome and, under a surveyed population, its design (WP8, WP10), when the
    family's ``inference_matrix`` accepts them.
    """
    step = form_step(pipeline)
    if step is None or table is None:
        return [], []
    rows = list(table.rows)
    names = [str(r["feature"]) for r in rows]
    where = _index(names)
    estimates = np.array([np.nan if r["estimate"] is None else r["estimate"] for r in rows], float)
    cov = getattr(table, "cov", None)
    info = dict(table.info or {})
    tests: list[dict[str, Any]] = []
    concerns: list[str] = []
    if task == "multiclass" or (task == "ordinal" and not getattr(family, "ordered_levels", False)):
        return [], ["Tests of an exposure's form are not computed for a multinomial model: each "
                    "level has its own curve."]
    if info.get("refused"):
        return [], []
    plan = step._plan()
    for column, (form, _) in plan.items():
        outputs = step._outputs(column)
        if not all(o in where for o in outputs):
            continue
        if form == "spline":
            idx = [where[o] for o in outputs]
            knots = knot_sentence(column, step.knots_[column], step.knot_notes_[column])
            for test_kind, chosen, what in (
                    ("overall", idx, f"all {len(idx)} terms of `{column}` are"),
                    ("nonlinear", idx[1:], f"the {len(idx) - 1} nonlinear "
                                           f"term{'s' if len(idx) > 2 else ''} of `{column}` "
                                           f"{'are' if len(idx) > 2 else 'is'}")):
                result = wald_test(estimates, cov, chosen, info, rows)
                if result is None:
                    if test_kind == "overall":
                        concerns.append(f"The spline tests for `{column}` need a Wald covariance, "
                                        f"which this table ({info.get('covariance')}) does not "
                                        f"give.")
                    continue
                caption = (f"Wald test that {what} zero ({result['basis']}): "
                           f"{_statistic_words(result)}. {knots[0].upper()}{knots[1:]}.")
                tests.append({"column": column, "form": "spline", "test": test_kind,
                              "statistic": result["statistic"], "df_num": result["df_num"],
                              "df_den": result["df_den"], "distribution": result["distribution"],
                              "p": result["p"], "estimate": None, "ci_low": None, "ci_high": None,
                              "knots": [float(v) for v in step.knots_[column]], "medians": None,
                              "caption": caption})
            continue
        # MS2: the global test that every quintile indicator is zero, a multi-df Wald test on the
        # table's own covariance (pooled by D1 under multiple imputation).
        idx = [where[o] for o in outputs]
        result = wald_test(estimates, cov, idx, info, rows)
        if result is not None:
            tests.append({"column": column, "form": "quintiles", "test": "global",
                          "statistic": result["statistic"], "df_num": result["df_num"],
                          "df_den": result["df_den"], "distribution": result["distribution"],
                          "p": result["p"], "estimate": None, "ci_low": None, "ci_high": None,
                          "knots": None, "medians": [float(v) for v in step.medians_[column]],
                          "caption": (f"Wald test that all {len(idx)} quintile indicators of "
                                      f"`{column}` are zero ({result['basis']}): "
                                      f"{_statistic_words(result)}.")})
        trend = _trend(family, pipeline, step, column, X, y, task=task, clusters=clusters,
                       outcome=outcome, survey=survey)
        if trend is None:
            concerns.append(f"The trend test for `{column}` could not be computed.")
            continue
        tests.append(trend)
    return tests, concerns


def _trend(family: Any, pipeline: Any, step: ExposureForms, column: str, X: pd.DataFrame, y: Any,
           *, task: str, clusters: Any, outcome: Any = None,
           survey: Any = None) -> dict[str, Any] | None:
    """The trend test on quintile medians: the model matrix with ``column``'s indicators replaced
    by one column, each row's quintile median, refit by the family's own inference table."""
    from turbotab.core.models.linear import model_matrix

    refit = getattr(family, "inference_matrix", None)
    if refit is None:
        return None
    matrix = model_matrix(pipeline, X)
    indicators = quintile_names(column)
    if not all(c in matrix.columns for c in indicators):
        return None
    raw = _before_form(pipeline, X)[column].to_numpy(dtype=float, na_value=np.nan)
    score_name = f"{column}{SCORE_SUFFIX}"
    at = list(matrix.columns).index(indicators[0])
    kept = matrix.drop(columns=indicators)
    kept.insert(at, score_name, step.scores(column, raw))
    classes = list(getattr(pipeline[-1], "classes_", [])) or None
    import inspect

    accepts = inspect.signature(refit).parameters
    extra = {name: value for name, value in (("outcome", outcome), ("survey", survey))
             if value is not None and name in accepts}
    if survey is not None and "survey" not in extra:
        return None  # a trend test without the design beside a design-based table would mislead
    table = refit(kept, y, task=task, classes=classes, clusters=clusters, **extra)
    row = next((r for r in table.rows if str(r["feature"]) == score_name), None)
    if row is None or row.get("p") is None:
        return None
    medians = ", ".join(f"{m:.4g}" for m in step.medians_[column])
    from turbotab.core.models.inference import format_p

    reference = "t" if row.get("df") is not None else "z"
    caption = (f"Trend across quintiles of `{column}`: each row scored by its quintile's median "
               f"({medians}, from the fitting rows), entered as one continuous term in place of "
               f"the four indicators; its coefficient {row['estimate']:+.4g} per unit, "
               f"p = {format_p(row['p'])} ({table.info.get('caption', '').rstrip('.')}).")
    if survey is not None:
        # The review of the modeling sequence (§1 step 5, quintiles): "Under survey design, state
        # whether cut points are weighted."
        caption += (" The cut points and medians are the analyzed rows' own, unweighted; the "
                    "trend's coefficient and interval are design-based.")
    return {"column": column, "form": "quintiles", "test": "trend",
            "statistic": (row["estimate"] / row["se"]) if row.get("se") else None,
            "df_num": 1, "df_den": row.get("df"), "distribution": reference, "p": row["p"],
            "estimate": row["estimate"], "ci_low": row.get("ci_low"), "ci_high": row.get("ci_high"),
            "knots": None, "medians": [float(m) for m in step.medians_[column]],
            "caption": caption}


def describe(forms: Mapping[str, Any]) -> str:
    """The pipeline step's detail line: which column takes which form, and where its knots or cut
    points come from."""
    parts = []
    learned = []
    for column, spec in forms.items():
        form, knots = _form_of(spec)
        if form == "spline":
            parts.append(f"a restricted cubic spline of {column} with {knots or DEFAULT_KNOTS} "
                         f"knots at Harrell's percentiles")
            if "knots" not in learned:
                learned.append("knots")
        elif form == "quintiles":
            parts.append(f"quintile indicators of {column} against its lowest fifth")
            if "cut points" not in learned:
                learned.append("cut points")
    if not parts:
        return "Every exposure enters as a straight line."
    joined = parts[0] if len(parts) == 1 else f"{'; '.join(parts[:-1])}; and {parts[-1]}"
    return (f"{joined[0].upper()}{joined[1:]}. The {' and '.join(learned)} are learned on the rows "
            f"each fit sees, every training fold included.")


__all__ = [
    "DEFAULT_KNOTS", "ExposureForms", "FORMS", "FRACTIED", "Form", "KNOT_CHOICES",
    "KNOT_PERCENTILES", "QUINTILES", "describe", "exposure_tests", "form_columns", "form_step",
    "formed_meanings",
    "adjusted_forms", "formed_name", "model_terms",
    "knot_sentence", "options", "quantile_cuts", "quantile_group", "quintile_names", "rcs_basis",
    "rcs_knots", "spline_names", "wald_test",
]
