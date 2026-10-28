"""``featurewise``: each exposure tested on its own, adjusted for the covariates, with
Benjamini–Hochberg false-discovery control (AUDIT_REPORT §5 WP11; closes ME-18).

Omics inference has no sound family on the shelf without it: at p ≥ n least squares has no unique
fit and elastic net's shrunk coefficients carry no confidence intervals. The field's workhorse is
per-feature testing with a multiple-testing correction — METABOLOMICS_PACK §08, SETTLED: "per-feature
testing with multiple-testing correction is expected, and its absence is a fatal flaw in review …
covariate-adjusted → linear model on log values … Benjamini–Hochberg at q < 0.05 is the field
standard" — and limma's design for expression data is the same linear model, one per gene.

The model, one per exposure ``j`` (the predictors with the role exposure; every other column of the
model matrix — covariates, energy, one-hot levels — is the adjustment set ``Z``):

* **numeric outcome** — ``outcome ~ x_j + Z`` by least squares; the estimate is the change in the
  outcome per unit of ``x_j`` (per log2 unit once the values are normalized to log scale);
* **two-level outcome** — ``x_j ~ event + Z`` by least squares, the limma design; the estimate is
  the difference in ``x_j``'s mean between the event and the other level, adjusted for ``Z`` (a
  log2 fold change on log2 values).

Standard errors: each test's own residual variance (classical least squares, limma's linear model
without its variance moderation; Smyth 2004), on t(n − rank[1, Z] − 1). Under the null that the
exposure is unrelated to the outcome given ``Z``, the residuals do not depend on the exposure, so
this t-test is exact for normal errors whatever the exposure's distribution — and the null is what
a false-discovery rate is about. The linear family's HC3 is not used here: Benjamini–Hochberg
works in the far tail (a p-value near 0.05/m), where an HC3 t-statistic is much heavier-tailed than
its t reference. Simulated (n = 40, 500 null exposures, one covariate; the acceptance suite,
``test_wp11_omics.py``, keeps the comparison): the chance of any false discovery after
Benjamini–Hochberg at q = 0.05 was 0.11 with HC3 on normal exposures and 0.74 on log-normal ones
(200 replicates), against 0.054 and 0.051 with these standard errors (2,000 replicates, Monte
Carlo error 0.005). The price: an interval around a real effect assumes equal residual variance.

When a unit's rows repeat: CR2 with Bell–McCaffrey degrees of freedom, the math layer's estimator
(up to :data:`CR2_MAX_FEATURES` exposures; beyond, the table is refused rather than reported as if
the rows were independent). p-values are two-sided; ``q`` is the Benjamini–Hochberg adjusted
p-value over the exposures tested, and a discovery is ``q <`` :data:`FDR`.

Every test is computed by Frisch–Waugh–Lovell: ``Z`` is projected out once, so 20,000 tests cost a
few matrix products. The family makes no predictions (``predicts = False``): the fit stage reports
its table and no cross-validated score.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import (
    Assessment,
    FamilyBase,
    Identity,
    InferenceDecl,
    Situation,
    register_family,
)

FDR = 0.05
LEVEL = 0.95
CR2_MAX_FEATURES = 200  # CR2 is one design per exposure; beyond this the table is refused when rows repeat
_BLOCK_FLOATS = 4_000_000  # n × exposures per block of the vectorized tests (32 MB)
INDICATOR = "missingindicator_"
OMICS = ("genomics", "metabolomics")
PLS_DA_ABSENT = ("PLS-DA is not offered: customary in metabolomics, it is sound only with nested "
                 "cross-validation and permutation tests, and VIP > 1 ranks features without "
                 "testing them.")
# Its refusal where it has nothing to test, at the models answer, or of its table at the fit (the
# swarm's C2; the table's, not the fit's, by the orchestrator's ruling (b) on the swarm's tail).
NO_STUDY_FACTOR = "featurewise_no_study_factor"
NO_STUDY_FACTOR_WORDS = ("No study factor to test: feature-wise regression tests the columns with "
                         "the role of what you study, one at a time, adjusted for the rest")
ROLES_WAY = {"label": "Give the columns you study the role of what you study (the roles "
                      "question)", "decision": None}
ANOTHER_FAMILY = {"label": "Choose another model family", "decision": None}
# … and its way forward where a study factor has nothing left once the covariates are projected
# out (:func:`not_estimable`): what each test adjusts for is the adjustment answers'.
ADJUSTMENT_WAY = {"label": "Revisit what each test adjusts for (the adjustment question)",
                  "decision": None}


# … and its table's refusal where a study factor's declared form enters it as other than one
# straight-line column (a spline's terms, indicators): no one column is its estimate per unit (the
# verifier's B1).
FORMED_STUDY_FACTOR = "featurewise_formed_study_factor"
FORMED_WORDS = ("Feature-wise regression tests each study factor as one straight-line column, one "
                "at a time, adjusted for the rest")
# Each form that is not a straight line: what it is, and the columns it enters the factor as
# (``methods.exposure_form.ExposureForms._outputs``).
FORM_ENTERS: dict[str, tuple[str, str]] = {
    "spline": ("a restricted cubic spline", "its spline terms"),
    "zero_spline": ("non-consumers apart beside a spline among consumers",
                    "a consumer indicator and its spline terms"),
    "quintiles": ("quintiles", "indicators of its quintiles"),
    "categories": ("categories at declared cut points", "indicators of its categories"),
    "optimal": ("a data-derived cut point", "an indicator of the values above the cut"),
}


def _who(named: Sequence[str], features: Sequence[str]) -> list[str]:
    """Each study factor by the column the roles answer gave the role (``named``), with the name
    the energy step gives it where the model matrix holds it under that one (``features``)."""
    return [f"`{n}`" if n == f else f"`{n}` (entered as `{f}` by the energy adjustment)"
            for n, f in zip(named, features)]


def no_study_factor(features: Sequence[str] | None,
                    named: Sequence[str] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """The table's refusal (its reason and ways forward) where the model matrix holds no column to
    test: no column has the role of what you study (``features`` empty), or none of those that
    have it is a column of the matrix by its name (a category enters as its indicators).
    ``named``: the columns the roles answer gave the role, beside ``features``, the names the
    matrix holds them under (an energy step's), so the words give the role to the column that has
    it. The table is refused, not the fit, as :func:`formed_study_factor`'s is: every other family
    chosen beside it is fit and shown (the orchestrator's ruling (b) on the swarm's tail)."""
    from turbotab.core.voice import listing

    sought = [str(f) for f in features or []]
    if not sought:
        return (f"{NO_STUDY_FACTOR_WORDS}; no column has that role.",
                [dict(ROLES_WAY), dict(ANOTHER_FAMILY)])
    raw = [str(n) for n in named] if named is not None and len(named) == len(sought) else sought
    one = len(sought) == 1
    return (f"{NO_STUDY_FACTOR_WORDS}; {listing(_who(raw, sought), ticked=False)} "
            f"{'has' if one else 'have'} that role, but the model matrix holds no column of "
            f"{'that name' if one else 'their names'}.", [dict(ANOTHER_FAMILY)])


def formed_study_factor(forms: Mapping[str, Any], named: Sequence[str],
                        features: Sequence[str]) -> tuple[str, list[dict[str, Any]]]:
    """The table's refusal (its reason and ways forward) of the study factors whose declared form
    is not a straight line (``forms``: the column the roles answer named -> its form): each enters
    the model matrix as its form's columns, a spline's terms or indicators, and testing one of them
    beside the others reports a part of the curve as the factor's estimate per unit (a spline's
    slope below its first knot). It names each column as the roles answer did, with the energy
    step's name for it (``named`` beside ``features``), and its form; its ways forward the straight
    line for each, and another family. The table is refused, not the fit: every other family
    chosen beside it is fit and shown."""
    from turbotab.core.decisions import SetExposureForm
    from turbotab.core.methods.exposure_form import _form_of
    from turbotab.core.voice import listing

    names = dict(zip(named, _who(named, features)))
    formed = [(str(c), *FORM_ENTERS[_form_of(f)[0]]) for c, f in forms.items()]
    if len(formed) == 1:
        (column, label, enters), = formed
        said = (f"{names.get(column, f'`{column}`')} has that role, and its declared form, {label}, "
                f"enters it as {enters}, not one straight-line column")
    else:
        each = "; ".join(f"{names.get(c, f'`{c}`')}, {label}, as {enters}"
                         for c, label, enters in formed)
        said = (f"{listing([c for c, _, _ in formed])} have that role, and their declared forms "
                f"enter each otherwise than as one straight-line column: {each}")
    exits = [{"label": f"Enter `{c}` as a straight line",
              "decision": SetExposureForm(column=c, form="linear").model_dump(mode="json")}
             for c, _, _ in formed]
    return f"{FORMED_WORDS}; {said}.", [*exits, dict(ANOTHER_FAMILY)]


def category_columns(pipeline: Any, categories: Sequence[str] | None = None
                     ) -> dict[str, list[str]]:
    """Each category among ``categories`` (every one the pipeline encodes when None) with the
    model-matrix columns its own encoding made of it, read from the fitted encoder, never from a
    name (the orchestrator's ruling (3) on the swarm's tail): the levels step's indicators (blanks a
    level of their own, ``MissingLevelEncoder``), or the one-hot step's, its levels in order with
    the first dropped where the coding drops it. A name-prefix rule gave `grp` the intercepts of a
    fixed-effects grouping `grp_site` and the spline terms of a covariate `grp_age`; the encoder
    gives each its own. A category the pipeline does not encode is not listed."""
    steps = dict(getattr(pipeline, "named_steps", None) or {})
    wanted = None if categories is None else {str(c) for c in categories}
    found: dict[str, list[str]] = {}
    levels = steps.get("levels")
    for column in (getattr(levels, "levels_", None) or {}):
        if wanted is None or str(column) in wanted:
            found[str(column)] = [str(name) for name, _ in levels._outputs(column)]
    onehot = steps.get("onehot")
    encoder = (getattr(onehot, "named_transformers_", {}) or {}).get("onehot") if onehot else None
    if encoder is not None and hasattr(encoder, "categories_"):
        names = [str(n) for n in encoder.get_feature_names_out()]
        drops = getattr(encoder, "drop_idx_", None)
        counts = [len(levels_i) - (0 if drops is None or drops[i] is None else 1)
                  for i, levels_i in enumerate(encoder.categories_)]
        if sum(counts) != len(names):
            raise ValueError("The one-hot step's columns do not follow its levels, so a category's "
                             "indicators cannot be read from it.")
        at = 0
        for column, k in zip((str(c) for c in encoder.feature_names_in_), counts):
            if wanted is None or column in wanted:
                found[column] = names[at:at + k]
            at += k
    return found


def nothing_left(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """For each column of ``X``, whether nothing of it is left once ``span(Q)`` is projected out
    (a constant, or a combination of the adjustment columns): its residual sum of squares at most
    1e-12 of its own, the tolerance :func:`classical_tests` holds a regressor to."""
    resid = X - Q @ (Q.T @ X)
    return np.sum(resid ** 2, axis=0) <= 1e-12 * np.maximum(np.sum(X ** 2, axis=0), 1e-300)


def not_estimable(named: Sequence[str], features: Sequence[str], adjust: Sequence[str]) -> str:
    """The sentence of the study factors with nothing left once the adjustment columns
    (``adjust``) are projected out (:func:`nothing_left`): not estimable given the covariates, so
    not tested, each named as the roles answer named it (``named`` beside ``features``, the
    matrix's names), the adjustment columns named too (the orchestrator's ruling (4))."""
    from turbotab.core.voice import listing

    one = len(named) == 1
    k = len(adjust)
    span = (f"the {k:,} adjustment column{'s' if k != 1 else ''} ({listing(list(adjust), limit=6)}) "
            f"{'is' if k == 1 else 'are'} projected out" if k else "the intercept is projected out")
    return (f"{listing(_who(named, features), limit=len(named), ticked=False)} "
            f"{'is' if one else 'are'} not estimable given the covariates: nothing of "
            f"{'it' if one else 'each'} is left once {span}, so {'it is not' if one else 'none is'} "
            f"tested.")


def not_tested(entered: Mapping[str, Sequence[str]] | None, adjust: Sequence[str]) -> str | None:
    """The sentence a table adds where a study factor enters the model matrix as indicators of its
    levels (``entered``: by its column, the columns that carry it), so is not tested: its
    indicators are among the adjustment columns (``adjust``) of each test. None where the matrix
    holds none of them (an unadjusted model of another study factor)."""
    from turbotab.core.voice import listing

    held = set(adjust)
    present = [(c, sum(1 for f in cols if f in held)) for c, cols in (entered or {}).items()]
    present = [(c, k) for c, k in present if k]
    if not present:
        return None
    if len(present) == 1:
        (column, k), = present
        where = (f"its {k:,} indicators are" if k != 1 else "its indicator is")
        return (f"`{column}` also has the role of what you study, but enters the model matrix as "
                f"indicators of its levels, not one column, so it is not tested: {where} among "
                f"the adjustment columns.")
    counts = listing([f"{k:,} of `{c}`" for c, k in present], limit=len(present), ticked=False)
    return (f"{listing([c for c, _ in present], limit=len(present))} also have the role of what "
            f"you study, but each enters the model matrix as indicators of its levels, not one "
            f"column, so {'neither' if len(present) == 2 else 'none'} is tested: their indicators "
            f"({counts}) are among the adjustment columns.")


_NOT_TESTED_SAID = re.compile(r"((?:`[^`]+`(?:, | and )?)+) also ha(?:s|ve) the role of what you "
                              r"study, but (?:each )?enters the model matrix as indicators of its "
                              r"levels")


def categories_said(caption: str) -> list[str]:
    """The study factors a table's caption says enter the model matrix as indicators of their
    levels, so are not tested (the sentence :func:`not_tested` writes), in its order: what the
    fit's artifact entry records of them, read back where only that entry is at hand (the served
    fit's family sentence, ``estimand.annotate_fit``)."""
    found = _NOT_TESTED_SAID.search(str(caption or ""))
    return re.findall(r"`([^`]+)`", found.group(1)) if found else []


def bh_adjust(p: Sequence[float]) -> np.ndarray:
    """Benjamini & Hochberg (1995) adjusted p-values: ``q_(i) = min_{k ≥ i} m p_(k) / k``, capped at
    1; a missing p stays missing and does not count among the m tests."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if not m:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def _basis(Z: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the column space of ``Z`` (rank-revealing: collinear columns add
    nothing)."""
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    tol = s.max() * max(Z.shape) * np.finfo(float).eps if s.size else 0.0
    return U[:, s > tol]


def split_columns(columns: Sequence[str], features: Sequence[str] | None) -> tuple[list[str], list[str]]:
    """(tested, adjustment): the exposures among ``columns``, and every other column but a tested
    exposure's own missing indicator."""
    names = [str(c) for c in columns]
    if features is None:
        return names, []
    wanted = set(features)
    tested = [c for c in names if c in wanted]
    mine = set(tested)
    adjust = [c for c in names if c not in mine
              and not (c.startswith(INDICATOR) and c[len(INDICATOR):] in mine)]
    return tested, adjust


def energy_names(columns: Sequence[str], spec: Any) -> list[str]:
    """``columns`` as the model matrix holds them after the design's energy step: a nutrient it
    adjusts under the name it gives it (the residual methods' ``{column}_adj``, the densities'
    ``{column}_per_{energy}``, a partition's ``kcal_from_{column}``), any other column as it is.
    Tested under its raw name, an adjusted study factor matched no column, and the fit refused
    it as no study factor to test (the swarm's C2 repair)."""
    from turbotab.core.methods.energy import _form, _matrix_name

    adjustment = spec.energy_adjustment()
    if adjustment is None:
        return [str(c) for c in columns]
    E = adjustment.energy_column
    form = _form(adjustment.method, bool(E and E in spec.predictors))
    adjusted = list(adjustment.nutrients or []) if form not in ("none", "standard") else []
    return [_matrix_name(str(c), form, adjusted, E) for c in columns]


def event_indicator(y: Any) -> tuple[np.ndarray, Any, Any]:
    """(0/1 event indicator, event level, other level) of a two-level outcome: 1 where the outcome
    is 1 when it is coded 0/1 (the fit stage codes the named event 1; the other level is then
    None, its name unknown here), else its last level."""
    values = pd.Series(np.asarray(y, dtype=object))
    levels = sorted(values.dropna().unique().tolist(), key=lambda v: str(v))
    if set(levels) <= {0, 1, 0.0, 1.0, True, False} and len(levels) == 2:
        return values.astype(float).to_numpy(), 1, None  # coded: the other level's name is not here
    if len(levels) != 2:
        raise ValueError(f"A two-level outcome needs two levels; this one has {len(levels)}.")
    return (values == levels[1]).to_numpy(dtype=float), levels[1], levels[0]


def classical_tests(response: np.ndarray, regressor: np.ndarray, Q: np.ndarray, df: float, *,
                    response_is_matrix: bool) -> tuple[np.ndarray, np.ndarray]:
    """(estimates, standard errors) of the regressor's coefficient in ``response ~ regressor +
    span(Q)``, one test per column, by Frisch–Waugh–Lovell.

    ``response_is_matrix``: the responses are the columns of ``response`` (n × m) and ``regressor``
    is one vector (the event); otherwise ``response`` is one vector (the outcome) and the regressors
    are the columns of ``regressor``. With the residualized regressor ``r`` and the full model's
    residuals ``e``: ``β = Σ r_i ỹ_i / Σ r_i²`` and ``Var = (Σ e_i² / df) / Σ r_i²``, the
    least-squares coefficient and its classical variance, ``df = n − rank[1, Z] − 1``.
    """
    if response_is_matrix:
        g = regressor - Q @ (Q.T @ regressor)
        sgg = float(g @ g)
        Y = response - Q @ (Q.T @ response)
        if sgg <= 0:
            return np.full(Y.shape[1], np.nan), np.full(Y.shape[1], np.nan)
        beta = (g @ Y) / sgg
        e = Y - np.outer(g, beta)
        return beta, np.sqrt(np.sum(e ** 2, axis=0) / df / sgg)
    X = regressor - Q @ (Q.T @ regressor)
    yt = response - Q @ (Q.T @ response)
    sxx = np.sum(X ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (yt @ X) / sxx
        e = yt[:, None] - X * beta
        var = np.sum(e ** 2, axis=0) / df / sxx
    # A column with nothing left once Z is projected out (constant, or a copy of a covariate)
    # cannot be tested: no estimate, rather than a number from rounding error.
    flat = sxx <= 1e-12 * np.maximum(np.sum(regressor ** 2, axis=0), 1e-300)
    return np.where(flat, np.nan, beta), np.sqrt(np.where(flat, np.nan, var))


def _p_and_ci(est: np.ndarray, se: np.ndarray, df: np.ndarray) -> tuple[np.ndarray, ...]:
    from scipy import stats

    ok = np.isfinite(est) & np.isfinite(se) & (se > 0) & np.isfinite(df) & (df > 0)
    p = np.full(est.shape, np.nan)
    lo = np.full(est.shape, np.nan)
    hi = np.full(est.shape, np.nan)
    if ok.any():
        t = est[ok] / se[ok]
        p[ok] = 2.0 * stats.t.sf(np.abs(t), df[ok])
        q = stats.t.ppf(0.5 + LEVEL / 2, df[ok])
        lo[ok] = est[ok] - q * se[ok]
        hi[ok] = est[ok] + q * se[ok]
    return p, lo, hi


def _clean(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def featurewise_table(matrix: pd.DataFrame, y: Any, task: str, features: Sequence[str] | None,
                      clusters: Any = None, event: str | None = None, *,
                      named: Sequence[str] | None = None,
                      forms: Mapping[str, Any] | None = None,
                      entered: Mapping[str, Sequence[str]] | None = None) -> Any:
    """The feature-wise inference table (an ``InferenceTable``) on the model matrix. ``named``:
    the columns the roles answer gave the role, beside ``features`` (the names the matrix holds
    them under), for the words of a refusal; ``forms``: the study factors whose declared form is
    not a straight line, by their ``named`` column, for which no table is made
    (:func:`formed_study_factor`). Where the matrix holds no study factor the table is refused the
    same way (:func:`no_study_factor`). A refused table with no test records the rows it was
    handed (``n_rows``), so no stage adds that it was estimated from them: nothing was.
    ``entered``: the study factors that enter the matrix as indicators of their levels, each with
    the columns its own encoding made of it (:func:`category_columns`); those it holds are
    adjustment columns, and the caption says so (:func:`not_tested`, the verifier's T2V-3). A study
    factor with nothing left once the adjustment columns are projected out is not estimable given
    them (:func:`not_estimable`): its row holds no number and the caption counts only the study
    factors tested, and where none is left to test the table is refused, naming each (the
    orchestrator's ruling (4))."""
    from turbotab.core.models.inference import (INDEPENDENT, InferenceTable, _info, floor_refusal,
                                                format_p)

    clusters = INDEPENDENT if clusters is None else clusters
    estimator = "feature-wise least squares"
    tested, adjust = split_columns(list(matrix.columns), features)
    if forms or not tested:
        sought = list(features or [])
        reason, exits = (formed_study_factor(forms, list(named) if named is not None else sought,
                                             sought) if forms else no_study_factor(features, named))
        info = _info(estimator, "none", f"No tests: {reason}", clusters, refused=reason,
                     exits=exits, n_rows=int(len(matrix)))
        return InferenceTable([], info, [reason])
    values = matrix.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError("The model matrix has missing values; feature-wise tests need complete "
                         "rows (choose complete cases or imputation).")
    n = len(matrix)
    Z = np.column_stack([np.ones(n), matrix[adjust].to_numpy(dtype=float)]) if adjust else np.ones((n, 1))
    Q = _basis(Z)
    df_resid = float(n - Q.shape[1] - 1)
    X = matrix[tested].to_numpy(dtype=float)
    binary = task == "binary"
    if binary:
        g, level, other = event_indicator(y)
        against = f"`{other}`" if other is not None else "the other level"
        estimand = f"the difference in each study factor's mean between `{event or level}` and {against}"
    elif task == "regression":
        yv = np.asarray(y, dtype=float)
        estimand = "the change in the outcome per unit of each study factor"
    else:
        raise ValueError("Feature-wise regression models a numeric or a two-level outcome.")
    m = len(tested)
    refusal = floor_refusal(clusters)
    if refusal is None and clusters.clustered and m > CR2_MAX_FEATURES:
        refusal = (f"`{clusters.column}` repeats, and cluster-robust intervals for {m:,} separate "
                   f"tests are beyond what TurboTab computes ({CR2_MAX_FEATURES:,} at most); "
                   f"intervals that treated the rows as independent would be too narrow, so none "
                   f"is reported.",
                   ({"label": f"Combine each `{clusters.column}`'s rows into one (the unit question, "
                              f"before the seal)", "decision": None},))
    est = np.full(m, np.nan)
    se = np.full(m, np.nan)
    df = np.full(m, df_resid)
    flat = np.zeros(m, dtype=bool)  # a study factor with nothing left once Z is projected out
    if refusal is None and df_resid < 1:
        refusal = (f"{n:,} rows leave no residual degrees of freedom after {Q.shape[1]:,} adjustment "
                   f"columns, so no test can be made.", ())
    if not clusters.clustered or refusal is not None:
        step = max(1, _BLOCK_FLOATS // max(n, 1))
        for lo in range(0, m, step):
            block = slice(lo, lo + step)
            if refusal is not None and df_resid < 1:
                break  # no residual degrees of freedom: nothing can be estimated
            flat[block] = nothing_left(X[:, block], Q)
            if binary:
                est[block], se[block] = classical_tests(X[:, block], g, Q, df_resid,
                                                        response_is_matrix=True)
            else:
                est[block], se[block] = classical_tests(yv, X[:, block], Q, df_resid,
                                                        response_is_matrix=False)
    else:
        flat = nothing_left(X, Q)
        est, se, df = _cr2_tests(X, g if binary else yv, Z, clusters.codes, binary)
    # No estimate of a study factor with nothing left (a copy of the covariates' span): the
    # numbers a fit gives it are rounding error, and under the limma design its "difference" is 0
    # with no variance. It is not estimable given the covariates (the orchestrator's ruling (4)).
    est, se = np.where(flat, np.nan, est), np.where(flat, np.nan, se)
    concerns = [clusters.note] if clusters.note else []
    if refusal is not None:
        reason, exits = refusal
        rows = [{"feature": f, "estimate": _clean(b), "ci_low": None, "ci_high": None, "p": None,
                 "se": None, "df": None, "q": None} for f, b in zip(tested, est)]
        said = not_tested(entered, adjust)
        info = _info(estimator, "none", f"No intervals: {reason}" + (f" {said}" if said else ""),
                     clusters, refused=reason, exits=[dict(e) for e in exits])
        return InferenceTable(rows, info, [reason, *concerns])
    who = dict(zip([str(f) for f in features or []],
                   [str(c) for c in named] if named is not None
                   and len(named) == len(features or []) else [str(f) for f in features or []]))
    left = [f for f, none in zip(tested, flat) if none]
    unestimable = (not_estimable([who.get(f, f) for f in left], left, adjust) if left else None)
    if left and len(left) == m:
        # Nothing to test: the table is refused, naming each study factor and the covariates.
        info = _info(estimator, "none", f"No tests: {unestimable}", clusters, refused=unestimable,
                     exits=[dict(ADJUSTMENT_WAY), dict(ANOTHER_FAMILY)], n_rows=int(n))
        return InferenceTable([], info, [unestimable, *concerns])
    p, low, high = _p_and_ci(est, se, df)
    q = bh_adjust(p)
    found = int(np.sum(q < FDR))
    m = m - len(left)  # the tests made: a study factor not estimable is not one of them
    rows = [{"feature": f, "estimate": _clean(b), "ci_low": _clean(a), "ci_high": _clean(c),
             "p": _clean(pv), "se": _clean(s), "df": _clean(d), "q": _clean(qv)}
            for f, b, a, c, pv, s, d, qv in zip(tested, est, low, high, p, se, df, q)]
    n_adjust = Q.shape[1] - 1
    adjusted = (f"with {n_adjust:,} adjustment column{'s' if n_adjust != 1 else ''}" if n_adjust
                else "with no adjustment columns")
    if n_adjust:
        estimand += ", adjusted for them"
    if clusters.clustered:
        how = (f"cluster-robust (CR2) by `{clusters.column}`, G = {clusters.n_clusters:,}, on t with "
               f"Bell–McCaffrey degrees of freedom")
        covariance = "CR2"
    else:
        how = (f"from each test's own residual variance (classical least squares, exact under "
               f"the null for normal errors), on t({df_resid:,.0f})")
        covariance = "model"
    caption = (f"Each of {m:,} study factors tested on its own {adjusted}; the estimate is {estimand}. "
               f"95% intervals {how}. Benjamini–Hochberg q < {FDR:g}: {found:,} of {m:,}.")
    if unestimable:
        caption = f"{caption} {unestimable}"
    said = not_tested(entered, adjust)
    if said:
        caption = f"{caption} {said}"
    smallest = np.nanmin(q) if np.isfinite(q).any() else float("nan")
    if found == 0 and np.isfinite(smallest):
        concerns.append(f"No study factor passes the false-discovery threshold (smallest q = "
                        f"{format_p(float(smallest))}): {m:,} tests at p < 0.05 would give about "
                        f"{0.05 * m:,.0f} false positives by chance alone.")
    return InferenceTable(rows, _info(estimator, covariance, caption, clusters), concerns)


def _cr2_tests(X: np.ndarray, other: np.ndarray, Z: np.ndarray, codes: np.ndarray,
               binary: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One CR2 fit per exposure, by the math layer's estimator (``inference.cr2``)."""
    from turbotab.core.models.inference import bread_of, cr2

    m = X.shape[1]
    est, se, df = np.full(m, np.nan), np.full(m, np.nan), np.full(m, np.nan)
    for j in range(m):
        if binary:
            D, resp = np.column_stack([Z, other]), X[:, j]
        else:
            D, resp = np.column_stack([Z, X[:, j]]), other
        bread = bread_of(D)
        beta = bread @ (D.T @ resp)
        e = resp - D @ beta
        V, dfs = cr2(D, e, codes, bread)
        est[j] = beta[-1]
        se[j] = float(np.sqrt(max(V[-1, -1], 0.0)))
        df[j] = float(dfs[-1]) if dfs is not None else float(int(codes.max()) + 1 - 1)
    return est, se, df


class FeatureWiseTests(BaseEstimator):
    """The model step: fitting it computes the feature-wise table on independent rows (the fit
    stage asks the family for the clustered one). It makes no predictions.

    ``features``: the study factors as the model matrix holds them; ``named``: the same, as the
    roles answer named them; ``forms``: those whose declared form is not a straight line, by
    their ``named`` column, whose table is refused (:func:`formed_study_factor`); ``categories``:
    those that enter the matrix as indicators of their levels. Their columns are the encoder's,
    which this step does not see: the family reads them off the fitted pipeline
    (:meth:`FeatureWise.inference`, :func:`category_columns`), and a step that holds them already
    (the effects stage's, ``entered_``) hands them on."""

    # A step pickled before ``named``, ``forms`` and ``categories`` reads them as given none.
    named: Sequence[str] | None = None
    forms: Mapping[str, Any] | None = None
    categories: Sequence[str] | None = None
    entered_: Mapping[str, Sequence[str]] | None = None

    def __init__(self, task: str = "regression", features: Sequence[str] | None = None,
                 named: Sequence[str] | None = None, forms: Mapping[str, Any] | None = None,
                 categories: Sequence[str] | None = None):
        self.task = task
        self.features = features
        self.named = named
        self.forms = forms
        self.categories = categories

    def fit(self, X: pd.DataFrame, y: Any) -> "FeatureWiseTests":
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray([str(c) for c in frame.columns], dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.tested_, self.adjust_ = split_columns(list(frame.columns), self.features)
        self.table_ = featurewise_table(frame, y, self.task, self.features, named=self.named,
                                        forms=self.forms)
        if self.task == "binary":
            _, event, other = event_indicator(y)
            self.classes_ = np.asarray([0 if other is None else other, event], dtype=object)
        return self


class FeatureWise(FamilyBase):
    key = "featurewise"
    label = "Feature-wise regression"
    tasks = ("regression", "binary")
    purposes = ("inference",)
    predicts = False
    linear_in_values = True
    inductive_bias = ("Each study factor tested on its own with the covariates; Benjamini–Hochberg "
                      "holds the false-discovery rate.")
    strengths = (
        "Valid tests with more features than rows.",
        "Holds the false-discovery rate across thousands of features (Benjamini–Hochberg).",
    )
    cautions = (
        "Each estimate ignores the other study factors it tests: one separate question per "
        "feature; a study factor that is a category is adjusted for, not tested.",
        "Tests only: it makes no predictions and has no cross-validated score.",
        "Intervals assume equal residual variance; robust ones fail at the thresholds it tests.",
    )
    needs_scaling = False
    handles_missing = False
    # Its tests run over more columns than an imputation model holds, so it pools no multiple
    # imputations: under that answer the fit holds its table with the ways forward (MS7 repair).
    pools_imputations = False
    # MODEL_FAMILY_CONTRACT §1 (§3.1's row for it). It makes no predictions, so its invariances,
    # raw scale, attribution and curve shape are not applicable (C3, C5, C10), and so is Harrell's
    # bootstrap, which has no score of its to correct: the fit skips the loop for it (C11).
    identity = Identity(kind="estimator", library="turbotab", estimator="FeatureWiseTests")
    flexible = False
    bootstrap_optimism = None
    inference_decl = InferenceDecl(table="intervals", intervals=("model", "CR2"),
                                   matrix_table=True, words="feature-wise regression")
    review_lenses = ("metabolomics", "genomics")
    consequence = ("Tests each factor on its own, adjusted for the covariates, with "
                   "Benjamini–Hochberg false-discovery control; no predictions.")
    card_label = "Feature-wise tests"

    def methods_label(self, task: Task | None) -> str:
        # WP11: one least-squares test per exposure, q-values by Benjamini–Hochberg
        return "feature-wise least-squares tests with Benjamini–Hochberg false-discovery control"

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        return FeatureWiseTests(task=task, features=None)

    def build_for(self, spec: Any, task: Task, purpose: Purpose | None, n_rows: int,
                  n_features: int) -> Any:
        """The model step for this design: the exposures are tested, everything else adjusts. An
        exposure the energy step adjusts is tested under the name that step gives it
        (:func:`energy_names`), the column the model matrix holds for it; where one's declared
        form is not a straight line the table is refused (:func:`formed_study_factor`). One that is
        a category enters as indicators of its levels (the shared steps' levels or one-hot step),
        so is not tested, and its table says so (:func:`not_tested`)."""
        from turbotab.core.methods.exposure_form import _form_of

        exposures = [c for c in spec.predictors if spec.roles.get(c) == "exposure"]
        declared = getattr(spec, "exposure_forms", None) or {}
        forms = {c: declared[c] for c in exposures
                 if c in declared and _form_of(declared[c])[0] != "linear"}
        coded = set(spec.categorical or []) | set(getattr(spec, "levels", None) or [])
        categories = [c for c in exposures if c in coded and c not in forms]
        return FeatureWiseTests(task=task, features=energy_names(exposures, spec),
                                named=exposures, forms=forms or None,
                                categories=categories or None)

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        if task == "binary":
            what = ("Regresses each study factor on the event and the covariates (the limma design): "
                    "the estimate is the adjusted difference in its mean.")
        else:
            what = ("Regresses the outcome on each study factor and the covariates, one study factor at a "
                    "time: the estimate is per unit of what you study.")
        return "Feature-wise least squares", (f"{what} Classical least-squares intervals, CR2 when "
                                              f"a unit's rows repeat; Benjamini–Hochberg q-values "
                                              f"across the study factors.")

    def _matrix(self, pipeline: Any, X: Any) -> pd.DataFrame:
        from turbotab.core.models.linear import model_matrix

        return model_matrix(pipeline, X)

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        from turbotab.core.models.linear import as_clusters

        return self.inference(pipeline, X, y, task=task, clusters=as_clusters(groups)).rows

    def inference(self, pipeline: Any, X: Any, y: Any, *, task: Task, clusters: Any,
                  event: str | None = None) -> Any:
        """The feature-wise table on the matrix the model step saw, clustered when rows repeat. A
        study factor that is a category enters with the columns the pipeline's own encoder made of
        it (:func:`category_columns`); a pipeline of the model step alone (the effects stage's, on
        a matrix) hands them on from the step (``entered_``)."""
        model = pipeline[-1]
        entered = getattr(model, "entered_", None)
        if len(getattr(pipeline, "steps", None) or ()) > 1 and getattr(model, "categories", None):
            entered = category_columns(pipeline, model.categories)
        return featurewise_table(self._matrix(pipeline, X), y, task,
                                 getattr(model, "features", None), clusters, event,
                                 named=getattr(model, "named", None),
                                 forms=getattr(model, "forms", None), entered=entered)

    def assess(self, s: Situation) -> Assessment:
        if s.purpose != "inference":
            return Assessment(0.0, "poor", ("Tests each study factor on its own and makes no "
                                            "predictions, so it adds nothing to a prediction.",))
        lenses = tuple(getattr(s, "lenses", ()) or ())
        omics = any(lens in OMICS for lens in lenses)
        if omics or s.n_features >= s.n_rows:
            # AUDIT_REPORT ME-18: PLS-DA is not on the shelf, and the one line says why.
            pls = (PLS_DA_ABSENT,) if "metabolomics" in lenses else ()
            return Assessment(3.5, "good", pls)
        return Assessment(2.0, "fair", ("Each study factor it tests is adjusted for the covariates but not "
                                        "for the other study factors it tests: a separate question "
                                        "for each.",))


FEATUREWISE = register_family(FeatureWise())


def _has_a_study_factor(decision: Any, ctx: Any) -> None:
    """The models answer that chooses feature-wise regression where no column has the role of what
    you study, once the roles are answered: it would have nothing to test, so it is refused here
    in the words the fit refuses its table in (:func:`no_study_factor`), its ways forward the
    roles question and the other families chosen beside it. A role not yet confirmed counts: the
    fit reads the roles as they stand when it is pressed, and refuses the table there where none
    is left."""
    from turbotab.core import decisions as d

    if FEATUREWISE.key not in (decision.models or []):
        return
    state = d._state(ctx)
    if state is None or getattr(state, "roles", None) is None:
        return
    target = getattr(state, "target", None)
    roles = [r for c, r in state.roles.items() if c != target]
    roles += [getattr(sc, "role", None) for sc in getattr(state, "scales", None) or []]
    if "exposure" in roles:
        return
    others = [m for m in decision.models if m != FEATUREWISE.key]
    keep = ({"label": (f"Fit the other {'family' if len(others) == 1 else 'families'} chosen, "
                       f"without feature-wise regression"),
             "decision": decision.model_copy(update={"models": others})}
            if others else ANOTHER_FAMILY)
    raise d.Refusal(NO_STUDY_FACTOR, f"{NO_STUDY_FACTOR_WORDS}; no column has that role.",
                    exits=[ROLES_WAY, keep])


def _register() -> None:
    from turbotab.core.decisions import register_validator

    register_validator("select_models", _has_a_study_factor)


_register()

__all__ = ["CR2_MAX_FEATURES", "FDR", "FEATUREWISE", "FORMED_STUDY_FACTOR", "FeatureWise",
           "FeatureWiseTests", "bh_adjust", "categories_said", "category_columns", "classical_tests",
           "energy_names", "event_indicator", "featurewise_table", "formed_study_factor",
           "no_study_factor", "not_estimable", "not_tested", "nothing_left", "split_columns"]
