"""Design-based estimation for complex surveys (AUDIT_REPORT §5 WP10: ME-06, and the minor D20/G20).

Under inference, when the user answers that the estimates describe the surveyed population
(``set_survey``, ``estimand = "population"``), the linear family's coefficient table is made here
rather than in :mod:`turbotab.core.models.inference`:

* **The estimate** solves the survey-weighted estimating equations ``Σ_i w_i s_i(β) = 0`` over the
  analysis rows (the *domain*), ``s_i`` each row's score: weighted least squares for a continuous
  outcome, weighted (pseudo-)maximum likelihood for a logistic or multinomial one (Binder 1983,
  *Int Stat Rev* 51:279).
* **The variance** is Taylor linearization (the sandwich ``D V̂{Ĝ(β)} D′``): ``D`` the inverse of
  the weighted information, and ``V̂{Ĝ}`` the design-based variance of a total, here the total of
  the weighted scores ``u_i = w_i s_i``, from the spread of PSU totals within strata:
  ``Σ_h n_h/(n_h − 1) Σ_j (z_hj − z̄_h)(z_hj − z̄_h)′``, ``z_hj`` the sum of ``u_i`` over PSU j of
  stratum h. PSUs are taken as drawn with replacement at the first stage (no finite-population
  correction), as NCHS's masked variance units are meant to be used. This is Stata's
  ``vce(linearized)`` ([SVY] *Variance estimation*, equation (1) and "Linearized/robust variance
  estimation"), R's ``survey::svyglm`` and SUDAAN's Taylor series option.
* **Domains, not deletions.** Every row of the working table keeps its stratum and PSU in the
  variance; rows outside the analysis (an eligibility rule, a missing value, a zero weight, a
  held-out row) contribute a score of zero. NHANES Analytic Guidelines 2011–2016 §3.2.3.1: "the
  entire set of data containing the appropriate weights for a particular survey cycle must be used
  to obtain the correct variance estimates. The estimation procedure must indicate which records
  are in the subgroup of interest."
* **Degrees of freedom** are the number of PSUs minus the number of strata, counting only those
  that hold analysis rows (§3.2.3.2: "If an analysis is performed on a subgroup of cases, the
  degrees of freedom should be based on the number of strata and PSUs containing the observations
  of interest"; Stata's ``d = n − L`` and its ``dofsubpop``). Intervals are ``β̂ ± t(d) SE``.
* **A stratum with a single PSU** has no spread of its own. Its PSU total is centered at the mean
  of all PSU totals, with ``n_h/(n_h − 1)`` taken as 1: R's ``options(survey.lonely.psu =
  "adjust")`` ("center the stratum at the population mean rather than the stratum mean") and
  Stata's ``singleunit(centered)``. It is conservative, and the table names every such stratum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from turbotab.core.models.inference import (
    InferenceTable,
    _and,
    _named,
    _row,
    _t_rows,
    separated_columns,
)

# NCHS Data Presentation Standards, via the Analytic Guidelines 2011–2016 §3.3.3: estimates on
# "fewer than 8 degrees of freedom be reviewed by a clearance official".
FEW_DESIGN_DF = 8
LONELY_METHOD = "centered"  # R survey.lonely.psu = "adjust"; Stata singleunit(centered)
_NEWTON_TOL = 1e-10
_NEWTON_MAX = 100


# ── the design ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SurveyDesign:
    """A survey design over every row of the working table, indexed by row id.

    ``weight`` is the analysis weight of each row (NaN where it has none), already built for pooled
    cycles (:func:`turbotab.core.methods.survey.analysis_weights`). ``stratum`` and ``psu`` are
    integer codes; a PSU code is unique across the whole design (PSUs nested in strata, as NHANES
    reuses ``SDMVPSU`` = 1, 2 in every stratum). Rows that cannot be placed (no stratum or PSU)
    are not in the design at all.
    """

    row_ids: np.ndarray
    weight: np.ndarray
    stratum: np.ndarray
    psu: np.ndarray
    weight_column: str | None
    strata_column: str | None
    psu_column: str | None
    weight_note: str | None = None  # how the weight was built, when cycles were pooled
    psu_note: str | None = None  # each row its own PSU, or the unit's rows one PSU
    n_unplaced: int = 0  # rows of the table with no stratum or PSU: outside the design
    stratum_labels: tuple[Any, ...] = ()  # code -> the stratum's own value

    def __post_init__(self) -> None:
        n = len(self.row_ids)
        if not (len(self.weight) == len(self.stratum) == len(self.psu) == n):
            raise ValueError("A survey design needs a weight, stratum and PSU for every row.")

    @property
    def n_rows(self) -> int:
        return int(len(self.row_ids))

    @property
    def n_psu(self) -> int:
        return int(len(np.unique(self.psu))) if len(self.psu) else 0

    @property
    def n_strata(self) -> int:
        return int(len(np.unique(self.stratum))) if len(self.stratum) else 0

    def lonely_strata(self) -> list[Any]:
        """The strata of the design with a single PSU, by their own values."""
        if not len(self.psu):
            return []
        per = pd.Series(self.psu).groupby(self.stratum).nunique()
        codes = [int(c) for c, k in per.items() if int(k) == 1]
        return [self.stratum_labels[c] if c < len(self.stratum_labels) else c for c in codes]


def build_design(frame: pd.DataFrame, weight: np.ndarray, *, weight_column: str | None,
                 strata_column: str | None, psu_column: str | None,
                 unit_codes: np.ndarray | None = None, unit_column: str | None = None,
                 weight_note: str | None = None) -> SurveyDesign:
    """The design over ``frame``'s rows (indexed by row id), with ``weight`` aligned to them.

    Without a PSU column, each unit is a PSU when ``unit_codes`` says which rows belong together (a
    person's repeated rows), else each row is its own PSU. Without a strata column the design has one
    stratum. A row with a missing stratum or PSU value cannot be placed and leaves the design.
    """
    n = len(frame)
    ids = np.asarray(frame.index, dtype=np.int64)
    placed = np.ones(n, dtype=bool)
    if strata_column is not None:
        strata_values = frame[strata_column]
        placed &= strata_values.notna().to_numpy()
    if psu_column is not None:
        placed &= frame[psu_column].notna().to_numpy()
    if strata_column is not None:
        stratum_codes, stratum_labels = pd.factorize(frame.loc[placed, strata_column], sort=True)
        labels = tuple(_plain(v) for v in stratum_labels)
    else:
        stratum_codes, labels = np.zeros(int(placed.sum()), dtype=np.int64), ("all rows",)
    psu_note = None
    if psu_column is not None:
        keys = pd.MultiIndex.from_arrays([stratum_codes, frame.loc[placed, psu_column].to_numpy()])
        psu_codes = pd.factorize(keys)[0]
    elif unit_codes is not None:
        keys = pd.MultiIndex.from_arrays([stratum_codes, np.asarray(unit_codes)[placed]])
        psu_codes = pd.factorize(keys)[0]
        psu_note = (f"no PSU column: each `{unit_column}`'s rows are one sampling unit"
                    if unit_column else "no PSU column: each unit's rows are one sampling unit")
    else:
        psu_codes = np.arange(int(placed.sum()))
        psu_note = "no PSU column: each row is its own sampling unit"
    return SurveyDesign(
        row_ids=ids[placed], weight=np.asarray(weight, dtype=float)[placed],
        stratum=np.asarray(stratum_codes, dtype=np.int64), psu=np.asarray(psu_codes, dtype=np.int64),
        weight_column=weight_column, strata_column=strata_column, psu_column=psu_column,
        weight_note=weight_note, psu_note=psu_note, n_unplaced=int((~placed).sum()),
        stratum_labels=labels,
    )


def _plain(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return int(f) if f.is_integer() else f
    return value


# ── the variance of a total of scores ────────────────────────────────────────


@dataclass
class DesignVariance:
    """``V̂{Ĝ}``, and what it rests on."""

    meat: np.ndarray  # P × P
    n_psu: int  # every PSU of the design
    n_strata: int
    df: int  # PSUs minus strata, counting those holding domain rows
    domain_psu: int
    domain_strata: int
    lonely: list[Any] = field(default_factory=list)  # strata with one PSU, by their own values


def total_variance(u: np.ndarray, design: SurveyDesign, domain: np.ndarray) -> DesignVariance:
    """The design-based variance of ``Σ_i u_i`` (``u``: design rows × P, zero outside the domain).

    ``Σ_h c_h Σ_j (z_hj − m_h)(z_hj − m_h)′`` over every PSU of the design: ``z_hj`` the PSU's
    total, ``m_h`` the stratum's mean PSU total and ``c_h = n_h/(n_h − 1)``; for a stratum with one
    PSU, ``m_h`` is the mean over every PSU of the design and ``c_h = 1`` (:data:`LONELY_METHOD`).
    """
    u = np.asarray(u, dtype=float)
    P = u.shape[1]
    G = int(design.psu.max()) + 1 if len(design.psu) else 0
    totals = np.zeros((G, P))
    for j in range(P):
        totals[:, j] = np.bincount(design.psu, weights=u[:, j], minlength=G)
    stratum_of = np.full(G, -1, dtype=np.int64)
    stratum_of[design.psu] = design.stratum
    H = int(stratum_of.max()) + 1 if G else 0
    per_stratum = np.bincount(stratum_of, minlength=H)
    sums = np.zeros((H, P))
    for j in range(P):
        sums[:, j] = np.bincount(stratum_of, weights=totals[:, j], minlength=H)
    means = sums / np.maximum(per_stratum, 1)[:, None]
    lonely = per_stratum == 1
    center = means[stratum_of]
    if lonely.any():
        grand = totals.mean(axis=0)
        center[lonely[stratum_of]] = grand
    n_h = per_stratum[stratum_of].astype(float)
    scale = np.where(n_h > 1, n_h / np.maximum(n_h - 1, 1), 1.0)
    deviations = totals - center
    meat = (deviations * scale[:, None]).T @ deviations
    in_domain = np.asarray(domain, dtype=bool)
    domain_psu = int(len(np.unique(design.psu[in_domain])))
    domain_strata = int(len(np.unique(design.stratum[in_domain])))
    labels = design.stratum_labels
    lonely_values = [labels[h] if h < len(labels) else h for h in np.flatnonzero(lonely)]
    return DesignVariance(meat=(meat + meat.T) / 2, n_psu=G, n_strata=H,
                          df=domain_psu - domain_strata, domain_psu=domain_psu,
                          domain_strata=domain_strata, lonely=lonely_values)


# ── weighted fits: estimate, scores and information ──────────────────────────


@dataclass
class WeightedFit:
    estimate: np.ndarray  # length P (class by class for a multinomial model)
    scores: np.ndarray  # domain rows × P: w_i s_i(β̂)
    information: np.ndarray  # P × P: Σ_i w_i I_i(β̂)
    converged: bool = True


def weighted_least_squares(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> WeightedFit:
    root = np.sqrt(w)
    beta = np.linalg.lstsq(X * root[:, None], y * root, rcond=None)[0]
    resid = y - X @ beta
    return WeightedFit(beta, X * (w * resid)[:, None], X.T @ (X * w[:, None]))


def weighted_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> WeightedFit:
    """Newton–Raphson on the weighted log-likelihood ``Σ w_i {y_i η_i − log(1 + e^η_i)}``, steps
    halved until it does not fall; converged when the Newton decrement is below the tolerance."""
    P = X.shape[1]
    beta = np.zeros(P)

    def loglik(b: np.ndarray) -> float:
        eta = X @ b
        return float(np.sum(w * (y * eta - np.logaddexp(0.0, eta))))

    current = loglik(beta)
    converged = False
    for _ in range(_NEWTON_MAX):
        mu = 0.5 * (1.0 + np.tanh(0.5 * (X @ beta)))
        score = X.T @ (w * (y - mu))
        info = X.T @ (X * (w * mu * (1.0 - mu))[:, None])
        step = np.linalg.lstsq(info, score, rcond=None)[0]
        if float(score @ step) < _NEWTON_TOL ** 2 * max(1.0, float(np.sum(w))):
            converged = True
            break
        for _half in range(40):
            trial = beta + step
            value = loglik(trial)
            if value >= current - 1e-12 * max(1.0, abs(current)):
                break
            step /= 2.0
        beta, current = trial, value
    mu = 0.5 * (1.0 + np.tanh(0.5 * (X @ beta)))
    info = X.T @ (X * (w * mu * (1.0 - mu))[:, None])
    return WeightedFit(beta, X * (w * (y - mu))[:, None], info, converged)


def _softmax(eta: np.ndarray) -> np.ndarray:
    """Class probabilities for linear predictors ``eta`` (n × q against a reference class 0)."""
    full = np.column_stack([np.zeros(len(eta)), eta])
    full -= full.max(axis=1, keepdims=True)
    e = np.exp(full)
    return e / e.sum(axis=1, keepdims=True)


def weighted_multinomial(X: np.ndarray, codes: np.ndarray, K: int, w: np.ndarray) -> WeightedFit:
    """Newton–Raphson for the weighted multinomial logit against class 0. Coefficients are ordered
    class by class (each class's P together), as :func:`~turbotab.core.models.inference.
    multinomial_working` orders them."""
    n, P = X.shape
    q = K - 1
    Y = (codes[:, None] == np.arange(1, K)[None, :]).astype(float)
    B = np.zeros((P, q))

    def loglik(b: np.ndarray) -> float:
        eta = X @ b
        full = np.column_stack([np.zeros(n), eta])
        return float(np.sum(w * (np.sum(Y * eta, axis=1) - np.logaddexp.reduce(full, axis=1))))

    def parts(b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pi = _softmax(X @ b)[:, 1:]
        score_rows = np.einsum("ia,ip->iap", (Y - pi), X).reshape(n, q * P)  # class-major
        info = np.zeros((q * P, q * P))
        for a in range(q):  # block (a, b) = Σ_i w_i W_i[a, b] x_i x_iᵀ, W_i = diag π_i − π_i π_iᵀ
            for b in range(a, q):
                cell = pi[:, a] * ((a == b) - pi[:, b])
                block = X.T @ (X * (w * cell)[:, None])
                info[a * P:(a + 1) * P, b * P:(b + 1) * P] = block
                info[b * P:(b + 1) * P, a * P:(a + 1) * P] = block.T
        return score_rows, info, pi

    current = loglik(B)
    converged = False
    for _ in range(_NEWTON_MAX):
        rows, info, _ = parts(B)
        score = (rows * w[:, None]).sum(axis=0)
        step = np.linalg.lstsq(info, score, rcond=None)[0]
        if float(score @ step) < _NEWTON_TOL ** 2 * max(1.0, float(np.sum(w))):
            converged = True
            break
        for _half in range(40):
            trial = B + step.reshape(q, P).T
            value = loglik(trial)
            if value >= current - 1e-12 * max(1.0, abs(current)):
                break
            step /= 2.0
        B, current = trial, value
    rows, info, _ = parts(B)
    return WeightedFit(B.T.ravel(), rows * w[:, None], info, converged)


# ── the table ────────────────────────────────────────────────────────────────


def _caption(design: SurveyDesign, var: DesignVariance) -> str:
    weight = f"weights `{design.weight_column}`" if design.weight_column else "equal weights"
    if design.psu_column:
        units = f"{var.n_psu:,} PSUs (`{design.psu_column}`)"
    else:
        units = f"{var.n_psu:,} sampling units ({design.psu_note})"
    strata = (f" in {var.n_strata:,} strata (`{design.strata_column}`)" if design.strata_column
              else " in one stratum")
    domain = ("" if var.domain_psu == var.n_psu and var.domain_strata == var.n_strata else
              f"; {var.domain_psu:,} PSUs in {var.domain_strata:,} strata hold the analysis rows")
    return (f"95% intervals by Taylor linearization over the survey design ({weight}; {units}"
            f"{strata}, drawn with replacement at the first stage{domain}), on t({var.df:,}): PSUs "
            f"minus strata.")


def _info(estimator: str, caption: str, design: SurveyDesign, var: DesignVariance | None,
          n_domain: int, **extra: Any) -> dict[str, Any]:
    survey = {
        "weight": design.weight_column, "strata": design.strata_column, "psu": design.psu_column,
        "weight_note": design.weight_note, "psu_note": design.psu_note,
        "n_design": design.n_rows, "n_domain": int(n_domain),
        "n_psu": var.n_psu if var else design.n_psu, "n_strata": var.n_strata if var else design.n_strata,
        "domain_psu": var.domain_psu if var else None, "domain_strata": var.domain_strata if var else None,
        "df": var.df if var else None,
        "lonely_strata": [str(s) for s in (var.lonely if var else design.lonely_strata())],
        "lonely_method": LONELY_METHOD,
    }
    return {"estimator": estimator, "covariance": "design", "caption": caption,
            "grouped_by": None, "n_clusters": None, "n_missing_ids": 0, "separated": [],
            "refused": None, "exits": [], "survey": survey, **extra}


def _refused(names: Sequence[str], est: np.ndarray | None, estimator: str, design: SurveyDesign,
             n_domain: int, reason: str, exits: Sequence[dict[str, Any]] = ()) -> InferenceTable:
    rows = [_row(n, b) for n, b in zip(names, est)] if est is not None else []
    info = _info(estimator, f"No intervals: {reason}", design, None, n_domain,
                 refused=reason, exits=[dict(e) for e in exits])
    info["covariance"] = "none"
    return InferenceTable(rows, info, [reason])


def blocked(reason: str, exits: Sequence[dict[str, Any]] = (),
            estimator: str = "not fitted: the survey question decides it") -> InferenceTable:
    """No table at all: the survey answer is missing, or its weight cannot be built (block and
    record, BLUEPRINT §11.3). The reason is the table's first concern."""
    info = {"estimator": estimator, "covariance": "none", "caption": f"No coefficients: {reason}",
            "grouped_by": None, "n_clusters": None, "n_missing_ids": 0, "separated": [],
            "refused": reason, "exits": [dict(e) for e in exits], "survey": None}
    return InferenceTable([], info, [reason])


def _concerns(design: SurveyDesign, var: DesignVariance, n_domain: int, n_coef: int,
              left: dict[str, int]) -> list[str]:
    out: list[str] = []
    outside = design.n_rows - n_domain
    if outside > 0:
        out.append(f"A domain analysis: the estimates use {n_domain:,} of the design's "
                   f"{design.n_rows:,} rows, and the other {outside:,} keep their strata and PSUs "
                   f"in the variance (NHANES Analytic Guidelines 2011–2016 §3.2.3).")
    if left.get("unweighted"):
        k = left["unweighted"]
        out.append(f"{k:,} analysis row{'s' if k != 1 else ''} with a zero, negative or missing "
                   f"`{design.weight_column}` represent{'s' if k == 1 else ''} no one in the "
                   f"population and {'is' if k == 1 else 'are'} left out of the estimate.")
    if left.get("unplaced"):
        k = left["unplaced"]
        out.append(f"{k:,} analysis row{'s' if k != 1 else ''} with no stratum or PSU "
                   f"{'is' if k == 1 else 'are'} outside the design and left out of the estimate.")
    if var.lonely:
        listed = _and([str(s) for s in var.lonely[:6]]) + (" and more" if len(var.lonely) > 6 else "")
        k = len(var.lonely)
        out.append(f"{k} {'stratum has' if k == 1 else 'strata have'} a single PSU ({listed}): "
                   f"{'its PSU total is' if k == 1 else 'each PSU total is'} centered at the mean "
                   f"of all PSU totals (R survey's lonely.psu \"adjust\", Stata's "
                   f"singleunit(centered)), which overstates rather than understates the variance.")
    if var.df < FEW_DESIGN_DF:
        out.append(f"Only {var.df} design degrees of freedom ({var.domain_psu} PSUs minus "
                   f"{var.domain_strata} strata): NCHS asks that estimates on fewer than "
                   f"{FEW_DESIGN_DF} be reviewed before they are published.")
    elif n_coef > var.df:
        out.append(f"{n_coef} coefficients rest on {var.df} design degrees of freedom: each "
                   f"interval holds, but no joint test of them all can be made.")
    return out


def survey_table(task: str, matrix: pd.DataFrame, y: Any, classes: Sequence[Any] | None,
                 design: SurveyDesign) -> InferenceTable:
    """The design-based coefficient table for the linear family on the matrix it saw.

    ``matrix`` holds the analysis rows (indexed by row id) and ``y`` their outcome. The domain is
    those of them in the design with a positive weight; every other design row contributes a
    score of zero.
    """
    import statsmodels.api as sm

    exog = sm.add_constant(matrix.astype(float), has_constant="add")
    names = ["(intercept)" if c == "const" else str(c) for c in exog.columns]
    ids = np.asarray(matrix.index, dtype=np.int64)
    position = pd.Index(design.row_ids).get_indexer(ids)
    placed = position >= 0
    weight = np.full(len(ids), np.nan)
    weight[placed] = design.weight[position[placed]]
    weighted = placed & np.isfinite(weight) & (weight > 0)
    left = {"unplaced": int((~placed).sum()), "unweighted": int((placed & ~weighted).sum())}
    X = exog.to_numpy(dtype=float)[weighted]
    yv = np.asarray(y)[weighted]
    n_domain = int(weighted.sum())
    # Weights scaled to average one over the domain: the estimate and the sandwich are unchanged by
    # a common factor, and the fits' tolerances stay on the scale of the rows.
    w = weight[weighted] * (n_domain / float(weight[weighted].sum())) if n_domain else weight[weighted]
    rows_at = position[weighted]
    estimator = {"regression": "survey-weighted least squares",
                 "binary": "survey-weighted logistic regression (pseudo-maximum likelihood)"}.get(
        task, "survey-weighted multinomial logistic regression (pseudo-maximum likelihood)")
    if n_domain <= X.shape[1]:
        return _refused(names, None, estimator, design, n_domain,
                        f"Only {n_domain:,} analysis rows carry a positive weight and a place in "
                        f"the design, too few for {X.shape[1]} coefficients.")
    labels = names
    if task == "regression":
        fit = weighted_least_squares(X, yv.astype(float), w)
    elif task == "binary":
        if classes is None or len(classes) != 2:
            raise ValueError("A binary outcome needs exactly two classes.")
        event = (yv == classes[1]).astype(float)
        separated = _named(names, separated_columns(X, event))
        if separated:
            verb = "separates" if len(separated) == 1 else "separate"
            return _refused(names, None, estimator, design, n_domain,
                            f"{_and(separated)} {verb} the outcome among the weighted rows, so the "
                            f"weighted log-odds is infinite and no design-based interval exists.",
                            ({"label": f"Leave {_and(separated)} out, or merge its rare levels",
                              "decision": None},
                             {"label": "Estimate for these participants instead (the survey "
                                       "question), where Firth's fit is available",
                              "decision": None}))
        fit = weighted_logistic(X, event, w)
    else:
        levels = list(classes or [])
        codes = pd.Categorical(yv, categories=levels).codes.astype(np.int64)
        K = len(levels)
        q = K - 1
        labels = [f"{n} [{levels[k + 1]}]" for k in range(q) for n in names]
        fit = weighted_multinomial(X, codes, K, w)
    u = np.zeros((design.n_rows, fit.scores.shape[1]))
    u[rows_at] = fit.scores
    domain = np.zeros(design.n_rows, dtype=bool)
    domain[rows_at] = True
    var = total_variance(u, design, domain)
    bread = np.linalg.pinv(fit.information)
    V = bread @ var.meat @ bread
    se = np.sqrt(np.clip(np.diag((V + V.T) / 2), 0, None))
    if var.df < 1:
        return _refused(labels, fit.estimate, estimator, design, n_domain,
                        f"The analysis rows lie in {var.domain_psu} PSU"
                        f"{'s' if var.domain_psu != 1 else ''} of {var.domain_strata} "
                        f"strat{'a' if var.domain_strata != 1 else 'um'}: no design degrees of "
                        f"freedom are left for an interval.")
    rows = _t_rows(labels, fit.estimate, se, np.full(len(fit.estimate), float(var.df)))
    concerns = _concerns(design, var, n_domain, len(fit.estimate), left)
    if not fit.converged:
        concerns.append("The weighted fit stopped before converging; treat these numbers with care.")
    table = InferenceTable(rows, _info(estimator, _caption(design, var), design, var, n_domain),
                           concerns)
    return table


def design_df(design: SurveyDesign, domain_ids: Sequence[int]) -> int:
    """PSUs minus strata among the design rows ``domain_ids`` (the NCHS rule), for sentences."""
    at = pd.Index(design.row_ids).get_indexer(np.asarray(domain_ids, dtype=np.int64))
    at = at[at >= 0]
    return int(len(np.unique(design.psu[at])) - len(np.unique(design.stratum[at])))


__all__ = [
    "FEW_DESIGN_DF", "LONELY_METHOD", "DesignVariance", "SurveyDesign", "WeightedFit", "blocked",
    "build_design", "design_df", "survey_table", "total_variance",
    "weighted_least_squares", "weighted_logistic", "weighted_multinomial",
]
