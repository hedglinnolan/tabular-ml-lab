"""Design-based estimation for complex surveys (AUDIT_REPORT §5 WP10: ME-06, and the minor D20/G20;
MODELING_SEQUENCE §0 ruling 6 and §5 MS4).

Under inference, when the user answers that the estimates describe the surveyed population
(``set_survey``, ``estimand = "population"``), every family and every display is design-based, or
it is blocked and recorded (MODELING_SEQUENCE §4, "population estimand without a design-based
estimator"). What is design-based:

* **The linear family** (least squares, logistic, multinomial logistic): :func:`survey_table`.
* **The Cox model**: Binder's (1992, *Int Stat Rev* 60:249) pseudo-likelihood, the partial
  likelihood with each row weighted, Efron's ties as R's ``coxph`` weights them
  (:func:`turbotab.core.models.survival.survey_cox_table`); R's ``survey::svycoxph``.
* **The proportional-odds model**: the weighted cumulative-logit likelihood
  (:func:`turbotab.core.models.ordinal.survey_ordinal_table`); R's ``survey::svyolr``.
* **The substitution curve** of the linear family: refit with the weights, averaged over the
  population, its band by linearization (:func:`design_curve`; Graubard & Korn 1999).
* **Joint tests** (a spline's overall and nonlinear tests) on a design-based covariance: the
  adjusted Wald F (:func:`adjusted_wald`).

**Every estimate** solves the survey-weighted estimating equations ``Σ_i w_i s_i(β) = 0`` over the
analysis rows (the *domain*), ``s_i`` each row's score (Binder 1983, *Int Stat Rev* 51:279).

**The variance** is Taylor linearization (the sandwich ``D V̂{Ĝ(β)} D′``): ``D`` the inverse of the
weighted information, and ``V̂{Ĝ}`` the design-based variance of a total, here the total of the
weighted scores ``u_i = w_i s_i``, from the spread of PSU totals within strata:
``Σ_h n_h/(n_h − 1) Σ_j (z_hj − z̄_h)(z_hj − z̄_h)′``, ``z_hj`` the sum of ``u_i`` over PSU j of
stratum h. PSUs are taken as drawn with replacement at the first stage (no finite-population
correction), as NCHS's masked variance units are meant to be used. This is Stata's
``vce(linearized)`` ([SVY] *Variance estimation*, equation (1) and "Linearized/robust variance
estimation"), R's ``survey::svyrecvar`` and SUDAAN's Taylor series option.

**Domains, not deletions.** Every row of the working table keeps its stratum and PSU in the
variance; rows outside the analysis (an eligibility rule, a missing value, a zero weight, a
held-out row) contribute a score of zero. NHANES Analytic Guidelines 2011–2016 §3.2.3.1: "the
entire set of data containing the appropriate weights for a particular survey cycle must be used
to obtain the correct variance estimates. The estimation procedure must indicate which records are
in the subgroup of interest."

**Degrees of freedom** are the number of PSUs minus the number of strata, counting only those that
hold analysis rows (§3.2.3.2: "If an analysis is performed on a subgroup of cases, the degrees of
freedom should be based on the number of strata and PSUs containing the observations of
interest"; Stata's ``d = n − L`` and its ``dofsubpop``). Intervals are ``β̂ ± t(d) SE``.

**A stratum with a single PSU** (a *lonely* PSU) has no spread of its own. The stated rule is R's
``options(survey.lonely.psu = "adjust")`` ("the data for the single-PSU stratum are centered at the
sample grand mean rather than the stratum mean"), as R's ``survey`` 4.5 computes it
(``onestage``/``onestrat``): the grand mean is the total of the scores over the number of PSUs in
the strata that hold analysis rows, ``n_h/(n_h − 1)`` is taken as 1, and a stratum that holds no
analysis row adds nothing. A stratum whose other PSUs hold no analysis row is not lonely: its
empty PSUs keep their zero totals (R's default ``survey.adjust.domain.lonely = FALSE``). The rule is
conservative, and every table names the strata it applied to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

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
LONELY_PSU = "adjust"  # the R option the rule matches, named in captions and the record
LONELY_RULE = ("centered at the mean PSU total of the strata that hold analysis rows, its "
               "n_h/(n_h − 1) taken as 1 (R survey's lonely.psu \"adjust\")")
_NEWTON_TOL = 1e-10
_NEWTON_MAX = 100
LEVEL = 0.95


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


def _psu_totals(u: np.ndarray, design: SurveyDesign) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(PSU totals G × P, each PSU's stratum, each stratum's number of PSUs in the design)."""
    P = u.shape[1]
    G = int(design.psu.max()) + 1 if len(design.psu) else 0
    totals = np.zeros((G, P))
    for j in range(P):
        totals[:, j] = np.bincount(design.psu, weights=u[:, j], minlength=G)
    stratum_of = np.full(G, -1, dtype=np.int64)
    stratum_of[design.psu] = design.stratum
    H = int(stratum_of.max()) + 1 if G else 0
    return totals, stratum_of, np.bincount(stratum_of, minlength=H)


def _deviations(totals: np.ndarray, stratum_of: np.ndarray, per_stratum: np.ndarray,
                present: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each PSU total less its center, the factor ``n_h/(n_h − 1)`` it is scaled by, and which
    strata are lonely and present (the stated rule; module docstring)."""
    H = len(per_stratum)
    P = totals.shape[1]
    sums = np.zeros((H, P))
    for j in range(P):
        sums[:, j] = np.bincount(stratum_of, weights=totals[:, j], minlength=H)
    means = sums / np.maximum(per_stratum, 1)[:, None]
    lonely = (per_stratum == 1) & present
    center = means[stratum_of]
    if lonely.any():
        # R's recentering: the total over every PSU, divided by the number of PSUs of the strata
        # that hold analysis rows (``onestage``: colSums(x) / sum of nPSU over its strata).
        grand = totals.sum(axis=0) / float(per_stratum[present].sum())
        center[lonely[stratum_of]] = grand
    n_h = per_stratum[stratum_of].astype(float)
    scale = np.where(n_h > 1, n_h / np.maximum(n_h - 1, 1), 1.0)
    deviations = (totals - center) * present[stratum_of][:, None]
    return deviations, scale, lonely


def total_variance(u: np.ndarray, design: SurveyDesign, domain: np.ndarray) -> DesignVariance:
    """The design-based variance of ``Σ_i u_i`` (``u``: design rows × P, zero outside the domain).

    ``Σ_h c_h Σ_j (z_hj − m_h)(z_hj − m_h)′`` over the PSUs of every stratum holding a domain row:
    ``z_hj`` the PSU's total, ``m_h`` the stratum's mean PSU total (its PSUs with no domain row
    count, at zero) and ``c_h = n_h/(n_h − 1)``; a lonely stratum is centered by the stated rule
    (module docstring) with ``c_h = 1``.
    """
    u = np.asarray(u, dtype=float)
    totals, stratum_of, per_stratum = _psu_totals(u, design)
    in_domain = np.asarray(domain, dtype=bool)
    present = np.zeros(len(per_stratum), dtype=bool)
    present[np.unique(design.stratum[in_domain])] = True
    deviations, scale, lonely = _deviations(totals, stratum_of, per_stratum, present)
    meat = (deviations * scale[:, None]).T @ deviations
    domain_psu = int(len(np.unique(design.psu[in_domain])))
    domain_strata = int(len(np.unique(design.stratum[in_domain])))
    labels = design.stratum_labels
    lonely_values = [labels[h] if h < len(labels) else h for h in np.flatnonzero(lonely)]
    return DesignVariance(meat=(meat + meat.T) / 2, n_psu=len(totals), n_strata=len(per_stratum),
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


# ── the domain and the design-based table ────────────────────────────────────


@dataclass
class Domain:
    """Where the analysis rows (indexed by row id) sit in the design.

    ``keep`` marks the analysis rows in the domain: placed in the design with a positive weight.
    ``at`` is each kept row's position among the design's rows, ``weight`` its weight scaled to
    average one over the domain (an estimate and its sandwich do not change with a common factor,
    and the fits' tolerances stay on the scale of the rows), ``raw`` its weight as built.
    """

    keep: np.ndarray
    at: np.ndarray
    weight: np.ndarray
    raw: np.ndarray
    left: dict[str, int]

    @property
    def n(self) -> int:
        return int(self.keep.sum())

    def mask(self, design: SurveyDesign) -> np.ndarray:
        """The domain over the design's rows."""
        out = np.zeros(design.n_rows, dtype=bool)
        out[self.at] = True
        return out


def domain_of(index: Any, design: SurveyDesign) -> Domain:
    """The domain of the analysis rows ``index`` (row ids) in ``design``."""
    ids = np.asarray(index, dtype=np.int64)
    position = pd.Index(design.row_ids).get_indexer(ids)
    placed = position >= 0
    weight = np.full(len(ids), np.nan)
    weight[placed] = design.weight[position[placed]]
    keep = placed & np.isfinite(weight) & (weight > 0)
    raw = weight[keep]
    n = int(keep.sum())
    scaled = raw * (n / float(raw.sum())) if n else raw
    return Domain(keep=keep, at=position[keep], weight=scaled, raw=raw,
                  left={"unplaced": int((~placed).sum()), "unweighted": int((placed & ~keep).sum())})


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


def survey_info(design: SurveyDesign, var: DesignVariance | None, n_domain: int) -> dict[str, Any]:
    """The ``survey`` field of a design-based table's ``inference`` (``SurveyInference``)."""
    return {
        "weight": design.weight_column, "strata": design.strata_column, "psu": design.psu_column,
        "weight_note": design.weight_note, "psu_note": design.psu_note,
        "n_design": design.n_rows, "n_domain": int(n_domain),
        "n_psu": var.n_psu if var else design.n_psu, "n_strata": var.n_strata if var else design.n_strata,
        "domain_psu": var.domain_psu if var else None, "domain_strata": var.domain_strata if var else None,
        "df": var.df if var else None,
        "lonely_strata": [str(s) for s in (var.lonely if var else design.lonely_strata())],
        "lonely_method": LONELY_METHOD,
    }


def _info(estimator: str, caption: str, design: SurveyDesign, var: DesignVariance | None,
          n_domain: int, **extra: Any) -> dict[str, Any]:
    return {"estimator": estimator, "covariance": "design", "caption": caption,
            "grouped_by": None, "n_clusters": None, "n_missing_ids": 0, "separated": [],
            "refused": None, "exits": [], "survey": survey_info(design, var, n_domain), **extra}


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
                   f"{'its PSU total is' if k == 1 else 'each PSU total is'} {LONELY_RULE}, as "
                   f"Stata's singleunit(centered) does, which overstates rather than understates "
                   f"the variance.")
    if var.df < FEW_DESIGN_DF:
        out.append(f"Only {var.df} design degrees of freedom ({var.domain_psu} PSUs minus "
                   f"{var.domain_strata} strata): NCHS asks that estimates on fewer than "
                   f"{FEW_DESIGN_DF} be reviewed before they are published.")
    elif n_coef > var.df:
        out.append(f"{n_coef} coefficients rest on {var.df} design degrees of freedom: each "
                   f"interval holds, but no joint test of them all can be made.")
    return out


def design_table(labels: Sequence[str], estimate: np.ndarray, scores: np.ndarray,
                 information: np.ndarray, design: SurveyDesign, domain: Domain, estimator: str,
                 *, converged: bool = True, bread: np.ndarray | None = None) -> InferenceTable:
    """The design-based table of an M-estimator fit on the domain's rows.

    ``scores`` are the kept rows' weighted scores ``w_i s_i(β̂)`` (domain rows × P), ``information``
    the weighted information ``Σ w_i I_i(β̂)`` (or ``bread``, its inverse, when the fit has it). The
    covariance is the sandwich over the design (module docstring); the intervals are on t(d)."""
    u = np.zeros((design.n_rows, scores.shape[1]))
    u[domain.at] = scores
    var = total_variance(u, design, domain.mask(design))
    B = np.linalg.pinv(information) if bread is None else bread
    V = B @ var.meat @ B
    V = (V + V.T) / 2
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    if var.df < 1:
        return _refused(labels, estimate, estimator, design, domain.n,
                        f"The analysis rows lie in {var.domain_psu} PSU"
                        f"{'s' if var.domain_psu != 1 else ''} of {var.domain_strata} "
                        f"strat{'a' if var.domain_strata != 1 else 'um'}: no design degrees of "
                        f"freedom are left for an interval.")
    rows = _t_rows(labels, estimate, se, np.full(len(estimate), float(var.df)))
    concerns = _concerns(design, var, domain.n, len(estimate), domain.left)
    if not converged:
        concerns.append("The weighted fit stopped before converging; treat these numbers with care.")
    return InferenceTable(rows, _info(estimator, _caption(design, var), design, var, domain.n),
                          concerns, cov=V)


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
    domain = domain_of(matrix.index, design)
    X = exog.to_numpy(dtype=float)[domain.keep]
    yv = np.asarray(y)[domain.keep]
    w = domain.weight
    estimator = {"regression": "survey-weighted least squares",
                 "binary": "survey-weighted logistic regression (pseudo-maximum likelihood)"}.get(
        task, "survey-weighted multinomial logistic regression (pseudo-maximum likelihood)")
    if domain.n <= X.shape[1]:
        return _refused(names, None, estimator, design, domain.n,
                        f"Only {domain.n:,} analysis rows carry a positive weight and a place in "
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
            return _refused(names, None, estimator, design, domain.n,
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
    return design_table(labels, fit.estimate, fit.scores, fit.information, design, domain,
                        estimator, converged=fit.converged)


def design_df(design: SurveyDesign, domain_ids: Sequence[int]) -> int:
    """PSUs minus strata among the design rows ``domain_ids`` (the NCHS rule), for sentences."""
    at = pd.Index(design.row_ids).get_indexer(np.asarray(domain_ids, dtype=np.int64))
    at = at[at >= 0]
    return int(len(np.unique(design.psu[at])) - len(np.unique(design.stratum[at])))


# ── joint tests on a design-based covariance ─────────────────────────────────


def adjusted_wald(W: float, q: int, d: int) -> tuple[float, float, float] | None:
    """The adjusted Wald test of q coefficients on d design degrees of freedom: ``F = (d − q + 1)
    W / (d q)`` on (q, d − q + 1) (Korn & Graubard 1990, *Am Stat* 44:270; Stata's ``test`` after
    ``svy``, SUDAAN's default). (F, denominator df, p); None when d < q, where no joint test can be
    made."""
    from scipy import stats

    den = d - q + 1
    if q < 1 or den < 1:
        return None
    F = (den * W) / (d * q)
    return F, float(den), float(stats.f.sf(F, q, den))


# ── families with no design-based estimator: block and record ────────────────

# What the population answer offers in their place, by task: a family that has one.
_DESIGN_FAMILY = {"regression": "linear", "binary": "linear", "multiclass": "linear",
                  "ordinal": "proportional_odds", "time_to_event": "cox"}
_DESIGN_LABEL = {
    ("linear", "regression"): "survey-weighted least squares",
    ("linear", "binary"): "survey-weighted logistic regression",
    ("linear", "multiclass"): "survey-weighted multinomial logistic regression",
    ("linear", "ordinal"): "survey-weighted multinomial logistic regression",
    ("proportional_odds", "ordinal"): "the survey-weighted proportional-odds model",
    ("cox", "time_to_event"): "survey-weighted Cox regression",
}
SAMPLE_EXIT = "Estimate for these participants instead: record the sample-only attestation"


def has_design_estimator(family: Any, task: str) -> bool:
    """Whether ``family`` estimates ``task`` design-based (its ``inference`` takes the design)."""
    import inspect

    fn = getattr(family, "inference", None)
    if fn is None or task not in getattr(family, "tasks", ()):
        return False
    return "survey" in inspect.signature(fn).parameters


def design_family(task: str) -> str | None:
    """The family that estimates ``task`` over the design, offered as the exit."""
    return _DESIGN_FAMILY.get(task)


def no_design_estimator(family: Any, task: str, models: Sequence[str] | None = None,
                        what: str = "coefficients") -> InferenceTable:
    """The table of a family with no design-based estimator under the population answer: blocked
    and recorded (MODELING_SEQUENCE §4). Its exits: the family that has one for this task (the
    chosen families with this one replaced), and the sample-only attestation."""
    label = getattr(family, "label", str(family))
    replacement = design_family(task)
    reason = (f"{label} has no design-based estimator, so its {what} would describe these "
              f"participants, not the surveyed population the survey answer names, and any "
              f"interval would ignore the strata and PSUs.")
    exits: list[dict[str, Any]] = []
    if replacement is not None and replacement != getattr(family, "key", None):
        chosen = list(models or [getattr(family, "key", "")])
        swapped = list(dict.fromkeys(replacement if m == getattr(family, "key", None) else m
                                     for m in chosen))
        exits.append({"label": f"Use {_DESIGN_LABEL.get((replacement, task), replacement)}",
                      "decision": {"kind": "select_models", "models": swapped}})
    exits.append({"label": SAMPLE_EXIT, "decision": {"kind": "set_survey", "estimand": "sample"}})
    return blocked(reason, exits, estimator="not fitted: no design-based estimator for the "
                                            "surveyed population")


def population_shelf(ranked: Sequence[tuple[Any, Any]], task: str) -> list[tuple[Any, Any]]:
    """The shelf under the population answer (MS4): every family with a design-based estimator for
    ``task`` first, in its order, then the rest, each with the concern that says its estimates
    will be blocked. The shelf is never shortened (BLUEPRINT §11.3: the menu stays whole)."""
    from turbotab.core.models.base import Assessment

    replacement = design_family(task)
    named = _DESIGN_LABEL.get((replacement, task), replacement) if replacement else None
    instead = f"; {named} has one" if named else ""
    out = []
    for family, judged in ranked:
        if has_design_estimator(family, task):
            out.append((family, judged))
            continue
        concern = (f"Under the surveyed population it has no design-based estimator, so its "
                   f"estimates are blocked and recorded{instead}.")
        out.append((family, Assessment(judged.score, judged.fit, (concern, *judged.concerns))))
    return sorted(out, key=lambda fa: not has_design_estimator(fa[0], task))


# ── the substitution curve over the design (Graubard & Korn 1999) ────────────


@dataclass
class DesignFit:
    """The linear family's design-based fit on a pipeline's model matrix, for curves.

    ``beta`` on the matrix with its intercept first; ``influence`` each kept row's influence on
    β̂ (kept rows × P: the bread times the row's weighted score), so ``Σ_i influence_i`` is β̂'s
    linearized error; ``link`` the inverse link, identity or logistic."""

    beta: np.ndarray
    influence: np.ndarray
    domain: Domain
    design: SurveyDesign
    link: str

    def mean(self, matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(prediction, its derivative with respect to η, the matrix with its intercept)."""
        X = np.column_stack([np.ones(len(matrix)), matrix.to_numpy(dtype=float)])
        eta = X @ self.beta
        if self.link == "identity":
            return eta, np.ones(len(eta)), X
        mu = 0.5 * (1.0 + np.tanh(0.5 * eta))
        return mu, mu * (1.0 - mu), X


def design_fit(task: str, matrix: pd.DataFrame, y: Any, classes: Sequence[Any] | None,
               design: SurveyDesign) -> DesignFit:
    """The fit :func:`survey_table` makes, kept for a curve. Raises ValueError, saying why, where
    the table would be refused."""
    import statsmodels.api as sm

    if task not in ("regression", "binary"):
        raise ValueError("A design-based curve follows a numeric or a yes/no outcome.")
    exog = sm.add_constant(matrix.astype(float), has_constant="add")
    domain = domain_of(matrix.index, design)
    X = exog.to_numpy(dtype=float)[domain.keep]
    yv = np.asarray(y)[domain.keep]
    if domain.n <= X.shape[1]:
        raise ValueError(f"Only {domain.n:,} analysis rows carry a positive weight and a place in "
                         f"the design, too few for {X.shape[1]} coefficients.")
    if task == "regression":
        fit = weighted_least_squares(X, yv.astype(float), domain.weight)
        link = "identity"
    else:
        if classes is None or len(classes) != 2:
            raise ValueError("A binary outcome needs exactly two classes.")
        event = (yv == classes[1]).astype(float)
        if separated_columns(X, event):
            raise ValueError("A column separates the outcome among the weighted rows, so the "
                             "weighted log-odds is infinite.")
        fit = weighted_logistic(X, event, domain.weight)
        link = "logit"
    bread = np.linalg.pinv(fit.information)
    return DesignFit(beta=fit.estimate, influence=fit.scores @ bread, domain=domain, design=design,
                     link=link)


def design_curve(fit: DesignFit, matrix_of: Callable[[pd.DataFrame], pd.DataFrame],
                 X: pd.DataFrame, shift: Any, ks: Sequence[float], live: Sequence[bool],
                 level: float = LEVEL) -> dict[str, Any]:
    """The curve over the surveyed population and its band, at each k.

    ``X`` holds the domain's rows (raw inputs, indexed by row id, in the order of ``fit.domain``'s
    kept rows); ``matrix_of`` is the pipeline up to its model step; ``shift`` the curve's
    :class:`~turbotab.core.methods.substitution.Shift`; ``live`` the point curve's own. The fixed
    population is the rows on support at every live k, as the point curve's is.

    At each k the estimate is the population mean of the change, ``θ_k = Σ w_i m_i Δ_i / Σ w_i
    m_i`` over the rows on support (``m_i``), ``Δ_i`` the change in the design-based fit's
    prediction. Its variance is the design variance of its linearization (Graubard & Korn 1999,
    *Biometrics* 55:652, predictive margins): ``z_i = w_i m_i (Δ_i − θ_k)/Σ w m + ψ_iᵀ g_k``, with
    ``ψ_i`` the row's influence on β̂ and ``g_k = Σ w m ∂Δ_i/∂β / Σ w m``, so it carries both the
    error of fitting the model and which people were sampled. The band is ``θ_k ± t(d) SE_k`` on
    the design degrees of freedom. Returns ``delta``, ``ci_low``, ``ci_high``, ``se``,
    ``fixed_delta``, ``fixed_ci_low``, ``fixed_ci_high``, ``df`` and the ``variance``."""
    from scipy import stats

    if len(X) != fit.domain.n:
        raise ValueError("The curve's rows are the domain's kept rows.")
    k_values = np.asarray(list(ks), dtype=float)
    live = np.asarray(list(live), dtype=bool)
    weight = fit.domain.raw
    valid = np.flatnonzero(shift.valid(X))
    base_frame = X.iloc[valid]
    base, base_d, base_X = fit.mean(matrix_of(base_frame))
    w = weight[valid]
    K = len(k_values)
    P = len(fit.beta)
    checked: dict[int, tuple[Any, np.ndarray]] = {}
    for i, k in enumerate(k_values):
        if live[i]:
            shifted, amount, composition = shift.checks(base_frame, float(k))
            checked[i] = (shifted, amount & composition)
    fixed = (np.logical_and.reduce([on for _, on in checked.values()]) if checked
             else np.zeros(len(valid), dtype=bool))
    z = np.zeros((fit.domain.n, 2 * K))
    theta = np.full(2 * K, np.nan)
    for i, (shifted, on) in checked.items():
        rows = np.flatnonzero(on)
        if not len(rows):
            continue
        moved, moved_d, moved_X = fit.mean(matrix_of(shifted.iloc[rows]))
        diff = moved - base[rows]
        grad = moved_X * moved_d[:, None] - base_X[rows] * base_d[rows][:, None]
        for slot, chosen in ((i, np.ones(len(rows), dtype=bool)), (K + i, fixed[rows])):
            if not chosen.any():
                continue
            wk = w[rows] * chosen
            total = float(wk.sum())
            value = float(wk @ diff) / total
            g = (wk @ grad) / total if P else np.zeros(0)
            theta[slot] = value
            z[valid[rows], slot] += wk * (diff - value) / total
            z[:, slot] += fit.influence @ g
    u = np.zeros((fit.design.n_rows, 2 * K))
    u[fit.domain.at] = z
    var = total_variance(u, fit.design, fit.domain.mask(fit.design))
    se = np.sqrt(np.clip(np.diag(var.meat), 0, None))
    df = var.df
    q = float(stats.t.ppf(0.5 + level / 2, df)) if df >= 1 else float("nan")

    def band(slots: slice) -> tuple[list[float | None], list[float | None], list[float | None]]:
        mid, low, high = [], [], []
        for t, s in zip(theta[slots], se[slots]):
            if not np.isfinite(t) or not np.isfinite(q):
                mid.append(None if not np.isfinite(t) else float(t))
                low.append(None)
                high.append(None)
                continue
            mid.append(float(t))
            low.append(float(t - q * s))
            high.append(float(t + q * s))
        return mid, low, high

    delta, ci_low, ci_high = band(slice(0, K))
    fixed_delta, fixed_low, fixed_high = band(slice(K, 2 * K))
    return {"delta": delta, "ci_low": ci_low, "ci_high": ci_high,
            "se": [float(s) if np.isfinite(t) else None for t, s in zip(theta[:K], se[:K])],
            "fixed_delta": fixed_delta, "fixed_ci_low": fixed_low, "fixed_ci_high": fixed_high,
            "df": int(df), "variance": var}


@dataclass
class PopulationCurve:
    """One family's substitution curve under the population answer: the point curve (the
    ``substitution_curve`` dict, weighted), its design-based ``band`` (:func:`design_curve`), or,
    for a family with no design-based estimator, why not (``refused``) and the ``exits``; then
    ``curve`` holds the support's counts only (they are the data's, not a model's)."""

    curve: dict[str, Any]
    band: dict[str, Any] | None = None
    refused: str | None = None
    exits: list[dict[str, Any]] = field(default_factory=list)


def population_curve(family: Any, task: str, pipeline: Any, X: pd.DataFrame, y: Any,
                     design: SurveyDesign, domain: Domain, models: Sequence[str] | None = None,
                     **curve_args: Any) -> PopulationCurve:
    """The substitution curve of ``family`` over the surveyed population (MS4).

    ``X`` and ``y`` are the domain's kept rows (every analyzed row placed in the design with a
    positive weight) and ``pipeline`` the family's fit on every analyzed row; ``curve_args`` are
    :func:`~turbotab.core.methods.substitution.substitution_curve`'s. A family with a design-based
    estimator for the task (the linear family) is refit with the weights on its own model matrix
    (:func:`design_fit`), the curve averages each row's change weighted by its survey weight, and
    the band is the design's (:func:`design_curve`). Any other family is blocked and recorded: no
    curve, its reason and exits (:func:`no_design_estimator`)."""
    from turbotab.core.methods.substitution import substitution_curve
    from turbotab.core.models.linear import model_matrix

    weights = domain.raw
    if not has_design_estimator(family, task) or task not in ("regression", "binary"):
        table = no_design_estimator(family, task, models, what="substitution curve")
        support = substitution_curve(lambda frame: np.zeros(len(frame)), X, weights=weights,
                                     **curve_args)
        return PopulationCurve(curve=support, refused=table.info["refused"],
                               exits=list(table.info["exits"]))

    def matrix_of(frame: pd.DataFrame) -> pd.DataFrame:
        return model_matrix(pipeline, frame)

    try:
        fit = design_fit(task, matrix_of(X), y, [0, 1] if task == "binary" else None, design)
    except ValueError as exc:
        support = substitution_curve(lambda frame: np.zeros(len(frame)), X, weights=weights,
                                     **curve_args)
        return PopulationCurve(curve=support, refused=f"No design-based curve: {exc}",
                               exits=[{"label": SAMPLE_EXIT,
                                       "decision": {"kind": "set_survey", "estimand": "sample"}}])
    curve = substitution_curve(lambda frame: fit.mean(matrix_of(frame))[0], X, weights=weights,
                               **curve_args)
    band = design_curve(fit, matrix_of, X, curve_args["shift"], curve["ks"], curve["live"])
    return PopulationCurve(curve=curve, band=band)


def curve_caption(design: SurveyDesign, var: DesignVariance, n_rows: int, level: float = LEVEL) -> str:
    """The saved figure's caption for a design-based band."""
    weight = f"`{design.weight_column}`" if design.weight_column else "equal weights"
    return (f"Shaded bands: {level:.0%} intervals by Taylor linearization over the survey design "
            f"(weights {weight}; {var.domain_psu:,} PSUs in {var.domain_strata:,} strata hold the "
            f"{n_rows:,} analysis rows), on t({var.df:,}), each curve the population mean of the "
            f"change in the survey-weighted fit's prediction; the band carries both the error of "
            f"the fit and which people were sampled (Graubard & Korn 1999).")


def pooled_design_caption(design: SurveyDesign, var: DesignVariance, n_rows: int, m: int,
                          supplied: bool = False, level: float = LEVEL) -> str:
    """The saved figure's caption for a design-based band pooled over imputed copies (MS2-MS4)."""
    weight = f"`{design.weight_column}`" if design.weight_column else "equal weights"
    what = f"the data's {m} imputed copies" if supplied else f"{m} imputations"
    return (f"Shaded bands: {level:.0%} intervals pooled over {what} by Rubin's rules, each copy's "
            f"variance by Taylor linearization over the survey design (weights {weight}; "
            f"{var.domain_psu:,} PSUs in {var.domain_strata:,} strata hold the {n_rows:,} analysis "
            f"rows), on Barnard–Rubin degrees of freedom with the design's {var.df:,} as the "
            f"complete-data degrees of freedom; each curve the population mean of the change in "
            f"the survey-weighted fit's prediction (Graubard & Korn 1999).")


# ── the methods sentences (BLUEPRINT §13: each contract's sentence) ──────────

# Each design-based estimator in the words the methods section writes it, by (family, task).
_ESTIMATOR_WORDS = {
    ("linear", "regression"): "least squares",
    ("linear", "binary"): "logistic regression (pseudo-maximum likelihood)",
    ("linear", "multiclass"): "multinomial logistic regression (pseudo-maximum likelihood)",
    ("linear", "ordinal"): "multinomial logistic regression (pseudo-maximum likelihood)",
    ("proportional_odds", "ordinal"): "the proportional-odds model (pseudo-maximum likelihood)",
    ("cox", "time_to_event"): "Cox regression (Binder's pseudo-likelihood, Efron ties)",
}


# A family with no design-based estimator, as the sentence names it (each one model).
_BLOCKED_WORDS = {
    "mixed": "the random-intercept mixed model",
    "gee": "the GEE model",
    "featurewise": "feature-wise regression",
    "elastic_net": "the elastic net",
    "boosted_trees": "the gradient-boosted tree model",
}


def _listing(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def population_answer(state: Any) -> bool:
    """Whether ``state`` answers the survey question "the surveyed population" under inference."""
    survey = getattr(state, "survey", None)
    return (getattr(state, "purpose", None) == "inference" and survey is not None
            and getattr(survey, "estimand", None) == "population")


def models_sentence(state: Any, models: Sequence[str], task: str | None) -> str | None:
    """What the population answer does to the chosen families, for the ``select_models``
    sentence: each design-based estimator named, weighted by the survey weight with Taylor-
    linearized standard errors; each family with none named as blocked. None unless the answer is
    the surveyed population under inference."""
    from turbotab.core.models import get_family

    survey = getattr(state, "survey", None)
    if not population_answer(state) or task is None:
        return None
    based: list[str] = []
    stopped: list[str] = []
    for key in models:
        try:
            family = get_family(key)
        except KeyError:
            continue
        words = _ESTIMATOR_WORDS.get((key, task))
        if words is not None and has_design_estimator(family, task):
            based.append(words)
        else:
            stopped.append(_BLOCKED_WORDS.get(key, f"`{key}`"))
    parts: list[str] = []
    weight = getattr(survey, "weight", None)
    if based:
        by = f" by `{weight}`" if weight else ""
        verb = "was" if len(based) == 1 else "were"
        parts.append(f"for the surveyed population, {_listing(based)} {verb} weighted{by}, with "
                     f"standard errors by Taylor linearization over the survey design")
    if stopped:
        verb = "has" if len(stopped) == 1 else "have"
        whose = "its" if len(stopped) == 1 else "their"
        parts.append(f"{_listing(stopped)} {verb} no design-based estimator, so {whose} estimates "
                     f"were blocked and not reported")
    if not parts:
        return None
    text = "; ".join(parts)
    return text[0].upper() + text[1:]


def substitution_clause(state: Any, n_boot: int = 0) -> str | None:
    """The ``set_substitution`` sentence's clause under the population answer (the curve's
    contract sentence), or None. ``n_boot``: the bootstrap refits the answer asked for, which the
    design's band replaces and the clause says are not drawn."""
    if not population_answer(state):
        return None
    from turbotab.core.voice import count

    undrawn = (f" (the {count(n_boot)} bootstrap refits asked for are not drawn: a row bootstrap "
               f"ignores the strata and PSUs)" if n_boot else "")
    return ("over the surveyed population its curve is the weighted mean of each participant's "
            "change in the survey-weighted fit, and its band comes from Taylor linearization over "
            f"the survey design{undrawn}")


# ── the method contracts (BLUEPRINT §13), in the one registry ─────────────────


def _register_contracts() -> None:
    """MS4's five contracts, in ``turbotab.core.contracts`` with its vocabulary: the population
    answer (the hub whose relations bind every family and display), the three design-based
    estimators, and the substitution curve over the population. Each writes its clause of the
    methods paragraph (``contracts.paragraph``) from ``run``: ``estimand`` ("population" or
    "sample"), ``weight``, ``strata``, ``psu``."""
    from turbotab.core.contracts import ContractOption as Option
    from turbotab.core.contracts import MethodContract, Relation, register_contract

    not_asked = ("Not asked: under prediction the scores describe the rows they were computed on, "
                 "and the weights are noted, not used.")
    model_scope = ("The weighted fit learns from the outcome and every study row in its domain "
                   "(the analysis rows); the strata and PSUs enter only its variance. Lockbox §06's "
                   "test: row i's fitted value moves with the outcome.")
    population_need = "the survey answer: the surveyed population (a weight, and strata and PSUs)"

    def estimator(key: str, label: str, family: str, tasks: str, words: str, reference: str,
                  storyboard: tuple[str, ...], diagnostic: Relation | None) -> MethodContract:
        def clause(run: Any, _words: str = words) -> str | None:
            # The population contract's clause states the weight and the linearization once.
            return _words if run.get("estimand", "population") == "population" else None
        relations = [
            Relation("implies", "design-based intervals",
                     "Every interval is design-based: Taylor linearization over the strata and "
                     "PSUs, on t with the design's degrees of freedom (the PSUs minus the strata "
                     "that hold the analysis rows).",
                     purposes=("inference",), enforced_by="turbotab.core.models.survey:design_table",
                     id="design_based_intervals"),
            Relation("implies", "lonely PSUs centered",
                     "A stratum with a single PSU is centered at the mean PSU total of the strata "
                     "that hold analysis rows (R survey's lonely.psu \"adjust\"), and the table "
                     "names it.",
                     purposes=("inference",), enforced_by="turbotab.core.models.survey:total_variance",
                     id="lonely_psus"),
            Relation("implies", "adjusted Wald F for joint tests",
                     "A spline's overall and nonlinear tests are adjusted Wald F tests on "
                     "(q, d − q + 1) degrees of freedom (Korn & Graubard 1990).",
                     purposes=("inference",), enforced_by="turbotab.core.models.survey:adjusted_wald",
                     id="adjusted_wald"),
        ]
        if diagnostic is not None:
            relations.append(diagnostic)
        return MethodContract(
            key=key, label=label, slot="model", scope="model", scope_note=model_scope,
            package="SURVEY",
            needs=(population_need, tasks, f"the {family} family chosen"),
            question=("Do the estimates describe the surveyed population or these participants? "
                      "(the survey question; this estimator answers \"the surveyed population\")"),
            place=("MODELING_SEQUENCE §1 step 9, the shelf: under the population answer the "
                   "families with a design-based estimator come first (\"design-based for "
                   "surveys\")"),
            decision="select_models", stage="fit", run_order=1.0,
            options=(
                Option("design_based", f"Design-based: {words}",
                       "NCHS's analytic guidelines: weights, strata and PSUs, Taylor linearization "
                       "(NHANES Analytic Guidelines 2011–2016 §3.2)",
                       {"inference": "Sound for the surveyed population: weighted estimating "
                                     "equations with a linearized variance (Binder 1983; Lumley "
                                     "2010).",
                        "prediction": not_asked},
                       {"inference": "recommended", "prediction": "not_offered"}),
                Option("unweighted", "Unweighted, with model-based standard errors",
                       "Common where an analysis is of the participants themselves",
                       {"inference": "Under the population answer it describes these participants "
                                     "and its interval ignores the strata and PSUs, so it is "
                                     "blocked and recorded; it is reported only with the "
                                     "sample-only attestation.",
                        "prediction": "The fit prediction uses: its scores describe these rows."},
                       {"inference": "block_and_record", "prediction": "recommended"}),
            ),
            leash={"inference": "recommended", "prediction": "not_offered"},
            storyboard=storyboard, relations=tuple(relations),
            sources=("Binder 1983, Int Stat Rev 51:279", "Lumley 2010, Complex Surveys",
                     reference),
            clause=clause, sentence="turbotab.core.models.survey:models_sentence")

    for c in (
        MethodContract(
            key="survey_population", label="The population estimand under a survey design",
            slot="model", scope="model", run_order=0.0, package="SURVEY",
            scope_note=("The answer chooses the outcome model's estimator. Each row's weight is its "
                        "own, but every design-based estimate and its variance read the outcome and "
                        "every PSU's rows."),
            needs=("design columns read as the weight, the strata and the PSUs (settled readings)",
                   "purpose: inference"),
            question="Do the estimates describe the surveyed population or these participants?",
            place=("Asked after the roles, as the survey question (the Router's survey step); "
                   "MODELING_SEQUENCE §1 step 9 orders the shelf by it"),
            decision="set_survey", stage="fit",
            options=(
                Option("population", "The surveyed population (weights, strata and PSUs)",
                       "NCHS's analytic guidelines for NHANES: the sample weights with the "
                       "design's strata and PSUs (NHANES Analytic Guidelines 2011–2016)",
                       {"inference": "Sound for a population estimand: every family and display "
                                     "is design-based, or blocked and recorded where none exists "
                                     "(MODELING_SEQUENCE §0 ruling 6).",
                        "prediction": not_asked},
                       {"inference": "recommended", "prediction": "not_offered"}),
                Option("sample", "These participants (the sample-only attestation)",
                       "Unweighted analyses of survey participants, stated as such",
                       {"inference": "Sound only as attested: the estimates describe these "
                                     "participants, unweighted, and their standard errors ignore "
                                     "the strata and PSUs; the record carries the attestation.",
                        "prediction": not_asked},
                       {"inference": "available", "prediction": "not_offered"}),
            ),
            leash={"inference": "recommended", "prediction": "not_offered"},
            storyboard=("The survey question: the surveyed population, or these participants",
                        "Each chosen family: its design-based estimator, or its block with exits",
                        "Each display: design-based (the coefficient table, the form tests, the "
                        "substitution curve, the relative effects), or blocked with exits",
                        "The record: each sentence restated on the answer as it stands"),
            relations=(
                Relation("implies", "every family and display",
                         "A design-based estimator, or block and record with the sample-only "
                         "attestation as its exit (MODELING_SEQUENCE §2).",
                         purposes=("inference",),
                         enforced_by="turbotab.core.models.survey:no_design_estimator",
                         condition="the survey answer \"the surveyed population\"",
                         id="design_based_or_blocked"),
                Relation("conflicts", "families with no design-based estimator",
                         "The mixed model, GEE, feature-wise tests, the elastic net and boosted "
                         "trees have no design-based estimator: their estimates are blocked and "
                         "recorded.",
                         purposes=("inference",), rung="block_and_record",
                         exits=("the design-based family for the task in its place, every other "
                                "chosen family kept", "the sample-only attestation"),
                         enforced_by="turbotab.core.models.survey:no_design_estimator",
                         id="no_design_estimator"),
                Relation("invalidates", "the substitution band from row resampling",
                         "A row bootstrap ignores the strata and PSUs, so the curve's band is the "
                         "design's linearization instead, and the refits asked for are not drawn.",
                         purposes=("inference",),
                         enforced_by="turbotab.core.models.survey:population_curve",
                         id="bootstrap_band"),
                Relation("implies", "multiple imputation on the design's degrees of freedom",
                         "Each completed copy is analyzed design-based, and Rubin's rules take the "
                         "design's degrees of freedom as the complete-data degrees of freedom "
                         "(MS2).",
                         purposes=("inference",),
                         enforced_by="turbotab.core.stages.modeling:pooled_table",
                         id="multiple_imputation"),
                Relation("conflicts", "regression calibration",
                         "Regression calibration has no design-based variance here (a bootstrap "
                         "by PSU within strata over the whole chain), so it is blocked and "
                         "recorded.",
                         purposes=("inference",), rung="block_and_record",
                         exits=("the sample-only attestation", "no correction"),
                         enforced_by="turbotab.core.stages.calibration:population_exits",
                         id="regression_calibration"),
                Relation("conflicts", "a scale's corrected coefficient",
                         "A scale's correction and the uncorrected coefficient beside it are fit "
                         "on the rows as sampled, so they are blocked and recorded.",
                         purposes=("inference",), rung="block_and_record",
                         exits=("the sample-only attestation",),
                         enforced_by="turbotab.core.stages.scales:population_block",
                         id="scale_correction"),
                Relation("implies", "the record restated",
                         "Every sentence that says what the survey answer does to a family, a "
                         "curve or a correction is restated in the methods text on the answer as "
                         "it stands; the Record keeps it as said.",
                         purposes=("inference",),
                         enforced_by="turbotab.core.provenance:restated", id="restated"),
                Relation("implies", "cross-validated scores labeled unweighted",
                         "Cross-validated scores, their calibration and the family comparisons "
                         "are about the fitting procedure on these rows: they are labeled "
                         "unweighted, not design-based (awaiting the owner's ruling).",
                         purposes=("inference",),
                         enforced_by="turbotab.core.stages.modeling:fit_stage",
                         id="unweighted_scores"),
            ),
            sources=("MODELING_SEQUENCE §0 ruling 6, §2, §4",
                     "NHANES Analytic Guidelines 2011–2016 §3.2",
                     "Korn & Graubard 1999, Analysis of Health Surveys"),
            clause=_population_clause, sentence="turbotab.core.voice:sentence_for"),
        estimator("survey_linear", "Survey-weighted linear, logistic and multinomial models",
                  "linear", "a continuous, yes/no or multiclass outcome",
                  "estimates came from survey-weighted least squares, logistic or multinomial "
                  "logistic regression (pseudo-maximum likelihood)", "R survey::svyglm",
                  ("Weight each row by the people it stands for",
                   "Solve the weighted estimating equations",
                   "Sum each PSU's weighted scores",
                   "Spread the PSU totals within strata",
                   "Read intervals on t, the PSUs minus the strata"),
                  None),
        estimator("survey_cox", "Survey-weighted Cox regression (Binder's pseudo-likelihood)",
                  "Cox", "a time-to-event outcome with its follow-up",
                  "hazard ratios came from survey-weighted Cox regression (Binder's "
                  "pseudo-likelihood, Efron ties)", "R survey::svycoxph",
                  ("Weight each row in every risk set",
                   "Solve the weighted partial-likelihood score (Efron ties)",
                   "Take each row's weighted score residual",
                   "Spread the PSU totals within strata"),
                  Relation("implies", "proportional hazards checked on these participants",
                           "The Schoenfeld check is a diagnostic on these participants, not a "
                           "design-based test.", purposes=("inference",),
                           enforced_by="turbotab.core.models.survival:survey_cox_table",
                           id="proportional_hazards")),
        estimator("survey_ordinal", "Survey-weighted proportional-odds model",
                  "proportional-odds", "an ordered outcome with its declared order",
                  "cumulative odds ratios came from a survey-weighted proportional-odds model "
                  "(pseudo-maximum likelihood)", "R survey::svyolr",
                  ("Weight each row's cumulative-logit likelihood",
                   "Solve the weighted score",
                   "Spread the PSU totals within strata"),
                  Relation("implies", "proportional odds checked on these participants",
                           "The Brant check is a diagnostic on these participants, not a "
                           "design-based test.", purposes=("inference",),
                           enforced_by="turbotab.core.models.ordinal:survey_ordinal_table",
                           id="proportional_odds")),
        MethodContract(
            key="survey_substitution", label="The substitution curve over the surveyed population",
            slot="evaluation", scope="model", scope_note=model_scope, package="SURVEY",
            needs=(population_need, "a substitution answer", "the linear family chosen"),
            question="Which energy substitution, in what steps? (the substitution question)",
            place="MODELING_SEQUENCE §1 step 11's displays, beside the coefficient table",
            decision="set_substitution", stage="substitution", run_order=1.0,
            options=(
                Option("design_band", "A band by Taylor linearization over the survey design",
                       "Predictive margins with survey data (Graubard & Korn 1999)",
                       {"inference": "Sound for the surveyed population: the band carries both the "
                                     "error of the weighted fit and which people were sampled.",
                        "prediction": not_asked},
                       {"inference": "recommended", "prediction": "not_offered"}),
                Option("row_bootstrap", "A band from refits on bootstrap resamples of rows",
                       "The band this app draws without a survey design",
                       {"inference": "Not under the population answer: a row bootstrap ignores the "
                                     "strata and PSUs, so the design's band replaces it.",
                        "prediction": "The band prediction draws on its training rows."},
                       {"inference": "not_offered", "prediction": "recommended"}),
            ),
            leash={"inference": "recommended", "prediction": "not_offered"},
            storyboard=("Refit the linear family with the weights",
                        "Move k kcal on every row on support",
                        "Average each row's change over the population (weighted)",
                        "Linearize the average over the design for its band"),
            relations=(
                Relation("invalidates", "the substitution band from row resampling",
                         "A row bootstrap ignores the strata and PSUs, so the band is the "
                         "design's instead.", purposes=("inference",),
                         enforced_by="turbotab.core.models.survey:design_curve",
                         id="bootstrap_band"),
                Relation("conflicts", "a curve from a family with no design-based estimator",
                         "Blocked and recorded: no curve, its reason and its exits.",
                         purposes=("inference",), rung="block_and_record",
                         exits=("the design-based family in its place, every other chosen family "
                                "kept", "the sample-only attestation"),
                         enforced_by="turbotab.core.models.survey:population_curve",
                         id="blocked_curve"),
            ),
            sources=("Graubard & Korn 1999, Biometrics 55:652", "R survey::svyglm + svycontrast"),
            clause=_substitution_contract_clause,
            sentence="turbotab.core.models.survey:substitution_clause"),
    ):
        register_contract(c)


def _population_clause(run: Any) -> str | None:
    """The population contract's clause of the methods paragraph."""
    from turbotab.core.survey import ATTESTATION

    if run.get("estimand") == "sample":
        return f"the estimates describe these participants: {ATTESTATION}"
    weight, strata, psu = run.get("weight"), run.get("strata"), run.get("psu")
    over = (f" over `{psu}` nested within `{strata}`" if psu and strata
            else (f" over `{psu}`" if psu else ""))
    return (f"the estimates describe the surveyed population: rows were weighted by `{weight}`, "
            f"and standard errors were estimated by Taylor series linearization{over}, with t "
            f"intervals on the PSUs minus the strata that hold the analysis rows")


def _substitution_contract_clause(run: Any) -> str | None:
    if run.get("estimand", "population") != "population":
        return None
    return ("the substitution curve is the population mean of each participant's change in the "
            "survey-weighted fit, its band by Taylor linearization over the survey design "
            "(Graubard & Korn 1999)")


SURVEY_CONTRACTS = ("survey_population", "survey_linear", "survey_cox", "survey_ordinal",
                    "survey_substitution")
_register_contracts()


__all__ = [
    "DesignFit", "DesignVariance", "Domain", "FEW_DESIGN_DF", "LONELY_METHOD",
    "LONELY_PSU", "LONELY_RULE", "SAMPLE_EXIT", "SURVEY_CONTRACTS", "SurveyDesign",
    "WeightedFit", "adjusted_wald", "blocked", "build_design", "curve_caption", "design_curve",
    "pooled_design_caption",
    "design_df", "design_family", "design_fit", "design_table", "domain_of", "has_design_estimator",
    "models_sentence", "no_design_estimator", "PopulationCurve", "population_answer",
    "population_curve", "population_shelf", "substitution_clause", "survey_info", "survey_table",
    "total_variance",
    "weighted_least_squares", "weighted_logistic", "weighted_multinomial",
]
