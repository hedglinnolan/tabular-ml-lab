"""Intervals that match how the rows were sampled (AUDIT_REPORT §5 WP2: MA-01, MA-06, MA-07, MA-08).

Under inference the linear family's coefficient table is made here, in four parts:

* **Which rows belong together** (:func:`resolve_clusters`) is read from the grain answer, the
  identifier roles and the analysis rows themselves, never from how the seal was drawn. A seal
  abandoned for too few units, or a grain answered "I don't know", still leaves an identifier that
  repeats, and intervals that ignore it are too narrow (Cameron & Miller 2015, *J Hum Resour*
  50:317). A missing identifier is its own unit, as the split maps it
  (``turbotab/core/seal.py::seal_inputs``).
* **Rows that repeat** get CR2 standard errors (bias-reduced linearization: Bell & McCaffrey 2002,
  *Surv Methodol* 28:169) and t reference distributions whose degrees of freedom are Bell and
  McCaffrey's Satterthwaite approximation under the working model of independent rows
  (Pustejovsky & Tipton 2018, *J Bus Econ Stat* 36:672; clubSandwich's ``test = "Satterthwaite"``).
  One engine, :func:`cr2`, serves OLS, logistic and multinomial models through their working
  linear models (McCaffrey & Bell 2006, *Stat Med* 25:4081). Below the unit floor no interval is
  reported: the table says why and offers the exits.
* **Independent rows** of a least-squares model get HC3 standard errors with t(n − p) reference
  distributions (Long & Ervin 2000, *Am Stat* 54:217), and a Breusch–Pagan check in Koenker's
  studentized form says when the residual spread is not constant. A coefficient that rests on a
  row of leverage one, which the fit passes through, has no HC3 standard error and is refused by
  row, naming it (:func:`resting_on_leverage_one`); the others are the other rows' HC3.
* **Separation** in a logistic model is found by linear programming (Konis 2007, *Linear
  programming algorithms for detecting separated data in binary logistic regression models*,
  DPhil thesis, Oxford), named, and the table becomes Firth's penalized likelihood (Firth 1993,
  *Biometrika* 80:27), whose estimates are finite (Heinze & Schemper 2002, *Stat Med* 21:2409).
  It reports those estimates alone, complete separation or quasi-complete, in words that name the
  separated columns (the cone :func:`separated_columns` finds, all of which left out ends the
  separation) and say why (:func:`separated_profile_words`): with a coefficient held at a value
  the penalized likelihood can then have more than one maximum, and the fit cannot certify that
  it found the highest, so no profile interval or likelihood-ratio p-value is reported. Every
  Firth profile quantity is refused on separated rows in typed words, whoever asks
  (:class:`FirthProfileSeparated`: :func:`firth_profile`, a combination's
  :func:`firth_combination`, the form tests' ``exposure_form.firth_lr_test``); on rows nothing
  separates :func:`firth_profile` computes the profile. The survey
  multinomial and the proportional-odds tables run the same linear programs on their own
  likelihoods (:func:`separated_multinomial`, :func:`separated_ordinal`) and refuse a separated
  table, naming the separating column. Under multiple
  imputation each completed copy's table, and each refit its form tests make on it, is made for
  Rubin's rules (:func:`for_pooling`), which read an estimate and its standard error: a Firth copy
  computes no profile, and the pooled table reports only the estimate of a coefficient that
  separates the outcome in any copy, withholding the standard error and the Wald interval and
  p-value built from it (:func:`pooled_firth`).

A table whose rows do not exceed its coefficients (the intercept among them) is refused before
any of these, its estimates with its intervals: those rows cannot both determine the estimates and
measure their uncertainty (:func:`_no_spare_rows`). The mixed model's and the GEE's tables refuse
them in the same words (``repeated``). The proportional-odds model's table is refused at the same
count, its cut-points among its coefficients, in its own words (``ordinal``): its cut-points are
shared by every row, so its likelihood there can have a unique maximum, and the words say only
that the rows are no more than its coefficients. Every table of a family whose linear predictor
is Xβ (least squares, logistic, Firth, multinomial, proportional odds, the GEE, and the survey
tables) is fit on its columns centered at their means and scaled to an SD of one, its slopes
scaled back and its intercept or cut-points mapped back to the columns' own origin, so no
column's unit or origin moves another's numbers (:func:`by_identified`); the Cox model's are
scaled and not centered (the Cox fit centers its columns itself), and the mixed model's REML
factors the design by QR, which already leaves the numbers unmoved. The near-singular concern the
GEE, Cox and mixed tables refuse in reads the columns centered too
(``linear.collinearity_concern``). An iterative fit's table (logistic, multinomial, proportional
odds, Firth, and their survey forms) is served only at its maximum: a fit that does not converge,
or whose Newton step from its estimate would move a coefficient by more than
:data:`SHORT_OF_MAXIMUM` of a standard error, is refused with no estimate (:func:`not_converged`,
:func:`at_maximum`), that distance taken from the fit's score and its working design's QR
(:func:`newton_distance`, :func:`firth_distance`), and a Firth profile that cannot be computed is
refused in its own words (:class:`FirthProfileFailed`). In each of them, and the mixed model's,
where the model matrix's columns are dependent at R's tolerance on the centered columns
(``repeated.aliased``: a column whose part outside the span of those before it is under 10⁻⁷ of
its length is aliased, as ``lm`` aliases it; which columns are in its dependency is read on the
same centered columns, each scaled to unit length) and the caller names what is studied
(``study``), a coefficient in no dependency is reported as the fit with the aliased columns left out gives it, as R's ``lm`` does, and each one in a
dependency is refused by row, with no estimate, in words that name the dependency's columns; the
whole table is refused, with no estimate, where a column named as studied, or every coefficient,
is in one, or where the caller names nothing studied (:func:`dependency_reason`'s words), before
a fit whose numbers for them would be rounding. The least-squares tables' HC3 and CR2 standard
errors, the CR2 ones of the logistic, multinomial and GEE working models, and the logistic,
multinomial and survey tables' inverse information are taken from the QR of the design or the
working design (:func:`qr_frame`), whose rounding is its condition number, not from ``XᵀX``,
whose rounding is that squared; the proportional-odds tables evaluate their information on the
QR's Q (``ordinal.on_qr``). A table whose fit gives some coefficient no finite,
positive standard error, or its t interval no finite, positive degrees of freedom, is refused in
words that name each and what the fit could not compute (:func:`no_interval`; every family's
table that makes Wald or t rows here, in ``repeated``, ``survival``, ``ordinal`` and ``survey``),
rather than served with the estimate beside no interval.

And on the outcome's own scale (AUDIT_REPORT §5 WP8: ME-07): every table says what one unit of an
input does, in words that name the outcome (:class:`Outcome`). A least-squares row is a difference
in the mean outcome. A logistic row's estimate is a log-odds; its ``ratio`` is the odds ratio
exp(β) for the named event against the named reference level, with exp of the interval's ends, to
be drawn on a log axis. A multinomial row's ``ratio`` is the relative-risk ratio exp(β_k) of its
class against the reference class (the first, as the model codes it), which the table names.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from turbotab.core import claims

LEVEL = 0.95
# Cameron & Miller (2015) §VI: "'few' may range from less than 20 clusters to less than 50
# clusters in the balanced case". Below this many, a concern names the number of clusters. The
# number is its claim's (``inference.few_clusters``), not the trial core's ten.
FEW_CLUSTERS = claims.value("inference.few_clusters", "few_clusters")
# The Breusch–Pagan check speaks below this p-value. HC3 is used whatever it says; the check only
# tells the reader the spread is not constant, so it is kept strict to stay silent on clean data.
HETERO_ALPHA = 0.01
HETERO_MATERIAL = 0.10  # …and only when classical standard errors would be off by this much
FIRTH_MAX_STEP = 5.0  # the largest Newton step on any coefficient (logistf's ``maxstep``)
_BATCH = 1_000_000  # floats per block of clusters in the CR2 engine (8 MB)
# The Bell–McCaffrey df cost about G·P³ operations (6.8 × 10⁹ took 0.3 s). Beyond this (say
# 100,000 clusters and 100 columns) they are not computed: the intervals fall back to t(G − 1),
# and the caption says so.
BM_BUDGET = 1e11
_EIG_FLOOR = 1e-10  # 1 − λ below this is a zero eigenvalue of I − H_gg: pseudo-inverted (clubSandwich)
# Set while each completed copy's table is made for Rubin's rules (:func:`for_pooling`).
_POOLING: ContextVar[bool] = ContextVar("inference_pooling", default=False)
FIRTH = "Firth-penalized logistic regression"
NAMED = 6  # the coefficients a refusal for want of a standard error names (:func:`no_interval`)

# CR1: the cluster sandwich with the G/(G − 1) correction, on t(G − 1), for a model with no
# working linear form here (the proportional-odds family, ``turbotab/core/models/ordinal.py``).
# CR0 is the Cox model's Lin–Wei cluster sandwich (``survival.py``); "model" includes a mixed
# model's REML covariance (``repeated.py``).
# design: Taylor linearization over a survey design (``turbotab/core/models/survey.py``, WP10).
Covariance = Literal["HC3", "CR2", "CR1", "CR0", "model", "profile", "design", "none"]
Scale = Literal["difference", "odds_ratio", "relative_risk_ratio"]
Rows = Literal["all", "training"]


def min_clusters() -> int:
    """The unit floor: the seal's own (``turbotab/core/seal.py::min_groups``), so the grain
    question, the seal and the intervals agree about how few units are too few."""
    from turbotab.core.seal import min_groups

    return min_groups()


# ── which rows belong together ───────────────────────────────────────────────


@dataclass(frozen=True)
class Clusters:
    """The clustering the intervals use, decided independently of the seal's grouping."""

    column: str | None = None
    codes: np.ndarray | None = None  # one integer per analysis row (aligned with them), or None
    n_clusters: int = 0
    n_missing: int = 0  # rows with no identifier: each is a unit of its own
    note: str | None = None  # the answers and the data disagree, said plainly
    refusal: str | None = None  # the rows repeat but nothing says which belong together
    exits: tuple[dict[str, Any], ...] = ()

    @property
    def clustered(self) -> bool:
        return self.codes is not None


INDEPENDENT = Clusters()


def cluster_columns(state: Any, available: Sequence[str], also: Sequence[str | None] = ()) -> list[str]:
    """Columns that may name the unit: the grain's named column first, then every identifier,
    then every cluster (a site or household: rows within one are not independent, so a repeating
    cluster clusters the intervals as a repeating identifier does; WP13 split it from the
    identifiers so that the seal never groups by it), then ``also`` (the column the split grouped
    by, when it is none of these)."""
    grain = getattr(state, "grain", None)
    named = getattr(grain, "id_column", None) if grain is not None else None
    roles = getattr(state, "roles", None) or {}
    identifiers = [c for c, r in roles.items() if r == "identifier"]
    clusters = [c for c, r in roles.items() if r == "cluster"]
    # WP17 (audit RO-08): the grouping the cluster question named, whatever its role.
    from turbotab.core.estimand import cluster_answer

    answered = cluster_answer(state)
    have = set(available)
    wanted = [*([named] if named else []), *identifiers, *clusters,
              *([answered] if answered else []), *[c for c in also if c]]
    return [c for c in dict.fromkeys(wanted) if c in have]


def _missing(values: pd.Series) -> np.ndarray:
    return values.isna().to_numpy(dtype=bool)


def resolve_clusters(state: Any, frame: pd.DataFrame, also: Sequence[str | None] = ()) -> Clusters:
    """How the analysis rows cluster, for the intervals: by the unit column whenever one repeats.

    ``frame`` holds the candidate columns (:func:`cluster_columns`) over the analysis rows,
    indexed by row id. The grain's named column is used when it repeats; otherwise the repeating
    identifier with the fewest units (the coarsest grouping, as the seal reads it). A missing
    identifier is its own unit, keyed by its row id, exactly as the split keys it.
    """
    n = int(len(frame))
    grain = getattr(state, "grain", None)
    answer = getattr(grain, "grain", None) if grain is not None else None
    named = getattr(grain, "id_column", None) if grain is not None else None
    repeating: list[tuple[int, int, str]] = []
    waiting_repeats: list[tuple[int, str]] = []
    # BLUEPRINT §14.3: which rows belong together is a cluster reading no value test settles (a
    # household's line number, a stratum, an interviewer repeat as a unit's identifier does), so
    # the intervals read it only from the user: the grain's named column and the split's grouping
    # (both the user's answers), a cluster or identifier role the user confirmed, or the column's
    # own confirmation. The grain answer always wins: an identifier on another column does not
    # cluster the intervals over the unit the grain names (the gate: MEPS ``PID``, G = 19, over
    # ``DUPERSID``), and a group of units the user named (a household) wins over an identifier
    # (the gate: a roster's ``person_no`` over the household the user confirmed).
    from turbotab.core.readings import cluster_exits, cluster_rank, cluster_reading

    # WP17 (audit RO-08): a grouping the user named at the cluster question is the user's answer,
    # as the grain's column and the split's grouping are.
    from turbotab.core.estimand import cluster_answer

    grouping = cluster_answer(state)
    answered = {named, grouping, *[c for c in also if c]}
    for column in cluster_columns(state, list(frame.columns), also):
        values = frame[column]
        units = int(values.nunique(dropna=True))
        if not 0 < units < int(values.notna().sum()):
            continue
        if column in answered:
            # The grouping above the person nests the units, so it is the level the intervals
            # cluster at (MODELING_SEQUENCE §2: "at the highest correlated level").
            rank = -1 if column == grouping else 0 if column == named else cluster_rank(state, column)
            repeating.append((rank, units, column))
            continue
        found = cluster_reading(state, column)
        if not found.settled:
            waiting_repeats.append((units, column))
        elif found.value == "yes":
            repeating.append((cluster_rank(state, column), units, column))
    repeating.sort()
    chosen = repeating[0][2] if repeating else None
    if chosen is None and waiting_repeats and not named:
        units, column = sorted(waiting_repeats)[0]
        roles = getattr(state, "roles", None) or {}
        role = roles.get(column)
        return Clusters(
            refusal=(f"`{column}` repeats ({units:,} values over {n:,} rows), and whether rows "
                     f"sharing it belong together is asked, not assumed: a unit's identifier, a "
                     f"household's line number, a stratum and an interviewer all repeat this way"
                     + (f" (its role, {role}, was not confirmed on its own)" if role else "")
                     + ". Intervals that cluster by it, or that ignore it, would each rest on a "
                       "guess."),
            exits=tuple([*(confirm_exits_for(state, column)), *cluster_exits(state, column)]))
    passed_over = [c for _, c in sorted(waiting_repeats)] if named else []
    if chosen is not None:
        units = next(u for _, u, c in repeating if c == chosen)
        values = frame[chosen].astype(object)
        missing = _missing(values)
        keys = np.asarray(values, dtype=object).copy()
        keys[missing] = [f"__missing_{rid}" for rid in np.asarray(frame.index)[missing]]
        codes = pd.factorize(pd.Series(keys, dtype=object))[0].astype(np.int64)
        seen = f"`{chosen}` repeats ({units:,} values over {n:,} rows)"
        so = f"so the intervals are cluster-robust by `{chosen}`."
        note = None
        if answer == "unknown":
            note = (f"Whether a unit can appear in more than one row was answered as not known, "
                    f"but {seen}, {so}")
        elif answer == "one_row_per_unit" and chosen == named:
            note = f"Each row was said to be a different unit, but {seen}, {so}"
        elif answer == "one_row_per_unit":
            note = (f"Each row was said to be a different unit, and {seen}, a grouping you "
                    f"confirmed, {so}")
        elif answer is None:
            note = f"{seen[0].upper()}{seen[1:]}, {so}"
        return Clusters(column=chosen, codes=codes, n_clusters=int(codes.max()) + 1 if n else 0,
                        n_missing=int(missing.sum()), note=note)
    if answer == "repeated" and not _combined(state) and (named is None or named not in frame.columns):
        said = f"by `{named}`, but the table has no such column" if named else \
            "but no column names the unit"
        return Clusters(
            refusal=(f"The rows were said to repeat {said}, so which rows belong together is "
                     f"unknown, and intervals that treat them as independent would be too narrow."),
            exits=({"label": "Name the column that identifies the unit (the grain question)",
                    "decision": None},))
    if passed_over:
        one = len(passed_over) == 1
        shown = ", ".join(f"`{c}`" for c in passed_over[:3])
        return Clusters(note=(f"The grain answer names `{named}` as the unit, so the intervals "
                              f"follow it; {shown} also repeat{'s' if one else ''} and "
                              f"cluster{'s' if one else ''} nothing unless you confirm that rows "
                              f"sharing {'it' if one else 'one'} belong together."))
    if answer == "unknown":
        return Clusters(note=("Whether a unit can appear in more than one row was answered as not "
                              "known, and no identifier repeats, so the intervals assume every row "
                              "is a different unit."))
    return INDEPENDENT


def confirm_exits_for(state: Any, column: str) -> list[dict[str, Any]]:
    """The column's role confirmation, when its role waits for one (the ledger's exits)."""
    from turbotab.core.readings import confirm_exits, unsettled

    return confirm_exits(state, [column]) if column in unsettled(state, [column]) else []


def _combined(state: Any) -> bool:
    """Each unit's rows were combined into one before the analysis (the unit question)."""
    return getattr(state, "unit", None) == "unit" and getattr(state, "aggregation", None) is not None


def floor_refusal(clusters: Clusters, task: str | None = None) -> tuple[str, tuple[dict[str, Any], ...]] | None:
    """Why no interval can be reported for these clusters, and the ways forward; None if one can.

    For a quantity (``task`` regression) the random-intercept mixed model is an exit: its
    variance is model-based, so it does not rest on unit-level residuals (WP12,
    ``turbotab/core/models/repeated.py``)."""
    if clusters.refusal:
        return clusters.refusal, clusters.exits
    if not clusters.clustered or clusters.n_clusters >= min_clusters():
        return None
    column = clusters.column
    combine = {"label": f"Combine each `{column}`'s rows into one (the unit question, before the "
                        f"seal)", "decision": None}
    if task == "regression":
        exits = (combine, {"label": "A random-intercept mixed model, whose intervals do not rest "
                                    "on unit-level residuals",
                           "decision": {"kind": "select_models", "models": ["mixed"]}})
    elif task == "time_to_event":  # one survival time per unit: combining rows is no answer
        exits = ({"label": "A shared-frailty Cox model (not yet in TurboTab)", "decision": None},)
    else:
        exits = (combine, {"label": "A random-intercept logistic model (not yet in TurboTab)",
                           "decision": None})
    return (
        f"`{column}` has {clusters.n_clusters} units, fewer than the {min_clusters()} TurboTab "
        f"requires for cluster-robust intervals: with so few, an interval rests on a handful of "
        f"unit-level residuals and its accuracy depends on how balanced the units are, while one "
        f"that ignores the repetition is far too narrow. No interval or p-value is reported.",
        exits,
    )


# ── CR2 with Bell–McCaffrey degrees of freedom ───────────────────────────────


def cr2(X: np.ndarray, e: np.ndarray, codes: np.ndarray, bread: np.ndarray,
        *, adjust: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """The CR2 covariance of a working linear model, and each coefficient's Bell–McCaffrey df.

    ``X`` (N × P) is the working design and ``e`` (N) the working residuals, both already on the
    scale where the working model is independent rows of equal variance (for OLS, the design and
    the residuals; for a GLM, ``W^½ X`` and the Pearson residuals). ``bread`` is ``(XᵀX)⁻¹``
    (a pseudo-inverse when singular). ``codes`` says which cluster each row is in.

    For each cluster ``A_g = (I − H_gg)^(−½)``, with ``H_gg = X_g (XᵀX)⁻¹ X_gᵀ`` (a pseudo-inverse
    square root when ``I − H_gg`` is singular, as clubSandwich does), and
    ``V = B (Σ_g X_gᵀ A_g e_g e_gᵀ A_g X_g) B``. ``H_gg`` has rank at most P, so ``A_g`` is
    applied through the eigenvectors of its P-dimensional part: no ``n_g × n_g`` matrix is formed.

    The degrees of freedom for coefficient j (:func:`_bell_mccaffrey`, for any linear combination
    of the coefficients): with ``p_g = A_g X_g B e_j`` and the working model's residual-maker
    ``I − H``, the variance estimate is ``Σ_g (u_gᵀ ε)²`` with ``u_g = (I − H) p_g``;
    Satterthwaite's ν = tr(Ω)² / tr(Ω²) for ``Ω_gh = u_gᵀu_h = δ_gh |p_g|² − w_gᵀ B w_h``,
    ``w_g = X_gᵀ p_g``. ``adjust=False`` gives CR0 (``A_g = I``), for checks. The df are None
    when computing them would cost more than :data:`BM_BUDGET` (the caller then uses t(G − 1)).
    """
    V, AX, blocks = _cr2_parts(X, e, codes, bread, adjust=adjust)
    P = np.asarray(X).shape[1]
    if not _bm_affordable(blocks, P):
        return V, None
    return V, _bell_mccaffrey(AX, np.asarray(X, dtype=float), bread, blocks)


def _cr2_parts(X: np.ndarray, e: np.ndarray, codes: np.ndarray, bread: np.ndarray, *,
               adjust: bool = True) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """:func:`cr2`'s covariance, with what its Bell–McCaffrey df are computed from: ``A_g X_g``
    for every cluster (N × P) and the clusters' rows in blocks (:func:`_size_blocks`)."""
    X = np.asarray(X, dtype=float)
    e = np.asarray(e, dtype=float)
    N, P = X.shape
    blocks = _size_blocks(np.asarray(codes), P)
    # Pass 1: A_g X_g for every cluster, kept as one N × P array (the size of the design itself).
    AX = X.copy()
    meat = np.zeros((P, P))
    for block in blocks:
        Xg = X[block]  # (k, m, P)
        if adjust:
            lengths = np.linalg.norm(Xg, axis=2)  # (k, m)
            graded = lengths.min(axis=1) < np.finfo(float).eps * lengths.max(axis=1)
            if graded.any():
                # A cluster with a row under rounding's share of its longest is factored longest
                # row first, Q put back in order: Householder is then accurate row by row
                # (:func:`qr_frame`), where a short row among the first keeps rounding in its Q.
                order = np.argsort(-lengths, axis=1, kind="stable")
                order[~graded] = np.arange(Xg.shape[1])
                Q_sorted, R = np.linalg.qr(np.take_along_axis(Xg, order[:, :, None], axis=1))
                Q = np.empty_like(Q_sorted)
                Q[np.arange(len(order))[:, None], order] = Q_sorted
            else:
                Q, R = np.linalg.qr(Xg)  # X_g = Q R, Q with orthonormal columns
            Bg = R @ bread @ np.swapaxes(R, 1, 2)
            lam, Z = np.linalg.eigh((Bg + np.swapaxes(Bg, 1, 2)) / 2)
            gap = 1.0 - lam
            f = np.where(gap > _EIG_FLOOR, 1.0 / np.sqrt(np.clip(gap, _EIG_FLOOR, None)) - 1.0, -1.0)
            U = Q @ Z  # eigenvectors of H_gg (k, m, r)
            AX[block] = Xg + U @ (f[:, :, None] * (np.swapaxes(U, 1, 2) @ Xg))
        s = np.einsum("kmp,km->kp", AX[block], e[block])
        meat += s.T @ s
    V = bread @ meat @ bread
    return (V + V.T) / 2, AX, blocks


def _bm_affordable(blocks: Sequence[np.ndarray], P: int, contrasts: int | None = None) -> bool:
    """Whether the Bell–McCaffrey df of ``contrasts`` linear combinations (P where None) cost no
    more than :data:`BM_BUDGET`."""
    G = sum(len(b) for b in blocks)
    return G * P ** 2 * (P if contrasts is None else contrasts) <= BM_BUDGET


def _bell_mccaffrey(AX: np.ndarray, X: np.ndarray, bread: np.ndarray, blocks: Sequence[np.ndarray],
                    contrasts: np.ndarray | None = None) -> np.ndarray:
    """The Bell–McCaffrey df (:func:`cr2`) of each coefficient, or of each column of
    ``contrasts`` (P × Q), a linear combination of them: the df depend only on the combination,
    so a coefficient's are the same in any parameterization that holds it."""
    P = X.shape[1]
    # Pass 2: the Satterthwaite pieces, a slice of coefficients at a time (S_j is P × P each).
    # T's row i, column j: p_g's entry for coefficient (or combination) j.
    T = AX @ bread if contrasts is None else AX @ (bread @ np.asarray(contrasts, dtype=float))
    Q = T.shape[1]
    df = np.full(Q, np.nan)
    width = max(1, min(Q, _BATCH // max(1, P * P)))
    for j0 in range(0, Q, width):
        cols = slice(j0, min(Q, j0 + width))
        J = cols.stop - cols.start
        sum_d = np.zeros(J)
        sum_k = np.zeros(J)
        sum_dd = np.zeros(J)
        sum_dk = np.zeros(J)
        S = np.zeros((J, P, P))  # S[j] = Σ_g w_g w_gᵀ
        for block in blocks:
            Tg = T[block][:, :, cols]  # (k, m, J)
            d = np.einsum("kmj,kmj->kj", Tg, Tg)
            Wg = np.swapaxes(X[block], 1, 2) @ Tg  # (k, P, J): w_g = X_gᵀ p_g
            k_gg = np.einsum("kaj,kaj->kj", Wg, bread @ Wg)
            sum_d += d.sum(0)
            sum_k += k_gg.sum(0)
            sum_dd += (d * d).sum(0)
            sum_dk += (d * k_gg).sum(0)
            Wj = np.transpose(Wg, (2, 0, 1))  # (J, k, P)
            S += np.swapaxes(Wj, 1, 2) @ Wj
        BS = bread[None, :, :] @ S
        tr_k2 = np.einsum("jab,jba->j", BS, BS)
        numerator = (sum_d - sum_k) ** 2
        denominator = sum_dd - 2.0 * sum_dk + tr_k2
        with np.errstate(divide="ignore", invalid="ignore"):
            df[cols] = np.where(denominator > 0, numerator / denominator, np.nan)
    return df


def _size_blocks(codes: np.ndarray, P: int) -> list[np.ndarray]:
    """The rows of each cluster as (k, m) index arrays, clusters of one size m together, at most
    :data:`_BATCH` floats of design per array."""
    N = len(codes)
    order = np.argsort(codes, kind="stable")
    ranked = codes[order]
    starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]]) if N else np.zeros(0, int)
    sizes = np.diff(np.r_[starts, N])
    out = []
    for m in np.unique(sizes):
        rows = order[starts[sizes == m][:, None] + np.arange(m)[None, :]]
        per = max(1, _BATCH // max(1, m * P + P * P))
        out.extend(rows[lo:lo + per] for lo in range(0, len(rows), per))
    return out


def scaled_inverse(A: np.ndarray) -> np.ndarray:
    """The pseudo-inverse of a symmetric positive semi-definite matrix (an information, or XᵀX),
    taken on its rows and columns scaled to a unit diagonal and scaled back:
    ``D⁻¹ (D⁻¹ A D⁻¹)⁺ D⁻¹``, with ``D`` the root of its diagonal. Where it is invertible this is
    its inverse; but a pseudo-inverse of ``A`` itself drops every direction below 10⁻¹⁵ of its
    largest eigenvalue, so a column in units 10⁷ or more times another's (a date in epoch seconds
    beside a standard-normal column) had its partner's variance cut to rounding. A zero diagonal
    entry is left at one."""
    A = np.asarray(A, dtype=float)
    d = np.sqrt(np.clip(np.diag(A), 0, None))
    d = np.where((d > 0) & np.isfinite(d), d, 1.0)
    inner = np.linalg.pinv(A / np.outer(d, d))
    return (inner + inner.T) / 2 / np.outer(d, d)


def bread_of(X: np.ndarray) -> np.ndarray:
    """``(XᵀX)⁻¹``, a pseudo-inverse where singular, on the column-scaled design
    (:func:`scaled_inverse`)."""
    return scaled_inverse(X.T @ X)


def qr_frame(X: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The design's QR as ``(Q, R⁻¹)``, ``X = QR`` with Q's columns orthonormal; None where R has a
    zero on its diagonal (the columns dependent). A covariance computed on Q, whose condition
    number is one, and carried to the coefficients by ``R⁻¹ · R⁻ᵀ`` has rounding of the design's
    condition number, where one from ``XᵀX`` has rounding of its square: beside ``x + 10⁻⁹·d``,
    statsmodels' HC3 leverages ``diag(X(XᵀX)⁻¹Xᵀ)`` put `v`'s HC3 standard error off by 2.6 times,
    and the pseudo-inverse of ``XᵀX`` dropped the near-null direction from CR2's (−1.3·10⁻³).

    Where some row's length is under rounding's share of the longest's (a working design whose
    weight rounds to zero on a row: a fitted probability of 0 or 1), the rows are factored longest
    first and Q put back in their order: Householder's QR is accurate row by row on rows so sorted
    (Powell & Reid 1969; Cox & Higham 1998, *BIT* 38:24), where a short row among the first ones
    keeps its row of Q at rounding, 10⁻¹⁷, not at its own scale. Times that row's Pearson residual,
    divided by the root of its weight, that rounding made the logistic CR2 standard errors beside
    one outlying row first in the data (its fitted probability 1.0, no event) 10¹³¹ and more,
    where they are 0.06 to 0.30 (CR2's QR of each cluster sorts its rows so too,
    :func:`_cr2_parts`). Other designs are factored as they come."""
    from scipy.linalg import solve_triangular

    X = np.asarray(X, dtype=float)
    lengths = np.linalg.norm(X, axis=1) if X.size else np.zeros(len(X))
    if len(lengths) and np.isfinite(lengths).all() and \
            lengths.min() < np.finfo(float).eps * lengths.max():
        order = np.argsort(-lengths, kind="stable")
        Q_sorted, R = np.linalg.qr(X[order])
        Q = np.empty_like(Q_sorted)
        Q[order] = Q_sorted
    else:
        Q, R = np.linalg.qr(X)
    if not np.all(np.isfinite(R)) or np.any(np.diag(R) == 0):
        return None
    return Q, solve_triangular(R, np.eye(R.shape[0]))


def gram_inverse(W: np.ndarray) -> np.ndarray | None:
    """``(WᵀW)⁻¹`` from W's QR (:func:`qr_frame`), ``R⁻¹R⁻ᵀ``, whose rounding is W's condition
    number, not its square; None where R is singular. With W a model's working design
    (:func:`logistic_working`, :func:`multinomial_working`), ``WᵀW`` is its information."""
    frame = qr_frame(W)
    if frame is None:
        return None
    Rinv = frame[1]
    V = Rinv @ Rinv.T
    return (V + V.T) / 2


# A row whose leverage is within this of one is one the fit passes through to working precision
# (:func:`hc3_covariance`): its residual and one less its leverage are both rounding, and HC3's
# e/(1 − h) is zero by zero. Above it, e/(1 − h) keeps its rounding under this share.
LEVERAGE_ONE = math.sqrt(np.finfo(float).eps)


def hc3_covariance(X: np.ndarray, e: np.ndarray) -> np.ndarray:
    """HC3's covariance (MacKinnon & White 1985), ``(XᵀX)⁻¹ Xᵀ diag(e²/(1 − h)²) X (XᵀX)⁻¹``,
    from the design's QR (:func:`qr_frame`): the leverages are the rows' squared lengths in Q, and
    the meat is taken on Q and carried by ``R⁻¹``; NaN throughout where R is singular. A row of
    leverage one (:data:`LEVERAGE_ONE`) adds nothing: its HC3 term is zero by zero (R's
    ``sandwich::vcovHC`` gives NaN), and a coefficient that rests on it has no HC3 standard error
    (:func:`resting_on_leverage_one`), while one that does not has the HC3 covariance of the fit
    without that row, whose leverages and residuals are the others' (the hat matrix is that row's
    alone plus theirs). Kept, a single row's indicator was served an HC3 standard error of 6.16,
    from rounding."""
    X = np.asarray(X, dtype=float)
    e = np.asarray(e, dtype=float)
    frame = qr_frame(X)
    if frame is None:
        return np.full((X.shape[1], X.shape[1]), np.nan)
    Q, Rinv = frame
    h = np.einsum("ij,ij->i", Q, Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(h >= 1.0 - LEVERAGE_ONE, 0.0, (e / (1.0 - h)) ** 2)
        V = Rinv @ ((Q * w[:, None]).T @ Q) @ Rinv.T
    return (V + V.T) / 2


def resting_on_leverage_one(X: np.ndarray) -> tuple[Callable[[np.ndarray], bool], int] | None:
    """Whether a combination ``c`` of the coefficients rests on the design's rows of leverage one
    (:data:`LEVERAGE_ONE`): more than ``repeated.IN_DEPENDENCY`` of its classical variance
    ``cᵀ(XᵀX)⁻¹c`` is theirs, ``Σ (cᵀR⁻¹q_i)²`` over their rows ``q_i`` of Q (beside a single row's
    indicator, all of the indicator's and none of the others'), and how many such rows there are;
    None where no row has leverage one or R is singular."""
    from turbotab.core.models.repeated import IN_DEPENDENCY

    frame = qr_frame(np.asarray(X, dtype=float))
    if frame is None:
        return None
    Q, Rinv = frame
    lone = np.einsum("ij,ij->i", Q, Q) >= 1.0 - LEVERAGE_ONE
    if not lone.any():
        return None
    G = Rinv @ Q[lone].T

    def rests(c: np.ndarray) -> bool:
        c = np.asarray(c, dtype=float)
        total = float(np.sum((c @ Rinv) ** 2))
        return float(np.sum((c @ G) ** 2)) > IN_DEPENDENCY * total

    return rests, int(lone.sum())


def leverage_words(n_lone: int, n_rows: int, named: Sequence[str], whole: bool = False) -> str:
    """Why the HC3 coefficients ``named``, which rest on the ``n_lone`` of the ``n_rows`` rows
    whose leverage is one (:func:`resting_on_leverage_one`), have no standard error: refused by
    row, or the whole table where ``whole`` (a coefficient studied among them)."""
    one = n_lone == 1
    lead = (f"{n_lone:,} of these {n_rows:,} rows {'has' if one else 'have'} leverage one: the fit "
            f"passes through {'it' if one else 'them'} exactly, and HC3 divides "
            f"{'its residual' if one else 'their residuals'} by one less "
            f"{'its' if one else 'their'} leverage, zero by zero")
    if whole:
        return (f"{lead}, so {_terms(list(named))} {'has' if len(named) == 1 else 'have'} no HC3 "
                f"standard error. No estimate, interval or p-value is reported.")
    return (f"{lead}, so no estimate, interval or p-value is reported for {_terms(list(named))}, "
            f"which {'rests' if len(named) == 1 else 'rest'} on {'it' if one else 'them'}; the "
            f"other coefficients' HC3 standard errors are those of the other rows.")


def logistic_working(X: np.ndarray, y: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A logistic fit's working linear model at its estimate: ``W^½ X`` and the Pearson residuals
    ``(y − μ)/√w`` (w = μ(1 − μ)). Its ``(XᵀX)⁻¹`` is the inverse information, and its meat is the
    sum of score outer products, so :func:`cr2` gives the logistic sandwich."""
    root = np.sqrt(np.clip(mu * (1.0 - mu), 1e-300, None))
    return X * root[:, None], (y - mu) / root


def multinomial_working(X: np.ndarray, outcome: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A multinomial logit's working linear model: each row becomes q = K − 1 rows,
    ``W_i^½ (I_q ⊗ x_iᵀ)`` and ``W_i^(−½) (y_i − π_i)`` with ``W_i = diag(π_i) − π_i π_iᵀ`` over the
    non-reference classes. Coefficients are ordered class by class (each class's P together).
    ``outcome`` holds class codes 0…K − 1 (0 the reference); ``probs`` is n × K."""
    n, P = X.shape
    q = probs.shape[1] - 1
    pi = probs[:, 1:]
    W = np.einsum("ia,ab->iab", pi, np.eye(q)) - pi[:, :, None] * pi[:, None, :]
    lam, vec = np.linalg.eigh(W)
    lam = np.clip(lam, 1e-300, None)
    half = vec @ (np.sqrt(lam)[:, :, None] * np.swapaxes(vec, 1, 2))
    inv_half = vec @ ((1.0 / np.sqrt(lam))[:, :, None] * np.swapaxes(vec, 1, 2))
    onehot = (np.asarray(outcome)[:, None] == np.arange(1, q + 1)[None, :]).astype(float)
    Xw = np.einsum("iab,ij->iabj", half, X).reshape(n * q, q * P)
    ew = np.einsum("iab,ib->ia", inv_half, onehot - pi).reshape(n * q)
    return Xw, ew


# ── separation and Firth ─────────────────────────────────────────────────────


def separated_columns(X: np.ndarray, y: np.ndarray, *, first: bool = False) -> list[int]:
    """Indices of the coefficients whose maximum-likelihood estimate is infinite; [] when none.

    Konis (2007): the data are separated when some β ≠ 0 has ``s_i x_iᵀβ ≥ 0`` for every row
    (``s_i = ±1`` for an event or not) with ``Xβ ≠ 0``. With every column scaled to at most one in
    absolute value and β boxed in [−1, 1], one linear program maximizing ``Σ s_i x_iᵀβ`` finds
    such a direction when one exists; coefficient j is infinite when some direction in that cone
    moves it (one linear program each way). Every direction a solver returns is checked again on
    the rows before it is believed. With a rank-deficient design, directions with ``Xβ = 0`` make
    the per-coefficient reading undefined, so nothing is reported (the collinearity concern says
    why the table cannot be computed). The cone is convex, so one separating direction moves every
    coefficient named, and none other: with all the named columns left out (the intercept kept,
    and both outcomes present), no direction separates what is left. With ``first``, only whether
    any is infinite is asked: the first one found alone.
    """
    from scipy.optimize import linprog

    X = np.asarray(X, dtype=float)
    s = np.where(np.asarray(y, dtype=float) > 0.5, 1.0, -1.0)
    n, P = X.shape
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    if n <= P or np.linalg.matrix_rank(Xs) < P:
        return []
    A = -(s[:, None] * Xs)
    b = np.zeros(n)
    bounds = [(-1.0, 1.0)] * P

    def direction(c: np.ndarray) -> np.ndarray | None:
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if result.status != 0 or result.x is None:
            return None
        beta = np.where(np.abs(result.x) < 1e-9, 0.0, result.x)
        margins = s * (Xs @ beta)
        return beta if margins.min() >= -1e-7 and margins.max() > 1e-6 else None

    if direction(-(s @ Xs)) is None:
        return []
    infinite = []
    for j in range(P):
        for sign in (1.0, -1.0):
            c = np.zeros(P)
            c[j] = -sign
            beta = direction(c)
            if beta is not None and sign * beta[j] > 1e-6:
                infinite.append(j)
                break
        if first and infinite:
            break
    return infinite


def _infinite_in_cone(M: np.ndarray, check: int, order: np.ndarray | None = None) -> list[int]:
    """:func:`separated_columns`' linear programs on any likelihood whose rows' terms stay away
    from zero along a direction ``z`` exactly where each margin ``M_r z`` is at least zero (and
    each of ``order``'s rows, which keep the cut-points in order, is too): with ``z`` boxed in
    [−1, 1], one program maximizing the margins' sum finds a direction with some margin positive,
    along which the likelihood rises without end, where one exists; each of the first ``check``
    entries of ``z`` that some such direction moves (one program each way) is infinite. Every
    direction a solver returns is checked again on the margins before it is believed."""
    from scipy.optimize import linprog

    V = M.shape[1]
    A = -M if order is None else -np.vstack([M, order])
    b = np.zeros(A.shape[0])
    bounds = [(-1.0, 1.0)] * V

    def direction(c: np.ndarray) -> np.ndarray | None:
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if result.status != 0 or result.x is None:
            return None
        z = np.where(np.abs(result.x) < 1e-9, 0.0, result.x)
        margins = M @ z
        ordered = order is None or (order @ z).min() >= -1e-7
        return z if ordered and margins.min() >= -1e-7 and margins.max() > 1e-6 else None

    if direction(-M.sum(axis=0)) is None:
        return []
    infinite = []
    for j in range(check):
        for sign in (1.0, -1.0):
            c = np.zeros(V)
            c[j] = -sign
            z = direction(c)
            if z is not None and sign * z[j] > 1e-6:
                infinite.append(j)
                break
    return infinite


def _boxed(X: np.ndarray) -> np.ndarray | None:
    """``X`` with every column scaled to at most one in absolute value, or None where its rows do
    not exceed its columns or it is rank-deficient (:func:`separated_columns`' reading)."""
    X = np.asarray(X, dtype=float)
    n, P = X.shape
    scale = np.abs(X).max(axis=0) if n else np.ones(P)
    scale[scale == 0] = 1.0
    Xs = X / scale
    if n <= P or np.linalg.matrix_rank(Xs) < P:
        return None
    return Xs


# Why a Firth profile quantity is not reported where the data separate (:func:`separated_columns`,
# complete or quasi-complete): held at a value, the penalized likelihood there can have more than
# one maximum, and a warm-started fit can stop on a lower one, so a bound is found inside the
# interval and a p-value overstated. Under complete separation `xs`'s upper bound beside
# ``sign(y)·(0.3 + |d|)`` was served at 16.53, where a point held there has a ratio of 3.02
# (T5aV2-1); with one row of the other outcome tied to the row nearest the cut (quasi-complete) at
# 19.04, where it has 3.28 (V5c2-1), each under χ²₁(0.95) = 3.84. A multi-start certification is a
# feature for later; until then nothing profiled is served on separated rows.
SEVERAL_MAXIMA = ("Firth's penalized likelihood can then have more than one maximum, and the fit "
                  "cannot certify that it found the highest")


def separated_profile_words(separated: Sequence[str], *,
                            held: str = "a coefficient held at a value",
                            unreported: str = "profile interval or p-value") -> str:
    """Why a Firth table on rows the data separate reports its penalized estimates alone (or a test
    on one, ``held`` and ``unreported`` its own words), naming the separated columns: the cone
    :func:`separated_columns` finds, one direction of which moves every one of them, so a cut on
    a combination of them puts every row on its outcome's side or on the cut, and all of which
    left out ends the separation. It is the table's refusal, which other surfaces quote where no
    estimate is reported either, so it says nothing of the estimates; the table's concern says
    that it reports Firth's fit."""
    one = len(separated) == 1
    cut = "a cut on it" if one else "a cut on a combination of them"
    return (f"{_and(separated)} {'separates' if one else 'separate'} the outcome: {cut} puts every "
            f"row on its outcome's side or on the cut itself. With {held}, {SEVERAL_MAXIMA}, so no "
            f"{unreported} is reported.")


def separated_multinomial(X: np.ndarray, codes: np.ndarray, K: int) -> list[int]:
    """The columns of ``X`` (its intercept among them) some class's maximum-likelihood
    coefficient on which is infinite in a multinomial logit against class 0; [] when none.

    The log-likelihood ``Σ_i log π_{i,y_i}`` rises without end along coefficients ``D`` (one
    column of ``X``'s length per class, class 0's zero) exactly where every row's own class has
    the largest linear predictor along them, ``x_iᵀ(D_{y_i} − D_k) ≥ 0`` for every other class k,
    and some row's strictly (Albert & Anderson 1984, *Biometrika* 71:1, for any number of
    classes): :func:`separated_columns`' linear programs on those margins
    (:func:`_infinite_in_cone`). A column whose coefficient for some class is moved is named."""
    Xs = _boxed(X)
    if Xs is None:
        return []
    n, P = Xs.shape
    q = K - 1
    codes = np.asarray(codes, dtype=np.int64)
    rows = []
    for k in range(K):  # x_iᵀ(D_{y_i} − D_k), D class-major over classes 1…K − 1
        other = codes != k
        if not other.any():
            continue
        M = np.zeros((int(other.sum()), q * P))
        own, Xo = codes[other], Xs[other]
        for c in range(1, K):
            M[own == c, (c - 1) * P:c * P] = Xo[own == c]
        if k > 0:
            M[:, (k - 1) * P:k * P] -= Xo
        rows.append(M)
    found = _infinite_in_cone(np.vstack(rows), q * P) if rows else []
    return sorted({j % P for j in found})


def separated_ordinal(X: np.ndarray, codes: np.ndarray) -> list[int]:
    """The columns of ``X`` whose maximum-likelihood coefficient in a proportional-odds
    (cumulative logit) model ``P(y ≤ j) = σ(θ_j − xᵀβ)`` is infinite; [] when none. Along
    ``(β, θ) + s·(b, t)`` a row of level j keeps its probability away from zero exactly where
    ``t_j − x_iᵀb ≥ 0`` (below the top level) and ``x_iᵀb − t_{j−1} ≥ 0`` (above the bottom), with
    the cut-points kept in order (``t_j ≥ t_{j−1}``); the likelihood rises without end where some
    row's margin is strict: :func:`separated_columns`' linear programs on those margins
    (:func:`_infinite_in_cone`), the levels the codes 0…K − 1 in order."""
    Xs = _boxed(X)
    if Xs is None:
        return []
    n, P = Xs.shape
    codes = np.asarray(codes, dtype=np.int64)
    K = int(codes.max()) + 1 if n else 0
    if K < 2:
        return []
    q = K - 1
    rows = []
    for x, j in zip(Xs, codes):
        if j < q:  # t_j − xᵀb
            m = np.concatenate([-x, np.zeros(q)])
            m[P + j] = 1.0
            rows.append(m)
        if j > 0:  # xᵀb − t_{j−1}
            m = np.concatenate([x, np.zeros(q)])
            m[P + j - 1] = -1.0
            rows.append(m)
    order = np.zeros((max(q - 1, 0), P + q))
    for j in range(1, q):
        order[j - 1, P + j], order[j - 1, P + j - 1] = 1.0, -1.0
    return _infinite_in_cone(np.array(rows), P, order if q > 1 else None)


def separation_words(separated: Sequence[str], what: str, among: str = "") -> str:
    """Why a multinomial or proportional-odds table that :func:`separated_multinomial` or
    :func:`separated_ordinal` finds separated reports no estimate: the columns named separate the
    outcome's ``what`` (classes, levels) ``among`` (the weighted rows), so a maximum-likelihood
    coefficient is infinite."""
    tail = ("(a maximum-likelihood coefficient is infinite), so no estimate, interval or p-value "
            "is reported.")
    if not separated:
        return f"The outcome's {what} are separated{among} {tail}"
    verb = "separates" if len(separated) == 1 else "separate"
    return f"{_and(separated)} {verb} the outcome's {what}{among} {tail}"


def separation_exits(separated: Sequence[str]) -> tuple[dict[str, Any], ...]:
    """The way out of a separated table: leave its separating columns out, or merge their rare
    levels."""
    named = _and(separated) if separated else "the separating column"
    return ({"label": f"Leave {named} out, or merge its rare levels", "decision": None},)


@dataclass
class FirthFit:
    beta: np.ndarray
    loglik: float  # the penalized log-likelihood at beta
    cov: np.ndarray  # the inverse Fisher information at beta
    iterations: int
    converged: bool


def _firth_parts(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> tuple[Any, ...]:
    """At ``beta``: the fitted probabilities, their weights ``w = μ(1 − μ)``, the QR of the
    weighted design ``√w·X = QR`` (``RᵀR`` the Fisher information), and the penalized
    log-likelihood ``l(β) + ½ log |I(β)|``, the penalty ``Σ log |R_jj|``: from the QR, its rounding
    is the design's condition number, where a determinant of the information has that squared."""
    eta = X @ beta
    mu = 0.5 * (1.0 + np.tanh(0.5 * eta))  # the logistic function, without overflow
    w = mu * (1.0 - mu)
    Q, R = np.linalg.qr(X * np.sqrt(w)[:, None])
    diagonal = np.abs(np.diag(R))
    loglik = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    penalized = (loglik + float(np.sum(np.log(diagonal)))
                 if np.all(np.isfinite(R)) and np.all(diagonal > 0) else -np.inf)
    return mu, w, Q, R, penalized


def _newton_step(R: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, float]:
    """The Newton step ``(RᵀR)⁻¹ s`` and its decrement ``sᵀ(RᵀR)⁻¹s = |R⁻ᵀs|²`` from the triangle of
    the weighted design's QR (two triangular solves); from a least-squares solve on ``RᵀR`` where
    R is singular."""
    from scipy.linalg import solve_triangular

    if R.shape[0] == 0:
        return np.zeros(0), 0.0
    if not np.all(np.isfinite(R)) or np.any(np.diag(R) == 0):
        step = np.linalg.lstsq(R.T @ R, score, rcond=None)[0]
        return step, float(score @ step)
    z = solve_triangular(R, score, trans="T")
    return solve_triangular(R, z), float(z @ z)


def _inverse_information(R: np.ndarray) -> np.ndarray:
    """``(RᵀR)⁻¹ = R⁻¹R⁻ᵀ`` from the QR's triangle (:func:`gram_inverse`'s form); a pseudo-inverse
    of ``RᵀR`` (:func:`scaled_inverse`) where R is singular."""
    from scipy.linalg import solve_triangular

    if R.shape[0] == 0:
        return np.zeros((0, 0))
    if not np.all(np.isfinite(R)) or np.any(np.diag(R) == 0):
        return scaled_inverse(R.T @ R)
    Rinv = solve_triangular(R, np.eye(R.shape[0]))
    V = Rinv @ Rinv.T
    return (V + V.T) / 2


def firth_fit(X: np.ndarray, y: np.ndarray, *, fixed: dict[int, float] | None = None,
              start: np.ndarray | None = None, max_iter: int = 300, tol: float = 1e-8) -> FirthFit:
    """Firth's bias-reduced logistic regression: the maximum of ``l(β) + ½ log |I(β)|``.

    Newton steps on the modified score ``Xᵀ(y − μ + h (½ − μ))`` (``h`` the hat values of the
    weighted fit), which is the exact gradient of the penalized log-likelihood, with steps capped
    at :data:`FIRTH_MAX_STEP` and halved until the penalized log-likelihood does not fall
    (logistf's algorithm). ``fixed`` holds coefficients at given values (the profile). Converged
    when the Newton decrement ``Uᵀ I⁻¹ U`` (invariant to the columns' units) is below ``tol``²,
    and only then. The step is the Fisher information's, not the penalized log-likelihood's
    Hessian, so near the maximum each step leaves a fixed share of the distance (its rate), and
    the decrement understates the distance left by about ``1/(1 − rate)``: a stop where the
    decrement fell by less than half in a step (read as rounding's floor) at most
    :data:`SHORT_OF_MAXIMUM` squared served completely separated designs, whose decrement falls
    to 0.52 to 0.72 of itself a step, up to 5.9·10⁻⁴ of a standard error from brglm2's maximum,
    where this rule serves them within 6·10⁻⁸. A fit that rounding keeps above ``tol``² runs out
    its steps and is refused, not converged. The step, the decrement, the hat values and the
    penalty are taken from the QR of the weighted design (:func:`_firth_parts`): from the
    information itself, whose rounding is the design's condition number squared, the decrement
    beside ``x + 3·10⁻⁷·d`` passed its tolerance far from the maximum, and the table served `v`'s
    penalized log-odds as −5·10⁻⁶ against the maximum's −0.353.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    P = X.shape[1]
    beta = np.zeros(P) if start is None else np.array(start, dtype=float)
    free = np.ones(P, dtype=bool)
    for j, value in (fixed or {}).items():
        free[j] = False
        beta[j] = value
    mu, w, Q, R, ll = _firth_parts(X, y, beta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        h = np.einsum("ij,ij->i", Q, Q)
        score = X.T @ (y - mu + h * (0.5 - mu))
        held = R if free.all() else np.linalg.qr((X * np.sqrt(w)[:, None])[:, free], mode="r")
        delta = np.zeros(P)
        delta[free], decrement = _newton_step(held, score[free])
        if decrement < tol * tol:
            converged = True
            break
        biggest = float(np.abs(delta).max()) if P else 0.0
        if biggest > FIRTH_MAX_STEP:
            delta *= FIRTH_MAX_STEP / biggest
        for _ in range(40):
            trial = beta + delta
            parts = _firth_parts(X, y, trial)
            if parts[4] >= ll - 1e-12 * max(1.0, abs(ll)):
                break
            delta /= 2.0
        beta = trial
        mu, w, Q, R, ll = parts
    return FirthFit(beta=beta, loglik=float(ll), cov=_inverse_information(R), iterations=it,
                    converged=converged)


# The most penalized Newton steps one coefficient's profile may take (:func:`firth_profile`,
# ``strict``), over every fit its bounds and its p-value make: a converged profile takes some
# 150; beside a near copy each penalized fit could run its 300 steps unconverged, for every
# doubling of the search and every step of the root-finder (the verifier timed one table at 215 s).
FIRTH_PROFILE_STEPS = 5_000


class FirthProfileFailed(ValueError):
    """A Firth profile (:func:`firth_profile`, ``strict``) that could not be computed: a
    penalized fit along it did not converge, its bound could not be bracketed, or it ran past
    :data:`FIRTH_PROFILE_STEPS`; or one refused because the data separate
    (:class:`FirthProfileSeparated`). Its message says which, in the words a table refuses in."""


class FirthProfileSeparated(FirthProfileFailed):
    """A Firth profile quantity (an interval, a likelihood-ratio p-value or test) asked for on
    rows the data separate (:func:`separated_columns`), refused whoever asks, strict or not: held
    at a value the penalized likelihood can then have more than one maximum, and the fit cannot
    certify that it found the highest (:data:`SEVERAL_MAXIMA`)."""


def is_separated(X: np.ndarray, y: np.ndarray) -> bool:
    """Whether the data separate the outcome (:func:`separated_columns` finds some coefficient
    infinite), its linear programs stopped at the first one found."""
    return bool(separated_columns(np.asarray(X, dtype=float), np.asarray(y, dtype=float),
                                  first=True))


def firth_profile(X: np.ndarray, y: np.ndarray, fit: FirthFit, j: int, *,
                  level: float = LEVEL, strict: bool = False, name: str | None = None
                  ) -> tuple[float | None, float | None, float]:
    """Coefficient j's profile penalized-likelihood interval and likelihood-ratio p-value, on rows
    nothing separates.

    The interval is every b with ``2 (l*(β̂) − max_{β: β_j = b} l*(β)) ≤ χ²₁(level)``
    (Heinze & Schemper 2002); a bound that is not reached within a wide search is None. On rows
    the data separate (:func:`is_separated`) it raises :class:`FirthProfileSeparated`, strict or
    not, in words that name the coefficient (``name``): held at a value the penalized likelihood
    can then have more than one maximum, and the warm-started fits here can stop on a lower one.
    With ``strict`` (a coefficient table's), a profile that cannot be computed raises
    :class:`FirthProfileFailed` in words that name it, rather than give a bound or a p-value from
    a penalized fit that did not converge: one that did not, a bound the search cannot bracket
    (whose solver raised an untyped ``ValueError``), or a profile past
    :data:`FIRTH_PROFILE_STEPS` Newton steps.
    """
    from scipy.optimize import brentq
    from scipy.stats import chi2

    term = _term(name) if name else "this coefficient"
    reported = "so no interval or p-value is reported"
    if is_separated(X, y):
        raise FirthProfileSeparated(
            f"Firth's profile of {term} is not computed: the data separate the outcome. With a "
            f"coefficient held at a value, {SEVERAL_MAXIMA}, {reported}.")
    crit = 0.5 * float(chi2.ppf(level, 1))
    centre = float(fit.beta[j])
    warm = {"beta": fit.beta.copy()}
    spent = {"steps": 0}
    profiled = f"Firth's profile of {term} could not be computed"

    def drop(b: float) -> float:
        profile = firth_fit(X, y, fixed={j: b}, start=warm["beta"])
        spent["steps"] += profile.iterations
        if strict and not profile.converged:
            raise FirthProfileFailed(
                f"{profiled}: the penalized fit with it held at {b:.4g} did not converge in "
                f"{profile.iterations:,} Newton steps, {reported}.")
        if strict and spent["steps"] > FIRTH_PROFILE_STEPS:
            raise FirthProfileFailed(
                f"{profiled} within {FIRTH_PROFILE_STEPS:,} penalized Newton steps, {reported}.")
        warm["beta"] = profile.beta
        return fit.loglik - profile.loglik

    se = math.sqrt(max(float(fit.cov[j, j]), 1e-12))
    bounds: list[float | None] = []
    for side in (-1.0, 1.0):
        warm["beta"] = fit.beta.copy()
        inner, reach = centre, 2.0 * se
        bound = None
        for _ in range(40):
            outer = centre + side * reach
            if drop(outer) >= crit:
                low, high = min(inner, outer), max(inner, outer)
                try:
                    bound = brentq(lambda b: drop(b) - crit, low, high, xtol=1e-10, rtol=1e-10)
                except FirthProfileFailed:
                    raise
                except ValueError:
                    if not strict:
                        raise
                    raise FirthProfileFailed(
                        f"{profiled}: its penalized likelihood ratio did not cross its bound "
                        f"between {low:.4g} and {high:.4g}, {reported}.") from None
                break
            inner, reach = outer, reach * 2.0
        bounds.append(bound)
    warm["beta"] = fit.beta.copy()
    statistic = max(0.0, 2.0 * drop(0.0))
    return bounds[0], bounds[1], float(chi2.sf(statistic, 1))


@contextmanager
def for_pooling() -> Iterator[None]:
    """While open, each table made is one completed copy's, for Rubin's rules
    (``imputation.pool_rows``, ``imputation.pool_scalar``), which read a row's estimate, standard
    error and df and make the interval and p-value themselves: a Firth table then gives each
    coefficient its estimate and standard error only, with no profile (each bound a search of
    penalized fits) for the pooling to discard. That holds for the copy's coefficient table and
    for each refit its form tests make (``exposure_form``'s trend and companion quintiles, which
    read a row's estimate and standard error where it has no p-value, :func:`pooling`). Its
    ``covariance`` stays ``profile``, so the joint form tests on it are still the penalized
    likelihood-ratio tests (``exposure_form.exposure_tests``), pooled by D2."""
    token = _POOLING.set(True)
    try:
        yield
    finally:
        _POOLING.reset(token)


def pooling() -> bool:
    """Whether the table being made is one completed copy's, for Rubin's rules
    (:func:`for_pooling`): a Firth row then has its estimate and standard error and no interval or
    p-value, which a refit reading a row (``exposure_form``'s trend and companion) allows for."""
    return _POOLING.get()


# ── the table ────────────────────────────────────────────────────────────────


@dataclass
class InferenceTable:
    """The coefficient rows, how their intervals were made (the ``inference`` artifact field), and
    the concerns the fit states first."""

    rows: list[dict[str, Any]]
    info: dict[str, Any]
    concerns: list[str] = field(default_factory=list)
    # The covariance the intervals rest on, over the rows in order (None when refused, and in one
    # completed copy's Firth table, whose rows hold an estimate and its standard error for Rubin's
    # rules). Joint tests read it (WP12a: a spline's
    # nonlinearity, ``turbotab/core/methods/exposure_form.py``); it is never serialized.
    cov: Any = None
    # The rows whose maximum-likelihood value the linear program finds infinite on this table's
    # matrix (``separated_columns``): the separated columns, the intercept where it is among them,
    # and an all-components relative effect whose refit finds it so
    # (``stages.modeling.relative_refit``). Rubin's rules withhold their pooled intervals
    # (:func:`pooled_firth`); never serialized.
    infinite: list[str] = field(default_factory=list)
    # The rows refused by row because their coefficients lie in a dependency of the model
    # matrix's columns (:func:`by_identified`), and the words that say so (also the first
    # concern); never serialized.
    unidentified: list[str] = field(default_factory=list)
    dependency: str | None = None
    # The rows of an HC3 table refused by row because their coefficients rest on rows of leverage
    # one (:func:`resting_on_leverage_one`), and how many such rows there are; never serialized.
    leverage_one: list[str] = field(default_factory=list)
    leverage_rows: int = 0
    # A linear combination ``c`` of the coefficients, at its estimate, as a row's fields (``se``,
    # ``df``, ``ci_low``, ``ci_high``, ``p``, and ``infinite`` where separation is checked), for a
    # table whose rows do not follow from its covariance and one df: CR2's Bell–McCaffrey df,
    # Firth's separation check, a multinomial table's covariance not kept. :func:`by_identified`
    # reads it for an intercept or cut-points on centered columns and drops it; never serialized.
    contrast: Callable[[np.ndarray, float], dict[str, Any]] | None = None
    # A Firth table on rows the data separate, on independent rows and at its penalized maximum,
    # whose estimates are served and every profile interval and p-value refused: the words it
    # refuses them in (its ``info["refused"]``, :func:`separated_profile_words`). A caller that
    # reads its estimates (the modification's Firth family) reads it; never serialized.
    profile_refusal: str | None = None


def _clean(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _row(name: str, estimate: float, se: float | None = None, df: float | None = None,
         low: float | None = None, high: float | None = None, p: float | None = None) -> dict[str, Any]:
    return {"feature": name, "estimate": _clean(estimate), "ci_low": None if low is None else _clean(low),
            "ci_high": None if high is None else _clean(high), "p": None if p is None else _clean(p),
            "se": None if se is None else _clean(se), "df": None if df is None else _clean(df)}


def _t_rows(names: Sequence[str], est: np.ndarray, se: np.ndarray, df: np.ndarray) -> list[dict[str, Any]]:
    """Rows with t intervals and two-sided p-values on each coefficient's own df."""
    from scipy import stats

    rows = []
    for name, b, s, v in zip(names, est, se, df):
        if not (np.isfinite(s) and s > 0 and np.isfinite(v) and v > 0):
            rows.append(_row(name, b, s if np.isfinite(s) else None))
            continue
        q = float(stats.t.ppf(0.5 + LEVEL / 2, v))
        p = float(2.0 * stats.t.sf(abs(b / s), v))
        rows.append(_row(name, b, s, v, b - q * s, b + q * s, p))
    return rows


def _z_rows(names: Sequence[str], est: np.ndarray, se: np.ndarray) -> list[dict[str, Any]]:
    from scipy import stats

    q = float(stats.norm.ppf(0.5 + LEVEL / 2))
    rows = []
    for name, b, s in zip(names, est, se):
        if not (np.isfinite(s) and s > 0):
            rows.append(_row(name, b, None))
            continue
        rows.append(_row(name, b, s, None, b - q * s, b + q * s, float(2.0 * stats.norm.sf(abs(b / s)))))
    return rows


def no_interval(names: Sequence[str], se: Any, df: Any = None, *, fit: str,
                df_kind: str | None = None) -> str | None:
    """Why a table reports no interval where some coefficient's standard error does not come out
    finite and positive (a covariance that is non-finite, unavailable or singular in its
    direction), or, for t intervals (``df`` given), its degrees of freedom do not; None where every
    one does. :func:`_t_rows` and :func:`_z_rows` give such a row its estimate and no interval, and
    a number is right or it is refused: the table builder refuses the whole table in these words
    rather than serve the estimate beside a missing interval. ``fit`` names what was computed (the
    covariance, or the fit); ``df_kind`` the degrees of freedom (Satterthwaite, Bell–McCaffrey)."""
    se = np.asarray(se, dtype=float)
    usable = np.isfinite(se) & (se > 0)
    no_se = [str(n) for n, ok in zip(names, usable) if not ok]
    no_df: list[str] = []
    if df is not None:
        df = np.asarray(df, dtype=float)
        no_df = [str(n) for n, ok, v in zip(names, usable, df)
                 if ok and not (np.isfinite(v) and v > 0)]
    if not no_se and not no_df:
        return None
    parts = []
    if no_se:
        parts.append(f"{_terms(no_se)} no finite, positive standard error")
    if no_df:
        kind = f"{df_kind} " if df_kind else ""
        parts.append(f"{_terms(no_df)} a standard error but no finite, positive {kind}degrees of "
                     f"freedom")
    return f"{fit} gives {', and '.join(parts)}, so no interval or p-value is reported."


def _terms(names: Sequence[str]) -> str:
    """The coefficients named, the first :data:`NAMED` of them where there are more."""
    listed = [_term(n) for n in names[:NAMED]]
    rest = len(names) - len(listed)
    if rest:
        return f"{', '.join(listed)} and {rest:,} more coefficient{'s' if rest != 1 else ''}"
    return listed[0] if len(listed) == 1 else f"{', '.join(listed[:-1])} and {listed[-1]}"


def _info(estimator: str, covariance: Covariance, caption: str, clusters: Clusters,
          **extra: Any) -> dict[str, Any]:
    return {"estimator": estimator, "covariance": covariance, "caption": caption,
            "grouped_by": clusters.column, "n_clusters": clusters.n_clusters if clusters.clustered else None,
            "n_missing_ids": clusters.n_missing, "separated": [], "refused": None, "exits": [], **extra}


def _cluster_caption(clusters: Clusters, bell_mccaffrey: bool = True) -> str:
    missing = (f", {clusters.n_missing:,} rows with no `{clusters.column}` counted as units of their "
               f"own" if clusters.n_missing else "")
    reference = ("on t with Bell–McCaffrey degrees of freedom" if bell_mccaffrey else
                 f"on t({clusters.n_clusters - 1:,}): Bell–McCaffrey degrees of freedom are too costly "
                 f"to compute at this size")
    return (f"95% intervals cluster-robust (CR2) by `{clusters.column}`, G = "
            f"{clusters.n_clusters:,} clusters{missing}, {reference}.")


def _clustered(names: Sequence[str], est: np.ndarray, estimator: str, X: np.ndarray, e: np.ndarray,
               codes: np.ndarray, clusters: Clusters) -> InferenceTable:
    """The CR2 table of a working linear model (``X``, ``e``), rows clustered by ``codes``; a
    linear combination of its coefficients gets its own Bell–McCaffrey df
    (:attr:`InferenceTable.contrast`)."""
    X = np.asarray(X, dtype=float)
    P = X.shape[1]
    frame = qr_frame(X)
    if frame is None:  # dependent columns: the pseudo-inverse's covariance, as before
        bread, carry, Z = bread_of(X), np.eye(P), X
    else:  # on Q, carried to the coefficients by R⁻¹ (:func:`qr_frame`)
        Z, carry = frame
        bread = np.eye(P)
    V, AX, blocks = _cr2_parts(Z, e, codes, bread)
    V = carry @ V @ carry.T
    V = (V + V.T) / 2
    exact = _bm_affordable(blocks, P)
    # A coefficient's df are those of its combination on Z's coefficients (``carry``'s row).
    df = (_bell_mccaffrey(AX, Z, bread, blocks, carry.T) if exact else
          np.full(len(est), float(clusters.n_clusters - 1)))
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    reason = no_interval(names, se, df, fit=f"The cluster-robust (CR2) covariance by "
                                            f"`{clusters.column}`",
                         df_kind="Bell–McCaffrey" if exact else None)
    if reason:
        return _refused(names, est, estimator, clusters, reason, ())
    rows = _t_rows(names, est, se, df)
    table = InferenceTable(rows, _info(estimator, "CR2", _cluster_caption(clusters, exact),
                                       clusters), _cluster_concerns(clusters), cov=V)
    if exact:
        def contrast(c: np.ndarray, estimate: float) -> dict[str, Any]:
            nu = _bell_mccaffrey(AX, Z, bread, blocks,
                                 (carry.T @ np.asarray(c, dtype=float))[:, None])[0]
            return combination_row(estimate, math.sqrt(max(float(c @ V @ c), 0.0)), nu)

        table.contrast = contrast
    return table


def _cluster_concerns(clusters: Clusters) -> list[str]:
    out = [clusters.note] if clusters.note else []
    G = clusters.n_clusters
    if clusters.note is None:
        out.append(f"Intervals are cluster-robust by `{clusters.column}` (CR2, G = {G:,}), because "
                   f"its rows repeat.")
    if G < FEW_CLUSTERS:
        out.append(f"Only {G} `{clusters.column}` clusters: the intervals use the CR2 small-sample "
                   f"correction with Bell–McCaffrey degrees of freedom, which Cameron & Miller "
                   f"(2015) recommend when clusters are few; for a continuous outcome, a "
                   f"random-intercept mixed model is the usual alternative.")
    return out


def _refused(names: Sequence[str], est: np.ndarray, estimator: str, clusters: Clusters,
             reason: str, exits: Sequence[dict[str, Any]]) -> InferenceTable:
    rows = [_row(n, b) for n, b in zip(names, est)]
    info = _info(estimator, "none", f"No intervals: {reason}", clusters,
                 refused=reason, exits=[dict(e) for e in exits])
    return InferenceTable(rows, info, [reason])


def format_p(p: float) -> str:
    """A p-value as a reader writes it: 0.004, or 3 × 10⁻⁸ below 0.001."""
    if p >= 0.001:
        return f"{p:.3f}"
    if p <= 0:
        return "< 10⁻³⁰⁰"
    exponent = math.floor(math.log10(p))
    mantissa = p / 10 ** exponent
    sup = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa:.0f} × 10{sup}"


def heteroskedasticity_concern(resid: np.ndarray, X: np.ndarray, names: Sequence[str],
                               classical: np.ndarray, robust: np.ndarray) -> str | None:
    """A sentence when the residual spread changes with the predictors and that matters here:
    Koenker's studentized Breusch–Pagan test below :data:`HETERO_ALPHA`, and classical standard
    errors off from the HC3 ones by at least :data:`HETERO_MATERIAL` for some coefficient."""
    from statsmodels.stats.diagnostic import het_breuschpagan

    if X.shape[1] < 2:
        return None
    try:
        _, p, _, _ = het_breuschpagan(resid, X, robust=True)
    except Exception:  # noqa: BLE001 - a check that cannot run is not a verdict
        return None
    if not np.isfinite(p) or p >= HETERO_ALPHA:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.asarray(classical, float) / np.asarray(robust, float)
    keep = [j for j, n in enumerate(names) if n != "(intercept)" and np.isfinite(ratio[j]) and ratio[j] > 0]
    if not keep:
        return None
    j = max(keep, key=lambda i: abs(math.log(ratio[i])))
    change = ratio[j] - 1.0
    if abs(change) < HETERO_MATERIAL:
        return None
    return (f"The residual spread is not constant (Breusch–Pagan p = {format_p(float(p))}): "
            f"classical standard errors would be {abs(change):.0%} "
            f"{'larger' if change > 0 else 'smaller'} for `{names[j]}`; these intervals use HC3 "
            f"robust standard errors, which allow for it.")


@dataclass(frozen=True)
class Outcome:
    """What the outcome and its levels are called, so a table's scale names them.

    ``labels`` maps a class as the model holds it to its level as the data spell it: a binary
    outcome whose event the user named is coded 1 for that level and 0 for the other
    (``turbotab/core/stages/modeling.py::coded_outcome``), so the model's classes are 0 and 1 and
    only this map says which level each is.
    """

    name: str | None = None
    labels: Mapping[Any, Any] | None = None

    def level(self, value: Any) -> str:
        labels = self.labels or {}
        for key in (value, str(value)):
            if key in labels:
                return str(labels[key])
        return str(value)


def _tick(value: Any) -> str:
    return f"`{value}`"


def _ratio(value: float | None) -> float | None:
    """exp(value), or None when it is missing or does not fit in a float."""
    if value is None or not math.isfinite(value) or value > 700:
        return None
    return math.exp(value)


def _on_scale(table: InferenceTable, task: str, classes: Sequence[Any] | None,
              outcome: Outcome | None) -> InferenceTable:
    """Name the table's scale (``info``: scale, axis, effect, event, reference) and, for a ratio
    scale, add each input's ratio and its interval to its row. The intercept is a baseline, not a
    ratio between two values of an input, so its row carries none."""
    outcome = outcome or Outcome()
    who = _tick(outcome.name) if outcome.name else "the outcome"
    per = "per unit of each input, holding the others"
    if task == "regression":
        table.info.update(scale="difference", axis="linear", event=None, reference=None,
                          effect=f"Difference in mean {who} {per}.")
        return table
    labels = list(classes or [])
    reference = outcome.level(labels[0]) if labels else None
    if task == "time_to_event":
        # WP12b × WP8 (repair round): a Cox table's rows are log hazard ratios, drawn on a log
        # axis like every other ratio scale; the event is the level coded 1.
        event = outcome.level(labels[1]) if len(labels) > 1 else None
        what = f"{who} being {_tick(event)}" if event is not None else f"the event in {who}"
        table.info.update(scale="hazard_ratio", axis="log", event=event, reference=None,
                          effect=f"Hazard ratio of {what} at any time during follow-up, {per}.")
        lead = "Estimates are log hazard ratios; each hazard ratio is exp(estimate)."
    elif task == "binary":
        event = outcome.level(labels[1]) if len(labels) > 1 else None
        table.info.update(scale="odds_ratio", axis="log", event=event, reference=reference,
                          effect=f"Odds ratio of {who} being {_tick(event)} rather than "
                                 f"{_tick(reference)}, {per}.")
        lead = (f"Estimates are log-odds of {_tick(event)} against {_tick(reference)}; each odds "
                f"ratio is exp(estimate).")
    else:
        table.info.update(scale="relative_risk_ratio", axis="log", event=None, reference=reference,
                          effect=f"Relative-risk ratio of each level of {who} against "
                                 f"{_tick(reference)}, {per}.")
        lead = (f"Estimates are log relative risks of each level against {_tick(reference)}; each "
                f"relative-risk ratio is exp(estimate).")
    for row in table.rows:
        if str(row["feature"]).startswith("(intercept)"):
            row.update(ratio=None, ratio_low=None, ratio_high=None)
            continue
        row.update(ratio=_ratio(row["estimate"]), ratio_low=_ratio(row["ci_low"]),
                   ratio_high=_ratio(row["ci_high"]))
    if lead not in table.info["caption"]:
        table.info["caption"] = f"{lead} {table.info['caption']}"
    return table


def _on_rows(table: InferenceTable, n: int, rows: Rows | None) -> InferenceTable:
    """Say how many rows the table was estimated from, and which."""
    table.info.update(n_rows=int(n), rows=rows)
    if rows == "all":
        table.info["caption"] += f" Estimated from all {n:,} analyzed rows."
    elif rows == "training":
        table.info["caption"] += f" Estimated from the {n:,} training rows."
    return table


def inference_table(task: str, matrix: pd.DataFrame, y: Any, classes: Sequence[Any] | None,
                    clusters: Clusters = INDEPENDENT, *, outcome: Outcome | None = None,
                    rows: Rows | None = None,
                    study: Sequence[str] | None = None) -> InferenceTable:
    """The inference coefficient table for the linear family on the model matrix it saw.

    ``outcome`` names the outcome and its levels for the scale's words; ``rows`` says whether
    the matrix holds every analyzed row (inference, BLUEPRINT §12 ruling 3) or the training rows;
    ``study`` names the matrix's columns for what is studied, whose coefficient not being
    identified refuses the whole table (:func:`by_identified`). A table refused with no estimate
    (:func:`_no_spare_rows`) is put on no scale and says only how many rows it was handed.
    """
    table = _table(task, matrix, y, classes, clusters, study=study)
    if not table.rows:
        table.info.update(n_rows=int(len(matrix)), rows=rows)
        return table
    return _on_rows(_on_scale(table, task, classes, outcome), len(matrix), rows)


ESTIMATORS = {"regression": "ordinary least squares",
              "binary": "logistic regression (maximum likelihood)"}
MULTINOMIAL = "multinomial logistic regression (maximum likelihood)"


def _no_spare_rows(task: str, n_rows: int, n_coefficients: int, clusters: Clusters = INDEPENDENT,
                   *, estimator: str | None = None, intercept: bool = True,
                   cut_points: int = 0, levels: int | None = None) -> InferenceTable | None:
    """The table refused, its estimates and intervals both, where its rows do not exceed its
    coefficients (the intercept among them, or a proportional-odds model's ``cut_points``, or
    neither where ``intercept`` is False; for several classes, those of each level against the
    reference); None where they do. A fit's own coefficients (a prediction model's) pass
    ``levels``, the blocks it shows, one for every level as a softmax has them: the count is said
    for each of them where there are several and for none where there is one, whatever ``task``.
    On n ≤ p rows least squares cannot do both: with its rank below p many coefficient vectors
    fit the rows equally well, and with a square matrix of full rank one fits them exactly,
    leaving no residual degree of freedom to measure its uncertainty; a logistic or multinomial
    likelihood there has no unique maximum (a rank below p, or a square matrix that separates the
    outcome), and REML has no residual degree of freedom to estimate a variance from. A
    proportional-odds likelihood can have one there (its cut-points are shared by every row, not
    free per row), so its words, refused at the same count, say only that the rows are no more
    than its coefficients. ``estimator`` is the table's own (the linear family's for ``task``
    where None). The ways forward: fewer columns, or more rows."""
    if n_rows > n_coefficients:
        return None
    if levels is not None:
        per = f" for each of its {levels:,} levels" if levels > 1 else ""
    else:
        per = " for each level against the reference" if task == "multiclass" else ""
    among = (f", its {cut_points:,} cut-point{'s' if cut_points != 1 else ''} among them," if
             cut_points else ", the intercept among them," if intercept else "")
    has = f"The model has {n_coefficients:,} coefficients{per}{among} and is fit on {n_rows:,} rows"
    if cut_points:
        reason = (f"{has}, no more than its coefficients, so no estimate or interval is reported "
                  f"here.")
    else:
        reason = (f"{has}: with no more rows than coefficients, the rows cannot both determine "
                  f"the estimates and measure their uncertainty, so no estimate or interval is "
                  f"reported.")
    exits = [{"label": "Leave columns out of the model, so it has fewer coefficients than rows",
              "decision": None},
             {"label": f"Fit the model on more than {n_coefficients:,} rows", "decision": None}]
    info = _info(estimator or ESTIMATORS.get(task, MULTINOMIAL), "none",
                 f"No coefficients: {reason}", clusters, refused=reason, exits=exits)
    return InferenceTable([], info, [reason])


DEPENDENT_EXITS = (
    {"label": "Leave one of the dependent columns out of the model", "decision": None},)


def singular_lead(n_rows: int, exact: bool) -> str:
    """How a refusal for a dependency opens: the model matrix singular on these rows, to working
    precision where ``exact``, else to within R's tolerance (``repeated.aliased``), whose share is
    said from ``effects.ALIAS_TOL``."""
    from turbotab.core.models.effects import ALIAS_TOL

    lead = f"On these {n_rows:,} rows the model matrix is singular"
    if exact:
        return lead
    return (f"{lead} to within 1 part in {round(1 / ALIAS_TOL):,} of a column's variation about "
            f"its mean (the tolerance of R's `lm`)")


def dependency_what(named: Sequence[str], count: int, beside: str, exact: bool = True) -> str:
    """What is dependent: the columns ``named`` (those in a dependency, :func:`_terms`), in
    ``count`` ways, with ``beside`` (the intercept, or a constant where cut-points or a baseline
    hazard take its place), so their coefficients are not identified; to R's tolerance, not at
    it, where the dependencies are not ``exact`` (:func:`singular_lead`)."""
    one = len(named) == 1
    ways = "" if count <= 1 else f", in {count:,} ways,"
    at = "" if exact else " to that precision"
    after = "" if exact else " at it"
    if not named:
        return (f"its columns and {beside} are dependent{at}, so their coefficients are not "
                f"identified{after}")
    return (f"{_terms(named)} {'is' if one else 'are'}{ways} "
            f"{'a fixed combination' if one else 'fixed combinations'} of the other columns and "
            f"{beside}{at}, so {'its coefficient is' if one else 'their coefficients are'} not "
            f"identified{after}")


def dependency_reason(n_rows: int, named: Sequence[str], count: int, *,
                      beside: str = "the intercept", exact: bool = True) -> str:
    """Why a table whose columns ``named`` are, ``count`` ways, dependent with ``beside`` on its
    ``n_rows`` rows (``repeated.aliased``) reports no coefficient at all."""
    leave = ("leave one of the dependent columns out" if count <= 1 else
             f"leave out one dependent column for each of the {count:,} ways")
    return (f"{singular_lead(n_rows, exact)}: {dependency_what(named, count, beside, exact)}. No "
            f"estimate, interval or p-value is reported; {leave}.")


def _nothing(estimator: str, clusters: Clusters, reason: str) -> InferenceTable:
    """A table with no coefficient, refused for a dependency in ``reason``'s words."""
    info = _info(estimator, "none", f"No coefficients: {reason}", clusters, refused=reason,
                 exits=[dict(e) for e in DEPENDENT_EXITS])
    return InferenceTable([], info, [reason])


# How a table's rows follow the model matrix's columns (:func:`by_identified`): "intercept", in
# blocks of the intercept and then every column (one block per level against the reference for a
# multinomial model); "cuts", every column and then the cut-points; "columns", every column alone
# (a Cox model, whose baseline hazard is its constant).
Layout = Literal["intercept", "cuts", "columns"]
CONSTANT_ROWS = {"intercept": "the intercept", "cuts": "the cut-points"}


def column_scales(values: np.ndarray) -> np.ndarray:
    """Each column's root mean square (1 where it is zero or not finite), of the columns centered
    at their means where :func:`by_identified` centers them (their SDs): a table's model is fit
    with each column divided by it and its coefficients scaled back, an identity in exact
    arithmetic that keeps a fit's numbers on columns of every unit at working precision.
    statsmodels' least squares, logistic and multinomial fits, and every covariance inverted from
    an unscaled information, served wrong estimates or standard errors once some column's unit was
    10⁵ or more times another's. Where squaring a column's entries would overflow or underflow
    (beyond about 10¹⁵⁰, or under 10⁻¹⁵⁰), its root mean square is taken over its largest
    absolute value (``repeated.peaks``): squared as it stood, a column of 10²⁰⁰'s was infinite and
    one of 10⁻²⁰⁰'s zero, each left unscaled."""
    from turbotab.core.models.repeated import peaks

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.ones(values.shape[1] if values.ndim == 2 else 0)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        rms = np.sqrt(np.mean(values ** 2, axis=0))
        peak = peaks(values)
        extreme = (peak > 1e150) | (peak < 1e-150)
        if extreme.any():
            rms = np.where(extreme, peak * np.sqrt(np.mean((values / peak) ** 2, axis=0)), rms)
    return np.where(np.isfinite(rms) & (rms > 0), rms, 1.0)


def _row_columns(table: InferenceTable, layout: Layout, columns: Sequence[str]
                 ) -> list[str | None]:
    """The model-matrix column each of ``table``'s rows is the coefficient of (None: the
    intercept, or a cut-point), as ``layout`` orders them."""
    P = len(columns)
    n = len(table.rows)
    if layout == "intercept":
        block = [None, *columns]
        if n % len(block):
            raise ValueError(f"{n} rows are not blocks of the intercept and {P} columns.")
        return block * (n // len(block))
    if layout == "cuts":
        return [*columns, *([None] * (n - P))]
    return list(columns)


def _unscaled(table: InferenceTable, layout: Layout, columns: Sequence[str],
              scale: Mapping[str, float]) -> InferenceTable:
    """``table``, fit on the columns divided by ``scale`` (:func:`column_scales`), on the
    columns' own units: each coefficient's estimate, standard error and interval divided by its
    column's scale (its df and p-value unchanged), and the covariance by both."""
    if not table.rows:
        return table
    factor = np.array([1.0 if c is None else float(scale[c])
                       for c in _row_columns(table, layout, columns)])
    for row, f in zip(table.rows, factor):
        for key in ("estimate", "se", "ci_low", "ci_high"):
            if row.get(key) is not None:
                row[key] = _clean(row[key] / f)
    if table.cov is not None:
        table.cov = np.asarray(table.cov, dtype=float) / np.outer(factor, factor)
    return table


def _by_row_reason(n_rows: int, named: Sequence[str], constant: str | None, count: int,
                   beside: str, left_out: Sequence[str], exact: bool = True) -> str:
    """Why the coefficients in a dependency (``named``, and ``constant``, the intercept or the
    cut-points, where the constant is in one) are refused by row, while the others are reported
    from the fit with ``left_out`` left out (:func:`by_identified`); at R's tolerance where the
    dependencies are not ``exact`` (:func:`singular_lead`)."""
    what = dependency_what(named, count, beside, exact)
    if constant:
        what += f", nor {'is' if constant == 'the intercept' else 'are'} {constant}"
    leave = ("leave one of the dependent columns out" if count <= 1 else
             f"leave out one dependent column for each of the {count:,} ways")
    unreported = [*(_term(n) for n in named[:NAMED]), *([constant] if constant else [])]
    rest = len(named) - min(len(named), NAMED)
    listed = (f"{', '.join(unreported)} and {rest:,} more coefficient{'s' if rest != 1 else ''}"
              if rest else unreported[0] if len(unreported) == 1 else
              f"{', '.join(unreported[:-1])} and {unreported[-1]}")
    return (f"{singular_lead(n_rows, exact)}: {what}. No estimate, interval "
            f"or p-value is reported for {listed}; the other coefficients, which these rows "
            f"identify, are estimated with {_terms(list(left_out))} left out. To report every "
            f"coefficient, {leave}.")


def by_identified(matrix: pd.DataFrame, build: Callable[[pd.DataFrame], InferenceTable], *,
                  layout: Layout, refuse: Callable[[str], InferenceTable],
                  beside: str = "the intercept", rows: np.ndarray | None = None,
                  study: Sequence[str] | None = None, scaled: bool = True) -> InferenceTable:
    """A coefficient table of a family whose linear predictor is Xβ, from ``build`` (its table on
    a model matrix whose columns are not dependent), on ``matrix``.

    The model is fit on the columns centered at their means and scaled to an SD of one, and its
    rows mapped back (:func:`_recentered`, :func:`_unscaled`): a column's unit then changes its
    own coefficient and standard error by the unit's factor and leaves every other row as it is,
    and its origin moves only the intercept (or the cut-points), to working precision, whatever
    the units and origins. Scaled alone, a date in epoch seconds spanning a day stayed within 10⁻⁵
    of a constant, so nearly a copy of the intercept, and the logistic and multinomial fits on it
    stopped short of their maximum. ``scaled`` False hands ``build`` the columns as they are, for
    a fit that is already invariant to them (the mixed model's REML, which factors the design by
    QR). A layout with no constant (``"columns"``, the Cox model, whose fit centers its columns
    itself) is scaled to a root mean square of one and not centered.

    Where the columns, with ``beside`` (the intercept, or a constant where cut-points or a
    baseline hazard take its place), are dependent on the rows ``rows`` marks (every row where
    None) at R's tolerance on the centered columns (``repeated.aliased``: a column whose part
    outside the span of the intercept and the columns kept before it is under 10⁻⁷ of its
    length), a coefficient whose column is in no dependency is estimable: its estimate and
    standard error are those of the fit with each aliased column left out, as R's ``lm`` and
    ``glm`` report it, whichever such columns are left out, to that precision. Each coefficient
    in a dependency (the intercept or the cut-points too, where the constant is in one) is
    refused by row, with no estimate, interval or p-value (:attr:`InferenceTable.unidentified`),
    in words that name the columns of the dependencies and count them, which are the table's
    first concern, and say R's tolerance where a dependency is not one to working precision
    (:func:`singular_lead`). That holds where the caller names what is studied (``study``, the
    columns of the study factor). The whole table is refused, with no coefficient, in
    ``refuse``'s table (:func:`dependency_reason`'s words), where a column of ``study`` is in a
    dependency, where the caller names nothing studied (``study`` None: whether what it reads is
    estimable is not known here), or where no coefficient is estimable."""
    from turbotab.core.models.repeated import aliased

    columns = [str(c) for c in matrix.columns]
    values = matrix.to_numpy(dtype=float)
    on = values if rows is None else values[np.asarray(rows)]
    shift = (column_means(on) if scaled and layout in CONSTANT_ROWS
             else np.zeros(len(columns)))
    factors = column_scales(on - shift) if scaled else np.ones(len(columns))
    scale = dict(zip(columns, factors))
    found = aliased(on)
    where = {c: j for j, c in enumerate(columns)}

    def fit(names: Sequence[str]) -> InferenceTable:
        at = [where[c] for c in names]
        part = matrix.iloc[:, at].astype(float)
        part = (part - shift[at]) / factors[at] if at else part
        table = _recentered(build(part), layout, names, shift[at] / factors[at])
        return _unscaled(table, layout, names, scale)

    if found.count == 0:
        return fit(columns)
    count, dependent = found.count, found.columns
    named = [columns[j - 1] for j in dependent if j > 0]
    constant = CONSTANT_ROWS.get(layout) if 0 in set(dependent.tolist()) else None
    estimable = [c for c in columns if c not in set(named)]
    has_constant = layout in CONSTANT_ROWS and constant is None
    if study is None or set(study) & set(named) or not (estimable or has_constant):
        return refuse(dependency_reason(len(on), named, count, beside=beside,
                                        exact=found.exact))
    keep = [columns[j - 1] for j in found.kept if j > 0]
    table = fit(keep)
    left_out = [c for c in columns if c not in set(keep)]
    reason = _by_row_reason(len(on), named, constant, count, beside, left_out, found.exact)
    table.concerns.insert(0, reason)
    table.dependency = reason
    if not table.rows:
        return table
    return _expanded(table, layout, keep, columns, set(named), constant is not None)


def column_means(values: np.ndarray) -> np.ndarray:
    """Each column's mean (0 where it is not finite), the origin :func:`by_identified` centers a
    column at."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(values.shape[1] if values.ndim == 2 else 0)
    with np.errstate(invalid="ignore", over="ignore"):
        mean = np.mean(values, axis=0)
    return np.where(np.isfinite(mean), mean, 0.0)


def combination_row(estimate: float, se: float | None, df: float | None = None
                    ) -> dict[str, Any]:
    """The fields of a row for one linear combination of a table's coefficients at ``estimate``
    with standard error ``se``: its interval and p-value on t(``df``), or on the normal where
    ``df`` is None (:func:`_t_rows`, :func:`_z_rows`)."""
    se = math.nan if se is None else float(se)
    row = (_z_rows(["c"], [estimate], [se]) if df is None else
           _t_rows(["c"], [estimate], [se], [float(df)]))[0]
    return {k: row[k] for k in ("se", "df", "ci_low", "ci_high", "p")}


def _recentered(table: InferenceTable, layout: Layout, columns: Sequence[str],
                origin: np.ndarray) -> InferenceTable:
    """``table``, fit on ``columns`` each less its ``origin`` (in the units the fit saw), with its
    intercept (each class's, for a multinomial model) or its cut-points those of the columns as
    they were. A slope does not move with its column's origin; the intercept, the linear predictor
    where every column is zero, moves by ``−Σ_j γ_j o_j``, and each cut-point, compared with
    ``−Xβ``, by ``+Σ_j γ_j o_j``. Each such row's estimate is that combination ``c`` of the fit's
    coefficients. A table whose rows follow from its covariance and one df gives the rest: where
    the row had a standard error it is ``√(cᵀVc)``, and where it had an interval it is on the
    row's own df (the same for every row of such a table) or the normal. A table whose rows do
    not (CR2's Bell–McCaffrey df, which are the combination's own; Firth's separation check; a
    multinomial table's covariance, not kept) gives the row's fields itself
    (:attr:`InferenceTable.contrast`), at every origin, and is then rid of it. The covariance
    becomes ``A V Aᵀ`` for the map ``A`` from the fit's coefficients to the rows."""
    contrast, table.contrast = table.contrast, None
    origin = np.asarray(origin, dtype=float)
    if not table.rows or layout not in CONSTANT_ROWS or not (np.any(origin) or contrast):
        return table
    n, P = len(table.rows), len(columns)
    roles = _row_columns(table, layout, columns)
    A = np.eye(n)
    if layout == "intercept":
        size = 1 + P
        for b in range(n // size):
            A[b * size, b * size + 1:(b + 1) * size] = -origin
    else:
        A[P:, :P] = origin
    est = np.array([np.nan if r.get("estimate") is None else r["estimate"] for r in table.rows],
                   dtype=float)
    V = None if table.cov is None else np.asarray(table.cov, dtype=float)
    for r, role in enumerate(roles):
        if role is not None:
            continue
        c = A[r]
        used = c != 0
        b = float(c[used] @ est[used])
        row = table.rows[r]
        row["estimate"] = _clean(b)
        if contrast is not None:
            fields = contrast(c, b)
        elif row.get("se") is not None:
            if V is None:
                raise ValueError(f"The row {row['feature']!r} has a standard error but its table "
                                 f"no covariance to move it to the columns' origin.")
            fields = combination_row(b, math.sqrt(max(float(c @ V @ c), 0.0)), row.get("df"))
            if all(row.get(k) is None for k in ("ci_low", "ci_high", "p")):
                fields = {"se": fields["se"]}
        else:
            fields = {}
        infinite = fields.pop("infinite", None)
        row.update(fields)
        if infinite is not None:
            label = str(row["feature"])
            table.infinite = [f for f in table.infinite if f != label]
            if infinite:
                table.infinite.insert(0, label)
    if V is not None:
        table.cov = A @ V @ A.T
    return table


def _expanded(table: InferenceTable, layout: Layout, keep: Sequence[str], columns: Sequence[str],
              named: set[str], constant: bool) -> InferenceTable:
    """``table`` (fit on the columns ``keep``) with a row for every one of ``columns``: a column in
    a dependency (``named``), and the intercept or the cut-points where the constant is in one,
    refused by row with no estimate, interval or p-value, the covariance not a number in its row
    and column."""
    source = _row_columns(table, layout, keep)
    template = dict.fromkeys(table.rows[0])

    def refused(label: str) -> dict[str, Any]:
        table.unidentified.append(label)
        return {**template, "feature": label}

    out: list[tuple[dict[str, Any], int | None]] = []
    if layout == "intercept":
        size = 1 + len(keep)
        for b in range(len(table.rows) // size):
            head = table.rows[b * size]
            suffix = str(head["feature"])[len("(intercept)"):]
            out.append((refused(str(head["feature"])), None) if constant else (head, b * size))
            for c in columns:
                at = b * size + 1 + list(keep).index(c) if c in keep else None
                out.append((refused(f"{c}{suffix}"), None) if c in named or at is None else
                           (table.rows[at], at))
    else:
        for c in columns:
            at = list(keep).index(c) if c in keep else None
            out.append((refused(c), None) if c in named or at is None else (table.rows[at], at))
        for i, c in enumerate(source):
            if c is None:
                row = table.rows[i]
                out.append((refused(str(row["feature"])), None) if constant else (row, i))
    if table.cov is not None:
        cov = np.full((len(out), len(out)), np.nan)
        served = [k for k, (_, at) in enumerate(out) if at is not None]
        at = [out[k][1] for k in served]
        cov[np.ix_(served, served)] = np.asarray(table.cov, dtype=float)[np.ix_(at, at)]
        table.cov = cov
    table.rows = [row for row, _ in out]
    return table


def _table(task: str, matrix: pd.DataFrame, y: Any, classes: Sequence[Any] | None,
           clusters: Clusters, *, study: Sequence[str] | None = None) -> InferenceTable:
    if clusters.clustered and len(clusters.codes) != len(matrix):
        raise ValueError(f"The clusters cover {len(clusters.codes):,} rows but the model matrix has "
                         f"{len(matrix):,}.")
    estimator = ESTIMATORS.get(task, MULTINOMIAL)
    spare = _no_spare_rows(task, len(matrix), matrix.shape[1] + 1, clusters)
    if spare is not None:
        return spare
    table = by_identified(matrix, lambda m: _fitted(task, m, y, classes, clusters),
                          layout="intercept", study=study,
                          refuse=lambda reason: _nothing(estimator, clusters, reason))
    resting = table.leverage_one
    if resting and study is not None and set(study) & set(resting):
        # A studied coefficient that rests on a row of leverage one refuses the whole table, as
        # one in a dependency does (:func:`by_identified`). Unlike a dependency's, the other
        # coefficients do not depend on which column a fit leaves out, so a caller that names
        # nothing studied is served them, the resting rows refused by row: the effects stage's
        # table beside a site held by one row serves the exposure, as its Cook's distance check
        # reads that row.
        reason = leverage_words(table.leverage_rows, len(matrix), resting, whole=True)
        cols = [c for c in resting if c != "(intercept)"]
        exits = [{"label": f"Leave {_and(cols)} out, or merge its rare levels", "decision": None}
                 ] if cols else []
        return InferenceTable([], _info(estimator, "none", f"No coefficients: {reason}", clusters,
                                        refused=reason, exits=exits), [reason])
    return table


def _fitted(task: str, matrix: pd.DataFrame, y: Any, classes: Sequence[Any] | None,
            clusters: Clusters) -> InferenceTable:
    """The linear family's table on a model matrix whose columns are not dependent."""
    import statsmodels.api as sm

    # The fits are handed the indexed frame, so which rows they saw stays checkable.
    exog = sm.add_constant(matrix.astype(float), has_constant="add")
    names = ["(intercept)" if c == "const" else str(c) for c in exog.columns]
    if task == "regression":
        return _least_squares(names, exog, np.asarray(y, dtype=float), clusters)
    if task == "binary":
        if classes is None or len(classes) != 2:
            raise ValueError("A binary outcome needs exactly two classes.")
        event = (np.asarray(y) == classes[1]).astype(float)
        return _logistic(names, exog, event, clusters)
    return _multinomial(names, exog, np.asarray(y), list(classes or []), clusters)


# The Newton steps statsmodels' logistic and multinomial fits take before they stop
# (:func:`not_converged`).
MAX_NEWTON = 200


def not_converged(model: str) -> str:
    """Why a logistic or multinomial table whose maximum-likelihood fit stopped before converging
    reports no estimate: its numbers are where the steps stopped, not the maximum's. Before the
    columns were centered, a date in epoch seconds spanning a day stopped the logistic fit there
    and served `x`'s log-odds as 59.05 (R's ``glm``: 1.06)."""
    steps = f"{MAX_NEWTON:,} Newton step{'s' if MAX_NEWTON != 1 else ''}"
    return (f"The {model} model's maximum-likelihood fit did not converge in {steps}, so no "
            f"estimate, interval or p-value is reported.")


# An iterative fit's table (logistic, multinomial, proportional odds, Firth, and their survey
# forms) is served only where the fit is at its maximum to within this share of a standard error
# (:func:`at_maximum`).
SHORT_OF_MAXIMUM = 1e-4


def newton_distance(working: np.ndarray, score: np.ndarray) -> float:
    """How far a fit is from its maximum, in standard errors: ``√(sᵀ I⁻¹ s)`` for the score ``s``
    and the information ``I`` at its estimate, with ``working`` its working design (``WᵀW = I``:
    :func:`logistic_working`, :func:`multinomial_working`). By Cauchy–Schwarz no coefficient, nor
    any combination of them, is moved by the Newton step from there by more than this many of its
    standard errors, and the step moves some combination by exactly that many. Taken from the
    triangle of the working design's QR (``W = QR``, so the distance is ``|R⁻ᵀs|``), whose rounding
    is its condition number, where an inverse of the information has that squared: the survey
    multinomial fit beside ``x + 1.1·10⁻⁷·d``, whose steps came from a least-squares solve on an
    information of condition number 10¹⁴, declared convergence where this distance was 0.57 and
    served the study factor's log-odds at −0.1200 against the maximum's −0.1177. The score is the
    fit's own (``Xᵀ(y − μ)``, each class's ``Xᵀ(y_k − π_k)``), never ``Qᵀr`` for the working
    residual ``r``, which divides each row by the root of its weight: where a fitted probability
    rounds to 0 or 1 that weight is zero, Householder's Q leaves the row at rounding, and a
    converged logistic fit beside one outlying row (its fitted probability 1.0, no event) had its
    table refused as 3.3·10¹³⁴ standard errors short of its maximum, where it is under 10⁻¹⁴ from
    it (V5c2-2). Infinite where the working design is singular or not finite."""
    from scipy.linalg import solve_triangular

    W = np.asarray(working, dtype=float)
    s = np.asarray(score, dtype=float)
    if W.shape[1] == 0:
        return 0.0
    if not (np.all(np.isfinite(W)) and np.all(np.isfinite(s))):
        return math.inf
    R = np.linalg.qr(W, mode="r")
    if not np.all(np.isfinite(R)) or np.any(np.diag(R) == 0):
        return math.inf
    return float(np.linalg.norm(solve_triangular(R, s, trans="T")))


def information_distance(score: np.ndarray, information: np.ndarray) -> float:
    """:func:`newton_distance` from a fit's score and its information, ``√(sᵀ I⁻¹ s)``, for a fit
    whose information is evaluated on the QR's Q (``ordinal.on_qr``), so is well conditioned."""
    s = np.asarray(score, dtype=float)
    if s.size == 0:
        return 0.0
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(information))):
        return math.inf
    value = float(s @ scaled_inverse(np.asarray(information, dtype=float)) @ s)
    return math.sqrt(value) if value >= 0 else math.inf


def _standard_errors(distance: float) -> str:
    """A distance in standard errors as the words say it: to two figures below a thousand, written
    out below a million, and above as a power of ten (2.9 × 10¹⁰¹), never as its every digit."""
    if distance < 1_000:
        return f"{distance:.2g}"
    if distance < 999_999.5:
        return f"{distance:,.0f}"
    exponent = math.floor(math.log10(distance))
    mantissa = distance / 10 ** exponent
    if round(mantissa, 1) >= 10.0:
        exponent, mantissa = exponent + 1, mantissa / 10
    sup = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa:.1f} × 10{sup}"


def at_maximum(fit: str, distance: float, *, converged: bool = True,
               steps: int | None = None) -> str | None:
    """Why an iterative fit's table (``fit`` names it, as "the proportional-odds fit") reports no
    estimate: it did not converge (in ``steps`` Newton steps, where given), or it stopped more than
    :data:`SHORT_OF_MAXIMUM` of a standard error from its maximum (:func:`newton_distance`), its
    numbers being where the steps stopped, not the maximum's; None where neither holds."""
    lead = fit[0].upper() + fit[1:]
    if not converged:
        within = "" if steps is None else f" in {steps:,} Newton step{'s' if steps != 1 else ''}"
        return (f"{lead} did not converge{within}, so no estimate, interval or p-value is "
                f"reported.")
    if math.isfinite(distance) and distance <= SHORT_OF_MAXIMUM:
        return None
    moved = ("the information there is singular" if not math.isfinite(distance) else
             f"one more Newton step would still move its estimates by up to "
             f"{_standard_errors(distance)} of a standard error")
    return (f"{lead} stopped short of its maximum: where it stopped, {moved}, and a table is "
            f"served only within 1/{round(1 / SHORT_OF_MAXIMUM):,} of one, so no estimate, "
            f"interval or p-value is reported.")


def _least_squares(names: list[str], exog: pd.DataFrame, y: np.ndarray,
                   clusters: Clusters) -> InferenceTable:
    import statsmodels.api as sm

    X = exog.to_numpy(dtype=float)
    fit = sm.OLS(y, exog).fit()
    est = np.asarray(fit.params, dtype=float)
    estimator = "ordinary least squares"
    refusal = floor_refusal(clusters, task="regression")
    if refusal:
        return _refused(names, est, estimator, clusters, *refusal)
    if clusters.clustered:
        return _clustered(names, est, estimator, X, np.asarray(fit.resid, dtype=float),
                          clusters.codes, clusters)
    # HC3 from the design's QR, not statsmodels' leverages from (XᵀX)⁻¹ (:func:`hc3_covariance`).
    cov = hc3_covariance(X, np.asarray(fit.resid, dtype=float))
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.diag(cov))
    dof = float(fit.df_resid)
    # A coefficient that rests on a row of leverage one has no HC3 standard error: refused by row
    # (the intercept, which centering moves, through the contrast as it stands).
    lone = resting_on_leverage_one(X)
    rests = lone[0] if lone else (lambda c: False)
    resting = [j for j, n in enumerate(names) if n != "(intercept)" and rests(np.eye(len(names))[j])]
    served = [j for j in range(len(names)) if j not in set(resting)]
    reason = no_interval([names[j] for j in served], se[served],
                         fit="The least-squares fit's HC3 covariance")
    if reason:
        return _refused(names, est, estimator, clusters, reason, ())
    rows = _t_rows(names, est, se, np.full(len(est), dof))
    concerns = [clusters.note] if clusters.note else []
    hetero = heteroskedasticity_concern(np.asarray(fit.resid, float), X, names,
                                        np.asarray(fit.bse, float),
                                        np.where(np.isin(np.arange(len(names)), resting), np.nan,
                                                 se))
    if hetero:
        concerns.append(hetero)
    caption = f"95% intervals from HC3 heteroskedasticity-robust standard errors, on t({dof:,.0f})."
    table = InferenceTable(rows, _info(estimator, "HC3", caption, clusters), concerns, cov=cov)
    if lone is None:
        return table
    blank = ("estimate", "se", "df", "ci_low", "ci_high", "p")
    table.leverage_rows = lone[1]
    for j in resting:
        rows[j].update(dict.fromkeys(blank))
        table.leverage_one.append(names[j])
    if resting:
        table.cov = np.array(cov, dtype=float)
        table.cov[resting, :] = np.nan
        table.cov[:, resting] = np.nan
    words = leverage_words(lone[1], len(X), table.leverage_one)
    if table.leverage_one:
        table.concerns.insert(0, words)

    def contrast(c: np.ndarray, b: float) -> dict[str, Any]:
        nonlocal words
        if rests(c):
            if table.concerns and table.concerns[0] == words:
                table.concerns.pop(0)
            table.leverage_one.insert(0, "(intercept)")
            words = leverage_words(lone[1], len(X), table.leverage_one)
            table.concerns.insert(0, words)
            return dict.fromkeys(blank)
        value = float(c @ est)
        return {"estimate": value, **combination_row(value, math.sqrt(max(float(c @ cov @ c),
                                                                          0.0)), dof)}

    table.contrast = contrast
    return table


def _named(names: Sequence[str], idx: Sequence[int]) -> list[str]:
    return [names[j] for j in idx if names[j] != "(intercept)"]


def _and(items: Sequence[str]) -> str:
    quoted = [f"`{i}`" for i in items]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _logistic(names: list[str], exog: pd.DataFrame, y: np.ndarray, clusters: Clusters) -> InferenceTable:
    import statsmodels.api as sm

    X = exog.to_numpy(dtype=float)
    found = separated_columns(X, y)
    separated = _named(names, found)
    if separated:
        table = _firth_table(names, X, y, clusters, separated)
        table.infinite = [names[j] for j in found]  # the intercept too, where it is infinite
        return table
    fit = sm.Logit(y, exog).fit(disp=0, maxiter=MAX_NEWTON)
    est = np.asarray(fit.params, dtype=float)
    estimator = "logistic regression (maximum likelihood)"
    if not fit.mle_retvals.get("converged", False):
        return _refused(names, np.full(len(names), np.nan), estimator, clusters,
                        not_converged("logistic"), ())
    mu = np.asarray(fit.predict(X), dtype=float)
    Xw, ew = logistic_working(X, y, mu)
    # The distance reads the score, Xᵀ(y − μ), never the Pearson residual (:func:`newton_distance`).
    short = at_maximum("the logistic model's maximum-likelihood fit",
                       newton_distance(Xw, X.T @ (y - mu)))
    if short:
        return _refused(names, np.full(len(names), np.nan), estimator, clusters, short, ())
    refusal = floor_refusal(clusters, task="binary")
    if refusal:
        return _refused(names, est, estimator, clusters, *refusal)
    if clusters.clustered:
        return _clustered(names, est, estimator, Xw, ew, clusters.codes, clusters)
    # The inverse information from the working design's QR (:func:`gram_inverse`), not
    # statsmodels' inverse of the Hessian, whose rounding is the design's condition number squared.
    cov = gram_inverse(Xw)
    cov = np.asarray(fit.cov_params(), dtype=float) if cov is None else cov
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    reason = no_interval(names, se, fit="The logistic model's information")
    if reason:
        return _refused(names, est, estimator, clusters, reason, ())
    rows = _z_rows(names, est, se)
    concerns = [clusters.note] if clusters.note else []
    return InferenceTable(rows, _info(estimator, "model",
                                      "95% Wald intervals from the logistic model's information.",
                                      clusters), concerns, cov=cov)


def separation_concern(separated: Sequence[str]) -> str:
    """A Firth table's first concern, naming the columns that separate the outcome in it, which a
    table pooled over completed copies replaces with its own (:attr:`PooledFirth.replaces`)."""
    one = len(separated) == 1
    return (f"{_and(separated)} {'separates' if one else 'separate'} the outcome (the "
            f"maximum-likelihood log-odds is infinite), so a Wald interval or p-value for "
            f"{'it' if one else 'them'} would be meaningless; the table reports Firth-penalized "
            f"logistic regression instead (Heinze & Schemper 2002).")


def _firth_table(names: list[str], X: np.ndarray, y: np.ndarray, clusters: Clusters,
                 separated: list[str]) -> InferenceTable:
    """Firth's table (:func:`_firth_rows`), whose intercept's row, the separation check of a linear
    combination where the columns were centered (and, in one completed copy's table made for
    Rubin's rules, its standard error), it gives through :attr:`InferenceTable.contrast`
    (:func:`firth_combination`, never its profile: the data separate). A table refused (its
    profile on separated rows among the reasons) gives its intercept's estimate alone."""
    fit = firth_fit(X, y)
    table = _firth_rows(names, X, y, clusters, separated, fit)
    refused = table.info.get("refused") is not None

    def contrast(c: np.ndarray, b: float) -> dict[str, Any]:
        return firth_combination(X, y, fit, c, profile=False, se=not refused)

    table.contrast = contrast
    return table


def firth_distance(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """:func:`newton_distance` for a Firth fit at ``beta``: ``|R⁻ᵀs|`` for its modified score
    ``s = Xᵀ(y − μ + h (½ − μ))`` and the triangle of its weighted design's QR (``√w·X = QR``), as
    :func:`_newton_step` takes its decrement, so no row's residual is divided by its weight. A row
    whose fitted probability rounds to exactly 0 or 1 has weight zero and carries no information,
    yet Householder's QR leaves its row of Q at rounding, not at zero: divided by
    ``√max(w, 10⁻³⁰⁰)``, its residual beside ``xs = sign(y)·(10⁻⁴ + |d| − min|d|)`` had a table
    whose fit had converged to 8.8·10⁻⁹ of a standard error from its maximum refused as up to
    2.9·10¹⁰¹ of one short of it (V5c2-2). Infinite where the triangle is singular or the score
    not finite."""
    from scipy.linalg import solve_triangular

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, _, Q, R, _ = _firth_parts(X, y, np.asarray(beta, dtype=float))
    h = np.einsum("ij,ij->i", Q, Q)
    score = X.T @ (y - mu + h * (0.5 - mu))
    if R.shape[0] == 0:
        return 0.0
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(score))) or np.any(np.diag(R) == 0):
        return math.inf
    return float(np.linalg.norm(solve_triangular(R, score, trans="T")))


def _firth_rows(names: list[str], X: np.ndarray, y: np.ndarray, clusters: Clusters,
                separated: list[str], fit: FirthFit) -> InferenceTable:
    """Firth's rows on rows the data separate (``separated``, the columns of the cone
    :func:`separated_columns` finds): refused for want of independent rows or of enough units, or
    with no estimate where the fit is not at its penalized maximum (:func:`at_maximum`); in one
    completed copy's table made for Rubin's rules, each estimate with its standard error from the
    information (:func:`for_pooling`); otherwise the penalized estimates alone, every profile
    interval and p-value refused in :func:`separated_profile_words`' words with the separation's
    exit (:attr:`InferenceTable.profile_refusal`)."""
    est = fit.beta
    estimator = FIRTH
    verb = "separates" if len(separated) == 1 else "separate"
    concern = separation_concern(separated)
    extra = {"separated": separated}
    refusal = floor_refusal(clusters, task="binary")
    if refusal is None and clusters.clustered:
        refusal = (f"{_and(separated)} {verb} the outcome and the rows repeat by `{clusters.column}`: "
                   f"Firth's profile intervals assume independent rows, so no interval or p-value "
                   f"is reported.",
                   ({"label": f"Combine each `{clusters.column}`'s rows into one (the unit question, "
                              f"before the seal)", "decision": None},
                    {"label": f"Leave {_and(separated)} out, or merge its rare levels", "decision": None}))
    short = at_maximum("Firth's penalized fit", firth_distance(X, y, est),
                       converged=fit.converged, steps=fit.iterations)
    if refusal is None and short:
        # Where it stopped is not the penalized maximum: no estimate (:func:`at_maximum`).
        refusal = (short, ())
        est = np.full(len(names), np.nan)
    if refusal:
        table = _refused(names, est, estimator, clusters, *refusal)
        table.info.update(extra)
        table.concerns.insert(0, concern)
        return table
    se = np.sqrt(np.clip(np.diag(fit.cov), 0, None))
    concerns = [concern] + ([clusters.note] if clusters.note else [])
    if _POOLING.get():
        # One completed copy's table, for Rubin's rules (:func:`for_pooling`): no profile.
        rows = [_row(name, est[j], se[j]) for j, name in enumerate(names)]
        return InferenceTable(rows, _info(estimator, "profile", POOLED_FIRTH_CAPTION, clusters,
                                          **extra), concerns)
    # Held at a value the penalized likelihood can have several maxima where the data separate,
    # and a warm-started profile can stop on a lower one (:data:`SEVERAL_MAXIMA`): the estimates
    # alone, complete separation or quasi-complete (V5c2-1).
    words = separated_profile_words(separated)
    table = _refused(names, est, estimator, clusters, words, separation_exits(separated))
    table.info.update(extra)
    table.concerns.insert(0, concern)
    table.profile_refusal = words
    return table


def firth_combination(X: np.ndarray, y: np.ndarray, fit: FirthFit, c: np.ndarray, *,
                      profile: bool = True, se: bool = True, strict: bool = False,
                      name: str | None = None) -> dict[str, Any]:
    """A linear combination ``cᵀβ`` of a Firth fit's coefficients as a row's fields: whether its
    maximum-likelihood value is infinite (:func:`separated_columns`), with ``se`` its standard
    error from the information, and with ``profile`` its profile penalized-likelihood interval and
    likelihood-ratio p-value (:func:`firth_profile`, ``strict`` and ``name`` passed on), which on
    rows the data separate raises :class:`FirthProfileSeparated`. Each is computed on the design
    reparametrized so that ``cᵀβ`` is its first coefficient, ``X [c/|c|², N]`` with ``N`` an
    orthonormal basis of the directions orthogonal to ``c`` (``β = c/|c|² δ₀ + N δ``): Firth's
    penalty, ½ log|I(β)|, moves by a constant under a linear reparametrization, so the profile is
    the same, and the design keeps the conditioning of ``X``'s. A combination that is one
    coefficient is that coefficient's row on ``X`` itself, its separation the fit's own reading."""
    c = np.asarray(c, dtype=float)
    out: dict[str, Any] = {}
    nonzero = np.flatnonzero(c)
    if len(nonzero) == 1 and c[nonzero[0]] == 1.0:
        j = int(nonzero[0])
        if se:
            out["se"] = _clean(math.sqrt(max(float(fit.cov[j, j]), 0.0)))
        if profile:
            low, high, p = firth_profile(X, y, fit, j, strict=strict, name=name)
            out.update(df=None, ci_low=None if low is None else _clean(low),
                       ci_high=None if high is None else _clean(high), p=_clean(p))
        return out
    _, _, vt = np.linalg.svd(c[None, :])
    T = np.vstack([c, vt[1:]])  # δ = T β
    inverse = np.column_stack([c / float(c @ c), vt[1:].T])
    Xd = np.asarray(X, dtype=float) @ inverse
    out["infinite"] = 0 in separated_columns(Xd, y)
    if se:
        out["se"] = _clean(math.sqrt(max(float(c @ fit.cov @ c), 0.0)))
    if profile:
        delta = T @ fit.beta
        moved = FirthFit(beta=delta, loglik=float(_firth_parts(Xd, y, delta)[4]),
                         cov=T @ fit.cov @ T.T, iterations=fit.iterations, converged=fit.converged)
        low, high, p = firth_profile(Xd, y, moved, 0, strict=strict, name=name)
        out.update(df=None, ci_low=None if low is None else _clean(low),
                   ci_high=None if high is None else _clean(high), p=_clean(p))
    return out


# What each completed copy's Firth table says it holds (:func:`for_pooling`), and so what the
# pooled table's intervals are.
POOLED_FIRTH_CAPTION = (
    "Firth-penalized logistic regression, because the maximum-likelihood estimate does not "
    "exist: each estimate with its standard error from the information at the penalized "
    "estimate, which Rubin's rules pool into Wald intervals and p-values; no profile interval is "
    "computed.")


@dataclass
class PooledFirth:
    """What a table pooled over completed copies says when some copy is Firth's
    (:func:`pooled_firth`): the caption that replaces the first copy's, the ``info`` fields it
    sets, the concern that names each coefficient whose interval is withheld, and the copies' own
    separation concerns (:func:`separation_concern`) it replaces, which the pooled table leaves
    out of the first copy's concerns."""

    caption: str
    info: dict[str, Any]
    concern: str
    replaces: list[str] = field(default_factory=list)


# What a withheld pooled row does not report (:func:`pooled_firth`): with its standard error and
# degrees of freedom (and its fraction of missing information, which with its Monte Carlo error
# gives the standard error back) the withheld Wald interval would be rebuilt as estimate ± t·se.
WITHHELD = ("se", "df", "fmi", "ci_low", "ci_high", "p")


def copies_words(k: int, m: int) -> str:
    """"in every one of the m imputed copies", or "in k of the m imputed copies"."""
    return (f"in every one of the {m:,} imputed copies" if k == m else
            f"in {k:,} of the {m:,} imputed copies")


def _in_copies(counts: Mapping[str, int], names: Sequence[str], m: int) -> str:
    """In how many of the ``m`` copies each of ``names`` is counted: once when they agree."""
    each = {counts[n] for n in names}
    if len(each) == 1:
        return copies_words(each.pop(), m)
    return (f"in some of the {m:,} imputed copies ("
            + ", ".join(f"{_term(n)} in {counts[n]:,}" for n in names) + ")")


def _term(name: str) -> str:
    return "the intercept" if name == "(intercept)" else f"`{name}`"


def withheld_reason(m: int, named: Sequence[str]) -> str:
    """Why a coefficient infinite in some of the ``m`` copies is reported by its estimate alone:
    the claim's card (``inference.pooled_separation``), with its citation."""
    from turbotab.core import claims

    because = claims.card("inference.pooled_separation", {"copies": m, "separated": list(named)})
    return f"{because} (Heinze & Schemper 2002)"


COMPLETE_CASES = "analyze complete cases, where one table is fit and none is pooled"


def pooled_firth(tables: Sequence[InferenceTable], rows: list[dict[str, Any]], *,
                 answer: Mapping[str, Any] | None = None,
                 sources: Mapping[str, Sequence[str] | set[str]] | None = None
                 ) -> PooledFirth | None:
    """Rubin's rules over completed copies some of whose logistic tables are Firth's (a column
    separates the outcome there), or None when none is.

    The pooled rows (``rows``, ``imputation.pool_rows``: changed in place) are Wald intervals and
    p-values on t from each copy's estimate and standard error. A row whose maximum-likelihood
    value is infinite in any copy (``separated_columns`` on that copy: the copy's
    ``info["separated"]``, and its ``infinite``, which adds the intercept and the all-components
    relative effects that rest on a separated column) keeps its pooled estimate (and its ratio,
    and its Monte Carlo error), and its standard error, degrees of freedom, fraction of missing
    information, interval, p-value and ratio interval are withheld (:data:`WITHHELD`; claim
    ``inference.pooled_separation``), with the ways forward: read the separated columns' source
    columns (``sources``: each model-matrix column's raw columns, as
    ``stages.effects.matrix_sources`` reads them; the column itself without) another way or
    combine their sparse levels, or complete cases (one table, nothing pooled) where ``answer``,
    the missing-values answer the copies were drawn by, is given. The caption is the first Firth
    copy's (:data:`POOLED_FIRTH_CAPTION`), or, where not every copy is Firth's, says how many of
    each kind there are and that only the Firth copies have no maximum-likelihood estimate. The
    concern replaces the Firth copies' own (:attr:`PooledFirth.replaces`), which say the table
    reports Firth's fit, true of a Firth copy and not of every pooled table."""
    firth = [t for t in tables if t.info.get("estimator") == FIRTH]
    m = len(tables)
    counts: dict[str, int] = {}  # the separated columns the copies' own concerns name
    held: dict[str, int] = {}  # every row withheld: those, the intercept, the relative effects
    for t in tables:
        for name in dict.fromkeys(str(n) for n in t.info.get("separated") or []):
            counts[name] = counts.get(name, 0) + 1
        for name in dict.fromkeys(str(n) for n in [*(t.info.get("separated") or []),
                                                   *t.infinite]):
            held[name] = held.get(name, 0) + 1
    if not held:
        return None
    for row in rows:
        if str(row["feature"]) in held:
            row.update(dict.fromkeys(WITHHELD))
            if "ratio" in row:
                row.update(ratio_low=None, ratio_high=None)
    order = [str(r["feature"]) for r in rows if str(r["feature"]) in held]
    names = [n for n in order if n in counts] or list(counts)
    others = [n for n in order if n not in counts]
    caption = (firth or tables)[0].info.get("caption", "")
    if firth and len(firth) < m:
        k, rest = len(firth), m - len(firth)
        fits = ("is a maximum-likelihood fit, its standard errors" if rest == 1 else
                "are maximum-likelihood fits, their standard errors")
        mixed = (f"In {k:,} of the {m:,} copies a column separates the outcome, so the "
                 f"maximum-likelihood estimate does not exist in {'it' if k == 1 else 'them'} and "
                 f"{'it' if k == 1 else 'each'} is fit by Firth's penalized likelihood, its "
                 f"standard errors from the information at the penalized estimate; the other "
                 f"{rest:,} {fits} from the logistic model's information. Rubin's rules pool each "
                 f"copy's estimates and standard errors into Wald intervals and p-values; no "
                 f"profile interval is computed.")
        caption = (caption.replace(POOLED_FIRTH_CAPTION, mixed) if POOLED_FIRTH_CAPTION in caption
                   else f"{caption} {mixed}")
    subject = names or others
    one = len(subject) == 1
    its = "its" if one else "their"
    because = withheld_reason(m, subject)
    unreported = ("standard error, interval and p-value" if one else
                  "standard errors, intervals and p-values")
    if names:
        concern = (f"{_and(names)} {'separates' if one else 'separate'} the outcome "
                   f"{_in_copies(counts, names, m)}, so {its} pooled {unreported} are not "
                   f"reported, only {its} estimate{'' if one else 's'}: {because}.")
    if others:
        single = len(others) == 1
        listed = [_term(n) for n in others]
        words = listed[0] if single else f"{', '.join(listed[:-1])} and {listed[-1]}"
        what = ("standard error, interval and p-value" if single else
                "standard errors, intervals and p-values")
        value = ("its maximum-likelihood value is" if single else
                 "their maximum-likelihood values are")
        where = _in_copies(held, others, m)
        concern = (f"{concern} The {what} of {words} are withheld too: {value} infinite "
                   f"{where}." if names else
                   f"The {what} of {words} are not reported, only "
                   f"{its} estimate{'' if single else 's'}: {value} infinite {where}, and "
                   f"{because}.")
    columns: list[str] = []
    for n in names or [o for o in others if o != "(intercept)"] or others:
        columns.extend(c for c in sorted(str(c) for c in (sources or {}).get(n) or [n])
                       if c not in columns)
    whose = "its" if len(columns) == 1 else "their"
    ways = f"read {_and(columns)} another way or combine {whose} sparse levels"
    if answer is not None:
        ways += f", or {COMPLETE_CASES}"
    concern += f" To report {'it' if len(order) == 1 else 'them'}, {ways}."
    exits: list[dict[str, Any]] = [
        {"label": f"Read {_and(columns)} another way, or combine {whose} sparse levels",
         "decision": None}]
    if answer is not None:
        exits.append({"label": "Complete cases, with their assumption stated",
                      "decision": {**dict(answer), "kind": "set_missing",
                                   "strategy": "complete_case"}})
    replaces = list(dict.fromkeys(separation_concern(t.info["separated"]) for t in firth
                                  if t.info.get("separated")))
    return PooledFirth(caption=caption, concern=concern, replaces=replaces,
                       info={"separated": list(counts), "covariance": "model", "exits": exits})


def _multinomial(names: list[str], exog: pd.DataFrame, y: np.ndarray, classes: list[Any],
                 clusters: Clusters) -> InferenceTable:
    import statsmodels.api as sm

    X = exog.to_numpy(dtype=float)
    codes = pd.Categorical(y, categories=classes).codes
    fit = sm.MNLogit(codes, exog).fit(disp=0, maxiter=MAX_NEWTON)
    B = np.asarray(fit.params, dtype=float)  # (P, K − 1), against the first class
    q = B.shape[1]
    labels = [f"{n} [{classes[k + 1] if classes else k + 1}]" for k in range(q) for n in names]
    est = B.ravel(order="F")  # class by class: the order of every vector below
    estimator = "multinomial logistic regression (maximum likelihood)"
    if not fit.mle_retvals.get("converged", False):
        return _refused(labels, np.full(len(labels), np.nan), estimator, clusters,
                        not_converged("multinomial"), ())
    probs = np.asarray(fit.predict(X), dtype=float)
    Xw, ew = multinomial_working(X, codes, probs)
    # Each class's score, Xᵀ(y_k − π_k), class by class as ``est`` (:func:`newton_distance`).
    onehot = (np.asarray(codes)[:, None] == np.arange(1, q + 1)[None, :]).astype(float)
    score = ((onehot - probs[:, 1:]).T @ X).ravel()
    short = at_maximum("the multinomial model's maximum-likelihood fit", newton_distance(Xw, score))
    if short:
        return _refused(labels, np.full(len(labels), np.nan), estimator, clusters, short, ())
    refusal = floor_refusal(clusters, task="multiclass")
    if refusal:
        return _refused(labels, est, estimator, clusters, *refusal)
    if clusters.clustered:
        return _clustered(labels, est, estimator, Xw, ew, np.repeat(clusters.codes, q), clusters)
    # The inverse information from the working design's QR (:func:`gram_inverse`), class by class
    # as ``est``, not statsmodels' inverse of the Hessian.
    C = gram_inverse(Xw)
    C = np.asarray(fit.cov_params(), dtype=float) if C is None else C
    se = np.sqrt(np.clip(np.diag(C), 0, None))
    reason = no_interval(labels, se, fit="The multinomial model's information")
    if reason:
        return _refused(labels, est, estimator, clusters, reason, ())
    concerns = [clusters.note] if clusters.note else []
    table = InferenceTable(_z_rows(labels, est, se),
                           _info(estimator, "model",
                                 "95% Wald intervals from the multinomial model's information.",
                                 clusters), concerns)
    # The covariance is ordered class by class, as ``est``; the table keeps none.
    table.contrast = lambda c, b: combination_row(b, math.sqrt(max(float(c @ C @ c), 0.0)))
    return table


__all__ = [
    "COMPLETE_CASES", "Clusters", "FEW_CLUSTERS", "FIRTH", "FirthFit", "FirthProfileSeparated",
    "INDEPENDENT",
    "InferenceTable", "Outcome", "POOLED_FIRTH_CAPTION", "PooledFirth", "Rows", "Scale", "WITHHELD",
    "bread_of", "by_identified", "cluster_columns", "column_scales", "copies_words", "cr2",
    "gram_inverse", "hc3_covariance", "qr_frame",
    "dependency_reason", "logistic_working", "scaled_inverse",
    "multinomial_working", "firth_fit", "firth_profile", "floor_refusal", "for_pooling",
    "no_interval",
    "format_p", "heteroskedasticity_concern", "pooled_firth", "pooling", "separation_concern",
    "inference_table", "is_separated", "min_clusters", "resolve_clusters", "separated_columns",
    "separated_profile_words", "withheld_reason",
]
